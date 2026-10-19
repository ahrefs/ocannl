(* gh-ocannl-1409: a budget-cut search that never measured the untuned default. With automatic
   scheduling on and [schedule_fission=false], no candidate reproduces the untuned default
   ([Autotune.default_seed_label] is [None]) and cc has no seeded plan, so nothing is exempt from
   the budget. A budget spent before the first candidate then leaves the serial baseline as the only
   timing; crowning it could ship and cache code slower than not tuning, so the search ships the
   untuned default and stores nothing. [schedule_fission] is read once per process, hence a process
   of its own (the cut, the floor and the replay rules are [autotune_budget]'s).

   The virtual clock and scripted timings are [autotune_budget]'s: the attempts so far, the baseline
   first, and a baseline timed fastest, so a search allowed to crown it would. *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module SC = Ir.Schedule_cache
module Cache_plant = Test_utils.Cache_plant

let budget_key = "autotune_budget_seconds"
let () = Hashtbl.set Utils.config_file_args ~key:"schedule_fission" ~data:"false"
let backend_name = Context.backend_name (Context.cpu ())
let chain = Fissioned_chain.chain_with_output ()

let untuned (qg, comp) =
  let ctx, routine = Context.compile (Context.cpu ()) comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  let values = Context.get_values ctx qg.Tensor.value in
  Context.release ctx;
  values

let reference = untuned chain
let cache_dir = "autotune_cache_budget_unfloored"

(* An earlier run's entry would turn the cold search into a replay. *)
let () =
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f))

(* One search under a budget of one attempt: the baseline's. *)
let search ~tag =
  let qg, comp = chain in
  let report = ref None and attempts = ref [] in
  let output, ios, identity =
    Exn.protect
      ~finally:(fun () -> Hashtbl.remove Utils.config_file_args budget_key)
      ~f:(fun () ->
        Hashtbl.set Utils.config_file_args ~key:budget_key ~data:"1";
        Autotune.Seam.with_value Autotune.on_candidate_measured (fun ~label ~digest:_ _ms ->
            if String.equal label "baseline" then 1. else 50.)
        @@ fun () ->
        Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label ->
            attempts := label :: !attempts)
        @@ fun () ->
        Autotune.Seam.with_value Autotune.budget_elapsed (fun _real ->
            Float.of_int (List.length !attempts))
        @@ fun () ->
        let ctx = Context.cpu () in
        let identity = Option.is_some (Context.timing_identity ctx) in
        let (ctx, routine), ios =
          SC.recording_cache_io (fun () ->
              Autotune.with_uncontended_test_windows (fun () ->
                  Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir
                    ~report:(fun r -> report := Some r)
                    ctx comp Ir.Indexing.Empty))
        in
        let ctx = Context.run ctx routine in
        let output = Context.get_values ctx qg.Tensor.value in
        Context.release ctx;
        (output, ios, identity))
  in
  match !report with
  | Some r ->
      Stdio.eprintf "%s (not part of the golden): %s, budget_cut %b, attempts %s, crown %s\n%!" tag
        (Autotune.outcome_name r.Autotune.outcome)
        r.Autotune.budget_cut
        (String.concat ~sep:"; " (List.rev !attempts))
        r.Autotune.best_label;
      (r, List.rev !attempts, output, ios, identity)
  | None -> failwith (tag ^ ": the search delivered no report")

let () =
  p "setup: no candidate reproduces the untuned default"
    (Option.is_none (Autotune.default_seed_label ~backend_name));
  let r, attempts, output, _ios, _identity = search ~tag:"cold" in
  p "cold: the budget cut the search before any candidate"
    ((match r.Autotune.outcome with Autotune.Searched -> true | _ -> false)
    && r.Autotune.budget_cut
    && List.equal String.equal attempts [ "baseline" ]);
  p_all2 "cold: the shipped routine computes the untuned output bit for bit" output reference
    ~f:Float.equal;
  (* Stored, the baseline crown would replay here. *)
  let warm, _, _, ios, identity = search ~tag:"warm" in
  Cache_plant.unless_unkeyed_or_refused ~identity ios
    "warm: nothing was stored, so the next budgeted call searches again"
    (match warm.Autotune.outcome with Autotune.Searched -> true | _ -> false)

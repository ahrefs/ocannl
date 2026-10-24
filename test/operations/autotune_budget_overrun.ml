(* gh-ocannl-1556: [autotune_budget_seconds] bounds a search's wall time, not only when its last
   attempt starts. A compile cannot be cancelled, so an attempt starts only where a compile as
   costly as the costliest one the search has made still ends within the budget (the projection);
   and a non-floor candidate's timing is abandoned, unranked, before its first batch past the
   budget.

   No real clock: every search reads a virtual one through [Autotune.budget_elapsed]. It advances
   only by what the test charges -- each non-floor candidate compile [compile_s] (charged as the
   compile's record reaches [Autotune.install_compile_observer], inside the compile's span), and
   each budget read made during one chosen candidate's timing [batch_s] (the reads the timing check
   makes, one before the warmup and one before each batch). The floors (the baseline, the untuned
   default seed) cost nothing: they are exempt from the budget, so the claims bound only what the
   budget governs. The timings are scripted ([Scripted_search]) so the ranking is the script's; the
   crowned routines are still compiled, run and compared with the untuned output. Searched on cc,
   over [Fissioned_chain]'s small menu. *)

open Base
open Verdict.Claims

let () = Fissioned_chain.small_menu ()
let budget_key = "autotune_budget_seconds"
let backend_name = Context.backend_name (Context.cpu ())
let default_label = Autotune.default_seed_label ~backend_name

let is_floor label =
  String.equal label "baseline" || Option.exists default_label ~f:(String.equal label)

(* The baseline slowest, the untuned default next, every other candidate a distinct faster time. *)
let cost label =
  if String.equal label "baseline" then 1000.
  else if Option.exists default_label ~f:(String.equal label) then 100.
  else 2. +. (Float.of_int (Stdlib.Hashtbl.hash label % 1000) /. 100.)

let chain = Fissioned_chain.chain_with_output ()
let reference = Scripted_search.untuned (fst chain) (snd chain)

(* The virtual clock and what one search charged to it. [timing_of] is the candidate whose timing
   reads are charged; [in_timing] is set by its pre-dispatch validation and cleared as the next
   attempt starts or its window is admitted. *)
type charges = {
  mutable clock : float;
  mutable compiles : int;  (** Non-floor candidate compiles charged. *)
  mutable timing_reads : int;  (** Budget reads charged during [timing_of]'s timing. *)
  mutable current : string;
  mutable in_timing : bool;
}

(* The search in flight's charges and per-compile charge. *)
let active : (charges * float) option ref = ref None

let () =
  Autotune.install_compile_observer (fun r ->
      match !active with
      | Some (c, compile_s)
        when String.equal r.Autotune.compile_stage "candidate"
             && not (is_floor r.Autotune.compile_attempt) ->
          c.compiles <- c.compiles + 1;
          c.clock <- c.clock +. compile_s
      | Some _ | None -> ())

(* One search under a budget, charging [compile_s] per non-floor candidate compile and [batch_s] per
   budget read during [timing_of]'s timing. *)
let search ?budget ?(compile_s = 0.) ?(batch_s = 0.) ?timing_of ~tag () =
  let c = { clock = 0.; compiles = 0; timing_reads = 0; current = ""; in_timing = false } in
  let qg, comp = chain in
  let attempt = Autotune.Seam.get Autotune.on_candidate_attempt in
  let preflight = Autotune.Seam.get Autotune.on_candidate_preflight in
  let measured = Autotune.Seam.get Autotune.on_candidate_measured in
  let o =
    Exn.protect
      ~finally:(fun () -> active := None)
      ~f:(fun () ->
        active := Some (c, compile_s);
        Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label ->
            c.current <- label;
            c.in_timing <- false;
            attempt label)
        @@ fun () ->
        Autotune.Seam.with_value Autotune.on_candidate_preflight (fun name ->
            c.in_timing <- Option.exists timing_of ~f:(String.equal c.current);
            preflight name)
        @@ fun () ->
        Autotune.Seam.with_value Autotune.on_candidate_measured (fun ~label ~digest reading ->
            c.in_timing <- false;
            measured ~label ~digest reading)
        @@ fun () ->
        Autotune.Seam.with_value Autotune.budget_elapsed (fun _real ->
            if c.in_timing then (
              c.timing_reads <- c.timing_reads + 1;
              c.clock <- c.clock +. batch_s);
            c.clock)
        @@ fun () ->
        Scripted_search.search
          ~settings:(Option.value_map budget ~default:[] ~f:(fun b -> [ (budget_key, b) ]))
          ~output:qg ~cache_dir:"" ~tag
          ~measure:(fun w -> cost w.Scripted_search.label)
          comp)
  in
  Stdio.eprintf
    "%s (not part of the golden): attempts %s | measured %s | clock %.1f, %d compiles charged, %d \
     timing reads, budget_cut %b, crown %s\n\
     %!"
    tag
    (String.concat ~sep:"; " o.attempts)
    (String.concat ~sep:"; " o.measured)
    c.clock c.compiles c.timing_reads o.report.Autotune.budget_cut o.report.Autotune.best_label;
  (o, c)

let non_floor attempts = List.filter attempts ~f:(fun l -> not (is_floor l))

let searched (o : Scripted_search.observed) =
  match o.report.Autotune.outcome with Autotune.Searched -> true | _ -> false

(* --- the control: no budget, every candidate attempted --- *)

let control, _ = search ~compile_s:3. ~tag:"control" ()
let control_non_floor = non_floor control.attempts
let control_measured = non_floor control.measured

let () =
  p "control: the menu offers at least five non-floor attempts" (List.length control_non_floor >= 5);
  p "control: nothing was cut" (searched control && not control.report.Autotune.budget_cut)

(* --- the projection: each non-floor compile costs 3 s against a 10 s budget --- *)

let () =
  let budget = 10. and compile_s = 3. in
  let o, c = search ~budget:"10" ~compile_s ~tag:"projection" () in
  (* At 9 s a fourth compile would end at 12 s: the projection refuses it, where a check of the
     elapsed time alone (9 < 10) started it. *)
  p "projection: the non-floor compiles ended within the budget" Float.(c.clock <= budget);
  p "projection: as many compiles started as fit in the budget"
    (c.compiles = Float.iround_towards_zero_exn (budget /. compile_s));
  p "projection: its attempts are a prefix of the control's, cut by the budget"
    (searched o && o.report.Autotune.budget_cut
    && List.is_prefix control_non_floor ~prefix:(non_floor o.attempts) ~equal:String.equal);
  p_all2 "projection: the crowned routine computes the untuned output bit for bit" o.output
    reference ~f:Float.equal

(* --- the timing check: the second non-floor candidate the control timed passes a 3 s budget in its
   timing, at 1 s per batch --- *)

let () =
  match List.nth control_measured 1 with
  | None -> p "timing: the control timed a second non-floor candidate" false
  | Some slow ->
      let budget = 3. and batch_s = 1. in
      let o, c = search ~budget:"3" ~batch_s ~timing_of:slow ~tag:"timing" () in
      p "timing: the slow candidate was attempted" (List.mem o.attempts slow ~equal:String.equal);
      p "timing: its timing was abandoned, unranked"
        (not (List.mem o.measured slow ~equal:String.equal));
      p "timing: the search completed, cut by its budget"
        (searched o && o.report.Autotune.budget_cut);
      (* One read before the warmup (1 s), then one per batch: the read that passed 3 s was the
         fourth, so batches ran before the stop and none ran after it. *)
      p "timing: it stopped at the first batch past the budget, not before its warmup"
        (c.timing_reads = Float.iround_towards_zero_exn (budget /. batch_s) + 1);
      p "timing: the clock passed the budget by one batch at most"
        Float.(c.clock > budget && c.clock <= budget +. batch_s);
      p "timing: no non-floor attempt started after it"
        (Option.equal String.equal (List.last (non_floor o.attempts)) (Some slow));
      p_all2 "timing: the crowned routine computes the untuned output bit for bit" o.output
        reference ~f:Float.equal

(* gh-ocannl-2134: a tuner candidate whose timing wait outlives its host-side bound is a typed
   decline of that candidate, and the search ends -- for a kernel that never ends and that no driver
   reports, so no classifier ever sees a status.

   No kernel hangs: the expiry is injected by a driver that wraps [Autotune.Context_driver] and
   answers [false] from the bounded wait ([await_within]) at one chosen candidate's first wait, the
   answer the production wait gives when its bound passes with the work outstanding. The real wait
   ran first, so nothing is left behind. Asserted: the expiry is counted once as a [wait_bound]
   decline, the search ends [Search_died] at that candidate's [Sync] with the decline's rendering,
   the timing lineage is poisoned, the candidate is never timed and no later candidate is attempted.
   The healthy control is the same search with no fault: no wait expires, and every wait's bound is
   at least the floor. The bound's arithmetic is pinned directly ([Autotune.wait_bound_s]).

   On a GPU backend a leg runs the healthy control on the device itself, through its own bounded
   wait (Metal's, or CUDA's and HIP's generic stream poll) (the production [await_within], which
   must not expire on a normal search). The real expiry path, with the device mark it leaves, is
   test/operations/wait_bound_device_wedged. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module SO = Ir.Schedule_outcome
open Verdict.Claims

(* Whether to answer [false] at the target's first wait: [false] for the healthy control. *)
let fault = ref false

(* Whether to aim the fault at the diagnostic untuned control instead (the search's one [compile]
   that is not a candidate's [compile_outcome], run only under [autotune_log]). *)
let at_control = ref false

(* The candidate the fault is aimed at, as in metal_hang_declines: chosen at its attempt, recognized
   by the routine its compile returns, fired at the first wait after that routine's first dispatch,
   then [fired] names it. *)
let target : string option ref = ref None
let target_routine : int option ref = ref None
let compiled : (string, int) Hashtbl.t = Hashtbl.create (module String)
let dispatched = ref false
let fired : (string * float) option ref = ref None
let poisoned : (string * exn) list ref = ref []

(* Every bounded wait of the search: its bound, and the production wait's own answer. *)
let waits : (float * bool) list ref = ref []

module Expiry_driver = struct
  include Autotune.Context_driver

  let compile_outcome ?name ?lowered_transform ?prelowered ?reuse ~provenance ?candidate ?retain ctx
      comp bindings =
    let outcome =
      Autotune.Context_driver.compile_outcome ?name ?lowered_transform ?prelowered ?reuse
        ~provenance ?candidate ?retain ctx comp bindings
    in
    (match (outcome, candidate) with
    | Ok (_, routine), Some c ->
        Hashtbl.set compiled ~key:c ~data:(routine_id routine);
        if Option.equal String.equal (Some c) !target && not !dispatched then
          target_routine := Some (routine_id routine)
    | _ -> ());
    outcome

  let compile ?name ?lowered_transform ?prelowered ?reuse ctx comp bindings =
    let ((_, routine) as compiled) =
      Autotune.Context_driver.compile ?name ?lowered_transform ?prelowered ?reuse ctx comp bindings
    in
    if !at_control && Option.is_none !target then (
      target := Some "untuned control";
      target_routine := Some (routine_id routine));
    compiled

  let run ?timing ctx routine =
    let ctx = Autotune.Context_driver.run ?timing ctx routine in
    if Option.equal Int.equal !target_routine (Some (routine_id routine)) && Option.is_none !fired
    then dispatched := true;
    ctx

  let await_within ctx ~seconds =
    let completed = Autotune.Context_driver.await_within ctx ~seconds in
    waits := (seconds, completed) :: !waits;
    match !target with
    | Some label when !fault && !dispatched && Option.is_none !fired ->
        fired := Some (label, seconds);
        false
    | _ -> completed

  let poison_lineage ctx ~routine_name exn =
    poisoned := (routine_name, exn) :: !poisoned;
    Autotune.Context_driver.poison_lineage ctx ~routine_name exn
end

module Bounded = Autotune.Make (Expiry_driver)

let n = 8

let comp, output =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "wb_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "wb_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t = relu mc in
  ignore mc;
  (Ll_test.unowned_copy (Train.forward t), t)

type run = {
  result : (float array, exn) Result.t;
  report : Autotune.report option;
  attempts : string list;
  measured : string list;
  fired_at : (string * float) option;
  poisonings : (string * exn) list;
  bounded_waits : (float * bool) list;
}

(* One search of [comp] through the wrapped driver on a fresh context from [make_ctx]. With
   [~expire:true], the fault is aimed at the first candidate attempted once the baseline is
   timed. *)
let search ?(make_ctx = fun () -> Context.cpu ()) ?(expire = false) ?(control = false) () =
  fault := expire || control;
  at_control := control;
  target := None;
  target_routine := None;
  Hashtbl.clear compiled;
  dispatched := false;
  fired := None;
  poisoned := [];
  waits := [];
  let report = ref None in
  let timed = ref 0 in
  let result, record =
    Autotune.Seam.with_value Autotune.on_candidate_timed (fun _ ~timed_so_far ->
        timed := timed_so_far)
    @@ fun () ->
    Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label ->
        if (not !at_control) && !timed >= 1 && Option.is_none !fired && not !dispatched then (
          target := Some label;
          target_routine := Hashtbl.find compiled label))
    @@ fun () ->
    Scripted_search.scripted
      ~measure:(fun w -> Float.of_int (10 + w.Scripted_search.ordinal))
      (fun () ->
        let ctx = make_ctx () in
        match
          Bounded.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"" ~log:control
            ~report:(fun r -> report := Some r)
            ctx comp Ir.Indexing.Empty
        with
        | ctx, routine ->
            let ctx = Context.run ctx routine in
            let values = Context.get_values ctx output.Tensor.value in
            Context.release ctx;
            Ok values
        | exception exn -> Error exn)
  in
  fault := false;
  at_control := false;
  {
    result;
    report = !report;
    attempts = record.Scripted_search.attempts;
    measured = Scripted_search.measured record.Scripted_search.windows;
    fired_at = !fired;
    poisonings = List.rev !poisoned;
    bounded_waits = List.rev !waits;
  }

let floor = Autotune.wait_bound_floor_s

let healthy_claims ~what ~expected (r : run) =
  p
    (what ^ ", the search returned a routine computing the untuned values")
    (match r.result with Ok got -> Array.equal Float.equal got expected | Error _ -> false);
  p_exists (what ^ ", the search timed candidates past the baseline") r.measured ~f:(fun l ->
      not (String.equal l "baseline"));
  p_all (what ^ ", no timing wait expired") r.bounded_waits ~f:snd;
  p_all (what ^ ", every timing wait was bounded by at least the floor") r.bounded_waits
    ~f:(fun (s, _) -> Float.(s >= floor));
  (* The measured waits, for the PR's record of how far below the bound a normal search runs. *)
  Stdio.eprintf "%s: %d bounded waits, bounds %s s (not part of the golden)\n%!" what
    (List.length r.bounded_waits)
    (String.concat ~sep:","
       (List.dedup_and_sort ~compare:Float.compare (List.map r.bounded_waits ~f:fst)
       |> List.map ~f:(Printf.sprintf "%g")))

let () =
  (* --- The bound's arithmetic. *)
  p "with no reference, the bound is the floor" Float.(Autotune.wait_bound_s ~depth:1 () = floor);
  p "the floor is 60 s and the multiple 20"
    Float.(floor = 60. && Autotune.wait_bound_multiple = 20.);
  p "a millisecond-scale reference keeps the floor"
    Float.(Autotune.wait_bound_s ~reference_ms:12.5 ~depth:16 () = floor);
  p "a 5 s single's wait is bounded at 20 times it"
    Float.(Autotune.wait_bound_s ~reference_ms:5000. ~depth:1 () = 100.);
  p "the bound scales with the launches the wait closes"
    Float.(Autotune.wait_bound_s ~reference_ms:5000. ~depth:2 () = 200.);
  p_all "a non-finite or non-positive reference keeps the floor"
    [ Float.nan; Float.infinity; 0.; -3. ] ~f:(fun ms ->
      Float.(Autotune.wait_bound_s ~reference_ms:ms ~depth:4 () = floor));
  let expected = Scripted_search.untuned output comp in
  (* --- The healthy control. *)
  healthy_claims ~what:"with no fault" ~expected (search ());
  (* --- The injected expiry. *)
  let backend = Context.backend_name (Context.cpu ()) in
  let bound_key = SO.Backend_rejected_key (backend, SO.wait_bound_stage, SO.Expected) in
  let expired = search ~expire:true () in
  let label = Option.value_map expired.fired_at ~default:"<the fault never fired>" ~f:fst in
  Stdio.eprintf "the expiry was injected at %s (not part of the golden)\n%!" label;
  p "the expiry was injected at a candidate's timing wait" (Option.is_some expired.fired_at);
  p "the expiry was aimed at a candidate, not the baseline" (not (String.equal label "baseline"));
  p "the baseline was timed before it" (List.mem expired.measured "baseline" ~equal:String.equal);
  p "the expired wait's bound was at least the floor"
    (match expired.fired_at with Some (_, s) -> Float.(s >= floor) | None -> false);
  p "the search raised the decline's rendering, naming the host-side bound"
    (match expired.result with
    | Error (Invalid_argument m) -> String.is_substring m ~substring:"host-side bound"
    | Ok _ | Error _ -> false);
  p "the expiry is counted once, as a wait_bound decline"
    (match
       Option.value_map expired.report ~default:[] ~f:(fun rep ->
           List.filter rep.Autotune.declines ~f:(fun d ->
               SO.equal_rejection_key d.Autotune.key bound_key))
     with
    | [ d ] -> d.Autotune.count = 1
    | _ -> false);
  p "the report ends the search at the expired candidate's Sync"
    (match Option.map expired.report ~f:(fun rep -> rep.Autotune.outcome) with
    | Some (Autotune.Search_died { phase = SO.Sync; candidate = Some c; _ }) -> String.equal c label
    | _ -> false);
  p_exists "the timing lineage was poisoned with the expiry's rendering" expired.poisonings
    ~f:(fun (_, exn) ->
      match exn with
      | Invalid_argument m -> String.is_substring m ~substring:"host-side bound"
      | _ -> false);
  p "the expired candidate was never timed"
    (not (List.mem expired.measured label ~equal:String.equal));
  p "no candidate was attempted after the expired one"
    (match List.last expired.attempts with Some last -> String.equal last label | None -> false);
  (* --- The diagnostic untuned control's wait expiring: not swallowed as a diagnostic's failure. *)
  let control = search ~control:true () in
  p "the expiry was injected at the diagnostic control's wait"
    (Option.exists control.fired_at ~f:(fun (l, _) -> String.equal l "untuned control"));
  p "an expiry at the diagnostic control propagates as the expiry itself"
    (match control.result with Error (SO.Wait_bound_expired _) -> true | Ok _ | Error _ -> false);
  p_exists "an expiry at the diagnostic control poisons the lineage with it" control.poisonings
    ~f:(fun (_, exn) -> match exn with SO.Wait_bound_expired _ -> true | _ -> false);
  (* --- The GPU leg: the healthy control through the device's own bounded wait. *)
  let ctx = Context.auto () in
  let backend_name = Context.backend_name ctx in
  let on_gpu = not (Ir.Backend_intf.renders_on_host_pool (Context.backend_facts ctx)) in
  Context.release ctx;
  let leg = Verdict.leg ~when_:on_gpu ~on:backend_name () in
  Verdict.in_leg ~leg "on the device, a normal search's bounded waits never expire" (fun claim ->
      let r = search ~make_ctx:Context.auto () in
      (* A search that failed contributes no waits, so its claim fails on the empty population. *)
      p_all claim (if Result.is_ok r.result then r.bounded_waits else []) ~f:snd)

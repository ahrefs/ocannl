(* gh-ocannl-2089: a tuner candidate whose kernel the GPU watchdog killed is a typed decline of that
   candidate, and the search still ends.

   No GPU is hung: the hang is the typed report [Metal_backend] raises for it
   ([Ir.Metal_failure.Command_buffer_failed], with the error text metal_gpu_hang_failure captured),
   raised from the timing sync of one chosen candidate by a driver that wraps
   [Autotune.Context_driver], and classified by the production Metal classifier the wrapper hands
   the tuner. So the search, the scripted ranking and the claims run on cc, on every box. Asserted:
   the candidate is counted once as a [gpu_hang] decline, the search ends [Search_died] at that
   candidate's [Sync] with the decline's rendering, the timing lineage is poisoned, the candidate is
   never timed and no later candidate is attempted. The healthy control is the same search with no
   fault; the attribution control raises the same text as a plain [Failure], which stays an
   unattributed fatal failure with no decline.

   Contained finals failures (gh-ocannl-2197) fail both recompiles, or both pre-dispatch runs, of
   one saved contender: each attempt counts under its cause key, while rollback and normal
   completion remain intact. The healthy run locates the first finals compile after search-only
   compiles that deduplicate without adding a reading.

   On Metal, a leg checks the wiring the wrapper stands in for: the backend's own
   [Context.failure_classifier] classifies the same typed report. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module SO = Ir.Schedule_outcome
module MF = Ir.Metal_failure
open Verdict.Claims

let hang_error = "Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)"

let typed_hang label =
  MF.Command_buffer_failed
    { what = "kernel " ^ label; runs_kernels = true; error = Some hang_error; first_report = true }

(* The fault in force for the current search: [None] for the healthy control. *)
type injected_fault = Report of (string -> exn) | Expire_wait

let fault : injected_fault option ref = ref None

(* The candidate the fault is aimed at: chosen at its attempt, recognized by the routine its compile
   returns (a baseline re-timing between two candidates dispatches another routine), fired at the
   first sync after that routine's first dispatch, then [fired] names it. *)
let target : string option ref = ref None
let target_routine : int option ref = ref None

(* Every candidate compile's routine, by the candidate's label: a candidate may be compiled ahead of
   its attempt. *)
let compiled : (string, int) Hashtbl.t = Hashtbl.create (module String)

(* Most recent compile by routine name: candidates can share that name. At a finals preflight, this
   is the contender just recompiled for this window, not a per-candidate map. *)
let compiled_routines : (string, int) Hashtbl.t = Hashtbl.create (module String)
let dispatched = ref false
let fired : string option ref = ref None
let poisoned : (string * exn) list ref = ref []

(* Contained faults are injected only after the healthy search's first finals recompile. Match its
   compile label, which can differ from the label used to time the saved contender. *)
type contained_fault = Recompile | Run

exception Contained_run

let contained : (contained_fault * int * string) option ref = ref None
let windows_completed = ref 0
let compile_trace : (int * string) list ref = ref []
let contained_attempts = ref 0
let contained_routine = ref None
let rollbacks = ref 0

let contained_cause stage =
  SO.Backend_rejected
    { backend = "fixture"; stage; severity = SO.Expected; detail = "contained finals fault" }

module Hang_driver = struct
  include Autotune.Context_driver

  let compile_outcome ?name ?lowered_transform ?prelowered ?reuse ~provenance ?candidate ?retain ctx
      comp bindings =
    Option.iter candidate ~f:(fun label ->
        compile_trace := (!windows_completed, label) :: !compile_trace);
    let aimed =
      match (!contained, candidate) with
      | Some (mode, before, label), Some c
        when !windows_completed >= before && String.equal c label && !contained_attempts < 2 ->
          Some mode
      | _ -> None
    in
    contained_routine := None;
    let outcome =
      match aimed with
      | Some Recompile ->
          Int.incr contained_attempts;
          Error
            (SO.Classified
               {
                 phase = SO.Backend_compile;
                 cause = contained_cause "finals_compile";
                 execution_effect = SO.No_device_writes;
               })
      | Some Run | None ->
          Autotune.Context_driver.compile_outcome ?name ?lowered_transform ?prelowered ?reuse
            ~provenance ?candidate ?retain ctx comp bindings
    in
    (match (aimed, outcome) with
    | Some Run, Ok (_, routine) -> contained_routine := Some (routine_id routine)
    | _ -> ());
    (match (outcome, candidate) with
    | Ok (_, routine), Some c ->
        Hashtbl.set compiled ~key:c ~data:(routine_id routine);
        Hashtbl.set compiled_routines ~key:(routine_name routine) ~data:(routine_id routine);
        if Option.equal String.equal (Some c) !target && not !dispatched then
          target_routine := Some (routine_id routine)
    | _ -> ());
    outcome

  let run ?timing ctx routine =
    if Option.equal Int.equal !contained_routine (Some (routine_id routine)) then (
      contained_routine := None;
      Int.incr contained_attempts;
      raise Contained_run);
    let ctx = Autotune.Context_driver.run ?timing ctx routine in
    if Option.equal Int.equal !target_routine (Some (routine_id routine)) && Option.is_none !fired
    then dispatched := true;
    ctx

  (* The tuner's timing waits are bounded (gh-ocannl-2134), so they go through [await_within]. *)
  let await_within ctx ~seconds =
    let completed = Autotune.Context_driver.await_within ctx ~seconds in
    match (!fault, !target, !fired) with
    | Some injection, Some label, None when !dispatched -> (
        fired := Some label;
        match injection with Report make -> raise (make label) | Expire_wait -> false)
    | _ -> completed

  let failure_classifier _ctx phase = function
    | Contained_run ->
        Some
          { SO.phase; cause = contained_cause "finals_run"; execution_effect = SO.No_device_writes }
    | exn -> MF.classify_command_buffer_failure ~backend:"metal" phase exn

  let rollback_execution ctx routine =
    Int.incr rollbacks;
    Autotune.Context_driver.rollback_execution ctx routine

  let poison_lineage ctx ~routine_name exn =
    poisoned := (routine_name, exn) :: !poisoned;
    Autotune.Context_driver.poison_lineage ctx ~routine_name exn
end

module Hung = Autotune.Make (Hang_driver)

let n = 8

let comp, output =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "hd_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "hd_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t = relu mc in
  ignore mc;
  (Ll_test.unowned_copy (Train.forward t), t)

type run = {
  result : (float array, exn) Result.t;
  report : Autotune.report option;
  attempts : string list;
  measured : string list;
  fired_at : string option;
  poisonings : (string * exn) list;
  compiles : (int * string) list;
  failed_attempts : int;
  rolled_back : int;
}

(* One search of [comp] through the wrapped driver on a fresh cc context. With [?injection], the
   fault is aimed at the first candidate attempted once the baseline is timed: a seed the search
   times itself, so the search has an incumbent when it hangs. (This small search's later contenders
   are re-timed by its finals from saved forms, a path with no attempt.) A candidate whose compile
   is declined never dispatches, and the aim moves on to the next attempt. *)
let search ?injection ?finals_at ?contained_fault ?(finals = false) () =
  contained := contained_fault;
  windows_completed := 0;
  compile_trace := [];
  contained_attempts := 0;
  contained_routine := None;
  rollbacks := 0;
  fault := injection;
  target := None;
  target_routine := None;
  Hashtbl.clear compiled;
  Hashtbl.clear compiled_routines;
  dispatched := false;
  fired := None;
  poisoned := [];
  let report = ref None in
  let timed = ref 0 in
  let windows = ref 0 in
  let result, record =
    Autotune.Seam.with_value Autotune.on_candidate_preflight (fun routine_name ->
        Option.iter finals_at ~f:(fun (before, label) ->
            if !windows = before && Option.is_none !fired then (
              target := Some label;
              target_routine := Hashtbl.find compiled_routines routine_name)))
    @@ fun () ->
    Autotune.Seam.with_value Autotune.on_candidate_timed (fun _ ~timed_so_far ->
        timed := timed_so_far)
    @@ fun () ->
    Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label ->
        if (not finals) && !timed >= 1 && Option.is_none !fired && not !dispatched then (
          target := Some label;
          target_routine := Hashtbl.find compiled label))
    @@ fun () ->
    Scripted_search.scripted
      ~settings:[ ("autotune_composite_playoff_width", "0"); ("autotune_finals_margin", "0.1") ]
      ~measure:(fun w ->
        Int.incr windows;
        Int.incr windows_completed;
        if finals then 10. else Float.of_int (10 + w.Scripted_search.ordinal))
      (fun () ->
        let ctx = Context.cpu () in
        match
          Hung.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
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
  fault := None;
  contained := None;
  {
    result;
    report = !report;
    attempts = record.Scripted_search.attempts;
    measured = Scripted_search.measured record.Scripted_search.windows;
    fired_at = !fired;
    poisonings = List.rev !poisoned;
    compiles = List.rev !compile_trace;
    failed_attempts = !contained_attempts;
    rolled_back = !rollbacks;
  }

let hang_key = SO.Backend_rejected_key ("metal", MF.gpu_hang_stage, SO.Expected)

let declines_for key (r : run) =
  Option.value_map r.report ~default:[] ~f:(fun rep ->
      List.filter rep.Autotune.declines ~f:(fun d -> SO.equal_rejection_key d.Autotune.key key))

let hang_declines = declines_for hang_key

let died_at_sync_of (r : run) label =
  match Option.map r.report ~f:(fun rep -> rep.Autotune.outcome) with
  | Some (Autotune.Search_died { phase = SO.Sync; candidate = Some c; _ }) -> String.equal c label
  | _ -> false

let fired_label r = Option.value r.fired_at ~default:"<the fault never fired>"

let () =
  let expected = Scripted_search.untuned output comp in
  (* --- The healthy control: no fault, the search ships a routine computing the right values. *)
  let healthy = search () in
  p "with no fault, the search returned a routine computing the untuned values"
    (match healthy.result with Ok got -> Array.equal Float.equal got expected | Error _ -> false);
  p_exists "with no fault, the search timed candidates past the baseline" healthy.measured
    ~f:(fun l -> not (String.equal l "baseline"));
  (* --- The typed hang. *)
  let hung = search ~injection:(Report typed_hang) () in
  let label = fired_label hung in
  Stdio.eprintf "the hang was injected at %s (not part of the golden)\n%!" label;
  p "the hang was injected at a candidate's timing sync" (Option.is_some hung.fired_at);
  p "the hang was aimed at a candidate, not the baseline" (not (String.equal label "baseline"));
  p "the baseline was timed before it" (List.mem hung.measured "baseline" ~equal:String.equal);
  p "the search raised the hang decline's rendering, naming the driver's hang code"
    (match hung.result with
    | Error (Invalid_argument m) ->
        String.is_substring m ~substring:"kIOGPUCommandBufferCallbackErrorHang"
    | Ok _ | Error _ -> false);
  p "the hang is counted once, as a gpu_hang decline"
    (match hang_declines hung with [ d ] -> d.Autotune.count = 1 | _ -> false);
  p "the report ends the search at the hung candidate's Sync" (died_at_sync_of hung label);
  p_exists "the timing lineage was poisoned with the hang's rendering" hung.poisonings
    ~f:(fun (_, exn) ->
      match exn with
      | Invalid_argument m -> String.is_substring m ~substring:hang_error
      | _ -> false);
  p "the hung candidate was never timed" (not (List.mem hung.measured label ~equal:String.equal));
  p "no candidate was attempted after the hung one"
    (match List.last hung.attempts with Some last -> String.equal last label | None -> false);
  (* gh-ocannl-2135: equal readings and no composite playoff yield beam finals. The healthy control
     locates their first window from the report, rather than guessing a seed count or confusing
     baseline re-timings with finals. Replay the same search and fail that window's wait. *)
  let control = search ~finals:true () in
  let finals_timed =
    Option.value_map control.report ~default:0 ~f:(fun r -> r.Autotune.finals_timed)
  in
  p "finals: the healthy control re-times at least two contenders" (finals_timed >= 4);
  let before = List.length control.measured - finals_timed in
  let target_label = Option.value (List.nth control.measured before) ~default:"<no finals>" in
  let finals =
    search ~injection:(Report typed_hang) ~finals:true ~finals_at:(before, target_label) ()
  in
  let finals_label = fired_label finals in
  p "finals: the healthy control computes the untuned values"
    (match control.result with Ok got -> Array.equal Float.equal got expected | Error _ -> false);
  p "finals: the hang fired after that contender had a search reading"
    (Option.is_some finals.fired_at && List.mem finals.measured finals_label ~equal:String.equal);
  p "finals: the report ends the search at the contender's Sync"
    (died_at_sync_of finals finals_label);
  p "finals: the failure is in the first finals window, with none admitted"
    (Option.exists finals.report ~f:(fun r ->
         r.Autotune.finals_timed = 1 && r.Autotune.finals_admitted = 0));
  p "finals: the hang is counted once, as a gpu_hang decline"
    (match hang_declines finals with [ d ] -> d.Autotune.count = 1 | _ -> false);
  p_exists "finals: the lineage is poisoned with the hang's rendering" finals.poisonings
    ~f:(fun (_, exn) ->
      match exn with
      | Invalid_argument m -> String.is_substring m ~substring:hang_error
      | _ -> false);
  p "finals: the failed re-time adds no admitted reading"
    (List.count finals.measured ~f:(String.equal finals_label) = 1);
  (* gh-ocannl-2197: fail both recompiles/windows of one saved contender. This leaves it without
     finals readings, so finals_crown falls back normally to the search ranking. *)
  (* The last compile before that first finals reading is its recompile: earlier compiles
     at the same admitted-window count can be search candidates that deduplicated away. *)
  let compile_label =
    List.find_map_exn (List.rev control.compiles) ~f:(fun (at, label) ->
        if at = before then Some label else None)
  in
  List.iter
    [ (Recompile, "finals_compile"); (Run, "finals_run") ]
    ~f:(fun (mode, stage) ->
      let r = search ~finals:true ~contained_fault:(mode, before, compile_label) () in
      Stdio.eprintf "%s target=%s attempts=%d rollbacks=%d (not part of the golden)\n%!" stage
        compile_label r.failed_attempts r.rolled_back;
      let claim suffix = stage ^ ": " ^ suffix in
      let key = SO.Backend_rejected_key ("fixture", stage, SO.Expected) in
      p (claim "both attempts of one finals contender failed") (r.failed_attempts = 2);
      p
        (claim "each failed attempt is counted under its cause key")
        (match declines_for key r with
        | [ d ] -> d.Autotune.count = r.failed_attempts && d.count = 2
        | _ -> false);
      p
        (claim "candidates_failed equals the decline sum including both failed attempts")
        (Option.exists r.report ~f:(fun rep ->
             rep.Autotune.candidates_failed
             = List.sum (module Int) rep.declines ~f:(fun d -> d.Autotune.count)
             && Option.exists control.report ~f:(fun base ->
                 rep.Autotune.candidates_failed = base.Autotune.candidates_failed + 2)));
      p
        (claim "the search ends normally with a crown computing the untuned values")
        (Option.exists r.report ~f:(fun rep ->
             match rep.Autotune.outcome with Autotune.Searched -> true | _ -> false)
        && match r.result with Ok got -> Array.equal Float.equal got expected | Error _ -> false);
      p_empty (claim "the lineage is not poisoned") ~over:r.attempts r.poisonings;
      p
        (claim "only run failures roll back their execution claims")
        (r.rolled_back = match mode with Recompile -> 0 | Run -> 2);
      p
        (claim "the finals admit only the other contender's two windows")
        (Option.exists r.report ~f:(fun rep -> rep.Autotune.finals_admitted = finals_timed - 2)));
  (* Returning false drives the host-side wait-bound classification itself, rather than substituting
     a backend exception for it. The same first finals window is targeted. *)
  let expired = search ~injection:Expire_wait ~finals:true ~finals_at:(before, target_label) () in
  let bound_key =
    SO.Backend_rejected_key (Context.backend_name (Context.cpu ()), SO.wait_bound_stage, SO.Expected)
  in
  p "finals expiry: the first finals wait expired, with none admitted"
    (Option.is_some expired.fired_at
    && Option.exists expired.report ~f:(fun r ->
        r.Autotune.finals_timed = 1 && r.Autotune.finals_admitted = 0));
  p "finals expiry: the expiry is counted once under the wait-bound key"
    (match declines_for bound_key expired with [ d ] -> d.Autotune.count = 1 | _ -> false);
  p "finals expiry: the report ends the search at the contender's Sync"
    (died_at_sync_of expired target_label);
  p "finals expiry: the search raises the host-side bound's rendering"
    (match expired.result with
    | Error (Invalid_argument m) -> String.is_substring m ~substring:"host-side bound"
    | Ok _ | Error _ -> false);
  p_exists "finals expiry: the lineage is poisoned with the expiry's rendering" expired.poisonings
    ~f:(fun (_, exn) ->
      match exn with
      | Invalid_argument m -> String.is_substring m ~substring:"host-side bound"
      | _ -> false);
  (* --- The attribution control: the same report text, untyped. *)
  let untyped =
    search
      ~injection:(Report (fun label -> Failure (Stdlib.Printexc.to_string (typed_hang label))))
      ()
  in
  let untyped_label = fired_label untyped in
  p "untyped, the injection fired too" (Option.is_some untyped.fired_at);
  p "untyped, the search raised the plain Failure itself"
    (match untyped.result with
    | Error (Failure m) -> String.is_substring m ~substring:hang_error
    | Ok _ | Error _ -> false);
  p "untyped, nothing is counted as a gpu_hang decline" (List.is_empty (hang_declines untyped));
  p "untyped, the search still ends at that candidate's Sync"
    (died_at_sync_of untyped untyped_label);
  (* --- The Metal leg: the backend's own classifier, which the wrapper stood in for. *)
  let ctx = Context.auto () in
  let backend_name = Context.backend_name ctx in
  let leg = Verdict.leg ~when_:(String.equal backend_name "metal") ~on:backend_name () in
  Verdict.in_leg ~leg "the Metal backend's classifier types the hang as a gpu_hang decline"
    (fun claim ->
      p claim
        (match Context.failure_classifier ctx SO.Sync (typed_hang "ocannl_gh2089_kernel") with
        | Some
            {
              cause = SO.Backend_rejected { stage; severity = SO.Expected; _ };
              execution_effect = SO.Writes_may_have_occurred;
              _;
            } ->
            String.equal stage MF.gpu_hang_stage
        | _ -> false));
  Context.release ctx

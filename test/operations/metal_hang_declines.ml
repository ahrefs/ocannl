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
let fault : (string -> exn) option ref = ref None

(* The candidate the fault is aimed at: chosen at its attempt, recognized by the routine its compile
   returns (a baseline re-timing between two candidates dispatches another routine), fired at the
   first sync after that routine's first dispatch, then [fired] names it. *)
let target : string option ref = ref None
let target_routine : int option ref = ref None

(* Every candidate compile's routine, by the candidate's label: a candidate may be compiled ahead of
   its attempt. *)
let compiled : (string, int) Hashtbl.t = Hashtbl.create (module String)
let compiled_routines : (string, int) Hashtbl.t = Hashtbl.create (module String)
let dispatched = ref false
let fired : string option ref = ref None
let poisoned : (string * exn) list ref = ref []

module Hang_driver = struct
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
        Hashtbl.set compiled_routines ~key:(routine_name routine) ~data:(routine_id routine);
        if Option.equal String.equal (Some c) !target && not !dispatched then
          target_routine := Some (routine_id routine)
    | _ -> ());
    outcome

  let run ?timing ctx routine =
    let ctx = Autotune.Context_driver.run ?timing ctx routine in
    if Option.equal Int.equal !target_routine (Some (routine_id routine)) && Option.is_none !fired
    then dispatched := true;
    ctx

  (* The tuner's timing waits are bounded (gh-ocannl-2134), so they go through [await_within]. *)
  let await_within ctx ~seconds =
    let completed = Autotune.Context_driver.await_within ctx ~seconds in
    (match (!fault, !target, !fired) with
    | Some make, Some label, None when !dispatched ->
        fired := Some label;
        raise (make label)
    | _ -> ());
    completed

  let failure_classifier _ctx = MF.classify_command_buffer_failure ~backend:"metal"

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
}

(* One search of [comp] through the wrapped driver on a fresh cc context. With [?raise_as], the
   fault is aimed at the first candidate attempted once the baseline is timed: a seed the search
   times itself, so the search has an incumbent when it hangs. (This small search's later contenders
   are re-timed by its finals from saved forms, a path with no attempt.) A candidate whose compile
   is declined never dispatches, and the aim moves on to the next attempt. *)
let search ?raise_as ?finals_at ?(finals = false) () =
  fault := raise_as;
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
  {
    result;
    report = !report;
    attempts = record.Scripted_search.attempts;
    measured = Scripted_search.measured record.Scripted_search.windows;
    fired_at = !fired;
    poisonings = List.rev !poisoned;
  }

let hang_key = SO.Backend_rejected_key ("metal", MF.gpu_hang_stage, SO.Expected)

let hang_declines (r : run) =
  Option.value_map r.report ~default:[] ~f:(fun rep ->
      List.filter rep.Autotune.declines ~f:(fun d -> SO.equal_rejection_key d.Autotune.key hang_key))

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
  let hung = search ~raise_as:typed_hang () in
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
  let finals = search ~raise_as:typed_hang ~finals:true ~finals_at:(before, target_label) () in
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
  (* --- The attribution control: the same report text, untyped. *)
  let untyped =
    search ~raise_as:(fun label -> Failure (Stdlib.Printexc.to_string (typed_hang label))) ()
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

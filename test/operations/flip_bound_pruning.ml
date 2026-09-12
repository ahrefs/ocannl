(* gh-ocannl-514: bound pruning in [Train.tune_placements]' flip chain, one level above the phase-4b
   sketch gate. The dune rule pins the cc backend, [autotune_bound_pruning=true] and a tiny envelope
   (model_peak_* = 1e3), making the partial-placement-vector roofline floor
   ([Autotune.placement_surface.ps_floor_ms]) astronomically larger than any measured time — so
   every [`Materialize] flip candidate is fathomed before its nested search when arm A has a usable
   incumbent, without consuming the budget. A refused incumbent or insufficient floor margin leaves
   that timing-dependent claim undecided. [`Inline] flips are never floor-pruned (committing to
   inline tightens nothing), so the number of flip searches observed through [flip_report] equals
   the number of [`Inline] candidates the surface reports, and the shipped routine still computes
   correct values (it is the plain A/B winner or an inline refinement of it).

   The control for "the same flips are measured when pruning is off" is the existing
   inline_flip_tune test, which runs the same driver without the gate. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Asgns = Ir.Assignments
open Verdict.Claims

let approx a b = Float.(abs (a -. b) < 1e-4)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Require a factor-two separation before this timing-dependent pruning oracle decides. The
   production comparison remains exact; this is a floor on the test's evidence. *)
let decisive ~incumbent ~floor =
  Float.is_finite incumbent && Float.(incumbent > 0. && floor /. incumbent >= 2.)

let n = 8

let () =
  p "pruning evidence requires a finite incumbent and a twofold floor margin"
    (decisive ~incumbent:1. ~floor:2.
    && (not (decisive ~incumbent:1. ~floor:1.99))
    && (not (decisive ~incumbent:Float.infinity ~floor:1000.))
    && not (decisive ~incumbent:0. ~floor:1000.));
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 5) -. 2.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t2 = relu mc in
  ignore mc;
  let comp = named "fbp" (Train.forward t2) in
  (* Reference values from a plain compile. *)
  let ctx_ref, routine_ref = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx_ref = Context.run ctx_ref routine_ref in
  let expected = Context.get_values ctx_ref t2.Tensor.value in
  (* The surface the chain will walk: how many flips of each kind exist. *)
  let surface = Autotune.placement_surface (Context.auto ()) comp Ir.Indexing.Empty in
  let mat_flips, inline_flips_on_surface =
    List.partition_tf surface.Autotune.ps_candidates ~f:(fun fc ->
        match fc.LL.fc_flip with `Materialize -> true | `Inline -> false)
  in
  p "the surface reports at least one materialize flip" (List.length mat_flips >= 1);
  (* Budget above the whole surface: every candidate is either measured or fathomed. *)
  let budget = List.length surface.Autotune.ps_candidates + 1 in
  let arm_reports = ref [] in
  let flip_reports = ref [] in
  let ctx_t, routine_t =
    Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> arm_reports := r :: !arm_reports)
      ~flip_report:(fun r -> flip_reports := r :: !flip_reports)
      ~inline_flips:budget (Context.auto ()) t2 comp Ir.Indexing.Empty
  in
  let ctx_t = Context.run ctx_t routine_t in
  let got = Context.get_values ctx_t t2.Tensor.value in
  p_all2 "tuned routine values match the plain compile" got expected ~f:approx;
  p "the public report callback keeps the positional A/B contract" (List.length !arm_reports = 2);
  let arm_a = List.last_exn !arm_reports in
  let incumbent = arm_a.Autotune.best_ms in
  let floors =
    List.map mat_flips ~f:(fun fc -> surface.Autotune.ps_floor_ms ~materialized:[ fc.LL.fc_tn ])
  in
  p_all "every materialize flip has a finite positive floor" floors ~f:(function
    | Some floor -> Float.is_finite floor && Float.(floor > 0.)
    | None -> false);
  let enough_margin =
    List.for_all floors ~f:(function Some floor -> decisive ~incumbent ~floor | None -> false)
  in
  Stdio.eprintf
    "bound pruning (not part of the golden): incumbent=%g ms, timed=%d refused=%d, searches=%d, \
     inline=%d, decisive=%b\n"
    incumbent arm_a.Autotune.candidates_timed arm_a.Autotune.timings_contended
    (List.length !flip_reports)
    (List.length inline_flips_on_surface)
    enough_margin;
  p "an absent incumbent has its own refused measurement evidence"
    ((Float.is_finite incumbent && Float.(incumbent > 0.))
    || Float.(incumbent = infinity)
       && arm_a.Autotune.candidates_timed = 0
       && arm_a.Autotune.timings_contended > 0);
  let claim = "every materialize flip was fathomed: only the inline flips reached a search" in
  if enough_margin then p claim (List.length !flip_reports = List.length inline_flips_on_surface)
  else (
    Stdio.eprintf "bound pruning: undecided — no finite incumbent with a twofold floor margin\n";
    skipped ~aggregation:`Environment ~backend:"cc" claim)

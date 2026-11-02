(* The advisory fallback of the model-picked untuned default (gh-ocannl-519).

   [Autotune.model_default] documents its schedule pick as advisory: any failure degrades to the
   ordinary default pipeline. The pick's own construction was guarded, but the checks that reject a
   bad pick — [Low_level.validate_parallel] above all — run inside backend codegen, past the
   transform seam, so a rejected pick used to escape the guard and abort the compile.

   The scenario here is that escape, reproduced with an explicit transform instead of a model pick
   (the model's picks are envelope- and backend-dependent; the failure mode is not): a kernel whose
   first nest is Grid+Workgroup-annotated and whose second is Grid-only, so the second nest's
   materialized write is not covered by all active hardware dimensions. The rejection is
   backend-independent (test/operations/hardware_axes_parity.ml pins it directly).

   Run by a rule (not a test stanza) with --ocannl_backend=cc and the model_peak_* envelope
   overrides pinned, like cost_model_selection.ml: the C backends carry no advisory envelope
   constants, so pinning them is what makes the [model_default] leg exercise live model scoring. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Asgns = Ir.Assignments
open Verdict.Claims

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Every top-level nest annotated Grid, with the FIRST one's inner loop additionally Workgroup: the
   launch then has an active workgroup dimension that the second nest's write does not cover. *)
let annotate_mixed (opt : LL.optimized) : LL.optimized list =
  let first = ref true in
  let rec map_inner (llc : LL.t) : LL.t =
    match llc with
    | LL.Seq (x, y) -> LL.Seq (map_inner x, map_inner y)
    | LL.For_loop fc -> LL.For_loop { fc with axis = LL.Workgroup }
    | other -> other
  in
  let rec map_outer (llc : LL.t) : LL.t =
    match llc with
    | LL.Seq (x, y) -> LL.Seq (map_outer x, map_outer y)
    | LL.For_loop fc ->
        let body = if !first then map_inner fc.body else fc.body in
        first := false;
        LL.For_loop { fc with axis = LL.Grid; body }
    | other -> other
  in
  [ { opt with llc = map_outer opt.llc } ]

let av = Array.init 32 ~f:(fun i -> Float.of_int i *. 0.5)
let bv = Array.init 32 ~f:(fun i -> Float.of_int (i % 7) -. 3.)
let ev = Array.init 48 ~f:(fun i -> Float.of_int i *. 0.25)
let fv = Array.init 48 ~f:(fun i -> Float.of_int ((i % 5) + 1))
let expected_c = Array.init 32 ~f:(fun i -> av.(i) +. bv.(i))
let expected_d = Array.init 48 ~f:(fun i -> ev.(i) *. fv.(i))

(* Two independent elementwise nests, so the annotation above has a second nest to leave uncovered.
   A fresh tensor set per leg: each leg compiles its own routine from a fresh context. *)
let pair tag =
  let a = TDSL.ndarray av ~label:[ "mdf_a_" ^ tag ] ~output_dims:[ 4; 8 ] () in
  let b = TDSL.ndarray bv ~label:[ "mdf_b_" ^ tag ] ~output_dims:[ 4; 8 ] () in
  let e = TDSL.ndarray ev ~label:[ "mdf_e_" ^ tag ] ~output_dims:[ 6; 8 ] () in
  let f = TDSL.ndarray fv ~label:[ "mdf_f_" ^ tag ] ~output_dims:[ 6; 8 ] () in
  let%op c = a + b in
  let%op d = e *. f in
  (c, d, named ("mdf_" ^ tag) (Asgns.sequence [ Train.forward c; Train.forward d ]))

(* Both readbacks against their references as one boolean, for a claim that stays one row:
   [Verdict.within] refuses an empty pair and names the failing readback on stderr. [name] is the
   claim's label. *)
let values name ctx (c : Tensor.t) (d : Tensor.t) =
  let tol _ = 1e-4 in
  Verdict.within ~what_ran:"c" name (Context.get_values ctx c.Tensor.value) expected_c ~tol
  && Verdict.within ~what_ran:"d" name (Context.get_values ctx d.Tensor.value) expected_d ~tol

let () =
  (* --- The escape: the rejection happens in codegen, past the transform seam --- *)
  let _c, _d, comp = pair "escape" in
  let raised =
    try
      ignore
        (Context.compile ~lowered_transform:annotate_mixed (Context.auto ()) comp Ir.Indexing.Empty
          : Context.t * Context.routine);
      None
    with Invalid_argument msg -> Some msg
  in
  p "an unguarded compile of the rejected transform raises out of Context.compile"
    (match raised with
    | Some msg -> String.is_substring msg ~substring:"all active hardware dimensions"
    | None -> false);

  let _c, _d, comp = pair "classified" in
  let classified =
    Context.compile_outcome ~lowered_transform:annotate_mixed
      ~provenance:Ir.Schedule_outcome.Candidate ~candidate:"bad-annotation" (Context.auto ()) comp
      Ir.Indexing.Empty
  in
  p "candidate compile retains the validation rejection key"
    (match classified with
    | Error
        (Ir.Schedule_outcome.Classified
           {
             phase = Ir.Schedule_outcome.Backend_codegen;
             cause =
               Ir.Schedule_outcome.Illegal_schedule
                 { check = "Low_level.validate_parallel"; detail = _ };
             execution_effect = Ir.Schedule_outcome.No_device_writes;
           }) ->
        true
    | Ok _ | Error (Ir.Schedule_outcome.Classified _) | Error (Ir.Schedule_outcome.Fatal _) -> false);

  (* --- The backstop: compile_advisory falls back and still returns a working routine --- *)
  let c, d, comp = pair "advisory" in
  let fell_back = ref None in
  let ctx, routine =
    Autotune.compile_advisory
      ~on_fallback:(fun exn -> fell_back := Some (Exn.to_string exn))
      annotate_mixed (Context.auto ()) comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  p "compile_advisory falls back on a transform rejected downstream of the seam"
    (match !fell_back with
    | Some msg -> String.is_substring msg ~substring:"all active hardware dimensions"
    | None -> false);
  p "the fallback routine computes the default pipeline's values"
    (values "the fallback routine computes the default pipeline's values" ctx c d);

  (* --- ... and not when the caller says there is nothing to fall back to (Codex P2) --- *)
  let _c, _d, comp = pair "nogate" in
  let fell_back = ref None in
  let raised =
    try
      ignore
        (Autotune.compile_advisory
           ~on_fallback:(fun exn -> fell_back := Some (Exn.to_string exn))
           ~fallback_if:(fun () -> false)
           annotate_mixed (Context.auto ()) comp Ir.Indexing.Empty
          : Context.t * Context.routine);
      None
    with Invalid_argument msg -> Some msg
  in
  p "fallback_if false re-raises instead of recompiling"
    ((match raised with
       | Some msg -> String.is_substring msg ~substring:"all active hardware dimensions"
       | None -> false)
    && Option.is_none !fell_back);

  (* Fatal compiler failures are not advisory declines: they propagate once, with their original
     backtrace, and never invoke the fallback callback. *)
  Stdlib.Printexc.record_backtrace true;
  let _c, _d, comp = pair "fatal_assert" in
  let fell_back = ref false in
  let assert_propagated_with_backtrace =
    try
      ignore
        (Autotune.compile_advisory
           ~on_fallback:(fun _ -> fell_back := true)
           (fun _ -> assert false)
           (Context.auto ()) comp Ir.Indexing.Empty
          : Context.t * Context.routine);
      false
    with
    | Assert_failure _ ->
        Stdlib.Printexc.raw_backtrace_length (Stdlib.Printexc.get_raw_backtrace ()) > 0
    | _ -> false
  in
  p "compile_advisory propagates assertions with a backtrace"
    (assert_propagated_with_backtrace && not !fell_back);
  let _c, _d, comp = pair "fatal_failure" in
  let fell_back = ref false in
  let strict_failure_propagated =
    try
      ignore
        (Autotune.compile_advisory
           ~on_fallback:(fun _ -> fell_back := true)
           (fun _ -> failwith "advisory compiler bug")
           (Context.auto ()) comp Ir.Indexing.Empty
          : Context.t * Context.routine);
      false
    with Failure msg -> String.equal msg "advisory compiler bug"
  in
  p "compile_advisory propagates unclassified failures in strict mode"
    (strict_failure_propagated && not !fell_back);

  (* --- ... and does not fire when the transform is fine (no blanket swallowing) --- *)
  let c, d, comp = pair "clean" in
  let fell_back = ref None in
  let ctx, routine =
    Autotune.compile_advisory
      ~on_fallback:(fun exn -> fell_back := Some (Exn.to_string exn))
      (fun opt -> [ opt ])
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  p "compile_advisory does not fall back on a transform that compiles" (Option.is_none !fell_back);
  p "the non-fallback routine computes correct values"
    (values "the non-fallback routine computes correct values" ctx c d)

let () =
  (* --- model_default keeps working end to end (envelope overrides make scoring live) --- *)
  let c, d, comp = pair "model" in
  let choice = ref None in
  let ctx, routine =
    Autotune.model_default
      ~report:(fun r -> choice := Some r)
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  p "model_default returns a working routine"
    (values "model_default returns a working routine" ctx c d);
  p "model_default reports a choice"
    (match !choice with
    | Some r -> (not (String.is_empty r.Autotune.mc_label)) && r.Autotune.mc_scored >= 1
    | None -> false)

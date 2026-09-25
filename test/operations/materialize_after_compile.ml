(* The load-bearing properties of [Train.tune_placements] (placement A/B), pinned deterministically
   via the two-level memory-mode split: tnode-level [memory_mode] is declared, semantics-bearing
   intent; placement decisions are context-level and functional ([Context.decide_materialized]
   returns a child context, leaving the argument context and the declared intent untouched). So the
   A/B arms are hermetic siblings and no virtual-to- materialized conversion (which the intent
   lattice rejects) is ever requested:

   1. A default-placement compile decides the intermediate Virtual only in its own lineage — the
   tnode's declared intent stays unspecified. 2. A compile from a [decide_materialized] sibling
   materializes the intermediate for real, still without touching intent. 3. The default-placement
   routine remains valid alongside it and computes the same values. 4.
   [Train.every_non_literal_materialized] (the intent-level strengthening used by the benchmark's
   "materialized" variant) stays legal after all of the above: it only strengthens unspecified
   intent. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module SC = Ir.Schedule_cache
open Verdict.Claims

let approx a b = Float.(abs (a -. b) < 1e-4)
let n = 8

let () =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t2 = relu mc in
  let comp = Train.forward t2 in
  let ctx = Context.auto () in
  p "intermediate intent unspecified before compiling" (Tn.mode_is_unspecified mc.Tensor.value);
  (* Arm A: the default-placement compile — the matmul intermediate inlines into the pointwise
     consumer, so this lineage decides it Virtual. The identity transform also captures the
     optimized code for the digest check below. *)
  let opt_a = ref None in
  let ctx_a, routine_a =
    Context.compile
      ~lowered_transform:(fun o ->
        opt_a := Some o;
        [ o ])
      ctx comp Ir.Indexing.Empty
  in
  p "A-arm lineage decides the intermediate virtual (not materialized)"
    (not (Tn.Placements.is_materialized_peek (Context.placements ctx_a) mc.Tensor.value));
  p "intermediate intent still unspecified after the A-arm compile"
    (Tn.mode_is_unspecified mc.Tensor.value);
  (* Arm B: a [decide_materialized] sibling materializes for real. *)
  let embedded = ref [] in
  Tensor.iter_embedded ~f:(fun tn -> embedded := tn :: !embedded) t2;
  let ctx_b, routine_b =
    Context.compile (Context.decide_materialized ctx !embedded) comp Ir.Indexing.Empty
  in
  p "B-arm lineage materializes the intermediate"
    (Tn.Placements.is_materialized_peek (Context.placements ctx_b) mc.Tensor.value);
  p "intermediate intent still unspecified after the B-arm compile"
    (Tn.mode_is_unspecified mc.Tensor.value);
  p "the parent context is unaffected by the B-arm's decisions"
    (Option.is_none (Tn.Placements.get (Context.placements ctx) mc.Tensor.value));
  (* Both arms coexist and agree. *)
  let ctx_a = Context.run ctx_a routine_a in
  let got_a = Context.get_values ctx_a t2.Tensor.value in
  let ctx_b = Context.run ctx_b routine_b in
  let got_b = Context.get_values ctx_b t2.Tensor.value in
  p_all2 "A-arm and B-arm routines compute the same values" got_a got_b ~f:approx;
  (* The autotune-cache identity must distinguish placements (Codex P1 on PR #140): optimized code
     can be identical while placements differ (Local scratch vs On_device buffer), and the A/B arms
     must not cache-hit each other's entries in that case. Pin the mechanism directly: flipping one
     node's placement class in a copied optimize_ctx changes the canonical digest. *)
  let opt = Option.value_exn !opt_a in
  let d1 = SC.digest (SC.canonicalize opt) in
  let flipped_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx in
  Tn.Placements.unsafe_restore flipped_ctx.LL.placements mc.Tensor.value
    (Some (Tn.On_device, Tn.Site "999:test-setup"));
  let d2 = SC.digest (SC.canonicalize { opt with LL.optimize_ctx = flipped_ctx }) in
  p "canonical digest distinguishes placement classes" (not (String.equal d1 d2));
  (* Intent-level strengthening (the "materialized" benchmark variant) stays legal afterwards: it
     only touches unspecified intent, never requesting virtual-to-materialized. *)
  let strengthened =
    match Train.every_non_literal_materialized t2 with () -> true | exception _ -> false
  in
  p "every_non_literal_materialized after both compiles does not raise" strengthened

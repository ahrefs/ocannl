(* gh-ocannl-805: the assertion at the [virtual_llc] / cleanup boundary.

   The broken shape cannot be produced by the complete optimizer today -- that is the invariant the
   assertion protects -- so the negative control honestly injects hand-built IR into the exposed
   seam validator with a fresh placement table. The positive control sends the same read through
   [Ll_test.optimize], proving the normal analysis + virtualization path decides it and reaches
   cleanup without tripping the seam. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
open Verdict.Claims

let () =
  let node = Ll_test.node_factory ~first_id:10600 ~dims:[| 1 |] () in
  let source = node "vdc_source" in
  let candidate = node "vdc_candidate" in
  let output = node "vdc_output" in
  Ll_test.materialize output;
  (* Setter first, reader second: without the seam, cleanup can commit [candidate] Virtual and drop
     its setter before its read asks for Never_virtual -- the order-dependent collision #805 is
     about. The source read also proves the positive path checks inside the inlined scope body. *)
  let program =
    Ll_test.seq
      (Ll_test.set_at candidate (Ll_test.fixed 0) (Ll_test.get source [| Ll_test.fixed 0 |]))
      (Ll_test.set_at output (Ll_test.fixed 0) (Ll_test.get candidate [| Ll_test.fixed 0 |]))
  in

  let undecided_ctx = LL.empty_optimize_ctx () in
  let rejection =
    match LL.validate_virtualization_decision_coverage undecided_ctx.LL.placements program with
    | () -> None
    | exception Invalid_argument msg -> Some msg
  in
  p "the seam rejects a surviving read with no virtualization decision" (Option.is_some rejection);
  let says substring =
    Option.value_map rejection ~default:false ~f:(String.is_substring ~substring)
  in
  p "the seam refusal names the pass boundary" (says "virtual_llc decision-coverage seam");
  p "the seam refusal names the undecided node"
    (Option.exists rejection ~f:(String.is_substring ~substring:(Tn.debug_name candidate)));
  p "the seam refusal states the missing decision" (says "no virtualization decision");
  p "the seam refusal states which reads carry the obligation" (says "Get / Get_dynamic");

  let optimized = Ll_test.optimize ~materialized:[ output ] ~name:"vdc_normal" program in
  p "normal optimization virtualizes the producer" (Ll_test.known_virtual optimized candidate);
  p "normal optimization replaces the candidate read" (Ll_test.count_get optimized candidate = 0);
  p "normal optimization decides the source read nested in the inlined body"
    (Ll_test.known_non_virtual optimized source && Ll_test.count_get optimized source = 1)

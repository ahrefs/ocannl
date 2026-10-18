(* [simplify_llc]'s constant-folding arm for a constant minus a constant-offset sum must keep the
   negation of the non-constant summand: [c1 - (c2 + x)] is [(c1 - c2) - x], not [(c1 - c2) + x].
   With [c1 = 1], [c2 = 2], [x = 3] the correct value is [-4]; dropping the negation gives [2]. All
   values are small integers, exact in every precision used here, so the executed results are exact.

   The arm is licensed for integer precisions whatever [simplify_fp_algebra] selects, so the int32
   legs simplify under the empty selection ["none"]: an integer program reaches it
   unconditionally. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
open Verdict.Claims

(* [c1 - sum], where [sum] places the constant [c2] on the left or the right of [x]. *)
let case ~prec ~first_id ~const_left label =
  let mk = Ll_test.node_factory ~prec ~first_id ~dims:[| 1 |] () in
  let input = mk (label ^ "_input") and output = mk (label ^ "_output") in
  Ll_test.materialize input;
  Ll_test.materialize output;
  let x = (LL.Get (input, [| Idx.Fixed_idx 0 |]), prec) in
  let c2 = (LL.Constant 2., prec) in
  let sum = if const_left then LL.Binop (Ops.Add, c2, x) else LL.Binop (Ops.Add, x, c2) in
  let rhs = LL.Binop (Ops.Sub, (LL.Constant 1., prec), (sum, prec)) in
  (Ll_test.set_at output (Idx.Fixed_idx 0) rhs, input, output)

(* The folded form: [(c1 - c2) - x] with the constant [-1] on the left of a subtraction. *)
let folded_subtractions llc =
  Ll_test.count_scalar llc ~f:(function
    | LL.Binop (Ops.Sub, (LL.Constant c, _), (LL.Get _, _)) -> Float.equal c (-1.)
    | _ -> false)

let added_reads llc =
  Ll_test.count_scalar llc ~f:(function
    | LL.Binop (Ops.Add, (LL.Get _, _), _) | LL.Binop (Ops.Add, _, (LL.Get _, _)) -> true
    | _ -> false)

let check ~prec ~fp_algebra ~first_id ~const_left label =
  let llc, input, output = case ~prec ~first_id ~const_left label in
  let simplified = LL.simplify_llc ~fp_algebra [] llc in
  p (label ^ ": folds to (c1 - c2) - x") (folded_subtractions simplified = 1);
  p (label ^ ": x is not added after folding") (added_reads simplified = 0);
  let name = "csub_" ^ label in
  let optimized = Ll_test.optimize ~materialized:[ input; output ] ~name llc in
  let got = Ll_test.execute ~name optimized ~seed:[ (input, [| 3. |]) ] ~read:[ output ] in
  p (label ^ ": 1 - (2 + x) at x = 3 evaluates to -4") (Ll_test.same got [ [| -4. |] ])

let () =
  check ~prec:Ops.single ~fp_algebra:"all" ~first_id:1239000 ~const_left:true "single_const_left";
  check ~prec:Ops.single ~fp_algebra:"all" ~first_id:1239010 ~const_left:false "single_const_right";
  check ~prec:Ops.int32 ~fp_algebra:"none" ~first_id:1239020 ~const_left:true "int32_const_left";
  check ~prec:Ops.int32 ~fp_algebra:"none" ~first_id:1239030 ~const_left:false "int32_const_right"

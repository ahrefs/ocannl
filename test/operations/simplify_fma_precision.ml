(* gh-ocannl-824: [simplify_llc]'s mul-add rewrite must stay restricted to floating-point
   evaluation. Integer precisions otherwise reach the C-family [fma] renderers through hand-built
   low-level IR, converting through a floating mantissa and changing exact integer arithmetic. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
open Verdict.Claims

let fmas llc =
  Ll_test.count_scalar llc ~f:(function LL.Ternop (Ops.FMA, _, _, _) -> true | _ -> false)

let scalar_case ~prec ~first_id label =
  let mk = Ll_test.node_factory ~prec ~first_id ~dims:[| 1 |] () in
  let input = mk (label ^ "_input") and output = mk (label ^ "_output") in
  Ll_test.materialize input;
  Ll_test.materialize output;
  let get = (LL.Get (input, [| Idx.Fixed_idx 0 |]), prec) in
  let one = (LL.Constant 1., prec) in
  let mul_add = (LL.Binop (Ops.Add, (LL.Binop (Ops.Mul, get, one), prec), one), prec) in
  let rhs = LL.Binop (Ops.Sub, mul_add, get) in
  (Ll_test.set_at output (Idx.Fixed_idx 0) rhs, input, output)

let () =
  let int_llc, input, output = scalar_case ~prec:Ops.int64 ~first_id:8240 "int64" in
  let float_llc, _, _ = scalar_case ~prec:Ops.single ~first_id:8250 "single" in
  let int_simplified = LL.simplify_llc ~fp_algebra:"all" [] int_llc in
  let float_simplified = LL.simplify_llc ~fp_algebra:"all" [] float_llc in
  p "int64 mul-add remains integer arithmetic" (fmas int_simplified = 0);
  p "single-precision mul-add still rewrites to FMA" (fmas float_simplified = 1);

  (* [2^53] is exactly representable as both int64 and double, but [2^53 + 1] is not. The intended
     integer evaluation [(x * 1 + 1) - x] therefore returns 1; an intervening double [fma] rounds
     the sum back to [2^53] and returns 0. *)
  let optimized = Ll_test.optimize ~materialized:[ input; output ] ~name:"fma_int64" int_llc in
  let got =
    Ll_test.execute ~name:"fma_int64" optimized
      ~seed:[ (input, [| 9007199254740992. |]) ]
      ~read:[ output ]
  in
  p "int64 mul-add preserves the unit past 2^53" (Ll_test.same got [ [| 1. |] ])

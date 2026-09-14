(* gh-ocannl-873: FMA is a floating-point-only operation at the C-family codegen boundary.
   Hand-built low-level IR can bypass the simplifier's gh-ocannl-824 float guard, so codegen must
   reject an integer FMA rather than render it through [fma]. A direct single-precision twin proves
   the supported path still renders and executes. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
open Verdict.Claims

let direct_fma_case ~prec ~first_id label =
  let mk = Ll_test.node_factory ~prec ~first_id ~dims:[| 1 |] () in
  let a = mk (label ^ "_a")
  and b = mk (label ^ "_b")
  and c = mk (label ^ "_c")
  and output = mk (label ^ "_output") in
  List.iter [ a; b; c; output ] ~f:Ll_test.materialize;
  let at_zero tn = (LL.Get (tn, [| Idx.Fixed_idx 0 |]), prec) in
  let rhs = LL.Ternop (Ops.FMA, at_zero a, at_zero b, at_zero c) in
  (Ll_test.set_at output (Idx.Fixed_idx 0) rhs, a, b, c, output)

let () =
  let cases = List.mapi Ops.integer_precs ~f:(fun i prec -> (prec, 8730 + (10 * i))) in
  p_all ~min:6 "every integer FMA is rejected at the C-family codegen boundary" cases
    ~f:(fun (prec, first_id) ->
      let label = Ops.prec_string prec in
      let llc, _, _, _, _ = direct_fma_case ~prec ~first_id label in
      let name = label ^ "_fma_codegen" in
      let optimized = Ll_test.optimize ~name llc in
      match Ll_test.link ~name optimized with
      | _ -> false
      | exception Invalid_argument message ->
          String.is_substring message
            ~substring:
              (Printf.sprintf "C_syntax.pp_scalar: FMA requires floating-point precision, got %s"
                 label));

  let first_id = 8730 + (10 * List.length cases) in
  let float_llc, a, b, c, output = direct_fma_case ~prec:Ops.single ~first_id "single" in
  let float_optimized = Ll_test.optimize ~name:"float_fma_codegen" float_llc in
  let got =
    Ll_test.execute ~name:"float_fma_codegen" float_optimized
      ~seed:[ (a, [| 2. |]); (b, [| 3. |]); (c, [| 4. |]) ]
      ~read:[ output ]
  in
  p "single-precision FMA renders and executes" (Ll_test.same got [ [| 10. |] ])

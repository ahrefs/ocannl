(* Residual Concat maps are structural probes only: lowering eliminates Concat before
   optimization/codegen. Its constituent symbols nevertheless affect recognition, just as
   Iterator/Affine symbols do (gh-ocannl-1187). *)
open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Idx = Ir.Indexing
module LL = Ir.Low_level

let () =
  let open Ll_test in
  let node = node_factory ~first_id:118700 ~dims:[| 8; 8 |] () in
  let d = node "concat_d" and a = node "concat_a" and b = node "concat_b" in
  let i = sym () and j = sym () and k = sym () in
  let di = [| iter i; iter j |] in
  let nest ~di ~ai =
    loop_n i 8
      (loop_n j 8
         (loop_n k 128 (set d di (add (get d di) (mul (get a ai) (get b [| iter k; iter j |]))))))
  in
  let ordinary = nest ~di ~ai:[| iter i; iter k |] in
  p "ordinary matmul remains recognized" (Option.is_some (Autotune.detect_matmul ordinary));
  let hidden_output = nest ~di ~ai:[| iter i; iter k; Idx.Concat [ j ] |] in
  p "Concat operand dependency on the output column refuses matmul"
    (Option.is_none (Autotune.detect_matmul hidden_output));
  let hidden_reduction = nest ~di ~ai:[| iter i; iter k; Idx.Concat [ k ] |] in
  p "Concat occurrence beside a plain reduction axis refuses matmul"
    (Option.is_none (Autotune.detect_matmul hidden_reduction));
  let concat_di = [| iter i; Idx.Concat [ j; k ] |] in
  let concat_output = nest ~di:concat_di ~ai:[| iter i; iter k |] in
  p "concatenated output is not a matmul contraction"
    (Option.is_none (Autotune.detect_matmul concat_output));
  (* Neither optimize nor codegen accepts residual Concat. Reuse an ordinary optimized lineage and
     replace only its code for this defensive site-detection probe. *)
  materialize d;
  let base = optimize ~name:"concat_dependency" ordinary in
  p "ordinary reduction remains eligible for split-reduce"
    (match Autotune.split_reduce_sites base with
    | [ site ] -> Idx.equal_symbol site.Autotune.sr_axis k
    | _ -> false);
  let residual = { base with LL.llc = concat_output } in
  p_empty "concatenated output yields no split-reduce site" ~over:(LL.loop_bounds residual.LL.llc)
    (Autotune.split_reduce_sites residual);
  let unrelated = sym () in
  p "Concat mentions each constituent and no unrelated symbol"
    (Idx.axis_index_mentions_symbol j (Idx.Concat [ j; k ])
    && Idx.axis_index_mentions_symbol k (Idx.Concat [ j; k ])
    && not (Idx.axis_index_mentions_symbol unrelated (Idx.Concat [ j; k ])))

(** Tests for ternary einsum: einsum3 and where with einsum spec. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
let get_vals ctx t = Context.get_values ctx t.Tensor.value

let get_grad ctx t =
  Context.get_values ctx (Option.value_exn ~here:[%here] t.Tensor.diff).grad

let fmt_arr vals =
  "[" ^ String.concat ~sep:" " (Array.to_list (Array.map vals ~f:(Printf.sprintf "%g"))) ^ "]"

(* ---- Test 1: einsum3 chain contraction matches binary chain (falsifier test) ---- *)
let () =
  Stdio.printf "Test 1: chain contraction\n";
  (* a = [[1,2],[3,4]] stored i=2(output), j=2(input), b = swap, c = identity *)
  (* Ternary: sum_{j,k} a[i,j]*b[j,k]*c[k,m]; Binary: (a @ b) @ c *)
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let a = PDSL.ndarray [| 1.; 2.; 3.; 4. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let b = PDSL.ndarray [| 0.; 1.; 1.; 0. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let c = PDSL.ndarray [| 1.; 0.; 0.; 1. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let tern = Operation.einsum3 "ij;jk;km=>im" ~grad_spec:Prohibit_grad a b c () in
  let ctx = Train.forward_once ~output_cd_file:false ctx tern in
  Stdio.printf "einsum3 = %s\n" (fmt_arr (get_vals ctx tern));

  Tensor.unsafe_reinitialize ();
  let ctx2 = Context.auto () in
  let a2 = NTDSL.ndarray [| 1.; 2.; 3.; 4. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let b2 = NTDSL.ndarray [| 0.; 1.; 1.; 0. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let c2 = NTDSL.ndarray [| 1.; 0.; 0.; 1. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let ab2 = Operation.einsum "ij;jk=>ik" a2 b2 ~grad_spec:Prohibit_grad () in
  let bin = Operation.einsum "ik;km=>im" ab2 c2 ~grad_spec:Prohibit_grad () in
  let ctx2 = Train.forward_once ~output_cd_file:false ctx2 bin in
  Stdio.printf "binary chain = %s\n" (fmt_arr (get_vals ctx2 bin));
  Stdio.printf "chain contraction match: %b\n"
    (Array.equal Float.equal (get_vals ctx tern) (get_vals ctx2 bin))

(* ---- Test 2: batched einsum3 matches binary chain ---- *)
let () =
  Stdio.printf "\nTest 2: batched chain contraction\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let a = PDSL.ndarray [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
            ~batch_dims:[ 2 ] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let b = PDSL.ndarray [| 0.; 1.; 1.; 0.; 1.; 0.; 0.; 1. |]
            ~batch_dims:[ 2 ] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let c = PDSL.ndarray [| 1.; 0.; 0.; 1.; 0.; 1.; 1.; 0. |]
            ~batch_dims:[ 2 ] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let tern = Operation.einsum3 "bij;bjk;bkm=>bim" ~grad_spec:Prohibit_grad a b c () in
  let ctx = Train.forward_once ~output_cd_file:false ctx tern in
  Stdio.printf "batched einsum3 = %s\n" (fmt_arr (get_vals ctx tern));

  Tensor.unsafe_reinitialize ();
  let ctx2 = Context.auto () in
  let a2 = NTDSL.ndarray [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
             ~batch_dims:[ 2 ] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let b2 = NTDSL.ndarray [| 0.; 1.; 1.; 0.; 1.; 0.; 0.; 1. |]
             ~batch_dims:[ 2 ] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let c2 = NTDSL.ndarray [| 1.; 0.; 0.; 1.; 0.; 1.; 1.; 0. |]
             ~batch_dims:[ 2 ] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let ab2 = Operation.einsum "bij;bjk=>bik" a2 b2 ~grad_spec:Prohibit_grad () in
  let bin = Operation.einsum "bik;bkm=>bim" ab2 c2 ~grad_spec:Prohibit_grad () in
  let ctx2 = Train.forward_once ~output_cd_file:false ctx2 bin in
  Stdio.printf "batched binary chain = %s\n" (fmt_arr (get_vals ctx2 bin));
  Stdio.printf "batched chain contraction match: %b\n"
    (Array.equal Float.equal (get_vals ctx tern) (get_vals ctx2 bin))

(* ---- Test 3: einsum3 gradient agreement with binary chain ---- *)
let () =
  Stdio.printf "\nTest 3: gradient agreement\n";
  (* a = I, b = swap, c = I: (a@b)@c = swap. With loss=sum(out), grads should match. *)
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let a = PDSL.ndarray [| 1.; 0.; 0.; 1. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let b = PDSL.ndarray [| 0.; 1.; 1.; 0. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let c = PDSL.ndarray [| 1.; 0.; 0.; 1. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  Train.set_materialized (Option.value_exn ~here:[%here] a.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] b.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] c.diff).grad;
  let tern = Operation.einsum3 "ij;jk;km=>im" ~grad_spec:Require_grad a b c () in
  let ctx = Train.update_once ~output_cd_file:false ctx tern in
  let ga_tern = get_grad ctx a in
  let gb_tern = get_grad ctx b in
  let gc_tern = get_grad ctx c in
  Stdio.printf "einsum3 grad_a = %s\n" (fmt_arr ga_tern);
  Stdio.printf "einsum3 grad_b = %s\n" (fmt_arr gb_tern);
  Stdio.printf "einsum3 grad_c = %s\n" (fmt_arr gc_tern);

  Tensor.unsafe_reinitialize ();
  let ctx2 = Context.auto () in
  let a2 = PDSL.ndarray [| 1.; 0.; 0.; 1. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let b2 = PDSL.ndarray [| 0.; 1.; 1.; 0. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let c2 = PDSL.ndarray [| 1.; 0.; 0.; 1. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  Train.set_materialized (Option.value_exn ~here:[%here] a2.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] b2.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] c2.diff).grad;
  let ab2 = Operation.einsum "ij;jk=>ik" a2 b2 ~grad_spec:If_needed () in
  let bin = Operation.einsum "ik;km=>im" ab2 c2 ~grad_spec:Require_grad () in
  let ctx2 = Train.update_once ~output_cd_file:false ctx2 bin in
  let ga_bin = get_grad ctx2 a2 in
  let gb_bin = get_grad ctx2 b2 in
  let gc_bin = get_grad ctx2 c2 in
  Stdio.printf "binary  grad_a = %s\n" (fmt_arr ga_bin);
  Stdio.printf "binary  grad_b = %s\n" (fmt_arr gb_bin);
  Stdio.printf "binary  grad_c = %s\n" (fmt_arr gc_bin);
  Stdio.printf "gradient a match: %b\n" (Array.equal Float.equal ga_tern ga_bin);
  Stdio.printf "gradient b match: %b\n" (Array.equal Float.equal gb_tern gb_bin);
  Stdio.printf "gradient c match: %b\n" (Array.equal Float.equal gc_tern gc_bin)

(* ---- Test 4: where with einsum spec "i;i;i=>i" equals pointwise where ---- *)
let () =
  Stdio.printf "\nTest 4: where with einsum spec\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let pred = NTDSL.ndarray [| 1.; 0.; 1.; 0. |] ~output_dims:[ 4 ] () in
  let a    = PDSL.ndarray   [| 10.; 20.; 30.; 40. |] ~output_dims:[ 4 ] () in
  let b    = PDSL.ndarray   [| 1.; 2.; 3.; 4. |] ~output_dims:[ 4 ] () in
  let wein = Operation.where ~spec:"i;i;i=>i" ~grad_spec:Prohibit_grad pred a b () in
  let ctx = Train.forward_once ~output_cd_file:false ctx wein in
  Stdio.printf "where einsum 'i;i;i=>i' = %s\n" (fmt_arr (get_vals ctx wein));

  Tensor.unsafe_reinitialize ();
  let ctx2 = Context.auto () in
  let pred2 = NTDSL.ndarray [| 1.; 0.; 1.; 0. |] ~output_dims:[ 4 ] () in
  let a2    = PDSL.ndarray   [| 10.; 20.; 30.; 40. |] ~output_dims:[ 4 ] () in
  let b2    = PDSL.ndarray   [| 1.; 2.; 3.; 4. |] ~output_dims:[ 4 ] () in
  let wpt   = Operation.where ~grad_spec:Prohibit_grad pred2 a2 b2 () in
  let ctx2 = Train.forward_once ~output_cd_file:false ctx2 wpt in
  Stdio.printf "where pointwise      = %s\n" (fmt_arr (get_vals ctx2 wpt));
  Stdio.printf "where einsum matches pointwise: %b\n"
    (Array.equal Float.equal (get_vals ctx wein) (get_vals ctx2 wpt))

(* ---- Test 5: where einsum gradient (pred=1 passes to a, pred=0 passes to b) ---- *)
let () =
  Stdio.printf "\nTest 5: where einsum gradient\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let pred = NTDSL.ndarray [| 1.; 0.; 1.; 0. |] ~output_dims:[ 4 ] () in
  let a    = PDSL.ndarray   [| 10.; 20.; 30.; 40. |] ~output_dims:[ 4 ] () in
  let b    = PDSL.ndarray   [| 1.; 2.; 3.; 4. |] ~output_dims:[ 4 ] () in
  Train.set_materialized (Option.value_exn ~here:[%here] a.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] b.diff).grad;
  let wein = Operation.where ~spec:"i;i;i=>i" ~grad_spec:Require_grad pred a b () in
  let ctx = Train.update_once ~output_cd_file:false ctx wein in
  Stdio.printf "where einsum grad_a (expect [1 0 1 0]) = %s\n"
    (fmt_arr (get_grad ctx a));
  Stdio.printf "where einsum grad_b (expect [0 1 0 1]) = %s\n"
    (fmt_arr (get_grad ctx b))

(* ---- Test 6: %cd dispatch fix -- mul3 with einsum spec compiles and runs ---- *)
let () =
  Stdio.printf "\nTest 6: %%cd dispatch fix\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let a = NTDSL.ndarray [| 1.; 0.; 0.; 1. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let b = NTDSL.ndarray [| 0.; 1.; 1.; 0. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let c = NTDSL.ndarray [| 1.; 0.; 0.; 1. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let%cd fwd = { out } =:+ mul3 a b c ~logic:"ij;jk;km=>im" in
  Train.set_materialized out.value;
  let ctx = Train.init_params ctx Train.IDX.empty out in
  let routine = Train.to_routine ctx Train.IDX.empty fwd in
  let ctx = Context.context routine in
  Train.run ctx routine;
  Stdio.printf "%%cd mul3 einsum (identity*swap*identity) = %s\n"
    (fmt_arr (get_vals ctx out));
  Stdio.printf "%%cd dispatch fix: OK\n"

(* ---- Test 7: wrong-arity spec raises a clear Shape_error ---- *)
let () =
  Stdio.printf "\nTest 7: wrong-arity spec raises Shape_error\n";
  Tensor.unsafe_reinitialize ();
  let a = NTDSL.ndarray [| 1.; 0.; 0.; 1. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let b = NTDSL.ndarray [| 1.; 0.; 0.; 1. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let c = NTDSL.ndarray [| 1.; 0.; 0.; 1. |] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  (try
     let _ = Operation.einsum3 "ij;jk=>im" ~grad_spec:If_needed a b c () in
     Stdio.printf "FAIL: expected Shape_error for two-RHS spec in einsum3\n"
   with Row.Shape_error (msg, _) ->
     Stdio.printf "Negative test (wrong arity): got expected Shape_error: %s\n"
       (String.prefix msg 60))

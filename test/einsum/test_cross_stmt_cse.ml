open Ocannl
open Nn_blocks.DSL_modules
open Stdio

(* Cross-statement CSE test.

   Compiles two outputs (e, f) that share a common virtual reduction (c = a * b)
   into a single routine via Ir.Assignments.sequence.

   After virtualization, both e's and f's For_loops contain alpha-equivalent
   Local_scope nodes for c's reduction. The hoist_cross_statement_cse pass:
   1. Fuses the sibling For_loops (same range) into one loop
   2. Detects the duplicate Local_scope nodes as siblings within the fused body
   3. Hoists the shared computation via Declare_local + body
   4. Both Set statements reference the same local via Get_local

   The .ll output should show ONE reduction loop body and both e[i] and f[i]
   referencing the same v_c local, not two separate reductions. *)

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let a = TDSL.range_of_shape ~label:[ "a" ] ~input_dims:[ 4 ] ~output_dims:[ 3 ] () in
  let b = TDSL.range_of_shape ~label:[ "b" ] ~output_dims:[ 4 ] () in
  let%op c = a * b in
  let%op e = c + 1.0 in
  let%op f = c - 1.0 in
  Train.set_hosted a.value;
  Train.set_hosted b.value;
  Train.set_hosted e.value;
  Train.set_hosted f.value;
  let ctx = Train.init_params ctx Train.IDX.empty e in
  let ctx = Train.init_params ctx Train.IDX.empty f in
  let fwd_e = Train.forward e in
  let fwd_f = Train.forward f in
  let combined = Ir.Assignments.sequence [ fwd_e; fwd_f ] in
  let routine = Train.to_routine ctx Train.IDX.empty combined in
  Train.run ctx routine;
  (* c = [0*0+1*1+2*2+3*3, 4*0+5*1+6*2+7*3, 8*0+9*1+10*2+11*3] = [14, 38, 62] *)
  printf "e = c + 1:\n%!";
  Tensor.print ~here:[%here] ~force:true ~with_code:false ~with_grad:false `Inline e;
  printf "f = c - 1:\n%!";
  Tensor.print ~here:[%here] ~force:true ~with_code:false ~with_grad:false `Inline f;
  printf "\n%!"

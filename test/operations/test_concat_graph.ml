open Base
open Ocannl
open Nn_blocks.DSL_modules
open Stdio

(** Test concatenation operation with forward and backprop computation graph.

    Uses sin of concatenated tensor to show both forward and backward pass. *)

let () =
  printf "=== Concatenation Forward and Backprop Graph Test ===\n\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create two input tensors with different sizes *)
  let x1 =
    PDSL.ndarray [| 1.0; 2.0; 3.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let x2 = PDSL.ndarray [| 4.0; 5.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in

  (* Concatenate and apply sin for a nice differentiable function *)
  let%op result = sin ((x1, x2) ++^ "a; b => a^b") in

  (* Sum to scalar for backprop *)
  let%op loss = result ++ "...|... => 0" in

  let ctx = Context.auto () in

  (* Set tensors as hosted to enable printing *)
  Train.set_hosted loss.value;
  Train.set_hosted result.value;
  Train.set_hosted (Option.value_exn ~here:[%here] x1.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] x2.diff).grad;

  (* Run forward and backward pass *)
  ignore (Train.update_once ~output_cd_file:false ctx loss);

  printf "--- Input tensors ---\n%!";
  printf "x1 = [1.0, 2.0, 3.0] (dim 3)\n%!";
  printf "x2 = [4.0, 5.0] (dim 2)\n%!";
  printf "\n--- Forward pass: sin((x1, x2) ++^ \"a; b => a^b\") ---\n%!";
  printf "Concatenated result (after sin):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result;

  printf "\n--- Backward pass: gradients ---\n%!";
  printf "Loss (sum of all sin values):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false loss;

  printf "\nGradient of x1 (cos of original values):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true x1;

  printf "\nGradient of x2:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true x2;

  printf "\n=== Three-way Concatenation Forward and Backprop Graph Test ===\n\n%!";
  (* Create three input tensors with different sizes *)
  let x1_3 = PDSL.ndarray [| 1.0; 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let x2_3 =
    PDSL.ndarray [| 3.0; 4.0; 5.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let x3_3 =
    PDSL.ndarray [| 6.0; 7.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in

  (* Concatenate and apply sin *)
  let%op result3 = sin ((x1_3, x2_3, x3_3) ++^ "a; b; c => a^b^c") in

  (* Sum to scalar for backprop *)
  let%op loss3 = result3 ++ "...|... => 0" in

  let ctx3 = Context.auto () in

  (* Set tensors as hosted to enable printing *)
  Train.set_hosted loss3.value;
  Train.set_hosted result3.value;
  Train.set_hosted (Option.value_exn ~here:[%here] x1_3.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] x2_3.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] x3_3.diff).grad;

  (* Run forward and backward pass *)
  ignore (Train.update_once ~output_cd_file:false ctx3 loss3);

  printf "--- Input tensors ---\n%!";
  printf "x1_3 = [1.0, 2.0] (dim 2)\n%!";
  printf "x2_3 = [3.0, 4.0, 5.0] (dim 3)\n%!";
  printf "x3_3 = [6.0, 7.0] (dim 2)\n%!";
  printf "\n--- Forward pass: sin((x1_3, x2_3, x3_3) ++^ \"a; b; c => a^b^c\") ---\n%!";
  printf "Concatenated result (after sin):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result3;

  printf "\n--- Backward pass: gradients ---\n%!";
  printf "Loss (sum of all sin values):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false loss3;

  printf "\nGradient of x1_3 (cos of original values):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true x1_3;

  printf "\nGradient of x2_3:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true x2_3;

  printf "\nGradient of x3_3:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true x3_3;

  printf "\n=== Three-way Concat With Unit Dimension Test ===\n\n%!";
  (* One concatenated dimension is 1 to exercise Fixed_idx 0 behavior. *)
  let x1_u = PDSL.ndarray [| 1.0; 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let x2_u = PDSL.ndarray [| 3.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] () in
  let x3_u =
    PDSL.ndarray [| 4.0; 5.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in

  let%op result_u = sin ((x1_u, x2_u, x3_u) ++^ "a; b; c => a^b^c") in
  let%op loss_u = result_u ++ "...|... => 0" in

  let ctx_u = Context.auto () in

  Train.set_hosted loss_u.value;
  Train.set_hosted result_u.value;
  Train.set_hosted (Option.value_exn ~here:[%here] x1_u.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] x2_u.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] x3_u.diff).grad;

  let update_ok =
    try
      ignore (Train.update_once ~output_cd_file:false ctx_u loss_u);
      true
    with exn ->
      printf "ERROR: %s\n%!" (Exn.to_string exn);
      false
  in

  if update_ok then (
    printf "--- Input tensors ---\n%!";
    printf "x1_u = [1.0, 2.0] (dim 2)\n%!";
    printf "x2_u = [3.0] (dim 1)\n%!";
    printf "x3_u = [4.0, 5.0] (dim 2)\n%!";
    printf "\n--- Forward pass: sin((x1_u, x2_u, x3_u) ++^ \"a; b; c => a^b^c\") ---\n%!";
    printf "Concatenated result (after sin):\n%!";
    Train.printf ~here:[%here] ~with_code:false ~with_grad:false result_u;

    printf "\n--- Backward pass: gradients ---\n%!";
    printf "Loss (sum of all sin values):\n%!";
    Train.printf ~here:[%here] ~with_code:false ~with_grad:false loss_u;

    printf "\nGradient of x1_u (cos of original values):\n%!";
    Train.printf ~here:[%here] ~with_code:false ~with_grad:true x1_u;

    printf "\nGradient of x2_u:\n%!";
    Train.printf ~here:[%here] ~with_code:false ~with_grad:true x2_u;

    printf "\nGradient of x3_u:\n%!";
    Train.printf ~here:[%here] ~with_code:false ~with_grad:true x3_u);

  printf "\n=== Nested Concatenation Test ===\n\n%!";
  (* Concatenate results of concatenations: (a^b)^c *)
  let n1 = PDSL.ndarray [| 1.0; 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let n2 =
    PDSL.ndarray [| 3.0; 4.0; 5.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let n3 = PDSL.ndarray [| 6.0; 7.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in

  (* First concatenate n1 and n2, then concatenate with n3 *)
  let%op inner = (n1, n2) ++^ "a; b => a^b" in
  let%op nested = sin ((inner, n3) ++^ "a; b => a^b") in
  let%op loss_n = nested ++ "...|... => 0" in

  let ctx_n = Context.auto () in

  Train.set_hosted loss_n.value;
  Train.set_hosted nested.value;
  Train.set_hosted (Option.value_exn ~here:[%here] n1.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] n2.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] n3.diff).grad;

  ignore (Train.update_once ~output_cd_file:false ctx_n loss_n);

  printf "--- Input tensors ---\n%!";
  printf "n1 = [1.0, 2.0] (dim 2)\n%!";
  printf "n2 = [3.0, 4.0, 5.0] (dim 3)\n%!";
  printf "n3 = [6.0, 7.0] (dim 2)\n%!";
  printf "\n--- Forward pass: sin(((n1, n2) ++^ \"a; b => a^b\", n3) ++^ \"a; b => a^b\") ---\n%!";
  printf "Nested concatenation result (after sin):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false nested;

  printf "\n--- Backward pass: gradients ---\n%!";
  printf "Loss:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false loss_n;

  printf "\nGradient of n1:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true n1;

  printf "\nGradient of n2:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true n2;

  printf "\nGradient of n3:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true n3;

  printf "\n=== Boundary Unit Dim: First Component Dim 1 ===\n\n%!";
  (* First component has dim 1 (boundary position) *)
  let b1 = PDSL.ndarray [| 1.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] () in
  let b2 =
    PDSL.ndarray [| 2.0; 3.0; 4.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let b3 = PDSL.ndarray [| 5.0; 6.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in

  let%op result_b1 = sin ((b1, b2, b3) ++^ "a; b; c => a^b^c") in
  let%op loss_b1 = result_b1 ++ "...|... => 0" in

  let ctx_b1 = Context.auto () in

  Train.set_hosted loss_b1.value;
  Train.set_hosted result_b1.value;
  Train.set_hosted (Option.value_exn ~here:[%here] b1.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] b2.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] b3.diff).grad;

  ignore (Train.update_once ~output_cd_file:false ctx_b1 loss_b1);

  printf "--- Input tensors ---\n%!";
  printf "b1 = [1.0] (dim 1)\n%!";
  printf "b2 = [2.0, 3.0, 4.0] (dim 3)\n%!";
  printf "b3 = [5.0, 6.0] (dim 2)\n%!";
  printf "\n--- Forward pass ---\n%!";
  printf "Concatenated result (after sin):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result_b1;

  printf "\n--- Backward pass: gradients ---\n%!";
  printf "\nGradient of b1:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true b1;

  printf "\nGradient of b2:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true b2;

  printf "\nGradient of b3:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true b3;

  printf "\n=== Boundary Unit Dim: Last Component Dim 1 ===\n\n%!";
  (* Last component has dim 1 (boundary position) *)
  let l1 = PDSL.ndarray [| 1.0; 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let l2 =
    PDSL.ndarray [| 3.0; 4.0; 5.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let l3 = PDSL.ndarray [| 6.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] () in

  let%op result_l = sin ((l1, l2, l3) ++^ "a; b; c => a^b^c") in
  let%op loss_l = result_l ++ "...|... => 0" in

  let ctx_l = Context.auto () in

  Train.set_hosted loss_l.value;
  Train.set_hosted result_l.value;
  Train.set_hosted (Option.value_exn ~here:[%here] l1.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] l2.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] l3.diff).grad;

  ignore (Train.update_once ~output_cd_file:false ctx_l loss_l);

  printf "--- Input tensors ---\n%!";
  printf "l1 = [1.0, 2.0] (dim 2)\n%!";
  printf "l2 = [3.0, 4.0, 5.0] (dim 3)\n%!";
  printf "l3 = [6.0] (dim 1)\n%!";
  printf "\n--- Forward pass ---\n%!";
  printf "Concatenated result (after sin):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result_l;

  printf "\n--- Backward pass: gradients ---\n%!";
  printf "\nGradient of l1:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true l1;

  printf "\nGradient of l2:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true l2;

  printf "\nGradient of l3:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true l3;

  printf "\n=== Concatenation Graph Test Complete ===\n%!"

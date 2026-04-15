open Base
open Ocannl
open Nn_blocks.DSL_modules
open Stdio

(* Block tensor literal syntax tests.
   Disambiguation: first-leaf heuristic.
   - Numeric literal first leaf → ndarray constant (existing behavior).
   - Non-numeric first leaf → block tensor (unsqueeze + concat).
   - Computed-number expressions like [Float.sin 1.0; 2.0] are reclassified as block tensors
     (accepted compatibility break; such patterns don't exist in the project). *)

let () =
  printf "=== Block Tensor Literal Tests ===\n\n%!";
  Tensor.unsafe_reinitialize ();

  (* --- Test 1: List — output axis stacking [x1; x2] --- *)
  printf "--- Test 1: List output axis [x1; x2] ---\n%!";
  let x1 =
    PDSL.ndarray [| 1.0; 2.0; 3.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let x2 =
    PDSL.ndarray [| 4.0; 5.0; 6.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let%op stacked = [x1; x2] in
  let ctx = Context.auto () in
  Train.set_hosted stacked.value;
  let ctx = Train.forward_once ctx stacked in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Inline stacked;

  (* --- Test 2: Array — batch axis stacking [|x1; x2|] --- *)
  printf "\n--- Test 2: Array batch axis [|x1; x2|] ---\n%!";
  let x3 =
    PDSL.ndarray [| 10.0; 20.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let x4 =
    PDSL.ndarray [| 30.0; 40.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let%op batched = [|x3; x4|] in
  Train.set_hosted batched.value;
  let ctx = Train.forward_once ctx batched in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Inline batched;

  (* --- Test 3: Tuple — input axis stacking (x1, x2) --- *)
  printf "\n--- Test 3: Tuple input axis (x1, x2) ---\n%!";
  let%op input_stack = (x1, x2) in
  Train.set_hosted input_stack.value;
  let ctx = Train.forward_once ctx input_stack in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Inline input_stack;

  (* --- Test 4: 3-way list [x1; x2; x3] --- *)
  printf "\n--- Test 4: 3-way list [x1; x2; x3] ---\n%!";
  let x5 =
    PDSL.ndarray [| 7.0; 8.0; 9.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let%op triple = [x1; x2; x5] in
  Train.set_hosted triple.value;
  let ctx = Train.forward_once ctx triple in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Inline triple;

  (* --- Test 5: Scalars in block tensor --- *)
  (* NOTE: Nesting like [[s1; s2]; [s3; s4]] or [row1; row2] where row1/row2 are
     themselves block tensor results is currently limited by shape inference:
     the Concat dimension types produced by inner blocks can't be reconciled
     with the row variable (...) in the outer concat spec. Use explicit ++^ for nesting. *)
  printf "\n--- Test 5: Scalars in block tensor ---\n%!";
  let s1 = PDSL.ndarray [| 1.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[] () in
  let s2 = PDSL.ndarray [| 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[] () in
  let s3 = PDSL.ndarray [| 3.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[] () in
  let%op scalar_stack = [s1; s2; s3] in
  Train.set_hosted scalar_stack.value;
  let ctx = Train.forward_once ctx scalar_stack in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Inline scalar_stack;

  (* --- Test 6: Single element [x1] — unsqueeze --- *)
  printf "\n--- Test 6: Single element [x1] ---\n%!";
  let%op unsqueezed = [x1] in
  Train.set_hosted unsqueezed.value;
  let ctx = Train.forward_once ctx unsqueezed in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Inline unsqueezed;

  (* --- Test 7: Gradient flow (2-way) --- *)
  printf "\n--- Test 7: Gradient flow (2-way) ---\n%!";
  let g1 =
    PDSL.ndarray [| 1.0; 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let g2 =
    PDSL.ndarray [| 3.0; 4.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let%op grad_result = sin [g1; g2] in
  let%op loss = grad_result ++ "...|... => 0" in
  Train.set_hosted loss.value;
  Train.set_hosted grad_result.value;
  Train.set_hosted (Option.value_exn ~here:[%here] g1.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] g2.diff).grad;
  ignore (Train.update_once ~output_cd_file:false ctx loss);

  printf "grad_result (sin of stacked):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false grad_result;
  printf "\nGradient of g1 (should be cos of original):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true g1;
  printf "\nGradient of g2:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true g2;

  (* --- Test 8: Gradient flow (3-way) --- *)
  printf "\n--- Test 8: Gradient flow (3-way) ---\n%!";
  let h1 =
    PDSL.ndarray [| 0.5; 1.5 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let h2 =
    PDSL.ndarray [| 1.0; 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let h3 =
    PDSL.ndarray [| 3.0; 0.1 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let%op grad3_result = sin [h1; h2; h3] in
  let%op loss3 = grad3_result ++ "...|... => 0" in
  Train.set_hosted loss3.value;
  Train.set_hosted grad3_result.value;
  Train.set_hosted (Option.value_exn ~here:[%here] h1.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] h2.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] h3.diff).grad;
  ignore (Train.update_once ~output_cd_file:false ctx loss3);

  printf "grad3_result (sin of 3-way stacked):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false grad3_result;
  printf "\nGradient of h1:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true h1;
  printf "\nGradient of h2:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true h2;
  printf "\nGradient of h3:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true h3;

  printf "\n=== Block Tensor Literal Tests Complete ===\n%!";
  ignore ctx

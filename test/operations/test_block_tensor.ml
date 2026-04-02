open Base
open Ocannl
open Nn_blocks.DSL_modules
open Stdio

let () =
  printf "=== Block Tensor Literal Syntax Tests ===\n\n%!";
  Tensor.unsafe_reinitialize ()

(* --- Test 1: List syntax [x1; x2] for output axis stacking --- *)
let () =
  printf "--- Test 1: List [x1; x2] output axis ---\n%!";
  let x1 =
    PDSL.ndarray [| 1.0; 2.0; 3.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let x2 =
    PDSL.ndarray [| 4.0; 5.0; 6.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let%op result = [x1; x2] in
  let ctx = Context.auto () in
  Train.set_hosted result.value;
  ignore (Train.forward_once ~output_cd_file:false ctx result);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result;
  printf "\n%!"

(* --- Test 2: Array syntax [|x1; x2|] for batch axis concatenation --- *)
let () =
  printf "--- Test 2: Array [|x1; x2|] batch axis ---\n%!";
  Tensor.unsafe_reinitialize ();
  let x1 =
    PDSL.ndarray [| 1.0; 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let x2 =
    PDSL.ndarray [| 3.0; 4.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let%op result = [|x1; x2|] in
  let ctx = Context.auto () in
  Train.set_hosted result.value;
  ignore (Train.forward_once ~output_cd_file:false ctx result);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result;
  printf "\n%!"

(* --- Test 3: Stacking 2D tensors with [x1; x2] --- *)
let () =
  printf "--- Test 3: Stacking 2D [x1; x2] ---\n%!";
  Tensor.unsafe_reinitialize ();
  let x1 =
    PDSL.ndarray [| 1.0; 2.0; 3.0; 4.0 |] ~batch_dims:[] ~input_dims:[ 2 ] ~output_dims:[ 2 ] ()
  in
  let x2 =
    PDSL.ndarray [| 5.0; 6.0; 7.0; 8.0 |] ~batch_dims:[] ~input_dims:[ 2 ] ~output_dims:[ 2 ] ()
  in
  let%op result = [x1; x2] in
  let ctx = Context.auto () in
  Train.set_hosted result.value;
  ignore (Train.forward_once ~output_cd_file:false ctx result);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result;
  printf "\n%!"

(* --- Test 4: Gradient through list block tensor --- *)
let () =
  printf "--- Test 4: Gradient through [x1; x2] ---\n%!";
  Tensor.unsafe_reinitialize ();
  let x1 =
    PDSL.ndarray [| 1.0; 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let x2 =
    PDSL.ndarray [| 3.0; 4.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  (* sin of block tensor, then sum to scalar for backprop *)
  let%op block = sin [x1; x2] in
  let%op loss = block ++ "...|... => 0" in
  let ctx = Context.auto () in
  Train.set_hosted loss.value;
  Train.set_hosted block.value;
  Train.set_hosted (Option.value_exn ~here:[%here] x1.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] x2.diff).grad;
  ignore (Train.update_once ~output_cd_file:false ctx loss);
  printf "Block (sin of concat):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false block;
  printf "\nGradients (should be cos of original values):\n%!";
  printf "x1 grad:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true x1;
  printf "x2 grad:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true x2;
  printf "\n%!"

(* --- Test 5: ndarray constant regression --- *)
let () =
  printf "--- Test 5: ndarray constant regression ---\n%!";
  Tensor.unsafe_reinitialize ();
  let%op c = [1.0; 2.0; 3.0] in
  let ctx = Context.auto () in
  Train.set_hosted c.value;
  ignore (Train.forward_once ~output_cd_file:false ctx c);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false c;
  printf "\n%!"

(* --- Test 6: Single element [x1] (unsqueeze) --- *)
let () =
  printf "--- Test 6: Single [x1] (unsqueeze) ---\n%!";
  Tensor.unsafe_reinitialize ();
  let x1 =
    PDSL.ndarray [| 2.0; 3.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let%op result = [x1] in
  let ctx = Context.auto () in
  Train.set_hosted result.value;
  ignore (Train.forward_once ~output_cd_file:false ctx result);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result;
  printf "\n%!"

(* --- Test 7: 3-way list stacking [x1; x2; x3] --- *)
let () =
  printf "--- Test 7: 3-way [x1; x2; x3] ---\n%!";
  Tensor.unsafe_reinitialize ();
  let x1 =
    PDSL.ndarray [| 1.0; 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let x2 =
    PDSL.ndarray [| 3.0; 4.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let x3 =
    PDSL.ndarray [| 5.0; 6.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let%op result = [x1; x2; x3] in
  let ctx = Context.auto () in
  Train.set_hosted result.value;
  ignore (Train.forward_once ~output_cd_file:false ctx result);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result;
  printf "\n%!"

(* --- Test 8: Tuple syntax (x1, x2) for input axis stacking --- *)
let () =
  printf "--- Test 8: Tuple (x1, x2) input axis ---\n%!";
  Tensor.unsafe_reinitialize ();
  let x1 =
    PDSL.ndarray [| 1.0; 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let x2 =
    PDSL.ndarray [| 3.0; 4.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let%op result = (x1, x2) in
  let ctx = Context.auto () in
  Train.set_hosted result.value;
  ignore (Train.forward_once ~output_cd_file:false ctx result);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result;
  printf "\n%!"

(* --- Test 9: Tuple ndarray constant regression --- *)
let () =
  printf "--- Test 9: Tuple ndarray constant regression ---\n%!";
  Tensor.unsafe_reinitialize ();
  let%op c = (1.0, 2.0, 3.0) in
  let ctx = Context.auto () in
  Train.set_hosted c.value;
  ignore (Train.forward_once ~output_cd_file:false ctx c);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false c;
  printf "\n%!"

(* --- Test 10: Ndarray with computed leaf regression --- *)
let () =
  printf "--- Test 10: Ndarray with computed leaf ---\n%!";
  Tensor.unsafe_reinitialize ();
  let v = 42.0 in
  let%op c = [1.0; v] in
  let ctx = Context.auto () in
  Train.set_hosted c.value;
  ignore (Train.forward_once ~output_cd_file:false ctx c);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false c;
  printf "\n%!"

(* --- Test 11: Negative ndarray literal regression --- *)
let () =
  printf "--- Test 11: Negative ndarray literal ---\n%!";
  Tensor.unsafe_reinitialize ();
  let%op c = [-1.0; 2.0; -3.0] in
  let ctx = Context.auto () in
  Train.set_hosted c.value;
  ignore (Train.forward_once ~output_cd_file:false ctx c);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false c;
  printf "\n%!"

(* --- Test 12: Negative tuple ndarray constant --- *)
let () =
  printf "--- Test 12: Negative tuple ndarray ---\n%!";
  Tensor.unsafe_reinitialize ();
  let%op c = (-1, 2) in
  let ctx = Context.auto () in
  Train.set_hosted c.value;
  ignore (Train.forward_once ~output_cd_file:false ctx c);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false c;
  printf "\n%!"

(* --- Test 13: Nested [[x1; x2]; [x3; x4]] block matrix --- *)
let () =
  printf "--- Test 13: Nested [[x1; x2]; [x3; x4]] ---\n%!";
  Tensor.unsafe_reinitialize ();
  let x1 = PDSL.ndarray [| 1.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] () in
  let x2 = PDSL.ndarray [| 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] () in
  let x3 = PDSL.ndarray [| 3.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] () in
  let x4 = PDSL.ndarray [| 4.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] () in
  let%op result = [[x1; x2]; [x3; x4]] in
  let ctx = Context.auto () in
  Train.set_hosted result.value;
  ignore (Train.forward_once ~output_cd_file:false ctx result);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result;
  printf "\n%!"

(* --- Test 14: Nested with vectors [[x1; x2]; [x3; x4]] --- *)
let () =
  printf "--- Test 14: Nested vectors [[x1; x2]; [x3; x4]] ---\n%!";
  Tensor.unsafe_reinitialize ();
  let x1 =
    PDSL.ndarray [| 1.0; 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let x2 =
    PDSL.ndarray [| 3.0; 4.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let x3 =
    PDSL.ndarray [| 5.0; 6.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let x4 =
    PDSL.ndarray [| 7.0; 8.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let%op result = [[x1; x2]; [x3; x4]] in
  let ctx = Context.auto () in
  Train.set_hosted result.value;
  ignore (Train.forward_once ~output_cd_file:false ctx result);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result;
  printf "\n%!"

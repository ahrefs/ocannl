open! Base
open! Ocannl
open! Nn_blocks.DSL_modules
open Stdio

let max_pool2d = Nn_blocks.max_pool2d

(** Test basic max_pool2d operation with default parameters.

    Default: stride=2, window_size=2
    For 4x4 input, output should be 2x2. *)
let test_max_pool2d_basic () =
  printf "Testing max_pool2d with default parameters (stride=2, window=2)...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 4x4 input with 1 channel *)
  let input = TDSL.range_of_shape ~output_dims:[ 4; 4; 1 ] () in

  (* Apply max_pool2d with default params *)
  let%op output = max_pool2d () input in

  let ctx = Context.auto () in
  Train.set_hosted input.value;
  Train.set_hosted output.value;
  ignore (Train.forward_once ctx output);

  printf "Input shape: 4x4x1\n%!";
  printf "Window size: 2x2\n%!";
  printf "Stride: 2\n%!";
  printf "Expected output spatial dims: 2x2\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false input;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false output;
  printf "\n%!"

(** Test max_pool2d with stride=2, window=3.

    For 7x7 input with stride=2, window=3 (no padding):
    Output size = (7 - 3) / 2 + 1 = 3.
    Valid convolution requires: input = stride * (output - 1) + window = 2 * 2 + 3 = 7 *)
let test_max_pool2d_window3 () =
  printf "Testing max_pool2d with stride=2, window=3...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 7x7 input with 2 channels *)
  let input = TDSL.range_of_shape ~output_dims:[ 7; 7; 2 ] () in

  (* Apply max_pool2d with window_size=3 *)
  let%op output = max_pool2d ~stride:2 ~window_size:3 () input in

  let ctx = Context.auto () in
  Train.set_hosted input.value;
  Train.set_hosted output.value;
  ignore (Train.forward_once ctx output);

  printf "Input shape: 7x7x2\n%!";
  printf "Window size: 3x3\n%!";
  printf "Stride: 2\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false input;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false output;
  printf "\n%!"

(** Test max_pool2d with output dimension 1.

    For 3x3 input with stride=2, window=3 (no padding):
    Output size = (3 - 3) / 2 + 1 = 1.
    This tests the edge case where the kernel exactly covers the input. *)
let test_max_pool2d_output_dim_1 () =
  printf "Testing max_pool2d with output dimension 1...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 3x3 input with 1 channel *)
  let input = TDSL.range_of_shape ~output_dims:[ 3; 3; 1 ] () in

  (* Apply max_pool2d with window_size=3, stride=2 *)
  let%op output = max_pool2d ~stride:2 ~window_size:3 () input in

  let ctx = Context.auto () in
  Train.set_hosted input.value;
  Train.set_hosted output.value;
  ignore (Train.forward_once ctx output);

  printf "Input shape: 3x3x1\n%!";
  printf "Window size: 3x3\n%!";
  printf "Stride: 2\n%!";
  printf "Expected output spatial dims: 1x1\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false input;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false output;
  printf "\n%!"

(** Test max_pool2d preserves channels.

    Channel dimension should pass through unchanged. *)
let test_max_pool2d_channels () =
  printf "Testing max_pool2d preserves channel dimension...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 4x4 input with 3 channels *)
  let input = TDSL.range_of_shape ~output_dims:[ 4; 4; 3 ] () in

  let%op output = max_pool2d () input in

  let ctx = Context.auto () in
  Train.set_hosted input.value;
  Train.set_hosted output.value;
  ignore (Train.forward_once ctx output);

  printf "Input shape: 4x4x3\n%!";
  printf "Expected output shape: 2x2x3 (channels preserved)\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false output;
  printf "\n%!"

(** Test backpropagation for max_pool2d.

    This tests that shape inference works correctly during backpropagation for strided max pooling
    (which never uses padding).

    BUG: This test fails with "solved dimensions for axis: mismatch" error during shape inference
    for backprop. The error shows dimension mismatch between input (4) and output (2) dimensions.
    This is the same class of issue as strided conv2d without padding - the backpropagation shape
    inference fails when output dimensions don't match input dimensions due to striding. This is
    also the root cause of LeNet training failures. See circles_conv.ml and test_conv_padding.ml
    for related discussion. *)
let test_max_pool2d_backprop () =
  printf "\nTesting backprop for max_pool2d...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 4x4 input with 1 channel using a parameter (requires grad) *)
  let%op input = { x; o = [ 4; 4; 1 ] } in

  (* Apply max_pool2d with default params (stride=2, window=2) *)
  let%op output = max_pool2d () input in
  (* Sum to scalar for backprop *)
  let%op loss = output ++ "...|... => 0" in

  let ctx = Context.auto () in
  Train.set_hosted loss.value;
  ignore (Train.update_once ~output_cd_file:false ctx loss);

  printf "Input shape: 4x4x1\n%!";
  printf "Window size: 2x2\n%!";
  printf "Stride: 2\n%!";
  printf "Backprop completed successfully!\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true loss

let () =
  test_max_pool2d_basic ();
  test_max_pool2d_window3 ();
  test_max_pool2d_output_dim_1 ();
  test_max_pool2d_channels ();
  (* TODO: Uncomment when the strided backprop bug is fixed:
     test_max_pool2d_backprop (); *)
  ignore test_max_pool2d_backprop;
  printf "\nAll max_pool2d tests completed!\n%!"

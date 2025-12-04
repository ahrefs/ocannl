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

    For 6x6 input with stride=2, window=3:
    Output size should be ceil((6-3)/2) + 1 = 2 or 3 depending on padding. *)
let test_max_pool2d_window3 () =
  printf "Testing max_pool2d with stride=2, window=3...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 6x6 input with 2 channels *)
  let input = TDSL.range_of_shape ~output_dims:[ 6; 6; 2 ] () in

  (* Apply max_pool2d with window_size=3 *)
  let%op output = max_pool2d ~stride:2 ~window_size:3 () input in

  let ctx = Context.auto () in
  Train.set_hosted input.value;
  Train.set_hosted output.value;
  ignore (Train.forward_once ctx output);

  printf "Input shape: 6x6x2\n%!";
  printf "Window size: 3x3\n%!";
  printf "Stride: 2\n%!";
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

let () =
  test_max_pool2d_basic ();
  test_max_pool2d_window3 ();
  test_max_pool2d_channels ();
  printf "All max_pool2d tests completed!\n%!"

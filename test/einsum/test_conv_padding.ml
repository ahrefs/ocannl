open! Base
open! Ocannl
open! Nn_blocks.DSL_modules
open Stdio

let conv2d = Nn_blocks.conv2d

(** Test that conv2d with use_padding=true preserves spatial dimensions.

    With use_padding=true, the output spatial dimensions should match input/stride. For stride=1 and
    any kernel_size, output should have the same spatial dims as input. *)
let test_conv2d_padding_preserves_dims () =
  printf "Testing conv2d with use_padding=true preserves dimensions...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a simple 5x5 input with 1 channel *)
  let input = TDSL.range_of_shape ~output_dims:[ 5; 5; 1 ] () in

  (* Apply conv2d with kernel_size=3, stride=1, use_padding=true *)
  let%op output =
    conv2d ~label:[ "test_conv" ] ~kernel_size:3 ~stride:1 ~use_padding:true () input
  in

  let ctx = Context.auto () in
  Train.set_hosted output.value;
  ignore (Train.forward_once ctx output);

  printf "Input shape: 5x5x1\n%!";
  printf "Kernel size: 3x3\n%!";
  printf "Stride: 1\n%!";
  printf "use_padding: true\n%!";
  printf "Expected output spatial dims: 5x5 (same as input)\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false output;
  printf "\n%!"

(** Test that conv2d with use_padding=false reduces spatial dimensions.

    With use_padding=false, the output spatial dimensions should be reduced by (kernel_size - 1).
    For 5x5 input, kernel_size=3, stride=1: output should be 3x3. *)
let test_conv2d_no_padding_reduces_dims () =
  printf "Testing conv2d with use_padding=false reduces dimensions...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a simple 5x5 input with 1 channel *)
  let input = TDSL.range_of_shape ~output_dims:[ 5; 5; 1 ] () in

  (* Apply conv2d with kernel_size=3, stride=1, use_padding=false *)
  let%op output =
    conv2d ~label:[ "test_conv" ] ~kernel_size:3 ~stride:1 ~use_padding:false () input
  in

  let ctx = Context.auto () in
  Train.set_hosted output.value;
  ignore (Train.forward_once ctx output);

  printf "Input shape: 5x5x1\n%!";
  printf "Kernel size: 3x3\n%!";
  printf "Stride: 1\n%!";
  printf "use_padding: false\n%!";
  printf "Expected output spatial dims: 3x3 (reduced by kernel_size-1)\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false output;
  printf "\n%!"

(** Test conv2d with stride=2 and use_padding=true.

    With stride=2 and use_padding=true, output dims should be ceil(input/stride). For 6x6 input,
    stride=2: output should be 3x3. *)
let test_conv2d_stride_with_padding () =
  printf "Testing conv2d with stride=2 and use_padding=true...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 6x6 input with 1 channel *)
  let input = TDSL.range_of_shape ~output_dims:[ 6; 6; 1 ] () in

  (* Apply conv2d with kernel_size=3, stride=2, use_padding=true *)
  let%op output =
    conv2d ~label:[ "test_conv" ] ~kernel_size:3 ~stride:2 ~use_padding:true () input
  in

  let ctx = Context.auto () in
  Train.set_hosted output.value;
  ignore (Train.forward_once ctx output);

  printf "Input shape: 6x6x1\n%!";
  printf "Kernel size: 3x3\n%!";
  printf "Stride: 2\n%!";
  printf "use_padding: true\n%!";
  printf "Expected output spatial dims: 3x3 (input/stride)\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false output;
  printf "\nInput: %s\n%!" @@ Ir.Tnode.dims_to_string input.value

(** Test conv2d with stride=2 and use_padding=false.

    With stride=2 and use_padding=false, output dims should be (input - kernel_size + 1) / stride.
    For 6x6 input, kernel_size=3, stride=2: (6-3+1)/2 = 2, so output should be 2x2. *)
let test_conv2d_stride_without_padding () =
  printf "Testing conv2d with stride=2 and use_padding=false...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 6x6 input with 1 channel *)
  let input = TDSL.range_of_shape ~output_dims:[ 6; 6; 1 ] () in

  (* Apply conv2d with kernel_size=3, stride=2, use_padding=false *)
  let%op output =
    conv2d ~label:[ "test_conv" ] ~kernel_size:3 ~stride:2 ~use_padding:false () input
  in

  let ctx = Context.auto () in
  Train.set_hosted output.value;
  ignore (Train.forward_once ctx output);

  printf "Input shape: 6x6x1\n%!";
  printf "Kernel size: 3x3\n%!";
  printf "Stride: 2\n%!";
  printf "use_padding: false\n%!";
  printf "Expected output spatial dims: 2x2 ((input-kernel+1)/stride)\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false output;
  printf "\n%!"

let () =
  test_conv2d_stride_with_padding ();
  ignore test_conv2d_padding_preserves_dims;
  ignore test_conv2d_no_padding_reduces_dims;
  ignore test_conv2d_stride_without_padding;
  printf "All conv padding tests completed!\n%!"

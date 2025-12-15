open! Base
open! Ocannl
open! Ocannl.Operation.DSL_modules
open Stdio

(** Test that padding margins are properly initialized and reset between operations.

    This test demonstrates the padding behavior with use_padding=true (= marker).
    We apply TWO DIFFERENT operations to the SAME input tensor, each requiring
    different padding margins. The input's padding must be properly reset between
    the two operations.

    - Max-pool-like operation: padding should be -infinity for correct max behavior
    - Conv-like operation: padding should be 0 for correct sum behavior

    By using negative input values, we can detect if padding is incorrectly 0 for max-pool. *)
let test_padding_reset () =
  printf "Testing padding margin initialization and reset...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 4x4 input with negative values: -16..-1
     This way, if padding margins are 0 (incorrect for max-pool with negative values),
     the max will incorrectly be 0 instead of the actual maximum negative value.
     Proper padding for max should be -infinity or at least very negative. *)
  let%op input = TDSL.range_of_shape ~output_dims:[ 4; 4 ] () - 16. in

  (* Max-pool-like operation on input with stride=1, window=3, use_padding=true.
     For max-pool, padding value should be -infinity so max ignores padding positions. *)
  let%op pooled =
    input @^+ "oh=+wh, ow=+ww; wh, ww => oh, ow" [ "wh"; "ww" ] (0.0 + 0.0)
  in
  Shape.set_dim wh 3;
  Shape.set_dim ww 3;

  (* Conv-like operation ALSO on input (not pooled!) with stride=1, kernel=3, use_padding=true.
     Kernel is all 1.0, so this sums 3x3 windows of input.
     For conv/sum, padding value should be 0 so sum ignores padding positions.

     KEY: Both operations use the SAME input tensor, but require DIFFERENT padding values.
     The input's padding margins must be reset between the two operations. *)
  let%op conv_out =
    input +* "oh=+kh, ow=+kw; kh, kw => oh, ow" [ "kh"; "kw" ] (1.0 + 0.0)
  in
  Shape.set_dim kh 3;
  Shape.set_dim kw 3;

  let ctx = Context.auto () in
  Train.set_hosted input.value;
  Train.set_hosted pooled.value;
  Train.set_hosted conv_out.value;

  (* Compile BOTH forward passes into a single routine using sequence.
     This tests that input's padding is properly reset between the two operations. *)
  let ctx = Train.init_params ctx Train.IDX.empty conv_out in
  let ctx = Train.init_params ctx Train.IDX.empty pooled in
  (* Get forward codes - order matters since consume_forward_code modifies state *)
  let fwd_pooled = Train.forward pooled in
  let fwd_conv = Train.forward conv_out in
  let combined = Ir.Assignments.sequence [ fwd_pooled; fwd_conv ] in
  let routine = Train.to_routine ctx Train.IDX.empty combined in

  printf "\n=== Forward pass (both operations in sequence) ===\n%!";
  Train.run ctx routine;

  printf "\nInput (4x4 logical, with padding margins shown):\n%!";
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Inline input;

  printf "\nPooled - max of 3x3 windows of INPUT (padding should be -inf for max):\n%!";
  Tensor.print ~here:[%here] ~force:true ~with_code:false ~with_grad:false `Inline pooled;

  printf "\nConv - sum of 3x3 windows of INPUT (padding should be 0 for sum):\n%!";
  Tensor.print ~here:[%here] ~force:true ~with_code:false ~with_grad:false `Inline conv_out;

  (* Run a second pass to check padding reset behavior across iterations *)
  printf "\n=== Second forward pass (checking reset across iterations) ===\n%!";
  Train.run ctx routine;

  printf "\nPooled after second pass:\n%!";
  Tensor.print ~here:[%here] ~force:true ~with_code:false ~with_grad:false `Inline pooled;

  printf "\nConv after second pass:\n%!";
  Tensor.print ~here:[%here] ~force:true ~with_code:false ~with_grad:false `Inline conv_out;

  (* Analysis of expected values:
     Input is 4x4 with values -16 to -1:
       -16  -15  -14  -13
       -12  -11  -10  -9
       -8   -7   -6   -5
       -4   -3   -2   -1

     For MAX-POOL at (0,0) with 3x3 window:
       - Window covers input[0..2, 0..2] = top-left 3x3
       - Values: -16,-15,-14,-12,-11,-10,-8,-7,-6
       - Correct max = -6
       - But with padding=0 at corners, window includes padding positions
       - If pad=0: max = 0 (BUG!)

     For CONV at (0,0) with 3x3 window:
       - Same window, but summing with kernel of 1s
       - With padding=0, sum = -16-15-14-12-11-10-8-7-6 = -99
       - Corner positions include fewer real values due to padding
       - sum at (0,0) = 0+0+0+0+(-16)+(-15)+0+(-12)+(-11) = -54

     The key test: if input's padding is not reset between max-pool and conv,
     the results will be wrong for one or both operations.
  *)

  printf "\n=== Expected Behavior Analysis ===\n%!";
  printf "Input values: -16 to -1 (all negative)\n%!";
  printf "\nFor MAX-POOL (padding should be -infinity):\n%!";
  printf "  With pad=0 (BUG): corners show 0 because max(0, negative) = 0\n%!";
  printf "  With pad=-inf (correct): pooled[0,0]=-6, pooled[1,1]=-6, etc.\n%!";
  printf "\nFor CONV/SUM (padding should be 0):\n%!";
  printf "  conv[0,0] = sum of 4 values (corner) = -16-15-12-11 = -54\n%!";
  printf "  conv[1,1] = sum of 9 values (center) = -16-15-14-12-11-10-8-7-6 = -99\n%!";
  printf "\nIf we see 0s in pooled corners, max-pool padding is wrong.\n%!";
  printf "If conv values are wrong, the padding reset between operations failed.\n%!"

let () =
  test_padding_reset ();
  printf "\nPadding reset test completed!\n%!"

open! Base
open! Ocannl
open! Nn_blocks.DSL_modules
open Stdio

(** Test tropical semiring (max-plus) operations with a learnable kernel.

    This tests backpropagation for tropical operations with both input (t1/rhs1)
    and kernel (t2/rhs2) gradients.

    LIMITATION: The current implementation using LHS projection for kernel gradients
    computes correct INPUT gradients (g1) but INCORRECT KERNEL gradients (g2).

    The issue: with `_lhs` suffix, `sum_lhs[oh,ow]` computes max over (wh,ww) for each
    output position, which equals `max_pool2d[oh,ow]`. So `cond_lhs` is always true,
    and every kernel position gets gradient from every output position.

    For correct kernel gradients, we would need tensors of shape (oh,ow,wh,ww) to track
    which specific kernel position achieved the argmax for each output. This would require
    either a different approach or explicit support for such "outer product" projections.

    For max_pool2d (zero kernel), this is fine since kernel gradients don't matter.
    For learnable tropical kernels, only the input gradients are reliable. *)

(** Create a tropical convolution-like operation with a learnable kernel.

    This is similar to max_pool2d but with a non-zero learnable kernel, allowing us to
    verify that g2 (kernel) gradients are computed correctly.

    The tropical operation computes: output[oh,ow] = max over (wh,ww) of (input[2*oh+wh, 2*ow+ww] + kernel[wh,ww])

    For backprop:
    - g1 (input grad): gradient flows to input positions that achieved the argmax
    - g2 (kernel grad): gradient flows to kernel positions that achieved the argmax *)
let tropical_conv2d ?(stride = 2) ?(window_size = 2) () =
  let%op op x kernel =
    Shape.set_dim wh window_size;
    Shape.set_dim ww window_size;
    x @^+ "... | stride*oh< + wh, stride*ow< + ww, ..c..; wh, ww => ... | oh, ow, ..c.." [ "wh"; "ww" ]
          kernel
  in
  op

(** Test tropical conv with learnable kernel - forward pass only.

    Verifies that the tropical operation correctly computes max(input + kernel) over windows. *)
let test_tropical_kernel_forward () =
  printf "Testing tropical conv with learnable kernel (forward)...\n%!";
  Tensor.unsafe_reinitialize ();

  (* 4x4 input with 1 channel *)
  let input = TDSL.range_of_shape ~output_dims:[ 4; 4; 1 ] () in

  (* 2x2 kernel with specific values - shape matches wh, ww dimensions *)
  let%op kernel = { k = [ [ 0; 1 ]; [ 2; 3 ] ] } in

  let%op output = tropical_conv2d () input kernel in

  let ctx = Context.auto () in
  Train.set_hosted input.value;
  Train.set_hosted kernel.value;
  Train.set_hosted output.value;
  ignore (Train.forward_once ctx output);

  printf "Input shape: 4x4x1 (values 0-15)\n%!";
  printf "Kernel shape: 2x2 (values [[0,1],[2,3]])\n%!";
  printf "Stride: 2, Window: 2x2\n%!";
  printf "Output formula: max over (wh,ww) of (input[2*oh+wh, 2*ow+ww] + kernel[wh,ww])\n%!";
  printf "\n%!";
  printf "Expected output calculations:\n%!";
  printf "  [0,0]: max(0+0, 1+1, 4+2, 5+3) = max(0, 2, 6, 8) = 8\n%!";
  printf "  [0,1]: max(2+0, 3+1, 6+2, 7+3) = max(2, 4, 8, 10) = 10\n%!";
  printf "  [1,0]: max(8+0, 9+1, 12+2, 13+3) = max(8, 10, 14, 16) = 16\n%!";
  printf "  [1,1]: max(10+0, 11+1, 14+2, 15+3) = max(10, 12, 16, 18) = 18\n%!";
  printf "\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false input;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false kernel;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false output;
  printf "\n%!"

(** Test tropical conv with learnable kernel - backprop.

    This is the key test: verifies that gradients flow correctly to both input and kernel.

    Input pattern (4x4, values designed so argmax varies):
    ```
      [[9, 0, 0, 0],
       [0, 0, 0, 8],
       [0, 7, 0, 0],
       [0, 0, 6, 0]]
    ```

    Kernel (2x2, small values so input determines argmax):
    ```
      [[0, 0],
       [0, 0]]
    ```

    With zero kernel, this is like max_pool2d - argmax is at input max positions.
    - Window [0,0]: max at (0,0)=9, argmax kernel position (0,0)
    - Window [0,1]: max at (1,3)=8, argmax kernel position (1,1)
    - Window [1,0]: max at (2,1)=7, argmax kernel position (0,1)
    - Window [1,1]: max at (3,2)=6, argmax kernel position (1,0)

    Expected input gradients: 1 at positions (0,0), (1,3), (2,1), (3,2); 0 elsewhere.

    Kernel gradients (LIMITATION): Due to the current implementation, all kernel positions
    get gradient 4.0 (each position is visited once per output, and cond_lhs is always true).
    Ideally, each position should get 1.0 since each is argmax for exactly one output. *)
let test_tropical_kernel_backprop_zero_kernel () =
  printf "Testing tropical conv backprop with zero kernel...\n%!";
  Tensor.unsafe_reinitialize ();

  (* 4x4 input designed so each 2x2 window has its max at a different position *)
  let%op input =
    {
      x =
        [
          [ [ 9 ]; [ 0 ]; [ 0 ]; [ 0 ] ];
          [ [ 0 ]; [ 0 ]; [ 0 ]; [ 8 ] ];
          [ [ 0 ]; [ 7 ]; [ 0 ]; [ 0 ] ];
          [ [ 0 ]; [ 0 ]; [ 6 ]; [ 0 ] ];
        ];
    }
  in

  (* Zero kernel - argmax determined purely by input *)
  let%op kernel = { k = [ [ 0; 0 ]; [ 0; 0 ] ] } in

  let%op output = tropical_conv2d () input kernel in
  let%op loss = output ++ "...|... => 0" in

  let ctx = Context.auto () in
  Train.set_hosted loss.value;
  Train.set_hosted (Option.value_exn ~here:[%here] input.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] kernel.diff).grad;
  ignore (Train.update_once ~output_cd_file:false ctx loss);

  printf "Input shape: 4x4x1\n%!";
  printf "Kernel shape: 2x2 (all zeros)\n%!";
  printf "Stride: 2, Window: 2x2\n%!";
  printf "Expected output: [[9, 8], [7, 6]]\n%!";
  printf "Expected loss: 9 + 8 + 7 + 6 = 30\n%!";
  printf "\n%!";
  printf "Expected input gradients: 1 at argmax positions, 0 elsewhere\n%!";
  printf "Kernel gradients (LIMITATION): all 4s due to cond_lhs always being true\n%!";
  printf "\n%!";
  Train.printf ~here:[%here] ~with_code:false loss;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true input;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true kernel;
  printf "\n%!"

(** Test tropical conv backprop with non-zero kernel that affects argmax.

    Input (4x4, uniform low values):
    ```
      [[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]]
    ```

    Kernel (2x2, large value at position (1,1)):
    ```
      [[0, 0],
       [0, 10]]
    ```

    With this kernel, the argmax for every window is at kernel position (1,1)
    because 1+10=11 > 1+0=1 for all other positions.

    Expected output: all 11 (value 1 + kernel 10 at position (1,1) of each window)
    Expected input gradients: 1 at positions (1,1), (1,3), (3,1), (3,3); 0 elsewhere
      (these are the input positions corresponding to kernel (1,1) in each window)

    Kernel gradients (LIMITATION): Due to the current implementation, all kernel positions
    get gradient 4.0 instead of the expected [0,0; 0,4]. The implementation doesn't track
    which kernel position achieved argmax for each output position. *)
let test_tropical_kernel_backprop_nonzero_kernel () =
  printf "Testing tropical conv backprop with non-zero kernel...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Uniform input - kernel determines argmax *)
  let%op input =
    {
      x =
        [
          [ [ 1 ]; [ 1 ]; [ 1 ]; [ 1 ] ];
          [ [ 1 ]; [ 1 ]; [ 1 ]; [ 1 ] ];
          [ [ 1 ]; [ 1 ]; [ 1 ]; [ 1 ] ];
          [ [ 1 ]; [ 1 ]; [ 1 ]; [ 1 ] ];
        ];
    }
  in

  (* Kernel with large value at (1,1) - this position wins for all windows *)
  let%op kernel = { k = [ [ 0; 0 ]; [ 0; 10 ] ] } in

  let%op output = tropical_conv2d () input kernel in
  let%op loss = output ++ "...|... => 0" in

  let ctx = Context.auto () in
  Train.set_hosted loss.value;
  Train.set_hosted (Option.value_exn ~here:[%here] input.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] kernel.diff).grad;
  ignore (Train.update_once ~output_cd_file:false ctx loss);

  printf "Input shape: 4x4x1 (all 1s)\n%!";
  printf "Kernel shape: 2x2 ([[0,0],[0,10]])\n%!";
  printf "Stride: 2, Window: 2x2\n%!";
  printf "Expected output: all 11 (1 + 10 at kernel position (1,1))\n%!";
  printf "Expected loss: 11 * 4 = 44\n%!";
  printf "\n%!";
  printf "Expected input gradients:\n%!";
  printf "  1 at positions (1,1), (1,3), (3,1), (3,3) - these correspond to kernel[1,1]\n%!";
  printf "  0 elsewhere\n%!";
  printf "Kernel gradients (LIMITATION): all 4s (should be [[0,0],[0,4]])\n%!";
  printf "\n%!";
  Train.printf ~here:[%here] ~with_code:false loss;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true input;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true kernel;
  printf "\n%!"

let () =
  test_tropical_kernel_forward ();
  test_tropical_kernel_backprop_zero_kernel ();
  test_tropical_kernel_backprop_nonzero_kernel ();
  printf "\nAll tropical kernel tests completed!\n%!"

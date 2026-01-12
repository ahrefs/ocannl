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

  printf "\n=== Concatenation Graph Test Complete ===\n%!";

  (* Three-way concatenation test - triggers ambiguous indices error during backprop.

     Root cause analysis (assignments.ml):
     ------------------------------------
     The error "Ambiguous indices in concatenation: multiple blocks viable for same
     position" occurs during Rev_sides lowering when multiple target tensors (gradient
     tensors for each concat input) are determined to be valid write targets.

     For 2-way concat (a, b) ++^ "a; b => a^b":
     - Product space has one entry with 2 dims [d1, d2] and 2 iterators [s1, s2]
     - Sequential loops iterate: first s1 (RHS[0] active), then s2 (RHS[1] active)
     - Iterator indices correctly raise Empty_block when not in current block_iters

     For 3-way concat (a, b, c) ++^ "a; b; c => a^b^c":
     - Should have 3 dims [d1, d2, d3] and 3 iterators [s1, s2, s3]
     - Sequential loops should iterate s1, then s2, then s3
     - However, during Rev_sides the filtering at lines 532-537 doesn't correctly
       exclude inactive RHSes, leading to targets array with length > 1

     The check at lines 540-543 catches this:
       if Array.length targets > 1 then
         raise @@ Utils.User_error "Ambiguous indices in concatenation..."

     Possible fix locations:
     1. shape.ml: How projections are generated for 3+ way concat
     2. assignments.ml: How Rev_sides filtering handles Concat indices with 3+ symbols
  *)
  printf "\n=== Three-way Concatenation Test (Known Limitation) ===\n\n%!";
  Tensor.unsafe_reinitialize ();

  let y1 =
    PDSL.ndarray [| 1.0; 2.0; 3.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let y2 = PDSL.ndarray [| 4.0; 5.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let y3 = PDSL.ndarray [| 6.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] () in

  let%op result3 = sin ((y1, y2, y3) ++^ "a; b; c => a^b^c") in
  let%op loss3 = result3 ++ "...|... => 0" in

  let ctx3 = Context.auto () in
  Train.set_hosted loss3.value;
  Train.set_hosted result3.value;
  Train.set_hosted (Option.value_exn ~here:[%here] y1.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] y2.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] y3.diff).grad;

  (try ignore (Train.update_once ~output_cd_file:false ctx3 loss3)
   with exn -> printf "ERROR: %s\n%!" (Exn.to_string exn));

  printf "\n=== Three-way Concatenation Test Complete ===\n%!"

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

  printf "\n=== Concatenation Graph Test Complete ===\n%!"

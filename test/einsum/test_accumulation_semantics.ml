open Base
open Ocannl
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL

module type Backend = Ir.Backend_intf.Backend

(** Test that einsum reductions properly accumulate values *)
let test_einsum_reduction () =
  Stdio.printf "\n=== Testing einsum reduction (surjective but not injective) ===\n";
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  (* Create input tensor with shape [batch=2, input=3, output=4] *)
  let input = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in

  (* The einsum "b|i->o => b|i" reduces over the output dimension This projection is: - Surjective:
     all positions in result tensor get written - NOT injective: multiple source positions map to
     same target (needs accumulation) *)
  let%op result = input ++ "b|i->o => b|i" in

  Train.set_hosted input.value;
  Train.set_hosted result.value;
  Train.every_non_literal_on_host result;

  ignore (Train.forward_once (module Backend) result);

  Stdio.printf "Input tensor (shape: batch=2, input=3, output=4):\n";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false input;

  Stdio.printf "\nResult after reduction 'b|i->o => b|i' (should sum over output dimension):\n";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result;

  (* Verify the accumulation is correct *)
  Stdio.printf "\nExpected values (summing over output dimension):\n";
  Stdio.printf "  Batch 0: [0+3+6+9=18, 1+4+7+10=22, 2+5+8+11=26]\n";
  Stdio.printf "  Batch 1: [12+15+18+21=66, 13+16+19+22=70, 14+17+20+23=74]\n"

(** Test diagonal tensor creation (not surjective, needs zero initialization) *)
let test_diagonal_tensor () =
  Stdio.printf "\n=== Testing diagonal tensor (not surjective) ===\n";
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  (* Create a diagonal tensor using einsum: i=>ii This projection is: - NOT surjective: off-diagonal
     positions never get written (need Zero_out) - Injective: each source position maps to exactly
     one target *)
  let input = TDSL.range 5 in
  let%op diagonal = input ++ "i=>ii" in

  Train.set_hosted diagonal.value;
  ignore (Train.forward_once (module Backend) diagonal);

  Stdio.printf "Input (1D tensor of size 5):\n";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false input;

  Stdio.printf "\nDiagonal tensor 'i=>ii' (5x5 with zeros off-diagonal):\n";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false diagonal;

  Stdio.printf "\nNote: Off-diagonal elements should be zero (initialized by Zero_out)\n"

(** Test fixed index projection (not surjective) *)
let test_fixed_index_projection () =
  Stdio.printf "\n=== Testing fixed index projection (not surjective) ===\n";
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  (* Create a sparse tensor using fixed index: i=>i0 This projection is: - NOT surjective: only
     column 0 gets written (need Zero_out for other columns) - Injective: each source position maps
     to exactly one target *)
  let input = TDSL.range 4 in
  let%op sparse = input ++ "i=>i0" in
  let%op _ = sparse ++ "i2=>i" in

  Train.set_hosted sparse.value;
  ignore (Train.forward_once (module Backend) sparse);

  Stdio.printf "Input (1D tensor of size 4):\n";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false input;

  Stdio.printf "\nSparse tensor 'i=>i0' (only first column populated):\n";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false sparse;

  Stdio.printf "\nNote: Only column 0 should have values, others should be zero\n"

(** Test bijective mapping (no initialization or accumulation needed) *)
let test_bijective_transpose () =
  Stdio.printf "\n=== Testing bijective transpose (optimization case) ===\n";
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  (* Simple transpose: ij=>ji This projection is: - Surjective: all target positions get written -
     Injective: each source maps to exactly one target - Therefore BIJECTIVE: can skip both Zero_out
     and accumulation *)
  let input = TDSL.range_of_shape ~output_dims:[ 3; 4 ] () in
  let%op transposed = input ++ "ij=>ji" in

  Train.set_hosted transposed.value;
  ignore (Train.forward_once (module Backend) transposed);

  Stdio.printf "Input (3x4 matrix):\n";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false input;

  Stdio.printf "\nTransposed 'ij=>ji' (4x3 matrix):\n";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false transposed;

  Stdio.printf "\nNote: Simple bijective mapping - no initialization or accumulation needed\n"

let () =
  test_einsum_reduction ();
  test_diagonal_tensor ();
  test_fixed_index_projection ();
  test_bijective_transpose ();
  Stdio.printf "\n=== All accumulation semantics tests completed ===\n"

open Base
open Ocannl
module IDX = Train.IDX
module CDSL = Train.CDSL
open Nn_blocks.DSL_modules

module type Backend = Ir.Backend_intf.Backend

let test_diagonal_tensor () =
  Stdio.printf "\nTesting diagonal tensor initialization:\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  (* Create a diagonal tensor using einsum: i->ii *)
  let input = TDSL.range 5 in
  let%op diagonal = input ++ "i=>ii" in

  (* Ensure the diagonal tensor is hosted *)
  Train.set_hosted diagonal.value;
  ignore (Train.forward_once ctx diagonal);

  (* Print the diagonal tensor *)
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false diagonal;
  Stdio.printf "\n"

let test_sparse_fixed_index () =
  Stdio.printf "\nTesting sparse assignment with fixed index:\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  (* Create a sparse tensor using fixed indices: i->i0j *)
  let input = TDSL.range 4 in
  let%op sparse = input ++ "i=>i0" in

  Train.set_hosted sparse.value;
  ignore (Train.forward_once ctx sparse);

  Train.printf ~here:[%here] ~with_code:false ~with_grad:false sparse;
  Stdio.printf "\n"

let test_multi_sparse () =
  Stdio.printf "\nTesting multiple sparse axes:\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  (* Create tensor with multiple dims and test sparse assignment *)
  let input = TDSL.range_of_shape ~output_dims:[ 3; 4 ] () in
  let%op result = input ++ "ij=>i1j" in

  Train.set_hosted result.value;
  ignore (Train.forward_once ctx result);

  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result;
  Stdio.printf "\n"

let _test_stride_gap () =
  Stdio.printf "\nTesting stride gap:\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  (* Create tensor with multiple dims and test sparse assignment *)
  let input = TDSL.range_of_shape ~output_dims:[ 2; 5 ] () in
  let%op result = input ++ "ij=>i+3*j" in

  Train.set_hosted result.value;
  ignore (Train.forward_once ctx result);

  Train.printf ~here:[%here] ~with_code:false ~with_grad:false result;
  Stdio.printf "\n"

let () =
  test_diagonal_tensor ();
  test_sparse_fixed_index ();
  test_multi_sparse ();
  (* FIXME(#354): Projections inference for convolution-style expressions not implemented yet. *)
  (* test_stride_gap (); *)
  Stdio.printf "All surjectivity tests completed.\n"

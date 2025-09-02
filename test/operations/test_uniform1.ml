open Base
module Tn = Ir.Tnode
module Asgns = Ir.Assignments
module Ndarray = Ir.Ndarray
open Ocannl.Operation.DSL_modules

let uniform1_basic_test () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let module O = TDSL.O in
  (* Create a simple 1D range tensor to use as input *)
  let range_tensor = TDSL.range 10 in

  (* Generate uniform1 values *)
  let uniform1_floats =
    O.threefry4x32
      (O.threefry4x32 (O.embed_self_id ()) (Ocannl.Tensor.get_random_seed ()))
      range_tensor
    |> O.uint4x32_to_prec_uniform1
  in

  (* Force precision to single *)
  Ir.Tnode.update_prec uniform1_floats.value Ir.Ops.single;

  (* Compile and run *)
  Ocannl.Train.set_hosted uniform1_floats.value;
  let ctx = Ocannl.Train.forward_once (module Backend) uniform1_floats in
  let result1 = Ir.Tnode.get_values uniform1_floats.value in

  (* Print first few values *)
  Stdio.printf "First 5 uniform1 random values:\n";
  for i = 0 to 4 do
    Stdio.printf "  [%d]: %.4g\n" i result1.(i)
  done;

  (* Check all values are in [0, 1) range *)
  let all_in_range = Array.for_all result1 ~f:(fun x -> Float.(x >= 0.0 && x < 1.0)) in
  Stdio.printf "All values in [0, 1) range: %b\n" all_in_range;

  (* Now test the vectorized version for comparison *)
  let uniform_floats =
    O.threefry4x32
      (O.threefry4x32 (O.embed_self_id ()) (Ocannl.Tensor.get_random_seed ()))
      range_tensor
    |> O.uint4x32_to_prec_uniform
  in

  Ir.Tnode.update_prec uniform_floats.value Ir.Ops.single;
  Ocannl.Train.set_hosted uniform_floats.value;
  ignore (Ocannl.Train.forward_once (module Backend) ~ctx uniform_floats);
  let result_vec = Ir.Tnode.get_values uniform_floats.value in

  Stdio.printf "\nFirst 5 uniform (vectorized) random values:\n";
  for i = 0 to Int.min 4 (Array.length result_vec - 1) do
    Stdio.printf "  [%d]: %.4g\n" i result_vec.(i)
  done;

  Stdio.printf "\nVectorized version produces %d values from 10 uint4x32 inputs\n"
    (Array.length result_vec);
  Stdio.printf "Non-vectorized version produces %d values from 10 uint4x32 inputs\n"
    (Array.length result1)

let uniform_at1_test () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let open Ocannl.Operation.DSL_modules in
  let module O = TDSL.O in
  (* Create a counter tensor *)
  let counter = TDSL.range 5 in

  (* Generate random values using uniform_at1 *)
  let uniform_at1_floats = O.uniform_at1 counter in
  Ir.Tnode.update_prec uniform_at1_floats.value Ir.Ops.single;

  (* Compile and run *)
  Ocannl.Train.set_hosted uniform_at1_floats.value;
  ignore (Ocannl.Train.forward_once (module Backend) uniform_at1_floats);
  let result = Ir.Tnode.get_values uniform_at1_floats.value in

  (* Print values *)
  Stdio.printf "uniform_at1 random values with counter:\n";
  for i = 0 to 4 do
    Stdio.printf "  [%d]: %.4g\n" i result.(i)
  done;

  (* Check all values are in [0, 1) range *)
  let all_in_range = Array.for_all result ~f:(fun x -> Float.(x >= 0.0 && x < 1.0)) in
  Stdio.printf "All values in [0, 1) range: %b\n" all_in_range;
  ()

let uniform1_shape_preservation_test () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let open Ocannl.Operation.DSL_modules in
  let module O = TDSL.O in
  (* Test that uniform1 preserves shape (1:1 mapping) *)
  let input_size = 24 in
  let range_tensor = TDSL.range input_size in

  let uniform1_tensor =
    O.threefry4x32
      (O.threefry4x32 (O.embed_self_id ()) (Ocannl.Tensor.get_random_seed ()))
      range_tensor
    |> O.uint4x32_to_prec_uniform1
  in

  Ir.Tnode.update_prec uniform1_tensor.value Ir.Ops.single;

  (* Compile and run *)
  Ocannl.Train.set_hosted uniform1_tensor.value;
  let ctx = Ocannl.Train.forward_once (module Backend) uniform1_tensor in
  let result = Ir.Tnode.get_values uniform1_tensor.value in

  Stdio.printf "Input size: %d elements\n" input_size;
  Stdio.printf "uniform1 output size: %d elements\n" (Array.length result);
  Stdio.printf "Shape preserved (1:1 mapping): %b\n" (Array.length result = input_size);

  (* Compare with vectorized version *)
  let uniform_vec =
    O.threefry4x32
      (O.threefry4x32 (O.embed_self_id ()) (Ocannl.Tensor.get_random_seed ()))
      range_tensor
    |> O.uint4x32_to_prec_uniform
  in

  Ir.Tnode.update_prec uniform_vec.value Ir.Ops.single;
  Ocannl.Train.set_hosted uniform_vec.value;
  ignore (Ocannl.Train.forward_once (module Backend) ~ctx uniform_vec);
  let result_vec = Ir.Tnode.get_values uniform_vec.value in

  Stdio.printf "\nVectorized uniform output size: %d elements\n" (Array.length result_vec);
  Stdio.printf "Vectorized expands by factor of %d\n" (Array.length result_vec / input_size)

let () =
  uniform1_basic_test ();
  uniform_at1_test ();
  uniform1_shape_preservation_test ()

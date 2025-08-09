open Base
module Tn = Ir.Tnode
module Asgns = Ir.Assignments
module Ndarray = Ir.Ndarray
module TDSL = Ocannl.Operation.TDSL
module O = TDSL.O

let%expect_test "threefry4x32 basic test" =
  let module Backend = (val Backends.fresh_backend ()) in
  (* Use the random seed as the key *)
  let key = Ocannl.Tensor.get_random_seed () in

  (* Create a counter tensor *)
  let counter = TDSL.range 10 in

  (* Generate random bits using threefry4x32 *)
  let random_bits = O.threefry4x32 key counter in

  (* Convert to uniform floats *)
  let uniform_floats = O.uint4x32_to_prec_uniform random_bits in
  Ir.Tnode.update_prec uniform_floats.value Ir.Ops.single;

  (* Compile and run *)
  Ocannl.Train.set_hosted uniform_floats.value;
  ignore (Ocannl.Train.forward_once (module Backend) uniform_floats);
  let result = Ir.Tnode.get_values uniform_floats.value in

  (* Print first few values *)
  Stdio.printf "First 5 uniform random values:\n";
  for i = 0 to 4 do
    Stdio.printf "  [%d]: %f\n" i result.(i)
  done;

  (* Check all values are in [0, 1) range *)
  let all_in_range = Array.for_all result ~f:(fun x -> Float.(x >= 0.0 && x < 1.0)) in
  Stdio.printf "All values in [0, 1) range: %b\n" all_in_range;

  [%expect
    {|
    First 5 uniform random values:
      [0]: 0.237546
      [1]: 0.120910
      [2]: 0.485426
      [3]: 0.235820
      [4]: 0.007262
    All values in [0, 1) range: true
    |}]

let%expect_test "uint4x32_to_prec_uniform different precisions" =
  let module Backend = (val Backends.fresh_backend ()) in
  (* TODO(#330): This is an opportunity to test that optimization context checking complains about
     the random seed being missing if it is not set here. *)
  Ocannl.Tensor.set_random_seed ();
  let key = Ocannl.Tensor.get_random_seed () in
  let counter = TDSL.range 5 in
  let random_bits = O.threefry4x32 key counter in
  let ctx = ref None in

  (* Test different target precisions *)
  let test_precision prec prec_name =
    let uniform = O.uint4x32_to_prec_uniform random_bits in
    Ir.Tnode.update_prec uniform.value prec;
    Ocannl.Train.set_hosted uniform.value;
    ctx := Some (Ocannl.Train.forward_once (module Backend) ?ctx:!ctx uniform);
    let result = Ir.Tnode.get_values uniform.value in
    Stdio.printf "%s precision - first value: %f, second value: %f\n" prec_name result.(0)
      result.(1);
    Stdio.printf "All values in [0, 1) range: %b\n"
      (Array.for_all result ~f:(fun x -> Float.(x >= 0.0 && x < 1.0)))
  in

  test_precision Ir.Ops.single "Single";
  (* Metal backend doesn't support double precision. *)
  (* test_precision Ir.Ops.double "Double"; *)
  test_precision Ir.Ops.half "Half";

  [%expect
    {|
    Single precision - first value: 0.237546, second value: 0.120910
    All values in [0, 1) range: true
    Half precision - first value: 0.831543, second value: 0.237549
    All values in [0, 1) range: true
    |}]

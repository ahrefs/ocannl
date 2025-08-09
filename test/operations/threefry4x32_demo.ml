(* A simple demo of using the Threefry4x32 PRNG in OCANNL *)

open Base
module Tensor = Ocannl.Tensor
module TDSL = Ocannl.Operation.TDSL
module O = TDSL.O

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  (* This is a stress-test for shape inference, usually we would use Tensor.number *)
  let key =
    Ocannl.Tensor.term ~label:[ "random_seed" ] ~grad_spec:Prohibit_grad
      ~fetch_op:(Ir.Assignments.Constant_fill [| 42. |]) ()
  in
  Ir.Tnode.update_prec key.value Ir.Ops.uint4x32;

  (* Create a counter tensor with values 0..5 *)
  let counter = TDSL.range 5 in

  (* Generate random bits using threefry4x32 *)
  let random_bits = O.threefry4x32 key counter in

  (* Convert to uniform floats in [0, 1) *)
  let uniform_floats = O.uint4x32_to_prec_uniform random_bits in
  Ir.Tnode.update_prec uniform_floats.value Ir.Ops.half;

  (* Compile and run *)
  Ocannl.Train.set_hosted uniform_floats.value;
  ignore (Ocannl.Train.forward_once (module Backend) uniform_floats);
  let result = Ir.Tnode.get_values uniform_floats.value in

  (* Print the results *)
  Stdio.printf "Generated %d uniform random numbers:\n" (Array.length result);
  Array.iteri result ~f:(fun i x -> Stdio.printf "  [%d]: %f\n" i x)

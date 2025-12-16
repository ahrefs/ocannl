(* A simple demo of using the Threefry4x32 PRNG in OCANNL *)

open Base
open Ocannl.Nn_blocks.DSL_modules
module O = TDSL.O

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  (* This is a stress-test for shape inference, usually we would use %op or %cd syntax *)
  let key =
    Ocannl.Tensor.term ~label:[ "random_seed" ] ~grad_spec:Prohibit_grad
      ~fetch_op:(Ir.Assignments.Constant_bits 42L) ()
  in
  Ir.Tnode.update_prec key.value Ir.Ops.uint4x32;
  let range = TDSL.range 5 in
  let%op random_bits = threefry4x32 key range in

  (* Convert to uniform floats in [0, 1) *)
  let%op uniform_floats = uint4x32_to_prec_uniform random_bits in
  Ir.Tnode.update_prec uniform_floats.value Ir.Ops.half;

  (* Compile and run *)
  Ocannl.Train.set_hosted uniform_floats.value;
  ignore (Ocannl.Train.forward_once ctx uniform_floats);
  let result = Ir.Tnode.get_values uniform_floats.value in

  (* Print the results *)
  Stdio.printf "Generated %d uniform random numbers:\n" (Array.length result);
  Array.iteri result ~f:(fun i x -> Stdio.printf "  [%d]: %.3g\n" i x)

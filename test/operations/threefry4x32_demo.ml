(* A simple demo of using the Threefry4x32 PRNG in OCANNL *)

open Base
module Backend = Arrayjit.Backends.Cc_backend
module TDSL = Operation.TDSL
module O = TDSL.O

let () =
  (* Use the random seed as the key *)
  let key = !Operation.random_seed in
  
  (* Create a counter tensor with values 0..9 *)
  let counter = TDSL.range 10 in
  
  (* Generate random bits using threefry4x32 *)
  let random_bits = O.threefry4x32 key counter in
  
  (* Convert to uniform floats in [0, 1) *)
  let uniform_floats = O.uint4x32_to_prec_uniform ~target_prec:(Ir.Ops.single ()) random_bits in
  
  (* Compile and run *)
  Train.set_hosted uniform_floats.value;
  let ctx = Train.forward_and_ctx Backend.runner uniform_floats in
  let result = Backend.to_float ctx uniform_floats in
  
  (* Print the results *)
  Stdio.printf "Generated %d uniform random numbers:\n" (Array.length result);
  Array.iteri result ~f:(fun i x ->
    Stdio.printf "  [%d]: %f\n" i x
  )
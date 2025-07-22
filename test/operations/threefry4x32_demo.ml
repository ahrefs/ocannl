(* A simple demo of using the Threefry4x32 PRNG in OCANNL *)

open Base
module TDSL = Ocannl.Operation.TDSL
module O = TDSL.O

let () =
  let module Backend = (val Backends.fresh_backend ()) in
  (* Use the random seed as the key *)
  let key = !Ocannl.Operation.random_seed in
  
  (* Create a counter tensor with values 0..9 *)
  let counter = TDSL.range 10 in
  
  (* Generate random bits using threefry4x32 *)
  let random_bits = O.threefry4x32 key counter in
  
  (* Convert to uniform floats in [0, 1) *)
  let uniform_floats = O.uint4x32_to_prec_uniform  random_bits in
  Ir.Tnode.update_prec uniform_floats.value Ir.Ops.single;
  
  (* Compile and run *)
  Ocannl.Train.set_hosted uniform_floats.value;
  ignore (Ocannl.Train.forward_once (module Backend) uniform_floats);
  let result = Ir.Tnode.get_values uniform_floats.value in
  
  (* Print the results *)
  Stdio.printf "Generated %d uniform random numbers:\n" (Array.length result);
  Array.iteri result ~f:(fun i x ->
    Stdio.printf "  [%d]: %f\n" i x
  )
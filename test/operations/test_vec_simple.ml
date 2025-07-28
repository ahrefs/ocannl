open Base
module TDSL = Ocannl.Operation.TDSL
module O = TDSL.O

let () =
  let module Backend = (val Backends.fresh_backend ()) in
  
  (* Create a simple test with a fixed index *)
  let key = Ocannl.Tensor.number ~label:[ "key" ] ~grad_spec:Prohibit_grad 42. in
  Ir.Tnode.update_prec key.value Ir.Ops.uint4x32;
  
  (* Single value counter *)
  let counter = Ocannl.Tensor.number ~label:[ "counter" ] ~grad_spec:Prohibit_grad 0. in
  
  (* Generate random bits *)
  let random_bits = O.threefry4x32 key counter in
  
  (* Convert to uniform floats *)
  let uniform_floats = O.uint4x32_to_prec_uniform random_bits in
  Ir.Tnode.update_prec uniform_floats.value Ir.Ops.single;
  
  (* Compile and run *)
  Ocannl.Train.set_hosted uniform_floats.value;
  ignore (Ocannl.Train.forward_once (module Backend) uniform_floats);
  let result = Ir.Tnode.get_values uniform_floats.value in
  
  (* Print results *)
  Stdio.printf "Generated %d uniform random numbers:\n" (Array.length result);
  Array.iteri result ~f:(fun i x -> Stdio.printf "  [%d]: %f\n" i x)
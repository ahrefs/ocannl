open Base
open Ocannl.Operation.DSL_modules

let () =
  let module Backend = (val Backends.fresh_backend ()) in
  (* Generate random bits *)
  let%op random_bits = threefry4x32 (embed_self_id ()) 42L in

  (* Convert to uniform floats *)
  let%op uniform_floats = uint4x32_to_prec_uniform random_bits in
  Ir.Tnode.update_prec uniform_floats.value Ir.Ops.single;

  (* Compile and run *)
  Ocannl.Train.set_hosted uniform_floats.value;
  ignore (Ocannl.Train.forward_once (module Backend) uniform_floats);
  let result = Ir.Tnode.get_values uniform_floats.value in

  (* Print results *)
  Stdio.printf "Generated %d uniform random numbers:\n" (Array.length result);
  Array.iteri result ~f:(fun i x -> Stdio.printf "  [%d]: %.4g\n" i x)

(* A simple demo of using the Threefry4x32 PRNG in OCANNL *)

open Base
module Tensor = Ocannl.Tensor
module Train = Ocannl.Train
module TDSL = Ocannl.Operation.TDSL
module Tn = Ir.Tnode

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let%op d = ("a" [ 2 ] + "b" [ 2 ]) *. "c" [ 2 ] in
  Tn.update_prec b.value Ir.Ops.half;
  Tn.update_prec d.value Ir.Ops.bfloat16;
  (* Even when the default precision is single, c is bfloat16 and a is half. *)
  Ocannl.Train.set_hosted d.value;
  ignore (Ocannl.Train.forward_once (module Backend) d);
  Train.printf_tree d

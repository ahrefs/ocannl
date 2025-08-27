open Base
module Tensor = Ocannl.Tensor
module Train = Ocannl.Train
module NTDSL = Ocannl.Operation.NTDSL
module Tn = Ir.Tnode

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  (* Test heterogeneous precision in where operation using %cd syntax *)
  let%cd condition = { cond } in
  let%cd then_val = { a } in
  let%cd else_val = { b } in
  let%cd result = where condition then_val else_val in

  (* Set different precisions for each argument *)
  Tn.update_prec condition.value Ir.Ops.byte;
  (* condition uses byte precision *)
  Tn.update_prec then_val.value Ir.Ops.half;
  (* then branch is half *)
  Tn.update_prec else_val.value Ir.Ops.bfloat16;
  (* else branch is bfloat16 *)
  Tn.update_prec result.value Ir.Ops.single;

  (* result is single *)

  (* Initialize on host (rather than on device). *)
  Tn.set_values cond.value [| 0.0 |];
  Tn.set_values a.value [| 1.0 |];
  Tn.set_values b.value [| 2.0 |];

  (* Set up values and run computation *)
  ignore (Train.forward_once (module Backend) result);
  Train.printf_tree result

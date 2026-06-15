module Train = Ocannl.Train
open Ocannl.Nn_blocks.DSL_modules
module Tn = Ir.Tnode

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
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

  (* Compile the forward computation, set the inputs on-device via the context, then run
     (gh-ocannl-333: values live on devices, set on demand through the context). *)
  let routine = Train.to_routine ctx Train.IDX.empty (Train.forward result) in
  let ctx = Context.context routine in
  let ctx = Context.set_values ctx cond.value [| 0.0 |] in
  let ctx = Context.set_values ctx a.value [| 1.0 |] in
  let ctx = Context.set_values ctx b.value [| 2.0 |] in
  Train.run ctx routine;
  Train.printf_tree ctx result

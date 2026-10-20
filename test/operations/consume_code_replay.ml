(* Replay comp handouts after constructing a later consumer. Recompile and execute with changed
   inputs so an empty replay or a stale device buffer cannot satisfy the numerical claims. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

let values ctx name node expected =
  p_close name (Context.get_values ctx node) expected ~tol:(fun _ -> 1e-5)

let () =
  let w = TDSL.param ~values:[| -2.; 3. |] "replay_w" ~output_dims:[ 2 ] () in
  let y = TDSL.O.( *. ) w w in
  let grad = (Option.value_exn w.Tensor.diff).grad in
  Train.set_materialized grad;
  let first_forward = Tensor.consume_forward_code y in
  let first_backprop = Tensor.consume_backprop_code y in
  let ctx = Train.init_params (Context.auto ()) Train.IDX.empty y in
  let ctx, first = Train.to_routine ctx Train.IDX.empty (Train.grad_update y) in
  let ctx = Context.run ctx first in
  values ctx "first handout computes squares" y.value [| 4.; 9. |];
  values ctx "first handout computes gradients" grad [| -4.; 6. |];
  (* Both sides of y have already been handed out. This consumer references y without acquiring its
     code; replay must leave the consumer available for a later handout of its own. *)
  let z = TDSL.O.( + ) y y in
  p "new consumer starts as a forward root" (Tensor.is_fwd_root z);
  p "new consumer starts as a backprop root" (Tensor.is_bprop_root z);
  p "forward replay keeps comp identity with a newer root"
    (phys_equal first_forward (Tensor.consume_forward_code y));
  p "backprop replay keeps comp identity with a newer root"
    (phys_equal first_backprop (Tensor.consume_backprop_code y));
  let ctx = Context.set_values ctx w.value [| 5.; -7. |] in
  let ctx, replay_forward = Train.to_routine ctx Train.IDX.empty (Train.forward y) in
  let ctx = Context.run ctx replay_forward in
  values ctx "recompiled forward reads changed inputs" y.value [| 25.; 49. |];
  let ctx, replay_update = Train.to_routine ctx Train.IDX.empty (Train.grad_update y) in
  let ctx = Context.run ctx replay_update in
  values ctx "recompiled backprop resets and recomputes gradients" grad [| 10.; -14. |];
  p "replay leaves the newer forward root intact" (Tensor.is_fwd_root z);
  p "replay leaves the newer backprop root intact" (Tensor.is_bprop_root z);
  let ctx, consumer = Train.to_routine ctx Train.IDX.empty (Train.forward z) in
  let ctx = Context.run ctx consumer in
  values ctx "new consumer executes after replay" z.value [| 50.; 98. |];
  (* Backpropagate the consumer first, then the retained y fragment, as a caller retaining the
     original comp would. Seed and zero explicitly: raw backprop does not include either. *)
  let ydiff = Option.value_exn y.Tensor.diff in
  let zdiff = Option.value_exn z.Tensor.diff in
  let zero_y = ydiff.zero_grads and zero_z = zdiff.zero_grads in
  let update =
    [%cd
      zero_y;
      zero_z;
      z.grad =: 1;
      z.backprop;
      y.backprop]
  in
  let update =
    { update with asgns = Ir.Assignments.Block_comment ("consumer replay", update.asgns) }
  in
  let ctx, consumer_update = Train.to_routine ctx Train.IDX.empty update in
  let ctx = Context.run ctx consumer_update in
  values ctx "new consumer and replayed fragment compose gradients" grad [| 20.; -28. |];
  Context.release ctx

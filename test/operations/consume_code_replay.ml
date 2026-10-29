(* Replay comp handouts after constructing a later consumer (gh-ocannl-955). A replay compiles as a
   sibling -- from a context that does not descend from the first handout's routine -- since a
   second copy of a fragment in one lineage is refused (gh-ocannl-1724). Recompile and execute with
   changed inputs so an empty replay or a stale device buffer cannot satisfy the numerical
   claims. *)
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
  let ctx0 = Train.init_params (Context.auto ()) Train.IDX.empty y in
  let ctx, first = Train.to_routine ctx0 Train.IDX.empty (Train.grad_update y) in
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
  (* In the lineage that compiled them, the replayed fragments are refused: rerun [first]
     instead. *)
  let refused name comp =
    match Train.to_routine ctx Train.IDX.empty comp with
    | _ -> p (name ^ " (accepted)") false
    | exception Utils.User_error msg ->
        p name (String.is_substring msg ~substring:"already compiled in this context's lineage")
  in
  refused "a forward replay in the first routine's lineage is refused" (Train.forward y);
  refused "a backprop replay in the first routine's lineage is refused" (Train.grad_update y);
  (* As siblings, from the context before [first], they compile and run. *)
  let ctx0 = Context.set_values ctx0 w.value [| 5.; -7. |] in
  let ctx_fwd, replay_forward = Train.to_routine ctx0 Train.IDX.empty (Train.forward y) in
  let ctx_fwd = Context.run ctx_fwd replay_forward in
  values ctx_fwd "recompiled forward reads changed inputs" y.value [| 25.; 49. |];
  let ctx, replay_update = Train.to_routine ctx0 Train.IDX.empty (Train.grad_update y) in
  let ctx = Context.run ctx replay_update in
  values ctx "recompiled backprop resets and recomputes gradients" grad [| 10.; -14. |];
  p "replay leaves the newer forward root intact" (Tensor.is_fwd_root z);
  p "replay leaves the newer backprop root intact" (Tensor.is_bprop_root z);
  let ctx, consumer = Train.to_routine ctx Train.IDX.empty (Train.forward z) in
  let ctx = Context.run ctx consumer in
  values ctx "new consumer executes after replay" z.value [| 50.; 98. |];
  (* Backpropagate the consumer first, then the retained y fragment, as a caller retaining the
     original comp would. Seed and zero explicitly: raw backprop does not include either. One
     routine carries both fragments, so [z]'s requirement of [y]'s backprop is met inside it; it
     compiles in the forward replay's lineage, which owns no backprop of [y]. *)
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
  p "the consumer's backprop requires y's, which the composed routine carries"
    (Set.mem update.requires (ydiff.grad, Ir.Assignments.Bprop)
    && Set.mem update.owned (ydiff.grad, Ir.Assignments.Bprop));
  let ctx, consumer_update = Train.to_routine ctx_fwd Train.IDX.empty update in
  let ctx = Context.run ctx consumer_update in
  values ctx "new consumer and replayed fragment compose gradients" grad [| 20.; -28. |];
  Context.release ctx

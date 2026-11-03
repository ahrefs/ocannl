(* The link refusal "The linked context lacks node <g>" names the missing reset when [g] is a
   gradient the computation accumulates into and nothing in its lineage zeroes (wave-20261031
   post-1-task-7, from lukstafi/ocannl-staging#1558). [Tensor.op] carries a gradient's zeroing in
   its tensor's first-constructed consumer only (gh-ocannl-1744); when that consumer was evaluated
   forward-only -- and, since gh-ocannl-1745, dropped and reclaimed -- a later loss over the
   parameter accumulates into a gradient no routine resets, and the link fails over it. The golden
   pins that message in full; the suggested remedy (a [Train.zero_params_grads] routine compiled
   first) is run here with exact values, and a step skipping it is refused (gh-ocannl-1781); and the
   other "lacks node" refusals keep their text: the bare one over a non-gradient node, and
   gh-ocannl-1724's hint over an unpropagated intermediate gradient. Values are small integers,
   compared bitwise. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module LC = Lineage_claims
module IDX = Train.IDX

let one t = Set.singleton (module Tensor) t
let compile ctx comp = Train.to_routine ctx IDX.empty comp
let run_all ctx routines = List.fold routines ~init:ctx ~f:Context.run
let reset_hint = "without zeroing it, and no routine of its lineage zeroes it"
let param prefix = TDSL.param ~values:[| 1.; 2.; 3. |] (prefix ^ "_w") ~output_dims:[ 3 ] ()

(* #1558's first consumer: built, evaluated forward-only, and dropped -- it carried [w]'s zeroing.
   Returns the context and an observer on the consumer. *)
let[@inline never] forward_only_consumer ctx w =
  let%op c = w *. 2. in
  (Train.forward_once ctx c, Gc_witness.witness c)

let () =
  case "a parameter whose first consumer was evaluated forward-only and dropped" (fun () ->
      let w = param "lh" in
      let ctx, c = forward_only_consumer (Context.auto ()) w in
      p "the forward-only first consumer is dropped and reclaimed" (Gc_witness.reclaimed c);
      claim "control: the parameter the case still holds is kept"
        (Gc_witness.kept (Gc_witness.witness w));
      let%op l = w *. w in
      let msg = LC.refusal "a loss over the parameter" (fun () -> Train.update_once ctx l) in
      Stdio.printf "The refusal:\n%s\n%!" msg;
      p "names the parameter's gradient, the missing reset and the remedy"
        (LC.has msg ("The linked context lacks node " ^ Ir.Tnode.label_name (LC.grad_of w))
        && LC.has msg reset_hint
        && LC.has msg "Train.zero_params_grads ~params");
      (* The remedy the message suggests. [l]'s [zero_grads] does not zero [w.grad] (its first
         consumer's did), so [Train.grad_update l] accumulates over the reset compiled before it. *)
      let ctx, zero = compile ctx (Train.zero_params_grads ~params:(one w) l) in
      let ctx, step = compile ctx (Train.grad_update l) in
      let ctx = run_all ctx [ zero; step; zero; step ] in
      LC.exact "reset before each step: w.grad = 2 w"
        (Context.get_values ctx (LC.grad_of w))
        [| 2.; 4.; 6. |];
      (* Without the reset the step would sum two steps' gradients: refused (gh-ocannl-1781). *)
      let msg = LC.refusal "a step without the reset" (fun () -> Context.run ctx step) in
      p "a step without the reset is refused, naming the reset routine"
        (LC.has msg (Ir.Tnode.label_name (LC.grad_of w)) && LC.has msg "has not run since");
      LC.exact "the refusal dispatched nothing"
        (Context.get_values ctx (LC.grad_of w))
        [| 2.; 4.; 6. |]);
  case "a parameter whose first consumer's update was compiled in a sibling" (fun () ->
      let w = param "lh_sib" in
      let%op c = w *. 2. in
      let%op l = w *. w in
      let ctx0 = Train.init_params (Context.auto ()) IDX.empty c in
      let _sibling = compile ctx0 (Train.grad_update c) in
      let msg =
        LC.refusal "a loss over the parameter, from the common ancestor" (fun () ->
            compile ctx0 (Train.grad_update l))
      in
      p "a reset compiled in a sibling does not count, and the hint names it"
        (LC.has msg ("The linked context lacks node " ^ Ir.Tnode.label_name (LC.grad_of w))
        && LC.has msg reset_hint));
  case "a node that is not a gradient: the bare message" (fun () ->
      let x = Tensor.term_init [| 1.; 2.; 3. |] ~label:[ "lh_x" ] ~output_dims:[ 3 ] () in
      let%op a = x *. 2. in
      let _ctx_a, _ = compile (Context.auto ()) (Train.forward a) in
      let%op b = a *. 3. in
      let msg =
        LC.refusal "a reader of [a] on a fresh context" (fun () ->
            compile (Context.auto ()) (Train.forward b))
      in
      p "the refusal is exactly the bare message"
        (String.equal msg ("The linked context lacks node " ^ Ir.Tnode.label_name a.value)));
  case "an unpropagated intermediate gradient: gh-ocannl-1724's hint" (fun () ->
      let w =
        Tensor.term_init [| 1.; 2.; 3. |] ~label:[ "lh_t_w" ] ~output_dims:[ 3 ]
          ~grad_spec:Require_grad ()
      in
      let%op h = w *. w in
      let h_fwd = Train.forward h in
      (* [l1], constructed first, carries [h]'s zeroing and backprop; its code is handed out (the
         conflicting-roots check wants the trunk's owner consumed first) and never compiled. *)
      let%op l1 = h *. 2. in
      let%op l2 = h *. 3. in
      let _l1_update = Train.grad_update l1 in
      Train.set_materialized (LC.grad_of h);
      let ctx, r = compile (Context.auto ()) h_fwd in
      let ctx = Context.run ctx r in
      let msg = LC.refusal "a loss over the trunk" (fun () -> Train.update_once ctx l2) in
      p "names the unpropagated gradient, and not the reset"
        (LC.has msg ("The linked context lacks node " ^ Ir.Tnode.label_name (LC.grad_of h))
        && LC.has msg "no routine in this lineage propagates from"
        && not (LC.has msg reset_hint)))

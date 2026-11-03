(* gh-ocannl-1746: a tensor's code fragment sequenced twice inside ONE computation. A comp's [owned]
   is a set, so the copy merged into one entry while [asgns] ran it twice -- unseen by
   [Context.compile]'s lineage check (gh-ocannl-1724), which compares sets. [Asgns.sequence] now
   refuses members that share a fragment, naming it.

   The probe is [t = w *. w], [loss = t *. 3] over a [Require_grad] leaf [w = 1;2;3], so [d loss / d
   w = 6 w] -- exact by construction, compared bitwise. Its backprop sequenced twice computes [18
   w]: the second copy adds [3] to [t.grad] again and propagates the sum, [2 w * 6]. The B legs
   build that composition by hand, past [sequence], to show the wrong answer the refusal prevents.

   The O legs pin the duplicate the check found in OCANNL itself: a [%cd] [op_asn] naming an operand
   tensor ([Operation.deinterleave_even]'s [id t1]) embedded the operand's forward code a second
   time, beside [Tensor.op]'s own embedding of it, and marked it taken, so a later
   [consume_forward_code] handed it out again instead of refusing. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module LC = Lineage_claims
module IDX = Train.IDX
module Asgns = Ir.Assignments

let shared_named msg tn ~side =
  LC.has msg "occurs in more than one member of this sequence"
  && LC.has msg (side ^ " " ^ Ir.Tnode.label_name tn)

let probe prefix =
  let w =
    Tensor.term_init [| 1.; 2.; 3. |]
      ~label:[ prefix ^ "_w" ]
      ~output_dims:[ 3 ] ~grad_spec:Require_grad ()
  in
  let%op t = w *. w in
  let%op loss = t *. 3. in
  Train.set_materialized (LC.grad_of w);
  (w, loss)

let w_grad_after w comp =
  let ctx, routine = Train.to_routine (Context.auto ()) IDX.empty comp in
  Context.get_values (Context.run ctx routine) (LC.grad_of w)

let () =
  case "B: a backprop fragment sequenced twice" (fun () ->
      let w, loss = probe "fst_b" in
      let update = Train.grad_update loss in
      (* [grad_update] took the loss's backprop; a later consumption hands the same comp out. *)
      let replay = Tensor.consume_backprop_code loss in
      let msg = LC.refusal "B" (fun () -> Asgns.sequence [ update; replay ]) in
      p "B: refused, naming the loss's backprop fragment"
        (shared_named msg (LC.grad_of loss) ~side:"propagating from" && LC.has msg "members 1, 2");
      LC.exact "B: the update carrying it once computes 6 w" (w_grad_after w update)
        [| 6.; 12.; 18. |];
      (* What the refusal prevents: the same two members run in sequence, past [sequence]. *)
      let doubled =
        Asgns.comp ~owned:update.owned ~requires:update.requires
          ~embedded:(Set.union update.embedded_nodes replay.embedded_nodes)
          (Asgns.Seq (update.asgns, replay.asgns))
      in
      LC.exact "B: run twice, it double-counts into 18 w" (w_grad_after w doubled)
        [| 18.; 36.; 54. |]);
  case "F: a forward fragment sequenced twice" (fun () ->
      let x = TDSL.ndarray [| 1.; 2.; 3. |] ~label:[ "fst_f_x" ] ~output_dims:[ 3 ] () in
      let%op y = x *. x in
      let fwd = Train.forward y in
      let msg = LC.refusal "F" (fun () -> Asgns.sequence [ fwd; Train.forward y ]) in
      p "F: refused, naming the forward fragment" (shared_named msg y.value ~side:"forward code of"));
  (* The members are different comps: [z] embeds [a]'s forward code, and [a.forward] carries it too.
     The check is on fragments, not on comps. *)
  case "E: a fragment shared through an embedding" (fun () ->
      let x = TDSL.ndarray [| 1.; 2.; 3. |] ~label:[ "fst_e_x" ] ~output_dims:[ 3 ] () in
      let%op a = x *. x in
      let%op z = a + 1. in
      let msg = LC.refusal "E" (fun () -> Asgns.sequence [ Train.forward z; a.Tensor.forward ]) in
      p "E: refused, naming the embedded fragment"
        (shared_named msg a.value ~side:"forward code of"));
  case "O: an op_asn naming its operand" (fun () ->
      let x =
        NTDSL.init ~l:"fst_o_x" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ 6 ]
          ~f:(function [| i |] -> Float.of_int (i + 1) | _ -> assert false)
          ()
      in
      let%op x2 = x *. 2. in
      let%op even = deinterleave_even x2 in
      p "O: the operand is no longer a forward root" (not (Tensor.is_fwd_root x2));
      p "O: and was not marked taken: consuming it is refused"
        (match Tensor.consume_forward_code x2 with
        | _ -> false
        | exception Tensor.Session_error (msg, _) -> LC.has msg "is not a forward root");
      let ctx = Train.forward_once (Context.auto ()) even in
      LC.exact "O: the even positions of 2 x" (Context.get_values ctx even.value) [| 2.; 6.; 10. |])

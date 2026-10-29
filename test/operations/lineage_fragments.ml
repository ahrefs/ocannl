(* gh-ocannl-1724: every code fragment generated from tensors is compiled into exactly one routine
   of a lineage, and every fragment a routine depends on is compiled somewhere in that lineage.
   [Context.compile] refuses a fragment the lineage already owns (a second copy of a backprop
   fragment double-counts a gradient); [Context.run] refuses a routine whose gradient writes flow
   into a backprop fragment no routine of its lineage owns (nothing would propagate from it, and the
   unread write is dead code -- a silent drop). Sibling compiles, from a context that does not
   descend from the owner, are alternatives: they never collide, and never satisfy a requirement.

   The probe legs are the issue's: [w] is a [Require_grad] leaf [1;2;3] or a parameter, [t = w *.
   w], [loss = t *. 3], so [d loss / d w = 6 w] -- exact by construction, compared bitwise. Owner
   presence is NOT proof of a correct gradient: the shared-trunk legs only assert the checks'
   verdicts (the zeroing order between two routines over one trunk is a separate hazard). *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module IDX = Train.IDX
module Asgns = Ir.Assignments

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad
let has msg substring = String.is_substring msg ~substring
let exact name got want = p_close name got want ~tol:(fun _ -> 0.)

let leaf name =
  Tensor.term_init [| 1.; 2.; 3. |] ~label:[ name ] ~output_dims:[ 3 ] ~grad_spec:Require_grad ()

(* The text of the [User_error] [f] raises; a failed claim when it raises none. *)
let refusal name f =
  match f () with
  | _ ->
      fail (name ^ ": was accepted");
      ""
  | exception Utils.User_error msg ->
      Stdio.eprintf "%s: %s (not part of the golden)\n%!" name msg;
      msg

let accepted name f =
  match f () with
  | x ->
      p name true;
      Some x
  | exception Utils.User_error msg ->
      p (name ^ " (refused: " ^ msg ^ ")") false;
      None

let duplicate_named msg tn =
  has msg "already compiled in this context's lineage" && has msg (Ir.Tnode.debug_name tn)

(* Either refusal of an unpropagated gradient: [Context.run]'s requirement check, or -- when the
   dangling gradient is materialized, so the link fails first -- the link error's lineage hint. *)
let unpropagated msg tn =
  let name = Ir.Tnode.debug_name tn in
  (has msg "no routine of its lineage carries" && has msg ("propagating from " ^ name))
  || has msg ("no routine in this lineage propagates from " ^ name)

(* [t = w *. w] and [loss = t *. 3] over a fresh [Require_grad] leaf. *)
let probe prefix =
  let w = leaf (prefix ^ "_w") in
  let%op t = w *. w in
  Train.set_materialized (grad_of w);
  (w, t)

let () =
  case "A: a fresh graph trains" (fun () ->
      let w, t = probe "lf_a" in
      let%op loss = t *. 3. in
      let ctx = Train.update_once (Context.auto ()) loss in
      exact "A: w.grad = 6 w" (Context.get_values ctx (grad_of w)) [| 6.; 12.; 18. |]);
  case "B: forward_once, then a loss over it" (fun () ->
      let _w, t = probe "lf_b" in
      let ctx = Train.forward_once (Context.auto ()) t in
      let%op loss = t *. 3. in
      let msg = refusal "B" (fun () -> Train.update_once ctx loss) in
      p "B: refused, naming t's backprop fragment" (unpropagated msg (grad_of t)));
  case "B': B with t.grad materialized" (fun () ->
      let _w, t = probe "lf_b2" in
      Train.set_materialized (grad_of t);
      let ctx = Train.forward_once (Context.auto ()) t in
      let%op loss = t *. 3. in
      let msg = refusal "B'" (fun () -> Train.update_once ctx loss) in
      p "B': the link error says no routine of the lineage propagates from t.grad"
        (has msg ("no routine in this lineage propagates from " ^ Ir.Tnode.debug_name (grad_of t))));
  case "C: forward_once twice" (fun () ->
      let _w, t = probe "lf_c" in
      let ctx0 = Context.auto () in
      let ctx1 = Train.forward_once ctx0 t in
      let msg = refusal "C" (fun () -> Train.forward_once ctx1 t) in
      p "C: the second, on the returned context, is refused naming t's forward fragment"
        (duplicate_named msg t.value);
      exact "C: the refusal left the first run's value" (Context.get_values ctx1 t.value)
        [| 1.; 4.; 9. |];
      Option.iter
        (accepted "C: the same compile as a sibling is accepted" (fun () ->
             Train.forward_once ctx0 t))
        ~f:(fun ctx ->
          exact "C: the sibling computes t" (Context.get_values ctx t.value) [| 1.; 4.; 9. |]));
  case "D: update_once twice" (fun () ->
      let w, t = probe "lf_d" in
      let%op loss = t *. 3. in
      let ctx0 = Context.auto () in
      let ctx1 = Train.update_once ctx0 loss in
      let msg = refusal "D" (fun () -> Train.update_once ctx1 loss) in
      p "D: the second, on the returned context, is refused naming loss's backprop fragment"
        (duplicate_named msg (grad_of loss));
      Option.iter
        (accepted "D: the same compile as a sibling is accepted" (fun () ->
             Train.update_once ctx0 loss))
        ~f:(fun ctx ->
          exact "D: the sibling trains w" (Context.get_values ctx (grad_of w)) [| 6.; 12.; 18. |]));
  case "E/F: Train.forward / Train.grad_update after a once-run" (fun () ->
      let _w, t = probe "lf_e" in
      let ctx0 = Context.auto () in
      let ctx1 = Train.forward_once ctx0 t in
      let msg = refusal "E" (fun () -> Train.to_routine ctx1 IDX.empty (Train.forward t)) in
      p "E: Train.forward t in the once-run's lineage is refused" (duplicate_named msg t.value);
      ignore
        (accepted "E: as a sibling it is accepted" (fun () ->
             Train.to_routine ctx0 IDX.empty (Train.forward t)));
      let _w, t = probe "lf_f" in
      let%op loss = t *. 3. in
      let ctx1 = Train.update_once ctx0 loss in
      let msg = refusal "F" (fun () -> Train.to_routine ctx1 IDX.empty (Train.grad_update loss)) in
      p "F: Train.grad_update loss in the once-run's lineage is refused"
        (duplicate_named msg (grad_of loss));
      ignore
        (accepted "F: as a sibling it is accepted" (fun () ->
             Train.to_routine ctx0 IDX.empty (Train.grad_update loss))));
  (* A shared non-parameter trunk [h]: [l1]'s construction embeds [h]'s backprop, so [l2]'s skips it
     and requires [(h.grad, Bprop)]. [h]'s forward is handed out first, as its own routine, so that
     both losses read a materialized [h] whichever is compiled first; [l1]'s code is handed out
     before [l2]'s (the session-level conflicting-roots check of [consume_forward_code] wants the
     trunk's owner consumed first). *)
  let trunk prefix =
    let w = TDSL.param ~values:[| 1.; 2.; 3. |] (prefix ^ "_w") ~output_dims:[ 3 ] () in
    let%op h = w *. w in
    let h_fwd = Train.forward h in
    let%op l1 = h *. 2. in
    let%op l2 = h *. 3. in
    (* [h]'s gradient zeroing, which owns no fragment: a routine of its own allocates [h.grad], so a
       loss accumulating into [h.grad] without carrying [h]'s backprop links, and the requirement is
       [Context.run]'s to check. *)
    let h_zero =
      let z = (Option.value_exn ~here:[%here] h.diff).zero_grads in
      { z with asgns = Asgns.Block_comment (prefix ^ "_h_zero", z.asgns) }
    in
    (* A fresh root with the parameters initialized, [h] computed and [h.grad] zeroed. *)
    let base () =
      let ctx = Train.init_params (Context.auto ()) IDX.empty l1 in
      let ctx, r = Train.to_routine ctx IDX.empty h_fwd in
      let ctx = Context.run ctx r in
      let ctx, r = Train.to_routine ctx IDX.empty h_zero in
      Context.run ctx r
    in
    (h, l1, l2, base)
  in
  case "shared trunk" (fun () ->
      let h, l1, l2, base = trunk "lf_st" in
      p "l2 requires h's backprop fragment"
        (Set.mem (Option.value_exn ~here:[%here] l2.diff).backprop.requires
           (grad_of h, Asgns.Bprop));
      let ctx0 = base () in
      (* The owner handed out but never compiled. *)
      let l1_update = Train.grad_update l1 in
      let msg = refusal "trunk, owner absent" (fun () -> Train.update_once ctx0 l2) in
      p "with no owner compiled, l2 is refused at run naming h's backprop fragment"
        (unpropagated msg (grad_of h)
        && has msg "accumulates into gradients"
        && has msg "no routine of this root context compiled it");
      let ctx_l1, r1 = Train.to_routine ctx0 IDX.empty l1_update in
      let ctx_l1 = Context.run ctx_l1 r1 in
      let msg = refusal "trunk, sibling owner" (fun () -> Train.update_once ctx0 l2) in
      p "an owner compiled -- and run -- in a sibling does not count, and is named"
        (unpropagated msg (grad_of h) && has msg "compiled outside this lineage");
      ignore
        (accepted "l2 compiled in l1's lineage is accepted (no claim on the gradient values)"
           (fun () -> Train.update_once ctx_l1 l2)));
  case "late owner" (fun () ->
      let h, l1, l2, base = trunk "lf_lo" in
      let ctx = base () in
      let l1_update = Train.grad_update l1 in
      let ctx_a, r2 = Train.to_routine ctx IDX.empty (Train.grad_update l2) in
      let ctx_b, _r1 = Train.to_routine ctx_a IDX.empty l1_update in
      let msg = refusal "late owner, run before it" (fun () -> Context.run ctx_a r2) in
      p "run from the context before the owner's compile, l2's routine is refused"
        (unpropagated msg (grad_of h) && has msg "compiled outside this lineage");
      ignore
        (accepted "run from the owner's descendant, l2's routine is accepted" (fun () ->
             Context.run ctx_b r2)));
  case "independent roots with coincident routine ids" (fun () ->
      let h, l1, l2, base = trunk "lf_ir" in
      (* Root X: init, h's forward, l1's update. Root Y: init, h's forward, l2's update. *)
      let l1_update = Train.grad_update l1 in
      let ctx_x, r1 = Train.to_routine (base ()) IDX.empty l1_update in
      let _ctx_y, r2 = Train.to_routine (base ()) IDX.empty (Train.grad_update l2) in
      p "the two roots' routines share an id" (r1.Context.routine_id = r2.Context.routine_id);
      let msg = refusal "foreign root" (fun () -> Context.run ctx_x r2) in
      p "another root's owner with the same id lends nothing"
        (unpropagated msg (grad_of h) && has msg "accumulates into gradients"));
  case "legitimate compositions" (fun () ->
      let pw = TDSL.param ~values:[| 1.; 2. |] "lf_two_w" ~output_dims:[ 2 ] () in
      let%op a = pw *. 2. in
      let%op b = pw *. 3. in
      p "a non-root parameter child generates no requirement"
        (Set.is_empty (Option.value_exn ~here:[%here] b.diff).backprop.requires);
      let ctx = Train.update_once (Context.auto ()) a in
      (* No value claim: [b]'s construction found [pw] no longer a backprop root and skipped its
         zeroing too, so [pw.grad] accumulates both losses' contributions -- the zeroing-order
         hazard the checks do not address. *)
      ignore
        (accepted "two losses over one parameter, one lineage" (fun () -> Train.update_once ctx b));
      let pv = TDSL.param ~values:[| 1.; 2. |] "lf_acc_w" ~output_dims:[ 2 ] () in
      let%op loss = pv *. pv in
      let ctx = Train.init_params (Context.auto ()) IDX.empty loss in
      let ctx, _ = Train.to_routine ctx IDX.empty (Train.zero_params_grads loss) in
      ignore
        (accepted "zero_params_grads beside grad_update ~accum_steps" (fun () ->
             Train.to_routine ctx IDX.empty (Train.grad_update ~accum_steps:2 loss)));
      let pm = TDSL.param ~values:[| 1.; 2. |] "lf_sgd_w" ~output_dims:[ 2 ] () in
      let%op loss = pm *. 3. in
      let ctx = Train.init_params (Context.auto ()) IDX.empty loss in
      let ctx, step =
        Train.to_routine ctx IDX.empty
          (Asgns.sequence
             [
               Train.grad_update loss;
               Train.sgd_update ~learning_rate:(TDSL.O.( !. ) 0.5) ~momentum:0.5 loss;
             ])
      in
      let ctx = Context.run ctx step in
      let ctx = Context.run ctx step in
      (* momentum m = 3, then 0.5 * 3 + 3 = 4.5; w = 1 - 0.5 * 3 - 0.5 * 4.5. *)
      exact "momentum SGD steps twice" (Context.get_values ctx pm.value) [| -2.75; -1.75 |];
      ignore
        (accepted "init_params ~reinit_all after training" (fun () ->
             Train.init_params ~reinit_all:true ctx IDX.empty loss)));
  case "metadata propagation" (fun () ->
      let w = leaf "lf_md_w" in
      let%op t = w *. w in
      let%op u = t + w in
      let fwd = u.forward.owned in
      p "forward owns its own and its embedded children's fragments, repeated operands once"
        (Set.equal fwd
           (Set.of_list
              (module Asgns.Fragment)
              [ (u.value, Asgns.Fwd); (t.value, Asgns.Fwd); (w.value, Asgns.Fwd) ]));
      let bwd = (Option.value_exn ~here:[%here] u.diff).backprop in
      p "backprop owns its own and its embedded children's fragments"
        (Set.mem bwd.owned (grad_of u, Asgns.Bprop) && Set.mem bwd.owned (grad_of t, Asgns.Bprop));
      let x = leaf "lf_md_x" in
      let%op y = x *. 2. in
      let acc = NTDSL.init ~l:"lf_md_acc" ~prec:Ir.Ops.single ~o:[ 3 ] ~f:(fun _ -> 0.) () in
      let embedding : Asgns.comp = [%cd acc =+ y] in
      p "a %cd embedding carries the operand's forward fragment"
        (Set.mem embedding.owned (y.value, Asgns.Fwd));
      p "a %cd assignment's own code owns nothing beyond its embedded operands' fragments"
        (Set.for_all embedding.owned ~f:(fun f ->
             Set.mem y.forward.owned f || Set.mem acc.forward.owned f)));
  (* The term exemption (decided on ahrefs/ocannl#1724, "exempt all terms"): a tensor built by
     [Tensor.term] -- a [Require_grad] leaf as much as a parameter -- has a constant-empty
     [grad_asn], so it owns no backprop fragment and a later consumer that finds it no longer a
     backprop root requires nothing of it: a second loss over a shared leaf compiles alone. *)
  case "term exemption" (fun () ->
      let v = leaf "lf_leaf_v" in
      Train.set_materialized v.value;
      Train.set_materialized (grad_of v);
      let%op a = v *. 2. in
      let%op b = v *. 3. in
      p "a term owns no backprop fragment"
        (Set.is_empty (Option.value_exn ~here:[%here] v.diff).backprop.owned);
      p "a consumer of a term that is no longer a backprop root requires nothing of it"
        (Set.is_empty (Option.value_exn ~here:[%here] b.diff).backprop.requires);
      (* [a]'s forward computes [v]; [v.grad] is zeroed by a routine of its own (zeroing owns
         nothing); [a]'s backprop is never compiled. *)
      let ctx = Train.forward_once (Context.auto ()) a in
      let v_zero =
        let z = (Option.value_exn ~here:[%here] v.diff).zero_grads in
        { z with asgns = Asgns.Block_comment ("lf_leaf_v_zero", z.asgns) }
      in
      let ctx, r = Train.to_routine ctx IDX.empty v_zero in
      let ctx = Context.run ctx r in
      Option.iter
        (accepted
           "a second loss over a shared Require_grad leaf trains without the first's backprop"
           (fun () -> Train.update_once ctx b))
        ~f:(fun ctx ->
          exact "the leaf's gradient is the second loss's"
            (Context.get_values ctx (grad_of v))
            [| 3.; 3.; 3. |]))

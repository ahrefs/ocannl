(* gh-ocannl-1743: a differentiable leaf that is not a parameter ([~grad_spec:Require_grad]: an
   optimized input, an adversarial perturbation, a saliency target) is served by the training
   helpers wherever being a parameter is not what the helper depends on. Being a parameter is a
   property of initialization (who sets the value); a gradient's life cycle -- accumulation and its
   per-cycle reset -- keys on the leaf, and the optimizer's DEFAULT stepped set stays the weights,
   with [?params] selecting an input. (The host read of such a leaf's gradient is pinned in
   [observable_grads], leg 4.)

   Every value here is dyadic and every operation pointwise (no device reduction), so the expected
   values are exact by construction; each is still claimed two-sided.

   1. Input optimization: [sgd_update ~params:{x}] (with the explicit-set checksum and clipping in
   the same routine) steps [x] along a host oracle for two steps -- the second step's gradient is
   taken at the updated [x], so a parameter initialization or a stale host literal touching [x]
   would show -- while the weight [w] is left alone; the checkpoint state set derives [x] from the
   routine that updates it.

   2. Accumulation: [grad_update ~accum_steps:2] over two micro-steps on DISTINCT data accumulates
   the input's gradient (the mean of the micro-step gradients), not only the weight's; the nonlinear
   intermediate is still zeroed every micro-step; a [Require_grad] leaf detached behind
   [stop_gradient] is neither required nor zeroed; and the DEFAULT [zero_params_grads] resets the
   input's gradient too, so the second cycle starts clean.

   3. A fetch-defined leaf (gh-ocannl-1776): [sgd_update ~params:{x}] refuses a [Tensor.term_init]
   leaf below [limit_constant_fill_size], naming it and [Operation.init] as the fix. The trap it
   guards is shown executed by bypassing the refusal through [sgd_one]: the second step lands where
   the first did, because the forward re-fetched [x]. The data-backed twin, same values, same loss,
   is accepted and steps along the host oracle.

   (A loss with no parameters at all -- the default stepped set empty, the refusal naming [?params],
   the default reset still selecting the leaf -- is pinned in test/training/loop_utils.ml, beside
   the gh-ocannl-670 refusals.) *)

open Base
open Stdio
open Ocannl
open Operation.DSL_modules
module IDX = Train.IDX
module Asgns = Ir.Assignments

let grad_of p = (Option.value_exn ~here:[%here] p.Tensor.diff).Tensor.grad
let values ctx tn = Context.get_values ctx tn
let close name got want = Verdict.p_close name got want ~tol:(fun _ -> 1e-6)

(* A data-backed leaf ([init_data]): its value is set once, not by forward code, so an optimizer can
   step it -- a [term_init] leaf below the constant-fill threshold is re-established by every
   forward run instead. *)
let data_leaf ~l ?(grad_spec = Tensor.Require_grad) values =
  Operation.init ~l ~prec:Ir.Ops.single
    ~o:[ Array.length values ]
    ~f:(function [| i |] -> values.(i) | _ -> assert false)
    ~grad_spec ()

let show label got =
  printf "%s:" label;
  Array.iter got ~f:(fun v -> printf " %.4f" v);
  printf "\n"

let input_optimization () =
  printf "=== 1. Input optimization ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let x = data_leaf ~l:"xin" [| 1.; 2. |] in
  (* Per cell: w * x^2, so dl/dx = 2 w x and dl/dw = x^2. *)
  let%op l = { w = [ 2.; 3. ] } *. x *. x in
  let w = Set.min_elt_exn l.Tensor.params in
  Verdict.p "x is not a parameter" (not (Set.mem l.Tensor.params x));
  Verdict.p "the default stepped set is the weight alone"
    (Set.equal (Train.trainable_params l) (Set.singleton (module Tensor) w));
  let xs = Set.singleton (module Tensor) x in
  let update = Train.grad_update l in
  let checksum, checksum_comp = Train.grad_checksum ~params:xs l in
  let clip = Train.clip_by_global_norm ~params:xs ~max_norm:100. l in
  let%op learning_rate = 0.125 in
  Train.every_non_literal_materialized learning_rate;
  let sgd = Train.sgd_update ~learning_rate ~params:xs l in
  let ctx = Train.init_params ctx IDX.empty l in
  let ctx, step =
    Context.compile ctx
      (Asgns.sequence [ update; checksum_comp; clip.Train.clip_comp; sgd ])
      IDX.empty
  in
  let ctx = Context.run ctx step in
  (* Step 1 at x = (1, 2): g = 2 w x = (4, 12); x <- x - g/8 = (0.5, 0.5). *)
  close "step 1: checksum over x's gradient = 4 + 12" (values ctx checksum.Tensor.value) [| 16. |];
  let norm = values ctx clip.Train.grad_norm.Tensor.value in
  eprintf "step 1 grad norm over x: %.9f (not part of the golden)\n" norm.(0);
  close "step 1: global norm over x's gradient = sqrt 160" norm [| Float.sqrt 160. |];
  close "step 1: x moved by lr * 2 w x" (values ctx x.Tensor.value) [| 0.5; 0.5 |];
  show "x after step 1" (values ctx x.Tensor.value);
  let ctx = Context.run ctx step in
  (* Step 2 at x = (0.5, 0.5): g = (2, 3); x <- (0.25, 0.125). *)
  close "step 2: x moved by the gradient taken at the updated x" (values ctx x.Tensor.value)
    [| 0.25; 0.125 |];
  show "x after step 2" (values ctx x.Tensor.value);
  close "the weight was not stepped" (values ctx w.Tensor.value) [| 2.; 3. |];
  let state = Checkpoint.training_state ~routines:[ step ] l in
  Verdict.p "the checkpoint state set holds x, carried by the routine that steps it"
    (Set.mem state x.Tensor.value)

let accumulation () =
  printf "=== 2. Accumulation ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let x = data_leaf ~l:"xacc" [| 1.; 2. |] in
  let z = data_leaf ~l:"zdet" [| 7.; 7. |] in
  let d = data_leaf ~l:"data" ~grad_spec:Tensor.Prohibit_grad [| 1.; 1. |] in
  Train.set_materialized d.Tensor.value;
  (* h = x d is an intermediate the loss is nonlinear in: dl/dh = 2 w h must be re-zeroed every
     micro-step. dl/dx = 2 w x d^2, dl/dw = h^2; [z] is detached. *)
  let%op h = x *. d in
  let%op l = ({ wacc = [ 2.; 3. ] } *. h *. h) + stop_gradient z in
  let w = Set.min_elt_exn l.Tensor.params in
  let micro = Train.grad_update ~accum_steps:2 l in
  let reset = Train.zero_params_grads l in
  let ctx = Train.init_params ctx IDX.empty l in
  (* The reset is compiled first: it is what runs first, at the start of each cycle. *)
  let ctx, reset = Context.compile ctx reset IDX.empty in
  let ctx, micro = Context.compile ctx micro IDX.empty in
  let set_data ctx v = Context.set_values ctx d.Tensor.value v in
  (* Cycle 1: micro-steps at d = (1, 1) then d = (2, 0.5). dl/dx: (4, 12) then (16, 3), mean (10,
     7.5); dl/dw: (1, 4) then (4, 1), mean (2.5, 2.5). *)
  let ctx = Context.run ctx reset in
  let ctx = Context.run (set_data ctx [| 1.; 1. |]) micro in
  let ctx = Context.run (set_data ctx [| 2.; 0.5 |]) micro in
  show "cycle 1 x gradient" (values ctx (grad_of x));
  close "cycle 1: x's gradient is the mean of its micro-step gradients"
    (values ctx (grad_of x))
    [| 10.; 7.5 |];
  close "cycle 1: w's gradient is the mean of its micro-step gradients"
    (values ctx (grad_of w))
    [| 2.5; 2.5 |];
  let ctx = Context.run ctx reset in
  close "the default reset zeroes x's gradient" (values ctx (grad_of x)) [| 0.; 0. |];
  close "the default reset zeroes w's gradient" (values ctx (grad_of w)) [| 0.; 0. |];
  (* Cycle 2: both micro-steps at d = (1, 1); a stale cycle-1 gradient would add (10, 7.5). *)
  let ctx = Context.run (set_data ctx [| 1.; 1. |]) micro in
  let ctx = Context.run ctx micro in
  close "cycle 2: x's gradient starts clean" (values ctx (grad_of x)) [| 4.; 12. |];
  close "cycle 2: w's gradient starts clean" (values ctx (grad_of w)) [| 1.; 4. |]

let fresh_learning_rate () =
  let%op learning_rate = 0.125 in
  Train.every_non_literal_materialized learning_rate;
  learning_rate

(* Compiles [grad_update l] followed by [step] into one routine and runs it twice, returning [x]
   after each run. *)
let two_steps l x step =
  let ctx = Train.init_params (Context.auto ()) IDX.empty l in
  let ctx, routine = Context.compile ctx (Asgns.sequence [ Train.grad_update l; step ]) IDX.empty in
  let ctx = Context.run ctx routine in
  let first = values ctx x.Tensor.value in
  let ctx = Context.run ctx routine in
  (first, values ctx x.Tensor.value)

let fetch_defined_leaf () =
  printf "=== 3. Fetch-defined leaf ===\n";
  Tensor.unsafe_reinitialize ();
  let x =
    Tensor.term_init [| 1.; 2. |] ~label:[ "xfetch" ] ~output_dims:[ 2 ]
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op l = { wfetch = [ 2.; 3. ] } *. x *. x in
  let learning_rate = fresh_learning_rate () in
  (match Train.sgd_update ~learning_rate ~params:(Set.singleton (module Tensor) x) l with
  | _ -> Verdict.p "sgd_update ~params:{x} refuses the fetch-defined leaf" false
  | exception Tensor.Session_error (msg, culprit) ->
      eprintf "refusal: %s (not part of the golden)\n" msg;
      Verdict.p "sgd_update ~params:{x} refuses the fetch-defined leaf" true;
      Verdict.p "the refusal names the leaf" (String.is_substring msg ~substring:"xfetch");
      Verdict.p "the refusal names Operation.init as the fix"
        (String.is_substring msg ~substring:"Operation.init");
      Verdict.p "the refusal carries the leaf"
        (Option.exists culprit ~f:(fun c -> Ir.Tnode.equal c.Tensor.value x.Tensor.value)));
  (* The trap, executed: [sgd_one] is the per-tensor step [sgd_update] maps, and it sees no loss to
     check against. Step 1 at x = (1, 2) gives (0.5, 0.5) as in section 1; the data-backed leaf
     would then reach (0.25, 0.125), but the forward re-fetches x = (1, 2) first. *)
  let first, second = two_steps l x (Train.sgd_one ~learning_rate x) in
  close "bypassed: step 1 moves x by lr * 2 w x" first [| 0.5; 0.5 |];
  close "bypassed: step 2 lands where step 1 did -- the forward re-fetched x" second [| 0.5; 0.5 |];
  (* The negative control: the data-backed twin. *)
  let xd = data_leaf ~l:"xdata" [| 1.; 2. |] in
  let%op ld = { wdata = [ 2.; 3. ] } *. xd *. xd in
  let learning_rate = fresh_learning_rate () in
  match Train.sgd_update ~learning_rate ~params:(Set.singleton (module Tensor) xd) ld with
  | exception Tensor.Session_error (msg, _) ->
      Verdict.p ("the data-backed twin is accepted (refused: " ^ msg ^ ")") false
  | step ->
      Verdict.p "the data-backed twin is accepted" true;
      let first, second = two_steps ld xd step in
      close "data-backed: step 1 moves x by lr * 2 w x" first [| 0.5; 0.5 |];
      close "data-backed: step 2 moves x by the gradient taken at the updated x" second
        [| 0.25; 0.125 |]

let () =
  input_optimization ();
  accumulation ();
  fetch_defined_leaf ()

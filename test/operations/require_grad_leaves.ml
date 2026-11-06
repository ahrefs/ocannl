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
   leaf below [limit_constant_fill_size], naming it and [Operation.init] as the fix. Bypassing that
   refusal no longer reaches the trap (gh-ocannl-1876): [Context.compile] refuses a routine writing
   a node a forward fetch in its lineage re-establishes -- the per-tensor [sgd_one] in the forward's
   own routine, a hand-written [%cd] step compiled after the forward's routine or before it, and a
   write through a slice of a fetch-defined parent. Writes [Context.compile] accepts: the
   data-backed twin, same values, same loss, steps along the host oracle; and a literal-initialized
   WEIGHT steps along its oracle beside the fetch-defined [x] (a parameter's initializer runs once,
   and the forward that re-fetches [x] writes nothing into it).

   A view in [?params] reads by the compile check's rule, [Assignments.forward_fetched]: over a
   fetch-defined parent [sgd_update] refuses it; an alias over data is accepted and its steps stay
   in the selected row of the parent, along the host oracle; a view lowering to a copy, which
   [sgd_update] cannot yet tell, is refused by [Context.compile], whatever its parent.

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

(* A fresh fetch-defined leaf [x] and the loss over it, per leg: [Train.grad_update] consumes the
   loss's forward code, so no two compiles share one. *)
let fetch_problem () =
  let x =
    Tensor.term_init [| 1.; 2. |] ~label:[ "xfetch" ] ~output_dims:[ 2 ]
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op l = { wfetch = [ 2.; 3. ] } *. x *. x in
  (x, l)

(* [compile] raises the refusal of gh-ocannl-1876, naming [node] and [Operation.init] as the fix. *)
let refused leg ?(node = "xfetch") ?(names = []) compile =
  match compile () with
  | () -> Verdict.p (leg ^ ": refused") false
  | exception Utils.User_error msg ->
      eprintf "%s: %s (not part of the golden)\n" leg msg;
      Verdict.p (leg ^ ": refused") true;
      List.iter (node :: "Operation.init" :: "gh-ocannl-1876" :: names) ~f:(fun substring ->
          Verdict.p (leg ^ ": the refusal names " ^ substring) (String.is_substring msg ~substring))

let fetch_defined_leaf () =
  printf "=== 3. Fetch-defined leaf ===\n";
  Tensor.unsafe_reinitialize ();
  let x, l = fetch_problem () in
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
  (* The bypass: [sgd_one] is the per-tensor step [sgd_update] maps, and it sees no loss to check
     against. Before gh-ocannl-1876 this compiled, and its second step landed where the first did
     (0.5, 0.5) -- the forward re-fetched x = (1, 2) first. *)
  refused "sgd_one in the forward's routine" (fun () ->
      ignore (two_steps l x (Train.sgd_one ~learning_rate x) : float array * float array));
  (* A hand-written step, compiled after the routine carrying x's forward fetch... *)
  let x, l = fetch_problem () in
  let ctx = Train.init_params (Context.auto ()) IDX.empty l in
  let ctx, _ = Context.compile ~name:"fwd_first" ctx (Train.grad_update l) IDX.empty in
  let%cd step = x =- x.grad in
  refused "a %cd step after the forward's routine" ~names:[ "fwd_first" ] (fun () ->
      ignore (Context.compile ~name:"xstep" ctx step IDX.empty : Context.t * Context.routine));
  (* ... and before it: the forward's routine is refused, naming the step's. *)
  let x, l = fetch_problem () in
  let ctx = Train.init_params (Context.auto ()) IDX.empty l in
  (* The step reads x and its gradient, which no routine has defined yet: seeded from the host. *)
  let ctx = Context.set_values ctx x.Tensor.value [| 1.; 2. |] in
  let ctx = Context.set_values ctx (grad_of x) [| 0.; 0. |] in
  let%cd step = x =- x.grad in
  let update = Train.grad_update l in
  let stepped, _ = Context.compile ~name:"step_first" ctx step IDX.empty in
  refused "the forward's routine after a %cd step" ~names:[ "step_first" ] (fun () ->
      ignore (Context.compile stepped update IDX.empty : Context.t * Context.routine));
  (* A sibling is an alternative: the same forward's routine, compiled from the context that does
     not descend from the step, is accepted. *)
  (match Context.compile ~name:"sibling" ctx update IDX.empty with
  | _ -> Verdict.p "the forward's routine, as a sibling of the step, is accepted" true
  | exception Utils.User_error msg ->
      Verdict.p ("the forward's routine, as a sibling of the step, is accepted: " ^ msg) false);
  (* A view takes after its parent: over a fetch-defined one the write is undone whether the slice
     lowers to an alias (the parent is fetched again) or a copy (it is copied again).
     [test_slice_alias] pins the accepted write-through, over a data-backed parent. *)
  let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let parent =
    Tensor.term_init [| 1.; 2.; 3.; 4. |] ~label:[ "pfetch" ] ~batch_dims:[ 2 ] ~output_dims:[ 2 ]
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let%op view = parent @| batch_n in
  let%op doubled = view *. 2 in
  let ctx, _ = Train.to_routine (Context.auto ()) bindings (Train.forward doubled) in
  let%cd through_view = view =: !.99.0 in
  refused "a write through a view of a fetch-defined parent" ~node:"view" (fun () ->
      ignore (Train.to_routine ctx bindings through_view : Context.t * Context.routine));
  (* The same reading in [sgd_update], where a view is selected through [?params]. Over a
     fetch-defined parent, refused early, naming the view. *)
  let batch_n, _ = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let parent =
    Tensor.term_init [| 1.; 2.; 3.; 4. |] ~label:[ "pfetchstep" ] ~batch_dims:[ 2 ]
      ~output_dims:[ 2 ] ~grad_spec:Tensor.Require_grad ()
  in
  let%op fetched_view = parent @| batch_n in
  let%op l = fetched_view *. fetched_view in
  let learning_rate = fresh_learning_rate () in
  (match Train.sgd_update ~learning_rate ~params:(Set.singleton (module Tensor) fetched_view) l with
  | _ -> Verdict.p "sgd_update refuses a view of a fetch-defined parent" false
  | exception Tensor.Session_error (msg, _) ->
      Verdict.p "sgd_update refuses a view of a fetch-defined parent" true;
      Verdict.p "that refusal names the view" (String.is_substring msg ~substring:"fetched_view"));
  (* Over data the step is real: per cell dl/dv = 2 v, so the selected row goes v <- 3/4 v, from the
     stepped value, and the other row stays. *)
  let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let parent =
    Operation.init ~l:"pdata" ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 2 ]
      ~f:(function [| i; j |] -> Float.of_int ((2 * i) + j + 1) | _ -> assert false)
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op data_view = parent @| batch_n in
  let%op l = data_view *. data_view in
  let learning_rate = fresh_learning_rate () in
  (match Train.sgd_update ~learning_rate ~params:(Set.singleton (module Tensor) data_view) l with
  | exception Tensor.Session_error (msg, _) ->
      Verdict.p ("sgd_update accepts a view of data (refused: " ^ msg ^ ")") false
  | step ->
      Verdict.p "sgd_update accepts a view of data" true;
      let ctx, routine =
        Train.to_routine (Context.auto ()) bindings (Asgns.sequence [ Train.grad_update l; step ])
      in
      Verdict.p "the stepped view is an alias" (Ir.Tnode.is_alias data_view.Tensor.value);
      IDX.find_exn routine.Context.bindings batch_n := 0;
      Train.run ctx routine;
      close "view of data: step 1 reaches the selected row of the parent"
        (values ctx parent.Tensor.value) [| 0.75; 1.5; 3.; 4. |];
      Train.run ctx routine;
      close "view of data: step 2 moves it from the stepped value" (values ctx parent.Tensor.value)
        [| 0.5625; 1.125; 3.; 4. |]);
  (* A padded view of data is a copy: the forward re-establishes it from the parent, and the step
     never reaches the parent. [sgd_update] runs before shapes settle and cannot tell; the compile
     can. *)
  let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let images =
    Operation.init ~l:"imgs" ~prec:Ir.Ops.single ~b:[ 2; 1 ] ~o:[ 3; 3; 1 ]
      ~f:(fun _ -> 1.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op copied_view = images @| batch_n in
  let processed =
    Nn_blocks.conv2d ~label:[ "processed" ] ~use_padding:true ~out_channels:1 () copied_view
  in
  let learning_rate = fresh_learning_rate () in
  let step =
    Train.sgd_update ~learning_rate ~params:(Set.singleton (module Tensor) copied_view) processed
  in
  let update = Train.grad_update processed in
  let ctx = Train.init_params (Context.auto ()) bindings processed in
  refused "a step on a view lowering to a copy, over data" ~node:"copied_view" (fun () ->
      ignore
        (Train.to_routine ctx bindings (Asgns.sequence [ update; step ])
          : Context.t * Context.routine));
  Verdict.p "that view cannot alias its parent"
    (not (Asgns.slice_alias_eligible ~array:copied_view.Tensor.value ~sliced:images.Tensor.value));
  (* Accepted: the weight, literal-initialized, stepped beside the fetch-defined x. Per cell dl/dw =
     x^2 = (1, 4) at the re-fetched x every step: w <- (2, 3) - (1, 4)/8 = (1.875, 2.5), then (1.75,
     2). *)
  let _, l = fetch_problem () in
  let w = Set.min_elt_exn l.Tensor.params in
  let learning_rate = fresh_learning_rate () in
  let first, second = two_steps l w (Train.sgd_update ~learning_rate l) in
  close "the literal-initialized weight: step 1 moves w by lr * x^2" first [| 1.875; 2.5 |];
  close "the literal-initialized weight: step 2 moves w again" second [| 1.75; 2. |];
  (* The data-backed twin. *)
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

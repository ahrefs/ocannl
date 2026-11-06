(* gh-ocannl-1804: a gradient's zeroing is carried by its first-constructed consumer
   (gh-ocannl-1744) that can still run. A consumer dropped and reclaimed with its backprop
   unconsumed cannot, so the leaves it absorbed, transitively, are backprop roots again and the
   next-constructed consumer carries their zeroing. The acceptance case itself is
   [lacks_grad_reset_hint]'s; this test pins what bounds the rule: only actual absorption gives
   back, only a dead root in the map does (a held copy, a compiled consumer or handed-out code keeps
   the ownership), tensors built earlier are not retrofitted, and the session's restores neither
   lose a leaf nor resurrect a handed-out one.

   Reclamation is the GC's to time, so every builder here drops its graph in an [@inline never]
   function ([Gc_witness]) and a case observes it before going on. What a consumer dropped but not
   yet collected does is the held case's outcome, by construction: its entry is alive. Values are
   small integers compared bitwise, each leaf's distinct, so a gradient cannot pass as another's. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module LC = Lineage_claims
module IDX = Train.IDX
module Asgns = Ir.Assignments

let values = [| 1.; 2.; 3. |]
let two_w = [| 2.; 4.; 6. |]
let param ?(values = values) name = TDSL.param ~values name ~output_dims:[ 3 ] ()
let compile ctx comp = Train.to_routine ctx IDX.empty comp
let is_leaf_root = Tensor.is_bprop_root
let hold (t : Tensor.t) = ignore (Stdlib.Sys.opaque_identity t : Tensor.t)
let zeroes (update : Asgns.comp) w = Set.mem update.zeroes (LC.grad_of w)

(* The next-constructed loss over a leaf carries its zeroing, and computes. *)
let trains name ctx w =
  let%op l = w *. w in
  let ctx = Train.update_once ctx l in
  LC.exact (name ^ ": w.grad = 2 w") (Context.get_values ctx (LC.grad_of w)) two_w

let[@inline never] forward_only ctx w =
  let%op c = w *. 2. in
  (Train.forward_once ctx c, Gc_witness.witness c)

(* [u] absorbs a computed [h], which absorbed both leaves. *)
let[@inline never] deep_consumer ctx w1 w2 =
  let%op h = w1 *. w2 in
  let%op u = h *. 2. in
  (Train.forward_once ctx u, Gc_witness.witness u, Gc_witness.witness h)

let[@inline never] unevaluated w =
  let%op c = w *. 2. in
  Gc_witness.witness c

(* [m] is built while the first consumer is alive. *)
let[@inline never] sibling_of_live_consumer ctx w =
  let%op c = w *. 2. in
  let ctx = Train.forward_once ctx c in
  let%op m = w *. 3. in
  (ctx, m, Gc_witness.witness c)

let[@inline never] copy_of_consumer ctx w =
  let%op c = w *. 2. in
  let ctx = Train.forward_once ctx c in
  let copy = { c with Tensor.children = c.Tensor.children } in
  (ctx, copy)

(* The update is compiled and the routine run: the code owning the zeroing outlives the tensor. *)
let[@inline never] compiled_consumer w =
  let%op c = w *. 2. in
  let ctx = Train.init_params (Context.auto ()) IDX.empty c in
  let ctx, step = compile ctx (Train.grad_update c) in
  (Context.run ctx step, step, Gc_witness.witness c)

(* The code is handed out and not kept here: a comp a caller keeps pins its tensor through the
   projections, so the dead-owner case a rewind must see is code that went on to a routine. The
   session's record of the handout is all that is left of it. *)
let[@inline never] handed_out_consumer w =
  let%op u = w *. 2. in
  let ck = Tensor.checkpoint ~label:"rc_ho" () in
  ignore (Tensor.consume_backprop_code u : Asgns.comp);
  (ck, Gc_witness.witness u)

let[@inline never] checkpointed_consumer label w =
  let%op u = w *. 2. in
  let ck = Tensor.checkpoint ~label () in
  (ck, Gc_witness.witness u)

(* Nothing of it stays: the leaf goes with its consumer. *)
let[@inline never] unheld_graph () =
  let w = param "rc_un_w" in
  let%op u = w *. 2. in
  let ck = Tensor.checkpoint ~label:"rc_un" () in
  (ck, Gc_witness.witness w, Gc_witness.witness u)

let[@inline never] a_closure () =
  let x = param "rc_cl_x" in
  let%op y = x *. 2. in
  ignore (Train.forward_once (Context.auto ()) y : Context.t)

let () =
  case "a deep consumer gives back every leaf it absorbed, and no computed tensor" (fun () ->
      let w1 = param "rc_d1" and w2 = param ~values:[| 5.; 7.; 11. |] "rc_d2" in
      let ctx, u, h = deep_consumer (Context.auto ()) w1 w2 in
      p "deep: the consumer and its child are reclaimed"
        (Gc_witness.reclaimed u && Gc_witness.reclaimed h);
      p "deep: both leaves are roots again" (is_leaf_root w1 && is_leaf_root w2);
      let%op l = (w1 *. w1) + (w2 *. w2) in
      let ctx = Train.update_once ctx l in
      LC.exact "deep: w1.grad = 2 w1" (Context.get_values ctx (LC.grad_of w1)) two_w;
      LC.exact "deep: w2.grad = 2 w2" (Context.get_values ctx (LC.grad_of w2)) [| 10.; 14.; 22. |]);
  case "a consumer dropped unevaluated gives its leaf back" (fun () ->
      let w = param "rc_ue" in
      let c = unevaluated w in
      p "unevaluated: the consumer is reclaimed" (Gc_witness.reclaimed c);
      trains "unevaluated" (Context.auto ()) w);
  case "the hand-back passes the zeroing on once per generation" (fun () ->
      let w = param "rc_gn" in
      let ctx, c1 = forward_only (Context.auto ()) w in
      p "generations: the first consumer is reclaimed" (Gc_witness.reclaimed c1);
      let ctx, c2 = forward_only ctx w in
      p "generations: the second consumer is reclaimed" (Gc_witness.reclaimed c2);
      trains "generations" ctx w);
  case "a tensor built before the hand-back is not retrofitted" (fun () ->
      let w = param "rc_sb" in
      let ctx, m, c = sibling_of_live_consumer (Context.auto ()) w in
      p "sibling: the first consumer is reclaimed" (Gc_witness.reclaimed c);
      p "sibling: its leaf is a root again" (is_leaf_root w);
      let msg = LC.refusal "sibling" (fun () -> Train.update_once ctx m) in
      p "sibling: the loss built while the consumer lived lacks the reset, and is told why"
        (LC.has msg "without zeroing it, and no routine of its lineage zeroes it"
        && LC.has msg "gh-ocannl-1804" && LC.has msg "Gc.full_major ()"
        && LC.has msg "Train.grad_update ~accum_steps:1");
      let%op l = w *. w in
      let%op k = w *. 5. in
      let update = Train.grad_update l in
      p "sibling: the next-constructed loss zeroes the gradient" (zeroes update w);
      p "sibling: the one after it does not: the first-constructed rule stands"
        (not (zeroes (Train.grad_update k) w));
      let ctx, step = compile ctx update in
      let ctx = Context.run ctx step in
      LC.exact "sibling: the next-constructed loss computes w.grad = 2 w"
        (Context.get_values ctx (LC.grad_of w))
        two_w);
  case "only what the entry absorbed comes back" (fun () ->
      let w = param "rc_no" in
      let%op owner = w *. 2. in
      let ctx = Train.forward_once (Context.auto ()) owner in
      let ctx, s = forward_only ctx w in
      p "non-owner: the sibling, which absorbed nothing, is reclaimed" (Gc_witness.reclaimed s);
      p "non-owner: the live owner keeps the leaf" (not (is_leaf_root w));
      hold owner;
      ignore (Stdlib.Sys.opaque_identity ctx : Context.t));
  case "an ownership-preserving copy keeps the leaf" (fun () ->
      let w = param "rc_cp" in
      let ctx, copy = copy_of_consumer (Context.auto ()) w in
      Gc_witness.settle ();
      p "copy: the consumer's entry lives on the copy, so the leaf stays absorbed"
        (Tensor.is_bprop_root copy && not (is_leaf_root w));
      hold copy;
      ignore (Stdlib.Sys.opaque_identity ctx : Context.t));
  case "a compiled consumer gives nothing back" (fun () ->
      let w = param "rc_cm" in
      let ctx, step, c = compiled_consumer w in
      p "compiled: the consumer is reclaimed" (Gc_witness.reclaimed c);
      p "compiled: its routine still owns the zeroing" (not (is_leaf_root w));
      let ctx = Context.run ctx step in
      LC.exact "compiled: the routine still runs: w.grad = 2"
        (Context.get_values ctx (LC.grad_of w))
        [| 2.; 2.; 2. |]);
  case "handed-out code gives nothing back, through a rewind either" (fun () ->
      let w = param "rc_ho" in
      let ck, u = handed_out_consumer w in
      p "handout: the consumer is reclaimed" (Gc_witness.reclaimed u);
      Tensor.rewind ck;
      p "handout: the rewind leaves the leaf with the code handed out" (not (is_leaf_root w)));
  case "a rewind to a checkpoint holding the dead entry gives the leaf back" (fun () ->
      let w = param "rc_rw" in
      let ck, u = checkpointed_consumer "rc_rw" w in
      p "rewind: the consumer is reclaimed" (Gc_witness.reclaimed u);
      let%op l = w *. w in
      hold l;
      p "rewind: the construction since took the leaf" (not (is_leaf_root w));
      Tensor.rewind ck;
      p "rewind: the leaf is a root again" (is_leaf_root w);
      trains "rewind" (Context.auto ()) w);
  case "a scoped restore keeps the hand-back, across a closure" (fun () ->
      let w = param "rc_sc" in
      let ctx, c = forward_only (Context.auto ()) w in
      Tensor.with_unchanged_roots ~f:(fun () ->
          p "scope: the consumer is reclaimed" (Gc_witness.reclaimed c);
          a_closure ();
          p "scope: the leaf is a root inside" (is_leaf_root w);
          let%op l = w *. w in
          hold l;
          p "scope: a construction took it" (not (is_leaf_root w)));
      p "scope: the restore leaves it a root" (is_leaf_root w);
      trains "scope" ctx w);
  case "the hand-back keeps nothing alive" (fun () ->
      let ck, w, u = unheld_graph () in
      p "retention: the consumer is reclaimed under a checkpoint" (Gc_witness.reclaimed u);
      p "retention: so is the leaf it would have given back" (Gc_witness.reclaimed w);
      ignore (Stdlib.Sys.opaque_identity ck : Tensor.checkpoint))

(* gh-ocannl-483: the online-softmax attention rewrite ([Ir.Online_softmax]), the first algebraic
   rewrite over lowered code -- a pattern-directed substitution that changes the computation where
   the schedule transforms only rearrange it.

   The composed attention lowers its softmax to a max-reduction and a sum-reduction over the key
   axis, and reads the probabilities once per value-width iteration of the final reduction; the
   scores and the probabilities are [seq^2]-shaped and both get materialized. The rewrite turns the
   reduction pair into one [Scan_loop] per row carrying the running max and the rescaled running sum
   (the online-softmax recurrence, gh-ocannl-696's founding use case), and hoists the probability
   read out of the loops it does not index, so the probability chain inlines into that one read and
   is never stored. The normalizer's summation is reassociated -- results move within rounding,
   which is why the pass sits behind the numerics-changing config key [online_softmax] -- and the
   hoist is exact.

   Every executed leg compares the rewritten model against the SAME model composed (AGENTS.md: a
   value-rewriting pass needs executed parity, not only structural pins). "The same model" is
   literal: the session is reinitialized before each build, so the two builds mint the same tensor
   ids and draw the same parameter initializations. Device floats stay off the golden: the claims
   are two-sided tolerance comparisons, and the exact digits go to stderr.

   Legs: 1. the gate -- the key off leaves the composed form, with no scan and the probabilities
   written; 2. causal attention with the head width under the recompute cap: parity, one scan, and
   NO [seq^2]-sized node written at all; 3. the head width above the cap: parity, and the scores are
   the one [seq^2] buffer left -- the cap, not the rewrite, decides it, pinned by raising the cap
   and watching that buffer go too; 4. a padding mask that leaves whole prefixes of keys masked (the
   [-inf] scores the recurrence has to survive): parity and finiteness; 6. training -- the rewritten
   forward under the composed backward, which reads the forward's intermediates through
   cross-routine splicing: parameter gradients agree -- last, since it runs the composed backward's
   own routine; 5. two stacked blocks: two scans. *)

open Base
open Stdio
module Train = Ocannl.Train
module Nn_blocks = Ocannl.Nn_blocks
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Online_softmax = Ir.Online_softmax

let batch = 2

(* Distinct from every other extent in the models, so a node with two axes of this extent is one of
   the attention's [seq, seq] intermediates and nothing else. *)
let seq = 7
let d_model = 16
let heads = 2

(* A mask over (query [s], key [t]) positions: causal, or causal with the first [prefix] keys masked
   for every query at or past [prefix] (so no row is fully masked, and rows from [prefix] on start
   with masked scores). *)
let mask ~prefix =
  NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
    ~f:(function
      | [| s; t |] -> if s >= t && (t >= prefix || s < prefix) then 1. else 0. | _ -> assert false)
    ()

(* [layers] attention blocks over a ramp input, residually stacked; the output keeps the model width
   through the residual. *)
let model ~layers ~d_k ~prefix () =
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ batch; seq ] ~input_dims:[]
      ~output_dims:[ d_model ] ()
  in
  let mask = mask ~prefix in
  let blocks =
    List.init layers ~f:(fun i ->
        Nn_blocks.multi_head_attention
          ~label:[ "attn" ^ Int.to_string i ]
          ~num_heads:heads ~d_k ~d_v:d_k ())
  in
  List.fold blocks ~init:x ~f:(fun x block ->
      let%op y = x + block ~train_step:None ~mask x in
      y)

(* The optimized lowering of [t]'s forward code, re-lowered in a fresh lineage after the run (the
   gh-343 recipe: [forward_once] first assembles the forward and settles memory modes). *)
let inspect (t : Tensor.t) : LL.optimized =
  Ir.Assignments.lower (LL.empty_optimize_ctx ()) ~unoptim_ll_source:None ~ll_source:None
    ~cd_source:None ~name:"probe" [] t.Tensor.forward.Ir.Assignments.asgns

let is_scan = function LL.Scan_loop _ -> true | _ -> false
let scans (o : LL.optimized) = Ll_test.count_stmt ~f:is_scan o.LL.llc

(* The nodes the optimized code writes that have two axes of extent [seq]: the attention's [seq,
   seq] intermediates (scores, probabilities) and nothing else in these models. *)
let square_buffers (o : LL.optimized) =
  Set.filter (LL.writes_of_stmt o.LL.llc) ~f:(fun tn ->
      Array.count (Lazy.force tn.Tn.dims) ~f:(fun d -> d = seq) >= 2)

type run = { values : float array; optimized : LL.optimized; raw : LL.t }

(* The raw lowering of [t]'s forward code, ahead of the rewrite tier. *)
let raw (t : Tensor.t) : LL.t = Ir.Assignments.to_low_level t.Tensor.forward.Ir.Assignments.asgns

(* Build and run the forward of [model ()] with the rewrite [on] or off. *)
let forward ~on ~layers ~d_k ~prefix =
  Tensor.unsafe_reinitialize ();
  Online_softmax.set_enabled (Some on);
  let t = model ~layers ~d_k ~prefix () in
  let ctx = Train.forward_once (Context.auto ()) t in
  let values = Context.get_values ctx t.Tensor.value in
  let optimized = inspect t and raw = raw t in
  Online_softmax.set_enabled None;
  { values; optimized; raw }

let close ~tol g w = Float.(abs (g -. w) <= tol *. max 1. (abs w))

let report label (got : float array) =
  eprintf "%s: %s (not part of the golden)\n%!" label
    (String.concat ~sep:" "
       (Array.to_list (Array.map (Array.sub got ~pos:0 ~len:8) ~f:(Printf.sprintf "%.9g"))))

let () =
  eprintf "backend: %s (not part of the golden)\n%!"
    (Utils.get_global_arg ~arg_name:"backend" ~default:"cc");
  printf "--- leg 1: the gate -- key off keeps the composed form ---\n";
  let composed = forward ~on:false ~layers:1 ~d_k:8 ~prefix:0 in
  p "composed: no scan in the routine" (scans composed.optimized = 0);
  p "composed: the routine writes seq^2-sized intermediates"
    (not (Set.is_empty (square_buffers composed.optimized)));
  p_all "composed: every output is finite" (Array.to_list composed.values) ~f:Float.is_finite;

  printf "--- leg 2: causal attention, head width under the recompute cap ---\n";
  let fused = forward ~on:true ~layers:1 ~d_k:8 ~prefix:0 in
  report "composed" composed.values;
  report "rewritten" fused.values;
  p "rewritten: exactly one scan, the attention's online normalizer" (scans fused.optimized = 1);
  (* The tier's member contract ([Ir.Rewrites]): the pass changes the raw lowering and is idempotent
     on its own output, which is what lets the tier run its members to a fixpoint. *)
  let once = Online_softmax.rewrite fused.raw in
  p "the pass changes the raw lowering" (not (LL.equal once fused.raw));
  p "the pass is idempotent on its own output" (LL.equal (Online_softmax.rewrite once) once);
  p "rewritten: no seq^2-sized node is written at all"
    (Set.is_empty (square_buffers fused.optimized));
  p_all2 "rewritten output matches the composed output within 1e-5 relative" fused.values
    composed.values ~f:(close ~tol:1e-5);

  printf "--- leg 3: head width above the recompute cap -- the cap owns the scores ---\n";
  let d_k = 32 in
  let composed = forward ~on:false ~layers:1 ~d_k ~prefix:0 in
  let fused = forward ~on:true ~layers:1 ~d_k ~prefix:0 in
  let n_fused = Set.length (square_buffers fused.optimized) in
  printf "seq^2 buffers: composed %d, rewritten %d\n"
    (Set.length (square_buffers composed.optimized))
    n_fused;
  p "rewritten: the scores are the one seq^2 buffer left (the probabilities are gone)"
    (n_fused = 1 && Set.length (square_buffers composed.optimized) > 1);
  p_all2 "rewritten output matches the composed output within 1e-5 relative" fused.values
    composed.values ~f:(close ~tol:1e-5);
  let cap = LL.virtualize_settings.LL.max_inline_reduction in
  LL.virtualize_settings.LL.max_inline_reduction <- d_k;
  let recomputed = forward ~on:true ~layers:1 ~d_k ~prefix:0 in
  LL.virtualize_settings.LL.max_inline_reduction <- cap;
  p "a cap admitting the head width recomputes the scores: no seq^2 buffer at all"
    (Set.is_empty (square_buffers recomputed.optimized));
  p_all2 "recomputed scores give the same output within 1e-5 relative" recomputed.values
    composed.values ~f:(close ~tol:1e-5);

  printf "--- leg 4: masked key prefixes -- the recurrence survives -inf scores ---\n";
  let composed = forward ~on:false ~layers:1 ~d_k:8 ~prefix:3 in
  let fused = forward ~on:true ~layers:1 ~d_k:8 ~prefix:3 in
  report "composed" composed.values;
  report "rewritten" fused.values;
  p_all "rewritten: every output is finite" (Array.to_list fused.values) ~f:Float.is_finite;
  p_all2 "rewritten output matches the composed output within 1e-5 relative" fused.values
    composed.values ~f:(close ~tol:1e-5);

  printf "--- leg 5: two stacked blocks -- one scan each ---\n";
  let composed = forward ~on:false ~layers:2 ~d_k:8 ~prefix:0 in
  let fused = forward ~on:true ~layers:2 ~d_k:8 ~prefix:0 in
  p "rewritten: two scans" (scans fused.optimized = 2);
  p "rewritten: no seq^2-sized node is written" (Set.is_empty (square_buffers fused.optimized));
  p_all2 "rewritten output matches the composed output within 1e-5 relative" fused.values
    composed.values ~f:(close ~tol:1e-5)

(* --- Leg 6: training -- the rewritten forward under the composed backward. --- *)

let train ~on =
  Tensor.unsafe_reinitialize ();
  Online_softmax.set_enabled (Some on);
  let y = model ~layers:1 ~d_k:8 ~prefix:2 () in
  let%op loss = (y *. y) ++ "... | ... => 0" in
  let params =
    Set.to_list y.Tensor.params
    |> List.sort ~compare:(fun a b -> Int.compare a.Tensor.value.Tn.id b.Tensor.value.Tn.id)
  in
  List.iter params ~f:(fun p -> Train.set_materialized (Option.value_exn p.Tensor.diff).Tensor.grad);
  let ctx = Train.update_once (Context.auto ()) loss in
  let grads =
    List.map params ~f:(fun p ->
        ( Tn.debug_name p.Tensor.value,
          Context.get_values ctx (Option.value_exn p.Tensor.diff).Tensor.grad ))
  in
  let loss_value = (Context.get_values ctx loss.Tensor.value).(0) in
  Online_softmax.set_enabled None;
  (loss_value, grads)

let () =
  printf "--- leg 6: training -- parameter gradients agree with the composed forward ---\n";
  let loss_c, grads_c = train ~on:false in
  let loss_f, grads_f = train ~on:true in
  eprintf "loss: composed %.9g rewritten %.9g (not part of the golden)\n%!" loss_c loss_f;
  p "the loss agrees within 1e-5 relative" (close ~tol:1e-5 loss_f loss_c);
  printf "parameters with gradients: %d\n" (List.length grads_f);
  p "the same parameters carry gradients in both runs"
    (List.equal String.equal (List.map grads_f ~f:fst) (List.map grads_c ~f:fst));
  List.iter2_exn grads_f grads_c ~f:(fun (name, gf) (_, gc) ->
      p_all2 (name ^ ".grad agrees within 1e-4 relative") gf gc ~f:(close ~tol:1e-4);
      p (name ^ ".grad is not identically zero") (Array.exists gc ~f:(fun v -> Float.(v <> 0.))))

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
   own routine; 5. two stacked blocks: two scans; 7. what the recognizer DECLINES, on hand-built raw
   lowerings the pipeline never emits but the exposed [rewrite] accepts: a pointwise definition
   outside the max..sum span, a read of the normalizer between its zeroing and the max -- each left
   untouched -- and the carried state taking each node's own precision. *)

open Base
open Stdio
module Train = Ocannl.Train
module Nn_blocks = Ocannl.Nn_blocks
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Online_softmax = Ir.Online_softmax

(* The dune rule [runtest-online_softmax_fast_math] reruns this executable on cc under
   [cc_backend_fast_math], which the [approximate] profile turns on beside the rewrite: every claim
   below holds there too, since the flag takes [-ffinite-math-only] back (its golden differs from
   this run's in the regime line alone). *)
let fast_math = Utils.get_global_flag ~default:false ~arg_name:"cc_backend_fast_math"
let backend_name = Utils.get_global_arg ~arg_name:"backend" ~default:"cc"
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
   through the residual. The ramp is scaled into [0, 1): the raw 0..223 ramp gives scores in the
   thousands, a one-hot softmax, and parameter gradients through the scores of order 1e-19 -- a
   parity claim on them is vacuous, and under a C compiler's fast-math licence the composed
   backward's cancellation ([dP / l + dl], two terms of order 1e5 whose difference is the gradient)
   leaves rounding garbage of order 10 where the exact value is 0. A non-saturated softmax makes
   every parity claim below a claim about the values. *)
let model ?mask_fill ~layers ~d_k ~prefix () =
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ batch; seq ] ~input_dims:[]
      ~output_dims:[ d_model ] ()
  in
  let%op x = x /. 224. in
  let mask = mask ~prefix in
  let blocks =
    List.init layers ~f:(fun i ->
        Nn_blocks.multi_head_attention
          ~label:[ "attn" ^ Int.to_string i ]
          ~num_heads:heads ~d_k ~d_v:d_k ?mask_fill ())
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
let scans_of (llc : LL.t) = Ll_test.count_stmt ~f:is_scan llc
let scans (o : LL.optimized) = scans_of o.LL.llc

(* The nodes the optimized code writes that have two axes of extent [seq]: the attention's [seq,
   seq] intermediates (scores, probabilities) and nothing else in these models. *)
let square_buffers ?(extent = seq) (o : LL.optimized) =
  LL.affine_accesses o.LL.llc
  |> List.filter_map ~f:(fun (a : Tn.t Ir.Affine.access) -> Option.some_if a.a_write a.a_tn)
  |> Set.of_list (module Tn)
  |> Set.filter ~f:(fun tn -> Array.count (Lazy.force tn.Tn.dims) ~f:(fun d -> d = extent) >= 2)

type run = { values : float array; optimized : LL.optimized; raw : LL.t }

(* The raw lowering of [t]'s forward code, ahead of the rewrite tier. *)
let raw (t : Tensor.t) : LL.t = Ir.Assignments.to_low_level t.Tensor.forward.Ir.Assignments.asgns

(* Build and run the forward of [model ()] with the rewrite [on] or off. *)
let forward ?mask_fill ~on ~layers ~d_k ~prefix () =
  Tensor.unsafe_reinitialize ();
  Online_softmax.set_enabled (Some on);
  let t = model ?mask_fill ~layers ~d_k ~prefix () in
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
  eprintf "backend: %s (not part of the golden)\n%!" backend_name;
  printf "--- run under cc_backend_fast_math=%b ---\n" fast_math;
  printf "--- leg 1: the gate -- key off keeps the composed form ---\n";
  let composed = forward ~on:false ~layers:1 ~d_k:8 ~prefix:0 () in
  p "composed: no scan in the routine" (scans composed.optimized = 0);
  p "composed: the routine writes seq^2-sized intermediates"
    (not (Set.is_empty (square_buffers composed.optimized)));
  p_all "composed: every output is finite" (Array.to_list composed.values) ~f:Float.is_finite;

  printf "--- leg 2: causal attention, head width under the recompute cap ---\n";
  let fused = forward ~on:true ~layers:1 ~d_k:8 ~prefix:0 () in
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
  let composed = forward ~on:false ~layers:1 ~d_k ~prefix:0 () in
  let fused = forward ~on:true ~layers:1 ~d_k ~prefix:0 () in
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
  let recomputed = forward ~on:true ~layers:1 ~d_k ~prefix:0 () in
  LL.virtualize_settings.LL.max_inline_reduction <- cap;
  p "a cap admitting the head width recomputes the scores: no seq^2 buffer at all"
    (Set.is_empty (square_buffers recomputed.optimized));
  p_all2 "recomputed scores give the same output within 1e-5 relative" recomputed.values
    composed.values ~f:(close ~tol:1e-5);

  printf "--- leg 4: masked key prefixes -- the recurrence survives -inf scores ---\n";
  let composed = forward ~on:false ~layers:1 ~d_k:8 ~prefix:3 () in
  let fused = forward ~on:true ~layers:1 ~d_k:8 ~prefix:3 () in
  report "composed" composed.values;
  report "rewritten" fused.values;
  p_all "rewritten: every output is finite" (Array.to_list fused.values) ~f:Float.is_finite;
  p_all2 "rewritten output matches the composed output within 1e-5 relative" fused.values
    composed.values ~f:(close ~tol:1e-5);

  printf "--- leg 5: two stacked blocks -- one scan each ---\n";
  let composed = forward ~on:false ~layers:2 ~d_k:8 ~prefix:0 () in
  let fused = forward ~on:true ~layers:2 ~d_k:8 ~prefix:0 () in
  p "rewritten: two scans" (scans fused.optimized = 2);
  p "rewritten: no seq^2-sized node is written" (Set.is_empty (square_buffers fused.optimized));
  p_all2 "rewritten output matches the composed output within 1e-5 relative" fused.values
    composed.values ~f:(close ~tol:1e-5)

(* --- Leg 6: training -- the rewritten forward under the composed backward. --- *)

let train ~on =
  Tensor.unsafe_reinitialize ();
  Online_softmax.set_enabled (Some on);
  (* The composed backward: the fused one (legs 9 to 11) is a gate of its own. *)
  Online_softmax.set_backward_enabled (Some false);
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
  Online_softmax.set_backward_enabled None;
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
      let worst =
        Array.fold2_exn gf gc ~init:0. ~f:(fun acc a b ->
            Float.max acc (Float.abs (a -. b) /. Float.max 1. (Float.abs b)))
      in
      let scale = Array.fold gc ~init:0. ~f:(fun acc b -> Float.max acc (Float.abs b)) in
      eprintf "%s.grad: worst relative difference %.3g at scale %.3g (not part of the golden)\n%!"
        name worst scale;
      p_all2 (name ^ ".grad agrees within 1e-4 relative") gf gc ~f:(close ~tol:1e-4);
      p (name ^ ".grad is not identically zero") (Array.exists gc ~f:(fun v -> Float.(v <> 0.))))

(* --- Leg 7: what the recognizer declines, on hand-built raw lowerings. --- *)

let () =
  printf "--- leg 7: the recognizer declines what it cannot prove, on hand-built lowerings ---\n";
  let module B = Ll_test in
  let n = 5 in
  (* One row of the composed softmax -- the four nests and the two initializations -- in the
     statement order [order], with the normalizer node at [l_prec]; [`Read_l] is a bystander
     statement reading the normalizer into an output. *)
  let build ?(l_prec = Ir.Ops.single) ?(l_dims = [| 1 |]) ?(m_prec = Ir.Ops.single)
      ?(x_prec = Ir.Ops.single) ?(e_prec = Ir.Ops.single) ?(n_prec = Ir.Ops.single) order =
    let mk = B.node_factory ~first_id:48300 ~dims:[| n |] () in
    let mk1 = B.node_factory ~first_id:48400 ~dims:[| 1 |] () in
    let mkl = B.node_factory ~prec:l_prec ~first_id:48500 ~dims:l_dims () in
    let mkm = B.node_factory ~prec:m_prec ~first_id:48800 ~dims:[| 1 |] () in
    let mkx = B.node_factory ~prec:x_prec ~first_id:48900 ~dims:[| n |] () in
    let mke = B.node_factory ~prec:e_prec ~first_id:49000 ~dims:[| n |] () in
    let mkn = B.node_factory ~prec:n_prec ~first_id:49100 ~dims:[| n |] () in
    let x = mkx "x" and nn = mkn "n" and e = mke "e" in
    (* An auxiliary chain off the same max that never reaches a sum. *)
    let n2 = mk "n2" and e2 = mk "e2" in
    List.iter [ n2; e2 ] ~f:B.materialize;
    let m = mkm "m" and y = mk1 "y" and l = mkl "l" in
    List.iter [ x; nn; e; m; y; l ] ~f:B.materialize;
    let v = mk1 "v" in
    B.virtualize v;
    (* A value reduction fed through a long elementwise chain from the probabilities. *)
    let width = 3 in
    let pp = mk "p" and ws = List.init 10 ~f:(fun k -> mk ("w" ^ Int.to_string k)) in
    let vals = B.node_factory ~first_id:48600 ~dims:[| n; width |] () "vals" in
    let out = B.node_factory ~first_id:48700 ~dims:[| width |] () "out" in
    (* A target whose cell is [t + j]: distinct (t, j) pairs collide. *)
    let out2 = B.node_factory ~first_id:48701 ~dims:[| n + width - 1 |] () "out2" in
    List.iter ((pp :: ws) @ [ vals; out; out2 ]) ~f:B.materialize;
    let op o args = LL.apply_op o args in
    let cell tn = B.get tn [| B.fixed 0 |] in
    let stmt = function
      | `A_init -> B.set_at m (B.fixed 0) (B.c Float.neg_infinity)
      | `A ->
          let t = B.sym () in
          B.loop_n t n
            (B.set_at m (B.fixed 0)
               (op (Ir.Ops.Binop Ir.Ops.Max) [| cell m; B.get x [| B.iter t |] |]))
      | `N ->
          let t = B.sym () in
          B.loop_n t n
            (B.set_at nn (B.iter t)
               (op (Ir.Ops.Binop Ir.Ops.Sub) [| B.get x [| B.iter t |]; cell m |]))
      | `E ->
          let t = B.sym () in
          B.loop_n t n
            (B.set_at e (B.iter t) (op (Ir.Ops.Unop Ir.Ops.Exp) [| B.get nn [| B.iter t |] |]))
      | `C_init -> B.zero l
      | `C ->
          let t = B.sym () in
          B.loop_n t n
            (B.set_at l (B.fixed 0)
               (op (Ir.Ops.Binop Ir.Ops.Add) [| cell l; B.get e [| B.iter t |] |]))
      | `Read_l -> B.set_at y (B.fixed 0) (cell l)
      | `C_init_cell -> B.set_at l (B.fixed 0) (B.c 0.)
      | `N_aux ->
          let t = B.sym () in
          B.loop_n t n
            (B.set_at n2 (B.iter t)
               (op (Ir.Ops.Binop Ir.Ops.Sub) [| B.get x [| B.iter t |]; cell m |]))
      | `E_aux ->
          let t = B.sym () in
          B.loop_n t n
            (B.set_at e2 (B.iter t) (op (Ir.Ops.Unop Ir.Ops.Exp) [| B.get n2 [| B.iter t |] |]))
      | `D ->
          let t = B.sym () in
          B.loop_n t n
            (B.set_at pp (B.iter t)
               (op (Ir.Ops.Binop Ir.Ops.Div) [| B.get e [| B.iter t |]; cell l |]))
      | `Chain ->
          LL.unflat_lines
            (List.mapi ws ~f:(fun k w ->
                 let src = if k = 0 then pp else List.nth_exn ws (k - 1) in
                 let t = B.sym () in
                 B.loop_n t n
                   (B.set_at w (B.iter t)
                      (op (Ir.Ops.Binop Ir.Ops.Add) [| B.get src [| B.iter t |]; B.c 0. |]))))
      | `PV ->
          let t = B.sym () and j = B.sym () in
          let w = List.last_exn ws in
          LL.unflat_lines
            [
              B.zero out;
              B.loop_n t n
                (B.loop_n j width
                   (B.set out
                      [| B.iter j |]
                      (op (Ir.Ops.Binop Ir.Ops.Add)
                         [|
                           B.get out [| B.iter j |];
                           op (Ir.Ops.Binop Ir.Ops.Mul)
                             [| B.get w [| B.iter t |]; B.get vals [| B.iter t; B.iter j |] |];
                         |])));
            ]
      | `PV_affine ->
          let t = B.sym () and j = B.sym () in
          let w = List.last_exn ws in
          let cell = [| B.aff [ (1, t); (1, j) ] 0 |] in
          LL.unflat_lines
            [
              B.zero out2;
              B.loop_n t n
                (B.loop_n j width
                   (B.set out2 cell
                      (op (Ir.Ops.Binop Ir.Ops.Add)
                         [|
                           B.get out2 cell;
                           op (Ir.Ops.Binop Ir.Ops.Mul)
                             [| B.get w [| B.iter t |]; B.get vals [| B.iter t; B.iter j |] |];
                         |])));
            ]
      | `PV_dead ->
          let t = B.sym () and j = B.sym () in
          let w = List.last_exn ws in
          LL.unflat_lines
            [
              B.zero out;
              B.loop_n t n
                (B.loop ~upto:(-1) j
                   (B.set out
                      [| B.iter j |]
                      (op (Ir.Ops.Binop Ir.Ops.Add)
                         [|
                           B.get out [| B.iter j |];
                           op (Ir.Ops.Binop Ir.Ops.Mul)
                             [| B.get w [| B.iter t |]; B.get vals [| B.iter t; B.iter j |] |];
                         |])));
            ]
      | `Opaque -> LL.Staged_compilation (fun () -> PPrint.empty)
      | `Tile ->
          B.tile_mma ~m:1 ~n:1 ~k:1
            ~d:(y, [| B.fixed 0 |])
            ~a:(x, [| B.fixed 0 |])
            ~b:(x, [| B.fixed 0 |])
            LL.Noop
      | `Scope_pure ->
          (* A scope whose body reads nothing of the chain and writes only its own local. *)
          let id = LL.get_scope v in
          B.set_at y (B.fixed 0)
            (LL.Local_scope
               {
                 id;
                 orig_indices = [||];
                 mint = LL.Inlined_computation;
                 body = LL.Set_local (id, B.c 7.);
               })
      | `Scope_read_l ->
          (* A scope whose body reads the normalizer. *)
          let id = LL.get_scope v in
          B.set_at y (B.fixed 0)
            (LL.Local_scope
               {
                 id;
                 orig_indices = [||];
                 mint = LL.Inlined_computation;
                 body = LL.Set_local (id, cell l);
               })
      | `Scope_write ->
          (* A scope whose body writes a tensor: impure by the optimizer's contract, but the tier
             runs ahead of that check -- the census has to see the write inside the body. *)
          B.set_at y (B.fixed 0)
            (LL.Local_scope
               {
                 id = LL.get_scope v;
                 orig_indices = [||];
                 mint = LL.Inlined_computation;
                 body = B.set_at x (B.fixed 0) (B.c 5.);
               })
      | `Opaque_cond ->
          (* Staged code reachable only through a guard's condition, inside a scope body. *)
          let scope =
            LL.Local_scope
              {
                id = LL.get_scope v;
                orig_indices = [||];
                mint = LL.Inlined_computation;
                body = LL.Staged_compilation (fun () -> PPrint.empty);
              }
          in
          LL.If { cond = (scope, Ir.Ops.single); body = LL.Noop }
    in
    (LL.unflat_lines (List.map order ~f:stmt), x, l)
  in
  (* A chain at one precision throughout. *)
  let uniform prec = (prec, prec, prec, prec, prec) in
  let build_at (x_prec, m_prec, n_prec, e_prec, l_prec) ?l_dims order =
    build ~x_prec ~m_prec ~n_prec ~e_prec ~l_prec ?l_dims order
  in
  let raw ?l_prec ?l_dims ?m_prec ?x_prec ?e_prec ?n_prec order =
    let llc, _, _ = build ?l_prec ?l_dims ?m_prec ?x_prec ?e_prec ?n_prec order in
    llc
  in
  let rewritten ?l_prec ?l_dims ?m_prec ?x_prec ?e_prec ?n_prec order =
    Online_softmax.rewrite (raw ?l_prec ?l_dims ?m_prec ?x_prec ?e_prec ?n_prec order)
  in
  let declined ?l_prec ?l_dims ?m_prec ?x_prec ?e_prec ?n_prec order =
    let raw = raw ?l_prec ?l_dims ?m_prec ?x_prec ?e_prec ?n_prec order in
    LL.equal (Online_softmax.rewrite raw) raw
  in
  let composed = [ `A_init; `A; `N; `E; `C_init; `C ] in
  p "the composed order is rewritten into one scan" (scans_of (rewritten composed) = 1);
  p "the subtraction ahead of the max is declined" (declined [ `N; `A_init; `A; `E; `C_init; `C ]);
  p "the exponential after the sum is declined" (declined [ `A_init; `A; `N; `C_init; `C; `E ]);
  p "a read of the normalizer between its zeroing and the max is declined"
    (declined [ `C_init; `Read_l; `A_init; `A; `N; `E; `C ]);
  p "a read of the normalizer between the max and the sum is declined"
    (declined [ `C_init; `A_init; `A; `Read_l; `N; `E; `C ]);
  p "the zeroing ahead of the max with nothing reading the normalizer in between is accepted"
    (scans_of (rewritten [ `C_init; `A_init; `A; `N; `E; `C ]) = 1);
  p "staged code inside the span the rewrite reorders is declined"
    (declined [ `A_init; `A; `Opaque; `N; `E; `C_init; `C ]);
  p "a Tile_mma inside the span is opaque even when its scalar fallback is empty"
    (declined [ `A_init; `A; `Tile; `N; `E; `C_init; `C ]);
  p "staged code outside that span is no objection"
    (scans_of (rewritten [ `Opaque; `A_init; `A; `N; `E; `C_init; `C ]) = 1);
  p "staged code reachable only through a guard's condition inside the span is declined too"
    (declined [ `A_init; `A; `N; `Opaque_cond; `E; `C_init; `C ]);
  p "a scope body writing the scores inside the span is in the census and declines the rewrite"
    (declined [ `A_init; `A; `Scope_write; `N; `E; `C_init; `C ]);
  p "a scope body reading the normalizer between the max and the sum is declined"
    (declined [ `C_init; `A_init; `A; `Scope_read_l; `N; `E; `C ]);
  p "a scope body inside the span touching nothing of the chain is no objection: the census sees it"
    (scans_of (rewritten [ `A_init; `A; `Scope_pure; `N; `E; `C_init; `C ]) = 1);
  p "a whole-node zeroing of a normalizer wider than the reduction's cells is declined"
    (declined ~l_dims:[| 2 |] [ `A_init; `A; `N; `E; `C_init; `C ]);
  p "the same reduction with a cell-wise zeroing is accepted: the scan writes what the fill did"
    (scans_of (rewritten ~l_dims:[| 2 |] [ `A_init; `A; `N; `E; `C_init_cell; `C ]) = 1);
  (* Executed special values, on the same hand-built row: a NaN score ahead of a masked one. [max]
     drops the NaN against [-inf] and then meets a genuinely all-masked step, whose branch must not
     overwrite the poison the NaN left in the normalizer. *)
  let normalizer label scores (llc, x, l) =
    let o = B.optimize ~name:label llc in
    (List.hd_exn (B.execute ~name:label o ~seed:[ (x, scores); (l, [| B.sentinel |]) ] ~read:[ l ])).(
    0)
  in
  let poisoned = [| Float.nan; Float.neg_infinity; 0.; 1.; 2. |] in
  let masked = [| Float.neg_infinity; Float.neg_infinity; 0.; 1.; 2. |] in
  let both ?(precs = uniform Ir.Ops.single) label scores =
    let ((llc, x, l) as built) = build_at precs composed in
    ( normalizer (label ^ "_composed") scores built,
      normalizer (label ^ "_online") scores (Online_softmax.rewrite llc, x, l) )
  in
  let c_nan, o_nan = both "os_nan_seq" poisoned in
  eprintf "normalizers for [nan; -inf; 0; 1; 2]: composed %g online %g (not part of the golden)\n%!"
    c_nan o_nan;
  p "composed: a NaN score ahead of a masked one poisons the normalizer" (Float.is_nan c_nan);
  p "rewritten: the all-masked step after the NaN keeps the poison" (Float.is_nan o_nan);
  let c_fin, o_fin = both "os_masked_seq" masked in
  eprintf
    "normalizers for [-inf; -inf; 0; 1; 2]: composed %.9g online %.9g (not part of the golden)\n%!"
    c_fin o_fin;
  p "a genuinely all-masked prefix leaves both normalizers finite and equal within 1e-6 relative"
    (Float.is_finite c_fin && close ~tol:1e-6 o_fin c_fin);
  (* The same prefix at f64: the floor is the largest finite double, not Base's [Float.max_value]
     (which is infinity -- a floor of [-inf] made the masked prefix NaN). Metal has no doubles. *)
  let f64 = not (String.equal (String.lowercase backend_name) "metal") in
  let c64, o64 =
    if f64 then both ~precs:(uniform Ir.Ops.double) "os_masked_f64" masked else (0., 0.)
  in
  if f64 then
    eprintf
      "normalizers for [-inf; -inf; 0; 1; 2] at f64: composed %.17g online %.17g (not part of the \
       golden)\n\
       %!"
      c64 o64;
  gated ~when_:f64 ~on:backend_name
    "an all-masked prefix at f64 leaves both normalizers finite and equal within 1e-12 relative"
    (Float.is_finite c64 && close ~tol:1e-12 o64 c64);
  (* Executed at a narrow uniform precision: the composed form rounds every intermediate to f16 per
     step while the carried pair lives at f32 -- the one difference beyond summation order, and it
     stays within f16's own resolution. *)
  let c_h, o_h = both ~precs:(uniform Ir.Ops.half) "os_half_chain" [| 0.5; 2.; -1.; 3.; 1.5 |] in
  eprintf "normalizers for an f16 chain: composed %.9g online %.9g (not part of the golden)\n%!" c_h
    o_h;
  p "an f16 chain: the f32-carried normalizer agrees with the per-step-rounded one within 2e-3"
    (Float.is_finite c_h && close ~tol:2e-3 o_h c_h);
  let c_min, o_min = both "os_min_finite" (Array.create ~len:n (-3.4028234663852886e38)) in
  eprintf
    "normalizers for a row at the lowest finite value: composed %.9g online %.9g (not part of the \
     golden)\n\
     %!"
    c_min o_min;
  p
    "a row of scores at the format's lowest finite value is a finite maximum in both forms, with \
     the row length as normalizer"
    (Float.is_finite c_min && close ~tol:1e-6 o_min c_min && close ~tol:1e-6 c_min (Float.of_int n));
  let c_all, o_all = both "os_all_masked" (Array.create ~len:n Float.neg_infinity) in
  eprintf "normalizers for an all-masked row: composed %g online %g (not part of the golden)\n%!"
    c_all o_all;
  p "a fully masked row leaves the composed normalizer NaN" (Float.is_nan c_all);
  p "and the rewritten normalizer stores that NaN too, though its carried state stayed zero"
    (Float.is_nan o_all);
  (* The hoist follows the probabilities through elementwise definitions however many: ten nests
     between the normalizer and the value reduction, and the reduction still gets its local. *)
  let locals llc = Ll_test.count_stmt ~f:(function LL.Declare_local _ -> true | _ -> false) llc in
  p "the value reduction is hoisted through a ten-nest elementwise chain from the probabilities"
    (locals (rewritten (composed @ [ `D; `Chain; `PV ])) = 2);
  p "a reduction whose target cell is [t + j] is not hoisted: distinct pairs share a cell"
    (locals (rewritten (composed @ [ `D; `Chain; `PV_affine ])) = 1);
  p "a dead moved loop is not hoisted: the original nest never reads the probability"
    (locals (rewritten (composed @ [ `D; `Chain; `PV_dead ])) = 1);
  p "an integer normalizer node is declined: its own reduction truncated after every step"
    (declined ~l_prec:Ir.Ops.int32 composed);
  p "an auxiliary subtraction and exponential ahead of the normalizer's own do not hide it"
    (scans_of (rewritten [ `A_init; `A; `N_aux; `E_aux; `N; `E; `C_init; `C ]) = 1);
  p
    "a max node narrower than the scores is declined: the composed max rounds where the state does \
     not"
    (declined ~m_prec:Ir.Ops.half composed);
  p "a max node of the scores' range but fewer digits (bf16 under f32) is declined too"
    (declined ~m_prec:Ir.Ops.bfloat16 composed);
  p "an exponential node narrower than the scores is declined: small terms round to zero there"
    (declined ~e_prec:Ir.Ops.half composed);
  p "a normalizer wider than the chain (f64 under f32) is declined: it would sum unrounded terms"
    (declined ~l_prec:Ir.Ops.double composed);
  p "scores narrower than the chain (f16 under f32) are declined as well: one precision throughout"
    (declined ~x_prec:Ir.Ops.half composed);
  p "a chain at one precision is accepted at f16"
    (scans_of
       (Online_softmax.rewrite
          (let llc, _, _ = build_at (uniform Ir.Ops.half) composed in
           llc))
    = 1);
  let rec carried_of = function
    | LL.Scan_loop { carried; _ } -> Some carried
    | LL.Seq (a, b) -> Option.first_some (carried_of a) (carried_of b)
    | LL.For_loop { body; _ } -> carried_of body
    | _ -> None
  in
  let state_precs precs =
    let llc, _, _ = build_at precs composed in
    Option.map
      (carried_of (Online_softmax.rewrite llc))
      ~f:(fun carried ->
        List.map carried ~f:(fun (c : LL.carried) -> Lazy.force c.prev.tn.Tn.storage_prec))
  in
  p "an f64 chain carries both states at double"
    (Option.equal (List.equal Ir.Ops.equal_prec)
       (state_precs (uniform Ir.Ops.double))
       (Some [ Ir.Ops.double; Ir.Ops.double ]));
  p "an f16 chain carries both states at single: widened, never per-step at the storage width"
    (Option.equal (List.equal Ir.Ops.equal_prec)
       (state_precs (uniform Ir.Ops.half))
       (Some [ Ir.Ops.single; Ir.Ops.single ]))

(* --- Leg 8: special values, and the analysis cache across sibling lowerings. --- *)

let () =
  printf
    "--- leg 8: a NaN mask fill poisons the rewritten rows exactly where it poisons the composed ---\n";
  (* A NaN where the mask fill goes: [max] drops it against [-inf], so an online normalizer that
     guarded on [m' = -inf] alone would discard the NaN at the head of every masked prefix and leave
     those rows finite, while the composed form's [exp (nan - max)] poisons every row that has a
     masked position -- which, under the prefix mask, is every row. *)
  let composed = forward ~mask_fill:Float.nan ~on:false ~layers:1 ~d_k:8 ~prefix:3 () in
  let fused = forward ~mask_fill:Float.nan ~on:true ~layers:1 ~d_k:8 ~prefix:3 () in
  p_all "the composed output is NaN in every row: every row has a NaN-filled position"
    (Array.to_list composed.values) ~f:Float.is_nan;
  p_all2 "the rewritten output is NaN exactly where the composed output is" fused.values
    composed.values ~f:(fun f c -> Bool.equal (Float.is_nan f) (Float.is_nan c));
  printf "--- leg 8b: sibling lowerings of one rewritten program hit the analysis cache ---\n";
  (* Placement arms and autotune candidates re-lower one program; the minted locals must be the same
     nodes each time, or the identity-keyed analysis cache misses on every sibling. *)
  Tensor.unsafe_reinitialize ();
  Online_softmax.set_enabled (Some true);
  let t = model ~layers:1 ~d_k:8 ~prefix:0 () in
  let _ctx = Train.forward_once (Context.auto ()) t in
  ignore (inspect t : LL.optimized);
  let h1, m1 = LL.analysis_cache_stats () in
  ignore (inspect t : LL.optimized);
  let h2, m2 = LL.analysis_cache_stats () in
  Online_softmax.set_enabled None;
  p "re-lowering the rewritten forward is an analysis-cache hit, not a miss" (h2 = h1 + 1 && m2 = m1)

(* --- Legs 9 to 11: the fused backward (gh-ocannl-1002). ---

   With [online_softmax_backward] on as well, the training step's composed attention backward -- the
   probabilities' gradient [dP], the chain down to the score gradient [dS], each a [seq, seq] buffer
   -- becomes a per-row reduction [D = sum (dO * O)] into a minted node and three nests recomputing
   each query-key pair's probability, [dP] cell and score gradient into scope locals: one over the
   query rows accumulating the query gradient, two over the keys accumulating the key and the value
   gradients. *)

(* The fused backward's per-row [D]: the minted node the rewrite labels [bwd_rowdot]. *)
let rowdot_writes (llc : LL.t) =
  Ll_test.count_stmt llc ~f:(function
    | LL.Set { tn; _ } -> ( match tn.Tn.label with "bwd_rowdot" :: _ -> true | _ -> false)
    | _ -> false)

let set_gates ~on ~bwd =
  Online_softmax.set_enabled (Some on);
  Online_softmax.set_backward_enabled (Some bwd)

let reset_gates () =
  Online_softmax.set_enabled None;
  Online_softmax.set_backward_enabled None

type step = {
  loss : float;
  grads : (string * float array) list;
  again : (string * float array) list;  (** The same routine's gradients on a second run. *)
  optimized : LL.optimized;  (** The training routine, as compiled. *)
  interface : Set.M(Tn).t;  (** Its arguments: the nodes it requires initialized and writes out. *)
  raw_step : LL.t;  (** Its raw lowering, ahead of the rewrite tier. *)
  cache : int * int;  (** Analysis-cache hits and misses of two sibling lowerings of the step. *)
}

(* Compile and run the training step of [model] ([Train.grad_update]: forward, gradient zeroing and
   backprop in one routine) under the two gates, capturing the routine as compiled. *)
let training ?mask_fill ?(layers = 1) ?(prefix = 2) ~d_k ~on ~bwd () =
  Tensor.unsafe_reinitialize ();
  set_gates ~on ~bwd;
  let y = model ?mask_fill ~layers ~d_k ~prefix () in
  let%op loss = (y *. y) ++ "... | ... => 0" in
  let params =
    Set.to_list y.Tensor.params
    |> List.sort ~compare:(fun a b -> Int.compare a.Tensor.value.Tn.id b.Tensor.value.Tn.id)
  in
  List.iter params ~f:(fun p -> Train.set_materialized (Option.value_exn p.Tensor.diff).Tensor.grad);
  let update = Train.grad_update loss in
  let ctx = Train.init_params (Context.auto ()) Ir.Indexing.Empty loss in
  let captured = ref None in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o ->
        captured := Some o;
        [ o ])
      ctx update Ir.Indexing.Empty
  in
  let read ctx =
    List.map params ~f:(fun p ->
        ( Tn.debug_name p.Tensor.value,
          Context.get_values ctx (Option.value_exn p.Tensor.diff).Tensor.grad ))
  in
  let ctx = Context.run ctx routine in
  let grads = read ctx
  and loss = Array.fold (Context.get_values ctx loss.Tensor.value) ~init:0. ~f:( +. ) in
  let again = read (Context.run ctx routine) in
  let lower () =
    ignore
      (Ir.Assignments.lower (LL.empty_optimize_ctx ()) ~unoptim_ll_source:None ~ll_source:None
         ~cd_source:None ~name:"probe_step" [] update.Ir.Assignments.asgns
        : LL.optimized)
  in
  lower ();
  let h1, m1 = LL.analysis_cache_stats () in
  lower ();
  let h2, m2 = LL.analysis_cache_stats () in
  let raw_step = Ir.Assignments.to_low_level update.Ir.Assignments.asgns in
  reset_gates ();
  {
    loss;
    grads;
    again;
    optimized = Option.value_exn !captured;
    interface = Set.union routine.Context.inputs routine.Context.outputs;
    raw_step;
    cache = (h2 - h1, m2 - m1);
  }

let worst_relative gf gc =
  Array.fold2_exn gf gc ~init:0. ~f:(fun acc a b ->
      Float.max acc (Float.abs (a -. b) /. Float.max 1. (Float.abs b)))

let grads_agree ~tol ~what (fused : (string * float array) list) composed =
  p "the same parameters carry gradients in both runs"
    (List.equal String.equal (List.map fused ~f:fst) (List.map composed ~f:fst));
  List.iter2_exn fused composed ~f:(fun (name, gf) (_, gc) ->
      let scale = Array.fold gc ~init:0. ~f:(fun acc b -> Float.max acc (Float.abs b)) in
      eprintf
        "%s %s.grad: worst relative difference %.3g at scale %.3g (not part of the golden)\n%!" what
        name (worst_relative gf gc) scale;
      p_all2
        (Printf.sprintf "%s: %s.grad agrees within %g relative" what name tol)
        gf gc ~f:(close ~tol);
      p
        (Printf.sprintf "%s: %s.grad is not identically zero" what name)
        (Array.exists gc ~f:(fun v -> Float.(v <> 0.))))

let () =
  printf "--- leg 9: training with the fused backward ---\n";
  let d_k = 8 in
  let composed = training ~d_k ~on:false ~bwd:false () in
  let forward_only = training ~d_k ~on:true ~bwd:false () in
  let fused = training ~d_k ~on:true ~bwd:true () in
  eprintf "loss: composed %.9g fused %.9g (not part of the golden)\n%!" composed.loss fused.loss;
  p "the loss agrees within 1e-5 relative" (close ~tol:1e-5 fused.loss composed.loss);
  grads_agree ~tol:1e-5 ~what:"head width 8" fused.grads composed.grads;
  let flat grads = Array.concat (List.map grads ~f:snd) in
  p_all2
    "a second run of the fused step recomputes the same gradients: the zeroings it accumulates \
     onto are kept"
    (flat fused.again) (flat fused.grads) ~f:Float.equal;
  p "the fused step has one per-row D, the forward-only step none"
    (rowdot_writes fused.optimized.LL.llc = 1 && rowdot_writes forward_only.optimized.LL.llc = 0);
  (* The backward gate alone: the forward rewritten, the backward composed -- leg 6's shape. *)
  p "the backward gate off leaves the composed backward, whose seq^2 buffers the step writes"
    (not (Set.is_empty (square_buffers forward_only.optimized)));
  p
    "under the recompute cap the fused step writes no seq^2-sized node at all (masked key prefixes \
     included)"
    (Set.is_empty (square_buffers fused.optimized));
  let square tn = Array.count (Lazy.force tn.Tn.dims) ~f:(fun d -> d = seq) >= 2 in
  p_none "no argument of the fused routine is seq^2-sized" (Set.to_list fused.interface) ~f:square;
  (* The tier's member contract on the whole step: the pass consumes its instance, and its output
     has no normalizer left to anchor a second match. *)
  set_gates ~on:true ~bwd:true;
  let once = Online_softmax.rewrite fused.raw_step in
  let twice = Online_softmax.rewrite once in
  reset_gates ();
  p "the pass fuses the raw training step" (rowdot_writes once = 1);
  (* Each gradient is written by one fused nest with ONE channel loop after its scalars -- the shape
     the default GPU annotator's lane geometry reads (gh-ocannl-1003 stage 1) -- and dV's scalars
     hold no loop (it needs [p] alone), where dQ's and dK's hold [dp]'s reduction. *)
  let writes_to name stmt =
    Ll_test.count_stmt stmt ~f:(function
      | LL.Set { tn; _ } -> String.equal (Tn.debug_name tn) name
      | _ -> false)
    > 0
  in
  let rec preamble_then_one_loop ~loop_free = function
    | LL.For_loop { body; _ } -> (
        match
          List.rev
            (List.filter (LL.flat_lines [ body ]) ~f:(function
              | LL.Noop | LL.Comment _ -> false
              | _ -> true))
        with
        | [ single ] -> preamble_then_one_loop ~loop_free single
        | LL.For_loop _ :: preamble ->
            (not (List.is_empty preamble))
            && List.for_all preamble ~f:(function
              | LL.Declare_local _ | LL.Set_local _ -> true
              | LL.For_loop _ -> not loop_free
              | _ -> false)
        | _ -> false)
    | _ -> false
  in
  let fused_nests name =
    List.filter (LL.flat_lines [ once ]) ~f:(fun s -> writes_to name s && rowdot_writes s = 0)
  in
  List.iter
    [ ("q.grad", false); ("k.grad", false); ("v.grad", true) ]
    ~f:(fun (name, loop_free) ->
      p
        (Printf.sprintf "%s: one fused nest, one channel loop after %s scalars" name
           (if loop_free then "loop-free" else "its"))
        (match fused_nests name with
        | [ nest ] -> preamble_then_one_loop ~loop_free nest
        | _ -> false));
  p "the pass is idempotent on the fused step" (LL.equal twice once);
  eprintf "sibling lowerings: %d hits, %d misses (not part of the golden)\n%!" (fst fused.cache)
    (snd fused.cache);
  p "re-lowering the fused step is an analysis-cache hit, not a miss"
    (Poly.equal fused.cache (1, 0));
  printf
    "--- leg 9b: head width above the recompute cap -- the scores are the one seq^2 buffer ---\n";
  let d_k = 32 in
  let composed = training ~d_k ~on:false ~bwd:false () in
  let forward_only = training ~d_k ~on:true ~bwd:false () in
  let fused = training ~d_k ~on:true ~bwd:true () in
  grads_agree ~tol:1e-5 ~what:"head width 32" fused.grads composed.grads;
  let composed_squares = Set.length (square_buffers forward_only.optimized) in
  let fused_squares = square_buffers fused.optimized in
  printf "seq^2 buffers in the training step: composed backward %d, fused backward %d\n"
    composed_squares (Set.length fused_squares);
  let names s = Set.to_list s |> List.map ~f:Tn.debug_name in
  p "the fused step writes one seq^2 buffer, the scores the forward-only step also stores"
    (match names fused_squares with
    | [ scores ] ->
        List.mem (names (square_buffers forward_only.optimized)) scores ~equal:String.equal
    | _ -> false);
  let square tn = Array.count (Lazy.force tn.Tn.dims) ~f:(fun d -> d = seq) >= 2 in
  p_none "no argument of the fused routine is seq^2-sized: the stored scores are routine scratch"
    (Set.to_list fused.interface) ~f:square;
  let cap = LL.virtualize_settings.LL.max_inline_reduction in
  LL.virtualize_settings.LL.max_inline_reduction <- d_k;
  let recomputed = training ~d_k ~on:true ~bwd:true () in
  LL.virtualize_settings.LL.max_inline_reduction <- cap;
  p "a cap admitting the head width recomputes the scores too: no seq^2 buffer at all"
    (Set.is_empty (square_buffers recomputed.optimized));
  grads_agree ~tol:1e-5 ~what:"recomputed scores" recomputed.grads composed.grads;
  printf "--- leg 9c: four SGD steps in one routine -- the loss trajectories agree ---\n";
  let trajectory ~on ~bwd =
    Tensor.unsafe_reinitialize ();
    set_gates ~on ~bwd;
    let y = model ~layers:1 ~d_k:8 ~prefix:2 () in
    let%op loss = (y *. y) ++ "... | ... => 0" in
    let update = Train.grad_update loss in
    let%op learning_rate = 0.001 in
    let sgd = Train.sgd_update ~learning_rate loss in
    let ctx = Train.init_params (Context.auto ()) Ir.Indexing.Empty loss in
    let captured = ref None in
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun o ->
          captured := Some o;
          [ o ])
        ctx
        (Ir.Assignments.sequence [ update; sgd ])
        Ir.Indexing.Empty
    in
    let losses =
      List.folding_map (List.range 0 4) ~init:ctx ~f:(fun ctx _ ->
          let ctx = Context.run ctx routine in
          (ctx, Array.fold (Context.get_values ctx loss.Tensor.value) ~init:0. ~f:( +. )))
    in
    reset_gates ();
    (losses, rowdot_writes (Option.value_exn !captured).LL.llc)
  in
  let composed, _ = trajectory ~on:false ~bwd:false in
  let fused, d_nests = trajectory ~on:true ~bwd:true in
  eprintf "losses: composed %s; fused %s (not part of the golden)\n%!"
    (String.concat ~sep:" " (List.map composed ~f:(Printf.sprintf "%.9g")))
    (String.concat ~sep:" " (List.map fused ~f:(Printf.sprintf "%.9g")));
  p "the step with the SGD update is fused too" (d_nests = 1);
  p "the loss moves across the steps"
    (not (Float.equal (List.hd_exn composed) (List.last_exn composed)));
  p_all2 "every step's loss agrees within 1e-5 relative" (Array.of_list fused)
    (Array.of_list composed) ~f:(close ~tol:1e-5);
  printf "--- leg 9d: two stacked blocks -- one D each ---\n";
  let composed = training ~layers:2 ~d_k:8 ~on:false ~bwd:false () in
  let fused = training ~layers:2 ~d_k:8 ~on:true ~bwd:true () in
  p "two per-row D reductions" (rowdot_writes fused.optimized.LL.llc = 2);
  p "no seq^2-sized node is written" (Set.is_empty (square_buffers fused.optimized));
  grads_agree ~tol:1e-5 ~what:"two blocks" fused.grads composed.grads;
  printf "--- leg 9e: a NaN mask fill poisons the fused gradients exactly where the composed ---\n";
  let composed = training ~mask_fill:Float.nan ~prefix:3 ~d_k:8 ~on:false ~bwd:false () in
  let fused = training ~mask_fill:Float.nan ~prefix:3 ~d_k:8 ~on:true ~bwd:true () in
  p "the fused step fired" (rowdot_writes fused.optimized.LL.llc = 1);
  p "the composed gradients carry NaNs"
    (List.exists composed.grads ~f:(fun (_, g) -> Array.exists g ~f:Float.is_nan));
  List.iter2_exn fused.grads composed.grads ~f:(fun (name, gf) (_, gc) ->
      p_all2 (name ^ ".grad is NaN exactly where the composed one is") gf gc ~f:(fun f c ->
          Bool.equal (Float.is_nan f) (Float.is_nan c)))

(* --- Leg 10: the attention's own gradients, on leaf queries, keys and values. ---

   The parameter gradients of the model above reach the attention through the projections; here [q],
   [k] and [v] are differentiable leaves, so their gradients ARE the fused nests' dQ, dK and dV.
   Their values vary with every index (and one query row is zero: its scores tie), and the loss
   weighs the output by an equally discriminating upstream gradient -- an all-ones one can hide a
   broken softmax adjoint. *)

let leaf_seq = 5
let leaf_heads = 2
let leaf_d = 3
let leaf_e = 4

let%op composed_max_softmax x =
  (* The softmax before gh-ocannl-1002's phase 0: the max stays differentiable, so the composed
     backward carries the max-gradient nests the fused recognizer declines. *)
  let max_vals = x @^^ " ... | t -> ... => ... | 0 -> ..." in
  let exp_vals = exp (x - max_vals) in
  exp_vals /. (exp_vals ++ " ... | t -> ... => ... | 0 -> ...")

let leaf_attention ~softmax ~fill ~mask q k v =
  let%op scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ]) /. sqrt (dim d)
  in
  let%op masked = where mask scores !.fill in
  let weights = softmax masked in
  let%op o = weights +* v " ... s | t -> h; ... t | h e => ... s | h e" [ "e" ] in
  o

(* A value varying with every index, of moderate magnitude, [salt] telling the tensors apart. *)
let wave salt idcs =
  Float.sin
    (Array.foldi idcs ~init:salt ~f:(fun i acc x ->
         acc +. (Float.of_int ((i + 2) * (x + 1)) *. 0.37)))

type leaf_run = {
  lloss : float;
  lgrads : (string * float array) list;
  loptimized : LL.optimized option;  (** The training routine, when [grads]. *)
}

let fused_count run = rowdot_writes (Option.value_exn run.loptimized).LL.llc

(* The leaf model's [seq, seq] intermediates: the only nodes with two axes of [leaf_seq], which no
   other extent of it equals. *)
let leaf_squares run = square_buffers ~extent:leaf_seq (Option.value_exn run.loptimized)

(* The loss [sum (O * up)] of one leaf attention, and the gradients of [q], [k] and [v]. [live s t]
   decides the mask; [tied] derives the keys from the queries ([k = q * 1], so the queries' gradient
   sums the fused query-gradient nest's contribution and the one flowing back from the keys');
   [bump] adds [h * wave] to one of the leaves, for the finite differences. At [prec] double every
   node is double, the gradients included. *)
let leaf ?(prec = Ir.Ops.single) ?(fill = Float.neg_infinity) ?(tied = false)
    ?(softmax = fun x -> Nn_blocks.softmax ~spec:" ... | t -> ..." () x) ?bump ?(grads = true) ~live
    ~on ~bwd () =
  Tensor.unsafe_reinitialize ();
  set_gates ~on ~bwd;
  let value_prec = !Tensor.default_value_prec and grad_prec = !Tensor.default_grad_prec in
  Tensor.default_value_prec := prec;
  Tensor.default_grad_prec := prec;
  let leaf_tensor name salt ~o =
    let f idcs =
      let base = if String.equal name "q" && idcs.(0) = 2 then 0. else wave salt idcs in
      match bump with
      | Some (which, h) when String.equal which name -> base +. (h *. wave (salt +. 5.) idcs)
      | _ -> base
    in
    Ocannl.Operation.init ~l:name ~prec ~b:[ leaf_seq ] ~o ~f ~grad_spec:Tensor.Require_grad ()
  in
  let q = leaf_tensor "q" 0.1 ~o:[ leaf_heads; leaf_d ] in
  let k =
    if tied then
      let%op k = q *. !.1. in
      k
    else leaf_tensor "k" 0.2 ~o:[ leaf_heads; leaf_d ]
  in
  let v = leaf_tensor "v" 0.3 ~o:[ leaf_heads; leaf_e ] in
  let mask =
    NTDSL.init ~l:"leaf_mask" ~prec:Ir.Ops.single ~b:[ leaf_seq ] ~i:[ leaf_seq ] ~o:[]
      ~f:(function [| s; t |] -> if live s t then 1. else 0. | _ -> assert false)
      ()
  in
  let up = NTDSL.init ~l:"up" ~prec ~b:[ leaf_seq ] ~o:[ leaf_heads; leaf_e ] ~f:(wave 0.4) () in
  let o = leaf_attention ~softmax ~fill ~mask q k v in
  let%op loss = (o *. up) ++ "... | ... => 0" in
  let leaves = if tied then [ q; v ] else [ q; k; v ] in
  (* Materialized, so that a leaf is uploaded rather than inlined into the code as a small constant:
     the finite differences change its values between builds. *)
  List.iter leaves ~f:(fun t ->
      Train.set_materialized t.Tensor.value;
      Train.set_materialized (Option.value_exn t.Tensor.diff).Tensor.grad);
  let captured = ref None in
  let ctx =
    if grads then
      let update = Train.grad_update loss in
      let ctx, routine =
        Context.compile
          ~lowered_transform:(fun o ->
            captured := Some o;
            [ o ])
          (Context.auto ()) update Ir.Indexing.Empty
      in
      Context.run ctx routine
    else Train.forward_once (Context.auto ()) loss
  in
  let lgrads =
    if grads then
      List.map leaves ~f:(fun t ->
          ( Tn.debug_name t.Tensor.value,
            Context.get_values ctx (Option.value_exn t.Tensor.diff).Tensor.grad ))
    else []
  in
  (* The loss keeps the query axis; its gradient is seeded with ones, so the total is the function
     the gradients are of. *)
  let lloss = Array.fold (Context.get_values ctx loss.Tensor.value) ~init:0. ~f:( +. ) in
  reset_gates ();
  Tensor.default_value_prec := value_prec;
  Tensor.default_grad_prec := grad_prec;
  { lloss; lgrads; loptimized = !captured }

let causal_prefix s t = s >= t && (t >= 2 || s < 2)

let () =
  printf "--- leg 10: leaf queries, keys and values -- dQ, dK and dV themselves ---\n";
  let composed = leaf ~live:causal_prefix ~on:false ~bwd:false () in
  let fused = leaf ~live:causal_prefix ~on:true ~bwd:true () in
  p "the fused backward fired on the leaf attention" (fused_count fused = 1);
  p "the composed leaf step writes seq^2-sized nodes, the fused one none"
    ((not (Set.is_empty (leaf_squares composed))) && Set.is_empty (leaf_squares fused));
  grads_agree ~tol:1e-5 ~what:"leaf" fused.lgrads composed.lgrads;
  let composed = leaf ~tied:true ~live:causal_prefix ~on:false ~bwd:false () in
  let fused = leaf ~tied:true ~live:causal_prefix ~on:true ~bwd:true () in
  p "keys derived from the queries: fused" (fused_count fused = 1);
  grads_agree ~tol:1e-5 ~what:"derived keys" fused.lgrads composed.lgrads;
  printf "--- leg 10b: a finite mask fill keeps the masked probabilities and their dV ---\n";
  (* Query row 1 has no live key and the last key is live for no query: under a finite fill row 1
     attends uniformly to every key, so the last key's dV comes from that row alone, while the
     mask's own gradient rule leaves dS -- hence row 1's dQ -- exactly zero. *)
  let live s t = s <> 1 && t < leaf_seq - 1 && t <= s in
  let composed = leaf ~fill:(-1e4) ~live ~on:false ~bwd:false () in
  let fused = leaf ~fill:(-1e4) ~live ~on:true ~bwd:true () in
  p "the fused backward fired" (fused_count fused = 1);
  grads_agree ~tol:1e-5 ~what:"finite fill" fused.lgrads composed.lgrads;
  let grad run name = List.Assoc.find_exn run.lgrads ~equal:String.equal name in
  let dv_last run =
    Array.filteri (grad run "v") ~f:(fun i _ -> i / (leaf_heads * leaf_e) = leaf_seq - 1)
  in
  let dq_row1 run = Array.filteri (grad run "q") ~f:(fun i _ -> i / (leaf_heads * leaf_d) = 1) in
  p_exists "the key every query masks still gets a dV under the finite fill: fused"
    (Array.to_list (dv_last fused))
    ~f:(fun g -> Float.(abs g > 1e-3));
  p_exists "and composed" (Array.to_list (dv_last composed)) ~f:(fun g -> Float.(abs g > 1e-3));
  p_all "the fully masked query row's dQ is exactly zero: fused"
    (Array.to_list (dq_row1 fused))
    ~f:(fun g -> Float.equal g 0.);
  p_all "and composed" (Array.to_list (dq_row1 composed)) ~f:(fun g -> Float.equal g 0.);
  printf "--- leg 10c: f64 finite differences of the fused dQ, dK and dV ---\n";
  (* Directional derivatives: the fused gradient against the wave direction [bump] adds, over a mask
     that is fixed (the loss is smooth in q, k and v away from none of its boundaries). *)
  let f64 = not (String.equal (String.lowercase backend_name) "metal") in
  let names = [ ("q", 0.1); ("k", 0.2); ("v", 0.3) ] in
  let fired, matches =
    if not f64 then (false, List.map names ~f:(fun _ -> false))
    else
      let prec = Ir.Ops.double in
      let fused = leaf ~prec ~live:causal_prefix ~on:true ~bwd:true () in
      let h = 1e-5 in
      ( fused_count fused = 1,
        List.map names ~f:(fun (name, salt) ->
            let at h' =
              (leaf ~prec ~bump:(name, h') ~grads:false ~live:causal_prefix ~on:true ~bwd:true ())
                .lloss
            in
            let fd = (at h -. at (-.h)) /. (2. *. h) in
            let g = List.Assoc.find_exn fused.lgrads ~equal:String.equal name in
            let dir =
              Array.init (Array.length g) ~f:(fun i ->
                  let per = Array.length g / leaf_seq in
                  let o = if String.equal name "v" then leaf_e else leaf_d in
                  wave (salt +. 5.) [| i / per; i % per / o; i % o |])
            in
            let analytic = Array.fold2_exn g dir ~init:0. ~f:(fun acc a b -> acc +. (a *. b)) in
            eprintf
              "d%s along the probe: finite difference %.12g, fused %.12g (not part of the golden)\n\
               %!"
              name fd analytic;
            Float.(abs analytic > 1e-3) && close ~tol:1e-6 analytic fd) )
  in
  (* Metal has no double precision. *)
  gated ~when_:f64 ~on:backend_name "the fused backward fired at f64" fired;
  List.iter2_exn names matches ~f:(fun (name, _) ok ->
      gated ~when_:f64 ~on:backend_name
        (Printf.sprintf "d%s: the fused gradient matches the f64 finite difference within 1e-6" name)
        ok)

(* --- Leg 11: what the fused backward declines. ---

   On the raw lowering of the leg-9 training step, edited the way a model variant or another flow
   would change it: each edit must leave the backward composed (no per-row D) while the forward
   rewrite still fires, and an edit outside the pattern's span must not. *)

let () =
  printf "--- leg 11: the fused backward declines what it cannot prove ---\n";
  let base = training ~d_k:8 ~on:true ~bwd:true () in
  let stmts = LL.flat_lines [ base.raw_step ] in
  let rec target = function
    | LL.For_loop { body; _ } -> target body
    | LL.Set { tn; _ } -> Some tn
    | _ -> None
  in
  let position name =
    fst
      (Option.value_exn
         (List.findi stmts ~f:(fun _ s ->
              Option.exists (target s) ~f:(fun tn -> String.equal (Tn.debug_name tn) name))))
  in
  let insert_after pos extra =
    List.concat_mapi stmts ~f:(fun i s -> if i = pos then [ s; extra ] else [ s ])
  in
  let rewritten llc =
    set_gates ~on:true ~bwd:true;
    let r = Online_softmax.rewrite llc in
    reset_gates ();
    r
  in
  let fused llc = rowdot_writes (rewritten llc) = 1 && scans_of (rewritten llc) = 1 in
  let declined llc = rowdot_writes (rewritten llc) = 0 && scans_of (rewritten llc) = 1 in
  let lines l = LL.unflat_lines l in
  p "the unedited step is fused" (fused base.raw_step);
  (* A flow compiling the backward as a routine of its own: nothing to anchor on. *)
  let backprop =
    let start =
      fst
        (Option.value_exn
           (List.findi stmts ~f:(fun _ -> function
             | LL.Comment c -> String.is_substring c ~substring:"backprop"
             | _ -> false)))
    in
    lines (List.drop stmts start)
  in
  p "a backward lowered without its forward is left as it is"
    (LL.equal (rewritten backprop) backprop && rowdot_writes backprop = 0);
  (* Another reader of the probabilities' gradient: fusing would leave it reading nothing. *)
  let a = position "softmax.grad" in
  let dp = Option.value_exn (target (List.nth_exn stmts a)) in
  let probe = Ll_test.node_factory ~first_id:100200 ~dims:[| 1 |] () "probe" in
  Ll_test.materialize probe;
  let extra_read =
    LL.Set
      {
        tn = probe;
        idcs = [| Ir.Indexing.Fixed_idx 0 |];
        llsc = LL.Get (dp, Array.map (Lazy.force dp.Tn.dims) ~f:(fun _ -> Ir.Indexing.Fixed_idx 0));
        debug = "";
      }
  in
  p "another reader of dP is declined" (declined (lines (insert_after a extra_read)));
  let staged = LL.Staged_compilation (fun () -> PPrint.empty) in
  p "staged code inside the backward's span is declined" (declined (lines (insert_after a staged)));
  p "staged code after the span is no objection" (fused (lines (stmts @ [ staged ])));
  (* A contraction whose channel loop never runs contributes nothing: not the pattern's. *)
  let rec kill_innermost = function
    | LL.For_loop ({ body = LL.Set _; _ } as l) -> LL.For_loop { l with to_ = -1 }
    | LL.For_loop l -> LL.For_loop { l with body = kill_innermost l.body }
    | s -> s
  in
  let k = position "q.grad" in
  p "a dead channel loop in the query-gradient contraction is declined"
    (declined (lines (List.mapi stmts ~f:(fun i s -> if i = k then kill_innermost s else s))));
  (* The composed max gradient: the softmax before phase 0. Declined, not matched as a variant --
     and the step it leaves still trains like the composed one. *)
  let composed = leaf ~softmax:composed_max_softmax ~live:causal_prefix ~on:false ~bwd:false () in
  let kept = leaf ~softmax:composed_max_softmax ~live:causal_prefix ~on:true ~bwd:true () in
  p "a backward with the composed max gradient is declined" (fused_count kept = 0);
  grads_agree ~tol:1e-5 ~what:"composed max gradient" kept.lgrads composed.lgrads

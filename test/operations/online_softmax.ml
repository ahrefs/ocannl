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
   through the residual. *)
let model ?mask_fill ~layers ~d_k ~prefix () =
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ batch; seq ] ~input_dims:[]
      ~output_dims:[ d_model ] ()
  in
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
let square_buffers (o : LL.optimized) =
  Set.filter (LL.writes_of_stmt o.LL.llc) ~f:(fun tn ->
      Array.count (Lazy.force tn.Tn.dims) ~f:(fun d -> d = seq) >= 2)

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

(* --- Leg 7: what the recognizer declines, on hand-built raw lowerings. --- *)

let () =
  printf "--- leg 7: the recognizer declines what it cannot prove, on hand-built lowerings ---\n";
  let module B = Ll_test in
  let n = 5 in
  (* One row of the composed softmax -- the four nests and the two initializations -- in the
     statement order [order], with the normalizer node at [l_prec]; [`Read_l] is a bystander
     statement reading the normalizer into an output. *)
  let build ?(l_prec = Ir.Ops.single) ?(l_dims = [| 1 |]) order =
    let mk = B.node_factory ~first_id:48300 ~dims:[| n |] () in
    let mk1 = B.node_factory ~first_id:48400 ~dims:[| 1 |] () in
    let mkl = B.node_factory ~prec:l_prec ~first_id:48500 ~dims:l_dims () in
    let x = mk "x" and nn = mk "n" and e = mk "e" in
    let m = mk1 "m" and y = mk1 "y" and l = mkl "l" in
    List.iter [ x; nn; e; m; y; l ] ~f:B.materialize;
    let v = mk1 "v" in
    B.virtualize v;
    (* A value reduction fed through a long elementwise chain from the probabilities. *)
    let width = 3 in
    let pp = mk "p" and ws = List.init 10 ~f:(fun k -> mk ("w" ^ Int.to_string k)) in
    let vals = B.node_factory ~first_id:48600 ~dims:[| n; width |] () "vals" in
    let out = B.node_factory ~first_id:48700 ~dims:[| width |] () "out" in
    List.iter ((pp :: ws) @ [ vals; out ]) ~f:B.materialize;
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
      | `Opaque -> LL.Staged_compilation (fun () -> PPrint.empty)
      | `Scope_write ->
          (* A scope whose body writes a tensor: impure by the optimizer's contract, but the tier
             runs ahead of that check and the write census does not enter scope bodies. *)
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
  let raw ?l_prec ?l_dims order =
    let llc, _, _ = build ?l_prec ?l_dims order in
    llc
  in
  let rewritten ?l_prec ?l_dims order = Online_softmax.rewrite (raw ?l_prec ?l_dims order) in
  let declined ?l_dims order =
    let raw = raw ?l_dims order in
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
  p "staged code outside that span is no objection"
    (scans_of (rewritten [ `Opaque; `A_init; `A; `N; `E; `C_init; `C ]) = 1);
  p "staged code reachable only through a guard's condition inside the span is declined too"
    (declined [ `A_init; `A; `N; `Opaque_cond; `E; `C_init; `C ]);
  p "a scope body inside the span is beyond the write census and declines the rewrite"
    (declined [ `A_init; `A; `Scope_write; `N; `E; `C_init; `C ]);
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
  let both label scores =
    let ((llc, x, l) as built) = build composed in
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
  let rec carried_of = function
    | LL.Scan_loop { carried; _ } -> Some carried
    | LL.Seq (a, b) -> Option.first_some (carried_of a) (carried_of b)
    | LL.For_loop { body; _ } -> carried_of body
    | _ -> None
  in
  let precs =
    Option.map
      (carried_of (rewritten ~l_prec:Ir.Ops.double composed))
      ~f:(fun carried ->
        List.map carried ~f:(fun (c : LL.carried) -> Lazy.force c.prev.tn.Tn.storage_prec))
  in
  p "the max's state stays single and the normalizer's takes its node's double"
    (Option.equal (List.equal Ir.Ops.equal_prec) precs (Some [ Ir.Ops.single; Ir.Ops.double ]))

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

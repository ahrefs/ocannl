(* gh-573: the transitive inline-fanin guard. A running sum (a transformer's residual stream) has
   per-cell read multiplicity within the visit cap — its consumers' copy-position reads are
   read-modify-write-exempt — and no reduction loops, so neither per-node cap
   ([virtualize_max_visits], [virtualize_max_inline_reduction]) ever materializes it; yet inlining
   it replays the entire prefix of the chain at every consumer, quadratic in depth. The guard caps
   the fan-in of the fully-inlined computation (the number of distinct materialized nodes it loads,
   accumulated through chains of virtual producers): the first chain node whose fan-in exceeds
   [virtualize_max_inline_fanin] is materialized (provenance 41), which resets the fan-in of
   everything downstream.

   Phase 1 pins the decision structurally on a hand-built [Ir.Low_level.t] chain through the
   [Ll_test] harness: with the default cap 8, a 10-link add chain materializes exactly the link
   whose fan-in first reaches 9 (x8), the links before and after stay virtual, and the
   materialization is reported as an [`Inline] flip whose recompute cost carries the fan-in (extent
   1 × multiplicity 1 × fan-in 9 — so the memory-budget planner does not rank the node among the
   cheapest to re-inline). Disabling the cap reproduces the old behavior (whole chain virtual). Both
   readings execute and must agree cell for cell with the OCaml reference (gh-ocannl-589: placement
   decisions need an executed leg, not just structural pins).

   Phase 1b pins that reads inside a [Local_scope] body in a setter's right-hand side count toward
   that setter's fan-in (review round 1): a producer computed through a scope body loading 9
   materialized nodes materializes under the cap.

   Phase 2 exercises the real [Assignments] pipeline on a tensor-graph chain built with [TDSL.O.( +
   )], asserting the same placement pattern through [Context.placements] and value parity between
   the default and cap-disabled compiles.

   Phase 3 pins the cross-routine reading (review round 1): the same chain split across two
   [Ll_test.optimize_in] calls sharing one [optimize_ctx] — the second routine reads a node the
   first committed [Virtual], whose fan-in is derived from the stored computation rather than from
   (absent) local setters, so the chain cannot escape the cap by being compiled in pieces. The
   executed leg is gh-ocannl-610's acceptance test: routine B's kernel parameters must include the
   leaves (x0, w1..w5) that reach it only through the spliced cross-routine computation.

   Phase 4 pins gh-ocannl-611: a routine whose entire content virtualizes away is legal — it
   optimizes to an empty schedule ([Noop]) whose stored computations persist in the lineage,
   compiles and runs as a no-op, and a later routine consumes the deferred chain, executed against
   the same reference. *)

open Base
open Ll_test
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode

let n_links = 10
let dim = 4

(* === The chain, shared by phases 1 and 3 ===

   x0, w1..w10 materialized; x_k = x_{k-1} + w_k; out = x_10. Fan-in of x_k is k+1 ({x0, w1..wk});
   the default cap 8 trips first at x8 (fan-in 9). *)

type chain = {
  x0 : Tn.t;
  ws : Tn.t array;
  xs : Tn.t array;
  out : Tn.t;
  links : LL.t list;
  consumer : LL.t;
  mk : string -> Tn.t;
}

let build_chain ~first_id () =
  let mk = node_factory ~first_id ~dims:[| dim |] () in
  let x0 = mk "x0" in
  materialize x0;
  let ws =
    Array.init n_links ~f:(fun k ->
        let w = mk (Printf.sprintf "w%d" (k + 1)) in
        materialize w;
        w)
  in
  let xs = Array.init n_links ~f:(fun k -> mk (Printf.sprintf "x%d" (k + 1))) in
  let out = mk "out" in
  materialize out;
  let link k =
    let prev = if k = 0 then x0 else xs.(k - 1) in
    let s = sym () in
    loop_n s dim (set xs.(k) [| iter s |] (add (get prev [| iter s |]) (get ws.(k) [| iter s |])))
  in
  let links = List.init n_links ~f:link in
  let consumer =
    let s = sym () in
    loop_n s dim (set out [| iter s |] (get xs.(n_links - 1) [| iter s |]))
  in
  { x0; ws; xs; out; links; consumer; mk }

let x0_vals = Array.init dim ~f:(fun i -> Float.of_int (i + 1))
let w_vals k = Array.init dim ~f:(fun i -> Float.of_int ((100 * (k + 1)) + (10 * (i + 1))))

let expected_out =
  Array.init dim ~f:(fun i ->
      Array.foldi (Array.init n_links ~f:w_vals) ~init:x0_vals.(i) ~f:(fun _ acc w -> acc +. w.(i)))

let chain_seed { x0; ws; out; _ } =
  ((x0, x0_vals) :: List.init n_links ~f:(fun k -> (ws.(k), w_vals k))) @ [ (out, blank dim) ]

let find_flip (o : LL.optimized) tn =
  List.find_map o.LL.flip_candidates ~f:(fun fc ->
      if Tn.equal fc.LL.fc_tn tn then Some (fc.LL.fc_flip, fc.LL.fc_recompute_cost) else None)

(* === Phase 1: hand-built chain, one routine === *)

let phase1 () =
  assert (LL.virtualize_settings.max_inline_fanin = 8);
  let c = build_chain ~first_id:3000 () in
  let llc = List.reduce_exn ~f:seq (c.links @ [ c.consumer ]) in
  let o = optimize ~name:"vcf_chain" llc in
  p "chain: x7 stays virtual (fan-in 8, at the cap)" (known_virtual o c.xs.(6));
  p "chain: x8 materialized by the fan-in cap" (known_non_virtual o c.xs.(7));
  p "chain: x9 and x10 virtual again past the reset"
    (known_virtual o c.xs.(8) && known_virtual o c.xs.(9));
  p "chain: x8 written once, read as a buffer downstream"
    (count_set o c.xs.(7) = 1 && count_get o c.xs.(7) >= 1);
  p "chain: x8 is an Inline flip charged its fan-in (cost 9)"
    (match find_flip o c.xs.(7) with Some (`Inline, 9) -> true | _ -> false);
  (* Executed parity: the guard is a placement decision, so both readings of the same program must
     produce the same cells. Discriminating producer values: vary with the link and the cell, off
     the zero-init and the sentinel. *)
  let seed = chain_seed c and read = [ c.out ] in
  let got = execute ~name:"vcf_chain" o ~seed ~read in
  p "chain: executed values match the reference" (same got [ expected_out ]);
  LL.virtualize_settings.max_inline_fanin <- -1;
  let o_off = optimize ~name:"vcf_chain_off" llc in
  LL.virtualize_settings.max_inline_fanin <- 8;
  p_all "cap disabled: whole chain virtual (the pre-fix behavior)" (Array.to_list c.xs)
    ~f:(known_virtual o_off);
  let got_off = execute ~name:"vcf_chain_off" o_off ~seed ~read in
  p "cap disabled: executed values agree with the capped arm" (same got got_off)

(* === Phase 1b: scope-body reads count toward the enclosing setter's fan-in === *)

let phase1b () =
  let mk = node_factory ~first_id:3100 ~dims:[| dim |] () in
  let ws =
    Array.init 9 ~f:(fun k ->
        let w = mk (Printf.sprintf "sw%d" (k + 1)) in
        materialize w;
        w)
  in
  let y = mk "y" and out = mk "sout" and lv = mk "lv" in
  materialize out;
  virtualize lv;
  let i = sym () and j = sym () in
  let id = LL.get_scope lv in
  (* y[i] = Local_scope{ lv := w1[i] + ... + w9[i] }: nine loads, all inside the scope body. *)
  let body =
    LL.Set_local (id, Array.fold ws ~init:(c 0.) ~f:(fun acc w -> add acc (get w [| iter i |])))
  in
  let producer =
    loop_n i dim
      (set y
         [| iter i |]
         (LL.Local_scope { id; body; orig_indices = [| iter i |]; mint = LL.Inlined_computation }))
  in
  let consumer = loop_n j dim (set out [| iter j |] (get y [| iter j |])) in
  let o = optimize ~name:"vcf_scope" (seq producer consumer) in
  p "scope: producer materialized (scope-body loads count, fan-in 9)" (known_non_virtual o y);
  let seed = (out, blank dim) :: (Array.to_list ws |> List.mapi ~f:(fun k w -> (w, w_vals k))) in
  let expected =
    Array.init dim ~f:(fun i ->
        Array.foldi (Array.init 9 ~f:w_vals) ~init:0. ~f:(fun _ acc w -> acc +. w.(i)))
  in
  let got = execute ~name:"vcf_scope" o ~seed ~read:[ out ] in
  p "scope: executed values match the reference" (same got [ expected ]);
  (* Review round 2: a scope body ACCUMULATING INTO the setter's own node must not record the
     self-read as a contributor. y2 is zero-initialized and set from a scope reading y2 itself plus
     exactly 8 materialized nodes: fan-in 8, at the cap — a phantom self contributor would make it 9
     and flip the decision. *)
  let y2 = mk "y2" and out2 = mk "sout2" and lv2 = mk "lv2" in
  materialize out2;
  virtualize lv2;
  let i2 = sym () and j2 = sym () in
  let id2 = LL.get_scope lv2 in
  let body2 =
    LL.Set_local
      ( id2,
        Array.fold (Array.sub ws ~pos:0 ~len:8)
          ~init:(get y2 [| iter i2 |])
          ~f:(fun acc w -> add acc (get w [| iter i2 |])) )
  in
  let producer2 =
    seq (zero y2)
      (loop_n i2 dim
         (set y2
            [| iter i2 |]
            (LL.Local_scope
               {
                 id = id2;
                 body = body2;
                 orig_indices = [| iter i2 |];
                 mint = LL.Inlined_computation;
               })))
  in
  let consumer2 = loop_n j2 dim (set out2 [| iter j2 |] (get y2 [| iter j2 |])) in
  let o2 = optimize ~name:"vcf_scope_rmw" (seq producer2 consumer2) in
  p "scope rmw: self-read excluded, producer stays virtual (fan-in 8)" (known_virtual o2 y2);
  let expected2 =
    Array.init dim ~f:(fun i ->
        Array.foldi (Array.init 8 ~f:w_vals) ~init:0. ~f:(fun _ acc w -> acc +. w.(i)))
  in
  let seed2 =
    (out2, blank dim)
    :: (Array.to_list (Array.sub ws ~pos:0 ~len:8) |> List.mapi ~f:(fun k w -> (w, w_vals k)))
  in
  let got2 = execute ~name:"vcf_scope_rmw" o2 ~seed:seed2 ~read:[ out2 ] in
  p "scope rmw: executed values match the reference" (same got2 [ expected2 ]);
  (* Review round 5: [Where]'s arms are mutually exclusive per evaluation
     ([Ops.ternop_conditionality] = [Cond_and_one_arm]), so a setter's fan-in charges the wider arm,
     not the union: mask + two 4-load arms is fan-in 5, within the cap — the union (9) would
     spuriously materialize. Both arms execute in the parity leg (mask alternates). *)
  let y4 = mk "y4" and out4 = mk "wout" and m = mk "m" in
  materialize out4;
  materialize m;
  let i4 = sym () and j4 = sym () in
  let arm pos =
    Array.fold (Array.sub ws ~pos ~len:4) ~init:(Ll_test.c 0.) ~f:(fun acc w ->
        add acc (get w [| iter i4 |]))
  in
  let producer4 =
    loop_n i4 dim
      (set y4
         [| iter i4 |]
         (LL.Ternop (Ir.Ops.Where, (get m [| iter i4 |], single), (arm 0, single), (arm 4, single))))
  in
  let consumer4 = loop_n j4 dim (set out4 [| iter j4 |] (get y4 [| iter j4 |])) in
  let o4 = optimize ~name:"vcf_where" (seq producer4 consumer4) in
  p "where: producer stays virtual (fan-in 5: mask + the wider arm)" (known_virtual o4 y4);
  let m_vals = Array.init dim ~f:(fun i -> Float.of_int (i % 2)) in
  let expected4 =
    Array.init dim ~f:(fun i ->
        let pos = if i % 2 <> 0 then 0 else 4 in
        Array.foldi
          (Array.init 4 ~f:(fun k -> w_vals (pos + k)))
          ~init:0.
          ~f:(fun _ acc w -> acc +. w.(i)))
  in
  let seed4 =
    (out4, blank dim) :: (m, m_vals) :: (Array.to_list ws |> List.mapi ~f:(fun k w -> (w, w_vals k)))
  in
  let got4 = execute ~name:"vcf_where" o4 ~seed:seed4 ~read:[ out4 ] in
  p "where: executed values match the reference (both arms exercised)" (same got4 [ expected4 ]);
  (* Review round 6: [Local_scope] bodies inside [Where] arms hoist to statement level and BOTH
     execute (see the operand-conditionality agent notes), so their reads union at the statement
     sink rather than joining the per-arm maximum: mask + two hoisted 4-load bodies is fan-in 9 —
     over the cap — where the arm-expression maximum alone would report 5. *)
  let y5 = mk "y5" and out5 = mk "hout" and lva = mk "lva" and lvb = mk "lvb" in
  materialize out5;
  virtualize lva;
  virtualize lvb;
  let i5 = sym () and j5 = sym () in
  let ida = LL.get_scope lva and idb = LL.get_scope lvb in
  let scope_arm sid pos =
    LL.Local_scope
      {
        id = sid;
        body =
          LL.Set_local
            ( sid,
              Array.fold (Array.sub ws ~pos ~len:4) ~init:(Ll_test.c 0.) ~f:(fun acc w ->
                  add acc (get w [| iter i5 |])) );
        orig_indices = [| iter i5 |];
        mint = LL.Inlined_computation;
      }
  in
  let producer5 =
    loop_n i5 dim
      (set y5
         [| iter i5 |]
         (LL.Ternop
            ( Ir.Ops.Where,
              (get m [| iter i5 |], single),
              (scope_arm ida 0, single),
              (scope_arm idb 4, single) )))
  in
  let consumer5 = loop_n j5 dim (set out5 [| iter j5 |] (get y5 [| iter j5 |])) in
  let o5 = optimize ~name:"vcf_where_scopes" (seq producer5 consumer5) in
  p "where scopes: producer materialized (hoisted bodies both charge, fan-in 9)"
    (known_non_virtual o5 y5);
  let seed5 =
    (out5, blank dim) :: (m, m_vals) :: (Array.to_list ws |> List.mapi ~f:(fun k w -> (w, w_vals k)))
  in
  let got5 = execute ~name:"vcf_where_scopes" o5 ~seed:seed5 ~read:[ out5 ] in
  p "where scopes: executed values match the reference" (same got5 [ expected4 ])

(* === Phase 2: the Assignments pipeline === *)

open Ocannl
open Ocannl.Operation.DSL_modules

let phase2 () =
  let build () =
    Utils.settings.fixed_state_for_init <- Some 42;
    Tensor.unsafe_reinitialize ();
    let init l k =
      NTDSL.init ~l ~prec:Ir.Ops.single ~o:[ dim ]
        ~f:(fun idcs -> Float.of_int ((100 * k) + idcs.(0) + 1))
        ()
    in
    let x0 = init "vcf_x0" 0 in
    let xs =
      List.folding_map
        (List.init n_links ~f:(fun k -> k + 1))
        ~init:x0
        ~f:(fun x k ->
          let x' = TDSL.O.( + ) x (init (Printf.sprintf "vcf_w%d" k) k) in
          (x', x'))
    in
    let last = List.last_exn xs in
    let%op loss = last ++ "i=>0" in
    Train.set_materialized loss.Tensor.value;
    Tn.set_observable loss.Tensor.value;
    (xs, loss)
  in
  let run ~cap =
    LL.virtualize_settings.max_inline_fanin <- cap;
    let xs, loss = build () in
    let ctx = Context.auto () in
    let ctx, routine = Context.compile ctx (Train.forward loss) Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    LL.virtualize_settings.max_inline_fanin <- 8;
    (Context.get_values ctx loss.Tensor.value, Context.placements ctx, xs)
  in
  let lv_default, plc_default, xs_default = run ~cap:8 in
  let lv_off, plc_off, xs_off = run ~cap:(-1) in
  let value k xs = (List.nth_exn xs (k - 1)).Tensor.value in
  p "pipeline: x7 stays virtual" (Tn.Placements.known_virtual plc_default (value 7 xs_default));
  p "pipeline: x8 materialized by the fan-in cap"
    (Tn.Placements.known_non_virtual plc_default (value 8 xs_default));
  p "pipeline: x9 and x10 virtual again past the reset"
    (Tn.Placements.known_virtual plc_default (value 9 xs_default)
    && Tn.Placements.known_virtual plc_default (value 10 xs_default));
  p_all "pipeline, cap disabled: whole chain virtual" xs_off ~f:(fun x ->
      Tn.Placements.known_virtual plc_off x.Tensor.value);
  let expected =
    Array.init dim ~f:(fun i ->
        List.fold
          (List.init n_links ~f:(fun k -> k + 1))
          ~init:(Float.of_int (i + 1))
          ~f:(fun acc k -> acc +. Float.of_int ((100 * k) + i + 1)))
    |> Array.fold ~init:0. ~f:( +. )
  in
  p "pipeline: loss matches the reference"
    (Array.length lv_default = 1 && Float.(abs (lv_default.(0) -. expected) <= 1e-3));
  p "pipeline: default and cap-disabled arms agree"
    (Array.length lv_off = 1 && Float.(abs (lv_default.(0) -. lv_off.(0)) <= 1e-3))

(* === Phase 3: the chain split across two routines in one lineage === *)

let phase3 () =
  let c = build_chain ~first_id:3200 () in
  let split = 5 in
  (* Routine A carries a materialized copy-out of x5, mirroring a real routine's observable root;
     its same-position read leaves x5 virtual. Phase 4 exercises the copy-out-free, fully-virtual
     variant (gh-ocannl-611). *)
  let out_a = c.mk "out_a" in
  materialize out_a;
  let copy_out =
    let s = sym () in
    loop_n s dim (set out_a [| iter s |] (get c.xs.(split - 1) [| iter s |]))
  in
  let llc_a = List.reduce_exn ~f:seq (List.take c.links split @ [ copy_out ]) in
  let llc_b = List.reduce_exn ~f:seq (List.drop c.links split @ [ c.consumer ]) in
  let ctx = LL.empty_optimize_ctx () in
  let o_a = optimize_in ctx ~name:"vcf_xchain_a" llc_a in
  p "cross-routine: routine A leaves x1..x5 virtual"
    (Array.for_alli c.xs ~f:(fun k tn -> k >= split || known_virtual o_a tn));
  let o_b = optimize_in ctx ~name:"vcf_xchain_b" llc_b in
  (* Routine B never sets x5, yet x5's stored computation is replayed when inlined: its fan-in (6:
     {x0, w1..w5}) is derived from the computation, so x6 sees 7, x7 sees 8, and x8 trips the cap
     exactly as in the single-routine reading. *)
  p "cross-routine: x7 stays virtual" (known_virtual o_b c.xs.(6));
  p "cross-routine: x8 materialized by the fan-in cap" (known_non_virtual o_b c.xs.(7));
  p "cross-routine: x9 and x10 virtual" (known_virtual o_b c.xs.(8) && known_virtual o_b c.xs.(9));
  (* gh-ocannl-610 acceptance: routine B executes correctly, its kernel declaring parameters for the
     leaves that reach it only through the spliced cross-routine computation. *)
  let got = execute ~name:"vcf_xchain_b" o_b ~seed:(chain_seed c) ~read:[ c.out ] in
  p "cross-routine: executed values match the reference" (same got [ expected_out ]);
  (* Review round 3: a routine that UPDATES an inherited virtual node must not lose the stored
     prefix behind the update's excluded self-read — the inherited component reads union with the
     local setter reads. Routine B': x5 += w6 (self-update of the inherited virtual), then three
     fresh links; x5's fan-in is {w6} ∪ {x0, w1..w5} = 7, so the second fresh link (fan-in 9) trips
     the cap while the first (fan-in 8) stays virtual. *)
  let c2 = build_chain ~first_id:3300 () in
  let llc_a2 =
    let out_a2 = c2.mk "out_a2" in
    materialize out_a2;
    let s = sym () in
    List.reduce_exn ~f:seq
      (List.take c2.links split
      @ [ loop_n s dim (set out_a2 [| iter s |] (get c2.xs.(split - 1) [| iter s |])) ])
  in
  let x5 = c2.xs.(split - 1) in
  let upd =
    let s = sym () in
    loop_n s dim (set x5 [| iter s |] (add (get x5 [| iter s |]) (get c2.ws.(5) [| iter s |])))
  in
  let link6 =
    let s = sym () in
    loop_n s dim
      (set c2.xs.(5) [| iter s |] (add (get x5 [| iter s |]) (get c2.ws.(6) [| iter s |])))
  in
  let link7 =
    let s = sym () in
    loop_n s dim
      (set c2.xs.(6) [| iter s |] (add (get c2.xs.(5) [| iter s |]) (get c2.ws.(7) [| iter s |])))
  in
  let link8 =
    let s = sym () in
    loop_n s dim
      (set c2.xs.(7) [| iter s |] (add (get c2.xs.(6) [| iter s |]) (get c2.ws.(8) [| iter s |])))
  in
  let consumer2 =
    let s = sym () in
    loop_n s dim (set c2.out [| iter s |] (get c2.xs.(7) [| iter s |]))
  in
  let ctx2 = LL.empty_optimize_ctx () in
  let _o_a2 = optimize_in ctx2 ~name:"vcf_xupd_a" llc_a2 in
  let o_b2 =
    optimize_in ctx2 ~name:"vcf_xupd_b"
      (List.reduce_exn ~f:seq [ upd; link6; link7; link8; consumer2 ])
  in
  p "cross-routine update: x6 stays virtual (fan-in 8, prefix counted through the update)"
    (known_virtual o_b2 c2.xs.(5));
  p "cross-routine update: x7 materialized by the fan-in cap" (known_non_virtual o_b2 c2.xs.(6));
  p "cross-routine update: x8 virtual past the reset" (known_virtual o_b2 c2.xs.(7))
(* Review round 4 note: [reads_of_proc] skips dead loops ([to_ < from_]) for parity with
   [trace_node_facts] — a dead body replays zero times. No boundary pin: the posited scenario (an
   earlier routine committing a Virtual whose stored computation holds a dead-loop setter) is not
   reachable through the pipeline — a zero-initialized node whose only setter is dead never
   virtualizes (it reaches later routines as an undecided read-only node and materializes by the
   read-only rule), so no stored computation with a dead loop exists to over-charge. *)

(* === Phase 4: a routine that optimizes to nothing (gh-ocannl-611) === *)

let phase4 () =
  let c = build_chain ~first_id:3400 () in
  let split = 5 in
  (* No copy-out: routine A is deferral-only — every target virtualizes and nothing remains. *)
  let llc_a = List.reduce_exn ~f:seq (List.take c.links split) in
  let llc_b = List.reduce_exn ~f:seq (List.drop c.links split @ [ c.consumer ]) in
  let ctx = LL.empty_optimize_ctx () in
  let o_a = optimize_in ctx ~name:"vcf_allvirt_a" llc_a in
  p "all-virtual: routine A optimizes to an empty schedule"
    (match o_a.LL.llc with LL.Noop -> true | _ -> false);
  (* Review round 1 (P2): the interface must match the empty schedule — the deferred computations'
     leaves are not phantom inputs of a routine that reads nothing. *)
  p "all-virtual: the empty schedule has an empty interface"
    (let (ins, outs), merge = LL.input_and_output_nodes o_a in
     Set.is_empty ins && Set.is_empty outs && Option.is_none merge);
  p "all-virtual: x1..x5 committed virtual"
    (Array.for_alli c.xs ~f:(fun k tn -> k >= split || known_virtual o_a tn));
  p "all-virtual: the deferred computations persist in the lineage"
    (Hashtbl.mem ctx.LL.computations c.xs.(split - 1));
  (* The empty routine must remain compilable and runnable end to end. *)
  let _ctx_a = run ~name:"vcf_allvirt_a" o_a ~seed:[] in
  p "all-virtual: routine A compiles and runs as a no-op" true;
  let o_b = optimize_in ctx ~name:"vcf_allvirt_b" llc_b in
  p "all-virtual: fan-in decisions unchanged in routine B (x8 trips the cap)"
    (known_virtual o_b c.xs.(6) && known_non_virtual o_b c.xs.(7) && known_virtual o_b c.xs.(8));
  (let (ins, _), _ = LL.input_and_output_nodes o_b in
   p_all "all-virtual: routine B declares the spliced leaves as inputs" (c.x0 :: Array.to_list c.ws)
     ~f:(Set.mem ins));
  let got = execute ~name:"vcf_allvirt_b" o_b ~seed:(chain_seed c) ~read:[ c.out ] in
  p "all-virtual: executed values match the reference" (same got [ expected_out ])

(* === Phase 5: spliced reads vs the consumer's own writes (review round 1) === *)

let phase5 () =
  (* P1: a spliced read PRECEDING the routine's own write of an already-traced node must flip it to
     read_before_write — the raw analysis alone sees only the write, concludes the node is
     output-only, and the incoming value would read as ignorable. Routine A defers v := ell + 100;
     routine B consumes v (splicing the ell read) and THEN overwrites ell. *)
  let mk = node_factory ~first_id:3500 ~dims:[| dim |] () in
  let ell = mk "ell" in
  materialize ell;
  let v = mk "v" and out = mk "vout" in
  materialize out;
  let ctx = LL.empty_optimize_ctx () in
  let llc_a =
    let s = sym () in
    loop_n s dim (set v [| iter s |] (add (get ell [| iter s |]) (c 100.)))
  in
  let o_a = optimize_in ctx ~name:"vcf_prewrite_a" llc_a in
  p "pre-write splice: routine A defers v" (known_virtual o_a v);
  let llc_b =
    let s = sym () and s2 = sym () in
    seq
      (loop_n s dim (set out [| iter s |] (get v [| iter s |])))
      (loop_n s2 dim (set ell [| iter s2 |] (ramp 1000. s2)))
  in
  let o_b = optimize_in ctx ~name:"vcf_prewrite_b" llc_b in
  p "pre-write splice: ell flips to read_before_write" (read_before_write o_b ell);
  p "pre-write splice: ell is an input of routine B"
    (let (ins, _), _ = LL.input_and_output_nodes o_b in
     Set.mem ins ell);
  let ell_old = Array.init dim ~f:(fun i -> 1. +. Float.of_int i) in
  let got =
    execute ~name:"vcf_prewrite_b" o_b ~seed:[ (ell, ell_old); (out, blank dim) ] ~read:[ out; ell ]
  in
  p "pre-write splice: out sees the incoming ell, ell holds the overwrite"
    (same got
       [
         Array.map ell_old ~f:(fun x -> x +. 100.);
         Array.init dim ~f:(fun i -> 1000. +. Float.of_int i);
       ]);
  (* P1: a stored computation reading the merge buffer must not be consumed by a LATER routine —
     merge-buffer contents are transient to the transfer-receiving routine, so the deferral would
     change which transfer the read observes. The splice raises instead of emitting an undeclared
     merge parameter. *)
  let msrc = mk "msrc" in
  materialize msrc;
  let mv = mk "mv" and mout = mk "mout" in
  materialize mout;
  let ctx2 = LL.empty_optimize_ctx () in
  let llc_ma =
    let s = sym () in
    loop_n s dim (set mv [| iter s |] (LL.Get_merge_buffer (msrc, [| iter s |])))
  in
  let o_ma = optimize_in ctx2 ~name:"vcf_merge_a" llc_ma in
  p "merge splice: routine A defers the merge-reading node" (known_virtual o_ma mv);
  p "merge splice: the deferred-away merge declaration is dropped from routine A"
    (Option.is_none o_ma.LL.merge_node);
  let llc_mb =
    let s = sym () in
    loop_n s dim (set mout [| iter s |] (get mv [| iter s |]))
  in
  let rejected =
    try
      ignore (optimize_in ctx2 ~name:"vcf_merge_b" llc_mb : LL.optimized);
      false
    with Utils.User_error msg -> String.is_substring msg ~substring:"merge buffer"
  in
  p "merge splice: consuming it in a later routine is rejected" rejected;
  (* Review round 2, P1: the consumer declaring the SAME merge source does not rescue the splice —
     the deferred read would observe the consumer's transfer, not the one it was written against.
     Detection is at consumption time (an entry-time snapshot in [virtual_llc]), because in the
     final code this case is indistinguishable from legitimate same-routine inlining of a declared
     merge read. *)
  let mout2 = mk "mout2" in
  materialize mout2;
  let llc_mb2 =
    let s = sym () in
    loop_n s dim
      (set mout2
         [| iter s |]
         (add (get mv [| iter s |]) (LL.Get_merge_buffer (msrc, [| iter s |]))))
  in
  let rejected2 =
    try
      ignore (optimize_in ctx2 ~name:"vcf_merge_b2" llc_mb2 : LL.optimized);
      false
    with Utils.User_error msg -> String.is_substring msg ~substring:"merge buffer"
  in
  p "merge splice: same-source consumer is rejected too" rejected2;
  (* Review round 5, P2: a legitimate same-routine merge read that survives optimization must not
     mint an ordinary traced entry for its SOURCE — the merge buffer is the parameter, and a phantom
     source entry would double the transfer buffer's allocation. *)
  let msrc3 = mk "msrc3" in
  materialize msrc3;
  let mtar = mk "mtar" in
  materialize mtar;
  let ctx7 = LL.empty_optimize_ctx () in
  let llc_m3 =
    let s = sym () in
    loop_n s dim (set mtar [| iter s |] (LL.Get_merge_buffer (msrc3, [| iter s |])))
  in
  let o_m3 = optimize_in ctx7 ~name:"vcf_merge_self" llc_m3 in
  p "same-routine merge read: source is the merge param, not an ordinary entry"
    ((match o_m3.LL.merge_node with Some m -> Tn.equal m msrc3 | None -> false)
    && not (Hashtbl.mem o_m3.LL.traced_store msrc3));
  (* Review round 2, P1: a write covering only SOME cells does not cover a later spliced read —
     coverage is per-cell and guard-aware, decided by [reads_covered_query] over the final code, not
     by syntactic write-before-read order. Routine B writes ell2[0] only, then consumes v2 = ell2 +
     100 at an OFFSET position (out2[i] = v2[i+1]): the spliced reads touch cells the write never
     covers, so ell2 must stay a routine input. The offset also keeps the read off the enclosing
     write's index position, so this case does not depend on how the [rmw_exempt] copy-position
     reads are judged — that split is gh-ocannl-618's, pinned in splice_semantics.ml (raw side) and
     by the same-position splice below (round 6). *)
  let ell2 = mk "ell2" in
  materialize ell2;
  let v2 = mk "v2" and out2 = mk "pout" in
  materialize out2;
  let ctx3 = LL.empty_optimize_ctx () in
  let llc_a3 =
    let s = sym () in
    loop_n s dim (set v2 [| iter s |] (add (get ell2 [| iter s |]) (c 100.)))
  in
  let o_a3 = optimize_in ctx3 ~name:"vcf_partial_a" llc_a3 in
  p "partial-write splice: routine A defers v2" (known_virtual o_a3 v2);
  let llc_b3 =
    let s = sym () in
    seq
      (set ell2 [| fixed 0 |] (c 5000.))
      (loop_n s (dim - 1) (set out2 [| iter s |] (get v2 [| aff [ (1, s) ] 1 |])))
  in
  let o_b3 = optimize_in ctx3 ~name:"vcf_partial_b" llc_b3 in
  p "partial-write splice: ell2 stays read_before_write (per-cell coverage)"
    (read_before_write o_b3 ell2);
  p "partial-write splice: ell2 is an input of routine B"
    (let (ins, _), _ = LL.input_and_output_nodes o_b3 in
     Set.mem ins ell2);
  let ell2_old = Array.init dim ~f:(fun i -> 1. +. Float.of_int i) in
  let got3 =
    execute ~name:"vcf_partial_b" o_b3 ~seed:[ (ell2, ell2_old); (out2, blank dim) ] ~read:[ out2 ]
  in
  (* out2[i] = ell2[i+1] + 100 for i in 0..2 (incoming cells — the write touched only cell 0);
     out2[3] keeps the sentinel, no writer covers it. *)
  let expected3 =
    Array.init dim ~f:(fun i -> if i < dim - 1 then ell2_old.(i + 1) +. 100. else sentinel)
  in
  p "partial-write splice: uncovered cells read the incoming values" (same got3 [ expected3 ]);
  (* Review round 3, P1: a DEAD loop's write must not supply coverage — the raw side drops dead
     accesses ([drop_dead_loop_accesses]) and the reconcile walk now mirrors it: a dead write does
     not enter the first-write tracking and dead accesses are filtered from the coverage query.
     Routine B: a dead full write of ell3, then the spliced read, then a live write. *)
  let ell3 = mk "ell3" in
  materialize ell3;
  let v3 = mk "v3" and out3 = mk "dout" in
  materialize out3;
  let ctx4 = LL.empty_optimize_ctx () in
  let llc_a4 =
    let s = sym () in
    loop_n s dim (set v3 [| iter s |] (add (get ell3 [| iter s |]) (c 100.)))
  in
  let o_a4 = optimize_in ctx4 ~name:"vcf_dead_a" llc_a4 in
  p "dead-write splice: routine A defers v3" (known_virtual o_a4 v3);
  (* ghost/ghost2 are mentioned ONLY inside a dead loop: since round 13 drops dead loops at
     virtualization, nothing about them survives — no registry entry, no parameter, no interface
     presence (rounds 4 and 13). *)
  let ghost = mk "ghost" in
  materialize ghost;
  let ghost2 = mk "ghost2" in
  materialize ghost2;
  let llc_b4 =
    let s0 = sym () and s = sym () and s2 = sym () and s4 = sym () in
    List.reduce_exn ~f:seq
      [
        loop ~upto:(-1) s0 (set ell3 [| iter s0 |] (c 9.));
        loop ~upto:(-1) s4 (set ghost [| iter s4 |] (get ghost2 [| iter s4 |]));
        loop_n s dim (set out3 [| iter s |] (get v3 [| iter s |]));
        loop_n s2 dim (set ell3 [| iter s2 |] (ramp 2000. s2));
      ]
  in
  let o_b4 = optimize_in ctx4 ~name:"vcf_dead_b" llc_b4 in
  p "dead-write splice: ell3 is read-before-write (a dead write supplies no coverage)"
    (read_before_write o_b4 ell3);
  p "dead-only mention: dead code is dropped wholesale, nothing registered"
    ((not (Hashtbl.mem o_b4.LL.traced_store ghost))
    && (not (Hashtbl.mem o_b4.LL.traced_store ghost2))
    &&
    let (ins, outs), _ = LL.input_and_output_nodes o_b4 in
    (not (Set.mem outs ghost)) && not (Set.mem ins ghost2));
  let ell3_old = Array.init dim ~f:(fun i -> 7. +. Float.of_int i) in
  let got4 =
    execute ~name:"vcf_dead_b" o_b4
      ~seed:[ (ell3, ell3_old); (out3, blank dim) ]
      ~read:[ out3; ell3 ]
  in
  p "dead-write splice: out sees incoming ell3, ell3 holds the live overwrite"
    (same got4
       [
         Array.map ell3_old ~f:(fun x -> x +. 100.);
         Array.init dim ~f:(fun i -> 2000. +. Float.of_int i);
       ]);
  (* Review round 3, P2: a projection's discarded operand is never rendered — a merge read inside it
     must not taint the deferred computation, and its ordinary reads must not become phantom
     parameters. v5 := Arg2(msrc.merge + extra, ell5): the whole first operand is discarded, so
     consuming v5 in a later routine is legal and computes the second operand. *)
  let ell5 = mk "ell5" in
  materialize ell5;
  let extra = mk "extra" in
  materialize extra;
  let v5 = mk "v5" and out5 = mk "aout" in
  materialize out5;
  let ctx5 = LL.empty_optimize_ctx () in
  let llc_a5 =
    let s = sym () in
    loop_n s dim
      (set v5
         [| iter s |]
         (binop Ir.Ops.Arg2
            (add (LL.Get_merge_buffer (msrc, [| iter s |])) (get extra [| iter s |]))
            (get ell5 [| iter s |])))
  in
  let o_a5 = optimize_in ctx5 ~name:"vcf_proj_a" llc_a5 in
  p "discarded operand: routine A defers v5" (known_virtual o_a5 v5);
  let llc_b5 =
    let s = sym () in
    loop_n s dim (set out5 [| iter s |] (get v5 [| iter s |]))
  in
  let o_b5 = optimize_in ctx5 ~name:"vcf_proj_b" llc_b5 in
  p "discarded operand: no phantom input from the discarded read"
    (let (ins, _), _ = LL.input_and_output_nodes o_b5 in
     (not (Set.mem ins extra)) && Set.mem ins ell5);
  let ell5_vals = Array.init dim ~f:(fun i -> 40. +. Float.of_int i) in
  let got5 =
    execute ~name:"vcf_proj_b" o_b5 ~seed:[ (ell5, ell5_vals); (out5, blank dim) ] ~read:[ out5 ]
  in
  p "discarded operand: executed values are the projected operand's" (same got5 [ ell5_vals ]);
  (* Review round 4, P1: a write under an [If] guard is never a DEFINITE write — when the guard is
     false at runtime the entry value is still required, so a guarded full write must not suppress
     the input flag of a later spliced read (the coverage query's guards-taken contract is for
     placement decisions, not the interface: guarded writes are pre-filtered). Both branches execute
     below; the offset read keeps the splice off the rmw_exempt position. *)
  let flag = mk ~dims:[| 1 |] "flag" in
  materialize flag;
  let ell7 = mk "ell7" in
  materialize ell7;
  let v7 = mk "v7" and out7 = mk "gout" in
  materialize out7;
  let ctx6 = LL.empty_optimize_ctx () in
  let llc_a6 =
    let s = sym () in
    loop_n s dim (set v7 [| iter s |] (add (get ell7 [| iter s |]) (c 100.)))
  in
  let o_a6 = optimize_in ctx6 ~name:"vcf_guard_a" llc_a6 in
  p "guarded-write splice: routine A defers v7" (known_virtual o_a6 v7);
  let llc_b6 =
    let sg = sym () and s = sym () in
    seq
      (LL.If
         {
           cond = (get flag [| fixed 0 |], single);
           body = loop_n sg dim (set ell7 [| iter sg |] (c 999.));
         })
      (loop_n s (dim - 1) (set out7 [| iter s |] (get v7 [| aff [ (1, s) ] 1 |])))
  in
  let o_b6 = optimize_in ctx6 ~name:"vcf_guard_b" llc_b6 in
  p "guarded-write splice: ell7 stays read_before_write (guarded write is not definite)"
    (read_before_write o_b6 ell7);
  p "guarded-write splice: ell7 is an input of routine B"
    (let (ins, _), _ = LL.input_and_output_nodes o_b6 in
     Set.mem ins ell7);
  let ell7_old = Array.init dim ~f:(fun i -> 3. +. Float.of_int i) in
  let got_false =
    execute ~name:"vcf_guard_b" o_b6
      ~seed:[ (flag, [| 0. |]); (ell7, ell7_old); (out7, blank dim) ]
      ~read:[ out7; ell7 ]
  in
  p "guarded-write splice, false branch: the entry values are observed"
    (same got_false
       [
         Array.init dim ~f:(fun i -> if i < dim - 1 then ell7_old.(i + 1) +. 100. else sentinel);
         ell7_old;
       ]);
  let got_true =
    execute ~name:"vcf_guard_b_taken" o_b6
      ~seed:[ (flag, [| 1. |]); (ell7, ell7_old); (out7, blank dim) ]
      ~read:[ out7; ell7 ]
  in
  p "guarded-write splice, true branch: the overwrite is observed"
    (same got_true
       [
         Array.init dim ~f:(fun i -> if i < dim - 1 then 1099. else sentinel);
         Array.create ~len:dim 999.;
       ]);
  (* Review round 6, P1: a same-position spliced read is a genuine read-modify-write — the
     multiplicity analyses exempt it ([rmw_exempt]) but the INTERFACE must not: after a partial
     write, ell9[i] = v9[i] splices to ell9[i] = ell9[i] + 100, and the cells the partial write
     missed require their incoming values. *)
  let ell9 = mk "ell9" in
  materialize ell9;
  let v9 = mk "v9" in
  let ctx8 = LL.empty_optimize_ctx () in
  let llc_a8 =
    let s = sym () in
    loop_n s dim (set v9 [| iter s |] (add (get ell9 [| iter s |]) (c 100.)))
  in
  let o_a8 = optimize_in ctx8 ~name:"vcf_rmw_a" llc_a8 in
  p "same-position splice: routine A defers v9" (known_virtual o_a8 v9);
  let llc_b8 =
    let s = sym () in
    seq
      (set ell9 [| fixed 0 |] (c 5000.))
      (loop_n s dim (set ell9 [| iter s |] (get v9 [| iter s |])))
  in
  let o_b8 = optimize_in ctx8 ~name:"vcf_rmw_b" llc_b8 in
  p "same-position splice: ell9 is read-before-write (rmw is not exempt from the interface)"
    (read_before_write o_b8 ell9);
  let ell9_old = Array.init dim ~f:(fun i -> 5. +. Float.of_int i) in
  let got8 = execute ~name:"vcf_rmw_b" o_b8 ~seed:[ (ell9, ell9_old) ] ~read:[ ell9 ] in
  p "same-position splice: updated in place over the incoming values"
    (same got8 [ Array.init dim ~f:(fun i -> if i = 0 then 5100. else ell9_old.(i) +. 100.) ]);
  (* Review round 6, P2: a shared-loop stored body carries SIBLING setters that inlining filters out
     — a sibling's merge read must not taint the candidate. Routine A's one loop computes both sib
     (materialized, from the merge buffer) and v10 (virtual, from ell10). *)
  let msrc4 = mk "msrc4" in
  materialize msrc4;
  let sib = mk "sib" in
  materialize sib;
  let ell10 = mk "ell10" in
  materialize ell10;
  let v10 = mk "v10" and out10 = mk "shout" in
  materialize out10;
  let ctx9 = LL.empty_optimize_ctx () in
  let llc_a9 =
    let s = sym () in
    loop_n s dim
      (seq
         (set sib [| iter s |] (LL.Get_merge_buffer (msrc4, [| iter s |])))
         (set v10 [| iter s |] (add (get ell10 [| iter s |]) (c 100.))))
  in
  let o_a9 = optimize_in ctx9 ~name:"vcf_shared_a" llc_a9 in
  p "sibling merge: v10 stays virtual in the shared loop" (known_virtual o_a9 v10);
  let llc_b9 =
    let s = sym () in
    loop_n s dim (set out10 [| iter s |] (get v10 [| iter s |]))
  in
  let o_b9 = optimize_in ctx9 ~name:"vcf_shared_b" llc_b9 in
  let ell10_vals = Array.init dim ~f:(fun i -> 60. +. Float.of_int i) in
  let got9 =
    execute ~name:"vcf_shared_b" o_b9
      ~seed:[ (ell10, ell10_vals); (out10, blank dim) ]
      ~read:[ out10 ]
  in
  p "sibling merge: consuming v10 is legal, only its own setter's reads splice"
    (same got9 [ Array.map ell10_vals ~f:(fun x -> x +. 100.) ]);
  (* Review round 7, P1: consumption through an UPDATE of the inherited virtual (the consumer has a
     local assignment to it) still records the spliced leaf reads — the strictness key is what the
     inlined bodies read, not whether the consumed node lacks local setters. *)
  let ell12 = mk "ell12" in
  materialize ell12;
  let d12 = mk "d12" in
  materialize d12;
  let v12 = mk "v12" and out12 = mk "uout" in
  materialize out12;
  let ctx10 = LL.empty_optimize_ctx () in
  let llc_a10 =
    let s = sym () in
    loop_n s dim (set v12 [| iter s |] (add (get ell12 [| iter s |]) (c 100.)))
  in
  let o_a10 = optimize_in ctx10 ~name:"vcf_upd_a" llc_a10 in
  p "update splice: routine A defers v12" (known_virtual o_a10 v12);
  let llc_b10 =
    let s = sym () and s2 = sym () and s3 = sym () in
    List.reduce_exn ~f:seq
      [
        loop_n s dim (set v12 [| iter s |] (add (get v12 [| iter s |]) (get d12 [| iter s |])));
        loop_n s2 dim (set out12 [| iter s2 |] (get v12 [| iter s2 |]));
        loop_n s3 dim (set ell12 [| iter s3 |] (ramp 3000. s3));
      ]
  in
  let o_b10 = optimize_in ctx10 ~name:"vcf_upd_b" llc_b10 in
  p "update splice: ell12 is read-before-write despite the local v12 assignment"
    (read_before_write o_b10 ell12);
  let ell12_old = Array.init dim ~f:(fun i -> 9. +. Float.of_int i) in
  let d12_vals = Array.init dim ~f:(fun i -> 0.5 +. Float.of_int i) in
  let got10 =
    execute ~name:"vcf_upd_b" o_b10
      ~seed:[ (ell12, ell12_old); (d12, d12_vals); (out12, blank dim) ]
      ~read:[ out12; ell12 ]
  in
  p "update splice: out sees the pre-overwrite leaf through the updated computation"
    (same got10
       [
         Array.init dim ~f:(fun i -> ell12_old.(i) +. 100. +. d12_vals.(i));
         Array.init dim ~f:(fun i -> 3000. +. Float.of_int i);
       ]);
  (* Review round 7, P1: strictness is PER SPLICED NODE — a routine that consumes an inherited
     computation must not re-judge its unrelated raw reads. ew is written then read inside ONE [If]
     body (guards-taken coverage: the false branch performs neither access), so it must stay off the
     interface even though the routine splices v13 elsewhere. *)
  let gflag = mk ~dims:[| 1 |] "gflag" in
  materialize gflag;
  let ew = mk "ew" in
  materialize ew;
  let ell13 = mk "ell13" in
  materialize ell13;
  let v13 = mk "v13" and out13 = mk "ewout" and out14 = mk "vout14" in
  materialize out13;
  materialize out14;
  let ctx11 = LL.empty_optimize_ctx () in
  let llc_a11 =
    let s = sym () in
    loop_n s dim (set v13 [| iter s |] (add (get ell13 [| iter s |]) (c 100.)))
  in
  let o_a11 = optimize_in ctx11 ~name:"vcf_raw_a" llc_a11 in
  p "raw-read isolation: routine A defers v13" (known_virtual o_a11 v13);
  let llc_b11 =
    let sg = sym () and sh = sym () and s = sym () in
    seq
      (LL.If
         {
           cond = (get gflag [| fixed 0 |], single);
           body =
             seq
               (loop_n sg dim (set ew [| iter sg |] (c 7.)))
               (loop_n sh dim (set out13 [| iter sh |] (get ew [| iter sh |])));
         })
      (loop_n s dim (set out14 [| iter s |] (get v13 [| iter s |])))
  in
  let o_b11 = optimize_in ctx11 ~name:"vcf_raw_b" llc_b11 in
  p "raw-read isolation: ew keeps its raw guards-taken verdict (not an input)"
    ((not (read_before_write o_b11 ew))
    && (not (Set.mem o_b11.LL.spliced_rbw ew))
    &&
    let (ins, _), _ = LL.input_and_output_nodes o_b11 in
    not (Set.mem ins ew));
  let ell13_vals = Array.init dim ~f:(fun i -> 21. +. Float.of_int i) in
  let got11 =
    execute ~name:"vcf_raw_b" o_b11
      ~seed:
        [
          (gflag, [| 1. |]);
          (ell13, ell13_vals);
          (ew, blank dim);
          (out13, blank dim);
          (out14, blank dim);
        ]
      ~read:[ out13; out14 ]
  in
  p "raw-read isolation: executed values match on the taken branch"
    (same got11 [ Array.create ~len:dim 7.; Array.map ell13_vals ~f:(fun x -> x +. 100.) ]);
  (* Review round 8, P2: a discarded projection operand inside an INHERITED computation must not
     mark its node as spliced — extra3 reaches routine B only through Arg2's dead first operand, so
     B's own raw guarded-write-then-read of extra3 keeps its raw guards-taken verdict. *)
  let extra3 = mk "extra3" in
  materialize extra3;
  let ell14 = mk "ell14" in
  materialize ell14;
  let v14 = mk "v14" and out16 = mk "pjout" and out17 = mk "pjout2" in
  materialize out16;
  materialize out17;
  let ctx12 = LL.empty_optimize_ctx () in
  let llc_a12 =
    let s = sym () in
    loop_n s dim
      (set v14 [| iter s |] (binop Ir.Ops.Arg2 (get extra3 [| iter s |]) (get ell14 [| iter s |])))
  in
  let o_a12 = optimize_in ctx12 ~name:"vcf_proj2_a" llc_a12 in
  p "discarded-splice: routine A defers v14" (known_virtual o_a12 v14);
  let llc_b12 =
    let sg = sym () and sh = sym () and s = sym () in
    seq
      (LL.If
         {
           cond = (get gflag [| fixed 0 |], single);
           body =
             seq
               (loop_n sg dim (set extra3 [| iter sg |] (c 3.)))
               (loop_n sh dim (set out16 [| iter sh |] (get extra3 [| iter sh |])));
         })
      (loop_n s dim (set out17 [| iter s |] (get v14 [| iter s |])))
  in
  let o_b12 = optimize_in ctx12 ~name:"vcf_proj2_b" llc_b12 in
  p "discarded-splice: extra3 keeps its raw verdict (discarded operand did not splice it)"
    ((not (read_before_write o_b12 extra3)) && not (Set.mem o_b12.LL.spliced_rbw extra3));
  let ell14_vals = Array.init dim ~f:(fun i -> 33. +. Float.of_int i) in
  let got12 =
    execute ~name:"vcf_proj2_b" o_b12
      ~seed:
        [
          (gflag, [| 1. |]);
          (ell14, ell14_vals);
          (extra3, blank dim);
          (out16, blank dim);
          (out17, blank dim);
        ]
      ~read:[ out16; out17 ]
  in
  p "discarded-splice: executed values are the projected operand's"
    (same got12 [ Array.create ~len:dim 3.; ell14_vals ]);
  (* Review round 8, P2: a guarded write DOMINATING the spliced read supplies coverage — inside one
     [If] body the write precedes the consume, so the false path performs neither access and no
     entry value is required. Contrast vcf_guard above, where the read sits OUTSIDE the guard. *)
  let leaf15 = mk "leaf15" in
  materialize leaf15;
  let v15 = mk "v15" and out18 = mk "dgout" in
  materialize out18;
  let ctx13 = LL.empty_optimize_ctx () in
  let llc_a13 =
    let s = sym () in
    loop_n s dim (set v15 [| iter s |] (add (get leaf15 [| iter s |]) (c 100.)))
  in
  let o_a13 = optimize_in ctx13 ~name:"vcf_domg_a" llc_a13 in
  p "dominated-guard splice: routine A defers v15" (known_virtual o_a13 v15);
  let llc_b13 =
    let sg = sym () and sh = sym () in
    LL.If
      {
        cond = (get gflag [| fixed 0 |], single);
        body =
          seq
            (loop_n sg dim (set leaf15 [| iter sg |] (ramp 500. sg)))
            (loop_n sh dim (set out18 [| iter sh |] (get v15 [| iter sh |])));
      }
  in
  let o_b13 = optimize_in ctx13 ~name:"vcf_domg_b" llc_b13 in
  p "dominated-guard splice: the same-guard write covers, leaf15 is not an input"
    ((not (read_before_write o_b13 leaf15))
    && (not (Set.mem o_b13.LL.spliced_rbw leaf15))
    &&
    let (ins, _), _ = LL.input_and_output_nodes o_b13 in
    not (Set.mem ins leaf15));
  let got13 =
    execute ~name:"vcf_domg_b" o_b13
      ~seed:[ (gflag, [| 1. |]); (leaf15, blank dim); (out18, blank dim) ]
      ~read:[ out18; leaf15 ]
  in
  p "dominated-guard splice: taken branch writes then consumes"
    (same got13
       [
         Array.init dim ~f:(fun i -> 600. +. Float.of_int i);
         Array.init dim ~f:(fun i -> 500. +. Float.of_int i);
       ]);
  (* Review round 9, P2: the CONSUMER's own projection around an inherited virtual — the discarded
     operand is never evaluated, so a merge-tainted deferred node there must not reject the routine.
     virtual_llc collapses Arg1/Arg2 to the selected operand (mirroring simplify), so the discarded
     inline is never attempted; mv is the merge-tainted deferred node from the merge-splice pins
     above, on the same lineage. *)
  let ell16 = mk "ell16" in
  materialize ell16;
  let out19 = mk "prjout3" in
  materialize out19;
  let llc_b14 =
    let s = sym () in
    loop_n s dim
      (set out19 [| iter s |] (binop Ir.Ops.Arg2 (get mv [| iter s |]) (get ell16 [| iter s |])))
  in
  let o_b14 = optimize_in ctx2 ~name:"vcf_proj3_b" llc_b14 in
  let ell16_vals = Array.init dim ~f:(fun i -> 71. +. Float.of_int i) in
  let got14 =
    execute ~name:"vcf_proj3_b" o_b14
      ~seed:[ (ell16, ell16_vals); (out19, blank dim) ]
      ~read:[ out19 ]
  in
  p "consumer projection: tainted node in the discarded operand is legal, values project"
    (same got14 [ ell16_vals ]);
  (* Review round 9, P2: a merge read inside a DEAD loop mirrors the raw tracer's dead-body skip —
     it neither validates against the declared merge node nor keeps a declaration alive. The
     routine's live merge read of msrc5 stays the declared node while the dead read of msrc6 is
     ignored. *)
  let msrc5 = mk "msrc5" in
  materialize msrc5;
  let msrc6 = mk "msrc6" in
  materialize msrc6;
  let mtar2 = mk "mtar2" and mtar3 = mk "mtar3" in
  materialize mtar2;
  materialize mtar3;
  let ctx14 = LL.empty_optimize_ctx () in
  let llc_m5 =
    let s = sym () and sd = sym () in
    seq
      (loop_n s dim (set mtar2 [| iter s |] (LL.Get_merge_buffer (msrc5, [| iter s |]))))
      (loop ~upto:(-1) sd (set mtar3 [| iter sd |] (LL.Get_merge_buffer (msrc6, [| iter sd |]))))
  in
  let o_m5 = optimize_in ctx14 ~name:"vcf_merge_dead" llc_m5 in
  p "dead merge read: ignored, the live declaration stands"
    (match o_m5.LL.merge_node with Some m -> Tn.equal m msrc5 | None -> false);
  (* Review round 10, P2: a merge read only inside a DEAD loop of a stored computation must not
     taint it — the dead component replays zero times when inlined, mirroring the raw tracer. *)
  let msrc7 = mk "msrc7" in
  materialize msrc7;
  let ell17 = mk "ell17" in
  materialize ell17;
  let v16 = mk "v16" and out20 = mk "dtout" in
  materialize out20;
  let ctx15 = LL.empty_optimize_ctx () in
  let lv16 = mk "lv16" in
  virtualize lv16;
  let id16 = LL.get_scope lv16 in
  let llc_a15 =
    let s = sym () and sd = sym () in
    (* The dead merge read lives inside the setter's scope body — a dead sub-loop around a
       [Set_local] — so v16 keeps one consistent setter and stays a virtualization candidate. *)
    loop_n s dim
      (set v16
         [| iter s |]
         (LL.Local_scope
            {
              id = id16;
              body =
                seq
                  (loop ~upto:(-1) sd
                     (LL.Set_local (id16, LL.Get_merge_buffer (msrc7, [| iter sd |]))))
                  (LL.Set_local (id16, add (get ell17 [| iter s |]) (c 100.)));
              orig_indices = [| iter s |];
              mint = LL.Inlined_computation;
            }))
  in
  let o_a15 = optimize_in ctx15 ~name:"vcf_dtaint_a" llc_a15 in
  p "dead-taint: routine A defers v16" (known_virtual o_a15 v16);
  let llc_b15 =
    let s = sym () in
    loop_n s dim (set out20 [| iter s |] (get v16 [| iter s |]))
  in
  let o_b15 = optimize_in ctx15 ~name:"vcf_dtaint_b" llc_b15 in
  let ell17_vals = Array.init dim ~f:(fun i -> 84. +. Float.of_int i) in
  let got15 =
    execute ~name:"vcf_dtaint_b" o_b15
      ~seed:[ (ell17, ell17_vals); (out20, blank dim) ]
      ~read:[ out20 ]
  in
  p "dead-taint: consuming is legal, the live component's values splice"
    (same got15 [ Array.map ell17_vals ~f:(fun x -> x +. 100.) ]);
  (* Review round 11, P2: when an INHERITED computation cannot be inlined at the consumer's read
     site (a fixed-cell producer read across a loop trips Non_virtual 13), the local-producer
     fallback — a materialized Get — is unavailable: the deferring routine already dropped the
     setters, so no routine writes the buffer. The consumer fails actionably instead. *)
  let ell18 = mk "ell18" in
  materialize ell18;
  let v17 = mk "v17" and out21 = mk "flout" in
  materialize out21;
  let ctx16 = LL.empty_optimize_ctx () in
  let llc_a16 = set v17 [| fixed 0 |] (add (get ell18 [| fixed 0 |]) (c 100.)) in
  let o_a16 = optimize_in ctx16 ~name:"vcf_fail_a" llc_a16 in
  p "failed-inline: routine A defers v17" (known_virtual o_a16 v17);
  let llc_b16 =
    let s = sym () in
    loop_n s dim (set out21 [| iter s |] (get v17 [| iter s |]))
  in
  let rejected_fail =
    try
      ignore (optimize_in ctx16 ~name:"vcf_fail_b" llc_b16 : LL.optimized);
      false
    with Utils.User_error msg -> String.is_substring msg ~substring:"could not be inlined"
  in
  p "failed-inline: consuming at an un-inlinable site fails actionably" rejected_fail;
  (* Review round 12, P2: an INHERITED node read as a dynamic-gather table — the Get_dynamic path
     never routes through inline_computation, and a dynamic read cannot be served by recomputation;
     pre-fix this died in cleanup's cryptic already-virtual collision. *)
  let ell21 = mk "ell21" in
  materialize ell21;
  let sel = mk ~dims:[| 1 |] "sel" in
  materialize sel;
  let v18 = mk "v18" and out22 = mk "gdout" in
  materialize out22;
  let ctx17 = LL.empty_optimize_ctx () in
  let llc_a17 =
    let s = sym () in
    loop_n s dim (set v18 [| iter s |] (add (get ell21 [| iter s |]) (c 100.)))
  in
  let o_a17 = optimize_in ctx17 ~name:"vcf_gd_a" llc_a17 in
  p "dynamic-table: routine A defers v18" (known_virtual o_a17 v18);
  let llc_b17 =
    let s = sym () in
    loop_n s dim
      (set out22
         [| iter s |]
         (LL.Get_dynamic
            {
              tn = v18;
              idcs = [| fixed 0 |];
              dyn_axis = 0;
              dyn_value = (get sel [| fixed 0 |], single);
            }))
  in
  let rejected_gd =
    try
      ignore (optimize_in ctx17 ~name:"vcf_gd_b" llc_b17 : LL.optimized);
      false
    with Utils.User_error msg -> String.is_substring msg ~substring:"dynamic-gather"
  in
  p "dynamic-table: inherited table is rejected actionably" rejected_gd;
  (* Review round 13, P2: a consumer's reference to a merge-tainted inherited node ONLY inside a
     dead loop must not reject the routine — dead loops are dropped at virtualization before any
     splice is attempted. mv is the tainted node from the merge-splice pins, same lineage. *)
  let ell22 = mk "ell22" in
  materialize ell22;
  let out23 = mk "dcout" in
  materialize out23;
  let llc_b18 =
    let sd = sym () and s = sym () in
    seq
      (loop ~upto:(-1) sd (set out23 [| iter sd |] (get mv [| iter sd |])))
      (loop_n s dim (set out23 [| iter s |] (get ell22 [| iter s |])))
  in
  let o_b18 = optimize_in ctx2 ~name:"vcf_deadcons_b" llc_b18 in
  let ell22_vals = Array.init dim ~f:(fun i -> 91. +. Float.of_int i) in
  let got18 =
    execute ~name:"vcf_deadcons_b" o_b18
      ~seed:[ (ell22, ell22_vals); (out23, blank dim) ]
      ~read:[ out23 ]
  in
  p "dead-consumer splice: tainted reference in dead code is legal, live values flow"
    (same got18 [ ell22_vals ])

(* === Phase 6: deferral-only comp through the ordinary pipeline (review round 3) === *)

let phase6 () =
  (* P1: [from_prior_context] is derived from the raw assignments, which over-approximate what the
     residual schedule needs — pre-fix, a deferral-only comp failed [verify_prior_context] on a
     fresh context, demanding leaves only its DEFERRED computations read. It is now filtered by the
     reconciled traced store: the routine links and runs as a no-op. *)
  Utils.settings.fixed_state_for_init <- Some 42;
  Tensor.unsafe_reinitialize ();
  let base =
    NTDSL.init ~l:"vcf_p6base" ~prec:Ir.Ops.single ~o:[ dim ]
      ~f:(fun idcs -> Float.of_int (10 * (idcs.(0) + 1)))
      ()
  in
  let w2 = TDSL.O.( + ) base base in
  (* Routine 1 computes w2 (materialized by [Train.forward]) on its own context chain, consuming
     w2's forward — the deferral comp below then references w2 as a context-sourced node rather than
     embedding it. *)
  let ctx_w = Context.auto () in
  let ctx_w, routine_w = Context.compile ctx_w (Train.forward w2) Ir.Indexing.Empty in
  let _ctx_w = Context.run ctx_w routine_w in
  let v6 = TDSL.O.( + ) w2 w2 in
  let comp_defer = Tensor.consume_forward_code v6 in
  let ctx_fresh = Context.auto () in
  let ctx_fresh, routine_defer =
    Context.compile ~name:"vcf_p6_defer" ctx_fresh comp_defer Ir.Indexing.Empty
  in
  let _ctx_ran = Context.run ctx_fresh routine_defer in
  p "pipeline deferral-only: links and runs as a no-op on a fresh context" true;
  p "pipeline deferral-only: the target stayed virtual"
    (Tn.Placements.known_virtual (Context.placements ctx_fresh) v6.Tensor.value);
  (* Review round 5, P1: the CONSUMER on this w2-less chain must be rejected at link time — its
     spliced input w2 now enters from_prior_context, where pre-fix it silently computed with a
     zero-filled fresh allocation. *)
  let out6 = TDSL.O.( + ) v6 v6 in
  let rejected =
    try
      let _ctx, _routine = Context.compile ctx_fresh (Train.forward out6) Ir.Indexing.Empty in
      false
    with Utils.User_error msg -> String.is_substring msg ~substring:"lacks node"
  in
  p "pipeline consumer on a w2-less chain: rejected at link (spliced input demanded)" rejected;
  (* The positive arc on ONE chain — compute the leaf, defer, consume — matches the reference: the
     ordinary-pipeline twin of phase 4's prelowered flow. *)
  Tensor.unsafe_reinitialize ();
  let base2 =
    NTDSL.init ~l:"vcf_p6base2" ~prec:Ir.Ops.single ~o:[ dim ]
      ~f:(fun idcs -> Float.of_int (10 * (idcs.(0) + 1)))
      ()
  in
  let w3 = TDSL.O.( + ) base2 base2 in
  let ctx1 = Context.auto () in
  let ctx1, r_w = Context.compile ctx1 (Train.forward w3) Ir.Indexing.Empty in
  let ctx1 = Context.run ctx1 r_w in
  let v8 = TDSL.O.( + ) w3 w3 in
  let ctx1, r_defer =
    Context.compile ~name:"vcf_p6_defer2" ctx1 (Tensor.consume_forward_code v8) Ir.Indexing.Empty
  in
  let ctx1 = Context.run ctx1 r_defer in
  let out8 = TDSL.O.( + ) v8 v8 in
  let ctx1, r_out = Context.compile ctx1 (Train.forward out8) Ir.Indexing.Empty in
  let ctx1 = Context.run ctx1 r_out in
  let got = Context.get_values ctx1 out8.Tensor.value in
  let expected = Array.init dim ~f:(fun i -> 80. *. Float.of_int (i + 1)) in
  p "pipeline incremental arc: compute leaf, defer, consume — matches the reference"
    (close got expected);
  (* Review round 6, P1: a consumer that WRITES a spliced leaf after consuming mentions the leaf
     only as an (embedded) write — the mention filter must not shield it from the prior-context
     demand, because the splice reads the leaf's ENTRY value before the overwrite. Hand-composed out
     of order (topo-sorted tensor forwards always init before consumers), with the fetch marked
     embedded so the raw set excludes the leaf. *)
  Tensor.unsafe_reinitialize ();
  let base3 =
    NTDSL.init ~l:"vcf_p6base3" ~prec:Ir.Ops.single ~o:[ dim ]
      ~f:(fun idcs -> Float.of_int (10 * (idcs.(0) + 1)))
      ()
  in
  let w4 = TDSL.O.( + ) base3 base3 in
  let ctx_w4 = Context.auto () in
  let ctx_w4, r_w4 = Context.compile ctx_w4 (Train.forward w4) Ir.Indexing.Empty in
  let _ctx_w4 = Context.run ctx_w4 r_w4 in
  let v11 = TDSL.O.( + ) w4 w4 in
  let ctx_f2 = Context.auto () in
  let ctx_f2, r_d2 =
    Context.compile ~name:"vcf_p6_defer3" ctx_f2 (Tensor.consume_forward_code v11) Ir.Indexing.Empty
  in
  let _ctx_d2 = Context.run ctx_f2 r_d2 in
  let out11 = TDSL.O.( + ) v11 v11 in
  Train.set_materialized out11.Tensor.value;
  let overwrite_w4 : Ir.Assignments.comp =
    {
      asgns =
        Ir.Assignments.Fetch
          { array = w4.Tensor.value; fetch_op = Ir.Assignments.Constant 0.; dims = lazy [| dim |] };
      embedded_nodes = Set.singleton (module Tn) w4.Tensor.value;
    }
  in
  let comp_consume_then_write =
    Ir.Assignments.sequence [ Tensor.consume_forward_code out11; overwrite_w4 ]
  in
  let rejected_w =
    try
      let _c, _r =
        Context.compile ~name:"vcf_p6_overwrite" ctx_f2 comp_consume_then_write Ir.Indexing.Empty
      in
      false
    with Utils.User_error msg -> String.is_substring msg ~substring:"lacks node"
  in
  p "pipeline write-mentioned spliced leaf: demanded despite the embedded-write mention" rejected_w

let () =
  phase1 ();
  phase1b ();
  phase2 ();
  phase3 ();
  phase4 ();
  phase5 ();
  phase6 ()

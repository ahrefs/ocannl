(* gh-617 + gh-618: the splice-semantics design pair surfaced by review of the gh-610/611 fixes.

   Phase 1 (gh-618) pins the raw-side split of the read-modify-write exemption by consumer. A read
   at its enclosing statement's write position is exempt from the visit-counting placement
   heuristics ([rmw_exempt] -- the statement's own store is not a visit, and virtual_chain_fanin
   relies on it for residual chains), but for a node that owns a buffer it is NOT interface
   coverage: in [ell[0] = 5000; out[i] = ell[i]] the copy-position reads consume ell's incoming
   cells 1.., and an accumulation with no preceding definite initialization consumes its own entry
   values. Both must classify [read_before_write] -- a routine input, excluded from buffer aliasing
   -- where pre-split they read as output-only. The strict classification closes over the SETTLED
   placements ([reconcile_traced_store], review round 1): a node that stays virtual is exempt by
   construction -- no interface, and exemption-dependent coverage is exactly the shape of the
   virtualizer's partial-write producers (affine_lowering.ml AC6) -- while a node decided
   non-virtual AFTER [decide_placements] (a [check_and_store_virtual] legality rejection, the fan-in
   guard) is still reached. Controls pin the boundaries: coverage by a real prior write still
   suppresses the flag (routine-complete lowered flows emit the initialization before accumulations,
   so they are unaffected), and a GUARDED full write still counts as raw coverage (the guards-taken
   contract; guard strictness is splice-only).

   Phase 2 (gh-617, decided as option 1) pins recompute-at-read as the semantics of deferred
   (virtual) computations: a virtual node is a named computation, not a snapshot -- inlining
   evaluates it at the consumption site with whatever its materialized inputs hold AT THAT MOMENT.
   At the arrayjit level the recompute-vs-materialize semantics is deliberately not fixed; the
   executed legs pin both readings of one program text: the virtual arm observes the consumer's (or
   an intervening routine's) overwrite of the leaf, while the materialized twin snapshots the leaf
   as of the deferring routine's execution. The two knobs users have -- the memory-mode intent and
   the choice of routine boundaries (routine execution is manual) -- are exactly what the arms
   toggle. See "Recompute-at-read" in docs/lowering_and_inlining.md. *)

open Base
open Ll_test
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode

let dim = 4

let inputs (o : LL.optimized) =
  let (ins, _), _ = LL.input_and_output_nodes o in
  ins

(* === Phase 1: gh-618 -- the rmw exemption is not interface coverage (raw reads) === *)

let phase1 () =
  let mk = node_factory ~first_id:3600 ~dims:[| dim |] () in
  (* Copy-position reads of a DIFFERENT node after a partial write: the exemption matches on the
     index map alone, whichever nodes are involved, so pre-split ell read as covered and classified
     output-only -- its incoming cells 1.. feed out yet the interface said the entry value was
     ignorable (aliasing-eligible, dropped from link-time input verification). *)
  let ell = mk "ell" in
  materialize ell;
  let out = mk "cpout" in
  materialize out;
  let llc_copy =
    let s = sym () in
    seq
      (set ell [| fixed 0 |] (c 5000.))
      (loop_n s dim (set out [| iter s |] (get ell [| iter s |])))
  in
  let o_copy = optimize ~name:"ssem_copypos" llc_copy in
  p "copy-position raw reads after a partial write: ell is read-before-write"
    (read_before_write o_copy ell);
  p "copy-position raw reads: ell is an input of the routine" (Set.mem (inputs o_copy) ell);
  let ell_old = Array.init dim ~f:(fun i -> 11. +. Float.of_int i) in
  let got =
    execute ~name:"ssem_copypos" o_copy ~seed:[ (ell, ell_old); (out, blank dim) ] ~read:[ out ]
  in
  p "copy-position raw reads: the uncovered cells read the incoming values"
    (same got [ Array.init dim ~f:(fun i -> if i = 0 then 5000. else ell_old.(i)) ]);
  (* True read-modify-write with no preceding initialization: x[i] = x[i] + a[i] consumes x's entry
     values, so x is an input -- the tracer-mirroring exemption must not classify it output-only. *)
  let x = mk "x" in
  materialize x;
  let a = mk "a" in
  materialize a;
  let llc_rmw =
    let s = sym () in
    loop_n s dim (set x [| iter s |] (add (get x [| iter s |]) (get a [| iter s |])))
  in
  let o_rmw = optimize ~name:"ssem_rmw" llc_rmw in
  p "uninitialized accumulation: x is read-before-write" (read_before_write o_rmw x);
  p "uninitialized accumulation: x is an input of the routine" (Set.mem (inputs o_rmw) x);
  let x_old = Array.init dim ~f:(fun i -> 21. +. Float.of_int i) in
  let a_vals = Array.init dim ~f:(fun i -> 100. *. (1. +. Float.of_int i)) in
  let got_rmw = execute ~name:"ssem_rmw" o_rmw ~seed:[ (x, x_old); (a, a_vals) ] ~read:[ x ] in
  p "uninitialized accumulation: updated in place over the incoming values"
    (same got_rmw [ Array.init dim ~f:(fun i -> x_old.(i) +. a_vals.(i)) ]);
  (* Control: coverage by a real prior write still suppresses the flag -- the strict interface
     verdict must not overreach onto routine-complete accumulations, whose lowering emits the
     initialization first. *)
  let y = mk "y" in
  materialize y;
  let b = mk "b" in
  materialize b;
  let llc_ctrl =
    let s = sym () in
    seq (zero y) (loop_n s dim (set y [| iter s |] (add (get y [| iter s |]) (get b [| iter s |]))))
  in
  let o_ctrl = optimize ~name:"ssem_zeroed" llc_ctrl in
  p "zero-initialized accumulation: covered by the real write, not read-before-write"
    (not (read_before_write o_ctrl y));
  p "zero-initialized accumulation: y is not an input" (not (Set.mem (inputs o_ctrl) y));
  let b_vals = Array.init dim ~f:(fun i -> 31. +. Float.of_int i) in
  let got_ctrl =
    execute ~name:"ssem_zeroed" o_ctrl ~seed:[ (y, blank dim); (b, b_vals) ] ~read:[ y ]
  in
  p "zero-initialized accumulation: the sentinel seed is overwritten" (same got_ctrl [ b_vals ]);
  (* Placement-freedom control: an UNDECIDED partial-write producer consumed at the copy position --
     the triangular-scatter genre of affine_lowering.ml's AC6 (injective, so no neutral init is
     emitted; off-region cells fall back to the init the INLINING prepends) -- must keep its
     virtualization eligibility: a virtual node has no interface, so the strict verdict binds only
     once a node is known non-virtual. The materialized reading of this genre is ell above; AC6
     executes both readings. *)
  let z = mk ~dims:[| 3; 4 |] "z" in
  let src = mk ~dims:[| 3; 2 |] "zsrc" in
  materialize src;
  let zout = mk ~dims:[| 3; 4 |] "zout" in
  materialize zout;
  let llc_und =
    let s1 = sym () and s2 = sym () and a = sym () and b = sym () in
    seq
      (loop_n s1 3
         (loop_n s2 2
            (set z [| iter s1; aff [ (1, s1); (1, s2) ] 0 |] (get src [| iter s1; iter s2 |]))))
      (loop_n a 3 (loop_n b 4 (set zout [| iter a; iter b |] (get z [| iter a; iter b |]))))
  in
  let o_und = optimize ~name:"ssem_undecided" llc_und in
  p "undecided partial-write producer: keeps its virtualization eligibility" (known_virtual o_und z);
  (* Late placement decision (review round 1, P1): an UNDECIDED node can become non-virtual AFTER
     [decide_placements] -- here [check_and_store_virtual] rejects the non-injective multi-affine
     scatter map [s1+s2] during [virtual_llc] ([Non_virtual 51]) -- and the strict verdict must
     still reach it: the classification closes over the settled placements in
     [reconcile_traced_store]. The scatter writes cells 0..3 of a 6-cell node, so the consumer's
     copy-position reads of cells 4..5 are exemption-dependent, and a decide-time-only strict
     verdict (which sees the node still undecided) would classify x2 output-only. The flip also
     promotes x2 [On_device] (review round 3): a late-rejected candidate is otherwise only
     [Never_virtual], which would default to [Local] -- routine scratch with no incoming contents,
     contradicting the entry values the reads consume. The executed leg is the pin: it seeds x2 and
     reads back the preserved cells, which a [Local] placement cannot even seed. *)
  let x2 = mk ~dims:[| 6 |] "x2" in
  Tn.set_observable x2;
  let a2 = mk ~dims:[| 3; 2 |] "a2" in
  materialize a2;
  let out2 = mk ~dims:[| 6 |] "lateout" in
  materialize out2;
  let llc_late =
    let s1 = sym () and s2 = sym () and t = sym () in
    seq
      (loop_n s1 3
         (loop_n s2 2 (set x2 [| aff [ (1, s1); (1, s2) ] 0 |] (get a2 [| iter s1; iter s2 |]))))
      (loop_n t 6 (set out2 [| iter t |] (get x2 [| iter t |])))
  in
  let o_late = optimize ~name:"ssem_late" llc_late in
  p "non-injective scatter, undecided: not virtualized" (not (known_virtual o_late x2));
  p "non-injective scatter, undecided: read-before-write despite deciding after the placement pass"
    (read_before_write o_late x2);
  p "non-injective scatter, undecided: an input of the routine" (Set.mem (inputs o_late) x2);
  (* Last-writer-wins over the s1-major, s2-minor iteration: cell c holds a2[s1_max, c - s1_max]
     with s1_max = min (c, 2); cells 4..5 keep the seed. *)
  let x2_old = Array.init 6 ~f:(fun i -> 61. +. Float.of_int i) in
  let a2_vals = [| 11.; 12.; 21.; 22.; 31.; 32. |] in
  let got_late =
    execute ~name:"ssem_late" o_late
      ~seed:[ (x2, x2_old); (a2, a2_vals); (out2, blank 6) ]
      ~read:[ out2 ]
  in
  p "non-injective scatter, undecided: uncovered cells read the incoming values"
    (same got_late [ [| 11.; 21.; 31.; 32.; x2_old.(4); x2_old.(5) |] ]);
  (* The UNOBSERVABLE twin pins the promotion itself (x2's declared observability would rescue its
     resolution on its own): without the [On_device] promotion the late-rejected candidate stays
     [Never_virtual], resolves to [Local] scratch, and the interface drops an input whose entry
     values the kernel consumes. *)
  let x3 = mk ~dims:[| 6 |] "x3" in
  let a3 = mk ~dims:[| 3; 2 |] "a3" in
  materialize a3;
  let out3 = mk ~dims:[| 6 |] "lateout2" in
  materialize out3;
  let llc_late2 =
    let s1 = sym () and s2 = sym () and t = sym () in
    seq
      (loop_n s1 3
         (loop_n s2 2 (set x3 [| aff [ (1, s1); (1, s2) ] 0 |] (get a3 [| iter s1; iter s2 |]))))
      (loop_n t 6 (set out3 [| iter t |] (get x3 [| iter t |])))
  in
  let o_late2 = optimize ~name:"ssem_late2" llc_late2 in
  p "unobservable late input: promoted to a persistent buffer (an interface input, not scratch)"
    (Set.mem (inputs o_late2) x3);
  (* Guards-taken control: the closing pass judges raw-positioned reads with the raw contract's
     query, so a GUARDED full write still counts as coverage (round 6 of the gh-610/611 PR:
     routine-wide guard strictness broke flows whose initialization runs under a flag; the
     guard-filtered strict query is for spliced reads only). *)
  let flag = mk ~dims:[| 1 |] "flag" in
  materialize flag;
  let xg = mk "xg" in
  materialize xg;
  let outg = mk "guardout" in
  materialize outg;
  let llc_guard =
    let s = sym () and t = sym () in
    seq
      (LL.If
         {
           cond = (get flag [| fixed 0 |], single);
           body = loop_n s dim (set xg [| iter s |] (ramp 100. s));
         })
      (loop_n t dim (set outg [| iter t |] (get xg [| iter t |])))
  in
  let o_guard = optimize ~name:"ssem_guardcov" llc_guard in
  p "guarded full write: raw guards-taken coverage keeps xg off read-before-write"
    (not (read_before_write o_guard xg));
  (* Cap-selected exemption-dependent producer (review round 4): the reduction cap materializes r
     (Never_virtual 39) before the closing pass promotes it On_device -- the promotion must not cost
     r its [`Inline] flip candidacy (a virtual reading has no interface to classify, so the
     placement search remains free to try it), and the interface must say input. *)
  assert (LL.virtualize_settings.max_inline_reduction = 16);
  let r = mk "r" in
  Tn.set_observable r;
  let ar = mk ~dims:[| 2; 20 |] "ar" in
  materialize ar;
  let outr = mk "capout" in
  materialize outr;
  let llc_cap =
    let s = sym () and k = sym () and t = sym () in
    seq
      (loop_n s 2
         (loop_n k 20
            (set r
               [| aff [ (2, s) ] 0 |]
               (add (get r [| aff [ (2, s) ] 0 |]) (get ar [| iter s; iter k |])))))
      (loop_n t dim (set outr [| iter t |] (get r [| iter t |])))
  in
  let o_cap = optimize ~name:"ssem_cap" llc_cap in
  p "cap-selected accumulator: read-before-write and an input"
    (read_before_write o_cap r && Set.mem (inputs o_cap) r);
  p "cap-selected accumulator: keeps its Inline flip candidacy"
    (List.exists o_cap.LL.flip_candidates ~f:(fun fc ->
         Tn.equal fc.LL.fc_tn r
         && List.exists fc.LL.fc_alternatives ~f:(fun fa -> LL.equal_reading fa.LL.fa_flip `Inline)));
  let r_seed = [| 71.; 72.; 73.; 74. |] in
  let ar_vals =
    Array.init 40 ~f:(fun i ->
        let s = i / 20 and k = i % 20 in
        (10. *. Float.of_int (s + 1)) +. Float.of_int (k + 1))
  in
  let got_cap =
    execute ~name:"ssem_cap" o_cap
      ~seed:[ (r, r_seed); (ar, ar_vals); (outr, blank dim) ]
      ~read:[ outr ]
  in
  (* Row sums: 20*10*(s+1) + 210; even cells accumulate over the seed, odd cells pass through. *)
  p "cap-selected accumulator: accumulates over the incoming values"
    (same got_cap [ [| 481.; 72.; 683.; 74. |] ]);
  (* Inherited-Local rejection (gh-631): an earlier routine of the lineage commits tmp as
     routine-local scratch. [link_finalized] takes that producer through real backend finalization,
     and [known_local] excludes the Virtual arm that [known_not_materialized] would also accept. A
     later routine's in-place update then consumes entry values a scratch buffer does not carry, so
     the guard is executed and rejects it with the materialize-before-first-use User_error -- not an
     internal placement-transition failure. *)
  let ctx4 = LL.empty_optimize_ctx () in
  let tmp = mk "tmp" in
  let a4 = mk "a4" in
  materialize a4;
  let outa = mk ~dims:[| dim - 1 |] "scrout" in
  materialize outa;
  let llc_a4 =
    let s = sym () and t = sym () in
    seq
      (loop_n s dim (set tmp [| iter s |] (add (get a4 [| iter s |]) (c 1000.))))
      (loop_n t (dim - 1)
         (set outa
            [| iter t |]
            (add (get tmp [| aff [ (1, t) ] 1 |]) (get tmp [| aff [ (1, t) ] 1 |]))))
  in
  let o_a4 = optimize_in ctx4 ~name:"ssem_scratch_a" llc_a4 in
  let a4_vals = Array.init dim ~f:(fun i -> 81. +. Float.of_int i) in
  let cctx4 = Context.auto () in
  let linked_a4 = link_finalized ~ctx:cctx4 ~placements:[ tmp ] ~name:"ssem_scratch_a" o_a4 in
  let _cctx4 = run_linked linked_a4 ~seed:[ (a4, a4_vals) ] in
  p "scratch producer: tmp resolved to routine-local scratch" (known_local o_a4 tmp);
  let b4 = mk "b4" in
  materialize b4;
  let llc_b4 =
    let s = sym () in
    loop_n s dim (set tmp [| iter s |] (add (get tmp [| iter s |]) (get b4 [| iter s |])))
  in
  let rejected_scratch =
    try
      ignore (optimize_in ctx4 ~name:"ssem_scratch_b" llc_b4 : LL.optimized);
      false
    with Utils.User_error msg ->
      String.is_substring msg ~substring:"routine-local scratch"
      && String.is_substring msg ~substring:"materialized"
  in
  p "in-place update of an earlier routine's Local scratch: materialize-first User_error"
    rejected_scratch

(* === Phase 2: gh-617 -- recompute-at-read, and the knobs that change the observation === *)

let phase2 () =
  let mk = node_factory ~first_id:3700 ~dims:[| dim |] () in
  (* Virtual arm, consumer's own overwrite: routine A defers v := ell + 100 (deferral-only, so A
     optimizes away); routine B overwrites ell in full and THEN consumes v. The splice evaluates
     f(ell) at the read -- B's new values -- and the full overwrite covers the spliced reads, so ell
     is not even an input of B. *)
  let ell = mk "ell2" in
  materialize ell;
  let v = mk "v" in
  let out = mk "rarout" in
  materialize out;
  let ctx = LL.empty_optimize_ctx () in
  let llc_a =
    let s = sym () in
    loop_n s dim (set v [| iter s |] (add (get ell [| iter s |]) (c 100.)))
  in
  let o_a = optimize_in ctx ~name:"ssem_rar_a" llc_a in
  p "recompute-at-read: routine A defers v" (known_virtual o_a v);
  let llc_b =
    let s = sym () and s2 = sym () in
    seq
      (loop_n s dim (set ell [| iter s |] (ramp 2000. s)))
      (loop_n s2 dim (set out [| iter s2 |] (get v [| iter s2 |])))
  in
  let o_b = optimize_in ctx ~name:"ssem_rar_b" llc_b in
  p "overwrite-then-consume: the full overwrite covers the spliced reads -- ell is not an input"
    ((not (read_before_write o_b ell)) && not (Set.mem (inputs o_b) ell));
  let ell_old = Array.init dim ~f:(fun i -> 41. +. Float.of_int i) in
  let got =
    execute ~name:"ssem_rar_b" o_b ~seed:[ (ell, ell_old); (out, blank dim) ] ~read:[ out; ell ]
  in
  p "overwrite-then-consume: the splice observes the NEW ell (recompute-at-read)"
    (same got
       [
         Array.init dim ~f:(fun i -> 2100. +. Float.of_int i);
         Array.init dim ~f:(fun i -> 2000. +. Float.of_int i);
       ]);
  (* Virtual arm, intervening routine: A defers v2 := ell3 + 100; routine C (between A and the
     consumer, sharing the lineage and the execution context) overwrites ell3; routine B consumes v2
     and observes C's values -- the deferred computation is index-parametric code, evaluated where
     it is read. *)
  let ell3 = mk "ell3" in
  materialize ell3;
  let v2 = mk "v2" in
  let out2 = mk "interout" in
  materialize out2;
  let ctx2 = LL.empty_optimize_ctx () in
  let llc_a2 =
    let s = sym () in
    loop_n s dim (set v2 [| iter s |] (add (get ell3 [| iter s |]) (c 100.)))
  in
  let o_a2 = optimize_in ctx2 ~name:"ssem_inter_a" llc_a2 in
  p "intervening write: routine A defers v2" (known_virtual o_a2 v2);
  let llc_c =
    let s = sym () in
    loop_n s dim (set ell3 [| iter s |] (ramp 3000. s))
  in
  let o_c = optimize_in ctx2 ~name:"ssem_inter_c" llc_c in
  let llc_b2 =
    let s = sym () in
    loop_n s dim (set out2 [| iter s |] (get v2 [| iter s |]))
  in
  let o_b2 = optimize_in ctx2 ~name:"ssem_inter_b" llc_b2 in
  p "intervening write: ell3 is an input of the consuming routine" (Set.mem (inputs o_b2) ell3);
  let ell3_old = Array.init dim ~f:(fun i -> 51. +. Float.of_int i) in
  let cctx = Context.auto () in
  let cctx = run ~ctx:cctx ~name:"ssem_inter_c" o_c ~seed:[ (ell3, ell3_old) ] in
  let got2 =
    execute ~ctx:cctx ~name:"ssem_inter_b" o_b2 ~seed:[ (out2, blank dim) ] ~read:[ out2 ]
  in
  p "intervening write: the consumer observes the intervening routine's values"
    (same got2 [ Array.init dim ~f:(fun i -> 3100. +. Float.of_int i) ]);
  (* Materialized twin -- the memory-mode-intent knob: the SAME program text with v3 declared
     materialized computes it in routine A, so the consumer observes the snapshot of ell4 as of A's
     execution, unaffected by B's overwrite. Placement selects which reading -- both are legal,
     which is exactly the option-1 stance: arrayjit does not fix the semantics. *)
  let ell4 = mk "ell4" in
  materialize ell4;
  let v3 = mk "v3" in
  materialize v3;
  let out3 = mk "snapout" in
  materialize out3;
  let ctx3 = LL.empty_optimize_ctx () in
  let llc_a3 =
    let s = sym () in
    loop_n s dim (set v3 [| iter s |] (add (get ell4 [| iter s |]) (c 100.)))
  in
  let o_a3 = optimize_in ctx3 ~name:"ssem_snap_a" llc_a3 in
  p "materialized twin: v3 stays non-virtual" (known_non_virtual o_a3 v3);
  let llc_b3 =
    let s = sym () and s2 = sym () in
    seq
      (loop_n s dim (set ell4 [| iter s |] (ramp 4000. s)))
      (loop_n s2 dim (set out3 [| iter s2 |] (get v3 [| iter s2 |])))
  in
  let o_b3 = optimize_in ctx3 ~name:"ssem_snap_b" llc_b3 in
  let ell4_old = Array.init dim ~f:(fun i -> 61. +. Float.of_int i) in
  let cctx3 = Context.auto () in
  let cctx3 = run ~ctx:cctx3 ~name:"ssem_snap_a" o_a3 ~seed:[ (ell4, ell4_old) ] in
  let got3 =
    execute ~ctx:cctx3 ~name:"ssem_snap_b" o_b3 ~seed:[ (out3, blank dim) ] ~read:[ out3; ell4 ]
  in
  p "materialized twin: the consumer observes the deferring routine's snapshot"
    (same got3
       [
         Array.map ell4_old ~f:(fun e -> e +. 100.);
         Array.init dim ~f:(fun i -> 4000. +. Float.of_int i);
       ])

let () =
  phase1 ();
  phase2 ()

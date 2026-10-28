(* Refusals at the Context API's entry points.

   gh-ocannl-1329: a context [Context.release] freed is a dead handle. Compile, run (on the released
   value, or on a live value outside its upload lineage with a routine compiled into it), sync, the
   transfers and the decision recorders all refuse it with a typed [Utils.User_error] naming the
   release, rather than reaching a freed pool. The release flag is shared exactly by the values that
   share pools: an earlier value of the same linear upload chain is refused too, while a compile
   sibling, the parent and an upload fork keep working. gh-ocannl-1462: so is a context one of whose
   lifecycle ancestors was released, while a child of a surviving upload fork keeps working.

   gh-ocannl-1339: [Context.run] refuses a READ-BEFORE-WRITE input that nothing ever wrote. The
   issue's case is a counter a routine reads and then overwrites, run with the counter never seeded:
   before, the input check accepted any node that merely had a buffer, so the routine read
   allocation-time zeros. It now has to have been written in the lineage -- by a run, an upload or a
   device copy -- and a caller that dropped the context [run] returned keeps that fact. A READ-ONLY
   input must be written the same way unless [Tnode.mark_zero_valid] declares zero its correct value
   (gh-ocannl-1462). gh-ocannl-1654: a node every routine only reads is a constant cached per root,
   so independently created contexts hold their own buffer for it; only a host-initialized literal
   is one device-wide buffer, and a write one root makes there is refused as stale to the others'
   routines until they re-seed or copy (gh-ocannl-1623). (The tensor-level form of the mistake -- an
   SGD step, which reads each parameter before writing it, compiled without [Train.init_params] --
   never gets this far: linking already refuses a parameter the prior context lacks.) *)

open Base
open Verdict.Claims
module LL = Ir.Low_level

let node = Ll_test.node_factory ~first_id:13290 ~dims:[| 4 |] ()

(* Whether [f] raises the typed refusal whose message contains [needle]. Any other outcome --
   success, or a different exception -- is a [false] claim rather than an aborted case, so each row
   reports on its own. *)
let refused ~needle f =
  match f () with
  | _ -> false
  | exception Utils.User_error msg -> String.is_substring msg ~substring:needle
  | exception _ -> false

let released = "was released by Context.release"

(* [dst[i] := src[i] + 1 + i]: values that identify their cell and stay off the zero-init. *)
let shift ~src ~dst =
  let i = Ll_test.sym () in
  Ll_test.loop_n i 4
    (Ll_test.set dst
       [| Ll_test.iter i |]
       (Ll_test.add (Ll_test.get src [| Ll_test.iter i |]) (Ll_test.tick i)))

let link_shift ?lctx ~ctx ~name ~src ~dst () =
  Ll_test.materialize src;
  Ll_test.materialize dst;
  let lctx = Option.value_or_thunk lctx ~default:LL.empty_optimize_ctx in
  let o = Ll_test.optimize_in lctx ~materialized:[ src; dst ] ~name (shift ~src ~dst) in
  Ll_test.link ~ctx ~name o

let seed = [| 10.; 20.; 30.; 40. |]
let shifted = [| 11.; 22.; 33.; 44. |]

(* gh-ocannl-1329. *)
let () =
  Verdict.case "chr_released" @@ fun () ->
  let base = Context.auto () in
  let x_a = node "chr_x_a" and out_a = node "chr_out_a" in
  let x_b = node "chr_x_b" and out_b = node "chr_out_b" in
  let c_a, r_a = link_shift ~ctx:base ~name:"chr_a" ~src:x_a ~dst:out_a () in
  let c_b, r_b = link_shift ~ctx:base ~name:"chr_b" ~src:x_b ~dst:out_b () in
  (* An upload into a buffer [c_a] already holds continues its linear chain: [c_a'] is the latest
     value, the one to release, and [c_a] shares its flag. *)
  let c_a' = Context.set_values c_a x_a seed in
  let c_a' = Context.run c_a' r_a in
  pf "chr_released: the routine ran before the release"
    (Poly.equal (Context.get_values c_a' out_a) shifted);
  Context.release c_a';
  let refused_here what f = pf "chr_released: %s is refused" what (refused ~needle:released f) in
  refused_here "run on the released value" (fun () -> Context.run c_a' r_a);
  refused_here "run on an earlier value of its upload chain" (fun () -> Context.run c_a r_a);
  refused_here "run on the live parent with a routine compiled into the released value" (fun () ->
      Context.run base r_a);
  refused_here "compile" (fun () -> link_shift ~ctx:c_a' ~name:"chr_a2" ~src:x_a ~dst:out_a ());
  refused_here "sync" (fun () -> Context.sync c_a');
  refused_here "to_host" (fun () -> Context.to_host c_a' out_a);
  refused_here "get_values" (fun () -> Context.get_values c_a' out_a);
  refused_here "set_values" (fun () -> Context.set_values c_a' x_a seed);
  refused_here "from_host" (fun () -> Context.from_host c_a' x_a (Context.to_host c_b x_b));
  refused_here "copy out of it" (fun () -> Context.copy ~src:c_a' ~dst:c_b x_a);
  refused_here "copy into it" (fun () -> Context.copy ~src:c_b ~dst:c_a' x_b);
  refused_here "decide_materialized" (fun () -> Context.decide_materialized c_a' [ out_a ]);
  refused_here "decide_inline" (fun () -> Context.decide_inline c_a' [ out_a ]);
  refused_here "decide_footprint" (fun () -> Context.decide_footprint c_a' [ out_a ]);
  (* The refusal names the context it refused. *)
  p "chr_released: the refusal names the entry point and the backend"
    (refused ~needle:"Context.run: context" (fun () -> Context.run c_a' r_a)
    && refused ~needle:("backend " ^ Context.backend_name c_a') (fun () -> Context.sync c_a'));
  (* The compile sibling keeps its own flag, and the parent stays live. *)
  let c_b = Context.run (Context.set_values c_b x_b seed) r_b in
  p "chr_released: the compile sibling still runs and reads back"
    (Poly.equal (Context.get_values c_b out_b) shifted);
  let x_c = node "chr_x_c" and out_c = node "chr_out_c" in
  let c_c, r_c = link_shift ~ctx:base ~name:"chr_c" ~src:x_c ~dst:out_c () in
  let c_c = Context.run (Context.set_values c_c x_c seed) r_c in
  p "chr_released: the live parent still compiles and runs"
    (Poly.equal (Context.get_values c_c out_c) shifted);
  Context.release c_b;
  Context.release c_c

(* An upload fork survives the release of the chain it forked from: uploading a node the routine's
   context lacks continues that context's linear chain the first time, and forks it the second time.
   The fork reference-counts the pools it holds, the routine's included, so after the chain is
   released the fork still runs the routine, while the chain's own values are refused. *)
let () =
  Verdict.case "chr_fork" @@ fun () ->
  let base = Context.auto () in
  let x = node "chr_fx" and out = node "chr_fout" and y = node "chr_fy" in
  Ll_test.materialize y;
  let c, r = link_shift ~ctx:base ~name:"chr_fork" ~src:x ~dst:out () in
  let chain = Context.set_values c y seed in
  let fork = Context.set_values c y shifted in
  Context.release chain;
  pf "chr_fork: the compile context, sharing the released chain's flag, is refused"
    (refused ~needle:released (fun () -> Context.run c r));
  let fork = Context.run (Context.set_values fork x seed) r in
  pf "chr_fork: the live fork still runs the routine and reads back"
    (Poly.equal (Context.get_values fork out) shifted
    && Poly.equal (Context.get_values fork y) shifted);
  Context.release fork

(* gh-ocannl-1462: a released ANCESTOR. [c2] is compiled from [c1] and binds [c1]'s buffer of [out1]
   (its source), and [c3] is compiled from [c2]; releasing the interior [c1] frees that buffer under
   both descendants. Every buffer-touching entry point refuses them, naming how far up the released
   ancestor is, before reaching the freed pool -- and a live context of another lineage running
   [c2]'s routine is refused too, by the routine's compile context. The descendants can still
   release their own pools. *)
let ancestor = "descends from a context released by Context.release"

let () =
  Verdict.case "chr_released_ancestor" @@ fun () ->
  let base = Context.auto () in
  let x1 = node "chr_ax1" and out1 = node "chr_aout1" in
  let out2 = node "chr_aout2" and out3 = node "chr_aout3" in
  let c1, r1 = link_shift ~ctx:base ~name:"chr_anc1" ~src:x1 ~dst:out1 () in
  let c1 = Context.run (Context.set_values c1 x1 seed) r1 in
  let c2, r2 = link_shift ~ctx:c1 ~name:"chr_anc2" ~src:out1 ~dst:out2 () in
  let c3, _r3 = link_shift ~ctx:c2 ~name:"chr_anc3" ~src:out2 ~dst:out3 () in
  let c2 = Context.run c2 r2 in
  pf "chr_released_ancestor: the descendant ran from its ancestor's buffer before the release"
    (Poly.equal (Context.get_values c2 out2) [| 12.; 24.; 36.; 48. |]);
  Context.release c1;
  let refused_here what f =
    pf "chr_released_ancestor: %s is refused" what (refused ~needle:ancestor f)
  in
  refused_here "compile from the child" (fun () ->
      link_shift ~ctx:c2 ~name:"chr_anc4" ~src:out2 ~dst:out3 ());
  refused_here "run on the child" (fun () -> Context.run c2 r2);
  refused_here "run on a live context of another lineage with the child's routine" (fun () ->
      Context.run base r2);
  refused_here "sync" (fun () -> Context.sync c2);
  refused_here "get_values" (fun () -> Context.get_values c2 out1);
  refused_here "set_values" (fun () -> Context.set_values c2 out1 seed);
  refused_here "decide_materialized" (fun () -> Context.decide_materialized c2 [ out2 ]);
  pf "chr_released_ancestor: the refusal names the entry point and the distance"
    (refused ~needle:"Context.sync: context" (fun () -> Context.sync c2)
    && refused ~needle:"ancestor 1 level up" (fun () -> Context.sync c2)
    && refused ~needle:"ancestor 2 levels up" (fun () -> Context.sync c3));
  p "chr_released_ancestor: the descendants still release their own pools"
    (match
       Context.release c3;
       Context.release c2
     with
    | () -> true
    | exception _ -> false)

(* An upload fork's lifecycle ancestors are those of the value it forked, not that value: a child
   compiled from a fork survives the release of the chain the fork came from. *)
let () =
  Verdict.case "chr_fork_child" @@ fun () ->
  let base = Context.auto () in
  let x = node "chr_fcx" and out = node "chr_fcout" and y = node "chr_fcy" in
  let out2 = node "chr_fcout2" in
  Ll_test.materialize y;
  let c, _r = link_shift ~ctx:base ~name:"chr_fc" ~src:x ~dst:out () in
  let chain = Context.set_values c y seed in
  let fork = Context.set_values c y shifted in
  let child, r_child = link_shift ~ctx:fork ~name:"chr_fc_child" ~src:y ~dst:out2 () in
  Context.release chain;
  let child = Context.run child r_child in
  pf "chr_fork_child: a child of the surviving fork still runs and reads back"
    (Poly.equal (Context.get_values child out2) [| 12.; 24.; 36.; 48. |]);
  Context.release child;
  Context.release fork

(* [ctr[i] := ctr[i] + 1 + i]: the counter is read before it is written. *)
let bump ctr = shift ~src:ctr ~dst:ctr

(* [ctr[i] := 1 + i]: writes the counter without reading it. *)
let fill ctr =
  let i = Ll_test.sym () in
  Ll_test.loop_n i 4 (Ll_test.set ctr [| Ll_test.iter i |] (Ll_test.tick i))

(* gh-ocannl-1339. *)
let () =
  Verdict.case "chr_unseeded" @@ fun () ->
  let base = Context.auto () in
  let ctr = node "chr_ctr" in
  Ll_test.materialize ctr;
  let o = Ll_test.optimize ~materialized:[ ctr ] ~name:"chr_bump" (bump ctr) in
  pf "chr_unseeded: the counter is read before written" (Ll_test.read_before_write o ctr);
  let c, r = Ll_test.link ~ctx:base ~name:"chr_bump" o in
  pf "chr_unseeded: the counter is both an input and an output of the routine"
    (Set.mem r.Context.inputs ctr && Set.mem r.Context.outputs ctr);
  pf "chr_unseeded: the counter has a buffer, which is not a value"
    (Context.mem c ctr && not (Context.is_initialized c ctr));
  pf "chr_unseeded: running it never seeded is refused, naming the counter as read before written"
    (refused ~needle:"chr_ctr (read before written)" (fun () -> Context.run c r));
  let c = Context.run (Context.set_values c ctr seed) r in
  pf "chr_unseeded: seeded, it runs from the seed" (Poly.equal (Context.get_values c ctr) shifted);
  Context.release c

let () =
  Verdict.case "chr_written_elsewhere" @@ fun () ->
  let base = Context.auto () in
  let ctr = node "chr_ctr_w" in
  Ll_test.materialize ctr;
  let lctx = LL.empty_optimize_ctx () in
  let o_fill = Ll_test.optimize_in lctx ~materialized:[ ctr ] ~name:"chr_fill" (fill ctr) in
  let c_fill, r_fill = Ll_test.link ~ctx:base ~name:"chr_fill" o_fill in
  pf "chr_written_elsewhere: the filler does not read the counter"
    (not (Set.mem r_fill.Context.inputs ctr));
  let o_bump = Ll_test.optimize_in lctx ~materialized:[ ctr ] ~name:"chr_bump_w" (bump ctr) in
  let c_bump, r_bump = Ll_test.link ~ctx:c_fill ~name:"chr_bump_w" o_bump in
  (* The filler's returned context is dropped: [c_bump] never learns from it that the counter was
     written, so only the lineage's record of the write can admit the run. *)
  ignore (Context.run c_fill r_fill : Context.t);
  pf "chr_written_elsewhere: the value it runs in has not tracked the write"
    (not (Context.is_initialized c_bump ctr));
  let c_bump = Context.run c_bump r_bump in
  pf "chr_written_elsewhere: written by another routine of the lineage, it runs from that write"
    (Poly.equal (Context.get_values c_bump ctr) [| 2.; 4.; 6.; 8. |]);
  Context.release c_bump

(* What vouches for a read-before-write input is the buffer the routine binds, not the node. Two
   compile siblings each allocate their own counter buffer: seeding one sibling's does not let the
   other's routine run, even in the seeded sibling's context, whose [initialized_nodes] does name
   the counter. *)
let () =
  Verdict.case "chr_sibling_seed" @@ fun () ->
  let base = Context.auto () in
  let ctr = node "chr_ctr_s" in
  Ll_test.materialize ctr;
  let link name =
    Ll_test.link ~ctx:base ~name (Ll_test.optimize ~materialized:[ ctr ] ~name (bump ctr))
  in
  let c_a, _r_a = link "chr_bump_sa" and c_b, r_b = link "chr_bump_sb" in
  let c_a = Context.set_values c_a ctr seed in
  pf "chr_sibling_seed: the seeded sibling's context names the counter initialized"
    (Context.is_initialized c_a ctr);
  pf "chr_sibling_seed: the other sibling's routine is refused even run in that context"
    (refused ~needle:"chr_ctr_s (read before written)" (fun () -> Context.run c_a r_b));
  Context.release c_a;
  Context.release c_b

(* A run [rollback_execution] withdraws -- one known not to have written -- withdraws its write
   claims too: the counter its writer never filled is refused again. (The initialization check runs
   before the dependency check, so the refusal names the counter rather than the rolled-back
   writer.) *)
let () =
  Verdict.case "chr_rollback" @@ fun () ->
  let base = Context.auto () in
  let ctr = node "chr_ctr_rb" in
  Ll_test.materialize ctr;
  let lctx = LL.empty_optimize_ctx () in
  let o_fill = Ll_test.optimize_in lctx ~materialized:[ ctr ] ~name:"chr_fill_rb" (fill ctr) in
  let c_fill, r_fill = Ll_test.link ~ctx:base ~name:"chr_fill_rb" o_fill in
  let o_bump = Ll_test.optimize_in lctx ~materialized:[ ctr ] ~name:"chr_bump_rb" (bump ctr) in
  let c_bump, r_bump = Ll_test.link ~ctx:c_fill ~name:"chr_bump_rb" o_bump in
  ignore (Context.run c_fill r_fill : Context.t);
  Context.sync c_fill;
  Context.rollback_execution c_fill r_fill.Context.routine_id;
  pf "chr_rollback: after the writer's run is rolled back, the counter is unwritten again"
    (refused ~needle:"chr_ctr_rb (read before written)" (fun () -> Context.run c_bump r_bump));
  Context.release c_bump

(* gh-ocannl-1462: a READ-ONLY input must be written too, unless allocation-time zero is declared
   its correct value. Before, it was accepted once allocated -- and [compile] allocates every input
   -- so a forgotten data input read zeros. *)
let ro_refused name f = refused ~needle:("not initialized: " ^ name) f

let () =
  Verdict.case "chr_read_only" @@ fun () ->
  let base = Context.auto () in
  let x = node "chr_ro_x" and out = node "chr_ro_out" in
  let c, r = link_shift ~ctx:base ~name:"chr_ro" ~src:x ~dst:out () in
  pf "chr_read_only: the source is a read-only input"
    (Set.mem r.Context.inputs x && not (Set.mem r.Context.outputs x));
  pf "chr_read_only: an allocated, never-written, unmarked read-only input is refused"
    (ro_refused "chr_ro_x." (fun () -> Context.run c r));
  let c = Context.run (Context.set_values c x seed) r in
  pf "chr_read_only: seeded, it runs from the seed" (Poly.equal (Context.get_values c out) shifted);
  Context.release c

let () =
  Verdict.case "chr_zero_valid" @@ fun () ->
  let base = Context.auto () in
  let x = node "chr_zv_x" and out = node "chr_zv_out" in
  Ir.Tnode.mark_zero_valid x;
  let c, r = link_shift ~ctx:base ~name:"chr_zv" ~src:x ~dst:out () in
  let c = Context.run c r in
  pf "chr_zero_valid: a read-only input marked zero-valid runs from its allocation-time zeros"
    (Poly.equal (Context.get_values c out) [| 1.; 2.; 3.; 4. |]);
  Context.release c;
  (* The mark does not excuse state the routine overwrites. *)
  let ctr = node "chr_zv_ctr" in
  Ll_test.materialize ctr;
  Ir.Tnode.mark_zero_valid ctr;
  let o = Ll_test.optimize ~materialized:[ ctr ] ~name:"chr_zv_bump" (bump ctr) in
  let c, r = Ll_test.link ~ctx:base ~name:"chr_zv_bump" o in
  pf "chr_zero_valid: a zero-valid read-before-write input is still refused until seeded"
    (refused ~needle:"chr_zv_ctr (read before written)" (fun () -> Context.run c r));
  Context.release c

(* What vouches for a read-only input is the buffer the routine binds, as for read-before-write
   state: a compile sibling's seeded buffer for the same node does not, and neither does a producer
   whose run was rolled back. *)
let () =
  Verdict.case "chr_read_only_buffer" @@ fun () ->
  let base = Context.auto () in
  let x = node "chr_rob_x" and out = node "chr_rob_out" in
  (* Context-owned, so each sibling binds working storage of its own: a node every routine only
     reads may otherwise live in one constant pool of their root, which the siblings share. *)
  Ir.Tnode.set_context_owned x;
  let c_a, _r_a = link_shift ~ctx:base ~name:"chr_rob_a" ~src:x ~dst:out () in
  let c_b, r_b = link_shift ~ctx:base ~name:"chr_rob_b" ~src:x ~dst:out () in
  let c_a = Context.set_values c_a x seed in
  pf "chr_read_only_buffer: a sibling seeding its own buffer does not let the other's routine run"
    (ro_refused "chr_rob_x." (fun () -> Context.run c_a r_b));
  Context.release c_a;
  Context.release c_b;
  let src = node "chr_rob_src" in
  Ll_test.materialize src;
  let lctx = LL.empty_optimize_ctx () in
  let o_fill = Ll_test.optimize_in lctx ~materialized:[ src ] ~name:"chr_rob_fill" (fill src) in
  let c_fill, r_fill = Ll_test.link ~ctx:base ~name:"chr_rob_fill" o_fill in
  let c_use, r_use =
    link_shift ~lctx ~ctx:c_fill ~name:"chr_rob_use" ~src ~dst:(node "chr_rob_use_out") ()
  in
  pf "chr_read_only_buffer: the consumer reads the producer's output read-only"
    (Set.mem r_use.Context.inputs src && not (Set.mem r_use.Context.outputs src));
  ignore (Context.run c_fill r_fill : Context.t);
  Context.sync c_fill;
  Context.rollback_execution c_fill r_fill.Context.routine_id;
  pf "chr_read_only_buffer: after the producer's run is rolled back, the input is unwritten again"
    (ro_refused "chr_rob_src." (fun () -> Context.run c_use r_use));
  Context.release c_use

(* gh-ocannl-1654: a node every routine only reads is a constant, cached PER ROOT unless it is a
   host-initialized literal. Two independently created contexts therefore bind buffers of their own
   for a data input: one root's seed, write or writer run never reaches the other, and a copy
   between them is a real transfer. Within one root the constant is shared as before. Before, every
   root on the device bound one buffer for it, and a write in one was refused as stale in the
   other. *)
let seed2 = [| 100.; 200.; 300.; 400. |]
let shifted2 = [| 101.; 202.; 303.; 404. |]

let () =
  Verdict.case "chr_root_constants" @@ fun () ->
  let x = node "chr_rc_x" and out_a = node "chr_rc_out_a" and out_b = node "chr_rc_out_b" in
  let lctx = LL.empty_optimize_ctx () in
  let c_a, r_a = link_shift ~lctx ~ctx:(Context.auto ()) ~name:"chr_rc_a" ~src:x ~dst:out_a () in
  let c_b, r_b = link_shift ~ctx:(Context.auto ()) ~name:"chr_rc_b" ~src:x ~dst:out_b () in
  pf "chr_root_constants: a data input is no shared read in either root"
    ((not (Set.mem r_a.Context.shared_reads x)) && not (Set.mem r_b.Context.shared_reads x));
  pf "chr_root_constants: unseeded, a root's consumer is refused"
    (ro_refused "chr_rc_x." (fun () -> Context.run c_b r_b));
  let c_a = Context.set_values c_a x seed in
  pf "chr_root_constants: seeded by the other root only, it is still refused"
    (ro_refused "chr_rc_x." (fun () -> Context.run c_b r_b));
  let c_b = Context.set_values c_b x seed2 in
  let c_a = Context.run c_a r_a in
  pf "chr_root_constants: each root runs from its own seed, the first"
    (Poly.equal (Context.get_values c_a out_a) shifted);
  let c_b = Context.run c_b r_b in
  pf "chr_root_constants: and the second" (Poly.equal (Context.get_values c_b out_b) shifted2);
  (* A writer compiled after [x] was bound as a constant writes the first root's buffer only. *)
  let o_w = Ll_test.optimize_in lctx ~materialized:[ x ] ~name:"chr_rc_w" (fill x) in
  let c_w, r_w = Ll_test.link ~ctx:c_a ~name:"chr_rc_w" o_w in
  let c_w = Context.run c_w r_w in
  let c_b = Context.run c_b r_b in
  pf "chr_root_constants: the first root's writer leaves the second root's value as it was"
    (Poly.equal (Context.get_values c_b out_b) shifted2);
  let c_b = Context.run (Context.copy ~src:c_w ~dst:c_b x) r_b in
  pf "chr_root_constants: a copy between the roots transfers the writer's value"
    (Poly.equal (Context.get_values c_b out_b) [| 2.; 4.; 6.; 8. |]);
  let c_w = Context.set_values c_w x seed in
  pf "chr_root_constants: and a later write by the source leaves the copy as it was"
    (Poly.equal (Context.get_values (Context.run c_b r_b) out_b) [| 2.; 4.; 6.; 8. |]);
  (* Within one root the constant is one buffer: a write through another context value of the root
     (here a descendant compiled from it) is what the root's routine reads. *)
  let c_s, _ = link_shift ~ctx:c_b ~name:"chr_rc_sib" ~src:x ~dst:(node "chr_rc_out_s") () in
  let c_s = Context.set_values c_s x seed2 in
  pf "chr_root_constants: a write by another context of the same root is what it reads"
    (Poly.equal (Context.get_values (Context.run c_b r_b) out_b) shifted2);
  List.iter [ c_w; c_s; c_b ] ~f:Context.release

(* A host-initialized literal is cached once per device: every root binds the same buffer for it
   (gh-ocannl-1654). It is writable all the same, so a write one root makes there changes the bytes
   the other's routine reads; that root is refused as stale until it re-seeds or copies
   (gh-ocannl-1623). *)
let literal name =
  let h = node name in
  let nd = Ir.Ndarray.create_array ~debug:name Ir.Ops.single ~dims:[| 4 |] ~padding:None in
  Ir.Ndarray.fill_from_float nd 5.;
  Ir.Host_inits.register h (lazy nd);
  h

let stale_refused name f =
  refused ~needle:(name ^ " are host-initialized literals, whose one device-wide buffer another") f

let () =
  Verdict.case "chr_shared_overwrite" @@ fun () ->
  let x = literal "chr_so_x" and out_a = node "chr_so_out_a" and out_b = node "chr_so_out_b" in
  let c_a, r_a = link_shift ~ctx:(Context.auto ()) ~name:"chr_so_a" ~src:x ~dst:out_a () in
  let c_b, r_b = link_shift ~ctx:(Context.auto ()) ~name:"chr_so_b" ~src:x ~dst:out_b () in
  pf "chr_shared_overwrite: the two roots read the same device-wide literal buffer"
    (Set.mem r_a.Context.shared_reads x && Set.mem r_b.Context.shared_reads x);
  let c_a = Context.set_values c_a x seed in
  pf "chr_shared_overwrite: after the first root writes, the other root's run is refused"
    (stale_refused "chr_so_x" (fun () -> Context.run c_b r_b));
  let c_b = Context.run (Context.copy ~src:c_a ~dst:c_b x) r_b in
  pf "chr_shared_overwrite: copying admits the other root, reading the written value"
    (Poly.equal (Context.get_values c_b out_b) shifted);
  let c_a = Context.set_values c_a x seed2 in
  pf "chr_shared_overwrite: after the first root writes again, the other root's run is refused"
    (stale_refused "chr_so_x" (fun () -> Context.run c_b r_b));
  let c_a = Context.run c_a r_a in
  pf "chr_shared_overwrite: the writer itself runs from its own write"
    (Poly.equal (Context.get_values c_a out_a) shifted2);
  let c_b = Context.run (Context.copy ~src:c_a ~dst:c_b x) r_b in
  pf "chr_shared_overwrite: copying again readmits the other root, reading the new value"
    (Poly.equal (Context.get_values c_b out_b) shifted2);
  let c_b = Context.run (Context.set_values c_b x seed) r_b in
  pf "chr_shared_overwrite: re-seeding readmits a root too"
    (Poly.equal (Context.get_values c_b out_b) shifted);
  pf "chr_shared_overwrite: and makes the first root stale in turn"
    (stale_refused "chr_so_x" (fun () -> Context.run c_a r_a));
  (* A third root copying from the stale first one adopts nothing: the bytes are not its
     source's. *)
  let out_c = node "chr_so_out_c" in
  let c_c, r_c = link_shift ~ctx:(Context.auto ()) ~name:"chr_so_c" ~src:x ~dst:out_c () in
  let c_c = Context.copy ~src:c_a ~dst:c_c x in
  pf "chr_shared_overwrite: a copy out of a stale root vouches for nothing"
    (stale_refused "chr_so_x" (fun () -> Context.run c_c r_c));
  (* Within one root the buffer is shared as before: a write through another context value of the
     same root (here a descendant compiled from it) is that root's own. *)
  let c_s, _ = link_shift ~ctx:c_b ~name:"chr_so_sib" ~src:x ~dst:(node "chr_so_out_s") () in
  let c_s = Context.set_values c_s x seed2 in
  pf "chr_shared_overwrite: a write by another context of the same root is not stale to it"
    (Poly.equal (Context.get_values (Context.run c_b r_b) out_b) shifted2);
  List.iter [ c_a; c_s; c_b; c_c ] ~f:Context.release

(* A routine that writes the shared literal -- compiled from a context where an earlier routine
   bound it as a constant -- makes the other root stale only by RUNNING, and a rollback of that run
   (known not to have written) restores the other root's currency. *)
let () =
  Verdict.case "chr_shared_writer" @@ fun () ->
  let x = literal "chr_sw_x" and out_a = node "chr_sw_out_a" and out_b = node "chr_sw_out_b" in
  let lctx = LL.empty_optimize_ctx () in
  let c_a, r_a = link_shift ~lctx ~ctx:(Context.auto ()) ~name:"chr_sw_a" ~src:x ~dst:out_a () in
  let o_w = Ll_test.optimize_in lctx ~materialized:[ x ] ~name:"chr_sw_w" (fill x) in
  let c_w, r_w = Ll_test.link ~ctx:c_a ~name:"chr_sw_w" o_w in
  pf "chr_shared_writer: the writer writes the node" (Set.mem r_w.Context.outputs x);
  let c_b, r_b = link_shift ~ctx:(Context.auto ()) ~name:"chr_sw_b" ~src:x ~dst:out_b () in
  (* The reader runs first: the writer waits on it (write after read). *)
  let c_w = Context.run (Context.set_values c_w x seed) r_a in
  let c_b = Context.run (Context.copy ~src:c_w ~dst:c_b x) r_b in
  pf "chr_shared_writer: compiled but not run, the writer leaves the other root current"
    (Poly.equal (Context.get_values (Context.run c_b r_b) out_b) shifted);
  let c_w = Context.run c_w r_w in
  Context.sync c_w;
  Context.rollback_execution c_w r_w.Context.routine_id;
  pf "chr_shared_writer: a rolled-back run of the writer leaves the other root current"
    (match Context.run c_b r_b with _ -> true | exception _ -> false);
  let c_w = Context.run c_w r_w in
  pf "chr_shared_writer: run, the writer makes the other root stale"
    (stale_refused "chr_sw_x" (fun () -> Context.run c_b r_b));
  List.iter [ c_w; c_a; c_b ] ~f:Context.release

(* Allocation-time zeros and a host initialization are values too. A zero-valid data input is cached
   per root, so another root's write leaves the zeros a reader relies on; a host literal is shared,
   so a reader relying on its host initialization is refused once another root wrote it. *)
let () =
  Verdict.case "chr_shared_implicit" @@ fun () ->
  let x = node "chr_si_zv" and out_a = node "chr_si_out_a" and out_b = node "chr_si_out_b" in
  Ir.Tnode.mark_zero_valid x;
  let c_a, _ = link_shift ~ctx:(Context.auto ()) ~name:"chr_si_a" ~src:x ~dst:out_a () in
  let c_b, r_b = link_shift ~ctx:(Context.auto ()) ~name:"chr_si_b" ~src:x ~dst:out_b () in
  let c_b = Context.run c_b r_b in
  pf "chr_shared_implicit: a zero-valid input runs from zeros"
    (Poly.equal (Context.get_values c_b out_b) [| 1.; 2.; 3.; 4. |]);
  let c_a = Context.set_values c_a x seed in
  pf "chr_shared_implicit: another root's write leaves the zero-valid reader its zeros"
    (Poly.equal (Context.get_values (Context.run c_b r_b) out_b) [| 1.; 2.; 3.; 4. |]);
  List.iter [ c_a; c_b ] ~f:Context.release;
  let h = literal "chr_si_host" and out_a = node "chr_si_hout_a" and out_b = node "chr_si_hout_b" in
  let c_a, _ = link_shift ~ctx:(Context.auto ()) ~name:"chr_si_ha" ~src:h ~dst:out_a () in
  let c_b, r_b = link_shift ~ctx:(Context.auto ()) ~name:"chr_si_hb" ~src:h ~dst:out_b () in
  pf "chr_shared_implicit: a host-initialized input is not a required input, but a shared read"
    ((not (Set.mem r_b.Context.inputs h)) && Set.mem r_b.Context.shared_reads h);
  let c_b = Context.run c_b r_b in
  pf "chr_shared_implicit: it runs from its host initialization"
    (Poly.equal (Context.get_values c_b out_b) [| 6.; 7.; 8.; 9. |]);
  let c_a = Context.set_values c_a h seed in
  pf "chr_shared_implicit: another root's write makes the host-initialized reader stale"
    (stale_refused "chr_si_host" (fun () -> Context.run c_b r_b));
  List.iter [ c_a; c_b ] ~f:Context.release

(* Both gradient creation sites mark the gradient zero-valid: a tensor's own ([Tensor.op]) and a
   parameter's ([Tensor.param], which replaces it). *)
let () =
  Verdict.case "chr_grad_zero_valid" @@ fun () ->
  let open Ocannl.Operation.DSL_modules in
  let grad_marked (t : Tensor.t) =
    Option.exists t.Tensor.diff ~f:(fun d -> Ir.Tnode.is_zero_valid d.Tensor.grad)
  in
  let t = Tensor.term_init [| 1.; 2. |] ~grad_spec:Tensor.Require_grad () in
  let w = TDSL.param "chr_w" () in
  pf "chr_grad_zero_valid: a tensor's gradient is zero-valid" (grad_marked t);
  pf "chr_grad_zero_valid: a parameter's gradient is zero-valid" (grad_marked w);
  pf "chr_grad_zero_valid: the value nodes are not" (not (Ir.Tnode.is_zero_valid w.Tensor.value))

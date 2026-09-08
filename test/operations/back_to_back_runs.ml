(* gh-ocannl-909: consecutive runs of one routine must execute in order on every backend, with no
   host synchronization between them. A routine's task waits only for the input events captured at
   link time, so back-to-back runs of the SAME routine — every training step, every
   [Train.sequential_loop] iteration — are ordered by the backend's queue semantics alone: FIFO on a
   CUDA/HIP stream, synchronous on cc, and on Metal by the wait each launch encodes for the previous
   all-work signal. The standalone probe of gh-ocannl-828 (`benchmarks/runners/ocannl/
   metal_queue_probe.ml`) reproduced that signal/wait shape against the Metal bindings alone and saw
   the two command buffers overlap, contradicting the backend's comment — a same-buffer race, since
   repeated runs reuse the same pools.

   This is the executed form of that invariant: a routine that reads a counter FIRST, spins long
   enough that a run started before the previous one finished would still read the stale value, then
   writes the counter back incremented. [runs] such runs with no sync between them must leave the
   counter at [runs]; an overlapping pair loses at least one increment. The increment is exact by
   construction: [outer * m] FMA steps of [1/(outer * m)] over a device-resident [ones] operand, so
   the compiler cannot fold the spin (it cannot know [ones] holds ones) and every partial sum is a
   dyadic rational the accumulator holds exactly up to the run count here. The read is data-bound to
   the first step ([acc = counter; acc = fma(acc, ones[i], steps[i])]), so no compiler can sink the
   load below the spin. *)

open Base
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Idx = Ir.Indexing
open Verdict.Claims

let single = Ir.Ops.single
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let node = Ll_test.node_factory ~first_id:967_000_000 ~dims:[| 1 |] ()
let m = 1024
let outer = 256
let runs = 16

(* [1/(outer * m)] = 2^-18: [outer * m] steps of it sum to exactly 1, and [runs + k * 2^-18] stays
   exact in f32 while [runs] is below 2^5 (5 + 18 < 24 mantissa bits). *)
let step = 1. /. Float.of_int (outer * m)
let counter = node "run_counter"
let acc = node "spin_acc"
let ones = node ~dims:[| m |] "spin_ones"
let steps = node ~dims:[| m |] "spin_steps"

let () =
  List.iter [ counter; ones; steps ] ~f:Ll_test.materialize;
  Tn.update_memory_mode acc Tn.Local 99

let f0 = Idx.Fixed_idx 0
let get tn idcs : LL.scalar_t = LL.Get (tn, idcs)

let llc =
  let j = Ll_test.sym () and i = Ll_test.sym () in
  let spin_step =
    LL.Set
      {
        tn = acc;
        idcs = [| f0 |];
        llsc =
          LL.Ternop
            ( Ir.Ops.FMA,
              (get acc [| f0 |], single),
              (get ones [| Idx.Iterator i |], single),
              (get steps [| Idx.Iterator i |], single) );
        debug = "";
      }
  in
  LL.Seq
    ( LL.Set { tn = acc; idcs = [| f0 |]; llsc = get counter [| f0 |]; debug = "" },
      LL.Seq
        ( LL.For_loop
            {
              index = j;
              from_ = 0;
              to_ = outer - 1;
              axis = LL.Serial;
              body =
                LL.For_loop
                  { index = i; from_ = 0; to_ = m - 1; axis = LL.Serial; body = spin_step };
            },
          LL.Set { tn = counter; idcs = [| f0 |]; llsc = get acc [| f0 |]; debug = "" } ) )

let compile ~name =
  let o = Ll_test.optimize ~materialized:[ counter; ones; steps ] ~name llc in
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ~name ~prelowered:o ctx Ir.Assignments.empty_comp Idx.Empty in
  let ctx = Context.set_values ctx ones (Array.create ~len:m 1.) in
  let ctx = Context.set_values ctx steps (Array.create ~len:m step) in
  let ctx = Context.set_values ctx counter [| 0. |] in
  (ctx, routine)

let () =
  (* The golden is backend-uniform; the backend is announced on stderr (gh-ocannl-622). *)
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" backend_name;
  (* The control: one run, synchronized, increments by exactly one — the routine itself is right,
     and a lost increment below is a lost RUN, not a wrong sum. *)
  let ctx, routine = compile ~name:"b2b_control" in
  let t0 = Unix.gettimeofday () in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  let single_run_s = Unix.gettimeofday () -. t0 in
  Stdio.eprintf "one synchronized run: %.3f ms (not part of the golden)\n%!" (single_run_s *. 1000.);
  p "one synchronized run of the self-incrementing routine adds exactly one"
    (Float.equal (Context.get_values ctx counter).(0) 1.);
  (* [runs] back-to-back, nothing between them, one host read at the end. *)
  let ctx, routine = compile ~name:"b2b_runs" in
  let ctx = List.fold (List.range 0 runs) ~init:ctx ~f:(fun ctx _ -> Context.run ctx routine) in
  let got = (Context.get_values ctx counter).(0) in
  Stdio.eprintf "counter after %d back-to-back runs: %g (not part of the golden)\n%!" runs got;
  p
    (Printf.sprintf
       "%d back-to-back runs of the self-incrementing routine, with no synchronization between \
        them, count every run: no run read the counter before the previous run wrote it"
       runs)
    (Float.equal got (Float.of_int runs))

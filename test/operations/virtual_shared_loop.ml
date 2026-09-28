(* Regression test for gh-ocannl-134: multiple virtual tensors sharing one traced for-loop.

   High-level lowering never places two distinct tensors in the same for-loop (each assignment
   lowers to its own [loop_over_dims]), so these cases are built directly as [Ir.Low_level.t] and
   run through the [Ll_test] harness (gh-ocannl-600) -- the same pipeline (trace_node_facts ->
   decide_placements -> virtual_llc -> cleanup_virtual_llc -> simplify -> CSE -> hoist) the backends
   use. We assert structurally on the optimized form and on the resulting [traced_array]/memory-mode
   facts, which precisely pin the #134 invariants:

   - shared loop symbols no longer force [is_complex]; - each candidate tensor in a shared loop gets
   its own stored computation and inlines downstream; - a surviving (materialized) sibling setter
   inlines a virtualized provider from the same loop; - a forward virtual->virtual chain is fully
   inlined, leaving no read of a dropped virtual node; - a reverse/read-before-write sibling read
   stays materialized (existing safety mechanism); - cleanup keeps non-virtual residual setters
   instead of dropping the whole loop.

   Structural pins say what the optimizer BUILT; virtualization rewrites what value a cell holds, so
   every case also has an EXECUTED leg (gh-ocannl-589): the very same [optimized] record is compiled
   through the [?prelowered] seam (gh-ocannl-562, worked out in [prelowered_seam.ml]), seeded, run,
   and its outputs checked against an OCaml reference. Where the case has a virtualization
   candidate, a second arm re-specializes the same code with that candidate's placement pre-decided
   [On_device] — the materialized reading of the same program — and the two arms must agree. *)

open Base
open Ll_test
open Verdict.Claims

let mk = node_factory ~first_id:1000 ~dims:[| 3 |] ()
let optimize ?materialized llc = optimize ?materialized ~name:"shared_loop" llc

(* === Case 1: two independent virtual siblings in one loop, read downstream === *)
let case_independent () =
  let a = mk "a" and b = mk "b" and oa = mk "oa" and ob = mk "ob" in
  materialize oa;
  materialize ob;
  let i = sym () and j = sym () and k = sym () in
  let shared = loop_n i 3 (seq (set a [| iter i |] (ramp 2. i)) (set b [| iter i |] (ramp 3. i))) in
  let use_a = loop_n j 3 (set oa [| iter j |] (get a [| iter j |])) in
  let use_b = loop_n k 3 (set ob [| iter k |] (get b [| iter k |])) in
  let llc = seq shared (seq use_a use_b) in
  let o = optimize llc in
  p "independent siblings both virtual" (known_virtual o a && known_virtual o b);
  p "independent siblings setters dropped" (count_set o a = 0 && count_set o b = 0);
  p "independent siblings inlined at use sites (no array reads survive)"
    (count_get o a = 0 && count_get o b = 0);
  (* Sharing symbol [i] alone must not make either sibling complex. Phrased as the claim that holds,
     so that a regression prints [false] rather than turning a designed negative into an undesigned
     one (gh-ocannl-601). *)
  p "no is_complex from sharing alone" (not (is_complex o a || is_complex o b));
  let seed = [ (oa, blank 3); (ob, blank 3) ] and read = [ oa; ob ] in
  let virt = execute ~name:"vsl_independent" o ~seed ~read in
  let mat = execute ~name:"vsl_independent_mat" (optimize ~materialized:[ a; b ] llc) ~seed ~read in
  p "independent siblings: each use site got its own sibling's cell"
    (same virt [ [| 2.; 3.; 4. |]; [| 3.; 4.; 5. |] ]);
  p "independent siblings: virtual and materialized arms agree" (same virt mat)

(* === Case 2: mixed loop -- one sibling virtual, one materialized === *)
let case_mixed () =
  let a = mk "a" and b = mk "b" and oa = mk "oa" in
  materialize b;
  materialize oa;
  let i = sym () and j = sym () in
  let shared = loop_n i 3 (seq (set a [| iter i |] (ramp 2. i)) (set b [| iter i |] (ramp 3. i))) in
  let use_a = loop_n j 3 (set oa [| iter j |] (get a [| iter j |])) in
  let llc = seq shared use_a in
  let o = optimize llc in
  p "mixed cleanup keeps b setter" (count_set o b = 1);
  p "mixed drops virtual a setter" (count_set o a = 0);
  p "mixed a virtual, b non-virtual" (known_virtual o a && known_non_virtual o b);
  let seed = [ (b, blank 3); (oa, blank 3) ] and read = [ b; oa ] in
  let virt = execute ~name:"vsl_mixed" o ~seed ~read in
  let mat = execute ~name:"vsl_mixed_mat" (optimize ~materialized:[ a ] llc) ~seed ~read in
  p "mixed: the surviving setter still stores, the virtual sibling still reaches its use"
    (same virt [ [| 3.; 4.; 5. |]; [| 2.; 3.; 4. |] ]);
  p "mixed: virtual and materialized arms agree" (same virt mat)

(* === Case 3: forward sibling provider inlined into a surviving materialized reader === *)
let case_forward_provider () =
  let a = mk "a" and b = mk "b" in
  materialize b;
  let i = sym () in
  (* a written, then b reads a -- both in one loop; b survives (materialized), a virtual. *)
  let shared =
    loop_n i 3
      (seq (set a [| iter i |] (ramp 2. i)) (set b [| iter i |] (add (get a [| iter i |]) (c 1.))))
  in
  let o = optimize shared in
  p "forward provider inlined into materialized reader"
    (known_virtual o a && count_set o a = 0 && count_get o a = 0 && count_set o b = 1);
  let seed = [ (b, blank 3) ] and read = [ b ] in
  let virt = execute ~name:"vsl_forward" o ~seed ~read in
  let mat = execute ~name:"vsl_forward_mat" (optimize ~materialized:[ a ] shared) ~seed ~read in
  p "forward provider: the reader stored the provider's own cell, not a stale one"
    (same virt [ [| 3.; 4.; 5. |] ]);
  p "forward provider: virtual and materialized arms agree" (same virt mat)

(* === Case 4: forward virtual->virtual chain consumed downstream === *)
let case_chain () =
  let a = mk "a" and b = mk "b" and out = mk "out" in
  materialize out;
  let i = sym () and j = sym () in
  (* a = f; b = g(a); both virtual. out = h(b), materialized, read downstream. *)
  let shared =
    loop_n i 3
      (seq (set a [| iter i |] (ramp 2. i)) (set b [| iter i |] (add (get a [| iter i |]) (c 1.))))
  in
  let use_b = loop_n j 3 (set out [| iter j |] (mul (get b [| iter j |]) (c 2.))) in
  let llc = seq shared use_b in
  let o = optimize llc in
  p "forward virtual-to-virtual chain both virtual" (known_virtual o a && known_virtual o b);
  p "forward virtual-to-virtual chain fully inlined"
    (count_set o a = 0
    && count_set o b = 0
    && count_get o a = 0
    && count_get o b = 0
    && known_non_virtual o out);
  let seed = [ (out, blank 3) ] and read = [ out ] in
  let virt = execute ~name:"vsl_chain" o ~seed ~read in
  let mat = execute ~name:"vsl_chain_mat" (optimize ~materialized:[ a; b ] llc) ~seed ~read in
  p "forward chain: the doubly-inlined value is (2 + i + 1) * 2" (same virt [ [| 6.; 8.; 10. |] ]);
  p "forward chain: virtual and materialized arms agree" (same virt mat)

(* === Case 5: loop-carried / read-before-write sibling read stays materialized === [a] is written
   at [i] but read at [i+1] in the same loop, so the read of [a[i+1]] precedes its write in trace
   order (read-before-write). The existing access analysis records this as a recurrent access and
   forces [a] materialized; the later writer must NOT be used to rewrite the earlier read. This is
   the safety mechanism the proposal relies on (#134). *)
let case_reverse () =
  (* [a] is one cell wider than the loop so that the read-ahead of the last iteration stays in
     bounds: the executed leg reads this array for real. *)
  let a = mk ~dims:[| 4 |] "a" and b = mk "b" in
  materialize b;
  let i = sym () in
  let read_ahead = get a [| aff [ (1, i) ] 1 |] in
  let shared =
    loop_n i 3 (seq (set a [| iter i |] (c 2.)) (set b [| iter i |] (add read_ahead (c 1.))))
  in
  let o = optimize shared in
  p "loop-carried provider kept materialized" (known_non_virtual o a);
  p "loop-carried provider read NOT rewritten (array read preserved)" (count_get o a >= 1);
  (* The values make the safety mechanism observable: each [b.(i)] must come from the INCOMING
     [a.(i+1)], never from the [2.] the same loop stores one iteration later (which would give [3.]
     throughout). *)
  let seed = [ (a, [| 10.; 20.; 30.; 40. |]); (b, blank 3) ] and read = [ b; a ] in
  let got = execute ~name:"vsl_reverse" o ~seed ~read in
  p "loop-carried: reads saw the incoming values, writes landed"
    (same got [ [| 21.; 31.; 41. |]; [| 2.; 2.; 2.; 40. |] ])

(* === Case 6: is_complex still set by a genuine complex scalar computation === *)
let case_complex () =
  let x = mk "x" and y = mk "y" and z = mk "z" in
  materialize x;
  materialize y;
  materialize z;
  let i = sym () in
  let l = loop_n i 3 (set z [| iter i |] (mul (get x [| iter i |]) (get y [| iter i |]))) in
  let o = optimize l in
  p "is_complex from genuine complex scalar" (is_complex o z);
  let got =
    execute ~name:"vsl_complex" o
      ~seed:[ (x, [| 1.; 2.; 3. |]); (y, [| 4.; 5.; 6. |]); (z, blank 3) ]
      ~read:[ z ]
  in
  p "genuine complex scalar: executed elementwise product" (same got [ [| 4.; 10.; 18. |] ])

(* === Case 7: two virtual providers + an in-loop materialized consumer (Codex P1) === c
   (materialized) reads BOTH a and b in the same loop. The storage pass for the first candidate (a)
   walks c's setter, which reads the not-yet-stored b; it must not call inline_computation on b
   before b is stored (that raised a stale optimize_ctx error). Both providers must virtualize and
   inline into c, and c's setter must survive. *)
let case_inloop_consumer () =
  let a = mk "a" and b = mk "b" and cons = mk "cons" in
  materialize cons;
  let i = sym () in
  let shared =
    loop_n i 3
      (seq
         (set a [| iter i |] (ramp 2. i))
         (seq
            (set b [| iter i |] (ramp 3. i))
            (set cons [| iter i |] (add (get a [| iter i |]) (get b [| iter i |])))))
  in
  let o = optimize shared in
  p "in-loop consumer: both providers virtual" (known_virtual o a && known_virtual o b);
  p "in-loop consumer: providers inlined (no array reads survive)"
    (count_get o a = 0 && count_get o b = 0);
  p "in-loop consumer: consumer setter kept" (count_set o cons = 1);
  let seed = [ (cons, blank 3) ] and read = [ cons ] in
  let virt = execute ~name:"vsl_inloop" o ~seed ~read in
  let mat = execute ~name:"vsl_inloop_mat" (optimize ~materialized:[ a; b ] shared) ~seed ~read in
  p "in-loop consumer: both inlined providers reached the sum" (same virt [ [| 5.; 7.; 9. |] ]);
  p "in-loop consumer: virtual and materialized arms agree" (same virt mat)

(* === Case 9: a write under a dead loop ([to_ < from_]) never executes === The retired tracer never
   enumerated dead loops; the structural facts pass and the metric views must likewise record
   nothing from them: the node stays read-only (a routine input, not a spurious output). *)
let case_dead_loop () =
  let d = mk "dead" and out = mk "dlo" in
  materialize out;
  let i = sym () and j = sym () in
  let dead_write = loop ~upto:(-1) i (set d [| iter i |] (c 7.)) in
  let consume = loop_n j 3 (set out [| iter j |] (get d [| iter j |])) in
  let o = optimize (seq dead_write consume) in
  p "dead loop: node stays read-only" (read_only o d);
  let (inputs, outputs), _merge = LL.input_and_output_nodes o in
  p "dead loop: node is a routine input, not an output" (Set.mem inputs d && not (Set.mem outputs d));
  (* Had the dead loop run, the consumer would have copied [7.] instead of the seeded values. *)
  let got =
    execute ~name:"vsl_dead_loop" o
      ~seed:[ (d, [| 11.; 12.; 13. |]); (out, blank 3) ]
      ~read:[ out; d ]
  in
  p "dead loop: the consumer read the seeded values and the dead write never happened"
    (same got [ [| 11.; 12.; 13. |]; [| 11.; 12.; 13. |] ])

(* === Case 9b: a dead write supplies no coverage === The coverage-side companion of case 9: the
   dead write is dropped from the metric views, so the fixed-position read is read-before-write and
   the node is a routine input. *)
let case_dead_non_traced () =
  let d = mk "deadnt" and out = mk "dnto" in
  materialize out;
  let i = sym () and j = sym () in
  let dead_write = loop ~upto:(-1) i (set d [| iter i |] (c 7.)) in
  let consume = loop_n j 3 (set out [| iter j |] (get d [| fixed 0 |])) in
  let o = optimize (seq dead_write consume) in
  p "dead-write coverage: read_before_write set" (read_before_write o d);
  let (inputs, _outputs), _merge = LL.input_and_output_nodes o in
  p "dead-write coverage: node is a routine input" (Set.mem inputs d);
  let got =
    execute ~name:"vsl_dead_coverage" o
      ~seed:[ (d, [| 11.; 12.; 13. |]); (out, blank 3) ]
      ~read:[ out ]
  in
  p "dead-write coverage: the fixed-position read served the seeded cell"
    (same got [ [| 11.; 11.; 11. |] ])

(* === Case 10: an If condition's read is not a read-modify-write self-read === The condition reads
   [a] at the same position the guarded body writes it, and shares the body's program path; the
   exemption must not fire (the read executes before the write), so [a] is read-before-write — a
   routine input whose prior contents must be preserved. *)
let case_if_cond_read () =
  let a = mk "guarded" in
  let i = sym () in
  let guarded_update =
    loop_n i 3
      (LL.If
         {
           cond = (binop Ops.Cmplt (get a [| iter i |]) (c 1.), single);
           body = set a [| iter i |] (c 1.);
         })
  in
  let o = optimize guarded_update in
  p "if-cond read: read_before_write set" (read_before_write o a);
  let (inputs, _outputs), _merge = LL.input_and_output_nodes o in
  p "if-cond read: node is a routine input" (Set.mem inputs a);
  (* Only the cells whose incoming value is below 1 are clamped; the rest keep what was seeded, so
     the executed leg fails if the input contents were not preserved. *)
  let got = execute ~name:"vsl_if_cond" o ~seed:[ (a, [| 0.5; 2.; -1. |]) ] ~read:[ a ] in
  p "if-cond read: only the cells failing the guard were updated" (same got [ [| 1.; 2.; 1. |] ])

(* Each row runs through [case], so a row that raises is a failed claim naming it, and the rows
   after it still run (gh-ocannl-1084; [footprint_materialization] is the exemplar). *)
let () =
  case "independent" case_independent;
  case "mixed" case_mixed;
  case "forward-provider" case_forward_provider;
  case "chain" case_chain;
  case "reverse" case_reverse;
  case "complex" case_complex;
  case "inloop-consumer" case_inloop_consumer;
  case "dead-loop" case_dead_loop;
  case "dead-non-traced" case_dead_non_traced;
  case "if-cond-read" case_if_cond_read;
  Stdio.printf "%!"

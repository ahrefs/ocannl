(* Regression test for gh-ocannl-133 Stage B: virtualize injective affine producers (multi-symbol
   affine LHS indices).

   High-level lowering never produces these shapes directly in a controllable way, so -- like
   [virtual_shared_loop.ml] -- the cases are built directly as [Ir.Low_level.t] and run through the
   [Ll_test] harness (gh-ocannl-600), i.e. the same trace_node_facts -> virtual_llc -> cleanup ->
   simplify pipeline the backends use. We assert structurally on the optimized form (which producers
   virtualize, that no intermediate array read/setter survives, and which stay materialized).

   Cases: - structural affine match: producer [2*oh+wh] consumed at the same affine structure
   inlines with no intermediate buffer; - unit-coefficient solving at a plain iterator: producer
   [2*oh+wh] consumed at [t] inlines (the residual [oh] loop is kept and range-guarded); -
   triangular [(s1, s1+s2)]: unit-coefficient solving after pinning s1; - non-injective [i+j] (both
   ranges > 1) stays non-virtual, preserving a producer array read; - Stage A diagonal [i;i] still
   virtualizes (no regression).

   Structural pins say what the optimizer BUILT; virtualization rewrites what value a cell holds, so
   every case also has an EXECUTED leg (gh-ocannl-589): the very same [optimized] record is compiled
   through the [?prelowered] seam (gh-ocannl-562, worked out in [prelowered_seam.ml]), seeded, run,
   and its outputs checked against an OCaml reference, plus — where the producer virtualizes —
   against a second arm that re-specializes the same code with the producer's placement pre-decided
   [On_device]. That second arm is what pins the solved index and its range guard to the
   materialized reading of the same program: exactly the cells the scatter would have written must
   carry the scattered value, and the rest the init value. *)

open Base
open Ll_test
open Verdict.Claims

let mk = node_factory ~first_id:2000 ~dims:[| 6 |] ()
let optimize ?materialized llc = optimize ?materialized ~name:"virtual_affine" llc

(* === Case 1: structural affine match === *)
let case_structural_match () =
  let tgt = mk ~dims:[| 6 |] "smatch" and out = mk ~dims:[| 6 |] "out1" in
  materialize out;
  let oh = sym () and wh = sym () and a = sym () and b = sym () in
  (* Injective + surjective scatter over [0, 6): no zero-init (lowering payoff). *)
  let prod = loop_n oh 3 (loop_n wh 2 (set tgt [| aff [ (2, oh); (1, wh) ] 0 |] (tag oh wh))) in
  let cons =
    loop_n a 3
      (loop_n b 2 (set out [| aff [ (2, a); (1, b) ] 0 |] (get tgt [| aff [ (2, a); (1, b) ] 0 |])))
  in
  let llc = seq prod cons in
  let o = optimize llc in
  p "structural-match producer virtual" (known_virtual o tgt);
  p "structural-match producer inlined (no array reads survive)" (count_get o tgt = 0);
  p "structural-match producer setter dropped" (count_set o tgt = 0);
  p "structural-match consumer setter kept" (count_set o out = 1);
  (* Same affine structure on both sides: bound pairwise, no range/equality guard. *)
  let wh, le, lt = count_guard_ops o in
  p "structural-match has no guard ops" (wh = 0 && le = 0 && lt = 0);
  (* The scatter is surjective onto [0, 6), so every cell carries the value of the producer
     iteration that pairwise-bound to it; the sentinel survives anywhere the inlined nest lost a
     cell the producer covered. *)
  let scattered = Array.init 6 ~f:(fun n -> Float.of_int (1 + (10 * (n / 2)) + (n % 2))) in
  let seed = [ (out, blank 6) ] and read = [ out ] in
  let virt = execute ~name:"va_structural" o ~seed ~read in
  let mat = execute ~name:"va_structural_mat" (optimize ~materialized:[ tgt ] llc) ~seed ~read in
  p "structural-match: every cell holds the value of the iteration bound to it"
    (same virt [ scattered ]);
  p "structural-match: virtual and materialized arms agree" (same virt mat)

(* === Case 2: unit-coefficient solving at a plain iterator === *)
let case_unit_solve_plain () =
  let tgt = mk ~dims:[| 6 |] "usolve" and out = mk ~dims:[| 6 |] "out2" in
  materialize out;
  let oh = sym () and wh = sym () and t = sym () in
  let prod = loop_n oh 3 (loop_n wh 2 (set tgt [| aff [ (2, oh); (1, wh) ] 0 |] (tag oh wh))) in
  let cons = loop_n t 6 (set out [| iter t |] (get tgt [| iter t |])) in
  let llc = seq prod cons in
  let o = optimize llc in
  p "unit-solve(plain) producer virtual" (known_virtual o tgt);
  p "unit-solve(plain) producer inlined (no array reads survive)" (count_get o tgt = 0);
  p "unit-solve(plain) producer setter dropped" (count_set o tgt = 0);
  p "unit-solve(plain) consumer setter kept" (count_set o out = 1);
  (* Solving [wh = t - 2*oh] keeps the [oh] loop and range-guards [0 <= t-2*oh < 2]: a [Where] over
     an eager 0/1 product of a [Cmple] lower bound and a [Cmplt] upper bound. *)
  let wh, le, lt = count_guard_ops o in
  p "unit-solve(plain) emits a range guard (Where + Cmple lower + Cmplt upper)"
    (wh >= 1 && le >= 1 && lt >= 1);
  (* The residual [oh] loop folds over the producer's fiber, so the range guard is what keeps each
     [t] to the single [oh] that scattered it. Because the producer value identifies its iteration,
     a bound admitting a neighbouring [oh] lands that neighbour's value here instead. *)
  let scattered = Array.init 6 ~f:(fun n -> Float.of_int (1 + (10 * (n / 2)) + (n % 2))) in
  let seed = [ (out, blank 6) ] and read = [ out ] in
  let virt = execute ~name:"va_unit_solve" o ~seed ~read in
  let mat = execute ~name:"va_unit_solve_mat" (optimize ~materialized:[ tgt ] llc) ~seed ~read in
  p "unit-solve(plain): every cell holds the value of the one iteration that scattered it"
    (same virt [ scattered ]);
  p "unit-solve(plain): virtual and materialized arms agree" (same virt mat)

(* === Case 3: triangular (s1, s1 + s2), unit-coefficient solving after pinning s1 === *)
let case_triangular () =
  let tgt = mk ~dims:[| 3; 4 |] "tri" and out = mk ~dims:[| 3; 4 |] "out3" in
  materialize out;
  let s1 = sym () and s2 = sym () and a = sym () and b = sym () in
  (* Triangular map is injective but not surjective, so it carries a zero-init. *)
  let prod =
    seq (zero tgt)
      (loop_n s1 3 (loop_n s2 2 (set tgt [| iter s1; aff [ (1, s1); (1, s2) ] 0 |] (tag s1 s2))))
  in
  let cons =
    loop_n a 3 (loop_n b 4 (set out [| iter a; iter b |] (get tgt [| iter a; iter b |])))
  in
  let llc = seq prod cons in
  let o = optimize llc in
  p "triangular producer virtual" (known_virtual o tgt);
  p "triangular producer inlined (no array reads survive)" (count_get o tgt = 0);
  p "triangular consumer setter kept" (count_set o out = 1);
  (* s2 = b - a is solved and range-guarded [0 <= b-a < 2]. *)
  let wh, le, lt = count_guard_ops o in
  p "triangular emits a range guard (Where + Cmple lower + Cmplt upper)"
    (wh >= 1 && le >= 1 && lt >= 1);
  (* The scatter is injective but not surjective, so the guard has to separate the band it covers
     from the cells that keep the zero-init, and within the band the solved [s2 = b - a] has to pick
     the right producer iteration: [out.(a, b)] is [1 + 10a + (b - a)] exactly on [0 <= b - a < 2],
     and 0 elsewhere. *)
  let expected =
    Array.init 12 ~f:(fun n ->
        let a = n / 4 and b = n % 4 in
        if b - a >= 0 && b - a < 2 then Float.of_int (1 + (10 * a) + (b - a)) else 0.)
  in
  let seed = [ (out, blank 12) ] and read = [ out ] in
  let virt = execute ~name:"va_triangular" o ~seed ~read in
  let mat = execute ~name:"va_triangular_mat" (optimize ~materialized:[ tgt ] llc) ~seed ~read in
  p "triangular: executed values are the scattered band over the zero-init" (same virt [ expected ]);
  p "triangular: virtual and materialized arms agree" (same virt mat)

(* === Case 4: non-injective i+j (both ranges > 1) stays non-virtual === *)
let case_noninjective () =
  let tgt = mk ~dims:[| 5 |] "ni" and out = mk ~dims:[| 5 |] "out4" in
  materialize out;
  let i = sym () and j = sym () and a = sym () and b = sym () in
  let prod = loop_n i 3 (loop_n j 3 (set tgt [| aff [ (1, i); (1, j) ] 0 |] (c 1.))) in
  let cons =
    loop_n a 3
      (loop_n b 3 (set out [| aff [ (1, a); (1, b) ] 0 |] (get tgt [| aff [ (1, a); (1, b) ] 0 |])))
  in
  let o = optimize (seq prod cons) in
  (* i+j with both ranges > 1 is not injective: the dropped producer loops fold over a fiber, so the
     producer must stay materialized (the reason is the injectivity soundness line). *)
  p "non-injective producer stays non-virtual" (known_non_virtual o tgt);
  p "non-injective producer array read preserved" (count_get o tgt >= 1);
  (* Staying materialized is already the safe reading, so this case has no second arm - the executed
     leg pins that the fiber the producer folds over reaches the consumer through a buffer. The
     producer value stays constant here, unlike the guarded cases: nothing selects an iteration, so
     a per-iteration value would only pin which write of the fiber lands last, which is the loop
     order rather than the property under test. Only [out] is read back: [tgt] is non-virtual but
     unobservable, so the pipeline places it [Local] (routine-scoped scratch with no context
     buffer). The second fact pins that reading it back raises (gh-ocannl-599) rather than answering
     with whatever a host write left behind — which is how this case reported an array of sentinels
     when its executed leg was first written. *)
  let ctx = run ~name:"va_noninjective" o ~seed:[ (out, blank 5) ] in
  p "non-injective: the scattered value reached the consumer through the buffer"
    (close (Context.get_values ctx out) (Array.create ~len:5 1.));
  p "non-injective: the Local producer refuses host readback"
    (refused_as_local (fun () -> Context.get_values ctx tgt))

(* === Case 5: Stage A diagonal [i;i] still virtualizes (no regression) === *)
let case_stage_a_diagonal () =
  let d = mk ~dims:[| 3; 3 |] "diag" and out = mk ~dims:[| 3; 3 |] "out5" in
  materialize out;
  let i = sym () and a = sym () and b = sym () in
  let prod = seq (zero d) (loop_n i 3 (set d [| iter i; iter i |] (add (c 4.) (embed i)))) in
  let cons = loop_n a 3 (loop_n b 3 (set out [| iter a; iter b |] (get d [| iter a; iter b |]))) in
  let llc = seq prod cons in
  let o = optimize llc in
  p "stage-a diagonal producer virtual" (known_virtual o d);
  p "stage-a diagonal inlined (no array reads survive)" (count_get o d = 0);
  let expected =
    Array.init 9 ~f:(fun n -> if n / 3 = n % 3 then Float.of_int (4 + (n / 3)) else 0.)
  in
  let seed = [ (out, blank 9) ] and read = [ out ] in
  let virt = execute ~name:"va_stage_a" o ~seed ~read in
  let mat = execute ~name:"va_stage_a_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "stage-a diagonal: executed values are the diagonal over the zero-init" (same virt [ expected ]);
  p "stage-a diagonal: virtual and materialized arms agree" (same virt mat)

(* Each row runs through [case], so a row that raises is a failed claim naming it, and the rows
   after it still run (gh-ocannl-1084; [footprint_materialization] is the exemplar). *)
let () =
  case "structural-match" case_structural_match;
  case "unit-solve(plain)" case_unit_solve_plain;
  case "triangular" case_triangular;
  case "non-injective" case_noninjective;
  case "stage-a diagonal" case_stage_a_diagonal;
  Stdio.printf "%!"

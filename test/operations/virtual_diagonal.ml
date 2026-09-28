(* Regression test for gh-ocannl-133 Stage A: virtualize producers whose index vector repeats a
   non-static symbol (diagonal [i;i] / partially-diagonal [i;j;i]) plus covered single-symbol affine
   positions.

   High-level lowering never produces a [Get] of a virtual diagonal with two distinct call-site
   symbols in one place (each assignment lowers to its own loop nest), so -- like
   [virtual_shared_loop] -- these cases are built directly as [Ir.Low_level.t] and run through the
   [Ll_test] harness (gh-ocannl-600), the same pipeline (trace_node_facts -> virtual_llc ->
   cleanup_virtual_llc -> simplify -> CSE -> hoist) the backends use. We assert structurally on the
   optimized form: that the diagonal producer virtualizes, that its reads are inlined, and that an
   equality guard ([Where (Cmpeq ...)]) is emitted exactly when the read uses distinct/dynamic
   indices and folded away when the read indices are syntactically equal.

   [Concat] virtualization stays out of scope: a [Concat] index is eliminated during lowering and
   [trace_node_facts] raises if one ever reaches this pass, so it cannot be exercised through
   [optimize] here. The "Concat remains rejected" criterion is the unchanged [check_idcs]
   [Non_virtual 52] branch plus the existing test_concat_graph / test_block_tensor coverage.

   Structural pins say what the optimizer BUILT; virtualization rewrites what value a cell holds, so
   every case also has an EXECUTED leg (gh-ocannl-589): the very same [optimized] record is compiled
   through the [?prelowered] seam (gh-ocannl-562, worked out in [prelowered_seam.ml]), seeded, run,
   and its outputs checked against an OCaml reference, plus against a second arm that re-specializes
   the same code with the producer's placement pre-decided [On_device]. That second arm is what pins
   the guarded inline to the materialized reading of the same program: the guard must reproduce,
   cell by cell, what the producer's buffer would have held. *)

open Base
open Ll_test
open Verdict.Claims

let mk = node_factory ~first_id:2000 ~dims:[| 3; 3 |] ()
let optimize ?materialized llc = optimize ?materialized ~name:"virtual_diagonal" llc

(* Every loop here has width 3, matching the default node dims. *)
let loop s body = loop_n s 3 body

(* === Case 1: diagonal producer read by a generic (distinct-symbol) consumer === d[i,i] = 1 + i
   (off-diagonal zero); a materialized consumer reads d[j,k]. The diagonal must virtualize, its read
   must be inlined, and exactly one equality guard must survive (j = k). *)
let case_diagonal_generic () =
  let d = mk "d" and o = mk "o" in
  materialize o;
  let i = sym () and j = sym () and k = sym () in
  let producer = seq (zero d) (loop i (set d [| iter i; iter i |] (tick i))) in
  let consumer = loop j (loop k (set o [| iter j; iter k |] (get d [| iter j; iter k |]))) in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "diagonal-generic: producer virtual" (known_virtual opt d);
  p "diagonal-generic: read inlined (no array read of d)" (count_get opt d = 0);
  p "diagonal-generic: one equality guard survives" (count_where opt >= 1);
  p "diagonal-generic: consumer read inlined under guard" (count_get opt d = 0);
  (* The guard is the whole content of the case: [o] must be the diagonal matrix, not the producer
     value smeared over every cell (guard dropped) and not all zeros (guard always failing). *)
  let expected =
    Array.init 9 ~f:(fun n -> if n / 3 = n % 3 then Float.of_int ((n / 3) + 1) else 0.)
  in
  let seed = [ (o, blank 9) ] and read = [ o ] in
  let virt = execute ~name:"vd_generic" opt ~seed ~read in
  let mat = execute ~name:"vd_generic_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "diagonal-generic: executed values are the diagonal" (same virt [ expected ]);
  p "diagonal-generic: virtual and materialized arms agree" (same virt mat)

(* === Case 2: diagonal producer read at equal indices -> guard simplifies away === *)
let case_diagonal_equal () =
  let d = mk "d" and o = mk "o" in
  materialize o;
  let i = sym () and j = sym () in
  let producer = seq (zero d) (loop i (set d [| iter i; iter i |] (tick i))) in
  (* read d[j,j]: the two call-site indices are syntactically equal, so no guard. *)
  let consumer = loop j (set o [| iter j; iter j |] (get d [| iter j; iter j |])) in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "diagonal-equal: producer virtual" (known_virtual opt d);
  p "diagonal-equal: read inlined (no array read of d)" (count_get opt d = 0);
  p "diagonal-equal: guard folded away (no Where)" (count_where opt = 0);
  (* Folding the guard away must not fold the producer's index dependence away with it: the consumer
     writes only its own diagonal, and each cell there holds that row's producer value. *)
  let expected =
    Array.init 9 ~f:(fun n -> if n / 3 = n % 3 then Float.of_int ((n / 3) + 1) else sentinel)
  in
  let seed = [ (o, blank 9) ] and read = [ o ] in
  let virt = execute ~name:"vd_equal" opt ~seed ~read in
  let mat = execute ~name:"vd_equal_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "diagonal-equal: executed values are the diagonal, off-diagonal untouched"
    (same virt [ expected ]);
  p "diagonal-equal: virtual and materialized arms agree" (same virt mat)

(* === Case 3: partially-diagonal producer [i;j;i] read generically === d[i,j,i] = 1 + 10i + j;
   consumer reads d[a,b,cc]. Repeated i guards (a = cc); j substituted normally. *)
let case_partial_diagonal () =
  let d = mk ~dims:[| 3; 3; 3 |] "pd" and o = mk ~dims:[| 3; 3; 3 |] "po" in
  materialize o;
  let i = sym () and j = sym () in
  let a = sym () and b = sym () and cc = sym () in
  let producer = seq (zero d) (loop i (loop j (set d [| iter i; iter j; iter i |] (tag i j)))) in
  let consumer =
    loop a
      (loop b (loop cc (set o [| iter a; iter b; iter cc |] (get d [| iter a; iter b; iter cc |]))))
  in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "partial-diagonal: producer virtual" (known_virtual opt d);
  p "partial-diagonal: read inlined (no array read of d)" (count_get opt d = 0);
  p "partial-diagonal: one equality guard survives" (count_where opt >= 1);
  (* Guarded on the repeated axis only, and the non-repeated [b] is substituted normally, so it
     shows through in the value: [o.(a, b, cc)] is [1 + 10a + b] when [a = cc], and the init
     otherwise. *)
  let expected =
    Array.init 27 ~f:(fun n ->
        let a = n / 9 and b = n / 3 % 3 and cc = n % 3 in
        if a = cc then Float.of_int (1 + (10 * a) + b) else 0.)
  in
  let seed = [ (o, blank 27) ] and read = [ o ] in
  let virt = execute ~name:"vd_partial" opt ~seed ~read in
  let mat = execute ~name:"vd_partial_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "partial-diagonal: executed values guard the repeated axis only" (same virt [ expected ]);
  p "partial-diagonal: virtual and materialized arms agree" (same virt mat)

(* === Case 4: static-vs-dynamic read of a diagonal producer === Read d[0, j] (a row slice): the
   first position is bound to Fixed_idx 0, the second is dynamic; the consistency must become a
   guard (0 = j), NOT a Non_virtual 13 rejection. *)
let case_static_dynamic () =
  let d = mk "d" and o = mk ~dims:[| 3 |] "o" in
  materialize o;
  let i = sym () and j = sym () in
  let producer = seq (zero d) (loop i (set d [| iter i; iter i |] (tick i))) in
  let consumer = loop j (set o [| iter j |] (get d [| fixed 0; iter j |])) in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "static-dynamic: producer virtual" (known_virtual opt d);
  p "static-dynamic: read inlined (no array read of d)" (count_get opt d = 0);
  p "static-dynamic: one equality guard survives" (count_where opt >= 1);
  let seed = [ (o, blank 3) ] and read = [ o ] in
  let virt = execute ~name:"vd_static_dynamic" opt ~seed ~read in
  let mat = execute ~name:"vd_static_dynamic_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "static-dynamic: executed values are row 0 of the diagonal" (same virt [ [| 1.; 0.; 0. |] ]);
  p "static-dynamic: virtual and materialized arms agree" (same virt mat)

(* === Case 5: covered single-symbol affine producer position === Producer d[i, i+1] (single-symbol
   affine, covered by the bare iterator i) read at d[j, j+1]; this must validate after substitution
   (no Non_virtual 13) and inline with no surviving guard. *)
let case_single_symbol_affine () =
  (* One column wider than the diagonal cases: the producer's [i + 1] position reaches column 3, and
     the materialized arm writes it for real. *)
  let d = mk ~dims:[| 3; 4 |] "d" and o = mk ~dims:[| 3 |] "o" in
  materialize o;
  let i = sym () and j = sym () in
  let plus1 s = aff [ (1, s) ] 1 in
  let producer = seq (zero d) (loop i (set d [| iter i; plus1 i |] (tick i))) in
  let consumer = loop j (set o [| iter j |] (get d [| iter j; plus1 j |])) in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "single-affine: producer virtual" (known_virtual opt d);
  p "single-affine: read inlined (no array read of d)" (count_get opt d = 0);
  p "single-affine: no guard for matching affine" (count_where opt = 0);
  (* Unguarded, but still per-index: [o.(j)] is the producer value at [i = j]. *)
  let seed = [ (o, blank 3) ] and read = [ o ] in
  let virt = execute ~name:"vd_affine" opt ~seed ~read in
  let mat = execute ~name:"vd_affine_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "single-affine: executed values track the producer index" (same virt [ [| 1.; 2.; 3. |] ]);
  p "single-affine: virtual and materialized arms agree" (same virt mat)

(* === Case 5b: covered single-symbol affine producer read at a MISMATCHED offset === Producer d[i,
   i+1] read at d[j, j+2]: the substituted producer index (j+1) does not equal the call-site index
   (j+2), but the position is covered (symbol j is bound), so it must virtualize with a SURVIVING
   equality guard (j+1 = j+2, which folds to the init value) -- NOT a Non_virtual 13 deferral to
   materialization. *)
let case_single_symbol_affine_mismatch () =
  (* Wide enough for both the producer's [j + 1] writes and the consumer's [j + 2] reads, which the
     materialized arm performs against a real buffer. *)
  let d = mk ~dims:[| 3; 5 |] "d" and o = mk ~dims:[| 3 |] "o" in
  materialize o;
  let i = sym () and j = sym () in
  let plus ofs s = aff [ (1, s) ] ofs in
  let producer = seq (zero d) (loop i (set d [| iter i; plus 1 i |] (tick i))) in
  let consumer = loop j (set o [| iter j |] (get d [| iter j; plus 2 j |])) in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "single-affine-mismatch: producer virtual" (known_virtual opt d);
  p "single-affine-mismatch: read inlined (no array read of d)" (count_get opt d = 0);
  p "single-affine-mismatch: equality guard survives" (count_where opt >= 1);
  (* The guard can never hold, so every cell falls back on the init value, and the materialized arm
     agrees, because [d(j, j + 2)] is off the written diagonal and holds the zero-init. *)
  let seed = [ (o, blank 3) ] and read = [ o ] in
  let virt = execute ~name:"vd_affine_mismatch" opt ~seed ~read in
  let mat = execute ~name:"vd_affine_mismatch_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "single-affine-mismatch: executed values are the init value" (same virt [ [| 0.; 0.; 0. |] ]);
  p "single-affine-mismatch: virtual and materialized arms agree" (same virt mat)

(* === Case 6: single-symbol (non-repeated) producer still virtualizes -- regression guard === *)
let case_single_symbol () =
  let d = mk ~dims:[| 3 |] "d" and o = mk ~dims:[| 3 |] "o" in
  materialize o;
  let i = sym () and j = sym () in
  let producer = loop i (set d [| iter i |] (tick i)) in
  let consumer = loop j (set o [| iter j |] (get d [| iter j |])) in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "single-symbol: producer virtual" (known_virtual opt d);
  p "single-symbol: read inlined (no array read of d)" (count_get opt d = 0);
  p "single-symbol: no guard" (count_where opt = 0);
  let seed = [ (o, blank 3) ] and read = [ o ] in
  let virt = execute ~name:"vd_single" opt ~seed ~read in
  let mat = execute ~name:"vd_single_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "single-symbol: executed values track the producer index" (same virt [ [| 1.; 2.; 3. |] ]);
  p "single-symbol: virtual and materialized arms agree" (same virt mat)

(* Each row runs through [case], so a row that raises is a failed claim naming it, and the rows
   after it still run (gh-ocannl-1084; [footprint_materialization] is the exemplar). *)
let () =
  case "diagonal-generic" case_diagonal_generic;
  case "diagonal-equal" case_diagonal_equal;
  case "partial-diagonal" case_partial_diagonal;
  case "static-dynamic" case_static_dynamic;
  case "single-affine" case_single_symbol_affine;
  case "single-affine-mismatch" case_single_symbol_affine_mismatch;
  case "single-symbol" case_single_symbol;
  Stdio.printf "%!"

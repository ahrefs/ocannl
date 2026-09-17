(* Regression test for gh-ocannl-674: a candidate whose captured computation sits inside a loop the
   capture does not contain used to virtualize, and the consumer then replayed the setter ONCE
   instead of once per iteration of that loop — the fold lost every iteration but one.

   [virtual_llc] captures a candidate's computation at the outermost [For_loop] whose index appears
   in the candidate's assignment indices ([track_symbol] / [reverse_node_map]). A reduction loop
   BELOW that point rides along inside the stored subtree, which is what makes the ordinary [x[t] +=
   a[s]] shape inlineable at all (and what [virtualize_max_inline_reduction] prices). A repetition
   loop ABOVE it does not ride along, and when the index map mentions no symbol at all — a
   fixed-index accumulator — no loop is ever a capture site, so the entire reduction is outside the
   stored statement. The fix threads the enclosing loops down the walk and rejects such a candidate
   as [Non_virtual 147] at store time; this is the [For_loop] half of the defect gh-ocannl-651 fixed
   for [If].

   Two positive controls guard the other direction, because a rejection is always SOUND and the only
   way this fix can be wrong is by rejecting too much: the ordinary inner-reduction shape must still
   inline, and a width-1 enclosing loop must stay exempt (replaying a single-iteration loop once is
   exact, so the width test has to be [> 1] rather than "any enclosing loop").

   Hand-built [Ir.Low_level.t] through the [Ll_test] harness (gh-ocannl-600). Each case executes
   both readings of the same program — the optimized one and the differential arm with the candidate
   pre-decided [On_device] — since a dropped repetition is invisible structurally: an inlined body
   missing an outer loop looks exactly like an inlined body that never had one. *)

open Base
open Ll_test
open Verdict.Claims

let mk = node_factory ~first_id:2800 ~dims:[| 4 |] ()
let n = 4

(* Every case reads back [out] under both placements, against a reference stated independently. *)
let both ~label ~llc ~cand ~out ~seed ~expected =
  let o = optimize ~name:label llc in
  let len = Array.length expected in
  let seed = (out, blank len) :: seed in
  let got = execute ~name:label o ~seed ~read:[ out ] in
  let mat =
    execute ~name:(label ^ "_mat")
      (optimize ~materialized:[ cand ] ~name:(label ^ "_mat") llc)
      ~seed ~read:[ out ]
  in
  p (label ^ ": executed values are the reference") (same got [ expected ]);
  p (label ^ ": virtual and materialized arms agree") (same got mat);
  o

(* === Case 1: a repetition loop above the capture point ===

   [x]'s index map mentions [t], so the capture is the [t] loop; the [k] loop above it repeats that
   whole nest into the same cells. Each cell must end at 2. *)
let case_repetition_above () =
  let x = mk "ucl_rep_x" and out = mk "ucl_rep_out" in
  materialize out;
  let k = sym () and t = sym () and u = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n k 2 (loop_n t n (set x [| iter t |] (add (get x [| iter t |]) (c 1.)))))
         (loop_n u n (set out [| iter u |] (get x [| iter u |]))))
  in
  let o =
    both ~label:"repetition_above" ~llc ~cand:x ~out ~seed:[] ~expected:(Array.create ~len:n 2.)
  in
  p "repetition above the capture: candidate rejected (stays non-virtual)" (known_non_virtual o x);
  p "repetition above the capture: rejected as Non_virtual 147"
    (Option.equal Tn.equal_provenance (rejection_code o x)
       (Some (Tn.Site "147:enclosing-repetition-loop")))

(* === Case 2: no capture point at all ===

   A symbol-free index map makes no loop a capture site, so the reduction loop is outside the stored
   statement entirely. The array-reduction spelling of this shape ([x[0] += a[s]]) is rejected by an
   unrelated arm — the sibling read [a[s]] escapes the captured statement, [Non_virtual 9] — so the
   case that reaches here accumulates a value the reduction symbol does not enter. Reading the
   accumulator ONCE is what gets past the visit cap; a second read site would trip it, and the cap
   is flippable policy rather than a legality guarantee. *)
let case_symbol_free_map () =
  let x = mk ~dims:[| 1 |] "ucl_sf_x" and out = mk ~dims:[| 1 |] "ucl_sf_out" in
  materialize out;
  let s = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n s n (set x [| fixed 0 |] (add (get x [| fixed 0 |]) (c 1.))))
         (set out [| fixed 0 |] (get x [| fixed 0 |])))
  in
  let o = both ~label:"symbol_free_map" ~llc ~cand:x ~out ~seed:[] ~expected:[| 4. |] in
  p "symbol-free index map: candidate rejected (stays non-virtual)" (known_non_virtual o x);
  p "symbol-free index map: rejected as Non_virtual 147"
    (Option.equal Tn.equal_provenance (rejection_code o x)
       (Some (Tn.Site "147:enclosing-repetition-loop")))

(* === Case 3 (positive control): the ordinary inner reduction still inlines ===

   Here the reduction loop is BELOW the capture point, so it is part of the stored computation and
   the consumer replays it in full. This is the shape the whole recompute-cost machinery
   ([virtualize_max_inline_reduction]) exists to price, so rejecting it would be a real loss. *)
let case_inner_reduction () =
  let a = mk "ucl_in_a" and x = mk "ucl_in_x" and out = mk "ucl_in_out" in
  materialize a;
  materialize out;
  let t = sym () and s = sym () and u = sym () in
  let avals = [| 1.; 2.; 3.; 4. |] in
  let llc =
    seq (zero x)
      (seq
         (loop_n t n
            (loop_n s n (set x [| iter t |] (add (get x [| iter t |]) (get a [| iter s |])))))
         (loop_n u n (set out [| iter u |] (get x [| iter u |]))))
  in
  let o =
    both ~label:"inner_reduction" ~llc ~cand:x ~out
      ~seed:[ (a, avals) ]
      ~expected:(Array.create ~len:n 10.)
  in
  p "inner reduction: candidate still virtual" (known_virtual o x);
  p "inner reduction: inlined (no array reads survive)" (count_get o x = 0)

(* === Case 4 (positive control): a width-1 enclosing loop stays exempt ===

   One iteration replayed once is exact, so the enclosing-loop test is on the WIDTH, not on the mere
   presence of an uncaptured loop. *)
let case_width_one_enclosing () =
  let x = mk "ucl_w1_x" and out = mk "ucl_w1_out" in
  materialize out;
  let k = sym () and t = sym () and u = sym () in
  let llc =
    seq
      (loop_n k 1 (loop_n t n (set x [| iter t |] (tick t))))
      (loop_n u n (set out [| iter u |] (get x [| iter u |])))
  in
  let o =
    both ~label:"width_one_enclosing" ~llc ~cand:x ~out ~seed:[]
      ~expected:(Array.init n ~f:(fun i -> 1. +. Float.of_int i))
  in
  p "width-1 enclosing loop: candidate still virtual" (known_virtual o x);
  p "width-1 enclosing loop: inlined (no array reads survive)" (count_get o x = 0)

let () =
  case_repetition_above ();
  case_symbol_free_map ();
  case_inner_reduction ();
  case_width_one_enclosing ();
  Stdio.printf "%!"

(* Regression test for gh-ocannl-651: a candidate whose setter nest sits inside an enclosing [If]
   used to virtualize, and [inline_computation] then replayed the stored computation WITHOUT the
   guard at every read site — the guard was silently dropped rather than the candidate rejected.

   The rejection the [Low_level.t] doc promises ("a conditional write is never a definite write;
   virtualization treats guarded computations as non-inlineable in v1") was enforced only for a
   guard INTERIOR to the captured subtree: [virtual_llc]'s [If] arm recursed plainly, and the
   setter/loop arms handed [check_and_store_virtual] the subtree rooted BELOW the guard, so the walk
   that raises [Non_virtual 142] never saw it. The fix threads the enclosing-guard flag down the
   walk and rejects at store time.

   These are hand-built [Ir.Low_level.t] cases through the [Ll_test] harness (gh-ocannl-600): the
   [Assignments] pipeline does emit pre-virtualization guards (interval guards for clamped-window
   pooling, gh-504; extent guards for symbolic extents, gh-490), but not in a shape that lets a test
   dial the guard's value and read the difference.

   Each case is EXECUTED at BOTH values of a runtime flag, because that is the only thing a dropped
   guard changes: structurally, an unguarded replay of a correct computation looks like successful
   inlining. The off-value run is the one that failed before the fix (it produced the on-value
   answer). Each case also carries the differential arm — the same program re-specialized with the
   candidate pre-decided [On_device] — so the rejected reading is pinned against an independently
   materialized one rather than against this test's own arithmetic alone.

   When guarded computations become inlineable (prepending the guard to the stored computation, the
   lift adjacent to the Block virtualizer's range-guard machinery), the value claims below hold
   unchanged and only the placement claims flip: that is the intended minor adjustment. *)

open Base
open Ll_test
open Verdict.Claims

let mk = node_factory ~first_id:2600 ~dims:[| 4 |] ()
let n = 4

(* Discriminating operand: varies with the index and stays off both 0 (the init the guard's
   off-branch leaves behind) and the sentinel. *)
let avals = Array.init n ~f:(fun i -> 2. +. Float.of_int i)

(** Runs [o] once per flag value, returning [(off, on)] readings of [out]. Linking once and seeding
    twice is what keeps the two readings the SAME compiled routine — a per-value recompile could
    differ for reasons other than the guard. *)
let run_both o ~name ~flag ~seed ~out =
  let linked = link ~name o in
  let read v =
    let ctx = run_linked linked ~seed:((flag, [| v |]) :: seed) in
    Context.get_values ctx out
  in
  let off = read 0. in
  let on = read 1. in
  (off, on)

(* === Case 1: the issue's reproducer — a guarded RMW update of a zero-initialized candidate ===

   [x] is written unguarded ([Zero_out]) and then updated under the guard, so both store sites see
   the candidate: the [Zero_out] arm unguarded, the [For_loop] candidate arm inside the [If]. *)
let case_guarded_rmw () =
  let flag = mk ~dims:[| 1 |] "grmw_flag"
  and a = mk "grmw_a"
  and x = mk "grmw_x"
  and out = mk "grmw_out" in
  List.iter [ flag; a; out ] ~f:materialize;
  let s = sym () and t = sym () in
  let llc =
    seq (zero x)
      (seq
         (if_
            (get flag [| fixed 0 |])
            (loop_n s n (set x [| iter s |] (add (get x [| iter s |]) (get a [| iter s |])))))
         (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  let o = optimize ~name:"vgs_rmw" llc in
  p "guarded RMW: candidate rejected (stays non-virtual)" (known_non_virtual o x);
  p "guarded RMW: the consumer reads the candidate through a buffer" (count_get o x >= 1);
  p "guarded RMW: the guard survives in the emitted code" (count_if o >= 1);
  let seed = [ (a, avals); (out, blank n) ] in
  let off, on = run_both o ~name:"vgs_rmw" ~flag ~seed ~out in
  p "guarded RMW: flag off leaves the zero init" (close off (Array.create ~len:n 0.));
  p "guarded RMW: flag on applies the update" (close on avals);
  let moff, mon =
    run_both
      (optimize ~materialized:[ x ] ~name:"vgs_rmw_mat" llc)
      ~name:"vgs_rmw_mat" ~flag ~seed ~out
  in
  p "guarded RMW: agrees with the materialized arm at both flag values"
    (close off moff && close on mon)

(* === Case 2: a guarded [Zero_out] resetting an unguarded full write ===

   The candidate is fully defined without the guard, so the off-value run has a well-defined answer
   whichever way the placement goes; what the guard decides is whether the reset happened. This
   exercises the [Zero_out] store site under an enclosing guard, which the [For_loop] arm of case 1
   does not reach. *)
let case_guarded_reset () =
  let flag = mk ~dims:[| 1 |] "grst_flag" and x = mk "grst_x" and out = mk "grst_out" in
  List.iter [ flag; out ] ~f:materialize;
  let s = sym () and t = sym () in
  let llc =
    seq
      (loop_n s n (set x [| iter s |] (tick s)))
      (seq
         (if_ (get flag [| fixed 0 |]) (zero x))
         (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  let o = optimize ~name:"vgs_reset" llc in
  p "guarded reset: candidate rejected (stays non-virtual)" (known_non_virtual o x);
  p "guarded reset: the consumer reads the candidate through a buffer" (count_get o x >= 1);
  p "guarded reset: the guard survives in the emitted code" (count_if o >= 1);
  let ticks = Array.init n ~f:(fun i -> 1. +. Float.of_int i) in
  let seed = [ (out, blank n) ] in
  let off, on = run_both o ~name:"vgs_reset" ~flag ~seed ~out in
  p "guarded reset: flag off keeps the unguarded write" (close off ticks);
  p "guarded reset: flag on applies the reset" (close on (Array.create ~len:n 0.));
  let moff, mon =
    run_both
      (optimize ~materialized:[ x ] ~name:"vgs_reset_mat" llc)
      ~name:"vgs_reset_mat" ~flag ~seed ~out
  in
  p "guarded reset: agrees with the materialized arm at both flag values"
    (close off moff && close on mon)

(* === Case 3: the interior guard, which already rejected — pinned as the contrast ===

   Here the [If] sits INSIDE the candidate's loop, so the captured subtree contains it and the
   walk's own [If] arm fires. Same verdict, different arm: the pair is what says the fix moved the
   enclosing case onto the documented contract rather than inventing a new one. *)
let case_interior_guard () =
  let mask = mk "gint_mask" and x = mk "gint_x" and out = mk "gint_out" in
  List.iter [ mask; out ] ~f:materialize;
  let s = sym () and t = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n s n (if_ (get mask [| iter s |]) (set x [| iter s |] (tick s))))
         (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  let o = optimize ~name:"vgs_interior" llc in
  p "interior guard: candidate rejected (stays non-virtual)" (known_non_virtual o x);
  p "interior guard: the consumer reads the candidate through a buffer" (count_get o x >= 1);
  (* Odd cells masked in, even cells masked out: a per-cell mask discriminates in a way a uniform
     flag cannot — a guard dropped per cell would show up as the even cells carrying [tick]. *)
  let maskv = Array.init n ~f:(fun i -> Float.of_int (i % 2)) in
  let expected = Array.init n ~f:(fun i -> if i % 2 = 1 then 1. +. Float.of_int i else 0.) in
  let seed = [ (mask, maskv); (out, blank n) ] in
  let got = execute ~name:"vgs_interior" o ~seed ~read:[ out ] in
  let mat =
    execute ~name:"vgs_interior_mat"
      (optimize ~materialized:[ x ] ~name:"vgs_interior_mat" llc)
      ~seed ~read:[ out ]
  in
  p "interior guard: only the masked-in cells carry the write" (same got [ expected ]);
  p "interior guard: agrees with the materialized arm" (same got mat)

(* Each row runs through [case], so a row that raises is a failed claim naming it, and the rows
   after it still run (gh-ocannl-1084; [footprint_materialization] is the exemplar). *)
let () =
  case "guarded RMW" case_guarded_rmw;
  case "guarded reset" case_guarded_reset;
  case "interior guard" case_interior_guard;
  Stdio.printf "%!"

(* gh-ocannl-2112: [check_half_prec_constants_cutoff] guards the VALUES a half-precision node
   computes, so it must read each constant at the precision its slot evaluates it at (the
   simplifier's discipline, gh-ocannl-1788), not at the storage precision of the node the statement
   writes. An index bound is evaluated at index precision: [Stage]'s per-axis edge guard [Where (idx
   < extent, load, 0)] compares at [Ops.index_prec ()], so an extent past the cutoff is a legal
   index bound, never an fp16 constant. The refusal named it as one ("Constant 190479. is too big
   for FP16") for a 190,475-cell half panel with hoisting off.

   The case is that guard's shape, hand-built: a half node of [n] cells whose first [bound] cells
   copy a half source and whose fringe stores 0, with [bound = 70000] past both the cutoff (2^14)
   and fp16's largest finite value (65504), and over a loop the interval analysis cannot decide the
   guard on. The source cells [1 + (i mod 1000)] are exact in half, nonzero (so the fringe's 0 is
   visible), and the destination is prefilled before the run (so a store skipped reads wrong).

   The control keeps the check honest: the same bound stored as a VALUE into the half node is still
   refused. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
open Verdict.Claims

let n = 70010
let bound = 70000
let src_value i = Float.of_int (1 + (i % 1000))
let refused_as_fp16 msg = String.is_substring msg ~substring:"is too big for FP16"

let guarded_copy ~first_id =
  let mk = Ll_test.node_factory ~prec:Ops.half ~first_id ~dims:[| n |] () in
  let src = mk "hib_src" and dst = mk "hib_dst" in
  Ll_test.materialize src;
  Ll_test.materialize dst;
  let i = Ll_test.sym () in
  let cond = Ll_test.lt (Ll_test.embed i) (Ll_test.ic bound) in
  let guarded =
    LL.Ternop
      ( Ops.Where,
        (cond, Ll_test.iprec ()),
        (Ll_test.get src [| Ll_test.iter i |], Ops.half),
        (LL.Constant 0., Ops.half) )
  in
  (Ll_test.loop_n i n (Ll_test.set_at dst (Ll_test.iter i) guarded), src, dst)

let check_guard () =
  let llc, src, dst = guarded_copy ~first_id:2112000 in
  let name = "hib_guard" in
  match Ll_test.optimize ~materialized:[ src; dst ] ~name llc with
  | exception Utils.User_error msg ->
      Stdio.prerr_endline ("refusal (not part of the golden): " ^ msg);
      p ("an index bound of " ^ Int.to_string bound ^ " in a half node compiles") false
  | optimized ->
      p "the guard is not folded away (the loop cannot decide it)"
        (Ll_test.count_scalar optimized.LL.llc ~f:(function
           | LL.Ternop (Ops.Where, _, _, _) -> true
           | _ -> false)
        > 0);
      let seed = Array.init n ~f:src_value in
      let got = Ll_test.execute ~name optimized ~seed:[ (src, seed) ] ~read:[ dst ] in
      let expected = Array.init n ~f:(fun i -> if i < bound then src_value i else 0.) in
      p "the guarded copy matches the host reference, fringe included"
        (Ll_test.same ~tol:0. got [ expected ])

(* A value the half node stores: as the whole right-hand side, and as a [Where] arm beside an
   index-precision condition (the arms evaluate at the store's precision, only the condition at its
   own). *)
let check_value_controls () =
  let mk = Ll_test.node_factory ~prec:Ops.half ~first_id:2112010 ~dims:[| 2 |] () in
  let refused ~label llsc =
    let dst = mk ("hib_" ^ label) in
    Ll_test.materialize dst;
    let i = Ll_test.sym () in
    let llc = Ll_test.loop_n i 2 (Ll_test.set_at dst (Ll_test.iter i) (llsc i)) in
    match Ll_test.optimize ~materialized:[ dst ] ~name:("hib_" ^ label) llc with
    | exception Utils.User_error msg -> refused_as_fp16 msg
    | _ -> false
  in
  let big = LL.Constant (Float.of_int bound) in
  p "the same bound stored as a half VALUE is still refused" (refused ~label:"value" (fun _ -> big));
  p "the same bound as a half Where ARM is still refused"
    (refused ~label:"arm" (fun i ->
         LL.Ternop
           ( Ops.Where,
             (Ll_test.lt (Ll_test.embed i) (Ll_test.ic 1), Ll_test.iprec ()),
             (big, Ops.half),
             (LL.Constant 0., Ops.half) )))

let () =
  check_guard ();
  check_value_controls ()

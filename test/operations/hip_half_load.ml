(* gh-ocannl-1182: the first GPT attention-softmax numerator was corrupted on gfx1102 while its
   scores, maxima and denominators remained identical. The emitted kernel packs the two half loads
   into one VGPR and updates its low half before the high load completes. A standalone replay failed
   in >128000/8388608 cells; waiting for outstanding loads made three controls pass without changing
   half arithmetic.

   This is the same consumer geometry and expression through the shipped compiler, with every input
   initialized and NaN-poisoned output. Constants are deliberate: exp(-1)/2 versus exp(+1)/2
   distinguishes the lost subtraction, and masked zero distinguishes corruption from ordinary half
   rounding. It is not an optimizer iteration-substitution test. Non-HIP runs use fewer query rows
   to keep CPU coverage inexpensive. The physical HIP shape exceeds the GPU cache's tiny-row
   regime. *)
open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module L = Ll_test
module LL = Ir.Low_level
module Ops = Ir.Ops

let () =
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let queries = if String.equal backend "hip" then 1024 else 32 in
  let heads = 8 and keys = 1024 in
  Stdio.eprintf "hip_half_load backend=%s queries=%d heads=%d keys=%d\n%!" backend queries heads
    keys;
  let make = L.node_factory ~prec:Ops.half ~first_id:1182000 ~dims:[| queries; heads; keys |] () in
  let scores = make "hhl_scores" and output = make "hhl_output" in
  let maxima = make ~dims:[| queries; heads; 1 |] "hhl_maxima" in
  let denominator = make ~dims:[| queries; heads; 1 |] "hhl_denominator" in
  let mask =
    L.node_factory ~prec:Ops.single ~first_id:1182100 ~dims:[| queries; keys |] () "hhl_mask"
  in
  let nodes = [ scores; output; maxima; denominator; mask ] in
  List.iter nodes ~f:L.materialize;
  let q = L.sym () and h = L.sym () and chunk = L.sym () and lane = L.sym () in
  let qi = L.iter q and hi = L.iter h and col = L.aff [ (256, chunk); (1, lane) ] 0 in
  let cell = [| qi; hi; col |] and row = [| qi; hi; L.fixed 0 |] in
  let half_bin op a b = LL.Binop (op, (a, Ops.half), (b, Ops.half)) in
  let score = half_bin Ops.Div (L.get scores cell) (L.c 5.656854249492381) in
  let masked =
    LL.Ternop
      ( Ops.Where,
        (L.get mask [| qi; col |], Ops.single),
        (score, Ops.half),
        (L.c Float.neg_infinity, Ops.half) )
  in
  let delta = half_bin Ops.Sub masked (L.get maxima row) in
  let numerator = LL.Unop (Ops.Exp, (delta, Ops.half)) in
  let llc =
    L.loop_n ~axis:LL.Grid q queries
      (L.loop_n ~axis:LL.Grid h heads
         (L.loop_n ~axis:LL.Grid chunk 4
            (L.loop_n ~axis:LL.Workgroup lane 256
               (L.set output cell (half_bin Ops.Div numerator (L.get denominator row))))))
  in
  let optimized = L.optimize ~materialized:nodes ~name:"hip_half_load" llc in
  let ctx, routine = L.link ~ctx ~name:"hip_half_load" optimized in
  let count = queries * heads * keys in
  let seed =
    [
      (scores, Array.create ~len:count 0.);
      (maxima, Array.create ~len:(queries * heads) 1.);
      (denominator, Array.create ~len:(queries * heads) 2.);
      (mask, Array.init (queries * keys) ~f:(fun i -> if i / keys >= i % keys then 1. else 0.));
      (output, Array.create ~len:count Float.nan);
    ]
  in
  let cases =
    List.init 3 ~f:(fun _ ->
        let ctx = L.run_linked (ctx, routine) ~seed in
        Context.get_values ctx output)
  in
  p_all "half-load replay writes every cell with a finite value" cases ~f:(fun values ->
      Array.for_all values ~f:Float.is_finite);
  p_all "half-load replay leaves every masked lane exactly zero" cases ~f:(fun values ->
      Array.for_alli values ~f:(fun i v -> i / (heads * keys) >= i % keys || Float.equal v 0.));
  p_all "half-load replay preserves max subtraction within half resolution" cases ~f:(fun values ->
      Array.for_alli values ~f:(fun i v ->
          i / (heads * keys) < i % keys || Float.(abs (v - (exp (-1.) / 2.)) < 0.001)))

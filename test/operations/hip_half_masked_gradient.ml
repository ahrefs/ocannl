(* gfx1102's masked backward branch can update a VGPR's low half while a D16 high-half load remains
   pending. A vector-load-only wait guard missed this path; the scalar mask wait also needs to drain
   vector loads. Keep all inputs finite, denominator squares small, and every reference partial sum
   far below half overflow. Vary the incoming gradients and operands with every index; six launches
   sample the asynchronous failure. This is a compiler hazard test, not an optimizer substitution
   test or a change to the GPT parity envelope. *)
open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module L = Ll_test
module LL = Ir.Low_level
module Ops = Ir.Ops

let () =
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let batches = 8 and heads = 8 and keys = 128 in
  let queries = if String.equal backend "hip" then 128 else 32 in
  Stdio.eprintf "hip_half_masked_gradient backend=%s shape=%dx%dx%dx%d\n%!" backend batches queries
    heads keys;
  let dims = [| batches; queries; heads; keys |] and row_dims = [| batches; queries; heads |] in
  let make = L.node_factory ~prec:Ops.half ~first_id:1182200 ~dims () in
  let scores = make "hmg_scores" and incoming = make "hmg_incoming" in
  let maxima = make ~dims:row_dims "hmg_maxima" in
  let denominator = make ~dims:row_dims "hmg_denominator" in
  let squared = make ~dims:row_dims "hmg_squared" and output = make ~dims:row_dims "hmg_output" in
  Ir.Tnode.update_memory_mode squared Ir.Tnode.Local (Site "1182:test-half-gradient-replay");
  let mask =
    L.node_factory ~prec:Ops.single ~first_id:1182300 ~dims:[| queries; keys |] () "hmg_mask"
  in
  let nodes = [ scores; incoming; maxima; denominator; mask; output ] in
  List.iter nodes ~f:L.materialize;
  let b = L.sym () and q = L.sym () and h = L.sym () and k = L.sym () in
  let row = [| L.iter b; L.iter q; L.iter h |] in
  let cell = [| L.iter b; L.iter q; L.iter h; L.iter k |] in
  let half_bin op a c = LL.Binop (op, (a, Ops.half), (c, Ops.half)) in
  let masked =
    LL.Ternop
      ( Ops.Where,
        (L.get mask [| L.iter q; L.iter k |], Ops.single),
        (half_bin Ops.Div (L.get scores cell) (L.c 5.656854249492381), Ops.half),
        (L.c Float.neg_infinity, Ops.half) )
  in
  let numerator = LL.Unop (Ops.Exp, (half_bin Ops.Sub masked (L.get maxima row), Ops.half)) in
  let derivative = half_bin Ops.Div (half_bin Ops.Mul (L.c (-1.)) numerator) (L.get squared row) in
  let update =
    L.set output row
      (half_bin Ops.Add (L.get output row) (half_bin Ops.Mul (L.get incoming cell) derivative))
  in
  let body =
    L.seq
      (L.set squared row (half_bin Ops.Mul (L.get denominator row) (L.get denominator row)))
      (L.seq (L.set output row (L.c 0.)) (L.loop_n k keys update))
  in
  let llc =
    L.loop_n ~axis:LL.Grid b batches
      (L.loop_n ~axis:LL.Grid q queries
         (L.loop_n ~axis:LL.Workgroup h 256 (L.if_ (L.lt (L.embed h) (L.c 8.)) body)))
  in
  let optimized = L.optimize ~materialized:nodes ~name:"hip_half_masked_gradient" llc in
  let ctx, routine = L.link ~ctx ~name:"hip_half_masked_gradient" optimized in
  let row_count = batches * queries * heads and count = batches * queries * heads * keys in
  let row_indices r = (r / (queries * heads), r / heads % queries, r % heads) in
  let score b q h k = Float.of_int (((b + q + h + k) % 16) - 8) /. 128. in
  let maximum b q h = 1. +. (Float.of_int ((4 * b) + q + (2 * h)) /. 128.) in
  let denom b q h = 2. +. (Float.of_int (b + q + h) /. 128.) in
  let grad b q h k = 1. +. (Float.of_int (b + q + h + k) /. 1024.) in
  let per_cell f =
    Array.init count ~f:(fun i ->
        let b, q, h = row_indices (i / keys) in
        f b q h (i % keys))
  in
  let per_row f =
    Array.init row_count ~f:(fun r ->
        let b, q, h = row_indices r in
        f b q h)
  in
  let seed =
    [
      (scores, per_cell score);
      (incoming, per_cell grad);
      (maxima, per_row maximum);
      (denominator, per_row denom);
      (mask, Array.init (queries * keys) ~f:(fun i -> if i / keys >= i % keys then 1. else 0.));
      (output, Array.create ~len:row_count Float.nan);
    ]
  in
  let reference =
    Array.init row_count ~f:(fun r ->
        let b, q, h = row_indices r in
        let d = denom b q h in
        List.sum
          (module Float)
          (List.init (q + 1) ~f:Fn.id)
          ~f:(fun k ->
            -.grad b q h k
            *. Float.exp ((score b q h k /. 5.656854249492381) -. maximum b q h)
            /. (d *. d)))
  in
  (* A conservative half-rounding bound: gamma_(keys+8) covers the 128-term recurrence and the
     surrounding operations. The absolute term covers half subnormal rounding. This tests a bounded
     synthetic derivative; the model retains its separately enforced parity envelope. *)
  let n_u = Float.of_int (keys + 8) /. 2048. in
  let gamma = n_u /. (1. -. n_u) and floor = Float.of_int keys *. Stdlib.ldexp 1. (-25) in
  let cases =
    List.init 6 ~f:(fun _ ->
        let ctx = L.run_linked (ctx, routine) ~seed in
        Context.get_values ctx output)
  in
  p_all "masked half-gradient replay writes only finite rows" cases ~f:(fun values ->
      Array.for_all values ~f:Float.is_finite);
  p_all "masked half-gradient replay stays within its half-rounding bound" cases ~f:(fun values ->
      Array.for_alli values ~f:(fun r v ->
          Float.(abs (v - reference.(r)) <= (gamma * abs reference.(r)) + floor)))

(* gh-ocannl-1602: the seed-parity reference's tf32 rounding, executed.

   A seed that computes an f32 contraction on a tf32 arm (CUDA under [tf32_matmuls]) multiplies its
   operands rounded to tf32, so Bench_seed_parity compares it with the default pipeline over a
   lowering whose matching accumulations round their multiplicands the same way
   ([Bench_seed_parity.tf32_round], built from existing operations: there is no tf32 op and no
   bitcast). Two legs:

   - Emulation, on every backend: a contraction [out[i] = x[i,0] * 1 + x[i,1] * 0] compiled through
   [Bench_seed_parity.compile_rounded] with [out] as the target, over a sweep of f32 values -- ties
   at tf32's 11-bit significand in both signs, quarter and three-quarter ulps, carries into the
   exponent, wide exponents, pseudo-random significands. [out] must equal the bit model of CUDA's
   [cvt.rna.tf32.f32] (add half of the 13 dropped bits' weight, then truncate: ties away from zero)
   bitwise; the products are exact, so any difference is the rounding's. The same contraction
   compiled plainly must differ from the model somewhere (the inputs discriminate), and the sweep
   must hold ties where ties-to-even would round the other way.

   - Hardware, where the backend's arm reads f32 operands as tf32 ([Bench_seed_parity.reads_tf32],
   the predicate the parity mode itself uses): a 32x16x32 matmul against a one-hot matrix, each of
   its seeds executed by [Bench_seed_parity.parity] at tolerance 0. Every output cell is one operand
   times 1 plus exact zeros, so a tensorized seed's cell is the hardware's own conversion of the
   operand, and the rounded reference must reproduce it bitwise. Against the plain reference
   ([~round:false]) a tensorized seed must differ: the rounding is load-bearing. The default run on
   CUDA skips this leg (tf32 off: no tf32 arm); the [-tf32] rule runs it.

   Per-cell and per-seed detail goes to stderr; the golden holds the claims. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module IDX = Train.IDX
module P = Bench_seed_parity

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Ir.Schedule.backend_is_cpu backend_name
let nd debug dims f = Ir.Ndarray.init_array ~debug Ir.Ops.single ~dims ~padding:None ~f
let f32_of_bits b = Stdlib.Int32.float_of_bits (Stdlib.Int32.of_int b)

(* CUDA's [cvt.rna.tf32.f32] on a finite normal f32: ties away from zero. *)
let cvt_rna v = Stdlib.Int32.(float_of_bits (logand (add (bits_of_float v) 0x1000l) 0xFFFFE000l))

(* The same rounding with ties to even, for the claim that the sweep separates the two. *)
let ties_to_even v =
  let bits = Stdlib.Int32.bits_of_float v in
  let low = Stdlib.Int32.(to_int (logand bits 0x1FFFl)) in
  let kept_odd = Stdlib.Int32.(to_int (logand bits 0x2000l)) <> 0 in
  if low = 0x1000 && not kept_odd then Stdlib.Int32.(float_of_bits (logand bits 0xFFFFE000l))
  else cvt_rna v

(* f32 bit patterns: sign, unbiased exponent, 23-bit significand field. *)
let pattern ~neg ~exp ~mant =
  f32_of_bits ((if neg then 1 lsl 31 else 0) lor ((exp + 127) lsl 23) lor (mant land 0x7FFFFF))

let sweep =
  let lows = [ 0x0000; 0x0001; 0x07FF; 0x0800; 0x0FFF; 0x1000; 0x1001; 0x17FF; 0x1800; 0x1FFF ] in
  let highs = [ 0x000; 0x3FF; 0x155; 0x2AA; 0x001; 0x200 ] in
  let exps = [ -100; -60; -20; -3; -1; 0; 1; 7; 20; 60; 100 ] in
  let structured =
    List.concat_map exps ~f:(fun exp ->
        List.concat_map highs ~f:(fun high ->
            List.concat_map lows ~f:(fun low ->
                List.map [ false; true ] ~f:(fun neg ->
                    pattern ~neg ~exp ~mant:((high lsl 13) lor low)))))
  in
  (* A fixed linear congruential sequence of significands at modest exponents. *)
  let state = ref 12345 in
  let next () =
    state := ((!state * 1103515245) + 12345) land 0x7FFFFFFF;
    !state
  in
  let random =
    List.init 256 ~f:(fun i ->
        let r = next () in
        pattern ~neg:(i % 2 = 1) ~exp:(((r lsr 24) % 41) - 20) ~mant:r)
  in
  Array.of_list (structured @ random)

let claims_emulation =
  [
    "emulation: the rounded compile rewrites the accumulation into the target";
    "emulation: every value rounds bitwise as cvt.rna.tf32.f32 does";
    "emulation: the plain compile leaves some value unrounded (the inputs discriminate)";
    "emulation: the sweep holds ties where ties-to-even rounds the other way";
  ]

let claims_hardware =
  [
    "hardware: at least one seed executes tensorized with its reference rounded to tf32";
    "hardware: every executed seed matches its reference bitwise";
    "hardware: no seed fails to reach its segment, is unjudged, or raises";
    "hardware control: against the plain reference, some tensorized seed differs";
  ]

let emulation () =
  let n = Array.length sweep in
  let x_nd = nd "trr_x" [| 1; n; 2 |] (fun i -> if i.(2) = 0 then sweep.(i.(1)) else 0.) in
  let w_nd = nd "trr_w" [| 1; 2 |] (fun i -> if i.(1) = 0 then 1. else 0.) in
  let xs = TDSL.rebatch ~l:"trr_xs" x_nd () in
  let ws = TDSL.rebatch ~l:"trr_ws" w_nd () in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:1 IDX.empty in
  let%op x = xs @| batch_n in
  let%op w = ws @| batch_n in
  let%op out = x +* "ij; j => i" w in
  Train.every_non_literal_materialized out;
  let comp = Train.forward out in
  let values ~rounded =
    let ctx, routine, stmts =
      if rounded then
        P.compile_rounded ~targets:[ out.Tensor.value ] (Context.auto ()) comp bindings
      else
        let ctx, routine = Context.compile (Context.auto ()) comp bindings in
        (ctx, routine, 0)
    in
    IDX.find_exn routine.Context.bindings batch_n := 0;
    let ctx = Context.run ctx routine in
    Context.sync ctx;
    let got = Context.get_values ctx out.Tensor.value in
    Context.release ctx;
    (got, stmts)
  in
  let rounded, stmts = values ~rounded:true in
  let plain, _ = values ~rounded:false in
  let want = Array.map sweep ~f:cvt_rna in
  Stdio.eprintf
    "emulation: %d values, %d accumulation statements rounded (not part of the golden)\n" n stmts;
  let wrong =
    List.filter (List.init n ~f:Fn.id) ~f:(fun i -> not (Float.equal rounded.(i) want.(i)))
  in
  List.iteri wrong ~f:(fun k i ->
      if k < 10 then
        Stdio.eprintf "emulation: %h rounded to %h, cvt.rna gives %h (not part of the golden)\n"
          sweep.(i) rounded.(i) want.(i));
  p (List.nth_exn claims_emulation 0) (stmts > 0);
  p_all (List.nth_exn claims_emulation 1) (List.init n ~f:Fn.id) ~f:(fun i ->
      Float.equal rounded.(i) want.(i));
  p_exists (List.nth_exn claims_emulation 2) (List.init n ~f:Fn.id) ~f:(fun i ->
      not (Float.equal plain.(i) want.(i)));
  p_exists (List.nth_exn claims_emulation 3) (Array.to_list sweep) ~f:(fun v ->
      not (Float.equal (cvt_rna v) (ties_to_even v)))

(* Whether this backend's arm reads f32 operands as tf32, and if not, what would lift that: the
   seeder's tensorized capability for an f32 site, then the parity mode's own tf32 predicate. *)
let tf32_arm limits =
  match
    Ll_test.tensorized_capability ~is_gpu:(not on_cpu) ~is_cpu:on_cpu ~limits ~a_prec:Ir.Ops.single
      ~b_prec:Ir.Ops.single ~d_prec:Ir.Ops.single
  with
  | `Withheld aggregation -> `Skip aggregation
  | `Advertised when P.f32_reads_tf32 limits -> `Reads
  | `Advertised ->
      `Skip (if P.f32_reads_tf32_under_tf32_policy limits then `Environment else `Backend)

let hardware () =
  let m = 32 and k = 16 and n = 32 in
  (* Operands of every magnitude multiply 1 exactly; keep them clear of overflow in f32 arithmetic
     around the tile anyway. A stride of 3 through the sweep (512 cells of its 1576) varies them
     along both axes. *)
  let a_vals =
    Array.init (m * k) ~f:(fun f ->
        let v = sweep.(f * 3) in
        if Float.(abs v > 1e30) then v *. 1e-60 else v)
  in
  let a_nd =
    nd "trr_a" [| 1; m; k |] (fun i -> a_vals.(Ll_test.flat ~dims:[| m; k |] [| i.(1); i.(2) |]))
  in
  let e_nd = nd "trr_e" [| 1; k; n |] (fun i -> if i.(1) = i.(2) % k then 1. else 0.) in
  let as_ = TDSL.rebatch ~l:"trr_as" a_nd () in
  let es = TDSL.rebatch ~l:"trr_es" e_nd () in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:1 IDX.empty in
  let%op a = as_ @| batch_n in
  let%op e = es @| batch_n in
  let%op out = a +* "ik; kj => ij" e in
  Train.every_non_literal_materialized out;
  let comp = Train.forward out in
  let ctx = Context.auto () in
  let static_indices = [ batch_n ] in
  let opt = Bench_harness.capture_lowering ctx comp bindings in
  let parity ~round =
    snd
      (P.parity ~tol:0. ~round ~select:P.any_seed
         ~on_result:(fun r ->
           Stdio.eprintf "%s%s (not part of the golden)\n"
             (if round then "" else "plain reference: ")
             (P.verdict_line r))
         ctx comp bindings
         ~bind:(fun r -> IDX.find_exn r.Context.bindings batch_n := 0)
         ~static_indices opt)
  in
  let executed = function { P.verdict = Match _ | Mismatch _; _ } -> true | _ -> false in
  let results = parity ~round:true in
  (* Tensorized by the compiled routine's own census, every statement on the tensor-core path: a
     statement that fell back to the scalar kernel computes f32, not tf32. *)
  let all_tensor_core r =
    Ir.C_syntax.equal_tensorization r.P.mma.Ir.C_syntax.tensorization Ir.C_syntax.Tensorized
    && r.P.mma.Ir.C_syntax.scalar_fallbacks = 0
  in
  p_exists (List.nth_exn claims_hardware 0) results ~f:(fun r ->
      executed r && all_tensor_core r && not (List.is_empty r.P.rounded));
  p_all (List.nth_exn claims_hardware 1) (List.filter results ~f:executed) ~f:(function
    | { P.verdict = Match { err; _ }; _ } -> Float.(err = 0.)
    | _ -> false);
  p_none (List.nth_exn claims_hardware 2) results ~f:(function
    | { P.verdict = Not_applied | Unjudged _ | Failed _ | Not_perturbed; _ } -> true
    | _ -> false);
  let plain = parity ~round:false in
  p_exists (List.nth_exn claims_hardware 3) plain ~f:(fun r ->
      (match r.P.verdict with Mismatch _ -> true | _ -> false) && all_tensor_core r)

let () =
  emulation ();
  match tf32_arm (Context.hardware_limits (Context.auto ())) with
  | `Reads -> hardware ()
  | `Skip aggregation ->
      List.iter claims_hardware ~f:(Verdict.skipped ~aggregation ~backend:backend_name)

(* Metal's [sin], [cos] and [tanh] are IEEE on special inputs and accurate near zero and at large
   magnitudes, at f32, half and bfloat16 (gh-ocannl-1435), and so are [exp], [log], [exp2] and
   [log2] up to a recorded subnormal flush (gh-ocannl-1515).

   Metal compiles with Safe arithmetic and fast math FUNCTIONS ([Compiler_options.metal]). Measured
   on M4 Max ([benchmarks/runners/ocannl/metal_f32_math_probe.ml]), the fast f32 [sin] and [cos]
   return 0 for inf, NaN, and every |x| from 2^21 pi (0x1.921fb8p+22) up. The fast f32 [tanh]
   returns 0 for x from about 43.7 and NaN from about 44.4 up (+inf included), +0 for -0, flushes
   |x| < 2^-26 to 0, and is off by up to 2050 ulps around 2^-12 and 26 ulps around 2^-5. Half and
   bfloat16 [sin]/[cos] are the f32 builtin narrowed (gh-ocannl-1288), so they inherited the
   special-value zeros. The backend now renders [sin]/[cos] through [ocannl_sin]/[ocannl_cos]
   ([Builtins_metal]: the fast function inside its domain, [precise::] outside it and for non-finite
   inputs) and f32/bfloat16 [tanh] as [precise::tanh]. The inputs below include every case above, so
   each claim fails under the plain fast spellings (checked on M4 Max against the previous
   rendering).

   The claims compare against the f64 function of the exact stored input: NaN where it is NaN, the
   signed zero bitwise where it is zero, and otherwise within a bound in ulps of the result
   precision -- 4 f32 ulps for [sin]/[cos] (3.4 measured, as for [precise::]), 3 for [tanh] (2
   measured), one ulp of half or bfloat16 (one rounding of an f32 result; the half [tanh] overload
   measured 0.502).

   [exp], [log], [exp2] and [log2] (gh-ocannl-1515) render bare: the same probe finds them IEEE on
   the signed zeros, infinities, NaN and negative inputs to [log]/[log2], under both policies, and
   [exp2]/[log2] bitwise their [precise::] forms. Two departures are recorded rather than fixed. The
   fast f32 [exp] is up to 59.5 ulps off at large |x| (the [precise::] one 1.4; CUDA's [__expf]
   promises [2 + 1.173 |x|]), hence its bound [2 + |x|]. And all four flush f32 subnormals -- the
   inputs of [log]/[log2] ([log(+-2^-140)] is -inf), the results of [exp]/[exp2] -- under
   [precise::] and the legacy policy too, so no spelling of the function avoids it, and CUDA's
   [--use_fast_math] implies [--ftz=true] alike; such a result passes against f64 at the flushed
   input or the flushed f32 result. Fixed at 4 ulps for [exp] and without the flush allowance, the
   f32 claim fails on M4 Max at exp(31.86), exp(50), exp(82.4), exp(-95), exp2(-140) and the four
   subnormal logs.

   They are Metal-only: CUDA and HIP render their own fast intrinsics (CUDA's [tanh.approx.f32] is a
   2^-11 relative approximation by contract; gh-ocannl-1336 audits their domains), and every backend
   still compiles and runs the kernels. *)
open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Stdio.eprintf "unary-math-special-values backend: %s\n" backend_name
let is_metal = String.equal backend_name "metal"
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let f16 x = Ops.half_to_single (Ops.single_to_half (f32 x))
let bf16 x = Ops.bfloat16_to_single (Ops.single_to_bfloat16 (f32 x))

(* Each op with its f64 reference, and whether Metal's f32 function flushes subnormals: the inputs
   of [log]/[log2], the results of [exp]/[exp2] (gh-ocannl-1515). *)
let ops : (Ops.unop * string * (float -> float) * bool) list =
  [
    (Sin, "sin", Float.sin, false);
    (Cos, "cos", Float.cos, false);
    (Tanh_approx, "tanh", Float.tanh, false);
    (Exp, "exp", Float.exp, true);
    (Log, "log", Float.log, true);
    (Exp2, "exp2", Stdlib.Float.exp2, true);
    (Log2, "log2", Stdlib.Float.log2, true);
  ]

let specials = [ -0.; 0.; Float.infinity; Float.neg_infinity; Float.nan ]

(* Per precision: its rounding, the spacing of its values at [v], the bound in those ulps for each
   op, and inputs exactly representable in it. *)
type case = {
  label : string;
  prec : Ops.prec;
  round : float -> float;
  ulp : float -> float;
  bound : string -> float -> float;
  inputs : float list;
}

(* The spacing at [|v|] of a binary format with [mant] stored fraction bits and smallest normal
   exponent [emin]: below 2^emin, the subnormal step. *)
let spacing ~mant ~emin v =
  let e = snd (Float.frexp (Float.abs v)) - 1 in
  Float.ldexp 1.0 (Int.max e emin - mant)

let cases =
  [
    {
      label = "f32";
      prec = Ops.single;
      round = f32;
      ulp = spacing ~mant:23 ~emin:(-126);
      (* The fast f32 [exp] is [exp2(x * log2(e))]-shaped: its error grows with |x| (59.5 ulps
         measured at 82.4), as CUDA's [__expf] contract's [2 + 1.173 |x|] does. *)
      bound =
        (fun name x ->
          match name with
          | "tanh" | "log2" -> 3.
          | "exp" -> 2. +. Float.abs x
          | "exp2" -> 2.
          | _ -> 4.);
      inputs =
        specials
        @ [
            0x1p-140;
            -0x1p-140;
            0x1p-126;
            0x1p-26;
            0x1p-20;
            -0x1p-12;
            0x1.ebp-6;
            0.5;
            -3.25;
            20.;
            43.75;
            50.;
            100.;
            0x1p22;
            0x1.921fb6p+22;
            0x1.921fb8p+22;
            0x1.e5p+23;
            0x1p30;
            0x1.2c4p+90;
            -0x1p100;
            0x1.fffffep+127;
            1.;
            -1.;
            2.;
            0x1.fdcp+4;
            0x1.49cp+6;
            88.75;
            -95.;
            128.;
            -140.;
            -150.;
          ];
    };
    {
      label = "half";
      prec = Ops.half;
      round = f16;
      ulp = spacing ~mant:10 ~emin:(-14);
      bound = (fun _ _ -> 1.);
      inputs =
        specials
        @ [
            1.;
            -1.;
            2.;
            11.;
            12.;
            -12.;
            -17.5;
            -24.;
            16.;
            0x1p-24;
            -0x1p-24;
            0x1.ff8p-15;
            0x1p-14;
            0x1p-10;
            -0x1.48p-7;
            0x1.ebp-6;
            0.5;
            -3.25;
            20.;
            43.75;
            50.;
            1000.;
            30000.;
            65504.;
          ];
    };
    {
      label = "bfloat16";
      prec = Ops.bfloat16;
      round = bf16;
      ulp = spacing ~mant:7 ~emin:(-126);
      bound = (fun _ _ -> 1.);
      inputs =
        (* No bfloat16 subnormal: Metal's float-to-bfloat narrowing, like its f32 arithmetic,
           flushes them to the signed zero, a property of the GPU's float ALU rather than of these
           functions (storage and widening keep them; [bf16_subnormal_storage]). *)
        specials
        @ [
            0x1p-126;
            0x1p-26;
            0x1p-12;
            0x1.ecp-6;
            0.5;
            43.75;
            50.;
            0x1.92p+22;
            0x1p23;
            0x1.2cp+90;
            -0x1p100;
            1.;
            -1.;
            2.;
            88.75;
            128.;
          ];
    };
  ]

(* One kernel at [c.prec] computing every op over every input: [out.(k * n + i) = op_k
   inputs.(i)]. *)
let run c ~first_id =
  let inputs = List.map c.inputs ~f:c.round in
  let n = List.length inputs and m = List.length ops in
  let name = "umsv_" ^ c.label in
  let node = Ll_test.node_factory ~prec:c.prec ~first_id ~dims:[| n |] () in
  let input = node (name ^ "_in") and out = node ~dims:[| n * m |] (name ^ "_out") in
  List.iter [ input; out ] ~f:Ll_test.materialize;
  let body =
    List.foldi ops ~init:LL.Noop ~f:(fun k acc (op, _, _, _) ->
        List.foldi inputs ~init:acc ~f:(fun i acc _ ->
            Ll_test.seq acc
              (Ll_test.set out
                 [| Ll_test.fixed ((k * n) + i) |]
                 (LL.Unop (op, (Ll_test.get input [| Ll_test.fixed i |], c.prec))))))
  in
  let o = Ll_test.optimize ~name body in
  ( inputs,
    List.hd_exn
      (Ll_test.execute ~name o
         ~seed:[ (input, Array.of_list inputs); (out, Ll_test.blank (n * m)) ]
         ~read:[ out ]) )

(* An f32 subnormal flushed to the zero of its sign. *)
let flush v = if Float.(v <> 0. && abs v < 0x1p-126) then Float.copysign 0. v else v

let () =
  Verdict.case "unary math special values" (fun () ->
      List.iteri cases ~f:(fun ci c ->
          let inputs, got = run c ~first_id:(19900 + (100 * ci)) in
          let n = List.length inputs in
          let flushed = ref 0 in
          gated_alli ~when_:is_metal ~on:backend_name
            (Printf.sprintf
               "Metal %s sin, cos, tanh, exp, log, exp2 and log2: NaN where f64 gives NaN, the \
                signed zero or infinity where f64 rounds to one, otherwise within the stated ulps \
                of f64 -- for exp, log, exp2 and log2, of f64 at the input or the f32 result \
                flushed from subnormal to zero"
               c.label) (Array.to_list got) ~f:(fun j v ->
              let _, name, reference, ftz = List.nth_exn ops (j / n) in
              let x = List.nth_exn inputs (j % n) in
              (* Whether [v] is [want] at [c]'s precision, and its error in ulps when finite. *)
              let check want =
                let rounded = c.round want in
                if Float.is_nan want then (Float.is_nan v, Float.nan)
                else if Float.(rounded = 0.) || not (Float.is_finite rounded) then
                  (Int64.equal (Int64.bits_of_float v) (Int64.bits_of_float rounded), Float.nan)
                else
                  let err = Float.abs (v -. want) /. c.ulp want in
                  (Float.(err <= c.bound name x), err)
              in
              let want = reference x in
              let ok, err = check want in
              let ok =
                ok
                || ftz
                   && (fst (check (reference (flush x))) || fst (check (flush (f32 want))))
                   &&
                   (Int.incr flushed;
                    true)
              in
              if not ok then
                Stdio.eprintf "(not part of the golden) %s %s(%h) = %h, f64 %h, %g ulps\n" c.label
                  name x v want err;
              ok);
          if is_metal then
            Stdio.eprintf
              "(not part of the golden) %s: %d results matched only with a subnormal flushed\n"
              c.label !flushed))

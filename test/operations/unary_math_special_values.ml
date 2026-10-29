(* The f32, half and bfloat16 [sin], [cos], [tanh], [exp], [log], [exp2], [log2], [sqrt] and [rsqrt]
   of every backend, over the domains where fast math functions go wrong (gh-ocannl-1336,
   gh-ocannl-1684): large-argument [sin]/[cos], subnormal and near-zero [log], [exp]/[exp2] next to
   overflow and underflow, [log] next to 1, and the special values. Every result is compared with
   the f64 function of the exact stored input: NaN where it is NaN, the signed zero or infinity
   bitwise where it rounds to one, and otherwise within a bound -- the "stated ulps" of the result
   precision (4 f32 ulps for [sin]/[cos]/[log], 3 for [tanh]/[log2], 2 for [exp2]/[rsqrt], 1 for
   [sqrt], [2 + |x|] for [exp], one ulp of half or bfloat16), or a backend's own documented
   contract. The binary [pow] and [fmod] (gh-ocannl-1684) run over a grid of special and dyadic
   operands whose results are exact, against C's [pow] and [fmod] (2 f32 ulps for [pow], [fmod]
   exactly), and [fmod] of constant operands once more, folded on the host before any backend sees
   it. Before gh-ocannl-1684 that fold raised on a negative modulus and gave a nonnegative remainder
   ([Base]'s float [%]), and cc's float [Mod] did not compile (C's [%] takes integers); half [fmod]
   did not compile on HIP either (two [__half] operands make the call ambiguous).

   cc (libm) is the reference: the stated ulps hold with subnormals kept. So does HIP, whose
   [-ffast-math] keeps ROCm's precise library functions and AMDGPU's f32 denormals, measured on
   gfx1102 (discrete memory), with one departure, fixed: ROCm's f32 [sinf]/[cosf] returned 0.974869
   and 0.222781 for +-inf (bfloat16, widened into them, likewise), hiding an overflow upstream, so
   f32 and bfloat16 [sin]/[cos] render through [Builtins_hip]'s NaN-preserving [ocannl_sinf]/
   [ocannl_cosf]; the special-value claim fails without them. Two more departures, fixed the same
   way (gh-ocannl-1684, measured on gfx1102 discrete; the fixed rendering passes on gfx1151 unified
   too): the f32 [sqrtf] -- and [1.0f / sqrtf], hence [rsqrt] -- read a subnormal input as the zero
   of its sign ([sqrt(2^-140)] is +0, [rsqrt] +inf), and [fmodf] returned the signed zero of [x] for
   [fmod(+-0, +-0)] and [fmod(+-inf, +-inf)], where C gives NaN. f32 and bfloat16 [sqrt]/[rsqrt] and
   f32, half and bfloat16 [fmod] render through [Builtins_hip]'s [ocannl_sqrtf] (a subnormal scaled
   into the normal range and back) and [ocannl_fmodf] (C's NaN domain selected as bits); without
   them the f32 stated-ulps claim and the [pow]/[fmod] claim of every precision fail. HIP's [pow] --
   the [ocannl_pow_f32] helper over ROCm's [powf] -- was already C's on the whole grid.

   On CUDA (RTX 5070 Ti), [sqrt], [rsqrt] and [fmod] are C's on the special values, f32 subnormals
   flushed as the CUDA claim allows. Its [pow] was not: [__powf] reads [pow(+-1, +-inf)] and [pow(1,
   NaN)] as NaN ([inf * log2(1)]) and [pow(-inf, 0.5)] as NaN ([log2] of a negative base), where C
   gives 1, 1 and +inf. The shared helper ([Builtins_cc.cuda_like_power_builtins]) now answers those
   cases itself, at every precision (half and bfloat16 widen into it); the host emulation
   [runtime_power_fast_math_emulation] pins them GPU-free. The f32 magnitude stays [__powf]'s,
   claimed against its contract ([cuda_powf]: 3 and 5 ulps measured at [pow(0.25, 3)] and [pow(0.25,
   -3)]).

   CUDA's [--use_fast_math] lowers f32 (and widened bfloat16) [sinf]/[cosf]/[expf]/[logf]/[exp2f]/
   [log2f] to its fast intrinsics, and [Tanh_approx] renders [__tanhf]; they keep the special values
   IEEE, so the special-value claim holds there. The stated ulps do not, so CUDA's f32 and bfloat16
   are claimed against the intrinsics' documented contracts instead, and the claim of the stated
   ulps is skipped there, each departure an accepted vendor behaviour (measured on RTX 5070 Ti): -
   [__sinf]/[__cosf] have an absolute-error contract, 2^-21.41 on [-pi, pi]: [sin(2^-26)] is 0 and
   [sin(2^-20)] 2% low. Past pi the error grows with |x|, within [2^-23 |x|] measured (0.7 ulps of
   the INPUT at 10000, the rounding a RoPE angle already carries). From 2^22, where the input's own
   spacing is 0.5, the argument reduction is lost ([sin(2^30)] is 0, [cos] 1): any value in [-1, 1]
   is accepted there. No training angle reaches it. - [__logf]/[__log2f] have an absolute-error
   contract on [0.5, 2], 2^-21.41 and 2^-22: next to 1 the result is up to 86% off relatively
   (2^-23.7 absolute) though of the right sign, which the dense band below pins for every backend,
   since [Nn_blocks.box_muller] takes [sqrt (-2 log u)] for [u] just below 1. - [__tanhf] is a
   2^-10.987 relative approximation; [__expf] is [2 + 1.173 |x|] ulps (exact at the overflow edge
   [0x1.62e42ep+6], measured). - [--use_fast_math] implies [--ftz=true]: f32 subnormal inputs and
   results flush, as on Metal. CUDA's half functions ([hsin], [hexp], ...) meet the stated ulps with
   subnormals kept.

   Metal compiles with Safe arithmetic and fast math FUNCTIONS ([Compiler_options.metal]). Measured
   on M4 Max ([benchmarks/runners/ocannl/metal_f32_math_probe.ml]), the fast f32 [sin] and [cos]
   return 0 for inf, NaN, and every |x| from 2^21 pi (0x1.921fb8p+22) up. The fast f32 [tanh]
   returns 0 for x from about 43.7 and NaN from about 44.4 up (+inf included), +0 for -0, flushes
   |x| < 2^-26 to 0, and is off by up to 2050 ulps around 2^-12 and 26 ulps around 2^-5. Half and
   bfloat16 [sin]/[cos] are the f32 builtin narrowed (gh-ocannl-1288), so they inherited the
   special-value zeros. The backend renders [sin]/[cos] through [ocannl_sin]/[ocannl_cos]
   ([Builtins_metal]: the fast function inside its domain, [precise::] outside it and for non-finite
   inputs) and f32/bfloat16 [tanh] as [precise::tanh] (gh-ocannl-1435). Each Metal claim fails under
   the plain fast spellings (checked on M4 Max against the previous rendering).

   Metal's [exp], [log], [exp2] and [log2] (gh-ocannl-1515) render bare: IEEE on the signed zeros,
   infinities, NaN and negative inputs to [log]/[log2], under both policies, and [exp2]/[log2]
   bitwise their [precise::] forms. Two departures are recorded rather than fixed. The fast f32
   [exp] is up to 59.5 ulps off at large |x| (the [precise::] one 1.4), hence its bound [2 + |x|].
   And all four flush f32 subnormals -- the inputs of [log]/[log2] ([log(+-2^-140)] is -inf), the
   results of [exp]/[exp2] -- under [precise::] and the legacy policy too, so no spelling of the
   function avoids it; such a result passes against f64 at the flushed input or the flushed f32
   result. Fixed at 4 ulps for [exp] and without the flush allowance, the Metal f32 claim fails on
   M4 Max at exp(31.86), exp(50), exp(82.4), exp(-95), exp2(-140) and the four subnormal logs.
   [sqrt] and [rsqrt] flush f32 subnormal inputs the same way (gh-ocannl-1684), and pass on the same
   allowance. Metal's fast [fmod] gave NaN for a finite [x] over an infinite [y] (C: [x]) and [+0]
   for an exact remainder of a negative [x] (C: [-0]), at every precision, so [Mod] renders as
   [precise::fmod], half widened; Metal's own [pow] was C's on the whole grid.

   A claim that does not apply to a backend is skipped there ([SKIPPED on <backend>] on stderr), and
   its label names where it is skipped and why. Every backend compiles and runs every kernel. *)
open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Stdio.eprintf "unary-math-special-values backend: %s\n" backend_name
let is_metal = String.equal backend_name "metal"
let is_cuda = String.equal backend_name "cuda"
let is_hip = String.equal backend_name "hip"
let is_cc = List.mem [ "cc"; "multidev_cc" ] backend_name ~equal:String.equal
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let f16 x = Ops.half_to_single (Ops.single_to_half (f32 x))
let bf16 x = Ops.bfloat16_to_single (Ops.single_to_bfloat16 (f32 x))

(* Each op with its f64 reference, and whether Metal's f32 function flushes subnormals: the inputs
   of [log]/[log2]/[sqrt]/[rsqrt], the results of [exp]/[exp2] (gh-ocannl-1515, gh-ocannl-1684). *)
let ops : (Ops.unop * string * (float -> float) * bool) list =
  [
    (Sin, "sin", Float.sin, false);
    (Cos, "cos", Float.cos, false);
    (Tanh_approx, "tanh", Float.tanh, false);
    (Exp, "exp", Float.exp, true);
    (Log, "log", Float.log, true);
    (Exp2, "exp2", Stdlib.Float.exp2, true);
    (Log2, "log2", Stdlib.Float.log2, true);
    (Sqrt, "sqrt", Float.sqrt, true);
    (Recip_sqrt, "rsqrt", (fun x -> 1. /. Float.sqrt x), true);
  ]

(* "a, b and c" over the ops' names, for the claim labels. *)
let names_phrase names =
  match List.rev names with
  | [] | [ _ ] -> String.concat names
  | last :: rest -> String.concat ~sep:", " (List.rev rest) ^ " and " ^ last

let op_names = names_phrase (List.map ops ~f:(fun (_, name, _, _) -> name))

let ftz_names =
  names_phrase (List.filter_map ops ~f:(fun (_, name, _, ftz) -> Option.some_if ftz name))

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
  cuda_intrinsics : bool;
      (** Whether CUDA renders the ops at this precision through its f32 fast intrinsics: f32, and
          bfloat16 widened into f32 ([Cuda_like_config]'s [bf16_func]); half has its own [h*]
          functions. *)
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
      cuda_intrinsics = true;
      round = f32;
      ulp = spacing ~mant:23 ~emin:(-126);
      (* The fast f32 [exp] is [exp2(x * log2(e))]-shaped: its error grows with |x| (59.5 ulps
         measured at 82.4), as CUDA's [__expf] contract's [2 + 1.173 |x|] does. *)
      bound =
        (fun name x ->
          match name with
          | "tanh" | "log2" -> 3.
          | "exp" -> 2. +. Float.abs x
          | "exp2" | "rsqrt" | "pow" -> 2.
          | "sqrt" -> 1.
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
            0x1.62e42ep+6;
            0x1.62e430p+6;
            -0x1.5d58ap+6;
            -0x1.9fe368p+6;
            0x1.fffffep+6;
            -149.;
            0x1p-149;
            -0x1p-149;
            0x1.000002p+0;
            0x1.fffffep-1;
            0x1.388p+13;
            0x1.921fb6p+1;
            -0x1.921fb6p+5;
            0x1p-100;
          ];
    };
    {
      label = "half";
      prec = Ops.half;
      cuda_intrinsics = false;
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
            0x1.62p+3;
            0x1.63p+3;
            0x1.004p+0;
            0x1.ffcp-1;
            -0x1.1p+4;
          ];
    };
    {
      label = "bfloat16";
      prec = Ops.bfloat16;
      cuda_intrinsics = true;
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

(* The f32 inputs within 2^-13 of 1, at the spacing of f32 on each side, and 1 itself: the arguments
   of [Nn_blocks.box_muller]'s [log], whose largest is [1 - 2^-24]. *)
let band =
  List.init 2048 ~f:(fun k -> 1. -. Float.ldexp (Float.of_int (k + 1)) (-24))
  @ [ 1. ]
  @ List.init 1024 ~f:(fun k -> 1. +. Float.ldexp (Float.of_int (k + 1)) (-23))

(* One f32 loop over [band]: [log] and [log2] of each input. *)
let run_band () =
  let n = List.length band in
  let name = "umsv_band" in
  let node = Ll_test.node_factory ~prec:Ops.single ~first_id:20400 ~dims:[| n |] () in
  let input = node (name ^ "_in") in
  let outs = List.map [ "log"; "log2" ] ~f:(fun op -> node (name ^ "_" ^ op)) in
  List.iter (input :: outs) ~f:Ll_test.materialize;
  let i = Ll_test.sym () in
  let at = [| Ll_test.iter i |] in
  let body =
    List.fold2_exn outs [ Ops.Log; Ops.Log2 ] ~init:LL.Noop ~f:(fun acc out op ->
        Ll_test.seq acc (Ll_test.set out at (LL.Unop (op, (Ll_test.get input at, Ops.single)))))
  in
  let o = Ll_test.optimize ~name (Ll_test.loop_n i n body) in
  Ll_test.execute ~name o
    ~seed:((input, Array.of_list band) :: List.map outs ~f:(fun out -> (out, Ll_test.blank n)))
    ~read:outs

(* The binary ops, each with its f64 reference -- C's [pow] and [fmod] -- and its operand grid:
   every pair of a left and a right operand. The finite operands are dyadic, chosen so every result
   is exact in all three precisions; the grids reach every special case C99 Annex F gives [pow] and
   [fmod], [pow(1, NaN) = pow(-1, +-inf) = 1] and [fmod(x, +-inf) = x] among them. A power renders
   through [ocannl_pow_f32]/[ocannl_pow_f64] on CUDA and HIP (the exponent is known only at run
   time), half and bfloat16 widened into the f32 one. *)
let binops : (Ops.binop * string * (float -> float -> float) * float list * float list) list =
  [
    ( ToPowOf,
      "pow",
      Float.( ** ),
      specials @ [ 1.; -1.; 0.25; -0.25; 4.; -4. ],
      specials @ [ 1.; -1.; 2.; -2.; 3.; -3.; 0.5; -0.5; 1.5 ] );
    ( Mod,
      "fmod",
      Float.mod_float,
      specials @ [ 5.5; -5.5; 4.; -4.; 0.25 ],
      specials @ [ 2.; -2.; 0.75; -0.75 ] );
  ]

let binop_names = names_phrase (List.map binops ~f:(fun (_, name, _, _, _) -> name))

(* Every binary op at [c.prec] over its grid, in one kernel: [out.(j) = op lhs.(j) rhs.(j)], the
   pairs of all the ops concatenated. Returns each result with its op, operands and reference. *)
let run_binary c ~first_id =
  let pairs =
    List.concat_map binops ~f:(fun (op, name, reference, xs, ys) ->
        List.concat_map xs ~f:(fun x ->
            List.map ys ~f:(fun y -> (op, name, reference, c.round x, c.round y))))
  in
  let n = List.length pairs in
  let name = "umsv_bin_" ^ c.label in
  let node = Ll_test.node_factory ~prec:c.prec ~first_id ~dims:[| n |] () in
  let lhs = node (name ^ "_lhs") and rhs = node (name ^ "_rhs") and out = node (name ^ "_out") in
  List.iter [ lhs; rhs; out ] ~f:Ll_test.materialize;
  let body =
    List.foldi pairs ~init:LL.Noop ~f:(fun j acc (op, _, _, _, _) ->
        let at = [| Ll_test.fixed j |] in
        Ll_test.seq acc
          (Ll_test.set out at
             (LL.Binop (op, (Ll_test.get lhs at, c.prec), (Ll_test.get rhs at, c.prec)))))
  in
  let o = Ll_test.optimize ~name body in
  let got =
    List.hd_exn
      (Ll_test.execute ~name o
         ~seed:
           [
             (lhs, Array.of_list (List.map pairs ~f:(fun (_, _, _, x, _) -> x)));
             (rhs, Array.of_list (List.map pairs ~f:(fun (_, _, _, _, y) -> y)));
             (out, Ll_test.blank n);
           ]
         ~read:[ out ])
  in
  List.map2_exn pairs (Array.to_list got) ~f:(fun (_, name, reference, x, y) v ->
      (name, reference, x, y, v))

(* [fmod] of constant operands, which the optimizer folds on the host ([Ops.interpret_binop]) before
   any backend sees it: the finite pairs of the [fmod] grid with a nonzero modulus and a nonzero
   result, negative operands among them. *)
let run_folded_fmod c ~first_id =
  let _, name, reference, xs, ys =
    List.find_exn binops ~f:(fun (op, _, _, _, _) -> Ops.equal_binop op Mod)
  in
  let pairs =
    List.concat_map xs ~f:(fun x ->
        List.filter_map ys ~f:(fun y ->
            let x = c.round x and y = c.round y in
            Option.some_if
              Float.(is_finite x && is_finite y && y <> 0. && reference x y <> 0.)
              (x, y)))
  in
  let n = List.length pairs in
  let kernel = "umsv_fold_" ^ c.label in
  let out = Ll_test.node_factory ~prec:c.prec ~first_id ~dims:[| n |] () (kernel ^ "_out") in
  Ll_test.materialize out;
  let body =
    List.foldi pairs ~init:LL.Noop ~f:(fun j acc (x, y) ->
        Ll_test.seq acc
          (Ll_test.set out
             [| Ll_test.fixed j |]
             (LL.Binop (Mod, (LL.Constant x, c.prec), (LL.Constant y, c.prec)))))
  in
  let o = Ll_test.optimize ~name:kernel body in
  let got =
    List.hd_exn (Ll_test.execute ~name:kernel o ~seed:[ (out, Ll_test.blank n) ] ~read:[ out ])
  in
  List.map2_exn pairs (Array.to_list got) ~f:(fun (x, y) v -> (name, reference, x, y, v))

(* What a result is held to, against the f64 value at the result's precision [c]. *)
type contract =
  | Ulps of float  (** Within this many ulps of [c]. *)
  | Absolute of float  (** Within this absolute error, plus one ulp of [c] for the rounding. *)
  | Relative of float  (** Within this relative error, plus one ulp of [c] for the rounding. *)
  | Unit_range  (** Any finite value in [-1, 1]: an accepted loss of the argument reduction. *)

(* Whether [v] meets [contract] against the f64 value [want]: NaN where [want] is NaN, the signed
   zero or infinity bitwise where [want] rounds to one at [c]'s precision. *)
let meets c contract ~want v =
  let rounded = c.round want in
  if Float.is_nan want then Float.is_nan v
  else if Float.(rounded = 0.) || not (Float.is_finite rounded) then
    Int64.equal (Int64.bits_of_float v) (Int64.bits_of_float rounded)
  else
    let err = Float.abs (v -. want) in
    match contract with
    | Ulps u -> Float.(err /. c.ulp want <= u)
    | Absolute a -> Float.(err <= a +. c.ulp want)
    | Relative r -> Float.(err <= (r *. abs want) +. c.ulp want)
    | Unit_range -> Float.(is_finite v && abs v <= 1.)

(* [fmod] is exact in every precision. *)
let stated c name x = Ulps (if String.equal name "fmod" then 0. else c.bound name x)

(* CUDA's documented contracts for its fast f32 intrinsics (CUDA Math API, "Intrinsic Functions"),
   with the growth of [__sinf]/[__cosf]'s error past pi as measured. *)
let cuda_fast c name x =
  let ax = Float.abs x in
  match name with
  | "sin" | "cos" ->
      if Float.(ax >= 0x1p22) then Unit_range
      else Absolute ((2. **. -21.41) +. if Float.(ax > Float.pi) then Float.ldexp ax (-23) else 0.)
  | "tanh" -> Relative (2. **. -10.987)
  | "exp" -> Ulps (2. +. (1.173 *. ax))
  | "log" when Float.(0.5 <= x && x <= 2.) -> Absolute (2. **. -21.41)
  | "log2" when Float.(0.5 <= x && x <= 2.) -> Absolute (2. **. -22.)
  | _ -> stated c name x

(* CUDA's [--use_fast_math] [powf] is [__powf], [exp2(y * log2(x))] through the intrinsics: their 2
   ulps each, the logarithm's relative error carried into the result scaled by [|y log2 |x||] (times
   [ln 2] and the product's rounding, together under 2). Measured at 5 ulps for [pow(0.25, -3)]. *)
let cuda_powf x y =
  (* [0 * log2 0] and [inf * log2 1] are NaN: their exact results ([pow(0, 0) = pow(1, inf) = 1])
     carry no logarithm error. *)
  let k = Float.abs (y *. Stdlib.Float.log2 (Float.abs x)) in
  Ulps (2. +. (2. *. if Float.is_finite k then k else 0.))

(* An f32 subnormal flushed to the zero of its sign. *)
let flush v = if Float.(v <> 0. && abs v < 0x1p-126) then Float.copysign 0. v else v

(* Whether [v = name x] meets [contract] at [c] -- with [~ftz], also against f64 at the flushed
   input or of the flushed f32 result -- reporting a failure on stderr. [flushed] counts the results
   that met it only by the flush. *)
let judge ?(ftz = false) ?flushed c contract (name, reference, x, v) =
  let want = reference x in
  let ok =
    meets c contract ~want v
    || ftz
       && (meets c contract ~want:(reference (flush x)) v
          || meets c contract ~want:(flush (f32 want)) v)
       &&
       (Option.iter flushed ~f:Int.incr;
        true)
  in
  if not ok then
    Stdio.eprintf "(not part of the golden) %s %s %s(%h) = %h, f64 %h, %g ulps\n" backend_name
      c.label name x v want
      (Float.abs (v -. want) /. c.ulp want);
  ok

let is_special x = Float.(x = 0.) || not (Float.is_finite x)

(* Whether [v = name x y] meets [contract] at [c], reporting a failure on stderr. *)
let judge2 c contract (name, reference, x, y, v) =
  let want = reference x y in
  let ok = meets c contract ~want v in
  if not ok then
    Stdio.eprintf "(not part of the golden) %s %s %s(%h, %h) = %h, f64 %h\n" backend_name c.label
      name x y v want;
  ok

let () =
  Verdict.case "unary math special values" (fun () ->
      List.iteri cases ~f:(fun ci c ->
          let inputs, got = run c ~first_id:(19900 + (100 * ci)) in
          let n = List.length inputs in
          let results =
            List.mapi (Array.to_list got) ~f:(fun j v ->
                let _, name, reference, ftz = List.nth_exn ops (j / n) in
                (ftz, (name, reference, List.nth_exn inputs (j % n), v)))
          in
          let specials, finite =
            List.partition_tf results ~f:(fun (_, (_, _, x, _)) -> is_special x)
          in
          let all = List.map ~f:snd in
          p_all
            (Printf.sprintf
               "%s %s of +-0, +-inf and NaN: NaN where f64 gives NaN, otherwise the signed zero or \
                infinity f64 gives, or within the stated ulps of it (every backend)"
               c.label op_names) (all specials) ~f:(fun ((name, _, x, _) as r) ->
              judge c (stated c name x) r);
          gated_all
            ~when_:(is_cc || is_hip || (is_cuda && not c.cuda_intrinsics))
            ~on:backend_name
            (Printf.sprintf
               "%s %s of finite nonzero inputs: within the stated ulps of f64 with subnormals kept \
                (skipped on Metal and on CUDA's f32 intrinsics, accepted vendor behaviour claimed \
                by their own contracts)"
               c.label op_names)
            (all finite)
            ~f:(fun ((name, _, x, _) as r) -> judge c (stated c name x) r);
          let flushed = ref 0 in
          gated_all ~when_:is_metal ~on:backend_name
            (Printf.sprintf
               "Metal %s %s: NaN where f64 gives NaN, the signed zero or infinity where f64 rounds \
                to one, otherwise within the stated ulps of f64 -- for %s, of f64 at the input or \
                the f32 result flushed from subnormal to zero"
               c.label op_names ftz_names) results ~f:(fun (ftz, ((name, _, x, _) as r)) ->
              judge ~ftz ~flushed c (stated c name x) r);
          if is_metal then
            Stdio.eprintf
              "(not part of the golden) %s: %d results matched only with a subnormal flushed\n"
              c.label !flushed;
          p_all
            (Printf.sprintf
               "%s %s over +-0, +-inf, NaN and dyadic operands: NaN where C gives NaN, otherwise \
                the signed zero or infinity C gives, or within the stated ulps of it -- CUDA's f32 \
                pow within __powf's 2 + 2 |y log2 |x|| -- and fmod exactly (every backend)"
               c.label binop_names)
            (run_binary c ~first_id:(20500 + (100 * ci)))
            ~f:(fun ((name, _, x, y, _) as r) ->
              judge2 c
                (if is_cuda && Ops.equal_prec c.prec Ops.single && String.equal name "pow" then
                   cuda_powf x y
                 else stated c name x)
                r);
          p_all
            (Printf.sprintf
               "%s fmod of constant operands, folded on the host: exactly C's fmod, the remainder \
                taking the sign of the dividend under a negative modulus too (every backend)"
               c.label)
            (run_folded_fmod c ~first_id:(20800 + (100 * ci)))
            ~f:(fun r -> judge2 c (Ulps 0.) r);
          if c.cuda_intrinsics then (
            let flushed = ref 0 in
            gated_all ~when_:is_cuda ~on:backend_name
              (Printf.sprintf
                 "CUDA %s %s under --use_fast_math: within the fast intrinsics' contracts -- sin \
                  and cos absolute 2^-21.41 (plus 2^-23 |x| past pi) below 2^22 and any value in \
                  [-1, 1] from there, tanh relative 2^-10.987, exp 2 + 1.173 |x| ulps, log and \
                  log2 absolute 2^-21.41 and 2^-22 on [0.5, 2] and the stated ulps elsewhere, f32 \
                  subnormals flushed"
                 c.label op_names) (all results) ~f:(fun ((name, _, x, _) as r) ->
                judge ~ftz:true ~flushed c (cuda_fast c name x) r);
            if is_cuda then
              Stdio.eprintf
                "(not part of the golden) CUDA %s: %d results matched only with a subnormal \
                 flushed, %d inputs from 2^22 held to [-1, 1] only\n"
                c.label !flushed
                (2 * List.count inputs ~f:(fun x -> Float.(is_finite x && abs x >= 0x1p22)))));
      let f32_case = List.hd_exn cases in
      let band_results =
        List.concat
          (List.map2_exn (run_band ())
             [ ("log", Float.log); ("log2", Stdlib.Float.log2) ]
             ~f:(fun got (name, reference) ->
               List.map2_exn band (Array.to_list got) ~f:(fun x v -> (name, reference, x, v))))
      in
      p_all
        "f32 log and log2 have the sign of x - 1 on every f32 within 2^-13 of 1, and are +0 at 1 \
         -- Box-Muller's sqrt (-2 log u) needs log u <= 0 (every backend)"
        band_results ~f:(fun (name, reference, x, v) ->
          let ok =
            if Float.(x < 1.) then Float.(v < 0.)
            else if Float.(x > 1.) then Float.(v > 0.)
            else Int64.equal (Int64.bits_of_float v) 0L
          in
          if not ok then
            Stdio.eprintf "(not part of the golden) %s band %s(%h) = %h, f64 %h\n" backend_name name
              x v (reference x);
          ok);
      p_all
        "f32 log and log2 on every f32 within 2^-13 of 1: within the backend's contract -- the \
         stated ulps of f64, CUDA's absolute 2^-21.41 and 2^-22 (every backend)"
        band_results ~f:(fun ((name, _, x, _) as r) ->
          judge f32_case (if is_cuda then cuda_fast f32_case name x else stated f32_case name x) r))

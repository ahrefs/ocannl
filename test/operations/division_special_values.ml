(* Division and reciprocal at f64, f32 and half on every backend (gh-ocannl-1782), over the operands
   where a reciprocal estimate departs from IEEE: +-0, +-inf and NaN on either side, subnormal
   divisors and dividends, quotients that overflow to infinity and that underflow into the
   subnormals or to zero. IEEE division is correctly rounded, so every result is held to C's
   quotient exactly: the host's f64 [x /. y] rounded to the precision (correct for f32 and half too,
   since f64 has more than twice their precision plus two bits), NaN where it is NaN and otherwise
   bitwise -- the sign of a zero included. [Recip] is [1 / x] over the same operands.

   HIP compiles under [-ffast-math], and hiprtc rendered every f64 [x / y] (and [1.0 / x]) as a
   reciprocal estimate refined by Newton steps: NaN for every [x / +-0], [x / +-inf], [+-inf / y]
   and [1 / +-0], [1 / +-inf], [+0] for [-0 / y], an infinite [x / x] for a subnormal [x], and
   [DBL_MAX / DBL_MAX] one ulp below 1 -- measured on gfx1151 (unified memory); both claims below
   failed there. [-fapprox-func] makes the estimate (dropping it alone fixed the special values) and
   [-freciprocal-math] the [x * (1 / y)] rounding (dropping both made every precision exact). f64
   now renders through [Builtins_hip.ocannl_div_f64], LLVM's correctly rounded expansion in target
   builtins, which no fast-math flag reaches, so HIP's f64 is held to both claims.

   HIP's f32 and half division (and bfloat16, through f32) stay fast math's estimate, as CUDA's f32
   is under [--use_fast_math]: a recorded boundary of the fast policy -- [hip_fast_math] off
   (gh-ocannl-1779, what [profile=reproducible] sets) drops the umbrella and makes them IEEE, held
   to both claims like every other precision (the [-reproducible] rule). Measured on
   gfx1151: quotients 1 ulp off ([3 / 3] is [1 - 2^-24], [1 / 0.75] [0x1.555554p+0]), a subnormal
   divisor's reciprocal overflowing to infinity, an f32 divisor from 2^126 up flushing its
   reciprocal to zero ([1 / FLT_MAX] is 0, [inf / FLT_MAX] NaN), half [+-0 / subnormal] NaN. Those
   backends are held to the estimate's own contract instead -- the special values C's and 2 ulps
   elsewhere, over the pairs with no subnormal operand or quotient and an f32 divisor below 2^126.

   The operand classes split the claims. With one operand among +-0, +-inf and NaN and the other
   normal (or special too), the quotient is decided by the special operand alone: every IEEE backend
   is held to C's value there, Metal included. The other pairs -- both finite and nonzero, or a
   subnormal on either side -- are where an estimate's rounding and a flushed subnormal show: the
   IEEE backends but Metal, which flushes f32 subnormals in its arithmetic, are held to the
   correctly rounded quotient there.

   The negative control runs on the device: HIP's f32 division is the same kind of estimate its f64
   division was, and the judgement of the special-operand claim must reject some of its results. *)
(* cc-fp-contract: no-contraction -- every kernel divides or takes a reciprocal of a stored operand;
   no product feeds an add, so contraction has nothing to fuse. *)
open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Stdio.eprintf "division-special-values backend: %s\n" backend_name
let is_metal = String.equal backend_name "metal"
let is_cuda = String.equal backend_name "cuda"
let is_hip = String.equal backend_name "hip"
let cuda_fast_math = Ir.Cuda_like_config.cuda_fast_math ()
let hip_fast_math = Ir.Cuda_like_config.hip_fast_math ()
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let f16 x = Ops.half_to_single (Ops.single_to_half (f32 x))
let specials = [ -0.; 0.; Float.infinity; Float.neg_infinity; Float.nan ]

(* Per precision: its rounding, its smallest normal magnitude, and operands exactly representable in
   it -- the special values, a quotient that rounds ([1 / 0.75]) and one that is exact ([-3 /
   0.75]), the smallest normal, the smallest subnormal and a larger one, the largest finite value,
   and a large and a small normal power of two whose quotients overflow and underflow. *)
type case = {
  label : string;
  prec : Ops.prec;
  round : float -> float;
  min_normal : float;
  ulp : float -> float;
  operands : float list;
}

(* The spacing at [|v|] of a binary format with [mant] stored fraction bits and smallest normal
   exponent [emin]: below 2^emin, the subnormal step. *)
let spacing ~mant ~emin v =
  let e = snd (Float.frexp (Float.abs v)) - 1 in
  Float.ldexp 1.0 (Int.max e emin - mant)

let cases =
  [
    {
      label = "f64";
      prec = Ops.double;
      round = Fn.id;
      min_normal = 0x1p-1022;
      ulp = spacing ~mant:52 ~emin:(-1022);
      operands =
        specials
        @ [
            1.;
            -3.;
            0.75;
            0x1p-1022;
            0x1p-1074;
            -0x1.8p-1061;
            0x1.fffffffffffffp+1023;
            -0x1p1000;
            0x1p-1000;
          ];
    };
    {
      label = "f32";
      prec = Ops.single;
      round = f32;
      min_normal = 0x1p-126;
      ulp = spacing ~mant:23 ~emin:(-126);
      operands =
        specials
        @ [ 1.; -3.; 0.75; 0x1p-126; 0x1p-149; -0x1.8p-136; 0x1.fffffep+127; -0x1p100; 0x1p-100 ];
    };
    {
      label = "half";
      prec = Ops.half;
      round = f16;
      min_normal = 0x1p-14;
      ulp = spacing ~mant:10 ~emin:(-14);
      operands =
        specials @ [ 1.; -3.; 0.75; 0x1p-14; 0x1p-24; -0x1.8p-20; 65504.; -0x1p12; 0x1p-10 ];
    };
  ]

let is_special x = Float.(x = 0.) || not (Float.is_finite x)
let is_subnormal c x = Float.(x <> 0. && abs x < c.min_normal)

(* One kernel at [c.prec]: [out.(j) = lhs.(j) / rhs.(j)] over every pair of operands, then [out.(m +
   i) = 1 / operands.(i)] through [Recip]. Returns each result as [(name, x, y, v)], [x = 1] for a
   reciprocal. *)
let run c ~first_id =
  let xs = c.operands in
  let pairs = List.concat_map xs ~f:(fun x -> List.map xs ~f:(fun y -> (x, y))) in
  let m = List.length pairs and n = List.length xs in
  let name = "dsv_" ^ c.label in
  let node = Ll_test.node_factory ~prec:c.prec ~first_id ~dims:[| m |] () in
  let lhs = node (name ^ "_lhs") and rhs = node (name ^ "_rhs") in
  let out = node ~dims:[| m + n |] (name ^ "_out") in
  List.iter [ lhs; rhs; out ] ~f:Ll_test.materialize;
  let get t j = (Ll_test.get t [| Ll_test.fixed j |], c.prec) in
  let set j v = Ll_test.set out [| Ll_test.fixed j |] v in
  let divisions =
    List.foldi pairs ~init:LL.Noop ~f:(fun j acc _ ->
        Ll_test.seq acc (set j (LL.Binop (Div, get lhs j, get rhs j))))
  in
  let body =
    List.foldi xs ~init:divisions ~f:(fun i acc _ ->
        Ll_test.seq acc (set (m + i) (LL.Unop (Recip, get rhs (i * (n + 1))))))
  in
  let o = Ll_test.optimize ~name body in
  let got =
    List.hd_exn
      (Ll_test.execute ~name o
         ~seed:
           [
             (lhs, Array.of_list (List.map pairs ~f:fst));
             (rhs, Array.of_list (List.map pairs ~f:snd));
             (out, Ll_test.blank (m + n));
           ]
         ~read:[ out ])
    |> Array.to_list
  in
  let quotients, reciprocals = List.split_n got m in
  List.map2_exn pairs quotients ~f:(fun (x, y) v -> ("div", x, y, v))
  @ List.map2_exn xs reciprocals ~f:(fun y v -> ("recip", 1., y, v))

(* Whether [v] is C's [x / y] at [c]'s precision: NaN where it is NaN, otherwise bitwise. *)
let exact c (_, x, y, v) =
  let want = c.round (x /. y) in
  if Float.is_nan want then Float.is_nan v
  else Int64.equal (Int64.bits_of_float v) (Int64.bits_of_float want)

(* {!exact}, reporting a failure on stderr. *)
let judge c ((name, x, y, v) as r) =
  let ok = exact c r in
  if not ok then
    Stdio.eprintf "(not part of the golden) %s %s %s(%h, %h) = %h, C %h\n" backend_name c.label name
      x y v
      (c.round (x /. y));
  ok

(* Pairs one special operand decides: a special operand, and no subnormal one. *)
let decided_by_special c (_, x, y, _) =
  (is_special x || is_special y) && not (is_subnormal c x || is_subnormal c y)

(* Whether this backend renders [c]'s division as a reciprocal estimate, [x * rcp(y)]: HIP's f32 and
   half under [-ffast-math] ([hip_fast_math]), and CUDA's f32 under [--use_fast_math]
   ([--prec-div=false]). *)
let estimates c =
  (is_hip && hip_fast_math && not (Ops.equal_prec c.prec Ops.double))
  || (is_cuda && cuda_fast_math && Ops.equal_prec c.prec Ops.single)

(* The pairs an estimate's contract covers: no subnormal operand or quotient, and for f32 a divisor
   below 2^126, whose reciprocal is subnormal -- flushed, so [inf / FLT_MAX] is [inf * 0], NaN, and
   [1 / FLT_MAX] is 0. *)
let estimate_domain c ((_, x, y, _) as r) =
  let q = c.round (x /. y) in
  (decided_by_special c r || not (is_subnormal c x || is_subnormal c y || is_subnormal c q))
  && not (Ops.equal_prec c.prec Ops.single && Float.(abs y >= 0x1p126 && is_finite y))

(* Whether [v] is within 2 ulps of C's [x / y] at [c]'s precision -- NaN, the signed zeros and the
   infinities still bitwise. *)
let within_estimate c ((name, x, y, v) as r) =
  let want = c.round (x /. y) in
  let ok =
    exact c r || Float.(is_finite want && want <> 0. && abs (v -. want) <= 2. *. c.ulp want)
  in
  if not ok then
    Stdio.eprintf "(not part of the golden) %s %s %s(%h, %h) = %h, C %h\n" backend_name c.label name
      x y v want;
  ok

let () =
  Verdict.case "division special values" (fun () ->
      let results = List.mapi cases ~f:(fun ci c -> lazy (run c ~first_id:(22000 + (100 * ci)))) in
      List.iter2_exn cases results ~f:(fun c results ->
          let has_prec = not (is_metal && Ops.equal_prec c.prec Ops.double) in
          let ieee = has_prec && not (estimates c) in
          Verdict.in_leg ~leg:(Verdict.leg ~when_:ieee ~on:backend_name ())
            (Printf.sprintf
               "%s x / y and 1 / y with an operand among +-0, +-inf and NaN and the other normal \
                or special: C's quotient, NaN where C gives NaN and otherwise bitwise (skipped on \
                Metal at f64, which it has not, and where the division is a reciprocal estimate: \
                HIP's f32 and half, CUDA's f32 under --use_fast_math)"
               c.label) (fun label ->
              p_all label (List.filter (Lazy.force results) ~f:(decided_by_special c)) ~f:(judge c));
          Verdict.in_leg
            ~leg:(Verdict.leg ~when_:(ieee && not is_metal) ~on:backend_name ())
            (Printf.sprintf
               "%s x / y and 1 / y of the other operands, subnormal, overflowing and underflowing \
                quotients among them: correctly rounded, bitwise C's (skipped where the division \
                is a reciprocal estimate, and on Metal, which flushes f32 subnormals)"
               c.label)
            (fun label ->
              p_all label
                (List.filter (Lazy.force results) ~f:(fun r -> not (decided_by_special c r)))
                ~f:(judge c));
          if not (Ops.equal_prec c.prec Ops.double) then
            Verdict.in_leg
              ~leg:(Verdict.leg ~when_:(estimates c) ~on:backend_name ())
              (Printf.sprintf
                 "%s x / y and 1 / y where the division is a reciprocal estimate (HIP under \
                  -ffast-math, CUDA's f32 under --use_fast_math), with no subnormal operand or \
                  quotient and an f32 divisor below 2^126: NaN, the signed zeros and the \
                  infinities C's, otherwise within 2 ulps of C's quotient"
                 c.label)
              (fun label ->
                p_all label
                  (List.filter (Lazy.force results) ~f:(estimate_domain c))
                  ~f:(within_estimate c)));
      (* The negative control, on the device: HIP's f32 division is still the reciprocal estimate
         its f64 division was, and the judgement of the first claim must reject some of its results
         -- [+-inf / FLT_MAX] is NaN there. [hip_fast_math] off (gh-ocannl-1779) turns the estimate
         off, so this leg opens only where it is on. *)
      let f32_case = List.nth_exn cases 1 and f32_results = List.nth_exn results 1 in
      Verdict.in_leg
        ~leg:(Verdict.leg ~when_:(is_hip && hip_fast_math) ~on:backend_name ())
        "negative control: HIP's f32 x / y under -ffast-math, a reciprocal estimate, departs from \
         C's quotient on some pair with a special operand, where the f32 claim above would fail it"
        (fun label ->
          p_exists label
            (List.filter (Lazy.force f32_results) ~f:(decided_by_special f32_case))
            ~f:(fun r -> not (exact f32_case r))))

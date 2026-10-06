(* Fractional and dynamic half powers (gh-ocannl-1198). Neither vendor header has a half pow, and
   both vendors' [hexp2] is unary, so CUDA and HIP widen both operands, take f32 [powf], and round
   back to half once, as cc's codegen does. The domain is f32 [powf]'s: a negative base under a
   fractional exponent is NaN. Known integer exponents take the integer-power helper instead
   ([integer_power_domain]); here every exponent is either fractional or only known at run time.

   Tolerance: one ulp at half precision, relative to an f64 reference over the half-rounded
   operands, i.e. [|got - want| <= 2^-10 * |want|]. That admits the f32 [powf] error (a few f32
   ulps, fast math included) plus the single rounding to half, and nothing coarser: the reference
   values are all normal halves, so a 2^-10 relative bound is one step of the half grid (two steps
   of the finer grid just below a power of two, when [want] is one). *)
open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
module Generated = Test_utils.Generated
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Stdio.eprintf "half-float-power backend: %s\n" backend_name
let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Generated.init ~backend_name
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let f16 x = Ops.half_to_single (Ops.single_to_half (f32 x))

(* The backends whose half pow is f32 [powf] rounded once; Metal's half [pow] is its own
   overload. *)
let widens_to_powf =
  List.mem [ "cc"; "multidev_cc"; "cuda"; "hip" ] backend_name ~equal:String.equal

let cuda_like = List.mem [ "cuda"; "hip" ] backend_name ~equal:String.equal

(* [Const]: the exponent is a literal in the kernel; [Dynamic]: it is read from a buffer. *)
type exponent = Const | Dynamic

let cases =
  List.concat_map [ 0.3; 1.7; 2.5; 10.; 100. ] ~f:(fun x ->
      List.map [ 0.5; 1.5; -0.75; 2.2 ] ~f:(fun e -> (x, e, Const)))
  @ [
      (0.3, 0.1, Dynamic);
      (1.7, -1.25, Dynamic);
      (2.5, 3., Dynamic);
      (10., 0.5, Dynamic);
      (100., -2., Dynamic);
      (0.01, 1.5, Dynamic);
      (7., 4.6, Dynamic);
    ]

(* Negative bases under fractional exponents: NaN in f32 [powf], fast math included. *)
let nan_cases = [ (-2., 0.5, Const); (-2.5, 1.5, Dynamic) ]
let is_normal_half v = Float.is_finite v && Float.(abs v >= 0x1p-14) && Float.(abs v <= 65504.)

(* Runs [cases] at [prec] under routine [name] and reads the results back. *)
let run ~prec ~name ~first_id cases =
  let len = List.length cases in
  let node = Ll_test.node_factory ~prec ~first_id ~dims:[| len |] () in
  let base = node (name ^ "_base") and exp = node (name ^ "_exp") and out = node (name ^ "_out") in
  List.iter [ base; exp; out ] ~f:Ll_test.materialize;
  let body =
    List.foldi cases ~init:LL.Noop ~f:(fun i acc (_, e, kind) ->
        let at = [| Ll_test.fixed i |] in
        let exponent =
          match kind with Const -> Ll_test.c (f16 e) | Dynamic -> Ll_test.get exp at
        in
        Ll_test.seq acc
          (Ll_test.set out at (LL.Binop (ToPowOf, (Ll_test.get base at, prec), (exponent, prec)))))
  in
  let o = Ll_test.optimize ~name body in
  let seed_of f = Array.of_list_map cases ~f in
  List.hd_exn
    (Ll_test.execute ~name o
       ~seed:
         [
           (base, seed_of (fun (x, _, _) -> f16 x));
           (exp, seed_of (fun (_, e, _) -> f16 e));
           (out, Ll_test.blank len);
         ]
       ~read:[ out ])

let () =
  Verdict.case "half powers" (fun () ->
      let got = run ~prec:Ops.half ~name:"hpow" ~first_id:19100 cases in
      let want = List.map cases ~f:(fun (x, e, _) -> f16 (Float.( ** ) (f16 x) (f16 e))) in
      p_all "references are normal halves, so the tolerance is one ulp at half precision" want
        ~f:is_normal_half;
      p_alli
        "fractional and dynamic half powers are within one ulp at half precision of the f64 \
         reference"
        cases ~f:(fun i (x, e, kind) ->
          let want = List.nth_exn want i in
          let ok = Float.(abs (got.(i) - want) <= 0x1p-10 * abs want) in
          if not ok then
            Stdio.eprintf "(not part of the golden) %s %h^%h: got %h want %h\n"
              (match kind with Const -> "const" | Dynamic -> "dynamic")
              x e got.(i) want;
          ok);
      (* The same operands at f32, narrowed to half on the host: one rounding of the same [powf]
         result. Pinned wherever that is the stated policy. *)
      let single = run ~prec:Ops.single ~name:"hpow_f32" ~first_id:19200 cases in
      gated_alli ~when_:widens_to_powf ~on:backend_name
        "half powers are f32 powf rounded to half once" (Array.to_list got) ~f:(fun i v ->
          let want = f16 single.(i) in
          let ok = Int64.equal (Int64.bits_of_float v) (Int64.bits_of_float want) in
          if not ok then
            Stdio.eprintf "(not part of the golden) case %d: half %h, f32 narrowed %h\n" i v want;
          ok);
      let src = Generated.read "hpow" in
      p "half powers never call the binary-misused hexp2"
        (not (String.is_substring src ~substring:"hexp2("));
      p "fractional and dynamic exponents bypass the integer-power helper"
        (not (String.is_substring src ~substring:"ocannl_powi_"));
      gated ~when_:cuda_like ~on:backend_name "half powers widen both operands into powf"
        (String.is_substring src ~substring:"__float2half(powf(__half2float("))

let () =
  Verdict.case "negative bases" (fun () ->
      let got = run ~prec:Ops.half ~name:"hpow_neg" ~first_id:19300 nan_cases in
      p_all "negative bases under fractional exponents are NaN" (Array.to_list got) ~f:Float.is_nan)

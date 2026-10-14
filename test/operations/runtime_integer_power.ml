(* Powers whose exponent is only known at run time (gh-ocannl-1240). A known integral exponent takes
   the integer-power helper ([integer_power_domain]); a run-time one reaches the vendor [pow], which
   CUDA's [--use_fast_math] lowers to [exp2(y * log2(x))]: NaN for every negative base, the integral
   exponents included, at f32 and -- through the widened f32 [powf] of gh-ocannl-1198 -- at f16
   alike. CUDA and HIP therefore route a run-time float power through a device helper
   ([Builtins_cc.runtime_power_builtins]) that takes an integral exponent on the base's magnitude
   and restores the sign from the exponent's parity; every other exponent keeps the vendor domain,
   so a negative base under a fractional exponent stays NaN as in IEEE/C [pow]. This is a domain
   fix, not a correctly rounded [pow]: the magnitude is still the vendor's [pow] of [|x|]. Metal
   keeps its own [pow]: under its fast math functions it already returns these powers (M4 Max, macOS
   26), and the claims below hold it to that. [runtime_power_fast_math_emulation] checks the
   helper's logic per PR, without a GPU.

   Oracle: the integer power in f64 of the precision-rounded base, rounded to the precision once.
   Tolerance, relative to it: 2^-18 at f32, which admits CUDA's documented fast [__powf] error (2 +
   floor(|1.173 y|) ulps, at most 4 ulps = 2^-21 here) eight times over and nothing a sign or domain
   error could hide in; one ulp at half precision (2^-10) at f16, as in [half_float_power].
   Device-produced digits go to stderr only. *)
open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
module Generated = Test_utils.Generated
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Stdio.eprintf "runtime-integer-power backend: %s\n" backend_name
let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Generated.init ~backend_name
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let f16 x = Ops.half_to_single (Ops.single_to_half (f32 x))
let cuda_like = List.mem [ "cuda"; "hip" ] backend_name ~equal:String.equal

(* The bitwise mirror claim: by construction where the helper runs, and measured for Metal's own
   [pow] (fast math functions, Safe math mode; M4 Max). cc's is the host libm's business. *)
let gpu_backend = cuda_like || String.equal backend_name "metal"
let bases = [ -1.5; -2.; -0.75; -1.25 ]
let exponents = [ 0; 1; 2; 3; -1; -2 ]

(* Every negative base under every exponent, then each base's positive mirror. *)
let negative = List.concat_map bases ~f:(fun x -> List.map exponents ~f:(fun n -> (x, n)))
let mirrored = List.map negative ~f:(fun (x, n) -> (Float.neg x, n))
let fractional = [ (-2., 0.5); (-1.5, -1.5); (-0.75, 2.5) ]

(* f32 subnormal exponents: CUDA's fast math flushes them to zero in arithmetic and comparisons, so
   the helper classifies them by their bits to keep them fractional. Metal's own [pow] flushes them
   too and answers as for a zero exponent (M4 Max), so the claim skips Metal; cc's libm keeps
   them. *)
let subnormal_fractional = [ (-2., 0x1p-149); (-1.5, -0x1p-130) ]
let is_metal = String.equal backend_name "metal"

(* Runs [x ** e] for each [(x, e)] at [prec], with both operands read from buffers. *)
let run ~prec ~narrow ~name ~first_id cases =
  let len = List.length cases in
  let node = Ll_test.node_factory ~prec ~first_id ~dims:[| len |] () in
  let base = node (name ^ "_base") and exp = node (name ^ "_exp") and out = node (name ^ "_out") in
  List.iter [ base; exp; out ] ~f:Ll_test.materialize;
  let body =
    List.foldi cases ~init:LL.Noop ~f:(fun i acc _ ->
        let at = [| Ll_test.fixed i |] in
        Ll_test.seq acc
          (Ll_test.set out at
             (LL.Binop (ToPowOf, (Ll_test.get base at, prec), (Ll_test.get exp at, prec)))))
  in
  let o = Ll_test.optimize ~name body in
  let seed_of f = Array.of_list_map cases ~f:(fun c -> narrow (f c)) in
  List.hd_exn
    (Ll_test.execute ~name o
       ~seed:[ (base, seed_of fst); (exp, seed_of snd); (out, Ll_test.blank len) ]
       ~read:[ out ])

let bits v = Int64.bits_of_float v

let check ?subnormal ~prec ~narrow ~tol ~name ~first_id ~call () =
  Verdict.case name (fun () ->
      let as_float (x, n) = (x, Float.of_int n) in
      let got = run ~prec ~narrow ~name ~first_id (List.map (negative @ mirrored) ~f:as_float) in
      let n_neg = List.length negative in
      p_alli (name ^ ": negative bases under run-time integral exponents match the integer power")
        negative ~f:(fun i (x, n) ->
          let want = narrow (Float.( ** ) (narrow x) (Float.of_int n)) in
          let ok = Float.(abs (got.(i) - want) <= tol * abs want) in
          if not ok then
            Stdio.eprintf "(not part of the golden) %s %h^%d: got %h want %h\n" name x n got.(i)
              want;
          ok);
      gated_alli ~when_:gpu_backend ~on:backend_name
        (name
       ^ ": a negative base's power is its mirror's, bitwise, negated exactly for odd exponents")
        negative ~f:(fun i (_, n) ->
          let mirror = got.(n_neg + i) in
          let want = if n % 2 = 0 then mirror else Float.neg mirror in
          Int64.equal (bits got.(i)) (bits want));
      let nan = run ~prec ~narrow ~name:(name ^ "_frac") ~first_id:(first_id + 50) fractional in
      p_all
        (name ^ ": negative bases under run-time fractional exponents stay NaN")
        (Array.to_list nan) ~f:Float.is_nan;
      Option.iter subnormal ~f:(fun cases ->
          let got = run ~prec ~narrow ~name:(name ^ "_subnormal") ~first_id:(first_id + 70) cases in
          Array.iter got ~f:(fun v ->
              Stdio.eprintf "(not part of the golden) %s subnormal exponent: %h\n" name v);
          gated_all ~when_:(not is_metal) ~on:backend_name
            (name ^ ": negative bases under run-time subnormal exponents stay NaN")
            (Array.to_list got) ~f:Float.is_nan);
      let zero_bases = [ -0.; 0.; Float.infinity; Float.neg_infinity; Float.nan ] in
      let ones =
        run ~prec ~narrow ~name:(name ^ "_zero") ~first_id:(first_id + 80)
          (List.map zero_bases ~f:(fun x -> (x, 0.)))
      in
      p_all (name ^ ": a run-time zero exponent gives 1 for every base, as in C pow")
        (Array.to_list ones) ~f:(fun v -> Float.equal v 1.);
      let src = Generated.read name in
      p
        (name ^ ": run-time exponents bypass the constant integer-power helper")
        (not (String.is_substring src ~substring:"ocannl_powi_"));
      gated ~when_:cuda_like ~on:backend_name
        (name ^ ": the kernel calls the run-time power helper")
        (String.is_substring src ~substring:call))

let () =
  check ~prec:Ops.single ~narrow:f32 ~tol:0x1p-18 ~name:"rpow_f32" ~first_id:19700
    ~subnormal:subnormal_fractional ~call:"ocannl_pow_f32(" ()

let () =
  check ~prec:Ops.half ~narrow:f16 ~tol:0x1p-10 ~name:"rpow_f16" ~first_id:19800
    ~call:"__float2half(ocannl_pow_f32(__half2float(" ()

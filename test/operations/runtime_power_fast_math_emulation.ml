(* The CUDA-like run-time power helper (gh-ocannl-1240), compiled by the host C compiler under an
   emulation of CUDA's [--use_fast_math] lowering of [powf]/[pow] to [exp2(y * log2(x))], so that
   every PR checks the helper's sign and domain handling without a GPU. [runtime_integer_power]
   executes the same helper on the real devices; this is the GPU-free per-PR pin of its logic.

   The negative control is the emulated vendor call itself: it must lose a negative base under an
   integral exponent, or the emulation is not reproducing the defect the helper exists for. *)
open Verdict.Claims
module B = Context.Builtins_cc
module Cc = Context.Cc_backend

let shims =
  {|
#include <math.h>
#include <string.h>
#define __device__
#define __forceinline__ static inline
static unsigned int __float_as_uint(float x) { unsigned int b; memcpy(&b, &x, sizeof b); return b; }
static long long __double_as_longlong(double x) { long long b; memcpy(&b, &x, sizeof b); return b; }
/* CUDA --use_fast_math's lowering of the vendor pow, which the helper must survive. */
#define powf(x, y) exp2f((y) * log2f(x))
#define pow(x, y) exp2((y) * log2(x))
|}

let probes =
  {|
double probe_f32(double x, double y) { return ocannl_pow_f32((float)x, (float)y); }
double probe_f64(double x, double y) { return ocannl_pow_f64(x, y); }
double probe_raw_f32(double x, double y) { return powf((float)x, (float)y); }
|}

let helper_source =
  String.concat "\n"
    (List.map
       (fun (_, def, _) -> def)
       (B.runtime_power_builtins ~prefix:"__device__ __forceinline__"))

let bits = Int64.bits_of_float
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let bases = [ -1.5; -2.; -0.75; -1.25 ]
let exponents = [ 0; 1; 2; 3; -1; -2 ]
let cases = List.concat_map (fun x -> List.map (fun n -> (x, n)) exponents) bases

let () =
  case "emulated fast-math run-time powers" @@ fun () ->
  let src = Filename.temp_file "ocannl_runtime_power_" ".c" in
  let dll = Filename.temp_file "ocannl_runtime_power_" (if Sys.win32 then ".dll" else ".so") in
  let log = Filename.temp_file "ocannl_runtime_power_" ".log" in
  let cleanup () =
    List.iter (fun p -> try Sys.remove p with Sys_error _ -> ()) [ src; dll; log ]
  in
  Fun.protect ~finally:cleanup (fun () ->
      let out = open_out_bin src in
      output_string out shims;
      output_string out helper_source;
      output_string out probes;
      close_out out;
      let command =
        Printf.sprintf "%s -O0 %s -o %s %s -lm >%s 2>&1" (Cc.compiler_command ())
          (Lazy.force Cc.kernel_link_flags) (Filename.quote dll) (Filename.quote src)
          (Filename.quote log)
      in
      if Sys.command command <> 0 then (
        let ic = open_in_bin log in
        prerr_string (really_input_string ic (in_channel_length ic));
        close_in ic;
        failwith ("cannot compile the run-time power probe: " ^ command));
      let lib = Dl.dlopen ~filename:dll ~flags:[ RTLD_NOW; RTLD_LOCAL ] in
      Fun.protect
        ~finally:(fun () -> Dl.dlclose ~handle:lib)
        (fun () ->
          let open Ctypes in
          let bind name = Foreign.foreign ~from:lib name (double @-> double @-> returning double) in
          let f32_pow = bind "probe_f32" and f64_pow = bind "probe_f64" in
          let raw = bind "probe_raw_f32" in
          p_all "the emulated vendor powf loses every negative base under an integral exponent"
            cases ~f:(fun (x, n) -> Float.is_nan (raw x (Float.of_int n)));
          let matches ~narrow ~tol pow =
           fun (x, n) ->
            let got = pow x (Float.of_int n) in
            let want = narrow (Float.pow x (Float.of_int n)) in
            let ok = Float.abs (got -. want) <= tol *. Float.abs want in
            if not ok then
              Printf.eprintf "(not part of the golden) %h^%d: got %h want %h\n" x n got want;
            ok
          in
          p_all "f32: negative bases under integral exponents match the integer power" cases
            ~f:(matches ~narrow:f32 ~tol:0x1p-18 f32_pow);
          p_all "f64: negative bases under integral exponents match the integer power" cases
            ~f:(matches ~narrow:Fun.id ~tol:0x1p-40 f64_pow);
          let mirrored pow (x, n) =
            let n = Float.of_int n in
            let neg = pow x n and pos = pow (-.x) n in
            bits neg = bits (if Float.rem n 2. = 0. then pos else -.pos)
          in
          p_all "f32: a negative base's power is its mirror's, negated exactly for odd exponents"
            cases ~f:(mirrored f32_pow);
          p_all "f64: a negative base's power is its mirror's, negated exactly for odd exponents"
            cases ~f:(mirrored f64_pow);
          (* Exponents around the integer-conversion guard: the largest odd integer each precision
             holds, and an even one past it. *)
          let parity =
            [
              (f32_pow, 16777215., -1.);
              (f32_pow, 33554432., 1.);
              (f64_pow, 9007199254740991., -1.);
              (f64_pow, 18014398509481984., 1.);
            ]
          in
          p_all "(-1)^n takes its sign from n's parity at the edge of the exact integers" parity
            ~f:(fun (pow, n, want) -> bits (pow (-1.) n) = bits want);
          (* The sign is read bitwise: -0 is a negative base. *)
          let zeros = [ (3., -0.); (2., 0.); (-1., Float.neg_infinity); (-2., Float.infinity) ] in
          p_all "negative zero's integral powers carry C pow's signs" zeros ~f:(fun (n, want) ->
              bits (f32_pow (-0.) n) = bits want && bits (f64_pow (-0.) n) = bits want);
          let zero_exponent = [ 0.; -0.; Float.infinity; Float.neg_infinity; Float.nan ] in
          p_all "a zero exponent gives 1 for every base, as in C pow" zero_exponent ~f:(fun x ->
              List.for_all
                (fun (pow, y) -> bits (pow x y) = bits 1.)
                [ (f32_pow, 0.); (f32_pow, -0.); (f64_pow, 0.); (f64_pow, -0.) ]);
          let fractional = [ (-2., 0.5); (-1.5, -1.5); (-0.75, 2.5); (-2., Float.nan) ] in
          p_all "negative bases under fractional or NaN exponents stay NaN" fractional
            ~f:(fun (x, y) -> Float.is_nan (f32_pow x y) && Float.is_nan (f64_pow x y))))

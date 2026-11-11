(* gh-ocannl-2137: a C expression's type does not determine its evaluation precision. For a = b = 1
   + 2^-10 and c = -(1 + 2^-9), round_half(a*b+c) = 2^-20, but round_half(round_half(a*b)+c) = 0.
   All inputs and both answers are exact.

   Probe excess precision independently, using (1 + 2^-11) - 1, then check the product witness under
   that evaluation method. GCC need not define __FLT16_EVAL_METHOD__, so its absence is diagnostic
   rather than a skip. The real cc kernel must match the two-rounding oracle under either method:
   its FLOAT_TO_HALF cast at each operator already removes excess precision. Contraction is disabled
   so this tests excess precision alone. *)
open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Cc = Context.Cc_backend
module LL = Ir.Low_level
module Ops = Ir.Ops

let () = Unix.putenv "OCANNL_BACKEND" "cc"
let () = Unix.putenv "OCANNL_CC_BACKEND_FP_CONTRACT" "off"
let () = Unix.putenv "OCANNL_CC_BACKEND_FAST_MATH" "false"

let source =
  {|
int half_eval_macro(void) {
#ifdef __FLT16_EVAL_METHOD__
  return __FLT16_EVAL_METHOD__;
#else
  return -1;
#endif
}
int float_eval_macro(void) { return __FLT_EVAL_METHOD__; }
int half_eval_probe(void) {
#ifdef __FLT16_MAX__
  volatile _Float16 a = 1, b = 0x1p-11;
  _Float16 v = (a + b) - a;
  return v == 0 ? 0 : v == b ? 1 : 2;
#else
  return -1;
#endif
}
double raw_product(double x, double y, double z) {
#ifdef __FLT16_MAX__
  _Float16 a = x, b = y, c = z;
  return (_Float16)(a * b + c);
#else
  return 0;
#endif
}
|}

let with_probe f =
  let src = Stdlib.Filename.temp_file "ocannl_half_eval_" ".c" in
  let dll = Stdlib.Filename.temp_file "ocannl_half_eval_" (if Sys.win32 then ".dll" else ".so") in
  let log = Stdlib.Filename.temp_file "ocannl_half_eval_" ".log" in
  Exn.protect
    ~finally:(fun () -> List.iter [ src; dll; log ] ~f:Stdlib.Sys.remove)
    ~f:(fun () ->
      Stdio.Out_channel.write_all src ~data:source;
      let command =
        Printf.sprintf "%s %s %s -o %s %s >%s 2>&1" (Cc.compiler_command ())
          (Cc.kernel_compiler_flags ()) (Stdlib.Filename.quote src) (Stdlib.Filename.quote dll)
          (Lazy.force Cc.kernel_link_flags) (Stdlib.Filename.quote log)
      in
      if Stdlib.Sys.command command <> 0 then
        failwith ("half evaluation probe failed: " ^ Stdio.In_channel.read_all log);
      let lib = Dl.dlopen ~filename:dll ~flags:[ RTLD_NOW; RTLD_LOCAL ] in
      Exn.protect ~finally:(fun () -> Dl.dlclose ~handle:lib) ~f:(fun () -> f lib))

let a = [| 1. +. (2. **. -10.); -.(1. +. (2. **. -10.)) |]
let b = Array.create ~len:2 a.(0)
let c = [| -.(1. +. (2. **. -9.)); 1. +. (2. **. -9.) |]
let once = [| 2. **. -20.; -.(2. **. -20.) |]
let twice = [| 0.; 0. |]
let half x = Ops.round_to_prec Ops.half x

let run () =
  let name = "hxp_collapsed" in
  let node = Ll_test.node_factory ~prec:Ops.half ~first_id:31000 ~dims:[| 2 |] () in
  let x = node (name ^ "_a") and y = node (name ^ "_b") and z = node (name ^ "_c") in
  let out = node (name ^ "_out") in
  List.iter [ x; y; z; out ] ~f:Ll_test.materialize;
  let s = Ll_test.sym () in
  let i = [| Ll_test.iter s |] in
  let at v = (v, Ops.half) in
  let product = LL.Binop (Ops.Mul, at (Ll_test.get x i), at (Ll_test.get y i)) in
  let sum v = LL.Binop (Ops.Add, at v, at (Ll_test.get z i)) in
  let raw = Ll_test.loop_n s 2 (Ll_test.set out i (sum product)) in
  let o = Ll_test.optimize_scoped ~name ~raw raw in
  List.hd_exn (Ll_test.execute ~name o ~seed:[ (x, a); (y, b); (z, c) ] ~read:[ out ])

let () =
  case "cc half excess precision" @@ fun () ->
  Test_utils.with_policy
    (fun p -> { p with Ir.Numerics.narrow_compute_f32 = false; fp16_arithmetic = Fp16_narrow })
    (fun () ->
      p_all "the witness inputs are exactly representable in half"
        (Array.to_list (Array.concat [ a; b; c ]))
        ~f:(fun x -> Float.equal (half x) x);
      p_all "the witness distinguishes one rounding from two" [ 0; 1 ] ~f:(fun i ->
          Float.equal (half ((a.(i) *. b.(i)) +. c.(i))) once.(i)
          && Float.equal (half (half (a.(i) *. b.(i)) +. c.(i))) twice.(i)
          && not (Float.equal once.(i) twice.(i)));
      with_probe (fun lib ->
          let open Ctypes in
          let get name = (Foreign.foreign ~from:lib name (void @-> returning int)) () in
          let method_ = get "half_eval_probe" in
          Stdio.eprintf
            "cc half evaluation: probe=%d __FLT_EVAL_METHOD__=%d __FLT16_EVAL_METHOD__=%d flags=%s \
             (not part of the golden)\n"
            method_ (get "float_eval_macro") (get "half_eval_macro") (Cc.kernel_compiler_flags ());
          p "the host evaluation probe reports absent, narrow or excess precision"
            (List.mem [ -1; 0; 1 ] method_ ~equal:Int.equal);
          let raw =
            lazy
              (let call =
                 Foreign.foreign ~from:lib "raw_product"
                   (double @-> double @-> double @-> returning double)
               in
               Array.init 2 ~f:(fun i -> call a.(i) b.(i) c.(i)))
          in
          let narrow =
            Verdict.leg ~when_:(method_ = 0) ~on:"host evaluates half operators at half precision"
              ()
          in
          let excess =
            Verdict.leg ~when_:(method_ = 1)
              ~on:"host evaluates half expressions with excess precision" ()
          in
          in_leg ~leg:narrow "native evaluation rounds the raw product before adding" (fun label ->
              p label (Array.equal Float.equal (Lazy.force raw) twice));
          in_leg ~leg:excess "excess evaluation rounds the raw expression only at its result"
            (fun label -> p label (Array.equal Float.equal (Lazy.force raw) once)));
      let collapsed = run () in
      p "cc's operator casts round even inside a collapsed narrow expression"
        (Array.equal Float.equal collapsed twice))

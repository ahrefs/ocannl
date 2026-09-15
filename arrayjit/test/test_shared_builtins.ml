(* Exercise the shipped host stubs against separately compiled kernel definitions, including a
   different fp16 ABI. No inlining: on ELF these calls must bind inside the kernel library. *)
open Verdict.Claims
module B = Context.Builtins_cc
module Cc = Context.Cc_backend

let probe =
  {|
int probe_native(void) { return HAS_NATIVE_FLOAT16; }
int probe_draw(int word) {
    uint4x32_t x = {{ (uint32_t)word, 0, 0, 0 }};
    return HALF_TO_UINT16(uint4x32_to_half_uniform(x));
}
int probe_half(double x) { return single_to_half((float)x); }
double probe_widen(int x) { return half_to_single((uint16_t)x); }
int probe_fp8(double x) { return double_to_fp8(x); }
int probe_fp8_single(double x) { return single_to_fp8((float)x); }
double probe_fp8_widen(int x) { return fp8_to_single((uint8_t)x); }
|}

let same_float x y =
  (Float.is_nan x && Float.is_nan y) || Int64.bits_of_float x = Int64.bits_of_float y

let run mode flags =
  let src = Filename.temp_file "ocannl_shared_builtins_" ".c" in
  let dll = Filename.temp_file "ocannl_shared_builtins_" (if Sys.win32 then ".dll" else ".so") in
  let log = Filename.temp_file "ocannl_shared_builtins_" ".log" in
  let cleanup () =
    List.iter (fun p -> try Sys.remove p with Sys_error _ -> ()) [ src; dll; log ]
  in
  Fun.protect ~finally:cleanup (fun () ->
      let out = open_out_bin src in
      output_string out B.source;
      output_string out probe;
      close_out out;
      let command =
        Printf.sprintf "%s -O0 -fno-inline %s %s -o %s %s -lm >%s 2>&1" (Cc.compiler_command ())
          flags (Lazy.force Cc.kernel_link_flags) (Filename.quote dll) (Filename.quote src)
          (Filename.quote log)
      in
      let code = Sys.command command in
      if code <> 0 then (
        let ic = open_in_bin log in
        prerr_string (really_input_string ic (in_channel_length ic));
        close_in ic;
        failwith ("cannot compile shared builtin probe: " ^ command));
      let lib = Dl.dlopen ~filename:dll ~flags:[ RTLD_NOW; RTLD_LOCAL ] in
      Fun.protect
        ~finally:(fun () -> Dl.dlclose ~handle:lib)
        (fun () ->
          let open Ctypes in
          let bind name typ = Foreign.foreign ~from:lib name typ in
          let native = bind "probe_native" (void @-> returning int) () in
          Printf.eprintf "shared builtins: backend=cc mode=%s native_half=%d compiler=%s\n%!" mode
            native (Cc.compiler_command ());
          let draw = bind "probe_draw" (int @-> returning int) in
          let half = bind "probe_half" (double @-> returning int) in
          let widen = bind "probe_widen" (int @-> returning double) in
          let fp8 = bind "probe_fp8" (double @-> returning int) in
          let fp8_single = bind "probe_fp8_single" (double @-> returning int) in
          let fp8_widen = bind "probe_fp8_widen" (int @-> returning double) in
          let label s = mode ^ ": " ^ s in
          if mode = "emulated" then p (label "the compiler selected emulated half") (native = 0);
          let words = [ 0; 0x10000000; 0x40000000; 0x55555555; 0x7fffffff; -1; -1073741824 ] in
          p_all (label "scalar half draws match the shipped host wrapper") words ~f:(fun w ->
              draw w = Ir.Ops.uint4x32_to_half_uniform [| w; 0; 0; 0 |]);
          p
            (label "a nonzero half draw preserves its bits across the C ABI")
            (draw 0x40000000 = 0x3400);
          let half_codes = List.init 65536 Fun.id in
          p_all (label "every half encoding widens as on the host") half_codes ~f:(fun x ->
              same_float (widen x) (Ir.Ops.half_to_single x));
          let values =
            [
              0.;
              -0.;
              0.25;
              -2.;
              1.00048828125;
              1.00146484375;
              65504.;
              Float.infinity;
              Float.neg_infinity;
              Float.nan;
              -.Float.nan;
            ]
          in
          p_all (label "half narrowing preserves finite values, ties and special values") values
            ~f:(fun x ->
              let got = half x and want = Ir.Ops.single_to_half x in
              if Float.is_nan x then got land 0x7c00 = 0x7c00 && got land 0x03ff <> 0
              else got = want);
          (* The rounding interval just above half of the smallest half subnormal (gh-ocannl-981):
             the emulated narrowing used to flush all of it. Expectations are an independent bit
             oracle over f32 patterns, not a second reading of either converter: the exact midpoint
             0x1p-25 ties down to signed zero, while its next f32 neighbour, 0x1.8p-25, the last
             pattern below 2^-24 and 2^-24 itself all round to the smallest subnormal. *)
          let f32_of_bits b = Int32.float_of_bits (Int32.of_int b) in
          let underflow =
            [
              (0x33000000, 0x0000);
              (0x33000001, 0x0001);
              (0x33400000, 0x0001);
              (0x337fffff, 0x0001);
              (0x33800000, 0x0001);
            ]
          in
          let signed =
            Array.of_list
              (List.concat_map
                 (fun (bits, h) -> [ (bits, h); (bits lor 0x80000000, h lor 0x8000) ])
                 underflow)
          in
          let want = Array.map snd signed in
          p_all2
            (label "half narrowing rounds the smallest-subnormal midpoint interval, both signs")
            (Array.map (fun (bits, _) -> half (f32_of_bits bits)) signed)
            want ~f:Int.equal;
          p_all2
            (label "the shipped host stub narrows that interval the same way")
            (Array.map (fun (bits, _) -> Ir.Ops.single_to_half (f32_of_bits bits)) signed)
            want ~f:Int.equal;
          let fp8_codes = List.init 256 Fun.id in
          p_all (label "every fp8 encoding widens as on the host") fp8_codes ~f:(fun x ->
              same_float (fp8_widen x) (Ir.Ops.fp8_to_single x));
          let positive_nan = Int64.float_of_bits 0x7ff8000000000000L in
          let negative_nan = Int64.float_of_bits 0xfff8000000000000L in
          p
            (label "both fp8 codecs preserve positive and negative NaN signs")
            (fp8 positive_nan = 0x7f
            && fp8 negative_nan = 0xff
            && fp8_single positive_nan = 0x7f
            && fp8_single negative_nan = 0xff);
          let edges =
            values
            @ [
                1.125;
                1.375;
                Float.pred 1.125;
                Float.succ 1.125;
                Float.ldexp 1. (-17);
                Float.ldexp 1. (-16);
                61440.;
                -61440.;
              ]
          in
          p_all (label "fp8 f64 narrowing preserves ties, underflow, overflow and NaN sign") edges
            ~f:(fun x -> fp8 x = Ir.Ops.double_to_fp8 x);
          p_all (label "fp8 f32 narrowing agrees with the shipped host codec") edges ~f:(fun x ->
              fp8_single x = Ir.Ops.single_to_fp8 x)))

let () =
  run "default" "";
  run "emulated" "-U__FLT16_MAX__"

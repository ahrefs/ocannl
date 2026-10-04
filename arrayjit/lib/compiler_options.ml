(* Pure compiler-option construction shared with the optional GPU backends.

   Keeping this below [arrayjit.ir] lets tests exercise the exact lists handed to an RTC compiler
   without linking its optional library or requiring a device. *)

(* Clang processes floating-point switches left-to-right, so every override here has to stay AFTER
   the umbrella flag or [-ffast-math] silently re-enables what it disables.

   [-fhonor-infinities] re-enables the one piece of IEEE behavior this codebase relies on as a VALUE
   rather than as a test: [C_syntax] deliberately emits [(-INFINITY)] for the neutral element of a
   [Max] accumulation and for [Nn_blocks.default_mask_fill], so a masked softmax's sentinel has to
   survive the subtraction and the [exp]. Under bare [-ffast-math] ([-ffinite-math-only]) that only
   ever held by accident of which optimization the compiler happened to pick -- adding
   [-fno-associative-math] for gh-ocannl-735 changed the pick, and HIP's causally masked half
   softmax started returning [exp(0)] and [exp(1)] where the mask demands exact zeros. NaN stays
   unhonored: it is only ever TESTED for here, and that has its own documented shape (a range
   compare of a runtime value). *)
let clang_fast_math_options ~reassociate =
  [ "-ffast-math" ]
  @ (if reassociate then [] else [ "-fno-associative-math" ])
  @ [ "-fhonor-infinities" ]

(* gh-ocannl-1182: on gfx1102, HIPRTC 9.0 (ROCm runtime 7.1.52801) emitted a half subtraction while
   a D16 high-half load into the same VGPR was outstanding. Replaying the emitted attention-softmax
   kernel returned exp(+1)/2, including in masked lanes; forcing load waits to zero made all three
   executed controls correct. This preserves arithmetic precision but conservatively reduces load
   overlap. Other architectures have not been measured. Keep unsupported-option failures visible
   through the normal HIPRTC error path; falling back to the corrupting compilation would be
   unsafe. *)
let hip_load_wait_options ~target_archs =
  if
    List.exists
      (fun arch -> String.equal (List.hd (String.split_on_char ':' arch)) "gfx1102")
      target_archs
  then [ "-mllvm"; "-amdgpu-waitcnt-load-forcezero" ]
  else []

let hiprtc ~target_archs ~hip_include_options ~rocwmma_include_options ~uses_rocwmma ~with_debug =
  hip_include_options
  @ (if uses_rocwmma then rocwmma_include_options @ [ "-std=c++17" ] else [])
  (* These are compiler options rather than kernel-body pragmas so they also govern the bf16 and f16
     operators while the HIP headers are parsed. *)
  @ clang_fast_math_options ~reassociate:false
  @ hip_load_wait_options ~target_archs
  @ if with_debug then [ "-g" ] else []

(* nvrtc's opt-IN for floating-point reassociation. nvrtc accepts it (13.3 answers a
   [--fassociative-math=false] with "does not accept any argument", so its parser knows the name and
   it is a bare flag), but NVIDIA's nvrtc option reference does not list it and there is no negative
   spelling to pair with it. It exists here only so [nvrtc] below can be checked never to emit it:
   the CUDA half of the reduction-order policy is a MEMBERSHIP claim, not an ordering one. *)
let nvrtc_reassociation_opt_in = "--fassociative-math"

(* nvrtc (CUDA). The left-to-right override discipline [clang_fast_math_options] documents does NOT
   apply here, because nvrtc has no umbrella-plus-overrides shape to defend against:
   [--use_fast_math] is a fixed expansion of four independent switches ([--ftz=true --prec-div=false
   --prec-sqrt=false --fmad=true]) and none of them concerns reassociation. The order is still
   pinned -- it is the state a compile is reproducible against -- but what this function actually
   guarantees is the absence of [nvrtc_reassociation_opt_in].

   Measured on the project's CUDA box for gh-ocannl-784 (nvrtc 13.3, driver 13030, GeForce RTX 5070
   Ti, sm_120) by compiling a 128-term float reduction through nvrtc and EXECUTING it: under
   [--use_fast_math], and also under [--use_fast_math] plus [--extra-device-vectorization], plus
   [--dopt=on --ptxas-options=-O3], and plus [--fmad=false], all three of the spellings that broke
   HIP -- a counted loop, a runtime-bound loop, and 128 repeated statements -- returned the
   bit-exact strictly-sequential value, as did an [(a+b)-a] cancellation probe. The same reduction
   compiled by host gcc at [-O3 -ffast-math] does reassociate and [-fno-associative-math] restores
   it, so the probe can see reassociation when it happens. Every clang-shaped guard
   ([-fno-associative-math], [-ffp-contract=off], [-fno-unsafe-math-optimizations], ...) is rejected
   outright by nvrtc as an unrecognized option.

   So CUDA's reduction-order safety is a DOCUMENTED BOUNDARY rather than a flag: nvrtc's fast math
   does not reassociate, nothing in its documented option list could re-guard it if a future nvrtc
   started to, and the one lever that names reassociation is an undocumented opt-in this list must
   never contain. A [reduction_forms] red on CUDA after a toolkit upgrade should re-run that probe
   before anything else: it is tools/nvrtc_reassoc_probe.c, which links nvrtc and the CUDA driver
   directly and carries its own build line. *)
let nvrtc ~cuda_include_options ~arch_options ~with_device_debug =
  cuda_include_options @ arch_options @ [ "--use_fast_math" ]
  @ if with_device_debug then [ "--device-debug" ] else []

(* Metal's MTLCompileOptions is a mutable Objective-C object rather than a string vector. Keep its
   assembly pure anyway: [metal] returns the ordered property writes and Metal_backend is only the
   interpreter that applies them. The complete sequence is then testable without linking Metal, and
   a compile failure can render the exact state it received.

   The math policy was measured for gh-ocannl-848 on an M4 Max, macOS 26.6.2 / Metal 4, MSL 3.1,
   with benchmarks/runners/ocannl/metal_reassoc_probe.ml. Default, Fast and Relaxed all rewrote
   [(a+b)-a] from the strict result 0 to [b] = 1, while Safe preserved 0. Three 128-term reduction
   spellings remained bit-exactly sequential and the runtime [-INFINITY] mask remained zero under
   every mode. The exact production sequence (Safe followed by Fast functions) cost 0.991x default
   on 262144 threads x 128 fixed-count terms and 1.006x on one thread x 1048576 runtime-bound terms
   (GPU-clock medians, 21 rotated interleaved repeats).

   On macOS 15 and later, use the modern split properties rather than deprecated
   [fastMathEnabled=false]. The legacy spelling also changes [mathFloatingPointFunctions] from Fast
   to Precise; [Math_mode_safe] plus [Math_functions_fast] blocks unsafe arithmetic reassociation
   while retaining the existing fast function family. Both modern writes are explicit so an SDK
   default change cannot move either half. On macOS 14 the split selectors do not exist, so the
   caller requests the legacy safe fallback explicitly. *)
type metal_math_api = Modern_split | Legacy

let metal_math_api ~selector_available =
  if selector_available "setMathMode:" && selector_available "setMathFloatingPointFunctions:" then
    Modern_split
  else Legacy

type metal_option =
  | Language_version_3_1
  | Language_version_3_2
  | Fast_math_enabled of bool
  | Math_mode_safe
  | Math_functions_fast
  | Enable_logging

let equal_metal_option a b =
  match (a, b) with
  | Language_version_3_1, Language_version_3_1
  | Language_version_3_2, Language_version_3_2
  | Math_mode_safe, Math_mode_safe
  | Math_functions_fast, Math_functions_fast
  | Enable_logging, Enable_logging ->
      true
  | Fast_math_enabled a, Fast_math_enabled b -> Bool.equal a b
  | _ -> false

let metal ~routine_logging ~math_api =
  [ (if routine_logging then Language_version_3_2 else Language_version_3_1) ]
  @ (match math_api with
    | Modern_split -> [ Math_mode_safe; Math_functions_fast ]
    | Legacy -> [ Fast_math_enabled false ])
  @ if routine_logging then [ Enable_logging ] else []

let render_metal options =
  String.concat " "
    (List.map
       (function
         | Language_version_3_1 -> "language-version=3.1"
         | Language_version_3_2 -> "language-version=3.2"
         | Fast_math_enabled enabled -> "fast-math-enabled=" ^ Bool.to_string enabled
         | Math_mode_safe -> "math-mode=safe"
         | Math_functions_fast -> "math-functions=fast"
         | Enable_logging -> "enable-logging=true")
       options)

(* One line, for diagnostics that have to travel through a log or an exception message: a compile's
   effective option vector is the state a numeric mismatch is reproducible against, and reading it
   off a failure is what turns "schedule-dependent numeric mismatch" into "optimizer flag". *)
let render options = String.concat " " options

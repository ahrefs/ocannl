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

(* gh-ocannl-1182: on gfx1102, HIPRTC 9.0 (ROCm runtime 7.1.52801) compiles 16-bit values into VGPR
   halves (LLVM's [real-true16] feature) and emitted a half subtraction into the low half while a
   D16 high-half load into the same VGPR was outstanding. Replaying the emitted attention-softmax
   kernel returned exp(+1)/2, including in masked lanes, and a backward masked branch corrupted its
   low half the same way. Without [real-true16] each half value gets a whole VGPR, no high-half
   loads are emitted, and that replay returns zero wrong cells in three launches.

   The first workaround, [-mllvm -amdgpu-waitcnt-forcezero], repaired both replays but silently
   miscompiled unrelated integer and bf16 kernels on the same target: Threefry outputs, fp8 round
   trips and tensorized bf16 matmuls all disagreed with their references (gh-ocannl-1222). Only
   gfx1102 has been measured; other RDNA3 targets keep the compiler's default. Keep
   unsupported-option failures visible through the normal HIPRTC error path; falling back to the
   corrupting compilation would be unsafe. *)
let hip_target_options ~target_archs =
  if
    List.exists
      (fun arch -> String.equal (List.hd (String.split_on_char ':' arch)) "gfx1102")
      target_archs
  then [ "-Xclang"; "-target-feature"; "-Xclang"; "-real-true16" ]
  else []

(* gh-ocannl-1779: the math switches, by the [hip_fast_math] key -- the HIP counterpart of
   [nvrtc_math_options]. On (the default, [performance] and [approximate]), the umbrella with its
   overrides, as above. Off (what [profile=reproducible] sets), no umbrella and the IEEE switches
   spelled out, so a hiprtc default cannot move one. Contraction is the one that matters most:
   [-ffp-contract=fast] is what loses the low part of ROCm's extended-precision f64 [pow] (204-206
   ulps, gh-ocannl-1730). Measured on gfx1151 for gh-ocannl-1779 with [unary_math_special_values]:
   this vector, and this vector without [-ffp-contract=off] (hiprtc's own default), hold f64 [pow]
   to 2 ulps; with [-ffp-contract=fast] in its place, [pow(2^-1000, 1)] is 204 ulps off again and
   f32 functions leave their stated ulps. hipcc's driver defaults to fast-honor-pragmas, so the
   default is not one to rely on: [-ffp-contract=off] is the counterpart of the [reproducible]
   payload's [cc_backend_fp_contract=off] and of nvrtc's [--fmad=false]. [-fno-associative-math] is
   clang's default without the umbrella; it stays spelled so the gh-ocannl-735 reduction-order guard
   is a member of both vectors. The off state is "clang's IEEE semantics, contraction off", not
   strict arithmetic in every respect: explicit intrinsics the codegen asks for ([__builtin_fma],
   the [Builtins_hip] wrappers, which are IEEE under either state) stay what they are. *)
let hip_math_options ~fast_math =
  if fast_math then clang_fast_math_options ~reassociate:false
  else
    [
      "-ffp-contract=off";
      "-fno-associative-math";
      "-fhip-fp32-correctly-rounded-divide-sqrt";
      "-fno-gpu-flush-denormals-to-zero";
      "-fno-gpu-approx-transcendentals";
    ]

(* gh-ocannl-1457: whether a clang-family option vector (cc's kernel flags, hiprtc's) pins
   contraction off. Clang applies floating-point switches left to right and [-ffast-math] implies
   [-ffp-contract=fast], so the last of the two decides; a vector naming neither leaves contraction
   to the toolchain's default, which is never a pin. *)
let clang_contraction_off options =
  List.fold_left
    (fun off option ->
      match option with
      | "-ffp-contract=off" -> true
      | "-ffast-math" | "-ffp-contract=fast" | "-ffp-contract=on"
      | "-ffp-contract=fast-honor-pragmas" ->
          false
      | _ -> off)
    false options

let hiprtc ~target_archs ~hip_include_options ~rocwmma_include_options ~uses_rocwmma ~fast_math
    ~with_debug ?(with_line_info = false) () =
  hip_include_options
  @ (if uses_rocwmma then rocwmma_include_options @ [ "-std=c++17" ] else [])
  (* These are compiler options rather than kernel-body pragmas so they also govern the bf16 and f16
     operators while the HIP headers are parsed. *)
  @ hip_math_options ~fast_math
  @ hip_target_options ~target_archs
  @ if with_debug then [ "-g" ] else if with_line_info then [ "-gline-tables-only" ] else []

(* nvrtc's opt-IN for floating-point reassociation. nvrtc accepts it (13.3 answers a
   [--fassociative-math=false] with "does not accept any argument", so its parser knows the name and
   it is a bare flag), but NVIDIA's nvrtc option reference does not list it and there is no negative
   spelling to pair with it. It exists here only so [nvrtc] below can be checked never to emit it:
   the CUDA half of the reduction-order policy is a MEMBERSHIP claim, not an ordering one. *)
let nvrtc_reassociation_opt_in = "--fassociative-math"

(* nvrtc (CUDA). The left-to-right override discipline [clang_fast_math_options] documents does NOT
   apply here, because nvrtc has no umbrella-plus-overrides shape to defend against:
   [--use_fast_math] is a fixed expansion of four independent switches ([--ftz=true --prec-div=false
   --prec-sqrt=false --fmad=true]) plus a substitution of the f32 library functions by their fast
   intrinsics ([sinf] by [__sinf], [expf] by [__expf], [logf], [exp2f], [log2f], [powf], ...), and
   none of it concerns reassociation. The order is still pinned -- it is the state a compile is
   reproducible against -- but what this function actually guarantees is the absence of
   [nvrtc_reassociation_opt_in].

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
(* gh-ocannl-1710: the math switches, by the [cuda_fast_math] key. On (the default, [performance]
   and [approximate]), the umbrella. Off (what [profile=reproducible] sets), no umbrella, so no
   library-function substitution, and all four of its switches spelled out at nvrtc's IEEE values --
   with contraction off as well, the CUDA counterpart of the [reproducible] payload's
   [cc_backend_fp_contract=off]: with [--fmad=true] nvrtc emits PTX [mul]/[add] without a rounding
   modifier, which the PTX ISA lets the target's JIT fuse into an [fma], so contraction would be a
   per-architecture decision. Every switch is explicit, so a toolkit's default cannot move one. The
   off state is "nvrtc's IEEE defaults, contraction off", not strict arithmetic in every respect:
   explicit intrinsics the codegen asks for ([fmaf], the half [h*] functions) stay what they are. *)
let nvrtc_math_options ~fast_math =
  if fast_math then [ "--use_fast_math" ]
  else [ "--ftz=false"; "--prec-div=true"; "--prec-sqrt=true"; "--fmad=false" ]

(* gh-ocannl-1457: whether an nvrtc option vector pins contraction off. [--use_fast_math] implies
   [--fmad=true], so only the spelled-out [--fmad=false] counts. *)
let nvrtc_contraction_off options = List.mem "--fmad=false" options

let nvrtc ~cuda_include_options ~arch_options ~fast_math ~with_device_debug
    ?(with_line_info = false) () =
  cuda_include_options @ arch_options @ nvrtc_math_options ~fast_math
  @ (if with_device_debug then [ "--device-debug" ] else [])
  @ if with_line_info then [ "-lineinfo" ] else []

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

(* gh-ocannl-1457: the Metal math regime as schedule-cache identity. The two option sequences are
   two arithmetic regimes (Legacy's [fastMathEnabled=false] also makes the math functions precise),
   picked by what the running OS's compile options accept, so a crown from one is not evidence for
   the other. Only the legacy regime adds a component, so every key crowned on macOS 15 or later
   stays valid; derived from the same builder the compile uses. *)
let metal_regime_tag math_api =
  match math_api with
  | Modern_split -> None
  | Legacy -> Some ("metal-math:" ^ render_metal (metal ~routine_logging:false ~math_api))

(* One line, for diagnostics that have to travel through a log or an exception message: a compile's
   effective option vector is the state a numeric mismatch is reproducible against, and reading it
   off a failure is what turns "schedule-dependent numeric mismatch" into "optimizer flag". *)
let render options = String.concat " " options

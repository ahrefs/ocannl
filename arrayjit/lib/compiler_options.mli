(* Pure compiler-option construction shared with the optional GPU backends.

   Keeping this below [arrayjit.ir] lets tests exercise the exact lists handed to an RTC compiler
   without linking its optional library or requiring a device. The per-toolchain math-policy
   rationale and measurements live with each builder in the implementation. *)

val hip_target_options : target_archs:string list -> string list
(** The target-feature workaround for the observed gfx1102 half-register hazard. Used by both the
    compiler and its schedule-cache regime. An empty list selects no workaround. A mixed set
    containing gfx1102 is covered because HIP compilation is backend-wide and its artifact can link
    on any device. *)

val hip_math_options : fast_math:bool -> string list
(** hiprtc's math switches under the [hip_fast_math] key (gh-ocannl-1779): the clang fast-math
    umbrella with its left-to-right overrides ([-fno-associative-math], [-fhonor-infinities]) when
    on; when off, no umbrella and clang's IEEE switches spelled out, contraction off
    ([-ffp-contract=off -fno-associative-math -fhip-fp32-correctly-rounded-divide-sqrt
      -fno-gpu-flush-denormals-to-zero -fno-gpu-approx-transcendentals]). Used by both the compiler
    and the HIP schedule-cache regime. *)

val clang_contraction_off : string list -> bool
(** Whether a clang-family option vector (cc's kernel flags split into words, {!hip_math_options})
    pins contraction off (gh-ocannl-1457): the last of [-ffp-contract=off] and the switches that
    turn contraction on ([-ffast-math], [-ffp-contract=fast|on|fast-honor-pragmas]) is
    [-ffp-contract=off]. A vector naming none of them leaves contraction to the toolchain's default,
    which is not a pin. [C_syntax_config.compiler_contraction_off] reads it. *)

val hiprtc :
  target_archs:string list ->
  hip_include_options:string list ->
  rocwmma_include_options:string list ->
  uses_rocwmma:bool ->
  fast_math:bool ->
  with_debug:bool ->
  ?with_line_info:bool ->
  unit ->
  string list
(** The hiprtc (HIP) option vector: includes, then {!hip_math_options}, the target's half-register
    workaround, then debug -- or, without it, [?with_line_info]'s line tables ([-gline-tables-only],
    default off): source-line attribution in the code object, which the debug-artifact runs carry so
    an ISA census can read the code that ran (gh-ocannl-1695). Line tables do not change the code
    generated. *)

val nvrtc_reassociation_opt_in : string
(** nvrtc's opt-IN for floating-point reassociation. It exists here only so {!nvrtc} can be checked
    never to emit it: the CUDA half of the reduction-order policy is a MEMBERSHIP claim, not an
    ordering one. *)

val nvrtc_math_options : fast_math:bool -> string list
(** nvrtc's math switches under the [cuda_fast_math] key (gh-ocannl-1710): [--use_fast_math] when
    on; when off, no umbrella (so no fast-intrinsic substitution of the f32 library functions) and
    [--ftz=false --prec-div=true --prec-sqrt=true --fmad=false], every switch explicit. Used by both
    the compiler and the CUDA schedule-cache regime. *)

val nvrtc_contraction_off : string list -> bool
(** Whether an nvrtc option vector pins contraction off: it holds [--fmad=false], which
    {!nvrtc_math_options} spells out when [cuda_fast_math] is off (gh-ocannl-1457). *)

val nvrtc :
  cuda_include_options:string list ->
  arch_options:string list ->
  fast_math:bool ->
  with_device_debug:bool ->
  ?with_line_info:bool ->
  unit ->
  string list
(** The nvrtc (CUDA) option vector: includes, arch, {!nvrtc_math_options}, debug, then
    [?with_line_info]'s [-lineinfo] (default off; source-line attribution in the PTX, which the
    debug-artifact runs carry for an ISA census, gh-ocannl-1695 -- it does not change the code
    generated). What this guarantees under either policy is the absence of
    {!nvrtc_reassociation_opt_in}: [--use_fast_math] is a fixed expansion of four switches plus a
    library-function substitution, none of which concerns reassociation (measured for gh-ocannl-784;
    see the implementation). *)

(** Which Metal math-policy API the host macOS offers: the modern split properties (macOS 15+), or
    the deprecated [fastMathEnabled] fallback (macOS 14). *)
type metal_math_api = Modern_split | Legacy

val metal_math_api : selector_available:(string -> bool) -> metal_math_api
(** Select the modern split policy only when both required Objective-C selectors are available. The
    callback keeps the capability seam injectable without making this pure module depend on the
    Metal runtime. *)

(** Ordered property writes to apply to an [MTLCompileOptions] object; [Metal_backend] is only the
    interpreter that applies them, so the complete sequence is testable without linking Metal. *)
type metal_option =
  | Language_version_3_1
  | Language_version_3_2
  | Fast_math_enabled of bool
  | Math_mode_safe
  | Math_functions_fast
  | Enable_logging

val equal_metal_option : metal_option -> metal_option -> bool
val metal : routine_logging:bool -> math_api:metal_math_api -> metal_option list
val render_metal : metal_option list -> string

val metal_regime_tag : metal_math_api -> string option
(** The Metal backend's schedule-cache regime component (gh-ocannl-1457): [None] for [Modern_split],
    the default on macOS 15 and later, so its keys stay valid; for [Legacy], the rendered {!metal}
    sequence, a different arithmetic regime ([fastMathEnabled=false] also makes the math functions
    precise). *)

val render : string list -> string
(** One line, for diagnostics that have to travel through a log or an exception message: a compile's
    effective option vector is the state a numeric mismatch is reproducible against, and reading it
    off a failure is what turns "schedule-dependent numeric mismatch" into "optimizer flag". *)

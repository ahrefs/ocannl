open Base

(** CUDA's math policy (gh-ocannl-1710): whether nvrtc compiles with [--use_fast_math]
    ([Compiler_options.nvrtc_math_options]) and f32 [Tanh_approx] renders the [__tanhf] intrinsic.
    On by default; [profile=reproducible] turns it off. The one reader of the key, so the compile,
    the rendering and the CUDA schedule-cache regime cannot disagree. *)
let cuda_fast_math () = Utils.get_global_flag ~default:true ~arg_name:"cuda_fast_math"

(** HIP's math policy (gh-ocannl-1779): whether hiprtc compiles with [-ffast-math]
    ([Compiler_options.hip_math_options]). On by default; [profile=reproducible] turns it off. The
    one reader of the key, so the compile and the HIP schedule-cache regime cannot disagree. Unlike
    CUDA's, it changes no rendering: [Builtins_hip]'s wrappers are IEEE under either state. *)
let hip_fast_math () = Utils.get_global_flag ~default:true ~arg_name:"hip_fast_math"

(** Diagnostic extra hiprtc compiler options (gh-ocannl-2344): whitespace-separated flags appended
    at the end of every hiprtc compile. Empty by default. The one reader of the key, so the compile,
    [Hip.compiler_contraction_off] and the HIP schedule-cache regime cannot disagree. *)
let hip_extra_compile_options () =
  Compiler_options.parse_extra_options
    (Utils.get_global_arg ~default:"" ~arg_name:"hip_extra_compile_options")

module Cuda = struct
  let header_prelude =
    {|#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

/* Define math constants that would normally come from <math.h> */
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#ifndef NAN
#define NAN __int_as_float(0x7fffffff)
#endif|}

  let name = "CUDA"
  let backend_facts = Backend_intf.cuda_facts
  let error_prefix = "Cuda_backend"
  let bf16_typ = "__nv_bfloat16"
  let fp8_typ = "__nv_fp8_e5m2"
  let bf16_arithmetic_intrinsics = true

  (* gh-ocannl-1695: at the shipping floor PTX target, ptxas (13.4, sm_120) if-converts the value
     selects in [branchless_select]'s census. Native sm_120 PTX instead branches on load-gating
     selects (gh-ocannl-1822); the no-branch claim is floor-target-only. CUDA keeps short-circuiting
     [?:] and [cuda_facts.select_blend_policy = No_blend]: blending would also change the inline
     fan-in guard's load accounting, without performance evidence here. *)
  let bf16_relu = ("__hmax_nan(__ushort_as_bfloat16((unsigned short)0x0000U), ", ")")

  (* [tanh.approx.f32], a hardware approximation, under fast math only: with [cuda_fast_math] off
     the f32 tanh is the library's, as it is on cc, HIP and Metal ([precise::tanh]). *)
  let tanh_approx () = if cuda_fast_math () then "__tanhf" else "tanhf"

  let compiler_contraction_off () =
    Compiler_options.nvrtc_contraction_off
      (Compiler_options.nvrtc_math_options ~fast_math:(cuda_fast_math ()))

  (* gh-ocannl-1779: CUDA's half functions ([hexp], [hsin], [hsqrt], [hrcp], [htanh_approx], ...)
     are hardware approximations ([ex2.approx], [sin.approx], [rcp.approx], [tanh.approx] in
     cuda_fp16.hpp) whatever nvrtc's switches say, so under [cuda_fast_math] off they render as the
     precise f32 function of the widened operand, narrowed once: the precise f32 result rounded to
     half, not a correctly rounded half function. *)
  let precise_half_functions () = not (cuda_fast_math ())

  (* Under [--use_fast_math] nvrtc lowers these to [__sinf]/[__cosf], which keep NaN for +-inf and
     NaN; their absolute-error contract and the large-argument loss are accepted (gh-ocannl-1336).
     With [cuda_fast_math] off they are the precise library functions (gh-ocannl-1710). *)
  let sin_f32 = "sinf"
  let cos_f32 = "cosf"
  let sqrt_f32 = "sqrtf"
  let fmod_f32 = "fmodf"
  let fmod_f64 = "fmod"
  let recip_sqrt_f64 = ("(1.0 / sqrt(", "))")

  (* nvrtc's [--use_fast_math] touches only f32 division ([--prec-div=false]): f64 [/] is IEEE. *)
  let div_f64 = None
  let fp8_from_float_syntax = ("(__nv_fp8_e5m2)(", ")")
  let conditional_includes = [ ("nvcuda::wmma", "#include <mma.h>") ]
  let prelude_headers () = Utils.included_headers header_prelude

  let conditional_headers () =
    List.concat_map conditional_includes ~f:(fun (_, directive) -> Utils.included_headers directive)

  (* gh-ocannl-707, gh-ocannl-1338: the names nvrtc fails to compile as a kernel name after the
     prelude, measured by [tools/vendor_idents.exe]. *)
  let vendor_routine_idents = Vendor_idents_cuda.names
  let workgroup_start_alignment = 1

  (* gh-ocannl-2426: CUDA's [CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES] reports static shared memory to 1
     B. Measured on rog-nv-linux (RTX 5070 Ti, run 20261122T165804Z-1226811): the 9x21, 9x11 and
     9x17 f16 fused products (378, 198, 306 B beside 4096 B of aligned tiles) reported 4474, 4294
     and 4402 B, and two-array byte/half probes (17+16, 17+17, 33+33 B) reported 33, 34 and 66 B. *)
  let workgroup_reporting_granule = 1

  let convert_precision ~typ_of_prec ~from ~to_ =
    match (from, to_) with
    | Ops.Double_prec _, Ops.Double_prec _
    | Single_prec _, Single_prec _
    | Half_prec _, Half_prec _
    | Byte_prec _, Byte_prec _
    | Uint16_prec _, Uint16_prec _
    | Int32_prec _, Int32_prec _
    | Int64_prec _, Int64_prec _
    | Uint4x32_prec _, Uint4x32_prec _
    | Bfloat16_prec _, Bfloat16_prec _
    | Fp8_prec _, Fp8_prec _
    | Void_prec, Void_prec ->
        ("", "")
    | Double_prec _, Half_prec _ -> ("__double2half(", ")")
    | Single_prec _, Half_prec _ -> ("__float2half(", ")")
    | Byte_prec _, Half_prec _ -> ("__ushort2half_rn((unsigned short int)", ")")
    | Double_prec _, Uint4x32_prec _ -> ("double_to_uint4x32(", ")")
    | Single_prec _, Uint4x32_prec _ -> ("single_to_uint4x32(", ")")
    (* A plain conversion out of a block is refused, as on every backend (gh-ocannl-1785): these
       arms used to read lane 0's bits. *)
    | Uint4x32_prec _, _ -> Ops.no_plain_uint4x32_conversion ~to_
    | Byte_prec _, Uint4x32_prec _ -> ("byte_to_uint4x32(", ")")
    | Uint16_prec _, Uint4x32_prec _ -> ("uint16_to_uint4x32(", ")")
    | Bfloat16_prec _, Uint4x32_prec _ -> ("bfloat16_to_uint4x32(", ")")
    | Half_prec _, Uint4x32_prec _ -> ("half_to_uint4x32(", ")")
    | Fp8_prec _, Uint4x32_prec _ -> ("fp8_to_uint4x32(", ")")
    | ( Fp8_prec _,
        (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Uint32_prec _ | Int64_prec _ | Uint64_prec _)
      ) ->
        (* __nv_fp8_e5m2 has no integer conversion operators: a C-style cast to an integer type
           would resolve through the saturating [operator unsigned char] (wrong for negative
           values). Convert via float, like the CC backend. *)
        ("(" ^ typ_of_prec to_ ^ ")((float)(", "))")
    (* The integer counter conversions MUST call the builtins, which spread the bits across all four
       uint4x32 lanes (golden-ratio / MMIX / rotation mixing). The raw struct literal below only
       fills lane 0, leaving lanes 1-3 zero; with the 2-round light threefry used for parameter init
       that produces near-identical outputs for consecutive counters (periodicity), so random inits
       diverge from CC/Metal. [Ops.index_prec] is signed [int32] (or [int64] under [large_models]),
       so the signed arms are the conversion hit by every PRNG init loop, e.g. centered [uniform1]
       parameter initialization (task-04f97340). *)
    | Int32_prec _, Uint4x32_prec _ -> ("int32_to_uint4x32(", ")")
    | Int64_prec _, Uint4x32_prec _ -> ("int64_to_uint4x32(", ")")
    | Uint32_prec _, Uint4x32_prec _ -> ("uint32_to_uint4x32(", ")")
    | Uint64_prec _, Uint4x32_prec _ -> ("uint64_to_uint4x32(", ")")
    | _, Uint4x32_prec _ -> ("{(unsigned int)(", "), 0, 0, 0}")
    | _ -> ("(" ^ typ_of_prec to_ ^ ")(", ")")
end

module Hip = struct
  (* hiprtc ships the HIP headers and needs no declaration for device-side printf. *)
  let header_prelude =
    {|#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
/* hip_fp8.h ships with ROCm >= 6.2 (hiprtc is clang, so __has_include is available); guarding
   keeps non-fp8 kernels compiling on older SDKs, where fp8 kernels still fail with unknown type
   __hip_fp8_e5m2. */
#if __has_include(<hip/hip_fp8.h>)
#include <hip/hip_fp8.h>
#endif

/* Define math constants that would normally come from <math.h> */
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif
#ifndef NAN
#define NAN __builtin_nanf("")
#endif|}

  let name = "HIP"
  let backend_facts = Backend_intf.hip_facts
  let error_prefix = "Hip_backend"
  let bf16_typ = "__hip_bfloat16"
  let fp8_typ = "__hip_fp8_e5m2"
  let bf16_arithmetic_intrinsics = false
  let bf16_relu = ("__float2bfloat16(fmaxf(0.0f, __bfloat162float(", ")))")
  let tanh_approx () = "tanhf"

  let compiler_contraction_off () =
    Compiler_options.clang_contraction_off
      (Compiler_options.hip_math_options ~fast_math:(hip_fast_math ())
      @ hip_extra_compile_options ())

  (* hip_fp16.h's half functions are ROCm's OCML f16 library functions ([__ocml_exp_f16], ...), and
     [htanh_approx] is [Builtins_hip]'s f32 route already. *)
  let precise_half_functions () = false

  (* [Builtins_hip]'s NaN-preserving wrappers: under [-ffast-math], ROCm's [sinf]/[cosf] return a
     finite value for +-inf (gh-ocannl-1336). *)
  let sin_f32 = "ocannl_sinf"
  let cos_f32 = "ocannl_cosf"

  (* [Builtins_hip]'s wrappers (gh-ocannl-1684): under [-ffast-math], [sqrtf] reads an f32 subnormal
     as zero, and [fmodf] returns a signed zero where C returns NaN. *)
  let sqrt_f32 = "ocannl_sqrtf"
  let fmod_f32 = "ocannl_fmodf"

  (* gh-ocannl-1730: [fmod] the same at f64 ([Builtins_hip.ocannl_fmod]); and under [-ffast-math],
     [1.0 / sqrt(x)] at f64 becomes a reciprocal-root estimate refined by Newton steps, NaN for +-0
     and +inf, where ROCm's own [rsqrt] is C's [1 / sqrt] there and correctly rounded elsewhere. *)
  let fmod_f64 = "ocannl_fmod"
  let recip_sqrt_f64 = ("rsqrt(", ")")

  (* gh-ocannl-1782: the former default's reciprocal estimate made f64 [/] NaN for [x / 0], [x /
     inf] and overflowing quotients; [Builtins_hip.ocannl_div_f64] supplies the IEEE quotient. Since
     gh-ocannl-1909 both policies disable reciprocal math and approximate functions. The explicit
     f64 expansion remains correct under either vector, independently of that policy. *)
  let div_f64 = Some "ocannl_div_f64"

  (* THE float-to-fp8 narrowing HIP emits — the operator bridges, which compute in float and narrow
     the result, and [convert_precision]'s conversions. It is ALWAYS the guarded helper, never the
     platform's own cast: ROCm's software float-to-e5m2 conversion shifts out of range for
     magnitudes around 4e-25 to 3.3e-24 and returns codes as large as 2^-14 where the answer is a
     signed zero (gh-ocannl-647, filed upstream as https://github.com/ROCm/rocm-systems/issues/10591
     — that fix landing is what retires this helper). The guard pre-rounds only that range, which
     rounds to zero anyway, so it is exact everywhere, and this backend's fp8 output matches CUDA,
     cc, Metal and the host codec on every input.

     One funnel rather than four call sites, because the first version of this guarded
     [convert_precision] alone and left fp8 ARITHMETIC results narrowing through bare casts — an fp8
     [**.] whose f32 result lands in ROCm's broken window still produced a spurious value (Codex P2
     on PR #372). A fifth narrowing site cannot repeat that without going through here.

     Which spelling narrows a value of precision [from]: the helpers are per-source-width on
     purpose, since a double handed to the float helper would narrow at the call and round twice
     (gh-ocannl-648), which the platform's own cast does not do. *)
  let fp8_from_prec_fn = function
    | Ops.Double_prec _ -> "ocannl_double_to_fp8_uniform"
    | _ -> "ocannl_single_to_fp8_uniform"

  let fp8_from_float_syntax = (fp8_from_prec_fn Ops.single ^ "((float)(", "))")
  let conditional_includes = [ ("rocwmma::", "#include <rocwmma/rocwmma.hpp>") ]
  let prelude_headers () = Utils.included_headers header_prelude

  let conditional_headers () =
    List.concat_map conditional_includes ~f:(fun (_, directive) -> Utils.included_headers directive)

  (* gh-ocannl-707, gh-ocannl-1338: the names hiprtc fails to compile as a kernel name after the
     prelude, measured by [tools/vendor_idents.exe]. *)
  let vendor_routine_idents = Vendor_idents_hip.names

  (* gh-ocannl-2465: LLVM AMDGPU can align LDS arrays to 16 B. Padding is between arrays: 17 B and
     15 B retain a 47 B layout on both gfx1102 and gfx1151. *)
  let workgroup_start_alignment = 16

  (* gh-ocannl-2426: HIP's [HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES] reports static shared memory to 1
     B on both discrete and unified memory. Measured on tuf-amd-linux (gfx1102 discrete, run
     20261122T165816Z-762875) and minix-amd-linux (gfx1151 unified, run 20261122T165826Z-1286312),
     byte-identical on both: the 9x21, 9x11 and 9x17 f16 fused products (378, 198, 306 B beside 4096
     B of aligned tiles) reported 4474, 4294 and 4402 B, and two-array byte/half probes reported odd
     totals (47, 49, 81 B). *)
  let workgroup_reporting_granule = 1

  let convert_precision ~typ_of_prec ~from ~to_ =
    match (from, to_) with
    | Ops.Double_prec _, Ops.Double_prec _
    | Single_prec _, Single_prec _
    | Half_prec _, Half_prec _
    | Byte_prec _, Byte_prec _
    | Uint16_prec _, Uint16_prec _
    | Int32_prec _, Int32_prec _
    | Int64_prec _, Int64_prec _
    | Uint4x32_prec _, Uint4x32_prec _
    | Bfloat16_prec _, Bfloat16_prec _
    | Fp8_prec _, Fp8_prec _
    | Void_prec, Void_prec ->
        ("", "")
    (* One rounding, not two through float (gh-ocannl-2075): hip_fp16.h has no [__double2half]. *)
    | Double_prec _, Half_prec _ -> ("ocannl_double_to_half(", ")")
    | Single_prec _, Half_prec _ -> ("__float2half(", ")")
    | Byte_prec _, Half_prec _ -> ("__ushort2half_rn((unsigned short int)", ")")
    | Double_prec _, Uint4x32_prec _ -> ("double_to_uint4x32(", ")")
    | Single_prec _, Uint4x32_prec _ -> ("single_to_uint4x32(", ")")
    (* A plain conversion out of a block is refused, as on every backend (gh-ocannl-1785): this arm
       used to read lane 0's bits. *)
    | Uint4x32_prec _, _ -> Ops.no_plain_uint4x32_conversion ~to_
    | Byte_prec _, Uint4x32_prec _ -> ("byte_to_uint4x32(", ")")
    | Uint16_prec _, Uint4x32_prec _ -> ("uint16_to_uint4x32(", ")")
    | Bfloat16_prec _, Uint4x32_prec _ -> ("bfloat16_to_uint4x32(", ")")
    | Half_prec _, Uint4x32_prec _ -> ("half_to_uint4x32(", ")")
    | Fp8_prec _, Uint4x32_prec _ -> ("fp8_to_uint4x32(", ")")
    (* The integer counter conversions MUST call the builtins, which spread the bits across all four
       uint4x32 lanes (golden-ratio / MMIX / rotation mixing). The raw struct literal below only
       fills lane 0, leaving lanes 1-3 zero; with the 2-round light threefry used for parameter init
       that produces near-identical outputs for consecutive counters (periodicity), so random inits
       diverge from CC/Metal. [Ops.index_prec] is signed [int32] (or [int64] under [large_models]),
       so the signed arms are the conversion hit by every PRNG init loop, e.g. centered [uniform1]
       parameter initialization (task-04f97340). *)
    | Int32_prec _, Uint4x32_prec _ -> ("int32_to_uint4x32(", ")")
    | Int64_prec _, Uint4x32_prec _ -> ("int64_to_uint4x32(", ")")
    | Uint32_prec _, Uint4x32_prec _ -> ("uint32_to_uint4x32(", ")")
    | Uint64_prec _, Uint4x32_prec _ -> ("uint64_to_uint4x32(", ")")
    | _, Uint4x32_prec _ -> ("{(unsigned int)(", "), 0, 0, 0}")
    (* [__hip_bfloat16] has constructors from float, double, short, unsigned short, ... and [__half]
       has several conversion operators, so C-style casts between them are ambiguous (observed with
       ROCm 7.1 hiprtc); route through float explicitly. Same precaution for the half <-> fp8
       pairs. *)
    | Half_prec _, Bfloat16_prec _ -> ("__float2bfloat16(__half2float(", "))")
    | Bfloat16_prec _, Half_prec _ -> ("__float2half(__bfloat162float(", "))")
    | Double_prec _, Bfloat16_prec _ -> ("ocannl_double_to_bfloat16(", ")")
    | (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Uint32_prec _), Bfloat16_prec _ ->
        ("ocannl_double_to_bfloat16((double)(", "))")
    | Int64_prec _, Bfloat16_prec _ -> ("__ushort_as_bfloat16(ocannl_int64_to_bfloat16_bits(", "))")
    | Uint64_prec _, Bfloat16_prec _ ->
        ("__ushort_as_bfloat16(ocannl_uint64_to_bfloat16_bits(", "))")
    | _, Bfloat16_prec _ -> ("__float2bfloat16((float)(", "))")
    (* Through the same funnel as the operator bridges. A helper and not a ternary wrapped around
       the operand: a [convert_precision] pair brackets ONE occurrence of its expression, and a
       ternary would name it twice. Float helper arguments state their conversion explicitly,
       including integer and bool sources; double sources retain their double helper. Match fp8
       before the general bf16 cast so the exact bf16-to-float widening still reaches the guard. *)
    | _, Fp8_prec _ -> (
        let fn = fp8_from_prec_fn from in
        match from with
        | Ops.Half_prec _ -> (fn ^ "(__half2float(", "))")
        | Ops.Bfloat16_prec _ -> (fn ^ "(__bfloat162float(", "))")
        | Ops.Double_prec _ -> (fn ^ "(", ")")
        | _ -> (fn ^ "((float)(", "))"))
    | Bfloat16_prec _, _ -> ("(" ^ typ_of_prec to_ ^ ")(__bfloat162float(", "))")
    | Fp8_prec _, Half_prec _ -> ("__float2half((float)(", "))")
    | ( Fp8_prec _,
        (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Uint32_prec _ | Int64_prec _ | Uint64_prec _)
      ) ->
        (* __hip_fp8_e5m2's integer conversion operators saturate (wrong for negative values into
           unsigned types) and overlap enough to make direct casts ambiguity-prone; convert via
           float, like the CC backend. *)
        ("(" ^ typ_of_prec to_ ^ ")((float)(", "))")
    | _ -> ("(" ^ typ_of_prec to_ ^ ")(", ")")
end

(** Shared CUDA/HIP scalar semantics. Hardware MMA, asynchronous copies and conversions with
    different vendor overload contracts stay in the backend; the common operator table lives here so
    fixes to narrow arithmetic reach both consumers. *)
module Make (Dialect : sig
  val name : string
  val backend_facts : Backend_intf.backend_facts
  val error_prefix : string
  val bf16_typ : string
  val fp8_typ : string
  val builtins : (string * string * string list) list
  val extra_blacklist : string list
  val bf16_arithmetic_intrinsics : bool
  val bf16_relu : string * string
  val tanh_approx : unit -> string

  (* The compile's contraction pin, derived from its option builder
     ([C_syntax_config.compiler_contraction_off], gh-ocannl-1457). *)
  val compiler_contraction_off : unit -> bool

  (* Whether the half unary functions ([Exp] ... [Tanh_approx]) render as the f32 rendering of the
     widened operand, narrowed once, rather than as the vendor's half functions. *)
  val precise_half_functions : unit -> bool

  (* The f32 (and, widened, bfloat16) [sin]/[cos] functions. *)
  val sin_f32 : string
  val cos_f32 : string

  (* The f32 (and, widened, bfloat16 -- and for [fmod], half) [sqrt] -- also under [Recip_sqrt] --
     and [fmod] functions. *)
  val sqrt_f32 : string
  val fmod_f32 : string

  (* The f64 [fmod] function, and the f64 [Recip_sqrt] rendering as a prefix and a suffix around the
     operand. *)
  val fmod_f64 : string
  val recip_sqrt_f64 : string * string

  (* The f64 [Div] (and [Recip], as [1.0 / x]) function, where a plain [/] is not IEEE. *)
  val div_f64 : string option
  val fp8_from_float_syntax : string * string
  val vendor_routine_idents : string list
  val workgroup_start_alignment : int
  val workgroup_reporting_granule : int

  val convert_precision :
    typ_of_prec:(Ops.prec -> string) -> from:Ops.prec -> to_:Ops.prec -> string * string
end) (Input : sig
  val procs : Low_level.t array
end) =
struct
  include C_syntax.Pure_C_config (struct
    let procs = Input.procs
    let full_printf_support = C_syntax.printf_support_unless_uniform ()
  end)

  let ident_blacklist =
    ident_blacklist @ C_syntax.cpp_keywords
    @ C_syntax.builtin_idents Dialect.builtins
    @ [ "threadIdx"; "blockIdx"; "blockDim"; "gridDim"; "warpSize" ]
    @ Dialect.extra_blacklist

  (* Qualified: [include C_syntax.Pure_C_config] above brings its own empty [vendor_routine_idents]
     into scope, which an unqualified reference here would pick up. *)
  let vendor_routine_idents = Dialect.vendor_routine_idents
  let compiler_contraction_off = Dialect.compiler_contraction_off

  (* gh-ocannl-1338: the vendor headers' global types and macros ([half2], [float4], [CUDARTAPI],
     [HIPRT_INF_FP16]) cannot be shadowed by, or expand into, a node's identifier. *)
  let node_ident_prefix = Dialect.backend_facts.Backend_intf.node_ident_prefix

  let fp8_from_float doc =
    let fp8_prefix, fp8_suffix = Dialect.fp8_from_float_syntax in
    PPrint.(group (string fp8_prefix ^^ doc ^^ string fp8_suffix))

  let main_kernel_prefix = "extern \"C\" __global__"
  let builtin_param_guards = Some C_syntax.Cxx_guards
  let result_type_probes = C_syntax.Cxx_probes "__device__ __forceinline__"

  (* An all-Serial kernel launches 1x1x1, so no single-thread guard is needed; annotated kernels
     need every thread (axis-types proposal §4). *)
  let kernel_prep_line = ""

  (* Use native types for loop indices and arguments instead of stdint.h types. Signed index
     arithmetic (docs/design/signed-index-precision.md). *)
  let loop_index_type = if Utils.settings.large_models then "long long " else "int "
  let arg_int_prefix = if Utils.settings.large_models then "const long long " else "const int "

  (* Hardware axis bindings (docs/design/axis-types-for-loops.md §5); the binding site casts the
     unsigned register to the signed [loop_index_type] (values fit by device limits and the per-node
     numel contract). *)
  let hardware_index ~kind ~slot =
    let base = match kind with `Grid -> "blockIdx" | `Workgroup -> "threadIdx" in
    match slot with
    | 0 -> Some (base ^ ".x")
    | 1 -> Some (base ^ ".y")
    | 2 -> Some (base ^ ".z")
    | _ -> None

  let barrier_syntax = Some "__syncthreads();"
  let shared_decl_prefix = Some "__shared__ "
  let workgroup_start_alignment = Dialect.workgroup_start_alignment
  let workgroup_reporting_granule = Dialect.workgroup_reporting_granule
  let restrict_keyword = Some "__restrict__"

  (* Warp-shuffle rendering of [Workgroup_reduce] accumulation loops (gh-ocannl-462):
     [ocannl_shfl_xor] wraps [__shfl_xor] with an explicit width of 32 (builtins_hip.ml). RDNA GPUs
     have 32-wide wavefronts natively; on wave64 (CDNA/GCN) devices the explicit width makes the
     shuffles reduce the same 32-lane groups, so 32 is correct everywhere. *)
  let warp_size = 32

  (* No vectorization pragmas in device code — SIMD-style gains on GPU come from memory
     transactions: eligible [Vectorized] loops render 128-bit packed loads/stores through the
     [__align__(16)] pack structs (gh-ocannl-463), and everything else falls back to plain serial
     loops. Local arrays live in registers/local memory; no alignment attribute needed (packed
     accesses require device-resident nodes). *)
  let vectorize_pragma = []
  let aligned_local_attr = None

  (* 16-byte packs that only move data: no vector arithmetic, so no register tile and no count. *)
  let vector_file = { Vector_file.bytes = 16; registers = 0 }
  let vector_style = `Packed_struct

  let typ_of_prec = function
    | Ops.Byte_prec _ -> "unsigned char"
    | Ops.Uint16_prec _ -> "unsigned short"
    | Ops.Int32_prec _ -> "int"
    | Ops.Int64_prec _ -> "long long"
    | Ops.Uint4x32_prec _ -> "uint4x32_t"
    | Ops.Half_prec _ -> "__half"
    | Ops.Bfloat16_prec _ -> Dialect.bf16_typ
    | Ops.Fp8_prec _ -> Dialect.fp8_typ
    | Ops.Single_prec _ -> "float"
    | Ops.Double_prec _ -> "double"
    | Ops.Void_prec -> "void"
    | Ops.Uint32_prec _ -> "unsigned int"
    | Ops.Uint64_prec _ -> "unsigned long long"

  let convert_precision = Dialect.convert_precision ~typ_of_prec

  let vec_typ_of_prec ~length prec =
    match (prec, length) with
    | Ops.Single_prec _, 4 -> "float4_t"
    | Ops.Double_prec _, 2 -> "double2_t"
    | Ops.Int32_prec _, 4 -> "int32x4_t"
    | Ops.Int64_prec _, 2 -> "int64x2_t"
    | Ops.Byte_prec _, 16 -> "int8x16_t"
    (* Fp8 needs [__hip_fp8_e5m2] elements: [Set_from_vec] assigns them to the fp8 array cells
       without a cast, and [__hip_fp8_e5m2] has no assignment from integer types. *)
    | Ops.Fp8_prec _, 16 -> "fp8x16_t"
    | Ops.Uint16_prec _, 8 -> "uint16x8_t"
    | Ops.Uint32_prec _, 4 -> "uint32x4_t"
    | Ops.Uint64_prec _, 2 -> "uint64x2_t"
    (* Like fp8, bfloat16 needs [__hip_bfloat16] elements rather than raw [unsigned short] bits:
       [Set_from_vec] assigns them to the array cells without a cast. Mirrors the CUDA backend. *)
    | Ops.Bfloat16_prec _, 8 -> "bfloat16x8_t"
    | Ops.Half_prec _, 8 -> "half8_t"
    | _, 1 -> typ_of_prec prec
    | _ -> invalid_arg (Dialect.error_prefix ^ ".vec_typ_of_prec: invalid combination")

  let rec binop_syntax prec v =
    (* The match stays exhaustive over (op, prec) -- that is what catches a newly added operator
       here -- but arms whose spelling is plain C delegate to {!C_syntax.default_binop_syntax}
       rather than restating the token. *)
    let open PPrint in
    let f op_str v1 v2 =
      group (parens (v1 ^^ string (" " ^ op_str) ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))))
    in
    let func fn v1 v2 =
      group (string fn ^^ parens (v1 ^^ comma ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))))
    in
    let widened_powf ~widen ~narrow v1 v2 =
      let w v = string widen ^^ parens v in
      group (string narrow ^^ parens (func "ocannl_pow_f32" (w v1) (w v2)))
    in
    (* The predicate consumes the condition once. The ternary still guards the payload and
       determines the result type. *)
    let sat01_gate ~zero predicate v2 =
      group
        (parens
           (group predicate
           ^^ nest 2 (break 1 ^^ string "? " ^^ v2 ^^ break 1 ^^ string (": " ^ zero))))
    in
    let sat01_compare ?(convert = "") lower upper ~zero v1 =
      let predicate =
        string "ocannl_sat01_open"
        ^^ parens (string convert ^^ parens v1 ^^ string (", " ^ lower ^ ", " ^ upper))
      in
      sat01_gate ~zero predicate
    in
    match (v, prec) with
    | Ops.Arg1, _ -> invalid_arg (Dialect.error_prefix ^ ".binop_syntax: Arg1 is not an operator")
    | Arg2, _ -> invalid_arg (Dialect.error_prefix ^ ".binop_syntax: Arg2 is not an operator")
    | _, Ops.Void_prec -> invalid_arg (Dialect.error_prefix ^ ".binop_syntax: Void precision")
    (* The RNG ops call the same builtins under the same precision contract on every C-family
       backend, so they render through the shared helper. Must precede the fp8 bridge: the Threefry
       errors should name the actual target precision, and the lane conversion's builtin already
       yields the target precision. *)
    | ((Threefry4x32_crypto | Threefry4x32_light | Uint4x32_to_prec_uniform_lane) as op), _ ->
        C_syntax.rng_binop_syntax ~backend:Dialect.name ~call:func prec op
    | _, Fp8_prec _ ->
        (* __hip_fp8_e5m2 defines no arithmetic operators, and its implicit conversion operators
           (float, double, int, char, ... in amd_hip_fp8.h) make the built-in operators ambiguous,
           so bridge fp8 math through float, mirroring the CC backend's fp8 handling. *)
        fun v1 v2 ->
          let fl v = string "(float)" ^^ parens v in
          fp8_from_float (binop_syntax Ops.single v (fl v1) (fl v2))
    | Add, Half_prec _ -> func "__hadd"
    | Sub, Half_prec _ -> func "__hsub"
    | Mul, Half_prec _ -> func "__hmul"
    | Div, Half_prec _ -> func "__hdiv"
    | Add, Bfloat16_prec _ when Dialect.bf16_arithmetic_intrinsics -> func "__hadd"
    | Sub, Bfloat16_prec _ when Dialect.bf16_arithmetic_intrinsics -> func "__hsub"
    | Mul, Bfloat16_prec _ when Dialect.bf16_arithmetic_intrinsics -> func "__hmul"
    | Div, Bfloat16_prec _ when Dialect.bf16_arithmetic_intrinsics -> func "__hdiv"
    | Add, _ -> f "+"
    | Sub, _ -> f "-"
    | Mul, _ -> f "*"
    | Div, Double_prec _ when Option.is_some Dialect.div_f64 ->
        func (Option.value_exn Dialect.div_f64)
    | Div, _ -> f "/"
    (* Through [Builtins_cc.cuda_like_power_builtins] (gh-ocannl-1240): fast math lowers the vendor
       [pow] to [exp2(y * log2(x))], NaN for a negative base even under an integral exponent known
       only at run time. Known integer exponents never reach here: [C_syntax.integer_power_doc]
       takes them. *)
    | ToPowOf, Double_prec _ -> func "ocannl_pow_f64"
    | ToPowOf, Single_prec _ -> func "ocannl_pow_f32"
    (* Neither vendor has a narrow-float pow (and both vendors' [hexp2] is unary, gh-ocannl-1198):
       widen both operands, take the f32 power above -- its domain, NaN for a negative base under a
       fractional exponent included -- and round back once, as cc's codegen does. *)
    | ToPowOf, Half_prec _ -> widened_powf ~widen:"__half2float" ~narrow:"__float2half"
    | ToPowOf, Bfloat16_prec _ -> widened_powf ~widen:"__bfloat162float" ~narrow:"__float2bfloat16"
    | ToPowOf, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Int64_prec _ | Uint4x32_prec _) ->
        invalid_arg
          (Dialect.error_prefix ^ ".binop_syntax: ToPowOf not supported for integer precisions")
    | Relu_gate, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Int64_prec _) ->
        fun v1 v2 ->
          group
            (parens
               (group (parens (v1 ^^ string " > 0"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "0")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "0"))))
    | Relu_gate, Bfloat16_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group (parens (string "__bfloat162float(" ^^ v1 ^^ string ") > 0.0f"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "__float2bfloat16(0.0f)")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "__float2bfloat16(0.0f)"))))
    | Relu_gate, Half_prec _ ->
        (* HIP's clang has no [0.0h] half literal; compare via [__hgt] against a bitcast zero. *)
        fun v1 v2 ->
          group
            (parens
               (group
                  (parens
                     (string "__hgt(" ^^ v1 ^^ string ", __ushort_as_half((unsigned short)0x0000U))"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                    ^^ string "__ushort_as_half((unsigned short)0x0000U)")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                       ^^ string "__ushort_as_half((unsigned short)0x0000U)"))))
    | Relu_gate, Single_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group (parens (v1 ^^ string " > 0.0f"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "0.0f")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "0.0f"))))
    | Relu_gate, Double_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group (parens (v1 ^^ string " > 0.0"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "0.0")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "0.0"))))
    | Relu_gate, Uint4x32_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group (parens (v1 ^^ string " > 0"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "0")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "0"))))
    | Satur01_gate, Byte_prec _ ->
        sat01_compare ~convert:"(float)" "0.0f" "1.0f" ~zero:"(unsigned char)0"
    | Satur01_gate, Half_prec _ ->
        fun v1 ->
          sat01_gate ~zero:"__ushort_as_half((unsigned short)0x0000U)"
            (string "ocannl_sat01_open_half" ^^ parens v1)
    | Satur01_gate, Single_prec _ -> sat01_compare "0.0f" "1.0f" ~zero:"0.0f"
    | Satur01_gate, Double_prec _ -> sat01_compare "0.0" "1.0" ~zero:"0.0"
    | Satur01_gate, Uint16_prec _ ->
        sat01_compare ~convert:"(float)" "0.0f" "1.0f" ~zero:"(unsigned short)0"
    | Satur01_gate, Int32_prec _ -> sat01_compare ~convert:"(float)" "0.0f" "1.0f" ~zero:"0"
    | Satur01_gate, Int64_prec _ -> sat01_compare ~convert:"(double)" "0.0" "1.0" ~zero:"0LL"
    | Satur01_gate, Uint4x32_prec _ -> sat01_compare ~convert:"(float)" "0.0f" "1.0f" ~zero:"0u"
    | Satur01_gate, Bfloat16_prec _ ->
        sat01_compare ~convert:"__bfloat162float" "0.0f" "1.0f" ~zero:"__float2bfloat16(0.0f)"
    (* Through the builtins' [ocannl_max] / [ocannl_min] templates, which compare in the operands'
       common type and are typed that (gh-ocannl-1883): the overloaded [max]/[min] resolve for the
       operand pairs they list only, and an integer arm's operand is a read, an integer-promoted
       sum, or a [bool] truth. *)
    | ( Max,
        (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Uint32_prec _ | Int64_prec _ | Uint64_prec _)
      ) ->
        func "ocannl_max"
    | ( Min,
        (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Uint32_prec _ | Int64_prec _ | Uint64_prec _)
      ) ->
        func "ocannl_min"
    | Max, Half_prec _ -> func "__hmax"
    | Max, Double_prec _ -> func "fmax"
    | Max, Single_prec _ -> func "fmaxf"
    | Max, Uint4x32_prec _ -> func "max"
    | Max, Bfloat16_prec _ -> func "__hmax"
    | Min, Half_prec _ -> func "__hmin"
    | Min, Double_prec _ -> func "fmin"
    | Min, Single_prec _ -> func "fminf"
    | Min, Uint4x32_prec _ -> func "min"
    | Min, Bfloat16_prec _ -> func "__hmin"
    | ( Mod,
        (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Uint32_prec _ | Int64_prec _ | Uint64_prec _)
      ) ->
        f "%"
    (* Like the libm calls in [unop_syntax]: [fmod] on bfloat16 operands returns float, which only
       fails once the placement inlines it into a bfloat16 binop (gh-ocannl-549). *)
    | Mod, Bfloat16_prec _ ->
        fun v1 v2 ->
          group
            (string ("__float2bfloat16(" ^ Dialect.fmod_f32 ^ "(__bfloat162float(")
            ^^ v1 ^^ string "), __bfloat162float(" ^^ v2 ^^ string ")))")
    (* A [__half] operand converts to both [float] and [double], so [fmod] on two of them is an
       ambiguous call (gh-ocannl-1684): widen, as [ToPowOf] does. The remainder is exact in half. *)
    | Mod, Half_prec _ ->
        fun v1 v2 ->
          group
            (string ("__float2half(" ^ Dialect.fmod_f32 ^ "(__half2float(")
            ^^ v1 ^^ string "), __half2float(" ^^ v2 ^^ string ")))")
    | Mod, Single_prec _ -> func Dialect.fmod_f32
    | Mod, Double_prec _ -> func Dialect.fmod_f64
    | Mod, _ -> func "fmod"
    (* Comparisons and logical connectives are precision-independent and spelled the same in HIP C++
       as in C, so they render through the shared default -- fp8 already bridged above. The
       constructors stay listed to keep the match exhaustiveness-checked. *)
    | ((Cmplt | Cmple | Cmpne | Cmpeq | Or | And) as op), _ -> C_syntax.default_binop_syntax prec op
    | ToPowOf, (Uint32_prec _ | Uint64_prec _) ->
        invalid_arg
          (Dialect.error_prefix ^ ".binop_syntax: ToPowOf not supported for integer precisions")
    | Relu_gate, Uint32_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group (parens (v1 ^^ string " > 0u"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "0u")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "0u"))))
    | Relu_gate, Uint64_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group (parens (v1 ^^ string " > 0ULL"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "0ULL")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "0ULL"))))
    | Satur01_gate, Uint32_prec _ -> sat01_compare "0u" "1u" ~zero:"0u"
    | Satur01_gate, Uint64_prec _ -> sat01_compare "0ULL" "1ULL" ~zero:"0ULL"

  let rec unop_syntax prec v =
    let open PPrint in
    let f prefix suffix expr = group (string prefix ^^ expr ^^ string suffix) in
    let func fn expr = group (string fn ^^ parens expr) in
    (* A libm call on a bfloat16 operand resolves (the operand converts to float) but *returns
       float*. Assigning that back to a bfloat16 cell is accepted -- __hip_bfloat16's converting
       constructor is implicit -- so it goes unnoticed until the placement that inlines the call
       instead makes the float an operand of a bfloat16 binop, where hiprtc reports "operator '+' is
       ambiguous ('__hip_bfloat16' and 'float')" (gh-ocannl-549). Bridge the result back the way
       [ToPowOf], [Relu], [Recip] and [Satur01] already do, so the emission is bfloat16-typed
       wherever it lands. *)
    let bf16_func fn = f ("__float2bfloat16(" ^ fn ^ "(__bfloat162float(") ")))" in
    match (v, prec) with
    | (Ops.Identity | Ops.Round_prec), _ -> f "" ""
    | Uint4x32_to_prec_uniform1, Ops.Uint4x32_prec _ ->
        invalid_arg
          (Dialect.error_prefix
         ^ ".unop_syntax: Uint4x32_to_prec_uniform1 not supported for Uint4x32")
    (* Heterogeneous op: the argument is uint4x32 whatever the result precision, so it must stay
       ahead of the fp8 float-bridging below; the fp8 builtin returns __hip_fp8_e5m2. *)
    | Uint4x32_to_prec_uniform1, _ -> func ("uint4x32_to_" ^ Ops.prec_string prec ^ "_uniform")
    | _, Ops.Fp8_prec _ ->
        (* __hip_fp8_e5m2 defines no arithmetic operators, and its implicit conversion operators
           (float, double, int, char, ... in amd_hip_fp8.h) make the built-in operators ambiguous,
           so bridge fp8 math through float, mirroring the CC backend's fp8 handling. *)
        fun expr -> fp8_from_float (unop_syntax Ops.single v (string "(float)" ^^ parens expr))
    (* The integer arms are integer-typed (gh-ocannl-1883): [fmax(0.0, v)], [truncf(v)] and [v ==
       0.0 ? 1.0 : 0.0] were float-typed over integral values, which [%] does not take and which
       round a wide operand. Their operand is never float-valued (C_syntax's [integer_op]). *)
    | Relu, p when Ops.is_integer p -> f "ocannl_max(0, " ")"
    | Satur01, p when Ops.is_integer p -> f "ocannl_max(0, ocannl_min(1, " "))"
    | Trunc, p when Ops.is_integer p -> f "(" ")"
    | Not, p when Ops.is_integer p -> f "(" " == 0)"
    | Relu, Ops.Single_prec _ -> f "fmaxf(0.0, " ")"
    | Relu, Ops.Half_prec _ -> f "__hmax_nan(__ushort_as_half((unsigned short)0x0000U), " ")"
    (* Mixing a [__hip_bfloat16] with a literal of another arithmetic type is ambiguous under hiprtc
       for the same reason as [fma] on bfloat16 operands (see [ternop_syntax] below), so [fmax(0.0,
       bf16)] does not compile. Bridge through float like the bf16 binops above, and produce the
       result with [__float2bfloat16]. *)
    | Relu, Ops.Bfloat16_prec _ ->
        let prefix, suffix = Dialect.bf16_relu in
        f prefix suffix
    | Relu, _ -> f "fmax(0.0, " ")"
    | Satur01, Bfloat16_prec _ ->
        f "__float2bfloat16(fmaxf(0.0f, fminf(1.0f, __bfloat162float(" "))))"
    | Satur01, Half_prec _ ->
        f
          "__hmax_nan(__ushort_as_half((unsigned short)0x0000U), \
           __hmin_nan(__ushort_as_half((unsigned short)0x3C00U), "
          "))"
    | Satur01, Single_prec _ -> f "fmaxf(0.0f, fminf(1.0f, " "))"
    | Satur01, _ -> f "fmax(0.0, fmin(1.0, " "))"
    | (Exp | Log | Exp2 | Log2 | Sin | Cos | Sqrt | Recip | Recip_sqrt | Tanh_approx), Half_prec _
      when Dialect.precise_half_functions () ->
        fun expr ->
          f "__float2half(" ")" (unop_syntax Ops.single v (string "__half2float" ^^ parens expr))
    | Exp, Half_prec _ -> func "hexp"
    | Exp, Double_prec _ -> func "exp"
    | Exp, Bfloat16_prec _ -> bf16_func "expf"
    | Exp, _ -> func "expf"
    | Log, Half_prec _ -> func "hlog"
    | Log, Double_prec _ -> func "log"
    | Log, Bfloat16_prec _ -> bf16_func "logf"
    | Log, _ -> func "logf"
    | Exp2, Half_prec _ -> func "hexp2"
    | Exp2, Double_prec _ -> func "exp2"
    | Exp2, Bfloat16_prec _ -> bf16_func "exp2f"
    | Exp2, _ -> func "exp2f"
    | Log2, Half_prec _ -> func "hlog2"
    | Log2, Double_prec _ -> func "log2"
    | Log2, Bfloat16_prec _ -> bf16_func "log2f"
    | Log2, _ -> func "log2f"
    | Sin, Half_prec _ -> func "hsin"
    | Sin, Double_prec _ -> func "sin"
    | Sin, Bfloat16_prec _ -> bf16_func Dialect.sin_f32
    | Sin, _ -> func Dialect.sin_f32
    | Cos, Half_prec _ -> func "hcos"
    | Cos, Double_prec _ -> func "cos"
    | Cos, Bfloat16_prec _ -> bf16_func Dialect.cos_f32
    | Cos, _ -> func Dialect.cos_f32
    | Sqrt, Half_prec _ -> func "hsqrt"
    | Sqrt, Double_prec _ -> func "sqrt"
    | Sqrt, Bfloat16_prec _ -> bf16_func Dialect.sqrt_f32
    | Sqrt, Single_prec _ -> func Dialect.sqrt_f32
    | Sqrt, _ -> func "sqrtf"
    | Recip, Byte_prec _ ->
        invalid_arg
          (Dialect.error_prefix ^ ".unop_syntax: Recip not supported for byte/integer precisions")
    | Recip, Half_prec _ -> func "hrcp"
    | Recip, Single_prec _ -> f "(1.0f / (" "))"
    | Recip, Double_prec _ -> (
        match Dialect.div_f64 with Some div -> f (div ^ "(1.0, ") ")" | None -> f "(1.0 / (" "))")
    (* [1 / bf16] is ambiguous: the int operand can pair with any of the bfloat16 conversions. *)
    | Recip, Bfloat16_prec _ -> f "__float2bfloat16(1.0f / __bfloat162float(" "))"
    | Recip, _ -> f "(1 / (" "))"
    | Recip_sqrt, Byte_prec _ ->
        invalid_arg
          (Dialect.error_prefix
         ^ ".unop_syntax: Recip_sqrt not supported for byte/integer precisions")
    | Recip_sqrt, Half_prec _ -> func "hrsqrt"
    | Recip_sqrt, Double_prec _ -> f (fst Dialect.recip_sqrt_f64) (snd Dialect.recip_sqrt_f64)
    | Recip_sqrt, Single_prec _ -> f ("(1.0f / " ^ Dialect.sqrt_f32 ^ "(") "))"
    | Recip_sqrt, Bfloat16_prec _ ->
        f ("__float2bfloat16(1.0f / " ^ Dialect.sqrt_f32 ^ "(__bfloat162float(") ")))"
    | Recip_sqrt, _ -> f "(1 / sqrtf(" "))"
    | Neg, _ -> f "(-(" "))"
    | Trunc, Double_prec _ -> func "trunc"
    (* [truncf] of a [__half] is a [float]: the value is a half's own, so the bridge is exact, and
       it keeps the result typed at half (gh-ocannl-2117). *)
    | Trunc, Half_prec _ -> f "__float2half(truncf(__half2float(" ")))"
    | Trunc, Bfloat16_prec _ -> bf16_func "truncf"
    | Trunc, _ -> func "truncf"
    (* Every integer precision: [tanh] has float and double overloads only, so an integer operand
       makes the call ambiguous under nvrtc (gh-ocannl-1731). *)
    | Tanh_approx, p when Ops.is_integer p ->
        invalid_arg
          (Dialect.error_prefix
         ^ ".unop_syntax: Tanh_approx not supported for byte/integer precisions")
    | Tanh_approx, Half_prec _ -> func "htanh_approx"
    | Tanh_approx, Single_prec _ -> func (Dialect.tanh_approx ())
    | Tanh_approx, Bfloat16_prec _ -> bf16_func "tanhf"
    | Tanh_approx, _ -> func "tanh"
    (* [bf16 == 0.0] is ambiguous for the same reason as [1 / bf16] above. *)
    | Not, Bfloat16_prec _ -> f "__float2bfloat16(__bfloat162float(" ") == 0.0f ? 1.0f : 0.0f)"
    (* [__half == 0.0] is ambiguous too: the built-in comparison and [__half]'s own [operator==]
       both match (gh-ocannl-1731). *)
    | Not, Half_prec _ -> f "__float2half(__half2float(" ") == 0.0f ? 1.0f : 0.0f)"
    | Not, Single_prec _ -> f "(" " == 0.0f ? 1.0f : 0.0f)"
    | Not, _ -> f "(" " == 0.0 ? 1.0 : 0.0)"

  let vec_unop_syntax prec op v =
    let open PPrint in
    match (op, prec) with
    | Ops.Uint4x32_to_prec_uniform, _ ->
        group (string ("uint4x32_to_" ^ Ops.prec_string prec ^ "_uniform_vec(") ^^ v ^^ rparen)

  let rec ternop_syntax prec v =
    let open PPrint in
    let func fn v1 v2 v3 = group (string fn ^^ parens (separate comma [ v1; v2; v3 ])) in
    match (v, prec) with
    | _, Ops.Fp8_prec _ ->
        (* __hip_fp8_e5m2 defines no arithmetic operators, and its implicit conversion operators
           (float, double, int, char, ... in amd_hip_fp8.h) make the built-in operators ambiguous,
           so bridge fp8 math through float, mirroring the CC backend's fp8 handling. *)
        fun v1 v2 v3 ->
          let fl v = string "(float)" ^^ parens v in
          fp8_from_float (ternop_syntax Ops.single v (fl v1) (fl v2) (fl v3))
    | Ops.Where, _ ->
        (* The whole ternary must be parenthesized, not just the condition: C's [?:] binds looser
           than the surrounding arithmetic, so for an expression like [where(c,a,b) + 1] the
           trailing [+ 1] would otherwise be absorbed into the else-branch, silently dropping it
           from the then-branch (see the CUDA backend, task-04f97340). *)
        fun v1 v2 v3 -> group (parens (parens v1 ^^ string " ? " ^^ v2 ^^ string " : " ^^ v3))
    | FMA, Ops.Half_prec _ -> func "__hfma"
    (* [__hip_bfloat16] has implicit conversion operators to float, __bf16, int, char, ... , so a
       plain [fma] call on bfloat16 operands is ambiguous under hiprtc: its float, double and
       _Float16 overloads (hiprtc_runtime.h) are reached through different conversion operators,
       which makes their conversion sequences indistinguishable. [__hfma] from amd_hip_bf16.h takes
       bfloat16 operands exactly, mirroring [__hmax] / [__hmin] above. *)
    | FMA, Ops.Bfloat16_prec _ -> func "__hfma"
    | FMA, Ops.Single_prec _ -> func "fmaf"
    | FMA, _ -> func "fma"
    | Mul3, _ -> fun v1 v2 v3 -> group (parens (v1 ^^ string " * " ^^ v2 ^^ string " * " ^^ v3))

  let select_blend_policy () =
    (Backend_intf.effective_facts Dialect.backend_facts).select_blend_policy

  (* gh-ocannl-1695: an admitted select evaluates its arms into [auto] locals and then selects with
     the dialect's own spelling. HIP's shipping policy leaves a loaded arm selected by the
     short-circuiting spelling, where AMDGPU can branch around it (gh-ocannl-2427). The locals keep
     each arm's type and the spelling is the dialect's own, so the type, the value and the gate's
     test are the [?:]'s, [__half] / [float] mixtures included. Not at fp8, whose operators bridge
     through [float] and no test exercises a blend of. *)
  let select_blend_syntax prec =
    if Backend_intf.equal_select_blend_policy (select_blend_policy ()) No_blend then None
    else
      match prec with
      | Ops.Half_prec _ | Ops.Bfloat16_prec _ | Ops.Single_prec _ | Ops.Double_prec _ ->
          Some (C_syntax.Blend_over_locals { decl = "auto"; wrap = `Lambda })
      | _ -> None

  let kernel_log_param = Some ("int", "log_id")
  let log_involves_file_management = false

  let pp_log_statement ~log_param_c_expr_doc ~base_message_literal ~args_docs =
    let open PPrint in
    let format_string_literal =
      let res = String.substr_replace_all base_message_literal ~pattern:"\n" ~with_:"$" in
      let res =
        if for_log_trace_tree && String.is_suffix res ~suffix:"$" then
          String.drop_suffix res 1 ^ "\\n"
        else res
      in
      !Utils.captured_log_prefix ^ "%d: " ^ res
    in
    let all_args =
      match log_param_c_expr_doc with
      | Some doc -> doc :: args_docs
      | None -> args_docs (* Should not happen if kernel_log_param is Some *)
    in
    group
      (string "printf("
      ^^ dquotes (string format_string_literal)
      ^^ comma
      ^^ nest 4 (break 1 ^^ separate (comma ^^ break 1) all_args)
      ^^ rparen ^^ semi)
end

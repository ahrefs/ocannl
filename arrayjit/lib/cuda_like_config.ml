open Base

module Cuda = struct
  let name = "CUDA"
  let error_prefix = "Cuda_backend"
  let bf16_typ = "__nv_bfloat16"
  let fp8_typ = "__nv_fp8_e5m2"
  let bf16_arithmetic_intrinsics = true
  let bf16_relu = ("__hmax_nan(__ushort_as_bfloat16((unsigned short)0x0000U), ", ")")
  let tanh_approx = "__tanhf"
  let fp8_from_float_fn = "(__nv_fp8_e5m2)"
  let conditional_includes = [ ("nvcuda::wmma", "#include <mma.h>") ]
end

module Hip = struct
  let name = "HIP"
  let error_prefix = "Hip_backend"
  let bf16_typ = "__hip_bfloat16"
  let fp8_typ = "__hip_fp8_e5m2"
  let bf16_arithmetic_intrinsics = false
  let bf16_relu = ("__float2bfloat16(fmaxf(0.0f, __bfloat162float(", ")))")
  let tanh_approx = "tanhf"

  (* ROCm's subnormal guard is one funnel for conversion and every operator arity. *)
  let fp8_from_prec_fn = function
    | Ops.Double_prec _ -> "ocannl_double_to_fp8_uniform"
    | _ -> "ocannl_single_to_fp8_uniform"

  let fp8_from_float_fn = fp8_from_prec_fn Ops.single
  let conditional_includes = [ ("rocwmma::", "#include <rocwmma/rocwmma.hpp>") ]
end

(** Shared CUDA/HIP scalar semantics. Hardware MMA, asynchronous copies and conversions with
    different vendor overload contracts stay in the backend; the common operator table lives here so
    fixes to narrow arithmetic reach both consumers. *)
module Make (Dialect : sig
  val name : string
  val error_prefix : string
  val bf16_typ : string
  val fp8_typ : string
  val builtins : (string * string * string list) list
  val extra_blacklist : string list
  val bf16_arithmetic_intrinsics : bool
  val bf16_relu : string * string
  val tanh_approx : string
  val fp8_from_float_fn : string
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

  let fp8_from_float doc = PPrint.(group (string Dialect.fp8_from_float_fn ^^ parens doc))
  let main_kernel_prefix = "extern \"C\" __global__"

  (* An all-Serial kernel launches 1x1x1, so no single-thread guard is needed; annotated kernels
     need every thread (axis-types proposal §4). *)
  let kernel_prep_line = ""

  (* Use native types for loop indices and arguments instead of stdint.h types. Signed index
     arithmetic (docs/proposals/signed-index-precision.md). *)
  let loop_index_type = if Utils.settings.large_models then "long long " else "int "
  let arg_int_prefix = if Utils.settings.large_models then "const long long " else "const int "

  (* Hardware axis bindings (docs/proposals/axis-types-for-loops.md §5); the binding site casts the
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
  let vector_bytes = 16
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
    | Div, _ -> f "/"
    | ToPowOf, Double_prec _ -> func "pow"
    | ToPowOf, Single_prec _ -> func "powf"
    | ToPowOf, Half_prec _ ->
        fun v1 v2 ->
          group
            (string "hexp2(hlog2(" ^^ v1 ^^ string "),"
            ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
            ^^ string ")")
    | ToPowOf, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Int64_prec _ | Uint4x32_prec _) ->
        invalid_arg
          (Dialect.error_prefix ^ ".binop_syntax: ToPowOf not supported for integer precisions")
    | ToPowOf, Bfloat16_prec _ ->
        fun v1 v2 ->
          group
            (string "__float2bfloat16(powf(__bfloat162float("
            ^^ v1 ^^ string "), __bfloat162float(" ^^ v2 ^^ string ")))")
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
        fun v1 v2 ->
          group
            (parens
               (group
                  (parens
                     (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                    ^^ string " < 1.0f"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "(unsigned char)0")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "(unsigned char)0"))))
    | Satur01_gate, Half_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group
                  (parens
                     (string "__hgt(" ^^ v1 ^^ comma
                     ^^ string " __ushort_as_half((unsigned short)0x0000U)) && __hlt("
                     ^^ v1 ^^ comma
                     ^^ string " __ushort_as_half((unsigned short)0x3C00U))"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                    ^^ string "__ushort_as_half((unsigned short)0x0000U)")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                       ^^ string "__ushort_as_half((unsigned short)0x0000U)"))))
    | Satur01_gate, Single_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group (parens (v1 ^^ string " > 0.0f && " ^^ v1 ^^ string " < 1.0f"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "0.0f")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "0.0f"))))
    | Satur01_gate, Double_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group (parens (v1 ^^ string " > 0.0 && " ^^ v1 ^^ string " < 1.0"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "0.0")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "0.0"))))
    | Satur01_gate, Uint16_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group
                  (parens
                     (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                    ^^ string " < 1.0f"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "(unsigned short)0")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "(unsigned short)0"))))
    | Satur01_gate, Int32_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group
                  (parens
                     (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                    ^^ string " < 1.0f"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "0")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "0"))))
    | Satur01_gate, Int64_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group
                  (parens
                     (string "(double)" ^^ v1 ^^ string " > 0.0 && (double)" ^^ v1
                    ^^ string " < 1.0"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "0LL")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "0LL"))))
    | Satur01_gate, Uint4x32_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group
                  (parens
                     (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                    ^^ string " < 1.0f"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "0u")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "0u"))))
    | Satur01_gate, Bfloat16_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group
                  (parens
                     (string "__bfloat162float(" ^^ v1
                     ^^ string ") > 0.0f && __bfloat162float("
                     ^^ v1 ^^ string ") < 1.0f"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "__float2bfloat16(0.0f)")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "__float2bfloat16(0.0f)"))))
    | Max, Byte_prec _ -> func "max"
    | Max, Half_prec _ -> func "__hmax"
    | Max, Double_prec _ -> func "fmax"
    | Max, Single_prec _ -> func "fmaxf"
    | Max, Uint16_prec _ -> func "max"
    | Max, Int32_prec _ -> func "max"
    | Max, Int64_prec _ -> func "max"
    | Max, Uint4x32_prec _ -> func "max"
    | Max, Bfloat16_prec _ -> func "__hmax"
    | Min, Byte_prec _ -> func "min"
    | Min, Half_prec _ -> func "__hmin"
    | Min, Double_prec _ -> func "fmin"
    | Min, Single_prec _ -> func "fminf"
    | Min, Uint16_prec _ -> func "min"
    | Min, Int32_prec _ -> func "min"
    | Min, Int64_prec _ -> func "min"
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
            (string "__float2bfloat16(fmodf(__bfloat162float("
            ^^ v1 ^^ string "), __bfloat162float(" ^^ v2 ^^ string ")))")
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
    | Satur01_gate, Uint32_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group (parens (v1 ^^ string " > 0u && " ^^ v1 ^^ string " < 1u"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "0u")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "0u"))))
    | Satur01_gate, Uint64_prec _ ->
        fun v1 v2 ->
          group
            (parens
               (group (parens (v1 ^^ string " > 0ULL && " ^^ v1 ^^ string " < 1ULL"))
               ^^ ifflat
                    (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                   ^^ string "0ULL")
                    (nest 2
                       (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                      ^^ string "0ULL"))))
    | Max, Uint32_prec _ -> func "max"
    | Max, Uint64_prec _ -> func "max"
    | Min, Uint32_prec _ -> func "min"
    | Min, Uint64_prec _ -> func "min"

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
    | Ops.Identity, _ -> f "" ""
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
    | Relu, Ops.Single_prec _ -> f "fmaxf(0.0, " ")"
    | Relu, Ops.Half_prec _ -> f "__hmax_nan(__ushort_as_half((unsigned short)0x0000U), " ")"
    | Relu, Ops.Byte_prec _ -> f "fmax(0, " ")"
    (* Mixing a [__hip_bfloat16] with a literal of another arithmetic type is ambiguous under hiprtc
       for the same reason as [fma] on bfloat16 operands (see [ternop_syntax] below), so [fmax(0.0,
       bf16)] does not compile. Bridge through float like the bf16 binops above, and produce the
       result with [__float2bfloat16]. *)
    | Relu, Ops.Bfloat16_prec _ ->
        let prefix, suffix = Dialect.bf16_relu in
        f prefix suffix
    | Relu, _ -> f "fmax(0.0, " ")"
    | Satur01, Byte_prec _ -> f "fmax(0, fmin(1, " "))"
    | Satur01, Bfloat16_prec _ ->
        f "__float2bfloat16(fmaxf(0.0f, fminf(1.0f, __bfloat162float(" "))))"
    | Satur01, Half_prec _ ->
        f
          "__hmax_nan(__ushort_as_half((unsigned short)0x0000U), \
           __hmin_nan(__ushort_as_half((unsigned short)0x3C00U), "
          "))"
    | Satur01, Single_prec _ -> f "fmaxf(0.0f, fminf(1.0f, " "))"
    | Satur01, _ -> f "fmax(0.0, fmin(1.0, " "))"
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
    | Sin, Bfloat16_prec _ -> bf16_func "sinf"
    | Sin, _ -> func "sinf"
    | Cos, Half_prec _ -> func "hcos"
    | Cos, Double_prec _ -> func "cos"
    | Cos, Bfloat16_prec _ -> bf16_func "cosf"
    | Cos, _ -> func "cosf"
    | Sqrt, Half_prec _ -> func "hsqrt"
    | Sqrt, Double_prec _ -> func "sqrt"
    | Sqrt, Bfloat16_prec _ -> bf16_func "sqrtf"
    | Sqrt, _ -> func "sqrtf"
    | Recip, Byte_prec _ ->
        invalid_arg
          (Dialect.error_prefix ^ ".unop_syntax: Recip not supported for byte/integer precisions")
    | Recip, Half_prec _ -> func "hrcp"
    | Recip, Single_prec _ -> f "(1.0f / (" "))"
    | Recip, Double_prec _ -> f "(1.0 / (" "))"
    (* [1 / bf16] is ambiguous: the int operand can pair with any of the bfloat16 conversions. *)
    | Recip, Bfloat16_prec _ -> f "__float2bfloat16(1.0f / __bfloat162float(" "))"
    | Recip, _ -> f "(1 / (" "))"
    | Recip_sqrt, Byte_prec _ ->
        invalid_arg
          (Dialect.error_prefix
         ^ ".unop_syntax: Recip_sqrt not supported for byte/integer precisions")
    | Recip_sqrt, Half_prec _ -> func "hrsqrt"
    | Recip_sqrt, Double_prec _ -> f "(1.0 / sqrt(" "))"
    | Recip_sqrt, Single_prec _ -> f "(1.0f / sqrtf(" "))"
    | Recip_sqrt, Bfloat16_prec _ -> f "__float2bfloat16(1.0f / sqrtf(__bfloat162float(" ")))"
    | Recip_sqrt, _ -> f "(1 / sqrtf(" "))"
    | Neg, _ -> f "(-(" "))"
    | Trunc, Double_prec _ -> func "trunc"
    | Trunc, Bfloat16_prec _ -> bf16_func "truncf"
    | Trunc, _ -> func "truncf"
    | Tanh_approx, Byte_prec _ ->
        invalid_arg
          (Dialect.error_prefix
         ^ ".unop_syntax: Tanh_approx not supported for byte/integer precisions")
    | Tanh_approx, Half_prec _ -> func "htanh_approx"
    | Tanh_approx, Single_prec _ -> func Dialect.tanh_approx
    | Tanh_approx, Bfloat16_prec _ -> bf16_func "tanhf"
    | Tanh_approx, _ -> func "tanh"
    (* [bf16 == 0.0] is ambiguous for the same reason as [1 / bf16] above. *)
    | Not, Bfloat16_prec _ -> f "__float2bfloat16(__bfloat162float(" ") == 0.0f ? 1.0f : 0.0f)"
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

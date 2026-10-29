open Base
open Verdict.Claims
module Cs = Ir.C_syntax
module Ops = Ir.Ops

module Input = struct
  let procs = [||]
end

module Cuda =
  Ir.Cuda_like_config.Make
    (struct
      include Ir.Cuda_like_config.Cuda

      let builtins = []
      let extra_blacklist = []
    end)
    (Input)

module Hip =
  Ir.Cuda_like_config.Make
    (struct
      include Ir.Cuda_like_config.Hip

      let builtins = []
      let extra_blacklist = []
    end)
    (Input)

let a = PPrint.string "a"
let b = PPrint.string "b"

let render doc =
  let buffer = Buffer.create 128 in
  PPrint.ToBuffer.pretty 1.0 110 buffer doc;
  Buffer.contents buffer

let () =
  p "CUDA bfloat16 addition keeps its intrinsic"
    (String.equal (render (Cuda.binop_syntax Ops.bfloat16 Ops.Add a b)) "__hadd(a, b)");
  p "HIP bfloat16 addition keeps its native operator"
    (String.equal (render (Hip.binop_syntax Ops.bfloat16 Ops.Add a b)) "(a + b)");
  p "CUDA approximate tanh keeps its intrinsic"
    (String.equal (render (Cuda.unop_syntax Ops.single Ops.Tanh_approx a)) "__tanhf(a)");
  p "HIP approximate tanh keeps its library function"
    (String.equal (render (Hip.unop_syntax Ops.single Ops.Tanh_approx a)) "tanhf(a)");
  (* gh-ocannl-1336: HIP's fast-math [sinf]/[cosf] map +-inf to a finite value, so f32 and bfloat16
     (widened into f32) route through [Builtins_hip]'s NaN-preserving wrappers; CUDA's keep NaN
     there and render bare. *)
  p_all "HIP f32 and bfloat16 sin and cos render through the NaN-preserving wrappers"
    [
      (Ops.single, Ops.Sin, "ocannl_sinf(a)");
      (Ops.single, Ops.Cos, "ocannl_cosf(a)");
      (Ops.bfloat16, Ops.Sin, "__float2bfloat16(ocannl_sinf(__bfloat162float(a)))");
      (Ops.bfloat16, Ops.Cos, "__float2bfloat16(ocannl_cosf(__bfloat162float(a)))");
    ]
    ~f:(fun (prec, op, want) -> String.equal (render (Hip.unop_syntax prec op a)) want);
  p_all "HIP's NaN-preserving sin and cos wrappers are defined in its builtins table"
    [ "ocannl_sinf"; "ocannl_cosf" ] ~f:(fun key ->
      List.exists Builtins_hip.builtins ~f:(fun (k, _, _) -> String.equal k key));
  (* gh-ocannl-1684: HIP's fast-math [sqrtf] reads an f32 subnormal as zero and its [fmodf] gives a
     signed zero where C gives NaN, so f32 and bfloat16 (widened) route through [Builtins_hip]'s
     wrappers, and half [fmod] widens into them too -- on both dialects, since two [__half] operands
     make [fmod] an ambiguous call. *)
  p_all "HIP sqrt, rsqrt and fmod render through their wrappers"
    [
      (Hip.unop_syntax Ops.single Ops.Sqrt a, "ocannl_sqrtf(a)");
      ( Hip.unop_syntax Ops.bfloat16 Ops.Sqrt a,
        "__float2bfloat16(ocannl_sqrtf(__bfloat162float(a)))" );
      (Hip.unop_syntax Ops.single Ops.Recip_sqrt a, "(1.0f / ocannl_sqrtf(a))");
      ( Hip.unop_syntax Ops.bfloat16 Ops.Recip_sqrt a,
        "__float2bfloat16(1.0f / ocannl_sqrtf(__bfloat162float(a)))" );
      (Hip.binop_syntax Ops.single Ops.Mod a b, "ocannl_fmodf(a, b)");
      ( Hip.binop_syntax Ops.bfloat16 Ops.Mod a b,
        "__float2bfloat16(ocannl_fmodf(__bfloat162float(a), __bfloat162float(b)))" );
      ( Hip.binop_syntax Ops.half Ops.Mod a b,
        "__float2half(ocannl_fmodf(__half2float(a), __half2float(b)))" );
    ]
    ~f:(fun (doc, want) -> String.equal (render doc) want);
  p_all "HIP's sqrt and fmod wrappers are defined in its builtins table"
    [ "ocannl_sqrtf"; "ocannl_fmodf" ] ~f:(fun key ->
      List.exists Builtins_hip.builtins ~f:(fun (k, _, _) -> String.equal k key));
  p_all "CUDA sqrt, rsqrt and fmod keep the library functions, half fmod widened"
    [
      (Cuda.unop_syntax Ops.single Ops.Sqrt a, "sqrtf(a)");
      (Cuda.unop_syntax Ops.single Ops.Recip_sqrt a, "(1.0f / sqrtf(a))");
      (Cuda.binop_syntax Ops.single Ops.Mod a b, "fmodf(a, b)");
      ( Cuda.binop_syntax Ops.half Ops.Mod a b,
        "__float2half(fmodf(__half2float(a), __half2float(b)))" );
    ]
    ~f:(fun (doc, want) -> String.equal (render doc) want);
  p "CUDA f32 sin and cos keep the library functions"
    (String.equal (render (Cuda.unop_syntax Ops.single Ops.Sin a)) "sinf(a)"
    && String.equal (render (Cuda.unop_syntax Ops.single Ops.Cos a)) "cosf(a)");
  p "CUDA bfloat16 ReLU keeps NaN-propagating half intrinsic"
    (String.equal
       (render (Cuda.unop_syntax Ops.bfloat16 Ops.Relu a))
       "__hmax_nan(__ushort_as_bfloat16((unsigned short)0x0000U), a)");
  p "HIP bfloat16 ReLU keeps its float bridge"
    (String.equal
       (render (Hip.unop_syntax Ops.bfloat16 Ops.Relu a))
       "__float2bfloat16(fmaxf(0.0f, __bfloat162float(a)))");
  let bridges unop binop ternop =
    [
      render (unop Ops.fp8 Ops.Exp a);
      render (binop Ops.fp8 Ops.Add a b);
      render (ternop Ops.fp8 Ops.FMA a b a);
    ]
  in
  p_all "every HIP arithmetic arity narrows fp8 through the guard"
    (bridges Hip.unop_syntax Hip.binop_syntax Hip.ternop_syntax)
    ~f:(String.is_prefix ~prefix:"ocannl_single_to_fp8_uniform(");
  p_all "every CUDA arithmetic arity narrows fp8 through the vendor type"
    (bridges Cuda.unop_syntax Cuda.binop_syntax Cuda.ternop_syntax)
    ~f:(String.is_prefix ~prefix:"(__nv_fp8_e5m2)(");
  p_all "both shared dialects preserve operand evaluation contracts"
    [
      (Cuda.binop_syntax, Cuda.ternop_syntax, Cuda.select_blend_syntax);
      (Hip.binop_syntax, Hip.ternop_syntax, Hip.select_blend_syntax);
    ]
    ~f:(fun (binop_syntax, ternop_syntax, select_blend_syntax) ->
      List.is_empty
        (Cs.operand_conditionality_violations ~binop_syntax ~ternop_syntax ~select_blend_syntax))

(* gh-ocannl-707: each dialect's vendor-header table reaches the EFFECTIVE config. It is checked
   through [kernel_ident] of the instantiated syntax rather than by membership in the list: [Make]
   includes [Pure_C_config], whose own empty [vendor_routine_idents] a same-named binding inside
   [Make] silently picked up, leaving the table built and unused (local Codex review). The tables
   are measured per toolchain (gh-ocannl-1338, [tools/vendor_idents.exe]), so the exemplars are each
   dialect's own: names its compiler rejected as a kernel name, one per class -- a type, a macro, a
   C-linkage function -- plus one of the reserved namespace, which a rule covers instead. *)
module Cuda_syntax = Cs.C_syntax (Cuda)
module Hip_syntax = Cs.C_syntax (Hip)

let cuda_exemplars =
  [
    ("half2", "half2__");
    ("nv_bfloat16", "nv_bfloat16__");
    ("CUDARTAPI", "CUDARTAPI__");
    ("rsqrtf", "rsqrtf__");
    ("__hadd", "k__hadd");
  ]

let hip_exemplars =
  [
    ("half2", "half2__");
    ("float4", "float4__");
    ("HIPRT_INF_FP16", "HIPRT_INF_FP16__");
    ("FLT_MAX", "FLT_MAX__");
    ("__hadd", "k__hadd");
  ]

let () =
  p_all "CUDA routine names avoid the names nvrtc rejects as a kernel name" cuda_exemplars
    ~f:(fun (name, emitted) -> String.equal (Cuda_syntax.kernel_ident name) emitted);
  p_all "HIP routine names avoid the names hiprtc rejects as a kernel name" hip_exemplars
    ~f:(fun (name, emitted) -> String.equal (Hip_syntax.kernel_ident name) emitted);
  p_all "every name of the CUDA table is mangled by the CUDA config"
    Ir.Cuda_like_config.Cuda.vendor_routine_idents ~f:(fun name ->
      not (String.equal (Cuda_syntax.kernel_ident name) name));
  p_all "every name of the HIP table is mangled by the HIP config"
    Ir.Cuda_like_config.Hip.vendor_routine_idents ~f:(fun name ->
      not (String.equal (Hip_syntax.kernel_ident name) name));
  p_all "ordinary CUDA and HIP routine names stay unmangled" [ "c_fwd"; "half_step"; "gh707" ]
    ~f:(fun name ->
      String.equal (Cuda_syntax.kernel_ident name) name
      && String.equal (Hip_syntax.kernel_ident name) name)

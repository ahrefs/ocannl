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

(* gh-ocannl-707: the vendor-header table reaches the EFFECTIVE configs. It is checked through
   [kernel_ident] of the instantiated syntax rather than by membership in the top-level list: [Make]
   includes [Pure_C_config], whose own empty [vendor_routine_idents] a same-named binding inside
   [Make] silently picked up, leaving the table built and unused (local Codex review). *)
module Cuda_syntax = Cs.C_syntax (Cuda)
module Hip_syntax = Cs.C_syntax (Hip)

let vendor_exemplars =
  [
    ("half2", "half2__");
    ("float4", "float4__");
    ("nv_bfloat16", "nv_bfloat16__");
    ("hip_fp8_storage_t", "hip_fp8_storage_t__");
    ("CUDART_INF_FP16", "CUDART_INF_FP16__");
    ("HIPRT_INF_FP16", "HIPRT_INF_FP16__");
    ("rsqrtf", "rsqrtf__");
    ("__hadd", "k__hadd");
  ]

let () =
  p_all "CUDA routine names avoid the vendor headers' identifiers" vendor_exemplars
    ~f:(fun (name, emitted) -> String.equal (Cuda_syntax.kernel_ident name) emitted);
  p_all "HIP routine names avoid the vendor headers' identifiers" vendor_exemplars
    ~f:(fun (name, emitted) -> String.equal (Hip_syntax.kernel_ident name) emitted);
  p_all "every name of the shared vendor-header table is mangled by both effective configs"
    Ir.Cuda_like_config.vendor_header_idents ~f:(fun name ->
      (not (String.equal (Cuda_syntax.kernel_ident name) name))
      && not (String.equal (Hip_syntax.kernel_ident name) name));
  p_all "ordinary CUDA and HIP routine names stay unmangled" [ "c_fwd"; "half_step"; "gh707" ]
    ~f:(fun name ->
      String.equal (Cuda_syntax.kernel_ident name) name
      && String.equal (Hip_syntax.kernel_ident name) name)

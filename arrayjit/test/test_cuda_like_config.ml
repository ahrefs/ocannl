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
  (* gh-ocannl-1710: the intrinsic under CUDA's fast math only. The key is stated per value (an
     ambient statement of it outranks the file level, so the expectation follows the reader). *)
  p_all "CUDA approximate tanh is its intrinsic under fast math, and the library's without it"
    [ "true"; "false" ] ~f:(fun value ->
      let previous = Hashtbl.find Utils.config_file_args "cuda_fast_math" in
      Hashtbl.set Utils.config_file_args ~key:"cuda_fast_math" ~data:value;
      let got = render (Cuda.unop_syntax Ops.single Ops.Tanh_approx a) in
      let want = if Ir.Cuda_like_config.cuda_fast_math () then "__tanhf(a)" else "tanhf(a)" in
      (match previous with
      | None -> Hashtbl.remove Utils.config_file_args "cuda_fast_math"
      | Some data -> Hashtbl.set Utils.config_file_args ~key:"cuda_fast_math" ~data);
      String.equal got want);
  p "HIP approximate tanh keeps its library function"
    (String.equal (render (Hip.unop_syntax Ops.single Ops.Tanh_approx a)) "tanhf(a)");
  (* gh-ocannl-1779: CUDA's half functions are hardware approximations under either switch set, so
     under [cuda_fast_math] off they render as the precise f32 function of the widened operand,
     narrowed once; HIP's half functions are ROCm's OCML ones whatever [hip_fast_math] says. *)
  let half_rows =
    [
      (Ops.Exp, "hexp(a)", "__float2half(expf(__half2float(a)))");
      (Ops.Log, "hlog(a)", "__float2half(logf(__half2float(a)))");
      (Ops.Exp2, "hexp2(a)", "__float2half(exp2f(__half2float(a)))");
      (Ops.Log2, "hlog2(a)", "__float2half(log2f(__half2float(a)))");
      (Ops.Sin, "hsin(a)", "__float2half(sinf(__half2float(a)))");
      (Ops.Cos, "hcos(a)", "__float2half(cosf(__half2float(a)))");
      (Ops.Sqrt, "hsqrt(a)", "__float2half(sqrtf(__half2float(a)))");
      (Ops.Recip, "hrcp(a)", "__float2half((1.0f / (__half2float(a))))");
      (Ops.Recip_sqrt, "hrsqrt(a)", "__float2half((1.0f / sqrtf(__half2float(a))))");
      (Ops.Tanh_approx, "htanh_approx(a)", "__float2half(tanhf(__half2float(a)))");
    ]
  in
  let with_key key value f =
    let previous = Hashtbl.find Utils.config_file_args key in
    Hashtbl.set Utils.config_file_args ~key ~data:value;
    let result = f () in
    (match previous with
    | None -> Hashtbl.remove Utils.config_file_args key
    | Some data -> Hashtbl.set Utils.config_file_args ~key ~data);
    result
  in
  let policies = [ "true"; "false" ] in
  p_all ~min:20
    "CUDA half unary functions are its intrinsics under fast math, and the precise f32 function of \
     the widened operand narrowed once without it" (List.cartesian_product policies half_rows)
    ~f:(fun (value, (op, intrinsic, widened)) ->
      with_key "cuda_fast_math" value (fun () ->
          String.equal
            (render (Cuda.unop_syntax Ops.half op a))
            (if Ir.Cuda_like_config.cuda_fast_math () then intrinsic else widened)));
  p_all ~min:20 "HIP half unary functions are ROCm's half functions under either hip_fast_math"
    (List.cartesian_product policies half_rows) ~f:(fun (value, (op, intrinsic, _)) ->
      with_key "hip_fast_math" value (fun () ->
          String.equal (render (Hip.unop_syntax Ops.half op a)) intrinsic));
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
  (* gh-ocannl-1730: at f64, HIP's [fmod] shares the f32 defect, and its fast-math [1.0 / sqrt(x)]
     is NaN for +-0 and +inf, where ROCm's [rsqrt] is not. *)
  p_all "HIP f64 rsqrt renders as ROCm's rsqrt and f64 fmod through its wrapper"
    [
      (Hip.unop_syntax Ops.double Ops.Recip_sqrt a, "rsqrt(a)");
      (Hip.binop_syntax Ops.double Ops.Mod a b, "ocannl_fmod(a, b)");
    ]
    ~f:(fun (doc, want) -> String.equal (render doc) want);
  p_all "HIP's sqrt and fmod wrappers are defined in its builtins table"
    [ "ocannl_sqrtf"; "ocannl_fmodf"; "ocannl_fmod" ] ~f:(fun key ->
      List.exists Builtins_hip.builtins ~f:(fun (k, _, _) -> String.equal k key));
  p_all "CUDA sqrt, rsqrt and fmod keep the library functions, half fmod widened"
    [
      (Cuda.unop_syntax Ops.single Ops.Sqrt a, "sqrtf(a)");
      (Cuda.unop_syntax Ops.single Ops.Recip_sqrt a, "(1.0f / sqrtf(a))");
      (Cuda.binop_syntax Ops.single Ops.Mod a b, "fmodf(a, b)");
      (Cuda.unop_syntax Ops.double Ops.Recip_sqrt a, "(1.0 / sqrt(a))");
      (Cuda.binop_syntax Ops.double Ops.Mod a b, "fmod(a, b)");
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
        (Cs.operand_conditionality_violations ~binop_syntax ~ternop_syntax ~select_blend_syntax));
  (* gh-ocannl-1695: HIP blends a value select over [auto] locals at every float compute precision
     but fp8, CUDA at none (ptxas if-converts the short-circuiting spelling itself), and the blend's
     truth is the dialect's own test -- the GPU [Satur01_gate]'s [0 < x && x < 1], not cc's
     [fabs(floor(x)) > 0] (ahrefs/ocannl#1647). *)
  let float_precs = [ Ops.half; Ops.bfloat16; Ops.single; Ops.double ] in
  p_all "HIP blends value selects over auto locals at f16, bf16, f32 and f64" float_precs
    ~f:(fun prec ->
      match Hip.select_blend_syntax prec with
      | Some (Cs.Blend_over_locals "auto") -> true
      | _ -> false);
  p_all "HIP keeps fp8's short-circuiting select, and CUDA every precision's"
    (Ops.fp8 :: float_precs) ~f:(fun prec ->
      Option.is_none (Cuda.select_blend_syntax prec)
      && ((not (Ops.equal_prec prec Ops.fp8)) || Option.is_none (Hip.select_blend_syntax prec)));
  let x = PPrint.string "x" and g = PPrint.string "g" in
  let gate =
    render
      (Cs.Over_locals.blend_gate ~decl:"auto" ~binop_syntax:Hip.binop_syntax Ops.single
         Ops.Satur01_gate x g)
  in
  p "the GPU Satur01 blend binds its operand, then keeps the dialect's own test over the local"
    (String.is_prefix gate ~prefix:"[&]() { auto ocannl_sel_a__ = g;"
    && String.is_substring gate ~substring:"x > 0.0f && x < 1.0f"
    && String.is_substring gate ~substring:"? ocannl_sel_a__ : 0.0f");
  let where =
    render
      (Cs.Over_locals.blend_where ~decl:"auto" ~ternop_syntax:Hip.ternop_syntax Ops.single x a b)
  in
  p "the GPU Where blend binds both arms, then selects between them with the dialect's own ternary"
    (String.equal where
       "[&]() { auto ocannl_sel_a__ = a; auto ocannl_sel_b__ = b; return ((x) ? ocannl_sel_a__ : \
        ocannl_sel_b__); }()")

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

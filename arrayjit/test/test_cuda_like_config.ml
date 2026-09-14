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
    [ (Cuda.binop_syntax, Cuda.ternop_syntax); (Hip.binop_syntax, Hip.ternop_syntax) ]
    ~f:(fun (binop_syntax, ternop_syntax) ->
      List.is_empty (Cs.operand_conditionality_violations ~binop_syntax ~ternop_syntax))

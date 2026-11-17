(* The NVRTC and missing-toolkit arms share the claims and their order. Only an open leg calls the
   probe; the missing arm supplies closed environment legs and no compiler implementation. *)
open Verdict.Claims

type probe =
  | Wide_fragment
  | Compute_75
  | Compute_70
  | Shipping_floor
  | Async_floor
  | Bf16_refusal
  | K8_statement
  | K8_compute_75
  | K8_shipping_floor
  | K8_loads_at_75
  | K8_refused_at_70
  | K16_loads_at_80
  | K16_refused_at_75

let run ~leg ~compute_70_leg ~jit_70_leg check =
  in_leg ~leg "the witness renders the wide fragment scope and both destination conversions"
    (fun label -> p label (check Wide_fragment));
  in_leg ~leg "compute_75 PTX targets sm_75 and accumulates WMMA f16 products in f32" (fun label ->
      p label (check Compute_75));
  in_leg ~leg:compute_70_leg "compute_70 PTX targets sm_70 and accumulates WMMA f16 products in f32"
    (fun label -> p label (check Compute_70));
  in_leg ~leg "the wide fragment alone ships at the toolkit-adjusted f16 floor" (fun label ->
      p label (check Shipping_floor));
  in_leg ~leg "cp.async still raises a wide-f16 kernel to the async-copy floor" (fun label ->
      p label (check Async_floor));
  in_leg ~leg "compute_75 refuses the otherwise compilable WMMA bf16 control" (fun label ->
      p label (check Bf16_refusal));
  (* gh-ocannl-2284: the per-statement m16n8k8 arm. NVRTC passes inline [asm] through to PTX
     unassembled, so a target's refusal of an [mma.sync] shape is the PTX assembler's: the driver's
     JIT, which checks each instruction against the module's [.target]. *)
  in_leg ~leg
    "at sm_75's resolution the per-statement witness renders the m16n8k8 arm and both destination \
     conversions" (fun label -> p label (check K8_statement));
  in_leg ~leg
    "compute_75 PTX of the m16n8k8 witness targets sm_75 and carries the m16n8k8 instruction"
    (fun label -> p label (check K8_compute_75));
  in_leg ~leg "the m16n8k8 witness ships at the arm's floor, compute_75" (fun label ->
      p label (check K8_shipping_floor));
  in_leg ~leg "the driver's PTX assembler loads the m16n8k8 witness at sm_75" (fun label ->
      p label (check K8_loads_at_75));
  in_leg ~leg:jit_70_leg
    "the driver's PTX assembler refuses the m16n8k8 witness retargeted to sm_70, naming the shape"
    (fun label -> p label (check K8_refused_at_70));
  in_leg ~leg "the driver's PTX assembler loads the m16n8k16 f16 statement control at sm_80"
    (fun label -> p label (check K16_loads_at_80));
  in_leg ~leg
    "the driver's PTX assembler refuses the m16n8k16 f16 statement control at sm_75, naming the \
     shape" (fun label -> p label (check K16_refused_at_75))

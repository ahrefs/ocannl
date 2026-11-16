(* The NVRTC and missing-toolkit arms share the claims and their order. Only an open leg calls the
   probe; the missing arm supplies closed environment legs and no compiler implementation. *)
open Verdict.Claims

type probe = Wide_fragment | Compute_75 | Compute_70 | Shipping_floor | Async_floor | Bf16_refusal

let run ~leg ~compute_70_leg check =
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
      p label (check Bf16_refusal))

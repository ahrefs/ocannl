(* gh-ocannl-514, the placement decision surface: the enablement prior, the ranking, and the
   partial-vector floor.

   The matmul reads a policy-virtual operand ([mbs], a pointwise scale of [mb]) and its result
   ([mc]) inlines into the relu consumer, so the default-placement lowering carries no recognizable
   matmul site at all — the site exists only in the all-materialized specialization of the decision
   surface, reading [mbs] into [mc]. With synthetic GPU limits advertising an (f32, f32, f32) mma
   format tile (classification is a pure function of the lowerings, so no GPU is needed — the
   sketch_family_tree harness), the enablement prior must promote exactly the site's flip
   candidates: materializing them is what makes the tensorized family expressible.

   The decoy [us] (a pointwise scale read with a 256-fold per-cell multiplicity by a broadcast
   consumer) carries a larger recompute cost than the site candidates — the gh-558 shape, where cost
   ordering buries the family-unlocking flips below a candidate that unlocks nothing and enablement
   ordering does not. The multiplicity is what makes it the second-dearest flip under the modeled
   recompute cost (gh-ocannl-637), above the partial reduction [n12], whose one instantiation
   replays a row of the matmul.

   The floor closure is asserted monotone in the committed materializations, under the envelope
   constants pinned by the rule's command line (cc carries none of its own).

   Printed candidate lists carry names, flip kinds, costs and enablement marks: the ranking is
   deterministic given the computation and the synthetic limits. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Asgns = Ir.Assignments
open Verdict.Claims

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let f32 = Ir.Backend_intf.Mma_f32

let gpu_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    mma =
      Some
        {
          Ir.Backend_intf.mma_simd_width = 32;
          mma_tile = (8, 8, 8);
          mma_format_tiles = [ ((f32, f32, f32), (8, 8, 8)) ];
          mma_f16_wide_acc_scopes = [];
          mma_bf16_wide_acc_scopes = [];
          mma_staged_layouts = [];
          mma_pipeline_depths = [];
        };
  }

let n = 8
let m = 256

let () =
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 5) -. 2.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let uv = Array.init m ~f:(fun i -> Float.of_int (i % 3) *. 0.25) in
  let wv = Array.init (m * m) ~f:(fun i -> Float.of_int (i % 11) *. 0.125) in
  let u = TDSL.ndarray uv ~label:[ "u" ] ~output_dims:[ m ] () in
  let w = TDSL.ndarray wv ~label:[ "w" ] ~output_dims:[ m; m ] () in
  (* The mma-site half: mbs and mc are policy-virtual. *)
  let%op mbs = mb *. 0.5 in
  let%op mc = ma * mbs in
  let%op t2 = relu mc in
  (* The decoy half: us is policy-virtual, read broadcast by every row of w. *)
  let%op us = u *. 2.0 in
  let%op d2 = w +* "ij; j => ij" us in
  let%op total = t2 ++ "ij => 0" + (d2 ++ "ij => 0") in
  let comp = named "ps" (Train.forward total) in
  let ctx = Context.auto () in
  let base = Context.lowered_for_decisions ctx comp Ir.Indexing.Empty in
  let candidates = base.LL.flip_candidates in
  let to_materialize =
    List.filter_map candidates ~f:(fun fc ->
        match fc.LL.fc_flip with `Materialize -> Some fc.LL.fc_tn | `Inline | `Footprint -> None)
  in
  let allmat =
    Context.lowered_for_decisions ~materialized:to_materialize ctx comp Ir.Indexing.Empty
  in
  let enablement, disablement =
    Autotune.placement_enablement ~limits:gpu_limits ~static_indices:[] ~base ~allmat
  in
  let mem set (t : Tensor.t) = Set.mem set t.Tensor.value in
  let mem_tn (fc : LL.flip_candidate) (t : Tensor.t) = Tn.equal fc.LL.fc_tn t.Tensor.value in
  p "the enablement set is nonempty" (not (Set.is_empty enablement));
  p "it contains the site operand mbs" (mem enablement mbs);
  p "it contains the site destination mc" (mem enablement mc);
  p "it does not contain the decoy us" (not (mem enablement us));
  p "no site is eligible under default placements (empty disablement)" (Set.is_empty disablement);
  let show ordering =
    let ranked = Autotune.rank_flip_candidates ~ordering ~enablement ~disablement candidates in
    List.iter ranked ~f:(fun fc ->
        Stdio.printf "  %-11s %-12s cost %-5d%s\n"
          (match fc.LL.fc_flip with
          | `Materialize -> "materialize"
          | `Inline -> "inline"
          | `Footprint -> "footprint")
          (Tn.debug_name fc.LL.fc_tn) fc.LL.fc_recompute_cost
          (if Set.mem enablement fc.LL.fc_tn then "  [enablement]" else ""));
    ranked
  in
  Stdio.printf "cost ranking:\n";
  let by_cost = show `Cost in
  Stdio.printf "enablement ranking:\n";
  let by_enablement = show `Enablement in
  let is_en fc = Set.mem enablement fc.LL.fc_tn in
  p "cost ranking buries the enablement candidates below the decoy"
    (match by_cost with fc :: _ -> not (is_en fc) | [] -> false);
  p "enablement ranking puts the family-unlocking materialize flip first"
    (match by_enablement with
    | fc :: _ -> (
        is_en fc && match fc.LL.fc_flip with `Materialize -> true | `Inline | `Footprint -> false)
    | [] -> false);
  p "enablement ranking puts the family-breaking inline flip last"
    (match List.last by_enablement with
    | Some fc -> (
        is_en fc && match fc.LL.fc_flip with `Inline -> true | `Materialize | `Footprint -> false)
    | None -> false);
  (* gh-ocannl-579, the profitability term: the prior above prices EXPRESSIBILITY only, so on a
     device where the family it unlocks has been MEASURED to lose, promoting its flips is pure
     opportunity cost — it displaces cheaper flips out of a small budget. The evidence is the
     placement A/B arms' own reports, here synthesized from gh-514's metal cell (best 7.5 ms, best
     tensorized 92 ms) and its hip cell (the arm's winner IS tensorized). *)
  let arm ~best_ms ~mma_timed ~mma_best_ms =
    {
      (Autotune.no_search_report ~timing:Autotune.Queued) with
      Autotune.best_ms;
      mma_timed;
      mma_best_ms;
    }
  in
  let losing =
    Autotune.family_profit_of_reports [ arm ~best_ms:7.5 ~mma_timed:3 ~mma_best_ms:92.0 ]
  in
  let paying =
    Autotune.family_profit_of_reports [ arm ~best_ms:1.28 ~mma_timed:16 ~mma_best_ms:1.28 ]
  in
  let ranked ordering profit =
    Autotune.rank_flip_candidates ~ordering ~profit ~enablement ~disablement candidates
  in
  let same a b = List.equal (fun x y -> Tn.equal x.LL.fc_tn y.LL.fc_tn) a b in
  p "a measured-losing family ranks the surface exactly as cost does"
    (same (ranked `Profitable losing) by_cost);
  p "a measured-paying family ranks it exactly as the enablement prior does"
    (same (ranked `Profitable paying) by_enablement);
  p "an unmeasured family leaves the enablement prior standing"
    (same (ranked `Profitable Autotune.Unmeasured) by_enablement);
  p "the pure enablement ordering ignores the evidence (the evaluation baseline)"
    (same (ranked `Enablement losing) by_enablement);
  (* The displacement, in miniature: the decoy is this surface's cheap-but-highest-cost flip, the
     analogue of metal's winning `inline n32_relu.grad` at cost 1024 / cost-rank 5. At a budget of
     two the promotion pushes it out of the chain; the profitability term hands the slot back. *)
  let budget = 2 in
  let prefix_has l t = List.exists (List.take l budget) ~f:(fun fc -> mem_tn fc t) in
  p_none "the enablement prior pushes the decoy out of a budget-2 chain"
    (List.take by_enablement budget) ~f:(fun fc -> mem_tn fc us);
  p "under a measured-losing family the decoy is back in the budget-2 chain"
    (prefix_has (ranked `Profitable losing) us);
  p_none "under a measured-paying family the promotion keeps the budget slot"
    (List.take (ranked `Profitable paying) budget)
    ~f:(fun fc -> mem_tn fc us);
  (* The floor closure, under the rule's pinned envelope: monotone in the commitments, and strictly
     above the empty commitment once the site nodes' traffic is certain. *)
  (* An unconditional ordering reads no evidence at all — including the margin, which a run pinned
     to a baseline must not be able to fail on. *)
  p "an unconditional ordering records no profitability evidence"
    (Option.is_none
       (Autotune.placement_surface ~ordering:`Enablement ctx comp Ir.Indexing.Empty)
         .Autotune.ps_profit);
  p "the profitable ordering records the verdict it ranked by"
    (match
       (Autotune.placement_surface ~ordering:`Profitable
          ~evidence:[ arm ~best_ms:7.5 ~mma_timed:3 ~mma_best_ms:92.0 ]
          ctx comp Ir.Indexing.Empty)
         .Autotune.ps_profit
     with
    | Some (Autotune.Loses _) -> true
    | _ -> false);
  let surface = Autotune.placement_surface ~ordering:`Enablement ctx comp Ir.Indexing.Empty in
  let f0 = surface.Autotune.ps_floor_ms ~materialized:[] in
  let f1 = surface.Autotune.ps_floor_ms ~materialized:[ mbs.Tensor.value ] in
  let f2 = surface.Autotune.ps_floor_ms ~materialized:[ mbs.Tensor.value; mc.Tensor.value ] in
  let ge a b = match (a, b) with Some a, Some b -> Float.(a >= b) | _ -> false in
  p "the floor is present under the pinned envelope" (Option.is_some f0);
  p "committing mbs does not lower the floor" (ge f1 f0);
  p "committing mc on top does not lower it either" (ge f2 f1);
  p "the two commitments strictly raise the floor"
    (match (f2, f0) with Some f2, Some f0 -> Float.(f2 > f0) | _ -> false)

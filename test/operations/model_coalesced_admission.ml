(* gh-ocannl-1600: model_default ranks the coalesced layouts where both terms that tell them apart
   are priced.

   A coalesced layout never changes which lanes read contiguous cells -- at every gh-ocannl-1479
   cell a coal-n or coal-k leaf measured as its uncoalesced twin, with equal memory transactions.
   What it adds is geometry the uncoalesced site cannot reach: the attention out projection's
   contraction merged into one k of 256 takes deep K tiles (32-256 per block), and those differ from
   their twins only in workgroup memory, which the occupancy factor charges. On HIP's unified
   gfx1151 (minix) the deep-K leaves were the fastest at outproj, and the occupancy factor at the
   physical 128 KiB per WGP made the score's argmin the fastest leaf.

   Here, at that outproj site (b8), lowered on the CPU and scheduled hermetically under a
   minix-shaped stand-in record (nothing compiles, so the claims run on every backend):
   [model_ranks_coalesced] holds exactly where both the transaction rate and the workgroup capacity
   resolve; [model_family_search] with [~coalesced:true] never returns a leaf ranked above the
   uncoalesced walk's pick, which [~coalesced:false] returns. (Without the capacity the menu's best
   would be a 256-deep tile measured 3.6x slower, ahead by a sub-1% op count: hence the gate.)

   Since gh-ocannl-1661 partitions a staged copy across the block's threads, the leaves near the
   outproj argmin scored within 0.1% of each other: the compute leg is flat across the staged block
   tiles. gh-ocannl-1821 adds what orders them -- the staged copies' volume, which a 64-wide block
   tile halves against a 32-wide one -- and, with it, the gate's second geometry term: on a
   rog-shaped record (CUDA's RTX 5070 Ti: no workgroup capacity, a staged-copy rate)
   [model_ranks_coalesced] holds through the copy rate, and at the outproj site at b256 -- where a
   64-wide coal-k tile measured 1.48x faster than the uncoalesced pick -- the walk over the
   coalesced menu returns a coal-k leaf ranked strictly below the uncoalesced walk's pick, which
   without the copy rate it does not. Ranks go to stderr: device constants' quotients. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module BI = Ir.Backend_intf

let minix_like =
  {
    BI.no_hardware_limits with
    max_threads_per_workgroup = Some 1024;
    max_workgroup_memory_bytes = Some 65536;
    max_workgroup_dims = Some (1024, 1024, 1024);
    max_grid_yz = Some 65535;
    simdgroup_width = Some 32;
    peak_flops = Some 3.0e13;
    peak_memory_bandwidth = Some 2.56e11;
    max_resident_threads = Some (20 * 2048);
    issue_rate = Some 2.8e12;
    dram_bandwidth = Some 2.35e11;
    multiprocessors = Some 20;
    fill_threads = Some 384;
    transaction_rate = Some 1.26e13;
    workgroup_bytes_per_multiprocessor = Some 131072;
  }

let without_capacity = { minix_like with workgroup_bytes_per_multiprocessor = None }
let without_rate = { minix_like with transaction_rate = None }

(* CUDA's advisories on rog-nv's RTX 5070 Ti laptop part (46 SMs of 1536 threads), with its
   staged-copy rate and no occupancy capacity. *)
let rog_like =
  {
    BI.no_hardware_limits with
    max_threads_per_workgroup = Some 1024;
    max_workgroup_memory_bytes = Some 49152;
    max_workgroup_dims = Some (1024, 1024, 64);
    max_grid_yz = Some 65535;
    simdgroup_width = Some 32;
    peak_flops = Some 1.5e13;
    peak_memory_bandwidth = Some 8.0e12;
    cache_capacity_bytes = Some (12 * 1024 * 1024 * 1024);
    cache_bandwidth = Some 1.8e12;
    issue_rate = Some 3.4e12;
    dram_bandwidth = Some 6.0e11;
    max_resident_threads = Some (46 * 1536);
    multiprocessors = Some 46;
    fill_threads = Some 256;
    transaction_rate = Some 1.0e12;
    staged_copy_rate = Some 1.0e12;
  }

let rog_without_copy_rate = { rog_like with staged_copy_rate = None }

let () =
  p "the coalesced layouts are ranked where the rate and the capacity both resolve"
    (Autotune.model_ranks_coalesced ~limits:minix_like);
  p "not without the workgroup capacity"
    (not (Autotune.model_ranks_coalesced ~limits:without_capacity));
  p "not without the transaction rate" (not (Autotune.model_ranks_coalesced ~limits:without_rate));
  p "not without the launch fill's multiprocessor count, which the occupancy factor reads"
    (not
       (Autotune.model_ranks_coalesced
          ~limits:{ minix_like with multiprocessors = None; fill_threads = None }));
  p "not without the device residency"
    (not (Autotune.model_ranks_coalesced ~limits:{ minix_like with max_resident_threads = None }));
  p "not without a lane width"
    (not (Autotune.model_ranks_coalesced ~limits:{ minix_like with simdgroup_width = None }))

(* gh-ocannl-1821: the copy rate is the other geometry term the gate accepts. *)
let () =
  p "the coalesced layouts are ranked through the staged-copy rate, without a capacity"
    (Autotune.model_ranks_coalesced ~limits:rog_like);
  p "not without the staged-copy rate (and no capacity)"
    (not (Autotune.model_ranks_coalesced ~limits:rog_without_copy_rate));
  p "not through the copy rate without the launch fill it prices with"
    (not
       (Autotune.model_ranks_coalesced
          ~limits:{ rog_like with multiprocessors = None; fill_threads = None }));
  p "not through the copy rate without the transaction rate"
    (not (Autotune.model_ranks_coalesced ~limits:{ rog_like with transaction_rate = None }))

let () =
  let ctx = Context.cpu () in
  let cell idx = Float.of_int (1 + (Array.fold idx ~init:0 ~f:( + ) % 7)) *. 0.125 in
  let w = NTDSL.init ~l:"mca_wout" ~prec:Ir.Ops.single ~o:[ 256 ] ~i:[ 8; 32 ] ~f:cell () in
  let x = NTDSL.init ~l:"mca_xout" ~prec:Ir.Ops.single ~b:[ 8; 128 ] ~o:[ 8; 32 ] ~f:cell () in
  let%op d = w * x in
  Train.set_materialized d.Tensor.value;
  let base =
    Context.lowered_for_decisions ~name:"mca_outproj" ctx (Train.forward d) Ir.Indexing.Empty
  in
  let capabilities = BI.no_codegen_capabilities in
  let ranked ~limits segs = Option.map (Autotune.ranking_ms ~capabilities ~limits segs) ~f:fst in
  let search ~limits ~coalesced () =
    let score = ranked ~limits in
    match
      Autotune.model_family_search ~coalesced ~capabilities ~is_gpu:true ~is_cpu:false ~limits
        ~static_indices:[] ~peak_flops:limits.peak_flops
        ~peak_memory_bandwidth:limits.peak_memory_bandwidth ~score ~incumbent:Float.infinity base
    with
    | Some (Some (p, rank), _, _) ->
        Stdio.eprintf "model_coalesced_admission: coalesced=%b pick %s rank %.6f ms%s\n" coalesced
          (Autotune.sketch_params_label p) rank " (not part of the golden)";
        Some (p, rank)
    | _ -> None
  in
  let hermetic (o : LL.optimized) =
    {
      o with
      LL.traced_store = Hashtbl.copy o.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
    }
  in
  (* Every coalesced leaf's rank, for the record. *)
  List.iter
    (Autotune.coalesced_seed_params ~capabilities ~is_gpu:true ~is_cpu:false ~limits:minix_like base)
    ~f:(fun p ->
      let rank =
        match
          Ir.Schedule.apply
            (Autotune.sketch_schedule ~accum_prec:Fn.id ~p (hermetic base))
            (hermetic base)
        with
        | exception e -> "refused: " ^ Exn.to_string e
        | post ->
            Option.value_map (Autotune.ranking_ms ~capabilities ~limits:minix_like [ post ])
              ~default:"n/a" ~f:(fun (r, _) -> Printf.sprintf "%.6f" r)
      in
      Stdio.eprintf "model_coalesced_admission: %s rank %s (not part of the golden)\n"
        (Autotune.sketch_params_label p) rank);
  let plain = search ~limits:minix_like ~coalesced:false () in
  let with_coal = search ~limits:minix_like ~coalesced:true () in
  p "the uncoalesced walk returns an uncoalesced leaf"
    (Option.exists plain ~f:(fun (p, _) -> Option.is_none p.Autotune.sk_coalesce));
  p "with the coalesced menu, the pick ranks no higher than the uncoalesced walk's"
    (match (plain, with_coal) with Some (_, r0), Some (_, r1) -> Float.(r1 <= r0) | _ -> false);
  ()

(* gh-ocannl-1821: at outproj b256 under the rog-shaped record, the real score's coalesced walk
   finds a coal-k leaf strictly below the uncoalesced pick; without the copy rate, nothing below
   it. *)
let () =
  let ctx = Context.cpu () in
  let cell idx = Float.of_int (1 + (Array.fold idx ~init:0 ~f:( + ) % 7)) *. 0.125 in
  let w = NTDSL.init ~l:"mca_wout256" ~prec:Ir.Ops.single ~o:[ 256 ] ~i:[ 8; 32 ] ~f:cell () in
  let x = NTDSL.init ~l:"mca_xout256" ~prec:Ir.Ops.single ~b:[ 256; 128 ] ~o:[ 8; 32 ] ~f:cell () in
  let%op d = w * x in
  Train.set_materialized d.Tensor.value;
  let base =
    Context.lowered_for_decisions ~name:"mca_outproj256" ctx (Train.forward d) Ir.Indexing.Empty
  in
  let capabilities = BI.no_codegen_capabilities in
  let search ~limits ~coalesced =
    let score segs = Option.map (Autotune.ranking_ms ~capabilities ~limits segs) ~f:fst in
    match
      Autotune.model_family_search ~coalesced ~capabilities ~is_gpu:true ~is_cpu:false ~limits
        ~static_indices:[] ~peak_flops:limits.BI.peak_flops
        ~peak_memory_bandwidth:limits.peak_memory_bandwidth ~score ~incumbent:Float.infinity base
    with
    | Some (Some (p, rank), _, _) ->
        Stdio.eprintf "model_coalesced_admission: b256 coalesced=%b pick %s rank %.6f ms%s\n"
          coalesced (Autotune.sketch_params_label p) rank " (not part of the golden)";
        Some (p, rank)
    | _ -> None
  in
  let is_coal_k (p : Autotune.sketch_params) =
    match p.sk_coalesce with Some Autotune.Coalesce_contraction -> true | _ -> false
  in
  let strictly_better ~limits =
    match (search ~limits ~coalesced:false, search ~limits ~coalesced:true) with
    | Some (p0, r0), Some (p, r1) ->
        Option.is_none p0.Autotune.sk_coalesce && is_coal_k p && Float.(r1 < r0)
    | _ -> false
  in
  p "under the staged-copy rate, a coal-k leaf strictly beats the uncoalesced walk's pick at b256"
    (strictly_better ~limits:rog_like);
  p "without it, the coalesced menu finds nothing strictly below the uncoalesced pick"
    (not (strictly_better ~limits:rog_without_copy_rate))

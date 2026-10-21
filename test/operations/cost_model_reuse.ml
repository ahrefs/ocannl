(* gh-ocannl-1253: the ranking score charges an untiled contraction for its operand re-reads.

   [Ir.Cost_model.analyze] charges distinct cells under a perfect cache, so it scores the untiled
   weight gradient [dw[o,i] += dy[k,o] * x[k,i]] exactly at its roofline floor however often each
   block re-reads the operands, and no reuse-tiled plan can rank below it. The ranking score
   ([Cost_model.ranking_seconds]) charges global reads per block -- the cells one block touches, its
   [Grid] loops standing still, times the blocks -- once the working set (per-block bytes times
   resident blocks) exceeds the cache capacity.

   Hand-built IR, scheduled by the default GPU preset and by the GPU sketch family under a
   GPU-shaped [hardware_limits] record, on whatever backend runs the test: nothing is compiled or
   executed, and every claim is an integer or a comparison of two scores. The envelope is passed
   explicitly (bandwidth-dominant, so the byte counts decide), never read from configuration.

   The claims: the untiled default plan requests between [M·N·K·elt] and [M·N·K·elt·(1 + 1/wg)] of
   its operands, [wg] its workgroup size; a [bm × bn] scalar blocktile requests exactly
   [M·N·K·elt·(1/bm + 1/bn)]; at or below capacity, and with no capacity, the score is [analyze]'s
   roofline; raising the capacity never raises the score; within capacity a cache tier charges the
   re-reads at the cache bandwidth, monotonically in it; and on every leaf of the site's GPU sketch
   tree and on the default plan, the branch-and-bound's bound ([completion_floor]'s roofline) is at
   most [analyze]'s roofline, which is at most the score -- the score can never make the bound prune
   a winner. *)

open Base
module LL = Ir.Low_level
module Sched = Ir.Schedule
module CM = Ir.Cost_model
module Tn = Ir.Tnode
open Verdict.Claims

let nk = 512
and nm = 128
and nn = 256

let elt = 4
let node = Ll_test.node_factory ~first_id:125300 ~dims:[| 1 |] ()
let dy = node ~dims:[| nk; nm |] "cmr_dy"
let x = node ~dims:[| nk; nn |] "cmr_x"
let dw = node ~dims:[| nm; nn |] "cmr_dw"
let () = List.iter [ dy; x; dw ] ~f:Ll_test.materialize
let it = Ll_test.iter

(* [dw[o,i] += dy[k,o] * x[k,i]], loops (o, i, k): the contraction innermost, as the matmul
   detection reads it. No [Zero_out]: the default GPU preset's race analysis declines a kernel
   zeroing a materialized node (the pipeline expands such zeros into a segment of their own). *)
let wgrad () =
  let o = Ll_test.sym () and i = Ll_test.sym () and k = Ll_test.sym () in
  Ll_test.loop_n o nm
    (Ll_test.loop_n i nn
       (Ll_test.loop_n k nk
          (Ll_test.set dw
             [| it o; it i |]
             (LL.Ternop
                ( Ir.Ops.FMA,
                  (Ll_test.get dy [| it k; it o |], Ll_test.single),
                  (Ll_test.get x [| it k; it i |], Ll_test.single),
                  (Ll_test.get dw [| it o; it i |], Ll_test.single) )))))

(* A GPU-shaped device: workgroups of up to 1024 threads, 48 KiB of shared memory, 32-wide
   simdgroups. The capacity and residency are passed to the score explicitly below. *)
let limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    max_threads_per_workgroup = Some 1024;
    max_workgroup_memory_bytes = Some 49152;
    max_workgroup_dims = Some (1024, 1024, 64);
    max_grid_yz = Some 65535;
    simdgroup_width = Some 32;
  }

let peak_flops = 1e15
and peak_memory_bandwidth = 1e11

let resident = 8192

(* The within-capacity tier: a cache 10x faster than DRAM, as the HIP boxes measured. *)
let cache = 10. *. peak_memory_bandwidth
let base = Ll_test.optimize ~name:"cmr_wgrad" (wgrad ())

let hermetic (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

let scheduled sched = Sched.apply sched (hermetic base)
let untiled = scheduled (Sched.default_gpu ~limits (hermetic base))

let requested (o : LL.optimized) =
  let r = CM.reuse_traffic ~max_resident_threads:resident ~is_global:(CM.global_in o) o.LL.llc in
  let of_ tn = List.Assoc.find r.CM.ru_requested tn ~equal:Tn.equal |> Option.value ~default:0 in
  (r, of_ dy + of_ x)

let rank ?cache_capacity_bytes ?cache_bandwidth (o : LL.optimized) =
  CM.ranking_seconds ~peak_flops ~peak_memory_bandwidth ?cache_capacity_bytes ?cache_bandwidth
    ~max_resident_threads:resident ~is_global:(CM.global_in o) o.LL.llc

let bound_of (o : LL.optimized) =
  let s = CM.analyze o.LL.llc in
  CM.roofline_seconds ~peak_flops ~peak_memory_bandwidth ~flops:s.CM.flops ~bytes:(CM.total_bytes s)
    ()

let floor_bound =
  let f = CM.completion_floor base.LL.llc in
  CM.roofline_seconds ~peak_flops ~peak_memory_bandwidth ~flops:f.CM.fr_flops ~bytes:f.CM.fr_bytes
    ()

let mnk = nm * nn * nk * elt
let dims a = String.concat ~sep:"x" (Array.to_list (Array.map a ~f:Int.to_string))
let le a b = match (a, b) with Some a, Some b -> Float.(a <= b) | _ -> false

let () =
  let launch = LL.launch_dims untiled.LL.llc in
  let wg = Array.fold launch.LL.block ~init:1 ~f:( * ) in
  let r, req = requested untiled in
  Stdio.eprintf "untiled: grid=%s block=%s requested=%d working_set=%d (not part of the golden)\n"
    (dims launch.LL.grid) (dims launch.LL.block) req r.CM.ru_working_set_bytes;
  p "the default plan is untiled: a Grid over the rows and a workgroup over the columns"
    (launch.LL.grid.(0) > 1 && wg > 1);
  p "untiled: the operands' requested bytes are at least M*N*K*elt" (req >= mnk);
  p "untiled: the operands' requested bytes are at most M*N*K*elt*(1 + 1/wg)"
    (req * wg <= mnk * (wg + 1))

(* The unfused scalar blocktile seeds: the numerics-preserving candidates the seeded default and the
   [auto] gate choose among. *)
let leaves = Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits base

let scalar_blocktiles =
  List.filter leaves ~f:(fun q ->
      q.Autotune.sk_gpu && (not q.Autotune.sk_mma) && (not q.Autotune.sk_epilogue)
      && Option.is_none q.Autotune.sk_coalesce
      && Option.is_none q.Autotune.sk_swizzle
      && q.Autotune.sk_depth = 1)

let applied q =
  match scheduled (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:q (hermetic base)) with
  | o -> Some o
  | exception (Invalid_argument _ | Ir.Schedule_outcome.Cause_at _) -> None

let () =
  let exact =
    List.filter_map scalar_blocktiles ~f:(fun q ->
        Option.map (applied q) ~f:(fun o ->
            let _, req = requested o in
            let want = (mnk / q.Autotune.sk_bm) + (mnk / q.Autotune.sk_bn) in
            Stdio.eprintf "blocktile %s: requested=%d want=%d (not part of the golden)\n"
              (Autotune.sketch_params_label q) req want;
            req = want))
  in
  p_all ~min:(List.length scalar_blocktiles)
    "every scalar blocktile requests exactly M*N*K*elt*(1/bm + 1/bn) of its operands" exact ~f:Fn.id

let () =
  let r, _ = requested untiled in
  let ws = r.CM.ru_working_set_bytes in
  p "the untiled plan's working set is positive" (ws > 0);
  let at_bound o c =
    match (rank ?cache_capacity_bytes:c o, bound_of o) with
    | Some a, Some b -> Float.equal a b
    | _ -> false
  in
  p "with no capacity the score is analyze's roofline" (at_bound untiled None);
  p "at capacity (working set = capacity) the score is analyze's roofline"
    (at_bound untiled (Some ws));
  p "past capacity the untiled score exceeds analyze's roofline"
    (match (rank ~cache_capacity_bytes:(ws - 1) untiled, bound_of untiled) with
    | Some a, Some b -> Float.(a > b)
    | _ -> false);
  let capacities = [ 0; ws / 4; ws / 2; ws - 1; ws; 2 * ws; Int.max_value / 2 ] in
  let plans = untiled :: List.filter_map scalar_blocktiles ~f:applied in
  p_all "raising the capacity never raises the score, with or without a cache tier, on every plan"
    (List.concat_map plans ~f:(fun o ->
         List.map
           (List.zip_exn (List.drop_last_exn capacities) (List.tl_exn capacities))
           ~f:(fun (lo, hi) -> (o, lo, hi))))
    ~f:(fun (o, lo, hi) ->
      le (rank ~cache_capacity_bytes:hi o) (rank ~cache_capacity_bytes:lo o)
      && le
           (rank ~cache_capacity_bytes:hi ~cache_bandwidth:cache o)
           (rank ~cache_capacity_bytes:lo ~cache_bandwidth:cache o));
  p "past capacity some scalar blocktile ranks strictly below the untiled plan"
    (List.exists (List.filter_map scalar_blocktiles ~f:applied) ~f:(fun o ->
         match (rank ~cache_capacity_bytes:0 o, rank ~cache_capacity_bytes:0 untiled) with
         | Some a, Some b -> Float.(a < b)
         | _ -> false));
  p_none ~min:(List.length scalar_blocktiles)
    "with no capacity no scalar blocktile ranks strictly below the untiled plan"
    (List.filter_map scalar_blocktiles ~f:applied) ~f:(fun o ->
      match (rank o, rank untiled) with Some a, Some b -> Float.(a < b) | _ -> false)

let () =
  (* Every leaf of the family, tensorized and twin flavors included, at capacity 0 -- the charge on
     wherever it can be -- and the default plan. *)
  let plans = untiled :: List.filter_map leaves ~f:applied in
  p_all
    ~min:(1 + List.length scalar_blocktiles)
    "the bound is at most analyze's roofline is at most the score, on every leaf" plans
    ~f:(fun o ->
      le floor_bound (bound_of o)
      && le (bound_of o) (rank ~cache_capacity_bytes:0 o)
      && le (bound_of o) (rank ~cache_bandwidth:cache o)
      && le (bound_of o) (rank o))

let () =
  (* The two tiers (the wave decision of 2026-10-18): within capacity, cross-block re-reads are
     charged at the cache bandwidth -- not for free, which is what tied the untiled and the seeded
     plans at batch 8 while the seeded one measured faster on both HIP boxes. *)
  let blocktiles = List.filter_map scalar_blocktiles ~f:applied in
  let huge = Int.max_value / 2 in
  let within ?(cb = cache) o = rank ~cache_capacity_bytes:huge ~cache_bandwidth:cb o in
  p "within capacity, the cache tier charges the untiled plan above analyze's roofline"
    (match (within untiled, bound_of untiled) with Some a, Some b -> Float.(a > b) | _ -> false);
  p "within capacity, the cache tier charges less than the DRAM tier past it"
    (match (within untiled, rank ~cache_capacity_bytes:0 ~cache_bandwidth:cache untiled) with
    | Some a, Some b -> Float.(a < b)
    | _ -> false);
  p "within capacity, some scalar blocktile ranks strictly below the untiled plan"
    (List.exists blocktiles ~f:(fun o ->
         match (within o, within untiled) with Some a, Some b -> Float.(a < b) | _ -> false));
  p_all
    ~min:(1 + List.length scalar_blocktiles)
    "raising the cache bandwidth never raises the score, on every plan" (untiled :: blocktiles)
    ~f:(fun o -> le (within ~cb:(2. *. cache) o) (within o));
  p "a cache bandwidth below DRAM's is read as DRAM's"
    (match (within ~cb:1. untiled, within ~cb:peak_memory_bandwidth untiled) with
    | Some a, Some b -> Float.equal a b
    | _ -> false)

let () =
  (* Residency comes from the kernel's ONE launch block, not a nest's own workgroup: a kernel of a
     256-thread nest and a 32-thread sibling launches 256-thread blocks for both, so the sibling has
     [resident / 256] blocks resident, not [resident / 32]. Hand-annotated, nothing scheduled. *)
  let z = node ~dims:[| 256 |] "cmr_z" and w = node ~dims:[| 256 |] "cmr_w" in
  let dwb = node ~dims:[| 128; 32 |] "cmr_dwb" in
  List.iter [ z; w; dwb ] ~f:Ll_test.materialize;
  let g = Ll_test.sym () and t = Ll_test.sym () and g' = Ll_test.sym () and t' = Ll_test.sym () in
  let k = Ll_test.sym () in
  let wide =
    Ll_test.loop_n ~axis:LL.Grid g' 1
      (Ll_test.loop_n ~axis:LL.Workgroup t' 256
         (Ll_test.set z [| it t' |] (Ll_test.get w [| it t' |])))
  in
  let narrow =
    Ll_test.loop_n ~axis:LL.Grid g 128
      (Ll_test.loop_n ~axis:LL.Workgroup t 32
         (Ll_test.loop_n k nk
            (Ll_test.set dwb
               [| it g; it t |]
               (Ll_test.add (Ll_test.get dwb [| it g; it t |]) (Ll_test.get x [| it k; it t |])))))
  in
  let ws code =
    (CM.reuse_traffic ~max_resident_threads:resident ~is_global:(fun _ -> true) code)
      .CM.ru_working_set_bytes
  in
  (* Per narrow block: x's 32 columns over the k rows, and the block's 32 accumulator cells. *)
  let per_block = ((nk * 32) + 32) * elt in
  p "a 32-thread nest launched alone keeps resident / 32 of its blocks"
    (ws narrow = per_block * min 128 (resident / 32));
  p "beside a 256-thread sibling, it keeps only resident / 256 (the launch block's)"
    (ws (Ll_test.seq wide narrow) = per_block * min 128 (resident / 256))

(* {2 The score's own rates and memory spaces (gh-ocannl-1442)}

   Ceilings shaped like CUDA's advisories ([peak_flops] 1.5e13, [peak_memory_bandwidth] 8e12) beside
   realistic rates (an issue rate of 3.4e12, a DRAM rate of 5.8e11, a cache tier of 1.8e12), within
   capacity. Not a calibration: numbers in the regime the backends' ceilings and rog's measurements
   put the score in. *)
let c_flops = 1.5e13
and c_bw = 8e12
and r_issue = 3.4e12
and r_dram = 5.8e11
and r_cache = 1.8e12

let huge = Int.max_value / 2

let rated ?issue_rate ?dram_bandwidth ?(cache_bandwidth = r_cache) (o : LL.optimized) =
  CM.ranking_seconds ~peak_flops:c_flops ~peak_memory_bandwidth:c_bw ~cache_capacity_bytes:huge
    ~cache_bandwidth ?issue_rate ?dram_bandwidth ~max_resident_threads:resident
    ~is_global:(CM.global_in o) o.LL.llc

let legs ?issue_rate ?dram_bandwidth ?(scratch = true) (o : LL.optimized) =
  let s = CM.analyze o.LL.llc in
  let r = CM.reuse_traffic ~max_resident_threads:resident ~is_global:(CM.global_in o) o.LL.llc in
  let r = if scratch then r else { r with CM.ru_scratch = [] } in
  CM.ranking_legs_of ~peak_flops:c_flops ~peak_memory_bandwidth:c_bw ~cache_capacity_bytes:huge
    ~cache_bandwidth:r_cache ?issue_rate ?dram_bandwidth s (Some r)

let c_bound (o : LL.optimized) =
  let s = CM.analyze o.LL.llc in
  CM.roofline_seconds ~peak_flops:c_flops ~peak_memory_bandwidth:c_bw ~flops:s.CM.flops
    ~bytes:(CM.total_bytes s) ()

let () =
  let blocktiles = List.filter_map scalar_blocktiles ~f:applied in
  let plans = untiled :: List.filter_map leaves ~f:applied in
  let eq a b = Option.equal Float.equal a b in
  p_all
    ~min:(1 + List.length scalar_blocktiles)
    "a rate at or above its ceiling reads as the ceiling: the score is the one without it, on \
     every leaf"
    plans
    ~f:(fun o ->
      eq (rated ~issue_rate:c_flops ~dram_bandwidth:c_bw o) (rated o)
      && eq (rated ~issue_rate:(4. *. c_flops) ~dram_bandwidth:(4. *. c_bw) o) (rated o));
  p_all
    ~min:(1 + List.length scalar_blocktiles)
    "raising the issue rate or the DRAM rate never raises the score, on every leaf" plans
    ~f:(fun o ->
      le
        (rated ~issue_rate:(2. *. r_issue) ~dram_bandwidth:r_dram o)
        (rated ~issue_rate:r_issue ~dram_bandwidth:r_dram o)
      && le
           (rated ~issue_rate:r_issue ~dram_bandwidth:(2. *. r_dram) o)
           (rated ~issue_rate:r_issue ~dram_bandwidth:r_dram o));
  p_all
    ~min:(1 + List.length scalar_blocktiles)
    "with the score's own rates, analyze's roofline under the ceilings is at most the score, on \
     every leaf"
    plans
    ~f:(fun o -> le (c_bound o) (rated ~issue_rate:r_issue ~dram_bandwidth:r_dram o));
  (* Why the two rates come together: priced at a realistic issue rate but the ceiling's DRAM, the
     untiled plan reads compute-bound, and no sketch -- the same operations plus its staging -- can
     be strictly below it; with the achieved DRAM rate too, the re-reads it pays for are back. *)
  let below ?issue_rate ?dram_bandwidth o =
    match (rated ?issue_rate ?dram_bandwidth o, rated ?issue_rate ?dram_bandwidth untiled) with
    | Some a, Some b -> Float.(a < b)
    | _ -> false
  in
  p_exists "under the ceilings, some scalar blocktile ranks strictly below the untiled plan"
    blocktiles ~f:(fun o -> below o);
  p_none ~min:1
    "an issue rate with the ceiling's DRAM leaves no scalar blocktile strictly below the untiled \
     plan"
    blocktiles ~f:(below ~issue_rate:r_issue);
  p_exists "with both rates, some scalar blocktile ranks strictly below the untiled plan" blocktiles
    ~f:(below ~issue_rate:r_issue ~dram_bandwidth:r_dram);
  (* Memory spaces: what the reuse model keeps off the DRAM leg is exactly the chip-resident nodes,
     and every global node's distinct bytes, writes included, stay on it. *)
  let globals_bytes (o : LL.optimized) =
    List.sum
      (module Int)
      (CM.analyze o.LL.llc).CM.per_node
      ~f:(fun (tn, fp) ->
        if CM.global_in o tn then fp.CM.fp_read_bytes + fp.CM.fp_write_bytes else 0)
  in
  p_all
    ~min:(1 + List.length scalar_blocktiles)
    "every leaf's memory leg is at least its global nodes' distinct bytes, writes included, at the \
     DRAM rate"
    plans
    ~f:(fun o ->
      match snd (legs ~dram_bandwidth:r_dram o) with
      | Some m -> Float.(m >= Float.of_int (globals_bytes o) /. r_dram)
      | None -> false);
  let scratch_of o =
    (CM.reuse_traffic ~max_resident_threads:resident ~is_global:(CM.global_in o) o.LL.llc)
      .CM.ru_scratch
  in
  p_all ~min:1 "the reuse extraction lists exactly the nodes no global placement holds" plans
    ~f:(fun o ->
      let off =
        List.filter_map (CM.analyze o.LL.llc).CM.per_node ~f:(fun (tn, _) ->
            if CM.global_in o tn then None else Some tn)
      in
      List.equal Tn.equal
        (List.sort ~compare:Tn.compare off)
        (List.sort ~compare:Tn.compare (scratch_of o)));
  let staged = List.filter blocktiles ~f:(fun o -> not (List.is_empty (scratch_of o))) in
  p_all ~min:1
    "a staged blocktile's tiles are off the DRAM leg: its memory leg is below the same extraction \
     charging them"
    staged ~f:(fun o ->
      match
        (snd (legs ~dram_bandwidth:r_dram o), snd (legs ~dram_bandwidth:r_dram ~scratch:false o))
      with
      | Some a, Some b -> Float.(a < b)
      | _ -> false);
  (* The compute leg on a hand-made summary: tile-MMA work stays at the ceiling, scalar work moves
     with the issue rate. *)
  let summary ~flops ~mma_flops =
    {
      CM.per_node = [];
      read_bytes = 0;
      write_bytes = 0;
      flops;
      mma_flops;
      flops_approx = false;
      opaque = false;
    }
  in
  let compute ?issue_rate s = fst (CM.ranking_legs_of ~peak_flops:c_flops ?issue_rate s None) in
  p "an all-tile-MMA kernel's compute leg does not depend on the issue rate"
    (eq
       (compute ~issue_rate:r_issue (summary ~flops:1_000_000 ~mma_flops:1_000_000))
       (compute (summary ~flops:1_000_000 ~mma_flops:1_000_000)));
  p "an all-scalar kernel's compute leg is its operations at the issue rate"
    (eq
       (compute ~issue_rate:r_issue (summary ~flops:1_000_000 ~mma_flops:0))
       (Some (1e6 /. r_issue)))

(* The staffed start re-decides each segment of the search's own segmentation under the gate, so its
   tag -- which joins the search trajectory's identity -- names every reading those decisions rest
   on: two gates differing only in the issue rate, or only in the DRAM rate, tag apart. *)
let () =
  let rated_limits = { limits with peak_flops = Some c_flops; peak_memory_bandwidth = Some c_bw } in
  let tag ?scalar_rate ?dram_rate () =
    Autotune.seeded_plan
      ~seed_gate:(Autotune.Ranked { capacity = Some 0; cache_rate = None; scalar_rate; dram_rate })
      ~backend:"cuda" ~limits:rated_limits ~accum_prec:Fn.id ~static_indices:[] (hermetic base)
    |> Option.bind ~f:(fun plan -> plan.Autotune.sp_start)
    |> Option.map ~f:(fun start -> start.Autotune.gate_tag)
  in
  let apart a b = match (a, b) with Some a, Some b -> not (String.equal a b) | _ -> false in
  p "the staffed start's gate tag tells two issue rates apart"
    (* At the achieved DRAM rate: an issue rate under the ceiling's DRAM seeds nothing here (above),
       and a plan that seeds nothing has no start to tag. *)
    (apart (tag ~dram_rate:r_dram ()) (tag ~scalar_rate:r_issue ~dram_rate:r_dram ()));
  p "the staffed start's gate tag tells two DRAM rates apart"
    (apart (tag ()) (tag ~dram_rate:r_dram ()))

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
   roofline; raising the capacity never raises the score; and on every leaf of the site's GPU sketch
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

let rank ?cache_capacity_bytes (o : LL.optimized) =
  CM.ranking_seconds ~peak_flops ~peak_memory_bandwidth ?cache_capacity_bytes
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
      && (not q.Autotune.sk_coalesce)
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
  p_all "raising the capacity never raises the score, on every plan"
    (List.concat_map plans ~f:(fun o ->
         List.map
           (List.zip_exn (List.drop_last_exn capacities) (List.tl_exn capacities))
           ~f:(fun (lo, hi) -> (o, lo, hi))))
    ~f:(fun (o, lo, hi) -> le (rank ~cache_capacity_bytes:hi o) (rank ~cache_capacity_bytes:lo o));
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
      && le (bound_of o) (rank o))

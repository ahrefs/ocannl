(* gh-ocannl-1479: the ranking score's launch fill.

   The compute leg used to divide a kernel's scalar operations by one issue rate, so every scalar
   leaf of a site scored the same whatever its launch: a plain leaf, its batch loops serial inside a
   few blocks, tied its [bgrid] twin, and measured 3-36x slower (gh-ocannl-1278). Each op-carrying
   statement's work is now attributed to its parallel context -- [B] blocks (its enclosing [Grid]
   loops) of [T] threads (its [Workgroup] loops) -- and priced at the issue rate times [min 1 (B *
   min T t / (P * t))].

   First on hand-built kernels, at a stand-in fill model of P = 4 multiprocessors saturating at t =
   64 threads and power-of-two rates, so every value compared is exact: the contexts and fills of a
   grid-bound nest, its serial-batch twin, a wide zero nest beside a one-block contraction (no work,
   no say: a kernel-wide launch reading would call that kernel full), a block wider than t, two
   compute nests (the leg sums per context), and accepted vs declined tile-MMA work (both filled).
   The score never drops below the roofline bound, and a fill model under which every context is
   full scores bitwise what no fill model does.

   Then at the q/k/v projection of gpt2_mini (b8), lowered here and scheduled by every scalar
   blocktile leaf of the GPU matmul family under a CUDA-like stand-in record (hermetic: nothing is
   compiled, so the claims run on every backend): with the record's launch-fill constants each plain
   leaf ranks strictly above its [bgrid] twin and the best-ranked leaf is a [bgrid] one; without
   them the twins tie, as before. Ranks go to stderr: device constants' quotients. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module CM = Ir.Cost_model
module BI = Ir.Backend_intf

let fresh_tn =
  let make = Ll_test.node_factory ~first_id:971_000_000 ~dims:[||] () in
  fun label dims -> make ~dims label

let it = Ll_test.iter
let fm = { CM.fm_multiprocessors = 4; fm_fill_threads = 64 }

(* [d[...] += x[...] * y[...]]: two operations per cell. *)
let mac d didx x xidx y yidx =
  Ll_test.set d didx
    (Ll_test.add (Ll_test.get d didx) (Ll_test.mul (Ll_test.get x xidx) (Ll_test.get y yidx)))

(* A batched product's nest: [nb] batch rows, [nr] blocks of [nw] threads, a serial reduction of
   [nk]; [batch_axis] binds the batch loop to the grid ([bgrid]) or leaves it serial (plain). *)
let nest ~tag ~batch_axis ~nb ~nr ~nw ~nk =
  let d = fresh_tn (tag ^ "_d") [| nb; nr; nw |]
  and x = fresh_tn (tag ^ "_x") [| nb; nr; nk |]
  and y = fresh_tn (tag ^ "_y") [| nk; nw |] in
  let b = Ll_test.sym () and r = Ll_test.sym () and w = Ll_test.sym () and k = Ll_test.sym () in
  let body =
    Ll_test.loop_n k nk (mac d [| it b; it r; it w |] x [| it b; it r; it k |] y [| it k; it w |])
  in
  match batch_axis with
  | LL.Grid ->
      Ll_test.loop_n ~axis:LL.Grid b nb
        (Ll_test.loop_n ~axis:LL.Grid r nr (Ll_test.loop_n ~axis:LL.Workgroup w nw body))
  | _ ->
      Ll_test.loop_n ~axis:LL.Grid r nr
        (Ll_test.loop_n ~axis:LL.Workgroup w nw (Ll_test.loop_n b nb body))

let ctx_list code =
  List.map (CM.launch_contexts code) ~f:(fun c ->
      (c.CM.lc_blocks, c.CM.lc_threads, c.CM.lc_tile_mma, c.CM.lc_ops))

let peak = 1024. (* operations/s and bytes/s: powers of two keep the quotients exact *)
let issue = 256.

let compute ?launch ?(mma_flops = 0) code =
  fst
    (CM.ranking_legs_of ~peak_flops:peak ~peak_memory_bandwidth:peak ~issue_rate:issue ~mma_flops
       ?launch (CM.analyze code) None)

let filled ?mma_flops code = compute ?mma_flops ~launch:(fm, CM.launch_contexts code) code

let () =
  let grid = nest ~tag:"cf_g" ~batch_axis:LL.Grid ~nb:2 ~nr:2 ~nw:64 ~nk:8 in
  let plain = nest ~tag:"cf_p" ~batch_axis:LL.Serial ~nb:2 ~nr:2 ~nw:64 ~nk:8 in
  let ops = 2 * 2 * 2 * 64 * 8 in
  p "the grid-bound nest is one context of 4 blocks of 64 threads holding all its operations"
    (List.equal Poly.equal (ctx_list grid) [ (4, 64, false, ops) ]);
  p "the serial-batch twin is one context of 2 blocks: its batch loop adds work, not blocks"
    (List.equal Poly.equal (ctx_list plain) [ (2, 64, false, ops) ]);
  p "4 blocks of 64 threads fill P=4 at t=64; 2 blocks fill half"
    (Float.equal (CM.launch_fill fm (CM.launch_contexts grid)) 1.
    && Float.equal (CM.launch_fill fm (CM.launch_contexts plain)) 0.5);
  p "without a fill model the twins tie on the compute leg"
    (Option.equal Float.equal (compute grid) (compute plain));
  p "with it the serial-batch twin's compute leg is exactly twice the grid-bound one's"
    (match (filled grid, filled plain) with
    | Some g, Some pl -> Float.(g = of_int ops /. issue) && Float.(pl = 2. *. g)
    | _ -> false);
  p "a fill model under which every context is full scores bitwise what no fill model does"
    (Option.equal Float.equal (filled grid) (compute grid))

let () =
  (* A wide zero nest -- 16 blocks of 64 threads, no operations -- beside a one-block
     contraction. *)
  let z = fresh_tn "cf_z" [| 16; 64 |] in
  let zb = Ll_test.sym () and zw = Ll_test.sym () in
  let zero =
    Ll_test.loop_n ~axis:LL.Grid zb 16
      (Ll_test.loop_n ~axis:LL.Workgroup zw 64 (Ll_test.set z [| it zb; it zw |] (Ll_test.c 0.)))
  in
  let narrow = nest ~tag:"cf_n" ~batch_axis:LL.Serial ~nb:2 ~nr:1 ~nw:64 ~nk:8 in
  let kernel = Ll_test.seq zero narrow in
  p "the zero nest contributes no context: only the contraction's 1 block of 64 threads"
    (List.equal Poly.equal (ctx_list kernel) [ (1, 64, false, 2 * 2 * 64 * 8) ]);
  p "so the kernel's fill is the contraction's quarter, though its launch spans 16 blocks"
    (Float.equal (CM.launch_fill fm (CM.launch_contexts kernel)) 0.25
    && Array.fold (LL.launch_dims kernel).LL.grid ~init:1 ~f:( * ) >= 16)

let () =
  (* Blocks wider than t: each counts one multiprocessor's saturating share. *)
  let wide nr =
    nest ~tag:(Printf.sprintf "cf_w%d" nr) ~batch_axis:LL.Grid ~nb:1 ~nr ~nw:128 ~nk:4
  in
  p
    "2 blocks of 128 threads fill half of P=4 at t=64 (not all of it: a block is on one \
     multiprocessor)"
    (Float.equal (CM.launch_fill fm (CM.launch_contexts (wide 2))) 0.5);
  p "8 blocks of 128 threads fill it"
    (Float.equal (CM.launch_fill fm (CM.launch_contexts (wide 8))) 1.)

let () =
  (* Two compute nests of equal work, one full and one at a quarter: the leg sums per context. *)
  let full = nest ~tag:"cf_f" ~batch_axis:LL.Grid ~nb:2 ~nr:2 ~nw:64 ~nk:8 in
  let quarter = nest ~tag:"cf_q" ~batch_axis:LL.Serial ~nb:4 ~nr:1 ~nw:64 ~nk:8 in
  let both = Ll_test.seq full quarter in
  let ops = 2 * 2 * 2 * 64 * 8 in
  p "two nests of equal work, at fills 1 and 1/4, price as 1 + 4 times one nest's work"
    (match filled both with
    | Some leg -> Float.(leg = (of_int ops +. (4. *. of_int ops)) /. issue)
    | None -> false)

let () =
  (* A tile-MMA statement inside one block of one lane: accepted work runs at the ceiling, declined
     work (its scalar fallback) at the issue rate, each times the fill of its context. *)
  let d = fresh_tn "cf_md" [| 2; 2 |]
  and a = fresh_tn "cf_ma" [| 2; 2 |]
  and b = fresh_tn "cf_mb" [| 2; 2 |] in
  let g = Ll_test.sym () in
  let at tn = (tn, [| Ll_test.fixed 0; Ll_test.fixed 0 |]) in
  let mma =
    Ll_test.loop_n ~axis:LL.Grid g 1 (Ll_test.tile_mma ~d:(at d) ~a:(at a) ~b:(at b) LL.Noop)
  in
  let tile_ops = (CM.analyze mma).CM.flops in
  let fill = CM.context_fill fm ~blocks:1 ~threads:1 in
  p "the tile-MMA statement is one tile context of 1 block"
    (List.equal Poly.equal (ctx_list mma) [ (1, 1, true, tile_ops) ]);
  p "accepted tile-MMA work prices at the ceiling times its context's fill"
    (Option.equal Float.equal (filled ~mma_flops:tile_ops mma)
       (Some (Float.of_int tile_ops /. fill /. peak)));
  p "declined tile-MMA work prices at the issue rate times its context's fill"
    (Option.equal Float.equal (filled mma) (Some (Float.of_int tile_ops /. fill /. issue)))

let () =
  (* Never below the bound: the compute leg is at least the operations at the ceiling. *)
  let kernels =
    [
      nest ~tag:"cf_b1" ~batch_axis:LL.Serial ~nb:8 ~nr:1 ~nw:16 ~nk:4;
      nest ~tag:"cf_b2" ~batch_axis:LL.Grid ~nb:8 ~nr:4 ~nw:64 ~nk:4;
    ]
  in
  p_all ~min:2 "the filled compute leg is never below the roofline's compute leg" kernels
    ~f:(fun k ->
      match filled k with
      | Some leg -> Float.(leg >= of_int (CM.analyze k).CM.flops /. peak)
      | None -> false)

(* The q/k/v projection at b8, scored per scalar blocktile leaf under a CUDA-like record. *)
let stand_in =
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
    max_resident_threads = Some (70 * 1536);
    issue_rate = Some 3.4e12;
    dram_bandwidth = Some 6.0e11;
    multiprocessors = Some 70;
    fill_threads = Some 128;
  }

let without_fill = { stand_in with multiprocessors = None; fill_threads = None }

let () =
  let ctx = Context.cpu () in
  let cell idx = Float.of_int (1 + (Array.fold idx ~init:0 ~f:( + ) % 7)) *. 0.125 in
  let w = NTDSL.init ~l:"cf_wqkv" ~prec:Ir.Ops.single ~o:[ 8; 32 ] ~i:[ 256 ] ~f:cell () in
  let x = NTDSL.init ~l:"cf_xqkv" ~prec:Ir.Ops.single ~b:[ 8; 128 ] ~o:[ 256 ] ~f:cell () in
  let%op d = w * x in
  Train.set_materialized d.Tensor.value;
  let lowered =
    Context.lowered_for_decisions ~name:"cf_qkv" ctx (Train.forward d) Ir.Indexing.Empty
  in
  let hermetic (o : LL.optimized) =
    {
      o with
      LL.traced_store = Hashtbl.copy o.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
    }
  in
  let leaves =
    match Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:stand_in lowered with
    | None -> []
    | Some tree -> List.filter (Ir.Schedule_space.leaves tree) ~f:Autotune.scalar_blocktile
  in
  let scored =
    List.filter_map leaves ~f:(fun p ->
        match
          Ir.Schedule.apply
            (Autotune.sketch_schedule ~accum_prec:Fn.id ~p (hermetic lowered))
            (hermetic lowered)
        with
        | exception _ -> None
        | post -> (
            match
              ( Autotune.ranking_ms ~limits:stand_in [ post ],
                Autotune.ranking_ms ~limits:without_fill [ post ] )
            with
            | Some (rank, bound), Some (rank_off, _) -> Some (p, rank, bound, rank_off)
            | _ -> None))
  in
  List.iter scored ~f:(fun (p, rank, bound, off) ->
      Stdio.eprintf
        "cost_model_fill: %s rank %.6f ms (no fill %.6f ms, bound %.6f ms) (not part of the golden)\n"
        (Autotune.sketch_params_label p) rank off bound);
  let twins =
    List.filter_map scored ~f:(fun ((p, _, _, _) as plain) ->
        if p.Autotune.sk_batch_grid then None
        else
          List.find scored ~f:(fun (q, _, _, _) ->
              q.Autotune.sk_batch_grid
              && Sexp.equal
                   (Autotune.sexp_of_sketch_params { q with sk_batch_grid = false })
                   (Autotune.sexp_of_sketch_params p))
          |> Option.map ~f:(fun grid -> (plain, grid)))
  in
  p_all ~min:2 "with the fill constants every plain leaf ranks strictly above its bgrid twin" twins
    ~f:(fun ((_, plain, _, _), (_, grid, _, _)) -> Float.(grid < plain));
  p_all ~min:2 "without them the twins tie, as before the launch fill" twins
    ~f:(fun ((_, _, _, plain_off), (_, _, _, grid_off)) -> Float.equal plain_off grid_off);
  p_all ~min:2 "every leaf's rank is at least its bound" scored ~f:(fun (_, rank, bound, _) ->
      Float.(rank >= bound));
  p "the best-ranked scalar leaf is a bgrid one"
    (match List.min_elt scored ~compare:(fun (_, a, _, _) (_, b, _, _) -> Float.compare a b) with
    | Some (p, _, _, _) -> p.Autotune.sk_batch_grid
    | None -> false)

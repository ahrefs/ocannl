(* gh-ocannl-2262: the ranking score's register-reuse term and the launch fill in lane groups.

   At one block tile, 2x2 and 4x4 register tiles tie in every other term (same staged copies, same
   fill when saturated, same work), while they measure apart in both directions: 2x2 slower at a
   32x32 block tile (256 against 64 threads, 1.2x on rog, 1.8x on tuf), 4x4 slower at 16x16 (16
   threads against 64, 1.9x on rog). Two terms account for both. Per k-step a thread of an [rm x rn]
   tile reads [rm + rn] staged cells for [rm * rn] multiply-adds, and the loads beyond the 4x4
   calibration tile's reuse are priced at a shared-load rate; a 16-thread block issues in 32-lane
   groups, and saturated does half a full group's work.

   First on hand-built kernels, at power-of-two rates and a stand-in fill model so every value
   compared is exact: the census counts each distinct staged cell once per straight-line body (a
   materialized tile's [rm + rn] a k-step), again per iteration of a serial loop, nothing for
   unstaged reads or in a [Tile_mma] kernel; the excess over the reference is nothing at 4x4 and [rm
   + rn - 2 * rm * rn / 4] cells a step at 2x2; the term adds the filled excess over the rate to the
   compute leg exactly, and leaves both legs bitwise without a rate; the fill in lane groups is
   bitwise the thread-only fill at a lane multiple and below saturation, and [T / L] of it
   saturated, the tail included.

   Then at the attention out projection of gpt2_mini (b8), lowered here and scheduled by the scalar
   blocktile leaves of the GPU matmul family (hermetic, nothing compiles): each leaf's census equals
   the register-tile arithmetic [M * N * K * (rm + rn) / (rm * rn)]; and through [model_explain]
   under a CUDA-shaped target, each limits field moves exactly the leaves it should: the rate the
   2x2 leaves (and orders the 32x32 pair as measured), the lane width the saturated 16-thread leaves
   (and orders the 16x16 pair as measured), every other leaf bitwise, the unsaturated 16-thread ones
   included. *)

(* cc-fp-contract: not-executed -- the claims compare cost-model scores computed in OCaml *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module CM = Ir.Cost_model
module BI = Ir.Backend_intf

let fresh_tn =
  let make = Ll_test.node_factory ~first_id:973_000_000 ~dims:[||] () in
  fun label dims -> make ~dims label

let it = Ll_test.iter
let fx = Ll_test.fixed

let fm =
  {
    CM.fm_multiprocessors = 4;
    fm_fill_threads = 64;
    fm_tail = false;
    fm_chains = 1;
    fm_lane_width = 1;
  }

(* [nb] blocks of [nw] threads, [ns] serial k-steps; per step every thread accumulates an [rm x rn]
   register tile from the staged [ta] (rows) and [tb] (columns), materialized: [rm * rn] statements
   at fixed tile coordinates. With [serial_cols], the column coordinate is a serial loop instead. *)
let tile_kernel ?(serial_cols = false) ~tag ~nb ~nw ~ns ~rm ~rn () =
  let d = fresh_tn (tag ^ "_d") [| nb; nw; rm; rn |]
  and ta = fresh_tn (tag ^ "_ta") [| rm; ns |]
  and tb = fresh_tn (tag ^ "_tb") [| ns; rn |] in
  let b = Ll_test.sym () and w = Ll_test.sym () and s = Ll_test.sym () in
  let update ri rj =
    let idx = [| it b; it w; fx ri; rj |] in
    Ll_test.set d idx
      (Ll_test.add (Ll_test.get d idx)
         (Ll_test.mul (Ll_test.get ta [| fx ri; it s |]) (Ll_test.get tb [| it s; rj |])))
  in
  let rows rj = List.reduce_exn (List.init rm ~f:(fun ri -> update ri rj)) ~f:Ll_test.seq in
  let step =
    if serial_cols then
      let j = Ll_test.sym () in
      Ll_test.loop_n j rn (rows (it j))
    else List.reduce_exn (List.init rn ~f:(fun rj -> rows (fx rj))) ~f:Ll_test.seq
  in
  let code =
    Ll_test.loop_n ~axis:LL.Grid b nb
      (Ll_test.loop_n ~axis:LL.Workgroup w nw (Ll_test.loop_n s ns step))
  in
  (code, fun tn -> Ir.Tnode.equal tn ta || Ir.Tnode.equal tn tb)

let loads ctxs = List.map ctxs ~f:(fun l -> (l.CM.sl_blocks, l.CM.sl_threads, l.CM.sl_loads))

let () =
  let nb, nw, ns = (2, 4, 3) in
  let census ?serial_cols ~rm ~rn tag =
    let code, is_staged = tile_kernel ?serial_cols ~tag ~nb ~nw ~ns ~rm ~rn () in
    (code, is_staged, CM.shared_loads ~is_staged code)
  in
  let c22, s22, l22 = census ~rm:2 ~rn:2 "crt_22" and c44, s44, l44 = census ~rm:4 ~rn:4 "crt_44" in
  let steps = Float.of_int (nb * nw * ns) in
  p "a materialized 2x2 tile reads rm + rn = 4 distinct staged cells a k-step"
    (Poly.equal (loads l22) [ (nb, nw, steps *. 4.) ]);
  p "a materialized 4x4 tile reads rm + rn = 8 distinct staged cells a k-step"
    (Poly.equal (loads l44) [ (nb, nw, steps *. 8.) ]);
  let _, _, serial = census ~serial_cols:true ~rm:2 ~rn:2 "crt_22s" in
  (* Per column iteration: the 2 row cells and the 1 column cell, recounted for each of 2. *)
  p "a serial loop recounts its body's cells per iteration"
    (Poly.equal (loads serial) [ (nb, nw, steps *. 6.) ]);
  p_empty "unstaged reads are not shared loads" ~over:l22
    (CM.shared_loads ~is_staged:(fun _ -> false) c22);
  let at tn = (tn, [| fx 0; fx 0 |]) in
  let mma =
    Ll_test.tile_mma
      ~d:(at (fresh_tn "crt_m_d" [| 2; 2 |]))
      ~a:(at (fresh_tn "crt_m_a" [| 2; 2 |]))
      ~b:(at (fresh_tn "crt_m_b" [| 2; 2 |]))
      LL.Noop
  in
  p_empty "a Tile_mma kernel has no shared loads (its rate prices them)" ~over:l22
    (CM.shared_loads ~is_staged:s22 (Ll_test.seq c22 mma));
  p "the calibration tile's reference reuse is 1/4 cell per scalar operation"
    (Float.equal CM.reference_loads_per_op 0.25);
  let excess code is_staged =
    CM.excess_shared_loads (CM.launch_contexts code) (CM.shared_loads ~is_staged code)
  in
  p_empty "a 4x4 tile has no loads beyond the reference" ~over:l44 (excess c44 s44);
  (* 4 cells against 1/4 of 4 multiply-adds' 8 operations a step: 2 beyond. *)
  p "a 2x2 tile has 2 cells a k-step beyond the reference"
    (Poly.equal (loads (excess c22 s22)) [ (nb, nw, steps *. 2.) ]);
  (* The legs: the 2-by-4 context fills 8 / (4 * 64) = 1/32 of the device. *)
  let peak = 1024. and issue = 256. in
  let legs ?shared_loads ?shared_load_rate ?(fill = fm) code =
    CM.ranking_legs_of ~peak_flops:peak ~peak_memory_bandwidth:peak ~issue_rate:issue
      ~launch:(fill, CM.launch_contexts code)
      ?shared_loads ?shared_load_rate (CM.analyze code) None
  in
  let legs_equal (c1, m1) (c2, m2) =
    Option.equal Float.equal c1 c2 && Option.equal Float.equal m1 m2
  in
  let ex22 = excess c22 s22 in
  let base = legs c22 in
  let priced = legs ~shared_loads:ex22 ~shared_load_rate:4. c22 in
  (* 48 excess cells over fill 1/32, at 4 cells/s: 384 s. *)
  p "the term adds the filled excess over the rate to the compute leg, exactly"
    (match (base, priced) with
    | (Some c0, m0), (Some c1, m1) -> Float.equal c1 (c0 +. 384.) && Option.equal Float.equal m0 m1
    | _ -> false);
  p "no rate leaves both legs bitwise" (legs_equal (legs ~shared_loads:ex22 c22) base);
  p "a rate of 0 leaves both legs bitwise"
    (legs_equal (legs ~shared_loads:ex22 ~shared_load_rate:0. c22) base);
  p "no excess leaves both legs bitwise"
    (legs_equal (legs ~shared_loads:[] ~shared_load_rate:4. c22) base);
  (* At one block tile of [nw * 4] outputs a thread, the 2x2 kernel has four times the 4x4 one's
     threads: same work. Both saturated (fill 1), the 4x4 leg is its work at the issue rate, the 2x2
     leg that plus the excess: the pair's ratio is [1 + excess / (rate * work / issue)]. *)
  let sat = { fm with CM.fm_fill_threads = 1; fm_multiprocessors = 1 } in
  (* [nw = 4] threads of 2x2 cover the one 4x4 thread's 16 outputs. *)
  let k22 = c22 and k44, z44 = tile_kernel ~tag:"crt_p44" ~nb ~nw:1 ~ns ~rm:4 ~rn:4 () in
  let leg ?shared_load_rate code is_staged =
    fst (legs ~fill:sat ~shared_loads:(excess code is_staged) ?shared_load_rate code)
  in
  p "unpriced, the 2x2 and 4x4 tiles at one block tile tie"
    (Option.equal Float.equal (leg k22 s22) (leg k44 z44));
  p "priced, the 2x2 tile ranks above the 4x4 one by exactly its excess over the rate"
    (match (leg ~shared_load_rate:4. k22 s22, leg ~shared_load_rate:4. k44 z44) with
    | Some c22, Some c44 -> Float.equal c22 (c44 +. (steps *. 2. /. 4.))
    | _ -> false)

(* The fill in lane groups. *)
let () =
  let lanes = { fm with CM.fm_lane_width = 32; fm_fill_threads = 256; fm_multiprocessors = 8 } in
  let threads_only = { lanes with fm_lane_width = 1 } in
  let both ?chains ~blocks ~threads () =
    ( CM.context_fill ?chains lanes ~blocks ~threads,
      CM.context_fill ?chains threads_only ~blocks ~threads )
  in
  p_all ~min:8 "a lane multiple fills bitwise as the thread-only fill, saturated or not"
    [ (1, 32); (4, 64); (64, 64); (1000, 256); (3, 96); (8, 512); (100, 32); (17, 128) ]
    ~f:(fun (blocks, threads) ->
      let l, t = both ~blocks ~threads () in
      Float.equal l t);
  (* 16 threads in 32 lanes: 8 blocks issue 256 lanes of the device's 2048. *)
  p "unsaturated, a sub-group block fills as the thread-only fill"
    (let l, t = both ~blocks:8 ~threads:16 () in
     Float.equal l t && Float.equal t (128. /. 2048.));
  p "saturated, a 16-thread block does half the thread-only fill's work"
    (let l, t = both ~blocks:4096 ~threads:16 () in
     Float.equal t 1. && Float.equal l 0.5);
  p "saturated, a 48-thread block does 48 of its 64 lanes' work"
    (let l, _ = both ~blocks:4096 ~threads:48 () in
     Float.equal l 0.75);
  (* Under the tail, 12 blocks of 16 lanes over 8 multiprocessors of 256: the busiest runs 2 of
     them, 64 lanes of 256 -- unsaturated, the thread-only fill. With 16 threads a block saturating
     its multiprocessor (256 * 16 blocks over 8), half. *)
  let tail = { lanes with fm_tail = true } and tail1 = { threads_only with fm_tail = true } in
  p "under the tail, unsaturated, a sub-group block fills as the thread-only fill"
    (Float.equal
       (CM.context_fill tail ~blocks:12 ~threads:16)
       (CM.context_fill tail1 ~blocks:12 ~threads:16));
  p "under the tail, saturated, a 16-thread block does half the thread-only fill's work"
    (Float.equal (CM.context_fill tail ~blocks:(8 * 16 * 2) ~threads:16) 0.5
    && Float.equal (CM.context_fill tail1 ~blocks:(8 * 16 * 2) ~threads:16) 1.)

let () =
  p_all ~min:2 "every default launch-sweep geometry is the calibration tile on lane multiples"
    Calibrate.default_launch_geometries ~f:(fun (bm, bn, _, tm, tn) ->
      Poly.equal (tm, tn) CM.calibration_register_tile && bm / tm * (bn / tn) % 32 = 0)

(* The out projection at b8: the real leaves. *)
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
    issue_rate = Some 3.4e12;
    multiprocessors = Some 46;
    fill_threads = Some 256;
    transaction_rate = Some 1.0e12;
    staged_copy_rate = Some 1.0e12;
  }

let () =
  let ctx = Context.cpu () in
  let cell idx = Float.of_int (1 + (Array.fold idx ~init:0 ~f:( + ) % 7)) *. 0.125 in
  let nb, ns, nj, nh, ne = (8, 128, 256, 8, 32) in
  let w = NTDSL.init ~l:"crt_wout" ~prec:Ir.Ops.single ~o:[ nj ] ~i:[ nh; ne ] ~f:cell () in
  let x = NTDSL.init ~l:"crt_xout" ~prec:Ir.Ops.single ~b:[ nb; ns ] ~o:[ nh; ne ] ~f:cell () in
  let%op d = w * x in
  Train.set_materialized d.Tensor.value;
  let lowered =
    Context.lowered_for_decisions ~name:"crt_outproj" ctx (Train.forward d) Ir.Indexing.Empty
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
  let m = nb * ns and n = nj and k = nh * ne in
  let counted =
    List.filter_map leaves ~f:(fun (sp : Autotune.sketch_params) ->
        match
          Ir.Schedule.apply
            (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:sp (hermetic lowered))
            (hermetic lowered)
        with
        | exception _ -> None
        | post ->
            let is_staged tn =
              Set.mem post.LL.workgroup_shared tn
              && (not (Set.mem post.LL.simdgroup_fragments tn))
              && not (Set.mem post.LL.masked_fragments tn)
            in
            if Set.is_empty post.LL.workgroup_shared then None
            else
              let shared = CM.shared_loads ~is_staged post.LL.llc in
              let cells = List.sum (module Float) shared ~f:(fun l -> l.CM.sl_loads) in
              let expected =
                Float.of_int (m * n * k / (sp.sk_tm * sp.sk_tn) * (sp.sk_tm + sp.sk_tn))
              in
              let excess =
                List.sum
                  (module Float)
                  (CM.excess_shared_loads
                     (CM.launch_contexts ~is_private:(CM.private_in post) post.LL.llc)
                     shared)
                  ~f:(fun l -> l.CM.sl_loads)
              in
              Stdio.eprintf
                "cost_model_register_tile: %s reads %.0f staged cells, register-tile arithmetic \
                 %.0f, %.0f beyond the reference (not part of the golden)\n"
                (Autotune.sketch_params_label sp) cells expected excess;
              Some (sp, cells, expected, excess))
  in
  p_all ~min:6 "every staged leaf's census equals M*N*K*(rm+rn)/(rm*rn) cells" counted
    ~f:(fun (_, cells, expected, _) -> Float.equal cells expected);
  let half_volume = Float.of_int (m * n * k) /. 2. in
  p_exists "a 2x2 leaf is among them" counted ~f:(fun (sp, _, _, _) -> sp.Autotune.sk_tm = 2);
  p_all ~min:6 "4x4 leaves load nothing beyond the reference, 2x2 leaves at most M*N*K/2 cells"
    counted ~f:(fun (sp, _, _, excess) ->
      if sp.Autotune.sk_tm = 4 && sp.sk_tn = 4 then Float.equal excess 0.
      else Float.(excess > 0.) && Float.(excess <= half_volume))

(* Through the chooser's own scoring: each field moves exactly its leaves. *)
let cuda_like =
  {
    stand_in with
    cache_capacity_bytes = Some (12 * 1024 * 1024 * 1024);
    cache_bandwidth = Some 1.8e12;
    max_resident_threads = Some (46 * 1536);
    dram_bandwidth = Some 6.0e11;
  }

let () =
  let cell idx = Float.of_int (1 + (Array.fold idx ~init:0 ~f:( + ) % 7)) *. 0.125 in
  let nb, ns, nj, nh, ne = (8, 128, 256, 8, 32) in
  let w = NTDSL.init ~l:"crt_wx" ~prec:Ir.Ops.single ~o:[ nj ] ~i:[ nh; ne ] ~f:cell () in
  let x = NTDSL.init ~l:"crt_xx" ~prec:Ir.Ops.single ~b:[ nb; ns ] ~o:[ nh; ne ] ~f:cell () in
  let%op d = w * x in
  Train.set_materialized d.Tensor.value;
  let explain ~tag limits =
    let target = Ll_test.cuda_model_target ~limits () in
    (Autotune.model_explain ~name:("crt_x_" ^ tag) ~target (Context.cpu ()) (Train.forward d)
       Ir.Indexing.Empty)
      .Autotune.x_routine
  in
  let base = explain ~tag:"base" cuda_like in
  let rated = explain ~tag:"rate" { cuda_like with shared_load_rate = Some 1.0e13 } in
  let laned = explain ~tag:"lane" { cuda_like with fill_lane_width = Some 32 } in
  let rank rows label =
    List.Assoc.find rows label ~equal:String.equal
    |> Option.bind ~f:(fun (e : Autotune.explain) -> e.Autotune.ex_rank_ms)
  in
  let threads label =
    (* The label's block tile over its register tile: [gpu BMxBNxBK/RMxRN ...]. *)
    Stdlib.Scanf.sscanf_opt label "W_sketch[gpu %dx%dx%d/%dx%d" (fun bm bn _ tm tn ->
        (bm / tm * (bn / tn), tm * tn))
  in
  let leaves =
    List.filter_map base ~f:(fun (l, _) -> Option.map (threads l) ~f:(fun t -> (l, t)))
  in
  List.iter leaves ~f:(fun (l, _) ->
      let show = Option.value_map ~default:"n/a" ~f:(Printf.sprintf "%.6f") in
      Stdio.eprintf
        "cost_model_register_tile: %s rank %s, rated %s, laned %s (not part of the golden)\n" l
        (show (rank base l))
        (show (rank rated l))
        (show (rank laned l)));
  let moved rows l = not (Option.equal Float.equal (rank rows l) (rank base l)) in
  let raised rows l =
    match (rank rows l, rank base l) with Some a, Some b -> Float.(a > b) | _ -> false
  in
  p_all ~min:4 "the shared-load rate raises the 2x2 leaves and leaves the 4x4 leaves bitwise" leaves
    ~f:(fun (l, (_, area)) -> if area = 4 then raised rated l else not (moved rated l));
  (* The plain leaves' serial batch loops leave them a sixth of the device: unsaturated, a sub-warp
     block fills as its threads; the [bgrid] leaves saturate it. *)
  let saturated l = String.is_substring l ~substring:" bgrid" in
  p_all ~min:4
    "the lane width raises the saturated sub-warp leaves and leaves the others bitwise, the \
     unsaturated sub-warp ones included"
    leaves ~f:(fun (l, (thr, _)) ->
      if thr % 32 <> 0 && saturated l then raised laned l else not (moved laned l));
  p_exists "an unsaturated sub-warp leaf is among them" leaves ~f:(fun (l, (thr, _)) ->
      thr % 32 <> 0 && not (saturated l));
  p "the seeded incumbent ranks bitwise under either field"
    (Option.is_some (rank base "seeded")
    && (not (moved rated "seeded"))
    && not (moved laned "seeded"));
  let order rows a b =
    match (rank rows a, rank rows b) with Some x, Some y -> Some (Float.compare x y) | _ -> None
  in
  let p22 = "W_sketch[gpu 32x32x16/2x2 bgrid]" and p44 = "W_sketch[gpu 32x32x8/4x4 bgrid]" in
  p "unpriced, the saturated 32x32 bgrid pair ties; priced, 4x4 ranks below 2x2 as measured"
    (Poly.equal (order base p22 p44) (Some 0) && Poly.equal (order rated p22 p44) (Some 1));
  let q22 = "W_sketch[gpu 16x16x8/2x2 bgrid]" and q44 = "W_sketch[gpu 16x16x8/4x4 bgrid]" in
  p "thread-only, the saturated 16x16 bgrid pair ties; in lanes, 2x2 ranks below 4x4 as measured"
    (Poly.equal (order base q22 q44) (Some 0) && Poly.equal (order laned q22 q44) (Some (-1)))

(* gh-ocannl-709: seeding and the pre-driver gate read a device's launch caps from ONE static
   predicate, so a geometry search never proposes a candidate the device would refuse.

   The asymmetry this closes: [Schedule.check_hardware_limits_classified] gates five launch
   dimensions (the workgroup's [.x]/[.y]/[.z] against [max_workgroup_dims], the grid's [.y] extent
   and folded [.z] product against [max_grid_yz]), and exactly one of the five — the [.z] fold — was
   also pre-filtered at seeding, in the seeder's own hand-written copy of that cap. The other four a
   search could only learn one wasted compile at a time, and the one that WAS filtered was a second
   encoding of a limit the gate already held: two places to keep in step as backends multiply.

   Now all callers consult [Schedule.launch_geometry_excess]. They differ only in where the geometry
   comes from — the gate reads it off the lowered code, the seeder predicts it from the parameters
   it is about to commit to — so this file pins three things:

   - the predicate's own rows: each of the five dimensions refuses one over its cap as its own typed
   resource, and accepts a candidate exactly AT the cap (without that arm a predicate that refuses
   everything reads as a pass), and an unpredicted dimension is exempt rather than refused; - the
   seeder's prediction is FAITHFUL: for every GPU seed of a real batched matmul, the predicted
   geometry is the one the applied schedule actually launches with. A prediction that drifted from
   what the builders emit would silently withhold legal candidates, which is worse than the wasted
   compile it saves; - parity, per dimension, on one real seed: a cap at the seed's own extent
   leaves it seeded AND accepted by the gate; a cap one below removes it from the seed list AND
   makes the gate refuse it — with the SAME sentence, so a refutation log and a decline log say the
   same thing about the same candidate.

   Limits are MOCKED throughout: the point is a cap below a candidate's geometry, and the fleet's
   real devices are nowhere near these extents (CUDA's [maxThreadsDim.z] of 64 against a 1024
   product cap is the one genuinely tight per-dimension cap, and no Apple part reproduces it). The
   lowering is real. The matmul site is captured directly; the conv sites are derived from the
   pre-schedule normal segment of [Schedule.fission_scheduled], because their [Zero_out] must be in
   a separate kernel before a GPU conv schedule applies. Only [Schedule.apply] and the seeding API
   consume those sites, so the claims remain backend-independent while the CUDA run exercises the
   real GPU lowering path. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SO = Ir.Schedule_outcome
module BI = Ir.Backend_intf
module Asgns = Ir.Assignments
open Verdict.Claims

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let geometry ?grid_y ?grid_z ?block_x ?block_y ?block_z () =
  {
    Sched.lg_grid_y = grid_y;
    lg_grid_z = grid_z;
    lg_block_x = block_x;
    lg_block_y = block_y;
    lg_block_z = block_z;
  }

let wg_caps dims = { BI.no_hardware_limits with max_workgroup_dims = Some dims }
let grid_cap n = { BI.no_hardware_limits with max_grid_yz = Some n }

(* Identity of a seed for membership questions: the parameters that decide its launch geometry, as
   plain scalars (the record carries a precision option, which no comparison here needs). *)
let key (q : Autotune.sketch_params) =
  ( q.Autotune.sk_gpu,
    q.Autotune.sk_mma,
    q.Autotune.sk_bm,
    q.Autotune.sk_bn,
    q.Autotune.sk_bk,
    q.Autotune.sk_tm,
    q.Autotune.sk_tn,
    q.Autotune.sk_simd,
    q.Autotune.sk_batch_grid,
    q.Autotune.sk_epilogue,
    q.Autotune.sk_depth )

let () =
  (* === The predicate's five rows, each over its own cap and each at it === *)
  let row ~what ~resource ~at ~over g =
    p
      (what ^ ": a geometry exactly at the cap passes the predicate")
      (Option.is_none (Sched.launch_geometry_excess ~limits:at g));
    p
      (what ^ ": one over the cap is refused as its own typed resource")
      (match Sched.launch_geometry_excess ~limits:over g with
      | Some x ->
          SO.equal_resource x.Sched.lx_resource resource
          && x.Sched.lx_requested = 128 && x.Sched.lx_limit = 127
      | None -> false)
  in
  row ~what:".x workgroup extent" ~resource:SO.Workgroup_x_extent
    ~at:(wg_caps (128, 1, 1))
    ~over:(wg_caps (127, 1024, 1024))
    (geometry ~block_x:128 ());
  row ~what:".y workgroup extent" ~resource:SO.Workgroup_y_extent
    ~at:(wg_caps (1, 128, 1))
    ~over:(wg_caps (1024, 127, 1024))
    (geometry ~block_y:128 ());
  (* CUDA's cliff: [maxThreadsDim] is (1024, 1024, 64) — re-verified through bin/device_props on the
     fleet's sm_120 part, which also reports [maxGridSize] (2^31-1, 65535, 65535) — so a 2 x 2 x 128
     workgroup has a perfectly legal 512-thread product and is still an invalid launch
     configuration. No in-tree annotator emits three [Workgroup] loops, so this row is exercised
     through the predicate directly; [launch_dim_gate.ml] carries the same geometry through the gate
     on a hand-built nest. *)
  row ~what:".z workgroup extent" ~resource:SO.Workgroup_z_extent
    ~at:(wg_caps (1, 1, 128))
    ~over:(wg_caps (1024, 1024, 127))
    (geometry ~block_z:128 ());
  row ~what:".y grid extent" ~resource:SO.Grid_y_extent ~at:(grid_cap 128) ~over:(grid_cap 127)
    (geometry ~grid_y:128 ());
  row ~what:".z grid fold" ~resource:SO.Grid_z_extent ~at:(grid_cap 128) ~over:(grid_cap 127)
    (geometry ~grid_z:128 ());
  p "the CUDA cliff: a 2 x 2 x 128 workgroup is refused on a (1024, 1024, 64) device"
    (match
       Sched.launch_geometry_excess
         ~limits:(wg_caps (1024, 1024, 64))
         (geometry ~block_x:2 ~block_y:2 ~block_z:128 ())
     with
    | Some x -> SO.equal_resource x.Sched.lx_resource SO.Workgroup_z_extent
    | None -> false);
  (* A dimension the caller does not predict is exempt, not refused: that is what lets a family
     predict only the part of its geometry it knows. *)
  p "an unpredicted dimension is exempt rather than refused"
    (Option.is_none
       (Sched.launch_geometry_excess ~limits:(wg_caps (1, 1, 1)) Sched.unknown_launch_geometry)
    && Option.is_none (Sched.launch_geometry_excess ~limits:(grid_cap 1) (geometry ~block_x:8 ())));
  (* An absent cap is what the C backends report, and must exempt rather than refuse. *)
  p "absent caps gate nothing"
    (Option.is_none
       (Sched.launch_geometry_excess ~limits:BI.no_hardware_limits
          (geometry ~grid_y:99999 ~grid_z:99999 ~block_x:9999 ~block_y:9999 ~block_z:9999 ())));

  (* === A real batched matmul site: the q/k/v rank-4 projection, two outer batch loops === *)
  let bb = 2 and hh = 4 and ss = 64 and jj = 32 and kk = 16 in
  let x () =
    NTDSL.init ~l:"lpp_x" ~prec:Ir.Ops.single ~o:[ bb; ss; kk ]
      ~f:(Ll_test.cycle ~dims:[| bb; ss; kk |] ~modulus:13 ~offset:0. ~stride:0.25)
      ()
  in
  let w () =
    NTDSL.init ~l:"lpp_w" ~prec:Ir.Ops.single ~o:[ hh; kk; jj ]
      ~f:(Ll_test.cycle ~dims:[| hh; kk; jj |] ~modulus:11 ~offset:(-5.) ~stride:0.5)
      ()
  in
  let captured = ref None in
  let _ctx, _r =
    let xv = x () and wv = w () in
    let%op out = xv +* "bsk;hkj=>bhsj" wv in
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      (Context.auto ())
      (named "lpp_site" (Train.forward out))
      Ir.Indexing.Empty
  in
  let opt = Option.value_exn ~here:[%here] !captured in
  let site = Option.value_exn ~here:[%here] (Autotune.detect_matmul opt.LL.llc) in
  (* A synthetic f32 mma capability so the tensorized pipeline seeds too: the prediction covers both
     GPU pipelines, whose workgroup geometries differ (register splits vs. the tensorization lane).
     Machine-independent by construction — no field here is read off a device. *)
  let mma_limits =
    {
      BI.no_hardware_limits with
      mma =
        Some
          {
            BI.mma_simd_width = 32;
            mma_tile = (8, 8, 8);
            mma_format_tiles = [ ((BI.Mma_f32, BI.Mma_f32, BI.Mma_f32), (8, 8, 8)) ];
            mma_f16_wide_acc_scopes = [];
            mma_bf16_wide_acc_scopes = [];
            mma_staged_layouts = [];
            mma_pipeline_depths = [];
          };
    }
  in
  let seeds limits = Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits opt in

  (* === The prediction is the geometry the schedule actually launches with === *)
  let all_gpu = List.filter (seeds mma_limits) ~f:(fun q -> q.Autotune.sk_gpu) in
  p "prediction: the site seeds GPU candidates of both pipelines"
    (List.exists all_gpu ~f:(fun q -> q.Autotune.sk_mma)
    && List.exists all_gpu ~f:(fun q -> not q.Autotune.sk_mma)
    && List.exists all_gpu ~f:(fun q -> q.Autotune.sk_batch_grid));
  let unfaithful =
    List.filter all_gpu ~f:(fun q ->
        match Sched.apply (Autotune.sketch_schedule ~p:q opt) opt with
        | o ->
            let actual = Sched.launch_geometry_of_dims (LL.launch_dims o.LL.llc) in
            let predicted = Autotune.matmul_launch_geometry site q in
            let bad = not (Poly.equal predicted actual) in
            if bad then
              Stdio.eprintf "prediction MISMATCH: mma=%b batch_grid=%b bm=%d bn=%d tm=%d tn=%d\n"
                q.Autotune.sk_mma q.Autotune.sk_batch_grid q.Autotune.sk_bm q.Autotune.sk_bn
                q.Autotune.sk_tm q.Autotune.sk_tn;
            bad
        | exception exn ->
            Stdio.eprintf "prediction: schedule FAILED: %s\n" (Exn.to_string exn);
            true)
  in
  p_empty "prediction: every GPU seed launches with exactly the geometry the seeder predicted"
    ~over:all_gpu unfaithful;

  (* === Parity, dimension by dimension, on one real seed ===

     The reference seeds are scalar-blocktile leaves (no mma capability in [limits], so the
     tensorized pipeline is refuted and the enumeration is the blocktile family alone): a 32x32x8
     tiling of this site launches a 1 x 2 grid of 8 x 8 workgroups, and its batch-grid twin folds b
     x h = 8 onto [.z]. The [.y] leg uses the serial-batch seed and the [.z] leg the twin, because
     one [max_grid_yz] field caps both grid dimensions — on the twin, a cap below the row-block
     count would also refuse the fold, and the claim could not say which row fired. *)
  let blocktile ~batch_grid =
    List.find_exn (seeds BI.no_hardware_limits) ~f:(fun q ->
        q.Autotune.sk_gpu && (not q.Autotune.sk_mma) && (not q.Autotune.sk_epilogue)
        && q.Autotune.sk_bm = 32 && q.Autotune.sk_tm = 4 && q.Autotune.sk_bk = 8
        && Bool.equal q.Autotune.sk_batch_grid batch_grid)
  in
  let serial_seed = blocktile ~batch_grid:false and twin_seed = blocktile ~batch_grid:true in
  let applied q = Sched.apply (Autotune.sketch_schedule ~p:q opt) opt in
  let sdims = LL.launch_dims (applied serial_seed).LL.llc in
  let tdims = LL.launch_dims (applied twin_seed).LL.llc in
  p
    "parity premise: the reference seed launches 8 x 8 workgroups over a .y grid of 2, its twin \
     folding 8 onto .z"
    (sdims.LL.block.(0) = 8
    && sdims.LL.block.(1) = 8
    && sdims.LL.grid.(1) = 2
    && sdims.LL.grid.(2) = 1
    && tdims.LL.grid.(2) = bb * hh);

  (* The gate's verdict on this seed's schedule: the typed resource and the sentence it reports. *)
  let gate q ~limits =
    match Sched.check_hardware_limits_classified ~name:"lpp" ~limits (applied q) with
    | () -> None
    | exception SO.Cause_at (_, SO.Resource_exceeded { resource; detail; _ }) ->
        Some (resource, detail)
    | exception exn -> Some (SO.Workgroup_threads, "unexpected: " ^ Exn.to_string exn)
  in
  (* Every witness the family tree refuses a candidate with, under these limits. *)
  let refutation_witnesses limits =
    match Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits opt with
    | Some tree -> List.map (Ir.Schedule_space.refutations tree) ~f:snd
    | None -> []
  in
  let parity ~what ~resource ~seed ~at ~over =
    let seeded limits = List.exists (seeds limits) ~f:(fun q -> Poly.equal (key q) (key seed)) in
    p (what ^ ": at the cap the candidate is still seeded") (seeded at);
    p (what ^ ": at the cap the gate accepts it") (Option.is_none (gate seed ~limits:at));
    p_empty
      (what ^ ": over the cap it is not seeded at all")
      ~over:(seeds at)
      (List.filter (seeds over) ~f:(fun q -> Poly.equal (key q) (key seed)));
    p
      (what ^ ": over the cap the gate refuses it as its own typed resource")
      (match gate seed ~limits:over with
      | Some (r, _) -> SO.equal_resource r resource
      | None -> false);
    (* The reason parity: the seeder's refutation witness is the gate's detail sentence, verbatim.
       Both are rendered from the one [launch_excess] the shared predicate returns, so a search's
       refutation log and its decline log describe the same candidate the same way. *)
    p
      (what ^ ": seeding and the gate refuse it with the same sentence")
      (match gate seed ~limits:over with
      | Some (_, detail) ->
          List.exists (refutation_witnesses over) ~f:(fun witness ->
              match String.chop_prefix witness ~prefix:"the candidate " with
              | Some phrase -> String.is_suffix detail ~suffix:phrase
              | None -> false)
      | None -> false)
  in
  parity ~what:"seed .x workgroup extent" ~resource:SO.Workgroup_x_extent ~seed:serial_seed
    ~at:(wg_caps (8, 1024, 1024))
    ~over:(wg_caps (7, 1024, 1024));
  parity ~what:"seed .y workgroup extent" ~resource:SO.Workgroup_y_extent ~seed:serial_seed
    ~at:(wg_caps (1024, 8, 1024))
    ~over:(wg_caps (1024, 7, 1024));
  parity ~what:"seed .y grid extent" ~resource:SO.Grid_y_extent ~seed:serial_seed ~at:(grid_cap 2)
    ~over:(grid_cap 1);
  parity ~what:"seed .z grid fold" ~resource:SO.Grid_z_extent ~seed:twin_seed
    ~at:(grid_cap (bb * hh))
    ~over:(grid_cap ((bb * hh) - 1));

  (* === Real convolution sites behind the fission seam (gh-ocannl-739) === *)
  let capture_conv_segment name y =
    let captured = ref None in
    let ctx = Context.auto () in
    let limits = Context.hardware_limits ctx in
    let _ctx, _routine =
      Context.compile
        ~lowered_transform:(fun opt ->
          let preset seg = Sched.default_gpu ~min_parallel:1 ~limits seg in
          let zero_sched tns = Sched.zero_expansion ~limits tns in
          let segments = Sched.fission_scheduled ~preset ~zero_sched ~static_indices:[] opt in
          List.iter segments ~f:(fun (_, pre, _, _) ->
              match Autotune.detect_conv pre.LL.llc with
              | Some _ -> captured := Some pre
              | None -> ());
          List.map segments ~f:(fun (_, _, _, post) -> post))
        ctx
        (named name (Train.forward y))
        Ir.Indexing.Empty
    in
    Option.value_exn ~here:[%here] !captured
  in
  let make_conv2 tag =
    let x =
      NTDSL.init ~l:(tag ^ "_x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 18; 18; 8 ]
        ~f:(fun idcs -> Float.of_int (Int.rem (Array.fold idcs ~init:0 ~f:( + )) 7))
        ()
    in
    let kernel =
      NTDSL.init ~l:(tag ^ "_k") ~prec:Ir.Ops.single ~i:[ 3; 3; 8 ] ~o:[ 16 ]
        ~f:(fun idcs -> Float.of_int (Int.rem (Array.fold idcs ~init:0 ~f:( + )) 5))
        ()
    in
    let%op y =
      x
      +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kernel
    in
    y
  in
  let conv_opt = capture_conv_segment "lpp_conv" (make_conv2 "lpp_c") in
  let conv_site = Option.value_exn ~here:[%here] (Autotune.detect_conv conv_opt.LL.llc) in
  let conv_gpu =
    Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:mma_limits conv_opt
    |> List.filter ~f:(fun q -> q.Autotune.sk_gpu && q.Autotune.sk_conv)
  in
  p_exists "conv prediction: the fission segment seeds a whole-extent GPU flavor" conv_gpu
    ~f:(fun q -> q.Autotune.sk_bm = 0);
  p_exists "conv prediction: the fission segment seeds a row-blocked GPU flavor" conv_gpu
    ~f:(fun q -> q.Autotune.sk_bm > 0);
  let lower_bound (predicted : Sched.launch_geometry) (actual : Sched.launch_geometry) =
    let le p a =
      match (p, a) with None, _ -> true | Some p, Some a -> p <= a | Some _, None -> false
    in
    le predicted.Sched.lg_grid_y actual.Sched.lg_grid_y
    && le predicted.Sched.lg_grid_z actual.Sched.lg_grid_z
    && le predicted.Sched.lg_block_x actual.Sched.lg_block_x
    && le predicted.Sched.lg_block_y actual.Sched.lg_block_y
    && le predicted.Sched.lg_block_z actual.Sched.lg_block_z
  in
  p_none "conv prediction: no GPU seed predicts a dimension above its applied launch geometry"
    conv_gpu ~f:(fun q ->
      match Sched.apply (Autotune.sketch_schedule ~p:q conv_opt) conv_opt with
      | applied ->
          let actual = Sched.launch_geometry_of_dims (LL.launch_dims applied.LL.llc) in
          not (lower_bound (Autotune.conv_launch_geometry conv_site q) actual)
      | exception exn ->
          Stdio.eprintf "conv prediction: schedule FAILED: %s\n" (Exn.to_string exn);
          true);

  (* Negative control: a 2-D conv whose outer [Grid] loops are the batch (65536) then the non-row
     spatial axis (2), with the row axis (16) blocked. The blocked flavor's row-block loop is the
     innermost grid coordinate, so the slot rule binds the row blocks to [.x] and the non-row
     spatial extent to [.y], leaving the batch ALONE to fold onto [.z] -- 65536, one past the 16-bit
     cap. Blocking is what makes this a [.z] question: unblocked, the same site has only two grid
     coordinates and the batch lands on [.y] instead. The small spatial extents keep the fixture
     cheap; the graph is compiled but never dispatched. *)
  let make_large_conv2 tag =
    let x =
      NTDSL.init ~l:(tag ^ "_x") ~prec:Ir.Ops.single ~b:[ 65_536 ] ~o:[ 3; 17; 2 ]
        ~f:(fun _ -> 1.)
        ()
    in
    let kernel =
      NTDSL.init ~l:(tag ^ "_k") ~prec:Ir.Ops.single ~i:[ 2; 2; 2 ] ~o:[ 2 ] ~f:(fun _ -> 1.) ()
    in
    let%op y =
      x
      +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kernel
    in
    y
  in
  let large_opt = capture_conv_segment "lpp_conv_overcap" (make_large_conv2 "lpp_co") in
  let large_site = Option.value_exn ~here:[%here] (Autotune.detect_conv large_opt.LL.llc) in
  let permissive_limits = { mma_limits with BI.max_grid_yz = Some 1_000_000 } in
  let capped_limits = { mma_limits with BI.max_grid_yz = Some 65_535 } in
  let gpu_seeds limits =
    Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits large_opt
    |> List.filter ~f:(fun q -> q.Autotune.sk_gpu && q.Autotune.sk_conv)
  in
  let permissive = gpu_seeds permissive_limits in
  let over_cap q =
    match
      Sched.launch_geometry_excess ~limits:capped_limits
        (Autotune.conv_launch_geometry large_site q)
    with
    | Some x -> SO.equal_resource x.Sched.lx_resource SO.Grid_z_extent
    | None -> false
  in
  p_exists
    "conv control: a permissive cap proposes a blocked seed folding the 65536 batch onto grid.z"
    permissive ~f:over_cap;
  let rejected_keys = List.filter permissive ~f:over_cap |> List.map ~f:key in
  let leaked =
    List.filter (gpu_seeds capped_limits) ~f:(fun q ->
        List.mem rejected_keys (key q) ~equal:Poly.equal)
  in
  p_empty "conv seeding: the 65535 grid.z cap omits every deliberately over-cap seed"
    ~over:rejected_keys leaked

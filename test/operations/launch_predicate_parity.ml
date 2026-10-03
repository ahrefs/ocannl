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
   pre-schedule normal segment of [Schedule.fission_scheduled], including the covering per-cell zero
   companion exposed before GPU fission. Only [Schedule.apply] and the seeding API consume those
   sites, so the claims remain backend-independent while the CUDA run exercises the real GPU
   lowering path. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SO = Ir.Schedule_outcome
module BI = Ir.Backend_intf
module Asgns = Ir.Assignments
open Verdict.Claims

(* The backend's accumulator residency, which a [Privatize] tile is minted at (gh-ocannl-1116). *)
let accum_prec =
  let caps = lazy (Context.codegen_capabilities (Context.auto ())) in
  fun p -> (Lazy.force caps).Ir.Backend_intf.accum_prec p

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
    q.Autotune.sk_batch_inner,
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
            BI.minimal_mma_capability with
            mma_tile = (8, 8, 8);
            mma_format_tiles = [ ((BI.Mma_f32, BI.Mma_f32, BI.Mma_f32), (8, 8, 8)) ];
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
        match Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:q opt) opt with
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

  (* The same faithfulness on the gpt2 projection's own layout, [d[b,s,h,e] += x[b,s,k] * w[h,e,k]]:
     its head loop is INTERIOR, so the bgrid-in twins (gh-ocannl-728) seed here — the interior batch
     between the row and column blocks, the row blocks folding with [b] onto [.z] — and their
     prediction is a different grid order from the batch-grid twins'. *)
  let pcaptured = ref None in
  let _ctx, _r =
    let xv = x () in
    let wv =
      NTDSL.init ~l:"lpp_wp" ~prec:Ir.Ops.single ~o:[ hh; jj; kk ]
        ~f:(Ll_test.cycle ~dims:[| hh; jj; kk |] ~modulus:11 ~offset:(-5.) ~stride:0.5)
        ()
    in
    let%op out = xv +* "bsk;hjk=>bshj" wv in
    Context.compile
      ~lowered_transform:(fun opt ->
        pcaptured := Some opt;
        [ opt ])
      (Context.auto ())
      (named "lpp_proj" (Train.forward out))
      Ir.Indexing.Empty
  in
  let popt = Option.value_exn ~here:[%here] !pcaptured in
  let psite = Option.value_exn ~here:[%here] (Autotune.detect_matmul popt.LL.llc) in
  let proj_gpu =
    List.filter (Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:mma_limits popt)
      ~f:(fun q -> q.Autotune.sk_gpu)
  in
  p "prediction: the projection site seeds bgrid-in twins of both pipelines"
    (List.exists proj_gpu ~f:(fun q -> q.Autotune.sk_batch_inner && q.Autotune.sk_mma)
    && List.exists proj_gpu ~f:(fun q -> q.Autotune.sk_batch_inner && not q.Autotune.sk_mma));
  p_all "prediction: every projection GPU seed launches with exactly the predicted geometry"
    proj_gpu ~f:(fun q ->
      match Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:q popt) popt with
      | o ->
          let actual = Sched.launch_geometry_of_dims (LL.launch_dims o.LL.llc) in
          let ok = Poly.equal (Autotune.matmul_launch_geometry psite q) actual in
          if not ok then
            Stdio.eprintf
              "projection prediction MISMATCH: mma=%b batch_grid=%b inner=%b bm=%d bn=%d\n"
              q.Autotune.sk_mma q.Autotune.sk_batch_grid q.Autotune.sk_batch_inner q.Autotune.sk_bm
              q.Autotune.sk_bn;
          ok
      | exception exn ->
          Stdio.eprintf "projection prediction: schedule FAILED: %s\n" (Exn.to_string exn);
          false);

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
  let applied q = Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:q opt) opt in
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
        ~f:(fun ix ->
          Float.of_int (1 + (3 * ix.(0)) + (5 * ix.(1)) + (7 * ix.(2)) + (11 * ix.(3))) /. 128.)
        ()
    in
    let kernel =
      NTDSL.init ~l:(tag ^ "_k") ~prec:Ir.Ops.single ~i:[ 3; 3; 8 ] ~o:[ 16 ]
        ~f:(fun ix ->
          Float.of_int (1 + (2 * ix.(0)) + (3 * ix.(1)) + (5 * ix.(2)) + (7 * ix.(3))) /. 64.)
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
  let apply_conv pre q =
    Sched.apply
      (Autotune.sketch_schedule ~accum_prec ~p:q pre)
      {
        pre with
        LL.traced_store = Hashtbl.copy pre.LL.traced_store;
        optimize_ctx = LL.copy_optimize_ctx pre.LL.optimize_ctx;
      }
  in
  let valid_conv pre q =
    match apply_conv pre q with
    | applied -> (
        match LL.validate_parallel applied.LL.optimize_ctx.placements applied.LL.llc with
        | () -> true
        | exception exn ->
            Stdio.eprintf "conv validation FAILED: %s\n" (Exn.to_string exn);
            false)
    | exception exn ->
        Stdio.eprintf "conv construction FAILED: %s\n" (Exn.to_string exn);
        false
  in
  p_all "conv coverage: every GPU seed maps the covering zero companion" conv_gpu
    ~f:(valid_conv conv_opt);
  let biased_conv = make_conv2 "lpp_cb" in
  Train.set_materialized biased_conv.Tensor.value;
  let bias =
    NTDSL.init ~l:"lpp_cb_bias" ~prec:Ir.Ops.single ~o:[ 16 ]
      ~f:(fun ix -> Float.of_int (1 + ix.(0)))
      ()
  in
  let%op biased = biased_conv + bias in
  Train.set_materialized biased.Tensor.value;
  let biased_opt = capture_conv_segment "lpp_conv_bias" biased in
  let biased_gpu =
    Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:mma_limits biased_opt
    |> List.filter ~f:(fun q -> q.Autotune.sk_gpu && q.Autotune.sk_conv)
  in
  p_all "conv+bias coverage: GPU seeds remain eligible and validate with the zero companion"
    biased_gpu ~f:(valid_conv biased_opt);
  let parity_label =
    "conv+bias coverage: GPU sketches match a materialized run with discriminating operands"
  in
  let backend = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc") in
  let real_gpu =
    if not (Sched.backend_is_gpu backend) then []
    else
      Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false
        ~limits:(Context.hardware_limits (Context.auto ()))
        biased_opt
      |> List.filter ~f:(fun q -> q.Autotune.sk_gpu && q.Autotune.sk_conv)
  in
  if List.is_empty real_gpu then Verdict.skipped ~backend parity_label
  else begin
    let outputs = [ biased_conv.Tensor.value; biased.Tensor.value ] in
    let seed =
      List.map outputs ~f:(fun tn -> (tn, Array.create ~len:(Ir.Tnode.num_elems tn) (-999.)))
    in
    let want =
      List.hd_exn
        (Ll_test.execute ~name:"lpp_cb_materialized" biased_opt ~seed ~read:[ biased.Tensor.value ])
    in
    p_all parity_label real_gpu ~f:(fun q ->
        let got =
          List.hd_exn
            (Ll_test.execute
               ~name:
                 ("lpp_cb_sketch_" ^ Int.to_string q.Autotune.sk_bm ^ "_"
                 ^ Bool.to_string q.Autotune.sk_epilogue
                 ^ "_" ^ Int.to_string q.Autotune.sk_depth)
               (apply_conv biased_opt q) ~seed ~read:[ biased.Tensor.value ])
        in
        Array.equal Float.equal got want)
  end;
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
      match apply_conv conv_opt q with
      | applied ->
          let actual = Sched.launch_geometry_of_dims (LL.launch_dims applied.LL.llc) in
          not (lower_bound (Autotune.conv_launch_geometry conv_site q) actual)
      | exception exn ->
          Stdio.eprintf "conv prediction: schedule FAILED: %s\n" (Exn.to_string exn);
          true);

  (* Negative controls: a 2-D conv whose outer [Grid] loops are the batch (65536) then the non-row
     spatial axis (2 = 3 - 2 + 1), over 16 rows (17 - 2 + 1). The two flavors put the SAME batch on
     different grid dimensions. Blocked ([sk_bm > 0]), the row-block loop is the innermost grid
     coordinate, so the slot rule binds the row blocks to [.x], the spatial extent to [.y], and
     folds the batch ALONE onto [.z] — 65536, one past the 16-bit cap. Unblocked ([sk_bm = 0]), the
     site has only its two outer grid coordinates: the spatial extent binds [.x], the batch lands on
     [.y] at the same 65536, and nothing folds onto [.z].

     Each claim below names the dimension, the requested extent and the cap (gh-ocannl-939): a
     resource-only claim ("some seed is refused on [.z]") holds for a wrong reading of the geometry
     too. Blocking is what moves the batch from [.y] to [.z], so each flavor is the other's negative
     control: the [.z] refusal claim must FAIL on every unblocked seed and the [.y] one on every
     blocked seed. The small spatial extents keep the fixture cheap; the graph is compiled but never
     dispatched. *)
  let batch = 65_536 and spatial = 2 and rows = 16 and device_cap = 65_535 in
  let make_large_conv2 tag =
    let x =
      NTDSL.init ~l:(tag ^ "_x") ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ 3; 17; 2 ]
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
  (* The fixture's derived extents, which the geometry claims below are written against: shape
     inference, not the fixture's literals, decides the spatial and row extents. *)
  pf
    "conv control premise: the site's outer grid loops are the %d batch then a non-row spatial %d, \
     over %d rows"
    batch spatial rows
    (List.equal Int.equal (List.map large_site.Autotune.c_outer ~f:snd) [ batch; spatial ]
    && large_site.Autotune.c_nrow = rows);
  let permissive_limits = { mma_limits with BI.max_grid_yz = Some 1_000_000 } in
  let capped_limits = { mma_limits with BI.max_grid_yz = Some device_cap } in
  let gpu_seeds limits =
    Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits large_opt
    |> List.filter ~f:(fun q -> q.Autotune.sk_gpu && q.Autotune.sk_conv)
  in
  let permissive = gpu_seeds permissive_limits in
  let blocked = List.filter permissive ~f:(fun q -> q.Autotune.sk_bm > 0) in
  let unblocked = List.filter permissive ~f:(fun q -> q.Autotune.sk_bm = 0) in
  let predicted q = Autotune.conv_launch_geometry large_site q in
  (* The claim each control makes: the capped predicate refuses the seed on exactly [resource], at
     exactly the batch extent, against exactly the device cap. Any other dimension or extent — a
     different first excess, a different requested value — makes it false. *)
  let is_batch_excess ~resource (x : Sched.launch_excess) =
    SO.equal_resource x.Sched.lx_resource resource
    && x.Sched.lx_requested = batch && x.Sched.lx_limit = device_cap
  in
  let refused_on resource q =
    match Sched.launch_geometry_excess ~limits:capped_limits (predicted q) with
    | Some x -> is_batch_excess ~resource x
    | None -> false
  in
  let grid_yz q = ((predicted q).Sched.lg_grid_y, (predicted q).Sched.lg_grid_z) in
  let same_yz (a : int option * int option) b = Poly.equal a b in
  p_all
    (Printf.sprintf
       "conv control: every blocked seed puts the spatial %d on grid.y and folds the %d batch \
        alone onto grid.z"
       spatial batch) blocked ~f:(fun q -> same_yz (grid_yz q) (Some spatial, Some batch));
  p_all
    (Printf.sprintf
       "conv control: every unblocked seed puts the %d batch on grid.y, folding nothing onto grid.z"
       batch) unblocked ~f:(fun q -> same_yz (grid_yz q) (Some batch, Some 1));
  p_all
    (Printf.sprintf
       "conv control: every blocked seed is refused on grid.z at %d, against the %d cap" batch
       device_cap)
    blocked ~f:(refused_on SO.Grid_z_extent);
  p_all
    (Printf.sprintf
       "conv control: every unblocked seed is refused on grid.y at %d, against the %d cap" batch
       device_cap)
    unblocked ~f:(refused_on SO.Grid_y_extent);
  (* The discrimination the two claims above rest on: each rejects the other flavor's refusal. *)
  p_none "conv control: the grid.z refusal claim fails on every seed refused on grid.y instead"
    unblocked ~f:(refused_on SO.Grid_z_extent);
  p_none "conv control: the grid.y refusal claim fails on every seed refused on grid.z instead"
    blocked ~f:(refused_on SO.Grid_y_extent);
  (* And the extent half: the matmul twin above is a real [.z] refusal at another extent (its 8-way
     batch fold against a cap of 7), which the [.z] claim must reject on the extent alone. *)
  p "conv control: the grid.z refusal claim fails on a grid.z refusal at another extent"
    (match
       Sched.launch_geometry_excess
         ~limits:(grid_cap ((bb * hh) - 1))
         (Autotune.matmul_launch_geometry site twin_seed)
     with
    | Some x ->
        SO.equal_resource x.Sched.lx_resource SO.Grid_z_extent
        && not (is_batch_excess ~resource:SO.Grid_z_extent x)
    | None -> false);
  let capped_keys = List.map (gpu_seeds capped_limits) ~f:key in
  let leaked flavor =
    List.filter flavor ~f:(fun q -> List.mem capped_keys (key q) ~equal:Poly.equal)
  in
  p_empty
    (Printf.sprintf "conv seeding: the %d cap omits every blocked seed it refuses on grid.z"
       device_cap)
    ~over:blocked (leaked blocked);
  p_empty
    (Printf.sprintf "conv seeding: the %d cap omits every unblocked seed it refuses on grid.y"
       device_cap)
    ~over:unblocked (leaked unblocked)

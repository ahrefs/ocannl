(* gh-ocannl-643: the rank-4 q/k/v geometry — batch and head axes of a batched/multi-head GEMM reach
   grid parallelism instead of running as serial loops inside each block.

   Mechanism under test, in two layers:

   - The [.z] grid fold (Low_level): [Grid] slots >= 2 are legal and share the hardware [.z]
   dimension — [launch_dims] multiplies their per-slot maxima into [grid.(2)], and each such loop
   binds [(z / stride) % cap] ([Low_level.grid_fold]; rendered by [C_syntax.hardware_binding]). On
   backends that bind no hardware register (cc) the loops keep their serial rendering, so the
   scheduled kernels execute correctly everywhere.

   - The [sk_batch_grid] twins (sketch families): each GPU matmul geometry of a batched (rank-3+)
   site is seeded twice — batch loops [Serial] (the pre-643 shape) and batch loops [Retype]d to
   [Grid] — with the zeroing nest and every companion nest carrying the same per-position
   annotation, interior batch loops hoisted identically ([companion_role_ops]). Twins rather than a
   replacement: block-count curves are non-monotone (gh-ocannl-569's probe), so the tuner measures
   both.

   Three lowered shapes: the q/k/v projection's leading-batch rank-4 site [out[b,h,s,j] += x[b,s,k]
   * w[h,k,j]] (two outer batch loops — the issue's mechanism); the same site with a materialized
   output feeding a bias+relu companion nest (companion coverage at full arity plus the companions'
   batch annotation); and the interior-batch rank-4 site [out[b,i,h,j] += att[b,i,h,k] * v[b,k,h,j]]
   (the head axis BETWEEN the tile roles, so the zero-nest and companion hoisting is load-bearing).

   Executed assertions compare each candidate against a serial reference computed from the same
   discriminating inputs; the input values vary with every index and keep all partial sums exactly
   representable in f32, so bitwise equality is required regardless of accumulation-order changes
   from the tiling. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let on_gpu = Sched.backend_is_gpu backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Zeros compare equal to zeros: pin every reference nonzero so the parity claims have content. *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let compile_serial ~name tensor =
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt -> [ opt ])
      (Context.auto ())
      (named name (Train.forward tensor))
      Ir.Indexing.Empty
  in
  nonzero name (Context.get_values (Context.run ctx routine) tensor.Tensor.value)

(* The scalar-blocktile GPU seeds of the routine, unfused, via the public seeding API. Synthetic
   no-limits keep the enumeration machine-independent; no mma capability, so the tensorized pipeline
   is refuted and the seeds are exactly the blocktile family. *)
let blocktile_seeds opt =
  Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:Ir.Backend_intf.no_hardware_limits
    opt
  |> List.filter ~f:(fun q ->
      q.Autotune.sk_gpu && (not q.Autotune.sk_mma) && not q.Autotune.sk_epilogue)

(* One leg: build the tensor twice (serial reference and candidate), enumerate the blocktile seeds,
   and check the batch-grid structure (launch dims, fold arithmetic, [validate_parallel]) on every
   seed's schedule applied as the pure IR transform it is — backend-independent, so this runs on cc
   too. On GPU backends every seed is additionally compiled, executed against the serial reference
   and — for the batch-grid twins — the generated source is checked for the folded [.z] bindings; cc
   cannot execute workgroup-shared staging, so those claims print as skipped there (the same gating
   as autotune_batched_companion). [batch_product] is the expected [.z] extent; [fold_div] and
   [fold_mod] the substrings the folded bindings must render. *)
let leg ~tag ~batch_product ~fold_div ~fold_mod ~build =
  let want = compile_serial ~name:(tag ^ "_serial") (build ()) in
  let cand = build () in
  let fwd = named (tag ^ "_sched") (Train.forward cand) in
  let captured = ref None in
  let _ctx, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      (Context.auto ()) fwd Ir.Indexing.Empty
  in
  let opt = Option.value_exn ~here:[%here] !captured in
  let seeds = blocktile_seeds opt in
  let serial_seeds, grid_seeds =
    List.partition_tf seeds ~f:(fun q -> not q.Autotune.sk_batch_grid)
  in
  p (tag ^ ": batch-grid twins are seeded") (not (List.is_empty grid_seeds));
  p
    (tag ^ ": one batch-grid twin per serial-batch geometry")
    (List.length grid_seeds = List.length serial_seeds);
  p
    (tag ^ ": batch-serial seeds precede their batch-grid twins")
    (List.is_sorted seeds ~compare:(fun a b ->
         Bool.compare a.Autotune.sk_batch_grid b.Autotune.sk_batch_grid));
  (* --- Structural checks on the pure transform, every seed, every backend. --- *)
  let grid_z_ok = ref true and fold_ok = ref true and valid_ok = ref true in
  List.iter seeds ~f:(fun q ->
      let o = Sched.apply (Autotune.sketch_schedule ~p:q opt) opt in
      let llc = o.LL.llc in
      (match LL.validate_parallel o.LL.optimize_ctx.LL.placements llc with
      | () -> ()
      | exception exn ->
          Stdio.eprintf "%s: validate_parallel FAILED: %s\n" tag (Exn.to_string exn);
          valid_ok := false);
      let dims = LL.launch_dims llc in
      let want_z = if q.Autotune.sk_batch_grid then batch_product else 1 in
      if dims.LL.grid.(2) <> want_z then grid_z_ok := false;
      if q.Autotune.sk_batch_grid then
        (* The fold arithmetic, straight off the annotated loops. Both legs have exactly two batch
           loops: the innermost batch slot (2) decodes with stride 1 under a modulo of its own
           extent, the outermost (3) with that extent as stride and no modulo. *)
        let axes = LL.hardware_axes llc in
        let max_slot =
          List.fold axes ~init:(-1) ~f:(fun m a ->
              match a.LL.ha_kind with `Grid -> max m a.LL.ha_slot | `Workgroup -> m)
        in
        let slot_max s =
          List.fold axes ~init:1 ~f:(fun m a ->
              match a.LL.ha_kind with `Grid when a.LL.ha_slot = s -> max m a.LL.ha_extent | _ -> m)
        in
        let stride_at slot = fst (LL.grid_fold axes ~slot) in
        let cap_at slot = snd (LL.grid_fold axes ~slot) in
        if
          not
            (max_slot = 3
            && stride_at 2 = 1
            && Poly.equal (cap_at 2) (Some (slot_max 2))
            && stride_at 3 = slot_max 2
            && Option.is_none (cap_at 3)
            && slot_max 2 * slot_max 3 = batch_product)
        then fold_ok := false);
  p (tag ^ ": every seed's schedule constructs and validates") !valid_ok;
  p
    (tag ^ ": batch-grid twins launch grid.z = batch product; serial twins launch grid.z = 1")
    !grid_z_ok;
  p (tag ^ ": the fold arithmetic decodes innermost-mod, outermost-div") !fold_ok;
  (* --- Executable parity, seed by seed, on backends that can run shared staging. --- *)
  if on_gpu then begin
    let n_ran = ref 0 and n_match = ref 0 in
    (* Every seed compiles under the one routine name [<tag>_sched], so each overwrites the previous
       seed's artifact. Arming before each compile makes the read below this seed's kernel or
       nothing -- until gh-ocannl-655 the fold was credited to whichever seed happened to have
       written last, which is how a post-loop grep for the fold found a later serial seed's kernel
       and read 0 occurrences (gh-ocannl-643). Each batch-grid seed is now checked in its own right
       rather than one standing for all. *)
    let n_fold_seeds = ref 0 and n_folded = ref 0 in
    List.iter seeds ~f:(fun q ->
        Generated.arm (tag ^ "_sched");
        match
          let ctx, routine =
            Context.compile
              ~lowered_transform:(fun o -> [ Sched.apply (Autotune.sketch_schedule ~p:q o) o ])
              (Context.auto ()) fwd Ir.Indexing.Empty
          in
          Context.get_values (Context.run ctx routine) cand.Tensor.value
        with
        | got ->
            Int.incr n_ran;
            if Array.for_all2_exn got want ~f:Float.equal then Int.incr n_match;
            if q.Autotune.sk_batch_grid then (
              Int.incr n_fold_seeds;
              let src = Generated.read (tag ^ "_sched") in
              if
                String.is_substring src ~substring:fold_div
                && String.is_substring src ~substring:fold_mod
              then Int.incr n_folded)
        | exception exn -> Stdio.eprintf "%s: seed FAILED: %s\n" tag (Exn.to_string exn));
    p (tag ^ ": every seed compiles and runs") (!n_ran = List.length seeds && !n_ran > 0);
    p (tag ^ ": every candidate matches the serial reference bitwise") (!n_ran = !n_match);
    p
      (tag ^ ": the generated source folds the batch axes onto the .z register")
      (!n_fold_seeds > 0 && !n_folded = !n_fold_seeds)
  end
  else begin
    Stdio.eprintf "%s: %s cannot execute workgroup-shared staging — execution legs skipped\n" tag
      backend_name;
    skipped (tag ^ ": every seed compiles and runs");
    skipped (tag ^ ": every candidate matches the serial reference bitwise");
    skipped (tag ^ ": the generated source folds the batch axes onto the .z register")
  end

let () =
  let bb = 2 and hh = 4 and ss = 64 and jj = 32 and kk = 16 in
  (* Discriminating inputs: values vary with every index (the linear-index strides are coprime with
     the moduli), and all products are small multiples of 1/8 with partial sums far below 2^24, so
     f32 addition is exact in any order — bitwise parity is well-defined across tilings. *)
  let x () =
    NTDSL.init ~l:"bg_x" ~prec:Ir.Ops.single ~o:[ bb; ss; kk ]
      ~f:(fun idcs ->
        Float.of_int (((idcs.(0) * ss * kk) + (idcs.(1) * kk) + idcs.(2)) % 13) *. 0.25)
      ()
  in
  let w () =
    NTDSL.init ~l:"bg_w" ~prec:Ir.Ops.single ~o:[ hh; kk; jj ]
      ~f:(fun idcs ->
        (Float.of_int (((idcs.(0) * kk * jj) + (idcs.(1) * jj) + idcs.(2)) % 11) -. 5.) *. 0.5)
      ()
  in
  (* --- The q/k/v shape: rank-4 output, two outer batch loops (batch, head) --- *)
  (* Slots: j-blocks 0, s-blocks 1, h 2 (extent 4: "% 4"), b 3 (stride 4: "/ 4"). *)
  (* This is intentionally dialect identity: the source assertion spells the folded grid-z
     register as MSL [gid.z] or CUDA/HIP [blockIdx.z]. *)
  let reg = if String.is_substring backend_name ~substring:"metal" then "gid.z" else "blockIdx.z" in
  leg ~tag:"qkv" ~batch_product:(bb * hh) ~fold_div:(reg ^ " / 4") ~fold_mod:(reg ^ " % 4")
    ~build:(fun () ->
      let xv = x () and wv = w () in
      let%op out = xv +* "bsk;hkj=>bhsj" wv in
      out);

  (* --- The q/k/v shape with a materialized output feeding a bias+relu companion nest --- *)
  let bias () =
    NTDSL.init ~l:"bg_bias" ~prec:Ir.Ops.single ~o:[ jj ]
      ~f:(fun idcs -> (Float.of_int (idcs.(0) % 3) -. 1.) *. 0.5)
      ()
  in
  leg ~tag:"qkv_companion" ~batch_product:(bb * hh) ~fold_div:(reg ^ " / 4") ~fold_mod:(reg ^ " % 4")
    ~build:(fun () ->
      let xv = x () and wv = w () and bv = bias () in
      let%op z = xv +* "bsk;hkj=>bhsj" wv in
      Train.set_materialized z.Tensor.value;
      let%op y = relu (z + bv) in
      y);

  (* --- Interior batch: the head axis BETWEEN the tile roles (gh-ocannl-528's shape) --- *)
  (* Slots: j-blocks 0, i-blocks 1, h 2 (extent 2: "% 2"), b 3 (stride 2: "/ 2"). The zero nest
     and the companionless site both hoist [h] above [i]; positional slot order must match. *)
  let bt = 2 and ss2 = 16 and hh2 = 2 and kk2 = 8 and jj2 = 32 in
  let att () =
    NTDSL.init ~l:"bg_att" ~prec:Ir.Ops.single ~o:[ bt; ss2; hh2; kk2 ]
      ~f:(fun idcs ->
        Float.of_int
          (((idcs.(0) * ss2 * hh2 * kk2) + (idcs.(1) * hh2 * kk2) + (idcs.(2) * kk2) + idcs.(3))
          % 11)
        *. 0.125)
      ()
  in
  let v () =
    NTDSL.init ~l:"bg_v" ~prec:Ir.Ops.single ~o:[ bt; kk2; hh2; jj2 ]
      ~f:(fun idcs ->
        (Float.of_int
           (((idcs.(0) * kk2 * hh2 * jj2) + (idcs.(1) * hh2 * jj2) + (idcs.(2) * jj2) + idcs.(3))
           % 7)
        -. 3.)
        *. 0.5)
      ()
  in
  leg ~tag:"interior" ~batch_product:(bt * hh2) ~fold_div:(reg ^ " / 2") ~fold_mod:(reg ^ " % 2")
    ~build:(fun () ->
      let a = att () and vv = v () in
      let%op out = a +* "bihk;bkhj=>bihj" vv in
      out);

  (* --- Interior batch with a companion: the companion nest's own head loop must hoist above its
     row loop ([companion_role_ops]'s Swaps on the companion's symbols) or its positional slot order
     diverges from the site's and validation rejects the write coverage. --- *)
  let bias2 () =
    NTDSL.init ~l:"bg_bias2" ~prec:Ir.Ops.single ~o:[ jj2 ]
      ~f:(fun idcs -> (Float.of_int (idcs.(0) % 3) -. 1.) *. 0.5)
      ()
  in
  leg ~tag:"interior_companion" ~batch_product:(bt * hh2) ~fold_div:(reg ^ " / 2")
    ~fold_mod:(reg ^ " % 2") ~build:(fun () ->
      let a = att () and vv = v () and bv = bias2 () in
      let%op z2 = a +* "bihk;bkhj=>bihj" vv in
      Train.set_materialized z2.Tensor.value;
      let%op y2 = relu (z2 + bv) in
      y2);

  (* --- The tensorized (mma) pipeline's batch-grid twins, construction and validation only --- A
     synthetic f32 mma capability makes the tensorized branch seedable machine-independently;
     execution stays with the real-capability legs of schedule_batched_mma (an f32 tile is not a
     hardware format on the wmma backends). What must hold structurally: the twins are seeded, the
     schedules construct, validate, and launch with the folded batch [.z] extent. *)
  let mma_limits =
    {
      Ir.Backend_intf.no_hardware_limits with
      mma =
        Some
          {
            Ir.Backend_intf.mma_simd_width = 32;
            mma_tile = (8, 8, 8);
            mma_format_tiles =
              [
                ( (Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32),
                  (8, 8, 8) );
              ];
            mma_f16_wide_acc_scopes = [];
            mma_bf16_wide_acc_scopes = [];
            mma_staged_layouts = [];
            mma_pipeline_depths = [];
          };
    }
  in
  let cand =
    let xv = x () and wv = w () in
    let%op out = xv +* "bsk;hkj=>bhsj" wv in
    out
  in
  let captured = ref None in
  let _ctx, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      (Context.auto ())
      (named "qkv_mma" (Train.forward cand))
      Ir.Indexing.Empty
  in
  let opt = Option.value_exn ~here:[%here] !captured in
  let mma_grid_seeds =
    Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:mma_limits opt
    |> List.filter ~f:(fun q ->
        q.Autotune.sk_mma && q.Autotune.sk_batch_grid && not q.Autotune.sk_epilogue)
  in
  p "qkv_mma: tensorized batch-grid twins are seeded" (not (List.is_empty mma_grid_seeds));
  p_all
    "qkv_mma: every tensorized batch-grid twin constructs, validates, and folds the batch onto .z"
    mma_grid_seeds ~f:(fun q ->
      match Sched.apply (Autotune.sketch_schedule ~p:q opt) opt with
      | o -> (
          match LL.validate_parallel o.LL.optimize_ctx.LL.placements o.LL.llc with
          | () -> (LL.launch_dims o.LL.llc).LL.grid.(2) = bb * hh
          | exception exn ->
              Stdio.eprintf "qkv_mma: validate FAILED: %s\n" (Exn.to_string exn);
              false)
      | exception exn ->
          Stdio.eprintf "qkv_mma: construct FAILED: %s\n" (Exn.to_string exn);
          false);

  (* --- The pre-driver gate for the launch dimensions: [validate_parallel] deliberately accepts any
     grid geometry (it is backend-independent), so [Schedule.check_hardware_limits_classified] is
     where a backend's [max_grid_yz] refuses an over-cap extent as a typed [Resource_exceeded] —
     covering hand-built schedules and future annotators, not only these seeds; and the seeding
     guard reads the same limit, so a tight cap also stops the twins from being proposed at all. One
     limit field (CUDA and HIP cap [gridDim.y] and [gridDim.z] at the same 65535), two typed
     resources, because the two extents are shrunk by different knobs. --- *)
  let bg_seed = List.find_exn (blocktile_seeds opt) ~f:(fun q -> q.Autotune.sk_batch_grid) in
  let o = Sched.apply (Autotune.sketch_schedule ~p:bg_seed opt) opt in
  let limits_yz n = { Ir.Backend_intf.no_hardware_limits with max_grid_yz = Some n } in
  p "limit gate: a folded .z extent at the device limit passes"
    (match
       Sched.check_hardware_limits_classified ~name:"qkv_bg" ~limits:(limits_yz (bb * hh)) o
     with
    | () -> true
    | exception _ -> false);
  p "limit gate: a folded .z extent beyond the device limit is a typed Resource_exceeded"
    (match
       Sched.check_hardware_limits_classified ~name:"qkv_bg" ~limits:(limits_yz ((bb * hh) - 1)) o
     with
    | () -> false
    | exception
        Ir.Schedule_outcome.Cause_at
          ( _,
            Ir.Schedule_outcome.Resource_exceeded
              { resource = Ir.Schedule_outcome.Grid_z_extent; _ } ) ->
        true
    | exception _ -> false);
  p_none "limit gate: a max_grid_yz below the batch product stops the twins at seeding"
    (Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:(limits_yz ((bb * hh) - 1)) opt)
    ~f:(fun q -> q.Autotune.sk_batch_grid);

  (* The [.y] gate is the same check one dimension over, and has nothing to do with the fold:
     [grid.(1)] is a blocktiled matmul's row-block count, which grows with the site's m-extent alone
     (at [bm = 16] an m-extent past ~1M rows is already over the 65535 cap). A serial-batch seed
     isolates it: its [.z] extent is 1, so only the [.y] check can fire, and the typed resource says
     which dimension asked too much. The seed is SELECTED for a row-block count above 1 rather than
     taken first: since gh-ocannl-730 the padded geometries include block tiles as wide as the whole
     site, and such a seed's single row block would leave the [.y] gate with nothing to compare. *)
  let y_ref =
    List.find_map (blocktile_seeds opt) ~f:(fun q ->
        if q.Autotune.sk_batch_grid then None
        else
          let os = Sched.apply (Autotune.sketch_schedule ~p:q opt) opt in
          let d = LL.launch_dims os.LL.llc in
          if d.LL.grid.(2) = 1 && d.LL.grid.(1) > 1 then Some (os, d) else None)
  in
  p "limit gate: the .y reference isolates the row blocks — its .z extent is 1 and its .y exceeds 1"
    (Option.is_some y_ref);
  let os, sdims = Option.value_exn ~here:[%here] y_ref in
  let grid_y = sdims.LL.grid.(1) in
  p "limit gate: a .y grid extent at the device limit passes"
    (match Sched.check_hardware_limits_classified ~name:"qkv_y" ~limits:(limits_yz grid_y) os with
    | () -> true
    | exception _ -> false);
  p "limit gate: a .y grid extent beyond the device limit is a typed Resource_exceeded"
    (match
       Sched.check_hardware_limits_classified ~name:"qkv_y" ~limits:(limits_yz (grid_y - 1)) os
     with
    | () -> false
    | exception
        Ir.Schedule_outcome.Cause_at
          ( _,
            Ir.Schedule_outcome.Resource_exceeded
              { resource = Ir.Schedule_outcome.Grid_y_extent; _ } ) ->
        true
    | exception _ -> false);

  (* The same gate one axis kind over: the WORKGROUP's per-dimension caps (gh-ocannl-679), which
     [max_threads_per_workgroup] does not imply because it caps only the thread product. The
     blocktile seed launches two Workgroup dimensions of [bm/tm] x [bn/tn] threads, so its [.y]
     entry is load-bearing on a schedule the seeding path actually produces — [launch_dim_gate.ml]
     covers the three-dimensional case (CUDA's [.z] cliff) on a hand-built nest, since no in-tree
     annotator emits three nested Workgroup loops. The product cap here is set to exactly the seed's
     own product, so the product row can never be what fires. *)
  let sblock = sdims.LL.block in
  let wg_limits dims =
    {
      Ir.Backend_intf.no_hardware_limits with
      max_threads_per_workgroup = Some (Array.fold sblock ~init:1 ~f:( * ));
      max_workgroup_dims = Some dims;
    }
  in
  p "limit gate: the blocktile seed's workgroup is 2-D — its .y thread extent exceeds 1"
    (sblock.(1) > 1);
  p "limit gate: per-dimension caps at the seed's own workgroup shape pass"
    (match
       Sched.check_hardware_limits_classified ~name:"qkv_wg"
         ~limits:(wg_limits (sblock.(0), sblock.(1), sblock.(2)))
         os
     with
    | () -> true
    | exception _ -> false);
  p
    "limit gate: a .y workgroup cap below the seed's extent is a typed Workgroup_y_extent, though \
     the thread product is legal"
    (match
       Sched.check_hardware_limits_classified ~name:"qkv_wg"
         ~limits:(wg_limits (sblock.(0), sblock.(1) - 1, sblock.(2)))
         os
     with
    | () -> false
    | exception
        Ir.Schedule_outcome.Cause_at
          ( _,
            Ir.Schedule_outcome.Resource_exceeded
              { resource = Ir.Schedule_outcome.Workgroup_y_extent; _ } ) ->
        true
    | exception _ -> false)

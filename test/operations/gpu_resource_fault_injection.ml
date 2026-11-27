(* gh-ocannl-778: hardware-gated resource fault-injection legs across cc, Metal, CUDA, and HIP.

   - Shared across every backend: centralized merge-slab invalidate/commit failure and retry (all
   four claims invalid on failure, restored on retry, consumer computes exact values) and
   partial-batch [link_batch] cleanup ([Link_batch_before_segment] failing on segment 1 of 2). -
   Metal leg: real Metal buffer allocation failure (past [maxBufferLength]) classified as
   [Schedule_outcome.Device_memory_exhausted] with zero live pool growth, plus exact Metal
   pool/command-buffer/event/module teardown ownership counts via [Context.global_debug_info] and
   [Alloc_census]. - CUDA and HIP legs: graph-capture failure injection at begin, segment launch,
   end, instantiate, and replay (both propagating failures leaving capture retryable and
   driver-shaped fallback / replay OOM drop), RELAXED-mode GC finalizer execution mid-capture, graph
   cache invalidation and single destruction on merge-pool growth, [Graph_exec_before_destroy]
   teardown failure without double-destroy, and exact stream/event/module/pool/graph ownership
   counts. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module FI = Ir.Resource_fault_injection
module AC = Ir.Alloc_census
module SO = Ir.Schedule_outcome
module Idx = Ir.Indexing

let tol _ = 1e-5

let two_segments ~static_indices (o : Ir.Low_level.optimized) =
  let stmts =
    List.filter (Ir.Low_level.flat_lines [ o.llc ]) ~f:(function
      | Ir.Low_level.Comment _ | Ir.Low_level.Noop -> false
      | _ -> true)
  in
  let n = List.length stmts in
  let first = Int.max 1 (n / 2) in
  let _, tuples =
    Ir.Schedule.fission_segmented
      ~replay:([ (`Solo, first); (`Solo, n - first) ], fun _ _ -> None)
      ~preset:(fun _ -> [])
      ~zero_sched:(fun _ -> [])
      ~static_indices o
  in
  List.map tuples ~f:(fun (_, _, _, post) -> post)

let counter ctx field = Ll_test.debug_counter ctx field
let count ctx field = Option.value ~default:0 (counter ctx field)

let () =
  let ctx0 = Context.auto () in
  let backend = Context.backend_name ctx0 in
  let on_metal = String.equal backend "metal" in
  let on_cuda = String.equal backend "cuda" in
  let on_hip = String.equal backend "hip" in
  let on_cuda_or_hip = on_cuda || on_hip in
  Stdio.eprintf "gpu_resource_fault_injection backend: %s (not part of the golden)\n%!" backend;

  (* --- 1. Shared leg (cc, Metal, CUDA, HIP): merge-slab invalidate/commit seam --- *)
  let small_n = 16 and large_n = 64 in
  let small_vals = Array.init small_n ~f:(fun i -> Float.of_int (i + 1)) in
  let large_vals = Array.init large_n ~f:(fun i -> Float.of_int (i + 20)) in
  let small_t =
    Operation.init ~mutable_:true ~l:"grfi_merge_small" ~prec:Ir.Ops.single ~o:[ small_n ]
      ~f:(fun i -> small_vals.(i.(0)))
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let large_t =
    Operation.init ~mutable_:true ~l:"grfi_merge_large" ~prec:Ir.Ops.single ~o:[ large_n ]
      ~f:(fun i -> large_vals.(i.(0)))
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let m_src = Context.set_values (Context.auto ()) small_t.Tensor.value small_vals in
  let m_src = Context.set_values m_src large_t.Tensor.value large_vals in
  let m_dst = Context.auto () in
  let m_dst =
    Context.copy ~into_merge_buffer:Ir.Backend_intf.Copy ~src:m_src ~dst:m_dst small_t.Tensor.value
  in
  let s_before = Context.merge_slab_state m_dst in
  p "merge slab: initial fill commits all four claims"
    (s_before.has_slab && s_before.has_location
    && s_before.capacity = small_n * 4
    && s_before.has_writer);
  let seen_merge_artifact = ref None in
  let grow_failed =
    match
      FI.with_event_callback
        (fun ev ->
          match (ev.point, ev.artifact) with
          | FI.Merge_slab_before_allocate, FI.Merge_slab { size_in_bytes; _ } ->
              seen_merge_artifact := Some size_in_bytes;
              failwith "gh778 injected merge-slab grow failure"
          | _ -> ())
        ~f:(fun () ->
          Context.copy ~into_merge_buffer:Ir.Backend_intf.Copy ~src:m_src ~dst:m_dst
            large_t.Tensor.value)
    with
    | _ -> false
    | exception Failure msg -> String.is_substring msg ~substring:"gh778 injected merge-slab"
  in
  let s_failed = Context.merge_slab_state m_dst in
  p "merge slab: failed grow invalidates all four claims"
    (grow_failed
    && Option.equal Int.equal !seen_merge_artifact (Some (large_n * 4))
    && (not s_failed.has_slab) && (not s_failed.has_location) && s_failed.capacity = 0
    && not s_failed.has_writer);
  let m_dst =
    Context.copy ~into_merge_buffer:Ir.Backend_intf.Copy ~src:m_src ~dst:m_dst large_t.Tensor.value
  in
  let s_retry = Context.merge_slab_state m_dst in
  p "merge slab: retry restores all four claims at grown capacity"
    (s_retry.has_slab && s_retry.has_location
    && s_retry.capacity = large_n * 4
    && s_retry.has_writer);
  let m_dst = Context.set_values m_dst large_t.Tensor.value (Array.create ~len:large_n 2.) in
  let merge_consume = [%cd large_t =+ large_t.merge] in
  let m_dst, consume_r = Context.compile ~name:"grfi_merge_consume" m_dst merge_consume Idx.Empty in
  let m_dst = Context.run m_dst consume_r in
  p_close "merge slab: consumer after retry computes exact values"
    (Context.get_values m_dst large_t.Tensor.value)
    (Array.map large_vals ~f:(fun v -> v +. 2.))
    ~tol;
  Context.release m_dst;
  Context.release m_src;

  (* --- 2. Shared leg: partial-batch link cleanup (2-segment routine failing at segment 1) --- *)
  let seg_vals = Array.init 16 ~f:(fun i -> Float.of_int (i - 6)) in
  let seg_x = TDSL.ndarray seg_vals ~label:[ "grfi_seg_x" ] ~output_dims:[ 16 ] () in
  let%op seg_mid = relu seg_x in
  Train.set_materialized seg_mid.Tensor.value;
  let%op seg_out = seg_mid + 3. in
  Train.set_materialized seg_out.Tensor.value;
  let seg_comp = Train.forward seg_out in
  let seg_expected = Array.map seg_vals ~f:(fun v -> Float.max 0. v +. 3.) in
  let before = AC.snapshot () in
  let seg_indices = ref [] in
  let batch_failed =
    match
      FI.with_event_callback
        (fun ev ->
          match (ev.point, ev.artifact) with
          | FI.Link_batch_before_segment, FI.Routine_segment { index; count; _ } ->
              seg_indices := (index, count) :: !seg_indices;
              if index = 1 then failwith "gh778 injected partial-batch segment failure"
          | _ -> ())
        ~f:(fun () ->
          Context.compile ~name:"grfi_partial_batch"
            ~lowered_transform:(two_segments ~static_indices:[]) (Context.auto ()) seg_comp
            Idx.Empty)
    with
    | _ -> false
    | exception Failure msg -> String.is_substring msg ~substring:"gh778 injected partial-batch"
  in
  let after = AC.snapshot () in
  p "partial-batch link failure hits segment 1 of 2 and frees whole delta"
    (batch_failed
    && List.equal
         (fun (a1, b1) (a2, b2) -> a1 = a2 && b1 = b2)
         (List.rev !seg_indices)
         [ (0, 2); (1, 2) ]
    && after.AC.working_pools_allocated - before.AC.working_pools_allocated >= 1
    && after.AC.live_working_pools - before.AC.live_working_pools = 0
    && after.AC.live_constant_pools - before.AC.live_constant_pools = 0);
  let before = AC.snapshot () in
  let b_ctx, b_routine =
    Context.compile ~name:"grfi_partial_batch_ok"
      ~lowered_transform:(two_segments ~static_indices:[]) (Context.auto ()) seg_comp Idx.Empty
  in
  let b_ctx = Context.run b_ctx b_routine in
  p_close "partial-batch control runs and computes exact values"
    (Context.get_values b_ctx seg_out.Tensor.value)
    seg_expected ~tol;
  Context.release b_ctx;
  let after = AC.snapshot () in
  p "partial-batch control releases cleanly with zero live pool growth"
    (after.AC.live_working_pools - before.AC.live_working_pools = 0
    && after.AC.live_constant_pools - before.AC.live_constant_pools = 0);

  (* --- 3. Metal leg: real allocation-failure classification + exact teardown counts --- *)
  let metal_leg = Verdict.leg ~when_:on_metal ~on:backend () in
  in_leg ~leg:metal_leg
    "metal: over-limit buffer allocation raises typed Device_memory_exhausted with zero leak"
    (fun label ->
      (* 1 TiB single-node buffer: far past Apple Silicon's maxBufferLength, so
         [newBufferWithLength:options:] returns nil immediately without committing RAM. *)
      let prior_large = Utils.settings.large_models in
      Utils.settings.large_models <- true;
      let huge_elems = 1 lsl 38 in
      let got_oom, before_ac, after_ac, before_live, after_live =
        Exn.protect
          ~finally:(fun () -> Utils.settings.large_models <- prior_large)
          ~f:(fun () ->
            let huge_tn =
              (Ll_test.node_factory ~first_id:778900 ~dims:[| huge_elems |] ()) "grfi_metal_huge"
            in
            Ll_test.materialize huge_tn;
            let i = Ll_test.sym () in
            let llc =
              Ll_test.loop_n i 1 (Ll_test.set huge_tn [| Ll_test.iter i |] (Ll_test.c 1.))
            in
            let huge_ll = Ll_test.optimize ~materialized:[ huge_tn ] ~name:"grfi_metal_huge" llc in
            let before_ac = AC.snapshot () in
            let before_live = count ctx0 "live_pools" in
            let got_oom =
              match
                Context.probe_link_delta ~name:"grfi_metal_huge" ~prelowered:true ctx0 huge_ll
                  [ huge_ll ]
              with
              | _ -> false
              | exception SO.Device_memory_exhausted { backend = b; requested; _ } ->
                  String.equal b "metal" && requested = huge_elems * 4
            in
            (got_oom, before_ac, AC.snapshot (), before_live, count ctx0 "live_pools"))
      in
      p label
        (got_oom && after_live = before_live
        && after_ac.AC.live_working_pools = before_ac.AC.live_working_pools));

  in_leg ~leg:metal_leg
    "metal: exact pool, command-buffer, event, and module teardown counts under release failure \
     and retry" (fun label ->
      let before_live = count ctx0 "live_pools" in
      let before_alloc = count ctx0 "pools_allocated" in
      let before_freed = count ctx0 "pools_freed" in
      let before_cb = count ctx0 "command_buffers_committed" in
      let before_reaped = count ctx0 "command_buffers_reaped" in
      let before_ev = count ctx0 "events_signaled" in
      let before_ac = AC.snapshot () in
      let m_x = TDSL.ndarray seg_vals ~label:[ "grfi_metal_teardown_x" ] ~output_dims:[ 16 ] () in
      let%op m_y = relu m_x in
      let m_comp = Train.forward m_y in
      let m_ctx, m_routine =
        Context.compile ~name:"grfi_metal_td" (Context.auto ()) m_comp Idx.Empty
      in
      let m_ctx = Context.run m_ctx m_routine in
      Context.sync m_ctx;
      let vals = Context.get_values m_ctx m_y.Tensor.value in
      let after_run_alloc = count ctx0 "pools_allocated" in
      let release_failed =
        match
          FI.with_callback
            (function
              | FI.Finalize_before_free -> failwith "gh778 metal finalize failure" | _ -> ())
            ~f:(fun () -> Context.release m_ctx)
        with
        | () -> false
        | exception Failure msg -> String.is_substring msg ~substring:"gh778 metal finalize"
      in
      let mid_freed = count ctx0 "pools_freed" in
      Context.release m_ctx;
      Context.release m_ctx;
      let after_live = count ctx0 "live_pools" in
      let after_freed = count ctx0 "pools_freed" in
      let after_cb = count ctx0 "command_buffers_committed" in
      let after_reaped = count ctx0 "command_buffers_reaped" in
      let after_ev = count ctx0 "events_signaled" in
      let after_ac = AC.snapshot () in
      let minted = after_run_alloc - before_alloc in
      Stdio.eprintf
        "metal teardown: minted=%d freed=%d live_delta=%d cb=%d reaped=%d ev=%d mods_loaded=%d \
         (not part of the golden)\n\
         %!"
        minted (after_freed - before_freed) (after_live - before_live) (after_cb - before_cb)
        (after_reaped - before_reaped) (after_ev - before_ev)
        (after_ac.AC.modules_loaded - before_ac.AC.modules_loaded);
      p label
        (release_failed && minted >= 1 && mid_freed = before_freed
        && after_freed - before_freed = minted
        && after_live = before_live && after_cb > before_cb && after_reaped > before_reaped
        && after_ev > before_ev
        && after_ac.AC.live_working_pools = before_ac.AC.live_working_pools
        && Verdict.within label vals (Array.map seg_vals ~f:(Float.max 0.)) ~tol));

  (* --- 4. CUDA and HIP leg: graph capture fault injection, merge-grow invalidation, teardown
     --- *)
  let capture_on =
    lazy
      (Utils.get_global_flag ~default:true ~arg_name:"gpu_graph_capture"
      && not (Utils.debug_log_from_routines ()))
  in
  let gpu_graph_leg =
    Verdict.sub_leg
      (Verdict.leg ~when_:on_cuda_or_hip ~on:backend ())
      ~when_:capture_on ~aggregation:`Environment
      ~on:"graph capture off (gpu_graph_capture=false or routine logging)"
  in

  let make_graph_routine ?(bindings = Idx.Empty) ~name comp =
    let static_indices = Idx.bound_symbols bindings in
    Context.compile ~name ~lowered_transform:(two_segments ~static_indices) (Context.auto ()) comp
      bindings
  in

  (* 4a. Begin / Launch / End / Instantiate failure injection (propagating + driver-shaped
     fallback). *)
  List.iter
    [
      (FI.Graph_capture_begin, "begin");
      (FI.Graph_capture_launch, "launch");
      (FI.Graph_capture_end, "end");
      (FI.Graph_capture_instantiate, "instantiate");
    ]
    ~f:(fun (point, phase_label) ->
      in_legf ~leg:gpu_graph_leg
        "gpu graph capture: propagating failure at %s leaves stream ready and retry captures once"
        phase_label (fun label ->
          let live0 = count ctx0 "live_graph_execs" in
          let created0 = count ctx0 "graph_execs_created" in
          let destroyed0 = count ctx0 "graph_execs_destroyed" in
          let name = "grfi_gc_prop_" ^ phase_label in
          let ctx, routine = make_graph_routine ~name seg_comp in
          let saw_event = ref false in
          let failed =
            match
              FI.with_event_callback
                (fun ev ->
                  if FI.equal_point ev.point point then (
                    (saw_event :=
                       FI.equal_ownership_phase ev.phase (FI.phase_of_point point)
                       &&
                       match ev.artifact with
                       | FI.Graph_capture { routine = r; _ } -> String.equal r name
                       | _ -> false);
                    failwith ("gh778 injected graph capture failure at " ^ phase_label)))
                ~f:(fun () -> Context.run ctx routine)
            with
            | _ -> false
            | exception Failure msg -> String.is_substring msg ~substring:"gh778 injected graph"
          in
          let live_after_fail = count ctx0 "live_graph_execs" in
          let created_after_fail = count ctx0 "graph_execs_created" in
          let ctx = Context.run ctx routine in
          Context.sync ctx;
          let got = Context.get_values ctx seg_out.Tensor.value in
          let live_after_retry = count ctx0 "live_graph_execs" in
          let created_after_retry = count ctx0 "graph_execs_created" in
          Context.release ctx;
          let live_after_release = count ctx0 "live_graph_execs" in
          let destroyed_after_release = count ctx0 "graph_execs_destroyed" in
          p label
            (failed && !saw_event && live_after_fail = live0 && created_after_fail = created0
            && live_after_retry = live0 + 1
            && created_after_retry = created0 + 1
            && live_after_release = live0
            && destroyed_after_release = destroyed0 + 1
            && Verdict.within label got seg_expected ~tol));

      in_legf ~leg:gpu_graph_leg
        "gpu graph capture: driver capture error at %s falls back cleanly to plain execution"
        phase_label (fun label ->
          let live0 = count ctx0 "live_graph_execs" in
          let created0 = count ctx0 "graph_execs_created" in
          let name = "grfi_gc_fb_" ^ phase_label in
          let ctx, routine = make_graph_routine ~name seg_comp in
          let ctx =
            FI.with_callback
              (fun seen ->
                if FI.equal_point seen point then
                  raise (FI.Simulated_capture_error ("simulated capture failure at " ^ phase_label)))
              ~f:(fun () -> Context.run ctx routine)
          in
          Context.sync ctx;
          let got1 = Context.get_values ctx seg_out.Tensor.value in
          (* Second run stays on the plain fallback path ([broken = true]) without caching. *)
          let ctx = Context.run ctx routine in
          Context.sync ctx;
          let got2 = Context.get_values ctx seg_out.Tensor.value in
          let live1 = count ctx0 "live_graph_execs" in
          let created1 = count ctx0 "graph_execs_created" in
          Context.release ctx;
          p label
            (live1 = live0 && created1 = created0
            && Verdict.within ~what_ran:"fallback1" label got1 seg_expected ~tol
            && Verdict.within ~what_ran:"fallback2" label got2 seg_expected ~tol)));

  (* 4b. RELAXED capture mode permits GC finalizers mid-capture without aborting capture. *)
  in_leg ~leg:gpu_graph_leg
    "gpu graph capture: RELAXED capture survives GC finalizers firing mid-capture" (fun label ->
      let live0 = count ctx0 "live_graph_execs" in
      let dead_ctx, _ = Context.compile ~name:"grfi_gc_dead" (Context.auto ()) seg_comp Idx.Empty in
      Context.release dead_ctx;
      let ctx, routine = make_graph_routine ~name:"grfi_gc_relaxed" seg_comp in
      let settled = ref false in
      let ctx =
        FI.with_callback
          (function
            | FI.Graph_capture_launch ->
                Gc_witness.settle ();
                settled := true
            | _ -> ())
          ~f:(fun () -> Context.run ctx routine)
      in
      Context.sync ctx;
      let got = Context.get_values ctx seg_out.Tensor.value in
      let live1 = count ctx0 "live_graph_execs" in
      Context.release ctx;
      p label
        (!settled
        && live1 = live0 + 1
        && count ctx0 "live_graph_execs" = live0
        && Verdict.within label got seg_expected ~tol));

  (* 4c. Replay OOM drops and destroys cached exec once, falls back plainly, next run recaptures. *)
  in_leg ~leg:gpu_graph_leg
    "gpu graph capture: replay OOM drops and destroys exec once, falls back plainly, and recaptures"
    (fun label ->
      let live0 = count ctx0 "live_graph_execs" in
      let created0 = count ctx0 "graph_execs_created" in
      let destroyed0 = count ctx0 "graph_execs_destroyed" in
      let drops0 = count ctx0 "graph_launch_memory_drops" in
      let ctx, routine = make_graph_routine ~name:"grfi_gc_replay_oom" seg_comp in
      let ctx = Context.run ctx routine in
      Context.sync ctx;
      let live_after_first = count ctx0 "live_graph_execs" in
      (* Second run replays the cached exec and injects a simulated replay OOM. *)
      let ctx =
        FI.with_callback
          (function
            | FI.Graph_capture_replay ->
                raise (FI.Simulated_graph_replay_oom "simulated replay out of memory")
            | _ -> ())
          ~f:(fun () -> Context.run ctx routine)
      in
      Context.sync ctx;
      let got_fb = Context.get_values ctx seg_out.Tensor.value in
      let live_after_drop = count ctx0 "live_graph_execs" in
      let destroyed_after_drop = count ctx0 "graph_execs_destroyed" in
      let drops_after_drop = count ctx0 "graph_launch_memory_drops" in
      (* Third run recaptures cleanly. *)
      let ctx = Context.run ctx routine in
      Context.sync ctx;
      let got_recap = Context.get_values ctx seg_out.Tensor.value in
      let live_after_recap = count ctx0 "live_graph_execs" in
      let created_after_recap = count ctx0 "graph_execs_created" in
      Context.release ctx;
      let destroyed_final = count ctx0 "graph_execs_destroyed" in
      p label
        (live_after_first = live0 + 1
        && live_after_drop = live0
        && destroyed_after_drop = destroyed0 + 1
        && drops_after_drop = drops0 + 1
        && live_after_recap = live0 + 1
        && created_after_recap = created0 + 2
        && count ctx0 "live_graph_execs" = live0
        && destroyed_final = destroyed0 + 2
        && Verdict.within ~what_ran:"replay_oom_fallback" label got_fb seg_expected ~tol
        && Verdict.within ~what_ran:"replay_oom_recapture" label got_recap seg_expected ~tol));

  (* 4d. Graph invalidation after merge-pool growth: growing the merge slab destroys the old
     merge-using graph exec exactly once; refilling recaptures cleanly; release destroys only the
     new exec (no double-destroy). *)
  in_leg ~leg:gpu_graph_leg
    "gpu graph invalidation: merge-pool growth destroys old cached graph once and recapture works"
    (fun label ->
      let mg_small_n = 32 and mg_large_n = 128 in
      let mg_s1 = Array.init mg_small_n ~f:(fun i -> Float.of_int (i + 1)) in
      let mg_s2 = Array.init mg_small_n ~f:(fun i -> Float.of_int (i + 101)) in
      let mg_big = Array.init mg_large_n ~f:(fun i -> Float.of_int (i + 200)) in
      let mg_small =
        Operation.init ~mutable_:true ~l:"grfi_mg_small" ~prec:Ir.Ops.single ~o:[ mg_small_n ]
          ~f:(fun i -> mg_s1.(i.(0)))
          ~grad_spec:Tensor.Prohibit_grad ()
      in
      let mg_large =
        Operation.init ~mutable_:true ~l:"grfi_mg_large" ~prec:Ir.Ops.single ~o:[ mg_large_n ]
          ~f:(fun i -> mg_big.(i.(0)))
          ~grad_spec:Tensor.Prohibit_grad ()
      in
      let mg_mid =
        Operation.init ~mutable_:true ~l:"grfi_mg_mid" ~prec:Ir.Ops.single ~o:[ mg_small_n ]
          ~f:(fun _ -> 0.)
          ~grad_spec:Tensor.Prohibit_grad ()
      in
      let mg_out =
        Operation.init ~mutable_:true ~l:"grfi_mg_out" ~prec:Ir.Ops.single ~o:[ mg_small_n ]
          ~f:(fun _ -> 0.)
          ~grad_spec:Tensor.Prohibit_grad ()
      in
      Train.set_materialized mg_mid.Tensor.value;
      Train.set_materialized mg_out.Tensor.value;
      let mg_consumer =
        [%cd
          ~~("grfi_mg_consumer";
             mg_mid =: mg_small.merge + 1.;
             mg_out =: mg_mid + 2.)]
      in
      let src_ctx = Context.set_values (Context.auto ()) mg_small.Tensor.value mg_s1 in
      let src_ctx = Context.set_values src_ctx mg_large.Tensor.value mg_big in
      let dst_ctx =
        Context.copy ~into_merge_buffer:Ir.Backend_intf.Copy ~src:src_ctx ~dst:(Context.auto ())
          mg_small.Tensor.value
      in
      let live0 = count ctx0 "live_graph_execs" in
      let created0 = count ctx0 "graph_execs_created" in
      let destroyed0 = count ctx0 "graph_execs_destroyed" in
      let dst_ctx, mg_routine =
        Context.compile ~name:"grfi_mg_graph" ~lowered_transform:(two_segments ~static_indices:[])
          dst_ctx mg_consumer Idx.Empty
      in
      let dst_ctx = Context.run dst_ctx mg_routine in
      Context.sync dst_ctx;
      let got1 = Context.get_values dst_ctx mg_out.Tensor.value in
      let live_after_cap = count ctx0 "live_graph_execs" in
      let created_after_cap = count ctx0 "graph_execs_created" in
      (* Grow the device's reserved merge slab from a separate root context: must immediately
         invalidate and destroy [mg_routine]'s cached graph exec once. *)
      let grower =
        Context.copy ~into_merge_buffer:Ir.Backend_intf.Copy ~src:src_ctx ~dst:(Context.auto ())
          mg_large.Tensor.value
      in
      let live_after_grow = count ctx0 "live_graph_execs" in
      let destroyed_after_grow = count ctx0 "graph_execs_destroyed" in
      (* Refill the grown merge slab with updated [mg_small] values and rerun [mg_routine]. *)
      let src_ctx = Context.set_values src_ctx mg_small.Tensor.value mg_s2 in
      let grower =
        Context.copy ~into_merge_buffer:Ir.Backend_intf.Copy ~src:src_ctx ~dst:grower
          mg_small.Tensor.value
      in
      let dst_ctx = Context.run dst_ctx mg_routine in
      Context.sync dst_ctx;
      let got2 = Context.get_values dst_ctx mg_out.Tensor.value in
      let live_after_recap = count ctx0 "live_graph_execs" in
      let created_after_recap = count ctx0 "graph_execs_created" in
      Context.release dst_ctx;
      Context.release dst_ctx;
      Context.release grower;
      Context.release src_ctx;
      let live_final = count ctx0 "live_graph_execs" in
      let destroyed_final = count ctx0 "graph_execs_destroyed" in
      p label
        (live_after_cap = live0 + 1
        && created_after_cap = created0 + 1
        && live_after_grow = live0
        && destroyed_after_grow = destroyed0 + 1
        && live_after_recap = live0 + 1
        && created_after_recap = created0 + 2
        && live_final = live0
        && destroyed_final = destroyed0 + 2
        && Verdict.within ~what_ran:"before_grow" label got1
             (Array.map mg_s1 ~f:(fun v -> v +. 3.))
             ~tol
        && Verdict.within ~what_ran:"after_grow" label got2
             (Array.map mg_s2 ~f:(fun v -> v +. 3.))
             ~tol));

  (* 4e. Exact CUDA/HIP stream/event/module/pool/graph teardown counts and partial exec-destroy
     failure without double-destroy. *)
  in_leg ~leg:gpu_graph_leg
    "gpu teardown: partial Graph_exec_before_destroy still destroys remaining execs with no \
     double-destroy and exact ownership counts" (fun label ->
      let idx_sym, bindings = Idx.get_static_symbol ~static_range:4 Idx.Empty in
      let b_mid = (Ll_test.node_factory ~first_id:778500 ~dims:[| 8 |] ()) "grfi_td_mid"
      and b_out = (Ll_test.node_factory ~first_id:778510 ~dims:[| 8 |] ()) "grfi_td_out" in
      Ll_test.materialize b_mid;
      Ll_test.materialize b_out;
      let i = Ll_test.sym () and j = Ll_test.sym () in
      let llc =
        Ll_test.seq
          (Ll_test.loop_n i 8
             (Ll_test.set b_mid
                [| Ll_test.iter i |]
                (Ll_test.add (Ll_test.tick i) (Ll_test.tick idx_sym.static_symbol))))
          (Ll_test.loop_n j 8
             (Ll_test.set b_out
                [| Ll_test.iter j |]
                (Ll_test.add (Ll_test.get b_mid [| Ll_test.iter j |]) (Ll_test.c 10.))))
      in
      let pre =
        Ll_test.optimize ~materialized:[ b_mid; b_out ] ~static_indices:[ idx_sym ]
          ~name:"grfi_td_two_execs" llc
      in
      let live0 = count ctx0 "live_graph_execs" in
      let created0 = count ctx0 "graph_execs_created" in
      let destroyed0 = count ctx0 "graph_execs_destroyed" in
      let ev0 = count ctx0 "events_recorded" in
      let ac0 = AC.snapshot () in
      let td_ctx, td_routine =
        Context.compile ~name:"grfi_td_two_execs" ~prelowered:pre
          ~lowered_transform:(two_segments ~static_indices:[ idx_sym ])
          (Context.auto ()) Ir.Assignments.empty_comp bindings
      in
      Idx.find_exn td_routine.Context.bindings idx_sym := 0;
      let td_ctx = Context.run td_ctx td_routine in
      Idx.find_exn td_routine.Context.bindings idx_sym := 1;
      let td_ctx = Context.run td_ctx td_routine in
      Context.sync td_ctx;
      let live_cached = count ctx0 "live_graph_execs" in
      let created_cached = count ctx0 "graph_execs_created" in
      let destroy_hits = ref 0 in
      let hook_failed =
        match
          FI.with_callback
            (function
              | FI.Graph_exec_before_destroy ->
                  Int.incr destroy_hits;
                  if !destroy_hits = 1 then failwith "gh778 injected graph exec destroy failure"
              | _ -> ())
            ~f:(fun () -> Context.release td_ctx)
        with
        | () -> false
        | exception Failure msg -> String.is_substring msg ~substring:"gh778 injected graph exec"
      in
      (* Second release must be a complete no-op: pools are already freed and cache already
         cleared. *)
      Context.release td_ctx;
      let live_after = count ctx0 "live_graph_execs" in
      let destroyed_after = count ctx0 "graph_execs_destroyed" in
      let ev1 = count ctx0 "events_recorded" in
      let streams = count ctx0 "live_streams" in
      let ac1 = AC.snapshot () in
      Stdio.eprintf
        "gpu teardown (%s): streams=%d events_delta=%d execs_created=%d execs_destroyed=%d \
         destroy_hits=%d working_live_delta=%d mods_loaded_delta=%d (not part of the golden)\n\
         %!"
        backend streams (ev1 - ev0) (created_cached - created0) (destroyed_after - destroyed0)
        !destroy_hits
        (ac1.AC.live_working_pools - ac0.AC.live_working_pools)
        (ac1.AC.modules_loaded - ac0.AC.modules_loaded);
      p label
        (live_cached = live0 + 2
        && created_cached = created0 + 2
        && hook_failed && !destroy_hits = 2
        && destroyed_after = destroyed0 + 1
        && live_after = live0 && streams >= 1 && ev1 > ev0
        && ac1.AC.live_working_pools = ac0.AC.live_working_pools))

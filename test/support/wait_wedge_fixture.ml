(* Finite work only: the zero-bound production wait is the sole writer of the device mark. Each
   client is a separate executable because the mark lasts for the process. A watchdog exits red even
   if a regressed wait blocks; it never abandons a deliberately hanging kernel. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module B = Context.Backends
module FI = Ir.Resource_fault_injection
module AC = Ir.Alloc_census

let n = 1024

let comp =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.25)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "wd_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "wd_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t = mc * mb * ma in
  ignore mc;
  Ll_test.unowned_copy (Train.forward t)

let wedged = function Ir.Backend_intf.Device_wedged _ -> true | _ -> false

let watchdog () =
  ignore
    (Domain.spawn (fun () ->
         Unix.sleepf 90.;
         Stdio.eprintf "wedge regression watchdog expired\n%!";
         Stdlib.exit 1)
      : unit Domain.t)

(* A synchronous CPU routine finishes inside [run], so only the worker-domain backend can exercise
   an expired wait there. GPU selections keep the selected physical device. *)
let make_ctx () =
  let ctx = Context.auto () in
  if Ir.Backend_intf.renders_on_host_pool (Context.backend_facts ctx) then Context.cpu ~threads:2 ()
  else ctx

let setup ?(make_context = make_ctx) () =
  watchdog ();
  let ctx, routine = Context.compile (make_context ()) comp Ir.Indexing.Empty in
  Stdio.eprintf "wedge fixture backend: %s\n%!" (Context.backend_name ctx);
  let ctx = Context.run ctx routine in
  p "the healthy wait completes" (Context.await_within ctx ~seconds:60.);
  let mark ?(before_wait = fun () -> ()) () =
    let ctx = Context.run ctx routine in
    before_wait ();
    p "a zero bound expires on finite work" (not (Context.await_within ctx ~seconds:0.))
  in
  let raw = B.make_context (B.get_backend ~backend_name:(Context.backend_name ctx) ()) in
  (ctx, raw, mark)

let small () =
  let x = TDSL.ndarray [| -2.; 1.; 3.; 4. |] ~output_dims:[ 4 ] () in
  let%op y = relu x in
  (y, Ll_test.unowned_copy (Train.forward y))

let refused f = match f () with _ -> false | exception exn -> wedged exn

let uncontainable_failures =
  [
    Stdlib.Sys.Break;
    Out_of_memory;
    Stack_overflow;
    Assert_failure ("wedge test", 1, 0);
    Utils.Invalid_config_value { key = "profile"; value = "invalid"; expected = "a profile" };
  ]

let preserves_failure point failure f =
  let fired = ref false in
  match
    FI.with_callback
      (fun seen ->
        if FI.equal_point seen point then (
          fired := true;
          raise failure))
      ~f
  with
  | _ -> false
  | exception exn -> !fired && phys_equal exn failure

let cleanup point =
  let _, _, mark = setup () in
  let _, small_comp = small () in
  let control_before = AC.snapshot () in
  let control_fired = ref false in
  let control_failed =
    match
      FI.with_callback
        (fun seen ->
          if FI.equal_point seen point && not !control_fired then (
            control_fired := true;
            failwith "healthy cleanup injection"))
        ~f:(fun () -> Context.compile (make_ctx ()) small_comp Ir.Indexing.Empty)
    with
    | _ -> false
    | exception Failure _ -> true
  in
  let control_after = AC.snapshot () in
  p "the healthy allocation cleanup preserves the original failure"
    (!control_fired && control_failed);
  let allocated =
    control_after.working_pools_allocated + control_after.constant_pools_allocated
    - control_before.working_pools_allocated - control_before.constant_pools_allocated
  in
  p "the healthy allocation cleanup frees every new pool"
    (allocated > 0 && control_after.pools_freed - control_before.pools_freed = allocated);
  let before = AC.snapshot () in
  let fired = ref false in
  let result =
    refused (fun () ->
        FI.with_callback
          (fun seen ->
            if FI.equal_point seen point && not !fired then (
              fired := true;
              mark ();
              failwith "wedge cleanup injection"))
          ~f:(fun () -> Context.compile (make_ctx ()) small_comp Ir.Indexing.Empty))
  in
  let after = AC.snapshot () in
  p "the allocation failure seam fired" !fired;
  p "the failed compile propagates Device_wedged" result;
  p "the failed compile allocated pools"
    (after.working_pools_allocated + after.constant_pools_allocated
    > before.working_pools_allocated + before.constant_pools_allocated);
  p "no pool is freed under outstanding work" (after.pools_freed = before.pools_freed);
  p_all "wedged delta cleanup preserves uncontainable exceptions" uncontainable_failures
    ~f:(fun failure ->
      preserves_failure point failure (fun () ->
          Context.compile (make_ctx ()) small_comp Ir.Indexing.Empty));
  p "uncontainable delta failures also leave every pool allocated"
    ((AC.snapshot ()).pools_freed = after.pools_freed)

let event () =
  let _, raw, mark = setup () in
  let (B.Unwrapped (impl, ctx)) = B.unwrap raw in
  let module Backend = (val impl.bi_module) in
  let event = Backend.all_work ctx.device in
  Backend.sync event;
  p "the healthy event sync completes" (Backend.is_done event);
  let pending = ref event in
  mark ~before_wait:(fun () -> pending := Backend.all_work ctx.device) ();
  p "event sync on its marked owner raises Device_wedged"
    (refused (fun () -> Backend.sync !pending));
  p "another bounded wait on the mark raises Device_wedged"
    (refused (fun () -> Backend.await_within ctx.device ~seconds:60.))

let transfer ~cleanup ~device_copy =
  let _, raw, mark = setup () in
  let (B.Unwrapped (impl, ctx)) = B.unwrap raw in
  let module Backend = (val impl.bi_module) in
  let tensor =
    Operation.init ~l:"wedge_upload" ~prec:Ir.Ops.single ~o:[ 4 ]
      ~f:(fun i -> Float.of_int (i.(0) + 1))
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let tn = tensor.Tensor.value in
  let host =
    Ir.Ndarray.init_array ~debug:"wedge upload" Ir.Ops.single ~dims:[| 4 |] ~padding:None
      ~f:(fun i -> Float.of_int (i.(0) + 1))
  in
  let src = Backend.init_from_host ctx tn host in
  let control_before = AC.snapshot () in
  let control =
    if device_copy then Backend.init_from_device ~dst:ctx ~src tn
    else Backend.init_from_host ctx tn host
  in
  Backend.await control.device;
  B.finalize (module Backend) control;
  let control_after = AC.snapshot () in
  p "the healthy transfer allocates and frees its destination pool"
    (control_after.working_pools_allocated = control_before.working_pools_allocated + 1
    && control_after.pools_freed = control_before.pools_freed + 1);
  (* [ctx] was a parent of the control; use a fresh root rather than a finalized upload arena. *)
  let ctx = Backend.make_context ctx.device in
  let before = AC.snapshot () in
  let fired = ref false and cleanup_seen = ref false in
  let point = if device_copy then FI.Init_from_device_before_await else FI.From_host_before_await in
  let result =
    refused (fun () ->
        FI.with_callback
          (fun seen ->
            if FI.equal_point seen FI.Transfer_cleanup_before_await then cleanup_seen := true;
            if FI.equal_point seen point && not !fired then (
              fired := true;
              mark ();
              if cleanup then failwith "wedge transfer injection"))
          ~f:(fun () ->
            if device_copy then Backend.init_from_device ~dst:ctx ~src tn
            else Backend.init_from_host ctx tn host))
  in
  let after = AC.snapshot () in
  p "the transfer seam expired the wait" !fired;
  p "the transfer cleanup was reached" !cleanup_seen;
  p "the transfer propagates Device_wedged" result;
  p "the transfer allocated a pool" (after.working_pools_allocated > before.working_pools_allocated);
  p "the transfer leaves its pool allocated" (after.pools_freed = before.pools_freed);
  (* Inject before the copy, so these additional checks submit no work to the marked device. *)
  p_all "wedged transfer cleanup preserves uncontainable exceptions" uncontainable_failures
    ~f:(fun failure ->
      preserves_failure FI.Transfer_pool_allocated failure (fun () ->
          Backend.init_from_host ctx tn host));
  p "uncontainable transfer failures also leave every pool allocated"
    ((AC.snapshot ()).pools_freed = after.pools_freed)

(* The arm keeps its original failure; the caller's later release deliberately propagates the
   independent device-wide refusal. No handler reclassifies a failed release as an arm's cause. *)
let placements () =
  let _, _, mark = setup () in
  let loss, small_comp = small () in
  let caller = make_ctx () in
  let fired = ref false in
  let reports = ref [] in
  let result =
    Autotune.Seam.with_value Autotune.on_candidate_preflight
      (fun _ ->
        if not !fired then (
          fired := true;
          mark ();
          raise
            (Ir.Schedule_outcome.Wait_bound_expired
               {
                 backend = Context.backend_name caller;
                 bound_s = 0.;
                 depth = 1;
                 reference_ms = None;
               })))
      (fun () ->
        match
          Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
            ~placement_store:false ~inline_flips:0 ~log:false ~progress:false
            ~report:(fun r -> reports := r :: !reports)
            caller loss small_comp Ir.Indexing.Empty
        with
        | _ -> None
        | exception exn -> Some exn)
  in
  p "the placement arm's injected failure fired after a real expiry" !fired;
  p "the placement search propagates the arm failure naming its bound"
    (Option.exists result ~f:(fun exn ->
         String.is_substring (Exn.to_string exn) ~substring:"host-side bound"));
  p_exists "the arm report retains the failed search" !reports ~f:(fun r ->
      match r.Autotune.outcome with
      | Autotune.Search_died f | Autotune.Pre_search_failure f ->
          String.is_substring f.detail ~substring:"host-side bound"
      | _ -> false);
  p "the caller's later release propagates Device_wedged"
    (refused (fun () -> Context.release caller))

(* Only multidev_cc: two real worker domains exercise the cross-device dependency path without
   requiring two physical GPUs. Each source expiry is finite work, never a hanging kernel. *)
let cross_device () =
  let _, _, mark = setup ~make_context:(fun () -> Context.cpu ~threads:2 ()) () in
  let module Backend = B.Multidev_cc_b in
  let src_root = Backend.make_context (Backend.get_device ~ordinal:0) in
  let dst_root = Backend.make_context (Backend.get_device ~ordinal:1) in
  let tensor =
    Operation.init ~l:"wedge_cross_device" ~prec:Ir.Ops.single ~o:[ 4 ]
      ~f:(fun i -> Float.of_int (i.(0) + 1))
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let tn = tensor.Tensor.value in
  let host value =
    Ir.Ndarray.init_array ~debug:"cross-device wedge" Ir.Ops.single ~dims:[| 4 |] ~padding:None
      ~f:value
  in
  let values = host (fun i -> Float.of_int (i.(0) + 1)) in
  let zeros = host (fun _ -> 0.) in
  let src = Backend.init_from_host src_root tn values in
  let dst = Backend.init_from_host dst_root tn zeros in
  let copy () =
    let routine = Option.value_exn (Backend.device_to_device tn ~into_merge_buffer:No ~dst ~src) in
    Ir.Task.run routine.schedule
  in
  copy ();
  Backend.await dst.device;
  let downloaded = host (fun _ -> -1.) in
  p "the healthy cross-device copy is available for readback" (Backend.to_host dst tn downloaded);
  Backend.await dst.device;
  p_all "the healthy cross-device copy computes the source values" [ 0; 1; 2; 3 ] ~f:(fun i ->
      Float.(Ir.Ndarray.get_as_float downloaded [| i |] = Float.of_int (Int.succ i)));
  let pending = ref (Backend.all_work src.device) in
  mark
    ~before_wait:(fun () ->
      let event = Backend.all_work src.device in
      pending := event;
      Hashtbl.set src.device.updating_for ~key:tn ~data:event)
    ();
  p "the marked source has a pending writer event" (not (Backend.is_done !pending));
  let before = AC.snapshot () in
  p "a cross-device copy of the marked source raises Device_wedged" (refused copy);
  p "the copy refusal frees no pools" ((AC.snapshot ()).pools_freed = before.pools_freed);
  p "a queued wait on the marked source event raises Device_wedged"
    (refused (fun () -> Backend.will_wait_for dst !pending));
  p "the destination still completes its bounded wait"
    (Backend.await_within dst.device ~seconds:60.);
  let healthy_event = Backend.all_work dst.device in
  Backend.sync healthy_event;
  p "an event from the other device remains usable" (Backend.is_done healthy_event);
  p "a marked destination refuses even a healthy source event"
    (refused (fun () -> Backend.will_wait_for src healthy_event))

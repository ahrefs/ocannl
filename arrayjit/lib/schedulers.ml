open Base
module Tn = Ir.Tnode
open Ir.Backend_intf
open Ir.Backend_impl

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_SCHEDULERS=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_SCHEDULERS"]

(* The CPU "tile-MMA capability" (gh-ocannl-469): no tensor cores, but the C backends render
   [Tile_mma] as the register-tiled vector micro-kernel ([C_syntax]'s [try_register_tile]), so the
   autotune menu may propose [Tensorize] over CPU serial triples. The register renderer peels edge
   tiles, so any block extents are legal (intrinsic tile 1×1×1); the lane loop renders serially, so
   a single lane suffices. [simd_vector_bytes] reports the register-tiling's vector width
   (gh-ocannl-479: autotune's seeding pre-filter checks micro-kernel extents against the lane
   count); the only CPU backend is cc, so its setting is authoritative here. *)
let cpu_mma_limits () =
  {
    no_hardware_limits with
    simd_vector_bytes = Cc_backend.vector_bytes_setting ();
    native_fp16_arithmetic = Cc_backend.has_native_fp16_arithmetic ();
    worker_pool_tag = Some (Cc_backend.pool_tag ());
    codegen_tag = Some (Cc_backend.codegen_tag ());
    (* [mma_format_tiles] is empty: the register tiling is not a tensor-core instruction — no
       per-format intrinsic tiles exist, and its precision gates (f32/f64) live in the seeding. *)
    mma =
      Some
        {
          mma_simd_width = 1;
          mma_tile = (1, 1, 1);
          mma_format_tiles = [];
          (* No format tiles, so the wide-f16/bf16 seeding gates never consult these; the CPU
             register tiling follows [cpu_compute_prec]/[accum_prec] directly (gh-ocannl-680). *)
          mma_f16_wide_acc_scopes = [];
          mma_bf16_wide_acc_scopes = [];
          mma_staged_layouts = [];
          mma_pipeline_depths = [];
        };
  }

module Multidev (Backend : For_add_scheduler) :
  With_scheduler with type buffer_ptr = Backend.buffer_ptr = struct
  include Backend
  module Domain = Domain [@warning "-3"]
  module Mut = Stdlib.Mutex

  module Device_config = struct
    type dev = CPU [@@deriving sexp_of]

    (* One worker domain with a plain mutex-protected FIFO queue per device. [multidev_cc] exists
       for debugging multi-device parallel workflows on any development machine, so the queue favors
       simplicity over throughput (the retired [Multicore] scheduler used a lock-free SPSC queue
       with idle/ready handshakes; kernel-level CPU parallelism does not come from the scheduler
       anymore, it comes from pool-rendered Grid loops in the cc codegen). [progress] is broadcast
       by the worker after every completed task and when it idles; [await] and event [sync] both
       wait on it. *)
    type device_state = {
      mutable keep_spinning : bool;
      mutable dev_error : exn option;
      queue : (Ir.Task.t Stdlib.Queue.t[@sexp.opaque]);
      mut : (Mut.t[@sexp.opaque]);
      work_available : (Stdlib.Condition.t[@sexp.opaque]);
      progress : (Stdlib.Condition.t[@sexp.opaque]);
      mutable busy : bool;
      ordinal : int;
    }
    [@@deriving sexp_of]

    type domain = unit Domain.t

    let sexp_of_domain (d : domain) = Sexp.Atom ("domain-" ^ Int.to_string (Domain.get_id d :> int))

    type runner = { state : device_state; domain : domain } [@@deriving sexp_of]
    type event = { dev_state : device_state; is_done : Utils.atomic_bool } [@@deriving sexp_of]

    let name = "multidev_" ^ Backend.name
  end

  module Device_types = Device_types_ll (Device_config)
  include Device (Device_types) (Make_slab (Device_types) (Backend))
  open Device_config

  let sync { dev_state; is_done } =
    Mut.lock dev_state.mut;
    while dev_state.keep_spinning && not (Atomic.get is_done) do
      Stdlib.Condition.wait dev_state.progress dev_state.mut
    done;
    (* A stopped worker is not completion: if the device died before the event's tick ran, surface
       its error to the waiter -- a cross-device consumer would otherwise proceed with stale or
       uninitialized source data. When this runs on a waiting device's worker (via [will_wait_for]),
       the re-raise fails that worker's task, propagating the failure to the waiting device. *)
    let err = if Atomic.get is_done then None else dev_state.dev_error in
    Mut.unlock dev_state.mut;
    Option.iter err ~f:(fun e ->
        Exn.reraise e [%string "Multidev_scheduler.sync: device %{dev_state.ordinal#Int} failed"])

  let is_done { dev_state = _; is_done } = Atomic.get is_done
  let get_used_memory _device = get_used_memory ()

  let is_idle device =
    let st = device.runner.state in
    (not st.busy) && Stdlib.Queue.is_empty st.queue

  let await device =
    assert (Domain.is_main_domain ());
    let st = device.runner.state in
    Mut.lock st.mut;
    while st.keep_spinning && (st.busy || not (Stdlib.Queue.is_empty st.queue)) do
      Stdlib.Condition.wait st.progress st.mut
    done;
    Mut.unlock st.mut;
    Option.iter st.dev_error ~f:(fun e -> Exn.reraise e @@ get_name device)

  let schedule_task device task =
    assert (Domain.is_main_domain ());
    let st = device.runner.state in
    Option.iter st.dev_error ~f:(fun e -> Exn.reraise e @@ get_name device);
    if not st.keep_spinning then invalid_arg "Multidev_scheduler: device not available";
    Mut.lock st.mut;
    Stdlib.Queue.add task st.queue;
    Stdlib.Condition.broadcast st.work_available;
    Mut.unlock st.mut

  let will_wait_for ctx ({ dev_state; is_done = _ } as event) =
    let work () = sync event in
    let task =
      Ir.Task.Task
        {
          context_lifetime = ();
          description =
            [%string "wait on %{get_name ctx.device} for device %{dev_state.ordinal#Int}"];
          work;
        }
    in
    schedule_task ctx.device task

  let all_work device =
    assert (Domain.is_main_domain ());
    let state = device.runner.state in
    let is_done = Atomic.make false in
    let work () = assert (Atomic.compare_and_set is_done false true) in
    (* The worker broadcasts [progress] right after this task completes, waking any [sync] waiting
       on the returned event. *)
    schedule_task device
    @@ Ir.Task.Task { context_lifetime = (); description = "all_work tick"; work };
    { dev_state = state; is_done }

  let spinup_runner ~ordinal : runner =
    let state =
      {
        keep_spinning = true;
        dev_error = None;
        queue = Stdlib.Queue.create ();
        mut = Mut.create ();
        work_available = Stdlib.Condition.create ();
        progress = Stdlib.Condition.create ();
        busy = false;
        ordinal;
      }
    in
    let worker (() : unit) : unit =
      assert (not @@ Domain.is_main_domain ());
      Mut.lock state.mut;
      let rec loop () =
        if state.keep_spinning then
          match Stdlib.Queue.take_opt state.queue with
          | Some task -> (
              state.busy <- true;
              Mut.unlock state.mut;
              let outcome = Result.try_with (fun () -> Ir.Task.run task) in
              Mut.lock state.mut;
              state.busy <- false;
              Stdlib.Condition.broadcast state.progress;
              match outcome with
              | Ok () -> loop ()
              | Error e ->
                  (* All exceptions are fatal to the device; [await] and [schedule_task] re-raise
                     the stored error on the main domain. *)
                  state.dev_error <- Some e;
                  state.keep_spinning <- false)
          | None ->
              Stdlib.Condition.broadcast state.progress;
              Stdlib.Condition.wait state.work_available state.mut;
              loop ()
      in
      loop ();
      Mut.unlock state.mut
    in
    { state; domain = Domain.spawn worker }

  let%track7_sexp cleanup_device (device : device) : unit =
    [%log "cleanup_device: await device"];
    await device;
    let r : runner = device.runner in
    Mut.lock r.state.mut;
    r.state.keep_spinning <- false;
    [%log "cleanup_device: broadcasting work_available to wake up the worker"];
    Stdlib.Condition.broadcast r.state.work_available;
    Mut.unlock r.state.mut;
    [%log "cleanup_device: joining the domain"];
    Domain.join r.domain

  (* The device count is fixed at first use: [multidev_num_devices] if configured positive,
     otherwise the effective pool width. Worker domains spin up lazily, one per device ordinal, on
     first [get_device] -- runs that never use multidev_cc pay nothing.

     Forcing the pool policy here, before any [Domain.spawn], is load-bearing on Linux:
     [sched_setaffinity(0)] restricts the calling thread and threads spawned afterwards, so a worker
     domain created before the restriction would keep the full-machine mask and its OpenMP teams
     would run on the mixed pool the policy exists to avoid. It also makes the device count
     affinity-respecting (gh-ocannl-530: [Domain.recommended_domain_count] is affinity-blind on
     Windows), so a restricted or pinned run does not oversubscribe its CPU subset with worker
     domains. *)
  let num_devices =
    let n =
      lazy
        (let policy_width = Cc_backend.effective_pool_width () in
         let configured =
           Int.of_string @@ Utils.get_global_arg ~default:"0" ~arg_name:"multidev_num_devices"
         in
         if configured > 0 then configured else policy_width)
    in
    fun () -> Lazy.force n

  let devices = lazy (Array.create ~len:(num_devices ()) (None : device option))

  let get_device ~ordinal =
    let devices = Lazy.force devices in
    let n = Array.length devices in
    if ordinal < 0 || ordinal >= n then
      invalid_arg
        [%string
          "Multidev_scheduler.get_device %{ordinal#Int}: only %{n#Int} devices (see the \
           multidev_num_devices setting)"];
    match devices.(ordinal) with
    | Some device -> device
    | None ->
        assert (Domain.is_main_domain ());
        let device = make_device CPU (spinup_runner ~ordinal) ~ordinal in
        Stdlib.Gc.finalise cleanup_device device;
        devices.(ordinal) <- Some device;
        device

  (* One entry per worker-domain device (gh-ocannl-710). This dump used to be the group atom
     followed by [(device_name CPU) (num_devices 16)] -- backend-level pairs, no device entries at
     all -- so a generic reader indexing the children reported two devices, neither of which
     existed, on the one backend whose entire purpose is multi-device debugging. The count is now
     the number of entries, which is the reading a count cannot give: an indexable device.

     Enumeration is static: it reports the ordinals this backend will serve, without forcing the
     lazy worker domains ([get_device] spins one up on first use, from the main domain only). *)
  let static_properties () =
    Sexp.List
      (Sexp.Atom (name ^ "_devices")
      :: List.init (num_devices ()) ~f:(fun ordinal ->
          Sexp.message "device"
            [
              ("device_name", Sexp.Atom "CPU");
              ("device_ordinal", [%sexp_of: int] ordinal);
              (* The device's own scheduling threads: one worker domain per ordinal. Not the
                 kernel-level width -- Grid loops render onto the process-global worker pool, which
                 is a backend fact and not a per-device one. *)
              ("threads", [%sexp_of: int] 1);
            ]))

  let timing_identity _ = Cc_backend.timing_identity ()
  let hardware_limits () = cpu_mma_limits ()
  let get_global_debug_info () = Sexp.message "global_debug" []
  let get_debug_info (device : device) = sexp_of_runner device.runner
end

(** A minimalistic wrapper creating backends where all calls run synchronously on the main thread.
    There is a single CPU device. *)
module Sync (Backend : For_add_scheduler) = struct
  include Backend

  module Device_config = struct
    type dev = CPU [@@deriving sexp_of]
    type runner = unit [@@deriving sexp_of]
    type event = unit [@@deriving sexp_of]

    let name = Backend.name
  end

  module Device_types = Device_types_ll (Device_config)
  include Device (Device_types) (Make_slab (Device_types) (Backend))
  open Device_config

  let sync () = ()
  let is_done () = true
  let will_wait_for _context () = ()

  (* Lazy like [Multidev]'s devices and for the same reason as the GPU backends' device discovery
     (PR #94): the singleton backend module initializes at program startup, and minting the device
     there would advance the process-global [device_id] counter in runs that never use cc --
     shifting other backends' stream names and log files (e.g. [multidev_cc-0-0.log]). *)
  let device : device Lazy.t = lazy (make_device CPU () ~ordinal:0)

  let get_device ~ordinal =
    if ordinal <> 0 then
      invalid_arg @@ "Sync_scheduler.get_device: there is only one device, but ordinal="
      ^ Int.to_string ordinal;
    Lazy.force device

  let num_devices () = 1
  let get_used_memory _ = Backend.get_used_memory ()
  let all_work _device = ()
  let is_idle _device = true
  let await _device = ()

  (* The single CPU device, [Sexp.message]-shaped like every other backend's entry (gh-ocannl-710):
     the pairs used to sit one nesting level deeper, wrapped in a list of their own. See
     [Backend_intf.parse_static_properties] for the contract. *)
  let static_properties () =
    Sexp.List
      [
        Sexp.Atom (name ^ "_devices");
        Sexp.message "device"
          [
            ("device_name", Sexp.Atom "CPU");
            ("device_ordinal", [%sexp_of: int] 0);
            (* Tasks run synchronously on the calling thread; see the [Multidev] note on why this is
               not the kernel-level width. *)
            ("threads", [%sexp_of: int] 1);
          ];
      ]

  let timing_identity _ = Cc_backend.timing_identity ()
  let hardware_limits () = cpu_mma_limits ()

  (* let global_run_no = ref 0 *)
  let schedule_task _device task = Ir.Task.run task
  let get_global_debug_info () = Sexp.message "global_debug" []
  let get_debug_info (device : device) = sexp_of_runner device.runner
end

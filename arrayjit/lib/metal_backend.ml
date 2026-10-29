open Base
open Ir
module Tn = Tnode
module Lazy = Utils.Lazy
module Me = Metal (* Alias for Metal module *)
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_METAL_BACKEND=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_METAL_BACKEND"]

type ullong = Unsigned.ULLong.t

let sexp_of_ullong x = Sexp.Atom (Unsigned.ULLong.to_string x)

(* gh-ocannl-344: number of pool base-pointer parameters every pooled kernel declares. A routine may
   reach at most this many distinct pools (working delta + ancestor deltas + constant pools +
   merge); together with the slot table, static-index and merge params this stays well under Metal's
   ~31 argument-buffer binding limit, independently of the tensor-node count. A routine needing more
   distinct pools raises a clear error at link time. *)
let metal_max_pools = 16

module Backend_buffer = struct
  type buffer_ptr = Me.Buffer.t (* Use the payload type from metal.ml *)

  (* Provide a sexp_of for Me.Buffer.t *)
  let sexp_of_buffer_ptr = Me.Buffer.sexp_of_t
end

module Device_config = struct
  include Backend_buffer

  (* Represents a physical Metal device *)
  type dev = Me.Device.t

  let sexp_of_dev = Me.Device.sexp_of_t

  (* Represents a command queue + event for synchronization *)
  type runner = {
    queue : Me.CommandQueue.t;
    event : Me.SharedEvent.t; (* Use SharedEvent for signalling *)
    mutable counter : ullong; (* Next value to signal *)
    mutable fused : (Me.CommandBuffer.t * Me.ComputeCommandEncoder.t) option; [@sexp.opaque]
        (* While [Some], kernel launches encode into this open serial compute pass instead of
           committing their own command buffer — set for the span of one fissioned routine's segment
           batch by [sequence_segments]. Only the task running on this queue touches it. *)
  }
  [@@deriving sexp_of]

  (* Represents a point in time on a specific runner *)
  type event = { shared : Me.SharedEvent.t; value : ullong } [@@deriving sexp_of]

  let name = "metal"
end

module Device_stream = Backend_impl.Device_types_ll (Device_config)

(* Bring types into scope *)
open Device_config

(* Manual tracking for allocated memory *)
let allocated_memory = Atomic.make 0

let track_allocation (buffer : Me.Buffer.t) =
  (* Relying on ARC + finalizer on the payload record *)
  let size = Me.Resource.get_allocated_size (Me.Buffer.super buffer) in
  Stdlib.Gc.finalise (fun _ -> ignore (Atomic.fetch_and_add allocated_memory (-size))) buffer;
  ignore (Atomic.fetch_and_add allocated_memory size)

(* gh-ocannl-344: every pool uses Shared storage (unified memory, WriteCombined for CPU writes), so
   host transfers are a direct memcpy with no staging blit, and a context delta never splits across
   storage modes. Hazards are Untracked: at pool granularity Metal's automatic tracking would
   serialize kernels touching *disjoint* tnodes in the same slab (false dependencies); OCANNL's own
   writer-event machinery ([updating_for]) already expresses the true cross-kernel ordering. The
   per-tnode private/shared selection (gh-ocannl-320) is retired -- there is no replacement
   config. *)
let shared_resource_options =
  Me.ResourceOptions.(
    storage_mode_shared + cpu_cache_mode_write_combined + hazard_tracking_mode_untracked)

(* The Metal slab allocator: a private [(device_id, pool_id) -> Metal.Buffer.t] table backing the
   shared {!Backend_intf.Slab_alloc}. All pools are Shared/Untracked; [?mode] is ignored. *)
module Slab = struct
  open Backend_intf

  type device = Device_stream.device
  type buffer_ptr = Me.Buffer.t

  let pools : (int * int, buffer_ptr) Hashtbl.Poly.t = Hashtbl.Poly.create ()

  let alloc_pool ?mode:_ (device : device) ~pool_id ~size_in_bytes ~alignment:_ =
    let buffer =
      Me.Buffer.on_device device.dev ~length:(max 1 size_in_bytes) shared_resource_options
    in
    track_allocation buffer;
    Hashtbl.set pools ~key:(device.device_id, pool_id) ~data:buffer

  (* Rely on ARC and the finalizer attached in track_allocation for the actual reclamation, but
     still drop the private table entry on finalization so the strong reference is released
     (otherwise ARC can never reclaim a context's tnode buffers). Re-allocating a pool also
     overwrites the entry. *)
  let free_pool =
    Some (fun (device : device) ~pool_id -> Hashtbl.remove pools (device.device_id, pool_id))

  let memset_zero (device : device) ~pool_id ~offset ~size_in_bytes =
    let buffer = Hashtbl.find_exn pools (device.device_id, pool_id) in
    let command_buffer = Me.CommandBuffer.on_queue device.runner.queue in
    let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
    Me.BlitCommandEncoder.fill_buffer blit_encoder buffer
      { location = offset; length = size_in_bytes }
      ~value:0;
    Me.BlitCommandEncoder.end_encoding blit_encoder;
    Me.CommandBuffer.commit command_buffer;
    Me.CommandBuffer.wait_until_completed command_buffer

  let resolve_pool (device : device) { pool_id; offset = _ } : buffer_ptr =
    (* Resolve the slab [Me.Buffer.t] for [pool_id]. Metal buffers are objects, not raw pointers, so
       the byte [offset] is NOT folded into the handle here; callers apply it via blit
       source/destination offsets, or fold it into a GPU address for the kernel pointer table. *)
    Hashtbl.find_exn pools (device.device_id, pool_id)

  let storage_mode_of_pool (device : device) ~pool_id =
    Me.Resource.get_storage_mode
      (Me.Buffer.super (Hashtbl.find_exn pools (device.device_id, pool_id)))
end

(* Module defining the backend. The exact public signature (Lowered_backend + storage_mode_of_pool)
   is sealed by metal_backend.mli. *)
module Impl = struct
  (* Include the device setup with types and allocation *)
  include Backend_impl.Device (Device_stream) (Slab)

  (* The concrete [buffer_ptr]/[buffer] + sexps for the impl-facing interface (no longer carried by
     the shared [Device_config_common]). *)
  include Backend_buffer

  let storage_mode_of_pool = Slab.storage_mode_of_pool

  (* Global state for Metal devices. Device discovery is lazy: the singleton [Impl] module
     initializes at program startup (Backends instantiates it eagerly for nameable types), and
     enumerating devices there would make runs that never use Metal depend on the Metal runtime
     (e.g. headless machines). Forced at first device use, where [Context.auto] can catch a failure
     per call. *)
  (* Deliberately NOT asserting the array is nonempty: on a Metal-linked machine with no Metal
     device the assertion fired here, before [get_device] could report [Backend_unavailable], and an
     [Assert_failure] is never a reason to try the next backend — so an unconfigured [Context.auto]
     would die instead of falling through to cc (Codex P1 on PR #256). The emptiness is handled
     where it means something; every indexing use is guarded by an ordinal check. *)
  let metal_devices : Me.Device.t array Lazy.t = lazy (Me.Device.copy_all_devices ())

  (* Captured routine logs per device_id (the device is its own single compute stream). Metal calls
     log handlers on its own dispatch threads, so the handler is ocaml-metal's native one, which
     buffers messages C-side and never runs OCaml there (gh-ocannl-1594); [await] drains it on the
     host thread. *)
  let stream_logs : (int, Me.LogState.log_buffer) Hashtbl.t = Hashtbl.create (module Int)

  (* Device Management *)
  let num_devs () = Array.length (Lazy.force metal_devices)
  let devices_cache = lazy (Array.create ~len:(num_devs ()) None)

  (* Builds the device's single compute runner (command queue + sync event), optionally with a
     debug-log-capturing queue. Returns the runner and the captured-log buffer (if logging is
     on). *)
  let spinup_runner metal_device =
    let queue =
      if Utils.debug_log_from_routines () then (
        let log_desc = Me.LogStateDescriptor.create () in
        Me.LogStateDescriptor.set_level log_desc Me.LogLevel.Debug;
        (* Capture all debug logs and above *)
        Me.LogStateDescriptor.set_buffer_size log_desc (1024 * 100);
        (* 100KB buffer *)
        let log_state = Me.LogState.on_device_with_descriptor metal_device log_desc in
        let log_buffer = Me.LogState.add_buffered_log_handler log_state in
        let queue_desc = Me.CommandQueueDescriptor.create () in
        Me.CommandQueueDescriptor.set_log_state queue_desc (Some log_state);
        (* The queue's OCaml value holds the log state, and with it the buffer, through the
           descriptor's lifetime chain; [stream_logs] holds the buffer too. *)
        let created_q = Me.CommandQueue.on_device_with_descriptor metal_device queue_desc in
        (created_q, Some log_buffer))
      else (Me.CommandQueue.on_device metal_device, None)
    in
    let actual_queue, opt_log_buffer = queue in
    let shared_event_obj = Me.SharedEvent.on_device metal_device in
    let counter = Unsigned.ULLong.one in
    (* Next value = 1 *)
    ({ queue = actual_queue; event = shared_event_obj; counter; fused = None }, opt_log_buffer)

  let get_device ~(ordinal : int) : device =
    (* See the note in [Cuda_backend.get_device]: no Metal device at all is the discovery failure
       [Context.auto] falls through on; a bad ordinal is a caller error. *)
    if num_devs () = 0 then
      raise
      @@ Backend_intf.Backend_unavailable
           { backend = name; detail = "no Metal device on this system" };
    if ordinal < 0 || num_devs () <= ordinal then
      invalid_arg [%string "Metal_backend.get_device %{ordinal#Int}: invalid ordinal"];
    let devices_cache = Lazy.force devices_cache in
    let default () =
      let metal_device = (Lazy.force metal_devices).(ordinal) in
      let runner, opt_log_buffer = spinup_runner metal_device in
      let result_device = make_device metal_device runner ~ordinal in
      (* The device is its own single compute stream; key captured logs by [device_id]. *)
      Option.iter opt_log_buffer ~f:(fun log_buffer ->
          Hashtbl.add_exn stream_logs ~key:result_device.device_id ~data:log_buffer);
      Stdlib.Gc.finalise (fun d -> Hashtbl.remove stream_logs d.device_id) result_device;
      devices_cache.(ordinal) <- Some result_device;
      result_device
    in
    Option.value_or_thunk devices_cache.(ordinal) ~default

  let num_devices () =
    (* FIXME: refactor the whole backend interface to use constant num_devices per backend
       instance *)
    num_devs ()

  (* --- Event Handling --- *)
  let is_done event =
    let current_value = Me.SharedEvent.get_signaled_value event.shared in
    Unsigned.ULLong.compare current_value event.value >= 0

  let sync event =
    if not (is_done event) then
      let timeout_max = Unsigned.ULLong.max_int in
      ignore
        (Me.SharedEvent.wait_until_signaled_value event.shared ~value:event.value
           ~timeout_ms:timeout_max)

  let will_wait_for context event =
    let device = context.device in
    let queue = device.runner.queue in
    let command_buffer = Me.CommandBuffer.on_queue queue in
    Me.CommandBuffer.encode_wait_for_event command_buffer
      (Me.SharedEvent.super event.shared)
      event.value;
    Me.CommandBuffer.commit command_buffer

  (* Commits a command buffer signalling the device's event after all current work; returns the
     event and that command buffer. *)
  let signal_after_all_work device =
    let queue = device.runner.queue in
    let shared_event = device.runner.event in
    let counter = device.runner.counter in
    let next_value = Unsigned.ULLong.add counter Unsigned.ULLong.one in
    device.runner.counter <- next_value;
    let command_buffer = Me.CommandBuffer.on_queue queue in
    Me.CommandBuffer.encode_signal_event command_buffer
      (Me.SharedEvent.super shared_event)
      next_value;
    Me.CommandBuffer.commit command_buffer;
    ({ shared = shared_event; value = next_value }, command_buffer)

  let all_work device = fst (signal_after_all_work device)

  let await device =
    (* Signal an event after all current work and wait for it. This ensures all previously submitted
       command buffers complete. *)
    let event, command_buffer = signal_after_all_work device in
    sync event;
    (* Drain the captured routine logs, if logging is on for this device. The event firing is a GPU
       timeline fact, while Metal hands log messages to the handler on the CPU side, so under
       logging [await] also waits for the signalling command buffer (the queue's last) to complete
       before draining: no Metal document promises delivery by either point, and completion is the
       later of the two. test/operations/metal_routine_log_delivery pins that every message of the
       work is drained by then (it held 10 of 10 runs even without this wait). A message that does
       arrive later is not lost: it is in the next [await]'s drain. *)
    if Utils.debug_log_from_routines () then
      match Hashtbl.find stream_logs device.device_id with
      | Some log_buffer ->
          Me.CommandBuffer.wait_until_completed command_buffer;
          (* The drain detaches the batch: if writing it out raises, the batch is lost, never
             written twice. *)
          let entries, dropped = Me.LogState.drain_log_buffer log_buffer in
          let messages = List.map entries ~f:(fun (e : Me.LogState.log_entry) -> e.message) in
          let messages =
            if dropped > 0 then
              messages
              @ [
                  [%string
                    "[ocannl] %{dropped#Int} Metal routine log messages were dropped (allocation \
                     failure)"];
                ]
            else messages
          in
          if not (List.is_empty messages) then
            Utils.log_debug_routine_logs ~log_contents:messages ~stream_name:(get_name device)
      | None -> () (* No log buffer for this device, logging likely not enabled for it *)

  let is_idle device =
    (* FIXME: store the latest CommandBuffer with the device runner and check that it's completed *)
    let counter = device.runner.counter in
    let current_signaled = Me.SharedEvent.get_signaled_value device.runner.event in
    let expected_signaled = Unsigned.ULLong.pred counter in
    Unsigned.ULLong.equal current_signaled expected_signaled

  (* --- Configuration and Info --- *)
  let get_used_memory _device = Atomic.get allocated_memory
  let get_free_and_total_memory _device = None

  (* [Sexp.message]-shaped device entries, like every other backend's (gh-ocannl-710): the pairs
     used to sit one nesting level deeper, wrapped in a list of their own, which is the same
     information at a path no other backend's reader knows. See
     [Backend_intf.parse_static_properties] for the contract. *)
  let static_properties () =
    let device_properties =
      Array.mapi (Lazy.force metal_devices) ~f:(fun ordinal device ->
          let attributes = Me.Device.get_attributes device in
          Sexp.message "device"
            [
              ("device_name", Sexp.Atom attributes.name);
              ("device_ordinal", [%sexp_of: int] ordinal);
              ("registry_id", Sexp.Atom (Unsigned.ULLong.to_string attributes.registry_id));
              ( "max_buffer_length",
                Sexp.Atom (Unsigned.ULong.to_string attributes.max_buffer_length) );
              ( "max_threadgroup_memory_length",
                Sexp.Atom (Unsigned.ULong.to_string attributes.max_threadgroup_memory_length) );
              (* The launch-dimension limit the schedule layer gates against: all three components,
                 not just [width]. [width] alone feeds [hardware_limits.max_threads_per_workgroup]
                 (the thread product), while the triple feeds [max_workgroup_dims] (per-dimension,
                 gh-ocannl-679). Surfaced so a run on hardware can read back what the gates compare
                 against -- [bin/device_props.ml] prints exactly this (gh-ocannl-684). *)
              ( "max_threads_per_threadgroup",
                [%sexp_of: int * int * int]
                  ( attributes.max_threads_per_threadgroup.width,
                    attributes.max_threads_per_threadgroup.height,
                    attributes.max_threads_per_threadgroup.depth ) );
              ( "recommended_max_working_set_size",
                Sexp.Atom (Unsigned.ULLong.to_string attributes.recommended_max_working_set_size) );
              ("is_low_power", [%sexp_of: bool] attributes.is_low_power);
              ("is_headless", [%sexp_of: bool] attributes.is_headless);
              ("has_unified_memory", [%sexp_of: bool] attributes.has_unified_memory);
              (* Backend-global, not per-device: the allocation counter this backend maintains
                 across its devices. Kept on the entry (rather than promoted to a backend-level
                 child, which the dump's shape does not have) because a single-GPU Mac is the only
                 configuration where the two readings differ at all. *)
              ("total_memory", [%sexp_of: int] (Atomic.get allocated_memory));
              ( "supported_gpu_families",
                Sexp.List
                  (List.map attributes.supported_gpu_families ~f:(fun gpu_family ->
                       Me.Device.GPUFamily.sexp_of_t gpu_family)) );
            ])
    in
    Sexp.List (Sexp.Atom "metal_devices" :: Array.to_list device_properties)

  (* Conservative per-workgroup device limits for the schedule layer (schedule-ir-optops §6):
     minimum across devices. [max_threads_per_threadgroup.width] is the device-level 1-D bound;
     a pipeline's [maxTotalThreadsPerThreadgroup] (checked at PSO creation in [compile_proc]) can
     restrict it further under register pressure. *)
  (* Memoized behind [lazy] like [metal_devices] itself: device enumeration must not run at
     backend-module initialization (see the [metal_devices] comment). *)
  (* The OS build identifies Metal's bundled compiler/driver. A host UUID disambiguates the
     device's registry ID across machines sharing the same cache directory. *)
  let timing_identity (device : device) =
    Option.map (Lazy.force Utils.macos_timing_environment) ~f:(fun (host, toolchain_signature) ->
        let attributes = Me.Device.get_attributes device.dev in
        {
          Backend_intf.device_signature =
            Sexp.to_string
              (Sexp.List
                 [
                   Sexp.Atom host;
                   Sexp.Atom attributes.name;
                   Sexp.Atom (Unsigned.ULLong.to_string attributes.registry_id);
                   Sexp.List
                     (List.map attributes.supported_gpu_families ~f:Me.Device.GPUFamily.sexp_of_t);
                   [%sexp_of: bool] attributes.has_unified_memory;
                 ]);
          toolchain_signature;
        })

  (* gh-ocannl-1638: the IORegistry's "gpu-core-count" of the GPU with this registry ID
     ([metal_iokit_stubs.c]); 0 where it has none. *)
  external gpu_core_count : int64 -> int = "ocannl_metal_gpu_core_count"

  let hardware_limits =
    let limits =
      lazy
        (let min_over f =
           Array.map (Lazy.force metal_devices) ~f:(fun d -> f (Me.Device.get_attributes d))
           |> Array.min_elt ~compare:Int.compare
         in
         (* The Apple-silicon class (Apple7+: M1 and later). [for_all] is vacuously true on no
            devices, which would advertise this class on a machine that has no Metal device at all.
            Unreachable in practice (compiling needs a device), but it used to be impossible by the
            discovery assertion, so it is stated rather than assumed. *)
         let apple_silicon =
           (not (Array.is_empty (Lazy.force metal_devices)))
           && Array.for_all (Lazy.force metal_devices) ~f:(fun d ->
               Me.Device.supports_family d Me.Device.GPUFamily.Apple7)
         in
         {
           Backend_intf.max_threads_per_workgroup =
             min_over (fun (a : Me.Device.attributes) -> a.max_threads_per_threadgroup.width);
           max_workgroup_memory_bytes =
             min_over (fun (a : Me.Device.attributes) ->
                 Unsigned.ULong.to_int a.max_threadgroup_memory_length);
           max_thread_private_bytes = None;
           (* Per-dimension threadgroup caps (gh-ocannl-679): [maxThreadsPerThreadgroup] is a 3-D
              [MTLSize] and only its [width] was being read (as the thread-product cap above). Apple
              parts report the three components equal, so this row does not fire alone here; it is
              filled because a schedule is gated by the record, not by the backend's name, and a
              per-dimension cap left [None] silently exempts the dimension. *)
           max_workgroup_dims =
             (match
                ( min_over (fun (a : Me.Device.attributes) -> a.max_threads_per_threadgroup.width),
                  min_over (fun (a : Me.Device.attributes) -> a.max_threads_per_threadgroup.height),
                  min_over (fun (a : Me.Device.attributes) -> a.max_threads_per_threadgroup.depth)
                )
              with
             | Some x, Some y, Some z -> Some (x, y, z)
             | _ -> None);
           (* Metal's threadgroups-per-grid dimensions are not 16-bit like CUDA/HIP's gridDim.y/z;
              no practical cap to enforce here. *)
           max_bound_pools = Some metal_max_pools;
           max_grid_yz = None;
           (* [simdgroup_matrix], available on Apple7+ (M1 and later): the device-free descriptor
              [Metal_mma.capability], whose format triples correspond to the arm table the hooks
              consult ([Metal_mma.mma_accumulator], gh-ocannl-1246). *)
           mma = (if apple_silicon then Some Metal_mma.capability else None);
           simd_vector = Ir.Vector_file.none;
           native_fp16_arithmetic = false;
           simdgroup_width = Some 32;
           (* gh-ocannl-1124: measured true on the M4 Max; see the field. *)
           lane_scalar_recompute_cheap = true;
           (* gh-ocannl-1168: measured on the M4 Max -- heads of 64 and 128 on the cross-simdgroup
              all-reduce train 3-18% and 31% faster; see the field. *)
           lane_all_reduce_simdgroups = 4;
           online_softmax_auto_block = 16;
           worker_pool_tag = None;
           (* gh-ocannl-572: no codegen or dispatch knob of this backend is configurable — the
              kernel source is a function of the lowered code, the device capabilities and the
              numerics policy, all of which the cache key already covers. *)
           codegen_tag = None;
           (* Advisory roofline envelope (gh-ocannl-491): documented rough constants for the
              Apple-silicon class ([Device.attributes] exposes no throughput numbers). Flops:
              mid-range M-series ~5 fp32 TFLOP/s — the model only ranks, so a class-typical number
              suffices. Bandwidth is a class CEILING per the [hardware_limits] contract
              (gh-ocannl-578): Ultra-tier unified memory reaches ~819 GB/s, and the previous
              mid-range 2.0e11 was demonstrably understated (a 4e11 B/s streaming rate is routine on
              an M4 Max), tripping the gh-514 agreement warning on every exact streaming row. *)
           peak_flops = Some 5.0e12;
           peak_memory_bandwidth = Some 1.0e12;
           (* gh-ocannl-1253: no ranking-score capacity until a Metal measurement (gh-ocannl-1404):
              the score charges distinct cells here. *)
           cache_capacity_bytes = None;
           max_resident_threads = None;
           cache_bandwidth = None;
           (* gh-ocannl-1442: the ranking score's own rates. The issue rate is the full-fill rate
              tools/calibrate_issue_rate.exe's launch sweep fits jointly with fill_threads below
              (gh-ocannl-1638; the two are one calibration): 2.21e12 on the M4 Max (40 GPU cores;
              2026-10-25, 20d048c3, largest relative residual 0.150), for the Apple-silicon class.
              The DRAM rate is unmeasured here. *)
           issue_rate = (if apple_silicon then Some 2.21e12 else None);
           dram_bandwidth = None;
           (* gh-ocannl-1495: unmeasured; simdgroup_load loads whole fragments. *)
           mma_rate = None;
           device_gather_rate = None;
           (* gh-ocannl-1479, gh-ocannl-1638: the launch fill. Metal publishes no GPU core count, so
              it is read from the IORegistry (the AGX accelerator's "gpu-core-count", by the
              device's registry ID), minimum over devices; [None] when any device has none (a
              non-AGX GPU). The threads per core at which scalar issue saturates is fitted by the
              launch sweep on the M4 Max (40 cores, 6 points, 5 latency-starved points left out;
              2026-10-25, 20d048c3), whose 512 x 512 calibration kernel is full under it. Both
              overridable by model_multiprocessors / model_fill_threads. *)
           multiprocessors =
             (let counts =
                Array.map (Lazy.force metal_devices) ~f:(fun d ->
                    gpu_core_count
                      (Unsigned.ULLong.to_int64 (Me.Device.get_attributes d).registry_id))
              in
              if Array.for_all counts ~f:(fun c -> c > 0) then
                Array.min_elt counts ~compare:Int.compare
              else None);
           fill_threads = (if apple_silicon then Some 320 else None);
           (* gh-ocannl-1600: unmeasured; configured (model_transaction_rate), never advisory. *)
           transaction_rate = None;
           workgroup_bytes_per_multiprocessor = None;
         })
    in
    fun () -> Lazy.force limits

  let get_global_debug_info () = Sexp.Atom "Metal global debug info NYI"

  let get_debug_info device =
    (* The captured shader logs wait C-side until [await] drains them; counting them would drain
       them, so this reports only whether capture is on. *)
    Sexp.message "Metal device debug info"
      [
        ("device_id", sexp_of_int device.device_id);
        ("shader_log_capture", sexp_of_bool (Hashtbl.mem stream_logs device.device_id));
      ]

  (* --- Copy Operations --- (transfers take {!Backend_intf.buffer_loc} and resolve to the concrete
     [Metal.Buffer.t] here, against the device's private pool table.) *)
  let offset_contents buffer ~offset =
    let contents = Me.Buffer.contents buffer in
    if offset = 0 then contents else Ctypes.(to_voidp (from_voidp uint8_t contents +@ offset))

  let from_host ~dst ~dst_loc hosted =
    (* Pools are Shared, so host transfer is a direct offset-aware memcpy. Avoid an asynchronous
       no-copy staging buffer here: the wrapped host pointer's lifetime/cache visibility is fragile
       and used to drop the non-zero pool offset round-trip on some systems. *)
    let dst_ptr = Slab.resolve_pool dst.device dst_loc in
    let size_in_bytes = Ndarray.size_in_bytes hosted in
    let host_fatptr = Ndarray.get_fatptr_not_managed hosted in
    let (Ctypes_static.CPointer dst_fatptr) = offset_contents dst_ptr ~offset:dst_loc.offset in
    if Ctypes_ptr.Fat.compare dst_fatptr host_fatptr <> 0 then
      Ctypes_memory_stubs.memcpy ~dst:dst_fatptr ~src:host_fatptr ~size:size_in_bytes

  let to_host ~src ~src_loc hosted =
    (* Pools are Shared, so host transfer is a direct offset-aware memcpy. *)
    let src_ptr = Slab.resolve_pool src.device src_loc in
    let size_in_bytes = Ndarray.size_in_bytes hosted in
    let host_fatptr = Ndarray.get_fatptr_not_managed hosted in
    let (Ctypes_static.CPointer src_fatptr) = offset_contents src_ptr ~offset:src_loc.offset in
    if Ctypes_ptr.Fat.compare src_fatptr host_fatptr <> 0 then
      Ctypes_memory_stubs.memcpy ~dst:host_fatptr ~src:src_fatptr ~size:size_in_bytes

  (* The merge buffer is the device's reserved single-tenant pool (id [merge_buffer_pool_id]); grow
     it in place when a larger node arrives ([Slab.alloc_pool] overwrites the reserved entry). The
     merge buffer holds a copy of [tn], so it inherits [tn]'s storage-mode classification. *)
  let opt_alloc_merge_buffer ?mode ~size_in_bytes (device : device) : unit =
    if device.merge_buffer_capacity < size_in_bytes then (
      Slab.alloc_pool ?mode device ~pool_id:merge_buffer_pool_id ~size_in_bytes ~alignment:1;
      device.merge_buffer_capacity <- size_in_bytes);
    device.merge_buffer := Some { pool_id = merge_buffer_pool_id; offset = 0 }

  let device_to_device tn ~into_merge_buffer ~dst_loc ~dst ~src_loc ~src =
    let size_in_bytes = Lazy.force tn.Tn.size_in_bytes in
    let src_ptr = Slab.resolve_pool src.device src_loc in

    let memcpy ~dst_ptr ~dst_offset =
      (* Always use explicit copy as Metal doesn't have peer-to-peer memory access like CUDA *)
      let command_buffer = Me.CommandBuffer.on_queue dst.device.runner.queue in
      let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
      Me.BlitCommandEncoder.copy_from_buffer blit_encoder ~source_buffer:src_ptr
        ~source_offset:src_loc.offset ~destination_buffer:dst_ptr ~destination_offset:dst_offset
        ~size:size_in_bytes;
      Me.BlitCommandEncoder.end_encoding blit_encoder;
      Me.CommandBuffer.commit command_buffer
    in

    match (into_merge_buffer, dst_loc) with
    | No, None -> invalid_arg "Metal_backend.device_to_device: missing dst_loc"
    | No, Some dst_loc ->
        memcpy ~dst_ptr:(Slab.resolve_pool dst.device dst_loc) ~dst_offset:dst_loc.offset
    | Copy, _ ->
        opt_alloc_merge_buffer
          ?mode:(Option.map tn.Tn.memory_mode_intent ~f:fst)
          ~size_in_bytes dst.device;
        let loc = Option.value_exn ~here:[%here] !(dst.device.merge_buffer) in
        memcpy ~dst_ptr:(Slab.resolve_pool dst.device loc) ~dst_offset:loc.offset

  (* --- Compilation and Linking --- *)
  (* gh-ocannl-1605: a library compiled for one device, with the function and compute pipeline state
     of each kernel it holds, created on a kernel's first link and kept beside the library
     ([pipeline_of]). It is what a kernel code cache entry answers, so a kernel an earlier compile
     produced skips both the library compile and the pipeline creation -- the larger share of a
     Metal link. *)
  type program = {
    library : Me.Library.t;
    pipelines : (string, Me.Function.t * Me.ComputePipelineState.t) Hashtbl.t;
    pipelines_lock : Stdlib.Mutex.t;
  }

  let new_program library =
    { library; pipelines = Hashtbl.create (module String); pipelines_lock = Stdlib.Mutex.create () }

  (* gh-ocannl-1440: the programs a code value's first link resolved its kernels to, on the device
     it linked for, each with the symbol its kernel is looked up by. A relink of retained code
     ([Context.relink_outcome]) reuses them: Metal's device compile happens at link, so without this
     a relink would compile the source a second time. Per code value, so it lives exactly as long as
     the code (a retained one, or a linked routine's), whatever the kernel code cache evicts. *)
  type resolved_memo = (Me.Device.t * (program * string) array) option ref

  let sexp_of_resolved_memo _ = Sexp.Atom "<resolved kernels>"

  (* What a link needs to compile any subset of a code value's kernels: Metal compiles at link, when
     a device is available, so the rendering is retained ([Compile_driver.render_batch]). *)
  type kernels = {
    artifact_name : string;  (** The name the routine's [.metal] artifact is written under. *)
    source : string;  (** Every kernel, in batch order: the routine's full source. *)
    symbols : string array;  (** Each kernel's emitted name. *)
    given_names : string array;
        (** The names the caller asked for: a link-time compile attributes a name-caused failure to
            them (gh-ocannl-707). *)
    assemble : (int list -> string[@sexp.opaque]);
    key_text : (int -> string[@sexp.opaque]);
    resolved : resolved_memo;
  }
  [@@deriving sexp_of]

  type code = {
    kernels : kernels;
    func_name : string;
    kparams : (string * kparam_source) list;
    bindings : Indexing.unit_bindings;
    launch : Low_level.launch_dims;
  }
  [@@deriving sexp_of]

  type code_batch = {
    batch_kernels : kernels;
    funcs : (string * (string * kparam_source) list * Low_level.launch_dims) array;
        (* func_name * kparams * launch dims *)
    bindings : Indexing.unit_bindings;
  }
  [@@deriving sexp_of]

  module C_syntax_config (Input : sig
    val procs : Low_level.t array
  end) =
  struct
    include C_syntax.Pure_C_config (struct
      let procs = Input.procs
      let full_printf_support = false
    end)

    open PPrint (* Open PPrint locally *)
    open Indexing.Doc_helpers (* Open our helpers *)

    let main_kernel_prefix = "kernel"
    let buffer_prefix = "device "
    let buffer_suffix = fun ~pos -> " [[buffer(" ^ Int.to_string pos ^ ")]]"

    (* Signed index arithmetic (docs/proposals/signed-index-precision.md). *)
    let arg_int_prefix =
      if Utils.settings.large_models then "const int64_t& " else "const int32_t& "

    (* gh-ocannl-344: pass materialized tensor nodes through [metal_max_pools] bound pool buffers +
       a slot table instead of one buffer per tnode, so a kernel reaching hundreds of nodes stays
       under Metal's ~31 argument-buffer binding limit. [link_proc] binds the pools and fills the
       slots. *)
    let ptr_param_style = `Pooled metal_max_pools

    let extra_args =
      [
        "uint3 gid [[threadgroup_position_in_grid]]"; "uint3 lid [[thread_position_in_threadgroup]]";
      ]

    (* Hardware axis bindings (docs/proposals/axis-types-for-loops.md §5): [Grid] loops bind
       threadgroup positions, [Workgroup] loops thread-in-threadgroup positions, both already passed
       via [extra_args]. The binding site casts to the signed [loop_index_type]. *)
    let hardware_index ~kind ~slot =
      let reg = match kind with `Grid -> "gid" | `Workgroup -> "lid" in
      match slot with
      | 0 -> Some (reg ^ ".x")
      | 1 -> Some (reg ^ ".y")
      | 2 -> Some (reg ^ ".z")
      | _ -> None

    (* Sibling nests can exchange device cells between lanes: in particular, distributed zeroing
       precedes the lane-0 Tile_mma fallback. Fence device writes as well as shared tile storage
       (gh-ocannl-963); a threadgroup-only fence does not order that exchange. *)
    let barrier_syntax =
      Some "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"

    let shared_decl_prefix = Some "threadgroup "

    (* Metal rounds each [threadgroup] array up to 4 bytes (measured on an M4 Max, gh-ocannl-1226:
       two odd-length [half] tiles whose exact sizes total the limit make pipeline creation request
       4 bytes more), so the schedule-level estimate rounds each allocation the same way
       (gh-ocannl-1330). *)
    let workgroup_allocation_alignment = 4

    (* Warp-shuffle rendering of [Workgroup_reduce] accumulation loops (gh-ocannl-462):
       [ocannl_shfl_xor] wraps [simd_shuffle_xor] (builtins_metal.ml). Apple-silicon simdgroups are
       32 threads wide (the same width [mma_syntax] relies on). *)
    let warp_size = 32

    (* MSL is Clang-based C++: [__restrict] applies to the pooled per-node pointers, which address
       disjoint slab sub-ranges (gh-ocannl-164). No vectorization pragmas / stack-array alignment:
       the GPU payoff is memory transactions — eligible [Vectorized] loops render 128-bit packed
       loads/stores through the [float4]-payload pack structs (gh-ocannl-463). *)
    let restrict_keyword = Some "__restrict"

    (* Metal shader-compiler miscompilation workaround; see {!C_syntax.C_syntax_config} and the
       standalone repro [benchmarks/runners/ocannl/bench_metal_bug.ml]. *)
    let volatile_serial_accumulation = true
    let vectorize_pragma = []
    let aligned_local_attr = None

    (* 16-byte packs that only move data: no vector arithmetic, so no register tile and no count. *)
    let vector_file = { Vector_file.bytes = 16; registers = 0 }
    let vector_style = `Packed_struct

    let ident_blacklist =
      ident_blacklist
      (* MSL is a C++14 dialect, so the C++ keywords are reserved here on top of the C ones
         {!Pure_C_config} contributes (gh-ocannl-686). *)
      @ C_syntax.cpp_keywords
      @ C_syntax.builtin_idents Builtins_metal.builtins
      @ [
          (* MSL address-space qualifiers and function attributes — highly plausible tensor
             labels *)
          "kernel";
          "device";
          "constant";
          "thread";
          "threadgroup";
          "threadgroup_imageblock";
          (* MSL function qualifiers. [fragment] is not merely a plausible label — the accumulator
             contraction mints it ([Schedule]'s [contract_around]), so every tensorized candidate
             whose fragment reaches a declaration would fail MSL parsing ("expected unqualified-id")
             rather than merely risking it. *)
          "vertex";
          "fragment";
          "visible";
          (* MSL primitive types that differ from C and would shadow type declarations *)
          "half";
          "uint";
          "ushort";
          "uchar";
          "ulong";
          "bfloat";
          (* MSL built-in variables *)
          "gid";
          "lid";
        ]

    (* gh-ocannl-1338: [metal_stdlib]'s global types and macros ([uint4] -- the storage type of
       every uint4x32 node --, [float4], [M_PI_F], [MAXFLOAT]) cannot be shadowed by, or expand
       into, a node's identifier. *)
    let node_ident_prefix = C_syntax.node_ident_prefix_for_backend "metal"

    (* gh-ocannl-707: the file-scope names [metal_stdlib] puts in the GLOBAL namespace, which a
       [kernel void <name>(] definition collides with ("redefinition of 'float4' as different kind
       of symbol", "expected unqualified-id"). Harvested empirically rather than by reading: every
       identifier token of the toolchain's [include/metal] headers was compiled as a kernel name
       through [MTLDevice newLibraryWithSource:] after [#include <metal_stdlib>] and [using
       namespace metal;] (the prelude's exact shape), bisecting batches of 64; what failed is below,
       as the grammar it instantiates. Functions are absent on purpose -- they live in namespace
       [metal] and C++ overloading lets a global kernel of the same name coexist with them ([dot],
       [exp], [clamp], [select], [simd_sum] all compiled) -- and so are the names other tables
       already hold: the C and C++ keywords, the C floor's macros and types, and this backend's own
       [ident_blacklist] above. *)
    let vendor_routine_idents =
      let scalars =
        [
          "bool";
          "char";
          "uchar";
          "short";
          "ushort";
          "int";
          "uint";
          "long";
          "ulong";
          "llong";
          "ullong";
          "half";
          "float";
          "double";
          "bfloat";
          "quad";
        ]
      in
      let float_kinds = [ "F"; "H"; "BF" ] in
      List.concat
        [
          (* Scalar type names, and the vector families over them ([float4], [packed_half2],
             [simd_uint3], [vector_bool8], ...). *)
          scalars;
          C_syntax.vector_type_names
            ~prefixes:[ ""; "packed_"; "simd_"; "simd_packed_"; "vector_" ]
            ~scalars ~widths:[ 2; 3; 4; 8; 16 ];
          (* Math constants per float kind: [M_PI_F], [M_SQRT2_H], [M_E_BF], ... *)
          List.concat_map
            [
              "1_PI";
              "2_PI";
              "2_SQRTPI";
              "E";
              "LN10";
              "LN2";
              "LOG10E";
              "LOG2E";
              "PI_2";
              "PI_4";
              "PI";
              "SQRT1_2";
              "SQRT2";
            ] ~f:(fun c -> List.map float_kinds ~f:(fun kind -> "M_" ^ c ^ "_" ^ kind));
          (* Float limits per kind: [FLT_MAX], [HALF_EPSILON], [BFLT_DIG], ... *)
          List.concat_map [ "FLT"; "HALF"; "BFLT" ] ~f:(fun kind ->
              List.map
                [
                  "DECIMAL_DIG";
                  "DIG";
                  "EPSILON";
                  "MANT_DIG";
                  "MAX";
                  "MAX_10_EXP";
                  "MAX_EXP";
                  "MIN";
                  "MIN_10_EXP";
                  "MIN_EXP";
                  "RADIX";
                ] ~f:(fun limit -> kind ^ "_" ^ limit));
          List.map float_kinds ~f:(fun kind -> "HUGE_VAL" ^ kind);
          [ "MAXFLOAT"; "MAXHALF"; "MAXBFLOAT" ];
          (* Integer limits. *)
          [
            "CHAR_BIT";
            "CHAR_MAX";
            "CHAR_MIN";
            "SCHAR_MAX";
            "SCHAR_MIN";
            "UCHAR_MAX";
            "SHRT_MAX";
            "SHRT_MIN";
            "USHRT_MAX";
            "INT_MAX";
            "INT_MIN";
            "UINT_MAX";
            "LONG_MAX";
            "LONG_MIN";
            "ULONG_MAX";
          ];
          (* The headers' own attribute macros. *)
          [
            "METAL_ALIGN";
            "METAL_ARG_UNIFORM";
            "METAL_ASM";
            "METAL_CONST";
            "METAL_CONST_ARG";
            "METAL_DEPRECATED";
            "METAL_ENABLE_IF";
            "METAL_EXPLICIT";
            "METAL_FUNC";
            "METAL_IMPLICIT";
            "METAL_INTERNAL";
            "METAL_MAYBE_UNDEF";
            "METAL_NON_NULL_RETURN";
            "METAL_NOTHROW";
            "METAL_OS_LOG_FORMAT";
            "METAL_PURE";
            "METAL_UNAVAILABLE";
            "METAL_VALID_COMPARE_EXCHANGE_FAILURE_ORDER";
            "METAL_VALID_COMPARE_EXCHANGE_SUCCESS_ORDER";
            "METAL_VALID_EXCHANGE_ORDER";
            "METAL_VALID_LEVEL_ARG";
            "METAL_VALID_LOAD_ORDER";
            "METAL_VALID_LOD_ARG";
            "METAL_VALID_MODIFY_ORDER";
            "METAL_VALID_RENDER_TARGET";
            "METAL_VALID_STORE_ORDER";
            "METAL_ZERO_ARG";
          ];
          (* Dialect keywords and global names outside the families above. [main] is the reserved
             entry point; [metal] and [simd] are namespaces. *)
          [
            "as_type";
            "coherent";
            "complex";
            "imaginary";
            "is_function_constant_defined";
            "main";
            "metal";
            "nullptr_t";
            "object_data";
            "ray_data";
            "simd";
          ];
        ]

    let metal_log_object_name = "os_log_default"

    let typ_of_prec = function
      | Ops.Byte_prec _ -> "uchar"
      | Ops.Uint16_prec _ -> "ushort"
      | Ops.Int32_prec _ -> "int"
      | Ops.Uint32_prec _ -> "uint"
      | Ops.Uint4x32_prec _ -> "uint4" (* Metal's uint4 type - 128-bit *)
      | Ops.Half_prec _ -> "half"
      | Ops.Bfloat16_prec _ -> "bfloat" (* Metal supports bfloat16 natively *)
      (* MSL has no fp8 type: an e5m2 node is stored as a byte, and [Builtins_metal]'s software
         codec bridges every load and store (see [compute_prec]). *)
      | Ops.Fp8_prec _ -> "uchar"
      | Ops.Single_prec _ -> "float"
      | Ops.Double_prec _ ->
          raise @@ Utils.User_error "Metal backend does not support double precision"
      | Ops.Int64_prec _ -> "long"
      | Ops.Uint64_prec _ -> "ulong"
      | Ops.Void_prec -> "void"

    let supports_f64 = false

    let vec_typ_of_prec ~length prec =
      match (prec, length) with
      | Ops.Single_prec _, 4 -> "float4_t"
      | Ops.Double_prec _, 2 ->
          raise @@ Utils.User_error "Metal backend does not support double precision"
      | Ops.Int32_prec _, 4 -> "int32x4_t"
      | Ops.Uint32_prec _, 4 -> "uint32x4_t"
      | Ops.Int64_prec _, 2 -> "int64x2_t"
      | Ops.Uint64_prec _, 2 -> "uint64x2_t"
      | (Ops.Byte_prec _ | Ops.Fp8_prec _), 16 -> "int8x16_t"
      | Ops.Uint16_prec _, 8 -> "uint16x8_t"
      (* [Set_from_vec] assigns elements directly, so bfloat16 blocks must contain native [bfloat]
         values rather than raw [ushort] bit patterns. *)
      | Ops.Bfloat16_prec _, 8 -> "bfloat16x8_t"
      | Ops.Half_prec _, 8 -> "half8_t"
      | _, 1 -> typ_of_prec prec
      | _ -> invalid_arg "Metal_backend.vec_typ_of_prec: invalid combination"

    (* The one combination resolver both MMA hooks consult, device-free in [Metal_mma] beside the
       descriptor it must correspond to (gh-ocannl-1246). *)
    let mma_accumulator = Metal_mma.mma_accumulator

    let simdgroup_fragment = function
      | Ops.Single_prec _ -> "simdgroup_float8x8"
      | Ops.Half_prec _ -> "simdgroup_half8x8"
      | Ops.Bfloat16_prec _ -> "simdgroup_bfloat8x8"
      | prec -> invalid_arg ("Metal_backend.simdgroup_fragment: " ^ Ops.prec_string prec)

    (* [(accumulator, a, b, destination)] fragment types of [mma_accumulator]'s arm. *)
    let mma_fragment_types ~d_prec ~a_prec ~b_prec =
      Option.map (mma_accumulator ~d_prec ~a_prec ~b_prec) ~f:(fun acc_prec ->
          ( simdgroup_fragment acc_prec,
            simdgroup_fragment a_prec,
            simdgroup_fragment b_prec,
            simdgroup_fragment d_prec ))

    let mma_arm = Metal_mma.mma_arm

    (* gh-ocannl-1456: how [Metal_mma.k_loop_lines], which both hooks render, loads its operands. *)
    let mma_operand_loads = Metal_mma.operand_loads
    let mma_operand_fragments = Metal_mma.operand_fragments

    (* MSL leaves the element-to-thread mapping unspecified, so a half/bfloat fragment cannot supply
       the coordinates of a float accumulator (gh-ocannl-1075). Load the row-major table into the
       accumulator's own type and convert scalar cells at the coordinates it names. [simdgroup_load]
       accepts only device/threadgroup memory (MSL 4.1 section 6.8.1), so the table is one tile's
       worth of [float]s (256 bytes) of threadgroup storage. It holds the same 0..63 for every
       scope, so it is declared and initialized ONCE per kernel, at kernel scope, when any accepted
       emission converts (gh-ocannl-1255) — not once per scope, which charged 256 bytes for every
       converted [Tile_mma] copy an [Unroll] or [Partition] made. A single thread writes it even
       when the workgroup contains several SIMD groups; each converted scope's opening barrier
       orders that write before its reads. *)
    let mma_rc_table_len = 8 * 8
    let mma_rc_table_prec = Ops.single

    (* Whether the destination boundary converts: the accumulator fragment type differs from the
       destination's. The one predicate behind the table, the boundary lines and the table's
       bytes. *)
    let mma_d_converts ~acc_frag ~d_frag = not (String.equal acc_frag d_frag)

    let triple_converts (d_prec, a_prec, b_prec) =
      match mma_fragment_types ~d_prec ~a_prec ~b_prec with
      | Some (acc_frag, _, _, d_frag) -> mma_d_converts ~acc_frag ~d_frag
      | None -> false

    let mma_kernel_prologue ~triples =
      if not (List.exists triples ~f:triple_converts) then []
      else
        let typ = typ_of_prec mma_rc_table_prec in
        [
          Printf.sprintf "threadgroup %s ocannl_mma_rc8[%d];" typ mma_rc_table_len;
          "if (all(lid == uint3(0))) {";
          Printf.sprintf "  for (int __rc = 0; __rc < %d; ++__rc) ocannl_mma_rc8[__rc] = (%s)__rc;"
            mma_rc_table_len typ;
          "}";
        ]

    (* The table's bytes, for the schedule-level shared-memory estimate (gh-ocannl-1205): the same
       predicate and declaration as [mma_kernel_prologue], once per kernel (gh-ocannl-1255). *)
    let mma_kernel_workgroup_bytes ~triples =
      if List.exists triples ~f:triple_converts then
        mma_rc_table_len * Ops.prec_in_bytes mma_rc_table_prec
      else 0

    let mma_d_boundary_lines ~dir ~acc_frag ~d_frag ~acc ~ptr ~ldd =
      if not (mma_d_converts ~acc_frag ~d_frag) then
        [
          (match dir with
          | `Load -> Printf.sprintf "simdgroup_load(%s, %s, (ulong)%d);" acc ptr ldd
          | `Store -> Printf.sprintf "simdgroup_store(%s, %s, (ulong)%d);" acc ptr ldd);
        ]
      else
        let d_elem =
          match d_frag with
          | "simdgroup_half8x8" -> "half"
          | "simdgroup_bfloat8x8" -> "bfloat"
          | _ -> invalid_arg ("Metal_backend.mma_d_boundary_lines: unconverted " ^ d_frag)
        in
        (* [thread_elements()] returns a pointer, whose sizeof is not the per-lane element count.
           Divide this fragment's logical storage by the advertised 32-thread SIMD-group width. *)
        let elt = Printf.sprintf "%s.thread_elements()[__ei]" acc in
        let cell = Printf.sprintf "(%s)[(__rc >> 3) * %d + (__rc & 7)]" ptr ldd in
        [
          "{ /* simdgroup converted d boundary: coordinates from ocannl_mma_rc8 */";
          Printf.sprintf "  %s __mma_rc;" acc_frag;
          "  simdgroup_load(__mma_rc, ocannl_mma_rc8, (ulong)8);";
          Printf.sprintf
            "  for (int __ei = 0; __ei < (int)(sizeof(%s::storage_type) / sizeof(float) / 32); \
             ++__ei) {"
            acc_frag;
          "    const int __rc = (int)__mma_rc.thread_elements()[__ei];";
          (match dir with
          | `Load -> Printf.sprintf "    %s = (float)%s;" elt cell
          | `Store -> Printf.sprintf "    %s = (%s)%s;" cell d_elem elt);
          "  }";
          "}";
        ]

    (* Cooperative tile-MMA emission for [Low_level.Tile_mma] via MSL [simdgroup_matrix]
       (docs/proposals/tensorize-mma.md §4): fragment blocks of 8×8 tiles held jointly by the
       32-thread simdgroup, loaded with [simdgroup_load] straight from [device] memory or
       [threadgroup] tiles (i.e. out of [Stage]'s shared staging), accumulated with
       [simdgroup_multiply_accumulate], stored once at the end — the accumulator fragments are
       resident across the whole [k] extent of the block statement. Transposed-stored operands
       ([ta]/[tb]) load with [simdgroup_load]'s [transpose_matrix] flag and swapped tile-offset
       arithmetic. Declines (fallback path) on: a precision combination [mma_fragment_types] does
       not resolve, extents not multiples of 8, thread-space (stack-array) operands (not loadable by
       [simdgroup_load]), and devices below the Apple7 family. *)
    let mma_syntax =
      Some
        (fun ~d_prec
          ~a_prec
          ~b_prec
          ~ta
          ~tb
          ~m
          ~n
          ~k
          ~d:(d_ptr, ldd, d_space, d_layout)
          ~a:(lda, a_space, a_layout)
          ~b:(ldb, b_space, b_layout)
        ->
          (* [simdgroup_matrix] loads take a plain pointer and leading dimension; Metal has no
             [ldmatrix] analogue, so a swizzled operand layout declines to the caller's scalar
             fallback (gh-ocannl-481 item 3, D2). *)
          let plain = function `Plain -> true | `Swizzled_b128 -> false in
          let tile = Metal_mma.tile in
          let frag_types =
            if plain d_layout && plain a_layout && plain b_layout then
              mma_fragment_types ~d_prec ~a_prec ~b_prec
            else None
          in
          let addr_space = function
            | `Device -> Some "device"
            | `Shared -> Some "threadgroup"
            | `Thread | `Fragment _ -> None
          in
          match (frag_types, d_space, addr_space a_space, addr_space b_space) with
          | Some (_acc_frag, a_frag, b_frag, _d_frag), `Fragment fragment, Some a_as, Some b_as
            when m % tile = 0
                 && n % tile = 0
                 && k % tile = 0
                 && Option.is_some (hardware_limits ()).Backend_intf.mma ->
              let ptr_decl name aspace elem ptr =
                string (Printf.sprintf "%s %s *%s = " aspace elem name) ^^ ptr ^^ semi
              in
              let body_lines =
                Metal_mma.k_loop_lines ~a_frag ~b_frag ~acc:fragment ~ta ~tb ~lda ~ldb ~m ~n ~k
              in
              let body ~a_ptr ~b_ptr =
                ptr_decl "__mma_ap" (Printf.sprintf "const %s" a_as) (typ_of_prec a_prec) a_ptr
                ^^ hardline
                ^^ ptr_decl "__mma_bp" (Printf.sprintf "const %s" b_as) (typ_of_prec b_prec) b_ptr
                ^^ hardline
                ^^ separate_map hardline string body_lines
              in
              Some
                (fun ~a_ptr ~b_ptr ->
                  group
                    (string (Printf.sprintf "{ /* tile_mma fragment update %dx%dx%d */" m n k)
                    ^^ nest 2 (hardline ^^ body ~a_ptr ~b_ptr)
                    ^^ hardline
                    ^^ string "threadgroup_barrier(mem_flags::mem_threadgroup);"
                    ^^ hardline ^^ rbrace))
          | _ -> (
              match (frag_types, addr_space d_space, addr_space a_space, addr_space b_space) with
              | Some (acc_frag, a_frag, b_frag, d_frag), Some d_as, Some a_as, Some b_as
                when m % tile = 0
                     && n % tile = 0
                     && k % tile = 0
                     && Option.is_some (hardware_limits ()).Backend_intf.mma ->
                  let mt = m / tile and nt = n / tile in
                  let ptr_decl name aspace elem ptr =
                    string (Printf.sprintf "%s %s *%s = " aspace elem name) ^^ ptr ^^ semi
                  in
                  (* Bracketing barriers: sibling statements (zeroing of [d], cooperative staging)
                     execute lane-partitioned, so the fragment loads must observe the other lanes'
                     writes, and later statements the stores. Uniform per the barrier-strength
                     validation. *)
                  let barrier =
                    "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"
                  in
                  let body_lines =
                    [
                      barrier;
                      Printf.sprintf "%s __mma_acc[%d][%d];" acc_frag mt nt;
                      Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                      Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                    ]
                    @ List.map
                        (mma_d_boundary_lines ~dir:`Load ~acc_frag ~d_frag
                           ~acc:"__mma_acc[__mi][__ni]"
                           ~ptr:
                             (Printf.sprintf "__mma_dp + __mi * %d * %d + __ni * %d" tile ldd tile)
                           ~ldd)
                        ~f:(fun line -> "    " ^ line)
                    @ [ "  }"; "}" ]
                    @ Metal_mma.k_loop_lines ~a_frag ~b_frag ~acc:"__mma_acc" ~ta ~tb ~lda ~ldb ~m
                        ~n ~k
                    @ [
                        Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                        Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      ]
                    @ List.map
                        (mma_d_boundary_lines ~dir:`Store ~acc_frag ~d_frag
                           ~acc:"__mma_acc[__mi][__ni]"
                           ~ptr:
                             (Printf.sprintf "__mma_dp + __mi * %d * %d + __ni * %d" tile ldd tile)
                           ~ldd)
                        ~f:(fun line -> "    " ^ line)
                    @ [ "  }"; "}"; barrier ]
                  in
                  let body ~a_ptr ~b_ptr =
                    ptr_decl "__mma_dp" d_as (typ_of_prec d_prec) d_ptr
                    ^^ hardline
                    ^^ ptr_decl "__mma_ap" (Printf.sprintf "const %s" a_as) (typ_of_prec a_prec)
                         a_ptr
                    ^^ hardline
                    ^^ ptr_decl "__mma_bp" (Printf.sprintf "const %s" b_as) (typ_of_prec b_prec)
                         b_ptr
                    ^^ hardline
                    ^^ separate_map hardline string body_lines
                  in
                  Some
                    (fun ~a_ptr ~b_ptr ->
                      group
                        (string (Printf.sprintf "{ /* tile_mma %dx%dx%d */" m n k)
                        ^^ nest 2 (hardline ^^ body ~a_ptr ~b_ptr)
                        ^^ hardline ^^ rbrace))
              | _ -> None))

    (* Cross-[k_o] accumulator residency (gh-ocannl-480): one marked local tile becomes a
       simdgroup-fragment array whose load/store bracket the whole serial reduction. The nested
       [Tile_mma] calls see [`Fragment fragment] and therefore emit update-only MMA steps. *)
    let mma_fragment_syntax =
      Some
        (fun ~d_prec
          ~a_prec
          ~b_prec
          ~m
          ~n
          ~k
          ~fragment
          ~target:(d_ptr, ldd, d_space, d_layout)
          ~a:(_, a_space, a_layout)
          ~b:(_, b_space, b_layout)
          ~body
        ->
          (* [simdgroup_matrix] loads take a plain pointer and leading dimension; Metal has no
             [ldmatrix] analogue, so a swizzled operand layout declines to the caller's scalar
             fallback (gh-ocannl-481 item 3, D2). *)
          let plain = function `Plain -> true | `Swizzled_b128 -> false in
          let tile = Metal_mma.tile in
          let frag_types =
            if plain d_layout && plain a_layout && plain b_layout then
              mma_fragment_types ~d_prec ~a_prec ~b_prec
            else None
          in
          let loadable = function `Device | `Shared -> true | `Thread | `Fragment _ -> false in
          match frag_types with
          | Some (acc_frag, _a_frag, _b_frag, d_frag)
            when m % tile = 0
                 && n % tile = 0
                 && k % tile = 0
                 && loadable d_space && loadable a_space && loadable b_space
                 && Option.is_some (hardware_limits ()).Backend_intf.mma ->
              let mt = m / tile and nt = n / tile in
              let d_as =
                match d_space with
                | `Device -> "device"
                | `Shared -> "threadgroup"
                | _ -> assert false
              in
              let barrier =
                "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"
              in
              let d_addr = Printf.sprintf "__mma_dp + __mi * %d * %d + __ni * %d" tile ldd tile in
              let lines_before =
                [
                  barrier;
                  Printf.sprintf "%s %s[%d][%d];" acc_frag fragment mt nt;
                  Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                  Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                ]
                @ List.map
                    (mma_d_boundary_lines ~dir:`Load ~acc_frag ~d_frag
                       ~acc:(Printf.sprintf "%s[__mi][__ni]" fragment)
                       ~ptr:d_addr ~ldd)
                    ~f:(fun line -> "    " ^ line)
                @ [ "  }"; "}"; "/* simdgroup fragment reduction body begins */" ]
              in
              let lines_after =
                [ "/* simdgroup fragment reduction body ends */" ]
                @ [
                    Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                    Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                  ]
                @ List.map
                    (mma_d_boundary_lines ~dir:`Store ~acc_frag ~d_frag
                       ~acc:(Printf.sprintf "%s[__mi][__ni]" fragment)
                       ~ptr:d_addr ~ldd)
                    ~f:(fun line -> "    " ^ line)
                @ [ "  }"; "}"; barrier ]
              in
              let d_decl =
                string (Printf.sprintf "%s %s *__mma_dp = " d_as (typ_of_prec d_prec))
                ^^ d_ptr ^^ semi
              in
              Some
                (group
                   (string (Printf.sprintf "{ /* simdgroup fragment %dx%d across k_o */" m n)
                   ^^ nest 2
                        (hardline ^^ d_decl ^^ hardline
                        ^^ separate_map hardline string lines_before
                        ^^ hardline ^^ body () ^^ hardline
                        ^^ separate_map hardline string lines_after)
                   ^^ hardline ^^ rbrace))
          | _ -> None)

    let metal_prec_suffix_float = function
      | Ops.Byte_prec _ -> ""
      | Ops.Uint16_prec _ -> ""
      | Ops.Int32_prec _ -> ""
      | Ops.Uint32_prec _ -> ""
      | Ops.Uint4x32_prec _ -> "" (* No specific suffix for uint4 *)
      | Ops.Half_prec _ -> "h"
      | Ops.Bfloat16_prec _ -> "bf" (* Verified: [0.0bf] is a bfloat literal MSL accepts. *)
      (* Not a capability limit but an invariant: [compute_prec] maps fp8 to f32, so no operator is
         ever rendered at fp8 and no fp8 literal is ever spelled. *)
      | Ops.Fp8_prec _ ->
          invalid_arg "Metal_backend: fp8 arithmetic renders at single precision (compute_prec)"
      | Ops.Single_prec _ -> "f"
      | Ops.Double_prec _ ->
          raise @@ Utils.User_error "Metal backend does not support double precision"
      | Ops.Int64_prec _ -> "l"
      | Ops.Uint64_prec _ -> "ul"
      | Ops.Void_prec -> ""

    (* MSL's math library has [float] and [half] overloads but no [bfloat] ones, so a builtin called
       on bfloat operands promotes them and returns [float] -- and unlike C, MSL rejects the
       narrowing assignment back to a bfloat destination ("assigning to 'bfloat' from incompatible
       type 'float'"). Every math builtin is affected, so the result is bridged back here rather
       than one operator at a time (gh-ocannl-549).

       The *call* is unambiguous: MSL's [bfloat] is a native scalar type with a single implicit
       conversion, unlike [__nv_bfloat16] / [__hip_bfloat16], whose several conversion operators
       make the call itself ambiguous on the other GPU backends. Conversely those backends accept
       the narrowing assignment (their converting constructor is implicit), which is why the same op
       tables fail there only where the float result becomes the operand of a bfloat16 binop -- i.e.
       only in the placement that inlines it.

       Arithmetic, comparisons, the ternary and the bfloat literal suffix ([0.0bf], used by the
       gates and by [Recip]) are all native for [bfloat] and need no bridging. *)
    let bf16_from_builtin prec doc =
      match prec with Ops.Bfloat16_prec _ -> group (string "(bfloat)" ^^ parens doc) | _ -> doc

    let ternop_syntax prec op =
      match op with
      | Ops.Where ->
          (* A short-circuiting ternary, exactly as on the C/CUDA/HIP backends — NOT [select]: MSL's
             [select] is a function call that evaluates both branches, so a range guard (an inlined
             concatenation's component guards, gh-504's clamped-window guards) would still evaluate
             its guarded out-of-range buffer read. The whole conditional is parenthesized (cf. the
             CUDA rendering's precedence note). *)
          fun v1 v2 v3 -> group (parens (parens v1 ^^ string " ? " ^^ v2 ^^ string " : " ^^ v3))
      | FMA ->
          fun v1 v2 v3 ->
            bf16_from_builtin prec
              (group (string "fma(" ^^ separate comma_sep [ v1; v2; v3 ] ^^ rparen))
      | Mul3 -> fun v1 v2 v3 -> group (parens (v1 ^^ string " * " ^^ v2 ^^ string " * " ^^ v3))

    let infix_binop op v1 v2 = parens (infix 2 1 (string op) v1 v2)

    let binop_syntax prec op =
      (* The match stays exhaustive over (op, prec) -- that is what catches a newly added operator
         here -- but arms whose spelling is plain C delegate to {!C_syntax.default_binop_syntax}
         rather than restating the token. *)
      let f = infix_binop in
      let func fn v1 v2 = group (string fn ^^ parens (v1 ^^ comma ^^ space ^^ v2)) in
      (* Math-library calls need the bfloat16 result bridged back; see [bf16_from_builtin]. *)
      let math fn v1 v2 = bf16_from_builtin prec (func fn v1 v2) in
      match (op, prec) with
      | Ops.Add, _ -> f "+"
      | Sub, _ -> f "-"
      | Mul, _ -> f "*"
      | Div, _ -> f "/"
      | ( Mod,
          ( Ops.Byte_prec _ | Ops.Uint16_prec _ | Ops.Int32_prec _ | Ops.Uint32_prec _
          | Ops.Int64_prec _ | Ops.Uint64_prec _ ) ) ->
          f "%"
      (* The fast [fmod] is [x - y * trunc(x / y)]-shaped (gh-ocannl-1684, M4 Max): NaN for a finite
         [x] over an infinite [y], where C returns [x], and [+0] for an exact remainder of a
         negative [x], where C returns [-0]. [precise::fmod] is C's; it has no half overload, so
         half widens (the remainder is exact in half: no rounding back). *)
      | Mod, Ops.Half_prec _ ->
          fun v1 v2 ->
            group
              (string "(half)precise::fmod(" ^^ string "(float)" ^^ parens v1 ^^ comma ^^ space
             ^^ string "(float)" ^^ parens v2 ^^ rparen)
      | Mod, _ -> math "precise::fmod"
      | Max, _ -> math "fmax"
      | Min, _ -> math "fmin"
      (* Comparisons and logical connectives are precision-independent and spelled the same in MSL
         as in C, so they render through the shared default. The constructors stay listed to keep
         the match exhaustiveness-checked. *)
      | ((Cmpeq | Cmpne | Cmplt | Cmple | And | Or) as op), _ ->
          C_syntax.default_binop_syntax prec op
      | Relu_gate, Ops.Half_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0h"))
                 ^^ space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                 ^^ string "0.0h"))
      | Relu_gate, Ops.Single_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0f"))
                 ^^ space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                 ^^ string "0.0f"))
      | Relu_gate, Ops.Double_prec _ ->
          raise @@ Utils.User_error "Metal backend does not support double precision"
      | Relu_gate, _ (* Byte_prec, Void_prec *) ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0"))
                 ^^ space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space ^^ string "0"
                 ))
      | Satur01_gate, p_res ->
          let s = metal_prec_suffix_float p_res in
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens (v1 ^^ string (" > 0.0" ^ s ^ " && ") ^^ v1 ^^ string (" < 1.0" ^ s)))
                 ^^ space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                 ^^ string ("0.0" ^ s)))
      | ToPowOf, Ops.Half_prec _ ->
          (* Widen both operands and round back once, as CUDA and HIP do (gh-ocannl-1198), so half
             pow is f32 [pow] narrowed under either compile-option math policy (the modern split on
             macOS 15+, the legacy [fastMathEnabled=false] on macOS 14) rather than whatever the
             half overload does there. *)
          fun v1 v2 ->
            group
              (string "(half)pow("
              ^^ separate comma_sep [ string "(float)" ^^ parens v1; string "(float)" ^^ parens v2 ]
              ^^ rparen)
      | ToPowOf, _ -> math "pow"
      (* The RNG ops call the same builtins under the same precision contract on every C-family
         backend, so they render through the shared helper. *)
      | ((Threefry4x32_crypto | Threefry4x32_light | Uint4x32_to_prec_uniform_lane) as op), _ ->
          C_syntax.rng_binop_syntax ~backend:"Metal" ~call:func prec op
      | Arg1, _ | Arg2, _ -> invalid_arg "Metal C_syntax_config: Arg1/Arg2 not operators"

    let unop_syntax prec op =
      let func_doc fn v = group (string fn ^^ parens v) in
      (* Math-library calls need the bfloat16 result bridged back; see [bf16_from_builtin]. *)
      let math_doc fn v = bf16_from_builtin prec (func_doc fn v) in
      (* Half operands widen and round back once, as half [ToPowOf] does (gh-ocannl-1288): a half
         result is the f32 builtin narrowed under whichever compile-option math policy is in force,
         rather than whatever the half overload does under it. Measured on M4 Max over all 65536
         half inputs ([benchmarks/runners/ocannl/metal_half_math_probe.ml]), the overloads already
         matched the widened form bitwise under both policies except [sin(-0)] (the overload drops
         the sign) and, under the modern policy only, [sin]/[cos] of inf and NaN (the overload gives
         NaN, the fast f32 functions 0 -- since gh-ocannl-1435 [ocannl_sin]/[ocannl_cos] guard
         those, see [Builtins_metal]). [Recip_sqrt] and [Tanh_approx] keep their overloads: see the
         backend-precision note. *)
      let widened_math fn =
        match prec with
        | Ops.Half_prec _ ->
            fun v -> group (string ("(half)" ^ fn ^ "((float)") ^^ parens v ^^ rparen)
        | _ -> math_doc fn
      in
      match (op, prec) with
      | Ops.Identity, _ -> fun v -> v
      | Neg, _ -> fun v -> string "-" ^^ v
      | Exp, _ -> widened_math "exp"
      | Log, _ -> widened_math "log"
      | Exp2, _ -> widened_math "exp2"
      | Log2, _ -> widened_math "log2"
      | Sin, _ -> widened_math "ocannl_sin"
      | Cos, _ -> widened_math "ocannl_cos"
      | Sqrt, _ -> widened_math "sqrt"
      | Relu, Ops.Half_prec _ -> fun v -> func_doc "max" (separate comma_sep [ string "0.0h"; v ])
      | Relu, Ops.Single_prec _ -> fun v -> func_doc "max" (separate comma_sep [ string "0.0f"; v ])
      | Relu, Ops.Bfloat16_prec _ ->
          (* Bridged like every other math builtin (see [bf16_from_builtin]), but spelled out
             because the other operand is a literal: an untyped [0] would resolve to the *integer*
             [max] and truncate the activation to an integer — with sub-unit activations that
             silently zeroes the whole forward pass. *)
          fun v -> group (string "(bfloat)max(0.0f, (float)" ^^ parens v ^^ rparen)
      | Relu, Ops.Double_prec _ ->
          raise @@ Utils.User_error "Metal backend does not support double precision"
      | Relu, _ (* Byte_prec, Void_prec *) ->
          fun v -> func_doc "max" (separate comma_sep [ string "0"; v ])
      | Satur01, Ops.Bfloat16_prec _ ->
          (* Like [Relu], spelled out rather than bridged: [clamp(bfloat, 0.0bf, 1.0bf)] — the
             suffixed form the generic arm below would emit — is itself ambiguous in MSL. *)
          fun v -> group (string "(bfloat)clamp((float)" ^^ parens v ^^ string ", 0.0f, 1.0f)")
      | Satur01, p ->
          let s = metal_prec_suffix_float p in
          fun v ->
            func_doc "clamp" (separate comma_sep [ v; string ("0.0" ^ s); string ("1.0" ^ s) ])
      | Recip, p ->
          let s = metal_prec_suffix_float p in
          fun v -> infix_binop "/" (string @@ "1.0" ^ s) v
      | Recip_sqrt, _ -> math_doc "rsqrt"
      | Trunc, _ -> math_doc "trunc"
      | Tanh_approx, Ops.Half_prec _ -> math_doc "tanh"
      (* gh-ocannl-1435: the fast f32 [tanh] the production policy selects returns 0 for x from
         about 43.7 and NaN from about 44.4 up (+inf included), +0 for -0, flushes |x| < 2^-26 to 0
         and is off by up to 2050 ulps near 2^-12
         (benchmarks/runners/ocannl/metal_f32_math_probe.ml). [precise::tanh] is within 2 ulps of
         f64 everywhere, IEEE on the special values, and bitwise the legacy policy's [tanh]; it
         costs 4x on 1024 back-to-back calls per thread (ALU-bound throughput) and nothing
         measurable on a load-call-store stream. bfloat16 widens explicitly so the call resolves to
         the f32 overload. *)
      | Tanh_approx, Ops.Bfloat16_prec _ ->
          fun v -> group (string "(bfloat)precise::tanh((float)" ^^ parens v ^^ rparen)
      | Tanh_approx, _ -> math_doc "precise::tanh"
      | Not, _ -> fun v -> string "!" ^^ v
      | Uint4x32_to_prec_uniform1, _ ->
          fun v -> func_doc ("uint4x32_to_" ^ Ops.prec_string prec ^ "_uniform") v
    (* Logical not *)

    (* Keep vec_unop_syntax same as in pure C syntax. *)

    let convert_precision ~from ~to_ =
      match (from, to_) with
      | Ops.Double_prec _, Ops.Double_prec _
      | Ops.Single_prec _, Ops.Single_prec _
      | Ops.Half_prec _, Ops.Half_prec _
      | Ops.Byte_prec _, Ops.Byte_prec _
      | Ops.Uint16_prec _, Ops.Uint16_prec _
      | Ops.Int32_prec _, Ops.Int32_prec _
      | Ops.Uint32_prec _, Ops.Uint32_prec _
      | Ops.Int64_prec _, Ops.Int64_prec _
      | Ops.Uint64_prec _, Ops.Uint64_prec _
      | Ops.Uint4x32_prec _, Ops.Uint4x32_prec _
      | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _
      | Ops.Fp8_prec _, Ops.Fp8_prec _
      | Ops.Void_prec, Ops.Void_prec ->
          ("", "")
      (* Uint4x32 conversions - special handling *)
      | Ops.Uint4x32_prec _, _ -> ("uint4x32_to_" ^ Ops.prec_string to_ ^ "_uniform(", ")")
      | _, Ops.Uint4x32_prec _ -> (Ops.prec_string from ^ "_to_uint4x32(", ")")
      (* fp8 conversions go through the software codec, which is where the storage/compute seam
         lands for e5m2 on Metal: a load widens once, a store narrows once, and the arithmetic in
         between is f32 ([compute_prec]). The spellings mirror {!Ops.c_convert_precision}'s fp8
         arms, so a tensor written by a C backend reads back identically here. *)
      | Ops.Fp8_prec _, (Ops.Single_prec _ | Ops.Double_prec _) -> ("fp8_to_single(", ")")
      | (Ops.Single_prec _ | Ops.Double_prec _), Ops.Fp8_prec _ -> ("single_to_fp8(", ")")
      | Ops.Fp8_prec _, _ -> ("(" ^ typ_of_prec to_ ^ ")(fp8_to_single(", "))")
      | _, Ops.Fp8_prec _ -> ("single_to_fp8((float)(", "))")
      (* Metal has no native double. Keep double tensor storage unsupported in [typ_of_prec], so a
         declared double buffer/local still fails instead of being silently degraded. This
         conversion case is only for scalar expression casts emitted by the shared lowering;
         currently the guarded dynamic gather uses [double] as a signed guard precision for
         float-precision ids, and Metal renders that scalar guard as [float] (exact only below 2^24
         -- prefer integer-precision ids, e.g. [Nn_blocks.class_ids_of_int_list], whose guard runs
         in native integer comparisons instead). *)
      | _, Ops.Double_prec _ -> ("(float)(", ")")
      (* Default case for all other conversions *)
      | _ -> ("(" ^ typ_of_prec to_ ^ ")(", ")")

    (* [half] and [bfloat] are native MSL scalars and compute where they store; fp8 is not a type at
       all here, so it takes gh-ocannl-517's seam instead — stored as a byte, computed in f32,
       converted once per load and once per store by [convert_precision] above. The same resolution
       [Numerics.cpu_compute_prec] gives fp8 on the CPU backends, and a function of the storage
       precision alone, as the signature requires. *)
    let compute_prec = function Ops.Fp8_prec _ -> Ops.single | prec -> prec

    (* gh-ocannl-663: the ordinary seeded f16/bf16 tiles accumulate in fragments of the storage
       precision, so serial and tensorized residency agree; fp8 has no accumulator format and
       computes in f32 above regardless of policy. Restated next to the [compute_prec] override
       because the pair binds at [include] time (signature coupling note).

       Under [Numerics.Fp16_wide], f16 accumulators reside in f32 here and in the tensorized path:
       MSL accepts half operand fragments with an f32 accumulator, while the two MMA lowering hooks
       convert the f16 destination only at their outer load/store boundary (gh-ocannl-837). *)
    let accum_prec prec =
      match prec with
      | Ops.Half_prec _ when Numerics.fp16_accum_wide () -> Ops.single
      (* gh-ocannl-838: likewise bf16 under [Numerics.Bf16_wide], where the uniform-bf16 MMA arm
         swaps to a float accumulator over bfloat operands (gh-ocannl-923). *)
      | Ops.Bfloat16_prec _ when Numerics.bf16_accum_wide () -> Ops.single
      | _ -> compute_prec prec

    (* MSL has no [long long]: its 64-bit signed scalar is [long] (what [int64_t] names here), so
       the portable-C default's [(long long)] / [%lld] pair would not compile. [os_log] checks the
       format string against the argument types at shader-compile time, so the width has to be right
       rather than merely wide enough. *)
    let log_index_arg doc =
      if Utils.settings.large_models then ("%ld", PPrint.(string "(long)" ^^ doc)) else ("%d", doc)

    (* If we wanted to reintroduce the log_id parameter: [Some ("const int&", "log_id")]. *)
    let kernel_log_param = None
    let log_involves_file_management = false

    let pp_log_statement ~log_param_c_expr_doc ~base_message_literal:base ~args_docs =
      assert (Option.is_none log_param_c_expr_doc);
      let open PPrint in
      (* Metal os_log handles newlines directly. Prefix with captured_log_prefix and log_id for
         consistency. *)
      let base = if String.is_suffix base ~suffix:"\n" then String.drop_suffix base 1 else base in
      let base =
        let with_ =
          (* if for_log_trace_tree then *) "$"
          (* else "\\n" *)
        in
        String.substr_replace_all base ~pattern:"\n" ~with_
      in
      let base_doc = dquotes (string base) in
      if List.is_empty args_docs then
        string metal_log_object_name ^^ string ".log_debug(" ^^ base_doc ^^ rparen ^^ semi
      else
        group
          (string metal_log_object_name ^^ string ".log_debug(" ^^ base_doc ^^ comma
          ^^ nest 4 (break 1 ^^ separate (comma ^^ break 1) args_docs)
          ^^ rparen ^^ semi)
  end

  let codegen_capabilities () =
    let module Config = C_syntax_config (struct
      let procs = [||]
    end) in
    C_syntax.codegen_capabilities (module Config)

  let runtime_math_api =
    (* These instance methods were added in macOS 15. Query the runtime capability rather than the
       host OS version: this is the capability whose absence matters, and keeps macOS 14 on the
       legacy safe spelling without ever sending an unavailable selector. The selector vocabulary
       and all-or-nothing decision live in [Compiler_options.metal_math_api], injectable in its
       GPU-free test; [Metal_math_api_runtime] owns the one real query shared with the standalone
       probe. *)
    lazy (Metal_math_api_runtime.get ())

  (* Whether a failed [newLibraryWithSource:] reported a diagnostic the compiler attached to the
     source, as ["program_source:LINE:COL: error: ..."]: the compiler read the generated code and
     rejected it, so the failure is that code's. *)
  let compiler_diagnosed_error msg =
    let prefix = "program_source:" in
    let n = String.length msg in
    let rec digits i = if i < n && Char.is_digit msg.[i] then digits (i + 1) else i in
    let diagnosed_at start =
      let after_line = digits start in
      after_line > start && after_line < n
      && Char.equal msg.[after_line] ':'
      &&
      let after_col = digits (after_line + 1) in
      after_col > after_line + 1
      && String.is_prefix (String.drop_prefix msg after_col) ~prefix:": error:"
    in
    List.exists (String.substr_index_all msg ~may_overlap:false ~pattern:prefix) ~f:(fun i ->
        diagnosed_at (i + String.length prefix))

  (* The compile options a link compiles under, computed once per link: what the kernel code cache
     keys on is what [compile_metal_source] applies. *)
  let metal_option_state () =
    Compiler_options.metal
      ~routine_logging:(Utils.debug_log_from_routines ())
      ~math_api:(Lazy.force runtime_math_api)

  (* Writes [source] as the [<name>.metal] build artifact when debug files are on. *)
  let write_source_artifact ~name source =
    if Utils.settings.output_debug_files_in_build_directory then (
      let metal_file = Utils.build_file (name ^ ".metal") in
      Stdio.Out_channel.write_all metal_file ~data:source;
      Some metal_file)
    else None

  (* The [MTLCompileOptions] an option state names. *)
  let metal_compile_options option_state =
    let options = Me.CompileOptions.init () in
    List.iter option_state ~f:(function
      | Compiler_options.Language_version_3_1 ->
          (* Version 3.1 is required for the [bfloat] type (bfloat16 precision). *)
          Me.CompileOptions.set_language_version options
            Me.CompileOptions.LanguageVersion.version_3_1
      | Language_version_3_2 ->
          Me.CompileOptions.set_language_version options
            Me.CompileOptions.LanguageVersion.version_3_2
      | Fast_math_enabled enabled -> Me.CompileOptions.set_fast_math_enabled options enabled
      | Math_mode_safe -> Me.CompileOptions.set_math_mode options Me.CompileOptions.MathMode.Safe
      | Math_functions_fast ->
          Me.CompileOptions.set_math_floating_point_functions options
            Me.CompileOptions.MathFloatingPointFunctions.Fast
      | Enable_logging -> Me.CompileOptions.set_enable_logging options true);
    options

  let%diagn_sexp compile_metal_source ~name ~routine_names ~source ~metal_file ~option_state ~device
      =
    let options = metal_compile_options option_state in
    [%log "metal options", (Compiler_options.render_metal option_state : string)];
    (* The kernels this library compiles, counted where Metal's compiler runs (gh-ocannl-1314):
       [Compile_driver] counted them as rendered only. *)
    Compile_cost.record_kernels ~rendered:0 ~compiled:(List.length routine_names);
    try Me.Library.on_device device ~source options
    with Failure msg ->
      let options_rendered = Compiler_options.render_metal option_state in
      let exn =
        if compiler_diagnosed_error msg && not (C_syntax.missing_include msg) then
          (* The compiler rejected the generated source: a typed decline of this kernel, as the cc
             backend raises at its compiler call and CUDA's [classify_failure] answers for
             [NVRTC_ERROR_COMPILATION] (ahrefs/ocannl#1387). Untyped, it was [Fatal] under
             [strict_failure_classification] and [User_schedule] provenance, so one rejected kernel
             ended a tuner search or the whole isolated per-kernel table. The detail carries the
             compiler's report, not the source: a declined candidate is counted and logged, and
             several of them must not each print a whole kernel. *)
          let source_note =
            match metal_file with
            | Some file -> "the generated .metal file at " ^ file
            | None ->
                "the generated .metal file (rerun with \
                 --ocannl_output_debug_files_in_build_directory=true to write it)"
          in
          Schedule_outcome.Cause_at
            ( Schedule_outcome.Backend_compile,
              Schedule_outcome.Backend_rejected
                {
                  backend = Device_config.name;
                  stage = "compiler";
                  severity = Schedule_outcome.Compiler_bug;
                  detail =
                    Printf.sprintf
                      "OCANNL metal backend: generated code for %s failed to compile.\n\
                       %s\n\
                       metal options: %s\n\
                       Compiler output:\n\
                       %s"
                      name
                      (C_syntax.codegen_bug_claim ~artifact:source_note)
                      options_rendered msg;
                } )
        else
          (* No compiler diagnostic, or a header the compiler could not open: nothing establishes
             that the failure is this kernel's rather than the environment's (an unavailable
             language version or system header fails every kernel alike), so it keeps propagating
             untyped. *)
          let error_msg =
            Printf.sprintf "Metal compilation failed for %s:\n%s\nmetal options: %s\nSource:\n%s"
              name msg options_rendered source
          in
          Stdio.prerr_endline error_msg;
          Failure error_msg
      in
      (* Metal compiles here, at link time, so the failure never crosses [Compile_driver]'s
         attribution; this is the same check (gh-ocannl-707). Attributed on [msg] alone: the untyped
         [error_msg] embeds the source, where every routine's declaration appears. *)
      let backtrace = Stdlib.Printexc.get_callstack 16 in
      C_syntax.attribute_routine_name_failure ~report:msg ~routine_names exn backtrace

  module Compile = C_syntax.Compile_driver (C_syntax_config)

  let metal_includes = {|#include <metal_stdlib>
using namespace metal;|}

  let conditional_includes = [ ("simdgroup_load", "#include <metal_simdgroup_matrix>") ]

  (* gh-ocannl-1338: the kernel-name harvest's view of this backend ([tools/vendor_idents.ml]): the
     prelude with the conditional includes ahead of it, compiled for the first device under the
     options a link compiles with. *)
  let routine_name_probe () =
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = [||]
    end))
    in
    let prelude =
      String.concat ~sep:"\n" (List.map conditional_includes ~f:snd) ^ "\n" ^ metal_includes
    in
    let device = (get_device ~ordinal:0).dev in
    let options = metal_compile_options (metal_option_state ()) in
    let compile source =
      match Me.Library.on_device device ~source options with
      | _ -> Ok ()
      | exception Failure msg -> Error msg
    in
    Syntax.routine_name_probe ~prelude ~compile

  (* Metal compiles at link time, when a device is available: the rendering is retained, so the link
     can compile any subset of the kernels (gh-ocannl-1605). [artifact_name] is the name the link's
     [.metal] artifact takes: the kernel's own, or the common prefix of a batch's names -- fissioned
     segments of one routine share the routine name as a prefix, and a fixed name would collide
     across routines. *)
  let render ~batch ~names bindings lowereds =
    (* Whether or not the cache is enabled: a test gated on this flag must then see reuse fail. *)
    Compile_cost.record_reuse_capable ();
    let r =
      Compile.render_batch ~names bindings lowereds ~includes:metal_includes
        ~builtins:Builtins_metal.builtins ~conditional_includes ()
    in
    let n = Array.length r.entries in
    (* Rendered here; compiled where Metal's compiler runs, at link ([compile_metal_source]). *)
    Compile_cost.record_kernels ~rendered:n ~compiled:0;
    let symbols = Array.map r.entries ~f:(fun (_, name, _) -> name) in
    let artifact_name =
      if batch then
        let prefix =
          String.(strip ~drop:(equal_char '_') @@ common_prefix (Array.to_list symbols))
        in
        if String.is_empty prefix then "batch" else prefix
      else symbols.(0)
    in
    ( r.entries,
      {
        artifact_name;
        source = r.assemble (List.init n ~f:Fn.id);
        symbols;
        given_names = names;
        assemble = r.assemble;
        key_text = r.key_text;
        resolved = ref None;
      } )

  let compile ~name bindings lowered =
    let entries, kernels = render ~batch:false ~names:[| name |] bindings [| lowered |] in
    let kparams, func_name, launch = entries.(0) in
    { kernels; func_name; kparams; bindings; launch }

  let compile_batch ~names bindings lowereds =
    let entries, batch_kernels = render ~batch:true ~names bindings lowereds in
    let funcs = Array.map entries ~f:(fun (kparams, name, launch) -> (name, kparams, launch)) in
    { batch_kernels; funcs; bindings }

  (* {2 The kernel code cache (gh-ocannl-1605)}

     The other backends' cache (gh-ocannl-1314, gh-ocannl-1529, gh-ocannl-1525), at Metal's link: a
     search candidate re-renders the whole routine and typically changes a handful of its kernels,
     so a link compiles only the kernels no earlier link in this process produced -- its cache
     misses -- as ONE library, and reuses the rest. Unlike the others, a Metal link spends most of
     its time after the library compile, creating each kernel's compute pipeline state (about 0.4 s
     of a 1.8 s [gpt2_mini_train] candidate compile was the library), so what the cache holds is a
     [program]: the library together with the pipelines its kernels' links created. A reused kernel
     skips both.

     A kernel's key is its canonical standalone source ([render_batch]'s [key_text i]: the includes
     and the builtins it uses verbatim, its definition with the names it owns renamed by first
     occurrence and its segment index dropped, [Kernel_code_cache.canonical_kernel]), the complete
     Metal compile option state the link applies ([metal_option_state]: language version, math mode,
     math API, logging) and the device, whose library and pipelines are its own. The reused library
     was compiled from the same kernel text up to that renaming, possibly beside other kernels and
     possibly under another name: the builtins are inline helpers and constants, so a companion
     changes nothing a kernel computes, and the function is looked up by the symbol the cache stored
     with it. A pipeline state depends only on its function and device (no descriptor, no function
     constants), and launch geometry is a dispatch argument: [link_proc] checks the current launch
     against the reused pipeline on every link. A pipeline creation that fails caches nothing.

     Process-local and bounded by [metal_kernel_cache_mib], weighted by the source bytes of each
     compiled library -- a proxy: Metal exposes neither a library's nor a pipeline's size, so it is
     no bound on driver memory. 0 disables it, and so does device debugging, where a GPU capture
     shows a library's source and a reused kernel's would be another routine's text: every link then
     compiles one library of all its kernels, as before. *)
  let kernel_cache_capacity_bytes =
    lazy
      (let mib = Utils.get_global_arg ~default:"64" ~arg_name:"metal_kernel_cache_mib" in
       match Int.of_string_opt (String.strip mib) with
       | Some m when m >= 0 -> m * 1024 * 1024
       | _ ->
           raise
           @@ Utils.User_error
                ("metal_kernel_cache_mib must be a non-negative integer (MiB); found: " ^ mib))

  let kernel_cache : (program * string) Kernel_code_cache.t =
    Kernel_code_cache.create ~capacity_bytes:(fun () -> Lazy.force kernel_cache_capacity_bytes)

  (* Held from the cache lookups to the publication of what they missed, so two domains linking the
     same kernels cannot both miss and both compile them. *)
  let kernel_compile_lock = Stdlib.Mutex.create ()

  (* Removes the [<name>.part<k>.metal] artifacts an earlier link of this name left, which are not
     this link's. [.part<k>] is a spelling no routine name takes. *)
  let clear_part_artifacts name =
    if Utils.settings.output_debug_files_in_build_directory then
      let dir = Stdlib.Filename.dirname (Utils.build_file (name ^ ".metal")) in
      try
        Array.iter (Stdlib.Sys.readdir dir) ~f:(fun file ->
            if
              String.is_prefix file ~prefix:(name ^ ".part")
              && String.is_suffix file ~suffix:".metal"
            then try Stdlib.Sys.remove (Stdlib.Filename.concat dir file) with Sys_error _ -> ())
      with Sys_error _ -> ()

  (* Every kernel of [k] as a program on [device] and the symbol to look it up by: from the kernel
     code cache where an earlier link produced it, the misses compiled as one library. *)
  let resolve_kernels_afresh (device : device) (k : kernels) =
    let n = Array.length k.symbols in
    let all = List.init n ~f:Fn.id in
    let option_state = metal_option_state () in
    (* The routine's full source is its [.metal] artifact whatever is compiled: the debug view of
       what codegen emitted (and what tests read) does not depend on what the cache held. *)
    let metal_file = write_source_artifact ~name:k.artifact_name k.source in
    (* One library of [idxs]. Every kernel in batch order is the routine's own source, already
       written; any other subset is written as [<name>.part<k>.metal], under the name its
       diagnostics then carry. *)
    let compile_group ~group ~n_groups idxs =
      let whole = n_groups = 1 && List.equal Int.equal idxs all in
      let name, source, metal_file =
        if whole then (k.artifact_name, k.source, metal_file)
        else
          let name = k.artifact_name ^ ".part" ^ Int.to_string group in
          let source = k.assemble idxs in
          (name, source, write_source_artifact ~name source)
      in
      let routine_names = List.map idxs ~f:(fun i -> (k.symbols.(i), k.given_names.(i))) in
      let library =
        compile_metal_source ~name ~routine_names ~source ~metal_file ~option_state
          ~device:device.dev
      in
      (new_program library, String.length source)
    in
    if not (Kernel_code_cache.enabled kernel_cache && not (Utils.with_runtime_debug ())) then
      Stdlib.Mutex.protect kernel_compile_lock (fun () ->
          clear_part_artifacts k.artifact_name;
          let program, _ = compile_group ~group:0 ~n_groups:1 all in
          Array.map k.symbols ~f:(fun symbol -> (program, symbol)))
    else
      let registry_id = (Me.Device.get_attributes device.dev).registry_id in
      let options =
        [
          Compiler_options.render_metal option_state;
          [%string
            "metal device %{device.ordinal#Int} registry %{Unsigned.ULLong.to_string registry_id}"];
        ]
      in
      let options = Array.create ~len:n options in
      let keys =
        Array.init n ~f:(fun i -> Kernel_code_cache.key ~parts:(options.(i) @ [ k.key_text i ]))
      in
      Kernel_code_cache.resolve kernel_cache ~lock:kernel_compile_lock ~keys ~options
        ~symbols:k.symbols ~always_lock:Utils.settings.output_debug_files_in_build_directory
        ~before_compile:(fun ~n_compiled:_ ~n_groups:_ -> clear_part_artifacts k.artifact_name)
        ~compile_group:(fun ~group ~n_groups _options idxs -> compile_group ~group ~n_groups idxs)
        ()

  (* At most once per code value and device ([resolved_memo]). Counted as a device compile at link
     time whether its kernels were compiled or taken from the cache, as an all-hit compile counts on
     the other backends: what [Context.backend_compiles] counts is the work a relink skips
     (gh-ocannl-1440). *)
  let resolve_kernels (device : device) (k : kernels) =
    match !(k.resolved) with
    | Some (d, resolved) when phys_equal d device.dev -> resolved
    | _ ->
        let resolved = resolve_kernels_afresh device k in
        Utils.count_backend_compile ();
        k.resolved := Some (device.dev, resolved);
        resolved

  (* gh-ocannl-344: from the routine's materialized nodes, assign each distinct pool an index (first
     use order), build the [(pool_index, byte_offset)] slot table (one [uint] pair per node,
     matching the kernel prologue order), and return [index_to_pool] for binding [__pool<i>]. Raises
     if the routine touches more than [metal_max_pools] distinct pools. The returned [CArray] is
     kept alive by the caller so its backing store outlives the buffer copy. *)
  let build_pool_binding (dev : device) (ctx_buffers : ctx_buffers) (tns : Tn.t list) =
    let tbl = Hashtbl.create (module Int) and index_to_pool = ref [] and next = ref 0 in
    let idx_of pool_id =
      match Hashtbl.find tbl pool_id with
      | Some i -> i
      | None ->
          let i = !next in
          Int.incr next;
          Hashtbl.set tbl ~key:pool_id ~data:i;
          index_to_pool := pool_id :: !index_to_pool;
          i
    in
    (* (pool_index, byte_offset) per node, in the slot order the shader prologue indexes. *)
    let pairs =
      List.map tns ~f:(fun tn ->
          let loc = Map.find_exn ctx_buffers tn in
          (idx_of loc.pool_id, loc.offset))
    in
    let max_pools = Option.value_exn (hardware_limits ()).Backend_intf.max_bound_pools in
    if !next > max_pools then
      raise
      @@ Utils.User_error
           (Printf.sprintf
              "Metal_backend: routine needs %d distinct pools, over the metal_max_pools=%d binding \
               budget. Upload or copy absent nodes into one destination context before compiling; \
               otherwise reduce the number of independently allocated pools the routine reads."
              !next max_pools);
    (* The slot-table element width must match the MSL type [C_syntax.pool_slot_msl_typ] declared in
       the shader: 64-bit under [large_models] (offsets may exceed UINT32_MAX once the 4 GB per-pool
       cap is lifted), else 32-bit. [keep] holds the backing [CArray] so it outlives the buffer
       copy. *)
    let slots_buf, keep =
      let make (type a) (elem : a Ctypes.typ) (of_int : int -> a) =
        let vals = List.concat_map pairs ~f:(fun (p, o) -> [ of_int p; of_int o ]) in
        let arr = Ctypes.CArray.of_list elem vals in
        let buf =
          Me.Buffer.on_device_with_bytes dev.dev
            ~bytes:Ctypes.(to_voidp (CArray.start arr))
            ~length:(List.length vals * Ctypes.sizeof elem)
            shared_resource_options
        in
        (buf, fun () -> ignore (Sys.opaque_identity arr))
      in
      if C_syntax.pool_slot_is_64 () then make Ctypes.ullong Unsigned.ULLong.of_int
      else make Ctypes.uint Unsigned.UInt.of_int
    in
    (Array.of_list (List.rev !index_to_pool), slots_buf, keep)

  (* Metal command buffers on one queue may begin executing while earlier ones are still running
     (untracked resources), and a routine's task only waits for the input events captured at link
     time — so back-to-back runs of one routine (e.g. training steps that read the loss only once
     per epoch, via [Train.grad_update ?accum_loss]) would race with themselves. Encode a wait for
     the latest all-work signal into the launch's own command buffer: every routine run is followed
     by an [all_work] signal ([Raise_backend.sync_routine]), so this orders the launch after all
     previously enqueued routine work at no extra command buffer — the FIFO-execution semantics CUDA
     streams have natively. No-op on a fresh queue: [all_work] signals [counter + 1] and stores it,
     so the latest signaled value is the current counter, and counter = 1 means no signal was issued
     yet. The wait has to be encoded BEFORE the compute pass, which is where the call below sits:
     [encodeWaitForEvent] orders only what follows it in the buffer, and a wait appended after an
     encoded kernel orders nothing (the shape gh-ocannl-828's probe measured overlapping,
     gh-ocannl-909). Executed pin: [test/operations/back_to_back_runs.ml], a self-incrementing
     routine run back-to-back with no host sync must count every run. *)
  let encode_wait_for_enqueued dev command_buffer =
    let counter = dev.runner.counter in
    if Unsigned.ULLong.compare counter Unsigned.ULLong.one > 0 then
      Me.CommandBuffer.encode_wait_for_event command_buffer
        (Me.SharedEvent.super dev.runner.event)
        counter

  (* Post-link validation (gh-ocannl-536 landing step 3, the portable half): the pre-compile
     [Schedule.check_hardware_limits] judges the schedule against DEVICE-wide limits, but the
     compiled pipeline can be stricter — [maxTotalThreadsPerThreadgroup] drops below the device
     width under register pressure — and knows its own threadgroup-memory footprint. These are typed
     [Resource_exceeded] causes rather than a bare [Utils.User_error]: to a tuner candidate they are
     an ordinary decline (and their key is what the blocker census tabulates), while a hand-written
     schedule still gets the same [Utils.User_error] rendering at the public [Context.compile]
     boundary. Untyped, strict classification would make them fatal and one register-heavy candidate
     would end the search. *)
  let reject_link resource ~requested ~limit detail =
    raise
    @@ Schedule_outcome.Cause_at
         ( Schedule_outcome.Backend_link,
           Schedule_outcome.Resource_exceeded { resource; requested; limit = Some limit; detail } )

  (* The function and compute pipeline state of the kernel [program] holds under [symbol], created
     on its first use and kept with the program, so every later link of the kernel -- this code
     value's or, through the kernel code cache, another's -- reuses them (gh-ocannl-1605).
     [func_name] is the kernel's current name, for diagnostics. A creation that fails keeps
     nothing. *)
  let pipeline_of (program : program) ~metal_device ~func_name symbol =
    Stdlib.Mutex.protect program.pipelines_lock (fun () ->
        match Hashtbl.find program.pipelines symbol with
        | Some func_and_pso -> func_and_pso
        | None ->
            let func = Me.Library.new_function_with_name program.library symbol in
            (* Over-limit threadgroup memory never reaches a pipeline to query: Metal refuses to
               create one, and the binding reports it as a bare [Failure] carrying the driver's
               sentence. It is typed here, at the same cause as [link_proc]'s post-link check, with
               the driver's own byte counts (gh-ocannl-1226): the driver applies per-allocation
               alignment the schedule-level estimate does not (each [threadgroup] array is rounded
               up to 4 bytes, so two odd-length [half] tiles summing exactly to the limit are
               refused), so an estimate at the limit can still be over it here. Any other
               pipeline-creation failure keeps propagating untyped: nothing here establishes that it
               is the candidate's rather than the environment's. *)
            let pso, _ =
              try Me.ComputePipelineState.on_device_with_function metal_device func
              with Failure msg as exn -> (
                let backtrace = Stdlib.Printexc.get_raw_backtrace () in
                let overflow =
                  Option.bind (String.substr_index msg ~pattern:"Threadgroup memory size (")
                    ~f:(fun pos ->
                      try
                        Some
                          (Stdlib.Scanf.sscanf (String.drop_prefix msg pos)
                             "Threadgroup memory size (%d) exceeds the maximum threadgroup memory \
                              allowed (%d)" (fun requested limit -> (requested, limit)))
                      with Stdlib.Scanf.Scan_failure _ | End_of_file | Failure _ -> None)
                in
                match overflow with
                | Some (requested, limit) ->
                    reject_link Schedule_outcome.Workgroup_memory ~requested ~limit
                      [%string
                        "Metal: pipeline creation for %{func_name} refused its threadgroup memory: \
                         %{requested#Int} bytes requested, over the device limit of %{limit#Int} \
                         bytes (driver: %{msg})"]
                | None -> Stdlib.Printexc.raise_with_backtrace exn backtrace)
            in
            Compile_cost.record_pipeline_created ();
            Hashtbl.set program.pipelines ~key:symbol ~data:(func, pso);
            (func, pso))

  let%debug4_sexp link_proc ~prior_context ~(program : program) ~symbol ~func_name
      ~(kparams : (string * kparam_source) list) ~(launch : Low_level.launch_dims) ~lowered_bindings
      ~(ctx_buffers : ctx_buffers) : Task.t =
    let dev = prior_context.device in
    let metal_device = dev.dev in
    let queue = dev.runner.queue in
    let runner_label = get_name dev in
    let _func, pso = pipeline_of program ~metal_device ~func_name symbol in
    let block_product = launch.block.(0) * launch.block.(1) * launch.block.(2) in
    let max_threads = Me.ComputePipelineState.get_max_total_threads_per_threadgroup pso in
    if block_product > max_threads then
      reject_link Schedule_outcome.Workgroup_threads ~requested:block_product ~limit:max_threads
        [%string
          "Metal: threadgroup size %{block_product#Int} for %{func_name} exceeds \
           maxTotalThreadsPerThreadgroup %{max_threads#Int}"];
    (* The compiled kernel's own static threadgroup allocation, which the schedule-level accounting
       (a sum over the staged tnodes) only approximates. A backstop, not the primary check: a kernel
       whose allocation is over the device limit never gets here, because pipeline creation
       ([pipeline_of]) already refused it (gh-ocannl-1226). What remains is a pipeline the driver
       accepted yet reports as over the limit the schedule layer used. *)
    Option.iter (hardware_limits ()).Backend_intf.max_workgroup_memory_bytes ~f:(fun max_bytes ->
        let static_bytes = Me.ComputePipelineState.get_static_threadgroup_memory_length pso in
        if static_bytes > max_bytes then
          reject_link Schedule_outcome.Workgroup_memory ~requested:static_bytes ~limit:max_bytes
            [%string
              "Metal: compiled kernel %{func_name} statically allocates %{static_bytes#Int} bytes \
               of threadgroup memory, exceeding the device limit of %{max_bytes#Int} bytes"]);
    (* Precompute the pool-index assignment + slot table once (addresses/offsets are stable after
       allocation). [Some (index_to_pool, slots_buf, arr)] iff this routine uses the pooled
       params. *)
    let pool_binding =
      List.find_map kparams ~f:(function _, Kparam_pool_slots tns -> Some tns | _ -> None)
      |> Option.map ~f:(build_pool_binding dev ctx_buffers)
    in

    let work () : unit =
      [%log3_result "Launching", func_name, "on", runner_label];
      (* Unlike CUDA, we don't use Utils.add_log_processor here. Logs are captured C-side by the
         buffered LogState handler installed on the CommandQueue, and [await] drains them to
         [Utils.log_debug_routine_logs]. *)
      try
        (* Inside a [sequence_segments] batch, encode into the routine's open serial compute pass
           (one command buffer for all segments); standalone, commit our own command buffer. *)
        let fused = dev.runner.fused in
        let command_buffer, encoder =
          match fused with
          | Some (cb, enc) -> (cb, enc)
          | None ->
              let cb = Me.CommandBuffer.on_queue queue in
              encode_wait_for_enqueued dev cb;
              (cb, Me.ComputeCommandEncoder.on_buffer cb)
        in
        Me.ComputeCommandEncoder.set_compute_pipeline_state encoder pso;

        (* Set arguments *)
        List.iteri kparams ~f:(fun index (_p_name, p_source) ->
            match p_source with
            | Kparam_ptr tn when Map.mem ctx_buffers tn ->
                (* gh-ocannl-344: the tnode is a region of a (possibly multi-tenant) pool slab; bind
                   the slab buffer at the tnode's byte offset so the kernel pointer parameter points
                   at this tnode, not the pool base. *)
                let loc = Map.find_exn ctx_buffers tn in
                Me.ComputeCommandEncoder.set_buffer encoder ~offset:loc.offset ~index
                  (Slab.resolve_pool dev loc)
            | Kparam_ptr tn ->
                (* After gh-ocannl-333 there is no host array to wrap as a constant buffer: every
                   in-context node (including constants) is allocated in [ctx_buffers] by the
                   allocator, which uploads any host initialization data there. *)
                failwith
                  [%string
                    "Kparam_ptr %{Tn.debug_name tn} not found in ctx_buffers for %{func_name}"]
            | Kparam_pool_slab i ->
                (* Bind pool param [i] to the slab assigned index [i]; the unused tail (i >= number
                   of distinct pools) duplicates pool 0 so the shader's local pool array is fully
                   bound without extra bindings. Pools are bound at offset 0 -- the per-node byte
                   offset is applied in-shader via the slot table. *)
                let index_to_pool, _, _ = Option.value_exn ~here:[%here] pool_binding in
                let pool_id =
                  if i < Array.length index_to_pool then index_to_pool.(i) else index_to_pool.(0)
                in
                Me.ComputeCommandEncoder.set_buffer encoder ~index
                  (Slab.resolve_pool dev { pool_id; offset = 0 })
            | Kparam_pool_slots _ ->
                let _, slots_buf, _ = Option.value_exn ~here:[%here] pool_binding in
                Me.ComputeCommandEncoder.set_buffer encoder ~index slots_buf
            | Static_idx s ->
                let value = !(Indexing.find_exn lowered_bindings s) in
                (* Bind-time validation (docs/proposals/signed-index-precision.md), mirroring the
                   CUDA backend's launch-side checks. *)
                Indexing.validate_bound_value ~width64:Utils.settings.large_models s value;
                let size = Ctypes.sizeof Ctypes.int in
                let bytes_ptr = Ctypes.(allocate int value |> to_voidp) in
                Me.ComputeCommandEncoder.set_bytes encoder ~bytes:bytes_ptr ~length:size ~index
            | Merge_buffer -> (
                match !(dev.merge_buffer) with
                | Some loc ->
                    Me.ComputeCommandEncoder.set_buffer encoder ~offset:loc.offset ~index
                      (Slab.resolve_pool dev loc)
                | None -> failwith [%string "Merge_buffer requested but not set for %{func_name}"])
            | Log_file_name ->
                (* TODO:We could tag logs with a run id. *)
                assert false);

        (* Launch dimensions derived from hardware-annotated loops (axis-types proposal §4);
           all-Serial kernels dispatch one threadgroup of one thread, as before. *)
        Me.ComputeCommandEncoder.dispatch_threadgroups encoder
          ~threadgroups_per_grid:
            { width = launch.grid.(0); height = launch.grid.(1); depth = launch.grid.(2) }
          ~threads_per_threadgroup:
            { width = launch.block.(0); height = launch.block.(1); depth = launch.block.(2) };

        match fused with
        | Some _ -> () (* [sequence_segments] ends the pass and commits once, after all segments. *)
        | None ->
            Me.ComputeCommandEncoder.end_encoding encoder;
            Me.CommandBuffer.commit command_buffer
        (* Make execution synchronous for debugging/simplicity, remove later *)
        (* Me.CommandBuffer.wait_until_completed command_buffer; *)
      with exn ->
        [%log "Exception during kernel launch:", (func_name : string), (exn : exn)];
        (* Raise after logging *)
        raise exn
    in
    Task.Task
      {
        context_lifetime = (program, pso, ctx_buffers, pool_binding);
        (* Keep the library, the PSO and the slot-table buffer alive *)
        description = "launches " ^ func_name ^ " on " ^ runner_label;
        work;
      }

  let link prior_context (code : code) ctx_buffers =
    let program, symbol = (resolve_kernels prior_context.device code.kernels).(0) in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols code.bindings) ~f:(fun s ->
          (s, Indexing.fresh_launch_param ()))
    in
    let task =
      link_proc ~prior_context ~program ~symbol ~func_name:code.func_name ~kparams:code.kparams
        ~launch:code.launch ~lowered_bindings ~ctx_buffers
    in
    (lowered_bindings, task)

  let link_batch prior_context code_batch ctx_buffers =
    let resolved = resolve_kernels prior_context.device code_batch.batch_kernels in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols code_batch.bindings) ~f:(fun s ->
          (s, Indexing.fresh_launch_param ()))
    in
    let tasks =
      Array.map2_exn code_batch.funcs resolved
        ~f:(fun (func_name, kparams, launch) (program, symbol) ->
          link_proc ~prior_context ~program ~symbol ~func_name ~kparams ~launch ~lowered_bindings
            ~ctx_buffers)
    in
    (lowered_bindings, tasks)

  (* One command buffer for a fissioned routine's whole segment batch: a [Serial]-dispatch compute
     pass executes its dispatches in encoding order — MTLDispatchType.serial semantics, independent
     of resource hazard tracking — which is exactly the grid-wide synchronization each fission
     boundary needs. This replaces per-segment command buffers plus two event command buffers per
     boundary (the dominant per-step cost of fissioned training steps; each command buffer costs
     ~0.2-0.5 ms of pipeline latency). The tasks are this backend's [link_batch] kernel launches:
     while [runner.fused] is set, [link_proc]'s work encodes into the open pass instead of
     committing its own buffer. *)
  let sequence_segments (context : context) ~name ~bindings:_ ~uses_merge_buffer:_
      (tasks : Task.t list) : (Task.t * (unit -> unit)) option =
    let dev = context.device in
    (* The command buffer is built afresh every run, so nothing outlives a run to release. *)
    Some
      ( Task.Task
          {
            context_lifetime = tasks;
            description = "fused segments of " ^ name ^ " on " ^ get_name dev;
            work =
              (fun () ->
                let command_buffer = Me.CommandBuffer.on_queue dev.runner.queue in
                encode_wait_for_enqueued dev command_buffer;
                let encoder =
                  Me.ComputeCommandEncoder.on_buffer_with_dispatch_type command_buffer
                    Me.ComputeCommandEncoder.DispatchType.Serial
                in
                dev.runner.fused <- Some (command_buffer, encoder);
                Exn.protect
                  ~f:(fun () -> List.iter tasks ~f:Task.run)
                  ~finally:(fun () ->
                    dev.runner.fused <- None;
                    Me.ComputeCommandEncoder.end_encoding encoder;
                    Me.CommandBuffer.commit command_buffer));
          },
        fun () -> () )
end

(* gh-ocannl-1038: the one capability this backend probes on the host rather than on a device --
   which math API the running Objective-C runtime's compile options accept, the query [Impl]'s
   [runtime_math_api] makes before every compile. It moves with an OS update, and it decides the
   option sequence every Metal compile uses. *)
let host_capabilities () =
  [
    ( "metal_math_api",
      match Metal_math_api_runtime.get () with
      | Compiler_options.Modern_split -> "modern_split"
      | Compiler_options.Legacy -> "legacy" );
  ]

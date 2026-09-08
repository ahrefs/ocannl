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

  (* Store for captured logs per device_id (the device is its own single compute stream). *)
  let stream_logs : (int, string list ref) Hashtbl.t = Hashtbl.create (module Int)

  (* Device Management *)
  let num_devs () = Array.length (Lazy.force metal_devices)
  let devices_cache = lazy (Array.create ~len:(num_devs ()) None)

  (* Builds the device's single compute runner (command queue + sync event), optionally with a
     debug-log-capturing queue. Returns the runner and the captured-log ref (if logging is on). *)
  let spinup_runner metal_device =
    let queue =
      if Utils.debug_log_from_routines () then (
        let log_entries_ref = ref [] in
        (* This ref will be captured by the log handler *)
        let log_desc = Me.LogStateDescriptor.create () in
        Me.LogStateDescriptor.set_level log_desc Me.LogLevel.Debug;
        (* Capture all debug logs and above *)
        Me.LogStateDescriptor.set_buffer_size log_desc (1024 * 100);
        (* 100KB buffer *)
        let log_state = Me.LogState.on_device_with_descriptor metal_device log_desc in
        Me.LogState.add_log_handler log_state (fun ~sub_system:_ ~category:_ ~level:_ ~message ->
            log_entries_ref := message :: !log_entries_ref);
        let queue_desc = Me.CommandQueueDescriptor.create () in
        Me.CommandQueueDescriptor.set_log_state queue_desc (Some log_state);
        (* The log_state and its handler (capturing log_entries_ref) are kept alive by the
           queue_desc / queue itself. *)
        let created_q = Me.CommandQueue.on_device_with_descriptor metal_device queue_desc in
        (created_q, Some log_entries_ref))
      else (Me.CommandQueue.on_device metal_device, None)
    in
    let actual_queue, opt_log_entries_ref = queue in
    let shared_event_obj = Me.SharedEvent.on_device metal_device in
    let counter = Unsigned.ULLong.one in
    (* Next value = 1 *)
    ({ queue = actual_queue; event = shared_event_obj; counter; fused = None }, opt_log_entries_ref)

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
      let runner, opt_log_entries_ref = spinup_runner metal_device in
      let result_device = make_device metal_device runner ~ordinal in
      (* The device is its own single compute stream; key captured logs by [device_id]. *)
      Option.iter opt_log_entries_ref ~f:(fun log_ref ->
          Hashtbl.add_exn stream_logs ~key:result_device.device_id ~data:log_ref);
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

  let all_work device =
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
    { shared = shared_event; value = next_value }

  let await device =
    (* Signal an event after all current work and wait for it. This ensures all previously submitted
       command buffers complete. *)
    let event = all_work device in
    sync event;
    (* Process captured logs if any *)
    if Utils.debug_log_from_routines () then
      match Hashtbl.find stream_logs device.device_id with
      | Some log_entries_ref ->
          let logs_to_process = List.rev !log_entries_ref in
          if not (List.is_empty logs_to_process) then
            Utils.log_debug_routine_logs ~log_contents:logs_to_process
              ~stream_name:(get_name device);
          log_entries_ref := [] (* Clear processed logs *)
      | None -> () (* No log bucket for this device, logging likely not enabled for it *)

  let is_idle device =
    (* FIXME: store the latest CommandBuffer with the device runner and check that it's completed *)
    let counter = device.runner.counter in
    let current_signaled = Me.SharedEvent.get_signaled_value device.runner.event in
    let expected_signaled = Unsigned.ULLong.pred counter in
    Unsigned.ULLong.equal current_signaled expected_signaled

  (* --- Configuration and Info --- *)
  let get_used_memory _device = Atomic.get allocated_memory

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
  let hardware_limits =
    let limits =
      lazy
        (let min_over f =
           Array.map (Lazy.force metal_devices) ~f:(fun d -> f (Me.Device.get_attributes d))
           |> Array.min_elt ~compare:Int.compare
         in
         {
           Backend_intf.max_threads_per_workgroup =
             min_over (fun (a : Me.Device.attributes) -> a.max_threads_per_threadgroup.width);
           max_workgroup_memory_bytes =
             min_over (fun (a : Me.Device.attributes) ->
                 Unsigned.ULong.to_int a.max_threadgroup_memory_length);
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
           max_grid_yz = None;
           (* [simdgroup_matrix] (docs/proposals/tensorize-mma.md): 8×8×8 tiles cooperatively held
              by the 32-thread simdgroup, available on Apple7+ (M1 and later). Supported storage
              precisions are decided per call by [mma_syntax] (f32/f16/bf16, uniform); under
              [Fp16_wide], the uniform-f16 storage arm uses an f32 accumulator fragment and converts
              only at the destination boundary (gh-ocannl-837). *)
           mma =
             (if
                (* [for_all] is vacuously true on no devices, which would advertise simdgroup
                   matrices on a machine that has no Metal device at all. Unreachable in practice
                   (compiling needs a device), but it used to be impossible by the discovery
                   assertion, so it is stated rather than assumed. *)
                (not (Array.is_empty (Lazy.force metal_devices)))
                && Array.for_all (Lazy.force metal_devices) ~f:(fun d ->
                    Me.Device.supports_family d Me.Device.GPUFamily.Apple7)
              then
                Some
                  {
                    Backend_intf.mma_simd_width = 32;
                    mma_tile = (8, 8, 8);
                    (* These triples describe storage formats. The wide-f16 exception below uses the
                       same f16 storage triple with an f32 accumulator fragment internally. *)
                    mma_format_tiles =
                      [
                        ( (Backend_intf.Mma_f32, Backend_intf.Mma_f32, Backend_intf.Mma_f32),
                          (8, 8, 8) );
                        ( (Backend_intf.Mma_f16, Backend_intf.Mma_f16, Backend_intf.Mma_f16),
                          (8, 8, 8) );
                        ( (Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_bf16),
                          (8, 8, 8) );
                      ];
                    (* MSL's generic [simdgroup_multiply_accumulate] accepts half A/B fragments and
                       an f32 accumulator. [mma_syntax]/[mma_fragment_syntax] bridge that
                       accumulator to f16 destination storage elementwise at the outer boundary
                       (gh-ocannl-837). *)
                    mma_f16_wide_acc_scopes =
                      [ Backend_intf.Mma_per_statement; Backend_intf.Mma_fragment_scope ];
                    (* Metal banks too, but [simdgroup_load] takes a plain pointer and leading
                       dimension — no [ldmatrix] analogue (gh-ocannl-481 item 3, D3). *)
                    mma_staged_layouts = [];
                    mma_pipeline_depths = [ 2 ];
                  }
              else None);
           simd_vector_bytes = 0;
           native_fp16_arithmetic = false;
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
         })
    in
    fun () -> Lazy.force limits

  let get_global_debug_info () = Sexp.Atom "Metal global debug info NYI"

  let get_debug_info device =
    let num_pending_logs =
      match Hashtbl.find stream_logs device.device_id with None -> 0 | Some r -> List.length !r
    in
    Sexp.message "Metal device debug info"
      [
        ("device_id", sexp_of_int device.device_id);
        ("pending_shader_logs", sexp_of_int num_pending_logs);
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
  type code = {
    metal_source : string; (* Store source, the library is built during link *)
    func_name : string;
    kparams : (string * kparam_source) list;
    bindings : Indexing.unit_bindings;
    launch : Low_level.launch_dims;
  }
  [@@deriving sexp_of]

  type code_batch = {
    metal_source : string; (* Store combined source, compiled per device at link time *)
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

    let barrier_syntax = Some "threadgroup_barrier(mem_flags::mem_threadgroup);"
    let shared_decl_prefix = Some "threadgroup "

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
    let vector_bytes = 16
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

    let mma_fragment_types ~d_prec ~a_prec ~b_prec =
      if not (Ops.equal_prec d_prec a_prec && Ops.equal_prec d_prec b_prec) then None
      else
        match d_prec with
        | Ops.Single_prec _ ->
            Some
              ( "simdgroup_float8x8",
                "simdgroup_float8x8",
                "simdgroup_float8x8",
                "simdgroup_float8x8" )
        | Ops.Half_prec _ when Numerics.fp16_accum_wide () ->
            Some
              ("simdgroup_float8x8", "simdgroup_half8x8", "simdgroup_half8x8", "simdgroup_half8x8")
        | Ops.Half_prec _ ->
            Some ("simdgroup_half8x8", "simdgroup_half8x8", "simdgroup_half8x8", "simdgroup_half8x8")
        | Ops.Bfloat16_prec _ ->
            Some
              ( "simdgroup_bfloat8x8",
                "simdgroup_bfloat8x8",
                "simdgroup_bfloat8x8",
                "simdgroup_bfloat8x8" )
        | _ -> None

    (* MSL exposes the distributed fragment elements through [thread_elements()]. On a 32-thread
       simdgroup each lane owns two elements of an 8x8 matrix, and the mapping depends only on the
       matrix dimensions, not its element type. A standalone runtime-compiler probe on an Apple M4
       Max established the half/f32 mixed multiply surface and this elementwise boundary
       (gh-ocannl-837). The assertion keeps the logical element-count premise adjacent to the copy;
       unlike a whole-[storage_type] conversion, the scalar spelling is accepted by current MSL. *)
    let mma_d_boundary_lines ~dir ~acc_frag ~d_frag ~acc ~ptr ~ldd =
      if String.equal acc_frag d_frag then
        [
          (match dir with
          | `Load -> Printf.sprintf "simdgroup_load(%s, %s, (ulong)%d);" acc ptr ldd
          | `Store -> Printf.sprintf "simdgroup_store(%s, %s, (ulong)%d);" acc ptr ldd);
        ]
      else
        let copy dst src cast =
          [
            Printf.sprintf "%s.thread_elements()[0] = (%s)%s.thread_elements()[0];" dst cast src;
            Printf.sprintf "%s.thread_elements()[1] = (%s)%s.thread_elements()[1];" dst cast src;
          ]
        in
        let assertion =
          "static_assert(sizeof(simdgroup_float8x8::storage_type) / sizeof(float) == "
          ^ "sizeof(simdgroup_half8x8::storage_type) / sizeof(half), "
          ^ "\"wide-f16 d boundary requires equal fragment element counts\");"
        in
        match dir with
        | `Load ->
            [ Printf.sprintf "%s __mma_dstage;" d_frag; assertion ]
            @ [ Printf.sprintf "simdgroup_load(__mma_dstage, %s, (ulong)%d);" ptr ldd ]
            @ copy acc "__mma_dstage" "float"
        | `Store ->
            [ Printf.sprintf "%s __mma_dstage;" d_frag; assertion ]
            @ copy "__mma_dstage" acc "half"
            @ [ Printf.sprintf "simdgroup_store(__mma_dstage, %s, (ulong)%d);" ptr ldd ]

    (* Cooperative tile-MMA emission for [Low_level.Tile_mma] via MSL [simdgroup_matrix]
       (docs/proposals/tensorize-mma.md §4): fragment blocks of 8×8 tiles held jointly by the
       32-thread simdgroup, loaded with [simdgroup_load] straight from [device] memory or
       [threadgroup] tiles (i.e. out of [Stage]'s shared staging), accumulated with
       [simdgroup_multiply_accumulate], stored once at the end — the accumulator fragments are
       resident across the whole [k] extent of the block statement. Transposed-stored operands
       ([ta]/[tb]) load with [simdgroup_load]'s [transpose_matrix] flag and swapped tile-offset
       arithmetic. Declines (fallback path) on: non-{f32,f16,bf16} precision, extents not multiples
       of 8, thread-space (stack-array) operands (not loadable by [simdgroup_load]), and devices
       below the Apple7 family. *)
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
          let tile = 8 in
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
          (* Tile load at role-space block (row_i, col_i): transposed storage keeps the operand's
             own leading dimension and indexes the STORED matrix at (col_i, row_i);
             [simdgroup_load]'s [transpose_matrix] flag transposes the 8×8 tile in flight. *)
          let load_line frag ptr ~row_i ~col_i ld ~transpose =
            if transpose then
              Printf.sprintf
                "    simdgroup_load(%s, %s + %s * %d * %d + %s * %d, (ulong)%d, ulong2(0), true);"
                frag ptr col_i tile ld row_i tile ld
            else
              Printf.sprintf "    simdgroup_load(%s, %s + %s * %d * %d + %s * %d, (ulong)%d);" frag
                ptr row_i tile ld col_i tile ld
          in
          match (frag_types, d_space, addr_space a_space, addr_space b_space) with
          | Some (_acc_frag, a_frag, b_frag, _d_frag), `Fragment fragment, Some a_as, Some b_as
            when m % tile = 0
                 && n % tile = 0
                 && k % tile = 0
                 && Option.is_some (hardware_limits ()).Backend_intf.mma ->
              let mt = m / tile and nt = n / tile and kt = k / tile in
              let ptr_decl name aspace elem ptr =
                string (Printf.sprintf "%s %s *%s = " aspace elem name) ^^ ptr ^^ semi
              in
              let body_lines =
                [
                  Printf.sprintf "for (int __ki = 0; __ki < %d; ++__ki) {" kt;
                  Printf.sprintf "  %s __mma_bf[%d];" b_frag nt;
                  Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                  load_line "__mma_bf[__ni]" "__mma_bp" ~row_i:"__ki" ~col_i:"__ni" ldb
                    ~transpose:tb;
                  "  }";
                  Printf.sprintf "  for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                  Printf.sprintf "    %s __mma_af;" a_frag;
                  load_line "__mma_af" "__mma_ap" ~row_i:"__mi" ~col_i:"__ki" lda ~transpose:ta;
                  Printf.sprintf "    for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                  Printf.sprintf
                    "      simdgroup_multiply_accumulate(%s[__mi][__ni], __mma_af, __mma_bf[__ni], \
                     %s[__mi][__ni]);"
                    fragment fragment;
                  "    }";
                  "  }";
                  "}";
                ]
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
                  let mt = m / tile and nt = n / tile and kt = k / tile in
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
                    @ [
                        "  }";
                        "}";
                        Printf.sprintf "for (int __ki = 0; __ki < %d; ++__ki) {" kt;
                        Printf.sprintf "  %s __mma_bf[%d];" b_frag nt;
                        Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                        load_line "__mma_bf[__ni]" "__mma_bp" ~row_i:"__ki" ~col_i:"__ni" ldb
                          ~transpose:tb;
                        "  }";
                        Printf.sprintf "  for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                        Printf.sprintf "    %s __mma_af;" a_frag;
                        load_line "__mma_af" "__mma_ap" ~row_i:"__mi" ~col_i:"__ki" lda
                          ~transpose:ta;
                        Printf.sprintf "    for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                        "      simdgroup_multiply_accumulate(__mma_acc[__mi][__ni], __mma_af, \
                         __mma_bf[__ni], __mma_acc[__mi][__ni]);";
                        "    }";
                        "  }";
                        "}";
                      ]
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
          let tile = 8 in
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
      | Mod, _ -> math "fmod"
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
      match (op, prec) with
      | Ops.Identity, _ -> fun v -> v
      | Neg, _ -> fun v -> string "-" ^^ v
      | Exp, _ -> math_doc "exp"
      | Log, _ -> math_doc "log"
      | Exp2, _ -> math_doc "exp2"
      | Log2, _ -> math_doc "log2"
      | Sin, _ -> math_doc "sin"
      | Cos, _ -> math_doc "cos"
      | Sqrt, _ -> math_doc "sqrt"
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
      | Tanh_approx, _ -> math_doc "tanh"
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

  let%diagn_sexp compile_metal_source ~name ~source ~device =
    let options = Me.CompileOptions.init () in
    let math_api = Lazy.force runtime_math_api in
    let option_state =
      Compiler_options.metal ~routine_logging:(Utils.debug_log_from_routines ()) ~math_api
    in
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
    [%log "metal options", (Compiler_options.render_metal option_state : string)];

    if Utils.settings.output_debug_files_in_build_directory then (
      let metal_file = Utils.build_file (name ^ ".metal") in
      Stdio.Out_channel.write_all metal_file ~data:source;
      [%log "Wrote metal source to file:", metal_file]);

    try Me.Library.on_device device ~source options
    with Failure msg ->
      let error_msg =
        Printf.sprintf "Metal compilation failed for %s:\n%s\nmetal options: %s\nSource:\n%s" name
          msg
          (Compiler_options.render_metal option_state)
          source
      in
      Stdio.prerr_endline error_msg;
      failwith error_msg

  (* The simdgroup-matrix types and functions are declared in <metal_simdgroup_matrix>.
     <metal_stdlib> is the umbrella header and pulls them in on current toolchains (the tensorized
     parity tests compile and run against it alone), but the dedicated header is the documented
     home, so inject it into kernels that emit the intrinsics — and only those, keeping every other
     kernel's source byte-identical (PR #101 review; mirrors [cuda_to_ptx]'s <mma.h> injection). *)
  let maybe_include_simdgroup_matrix source =
    if String.is_substring source ~substring:"simdgroup_load" then
      "#include <metal_simdgroup_matrix>\n" ^ source
    else source

  let compile ~name bindings lowered =
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = [| lowered.Low_level.llc |]
    end))
    in
    (* gh-ocannl-686: normalize the user-supplied routine name into a legal MSL identifier ONCE,
       here, so the emitted symbol, [new_function_with_name] and the [.metal] artifact agree. *)
    let name = Syntax.kernel_ident name in
    let idx_params = Indexing.bound_symbols bindings in
    (* Add Metal address space qualifiers *)
    let kparams, proc_doc, launch = Syntax.compile_proc ~name idx_params lowered in
    let metal_includes = {|#include <metal_stdlib>
using namespace metal;|} in
    let source =
      maybe_include_simdgroup_matrix
      @@ Syntax.filter_and_prepend_builtins ~routine_names:[ name ] ~includes:metal_includes
           ~builtins:Builtins_metal.builtins ~proc_doc
    in
    { metal_source = source; func_name = name; kparams; bindings; launch }

  let compile_batch ~names bindings lowereds =
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = Array.map lowereds ~f:(fun l -> l.Low_level.llc)
    end))
    in
    (* gh-ocannl-686: normalize the user-supplied routine name into a legal MSL identifier ONCE,
       here, so the emitted symbol, [new_function_with_name] and the [.metal] artifact agree. *)
    let names = Array.map names ~f:Syntax.kernel_ident in
    let idx_params = Indexing.bound_symbols bindings in
    let funcs_and_docs =
      Array.map2_exn names lowereds ~f:(fun name lowered ->
          let kparams, doc, launch = Syntax.compile_proc ~name idx_params lowered in
          ((name, kparams, launch), doc))
    in
    let all_proc_docs = List.map (Array.to_list funcs_and_docs) ~f:snd in
    let final_doc = PPrint.(separate hardline all_proc_docs) in
    let metal_includes = {|#include <metal_stdlib>
using namespace metal;|} in
    let source =
      maybe_include_simdgroup_matrix
      @@ Syntax.filter_and_prepend_builtins ~routine_names:(Array.to_list names)
           ~includes:metal_includes ~builtins:Builtins_metal.builtins ~proc_doc:final_doc
    in
    let funcs = Array.map funcs_and_docs ~f:fst in
    { metal_source = source; funcs; bindings }

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
    if !next > metal_max_pools then
      raise
      @@ Utils.User_error
           (Printf.sprintf
              "Metal_backend: routine needs %d distinct pools, over the metal_max_pools=%d binding \
               budget"
              !next metal_max_pools);
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
     yet. *)
  let encode_wait_for_enqueued dev command_buffer =
    let counter = dev.runner.counter in
    if Unsigned.ULLong.compare counter Unsigned.ULLong.one > 0 then
      Me.CommandBuffer.encode_wait_for_event command_buffer
        (Me.SharedEvent.super dev.runner.event)
        counter

  let%debug4_sexp link_proc ~prior_context ~library ~func_name
      ~(kparams : (string * kparam_source) list) ~(launch : Low_level.launch_dims) ~lowered_bindings
      ~(ctx_buffers : ctx_buffers) : Task.t =
    let dev = prior_context.device in
    let metal_device = dev.dev in
    let queue = dev.runner.queue in
    let runner_label = get_name dev in
    let func = Me.Library.new_function_with_name library func_name in
    let pso, _ = Me.ComputePipelineState.on_device_with_function metal_device func in
    (* Post-link validation (gh-ocannl-536 landing step 3, the portable half): the pre-compile
       [Schedule.check_hardware_limits] judges the schedule against DEVICE-wide limits, but the
       compiled pipeline can be stricter — [maxTotalThreadsPerThreadgroup] drops below the device
       width under register pressure — and knows its own threadgroup-memory footprint. These are
       typed [Resource_exceeded] causes rather than a bare [Utils.User_error]: to a tuner candidate
       they are an ordinary decline (and their key is what the blocker census tabulates), while a
       hand-written schedule still gets the same [Utils.User_error] rendering at the public
       [Context.compile] boundary. Untyped, strict classification would make them fatal and one
       register-heavy candidate would end the search. *)
    let reject resource ~requested ~limit detail =
      raise
      @@ Schedule_outcome.Cause_at
           ( Schedule_outcome.Backend_link,
             Schedule_outcome.Resource_exceeded { resource; requested; limit = Some limit; detail }
           )
    in
    let block_product = launch.block.(0) * launch.block.(1) * launch.block.(2) in
    let max_threads = Me.ComputePipelineState.get_max_total_threads_per_threadgroup pso in
    if block_product > max_threads then
      reject Schedule_outcome.Workgroup_threads ~requested:block_product ~limit:max_threads
        [%string
          "Metal: threadgroup size %{block_product#Int} for %{func_name} exceeds \
           maxTotalThreadsPerThreadgroup %{max_threads#Int}"];
    (* The compiled kernel's own static threadgroup allocation, which the schedule-level accounting
       (a sum over the staged tnodes) only approximates. Checked against the same device limit the
       schedule layer used, so this rejects exactly what could not launch. *)
    Option.iter (hardware_limits ()).Backend_intf.max_workgroup_memory_bytes ~f:(fun max_bytes ->
        let static_bytes = Me.ComputePipelineState.get_static_threadgroup_memory_length pso in
        if static_bytes > max_bytes then
          reject Schedule_outcome.Workgroup_memory ~requested:static_bytes ~limit:max_bytes
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
      (* Unlike CUDA, we don't use Utils.add_log_processor here. Logs are captured by the LogState
         handler installed on the CommandQueue. They will be processed by Utils.log_trace_tree in
         `await`. *)
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
        context_lifetime = (library, pso, ctx_buffers, pool_binding);
        (* Keep library, PSO and the slot-table buffer alive *)
        description = "launches " ^ func_name ^ " on " ^ runner_label;
        work;
      }

  let link prior_context code ctx_buffers =
    let device = prior_context.device.dev in
    let library = compile_metal_source ~name:code.func_name ~source:code.metal_source ~device in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols code.bindings) ~f:(fun s -> (s, ref 0))
    in
    let task =
      link_proc ~prior_context ~library ~func_name:code.func_name ~kparams:code.kparams
        ~launch:code.launch ~lowered_bindings ~ctx_buffers
    in
    (lowered_bindings, task)

  let link_batch prior_context code_batch ctx_buffers =
    let device = prior_context.device.dev in
    (* Name the (debug) source file from the batch's function names — fissioned segments of one
       routine share the routine name as a prefix; a fixed "batch" name would collide across
       routines. *)
    let base_name =
      String.(
        strip ~drop:(equal_char '_')
        @@ common_prefix (List.map (Array.to_list code_batch.funcs) ~f:(fun (n, _, _) -> n)))
    in
    let base_name = if String.is_empty base_name then "batch" else base_name in
    let library = compile_metal_source ~name:base_name ~source:code_batch.metal_source ~device in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols code_batch.bindings) ~f:(fun s -> (s, ref 0))
    in
    let tasks =
      Array.map code_batch.funcs ~f:(fun (func_name, kparams, launch) ->
          link_proc ~prior_context ~library ~func_name ~kparams ~launch ~lowered_bindings
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
      (tasks : Task.t list) : Task.t option =
    let dev = context.device in
    Some
      (Task.Task
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
         })
end

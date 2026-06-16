open Base
open Ir
module Tn = Tnode
module Lazy = Utils.Lazy
module Me = Metal (* Alias for Metal module *)
open Backend_intf
module Impl = Backend_impl (* Alias for Backend_impl *)

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_METAL_BACKEND=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_METAL_BACKEND"]

type ullong = Unsigned.ULLong.t

let sexp_of_ullong x = Sexp.Atom (Unsigned.ULLong.to_string x)

module Backend_buffer = struct
  type buffer_ptr = Me.Buffer.t (* Use the payload type from metal.ml *)

  (* Provide a sexp_of for Me.Buffer.t *)
  let sexp_of_buffer_ptr = Me.Buffer.sexp_of_t

  include Impl.Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]
  end)
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

(* Use Shared storage for unified memory, WriteCombined for CPU writes, Tracked hazards. Shared
   buffers are accessible by both the CPU and the GPU. *)
let shared_resource_options =
  Me.ResourceOptions.(
    storage_mode_shared + cpu_cache_mode_write_combined + hazard_tracking_mode_tracked)

(* Private storage is GPU-only: the CPU cannot map it, so there is no CPU cache mode to set.
   Choosing private for GPU-only buffers avoids CPU cache-coherency traffic and lets Metal pick a
   GPU-friendly layout. *)
let private_resource_options =
  Me.ResourceOptions.(storage_mode_private + hazard_tracking_mode_tracked)

(* GPU-only tnodes ([Local], [Device_only], [On_device]) do not require persistent CPU-visible
   storage. After gh-ocannl-333 there is no [Hosted] mode; on-demand host read-back / upload for
   [On_device] nodes goes through the backend's blit-based [to_host]/[from_host], so private storage
   remains valid for them. The materialization-request modes -- [Materialized], [Effectively_constant],
   the partially-resolved [Never_virtual], and the [None] default -- may still be initialized or
   wrapped via shared memory (e.g. [use_host_memory]) and stay shared. [Virtual] tnodes are inlined
   and never reach the allocator; they map to shared defensively. *)
let storage_mode_for_memory_mode : Tn.memory_mode option -> Me.Resource.StorageMode.t = function
  | Some (Local | Device_only | On_device) -> Me.Resource.StorageMode.Private
  | Some (Effectively_constant | Virtual | Never_virtual | Materialized) | None ->
      Me.Resource.StorageMode.Shared

let resource_options_for_mode (mode : Tn.memory_mode option) =
  match storage_mode_for_memory_mode mode with
  | Me.Resource.StorageMode.Private -> private_resource_options
  | Shared | Managed | Memoryless -> shared_resource_options

(* The Metal slab allocator: a private [(device_id, pool_id) -> Metal.Buffer.t] table backing the
   shared {!Backend_intf.Slab_alloc}. Storage mode (private vs. shared) is a per-pool property carried
   by [?mode]. *)
module Slab = struct
  open Backend_intf

  type device = Device_stream.device
  type buffer_ptr = Me.Buffer.t

  let pools : (int * int, buffer_ptr) Hashtbl.Poly.t = Hashtbl.Poly.create ()

  let alloc_pool ?mode (device : device) ~pool_id ~size_in_bytes ~alignment:_ =
    let buffer =
      Me.Buffer.on_device device.dev ~length:(max 1 size_in_bytes) (resource_options_for_mode mode)
    in
    track_allocation buffer;
    Hashtbl.set pools ~key:(device.device_id, pool_id) ~data:buffer

  (* Rely on ARC and the finalizer attached in track_allocation for the actual reclamation, but still
     drop the private table entry on finalization so the strong reference is released (otherwise ARC
     can never reclaim a context's tnode buffers). Re-allocating a pool also overwrites the entry. *)
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

  let resolve_pool (device : device) { pool_id; offset } : buffer_ptr =
    (* Phase-1 policy: one pool per tnode at offset 0. *)
    assert (offset = 0);
    Hashtbl.find_exn pools (device.device_id, pool_id)

  let storage_mode_of_pool (device : device) ~pool_id =
    Me.Resource.get_storage_mode
      (Me.Buffer.super (Hashtbl.find_exn pools (device.device_id, pool_id)))
end

(* Functor defining the backend. The exact public signature (Lowered_backend + storage_mode_of_pool)
   is sealed by metal_backend.mli. *)
module Fresh () = struct
  (* Include the device setup with types and allocation *)
  include Backend_impl.Device (Device_stream) (Slab)

  (* The concrete [buffer_ptr]/[buffer] + sexps for the impl-facing interface (no longer carried by
     the shared [Device_config_common]). *)
  include Backend_buffer

  let storage_mode_of_pool = Slab.storage_mode_of_pool

  (* Global state for Metal devices *)
  let metal_devices : Me.Device.t array = Me.Device.copy_all_devices ()
  let () = assert (Array.length metal_devices > 0)

  (* Store for captured logs per device_id (the device is its own single compute stream). *)
  let stream_logs : (int, string list ref) Hashtbl.t = Hashtbl.create (module Int)

  (* Metal has unified memory on Apple Silicon, so we can use host memory *)
  let get_buffer_for_ptr device ~size_in_bytes bytes =
    Me.Buffer.on_device_with_bytes_no_copy device ~bytes ~length:size_in_bytes
      Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache)

  let use_host_memory = Some (get_buffer_for_ptr metal_devices.(0))

  (* Device Management *)
  let num_devs = Array.length metal_devices
  let devices_cache = Array.create ~len:num_devs None

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
    ({ queue = actual_queue; event = shared_event_obj; counter }, opt_log_entries_ref)

  let get_device ~(ordinal : int) : device =
    if ordinal < 0 || num_devs <= ordinal then
      invalid_arg [%string "Metal_backend.get_device %{ordinal#Int}: invalid ordinal"];
    let default () =
      let metal_device = metal_devices.(ordinal) in
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
    num_devs

  let new_stream (device : device) : device = device

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

  let static_properties =
    let device_properties =
      Array.mapi metal_devices ~f:(fun ordinal device ->
          let attributes = Me.Device.get_attributes device in
          Sexp.List
            [
              Sexp.Atom "device";
              Sexp.List
                [
                  Sexp.List [ Sexp.Atom "device_name"; Sexp.Atom attributes.name ];
                  Sexp.List [ Sexp.Atom "device_ordinal"; Sexp.Atom (Int.to_string ordinal) ];
                  Sexp.List
                    [
                      Sexp.Atom "registry_id";
                      Sexp.Atom (Unsigned.ULLong.to_string attributes.registry_id);
                    ];
                  Sexp.List
                    [
                      Sexp.Atom "max_buffer_length";
                      Sexp.Atom (Unsigned.ULong.to_string attributes.max_buffer_length);
                    ];
                  Sexp.List
                    [
                      Sexp.Atom "max_threadgroup_memory_length";
                      Sexp.Atom (Unsigned.ULong.to_string attributes.max_threadgroup_memory_length);
                    ];
                  Sexp.List
                    [
                      Sexp.Atom "recommended_max_working_set_size";
                      Sexp.Atom
                        (Unsigned.ULLong.to_string attributes.recommended_max_working_set_size);
                    ];
                  Sexp.List
                    [ Sexp.Atom "is_low_power"; Sexp.Atom (Bool.to_string attributes.is_low_power) ];
                  Sexp.List
                    [ Sexp.Atom "is_headless"; Sexp.Atom (Bool.to_string attributes.is_headless) ];
                  Sexp.List
                    [
                      Sexp.Atom "has_unified_memory";
                      Sexp.Atom (Bool.to_string attributes.has_unified_memory);
                    ];
                  Sexp.List
                    [
                      Sexp.Atom "total_memory";
                      Sexp.Atom (Int.to_string (Atomic.get allocated_memory));
                    ];
                  Sexp.List
                    [
                      Sexp.Atom "supported_gpu_families";
                      Sexp.List
                        (List.map attributes.supported_gpu_families ~f:(fun gpu_family ->
                             Me.Device.GPUFamily.sexp_of_t gpu_family));
                    ];
                ];
            ])
    in
    Sexp.List (Sexp.Atom "metal_devices" :: Array.to_list device_properties)

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
  let from_host ~dst ~dst_loc hosted =
    (* Copy from host memory to Metal buffer *)
    let dst_ptr = Slab.resolve_pool dst.device dst_loc in
    let size_in_bytes = Ndarray.size_in_bytes hosted in
    let command_buffer = Me.CommandBuffer.on_queue dst.device.runner.queue in

    (* Get host memory pointer *)
    let host_ptr = Ndarray.get_fatptr_not_managed hosted in
    let (Ctypes_static.CPointer dst_fatptr) = Me.Buffer.contents dst_ptr in
    if Ctypes_ptr.Fat.compare dst_fatptr host_ptr <> 0 then (
      (* Create a temporary host buffer to bridge the gap *)
      let temp_buffer =
        Me.Buffer.on_device_with_bytes_no_copy dst.device.dev
          ~bytes:(Ctypes_static.CPointer host_ptr) ~length:size_in_bytes
          Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache)
      in
      let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
      Me.BlitCommandEncoder.copy_from_buffer blit_encoder ~source_buffer:temp_buffer
        ~source_offset:0 ~destination_buffer:dst_ptr ~destination_offset:0 ~size:size_in_bytes;
      Me.BlitCommandEncoder.end_encoding blit_encoder;
      Me.CommandBuffer.commit command_buffer)

  let to_host ~src ~src_loc hosted =
    (* Copy from Metal buffer to host memory *)
    let src_ptr = Slab.resolve_pool src.device src_loc in
    let size_in_bytes = Ndarray.size_in_bytes hosted in
    let command_buffer = Me.CommandBuffer.on_queue src.device.runner.queue in

    (* Get host memory pointer *)
    let host_ptr = Ndarray.get_fatptr_not_managed hosted in
    let (Ctypes_static.CPointer src_fatptr) = Me.Buffer.contents src_ptr in
    if Ctypes_ptr.Fat.compare src_fatptr host_ptr <> 0 then (
      (* Create a temporary host buffer to bridge the gap *)
      let temp_buffer =
        Me.Buffer.on_device_with_bytes_no_copy src.device.dev
          ~bytes:(Ctypes_static.CPointer host_ptr) ~length:size_in_bytes
          Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache)
      in
      let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
      Me.BlitCommandEncoder.copy_from_buffer blit_encoder ~source_buffer:src_ptr ~source_offset:0
        ~destination_buffer:temp_buffer ~destination_offset:0 ~size:size_in_bytes;
      Me.BlitCommandEncoder.end_encoding blit_encoder;
      Me.CommandBuffer.commit command_buffer)

  (* The merge buffer is the device's reserved single-tenant pool (id [merge_buffer_pool_id]); grow it
     in place when a larger node arrives ([Slab.alloc_pool] overwrites the reserved entry). The merge
     buffer holds a copy of [tn], so it inherits [tn]'s storage-mode classification. *)
  let opt_alloc_merge_buffer ?mode ~size_in_bytes (device : device) : unit =
    if device.merge_buffer_capacity < size_in_bytes then (
      Slab.alloc_pool ?mode device ~pool_id:merge_buffer_pool_id ~size_in_bytes ~alignment:1;
      device.merge_buffer_capacity <- size_in_bytes);
    device.merge_buffer := Some { pool_id = merge_buffer_pool_id; offset = 0 }

  let device_to_device tn ~into_merge_buffer ~dst_loc ~dst ~src_loc ~src =
    let size_in_bytes = Lazy.force tn.Tn.size_in_bytes in
    let src_ptr = Slab.resolve_pool src.device src_loc in

    let memcpy ~dst_ptr =
      (* Always use explicit copy as Metal doesn't have peer-to-peer memory access like CUDA *)
      let command_buffer = Me.CommandBuffer.on_queue dst.device.runner.queue in
      let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
      Me.BlitCommandEncoder.copy_from_buffer blit_encoder ~source_buffer:src_ptr ~source_offset:0
        ~destination_buffer:dst_ptr ~destination_offset:0 ~size:size_in_bytes;
      Me.BlitCommandEncoder.end_encoding blit_encoder;
      Me.CommandBuffer.commit command_buffer
    in

    match (into_merge_buffer, dst_loc) with
    | No, None -> invalid_arg "Metal_backend.device_to_device: missing dst_loc"
    | No, Some dst_loc -> memcpy ~dst_ptr:(Slab.resolve_pool dst.device dst_loc)
    | Copy, _ ->
        opt_alloc_merge_buffer
          ?mode:(Option.map tn.Tn.memory_mode ~f:fst)
          ~size_in_bytes dst.device;
        let loc = Option.value_exn ~here:[%here] !(dst.device.merge_buffer) in
        memcpy ~dst_ptr:(Slab.resolve_pool dst.device loc)

  (* --- Compilation and Linking --- *)
  type code = {
    metal_source : string; (* Store source, compile during link if not already compiled *)
    compiled_code : Me.Library.t option array; (* Store compiled code per device *)
    func_name : string;
    kparams : (string * kparam_source) list;
    bindings : Indexing.unit_bindings;
    traced_store : Low_level.traced_store;
  }
  [@@deriving sexp_of]

  type code_batch = {
    metal_source : string; (* Store combined source *)
    compiled_code : Me.Library.t option array; (* Store compiled code per device *)
    funcs : (string * (string * kparam_source) list) option array; (* func_name * kparams *)
    bindings : Indexing.unit_bindings;
    traced_stores : Low_level.traced_store option array;
  }
  [@@deriving sexp_of]

  module C_syntax_config (Input : sig
    val procs : Low_level.optimized array
  end) =
  struct
    include C_syntax.Pure_C_config (struct
      type nonrec buffer_ptr = buffer_ptr

      let use_host_memory = use_host_memory
      let procs = Input.procs
      let full_printf_support = false
    end)

    open PPrint (* Open PPrint locally *)
    open Indexing.Doc_helpers (* Open our helpers *)

    let main_kernel_prefix = "kernel"
    let buffer_prefix = "device "
    let buffer_suffix = fun ~pos -> " [[buffer(" ^ Int.to_string pos ^ ")]]"

    let arg_int_prefix =
      if Utils.settings.big_models then "const uint64_t& " else "const uint32_t& "

    let extra_args =
      [
        "uint3 gid [[threadgroup_position_in_grid]]"; "uint3 lid [[thread_position_in_threadgroup]]";
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
      | Ops.Fp8_prec _ -> invalid_arg "Metal backend does not support FP8 precision"
      | Ops.Single_prec _ -> "float"
      | Ops.Double_prec _ ->
          raise @@ Utils.User_error "Metal backend does not support double precision"
      | Ops.Int64_prec _ -> "long"
      | Ops.Uint64_prec _ -> "ulong"
      | Ops.Void_prec -> "void"

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
      | (Ops.Uint16_prec _ | Ops.Bfloat16_prec _), 8 -> "uint16x8_t"
      | Ops.Half_prec _, 8 -> "half8_t"
      | _, 1 -> typ_of_prec prec
      | _ -> invalid_arg "Metal_backend.vec_typ_of_prec: invalid combination"

    let metal_prec_suffix_float = function
      | Ops.Byte_prec _ -> ""
      | Ops.Uint16_prec _ -> ""
      | Ops.Int32_prec _ -> ""
      | Ops.Uint32_prec _ -> ""
      | Ops.Uint4x32_prec _ -> "" (* No specific suffix for uint4 *)
      | Ops.Half_prec _ -> "h"
      | Ops.Bfloat16_prec _ -> "bf" (* TODO: Verify actual Metal suffix for bfloat16 *)
      | Ops.Fp8_prec _ -> invalid_arg "Metal backend does not support FP8 precision"
      | Ops.Single_prec _ -> "f"
      | Ops.Double_prec _ ->
          raise @@ Utils.User_error "Metal backend does not support double precision"
      | Ops.Int64_prec _ -> "l"
      | Ops.Uint64_prec _ -> "ul"
      | Ops.Void_prec -> ""

    let ternop_syntax _prec op =
      match op with
      | Ops.Where ->
          fun v1 v2 v3 -> group (string "select(" ^^ separate comma_sep [ v3; v2; v1 ] ^^ rparen)
      | FMA -> fun v1 v2 v3 -> group (string "fma(" ^^ separate comma_sep [ v1; v2; v3 ] ^^ rparen)

    let infix_binop op v1 v2 = parens (infix 2 1 (string op) v1 v2)

    let binop_syntax prec op =
      let f = infix_binop in
      let func fn v1 v2 = group (string fn ^^ parens (v1 ^^ comma ^^ space ^^ v2)) in
      match (op, prec) with
      | Ops.Add, _ -> f "+"
      | Sub, _ -> f "-"
      | Mul, _ -> f "*"
      | Div, _ -> f "/"
      | Mod, _ -> func "fmod"
      | Max, _ -> func "fmax"
      | Min, _ -> func "fmin"
      | Cmpeq, _ -> f "=="
      | Cmpne, _ -> f "!="
      | Cmplt, _ -> f "<"
      | And, _ -> f "&&"
      | Or, _ -> f "||"
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
      | ToPowOf, _ -> func "pow"
      | Threefry4x32_crypto, _ -> (
          (* Threefry4x32_crypto must output to uint4x32 precision *)
          match prec with
          | Ops.Uint4x32_prec _ -> func "arrayjit_threefry4x32_crypto"
          | _ ->
              raise
              @@ Utils.User_error
                   (Printf.sprintf
                      "Metal backend: Threefry4x32_crypto requires target precision to be \
                       uint4x32, but got %s"
                      (Ops.prec_string prec)))
      | Threefry4x32_light, _ -> (
          (* Threefry4x32_light must output to uint4x32 precision *)
          match prec with
          | Ops.Uint4x32_prec _ -> func "arrayjit_threefry4x32_light"
          | _ ->
              raise
              @@ Utils.User_error
                   (Printf.sprintf
                      "Metal backend: Threefry4x32_light requires target precision to be uint4x32, \
                       but got %s"
                      (Ops.prec_string prec)))
      | Arg1, _ | Arg2, _ -> invalid_arg "Metal C_syntax_config: Arg1/Arg2 not operators"

    let unop_syntax prec op =
      let func_doc fn v = group (string fn ^^ parens v) in
      match (op, prec) with
      | Ops.Identity, _ -> fun v -> v
      | Neg, _ -> fun v -> string "-" ^^ v
      | Exp, _ -> func_doc "exp"
      | Log, _ -> func_doc "log"
      | Exp2, _ -> func_doc "exp2"
      | Log2, _ -> func_doc "log2"
      | Sin, _ -> func_doc "sin"
      | Cos, _ -> func_doc "cos"
      | Sqrt, _ -> func_doc "sqrt"
      | Relu, Ops.Half_prec _ -> fun v -> func_doc "max" (separate comma_sep [ string "0.0h"; v ])
      | Relu, Ops.Single_prec _ -> fun v -> func_doc "max" (separate comma_sep [ string "0.0f"; v ])
      | Relu, Ops.Double_prec _ ->
          raise @@ Utils.User_error "Metal backend does not support double precision"
      | Relu, _ (* Byte_prec, Void_prec *) ->
          fun v -> func_doc "max" (separate comma_sep [ string "0"; v ])
      | Satur01, p ->
          let s = metal_prec_suffix_float p in
          fun v ->
            func_doc "clamp" (separate comma_sep [ v; string ("0.0" ^ s); string ("1.0" ^ s) ])
      | Recip, p ->
          let s = metal_prec_suffix_float p in
          fun v -> infix_binop "/" (string @@ "1.0" ^ s) v
      | Recip_sqrt, _ -> func_doc "rsqrt"
      | Tanh_approx, _ -> func_doc "tanh"
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
      (* Default case for all other conversions *)
      | _ -> ("(" ^ typ_of_prec to_ ^ ")(", ")")

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
          (string metal_log_object_name ^^ string ".log_debug(" ^^ base_doc
          ^^ comma ^^ nest 4 (break 1 ^^ separate (comma ^^ break 1) args_docs)
          ^^ rparen ^^ semi)
  end

  let%diagn_sexp compile_metal_source ~name ~source ~device =
    let options = Me.CompileOptions.init () in
    if Utils.debug_log_from_routines () then (
      Me.CompileOptions.set_language_version options Me.CompileOptions.LanguageVersion.version_3_2;
      Me.CompileOptions.set_enable_logging options true)
    else
      Me.CompileOptions.set_language_version options Me.CompileOptions.LanguageVersion.version_3_1
      (* Version 3.1 is required for the `bfloat` type (bfloat16 precision). *)
      (* Logging is disabled by default in CompileOptions, so no need to explicitly set it to
         false *);

    if Utils.settings.output_debug_files_in_build_directory then (
      let metal_file = Utils.build_file (name ^ ".metal") in
      Stdio.Out_channel.write_all metal_file ~data:source;
      [%log "Wrote metal source to file:", metal_file]);

    try Me.Library.on_device device ~source options
    with Failure msg ->
      let error_msg =
        Printf.sprintf "Metal compilation failed for %s:\n%s\nSource:\n%s" name msg source
      in
      Stdio.prerr_endline error_msg;
      failwith error_msg

  let compile ~name bindings lowered =
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = [| lowered |]
    end))
    in
    let idx_params = Indexing.bound_symbols bindings in
    (* Add Metal address space qualifiers *)
    let kparams, proc_doc = Syntax.compile_proc ~name idx_params lowered in
    let metal_includes = {|#include <metal_stdlib>
using namespace metal;|} in
    let source =
      Syntax.filter_and_prepend_builtins ~includes:metal_includes ~builtins:Builtins_metal.builtins
        ~proc_doc
    in
    {
      metal_source = source;
      compiled_code = Array.create ~len:num_devs None;
      (* One slot per device *)
      func_name = name;
      kparams;
      bindings;
      traced_store = lowered.traced_store;
    }

  let compile_batch ~names bindings lowereds =
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = Array.filter_opt lowereds
    end))
    in
    let idx_params = Indexing.bound_symbols bindings in
    let funcs_and_docs =
      Array.map2_exn names lowereds
        ~f:
          (Option.map2 ~f:(fun name lowered ->
               let kparams, doc = Syntax.compile_proc ~name idx_params lowered in
               ((name, kparams), doc)))
    in
    let all_proc_docs = List.filter_map (Array.to_list funcs_and_docs) ~f:(Option.map ~f:snd) in
    let final_doc = PPrint.(separate hardline all_proc_docs) in
    let metal_includes = {|#include <metal_stdlib>
using namespace metal;|} in
    let source =
      Syntax.filter_and_prepend_builtins ~includes:metal_includes ~builtins:Builtins_metal.builtins
        ~proc_doc:final_doc
    in
    let traced_stores = Array.map lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store)) in
    let funcs = Array.map funcs_and_docs ~f:(Option.map ~f:fst) in
    {
      metal_source = source;
      compiled_code = Array.create ~len:num_devs None;
      (* One slot per device *)
      funcs;
      bindings;
      traced_stores;
    }

  let%debug4_sexp link_proc ~prior_context ~library ~func_name
      ~(kparams : (string * kparam_source) list) ~lowered_bindings
      ~(ctx_arrays : buffer_ptr Tn.t_map) : Task.t =
    let dev = prior_context.device in
    let metal_device = dev.dev in
    let queue = dev.runner.queue in
    let runner_label = get_name dev in
    let func = Me.Library.new_function_with_name library func_name in
    let pso, _ = Me.ComputePipelineState.on_device_with_function metal_device func in

    let work () : unit =
      [%log3_result "Launching", func_name, "on", runner_label];
      (* Unlike CUDA, we don't use Utils.add_log_processor here. Logs are captured by the LogState
         handler installed on the CommandQueue. They will be processed by Utils.log_trace_tree in
         `await`. *)
      try
        let command_buffer = Me.CommandBuffer.on_queue queue in
        let encoder = Me.ComputeCommandEncoder.on_buffer command_buffer in
        Me.ComputeCommandEncoder.set_compute_pipeline_state encoder pso;

        (* Set arguments *)
        List.iteri kparams ~f:(fun index (_p_name, p_source) ->
            match p_source with
            | Kparam_ptr tn when Map.mem ctx_arrays tn ->
                let buffer = Map.find_exn ctx_arrays tn in
                Me.ComputeCommandEncoder.set_buffer encoder ~index buffer
            | Kparam_ptr tn ->
                (* After gh-ocannl-333 there is no host array to wrap as a constant buffer: every
                   in-context node (including constants) is allocated in [ctx_arrays] by
                   [alloc_if_needed], which uploads any host initialization data there. *)
                failwith
                  [%string
                    "Kparam_ptr %{Tn.debug_name tn} not found in ctx_arrays for %{func_name}"]
            | Static_idx s ->
                let value = !(Indexing.find_exn lowered_bindings s) in
                let size = Ctypes.sizeof Ctypes.int in
                let bytes_ptr = Ctypes.(allocate int value |> to_voidp) in
                Me.ComputeCommandEncoder.set_bytes encoder ~bytes:bytes_ptr ~length:size ~index
            | Merge_buffer -> (
                match !(dev.merge_buffer) with
                | Some loc ->
                    Me.ComputeCommandEncoder.set_buffer encoder ~index (Slab.resolve_pool dev loc)
                | None -> failwith [%string "Merge_buffer requested but not set for %{func_name}"])
            | Log_file_name ->
                (* TODO:We could tag logs with a run id. *)
                assert false);

        (* Dispatch - TODO: Determine grid/group sizes properly *)
        let max_threads = Me.ComputePipelineState.get_max_total_threads_per_threadgroup pso in
        let width = Int.min max_threads (* 32 *) 1 in
        (* Example: Use a small group size *)
        (* Example: single group *)
        Me.ComputeCommandEncoder.dispatch_threadgroups encoder
          ~threadgroups_per_grid:{ width = 1; height = 1; depth = 1 }
          ~threads_per_threadgroup:{ width; height = 1; depth = 1 };

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
        context_lifetime = (library, pso, ctx_arrays);
        (* Keep library and PSO alive *)
        description = "launches " ^ func_name ^ " on " ^ runner_label;
        work;
      }

  let link prior_context code ctx_buffers =
    let device = prior_context.device.dev in
    let library = compile_metal_source ~name:code.func_name ~source:code.metal_source ~device in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols code.bindings) ~f:(fun s -> (s, ref 0))
    in
    let ctx_arrays = Map.map ctx_buffers ~f:(Slab.resolve_pool prior_context.device) in
    let task =
      link_proc ~prior_context ~library ~func_name:code.func_name ~kparams:code.kparams
        ~lowered_bindings ~ctx_arrays
    in
    (lowered_bindings, task)

  let link_batch prior_context code_batch ctx_buffers_opts =
    let device = prior_context.device.dev in
    let library = compile_metal_source ~name:"batch" ~source:code_batch.metal_source ~device in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols code_batch.bindings) ~f:(fun s -> (s, ref 0))
    in

    let tasks =
      Array.mapi code_batch.funcs ~f:(fun i func_opt ->
          Option.bind func_opt ~f:(fun (func_name, kparams) ->
              Option.map ctx_buffers_opts.(i) ~f:(fun ctx_buffers ->
                  let ctx_arrays =
                    Map.map ctx_buffers ~f:(Slab.resolve_pool prior_context.device)
                  in
                  link_proc ~prior_context ~library ~func_name ~kparams ~lowered_bindings
                    ~ctx_arrays)))
    in
    (lowered_bindings, tasks)
end

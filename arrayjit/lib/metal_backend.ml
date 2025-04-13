open Base
open Ir
module Tn = Tnode
module Lazy = Utils.Lazy
module Me = Metal
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

(* Placeholder for error checking, adapt Metal's error handling as needed *)
let check_metal_error label = function
  | err_id when Metal.is_nil err_id -> ()
  | err_id ->
      let desc = Metal.get_error_description err_id in
      failwith [%string "%{label} failed: %{desc}"]

(* Shared event counter for signaling *)
let next_event_value = Atomic.make Unsigned.ULLong.one

module Backend_buffer = struct
  type buffer_ptr = Me.Buffer.t

  let sexp_of_buffer_ptr = Me.Buffer.sexp_of_t

  include Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]
  end)
end

module Device_config = struct
  include Backend_buffer

  type dev = Me.Device.t [@@deriving sexp_of]
  type runner = Me.CommandQueue.t [@@deriving sexp_of]

  (* Event is a shared event and the value it should reach *)
  type event = Me.SharedEvent.t * Unsigned.ULLong.t [@@deriving sexp_of]

  let name = "metal"
end

module Device_stream = Backend_impl.Device_types (Device_config)
open Device_config

module Alloc_buffer = struct
  include Device_stream

  let use_host_memory =
    (* Determine if the default device has unified memory *)
    let check () =
      let device = Me.Device.create_system_default () in
      let attrs = Me.Device.get_attributes device in
      attrs.has_unified_memory
    in
    if check () then Some Metal.Buffer.contents else None

  let alloc_buffer ?old_buffer ~size_in_bytes stream =
    let device = stream.device.dev in
    match old_buffer with
    | Some ({ size_in_bytes = old_size; _ } as buffer) when size_in_bytes <= old_size -> buffer
    | _ ->
        (* TODO: Choose options based on use_host_memory and device properties *)
        let options = Me.ResourceOptions.storage_mode_shared in
        let new_ptr = Me.Buffer.on_device device ~length:size_in_bytes options in
        { ptr = new_ptr; size_in_bytes }

  let alloc_zero_init_array prec ~dims stream =
    let size_in_bytes =
      (if Array.length dims = 0 then 0 else Array.reduce_exn dims ~f:( * )) * Ops.prec_in_bytes prec
    in
    let device = stream.device.dev in
    (* Metal buffers are zero-initialized by default. *)
    let options = Me.ResourceOptions.storage_mode_shared in
    Me.Buffer.on_device device ~length:size_in_bytes options

  (* Metal uses ARC; explicit free is likely not needed if bindings are correct. *)
  let free_buffer = None
end

let initialized_devices = Hash_set.create (module Int)
let the_device = Lazy.from_fun Metal.Device.create_system_default

module Fresh
    ()
(* : Backend_impl.Lowered_backend with type buffer_ptr = Device_config.buffer_ptr and type dev =
   Device_config.dev and type runner = Device_config.runner and type event = Device_config.event and
   type code = Low_level.optimized and type code_batch = Low_level.optimized array *) =
struct
  include Backend_impl.Device (Device_stream) (Alloc_buffer)

  let use_host_memory = Alloc_buffer.use_host_memory
  let global_config = ref For_parallel_copying

  let is_initialized, initialize =
    let initialized = ref false in
    let init (config : config) : unit =
      initialized := true;
      global_config := config
    in
    ((fun () -> !initialized), init)

  (* Metal Bindings TODO: Need MTLCopyAllDevices or equivalent *)
  let num_devices () = 1
  let devices = ref @@ Array.create ~len:1 None

  let get_device ~(ordinal : int) : device =
    if ordinal <> 0 then
      invalid_arg [%string "Metal backend currently only supports the default device (ordinal 0)"];
    let default () =
      let dev = Lazy.force the_device in
      Hash_set.add initialized_devices ordinal;
      let result = make_device dev ~ordinal in
      (* TODO: Metal does not require explicit device finalization like CUDA context sync *)
      (* Stdlib.Gc.finalise finalize_device result; *)
      !devices.(ordinal) <- Some result;
      result
    in
    Option.value_or_thunk !devices.(ordinal) ~default

  let new_stream (device : device) : stream =
    let queue = Me.CommandQueue.on_device device.dev in
    make_stream device queue

  let metal_properties =
    let cache =
      lazy
        (let dev = get_device ~ordinal:0 in
         Me.Device.get_attributes dev.dev)
    in
    fun (_device : device) -> Lazy.force cache

  let suggested_num_streams device =
    match !global_config with
    | Only_devices_parallel -> 1
    | For_parallel_copying -> 1 (* Metal has implicit host/device transfer queues *)
    | Most_parallel_streams ->
        (* No direct core count, use a heuristic or default *)
        let attrs = metal_properties device in
        (* Example: Use max threads per threadgroup as a proxy? Needs tuning *)
        (attrs.max_threads_per_threadgroup.width / 64) (* Guess *) + 1

  (* Event Management *)
  let shared_events = Hashtbl.create (module Int) (* ordinal -> SharedEvent.t *)

  let get_shared_event (device : device) : Me.SharedEvent.t =
    Hashtbl.find_or_add shared_events device.ordinal ~default:(fun () ->
        Me.SharedEvent.on_device device.dev)

  let sync ((event, target_val) : event) : unit =
    let current_val = Me.SharedEvent.signaled_value event in
    if Unsigned.ULLong.compare current_val target_val < 0 then (
      (* Metal Bindings TODO: Need synchronous wait or better notification *)
      (* Workaround: Use a listener with a condition variable *)
      let mutex = Caml_threads.Mutex.create () in
      let cond = Caml_threads.Condition.create () in
      let notified = ref false in
      let callback _event signaled_value =
        Caml_threads.Mutex.lock mutex;
        if Unsigned.ULLong.compare signaled_value target_val >= 0 then (
          notified := true;
          Caml_threads.Condition.signal cond);
        Caml_threads.Mutex.unlock mutex
      in
      Me.SharedEvent.notify_listener event
        (Lazy.force Utils.metal_event_listener)
        target_val callback;
      (* Check again in case event was signaled between check and listener setup *)
      let current_val' = Me.SharedEvent.signaled_value event in
      if Unsigned.ULLong.compare current_val' target_val < 0 then (
        Caml_threads.Mutex.lock mutex;
        while not !notified do
          Caml_threads.Condition.wait cond mutex
        done;
        Caml_threads.Mutex.unlock mutex))

  let is_done ((event, target_val) : event) : bool =
    Unsigned.ULLong.compare (Me.SharedEvent.signaled_value event) target_val >= 0

  (* Store last command buffer for await/is_idle *)
  let last_command_buffer : Me.CommandBuffer.t option ref = ref None

  let commit_buffer (stream : stream) (cmdbuf : Me.CommandBuffer.t) =
    last_command_buffer := Some cmdbuf;
    Me.CommandBuffer.commit cmdbuf;
    (* Check for errors *after* commit, might need completion handler for robust check *)
    let err = Me.CommandBuffer.error cmdbuf in
    check_metal_error (get_name stream ^ " command buffer commit") err

  let await stream : unit =
    (* This waits only for the *last submitted* buffer. Might need more robust tracking if
       out-of-order execution matters. *)
    match !last_command_buffer with
    | None -> ()
    | Some cmdbuf ->
        (* TODO: only wait if cmdbuf belongs to this stream? *)
        Me.CommandBuffer.wait_until_completed cmdbuf;
        check_metal_error
          (get_name stream ^ " command buffer await")
          (Me.CommandBuffer.error cmdbuf);
        last_command_buffer := None;
        (* Assume awaited buffer is done *)
        Option.iter !Utils.advance_captured_logs ~f:(fun callback -> callback ())

  let is_idle _stream =
    (* Metal Bindings TODO: Need queue idle check or robust last buffer status check *)
    match !last_command_buffer with
    | None -> true
    | Some cmdbuf ->
        let status = Objc.msg_send ~self:cmdbuf ~cmd:(selector "status") ~typ:(returning int) in
        status >= 4 (* MTLCommandBufferStatusCompleted or Error *)

  let all_work stream : event =
    let shared_event = get_shared_event stream.device in
    let target_val = Atomic.fetch_and_add next_event_value Unsigned.ULLong.one in
    let cmdbuf = Me.CommandQueue.command_buffer stream.runner in
    Me.CommandBuffer.encode_signal_event cmdbuf shared_event target_val;
    commit_buffer stream cmdbuf;
    (shared_event, target_val)

  let will_wait_for (context : context) ((event, target_val) : event) : unit =
    (* This wait needs to be encoded into a command buffer. We assume it affects the *next* buffer created. *)
    (* This requires context/stream to hold pending operations or modify buffer creation. *)
    (* Hack: Store pending wait on stream, apply in next buffer creation/encoding *)
    (* A better approach needs changes to backend_intf or careful state management *)
    failwith
      "Metal: will_wait_for needs implementation adjustment - requires command buffer encoding"
  (* Placeholder: let cmdbuf = Me.CommandQueue.command_buffer context.stream.runner in
     Me.CommandBuffer.encode_wait_for_event cmdbuf event target_val; commit_buffer context.stream
     cmdbuf *)

  (* Data Transfer *)
  let get_blit_encoder (stream : stream) =
    let cmdbuf = Me.CommandQueue.command_buffer stream.runner in
    let encoder = Me.CommandBuffer.blit_command_encoder cmdbuf in
    (cmdbuf, encoder)
  (* Return buffer to commit later *)

  let from_host ~dst_ptr ~dst hosted =
    match use_host_memory with
    | Some _
      when Me.ResourceOptions.equal
             (Objc.msg_send ~self:dst_ptr ~cmd:(selector "storageMode") ~typ:(returning uint64_t))
             Me.ResourceOptions.storage_mode_shared ->
        (* If shared, copy directly using Ctypes pointers *)
        let host_ptr = Ndarray.to_ ctypes_voidptr hosted in
        let size = Ndarray.size_in_bytes hosted in
        let device_contents_ptr = Me.Buffer.contents dst_ptr in
        Ctypes.memcpy ~dst:device_contents_ptr ~src:host_ptr ~size
        (* Requires didModifyRange for managed mode, not applicable here *)
    | _ ->
        (* Use blit encoder *)
        let size = Ndarray.size_in_bytes hosted in
        let host_buf = Ndarray.get_managed_bytes_readonly_ptr hosted in
        (* Ctypes needs pointer *)
        let temp_host_buffer =
          Me.Buffer.on_device dst.stream.device.dev ~length:size
            Me.ResourceOptions.storage_mode_shared
        in
        let temp_contents = Me.Buffer.contents temp_host_buffer in
        Ctypes.memcpy ~dst:temp_contents ~src:(Ctypes.to_voidp host_buf) ~size;
        let cmdbuf, encoder = get_blit_encoder dst.stream in
        Me.BlitCommandEncoder.copy_from_buffer encoder ~source_buffer:temp_host_buffer
          ~source_offset:0 ~destination_buffer:dst_ptr ~destination_offset:0 ~size;
        Me.BlitCommandEncoder.end_encoding encoder;
        commit_buffer dst.stream cmdbuf
  (* Temp host buffer will be GC'd *)

  let to_host ~src_ptr ~src hosted =
    match use_host_memory with
    | Some _
      when Me.ResourceOptions.equal
             (Objc.msg_send ~self:src_ptr ~cmd:(selector "storageMode") ~typ:(returning uint64_t))
             Me.ResourceOptions.storage_mode_shared ->
        let host_ptr = Ndarray.to_ ctypes_voidptr hosted in
        let size = Ndarray.size_in_bytes hosted in
        let device_contents_ptr = Me.Buffer.contents src_ptr in
        (* Ensure GPU writes are visible *)
        let cmdbuf, encoder = get_blit_encoder src.stream in
        Me.BlitCommandEncoder.synchronize_resource encoder ~resource:src_ptr;
        Me.BlitCommandEncoder.end_encoding encoder;
        commit_buffer src.stream cmdbuf;
        await src.stream;
        (* Wait for sync to complete before CPU read *)
        Ctypes.memcpy ~dst:host_ptr ~src:device_contents_ptr ~size
    | _ ->
        let size = Ndarray.size_in_bytes hosted in
        let host_buf_ptr = Ndarray.get_managed_bytes_writeonly_ptr hosted in
        (* Ctypes needs pointer *)
        let temp_host_buffer =
          Me.Buffer.on_device src.stream.device.dev ~length:size
            Me.ResourceOptions.storage_mode_shared
        in
        let cmdbuf, encoder = get_blit_encoder src.stream in
        Me.BlitCommandEncoder.copy_from_buffer encoder ~source_buffer:src_ptr ~source_offset:0
          ~destination_buffer:temp_host_buffer ~destination_offset:0 ~size;
        Me.BlitCommandEncoder.end_encoding encoder;
        commit_buffer src.stream cmdbuf;
        await src.stream;
        (* Wait for copy to host-visible buffer *)
        let temp_contents = Me.Buffer.contents temp_host_buffer in
        Ctypes.memcpy ~dst:(Ctypes.to_voidp host_buf_ptr) ~src:temp_contents ~size
  (* Temp host buffer will be GC'd *)

  let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src =
    let size_in_bytes = Lazy.force tn.Tn.size_in_bytes in
    let cmdbuf, encoder = get_blit_encoder dst.stream in
    (* Use dst stream queue *)
    (* TODO: Add wait logic for src event associated with src_ptr *)
    (* TODO: Add wait logic for dst event associated with dst_ptr if overwriting *)
    (match (into_merge_buffer, dst_ptr) with
    | No, None -> invalid_arg "Metal_backend.device_to_device: missing dst_ptr"
    | No, Some dst_ptr_val ->
        Me.BlitCommandEncoder.copy_from_buffer encoder ~source_buffer:src_ptr ~source_offset:0
          ~destination_buffer:dst_ptr_val ~destination_offset:0 ~size:size_in_bytes
    (* TODO: Update writer event for dst_ptr_val *)
    | Streaming_for task, _ ->
        (* For streaming, conceptually place src_ptr into dst's merge buffer state *)
        (* Actual copy might happen implicitly or via kernel args *)
        dst.stream.merge_buffer := Some { ptr = src_ptr; size_in_bytes };
        dst.stream.updating_for_merge_buffer <- Some (tn, None);
        (* Event set after task *)
        Task.run task
        (* Run the task that uses the merge buffer *)
        (* TODO: After task completes, signal an event and store with updating_for_merge_buffer *)
    | Copy, _ ->
        (* Allocate or reuse merge buffer on dst stream *)
        let merge_buffer =
          match !(dst.stream.merge_buffer) with
          | Some buf when buf.size_in_bytes >= size_in_bytes -> buf
          | _ ->
              let new_buf = alloc_buffer ~size_in_bytes dst.stream in
              dst.stream.merge_buffer := Some new_buf;
              new_buf
        in
        Me.BlitCommandEncoder.copy_from_buffer encoder ~source_buffer:src_ptr ~source_offset:0
          ~destination_buffer:merge_buffer.ptr ~destination_offset:0 ~size:size_in_bytes;
        (* TODO: Update writer event for the merge buffer *)
        dst.stream.updating_for_merge_buffer <- Some (tn, None)
        (* Event set after copy completes *));
    Me.BlitCommandEncoder.end_encoding encoder;
    commit_buffer dst.stream cmdbuf;
    (* TODO: If Copy or Streaming, record event from this cmdbuf and associate with merge buffer
       state *)
    ()

  (* Compilation and Linking *)
  type code = {
    library : Me.Library.t;
    bindings : Indexing.unit_bindings;
    optimized_op : Low_level.optimized; (* Keep original op for linking *)
    name : string; (* Function name *)
  }
  [@@deriving sexp_of]

  type code_batch = {
    library : Me.Library.t;
    bindings : Indexing.unit_bindings;
    optimized_ops : Low_level.optimized option array; (* Keep original ops *)
    names : string option array; (* Function names *)
  }
  [@@deriving sexp_of]

  (* Metal Bindings TODO: Implement C_syntax for MSL *)
  module C_syntax_config (Input : sig
    val procs : Low_level.optimized array
  end) =
  struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]

    let procs = Input.procs
    let use_host_memory = use_host_memory
    let logs_to_stdout = true
    let main_kernel_prefix = "kernel void" (* MSL kernel declaration *)
    let kernel_prep_line = "" (* No equivalent needed? Thread indexing handles this *)
    let includes = [ "<metal_stdlib>"; "#include <metal_compute>" ] (* Basic MSL includes *)

    let typ_of_prec = function
      | Ops.Byte_prec _ -> "uchar"
      | Half_prec _ -> "half"
      | Single_prec _ -> "float"
      | Double_prec _ -> "double" (* Requires enabling double support *)
      | Void_prec -> "void"

    (* Metal Bindings TODO: Define MSL syntax for all Ops *)
    let binop_syntax _prec _v = failwith "MSL binop syntax TBD"
    let unop_syntax _prec _v = failwith "MSL unop syntax TBD"
    let ternop_syntax _prec _v = failwith "MSL ternop syntax TBD"

    let convert_precision ~from ~to_ =
      if Ops.equal_prec from to_ then ("", "") else ("(" ^ typ_of_prec to_ ^ ")(", ")")
  end

  let compile ~name bindings lowered =
    (* failwith "Metal compilation (MSL generation) not yet implemented" *)
    (* Placeholder structure: *)
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = [| lowered |]
    end)) in
    let idx_params = Indexing.bound_symbols bindings in
    let b = Buffer.create 4096 in
    let ppf = Stdlib.Format.formatter_of_buffer b in
    Syntax.print_includes ppf;
    let _params = Syntax.compile_proc ~name ppf idx_params lowered in
    (* Need MSL generation *)
    let msl_source = Buffer.contents b in
    (* Add MSL-specific compilation options *)
    let compile_options = Me.CompileOptions.init () in
    Me.CompileOptions.set_language_version compile_options
      Me.CompileOptions.LanguageVersion.version_3_1;
    let library = Me.Library.on_device (Lazy.force the_device) ~source:msl_source compile_options in
    { library; bindings; optimized_op = lowered; name }
  (* TODO: Add MSL generation *)

  let compile_batch ~names bindings lowereds =
    (* failwith "Metal batch compilation (MSL generation) not yet implemented" *)
    (* Placeholder structure: *)
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = Array.filter_opt lowereds
    end)) in
    (* ... generate combined MSL source ... *)
    let msl_source = Buffer.contents b in
    let compile_options = Me.CompileOptions.init () in
    (* ... set options ... *)
    let library = Me.Library.on_device (Lazy.force the_device) ~source:msl_source compile_options in
    { library; bindings; optimized_ops = lowereds; names = name_opts }
  (* TODO: Add MSL generation *)

  let link prior_context (code : code) ctx_arrays =
    (* failwith "Metal linking not yet implemented" *)
    (* Placeholder structure: *)
    let stream = prior_context.stream in
    let device = stream.device.dev in
    let func = Me.Library.new_function_with_name code.library code.name in
    let pipeline_state = Me.ComputePipelineState.on_device device func in
    let idx_params = Indexing.bound_symbols code.bindings in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map idx_params ~f:(fun s -> (s, ref 0))
    in

    let work () =
      let cmdbuf = Me.CommandQueue.command_buffer stream.runner in
      let encoder = Me.CommandBuffer.compute_command_encoder cmdbuf in
      Me.ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline_state;
      (* Set buffers using ctx_arrays and code.optimized_op.body args *)
      List.iteri code.optimized_op.body.args ~f:(fun i arg ->
          match arg.Tn.op with
          | Param ->
              (* Assume Param maps to buffer *)
              let buffer = Map.find_exn ctx_arrays arg in
              Me.ComputeCommandEncoder.set_buffer encoder buffer 0 i
          | _ -> ()
          (* Handle other args if necessary *));
      (* Calculate grid/threadgroup sizes *)
      let threads_per_group = Me.ComputeCommandEncoder.Size.make ~width:32 ~height:1 ~depth:1 in
      (* Example *)
      let grid_size = Me.ComputeCommandEncoder.Size.make ~width:1024 ~height:1 ~depth:1 in
      (* Example *)
      Me.ComputeCommandEncoder.dispatch_threads encoder ~threads_per_grid:grid_size
        ~threads_per_threadgroup:threads_per_group;
      Me.ComputeCommandEncoder.end_encoding encoder;
      commit_buffer stream cmdbuf
      (* TODO: associate event from this buffer with output nodes *)
    in
    let task =
      Task.Task
        {
          context_lifetime = (pipeline_state, ctx_arrays);
          description = "launch " ^ code.name;
          work;
        }
    in
    (lowered_bindings, task)
  (* TODO: Add MSL generation *)

  let link_batch prior_context (code_batch : code_batch) ctx_arrays_opts =
    failwith "Metal batch linking not yet implemented"
  (* Placeholder structure: Similar to link, but loop through code_batch.names/ops and
     ctx_arrays_opts *)

  let%debug2_sexp get_used_memory (device : device) =
    (* Metal Bindings TODO: Need memory info API *)
    [%log "Metal: get_used_memory not implemented"];
    0

  let get_global_debug_info () = Sexp.List []
  let get_debug_info (_stream : stream) = Sexp.List []
end

open Base
module Tn = Ir.Tnode
module Lazy = Utils.Lazy
module Mtl = Metal
open Ir
open Ir.Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

(* TODO: Add Metal call logging/hook similar to CUDA if needed *)

module Backend_buffer = struct
  type buffer_ptr = Mtl.Buffer.t

  (* Sexp representation for opaque Metal objects *)
  let sexp_of_buffer_ptr (_ptr : buffer_ptr) = Sexp.Atom "<metal_buffer>"

  include Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]
  end)
end

module Device_config = struct
  include Backend_buffer

  type dev = { dev : Mtl.Device.t (* Metal device might implicitly handle context management *) }
  [@@deriving sexp_of]

  type runner = Mtl.CommandQueue.t [@@deriving sexp_of]

  (* Using CommandBuffer as the event seems reasonable, as its completion signifies the work is
     done. *)
  type event = Mtl.CommandBuffer.t [@@deriving sexp_of]

  let sexp_of_event _e = Sexp.Atom "<metal_command_buffer_event>"
  let name = "metal"
end

module Fresh () = struct
  include Backend_impl.Device_types (Device_config)
  open Device_config

  (* Metal on Apple Silicon uses Unified Memory, suggesting true host memory usage. However,
     directly mapping Ctypes pointers to Metal buffers isn't trivial/safe. We create Metal buffers
     and copy. So, setting to None for now. *)
  let use_host_memory = None

  module Alloc_buffer = struct
    include Device_stream

    let buffer_options = Mtl.ResourceOptions.(storage_mode_shared + cpu_cache_mode_write_combined)

    let alloc_buffer ?old_buffer ~size_in_bytes stream =
      match old_buffer with
      | Some ({ size_in_bytes = old_size; _ } as buffer) when size_in_bytes <= old_size -> buffer
      | Some _ (* Buffer exists but too small *) | None ->
          (* Metal uses ARC; old buffer will be released if no longer referenced. *)
          let new_ptr =
            Mtl.Device.new_buffer_with_length stream.device.dev.dev ~length:size_in_bytes
              buffer_options
          in
          { ptr = new_ptr; size_in_bytes }

    let alloc_zero_init_array prec ~dims stream =
      let size_in_bytes =
        (if Array.length dims = 0 then 0 else Array.reduce_exn dims ~f:( * ))
        * Ops.prec_in_bytes prec
      in
      (* Metal buffers allocated this way are not zero-initialized by default. Zeroing would require
         a separate blit command. For now, just allocate. *)
      let new_ptr =
        Mtl.Device.new_buffer_with_length stream.device.dev.dev ~length:size_in_bytes buffer_options
      in
      new_ptr

    (* Metal uses ARC via OCaml's GC finalizers provided by ctypes-foreign. Explicit freeing is not
       typically needed/done here. *)
    let free_buffer = None
  end

  include Backend_impl.Device (Device_stream) (Alloc_buffer)

  let ctx_of (context : context) = context.stream.device.dev.dev (* Return the Metal device *)

  (* --- Initialization --- *)
  let global_config = ref For_parallel_copying
  let initialized = ref false

  let initialize (config : config) : unit =
    initialized := true;
    global_config := config

  let is_initialized () = !initialized

  (* --- Device Management --- *)
  let the_device = Lazy.from_fun Mtl.Device.create_system_default
  let devices = ref [||] (* We'll cache the single default device *)

  let%track3_sexp get_device ~(ordinal : int) : device =
    if ordinal <> 0 then
      invalid_arg [%string "Metal backend currently only supports the default device (ordinal 0)"];
    if Array.is_empty !devices then (
      let metal_dev = Lazy.force the_device in
      let dev_record = { dev = metal_dev } in
      let backend_dev = make_device dev_record ~ordinal:0 in
      (* TODO: Add finalizer if needed, though ARC should handle Objective-C objects *)
      devices := [| Some backend_dev |];
      backend_dev)
    else Option.value_exn !devices.(0) ~message:"Metal device cache error"

  let num_devices () = 1 (* Assume only one default device for now *)

  let%track3_sexp new_stream (device : device) : stream =
    let command_queue = Mtl.Device.new_command_queue device.dev.dev in
    make_stream device command_queue

  let suggested_num_streams _device =
    match !global_config with
    | Only_devices_parallel -> 1
    | For_parallel_copying -> 2 (* Example: 1 compute, 1 copy *)
    | Most_parallel_streams -> 8 (* Placeholder, Metal doesn't expose cores like CUDA *)

  (* --- Synchronization --- *)
  let await stream : unit =
    (* There's no direct stream sync in Metal. We sync the *queue*.
       This waits for all *scheduled* command buffers to complete. *)
    (* Mtl.CommandQueue.waitUntilCompleted() - No such function exists.
       We need to sync individual command buffers (events).
       Awaiting a stream means awaiting the *last submitted event* conceptually.
       This requires tracking the last event per stream. Let's punt for now.
       Users should await specific events (command buffers). *)
    failwith "Metal backend: await stream not yet implemented"

  let is_idle stream =
    (* Similar to await, we check the status of the last submitted command buffer. Requires
       tracking. Return true placeholder. *)
    true
  (* Placeholder *)

  let sync (event : event) = Mtl.CommandBuffer.wait_until_completed event

  let is_done (event : event) =
    match Mtl.CommandBuffer.status event with
    | Completed -> true
    | Error -> (* Raise or log? Let's consider error completed for now *) true
    | _ -> false (* NotEnqueued, Enqueued, Committed *)

  let will_wait_for (context : context) (event : event) =
    (* Metal command buffers execute sequentially on a queue by default. Explicit dependencies
       between queues or events across queues are needed for waits. Simple approach: Commit the
       event's command buffer *before* subsequent work on the context's stream (queue). The queue
       ensures order. If events are on *different* queues, we need MTLSharedEvent. Not implemented
       yet. Assume same queue for now, so this is a no-op for scheduling, but commit is needed. We
       don't commit here, the Task work function does. *)
    ignore (context, event)
  (* No-op for same-queue execution *)

  let all_work stream : event =
    (* Return the *last completed* or *last submitted* command buffer? Let's return a dummy/sentinel
       event for now. Needs proper tracking. *)
    failwith "Metal backend: all_work not yet implemented"

  (* --- Memory Info --- *)
  let get_used_memory device =
    (* Metal doesn't offer a simple API like CUDA's cuMemGetInfo. Could try to query
       currentAllocatedSize on the device, but might be expensive/complex. Return 0 for now. *)
    ignore device;
    0

  (* --- Data Transfer --- *)
  let ensure_command_buffer_and_encoder stream =
    let cmd_buffer = Mtl.CommandQueue.command_buffer stream.runner in
    let blit_encoder = Mtl.CommandBuffer.blit_command_encoder cmd_buffer in
    (cmd_buffer, blit_encoder)

  let commit_blit_encoder cmd_buffer blit_encoder =
    Mtl.BlitCommandEncoder.end_encoding blit_encoder;
    Mtl.CommandBuffer.commit cmd_buffer;
    cmd_buffer (* Return the command buffer as the event *)

  let from_host ~dst_ptr ~dst (hosted : Ndarray.t) =
    let size = Ndarray.size_in_bytes hosted in
    let host_ptr = Ndarray.get_voidptr hosted in
    let dst_contents_ptr = Mtl.Buffer.contents dst_ptr in
    Ctypes_memory_stubs.memcpy ~dst:dst_contents_ptr ~src:host_ptr ~size;
    (* If using shared memory, we might need didModifyRange, but not for StorageModeShared. We still
       need to signal completion via an event. A simple copy doesn't naturally create a command
       buffer. How to synchronize CPU writes with GPU reads? For StorageModeShared, CPU writes are
       visible *eventually*. A NOP blit command might enforce ordering? Or rely on command buffer
       submission order. Let's assume command buffer submission order is sufficient for now. This
       function needs to return *no* event. Syncing happens elsewhere. *)
    ignore dst (* Context currently unused, might be needed for sync later *)

  let to_host ~src_ptr ~src (hosted : Ndarray.t) =
    (* Need to ensure GPU work writing to src_ptr is done *before* CPU reads. *)
    let event =
      match Hashtbl.find src.stream.updating_for (Tnode.find src_ptr) with
      (* Assuming Tnode.find works *)
      | Some e -> e
      | None -> failwith "to_host: No event found for source buffer write completion"
    in
    sync event;

    (* Wait for the GPU write to complete *)
    let size = Ndarray.size_in_bytes hosted in
    let host_ptr = Ndarray.get_voidptr hosted in
    let src_contents_ptr = Mtl.Buffer.contents src_ptr in
    Ctypes_memory_stubs.memcpy ~dst:host_ptr ~src:src_contents_ptr ~size

  let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src =
    let size_in_bytes = Tnode.size_in_bytes_known ~here:[%here] tn in
    let cmd_buffer = Mtl.CommandQueue.command_buffer dst.stream.runner in
    let blit_encoder = Mtl.CommandBuffer.blit_command_encoder cmd_buffer in

    let final_dst_ptr =
      match (into_merge_buffer, dst_ptr) with
      | No, None -> invalid_arg "Metal_backend.device_to_device: missing dst_ptr"
      | No, Some ptr -> ptr
      | Streaming_for _, _ ->
          (* Set merge buffer conceptually. Actual copy might not happen yet. *)
          (* TODO: Track source for streaming merge buffer *)
          dst.stream.merge_buffer := Some { ptr = src_ptr; size_in_bytes };
          (* No blit needed here, the compute kernel will read src_ptr *)
          Mtl.BlitCommandEncoder.end_encoding blit_encoder;
          (* End empty encoder *)
          Mtl.CommandBuffer.commit cmd_buffer;
          (* Commit empty buffer *)
          (* TODO: How to handle task? *)
          failwith "Streaming_for merge buffer not implemented"
      | Copy, _ ->
          (* Allocate or reuse merge buffer *)
          let merge_buf_record =
            Alloc_buffer.alloc_buffer ?old_buffer:!(dst.stream.merge_buffer) ~size_in_bytes
              dst.stream
          in
          dst.stream.merge_buffer := Some merge_buf_record;
          merge_buf_record.ptr
    in

    if phys_equal final_dst_ptr src_ptr then
      (* Copying to self, do nothing *)
      Mtl.BlitCommandEncoder.end_encoding blit_encoder
    else (
      Mtl.BlitCommandEncoder.copy_from_buffer ~self:blit_encoder ~source_buffer:src_ptr
        ~source_offset:0 ~destination_buffer:final_dst_ptr ~destination_offset:0 ~size:size_in_bytes;
      (* TODO: Add synchronization if using StorageModeManaged *)
      (* Mtl.BlitCommandEncoder.synchronize_resource ~self:blit_encoder ~resource:(Obj.magic final_dst_ptr); *)
      Mtl.BlitCommandEncoder.end_encoding blit_encoder);

    Mtl.CommandBuffer.commit cmd_buffer;

    (* Update writer event. The committed buffer *is* the event. *)
    let node_dst = Tnode.find final_dst_ptr in
    (* Assuming Tnode.find works *)
    let node_merge =
      match !dst.stream.merge_buffer with Some b -> Tnode.find b.ptr | None -> node_dst
    in
    (* Get merge node if used *)

    match into_merge_buffer with
    | No -> Hashtbl.set dst.stream.updating_for ~key:node_dst ~data:cmd_buffer
    | Copy -> dst.stream.updating_for_merge_buffer <- Some (node_merge, Some cmd_buffer)
    | Streaming_for _ -> dst.stream.updating_for_merge_buffer <- Some (node_merge, None)
  (* Event comes later *)

  (* --- Compilation and Linking --- *)
  type code = {
    library : Mtl.Library.t;
    func_name : string;
    params : (string * param_source) list;
    bindings : Indexing.unit_bindings;
    traced_store : Low_level.traced_store; (* For debug/info *)
  }
  [@@deriving sexp_of]

  let sexp_of_code c =
    Sexp.List
      [
        Sexp.Atom c.func_name;
        [%sexp_of: (string * param_source) list] c.params;
        Sexp.Atom "<metal_library>";
      ]

  type code_batch = {
    libraries : Mtl.Library.t option array; (* Can batches share a library? *)
    params_and_names : ((string * param_source) list * string) option array;
    bindings : Indexing.unit_bindings;
    traced_stores : Low_level.traced_store option array;
  }
  [@@deriving sexp_of]

  let sexp_of_code_batch c =
    Sexp.List
      [
        Sexp.Atom "code_batch";
        [%sexp_of: bool array] (Array.map c.libraries ~f:Option.is_some);
        [%sexp_of: Indexing.unit_bindings] c.bindings;
      ]

  let metal_compile_options =
    let opts = Mtl.CompileOptions.init () in
    (* Set desired options, e.g., language version *)
    Mtl.CompileOptions.set_language_version opts Mtl.CompileOptions.LanguageVersion.version_3_1;
    (* Mtl.CompileOptions.set_fast_math_enabled opts true; *)
    opts

  let compile ~name bindings ({ Low_level.traced_store; _ } as lowered) =
    (* TODO: Implement C_syntax_config for Metal Shading Language (MSL) *)
    (* TODO: Generate MSL source from Low_level.optimized *)
    let msl_source =
      "// MSL Source Placeholder for " ^ name ^ "\nkernel void " ^ name ^ "() {}\n"
    in

    (* For compilation, we need a device *)
    let device = get_device ~ordinal:0 in

    (* Compile MSL to Metal Library *)
    if Utils.settings.output_debug_files_in_build_directory then (
      let oc = Out_channel.open_text @@ Utils.build_file @@ name ^ ".metal" in
      Stdio.Out_channel.output_string oc msl_source;
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc);

    let library =
      try Mtl.Device.new_library_with_source device.dev.dev ~source:msl_source metal_compile_options
      with Failure msg ->
        failwith [%string "Metal library compilation failed for %{name}: %{msg}"]
    in
    (* TODO: Extract params from the compilation process/lowered IR *)
    let params = [] in
    { library; func_name = name; params; bindings; traced_store }

  let compile_batch ~names bindings lowereds =
    (* TODO: Implement batch compilation - potentially generating one MSL file *)
    let device = get_device ~ordinal:0 in
    let libraries =
      Array.map lowereds
        ~f:(Option.map ~f:(fun l -> (compile ~name:"placeholder" bindings l).library))
    in
    let params_and_names = Array.map names ~f:(Option.map ~f:(fun n -> ([], n))) in
    let traced_stores = Array.map lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store)) in
    { libraries; params_and_names; bindings; traced_stores }
  (* Placeholder implementation *)

  let link prior_context code ctx_arrays =
    let device = prior_context.stream.device.dev.dev in
    let work () =
      (* 1. Get function & create pipeline state *)
      let mtl_function = Mtl.Library.new_function_with_name code.library code.func_name in
      let pipeline_state =
        Mtl.Device.new_compute_pipeline_state_with_function device mtl_function
      in

      (* 2. Create command buffer and encoder *)
      let command_buffer = Mtl.CommandQueue.command_buffer prior_context.stream.runner in
      let encoder = Mtl.CommandBuffer.compute_command_encoder command_buffer in
      Mtl.CommandEncoder.set_label encoder code.func_name;

      (* 3. Set pipeline state *)
      Mtl.ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline_state;

      (* 4. Set buffers *)
      List.iteri code.params ~f:(fun index (_name, param_source) ->
          match param_source with
          | Param_ptr tn ->
              let buffer = Map.find_exn ctx_arrays tn in
              Mtl.ComputeCommandEncoder.set_buffer encoder buffer 0 index
          | Merge_buffer ->
              let merge_buf = Option.value_exn ~here:[%here] !(prior_context.stream.merge_buffer) in
              Mtl.ComputeCommandEncoder.set_buffer encoder merge_buf.ptr 0 index
          | Log_file_name | Static_idx _ -> () (* Handled by kernel source generation *));

      (* 5. Dispatch threads - TODO: Calculate grid/threadgroup sizes based on bindings *)
      let threadgroup_size =
        Mtl.ComputePipelineState.max_total_threads_per_threadgroup pipeline_state
      in
      let threads_per_threadgroup =
        Mtl.Size.make ~width:(Unsigned.ULong.to_int threadgroup_size) ~height:1 ~depth:1
      in
      (* Example *)
      let grid_size = Mtl.Size.make ~width:1024 ~height:1 ~depth:1 in
      (* Example size *)

      Mtl.ComputeCommandEncoder.dispatch_threads encoder ~threads_per_grid:!@grid_size
        ~threads_per_threadgroup:!@threads_per_threadgroup;

      (* 6. End encoding and commit *)
      Mtl.ComputeCommandEncoder.end_encoding encoder;
      Mtl.CommandBuffer.commit command_buffer;

      (* 7. Update writer events *)
      (* The committed buffer *is* the event. *)
      (* TODO: Identify output nodes accurately *)
      Map.iteri ctx_arrays ~f:(fun ~key:tnode ~data:_ ->
          Hashtbl.set prior_context.stream.updating_for ~key:tnode ~data:command_buffer);
      Option.iter !prior_context.stream.merge_buffer ~f:(fun buf ->
          match prior_context.stream.updating_for_merge_buffer with
          | Some (tn, None) when Tnode.equal tn (Tnode.find buf.ptr) ->
              prior_context.stream.updating_for_merge_buffer <- Some (tn, Some command_buffer)
          | _ -> () (* Only update if it was waiting for streaming event *))
    in
    let task =
      Task.(Task { context_lifetime = (); description = "Metal kernel " ^ code.func_name; work })
    in
    (* TODO: Return proper bindings *)
    let lowered_bindings = [] in
    (lowered_bindings, task)

  let link_batch prior_context code_batch ctx_arrays_opts =
    (* TODO: Implement batch linking *)
    let lowered_bindings = [] in
    let tasks = Array.map ctx_arrays_opts ~f:(fun _ -> None) in
    (* Placeholder *)
    (lowered_bindings, tasks)

  (* --- Debug Info --- *)
  let get_global_debug_info () = Sexp.List [ Sexp.Atom "metal_global_debug_placeholder" ]

  let get_debug_info stream =
    Sexp.List
      [
        Sexp.Atom "metal_stream_debug";
        Sexp.List [ Sexp.Atom "stream_id"; [%sexp_of: int] stream.stream_id ];
        Sexp.List [ Sexp.Atom "device_ordinal"; [%sexp_of: int] stream.device.ordinal ];
      ]
end

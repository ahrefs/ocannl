(* arrayjit/lib/metal_backend.ml *)
open Base
open Ir
module Tn = Tnode
module Lazy = Utils.Lazy
module Me = Metal
open Backend_intf
module Impl = Backend_impl

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

type ullong = Unsigned.ULLong.t
let sexp_of_ullong x = Sexp.Atom (Unsigned.ULLong.to_string x)

(* TODO: Add Metal-specific debug hooks if needed *)

module Backend_buffer = struct
  (* A Metal buffer object *)
  type buffer_ptr = Me.Buffer.t [@@deriving sexp_of]

  include Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]
  end)
end

module Device_config = struct
  include Backend_buffer

  (* A Metal device object *)
  type dev = Me.Device.t [@@deriving sexp_of]

  (* A Metal command queue *)
  type runner = Me.CommandQueue.t [@@deriving sexp_of]

  (* A Metal shared event and the value it should reach *)
  type event = Me.SharedEvent.t * ullong [@@deriving sexp_of]

  let name = "metal"

  (* Metal on Apple Silicon has unified memory *)
  let use_host_memory =
    Some
      (fun ptr ->
        let length = Bigarray.Array1.dim (Ctypes.bigarray_of_ptr Ctypes.array1 ptr Bigarray.Char) in
        (* Assuming default device for this mapping. This might need refinement if multiple devices
           were supported. *)
        let device = Me.Device.create_system_default () in
        (* Create a buffer sharing the host memory without copying. The lifetime is tied to the
           OCaml GC via the payload type unless a custom deallocator is used. *)
        Me.Buffer.on_device_with_bytes_no_copy device ~bytes:(Ctypes.to_voidp ptr) ~length
          Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_write_combined)
          ~deallocator:(fun () -> [%log "Host memory buffer deallocated"]))
end

module Device_stream = Impl.Device_types (Device_config)

module Alloc_buffer = struct
  open Device_stream
  open Device_config

  let%track7_sexp alloc_buffer ?old_buffer ~size_in_bytes stream =
    let device = stream.device.dev in
    let options = Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_write_combined) in
    match old_buffer with
    | Some ({ size_in_bytes = old_size; _ } as buffer) when size_in_bytes <= old_size ->
        (* Can reuse existing buffer if large enough *)
        buffer
    | Some _old_buffer ->
        (* TODO: Could potentially reuse the old ptr if storage options match and no custom
           deallocator? For simplicity, always reallocate for now. Metal's ARC should handle freeing
           the old buffer when its payload record is GC'd. *)
        { ptr = Me.Buffer.on_device device ~length:size_in_bytes options; size_in_bytes }
    | None -> { ptr = Me.Buffer.on_device device ~length:size_in_bytes options; size_in_bytes }

  let%track7_sexp alloc_zero_init_array prec ~dims stream =
    let size_in_bytes =
      (if Array.length dims = 0 then 1 (* Avoid 0-size alloc *) else Array.reduce_exn dims ~f:( * ))
      * Ops.prec_in_bytes prec
    in
    let device = stream.device.dev in
    let options = Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_write_combined) in
    let buffer = Me.Buffer.on_device device ~length:size_in_bytes options in
    (* Schedule a fill operation to zero the buffer *)
    (* TODO: This requires a command buffer and encoder. Ideally, this should return a task or be
       integrated differently. For now, we rely on Low_level.Zero_out being handled in the kernel or
       by explicit initialization. We return the buffer pointer directly. *)
    buffer

  (* Metal uses ARC, explicit freeing via this API is not the standard way. The payload record in
     metal.ml handles GC. *)
  let free_buffer = None
end

module Fresh () : Impl.Lowered_backend = struct
  include Device_stream
  include Impl.Device (Device_stream) (Alloc_buffer)

  let use_host_memory = Device_config.use_host_memory

  (* --- State Management --- *)
  let initialized = ref false
  let devices = ref [||]
  let global_config = ref Backend_intf.For_parallel_copying (* Default config *)
  let next_event_value = Atomic.make Unsigned.ULLong.zero

  let get_next_event_value () =
    let v = Atomic.get next_event_value in
    Atomic.set next_event_value Unsigned.ULLong.(v + one);
    Unsigned.ULLong.(v + one)

  (* --- Initialization --- *)
  let initialize config =
    global_config := config;
    (* Metal doesn't require explicit library init like CUDA's cuInit *)
    initialized := true;
    [%log "Metal backend initialized"]

  let is_initialized () = !initialized

  (* --- Device Handling --- *)
  let num_devices () =
    (* MTLCopyAllDevices would be needed here. For now, assume 1 default device. *)
    1

  let get_device ~ordinal =
    if ordinal <> 0 then
      invalid_arg [%string "Metal backend currently only supports device ordinal 0"];
    if Array.is_empty !devices then (
      let dev = Me.Device.create_system_default () in
      let device = make_device dev ~ordinal:0 in
      Stdlib.Gc.finalise
        (fun d -> [%log "Finalizing Metal device"; ignore d]) (* TODO: Proper cleanup? *)
        device;
      devices := [| Some device |];
      device)
    else Option.value_exn !devices.(0)

  let suggested_num_streams _device =
    (* Metal doesn't expose fine-grained parallelism details like CUDA. *)
    match !global_config with Only_devices_parallel -> 1 | _ -> 2 (* Host + 1 GPU queue *)

  let new_stream device =
    let runner = Me.CommandQueue.on_device device.dev in
    make_stream device runner

  (* --- Event Handling --- *)

  let sync ((event, value) : event) =
    [%log_result "Waiting for event value", value];
    (* Use a very long timeout (effectively infinite for practical purposes) *)
    let timeout_ms = Unsigned.ULLong.max_int in
    let signaled = Me.SharedEvent.wait_until_signaled_value event ~value ~timeout_ms in
    if not signaled then [%log "Warning: Event sync timed out (should be rare)"];
    [%log_result "Event sync completed for value", value]

  let is_done ((event, value) : event) =
    let current_value = Me.SharedEvent.get_signaled_value event in
    Unsigned.ULLong.(current_value >= value)

  (* --- Stream/Queue Operations --- *)

  (* Helper to get or create a shared event for a stream *)
  let stream_event_map : (int, Me.SharedEvent.t) Hashtbl.t = Hashtbl.create (module Int)

  let get_stream_event stream =
    Hashtbl.find_or_add stream_event_map stream.stream_id ~default:(fun () ->
        Me.SharedEvent.on_device stream.device.dev)

  (* Helper to manage command buffers per stream task *)
  (* This is tricky: Linking creates a task, but event/copy operations need to be encoded *before*
     or *after* that task's command buffer. This suggests a more integrated command buffer
     management or a different event strategy. *)
  (* Approach: Submit small, dedicated command buffers for sync/copy operations. *)

  let submit_sync_command stream (f : Me.CommandBuffer.t -> unit) =
    let cmdbuf = Me.CommandBuffer.on_queue stream.runner in
    f cmdbuf;
    Me.CommandBuffer.commit cmdbuf;
    (* We might need to wait for *this* specific buffer if subsequent CPU operations depend on it.*)
    Me.CommandBuffer.wait_until_completed cmdbuf

  let will_wait_for context ((event, value) : event) =
    [%log_result "Scheduling wait for event value", value, "on stream", context.stream.stream_id];
    submit_sync_command context.stream (fun cmdbuf ->
        Me.CommandBuffer.encode_wait_for_event cmdbuf (Me.SharedEvent.super event) value)

  let all_work stream =
    let event = get_stream_event stream in
    let value = get_next_event_value () in
    [%log_result "Scheduling signal event value", value, "on stream", stream.stream_id];
    submit_sync_command stream (fun cmdbuf ->
        Me.CommandBuffer.encode_signal_event cmdbuf (Me.SharedEvent.super event) value);
    (event, value)

  let await stream =
    [%log "Awaiting stream", stream.stream_id];
    (* Create a command buffer, enqueue a signal, commit, and wait for *that* signal. *)
    let event, value = all_work stream in
    sync (event, value);
    [%log "Stream awaited", stream.stream_id]

  let is_idle stream =
    (* Check if the latest event signaled by this stream is done. This assumes events are signaled
       monotonically. *)
    let event = get_stream_event stream in
    let last_signaled = Me.SharedEvent.get_signaled_value event in
    let expected_next = Atomic.get next_event_value in (* This is global, needs refinement *)
    (* Heuristic: if the last signaled value is close to the globally expected next one, assume not idle?
       A better way is needed. Querying command queue status isn't directly available.
       Check the status of the *last submitted command buffer*? Requires tracking it.
       For now, use event query as a proxy. If the last known event value for this stream is done, assume idle.
       This requires tracking the last event value *per stream*.
    *)
    (* Let's track last event per stream *)
    let last_event_value = ref Unsigned.ULLong.zero in (* Needs to be per stream state *)
    (* TODO: Store last_event_value in stream ref *)
    is_done (event, !last_event_value) (* Placeholder logic *)

  (* --- Memory Info --- *)
  let get_used_memory device =
    (* Metal API via metal.ml doesn't expose detailed memory usage. *)
    let attrs = Me.Device.get_attributes device.dev in
    (* Return recommended working set size as a very rough proxy? Or 0? *)
    Int64.to_int_exn @@ Unsigned.ULLong.to_int64 attrs.recommended_max_working_set_size
  (* Return 0 *)

  let get_global_debug_info () = Sexp.Atom "Metal global debug info not implemented"
  let get_debug_info stream = Sexp.message "Metal stream debug info" [ ("stream_id", [%sexp_of: int] stream.stream_id) ]

  (* --- Data Transfer --- *)

  (* Helper for blit operations *)
  let submit_blit_command stream (f : Me.BlitCommandEncoder.t -> unit) =
     let cmdbuf = Me.CommandBuffer.on_queue stream.runner in
     let encoder = Me.BlitCommandEncoder.on_buffer cmdbuf in
     f encoder;
     Me.BlitCommandEncoder.end_encoding encoder;
     Me.CommandBuffer.commit cmdbuf;
     (* Wait immediately for blit commands to simplify synchronization logic for now *)
     Me.CommandBuffer.wait_until_completed cmdbuf

  let from_host ~dst_ptr ~dst hosted =
    let size_in_bytes = Ndarray.size_in_bytes hosted in
    let host_ptr = Ndarray.get_voidptr hosted in
    [%log_result "Copying from host", size_in_bytes, "bytes"];
    if phys_equal dst_ptr.ptr host_ptr then [%log "Skipping host copy (pointers identical)"]
    else
      let contents_ptr = Me.Buffer.contents dst_ptr in
      if Ctypes.is_null contents_ptr then failwith "Buffer contents pointer is null";
      Ctypes_memory_stubs.memcpy ~dst:contents_ptr ~src:host_ptr ~size:size_in_bytes;
      (* Inform Metal that the buffer range was modified by the CPU *)
      Me.Buffer.did_modify_range dst_ptr { location = 0; length = size_in_bytes }

  let to_host ~src_ptr ~src hosted =
    let size_in_bytes = Ndarray.size_in_bytes hosted in
    let host_ptr = Ndarray.get_voidptr hosted in
    [%log_result "Copying to host", size_in_bytes, "bytes"];
     if phys_equal src_ptr.ptr host_ptr then [%log "Skipping host copy (pointers identical)"]
     else
      (* Ensure GPU work is done before reading *)
      await src.stream;
      let contents_ptr = Me.Buffer.contents src_ptr in
       if Ctypes.is_null contents_ptr then failwith "Buffer contents pointer is null";
      Ctypes_memory_stubs.memcpy ~dst:host_ptr ~src:contents_ptr ~size:size_in_bytes

  let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src =
    let size_in_bytes = Lazy.force tn.Tn.size_in_bytes in
    let same_device = dst.stream.device.ordinal = src.stream.device.ordinal in
    if not same_device then
      failwith "Metal backend does not support device-to-device copy across different devices";

    match into_merge_buffer with
    | No -> (
        match dst_ptr with
        | None -> invalid_arg "Metal_backend.device_to_device: No and missing dst_ptr"
        | Some dst_ptr_val ->
            [%log_result "Copying D2D", size_in_bytes, "bytes"];
            (* Use BlitCommandEncoder for D2D copy *)
            submit_blit_command dst.stream (fun encoder ->
                Me.BlitCommandEncoder.copy_from_buffer encoder ~source_buffer:src_ptr
                  ~source_offset:0 ~destination_buffer:dst_ptr_val ~destination_offset:0
                  ~size:size_in_bytes)
            )
    | Streaming_for task ->
        (* Just set the merge buffer reference; assumes buffer lifetime is managed correctly. *)
        dst.stream.merge_buffer := Some { ptr = src_ptr; size_in_bytes };
         (* Schedule the task that uses the merge buffer *)
        Impl.schedule_task dst.stream task;
        (* TODO: Need event management for streaming *)
        ()
    | Copy ->
        [%log_result "Copying D2D to Merge Buffer", size_in_bytes, "bytes"];
        opt_alloc_merge_buffer ~size_in_bytes dst.stream.device.dev dst.stream;
        let merge_buf = Option.value_exn ~here:[%here] !(dst.stream.merge_buffer) in
        submit_blit_command dst.stream (fun encoder ->
            Me.BlitCommandEncoder.copy_from_buffer encoder ~source_buffer:src_ptr
              ~source_offset:0 ~destination_buffer:merge_buf.ptr ~destination_offset:0
              ~size:size_in_bytes)
        (* TODO: Need event management for merge buffer copy completion *)

  (* --- Compilation --- *)
  type compiled_kernel = {
    name : string;
    params : (string * param_source) list;
    pso : Me.ComputePipelineState.t;
    bindings : Indexing.unit_bindings;
    (* Add threadgroup info if needed *)
  }
  [@@deriving sexp_of]

  type code = {
    msl_source : string;
    kernel_name : string;
    params : (string * param_source) list;
    bindings : Indexing.unit_bindings;
    traced_store : Low_level.traced_store; (* Keep for debug/linking info *)
  }
  [@@deriving sexp_of]

  type code_batch = {
    msl_source : string; (* Combined source for all kernels *)
    kernels : (code option) array; (* Info per kernel *)
    bindings : Indexing.unit_bindings;
  }
  [@@deriving sexp_of]

  (* --- C Syntax Config for Metal --- *)
   module C_syntax_config (Input : sig val procs : Low_level.optimized array end) = struct
     type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]

     let procs = Input.procs
     let use_host_memory = use_host_memory
     let logs_to_stdout = true (* Metal printf goes to stdout *)
     let main_kernel_prefix = "kernel"
     let kernel_prep_line =
        (* Basic single thread execution for now *)
       "if (thread_position_in_grid.x != 0 || thread_position_in_grid.y != 0 || thread_position_in_grid.z != 0) { return; }"

     let includes = [ "#include <metal_stdlib>"; "using namespace metal;" ]

     let typ_of_prec = function
       | Ops.Byte_prec _ -> "uchar"
       | Half_prec _ -> "half"
       | Single_prec _ -> "float"
       | Double_prec _ -> (* Metal doesn't fully support double in kernels by default *)
           "float" (* Fallback or error? Check Metal spec. Let's fallback to float *)
       | Void_prec -> "void"

     let binop_syntax prec v =
       let standard_op op = ("(", " " ^ op, ")") in
       match (v, prec) with
       | Ops.Arg1, _ | Arg2, _ -> invalid_arg "C_syntax_config: Arg1/Arg2 not operators"
       | _, Ops.Void_prec -> invalid_arg "C_syntax_config: Void precision binop"
       | Add, _ -> standard_op "+"
       | Sub, _ -> standard_op "-"
       | Mul, _ -> standard_op "*"
       | Div, _ -> standard_op "/"
       | ToPowOf, Ops.Half_prec _ -> ("pow(", ",", ")") (* half has pow *)
       | ToPowOf, _ -> ("pow(", ",", ")") (* float/double (fallback) has pow *)
       | Relu_gate, Ops.Byte_prec _ -> ("(", " > 0 ?", " : 0)")
       | Relu_gate, _ -> ("(", " > 0.0h ?", " : half(0.0h))") (* Assuming half/float *)
       | Satur01_gate, Ops.Byte_prec _ -> ("(abs(", ") > 0 ? 0 : (", "))") (* Needs check *)
       | Satur01_gate, _ -> ("(fabs(trunc(", ")) > 0.0h ? half(0.0h) : (", "))")
       | Max, _ -> ("max(", ",", ")")
       | Min, _ -> ("min(", ",", ")")
       | Mod, Ops.Byte_prec _ -> standard_op "%"
       | Mod, _ -> ("fmod(", ",", ")") (* MSL has fmod *)
       | Cmplt, _ -> standard_op "<"
       | Cmpne, _ -> standard_op "!="
       | Cmpeq, _ -> standard_op "=="
       | Or, _ -> standard_op "||"
       | And, _ -> standard_op "&&"

      let unop_syntax prec v =
       let standard_fn fn = (fn ^ "(", ")") in
       match (v, prec) with
       | Ops.Identity, _ -> ("", "")
       | Relu, _ -> ("max(", ", 0.0h)") (* Generic max should work *)
       | Satur01, _ -> ("clamp(", ", 0.0h, 1.0h)") (* MSL clamp *)
       | Exp, _ -> standard_fn "exp"
       | Log, _ -> standard_fn "log"
       | Exp2, _ -> standard_fn "exp2"
       | Log2, _ -> standard_fn "log2"
       | Sin, _ -> standard_fn "sin"
       | Cos, _ -> standard_fn "cos"
       | Sqrt, _ -> standard_fn "sqrt"
       | Recip, _ -> ("(1.0h / ", ")") (* half division *)
       | Recip_sqrt, _ -> standard_fn "rsqrt" (* MSL rsqrt *)
       | Neg, _ -> ("(-", ")")
       | Tanh_approx, _ -> standard_fn "tanh" (* MSL tanh *)
       | Not, _ -> ("(!", ")") (* Logical not? Or bitwise? Assuming logical. *)

     let ternop_syntax prec v =
       match (v, prec) with
       | Ops.Where, _ -> ("(", " ? ", " : ", ")") (* Standard C ternary *)
       | FMA, _ -> ("fma(", ", ", ", ", ")") (* MSL fma *)

     let convert_precision ~from ~to_ =
       match (from, to_) with
        (* No-op conversion *)
       | (p1, p2) when Ops.equal_prec p1 p2 -> ("", "")
       | (_, Ops.Void_prec) -> ("", "") (* Cast to void? *)
       | (_, _) -> ("(" ^ typ_of_prec to_ ^ ")(", ")") (* C-style cast *)

   end

  let compile ~name bindings ({ Low_level.traced_store; llc; merge_node } as lowered) =
    let module Syntax = C_syntax_config (struct let procs = [| lowered |] end) in
    let module Pp = C_syntax.C_syntax (Syntax) in
    let idx_params = Indexing.bound_symbols bindings in
    let b = Buffer.create 4096 in
    let ppf = Stdlib.Format.formatter_of_buffer b in
    Pp.print_includes ppf;
    (* Add thread_position_in_grid etc. to parameter list? No, they are built-in. *)
    let params = Pp.compile_proc ~name ppf idx_params lowered in
    Stdlib.Format.pp_print_flush ppf ();
    let msl_source = Buffer.contents b in
    { msl_source; kernel_name = name; params; bindings; traced_store }

  let compile_batch ~names bindings lowereds =
     let module Syntax = C_syntax_config (struct let procs = Array.filter_opt lowereds end) in
     let module Pp = C_syntax.C_syntax (Syntax) in
     let idx_params = Indexing.bound_symbols bindings in
     let b = Buffer.create 4096 in
     let ppf = Stdlib.Format.formatter_of_buffer b in
     Pp.print_includes ppf;
     let kernels =
       Array.map2_exn names lowereds ~f:(fun name_opt lowered_opt ->
           Option.both name_opt lowered_opt
           |> Option.map ~f:(fun (name, lowered) ->
                  let params = Pp.compile_proc ~name ppf idx_params lowered in
                  {
                    msl_source = ""; (* Source is combined *)
                    kernel_name = name;
                    params;
                    bindings;
                    traced_store = lowered.traced_store;
                  }))
     in
     Stdlib.Format.pp_print_flush ppf ();
     let msl_source = Buffer.contents b in
     (* Update source in individual kernel infos - slightly redundant *)
     Array.iter kernels ~f:(Option.iter ~f:(fun k -> ignore k)); (* Side effect dummy *)
     let kernels = Array.map kernels ~f:(Option.map ~f:(fun k -> { k with msl_source })) in
     { msl_source; kernels; bindings }

  (* --- Linking --- *)
  let link_kernel prior_context (kernel_info : code) ctx_arrays =
     let device = prior_context.stream.device.dev in
     (* Compile MSL source to Library *)
     (* TODO: Cache libraries based on source? *)
     let compile_options = Me.CompileOptions.init () in
      (* Set desired Metal language version if needed, e.g.: *)
      (* Me.CompileOptions.(set_language_version compile_options LanguageVersion.version_3_1); *)
     let library = Me.Library.on_device device ~source:kernel_info.msl_source compile_options in

     (* Get Function *)
     let func = Me.Library.new_function_with_name library kernel_info.kernel_name in

     (* Create Compute Pipeline State *)
     let pso, _reflection = Me.ComputePipelineState.on_device_with_function device func in

     let compiled_kernel =
       { name = kernel_info.kernel_name; params = kernel_info.params; pso; bindings = kernel_info.bindings }
     in

     let lowered_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols kernel_info.bindings) ~f:(fun s -> (s, ref 0))
     in

     (* --- Create Task --- *)
     let work () : unit =
       [%log_result "Launching Metal kernel", compiled_kernel.name];
       let stream = prior_context.stream in
       let cmdbuf = Me.CommandBuffer.on_queue stream.runner in
       let encoder = Me.ComputeCommandEncoder.on_buffer cmdbuf in

       Me.ComputeCommandEncoder.set_compute_pipeline_state encoder compiled_kernel.pso;

       (* Set arguments *)
       List.iteri compiled_kernel.params ~f:(fun index (_p_name, param_source) ->
           match param_source with
           | Param_ptr tn ->
               let buffer = Map.find_exn ctx_arrays tn in
               Me.ComputeCommandEncoder.set_buffer encoder ~index buffer
           | Merge_buffer ->
               let buffer = Option.value_exn ~here:[%here] !(stream.merge_buffer) in
               Me.ComputeCommandEncoder.set_buffer encoder ~index buffer.ptr
           | Static_idx s ->
               let value_ref = List.Assoc.find_exn lowered_bindings ~equal:Indexing.equal_static_symbol s in
               let value = !value_ref in
               (* Pass integer index using setBytes *)
               let idx_ptr = Ctypes.allocate Ctypes.int value in
               Me.ComputeCommandEncoder.set_bytes encoder ~bytes:(Ctypes.to_voidp idx_ptr)
                 ~length:(Ctypes.sizeof Ctypes.int) ~index
           | Log_file_name ->
               (* Metal printf goes to stdout, log_id passed differently if needed *)
               () );

       (* Dispatch - Use simple 1 thread for now *)
       let threads_per_group = Me.Size.make ~width:1 ~height:1 ~depth:1 in
       let groups_per_grid = Me.Size.make ~width:1 ~height:1 ~depth:1 in
       Me.ComputeCommandEncoder.dispatch_threadgroups encoder ~threadgroups_per_grid:groups_per_grid
         ~threads_per_threadgroup:threads_per_group;

       Me.ComputeCommandEncoder.end_encoding encoder;
       Me.CommandBuffer.commit cmdbuf;
       [%log "Metal kernel committed", compiled_kernel.name]
     in

     let task = Task.Task {
       context_lifetime = (pso, library, compile_options, ctx_arrays); (* Keep PSO and Library alive *)
       description = "launch " ^ compiled_kernel.name;
       work;
     } in
     (lowered_bindings, task)

  let link prior_context code ctx_arrays = link_kernel prior_context code ctx_arrays

  let link_batch prior_context code_batch ctx_arrays_opts =
     let device = prior_context.stream.device.dev in
     let compile_options = Me.CompileOptions.init () in
     (* Compile the combined MSL source once *)
     let library = Me.Library.on_device device ~source:code_batch.msl_source compile_options in

     let lowered_bindings : Indexing.lowered_bindings =
        List.map (Indexing.bound_symbols code_batch.bindings) ~f:(fun s -> (s, ref 0))
     in

     let tasks =
       Array.mapi code_batch.kernels ~f:(fun i kernel_opt ->
           Option.both kernel_opt ctx_arrays_opts.(i)
           |> Option.map ~f:(fun (kernel_info, ctx_arrays) ->
                  let func = Me.Library.new_function_with_name library kernel_info.kernel_name in
                  let pso, _ = Me.ComputePipelineState.on_device_with_function device func in
                  let compiled_kernel =
                    { name = kernel_info.kernel_name; params = kernel_info.params; pso; bindings = kernel_info.bindings }
                  in
                  (* Create Task - duplicated logic from link_kernel *)
                   let work () : unit =
                     [%log_result "Launching Metal kernel", compiled_kernel.name];
                     let stream = prior_context.stream in
                     let cmdbuf = Me.CommandBuffer.on_queue stream.runner in
                     let encoder = Me.ComputeCommandEncoder.on_buffer cmdbuf in
                     Me.ComputeCommandEncoder.set_compute_pipeline_state encoder compiled_kernel.pso;
                     List.iteri compiled_kernel.params ~f:(fun index (_p_name, param_source) ->
                         match param_source with
                         | Param_ptr tn ->
                             let buffer = Map.find_exn ctx_arrays tn in
                             Me.ComputeCommandEncoder.set_buffer encoder ~index buffer
                         | Merge_buffer ->
                             let buffer = Option.value_exn ~here:[%here] !(stream.merge_buffer) in
                             Me.ComputeCommandEncoder.set_buffer encoder ~index buffer.ptr
                         | Static_idx s ->
                             let value_ref = List.Assoc.find_exn lowered_bindings ~equal:Indexing.equal_static_symbol s in
                             let value = !value_ref in
                             let idx_ptr = Ctypes.allocate Ctypes.int value in
                             Me.ComputeCommandEncoder.set_bytes encoder ~bytes:(Ctypes.to_voidp idx_ptr)
                               ~length:(Ctypes.sizeof Ctypes.int) ~index
                          | Log_file_name -> () );
                     let threads_per_group = Me.Size.make ~width:1 ~height:1 ~depth:1 in
                     let groups_per_grid = Me.Size.make ~width:1 ~height:1 ~depth:1 in
                     Me.ComputeCommandEncoder.dispatch_threadgroups encoder ~threadgroups_per_grid:groups_per_grid
                       ~threads_per_threadgroup:threads_per_group;
                     Me.ComputeCommandEncoder.end_encoding encoder;
                     Me.CommandBuffer.commit cmdbuf;
                     [%log "Metal kernel committed", compiled_kernel.name]
                   in
                   Task.Task {
                     context_lifetime = (pso, library, compile_options, ctx_arrays);
                     description = "launch " ^ compiled_kernel.name;
                     work;
                   }
           ))
     in
     (lowered_bindings, tasks)

end
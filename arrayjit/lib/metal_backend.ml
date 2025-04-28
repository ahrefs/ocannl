open Base
open Ir
module Tn = Tnode
module Lazy = Utils.Lazy
module Me = Metal (* Alias for Metal module *)
open Backend_intf
module Impl = Backend_impl (* Alias for Backend_impl *)

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

type ullong = Unsigned.ULLong.t
let sexp_of_ullong x = Sexp.Atom (Unsigned.ULLong.to_string x)

module Backend_buffer = struct
  type buffer_ptr = Me.Buffer.t

  let sexp_of_buffer_ptr ptr = Sexp.Atom (Me.Buffer.get_label ptr)

  include Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]
  end)
end

module Device_config = struct
  include Backend_buffer

  type dev = { dev : Me.Device.t; primary_context : Me.CommandQueue.t } [@@deriving sexp_of]
  type runner = Me.CommandQueue.t [@@deriving sexp_of]
  type event = { event : Me.SharedEvent.t; value : ullong } [@@deriving sexp_of]

  let name = "metal"
end

module Device_stream = Backend_impl.Device_types (Device_config)
open Device_config

let set_ctx ctx =
  (* Metal doesn't need context switching like CUDA, this is a no-op *)
  ()

module Alloc_buffer = struct
  include Device_stream

  (* Metal has unified memory on Apple Silicon, so we can use host memory *)
  let use_host_memory = Some (fun ptr -> 
    (* Need to create a Metal buffer that wraps the host memory *)
    let device = Me.Device.create_system_default () in
    let length = 0 (* We don't know the size from just the pointer, will be handled elsewhere *) in
    Me.Buffer.on_device_with_bytes_no_copy device ~bytes:ptr ~length ~deallocator:None
      Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache)
  )

  let alloc_buffer ?old_buffer ~size_in_bytes stream =
    match old_buffer with
    | Some ({ size_in_bytes = old_size; _ } as buffer) when size_in_bytes <= old_size -> buffer
    | Some { ptr; _ } ->
        (* We don't need to free explicitly as Metal's ARC will handle it *)
        { ptr = Me.Buffer.on_device stream.device.dev ~length:size_in_bytes 
            Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache); 
          size_in_bytes }
    | None ->
        { ptr = Me.Buffer.on_device stream.device.dev ~length:size_in_bytes 
            Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache); 
          size_in_bytes }

  let alloc_zero_init_array prec ~dims stream =
    let size_in_bytes =
      (if Array.length dims = 0 then 0 else Array.reduce_exn dims ~f:( * )) * Ops.prec_in_bytes prec
    in
    let buffer = Me.Buffer.on_device stream.device.dev ~length:size_in_bytes 
      Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache) in
    
    (* Zero initialize the buffer using a blit command encoder *)
    let command_buffer = Me.CommandBuffer.on_queue stream.runner in
    let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
    Me.BlitCommandEncoder.fill_buffer blit_encoder buffer 
      (Me.Range.make ~location:0 ~length:size_in_bytes) ~value:0;
    Me.BlitCommandEncoder.end_encoding blit_encoder;
    Me.CommandBuffer.commit command_buffer;
    Me.CommandBuffer.wait_until_completed command_buffer;
    
    buffer

  (* Metal's ARC handles memory management, but we provide a way to manually free if needed *)
  let free_buffer = None
end

(* [initialized_devices] never forgets its entries. *)
let initialized_devices = Hash_set.create (module Int)
let metal_initialized = ref false

module Fresh (Config : sig
  val config : Ir.Backend_intf.config
end) =
struct
  include Backend_impl.Device (Device_stream) (Alloc_buffer)

  let use_host_memory = Alloc_buffer.use_host_memory
  let ctx_of (context : context) = context.stream.device.dev.primary_context
  
  let is_done event = 
    (* Metal SharedEvent's signaled value must be >= our threshold *)
    let current_value = Me.SharedEvent.get_signaled_value event.event in
    Unsigned.ULLong.compare current_value event.value >= 0
  
  let will_wait_for context event =
    (* Encode waiting for the event in a command buffer *)
    let command_buffer = Me.CommandBuffer.on_queue context.stream.runner in
    Me.CommandBuffer.encode_wait_for_event command_buffer event.event event.value;
    Me.CommandBuffer.commit command_buffer

  let sync event = 
    (* Wait for the event to be signaled with the specified value *)
    let timeout = Unsigned.ULLong.of_int 0xFFFFFFFF in (* ~infinite timeout *)
    let _ = Me.SharedEvent.wait_until_signaled_value event.event ~value:event.value ~timeout_ms:timeout in
    ()

  let all_work stream = 
    (* Create a new shared event and signal it at the end of all pending work *)
    let device = stream.device.dev.dev in
    let shared_event = Me.SharedEvent.on_device device in
    let value = Unsigned.ULLong.succ (Me.SharedEvent.get_signaled_value shared_event) in
    let command_buffer = Me.CommandBuffer.on_queue stream.runner in
    Me.CommandBuffer.encode_signal_event command_buffer shared_event value;
    Me.CommandBuffer.commit command_buffer;
    { event = shared_event; value }

  let initialize config =
    metal_initialized := true;
    Config.config |> ignore

  let is_initialized () = !metal_initialized

  let num_devices () = 
    let devices = Me.Device.copy_all_devices () in
    Array.length devices
    
  let devices = ref @@ Array.create ~len:(num_devices ()) None

  let get_used_memory (device : device) =
    (* Metal doesn't provide a direct way to get memory usage like CUDA does *)
    (* This is a workaround - we could track manually *)
    0 (* Placeholder *)

  let opt_alloc_merge_buffer ~size_in_bytes dev stream : unit =
    if
      Option.value_map ~default:true !(stream.merge_buffer) ~f:(fun buffer ->
          buffer.size_in_bytes < size_in_bytes)
    then (
      set_ctx dev.primary_context;
      Option.iter !(stream.merge_buffer) ~f:(fun _buffer -> ());
      stream.merge_buffer := Some { 
        ptr = Me.Buffer.on_device dev.dev ~length:size_in_bytes 
          Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache); 
        size_in_bytes 
      })

  let%track4_sexp finalize_device (device : device) =
    (* Metal's ARC handles resource cleanup, but we need to release cross_stream_candidates *)
    Hashtbl.iter device.cross_stream_candidates ~f:(fun _buffer_ptr -> ());
    Option.iter !Utils.advance_captured_logs ~f:(fun callback -> callback ())

  let%track3_sexp get_device ~(ordinal : int) : device =
    if num_devices () <= ordinal then
      invalid_arg [%string "Metal_backend.get_device %{ordinal#Int}: not enough devices"];
    
    (if Array.length !devices <= ordinal then
       let old, len = (!devices, Array.length !devices) in
       devices := Array.init (ordinal + 1) ~f:(fun i -> if i < len then old.(i) else None));
    
    let default () =
      let all_devices = Me.Device.copy_all_devices () in
      let dev = if ordinal < Array.length all_devices then all_devices.(ordinal)
                else invalid_arg [%string "Metal_backend.get_device %{ordinal#Int}: invalid device index"] in
      let primary_context = Me.CommandQueue.on_device dev in
      let dev = { dev; primary_context } in
      if Utils.debug_log_from_routines () && not (Hash_set.mem initialized_devices ordinal) then
        Hash_set.add initialized_devices ordinal;
      let result = make_device dev ~ordinal in
      Stdlib.Gc.finalise finalize_device result;
      !devices.(ordinal) <- Some result;
      result
    in
    Option.value_or_thunk !devices.(ordinal) ~default

  let%track3_sexp new_stream (device : device) : stream =
    let queue = Me.CommandQueue.on_device device.dev.dev in
    make_stream device queue

  let get_metal_device_info device =
    let props = Me.Device.get_attributes device.dev.dev in
    props

  let suggested_num_streams device =
    let props = get_metal_device_info device in
    match Config.config with
    | Only_devices_parallel -> 1
    | For_parallel_copying -> 2 (* Metal doesn't expose async_engine_count directly *)
    | Most_parallel_streams -> 4 (* Based on typical number of compute units *)

  let await stream : unit =
    (* Wait for all queued commands to complete *)
    (* We can achieve this by signaling and waiting for an event *)
    let event = all_work stream in
    sync event;
    Option.iter !Utils.advance_captured_logs ~f:(fun callback -> callback ())

  let is_idle stream = 
    (* Check if the command queue has no pending work *)
    (* This is an approximation - Metal doesn't expose a direct way to check this *)
    true (* Default to true as we can't easily check *)

  let from_host ~dst_ptr ~dst hosted =
    set_ctx @@ ctx_of dst;
    (* Copy from host memory to Metal buffer *)
    let size_in_bytes = Ndarray.size_in_bytes hosted in
    let command_buffer = Me.CommandBuffer.on_queue dst.stream.runner in
    
    (* Get host memory pointer *)
    let host_ptr = Ndarray.get_voidptr_not_managed hosted in
    
    (* Create a temporary host buffer to bridge the gap *)
    let temp_buffer = Me.Buffer.on_device_with_bytes dst.stream.device.dev.dev 
      ~bytes:host_ptr ~length:size_in_bytes 
      Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache) in
    
    (* Copy from temp buffer to destination buffer *)
    let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
    Me.BlitCommandEncoder.copy_from_buffer blit_encoder
      ~source_buffer:temp_buffer ~source_offset:0
      ~destination_buffer:dst_ptr ~destination_offset:0
      ~size:size_in_bytes;
    Me.BlitCommandEncoder.end_encoding blit_encoder;
    
    (* Submit the command buffer *)
    Me.CommandBuffer.commit command_buffer;
    
    (* No need to wait, as Metal will handle the dependency tracking *)

  let to_host ~src_ptr ~src hosted =
    set_ctx @@ ctx_of src;
    (* Copy from Metal buffer to host memory *)
    let size_in_bytes = Ndarray.size_in_bytes hosted in
    let command_buffer = Me.CommandBuffer.on_queue src.stream.runner in
    
    (* Get host memory pointer *)
    let host_ptr = Ndarray.get_voidptr_not_managed hosted in
    
    (* Create a temporary host buffer to bridge the gap *)
    let temp_buffer = Me.Buffer.on_device_with_bytes src.stream.device.dev.dev 
      ~bytes:host_ptr ~length:size_in_bytes 
      Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache) in
    
    (* Copy from source buffer to temp buffer *)
    let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
    Me.BlitCommandEncoder.copy_from_buffer blit_encoder
      ~source_buffer:src_ptr ~source_offset:0
      ~destination_buffer:temp_buffer ~destination_offset:0
      ~size:size_in_bytes;
    Me.BlitCommandEncoder.end_encoding blit_encoder;
    
    (* Submit and wait for completion to ensure host memory is updated *)
    Me.CommandBuffer.commit command_buffer;
    Me.CommandBuffer.wait_until_completed command_buffer;

  let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src =
    let dev = dst.stream.device.dev in
    let same_device = dev.dev == src.stream.device.dev.dev in
    let size_in_bytes = Lazy.force tn.Tn.size_in_bytes in
    
    let memcpy ~dst_ptr =
      (* Always use explicit copy as Metal doesn't have peer-to-peer memory access like CUDA *)
      let command_buffer = Me.CommandBuffer.on_queue dst.stream.runner in
      let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
      Me.BlitCommandEncoder.copy_from_buffer blit_encoder
        ~source_buffer:src_ptr ~source_offset:0
        ~destination_buffer:dst_ptr ~destination_offset:0
        ~size:size_in_bytes;
      Me.BlitCommandEncoder.end_encoding blit_encoder;
      Me.CommandBuffer.commit command_buffer
    in
    
    match (into_merge_buffer, dst_ptr) with
    | No, None -> invalid_arg "Metal_backend.device_to_device: missing dst_ptr"
    | No, Some dst_ptr ->
        set_ctx (ctx_of dst);
        memcpy ~dst_ptr
    | Streaming_for _, _ ->
        (* Metal doesn't support streaming as directly as CUDA *)
        if same_device then
          dst.stream.merge_buffer := Some { ptr = src_ptr; size_in_bytes }
        else
          (* Fall back to copy for different devices *)
          set_ctx (ctx_of dst);
          opt_alloc_merge_buffer ~size_in_bytes dev.dev dst.stream;
          let buffer = Option.value_exn ~here:[%here] !(dst.stream.merge_buffer) in
          memcpy ~dst_ptr:buffer.ptr
    | Copy, _ ->
        set_ctx (ctx_of dst);
        opt_alloc_merge_buffer ~size_in_bytes dev.dev dst.stream;
        let buffer = Option.value_exn ~here:[%here] !(dst.stream.merge_buffer) in
        memcpy ~dst_ptr:buffer.ptr

  type code = {
    traced_store : Low_level.traced_store;
    msl : string;  (* Metal Shading Language source *)
    params : (string * param_source) list;
    bindings : Indexing.unit_bindings;
    name : string;
  }
  [@@deriving sexp_of]

  type code_batch = {
    traced_stores : Low_level.traced_store option array;
    msl : string;  (* Combined Metal Shading Language source *)
    bindings : Indexing.unit_bindings;
    params_and_names : ((string * param_source) list * string) option array;
  }
  [@@deriving sexp_of]

  let%diagn2_sexp metal_to_msl ~name cu_src =
    let name_msl = name ^ ".metal" in
    if Utils.settings.output_debug_files_in_build_directory then (
      let oc = Out_channel.open_text @@ Utils.build_file name_msl in
      Stdio.Out_channel.output_string oc cu_src;
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc);
    [%log "compiling MSL source"];
    cu_src (* Just return the source, we'll compile it later *)

  module C_syntax_config (Input : sig
    val procs : Low_level.optimized array
  end) =
  struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]

    let procs = Input.procs
    let use_host_memory = use_host_memory
    let logs_to_stdout = true
    let main_kernel_prefix = "kernel" (* Metal kernels use 'kernel' attribute *)
    let kernel_prep_line = ""

    let includes = []  (* Metal doesn't use includes like C/C++ *)

    let typ_of_prec = function
      | Ops.Byte_prec _ -> "uchar"  (* Metal uses uchar rather than unsigned char *)
      | Half_prec _ -> "half"
      | Single_prec _ -> "float"
      | Double_prec _ -> "double"
      | Void_prec -> "void"

    let binop_syntax prec v =
      match (v, prec) with
      | Ops.Arg1, _ -> invalid_arg "Metal_backend.binop_syntax: Arg1 is not an operator"
      | Arg2, _ -> invalid_arg "Metal_backend.binop_syntax: Arg2 is not an operator"
      | _, Ops.Void_prec -> invalid_arg "Metal_backend.binop_syntax: Void precision"
      | Add, _ -> ("(", " +", ")")
      | Sub, _ -> ("(", " -", ")")
      | Mul, _ -> ("(", " *", ")")
      | Div, _ -> ("(", " /", ")")
      | ToPowOf, Double_prec _ -> ("pow(", ", ", ")")
      | ToPowOf, Single_prec _ -> ("pow(", ", ", ")")
      | ToPowOf, Half_prec _ -> ("pow(", ", ", ")")
      | ToPowOf, Byte_prec _ ->
          invalid_arg "Metal_backend.binop_syntax: ToPowOf not supported for byte/integer precisions"
      | Relu_gate, Byte_prec _ -> ("(", " > 0 ?", " : 0)")
      | Relu_gate, _ -> ("(", " > 0.0 ?", " : 0.0)")
      | Satur01_gate, Byte_prec _ -> ("(abs(", ") > 0 ? 0 : (", ")")
      | Satur01_gate, _ -> ("(fabs(trunc(", ")) > 0.0 ? 0.0 : (", "))")
      | Max, _ -> ("max(", ", ", ")")
      | Min, _ -> ("min(", ", ", ")")
      | Mod, Byte_prec _ -> ("(", " % ", ")")
      | Mod, _ -> ("fmod(", ", ", ")")
      | Cmplt, _ -> ("(", " < ", ")")
      | Cmpne, _ -> ("(", " != ", ")")
      | Cmpeq, _ -> ("(", " == ", ")")
      | Or, _ -> ("(", " || ", ")")
      | And, _ -> ("(", " && ", ")")

    let unop_syntax prec v =
      match (v, prec) with
      | Ops.Identity, _ -> ("", "")
      | Relu, _ -> ("max(0.0, ", ")")
      | Satur01, _ -> ("max(0.0, min(1.0, ", "))")
      | Exp, _ -> ("exp(", ")")
      | Log, _ -> ("log(", ")")
      | Exp2, _ -> ("exp2(", ")")
      | Log2, _ -> ("log2(", ")")
      | Sin, _ -> ("sin(", ")")
      | Cos, _ -> ("cos(", ")")
      | Sqrt, _ -> ("sqrt(", ")")
      | Recip, Byte_prec _ ->
          invalid_arg "Metal_backend.unop_syntax: Recip not supported for byte/integer precisions"
      | Recip, _ -> ("(1.0 / (", "))")
      | Recip_sqrt, Byte_prec _ ->
          invalid_arg
            "Metal_backend.unop_syntax: Recip_sqrt not supported for byte/integer precisions"
      | Recip_sqrt, _ -> ("(1.0 / sqrt(", "))")
      | Neg, _ -> ("(-(", "))")
      | Tanh_approx, Byte_prec _ ->
          invalid_arg
            "Metal_backend.unop_syntax: Tanh_approx not supported for byte/integer precisions"
      | Tanh_approx, _ -> ("tanh(", ")")
      | Not, _ -> ("(", " == 0.0 ? 1.0 : 0.0)")

    let ternop_syntax prec v =
      match (v, prec) with
      | Ops.Where, _ -> ("(", " ? ", " : ", ")")
      | FMA, _ -> ("fma(", ", ", ", ", ")")

    let convert_precision ~from ~to_ =
      match (from, to_) with
      | Ops.Double_prec _, Ops.Double_prec _
      | Single_prec _, Single_prec _
      | Half_prec _, Half_prec _
      | Byte_prec _, Byte_prec _
      | Void_prec, Void_prec ->
          ("", "")
      | _ -> ("(" ^ typ_of_prec to_ ^ ")(", ")")
  end

  let compile ~name bindings ({ Low_level.traced_store; _ } as lowered) =
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = [| lowered |]
    end)) in
    let idx_params = Indexing.bound_symbols bindings in
    let b = Buffer.create 4096 in
    let ppf = Stdlib.Format.formatter_of_buffer b in
    Syntax.print_includes ppf;
    let params = Syntax.compile_proc ~name ppf idx_params lowered in
    let msl = metal_to_msl ~name @@ Buffer.contents b in
    { traced_store; msl; params; bindings; name }

  let compile_batch ~names bindings lowereds =
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = Array.filter_opt lowereds
    end)) in
    let idx_params = Indexing.bound_symbols bindings in
    let b = Buffer.create 4096 in
    let ppf = Stdlib.Format.formatter_of_buffer b in
    Syntax.print_includes ppf;
    let params_and_names =
      Array.map2_exn names lowereds
        ~f:
          (Option.map2 ~f:(fun name lowered ->
               (Syntax.compile_proc ~name ppf idx_params lowered, name)))
    in
    let name : string =
      String.(
        strip ~drop:(equal_char '_')
        @@ common_prefix (Array.to_list names |> List.concat_map ~f:Option.to_list))
    in
    let msl = metal_to_msl ~name @@ Buffer.contents b in
    let traced_stores = Array.map lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store)) in
    { traced_stores; msl; params_and_names; bindings }

  let get_global_run_id =
    let next_id = ref 0 in
    fun () ->
      Int.incr next_id;
      if !next_id < 0 then next_id := 0;
      !next_id

  let link_proc ~prior_context ~name ~(params : (string * param_source) list) ~ctx_arrays
      lowered_bindings msl_source =
    let device = prior_context.stream.device.dev.dev in
    let stream = prior_context.stream in
    let runner_label = get_name stream in
    
    (* Compile MSL source code to a Metal library *)
    let compile_options = Me.CompileOptions.init () in
    Me.CompileOptions.set_fast_math_enabled compile_options true;
    let library = Me.Library.on_device device ~source:msl_source compile_options in
    
    (* Get the compute function (kernel) from the library *)
    let func = Me.Library.new_function_with_name library name in
    
    (* Create compute pipeline state *)
    let pipeline_state, _ = Me.ComputePipelineState.on_device_with_function device func in
    
    let%diagn3_sexp work () : unit =
      let log_id = get_global_run_id () in
      let log_id_prefix = Int.to_string log_id ^ ": " in
      [%log_result "Launching", name, "on", runner_label, (log_id : int), (params : (string * param_source) list)];
      
      (* Create a command buffer and encoder *)
      let command_buffer = Me.CommandBuffer.on_queue stream.runner in
      let compute_encoder = Me.ComputeCommandEncoder.on_buffer command_buffer in
      
      (* Set the compute pipeline state *)
      Me.ComputeCommandEncoder.set_compute_pipeline_state compute_encoder pipeline_state;
      
      (* Set the buffers from parameters *)
      let idx = ref 0 in
      List.iter params ~f:(function
        | _name, Param_ptr tn ->
            let arr = Option.value_exn ~here:[%here] @@ Map.find ctx_arrays tn in
            Me.ComputeCommandEncoder.set_buffer compute_encoder ~index:!idx arr;
            Int.incr idx
        | _name, Log_file_name -> 
            (* Metal doesn't support passing string literals directly, would need a buffer *)
            Int.incr idx
        | _name, Merge_buffer ->
            let buf = Option.value_exn ~here:[%here] !(stream.merge_buffer) in
            Me.ComputeCommandEncoder.set_buffer compute_encoder ~index:!idx buf.ptr;
            Int.incr idx
        | _name, Static_idx s ->
            let i = Indexing.find_exn lowered_bindings s in
            if !i < 0 then
              raise
              @@ Utils.User_error
                   [%string
                     "metal: static index %{Indexing.symbol_ident s.static_symbol} is negative: \
                      %{!i#Int}"];
            Option.iter s.static_range ~f:(fun upto ->
                if !i >= upto then
                  raise
                  @@ Utils.User_error
                       [%string
                         "metal: static index %{Indexing.symbol_ident s.static_symbol} is too \
                          big: %{upto#Int}"]);
            (* Set the static index as a buffer value *)
            Me.ComputeCommandEncoder.set_bytes compute_encoder 
              ~bytes:(Ctypes.addr !i) ~length:(Ctypes.sizeof Ctypes.int) ~index:!idx;
            Int.incr idx);
      
      (* Dispatch the kernel with a single threadgroup *)
      let threads_per_group = Me.Size.make ~width:1 ~height:1 ~depth:1 in
      let grid_size = Me.Size.make ~width:1 ~height:1 ~depth:1 in
      Me.ComputeCommandEncoder.dispatch_threadgroups compute_encoder
        ~threadgroups_per_grid:(Me.Size.from_struct grid_size)
        ~threads_per_threadgroup:(Me.Size.from_struct threads_per_group);
      
      (* End encoding and commit the command buffer *)
      Me.ComputeCommandEncoder.end_encoding compute_encoder;
      Me.CommandBuffer.commit command_buffer;
      
      [%log "kernel launched"]
    in
    Task.Task
      {
        context_lifetime = (library, ctx_arrays);
        description = "launches " ^ name ^ " on " ^ runner_label;
        work;
      }

  let%track3_sexp link prior_context (code : code) ctx_arrays =
    let idx_params = Indexing.bound_symbols code.bindings in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map idx_params ~f:(fun s -> (s, ref 0))
    in
    let task =
      link_proc ~prior_context ~name:code.name ~params:code.params ~ctx_arrays lowered_bindings
        code.msl
    in
    (lowered_bindings, task)

  let%track3_sexp link_batch prior_context (code_batch : code_batch) ctx_arrays =
    let idx_params = Indexing.bound_symbols code_batch.bindings in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map idx_params ~f:(fun s -> (s, ref 0))
    in
    
    let procs =
      Array.mapi code_batch.params_and_names ~f:(fun i pns ->
          Option.value ~default:None
          @@ Option.map2 pns ctx_arrays.(i) ~f:(fun (params, name) ctx_arrays ->
                 let task =
                   link_proc ~prior_context ~name ~params ~ctx_arrays lowered_bindings
                     code_batch.msl
                 in
                 Some task))
    in
    (lowered_bindings, procs)

  let get_global_debug_info () =
    (* Metal doesn't provide as much debug info as CUDA *)
    Sexp.message "metal_global_debug" []

  let get_debug_info (_stream : stream) =
    (* Metal doesn't provide as much stream debug info as CUDA *)
    Sexp.message "metal_stream_debug" []
end

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

module Backend_buffer = struct
  type buffer_ptr = Me.Buffer.t

  let sexp_of_buffer_ptr ptr = 
    Sexp.message "<MetalBuffer>" [("id", Me.Buffer.sexp_of_t ptr)]

  include Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]
  end)
end

module Device_config = struct
  include Backend_buffer

  type dev = { 
    dev : Me.Device.t; 
    command_queue : Me.CommandQueue.t;
    compute_library : Me.Library.t;
  } [@@deriving sexp_of]
  
  type runner = Me.CommandBuffer.t [@@deriving sexp_of]
  type event = Me.SharedEvent.t [@@deriving sexp_of]

  let name = "metal"
end

module Device_stream = Impl.Device_types(Device_config)
open Device_config

module Fresh () = struct
  include Impl.Device(Device_stream)(struct
    include Device_stream

    (* It's not actually used, but it's required by the [Backend] interface. *)
    let alloc_buffer ?old_buffer ~size_in_bytes stream =
      match old_buffer with
      | Some ({ size_in_bytes = old_size; _ } as buffer) when size_in_bytes <= old_size -> buffer
      | Some { ptr; _ } ->
          (* Old buffer will be freed by Metal's reference counting system *)
          { ptr = Me.Buffer.on_device stream.device.dev.dev ~length:size_in_bytes 
                 (Me.ResourceOptions.make ()); 
            size_in_bytes }
      | None ->
          { ptr = Me.Buffer.on_device stream.device.dev.dev ~length:size_in_bytes 
                 (Me.ResourceOptions.make ()); 
            size_in_bytes }

    let alloc_zero_init_array prec ~dims stream =
      let size_in_bytes =
        (if Array.length dims = 0 then 0 else Array.reduce_exn dims ~f:( * )) 
        * Ops.prec_in_bytes prec
      in
      (* Create a buffer and initialize with zeros *)
      let zero_buffer = Bytes.make size_in_bytes '\000' in
      let ptr = Ctypes.(to_voidp @@ allocate_n char ~count:size_in_bytes) in
      (* Ctypes.memset ptr 0 (Unsigned.Size_t.of_int size_in_bytes); *)
      Me.Buffer.on_device_with_bytes stream.device.dev.dev 
        ~bytes:ptr ~length:size_in_bytes (Me.ResourceOptions.make ())

    let free_buffer = None (* Metal handles memory with reference counting *)
  end)

  let use_host_memory = Some (fun host_ptr -> 
      (* Create a buffer that wraps the host memory *)
      let device = Me.Device.create_system_default() in
      let size = 0 (* This is just a placeholder; real size should be passed *)
      in
      Me.Buffer.on_device_with_bytes_no_copy 
        device 
        ~bytes:host_ptr
        ~length:size
        ?deallocator:None
        (Me.ResourceOptions.make ())
    )

  (* Tracks initialized devices *)
  let initialized_devices = Hash_set.create (module Int)

  let global_config = ref For_parallel_copying

  (* Initialize Metal *)
  let is_initialized, initialize =
    let initialized = ref false in
    let init (config : config) : unit =
      initialized := true;
      global_config := config
    in
    ((fun () -> !initialized), init)

  let num_devices () = 
    (* Metal can only access the system default device in most cases *)
    1

  let devices = ref @@ Array.create ~len:(num_devices ()) None

  let get_used_memory (device : device) =
    (* Metal doesn't provide direct memory usage querying like CUDA *)
    (* We could keep track of allocated buffers ourselves *)
    0 (* Placeholder *)

  let opt_alloc_merge_buffer ~size_in_bytes dev stream : unit =
    if Option.value_map ~default:true !(stream.merge_buffer) ~f:(fun buffer ->
        buffer.size_in_bytes < size_in_bytes)
    then (
      Option.iter !(stream.merge_buffer) ~f:(fun _buffer -> ()); (* No manual freeing needed *)
      stream.merge_buffer := Some { 
        ptr = Me.Buffer.on_device dev.dev ~length:size_in_bytes (Me.ResourceOptions.make ()); 
        size_in_bytes 
      })

  let%track4_sexp finalize_device (device : device) =
    (* No explicit context synchronization needed *)
    Option.iter !Utils.advance_captured_logs ~f:(fun callback -> callback ());
    (* Release is automatic through reference counting *)
    Hashtbl.iter device.cross_stream_candidates ~f:(fun _buffer_ptr -> ())

  let%track3_sexp get_device ~(ordinal : int) : device =
    if num_devices () <= ordinal then
      invalid_arg [%string "Metal_backend.get_device %{ordinal#Int}: not enough devices"];
    
    (if Array.length !devices <= ordinal then
       let old, len = (!devices, Array.length !devices) in
       devices := Array.init (ordinal + 1) ~f:(fun i -> if i < len then old.(i) else None));
    
    let default () =
      let metal_device = Me.Device.create_system_default () in
      let command_queue = Me.CommandQueue.on_device metal_device in
      
      (* Create a default compute library with Metal standard library functions *)
      let default_functions = [
        "__half hexp(__half x) { return __float2half(exp(__half2float(x))); }";
        "__half hlog(__half x) { return __float2half(log(__half2float(x))); }";
        "__half hexp2(__half x) { return __float2half(exp2(__half2float(x))); }";
        "__half hlog2(__half x) { return __float2half(log2(__half2float(x))); }";
        "__half hsin(__half x) { return __float2half(sin(__half2float(x))); }";
        "__half hcos(__half x) { return __float2half(cos(__half2float(x))); }";
        "__half hsqrt(__half x) { return __float2half(sqrt(__half2float(x))); }";
        "__half hrcp(__half x) { return __float2half(1.0/__half2float(x)); }";
        "__half hrsqrt(__half x) { return __float2half(1.0/sqrt(__half2float(x))); }";
        "__half htanh_approx(__half x) { return __float2half(tanh(__half2float(x))); }";
        "__half __hmax_nan(__half a, __half b) { return __half2float(a) > __half2float(b) ? a : b; }";
        "__half __hmin_nan(__half a, __half b) { return __half2float(a) < __half2float(b) ? a : b; }";
      ] in
      
      let lib_source = String.concat ~sep:"\n" [
        "#include <metal_stdlib>"; 
        "#include <metal_math>";
        "using namespace metal;";
        String.concat ~sep:"\n" default_functions;
      ] in
      
      let options = Me.CompileOptions.init () in
      Me.CompileOptions.set_language_version options Me.CompileOptions.LanguageVersion.version_2_4;
      let compute_library = Me.Library.on_device metal_device ~source:lib_source options in
      
      let dev = { dev = metal_device; command_queue; compute_library } in
      
      if Utils.debug_log_from_routines () && not (Hash_set.mem initialized_devices ordinal) then
        ();
      
      Hash_set.add initialized_devices ordinal;
      let result = make_device dev ~ordinal in
      Stdlib.Gc.finalise finalize_device result;
      !devices.(ordinal) <- Some result;
      result
    in
    Option.value_or_thunk !devices.(ordinal) ~default

  let%track3_sexp new_stream (device : device) : stream =
    let command_buffer = Me.CommandBuffer.on_queue device.dev.command_queue in
    make_stream device command_buffer

  let metal_properties =
    let cache =
      let%debug2_sexp f (ordinal : int) =
        let dev = get_device ~ordinal in
        lazy (Me.Device.get_attributes dev.dev.dev)
      in
      lazy (Array.init (num_devices ()) ~f)
    in
    let%debug2_sexp get_props (device : device) : Me.Device.attributes =
      let cache = Lazy.force cache in
      Lazy.force cache.(device.ordinal)
    in
    get_props

  let suggested_num_streams device =
    match !global_config with
    | Only_devices_parallel -> 1
    | For_parallel_copying -> 2  (* Metal doesn't expose async engine count *)
    | Most_parallel_streams -> max 4 (metal_properties device).max_buffer_length / (1024 * 1024)
                               (* Rough estimate based on device memory *)

  let await stream : unit =
    Me.CommandBuffer.wait_until_completed stream.runner;
    Option.iter !Utils.advance_captured_logs ~f:(fun callback -> callback ())

  let is_done event = 
    (* Returns true if the event has been signaled with a value >= the wait value *)
    let current_value = Me.SharedEvent.get_signaled_value event in
    Unsigned.ULLong.compare current_value Unsigned.ULLong.one >= 0

  let is_idle stream = 
    Me.CommandBuffer.get_status stream.runner = Me.CommandBuffer.Status.Completed

  let will_wait_for context event =
    (* Encode a wait command into the context's command buffer *)
    Me.CommandBuffer.encode_wait_for_event context.stream.runner event Unsigned.ULLong.one

  let sync event =
    (* Wait for the event to be signaled *)
    ignore (Me.SharedEvent.wait_until_signaled_value 
              event 
              ~value:Unsigned.ULLong.one 
              ~timeout_ms:(Unsigned.ULLong.of_int 1000000));
    ()

  let all_work stream =
    (* Create a new shared event and signal it when the command buffer completes *)
    let event = Me.SharedEvent.on_device stream.device.dev.dev in
    let _ = Me.CommandBuffer.add_completed_handler stream.runner (fun _buffer ->
        Me.SharedEvent.set_signaled_value event Unsigned.ULLong.one
      ) in
    Me.CommandBuffer.commit stream.runner;
    event

  let from_host ~dst_ptr ~dst hosted =
    let f dst_buffer = 
      let host_ptr = Ndarray.get_voidptr_not_managed hosted in
      let size_in_bytes = Ndarray.size_in_bytes hosted in
      
      (* Create a temporary blitter to copy from host to device *)
      let blit_buffer = Me.CommandBuffer.on_queue dst.stream.device.dev.command_queue in
      let blit_encoder = Me.BlitCommandEncoder.on_buffer blit_buffer in
      
      (* Copy the bytes from host to device buffer *)
      let temp_buffer = Me.Buffer.on_device_with_bytes dst.stream.device.dev.dev 
                          ~bytes:host_ptr ~length:size_in_bytes (Me.ResourceOptions.make ()) in
      
      Me.BlitCommandEncoder.copy_from_buffer blit_encoder 
        ~source_buffer:temp_buffer ~source_offset:0
        ~destination_buffer:dst_buffer ~destination_offset:0
        ~size:size_in_bytes;
      
      Me.BlitCommandEncoder.end_encoding blit_encoder;
      Me.CommandBuffer.commit blit_buffer;
      Me.CommandBuffer.wait_until_completed blit_buffer;
    in
    f dst_ptr

  let to_host ~src_ptr ~src hosted =
    let f dst_host_array = 
      let host_ptr = Ndarray.get_voidptr_not_managed dst_host_array in
      let size_in_bytes = Ndarray.size_in_bytes dst_host_array in
      
      (* Create a temporary buffer to receive the data *)
      let temp_buffer = Me.Buffer.on_device src.stream.device.dev.dev
                          ~length:size_in_bytes (Me.ResourceOptions.make ()) in
      
      (* Create a blitter to copy from device to temp buffer *)
      let blit_buffer = Me.CommandBuffer.on_queue src.stream.device.dev.command_queue in
      let blit_encoder = Me.BlitCommandEncoder.on_buffer blit_buffer in
      
      Me.BlitCommandEncoder.copy_from_buffer blit_encoder
        ~source_buffer:src_ptr ~source_offset:0
        ~destination_buffer:temp_buffer ~destination_offset:0
        ~size:size_in_bytes;
      
      Me.BlitCommandEncoder.end_encoding blit_encoder;
      Me.CommandBuffer.commit blit_buffer;
      Me.CommandBuffer.wait_until_completed blit_buffer;
      
      (* Now read from temp buffer to host memory *)
      let contents_ptr = Me.Buffer.contents temp_buffer in
      Ctypes.memcpy (Ctypes.to_voidp host_ptr) contents_ptr (Unsigned.Size_t.of_int size_in_bytes);
    in
    f hosted

  let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src =
    let dev = dst.stream.device in
    let same_device = dev.ordinal = src.stream.device.ordinal in
    let size_in_bytes = Lazy.force tn.Tn.size_in_bytes in
    
    let memcpy ~dst_ptr =
      if same_device then
        (* Using blit encoder to copy within the same device *)
        let blit_buffer = Me.CommandBuffer.on_queue dst.stream.device.dev.command_queue in
        let blit_encoder = Me.BlitCommandEncoder.on_buffer blit_buffer in
        
        Me.BlitCommandEncoder.copy_from_buffer blit_encoder
          ~source_buffer:src_ptr ~source_offset:0
          ~destination_buffer:dst_ptr ~destination_offset:0
          ~size:size_in_bytes;
        
        Me.BlitCommandEncoder.end_encoding blit_encoder;
        Me.CommandBuffer.commit blit_buffer
      else
        (* Cross-device copy requires reading to host memory first then writing to destination *)
        (* For now we assume size is small enough to use a temporary host buffer *)
        let host_buffer = Bytes.create size_in_bytes in
        let host_ptr = Ctypes.(to_voidp @@ allocate_n char ~count:size_in_bytes) in
        
        (* Copy from source device to host *)
        let temp_src_buffer = Me.Buffer.on_device_with_bytes_no_copy 
                                src.stream.device.dev.dev
                                ~bytes:host_ptr
                                ~length:size_in_bytes
                                ~deallocator:None
                                (Me.ResourceOptions.make ()) in
        
        (* Create a blitter to copy from source to temp *)
        let blit_src = Me.CommandBuffer.on_queue src.stream.device.dev.command_queue in
        let blit_src_encoder = Me.BlitCommandEncoder.on_buffer blit_src in
        
        Me.BlitCommandEncoder.copy_from_buffer blit_src_encoder
          ~source_buffer:src_ptr ~source_offset:0
          ~destination_buffer:temp_src_buffer ~destination_offset:0
          ~size:size_in_bytes;
        
        Me.BlitCommandEncoder.end_encoding blit_src_encoder;
        Me.CommandBuffer.commit blit_src;
        Me.CommandBuffer.wait_until_completed blit_src;
        
        (* Copy from host to destination device *)
        let temp_dst_buffer = Me.Buffer.on_device_with_bytes
                                dst.stream.device.dev.dev
                                ~bytes:host_ptr
                                ~length:size_in_bytes
                                (Me.ResourceOptions.make ()) in
        
        (* Create a blitter to copy from temp to destination *)
        let blit_dst = Me.CommandBuffer.on_queue dst.stream.device.dev.command_queue in
        let blit_dst_encoder = Me.BlitCommandEncoder.on_buffer blit_dst in
        
        Me.BlitCommandEncoder.copy_from_buffer blit_dst_encoder
          ~source_buffer:temp_dst_buffer ~source_offset:0
          ~destination_buffer:dst_ptr ~destination_offset:0
          ~size:size_in_bytes;
        
        Me.BlitCommandEncoder.end_encoding blit_dst_encoder;
        Me.CommandBuffer.commit blit_dst
    in
    
    match (into_merge_buffer, dst_ptr) with
    | No, None -> invalid_arg "Metal_backend.device_to_device: missing dst_ptr"
    | No, Some dst_ptr ->
        memcpy ~dst_ptr
    | Streaming_for _, _ ->
        assert same_device;
        dst.stream.merge_buffer := Some { ptr = src_ptr; size_in_bytes }
    | Copy, _ ->
        opt_alloc_merge_buffer ~size_in_bytes dev.dev dst.stream;
        let buffer = Option.value_exn ~here:[%here] !(dst.stream.merge_buffer) in
        memcpy ~dst_ptr:buffer.ptr

  type code = {
    traced_store : Low_level.traced_store;
    metal_lib : Me.Library.t;
    params : (string * param_source) list;
    bindings : Indexing.unit_bindings;
    name : string;
  }
  [@@deriving sexp_of]

  type code_batch = {
    traced_stores : Low_level.traced_store option array;
    metal_lib : Me.Library.t;
    bindings : Indexing.unit_bindings;
    params_and_names : ((string * param_source) list * string) option array;
  }
  [@@deriving sexp_of]

  let%diagn2_sexp metal_to_lib ~name msl_src =
    let name_msl = name ^ ".metal" in
    if Utils.settings.output_debug_files_in_build_directory then (
      let oc = Out_channel.open_text @@ Utils.build_file name_msl in
      Stdio.Out_channel.output_string oc msl_src;
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc);
    
    [%log "compiling to Metal library"];
    
    let options = Me.CompileOptions.init () in
    Me.CompileOptions.set_language_version options Me.CompileOptions.LanguageVersion.version_2_4;
    
    let device = Me.Device.create_system_default () in
    let lib = Me.Library.on_device device ~source:msl_src options in
    
    lib

  module C_syntax_config (Input : sig
    val procs : Low_level.optimized array
  end) =
  struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]

    let procs = Input.procs
    let use_host_memory = use_host_memory
    let logs_to_stdout = true
    let main_kernel_prefix = "kernel"
    
    (* Handle thread indexing for Metal kernels *)
    let kernel_prep_line = 
      "uint tid = (uint)threadgroup_position_in_grid.x * threadgroups_per_grid.x + (uint)thread_position_in_threadgroup.x;"

    let includes = []  (* Metal includes are different from C/CUDA *)

    let typ_of_prec = function
      | Ops.Byte_prec _ -> "uchar"
      | Half_prec _ -> "half"
      | Single_prec _ -> "float"
      | Double_prec _ -> "float" (* Metal doesn't support double, using float as fallback *)
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
      | ToPowOf, _ -> ("pow(", ",", ")")
      | Relu_gate, Byte_prec _ -> ("(", " > 0 ?", " : 0)")
      | Relu_gate, _ -> ("(", " > 0.0 ?", " : 0.0)")
      | Satur01_gate, Byte_prec _ -> ("(abs(", ") > 0 ? 0 : (", ")")
      | Satur01_gate, _ -> ("(abs(", ") > 0.0 ? 0.0 : (", ")")
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
      | Recip, _ -> ("(1.0 / (", "))")
      | Recip_sqrt, _ -> ("(1.0 / sqrt(", "))")
      | Neg, _ -> ("(-(", "))")
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
      | _, _ -> ("(" ^ typ_of_prec to_ ^ ")(", ")")
  end

  let compile ~name bindings ({ Low_level.traced_store; _ } as lowered) =
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = [| lowered |]
    end)) in
    let idx_params = Indexing.bound_symbols bindings in
    let b = Buffer.create 4096 in
    let ppf = Stdlib.Format.formatter_of_buffer b in
    
    (* Add Metal-specific includes and declarations *)
    Stdlib.Format.fprintf ppf "#include <metal_stdlib>\n";
    Stdlib.Format.fprintf ppf "#include <metal_math>\n";
    Stdlib.Format.fprintf ppf "using namespace metal;\n\n";
    
    let params = Syntax.compile_proc ~name ppf idx_params lowered in
    let metal_lib = metal_to_lib ~name @@ Buffer.contents b in
    { traced_store; metal_lib; params; bindings; name }

  let compile_batch ~names bindings lowereds =
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = Array.filter_opt lowereds
    end)) in
    let idx_params = Indexing.bound_symbols bindings in
    let b = Buffer.create 4096 in
    let ppf = Stdlib.Format.formatter_of_buffer b in
    
    (* Add Metal-specific includes and declarations *)
    Stdlib.Format.fprintf ppf "#include <metal_stdlib>\n";
    Stdlib.Format.fprintf ppf "#include <metal_math>\n";
    Stdlib.Format.fprintf ppf "using namespace metal;\n\n";
    
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
    
    let metal_lib = metal_to_lib ~name @@ Buffer.contents b in
    let traced_stores = Array.map lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store)) in
    { traced_stores; metal_lib; params_and_names; bindings }

  let get_global_run_id =
    let next_id = ref 0 in
    fun () ->
      Int.incr next_id;
      if !next_id < 0 then next_id := 0;
      !next_id

  let link_proc ~prior_context ~name ~(params : (string * param_source) list) ~ctx_arrays
      lowered_bindings metal_lib =
    
    let func = Me.Library.new_function_with_name metal_lib name in
    let stream = prior_context.stream in
    let runner_label = get_name stream in
    
    let%diagn3_sexp work () : unit =
      let log_id = get_global_run_id () in
      let log_id_prefix = Int.to_string log_id ^ ": " in
      
      [%log_result "Launching", name, "on", runner_label, (log_id : int), 
                   (params : (string * param_source) list)];
      
      (* Create a compute pipeline state for the function *)
      let compute_pipeline, _ = 
        Me.ComputePipelineState.on_device_with_function stream.device.dev.dev func in
      
      (* Create a command buffer and compute encoder *)
      let command_buffer = Me.CommandBuffer.on_queue stream.device.dev.command_queue in
      let compute_encoder = Me.ComputeCommandEncoder.on_buffer command_buffer in
      
      (* Set the compute pipeline state *)
      Me.ComputeCommandEncoder.set_compute_pipeline_state compute_encoder compute_pipeline;
      
      (* Set buffer arguments *)
      List.iteri params ~f:(fun i -> function
        | _name, Param_ptr tn ->
            let arr = Option.value_exn ~here:[%here] @@ Map.find ctx_arrays tn in
            Me.ComputeCommandEncoder.set_buffer compute_encoder ~index:i arr
        | _name, Log_file_name -> 
            (* For logging, we could use a buffer to collect logs *)
            ()
        | _name, Merge_buffer ->
            let buf = Option.value_exn ~here:[%here] !(stream.merge_buffer) in
            Me.ComputeCommandEncoder.set_buffer compute_encoder ~index:i buf.ptr
        | _name, Static_idx s ->
            let i_val = Indexing.find_exn lowered_bindings s in
            if !i_val < 0 then
              raise
              @@ Utils.User_error
                   [%string
                     "metal: static index %{Indexing.symbol_ident s.static_symbol} is negative: \
                      %{!i_val#Int}"];
            Option.iter s.static_range ~f:(fun upto ->
                if !i_val >= upto then
                  raise
                  @@ Utils.User_error
                       [%string
                         "metal: static index %{Indexing.symbol_ident s.static_symbol} is too \
                          big: %{upto#Int}"]);
            
            (* Set scalar constant i_val into constant memory *)
            Me.ComputeCommandEncoder.set_bytes compute_encoder 
              ~bytes:(Ctypes.addr !i_val) ~length:(Ctypes.sizeof Ctypes.int) ~index:i);
      
      (* Calculate grid and threadgroup size *)
      let threads_per_threadgroup = Me.Size.make ~width:64 ~height:1 ~depth:1 in
      let threadgroups_per_grid = Me.Size.make ~width:1 ~height:1 ~depth:1 in
      
      (* Dispatch the compute function *)
      Me.ComputeCommandEncoder.dispatch_threadgroups compute_encoder
        ~threadgroups_per_grid ~threads_per_threadgroup;
      
      (* End encoding and commit the command buffer *)
      Me.ComputeCommandEncoder.end_encoding compute_encoder;
      Me.CommandBuffer.commit command_buffer;
      
      [%log "kernel launched"]
    in
    
    Task.Task
      {
        context_lifetime = (metal_lib, ctx_arrays);
        description = "launches " ^ name ^ " on " ^ runner_label;
        work;
      }

  let%track3_sexp link prior_context (code : code) ctx_arrays =
    let lowered_bindings : Indexing.lowered_bindings =
      let idx_params = Indexing.bound_symbols code.bindings in
      List.map idx_params ~f:(fun s -> (s, ref 0))
    in
    let task =
      link_proc ~prior_context ~name:code.name ~params:code.params ~ctx_arrays 
        lowered_bindings code.metal_lib
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
                   link_proc ~prior_context ~name ~params ~ctx_arrays 
                     lowered_bindings code_batch.metal_lib
                 in
                 Some task))
    in
    (lowered_bindings, procs)

  let get_global_debug_info () =
    Sexp.message "metal_global_debug" []

  let get_debug_info (_stream : stream) =
    Sexp.message "metal_stream_debug" []
end

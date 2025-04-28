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
(* Global state for Metal devices *)
let metal_devices : Me.Device.t array option ref = ref None

let initialize_device_list () =
  if Option.is_none !metal_devices then
    metal_devices := Some (Me.Device.copy_all_devices ())

let get_all_metal_devices () =
  initialize_device_list ();
  Option.value_exn ~message:"Metal devices not initialized" !metal_devices

module Backend_buffer = struct
  type buffer_ptr = Me.Buffer.t (* Use the payload type from metal.ml *)

  (* Provide a sexp_of for Me.Buffer.t *)
  let sexp_of_buffer_ptr = Me.Buffer.sexp_of_t

  include Buffer_types (struct
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
    counter : Utils.atomic_ullong; (* Next value to signal *)
  }

  let sexp_of_runner r =
    Sexp.List
      [
        Sexp.Atom "<runner>";
        Me.CommandQueue.sexp_of_t r.queue;
        Me.SharedEvent.sexp_of_t r.event;
        Sexp.Atom ("counter=" ^ Unsigned.ULLong.to_string (Utils.atomic_get r.counter));
      ]

  (* Represents a point in time on a specific runner *)
  type event = Me.SharedEvent.t * ullong (* Pair of SharedEvent and target value *)

  let sexp_of_event (evt, value) =
    Sexp.List [ Me.SharedEvent.sexp_of_t evt; sexp_of_ullong value ]

  let name = "metal"
end

module Device_stream = Backend_impl.Device_types (Device_config)

(* Bring types into scope *)
open Device_config

(* Manual tracking for allocated memory *)
let allocated_memory = Atomic.make 0

let track_allocation (buffer : Me.Buffer.t) =
  (* Relying on ARC + finalizer on the payload record *)
  try
    let size = Me.Resource.get_allocated_size (Me.Buffer.super buffer) in
    Stdlib.Gc.finalise
      (fun _ -> ignore (Atomic.fetch_and_add allocated_memory (-size)))
      buffer;
    ignore (Atomic.fetch_and_add allocated_memory size)
  with exn ->
    [%log "Exception during track_allocation finalizer setup:", (exn : exn)];
    (* Log and ignore: Finalizers shouldn't prevent allocation *)
    ()

module Alloc_buffer = struct
  include Device_stream

  let resource_options =
    (* Use Shared storage for unified memory, WriteCombined for CPU writes, Tracked hazards *)
    Me.ResourceOptions.(
      storage_mode_shared + cpu_cache_mode_write_combined + hazard_tracking_mode_tracked)

  let alloc_buffer ?old_buffer ~size_in_bytes stream =
    let device = stream.device.dev in
    match old_buffer with
    | Some ({ size_in_bytes = old_size; _ } as buffer) when size_in_bytes <= old_size -> buffer
    | Some _old ->
        (* ARC should handle the old buffer *)
        let new_buffer_obj = Me.Buffer.on_device device ~length:size_in_bytes resource_options in
        track_allocation new_buffer_obj;
        { ptr = new_buffer_obj; size_in_bytes }
    | None ->
        let new_buffer_obj = Me.Buffer.on_device device ~length:size_in_bytes resource_options in
        track_allocation new_buffer_obj;
        { ptr = new_buffer_obj; size_in_bytes }

  let alloc_zero_init_array prec ~dims stream =
    let size_in_bytes =
      (if Array.length dims = 0 then 0 else Array.reduce_exn dims ~f:( * )) * Ops.prec_in_bytes prec
    in
    let device = stream.device.dev in
    let buffer_obj = Me.Buffer.on_device device ~length:size_in_bytes resource_options in
    track_allocation buffer_obj;
    (* Don't zero-init here. Kernels using must handle initialization (e.g., via memset equivalent or first write). *)
    (* Alternatively, could enqueue a fillBuffer command here, but makes alloc async. *)
    buffer_obj

  (* Rely on ARC and the finalizer attached in track_allocation *)
  let free_buffer = None
end

(* Functor defining the backend *)
module Fresh (Config : sig
  val config : Ir.Backend_intf.config
end) : sig
  (* Explicit signature matching the user request *)
  type buffer_ptr = Device_config.buffer_ptr
  type dev = Device_config.dev
  type runner = Device_config.runner
  type event = Device_config.event

  val use_host_memory : (unit Ctypes.ptr -> buffer_ptr) option
  val sexp_of_dev : dev -> Sexp.t
  val sexp_of_runner : runner -> Sexp.t
  val sexp_of_event : event -> Sexp.t
  val name : string
  type nonrec device = (buffer_ptr, dev, runner, event) Backend_intf.device
  val sexp_of_device : device -> Sexp.t
  type nonrec stream = (buffer_ptr, dev, runner, event) Backend_intf.stream
  val sexp_of_stream : stream -> Sexp.t
  type nonrec context = (buffer_ptr, stream) Backend_intf.context
  val sexp_of_context : context -> Sexp.t

  val alloc_buffer :
    ?old_buffer:buffer_ptr Backend_intf.buffer -> size_in_bytes:int -> stream -> buffer_ptr Backend_intf.buffer

  val alloc_zero_init_array : Ops.prec -> dims:int array -> stream -> buffer_ptr
  val free_buffer : (stream -> buffer_ptr -> unit) option
  val make_device : dev -> ordinal:int -> device
  val make_stream : device -> runner -> stream
  val make_context : ?ctx_arrays:buffer_ptr Backend_intf.ctx_arrays -> stream -> context
  val make_child : ?ctx_arrays:buffer_ptr Backend_intf.ctx_arrays -> context -> context
  val get_name : stream -> string
  val sexp_of_buffer_ptr : buffer_ptr -> Sexp.t
  type nonrec buffer = buffer_ptr Backend_intf.buffer
  val sexp_of_buffer : buffer -> Sexp.t
  type nonrec ctx_arrays = buffer_ptr Backend_intf.ctx_arrays
  val sexp_of_ctx_arrays : ctx_arrays -> Sexp.t
  val initialize : Backend_intf.config -> unit
  val is_initialized : unit -> bool
  val sync : event -> unit
  val is_done : event -> bool
  val will_wait_for : context -> event -> unit
  val get_used_memory : device -> int
  val get_global_debug_info : unit -> Sexp.t
  val get_debug_info : stream -> Sexp.t
  val await : stream -> unit
  val all_work : stream -> event
  val is_idle : stream -> bool
  val get_device : ordinal:int -> device
  val num_devices : unit -> int
  val suggested_num_streams : device -> int
  val new_stream : device -> stream
  val from_host : dst_ptr:buffer_ptr -> dst:context -> Ndarray.t -> unit
  val to_host : src_ptr:buffer_ptr -> src:context -> Ndarray.t -> unit

  val device_to_device :
    Tnode.t ->
    into_merge_buffer:Backend_intf.merge_buffer_use ->
    dst_ptr:buffer_ptr option ->
    dst:context ->
    src_ptr:buffer_ptr ->
    src:context ->
    unit

  type code = {
    metal_source : string;
    func_name : string;
    params : (string * param_source) list;
    bindings : Indexing.unit_bindings;
    traced_store : Low_level.traced_store;
  }
  val sexp_of_code : code -> Sexp.t

  type code_batch = {
    metal_source : string;
    funcs : (string * (string * param_source) list) option array;
    bindings : Indexing.unit_bindings;
    traced_stores : Low_level.traced_store option array;
  }
  val sexp_of_code_batch : code_batch -> Sexp.t

  val compile : name:string -> Indexing.unit_bindings -> Low_level.optimized -> code

  val compile_batch :
    names:string option array ->
    Indexing.unit_bindings ->
    Low_level.optimized option array ->
    code_batch

  val link : context -> code -> ctx_arrays -> Indexing.lowered_bindings * Task.t

  val link_batch :
    context ->
    code_batch ->
    ctx_arrays option array ->
    Indexing.lowered_bindings * Task.t option array
end = struct
  (* Include the device setup with types and allocation *)
  include Backend_impl.Device (Device_stream) (Alloc_buffer)

  (* Metal on Apple Silicon usually has unified memory *)
  let use_host_memory =
    let check_device_memory () =
      try
        let devices = get_all_metal_devices () in
        if Array.length devices > 0 then (Me.Device.get_attributes devices.(0)).has_unified_memory
        else false
      with _ -> false (* Default to false if device check fails *)
    in
    if check_device_memory () then None (* Keep None: host ptr -> buffer_ptr conversion needed *)
    else None

  (* Initialization *)
  let initialized = ref false

  let initialize _config =
    if not !initialized then (
      initialize_device_list (); (* Ensure devices are loaded *)
      initialized := true)

  let is_initialized () = !initialized

  (* Device Management *)
  let num_devs = lazy (Array.length (get_all_metal_devices ()))
  let devices_cache = ref @@ Array.create ~len:0 None (* Resize as needed *)

  let get_device ~(ordinal : int) : device =
    if not (is_initialized ()) then raise (Utils.User_error "Metal backend not initialized");
    let num_devices = Lazy.force num_devs in
    if ordinal < 0 || num_devices <= ordinal then
      invalid_arg [%string "Metal_backend.get_device %{ordinal#Int}: invalid ordinal"];
    if Array.length !devices_cache <= ordinal then (
      let old_cache, old_len = (!devices_cache, Array.length !devices_cache) in
      devices_cache :=
        Array.init (ordinal + 1) ~f:(fun i -> if i < old_len then old_cache.(i) else None));
    let default () =
      let metal_device = (get_all_metal_devices ()).(ordinal) in
      let result_device = make_device metal_device ~ordinal in
      !devices_cache.(ordinal) <- Some result_device;
      result_device
    in
    Option.value_or_thunk !devices_cache.(ordinal) ~default

  let num_devices () = if not (is_initialized ()) then 0 else Lazy.force num_devs

  let new_stream (device_wrapper : device) : stream =
    let metal_device = device_wrapper.dev in
    let queue = Me.CommandQueue.on_device metal_device in
    let shared_event_obj = Me.SharedEvent.on_device metal_device in
    let counter = Utils.atomic_make Unsigned.ULLong.one in (* Next value = 1 *)
    let runner = { queue; event = shared_event_obj; counter } in
    make_stream device_wrapper runner

  (* --- Event Handling --- *)
  let get_runner_event stream = stream.runner.event
  let get_runner_counter stream = stream.runner.counter
  let get_runner_queue stream = stream.runner.queue

  let is_done event =
    let shared_event, target_value = event in
    try
      let current_value = Me.SharedEvent.get_signaled_value shared_event in
      Unsigned.ULLong.(current_value >= target_value)
    with exn ->
      [%log "Exception in is_done:", (exn : exn)];
      false (* Assume not done if check fails *)

  let sync event =
    if not (is_done event) then
      let shared_event, target_value = event in
      let timeout_max = Unsigned.ULLong.max_int in
      try
        ignore
          (Me.SharedEvent.wait_until_signaled_value shared_event ~value:target_value
             ~timeout_ms:timeout_max)
      with exn -> [%log "Exception in sync:", (exn : exn)] (* Log and continue *)

  let will_wait_for context event =
    let stream = context.stream in
    let queue = get_runner_queue stream in
    let shared_event, target_value = event in
    try
      let command_buffer = Me.CommandBuffer.on_queue queue in
      Me.CommandBuffer.encode_wait_for_event command_buffer shared_event target_value;
      Me.CommandBuffer.commit command_buffer
    with exn -> [%log "Exception in will_wait_for:", (exn : exn)]

  let all_work stream =
    let queue = get_runner_queue stream in
    let shared_event = get_runner_event stream in
    let counter = get_runner_counter stream in
    let next_value = Utils.atomic_fetch_and_add counter Unsigned.ULLong.one in
    try
      let command_buffer = Me.CommandBuffer.on_queue queue in
      Me.CommandBuffer.encode_signal_event command_buffer shared_event next_value;
      Me.CommandBuffer.commit command_buffer;
      (shared_event, next_value)
    with exn ->
      [%log "Exception in all_work:", (exn : exn)];
      (* Return a potentially problematic event *)
      (shared_event, next_value)

  let await stream =
    try
      let queue = get_runner_queue stream in
      let command_buffer = Me.CommandBuffer.on_queue queue in
      Me.CommandBuffer.commit command_buffer;
      Me.CommandBuffer.wait_until_completed command_buffer
    with exn -> [%log "Exception in await:", (exn : exn)]

  let is_idle stream =
    (* Approximate check *)
    try
      let shared_event = get_runner_event stream in
      let counter = get_runner_counter stream in
      let current_signaled = Me.SharedEvent.get_signaled_value shared_event in
      let expected_signaled = Unsigned.ULLong.pred (Utils.atomic_get counter) in
      Unsigned.ULLong.equal current_signaled expected_signaled
    with exn ->
      [%log "Exception in is_idle:", (exn : exn)];
      false

  (* --- Configuration and Info --- *)
  let suggested_num_streams _device =
    match Config.config with
    | Only_devices_parallel | For_parallel_copying | Most_parallel_streams -> 1

  let get_used_memory _device = Atomic.get allocated_memory
  let get_global_debug_info () = Sexp.Atom "Metal global debug info NYI"
  let get_debug_info stream = Sexp.message "Metal stream debug info NYI" [ ("stream_id", sexp_of_int stream.stream_id) ]

  (* --- Copy Operations --- *)
  let commit_and_wait cmd_buffer =
    try
      Me.CommandBuffer.commit cmd_buffer;
      Me.CommandBuffer.wait_until_completed cmd_buffer
    with exn -> [%log "Exception during commit_and_wait:", (exn : exn)]

  let from_host ~dst_ptr ~dst host_nd =
    let stream = dst.stream in
    let queue = get_runner_queue stream in
    let size_in_bytes = Ndarray.size_in_bytes host_nd in
    try
      if size_in_bytes = 0 then () (* Skip empty copy *)
      else if Option.is_some use_host_memory then (* Assumes Shared Memory *)
         let dst_contents = Me.Buffer.contents dst_ptr in
         let src_ptr = Ndarray.get_voidptr_not_managed host_nd in
         Ctypes_memory_stubs.memcpy ~dst:dst_contents ~src:src_ptr ~size:size_in_bytes
         (* TODO: Need `didModifyRange` if using managed buffers? *)
      else (* Use Blit Copy via Temp Buffer *)
         let command_buffer = Me.CommandBuffer.on_queue queue in
         let temp_options = Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_write_combined) in
         let temp_buffer = Me.Buffer.on_device stream.device.dev ~bytes:(Ndarray.get_voidptr_not_managed host_nd) ~length:size_in_bytes temp_options in
         let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
         Me.BlitCommandEncoder.copy_from_buffer blit_encoder
           ~source_buffer:temp_buffer ~source_offset:0
           ~destination_buffer:dst_ptr ~destination_offset:0
           ~size:size_in_bytes;
         Me.BlitCommandEncoder.end_encoding blit_encoder;
         commit_and_wait command_buffer (* Synchronous for now *)
    with exn -> [%log "Exception in from_host:", (exn : exn)]


  let to_host ~src_ptr ~src host_nd =
     let stream = src.stream in
     let queue = get_runner_queue stream in
     let size_in_bytes = Ndarray.size_in_bytes host_nd in
     try
       if size_in_bytes = 0 then () (* Skip empty copy *)
       else if Option.is_some use_host_memory then (* Assumes Shared Memory *)
         let command_buffer = Me.CommandBuffer.on_queue queue in
         let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
         Me.BlitCommandEncoder.synchronize_resource blit_encoder (Me.Buffer.super src_ptr); (* Ensure GPU writes visible *)
         Me.BlitCommandEncoder.end_encoding blit_encoder;
         commit_and_wait command_buffer; (* Wait for sync *)
         let src_contents = Me.Buffer.contents src_ptr in
         let dst_ptr = Ndarray.get_voidptr_not_managed host_nd in
         Ctypes_memory_stubs.memcpy ~dst:dst_ptr ~src:src_contents ~size:size_in_bytes
       else (* Use Blit Copy via Temp Buffer *)
         let command_buffer = Me.CommandBuffer.on_queue queue in
         let temp_options = Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache) in (* Read back *)
         let temp_buffer = Me.Buffer.on_device stream.device.dev ~length:size_in_bytes temp_options in
         let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
         Me.BlitCommandEncoder.copy_from_buffer blit_encoder
           ~source_buffer:src_ptr ~source_offset:0
           ~destination_buffer:temp_buffer ~destination_offset:0
           ~size:size_in_bytes;
         Me.BlitCommandEncoder.synchronize_resource blit_encoder (Me.Buffer.super temp_buffer); (* Sync temp buffer *)
         Me.BlitCommandEncoder.end_encoding blit_encoder;
         commit_and_wait command_buffer; (* Wait for blit and sync *)
         let temp_contents = Me.Buffer.contents temp_buffer in
         Ctypes_memory_stubs.memcpy ~dst:(Ndarray.get_voidptr_not_managed host_nd) ~src:temp_contents ~size:size_in_bytes
     with exn -> [%log "Exception in to_host:", (exn : exn)]

  let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src =
    let dst_stream = dst.stream in
    let src_stream = src.stream in
    let dst_queue = get_runner_queue dst_stream in
    let size_in_bytes = Lazy.force tn.Tn.size_in_bytes in

    try
      if size_in_bytes = 0 then () (* Skip empty copy *)
      else
        let actual_dst_ptr =
          match into_merge_buffer with
          | No -> Option.value_exn ~message:"device_to_device: No and dst_ptr=None" dst_ptr
          | Copy ->
              let merge_buf_ref = dst_stream.merge_buffer in
              let current_merge_buf = !merge_buf_ref in
              let target_buf =
                match current_merge_buf with
                | Some buf when buf.size_in_bytes >= size_in_bytes -> buf.ptr
                | _ ->
                    let new_buf = Alloc_buffer.alloc_buffer ~size_in_bytes dst_stream in
                    merge_buf_ref := Some new_buf;
                    new_buf.ptr
              in
              dst_stream.updating_for_merge_buffer <- Some (tn, None);
              target_buf
          | Streaming_for _task ->
              if dst_stream.device.ordinal <> src_stream.device.ordinal then
                failwith "Streaming_for across different Metal devices not yet supported";
              dst_stream.merge_buffer := Some { ptr = src_ptr; size_in_bytes };
              dst_stream.updating_for_merge_buffer <- Some (tn, None);
              src_ptr (* Indicate skip physical copy *)
        in

        (* Skip copy if source and destination pointers are physically the same *)
        if not (phys_equal actual_dst_ptr src_ptr) then (
          let command_buffer = Me.CommandBuffer.on_queue dst_queue in
          let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
          (* TODO: Add proper event synchronization between streams *)
          Me.BlitCommandEncoder.copy_from_buffer blit_encoder ~source_buffer:src_ptr
            ~source_offset:0 ~destination_buffer:actual_dst_ptr ~destination_offset:0
            ~size:size_in_bytes;
          Me.BlitCommandEncoder.end_encoding blit_encoder;
          commit_and_wait command_buffer (* Synchronous for now *)
        )
    with exn -> [%log "Exception in device_to_device:", (exn : exn)]

  (* --- Compilation and Linking --- *)
  type code = {
    metal_source : string; (* Store source, compile during link *)
    func_name : string;
    params : (string * param_source) list;
    bindings : Indexing.unit_bindings;
    traced_store : Low_level.traced_store;
  }
  [@@deriving sexp_of]

  type code_batch = {
    metal_source : string; (* Store combined source *)
    funcs : (string * (string * param_source) list) option array; (* func_name * params *)
    bindings : Indexing.unit_bindings;
    traced_stores : Low_level.traced_store option array;
  }
  [@@deriving sexp_of]

  module C_syntax_config (Input : sig
    val procs : Low_level.optimized array
  end) = struct
    (* Same as CUDA for now, adjust specific syntax/types *)
    type nonrec buffer_ptr = buffer_ptr

    let procs = Input.procs
    let use_host_memory = use_host_memory
    let logs_to_stdout = true (* Metal printf goes to stdout/stderr *)
    let main_kernel_prefix = "kernel"
    let kernel_prep_line = "" (* Metal grid setup is external *)

    let includes =
      [ "<metal_stdlib>"; "<metal_math>"; "<metal_compute>"; "<metal_atomic>"; "<metal_half>" ]

    let typ_of_prec = function
      | Ops.Byte_prec _ -> "uint8_t"
      | Half_prec _ -> "half"
      | Single_prec _ -> "float"
      | Double_prec _ -> "double"
      | Void_prec -> "void"

    let ternop_syntax _prec op =
      match op with
      | Ops.Where -> ("select(", ", ", ", ", ")") (* select(false, true, condition) *)
      | FMA -> ("fma(", ", ", ", ", ")")

    let binop_syntax prec op =
      let f op_str = ("(", " " ^ op_str ^ " ", ")") in
      let func fn = (fn ^ "(", ", ", ")") in
      match (op, prec) with
      | Add, _ -> f "+" | Sub, _ -> f "-" | Mul, _ -> f "*" | Div, _ -> f "/"
      | Mod, _ -> func "fmod"
      | Max, _ -> func "fmax" | Min, _ -> func "fmin"
      | Cmpeq, _ -> f "==" | Cmpne, _ -> f "!=" | Cmplt, _ -> f "<"
      | And, _ -> f "&&" | Or, _ -> f "||"
      | Relu_gate, _ -> ("(", " > 0 ?", " : 0)")
      | Satur01_gate, _ -> ("(abs(", ") > 0 ? 0 : (", ")")
      | ToPowOf, _ -> func "pow"
      | Arg1 | Arg2 -> invalid_arg "Metal C_syntax_config: Arg1/Arg2 not operators"

    let unop_syntax prec op =
      let f fn = (fn ^ "(", ")") in
      match (op, prec) with
      | Identity, _ -> ("", "") | Neg, _ -> ("-", "") (* Prefix negation *)
      | Exp, _ -> f "exp" | Log, _ -> f "log" | Exp2, _ -> f "exp2" | Log2, _ -> f "log2"
      | Sin, _ -> f "sin" | Cos, _ -> f "cos" | Sqrt, _ -> f "sqrt"
      | Relu, Half_prec _ -> ("max(0.0h", ", ", ")")
      | Relu, _ -> ("max(0.0f", ", ", ")")
      | Satur01, _ -> ("clamp(", ", 0.0, 1.0)")
      | Recip, _ -> ("(1.0 / ", ")")
      | Recip_sqrt, _ -> f "rsqrt"
      | Tanh_approx, _ -> f "tanh"
      | Not, _ -> ("!", "") (* Logical not *)

    let convert_precision ~from ~to_ =
      if Ops.equal_prec from to_ then ("", "")
      else ("(" ^ typ_of_prec to_ ^ ")(", ")")
  end

  let compile_metal_source ~name ~source ~device =
    let options = Me.CompileOptions.init () in
    Me.CompileOptions.set_language_version options Me.CompileOptions.LanguageVersion.version_3_1;
    if Utils.debug_log_from_routines () then Me.CompileOptions.set_enable_logging options true;

    if Utils.settings.output_debug_files_in_build_directory then (
      let metal_file = Utils.build_file (name ^ ".metal") in
      Out_channel.write_all metal_file ~data:source;
      [%log "Wrote metal source to file: %s" metal_file]);

    try Me.Library.on_device device ~source options
    with Failure msg ->
      let error_msg = Printf.sprintf "Metal compilation failed for %s:\n%s\nSource:\n%s" name msg source in
      Stdio.prerr_endline error_msg;
      failwith error_msg

  let compile ~name bindings lowered =
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = [| lowered |]
    end)) in
    let idx_params = Indexing.bound_symbols bindings in
    let b = Buffer.create 4096 in
    let ppf = Stdlib.Format.formatter_of_buffer b in
    Syntax.print_includes ppf;
    (* Add Metal address space qualifiers *)
    let add_addr_space p_decl =
      if String.is_suffix p_decl ~suffix:"*" then
        String.substr_replace_all p_decl ~pattern:"*" ~with_:"device *"
      else p_decl
    in
    let params =
      Syntax.compile_proc ~name ppf idx_params
        {
          lowered with
          params = List.map lowered.params ~f:(fun (p, s) -> (add_addr_space p, s));
        }
    in
    let source = Buffer.contents b in
    { metal_source = source; func_name = name; params; bindings; traced_store = lowered.traced_store }

  let compile_batch ~names bindings lowereds =
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = Array.filter_opt lowereds
    end)) in
    let idx_params = Indexing.bound_symbols bindings in
    let b = Buffer.create 4096 in
    let ppf = Stdlib.Format.formatter_of_buffer b in
    Syntax.print_includes ppf;
    let add_addr_space p_decl =
      if String.is_suffix p_decl ~suffix:"*" then
        String.substr_replace_all p_decl ~pattern:"*" ~with_:"device *"
      else p_decl
    in
    let funcs =
      Array.map2_exn names lowereds
        ~f:
          (Option.map2 ~f:(fun name lowered ->
               let params =
                 Syntax.compile_proc ~name ppf idx_params
                   {
                     lowered with
                     params = List.map lowered.params ~f:(fun (p, s) -> (add_addr_space p, s));
                   }
               in
               (name, params)))
    in
    let source = Buffer.contents b in
    let traced_stores = Array.map lowereds ~f:(Option.map ~f:(fun l -> l.traced_store)) in
    { metal_source = source; funcs; bindings; traced_stores }


  let link_proc ~prior_context ~library ~func_name ~params ~lowered_bindings ~ctx_arrays
      run_log_id =
    let stream = prior_context.stream in
    let device = stream.device.dev in
    let queue = get_runner_queue stream in
    let runner_label = get_name stream in
    let func = Me.Library.new_function_with_name library func_name in
    if Me.Objc.is_nil func then
      failwith ("Failed to get function " ^ func_name ^ " from Metal library");
    let pso, _ = Me.ComputePipelineState.on_device_with_function device func in
    if Me.Objc.is_nil pso then
       failwith ("Failed to create PSO for function " ^ func_name);


    let work () : unit =
      [%log_result "Launching", func_name, "on", runner_label, (run_log_id : int)];
      try
        let command_buffer = Me.CommandBuffer.on_queue queue in
        let encoder = Me.ComputeCommandEncoder.on_buffer command_buffer in
        Me.ComputeCommandEncoder.set_compute_pipeline_state encoder pso;

        (* Set arguments *)
        List.iteri params ~f:(fun index (_p_name, p_source) ->
            match p_source with
            | Param_ptr tn -> (
                try
                  let buffer = Map.find_exn ctx_arrays tn in
                  Me.ComputeCommandEncoder.set_buffer encoder ~index buffer
                with Not_found_s _ ->
                  failwith
                    [%string
                      "Param_ptr %{Tn.debug_name tn} not found in ctx_arrays for %{func_name}"]
              )
            | Static_idx s ->
                let value = !(Indexing.find_exn lowered_bindings s) in
                let size = Ctypes.sizeof Ctypes.int in
                let bytes_ptr = Ctypes.(allocate int value |> to_voidp) in
                Me.ComputeCommandEncoder.set_bytes encoder ~bytes:bytes_ptr ~length:size ~index
            | Merge_buffer -> (
                match !(stream.merge_buffer) with
                | Some merge_buf -> Me.ComputeCommandEncoder.set_buffer encoder ~index merge_buf.ptr
                | None -> failwith [%string "Merge_buffer requested but not set for %{func_name}"]
              )
            | Log_file_name ->
                let size = Ctypes.sizeof Ctypes.int in
                let bytes_ptr = Ctypes.(allocate int run_log_id |> to_voidp) in
                Me.ComputeCommandEncoder.set_bytes encoder ~bytes:bytes_ptr ~length:size ~index
          );

        (* Dispatch - TODO: Determine grid/group sizes properly *)
        let max_threads = Me.ComputePipelineState.get_max_total_threads_per_threadgroup pso in
        let width = Int.min max_threads 32 in (* Example: Use a small group size *)
        let threads_per_group = Me.Size.make ~width ~height:1 ~depth:1 in
        let groups_per_grid = Me.Size.make ~width:1 ~height:1 ~depth:1 in (* Example: single group *)
        Me.ComputeCommandEncoder.dispatch_threadgroups encoder ~threadgroups_per_grid:groups_per_grid
          ~threads_per_threadgroup:threads_per_group;

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
        context_lifetime = (library, pso, ctx_arrays); (* Keep library and PSO alive *)
        description = "launches " ^ func_name ^ " on " ^ runner_label;
        work;
      }

  let link prior_context code ctx_arrays =
    let device = prior_context.stream.device.dev in
    let library = compile_metal_source ~name:code.func_name ~source:code.metal_source ~device in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols code.bindings) ~f:(fun s -> (s, ref 0))
    in
    let run_log_id = if Utils.debug_log_from_routines () then Utils.get_global_run_id () else 0 in
    let task =
      link_proc ~prior_context ~library ~func_name:code.func_name ~params:code.params
        ~lowered_bindings ~ctx_arrays run_log_id
    in
    (lowered_bindings, task)

  let link_batch prior_context code_batch ctx_arrays_opts =
    let device = prior_context.stream.device.dev in
    let library = compile_metal_source ~name:"batch" ~source:code_batch.metal_source ~device in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols code_batch.bindings) ~f:(fun s -> (s, ref 0))
    in
    let run_log_id = if Utils.debug_log_from_routines () then Utils.get_global_run_id () else 0 in

    let tasks =
      Array.mapi code_batch.funcs ~f:(fun i func_opt ->
          Option.bind func_opt (fun (func_name, params) ->
              Option.map ctx_arrays_opts.(i) ~f:(fun ctx_arrays ->
                  link_proc ~prior_context ~library ~func_name ~params ~lowered_bindings
                    ~ctx_arrays run_log_id)))
    in
    (lowered_bindings, tasks)
end
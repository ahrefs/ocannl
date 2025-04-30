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
    counter : ullong; (* Next value to signal *)
  }
  [@@deriving sexp_of]

  (* Represents a point in time on a specific runner *)
  type event = { shared : Me.SharedEvent.t; value : ullong } [@@deriving sexp_of]

  let name = "metal"
end

module Device_stream = Backend_impl.Device_types (Device_config)

(* Bring types into scope *)
open Device_config

(* Manual tracking for allocated memory *)
let allocated_memory = Atomic.make 0

let track_allocation (buffer : Me.Buffer.t) =
  (* Relying on ARC + finalizer on the payload record *)
  let size = Me.Resource.get_allocated_size (Me.Buffer.super buffer) in
  Stdlib.Gc.finalise (fun _ -> ignore (Atomic.fetch_and_add allocated_memory (-size))) buffer;
  ignore (Atomic.fetch_and_add allocated_memory size)

module Alloc_buffer = struct
  include Device_stream

  let resource_options =
    (* Use Shared storage for unified memory, WriteCombined for CPU writes, Tracked hazards *)
    Me.ResourceOptions.(
      storage_mode_shared + cpu_cache_mode_write_combined + hazard_tracking_mode_tracked)

  let alloc_buffer ?old_buffer ~size_in_bytes (stream : stream) =
    let device = stream.device.dev in
    match old_buffer with
    | Some ({ size_in_bytes = old_size; _ } as buffer) when size_in_bytes <= old_size -> buffer
    | _ ->
        (* ARC should handle the old buffer *)
        let new_buffer_obj = Me.Buffer.on_device device ~length:size_in_bytes resource_options in
        track_allocation new_buffer_obj;
        { ptr = new_buffer_obj; size_in_bytes }

  let alloc_zero_init_array prec ~dims (stream : stream) =
    let size_in_bytes =
      (if Array.length dims = 0 then 0 else Array.reduce_exn dims ~f:( * )) * Ops.prec_in_bytes prec
    in
    let device = stream.device.dev in
    let buffer = Me.Buffer.on_device device ~length:size_in_bytes resource_options in
    track_allocation buffer;
    (* Zero initialize the buffer using a blit command encoder *)
    let command_buffer = Me.CommandBuffer.on_queue stream.runner.queue in
    let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
    Me.BlitCommandEncoder.fill_buffer blit_encoder buffer
      { location = 0; length = size_in_bytes }
      ~value:0;
    Me.BlitCommandEncoder.end_encoding blit_encoder;
    Me.CommandBuffer.commit command_buffer;
    Me.CommandBuffer.wait_until_completed command_buffer;
    buffer

  (* Rely on ARC and the finalizer attached in track_allocation *)
  let free_buffer = None
end

(* Functor defining the backend *)
module Fresh (Config : sig
  val config : Ir.Backend_intf.config
end) =
struct
  (* Include the device setup with types and allocation *)
  include Backend_impl.Device (Device_stream) (Alloc_buffer)

  (* Global state for Metal devices *)
  let metal_devices : Me.Device.t array = Me.Device.copy_all_devices ()
  let () = assert (Array.length metal_devices > 0)

  (* Metal has unified memory on Apple Silicon, so we can use host memory *)
  let use_host_memory =
    Some
      (fun ~size_in_bytes:length ptr ->
        (* Need to create a Metal buffer that wraps the host memory *)
        let device = metal_devices.(0) in
        Me.Buffer.on_device_with_bytes_no_copy device ~bytes:ptr ~length
          Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache))

  (* Device Management *)
  let num_devs = Array.length metal_devices
  let devices_cache = Array.create ~len:num_devs None

  let get_device ~(ordinal : int) : device =
    if ordinal < 0 || num_devs <= ordinal then
      invalid_arg [%string "Metal_backend.get_device %{ordinal#Int}: invalid ordinal"];
    let default () =
      let metal_device = metal_devices.(ordinal) in
      let result_device = make_device metal_device ~ordinal in
      devices_cache.(ordinal) <- Some result_device;
      result_device
    in
    Option.value_or_thunk devices_cache.(ordinal) ~default

  let num_devices () =
    (* FIXME: refactor the whole backend interface to use constant num_devices per backend
       instance *)
    num_devs

  let new_stream (device_wrapper : device) : stream =
    let metal_device = device_wrapper.dev in
    let queue = Me.CommandQueue.on_device metal_device in
    let shared_event_obj = Me.SharedEvent.on_device metal_device in
    let counter = Unsigned.ULLong.one in
    (* Next value = 1 *)
    let runner = { queue; event = shared_event_obj; counter } in
    make_stream device_wrapper runner

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
    let stream = context.stream in
    let queue = stream.runner.queue in
    let command_buffer = Me.CommandBuffer.on_queue queue in
    Me.CommandBuffer.encode_wait_for_event command_buffer
      (Me.SharedEvent.super event.shared)
      event.value;
    Me.CommandBuffer.commit command_buffer

  let all_work stream =
    let queue = stream.runner.queue in
    let shared_event = stream.runner.event in
    let counter = stream.runner.counter in
    let next_value = Unsigned.ULLong.add counter Unsigned.ULLong.one in
    let command_buffer = Me.CommandBuffer.on_queue queue in
    Me.CommandBuffer.encode_signal_event command_buffer
      (Me.SharedEvent.super shared_event)
      next_value;
    Me.CommandBuffer.commit command_buffer;
    { shared = shared_event; value = next_value }

  let await stream =
    let queue = stream.runner.queue in
    let command_buffer = Me.CommandBuffer.on_queue queue in
    Me.CommandBuffer.commit command_buffer;
    Me.CommandBuffer.wait_until_completed command_buffer

  let is_idle stream =
    (* FIXME: store the latest CommandBuffer with the stream runner and check that it's completed *)
    let counter = stream.runner.counter in
    let current_signaled = Me.SharedEvent.get_signaled_value stream.runner.event in
    let expected_signaled = Unsigned.ULLong.pred counter in
    Unsigned.ULLong.equal current_signaled expected_signaled

  (* --- Configuration and Info --- *)
  let suggested_num_streams _device =
    match Config.config with
    | Only_devices_parallel | For_parallel_copying | Most_parallel_streams -> 1

  let get_used_memory _device = Atomic.get allocated_memory
  let get_global_debug_info () = Sexp.Atom "Metal global debug info NYI"

  let get_debug_info stream =
    Sexp.message "Metal stream debug info NYI" [ ("stream_id", sexp_of_int stream.stream_id) ]

  (* --- Copy Operations --- *)
  let commit_and_wait cmd_buffer =
    Me.CommandBuffer.commit cmd_buffer;
    Me.CommandBuffer.wait_until_completed cmd_buffer

  let from_host ~dst_ptr ~dst hosted =
    (* Copy from host memory to Metal buffer *)
    let size_in_bytes = Ndarray.size_in_bytes hosted in
    let command_buffer = Me.CommandBuffer.on_queue dst.stream.runner in

    (* Get host memory pointer *)
    let host_ptr = Ndarray.get_fatptr_not_managed hosted in
    let (Ctypes_static.CPointer dst_fatptr) = Me.Buffer.contents dst_ptr in
    if Ctypes_ptr.Fat.compare dst_fatptr host_ptr <> 0 then (
      (* Create a temporary host buffer to bridge the gap *)
      let temp_buffer =
        Me.Buffer.on_device_with_bytes_no_copy dst.stream.device.dev
          ~bytes:(Ctypes_static.CPointer host_ptr) ~length:size_in_bytes
          Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache)
      in
      let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
      Me.BlitCommandEncoder.copy_from_buffer blit_encoder ~source_buffer:temp_buffer
        ~source_offset:0 ~destination_buffer:dst_ptr ~destination_offset:0 ~size:size_in_bytes;
      Me.BlitCommandEncoder.end_encoding blit_encoder;
      Me.CommandBuffer.commit command_buffer)

  let to_host ~src_ptr ~src hosted =
    (* Copy from Metal buffer to host memory *)
    let size_in_bytes = Ndarray.size_in_bytes hosted in
    let command_buffer = Me.CommandBuffer.on_queue src.stream.runner in

    (* Get host memory pointer *)
    let host_ptr = Ndarray.get_fatptr_not_managed hosted in
    let (Ctypes_static.CPointer src_fatptr) = Me.Buffer.contents src_ptr in
    if Ctypes_ptr.Fat.compare src_fatptr host_ptr <> 0 then (
      (* Create a temporary host buffer to bridge the gap *)
      let temp_buffer =
        Me.Buffer.on_device_with_bytes_no_copy src.stream.device.dev
          ~bytes:(Ctypes_static.CPointer host_ptr) ~length:size_in_bytes
          Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache)
      in
      let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
      Me.BlitCommandEncoder.copy_from_buffer blit_encoder ~source_buffer:src_ptr ~source_offset:0
        ~destination_buffer:temp_buffer ~destination_offset:0 ~size:size_in_bytes;
      Me.BlitCommandEncoder.end_encoding blit_encoder;
      Me.CommandBuffer.commit command_buffer)

  let opt_alloc_merge_buffer ~size_in_bytes stream : unit =
    stream.merge_buffer :=
      Some (alloc_buffer ?old_buffer:!(stream.merge_buffer) ~size_in_bytes stream)

  let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src =
    let same_device = dst.stream.device.ordinal = src.stream.device.ordinal in
    let size_in_bytes = Lazy.force tn.Tn.size_in_bytes in

    let memcpy ~dst_ptr =
      (* Always use explicit copy as Metal doesn't have peer-to-peer memory access like CUDA *)
      let command_buffer = Me.CommandBuffer.on_queue dst.stream.runner.queue in
      let blit_encoder = Me.BlitCommandEncoder.on_buffer command_buffer in
      Me.BlitCommandEncoder.copy_from_buffer blit_encoder ~source_buffer:src_ptr ~source_offset:0
        ~destination_buffer:dst_ptr ~destination_offset:0 ~size:size_in_bytes;
      Me.BlitCommandEncoder.end_encoding blit_encoder;
      Me.CommandBuffer.commit command_buffer
    in

    match (into_merge_buffer, dst_ptr) with
    | No, None -> invalid_arg "Metal_backend.device_to_device: missing dst_ptr"
    | No, Some dst_ptr -> memcpy ~dst_ptr
    | Streaming_for _, _ ->
        if same_device then dst.stream.merge_buffer := Some { ptr = src_ptr; size_in_bytes }
        else (
          (* Fall back to copy for different devices *)
          opt_alloc_merge_buffer ~size_in_bytes dst.stream;
          let buffer = Option.value_exn ~here:[%here] !(dst.stream.merge_buffer) in
          memcpy ~dst_ptr:buffer.ptr)
    | Copy, _ ->
        opt_alloc_merge_buffer ~size_in_bytes dst.stream;
        let buffer = Option.value_exn ~here:[%here] !(dst.stream.merge_buffer) in
        memcpy ~dst_ptr:buffer.ptr

  (* --- Compilation and Linking --- *)
  type code = {
    metal_source : string; (* Store source, compile during link if not already compiled *)
    compiled_code : Me.Library.t option array; (* Store compiled code per device *)
    func_name : string;
    params : (string * param_source) list;
    bindings : Indexing.unit_bindings;
    traced_store : Low_level.traced_store;
  }
  [@@deriving sexp_of]

  type code_batch = {
    metal_source : string; (* Store combined source *)
    compiled_code : Me.Library.t option array; (* Store compiled code per device *)
    funcs : (string * (string * param_source) list) option array; (* func_name * params *)
    bindings : Indexing.unit_bindings;
    traced_stores : Low_level.traced_store option array;
  }
  [@@deriving sexp_of]

  module C_syntax_config (Input : sig
    val procs : Low_level.optimized array
  end) =
  struct
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
      | Relu_gate, _ -> ("(", " > 0 ?", " : 0)")
      | Satur01_gate, _ -> ("(abs(", ") > 0 ? 0 : (", ")")
      | ToPowOf, _ -> func "pow"
      | Arg1, _ | Arg2, _ -> invalid_arg "Metal C_syntax_config: Arg1/Arg2 not operators"

    let unop_syntax prec op =
      let f fn = (fn ^ "(", ")") in
      match (op, prec) with
      | Ops.Identity, _ -> ("", "")
      | Neg, _ -> ("-", "") (* Prefix negation *)
      | Exp, _ -> f "exp"
      | Log, _ -> f "log"
      | Exp2, _ -> f "exp2"
      | Log2, _ -> f "log2"
      | Sin, _ -> f "sin"
      | Cos, _ -> f "cos"
      | Sqrt, _ -> f "sqrt"
      | Relu, Ops.Half_prec _ -> ("max(0.0h, ", ")")
      | Relu, _ -> ("max(0.0f, ", ")")
      | Satur01, _ -> ("clamp(", ", 0.0, 1.0)")
      | Recip, _ -> ("(1.0 / ", ")")
      | Recip_sqrt, _ -> f "rsqrt"
      | Tanh_approx, _ -> f "tanh"
      | Not, _ -> ("!", "")
    (* Logical not *)

    let convert_precision ~from ~to_ =
      if Ops.equal_prec from to_ then ("", "") else ("(" ^ typ_of_prec to_ ^ ")(", ")")
  end

  let%diagn_sexp compile_metal_source ~name ~source ~device =
    let options = Me.CompileOptions.init () in
    Me.CompileOptions.set_language_version options Me.CompileOptions.LanguageVersion.version_3_1;
    if Utils.debug_log_from_routines () then Me.CompileOptions.set_enable_logging options true;

    if Utils.with_runtime_debug () then (
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
    end)) in
    let idx_params = Indexing.bound_symbols bindings in
    let b = Buffer.create 4096 in
    let ppf = Stdlib.Format.formatter_of_buffer b in
    Syntax.print_includes ppf;
    (* Add Metal address space qualifiers *)
    let params = Syntax.compile_proc ~name ppf idx_params lowered in
    let source = Buffer.contents b in
    {
      metal_source = source;
      compiled_code = Array.create ~len:1 None;
      func_name = name;
      params;
      bindings;
      traced_store = lowered.traced_store;
    }

  let compile_batch ~names bindings lowereds =
    let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
      let procs = Array.filter_opt lowereds
    end)) in
    let idx_params = Indexing.bound_symbols bindings in
    let b = Buffer.create 4096 in
    let ppf = Stdlib.Format.formatter_of_buffer b in
    Syntax.print_includes ppf;
    let funcs =
      Array.map2_exn names lowereds
        ~f:
          (Option.map2 ~f:(fun name lowered ->
               let params = Syntax.compile_proc ~name ppf idx_params lowered in
               (name, params)))
    in
    let source = Buffer.contents b in
    let traced_stores = Array.map lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store)) in
    {
      metal_source = source;
      compiled_code = Array.create ~len:(Array.length lowereds) None;
      funcs;
      bindings;
      traced_stores;
    }

  let%diagn_sexp link_proc ~prior_context ~library ~func_name ~params ~lowered_bindings ~ctx_arrays
      run_log_id =
    let stream = prior_context.stream in
    let device = stream.device.dev in
    let queue = stream.runner.queue in
    let runner_label = get_name stream in
    let func = Me.Library.new_function_with_name library func_name in
    let pso, _ = Me.ComputePipelineState.on_device_with_function device func in

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
                      "Param_ptr %{Tn.debug_name tn} not found in ctx_arrays for %{func_name}"])
            | Static_idx s ->
                let value = !(Indexing.find_exn lowered_bindings s) in
                let size = Ctypes.sizeof Ctypes.int in
                let bytes_ptr = Ctypes.(allocate int value |> to_voidp) in
                Me.ComputeCommandEncoder.set_bytes encoder ~bytes:bytes_ptr ~length:size ~index
            | Merge_buffer -> (
                match !(stream.merge_buffer) with
                | Some merge_buf -> Me.ComputeCommandEncoder.set_buffer encoder ~index merge_buf.ptr
                | None -> failwith [%string "Merge_buffer requested but not set for %{func_name}"])
            | Log_file_name ->
                let size = Ctypes.sizeof Ctypes.int in
                let bytes_ptr = Ctypes.(allocate int run_log_id |> to_voidp) in
                Me.ComputeCommandEncoder.set_bytes encoder ~bytes:bytes_ptr ~length:size ~index);

        (* Dispatch - TODO: Determine grid/group sizes properly *)
        let max_threads = Me.ComputePipelineState.get_max_total_threads_per_threadgroup pso in
        let width = Int.min max_threads 32 in
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

  let get_global_run_id =
    let next_id = ref 0 in
    fun () ->
      Int.incr next_id;
      if !next_id < 0 then next_id := 0;
      !next_id

  let link prior_context code ctx_arrays =
    let device = prior_context.stream.device.dev in
    let library = compile_metal_source ~name:code.func_name ~source:code.metal_source ~device in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols code.bindings) ~f:(fun s -> (s, ref 0))
    in
    let run_log_id = if Utils.debug_log_from_routines () then get_global_run_id () else 0 in
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
    let run_log_id = if Utils.debug_log_from_routines () then get_global_run_id () else 0 in

    let tasks =
      Array.mapi code_batch.funcs ~f:(fun i func_opt ->
          Option.bind func_opt ~f:(fun (func_name, params) ->
              Option.map ctx_arrays_opts.(i) ~f:(fun ctx_arrays ->
                  link_proc ~prior_context ~library ~func_name ~params ~lowered_bindings ~ctx_arrays
                    run_log_id)))
    in
    (lowered_bindings, tasks)
end

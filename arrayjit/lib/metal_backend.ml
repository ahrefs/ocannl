open Base
open Ir
module Tn = Tnode
module Lazy = Utils.Lazy
module Mt = Metal
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

(* Metal doesn't need hook setup like CUDA *)
let () = ()

module Backend_buffer = struct
  type buffer_ptr = Mt.Buffer.t

  let sexp_of_buffer_ptr _ptr = Sexp.Atom "metal_buffer"

  include Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]
  end)
end

module Device_config = struct
  include Backend_buffer

  type dev = { dev : Mt.Device.t; primary_context : Mt.Device.t } [@@deriving sexp_of]
  type runner = Mt.CommandQueue.t [@@deriving sexp_of]
  type event = Mt.SharedEvent.t [@@deriving sexp_of]

  let name = "metal"
end

module Device_stream = Backend_impl.Device_types (Device_config)
open Device_config

(* In Metal we don't need to set context explicitly like in CUDA *)
let set_ctx _ctx = ()

module Alloc_buffer = struct
  include Device_stream

  let alloc_buffer ?old_buffer ~size_in_bytes stream =
    match old_buffer with
    | Some ({ size_in_bytes = old_size; _ } as buffer) when size_in_bytes <= old_size -> buffer
    | Some { ptr = _; _ } ->
        (* Metal buffers are reference counted, no explicit free needed *)
        {
          ptr =
            Mt.Buffer.on_device stream.device.dev.primary_context ~length:size_in_bytes
              Mt.ResourceOptions.storage_mode_shared;
          size_in_bytes;
        }
    | None ->
        {
          ptr =
            Mt.Buffer.on_device stream.device.dev.primary_context ~length:size_in_bytes
              Mt.ResourceOptions.storage_mode_shared;
          size_in_bytes;
        }

  let alloc_zero_init_array prec ~dims stream =
    let size_in_bytes =
      (if Array.length dims = 0 then 0 else Array.reduce_exn dims ~f:( * )) * Ops.prec_in_bytes prec
    in
    Mt.Buffer.on_device stream.device.dev.primary_context ~length:size_in_bytes
      Mt.ResourceOptions.storage_mode_shared

  (* Metal doesn't need explicit free, but we provide None for the interface *)
  let free_buffer = None
end

(* [initialized_devices] never forgets its entries, just like in CUDA backend *)
let initialized_devices = Hash_set.create (module Int)

module Fresh () = struct
  include Backend_impl.Device (Device_stream) (Alloc_buffer)

  (* Since Metal has unified memory architecture on Apple Silicon, we could use host memory
     directly, but for consistency we'll use Metal's storage_mode_shared buffers *)
  let use_host_memory = None
  let ctx_of (context : context) = context.stream.device.dev.primary_context

  (* Event handling *)
  let is_done event =
    (* Check if the event has been signaled with a value > 0 *)
    Unsigned.ULLong.compare (Mt.SharedEvent.signaled_value event) (Unsigned.ULLong.of_int 0) > 0

  let will_wait_for context event =
    (* Schedule a wait for this event in the stream *)
    let command_buffer = Mt.CommandQueue.command_buffer context.stream.runner in
    Mt.CommandBuffer.encode_wait_for_event command_buffer event (Unsigned.ULLong.of_int 1);
    Mt.CommandBuffer.commit command_buffer

  let sync event =
    (* Wait for the event to complete - metal doesn't have a direct synchronize method, so we
       implement polling for simplicity *)
    while not (is_done event) do
      Unix.nanosleep 0.001 |> ignore
    done

  let all_work stream =
    (* Signal an event after all current work in the stream *)
    let event = Mt.SharedEvent.on_device stream.device.dev.primary_context in
    let command_buffer = Mt.CommandQueue.command_buffer stream.runner in
    Mt.CommandBuffer.encode_signal_event command_buffer event (Unsigned.ULLong.of_int 1);
    Mt.CommandBuffer.commit command_buffer;
    event

  let global_config = ref For_parallel_copying

  (* No explicit Metal initialization needed *)
  let is_initialized, initialize =
    let initialized = ref false in
    let init (config : config) : unit =
      initialized := true;
      global_config := config
    in
    ((fun () -> !initialized), init)

  let num_devices =
   (* Metal typically has 1 device per system *)
   fun () ->
    1

  let devices = ref @@ Array.create ~len:(num_devices ()) None

  let get_used_memory (_device : device) =
    (* Metal doesn't provide direct memory usage info like CUDA, we would need to track it ourselves
       - returning 0 as a placeholder *)
    0

  let opt_alloc_merge_buffer ~size_in_bytes dev stream : unit =
    if
      Option.value_map ~default:true !(stream.merge_buffer) ~f:(fun buffer ->
          buffer.size_in_bytes < size_in_bytes)
    then
      (* No explicit free needed, just replace the reference *)
      stream.merge_buffer :=
        Some
          {
            ptr =
              Mt.Buffer.on_device dev.primary_context ~length:size_in_bytes
                Mt.ResourceOptions.storage_mode_shared;
            size_in_bytes;
          }

  let%track4_sexp finalize_device (_device : device) =
    (* Metal devices don't need explicit finalization like CUDA contexts *)
    Option.iter !Utils.advance_captured_logs ~f:(fun callback -> callback ())

  let%track3_sexp get_device ~(ordinal : int) : device =
    if num_devices () <= ordinal then
      invalid_arg [%string "Metal_backend.get_device %{ordinal#Int}: not enough devices"];
    (if Array.length !devices <= ordinal then
       let old, len = (!devices, Array.length !devices) in
       devices := Array.init (ordinal + 1) ~f:(fun i -> if i < len then old.(i) else None));
    let default () =
      let metal_device = Mt.Device.create_system_default () in
      let dev = { dev = metal_device; primary_context = metal_device } in
      if Utils.debug_log_from_routines () && not (Hash_set.mem initialized_devices ordinal) then ();
      (* Metal doesn't need PRINTF_FIFO_SIZE configuration *)
      Hash_set.add initialized_devices ordinal;
      let result = make_device dev ~ordinal in
      Stdlib.Gc.finalise finalize_device result;
      !devices.(ordinal) <- Some result;
      result
    in
    Option.value_or_thunk !devices.(ordinal) ~default

  let%track3_sexp new_stream (device : device) : stream =
    let command_queue = Mt.CommandQueue.on_device device.dev.primary_context in
    make_stream device command_queue

  let metal_properties =
    let cache =
      let%debug2_sexp f (ordinal : int) =
        let dev = get_device ~ordinal in
        lazy (Mt.Device.get_attributes dev.dev.dev)
      in
      lazy (Array.init (num_devices ()) ~f)
    in
    let%debug2_sexp get_props (device : device) : Mt.Device.attributes =
      let cache = Lazy.force cache in
      Lazy.force cache.(device.ordinal)
    in
    get_props

  let suggested_num_streams device =
    match !global_config with
    | Only_devices_parallel -> 1
    | For_parallel_copying ->
        2 (* Metal doesn't have async_engine_count, using a reasonable value *)
    | Most_parallel_streams ->
        (* Use the max_threads_per_threadgroup as a hint for parallelism capability *)
        let props = metal_properties device in
        props.max_threads_per_threadgroup.width / 32

  let await stream : unit =
    (* Create and execute a command buffer that we'll wait on *)
    let command_buffer = Mt.CommandQueue.command_buffer stream.runner in
    Mt.CommandBuffer.commit command_buffer;
    Mt.CommandBuffer.wait_until_completed command_buffer;
    Option.iter !Utils.advance_captured_logs ~f:(fun callback -> callback ())

  let is_idle _stream =
    (* Metal doesn't have a direct equivalent to CUDA's is_ready. Metal CommandQueues don't provide
       status information, so we assume it's ready *)
    true

  let from_host ~dst_ptr ~dst hosted =
    let command_buffer = Mt.CommandQueue.command_buffer dst.stream.runner in

    (* Copy data from host to device using Metal's storage_mode_shared *)
    let src_ptr = Ndarray.get_fatptr hosted in
    let size = Ndarray.size_in_bytes hosted in

    (* Get pointer to Metal buffer contents and copy data *)
    let dst_contents_ptr = Mt.Buffer.contents dst_ptr in
    Ctypes.memcpy ~dst:dst_contents_ptr ~src:src_ptr ~size;

    (* Notify Metal that we've modified the buffer *)
    let range = Mt.Buffer.NSRange.make ~location:0 ~length:size in
    Mt.Buffer.did_modify_range dst_ptr range;

    (* Create a blit encoder to ensure synchronization *)
    let encoder = Mt.CommandBuffer.blit_command_encoder command_buffer in
    Mt.BlitCommandEncoder.synchronize_resource ~self:encoder ~resource:(Obj.magic dst_ptr);
    Mt.BlitCommandEncoder.end_encoding encoder;

    Mt.CommandBuffer.commit command_buffer

  let to_host ~src_ptr ~src hosted =
    let command_buffer = Mt.CommandQueue.command_buffer src.stream.runner in

    (* Create blit encoder to synchronize the resource for CPU access *)
    let encoder = Mt.CommandBuffer.blit_command_encoder command_buffer in
    Mt.BlitCommandEncoder.synchronize_resource ~self:encoder ~resource:(Obj.magic src_ptr);
    Mt.BlitCommandEncoder.end_encoding encoder;

    (* Execute and wait for completion *)
    Mt.CommandBuffer.commit command_buffer;
    Mt.CommandBuffer.wait_until_completed command_buffer;

    (* Copy data from Metal buffer to host array *)
    let dst_ptr = Ndarray.get_fatptr hosted in
    let size = Ndarray.size_in_bytes hosted in
    let src_contents_ptr = Mt.Buffer.contents src_ptr in
    Ctypes.memcpy ~dst:dst_ptr ~src:src_contents_ptr ~size

  let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src =
    let dev = dst.stream.device in
    let same_device = dev.ordinal = src.stream.device.ordinal in
    let size_in_bytes = Lazy.force tn.Tn.size_in_bytes in

    let copy_buffers ~dst_ptr =
      if same_device then (
        (* Use blit encoder to copy between buffers *)
        let command_buffer = Mt.CommandQueue.command_buffer dst.stream.runner in
        let encoder = Mt.CommandBuffer.blit_command_encoder command_buffer in
        Mt.BlitCommandEncoder.copy_from_buffer ~self:encoder ~source_buffer:src_ptr ~source_offset:0
          ~destination_buffer:dst_ptr ~destination_offset:0 ~size:size_in_bytes;
        Mt.BlitCommandEncoder.end_encoding encoder;
        Mt.CommandBuffer.commit command_buffer)
      else
        (* Cross-device copy - Metal doesn't support this directly like CUDA *)
        (* We'd need to go through host memory *)
        invalid_arg "Metal_backend.device_to_device: cross-device copy not supported"
    in

    match (into_merge_buffer, dst_ptr) with
    | No, None -> invalid_arg "Metal_backend.device_to_device: missing dst_ptr"
    | No, Some dst_ptr -> copy_buffers ~dst_ptr
    | Streaming_for _, _ ->
        assert same_device;
        dst.stream.merge_buffer := Some { ptr = src_ptr; size_in_bytes }
    | Copy, _ ->
        opt_alloc_merge_buffer ~size_in_bytes dev.dev dst.stream;
        let buffer = Option.value_exn ~here:[%here] !(dst.stream.merge_buffer) in
        copy_buffers ~dst_ptr:buffer.ptr
end

type code = {
  traced_store : Low_level.traced_store;
  msl : string; (* Metal Shading Language code *)
  params : (string * param_source) list;
  bindings : Indexing.unit_bindings;
  name : string;
}
[@@deriving sexp_of]

type code_batch = {
  traced_stores : Low_level.traced_store option array;
  msl : string; (* Metal Shading Language code for all functions *)
  bindings : Indexing.unit_bindings;
  params_and_names : ((string * param_source) list * string) option array;
}
[@@deriving sexp_of]

type compiled_library = { library : Mt.Library.t; compile_options : Mt.CompileOptions.t }

let%diagn2_sexp metal_from_c ~name c_src =
  (* Convert C-like code to Metal Shading Language *)
  (* In a real implementation, we would do proper translation *)
  let name_metal = name ^ ".metal" in
  if Utils.settings.output_debug_files_in_build_directory then (
    let oc = Out_channel.open_text @@ Utils.build_file name_metal in
    Stdio.Out_channel.output_string oc c_src;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc);

  (* Metal kernel entry points have different syntax from CUDA *)
  let c_src_modified =
    Str.global_replace (Str.regexp "extern \"C\" __global__ void") "kernel void" c_src
  in

  (* Replace CUDA-specific intrinsics with Metal equivalents *)
  let msl_src =
    (* These are simplified examples - real translation would be more complex *)
    let src =
      Str.global_replace (Str.regexp "__syncthreads()") "threadgroup_barrier(mem_flags::mem_device)"
        c_src_modified
    in
    let src = Str.global_replace (Str.regexp "atomicAdd") "atomic_fetch_add_explicit" src in
    let src =
      Str.global_replace
        (Str.regexp "#include <cuda_fp16.h>")
        "#include <metal_stdlib>\n#include <metal_atomic>\nusing namespace metal;" src
    in
    src
  in

  msl_src

module C_syntax_config (Input : sig
  val procs : Low_level.optimized array
end) =
struct
  type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]

  let procs = Input.procs
  let use_host_memory = use_host_memory
  let logs_to_stdout = true

  (* Metal uses 'kernel' instead of CUDA's 'extern "C" __global__' *)
  let main_kernel_prefix = "kernel"

  (* Metal doesn't need thread ID check like CUDA *)
  let kernel_prep_line = ""

  (* Metal includes *)
  let includes = [ "#include <metal_stdlib>\n#include <metal_atomic>\nusing namespace metal;" ]

  (* Type mapping to Metal types *)
  let typ_of_prec = function
    | Ops.Byte_prec _ -> "uchar"
    | Half_prec _ -> "half"
    | Single_prec _ -> "float"
    | Double_prec _ -> "float" (* Metal doesn't fully support double precision *)
    | Void_prec -> "void"

  (* Metal binary operators *)
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
    | Relu_gate, _ -> ("(", " > 0.0f ?", " : 0.0f)")
    | Satur01_gate, Byte_prec _ -> ("(abs(", ") > 0 ? 0 : (", ")")
    | Satur01_gate, _ -> ("(abs(trunc(", ")) > 0.0f ? 0.0f : (", "))")
    | Max, _ -> ("max(", ", ", ")")
    | Min, _ -> ("min(", ", ", ")")
    | Mod, Byte_prec _ -> ("(", " % ", ")")
    | Mod, _ -> ("fmod(", ", ", ")")
    | Cmplt, _ -> ("(", " < ", ")")
    | Cmpne, _ -> ("(", " != ", ")")
    | Cmpeq, _ -> ("(", " == ", ")")
    | Or, _ -> ("(", " || ", ")")
    | And, _ -> ("(", " && ", ")")

  (* Metal unary operators *)
  let unop_syntax prec v =
    match (v, prec) with
    | Ops.Identity, _ -> ("", "")
    | Relu, _ -> ("max(0.0f, ", ")")
    | Satur01, _ -> ("max(0.0f, min(1.0f, ", "))")
    | Exp, _ -> ("exp(", ")")
    | Log, _ -> ("log(", ")")
    | Exp2, _ -> ("exp2(", ")")
    | Log2, _ -> ("log2(", ")")
    | Sin, _ -> ("sin(", ")")
    | Cos, _ -> ("cos(", ")")
    | Sqrt, _ -> ("sqrt(", ")")
    | Recip, _ -> ("(1.0f / (", "))")
    | Recip_sqrt, _ -> ("(1.0f / sqrt(", "))")
    | Neg, _ -> ("(-(", "))")
    | Tanh_approx, _ -> ("tanh(", ")")
    | Not, _ -> ("(", " == 0.0f ? 1.0f : 0.0f)")

  (* Metal ternary operators *)
  let ternop_syntax prec v =
    match (v, prec) with
    | Ops.Where, _ -> ("(", " ? ", " : ", ")")
    | FMA, _ -> ("fma(", ", ", ", ", ")")

  (* Precision conversion *)
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
  (* Use C_syntax to generate code, then convert to Metal Shading Language *)
  let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
    let procs = [| lowered |]
  end)) in
  let idx_params = Indexing.bound_symbols bindings in
  let b = Buffer.create 4096 in
  let ppf = Stdlib.Format.formatter_of_buffer b in

  (* Add debug printing if needed *)
  if Utils.debug_log_from_routines () then
    Stdlib.Format.fprintf ppf "@,device int printf(constant char* format, ...);@,";

  Syntax.print_includes ppf;
  let params = Syntax.compile_proc ~name ppf idx_params lowered in
  let msl = metal_from_c ~name @@ Buffer.contents b in
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
  let msl = metal_from_c ~name @@ Buffer.contents b in
  let traced_stores = Array.map lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store)) in
  { traced_stores; msl; params_and_names; bindings }

let get_global_run_id =
  let next_id = ref 0 in
  fun () ->
    Int.incr next_id;
    if !next_id < 0 then next_id := 0;
    !next_id

(* Compile Metal shader code and link procedure *)
let%debug2_sexp link_proc ~prior_context ~name ~(params : (string * param_source) list) ~ctx_arrays
    lowered_bindings (library, compile_options) =
  (* Get function from library and create pipeline state *)
  let function_obj = Mt.Library.new_function_with_name library name in
  let pipeline_state =
    Mt.ComputePipelineState.on_device prior_context.stream.device.dev.primary_context function_obj
  in
  let stream = prior_context.stream in
  let runner_label = get_name stream in

  let work () : unit =
    let log_id = get_global_run_id () in
    let log_id_prefix = Int.to_string log_id ^ ": " in
    [%log_result
      "Launching", name, "on", runner_label, (log_id : int), (params : (string * param_source) list)];

    (* Create command buffer and encoder *)
    let command_buffer = Mt.CommandQueue.command_buffer stream.runner in
    let encoder = Mt.CommandBuffer.compute_command_encoder command_buffer in

    (* Set compute pipeline state *)
    Mt.ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline_state;

    (* Set kernel parameters *)
    List.iteri params ~f:(fun idx (param_name, param_source) ->
        match param_source with
        | Param_ptr tn ->
            let buffer = Option.value_exn ~here:[%here] @@ Map.find ctx_arrays tn in
            Mt.ComputeCommandEncoder.set_buffer encoder buffer 0 idx
        | Log_file_name ->
            (* Metal doesn't support dynamic string parameters easily - we'd need a different
               approach *)
            ()
        | Merge_buffer ->
            let buf = Option.value_exn ~here:[%here] !(stream.merge_buffer) in
            Mt.ComputeCommandEncoder.set_buffer encoder buf.ptr 0 idx
        | Static_idx s ->
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
                         "metal: static index %{Indexing.symbol_ident s.static_symbol} is too big: \
                          %{upto#Int}"]);

            (* Metal doesn't have a direct way to set integers - we need to use a buffer *)
            let int_buffer =
              Mt.Buffer.on_device stream.device.dev.primary_context ~length:4
                Mt.ResourceOptions.storage_mode_shared
            in
            let int_ptr = Mt.Buffer.contents int_buffer in
            Ctypes.(( !@ ) (int_ptr |> from_voidp int) <-@ !i);
            Mt.Buffer.did_modify_range int_buffer (Mt.Buffer.NSRange.make ~location:0 ~length:4);
            Mt.ComputeCommandEncoder.set_buffer encoder int_buffer 0 idx);

    (* Dispatch compute kernel *)
    [%log "launching the kernel"];
    if Utils.debug_log_from_routines () then (
      Utils.add_log_processor ~prefix:log_id_prefix @@ fun _output ->
      ([%log_block
         runner_label;
         Utils.log_trace_tree _output]);

      (* Determine threadgroup size and grid size *)
      let threadgroup_size = Mt.ComputeCommandEncoder.Size.make ~width:1 ~height:1 ~depth:1 in
      let grid_size = Mt.ComputeCommandEncoder.Size.make ~width:1 ~height:1 ~depth:1 in

      (* Dispatch the compute kernel *)
      Mt.ComputeCommandEncoder.dispatch_threadgroups encoder ~threadgroups_per_grid:grid_size
        ~threads_per_threadgroup:threadgroup_size;

      (* End encoding and commit command buffer *)
      Mt.ComputeCommandEncoder.end_encoding encoder;
      Mt.CommandBuffer.commit command_buffer;

      [%log "kernel launched"])
  in

  Task.Task
    {
      context_lifetime = ((library, compile_options), ctx_arrays);
      description = "launches " ^ name ^ " on " ^ runner_label;
      work;
    }

(* Metal compilation options *)
let compile_options () =
  let options = Mt.CompileOptions.init () in
  if Utils.with_runtime_debug () then
    Mt.CompileOptions.set_optimization_level options Mt.CompileOptions.OptimizationLevel.default
  else
    Mt.CompileOptions.set_optimization_level options Mt.CompileOptions.OptimizationLevel.performance;
  Mt.CompileOptions.set_fast_math_enabled options true;
  options

(* Link a single code object *)
let%track3_sexp link prior_context (code : code) ctx_arrays =
  let device = prior_context.stream.device.dev.primary_context in
  let compile_opts = compile_options () in

  (* Compile Metal code *)
  let library = Mt.Library.on_device device ~source:code.msl compile_opts in

  (* Setup bindings *)
  let idx_params = Indexing.bound_symbols code.bindings in
  let lowered_bindings : Indexing.lowered_bindings = List.map idx_params ~f:(fun s -> (s, ref 0)) in

  (* Link procedure *)
  let task =
    link_proc ~prior_context ~name:code.name ~params:code.params ~ctx_arrays lowered_bindings
      (library, compile_opts)
  in
  (lowered_bindings, task)

(* Link a batch of code objects *)
let%track3_sexp link_batch prior_context (code_batch : code_batch) ctx_arrays =
  let idx_params = Indexing.bound_symbols code_batch.bindings in
  let lowered_bindings : Indexing.lowered_bindings = List.map idx_params ~f:(fun s -> (s, ref 0)) in

  let device = prior_context.stream.device.dev.primary_context in
  let compile_opts = compile_options () in

  (* Compile Metal code *)
  let library = Mt.Library.on_device device ~source:code_batch.msl compile_opts in

  (* Link procedures *)
  let procs =
    Array.mapi code_batch.params_and_names ~f:(fun i pns ->
        Option.value ~default:None
        @@ Option.map2 pns ctx_arrays.(i) ~f:(fun (params, name) ctx_arrays ->
               let task =
                 link_proc ~prior_context ~name ~params ~ctx_arrays lowered_bindings
                   (library, compile_opts)
               in
               Some task))
  in
  (lowered_bindings, procs)

(* Debug info *)
let get_global_debug_info () =
  Sexp.message "metal_global_debug"
    [ ("initialized_devices", [%sexp_of: int] @@ Hash_set.length initialized_devices) ]

let get_debug_info (_stream : stream) =
  (* Metal doesn't provide detailed stream status info like CUDA *)
  Sexp.message "metal_stream_debug" []

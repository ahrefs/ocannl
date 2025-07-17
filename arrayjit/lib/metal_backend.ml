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
end) : Ir.Backend_impl.Lowered_backend = struct
  (* Include the device setup with types and allocation *)
  include Backend_impl.Device (Device_stream) (Alloc_buffer)

  (* Global state for Metal devices *)
  let metal_devices : Me.Device.t array = Me.Device.copy_all_devices ()
  let () = assert (Array.length metal_devices > 0)

  (* Store for captured logs per stream_id *)
  let stream_logs : (int, string list ref) Hashtbl.t = Hashtbl.create (module Int)

  (* Metal has unified memory on Apple Silicon, so we can use host memory *)
  let get_buffer_for_ptr device ~size_in_bytes bytes =
    Me.Buffer.on_device_with_bytes_no_copy device ~bytes ~length:size_in_bytes
      Me.ResourceOptions.(storage_mode_shared + cpu_cache_mode_default_cache)

  let use_host_memory = Some (get_buffer_for_ptr metal_devices.(0))

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
        (* Store the log_entries_ref for later retrieval, associated with the stream_id which will
           be assigned by make_stream shortly. We'll add it after make_stream. *)
        (created_q, Some log_entries_ref))
      else (Me.CommandQueue.on_device metal_device, None)
    in
    let actual_queue, opt_log_entries_ref = queue in
    let shared_event_obj = Me.SharedEvent.on_device metal_device in
    let counter = Unsigned.ULLong.one in
    (* Next value = 1 *)
    let runner = { queue = actual_queue; event = shared_event_obj; counter } in
    let stream_obj = make_stream device_wrapper runner in
    (* Finalize linking log_entries_ref to stream_id and set up GC finalizer *)
    Option.iter opt_log_entries_ref ~f:(fun log_ref ->
        Hashtbl.add_exn stream_logs ~key:stream_obj.stream_id ~data:log_ref);
    Stdlib.Gc.finalise (fun s -> Hashtbl.remove stream_logs s.stream_id) stream_obj;
    stream_obj

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
    Me.CommandBuffer.wait_until_completed command_buffer;
    (* Process captured logs if any *)
    if Utils.debug_log_from_routines () then
      match Hashtbl.find stream_logs stream.stream_id with
      | Some log_entries_ref ->
          let logs_to_process = List.rev !log_entries_ref in
          if not (List.is_empty logs_to_process) then
            Utils.log_debug_routine_logs ~log_contents:logs_to_process
              ~stream_name:(get_name stream);
          log_entries_ref := [] (* Clear processed logs *)
      | None -> () (* No log bucket for this stream, logging likely not enabled for it *)

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

  let get_debug_info stream =
    let num_pending_logs =
      match Hashtbl.find stream_logs stream.stream_id with None -> 0 | Some r -> List.length !r
    in
    Sexp.message "Metal stream debug info"
      [
        ("stream_id", sexp_of_int stream.stream_id);
        ("pending_shader_logs", sexp_of_int num_pending_logs);
      ]

  (* --- Copy Operations --- *)
  let from_host ~dst_ptr ~dst hosted =
    (* Copy from host memory to Metal buffer *)
    let size_in_bytes = Ndarray.size_in_bytes hosted in
    let command_buffer = Me.CommandBuffer.on_queue dst.stream.runner.queue in

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
    let command_buffer = Me.CommandBuffer.on_queue src.stream.runner.queue in

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
    let arg_int_prefix = "const int& "

    let extra_args =
      [
        "uint3 gid [[threadgroup_position_in_grid]]"; "uint3 lid [[thread_position_in_threadgroup]]";
      ]

    let includes =
      [ "<metal_stdlib>"; "<metal_math>"; "<metal_logging>"; "<metal_compute>"; "<metal_atomic>" ]

    let metal_log_object_name = "os_log_default"
    let extra_declarations = [ "using namespace metal;" ]

    let typ_of_prec = function
      | Ops.Byte_prec _ -> "uchar"
      | Ops.Uint16_prec _ -> "ushort"
      | Ops.Int32_prec _ -> "int"
      | Ops.Uint4x32_prec _ -> "uint4" (* Metal's uint4 type - 128-bit *)
      | Ops.Half_prec _ -> "half"
      | Ops.Bfloat16_prec _ -> "bfloat" (* Metal supports bfloat16 natively *)
      | Ops.Fp8_prec _ -> invalid_arg "Metal backend does not support FP8 precision"
      | Ops.Single_prec _ -> "float"
      | Ops.Double_prec _ -> "double"
      | Ops.Void_prec -> "void"

    let metal_prec_suffix_float = function
      | Ops.Byte_prec _ -> ""
      | Ops.Uint16_prec _ -> ""
      | Ops.Int32_prec _ -> ""
      | Ops.Uint4x32_prec _ -> "" (* No specific suffix for uint4 *)
      | Ops.Half_prec _ -> "h"
      | Ops.Bfloat16_prec _ -> "bf" (* TODO: Verify actual Metal suffix for bfloat16 *)
      | Ops.Fp8_prec _ -> invalid_arg "Metal backend does not support FP8 precision"
      | Ops.Single_prec _ -> "f"
      | Ops.Double_prec _ -> ""
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
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0"))
                 ^^ space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                 ^^ string "0.0"))
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
      | Threefry4x32, _ -> func "arrayjit_threefry4x32"
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
      | Relu, Ops.Double_prec _ -> fun v -> func_doc "max" (separate comma_sep [ string "0.0"; v ])
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
      | Uint4x32_to_prec_uniform target_prec, _ ->
          let conv_func = match target_prec with
            | Ops.Single_prec _ -> "uint4x32_to_fp32_uniform"
            | Double_prec _ -> "uint4x32_to_fp64_uniform" (* Metal doesn't support double, but function exists *)
            | Half_prec _ -> "uint4x32_to_fp16_uniform"
            | Bfloat16_prec _ -> "uint4x32_to_bf16_uniform"
            | Byte_prec _ -> "uint4x32_to_u8_uniform"
            | Uint16_prec _ -> "uint4x32_to_u32_uniform" (* Should probably be u16 *)
            | Int32_prec _ -> "uint4x32_to_i32_uniform"
            | _ -> "/* unsupported conversion from uint4x32 */ 0"
          in
          func_doc conv_func
    (* Logical not *)

    let convert_precision ~from ~to_ =
      if Ops.equal_prec from to_ then ("", "") else ("(" ^ typ_of_prec to_ ^ ")(", ")")

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
        string metal_log_object_name ^^ string ".log_debug(" ^^ base_doc ^^ comma ^^ space
        ^^ separate (comma ^^ space) args_docs
        ^^ rparen ^^ semi
  end

  let%diagn_sexp compile_metal_source ~name ~source ~device =
    let options = Me.CompileOptions.init () in
    if Utils.debug_log_from_routines () then (
      Me.CompileOptions.set_language_version options Me.CompileOptions.LanguageVersion.version_3_2;
      Me.CompileOptions.set_enable_logging options true)
    else
      Me.CompileOptions.set_language_version options Me.CompileOptions.LanguageVersion.version_3_0
      (* Logging is disabled by default in CompileOptions, so no need to explicitly set it to
         false *);

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
    (* Read and prepend the Metal builtins file *)
    let builtins_path = Stdlib.Filename.concat (Stdlib.Filename.dirname __FILE__) "arrayjit_builtins.msl" in
    (try
       let builtins_content = Stdio.In_channel.read_all builtins_path in
       Buffer.add_string b builtins_content;
       Buffer.add_string b "\n\n"
     with _ -> ()); (* Silently skip if file not found *)
    let declarations_doc = Syntax.print_declarations () in
    (* Add Metal address space qualifiers *)
    let params, proc_doc = Syntax.compile_proc ~name idx_params lowered in
    let final_doc = PPrint.(declarations_doc ^^ proc_doc) in
    PPrint.ToBuffer.pretty 1.0 110 b final_doc;
    let source = Buffer.contents b in
    {
      metal_source = source;
      compiled_code = Array.create ~len:num_devs None;
      (* One slot per device *)
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
    (* Read and prepend the Metal builtins file *)
    let builtins_path = Stdlib.Filename.concat (Stdlib.Filename.dirname __FILE__) "arrayjit_builtins.msl" in
    (try
       let builtins_content = Stdio.In_channel.read_all builtins_path in
       Buffer.add_string b builtins_content;
       Buffer.add_string b "\n\n"
     with _ -> ()); (* Silently skip if file not found *)
    let declarations_doc = Syntax.print_declarations () in
    let funcs_and_docs =
      Array.map2_exn names lowereds
        ~f:
          (Option.map2 ~f:(fun name lowered ->
               let params, doc = Syntax.compile_proc ~name idx_params lowered in
               ((name, params), doc)))
    in
    let all_proc_docs = List.filter_map (Array.to_list funcs_and_docs) ~f:(Option.map ~f:snd) in
    let final_doc = PPrint.(declarations_doc ^^ separate hardline all_proc_docs) in
    PPrint.ToBuffer.pretty 1.0 110 b final_doc;
    let source = Buffer.contents b in
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

  let%diagn2_sexp link_proc ~prior_context ~library ~func_name ~params ~lowered_bindings ~ctx_arrays
      =
    let stream = prior_context.stream in
    let device = stream.device.dev in
    let queue = stream.runner.queue in
    let runner_label = get_name stream in
    let func = Me.Library.new_function_with_name library func_name in
    let pso, _ = Me.ComputePipelineState.on_device_with_function device func in

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
        List.iteri params ~f:(fun index (_p_name, p_source) ->
            match p_source with
            | Param_ptr tn when Map.mem ctx_arrays tn ->
                let buffer = Map.find_exn ctx_arrays tn in
                Me.ComputeCommandEncoder.set_buffer encoder ~index buffer
            | Param_ptr tn when Tn.known_constant tn && Tn.is_hosted_force tn 48 ->
                let buffer =
                  Hashtbl.find_or_add stream.device.cross_stream_candidates tn ~default:(fun () ->
                      get_buffer_for_ptr device ~size_in_bytes:(Lazy.force tn.size_in_bytes)
                      @@ Ndarray.get_voidptr_not_managed
                      @@ Option.value_exn ~here:[%here]
                      @@ Lazy.force tn.array)
                in
                Me.ComputeCommandEncoder.set_buffer encoder ~index buffer
            | Param_ptr tn ->
                failwith
                  [%string "Param_ptr %{Tn.debug_name tn} not found in ctx_arrays for %{func_name}"]
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

  let link prior_context code ctx_arrays =
    let device = prior_context.stream.device.dev in
    let library = compile_metal_source ~name:code.func_name ~source:code.metal_source ~device in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols code.bindings) ~f:(fun s -> (s, ref 0))
    in
    let task =
      link_proc ~prior_context ~library ~func_name:code.func_name ~params:code.params
        ~lowered_bindings ~ctx_arrays
    in
    (lowered_bindings, task)

  let link_batch prior_context code_batch ctx_arrays_opts =
    let device = prior_context.stream.device.dev in
    let library = compile_metal_source ~name:"batch" ~source:code_batch.metal_source ~device in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols code_batch.bindings) ~f:(fun s -> (s, ref 0))
    in

    let tasks =
      Array.mapi code_batch.funcs ~f:(fun i func_opt ->
          Option.bind func_opt ~f:(fun (func_name, params) ->
              Option.map ctx_arrays_opts.(i) ~f:(fun ctx_arrays ->
                  link_proc ~prior_context ~library ~func_name ~params ~lowered_bindings ~ctx_arrays)))
    in
    (lowered_bindings, tasks)
end

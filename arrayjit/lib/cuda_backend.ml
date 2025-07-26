open Base
open Ir
module Tn = Tnode
module Lazy = Utils.Lazy
module Cu = Cuda
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

let () =
  Cu.cuda_call_hook :=
    Some
      (fun ~message:_message ~status:_status ->
        [%debug_sexp
          [%log5_block
            _message;
            if not @@ Cu.is_success _status then [%log (_status : Cu.result)]]])

let _suspended () =
  Cu.cuda_call_hook := Some (fun ~message ~status:_ -> Stdlib.Printf.printf "CUDA %s\n" message)

module Backend_buffer = struct
  type buffer_ptr = Cu.Deviceptr.t

  let sexp_of_buffer_ptr ptr = Sexp.Atom (Cu.Deviceptr.string_of ptr)

  include Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]
  end)
end

module Device_config = struct
  include Backend_buffer

  type dev = {
    dev : Cu.Device.t;
    primary_context : Cu.Context.t;
    set_builtins_in : Cu.Module.t -> unit;
  }
  [@@deriving sexp_of]

  type runner = Cu.Stream.t [@@deriving sexp_of]
  type event = Cu.Delimited_event.t [@@deriving sexp_of]

  let name = "cuda"
end

module Device_stream = Backend_impl.Device_types_ll (Device_config)
open Device_config

let set_ctx ctx = Cu.Context.set_current ctx

module Alloc_buffer = struct
  include Device_stream

  (* It's not actually used, but it's required by the [Backend] interface. *)
  let alloc_buffer ?old_buffer ~size_in_bytes stream =
    match old_buffer with
    | Some ({ size_in_bytes = old_size; _ } as buffer) when size_in_bytes <= old_size -> buffer
    | Some { ptr; _ } ->
        set_ctx stream.device.dev.primary_context;
        Cu.Deviceptr.mem_free ptr;
        { ptr = Cu.Deviceptr.mem_alloc ~size_in_bytes; size_in_bytes }
    | None ->
        set_ctx stream.device.dev.primary_context;
        { ptr = Cu.Deviceptr.mem_alloc ~size_in_bytes; size_in_bytes }

  let alloc_zero_init_array prec ~dims stream =
    let size_in_bytes =
      (if Array.length dims = 0 then 0 else Array.reduce_exn dims ~f:( * )) * Ops.prec_in_bytes prec
    in
    set_ctx stream.device.dev.primary_context;
    let ptr = Cu.Deviceptr.mem_alloc ~size_in_bytes in
    (* TODO: consider using memset_d8 to zero-initialize the memory. *)
    (* if size_in_bytes > 0 then
      Cu.Stream.memset_d8 ptr Unsigned.UChar.zero ~length:size_in_bytes stream.runner; *)
    ptr

  let free_buffer = Some (fun _stream ptr -> Cu.Deviceptr.mem_free ptr)
end

(* [initialized_devices] never forgets its entries. *)
let initialized_devices = Hash_set.create (module Int)
let initialized = ref false

module Fresh (Config : sig
  val config : Ir.Backend_intf.config
end) : Ir.Backend_impl.Lowered_backend = struct
  include Backend_impl.Device (Device_stream) (Alloc_buffer)

  let use_host_memory = None
  let ctx_of (context : context) = context.stream.device.dev.primary_context
  let is_done event = Cu.Delimited_event.query event
  let will_wait_for context event = Cu.Delimited_event.wait context.stream.runner event
  let sync event = Cu.Delimited_event.synchronize event
  let all_work stream = Cu.Delimited_event.record stream.runner

  let () =
    if not !initialized then (
      Cu.init ();
      initialized := true)

  let num_devices = Cu.Device.get_count

  (* [devices] is mutable to support plugging in new devices. *)
  let devices = ref @@ Array.create ~len:(num_devices ()) None

  let get_used_memory (device : device) =
    set_ctx device.dev.primary_context;
    let free, total = Cu.Device.get_free_and_total_mem () in
    total - free

  let opt_alloc_merge_buffer ~size_in_bytes dev stream : unit =
    if
      Option.value_map ~default:true !(stream.merge_buffer) ~f:(fun buffer ->
          buffer.size_in_bytes < size_in_bytes)
    then (
      set_ctx dev.primary_context;
      Option.iter !(stream.merge_buffer) ~f:(fun buffer -> Cu.Deviceptr.mem_free buffer.ptr);
      stream.merge_buffer := Some { ptr = Cu.Deviceptr.mem_alloc ~size_in_bytes; size_in_bytes })

  let%track4_sexp finalize_device (device : device) =
    Cu.Context.set_current device.dev.primary_context;
    Cu.Context.synchronize ();
    (* Note: this is not necessary as releasing the primary context by GC will reset the context. *)
    Hashtbl.iter device.cross_stream_candidates ~f:(fun buffer_ptr ->
        Cu.Deviceptr.mem_free buffer_ptr)

  let%diagn2_sexp cuda_to_ptx ~name cu_src =
    let name_cu = name ^ ".cu" in
    if Utils.settings.output_debug_files_in_build_directory then (
      let build_file = Utils.open_build_file ~base_name:name ~extension:".cu" in
      Stdio.Out_channel.output_string build_file.oc cu_src;
      build_file.finalize ());
    [%log "compiling to PTX"];
    let with_debug =
      Utils.settings.output_debug_files_in_build_directory || Utils.settings.log_level > 0
    in
    let options =
      "--use_fast_math" :: (if Utils.with_runtime_debug () then [ "--device-debug" ] else [])
    in
    (* FIXME: every now and then the compilation crashes because the options are garbled. *)
    (* Stdio.printf "PTX options %s\n%!" @@ String.concat ~sep:", " options; *)
    let ptx = Nvrtc.compile_to_ptx ~cu_src ~name:name_cu ~options ~with_debug in
    if Utils.settings.output_debug_files_in_build_directory then (
      let oc = Out_channel.open_text @@ Utils.build_file @@ name ^ ".ptx" in
      Stdio.Out_channel.output_string oc @@ Nvrtc.string_from_ptx ptx;
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc;
      let oc = Out_channel.open_text @@ Utils.build_file @@ name ^ ".cu_log" in
      Stdio.Out_channel.output_string oc
      @@ Option.value_exn ~here:[%here] (Nvrtc.compilation_log ptx);
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc);
    ptx

  let run_options () =
    if Utils.with_runtime_debug () then
      Cu.Module.[ GENERATE_DEBUG_INFO true; GENERATE_LINE_INFO true ]
    else []

  let set_ptr_in_kernel kernel_module src name =
    let dst, _ = Cuda.Module.get_global kernel_module ~name in
    (* Copy the helper function address to the kernel's function pointer variable *)
    Cuda.Deviceptr.memcpy_D_to_D ~dst ~src ~size_in_bytes:8 (* pointer size *) ()

  let set_builtins_for_device =
    assert !initialized;
    let builtins_path =
      Stdlib.Filename.concat (Stdlib.Filename.dirname Stdlib.__FILE__) "builtins_large.cu"
    in
    let cu_src = Stdio.In_channel.read_all builtins_path in
    let code = cuda_to_ptx ~name:"builtins_large" cu_src in
    fun ~primary_context ->
      set_ctx primary_context;
      let run_module = Cu.Module.load_data_ex code (run_options ()) in
      let threefry4x32_ptr, _ = Cu.Module.get_global run_module ~name:"arrayjit_threefry4x32" in
      fun kernel_module -> set_ptr_in_kernel kernel_module threefry4x32_ptr "arrayjit_threefry4x32"

  let%track3_sexp get_device ~(ordinal : int) : device =
    if num_devices () <= ordinal then
      invalid_arg [%string "Exec_as_cuda.get_device %{ordinal#Int}: not enough devices"];
    (if Array.length !devices <= ordinal then
       let old, len = (!devices, Array.length !devices) in
       devices := Array.init (ordinal + 1) ~f:(fun i -> if i < len then old.(i) else None));
    let default () =
      let dev = Cu.Device.get ~ordinal in
      let primary_context : Cu.Context.t = Cu.Context.get_primary dev in
      let set_builtins_in = set_builtins_for_device ~primary_context in
      let dev = { dev; primary_context; set_builtins_in } in
      set_ctx primary_context;
      if Utils.debug_log_from_routines () && not (Hash_set.mem initialized_devices ordinal) then
        Int.of_string_opt @@ Utils.get_global_arg ~arg_name:"cuda_printf_fifo_size" ~default:""
        |> Option.iter ~f:Cu.Context.(set_limit PRINTF_FIFO_SIZE);
      Hash_set.add initialized_devices ordinal;
      let result = make_device dev ~ordinal in
      Stdlib.Gc.finalise finalize_device result;
      !devices.(ordinal) <- Some result;
      result
    in
    Option.value_or_thunk !devices.(ordinal) ~default

  let%track3_sexp new_stream (device : device) : stream =
    (* Strange that we need ctx_set_current even with a single device! *)
    set_ctx device.dev.primary_context;
    let cu_stream = Cu.Stream.create ~non_blocking:true () in
    make_stream device cu_stream

  let cuda_properties =
    let cache =
      let%debug2_sexp f (ordinal : int) =
        let dev = get_device ~ordinal in
        lazy (Cu.Device.get_attributes dev.dev.dev)
      in
      lazy (Array.init (num_devices ()) ~f)
    in
    let%debug2_sexp get_props (device : device) : Cu.Device.attributes =
      let cache = Lazy.force cache in
      Lazy.force cache.(device.ordinal)
    in
    get_props

  let suggested_num_streams device =
    match Config.config with
    | Only_devices_parallel -> 1
    | For_parallel_copying -> 1 + (cuda_properties device).async_engine_count
    | Most_parallel_streams -> (cuda_properties device).multiprocessor_count

  let await stream : unit =
    set_ctx stream.device.dev.primary_context;
    Cu.Stream.synchronize stream.runner

  let is_idle stream = Cu.Stream.is_ready stream.runner

  let from_host ~dst_ptr ~dst hosted =
    set_ctx @@ ctx_of dst;
    let f src = Cu.Stream.memcpy_H_to_D ~dst:dst_ptr ~src dst.stream.runner in
    Ndarray.apply { f } hosted

  let to_host ~src_ptr ~src hosted =
    set_ctx @@ ctx_of src;
    let f dst = Cu.Stream.memcpy_D_to_H ~dst ~src:src_ptr src.stream.runner in
    Ndarray.apply { f } hosted

  let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src =
    let dev = dst.stream.device in
    let same_device = dev.ordinal = src.stream.device.ordinal in
    let size_in_bytes = Lazy.force tn.Tn.size_in_bytes in
    let memcpy ~dst_ptr =
      (* FIXME: coming in cudajit.0.6.2. *)
      (* if same_device && Cu.Deviceptr.equal dst_ptr src_ptr then () else *)
      if same_device then
        Cu.Stream.memcpy_D_to_D ~size_in_bytes ~dst:dst_ptr ~src:src_ptr dst.stream.runner
      else
        Cu.Stream.memcpy_peer ~size_in_bytes ~dst:dst_ptr ~dst_ctx:(ctx_of dst) ~src:src_ptr
          ~src_ctx:(ctx_of src) dst.stream.runner
    in
    match (into_merge_buffer, dst_ptr) with
    | No, None -> invalid_arg "Cuda_backend.device_to_device: missing dst_ptr"
    | No, Some dst_ptr ->
        set_ctx @@ ctx_of dst;
        memcpy ~dst_ptr
    | Streaming_for _, _ ->
        assert same_device;
        dst.stream.merge_buffer := Some { ptr = src_ptr; size_in_bytes }
    | Copy, _ ->
        set_ctx @@ ctx_of dst;
        opt_alloc_merge_buffer ~size_in_bytes dev.dev dst.stream;
        let buffer = Option.value_exn ~here:[%here] !(dst.stream.merge_buffer) in
        memcpy ~dst_ptr:buffer.ptr

  type code = {
    traced_store : Low_level.traced_store;
    ptx : Nvrtc.compile_to_ptx_result;
    params : (string * param_source) list;
    bindings : Indexing.unit_bindings;
    name : string;
  }
  [@@deriving sexp_of]

  type code_batch = {
    traced_stores : Low_level.traced_store option array;
    ptx : Nvrtc.compile_to_ptx_result;
    bindings : Indexing.unit_bindings;
    params_and_names : ((string * param_source) list * string) option array;
  }
  [@@deriving sexp_of]

  module Cuda_syntax_config (Input : sig
    val procs : Low_level.optimized array
  end) =
  struct
    include C_syntax.Pure_C_config (struct
      type nonrec buffer_ptr = buffer_ptr

      let use_host_memory = None
      let procs = Input.procs

      let full_printf_support =
        not @@ Utils.get_global_flag ~default:false ~arg_name:"prefer_backend_uniformity"
    end)

    let main_kernel_prefix = "extern \"C\" __global__"

    let kernel_prep_line =
      "/* FIXME: single-threaded for now. */if (threadIdx.x != 0 || blockIdx.x != 0) { return; }"

    let includes = [ "<cuda_fp16.h>" ]

    let typ_of_prec = function
      | Ops.Byte_prec _ -> "unsigned char"
      | Ops.Uint16_prec _ -> "unsigned short"
      | Ops.Int32_prec _ -> "int"
      | Ops.Uint4x32_prec _ -> "uint4x32_t"
      | Ops.Half_prec _ -> "__half"
      | Ops.Bfloat16_prec _ -> "__nv_bfloat16" (* CUDA bfloat16 type *)
      | Ops.Fp8_prec _ -> "__nv_fp8_e5m2" (* CUDA FP8 type (E5M2 format) *)
      | Ops.Single_prec _ -> "float"
      | Ops.Double_prec _ -> "double"
      | Ops.Void_prec -> "void"

    let vec_typ_of_prec ~length prec =
      ignore length;
      (* FIXME: NOT IMPLEMENTED YET *)
      failwith "NOT IMPLEMENTED YET"

    let binop_syntax prec v =
      (* TODO: consider using binop_syntax inherited from Pure_C_config and overriding only where
         different. *)
      let open PPrint in
      let f op_str v1 v2 =
        group
          (parens (v1 ^^ string (" " ^ op_str) ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))))
      in
      let func fn v1 v2 =
        group (string fn ^^ parens (v1 ^^ comma ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))))
      in
      match (v, prec) with
      | Ops.Arg1, _ -> invalid_arg "Cuda_backend.binop_syntax: Arg1 is not an operator"
      | Arg2, _ -> invalid_arg "Cuda_backend.binop_syntax: Arg2 is not an operator"
      | _, Ops.Void_prec -> invalid_arg "Cuda_backend.binop_syntax: Void precision"
      | Add, Half_prec _ -> func "__hadd"
      | Sub, Half_prec _ -> func "__hsub"
      | Mul, Half_prec _ -> func "__hmul"
      | Div, Half_prec _ -> func "__hdiv"
      | Add, _ -> f "+"
      | Sub, _ -> f "-"
      | Mul, _ -> f "*"
      | Div, _ -> f "/"
      | ToPowOf, Double_prec _ -> func "pow"
      | ToPowOf, Single_prec _ -> func "powf"
      | ToPowOf, Half_prec _ ->
          fun v1 v2 ->
            group
              (string "hexp2(hlog2(" ^^ v1 ^^ string "),"
              ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
              ^^ string ")")
      | ToPowOf, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _ | Uint4x32_prec _) ->
          invalid_arg "Cuda_backend.binop_syntax: ToPowOf not supported for integer precisions"
      | ToPowOf, Bfloat16_prec _ ->
          fun v1 v2 ->
            group
              (string "__float2bfloat16(powf(__bfloat162float("
              ^^ v1 ^^ string "), __bfloat162float(" ^^ v2 ^^ string ")))")
      | Relu_gate, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0"))))
      | Relu_gate, Bfloat16_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (string "__bfloat162float(" ^^ v1 ^^ string ") > 0.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "__float2bfloat16(0.0f)")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "__float2bfloat16(0.0f)"))))
      | Relu_gate, Half_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0h"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0.0h")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0.0h"))))
      | Relu_gate, Single_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0.0f")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0.0f"))))
      | Relu_gate, Double_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0.0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0.0"))))
      | Relu_gate, Uint4x32_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0"))))
      | Satur01_gate, Byte_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                      ^^ string " < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "(unsigned char)0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "(unsigned char)0"))))
      | Satur01_gate, Half_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "__hgt(" ^^ v1 ^^ comma
                       ^^ string " __ushort_as_half((unsigned short)0x0000U)) && __hlt("
                       ^^ v1 ^^ comma
                       ^^ string " __ushort_as_half((unsigned short)0x3C00U))"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                      ^^ string "__ushort_as_half((unsigned short)0x0000U)")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                         ^^ string "__ushort_as_half((unsigned short)0x0000U)"))))
      | Satur01_gate, Single_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0f && " ^^ v1 ^^ string " < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0.0f")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0.0f"))))
      | Satur01_gate, Double_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0 && " ^^ v1 ^^ string " < 1.0"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0.0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0.0"))))
      | Satur01_gate, Uint16_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                      ^^ string " < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "(unsigned short)0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "(unsigned short)0"))))
      | Satur01_gate, Int32_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                      ^^ string " < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0"))))
      | Satur01_gate, Uint4x32_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                      ^^ string " < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0u")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0u"))))
      | Satur01_gate, Bfloat16_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "__bfloat162float(" ^^ v1
                       ^^ string ") > 0.0f && __bfloat162float("
                       ^^ v1 ^^ string ") < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "__float2bfloat16(0.0f)")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "__float2bfloat16(0.0f)"))))
      | Satur01_gate, Fp8_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                      ^^ string " < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "(unsigned char)0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "(unsigned char)0"))))
      | Max, Byte_prec _ -> func "max"
      | Max, Half_prec _ -> func "__hmax"
      | Max, Double_prec _ -> func "fmax"
      | Max, Single_prec _ -> func "fmaxf"
      | Max, Uint16_prec _ -> func "max"
      | Max, Int32_prec _ -> func "max"
      | Max, Uint4x32_prec _ -> func "max"
      | Max, Bfloat16_prec _ ->
          (* FIXME: This might be wrong, definitely verify and maybe fix, here and elsewhere *)
          func "__hmax"
      | Max, Fp8_prec _ -> func "max"
      | Min, Byte_prec _ -> func "min"
      | Min, Half_prec _ -> func "__hmin"
      | Min, Double_prec _ -> func "fmin"
      | Min, Single_prec _ -> func "fminf"
      | Min, Uint16_prec _ -> func "min"
      | Min, Int32_prec _ -> func "min"
      | Min, Uint4x32_prec _ -> func "min"
      | Min, Bfloat16_prec _ -> func "__hmin"
      | Min, Fp8_prec _ -> func "min"
      | Mod, Byte_prec _ -> f "%"
      | Mod, _ -> func "fmod"
      | Cmplt, _ -> f "<"
      | Cmpne, _ -> f "!="
      | Cmpeq, _ -> f "=="
      | Or, _ -> f "||"
      | And, _ -> f "&&"
      | Threefry4x32, _ -> func "arrayjit_threefry4x32"

    let unop_syntax prec v =
      let open PPrint in
      let f prefix suffix expr = group (string prefix ^^ expr ^^ string suffix) in
      let func fn expr = group (string fn ^^ parens expr) in
      match (v, prec) with
      | Ops.Identity, _ -> f "" ""
      | Relu, Ops.Single_prec _ -> f "fmaxf(0.0, " ")"
      | Relu, Ops.Half_prec _ -> f "__hmax_nan(__ushort_as_half((unsigned short)0x0000U), " ")"
      | Relu, Ops.Byte_prec _ -> f "fmax(0, " ")"
      | Relu, _ -> f "fmax(0.0, " ")"
      | Satur01, Byte_prec _ -> f "fmax(0, fmin(1, " "))"
      | Satur01, Half_prec _ ->
          f
            "__hmax_nan(__ushort_as_half((unsigned short)0x0000U), \
             __hmin_nan(__ushort_as_half((unsigned short)0x3C00U), "
            "))"
      | Satur01, Single_prec _ -> f "fmaxf(0.0f, fminf(1.0f, " "))"
      | Satur01, _ -> f "fmax(0.0, fmin(1.0, " "))"
      | Exp, Half_prec _ -> func "hexp"
      | Exp, Double_prec _ -> func "exp"
      | Exp, _ -> func "expf"
      | Log, Half_prec _ -> func "hlog"
      | Log, Double_prec _ -> func "log"
      | Log, _ -> func "logf"
      | Exp2, Half_prec _ -> func "hexp2"
      | Exp2, Double_prec _ -> func "exp2"
      | Exp2, _ -> func "exp2f"
      | Log2, Half_prec _ -> func "hlog2"
      | Log2, Double_prec _ -> func "log2"
      | Log2, _ -> func "log2f"
      | Sin, Half_prec _ -> func "hsin"
      | Sin, Double_prec _ -> func "sin"
      | Sin, _ -> func "sinf"
      | Cos, Half_prec _ -> func "hcos"
      | Cos, Double_prec _ -> func "cos"
      | Cos, _ -> func "cosf"
      | Sqrt, Half_prec _ -> func "hsqrt"
      | Sqrt, Double_prec _ -> func "sqrt"
      | Sqrt, _ -> func "sqrtf"
      | Recip, Byte_prec _ ->
          invalid_arg "Cuda_backend.unop_syntax: Recip not supported for byte/integer precisions"
      | Recip, Half_prec _ -> func "hrcp"
      | Recip, Single_prec _ -> f "(1.0f / (" "))"
      | Recip, Double_prec _ -> f "(1.0 / (" "))"
      | Recip, _ -> f "(1 / (" "))"
      | Recip_sqrt, Byte_prec _ ->
          invalid_arg
            "Cuda_backend.unop_syntax: Recip_sqrt not supported for byte/integer precisions"
      | Recip_sqrt, Half_prec _ -> func "hrsqrt"
      | Recip_sqrt, Double_prec _ -> f "(1.0 / sqrt(" "))"
      | Recip_sqrt, Single_prec _ -> f "(1.0f / sqrtf(" "))"
      | Recip_sqrt, _ -> f "(1 / sqrtf(" "))"
      | Neg, _ -> f "(-(" "))"
      | Tanh_approx, Byte_prec _ ->
          invalid_arg
            "Cuda_backend.unop_syntax: Tanh_approx not supported for byte/integer precisions"
      | Tanh_approx, Half_prec _ -> func "htanh_approx"
      | Tanh_approx, Single_prec _ -> func "__tanhf"
      | Tanh_approx, _ -> func "tanh"
      | Not, _ -> f "(" " == 0.0 ? 1.0 : 0.0)"
      | Uint4x32_to_prec_uniform, _ -> func ("uint4x32_to_" ^ Ops.prec_string prec ^ "_uniform")

    let ternop_syntax prec v =
      let open PPrint in
      let func fn v1 v2 v3 = group (string fn ^^ parens (separate comma [ v1; v2; v3 ])) in
      match (v, prec) with
      | Ops.Where, _ -> fun v1 v2 v3 -> group (parens v1 ^^ string " ? " ^^ v2 ^^ string " : " ^^ v3)
      | FMA, Ops.Half_prec _ -> func "__hfma"
      | FMA, Ops.Single_prec _ -> func "fmaf"
      | FMA, _ -> func "fma"

    let convert_precision ~from ~to_ =
      match (from, to_) with
      | Ops.Double_prec _, Ops.Double_prec _
      | Single_prec _, Single_prec _
      | Half_prec _, Half_prec _
      | Byte_prec _, Byte_prec _
      | Void_prec, Void_prec ->
          ("", "")
      | Double_prec _, Half_prec _ -> ("__double2half(", ")")
      | Single_prec _, Half_prec _ -> ("__float2half(", ")")
      | Byte_prec _, Half_prec _ -> ("__ushort2half_rn((unsigned short int)", ")")
      | _ -> ("(" ^ typ_of_prec to_ ^ ")(", ")")

    let kernel_log_param = Some ("int", "log_id")
    let log_involves_file_management = false

    let pp_log_statement ~log_param_c_expr_doc ~base_message_literal ~args_docs =
      let open PPrint in
      let format_string_literal =
        let res = String.substr_replace_all base_message_literal ~pattern:"\n" ~with_:"$" in
        let res =
          if for_log_trace_tree && String.is_suffix res ~suffix:"$" then
            String.drop_suffix res 1 ^ "\\n"
          else res
        in
        !Utils.captured_log_prefix ^ "%d: " ^ res
      in
      let all_args =
        match log_param_c_expr_doc with
        | Some doc -> doc :: args_docs
        | None -> args_docs (* Should not happen if kernel_log_param is Some *)
      in
      string "printf("
      ^^ dquotes (string format_string_literal)
      ^^ comma ^^ space
      ^^ separate (comma ^^ space) all_args
      ^^ rparen ^^ semi
  end

  let builtins_large_header =
    {|
  __device__ uint4x32_t ( *arrayjit_threefry4x32)(uint4x32_t key, uint4x32_t counter) = nullptr;
  |}

  let prepend_builtins b =
    if Utils.debug_log_from_routines () then
      Buffer.add_string b "__device__ int printf (const char * format, ... );\n";
    Buffer.add_string b "\n\n";
    let builtins_path =
      Stdlib.Filename.concat (Stdlib.Filename.dirname Stdlib.__FILE__) "builtins_small.cu"
    in
    let builtins_content = Stdio.In_channel.read_all builtins_path in
    Buffer.add_string b builtins_content;
    (* Needs to be after the small builtins, because uses uint4x32_t. *)
    Buffer.add_string b builtins_large_header;
    Buffer.add_string b "\n\n"

  let%diagn2_sexp compile ~name bindings ({ Low_level.traced_store; _ } as lowered) =
    (* TODO: The following link seems to claim it's better to expand into loops than use memset.
       https://stackoverflow.com/questions/23712558/how-do-i-best-initialize-a-local-memory-array-to-0 *)
    let module Syntax = C_syntax.C_syntax (Cuda_syntax_config (struct
      let procs = [| lowered |]
    end)) in
    let idx_params = Indexing.bound_symbols bindings in
    let b = Buffer.create 4096 in
    prepend_builtins b;
    let declarations_doc = Syntax.print_declarations () in
    let params, proc_doc = Syntax.compile_proc ~name idx_params lowered in
    let final_doc = PPrint.(declarations_doc ^^ proc_doc) in
    PPrint.ToBuffer.pretty 1.0 110 b final_doc;
    let ptx = cuda_to_ptx ~name (Buffer.contents b) in
    { traced_store; ptx; params; bindings; name }

  let%diagn2_sexp compile_batch ~names bindings lowereds =
    let module Syntax = C_syntax.C_syntax (Cuda_syntax_config (struct
      let procs = Array.filter_opt lowereds
    end)) in
    let idx_params = Indexing.bound_symbols bindings in
    let b = Buffer.create 4096 in
    prepend_builtins b;
    let declarations_doc = Syntax.print_declarations () in
    let params_and_docs =
      Array.map2_exn names lowereds
        ~f:
          (Option.map2 ~f:(fun name lowered ->
               let params, doc = Syntax.compile_proc ~name idx_params lowered in
               ((params, name), doc)))
    in
    let all_proc_docs = List.filter_map (Array.to_list params_and_docs) ~f:(Option.map ~f:snd) in
    let final_doc = PPrint.(declarations_doc ^^ separate hardline all_proc_docs) in
    PPrint.ToBuffer.pretty 1.0 110 b final_doc;

    let name : string =
      String.(
        strip ~drop:(equal_char '_')
        @@ common_prefix (Array.to_list names |> List.concat_map ~f:Option.to_list))
    in
    let ptx = cuda_to_ptx ~name (Buffer.contents b) in
    let traced_stores = Array.map lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store)) in
    let params_and_names = Array.map params_and_docs ~f:(Option.map ~f:fst) in
    { traced_stores; ptx; params_and_names; bindings }

  let get_global_run_id =
    let next_id = ref 0 in
    fun () ->
      Int.incr next_id;
      if !next_id < 0 then next_id := 0;
      !next_id

  let link_proc ~prior_context ~name ~(params : (string * param_source) list) ~ctx_arrays
      lowered_bindings run_module =
    let func = Cu.Module.get_function run_module ~name in
    let stream = prior_context.stream in
    let stream_name = get_name stream in
    let%diagn3_sexp work () : unit =
      let log_id = get_global_run_id () in
      let log_id_prefix = Int.to_string log_id ^ ": " in
      [%log_result
        "Launching",
        name,
        "on",
        stream_name,
        (log_id : int),
        (params : (string * param_source) list)];
      let module S = Cu.Stream in
      let args : S.kernel_param list =
        (* TODO: should we prohibit or warn about local-only tensors that are in
           prior_context.ctx_arrays? *)
        List.map params ~f:(function
          | _name, Param_ptr tn ->
              let arr = Option.value_exn ~here:[%here] @@ Map.find ctx_arrays tn in
              S.Tensor arr
          | _name, Log_file_name -> S.Int log_id
          | _name, Merge_buffer ->
              let buf = Option.value_exn ~here:[%here] !(stream.merge_buffer) in
              S.Tensor buf.ptr
          | _name, Static_idx s ->
              let i = Indexing.find_exn lowered_bindings s in
              if !i < 0 then
                raise
                @@ Utils.User_error
                     [%string
                       "cuda: static index %{Indexing.symbol_ident s.static_symbol} is negative: \
                        %{!i#Int}"];
              Option.iter s.static_range ~f:(fun upto ->
                  if !i >= upto then
                    raise
                    @@ Utils.User_error
                         [%string
                           "cuda: static index %{Indexing.symbol_ident s.static_symbol} is too \
                            big: %{upto#Int}"]);
              S.Int !i)
      in
      set_ctx @@ ctx_of prior_context;
      (* FIXME: this happens inside the kernel. *)
      (* Map.iteri ctx_arrays ~f:(fun ~key ~data:ptr -> if key.Low_level.zero_initialized then
       Cu.Stream.memset_d8 ptr Unsigned.UChar.zero ~length:(Tn.size_in_bytes key.Low_level.tn)); *)
      [%log "launching the kernel"];
      (* Stdio.printf "launching %s\n" name; *)
      (if Utils.debug_log_from_routines () then
         Utils.add_log_processor ~prefix:log_id_prefix @@ fun log_contents ->
         Utils.log_debug_routine_logs ~log_contents ~stream_name);
      S.launch_kernel func ~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0 stream.runner args;
      [%log "kernel launched"]
    in
    Task.Task
      {
        context_lifetime = (run_module, ctx_arrays);
        description = "launches " ^ name ^ " on " ^ stream_name;
        work;
      }

  let%track3_sexp link prior_context (code : code) ctx_arrays =
    let ctx = ctx_of prior_context in
    set_ctx ctx;
    let run_module = Cu.Module.load_data_ex code.ptx (run_options ()) in
    prior_context.stream.device.dev.set_builtins_in run_module;
    let idx_params = Indexing.bound_symbols code.bindings in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map idx_params ~f:(fun s -> (s, ref 0))
    in
    let task =
      link_proc ~prior_context ~name:code.name ~params:code.params ~ctx_arrays lowered_bindings
        run_module
    in
    (lowered_bindings, task)

  let%track3_sexp link_batch prior_context (code_batch : code_batch) ctx_arrays =
    let idx_params = Indexing.bound_symbols code_batch.bindings in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map idx_params ~f:(fun s -> (s, ref 0))
    in
    let ctx = ctx_of prior_context in
    set_ctx ctx;
    let run_module = Cu.Module.load_data_ex code_batch.ptx (run_options ()) in
    prior_context.stream.device.dev.set_builtins_in run_module;
    let procs =
      Array.mapi code_batch.params_and_names ~f:(fun i pns ->
          Option.value ~default:None
          @@ Option.map2 pns ctx_arrays.(i) ~f:(fun (params, name) ctx_arrays ->
                 let task =
                   link_proc ~prior_context ~name ~params ~ctx_arrays lowered_bindings run_module
                 in
                 Some task))
    in
    (lowered_bindings, procs)

  let get_global_debug_info () =
    Sexp.message "cuda_global_debug"
      [ ("live_streams", [%sexp_of: int] @@ Cu.Stream.get_total_live_streams ()) ]

  let static_properties =
    let device_properties =
      Array.init (num_devices ()) ~f:(fun ordinal ->
          let dev = Cu.Device.get ~ordinal in
          let attributes = Cu.Device.get_attributes dev in
          let props =
            [
              ("device_name", Sexp.Atom attributes.name);
              ("device_ordinal", [%sexp_of: int] ordinal);
              ("multiprocessor_count", [%sexp_of: int] attributes.multiprocessor_count);
              ("clock_rate", [%sexp_of: int] attributes.clock_rate);
              ("async_engine_count", [%sexp_of: int] attributes.async_engine_count);
              ("compute_capability_major", [%sexp_of: int] attributes.compute_capability_major);
              ("compute_capability_minor", [%sexp_of: int] attributes.compute_capability_minor);
              ("max_threads_per_block", [%sexp_of: int] attributes.max_threads_per_block);
              ("unified_addressing", [%sexp_of: bool] attributes.unified_addressing);
            ]
          in
          Sexp.message "device" props)
    in
    Sexp.List (Sexp.Atom "cuda_devices" :: Array.to_list device_properties)

  let get_debug_info (stream : stream) =
    let tot, unr, unf = Cu.Stream.total_unreleased_unfinished_delimited_events stream.runner in
    let i2s = [%sexp_of: int] in
    Sexp.message "cuda_stream_debug"
      [ ("total_events", i2s tot); ("unreleased_events", i2s unr); ("unfinished_events", i2s unf) ]
end

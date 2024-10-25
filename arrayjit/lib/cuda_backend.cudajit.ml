open Base
module Tn = Tnode
module Lazy = Utils.Lazy
module Cu = Cudajit
module Debug_runtime = Utils.Debug_runtime
open Backend_types

let _get_local_debug_runtime = Utils._get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

let () =
  Cu.cuda_call_hook :=
    Some
      (fun ~message ~status ->
        [%debug_l_sexp
          [%log5_block
            message;
            if not @@ Cu.is_success status then [%log (status : Cu.result)]]])

module Backend_buffer = struct
  type buffer_ptr = Cu.Deviceptr.t

  let sexp_of_buffer_ptr ptr = Sexp.Atom (Cu.Deviceptr.string_of ptr)
  let c_ptr_to_string = None

  include Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]
  end)
end

module Device_config = struct
  include Backend_buffer

  type device = {
    dev : Cu.Device.t;
    ordinal : int;
    primary_context : Cu.Context.t;
    mutable copy_merge_buffer : buffer_ptr;
    mutable copy_merge_buffer_capacity : int;
    used_names : Hash_set.M(String).t;  (** Unique names of streams. *)
    released : Utils.atomic_bool;
    cross_stream_candidates : buffer_ptr Hashtbl.M(Tn).t;
        (** Freshly created arrays that might be shared across streams. The map can both grow and
            shrink. See the explanation on top of this file. *)
    owner_streams : string Hashtbl.M(Tn).t;
        (** The streams owning the given nodes. This map can only grow. *)
  }
  [@@deriving sexp_of]

  type stream_state = unit [@@deriving sexp_of]
  type runner = Cu.Stream.t [@@deriving sexp_of]
  type event = Cu.Delimited_event.t [@@deriving sexp_of]
end

module Device_stream = Device_types (Device_config)

let set_ctx ctx = Cu.Context.set_current ctx

module Alloc_buffer = struct
  include Device_stream

  (* It's not actually used, but it's required by the [Backend] interface. *)
  let alloc_buffer ?old_buffer ~size_in_bytes stream =
    match old_buffer with
    | Some ({ size_in_bytes = old_size; _ } as buffer) when size_in_bytes <= old_size -> buffer
    | Some { ptr; _ } ->
        set_ctx stream.device.Device_config.primary_context;
        Cu.Deviceptr.mem_free ptr;
        { ptr = Cu.Deviceptr.mem_alloc ~size_in_bytes; size_in_bytes }
    | None ->
        set_ctx stream.device.primary_context;
        { ptr = Cu.Deviceptr.mem_alloc ~size_in_bytes; size_in_bytes }

  let alloc_zero_init_array prec ~dims stream =
    let size_in_bytes =
      (if Array.length dims = 0 then 0 else Array.reduce_exn dims ~f:( * )) * Ops.prec_in_bytes prec
    in
    set_ctx stream.device.Device_config.primary_context;
    Cu.Deviceptr.mem_alloc ~size_in_bytes
end

include Device (Device_stream) (Alloc_buffer)

type context = {
  label : string;
  ctx : Cu.Context.t;  (** Currently, this is always the same as [stream.device.primary_context]. *)
  stream : stream;
  parent : context option;
  run_module : (Cu.Module.t[@sexp.opaque]) option;
      (** Code jitted for this context, typically independent of the parent and child contexts, but
          shared by batch linked contexts. *)
  ctx_arrays : ctx_arrays;
      (** This map contains arrays used in this context or an ancestor context (they might be unique
          but might also be cross-stream shared. *)
  finalized : Utils.atomic_bool;
}
[@@deriving sexp_of]

let ctx_arrays ctx = ctx.ctx_arrays
let global_config = ref For_parallel_copying
let is_done event = Cu.Delimited_event.query event
let will_wait_for context event = Cu.Delimited_event.wait context.stream.runner event
let sync event = Cu.Delimited_event.synchronize event
let all_work stream = Cu.Delimited_event.record stream.runner
let scheduled_merge_node stream = Option.map ~f:snd !(stream.merge_buffer)

let is_initialized, initialize =
  let initialized = ref false in
  let init (config : config) : unit =
    if not !initialized then Cu.init ();
    initialized := true;
    global_config := config
  in
  ((fun () -> !initialized), init)

let num_devices = Cu.Device.get_count
let devices = ref @@ Stdlib.Weak.create 0

(* Unlike [devices] above, [initialized_devices] never forgets its entries. *)
let initialized_devices = Hash_set.create (module Int)

let get_used_memory dev =
  set_ctx dev.Device_config.primary_context;
  let free, total = Cudajit.Device.get_free_and_total_mem () in
  total - free

let opt_alloc_merge_buffer ~size_in_bytes dev =
  if dev.Device_config.copy_merge_buffer_capacity < size_in_bytes then (
    set_ctx dev.primary_context;
    Cu.Deviceptr.mem_free dev.copy_merge_buffer;
    dev.copy_merge_buffer <- Cu.Deviceptr.mem_alloc ~size_in_bytes;
    dev.copy_merge_buffer_capacity <- size_in_bytes)

let%track3_sexp cleanup_device (device : device) =
  Cu.Context.set_current device.primary_context;
  Cu.Context.synchronize ();
  Option.iter !Utils.advance_captured_logs ~f:(fun callback -> callback ());
  (* Note: this is not necessary as releasing the primary context by GC will reset the context. *)
  Cu.Deviceptr.mem_free device.copy_merge_buffer;
  Hashtbl.iter device.cross_stream_candidates ~f:(fun buffer_ptr ->
      Cu.Deviceptr.mem_free buffer_ptr)

let%track5_sexp finalize_device device =
  if Atomic.compare_and_set device.Device_config.released false true then cleanup_device device

let%track3_sexp get_device ~(ordinal : int) : device =
  if num_devices () <= ordinal then
    invalid_arg [%string "Exec_as_cuda.get_device %{ordinal#Int}: not enough devices"];
  if Stdlib.Weak.length !devices <= ordinal then (
    let old = !devices in
    devices := Stdlib.Weak.create (ordinal + 1);
    Stdlib.Weak.blit old 0 !devices 0 (Stdlib.Weak.length old));
  let default () =
    let dev : Cu.Device.t = Cu.Device.get ~ordinal in
    let primary_context : Cu.Context.t = Cu.Context.get_primary dev in
    let copy_merge_buffer_capacity = 8 in
    set_ctx primary_context;
    if Utils.debug_log_from_routines () && not (Hash_set.mem initialized_devices ordinal) then
      Option.iter Utils.settings.cuda_printf_fifo_size ~f:Cu.Context.(set_limit PRINTF_FIFO_SIZE);
    Hash_set.add initialized_devices ordinal;
    let copy_merge_buffer = Cu.Deviceptr.mem_alloc ~size_in_bytes:copy_merge_buffer_capacity in
    let result =
      Device_config.
        {
          dev;
          ordinal;
          used_names = Hash_set.create (module String);
          primary_context;
          copy_merge_buffer;
          copy_merge_buffer_capacity;
          released = Atomic.make false;
          cross_stream_candidates = (Hashtbl.create (module Tn) : buffer_ptr Hashtbl.M(Tn).t);
          owner_streams = Hashtbl.create (module Tn);
        }
    in
    Stdlib.Gc.finalise finalize_device result;
    Stdlib.Weak.set !devices ordinal (Some result);
    result
  in
  let result = Option.value_or_thunk (Stdlib.Weak.get !devices ordinal) ~default in
  (* We need this: there can be an arbitrary gap between the finalizer run and the deallocation. *)
  if Atomic.get result.released then default () else result

let%track3_sexp new_stream (device : device) : stream =
  let rec unique_name suffix =
    let name = "stream " ^ Int.to_string suffix in
    if Hash_set.mem device.used_names name then unique_name (suffix + 1) else name
  in
  let unique_name = unique_name 0 in
  Hash_set.add device.used_names unique_name;
  (* Strange that we need ctx_set_current even with a single device! *)
  set_ctx device.primary_context;
  let cu_stream = Cu.Stream.create ~non_blocking:true () in
  make_stream ~device ~state:() ~unique_name ~runner:cu_stream

let cuda_properties =
  let cache =
    lazy
      (Array.init (num_devices ()) ~f:(fun ordinal ->
           let dev = get_device ~ordinal in
           lazy (Cu.Device.get_attributes dev.dev)))
  in
  fun device ->
    if not @@ is_initialized () then invalid_arg "cuda_properties: CUDA not initialized";
    let cache = Lazy.force cache in
    Lazy.force cache.(device.Device_config.ordinal)

let suggested_num_streams device =
  match !global_config with
  | Only_devices_parallel -> 1
  | For_parallel_copying -> 1 + (cuda_properties device).async_engine_count
  | Most_parallel_streams -> (cuda_properties device).multiprocessor_count

let get_ctx_stream { stream; _ } = stream
let get_stream_device { device; _ } = device
let to_ordinal Device_config.{ ordinal; _ } = ordinal
let get_name stream = stream.unique_name

let await stream : unit =
  set_ctx stream.device.Device_config.primary_context;
  Cu.Stream.synchronize stream.runner;
  Option.iter !Utils.advance_captured_logs ~f:(fun callback -> callback ())

let is_idle stream = Cu.Stream.is_ready stream.runner

let%track3_sexp finalize (ctx : context) : unit =
  if
    Atomic.compare_and_set ctx.finalized false true && (not @@ Atomic.get ctx.stream.device.released)
  then (
    (* await does this: set_ctx ctx.stream.device.primary_context; *)
    await ctx.stream;
    (* Cudajit's contexts, streams and events are destroyed by their respective finalizers. *)
    Option.iter ctx.run_module ~f:Cu.Module.unload;
    Map.iteri ctx.ctx_arrays ~f:(fun ~key ~data ->
        if
          (not (Option.exists ctx.parent ~f:(fun pc -> Map.mem pc.ctx_arrays key)))
          && not (Hashtbl.mem ctx.stream.device.cross_stream_candidates key)
        then Cu.Deviceptr.mem_free data))

let init stream =
  let ctx =
    {
      label = "on dev " ^ get_name stream;
      ctx = stream.device.Device_config.primary_context;
      stream;
      parent = None;
      ctx_arrays = Map.empty (module Tn);
      run_module = None;
      finalized = Atomic.make false;
    }
  in
  Stdlib.Gc.finalise finalize ctx;
  ctx

let from_host ~dst_ptr ~dst hosted =
  set_ctx dst.ctx;
  let f src = Cu.Stream.memcpy_H_to_D ~dst:dst_ptr ~src dst.stream.runner in
  Ndarray.map { f } hosted

let to_host ~src_ptr ~src hosted =
  set_ctx src.ctx;
  let f dst = Cu.Stream.memcpy_D_to_H ~dst ~src:src_ptr src.stream.runner in
  Ndarray.map { f } hosted

let device_to_device tn ~into_merge_buffer ~dst_ptr ~dst ~src_ptr ~src =
  let same_device = dst.stream.device.ordinal = src.stream.device.ordinal in
  let memcpy ~dst_ptr =
    if same_device then
      Cu.Stream.memcpy_D_to_D ~size_in_bytes:(Tn.size_in_bytes tn) ~dst:dst_ptr ~src:src_ptr
        dst.stream.runner
    else
      Cu.Stream.memcpy_peer ~size_in_bytes:(Tn.size_in_bytes tn) ~dst:dst_ptr ~dst_ctx:dst.ctx
        ~src:src_ptr ~src_ctx:src.ctx dst.stream.runner
  in
  match (into_merge_buffer, dst_ptr) with
  | No, None -> invalid_arg "Cuda_backend.device_to_device: missing dst_ptr"
  | No, Some dst_ptr ->
      set_ctx dst.ctx;
      memcpy ~dst_ptr
  | Streaming, _ ->
      assert same_device;
      dst.stream.merge_buffer := Some (src_ptr, tn)
  | Copy, _ ->
      set_ctx dst.ctx;
      let size_in_bytes = Tn.size_in_bytes tn in
      opt_alloc_merge_buffer ~size_in_bytes dst.stream.device;
      memcpy ~dst_ptr:dst.stream.device.copy_merge_buffer;
      dst.stream.merge_buffer := Some (dst.stream.device.copy_merge_buffer, tn)

type code = {
  traced_store : Low_level.traced_store;
  ptx : Cu.Nvrtc.compile_to_ptx_result;
  params : (string * param_source) list;
  bindings : Indexing.unit_bindings;
  name : string;
}
[@@deriving sexp_of]

type code_batch = {
  traced_stores : Low_level.traced_store option array;
  ptx : Cu.Nvrtc.compile_to_ptx_result;
  bindings : Indexing.unit_bindings;
  params_and_names : ((string * param_source) list * string) option array;
}
[@@deriving sexp_of]

let%diagn2_sexp cuda_to_ptx ~name cu_src =
  let name_cu = name ^ ".cu" in
  if Utils.settings.output_debug_files_in_build_directory then (
    let oc = Out_channel.open_text @@ Utils.build_file name_cu in
    Stdio.Out_channel.output_string oc cu_src;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc);
  [%log "compiling to PTX"];
  let module Cu = Cudajit in
  let with_debug =
    Utils.settings.output_debug_files_in_build_directory || Utils.settings.log_level > 0
  in
  let options =
    "--use_fast_math" :: (if Utils.with_runtime_debug () then [ "--device-debug" ] else [])
  in
  let ptx = Cu.Nvrtc.compile_to_ptx ~cu_src ~name:name_cu ~options ~with_debug in
  if Utils.settings.output_debug_files_in_build_directory then (
    let oc = Out_channel.open_text @@ Utils.build_file @@ name ^ ".ptx" in
    Stdio.Out_channel.output_string oc @@ Cu.Nvrtc.string_from_ptx ptx;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc;
    let oc = Out_channel.open_text @@ Utils.build_file @@ name ^ ".cu_log" in
    Stdio.Out_channel.output_string oc
    @@ Option.value_exn ~here:[%here] (Cu.Nvrtc.compilation_log ptx);
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc);
  ptx

let is_in_context node =
  Tnode.default_to_most_local node.Low_level.tn 33;
  match node.tn.memory_mode with Some ((Virtual | Local), _) -> false | _ -> true

module C_syntax_config (Input : sig
  val for_lowereds : Low_level.optimized array
end) =
struct
  let for_lowereds = Input.for_lowereds

  type nonrec buffer_ptr = buffer_ptr [@@deriving sexp_of]

  let opt_ctx_arrays = None
  let hardcoded_context_ptr = None
  let is_in_context = is_in_context
  let host_ptrs_for_readonly = false
  (* GPUs cannot access host memory pointers directly. *)

  let logs_to_stdout = true
  let main_kernel_prefix = "extern \"C\" __global__"

  let kernel_prep_line =
    "/* FIXME: single-threaded for now. */if (threadIdx.x != 0 || blockIdx.x != 0) { return; }"

  let include_lines = [ "#include <cuda_fp16.h>" ]

  let typ_of_prec = function
    | Ops.Byte_prec _ -> "unsigned char"
    | Half_prec _ -> "__half"
    | Single_prec _ -> "float"
    | Double_prec _ -> "double"
    | Void_prec -> "void"

  let binop_syntax prec v =
    match (v, prec) with
    | Ops.Arg1, _ -> invalid_arg "Cuda_backend.binop_syntax: Arg1 is not an operator"
    | Arg2, _ -> invalid_arg "Cuda_backend.binop_syntax: Arg2 is not an operator"
    | _, Ops.Void_prec -> invalid_arg "Cuda_backend.binop_syntax: Void precision"
    | Add, Half_prec _ -> ("__hadd(", ", ", ")")
    | Sub, Half_prec _ -> ("__hsub(", ", ", ")")
    | Mul, Half_prec _ -> ("__hmul(", ", ", ")")
    | Div, Half_prec _ -> ("__hdiv(", ", ", ")")
    | Add, _ -> ("(", " +", ")")
    | Sub, _ -> ("(", " -", ")")
    | Mul, _ -> ("(", " *", ")")
    | Div, _ -> ("(", " /", ")")
    | ToPowOf, Double_prec _ -> ("pow(", ",", ")")
    | ToPowOf, Single_prec _ -> ("powf(", ",", ")")
    | ToPowOf, Half_prec _ -> ("hexp2(hlog2(", "), ", ")")
    | ToPowOf, Byte_prec _ ->
        invalid_arg "Cuda_backend.binop_syntax: ToPowOf not supported for byte/integer precisions"
    | Relu_gate, Byte_prec _ -> ("(", " > 0 ?", " : 0)")
    | Relu_gate, Half_prec _ ->
        ( "(__hgt(",
          ", __ushort_as_half((unsigned short)0x0000U)) ?",
          " : __ushort_as_half((unsigned short)0x0000U))" )
    | Relu_gate, _ -> ("(", " > 0.0 ?", " : 0.0)")

  let unop_syntax prec v =
    match (v, prec) with
    | Ops.Identity, _ -> ("", "")
    | Relu, Ops.Single_prec _ -> ("fmaxf(0.0, ", ")")
    | Relu, Ops.Half_prec _ -> ("__hmax_nan(__ushort_as_half((unsigned short)0x0000U), ", ")")
    | Relu, Ops.Byte_prec _ -> ("fmax(0, ", ")")
    | Relu, _ -> ("fmax(0.0, ", ")")

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
end

let compile ~name bindings ({ Low_level.traced_store; _ } as lowered) =
  (* TODO: The following link seems to claim it's better to expand into loops than use memset.
     https://stackoverflow.com/questions/23712558/how-do-i-best-initialize-a-local-memory-array-to-0 *)
  let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
    let for_lowereds = [| lowered |]
  end)) in
  let idx_params = Indexing.bound_symbols bindings in
  let b = Buffer.create 4096 in
  let ppf = Stdlib.Format.formatter_of_buffer b in
  if Utils.debug_log_from_routines () then
    Stdlib.Format.fprintf ppf "@,__device__ int printf (const char * format, ... );@,";
  let is_global = Syntax.compile_globals ppf in
  let params = Syntax.compile_proc ~name ~is_global ppf idx_params lowered in
  let ptx = cuda_to_ptx ~name @@ Buffer.contents b in
  { traced_store; ptx; params; bindings; name }

let compile_batch ~names bindings lowereds =
  let for_lowereds = Array.filter_map ~f:Fn.id lowereds in
  let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
    let for_lowereds = for_lowereds
  end)) in
  let idx_params = Indexing.bound_symbols bindings in
  let b = Buffer.create 4096 in
  let ppf = Stdlib.Format.formatter_of_buffer b in
  let is_global = Syntax.compile_globals ppf in
  let params_and_names =
    Array.map2_exn names lowereds
      ~f:
        (Option.map2 ~f:(fun name lowered ->
             (Syntax.compile_proc ~name ~is_global ppf idx_params lowered, name)))
  in
  let name : string =
    String.(
      strip ~drop:(equal_char '_')
      @@ common_prefix (Array.to_list names |> List.concat_map ~f:Option.to_list))
  in
  let ptx = cuda_to_ptx ~name @@ Buffer.contents b in
  let traced_stores = Array.map lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store)) in
  { traced_stores; ptx; params_and_names; bindings }

let get_global_run_id =
  let next_id = ref 0 in
  fun () ->
    Int.incr next_id;
    if !next_id < 0 then next_id := 0;
    !next_id

let link_proc ~prior_context ~name ~(params : (string * param_source) list) ~ctx_arrays
    _traced_store lowered_bindings run_module =
  let module Cu = Cudajit in
  let func = Cu.Module.get_function run_module ~name in
  let context =
    { prior_context with parent = Some prior_context; run_module = Some run_module; ctx_arrays }
  in
  Stdlib.Gc.finalise finalize context;
  let%diagn3_l_sexp work () : unit =
    let log_id = get_global_run_id () in
    let log_id_prefix = Int.to_string log_id ^ ": " in
    [%log_result
      "Launching", name, context.label, (log_id : int), (params : (string * param_source) list)];
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
            let ptr = fst @@ Option.value_exn ~here:[%here] !(context.stream.merge_buffer) in
            S.Tensor ptr
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
                         "cuda: static index %{Indexing.symbol_ident s.static_symbol} is too big: \
                          %{upto#Int}"]);
            S.Int !i)
    in
    set_ctx context.ctx;
    (* FIXME: this happens inside the kernel. *)
    (* Map.iteri ctx_arrays ~f:(fun ~key ~data:ptr -> if key.Low_level.zero_initialized then
       Cu.Stream.memset_d8 ptr Unsigned.UChar.zero ~length:(Tn.size_in_bytes key.Low_level.tn)); *)
    [%log "launching the kernel"];
    (if Utils.debug_log_from_routines () then
       Utils.add_log_processor ~prefix:log_id_prefix @@ fun _output ->
       [%log_block
         context.label;
         Utils.log_trace_tree _output]);
    S.launch_kernel func ~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0 context.stream.runner args;
    [%log "kernel launched"]
  in
  ( context,
    Task.Task
      {
        context_lifetime = context;
        description = "launches " ^ name ^ " on " ^ context.label;
        work;
      } )

let%track3_sexp alloc_if_needed ctx stream ~key ~data:node ctx_arrays =
  if is_in_context node && not (Map.mem ctx_arrays key) then (
    [%log2 Tn.debug_name key, "read_only", (node.read_only : bool)];
    [%log3 (key : Tn.t)];
    let default () : buffer_ptr =
      set_ctx ctx;
      Cu.Deviceptr.mem_alloc ~size_in_bytes:(Tn.size_in_bytes key)
    in
    let add_new () = Map.add_exn ctx_arrays ~key ~data:(default ()) in
    let device = stream.device in
    if node.read_only then
      if Tn.known_non_cross_stream key then add_new ()
      else (
        if Hashtbl.mem device.Device_config.cross_stream_candidates key then
          Tn.update_memory_sharing key Tn.Shared_cross_stream 40;
        let data = Hashtbl.find_or_add device.cross_stream_candidates key ~default in
        Map.add_exn ctx_arrays ~key ~data)
    else if Tn.known_shared_cross_stream key then (
      if Hashtbl.mem device.owner_streams key then
        if not @@ String.equal stream.unique_name @@ Hashtbl.find_exn device.owner_streams key then
          raise
          @@ Utils.User_error
               ("Cuda_backend.alloc_if_needed: node " ^ Tn.debug_name key
              ^ " assumed to be cross-stream-shared but then written to on multiple devices")
        else Hashtbl.add_exn device.owner_streams ~key ~data:stream.unique_name;
      let data = Hashtbl.find_exn device.cross_stream_candidates key in
      Map.add_exn ctx_arrays ~key ~data)
    else (
      Tn.update_memory_sharing key Tn.Per_stream 41;
      Hashtbl.remove device.cross_stream_candidates key;
      add_new ()))
  else ctx_arrays

let run_options () =
  if Utils.with_runtime_debug () then
    Cu.Module.[ GENERATE_DEBUG_INFO true; GENERATE_LINE_INFO true ]
  else []

let%track3_sexp link prior_context (code : code) : context * _ * _ =
  let ctx = prior_context.ctx in
  set_ctx ctx;
  let ctx_arrays =
    Hashtbl.fold ~init:prior_context.ctx_arrays code.traced_store
      ~f:(alloc_if_needed ctx prior_context.stream)
  in
  let run_module = Cu.Module.load_data_ex code.ptx (run_options ()) in
  let idx_params = Indexing.bound_symbols code.bindings in
  let lowered_bindings : Indexing.lowered_bindings = List.map idx_params ~f:(fun s -> (s, ref 0)) in
  let context, task =
    link_proc ~prior_context ~name:code.name ~params:code.params ~ctx_arrays code.traced_store
      lowered_bindings run_module
  in
  (context, lowered_bindings, task)

let%track3_sexp link_batch prior_context (code_batch : code_batch) : context * _ * _ =
  let idx_params = Indexing.bound_symbols code_batch.bindings in
  let lowered_bindings : Indexing.lowered_bindings = List.map idx_params ~f:(fun s -> (s, ref 0)) in
  let module Cu = Cudajit in
  let ctx = prior_context.ctx in
  set_ctx ctx;
  let run_module = Cu.Module.load_data_ex code_batch.ptx (run_options ()) in
  let (context, _ctx_arrays), procs =
    Array.fold_mapi code_batch.params_and_names ~init:(prior_context, prior_context.ctx_arrays)
      ~f:(fun i (context, ctx_arrays) pns ->
        Option.value ~default:((context, ctx_arrays), None)
        @@ Option.map2 pns code_batch.traced_stores.(i) ~f:(fun (params, name) traced_store ->
               let ctx_arrays =
                 Hashtbl.fold ~init:ctx_arrays traced_store
                   ~f:(alloc_if_needed ctx prior_context.stream)
               in
               let context, task =
                 link_proc ~prior_context:context ~name ~params ~ctx_arrays traced_store
                   lowered_bindings run_module
               in
               ((context, ctx_arrays), Some task)))
  in
  (context, lowered_bindings, procs)

let name = "cuda"

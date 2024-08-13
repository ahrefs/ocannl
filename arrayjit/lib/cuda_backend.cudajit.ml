open Base
module Tn = Tnode
module Lazy = Utils.Lazy
module Debug_runtime = Utils.Debug_runtime
open Backend_utils.Types

let _get_local_debug_runtime = Utils._get_local_debug_runtime

[%%global_debug_log_level Nothing]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

type buffer_ptr = Cudajit.deviceptr

let sexp_of_buffer_ptr ptr = Sexp.Atom (Cudajit.string_of_deviceptr ptr)

type ctx_array = Cudajit.deviceptr

let sexp_of_ctx_array = sexp_of_buffer_ptr

type physical_device = {
  dev : (Cudajit.device[@sexp.opaque]);
  ordinal : int;
  primary_context : (Cudajit.context[@sexp.opaque]);
  mutable copy_merge_buffer : buffer_ptr;
  mutable copy_merge_buffer_capacity : int;
  released : Utils.atomic_bool;
}
[@@deriving sexp_of]

and device = {
  physical : physical_device;
  stream : (Cudajit.stream[@sexp.opaque]);
  subordinal : int;
  mutable merge_buffer : (buffer_ptr * Tn.t) option;
}

and context = {
  label : string;
  ctx : (Cudajit.context[@sexp.opaque]);
  device : device;
  parent : context option;
  run_module : (Cudajit.module_[@sexp.opaque]) option;
      (** Code jitted for this context, typically independent of the parent and child contexts, but
          shared by batch linked contexts. *)
  global_arrays : ctx_array Map.M(Tn).t;
      (** This map contains only the global arrays, where [all_arrays.(key).global] is [Some name]. *)
  finalized : Utils.atomic_bool;
}
[@@deriving sexp_of]

let ctx_arrays ctx = ctx.global_arrays
let global_config = ref For_parallel_copying

let is_initialized, initialize =
  let initialized = ref false in
  let%track_sexp init (config : config) : unit =
    initialized := true;
    global_config := config;
    Cudajit.init ()
  in
  ((fun () -> !initialized), init)

let num_physical_devices = Cudajit.device_get_count
let devices = ref @@ Core.Weak.create 0

let set_ctx ctx =
  let cur_ctx = Cudajit.ctx_get_current () in
  if not @@ phys_equal ctx cur_ctx then Cudajit.ctx_set_current ctx

(* It's not actually used, but it's required by the [Backend] interface. *)
let alloc_buffer ?old_buffer ~size_in_bytes device =
  match old_buffer with
  | Some (old_ptr, old_size) when size_in_bytes <= old_size -> old_ptr
  | Some (old_ptr, _old_size) ->
      set_ctx device.physical.primary_context;
      Cudajit.mem_free old_ptr;
      Cudajit.mem_alloc ~size_in_bytes
  | None ->
      set_ctx device.physical.primary_context;
      Cudajit.mem_alloc ~size_in_bytes

let opt_alloc_merge_buffer ~size_in_bytes phys_dev =
  if phys_dev.copy_merge_buffer_capacity < size_in_bytes then (
    set_ctx phys_dev.primary_context;
    Cudajit.mem_free phys_dev.copy_merge_buffer;
    phys_dev.copy_merge_buffer <- Cudajit.mem_alloc ~size_in_bytes;
    phys_dev.copy_merge_buffer_capacity <- size_in_bytes)

let%track_sexp get_device ~(ordinal : int) : physical_device =
  if num_physical_devices () <= ordinal then
    invalid_arg [%string "Exec_as_cuda.get_device %{ordinal#Int}: not enough devices"];
  if Core.Weak.length !devices <= ordinal then (
    let old = !devices in
    devices := Core.Weak.create (ordinal + 1);
    Core.Weak.blit old 0 !devices 0 (Core.Weak.length old));
  Option.value_or_thunk (Core.Weak.get !devices ordinal) ~default:(fun () ->
      let dev = Cudajit.device_get ~ordinal in
      let primary_context = Cudajit.device_primary_ctx_retain dev in
      let copy_merge_buffer_capacity = 8 in
      set_ctx primary_context;
      let copy_merge_buffer = Cudajit.mem_alloc ~size_in_bytes:copy_merge_buffer_capacity in
      let result =
        {
          dev;
          ordinal;
          primary_context;
          copy_merge_buffer;
          copy_merge_buffer_capacity;
          released = Atomic.make false;
        }
      in
      Core.Weak.set !devices ordinal (Some result);
      result)

let new_virtual_device physical =
  let subordinal = 0 in
  (* Strange that we need ctx_set_current even with a single device! *)
  set_ctx physical.primary_context;
  let stream = Cudajit.stream_create ~non_blocking:true () in
  { physical; stream; subordinal; merge_buffer = None }

let cuda_properties =
  let cache =
    lazy
      (Array.init (num_physical_devices ()) ~f:(fun ordinal ->
           let dev = get_device ~ordinal in
           lazy (Cudajit.device_get_attributes dev.dev)))
  in
  fun physical ->
    if not @@ is_initialized () then invalid_arg "cuda_properties: CUDA not initialized";
    let cache = Lazy.force cache in
    Lazy.force cache.(physical.ordinal)

let suggested_num_virtual_devices device =
  match !global_config with
  | Physical_devices_only -> 1
  | For_parallel_copying -> 1 + (cuda_properties device).async_engine_count
  | Most_parallel_devices -> (cuda_properties device).multiprocessor_count

let get_ctx_device { device; _ } = device
let get_physical_device { physical; _ } = physical
let to_ordinal { ordinal; _ } = ordinal
let to_subordinal { subordinal; _ } = subordinal

let get_name device =
  Int.to_string (to_ordinal device.physical) ^ "_" ^ Int.to_string (to_subordinal device)

let await device : unit =
  set_ctx device.physical.primary_context;
  Cudajit.stream_synchronize device.stream

let is_idle device = Cudajit.stream_is_ready device.stream

let finalize ctx =
  if
    Atomic.compare_and_set ctx.finalized false true
    && (not @@ Atomic.get ctx.device.physical.released)
  then (
    (* await does this: set_ctx ctx.device.physical.primary_context; *)
    await ctx.device;
    Option.iter ctx.run_module ~f:Cudajit.module_unload;
    Map.iteri ctx.global_arrays ~f:(fun ~key ~data:ptr ->
        if not @@ Option.exists ctx.parent ~f:(fun pc -> Map.mem pc.global_arrays key) then
          Cudajit.mem_free ptr);
    Cudajit.stream_destroy ctx.device.stream)

let init device =
  let ctx =
    {
      label = "on dev " ^ get_name device;
      ctx = device.physical.primary_context;
      device;
      parent = None;
      global_arrays = Map.empty (module Tn);
      run_module = None;
      finalized = Atomic.make false;
    }
  in
  Stdlib.Gc.finalise finalize ctx;
  ctx

let unsafe_cleanup () =
  let len = Core.Weak.length !devices in
  (* NOTE: releasing the context should free its resources, there's no need to finalize the
     remaining contexts, and [finalize] will not do anything for a [released] physical device. *)
  for i = 0 to len - 1 do
    Option.iter (Core.Weak.get !devices i) ~f:(fun device ->
        if Atomic.compare_and_set device.released false true then (
          Cudajit.ctx_set_current device.primary_context;
          Cudajit.ctx_synchronize ();
          Cudajit.device_primary_ctx_release device.dev))
  done;
  Core.Weak.fill !devices 0 len None

let%diagn_sexp from_host ?(rt : (module Minidebug_runtime.Debug_runtime) option) (ctx : context) tn
    =
  match (tn, Map.find ctx.global_arrays tn) with
  | { Tn.array = (lazy (Some hosted)); _ }, Some dst ->
      set_ctx ctx.ctx;
      (if Utils.settings.with_debug_level > 0 then
         let module Debug_runtime =
           (val Option.value_or_thunk rt ~default:(fun () -> (module Debug_runtime)))
         in
         [%log "copying", Tn.debug_name tn, "to", (dst : ctx_array), "from host"]);
      let f src = Cudajit.memcpy_H_to_D_async ~dst ~src ctx.device.stream in
      Ndarray.map { f } hosted;
      true
  | _ -> false

let%track_sexp to_host ?(rt : (module Minidebug_runtime.Debug_runtime) option) (ctx : context)
    (tn : Tn.t) =
  match (tn, Map.find ctx.global_arrays tn) with
  | { Tn.array = (lazy (Some hosted)); _ }, Some src ->
      set_ctx ctx.ctx;
      (if Utils.settings.with_debug_level > 0 then
         let module Debug_runtime =
           (val Option.value_or_thunk rt ~default:(fun () ->
                    (module Debug_runtime : Minidebug_runtime.Debug_runtime)))
         in
         [%log "copying", Tn.debug_name tn, "at", (src : ctx_array), "to host"]);
      let f dst = Cudajit.memcpy_D_to_H_async ~dst ~src ctx.device.stream in
      Ndarray.map { f } hosted;
      true
  | _ -> false

let%track_sexp rec device_to_device ?(rt : (module Minidebug_runtime.Debug_runtime) option)
    (tn : Tn.t) ~into_merge_buffer ~(dst : context) ~(src : context) =
  let memcpy ~d_arr ~s_arr =
    if phys_equal dst.device.physical src.device.physical then
      Cudajit.memcpy_D_to_D_async ~size_in_bytes:(Tn.size_in_bytes tn) ~dst:d_arr ~src:s_arr
        dst.device.stream
    else
      Cudajit.memcpy_peer_async ~size_in_bytes:(Tn.size_in_bytes tn) ~dst:d_arr ~dst_ctx:dst.ctx
        ~src:s_arr ~src_ctx:src.ctx dst.device.stream
  in
  match Map.find src.global_arrays tn with
  | None -> false
  | Some s_arr -> (
      match into_merge_buffer with
      | No -> (
          match Map.find dst.global_arrays tn with
          | None -> false
          | Some d_arr ->
              set_ctx dst.ctx;
              memcpy ~d_arr ~s_arr;
              (if Utils.settings.with_debug_level > 0 then
                 let module Debug_runtime =
                   (val Option.value_or_thunk rt ~default:(fun () ->
                            (module Debug_runtime : Minidebug_runtime.Debug_runtime)))
                 in
                 [%log
                   "copied",
                     Tn.debug_name tn,
                     "from",
                     src.label,
                     "at",
                     (s_arr : ctx_array),
                     "to",
                     (d_arr : ctx_array)]);
              true)
      | Streaming ->
          if phys_equal dst.device.physical src.device.physical then (
            dst.device.merge_buffer <- Some (s_arr, tn);
            (if Utils.settings.with_debug_level > 0 then
               let module Debug_runtime =
                 (val Option.value_or_thunk rt ~default:(fun () ->
                          (module Debug_runtime : Minidebug_runtime.Debug_runtime)))
               in
               [%log "using merge buffer for", Tn.debug_name tn, "from", src.label]);
            true)
          else
            (* TODO: support proper streaming, but it might be difficult. *)
            device_to_device ?rt tn ~into_merge_buffer:Copy ~dst ~src
      | Copy ->
          set_ctx dst.ctx;
          let size_in_bytes = Tn.size_in_bytes tn in
          opt_alloc_merge_buffer ~size_in_bytes dst.device.physical;
          memcpy ~d_arr:dst.device.physical.copy_merge_buffer ~s_arr;
          dst.device.merge_buffer <- Some (dst.device.physical.copy_merge_buffer, tn);
          (if Utils.settings.with_debug_level > 0 then
             let module Debug_runtime =
               (val Option.value_or_thunk rt ~default:(fun () ->
                        (module Debug_runtime : Minidebug_runtime.Debug_runtime)))
             in
             [%log "copied into merge buffer", Tn.debug_name tn, "from", src.label]);
          true)

type code = {
  traced_store : Low_level.traced_store;
  ptx : (Cudajit.compile_to_ptx_result[@sexp.opaque]);
  params : (string * param_source) list;
  bindings : Indexing.unit_bindings;
  name : string;
}
[@@deriving sexp_of]

type code_batch = {
  traced_stores : Low_level.traced_store option array;
  ptx : (Cudajit.compile_to_ptx_result[@sexp.opaque]);
  bindings : Indexing.unit_bindings;
  params_and_names : ((string * param_source) list * string) option array;
}
[@@deriving sexp_of]

let%diagn_sexp cuda_to_ptx ~name cu_src =
  let name_cu = name ^ ".cu" in
  if Utils.settings.output_debug_files_in_build_directory then (
    let oc = Out_channel.open_text @@ Utils.build_file name_cu in
    Stdio.Out_channel.output_string oc cu_src;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc);
  [%log "compiling to PTX"];
  let module Cu = Cudajit in
  let with_debug =
    Utils.settings.output_debug_files_in_build_directory || Utils.settings.with_debug_level > 0
  in
  let options =
    "--use_fast_math" :: (if Utils.with_runtime_debug () then [ "--device-debug" ] else [])
  in
  let ptx = Cu.compile_to_ptx ~cu_src ~name:name_cu ~options ~with_debug in
  if Utils.settings.output_debug_files_in_build_directory then (
    let oc = Out_channel.open_text @@ Utils.build_file @@ name ^ ".ptx" in
    Stdio.Out_channel.output_string oc @@ Cu.string_from_ptx ptx;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc;
    let oc = Out_channel.open_text @@ Utils.build_file @@ name ^ ".cu_log" in
    Stdio.Out_channel.output_string oc @@ Option.value_exn ~here:[%here] (Cu.compilation_log ptx);
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc);
  ptx

let is_in_context node =
  Tnode.default_to_most_local node.Low_level.tn 33;
  match node.tn.memory_mode with Some ((Virtual | Local), _) -> false | _ -> true

let compile ~name bindings ({ Low_level.traced_store; _ } as lowered) =
  (* TODO: The following link seems to claim it's better to expand into loops than use memset.
     https://stackoverflow.com/questions/23712558/how-do-i-best-initialize-a-local-memory-array-to-0 *)
  let module Syntax = Backend_utils.C_syntax (struct
    let for_lowereds = [| lowered |]

    type nonrec ctx_array = buffer_ptr

    let opt_ctx_arrays = None
    let hardcoded_context_ptr = None
    let is_in_context = is_in_context
    let host_ptrs_for_readonly = true
    let logs_to_stdout = true
    let main_kernel_prefix = "extern \"C\" __global__"

    let kernel_prep_line =
      "/* FIXME: single-threaded for now. */if (threadIdx.x != 0 || blockIdx.x != 0) { return; }"
  end) in
  let idx_params = Indexing.bound_symbols bindings in
  let b = Buffer.create 4096 in
  let ppf = Stdlib.Format.formatter_of_buffer b in
  if Utils.settings.debug_log_from_routines then
    Stdlib.Format.fprintf ppf "@,__device__ int printf (const char * format, ... );@,";
  let is_global = Hash_set.create (module Tn) in
  let params = Syntax.compile_proc ~name ~is_global ppf idx_params lowered in
  let ptx = cuda_to_ptx ~name @@ Buffer.contents b in
  { traced_store; ptx; params; bindings; name }

let compile_batch ~names bindings lowereds =
  let for_lowereds = Array.filter_map ~f:Fn.id lowereds in
  let module Syntax = Backend_utils.C_syntax (struct
    let for_lowereds = for_lowereds

    type nonrec ctx_array = buffer_ptr

    let opt_ctx_arrays = None
    let hardcoded_context_ptr = None
    let is_in_context = is_in_context
    let host_ptrs_for_readonly = true
    let logs_to_stdout = true
    let main_kernel_prefix = "extern \"C\" __global__"

    let kernel_prep_line =
      "/* FIXME: single-threaded for now. */if (threadIdx.x != 0 || blockIdx.x != 0) { return; }"
  end) in
  let idx_params = Indexing.bound_symbols bindings in
  let b = Buffer.create 4096 in
  let ppf = Stdlib.Format.formatter_of_buffer b in
  let is_global = Hash_set.create (module Tn) in
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

let link_proc ~prior_context ~name ~(params : (string * param_source) list) ~global_arrays
    lowered_bindings run_module =
  let module Cu = Cudajit in
  let func = Cu.module_get_function run_module ~name in
  let context =
    { prior_context with parent = Some prior_context; run_module = Some run_module; global_arrays }
  in
  Stdlib.Gc.finalise finalize context;
  let%diagn_this_l_sexp work () : unit =
    let log_id = get_global_run_id () in
    let log_id_prefix = Int.to_string log_id ^ ": " in
    if Utils.settings.with_debug_level > 0 then
      [%log_result
        "Launching", name, context.label, (log_id : int), (params : (string * param_source) list)];
    let module Cu = Cudajit in
    let args : Cu.kernel_param list =
      (* TODO: should we prohibit or warn about local-only tensors that are in
         prior_context.global_arrays? *)
      List.map params ~f:(function
        | _name, Param_ptr tn ->
            let ptr = Option.value_exn ~here:[%here] @@ Map.find global_arrays tn in
            Cu.Tensor ptr
        | _name, Log_file_name -> Cu.Int log_id
        | _name, Merge_buffer ->
            let ptr = fst @@ Option.value_exn ~here:[%here] context.device.merge_buffer in
            Cu.Tensor ptr
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
            Cu.Int !i)
    in
    set_ctx context.ctx;
    (* FIXME: this happens inside the kernel. *)
    (* Map.iteri global_arrays ~f:(fun ~key ~data:ptr -> if key.Low_level.zero_initialized then
       Cu.memset_d8_async ptr Unsigned.UChar.zero ~length:(Tn.size_in_bytes key.Low_level.tn)); *)
    [%log "launching the kernel"];
    (if Utils.settings.debug_log_from_routines then
       Utils.add_log_processor ~prefix:log_id_prefix @@ fun output ->
       [%log_entry
         context.label;
         Utils.log_trace_tree output]);
    (* if Utils.settings.debug_log_from_routines then Cu.ctx_set_limit CU_LIMIT_PRINTF_FIFO_SIZE
       4096; *)
    Cu.launch_kernel func ~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0 context.device.stream
      args;
    [%log "kernel launched"]
  in
  ( context,
    Tn.Task
      {
        context_lifetime = context;
        description = "launches " ^ name ^ " on " ^ context.label;
        work;
      } )

let%diagn_sexp alloc_if_needed ctx ~key ~data:node globals =
  if is_in_context node then (
    if Utils.settings.with_debug_level > 0 then
      [%log "mem_alloc", Tn.debug_name node.tn, (not @@ Map.mem globals key : bool)];
    set_ctx ctx;
    let ptr () = Cudajit.mem_alloc ~size_in_bytes:(Tn.size_in_bytes node.tn) in
    Map.update globals key ~f:(fun old -> Option.value_or_thunk old ~default:ptr))
  else globals

let run_options () =
  if Utils.with_runtime_debug () then Cudajit.[ GENERATE_DEBUG_INFO true; GENERATE_LINE_INFO true ]
  else []

let%diagn_sexp link prior_context (code : code) =
  let ctx = prior_context.ctx in
  set_ctx ctx;
  let global_arrays =
    Hashtbl.fold ~init:prior_context.global_arrays code.traced_store ~f:(alloc_if_needed ctx)
  in
  let run_module = Cudajit.module_load_data_ex code.ptx (run_options ()) in
  let idx_params = Indexing.bound_symbols code.bindings in
  let lowered_bindings : Indexing.lowered_bindings = List.map idx_params ~f:(fun s -> (s, ref 0)) in
  let context, task =
    link_proc ~prior_context ~name:code.name ~params:code.params ~global_arrays lowered_bindings
      run_module
  in
  (context, lowered_bindings, task)

let%track_sexp link_batch prior_context (code_batch : code_batch) =
  let idx_params = Indexing.bound_symbols code_batch.bindings in
  let lowered_bindings : Indexing.lowered_bindings = List.map idx_params ~f:(fun s -> (s, ref 0)) in
  let module Cu = Cudajit in
  let ctx = prior_context.ctx in
  set_ctx ctx;
  let run_module = Cu.module_load_data_ex code_batch.ptx (run_options ()) in
  let (context, _global_arrays), procs =
    Array.fold_mapi code_batch.params_and_names ~init:(prior_context, prior_context.global_arrays)
      ~f:(fun i (context, global_arrays) pns ->
        Option.value ~default:((context, global_arrays), None)
        @@ Option.map2 pns code_batch.traced_stores.(i) ~f:(fun (params, name) traced_store ->
               let global_arrays =
                 Hashtbl.fold ~init:global_arrays traced_store ~f:(alloc_if_needed ctx)
               in
               let context, task =
                 link_proc ~prior_context:context ~name ~params ~global_arrays lowered_bindings
                   run_module
               in
               ((context, global_arrays), Some task)))
  in
  (context, lowered_bindings, procs)

let to_buffer ?rt:_ _tn ~dst:_ ~src:_ = failwith "CUDA low-level: NOT IMPLEMENTED YET"
let host_to_buffer ?rt:_ _tn ~dst:_ = failwith "CUDA low-level: NOT IMPLEMENTED YET"
let buffer_to_host ?rt:_ _tn ~src:_ = failwith "CUDA low-level: NOT IMPLEMENTED YET"
let get_buffer _tn _context = failwith "CUDA low-level: NOT IMPLEMENTED YET"

(** In the current design of the CUDA backend, unlike in the CPU backends, context arrays for
    incomparable contexts do not need be disjoint, as long as they share a physical device. If a
    tensor node is read-only for all contexts, its array will be shared even by incomparable
    contexts. The particular design is as follows, within a single physical device:
    - If a tensor node is read-only for a context, and not otherwise recorded, it is stored as a
      cross-device sharing candidate.
    - If a cross-device sharing candidate is read-only for another context, whose parent does not
      have the corresponding array (i.e. it is a different virtual device), it is recorded as
      cross-device shared, and the same array is reused.
    - If a tensor node is writable by a context, and it is not cross-device shared, it is marked as
      non-cross-device, the array is removed from cross-device sharing candidates if present. If it
      is cross-device shared, it is recorded as owned by the corresponding virtual device. It is an
      error if the node was already owned by a different device.

    If a tensor node is cross-device shared, within-physical-device copying is a NOOP as source and
    destination pointers are in that case identical.

    FIXME(#286): this should be controllable via {!Tnode.memory_mode}. *)

open Base
module Tn = Tnode
module Lazy = Utils.Lazy
module Cu = Cudajit
module Debug_runtime = Utils.Debug_runtime
open Backend_utils.Types

let _get_local_debug_runtime = Utils._get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

type buffer_ptr = Cu.Deviceptr.t

let sexp_of_buffer_ptr ptr = Sexp.Atom (Cu.Deviceptr.string_of ptr)

type event = Cu.Delimited_event.t

type ctx_array = { ptr : buffer_ptr; mutable tracking : (event[@sexp.opaque]) option }
[@@deriving sexp_of]

type physical_device = {
  dev : (Cu.Device.t[@sexp.opaque]);
  ordinal : int;
  primary_context : (Cu.Context.t[@sexp.opaque]);
  mutable copy_merge_buffer : buffer_ptr;
  mutable copy_merge_buffer_capacity : int;
  mutable latest_subordinal : int;
  released : Utils.atomic_bool;
  cross_device_candidates : ctx_array Hashtbl.M(Tn).t;
      (** Freshly created arrays that might be shared across virtual devices. The map can both grow
          and shrink. See the explanation on top of this file. *)
  cross_device_shared : Hash_set.M(Tn).t;
      (** Tensor nodes known to be cross-device shared. This set can only grow. *)
  non_cross_device : Hash_set.M(Tn).t;
      (** Tensor nodes known to not be cross-device shared. This set can only grow. *)
  owner_device_subordinal : int Hashtbl.M(Tn).t;
      (** The virtual devices owning the given nodes. This map can only grow. *)
}
[@@deriving sexp_of]

and device = {
  physical : physical_device;
  stream : (Cu.Stream.t[@sexp.opaque]);
  subordinal : int;
  mutable merge_buffer : (buffer_ptr * Tn.t) option;
}

and context = {
  label : string;
  ctx : (Cu.Context.t[@sexp.opaque]);
      (** Currently, this is always the same as [device.physical.primary_context]. *)
  device : device;
  parent : context option;
  run_module : (Cu.Module.t[@sexp.opaque]) option;
      (** Code jitted for this context, typically independent of the parent and child contexts, but
          shared by batch linked contexts. *)
  ctx_arrays : ctx_array Map.M(Tn).t;
      (** This map contains arrays used in this context or an ancestor context (they might be unique
          but might also be cross-device shared. *)
  finalized : Utils.atomic_bool;
}
[@@deriving sexp_of]

let ctx_arrays ctx = ctx.ctx_arrays
let global_config = ref For_parallel_copying

let work_for ctx tn =
  match Map.find ctx.ctx_arrays tn with
  | None -> None
  | Some { tracking = Some event; _ } -> Some event
  | Some ctx_array ->
      ctx_array.tracking <- Some (Cu.Delimited_event.record ctx.device.stream);
      ctx_array.tracking

let is_done event = Cu.Delimited_event.query event
let will_wait_for context event = Cu.Delimited_event.wait context.device.stream event
let sync event = Cu.Delimited_event.synchronize event
let all_work device = Cu.Delimited_event.record device.stream

let is_initialized, initialize =
  let initialized = ref false in
  let init (config : config) : unit =
    initialized := true;
    global_config := config;
    Cu.init ()
  in
  ((fun () -> !initialized), init)

let num_physical_devices = Cu.Device.get_count
let devices = ref @@ Core.Weak.create 0

(* Unlike [devices] above, [initialized_devices] never forgets its entries. *)
let initialized_devices = Hash_set.create (module Int)

let set_ctx ctx =
  let cur_ctx = Cu.Context.get_current () in
  if not @@ phys_equal ctx cur_ctx then Cu.Context.set_current ctx

(* It's not actually used, but it's required by the [Backend] interface. *)
let alloc_buffer ?old_buffer ~size_in_bytes device =
  match old_buffer with
  | Some (old_ptr, old_size) when size_in_bytes <= old_size -> old_ptr
  | Some (old_ptr, _old_size) ->
      set_ctx device.physical.primary_context;
      Cu.Deviceptr.mem_free old_ptr;
      Cu.Deviceptr.mem_alloc ~size_in_bytes
  | None ->
      set_ctx device.physical.primary_context;
      Cu.Deviceptr.mem_alloc ~size_in_bytes

let opt_alloc_merge_buffer ~size_in_bytes phys_dev =
  if phys_dev.copy_merge_buffer_capacity < size_in_bytes then (
    set_ctx phys_dev.primary_context;
    Cu.Deviceptr.mem_free phys_dev.copy_merge_buffer;
    phys_dev.copy_merge_buffer <- Cu.Deviceptr.mem_alloc ~size_in_bytes;
    phys_dev.copy_merge_buffer_capacity <- size_in_bytes)

let get_device ~(ordinal : int) : physical_device =
  if num_physical_devices () <= ordinal then
    invalid_arg [%string "Exec_as_cuda.get_device %{ordinal#Int}: not enough devices"];
  if Core.Weak.length !devices <= ordinal then (
    let old = !devices in
    devices := Core.Weak.create (ordinal + 1);
    Core.Weak.blit old 0 !devices 0 (Core.Weak.length old));
  Option.value_or_thunk (Core.Weak.get !devices ordinal) ~default:(fun () ->
      let dev = Cu.Device.get ~ordinal in
      let primary_context = Cu.Context.get_primary dev in
      let copy_merge_buffer_capacity = 8 in
      set_ctx primary_context;
      if Utils.debug_log_from_routines () && not (Hash_set.mem initialized_devices ordinal) then
        Option.iter Utils.settings.cuda_printf_fifo_size ~f:Cu.Context.(set_limit PRINTF_FIFO_SIZE);
      Hash_set.add initialized_devices ordinal;
      let copy_merge_buffer = Cu.Deviceptr.mem_alloc ~size_in_bytes:copy_merge_buffer_capacity in
      let result =
        {
          dev;
          ordinal;
          latest_subordinal = 0;
          primary_context;
          copy_merge_buffer;
          copy_merge_buffer_capacity;
          released = Atomic.make false;
          cross_device_candidates = (Hashtbl.create (module Tn) : ctx_array Hashtbl.M(Tn).t);
          cross_device_shared = Hash_set.create (module Tn);
          non_cross_device = Hash_set.create (module Tn);
          owner_device_subordinal = Hashtbl.create (module Tn);
        }
      in
      Core.Weak.set !devices ordinal (Some result);
      result)

let new_virtual_device physical =
  let subordinal = physical.latest_subordinal in
  physical.latest_subordinal <- physical.latest_subordinal + 1;
  (* Strange that we need ctx_set_current even with a single device! *)
  set_ctx physical.primary_context;
  let stream = Cu.Stream.create ~non_blocking:true () in
  { physical; stream; subordinal; merge_buffer = None }

let cuda_properties =
  let cache =
    lazy
      (Array.init (num_physical_devices ()) ~f:(fun ordinal ->
           let dev = get_device ~ordinal in
           lazy (Cu.Device.get_attributes dev.dev)))
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
  Cu.Stream.synchronize device.stream;
  Option.iter !Utils.advance_captured_logs ~f:(fun callback -> callback ())

let is_idle device = Cu.Stream.is_ready device.stream

let finalize ctx =
  if
    Atomic.compare_and_set ctx.finalized false true
    && (not @@ Atomic.get ctx.device.physical.released)
  then (
    (* await does this: set_ctx ctx.device.physical.primary_context; *)
    await ctx.device;
    (* Cudajit's contexts, streams and events are destroyed by their respective finalizers. *)
    Option.iter ctx.run_module ~f:Cu.Module.unload;
    Map.iteri ctx.ctx_arrays ~f:(fun ~key ~data ->
        if
          (not (Option.exists ctx.parent ~f:(fun pc -> Map.mem pc.ctx_arrays key)))
          && not (Hashtbl.mem ctx.device.physical.cross_device_candidates key)
        then Cu.Deviceptr.mem_free data.ptr))

let init device =
  let ctx =
    {
      label = "on dev " ^ get_name device;
      ctx = device.physical.primary_context;
      device;
      parent = None;
      ctx_arrays = Map.empty (module Tn);
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
          Cu.Context.set_current device.primary_context;
          Cu.Context.synchronize ();
          Option.iter !Utils.advance_captured_logs ~f:(fun callback -> callback ());
          Hashtbl.iter device.cross_device_candidates ~f:(fun ctx_array ->
              Cu.Deviceptr.mem_free ctx_array.ptr);
          Cu.Device.primary_ctx_release device.dev))
  done;
  Core.Weak.fill !devices 0 len None

let%diagn_l_sexp from_host (ctx : context) tn =
  match (tn, Map.find ctx.ctx_arrays tn) with
  | { Tn.array = (lazy (Some hosted)); _ }, Some dst ->
      set_ctx ctx.ctx;
      [%log "copying", Tn.debug_name tn, "to", (dst : ctx_array), "from host"];
      let f src = Cu.Stream.memcpy_H_to_D ~dst:dst.ptr ~src ctx.device.stream in
      Ndarray.map { f } hosted;
      true
  | _ -> false

let%track_l_sexp to_host (ctx : context) (tn : Tn.t) =
  match (tn, Map.find ctx.ctx_arrays tn) with
  | { Tn.array = (lazy (Some hosted)); _ }, Some src ->
      set_ctx ctx.ctx;
      [%log "copying", Tn.debug_name tn, "at", (src : ctx_array), "to host"];
      let f dst = Cu.Stream.memcpy_D_to_H ~dst ~src:src.ptr ctx.device.stream in
      Ndarray.map { f } hosted;
      true
  | _ -> false

let%track_l_sexp rec device_to_device (tn : Tn.t) ~into_merge_buffer ~(dst : context)
    ~(src : context) =
  let same_physical = phys_equal dst.device.physical src.device.physical in
  let memcpy ~d_arr ~s_arr =
    if same_physical then
      Cu.Stream.memcpy_D_to_D ~size_in_bytes:(Tn.size_in_bytes tn) ~dst:d_arr.ptr ~src:s_arr.ptr
        dst.device.stream
    else
      Cu.Stream.memcpy_peer ~size_in_bytes:(Tn.size_in_bytes tn) ~dst:d_arr.ptr ~dst_ctx:dst.ctx
        ~src:s_arr.ptr ~src_ctx:src.ctx dst.device.stream
  in
  if
    same_physical
    && (src.device.subordinal = dst.device.subordinal
       || Hash_set.mem dst.device.physical.cross_device_shared tn)
  then false
  else
    match Map.find src.ctx_arrays tn with
    | None -> false
    | Some s_arr -> (
        match into_merge_buffer with
        | No -> (
            match Map.find dst.ctx_arrays tn with
            | None -> false
            | Some d_arr ->
                set_ctx dst.ctx;
                memcpy ~d_arr ~s_arr;
                [%log
                  "copied",
                    Tn.debug_name tn,
                    "from",
                    src.label,
                    "at",
                    (s_arr : ctx_array),
                    "to",
                    (d_arr : ctx_array)];
                true)
        | Streaming ->
            if phys_equal dst.device.physical src.device.physical then (
              dst.device.merge_buffer <- Some (s_arr.ptr, tn);
              [%log "using merge buffer for", Tn.debug_name tn, "from", src.label];
              true)
            else
              (* TODO: support proper streaming, but it might be difficult. *)
              device_to_device tn ~into_merge_buffer:Copy ~dst ~src
        | Copy ->
            set_ctx dst.ctx;
            let size_in_bytes = Tn.size_in_bytes tn in
            opt_alloc_merge_buffer ~size_in_bytes dst.device.physical;
            memcpy ~d_arr:{ ptr = dst.device.physical.copy_merge_buffer; tracking = None } ~s_arr;
            dst.device.merge_buffer <- Some (dst.device.physical.copy_merge_buffer, tn);
            [%log "copied into merge buffer", Tn.debug_name tn, "from", src.label];
            true)

type code = {
  traced_store : Low_level.traced_store;
  ptx : (Cu.Nvrtc.compile_to_ptx_result[@sexp.opaque]);
  params : (string * param_source) list;
  bindings : Indexing.unit_bindings;
  name : string;
}
[@@deriving sexp_of]

type code_batch = {
  traced_stores : Low_level.traced_store option array;
  ptx : (Cu.Nvrtc.compile_to_ptx_result[@sexp.opaque]);
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

  type nonrec ctx_array = buffer_ptr

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
  let module Syntax = Backend_utils.C_syntax (C_syntax_config (struct
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
  let module Syntax = Backend_utils.C_syntax (C_syntax_config (struct
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

let link_proc ~prior_context ~name ~(params : (string * param_source) list) ~ctx_arrays traced_store
    lowered_bindings run_module =
  let module Cu = Cudajit in
  let func = Cu.Module.get_function run_module ~name in
  let context =
    { prior_context with parent = Some prior_context; run_module = Some run_module; ctx_arrays }
  in
  Stdlib.Gc.finalise finalize context;
  let%diagn_l_sexp work () : unit =
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
            S.Tensor arr.ptr
        | _name, Log_file_name -> S.Int log_id
        | _name, Merge_buffer ->
            let ptr = fst @@ Option.value_exn ~here:[%here] context.device.merge_buffer in
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
    S.launch_kernel func ~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0 context.device.stream args;
    Map.iteri ctx_arrays ~f:(fun ~key ~data ->
        (* Note: a tensor node can only be a context array if it is materialized. *)
        if Option.is_some data.tracking then
          let traced = Low_level.get_node traced_store key in
          if not traced.read_only then
            data.tracking <- Some (Cu.Delimited_event.record context.device.stream));
    [%log "kernel launched"]
  in
  ( context,
    Tn.Task
      {
        context_lifetime = context;
        description = "launches " ^ name ^ " on " ^ context.label;
        work;
      } )

let%diagn_sexp alloc_if_needed ctx device ~key ~data:node ctx_arrays =
  if is_in_context node && not (Map.mem ctx_arrays key) then (
    [%log Tn.debug_name key, "read_only", (node.read_only : bool)];
    let default () =
      set_ctx ctx;
      let ptr = Cu.Deviceptr.mem_alloc ~size_in_bytes:(Tn.size_in_bytes key) in
      { ptr; tracking = None }
    in
    let add_new () = Map.add_exn ctx_arrays ~key ~data:(default ()) in
    let physical = device.physical in
    if node.read_only then
      if Hash_set.mem physical.non_cross_device key then add_new ()
      else (
        if Hashtbl.mem physical.cross_device_candidates key then
          Hash_set.add physical.cross_device_shared key;
        let data = Hashtbl.find_or_add physical.cross_device_candidates key ~default in
        Map.add_exn ctx_arrays ~key ~data)
    else if Hash_set.mem physical.cross_device_shared key then (
      if Hashtbl.mem physical.owner_device_subordinal key then
        if Hashtbl.find_exn physical.owner_device_subordinal key <> device.subordinal then
          raise
          @@ Utils.User_error
               ("Cuda_backend.alloc_if_needed: node " ^ Tn.debug_name key
              ^ " assumed to be cross-device-shared but then written to on multiple devices")
        else Hashtbl.add_exn physical.owner_device_subordinal ~key ~data:device.subordinal;
      let data = Hashtbl.find_exn physical.cross_device_candidates key in
      Map.add_exn ctx_arrays ~key ~data)
    else (
      Hash_set.add physical.non_cross_device key;
      Hashtbl.remove physical.cross_device_candidates key;
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
      ~f:(alloc_if_needed ctx prior_context.device)
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
                   ~f:(alloc_if_needed ctx prior_context.device)
               in
               let context, task =
                 link_proc ~prior_context:context ~name ~params ~ctx_arrays traced_store
                   lowered_bindings run_module
               in
               ((context, ctx_arrays), Some task)))
  in
  (context, lowered_bindings, procs)

let to_buffer _tn ~dst:_ ~src:_ = failwith "CUDA low-level: NOT IMPLEMENTED YET"
let host_to_buffer _tn ~dst:_ = failwith "CUDA low-level: NOT IMPLEMENTED YET"
let buffer_to_host _tn ~src:_ = failwith "CUDA low-level: NOT IMPLEMENTED YET"
let get_buffer _tn _context = failwith "CUDA low-level: NOT IMPLEMENTED YET"

open Base
module Tn = Tnode
module Lazy = Utils.Lazy
module Debug_runtime = Utils.Debug_runtime

[%%global_debug_log_level Nothing]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

type mem_properties =
  | Local_only
      (** The array is only needed for a single computation and is allocated locally (or spilled). *)
  | Global
      (** Could not perform optimizations: the array is computed directly in the global memory. *)
[@@deriving sexp, equal, compare, variants]

type tn_info = {
  tn : Tn.t;
  global : string option;
      (** A global device array, if any. This becomes [Cudajit.deviceptr] in a context. *)
  local : string option;  (** A local name, if any. *)
  mem : mem_properties;
  dims : int array;
  num_typ : string;
      (** The type of the stored values: [short] (precision [Half]), [float] (precision [Single]),
          [double] (precision [Double]). *)
  zero_initialized : bool;
}
[@@deriving sexp_of]

type physical_device = {
  dev : (Cudajit.device[@sexp.opaque]);
  ordinal : int;
  primary_context : (Cudajit.context[@sexp.opaque]);
  mutable postprocess_queue : (context * (output:string list -> unit)) list;
}
[@@deriving sexp_of]

and device = {
  physical : physical_device;
  stream : (Cudajit.stream[@sexp.opaque]);
  subordinal : int;
}

and context = {
  label : string;
  ctx : (Cudajit.context[@sexp.opaque]);
  device : device;
  run_module : (Cudajit.module_[@sexp.opaque]) option;
      (** Code jitted for this context, independent of the parent and child contexts. *)
  all_arrays : tn_info Map.M(Tn).t;
  global_arrays : (Cudajit.deviceptr[@sexp.opaque]) Map.M(Tn).t;
      (** This map contains only the global arrays, where [all_arrays.(key).global] is [Some name]. *)
}
[@@deriving sexp_of]

type info_nodes = {
  nodes : tn_info Hashtbl.M(Tn).t;
  used_tensors : Hash_set.M(Tn).t;
  get_ident : Tn.t -> string;
}
[@@deriving sexp_of]

let get_name { physical = { ordinal; _ }; subordinal; _ } =
  Int.to_string ordinal ^ "_" ^ Int.to_string subordinal

let global_config = ref Backend_types.For_parallel_copying

let init device =
  {
    label = "cuda " ^ get_name device;
    ctx = device.physical.primary_context;
    device;
    all_arrays = Map.empty (module Tn);
    global_arrays = Map.empty (module Tn);
    run_module = None;
  }

let is_initialized, initialize =
  let initialized = ref false in
  ( (fun () -> !initialized),
    fun config ->
      initialized := true;
      global_config := config;
      Cudajit.init () )

let num_physical_devices = Cudajit.device_get_count
let devices = ref @@ Core.Weak.create 0

let get_device ~ordinal =
  if num_physical_devices () <= ordinal then
    invalid_arg [%string "Exec_as_cuda.get_device %{ordinal#Int}: not enough devices"];
  if Core.Weak.length !devices <= ordinal then (
    let old = !devices in
    devices := Core.Weak.create (ordinal + 1);
    Core.Weak.blit old 0 !devices 0 (Core.Weak.length old));
  Option.value_or_thunk (Core.Weak.get !devices ordinal) ~default:(fun () ->
      let dev = Cudajit.device_get ~ordinal in
      let primary_context = Cudajit.device_primary_ctx_retain dev in
      let result = { dev; ordinal; primary_context; postprocess_queue = [] } in
      Core.Weak.set !devices ordinal (Some result);
      result)

let new_virtual_device physical =
  (* FIXME: *)
  let subordinal = 0 in
  let stream = Cudajit.no_stream in
  { physical; stream; subordinal }

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

let set_ctx ctx =
  let cur_ctx = Cudajit.ctx_get_current () in
  if not @@ phys_equal ctx cur_ctx then Cudajit.ctx_set_current ctx

let capture_stdout arg =
  Stdlib.flush Stdlib.stdout;
  let exitp, entrancep = Unix.pipe () and backup = Unix.dup Unix.stdout in
  Unix.dup2 entrancep Unix.stdout;
  let _ = arg () in
  Stdlib.flush Stdlib.stdout;
  Unix.close entrancep;
  Unix.dup2 backup Unix.stdout;
  let ls = ref [] and channel = Unix.in_channel_of_descr exitp in
  try
    while true do
      let line = Stdlib.input_line channel in
      ls := line :: !ls
    done;
    []
  with _ -> List.rev !ls

let finalize ctx =
  if Option.is_some @@ Core.Weak.get !devices ctx.device.physical.ordinal then (
    set_ctx ctx.device.physical.primary_context;
    let output = capture_stdout @@ fun () -> Option.iter ctx.run_module ~f:Cudajit.module_unload in
    Exn.protect
      ~f:(fun () ->
        List.iter ctx.device.physical.postprocess_queue ~f:(fun (f_ctx, f) ->
            if phys_equal f_ctx ctx then f ~output))
      ~finally:(fun () ->
        ctx.device.physical.postprocess_queue <-
          List.filter ctx.device.physical.postprocess_queue ~f:(fun (f_ctx, _) ->
              phys_equal f_ctx ctx));
    Map.iter ctx.global_arrays ~f:(fun ptr -> Cudajit.mem_free ptr))

let unsafe_cleanup ?unsafe_shutdown:_ () =
  let len = Core.Weak.length !devices in
  (* TODO: maybe better to do device_primary_ctx_reset if [unsafe_shutdown=false]. *)
  for i = 0 to len - 1 do
    Option.iter (Core.Weak.get !devices i) ~f:(fun device ->
        Cudajit.device_primary_ctx_release device.dev)
  done;
  Core.Weak.fill !devices 0 len None

let await device =
  (* FIXME: NOT IMPLEMENTED YET *)
  let device = device.physical in
  set_ctx device.primary_context;
  let output = capture_stdout Cudajit.ctx_synchronize in
  Exn.protect
    ~f:(fun () -> List.iter device.postprocess_queue ~f:(fun (_, f) -> f ~output))
    ~finally:(fun () -> device.postprocess_queue <- [])

let is_idle _device = failwith "NOT IMPLEMENTED YET"

let%diagn_sexp from_host ?(rt : (module Minidebug_runtime.Debug_runtime) option) (ctx : context) tn
    =
  match (Map.find ctx.all_arrays tn, Map.find ctx.global_arrays tn) with
  | Some { tn = { Tn.array = (lazy (Some hosted)); _ }; _ }, Some dst ->
      set_ctx ctx.ctx;
      (* FIXME: asynchronous *)
      let f src = Cudajit.memcpy_H_to_D ~dst ~src () in
      Ndarray.map { f } hosted;
      (if Utils.settings.with_debug_level > 0 then
         let module Debug_runtime =
           (val Option.value_or_thunk rt ~default:(fun () -> (module Debug_runtime)))
         in
         [%log "copied", Tn.label tn, Tn.name tn, "from host"]);
      true
  | _ -> false

let%diagn_sexp to_host ?(rt : (module Minidebug_runtime.Debug_runtime) option) (ctx : context) tn =
  match (Map.find ctx.all_arrays tn, Map.find ctx.global_arrays tn) with
  | Some { tn = { Tn.array = (lazy (Some hosted)); _ }; _ }, Some src ->
      set_ctx ctx.ctx;
      (* FIXME: asynchronous *)
      let f dst = Cudajit.memcpy_D_to_H ~dst ~src () in
      Ndarray.map { f } hosted;
      if Utils.settings.with_debug_level > 0 then (
        let module Debug_runtime =
          (val Option.value_or_thunk rt ~default:(fun () -> (module Debug_runtime)))
        in
        [%log "copied", Tn.label tn, Tn.name tn, "to host"];
        if Utils.settings.with_debug_level > 1 then
          [%log_printbox
            let indices = Array.init (Array.length @@ Lazy.force tn.dims) ~f:(fun i -> i - 5) in
            Ndarray.render_array ~indices hosted]);
      true
  | _ -> false

let%diagn_sexp device_to_device ?(rt : (module Minidebug_runtime.Debug_runtime) option) tn
    ~into_merge_buffer ~dst ~src =
  Option.value ~default:false
  @@ Option.map (Map.find src.global_arrays tn) ~f:(fun s_arr ->
         Option.value ~default:false
         @@ Option.map (Map.find dst.global_arrays tn) ~f:(fun d_arr ->
                match into_merge_buffer with
                | Backend_types.No ->
                    set_ctx dst.ctx;
                    Cudajit.memcpy_D_to_D ~dst:d_arr ~src:s_arr ();
                    (if Utils.settings.with_debug_level > 0 then
                       let module Debug_runtime =
                         (val Option.value_or_thunk rt ~default:(fun () -> (module Debug_runtime)))
                       in
                       [%log
                         "copied",
                           Tn.label tn,
                           Tn.name tn,
                           "using merge buffer",
                           (into_merge_buffer : bool),
                           "from device",
                           get_ctx_device src |> get_name]);
                    true (* FIXME: *)
                | Streaming -> failwith "NOT IMPLEMENTED YET"
                | Copy -> failwith "NOT IMPLEMENTED YET"))

(* let pp_semi ppf () = Stdlib.Format.fprintf ppf ";@ " *)
let pp_comma ppf () = Stdlib.Format.fprintf ppf ",@ "

(* let pp_symbol ppf sym = Stdlib.Format.fprintf ppf "%s" @@ Indexing.symbol_ident sym *)
let pp_index ppf sym = Stdlib.Format.fprintf ppf "%s" @@ Indexing.symbol_ident sym

let pp_index_axis ppf = function
  | Indexing.Iterator it -> pp_index ppf it
  | Fixed_idx i -> Stdlib.Format.fprintf ppf "%d" i

let pp_array_offset ppf (idcs, dims) =
  let open Stdlib.Format in
  assert (not @@ Array.is_empty idcs);
  for _ = 0 to Array.length idcs - 3 do
    fprintf ppf "@[<1>("
  done;
  for i = 0 to Array.length idcs - 1 do
    let dim = dims.(i) in
    if i = 0 then fprintf ppf "%a" pp_index_axis idcs.(i)
    else if i = Array.length idcs - 1 then fprintf ppf " * %d +@ %a" dim pp_index_axis idcs.(i)
    else fprintf ppf " * %d +@ %a@])" dim pp_index_axis idcs.(i)
  done

let array_offset_to_string (idcs, dims) =
  let b = Buffer.create 32 in
  let ppf = Stdlib.Format.formatter_of_buffer b in
  pp_array_offset ppf (idcs, dims);
  Stdlib.Format.pp_print_flush ppf ();
  Buffer.contents b

let get_run_ptr array =
  match (array.global, array.local) with
  | _, Some lv -> lv
  | Some rv, _ -> rv
  | None, None -> assert false

let get_run_ptr_debug array =
  match (array.global, array.local) with
  | _, Some lv -> "local_" ^ lv
  | Some rv, _ -> "global_" ^ rv
  | None, None -> assert false

(* let compute_array_offset ~idcs ~dims = Array.fold2_exn idcs dims ~init:0 ~f:(fun offset idx dim
   -> idx + (offset * dim)) *)

let%debug_sexp prepare_node traced_store info tn =
  Hash_set.add info.used_tensors tn;
  Hashtbl.update info.nodes tn ~f:(function
    | Some old -> old
    | None ->
        (* let tn = Low_level.get_node traced_store v in *)
        (* TODO: We will need tn to perform more refined optimizations. *)
        let dims = Lazy.force tn.dims in
        let size_in_elems = Array.fold ~init:1 ~f:( * ) dims in
        let prec = tn.prec in
        let size_in_bytes = size_in_elems * Ops.prec_in_bytes prec in
        let is_on_host = Tn.is_hosted_force tn 31 in
        let is_materialized = Tn.is_hosted_force tn 32 in
        assert (Bool.(Option.is_some (Lazy.force tn.array) = is_on_host));
        let num_typ = Ops.cuda_typ_of_prec prec in
        let mem = if not is_materialized then Local_only else Global in
        let global = if is_local_only mem then None else Some (Tn.name tn) in
        let local = Option.some_if (is_local_only mem) @@ Tn.name tn ^ "_local" in
        let backend_info = sexp_of_mem_properties mem in
        if Utils.settings.with_debug_level > 0 then
          [%log
            "creating",
              (tn.id : int),
              Tn.label tn,
              "mem",
              (backend_info : Sexp.t),
              "prec",
              (prec : Ops.prec),
              "on-host",
              (is_on_host : bool),
              "is-global",
              (Option.is_some global : bool)];
        if not @@ Utils.sexp_mem ~elem:backend_info tn.backend_info then
          tn.backend_info <- Utils.sexp_append ~elem:backend_info tn.backend_info;
        let zero_initialized = (Hashtbl.find_exn traced_store tn).Low_level.zero_initialized in
        { tn; local; mem; dims; size_in_bytes; size_in_elems; num_typ; global; zero_initialized })

let compile_main traced_store info ppf llc : unit =
  let open Stdlib.Format in
  let get_node = Hashtbl.find_exn info.nodes in
  let visited = Hash_set.create (module Tn) in
  let rec pp_ll ppf c : unit =
    match c with
    | Low_level.Noop -> ()
    | Seq (c1, c2) ->
        (* Note: no separator. Filter out some entries known to not generate code to avoid
           whitespace. *)
        fprintf ppf "@[<v 0>%a@]" (pp_print_list pp_ll)
          (List.filter [ c1; c2 ] ~f:(function
            | Noop -> false
            | Zero_out ptr -> not Low_level.(get_node traced_store ptr).zero_initialized
            | _ -> true))
    | For_loop { index = i; from_; to_; body; trace_it = _ } ->
        fprintf ppf "@[<2>for (int@ %a = %d;@ %a <= %d;@ ++%a) {@ %a@]@ }@," pp_index i from_
          pp_index i to_ pp_index i pp_ll body
    | Zero_out tn ->
        if Hash_set.mem visited tn then
          pp_ll ppf
          @@ Low_level.loop_over_dims (Lazy.force tn.dims) ~body:(fun idcs ->
                 Set { tn; idcs; llv = Constant 0.0; debug = "zero_out" })
        else
          let traced = Low_level.(get_node traced_store tn) in
          assert traced.zero_initialized (* The initialization will be emitted by prepare_node. *)
    | Set { tn; idcs; llv; debug } ->
        Hash_set.add visited tn;
        let node = get_node tn in
        let loop_f = pp_float ~num_typ:node.num_typ tn.prec in
        let loop_debug_f = debug_float ~num_typ:node.num_typ tn.prec in
        let num_closing_braces = pp_top_locals ppf llv in
        let num_typ = Ops.cuda_typ_of_prec tn.prec in
        if Utils.settings.debug_log_from_routines then (
          fprintf ppf "@[<2>{@ @[<2>%s new_set_v =@ %a;@]@ " num_typ loop_f llv;
          let v_code, v_idcs = loop_debug_f llv in
          let pp_args =
            pp_print_list @@ fun ppf -> function
            | `Accessor idx ->
                pp_comma ppf ();
                pp_array_offset ppf idx
            | `Value v ->
                pp_comma ppf ();
                pp_print_string ppf v
          in
          let run_ptr_debug = get_run_ptr_debug node in
          let run_ptr = get_run_ptr node in
          let offset = (idcs, node.dims) in
          let debug_line =
            "# " ^ String.substr_replace_all debug ~pattern:"\n" ~with_:"$" ^ "\\n"
          in
          fprintf ppf
            "@ @[<2>if @[<2>(threadIdx.x == 0 && blockIdx.x == 0@]) {@ printf(\"%%d: %s\", \
             log_id);@ printf(@[<h>\"%%d: %s[%%u] = %%f = %s\\n\"@], log_id,@ %a,@ %s[%a]%a);@ @]}"
            debug_line run_ptr_debug v_code pp_array_offset offset run_ptr pp_array_offset offset
            pp_args v_idcs;
          fprintf ppf "@[<2>%s[@,%a] =@ new_set_v;@]@ " (get_run_ptr node) pp_array_offset
            (idcs, node.dims))
        else
          (* No idea why adding any cut hint at the end of the assign line breaks formatting! *)
          fprintf ppf "@[<2>%s[@,%a] =@ %a;@]@ " (get_run_ptr node) pp_array_offset
            (idcs, node.dims) loop_f llv;
        for _ = 1 to num_closing_braces do
          fprintf ppf "@]@ }@,"
        done
    | Comment message ->
        if Utils.settings.debug_log_from_routines then
          fprintf ppf
            "@[<2>if @[<2>(threadIdx.x == 0 && blockIdx.x == 0@]) {@ printf(@[<h>\"%%d: COMMENT: \
             %s\\n\", log_id@]);@ @]}"
            (String.substr_replace_all ~pattern:"%" ~with_:"%%" message)
        else fprintf ppf "/* %s */@ " message
    | Staged_compilation callback -> callback ()
    | Set_local (Low_level.{ scope_id; tn = { prec; _ } }, value) ->
        let num_typ = Ops.cuda_typ_of_prec prec in
        let num_closing_braces = pp_top_locals ppf value in
        fprintf ppf "@[<2>v%d =@ %a;@]" scope_id (pp_float ~num_typ prec) value;
        for _ = 1 to num_closing_braces do
          fprintf ppf "@]@ }@,"
        done
  and pp_top_locals ppf (vcomp : Low_level.float_t) : int =
    match vcomp with
    | Local_scope { id = { scope_id = i; tn = { prec; _ } }; body; orig_indices = _ } ->
        let num_typ = Ops.cuda_typ_of_prec prec in
        (* Arrays are initialized to 0 by default. However, there is typically an explicit
           initialization for virtual nodes. *)
        fprintf ppf "@[<2>{@ %s v%d = 0;@ " num_typ i;
        pp_ll ppf body;
        pp_print_space ppf ();
        1
    | Get_local _ | Get_global _ | Get _ | Constant _ | Embed_index _ -> 0
    | Binop (Arg1, v1, _v2) -> pp_top_locals ppf v1
    | Binop (Arg2, _v1, v2) -> pp_top_locals ppf v2
    | Binop (_, v1, v2) -> pp_top_locals ppf v1 + pp_top_locals ppf v2
    | Unop (_, v) -> pp_top_locals ppf v
  and pp_float ~num_typ prec ppf value =
    let loop = pp_float ~num_typ prec in
    match value with
    | Local_scope { id; _ } ->
        (* Embedding of Local_scope is done by pp_top_locals. *)
        loop ppf @@ Get_local id
    | Get_local id ->
        let get_typ = Ops.cuda_typ_of_prec id.tn.prec in
        if not @@ String.equal num_typ get_typ then fprintf ppf "(%s)" num_typ;
        fprintf ppf "v%d" id.scope_id
    | Get_global (Merge_buffer { source_node_id }, Some idcs) ->
        let tn = Option.value_exn ~here:[%here] @@ Tnode.find ~id:source_node_id in
        fprintf ppf "@[<2>merge_buffer[%a@]]" pp_array_offset (idcs, Lazy.force tn.dims)
    | Get_global _ -> failwith "Exec_as_cuda: Get_global / FFI NOT IMPLEMENTED YET"
    | Get (tn, idcs) ->
        Hash_set.add visited tn;
        let node = get_node tn in
        fprintf ppf "@[<2>%s[%a@]]" (get_run_ptr node) pp_array_offset (idcs, node.dims)
    | Constant c -> fprintf ppf "(%f)" c
    | Embed_index idx ->
        if not @@ List.exists ~f:(String.equal num_typ) [ "int"; "size_t" ] then
          fprintf ppf "(%s)" num_typ;
        pp_index_axis ppf idx
    | Binop (Arg1, v1, _v2) -> loop ppf v1
    | Binop (Arg2, _v1, v2) -> loop ppf v2
    | Binop (op, v1, v2) ->
        let prefix, infix, postfix = Ops.binop_C_syntax prec op in
        fprintf ppf "@[<1>%s%a%s@ %a@]%s" prefix loop v1 infix loop v2 postfix
    | Unop (Identity, v) -> loop ppf v
    | Unop (Relu, v) ->
        (* FIXME: don't recompute v *)
        fprintf ppf "@[<1>(%a > 0.0 ?@ %a : 0.0@])" loop v loop v
  and debug_float ~num_typ prec (value : Low_level.float_t) : string * 'a list =
    let loop = debug_float ~num_typ prec in
    match value with
    | Local_scope { id; _ } ->
        (* Not printing the inlined definition: (1) code complexity; (2) don't overload the debug
           logs. *)
        loop @@ Get_local id
    | Get_local id ->
        let get_typ = Ops.cuda_typ_of_prec id.tn.prec in
        let v =
          (if not @@ String.equal num_typ get_typ then "(" ^ num_typ ^ ")" else "")
          ^ "v" ^ Int.to_string id.scope_id
        in
        (v ^ "{=%f}", [ `Value v ])
    | Get_global (Merge_buffer { source_node_id }, Some idcs) ->
        let tn = Option.value_exn ~here:[%here] @@ Tnode.find ~id:source_node_id in
        let v =
          sprintf "@[<2>merge_buffer[%s@]]" (array_offset_to_string (idcs, Lazy.force tn.dims))
        in
        ( "merge " ^ Tn.get_debug_name tn ^ "[%u]{=%f}",
          [ `Accessor (idcs, Lazy.force tn.dims); `Value v ] )
    | Get_global _ -> failwith "Exec_as_cuda: Get_global / FFI NOT IMPLEMENTED YET"
    | Get (tn, idcs) ->
        let node = get_node tn in
        let v =
          sprintf "@[<2>%s[%s@]]" (get_run_ptr node) (array_offset_to_string (idcs, node.dims))
        in
        (get_run_ptr_debug node ^ "[%u]{=%f}", [ `Accessor (idcs, node.dims); `Value v ])
    | Constant c -> (Float.to_string c, [])
    | Embed_index (Fixed_idx i) -> (Int.to_string i, [])
    | Embed_index (Iterator s) -> (Indexing.symbol_ident s, [])
    | Binop (Arg1, v1, _v2) -> loop v1
    | Binop (Arg2, _v1, v2) -> loop v2
    | Binop (op, v1, v2) ->
        let prefix, infix, postfix = Ops.binop_C_syntax prec op in
        let v1, idcs1 = loop v1 in
        let v2, idcs2 = loop v2 in
        (String.concat [ prefix; v1; infix; " "; v2; postfix ], idcs1 @ idcs2)
    | Unop (Identity, v) -> loop v
    | Unop (Relu, v) ->
        let v, idcs = loop v in
        (String.concat [ "("; v; " > 0.0 ? "; v; " : 0.0)" ], idcs @ idcs)
  in
  pp_ll ppf llc

let prepare_nodes traced_store info (llc : Low_level.t) =
  let prepare_node = prepare_node traced_store info in
  let rec loop llc =
    match llc with
    | Low_level.Noop | Low_level.Comment _ | Low_level.Staged_compilation _ -> ()
    | Low_level.Seq (c1, c2) ->
        loop c1;
        loop c2
    | Low_level.For_loop { body; _ } -> loop body
    | Low_level.Zero_out tn -> prepare_node tn
    | Low_level.Set { tn; llv; _ } ->
        prepare_node tn;
        loop_float llv
    | Low_level.Set_local (_, llv) -> loop_float llv
  and loop_float llv =
    match llv with
    | Low_level.Local_scope { body; _ } -> loop body
    | Low_level.Get_local _ | Low_level.Get_global (_, _) -> ()
    | Low_level.Get (tn, _) -> prepare_node tn
    | Low_level.Binop (_, v1, v2) ->
        loop_float v1;
        loop_float v2
    | Low_level.Unop (_, v) -> loop_float v
    | Low_level.Constant _ | Low_level.Embed_index _ -> ()
  in
  loop llc

type code = {
  ptx : (Cudajit.compile_to_ptx_result[@sexp.opaque]);
  info : info_nodes;
  bindings : Indexing.unit_bindings;
  name : string;
}
[@@deriving sexp_of]

type code_batch = {
  ptx : (Cudajit.compile_to_ptx_result[@sexp.opaque]);
  infos : info_nodes option array;
  bindings : Indexing.unit_bindings;
  names : string option array;
}
[@@deriving sexp_of]

let%track_sexp compile_proc ~name ~get_ident ppf idx_params
    Low_level.{ traced_store; llc; merge_node } =
  let open Stdlib.Format in
  let info =
    { nodes = Hashtbl.create (module Tn); used_tensors = Hash_set.create (module Tn); get_ident }
  in
  prepare_nodes traced_store info llc;
  let arrays = Hash_set.to_list info.used_tensors in
  let params =
    List.filter_map arrays ~f:(fun tn ->
        let node = Hashtbl.find_exn info.nodes tn in
        if Utils.settings.with_debug_level > 0 then
          [%log "array-used:", (tn : Tn.t), Tn.label tn, (node.mem : mem_properties)];
        match node.mem with
        | Local_only -> None
        | Global -> Option.map node.global ~f:(fun n -> node.num_typ ^ " *" ^ n))
  in
  let idx_params =
    List.map idx_params ~f:(fun { Indexing.static_symbol; _ } ->
        "int " ^ Indexing.symbol_ident static_symbol)
  in
  let merge_buffer_param =
    Option.to_list merge_node
    |> List.map ~f:(fun tn -> Ops.cuda_typ_of_prec tn.prec ^ " *merge_buffer")
  in
  let log_id = if Utils.settings.debug_log_from_routines then [ "int log_id" ] else [] in
  fprintf ppf "extern \"C\" __global__ void %s(%a) {@," name
    (pp_print_list ~pp_sep:pp_comma pp_print_string)
  @@ log_id @ merge_buffer_param @ idx_params @ params;
  fprintf ppf
    "/* FIXME: single-threaded for now. */@,if (threadIdx.x != 0 || blockIdx.x != 0) { return; }@ ";
  (* TODO: The following link seems to claim it's better to expand into loops.
     https://stackoverflow.com/questions/23712558/how-do-i-best-initialize-a-local-memory-array-to-0 *)
  fprintf ppf "/* Thread-local declarations and initialization. */@,";
  List.iter arrays ~f:(fun tn ->
      let node = Hashtbl.find_exn info.nodes tn in
      match node.mem with
      | Local_only ->
          Option.iter node.local ~f:(fun t_name ->
              fprintf ppf "%s %s[%d]%s;@," node.num_typ t_name node.size_in_elems
                (if (Hashtbl.find_exn traced_store tn).zero_initialized then " = {0}" else ""))
      | Global when node.zero_initialized ->
          Option.iter node.global ~f:(fun t_name ->
              Stdlib.Format.fprintf ppf "@[<2>memset(%s, 0, %d);@]@ " t_name node.size_in_bytes)
      | _ -> ());
  fprintf ppf "/* Main logic. */@,";
  compile_main traced_store info ppf llc;
  fprintf ppf "@,}@.";
  info

let%diagn_sexp cuda_to_ptx ~name cu_src =
  let f_name = name ^ "-cudajit-debug" in
  if Utils.settings.output_debug_files_in_run_directory then (
    let oc = Out_channel.open_text @@ f_name ^ ".cu" in
    Stdio.Out_channel.output_string oc cu_src;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc);
  [%log "compiling to PTX"];
  let module Cu = Cudajit in
  let with_debug =
    Utils.settings.output_debug_files_in_run_directory || Utils.settings.with_debug_level > 0
  in
  let ptx = Cu.compile_to_ptx ~cu_src ~name ~options:[ "--use_fast_math" ] ~with_debug in
  if Utils.settings.output_debug_files_in_run_directory then (
    let f_name = name ^ "-cudajit-debug" in
    let oc = Out_channel.open_text @@ f_name ^ ".ptx" in
    Stdio.Out_channel.output_string oc @@ Cu.string_from_ptx ptx;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc;
    let oc = Out_channel.open_text @@ f_name ^ ".cu_log" in
    Stdio.Out_channel.output_string oc @@ Option.value_exn ~here:[%here] ptx.log;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc);
  ptx

type buffer_ptr = Cudajit.deviceptr

let sexp_of_buffer_ptr (Cudajit.Deviceptr ptr : buffer_ptr) =
  Sexp.Atom (Unsigned.UInt64.to_hexstring ptr)

let alloc_buffer ?old_buffer ~size_in_bytes () =
  match old_buffer with
  | Some (old_ptr, old_size) when size_in_bytes <= old_size -> old_ptr
  | Some (old_ptr, _old_size) ->
      Cudajit.mem_free old_ptr;
      Cudajit.mem_alloc ~byte_size:size_in_bytes
  | None -> Cudajit.mem_alloc ~byte_size:size_in_bytes

let%diagn_sexp link_proc (old_context : context) ~name info ptx =
  let module Cu = Cudajit in
  let ctx = old_context.ctx in
  set_ctx ctx;
  let global_arrays =
    Hashtbl.fold ~init:old_context.global_arrays info.nodes ~f:(fun ~key ~data:node globals ->
        Option.value_map ~default:globals node.global ~f:(fun _name ->
            if Utils.settings.with_debug_level > 0 then [%log "mem_alloc", _name];
            set_ctx ctx;
            let ptr () = Cu.mem_alloc ~byte_size:node.size_in_bytes in
            Map.update globals key ~f:(fun old -> Option.value_or_thunk old ~default:ptr)))
  in
  let run_module = Cu.module_load_data_ex ptx [] in
  let func = Cu.module_get_function run_module ~name in
  [%log "compilation finished"];
  (func, global_arrays, run_module)

let compile ?name bindings ({ Low_level.llc; _ } as lowered) =
  let get_ident = Low_level.get_ident_within_code ~no_dots:true [| llc |] in
  let name : string =
    Option.value_or_thunk name ~default:(fun () -> Low_level.extract_block_name [ llc ])
  in
  let idx_params = Indexing.bound_symbols bindings in
  let b = Buffer.create 4096 in
  let ppf = Stdlib.Format.formatter_of_buffer b in
  if Utils.settings.debug_log_from_routines then
    Stdlib.Format.fprintf ppf "@,__device__ int printf (const char * format, ... );@,";
  let info = compile_proc ~name ~get_ident ppf idx_params lowered in
  let ptx = cuda_to_ptx ~name @@ Buffer.contents b in
  { ptx; info; bindings; name }

let compile_batch ~names bindings lowereds =
  let get_ident =
    Low_level.get_ident_within_code ~no_dots:true
    @@ Array.filter_map lowereds ~f:(Option.map ~f:(fun { Low_level.llc; _ } -> llc))
  in
  let idx_params = Indexing.bound_symbols bindings in
  let b = Buffer.create 4096 in
  let ppf = Stdlib.Format.formatter_of_buffer b in
  let infos =
    Array.map2_exn names lowereds
      ~f:(Option.map2 ~f:(fun name lowered -> compile_proc ~name ~get_ident ppf idx_params lowered))
  in
  let name : string =
    String.(
      strip ~drop:(equal_char '_')
      @@ common_prefix (Array.to_list names |> List.concat_map ~f:Option.to_list))
  in
  let ptx = cuda_to_ptx ~name @@ Buffer.contents b in
  { ptx; infos; bindings; names }

let get_global_run_id =
  let next_id = ref 0 in
  fun () ->
    Int.incr next_id;
    if !next_id < 0 then next_id := 0;
    !next_id

let link old_context (code : code) =
  let all_arrays = Map.of_alist_exn (module Tn) @@ Hashtbl.to_alist code.info.nodes in
  let func, global_arrays, run_module = link_proc old_context ~name:code.name code.info code.ptx in
  let context = { old_context with run_module = Some run_module; global_arrays; all_arrays } in
  let idx_params = Indexing.bound_symbols code.bindings in
  let idx_args = List.map idx_params ~f:(fun s -> (s, ref 0)) in
  let%diagn_rt_sexp work () : unit =
    let log_id = get_global_run_id () in
    let log_id_prefix = Int.to_string log_id ^ ": " in
    [%log_result "Launching", code.name, context.label, (log_id : int)];
    let module Cu = Cudajit in
    let log_arg = if Utils.settings.debug_log_from_routines then [ Cu.Int log_id ] else [] in
    let idx_args =
      List.map idx_args ~f:(fun ({ static_symbol; static_range }, i) ->
          if !i < 0 then
            raise
            @@ Utils.User_error
                 [%string
                   "Exec_as_cuda: static index %{Indexing.symbol_ident static_symbol} is negative: \
                    %{!i#Int}"];
          Option.iter static_range ~f:(fun upto ->
              if !i >= upto then
                raise
                @@ Utils.User_error
                     [%string
                       "Exec_as_cuda: static index %{Indexing.symbol_ident static_symbol} is too \
                        big: %{upto#Int}"]);
          Cu.Int !i)
    in
    let args =
      (* TODO: should we prohibit or warn about Local_only tensors that are in
         old_context.global_arrays? *)
      let arrays = Hash_set.to_list code.info.used_tensors in
      List.filter_map arrays ~f:(fun tn ->
          let node = Hashtbl.find_exn code.info.nodes tn in
          match (node.mem, node.global, Map.find global_arrays tn) with
          | Global, Some _, Some ptr -> Some (Cu.Tensor ptr)
          | _ -> None)
    in
    [%log "zeroing-out global memory"];
    set_ctx context.ctx;
    Map.iteri global_arrays ~f:(fun ~key ~data:ptr ->
        if Hash_set.mem code.info.used_tensors key then
          let node = Map.find_exn all_arrays key in
          if node.zero_initialized then
            Cu.memset_d8 ptr Unsigned.UChar.zero ~length:node.size_in_bytes);
    [%log "launching the kernel"];
    (* if Utils.settings.debug_log_from_routines then Cu.ctx_set_limit CU_LIMIT_PRINTF_FIFO_SIZE
       4096; *)
    Cu.launch_kernel func ~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0 Cu.no_stream
    @@ log_arg @ idx_args @ args;
    [%log "kernel launched"];
    if Utils.settings.debug_log_from_routines then
      let postprocess_logs ~output =
        let output = List.filter_map output ~f:(String.chop_prefix ~prefix:log_id_prefix) in
        [%log_entry
          context.label;
          Utils.log_trace_tree _debug_runtime output]
      in
      context.device.physical.postprocess_queue <-
        (context, postprocess_logs) :: context.device.physical.postprocess_queue
  in
  (context, idx_args, Tn.{ description = "launches " ^ code.name ^ " on " ^ context.label; work })

let link_batch old_context (code_batch : code_batch) =
  let idx_params = Indexing.bound_symbols code_batch.bindings in
  let idx_args = List.map idx_params ~f:(fun s -> (s, ref 0)) in
  (* FIXME: NOT IMPLEMENTED YET *)
  (old_context, idx_args, [||])

let to_buffer ?rt:_ _tn ~dst:_ ~src:_ = failwith "CUDA low-level: NOT IMPLEMENTED YET"
let host_to_buffer ?rt:_ _tn ~dst:_ = failwith "CUDA low-level: NOT IMPLEMENTED YET"
let buffer_to_host ?rt:_ _tn ~src:_ = failwith "CUDA low-level: NOT IMPLEMENTED YET"
let get_buffer _tn _context = failwith "CUDA low-level: NOT IMPLEMENTED YET"

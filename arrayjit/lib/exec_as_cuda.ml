open Base

type mem_properties =
  | Local_only
      (** The array is only needed for a single computation and is allocated locally (or spilled). *)
  | Global  (** Could not perform optimizations: the array is computed directly in the global memory. *)
[@@deriving sexp, equal, compare, variants]

type ndarray = {
  hosted : Ndarray.t option;
  global : (string * (Cudajit.deviceptr[@sexp.opaque])) option;  (** A global device array, if any. *)
  local : string option;  (** A local name, if any. *)
  mem : mem_properties;
  dims : int array;
  size_in_bytes : int;
  size_in_elems : int;
  num_typ : string;
      (** The type of the stored values:
          [short] (precision [Half]), [float] (precision [Single]), [double] (precision [Double]). *)
  is_double : bool;
}
[@@deriving sexp_of]

module LA = Lazy_array

type device = { dev : Cudajit.device; ordinal : int; primary_context : Cudajit.context }

type context = {
  ctx : Cudajit.context;
  device : device;
  run_module : Cudajit.module_ option;
      (** Code jitted for this context, independent of the parent and child contexts. *)
  arrays : ndarray Map.M(LA).t;
}

type ctx_info = {
  ctx : Cudajit.context;  (** Context for jitting, independent of the parent and child contexts. *)
  mutable ctx_arrays : ndarray Map.M(LA).t;
  used_tensors : Hash_set.M(LA).t;
}

let init device = { ctx = device.primary_context; device; arrays = Map.empty (module LA); run_module = None }

let is_initialized, initialize =
  let initialized = ref false in
  ( (fun () -> !initialized),
    fun () ->
      initialized := true;
      Cudajit.init () )

let num_devices = Cudajit.device_get_count
let devices = Res.Weak.empty ()

let get_device ~ordinal =
  if num_devices () <= ordinal then
    invalid_arg [%string "Exec_as_cuda.get_device %{ordinal#Int}: not enough devices"];
  let module Array = Res.Weak in
  Option.value_or_thunk devices.(ordinal) ~default:(fun () ->
      let dev = Cudajit.device_get ~ordinal in
      let primary_context = Cudajit.device_primary_ctx_retain dev in
      let result = { dev; ordinal; primary_context } in
      devices.(ordinal) <- Some result;
      result)

let get_ctx_device { device; _ } = device
let to_ordinal { ordinal; _ } = ordinal

let set_ctx ctx =
  let cur_ctx = Cudajit.ctx_get_current () in
  if not @@ phys_equal ctx cur_ctx then Cudajit.ctx_set_current ctx

let finalize ctx =
  if Option.is_some @@ Res.Weak.get devices ctx.device.ordinal then (
    set_ctx ctx.device.primary_context;
    Option.iter ctx.run_module ~f:Cudajit.module_unload;
    Map.iter ctx.arrays ~f:(fun array -> Option.iter array.global ~f:(fun (_, ptr) -> Cudajit.mem_free ptr)))

let unsafe_cleanup () =
  (* TODO: maybe better to do device_primary_ctx_reset. *)
  Res.Weak.iter (function None -> () | Some device -> Cudajit.device_primary_ctx_release device.dev) devices;
  Res.Weak.clear devices

let await device =
  set_ctx device.primary_context;
  Cudajit.ctx_synchronize ()

let from_host (ctx : context) la =
  match Map.find ctx.arrays la with
  | None -> false
  | Some array -> (
      match (array.hosted, array.global) with
      | Some hosted, Some (_, dst) ->
          set_ctx ctx.ctx;
          let f src = Cudajit.memcpy_H_to_D ~dst ~src () in
          Ndarray.map { f } hosted;
          true
      | _ -> false)

let to_host (ctx : context) la =
  match Map.find ctx.arrays la with
  | None -> false
  | Some array -> (
      match (array.hosted, array.global) with
      | Some hosted, Some (_, src) ->
          set_ctx ctx.ctx;
          let f dst = Cudajit.memcpy_D_to_H ~dst ~src () in
          Ndarray.map { f } hosted;
          true
      | _ -> false)

let merge ?(name_suffix = "") la ~dst ~accum ~src =
  let ord ctx = ctx.device.ordinal in
  let name =
    [%string
      "merge_into_%{la.Lazy_array.id#Int}_from_dev_%{ord src#Int}_to_dev_%{ord dst#Int}_%{name_suffix}"]
  in
  ignore (name, accum);
  failwith "NOT IMPLEMENTED YET"

let pp_semi ppf () = Caml.Format.fprintf ppf ";@ "
let pp_comma ppf () = Caml.Format.fprintf ppf ",@ "
let pp_symbol ppf sym = Caml.Format.fprintf ppf "%s" @@ Indexing.symbol_ident sym
let pp_index ppf sym = Caml.Format.fprintf ppf "%s" @@ Indexing.symbol_ident sym

let pp_index_axis ppf = function
  | Indexing.Iterator it -> pp_index ppf it
  | Fixed_idx i -> Caml.Format.fprintf ppf "%d" i

let pp_array_offset ppf (idcs, dims) =
  let open Caml.Format in
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
  let ppf = Caml.Format.formatter_of_buffer b in
  pp_array_offset ppf (idcs, dims);
  Caml.Format.pp_print_flush ppf ();
  Buffer.contents b

let get_run_ptr array =
  match (array.global, array.local) with
  | _, Some lv -> lv
  | Some (rv, _), _ -> rv
  | None, None -> assert false

let prec_to_c_type = function
  | Ops.Void_prec -> "void"
  | Byte_prec _ -> "uint8"
  | Half_prec _ -> (* FIXME: *) "uint16"
  | Single_prec _ -> "float"
  | Double_prec _ -> "double"

let compute_array_offset ~idcs ~dims =
  Array.fold2_exn idcs dims ~init:0 ~f:(fun offset idx dim -> idx + (offset * dim))

let get_array ~traced_store:_ ctx_info (key : LA.t) =
  let default () =
    (* let tn = Low_level.get_node traced_store v in *)
    (* TODO: We will need tn to perform more refined optimizations. *)
    (* let host_size_in_bytes = Ndarray.size_in_bytes key.array in *)
    let dims = Lazy.force key.dims in
    let size_in_elems = Array.fold ~init:1 ~f:( * ) dims in
    let hosted = Lazy.force key.array in
    let size_in_bytes = size_in_elems * Ops.prec_in_bytes key.prec in
    let is_on_host = !(key.hosted) in
    assert (Bool.(Option.is_some hosted = is_on_host));
    let is_double = Ops.is_double_prec key.prec in
    let num_typ = prec_to_c_type key.prec in
    let mem = if not is_on_host then Local_only else Global in
    let global =
      if is_local_only mem then None
      else (
        if !Low_level.with_debug then Stdio.printf "Exec_as_cuda.get_array: mem_alloc %s\n%!" (LA.name key);
        set_ctx ctx_info.ctx;
        Some (LA.name key, Cudajit.mem_alloc ~byte_size:size_in_bytes))
    in
    let local = Option.some_if (is_local_only mem) @@ LA.name key ^ "_local" in
    let backend_info = (Sexp.to_string_hum @@ sexp_of_mem_properties mem) ^ ";" in
    if not @@ String.is_substring key.backend_info ~substring:backend_info then
      key.backend_info <- key.backend_info ^ backend_info;
    let data = { hosted; local; mem; dims; size_in_bytes; size_in_elems; num_typ; is_double; global } in
    ctx_info.ctx_arrays <- Map.add_exn ctx_info.ctx_arrays ~key ~data;
    data
  in
  Option.value_or_thunk (Map.find ctx_info.ctx_arrays key) ~default

let jit_binop ~num_typ:_ ~is_double op =
  match op with
  | Ops.Arg1 -> assert false
  | Arg2 -> assert false
  | Add -> ("(", " +", ")")
  | Sub -> ("(", " -", ")")
  | Mul -> ("(", " *", ")")
  | Div -> ("(", " /", ")")
  | ToPowOf when is_double -> ("pow(", ",", ")")
  | ToPowOf -> ("powf(", ",", ")")
  | Relu_gate -> ("(", " > 0.0 ?", " : 0.0)")
(* "((int)(", "> 0.0) * ", ")" *)

let jit_code ~traced_store info ppf llc : unit =
  let open Caml.Format in
  let locals = ref Map.Poly.empty in
  let rec pp_ll ppf c : unit =
    match c with
    | Low_level.Noop -> ()
    | Seq (c1, c2) ->
        (* Note: no separator. Filter out some entries known to not generate code to avoid whitespace. *)
        fprintf ppf "@[<v 0>%a@]" (pp_print_list pp_ll)
          (List.filter [ c1; c2 ] ~f:(function
            | Noop -> false
            | Zero_out ptr -> not Low_level.(get_node traced_store ptr).zero_initialized
            | _ -> true))
    | For_loop { index = i; from_; to_; body; trace_it = _ } ->
        fprintf ppf "@[<2>for (unsigned int@ %a = %d;@ %a <= %d;@ ++%a) {@ %a@]@ }@," pp_index i from_
          pp_index i to_ pp_index i pp_ll body
    | Zero_out array ->
        if Map.mem info.ctx_arrays array then
          failwith
            ("exec_as_cuda: Non-initialization zeroing-out NOT IMPLEMENTED YET: " ^ Sexp.to_string_hum
            @@ [%sexp_of: LA.t] array);
        let tn = Low_level.(get_node traced_store array) in
        assert tn.zero_initialized
        (* The initialization will be emitted by get_array. *)
    | Set (array, idcs, v) ->
        let array = get_array ~traced_store info array in
        let old_locals = !locals in
        let loop_f = pp_float ~num_typ:array.num_typ ~is_double:array.is_double in
        let loop_debug_f = debug_float ~num_typ:array.num_typ ~is_double:array.is_double in
        let num_closing_braces = pp_top_locals ppf v in
        (* No idea why adding any cut hint at the end of the assign line breaks formatting! *)
        fprintf ppf "@[<2>%s[@,%a] =@ %a;@]@ " (get_run_ptr array) pp_array_offset (idcs, array.dims) loop_f v;
        (if !Low_level.debug_verbose_trace then
           let v_code, v_idcs = loop_debug_f v in
           fprintf ppf
             "@ @[<2>if @[<2>(threadIdx.x == 0 && blockIdx.x == 0@]) {@ printf(@[<h>\"TRACE: %s[%%u] = %%f = \
              %s\\n\"@],@ %a,@ %s[%a]%a);@ @]}"
             (get_run_ptr array) v_code pp_array_offset (idcs, array.dims) (get_run_ptr array) pp_array_offset
             (idcs, array.dims)
             ( pp_print_list @@ fun ppf -> function
               | `Accessor idx ->
                   pp_comma ppf ();
                   pp_array_offset ppf idx
               | `Value v ->
                   pp_comma ppf ();
                   pp_print_string ppf v )
             v_idcs);
        for _ = 1 to num_closing_braces do
          fprintf ppf "@]@ }@,"
        done;
        locals := old_locals
    | Comment message ->
        fprintf ppf "/* %s */@ " message;
        if !Low_level.debug_verbose_trace then
          fprintf ppf
            "@[<2>if @[<2>(threadIdx.x == 0 && blockIdx.x == 0@]) {@ printf(@[<h>\"TRACE: %s\\n\"@]);@ @]}"
            (String.substr_replace_all ~pattern:"%" ~with_:"%%" message)
    | Staged_compilation callback -> callback ()
    | Set_local (({ scope_id; _ } as id), value) ->
        let num_typ, is_double = Map.find_exn !locals id in
        let old_locals = !locals in
        let num_closing_braces = pp_top_locals ppf value in
        fprintf ppf "@[<2>v%d =@ %a;@]" scope_id (pp_float ~num_typ ~is_double) value;
        for _ = 1 to num_closing_braces do
          fprintf ppf "@]@ }@,"
        done;
        locals := old_locals
  and pp_top_locals ppf (vcomp : Low_level.float_t) : int =
    match vcomp with
    | Local_scope { id = { scope_id = i; _ } as id; prec; body; orig_indices = _ } ->
        let typ = prec_to_c_type prec in
        (* Arrays are initialized to 0 by default. However, there is typically an explicit
           initialization for virtual nodes. *)
        fprintf ppf "@[<2>{@ %s v%d = 0;@ " typ i;
        locals := Map.update !locals id ~f:(fun _ -> (typ, Ops.is_double_prec prec));
        pp_ll ppf body;
        pp_print_space ppf ();
        1
    | Get_local _ | Get_global _ | Get _ | Constant _ -> 0
    | Binop (Arg1, v1, _v2) -> pp_top_locals ppf v1
    | Binop (Arg2, _v1, v2) -> pp_top_locals ppf v2
    | Binop (_, v1, v2) -> pp_top_locals ppf v1 + pp_top_locals ppf v2
    | Unop (_, v) -> pp_top_locals ppf v
  and pp_float ~num_typ ~is_double ppf value =
    let loop = pp_float ~num_typ ~is_double in
    match value with
    | Local_scope { id; _ } ->
        (* Embedding of Local_scope is done by pp_top_locals. *)
        loop ppf @@ Get_local id
    | Get_local id ->
        let typ, _local_is_double = Map.find_exn !locals id in
        if not @@ String.equal num_typ typ then fprintf ppf "(%s)" num_typ;
        fprintf ppf "v%d" id.scope_id
    | Get_global _ -> failwith "Exec_as_cuda: Get_global / FFI NOT IMPLEMENTED YET"
    | Get (array, idcs) ->
        let array = get_array ~traced_store info array in
        fprintf ppf "@[<2>%s[%a@]]" (get_run_ptr array) pp_array_offset (idcs, array.dims)
    | Constant c -> fprintf ppf "(%f)" c
    | Binop (Arg1, v1, _v2) -> loop ppf v1
    | Binop (Arg2, _v1, v2) -> loop ppf v2
    | Binop (op, v1, v2) ->
        let prefix, infix, postfix = jit_binop ~num_typ ~is_double op in
        fprintf ppf "@[<1>%s%a%s@ %a@]%s" prefix loop v1 infix loop v2 postfix
    | Unop (Identity, v) -> loop ppf v
    | Unop (Relu, v) ->
        (* FIXME: don't recompute v *)
        fprintf ppf "@[<1>(%a > 0.0 ?@ %a : 0.0@])" loop v loop v
  and debug_float ~num_typ ~is_double (value : Low_level.float_t) : string * 'a list =
    let loop = debug_float ~num_typ ~is_double in
    match value with
    | Local_scope { id; _ } ->
        (* Not printing the inlined definition: (1) code complexity; (2) don't overload the debug logs. *)
        loop @@ Get_local id
    | Get_local id ->
        let typ, _local_is_double = Map.find_exn !locals id in
        let v =
          (if not @@ String.equal num_typ typ then "(" ^ num_typ ^ ")" else "")
          ^ "v" ^ Int.to_string id.scope_id
        in
        (v ^ "{=%f}", [ `Value v ])
    | Get_global _ -> failwith "Exec_as_cuda: Get_global / FFI NOT IMPLEMENTED YET"
    | Get (ptr, idcs) ->
        let array = get_array ~traced_store info ptr in
        let v = sprintf "@[<2>%s[%s@]]" (get_run_ptr array) (array_offset_to_string (idcs, array.dims)) in
        (get_run_ptr array ^ "[%u]{=%f}", [ `Accessor (idcs, array.dims); `Value v ])
    | Constant c -> (Float.to_string c, [])
    | Binop (Arg1, v1, _v2) -> loop v1
    | Binop (Arg2, _v1, v2) -> loop v2
    | Binop (op, v1, v2) ->
        let prefix, infix, postfix = jit_binop ~num_typ ~is_double op in
        let v1, idcs1 = loop v1 in
        let v2, idcs2 = loop v2 in
        (String.concat [ prefix; v1; infix; " "; v2; postfix ], idcs1 @ idcs2)
    | Unop (Identity, v) -> loop v
    | Unop (Relu, v) ->
        let v, idcs = loop v in
        (String.concat [ "("; v; " > 0.0 ? "; v; " : 0.0)" ], idcs @ idcs)
  in
  pp_ll ppf llc

let jit_func ~name ?(verbose = false) (old_context : context) idx_params (traced_store, llc) =
  set_ctx old_context.ctx;
  if verbose then Stdio.printf "Exec_as_cuda.jit: generating the .cu source\n%!";
  let info =
    { ctx = old_context.ctx; ctx_arrays = old_context.arrays; used_tensors = Hash_set.create (module LA) }
  in
  let b = Buffer.create 4096 in
  let ppf = Caml.Format.formatter_of_buffer b in
  jit_code ~traced_store info ppf llc;
  Caml.Format.pp_print_newline ppf ();
  let cu_body = Buffer.contents b in
  let arrays = Hash_set.to_list info.used_tensors in
  let params, args =
    List.unzip
    @@ List.filter_map arrays ~f:(fun la ->
           let tn = Map.find_exn info.ctx_arrays la in
           match tn.mem with
           | Local_only -> None
           | Global -> Option.map tn.global ~f:(fun (n, ptr) -> (tn.num_typ ^ " *" ^ n, ptr)))
  in
  let idx_params =
    List.map idx_params ~f:(fun (Indexing.Static_symbol s) -> "int " ^ Indexing.symbol_ident s)
  in
  (* TODO: optimize zero-initializations? E.g.
     https://stackoverflow.com/questions/23712558/how-do-i-best-initialize-a-local-memory-array-to-0 *)
  let thread_decls =
    List.filter_map arrays ~f:(fun la ->
        let tn = Map.find_exn info.ctx_arrays la in
        match tn.mem with
        | Local_only ->
            Option.map tn.local ~f:(fun t_name ->
                tn.num_typ ^ " " ^ t_name ^ "[" ^ Int.to_string tn.size_in_elems
                ^ if (Hashtbl.find_exn traced_store la).zero_initialized then "] = {0};" else "];")
        | _ -> None)
  in
  let cu_src =
    [%string
      {|
%{if !Low_level.debug_verbose_trace then "__device__ int printf (const char * format, ... );" else ""}
extern "C" __global__ void %{name}(%{String.concat ~sep:", " @@ idx_params @ params}) {
  /* TODO: this initial toy prototype is single-threaded. */
  if (threadIdx.x != 0 || blockIdx.x != 0) { return; }
  
  /* Thread-local declarations. */
  %{String.concat ~sep:"\n  " thread_decls}

  /* Main logic. */
  %{String.substr_replace_all cu_body ~pattern:"\n" ~with_:"\n  "}
}
|}]
  in
  let f_name = name ^ "-cudajit-debug" in
  if !Low_level.with_debug && !Low_level.keep_files_in_run_directory then (
    let oc = Out_channel.open_text @@ f_name ^ ".cu" in
    Stdio.Out_channel.output_string oc cu_src;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc);
  if verbose then Stdio.printf "Exec_as_cuda.jit: compiling to PTX\n%!";
  let module Cu = Cudajit in
  let ptx =
    Cu.compile_to_ptx ~cu_src ~name ~options:[ "--use_fast_math" ] ~with_debug:!Low_level.with_debug
  in
  if !Low_level.with_debug && !Low_level.keep_files_in_run_directory then (
    let f_name = name ^ "-cudajit-debug" in
    let oc = Out_channel.open_text @@ f_name ^ ".ptx" in
    Stdio.Out_channel.output_string oc @@ Cu.string_from_ptx ptx;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc;
    let oc = Out_channel.open_text @@ f_name ^ ".cu_log" in
    Stdio.Out_channel.output_string oc @@ Option.value_exn ptx.log;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc);
  let run_module = Cu.module_load_data_ex ptx [] in
  let func = Cu.module_get_function run_module ~name in
  let args = List.map args ~f:(fun p -> Cu.Tensor p) in
  if verbose then Stdio.printf "Exec_as_cuda.jit: compilation finished\n%!";
  ( func,
    args,
    { ctx = info.ctx; device = old_context.device; run_module = Some run_module; arrays = info.ctx_arrays } )

type jitted = { context : context; run : unit -> unit; params : unit Indexing.bindings }

let jit ~name ?(verbose = false) old_context params ((traced_store, _) as compiled) =
  let idx_params, idx_args = List.unzip @@ Indexing.assoc_of_bindings params in
  let func, args, context = jit_func ~name ~verbose old_context idx_params compiled in
  let run () =
    if verbose then Stdio.printf "Exec_as_cuda.jit: zeroing-out global memory\n%!";
    set_ctx context.ctx;
    let module Cu = Cudajit in
    Map.iteri context.arrays ~f:(fun ~key:ptr ~data ->
        match data with
        | { global = Some (_, dev_ptr); size_in_bytes; _ } ->
            let tn = Hashtbl.find_exn traced_store ptr in
            if tn.zero_initialized then Cu.memset_d8 dev_ptr Unsigned.UChar.zero ~length:size_in_bytes
        | _ -> ());
    if verbose then Stdio.printf "Exec_as_cuda.jit: launching the kernel\n%!";
    (* if !Low_level.debug_verbose_trace then Cu.ctx_set_limit CU_LIMIT_PRINTF_FIFO_SIZE 4096; *)
    let idx_args = List.map idx_args ~f:(fun i -> Cu.Int !i) in
    Cu.launch_kernel func ~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0 Cu.no_stream @@ idx_args @ args;
    if verbose then Stdio.printf "Exec_as_cuda.jit: kernel launched\n%!"
  in
  { context; run; params }

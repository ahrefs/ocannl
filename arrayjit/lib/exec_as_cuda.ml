open Base

let sync_threads_on_update = ref true

(* let session_context = *)

type mem_properties =
  | Local_only
      (** The array is only needed for a single computation and is allocated locally (or spilled). *)
  | Device_finally_host
      (** The array is computed on-device and then copied to host, if the flag [is_final] is true. *)
  | Constant
      (** The array is accessed directly in the global memory but is not modified by the computation. *)
  | From_host
      (** The array is copied from host when [is_initial], modified in the global memory, and copied
          to host when [is_final]. *)
[@@deriving sexp, equal, compare, variants]

type ndarray = {
  hosted : Ndarray.t option;
  global : string option;  (** A global device array, if any. *)
  global_ptr : (Cudajit.deviceptr Lazy.t[@sexp.opaque]) option;
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

(* let session_results = ref [] *)
let hoist_dynamic_indices = ref false
module LA = Low_level.Lazy_array

type session_state = {
  mutable ctx : Cudajit.context option;
  arrays : (LA.t, ndarray) Hashtbl.t;
  mutable last_module : Cudajit.module_ option;
}

let session_state = { ctx = None; arrays = Hashtbl.create (module LA); last_module = None }
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
  match (array.global, array.local) with _, Some lv -> lv | Some rv, _ -> rv | None, None -> assert false

let prec_to_c_type = function
  | Ndarray.Void_prec -> "void"
  | Half_prec _ -> (* FIXME: *) "uint16"
  | Single_prec _ -> "float"
  | Double_prec _ -> "double"

let compute_array_offset ~idcs ~dims =
  Array.fold2_exn idcs dims ~init:0 ~f:(fun offset idx dim -> idx + (offset * dim))

let get_array ~traced_store v =
  let { arrays; _ } = session_state in
  Hashtbl.find_or_add arrays v ~default:(fun () ->
      let tn = Low_level.get_node traced_store v in
      (* let host_size_in_bytes = Ndarray.size_in_bytes v.array in *)
      let dims = Lazy.force v.dims in
      let size_in_elems = Array.fold ~init:1 ~f:( * ) dims in
      let hosted = Lazy.force v.array in
      let size_in_bytes = size_in_elems * Ndarray.prec_in_bytes v.prec in
      let is_on_host = !(v.materialized) in
      let is_double = Ndarray.is_double_prec v.prec in
      let num_typ = prec_to_c_type v.prec in
      let mem =
        if not is_on_host then Local_only
        else if tn.read_only then Constant
        else if not tn.read_before_write then Device_finally_host
        else From_host
      in
      let global_ptr =
        Option.some_if (not @@ is_local_only mem)
        @@
        match mem with
        | Constant ->
            lazy
              (let ptr, size =
                 (* Defer till after compilation, to access the compiled-into module. *)
                 Cudajit.module_get_global
                   (Option.value_exn session_state.last_module)
                   ~name:(LA.name v)
               in
               assert (Unsigned.Size_t.to_int size = size_in_bytes);
               ptr)
        | _ ->
            (* The general case does not require laziness, but it should be OK. *)
            lazy
              (if !Low_level.with_debug then
                 Stdio.printf "Exec_as_cuda.get_array: mem_alloc %s\n%!" (LA.name v);
               Cudajit.mem_alloc ~byte_size:size_in_bytes)
      in
      let global = Option.some_if (not @@ is_local_only mem) @@ LA.name v in
      let local = Option.some_if (is_local_only mem) @@ LA.name v ^ "_local" in
      let backend_info = (Sexp.to_string_hum @@ sexp_of_mem_properties mem) ^ ";" in
      if not @@ String.is_substring v.backend_info ~substring:backend_info then
        v.backend_info <- v.backend_info ^ backend_info;
      { hosted; local; mem; dims; size_in_bytes; size_in_elems; num_typ; is_double; global; global_ptr })

let jit_binop ~num_typ:_ ~is_double op =
  match op with
  | Low_level.Arg1 -> assert false
  | Arg2 -> assert false
  | Add -> ("(", " +", ")")
  | Mul -> ("(", " *", ")")
  | ToPowOf when is_double -> ("pow(", ",", ")")
  | ToPowOf -> ("powf(", ",", ")")
  | Relu_gate -> ("(", " > 0.0 ?", " : 0.0)")
(* "((int)(", "> 0.0) * ", ")" *)

let jit_code ~traced_store ppf llc : unit =
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
        if Hashtbl.mem session_state.arrays array then
          failwith
            ("exec_as_cuda: Non-initialization zeroing-out NOT IMPLEMENTED YET: " ^ Sexp.to_string_hum
            @@ [%sexp_of: LA.t] array);
        let tn = Low_level.(get_node traced_store array) in
        assert tn.zero_initialized
        (* The initialization will be emitted by get_array. *)
    | Set (array, idcs, v) ->
        let array = get_array ~traced_store array in
        let old_locals = !locals in
        let loop_f = pp_float ~num_typ:array.num_typ ~is_double:array.is_double in
        let loop_debug_f = debug_float ~num_typ:array.num_typ ~is_double:array.is_double in
        let num_closing_braces = pp_top_locals ppf v in
        (* No idea why adding any cut hint at the end of the assign line breaks formatting! *)
        fprintf ppf "@[<2>%s[@,%a] =@ %a;@]@ " (get_run_ptr array) pp_array_offset (idcs, array.dims) loop_f
          v;
        (if !Low_level.debug_verbose_trace then
           let v_code, v_idcs = loop_debug_f v in
           fprintf ppf
             "@ @[<2>if @[<2>(threadIdx.x == 0 && blockIdx.x == 0@]) {@ printf(@[<h>\"TRACE: %s[%%u] = %%f = \
              %s\\n\"@],@ %a,@ %s[%a]%a);@ @]}"
             (get_run_ptr array) v_code pp_array_offset (idcs, array.dims) (get_run_ptr array)
             pp_array_offset (idcs, array.dims)
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
        locals := Map.update !locals id ~f:(fun _ -> (typ, Ndarray.is_double_prec prec));
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
        let array = get_array ~traced_store array in
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
        let array = get_array ~traced_store ptr in
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

let new_context ?(device_num = 0) () =
  let num_devices = Cudajit.device_get_count () in
  if num_devices <= device_num then None
  else
    let device = Cudajit.device_get ~ordinal:device_num in
    Some (Cudajit.ctx_create ~flags:0 device)

let cleanup_session () =
  if Option.is_none session_state.ctx then Cudajit.init ();
  Option.iter session_state.last_module ~f:Cudajit.module_unload;
  session_state.last_module <- None;
  Hashtbl.iter session_state.arrays ~f:(fun array ->
      Option.iter array.global_ptr ~f:(fun (lazy ptr) ->
          if not @@ is_constant array.mem then (
            if !Low_level.with_debug then
              Stdio.printf "Exec_as_cuda.cleanup_session: mem_free %s\n%!" (Option.value_exn array.global);
            Cudajit.mem_free ptr)));
  Hashtbl.clear session_state.arrays;
  Option.iter session_state.ctx ~f:Cudajit.ctx_destroy;
  (* For now we stick with device 0. *)
  session_state.ctx <- new_context ()

let jit_func ~name ?(verbose = false) (traced_store, llc) =
  let module Cu = Cudajit in
  Hashtbl.filter_inplace session_state.arrays ~f:(fun array -> not @@ is_constant array.mem);
  Option.iter session_state.last_module ~f:Cu.module_unload;
  session_state.last_module <- None;
  if Option.is_none session_state.ctx then (
    if verbose then Stdio.printf "Exec_as_cuda.jit: initializing the CUDA context\n%!";
    cleanup_session ());
  if Option.is_none session_state.ctx then invalid_arg "Exec_as_cuda: no device found";
  if verbose then Stdio.printf "Exec_as_cuda.jit: generating the .cu source\n%!";
  let b = Buffer.create 4096 in
  let ppf = Caml.Format.formatter_of_buffer b in
  jit_code ~traced_store ppf llc;
  Caml.Format.pp_print_newline ppf ();
  let cu_body = Buffer.contents b in
  let arrays = Hashtbl.to_alist session_state.arrays in
  let params, args =
    List.unzip
    @@ List.filter_map arrays ~f:(fun (_, tn) ->
           match tn.mem with
           | Local_only | Constant -> None
           | From_host | Device_finally_host ->
               Option.map tn.global ~f:(fun t_name -> (tn.num_typ ^ " *" ^ t_name, tn.global_ptr)))
  in
  (* TODO: optimize zero-initializations? E.g.
     https://stackoverflow.com/questions/23712558/how-do-i-best-initialize-a-local-memory-array-to-0 *)
  let constant_defs =
    List.filter_map arrays ~f:(fun (ptr, tn) ->
        match tn.mem with
        | Constant ->
            Option.map tn.global ~f:(fun t_name ->
                "__constant__ " ^ tn.num_typ ^ " " ^ t_name ^ "[" ^ Int.to_string tn.size_in_elems
                ^ if (Hashtbl.find_exn traced_store ptr).zero_initialized then "] = {0};" else "];")
        | _ -> None)
  in
  let thread_decls =
    List.filter_map arrays ~f:(fun (ptr, tn) ->
        match tn.mem with
        | Local_only ->
            Option.map tn.local ~f:(fun t_name ->
                tn.num_typ ^ " " ^ t_name ^ "[" ^ Int.to_string tn.size_in_elems
                ^ if (Hashtbl.find_exn traced_store ptr).zero_initialized then "] = {0};" else "];")
        | _ -> None)
  in
  let finalizers =
    Array.of_list arrays
    |> Array.filter_map ~f:(fun (_, tn) ->
           match tn.mem with
           | Device_finally_host ->
               Option.map2 tn.local tn.global ~f:(fun l_name g_name ->
                   let b = Buffer.create 4096 in
                   let ppf = Caml.Format.formatter_of_buffer b in
                   let body idcs =
                     Low_level.Staged_compilation
                       (fun () ->
                         Caml.Format.fprintf ppf "@[<2>%s[%a] =@ %s[%a];@]" g_name pp_array_offset
                           (idcs, tn.dims) l_name pp_array_offset (idcs, tn.dims))
                   in
                   let loops = Low_level.loop_over_dims tn.dims ~body in
                   jit_code ~traced_store ppf loops;
                   Caml.Format.pp_print_newline ppf ();
                   Buffer.contents b)
           | _ -> None)
  in
  let cu_src =
    [%string
      {|
%{if !Low_level.debug_verbose_trace then "__device__ int printf (const char * format, ... );" else ""}
%{String.concat ~sep:"\n" constant_defs}
extern "C" __global__ void %{name}(bool is_final, %{String.concat ~sep:", " params}) {
  /* TODO: this initial toy prototype is single-threaded. */
  if (threadIdx.x != 0 || blockIdx.x != 0) { return; }
  
  /* Thread-local declarations. */
  %{String.concat ~sep:"\n  " thread_decls}

  /* Main logic. */
  %{String.substr_replace_all cu_body ~pattern:"\n" ~with_:"\n  "}

  /* Finalization: copy local-to-global. */
  if (is_final) {
    %{String.concat_array ~sep:"\n    "
    @@ Array.map finalizers ~f:(String.substr_replace_all ~pattern:"\n" ~with_:"\n    ")}
  }
}
|}]
  in
  (* Constants will be referred to via cuModuleGetGlobal. *)
  let f_name = name ^ "-cudajit-debug" in
  if !Low_level.with_debug && !Low_level.keep_files_in_run_directory then (
    let oc = Out_channel.open_text @@ f_name ^ ".cu" in
    Stdio.Out_channel.output_string oc cu_src;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc);
  if verbose then Stdio.printf "Exec_as_cuda.jit: compiling to PTX\n%!";
  let ptx =
    Cu.compile_to_ptx ~cu_src ~name ~options:[ "--use_fast_math" ] ~with_debug:!Low_level.with_debug
  in
  if !Low_level.with_debug && !Low_level.keep_files_in_run_directory then (
    let f_name = name ^ "-cudajit-debug" in
    let oc = Out_channel.open_text @@ f_name ^ ".ptx" in
    Stdio.Out_channel.output_string oc @@ Cudajit.string_from_ptx ptx;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc;
    let oc = Out_channel.open_text @@ f_name ^ ".cu_log" in
    Stdio.Out_channel.output_string oc @@ Option.value_exn ptx.log;
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc);
  let module_ = Cu.module_load_data_ex ptx [] in
  session_state.last_module <- Some module_;
  let func = Cu.module_get_function module_ ~name in
  let args = List.map args ~f:(function Some (lazy p) -> Cu.Tensor p | None -> assert false) in
  if verbose then Stdio.printf "Exec_as_cuda.jit: compilation finished\n%!";
  fun ~is_initial ~is_final ->
    if is_initial then (
      if verbose then Stdio.printf "Exec_as_cuda.jit: copying host-to-device\n%!";
      List.iter arrays ~f:(function
        | ptr, { hosted = Some ndarray; global = Some name; global_ptr = Some (lazy dst); size_in_elems; _ }
          ->
            let tn = Hashtbl.find_exn traced_store ptr in
            if tn.read_before_write then (
              let f src = Cu.memcpy_H_to_D ~length:size_in_elems ~dst ~src () in
              if verbose && !Low_level.with_debug then
                Stdio.printf "Exec_as_cuda.jit: memcpy_H_to_D for %s, length: %d\n%!" name size_in_elems;
              Ndarray.map { f } ndarray)
        | _ -> ()));
    if verbose then Stdio.printf "Exec_as_cuda.jit: zeroing-out global memory\n%!";
    List.iter arrays ~f:(function
      | ptr, { global_ptr = Some (lazy device); size_in_bytes; _ } ->
          let tn = Hashtbl.find_exn traced_store ptr in
          if tn.zero_initialized then Cu.memset_d8 device Unsigned.UChar.zero ~length:size_in_bytes
      | _ -> ());
    if verbose then Stdio.printf "Exec_as_cuda.jit: running the kernel\n%!";
    (* if !Low_level.debug_verbose_trace then Cu.ctx_set_limit CU_LIMIT_PRINTF_FIFO_SIZE 4096; *)
    Cu.launch_kernel func ~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0 Cu.no_stream args;
    Cu.ctx_synchronize ();
    if is_final then (
      if verbose then Stdio.printf "Exec_as_cuda.jit: copying device-to-host\n%!";
      List.iter arrays ~f:(function
        | ptr, { hosted = Some ndarray; global = Some name; global_ptr = Some (lazy src); size_in_elems; _ }
          ->
            let tn = Hashtbl.find_exn traced_store ptr in
            if not tn.read_only then (
              let f dst = Cu.memcpy_D_to_H ~length:size_in_elems ~dst ~src () in
              if verbose && !Low_level.with_debug then
                Stdio.printf "Exec_as_cuda.jit: memcpy_D_to_H for %s\n%!" name;
              Ndarray.map { f } ndarray)
        | _ -> ()));
    if verbose then Stdio.printf "Exec_as_cuda.jit: kernel run finished\n%!"

let jit = jit_func

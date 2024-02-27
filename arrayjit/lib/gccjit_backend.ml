open Base

let optimization_level = ref 3

type mem_properties =
  | Local_only  (** The array is only needed for a local computation, is allocated on the stack. *)
  | Global  (** The array has a copy allocated per-cpu-device, in addition to existing on the host. *)
  | Constant  (** The array is read directly from the host. *)
[@@deriving sexp, equal, compare, variants]

let root_ctx = ref None

module Tn = Tnode

type context = {
  label : string;
  arrays : Ndarray.t Map.M(Tn).t;
  result : (Gccjit.result option[@sexp.opaque]);
}
[@@deriving sexp_of]

let unsafe_cleanup ?(unsafe_shutdown = false) () =
  let open Gccjit in
  Option.iter ~f:Context.release !root_ctx;
  if unsafe_shutdown then root_ctx := None
  else
    let ctx = Context.create () in
    Context.set_option ctx Optimization_level !optimization_level;
    root_ctx := Some ctx

let is_initialized, initialize =
  let initialized = ref false in
  ( (fun () -> !initialized),
    fun () ->
      initialized := true;
      unsafe_cleanup () )

let finalize ctx =
  let open Gccjit in
  Option.iter ctx.result ~f:Result.release

let init ~label =
  let result = { label; result = None; arrays = Map.empty (module Tn) } in
  Core.Gc.Expert.add_finalizer_exn result finalize;
  result

type ndarray = {
  nd : Tn.t;  (** The original array. *)
  hosted_ptr : (Gccjit.rvalue[@sexp.opaque]) option;
      (** Pointer to the first value of the associated hosted [Ndarray]. *)
  global_ptr : (Gccjit.rvalue[@sexp.opaque]) option;
      (** Pointer to the first value of the associated [context.arrays]. *)
  local : (Gccjit.lvalue[@sexp.opaque]) option;  (** A local array, if any. *)
  mem : mem_properties;
  dims : int array;
  size_in_bytes : int;
  num_typ : (Gccjit.type_[@sexp.opaque]);
      (** The type of the stored values: [short] (precision [Half]), [float] (precision [Single]),
          [double] (precision [Double]). *)
  is_double : bool;
}
[@@deriving sexp_of]

type ctx_info = {
  ctx : (Gccjit.context[@sexp.opaque]);
  func : (Gccjit.function_[@sexp.opaque]);
  traced_store : (Low_level.traced_store[@sexp.opaque]);
  init_block : (Gccjit.block[@sexp.opaque]);
  mutable ctx_arrays : Ndarray.t Map.M(Tn).t;
  arrays : (Tn.t, ndarray) Hashtbl.t;
}
[@@deriving sexp_of]

let jit_array_offset ctx ~idcs ~dims =
  let open Gccjit in
  let c_index = Type.get ctx Type.Int in
  Array.fold2_exn idcs dims ~init:(RValue.zero ctx c_index) ~f:(fun offset idx dim ->
      RValue.binary_op ctx Plus c_index idx
      @@ RValue.binary_op ctx Mult c_index offset (RValue.int ctx c_index dim))

let zero_out ctx block arr =
  let open Gccjit in
  let c_void_ptr = Type.(get ctx Type.Void_ptr) in
  let c_index = Type.get ctx Type.Size_t in
  let c_int = Type.get ctx Type.Int in
  let cast_void rv = RValue.cast ctx rv c_void_ptr in
  List.find_map ~f:Fn.id
    [ Option.map arr.local ~f:(Fn.compose cast_void LValue.address); arr.global_ptr; arr.hosted_ptr ]
  |> Option.iter ~f:(fun rv_ptr ->
         Block.eval block
         @@ RValue.call ctx (Function.builtin ctx "memset")
              [ rv_ptr; RValue.zero ctx c_int; RValue.int ctx c_index arr.size_in_bytes ])

module Debug_runtime = Utils.Debug_runtime

let%track_sexp get_array ({ ctx; func; arrays; ctx_arrays; traced_store; init_block } as ctx_info)
    (key : Tn.t) : ndarray =
  let open Gccjit in
  Hashtbl.find_or_add arrays key ~default:(fun () ->
      let ta = Low_level.(get_node traced_store key) in
      let dims = Lazy.force key.dims in
      let size_in_elems = Array.fold ~init:1 ~f:( * ) dims in
      let size_in_bytes = size_in_elems * Ops.prec_in_bytes key.prec in
      let is_on_host = Tn.is_hosted_force key 33 in
      let is_materialized = Tn.is_materialized_force key 34 in
      assert (Bool.(Option.is_some (Lazy.force key.array) = is_on_host));
      (* TODO: is the complexity of introducing this function and matching on Ndarray.t needed? *)
      let array c_typ is_double arr =
        let num_typ = Type.(get ctx c_typ) in
        let get_c_ptr ba =
          RValue.ptr ctx (Type.pointer num_typ) @@ Ctypes.bigarray_start Ctypes_static.Genarray ba
        in
        let hosted_ptr = Option.map arr ~f:get_c_ptr in
        let mem =
          if not is_materialized then Local_only else if is_on_host && ta.read_only then Constant else Global
        in
        let arr_typ = Type.array ctx num_typ size_in_elems in
        let local = if is_local_only mem then Some (Function.local func arr_typ @@ Tn.name key) else None in
        (if is_global mem && (not @@ Map.mem ctx_arrays key) then
           let data =
             Ndarray.create_array key.Tn.prec ~dims @@ Constant_fill { values = [| 0. |]; strict = false }
           in
           ctx_info.ctx_arrays <- Map.add_exn ~key ~data ctx_arrays);
        let global_ptr = Option.map (Map.find ctx_info.ctx_arrays key) ~f:Ndarray.(map { f = get_c_ptr }) in
        let backend_info = sexp_of_mem_properties mem in
        let comment_on ptr = Option.value ~default:"not" @@ Option.map ptr ~f:RValue.to_string in
        Block.comment init_block
          [%string
            "Array #%{key.id#Int} %{Tn.label key}: %{Sexp.to_string_hum @@ backend_info} %{comment_on \
             hosted_ptr} hosted, %{comment_on global_ptr} global, %{comment_on @@ Option.map \
             ~f:RValue.lvalue local} local."];
        let result =
          { nd = key; hosted_ptr; global_ptr; local; mem; dims; size_in_bytes; num_typ; is_double }
        in
        if ta.zero_initialized then zero_out ctx init_block result;
        if not @@ Utils.sexp_mem ~elem:backend_info key.backend_info then
          key.backend_info <- Utils.sexp_append ~elem:backend_info key.backend_info;
        result
      in
      match (key.prec, Lazy.force key.array) with
      | _, Some (Byte_nd arr) -> array Type.Unsigned_char false (Some arr)
      | _, Some (Half_nd arr) -> (* FIXME: *) array Type.Float false (Some arr)
      | _, Some (Single_nd arr) -> array Type.Float false (Some arr)
      | _, Some (Double_nd arr) -> array Type.Double true (Some arr)
      | Byte_prec _, None -> array Type.Unsigned_char false None
      | Half_prec _, None -> (* FIXME: *) array Type.Float false None
      | Single_prec _, None -> array Type.Float false None
      | Double_prec _, None -> array Type.Double true None
      | Void_prec, None -> assert false)

let prec_to_kind prec =
  let open Gccjit in
  match prec with
  | Ops.Void_prec -> Type.Void
  | Byte_prec _ -> Type.Unsigned_char
  | Half_prec _ -> (* FIXME: *) Type.Unsigned_short
  | Single_prec _ -> Type.Float
  | Double_prec _ -> Type.Double

let prec_is_double = function Ops.Double_prec _ -> true | _ -> false
let is_builtin_op = function Ops.Add | Sub | Mul | Div -> true | ToPowOf | Relu_gate | Arg2 | Arg1 -> false

let builtin_op = function
  | Ops.Add -> Gccjit.Plus
  | Sub -> Gccjit.Minus
  | Mul -> Gccjit.Mult
  | Div -> Gccjit.Divide
  | ToPowOf | Relu_gate | Arg2 | Arg1 -> invalid_arg "Exec_as_gccjit.builtin_op: not a builtin"

let get_ptr array =
  match array.local with
  | Some lv -> Gccjit.RValue.lvalue lv
  | None -> Option.value_exn @@ Option.first_some array.global_ptr array.hosted_ptr

let get_ptr_debug array =
  match (array.local, array.global_ptr, array.hosted_ptr) with
  | Some _, _, _ -> "local_" ^ Tn.name array.nd
  | None, Some _, _ -> "global_" ^ Tn.name array.nd
  | None, None, Some _ -> "hosted_" ^ Tn.name array.nd
  | None, None, None -> assert false

let%track_sexp jit_code ~name ~log_file ~env ({ ctx; func; _ } as info) initial_block (body : Low_level.t) =
  let open Gccjit in
  let c_int = Type.get ctx Type.Int in
  let c_index = c_int in
  let lookup env indices =
    let open Gccjit in
    try
      Array.map indices ~f:(function
        | Indexing.Fixed_idx i -> RValue.int ctx c_index i
        | Iterator s -> Map.find_exn env s)
    with e ->
      Stdlib.Format.eprintf "exec_as_gccjit: missing index from@ %a@ among environment keys:@ %a\n%!"
        Sexp.pp_hum
        ([%sexp_of: Indexing.axis_index array] indices)
        Sexp.pp_hum
        ([%sexp_of: Indexing.symbol list] @@ Map.keys env);
      raise e
  in
  let c_float = Type.get ctx Type.Float in
  let c_double = Type.get ctx Type.Double in
  let cast_bool num_typ v = RValue.cast ctx (RValue.cast ctx v c_int) num_typ in
  let to_d v = RValue.cast ctx v c_double in
  (* Source of unique identifiers. E.g. local scope ids can be non-unique due to inlining.
     We also need unique ids for computation ordering lvalues. *)
  let uid = ref 0 in
  let get_uid () =
    let id =
      Int.incr uid;
      !uid
    in
    Int.to_string id
  in
  let locals = ref @@ Map.empty (module Low_level.Scope_id) in
  let debug_locals = ref !locals in
  let current_block = ref initial_block in
  let loop_binop op ~num_typ ~is_double ~v1 ~v2 =
    match op with
    | Ops.Add -> RValue.binary_op ctx Plus num_typ v1 v2
    | Sub -> RValue.binary_op ctx Minus num_typ v1 v2
    | Mul -> RValue.binary_op ctx Mult num_typ v1 v2
    | Div -> RValue.binary_op ctx Divide num_typ v1 v2
    | ToPowOf when is_double ->
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "pow") [ to_d v1; to_d v2 ]) num_typ
    | ToPowOf ->
        let base = RValue.cast ctx v1 c_float in
        let expon = RValue.cast ctx v2 c_float in
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "powf") [ base; expon ]) num_typ
    | Relu_gate ->
        let cmp = cast_bool num_typ @@ RValue.comparison ctx Lt (RValue.zero ctx num_typ) v1 in
        RValue.binary_op ctx Mult num_typ cmp @@ v2
    | Arg2 -> v2
    | Arg1 -> v1
  in
  let fprintf = Option.map log_file ~f:(fun _ -> Function.builtin ctx "fprintf") in
  let log_comment c =
    match (log_file, fprintf) with
    | Some f, Some p ->
        Block.eval !current_block
        @@ RValue.call ctx p [ f; RValue.string_literal ctx ("\nCOMMENT: " ^ c ^ "\n") ]
    | _ -> Block.comment !current_block c
  in
  let rec debug_float ~env ~is_double value =
    let loop = debug_float ~env ~is_double in
    match value with
    | Low_level.Local_scope { id; _ } ->
        (* Not printing the inlined definition: (1) code complexity; (2) don't overload the debug logs. *)
        loop @@ Get_local id
    | Get_local id ->
        let lvalue, _typ, _local_is_double = Map.find_exn !debug_locals id in
        (* FIXME(194): Convert according to _typ ?= num_typ. *)
        (LValue.to_string lvalue ^ "{=%g}", [ to_d @@ RValue.lvalue lvalue ])
    | Get_global (C_function f_name, None) -> ("<calls " ^ f_name ^ ">", [])
    | Get_global (External_unsafe { ptr; dims = (lazy dims); prec }, Some idcs) ->
        let idcs = lookup env idcs in
        let typ = Type.get ctx @@ prec_to_kind prec in
        let ptr = RValue.ptr ctx (Type.pointer typ) ptr in
        let offset = jit_array_offset ctx ~idcs ~dims in
        (* FIXME(194): Convert according to typ ?= num_typ. *)
        let v = to_d @@ RValue.lvalue @@ LValue.access_array ptr offset in
        ("external " ^ RValue.to_string ptr ^ "[%d]{=%g}", [ offset; v ])
    | Get_global _ -> failwith "NOT IMPLEMENTED YET"
    | Get (ptr, idcs) ->
        let array = get_array info ptr in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:array.dims in
        (* FIXME(194): Convert according to array.typ ?= num_typ. *)
        let v = to_d @@ RValue.lvalue @@ LValue.access_array (get_ptr array) offset in
        (get_ptr_debug array ^ "[%d]{=%g}", [ offset; v ])
    | Constant c -> (Float.to_string c, [])
    | Embed_index (Fixed_idx i) -> (Int.to_string i, [])
    | Embed_index (Iterator s) -> (Indexing.symbol_ident s ^ "{=%d}", [ Map.find_exn env s ])
    | Binop (Arg1, v1, _v2) -> loop v1
    | Binop (Arg2, _v1, v2) -> loop v2
    | Binop (op, v1, v2) ->
        let prefix, infix, postfix = Ops.binop_C_syntax ~is_double op in
        let v1, fillers1 = loop v1 in
        let v2, fillers2 = loop v2 in
        (String.concat [ prefix; v1; infix; " "; v2; postfix ], fillers1 @ fillers2)
    | Unop (Identity, v) -> loop v
    | Unop (Relu, v) ->
        let v, fillers = loop v in
        (String.concat [ "("; v; " > 0.0 ? "; v; " : 0.0)" ], fillers @ fillers)
  in
  let fflush =
    Option.map log_file ~f:(fun _ ->
        let f_ptr = Type.get ctx Type.File_ptr in
        Function.create ctx Imported (Type.get ctx Void) "fflush" [ Param.create ctx f_ptr "f" ])
  in
  let debug_log_assignment ~env debug idcs array accum_op value v_code =
    match (log_file, fprintf, fflush) with
    | Some f, Some p, Some fl ->
        let v_format, v_fillers = debug_float ~env ~is_double:array.is_double v_code in
        let offset = jit_array_offset ctx ~idcs ~dims:array.dims in
        let debug_line = "# " ^ String.substr_replace_all debug ~pattern:"\n" ~with_:"$" ^ "\n" in
        Block.eval !current_block @@ RValue.call ctx p @@ [ f; RValue.string_literal ctx debug_line ];
        Block.eval !current_block @@ RValue.call ctx p
        @@ f
           :: RValue.string_literal ctx
                [%string
                  {|%{get_ptr_debug array}[%d]{=%g} %{Ops.assign_op_C_syntax accum_op} %g = %{v_format}
|}]
           :: (to_d @@ RValue.lvalue @@ LValue.access_array (get_ptr array) offset)
           :: offset :: to_d value :: v_fillers;
        Block.eval !current_block @@ RValue.call ctx fl [ f ]
    | _ -> ()
  in
  let rec loop_proc ~env ~name (body : Low_level.t) : unit =
    let loop = loop_proc ~env ~name in
    match body with
    | Noop -> ()
    | Low_level.Seq (c1, c2) ->
        loop c1;
        loop c2
    | For_loop { index; from_; to_; body; trace_it = _ } -> jit_for_loop ~env index ~from_ ~to_ body
    | Set { llv = Binop (Arg2, Get (_, _), _); _ } -> assert false
    | Set { array; idcs; llv = Binop (op, Get (array2, idcs2), c2); debug }
      when Tn.equal array array2 && [%equal: Indexing.axis_index array] idcs idcs2 && is_builtin_op op ->
        (* FIXME: maybe it's not worth it? *)
        let array = get_array info array in
        let value = loop_float ~name ~env ~num_typ:array.num_typ ~is_double:array.is_double c2 in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:array.dims in
        let lhs = LValue.access_array (get_ptr array) offset in
        debug_log_assignment ~env debug idcs array op value c2;
        Block.assign_op !current_block lhs (builtin_op op) value
    | Set { array; idcs; llv; debug } ->
        let array = get_array info array in
        let value = loop_float ~name ~env ~num_typ:array.num_typ ~is_double:array.is_double llv in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:array.dims in
        let lhs = LValue.access_array (get_ptr array) offset in
        debug_log_assignment ~env debug idcs array Ops.Arg2 value llv;
        Block.assign !current_block lhs value
    | Zero_out array ->
        if Hashtbl.mem info.arrays array then
          zero_out ctx !current_block @@ Hashtbl.find_exn info.arrays array
        else
          let tn = Low_level.(get_node info.traced_store array) in
          assert tn.zero_initialized (* The initialization is emitted by get_array. *)
    | Set_local (id, llv) ->
        let lhs, num_typ, is_double = Map.find_exn !locals id in
        let value = loop_float ~name ~env ~num_typ ~is_double llv in
        Block.assign !current_block lhs value
    | Comment c -> log_comment c
    | Staged_compilation exp -> exp ()
  and loop_float ~name ~env ~num_typ ~is_double v_code =
    let loop = loop_float ~name ~env ~num_typ ~is_double in
    match v_code with
    | Local_scope { id = { scope_id = i; _ } as id; prec; body; orig_indices = _ } ->
        let typ = Type.get ctx @@ prec_to_kind prec in
        (* Scope ids can be non-unique due to inlining. *)
        let v_name = Int.("v" ^ to_string i ^ "_" ^ get_uid ()) in
        let lvalue = Function.local func typ v_name in
        (* Arrays are initialized to 0 by default. However, there is typically an explicit
           initialization for virtual nodes. *)
        Block.assign !current_block lvalue @@ RValue.zero ctx typ;
        let old_locals = !locals in
        locals := Map.update !locals id ~f:(fun _ -> (lvalue, typ, prec_is_double prec));
        loop_proc ~env ~name:(name ^ "_at_" ^ v_name) body;
        debug_locals := Map.update !debug_locals id ~f:(fun _ -> (lvalue, typ, prec_is_double prec));
        locals := old_locals;
        RValue.lvalue lvalue
    | Get_local id ->
        let lvalue, _typ, _local_is_double = Map.find_exn !locals id in
        (* FIXME(194): Convert according to _typ ?= num_typ. *)
        RValue.lvalue lvalue
    | Get_global (C_function f_name, None) ->
        (* TODO: this is too limiting. *)
        let f = Function.builtin ctx f_name in
        RValue.call ctx f []
    | Get_global (External_unsafe { ptr; dims = (lazy dims); prec }, Some idcs) ->
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims in
        let typ = Type.get ctx @@ prec_to_kind prec in
        let ptr = RValue.ptr ctx (Type.pointer typ) ptr in
        (* FIXME(194): Convert according to typ ?= num_typ. *)
        RValue.lvalue @@ LValue.access_array ptr offset
    | Get_global _ -> failwith "NOT IMPLEMENTED YET"
    | Get (array, idcs) ->
        let array = get_array info array in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:array.dims in
        (* FIXME(194): Convert according to array.typ ?= num_typ. *)
        RValue.lvalue @@ LValue.access_array (get_ptr array) offset
    | Embed_index (Fixed_idx i) -> RValue.cast ctx (RValue.int ctx num_typ i) num_typ
    | Embed_index (Iterator s) -> (
        try RValue.cast ctx (Map.find_exn env s) num_typ
        with e ->
          Stdlib.Format.eprintf "exec_as_gccjit: missing index %a@ among environment keys:@ %a\n%!"
            Sexp.pp_hum
            ([%sexp_of: Indexing.symbol] s)
            Sexp.pp_hum
            ([%sexp_of: Indexing.symbol list] @@ Map.keys env);
          raise e)
    | Binop (Arg2, _, c2) -> loop c2
    | Binop (Arg1, c1, _) -> loop c1
    | Binop (op, c1, c2) -> loop_binop op ~num_typ ~is_double ~v1:(loop c1) ~v2:(loop c2)
    | Unop (Identity, c) -> loop c
    | Unop (Relu, c) ->
        (* FIXME: don't recompute c *)
        let cmp = cast_bool num_typ @@ RValue.comparison ctx Lt (RValue.zero ctx num_typ) @@ loop c in
        RValue.binary_op ctx Mult num_typ cmp @@ loop c
    | Constant v -> RValue.double ctx num_typ v
  and jit_for_loop ~env key ~from_ ~to_ body =
    let open Gccjit in
    let i = Indexing.symbol_ident key in
    let index = Function.local func c_index i in
    let env = Map.add_exn ~key ~data:(RValue.lvalue index) env in
    let b_loop_cond = Block.create ~name:("loop_cond_" ^ i) func in
    let b_loop_body = Block.create ~name:("loop_body_" ^ i) func in
    let b_after_loop = Block.create ~name:("after_loop_" ^ i) func in
    Block.assign !current_block index (RValue.int ctx c_index from_);
    Block.jump !current_block b_loop_cond;
    let guard = RValue.comparison ctx Gt (RValue.lvalue index) (RValue.int ctx c_index to_) in
    Block.cond_jump b_loop_cond guard b_after_loop (* on true *) b_loop_body (* on false *);
    current_block := b_loop_body;
    loop_proc ~env ~name body;
    Block.assign_op !current_block index Plus (RValue.one ctx c_index);
    Block.jump !current_block b_loop_cond;
    current_block := b_after_loop
  in

  loop_proc ~name ~env body;
  !current_block

let%track_sexp jit_func ~name ~log_file_name (context : context) ctx bindings (traced_store, proc) : ctx_info
    =
  let open Gccjit in
  let c_index = Type.get ctx Type.Int in
  let fkind = Function.Exported in
  let symbols = Indexing.bound_symbols bindings in
  let static_indices =
    List.map symbols ~f:(fun { static_symbol; _ } ->
        Param.create ctx c_index @@ Indexing.symbol_ident static_symbol)
  in
  let func = Function.create ctx fkind (Type.get ctx Void) name static_indices in
  let env =
    Map.of_alist_exn (module Indexing.Symbol)
    @@ List.mapi symbols ~f:(fun pos { Indexing.static_symbol; _ } ->
           (static_symbol, RValue.param @@ Function.param func pos))
  in
  let init_block = Block.create ~name:("init_" ^ name) func in
  let log_file () =
    let file_ptr = Type.(get ctx File_ptr) in
    let c_str = Type.(get ctx Const_char_ptr) in
    let log_file = Function.local func file_ptr "log_file" in
    let fopen =
      Function.create ctx Imported file_ptr "fopen"
        [ Param.create ctx c_str "filename"; Param.create ctx c_str "mode" ]
    in
    Block.assign init_block log_file
    @@ RValue.call ctx fopen [ RValue.string_literal ctx log_file_name; RValue.string_literal ctx "w" ];
    RValue.lvalue log_file
  in
  let log_file = if Utils.settings.debug_log_jitted then Some (log_file ()) else None in
  let main_block = Block.create ~name func in
  let ctx_info =
    { ctx; func; traced_store; init_block; ctx_arrays = context.arrays; arrays = Hashtbl.create (module Tn) }
  in
  let after_proc = jit_code ~name ~log_file ~env ctx_info main_block proc in
  (match log_file with
  | Some f ->
      (* FIXME: should be Imported? *)
      let file_ptr = Type.(get ctx File_ptr) in
      let fclose =
        Function.create ctx Imported Type.(get ctx Type.Void_ptr) "fclose" [ Param.create ctx file_ptr "f" ]
      in
      Block.eval after_proc @@ RValue.call ctx fclose [ f ]
  | None -> ());
  Block.jump init_block main_block;
  Block.return_void after_proc;
  (if Utils.settings.output_debug_files_in_run_directory then
     let f_name = name ^ "-gccjit-debug.c" in
     Context.dump_to_file ctx ~update_locs:true f_name);
  ctx_info

let header_sep =
  let open Re in
  compile (seq [ str " "; opt any; str "="; str " " ])

let%track_sexp jit (old_context : context) ~(name : string) bindings
    (compiled : Low_level.traced_store * Low_level.t) : context * _ * _ =
  let open Gccjit in
  if Option.is_none !root_ctx then initialize ();
  let ctx = Context.create_child @@ Option.value_exn !root_ctx in
  Context.set_option ctx Context.Optimization_level !optimization_level;
  (*
  if Utils.settings.with_debug && Utils.settings.output_debug_files_in_run_directory then (
    Context.set_option ctx Context.Keep_intermediates true;
    Context.set_option ctx Context.Dump_everything true);
  *)
  let log_file_name = [%string "debug-%{old_context.label}-%{name}.log"] in
  let ctx_info = jit_func ~name ~log_file_name old_context ctx bindings compiled in
  let result = Context.compile ctx in
  let context = { label = old_context.label; arrays = ctx_info.ctx_arrays; result = Some result } in
  let%diagn_sexp run_variadic =
    let rec link : 'a. 'a Indexing.bindings -> 'a Ctypes.fn -> 'a Indexing.variadic =
     fun (type b) (bs : b Indexing.bindings) (cs : b Ctypes.fn) ->
      match bs with
      | Empty -> Indexing.Result (Result.code result name Ctypes.(void @-> cs))
      | Bind (_, more) -> Param (ref 0, link more Ctypes.(int @-> cs))
    in
    link bindings Ctypes.(returning void)
  in
  let%diagn_rt_sexp run () =
    let module Debug_runtime = (val _debug_runtime) in
    [%log_result "gccjit-run", old_context.label, name];
    Indexing.apply run_variadic;
    if Utils.settings.debug_log_jitted then
      let rec loop = function
        | [] -> ()
        | line :: more when String.is_empty line -> loop more
        | comment :: more when String.is_prefix comment ~prefix:"COMMENT: " ->
            [%log String.chop_prefix_exn ~prefix:"COMMENT: " comment];
            loop more
        | source :: trace :: more when String.is_prefix source ~prefix:"# " ->
            (let source =
               String.concat ~sep:"\n" @@ String.split ~on:'$' @@ String.chop_prefix_exn ~prefix:"# " source
             in
             match Utils.split_with_seps header_sep trace with
             | [] | [ "" ] -> [%log source]
             | header1 :: assign1 :: header2 :: body ->
                 let header = String.concat [ header1; assign1; header2 ] in
                 let body = String.concat body in
                 let message = Sexp.(List [ Atom header; Atom source; Atom body ]) in
                 [%log (message : Sexp.t)]
             | _ -> [%log source, trace]);
            loop more
        | line :: more ->
            [%log line];
            loop more
      in
      loop (Stdio.In_channel.read_lines log_file_name)
  in
  Context.release ctx;
  (context, Indexing.jitted_bindings bindings run_variadic, run)

let from_host (context : context) la =
  match Map.find context.arrays la with
  | None -> false
  | Some c_arr -> (
      match la.Tn.array with
      | (lazy (Some h_arr)) ->
          Ndarray.map2 { f2 = Ndarray.A.blit } h_arr c_arr;
          true
      | (lazy None) -> false)

let to_host (context : context) la =
  match Map.find context.arrays la with
  | None -> false
  | Some c_arr -> (
      match la.Tn.array with
      | (lazy (Some h_arr)) ->
          Ndarray.map2 { f2 = Ndarray.A.blit } c_arr h_arr;
          true
      | _ -> false)

let%track_sexp merge_from_global ?(name_suffix = "") (context : context) ~dst ~accum ~src bindings =
  let body idcs =
    Low_level.(
      Set
        {
          array = dst;
          idcs;
          llv =
            Binop
              ( accum,
                Get (dst, idcs),
                Get_global (External_unsafe { ptr = src; prec = dst.Tn.prec; dims = dst.dims }, Some idcs) );
          debug = "";
        })
  in
  let llc = Low_level.loop_over_dims (Lazy.force dst.dims) ~body in
  let name = [%string "merge_into_%{Tn.name dst}_%{name_suffix}"] in
  jit context ~name bindings (Low_level.compile_proc ~name [] llc)

let%track_sexp merge ?name_suffix la ~dst ~accum ~(src : context) bindings =
  Option.map (Map.find src.arrays la) ~f:(fun (src : Ndarray.t) ->
      merge_from_global ?name_suffix dst ~dst:la ~accum ~src:(Ndarray.get_voidptr src) bindings)

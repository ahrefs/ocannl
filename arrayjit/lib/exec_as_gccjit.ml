open Base

let optimization_level = ref 3

type mem_properties =
  | Local_only  (** The array is only needed for a local computation, is allocated on the stack. *)
  | Global  (** The array has a copy allocated per-cpu-device, in addition to existing on the host. *)
  | Constant  (** The array is read directly from the host. *)
[@@deriving sexp, equal, compare, variants]

let root_ctx = ref None

module LA = Lazy_array

type context = { arrays : Ndarray.t Map.M(LA).t; result : Gccjit.result option }

let unsafe_cleanup () =
  let open Gccjit in
  Option.iter ~f:Context.release !root_ctx;
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

let init () =
  let result = { result = None; arrays = Map.empty (module LA) } in
  Core.Gc.Expert.add_finalizer_exn result finalize;
  result

type ndarray = {
  hosted_ptr : Gccjit.rvalue option;  (** Pointer to the first value of the associated hosted [Ndarray]. *)
  global_ptr : Gccjit.rvalue option;  (** Pointer to the first value of [context.arrays]. *)
  local : Gccjit.lvalue option;  (** A local array, if any. *)
  mem : mem_properties;
  dims : int array;
  size_in_bytes : int;
  num_typ : Gccjit.type_;
      (** The type of the stored values: [short] (precision [Half]), [float] (precision [Single]),
          [double] (precision [Double]). *)
  is_double : bool;
}

type ctx_info = {
  ctx : Gccjit.context;
  func : Gccjit.function_;
  traced_store : Low_level.traced_store;
  init_block : Gccjit.block;
  mutable ctx_arrays : Ndarray.t Map.M(LA).t;
  arrays : (LA.t, ndarray) Hashtbl.t;
}

let jit_array_offset ctx ~idcs ~dims =
  let open Gccjit in
  let c_index = Type.get ctx Type.Int in
  Array.fold2_exn idcs dims ~init:(RValue.zero ctx c_index) ~f:(fun offset idx dim ->
      RValue.binary_op ctx Plus c_index idx
      @@ RValue.binary_op ctx Mult c_index offset (RValue.int ctx c_index dim))

let get_array ({ ctx; func; arrays; ctx_arrays; traced_store; init_block } as ctx_info) key : ndarray =
  let open Gccjit in
  Hashtbl.find_or_add arrays key ~default:(fun () ->
      let tn = Low_level.(get_node traced_store key) in
      let dims = Lazy.force key.dims in
      let size_in_elems = Array.fold ~init:1 ~f:( * ) dims in
      let size_in_bytes = size_in_elems * Ops.prec_in_bytes key.prec in
      let is_on_host = !(key.hosted) in
      assert (Bool.(Option.is_some (Lazy.force key.array) = is_on_host));
      let c_void_ptr = Type.(get ctx Type.Void_ptr) in
      let c_index = Type.get ctx Type.Size_t in
      let c_int = Type.get ctx Type.Int in
      (* TODO: is the complexity of introducing this function and matching on Ndarray.t needed? *)
      let array c_typ is_double arr =
        let num_typ = Type.(get ctx c_typ) in
        let get_c_ptr ba =
          RValue.ptr ctx (Type.pointer num_typ) @@ Ctypes.bigarray_start Ctypes_static.Genarray ba
        in
        let hosted_ptr = Option.map arr ~f:get_c_ptr in
        let mem = if not is_on_host then Local_only else if tn.read_only then Constant else Global in
        let arr_typ = Type.array ctx num_typ size_in_elems in
        let local = if is_local_only mem then Some (Function.local func arr_typ @@ LA.name key) else None in
        (if is_global mem && (not @@ Map.mem ctx_arrays key) then
           let data =
             Ndarray.create_array key.LA.prec ~dims @@ Constant_fill { values = [| 0. |]; strict = false }
           in
           ctx_info.ctx_arrays <- Map.add_exn ~key ~data ctx_arrays);
        let global_ptr = Option.map (Map.find ctx_info.ctx_arrays key) ~f:Ndarray.(map { f = get_c_ptr }) in
        let cast_void rv = RValue.cast ctx rv c_void_ptr in
        let backend_info = (Sexp.to_string_hum @@ sexp_of_mem_properties mem) ^ ";" in
        let comment_on ptr = Option.value ~default:"not" @@ Option.map ptr ~f:RValue.to_string in
        Block.comment init_block
          [%string
            "Array #%{key.id#Int} %{key.label}: %{backend_info} %{comment_on hosted_ptr} hosted, \
             %{comment_on global_ptr} global, %{comment_on @@ Option.map ~f:RValue.lvalue local} local."];
        if tn.zero_initialized then
          List.find_map ~f:Fn.id
            [ Option.map local ~f:(Fn.compose cast_void LValue.address); global_ptr; hosted_ptr ]
          |> Option.iter ~f:(fun rv_ptr ->
                 Block.eval init_block
                 @@ RValue.call ctx (Function.builtin ctx "memset")
                      [ rv_ptr; RValue.zero ctx c_int; RValue.int ctx c_index size_in_bytes ]);
        if not @@ String.is_substring key.backend_info ~substring:backend_info then
          key.backend_info <- key.backend_info ^ backend_info;
        { hosted_ptr; global_ptr; local; mem; dims; size_in_bytes; num_typ; is_double }
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

let jit_code ~name ~(env : Gccjit.rvalue Indexing.environment) ({ ctx; func; _ } as info) initial_block body =
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
  let locals = ref Map.Poly.empty in
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
  let log_comment c =
    if !Low_level.executor_print_comments then
       let f = Function.builtin ctx "printf" in
       Block.eval !current_block @@ RValue.call ctx f [ RValue.string_literal ctx ("\nCOMMENT: " ^ c ^ "\n") ]
    else Block.comment !current_block c
  in
  let rec debug_float ~is_double (value : Low_level.float_t) : string * 'a list =
    let loop = debug_float ~is_double in
    match value with
    | Local_scope { id; _ } ->
        (* Not printing the inlined definition: (1) code complexity; (2) don't overload the debug logs. *)
        loop @@ Get_local id
    | Get_local id ->
        let lvalue, _typ, _local_is_double = Map.find_exn !locals id in
        (* FIXME(194): Convert according to _typ ?= num_typ. *)
        ("v" ^ Int.to_string id.scope_id ^ "{=%g}", [ to_d @@ RValue.lvalue lvalue ])
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
        (RValue.to_string (get_ptr array) ^ "[%d]{=%g}", [ offset; v ])
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
  let printf = Function.builtin ctx "printf" in
  let f_ptr = Type.get ctx Type.File_ptr in
  let fflush = Function.create ctx Imported (Type.get ctx Void) "fflush" [ Param.create ctx f_ptr "f" ] in
  let c_stdout = LValue.global ctx LValue.Imported f_ptr "stdout" in
  let log_trace_assignment idcs array accum_op value v_code =
    if !Low_level.debug_verbose_trace then (
      let v_format, v_fillers = debug_float ~is_double:array.is_double v_code in
      let offset = jit_array_offset ctx ~idcs ~dims:array.dims in
      Block.eval !current_block @@ RValue.call ctx printf
      @@ RValue.string_literal ctx
           [%string
             {|TRACE: %{RValue.to_string @@ get_ptr array}[%d]{=%g} %{Ops.assign_op_C_syntax accum_op} %g = %{v_format}
|}]
         :: (to_d @@ RValue.lvalue @@ LValue.access_array (get_ptr array) offset)
         :: offset :: to_d value :: v_fillers;
      Block.eval !current_block @@ RValue.call ctx fflush [ RValue.lvalue c_stdout ])
  in
  let rec loop_proc ~(env : rvalue Indexing.environment) ~name (body : Low_level.t) : unit =
    let loop = loop_proc ~env ~name in
    match body with
    | Noop -> ()
    | Low_level.Seq (c1, c2) ->
        loop c1;
        loop c2
    | For_loop { index; from_; to_; body; trace_it = _ } -> jit_for_loop ~env index ~from_ ~to_ body
    | Set (_, _, Binop (Arg2, Get (_, _), _)) -> assert false
    | Set (array, idcs, Binop (op, Get (array2, idcs2), c2))
      when LA.equal array array2 && [%equal: Indexing.axis_index array] idcs idcs2 && is_builtin_op op ->
        (* FIXME: maybe it's not worth it? *)
        let array = get_array info array in
        let value = loop_float ~name ~env ~num_typ:array.num_typ ~is_double:array.is_double c2 in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:array.dims in
        let lhs = LValue.access_array (get_ptr array) offset in
        log_trace_assignment idcs array op value c2;
        Block.assign_op !current_block lhs (builtin_op op) value
    | Set (array, idcs, v_code) ->
        let array = get_array info array in
        let value = loop_float ~name ~env ~num_typ:array.num_typ ~is_double:array.is_double v_code in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:array.dims in
        let lhs = LValue.access_array (get_ptr array) offset in
        log_trace_assignment idcs array Ops.Arg2 value v_code;
        Block.assign !current_block lhs value
    | Zero_out array ->
        if Hashtbl.mem info.arrays array then
          failwith
            ("exec_as_gccjit: Non-initialization zeroing-out NOT IMPLEMENTED YET: " ^ Sexp.to_string_hum
            @@ [%sexp_of: LA.t] array);
        let tn = Low_level.(get_node info.traced_store array) in
        assert tn.zero_initialized
        (* The initialization will be emitted by get_array. *)
    | Set_local (id, value) ->
        let lhs, num_typ, is_double = Map.find_exn !locals id in
        let value = loop_float ~name ~env ~num_typ ~is_double value in
        Block.assign !current_block lhs value
    | Comment c -> log_comment c
    | Staged_compilation exp -> exp ()
  and loop_float ~name ~env ~num_typ ~is_double v_code : rvalue =
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
    | Embed_index (Fixed_idx i) -> RValue.int ctx num_typ i
    | Embed_index (Iterator s) -> (
        try Map.find_exn env s
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
  and jit_for_loop ~env key ~from_ ~to_ body : unit =
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

let jit_func ~name (context : context) ctx bindings (traced_store, proc) =
  let open Gccjit in
  let c_index = Type.get ctx Type.Int in
  let fkind = Function.Exported in
  let bindings = Indexing.assoc_of_bindings bindings in
  let static_indices =
    List.map bindings ~f:(fun ({ static_symbol; _ }, _) ->
        Param.create ctx c_index @@ Indexing.symbol_ident static_symbol)
  in
  let func = Function.create ctx fkind (Type.get ctx Void) name static_indices in
  let env =
    Map.of_alist_exn (module Indexing.Symbol)
    @@ List.mapi bindings ~f:(fun pos ({ Indexing.static_symbol; _ }, _) ->
           (static_symbol, RValue.param @@ Function.param func pos))
  in
  let init_block = Block.create ~name:("init_" ^ name) func in
  let main_block = Block.create ~name func in
  let ctx_info =
    { ctx; func; traced_store; init_block; ctx_arrays = context.arrays; arrays = Hashtbl.Poly.create () }
  in
  let after_proc = jit_code ~name ~env ctx_info main_block proc in
  Block.jump init_block main_block;
  Block.return_void after_proc;
  (if !Low_level.with_debug then
     let suf = "-gccjit-debug.c" in
     let f_name =
       if !Low_level.keep_files_in_run_directory then name ^ suf
       else Stdlib.Filename.temp_file (name ^ "-") suf
     in
     Context.dump_to_file ctx ~update_locs:true f_name);
  ctx_info

type jitted = { context : context; run : unit -> unit; bindings : unit Indexing.bindings }

let jit old_context ~name ?verbose:_ bindings compiled =
  (* TODO: add verbose logs *)
  let open Gccjit in
  if Option.is_none !root_ctx then initialize ();
  let ctx = Context.create_child @@ Option.value_exn !root_ctx in
  Context.set_option ctx Context.Optimization_level !optimization_level;
  (*
  if !Low_level.with_debug && !Low_level.keep_files_in_run_directory then (
    Context.set_option ctx Context.Keep_intermediates true;
    Context.set_option ctx Context.Dump_everything true);
  *)
  let ctx_info = jit_func ~name old_context ctx bindings compiled in
  let result = Context.compile ctx in
  let context = { arrays = ctx_info.ctx_arrays; result = Some result } in
  let run : unit -> unit =
    let rec link : 'a. 'a Indexing.bindings -> 'a Ctypes.fn -> 'a Indexing.variadic =
     fun (type b) (bs : b Indexing.bindings) (cs : b Ctypes.fn) ->
      match bs with
      | Empty -> Indexing.Result (Result.code result name Ctypes.(void @-> cs))
      | Bind ({ static_range; static_symbol }, i, more) ->
          if !i < 0 then
            raise
            @@ Ndarray.User_error
                 [%string
                   "Exec_as_gccjit: static index %{Indexing.symbol_ident static_symbol} is negative: \
                    %{!i#Int}"];
          Option.iter static_range ~f:(fun upto ->
              if !i >= upto then
                raise
                @@ Ndarray.User_error
                     [%string
                       "Exec_as_gccjit: static index %{Indexing.symbol_ident static_symbol} is too big: \
                        %{upto#Int}"]);
          Param (i, link more Ctypes.(int @-> cs))
    in
    let runf = link bindings Ctypes.(returning void) in
    fun () -> Indexing.apply runf
  in
  Context.release ctx;
  { context; bindings; run }

let from_host (context : context) la =
  match (la.LA.array, Map.find context.arrays la) with
  | (lazy (Some h_arr)), Some c_arr ->
      Ndarray.map2 { f2 = Ndarray.A.blit } h_arr c_arr;
      true
  | _ -> false

let to_host (context : context) la =
  match (la.LA.array, Map.find context.arrays la) with
  | (lazy (Some h_arr)), Some c_arr ->
      Ndarray.map2 { f2 = Ndarray.A.blit } c_arr h_arr;
      true
  | _ -> false

let merge_from_global ?(name_suffix = "") (context : context) ~dst ~accum ~src =
  let body idcs =
    Low_level.(
      Set
        ( dst,
          idcs,
          Binop
            ( accum,
              Get (dst, idcs),
              Get_global (External_unsafe { ptr = src; prec = dst.LA.prec; dims = dst.dims }, Some idcs) ) ))
  in
  let llc = Low_level.loop_over_dims (Lazy.force dst.dims) ~body in
  let name = [%string "merge_into_%{dst.Lazy_array.id#Int}%{name_suffix}"] in
  jit context ~name ~verbose:false Indexing.Empty (Low_level.compile_proc ~name llc)

let merge ?name_suffix la ~dst ~accum ~(src : context) =
  Option.map (Map.find src.arrays la) ~f:(fun src ->
      merge_from_global ?name_suffix dst ~dst:la ~accum ~src:(Ndarray.get_voidptr src))

open Base
module Lazy = Utils.Lazy
module Debug_runtime = Utils.Debug_runtime

[%%global_debug_log_level Nothing]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

let optimization_level = ref 3

type mem_properties =
  | Local_only  (** The array is only needed for a local computation, is allocated on the stack. *)
  | From_context  (** The array has a copy allocated per-cpu-device, may or may not exist on the host. *)
  | Constant_from_host  (** The array is read directly from the host. *)
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

type tn_info = {
  tn : Tn.t;  (** The original array. *)
  mutable ptr : (Gccjit.rvalue[@sexp.opaque]) Lazy.t;
      (** Pointer to the first value of the associated array.
          - if [mem = Constant_from_host], the pointer to the first element of the hosted [Ndarray],
          - if [mem = From_context], either a pointer to [Ndarray] from [context.arrays] when [~shared:false],
            or the function parameter when [~shared:true],
          - if [mem = Local_only], the address of the on-the-stack array. *)
  mem : mem_properties;
  dims : int array;
  size_in_bytes : int;
  num_typ : (Gccjit.type_[@sexp.opaque]);
      (** The type of the stored values: [short] (precision [Half]), [float] (precision [Single]), [double]
          (precision [Double]). *)
  prec : Ops.prec;
}
[@@deriving sexp_of]

type info = {
  ctx : (Gccjit.context[@sexp.opaque]);
  func : (Gccjit.function_[@sexp.opaque]);
  traced_store : (Low_level.traced_store[@sexp.opaque]);
  init_block : (Gccjit.block[@sexp.opaque]);
  nodes : (Tn.t, tn_info) Hashtbl.t;
}
[@@deriving sexp_of]

type param_source = Log_file_name | Param_ptr of Tn.t | Static_idx of Indexing.static_symbol
[@@deriving sexp_of]

type routine = {
  info : info;
  bindings : Indexing.unit_bindings;
  name : string;
  result : (Gccjit.result[@sexp.opaque]);
  opt_ctx_arrays : Ndarray.t Map.M(Tn).t option;
  params : param_source list;
}
[@@deriving sexp_of]

type code =
  | Postponed of {
      compiled : Low_level.traced_store * Low_level.t;
      bindings : Indexing.unit_bindings;
      name : string;
    }
  | Compiled of routine
[@@deriving sexp_of]

type gccjit_param = Gccjit.param
type gccjit_lvalue = Gccjit.lvalue
type gccjit_rvalue = Gccjit.rvalue

let sexp_of_gccjit_param p = Sexp.Atom (Gccjit.Param.to_string p)
let sexp_of_gccjit_lvalue v = Sexp.Atom (Gccjit.LValue.to_string v)
let sexp_of_gccjit_rvalue v = Sexp.Atom (Gccjit.RValue.to_string v)

type ctx_nodes =
  | Ctx_arrays of Ndarray.t Map.M(Tn).t ref
  | Param_ptrs of (gccjit_param * param_source) list ref
[@@deriving sexp_of]

let jit_array_offset ctx ~idcs ~dims =
  let open Gccjit in
  let c_index = Type.get ctx Type.Int in
  Array.fold2_exn idcs dims ~init:(RValue.zero ctx c_index) ~f:(fun offset idx dim ->
      RValue.binary_op ctx Plus c_index idx
      @@ RValue.binary_op ctx Mult c_index offset (RValue.int ctx c_index dim))

let zero_out ctx block node =
  let open Gccjit in
  let c_index = Type.get ctx Type.Size_t in
  let c_int = Type.get ctx Type.Int in
  Block.eval block
  @@ RValue.call ctx (Function.builtin ctx "memset")
       [ Lazy.force node.ptr; RValue.zero ctx c_int; RValue.int ctx c_index node.size_in_bytes ]

let get_c_ptr ctx num_typ ba =
  Gccjit.(RValue.ptr ctx (Type.pointer num_typ) @@ Ctypes.bigarray_start Ctypes_static.Genarray ba)

let prepare_node ~debug_log_zero_out ctx nodes traced_store ctx_nodes initializations (key : Tn.t) =
  let open Gccjit in
  Hashtbl.update nodes key ~f:(function
    | Some old -> old
    | None ->
        let traced = Low_level.(get_node traced_store key) in
        let dims = Lazy.force key.dims in
        let size_in_elems = Array.fold ~init:1 ~f:( * ) dims in
        let size_in_bytes = size_in_elems * Ops.prec_in_bytes key.prec in
        let is_on_host = Tn.is_hosted_force key 33 in
        let is_materialized = Tn.is_materialized_force key 331 in
        let is_constant = Tn.is_hosted_force ~specifically:Constant key 332 in
        assert (Bool.(Option.is_some (Lazy.force key.array) = is_on_host));
        let prec = key.prec in
        let c_typ = Ops.gcc_typ_of_prec prec in
        let num_typ = Type.(get ctx c_typ) in
        let ptr_typ = Type.pointer num_typ in
        let mem =
          if not is_materialized then Local_only
          else if is_constant && traced.read_only then Constant_from_host
          else From_context
        in
        let name = Tn.name key in
        let ptr =
          match (mem, ctx_nodes) with
          | From_context, Ctx_arrays ctx_arrays -> (
              match Map.find !ctx_arrays key with
              | None ->
                  let data =
                    Ndarray.create_array key.Tn.prec ~dims
                    @@ Constant_fill { values = [| 0. |]; strict = false }
                  in
                  ctx_arrays := Map.add_exn !ctx_arrays ~key ~data;
                  let f arr = get_c_ptr ctx num_typ arr in
                  Lazy.from_val @@ Ndarray.(map { f } data)
              | Some data ->
                  let f arr = get_c_ptr ctx num_typ arr in
                  Lazy.from_val @@ Ndarray.(map { f } data))
          | From_context, Param_ptrs ptrs ->
              let p = Param.create ctx ptr_typ name in
              ptrs := (p, Param_ptr key) :: !ptrs;
              Lazy.from_val (RValue.param p)
          | Constant_from_host, _ -> (
              let addr arr = Lazy.from_val @@ get_c_ptr ctx num_typ arr in
              match Lazy.force key.array with
              | Some (Byte_nd arr) -> addr arr
              | Some (Half_nd arr) -> addr arr
              | Some (Single_nd arr) -> addr arr
              | Some (Double_nd arr) -> addr arr
              | None -> assert false)
          | Local_only, _ ->
              let arr_typ = Type.array ctx num_typ size_in_elems in
              let v = ref None in
              let initialize _init_block func = v := Some (Function.local func arr_typ name) in
              initializations := initialize :: !initializations;
              (* The array is the pointer but the address of the array is the same pointer. *)
              lazy (RValue.cast ctx (LValue.address @@ Option.value_exn !v) ptr_typ)
        in
        let result = { tn = key; ptr; mem; dims; size_in_bytes; num_typ; prec } in
        let backend_info = sexp_of_mem_properties mem in
        let initialize init_block _func =
          Block.comment init_block
            [%string
              "Array #%{key.id#Int} %{Tn.label key}: %{Sexp.to_string_hum @@ backend_info}; ptr: \
               %{Sexp.to_string_hum @@ sexp_of_gccjit_rvalue @@ Lazy.force ptr}."];
          if traced.zero_initialized then (
            debug_log_zero_out init_block result;
            zero_out ctx init_block result)
        in
        initializations := initialize :: !initializations;
        if not @@ Utils.sexp_mem ~elem:backend_info key.backend_info then
          key.backend_info <- Utils.sexp_append ~elem:backend_info key.backend_info;
        result)

let prec_to_kind prec =
  let open Gccjit in
  match prec with
  | Ops.Void_prec -> Type.Void
  | Byte_prec _ -> Type.Unsigned_char
  | Half_prec _ -> (* FIXME: *) Type.Unsigned_short
  | Single_prec _ -> Type.Float
  | Double_prec _ -> Type.Double

let is_builtin_op = function Ops.Add | Sub | Mul | Div -> true | ToPowOf | Relu_gate | Arg2 | Arg1 -> false

let builtin_op = function
  | Ops.Add -> Gccjit.Plus
  | Sub -> Gccjit.Minus
  | Mul -> Gccjit.Mult
  | Div -> Gccjit.Divide
  | ToPowOf | Relu_gate | Arg2 | Arg1 -> invalid_arg "Exec_as_gccjit.builtin_op: not a builtin"

let node_debug_name node =
  let memloc =
    if Utils.settings.debug_memory_locations && Lazy.is_val node.ptr then
      "@" ^ Gccjit.RValue.to_string (Lazy.force node.ptr)
    else ""
  in
  Tn.name node.tn ^ memloc

let debug_log_zero_out ctx log_functions block node =
  let open Gccjit in
  let c_index = Type.get ctx Type.Int in
  match Lazy.force log_functions with
  | Some (lf, pf, ff) ->
      let c_double = Type.get ctx Type.Double in
      let to_d v = RValue.cast ctx v c_double in
      Block.eval block @@ RValue.call ctx pf
      @@ lf
         :: RValue.string_literal ctx
              [%string {|memset_zero(%{node_debug_name node}) where before first element = %g
|}]
         :: [ to_d @@ RValue.lvalue @@ LValue.access_array (Lazy.force node.ptr) @@ RValue.zero ctx c_index ];
      Block.eval block @@ RValue.call ctx ff [ lf ]
  | _ -> ()

let debug_log_index ctx log_functions =
  let open Gccjit in
  match log_functions with
  | Some (lf, pf, ff) ->
      fun block i index ->
        Block.eval block @@ RValue.call ctx pf
        @@ (lf :: RValue.string_literal ctx [%string {|index %{i} = %d
|}] :: [ index ]);
        Block.eval block @@ RValue.call ctx ff [ lf ]
  | _ -> fun _block _i _index -> ()

let compile_main ~name ~log_functions ~env ({ ctx; nodes; _ } as info) func initial_block (body : Low_level.t)
    =
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
  (* Source of unique identifiers. E.g. local scope ids can be non-unique due to inlining. We also need unique
     ids for computation ordering lvalues. *)
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
  let loop_binop op ~num_typ prec ~v1 ~v2 =
    match (op, prec) with
    | _, Ops.Void_prec -> raise @@ Utils.User_error "gccjit_backend: binary operation on Void_prec"
    | Ops.Add, _ -> RValue.binary_op ctx Plus num_typ v1 v2
    | Sub, _ -> RValue.binary_op ctx Minus num_typ v1 v2
    | Mul, _ -> RValue.binary_op ctx Mult num_typ v1 v2
    | Div, _ -> RValue.binary_op ctx Divide num_typ v1 v2
    | ToPowOf, Double_prec _ ->
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "pow") [ to_d v1; to_d v2 ]) num_typ
    | ToPowOf, (Single_prec _ | Half_prec _ (* FIXME: *)) ->
        let base = RValue.cast ctx v1 c_float in
        let expon = RValue.cast ctx v2 c_float in
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "powf") [ base; expon ]) num_typ
    | ToPowOf, Byte_prec _ -> raise @@ Utils.User_error "gccjit_backend: Byte_prec does not support ToPowOf"
    | Relu_gate, _ ->
        let cmp = cast_bool num_typ @@ RValue.comparison ctx Lt (RValue.zero ctx num_typ) v1 in
        RValue.binary_op ctx Mult num_typ cmp @@ v2
    | Arg2, _ -> v2
    | Arg1, _ -> v1
  in
  let log_comment c =
    match log_functions with
    | Some (lf, pf, ff) ->
        Block.eval !current_block
        @@ RValue.call ctx pf [ lf; RValue.string_literal ctx ("\nCOMMENT: " ^ c ^ "\n") ];
        Block.eval !current_block @@ RValue.call ctx ff [ lf ]
    | _ -> Block.comment !current_block c
  in
  let get_node = Hashtbl.find_exn nodes in
  let rec debug_float ~env prec value =
    let loop = debug_float ~env prec in
    match value with
    | Low_level.Local_scope { id; _ } ->
        (* Not printing the inlined definition: (1) code complexity; (2) don't overload the debug logs. *)
        loop @@ Get_local id
    | Get_local id ->
        let lvalue = Map.find_exn !debug_locals id in
        (LValue.to_string lvalue ^ "{=%g}", [ to_d @@ RValue.lvalue lvalue ])
    | Get_global (C_function f_name, None) -> ("<calls " ^ f_name ^ ">", [])
    | Get_global (External_unsafe { ptr; dims = (lazy dims); prec }, Some idcs) ->
        let idcs = lookup env idcs in
        let typ = Type.get ctx @@ prec_to_kind prec in
        let ptr = RValue.ptr ctx (Type.pointer typ) ptr in
        let offset = jit_array_offset ctx ~idcs ~dims in
        let v = to_d @@ RValue.lvalue @@ LValue.access_array ptr offset in
        ("external " ^ RValue.to_string ptr ^ "[%d]{=%g}", [ offset; v ])
    | Get_global (External_unsafe _, None) -> assert false
    | Get_global (C_function _, Some _) -> failwith "gccjit_backend: FFI with parameters NOT IMPLEMENTED YET"
    | Get (tn, idcs) ->
        let node = get_node tn in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:node.dims in
        let v = to_d @@ RValue.lvalue @@ LValue.access_array (Lazy.force node.ptr) offset in
        (node_debug_name node ^ "[%d]{=%g}", [ offset; v ])
    | Constant c -> (Float.to_string c, [])
    | Embed_index (Fixed_idx i) -> (Int.to_string i, [])
    | Embed_index (Iterator s) -> (Indexing.symbol_ident s ^ "{=%d}", [ Map.find_exn env s ])
    | Binop (Arg1, v1, _v2) -> loop v1
    | Binop (Arg2, _v1, v2) -> loop v2
    | Binop (op, v1, v2) ->
        let prefix, infix, postfix = Ops.binop_C_syntax prec op in
        let v1, fillers1 = loop v1 in
        let v2, fillers2 = loop v2 in
        (String.concat [ prefix; v1; infix; " "; v2; postfix ], fillers1 @ fillers2)
    | Unop (Identity, v) -> loop v
    | Unop (Relu, v) ->
        let v, fillers = loop v in
        (String.concat [ "("; v; " > 0.0 ? "; v; " : 0.0)" ], fillers @ fillers)
  in
  let debug_log_assignment ~env debug idcs node accum_op value v_code =
    match log_functions with
    | Some (lf, pf, ff) ->
        let v_format, v_fillers = debug_float ~env node.prec v_code in
        let offset = jit_array_offset ctx ~idcs ~dims:node.dims in
        let debug_line = "# " ^ String.substr_replace_all debug ~pattern:"\n" ~with_:"$" ^ "\n" in
        Block.eval !current_block @@ RValue.call ctx pf @@ [ lf; RValue.string_literal ctx debug_line ];
        Block.eval !current_block @@ RValue.call ctx pf
        @@ lf
           :: RValue.string_literal ctx
                [%string
                  {|%{node_debug_name node}[%d]{=%g} %{Ops.assign_op_C_syntax accum_op} %g = %{v_format}
|}]
           :: (to_d @@ RValue.lvalue @@ LValue.access_array (Lazy.force node.ptr) offset)
           :: offset :: to_d value :: v_fillers;
        Block.eval !current_block @@ RValue.call ctx ff [ lf ]
    | _ -> ()
  in
  let debug_log_index = debug_log_index ctx log_functions in
  let rec loop_proc ~toplevel ~env ~name (body : Low_level.t) : unit =
    let loop = loop_proc ~toplevel ~env ~name in
    match body with
    | Noop -> ()
    | Low_level.Seq (c1, c2) ->
        loop c1;
        loop c2
    | For_loop { index; from_; to_; body; trace_it = _ } ->
        loop_for_loop ~toplevel ~env index ~from_ ~to_ body
    | Set { llv = Binop (Arg2, Get (_, _), _); _ } -> assert false
    | Set { tn; idcs; llv = Binop (op, Get (tn2, idcs2), c2); debug }
      when Tn.equal tn tn2 && [%equal: Indexing.axis_index array] idcs idcs2 && is_builtin_op op ->
        (* FIXME: maybe it's not worth it? *)
        let node = get_node tn in
        let value = loop_float ~name ~env ~num_typ:node.num_typ node.prec c2 in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:node.dims in
        let lhs = LValue.access_array (Lazy.force node.ptr) offset in
        debug_log_assignment ~env debug idcs node op value c2;
        Block.assign_op !current_block lhs (builtin_op op) value
    | Set { tn; idcs; llv; debug } ->
        let node = get_node tn in
        let value = loop_float ~name ~env ~num_typ:node.num_typ node.prec llv in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:node.dims in
        let lhs = LValue.access_array (Lazy.force node.ptr) offset in
        debug_log_assignment ~env debug idcs node Ops.Arg2 value llv;
        Block.assign !current_block lhs value
    | Zero_out array ->
        if Hashtbl.mem info.nodes array then (
          let array = Hashtbl.find_exn info.nodes array in
          debug_log_zero_out ctx (lazy log_functions) !current_block array;
          zero_out ctx !current_block array)
        else
          let tn = Low_level.(get_node info.traced_store array) in
          assert tn.zero_initialized (* The initialization is emitted by prepare_nodes. *)
    | Set_local (id, llv) ->
        let lhs = Map.find_exn !locals id in
        let local_typ = Ops.gcc_typ_of_prec id.tn.prec in
        let num_typ = Type.get ctx local_typ in
        let value = loop_float ~name ~env ~num_typ id.tn.prec llv in
        Block.assign !current_block lhs value
    | Comment c -> log_comment c
    | Staged_compilation exp -> exp ()
  and loop_float ~name ~env ~num_typ prec v_code =
    let loop = loop_float ~name ~env ~num_typ prec in
    match v_code with
    | Local_scope { id = { scope_id = i; tn = { prec; _ } } as id; body; orig_indices = _ } ->
        let typ = Type.get ctx @@ prec_to_kind prec in
        (* Scope ids can be non-unique due to inlining. *)
        let v_name = Int.("v" ^ to_string i ^ "_" ^ get_uid ()) in
        let lvalue = Function.local func typ v_name in
        (* Arrays are initialized to 0 by default. However, there is typically an explicit initialization for
           virtual nodes. *)
        Block.assign !current_block lvalue @@ RValue.zero ctx typ;
        let old_locals = !locals in
        locals := Map.update !locals id ~f:(fun _ -> lvalue);
        loop_proc ~toplevel:false ~env ~name:(name ^ "_at_" ^ v_name) body;
        debug_locals := Map.update !debug_locals id ~f:(fun _ -> lvalue);
        locals := old_locals;
        RValue.lvalue lvalue
    | Get_local id ->
        let lvalue = Map.find_exn !locals id in
        let rvalue = RValue.lvalue lvalue in
        let local_typ = Ops.gcc_typ_of_prec id.tn.prec in
        let num_typ = Type.get ctx local_typ in
        if not @@ Ops.equal_prec prec id.tn.prec then RValue.cast ctx rvalue num_typ else rvalue
    | Get_global (C_function f_name, None) ->
        (* TODO: this is too limiting. *)
        let f = Function.builtin ctx f_name in
        RValue.call ctx f []
    | Get_global (External_unsafe { ptr; dims = (lazy dims); prec = local_prec }, Some idcs) ->
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims in
        let typ = Type.get ctx @@ prec_to_kind prec in
        let ptr = RValue.ptr ctx (Type.pointer typ) ptr in
        let rvalue = RValue.lvalue @@ LValue.access_array ptr offset in
        let local_typ = Ops.gcc_typ_of_prec local_prec in
        let num_typ = Type.get ctx local_typ in
        if not @@ Ops.equal_prec prec local_prec then RValue.cast ctx rvalue num_typ else rvalue
    | Get_global (External_unsafe _, None) -> assert false
    | Get_global (C_function _, Some _) -> failwith "gccjit_backend: FFI with parameters NOT IMPLEMENTED YET"
    | Get (tn, idcs) ->
        let node = get_node tn in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:node.dims in
        let rvalue = RValue.lvalue @@ LValue.access_array (Lazy.force node.ptr) offset in
        let local_typ = Ops.gcc_typ_of_prec tn.prec in
        let num_typ = Type.get ctx local_typ in
        if not @@ Ops.equal_prec prec tn.prec then RValue.cast ctx rvalue num_typ else rvalue
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
    | Binop (op, c1, c2) -> loop_binop op ~num_typ prec ~v1:(loop c1) ~v2:(loop c2)
    | Unop (Identity, c) -> loop c
    | Unop (Relu, c) ->
        (* FIXME: don't recompute c *)
        let cmp = cast_bool num_typ @@ RValue.comparison ctx Lt (RValue.zero ctx num_typ) @@ loop c in
        RValue.binary_op ctx Mult num_typ cmp @@ loop c
    | Constant v -> RValue.double ctx num_typ v
  and loop_for_loop ~toplevel ~env key ~from_ ~to_ body =
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
    (* Currently we don't log local computations. *)
    if toplevel then debug_log_index !current_block i (RValue.lvalue index);
    loop_proc ~toplevel ~env ~name body;
    Block.assign_op !current_block index Plus (RValue.one ctx c_index);
    Block.jump !current_block b_loop_cond;
    current_block := b_after_loop
  in

  loop_proc ~toplevel:true ~name ~env body;
  !current_block

let prepare_nodes ctx ~log_functions nodes traced_store ctx_nodes initializations (llc : Low_level.t) =
  let debug_log_zero_out = debug_log_zero_out ctx log_functions in
  let prepare_node = prepare_node ctx ~debug_log_zero_out nodes traced_store ctx_nodes initializations in
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

let%track_sexp compile_func ~name ~opt_ctx_arrays ctx bindings (traced_store, proc) :
    info * Ndarray.t Base.Map.M(Tn).t * (gccjit_param * param_source) list =
  let open Gccjit in
  let c_index = Type.get ctx Type.Int in
  let fkind = Function.Exported in
  let c_str = Type.(get ctx Const_char_ptr) in
  let log_file_name =
    if Utils.settings.debug_log_from_routines then Some (Param.create ctx c_str "log_file_name", Log_file_name)
    else None
  in
  let symbols = Indexing.bound_symbols bindings in
  let static_indices =
    List.map symbols ~f:(fun ({ static_symbol; _ } as s) ->
        (Param.create ctx c_index @@ Indexing.symbol_ident static_symbol, Static_idx s))
  in
  let ctx_arrays : Ndarray.t Map.M(Tn).t ref =
    ref @@ Option.value opt_ctx_arrays ~default:(Map.empty (module Tn))
  in
  let params : (gccjit_param * param_source) list ref = ref (Option.to_list log_file_name @ static_indices) in
  let ctx_nodes : ctx_nodes =
    if Option.is_none opt_ctx_arrays then Param_ptrs params else Ctx_arrays ctx_arrays
  in
  let initializations = ref [] in
  let nodes = Hashtbl.create (module Tn) in
  let log_functions_ref = ref None in
  let log_functions = lazy !log_functions_ref in
  prepare_nodes ~log_functions ctx nodes traced_store ctx_nodes initializations proc;
  let params : (gccjit_param * param_source) list =
    match ctx_nodes with Param_ptrs ps -> !ps | Ctx_arrays _ -> !params
  in
  let func = Function.create ctx fkind (Type.get ctx Void) name @@ List.map ~f:fst params in
  let env =
    Map.of_alist_exn (module Indexing.Symbol)
    @@ List.map2_exn symbols static_indices ~f:(fun { Indexing.static_symbol; _ } (p_ind, _) ->
           (static_symbol, RValue.param p_ind))
  in
  let init_block = Block.create ~name:("init_" ^ name) func in
  Option.iter log_file_name ~f:(fun log_file_name ->
      let file_ptr = Type.(get ctx File_ptr) in
      let log_file = Function.local func file_ptr "log_file" in
      let fopen =
        Function.create ctx Imported file_ptr "fopen"
          [ Param.create ctx c_str "filename"; Param.create ctx c_str "mode" ]
      in
      Block.assign init_block log_file
      @@ RValue.call ctx fopen [ RValue.param @@ fst log_file_name; RValue.string_literal ctx "w" ];
      let log_file = RValue.lvalue log_file in
      let fprintf = Function.builtin ctx "fprintf" in
      let fflush =
        let f_ptr = Type.get ctx Type.File_ptr in
        Function.create ctx Imported (Type.get ctx Void) "fflush" [ Param.create ctx f_ptr "f" ]
      in
      log_functions_ref := Some (log_file, fprintf, fflush));
  let log_functions = Lazy.force log_functions in
  let debug_log_index = debug_log_index ctx log_functions in
  Map.iteri env ~f:(fun ~key:sym ~data:idx -> debug_log_index init_block (Indexing.symbol_ident sym) idx);
  (* Do initializations in the order they were scheduled. *)
  List.iter (List.rev !initializations) ~f:(fun init -> init init_block func);
  let main_block = Block.create ~name func in
  let ctx_info : info = { ctx; traced_store; init_block; func; nodes } in
  let after_proc = compile_main ~name ~log_functions ~env ctx_info func main_block proc in
  (match log_functions with
  | Some (lf, _, _) ->
      (* FIXME: should be Imported? *)
      let file_ptr = Type.(get ctx File_ptr) in
      let fclose =
        Function.create ctx Imported Type.(get ctx Type.Void_ptr) "fclose" [ Param.create ctx file_ptr "f" ]
      in
      Block.eval after_proc @@ RValue.call ctx fclose [ lf ]
  | None -> ());
  Block.jump init_block main_block;
  Block.return_void after_proc;
  (ctx_info, !ctx_arrays, params)

let header_sep =
  let open Re in
  compile (seq [ str " "; opt any; str "="; str " " ])

let%track_sexp compile ~(name : string) ~opt_ctx_arrays bindings
    (compiled : Low_level.traced_store * Low_level.t) =
  let open Gccjit in
  if Option.is_none !root_ctx then initialize ();
  let ctx = Context.create_child @@ Option.value_exn !root_ctx in
  Context.set_option ctx Context.Optimization_level !optimization_level;
  (* if Utils.settings.with_debug && Utils.settings.output_debug_files_in_run_directory then (
     Context.set_option ctx Context.Keep_intermediates true; Context.set_option ctx Context.Dump_everything
     true); *)
  let info, ctx_arrays, params = compile_func ~name ~opt_ctx_arrays ctx bindings compiled in
  (if Utils.settings.output_debug_files_in_run_directory then
     let f_name = name ^ "-gccjit-debug.c" in
     Context.dump_to_file ctx ~update_locs:true f_name);
  let opt_ctx_arrays = if Map.is_empty ctx_arrays then None else Some ctx_arrays in
  let result = Context.compile ctx in
  Context.release ctx;
  { info; result; bindings; name; opt_ctx_arrays; params = List.map ~f:snd params }

let%track_sexp compile_batch ~(names : string array) ~opt_ctx_arrays bindings
    (compileds : (Low_level.traced_store * Low_level.t) array) =
  let open Gccjit in
  if Option.is_none !root_ctx then initialize ();
  let ctx = Context.create_child @@ Option.value_exn !root_ctx in
  Context.set_option ctx Context.Optimization_level !optimization_level;
  (* if Utils.settings.with_debug && Utils.settings.output_debug_files_in_run_directory then (
     Context.set_option ctx Context.Keep_intermediates true; Context.set_option ctx Context.Dump_everything
     true); *)
  let funcs =
    Array.map2_exn names compileds ~f:(fun name compiled ->
        compile_func ~name ~opt_ctx_arrays ctx bindings compiled)
  in
  (if Utils.settings.output_debug_files_in_run_directory then
     let f_name =
       String.(strip ~drop:(equal_char '_') @@ common_prefix (Array.to_list names)) ^ "-gccjit-debug.c"
     in
     Context.dump_to_file ctx ~update_locs:true f_name);
  let result = Context.compile ctx in
  Context.release ctx;
  Array.map2_exn names funcs ~f:(fun name (info, ctx_arrays, params) ->
      let opt_ctx_arrays = if Map.is_empty ctx_arrays then None else Some ctx_arrays in
      { info; result; bindings; name; opt_ctx_arrays; params = List.map ~f:snd params })

let%track_sexp link_compiled (old_context : context) (code : routine) : context * _ * _ * string =
  let label : string = old_context.label in
  let name : string = code.name in
  let arrays : Ndarray.t Base.Map.M(Tn).t =
    match code with
    | { opt_ctx_arrays = Some arrays; _ } -> arrays
    | { params; _ } ->
        List.fold params ~init:old_context.arrays ~f:(fun ctx_arrays -> function
          | Param_ptr tn ->
              let f = function
                | Some arr -> arr
                | None ->
                    Ndarray.create_array tn.Tn.prec ~dims:(Lazy.force tn.dims)
                    @@ Constant_fill { values = [| 0. |]; strict = false }
              in
              Map.update ctx_arrays tn ~f
          | _ -> ctx_arrays)
  in
  let context = { label; arrays; result = Some code.result } in
  let log_file_name = [%string "debug-%{label}-%{code.name}.log"] in
  let run_variadic =
    [%log_level
      Nothing;
      let rec link :
            'a 'b 'idcs.
            'idcs Indexing.bindings ->
            param_source list ->
            ('a -> 'b) Ctypes.fn ->
            ('a -> 'b, 'idcs, 'p1, 'p2) Indexing.variadic =
       fun (type a b idcs) (binds : idcs Indexing.bindings) params (cs : (a -> b) Ctypes.fn) ->
        match (binds, params) with
        | Empty, [] -> Indexing.Result (Gccjit.Result.code code.result name cs)
        | Bind _, [] -> invalid_arg "Gccjit_backend.link: too few static index params"
        | Bind (_, bs), Static_idx _ :: ps -> Param_idx (ref 0, link bs ps Ctypes.(int @-> cs))
        | Empty, Static_idx _ :: _ -> invalid_arg "Gccjit_backend.link: too many static index params"
        | bs, Log_file_name :: ps -> Param_1 (ref (Some log_file_name), link bs ps Ctypes.(string @-> cs))
        | bs, Param_ptr tn :: ps ->
            let nd = match Map.find arrays tn with Some nd -> nd | None -> assert false in
            (* let f ba = Ctypes.bigarray_start Ctypes_static.Genarray ba in let c_ptr = Ndarray.(map { f }
               nd) in *)
            let c_ptr = Ndarray.get_voidptr nd in
            Param_2 (ref (Some c_ptr), link bs ps Ctypes.(ptr void @-> cs))
      in
      (* Folding by [link] above reverses the input order. Important: [code.bindings] are traversed in the
         wrong order but that's OK because [link] only uses them to check the number of indices. *)
      link code.bindings (List.rev code.params) Ctypes.(void @-> returning void)]
  in
  let schedule () =
    let callback = Indexing.apply run_variadic in
    let%diagn_rt_sexp work () : unit =
      [%log_result name];
      callback ();
      if Utils.settings.debug_log_from_routines then (
        Utils.log_trace_tree _debug_runtime (Stdio.In_channel.read_lines log_file_name);
        Stdlib.Sys.remove log_file_name)
    in
    Tn.Work work
  in
  (context, Indexing.compiled_bindings code.bindings run_variadic, schedule, name)

let link (old_context : context) code =
  match code with
  | Postponed { compiled; bindings; name } ->
      let code = compile ~name ~opt_ctx_arrays:(Some old_context.arrays) bindings compiled in
      link_compiled old_context code
  | Compiled code -> link_compiled old_context code

let%track_sexp from_host (context : context) (la : Tn.t) : bool =
  match Map.find context.arrays la with
  | None -> false
  | Some c_arr -> (
      match la.Tn.array with
      | (lazy (Some h_arr)) ->
          Ndarray.map2 { f2 = Ndarray.A.blit } h_arr c_arr;
          [%log_printbox
            let indices = Array.init (Array.length @@ Lazy.force la.dims) ~f:(fun i -> i - 5) in
            Ndarray.render_array ~indices c_arr];
          true
      | (lazy None) -> false)

let%track_sexp to_host (context : context) (la : Tn.t) : bool =
  match Map.find context.arrays la with
  | None -> false
  | Some c_arr -> (
      match la.Tn.array with
      | (lazy (Some h_arr)) ->
          Ndarray.map2 { f2 = Ndarray.A.blit } c_arr h_arr;
          [%log_printbox
            let indices = Array.init (Array.length @@ Lazy.force la.dims) ~f:(fun i -> i - 5) in
            Ndarray.render_array ~indices h_arr];
          true
      | _ -> false)

let%track_sexp merge_from_global ~unoptim_ll_source ~ll_source ~name ~dst ~accum ~src =
  let body idcs =
    Low_level.(
      Set
        {
          tn = dst;
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
  Low_level.compile_proc ~unoptim_ll_source ~ll_source ~name [] llc

let%track_sexp merge ?name_prefix la ~accum ~(src : context) bindings =
  (* FIXME: reconstruct name if missing *)
  let name = Option.value name_prefix ~default:"" in
  let unoptim_ll_source = Utils.get_debug_formatter ~fname:(name ^ "-unoptimized.ll") in
  let ll_source = Utils.get_debug_formatter ~fname:(name ^ ".ll") in
  let name_prefix : string = Option.value ~default:"" @@ Option.map name_prefix ~f:(fun s -> s ^ "_") in
  Option.map (Map.find src.arrays la) ~f:(fun (src : Ndarray.t) ->
      let name = [%string "%{name_prefix}into_%{Tn.name la}"] in
      compile ~opt_ctx_arrays:None ~name bindings
      @@ merge_from_global ~unoptim_ll_source ~ll_source ~name ~dst:la ~accum ~src:(Ndarray.get_voidptr src))

let%track_sexp merge_batch ?(name_prefixes : string array option) ~occupancy tns ~accum
    ~(srcs : context array) bindings =
  (* FIXME: reconstruct name if missing *)
  let name =
    String.(
      strip ~drop:(equal_char '_')
      @@ common_prefix @@ Array.to_list
      @@ Option.value name_prefixes ~default:[||])
  in
  let unoptim_ll_source = Utils.get_debug_formatter ~fname:(name ^ "-unoptimized.ll") in
  let ll_source = Utils.get_debug_formatter ~fname:(name ^ ".ll") in
  let complete =
    Array.concat_map (Array.of_list tns) ~f:(fun tn ->
        Array.mapi srcs ~f:(fun i src ->
            match (occupancy tn ~src_n:i ~src, Map.find src.arrays tn) with
            | Utils.Skip, _ -> None
            | Optional { callback_if_missing }, None ->
                callback_if_missing ();
                None
            | Required, None ->
                failwith @@ "Gccjit_backend.merge_batch: missing tnode " ^ Tn.name tn ^ " in context "
                ^ src.label
            | _, Some src ->
                let prefix = match name_prefixes with Some ns -> ns.(i) ^ "_" | None -> "" in
                let name = [%string "%{prefix}into_%{Tn.name tn}"] in
                Some
                  ( (tn, i),
                    ( name,
                      merge_from_global ~unoptim_ll_source ~ll_source ~name ~dst:tn ~accum
                        ~src:(Ndarray.get_voidptr src) ) )))
  in
  let ids, compileds = Array.unzip @@ Array.filter_opt complete in
  let len = Array.length srcs in
  let together =
    Hashtbl.of_alist_exn (module Tn) @@ List.map tns ~f:(fun tn -> (tn, Array.create ~len None))
  in
  if Array.is_empty compileds then together
  else
    let names, compileds = Array.unzip compileds in
    let result = compile_batch ~opt_ctx_arrays:None ~names bindings compileds in
    Array.iter2_exn ids result ~f:(fun (tn, i) res ->
        let r = Hashtbl.find_exn together tn in
        assert (Option.is_none r.(i));
        r.(i) <- Some res);
    together

open Base
open Ir
module Lazy = Utils.Lazy

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

include Backend_impl.No_device_buffer_and_copying ()
open Backend_intf

let name = "gccjit"

let optimization_level () =
  Int.of_string @@ Utils.get_global_arg ~default:"3" ~arg_name:"gccjit_backend_optimization_level"

let root_ctx =
  let open Gccjit in
  let ctx = Context.create () in
  Context.set_option ctx Optimization_level (optimization_level ());
  ctx

module Tn = Tnode

type tn_info = {
  tn : Tn.t;  (** The original array. *)
  ptr : (Gccjit.rvalue[@sexp.opaque]) Lazy.t;
      (** Pointer to the first value of the associated array, any of: hosted, in context, local. *)
  dims : int array;
  num_typ : (Gccjit.type_[@sexp.opaque]);
      (** The type of the stored values: [short] (precision [Half]), [float] (precision [Single]),
          [double] (precision [Double]). *)
  prec : Ops.prec;
  zero_initialized : bool;
}
[@@deriving sexp_of]

type gccjit_param = Gccjit.param

(* let sexp_of_gccjit_param p = Sexp.Atom (Gccjit.Param.to_string p) *)
let sexp_of_gccjit_rvalue v = Sexp.Atom (Gccjit.RValue.to_string v)

type info_nodes = {
  ctx : (Gccjit.context[@sexp.opaque]);
  func : (Gccjit.function_[@sexp.opaque]);
  traced_store : (Low_level.traced_store[@sexp.opaque]);
  init_block : (Gccjit.block[@sexp.opaque]);
  nodes : (Tn.t, tn_info) Hashtbl.t;
  get_ident : Tn.t -> string;
  merge_node : (Gccjit.rvalue[@sexp.opaque]) option;
}
[@@deriving sexp_of]

type procedure = {
  info : info_nodes;
  bindings : Indexing.unit_bindings;
  name : string;
  result : (Gccjit.result[@sexp.opaque]);
  params : param_source list;
}
[@@deriving sexp_of]

(* let use_host_memory = true *)

let gcc_typ_of_prec =
  let open Gccjit in
  function
  | Ops.Byte_prec _ -> Type.Unsigned_char
  | Half_prec _ -> (* FIXME: *) Type.Float
  | Single_prec _ -> Type.Float
  | Double_prec _ -> Type.Double
  | Void_prec -> Type.Void

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
       [
         Lazy.force node.ptr;
         RValue.zero ctx c_int;
         RValue.int ctx c_index @@ Lazy.force node.tn.size_in_bytes;
       ]

let get_c_ptr ctx num_typ ptr = Gccjit.(RValue.ptr ctx (Type.pointer num_typ) ptr)

let prepare_node ~debug_log_zero_out ~get_ident ctx traced_store ~param_ptrs initializations
    (tn : Tn.t) =
  let open Gccjit in
  let traced = Low_level.(get_node traced_store tn) in
  let dims = Lazy.force tn.dims in
  let size_in_elems = Array.fold ~init:1 ~f:( * ) dims in
  let is_on_host = Tn.is_hosted_force tn 33 in
  assert (not @@ Tn.is_virtual_force tn 330);
  assert (Bool.(Option.is_some (Lazy.force tn.array) = is_on_host));
  let prec = Lazy.force tn.prec in
  let zero_initialized = traced.zero_initialized in
  let c_typ = gcc_typ_of_prec prec in
  let num_typ = Type.(get ctx c_typ) in
  let ptr_typ = Type.pointer num_typ in
  let ident = get_ident tn in
  let hosted = Tn.is_hosted_force tn 344 in
  let in_ctx = Tn.is_in_context_force ~use_host_memory tn 45 in
  let ptr =
    match (in_ctx, hosted) with
    | true, _ ->
        let p = Param.create ctx ptr_typ ident in
        param_ptrs := (p, Param_ptr tn) :: !param_ptrs;
        Lazy.from_val (RValue.param p)
    | false, true -> (
        let addr arr =
          Lazy.from_val @@ get_c_ptr ctx num_typ @@ Ctypes.bigarray_start Ctypes_static.Genarray arr
        in
        match Lazy.force tn.array with
        | Some (Byte_nd arr) -> addr arr
        | Some (Half_nd arr) -> addr arr
        | Some (Single_nd arr) -> addr arr
        | Some (Double_nd arr) -> addr arr
        | None -> assert false)
    | false, false ->
        let arr_typ = Type.array ctx num_typ size_in_elems in
        let v = ref None in
        let initialize _init_block func = v := Some (Function.local func arr_typ ident) in
        initializations := initialize :: !initializations;
        (* The array is the pointer but the address of the array is the same pointer. *)
        lazy (RValue.cast ctx (LValue.address @@ Option.value_exn ~here:[%here] !v) ptr_typ)
  in
  let result = { tn; ptr; dims; num_typ; prec; zero_initialized } in
  let backend_info = [%sexp_of: string * bool] (ident, zero_initialized) in
  let initialize init_block _func =
    Block.comment init_block
      [%string
        "Array #%{tn.id#Int} %{Tn.label tn}: %{Sexp.to_string_hum @@ backend_info}; ptr: \
         %{Sexp.to_string_hum @@ sexp_of_gccjit_rvalue @@ Lazy.force ptr}."];
    if zero_initialized then (
      debug_log_zero_out init_block result;
      zero_out ctx init_block result)
  in
  initializations := initialize :: !initializations;
  if not @@ Utils.sexp_mem ~elem:backend_info tn.backend_info then
    tn.backend_info <- Utils.sexp_append ~elem:backend_info tn.backend_info;
  result

let prec_to_kind prec =
  let open Gccjit in
  match prec with
  | Ops.Void_prec -> Type.Void
  | Byte_prec _ -> Type.Unsigned_char
  | Half_prec _ -> (* FIXME: *) Type.Unsigned_short
  | Single_prec _ -> Type.Float
  | Double_prec _ -> Type.Double

let is_builtin_op = function
  | Ops.Add | Sub | Mul | Div -> true
  | ToPowOf | Relu_gate | Satur01_gate | Arg2 | Arg1 | Max | Min | Mod | Cmplt | Cmpne | Cmpeq | Or
  | And ->
      false

let builtin_op = function
  | Ops.Add -> Gccjit.Plus
  | Sub -> Gccjit.Minus
  | Mul -> Gccjit.Mult
  | Div -> Gccjit.Divide
  | ToPowOf | Relu_gate | Satur01_gate | Arg2 | Arg1 | Max | Min | Mod | Cmplt | Cmpne | Cmpeq | Or
  | And ->
      invalid_arg "Exec_as_gccjit.builtin_op: not a builtin"

let node_debug_name get_ident node = get_ident node.tn

let debug_log_zero_out ctx log_functions get_ident block node =
  let open Gccjit in
  let c_index = Type.get ctx Type.Int in
  match Lazy.force log_functions with
  | Some (lf, pf, ff) ->
      let c_double = Type.get ctx Type.Double in
      let to_d v = RValue.cast ctx v c_double in
      Block.eval block @@ RValue.call ctx pf
      @@ lf
         :: RValue.string_literal ctx
              [%string
                {|memset_zero(%{node_debug_name get_ident node}) where before first element = %g
|}]
         :: [
              to_d @@ RValue.lvalue
              @@ LValue.access_array (Lazy.force node.ptr)
              @@ RValue.zero ctx c_index;
            ];
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

let assign_op_c_syntax = function
  | Ops.Arg1 -> invalid_arg "Gcc_backend.assign_op_c_syntax: Arg1 is not a C assignment operator"
  | Arg2 -> "="
  | Add -> "+="
  | Sub -> "-="
  | Mul -> "*="
  | Div -> "/="
  | Mod -> "%="
  (* | Shl -> "<<=" *)
  (* | Shr -> ">>=" *)
  | _ -> invalid_arg "Gcc_backend.assign_op_c_syntax: not a C assignment operator"

let compile_main ~name ~log_functions ~env { ctx; nodes; get_ident; merge_node; _ } func
    initial_block (body : Low_level.t) =
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
      Stdlib.Format.eprintf
        "exec_as_gccjit: missing index from@ %a@ among environment keys:@ %a\n%!" Sexp.pp_hum
        ([%sexp_of: Indexing.axis_index array] indices)
        Sexp.pp_hum
        ([%sexp_of: Indexing.symbol list] @@ Map.keys env);
      raise e
  in
  let c_float = Type.get ctx Type.Float in
  let c_double = Type.get ctx Type.Double in
  let cast_bool num_typ v = RValue.cast ctx (RValue.cast ctx v c_int) num_typ in
  let to_d v = RValue.cast ctx v c_double in
  (* Unique identifiers for local scope ids, can be non-unique globally due to inlining. *)
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
    | ToPowOf, Byte_prec _ ->
        raise @@ Utils.User_error "gccjit_backend: Byte_prec does not support ToPowOf"
    | Relu_gate, _ ->
        let cmp = cast_bool num_typ @@ RValue.comparison ctx Lt (RValue.zero ctx num_typ) v1 in
        RValue.binary_op ctx Mult num_typ cmp @@ v2
    | Satur01_gate, _ ->
        let cmp =
          cast_bool num_typ
          @@ RValue.binary_op ctx Logical_and (Type.get ctx Type.Bool)
               (RValue.comparison ctx Lt (RValue.zero ctx num_typ) v1)
               (RValue.comparison ctx Lt v1 (RValue.one ctx num_typ))
        in
        RValue.binary_op ctx Mult num_typ cmp @@ v2
    | Arg2, _ -> v2
    | Arg1, _ -> v1
    | Max, Double_prec _ ->
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "fmax") [ to_d v1; to_d v2 ]) num_typ
    | Max, Single_prec _ ->
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "fmaxf") [ v1; v2 ]) num_typ
    | Max, Half_prec _ ->
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "fmaxf") [ v1; v2 ]) num_typ
    | Max, Byte_prec _ ->
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "max") [ v1; v2 ]) num_typ
    | Min, Double_prec _ ->
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "fmin") [ to_d v1; to_d v2 ]) num_typ
    | Min, Single_prec _ ->
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "fminf") [ v1; v2 ]) num_typ
    | Min, Half_prec _ ->
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "fminf") [ v1; v2 ]) num_typ
    | Min, Byte_prec _ ->
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "min") [ v1; v2 ]) num_typ
    | Mod, _ -> RValue.binary_op ctx Modulo num_typ v1 v2
    | Cmplt, _ -> RValue.comparison ctx Lt v1 v2
    | Cmpne, _ -> RValue.comparison ctx Ne v1 v2
    | Cmpeq, _ -> RValue.comparison ctx Eq v1 v2
    | Or, _ -> RValue.binary_op ctx Logical_or num_typ v1 v2
    | And, _ -> RValue.binary_op ctx Logical_and num_typ v1 v2
  in
  let log_comment c =
    match log_functions with
    | Some (lf, pf, ff) ->
        Block.eval !current_block
        @@ RValue.call ctx pf [ lf; RValue.string_literal ctx ("\nCOMMENT: " ^ c ^ "\n") ];
        Block.eval !current_block @@ RValue.call ctx ff [ lf ]
    | _ -> Block.comment !current_block c
  in
  let get_node tn = Hashtbl.find_exn nodes tn in
  let rec debug_float ~env prec value =
    let loop = debug_float ~env prec in
    match value with
    | Low_level.Local_scope { id; _ } ->
        (* Not printing the inlined definition: (1) code complexity; (2) don't overload the logs. *)
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
    | Get_global (Merge_buffer _, None) -> assert false
    | Get_global (Merge_buffer { source_node_id }, Some idcs) ->
        let tn = Option.value_exn ~here:[%here] @@ Tn.find ~id:source_node_id in
        let idcs = lookup env idcs in
        let ptr = Option.value_exn ~here:[%here] merge_node in
        let offset = jit_array_offset ctx ~idcs ~dims:(Lazy.force tn.dims) in
        let v = to_d @@ RValue.lvalue @@ LValue.access_array ptr offset in
        (get_ident tn ^ ".merge[%d]{=%g}", [ offset; v ])
    | Get_global (C_function _, Some _) ->
        failwith "gccjit_backend: FFI with parameters NOT IMPLEMENTED YET"
    | Get (tn, idcs) ->
        let node = get_node tn in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:(Lazy.force tn.dims) in
        let v = to_d @@ RValue.lvalue @@ LValue.access_array (Lazy.force node.ptr) offset in
        (node_debug_name get_ident node ^ "[%d]{=%g}", [ offset; v ])
    | Constant c -> (Float.to_string c, [])
    | Embed_index (Fixed_idx i) -> (Int.to_string i, [])
    | Embed_index (Iterator s) -> (Indexing.symbol_ident s ^ "{=%d}", [ Map.find_exn env s ])
    | Binop (Arg1, v1, _v2) -> loop v1
    | Binop (Arg2, _v1, v2) -> loop v2
    | Binop (op, v1, v2) ->
        let prefix, infix, postfix = Ops.binop_c_syntax prec op in
        let v1, fillers1 = loop v1 in
        let v2, fillers2 = loop v2 in
        (String.concat [ prefix; v1; infix; " "; v2; postfix ], fillers1 @ fillers2)
    | Unop (Identity, v) -> loop v
    | Unop (Relu, v) ->
        let v, fillers = loop v in
        (String.concat [ "("; v; " > 0.0 ? "; v; " : 0.0)" ], fillers @ fillers)
    | Unop (op, v) ->
        let prefix, postfix = Ops.unop_c_syntax prec op in
        let v, fillers = loop v in
        (String.concat [ prefix; v; postfix ], fillers)
    | Ternop (op, cond_v, then_v, else_v) ->
        let prefix, infix1, infix2, postfix = Ops.ternop_c_syntax prec op in
        let cond, fillers1 = loop cond_v in
        let then_, fillers2 = loop then_v in
        let else_, fillers3 = loop else_v in
        ( String.concat [ prefix; cond; infix1; then_; infix2; else_; postfix ],
          fillers1 @ fillers2 @ fillers3 )
  in
  let debug_log_assignment ~env debug idcs node accum_op value v_code =
    match log_functions with
    | Some (lf, pf, ff) ->
        let v_format, v_fillers = debug_float ~env node.prec v_code in
        let offset = jit_array_offset ctx ~idcs ~dims:node.dims in
        let debug_line = "# " ^ String.substr_replace_all debug ~pattern:"\n" ~with_:"$" ^ "\n" in
        Block.eval !current_block @@ RValue.call ctx pf
        @@ [ lf; RValue.string_literal ctx debug_line ];
        Block.eval !current_block @@ RValue.call ctx pf
        @@ lf
           :: RValue.string_literal ctx
                [%string
                  {|%{node_debug_name get_ident node}[%d]{=%g} %{assign_op_c_syntax accum_op} %g = %{v_format}
|}]
           :: (to_d @@ RValue.lvalue @@ LValue.access_array (Lazy.force node.ptr) offset)
           :: offset :: to_d value :: v_fillers;
        Block.eval !current_block @@ RValue.call ctx ff [ lf ]
    | _ -> ()
  in
  let debug_log_index = debug_log_index ctx log_functions in
  let visited = Hash_set.create (module Tn) in
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
        Hash_set.add visited tn;
        let node = get_node tn in
        let value = loop_float ~name ~env ~num_typ:node.num_typ node.prec c2 in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:node.dims in
        let lhs = LValue.access_array (Lazy.force node.ptr) offset in
        debug_log_assignment ~env debug idcs node op value c2;
        Block.assign_op !current_block lhs (builtin_op op) value
    | Set { tn; idcs; llv; debug } ->
        Hash_set.add visited tn;
        let node = get_node tn in
        let value = loop_float ~name ~env ~num_typ:node.num_typ node.prec llv in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:node.dims in
        let lhs = LValue.access_array (Lazy.force node.ptr) offset in
        debug_log_assignment ~env debug idcs node Ops.Arg2 value llv;
        Block.assign !current_block lhs value
    | Zero_out tn ->
        if not @@ Hash_set.mem visited tn then (
          let node = get_node tn in
          debug_log_zero_out ctx (lazy log_functions) get_ident !current_block node;
          zero_out ctx !current_block node)
    | Set_local (id, llv) ->
        let lhs = Map.find_exn !locals id in
        let local_prec = Lazy.force id.tn.prec in
        let local_typ = gcc_typ_of_prec local_prec in
        let num_typ = Type.get ctx local_typ in
        let value = loop_float ~name ~env ~num_typ local_prec llv in
        Block.assign !current_block lhs value
    | Comment c -> log_comment c
    | Staged_compilation exp -> exp ()
  and loop_float ~name ~env ~num_typ prec v_code =
    let loop = loop_float ~name ~env ~num_typ prec in
    match v_code with
    | Local_scope { id = { scope_id = i; tn = { prec; _ } } as id; body; orig_indices = _ } ->
        let typ = Type.get ctx @@ prec_to_kind @@ Lazy.force prec in
        (* Scope ids can be non-unique due to inlining. *)
        let v_name = Int.("v" ^ to_string i ^ "_" ^ get_uid ()) in
        let lvalue = Function.local func typ v_name in
        (* Arrays are initialized to 0 by default. However, there is typically an explicit
           initialization for virtual nodes. *)
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
        let local_prec = Lazy.force id.tn.prec in
        let local_typ = gcc_typ_of_prec local_prec in
        let num_typ = Type.get ctx local_typ in
        if not @@ Ops.equal_prec prec local_prec then RValue.cast ctx rvalue num_typ else rvalue
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
        let local_typ = gcc_typ_of_prec local_prec in
        let num_typ = Type.get ctx local_typ in
        if not @@ Ops.equal_prec prec local_prec then RValue.cast ctx rvalue num_typ else rvalue
    | Get_global ((External_unsafe _ | Merge_buffer _), None) -> assert false
    | Get_global (Merge_buffer { source_node_id }, Some idcs) ->
        let tn = Option.value_exn ~here:[%here] @@ Tnode.find ~id:source_node_id in
        let ptr = Option.value_exn ~here:[%here] merge_node in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:(Lazy.force tn.dims) in
        let rvalue = RValue.lvalue @@ LValue.access_array ptr offset in
        let local_prec = Lazy.force tn.prec in
        let local_typ = gcc_typ_of_prec local_prec in
        let num_typ = Type.get ctx local_typ in
        if not @@ Ops.equal_prec prec local_prec then RValue.cast ctx rvalue num_typ else rvalue
    | Get_global (C_function _, Some _) ->
        failwith "gccjit_backend: FFI with parameters NOT IMPLEMENTED YET"
    | Get (tn, idcs) ->
        Hash_set.add visited tn;
        let node = get_node tn in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:node.dims in
        let rvalue = RValue.lvalue @@ LValue.access_array (Lazy.force node.ptr) offset in
        let local_prec = Lazy.force tn.prec in
        let local_typ = gcc_typ_of_prec local_prec in
        let num_typ = Type.get ctx local_typ in
        if not @@ Ops.equal_prec prec local_prec then RValue.cast ctx rvalue num_typ else rvalue
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
        let v = loop c in
        let cmp = cast_bool num_typ @@ RValue.comparison ctx Lt (RValue.zero ctx num_typ) v in
        RValue.binary_op ctx Mult num_typ cmp v
    | Unop (Satur01, c) ->
        let v = loop c in
        let zero = RValue.zero ctx num_typ in
        let one = RValue.double ctx num_typ 1.0 in
        let min =
          RValue.binary_op ctx Plus num_typ zero
            (RValue.binary_op ctx Mult num_typ (RValue.comparison ctx Lt v zero)
               (RValue.binary_op ctx Minus num_typ zero v))
        in
        RValue.binary_op ctx Plus num_typ min
          (RValue.binary_op ctx Mult num_typ (RValue.comparison ctx Gt v one)
             (RValue.binary_op ctx Minus num_typ one v))
    | Unop (((Exp | Log | Exp2 | Log2 | Sin | Cos | Sqrt | Tanh_approx) as op), c) ->
        let prefix, suffix = Ops.unop_c_syntax prec op in
        assert (
          String.is_suffix ~suffix:"(" prefix
          && String.equal suffix ")"
          && not (String.is_prefix ~prefix:"(" prefix));
        let f = Function.builtin ctx (String.drop_suffix prefix 1) in
        RValue.call ctx f [ loop c ]
    | Ternop ((FMA as op), c1, c2, c3) ->
        let prefix, _, _, _ = Ops.ternop_c_syntax prec op in
        let f = Function.builtin ctx (String.drop_suffix prefix 1) in
        RValue.call ctx f [ loop c1; loop c2; loop c3 ]
    | Ternop (Where, c1, c2, c3) ->
        let cond = loop c1 in
        let zero = RValue.zero ctx num_typ in
        let cmp = RValue.comparison ctx Eq cond zero in
        let v1 = loop c2 in
        let v2 = loop c3 in
        RValue.binary_op ctx Plus num_typ
          (RValue.binary_op ctx Mult num_typ cmp v2)
          (RValue.binary_op ctx Mult num_typ
             (RValue.binary_op ctx Minus num_typ (RValue.one ctx num_typ) cmp)
             v1)
    | Constant v -> RValue.double ctx num_typ v
    | Unop (Recip, c) ->
        let v = loop c in
        RValue.binary_op ctx Divide num_typ (RValue.one ctx num_typ) v
    | Unop (Recip_sqrt, c) ->
        let v = loop c in
        RValue.binary_op ctx Divide num_typ (RValue.one ctx num_typ)
          (RValue.call ctx (Function.builtin ctx "sqrtf") [ v ])
    | Unop (Neg, c) ->
        let v = loop c in
        RValue.unary_op ctx Negate num_typ v
    | Unop (Not, c) ->
        let v = loop c in
        cast_bool num_typ
        @@ RValue.unary_op ctx Logical_negate (Type.get ctx Type.Bool)
             (RValue.comparison ctx Eq v (RValue.zero ctx num_typ))
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

let%diagn_sexp compile_proc ~name ctx bindings ~get_ident
    Low_level.{ traced_store; llc = proc; merge_node } =
  let open Gccjit in
  let c_index = Type.get ctx Type.Int in
  let fkind = Function.Exported in
  let c_str = Type.(get ctx Const_char_ptr) in
  let log_file_name =
    if Utils.debug_log_from_routines () then
      Some (Param.create ctx c_str "log_file_name", Log_file_name)
    else None
  in
  let symbols = Indexing.bound_symbols bindings in
  let static_indices =
    List.map symbols ~f:(fun ({ static_symbol; _ } as s) ->
        (Param.create ctx c_index @@ Indexing.symbol_ident static_symbol, Static_idx s))
  in
  let merge_param =
    Option.map merge_node ~f:(fun tn ->
        let c_typ = gcc_typ_of_prec @@ Lazy.force tn.prec in
        let num_typ = Type.(get ctx c_typ) in
        let ptr_typ = Type.pointer num_typ in
        (Param.create ctx ptr_typ "merge_buffer", Merge_buffer))
  in
  let merge_node = Option.map merge_param ~f:(fun (p, _) -> RValue.param p) in
  let param_ptrs : (gccjit_param * param_source) list ref =
    ref (Option.to_list log_file_name @ Option.to_list merge_param @ static_indices)
  in
  let initializations = ref [] in
  let nodes = Hashtbl.create (module Tn) in
  let log_functions_ref = ref None in
  let log_functions = lazy !log_functions_ref in
  Hashtbl.iter_keys traced_store ~f:(fun tn ->
      if not (Tn.is_virtual_force tn 330) then
        let data =
          prepare_node
            ~debug_log_zero_out:(debug_log_zero_out ctx log_functions get_ident)
            ~get_ident ctx traced_store ~param_ptrs initializations tn
        in
        Hashtbl.add_exn nodes ~key:tn ~data);
  let params : (gccjit_param * param_source) list = !param_ptrs in
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
  Map.iteri env ~f:(fun ~key:sym ~data:idx ->
      debug_log_index init_block (Indexing.symbol_ident sym) idx);
  (* Do initializations in the order they were scheduled. *)
  List.iter (List.rev !initializations) ~f:(fun init -> init init_block func);
  let main_block = Block.create ~name func in
  let ctx_info : info_nodes =
    { ctx; traced_store; init_block; func; nodes; get_ident; merge_node }
  in
  let after_proc = compile_main ~name ~log_functions ~env ctx_info func main_block proc in
  (match log_functions with
  | Some (lf, _, _) ->
      (* FIXME: should be Imported? *)
      let file_ptr = Type.(get ctx File_ptr) in
      let fclose =
        Function.create ctx Imported
          Type.(get ctx Type.Void_ptr)
          "fclose"
          [ Param.create ctx file_ptr "f" ]
      in
      Block.eval after_proc @@ RValue.call ctx fclose [ lf ]
  | None -> ());
  Block.jump init_block main_block;
  Block.return_void after_proc;
  (ctx_info, params)

let compile ~(name : string) bindings (lowered : Low_level.optimized) =
  let get_ident = Low_level.get_ident_within_code ~no_dots:true [| lowered.llc |] in
  let open Gccjit in
  let ctx = Context.create_child root_ctx in
  Context.set_option ctx Context.Optimization_level (optimization_level ());
  (* if Utils.settings.with_debug && Utils.settings.output_debug_files_in_build_directory then (
     Context.set_option ctx Context.Keep_intermediates true; Context.set_option ctx
     Context.Dump_everything true); *)
  let info, params = compile_proc ~name ctx bindings ~get_ident lowered in
  (if Utils.settings.output_debug_files_in_build_directory then
     let f_name = Utils.build_file @@ name ^ "-gccjit-debug.c" in
     Context.dump_to_file ctx ~update_locs:true f_name);
  let%track7_sexp finalize result = Result.release result in
  let result = Context.compile ctx in
  Stdlib.Gc.finalise finalize result;
  Context.release ctx;
  { info; result; bindings; name; params = List.map ~f:snd params }

let%diagn_sexp compile_batch ~(names : string option array) bindings
    (lowereds : Low_level.optimized option array) =
  let get_ident =
    Low_level.get_ident_within_code ~no_dots:true
    @@ Array.filter_map lowereds ~f:(Option.map ~f:(fun { Low_level.llc; _ } -> llc))
  in
  let open Gccjit in
  let ctx = Context.create_child root_ctx in
  Context.set_option ctx Context.Optimization_level (optimization_level ());
  (* if Utils.settings.with_debug && Utils.settings.output_debug_files_in_build_directory then (
     Context.set_option ctx Context.Keep_intermediates true; Context.set_option ctx
     Context.Dump_everything true); *)
  let funcs =
    Array.mapi lowereds ~f:(fun i lowered ->
        match (names.(i), lowered) with
        | Some name, Some lowered ->
            let info, params = compile_proc ~name ctx bindings ~get_ident lowered in
            Some (info, params)
        | _ -> None)
  in
  (if Utils.settings.output_debug_files_in_build_directory then
     let f_name =
       String.(
         strip ~drop:(equal_char '_')
         @@ common_prefix (Array.to_list @@ Array.concat_map ~f:Option.to_array names))
       ^ "-gccjit-debug.c"
     in
     Context.dump_to_file ctx ~update_locs:true @@ Utils.build_file f_name);
  let result = Context.compile ctx in
  Context.release ctx;
  Array.mapi funcs ~f:(fun i ->
      Option.map2 names.(i) ~f:(fun name (info, params) ->
          { info; result; bindings; name; params = List.map ~f:snd params }))

let%track3_sexp link_compiled ~merge_buffer ~runner_label ctx_arrays (code : procedure) =
  let name : string = code.name in
  List.iter code.params ~f:(function
    (* FIXME: see cc_backend.ml *)
    | Param_ptr tn when not (Tn.known_shared_cross_streams tn) -> assert (Map.mem ctx_arrays tn)
    | _ -> ());
  let log_file_name = Utils.diagn_log_file [%string "debug-%{runner_label}-%{code.name}.log"] in
  let run_variadic =
    [%log_level
      0;
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
        | Empty, Static_idx _ :: _ ->
            invalid_arg "Gccjit_backend.link: too many static index params"
        | bs, Log_file_name :: ps ->
            Param_1 (ref (Some log_file_name), link bs ps Ctypes.(string @-> cs))
        | bs, Param_ptr tn :: ps ->
            let c_ptr =
              match Map.find ctx_arrays tn with
              | None ->
                  Ndarray.get_voidptr_not_managed
                  @@ Option.value_exn ~here:[%here]
                       ~message:
                         [%string
                           "Gcc_backend.link_compiled: node %{Tn.debug_name tn} missing from \
                            context: %{Tn.debug_memory_mode tn.Tn.memory_mode}"]
                  @@ Lazy.force tn.array
              | Some arr -> arr
            in
            Param_2 (ref (Some c_ptr), link bs ps Ctypes.(ptr void @-> cs))
        | bs, Merge_buffer :: ps ->
            let get_ptr buf = buf.Backend_intf.ptr in
            Param_2f (get_ptr, merge_buffer, link bs ps Ctypes.(ptr void @-> cs))
      in
      (* Folding by [link] above reverses the input order. Important: [code.bindings] are traversed
         in the wrong order but that's OK because [link] only uses them to check the number of
         indices. *)
      link code.bindings (List.rev code.params) Ctypes.(void @-> returning void)]
  in
  let%diagn_sexp work () : unit =
    [%log_result name];
    Indexing.apply run_variadic ();
    if Utils.debug_log_from_routines () then (
      Utils.log_trace_tree (Stdio.In_channel.read_lines log_file_name);
      Stdlib.Sys.remove log_file_name)
  in
  ( Indexing.lowered_bindings code.bindings run_variadic,
    Task.Task
      {
        context_lifetime = (ctx_arrays, code.result);
        description = "executes " ^ code.name ^ " on " ^ runner_label;
        work;
      } )

open Base

let optimization_level = ref 3

let session_context =
  let open Gccjit.Context in
  let ctx = create () in
  set_option ctx Optimization_level !optimization_level;
  ref ctx

type mem_properties =
  | Local_only  (** The array is only needed for a local computation and does not exist on host. *)
  | Local_finally_host
      (** The array is computed locally and then copied to host, if the flag [is_final] is true. *)
  | Host_only  (** The array is read from or updated directly on the host. *)
[@@deriving sexp, equal, compare, variants]

type ndarray = {
  hosted_ptr : Gccjit.rvalue option;
      (** Pointer to the first value of the associated [Bigarray], if hosted. *)
  local : Gccjit.lvalue option;  (** A local array, if any. *)
  mem : mem_properties;
  dims : int array;
  size_in_bytes : int;
  num_typ : Gccjit.type_;
      (** The type of the stored values: [short] (precision [Half]), [float] (precision [Single]),
          [double] (precision [Double]). *)
  is_double : bool;
}

let session_results : Gccjit.result list ref = ref []
let hoist_dynamic_indices = ref false

module LA = Lazy_array

type state = {
  ctx : Gccjit.context;
  func : Gccjit.function_;
  arrays : (LA.t, ndarray) Hashtbl.t;
  traced_store : Low_level.traced_store;
  init_block : Gccjit.block;
  finalize_block : Gccjit.block;  (** Only executed when [is_final] is true. *)
  is_final : Gccjit.param option;
}

let jit_array_offset ctx ~idcs ~dims =
  let open Gccjit in
  let c_index = Type.get ctx Type.Int in
  Array.fold2_exn idcs dims ~init:(RValue.zero ctx c_index) ~f:(fun offset idx dim ->
      RValue.binary_op ctx Plus c_index idx
      @@ RValue.binary_op ctx Mult c_index offset (RValue.int ctx c_index dim))

let get_array { ctx; func; arrays; traced_store; init_block; finalize_block; is_final = _ } v : ndarray =
  let open Gccjit in
  Hashtbl.find_or_add arrays v ~default:(fun () ->
      let tn = Low_level.(get_node traced_store v) in
      let dims = Lazy.force v.dims in
      let size_in_elems = Array.fold ~init:1 ~f:( * ) dims in
      let size_in_bytes = size_in_elems * Ndarray.prec_in_bytes v.prec in
      let is_on_host = !(v.materialized) in
      let c_void_ptr = Type.(get ctx Type.Void_ptr) in
      let c_index = Type.get ctx Type.Size_t in
      let c_int = Type.get ctx Type.Int in
      (* TODO: is the complexity of introducing this function and matching on Ndarray.t needed? *)
      let array c_typ is_double arr =
        let num_typ = Type.(get ctx c_typ) in
        let hosted_ptr =
          Option.map arr
            ~f:
              (Fn.compose
                 (RValue.ptr ctx @@ Type.pointer num_typ)
                 (Ctypes.bigarray_start Ctypes_static.Genarray))
        in
        let mem =
          if not is_on_host then Local_only
          else if not tn.read_before_write then Local_finally_host
          else Host_only
        in
        let arr_typ = Type.array ctx num_typ size_in_elems in
        let local = if is_host_only mem then None else Some (Function.local func arr_typ @@ LA.name v) in
        let cast_void rv = RValue.cast ctx rv c_void_ptr in
        if tn.zero_initialized then
          Option.first_some (Option.map local ~f:(Fn.compose cast_void LValue.address)) hosted_ptr
          |> Option.iter ~f:(fun rv_ptr ->
                 Block.eval init_block
                 @@ RValue.call ctx (Function.builtin ctx "memset")
                      [ rv_ptr; RValue.zero ctx c_int; RValue.int ctx c_index size_in_bytes ]);
        Option.iter hosted_ptr ~f:(fun hosted_ptr ->
            if is_local_finally_host mem then
              Block.eval finalize_block
              @@ RValue.call ctx (Function.builtin ctx "memcpy")
                   [
                     cast_void hosted_ptr;
                     cast_void @@ LValue.address @@ Option.value_exn local;
                     RValue.int ctx c_index size_in_bytes;
                   ]);
        let backend_info = (Sexp.to_string_hum @@ sexp_of_mem_properties mem) ^ ";" in
        if not @@ String.is_substring v.backend_info ~substring:backend_info then
          v.backend_info <- v.backend_info ^ backend_info;
        { hosted_ptr; local; mem; dims; size_in_bytes; num_typ; is_double }
      in
      match v.prec, Lazy.force v.array with
      | _, Some (Half_nd arr) -> (* FIXME: *) array Type.Float false (Some arr)
      | _, Some (Single_nd arr) -> array Type.Float false (Some arr)
      | _, Some (Double_nd arr) -> array Type.Double true (Some arr)
      | Half_prec _, None -> (* FIXME: *) array Type.Float false None
      | Single_prec _, None -> array Type.Float false None
      | Double_prec _, None -> array Type.Double true None
      | Void_prec, None -> assert false
      )

let cleanup_session () =
  let open Gccjit in
  List.iter !session_results ~f:Result.release;
  Context.release !session_context;
  session_context := Context.create ();
  Context.set_option !session_context Optimization_level !optimization_level;
  session_results := []

let prec_to_kind prec =
  let open Gccjit in
  match prec with
  | Ndarray.Void_prec -> Type.Void
  | Half_prec _ -> (* FIXME: *) Type.Unsigned_short
  | Single_prec _ -> Type.Float
  | Double_prec _ -> Type.Double

let prec_is_double = function Ndarray.Double_prec _ -> true | _ -> false

let is_builtin_op = function
  | Low_level.Add | Low_level.Mul -> true
  | Low_level.ToPowOf | Low_level.Relu_gate | Low_level.Arg2 | Low_level.Arg1 -> false

let builtin_op = function
  | Low_level.Add -> Gccjit.Plus
  | Low_level.Mul -> Gccjit.Mult
  | Low_level.ToPowOf | Low_level.Relu_gate | Low_level.Arg2 | Low_level.Arg1 ->
      invalid_arg "Exec_as_gccjit.builtin_op: not a builtin"

let get_ptr array =
  match array.local with Some lv -> Gccjit.RValue.lvalue lv | None -> Option.value_exn array.hosted_ptr

let jit_code ~name ~(env : Gccjit.rvalue Low_level.environment) ({ ctx; func; _ } as state) initial_block body
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
      Caml.Format.eprintf "exec_as_gccjit: missing index from@ %a@ among environment keys:@ %a\n%!"
        Sexp.pp_hum
        ([%sexp_of: Indexing.axis_index array] indices)
        Sexp.pp_hum
        ([%sexp_of: Indexing.symbol list] @@ Map.keys env);
      raise e
  in
  let c_float = Type.get ctx Type.Float in
  let c_double = Type.get ctx Type.Double in
  let cast_bool num_typ v = RValue.cast ctx (RValue.cast ctx v c_int) num_typ in
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
    | Low_level.Add -> RValue.binary_op ctx Plus num_typ v1 v2
    | Low_level.Mul -> RValue.binary_op ctx Mult num_typ v1 v2
    | Low_level.ToPowOf when is_double ->
        let base = RValue.cast ctx v1 c_double in
        let expon = RValue.cast ctx v2 c_double in
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "pow") [ base; expon ]) num_typ
    | Low_level.ToPowOf ->
        let base = RValue.cast ctx v1 c_float in
        let expon = RValue.cast ctx v2 c_float in
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "powf") [ base; expon ]) num_typ
    | Low_level.Relu_gate ->
        let cmp = cast_bool num_typ @@ RValue.comparison ctx Lt (RValue.zero ctx num_typ) v1 in
        RValue.binary_op ctx Mult num_typ cmp @@ v2
    | Low_level.Arg2 -> v2
    | Low_level.Arg1 -> v1
  in
  let log_comment c =
    (if !Low_level.with_debug && !Low_level.executor_print_comments then
       let f = Function.builtin ctx "printf" in
       Block.eval !current_block @@ RValue.call ctx f [ RValue.string_literal ctx ("\nComment: " ^ c ^ "\n") ]);
    Block.comment !current_block c
  in
  let rec loop_proc ~(env : rvalue Low_level.environment) ~name (body : Low_level.t) : unit =
    let loop = loop_proc ~env ~name in
    match body with
    | Noop -> ()
    | Low_level.Seq (c1, c2) ->
        loop c1;
        loop c2
    | For_loop { index; from_; to_; body; trace_it = _ } -> jit_for_loop ~env index ~from_ ~to_ body
    | Set (_, _, Binop (Arg2, Get (_, _), _)) -> assert false
    | Set (array, idcs, Binop (op, Get (array2, idcs2), c2))
      when LA.equal array array2
           && [%equal: Indexing.axis_index array] idcs idcs2
           && is_builtin_op op ->
        (* FIXME: maybe it's not worth it? *)
        let array = get_array state array in
        let value = loop_float ~name ~env ~num_typ:array.num_typ ~is_double:array.is_double c2 in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:array.dims in
        let lhs = LValue.access_array (get_ptr array) offset in
        Block.assign_op !current_block lhs (builtin_op op) value
    | Set (array, idcs, value) ->
        let array = get_array state array in
        let value = loop_float ~name ~env ~num_typ:array.num_typ ~is_double:array.is_double value in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:array.dims in
        let lhs = LValue.access_array (get_ptr array) offset in
        Block.assign !current_block lhs value
    | Zero_out array ->
        if Hashtbl.mem state.arrays array then
          failwith
            ("exec_as_gccjit: Non-initialization zeroing-out NOT IMPLEMENTED YET: " ^ Sexp.to_string_hum
            @@ [%sexp_of: LA.t] array);
        let tn = Low_level.(get_node state.traced_store array) in
        assert tn.zero_initialized
        (* The initialization will be emitted by get_array. *)
    | Set_local (id, value) ->
        let lhs, num_typ, is_double = Map.find_exn !locals id in
        let value = loop_float ~name ~env ~num_typ ~is_double value in
        Block.assign !current_block lhs value
    | Comment c -> log_comment c
    | Staged_compilation exp -> exp ()
  and loop_float ~name ~env ~num_typ ~is_double value : rvalue =
    let loop = loop_float ~name ~env ~num_typ ~is_double in
    match value with
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
        (* FIXME: Convert according to local_is_double ?= is_double. *)
        RValue.lvalue lvalue
    | Get_global (C_function f_name) ->
        (* TODO: this is too limiting. *)
        let f = Function.builtin ctx f_name in
        RValue.call ctx f []
    | Get (array, idcs) ->
        let array = get_array state array in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:array.dims in
        RValue.lvalue @@ LValue.access_array (get_ptr array) offset
    | Binop (Low_level.Arg2, _, c2) -> loop c2
    | Binop (Low_level.Arg1, c1, _) -> loop c1
    | Binop (op, c1, c2) -> loop_binop op ~num_typ ~is_double ~v1:(loop c1) ~v2:(loop c2)
    | Unop (Low_level.Identity, c) -> loop c
    | Unop (Low_level.Relu, c) ->
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

let jit_func ~name ctx (traced_store, proc) =
  let open Gccjit in
  let fkind = Function.Exported in
  let is_final = Param.create ctx Type.(get ctx Bool) "is_final" in
  let func = Function.create ctx fkind (Type.get ctx Void) name [ is_final ] in
  let init_block = Block.create ~name:("init_" ^ name) func in
  let finalize_block = Block.create ~name:("finalize_" ^ name) func in
  let main_block = Block.create ~name func in
  let env = Low_level.empty_env in
  let state =
    {
      ctx;
      func;
      traced_store;
      init_block;
      finalize_block;
      arrays = Hashtbl.Poly.create ();
      is_final = Some is_final;
    }
  in
  let after_proc = jit_code ~name ~env state main_block proc in
  Block.jump init_block main_block;
  let b_after_if = Block.create ~name:("after_finalize_replicated_" ^ name) func in
  let guard = RValue.param is_final in
  Block.cond_jump after_proc guard finalize_block (* on true *) b_after_if (* on false *);
  Block.jump finalize_block b_after_if;
  Block.return_void b_after_if;
  if !Low_level.with_debug then
    let suf = "-gccjit-debug.c" in
    let f_name =
      if !Low_level.keep_files_in_run_directory then name ^ suf else Caml.Filename.temp_file (name ^ "-") suf
    in
    Context.dump_to_file ctx ~update_locs:true f_name

let jit ~name ?verbose:_ compiled =
  (* TODO: add verbose logs *)
  let open Gccjit in
  let ctx = Context.create_child !session_context in
  Context.set_option ctx Context.Optimization_level !optimization_level;
  (*
  if !Low_level.with_debug && !Low_level.keep_files_in_run_directory then (
    Context.set_option ctx Context.Keep_intermediates true;
    Context.set_option ctx Context.Dump_everything true);
  *)
  jit_func ~name ctx compiled;
  let result = Context.compile ctx in
  session_results := result :: !session_results;
  let routine = Result.code result name Ctypes.(bool @-> returning void) in
  Context.release ctx;
  fun ~is_initial:_ ~is_final -> routine is_final

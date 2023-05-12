open Base

let optimization_level = ref 3

let session_context =
  let open Gccjit.Context in
  let ctx = create () in
  set_option ctx Optimization_level !optimization_level;
  ref ctx

type tensor = {
  ptr : Gccjit.rvalue;  (** Pointer to the first value of the associated [Bigarray]. *)
  dims : int array;  (** Dimensions (shape) of the tensor. *)
  num_typ : Gccjit.type_;
      (** The type of the stored values: [signed char] (corresponds to precision [Byte]),
      [short] (precision [Half]), [float] (precision [Single]), [double] (precision [Double]). *)
  is_double : bool;
}

let session_results : Gccjit.result list ref = ref []
let compiled_session_globals : Gccjit.result option ref = ref None
let hoist_dynamic_indices = ref false

let get_tensor ctx data : tensor =
  let open Node in
  let open Gccjit in
  let tensor c_typ is_double arr =
    let num_typ = Type.(get ctx c_typ) in
    let ptr = RValue.ptr ctx (Type.pointer num_typ) @@ Ctypes.bigarray_start Ctypes_static.Genarray arr in
    let dims = Bigarray.Genarray.dims arr in
    Option.iter !compiled_session_globals ~f:Result.release;
    compiled_session_globals := None;
    { ptr; dims; num_typ; is_double }
  in
  let open NodeUI in
  let arr = Option.value_exn @@ get_tensor data in
  match arr with
  | Byte_as_int_nd arr -> tensor Type.Signed_char false arr
  | Half_as_int_nd arr -> tensor Type.Short false arr
  | Single_nd arr -> tensor Type.Float false arr
  | Double_nd arr -> tensor Type.Double true arr

let cleanup_session () =
  let open Gccjit in
  List.iter !session_results ~f:Result.release;
  Option.iter !compiled_session_globals ~f:Result.release;
  compiled_session_globals := None;
  Context.release !session_context;
  session_context := Context.create ();
  Context.set_option !session_context Optimization_level !optimization_level;
  session_results := []

let jit_array_offset ctx ~idcs ~dims =
  let open Gccjit in
  let c_index = Type.get ctx Type.Int in
  Array.fold2_exn idcs dims ~init:(RValue.zero ctx c_index) ~f:(fun offset idx dim ->
      RValue.binary_op ctx Plus c_index idx
      @@ RValue.binary_op ctx Mult c_index offset (RValue.int ctx c_index dim))

let prec_to_kind prec =
  let open Gccjit in
  match prec with
  | NodeUI.Void_prec -> Type.Void
  | Byte_as_int_prec _ -> Type.Signed_char
  | Half_as_int_prec _ -> Type.Short
  | Single_prec _ -> Type.Float
  | Double_prec _ -> Type.Double

let prec_is_double = function NodeUI.Double_prec _ -> true | _ -> false

let is_builtin_op = function
  | Code.Add | Code.Mul -> true
  | Code.ToPowOf | Code.Relu_gate | Code.Arg2 | Code.Arg1 -> false

let builtin_op = function
  | Code.Add -> Gccjit.Plus
  | Code.Mul -> Gccjit.Mult
  | Code.ToPowOf | Code.Relu_gate | Code.Arg2 | Code.Arg1 ->
      invalid_arg "Exec_as_gccjit.builtin_op: not a builtin"

let tensor_ptr_name = function
  | NodeUI.{ id; field = Value } -> "n" ^ Int.to_string id ^ "_value"
  | NodeUI.{ id; field = Grad } -> "n" ^ Int.to_string id ^ "_grad"

let jit_code ~name ~env ~task_id ctx func initial_block (body : unit Code.low_level) : Gccjit.block =
  let open Gccjit in
  let c_int = Type.get ctx Type.Int in
  let c_index = c_int in
  let c_float = Type.get ctx Type.Float in
  let c_double = Type.get ctx Type.Double in
  let cast_bool num_typ v = RValue.cast ctx (RValue.cast ctx v c_int) num_typ in
  (* Source of unique identifiers. E.g. local scope ids can be non-unique due to inlining.
     We also need unique ids for computation ordering lvalues. *)
  let uid = ref 0 in
  let get_uid () =
    let id = !uid in
    Int.to_string id
  in
  let locals = ref Map.Poly.empty in
  let current_block = ref initial_block in
  let loop_binop op ~num_typ ~is_double ~v1 ~v2 =
    match op with
    | Code.Add -> RValue.binary_op ctx Plus num_typ v1 v2
    | Code.Mul -> RValue.binary_op ctx Mult num_typ v1 v2
    | Code.ToPowOf when is_double ->
        let base = RValue.cast ctx v1 c_double in
        let expon = RValue.cast ctx v2 c_double in
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "pow") [ base; expon ]) num_typ
    | Code.ToPowOf ->
        let base = RValue.cast ctx v1 c_float in
        let expon = RValue.cast ctx v2 c_float in
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "powf") [ base; expon ]) num_typ
    | Code.Relu_gate ->
        let cmp = cast_bool num_typ @@ RValue.comparison ctx Lt (RValue.zero ctx num_typ) v1 in
        RValue.binary_op ctx Mult num_typ cmp @@ v2
    | Code.Arg2 -> v2
    | Code.Arg1 -> v1
  in
  let lookup ?provider_dim (env, dyn_env) indices =
    Array.map indices ~f:(function
      | Shape.Fixed_idx i -> RValue.int ctx c_index i
      | Iterator s -> Map.find_exn env s
      | Task_id -> RValue.param task_id
      | Dynamic_recipient s -> Map.find_exn dyn_env s
      | Dynamic_provider _ -> Option.value_exn provider_dim)
  in
  let log_comment c =
    (if !Code.with_debug && !Code.executor_print_comments then
       let f = Function.builtin ctx "printf" in
       Block.eval !current_block
       @@ RValue.call ctx f
            [ RValue.string_literal ctx ("\nComment for task %d: " ^ c ^ "\n"); RValue.param task_id ]);
    Block.comment !current_block c
  in
  let rec loop_proc ~name ~env (body : unit Code.low_level) : unit =
    match body with
    | Code.Lines lines ->
        Array.iteri lines ~f:(fun i line -> loop_proc ~name:(name ^ "_at_line_" ^ Int.to_string i) ~env line)
    | For_loop { index; from_; to_; body } -> jit_for_loop ~env index ~from_ ~to_ (Either.First body)
    | If_task_id_is { for_task_id = _; body } when !Shape.num_parallel_tasks <= 1 -> loop_proc ~name ~env body
    | If_task_id_is { for_task_id; body } ->
        let open Gccjit in
        let id = get_uid () in
        let b_if_body =
          Block.create ~name:("body_if_task_id_is_" ^ Int.to_string for_task_id ^ "_" ^ id) func
        in
        let b_after_if =
          Block.create ~name:("after_if_task_id_is_" ^ Int.to_string for_task_id ^ "_" ^ id) func
        in
        let guard = RValue.comparison ctx Eq (RValue.param task_id) (RValue.int ctx c_index for_task_id) in
        Block.cond_jump !current_block guard b_if_body (* on true *) b_after_if (* on false *);
        current_block := b_if_body;
        loop_proc ~name ~env body;
        Block.jump !current_block b_after_if;
        current_block := b_after_if
    | Synchronize _ when !Shape.num_parallel_tasks <= 1 -> ()
    | Synchronize info ->
        (* FIXME: lock-free implementation with an int for each task that counts the stage the task
           is at, and a busy loop waiting for all other stages to arrive at the current task's
           (incremented) stage. *)
        (* failwith ("Exec_as_gccjit.jit_code.Synchronize: NOT IMPLEMENTED YET -- at " ^ info.info) *)
        log_comment ("NOT IMPLEMENTED SYNCHRONIZATION: " ^ info.info)
    | Reset_synchronizer when !Shape.num_parallel_tasks <= 1 -> ()
    | Reset_synchronizer ->
        (* failwith "Exec_as_gccjit.jit_code.Reset_synchronizer: NOT IMPLEMENTED YET" *)
        log_comment ("NOT IMPLEMENTED RESET SYNCHRONIZATER")
    | Set (data_node, idcs, Binop (op, Get (tensor, idcs2), c2))
      when NodeUI.equal_tensor_ptr data_node tensor
           && [%equal: Code.index array] idcs idcs2
           && is_builtin_op op ->
        let tensor = get_tensor ctx data_node in
        let value = loop_float ~name ~env ~num_typ:tensor.num_typ ~is_double:tensor.is_double c2 in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:tensor.dims in
        let lhs = LValue.access_array tensor.ptr offset in
        Block.assign_op !current_block lhs (builtin_op op) value
    | Set (data_node, idcs, Binop (op, (Get (tensor, _) as c1), c2))
      when NodeUI.equal_tensor_ptr data_node tensor ->
        let tensor = get_tensor ctx data_node in
        let v2 = loop_float ~name ~env ~num_typ:tensor.num_typ ~is_double:tensor.is_double c2 in
        (* Force the ordering of computations to reduce race conditions. *)
        let l2 = Function.local func tensor.num_typ (tensor_ptr_name data_node ^ "_" ^ get_uid ()) in
        Block.assign !current_block l2 v2;
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:tensor.dims in
        let lhs = LValue.access_array tensor.ptr offset in
        let v1 = loop_float ~name ~env ~num_typ:tensor.num_typ ~is_double:tensor.is_double c1 in
        Block.assign !current_block lhs
        @@ loop_binop op ~num_typ:tensor.num_typ ~is_double:tensor.is_double ~v1 ~v2:(RValue.lvalue l2)
    | Set (data_node, idcs, value) ->
        let tensor = get_tensor ctx data_node in
        let value = loop_float ~name ~env ~num_typ:tensor.num_typ ~is_double:tensor.is_double value in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:tensor.dims in
        let lhs = LValue.access_array tensor.ptr offset in
        Block.assign !current_block lhs value
    | Set_local (id, value) ->
        let lhs, num_typ, is_double = Map.find_exn !locals id in
        let value = loop_float ~name ~env ~num_typ ~is_double value in
        Block.assign !current_block lhs value
    | Comment c -> log_comment c
    | Dynamic_indices { tensor; tensor_idcs; dynamic_idcs; target_dims; body } ->
        jit_dynamic_indices ~name ~env tensor ~tensor_idcs ~dynamic_idcs ~target_dims body
  and loop_float ~name ~env ~num_typ ~is_double value : rvalue =
    let loop = loop_float ~name ~env ~num_typ ~is_double in
    match value with
    | Local_scope { id = { scope_id = i; tensor } as id; prec; body; orig_indices } ->
        let typ = Type.get ctx @@ prec_to_kind prec in
        (* Scope ids can be non-unique due to inlining. *)
        let v_name = Int.("v" ^ to_string i ^ "_" ^ get_uid ()) in
        let lvalue = Function.local func typ v_name in
        (* Tensors are initialized to 0 by default. However, there is typically an explicit
           initialization for virtual nodes. *)
        Block.assign !current_block lvalue @@ RValue.zero ctx typ;
        let old_locals = !locals in
        locals := Map.update !locals id ~f:(fun _ -> (lvalue, typ, prec_is_double prec));
        loop_proc ~name:(name ^ "_at_" ^ v_name) ~env body;
        locals := old_locals;
        (if !Code.debug_virtual_nodes then
           let tensor = get_tensor ctx tensor in
           let idcs = lookup env orig_indices in
           let offset = jit_array_offset ctx ~idcs ~dims:tensor.dims in
           let lhs = LValue.access_array tensor.ptr offset in
           Block.assign !current_block lhs @@ RValue.lvalue lvalue);
        RValue.lvalue lvalue
    | Get_local id ->
        let lvalue, _typ, _local_is_double = Map.find_exn !locals id in
        (* FIXME: Convert according to local_is_double ?= is_double. *)
        RValue.lvalue lvalue
    | Get_global Task_id -> RValue.cast ctx (RValue.param task_id) num_typ
    | Get_global (C_function f_name) ->
        (* TODO: this is too limiting. *)
        let f = Function.builtin ctx f_name in
        RValue.call ctx f []
    | Get (tensor, idcs) ->
        let tensor = get_tensor ctx tensor in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:tensor.dims in
        RValue.lvalue @@ LValue.access_array tensor.ptr offset
    | Binop (Code.Arg2, _, c2) -> loop c2
    | Binop (Code.Arg1, c1, _) -> loop c1
    | Binop (op, c1, c2) -> loop_binop op ~num_typ ~is_double ~v1:(loop c1) ~v2:(loop c2)
    | Unop (Code.Identity, c) -> loop c
    | Unop (Code.Relu, c) ->
        let cmp = cast_bool num_typ @@ RValue.comparison ctx Lt (RValue.zero ctx num_typ) @@ loop c in
        RValue.binary_op ctx Mult num_typ cmp @@ loop c
    | Constant v -> RValue.double ctx num_typ v
  and jit_for_loop ~env (Code.{ sym = Shape.Symbol s; uid } as key) ~from_ ~to_ body : unit =
    let open Gccjit in
    let i = "i" ^ Int.to_string s ^ "_" ^ Int.to_string uid in
    let index = Function.local func c_index i in
    let env = (Map.add_exn ~key ~data:(RValue.lvalue index) @@ fst env, snd env) in
    let b_loop_cond = Block.create ~name:("loop_cond_" ^ i) func in
    let b_loop_body = Block.create ~name:("loop_body_" ^ i) func in
    let b_after_loop = Block.create ~name:("after_loop_" ^ i) func in
    Block.assign !current_block index (RValue.int ctx c_index from_);
    Block.jump !current_block b_loop_cond;
    let guard = RValue.comparison ctx Gt (RValue.lvalue index) (RValue.int ctx c_index to_) in
    Block.cond_jump b_loop_cond guard b_after_loop (* on true *) b_loop_body (* on false *);
    current_block := b_loop_body;
    (match body with
    | Either.First body -> loop_proc ~name ~env body
    | Second callback -> callback (RValue.lvalue index));
    Block.assign_op !current_block index Plus (RValue.one ctx c_index);
    Block.jump !current_block b_loop_cond;
    current_block := b_after_loop
  and jit_dynamic_indices ~name ~env tensor ~tensor_idcs ~dynamic_idcs ~target_dims body =
    let tensor = get_tensor ctx tensor in
    let env =
      Array.foldi dynamic_idcs ~init:env ~f:(fun provider_dim env (Symbol s as key) ->
          let target_dim =
            RValue.int ctx c_int
              (match target_dims.(provider_dim) with
              | Shape.Dim d -> d
              | Parallel -> !Shape.num_parallel_tasks)
          in
          let provider_dim = RValue.int ctx c_int provider_dim in
          let idcs = lookup ~provider_dim env tensor_idcs in
          let prov_offset = jit_array_offset ctx ~idcs ~dims:tensor.dims in
          let dyn_index = RValue.lvalue @@ LValue.access_array tensor.ptr prov_offset in
          let dyn_index = RValue.cast ctx dyn_index c_index in
          let dyn_index = RValue.binary_op ctx Modulo c_index dyn_index target_dim in
          let data =
            if !hoist_dynamic_indices then (
              let sym_index = Function.local func c_index ("i" ^ Int.to_string s) in
              Block.assign !current_block sym_index dyn_index;
              RValue.lvalue sym_index)
            else dyn_index
          in
          (fst env, Map.add_exn ~key ~data @@ snd env))
    in
    loop_proc ~name ~env body
  in

  loop_proc ~name ~env body;
  !current_block

let jit_func ~name ctx proc =
  let open Gccjit in
  let fkind = Function.Exported in
  let env = (Map.Poly.empty, Map.Poly.empty) in
  let task_id = Param.create ctx Type.(get ctx Int) "task_id" in
  let func = Function.create ctx fkind (Type.get ctx Void) name [ task_id ] in
  let block = Block.create ~name func in
  let after_proc = jit_code ~name ~env ~task_id ctx func block proc in
  Block.return_void after_proc;
  if !Code.with_debug then
    let suf = "-gccjit-debug.c" in
    let f_name =
      if !Code.keep_files_in_run_directory then name ^ suf else Caml.Filename.temp_file (name ^ "-") suf
    in
    Context.dump_to_file ctx ~update_locs:true f_name
(* match !compiled_session_globals with
   | None ->
     Context.dump_to_file !session_context ~update_locs:true "globals-gccjit-debug.c";
     let globals = Context.compile !session_context in
      compiled_session_globals := Some globals
   | Some _ -> () *)

let error_message ~name ~prefix ?extra_error_msg ~contents exc =
  let backtrace = Caml.Printexc.get_backtrace () in
  let exc_str = Caml.Printexc.to_string exc in
  let message = Buffer.create (String.length contents + String.length backtrace + String.length exc_str) in
  let msg = Buffer.add_string message in
  msg name;
  msg ": ";
  msg prefix;
  msg exc_str;
  msg "\n";
  msg backtrace;
  (match extra_error_msg with
  | None -> ()
  | Some extra ->
      msg "\nIn the context of:\n";
      msg extra);
  (* let from_pos, to_pos = first_file_span ~contents ~message:(Buffer.contents message) in
     let from_pos = Int.min from_pos @@ String.length contents - 1 in
     let to_pos = Int.min to_pos @@ String.length contents - 1 in
     msg "\nIn code span ";
     msg error_opening_delimiter; msg "..."; msg error_closing_delimiter; msg ":\n";
     msg @@ String.sub contents ~pos:0 ~len:from_pos;
     msg error_opening_delimiter;
     msg @@ String.sub contents ~pos:from_pos ~len:(to_pos - from_pos);
     msg error_closing_delimiter;
     msg @@ String.sub contents ~pos:to_pos ~len:(String.length contents - to_pos); *)
  msg contents;
  Buffer.contents message

let jit_task_id_func ~name compiled =
  let open Gccjit in
  let ctx = Context.create_child !session_context in
  Context.set_option ctx Context.Optimization_level !optimization_level;
  (*
  if !Code.with_debug && !Code.keep_files_in_run_directory then (
    Context.set_option ctx Context.Keep_intermediates true;
    Context.set_option ctx Context.Dump_everything true);
  *)
  jit_func ~name ctx compiled;
  let result = Context.compile ctx in
  session_results := result :: !session_results;
  let routine = Result.code result name Ctypes.(int @-> returning void) in
  Context.release ctx;
  fun ~task_id -> routine task_id

let jit_unit_func ~name compiled =
  let open Gccjit in
  let ctx = Context.create_child !session_context in
  Context.set_option ctx Context.Optimization_level !optimization_level;
  (*
  if !Code.with_debug && !Code.keep_files_in_run_directory then (
    Context.set_option ctx Context.Keep_intermediates true;
    Context.set_option ctx Context.Dump_everything true);
  *)
  jit_func ~name ctx compiled;
  let result = Context.compile ctx in
  session_results := result :: !session_results;
  let routine = Result.code result name Ctypes.(void @-> returning void) in
  Context.release ctx;
  routine

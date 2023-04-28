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

let get_tensor acc ctx id : tensor =
  let open Ocannl_runtime.Node in
  let open Gccjit in
  let tensor c_typ is_double arr =
    let num_typ = Type.(get ctx c_typ) in
    let ptr = RValue.ptr ctx (Type.pointer num_typ) @@ Ctypes.bigarray_start Ctypes_static.Genarray arr in
    let dims = Bigarray.Genarray.dims arr in
    Option.iter !compiled_session_globals ~f:Result.release;
    compiled_session_globals := None;
    { ptr; dims; num_typ; is_double }
  in
  let n = get id in
  match acc n with
  | Byte_as_int_nd arr -> tensor Type.Signed_char false arr
  | Half_as_int_nd arr -> tensor Type.Short false arr
  | Single_nd arr -> tensor Type.Float false arr
  | Double_nd arr -> tensor Type.Double true arr

let get_value_tensor = get_tensor (fun n -> n.value)
let get_grad_tensor = get_tensor (fun n -> Option.value_exn n.grad)

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
  | Code.Byte_as_int_prec _ -> Type.Signed_char
  | Half_as_int_prec _ -> Type.Short
  | Single_prec _ -> Type.Float
  | Double_prec _ -> Type.Double

let prec_is_double = function Code.Double_prec _ -> true | _ -> false

let jit_code ~name ~env ctx func block (body : unit Code.low_level) : Gccjit.block =
  let open Gccjit in
  let c_int = Type.get ctx Type.Int in
  let c_index = c_int in
  let c_float = Type.get ctx Type.Float in
  let c_double = Type.get ctx Type.Double in
  let cast_bool num_typ v = RValue.cast ctx (RValue.cast ctx v c_int) num_typ in
  let locals = Hashtbl.Poly.create () in
  let current_block = ref initial_block in
  let lookup ?provider_dim env indices =
    Array.map indices
      ~f:
        Shape.(
          function
          | Fixed_idx i -> RValue.int ctx c_index i
          | Iterator s | Dynamic_recipient s -> Map.find_exn env s
          | Dynamic_provider _ -> Option.value_exn provider_dim)
  in
  let rec loop_proc ~name ~env (body : unit Code.low_level) : unit =
    (* TODO: consider matching gccjit's assign_op-style pattern (but probably no benefit). *)
    match body with
    | Code.Lines lines ->
        Array.iteri lines ~f:(fun i line -> loop_proc ~name:(name ^ "_at_line_" ^ Int.to_string i) ~env line)
    | For_loop { index; from_; to_; body } -> jit_for_loop ~env index ~from_ ~to_ (Either.First body)
    | Set (((Value_at_node_id id | Gradient_at_node_id id) as tensor), idcs, value) ->
        let tensor =
          if Code.is_value_at_node_id tensor then get_value_tensor ctx id else get_grad_tensor ctx id
        in
        let value = loop_float ~name ~env ~num_typ:tensor.num_typ ~is_double:tensor.is_double value in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:tensor.dims in
        let lhs = LValue.access_array tensor.ptr offset in
        Block.assign !current_block lhs value
    | Set_local (id, value) ->
        let lhs, num_typ, is_double = Hashtbl.find_exn locals id in
        let value = loop_float ~name ~env ~num_typ ~is_double value in
        Block.assign !current_block lhs value
    | Comment c -> Block.comment !current_block c
    | Fill { tensor = (Value_at_node_id id | Gradient_at_node_id id) as tensor; value } ->
        let tensor =
          if Code.is_value_at_node_id tensor then get_value_tensor ctx id else get_grad_tensor ctx id
        in
        let size_m_1 = Array.fold tensor.dims ~init:1 ~f:( * ) - 1 in
        let value = loop_float ~name ~env ~num_typ:tensor.num_typ ~is_double:tensor.is_double value in
        let callback offset =
          let lhs = LValue.access_array tensor.ptr offset in
          Block.assign !current_block lhs value
        in
        jit_for_loop ~env (Shape.get_symbol ()) ~from_:0 ~to_:size_m_1 (Either.Second callback)
    | Dynamic_indices { tensor; tensor_idcs; dynamic_idcs; target_dims; body } ->
        jit_dynamic_indices ~name ~env tensor ~tensor_idcs ~dynamic_idcs ~target_dims body
  and loop_float ~name ~env ~num_typ ~is_double value : rvalue =
    let loop = loop_float ~name ~env ~num_typ ~is_double in
    match value with
    | Local_scope ((Scope_id i as id), prec, body) ->
        let typ = Type.get ctx @@ prec_to_kind prec in
        let v_name = "v" ^ Int.to_string i in
        let lvalue = Function.local func typ v_name in
        Hashtbl.add_exn locals ~key:id ~data:(lvalue, typ, prec_is_double prec);
        loop_proc ~name:(name ^ "_at_" ^ v_name) ~env body;
        RValue.lvalue lvalue
    | Get_local id ->
        let lvalue, _typ, _is_double = Hashtbl.find_exn locals id in
        RValue.lvalue lvalue
    | Get (((Value_at_node_id id | Gradient_at_node_id id) as tensor), idcs) ->
        let tensor =
          if Code.is_value_at_node_id tensor then get_value_tensor ctx id else get_grad_tensor ctx id
        in
        let idcs = lookup env idcs in
        let offset = jit_array_offset ctx ~idcs ~dims:tensor.dims in
        RValue.lvalue @@ LValue.access_array tensor.ptr offset
    | Binop (Code.Add, c1, c2) -> RValue.binary_op ctx Plus num_typ (loop c1) (loop c2)
    | Binop (Code.Mul, c1, c2) -> RValue.binary_op ctx Mult num_typ (loop c1) (loop c2)
    | Binop (Code.ToPowOf, c1, c2) when is_double ->
        let base = RValue.cast ctx (loop c1) c_double in
        let expon = RValue.cast ctx (loop c2) c_double in
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "pow") [ base; expon ]) num_typ
    | Binop (Code.ToPowOf, c1, c2) ->
        let base = RValue.cast ctx (loop c1) c_float in
        let expon = RValue.cast ctx (loop c2) c_float in
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "powf") [ base; expon ]) num_typ
    | Binop (Code.Relu_gate, c1, c2) ->
        let cmp = cast_bool num_typ @@ RValue.comparison ctx Lt (RValue.zero ctx num_typ) @@ loop c1 in
        RValue.binary_op ctx Mult num_typ cmp @@ loop c2
    | Binop (Code.Arg2, _, c2) -> loop c2
    | Binop (Code.Arg1, c1, _) -> loop c1
    | Unop (Code.Identity, c) -> loop c
    | Unop (Code.Relu, c) ->
        let cmp = cast_bool num_typ @@ RValue.comparison ctx Lt (RValue.zero ctx num_typ) @@ loop c in
        RValue.binary_op ctx Mult num_typ cmp @@ loop c
    | Constant v -> RValue.double ctx num_typ v
  and jit_for_loop ~env (Shape.Symbol s as symbol) ~from_ ~to_ body : unit =
    let open Gccjit in
    let i = "i" ^ Int.to_string s in
    let index = Function.local func c_index i in
    let env = Map.add_exn env ~key:symbol ~data:(RValue.lvalue index) in
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
  and jit_dynamic_indices ~name ~env ((Value_at_node_id id | Gradient_at_node_id id) as tensor) ~tensor_idcs
      ~dynamic_idcs ~target_dims body =
    let tensor =
      if Code.is_value_at_node_id tensor then get_value_tensor ctx id else get_grad_tensor ctx id
    in
    let env =
      Array.foldi dynamic_idcs ~init:env ~f:(fun provider_dim env (Symbol s as key) ->
          let target_dim = RValue.int ctx c_int @@ target_dims.(provider_dim) in
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
          Map.add_exn ~key ~data env)
    in
    loop_proc ~name ~env body
  in

  loop_proc ~name ~env body;
  !current_block

let jit_ll_prog ~name ctx prog =
  let open Gccjit in
  let fkind = Function.Exported in
  let env = Map.empty (module Shape.Symbol) in
  let msg = ref None in
  let emit_routine proc suffix =
    let name = name ^ suffix in
    let func = Function.create ctx fkind (Type.get ctx Void) name [] in
    let block = Block.create ~name func in
    (let after_proc = jit_code ~name ~env ctx func block proc in
     Block.return_void after_proc;
     if !Code.with_debug then (
       let suf = "-gccjit-debug.c" in
       let f_name =
         if !Code.keep_files_in_run_directory then name ^ suf else Caml.Filename.temp_file (name ^ "-") suf
       in
       Context.dump_to_file ctx ~update_locs:true f_name;
       msg := Some (Stdio.In_channel.read_all f_name)))
    (* match !compiled_session_globals with
       | None ->
         Context.dump_to_file !session_context ~update_locs:true "globals-gccjit-debug.c";
         let globals = Context.compile !session_context in
          compiled_session_globals := Some globals
       | Some _ -> () *);
    let result = Context.compile ctx in
    session_results := result :: !session_results;
    Result.code result name Ctypes.(void @-> returning void)
  in
  let open Ocannl_runtime.Node in
  (match prog with
  | Code.Assign_suspension proc -> most_recent_suspension := Some (emit_routine proc @@ "suspension")
  | Assign_session_step_update proc -> global.session_step_update := Some (emit_routine proc @@ "step_update"));
  !msg

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

let jit_program (prog : Code.program) =
  let open Gccjit in
  let ctx = Context.create_child !session_context in
  Context.set_option ctx Context.Optimization_level !optimization_level;
  (*
  if !Code.with_debug then (
    Context.set_option ctx Context.Keep_intermediates true;
    Context.set_option ctx Context.Dump_everything true);
  *)
  let msg = jit_ll_prog ~name:"" ctx (Code.to_low_level_program prog) in
  Context.release ctx;
  msg

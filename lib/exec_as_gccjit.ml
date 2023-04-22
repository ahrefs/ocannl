open Base

let session_context =
  let open Gccjit.Context in
  let ctx = create () in
  set_option ctx Optimization_level 3;
  ref ctx

type tensor = {
  ptr : Gccjit.rvalue;  (** Pointer to the first value of the underlying array. *)
  dims : int array;  (** Dimensions (shape) of the tensor. *)
  num_typ : Gccjit.type_;
      (** The type of the stored values: [signed char] (corresponds to precision [Byte]),
      [short] (precision [Half]), [float] (precision [Single]), [double] (precision [Double]). *)
}

let value_tensors_cache : (int, tensor) Hashtbl.t = Hashtbl.create (module Int)
let grad_tensors_cache : (int, tensor) Hashtbl.t = Hashtbl.create (module Int)
let session_results : Gccjit.result list ref = ref []
let hoist_dynamic_indices = ref false

let get_tensor acc cache id : tensor =
  let open Ocannl_runtime.Node in
  let open Gccjit in
  let tensor c_typ arr =
    let ptr = Ctypes.bigarray_start Ctypes_static.Genarray arr in
    let dims = Bigarray.Genarray.dims arr in
    let num_typ = Type.(get !session_context c_typ) in
    let ptr = RValue.ptr !session_context (Type.pointer num_typ) ptr in
    (* let () = session_result := Some (Context.compile !session_context) in *)
    { ptr; dims; num_typ }
  in
  let default () =
    let n = get id in
    match acc n with
    | Byte_as_int_nd arr -> tensor Type.Signed_char arr
    | Half_as_int_nd arr -> tensor Type.Short arr
    | Single_nd arr -> tensor Type.Float arr
    | Double_nd arr -> tensor Type.Double arr
  in
  Hashtbl.find_or_add cache id ~default

let get_value_tensor = get_tensor (fun n -> n.value) value_tensors_cache
let get_grad_tensor = get_tensor (fun n -> (Option.value_exn n.form).grad) grad_tensors_cache

let cleanup_session () =
  let open Gccjit in
  List.iter !session_results ~f:Result.release;
  Hashtbl.clear value_tensors_cache;
  Hashtbl.clear grad_tensors_cache;
  Context.release !session_context;
  session_context := Context.create ();
  Context.set_option !session_context Optimization_level 3;
  session_results := []

let jit_array_offset ctx ~idcs ~dims =
  let open Gccjit in
  let c_int = Type.get ctx Type.Int in
  Array.fold2_exn idcs dims ~init:(RValue.zero ctx c_int) ~f:(fun offset idx dim ->
      RValue.binary_op ctx Plus c_int idx @@ RValue.binary_op ctx Mult c_int offset (RValue.int ctx c_int dim))

let jit_code ~name ~env ctx func block (body : unit Code.low_level) : Gccjit.block =
  let open Gccjit in
  let c_int = Type.get ctx Type.Int in
  let lookup ?provider_dim env indices =
    Array.map indices
      ~f:
        Shape.(
          function
          | Fixed_idx i -> RValue.int ctx c_int i
          | Iterator s | Dynamic_recipient s -> Map.find_exn env s
          | Dynamic_provider _ -> Option.value_exn provider_dim)
  in
  let rec loop_proc ~name ~env ~block (body : unit Code.low_level) : Gccjit.block =
    (* TODO: consider matching gccjit's assign_op-style pattern (but probably no benefit). *)
    match body with
    | Code.Lines lines ->
        Array.foldi lines ~init:block ~f:(fun i block line ->
            loop_proc ~name:(name ^ ":" ^ Int.to_string i) ~env ~block line)
    | For_loop { index; from_; to_; body } -> jit_for_loop ~env index ~from_ ~to_ ~block (Either.First body)
    | Set (((Value_at_node_id id | Gradient_at_node_id id) as tensor), idcs, value) ->
        let tensor = if Code.is_value_at_node_id tensor then get_value_tensor id else get_grad_tensor id in
        let value = loop_float ~name ~env value in
        let idcs = lookup env idcs in
        let lhs_offset = jit_array_offset ctx ~idcs ~dims:tensor.dims in
        let lhs = LValue.access_array tensor.ptr lhs_offset in
        Block.assign block lhs value;
        block
    | Comment c ->
        Block.comment block c;
        block
    | Fill { tensor = (Value_at_node_id id | Gradient_at_node_id id) as tensor; value } ->
        let tensor = if Code.is_value_at_node_id tensor then get_value_tensor id else get_grad_tensor id in
        let size_m_1 = Array.fold tensor.dims ~init:1 ~f:( * ) - 1 in
        let value = loop_float ~name ~env value in
        let callback after_body lhs_offset =
          let lhs = LValue.access_array tensor.ptr lhs_offset in
          Block.assign after_body lhs value
        in
        jit_for_loop ~env (Shape.get_symbol ()) ~from_:0 ~to_:size_m_1 ~block (Either.Second callback)
    | Dynamic_indices { tensor; tensor_idcs; dynamic_idcs; target_dims; body } ->
        jit_dynamic_indices ~name ~env ~block tensor ~tensor_idcs ~dynamic_idcs ~target_dims body
  and loop_float ~name ~env value =
    ignore (name, env);
    match value with
    | Get (_, _) -> failwith "NOT IMPLEMENTED"
    | Binop (_, _, _) -> failwith "NOT IMPLEMENTED"
    | Unop (_, _) -> failwith "NOT IMPLEMENTED"
    | Constant _v -> failwith "NOT IMPLEMENTED"
  and jit_for_loop ~env (Shape.Symbol s as symbol) ~from_ ~to_ ~block body : Gccjit.block =
    let open Gccjit in
    let index = Function.local func c_int ("i" ^ Int.to_string s) in
    let env = Map.add_exn env ~key:symbol ~data:(RValue.lvalue index) in
    (* TODO: maybe propagate more informative names *)
    let b_loop_cond = Block.create ~name:"loop_cond" func in
    let b_loop_body = Block.create ~name:"loop_body" func in
    let b_after_loop = Block.create ~name:"after_loop" func in
    Block.assign block index (RValue.int ctx c_int from_);
    Block.jump block b_loop_cond;
    let guard = RValue.comparison ctx Ge (RValue.lvalue index) (RValue.int ctx c_int to_) in
    Block.cond_jump b_loop_cond guard b_after_loop (* on true *) b_loop_body (* on false *);
    let after_body =
      match body with
      | Either.First body -> loop_proc ~name:"" ~env ~block:b_loop_body body
      | Second callback ->
          callback b_loop_body (RValue.lvalue index);
          b_loop_body
    in
    Block.assign_op after_body index Plus (RValue.one ctx c_int);
    Block.jump after_body b_loop_cond;
    b_after_loop
  and jit_dynamic_indices ~name ~env ~block ((Value_at_node_id id | Gradient_at_node_id id) as tensor)
      ~tensor_idcs ~dynamic_idcs ~target_dims body =
    let tensor = if Code.is_value_at_node_id tensor then get_value_tensor id else get_grad_tensor id in
    let env =
      Array.foldi dynamic_idcs ~init:env ~f:(fun provider_dim env (Symbol s as key) ->
          let target_dim = RValue.int ctx c_int @@ target_dims.(provider_dim) in
          let provider_dim = RValue.int ctx c_int provider_dim in
          let idcs = lookup ~provider_dim env tensor_idcs in
          let prov_offset = jit_array_offset ctx ~idcs ~dims:tensor.dims in
          let dyn_index = RValue.lvalue @@ LValue.access_array tensor.ptr prov_offset in
          let dyn_index = RValue.binary_op ctx Modulo tensor.num_typ dyn_index target_dim in
          let data =
            if !hoist_dynamic_indices then (
              let sym_index = Function.local func c_int ("i" ^ Int.to_string s) in
              Block.assign block sym_index dyn_index;
              RValue.lvalue sym_index)
            else dyn_index
          in
          Map.add_exn ~key ~data env)
    in
    loop_proc ~name ~env ~block body
  in

  loop_proc ~name ~env ~block body

let jit_ll_prog ~with_debug ~name ctx prog =
  let open Gccjit in
  let fkind = Function.Exported in
  let env = Map.empty (module Shape.Symbol) in
  let msg = ref None in
  let emit_routine proc suffix =
    let name = name ^ suffix in
    let func = Function.create ctx fkind (Type.get ctx Void) name [] in
    let block = Block.create ~name func in
    let after_proc = jit_code ~name ~env ctx func block proc in
    Block.return_void after_proc;
    let result = Context.compile ctx in
    if with_debug then msg := Some (Function.to_string func);
    session_results := result :: !session_results;
    Result.code result name Ctypes.(void @-> returning void)
  in
  let open Ocannl_runtime.Node in
  (match prog with
  | Code.Perform proc -> emit_routine proc "_init" ()
  | Assign_routine ({ id; field = `Forward }, proc) ->
      (get_form id).forward := Some (emit_routine proc @@ "_forward_" ^ Int.to_string id)
  | Assign_routine ({ id; field = `Backprop }, proc) ->
      (get_form id).backprop := Some (emit_routine proc @@ "_backprop_" ^ Int.to_string id)
  | Assign_suspension proc -> most_recent_suspension := Some (emit_routine proc @@ "_suspension")
  | Assign_session_prepare_step proc ->
      global.session_prepare_step := Some (emit_routine proc @@ "_prepare_step"));
  !msg

let error_message prefix ?extra_error_msg ~contents exc =
  let backtrace = Caml.Printexc.get_backtrace () in
  let exc_str = Caml.Printexc.to_string exc in
  let message = Buffer.create (String.length contents + String.length backtrace + String.length exc_str) in
  let msg = Buffer.add_string message in
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

let jit_program ?(with_debug = true) (prog : Code.program) =
  let open Gccjit in
  let ctx = Context.create_child !session_context in
  Context.set_option ctx Context.Optimization_level 3;
  (* if with_debug then Context.set_option ctx Context.Dump_initial_gimple true; *)
  let msg = jit_ll_prog ~with_debug ~name:"" ctx (Code.to_low_level_program prog) in
  Context.release ctx;
  msg

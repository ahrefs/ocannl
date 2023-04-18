open Base

let session_context = ref @@ Gccjit.Context.create ()

type tensor = {
  ptr: Gccjit.rvalue;
  (** Pointer to the first value of the underlying array. *)
  dims: int array;
  (** Dimensions (shape) of the tensor. *)
  num_typ: Gccjit.type_;
  (** The type of the stored values: [signed char] (corresponds to precision [Byte]),
      [short] (precision [Half]), [float] (precision [Signle]), [double] (precision [Double]). *)
}

let value_tensors_cache: (int, tensor) Hashtbl.t = Hashtbl.create (module Int)
let grad_tensors_cache: (int, tensor) Hashtbl.t = Hashtbl.create (module Int)
let session_results: Gccjit.result list ref = ref []

let get_tensor acc cache id: tensor =
  let open Ocannl_runtime.Node in
  let open Gccjit in
  let tensor c_typ arr =
    let ptr = Ctypes.bigarray_start Ctypes_static.Genarray arr in
    let dims = Bigarray.Genarray.dims arr in
    let num_typ = Type.(get !session_context c_typ) in
    let ptr = RValue.ptr !session_context num_typ ptr in
    {ptr; dims; num_typ} in
  let default() =
    let n = get id in
    match acc n with
    | Byte_as_int_nd arr -> tensor Type.Signed_char arr
    | Half_as_int_nd arr -> tensor Type.Short arr
    | Single_nd arr -> tensor Type.Float arr
    | Double_nd arr -> tensor Type.Double arr in
  Hashtbl.find_or_add cache id ~default

let get_value_tensor = get_tensor (fun n->n.value) value_tensors_cache

let get_grad_tensor = get_tensor (fun n-> (Option.value_exn n.form).grad) grad_tensors_cache

let cleanup_session () =
  let open Gccjit in
  List.iter !session_results ~f:Result.release;
  Hashtbl.clear value_tensors_cache;
  Hashtbl.clear grad_tensors_cache;
  Context.release !session_context;
  session_context := Gccjit.Context.create ();
  session_results := []

let rec jit_code ~name ~env ctx func ~b_initial (body: unit Code.low_level): Gccjit.block =
  let open Gccjit in
  (* TODO: match gccjit's assign_op-style pattern *)
  (match body with
   | Code.Lines lines ->
     Array.foldi lines ~init:b_initial
       ~f:(fun i b_initial line ->
           jit_code ~name:(name^":"^Int.to_string i) ~env ctx func ~b_initial line)
   | Code.For_loop {index; from_; to_; body} ->
     jit_for_loop ~env ctx func index from_ to_ ~b_initial body
   | Code.Set (Value_at_node_id id, idcs, expr) ->
     let tensor = get_value_tensor id in
     ignore (tensor, idcs, expr);
     failwith "NOT IMPLEMENTED"
   | Code.Set (Gradient_at_node_id id, idcs, expr) ->
     let tensor = get_grad_tensor id in
     ignore (tensor, idcs, expr);
     failwith "NOT IMPLEMENTED"
   | Code.Comment c -> Block.comment b_initial c; b_initial
   | Code.Fill {tensor=Value_at_node_id id; value} ->
     let tensor = get_value_tensor id in
     ignore (tensor, value);
     failwith "NOT IMPLEMENTED"
   | Code.Fill {tensor=Gradient_at_node_id id; value} ->
     let tensor = get_value_tensor id in
     ignore (tensor, value);
     failwith "NOT IMPLEMENTED"
   | Dynamic_indices _ -> failwith "NOT IMPLEMENTED"
  )
(* 
  | Code.Value_at_node_id _ -> _
  | Code.Gradient_at_node_id _ -> _
  | Code.Get (_, _) -> _
  | Code.Binop (_, _, _) -> _
  | Code.Unop (_, _) -> _
 *)

and jit_for_loop ~env ctx func (Shape.Symbol s as symbol) from_ to_ ~b_initial body: Gccjit.block =
  let open Gccjit in
  let int_typ = Type.(get ctx Int) in
  let index = Function.local func int_typ ("i"^Int.to_string s) in
  let env = Map.add_exn env ~key:symbol ~data:index in
  (* TODO: maybe propagate more informative names *)
  let b_loop_cond = Block.create ~name:"loop_cond" func in
  let b_loop_body = Block.create ~name:"loop_body" func in
  let b_after_loop = Block.create ~name:"after_loop" func in
  Block.assign b_initial index (RValue.int ctx int_typ from_);
  Block.jump b_initial b_loop_cond;
  let guard = RValue.comparison ctx Ge (RValue.lvalue index) (RValue.int ctx int_typ to_) in
  Block.cond_jump b_loop_cond guard b_after_loop (* on true *) b_loop_body; (* on false *)
  let after_body = jit_code ~name:"" ~env ctx func ~b_initial:b_loop_body body in
  Block.assign_op after_body index Plus (RValue.one ctx int_typ);
  Block.jump after_body b_loop_cond;
  b_after_loop

let jit_ll_prog ~name ctx prog =
  let open Gccjit in
  let fkind = Function.Exported in
  let env = Map.empty (module Shape.Symbol) in
  let emit_routine proc suffix =
    let name = name^suffix in
    let func = Function.create ctx fkind (Type.get ctx Void) name [] in
    let b_initial = Block.create ~name func in
    let after_proc = jit_code ~name ~env ctx func ~b_initial proc in
    Block.return_void after_proc;
    let result = Context.compile ctx in
    Result.code result name Ctypes.(void @-> returning void) in
  let open Ocannl_runtime.Node in
  (match prog with
   | Code.Perform proc -> emit_routine proc "_init" ()
   | Assign_routine ({id; field=`Forward}, proc) ->
     (get_form id).forward := Some (emit_routine proc @@ "_forward_" ^ Int.to_string id)
   | Assign_routine ({id; field=`Backprop}, proc) ->
     (get_form id).backprop := Some (emit_routine proc @@ "_backprop_" ^ Int.to_string id)
   | Assign_suspension proc ->
     most_recent_suspension := Some (emit_routine proc @@ "_suspension")
   | Assign_session_prepare_step proc ->
     global.session_prepare_step := Some (emit_routine proc @@ "_prepare_step")
  )

let error_message prefix ?extra_error_msg ~contents exc =
  let backtrace = Caml.Printexc.get_backtrace() in
  let exc_str = Caml.Printexc.to_string exc in
  let message =
    Buffer.create (String.length contents + String.length backtrace + String.length exc_str) in
  let msg = Buffer.add_string message in
  msg prefix; msg exc_str; msg "\n"; msg backtrace;
  (match extra_error_msg with None -> () | Some extra ->
      msg "\nIn the context of:\n"; msg extra);
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

let jit_program ?(with_debug=true) (prog: Code.program) =
  let open Gccjit in
  let ctx = Context.create_child !session_context in
  Context.set_option ctx Context.Optimization_level 3;
  if with_debug then Context.set_option ctx Context.Dump_initial_gimple true;
  let msg = "" in
  ignore (prog, jit_code);
  jit_ll_prog ~name:"" ctx (Code.to_low_level_program prog);
  Context.release ctx;
  if with_debug then Some msg
  else None

open Base

let session_results: Gccjit.result list ref = ref []

let cleanup_session () =
  List.iter !session_results ~f:Gccjit.Result.release;
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
  | Code.LLCreate {tensor=Value_at_node_id _id; dims=_; init_op=_} -> failwith "NOT IMPLEMENTED"
  | Code.LLCreate {tensor=Gradient_at_node_id _id; dims=_; init_op=_} -> failwith "NOT IMPLEMENTED"
  | Code.LLFetch {tensor=Value_at_node_id _id; fetch_op=_} -> failwith "NOT IMPLEMENTED"
  | Code.LLFetch {tensor=Gradient_at_node_id _id; fetch_op=_} -> failwith "NOT IMPLEMENTED"
  | Code.Unoptimized_set (_, _, _) -> failwith "NOT IMPLEMENTED"
  | Code.Comment c -> Block.comment b_initial c; b_initial
  )
(* 
  | Code.Value_at_node_id _ -> _
  | Code.Gradient_at_node_id _ -> _
  | Code.Unoptimized_get (_, _) -> _
  | Code.Unoptimized_binop (_, _, _) -> _
  | Code.Unoptimized_unop (_, _) -> _
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
  let ctx = Context.create () in
  Context.set_option ctx Context.Optimization_level 3;
  if with_debug then Context.set_option ctx Context.Dump_initial_gimple true;
  let msg = "" in
  ignore (prog, jit_code);
  jit_ll_prog ~name:"" ctx (Code.unoptimized_program prog);
  Context.release ctx;
  if with_debug then Some msg
  else None
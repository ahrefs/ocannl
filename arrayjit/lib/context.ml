open Base
module Asgns = Ir.Assignments
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module BI = Ir.Backend_intf
module Backends_deprecated = Backends

(** Existential wrapper to hide backend types *)
type backend_wrapper =
  | Wrapper : {
      backend :
        (module BI.Backend
           with type buffer_ptr = 'buffer_ptr
            and type dev = 'dev
            and type runner = 'runner
            and type event = 'event
            and type optimize_ctx = 'optimize_ctx);
      device : ('buffer_ptr, 'dev, 'runner, 'event) BI.device;
      device_id : int;
      stream : ('buffer_ptr, 'dev, 'runner, 'event) BI.stream;
      context :
        ('buffer_ptr, ('buffer_ptr, 'dev, 'runner, 'event) BI.stream, 'optimize_ctx) BI.context;
    }
      -> backend_wrapper

type t = {
  backend_wrapper : (backend_wrapper[@sexp.opaque]);
  device_id : int;
  backend_name : string;
  initialized_nodes : Set.M(Tn).t; (* Track which nodes have been initialized *)
}
[@@deriving sexp_of]

type routine = {
  (* TODO: Remove commented out fields if they prove to be unnecessary *)
  context : t;
  task : Ir.Task.t;
  bindings : Idx.lowered_bindings;
  (* name : string; *)
  inputs : Set.M(Tn).t; (* Nodes that need to be initialized before running *)
  outputs : Set.M(Tn).t; (* Nodes that will be initialized after running *)
}

let bindings r = r.bindings
let context r = r.context

(** Create a context from a backend name *)
let create_from_backend_name ~device_id backend_name =
  let backend_module = Backends.fresh_backend ~backend_name () in
  let module Backend = (val backend_module : Ir.Backend_intf.Backend) in
  let device = Backend.get_device ~ordinal:device_id in
  let stream = Backend.new_stream device in
  let context = Backend.make_context ~optimize_ctx:(Backend.empty_optimize_ctx ()) stream in

  let backend_wrapper =
    Wrapper { backend = (module Backend); device; device_id; stream; context }
  in

  { backend_wrapper; device_id = 0; backend_name; initialized_nodes = Set.empty (module Tn) }

let cuda ?device_id () =
  let device_id = Option.value device_id ~default:0 in
  let ctx = create_from_backend_name ~device_id "cuda" in
  { ctx with device_id }

let metal ?device_id () =
  let device_id = Option.value device_id ~default:0 in
  let ctx = create_from_backend_name ~device_id "metal" in
  { ctx with device_id }

let cpu ?threads () =
  let backend_name = match threads with None | Some 1 -> "sync_cc" | Some _ -> "multicore_cc" in
  create_from_backend_name ~device_id:0 backend_name

let auto () =
  (* First check if a backend is configured globally *)
  match Utils.get_global_arg ~arg_name:"backend" ~default:"" with
  | "" ->
      (* No global config, try backends in order of preference *)
      let backends_to_try = [ "metal"; "cuda"; "multicore_cc"; "sync_cc" ] in
      let rec try_backends = function
        | [] -> failwith "No backend available"
        | name :: rest -> (
            try create_from_backend_name ~device_id:0 name with _ -> try_backends rest)
      in
      try_backends backends_to_try
  | backend_name -> (
      (* Use the configured backend *)
      try create_from_backend_name ~device_id:0 backend_name
      with _ -> invalid_arg ("Unknown backend: " ^ backend_name))

let compile ctx comp bindings =
  let (Wrapper wrapper) = ctx.backend_wrapper in
  let module Backend = (val wrapper.backend) in
  (* Compile and link following train.ml pattern *)
  let code = Backend.compile wrapper.context.optimize_ctx bindings comp in
  let backend_routine = Backend.link wrapper.context code in

  (* Calculate required inputs: context nodes minus embedded nodes *)
  (* Following backends.ml line 445 *)
  let context_nodes = Asgns.context_nodes ~use_host_memory:None comp.Asgns.asgns in
  let inputs = Set.diff context_nodes comp.Asgns.embedded_nodes in

  (* Outputs are all nodes written by the computation *)
  let outputs = backend_routine.outputs in

  let updated_wrapper = Wrapper { wrapper with context = backend_routine.context } in

  let updated_ctx = { ctx with backend_wrapper = updated_wrapper } in

  let routine =
    {
      context = updated_ctx;
      task = backend_routine.schedule;
      bindings = backend_routine.bindings;
      (* name = backend_routine.name; *)
      inputs;
      outputs;
    }
  in

  (updated_ctx, routine)

let run ctx routine =
  (* Check that all required inputs are initialized *)
  let missing_inputs = Set.diff routine.inputs ctx.initialized_nodes in
  (if not (Set.is_empty missing_inputs) then
     let missing_names =
       Set.to_list missing_inputs |> List.map ~f:Tn.debug_name |> String.concat ~sep:", "
     in
     failwith (Printf.sprintf "Context.run: required input nodes not initialized: %s" missing_names));

  (* Run the routine's task/schedule *)
  Ir.Task.run routine.task;

  (* Mark outputs as initialized and return updated context *)
  let initialized_nodes = Set.union ctx.initialized_nodes routine.outputs in
  { ctx with initialized_nodes }

let copy ~src ~dst _tnode =
  (* Device-to-device copy *)
  let (Wrapper _src_wrapper) = src.backend_wrapper in

  (* For now, only support same backend type *)
  if not (String.equal src.backend_name dst.backend_name) then
    failwith "Context.copy: cross-backend copy not yet supported";

  (* This is a simplified placeholder - proper implementation needs device_to_device *)
  (* The challenge is that src and dst may have different buffer_ptr types *)
  failwith "Context.copy: not yet implemented - needs proper device_to_device integration"

(* Internal helper - not exposed in interface to maintain invariants *)
let mark_initialized ctx nodes =
  { ctx with initialized_nodes = Set.union ctx.initialized_nodes nodes }

let init_from_host_deprecated ctx tnode =
  let (Wrapper wrapper) = ctx.backend_wrapper in
  let module Backend = (val wrapper.backend) in
  (* Use the backend's init_from_host function *)
  (* This assumes the node has hosted data already set *)
  let new_backend_context = Backend.init_from_host wrapper.context tnode in

  (* Update the wrapper with the new context *)
  let updated_wrapper = Wrapper { wrapper with context = new_backend_context } in

  (* Mark this node as initialized and update context *)
  let updated_ctx = { ctx with backend_wrapper = updated_wrapper } in
  mark_initialized updated_ctx (Set.singleton (module Tn) tnode)

let is_initialized ctx node = Set.mem ctx.initialized_nodes node
let backend_name ctx = ctx.backend_name
let device_id ctx = ctx.device_id

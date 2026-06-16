open Base
module Asgns = Ir.Assignments
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Idx = Ir.Indexing
module BI = Ir.Backend_intf
module Backends_deprecated = Backends

(** Existential wrapper to hide backend types *)
type backend_wrapper =
  | Wrapper : {
      backend :
        (module BI.Backend
           with type dev = 'dev
            and type runner = 'runner
            and type event = 'event
            and type optimize_ctx = 'optimize_ctx);
      device : ('dev, 'runner, 'event) BI.device;
      device_id : int;
      context : ('dev, 'runner, 'event, 'optimize_ctx) BI.context;
    }
      -> backend_wrapper

type compile_frontier = {
  last_writer : int Map.M(Tn).t;
      (** For each tnode, the routine_id of the most recent routine that writes it. *)
  last_readers : Set.M(Int).t Map.M(Tn).t;
      (** For each tnode, the set of routine_ids that read it since the last write. *)
}
(** Immutable compile-time frontier for execution dependency tracking. Each context carries its own
    frontier; only the context returned by [compile] receives the updated frontier. The original
    context is unchanged. This ensures that sibling compiles (from the same context) produce
    independent routines. *)

type execution_ledger = {
  mutable next_id : int;
  routine_names : string Hashtbl.M(Int).t;
  mutable executed : Set.M(Int).t;
}
(** Shared mutable state for execution tracking, allocated once per root context. Shared by
    reference across all contexts derived from the same root. *)

let empty_frontier = { last_writer = Map.empty (module Tn); last_readers = Map.empty (module Tn) }

let create_ledger () =
  { next_id = 0; routine_names = Hashtbl.create (module Int); executed = Set.empty (module Int) }

type t = {
  backend_wrapper : (backend_wrapper[@sexp.opaque]);
  device_id : int;
  backend_name : string;
  initialized_nodes : Set.M(Tn).t;
  frontier : (compile_frontier[@sexp.opaque]);
  ledger : (execution_ledger[@sexp.opaque]);
}
[@@deriving sexp_of]

type routine = {
  context : t;
  task : Ir.Task.t;
  bindings : Idx.lowered_bindings;
  name : string;
  inputs : Set.M(Tn).t;
  outputs : Set.M(Tn).t;
  routine_id : int;
  execution_deps : Set.M(Int).t;
}

let bindings r = r.bindings
let context r = r.context
let routine_id r = r.routine_id
let routine_name r = r.name
let execution_deps r = Set.to_list r.execution_deps
let can_run ctx routine = Set.is_subset routine.execution_deps ~of_:ctx.ledger.executed

(** Create a context from a backend name *)
let create_from_backend_name ~device_id backend_name =
  let backend_module = Backends.fresh_backend ~backend_name () in
  let module Backend = (val backend_module : Ir.Backend_intf.Backend) in
  let device = Backend.get_device ~ordinal:device_id in
  let context = Backend.make_context ~optimize_ctx:(Backend.empty_optimize_ctx ()) device in

  let backend_wrapper =
    Wrapper { backend = (module Backend); device; device_id; context }
  in

  {
    backend_wrapper;
    device_id = 0;
    backend_name;
    initialized_nodes = Set.empty (module Tn);
    frontier = empty_frontier;
    ledger = create_ledger ();
  }

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

  (* Allocate unique ID from shared ledger *)
  let id = ctx.ledger.next_id in
  ctx.ledger.next_id <- id + 1;

  (* Use backend routine's precise access sets for dependency tracking. backend_routine.inputs =
     materialized read-only and read-before-write nodes. backend_routine.outputs = all materialized
     written-to nodes. *)
  let backend_inputs = backend_routine.inputs in
  let backend_outputs = backend_routine.outputs in
  let frontier = ctx.frontier in
  let empty_int_set = Set.empty (module Int) in

  (* RAW: for each backend input, depend on its last writer *)
  let deps =
    Set.fold backend_inputs ~init:empty_int_set ~f:(fun deps tn ->
        match Map.find frontier.last_writer tn with
        | Some writer_id -> Set.add deps writer_id
        | None -> deps)
  in

  (* WAW + WAR: for each backend output, depend on last writer and all last readers *)
  let deps =
    Set.fold backend_outputs ~init:deps ~f:(fun deps tn ->
        let deps =
          match Map.find frontier.last_writer tn with
          | Some writer_id -> Set.add deps writer_id
          | None -> deps
        in
        match Map.find frontier.last_readers tn with
        | Some readers -> Set.union deps readers
        | None -> deps)
  in

  (* Build updated frontier (immutable — only in returned context) *)
  let new_last_writer =
    Set.fold backend_outputs ~init:frontier.last_writer ~f:(fun lw tn ->
        Map.set lw ~key:tn ~data:id)
  in
  let new_last_readers =
    Set.fold backend_outputs ~init:frontier.last_readers ~f:(fun lr tn -> Map.remove lr tn)
  in
  let pure_inputs = Set.diff backend_inputs backend_outputs in
  let new_last_readers =
    Set.fold pure_inputs ~init:new_last_readers ~f:(fun lr tn ->
        let existing = Option.value (Map.find lr tn) ~default:empty_int_set in
        Map.set lr ~key:tn ~data:(Set.add existing id))
  in
  let new_frontier = { last_writer = new_last_writer; last_readers = new_last_readers } in

  (* Register in shared ledger *)
  let name = backend_routine.name in
  Hashtbl.set ctx.ledger.routine_names ~key:id ~data:name;

  (* Required inputs for the initialization check below. Nodes with registered host initialization
     data (ndarray-backed literals, loaded tensors) self-initialize at link time from [Host_inits]
     (gh-ocannl-333), so they are excluded. *)
  let context_nodes = Asgns.context_nodes ~use_host_memory:None comp.Asgns.asgns in
  let inputs =
    Set.filter (Set.diff context_nodes comp.Asgns.embedded_nodes) ~f:(fun tn ->
        not (Ir.Host_inits.mem tn))
  in

  (* Outputs are all nodes written by the computation *)
  let outputs = backend_routine.outputs in

  let updated_wrapper = Wrapper { wrapper with context = backend_routine.context } in
  let updated_ctx = { ctx with backend_wrapper = updated_wrapper; frontier = new_frontier } in

  let routine =
    {
      context = updated_ctx;
      task = backend_routine.schedule;
      bindings = backend_routine.bindings;
      name;
      inputs;
      outputs;
      routine_id = id;
      execution_deps = deps;
    }
  in

  (updated_ctx, routine)

let run ctx routine =
  (* Check that all required inputs are initialized. A node counts as initialized if it was produced
     by a prior routine ([initialized_nodes]) or is already allocated in the running context's device
     buffers ([in_backend]): such inputs are either user-set via [set_values]/[from_host] (which
     write the allocated buffer in place) or zero-initialized at allocation, which is the correct
     identity for read-only accumulators (e.g. gradients). NOTE (Codex P1): this does not distinguish
     a forgotten non-zero data input from a zero-valid accumulator — both are [alloc_zeros]'d
     read-only buffers — so a forgotten data input reads zeros rather than failing. Catching that
     precisely needs per-node "needs-nonzero-init" metadata OCANNL does not currently carry; a
     stricter check produces false positives on read-only accumulator gradients (zero2hero_1of7,
     primitive_ops). *)
  let (Wrapper run_wrapper) = ctx.backend_wrapper in
  let in_backend tn = Map.mem run_wrapper.context.BI.ctx_buffers tn in
  let missing_inputs =
    Set.filter routine.inputs ~f:(fun tn ->
        not (Set.mem ctx.initialized_nodes tn || in_backend tn))
  in
  (if not (Set.is_empty missing_inputs) then
     let missing_names =
       Set.to_list missing_inputs |> List.map ~f:Tn.debug_name |> String.concat ~sep:", "
     in
     failwith (Printf.sprintf "Context.run: required input nodes not initialized: %s" missing_names));

  (* Check execution dependencies *)
  let missing_deps = Set.diff routine.execution_deps ctx.ledger.executed in
  (if not (Set.is_empty missing_deps) then
     let dep_names =
       Set.to_list missing_deps
       |> List.filter_map ~f:(fun dep_id ->
           Option.map (Hashtbl.find ctx.ledger.routine_names dep_id) ~f:(fun n ->
               Printf.sprintf "%s (id=%d)" n dep_id))
       |> String.concat ~sep:", "
     in
     failwith
       (Printf.sprintf "Context.run: routine %s (id=%d) has unexecuted dependencies: %s"
          routine.name routine.routine_id dep_names));

  (* Run the routine's task/schedule *)
  Ir.Task.run routine.task;

  (* Mark executed in shared ledger *)
  ctx.ledger.executed <- Set.add ctx.ledger.executed routine.routine_id;

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
  failwith "Context.copy: not yet implemented - needs proper device_to_device integration"

(* Internal helper - not exposed in interface to maintain invariants *)
let mark_initialized ctx nodes =
  { ctx with initialized_nodes = Set.union ctx.initialized_nodes nodes }

(* {2 On-demand host access (gh-ocannl-333)}

   All CPU-side value access goes through these context-mediated transfers. There is no host copy
   stored on the tensor node, and there is no cache: each call allocates a fresh temporary host
   buffer and performs a device-to-host (or host-to-device) transfer. This is intentionally
   expensive on non-unified-memory backends — callers should batch access rather than poll. *)

(* A fresh temporary host buffer matching the node's (padded) device buffer. *)
let host_buffer (tn : Tn.t) =
  Nd.create_array
    ~debug:("Context host buffer for " ^ Tn.debug_name tn)
    (Lazy.force tn.Tn.prec) ~dims:(Lazy.force tn.Tn.dims) ~padding:(Lazy.force tn.Tn.padding)

(** Whether [tn] has a device buffer allocated in this context. *)
let mem ctx (tn : Tn.t) : bool =
  let (Wrapper wrapper) = ctx.backend_wrapper in
  Map.mem wrapper.context.BI.ctx_buffers tn

(* For-print proxies (gh-ocannl-333 AC 5): when a tensor's node is not materialized in a context,
   [Train.printf] recompiles a copy ([%cd "for_print" =: t]) into a fresh node and registers it here
   as a proxy for the source node, so {!to_host} can read the source's value through the copy. The
   table is keyed by the source node's id and holds the proxy node; it is read-only from [to_host]'s
   point of view and is for printing only — never a general host cache. *)
let for_print_proxies : Tn.t Hashtbl.M(Int).t = Hashtbl.create (module Int)

let register_for_print ~(src : Tn.t) ~(proxy : Tn.t) =
  Hashtbl.set for_print_proxies ~key:src.Tn.id ~data:proxy

(* A deep copy of a host [Ndarray] (same precision, dims, and layout). Used so reads of shared
   initialization buffers hand the caller a private buffer it may mutate. *)
let copy_nd (src : Nd.t) : Nd.t =
  Nd.apply_with_prec
    {
      f =
        (fun prec arr ->
          let dst =
            Bigarray.Genarray.create (Bigarray.Genarray.kind arr) Bigarray.c_layout
              (Bigarray.Genarray.dims arr)
          in
          Bigarray.Genarray.blit arr dst;
          Nd.as_array prec dst);
    }
    src

(** Transfers [tn]'s device buffer into a fresh host [Ndarray] and returns it. Raises if the node is
    not present in the context (and has no host-init data or for-print proxy). *)
let to_host ctx (tn : Tn.t) : Nd.t =
  let (Wrapper wrapper) = ctx.backend_wrapper in
  let module Backend = (val wrapper.backend) in
  (* Ensure pending device writes feeding [tn] have completed before reading it back. *)
  Backend.await wrapper.context.device;
  let nd = host_buffer tn in
  if Backend.to_host wrapper.context tn nd then (
    (* Ensure the device-to-host copy itself has completed before the host buffer is read. *)
    Backend.await wrapper.context.device;
    nd)
  else
    match Ir.Host_inits.find tn with
    | Some init ->
        (* An ndarray-backed literal that is not part of any computation in this context (so it was
           never allocated on the device): its value is its registered host initialization data.
           Return a private copy so a mutating caller (e.g. [set_value]'s read-modify-write) cannot
           corrupt the shared initialization buffer used to initialize other contexts. *)
        copy_nd (Lazy.force init)
    | None -> (
        (* Read through a for-print proxy, if a copy of [tn] was materialized for printing. *)
        match Hashtbl.find for_print_proxies tn.Tn.id with
        | Some proxy when Backend.to_host wrapper.context proxy nd ->
            Backend.await wrapper.context.device;
            nd
        | _ ->
            raise
            @@ Utils.User_error
                 (Printf.sprintf "Context.to_host: node %s is not present in context (backend %s)"
                    (Tn.debug_name tn) ctx.backend_name))

(** Uploads the host buffer [nd] into [tn]'s device buffer, allocating it if needed, and returns a
    context in which [tn] is marked initialized (so a subsequent {!run} reading [tn] succeeds). *)
let from_host ctx (tn : Tn.t) (nd : Nd.t) : t =
  let (Wrapper wrapper) = ctx.backend_wrapper in
  let module Backend = (val wrapper.backend) in
  let ctx =
    if Backend.from_host wrapper.context tn nd then ctx
    else
      let new_backend_context = Backend.init_from_host wrapper.context tn nd in
      let updated_wrapper = Wrapper { wrapper with context = new_backend_context } in
      { ctx with backend_wrapper = updated_wrapper }
  in
  Backend.await wrapper.context.device;
  mark_initialized ctx (Set.singleton (module Tn) tn)

let get_values ctx (tn : Tn.t) : float array =
  let nd = to_host ctx tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  Nd.retrieve_flat_values ?padding nd

let set_values ctx (tn : Tn.t) (values : float array) : t =
  let nd = host_buffer tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  Nd.set_flat_values ?padding nd values;
  from_host ctx tn nd

let get_value ctx (tn : Tn.t) (idx : int array) : float =
  let nd = to_host ctx tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  let idx =
    if Array.length (Lazy.force tn.Tn.dims) = 0 && Array.length idx = 1 then
      if idx.(0) = 0 then [||] else invalid_arg "Context.get_value: index out of bounds"
    else idx
  in
  Nd.get_as_float ?padding nd idx

(* Reads the current device buffer, sets one element, and uploads the whole buffer back, so that
   the other elements are preserved. *)
let set_value ctx (tn : Tn.t) (idx : int array) (v : float) : t =
  let nd = to_host ctx tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  Nd.set_from_float ?padding nd idx v;
  from_host ctx tn nd

let points_1d ?from_axis ~xdim ctx (tn : Tn.t) =
  let nd = to_host ctx tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  Nd.retrieve_1d_points ?from_axis ?padding ~xdim nd

let points_2d ?from_axis ~xdim ~ydim ctx (tn : Tn.t) =
  let nd = to_host ctx tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  Nd.retrieve_2d_points ?from_axis ?padding ~xdim ~ydim nd

let is_initialized ctx node = Set.mem ctx.initialized_nodes node
let backend_name ctx = ctx.backend_name
let device_id ctx = ctx.device_id

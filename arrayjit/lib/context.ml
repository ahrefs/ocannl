open Base
module Asgns = Ir.Assignments
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Idx = Ir.Indexing
module BI = Ir.Backend_intf
module Backends = Backends
module Cc_backend = Cc_backend
module Builtins_cc = Builtins_cc

module Buffer_key = struct
  module T = struct
    type t = int * BI.buffer_loc [@@deriving compare, sexp]
  end

  include T
  include Comparator.Make (T)
end

(* The backend context rides in [Backends.wrapped_context] -- a closed disjunction over the backend
   singletons' context types (no existential): [Backends.query]/[Backends.with_backend] dispatch
   generic operations, and [copy] correlates two of them ([Backends.pair_contexts]) to recover type
   equality for same-backend transfers. Nothing here matches the constructors: each dispatcher goes
   through the one match [Backends] keeps per direction. *)

type compile_frontier = {
  last_writer : int Map.M(Tn).t;
      (** For each tnode, the routine_id of the most recent routine that writes it. *)
  last_readers : Set.M(Int).t Map.M(Tn).t;
      (** For each tnode, the set of routine_ids that read it since the last write. *)
  merge_buffer_writer : (Tn.t * int) option;
      (** The node and synthetic routine id of the most recent merge-buffer transfer in this
          compilation lineage. Unlike [last_writer], this names the transient merge slab write, not
          a write to the node's ordinary context buffer. *)
}
(** Immutable compile-time frontier for execution dependency tracking. Each context carries its own
    frontier; only the context returned by [compile] receives the updated frontier. The original
    context is unchanged. This ensures that sibling compiles (from the same context) produce
    independent routines. *)

type execution_ledger = {
  mutable next_id : int;
  mutable next_context_id : int;
  routine_names : string Hashtbl.M(Int).t;
  mutable executed : Set.M(Int).t;
  mutable current_writers : int Map.M(Buffer_key).t;
      (** The routine whose execution most recently wrote each concrete ordinary buffer. Unlike the
          immutable compile frontier, this follows re-execution order without conflating sibling
          contexts that allocate distinct buffers for the same tnode. External writes remove their
          buffer. *)
  merge_buffer_readers : (int * int) list Hashtbl.M(Int).t;
      (** [(consumer routine id, owning context id)] by synthetic merge-transfer id. Ownership lets
          [release] unregister an abandoned candidate that can no longer execute. *)
  mutable current_merge_buffer_writer : int option;
      (** The synthetic transfer id whose generation currently occupies this lineage's merge slab.
          Replacing it removes the old id from [executed], invalidating its compiled consumers. *)
  mutable poisoned : (string * exn) option;
      (** Set when a launch or synchronization failed in a way that may have left device buffers of
          this lineage partially written (gh-ocannl-536). [Context.run] marks a routine executed
          before the later [sync] can report an asynchronous failure, so the ledger would otherwise
          claim a routine completed that did not. There is no restore API, so the lineage stays
          unusable: every entrypoint re-raises the stored failure, naming the routine. Scratch
          lineages ([timing_ctx]) are separate, so poisoning one does not condemn the caller's
          context. *)
}
(** Shared mutable state for execution tracking, allocated once per root context. Shared by
    reference across all contexts derived from the same root. *)

let empty_frontier =
  {
    last_writer = Map.empty (module Tn);
    last_readers = Map.empty (module Tn);
    merge_buffer_writer = None;
  }

let create_ledger () =
  {
    next_id = 0;
    next_context_id = 1;
    routine_names = Hashtbl.create (module Int);
    executed = Set.empty (module Int);
    current_writers = Map.empty (module Buffer_key);
    merge_buffer_readers = Hashtbl.create (module Int);
    current_merge_buffer_writer = None;
    poisoned = None;
  }

type t = {
  wrapped : (Backends.wrapped_context[@sexp.opaque]);
  context_id : int;
      (** Identity of this logical backend-context child, used only to retire ledger registrations
          when its leaf is released. [run] preserves it; operations that make a child mint one
          through [derive] below, the only constructor of derived contexts. *)
  ordinal : int;
      (** The backend's device ordinal this context runs on -- what {!Backends.make_context} was
          given. NOT {!Ir.Backend_intf.device.device_id}, which is a process-global counter across
          all backends. *)
  initialized_nodes : Set.M(Tn).t;
  frontier : (compile_frontier[@sexp.opaque]);
  ledger : (execution_ledger[@sexp.opaque]);
}
[@@deriving sexp_of]

let backend_name ctx = Backends.backend_name (Backends.wrapped_backend ctx.wrapped)

(* The one constructor for a derived logical context — the result of any operation that makes a
   child (compiling, transfers that touch the wrapped context, decision recording). Minting the
   fresh [context_id] here, and only here, is what lets [release] retire exactly the leaf's own
   ledger registrations: a hand-assembled child that kept its parent's id would retire the parent's
   merge-buffer reads along with its own. Frontier updates ride along when the operation produced
   one. Operations that evolve the SAME logical context ([run], [mark_initialized],
   [record_external_writes]) stay plain record updates: they must preserve the identity. *)
let derive ?frontier ctx wrapped =
  let context_id = ctx.ledger.next_context_id in
  ctx.ledger.next_context_id <- context_id + 1;
  { ctx with wrapped; context_id; frontier = Option.value frontier ~default:ctx.frontier }

let buffer_key ctx tn =
  Backends.query ctx.wrapped
    {
      q =
        (fun _ c ->
          Option.map (Map.find c.BI.ctx_buffers tn) ~f:(fun loc -> (c.BI.device.device_id, loc)));
    }

(* The context's backend as a first-class module, for the queries that read nothing FROM the
   context: [classify_failure], [static_properties] and [hardware_limits] are backend-level
   functions, so they need no type-recovering dispatch over the wrapped context. *)
let backend_module ctx : (module BI.Backend) =
  Backends.backend_module (Backends.wrapped_backend ctx.wrapped)

type task_handle = Ir.Task.t

type routine = {
  context : t;
  task : task_handle;
  bindings : Idx.lowered_bindings;
  name : string;
  inputs : Set.M(Tn).t;
  outputs : Set.M(Tn).t;
  routine_id : int;
  execution_deps : Set.M(Int).t;
  mma : Ir.C_syntax.mma_summary;
  peel : Ir.C_syntax.peel_summary;
  volatility : Ir.C_syntax.volatility_summary;
}

let can_run ctx routine = Set.is_subset routine.execution_deps ~of_:ctx.ledger.executed

(** Create a context from a backend name *)
let create_from_backend_name ~ordinal backend_name =
  let backend = Backends.get_backend ~backend_name () in
  let ledger = create_ledger () in
  {
    wrapped = Backends.make_context ~ordinal backend;
    context_id = 0;
    ordinal;
    initialized_nodes = Set.empty (module Tn);
    frontier = empty_frontier;
    ledger;
  }

let cuda ?ordinal () = create_from_backend_name ~ordinal:(Option.value ordinal ~default:0) "cuda"
let hip ?ordinal () = create_from_backend_name ~ordinal:(Option.value ordinal ~default:0) "hip"
let metal ?ordinal () = create_from_backend_name ~ordinal:(Option.value ordinal ~default:0) "metal"

let cpu ?threads () =
  (* Kernel-level CPU parallelism is automatic on both cc backends (pool-rendered Grid loops, see
     [automatic_cpu_schedule]); [threads] > 1 selects the multidev_cc debugging backend, which
     exposes multiple worker-domain devices. *)
  let backend_name = match threads with None | Some 1 -> "cc" | Some _ -> "multidev_cc" in
  create_from_backend_name ~ordinal:0 backend_name

(* gh-ocannl-536 landing step 5: backend selection is not candidate compilation, so it does not use
   the compile-phase policy — but it used to catch everything, which turned a broken driver, an
   assertion failure, or an interrupt into a silent downgrade to another backend (and, on the
   configured arm, into the misleading "Unknown backend"). Only {!BI.Backend_unavailable} — raised
   by device discovery when the library is not linked in or the driver reports no devices — advances
   to the next candidate; everything else propagates with its original backtrace. *)
let advances_to_next_backend = function BI.Backend_unavailable _ -> true | _ -> false

let auto () =
  (* First check if a backend is configured globally *)
  match Utils.get_global_arg ~arg_name:"backend" ~default:"" with
  | "" ->
      (* No global config, try backends in order of preference *)
      let backends_to_try = [ "metal"; "cuda"; "hip"; "cc" ] in
      let rec try_backends unavailable = function
        | [] ->
            failwith
              ("Context.auto: no backend available; tried "
              ^ String.concat ~sep:", " (List.rev unavailable))
        | name :: rest -> (
            match create_from_backend_name ~ordinal:0 name with
            | ctx -> ctx
            | exception exn when advances_to_next_backend exn ->
                try_backends (Exn.to_string exn :: unavailable) rest)
      in
      try_backends [] backends_to_try
  | backend_name ->
      (* Use the configured backend. An unknown name already raises a message naming it
         ([Backends.get_backend]); an unusable one keeps its own failure rather than being relabeled
         as a spelling mistake. *)
      create_from_backend_name ~ordinal:0 backend_name

let compile_outcome ?name ?lowered_transform ?prelowered ~provenance ?candidate ctx comp bindings =
  (* Compile and link on the wrapped backend context; only backend-independent routine components
     (and, via [with_backend]'s rebuilt constructor, the updated context) escape the dispatch. *)
  let wrapped, backend_outcome =
    Backends.with_backend ctx.wrapped
      {
        f =
          (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) bctx ->
            (* The [Tile_mma] rendering census is collected HERE, once, around this routine's
               codegen (gh-ocannl-626): whether a routine tensorized is a property of the compiled
               routine, not of whichever timing harness remembered to bracket the global. Fissioned
               segments compile inside this bracket, so their kernels land in the same summary. *)
            let ((outcome, mma), peel), volatility =
              (* And the reduction peel's own census (gh-ocannl-733), bracketed the same way and for
                 the same reason: which decision produced a kernel is a property of the compiled
                 routine, not of whichever test remembered to collect it. Likewise the volatility
                 census (gh-ocannl-782): which of this routine's serial accumulations the Metal
                 compiler-bug workaround pinned to memory, and therefore which of them are not
                 register-resident. *)
              Ir.C_syntax.with_volatility_census @@ fun () ->
              Ir.C_syntax.with_peel_census @@ fun () ->
              Ir.C_syntax.with_census (fun () ->
                  Ir.Schedule_outcome.protect ~classify_backend:Backend.classify_failure ~provenance
                    ~phase:Ir.Schedule_outcome.Transform ?candidate (fun () ->
                      let code =
                        Backend.compile ?name ?lowered_transform ?prelowered bctx.BI.optimize_ctx
                          bindings comp
                      in
                      Ir.Schedule_outcome.tag Ir.Schedule_outcome.Backend_link (fun () ->
                          Backend.link bctx code)))
            in
            match outcome with
            | Ok r ->
                ( r.BI.context,
                  Ok
                    ( r.BI.schedule,
                      r.BI.bindings,
                      r.BI.name,
                      r.BI.inputs,
                      r.BI.merge_buffer_input,
                      r.BI.outputs,
                      mma,
                      peel,
                      volatility ) )
            | Error failure -> (bctx, Error failure));
      }
  in
  match backend_outcome with
  | Error failure -> Error failure
  | Ok
      ( task,
        lowered_bindings,
        name,
        backend_inputs,
        merge_buffer_input,
        backend_outputs,
        mma,
        peel,
        volatility ) ->
      (* Allocate unique ID from shared ledger *)
      let id = ctx.ledger.next_id in
      ctx.ledger.next_id <- id + 1;

      (* Use the backend routine's precise access sets for dependency tracking. [backend_inputs] =
         materialized read-only and read-before-write nodes. A merge-buffer input is a separate read
         of the transfer slab: it depends on the synthetic transfer routine recorded by [copy], not
         on a writer of the node's ordinary context buffer. [backend_outputs] = all materialized
         written-to nodes. *)
      let frontier = ctx.frontier in
      let empty_int_set = Set.empty (module Int) in

      (* RAW: for each ordinary input, depend on its last writer. *)
      let deps =
        Set.fold backend_inputs ~init:empty_int_set ~f:(fun deps tn ->
            match Map.find frontier.last_writer tn with
            | Some writer_id -> Set.add deps writer_id
            | None -> deps)
      in

      (* The backend's static merge-node check has already established the same node identity on its
         wrapped context. Mirror that invariant in the execution frontier and attach the consumer to
         the transfer routine that actually filled the slab. *)
      let deps =
        match (merge_buffer_input, frontier.merge_buffer_writer) with
        | None, _ -> deps
        | Some input, Some (transferred, transfer_id) when Tn.equal input transferred ->
            Set.add deps transfer_id
        | Some input, _ ->
            failwith
              (Printf.sprintf
                 "Context.compile: backend accepted merge-buffer input %s without a matching \
                  execution-frontier transfer"
                 (Tn.debug_name input))
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
      let new_frontier =
        { frontier with last_writer = new_last_writer; last_readers = new_last_readers }
      in

      (* Register in shared ledger *)
      Hashtbl.set ctx.ledger.routine_names ~key:id ~data:name;
      let updated_ctx = derive ~frontier:new_frontier ctx wrapped in
      Option.iter merge_buffer_input ~f:(fun _ ->
          let _, transfer_id = Option.value_exn ~here:[%here] frontier.merge_buffer_writer in
          Hashtbl.update ctx.ledger.merge_buffer_readers transfer_id ~f:(fun readers ->
              (id, updated_ctx.context_id) :: Option.value readers ~default:[]));

      (* Required inputs for the initialization check below: the backend routine's materialized
         read-only / read-before-write nodes, resolved against this compile's placements (the
         context-scoped memory-modes split removed the pre-lowering [context_nodes] settlement).
         Nodes with registered host initialization data (ndarray-backed literals, loaded tensors)
         self-initialize at link time from [Host_inits] (gh-ocannl-333), so they are excluded. *)
      let inputs =
        Set.filter (Set.diff backend_inputs comp.Asgns.embedded_nodes) ~f:(fun tn ->
            not (Ir.Host_inits.mem tn))
      in

      (* Outputs are all nodes written by the computation *)
      let outputs = backend_outputs in

      let routine =
        {
          context = updated_ctx;
          task;
          bindings = lowered_bindings;
          name;
          inputs;
          outputs;
          routine_id = id;
          execution_deps = deps;
          mma;
          peel;
          volatility;
        }
      in

      Ok (updated_ctx, routine)

let compile ?name ?lowered_transform ?prelowered ctx comp bindings =
  match
    compile_outcome ?name ?lowered_transform ?prelowered
      ~provenance:Ir.Schedule_outcome.User_schedule ctx comp bindings
  with
  | Ok result -> result
  | Error failure -> Ir.Schedule_outcome.raise_failure failure

(* {2 Failure classification at the launch/sync boundary (gh-ocannl-536)}

   The compile path passes [Backend.classify_failure] into [protect] through the same first-class
   module dispatch as compilation itself. Launch and sync need it too — that is where a driver
   reports a candidate's kernel as unrunnable — but they are not compile-shaped: the public
   [run]/[sync] contracts stay raising APIs, and callers that want typed outcomes (the autotuner)
   wrap them with the classifier this accessor hands out. Without it, a backend's classifier is
   simply never consulted for a launch failure, and every such failure is fatal by phase default. *)
let failure_classifier ctx :
    Ir.Schedule_outcome.phase -> exn -> Ir.Schedule_outcome.classified_cause option =
  let (module Backend) = backend_module ctx in
  Backend.classify_failure

let poisoned_failure ctx =
  Option.map ctx.ledger.poisoned ~f:(fun (name, exn) ->
      Failure
        (Printf.sprintf
           "Context: this execution lineage was poisoned by a failure of routine %s that may have \
            written device buffers; there is no restore API, so it cannot be reused. Original \
            failure: %s"
           name (Exn.to_string exn)))

let check_not_poisoned ctx = Option.iter (poisoned_failure ctx) ~f:raise

(** Undoes [run]'s optimistic execution marking for a routine whose failure is known not to have
    written device buffers, so the next candidate does not inherit a dependency that never ran. *)
let rollback_execution ctx routine_id =
  ctx.ledger.executed <- Set.remove ctx.ledger.executed routine_id;
  ctx.ledger.current_writers <-
    Map.filter ctx.ledger.current_writers ~f:(fun writer_id -> writer_id <> routine_id)

let poison_lineage ctx ~routine_name exn =
  if Option.is_none ctx.ledger.poisoned then ctx.ledger.poisoned <- Some (routine_name, exn)

(* The LINEAGE-WIDE half of [check_runnable]: a poisoned lineage, an uninitialized input, an
   unexecuted dependency. Each is a property of the context and of the computation, identical for
   every candidate a search compiles from it, so a genuine one fails every candidate of every arm at
   once — which is why it must reach the caller rather than being absorbed as a per-candidate
   decline (gh-ocannl-569). Contained, a search whose serial baseline is not dispatched (every GPU
   search, gh-ocannl-532) declines every candidate for this one reason, times nothing, and ships the
   untuned default out of an unusable lineage under a COMPLETED report — the caller never learns
   about the one-line fix. That is the reasoning the poisoned-lineage check was already raised
   outside the [Preflight] region for; these two belong beside it. On the C backends the dispatched
   serial baseline hit this first and took the arm down with it, which is why the divergence was
   CPU-invisible until HIP ran the suite. *)
let check_lineage_runnable ctx routine =
  check_not_poisoned ctx;
  (* Check that all required inputs are initialized. A node counts as initialized if it was produced
     by a prior routine ([initialized_nodes]) or is already allocated in the running context's
     device buffers ([in_backend]): such inputs are either user-set via [set_values]/[from_host]
     (which write the allocated buffer in place) or zero-initialized at allocation, which is the
     correct identity for read-only accumulators (e.g. gradients). NOTE (Codex P1): this does not
     distinguish a forgotten non-zero data input from a zero-valid accumulator — both are read-only
     buffers allocated with [~zero_init:true] — so a forgotten data input reads zeros rather than
     failing. Catching that precisely needs per-node "needs-nonzero-init" metadata OCANNL does not
     currently carry; a stricter check produces false positives on read-only accumulator gradients
     (zero2hero_1of7, primitive_ops). *)
  let ctx_buffers = Backends.query ctx.wrapped { q = (fun _ c -> c.BI.ctx_buffers) } in
  let in_backend tn = Map.mem ctx_buffers tn in
  let missing_inputs =
    Set.filter routine.inputs ~f:(fun tn -> not (Set.mem ctx.initialized_nodes tn || in_backend tn))
  in
  (if not (Set.is_empty missing_inputs) then
     let missing_names =
       Set.to_list missing_inputs |> List.map ~f:Tn.debug_name |> String.concat ~sep:", "
     in
     failwith (Printf.sprintf "Context.run: required input nodes not initialized: %s" missing_names));

  (* Check execution dependencies *)
  let missing_deps = Set.diff routine.execution_deps ctx.ledger.executed in
  if not (Set.is_empty missing_deps) then
    let dep_names =
      Set.to_list missing_deps
      |> List.filter_map ~f:(fun dep_id ->
          Option.map (Hashtbl.find ctx.ledger.routine_names dep_id) ~f:(fun n ->
              Printf.sprintf "%s (id=%d)" n dep_id))
      |> String.concat ~sep:", "
    in
    failwith
      (Printf.sprintf "Context.run: routine %s (id=%d) has unexecuted dependencies: %s" routine.name
         routine.routine_id dep_names)

(* The PER-CANDIDATE half: bind-time validation of launch parameters
   (docs/proposals/signed-index-precision.md) — each bound value must be non-negative, within its
   declared static range, and within the index width. Unlike the lineage checks above this reads the
   bindings the caller just wrote for THIS routine, and candidates differ in their static ranges, so
   one candidate can fail it while its siblings time cleanly. That makes it the half a search should
   contain as a decline (gh-ocannl-564). *)
let check_launch_bindings routine =
  Idx.validate_lowered_bindings ~width64:Utils.settings.large_models routine.bindings

(* The pre-dispatch validation of {!run}, callable on its own (gh-ocannl-550): everything here
   happens BEFORE [Ir.Task.run], so a failure it raises proves the routine was never dispatched and
   the device wrote nothing. Callers that need the two halves apart — the autotuner's timing runs,
   which contain only the per-candidate one — use them directly. *)
let check_runnable ctx routine =
  check_lineage_runnable ctx routine;
  check_launch_bindings routine

let run ctx routine =
  check_runnable ctx routine;

  (* Run the routine's task/schedule *)
  Ir.Task.run routine.task;

  (* Mark executed in shared ledger *)
  ctx.ledger.executed <- Set.add ctx.ledger.executed routine.routine_id;
  ctx.ledger.current_writers <-
    Set.fold routine.outputs ~init:ctx.ledger.current_writers ~f:(fun writers tn ->
        Option.value_map (buffer_key routine.context tn) ~default:writers ~f:(fun key ->
            Map.set writers ~key ~data:routine.routine_id));

  (* Mark outputs as initialized and return updated context *)
  let initialized_nodes = Set.union ctx.initialized_nodes routine.outputs in
  { ctx with initialized_nodes }

let sync ctx =
  check_not_poisoned ctx;
  Backends.query ctx.wrapped
    {
      q =
        (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
          Backend.await c.BI.device);
    }

let static_properties ctx =
  let (module Backend) = backend_module ctx in
  Backend.static_properties ()

let hardware_limits ctx =
  let (module Backend) = backend_module ctx in
  Backend.hardware_limits ()

let codegen_capabilities ctx =
  let (module Backend) = backend_module ctx in
  Backend.codegen_capabilities ()

(* Internal helper - not exposed in interface to maintain invariants *)
let mark_initialized ctx nodes =
  { ctx with initialized_nodes = Set.union ctx.initialized_nodes nodes }

(* A completed/enqueued write outside [Context.run] supersedes the compiled access history for its
   destination nodes. Future routines compiled from the returned context must observe the external
   value, not wait for an older writer that the caller deliberately replaced; likewise an older
   unexecuted reader cannot impose a WAR edge on a write that already happened. The merge-buffer
   frontier is separate storage and is deliberately unaffected. *)
let record_external_writes ctx nodes =
  ctx.ledger.current_writers <-
    Set.fold nodes ~init:ctx.ledger.current_writers ~f:(fun writers tn ->
        Option.value_map (buffer_key ctx tn) ~default:writers ~f:(Map.remove writers));
  let frontier =
    Set.fold nodes ~init:ctx.frontier ~f:(fun frontier tn ->
        {
          frontier with
          last_writer = Map.remove frontier.last_writer tn;
          last_readers = Map.remove frontier.last_readers tn;
        })
  in
  mark_initialized { ctx with frontier } nodes

(* {2 On-demand host access (gh-ocannl-333)}

   All CPU-side value access goes through these context-mediated transfers. There is no host copy
   stored on the tensor node, and there is no cache: each call allocates a fresh temporary host
   buffer and performs a device-to-host (or host-to-device) transfer. This is intentionally
   expensive on non-unified-memory backends — callers should batch access rather than poll. *)

(* A fresh temporary host buffer matching the node's (padded) device buffer. *)
let host_buffer (tn : Tn.t) =
  Nd.create_array
    ~debug:("Context host buffer for " ^ Tn.debug_name tn)
    (Lazy.force tn.Tn.storage_prec) ~dims:(Lazy.force tn.Tn.dims)
    ~padding:(Lazy.force tn.Tn.padding)

(** Whether [tn] has a device buffer allocated in this context. *)
let mem ctx (tn : Tn.t) : bool =
  Backends.query ctx.wrapped { q = (fun _ c -> Map.mem c.BI.ctx_buffers tn) }

let placements ctx =
  Backends.query ctx.wrapped { q = (fun _ c -> c.BI.optimize_ctx.Ir.Low_level.placements) }

(* gh-ocannl-599: [Local] is the unobservable placement class -- routine-scoped scratch the backend
   may keep in registers or on the stack, never a context buffer a routine writes back. Host access
   to such a node is meaningless, and nothing about it looks meaningless from the outside:
   [from_host] allocates a context buffer for it (the [init_from_host] fallback), the routine
   computes into its local storage, and [to_host] then hands back exactly the bytes that were
   uploaded -- plausible numbers no kernel wrote. Both directions therefore refuse.

   The check reads the effective placement, so it fires only where a decision (or a declared [Local]
   intent) exists: a node no routine of this lineage has mentioned is undecided and keeps the
   ordinary behavior, including the "not present in context" refusal. *)
let local_placement ctx (tn : Tn.t) : Tn.provenance option =
  match Tn.Placements.get (placements ctx) tn with Some (Tn.Local, prov) -> Some prov | _ -> None

let refuse_local ~fn ctx (tn : Tn.t) prov =
  raise
  @@ Utils.User_error
       (Printf.sprintf
          "Context.%s: node %s is placed Local in this context's lineage (provenance %s): \
           routine-scoped scratch with no context buffer, so host access to it cannot observe (or \
           reach) what the routines compute. Request materialization -- e.g. \
           Train.set_materialized, Context.decide_materialized, or Tnode.set_observable -- before \
           the first routine using the node is compiled. Backend: %s"
          fn (Tn.debug_name tn) prov (backend_name ctx))

(* For-print proxies (gh-ocannl-333 AC 5): when a tensor's node is not materialized in a context,
   [Train.printf] recompiles a copy ([%cd "for_print" =: t]) into a fresh node and registers it here
   as a proxy for the source node, so {!to_host} can read the source's value through the copy. The
   table is keyed by the source node and holds the proxy node; it is read-only from [to_host]'s
   point of view and is for printing only — never a general host cache. (Keyed by the tnode, whose
   identity is the never-reused [uid]: an id-keyed table here would resolve stale proxies for reused
   ids after [Tensor.unsafe_reinitialize].) *)
let for_print_proxies : Tn.t Hashtbl.M(Tn).t = Hashtbl.create (module Tn)

let register_for_print ~(src : Tn.t) ~(proxy : Tn.t) =
  Hashtbl.set for_print_proxies ~key:src ~data:proxy

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
  check_not_poisoned ctx;
  (* An [\@|] slice view is addressed through its parent (gh-ocannl-293 293a): an eligible slice
     owns no buffer, and an ineligible (copy) slice's value is recomputed from the parent each run.
     Reject direct host reads uniformly -- read the parent tensor instead. [slice_of] is set eagerly
     at construction, so this also covers the window before lowering decides eligibility. *)
  (match Tn.slice_of tn with
  | Some (parent, _) ->
      raise
      @@ Utils.User_error
           (Printf.sprintf
              "Context.to_host: node %s is an @| slice view; read its parent %s instead"
              (Tn.debug_name tn) (Tn.debug_name parent))
  | None -> ());
  let nd = host_buffer tn in
  (* [transfer] awaits pending device writes feeding the node, attempts the device-to-host copy, and
     awaits its completion before the host buffer is read. *)
  let transfer node =
    Backends.query ctx.wrapped
      {
        q =
          (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
            Backend.await c.BI.device;
            if Backend.to_host c node nd then (
              Backend.await c.BI.device;
              true)
            else false);
      }
  in
  (* Read through a for-print proxy, if a copy of [tn] was materialized for printing. *)
  let from_proxy () =
    match Hashtbl.find for_print_proxies tn with
    | Some proxy when transfer proxy -> Some nd
    | _ -> None
  in
  match local_placement ctx tn with
  | Some prov -> (
      (* A [Local] node's own buffer, if the context has one at all, holds only what a host write
         put there, and its host-init data predates every run -- neither is what the routines
         computed. The one honest read is a for-print proxy: a separate, materialized node
         recomputing the value. *)
      match from_proxy () with
      | Some nd -> nd
      | None -> refuse_local ~fn:"to_host" ctx tn prov)
  | None -> (
      if transfer tn then nd
      else
        match Ir.Host_inits.find tn with
        | Some init ->
            (* An ndarray-backed literal that is not part of any computation in this context (so it
               was never allocated on the device): its value is its registered host initialization
               data. Return a private copy so a mutating caller (e.g. [set_value]'s
               read-modify-write) cannot corrupt the shared initialization buffer used to initialize
               other contexts. *)
            copy_nd (Lazy.force init)
        | None -> (
            match from_proxy () with
            | Some nd -> nd
            | None ->
                raise
                @@ Utils.User_error
                     (Printf.sprintf
                        "Context.to_host: node %s is not present in context (backend %s)"
                        (Tn.debug_name tn) (backend_name ctx))))

(** Uploads the host buffer [nd] into [tn]'s device buffer, allocating it if needed, and returns a
    context in which [tn] is marked initialized (so a subsequent {!run} reading [tn] succeeds). *)
let from_host ctx (tn : Tn.t) (nd : Nd.t) : t =
  check_not_poisoned ctx;
  (* Reject direct host writes to an [\@|] slice view (gh-ocannl-293 293a). Critically, [slice_of]
     is set eagerly at construction, so this fires even when the slice has not been lowered yet and
     [alias_of] is still [None] -- without it the [init_from_host] fallback below would allocate a
     fresh detached buffer for the slice that later alias lowering orphans (the host write would
     update neither the parent nor any buffer the kernels read). Write the parent instead. *)
  (match Tn.slice_of tn with
  | Some (parent, _) ->
      raise
      @@ Utils.User_error
           (Printf.sprintf
              "Context.from_host: node %s is an @| slice view; write its parent %s instead"
              (Tn.debug_name tn) (Tn.debug_name parent))
  | None -> ());
  (* gh-ocannl-599: refuse the write too, rather than only reporting on the read. Seeding a [Local]
     node is a no-op from the routine's point of view -- the routine reads its own local storage --
     so the upload is silently lost either way; refusing here is also what keeps the context from
     acquiring a buffer for a node with no observable buffer, which is what made the later read look
     legitimate. *)
  Option.iter (local_placement ctx tn) ~f:(refuse_local ~fn:"from_host" ctx tn);
  (* Interval analysis, Phase B: a host write acts as a writer around the bounds-settlement point --
     pre-settlement it proposes the scanned [min, max] into the node's bounds candidate,
     post-settlement it validates against the settled bounds (or raises). See
     [Tnode.bounds_state]. *)
  Tn.propose_bounds_from_host tn nd;
  let wrapped, () =
    Backends.with_backend ctx.wrapped
      {
        f =
          (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
            (* Await pending device work BEFORE the upload, mirroring [to_host]: backends with
               host-visible (Shared) buffers implement [from_host] as a direct CPU memcpy, which
               already-queued kernels writing [tn] (e.g. a just-scheduled parameter initialization)
               would otherwise execute after and overwrite — [set_values] right after
               [Train.init_params] silently lost its writes on Metal. *)
            Backend.await c.BI.device;
            let c = if Backend.from_host c tn nd then c else Backend.init_from_host c tn nd in
            Backend.await c.BI.device;
            (c, ()));
      }
  in
  record_external_writes (derive ctx wrapped) (Set.singleton (module Tn) tn)

(** Copies [tn]'s device buffer from [src] into [dst] (or into [dst]'s stream's merge buffer for
    [~into_merge_buffer:Copy]), returning the updated destination context. When both contexts come
    from the same backend, {!Backends.pair_contexts} recovers type equality and the copy dispatches
    to the backend's [device_to_device] transfer machinery; otherwise it falls back to a host
    round-trip. *)
let copy ?(into_merge_buffer = BI.No) ~src ~dst tn =
  (* Both lineages, and BEFORE dispatch: the same-backend path runs the transfer schedule directly
     rather than through [to_host], so checking only the host round-trip would let a poisoned source
     export a possibly partially written buffer into a clean lineage — exactly what poisoning is for
     (Codex P2 on PR #256). Reading from a condemned lineage and running transfer work on one are
     both refused. *)
  check_not_poisoned src;
  check_not_poisoned dst;
  (* The fallback also serves nodes with no device buffer in [src]: [to_host] reads host-init
     literals and for-print proxies. A merge buffer cannot be filled host-side, so [Copy] raises
     where the fallback would engage. *)
  let host_roundtrip what =
    match into_merge_buffer with
    | BI.No -> from_host dst tn (to_host src tn)
    | BI.Copy ->
        raise
        @@ Utils.User_error
             (Printf.sprintf "Context.copy: cannot fill the merge buffer with node %s: %s"
                (Tn.debug_name tn) what)
  in
  (* A same-backend merge transfer reads the source node immediately. If that source context was
     obtained by compiling a writer but the writer has not run, device synchronization has no event
     to wait for and the transfer would silently copy the old/zero buffer. This is a cross-lineage
     edge, so it cannot live in the destination routine's integer dependency set; enforce it before
     scheduling the transfer. *)
  let check_merge_source_ready () =
    let current_writer = Option.bind (buffer_key src tn) ~f:(Map.find src.ledger.current_writers) in
    let pending_writer =
      match current_writer with
      | Some writer_id when Set.mem src.ledger.executed writer_id -> None
      | Some writer_id -> Some writer_id
      | None when Set.mem src.initialized_nodes tn -> None
      | None -> (
          match Map.find src.frontier.last_writer tn with
          | Some writer_id when not (Set.mem src.ledger.executed writer_id) -> Some writer_id
          | _ -> None)
    in
    match pending_writer with
    | Some writer_id ->
        let writer_name =
          Option.value (Hashtbl.find src.ledger.routine_names writer_id) ~default:"<unknown>"
        in
        failwith
          (Printf.sprintf
             "Context.copy: cannot fill the merge buffer from node %s before source writer %s \
              (id=%d) executes"
             (Tn.debug_name tn) writer_name writer_id)
    | _ -> ()
  in
  (* The merge buffer is single-tenant. Compiling a consumer registers it against the current
     transfer generation; overwriting before every such reader has executed would leave that
     consumer runnable while changing the bytes it will read. *)
  let check_merge_destination_ready () =
    Option.iter dst.ledger.current_merge_buffer_writer ~f:(fun transfer_id ->
        let readers =
          Option.value (Hashtbl.find dst.ledger.merge_buffer_readers transfer_id) ~default:[]
        in
        let pending =
          List.filter_map readers ~f:(fun (reader_id, _) ->
              if Set.mem dst.ledger.executed reader_id then None else Some reader_id)
          |> Set.of_list (module Int)
        in
        if not (Set.is_empty pending) then
          let pending_names =
            Set.to_list pending
            |> List.map ~f:(fun reader_id ->
                Printf.sprintf "%s (id=%d)"
                  (Option.value
                     (Hashtbl.find dst.ledger.routine_names reader_id)
                     ~default:"<unknown>")
                  reader_id)
            |> String.concat ~sep:", "
          in
          failwith
            (Printf.sprintf
               "Context.copy: cannot overwrite the merge buffer before pending consumers execute: \
                %s"
               pending_names))
  in
  (* [Context.copy] schedules transfers eagerly, so by the time it returns this synthetic routine is
     executed. Keeping its id in the immutable destination frontier still gives a later merge-buffer
     consumer the real read edge; it also preserves sibling-compilation semantics. *)
  let record_merge_transfer dst ~name =
    Option.iter dst.ledger.current_merge_buffer_writer ~f:(fun old_transfer_id ->
        dst.ledger.executed <- Set.remove dst.ledger.executed old_transfer_id;
        Hashtbl.remove dst.ledger.merge_buffer_readers old_transfer_id);
    let transfer_id = dst.ledger.next_id in
    dst.ledger.next_id <- transfer_id + 1;
    Hashtbl.set dst.ledger.routine_names ~key:transfer_id ~data:name;
    dst.ledger.executed <- Set.add dst.ledger.executed transfer_id;
    dst.ledger.current_merge_buffer_writer <- Some transfer_id;
    { dst with frontier = { dst.frontier with merge_buffer_writer = Some (tn, transfer_id) } }
  in
  let same (type d r e) (impl : (d, r, e) Backends.backend_impl) (sctx : (d, r, e) BI.context)
      (dctx : (d, r, e) BI.context) =
    let (module Backend) = impl.Backends.bi_module in
    let rewrap = impl.Backends.bi_wrap in
    match Backend.device_to_device tn ~into_merge_buffer ~dst:dctx ~src:sctx with
    | Some r -> (
        (* The transfer routine's schedule is ordered on [dst]'s stream; host reads await the device
           as usual. For [Copy], the rewrapped [r.context] is what carries [merge_buffer_node = Some
           tn] into the next [compile]'s static merge-node check (gh-ocannl-288). *)
        (match into_merge_buffer with
        | BI.No -> ()
        | BI.Copy ->
            check_merge_source_ready ();
            check_merge_destination_ready ());
        Ir.Task.run r.BI.schedule;
        let dst = derive dst (rewrap r.BI.context) in
        match into_merge_buffer with
        | BI.No -> record_external_writes dst (Set.singleton (module Tn) tn)
        | BI.Copy -> record_merge_transfer dst ~name:r.BI.name)
    | None ->
        if not (Map.mem sctx.BI.ctx_buffers tn) then
          host_roundtrip "the node is absent from the source context"
        else if not (Map.mem dctx.BI.ctx_buffers tn) then
          (* Present in [src], absent in [dst]: allocate in [dst] and schedule the copy. *)
          record_external_writes
            (derive dst (rewrap (Backend.init_from_device tn ~dst:dctx ~src:sctx)))
            (Set.singleton (module Tn) tn)
        else
          (* The source and destination buffers are physically the same: nothing to transfer. *)
          mark_initialized dst (Set.singleton (module Tn) tn)
  in
  match Backends.pair_contexts src.wrapped dst.wrapped with
  | Backends.Same_backend (impl, s, d) -> same impl s d
  | Backends.Cross_backend ->
      host_roundtrip
        (Printf.sprintf "cross-backend transfer (%s to %s)" (backend_name src) (backend_name dst))

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

(* Reads the current device buffer, sets one element, and uploads the whole buffer back, so that the
   other elements are preserved. *)
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
let ordinal ctx = ctx.ordinal

let get_used_memory ctx =
  Backends.query ctx.wrapped
    {
      q =
        (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
          Backend.get_used_memory c.BI.device);
    }

let release ctx =
  Backends.query ctx.wrapped
    {
      q =
        (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
          Backends.finalize (module Backend) c);
    };
  Hashtbl.to_alist ctx.ledger.merge_buffer_readers
  |> List.iter ~f:(fun (transfer_id, readers) ->
      let remaining =
        List.filter readers ~f:(fun (_, context_id) -> context_id <> ctx.context_id)
      in
      if List.is_empty remaining then Hashtbl.remove ctx.ledger.merge_buffer_readers transfer_id
      else Hashtbl.set ctx.ledger.merge_buffer_readers ~key:transfer_id ~data:remaining)

(* gh-560: the analyze-only entry points — lowering and optimization without backend codegen or
   linking. [Backends.lower_assignments] forks the lineage state itself, so the result is read off a
   hermetic sibling: the argument context, its ledger and frontier are unaffected. With the analysis
   cache (gh-560), a context that already compiled this routine (e.g. the tuner's arms) pays only
   the [specialize_proc] replay here. *)
let lowered_for_decisions ?name ?(materialized = []) ?(inline = []) ctx comp bindings =
  let optim_ctx = Backends.query ctx.wrapped { q = (fun _ c -> c.BI.optimize_ctx) } in
  let optim_ctx = Ir.Low_level.copy_optimize_ctx optim_ctx in
  (* The same decision recording as [decide_materialized] / [decide_inline] below, applied to the
     hermetic fork rather than a child context. *)
  Ir.Low_level.decide_materialized optim_ctx materialized;
  List.iter inline ~f:(Hash_set.add optim_ctx.Ir.Low_level.inline_preferences);
  let _name, (lowered : Ir.Low_level.optimized) =
    Backends.lower_assignments optim_ctx ?name bindings comp.Asgns.asgns
  in
  lowered

let decision_surface ?name ctx comp bindings =
  (lowered_for_decisions ?name ctx comp bindings).Ir.Low_level.flip_candidates

let decide_materialized ctx tns =
  let wrapped, () =
    Backends.with_backend ctx.wrapped
      {
        f =
          (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) bctx ->
            (* Fork the lineage state exactly like a compile would, then record the decisions in the
               fork: the argument context and its other descendants are unaffected. *)
            let optimize_ctx = Ir.Low_level.copy_optimize_ctx bctx.BI.optimize_ctx in
            Ir.Low_level.decide_materialized optimize_ctx tns;
            (Backend.make_child ~optimize_ctx bctx, ()));
      }
  in
  derive ctx wrapped

let decide_inline ctx tns =
  let wrapped, () =
    Backends.with_backend ctx.wrapped
      {
        f =
          (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) bctx ->
            (* Fork like [decide_materialized]; the preference is recorded rather than a placement
               decided, because inlining legality is settled only during optimization
               ([check_and_store_virtual]) — a preferred node the virtualizer rejects still
               materializes. A node whose placement THIS lineage already decided (e.g. a cap
               materialization from an earlier compile of a routine setting it) keeps that decision:
               decisions are final within a lineage — already-compiled routines depend on them (a
               consumer compiled against the node's buffer must find it written) — so the preference
               only steers placements not yet decided. Callers wanting the exemption to take effect
               fork a pre-compile sibling, as [Train.tune_placements] does. *)
            let optimize_ctx = Ir.Low_level.copy_optimize_ctx bctx.BI.optimize_ctx in
            List.iter tns ~f:(Hash_set.add optimize_ctx.Ir.Low_level.inline_preferences);
            (Backend.make_child ~optimize_ctx bctx, ()));
      }
  in
  derive ctx wrapped

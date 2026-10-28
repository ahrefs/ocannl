open Base
module Asgns = Ir.Assignments
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Idx = Ir.Indexing
module BI = Ir.Backend_intf
module Backends = Backends
module Cc_backend = Cc_backend
module Builtins_cc = Builtins_cc

(* gh-ocannl-1038: every backend's host-side probes, in one record. Device-free by construction:
   each arm is a function its backend exposes for exactly this, and none of them opens a driver. *)
let host_capabilities () =
  Cc_backend.host_capabilities ()
  @ Cuda_backend_impl.host_capabilities ()
  @ Hip_backend_impl.host_capabilities ()
  @ Metal_backend_impl.host_capabilities ()

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
          buffer. Entries outlive a {!release}, and a released upload leaf's arena tail is handed to
          a survivor at the same location (gh-ocannl-1328); such an entry cannot mislead: only
          [upload_slot] re-issues a location, and both of its callers ([from_host], [copy]) record
          an external write over it, and every writer here is executed ([run] adds both,
          [rollback_execution] removes both), so it could only ever report "ready". *)
  merge_buffer_readers : (int * int) list Hashtbl.M(Int).t;
      (** [(consumer routine id, owning context id)] by synthetic merge-transfer id, one entry per
          release unit that keeps the consumer live (an upload fork adds one, see [derive]).
          Ownership lets [release] unregister an abandoned candidate that can no longer execute; a
          consumer stays pending while any of its entries remains. *)
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
  mutable written_buffers : Set.M(Buffer_key).t;
      (** Every concrete ordinary buffer of this lineage that something has written: a {!run} (its
          outputs), a host upload or a device copy. Unlike [current_writers] it survives external
          writes, because it answers a different question — not "which routine must this one wait
          for" but "has this buffer ever held a value anyone put there", which is what
          [check_lineage_runnable] asks of a read-before-write input (gh-ocannl-1339). Ledger-wide
          and keyed by the concrete buffer rather than by the node or the immutable context value: a
          caller that discarded the context [run] returned does not lose the fact, and a sibling
          that wrote ITS buffer for the same node does not vouch for this one. Shrinks only through
          [rollback_execution], by [fresh_writes]. *)
  fresh_writes : Buffer_key.t list Hashtbl.M(Int).t;
      (** Per routine id, the [written_buffers] entries the routine's latest {!run} introduced —
          what [rollback_execution] withdraws for a run known not to have written. *)
  mutable observed_tokens : int Map.M(Buffer_key).t;
      (** For each device-wide constant buffer this lineage wrote or adopted, the write token it
          left there or adopted ([shared_write_tokens], gh-ocannl-1623). A buffer whose current
          token differs was overwritten by another root since. *)
  token_undo : token_undo list Hashtbl.M(Int).t;
      (** Per routine id, the tokens the routine's latest {!run} minted — what [rollback_execution]
          restores for a run known not to have written. *)
}
(** Shared mutable state for execution tracking, allocated once per root context. Shared by
    reference across all contexts derived from the same root. *)

(* gh-ocannl-1623: write tokens for the buffers independently created roots can SHARE. A node every
   routine of a compile only reads (and that is not [context_owned]) is laid out in its device's
   [constant_buffer_cache], which every root on the device resolves to the same location -- the same
   [Buffer_key] -- while each root keeps a ledger of its own. A write one root makes there
   ([set_values], a device copy, a routine whose later compile writes the node) changes the bytes
   another root's routine reads, and that root's ledger, which saw its own earlier write, still says
   "written": a silent wrong value. Every write into such a buffer mints a fresh token, kept here
   and in the writing ledger's [observed_tokens]; a run whose ledger holds a different token for a
   buffer it reads is refused ([check_lineage_runnable]). Within one root the ledger is shared, so
   its compile siblings always agree and keep sharing as before. Keys are constant-cache locations
   only: they live as long as the device, and no other location is ever shared across roots (pool
   ids are minted per device and never reused). Ordering needs no extra edge: roots on one device
   share its single FIFO runner. *)
and token_undo = {
  tu_key : Buffer_key.t;
  tu_minted : int;
  tu_prev_device : int option;
  tu_prev_observed : int option;
}

let shared_write_tokens : int Map.M(Buffer_key).t ref = ref (Map.empty (module Buffer_key))
let next_write_token = ref 0

(* Mint a token for [key] in [ledger]'s name; the record says how to undo it. *)
let mint_write_token ledger key =
  let token = !next_write_token in
  Int.incr next_write_token;
  let undo =
    {
      tu_key = key;
      tu_minted = token;
      tu_prev_device = Map.find !shared_write_tokens key;
      tu_prev_observed = Map.find ledger.observed_tokens key;
    }
  in
  shared_write_tokens := Map.set !shared_write_tokens ~key ~data:token;
  ledger.observed_tokens <- Map.set ledger.observed_tokens ~key ~data:token;
  undo

(* Undo a mint known not to have written, as far as nothing wrote since. *)
let undo_write_token ledger { tu_key = key; tu_minted; tu_prev_device; tu_prev_observed } =
  let restore m prev =
    match prev with None -> Map.remove m key | Some data -> Map.set m ~key ~data
  in
  if Option.equal Int.equal (Map.find !shared_write_tokens key) (Some tu_minted) then
    shared_write_tokens := restore !shared_write_tokens tu_prev_device;
  if Option.equal Int.equal (Map.find ledger.observed_tokens key) (Some tu_minted) then
    ledger.observed_tokens <- restore ledger.observed_tokens tu_prev_observed

(* Whether another root wrote [key] since [ledger] last wrote or adopted it. *)
let stale_in ledger key =
  match Map.find !shared_write_tokens key with
  | None -> false
  | Some token -> not (Option.equal Int.equal (Map.find ledger.observed_tokens key) (Some token))

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
    written_buffers = Set.empty (module Buffer_key);
    fresh_writes = Hashtbl.create (module Int);
    observed_tokens = Map.empty (module Buffer_key);
    token_undo = Hashtbl.create (module Int);
  }

type t = {
  wrapped : (Backends.wrapped_context[@sexp.opaque]);
  context_id : int;
      (** Identity of this context's release unit, used only to retire ledger registrations when its
          leaf is released. [run] preserves it; [derive] below, the only constructor of derived
          contexts, preserves it for a value of the same backend lifecycle (a linear upload) and
          mints one for a lifecycle child or an upload fork. *)
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

(* The backend facts that decide a derived context's release unit: its release flag, shared along a
   linear upload chain and fresh for a lifecycle child or an upload fork, and its upload lineage,
   shared by an upload fork and fresh for a lifecycle child. *)
let release_flag wrapped = Backends.query wrapped { q = (fun _ c -> c.BI.finalized) }
let upload_lineage wrapped = Backends.query wrapped { q = (fun _ c -> c.BI.upload_arenas) }

(* gh-ocannl-1329: a released context is a dead handle (see {!release}); every public entry point
   that would touch its buffers -- compile, run, sync, the transfers, the decision recorders --
   refuses it here, by its own release flag, rather than reaching a freed pool or pointer. The flag
   is shared exactly by the values that share pools (a linear upload chain), so a still-live sibling
   (a compile sibling, an upload fork) keeps its own flag and keeps working. One atomic read per
   call, at entry points only: nothing below them re-checks.

   gh-ocannl-1462: a released ANCESTOR is refused too. A descendant has a flag of its own, but it
   inherits its ancestors' buffer locations, so releasing an interior node frees pools the
   descendant still binds. The backend context already records the lifecycle chain
   ([BI.context.parent], set by [make_child]) with each node's flag, so the walk below reads it
   rather than keeping a second registry: one atomic read per lifecycle ancestor. An upload fork
   keeps its source's PARENT, not the source, so a fork outliving the chain it forked from is not
   refused. [release] itself does not check: a descendant must still be able to free its own
   pools. *)
let released_ancestor_depth wrapped =
  Backends.query wrapped
    {
      q =
        (fun _ c ->
          let rec walk depth p =
            match p with
            | None -> None
            | Some p when Atomic.get p.BI.finalized -> Some depth
            | Some p -> walk (depth + 1) p.BI.parent
          in
          walk 1 c.BI.parent);
    }

let check_live ~fn ctx =
  if Atomic.get (release_flag ctx.wrapped) then
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Context.%s: context %d (backend %s, device %d) was released by Context.release and is \
             a dead handle: its device buffers are freed. Use a live context -- e.g. a sibling, or \
             recompile from a live ancestor."
            fn ctx.context_id (backend_name ctx) ctx.ordinal);
  Option.iter (released_ancestor_depth ctx.wrapped) ~f:(fun depth ->
      raise
      @@ Utils.User_error
           (Printf.sprintf
              "Context.%s: context %d (backend %s, device %d) descends from a context released by \
               Context.release (its lifecycle ancestor %d level%s up): the device buffers it \
               inherits from that ancestor are freed. Release leaves before their ancestors, or \
               recompile from a live ancestor."
              fn ctx.context_id (backend_name ctx) ctx.ordinal depth
              (if depth = 1 then "" else "s")))

(* The one constructor for a derived logical context — the result of any operation that makes a
   child (compiling, transfers that touch the wrapped context, decision recording). [context_id]
   names a RELEASE UNIT, and the backend context decides which one the result belongs to
   (gh-ocannl-1203).

   A result sharing its argument's release flag is the same unit: an upload continuing a linear
   chain ([Backend_intf.evolve_with_buffer]), or an upload into an existing buffer, which returns
   the backend context unchanged. Releasing either value frees the same pools and sets the same
   flag, so both must retire the same ledger registrations, and the id is preserved: a fresh one
   would strand every merge-buffer reader compiled before the upload, refusing all later refills.

   Any other result is a new unit with a fresh id, so [release] retires exactly the leaf's own
   registrations: a hand-assembled child that kept its parent's id would retire the parent's
   merge-buffer reads along with its own. An upload FORK (same upload lineage, its own flag) is an
   independent leaf keeping the pools of the value it forked, so the readers compiled into that
   value stay live until every leaf holding them is released: their registrations are extended to
   the fork's id.

   Frontier updates ride along when the operation produced one. Operations that evolve the SAME
   logical context without touching the wrapped one ([run], [mark_initialized],
   [record_external_writes]) stay plain record updates. *)
let derive ?frontier ctx wrapped =
  let frontier = Option.value frontier ~default:ctx.frontier in
  if phys_equal (release_flag wrapped) (release_flag ctx.wrapped) then
    { ctx with wrapped; frontier }
  else
    let context_id = ctx.ledger.next_context_id in
    ctx.ledger.next_context_id <- context_id + 1;
    (if phys_equal (upload_lineage wrapped) (upload_lineage ctx.wrapped) then
       let readers = ctx.ledger.merge_buffer_readers in
       Hashtbl.map_inplace readers ~f:(fun registered ->
           List.filter_map registered ~f:(fun (reader_id, owner) ->
               Option.some_if (owner = ctx.context_id) (reader_id, context_id))
           @ registered));
    { ctx with wrapped; context_id; frontier }

let buffer_key ctx tn =
  Backends.query ctx.wrapped
    {
      q =
        (fun _ c ->
          Option.map (Map.find c.BI.ctx_buffers tn) ~f:(fun loc -> (c.BI.device.device_id, loc)));
    }

(* [tn]'s buffer key in [ctx] when that buffer is its device's cached constant for [tn] -- the one
   kind of buffer independently created roots can share (gh-ocannl-1623, [shared_write_tokens]). *)
let shared_constant_key ctx tn =
  Backends.query ctx.wrapped
    {
      q =
        (fun _ c ->
          match Map.find c.BI.ctx_buffers tn with
          | Some loc
            when Option.exists
                   (Hashtbl.find c.BI.device.constant_buffer_cache tn)
                   ~f:(BI.equal_buffer_loc loc) ->
              Some (c.BI.device.device_id, loc)
          | _ -> None);
    }

(* gh-ocannl-1623: how many [BI.context.parent] links [check_live] walks from [ctx]. *)
let lifecycle_depth ctx =
  Backends.query ctx.wrapped
    {
      q =
        (fun _ c ->
          let rec walk depth p =
            match p with None -> depth | Some p -> walk (depth + 1) p.BI.parent
          in
          walk 0 c.BI.parent);
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
  shared_reads : Set.M(Tn).t;
  outputs : Set.M(Tn).t;
  routine_id : int;
  execution_deps : Set.M(Int).t;
  mma : Ir.C_syntax.mma_summary;
  peel : Ir.C_syntax.peel_summary;
  volatility : Ir.C_syntax.volatility_summary;
  segments : Ir.Low_level.optimized list;
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

(* What a backend link hands back, minus the backend-typed context (already rewrapped). *)
type link_payload = {
  lp_task : task_handle;
  lp_bindings : Idx.lowered_bindings;
  lp_name : string;
  lp_inputs : Set.M(Tn).t;
  lp_merge_buffer_input : Tn.t option;
  lp_outputs : Set.M(Tn).t;
}

type census =
  Ir.C_syntax.mma_summary
  * Ir.C_syntax.peel_summary
  * Ir.C_syntax.volatility_summary
  * Ir.Low_level.optimized list

(* gh-ocannl-1440: compiled code kept for a later link on the context it was compiled from. The
   closure holds the backend's code and the pre-link backend context, typed inside the dispatch that
   produced them; neither holds working pools, which a link allocates. *)
type retained = {
  rt_ctx : t;
  rt_comp : Asgns.comp;
  rt_census : census;
  rt_link : unit -> (Backends.wrapped_context * link_payload) Ir.Schedule_outcome.outcome;
  rt_gate : unit -> unit;
      (** The compile's frontend gate (gh-ocannl-1301), which a relink passes again: the session or
          a node may have been discarded since the code was compiled. *)
  rt_relinked : bool Atomic.t;
}

(* gh-ocannl-1440: [Utils.backend_compiles], counted here for code generation and by a backend for a
   device compile it defers to link time. *)
let backend_compiles = Utils.backend_compiles

(* The ledger, frontier and routine record of a link, shared by a compile and a relink. *)
let finish_link ctx comp wrapped
    {
      lp_task = task;
      lp_bindings = lowered_bindings;
      lp_name = name;
      lp_inputs = backend_inputs;
      lp_merge_buffer_input = merge_buffer_input;
      lp_outputs = backend_outputs;
    } ((mma, peel, volatility, segments) : census) =
  (* Allocate unique ID from shared ledger *)
  let id = ctx.ledger.next_id in
  ctx.ledger.next_id <- id + 1;

  (* Use the backend routine's precise access sets for dependency tracking. [backend_inputs] =
     materialized read-only and read-before-write nodes. A merge-buffer input is a separate read of
     the transfer slab: it depends on the synthetic transfer routine recorded by [copy], not on a
     writer of the node's ordinary context buffer. [backend_outputs] = all materialized written-to
     nodes. *)
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
     wrapped context. Mirror that invariant in the execution frontier and attach the consumer to the
     transfer routine that actually filled the slab. *)
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
     context-scoped memory-modes split removed the pre-lowering [context_nodes] settlement). Nodes
     with registered host initialization data (ndarray-backed literals, loaded tensors)
     self-initialize at link time from [Host_inits] (gh-ocannl-333), so they are excluded. *)
  let read_inputs = Set.diff backend_inputs comp.Asgns.embedded_nodes in
  let inputs = Set.filter read_inputs ~f:(fun tn -> not (Ir.Host_inits.mem tn)) in
  (* gh-ocannl-1623: the reads whose buffers may be device-wide constants another root overwrites,
     host-initialized ones included -- [run] refuses a stale one ([shared_write_tokens]). *)
  let shared_reads =
    Set.filter read_inputs ~f:(fun tn -> Option.is_some (shared_constant_key updated_ctx tn))
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
      shared_reads;
      outputs;
      routine_id = id;
      execution_deps = deps;
      mma;
      peel;
      volatility;
      segments;
    }
  in

  (updated_ctx, routine)

let payload_of (r : _ BI.routine) =
  {
    lp_task = r.BI.schedule;
    lp_bindings = r.BI.bindings;
    lp_name = r.BI.name;
    lp_inputs = r.BI.inputs;
    lp_merge_buffer_input = r.BI.merge_buffer_input;
    lp_outputs = r.BI.outputs;
  }

(* The frontend's session gate and stale-node check (gh-ocannl-1301), run by every entry point that
   lowers a program: a tainted inference session lowers nothing, and no program is lowered over a
   node the frontend discarded -- lowering forces the nodes' dims, which for a discarded node would
   fail as an unrelated shape error rather than name the rewind (gh-ocannl-1505). *)
let frontend_gate ?prelowered comp =
  !Tn.session_gate ();
  Option.iter !Tn.validity_check ~f:(fun check ->
      let read, written = Ir.Assignments.collect_nodes_guess_output comp.Ir.Assignments.asgns in
      Set.iter
        (Set.union_list (module Tn) [ read; written; comp.Ir.Assignments.embedded_nodes ])
        ~f:check;
      (* A prelowered program replaces the comp's lowering: its nodes are the ones that run. *)
      Option.iter prelowered ~f:(fun (p : Ir.Low_level.optimized) ->
          Hashtbl.iter_keys p.traced_store ~f:check))

(* The stale-node check alone, for the entry points over one node (host transfers, copies): run
   before anything of [tn] is forced, for the same reason (gh-ocannl-1505). *)
let check_node_valid tn = Option.iter !Tn.validity_check ~f:(fun check -> check tn)

let compile_outcome ?name ?lowered_transform ?prelowered ~provenance ?candidate ?retain ctx comp
    bindings =
  (* Raised, not returned as an outcome: a dead handle is the caller's error, not a property of any
     candidate, and no search over it can succeed. *)
  check_live ~fn:"compile" ctx;
  frontend_gate ?prelowered comp;
  (* Compile and link on the backend context; only backend-independent routine components (and the
     rewrapped context) escape the dispatch. Unwrapped rather than through [Backends.with_backend]
     so that a [retain]ed link (gh-ocannl-1440) can rewrap the context it links later. *)
  match Backends.unwrap ctx.wrapped with
  | Backends.Unwrapped (impl, bctx) -> (
      let module Backend = (val impl.Backends.bi_module) in
      let link code () =
        let r = Ir.Compile_cost.time_link (fun () -> Backend.link bctx code) in
        (impl.Backends.bi_wrap r.BI.context, payload_of r)
      in
      (* The [Tile_mma] rendering census is collected HERE, once, around this routine's codegen
         (gh-ocannl-626): whether a routine tensorized is a property of the compiled routine, not of
         whichever timing harness remembered to bracket the global. Fissioned segments compile
         inside this bracket, so their kernels land in the same summary. *)
      let (((outcome, mma), peel), volatility), segments =
        (* And the reduction peel's own census (gh-ocannl-733), bracketed the same way and for the
           same reason: which decision produced a kernel is a property of the compiled routine, not
           of whichever test remembered to collect it. Likewise the volatility census
           (gh-ocannl-782): which of this routine's serial accumulations the Metal compiler-bug
           workaround pinned to memory, and therefore which of them are not register-resident. And
           the shipped kernel segments themselves (gh-ocannl-1006). *)
        Backends.with_segments_census @@ fun () ->
        Ir.C_syntax.with_volatility_census @@ fun () ->
        Ir.C_syntax.with_peel_census @@ fun () ->
        Ir.C_syntax.with_census (fun () ->
            Ir.Schedule_outcome.protect ~classify_backend:Backend.classify_failure ~provenance
              ~phase:Ir.Schedule_outcome.Transform ?candidate (fun () ->
                let code =
                  Backend.compile ?name ?lowered_transform ?prelowered bctx.BI.optimize_ctx bindings
                    comp
                in
                Utils.count_backend_compile ();
                (code, Ir.Schedule_outcome.tag Ir.Schedule_outcome.Backend_link (link code))))
      in
      match outcome with
      | Error failure -> Error failure
      | Ok (code, (wrapped, payload)) ->
          let census = (mma, peel, volatility, segments) in
          (* After the link, which has allocated the routine's working pools: a [retain] that raises
             leaves the caller no context to release them through, so they are freed here before its
             exception propagates (review round 1). *)
          let release_linked () =
            Backends.query wrapped
              {
                q =
                  (fun (type d r e) ((module B) : (d, r, e) Backends.backend_module) c ->
                    Backends.finalize (module B) c);
              }
          in
          Option.iter retain ~f:(fun keep ->
              try
                keep
                  {
                    rt_ctx = ctx;
                    rt_comp = comp;
                    rt_census = census;
                    rt_link =
                      (fun () ->
                        Ir.Schedule_outcome.protect ~classify_backend:Backend.classify_failure
                          ~provenance ~phase:Ir.Schedule_outcome.Backend_link ?candidate (link code));
                    rt_gate = (fun () -> frontend_gate ?prelowered comp);
                    rt_relinked = Atomic.make false;
                  }
              with exn ->
                let backtrace = Stdlib.Printexc.get_raw_backtrace () in
                (try release_linked () with _ -> ());
                Stdlib.Printexc.raise_with_backtrace exn backtrace);
          Ok (finish_link ctx comp wrapped payload census))

let relink_outcome rt =
  check_live ~fn:"relink" rt.rt_ctx;
  rt.rt_gate ();
  if not (Atomic.compare_and_set rt.rt_relinked false true) then
    invalid_arg "Context.relink_outcome: this retained code was already relinked";
  match rt.rt_link () with
  | Error failure -> Error failure
  | Ok (wrapped, payload) -> Ok (finish_link rt.rt_ctx rt.rt_comp wrapped payload rt.rt_census)

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
    Map.filter ctx.ledger.current_writers ~f:(fun writer_id -> writer_id <> routine_id);
  (* Only the write claims this run introduced: a buffer an earlier writer had filled keeps its
     value, so it stays written. *)
  Option.iter (Hashtbl.find_and_remove ctx.ledger.fresh_writes routine_id) ~f:(fun keys ->
      ctx.ledger.written_buffers <- List.fold keys ~init:ctx.ledger.written_buffers ~f:Set.remove);
  Option.iter (Hashtbl.find_and_remove ctx.ledger.token_undo routine_id) ~f:(fun undos ->
      List.iter undos ~f:(undo_write_token ctx.ledger))

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
  (* Both handles: [ctx] is the lineage value the caller runs in, [routine.context] holds the buffer
     locations the task actually binds. The second is waived for a live [ctx] of the routine's own
     upload lineage: an upload fork derived from [routine.context] reference-counts the pools it
     holds, so it keeps the routine's buffers alive after the chain it forked from is released (and
     [derive] keeps its merge-buffer consumers registered to it, so refusing them would leave them
     blocking refills forever). Any other [ctx] -- the parent, a compile sibling -- does not hold
     those pools. *)
  check_live ~fn:"run" ctx;
  if not (phys_equal (upload_lineage ctx.wrapped) (upload_lineage routine.context.wrapped)) then
    check_live ~fn:"run (the routine's compile context)" routine.context;
  check_not_poisoned ctx;
  (* Check that all required inputs are initialized, in two classes.

     A READ-BEFORE-WRITE input (the routine both reads it first and writes it: a counter, a running
     statistic, a recurrent state) must have held a value someone put there: the very buffer the
     routine binds must have been written, in this lineage, by a run, upload or device copy
     ([written_buffers]). Not the run context's node-level [initialized_nodes]: [ctx] need not be
     [routine.context], and a sibling that seeded its own buffer for the same node says nothing
     about this one. Every path that marks a node initialized also records its buffer, so nothing is
     lost by asking the buffer. Its allocation-time zeros do not count (gh-ocannl-1339): an
     uninitialized read of state the routine then overwrites is a user bug the context can see.
     State a routine is meant to start from zero carries that in its own computation -- an inline
     declaration is an embedded node, excluded from [inputs] at compile ([Train.sgd_one]'s momentum
     is the pattern) -- or is seeded explicitly.

     A READ-ONLY input must have been written the same way, unless allocation-time zero is declared
     a correct value of it ([Tn.is_zero_valid], gh-ocannl-1462): every gradient is marked at
     creation, being an accumulator, and [Tn.mark_zero_valid] marks any other. Before, a read-only
     input was accepted once merely allocated, and since [compile] allocates every input the check
     accepted nearly anything, so a forgotten data input read zeros. Neither the running context's
     allocation nor its node-level [initialized_nodes] vouches for the buffer either: a compile
     sibling that seeded its own buffer for the node says nothing about this one. The mark excuses
     only the read-only class: a read-before-write input is state the routine overwrites, and its
     entry value is the caller's to give. *)
  let accepted tn =
    match buffer_key routine.context tn with
    | None -> false
    | Some key ->
        Set.mem ctx.ledger.written_buffers key
        || ((not (Set.mem routine.outputs tn)) && Tn.is_zero_valid tn)
  in
  let missing_inputs = Set.filter routine.inputs ~f:(fun tn -> not (accepted tn)) in
  (if not (Set.is_empty missing_inputs) then
     let missing_names =
       Set.to_list missing_inputs
       |> List.map ~f:(fun tn ->
           Tn.debug_name tn ^ if Set.mem routine.outputs tn then " (read before written)" else "")
       |> String.concat ~sep:", "
     in
     raise
     @@ Utils.User_error
          (Printf.sprintf
             "Context.run: routine %s: required input nodes not initialized: %s. Seed them (e.g. \
              Context.set_values, or a routine writing them run first) before running; a read-only \
              input whose allocation-time zeros are its correct value can instead be declared so \
              with Ir.Tnode.mark_zero_valid."
             routine.name missing_names));
  (* gh-ocannl-1623: a buffer this lineage did write, or may read as zero or as its host init, but
     that lives in a device-wide constant pool another root has written since: the bytes are that
     root's, not this lineage's ([shared_write_tokens]). Checked after the class above, so a node
     this lineage never wrote is reported as uninitialized, which is the more basic mistake. *)
  let stale_inputs =
    Set.filter routine.shared_reads ~f:(fun tn ->
        Option.exists (shared_constant_key routine.context tn) ~f:(stale_in ctx.ledger))
  in
  (if not (Set.is_empty stale_inputs) then
     let stale_names =
       Set.to_list stale_inputs |> List.map ~f:Tn.debug_name |> String.concat ~sep:", "
     in
     raise
     @@ Utils.User_error
          (Printf.sprintf
             "Context.run: routine %s: input nodes %s live in a device-wide constant buffer that \
              another independently created context wrote after this lineage last wrote or copied \
              it, so they no longer hold this lineage's values. Re-seed them in this lineage (e.g. \
              Context.set_values), copy them from the writer with Context.copy, or give each \
              context its own buffer with Ir.Tnode.set_context_owned before compiling."
             routine.name stale_names));

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
   (docs/proposals/signed-index-precision.md) — each one must have been assigned (gh-ocannl-940:
   linking leaves them unassigned, so a forgotten one is refused rather than run at 0), and each
   bound value must be non-negative, within its declared static range, and within the index width.
   Unlike the lineage checks above this reads the bindings the caller just wrote for THIS routine,
   and candidates differ in their static ranges, so one candidate can fail it while its siblings
   time cleanly. That makes it the half a search should contain as a decline (gh-ocannl-564). *)
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
  let fresh =
    Set.fold routine.outputs ~init:[] ~f:(fun fresh tn ->
        match buffer_key routine.context tn with
        | None -> fresh
        | Some key ->
            ctx.ledger.current_writers <-
              Map.set ctx.ledger.current_writers ~key ~data:routine.routine_id;
            if Set.mem ctx.ledger.written_buffers key then fresh
            else (
              ctx.ledger.written_buffers <- Set.add ctx.ledger.written_buffers key;
              key :: fresh))
  in
  Hashtbl.set ctx.ledger.fresh_writes ~key:routine.routine_id ~data:fresh;
  let minted =
    Set.fold routine.outputs ~init:[] ~f:(fun minted tn ->
        match shared_constant_key routine.context tn with
        | None -> minted
        | Some key -> mint_write_token ctx.ledger key :: minted)
  in
  Hashtbl.set ctx.ledger.token_undo ~key:routine.routine_id ~data:minted;

  (* Mark outputs as initialized and return updated context *)
  let initialized_nodes = Set.union ctx.initialized_nodes routine.outputs in
  { ctx with initialized_nodes }

let sync ctx =
  check_live ~fn:"sync" ctx;
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

let timing_identity ctx =
  Backends.query ctx.wrapped
    {
      q =
        (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
          Backend.timing_identity c.BI.device);
    }

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
  Set.iter nodes ~f:(fun tn ->
      Option.iter (buffer_key ctx tn) ~f:(fun key ->
          ctx.ledger.current_writers <- Map.remove ctx.ledger.current_writers key;
          ctx.ledger.written_buffers <- Set.add ctx.ledger.written_buffers key);
      Option.iter (shared_constant_key ctx tn) ~f:(fun key ->
          ignore (mint_write_token ctx.ledger key : token_undo)));
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

let inherits_computation ctx tn =
  Backends.query ctx.wrapped
    { q = (fun _ c -> Hashtbl.mem c.BI.optimize_ctx.Ir.Low_level.computations tn) }

let stored_components ctx tn =
  Backends.query ctx.wrapped
    {
      q =
        (fun _ c ->
          Option.value_map ~default:0 ~f:List.length
            (Hashtbl.find c.BI.optimize_ctx.Ir.Low_level.computations tn));
    }

let stored_component_total ctx =
  Backends.query ctx.wrapped
    {
      q =
        (fun _ c ->
          Hashtbl.fold c.BI.optimize_ctx.Ir.Low_level.computations ~init:0
            ~f:(fun ~key:_ ~data acc -> acc + List.length data));
    }

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
  (* gh-ocannl-609: when the decision began as a heuristic cap, materializing is not the only remedy
     -- raising the cap can leave the node virtual and pay recompute instead of memory. The
     provenance is what distinguishes that situation, so the message names the setting rather than
     advising materialization unconditionally.

     The advice is deliberately hedged. [decide_placements] records only the FIRST cap that fires
     (each is guarded on the placement still being undecided), and the legality rejections of
     [check_and_store_virtual] / [inline_computation] are asked afterwards -- so raising the named
     setting can merely expose the next cap or a rejection the caps preempted. What the tag supports
     is "this cap is why the decision was taken", not "this cap is the only thing in the way". *)
  let alternative =
    match Ir.Low_level.cap_provenance_setting (Tn.leading_provenance prov) with
    | None -> ""
    | Some setting ->
        Printf.sprintf
          " That decision was a heuristic cap, not a legality or observability verdict, so raising \
           %s may instead leave the node virtual (paying recompute rather than memory) -- though \
           another cap or an inlining rejection can still force materialization."
          setting
  in
  raise
  @@ Utils.User_error
       (Printf.sprintf
          "Context.%s: node %s is placed Local in this context's lineage (provenance %s): \
           routine-scoped scratch with no context buffer, so host access to it cannot observe (or \
           reach) what the routines compute. Request materialization -- e.g. \
           Train.set_materialized, Context.decide_materialized, or Tnode.set_observable -- before \
           the first routine using the node is compiled.%s Backend: %s"
          fn (Tn.debug_name tn) (Tn.provenance_to_string prov) alternative (backend_name ctx))

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
  check_live ~fn:"to_host" ctx;
  check_node_valid tn;
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
  check_live ~fn:"from_host" ctx;
  check_node_valid tn;
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
  check_live ~fn:"copy (source)" src;
  check_live ~fn:"copy (destination)" dst;
  (* Up front: the same-backend path never reaches [to_host]/[from_host]'s check
     (gh-ocannl-1505). *)
  check_node_valid tn;
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
        else (
          (* The source and destination buffers are physically the same: nothing to transfer. Two
             independently rooted contexts can share it (a cached constant buffer) while keeping
             separate ledgers, so the destination's ledger learns the buffer holds a value -- the
             one [check_lineage_runnable] asks of a read-only input (gh-ocannl-1462) -- when the
             source's ledger, or the source context, says it was written. Nothing was written now,
             so the access history stays as it is. *)
          (* gh-ocannl-1623: the destination adopts the buffer's current write token with it, and
             only from a source still holding its own value: a source another root overwrote since
             vouches for nothing, so the destination stays as unwritten (or as stale) as it was. *)
          if
            (Set.mem src.initialized_nodes tn
            || Option.exists (buffer_key src tn) ~f:(Set.mem src.ledger.written_buffers))
            && not (Option.exists (shared_constant_key src tn) ~f:(stale_in src.ledger))
          then
            Option.iter (buffer_key dst tn) ~f:(fun key ->
                dst.ledger.written_buffers <- Set.add dst.ledger.written_buffers key;
                Option.iter (Map.find !shared_write_tokens key) ~f:(fun token ->
                    dst.ledger.observed_tokens <-
                      Map.set dst.ledger.observed_tokens ~key ~data:token));
          mark_initialized dst (Set.singleton (module Tn) tn))
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
  (* Before [host_buffer] forces the node's dims: [from_host] checks only after (gh-ocannl-1505). *)
  check_live ~fn:"set_values" ctx;
  check_node_valid tn;
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

let get_free_and_total_memory ctx =
  Backends.query ctx.wrapped
    {
      q =
        (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
          Backend.get_free_and_total_memory c.BI.device);
    }

type link_delta = Ir.Backend_intf.link_delta = {
  ld_working_bytes : int;
  ld_constant_bytes : int;
  ld_pools : int;
  ld_liveness_planned : bool;
}
[@@deriving sexp_of, equal]

let link_delta_impl ~fn ~probe ?name ?prelowered ctx lowered segments =
  check_live ~fn ctx;
  Backends.query ctx.wrapped
    {
      q =
        (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
          Backend.link_delta c ?name ?prelowered ~probe lowered segments);
    }

let link_delta ?name ?prelowered ctx lowered segments =
  link_delta_impl ~fn:"link_delta" ~probe:false ?name ?prelowered ctx lowered segments

let probe_link_delta ?name ?prelowered ctx lowered segments =
  link_delta_impl ~fn:"probe_link_delta" ~probe:true ?name ?prelowered ctx lowered segments

type link_floor = Ir.Backend_intf.link_floor = {
  lf_working_bytes : int;
  lf_constant_bytes : int;
  lf_pools : int;
  lf_off_device_bytes : int;
  lf_replaceable_bytes : int;
  lf_overlap_bytes : int;
}
[@@deriving sexp_of, equal]

let link_floor_impl ~fn ~probe ?name ?prelowered ctx lowered =
  check_live ~fn ctx;
  Backends.query ctx.wrapped
    {
      q =
        (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
          Backend.link_floor c ?name ?prelowered ~probe lowered);
    }

let link_floor ?name ?prelowered ctx lowered =
  link_floor_impl ~fn:"link_floor" ~probe:false ?name ?prelowered ctx lowered

let probe_link_floor ?name ?prelowered ctx lowered =
  link_floor_impl ~fn:"probe_link_floor" ~probe:true ?name ?prelowered ctx lowered

let global_debug_info ctx =
  Backends.query ctx.wrapped
    {
      q =
        (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) _c ->
          Backend.get_global_debug_info ());
    }

let release ctx =
  let retire_readers () =
    Hashtbl.to_alist ctx.ledger.merge_buffer_readers
    |> List.iter ~f:(fun (transfer_id, readers) ->
        let remaining =
          List.filter readers ~f:(fun (_, context_id) -> context_id <> ctx.context_id)
        in
        if List.is_empty remaining then Hashtbl.remove ctx.ledger.merge_buffer_readers transfer_id
        else Hashtbl.set ctx.ledger.merge_buffer_readers ~key:transfer_id ~data:remaining)
  in
  match
    Backends.query ctx.wrapped
      {
        q =
          (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
            Backends.finalize (module Backend) c);
      }
  with
  | () -> retire_readers ()
  | exception exn ->
      (* gh-ocannl-1445: a release hook (graph execs) that fails runs after the pools were freed and
         leaves the context finalized, so it is a released consumer all the same. *)
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      if Backends.query ctx.wrapped { q = (fun _ c -> Atomic.get c.BI.finalized) } then
        retire_readers ();
      Stdlib.Printexc.raise_with_backtrace exn backtrace

(* gh-560: the analyze-only entry points — lowering and optimization without backend codegen or
   linking. [Backends.lower_assignments] forks the lineage state itself, so the result is read off a
   hermetic sibling: the argument context, its ledger and frontier are unaffected. With the analysis
   cache (gh-560), a context that already compiled this routine (e.g. the tuner's arms) pays only
   the [specialize_proc] replay here. *)
let lowered_for_decisions ?name ?(materialized = []) ?(inline = []) ?(footprint = []) ctx comp
    bindings =
  (* Lowering forces dims like a compile does: the same gate first. *)
  frontend_gate comp;
  let optim_ctx = Backends.query ctx.wrapped { q = (fun _ c -> c.BI.optimize_ctx) } in
  let optim_ctx = Ir.Low_level.copy_optimize_ctx optim_ctx in
  (* The same decision recording as [decide_materialized] / [decide_inline] / [decide_footprint]
     below, applied to the hermetic fork rather than a child context. *)
  Ir.Low_level.decide_materialized optim_ctx materialized;
  Ir.Low_level.prefer_inline optim_ctx inline;
  Ir.Low_level.prefer_footprint optim_ctx footprint;
  let _name, (lowered : Ir.Low_level.optimized) =
    Backends.lower_assignments ~limits:(hardware_limits ctx) optim_ctx ?name bindings
      comp.Asgns.asgns
  in
  lowered

let decision_surface ?name ctx comp bindings =
  (lowered_for_decisions ?name ctx comp bindings).Ir.Low_level.flip_candidates

let decide_materialized ctx tns =
  (* A child of a dead handle would inherit its freed buffer locations under a fresh, unset flag. *)
  check_live ~fn:"decide_materialized" ctx;
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
  check_live ~fn:"decide_inline" ctx;
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
            Ir.Low_level.prefer_inline optimize_ctx tns;
            (Backend.make_child ~optimize_ctx bctx, ()));
      }
  in
  derive ctx wrapped

let decide_footprint ctx tns =
  check_live ~fn:"decide_footprint" ctx;
  let wrapped, () =
    Backends.with_backend ctx.wrapped
      {
        f =
          (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) bctx ->
            (* Fork like [decide_inline]; a preference for the same reasons — the footprint form is
               served per read site by the virtualizer, which may still decline one. *)
            let optimize_ctx = Ir.Low_level.copy_optimize_ctx bctx.BI.optimize_ctx in
            Ir.Low_level.prefer_footprint optimize_ctx tns;
            (Backend.make_child ~optimize_ctx bctx, ()));
      }
  in
  derive ctx wrapped

let decide_reading ctx (reading : Ir.Low_level.reading) tns =
  match reading with
  | `Materialize -> decide_materialized ctx tns
  | `Inline -> decide_inline ctx tns
  | `Footprint -> decide_footprint ctx tns

let decide_readings ?(skip_empty = true) ctx (materialized, inline, footprint) =
  List.fold
    [ (`Materialize, materialized); (`Inline, inline); (`Footprint, footprint) ]
    ~init:ctx
    ~f:(fun c (r, tns) -> if skip_empty && List.is_empty tns then c else decide_reading c r tns)

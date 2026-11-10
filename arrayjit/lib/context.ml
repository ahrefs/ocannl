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

(* A fragment as the lineage records it (gh-ocannl-1724): by its node's uid -- never reused -- and
   side, so a context does not keep the nodes of every fragment it compiled alive for diagnostics
   (the refusals take names from the comp or routine at hand). *)
module Fragment_key = struct
  module T = struct
    type t = int * Asgns.side [@@deriving compare, sexp_of]
  end

  include T
  include Comparator.Make (T)

  let of_fragment ((tn, side) : Asgns.Fragment.t) = (tn.Tn.uid, side)
end

(* A routine's tensor-generated part in one gradient's life cycle (gh-ocannl-1744), as the lineage
   records it: whether it zeroes the gradient, and whether its backprop accumulates into it. *)
type grad_role = { gr_id : int; gr_name : string; gr_zeroes : bool; gr_accumulates : bool }

type compile_frontier = {
  last_writer : int Map.M(Tn).t;
      (** For each tnode, the routine_id of the most recent routine that writes it. *)
  last_readers : Set.M(Int).t Map.M(Tn).t;
      (** For each tnode, the set of routine_ids that read it since the last write. *)
  merge_buffer_writer : (Tn.t * int) option;
      (** The node and synthetic routine id of the most recent merge-buffer transfer in this
          compilation lineage. Unlike [last_writer], this names the transient merge slab write, not
          a write to the node's ordinary context buffer. *)
  owners : (int * string) Map.M(Fragment_key).t;
      (** For each tensor-generated code fragment compiled in this lineage, the routine (id, name)
          that owns it (gh-ocannl-1724). Only a successful link adds to it, and nothing removes from
          it: an external write supersedes a value, not the code that computes it. *)
  lineage_routines : Set.M(Int).t;
      (** The routines compiled in this lineage: this context's compile ancestry. *)
  grad_roles : grad_role list Map.M(Int).t;
      (** For each gradient (by node uid) that a routine of this lineage zeroes or accumulates into
          through tensor-generated code ({!Asgns.field-zeroes}, {!Asgns.field-accumulates}), those
          routines, latest first (gh-ocannl-1744). Like [owners], only a successful link adds to it.
      *)
  fetch_defined : (int * string) Map.M(Int).t;
      (** For each node (by uid) whose forward code a routine of this lineage carries as a [Fetch]
          ([forward_fetched]), that routine (id, name) (gh-ocannl-1876): every run of it
          re-establishes the node. Like [owners], only a successful link adds to it. *)
  overwriters : (int * string) Map.M(Int).t;
      (** For each node (by uid) that a routine of this lineage writes other than by a [Fetch]
          ([overwritten]), the latest such routine (gh-ocannl-1876). Like [owners], only a
          successful link adds to it. *)
}
(** Immutable compile-time frontier for execution dependency tracking. Each context carries its own
    frontier; only the context returned by [compile] receives the updated frontier. The original
    context is unchanged. This ensures that sibling compiles (from the same context) produce
    independent routines. *)

(* gh-ocannl-1762: a lifecycle ancestor a wrapper minted on its caller's behalf and returns only a
   descendant of -- [Train.forward_once]'s initialization context, a memory-budget decision context
   -- which the caller therefore cannot release ({!adopt_ancestor}). [holders] counts the release
   units holding it; [release_ancestor] releases it once none does. *)
type minted_hold = { release_ancestor : unit -> unit; depth : int; holders : int Atomic.t }

(* A release unit's holds, and whether its release gave them back yet (once per unit). *)
type minted_entry = { holds : minted_hold list; mutable given_back : bool }

(* A released writer's values can remain in an ancestor-owned buffer. Keep readiness without
   retaining the dead routine id, until another write or the buffer's retirement replaces it. *)
type buffer_writer = Routine_writer of int | Released_writer

type execution_ledger = {
  mutable next_id : int;
  mutable next_context_id : int;
  minted_holds : minted_entry Hashtbl.M(Int).t;
      (** By release unit ([context_id]), the ancestors the unit holds ({!adopt_ancestor},
          gh-ocannl-1762): every value of the unit -- a [run] result, a linear upload, the routine's
          compile context -- reaches them through the id. An upload fork takes holds of its own
          ([derive]); a lifecycle child none, since it is released before its ancestors. An entry
          goes once its unit's release has released what it no longer holds. *)
  routine_units : Set.M(Int).t Hashtbl.M(Int).t;
      (** Routine ancestry held by each live release unit. Descendants and upload forks each hold
          their ancestry; record copies sharing the release flag share this entry. *)
  routine_holders : int Hashtbl.M(Int).t;
      (** Number of live release units whose ancestry contains this routine. The last release
          retires its per-routine records without undoing writes to surviving buffers. *)
  routine_names : string Hashtbl.M(Int).t;
      (** Names of routines held by live release units, plus the current synthetic merge fill.
          Released routines disappear once no descendant or upload fork holds their ancestry. *)
  mutable executed : Set.M(Int).t;
  mutable current_writers : buffer_writer Map.M(Buffer_key).t;
      (** The routine whose execution most recently wrote each concrete ordinary buffer, or
          [Released_writer] once its last release unit gives it up: readiness without a routine id.
          Unlike the immutable compile frontier, this follows re-execution order without conflating
          sibling contexts that allocate distinct buffers for the same tnode. External writes remove
          their buffer. A {!release} retires the entries of the locations it gives up, as it does
          [written_buffers]'s ([retire_written], gh-ocannl-1762), so the table does not grow by one
          entry per freed buffer for the life of the root. Nothing depended on the entry before
          that: only [upload_slot] re-issues a location (a released upload leaf's arena tail, to a
          survivor, gh-ocannl-1328), and both of its callers ([from_host], [copy]) record an
          external write over it, and every routine writer here is executed ([run] adds both,
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
          that wrote ITS buffer for the same node does not vouch for this one. Shrinks through
          [rollback_execution], by [fresh_writes], and through {!release}, by the locations it gave
          up ([retire_written], gh-ocannl-1738). *)
  fresh_writes : Buffer_key.t list Hashtbl.M(Int).t;
      (** Per routine id, the [written_buffers] entries the routine's latest {!run} introduced —
          what [rollback_execution] withdraws for a run known not to have written. *)
  mutable observed_tokens : int Map.M(Buffer_key).t;
      (** For each device-wide host-literal buffer this lineage wrote or adopted, the write token it
          left there or adopted ([shared_write_tokens], gh-ocannl-1623). A buffer whose current
          token differs was overwritten by another root since. *)
  token_undo : token_undo list Hashtbl.M(Int).t;
      (** Per routine id, the tokens the routine's latest {!run} minted — what [rollback_execution]
          restores for a run known not to have written. *)
  mutable fragment_registry : (int * string) list Map.M(Fragment_key).t;
      (** Routines (id, name) of this root owning each fragment, sibling lineages included
          (gh-ocannl-1724), retired with the last release unit holding each routine. Diagnostic
          only: it lets a requirement refusal name an owner compiled in a sibling lineage. Ownership
          itself is the frontier's [owners] -- a sibling never satisfies a requirement. *)
  mutable zeroing_registry : (int * string) list Map.M(Int).t;
      (** Every routine (id, name) of this root whose computation zeroes each gradient, by the
          gradient's node uid ({!Asgns.field-zeroes}), sibling lineages included (gh-ocannl-1797).
          Diagnostic only, like [fragment_registry], and retired with the last release unit holding
          each routine: it lets the missing-reset hint of [lacks_node_error] name a zeroing compiled
          outside the lineage. Whether a lineage resets a gradient is its frontier's [grad_roles].
      *)
  grad_buffers : grad_buffers Hashtbl.M(Int).t;
      (** Per routine id, the gradient buffers its computation resets and the ones it accumulates
          into without resetting (gh-ocannl-1781), resolved at link -- buffer keys and names, so the
          ledger keeps no node alive. Absent for a routine with neither. *)
  mutable accumulated : int Map.M(Int).t Map.M(Buffer_key).t;
      (** For each gradient buffer, the runs since its last reset of each routine that accumulates
          into it without zeroing it, by routine id (gh-ocannl-1781). A run of a routine zeroing the
          gradient, and a write from outside a run ([record_external_writes]), reset it. Keyed by
          the concrete buffer, like [written_buffers]: a sibling's buffer for the same node is
          another history. Counted per routine, so two routines contributing once each per reset
          (gh-ocannl-1744's common reset) stay legal. *)
  accumulation_undo : accumulation_undo list Hashtbl.M(Int).t;
      (** Per routine id, the [accumulated] entries the routine's latest {!run} changed — what
          [rollback_execution] restores, unless something changed them since. *)
  persist_roles : persist_roles Hashtbl.M(Int).t;
      (** Per routine id, the buffers its computation writes meaning the value to last
          ({!Asgns.field-persists}) and the ones its forward code re-establishes without reading
          them (gh-ocannl-2027), resolved at link. Absent for a routine with none. *)
  mutable persisted : persist_mark Map.M(Buffer_key).t;
      (** For each buffer whose value a run wrote meaning it to last, that write (gh-ocannl-2027),
          until a write from outside a run ([record_external_writes]) replaces the value
          deliberately. A run of a routine whose forward code re-establishes it is refused instead,
          and a run of any other routine does not supersede it: whether a write covers the whole
          buffer is not known here. Keeps the stepping routine's name: its value outlives the
          routine's release. *)
  persist_undo : persist_undo list Hashtbl.M(Int).t;
      (** Per routine id, the [persisted] entries the routine's latest {!run} changed, for
          [rollback_execution]. *)
}
(** Shared mutable state for execution tracking, allocated once per root context. Shared by
    reference across all contexts derived from the same root. *)

(* gh-ocannl-1781: a gradient a routine accumulates into without zeroing it, as the ledger records
   it at link: its buffer, its node's uid (for the lineage's [grad_roles]) and name, and the runs
   one reset of it spans ([Asgns.field-accumulation_runs], one by default). *)
and unreset_accumulation = { ua_key : Buffer_key.t; ua_uid : int; ua_name : string; ua_runs : int }
and grad_buffers = { gb_resets : Buffer_key.t list; gb_accumulates : unreset_accumulation list }

and accumulation_undo = {
  au_key : Buffer_key.t;
  au_before : int Map.M(Int).t option;
  au_after : int Map.M(Int).t option;
}

(* gh-ocannl-2027: a buffer and its node's name, as [persist_roles] records it. *)
and persist_buffer = { pb_key : Buffer_key.t; pb_name : string }
and persist_roles = { pr_persists : persist_buffer list; pr_recomputes : persist_buffer list }
and persist_mark = { pm_routine_id : int; pm_routine : string }

and persist_undo = {
  pu_key : Buffer_key.t;
  pu_before : persist_mark option;
  pu_after : persist_mark option;
}

(* gh-ocannl-1623: write tokens for the buffers independently created roots SHARE. A node every
   routine of a compile only reads (and that is not [context_owned]) is laid out in its device's
   [constant_buffer_cache]. Since gh-ocannl-1654 only a host-initialized literal is cached
   device-wide there, uploaded once and resolved by every root on the device to the same location --
   the same [Buffer_key] -- while each root keeps a ledger of its own; any other constant is cached
   per root, so no write can reach another root through it. Host literals are writable through the
   API all the same, and a write one root makes to one ([set_values], a device copy, a routine whose
   later compile writes the node) changes the bytes another root's routine reads, and that root's
   ledger, which saw its own earlier write, still says "written": a silent wrong value. Every write
   into such a buffer mints a fresh token, kept here and in the writing ledger's [observed_tokens];
   a run whose ledger holds a different token for a buffer it reads is refused
   ([check_lineage_runnable]). Within one root the ledger is shared, so its compile siblings always
   agree and keep sharing as before. Keys are constant-cache locations only, and no other location
   is shared across roots (pool ids are minted per device and never reused); a key goes with its
   pool, which the release of its last binder frees (gh-ocannl-1762, [retire_written]). Ordering
   needs no extra edge: roots on one device share its single FIFO runner. *)
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
    owners = Map.empty (module Fragment_key);
    lineage_routines = Set.empty (module Int);
    grad_roles = Map.empty (module Int);
    fetch_defined = Map.empty (module Int);
    overwriters = Map.empty (module Int);
  }

let create_ledger () =
  {
    next_id = 0;
    next_context_id = 1;
    minted_holds = Hashtbl.create (module Int);
    routine_units = Hashtbl.create (module Int);
    routine_holders = Hashtbl.create (module Int);
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
    fragment_registry = Map.empty (module Fragment_key);
    zeroing_registry = Map.empty (module Int);
    grad_buffers = Hashtbl.create (module Int);
    accumulated = Map.empty (module Buffer_key);
    accumulation_undo = Hashtbl.create (module Int);
    persist_roles = Hashtbl.create (module Int);
    persisted = Map.empty (module Buffer_key);
    persist_undo = Hashtbl.create (module Int);
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

   The ancestors a wrapper handed the unit ([minted_holds], gh-ocannl-1762) follow the same split:
   the same unit keeps its holds, by its id; an upload fork keeps the source's lifecycle parent, so
   it binds those ancestors' buffers too and takes a hold of its own on each, the last holder's
   release releasing them; a lifecycle child takes none, since its ancestor is released after it
   (release leaves first).

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
    if not (Set.is_empty frontier.lineage_routines) then (
      Hashtbl.set ctx.ledger.routine_units ~key:context_id ~data:frontier.lineage_routines;
      Set.iter frontier.lineage_routines ~f:(fun id ->
          Hashtbl.update ctx.ledger.routine_holders id ~f:(fun count ->
              1 + Option.value count ~default:0)));
    let fork = phys_equal (upload_lineage wrapped) (upload_lineage ctx.wrapped) in
    (if fork then
       let readers = ctx.ledger.merge_buffer_readers in
       Hashtbl.map_inplace readers ~f:(fun registered ->
           List.filter_map registered ~f:(fun (reader_id, owner) ->
               Option.some_if (owner = ctx.context_id) (reader_id, context_id))
           @ registered));
    (if fork then
       match Hashtbl.find ctx.ledger.minted_holds ctx.context_id with
       | Some { holds; given_back = false } ->
           List.iter holds ~f:(fun h -> Atomic.incr h.holders);
           Hashtbl.set ctx.ledger.minted_holds ~key:context_id ~data:{ holds; given_back = false }
       | Some { given_back = true; _ } | None -> ());
    { ctx with wrapped; context_id; frontier }

let buffer_key ctx tn =
  Backends.query ctx.wrapped
    {
      q =
        (fun _ c ->
          Option.map (Map.find c.BI.ctx_buffers tn) ~f:(fun loc -> (c.BI.device.device_id, loc)));
    }

let buffer_location ctx tn =
  Option.map (buffer_key ctx tn) ~f:(fun (_, (loc : BI.buffer_loc)) -> (loc.pool_id, loc.offset))

(* [tn]'s buffer key in [ctx] when that buffer is its device's cached host literal for [tn] -- the
   one kind of buffer independently created roots share (gh-ocannl-1623, [shared_write_tokens]):
   every other constant is cached per root (gh-ocannl-1654, [BI.Constant_key]). *)
let shared_constant_key ctx tn =
  Backends.query ctx.wrapped
    {
      q =
        (fun _ c ->
          match Map.find c.BI.ctx_buffers tn with
          | Some loc
            when Option.exists (BI.find_shared_constant c.BI.device tn) ~f:(BI.equal_buffer_loc loc)
            ->
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
  written_first : Set.M(Tn).t;
  read_first_undeclared : Set.M(Tn).t;
  routine_id : int;
  execution_deps : Set.M(Int).t;
  requires : Set.M(Asgns.Fragment).t;
  mma : Ir.C_syntax.mma_summary;
  peel : Ir.C_syntax.peel_summary;
  volatility : Ir.C_syntax.volatility_summary;
  grid : Ir.C_syntax.grid_summary;
  segments : Ir.Low_level.optimized list;
}

let can_run ctx routine = Set.is_subset routine.execution_deps ~of_:ctx.ledger.executed

(** Create a context from a backend name *)
let create_from_backend_name ~ordinal ~backend_name =
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

let cuda ?ordinal () =
  create_from_backend_name ~ordinal:(Option.value ordinal ~default:0) ~backend_name:"cuda"

let hip ?ordinal () =
  create_from_backend_name ~ordinal:(Option.value ordinal ~default:0) ~backend_name:"hip"

let metal ?ordinal () =
  create_from_backend_name ~ordinal:(Option.value ordinal ~default:0) ~backend_name:"metal"

let cpu ?threads () =
  (* Kernel-level CPU parallelism is automatic on both cc backends (pool-rendered Grid loops, see
     [automatic_cpu_schedule]); [threads] > 1 selects the multidev_cc debugging backend, which
     exposes multiple worker-domain devices. *)
  let backend_name = match threads with None | Some 1 -> "cc" | Some _ -> "multidev_cc" in
  create_from_backend_name ~ordinal:0 ~backend_name

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
            match create_from_backend_name ~ordinal:0 ~backend_name:name with
            | ctx -> ctx
            | exception exn when advances_to_next_backend exn ->
                try_backends (Exn.to_string exn :: unavailable) rest)
      in
      try_backends [] backends_to_try
  | backend_name ->
      (* Use the configured backend. An unknown name already raises a message naming it
         ([Backends.get_backend]); an unusable one keeps its own failure rather than being relabeled
         as a spelling mistake. *)
      create_from_backend_name ~ordinal:0 ~backend_name

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
  * Ir.C_syntax.grid_summary
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

(* gh-ocannl-1876: {!Asgns.forward_fetched} over a lineage: [frontier] answers for a parent another
   routine of it fetches. Shapes are settled here, lowering being the next step. *)
let forward_fetched frontier comp =
  Asgns.forward_fetched
    ~fetched_elsewhere:(fun tn -> Map.mem frontier.fetch_defined tn.Tn.uid)
    ~view_aliases:Asgns.slice_alias_eligible comp

(* The nodes [comp] writes other than by a [Fetch]: [Rev_sides]' buffers, any other [Accum_op]'s and
   [Set_vec_unop]'s [lhs]. *)
let overwritten (comp : Asgns.comp) =
  let add acc = function Asgns.Node tn -> Set.add acc tn | Asgns.Merge_buffer _ -> acc in
  Asgns.fold_leaves comp.asgns
    ~init:(Set.empty (module Tn))
    ~f:(fun acc -> function
      | Asgns.Accum_op { rhs = Rev_sides { lhses; _ }; _ } -> Array.fold lhses ~init:acc ~f:add
      | Asgns.Accum_op { lhs; _ } | Asgns.Set_vec_unop { lhs; _ } -> Set.add acc lhs
      | Asgns.Fetch _ | Asgns.Noop | Asgns.Seq _ | Asgns.Block_comment _ -> acc)

(* gh-ocannl-2027: the buffer a write to [tn] reaches -- a confirmed slice alias's parent's. *)
let rec persist_storage tn =
  match Tn.alias_of tn with Some (parent, _) -> persist_storage parent | None -> tn

(* gh-ocannl-2027: the buffers [comp] writes meaning the value to last, and the ones its forward
   code re-establishes without reading them ({!Asgns.forward_recomputed} over a node whose [Fwd]
   fragment [comp] owns), the recomputation [run] refuses over a persisting write's value. A node
   with no buffer in [ctx] -- virtual, [Local] -- holds nothing across runs. *)
let record_persist_roles ctx id (comp : Asgns.comp) =
  let buffers nodes =
    Set.to_list (Set.map (module Tn) nodes ~f:persist_storage)
    |> List.filter_map ~f:(fun tn ->
        Option.map (buffer_key ctx tn) ~f:(fun pb_key ->
            {
              pb_key;
              pb_name = Tn.debug_name tn [@message_only "the refusal names the node by it"];
            }))
  in
  let pr_persists = buffers comp.Asgns.persists in
  let pr_recomputes =
    buffers
      (Set.filter
         (Set.map (module Tn) (Asgns.forward_recomputed comp) ~f:persist_storage)
         ~f:(fun tn -> Set.mem comp.Asgns.owned (tn, Asgns.Fwd)))
  in
  if not (List.is_empty pr_persists && List.is_empty pr_recomputes) then
    Hashtbl.set ctx.ledger.persist_roles ~key:id ~data:{ pr_persists; pr_recomputes }

(* The ledger, frontier and routine record of a link, shared by a compile and a relink. *)
let finish_link ctx comp wrapped
    {
      lp_task = task;
      lp_bindings = lowered_bindings;
      lp_name = name;
      lp_inputs = backend_inputs;
      lp_merge_buffer_input = merge_buffer_input;
      lp_outputs = backend_outputs;
    } ((mma, peel, volatility, grid, segments) : census) =
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
  (* gh-ocannl-1724: the fragments this routine owns join the returned lineage only (a sibling
     compiled from [ctx] does not see them), and the root's diagnostic registry. *)
  let owners =
    Set.fold comp.Asgns.owned ~init:frontier.owners ~f:(fun owners fragment ->
        Map.set owners ~key:(Fragment_key.of_fragment fragment) ~data:(id, name))
  in
  ctx.ledger.fragment_registry <-
    Set.fold comp.Asgns.owned ~init:ctx.ledger.fragment_registry ~f:(fun registry fragment ->
        Map.add_multi registry ~key:(Fragment_key.of_fragment fragment) ~data:(id, name));
  ctx.ledger.zeroing_registry <-
    Set.fold comp.Asgns.zeroes ~init:ctx.ledger.zeroing_registry ~f:(fun registry grad ->
        Map.add_multi registry ~key:grad.Tn.uid ~data:(id, name));
  let grad_roles =
    Set.fold (Set.union comp.Asgns.zeroes comp.Asgns.accumulates) ~init:frontier.grad_roles
      ~f:(fun roles grad ->
        Map.add_multi roles ~key:grad.Tn.uid
          ~data:
            {
              gr_id = id;
              gr_name = name;
              gr_zeroes = Set.mem comp.Asgns.zeroes grad;
              gr_accumulates = Set.mem comp.Asgns.accumulates grad;
            })
  in
  (* gh-ocannl-1876: likewise the nodes this routine fetch-defines and the ones it overwrites. *)
  let record nodes map =
    Set.fold nodes ~init:map ~f:(fun map tn -> Map.set map ~key:tn.Tn.uid ~data:(id, name))
  in
  let new_frontier =
    {
      frontier with
      last_writer = new_last_writer;
      last_readers = new_last_readers;
      owners;
      lineage_routines = Set.add frontier.lineage_routines id;
      grad_roles;
      fetch_defined = record (forward_fetched frontier comp) frontier.fetch_defined;
      overwriters = record (overwritten comp) frontier.overwriters;
    }
  in

  (* Register in shared ledger *)
  Hashtbl.set ctx.ledger.routine_names ~key:id ~data:name;
  let updated_ctx = derive ~frontier:new_frontier ctx wrapped in
  (* gh-ocannl-1781: the gradient buffers this routine resets and accumulates into without zeroing
     -- the ones its context holds; a gradient left [Local] or virtual lives within one run. *)
  (let key tn = buffer_key updated_ctx tn in
   let gb_resets = List.filter_map (Set.to_list comp.Asgns.zeroes) ~f:key in
   let gb_accumulates =
     Set.to_list (Set.diff comp.Asgns.accumulates comp.Asgns.zeroes)
     |> List.filter_map ~f:(fun tn ->
         Option.map (key tn) ~f:(fun ua_key ->
             {
               ua_key;
               ua_uid = tn.Tn.uid;
               ua_name =
                 Tn.debug_name tn [@message_only "the refusal shows it as the gradient's name"];
               ua_runs =
                 Option.value (Map.find comp.Asgns.accumulation_runs tn) ~default:1 |> Int.max 1;
             }))
   in
   if not (List.is_empty gb_resets && List.is_empty gb_accumulates) then
     Hashtbl.set ctx.ledger.grad_buffers ~key:id ~data:{ gb_resets; gb_accumulates });
  record_persist_roles updated_ctx id comp;
  Option.iter merge_buffer_input ~f:(fun _ ->
      let _, transfer_id = Option.value_exn ~here:[%here] frontier.merge_buffer_writer in
      Hashtbl.update ctx.ledger.merge_buffer_readers transfer_id ~f:(fun readers ->
          (id, updated_ctx.context_id) :: Option.value readers ~default:[]));

  (* Required inputs for the initialization check below: the backend routine's materialized
     read-only / read-before-write nodes, resolved against this compile's placements (the
     context-scoped memory-modes split removed the pre-lowering [context_nodes] settlement). Nodes
     with registered host initialization data (ndarray-backed literals, loaded tensors)
     self-initialize at link time from [Host_inits] (gh-ocannl-333), so they are excluded. *)
  (* Embedding only excuses nodes the routine writes: a read-only embedded node has no
     initialization in this routine (gh-ocannl-2023). It owes the same written-buffer or
     zero-valid evidence as any other input. *)
  let read_inputs = Set.diff backend_inputs (Set.inter comp.Asgns.embedded_nodes backend_outputs) in
  let inputs = Set.filter read_inputs ~f:(fun tn -> not (Ir.Host_inits.mem tn)) in
  (* gh-ocannl-1623: the reads whose buffers are device-wide host literals, which another root may
     overwrite -- [run] refuses a stale one ([shared_write_tokens]). *)
  let shared_reads =
    Set.filter read_inputs ~f:(fun tn -> Option.is_some (shared_constant_key updated_ctx tn))
  in

  (* Outputs are all nodes written by the computation *)
  let outputs = backend_outputs in
  (* Against the backend's raw read set, not [inputs]: an embedded node a routine reads first is
     state it starts from allocation's zero ([Train.sgd_one]'s momentum), left out of [inputs]
     because the routine carries that initialization, not because it is written first. *)
  let written_first =
    Set.filter (Set.diff outputs backend_inputs) ~f:(fun tn -> not (Ir.Host_inits.mem tn))
  in
  (* gh-ocannl-1966: the embedded nodes the raw read set calls read first and nothing declares to
     start from zero. Not refused: the raw read set is conservative, so this holds correct routines
     too (a vectorized write whose cell reads have unknown coverage, a read of a padded margin). An
     accumulation opening with [=+] declares its own start (gh-ocannl-2013). *)
  let read_first_undeclared =
    let from_zero = Asgns.accumulates_from_zero comp in
    Set.filter
      (Set.inter (Set.inter outputs backend_inputs) comp.Asgns.embedded_nodes)
      ~f:(fun tn -> not (Ir.Host_inits.mem tn || Tn.is_zero_valid tn || Set.mem from_zero tn))
  in
  let routine =
    {
      context = updated_ctx;
      task;
      bindings = lowered_bindings;
      name;
      inputs;
      shared_reads;
      outputs;
      written_first;
      read_first_undeclared;
      routine_id = id;
      execution_deps = deps;
      requires = comp.Asgns.requires;
      mma;
      peel;
      volatility;
      grid;
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
      (* The fragment metadata's nodes too (gh-ocannl-1724): optimization may have removed the code
         that mentions them, and a stale handle must not get past the gate as metadata. *)
      let fragment_nodes fragments =
        Set.map (module Tn) fragments ~f:(fun ((tn, _) : Asgns.Fragment.t) -> tn)
      in
      Set.iter
        (Set.union_list
           (module Tn)
           [
             read;
             written;
             comp.Ir.Assignments.embedded_nodes;
             fragment_nodes comp.Asgns.owned;
             fragment_nodes comp.Asgns.requires;
             comp.Asgns.zeroes;
             comp.Asgns.accumulates;
           ])
        ~f:check;
      (* A prelowered program replaces the comp's lowering: its nodes are the ones that run. *)
      Option.iter prelowered ~f:(fun (p : Ir.Low_level.optimized) ->
          Hashtbl.iter_keys p.traced_store ~f:check))

(* gh-ocannl-2027: a computation that re-establishes, without reading it, a node it means a write to
   last ({!Asgns.persists_undone}) undoes that write on every run -- its own forward recomputing the
   node it steps, a write sequenced into a forward, a step sequenced before the forward. Unlike a
   separate forward ([run]'s refusal), this one cannot have run once before the steps: it carries
   them. Decided over the whole computation, before lowering. *)
let check_persists_kept ?name (comp : Asgns.comp) =
  let undone = Asgns.persists_undone comp in
  if not (Set.is_empty undone) then
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Context.compile%s: this computation writes nodes meaning the write to last (an \
             optimizer step, gh-ocannl-2027) and also re-establishes them without reading them, so \
             each run undoes the step: %s. A computed node -- its forward code in this computation \
             recomputes it -- does not hold a step; materializing it changes nothing. Step what it \
             is computed from instead, or optimize a data-backed input (Operation.init \
             ~grad_spec:Require_grad) or a parameter."
            (match name with None -> "" | Some n -> " (" ^ n ^ ")")
            (String.concat ~sep:", "
               (List.map (Set.to_list undone) ~f:(fun tn ->
                    (Tn.debug_name tn [@message_only "the refusal names the node by it"])))))

(* The stale-node check alone, for the entry points over one node (host transfers, copies): run
   before anything of [tn] is forced, for the same reason (gh-ocannl-1505). *)
let check_node_valid tn = Option.iter !Tn.validity_check ~f:(fun check -> check tn)

(* gh-ocannl-1876: a routine's forward [Fetch] re-establishes a fetch-defined node on every run, so
   any other write to it in the lineage -- an optimizer step ([Train.sgd_one], which sees no loss to
   check against), a hand-written [%cd] update -- is silently undone the next time that routine
   runs, or at once when the write precedes the fetch. Refused in either compile order, and within
   one routine, before lowering. Siblings are alternatives and never collide. A host write, which no
   routine carries, is {!check_host_write_lasts}'s (gh-ocannl-1959). *)
let check_no_fetch_overwrite ?name ctx (comp : Asgns.comp) =
  let fetched = forward_fetched ctx.frontier comp and written = overwritten comp in
  let routine (id, name) = Printf.sprintf "routine %s (id=%d)" name id in
  let conflicts =
    Set.to_list (Set.union fetched written)
    |> List.filter_map ~f:(fun tn ->
        let uid = tn.Tn.uid in
        match
          ( Set.mem fetched tn,
            Set.mem written tn,
            Map.find ctx.frontier.fetch_defined uid,
            Map.find ctx.frontier.overwriters uid )
        with
        | true, true, _, _ ->
            Some
              (Tn.debug_name tn
             ^ ": this computation carries its forward code, a fetch, and also writes it")
        | false, true, Some owner, _ ->
            Some
              (Printf.sprintf
                 "%s: %s carries its forward code, a fetch, and this computation writes it"
                 (Tn.debug_name tn) (routine owner))
        | true, false, _, Some writer ->
            Some
              (Printf.sprintf
                 "%s: %s writes it, and this computation carries its forward code, a fetch"
                 (Tn.debug_name tn) (routine writer))
        | _ -> None)
  in
  if not (List.is_empty conflicts) then
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Context.compile%s: this computation and this context's lineage write nodes that a \
             forward fetch re-establishes on every run (gh-ocannl-1876): %s. A fetch-defined node \
             -- a Tensor.term_init or literal with at most limit_constant_fill_size values, a \
             range, an embedded symbol, a slice of one or a slice lowering to a copy -- is code, \
             not data: each run of the routine carrying its forward code fetches its value again, \
             silently undoing the write (an optimizer step on it never accumulates). Build a node \
             that is to be updated data-backed, with Operation.init (~grad_spec:Require_grad for \
             an optimized input), or make it a parameter, whose initializer runs once \
             (Train.init_params); write a sliced node through its parent. A sibling, compiled from \
             a context that does not descend from the other routine, is an alternative and is not \
             refused."
            (match name with None -> "" | Some n -> " (" ^ n ^ ")")
            (String.concat ~sep:"; " conflicts))

(* The literal a write to [tn] would land in: [tn] itself or, through its [\@|] slice views, the
   parent it is a view of, when that node's values are declared forever equal to its registered
   host-init data ([Tn.known_host_constant]: a [TDSL.ndarray] marked [host_constant], or a
   [TDSL.number] with its [Effectively_constant] intent). The one predicate behind the routine
   refusal {!check_no_literal_write} and the host refusal {!check_host_write_lasts}. *)
let rec literal_of tn =
  if Tn.known_host_constant tn then Some tn
  else Option.bind (Tn.slice_of tn) ~f:(fun (parent, _) -> literal_of parent)

(* gh-ocannl-2104: {!check_host_write_lasts}'s gh-ocannl-2021 rule for a routine write. A literal's
   values are declared forever equal to its registered host-init data ([Tn.known_host_constant]),
   and every consumer of that contract reads the registered values, not the buffer a routine wrote:
   a hoisted [Stage] packs them at link time ([Schedule.hoistable_constant]), and a materialized
   small literal self-initializes from them at link time
   ([Low_level.hosted_constant_inits_to_link_time]). So a write other than the literal's own fetch
   -- an [Accum_op], [Set_vec_unop] or [Rev_sides] target, directly or through a slice view of it --
   is refused, at any size, in any lineage, before lowering. A small literal whose forward the same
   computation carries was already refused as fetch-defined (gh-ocannl-1876); this check runs first,
   so a literal is named as one at every size. *)
let check_no_literal_write ?name (comp : Asgns.comp) =
  let conflicts =
    Set.to_list (overwritten comp)
    |> List.filter_map ~f:(fun tn ->
        Option.map (literal_of tn) ~f:(fun lit ->
            Printf.sprintf "%s%s" (Tn.debug_name tn)
              (if Tn.equal lit tn then "" else Printf.sprintf " (a view of %s)" (Tn.debug_name lit))))
  in
  if not (List.is_empty conflicts) then
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Context.compile%s: this computation writes a literal with constant host values \
             (gh-ocannl-2104): %s. A literal's values are declared fixed at construction: a \
             hoisted schedule reads its registered values, so a routine reading it would silently \
             ignore the write, regardless of limit_constant_fill_size. Build a node that is to be \
             updated as mutable data, with Operation.init (~grad_spec:Require_grad for an \
             optimized input) or Train.host_scalar, use TDSL.wrap or TDSL.rebatch for ndarray \
             data, or make it a parameter (TDSL.param ~param_init:(TDSL.ndarray ...))."
            (match name with None -> "" | Some n -> " (" ^ n ^ ")")
            (String.concat ~sep:"; " conflicts))

(* gh-ocannl-1724: every tensor-generated fragment is compiled into exactly one routine of a
   lineage. A backprop fragment is a [+=], so a second copy double-counts a gradient; a second
   forward copy is a reuse the caller meant as [Context.run] of the first routine. Siblings
   (compiled from a context that does not descend from the owner) are alternatives and never
   collide. *)
let check_no_duplicate_fragments ?name ctx (comp : Asgns.comp) =
  let duplicates =
    Set.to_list comp.owned
    |> List.filter_map ~f:(fun fragment ->
        Option.map
          (Map.find ctx.frontier.owners (Fragment_key.of_fragment fragment))
          ~f:(fun owner -> (fragment, owner)))
  in
  if not (List.is_empty duplicates) then
    let listed =
      List.map duplicates ~f:(fun (fragment, (id, owner)) ->
          Printf.sprintf "%s (compiled by routine %s, id=%d)"
            (Asgns.Fragment.describe fragment)
            owner id)
      |> String.concat ~sep:"; "
    in
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Context.compile%s: this computation carries code fragments already compiled in this \
             context's lineage: %s. A fragment is compiled into one routine of a lineage -- a \
             second copy of a backprop fragment would double-count a gradient. Run the earlier \
             routine again with Context.run instead of recompiling it; to compile an alternative \
             version, compile from a context that does not descend from that routine (a sibling)."
            (match name with None -> "" | Some n -> " (" ^ n ^ ")")
            listed)

(* gh-ocannl-1744: [Tensor.op] carries a shared operand's gradient zeroing in its first-constructed
   consumer only -- together with the operand's backprop -- while every consumer's own gradient code
   accumulates into it. Two routines of a lineage that accumulate into one gradient are therefore
   correct together only under a common reset: a zeroing compiled before both (so it is an execution
   dependency of the first), carried by neither, and with no other zeroing after the first
   accumulation (which would discard it). And a contribution compiled after the routine that
   propagates from the gradient never reaches the gradients below it: the propagation is ordered
   before it, and the next cycle's reset discards it. Both are refused here, before lowering;
   [Context.join] (gh-ocannl-1742) is to lift the first by hoisting the zeroing. Siblings are
   alternatives and never collide. This is compile-time compatibility only: the order of later
   [run]s is the caller's. *)
let check_shared_gradient_reset ?name ctx (comp : Asgns.comp) =
  let this =
    Option.value_or_thunk name ~default:(fun () ->
        match Asgns.get_name_exn comp.asgns with
        | derived -> derived
        | exception Invalid_argument _ -> "(unnamed)")
  in
  let routine { gr_id; gr_name; _ } = Printf.sprintf "%s (id=%d)" gr_name gr_id in
  let conflict grad =
    let roles = Map.find_multi ctx.frontier.grad_roles grad.Tn.uid in
    let accumulators = List.filter roles ~f:(fun r -> r.gr_accumulates) in
    let propagation = Map.find ctx.frontier.owners (Fragment_key.of_fragment (grad, Asgns.Bprop)) in
    let shared reason (other : grad_role) =
      Some
        (Printf.sprintf "%s, also accumulated into by routine %s: %s" (Tn.debug_name grad)
           (routine other) reason)
    in
    match (propagation, List.last accumulators) with
    | Some (owner_id, owner), _ when not (Set.mem comp.owned (grad, Asgns.Bprop)) ->
        Some
          (Printf.sprintf
             "%s: routine %s (id=%d) already propagates from it, so this contribution is compiled \
              after the propagation and never reaches the gradients below it -- compile the \
              contributing routine first and the one carrying that backprop last"
             (Tn.debug_name grad) owner owner_id)
    | _, None -> None
    | _, Some first -> (
        let zeroers = List.filter roles ~f:(fun r -> r.gr_zeroes && not r.gr_accumulates) in
        match
          ( Set.mem comp.zeroes grad,
            List.find accumulators ~f:(fun r -> r.gr_zeroes),
            List.find zeroers ~f:(fun z -> z.gr_id > first.gr_id) )
        with
        | true, _, _ ->
            shared
              "this computation zeroes it itself, discarding the other's contribution whenever it \
               runs after it"
              first
        | false, Some zeroing, _ ->
            shared
              "that routine zeroes it itself, discarding this computation's contribution whenever \
               it runs after it"
              zeroing
        | false, None, Some late ->
            shared
              (Printf.sprintf
                 "routine %s zeroes it after the first accumulation was compiled, discarding that \
                  contribution"
                 (routine late))
              first
        | false, None, None when List.is_empty zeroers ->
            shared "no zeroing of it is compiled in this lineage before both" first
        | false, None, None -> None)
  in
  let conflicts = List.filter_map (Set.to_list comp.accumulates) ~f:conflict in
  if not (List.is_empty conflicts) then
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Context.compile (%s): this computation's backprop accumulates into gradients that \
             another routine of this context's lineage also accumulates into, or already \
             propagates from, without a common zeroing ordered before both (gh-ocannl-1744): %s. A \
             tensor shared by two losses (a Require_grad input, a parameter, a trunk) receives \
             each loss's contribution from that loss's routine, but only its first-constructed \
             consumer zeroes and propagates its gradient. Compute the objectives in one routine \
             (one loss over their sum, one Train.grad_update), or compile a zeroing of the shared \
             gradient first (Train.zero_params_grads ~params) and then routines accumulating into \
             it that do not zero it (Train.grad_update ~accum_steps). A sibling, compiled from a \
             context that does not descend from the other routine, is an alternative and is not \
             refused."
            this
            (String.concat ~sep:"; " conflicts))

(* The frontier deciding which routines count as a routine's lineage at run time: the running
   context's, when it descends from the routine's compile -- same root, and the routine among its
   compile ancestry -- so an owner compiled after the routine counts; otherwise the routine's own
   compile context's, i.e. the routine and its ancestors. A sibling's never counts. *)
let lineage_frontier ctx routine =
  if
    phys_equal ctx.ledger routine.context.ledger
    && Set.mem ctx.frontier.lineage_routines routine.routine_id
  then ctx.frontier
  else routine.context.frontier

let sibling_owners ctx frontier fragment =
  Map.find_multi ctx.ledger.fragment_registry (Fragment_key.of_fragment fragment)
  |> List.filter ~f:(fun (id, _) -> not (Set.mem frontier.lineage_routines id))

let describe_owners owners =
  List.rev owners
  |> List.map ~f:(fun (id, name) -> Printf.sprintf "%s (id=%d)" name id)
  |> String.concat ~sep:", "

(* gh-ocannl-1724: a routine whose gradient writes flow into a backprop fragment it does not carry
   needs that fragment owned in its own lineage -- otherwise nothing ever propagates from that
   gradient, and the drop is silent (the dangling write is dead code). Checked at run, not at
   compile: the owner may legitimately be compiled after the routine. *)
(* A required fragment's description, with any owner the root compiled outside [frontier]. *)
let describe_unmet ctx frontier fragment =
  let described = Asgns.Fragment.describe fragment in
  match sibling_owners ctx frontier fragment with
  | [] -> described ^ " (no routine of this root context compiled it)"
  | owners ->
      Printf.sprintf
        "%s (owned by %s, compiled outside this lineage -- in a sibling, or after the context this \
         runs from -- which does not count: a sibling is an alternative, not a part of this \
         computation)"
        described (describe_owners owners)

let check_requirements ctx routine =
  let frontier = lineage_frontier ctx routine in
  let unmet =
    Set.filter routine.requires ~f:(fun f ->
        not (Map.mem frontier.owners (Fragment_key.of_fragment f)))
  in
  if not (Set.is_empty unmet) then
    let listed =
      Set.to_list unmet |> List.map ~f:(describe_unmet ctx frontier) |> String.concat ~sep:"; "
    in
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Context.run: routine %s accumulates into gradients whose backprop code no routine of \
             its lineage carries: %s. That backprop was embedded in another tensor's when it was \
             built, or handed out to an earlier Train.grad_update. Compile the routine carrying it \
             in this lineage: compile one of the two routines from the context the other's compile \
             returned, and run from the later context."
            routine.name listed)

(* The public form of a backend's missing-prior-node failure, with what the lineage knows about the
   node. When the computation requires a backprop fragment propagating from it (gh-ocannl-1724), the
   dangling gradient is materialized, so the link fails before [run]'s requirement check could.
   Otherwise, when the computation's own gradient code accumulates into it, the missing node is a
   gradient whose reset this computation does not carry: [Tensor.op] embeds an operand's gradient
   (and carries its zeroing) only while the operand is still a backprop root, and an embedded node
   is allocated by the routine rather than demanded of the context. So no accumulating operation of
   this computation found the operand a root: an earlier-constructed consumer had taken its backprop
   and zeroing (gh-ocannl-1744), and -- as no routine of the lineage zeroes it -- that consumer is
   not compiled here. The lineage cannot tell which consumer, nor whether it was evaluated
   forward-only or compiled elsewhere, so the message names the typical case as typical. A consumer
   reclaimed with its backprop unconsumed gives the reset back (gh-ocannl-1804), so a computation
   built after that lands here only if the GC had not got to it yet: the message says so. *)
let lacks_node_error ctx (comp : Asgns.comp) tn =
  let base = "The linked context lacks node " ^ Tn.debug_name tn in
  let fragment = (tn, Asgns.Bprop) in
  if
    Set.mem comp.requires fragment
    && not (Map.mem ctx.frontier.owners (Fragment_key.of_fragment fragment))
  then
    Utils.User_error
      (Printf.sprintf
         "%s: no routine in this lineage propagates from %s -- this computation accumulates into \
          it, but %s. Compile the routine carrying that backprop first, and compile this one from \
          the context its compile returned."
         base (Tn.debug_name tn)
         (describe_unmet ctx ctx.frontier fragment))
  else if
    Set.mem comp.accumulates tn
    && (not (Set.mem comp.zeroes tn))
    && (not (Set.mem comp.embedded_nodes tn))
    && not
         (List.exists (Map.find_multi ctx.frontier.grad_roles tn.Tn.uid) ~f:(fun r -> r.gr_zeroes))
  then
    (* gh-ocannl-1797: the root's routines zeroing it outside this lineage, as [sibling_owners]
       finds a fragment's owners (the lineage's own zero none of it, by the test above). *)
    let outside =
      match
        Map.find_multi ctx.ledger.zeroing_registry tn.Tn.uid
        |> List.filter ~f:(fun (id, _) -> not (Set.mem ctx.frontier.lineage_routines id))
      with
      | [] -> ""
      | zeroers ->
          Printf.sprintf
            " %s %s %s it, compiled outside this lineage -- in a sibling, or after the context \
             this routine is compiled from -- which does not count: a sibling is an alternative, \
             not a part of this computation."
            (match zeroers with [ _ ] -> "Routine" | _ -> "Routines")
            (describe_owners zeroers)
            (match zeroers with [ _ ] -> "zeroes" | _ -> "zero")
    in
    Utils.User_error
      (Printf.sprintf
         "%s: this computation's backprop accumulates into the gradient %s without zeroing it, and \
          no routine of its lineage zeroes it. Its reset is carried by an earlier-constructed \
          consumer of its tensor, beside that consumer's backprop (gh-ocannl-1744), and that \
          consumer is not compiled in this lineage -- typically a parameter's first consumer that \
          was evaluated forward-only (Train.forward_once, Train.forward), or one compiled outside \
          this lineage.%s Compile a zeroing of the gradient first -- Train.zero_params_grads \
          ~params for a parameter's -- then compile this routine from the context that compile \
          returned, and run the zeroing before each step. A consumer dropped with its backprop \
          unconsumed gives the reset back once the GC reclaims it, to tensors constructed after \
          that (gh-ocannl-1804), so until then this outcome depends on GC timing: rebuilding this \
          computation after Gc.full_major () settles it. Losses summed over one reset spell \
          Train.grad_update ~accum_steps:1, which holds whether or not the reset came back."
         base (Tn.debug_name tn) outside)
  else Utils.User_error base

let compile_outcome ?name ?lowered_transform ?prelowered ?reuse ~provenance ?candidate ?retain ctx
    comp bindings =
  (* Raised, not returned as an outcome: a dead handle is the caller's error, not a property of any
     candidate, and no search over it can succeed. *)
  check_live ~fn:"compile" ctx;
  frontend_gate ?prelowered comp;
  (* Likewise lineage-wide, and after the gate, so a discarded node is reported as such. *)
  check_no_duplicate_fragments ?name ctx comp;
  check_shared_gradient_reset ?name ctx comp;
  check_no_literal_write ?name comp;
  check_no_fetch_overwrite ?name ctx comp;
  check_persists_kept ?name comp;
  (* Compile and link on the backend context; only backend-independent routine components (and the
     rewrapped context) escape the dispatch. Unwrapped rather than through [Backends.with_backend]
     so that a [retain]ed link (gh-ocannl-1440) can rewrap the context it links later. *)
  match Backends.unwrap ctx.wrapped with
  | Backends.Unwrapped (impl, bctx) -> (
      let module Backend = (val impl.Backends.bi_module) in
      let link code () =
        let r =
          try Ir.Compile_cost.time_link (fun () -> Backend.link bctx code)
          with Backends.Prior_context_lacks tn ->
            let backtrace = Stdlib.Printexc.get_raw_backtrace () in
            Stdlib.Printexc.raise_with_backtrace (lacks_node_error ctx comp tn) backtrace
        in
        (impl.Backends.bi_wrap r.BI.context, payload_of r)
      in
      (* The [Tile_mma] rendering census is collected HERE, once, around this routine's codegen
         (gh-ocannl-626): whether a routine tensorized is a property of the compiled routine, not of
         whichever timing harness remembered to bracket the global. Fissioned segments compile
         inside this bracket, so their kernels land in the same summary. *)
      let ((((outcome, mma), peel), volatility), grid), segments =
        (* And the reduction peel's own census (gh-ocannl-733), bracketed the same way and for the
           same reason: which decision produced a kernel is a property of the compiled routine, not
           of whichever test remembered to collect it. Likewise the volatility census
           (gh-ocannl-782): which of this routine's serial accumulations the Metal compiler-bug
           workaround pinned to memory, and therefore which of them are not register-resident. And
           the shipped kernel segments themselves (gh-ocannl-1006). And the Grid census
           (gh-ocannl-1668): which outermost [Grid] loops rendered on the CPU pool, and why each
           other one stayed serial. *)
        Backends.with_segments_census @@ fun () ->
        Ir.C_syntax.with_grid_census @@ fun () ->
        Ir.C_syntax.with_volatility_census @@ fun () ->
        Ir.C_syntax.with_peel_census @@ fun () ->
        Ir.C_syntax.with_census (fun () ->
            Ir.Schedule_outcome.protect ~classify_backend:Backend.classify_failure ~provenance
              ~phase:Ir.Schedule_outcome.Transform ?candidate (fun () ->
                let code =
                  Backend.compile ?name ?lowered_transform ?prelowered ?reuse bctx.BI.optimize_ctx
                    bindings comp
                in
                Utils.count_backend_compile ();
                (code, Ir.Schedule_outcome.tag Ir.Schedule_outcome.Backend_link (link code))))
      in
      match outcome with
      | Error failure -> Error failure
      | Ok (code, (wrapped, payload)) ->
          let census = (mma, peel, volatility, grid, segments) in
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

let compile ?name ?lowered_transform ?prelowered ?reuse ctx comp bindings =
  match
    compile_outcome ?name ?lowered_transform ?prelowered ?reuse
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
let equal_persist_mark a b = a.pm_routine_id = b.pm_routine_id

(** Undoes [run]'s optimistic execution marking for a routine whose failure is known not to have
    written device buffers, so the next candidate does not inherit a dependency that never ran. *)
let rollback_execution ctx routine_id =
  ctx.ledger.executed <- Set.remove ctx.ledger.executed routine_id;
  ctx.ledger.current_writers <-
    Map.filter ctx.ledger.current_writers ~f:(function
      | Routine_writer id -> id <> routine_id
      | Released_writer -> true);
  (* Only the write claims this run introduced: a buffer an earlier writer had filled keeps its
     value, so it stays written. *)
  Option.iter (Hashtbl.find_and_remove ctx.ledger.fresh_writes routine_id) ~f:(fun keys ->
      ctx.ledger.written_buffers <- List.fold keys ~init:ctx.ledger.written_buffers ~f:Set.remove);
  Option.iter (Hashtbl.find_and_remove ctx.ledger.token_undo routine_id) ~f:(fun undos ->
      List.iter undos ~f:(undo_write_token ctx.ledger));
  (* gh-ocannl-1781: the accumulation counts and resets this run recorded, each only while its entry
     is still the one the run left -- a later reset or run is not undone with it. *)
  Option.iter (Hashtbl.find_and_remove ctx.ledger.accumulation_undo routine_id) ~f:(fun undos ->
      List.iter undos ~f:(fun { au_key; au_before; au_after } ->
          if Option.equal (Map.equal Int.equal) (Map.find ctx.ledger.accumulated au_key) au_after
          then
            ctx.ledger.accumulated <-
              Map.change ctx.ledger.accumulated au_key ~f:(fun _ -> au_before)));
  Option.iter (Hashtbl.find_and_remove ctx.ledger.persist_undo routine_id) ~f:(fun undos ->
      List.iter undos ~f:(fun { pu_key; pu_before; pu_after } ->
          if Option.equal equal_persist_mark (Map.find ctx.ledger.persisted pu_key) pu_after then
            ctx.ledger.persisted <- Map.change ctx.ledger.persisted pu_key ~f:(fun _ -> pu_before)))

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
  (* Ahead of the initialization and dependency checks: an unpropagated gradient is the more basic
     mistake, and its missing producer can surface below as an unrelated-looking input. *)
  check_requirements ctx routine;
  (* Check that all required inputs are initialized, in two classes.

     A READ-BEFORE-WRITE input (the routine both reads it first and writes it: a counter, a running
     statistic, a recurrent state) must have held a value someone put there: the very buffer the
     routine binds must have been written, in this lineage, by a run, upload or device copy
     ([written_buffers]). Not the run context's node-level [initialized_nodes]: [ctx] need not be
     [routine.context], and a sibling that seeded its own buffer for the same node says nothing
     about this one. Every path that marks a node initialized also records its buffer, so nothing is
     lost by asking the buffer. Its allocation-time zeros do not count (gh-ocannl-1339): an
     uninitialized read of state the routine then overwrites is a user bug the context can see.
     State a routine is meant to start from zero carries that in its own computation -- an embedded
     node, excluded from [inputs] at compile, declared zero-valid so that [debug_fill_fresh_outputs]
     leaves it at zero ([Train.sgd_one]'s momentum is the pattern) -- or is seeded explicitly.

     A READ-ONLY input must have been written the same way, unless allocation-time zero is declared
     a correct value of it ([Tn.is_zero_valid], gh-ocannl-1462): every gradient is marked at
     creation, being an accumulator, and [Tn.mark_zero_valid] marks any other. Before, a read-only
     input was accepted once merely allocated, and since [compile] allocates every input the check
     accepted nearly anything, so a forgotten data input read zeros. Neither the running context's
     allocation nor its node-level [initialized_nodes] vouches for the buffer either: a compile
     sibling that seeded its own buffer for the node says nothing about this one. The mark excuses
     only the read-only class of [inputs]: a read-before-write input is state the routine
     overwrites, and its entry value is the caller's to give. *)
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
  (* gh-ocannl-1623: a host literal this lineage did write or reads as its host init, but whose
     device-wide buffer another root has written since: the bytes are that root's, not this
     lineage's ([shared_write_tokens]). Checked after the class above, so a node this lineage never
     wrote is reported as uninitialized, which is the more basic mistake. *)
  let stale_inputs =
    Set.filter routine.shared_reads ~f:(fun tn ->
        Option.exists (shared_constant_key routine.context tn) ~f:(stale_in ctx.ledger))
  in
  (if not (Set.is_empty stale_inputs) then
     let stale_names =
       String.concat ~sep:", "
         (List.map (Set.to_list stale_inputs) ~f:Tn.debug_name
          [@message_only "names the stale inputs in the User_error below"])
     in
     raise
     @@ Utils.User_error
          (Printf.sprintf
             "Context.run: routine %s: input nodes %s are host-initialized literals, whose one \
              device-wide buffer another independently created context wrote after this lineage \
              last wrote or copied it, so they no longer hold this lineage's values. Re-seed them \
              in this lineage (e.g. Context.set_values), copy them from the writer with \
              Context.copy, or give each context its own buffer with Ir.Tnode.set_context_owned \
              before compiling."
             routine.name stale_names));

  (* Check execution dependencies *)
  let missing_deps = Set.diff routine.execution_deps ctx.ledger.executed in
  if not (Set.is_empty missing_deps) then
    let dep_names =
      Set.to_list missing_deps
      |> List.map ~f:(fun dep_id ->
          match Hashtbl.find ctx.ledger.routine_names dep_id with
          | Some n -> Printf.sprintf "%s (id=%d)" n dep_id
          | None ->
              (* gh-ocannl-1654: among a live routine's dependencies, only a retired merge-buffer
                 transfer loses its name: the live routine's ancestry holds its routine deps. *)
              Printf.sprintf
                "a merge-buffer fill (id=%d) that a later fill superseded (refill the merge buffer \
                 and compile the consumer against the new fill)"
                dep_id)
      |> String.concat ~sep:", "
    in
    failwith
      (Printf.sprintf "Context.run: routine %s (id=%d) has unexecuted dependencies: %s" routine.name
         routine.routine_id dep_names)

(* The PER-CANDIDATE half: bind-time validation of launch parameters
   (docs/design/signed-index-precision.md) — each one must have been assigned (gh-ocannl-940:
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

(* gh-ocannl-1781: a routine accumulating into a gradient without zeroing it must not run once more
   than its declared span (one, unless [Train.grad_update ~accum_steps:k] states [k]) since the
   gradient's buffer was last reset: each further run adds another step's gradient to the sum of the
   previous ones, the classic silent divergence. Counted per routine, so routines contributing once
   each per reset (gh-ocannl-1744's common reset) stay legal. Read-only: [run] records. *)
let check_accumulation_resets ctx routine gb =
  let runs_since_reset ua =
    Option.bind (Map.find ctx.ledger.accumulated ua.ua_key) ~f:(fun counts ->
        Map.find counts routine.routine_id)
    |> Option.value ~default:0
  in
  let exhausted =
    List.filter_map gb.gb_accumulates ~f:(fun ua ->
        let runs = runs_since_reset ua in
        Option.some_if (runs >= ua.ua_runs) (ua, runs))
  in
  if not (List.is_empty exhausted) then
    let frontier = lineage_frontier ctx routine in
    let describe (ua, runs) =
      let zeroers =
        Map.find_multi frontier.grad_roles ua.ua_uid
        |> List.filter ~f:(fun r -> r.gr_zeroes)
        |> List.rev
        |> List.map ~f:(fun r -> Printf.sprintf "%s (id=%d)" r.gr_name r.gr_id)
      in
      Printf.sprintf "%s (%d run(s) of this routine since its last reset, %d allowed; %s)"
        ua.ua_name runs ua.ua_runs
        (match zeroers with
        | [] ->
            "no routine of this lineage zeroes it -- compile a zeroing of it in this lineage \
             (Train.zero_params_grads ~params for a parameter's or a Require_grad leaf's) and run \
             it before each step, or reset it with Context.set_values"
        | zs ->
            "routine " ^ String.concat ~sep:", " zs
            ^ " of this lineage zeroes it, and has not run since -- run it before each step")
    in
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Context.run: routine %s (id=%d) accumulates into gradients without zeroing them, and \
             has already done so as many times as it may since their last reset (gh-ocannl-1781): \
             %s. Running it again would add this step's gradient to the sum of the previous ones. \
             A gradient is reset by a run of a routine zeroing it, or by a write from outside a \
             run (Context.set_values, Context.from_host, Context.copy). To accumulate deliberately \
             over k micro-steps per reset, build the update with Train.grad_update ~accum_steps:k, \
             which allows k runs between resets."
            routine.name routine.routine_id
            (String.concat ~sep:"; " (List.map exhausted ~f:describe)))

(* gh-ocannl-2027: a routine whose forward code re-establishes a buffer without reading it must not
   run while the buffer holds a value a run wrote meaning it to last ([persisted]): the run would
   silently undo it -- an optimizer step on a computed node, taken by bare [Train.sgd_one], a
   hand-written step or [Train.sgd_update], undone by the next forward. Decided here and not at
   compile: a forward run once before the steps is legal, and its routine is the same one. A routine
   both recomputing and stepping a node is the same case from its second run. Read-only: [run]
   records. *)
let check_persisted_writes ctx routine roles =
  let undone =
    List.filter_map roles.pr_recomputes ~f:(fun pb ->
        Option.map (Map.find ctx.ledger.persisted pb.pb_key) ~f:(fun mark -> (pb, mark)))
  in
  if not (List.is_empty undone) then
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Context.run: routine %s (id=%d) recomputes, from its forward code and without reading \
             them, nodes whose buffers hold a value written to last (gh-ocannl-2027): %s. Running \
             it would silently undo that write -- an optimizer step on a computed node never \
             accumulates. Step what the node is computed from instead, or optimize a data-backed \
             input (Operation.init ~grad_spec:Require_grad) or a parameter; a forward that is to \
             run once must run before the steps. To discard the steps deliberately, replace the \
             value first: Context.set_values, Context.from_host or Context.copy."
            routine.name routine.routine_id
            (String.concat ~sep:"; "
               (List.map undone ~f:(fun (pb, mark) ->
                    Printf.sprintf "%s (written to last by routine %s (id=%d))" pb.pb_name
                      mark.pm_routine mark.pm_routine_id))))

(* [run]'s record of a routine's persisting writes, with the undo [rollback_execution] applies. *)
let record_persisted ledger routine roles =
  let undo = ref [] in
  let set pu_key pu_after =
    let pu_before = Map.find ledger.persisted pu_key in
    undo := { pu_key; pu_before; pu_after } :: !undo;
    ledger.persisted <- Map.change ledger.persisted pu_key ~f:(fun _ -> pu_after)
  in
  List.iter roles.pr_persists ~f:(fun pb ->
      set pb.pb_key (Some { pm_routine_id = routine.routine_id; pm_routine = routine.name }));
  Hashtbl.set ledger.persist_undo ~key:routine.routine_id ~data:!undo

let save_accumulation_history ctx =
  let ledger = ctx.ledger in
  let saved = ledger.accumulated in
  (* gh-ocannl-2027: the persisting writes too, which the restoring uploads clear and the timing
     runs extend. *)
  let saved_persisted = ledger.persisted in
  fun () ->
    ledger.persisted <-
      Map.filter_keys saved_persisted ~f:(fun key -> Set.mem ledger.written_buffers key);
    (* A caller may have released a writer since saving. Restoring its values cannot resurrect that
       routine, nor a freed buffer's accounting. Live writers keep their exact counts. *)
    ledger.accumulated <-
      Map.filter_mapi saved ~f:(fun ~key ~data ->
          Option.some_if
            (Set.mem ledger.written_buffers key)
            (Map.filter_keys data ~f:(Hashtbl.mem ledger.routine_holders)))

(* [run]'s record of a routine's resets and accumulations, with the undo [rollback_execution]
   applies. *)
let record_accumulations ledger routine_id gb =
  let undo = ref [] in
  let set au_key au_after =
    let au_before = Map.find ledger.accumulated au_key in
    undo := { au_key; au_before; au_after } :: !undo;
    ledger.accumulated <- Map.change ledger.accumulated au_key ~f:(fun _ -> au_after)
  in
  List.iter gb.gb_resets ~f:(fun key -> if Map.mem ledger.accumulated key then set key None);
  List.iter gb.gb_accumulates ~f:(fun ua ->
      let counts =
        Option.value (Map.find ledger.accumulated ua.ua_key) ~default:(Map.empty (module Int))
      in
      set ua.ua_key
        (Some (Map.update counts routine_id ~f:(fun n -> 1 + Option.value n ~default:0))));
  Hashtbl.set ledger.accumulation_undo ~key:routine_id ~data:!undo

(* A fresh temporary host buffer matching the node's (padded) device buffer. *)
let host_buffer (tn : Tn.t) =
  Nd.create_array
    ~debug:("Context host buffer for " ^ Tn.debug_name tn)
    (Lazy.force tn.Tn.storage_prec) ~dims:(Lazy.force tn.Tn.dims)
    ~padding:(Lazy.force tn.Tn.padding)

let debug_fill_fresh_outputs =
  lazy (Utils.get_global_flag ~default:false ~arg_name:"debug_fill_fresh_outputs")

(* gh-ocannl-1873: the [written_first] nodes of [routine] whose very buffer nothing has written yet
   start from {!Ir.Ops.debug_fill_value} instead of allocation's zero, so a write the routine should
   have done and did not -- a zero a transform dropped, a seed reading the node, a cell no writer
   covers -- shows in its results rather than computing the same values from a fresh zero. So do its
   [read_first_undeclared] nodes (gh-ocannl-1966): an accumulation that lost its init reads its
   target first, and a correct routine there writes over the fill before reading.

   Asks the ledger's [written_buffers], so a repeated run, a seeded node and a node an earlier
   routine wrote all keep their values. Goes through [Backend.overwrite_from_host] rather than
   {!from_host}, because this is no write of a value: the ledger's answer stays [run]'s to give, so
   a run [rollback_execution] withdraws leaves the buffer unwritten, as gh-ocannl-1339's check asks.
   The margins of a padded node keep their padding value, which [host_buffer] gives them: writers
   refill them, and a reader of a margin reads the padding.

   A liveness-aliased buffer is filled over its whole range, which may reach bytes of a node whose
   own buffer is not fresh. That destroys nothing: bytes are shared only among alias-eligible nodes
   ([Backends.alias_eligible]), each written first in the routine that computes it, whose values the
   host and every later routine are refused, and a padded one gets its margins back from its writer.
   Only a range's first occupant reads the fill, though: a later one that lost a zero reads what its
   predecessor left there.

   Left out: a node with no scalar storage. [written_first] and [read_first_undeclared] have already
   left out every node with host initialization data, which is what makes a buffer a device-wide
   literal other roots read ([BI.Constant_key]). *)
let fill_fresh_outputs ctx routine =
  let fresh tn =
    Option.exists (buffer_key routine.context tn) ~f:(fun key ->
        not (Set.mem ctx.ledger.written_buffers key))
  in
  let targets =
    Set.filter (Set.union routine.written_first routine.read_first_undeclared) ~f:fresh
  in
  if not (Set.is_empty targets) then
    Backends.query routine.context.wrapped
      {
        q =
          (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
            (* As {!from_host}: on a host-visible buffer the upload is a direct memcpy. No queued
               kernel writes a buffer the ledger calls unwritten, but one may still be reading it,
               at the allocation zero [Tn.is_zero_valid] declares correct. *)
            Backend.await c.BI.device;
            Set.iter targets ~f:(fun tn ->
                let prec = Lazy.force tn.Tn.storage_prec in
                match Ir.Ops.debug_fill_value prec with
                | Some v ->
                    let nd = host_buffer tn in
                    let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
                    Nd.set_flat_values ?padding ~cycle:true nd [| v |];
                    ignore (Backend.overwrite_from_host c tn nd : bool)
                | _ -> ());
            Backend.await c.BI.device);
      }

let run ?(timing = false) ctx routine =
  check_runnable ctx routine;
  let grad_buffers = Hashtbl.find routine.context.ledger.grad_buffers routine.routine_id in
  (* The autotuner's timing loop dispatches one routine repeatedly by design (and restores the
     lineage's state around its windows, gh-ocannl-1621): its runs count, but are not refused. *)
  if not timing then Option.iter grad_buffers ~f:(check_accumulation_resets ctx routine);
  let persist_roles = Hashtbl.find routine.context.ledger.persist_roles routine.routine_id in
  if not timing then Option.iter persist_roles ~f:(check_persisted_writes ctx routine);
  if Lazy.force debug_fill_fresh_outputs then fill_fresh_outputs ctx routine;

  Ir.Resource_fault_injection.hit Run_before_launch;

  (* Run the routine's task/schedule *)
  Ir.Task.run routine.task;
  Option.iter grad_buffers ~f:(record_accumulations ctx.ledger routine.routine_id);
  Option.iter persist_roles ~f:(record_persisted ctx.ledger routine);

  (* Mark executed in shared ledger *)
  ctx.ledger.executed <- Set.add ctx.ledger.executed routine.routine_id;
  let fresh =
    Set.fold routine.outputs ~init:[] ~f:(fun fresh tn ->
        match buffer_key routine.context tn with
        | None -> fresh
        | Some key ->
            ctx.ledger.current_writers <-
              Map.set ctx.ledger.current_writers ~key ~data:(Routine_writer routine.routine_id);
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

let backend_facts ctx = Backends.backend_facts (Backends.wrapped_backend ctx.wrapped)

let backend_facts_of_name backend_name =
  Backends.backend_facts (Backends.get_backend ~backend_name ())

let same_backend c1 c2 =
  Backends.equal_backend (Backends.wrapped_backend c1.wrapped) (Backends.wrapped_backend c2.wrapped)

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
          (* gh-ocannl-1781: a deliberate new value -- a reset of a gradient's accumulation, zero or
             not (nothing reads it to check). *)
          ctx.ledger.accumulated <- Map.remove ctx.ledger.accumulated key;
          (* gh-ocannl-2027: and a deliberate replacement of a value written to last. *)
          ctx.ledger.persisted <- Map.remove ctx.ledger.persisted key;
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

let buffer_written ctx tn =
  Option.exists (buffer_key ctx tn) ~f:(Set.mem ctx.ledger.written_buffers)

let ledger_entries ctx =
  let l = ctx.ledger in
  Hashtbl.length l.routine_names + Set.length l.executed
  + Hashtbl.length l.merge_buffer_readers
  + Hashtbl.length l.fresh_writes + Hashtbl.length l.token_undo

let ledger_census ctx =
  let l = ctx.ledger in
  let entry name count value = (name, count, Stdlib.Obj.reachable_words (Stdlib.Obj.repr value)) in
  [
    entry "minted_holds" (Hashtbl.length l.minted_holds) l.minted_holds;
    entry "routine_units" (Hashtbl.length l.routine_units) l.routine_units;
    entry "routine_holders" (Hashtbl.length l.routine_holders) l.routine_holders;
    entry "routine_names" (Hashtbl.length l.routine_names) l.routine_names;
    entry "executed" (Set.length l.executed) l.executed;
    entry "current_writers" (Map.length l.current_writers) l.current_writers;
    entry "merge_buffer_readers" (Hashtbl.length l.merge_buffer_readers) l.merge_buffer_readers;
    entry "written_buffers" (Set.length l.written_buffers) l.written_buffers;
    entry "fresh_writes" (Hashtbl.length l.fresh_writes) l.fresh_writes;
    entry "observed_tokens" (Map.length l.observed_tokens) l.observed_tokens;
    entry "token_undo" (Hashtbl.length l.token_undo) l.token_undo;
    entry "fragment_registry" (Map.length l.fragment_registry) l.fragment_registry;
    entry "zeroing_registry" (Map.length l.zeroing_registry) l.zeroing_registry;
    entry "grad_buffers" (Hashtbl.length l.grad_buffers) l.grad_buffers;
    entry "accumulated" (Map.length l.accumulated) l.accumulated;
    entry "accumulation_undo" (Hashtbl.length l.accumulation_undo) l.accumulation_undo;
    entry "persist_roles" (Hashtbl.length l.persist_roles) l.persist_roles;
    entry "persisted" (Map.length l.persisted) l.persisted;
    entry "persist_undo" (Hashtbl.length l.persist_undo) l.persist_undo;
  ]

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

(* {2 Observability (gh-ocannl-777)}

   Whether a host read of a node can be served from this context, answered from the lineage's
   metadata alone -- placements, the context's buffers, [Host_inits] and the stored computations --
   without compiling. A [Virtual] node is observable by recomputation exactly when every node one
   replay of its stored computations reads is: materialized and present (which shields whatever
   produced it, a [Local] dependency included), host-initialized, or itself virtual and observable.
   The walk reads what the inliner would replay ({!Ir.Low_level.replay_facts}), not the cost
   collector's approximation. *)
type observability =
  | Stored
  | Host_init
  | Recomputable
  | Needs_context of Tn.t list
  | Unobservable_local of Tn.t list
  | Unobservable_aliased of Tn.t list
  | Unobservable_merge of Tn.t list
  | Launch_bound of Tn.t list * Idx.symbol list
  | Slice_view of Tn.t
  | Not_computed

let aliased ctx (tn : Tn.t) =
  Backends.query ctx.wrapped { q = (fun _ c -> Backends.buffer_aliased c.BI.ctx_buffers tn) }

let computations ctx =
  Backends.query ctx.wrapped { q = (fun _ c -> c.BI.optimize_ctx.Ir.Low_level.computations) }

(* The verdict on one virtual node's stored computations, in precedence order: an unobservable
   dependency is reported over a launch dependence, which is reported over absent leaves -- the
   first cannot be fixed by reading from another context, the last can. Paths start at the node the
   verdict is about. *)
type walk_verdict =
  | W_local of Tn.t list
  | W_aliased of Tn.t list
  | W_merge of Tn.t list
  | W_launch of Tn.t list * Idx.symbol list
  | W_missing of Set.M(Tn).t
  | W_ok

let verdict_rank = function
  | W_local _ -> 0
  | W_aliased _ -> 1
  | W_merge _ -> 2
  | W_launch _ -> 3
  | W_missing _ -> 4
  | W_ok -> 5

(* Combines two verdicts: the higher-precedence one wins, the first (program order) on a tie, and
   absent leaves accumulate. *)
let combine_verdicts v1 v2 =
  match (v1, v2) with
  | W_missing a, W_missing b -> W_missing (Set.union a b)
  | _ -> if verdict_rank v2 < verdict_rank v1 then v2 else v1

let observability_internal_error tn what =
  failwith
    (Printf.sprintf
       "Context.observability: internal error: %s %s (a compile that stored the computation \
        reading it should have decided it)"
       (Tn.debug_name tn) what)

let walk_virtual ctx (root : Tn.t) : walk_verdict =
  let plc = placements ctx in
  let comps = computations ctx in
  let memo = Hashtbl.create (module Tn) in
  let present tn = mem ctx tn || Ir.Host_inits.mem tn in
  let prefix v = function
    | W_local path -> W_local (v :: path)
    | W_aliased path -> W_aliased (v :: path)
    | W_merge path -> W_merge (v :: path)
    | W_launch (path, syms) -> W_launch (v :: path, syms)
    | (W_missing _ | W_ok) as w -> w
  in
  let rec walk (v : Tn.t) : walk_verdict =
    match Hashtbl.find memo v with
    | Some w -> w
    | None ->
        (* A cycle through virtual nodes cannot be stored (a node's replay excludes itself, and a
           read of a virtual node inlines its already-stored template); the in-progress mark only
           keeps the walk finite. *)
        Hashtbl.set memo ~key:v ~data:W_ok;
        let components =
          match Hashtbl.find comps v with
          | Some (_ :: _ as cs) -> cs
          | Some [] | None -> observability_internal_error v "is virtual with no stored computation"
        in
        (* Stored newest first; walked oldest first, so "first" means program order. *)
        let w =
          List.fold (List.rev components) ~init:W_ok ~f:(fun acc (_, body) ->
              let facts = Ir.Low_level.replay_facts ~self:v body in
              let acc =
                if facts.replay_merge_reads then combine_verdicts acc (W_merge [ v ]) else acc
              in
              let acc =
                if Set.is_empty facts.replay_free_symbols then acc
                else combine_verdicts acc (W_launch ([ v ], Set.to_list facts.replay_free_symbols))
              in
              Set.fold facts.replay_reads ~init:acc ~f:(fun acc r ->
                  combine_verdicts acc (read_verdict v r)))
        in
        Hashtbl.set memo ~key:v ~data:w;
        w
  and read_verdict v r =
    match Tn.Placements.get plc r with
    | Some (Tn.Local, _) -> W_local [ v; r ]
    | _ when aliased ctx r -> W_aliased [ v; r ]
    | _ when present r -> W_ok
    | Some (Tn.Virtual, _) -> prefix v (walk r)
    | Some (Tn.Effectively_constant, _) when Hashtbl.mem comps r -> prefix v (walk r)
    | Some (Tn.On_device, _) -> W_missing (Set.singleton (module Tn) r)
    | Some ((Tn.Effectively_constant | Tn.Never_virtual), _) | None ->
        observability_internal_error r
          "is read by a stored computation but has neither an available buffer nor a replayable \
           computation"
  in
  walk root

let observability ctx (tn : Tn.t) : observability =
  check_live ~fn:"observability" ctx;
  check_node_valid tn;
  match Tn.slice_of tn with
  | Some (parent, _) -> Slice_view parent
  | None -> (
      let plc = placements ctx in
      match Tn.Placements.get plc tn with
      | Some (Tn.Local, _) -> Unobservable_local [ tn ]
      | placement -> (
          if aliased ctx tn then Unobservable_aliased [ tn ]
          else if mem ctx tn then Stored
          else if Ir.Host_inits.mem tn then Host_init
          else
            match placement with
            | Some ((Tn.Virtual | Tn.Effectively_constant), _)
              when Hashtbl.mem (computations ctx) tn -> (
                match walk_virtual ctx tn with
                | W_ok -> Recomputable
                | W_local path -> Unobservable_local path
                | W_aliased path -> Unobservable_aliased path
                | W_merge path -> Unobservable_merge path
                | W_launch (path, syms) -> Launch_bound (path, syms)
                | W_missing leaves -> Needs_context (Set.to_list leaves))
            | Some (Tn.On_device, _) -> Needs_context [ tn ]
            | Some ((Tn.Virtual | Tn.Effectively_constant | Tn.Never_virtual | Tn.Local), _) | None
              ->
                Not_computed))

let path_to_string path =
  String.concat ~sep:" -> "
    (List.map path ~f:Tn.debug_name [@message_only "spells a dependency path for the messages"])

let observability_tag = function
  | Stored | Host_init | Recomputable -> None
  | Needs_context leaves ->
      Some
        ("<not-in-context: needs "
        ^ String.concat ~sep:", " (List.map leaves ~f:Tn.debug_name)
        ^ ">")
  | Unobservable_local [ _ ] -> Some "<unobservable: Local>"
  | Unobservable_local path ->
      Some ("<unobservable: reads Local " ^ Tn.debug_name (List.last_exn path) ^ ">")
  | Unobservable_aliased path ->
      Some ("<unobservable: buffer of " ^ Tn.debug_name (List.last_exn path) ^ " aliased>")
  | Unobservable_merge path ->
      Some ("<unobservable: " ^ Tn.debug_name (List.last_exn path) ^ " reads a merge buffer>")
  | Launch_bound (_, syms) ->
      Some
        ("<unobservable: depends on launch index "
        ^ String.concat ~sep:", " (List.map syms ~f:Idx.symbol_ident)
        ^ ">")
  | Slice_view parent -> Some ("<slice-view: read " ^ Tn.debug_name parent ^ ">")
  | Not_computed -> Some "<not-computed>"

let explain_observability ctx (tn : Tn.t) (obs : observability) : string option =
  let name = (Tn.debug_name tn [@message_only "names the node in the explanation"]) in
  match obs with
  | Stored | Host_init | Recomputable -> None
  | Needs_context [ leaf ] when Tn.equal leaf tn ->
      Some
        (Printf.sprintf
           "node %s is materialized (placed On_device) but has no buffer in this context: read it \
            from a context its computing routine ran in"
           name)
  | Needs_context leaves ->
      Some
        (Printf.sprintf
           "node %s is virtual and would be recomputed, but its computation reads %s, materialized \
            in this lineage yet absent from this context: read it from a context holding %s"
           name
           (String.concat ~sep:", " (List.map leaves ~f:Tn.debug_name))
           (if List.length leaves = 1 then "that node" else "those nodes"))
  | Unobservable_local [ _ ] ->
      (* The root itself is Local: [to_host] reports it through [refuse_local], with its provenance
         and the cap advice. *)
      Some (Printf.sprintf "node %s is placed Local (routine-scoped scratch)" name)
  | Unobservable_local path ->
      let local = List.last_exn path in
      let prov =
        match Tn.Placements.get (placements ctx) local with
        | Some (_, prov) -> Tn.provenance_to_string prov
        | None -> "undecided"
      in
      Some
        (Printf.sprintf
           "node %s is virtual, and recomputing it reads %s, which this lineage placed Local \
            (provenance %s): routine-scoped scratch, kept by no routine, so %s cannot be observed \
            (dependency path %s). Materialize %s -- e.g. Train.set_materialized or \
            Context.decide_materialized -- before the first routine using it is compiled, or \
            materialize %s itself."
           name (Tn.debug_name local) prov name (path_to_string path) (Tn.debug_name local) name)
  | Unobservable_aliased path ->
      let victim = List.last_exn path in
      Some
        (Printf.sprintf
           "%s: tensor node %s was buffer-aliased by the liveness memory planner (config \
            buffer_aliasing): its buffer is shared with other nodes and its values are not \
            preserved past its last read within the routine that computes it (dependency path %s). \
            Mark %s observable (Tnode.set_observable) before compiling that routine, or disable \
            buffer_aliasing."
           (if Tn.equal victim tn then "node " ^ name ^ " cannot be read"
            else "node " ^ name ^ " is virtual, and recomputing it reads " ^ Tn.debug_name victim)
           (Tn.debug_name victim) (path_to_string path) (Tn.debug_name victim))
  | Unobservable_merge path ->
      Some
        (Printf.sprintf
           "node %s is virtual, and recomputing it replays a read of a merge buffer (in the stored \
            computation of %s, dependency path %s): merge-buffer contents are transient to the \
            routine receiving the transfer, so %s cannot be observed by recomputation. Materialize \
            %s to observe it."
           name
           (Tn.debug_name (List.last_exn path))
           (path_to_string path) name name)
  | Launch_bound (path, syms) ->
      Some
        (Printf.sprintf
           "node %s is virtual, and its stored computation (dependency path %s) depends on the \
            launch index %s, whose value is bound only for a launch of the routine that stored it: \
            a host read has no binding to supply, so %s cannot be observed by recomputation. \
            Materialize %s to observe it."
           name (path_to_string path)
           (String.concat ~sep:", " (List.map syms ~f:Idx.symbol_ident))
           name name)
  | Slice_view parent ->
      Some
        (Printf.sprintf "node %s is an @| slice view; read its parent %s instead" name
           (Tn.debug_name parent))
  | Not_computed ->
      Some
        (Printf.sprintf
           "node %s has no buffer in this context, no host initialization data, and no computation \
            stored by any routine of this context's compilation lineage: compile (and run) a \
            routine computing it into this context first"
           name)

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

(* gh-ocannl-1738: the written records of the locations a release gave up. A freed pool's locations
   are never handed out again (pool ids are minted per device and never reused), but a trimmed
   upload-arena tail is, at the same locations, to the next upload of a surviving sibling: an entry
   left behind would vouch for a buffer nobody wrote since. Retired from [fresh_writes] too, so a
   later rollback of a routine that wrote there cannot withdraw a later writer's entry. *)
let retire_written ledger ~device_id retired =
  let gone (d, (loc : BI.buffer_loc)) =
    d = device_id
    && List.exists retired ~f:(function
      | Backends.Pool pool_id -> loc.pool_id = pool_id
      | Backends.Location l -> BI.equal_buffer_loc l loc)
  in
  if not (List.is_empty retired) then (
    ledger.written_buffers <- Set.filter ledger.written_buffers ~f:(fun key -> not (gone key));
    (* gh-ocannl-1762: the writer table and the host-literal write tokens go with the location too
       -- a freed constant pool is a literal's last binder released, and its tokens would otherwise
       stay for the process. *)
    ledger.current_writers <- Map.filter_keys ledger.current_writers ~f:(fun key -> not (gone key));
    ledger.observed_tokens <- Map.filter_keys ledger.observed_tokens ~f:(fun key -> not (gone key));
    shared_write_tokens := Map.filter_keys !shared_write_tokens ~f:(fun key -> not (gone key));
    ledger.accumulated <- Map.filter_keys ledger.accumulated ~f:(fun key -> not (gone key));
    ledger.persisted <- Map.filter_keys ledger.persisted ~f:(fun key -> not (gone key));
    Hashtbl.map_inplace ledger.fresh_writes ~f:(List.filter ~f:(fun key -> not (gone key))))

(* gh-ocannl-1762: once [ctx]'s unit is released, its holds on the ancestors a wrapper minted for it
   are given back (once per unit), and every ancestor no unit holds any more is released, deepest
   first. Releasing an ancestor is idempotent, so the entry stays until every such release
   succeeded: a release that raised part way is resumed by the next [release] of any value of the
   unit. The first failure is re-raised once all were tried. *)
let release_minted ctx =
  let table = ctx.ledger.minted_holds in
  match Hashtbl.find table ctx.context_id with
  | Some entry when Atomic.get (release_flag ctx.wrapped) -> (
      if not entry.given_back then (
        entry.given_back <- true;
        List.iter entry.holds ~f:(fun h -> Atomic.decr h.holders));
      let unheld =
        List.filter entry.holds ~f:(fun h -> Atomic.get h.holders <= 0)
        |> List.sort ~compare:(fun a b -> Int.compare b.depth a.depth)
      in
      let first_failure =
        List.fold unheld ~init:None ~f:(fun failure h ->
            match h.release_ancestor () with
            | () -> failure
            | exception exn -> (
                match failure with
                | Some _ -> failure
                | None -> Some (exn, Stdlib.Printexc.get_raw_backtrace ())))
      in
      match first_failure with
      | None -> Hashtbl.remove table ctx.context_id
      | Some (exn, backtrace) -> Stdlib.Printexc.raise_with_backtrace exn backtrace)
  | Some _ | None -> ()

(* Retirement is not rollback: a released routine may have written a surviving ancestor's buffer.
   Its values and write tokens remain valid; only records no live routine can use go. *)
let retire_routine_unit ctx =
  let l = ctx.ledger in
  Option.iter (Hashtbl.find_and_remove l.routine_units ctx.context_id) ~f:(fun ids ->
      Set.iter ids ~f:(fun id ->
          match Hashtbl.find l.routine_holders id with
          | Some n when n > 1 -> Hashtbl.set l.routine_holders ~key:id ~data:(n - 1)
          | Some _ ->
              Hashtbl.remove l.routine_holders id;
              Hashtbl.remove l.routine_names id;
              Hashtbl.remove l.fresh_writes id;
              Hashtbl.remove l.token_undo id;
              Hashtbl.remove l.grad_buffers id;
              Hashtbl.remove l.accumulation_undo id;
              Hashtbl.remove l.persist_roles id;
              Hashtbl.remove l.persist_undo id;
              (* Other live routines may roll back a reset of these buffers. Scrub both snapshots as
                 well as the current counts, preserving the equality check that prevents a rollback
                 from undoing a later write. *)
              let without_count counts = Map.remove counts id in
              l.accumulated <- Map.map l.accumulated ~f:without_count;
              Hashtbl.map_inplace l.accumulation_undo
                ~f:
                  (List.map ~f:(fun undo ->
                       {
                         undo with
                         au_before = Option.map undo.au_before ~f:without_count;
                         au_after = Option.map undo.au_after ~f:without_count;
                       }));
              l.executed <- Set.remove l.executed id;
              l.current_writers <-
                Map.map l.current_writers ~f:(function
                  | Routine_writer writer when writer = id -> Released_writer
                  | writer -> writer);
              let without_id records =
                let remaining = List.filter records ~f:(fun (owner, _) -> owner <> id) in
                Option.some_if (not (List.is_empty remaining)) remaining
              in
              l.fragment_registry <- Map.filter_map l.fragment_registry ~f:without_id;
              l.zeroing_registry <- Map.filter_map l.zeroing_registry ~f:without_id
          | None -> ()))

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
  let retired = ref [] in
  let device_id = Backends.query ctx.wrapped { q = (fun _ c -> c.BI.device.device_id) } in
  match
    Backends.query ctx.wrapped
      {
        q =
          (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
            Backends.finalize (module Backend) c ~on_retire:(fun r -> retired := r :: !retired));
      }
  with
  | () ->
      retire_written ctx.ledger ~device_id !retired;
      retire_readers ();
      retire_routine_unit ctx;
      release_minted ctx
  | exception exn ->
      (* gh-ocannl-1445: a release hook (graph execs) that fails runs after the pools were freed and
         leaves the context finalized, so it is a released consumer all the same. A cleanup that
         raised part way reported only what it did give up, so that is retired either way. *)
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      retire_written ctx.ledger ~device_id !retired;
      if Backends.query ctx.wrapped { q = (fun _ c -> Atomic.get c.BI.finalized) } then (
        retire_readers ();
        retire_routine_unit ctx;
        try release_minted ctx with _ -> ());
      Stdlib.Printexc.raise_with_backtrace exn backtrace

let same_release_unit a b = phys_equal (release_flag a.wrapped) (release_flag b.wrapped)

(* gh-ocannl-1762: see the interface. The ancestor is found on [ctx]'s lifecycle chain by its
   release flag, the identity of its unit, as [released_ancestor_depth] finds a released one. *)
let is_lifecycle_ancestor ctx ~ancestor =
  let flag = release_flag ancestor.wrapped in
  Backends.query ctx.wrapped
    {
      q =
        (fun _ c ->
          let rec walk p =
            match p with
            | None -> false
            | Some p -> phys_equal p.BI.finalized flag || walk p.BI.parent
          in
          walk c.BI.parent);
    }

let adopt_ancestor ctx ~ancestor =
  check_live ~fn:"adopt_ancestor" ctx;
  if not (is_lifecycle_ancestor ctx ~ancestor) then
    invalid_arg
      (Printf.sprintf
         "Context.adopt_ancestor: context %d is not a lifecycle ancestor of context %d (a context \
          it was compiled or decided from, directly or not)"
         ancestor.context_id ctx.context_id);
  if not (phys_equal ctx.ledger ancestor.ledger) then
    invalid_arg "Context.adopt_ancestor: the ancestor's context value belongs to another root";
  let hold =
    {
      release_ancestor = (fun () -> release ancestor);
      depth = lifecycle_depth ancestor;
      holders = Atomic.make 1;
    }
  in
  Hashtbl.update ctx.ledger.minted_holds ctx.context_id ~f:(function
    | None -> { holds = [ hold ]; given_back = false }
    | Some entry -> { entry with holds = entry.holds @ [ hold ] })

(* {2 Recompute-on-read (gh-ocannl-777)}

   A [Recomputable] node is read through a throwaway routine [proxy =: tn], compiled from the
   reading context so its lowering inlines [tn]'s stored computation (recompute-at-read, gh-617),
   built as plain assignments: no tensor, so no shape inference and no forward root. [proxy] is a
   fresh materialized node with [tn]'s precision and layout (unpadded dims and padding), so the
   transfer fills a host buffer [get_values] reads with [tn]'s own padding. The child context is
   released whatever happens, and its private ledger records retired, so repeated reads leave the
   parent's buffers, frontier, placements and ledger as they were. *)
let recompute_namespace = "observe"
let recompute_provenance = Tn.Site "777:recompute-on-read"

let recompute_id =
  let c = ref (-1) in
  fun () ->
    Int.incr c;
    !c

let identity_copy ~(lhs : Tn.t) ~(rhs : Tn.t) : Asgns.t =
  let dims = Tn.dims_without_padding rhs in
  let axes =
    Array.map dims ~f:(fun d ->
        if d > 1 then
          let s = Idx.get_symbol () in
          (Idx.Iterator s, Some [ (d, s) ])
        else (Idx.Fixed_idx 0, None))
  in
  let idcs = Array.map axes ~f:fst in
  let projections : Idx.projections =
    {
      components = Array.filter_map axes ~f:snd;
      lhs_dims = dims;
      rhs_dims = [| dims |];
      project_lhs = idcs;
      project_rhs = [| idcs |];
      extent_syms = [];
      debug_info =
        {
          spec = "recompute-on-read";
          derived_for =
            Sexp.Atom (Tn.debug_name rhs [@message_only "the projection's debug provenance"]);
          trace = [];
        };
    }
  in
  Asgns.Accum_op
    {
      initialize_neutral = false;
      accum = Ir.Ops.Arg2;
      lhs;
      rhs = Asgns.Unop { op = Ir.Ops.Identity; rhs = Asgns.Node rhs };
      projections = Lazy.from_val projections;
      projections_debug = "recompute-on-read";
    }

(* Undoes the throwaway routine's ledger records after its context is released: they are private to
   it -- nothing is compiled from the child, and its output is the fresh [proxy]. (The release
   itself already retired the freed proxy buffer's [written_buffers] entry, gh-ocannl-1738.) *)
let retire_recompute_records ctx (routine : routine) =
  rollback_execution ctx routine.routine_id;
  Hashtbl.remove ctx.ledger.routine_names routine.routine_id

(* A throwaway context whose release raised before its pools were freed (the backend keeps a retry
   ledger: [finalize] resumes where it stopped). The caller never held the handle, so ownership
   stays here: each later recompute retries these first, and its records are retired once the
   release commits. A release whose pools were freed but whose release hook then raised is
   committed: its records are retired at once, as [release] retires merge readers in that case. *)
let pending_recompute_releases : (t * t * routine) list ref = ref []
let pending_recompute_mutex = Stdlib.Mutex.create ()

let release_recompute ~parent child (routine : routine) =
  match release child with
  | () -> retire_recompute_records parent routine
  | exception exn ->
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      if Atomic.get (release_flag child.wrapped) then retire_recompute_records parent routine
      else
        Stdlib.Mutex.protect pending_recompute_mutex (fun () ->
            pending_recompute_releases := (child, parent, routine) :: !pending_recompute_releases);
      Stdlib.Printexc.raise_with_backtrace exn backtrace

let retry_pending_recompute_releases () =
  let pending =
    Stdlib.Mutex.protect pending_recompute_mutex (fun () ->
        let p = !pending_recompute_releases in
        pending_recompute_releases := [];
        p)
  in
  (* Each failure was reported when it happened, and a failed retry is queued again, so a retry's
     own exception is not this read's to raise. *)
  List.iter pending ~f:(fun (child, parent, routine) ->
      try release_recompute ~parent child routine with _ -> ())

(* gh-ocannl-1737: every node of [reads] is recomputed by ONE throwaway routine -- a [proxy_i =:
   tn_i] copy per node, sequenced -- so reading a whole tree compiles once rather than once per
   node. Each copy inlines its node's stored computation against the leaves as they are now, exactly
   as a single-node read does; the copies only share the routine. *)
let recompute_all_to_host ctx (reads : (Tn.t * Nd.t) list) : Nd.t list =
  retry_pending_recompute_releases ();
  let reads =
    List.map reads ~f:(fun (tn, nd) ->
        let proxy =
          Tn.create ~namespace:recompute_namespace
            (Tn.Specified (Lazy.force tn.Tn.storage_prec))
            ~id:(recompute_id ()) ~label:("recompute" :: tn.Tn.label)
            ~unpadded_dims:(lazy (Tn.dims_without_padding tn))
            ~padding:tn.Tn.padding ()
        in
        Tn.update_memory_mode proxy Tn.On_device recompute_provenance;
        (tn, proxy, nd))
  in
  let comp =
    Asgns.sequence
      (List.map reads ~f:(fun (tn, proxy, _) ->
           Asgns.comp ~embedded:(Set.singleton (module Tn) proxy) (identity_copy ~lhs:proxy ~rhs:tn)))
  in
  let describe =
    match reads with
    | [ (tn, _, _) ] -> Tn.debug_name tn [@message_only "describes the recomputed read"]
    | _ ->
        String.concat ~sep:", "
          (List.map reads ~f:(fun (tn, _, _) ->
               (Tn.debug_name tn [@message_only "describes the recomputed reads"])))
  in
  let name =
    match reads with
    | [ (tn, _, _) ] -> Printf.sprintf "recompute_%d" tn.Tn.uid
    | (tn, _, _) :: _ -> Printf.sprintf "recompute_%d_and_%d_more" tn.Tn.uid (List.length reads - 1)
    | [] -> invalid_arg "Context.recompute_all_to_host: nothing to read"
  in
  let child, routine = compile ~name ctx comp Idx.Empty in
  let child = ref child in
  let read () =
    child := run !child routine;
    List.map reads ~f:(fun (tn, proxy, nd) ->
        let copied =
          Backends.query !child.wrapped
            {
              q =
                (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
                  Backend.await c.BI.device;
                  if Backend.to_host c proxy nd then (
                    Backend.await c.BI.device;
                    true)
                  else false);
            }
        in
        if not copied then
          failwith
            (Printf.sprintf
               "Context.to_host: internal error: the recompute of %s left no buffer for it"
               (Tn.debug_name tn));
        (* The fringe of a padded node holds its committed padding value, whatever the scratch
           buffer's fringe held: the interior is moved into a freshly allocated buffer, which
           [host_buffer] fills with that value. *)
        match Lazy.force tn.Tn.padding with
        | None -> nd
        | Some (padding, _) ->
            let fresh = host_buffer tn in
            Nd.set_flat_values ~padding fresh (Nd.retrieve_flat_values ~padding nd);
            fresh)
  in
  match read () with
  | nds ->
      release_recompute ~parent:ctx !child routine;
      nds
  | exception exn ->
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      (try release_recompute ~parent:ctx !child routine
       with release_exn ->
         Stdio.eprintf
           "Context.to_host: releasing the recompute context of %s also failed (retried at the \
            next recompute): %s\n\
            %!"
           describe (Exn.to_string release_exn));
      Stdlib.Printexc.raise_with_backtrace exn backtrace

let recompute_to_host ctx (tn : Tn.t) (nd : Nd.t) : Nd.t =
  match recompute_all_to_host ctx [ (tn, nd) ] with [ nd ] -> nd | _ -> assert false

(** Transfers [tn]'s value into a fresh host [Ndarray] and returns it: its device buffer, its
    host-init data, or its recomputation ({!observability}); refuses with the reason otherwise. *)
let to_host ctx (tn : Tn.t) : Nd.t =
  check_live ~fn:"to_host" ctx;
  check_node_valid tn;
  check_not_poisoned ctx;
  let refuse reason =
    raise
    @@ Utils.User_error
         (Printf.sprintf "Context.to_host: %s (backend %s)" reason (backend_name ctx))
  in
  let obs = observability ctx tn in
  match obs with
  | Stored ->
      (* [transfer] awaits pending device writes feeding the node, attempts the device-to-host copy,
         and awaits its completion before the host buffer is read. *)
      let nd = host_buffer tn in
      let copied =
        Backends.query ctx.wrapped
          {
            q =
              (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) c ->
                Backend.await c.BI.device;
                if Backend.to_host c tn nd then (
                  Backend.await c.BI.device;
                  true)
                else false);
          }
      in
      if copied then nd
      else
        failwith
          (Printf.sprintf "Context.to_host: internal error: no transfer for %s, which has a buffer"
             (Tn.debug_name tn))
  | Host_init ->
      (* An ndarray-backed literal that is not part of any computation in this context (so it was
         never allocated on the device): its value is its registered host initialization data.
         Return a private copy so a mutating caller (e.g. [set_value]'s read-modify-write) cannot
         corrupt the shared initialization buffer used to initialize other contexts. *)
      copy_nd (Lazy.force (Option.value_exn ~here:[%here] (Ir.Host_inits.find tn)))
  | Recomputable -> recompute_to_host ctx tn (host_buffer tn)
  | Unobservable_local [ _ ] ->
      (* A [Local] node's own buffer, if the context has one at all, holds only what a host write
         put there, and its host-init data predates every run -- neither is what the routines
         computed. *)
      refuse_local ~fn:"to_host" ctx tn (Option.value_exn ~here:[%here] (local_placement ctx tn))
  | Needs_context _ | Not_computed ->
      refuse
        (Printf.sprintf "node %s is not present in context: %s" (Tn.debug_name tn)
           (Option.value_exn ~here:[%here] (explain_observability ctx tn obs)))
  | Unobservable_local _ | Unobservable_aliased _ | Unobservable_merge _ | Launch_bound _
  | Slice_view _ ->
      refuse (Option.value_exn ~here:[%here] (explain_observability ctx tn obs))

(** {!to_host} over several nodes, all [Recomputable] ones through one shared throwaway routine
    (gh-ocannl-1737). Every node is classified before anything is read or compiled: the first one
    {!to_host} would refuse, in list order, raises its refusal. *)
let to_host_all ctx (tns : Tn.t list) : Nd.t list =
  check_live ~fn:"to_host_all" ctx;
  List.iter tns ~f:check_node_valid;
  check_not_poisoned ctx;
  let distinct = List.dedup_and_sort tns ~compare:Tn.compare in
  let classified = List.map distinct ~f:(fun tn -> (tn, observability ctx tn)) in
  (* A refused node raises exactly as its own read would, before any other node is read. *)
  List.iter tns ~f:(fun tn ->
      match List.Assoc.find_exn classified ~equal:Tn.equal tn with
      | Stored | Host_init | Recomputable -> ()
      | _ -> ignore (to_host ctx tn : Nd.t));
  let recomputed =
    match List.filter_map classified ~f:(function tn, Recomputable -> Some tn | _, _ -> None) with
    | [] -> []
    | recomputable ->
        List.zip_exn recomputable
          (recompute_all_to_host ctx (List.map recomputable ~f:(fun tn -> (tn, host_buffer tn))))
  in
  let read = Hashtbl.create (module Tn) in
  List.iter recomputed ~f:(fun (tn, nd) -> Hashtbl.set read ~key:tn ~data:nd);
  List.map tns ~f:(fun tn -> Hashtbl.find_or_add read tn ~default:(fun () -> to_host ctx tn))

(* gh-ocannl-1959: {!check_no_fetch_overwrite}'s rule for a host write. Only once the forward is
   compiled: a seed before that is what a routine reading the node runs on until the forward first
   runs ([shared_grad_refusal.ml], [require_grad_leaves.ml]). A materialized literal the routine
   reads has lost its in-kernel fetch (gh-ocannl-633), so a write would stay in its buffer, but the
   literal's values are declared constant: a schedule hoisting it reads the registered ones
   ([Schedule.hoistable_constant]), so the write is refused all the same. {!copy} has its own rule,
   {!check_copy_lasts}: it moves a node's value between contexts, which may be the fetch's own. Runs
   before anything of the write happens, so a refused [set_values] leaves the node's host buffer as
   it was. gh-ocannl-2021: a literal ({!literal_of}, the predicate of the routine refusal too) is
   immutable even without a fetch or a compiled forward: above the fill limit it is Reshape-backed,
   but hoisting still reads its registered values, and a [TDSL.number] a routine inlines. A view of
   a literal is refused as one, naming the literal: writing the parent instead, which {!upload}'s
   refusal of any other view says, is refused too. *)
let check_host_write_lasts ~fn ctx (tn : Tn.t) =
  match (Tn.slice_of tn, Map.find ctx.frontier.fetch_defined tn.Tn.uid) with
  | None, Some (id, name) ->
      raise
      @@ Utils.User_error
           (Printf.sprintf
              "Context.%s: node %s is fetch-defined in this lineage: routine %s (id=%d) carries \
               its forward code, a fetch, so this write would be silently lost: a run fetches the \
               node again, or reads a literal's own values, inlined or hoisted (gh-ocannl-1959). A \
               fetch-defined node -- a Tensor.term_init or literal with at most \
               limit_constant_fill_size values, a range, an embedded symbol -- is code, not data. \
               Build a node the host writes data-backed, with Operation.init or Train.host_scalar, \
               or make it a parameter. A seed for a routine that reads the node before the forward \
               runs goes in before the forward's routine is compiled."
              fn (Tn.debug_name tn) name id)
  | _ -> (
      match literal_of tn with
      | None -> ()
      | Some lit ->
          raise
          @@ Utils.User_error
               (Printf.sprintf
                  "Context.%s: node %s is %s with constant host values (gh-ocannl-2021). A hoisted \
                   schedule reads its registered values, and a routine inlines or fetches them, so \
                   a host write would be silently ignored, regardless of limit_constant_fill_size \
                   or whether a routine is compiled. Build mutable data with Operation.init or \
                   Train.host_scalar, use TDSL.wrap or TDSL.rebatch for ndarray data, or make it a \
                   parameter."
                  fn (Tn.debug_name tn)
                  (if Tn.equal lit tn then "a literal"
                   else "an @| slice view of " ^ Tn.debug_name lit ^ ", a literal")))

(* gh-ocannl-2022: {!check_host_write_lasts}'s rule for a {!copy} into an ordinary buffer. A copy
   into a node [dst]'s lineage fetch-defines lasts only until the routine carrying the fetch next
   runs, so it is refused unless it brings the fetch's own value -- then the fetch re-establishes
   what is already there. Provenance, from what the ledger already keeps: a source holding no buffer
   of the node is served by the host round trip, which reads its registered host-init values or
   recomputes it from its own forward, neither of which a host write reaches ([host_buffer] is fresh
   per write); a source holding one must have it last written ([current_writers]) by a run of the
   routine of its own lineage that carries the fetch. A seed, a host write, a copy and a sibling's
   write to the same buffer all drop that entry, and a source compiled with the forward but not run
   still holds what was there before. Accepted regardless: a copy between two contexts binding the
   same buffer moves nothing, and a literal with registered host-init data always holds its
   registered values (gh-ocannl-2021 refuses host writes to it; its link-time upload leaves no
   [current_writers] entry). That is {!literal_of}'s contract, [Tn.known_host_constant], on the node
   itself and only with [Host_inits] data: unlike a write, a copy reads the source's buffer, and
   only the link-time upload puts the registered values in a buffer no run wrote. A view's buffer is
   its slice's fetch, and a [TDSL.number] registers no host-init data, so a source compiled with
   their fetch and not run holds nothing of the literal: those keep the provenance rule. The node's
   own uid is looked up, a view included: a view that is not an alias is fetch-defined itself, by
   its slice, and the device path writes its own buffer. Runs before the copy dispatches, so a
   refused copy allocates, reads and records nothing. *)
let check_copy_lasts ~src ~dst (tn : Tn.t) =
  match Map.find dst.frontier.fetch_defined tn.Tn.uid with
  | None -> ()
  | Some _ when Tn.known_host_constant tn && Ir.Host_inits.mem tn -> ()
  | Some (id, name) -> (
      let src_key = buffer_key src tn in
      let no_op =
        Option.is_some src_key
        && Option.equal (fun a b -> Buffer_key.compare a b = 0) src_key (buffer_key dst tn)
      in
      let src_fetch = Map.find src.frontier.fetch_defined tn.Tn.uid in
      let reason =
        match (src_key, src_fetch) with
        | _ when no_op -> None
        | None, _ -> None
        | Some key, Some (src_id, src_name) -> (
            match Map.find src.ledger.current_writers key with
            | Some (Routine_writer writer) when writer = src_id -> None
            | _ ->
                Some
                  (Printf.sprintf
                     "the source's buffer of it was not last written by a run of %s (id=%d), the \
                      routine of the source's lineage carrying the same fetch: it holds a seed, a \
                      host write or a copy, or the routine has not run since"
                     src_name src_id))
        | Some _, None ->
            Some
              "the source's lineage carries no fetch of it, so the source's buffer holds a value \
               the caller put there"
      in
      match reason with
      | None -> ()
      | Some reason ->
          raise
          @@ Utils.User_error
               (Printf.sprintf
                  "Context.copy: node %s is fetch-defined in the destination's lineage: routine %s \
                   (id=%d) carries its forward code, a fetch, so a copy of any other value would \
                   be silently undone by its next run (gh-ocannl-2022, gh-ocannl-1959), and %s. A \
                   copy into it is accepted when it brings the fetch's own value: from a context \
                   holding no buffer of the node, or one whose buffer a run of its own fetching \
                   routine last wrote. A fetch-defined node -- a Tensor.term_init or literal with \
                   at most limit_constant_fill_size values, a range, an embedded symbol -- is \
                   code, not data: build a node whose values move between contexts data-backed, \
                   with Operation.init or Train.host_scalar, or make it a parameter."
                  (Tn.debug_name tn) name id reason))

(** Uploads the host buffer [nd] into [tn]'s device buffer, allocating it if needed, and returns a
    context in which [tn] is marked initialized (so a subsequent {!run} reading [tn] succeeds). *)
let upload ctx (tn : Tn.t) (nd : Nd.t) : t =
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

let from_host ctx (tn : Tn.t) (nd : Nd.t) : t =
  check_live ~fn:"from_host" ctx;
  check_host_write_lasts ~fn:"from_host" ctx tn;
  upload ctx tn nd

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
  (* gh-ocannl-2022: before [pair_contexts] and any transfer construction, which may allocate. A
     merge-buffer fill writes the transient slab, not the node. *)
  (match into_merge_buffer with
  | BI.No -> check_copy_lasts ~src ~dst tn
  | BI.Copy -> ());
  (* The fallback also serves nodes with no device buffer in [src]: [to_host] reads host-init
     literals and recomputes virtual nodes. A merge buffer cannot be filled host-side, so [Copy]
     raises where the fallback would engage. *)
  let host_roundtrip what =
    match into_merge_buffer with
    | BI.No -> upload dst tn (to_host src tn)
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
      | Some Released_writer -> None
      | Some (Routine_writer writer_id) when Set.mem src.ledger.executed writer_id -> None
      | Some (Routine_writer writer_id) -> Some writer_id
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
    (* gh-ocannl-1654: the retired transfer leaves no record behind -- its name included, so a long
       merge loop keeps the ledger bounded. Its id is never reused: a consumer compiled against it
       keeps depending on it, and is refused as depending on a superseded fill (see
       [check_lineage_runnable]), the only kind of id the ledger forgets the name of. *)
    Option.iter dst.ledger.current_merge_buffer_writer ~f:(fun old_transfer_id ->
        dst.ledger.executed <- Set.remove dst.ledger.executed old_transfer_id;
        Hashtbl.remove dst.ledger.merge_buffer_readers old_transfer_id;
        Hashtbl.remove dst.ledger.routine_names old_transfer_id);
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
             independently rooted contexts can share it (a device-wide host literal, gh-ocannl-1654)
             while keeping separate ledgers, so the destination's ledger learns the buffer holds a
             value -- the
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
  check_host_write_lasts ~fn:"set_values" ctx tn;
  let nd = host_buffer tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  Nd.set_flat_values ?padding nd values;
  upload ctx tn nd

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
  check_live ~fn:"set_value" ctx;
  check_host_write_lasts ~fn:"set_value" ctx tn;
  let nd = to_host ctx tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  Nd.set_from_float ?padding nd idx v;
  upload ctx tn nd

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

(* gh-560: the analyze-only entry points — lowering and optimization without backend codegen or
   linking. [Backends.lower_assignments] forks the lineage state itself, so the result is read off a
   hermetic sibling: the argument context, its ledger and frontier are unaffected. With the analysis
   cache (gh-560), a context that already compiled this routine (e.g. the tuner's arms) pays only
   the [specialize_proc] replay here. *)
let lowered_for_decisions ?name ?(materialized = []) ?(inline = []) ?(footprint = []) ?facts ?limits
    ctx comp bindings =
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
    Backends.lower_assignments
      ~facts:(Option.value_or_thunk facts ~default:(fun () -> backend_facts ctx))
      ~limits:(Option.value_or_thunk limits ~default:(fun () -> hardware_limits ctx))
      optim_ctx ?name bindings comp.Asgns.asgns
  in
  lowered

let lowered_transform_input = Backends.maybe_sink_zeros

let decision_surface ?name ctx comp bindings =
  (lowered_for_decisions ?name ctx comp bindings).Ir.Low_level.flip_candidates

(* The child a decision is recorded in: the lineage state forked exactly like a compile would, then
   [record] in the fork, so the argument context and its other descendants are unaffected. *)
let decided ~fn ctx record =
  (* A child of a dead handle would inherit its freed buffer locations under a fresh, unset flag. *)
  check_live ~fn ctx;
  let wrapped, () =
    Backends.with_backend ctx.wrapped
      {
        f =
          (fun (type d r e) ((module Backend) : (d, r, e) Backends.backend_module) bctx ->
            let optimize_ctx = Ir.Low_level.copy_optimize_ctx bctx.BI.optimize_ctx in
            record optimize_ctx;
            (Backend.make_child ~optimize_ctx bctx, ()));
      }
  in
  derive ctx wrapped

let decide_materialized ctx tns =
  decided ~fn:"decide_materialized" ctx (fun optimize_ctx ->
      Ir.Low_level.decide_materialized optimize_ctx tns)

(* The preference is recorded rather than a placement decided, because inlining legality is settled
   only during optimization ([check_and_store_virtual]) — a preferred node the virtualizer rejects
   still materializes. A node whose placement THIS lineage already decided (e.g. a cap
   materialization from an earlier compile of a routine setting it) keeps that decision: decisions
   are final within a lineage — already-compiled routines depend on them (a consumer compiled
   against the node's buffer must find it written) — so the preference only steers placements not
   yet decided. Callers wanting the exemption to take effect fork a pre-compile sibling, as
   [Train.tune_placements] does. *)
let decide_inline ctx tns =
  decided ~fn:"decide_inline" ctx (fun optimize_ctx -> Ir.Low_level.prefer_inline optimize_ctx tns)

(* A preference like [decide_inline], for the same reasons: the footprint form is served per read
   site by the virtualizer, which may still decline one. *)
let decide_footprint ctx tns =
  decided ~fn:"decide_footprint" ctx (fun optimize_ctx ->
      Ir.Low_level.prefer_footprint optimize_ctx tns)

let decide_reading ctx (reading : Ir.Low_level.reading) tns =
  match reading with
  | `Materialize -> decide_materialized ctx tns
  | `Inline -> decide_inline ctx tns
  | `Footprint -> decide_footprint ctx tns

(* gh-ocannl-1859: one child for all three, in the order the separate recorders would apply them: a
   chain of three would leave the wrapper that adopts what it minted holding only the last. *)
let decide_readings ?(skip_empty = true) ctx (materialized, inline, footprint) =
  if skip_empty && List.for_all [ materialized; inline; footprint ] ~f:List.is_empty then ctx
  else
    decided ~fn:"decide_readings" ctx (fun optimize_ctx ->
        Ir.Low_level.decide_materialized optimize_ctx materialized;
        Ir.Low_level.prefer_inline optimize_ctx inline;
        Ir.Low_level.prefer_footprint optimize_ctx footprint)

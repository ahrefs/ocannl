open Base
module Lazy = Utils.Lazy
module Nd = Ir.Ndarray
module Tn = Ir.Tnode
module Asgns = Ir.Assignments
module Idx = Ir.Indexing

type ndarray = Nd.t
type tn = Tn.t
type tn_set = Set.M(Tn).t
type asgns = Asgns.t
type comp = Asgns.comp
type fetch_op = Asgns.fetch_op

type projections = {
  projections_debug : string;
  projections : Ir.Indexing.projections Lazy.t;
  product_shape : Shape.t Lazy.t;
      (** The product-space proxy shape of the operation (gh-512), for [%cd] [*_pspace]
          intermediates; see {!Shape.product_space_shape}. Forcing it emits shape constraints, so it
          must be forced (if at all) while the operation's code is being built. Unlike
          [projections], it does not need the operation's shape update step, which {!op} creates
          only after [op_asn] returns: forcing [projections] from within an [op_asn] is rejected. *)
}

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_TENSOR=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_TENSOR"]

type diff = { grad : (Tn.t[@sexp.opaque]); zero_grads : Asgns.comp; backprop : Asgns.comp }
[@@deriving sexp_of]

(* What became of the tensor's code once it stopped being a root. A root leaves the session's root
   map when a consumer embeds it, when it is handed out ([consume_forward_code], [take_forward_code]
   -- the [%cd] embedding path -- or [consume_backprop_code]); repeated handout is allowed for
   [Taken], so every non-embedding removal records its kind here. A second copy of handed-out code
   in one compilation lineage is the lineage's to refuse (gh-ocannl-1724). The cell is mutable and
   shared by record copies that preserve code ownership; [param] starts a fresh cell when it changes
   ownership. Markers stay on tensors rather than accumulating in session-global storage. *)
type handout = Not_handed_out | Taken [@@deriving sexp_of, equal]

(* gh-ocannl-1796: the forward owner cell of a tensor -- a GC-independent shadow of "this tensor is
   a forward root", which [consume_forward_code]'s conflicting-roots check reads instead of the
   session's (weak) forward-root map. [owners] are the cells of the tensors whose forward code
   embeds this tensor's code: an edge per embedding, never a chain's top, so a computation replayed
   into two roots is owned by both and neither owns the other's result. A read is pending while any
   cell reachable through [owners] is a root. A cell holds only cells and its tensor's stable name:
   never a tensor record, a comp, a [consumption] cell or a node, so a reader's cells keep no graph
   alive. A root dropped before its code was consumed keeps [is_root]: its code is gone, so its
   readers stay refused whether or not the GC reclaimed it. *)
type owner_cell = {
  oc_uid : int;  (** Unique per cell, for the traversal's visited set. *)
  oc_name : string;  (** The tensor's [Tn.label_name], for the refusal. *)
  mutable is_root : bool;
  mutable resolved_epoch : int;
      (** The closure epoch in which [is_root] was last cleared: an owner resolved before the last
          closure can never become a root again, so it can be compacted away. *)
  mutable owners : owner_cell list;
}

type consumption = {
  mutable fwd : handout;
  mutable bprop : handout;
  fwd_owner : (owner_cell[@sexp.opaque]);
}
[@@deriving sexp_of]

module rec Self : sig
  type t = {
    params : (t, Self_comparator.comparator_witness) Set.t;
    forward : comp;
    diff : diff option;
    value : tn;
    top_down_prec : bool;
    shape : Shape.t;
    children : subtensor list;
    consumption : consumption;
  }
  [@@deriving sexp_of]

  and subtensor = { subtensor : t; embedded : bool }

  val compare : t -> t -> int
end = struct
  type t = {
    params : (t, Self_comparator.comparator_witness) Set.t;
    forward : comp;
    diff : diff option;
    value : tn;
    top_down_prec : bool;
    shape : Shape.t;
    children : subtensor list;
    consumption : consumption;
  }

  and subtensor = { subtensor : t; embedded : bool }

  let rec sexp_of_t t =
    Sexp.message "Tensor"
      [
        ("id", sexp_of_int t.value.Tn.id);
        ("label", [%sexp_of: string list] t.value.label);
        ("forward", [%sexp_of: Asgns.comp] t.forward);
        ("diff", [%sexp_of: diff option] t.diff);
        ("children", [%sexp_of: subtensor list] t.children);
      ]

  and sexp_of_subtensor ch =
    Sexp.message "child"
      [
        (if ch.embedded then ("", sexp_of_t ch.subtensor)
         else ("ref-id", sexp_of_int ch.subtensor.value.Tn.id));
      ]

  (* Tensor identity follows the value node's process-unique [uid] (never reused across
     [unsafe_reinitialize], unlike the session-scoped [value.id]). *)
  let compare t1 t2 = Int.compare t1.value.Tn.uid t2.value.Tn.uid
end

and Self_comparator : (Comparator.S with type t := Self.t) = Comparator.Make (Self)

include Self
include Self_comparator

module T = struct
  include Self
  include Self_comparator
end

let sexp_of_comparator_witness _ = Sexp.Atom "comparator_witness"

(* gh-ocannl-1745: a backprop-root entry does not own its tensor. The entry is an ephemeron keyed by
   the tensor's [consumption] cell -- the code-ownership identity every ownership-preserving record
   copy shares -- with the registered tensor as its data, plus a weak pointer to the key so the
   session can read the entry without holding it. The entry lives exactly while some such copy is
   reachable from outside the root maps; once the caller drops the graph, the GC reclaims it and the
   entry reads as absent. Entries are immutable once made: the persistent maps holding them are
   shared with checkpoints.

   gh-ocannl-1804: a backprop root also remembers the leaves it absorbed, transitively, as the
   leaves' own entries: a weak handle each, shared rather than copied, so absorbing costs one node
   and an absorbed computed root's entry is not retained. A root that dies while still in the map
   can never run, so {!prune} gives those leaves back. A root that left the map gave nothing: its
   code was handed out, or a live consumer embedded it, and that code still owns the zeroing. *)
module Weak_root : sig
  type entry
  type map = entry Map.M(Int).t

  val make : ?absorbed:entry list -> t -> entry
  (** [absorbed]: the entries of the backprop roots whose code [t]'s backprop embeds. *)

  val cell : entry -> owner_cell
  (** The tensor's forward owner cell, held strongly: a restore resets the cell of a root the GC
      reclaimed as deterministically as a live one's (gh-ocannl-1796). *)

  val grad_uid : entry -> int option
  (** The gradient of the tensor registered, readable after the GC reclaimed it. *)

  val find : map -> int -> entry option
  (** Present and live. *)

  val mem : map -> int -> bool
  (** Present and live. *)

  val live : map -> t list

  val is_leaf : t -> bool
  (** Differentiable, with a backprop owning no fragment: a term, parameters included. *)

  val prune : map -> map
  (** Drops the dead entries and gives back the live leaves they absorbed, never replacing an entry
      in the map; physically the same map when there are no dead entries. *)
end = struct
  type entry = {
    anchor : consumption Stdlib.Weak.t;
    root : (consumption, t) Stdlib.Ephemeron.K1.t;
    cell : owner_cell;
    grad_uid : int option;
    is_leaf : bool;
    leaves : leaves;
  }

  and leaves = One of entry | Many of leaves list

  type map = entry Map.M(Int).t

  (* The test [Tensor.op] applies for gh-ocannl-1724's [requires]. *)
  let is_leaf t =
    match t.diff with
    | Some d -> not (Set.mem d.backprop.owned (d.grad, Asgns.Bprop))
    | None -> false

  let make ?(absorbed = []) t =
    let anchor = Stdlib.Weak.create 1 in
    Stdlib.Weak.set anchor 0 (Some t.consumption);
    let leaves =
      List.filter_map absorbed ~f:(fun e ->
          match e with
          | { is_leaf = true; _ } -> Some (One e)
          | { leaves = Many []; _ } -> None
          | { leaves; _ } -> Some leaves)
    in
    {
      anchor;
      root = Stdlib.Ephemeron.K1.make t.consumption t;
      cell = t.consumption.fwd_owner;
      grad_uid = Option.map t.diff ~f:(fun d -> d.grad.Tn.uid);
      is_leaf = is_leaf t;
      leaves = Many leaves;
    }

  let cell e = e.cell
  let grad_uid e = e.grad_uid

  let get { anchor; root; _ } =
    match Stdlib.Weak.get anchor 0 with
    | None -> None
    | Some key -> Stdlib.Ephemeron.K1.query root key

  let find m id = Option.bind (Map.find m id) ~f:(fun e -> Option.map (get e) ~f:(fun _ -> e))
  let mem m id = Option.is_some (find m id)
  let live m = List.filter_map (Map.data m) ~f:get

  let rec hand_back m = function
    | Many leaves -> List.fold leaves ~init:m ~f:hand_back
    | One e -> (
        match get e with
        | Some t when not (Map.mem m t.value.id) -> Map.add_exn m ~key:t.value.id ~data:e
        | Some _ | None -> m)

  let prune m =
    if Map.for_all m ~f:(fun e -> Option.is_some (get e)) then m
    else
      let dead, live = Map.partition_tf m ~f:(fun e -> Option.is_none (get e)) in
      Map.fold dead ~init:live ~f:(fun ~key:_ ~data m -> hand_back m data.leaves)
end

(* The session's graph state. Its writes are held to the mutation-serial obligation field by field
   (gh-ocannl-1573): the root maps count as session mutations, the id counter does not.

   Both root maps are weak ({!Weak_root}): backprop roots since gh-ocannl-1745, forward roots since
   gh-ocannl-1796. A graph the caller dropped is reclaimed whether or not its code was ever
   evaluated, and a checkpoint's capture of the maps holds nothing either. [consume_forward_code]'s
   conflicting-roots check, the only guard against compiling a reader of a node whose forward code
   an unconsumed root owns, does not read the map: it reads the owner cells ({!owner_cell}), which
   remember a dropped unconsumed root as pending, so the refusal does not depend on GC timing. *)
type session_state = {
  mutable next_id : int;
      [@mutation_serial.exempt
        "a fresh-id counter: ids are never reused, so a rewind need not restore it (an id-neutral \
         window, Scheme.with_id_neutral, allocates from a reserved range and restores it)"]
  mutable forward_roots : Weak_root.map;
  mutable backprop_roots : Weak_root.map;
}

let session_state =
  { next_id = 0; forward_roots = Map.empty (module Int); backprop_roots = Map.empty (module Int) }

let bump_next_id id = session_state.next_id <- max session_state.next_id (id + 1)
let get_next_id () = session_state.next_id
let is_fwd_root t = Weak_root.mem session_state.forward_roots t.value.id

(** {2 Forward owner cells (gh-ocannl-1796)} *)

(* Bumped by every closure, before it captures its checkpoint: an owner cell resolved in an earlier
   epoch was not a root at the last closure, so no restore can make it one again. *)
let closure_epoch = ref 0
[@@mutation_serial.exempt "a closure counter: it only grows, and no rewind restores an older one"]

(* While a [with_unchanged_roots] scope is open, its restore must find every forward root its body
   created in the current map, and every owner edge intact: the forward map is not pruned and owner
   lists are not compacted. *)
let unchanged_roots_depth = ref 0
[@@mutation_serial.exempt
  "the nesting of with_unchanged_roots scopes, restored by the scope's own finally"]

let next_cell_uid = ref 0
[@@mutation_serial.exempt "a fresh-id counter for owner cells: ids are never reused"]

let fresh_owner_cell (tn : Tn.t) =
  Int.incr next_cell_uid;
  {
    oc_uid = !next_cell_uid;
    oc_name = Tn.label_name tn;
    is_root = false;
    resolved_epoch = !closure_epoch;
    owners = [];
  }

(* The owner cell of each value node, and of each node an operation's own code embeds: the cell of
   the tensor whose forward code computes it. Keyed weakly by the node, so an entry lives while
   anything can still read the node -- in particular while a reader's code does. *)
module Owner_cells = Stdlib.Ephemeron.K1.Make (struct
  type t = Tn.t

  let equal = phys_equal
  let hash (tn : Tn.t) = Int.hash tn.uid
end)

let owner_cells : owner_cell Owner_cells.t = Owner_cells.create 256
[@@mutation_serial.exempt
  "a weak registry of each node's owner cell: constructions add entries for the nodes they create, \
   which a rewind discards with their nodes, so it restores nothing"]

let set_not_root c =
  if c.is_root then (
    c.is_root <- false;
    c.resolved_epoch <- !closure_epoch)

(* An owner that can never claim a node again: not a root, resolved before the last closure (so no
   restore revives it), and owned by nothing (its own list as last compacted). Dropping it keeps a
   long-held node's owner list to the claims of one closure period. [compact] is one level;
   [compact_deep] and [pending_owners] compact bottom-up over the cells above, each cell once per
   [seen] set. Not while a [with_unchanged_roots] scope is open. *)
let stale c = (not c.is_root) && c.resolved_epoch < !closure_epoch && List.is_empty c.owners

let compact c =
  if !unchanged_roots_depth = 0 && not (List.is_empty c.owners) then
    c.owners <- List.filter c.owners ~f:(Fn.non stale)

(* [seen] is created on first need: most cells gain their first owner with an empty list. *)
let rec compact_deep ~(seen : Hash_set.M(Int).t option ref) c =
  if !unchanged_roots_depth = 0 && not (List.is_empty c.owners) then
    let visited =
      match !seen with
      | Some visited -> visited
      | None ->
          let visited = Hash_set.create (module Int) in
          seen := Some visited;
          visited
    in
    if not (Hash_set.mem visited c.oc_uid) then (
      Hash_set.add visited c.oc_uid;
      List.iter c.owners ~f:(compact_deep ~seen);
      compact c)

let add_owner ~seen c ~owner =
  if not (phys_equal c owner || List.mem c.owners owner ~equal:phys_equal) then (
    compact_deep ~seen c;
    c.owners <- owner :: c.owners)

(* [tn] is computed by the code of the tensor owning [cell]: a fresh node is registered to it, a
   node some other tensor's code computes gains [cell] as an owner of that tensor's cell. *)
let register_owned ~seen ~cell (tn : Tn.t) =
  match Owner_cells.find_opt owner_cells tn with
  | None -> Owner_cells.replace owner_cells tn cell
  | Some c -> add_owner ~seen c ~owner:cell

(* The root cells reachable from the read nodes' cells, other than [self]'s, in creation order. *)
let pending_owners ~self (reads : Set.M(Tn).t) =
  let seen = Hash_set.create (module Int) in
  let found = ref [] in
  let rec visit c =
    if not (Hash_set.mem seen c.oc_uid) then (
      Hash_set.add seen c.oc_uid;
      if c.is_root && not (phys_equal c self) then found := c :: !found;
      List.iter c.owners ~f:visit;
      compact c)
  in
  Set.iter reads ~f:(fun tn -> Option.iter (Owner_cells.find_opt owner_cells tn) ~f:visit);
  List.sort !found ~compare:(fun c1 c2 -> Int.compare c1.oc_uid c2.oc_uid)

(* Clears the root marks of the current map's cells and sets those of [restored]'s, dead entries
   included: an entry holds its cell, so a root the GC reclaimed is reset like a live one. *)
let restore_root_cells ~restored =
  Map.iter session_state.forward_roots ~f:(fun e -> set_not_root (Weak_root.cell e));
  Map.iter restored ~f:(fun e -> (Weak_root.cell e).is_root <- true)

(* The tensors made while a traced [%op] body runs (gh-ocannl-1801), newest first: the candidates
   for its results, held strongly until the trace reads them. [None] outside such a body. *)
let made_by_traced_body : t list ref option ref = ref None
[@@mutation_serial.exempt
  "the result candidates of a running trace, restored by Scheme.traced_results"]

(* Root-map and handout changes count as session mutations (gh-ocannl-1301): a construction that
   embedded a root and then failed has to be rewound. Every tensor made is installed here. *)
let add_fwd_root t =
  Row.note_mutation ();
  Option.iter !made_by_traced_body ~f:(fun made -> made := t :: !made);
  session_state.forward_roots <-
    Map.add_exn session_state.forward_roots ~key:t.value.id ~data:(Weak_root.make t);
  t.consumption.fwd_owner.is_root <- true

(* The cell cleared is the map entry's: [param] passes a record with a fresh [consumption]. *)
let remove_fwd_root t =
  Row.note_mutation ();
  Option.iter (Map.find session_state.forward_roots t.value.id) ~f:(fun e ->
      set_not_root (Weak_root.cell e));
  session_state.forward_roots <- Map.remove session_state.forward_roots t.value.id

(* The map as it stands. [Tensor.op] reads this, after settling its operands once: see there. *)
let in_bprop_roots t = Weak_root.mem session_state.backprop_roots t.value.id

(* gh-ocannl-1804: a reclaimed root's leaves go back before anything reads the map for them. A
   closure does this too, but the next consumer need not come after one. *)
let hand_back_reclaimed () =
  let pruned = Weak_root.prune session_state.backprop_roots in
  if not (phys_equal pruned session_state.backprop_roots) then (
    Row.note_mutation ();
    session_state.backprop_roots <- pruned)

let is_bprop_root t =
  hand_back_reclaimed ();
  in_bprop_roots t

let remove_bprop_root t =
  Row.note_mutation ();
  session_state.backprop_roots <- Map.remove session_state.backprop_roots t.value.id

let take_forward_code t =
  remove_fwd_root t;
  t.consumption.fwd <- Taken;
  t.forward

(** {2 Checkpoints and rewind (gh-ocannl-1301, gh-ocannl-903)}

    A checkpoint snapshots the open frontier ({!Shape.Checkpoint} holds the shape-inference half):
    here, the graph's module-level state -- the root maps, the random-seed singleton and the open
    nodes, persistent values captured as pointers by [capture_globals] -- and every mutable field of
    each open node ({!Tn.snapshot_open}). A rewind restores them and enters the nodes the window
    allocated in [discarded_nodes], a weak registry; every use of a discarded tensor or node is
    refused with the rewind's provenance. The nodes are tracked exactly rather than by an id/uid
    range (gh-ocannl-1505): a node created in the window by anyone else -- a hand-made [Tn.create],
    a [Persistence.load] -- may carry an id inside the window's range, and is not open inference
    work.

    [last_closure] is recorded at the end of every successful [Shape.finish_inference] (through
    [Shape.set_on_closure]): with an empty frontier, it is the base checkpoint
    [rewind_to_last_closure] returns to. *)

(* Global singleton for random seed, used in init_params and random number generation *)
let random_seed : t option ref = ref None

(* The distinct seed values [set_random_seed] was given this session, newest first: a graph built
   under a scoped seed ([with_saved_random_seed]) keeps drawing from it after the singleton is
   restored, so the singleton alone does not say which seeds the session's graphs draw from
   (gh-ocannl-1727). The values, not the seed tensors: holding those would keep every dropped
   graph's seed for the session (gh-ocannl-1859), and holding them weakly would make what a
   checkpoint records depend on when the GC ran. *)
let random_seeds_created : int list ref = ref []

(* The nodes the frontend allocated since the last closure, newest first ([note_open_node] at every
   frontend [Tn.create]). A closure closes every shape of the open work, so it empties this. *)
let open_nodes : Tn.t list ref = ref []
[@@mutation_serial.exempt
  "a record of what the open work allocated: an entry left by a failure that did not taint is a \
   node nobody holds, which a later rewind discards and a closure forgets"]

let note_open_node (tn : Tn.t) = open_nodes := tn :: !open_nodes

(* External handouts since the last closure: code given to a caller ([consume_forward_code],
   [consume_backprop_code]) is a closed-world act, so a rewind never resurrects a root whose code
   the caller holds. Forward code by the value node's id; backprop code by the gradient node's uid,
   since a parameter over an initializer shares its value node but owns a fresh gradient. Embeddings
   by constructions are not recorded -- those constructions are what a rewind discards. *)
type handouts = {
  mutable fwd_out : Set.M(Int).t;
      [@mutation_serial.kept "a record of code given to callers, which no rewind takes back"]
  mutable bprop_out : Set.M(Int).t;
      [@mutation_serial.kept "a record of code given to callers, which no rewind takes back"]
}

let handouts = { fwd_out = Set.empty (module Int); bprop_out = Set.empty (module Int) }

let note_external_handout ~fwd t =
  if fwd then handouts.fwd_out <- Set.add handouts.fwd_out t.value.id
  else
    Option.iter t.diff ~f:(fun d -> handouts.bprop_out <- Set.add handouts.bprop_out d.grad.Tn.uid)
[@@mutation_serial.boundary
  "a closed-world act: records that work left the session, so a rewind does not resurrect it; not \
   open inference state"]

(* The graph's module-level session state, captured as pointers. Every such binding of this file is
   captured here, or classified [kept] or [exempt] (test/operations/mutation_serial_scan holds the
   relationship). *)
type globals = {
  g_forward_roots : Weak_root.map;
  g_backprop_roots : Weak_root.map;
  g_random_seed : t option;
  g_random_seeds_created : int list;
  g_open_nodes : Tn.t list;
}

let capture_globals () =
  {
    g_forward_roots = session_state.forward_roots;
    g_backprop_roots = session_state.backprop_roots;
    g_random_seed = !random_seed;
    g_random_seeds_created = !random_seeds_created;
    g_open_nodes = !open_nodes;
  }

type checkpoint = {
  ck_shape : Shape.Checkpoint.t;
  ck_globals : globals;
  ck_nodes : Tn.open_snapshot list;  (** The open nodes' fields; none for a closure's. *)
}

let take_checkpoint ~closure ~label =
  let id_mark = session_state.next_id in
  {
    ck_shape =
      (if closure then Shape.closure_checkpoint ~id_mark else Shape.checkpoint ~label ~id_mark);
    ck_globals = capture_globals ();
    ck_nodes = (if closure then [] else List.map !open_nodes ~f:Tn.snapshot_open);
  }

let last_closure = ref (take_checkpoint ~closure:true ~label:"")
[@@mutation_serial.kept
  "the last closure's checkpoint: a closure records it, and no rewind restores an older one"]

let record_checkpoint () =
  (* The dead backprop-root entries of graphs the caller dropped (gh-ocannl-1745) are pruned at the
     closure, before it is captured: a forward-only evaluation loop closes on every compile. Pruning
     changes the status of no root anyone can name, but the leaves those entries absorbed are roots
     again (gh-ocannl-1804). *)
  session_state.backprop_roots <- Weak_root.prune session_state.backprop_roots;
  (* So are the dead forward roots (gh-ocannl-1796): their cells keep the roots' status, and no
     restore reads them -- except an open [with_unchanged_roots] scope's, which needs the roots its
     body created, so it defers the pruning. The epoch moves first: an owner cell resolved before
     this closure was not a root at it. *)
  Int.incr closure_epoch;
  if !unchanged_roots_depth = 0 then
    session_state.forward_roots <- Weak_root.prune session_state.forward_roots;
  open_nodes := [];
  handouts.fwd_out <- Set.empty (module Int);
  handouts.bprop_out <- Set.empty (module Int);
  last_closure := take_checkpoint ~closure:true ~label:""
[@@mutation_serial.boundary
  "the closure's record: run at the commit of a successful finish_inference, and to re-anchor \
   after a rewind or a session reset"]

let () = Shape.set_on_closure record_checkpoint

(* The nodes discarded by rewinds, with the provenance of the rewind that discarded each, held
   weakly (gh-ocannl-1789): an entry lasts only while its node is reachable, so a long session's
   rewinds -- a scheme traced per closure, a REPL re-running cells -- accumulate nothing beyond the
   stale handles its caller still holds. Keyed by uid (never reused, across namespaces and
   reinitializations alike). Only the nodes the frontend allocated are entered (a node made with
   [Tn.create] directly, whatever its id, is not the frontend's). *)
module Weak_discards = Stdlib.Ephemeron.K1.Make (struct
  type t = Tn.t

  let equal (tn1 : Tn.t) (tn2 : Tn.t) = tn1.uid = tn2.uid
  let hash (tn : Tn.t) = Int.hash tn.uid
end)

let discarded_nodes_table : string Weak_discards.t option ref = ref None
[@@mutation_serial.kept
  "a registry of the nodes rewinds discarded, held weakly: rewinds extend it, never restore it"]

let discarded_nodes () =
  match !discarded_nodes_table with
  | Some table -> table
  | None ->
      let table = Weak_discards.create 64 in
      discarded_nodes_table := Some table;
      table
[@@mutation_serial.boundary
  "the weak registry's creation on first use: an empty registry, which no rewind restores"]

(* The seed literals [set_random_seed] created, each with its forward code -- the [Constant_bits]
   fetch writing it -- held weakly by the node (gh-ocannl-2119). A graph built under a scoped or
   replaced singleton keeps reading its own seed, which may no longer be the singleton; this is how
   [init_params] recognizes such a seed among the nodes its program reads, and replays its literal.
   Holding the entries weakly keeps no dropped graph's seed alive (gh-ocannl-1859). *)
let seed_literals : Asgns.comp Weak_discards.t = Weak_discards.create 16
[@@mutation_serial.kept
  "a weak registry of seed literals and their constant fetches: [set_random_seed] extends it, and \
   no rewind restores it -- a discarded seed is refused by the validity check, not by its absence"]

(* Id-neutral windows (gh-ocannl-1751), see [Scheme.with_id_neutral]: a window that is always
   rewound -- a shape-scheme trace, a definition check -- allocates its tensors' ids from a reserved
   range above every session id. Every node such a window allocated is discarded when it ends, and
   then reported under the window's own label. *)
let reserved_id_base = Shape.reserved_id_base

let id_neutral_depth = ref 0
[@@mutation_serial.exempt "the nesting of id-neutral windows, restored by their own finally"]

(* The nodes rewinds discarded in the current id-neutral window. *)
let window_discarded_nodes : Tn.t list ref = ref []
[@@mutation_serial.exempt "scratch of the current id-neutral window, emptied when it ends"]

let discarded_by (tn : Tn.t) = Weak_discards.find_opt (discarded_nodes ()) tn

let with_unchanged_roots ~f =
  let fwd_roots = session_state.forward_roots in
  let bprop_roots = session_state.backprop_roots in
  let fwd_out = handouts.fwd_out and bprop_out = handouts.bprop_out in
  let closure = !last_closure in
  Int.incr unchanged_roots_depth;
  let finally () =
    Int.decr unchanged_roots_depth;
    (* The owner cells follow the maps, the roots [f] created and dropped included: the scope kept
       them in the map ({!record_checkpoint}), so the result does not depend on the GC. *)
    restore_root_cells ~restored:fwd_roots;
    session_state.forward_roots <- fwd_roots;
    session_state.backprop_roots <- bprop_roots;
    (* Handouts inside [f] are undone, so their records are too (gh-ocannl-1301); after a closure
       inside [f], the restored maps are the closed state's roots. *)
    if phys_equal !last_closure closure then (
      handouts.fwd_out <- fwd_out;
      handouts.bprop_out <- bprop_out)
    else (
      (* The closure restarted the record; what [f] handed out since is undone with the maps. *)
      handouts.fwd_out <- Set.empty (module Int);
      handouts.bprop_out <- Set.empty (module Int);
      last_closure :=
        {
          !last_closure with
          ck_globals =
            {
              !last_closure.ck_globals with
              g_forward_roots = fwd_roots;
              g_backprop_roots = bprop_roots;
            };
        });
    (* A root's code can only have been taken inside [f] if it was a root at entry, and a root is by
       construction not yet taken: restoring the maps restores the markers. *)
    List.iter (Weak_root.live fwd_roots) ~f:(fun t -> t.consumption.fwd <- Not_handed_out);
    List.iter (Weak_root.live bprop_roots) ~f:(fun t -> t.consumption.bprop <- Not_handed_out)
      [@@mutation_serial.boundary
        "a scoped restore: puts back the roots saved at entry, on return and on a raise alike"]
  in
  Exn.protectx ~f ~finally ()

let iter_embedded ~f t =
  Set.iter ~f t.forward.embedded_nodes;
  Option.iter t.diff ~f:(fun diff -> Set.iter ~f diff.backprop.embedded_nodes)

let%debug7_sexp rec init_params_body ?skip (t : t) : Asgns.comp =
  let more_embedded = ref @@ Set.empty (module Tn) in
  let params : t list =
    Set.to_list t.params
    |> (match skip with
      | None -> Fn.id
      | Some skip -> List.filter ~f:(fun p -> not (Map.mem skip p.value)))
      (* Compare to ordered_ts in op -- we need to sort to avoid computed-after-use bugs! *)
    |> List.sort ~compare:(fun p1 p2 -> Int.ascending p1.value.id p2.value.id)
  in
  let asgns =
    Asgns.Block_comment
      ( "init params for " ^ Tn.label_name t.value,
        List.fold_right params ~init:Asgns.Noop ~f:(fun param acc ->
            if Set.is_empty param.params then Asgns.Seq (param.forward.asgns, acc)
            else
              let comp = init_params_body ?skip param in
              more_embedded := Set.union !more_embedded comp.Asgns.embedded_nodes;
              Seq (Seq (comp.Asgns.asgns, param.forward.asgns), acc)) )
  in
  let embedded_nodes =
    List.fold params ~init:!more_embedded ~f:(fun acc p ->
        Set.add (Set.union acc p.forward.embedded_nodes) p.value)
  in
  (* Another graph (including an unused branch) may have absorbed the singleton seed's initializer.
     When adding that seed for linking, also initialize the library-owned literal before the
     parameter draws (gh-ocannl-2023). This replays a Constant_bits fetch, not PRNG stream state:
     random operations derive their counters from node ids, offsets and explicit counter inputs.
     Other unwritten embedded reads still require a caller-provided write. This clause also writes
     the singleton for a forward graph reading it; a seed the initializers read that is no longer
     the singleton is [init_params]'s, below (gh-ocannl-2119). *)
  let embedded_nodes, asgns =
    match !random_seed with
    | None -> (embedded_nodes, asgns)
    | Some rs ->
        if (not (is_fwd_root rs)) && not (Set.mem embedded_nodes rs.value) then
          ( Set.add (Set.union embedded_nodes rs.forward.embedded_nodes) rs.value,
            Asgns.Seq (rs.forward.asgns, asgns) )
        else (embedded_nodes, asgns)
  in
  (* Initialization owns no fragment (gh-ocannl-1724): it is re-run on purpose ([reinit_all], and
     [Train.run_once] initializing missing parameters on every call), and the seed is a singleton --
     so the parameters' forward fragments are not carried here. *)
  Asgns.comp ~embedded:embedded_nodes asgns

(* A seed the selected initializers read is a literal whose code may live elsewhere: absorbed by a
   discarded draw, while the singleton has since moved on -- a [with_saved_random_seed] scope
   restored another one, or [set_random_seed] replaced it (gh-ocannl-2119). Every registered seed
   the program reads and does not write gets its [Constant_bits] fetch prepended, so the draws are
   those of a program owning the seed, whichever singleton is current. Checked once, on the whole
   program: a seed one included parameter's initializer writes is not missing for another's. *)
let init_params ?skip (t : t) : Asgns.comp =
  let comp = init_params_body ?skip t in
  let read, _ = Asgns.collect_nodes_guess_output comp.Asgns.asgns in
  let missing =
    Set.filter
      (Set.diff read (Asgns.collect_written comp.Asgns.asgns))
      ~f:(Weak_discards.mem seed_literals)
  in
  if Set.is_empty missing then comp
  else
    let embedded_nodes, asgns =
      Set.fold_right missing ~init:(comp.Asgns.embedded_nodes, comp.Asgns.asgns)
        ~f:(fun tn (embedded, asgns) ->
          let seed = Weak_discards.find seed_literals tn in
          ( Set.add (Set.union embedded seed.Asgns.embedded_nodes) tn,
            Asgns.Seq (seed.Asgns.asgns, asgns) ))
    in
    Asgns.comp ~embedded:embedded_nodes asgns

let initial_default_prec =
  Ir.Ops.prec_of_string (Utils.get_global_arg ~default:"single" ~arg_name:"default_prec")

let default_value_prec = ref initial_default_prec
[@@mutation_serial.exempt "user configuration read when a tensor is built, not inference state"]

let default_grad_prec = ref initial_default_prec
[@@mutation_serial.exempt "user configuration read when a tensor is built, not inference state"]

exception Session_error of string * t option [@@deriving sexp]

let session_error_printer = function
  | Session_error (msg, None) -> Some msg
  | Session_error (msg, Some m) ->
      Some
        ([%string "For #%{m.value.id#Int} %{Tn.debug_name m.value}: %{msg}"]
        [@message_only "the exception printer's message"])
  | _ -> None

let () = Stdlib.Printexc.register_printer session_error_printer

let discarded_message (tn : Tn.t) provenance =
  [%string
    "tensor node %{Tn.debug_name tn} was discarded by the rewind after %{provenance}; re-run its \
     definition"]
  [@message_only "the refusal its callers raise at once"]

(** Raises [Session_error] when [t] (its value or gradient node) was discarded by a rewind. Shared
    with the explicit checkpoints of gh-ocannl-903. *)
let check_not_discarded ~what t =
  let nodes = t.value :: Option.value_map t.diff ~default:[] ~f:(fun d -> [ d.grad ]) in
  List.iter nodes ~f:(fun tn ->
      Option.iter (discarded_by tn) ~f:(fun provenance ->
          raise @@ Session_error (what ^ ": " ^ discarded_message tn provenance, Some t)))

(** Sets an operand's precision ahead of an operation that requires it (e.g. [uint4x32] bits). A
    closed operand's precision is part of its committed state (gh-ocannl-1301): an update that would
    change it is refused before writing; an equal one (or a settled precision, which
    [Tn.update_prec] checks itself) passes. *)
let prec_update_is_noop t prec =
  let tn = t.value in
  if Lazy.is_val tn.Tn.storage_prec then Ir.Ops.equal_prec (Lazy.force tn.Tn.storage_prec) prec
  else match Tn.get_specified_prec tn with Some p -> Ir.Ops.equal_prec p prec | None -> false

(* A settled precision cannot change either ([Tn.update_prec] would raise): validating it here keeps
   a multi-operand update from writing one operand before another is refused. *)
let check_operand_prec t prec =
  let tn = t.value in
  if (t.shape.Shape.closed || Lazy.is_val tn.Tn.storage_prec) && not (prec_update_is_noop t prec)
  then
    raise
    @@ Session_error
         ( [%string
             "the operand %{Tn.debug_name tn} is closed (committed by an earlier \
              finish_inference), but this operation would set its precision to \
              %{Ir.Ops.prec_string prec}; build the operation before closing the operand"],
           Some t )

(** Sets the operands' precision; every operand is validated before any is written. *)
let update_operands_prec ts prec =
  (* Every refusal an operation would raise on these operands comes before the first write. *)
  Shape.check_untainted ();
  List.iter ts ~f:(check_not_discarded ~what:"operand");
  List.iter ts ~f:(fun t -> check_operand_prec t prec);
  (* A write that changes a precision counts itself, through [Tn.prec_write_hook]. *)
  List.iter ts ~f:(fun t -> Tn.update_prec t.value prec)

let update_operand_prec t prec = update_operands_prec [ t ] prec

let () =
  Tn.session_gate := Shape.check_untainted;
  (* Every write that changes a node's delayed precision is a session mutation (gh-ocannl-1624):
     top-down links, operand precisions, and a precision set inside a user's construction alike. *)
  Tn.prec_write_hook := Row.note_mutation;
  (* Installed only once something is discarded, so a session that never rewinds pays nothing. *)
  Tn.validity_check := None

let lazy_to_dims shape = lazy (Shape.to_dims shape)

let fetch_zeros array shape =
  Asgns.Fetch { array; fetch_op = Constant 0.; dims = lazy_to_dims shape }

let max_sublabel_length = ref 25
[@@mutation_serial.exempt "user configuration of label printing, not inference state"]

(** Creates the shape update step of an operation, complete, and registers it with shape inference
    (propagating shapes eagerly). A step carries the neutral element of its operation's accumulation
    from birth: {!Shape.derive_projections} reads it, so there is no window in which a derivation
    could observe a step without it. {!raw_accum} knows its accumulation up-front; {!op} reads the
    neutral element off the assignments [op_asn] built, and creates the step afterwards. *)
let register_update_step ~neutral_elem ~shape ~shape_logic : Shape.update_step =
  let update_step = Shape.make_update_step ~shape ~logic:shape_logic ~neutral_elem in
  Shape.propagate_shapes update_step;
  update_step

(** The lazy projections handle of an operation, over [get_step], which yields its update step. The
    step need not exist yet: only the derived projections need it, and they are forced during
    lowering, while the product-space proxy shape is a function of the operation's shape and logic
    alone. This is what lets {!op} hand the handle to [op_asn] before the step exists. *)
let make_projections ~shape ~shape_logic ~(get_step : unit -> Shape.update_step) : projections =
  {
    projections_debug = Shape.logic_to_spec shape_logic;
    projections = lazy (Shape.get_projections (get_step ()));
    product_shape = lazy (Shape.product_space_shape ~shape ~logic:shape_logic);
  }

(** The node an operand triple [(t, is_grad, is_merge)] denotes, as an assignment buffer. *)
let buffer_of ~is_grad ~is_merge (t : t) : Asgns.buffer =
  let tn = if is_grad then (Option.value_exn ~here:[%here] t.diff).grad else t.value in
  if is_merge then Asgns.Merge_buffer tn else Asgns.Node tn

let raw_accum ~initialize_neutral ~accum ~(t : t) ~(lhs_is_grad : bool) ~shape_logic
    ~(rhs : Asgns.accum_rhs) : Asgns.t =
  check_not_discarded ~what:"%cd assignment" t;
  (* Everything that can refuse comes before the registration; the rest of a [%cd] assignment's
     construction is a taint boundary like [Tensor.op]'s (gh-ocannl-1301). *)
  let lhs = if lhs_is_grad then (Option.value_exn ~here:[%here] t.diff).grad else t.value in
  Shape.with_taint_on_failure ~describe:(fun exn ->
      [%string "%cd assignment to %{Tn.label_name lhs}: %{Shape.exn_summary exn}"])
  @@ fun () ->
  let update_step =
    register_update_step
      ~neutral_elem:(Some (Ir.Ops.neutral_elem accum))
      ~shape:t.shape ~shape_logic
  in
  let projections =
    make_projections ~shape:t.shape ~shape_logic ~get_step:(fun () -> update_step)
  in
  Asgns.Accum_op
    {
      initialize_neutral;
      accum;
      lhs;
      rhs;
      projections = projections.projections;
      projections_debug = projections.projections_debug;
    }

(** Kahn-style topological layering of code fragments (gh-461). Each item comes tagged with the
    nodes it [needs] from its siblings and the nodes it [provides] to them; an item stays blocked
    while any still-unemitted item provides something it needs, and the ready items are emitted as a
    layer preserving their relative order. [on_cycle] is called with the remaining items when a
    layer comes out empty; it is expected to raise, since a fragment-level dependency cycle would
    need statement-level interleaving to schedule correctly, which is not supported: any
    whole-fragment order would read some values before they are computed. *)
let topo_layers (type a) (tagged : (a * tn_set * tn_set) list)
    ~(on_cycle : (a * tn_set * tn_set) list -> a list) : a list =
  let rec order acc remaining =
    match remaining with
    | [] -> List.concat @@ List.rev acc
    | _ ->
        let blocked_by (_, needs, _) =
          List.exists remaining ~f:(fun (_, _, provides) ->
              not (Set.is_empty (Set.inter needs provides)))
        in
        let ready, blocked = List.partition_tf remaining ~f:(Fn.non blocked_by) in
        if List.is_empty ready then on_cycle remaining
        else order (List.map ready ~f:(fun (item, _, _) -> item) :: acc) blocked
  in
  order [] tagged

type grad_spec = Require_grad | Prohibit_grad | If_needed [@@deriving sexp, equal, variants]

(* Registers [values] as [tn]'s host initialization data (gh-ocannl-633). The buffer is shaped only
   when forced, i.e. after shape inference: interior cells are filled row-major cycling over
   [values] ([Constant_fill]'s in-kernel semantics; the [Total_elems] shape constraint pins the
   element count to the values length, so cycling matters only for the 1-element [Constant] case,
   which broadcasts, including [Broadcast] host-init data), and padding regions are filled with the
   committed padding value, matching [reset_padding_regions]. Both cases write the interior
   directly, so neither allocates anything the size of the buffer (gh-ocannl-1218,
   gh-ocannl-1258). *)
let register_constant_host_init tn values =
  Ir.Host_inits.register tn
    (lazy
      (let prec = Lazy.force tn.Tn.storage_prec in
       let dims = Lazy.force tn.Tn.dims in
       let padding = Tn.get_padding tn in
       let debug = "Host init for " ^ Tn.label_name tn in
       let nd = Nd.create_array ~debug prec ~dims ~padding in
       let pads = Option.map padding ~f:fst in
       (match values with
       | [| value |] -> Nd.fill_from_float ?padding:pads nd value
       | _ -> Nd.set_flat_values ?padding:pads ~cycle:true nd values);
       nd))

let debug_gc_backprop_roots =
  lazy (Utils.get_global_flag ~default:false ~arg_name:"debug_gc_backprop_roots")

(* [terminal_backprop] (gh-ocannl-1724): the tensor's backprop is terminal by construction -- set by
   {!term}, whose [grad_asn] is the constant [empty_comp] -- so it owns no backprop fragment, and a
   later consumer finding it no longer a backprop root generates no requirement for it. Parameters
   are built by [term] too (and [force_param_diff] gives them an empty backprop besides). Decided by
   the maintainer ("exempt all terms", ahrefs/ocannl#1724): there is no code to propagate. *)
let%track7_sexp op ~(label : string list) ?(ternary_op = Shape.Pointwise_tern)
    ?(compose_op = Shape.Pointwise_bin) ?(transpose_op = Shape.Pointwise_un) ?terminal_op
    ?shape_logic ?(terminal_backprop = false) ~op_asn ~grad_asn ?(grad_spec = If_needed)
    ?(top_down_prec = false) make_shape (orig_ts : t list) : t =
  List.iter orig_ts ~f:(fun t ->
      if t.value.id >= session_state.next_id then (
        (* A trace's tensors have ids from a reserved range above every session id (gh-ocannl-1751):
           a handle that outlived its trace is reported as the discarded node it is. *)
        check_not_discarded ~what:"Tensor.op: operand" t;
        raise
        @@ Session_error
             ( [%string
                 "Tensor #%{t.value.id#Int} %{Tn.debug_name t.value} has an id greater than the \
                  last id #%{session_state.next_id - 1#Int} -- check your uses of \
                  Tensor.unsafe_reinitialize, if all your uses are valid, report this as a bug."],
               Some t )));
  Shape.check_untainted ();
  List.iter orig_ts ~f:(check_not_discarded ~what:"Tensor.op: operand");
  let entry_id = session_state.next_id in
  Shape.with_taint_on_failure ~describe:(fun exn ->
      let msg = match exn with Session_error (msg, _) -> msg | exn -> Shape.exn_summary exn in
      [%string "construction of #%{entry_id#Int} %{String.concat ~sep:\" \" label}: %{msg}"])
  @@ fun () ->
  (* The code needs to be included in the order it was computed due to potential non-tree DAGs. *)
  let ordered_ts =
    List.dedup_and_sort orig_ts ~compare:(fun t1 t2 -> Int.ascending t1.value.id t2.value.id)
  in
  (* Id-ascending order does not suffice to avoid computed-after-use bugs across sibling forward
     fragments: the code of a shared subtensor is embedded in the forward of its first-constructed
     consumer, while siblings are constructed in the application order of the current operation
     (right-to-left for OCaml arguments), so an earlier-id sibling can read a node whose defining
     code is embedded under a later-id sibling (gh-461). Re-order the forward-root children
     topologically: a fragment embedding a node must precede fragments that read it. Non-root
     children keep their positions -- their forward code is owned elsewhere, so their (stale)
     embedded_nodes must not contribute dependency edges. The backprop code below includes the
     backprop fragments via [List.rev ordered_ts], so it inherits the mirrored owner-last order (for
     children that are backprop roots but not forward roots, this still degenerates to the id order,
     see gh-461). *)
  let ordered_ts =
    match List.filter ordered_ts ~f:is_fwd_root with
    | [] | [ _ ] -> ordered_ts
    | roots ->
        let tagged =
          List.map roots ~f:(fun ti ->
              (* A node only constrains the order if the owning fragment actually computes it:
                 data-backed terminals are uploaded at link time (Host_inits) and have no forward
                 code, so embedding them must not create edges (e.g. an input shared between two
                 RoPE applications is not a scheduling constraint). *)
              let defines =
                Set.inter ti.forward.embedded_nodes (Asgns.collect_written ti.forward.asgns)
              in
              let requires =
                Set.diff
                  (fst @@ Asgns.collect_nodes_guess_output ti.forward.asgns)
                  ti.forward.embedded_nodes
              in
              (ti, requires, defines))
        in
        let on_cycle remaining =
          (* Restructure the program so that shared subexpressions are first consumed in a
             consistent order across the conflicting operands. *)
          List.iter remaining ~f:(fun (ti, requires, _) ->
              let blockers =
                List.concat_map remaining ~f:(fun (other, _, other_defines) ->
                    Set.inter requires other_defines |> Set.to_list
                    |> List.map ~f:(fun tn ->
                        ([%string "%{Tn.debug_name tn} (in #%{other.value.id#Int})"]
                        [@message_only "the fragment-cycle diagnostic printed below"])))
              in
              Stdlib.Printf.eprintf "  fragment #%d %s waits for: %s\n%!" ti.value.id
                (Tn.debug_name ti.value)
                (String.concat ~sep:", " blockers));
          raise
          @@ Session_error
               ( [%string
                   "Tensor.op: forward-code fragments of the operands of tensor \
                    #%{session_state.next_id#Int} have a dependency cycle: each fragment reads a \
                    node whose defining code is embedded in another"],
                 Some ((fun (ti, _, _) -> ti) @@ List.hd_exn remaining) )
        in
        let sorted_roots = topo_layers tagged ~on_cycle in
        (* Splice the re-ordered roots back into the root positions, keeping non-roots put. *)
        let rec splice roots_left ts =
          match (ts, roots_left) with
          | [], [] -> []
          | [], _ :: _ -> assert false
          | ti :: rest, _ when not (is_fwd_root ti) -> ti :: splice roots_left rest
          | _ :: rest, r :: roots_left -> r :: splice roots_left rest
          | _ :: _, [] -> assert false
        in
        splice sorted_roots ordered_ts
  in
  let id : int = session_state.next_id in
  session_state.next_id <- session_state.next_id + 1;
  let _session_state_next_id : int = session_state.next_id in
  let shape = make_shape ~debug_name:(Tn.get_debug_name ~id ~label ()) ~id in
  (* Split subtensors by whether they use top-down precision inference *)
  let top_down_ts = List.filter ordered_ts ~f:(fun t -> t.top_down_prec) in
  let delayed_prec_for default get =
    if top_down_prec then
      (* For top-down precision, don't promote from inputs *)
      Tn.Default default
    else
      (* For bottom-up precision, only promote from non-top-down subtensors *)
      let lazy_v_precs =
        List.filter_map ordered_ts ~f:(fun ti ->
            Option.map (get ti) ~f:(fun v ->
                if ti.top_down_prec then lazy (Tn.get_specified_prec v)
                else lazy (Some (Tn.bottom_up_prec v))))
      in
      Tn.Inferred
        (lazy
          (List.filter_map lazy_v_precs ~f:Lazy.force
          |> List.reduce ~f:Ir.Ops.promote_prec
          |> Option.value ~default))
  in
  let delayed_prec = delayed_prec_for !default_value_prec (fun t -> Some t.value) in
  let terminal_logic () =
    let open Shape in
    (* Note: parameters will get their terminal logic set via Shape.set_terminal. *)
    let is_param = false in
    match terminal_op with
    | None -> Terminal { is_param; logic = Fetch (Asgns.Constant 0.0) }
    | Some (Fetch fetch_op) -> Terminal { is_param; logic = Fetch fetch_op }
    | Some (Data init_data) -> Terminal { is_param; logic = Data init_data }
  in
  let unpadded_dims = lazy_to_dims shape in
  let padding = lazy (Shape.to_padding shape) in
  let v =
    (* For ndarray-backed literals the init buffer is no longer stored on the tensor node
       (gh-ocannl-333); we register it in [Host_inits] so each context uploads it at link time. *)
    match terminal_op with
    | Some (Shape.Data (Asgns.Broadcast value)) ->
        let tn = Tn.create delayed_prec ~id ~label ~unpadded_dims ~padding () in
        Tn.update_memory_mode tn On_device (Site "49:ndarray-backed");
        register_constant_host_init tn [| value |];
        tn
    | Some (Shape.Data (Asgns.Reshape data)) ->
        let tn, init =
          Tn.create_with_reshape ~id ~label ~unpadded_dims ~padding ~from_padded:false
            ~base_ndarray:data ()
        in
        Ir.Host_inits.register tn init;
        tn
    | Some (Shape.Data (Asgns.Keep_shape_no_padding data)) ->
        let tn, init = Tn.create_from_padded ~id ~label ~ndarray:data ~padding:None () in
        Ir.Host_inits.register tn init;
        tn
    | Some (Shape.Data (Asgns.Padded { data; padding = padding_spec; padded_value })) ->
        let padding = Some (padding_spec, padded_value) in
        let tn, init = Tn.create_from_padded ~id ~label ~ndarray:data ~padding () in
        Ir.Host_inits.register tn init;
        tn
    | Some (Shape.Fetch _) | None -> Tn.create delayed_prec ~id ~label ~unpadded_dims ~padding ()
  in
  note_open_node v;
  let update_infer_prec ti tn prec =
    (* A closed operand takes no new links (gh-ocannl-1301): its committed state must not depend on
       a consumer a rewind can discard. Layout readiness alone says nothing about precision
       (gh-ocannl-2026); the precision cell itself refuses changing a forced, non-Specified
       precision and counts writes via prec_write_hook. *)
    if not ti.shape.Shape.closed then Tn.update_infer_prec tn prec
  in
  let transpose_op =
    match transpose_op with
    | Uint4x32_to_prec _ -> Shape.Uint4x32_to_prec v.Tn.storage_prec
    | _ -> transpose_op
  in
  let shape_logic =
    match shape_logic with
    | Some logic -> logic
    | None -> (
        match orig_ts with
        | [] -> terminal_logic ()
        | [ t1 ] -> Shape.Transpose (transpose_op, t1.shape)
        | [ t1; t2 ] -> Shape.Broadcast (compose_op, t1.shape, t2.shape)
        | [ t1; t2; t3 ] -> Shape.Broadcast_tern (ternary_op, t1.shape, t2.shape, t3.shape)
        | _ ->
            (* Let's implement what we need when we need it. *)
            assert false)
  in
  let embedded_nodes = ref @@ Set.singleton (module Tn) v in
  let children =
    List.folding_map orig_ts
      ~init:(Set.empty (module Int))
      ~f:(fun used ti ->
        let embedded = is_fwd_root ti && not (Set.mem used ti.value.id) in
        if embedded then
          embedded_nodes := Set.add (Set.union !embedded_nodes ti.forward.embedded_nodes) ti.value;
        (Set.add used ti.value.id, { subtensor = ti; embedded }))
  in
  let params = Set.union_list (module T) @@ List.map ordered_ts ~f:(fun ti -> ti.params) in
  let t =
    {
      params;
      forward = Asgns.empty_comp;
      diff = None;
      value = v;
      top_down_prec;
      shape;
      children;
      consumption = { fwd = Not_handed_out; bprop = Not_handed_out; fwd_owner = fresh_owner_cell v };
    }
  in
  (* The operation's shape update step is created only once [op_asn] has returned: the step carries
     the neutral element of the operation's accumulation, which is read off the assignments [op_asn]
     builds, and {!Shape.derive_projections} consumes it (the clamped-window decision and the
     margins' neutral). Creating the step complete, rather than creating it first and filling the
     neutral element in afterwards, means no derivation can ever run against a step whose neutral
     element is not yet known -- there is no such step. The [projections] handle [op_asn] receives
     is a promise for the step: it is only redeemed during lowering.

     Forcing the handle from within [op_asn] is therefore rejected: this operation's step is not
     registered and nothing is derived for it. So is finalizing inference by any other route
     ([Shape.to_dims]/[to_padding]): [op_asn] runs under {!Shape.with_construction_window}, and
     [Shape.finish_inference] refuses while it is open, since this operation's constraints are not
     registered yet and its operands would close without them. The rejection is a bug report, not a
     recovery API. Shape inference is session-global and not transactional -- an [op_asn] may have
     emitted constraints, set delayed references or propagated shapes before forcing, and any
     construction that fails mid-way (an ordinary shape error at registration included) leaves the
     solver partly updated -- so the error means "fix the [op_asn]", and the session is reset with
     {!unsafe_reinitialize} rather than continued; the transaction boundary that would make a
     rejected construction reusable is gh-ocannl-903, not this seam. There is no supported reason to
     force it there: forcing the projections runs {!Shape.finish_inference}, which solves and
     derives for EVERY active update step in the session, mid-construction of this tensor; no
     [op_asn] in this repository does it, and a caller that needs shape information has
     [Shape.set_dim] and the einsum capture syntax instead. The product-space proxy shape, which
     [%cd] [*_pspace] intermediates force, is a function of the shape and logic alone and stays
     available. Captured dimension and row references are bound at registration, so relating them
     ([Shape.set_equal]/[set_scale]) belongs at the operation-expression level, where the [%op]
     capture syntax puts it, not in an [op_asn]. *)
  let update_step : Shape.update_step option ref = ref None in
  let forced_early = ref false in
  let get_step () =
    match !update_step with
    | Some update_step -> update_step
    | None ->
        forced_early := true;
        raise
        @@ Session_error
             ( [%string
                 "Tensor.op: the projections of tensor #%{id#Int} were forced from within \
                  [op_asn], before the operation's assignments (and so its neutral element) exist \
                  -- not supported, as it would finalize shape inference for the whole session \
                  mid-construction. The operation is not registered, but whatever else the \
                  [op_asn] did before forcing stays in the session: fix the [op_asn] and rewind \
                  the session (Tensor.rewind_to_last_closure) rather than reusing the operands. If \
                  the [op_asn] is OCANNL's own, report this as a bug."],
               Some t )
  in
  let projections = make_projections ~shape ~shape_logic ~get_step in
  let fwd_root_ts = List.filter ordered_ts ~f:is_fwd_root in
  let fwds = List.map fwd_root_ts ~f:(fun ti -> ti.forward) in
  (* gh-ocannl-1746: [fwds] already carries the root operands' forward code, so they are hidden from
     the root map while [op_asn] runs: a [%cd] [op_asn] naming an operand tensor (as
     [Operation.interleave] does) would otherwise embed its code a second time -- and mark it taken.
     Restored on the way out, so a failing [op_asn] leaves the roots as they were; they are removed
     for good below, once the operation is registered. *)
  let this_op_asn =
    let hidden =
      List.filter_map fwd_root_ts ~f:(fun ti ->
          Option.map (Map.find session_state.forward_roots ti.value.id) ~f:(fun e ->
              (ti.value.id, e)))
    in
    session_state.forward_roots <-
      List.fold hidden ~init:session_state.forward_roots ~f:(fun roots (id, _) ->
          Map.remove roots id);
    Exn.protect
      ~f:(fun () -> Shape.with_construction_window (fun () -> op_asn ~t ~projections))
      ~finally:(fun () ->
        session_state.forward_roots <-
          List.fold hidden ~init:session_state.forward_roots ~f:(fun roots (id, e) ->
              Map.set roots ~key:id ~data:e))
  in
  (* An [op_asn] that forced the handle and swallowed the rejection has built its assignments over a
     poisoned thunk (a forced lazy re-raises); the operation is unusable either way. *)
  if !forced_early then
    raise
    @@ Session_error
         ( [%string
             "Tensor.op: the [op_asn] of tensor #%{id#Int} forced its projections and caught the \
              rejection; the operation cannot be built. Fix the [op_asn] and rewind the session \
              (Tensor.rewind_to_last_closure) rather than reusing the operands."],
           Some t );
  let forward = Asgns.sequence @@ fwds @ [ this_op_asn ] in
  (* Extract the neutral element from THIS operation's assignments only, not the whole forward
     chain. The dependencies may have different accumulation operations which would cause false
     conflicts. *)
  let neutral_elem = Asgns.collect_neutral_elem this_op_asn.asgns in
  update_step := Some (register_update_step ~neutral_elem ~shape ~shape_logic);
  (* Apply delayed top-down precision updates to parameter subtensors. Deferred until the operation
     is registered (gh-ocannl-1301): a construction that fails before that leaves no link on its
     operands, so it does not taint the session. *)
  List.iter top_down_ts ~f:(fun ti -> update_infer_prec ti ti.value v.Tn.storage_prec);
  (* The forward fragment of [t] (gh-ocannl-1724): [(v, Fwd)] plus, through [Asgns.sequence], the
     fragments of the forward-root children embedded above. *)
  let forward =
    {
      forward with
      embedded_nodes = Set.union forward.embedded_nodes !embedded_nodes;
      owned = Set.add forward.owned (v, Asgns.Fwd);
    }
  in
  List.iter ordered_ts ~f:(fun ti -> remove_fwd_root ti);
  (* The owner cells (gh-ocannl-1796): [t]'s code computes [v], embeds the root children's code, and
     whatever its own [op_asn] embedded ([%cd]'s takes and [.forward] replays, inline tensors). One
     owner edge per embedded cell, so the relation stays exact when code is replayed into several
     roots. *)
  let cell = t.consumption.fwd_owner and seen = ref None in
  register_owned ~seen ~cell v;
  List.iter fwd_root_ts ~f:(fun ti -> add_owner ~seen ti.consumption.fwd_owner ~owner:cell);
  Set.iter this_op_asn.embedded_nodes ~f:(register_owned ~seen ~cell);
  let t = { t with forward } in
  if
    is_prohibit_grad grad_spec
    || Fn.non is_require_grad grad_spec
       && List.for_all orig_ts ~f:(fun ti -> Option.is_none ti.diff)
  then (
    (* Installing the result as a root is a session mutation (gh-ocannl-1301), counted by
       [add_fwd_root]: nothing after it can fail. *)
    add_fwd_root t;
    t)
  else
    let get ti = Option.map ti.diff ~f:(fun d -> d.grad) in
    let delayed_prec = delayed_prec_for !default_grad_prec get in
    let grad_id = session_state.next_id in
    session_state.next_id <- session_state.next_id + 1;
    let g =
      Tn.create delayed_prec ~id:grad_id ~label:("grad" :: label) ~unpadded_dims
        ~padding:(lazy (Shape.to_padding shape))
        ()
    in
    (* A gradient is an accumulator: zero is its correct entry value, so a separately compiled
       routine may read it before any backprop wrote it (gh-ocannl-1462). *)
    Tn.mark_zero_valid g;
    (* A differentiable term's gradient (a [Require_grad] leaf: an optimized input, a saliency
       target) has no reader inside its routine -- no backprop code reads a terminal gradient -- so
       its only readers are the host and later routines (an optimizer step). Declare the observation
       intent [param] declares for parameter gradients, which keeps it out of the unobservable
       [Local] class and the liveness planner's aliasing while leaving [Virtual] recomputation
       available (gh-ocannl-1743). The leaf's value carries the intent too: a parameter's value is
       [On_device] by its initialization, while a term's value is placed like any forward result and
       could settle [Local] -- and a [Virtual] gradient recomputes from it, so a [Local] value would
       leave the gradient unobservable after all. *)
    if terminal_backprop then (
      Tn.set_observable g;
      Tn.set_observable t.value);
    note_open_node g;
    (* Apply delayed top-down precision updates to parameter gradient subtensors *)
    List.iter top_down_ts ~f:(fun ti ->
        Option.iter ti.diff ~f:(fun d -> update_infer_prec ti d.grad g.Tn.storage_prec));
    (* Which operands are backprop roots is settled once, here (gh-ocannl-1804): whether a dropped
       consumer gives a leaf back moves with the GC, and with the nested ops [grad_asn] builds, so a
       second read of the map could embed a gradient whose zeroing the first read skipped. The scan
       runs only for an operand it can change. *)
    if List.exists ordered_ts ~f:(fun ti -> Weak_root.is_leaf ti && not (in_bprop_roots ti)) then (
      (* The hand-back is the read that moves with the GC (gh-ocannl-2062). Stress only this branch:
         the operand entries checked above are live, and only pruning dead consumers can restore a
         missing leaf. *)
      if Lazy.force debug_gc_backprop_roots then Stdlib.Gc.full_major ();
      hand_back_reclaimed ());
    let bck_roots =
      List.filter_map ordered_ts ~f:(fun ti ->
          Option.map (Weak_root.find session_state.backprop_roots ti.value.id) ~f:(fun e ->
              (ti.value.id, e)))
    in
    let is_bck_root ti = List.Assoc.mem bck_roots ~equal:Int.equal ti.value.id in
    let zero_grads =
      let zero_g ti =
        Option.value_map ti.diff ~default:Asgns.empty_comp ~f:(fun diff -> diff.zero_grads)
      in
      let zeros =
        List.map ordered_ts ~f:(fun ti -> if is_bck_root ti then zero_g ti else Asgns.empty_comp)
      in
      (* The zeroed gradients are embedded: a routine built out of this code alone (as
         {!Train.zero_params_grads} does) must be able to allocate them rather than require them of
         a prior context. *)
      Asgns.sequence @@ zeros
      @ [
          Asgns.comp
            ~embedded:(Set.singleton (module Tn) g)
            ~zeroes:(Set.singleton (module Tn) g)
            (fetch_zeros g shape);
        ]
    in
    let embedded_nodes = ref @@ Set.singleton (module Tn) g in
    (* The code needs to be included in the reverse order to which it was computed. This guarantees
       that all ancestors of a node are backpropagated before the node is backpropagated, even for
       non-tree DAGs. For repeating subtensors, the first one to-be-included should actually be
       included! That's why we reverse the order up-front prior to processing. *)
    let ordered_ts = List.rev ordered_ts in
    (* A backprop that would add nothing to the sequence but a [Noop]: no code, nothing embedded,
       owned, required, zeroed or accumulated. That is exactly a parameter's ([force_param_diff]'s
       [empty_comp]); a term built by [op] embeds its own gradient, so it never is. Leaving it out
       keeps a parameter operand from adding a [Noop] leaf to the assembled backprop
       (gh-ocannl-1797: the parameter test this replaces had no other effect). *)
    let inert (c : Asgns.comp) =
      (match c.asgns with Asgns.Noop -> true | _ -> false)
      && Set.is_empty c.embedded_nodes && Set.is_empty c.owned && Set.is_empty c.requires
      && Set.is_empty c.zeroes && Set.is_empty c.accumulates && Map.is_empty c.accumulation_runs
      && Set.is_empty c.persists
    in
    let bprop ti =
      Option.bind ti.diff ~f:(fun diff ->
          (* The gradient is embedded even when the child's backprop is terminal. *)
          embedded_nodes :=
            Set.add (Set.union !embedded_nodes diff.backprop.embedded_nodes) diff.grad;
          Option.some_if (not (inert diff.backprop)) diff.backprop)
    in
    (* The backprop fragments this one's gradient writes flow into without carrying them
       (gh-ocannl-1724): a differentiable non-parameter child that is no longer a backprop root had
       its backprop embedded by an earlier consumer, or handed out, so some other routine of the
       lineage must propagate from its gradient. Derived from the child's own ownership: a child
       whose backprop owns no fragment (a term, parameters included -- [terminal_backprop])
       generates no requirement; nor does one whose gradient this operation's [grad_asn] does not
       write (a [where] predicate, a comparison's operands), filtered below. *)
    let requires = ref Asgns.no_fragments in
    let bcks =
      List.filter_map ordered_ts ~f:(fun ti ->
          if is_bck_root ti then bprop ti
          else (
            (match ti.diff with
            | Some diff when Set.mem diff.backprop.owned (diff.grad, Asgns.Bprop) ->
                requires := Set.add !requires (diff.grad, Asgns.Bprop)
            | _ -> ());
            None))
    in
    (* Mirror of the forward-fragment ordering above (gh-461), with the edge reversed: a fragment
       accumulating into a node's gradient must precede the fragment that embeds the node's backprop
       code. The reversed [ordered_ts] already handles this whenever backprop ownership coincides
       with forward ownership (both belong to the first-constructed consumer); this pass also covers
       children whose forward root was consumed separately (e.g. by a non-differentiable operation),
       for which the forward ordering gives no constraint. *)
    let bcks =
      match bcks with
      | [] | [ _ ] -> bcks
      | _ ->
          let tagged =
            List.map bcks ~f:(fun c ->
                ( c,
                  c.Asgns.embedded_nodes,
                  Set.diff (Asgns.collect_written c.Asgns.asgns) c.Asgns.embedded_nodes ))
          in
          let on_cycle _remaining =
            raise
            @@ Session_error
                 ( [%string
                     "Tensor.op: backprop-code fragments of the operands of tensor #%{id#Int} have \
                      a dependency cycle: each fragment accumulates into a gradient whose backprop \
                      code is embedded in another"],
                   Some t )
          in
          topo_layers tagged ~on_cycle
    in
    let diff = Some { grad = g; zero_grads; backprop = Asgns.empty_comp } in
    let t = { t with diff } in
    (* [grad_asn] runs inside the construction window too: the operands' roots are already removed
       and the result's not yet installed, so a closure here would checkpoint an incomplete graph
       (gh-ocannl-1301). *)
    let own_grad_asn = Shape.with_construction_window (fun () -> grad_asn ~t ~g ~projections) in
    let backprop = Asgns.sequence (own_grad_asn :: bcks) in
    let written = Asgns.collect_written own_grad_asn.asgns in
    let requires =
      Set.filter !requires ~f:(fun ((grad, _) : Asgns.Fragment.t) -> Set.mem written grad)
    in
    (* The operands' gradients this operation's own gradient code writes (gh-ocannl-1744), whether
       or not the operand is still a backprop root: a non-root's zeroing was skipped above together
       with its backprop, so [Context.compile] needs to know who else accumulates into it. The
       embedded children's come through [Asgns.sequence]. *)
    let accumulates =
      List.filter_map ordered_ts ~f:(fun ti ->
          Option.bind ti.diff ~f:(fun diff -> Option.some_if (Set.mem written diff.grad) diff.grad))
      |> Set.of_list (module Tn)
    in
    (* The backprop fragment of [t]: [(g, Bprop)] plus the embedded children's fragments. *)
    let backprop =
      {
        backprop with
        embedded_nodes = Set.union backprop.embedded_nodes !embedded_nodes;
        owned =
          (if terminal_backprop then backprop.owned else Set.add backprop.owned (g, Asgns.Bprop));
        requires = Set.union backprop.requires requires;
        accumulates = Set.union backprop.accumulates accumulates;
      }
    in
    (* As in the forward-only branch: the root changes below count, and nothing after them fails. *)
    Row.note_mutation ();
    List.iter bck_roots ~f:(fun (ti_id, _) ->
        session_state.backprop_roots <- Map.remove session_state.backprop_roots ti_id);
    let diff = Some { grad = g; zero_grads; backprop } in
    let t = { t with diff } in
    add_fwd_root t;
    session_state.backprop_roots <-
      Map.add_exn session_state.backprop_roots ~key:id
        ~data:(Weak_root.make ~absorbed:(List.map bck_roots ~f:snd) t);
    t

type param_op_fun =
  ?input_dims:int list ->
  ?output_dims:int list ->
  ?input_axes:(string * int) list ->
  ?output_axes:(string * int) list ->
  ?deduced:Shape.deduce_within_shape ->
  unit ->
  t

type op_fun =
  ?label:string list ->
  ?top_down_prec:bool ->
  ?batch_dims:int list ->
  ?batch_axes:(string * int) list ->
  param_op_fun

let%track7_sexp binop ?op_label ?compose_op ~op_asn ~grad_asn ?grad_spec t1 t2 ?(label = [])
    ?top_down_prec ?batch_dims ?batch_axes ?input_dims ?output_dims ?input_axes ?output_axes
    ?deduced () : t =
  let op_asn ~t ~projections = op_asn ~t ~t1 ~t2 ~projections in
  let grad_asn ~t ~g ~projections = grad_asn ~t ~g ~t1 ~t2 ~projections in
  op
    ~label:(Option.to_list op_label @ label)
    ?compose_op ?transpose_op:None ~op_asn ~grad_asn ?grad_spec ?top_down_prec
    (Shape.make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes ?deduced
       ())
    [ t1; t2 ]

let%track7_sexp ternop ?op_label ?ternary_op ~op_asn ~grad_asn ?grad_spec t1 t2 t3 ?(label = [])
    ?top_down_prec ?batch_dims ?batch_axes ?input_dims ?output_dims ?input_axes ?output_axes
    ?deduced () : t =
  let op_asn ~t ~projections = op_asn ~t ~t1 ~t2 ~t3 ~projections in
  let grad_asn ~t ~g ~projections = grad_asn ~t ~g ~t1 ~t2 ~t3 ~projections in
  op
    ~label:(Option.to_list op_label @ label)
    ?ternary_op ?compose_op:None ~op_asn ~grad_asn ?grad_spec ?top_down_prec
    (Shape.make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes ?deduced
       ())
    [ t1; t2; t3 ]

let%track7_sexp blockop ?op_label ~spec ~delayed_vars ~op_asn ~grad_asn ?grad_spec rhses
    ?(label = []) ?top_down_prec ?batch_dims ?batch_axes ?input_dims ?output_dims ?input_axes
    ?output_axes ?deduced () : t =
  let rhses_list = Array.to_list rhses in
  let rhs_shapes = List.map rhses_list ~f:(fun rhs -> rhs.shape) in
  let shape_logic = Shape.Block { spec; delayed_vars; rhses = rhs_shapes } in
  op
    ~label:(Option.to_list op_label @ label)
    ~shape_logic ~op_asn ~grad_asn ?grad_spec ?top_down_prec
    (Shape.make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes ?deduced
       ())
    rhses_list

let%track7_sexp unop ?op_label ?transpose_op ~op_asn ~grad_asn ?grad_spec t1 ?(label = [])
    ?top_down_prec ?batch_dims ?batch_axes ?input_dims ?output_dims ?input_axes ?output_axes
    ?deduced () : t =
  let op_asn ~t ~projections = op_asn ~t ~t1 ~projections in
  let grad_asn ~t ~g ~projections = grad_asn ~t ~g ~t1 ~projections in
  op
    ~label:(Option.to_list op_label @ label)
    ?compose_op:None ?transpose_op ~op_asn ~grad_asn ?grad_spec ?top_down_prec
    (Shape.make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes ?deduced
       ())
    [ t1 ]

let%track7_sexp term ?init_data ?fetch_op ?grad_spec ?(label = []) ?(top_down_prec = true)
    ?batch_dims ?batch_axes ?input_dims ?output_dims ?input_axes ?output_axes ?deduced () : t =
  let terminal_op =
    match (init_data, fetch_op) with
    | Some _, Some _ -> invalid_arg "Tensor.term: both init_data and fetch_op are provided"
    | Some init_data, None -> Some (Shape.Data init_data)
    | None, Some fetch_op -> Some (Shape.Fetch fetch_op)
    | None, None -> None
  in
  let op_asn ~t ~projections =
    let open Asgns in
    let dims = lazy (Lazy.force projections.projections).Idx.lhs_dims in
    match fetch_op with
    | None -> Asgns.empty_comp
    | Some
        (( Constant _ | Constant_bits _ | Slice _ | Embed_symbol _ | Embed_dim _ | Embed_self_id
         | Range_over_offsets | Constant_fill _ ) as fetch_op) ->
        Asgns.to_comp @@ Fetch { array = t.value; fetch_op; dims }
  in
  let grad_asn ~t:_ ~g:_ ~projections:_ = Asgns.empty_comp in
  let make_shape =
    Shape.make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes ?deduced ()
  in
  let grad_spec = Option.value grad_spec ~default:If_needed in
  (* Note: terminal_op is used for both tensor creation and shape inference. *)
  op ~label ?compose_op:None ?transpose_op:None ?terminal_op ~terminal_backprop:true ~op_asn
    ~grad_asn ~grad_spec ~top_down_prec make_shape []

let float_to_label v = Float.to_string v |> String.chop_suffix_if_exists ~suffix:"."

let%track7_sexp number ?(label = []) ?axis_basis ?(grad_spec = Prohibit_grad) c : t =
  (* Note: broadcastable scalar basis ([bcast_if_1], the broadcast top) by default, so a scalar
     constant broadcasts into a parameter of any shape rather than fusing as a fixed atom. *)
  let label = float_to_label c :: label in
  let fetch_op = Ir.Assignments.Constant c in
  let t = term ~label ~grad_spec ~batch_dims:[] ~input_dims:[] ~fetch_op in
  let t =
    match axis_basis with
    | None -> t ~output_axes:[ (Row.bcast_if_1, 1) ] ()
    | Some axis_basis -> t ~output_axes:[ (axis_basis, 1) ] ()
  in
  Tn.update_memory_mode t.value Effectively_constant (Site "24:literal-constant");
  (* FIXME: make this always pick a matching precision. *)
  Ir.Ops.(
    if exceeds_fp16_cutoff c then Tn.update_infer_prec ~only_if:is_up_to_fp16 t.value (lazy single));
  t

let%track7_sexp bits ?(label = []) ?axis_basis ?(grad_spec = Prohibit_grad) i : t =
  (* Use Constant_bits for exact bit representation, primarily for uint4x32 *)
  let label = Int64.to_string i :: label in
  let fetch_op = Ir.Assignments.Constant_bits i in
  let t = term ~label ~grad_spec ~batch_dims:[] ~input_dims:[] ~fetch_op in
  let t =
    match axis_basis with
    | None -> t ~output_axes:[ (Row.bcast_if_1, 1) ] ()
    | Some axis_basis -> t ~output_axes:[ (axis_basis, 1) ] ()
  in
  Tn.update_memory_mode t.value Effectively_constant (Site "24:literal-constant");
  t

let constant_fill ~debug values =
  let n = Array.length values in
  if n = 0 then (None, None)
  else
    let limit =
      Int.of_string @@ Utils.get_global_arg ~default:"16" ~arg_name:"limit_constant_fill_size"
    in
    if n <= limit then
      if n = 1 then (None, Some (Asgns.Constant values.(0)))
      else (None, Some (Asgns.Constant_fill values))
    else if n = 1 then (Some (Asgns.Broadcast values.(0)), None)
    else
      let nd = Nd.create_array ~debug ~dims:[| n |] ~padding:None !default_value_prec in
      Nd.set_flat_values nd values;
      (Some (Asgns.Reshape nd), None)

let ndarray ?(grad_spec = Prohibit_grad) values ?(label = []) ?top_down_prec ?batch_dims ?batch_axes
    ?input_dims ?output_dims ?input_axes ?output_axes ?deduced () =
  let num_label =
    String.concat ~sep:"," @@ List.map ~f:float_to_label @@ Fn.flip List.take 5
    @@ Array.to_list values
  in
  let num_label = if Array.length values > 5 then num_label ^ "..." else num_label in
  let label = num_label :: label in
  let batch_dims = Option.first_some batch_dims @@ Option.some_if (Option.is_none batch_axes) [] in
  let input_dims = Option.first_some input_dims @@ Option.some_if (Option.is_none input_axes) [] in
  let output_dims =
    Option.first_some output_dims @@ Option.some_if (Option.is_none output_axes) []
  in
  let init_data, fetch_op = constant_fill ~debug:"Tensor.ndarray" values in
  (* Ideally, while the shape is known, the deduced argument will be used for verification. *)
  let t =
    term ?init_data ?fetch_op ~grad_spec ?batch_dims ?batch_axes ~label ?top_down_prec ?input_dims
      ?output_dims ?input_axes ?output_axes ?deduced ()
  in
  Tn.update_memory_mode t.value Effectively_constant (Site "24:literal-constant");
  (* The ndarray-backed path mints the node [On_device] (provenance 49), so the update above cannot
     stick as [Effectively_constant] there; the explicit marker carries the constancy (consumed by
     hoisted operand packing, gh-ocannl-470). *)
  Tn.set_host_constant t.value;
  (* gh-ocannl-633: a below-threshold literal keeps its in-kernel init (the recipe virtualization
     inlines), but its values are also registered as host-init data like the ndarray-backed path
     above the threshold: a lineage that materializes the node drops the in-kernel init and
     self-initializes at link time from this entry ([Low_level.hosted_constant_inits_to_link_time]),
     and hoisted operand packing ([Schedule.hoistable_constant]) can read the values below the size
     cutoff. *)
  (match fetch_op with
  | Some (Asgns.Constant _ | Asgns.Constant_fill _) -> register_constant_host_init t.value values
  | _ -> ());
  let max_abs = Array.fold values ~init:0. ~f:(fun acc v -> Float.(max acc @@ abs v)) in
  Ir.Ops.(
    if exceeds_fp16_cutoff max_abs then
      Tn.update_infer_prec ~only_if:is_up_to_fp16 t.value (lazy single));
  t

let term_init ?grad_spec values ?(label = []) ?top_down_prec ?batch_dims ?batch_axes ?input_dims
    ?output_dims ?input_axes ?output_axes ?deduced () =
  let init_data, fetch_op = constant_fill ~debug:"Tensor.term_init" values in
  term ?init_data ?fetch_op ?grad_spec ?batch_dims ?batch_axes ~label ?top_down_prec ?input_dims
    ?output_dims ?input_axes ?output_axes ?deduced ()

let force_param_diff t =
  Option.iter t.diff ~f:(fun _ -> remove_bprop_root t);
  let grad_id = session_state.next_id in
  session_state.next_id <- session_state.next_id + 1;
  let g =
    Tn.create (Tn.Default !default_grad_prec) ~id:grad_id ~label:("grad" :: t.value.label)
      ~unpadded_dims:(lazy_to_dims t.shape)
      ~padding:(lazy (Shape.to_padding t.shape))
      ()
  in
  Tn.mark_zero_valid g;
  note_open_node g;
  let diff =
    {
      grad = g;
      zero_grads =
        Asgns.comp
          ~embedded:(Set.singleton (module Tn) g)
          ~zeroes:(Set.singleton (module Tn) g)
          (fetch_zeros g t.shape);
      backprop = Asgns.empty_comp;
    }
  in
  let t = { t with diff = Some diff } in
  Row.note_mutation ();
  session_state.backprop_roots <-
    Map.set session_state.backprop_roots ~key:t.value.id ~data:(Weak_root.make t);
  t

let strip_param_diff t =
  Option.iter t.diff ~f:(fun _ -> remove_bprop_root t);
  { t with diff = None }

(** Post-processing hook applied by {!param} to each fully-constructed parameter before it is
    returned to user code. Mixed-precision recipes install a wrapper here that returns a
    reduced-precision "cast twin" consuming the parameter (gh-ocannl-492 master weights): the graph
    then reads the twin while the optimizer keeps updating the master parameter, which remains the
    sole member of the result's {!field:params}. Reset by {!unsafe_reinitialize}. *)
let param_postprocess : (t -> t) ref = ref Fn.id
[@@mutation_serial.exempt "a user-installed hook applied to new parameters, not inference state"]

(* gh-ocannl-544: an initialization expression's job is to fill the parameter's shape, so its
   intermediate results resolve at their use sites — they widen to the parameter instead of closing
   down and broadcasting (which would repeat random values along the broadcast axes). Walks the init
   expression's subgraph, stopping at parameters (their own terminal marking governs them).

   gh-ocannl-1967: for the same reason its float nodes are at least as precise as the parameter,
   whatever the parameter's precision comes from -- a narrower draw would quantize the initial
   values. Bottom-up inference alone stops at the first node none of whose operands states a
   precision, which settles at the session default. The link promotes and does not narrow: a
   parameter narrower than its initializer rounds the wider values once, as it did. A parameter that
   is not top-down reads its initializer's precisions instead, and a settled or closed node takes no
   link (gh-ocannl-1301). The parameter may in turn be inferred from a consumer reading one of these
   nodes, which is why the link is [Tn.follow_prec]. *)
let mark_init_cone (t : t) =
  let visited = Hash_set.create (module Int) in
  let rec loop sub =
    let uid = sub.value.Tn.uid in
    if not (Hash_set.mem visited uid) then (
      Hash_set.add visited uid;
      if not (Set.mem sub.params sub) then (
        let follows =
          t.top_down_prec && not (Lazy.is_val sub.value.Tn.storage_prec || sub.shape.Shape.closed)
        in
        Shape.set_resolve_at_use sub.shape;
        if follows then Tn.follow_prec ~only_if:Ir.Ops.is_float sub.value t.value.Tn.storage_prec;
        List.iter sub.children ~f:(fun { subtensor; _ } -> loop subtensor)))
  in
  List.iter t.children ~f:(fun { subtensor; _ } -> loop subtensor)

let%debug7_sexp param ?(require_grad = true) ~t (name : string) ?(more_label = []) ?input_dims
    ?output_dims ?input_axes ?output_axes ?deduced () : t =
  let t =
    t
      ?label:(Some (name :: more_label))
      ?top_down_prec:(Some true) ?batch_dims:(Some []) ?batch_axes:None ?input_dims ?output_dims
      ?input_axes ?output_axes ?deduced ()
  in
  (* Parameterization writes the value node (memory mode, ownership), the gradient and the shape's
     terminal constraints: a tensor whose inference is closed cannot become a parameter
     (gh-ocannl-1301), since a closed tensor's committed state does not change. *)
  check_not_discarded ~what:("Tensor.param " ^ name) t;
  if t.shape.Shape.closed then
    raise
    @@ Session_error
         ( [%string
             "Tensor.param %{name}: the initializer returned #%{t.value.id#Int} %{Tn.debug_name \
              t.value}, whose shape inference is already closed; a parameter must be a tensor \
              built for it"],
           Some t );
  (* Parameterization's writes, and the post-processing hook, are a taint boundary like
     [Tensor.op]'s (gh-ocannl-1301). *)
  Shape.with_taint_on_failure ~describe:(fun exn ->
      let msg = match exn with Session_error (msg, _) -> msg | exn -> Shape.exn_summary exn in
      [%string "parameterization of %{name}: %{msg}"])
  @@ fun () ->
  Row.note_mutation ();
  (* Parameterization changes code ownership (and replaces the gradient program). Its handout state
     must not alias an initializer returned by a custom [t], which may already be Taken. Keep that
     original initializer replayable while the parameter starts with fresh ownership. *)
  let t =
    {
      t with
      consumption =
        { fwd = Not_handed_out; bprop = Not_handed_out; fwd_owner = fresh_owner_cell t.value };
    }
  in
  let t = if require_grad then force_param_diff t else strip_param_diff t in
  let v = t.value in
  (* Parameters live on device and are materialized; CPU access (init, inspection) is on-demand via
     the context (gh-ocannl-333). *)
  Tn.update_memory_mode v On_device (Site "241:param-value");
  Tn.set_context_owned v;
  (* An ndarray initializer becomes mutable with the parameter: its registered initial values no
     longer justify literal host-write refusal or hoisted packing (gh-ocannl-2021). *)
  Tn.clear_host_constant v;
  (* Never_virtual audit resolution (context-scoped memory modes): parameter gradients carry
     observation intent, not a materialization requirement -- users print and inspect them, and the
     optimizer step's read is an ordinary cross-routine use the lineage can serve (a fused
     forward+backprop+update block may still inline them entirely). [set_observable] only forbids
     resolving them into the [Local] unobservable class; [Virtual] stays available via
     recomputation. A differentiable non-parameter term gets the same intent in [op]
     (gh-ocannl-1743); this one is still needed because [force_param_diff] mints a fresh
     gradient. *)
  (match t.diff with
  | Some diff -> Tn.set_observable diff.grad
  | None -> ());
  Shape.set_terminal ~is_param:(Option.is_some t.diff) t.shape;
  mark_init_cone t;
  remove_fwd_root t;
  !param_postprocess { t with params = Set.singleton (module T) t }

let debug_name t = Tn.debug_name t.value
let debug_grad t = Tn.debug_name (Option.value_exn t.diff).grad

(* Taken code can be handed out repeatedly. Other non-roots still need a reason: a parameter's
   absent forward code, or code owned by a tensor consumer. *)
let not_a_root_reason ~(handout : handout) ~what ~name t =
  match handout with
  | Taken -> assert false (* The consume functions return before reaching this rejection. *)
  | Not_handed_out when String.equal what "forward" && Set.mem t.params t ->
      [%string
        "Tensor.consume_%{what}_code: %{name} is a parameter, which owns no %{what} code of its \
         own (its value is set by Train.init_params or Context.set_values); consume a tensor \
         computed from it instead"]
  | Not_handed_out ->
      [%string
        "Tensor.consume_%{what}_code: %{name} is not a %{what} root: its %{what} code is embedded \
         in a tensor constructed from it (a shared subtensor's code is owned by its \
         first-constructed consumer); consume that consumer instead"]

(* [t]'s non-embedded reads: the forward requirements of its code. *)
let forward_reads t =
  Set.diff (fst @@ Asgns.collect_nodes_guess_output t.forward.asgns) t.forward.embedded_nodes

(* The reads whose code an unconsumed forward root still owns, read off the owner cells
   (gh-ocannl-1796): a root the caller dropped before its code was consumed stays pending, whatever
   the GC has done. *)
let forward_conflicts t = pending_owners ~self:t.consumption.fwd_owner (forward_reads t)

(* The rule the owner cells replaced (gh-ocannl-1796): the live roots whose embedded nodes [t]'s
   code reads without embedding. It sees no root the GC reclaimed, so it can name fewer roots than
   the cells, never more: a root only it names is an owner edge some source of [embedded_nodes]
   failed to add. *)
let forward_conflicts_by_live_roots t =
  let reads = forward_reads t in
  Weak_root.live session_state.forward_roots
  |> List.filter ~f:(fun r ->
      r.value.id <> t.value.id && not (Set.is_empty (Set.inter reads r.forward.embedded_nodes)))

let debug_check_forward_root_owners =
  lazy (Utils.get_global_flag ~default:false ~arg_name:"debug_check_forward_root_owners")

(* Fails rather than refuses, ahead of the refusal the verdict decides: a missed root is a bug here,
   not a session the caller can fix, and no handler of [Session_error] should absorb it. *)
let check_forward_root_owners t ~conflicting_roots =
  if Lazy.force debug_check_forward_root_owners then
    let missed =
      List.filter (forward_conflicts_by_live_roots t) ~f:(fun r ->
          not (List.mem conflicting_roots r.consumption.fwd_owner ~equal:phys_equal))
    in
    if not (List.is_empty missed) then
      let names = String.concat ~sep:", " @@ List.map missed ~f:(fun r -> Tn.label_name r.value) in
      failwith
        [%string
          "Tensor.consume_forward_code (debug_check_forward_root_owners) for %{debug_name t}: the \
           live forward roots %{names} own nodes it reads, and no owner cell says so: a source of \
           these roots' embedded nodes added no owner edge"]

let consume_forward_code t =
  Shape.check_untainted ();
  check_not_discarded ~what:"Tensor.consume_forward_code" t;
  (* Handing out the same immutable comp is equivalent to the caller retaining it. In particular,
     roots built since the first handout must neither veto replay nor be consumed by it. *)
  if equal_handout t.consumption.fwd Taken then (
    (* Taken by an embedding (e.g. [%cd]) and now handed out: a handout all the same. *)
    note_external_handout ~fwd:true t;
    t.forward)
  else (
    if not @@ is_fwd_root t then
      raise
      @@ Session_error
           ( not_a_root_reason ~handout:t.consumption.fwd ~what:"forward"
               ~name:(Tn.debug_name t.value [@message_only "the refusal raised here"])
               t,
             Some t );
    let conflicting_roots = forward_conflicts t in
    check_forward_root_owners t ~conflicting_roots;
    if not @@ List.is_empty conflicting_roots then
      raise
      @@ Session_error
           ( [%string
               {|Tensor.consume_forward_code for %{debug_name t}:
found conflicting roots with shared non-embedded descendants: %{String.concat ~sep:", " @@ List.map ~f:(fun c -> c.oc_name) conflicting_roots}
(consume them first; a root dropped before its forward code was consumed can never be, so rebuild this tensor's computation)|}],
             Some t );
    note_external_handout ~fwd:true t;
    take_forward_code t)

let consume_backprop_code t =
  Shape.check_untainted ();
  check_not_discarded ~what:"Tensor.consume_backprop_code" t;
  let diff =
    Option.value_or_thunk t.diff ~default:(fun () ->
        raise
        @@ Session_error
             ( "Tensor.consume_backprop_code: tensor is not differentiable for value: "
               ^ debug_name t,
               Some t ))
  in
  if equal_handout t.consumption.bprop Taken then (
    note_external_handout ~fwd:false t;
    diff.backprop)
  else (
    if not @@ is_bprop_root t then
      raise
      @@ Session_error
           ( not_a_root_reason ~handout:t.consumption.bprop ~what:"backprop"
               ~name:(debug_grad t [@message_only "the refusal raised here"])
               t,
             Some t );
    (* Check if any non-embedded grad descendants of t are embedded in other roots *)
    let all_read = fst @@ Asgns.collect_nodes_guess_output diff.backprop.asgns in
    let non_embedded_grad_descendants = Set.diff all_read diff.backprop.embedded_nodes in
    (* Live roots only (gh-ocannl-1745): a root the caller dropped can never be consumed, so it is
       not the fix a refusal would ask for; a dependence on its propagation is refused by the
       lineage's uid-keyed [requires] check at [Context.run]. *)
    let other_roots =
      Weak_root.live session_state.backprop_roots
      |> List.filter ~f:(fun r -> r.value.id <> t.value.id)
    in
    let conflicting_roots =
      List.filter other_roots ~f:(fun root ->
          match root.diff with
          | Some rdiff ->
              not
                (Set.is_empty
                   (Set.inter non_embedded_grad_descendants rdiff.backprop.embedded_nodes))
          | None -> false)
    in
    if not @@ List.is_empty conflicting_roots then
      raise
      @@ Session_error
           ( [%string
               {|Tensor.consume_backprop_code for %{debug_grad t}:
found conflicting roots with shared non-embedded grad descendants: %{String.concat ~sep:", " @@ List.map ~f:debug_grad conflicting_roots}|}],
             Some t );
    note_external_handout ~fwd:false t;
    remove_bprop_root t;
    t.consumption.bprop <- Taken;
    diff.backprop)

let set_random_seed ?seed () =
  let seed =
    Option.value ~default:42 @@ Option.first_some seed Utils.settings.fixed_state_for_init
  in
  let res = bits ~label:[ "random_seed" ] ~grad_spec:Prohibit_grad (Int64.of_int seed) in
  Tn.update_prec res.value Ir.Ops.uint4x32;
  (* The seed singleton is restored by a rewind, so replacing it counts (gh-ocannl-1573); [bits]
     above already counted, through [op]. *)
  Row.note_mutation ();
  if not (List.mem !random_seeds_created seed ~equal:Int.equal) then
    random_seeds_created := seed :: !random_seeds_created;
  Weak_discards.replace seed_literals res.value res.forward;
  random_seed := Some res

let rec get_random_seed () =
  match !random_seed with
  | Some res -> res
  | None ->
      set_random_seed ();
      get_random_seed ()

let with_saved_random_seed f =
  let saved = !random_seed in
  let closure = !last_closure in
  Exn.protectx ~f
    ~finally:(fun () ->
      random_seed := saved;
      (* A closure inside [f] checkpointed [f]'s scoped seed: the restored one is the closed state's
         (gh-ocannl-1301). *)
      if not (phys_equal !last_closure closure) then
        last_closure :=
          {
            !last_closure with
            ck_globals = { !last_closure.ck_globals with g_random_seed = saved };
          })
    ()
[@@mutation_serial.boundary
  "a scoped restore: puts back the seed saved at entry, on return and on a raise alike"]

let random_seeds () = List.rev !random_seeds_created

let install_validity_check () =
  Tn.validity_check :=
    if Weak_discards.length (discarded_nodes ()) = 0 then None
    else
      Some
        (fun tn ->
          Option.iter (discarded_by tn) ~f:(fun provenance ->
              raise @@ Session_error (discarded_message tn provenance, None)))

(* The end of an id-neutral window, whose every node a rewind discarded: those nodes are reported
   under the window's provenance. *)
let end_id_neutral_window ~label =
  List.iter !window_discarded_nodes ~f:(fun (tn : Tn.t) ->
      Weak_discards.replace (discarded_nodes ()) tn label);
  window_discarded_nodes := [];
  install_validity_check ()

(* Restores the graph half of [ck] and the shape-inference half, discarding what the window
   allocated. The caller has validated [ck]. [omit] skips one restore category: negative controls
   only ([For_testing.rewind_omitting]). *)
let restore_checkpoint ?omit ~provenance ck =
  let skip c = Option.exists omit ~f:(String.equal c) in
  let { g_forward_roots; g_backprop_roots; g_random_seed; g_random_seeds_created; g_open_nodes } =
    ck.ck_globals
  in
  if not (skip "node discard") then (
    (* The nodes allocated since the checkpoint: [open_nodes] grows by consing onto the list the
       checkpoint captured. *)
    let rec since acc = function
      | l when phys_equal l g_open_nodes -> acc
      | [] -> acc
      | (tn : Tn.t) :: rest -> since (tn :: acc) rest
    in
    List.iter (since [] !open_nodes) ~f:(fun (tn : Tn.t) ->
        Weak_discards.replace (discarded_nodes ()) tn provenance;
        if !id_neutral_depth > 0 then window_discarded_nodes := tn :: !window_discarded_nodes);
    install_validity_check ());
  if not (skip "roots") then (
    (* The checkpoint's roots, wholesale: a root a discarded construction embedded (via [Tensor.op]
       or [%cd]'s [take_forward_code]) is a root again, with its handout marker reset -- only a root
       is ever taken, so these are all the markers the window could have set. External handouts are
       not taken back. *)
    let fwd = Map.filter_keys g_forward_roots ~f:(fun id -> not (Set.mem handouts.fwd_out id)) in
    (* The surviving backprop roots (gh-ocannl-1745): a root the GC reclaimed since the checkpoint
       had no holder, so no one can name it again, and its leaves go back (gh-ocannl-1804) -- unless
       its code was handed out, which a routine outlives the tensor with. So the handouts are
       filtered first, and again for a leaf handed out after an earlier hand-back. *)
    let not_handed_out =
      Map.filter ~f:(fun e ->
          not (Option.exists (Weak_root.grad_uid e) ~f:(Set.mem handouts.bprop_out)))
    in
    let bprop = not_handed_out (Weak_root.prune (not_handed_out g_backprop_roots)) in
    (* The owner cells follow (gh-ocannl-1796), a reclaimed root's included: its entry holds its
       cell, so whether the GC got to it changes nothing about its readers. *)
    restore_root_cells ~restored:fwd;
    session_state.forward_roots <- fwd;
    session_state.backprop_roots <- bprop;
    List.iter (Weak_root.live fwd) ~f:(fun t -> t.consumption.fwd <- Not_handed_out);
    List.iter (Weak_root.live bprop) ~f:(fun t -> t.consumption.bprop <- Not_handed_out));
  (* The seed singleton as it was at the checkpoint: one created since is dropped, one replaced
     since ([set_random_seed]) comes back. *)
  if not (skip "seed") then (
    random_seed := g_random_seed;
    random_seeds_created := g_random_seeds_created);
  open_nodes := g_open_nodes;
  if not (skip "nodes") then List.iter ck.ck_nodes ~f:Tn.restore_open;
  Shape.Checkpoint.rewind ?omit ck.ck_shape ~provenance
[@@mutation_serial.boundary "the rewind: it restores the checkpoint and clears the taint"]

let rewind_provenance ~default reason =
  match Shape.taint_provenance () with
  | Some provenance -> provenance
  | None -> Option.value reason ~default

let rewind_to_last_closure ?reason () =
  Shape.check_can_rewind ();
  let provenance = rewind_provenance ~default:"an explicit Tensor.rewind_to_last_closure" reason in
  restore_checkpoint ~provenance !last_closure;
  record_checkpoint ()

let checkpoint ?label () =
  Shape.check_can_rewind ~what:"Tensor.checkpoint" ();
  (* A checkpoint records a consistent session: rewind the failure first. *)
  Shape.check_untainted ();
  take_checkpoint ~closure:false ~label:(Option.value label ~default:"")

let checkpoint_refusal ck =
  match Shape.Checkpoint.refusal ck.ck_shape with
  | Some reason -> Some reason
  | None ->
      List.find_map ck.ck_nodes ~f:(fun s ->
          if Tn.open_snapshot_settled s then
            let name =
              (Tn.debug_name (Tn.open_snapshot_node s)
               [@message_only "the refusal checkpoint raises at once"])
            in
            Some
              [%string
                "the precision of %{name}, open at the checkpoint, began settling after it; a \
                 settled precision is memoized and no rewind can reset it -- rewind to the last \
                 closure instead"]
          else None)

let rewind_impl ?omit ?reason ck =
  Shape.check_can_rewind ~what:"Tensor.rewind" ();
  let name = Shape.Checkpoint.label ck.ck_shape in
  Option.iter (checkpoint_refusal ck) ~f:(fun why ->
      raise
      @@ Session_error
           ( [%string
               "Tensor.rewind: cannot rewind to %{name}: %{why}. Tensor.rewind_to_last_closure () \
                is always available"],
             None ));
  let provenance =
    rewind_provenance ~default:[%string "an explicit Tensor.rewind to %{name}"] reason
  in
  restore_checkpoint ?omit ~provenance ck

let rewind ?reason ck = rewind_impl ?reason ck

let with_checkpoint ?label f =
  let ck = checkpoint ?label () in
  try f ()
  with exn ->
    let bt = Stdlib.Printexc.get_raw_backtrace () in
    (* [f]'s exception wins: a checkpoint [f] invalidated (by closing inference) cannot be rewound
       to, and the session keeps whatever taint [f] left, naming its failure. *)
    (try rewind ck with Session_error _ -> ());
    Stdlib.Printexc.raise_with_backtrace exn bt

let with_rewind ?label f =
  let ck = checkpoint ?label () in
  match f () with
  | result ->
      (* Loud on purpose: a trace that cannot be rewound has left its work in the session. *)
      rewind ck;
      result
  | exception exn ->
      let bt = Stdlib.Printexc.get_raw_backtrace () in
      (try rewind ck with Session_error _ -> ());
      Stdlib.Printexc.raise_with_backtrace exn bt

(** {2 Shape schemes (gh-ocannl-404)} *)

module Scheme = struct
  type tensor = t
  type t = Shape.Scheme.t

  let label = Shape.Scheme.label
  let arity = Shape.Scheme.arity
  let num_results = Shape.Scheme.num_results
  let to_string = Shape.Scheme.to_string

  let applications = ref 0
  [@@mutation_serial.exempt "a counter naming default application labels, not inference state"]

  (* Id neutrality (gh-ocannl-1751). A rewind does not restore the tensor id counter: ids are never
     reused. So a window that is always rewound -- a trace, a definition check -- allocates its
     tensors' ids from a reserved range above every session id, with a counter of its own that is
     never reused either, and restores the session counter on exit. Ids the session hands out are
     then the same whether or not the window ran: printed ids, generated identifiers, cache identity
     and [Embed_self_id]-seeded random draws do not move. Node uids need no such range: they are
     identity only, and the compiler's choices do not depend on their gaps (gh-ocannl-2053). Both id
     invariants hold: a checkpoint's [id_mark] orders pre-existing shapes before later ones within
     either range, and [op]'s [id >= next_id] check catches a handle that outlived its window. *)
  let reserved_next_id = ref reserved_id_base
  [@@mutation_serial.exempt
    "a fresh-id counter of the id-neutral windows: never reused, so never restored"]

  let id_neutral_sabotaged = ref false
  [@@mutation_serial.exempt "a negative control's switch, set by tests only"]

  (* Every use rewinds all the window allocated: [label] is the provenance its stale handles
     report. *)
  let with_id_neutral ~label f =
    if !id_neutral_depth > 0 || !id_neutral_sabotaged then f ()
    else (
      assert (session_state.next_id < reserved_id_base);
      let saved = session_state.next_id in
      let first = !reserved_next_id in
      session_state.next_id <- first;
      Int.incr id_neutral_depth;
      Exn.protect
        ~f:(fun () -> Row.with_reserved_vars f)
        ~finally:(fun () ->
          Int.decr id_neutral_depth;
          let last = session_state.next_id in
          reserved_next_id := max !reserved_next_id last;
          assert (!reserved_next_id < Int.max_value / 2);
          session_state.next_id <- saved;
          end_id_neutral_window ~label))

  (* The trace proper, to run inside a rewound window. *)
  let trace_core ?sabotage ~label arity f =
    let var_mark = Row.var_mark () in
    (* The tracers' terminal steps are the trace's work, discarded with it: not inline
       registrations. Made after [var_mark], so that their variables are generalized, and before the
       trace's constraint window opens. *)
    let tracers =
      Shape.as_tracing ~label (fun () ->
          List.init arity ~f:(fun i ->
              term ~grad_spec:Prohibit_grad ~label:[ [%string "%{label}.arg%{i#Int}"] ] ()))
    in
    Shape.Scheme.trace ?sabotage ~var_mark ~label
      ~args:(List.map tracers ~f:(fun t -> t.shape))
      (fun () -> List.map (f tracers) ~f:(fun t -> t.shape))

  let trace_impl ?sabotage ~label arity f =
    try
      let label' = [%string "the trace of the shape scheme %{label}"] in
      with_id_neutral ~label:label' (fun () ->
          with_rewind ~label:label' (fun () -> trace_core ?sabotage ~label arity f))
    with Shape.Untraceable_scheme msg ->
      (* The explicit layer keeps its contract: every refusal is a shape error. *)
      let bt = Stdlib.Printexc.get_raw_backtrace () in
      Stdlib.Printexc.raise_with_backtrace (Row.Shape_error (msg, [])) bt

  let trace ~label arity f = trace_impl ~label arity f

  let instantiate_impl ?sabotage ?application s ~args ~results =
    let application =
      match application with
      | Some a -> a
      | None ->
          Int.incr applications;
          [%string "application #%{!applications#Int}"]
    in
    Shape.Scheme.instantiate ?sabotage s ~application ~args ~results

  let instantiate ?application s ~args ~results = instantiate_impl ?application s ~args ~results

  (* {3 Automatic schemes for [%op] functions (gh-ocannl-1751)} *)

  type event = Traced of t | Unschemed of string * string | Checked of string

  (* Observers of the automatic schemes' events, innermost first: [traced_by] and [check_definition]
     collect the schemes traced, and tests read the log. *)
  let observers : (event -> unit) list ref = ref []
  [@@mutation_serial.exempt "observers of scheme events, restored by their installers' finally"]

  let emit ev = List.iter !observers ~f:(fun f -> f ev)

  let observing f ~on_event =
    observers := on_event :: !observers;
    Exn.protect ~f ~finally:(fun () ->
        match !observers with _ :: rest -> observers := rest | [] -> ())

  (* Session resets so far: a scheme traced before one is traced again. *)
  let generation = ref 0
  [@@mutation_serial.exempt "a count of session resets, which only retires schemes"]

  type mode = [ `Check | `Off ]

  let mode : mode Lazy.t =
    lazy
      (match
         String.lowercase (Utils.get_global_arg ~default:"check" ~arg_name:"op_shape_schemes")
       with
      | "check" -> `Check
      | "off" -> `Off
      | other ->
          raise
          @@ Utils.User_error
               [%string "Invalid op_shape_schemes setting %{other}: expected check or off"])

  let mode_override : mode option ref = ref None
  [@@mutation_serial.exempt "a test override of a process-lifetime setting"]

  let current_mode () = Option.value !mode_override ~default:(Lazy.force mode)
  let enabled () = match current_mode () with `Check -> true | `Off -> false
  let registration_counts = Shape.registration_counts
  let reset_registration_counts = Shape.reset_registration_counts
  let registration_times = Shape.registration_times
  let reset_registration_times = Shape.reset_registration_times

  type state =
    | Pending
    | Schemed of { scheme : t; generation : int }
    | Unschemed of string  (** The reason. *)

  (* One scheme per presence pattern (gh-ocannl-1792): which of the function's option-valued tensor
     parameters are [Some]. [pattern] covers every position, a required one always present. *)
  type slot = { pattern : bool list; label : string; mutable state : state }

  type handle = {
    body : tensor option list -> tensor list;
        (** The function applied to tracers, returning its result tensors. *)
    slots : slot list;
  }

  (* [body] applied to the tracers [args], returning its result tensors (gh-ocannl-1801). The ppx
     cannot name a function's result type, so they are found by physical identity, among the tensors
     the body made and the tracers: the value itself, or each field of a returned tuple or record
     when every field is one. A copy of a tensor's record ([{ t with forward = ... }], as
     [Nn_blocks]' batch norm returns in training) is that tensor: a block of its size holding its
     node where it does. Any other value -- a function, a list, a tensor made before the body ran --
     gives no results. [Obj] only compares pointers and reads the fields of tag-0 blocks, each a
     value. *)
  let traced_results body args =
    let made = ref [] in
    let outer = !made_by_traced_body in
    made_by_traced_body := Some made;
    let value =
      Exn.protect ~f:(fun () -> body args) ~finally:(fun () -> made_by_traced_body := outer)
    in
    let module Obj = Stdlib.Obj in
    let fields r =
      if Obj.is_block r && Obj.tag r = 0 then List.init (Obj.size r) ~f:(Obj.field r) else []
    in
    (* A copy of [c]'s record: a block of its size holding its node where [c] does. *)
    let is_copy r (c : tensor) =
      let rc = Obj.repr c and node = Obj.repr c.value in
      Obj.is_block r
      && Obj.tag r = 0
      && Obj.size r = Obj.size rc
      && List.exists2_exn (fields rc) (fields r) ~f:(fun fc fr ->
          phys_equal fc node && phys_equal fr node)
    in
    (* Newest first: a result is usually the last tensor made. *)
    let candidates = !made @ List.filter_opt args in
    let find r =
      match List.find candidates ~f:(fun c -> phys_equal r (Obj.repr c)) with
      | Some _ as found -> found
      | None -> List.find candidates ~f:(is_copy r)
    in
    let r = Obj.repr value in
    match find r with
    | Some t -> [ t ]
    | None -> Option.value ~default:[] (Option.all (List.map (fields r) ~f:find))

  let unscheme s reason =
    s.state <- Unschemed reason;
    emit (Unschemed (s.label, reason))

  (* The refusals already reported, by label and rendered reason: a model building one block many
     times (a dropout layer per transformer block) is told once, and a different refusal behind the
     same label still is. Bounded by the distinct pairs. *)
  let warned = Hash_set.create (module String)
  [@@mutation_serial.exempt
    "the refusal warnings already printed: diagnostics, not inference state"]

  (* A trace refused by design leaves the function unschemed, which it does not choose
     (gh-ocannl-1791): said on stderr, with the reason rendered as a shape error is
     (gh-ocannl-1799), so the reserved-range ids of the trace's tensors and variables read by first
     appearance. *)
  let unscheme_refused s reason =
    unscheme s reason;
    (* The label is named once: the reason's own prefix would repeat it. *)
    let reason =
      Option.value ~default:reason
        (String.chop_prefix reason ~prefix:[%string "tracing the shape scheme %{s.label}: "])
    in
    let text = Shape.error_to_string_hum reason [] in
    let key = s.label ^ "\000" ^ text in
    if not (Hash_set.mem warned key) then (
      Hash_set.add warned key;
      Stdio.eprintf
        "OCANNL warning: the shape scheme of %s was refused, so it stays unschemed (its shape \
         errors surface only when its body runs): %s\n\
         %!"
        s.label text)

  (* A rewind that must succeed for the fallback to be safe. When it is refused -- the trace read a
     pre-existing open shape's committed state (forcing its dims), and a lazy over that read may
     hold the failure, which no rewind can reset -- the function cannot degrade silently: loud,
     naming the function and the remedy. *)
  let rewind_or_reraise s ck exn bt =
    match rewind ck with
    | () -> ()
    | exception Session_error (why, _) ->
        Stdlib.Printexc.raise_with_backtrace
          (Row.Shape_error
             ( [%string
                 "%{Shape.exn_summary exn}; and the rewind of the trace of the shape scheme \
                  %{s.label} was refused (%{why}), so the function cannot be left unschemed -- opt \
                  it out of its scheme with [@@ocannl.no_scheme]"],
               [] ))
          bt

  (* The body's arguments for a pattern: the tracers in the present positions, in order. *)
  let rec spread pattern tracers =
    match (pattern, tracers) with
    | [], _ -> []
    | true :: pattern, t :: tracers -> Some t :: spread pattern tracers
    | true :: _, [] -> assert false
    | false :: pattern, tracers -> None :: spread pattern tracers

  let trace_slot h s =
    if not (Shape.can_checkpoint ()) then
      (* Tainted, or mid-construction: the body itself reports what it reports, as without a scheme.
         Left pending, to trace when a checkpoint is possible. *)
      ()
    else
      let label = [%string "the trace of the shape scheme %{s.label}"] in
      Shape.timed `Tracing @@ fun () ->
      with_id_neutral ~label (fun () ->
          let ck = checkpoint ~label () in
          match
            trace_core ~label:s.label (List.count s.pattern ~f:Fn.id) (fun ts ->
                h.body (spread s.pattern ts))
          with
          | scheme ->
              rewind ck;
              s.state <- Schemed { scheme; generation = !generation };
              emit (Traced scheme)
          | exception (Row.Shape_error _ as exn) ->
              (* A contradiction that does not depend on the arguments: loud, naming the
                 function. *)
              let bt = Stdlib.Printexc.get_raw_backtrace () in
              (try rewind ck with Session_error _ -> ());
              Stdlib.Printexc.raise_with_backtrace exn bt
          | exception (Shape.Untraceable_scheme msg as exn) ->
              let bt = Stdlib.Printexc.get_raw_backtrace () in
              rewind_or_reraise s ck exn bt;
              unscheme_refused s msg
          | exception exn ->
              (* The body cannot run on tracers (it reads concrete dims, say): degrade only once the
                 session is verifiably back where it was. *)
              let bt = Stdlib.Printexc.get_raw_backtrace () in
              rewind_or_reraise s ck exn bt;
              unscheme_refused s
                [%string
                  "tracing the shape scheme %{s.label}: the function raised %{Shape.exn_summary \
                   exn}"])

  (* Every presence pattern of the positions: an option-valued one ([Some name]) present first, then
     absent; a required one ([None]) always present. The all-present pattern comes first. *)
  let rec patterns = function
    | [] -> [ [] ]
    | None :: rest -> List.map (patterns rest) ~f:(fun p -> true :: p)
    | Some _ :: rest ->
        let ps = patterns rest in
        List.map ps ~f:(fun p -> true :: p) @ List.map ps ~f:(fun p -> false :: p)

  (* [name] when every parameter is required; otherwise [name [train_step, no mask]]. *)
  let slot_label ~name optional pattern =
    match List.filter_opt optional with
    | [] -> name
    | _ ->
        let parts =
          List.filter_map (List.zip_exn optional pattern) ~f:(function
            | None, _ -> None
            | Some n, true -> Some n
            | Some n, false -> Some ("no " ^ n))
        in
        let parts = String.concat ~sep:", " parts in
        [%string "%{name} [%{parts}]"]

  let make ~name ~optional body =
    {
      body = traced_results body;
      slots =
        List.map (patterns optional) ~f:(fun pattern ->
            { pattern; label = slot_label ~name optional pattern; state = Pending });
    }

  module For_ppx = struct
    let closure_variants ~name ~optional body =
      let h = make ~name ~optional body in
      if not (enabled ()) then
        List.iter h.slots ~f:(fun s -> s.state <- Unschemed "automatic shape schemes are off")
      else if Shape.tracing_in_progress () then
        List.iter h.slots ~f:(fun s -> unscheme s "created inside a shape-scheme trace")
      else List.iter h.slots ~f:(trace_slot h);
      h

    let deferred_variants ~name ~optional body =
      let h = make ~name ~optional body in
      if Shape.tracing_in_progress () then
        List.iter h.slots ~f:(fun s -> unscheme s "created inside a shape-scheme trace");
      h

    let all_required arity = List.init arity ~f:(fun _ -> None)

    let present_all body args =
      body (List.map args ~f:(function Some t -> t | None -> assert false))

    let closure ~name ~arity body =
      closure_variants ~name ~optional:(all_required arity) (present_all body)

    let deferred ~name ~arity body =
      deferred_variants ~name ~optional:(all_required arity) (present_all body)

    let application_name args =
      match args with
      | [] -> "no arguments"
      | _ -> String.concat ~sep:", " (List.map args ~f:(fun t -> Tn.label_name t.value))

    (* Whether the body of a checked application is running (gh-ocannl-1814). Its trace ran every
       schemed application in that body plain, so the scheme just checked holds their constraints:
       one checked below it would re-solve a renamed subset of what the outer check found
       consistent. Such an application only runs its body, and an argument its scheme would refuse
       has already been refused naming the outer function. What the outer window does not hold is
       the patterns that application does not take: a first application still traces those. What
       this rests on is gh-ocannl-1751's determinism premise, that a schemed body's graph depends
       only on its closure and its arguments' shapes; a body breaking it still registers its
       constraints inline, so an error moves to the body or the closure and is not lost. *)
    let in_checked_body = ref false
    [@@mutation_serial.exempt
      "the extent of a checked application's body, restored by run_checked_body's finally"]

    let recheck_nested = ref false
    [@@mutation_serial.exempt "a negative control's switch, set by tests only"]

    let run_checked_body body =
      let outer = !in_checked_body in
      in_checked_body := true;
      Exn.protect ~f:body ~finally:(fun () -> in_checked_body := outer)

    (* The check of one application: the scheme's constraints against the arguments, solved, and
       then put back -- the session afterwards is what the body alone makes. False when no check
       could run, so that nothing vouches for the body's applications. *)
    let check s scheme args =
      Shape.can_checkpoint ()
      &&
      ((* Its renamed variables die with it: minted from the reserved range. *)
       Row.with_reserved_vars (fun () ->
           Shape.Scheme.check scheme ~application:(application_name args)
             ~args:(List.map args ~f:(fun t -> t.shape)));
       Shape.note_checked_application ();
       emit (Checked s.label);
       true)

    let due s =
      match s.state with
      | Pending -> true
      | Schemed { scheme; generation = g } -> g <> !generation || Shape.Scheme.stale scheme
      | Unschemed _ -> false

    let apply_variants h args body =
      if (not (enabled ())) || Shape.tracing_in_progress () then body ()
      else
        let pattern = List.map args ~f:Option.is_some in
        match List.find h.slots ~f:(fun s -> List.equal Bool.equal s.pattern pattern) with
        | None -> body ()
        | Some s ->
            let first =
              List.for_all h.slots ~f:(fun s -> match s.state with Pending -> true | _ -> false)
            in
            if !in_checked_body && not !recheck_nested then (
              if first && List.length h.slots > 1 then List.iter h.slots ~f:(trace_slot h);
              body ())
            else (
              (* Tracing begins with every pattern (a function without a [()] stage, or a closure
                 built when no checkpoint was possible); afterwards, each slot re-traces on its
                 own. *)
              if first then List.iter h.slots ~f:(trace_slot h) else if due s then trace_slot h s;
              let checked =
                match s.state with
                | Schemed { scheme; _ } -> check s scheme (List.filter_opt args)
                | Pending | Unschemed _ -> false
              in
              if checked then run_checked_body body else body ())

    let apply h args body = apply_variants h (List.map args ~f:Option.some) body
  end

  let slot_status s =
    match s.state with
    | Pending -> "pending"
    | Schemed _ -> "schemed"
    | Unschemed reason -> "unschemed: " ^ reason

  let status h =
    match h.slots with
    | [ s ] -> slot_status s
    | slots ->
        String.concat ~sep:"; "
          (List.map slots ~f:(fun s -> [%string "%{s.label}: %{slot_status s}"]))

  let traced_by f =
    let traced = ref [] in
    let result = observing f ~on_event:(function Traced s -> traced := s :: !traced | _ -> ()) in
    (result, List.rev !traced)

  let check_definition ~name f =
    try
      let label = [%string "the check of the definition %{name}"] in
      snd (with_id_neutral ~label (fun () -> with_rewind ~label (fun () -> traced_by f)))
    with Row.Shape_error (msg, trace) ->
      let bt = Stdlib.Printexc.get_raw_backtrace () in
      Stdlib.Printexc.raise_with_backtrace
        (Row.Shape_error ([%string "checking the definition %{name}: %{msg}"], trace))
        bt
end

module For_testing = struct
  let restore_categories =
    [ "node discard"; "roots"; "seed"; "nodes" ] @ Shape.Checkpoint.categories

  let trace_scheme ~sabotage ~label arity f = Scheme.trace_impl ~sabotage ~label arity f

  let instantiate_scheme ~sabotage ?application s ~args ~results =
    Scheme.instantiate_impl ~sabotage ?application s ~args ~results

  let observe_schemes f ~on_event = Scheme.observing f ~on_event
  let set_op_shape_schemes v = Scheme.mode_override := v

  let discarded_registry_sizes () =
    Weak_discards.clean (discarded_nodes ());
    ("discarded nodes", Weak_discards.length (discarded_nodes ()))
    :: Shape.discarded_registry_sizes ()

  let with_nested_rechecks f =
    Scheme.For_ppx.recheck_nested := true;
    Exn.protect ~f ~finally:(fun () -> Scheme.For_ppx.recheck_nested := false)

  let without_id_neutrality f =
    Scheme.id_neutral_sabotaged := true;
    Exn.protect ~f ~finally:(fun () -> Scheme.id_neutral_sabotaged := false)

  let rewind_omitting ~omit ck =
    if not (List.mem restore_categories omit ~equal:String.equal) then
      invalid_arg ("Tensor.For_testing.rewind_omitting: unknown category " ^ omit);
    rewind_impl ~omit ck

  let checkpoint_differences ck =
    let { g_forward_roots; g_backprop_roots; g_random_seed; g_random_seeds_created; g_open_nodes } =
      ck.ck_globals
    in
    let now = capture_globals () in
    List.filter_map
      [
        ("Tensor forward roots", phys_equal now.g_forward_roots g_forward_roots);
        ("Tensor backprop roots", phys_equal now.g_backprop_roots g_backprop_roots);
        ("Tensor random seed", phys_equal now.g_random_seed g_random_seed);
        ("Tensor random seeds created", phys_equal now.g_random_seeds_created g_random_seeds_created);
        ("Tensor open nodes", phys_equal now.g_open_nodes g_open_nodes);
      ]
      ~f:(fun (name, same) -> if same then None else Some name)
    @ List.concat_map ck.ck_nodes ~f:Tn.open_snapshot_differences
    @ Shape.Checkpoint.differences ck.ck_shape

  let forward_conflicts_by_cells t = List.map (forward_conflicts t) ~f:(fun c -> c.oc_name)

  let forward_conflicts_by_live_roots t =
    List.map (forward_conflicts_by_live_roots t) ~f:(fun r -> Tn.label_name r.value)

  let owner_count t = List.length t.consumption.fwd_owner.owners
  let drop_owner_edges t = t.consumption.fwd_owner.owners <- []

  let checkpoint_sizes ck =
    ("open nodes", List.length ck.ck_globals.g_open_nodes) :: Shape.Checkpoint.sizes ck.ck_shape
end

let%track5_sexp unsafe_reinitialize ?(namespace = Tn.default_namespace) () : unit =
  Tn.set_current_namespace namespace;
  session_state.next_id <- 0;
  Int.incr Scheme.generation;
  session_state.forward_roots <- Map.empty (module Int);
  session_state.backprop_roots <- Map.empty (module Int);
  Owner_cells.reset owner_cells;
  (* A node an earlier session's rewind discarded stays refused: its record goes with the node. *)
  Weak_discards.clean (discarded_nodes ());
  install_validity_check ();
  param_postprocess := Fn.id;
  random_seed := None;
  random_seeds_created := [];
  Tn.Registry.clear Tn.registry;
  (* The analysis cache retains recent routines' lowered code (gh-560): release the old session's
     nodes promptly. Stale entries could never alias fresh nodes anyway (uids are not reused). *)
  Ir.Low_level.clear_analysis_cache ();
  (* The rewrite tier's members memoize the scope-local nodes they mint per source node, so that
     sibling lowerings share one analysis-cache key; the memo retains them like the cache does. *)
  Ir.Rewrites.reset ();
  Shape.unsafe_reinitialize ();
  record_checkpoint ()
[@@mutation_serial.boundary "the session reset: it discards all state and clears the taint"]

let header t =
  let v_dims_s = Tn.dims_to_string t.value in
  let g_dims_s =
    match t.diff with None -> "<no-grad>" | Some diff -> Tn.dims_to_string diff.grad
  in
  let dims_s =
    if String.equal v_dims_s g_dims_s then "dims " ^ v_dims_s
    else "dims val " ^ v_dims_s ^ " grad " ^ g_dims_s
  in
  "#" ^ Int.to_string t.value.id ^ " " ^ Tn.label t.value ^ " " ^ dims_s ^ " ["
  ^ String.concat ~sep:","
      (List.map t.children ~f:(fun { subtensor; _ } -> Int.to_string subtensor.value.id))
  ^ "]"
(*^" "^PrintBox_text.to_string (PrintBox.Simple.to_box v.label)*)

(* On-demand value retrieval for printing (gh-ocannl-333). Values live only on devices; rendering
   them requires an explicit context, off by default for [Tensor.print] and on for [Train.printf].
   gh-ocannl-777: whether the node can be read is asked first ({!Context.observability}); a readable
   node is read -- a virtual one recomputed -- and any failure of that read propagates, while an
   unreadable one is shown as a placeholder naming why. No context: [<no-context>]. gh-ocannl-1737:
   a printer names every node it will show up front and reads the readable ones together
   ({!Context.to_host_all}), so all the virtual nodes of a tree are recomputed by one routine. The
   result answers for exactly the nodes it was given. *)
let retrieve_all_for_print ?ctx (tns : Tn.t list) : Tn.t -> (Nd.t, string) Either.t =
  match ctx with
  | None -> fun _ -> Second "<no-context>"
  | Some ctx ->
      let answers = Hashtbl.create (module Tn) in
      let readable =
        List.filter (List.dedup_and_sort tns ~compare:Tn.compare) ~f:(fun tn ->
            match Context.observability_tag (Context.observability ctx tn) with
            | None -> true
            | Some tag ->
                Hashtbl.set answers ~key:tn ~data:(Either.Second tag);
                false)
      in
      List.iter2_exn readable (Context.to_host_all ctx readable) ~f:(fun tn nd ->
          Hashtbl.set answers ~key:tn ~data:(Either.First nd));
      Hashtbl.find_exn answers

let optional_payload ~retrieve ~present ~missing v =
  match (retrieve v : (Nd.t, string) Either.t) with
  | First nd -> present nd
  | Second tag -> `Vlist (false, [ `Text (missing ()); `Text tag ])

type array_print_style =
  [ `Default | `Inline | `Label_layout of (string * int) list | `N5_layout of string ]
[@@deriving sexp_of]

let%debug5_sexp to_dag ?(single_node = false) ?(embedded_only = false) ?entries_per_axis ?ctx
    ~with_shape ~with_id ~with_value ~with_grad t =
  check_not_discarded ~what:"Tensor printing" t;
  (* First scan to identify which tensors appear embedded anywhere *)
  let tensors_with_embedded_occurrence = Hash_set.create (module Int) in
  let rec scan_for_embedded { subtensor = t; embedded } =
    if embedded then Hash_set.add tensors_with_embedded_occurrence t.value.id;
    if not single_node then List.iter ~f:scan_for_embedded t.children
  in
  if not embedded_only then scan_for_embedded { subtensor = t; embedded = true };

  (* gh-ocannl-1737: the dag is rendered twice when values are read -- a first pass only records the
     nodes it shows, so that the second reads all of them at once. *)
  let retrieve = ref (retrieve_all_for_print []) in
  let visited = if embedded_only then None else Some (Hash_set.create (module Int)) in
  let rec to_dag { subtensor = t; embedded } : PrintBox_utils.dag =
    let id = Int.to_string t.value.id in
    let children = if single_node then [] else List.map ~f:to_dag t.children in
    let indices = Shape.default_display_indices t.shape in
    (* Reserved (claim-free) tags ([default] / [bcast_if_1]) are not user-meaningful axis names;
       display them as blank, as the old [None] basis was displayed. *)
    let labels =
      Array.map (Shape.to_bases t.shape) ~f:(fun l -> if Row.is_reserved_basis l then "" else l)
    in
    let where_located a = Tn.(debug_memory_mode a.memory_mode_intent) in
    let txt =
      if with_id then "#" ^ id ^ " " ^ Tn.label t.value (* ^ " DEBUG: " ^ where_located t.value *)
      else Tn.label t.value
    in
    let grad_txt diff =
      let label = Tn.label diff.grad in
      assert (String.is_prefix label ~prefix:"grad");
      if with_id then
        "#" ^ Int.to_string diff.grad.id ^ " " ^ label (* ^ " DEBUG: " ^ where_located diff.grad *)
      else label
    in
    let add_shape nodes =
      if with_shape then
        let shape = `Box (PrintBox.asprintf "%a" Sexp.pp_hum ([%sexp_of: Shape.t] t.shape)) in
        `Vlist (false, nodes @ [ shape ])
      else `Vlist (false, nodes)
    in
    let should_elide =
      if embedded_only then not embedded
      else if
        (* If this tensor appears embedded anywhere, use embedded logic for consistency *)
        Hash_set.mem tensors_with_embedded_occurrence t.value.id
      then not embedded
      else
        (* Only use visited tracking for tensors that are never embedded anywhere *)
        match visited with
        | None -> not embedded
        | Some visited_set ->
            if Hash_set.mem visited_set t.value.id then true
            else (
              Hash_set.add visited_set t.value.id;
              false)
    in
    let txt = txt ^ if (not should_elide) && not embedded then " non-emb" else "" in
    match (should_elide, with_value, with_grad, t.diff) with
    | true, _, _, _ -> `Embed_subtree_ID txt
    | _, false, false, _ | _, false, true, None ->
        `Subtree_with_ID (id, `Tree (add_shape [ `Text txt ], children))
    | _, true, false, _ | _, true, true, None ->
        let node =
          optional_payload ~retrieve:!retrieve t.value
            ~present:(fun v_array ->
              `Box
                (Nd.render_array ~brief:true ~prefix:txt ?entries_per_axis ~labels ~indices v_array))
            ~missing:(fun () -> txt ^ " " ^ where_located t.value)
        in
        `Subtree_with_ID (id, `Tree (add_shape [ node ], children))
    | _, false, true, Some diff ->
        let prefix =
          grad_txt diff ^ if (not should_elide) && not embedded then " non-emb" else ""
        in
        let node =
          optional_payload ~retrieve:!retrieve diff.grad
            ~present:(fun g_array ->
              `Box (Nd.render_array ~brief:true ~prefix ?entries_per_axis ~labels ~indices g_array))
            ~missing:(fun () -> prefix ^ " " ^ where_located diff.grad)
        in
        `Subtree_with_ID (id, `Tree (add_shape [ node ], children))
    | _, true, true, Some diff ->
        let node =
          let value =
            optional_payload ~retrieve:!retrieve t.value
              ~present:(fun v_array ->
                `Box
                  (Nd.render_array ~brief:true ~prefix:txt ?entries_per_axis ~labels ~indices
                     v_array))
              ~missing:(fun () -> txt ^ " " ^ where_located t.value)
          in
          let grad =
            optional_payload ~retrieve:!retrieve diff.grad
              ~present:(fun g_array ->
                `Box
                  (Nd.render_array ~brief:true ~prefix:(grad_txt diff) ?entries_per_axis ~labels
                     ~indices g_array))
              ~missing:(fun () -> grad_txt diff ^ " " ^ where_located diff.grad)
          in
          `Vlist (false, [ value; grad ])
        in
        `Subtree_with_ID (id, `Tree (add_shape [ node ], children))
  in
  let root = { subtensor = t; embedded = true } in
  Option.iter ctx ~f:(fun _ ->
      let shown = ref [] in
      (retrieve :=
         fun tn ->
           shown := tn :: !shown;
           Second "");
      ignore (to_dag root : PrintBox_utils.dag);
      Option.iter visited ~f:Hash_set.clear;
      retrieve := retrieve_all_for_print ?ctx !shown);
  to_dag root

let to_printbox ?single_node ?embedded_only ?entries_per_axis ?(with_id = false) ?(force = false)
    ?ctx ?(with_shape = false) ?(with_value = true) ~with_grad ~depth t =
  ignore force;
  to_dag ?single_node ?embedded_only ?entries_per_axis ?ctx ~with_id ~with_shape ~with_value
    ~with_grad t
  |> PrintBox_utils.reformat_dag depth

let%debug_sexp log_debug_info ~from_log_level t =
  [%diagn_sexp
    let%diagn_sexp log_child { subtensor = _subtensor; embedded = _embedded } =
      [%logN_block
        from_log_level
          ((if _embedded then "Embedded " else "Non-embedded ")
          ^ (Tn.debug_name _subtensor.value [@message_only "a debug-log block's title"]));
        Tn.log_debug_info ~from_log_level _subtensor.value]
    in
    [%logN_block
      from_log_level ("Tensor " ^ Tn.dims_to_string t.value);
      Tn.log_debug_info ~from_log_level t.value;
      Option.iter t.diff ~f:(fun diff ->
          [%log_block
            "Gradient";
            Tn.log_debug_info ~from_log_level diff.grad]);
      List.iter ~f:log_child t.children]]

let%debug5_sexp to_doc ?ctx ~force ~with_grad ~with_code ?(with_low_level = false)
    (style : array_print_style) t =
  ignore force;
  let sh = t.shape in
  let label = Tn.label t.value in
  let prefix_str =
    "[" ^ Int.to_string t.value.id ^ "]: " ^ label ^ " shape "
    ^ Shape.to_string_hum ~style:Row.Axis_number_and_size sh
    ^ " "
  in
  let grad_txt diff =
    let label = Tn.label diff.grad in
    assert (String.is_prefix label ~prefix:"grad");
    label
  in
  (* Reserved (claim-free) tags display as blank (see [to_dag]); also keeps [Label_layout] matching
     and [needs_spec] behaving as they did under the old [None] basis. *)
  let labels =
    Array.map (Shape.to_bases t.shape) ~f:(fun l -> if Row.is_reserved_basis l then "" else l)
  in
  let indices =
    match style with
    | `Default -> Shape.default_display_indices sh
    | `N5_layout priorities -> Shape.parse_n5_layout priorities
    | `Label_layout label_idcs ->
        let inv_labels =
          Array.mapi labels ~f:(fun i l -> (l, i)) |> Array.to_list |> Map.of_alist (module String)
        in
        let inv_labels =
          match inv_labels with
          | `Duplicate_key l ->
              raise @@ Session_error ("`Label_layout found a repeating label: " ^ l, Some t)
          | `Ok inv_labels -> inv_labels
        in
        let result = Array.create ~len:(Array.length labels) 0 in
        List.iter label_idcs ~f:(fun (l, priority) ->
            match Map.find inv_labels l with
            | Some pos -> result.(pos) <- priority
            | None -> raise @@ Session_error ("`Label_layout label not found in shape: " ^ l, Some t));
        result
    | `Inline -> [||]
  in
  let needs_spec =
    (* [labels] already blanks reserved (claim-free) tags, so a non-empty label is a real named
       basis that warrants showing the spec — as under the old [None] basis. *)
    Array.exists labels ~f:(Fn.non String.is_empty)
    || Shape.(List.exists ~f:Row.(equal_dim @@ get_default_dim ~d:1 ()) sh.input.dims)
  in
  let axes_spec =
    if needs_spec then Some (Shape.to_string_hum ~style:Row.Only_bases sh) else None
  in
  let num_batch_axes = List.length (sh.batch.beg_dims @ sh.batch.dims) in
  let num_input_axes = List.length (sh.input.beg_dims @ sh.input.dims) in
  let num_output_axes = List.length (sh.output.beg_dims @ sh.output.dims) in

  let open PPrint in
  (* Create document for tensor value *)
  let has_grad = with_grad && Option.is_some t.diff in
  let retrieve =
    retrieve_all_for_print ?ctx
      (t.value :: (match t.diff with Some diff when with_grad -> [ diff.grad ] | _ -> []))
  in
  let value_doc =
    match retrieve t.value with
    | Second tag -> string prefix_str ^^ string (" " ^ tag) ^^ if has_grad then break 1 else empty
    | First arr -> (
        match style with
        | `Inline ->
            string prefix_str ^^ space
            ^^ Nd.to_doc_inline ~num_batch_axes ~num_input_axes ~num_output_axes ?axes_spec arr
            ^^ if has_grad then break 1 else empty
        | _ ->
            Nd.to_doc ~prefix:prefix_str ~labels ~indices arr ^^ if has_grad then break 1 else empty
        )
  in

  (* Create document for gradient *)
  let grad_doc =
    if with_grad then
      match t.diff with
      | Some diff -> (
          match retrieve diff.grad with
          | Second tag -> string (grad_txt diff) ^^ string (" " ^ tag)
          | First arr -> (
              match style with
              | `Inline ->
                  string (grad_txt diff)
                  ^^ space
                  ^^ Nd.to_doc_inline ~num_batch_axes ~num_input_axes ~num_output_axes ?axes_spec
                       arr
              | `Default | `N5_layout _ | `Label_layout _ ->
                  let prefix = prefix_str ^ " " ^ grad_txt diff in
                  Nd.to_doc ~prefix ~labels ~indices arr))
      | None -> empty
    else empty
  in

  (* Create document for code *)
  let code_doc =
    if with_code then
      let fwd_doc =
        match t.forward.asgns with
        | Noop -> empty
        | fwd_code ->
            group (string "Current forward body:" ^^ nest 2 (hardline ^^ Asgns.to_doc () fwd_code))
            ^^ hardline
      in
      let bwd_doc =
        match t.diff with
        | Some { backprop = { asgns = Noop; _ }; _ } -> empty
        | Some { backprop = { asgns = bwd_code; _ }; _ } ->
            group (string "Current backprop body:" ^^ nest 2 (hardline ^^ Asgns.to_doc () bwd_code))
            ^^ hardline
        | None -> empty
      in
      fwd_doc ^^ bwd_doc
    else empty
  in

  (* Create document for low-level code *)
  let low_level_doc =
    if with_low_level then
      let fwd_doc =
        match t.forward.asgns with
        | Noop -> empty
        | fwd_code ->
            group
              (string "Current forward low-level body:"
              ^^ nest 2 (hardline ^^ Ir.Low_level.to_doc () (Asgns.to_low_level fwd_code)))
            ^^ hardline
      in
      let bwd_doc =
        match t.diff with
        | Some { backprop = { asgns = Noop; _ }; _ } -> empty
        | Some { backprop = { asgns = bwd_code; _ }; _ } ->
            group
              (string "Current backprop low-level body:"
              ^^ nest 2 (hardline ^^ Ir.Low_level.to_doc () (Asgns.to_low_level bwd_code)))
            ^^ hardline
        | None -> empty
      in
      fwd_doc ^^ bwd_doc
    else empty
  in

  (* Combine all documents and print *)
  group
    (value_doc ^^ grad_doc
    ^^ (if is_empty value_doc && is_empty grad_doc then empty else hardline)
    ^^ code_doc ^^ low_level_doc)

let print ?here ?(force = false) ?ctx ~with_grad ~with_code ?(with_low_level = false)
    (style : array_print_style) t =
  Option.iter here ~f:(fun here ->
      Stdio.printf "HERE: %s\n%!" (Source_code_position.to_string here));
  check_not_discarded ~what:"Tensor.print" t;
  let doc = to_doc ?ctx ~force ~with_grad ~with_code ~with_low_level style t in
  PPrint.ToChannel.pretty 0.7 100 Stdio.stdout doc;
  Stdio.Out_channel.flush Stdio.stdout

let print_tree ?here ?(force = false) ?ctx ?entries_per_axis ?(with_backend_info = false)
    ?(with_id = true) ?(with_shape = false) ?(with_value = true) ?embedded_only ~with_grad ~depth t
    =
  ignore force;
  Option.iter here ~f:(fun here ->
      Stdio.printf "HERE: %s\n%!" (Source_code_position.to_string here));
  (* FIXME: print backend info *)
  ignore with_backend_info;
  PrintBox_text.output Stdio.stdout @@ PrintBox_utils.dag_to_box @@ PrintBox_utils.boxify depth
  @@ to_dag ?entries_per_axis ?embedded_only ?ctx ~with_id ~with_shape ~with_value ~with_grad t

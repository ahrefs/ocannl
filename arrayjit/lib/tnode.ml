open Base
module Lazy = Utils.Lazy
module Nd = Ndarray

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_TNODE=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_TNODE"]

type memory_mode =
  | Effectively_constant  (** A constant, or a subset of [Virtual]. *)
  | Virtual
      (** The tensor node's computations are inlined on a per-scalar basis. The node has no buffer
          in any context; its defining computation is tracked (in [Low_level.optimize_ctx]), so it
          remains observable: its value can be recomputed on demand, including by later routines and
          for printing. Observability is inductive, not intrinsic: recomputation requires the nodes
          the tracked computation reads to be observable themselves, so a [Virtual] node that
          depends -- directly or transitively through other [Virtual] nodes -- on a [Local] node
          inherits its unobservability (the recompilation raises the same [User_error]). *)
  | Never_virtual  (** An as-yet-unresolved request; resolves to [Local] or [On_device]. *)
  | Local
      (** Routine-scoped scratch: the tensor node exists only for the duration of a single call to a
          compiled function, stored to whatever degree the optimizer decides on (e.g. a stack
          array), and is not persisted across calls. It is not materialized (owns no context buffer)
          and is not available for merging across devices. Unlike [Virtual] -- which stays
          observable via recomputation -- [Local] is {b unobservable}, and the sole source of
          unobservability: its computation is not tracked, and compiling a later routine that reads
          it raises a [User_error] directing to mark it as materialized before its first use. This
          mode is only ever assigned by the compiler (never requested), reserving the freedom to
          optimize placement of nodes whose lifetime is confined to one routine. *)
  | On_device
      (** The tensor node is stored on the devices that compute with it and persisted across
          function calls. It is available for merging across devices (for devices that support
          merging / P2P). CPU-side access (printing, persistence, inspection) is on-demand via
          context-mediated device-to-host transfers; no host copy is stored on the node. *)
[@@deriving sexp, compare, equal]

(** Why a memory mode was requested or decided (gh-ocannl-609).

    Two kinds of tag, and the split is a layering decision, not a convenience. A decision that only
    ever gets {e recorded} is a {!Site} carrying its own explanation: the ~60 of those are minted
    across nine modules, from the C renderer's storage queries to the scheduler's tile placements,
    and giving each a constructor would make this type -- which sits at the bottom of the dependency
    graph, since {!module-Placements} is here -- enumerate the vocabulary of every module above it.
    A tag that some other code {e reads back} gets a constructor instead, so the reading is an
    exhaustive match rather than a string comparison that can silently stop matching.

    A [Site]'s string is spelled ["<code>:<kebab-case-reason>"], and so is every constructor's
    rendering: the numeric code is the integer this used to be, kept so that issues, comments and
    goldens citing e.g. [Non_virtual 13] or "provenance 39" still resolve. Codes are not unique --
    two sites that were the same integer stay the same integer and are told apart by their reasons.
*)
type provenance =
  | Visit_cap
      (** Per-cell visits above [virtualize_max_visits], or an uncovered read. One of the three
          heuristic caps of [Low_level.decide_placements]: policy priors, which the decision-vector
          search may flip back to inlining, unlike a legality or observability verdict. *)
  | Inline_reduction_cap  (** Reduction extent above [virtualize_max_inline_reduction]. *)
  | Inline_fanin_cap  (** Transitive fan-in above [virtualize_max_inline_fanin]. *)
  | Read_before_write
      (** An uncovered read within the routine: the node is an input, so it owns a device buffer
          whose prior contents are preserved. Minted from two sites -- the lenient verdict of
          [decide_placements] and the strict re-classification in [reconcile_traced_store]. *)
  | Scope_local  (** A scope local the inliner minted, committed [Virtual] by cleanup. *)
  | Surviving_read
      (** A read that survived inlining, so its target must be materialized. Cleanup is the
          commitment point, not a re-assertion: a node read here but written only under a
          virtualized setter is decided right now. *)
  | Site of string
      (** A decision no other code reads back, explained by its own tag. Keeping these out of the
          constructor list is what stops this type from growing a case per backend query site. *)
  | Refined of provenance * provenance
      (** A decision that defaulted an earlier one into a concrete placement
          ({!Placements.default_to_most_local}): the prior decision, then the one that resolved it.
          Renders as ["39:inline-reduction-cap -> 432:is-local-materialized-query"] -- what the
          retired [1000 * prior + own] arithmetic encoded as the undecodable [39432]. *)
[@@deriving sexp_of, equal]

let rec provenance_to_string = function
  | Visit_cap -> "1:visit-cap"
  | Inline_reduction_cap -> "39:inline-reduction-cap"
  | Inline_fanin_cap -> "41:inline-fanin-cap"
  | Read_before_write -> "36:read-before-write"
  | Scope_local -> "16:scope-local"
  | Surviving_read -> "17:surviving-read"
  | Site s -> s
  | Refined (prior, refinement) ->
      provenance_to_string prior ^ " -> " ^ provenance_to_string refinement

(** The decision the later ones refined: the leftmost tag of a {!Refined} chain, which
    {!Placements.default_to_most_local} builds left-nested. *)
let rec leading_provenance = function
  | Refined (prior, _) -> leading_provenance prior
  | ( Visit_cap | Inline_reduction_cap | Inline_fanin_cap | Read_before_write | Scope_local
    | Surviving_read | Site _ ) as p ->
      p

type delayed_prec = Default of Ops.prec | Inferred of Ops.prec Lazy.t | Specified of Ops.prec
[@@deriving sexp, equal]

(** Per-tensor scalar value bounds: the interprocedural layer of the interval analysis
    (docs/proposals/interval-analysis-scalar-t.md), the third instance of the {!delayed_prec}
    propose/settle lifecycle. Writers {e propose} bounds (joined, like [Inferred] promotion); a
    reader that discharges a guard against the candidate {e settles} it (like forcing the prec
    lazy); a post-settlement proposal that does not fit the settled interval is an error (like
    {!update_prec} on a settled precision) -- otherwise already-generated code that folded a bounds
    guard away would become unsound.

    Phase B v1 execution anchoring (binding constraint 3): every compiled device write of a node
    proposes [Interval.top] at lowering time (see [Low_level.optimize_proc]), so a candidate can
    only be narrower than [top] for host-initialized, never-device-written tensors; guard folds
    therefore never depend on device-computed values, sidestepping the
    runs-never/runs-later/runs-repeatedly hazards. *)
type bounds_state =
  | Bounds_unknown  (** No proposal yet: readers fall back to the precision's machine range. *)
  | Bounds_proposed of Interval.t  (** Join of all proposals so far. *)
  | Bounds_settled of Interval.t
      (** Consumed by a guard fold; subsequent proposals must fit or raise. *)
[@@deriving sexp, equal]

let default_namespace = "ocannl"

(** Namespaces must be legal C-family identifiers so they can prefix generated-code identifiers and
    debug names verbatim (rendered as [ns__n42]); see docs/proposals gh-ocannl-372. *)
let validate_namespace ns =
  let ok_first c = Char.is_alpha c || Char.equal c '_' in
  let ok c = Char.is_alphanum c || Char.equal c '_' in
  if String.is_empty ns || not (ok_first ns.[0] && String.for_all ns ~f:ok) then
    invalid_arg
      [%string "Tnode: invalid namespace %{String.escaped ns}: must match [A-Za-z_][A-Za-z0-9_]*"]

(* The ambient namespace stamped on newly created tnodes. Only [Tensor.unsafe_reinitialize
   ~namespace] is meant to change it (via {!set_current_namespace}); explicitly-namespaced creation
   ([Persistence.load ~prefix_namespace], schedule-internal tile nodes) bypasses it. *)
let current_namespace = ref default_namespace

let set_current_namespace ns =
  validate_namespace ns;
  current_namespace := ns

let get_current_namespace () = !current_namespace

type t = {
  storage_prec : Ops.prec Lazy.t;
  dims : int array Lazy.t;
  padding : (Ops.axis_padding array * float) option Lazy.t;
      (** If the tensor node is pre-padded, this is the pair of (left padding, right padding) per
          axis and the padding/neutral value. Both the margins and the neutral value are part of the
          node's identity: they commit when this lazy is forced (the node's first compilation), the
          margins permanently hold the neutral value, and later conflicting demands — more padding,
          or margin-touching operations expecting a different neutral — are rejected at
          shape-inference time. *)
  size_in_bytes : int Lazy.t;
  id : int;
      (** The within-session id ("s_id" of gh-ocannl-372): consecutive number for nodes created in
          the current session, restarting at 0 on [Tensor.unsafe_reinitialize]. Full node identity
          for presentation and persistence is the pair ({!field-namespace}, [id]); in-memory
          identity is {!field-uid}. *)
  namespace : (string[@sexp_drop_if String.equal default_namespace]);
      (** The namespace qualifying {!field-id}, stamped at creation from the ambient
          {!current_namespace} unless created with an explicit [?namespace] (persistence loads,
          schedule-internal nodes). Grads share their value node's session namespace. Elided from
          sexps and renderings when it is {!default_namespace}. *)
  uid : (int[@sexp_drop_if fun _ -> true]);
      (** Process-unique identity, from a counter that {b no} reinitialization ever resets -- unlike
          {!field-id}, which restarts at 0 on [Tensor.unsafe_reinitialize] for deterministic
          printing. All comparison/hashing (hence every tnode-keyed map, set and cache in the
          process) uses [uid], so a stale entry surviving a reinitialization can never alias a fresh
          tnode that reuses its [id]. Excluded from sexps to keep debug output
          reinitialization-deterministic. *)
  label : string list;
      (** Display information. It is better if the last element of the list is the most narrow or
          alphanumeric, e.g. an identifier. *)
  mutable delayed_prec_unsafe : delayed_prec;
      (** Participates in the computation of {!field-storage_prec}. *)
  mutable bounds : bounds_state;
      (** Scalar value bounds summary; see {!bounds_state} for the lifecycle. *)
  mutable memory_mode_intent : (memory_mode * provenance) option;
      (** The tnode's {e declared intent} -- requests made at graph-construction time (parameter and
          constant marking, [Train.set_materialized], op-support [Never_virtual]), paired with a
          {!provenance} tag saying which request it was. Since the context-scoped memory-modes split
          (docs/proposals/context-scoped-memory-modes.md) this is monotone, side-effect free to
          read, and never written by the compilation pipeline: placement {e decisions} are
          per-compilation-lineage, recorded in {!module-Placements} tables riding
          [Low_level.optimize_ctx]. *)
  mutable observable : bool;
      (** Declared host-observation intent (docs/proposals/context-scoped-memory-modes.md category
          2): someone intends to read this node's values (printing, persistence, inspection).
          Monotone-upward (set, never cleared) and side-effect free to read. Observation does
          {b not} require materialization -- a [Virtual] resolution stays observable via
          recomputation -- so the only constraint this imposes on placement is: do not resolve the
          node into the [Local]-dependent unobservable class ({!Placements.default_to_most_local}
          resolves [Never_virtual] to [On_device] rather than [Local] for observable nodes). *)
  mutable host_constant : bool;
      (** Declared value-constancy: the node's values are fixed at construction and always equal its
          registered host-init data (an ndarray-backed literal). Set by [Tensor.ndarray] (and
          eligible loaders), never cleared. This carries the constancy that [Effectively_constant]
          intent cannot: ndarray-backed nodes are minted [On_device] (provenance
          [Site "49:ndarray-backed"]), so the memory-mode lattice has no room left for the constant
          marking. Consumed by [Schedule.Stage]'s hoisted packing (gh-ocannl-470) to justify
          materializing a repacked copy once per device. *)
  mutable alias_of : ((t * Indexing.static_symbol) option[@sexp.opaque]);
      (** When [Some (parent, batch_idx)], this node is a zero-copy slice-alias *view* of [parent]:
          it owns no buffer of its own, and every read/write of it is redirected (during lowering)
          to [parent] with [batch_idx] prepended as the leading index. Set by {!Assignments.lower}
          for alias-eligible [Fetch.Slice]s; orthogonal to {!field-memory_mode_intent}. The strong
          reference to [parent] also keeps it reachable for as long as the alias is. *)
  mutable slice_of : ((t * Indexing.static_symbol) option[@sexp.opaque]);
      (** When [Some (parent, batch_idx)], this node is an [\@|] sub-tensor slice of [parent]. Set
          *eagerly at construction* (independent of alias eligibility), so it is a superset of
          {!field-alias_of}: every confirmed alias is a slice, but an ineligible slice (precision-
          converting, padded, virtual parent) is still a slice that falls back to a materializing
          copy. Used to reject direct host access (read/write the parent instead) -- including the
          window before lowering decides eligibility, where {!field-alias_of} is still [None]. *)
  mutable backend_info : Sexp.t;
  mutable code_name : string option;
}
[@@deriving sexp_of]

let compare a1 a2 = compare_int a1.uid a2.uid

(* The [uid] counter deliberately lives outside any reinitializable session state. *)
let next_uid = ref 0

let fresh_uid () =
  let uid = !next_uid in
  next_uid := uid + 1;
  uid

let num_elems tn =
  let dims = Lazy.force tn.dims in
  Array.fold dims ~init:1 ~f:( * )

let dims_without_padding tn =
  match Lazy.force tn.padding with
  | None -> Lazy.force tn.dims
  | Some (padding, _) ->
      let dims = Lazy.force tn.dims in
      Array.map2_exn dims padding ~f:(fun dim { left; right } -> dim - left - right)

let get_padding tn = Lazy.force tn.padding

(* The "n" prefix of numeric idents, qualified by the namespace when non-default: [n42] vs
   [snap1__n42]. Namespace validation guarantees the result is a legal C-family identifier. *)
let ident_prefix namespace =
  if String.equal namespace default_namespace then "n" else namespace ^ "__n"

let id { id; namespace; _ } = ident_prefix namespace ^ Int.to_string id
let label a = String.concat ~sep:"_" a.label

let is_alphanum_ s =
  (not (String.is_empty s)) && String.for_all s ~f:(fun c -> Char.equal c '_' || Char.is_alphanum c)

let collapse_consecutive = function
  | [] -> []
  | first :: rest ->
      let emit ident count acc =
        (if count = 1 then ident else ident ^ Int.to_string count) :: acc
      in
      let acc, last_ident, last_count =
        List.fold rest ~init:([], first, 1) ~f:(fun (acc, cur, cnt) s ->
            if String.equal s cur then (acc, cur, cnt + 1) else (emit cur cnt acc, s, 1))
      in
      List.rev (emit last_ident last_count acc)

let get_debug_name ?code_name ?(namespace = default_namespace) ~id ~label () =
  match code_name with
  | Some code_name -> (
      match String.chop_suffix code_name ~suffix:"_grad" with
      | None -> code_name
      | Some ident -> ident ^ ".grad")
  | None -> (
      let components = List.filter ~f:is_alphanum_ label in
      let components, is_grad =
        match components with "grad" :: components -> (components, true) | _ -> (components, false)
      in
      let components = collapse_consecutive components in
      let ident_label =
        if List.is_empty components then None else Some (String.concat ~sep:"_" components)
      in
      let opt_grad = if is_grad then ".grad" else "" in
      let n = ident_prefix namespace in
      match ident_label with
      | Some ident -> [%string "%{ident}%{opt_grad}"]
      | None when is_grad -> [%string "%{n}%{id - 1#Int}%{opt_grad}"]
      | None -> n ^ Int.to_string id)

let debug_name tn =
  let id = tn.id and label = tn.label and code_name = tn.code_name in
  get_debug_name ?code_name ~namespace:tn.namespace ~id ~label ()

let debug_memory_mode = function
  | None -> "unknown"
  | Some (mem, prov) ->
      (match mem with
        | Effectively_constant -> "Const"
        | Virtual -> "Virt"
        | Never_virtual -> "Non-virt"
        | Local -> "Local"
        | On_device -> "On-dev")
      ^ "/" ^ provenance_to_string prov

let log_debug_info ~from_log_level tn =
  [%debug_sexp
    [%logN_block
      from_log_level (debug_name tn);
      [%log
        "id:",
        (tn.id : int),
        "label:",
        (tn.label : string list),
        "mem:",
        debug_memory_mode tn.memory_mode_intent,
        "backends:",
        (tn.backend_info : Sexp.t)]]]

(** The mode [Never_virtual] resolves to when defaulted: [Local] if the node fits the stack
    threshold, [On_device] otherwise. *)
let most_local_materialized_mode tn =
  let stack_threshold_in_bytes =
    Int.of_string @@ Utils.get_global_arg ~default:"16384" ~arg_name:"stack_threshold_in_bytes"
  in
  if
    stack_threshold_in_bytes > 0
    && num_elems tn > stack_threshold_in_bytes / (Ops.prec_in_bytes @@ Lazy.force tn.storage_prec)
  then On_device
  else Local

(* Note (context-scoped memory modes): the tnode-level forcing family ([is_virtual_force],
   [is_materialized_force], [is_in_context_force], [default_to_most_local], [is_materialized_peek])
   was removed -- {!field-memory_mode_intent} now holds only declared intent, side-effect free to
   read. Settlement lives in {!module-Placements}, per compilation lineage. *)

let is_observable tn = tn.observable

(** Declares host-observation intent (see {!field-observable}). Monotone: there is no unset. *)
let set_observable tn = tn.observable <- true

(** A slice-alias view (see {!field-alias_of}). Such a node owns no buffer of its own; its accesses
    are redirected to its parent during lowering. *)
let is_alias tn = Option.is_some tn.alias_of

let alias_of tn = tn.alias_of

(** Marks [tn] as a zero-copy slice-alias view of [parent] with leading index [batch_idx].
    Idempotent when re-marked with the same parent. *)
let set_alias_of tn ~parent ~batch_idx = tn.alias_of <- Some (parent, batch_idx)

(** Whether [tn] is an [\@|] sub-tensor slice (see {!field-slice_of}) -- set eagerly at
    construction, independent of alias eligibility. A superset of {!is_alias}. *)
let is_slice tn = Option.is_some tn.slice_of

let slice_of tn = tn.slice_of

(** Marks [tn] as an [\@|] slice of [parent] eagerly at construction (before alias eligibility is
    known). Idempotent. *)
let set_slice_of tn ~parent ~batch_idx = tn.slice_of <- Some (parent, batch_idx)

let known_not_materialized tn =
  match tn.memory_mode_intent with Some ((Virtual | Local), _) -> true | _ -> false

let known_constant tn =
  match tn.memory_mode_intent with Some (Effectively_constant, _) -> true | _ -> false

(** Whether [tn]'s values are declared fixed at construction, forever equal to its registered
    host-init data (see {!field-host_constant}). Includes [known_constant] intent: small constants
    keep the [Effectively_constant] marking (their creation does not pass through the
    [On_device]-minting ndarray-backed path). *)
let known_host_constant tn = tn.host_constant || known_constant tn

(** Declares that [tn]'s values are fixed at construction (host-init-backed literal). Monotone: set,
    never cleared. *)
let set_host_constant tn = tn.host_constant <- true

let known_non_virtual tn =
  match tn.memory_mode_intent with
  | None | Some ((Virtual | Effectively_constant), _) -> false
  | _ -> true

let known_virtual tn = match tn.memory_mode_intent with Some (Virtual, _) -> true | _ -> false

let mode_is_unspecified tn =
  match tn.memory_mode_intent with
  | None | Some ((Never_virtual | Effectively_constant), _) -> true
  | _ -> false

(** The pure memory-mode lattice transition, shared between {!update_memory_mode} (tnode-level
    declared intent) and per-context placement resolution ({!module-Placements}). Returns the new
    state; raises on conflicting requests. *)
let transition_memory_mode ~debug_name:name current mode provenance =
  match (current, mode) with
  | None, _ -> (mode, provenance)
  | Some ((m1, _) as cur), m2 when equal_memory_mode m1 m2 -> cur
  | Some (Never_virtual, prov2), Virtual ->
      raise
      @@ Utils.User_error
           [%string
             "Tnode.update_memory_mode: update %{provenance_to_string prov2} -> \
              %{provenance_to_string provenance} for %{name}: cannot be virtual"]
  | Some ((Virtual, _) as cur), Effectively_constant -> cur
  | Some (Never_virtual, _), Effectively_constant | Some (Effectively_constant, _), Never_virtual ->
      (* A constant that must be persisted is just a materialized (device-resident) node now; there
         is no separate hosted-constant state. *)
      (On_device, provenance)
  | Some ((On_device, _) as cur), Effectively_constant -> cur
  | Some (Effectively_constant, _), Virtual -> (mode, provenance)
  | Some (Effectively_constant, _), On_device -> (On_device, provenance)
  | Some (Never_virtual, _), mode -> (mode, provenance)
  | Some (Virtual, prov2), Never_virtual ->
      raise
      @@ Utils.User_error
           [%string
             "Tnode.update_memory_mode: update %{provenance_to_string prov2} -> \
              %{provenance_to_string provenance} for %{name} is already virtual"]
  | Some ((_, _) as cur), Never_virtual -> cur
  | Some (_, prov2), _ ->
      invalid_arg
        [%string
          "Tnode.update_memory_mode: update %{provenance_to_string prov2} -> \
           %{provenance_to_string provenance} inconsistent for %{name}"]

let update_memory_mode tn mode provenance =
  tn.memory_mode_intent <-
    Some (transition_memory_mode ~debug_name:(debug_name tn) tn.memory_mode_intent mode provenance)

let update_prec ?only_if tn prec =
  let do_update =
    match only_if with
    | None -> true
    | Some cond -> (
        match tn.delayed_prec_unsafe with
        | Specified old_prec -> cond old_prec
        | Default old_prec -> cond old_prec
        | Inferred old_prec when Lazy.is_val old_prec -> cond @@ Lazy.force old_prec
        | _ -> true)
  in
  if do_update then
    if Lazy.is_val tn.storage_prec then (
      if not @@ Ops.equal_prec (Lazy.force tn.storage_prec) prec then
        raise
        @@ Utils.User_error
             (String.concat
                [
                  "Tnode.update_prec: setting precision ";
                  Ops.prec_string prec;
                  " for ";
                  debug_name tn;
                  " but the settled precision is ";
                  Ops.prec_string (Lazy.force tn.storage_prec);
                ]))
    else
      match (tn.delayed_prec_unsafe, only_if) with
      | Specified old_prec, _ when not @@ Ops.equal_prec old_prec prec ->
          raise
          @@ Utils.User_error
               (String.concat
                  [
                    "Tnode.update_prec: setting precision ";
                    Ops.prec_string prec;
                    " for ";
                    debug_name tn;
                    ", but the precision is already set to ";
                    Ops.prec_string (Lazy.force tn.storage_prec);
                  ])
      | Inferred old_prec, Some cond ->
          tn.delayed_prec_unsafe <-
            Inferred
              (lazy
                (let old = Lazy.force old_prec in
                 if cond old then prec else old))
      | Default old_prec, Some cond ->
          tn.delayed_prec_unsafe <- (if cond old_prec then Specified prec else Default old_prec)
      | _ -> tn.delayed_prec_unsafe <- Specified prec

let update_infer_prec ?only_if tn delayed_prec =
  let do_update =
    match only_if with
    | None -> true
    | Some cond -> (
        match tn.delayed_prec_unsafe with
        | Specified old_prec -> cond old_prec
        | Default old_prec -> cond old_prec
        | Inferred old_prec when Lazy.is_val old_prec -> cond @@ Lazy.force old_prec
        | _ -> true)
  in
  if do_update then
    if Lazy.is_val tn.storage_prec then
      raise
      @@ Utils.User_error
           (String.concat
              [
                "Tnode.update_infer_prec: cannot update precision for ";
                debug_name tn;
                " because it has already been forced";
              ])
    else
      match (tn.delayed_prec_unsafe, only_if) with
      | Specified _, _ -> () (* User-specified precision has higher priority *)
      | Default old_prec, Some cond ->
          tn.delayed_prec_unsafe <-
            (if cond old_prec then Inferred delayed_prec else Default old_prec)
      | Default _, None -> tn.delayed_prec_unsafe <- Inferred delayed_prec
      | Inferred old_prec, Some cond ->
          tn.delayed_prec_unsafe <-
            Inferred
              (lazy
                (let old = Lazy.force old_prec in
                 if cond old then Ops.promote_prec old (Lazy.force delayed_prec) else old))
      | Inferred old_prec, None ->
          tn.delayed_prec_unsafe <-
            Inferred (lazy (Ops.promote_prec (Lazy.force old_prec) (Lazy.force delayed_prec)))

let get_specified_prec tn =
  match tn.delayed_prec_unsafe with Specified prec -> Some prec | _ -> None

(** {2 Value-bounds lifecycle (see {!bounds_state})} *)

(** Joins [iv] into the node's candidate bounds. Post-settlement, instead validates that [iv] fits
    the settled interval and raises {!Utils.User_error} otherwise -- a wider write after a reader
    already folded a guard against the bounds would leave compiled code unsound. [what] names the
    proposing write for the error message. *)
let propose_bounds ~what tn iv =
  match tn.bounds with
  | Bounds_unknown -> tn.bounds <- Bounds_proposed iv
  | Bounds_proposed old -> tn.bounds <- Bounds_proposed (Interval.join old iv)
  | Bounds_settled s ->
      if not (Interval.is_within ~outer:s iv) then
        raise
        @@ Utils.User_error
             (String.concat
                [
                  "Tnode.propose_bounds: ";
                  what;
                  " of ";
                  debug_name tn;
                  " has value bounds ";
                  Sexp.to_string_hum (Interval.sexp_of_t iv);
                  " that do not fit the settled bounds ";
                  Sexp.to_string_hum (Interval.sexp_of_t s);
                  " -- already-compiled code discharged an in-range guard against the settled \
                   bounds; write the wider data before compiling readers, or avoid narrowing host \
                   initializations";
                ])

(** Pins the candidate to [Interval.top]: Phase B v1 execution anchoring for compiled device writes.
    Post-settlement (of non-top bounds) this raises like any non-fitting proposal. *)
let pin_bounds_top ~what tn = propose_bounds ~what tn Interval.top

(** The current candidate (or settled) bounds, if any proposal was made. Readers should intersect
    with the precision's machine range themselves. *)
let bounds_candidate tn =
  match tn.bounds with Bounds_unknown -> None | Bounds_proposed iv | Bounds_settled iv -> Some iv

(** Marks the candidate as consumed by a guard fold. Idempotent; raises if nothing was ever proposed
    (folds must only consume existing candidates). *)
let settle_bounds tn =
  match tn.bounds with
  | Bounds_settled _ -> ()
  | Bounds_proposed iv -> tn.bounds <- Bounds_settled iv
  | Bounds_unknown ->
      invalid_arg @@ "Tnode.settle_bounds: no bounds were proposed for " ^ debug_name tn

(** Whether host-upload scans can profit for this precision: integer storage only -- the current
    consumers (gather-guard discharge, index-width selection) act on integer-valued facts, and the
    Phase A folding policy ignores float bounds (binding constraint 8: gate the O(n) scans). *)
let bounds_scan_worthwhile (prec : Ops.prec) =
  match prec with
  | Ops.Byte_prec _ | Ops.Uint16_prec _ | Ops.Int32_prec _ | Ops.Uint32_prec _ | Ops.Int64_prec _
  | Ops.Uint64_prec _ ->
      true
  | Ops.Void_prec | Ops.Uint4x32_prec _ | Ops.Half_prec _ | Ops.Bfloat16_prec _ | Ops.Fp8_prec _
  | Ops.Single_prec _ | Ops.Double_prec _ ->
      false

(* Single pass directly over the bigarray (binding constraint 8), whole buffer including any padding
   margins -- the upload copies the margins too, so the halo fill participates in the node's value
   domain automatically (binding constraint 4). The float view of unsigned storage reinterprets the
   sign bit ([Nd.fold_as_float] reads uint32/uint64 through [Int32.to_float]/[Int64.to_float]), so
   map negative readings back to the unsigned value. Endpoints at or above 2^53 are inexact (strict
   cutoff, binding constraint 6) and nudged outward, covering both the int64-to-float conversion
   error and the unsigned fixup rounding. *)
let scan_host_bounds (prec : Ops.prec) (nd : Nd.t) : Interval.t =
  let fixup =
    match prec with
    | Ops.Uint32_prec _ -> fun v -> if Float.(v < 0.) then v +. 4294967296. else v
    | Ops.Uint64_prec _ -> fun v -> if Float.(v < 0.) then v +. 1.8446744073709552e19 else v
    | _ -> Fn.id
  in
  let lo = ref Float.infinity and hi = ref Float.neg_infinity in
  let integral = ref true and exact = ref true in
  Nd.fold_as_float nd ~init:() ~f:(fun () _idx v ->
      let v = fixup v in
      if Float.(v < !lo) then lo := v;
      if Float.(v > !hi) then hi := v;
      if not (Float.is_integer v) then integral := false;
      if not Float.(abs v < Interval.exact_int_cutoff) then exact := false);
  if Float.(!lo > !hi) then (* no elements *) Interval.top
  else
    let iv = { Interval.lo = !lo; hi = !hi; integral = !integral; exact = !exact } in
    if !exact then iv else Interval.round_out iv

(** Scans a host buffer about to initialize [tn]'s device data and proposes the observed bounds
    (pre-settlement) or validates them (post-settlement). Call on every host-write path
    ([Context.from_host]-mediated writes and link-time [Host_inits] uploads). No-op for float and
    opaque precisions, and when the candidate is already pinned to [top]. *)
let propose_bounds_from_host tn (nd : Nd.t) =
  let prec = Lazy.force tn.storage_prec in
  if bounds_scan_worthwhile prec then
    match tn.bounds with
    | (Bounds_proposed iv | Bounds_settled iv) when Interval.is_top iv ->
        () (* Pinned or settled-top: scanning cannot narrow nor violate anything. *)
    | Bounds_unknown | Bounds_proposed _ | Bounds_settled _ ->
        propose_bounds ~what:"host upload" tn (scan_host_bounds prec nd)

include Comparator.Make (struct
  type nonrec t = t

  let compare = compare
  let sexp_of_t = sexp_of_t
end)

let equal a1 a2 = equal_int a1.uid a2.uid
let hash nd = Int.hash nd.uid
let hash_fold_t acc nd = hash_fold_int acc nd.uid
let hash_t = hash

module Comp = struct
  type nonrec t = t
  type nonrec comparator_witness = comparator_witness
end

(** {2 Per-context placement resolution}

    The context-scoped side of the memory-mode split
    (docs/proposals/context-scoped-memory-modes.md): {!field-memory_mode_intent} on the tnode is
    *declared intent* -- requests made at graph-construction time (user [set_materialized],
    parameter/constant marking, op-support [Never_virtual]) -- while the *decisions* (resolving to
    [Virtual] / [Local] / [On_device]) are recorded here, per compilation lineage. A [Placements]
    table rides [Low_level.optimize_ctx]: it is copied at the start of each backend [compile], so
    sibling compiles from the same context are hermetic (a candidate compile cannot poison another's
    placement resolution), while child contexts inherit the lineage's decisions.

    Lookups fall back to the tnode's intent when the lineage has not yet decided; updates apply the
    same lattice as {!update_memory_mode} but never write the tnode. Intent strengthened after a
    lineage compiled does not invalidate that lineage. *)
module Placements = struct
  open struct
    type tn = t

    let sexp_of_tn = sexp_of_t
  end

  module Key = struct
    type t = tn

    let compare = compare
    let sexp_of_t = sexp_of_tn
    let hash = hash
  end

  type nonrec t = { table : (tn, memory_mode * provenance) Hashtbl.t }

  let sexp_of_t p =
    [%sexp_of: (string * (memory_mode * provenance)) list]
      (List.map (Hashtbl.to_alist p.table) ~f:(fun (tn, d) -> (debug_name tn, d)))

  let create () = { table = Hashtbl.create (module Key) }
  let copy p = { table = Hashtbl.copy p.table }

  (** The effective placement state: the lineage's decision if any, otherwise the tnode's declared
      intent. Side-effect free. *)
  let get p tn =
    match Hashtbl.find p.table tn with Some e -> Some e | None -> tn.memory_mode_intent

  let update p tn mode provenance =
    Hashtbl.set p.table ~key:tn
      ~data:(transition_memory_mode ~debug_name:(debug_name tn) (get p tn) mode provenance)

  (** Kernel-fission escape hatch: strengthen a routine-scoped [Local] decision to [On_device].
      [Local] is only ever a compiler decision (never declared intent), premised on the node's
      lifetime being confined to one kernel launch; when the schedule layer splits a routine into
      multiple kernels at a point where the node's live range crosses, that premise is withdrawn and
      the node must own a context buffer. The {!update} lattice deliberately rejects [Local] ->
      [On_device] (decisions are final within a lineage); this is the one sanctioned override, sound
      exactly because fission runs between optimization and code generation — before any consumer of
      the decision (codegen parameter lists, context allocation) has read it. *)
  let promote_local_to_device p tn provenance =
    match get p tn with
    | Some (On_device, _) -> ()
    | None | Some ((Local | Never_virtual), _) ->
        Hashtbl.set p.table ~key:tn ~data:(On_device, provenance)
    | Some ((Virtual | Effectively_constant), _) ->
        invalid_arg
          ("Tnode.Placements.promote_local_to_device: " ^ debug_name tn
         ^ " is virtual or constant, not routine-scoped scratch")

  (** Fission bookkeeping around {!promote_local_to_device}: the raw lineage entry (no intent
      fallback), captured before a promotion so {!unsafe_restore} can undo it when segment
      coalescing removes the crossing that motivated it. Undoing is sound in that direction only:
      schedules computed under the (stricter) materialized view remain valid for a [Local] node,
      while the converse would miss coverage requirements. *)
  let raw_entry p tn = Hashtbl.find p.table tn

  let unsafe_restore p tn prior =
    match prior with
    | Some entry -> Hashtbl.set p.table ~key:tn ~data:entry
    | None -> Hashtbl.remove p.table tn

  let debug p tn = debug_memory_mode (get p tn)

  (** Mirrors the retired tnode-level [default_to_most_local], resolving into the placements table,
      with two observation guards (docs/proposals/context-scoped-memory-modes.md):

      - An observable node never defaults to [Local]: [Local] is the unobservable class, and unlike
        [Virtual] it cannot be served by recomputation.
      - An observable node still undecided ([None]) at a forcing point materializes instead of
        defaulting to [Virtual]: a node that stayed undecided through optimization has no tracked
        computation (the virtualizer commits every stored candidate at cleanup), so a [Virtual]
        resolution would be unobservable in practice -- there would be nothing to recompute from.
        [Effectively_constant] keeps folding to [Virtual]: observation of constants is served by
        their registered host-init data. *)
  let default_to_most_local p tn provenance =
    let provenance =
      match get p tn with Some (_, prior) -> Refined (prior, provenance) | None -> provenance
    in
    match get p tn with
    | None when is_observable tn -> Hashtbl.set p.table ~key:tn ~data:(On_device, provenance)
    | None | Some (Effectively_constant, _) ->
        Hashtbl.set p.table ~key:tn ~data:(Virtual, provenance)
    | Some (Never_virtual, _) ->
        let mode = if is_observable tn then On_device else most_local_materialized_mode tn in
        Hashtbl.set p.table ~key:tn ~data:(mode, provenance)
    | Some ((Virtual | Local | On_device), _) -> ()

  (** The forcing family -- {!is_virtual_force}, {!is_materialized_force} and {!is_in_context_force}
      -- answers a placement question and, where the placement is still open, resolves the node on
      the spot.

      A [provenance] is recorded exactly where the query WRITES the table, and each query writes on
      its own set of open states:

      - {!is_virtual_force} records on [None] and [Effectively_constant]. [Never_virtual] reaches
        its catch-all [false] arm instead, and the tag is discarded.
      - {!is_materialized_force} records on [Never_virtual] and [Effectively_constant]. [None] is an
        assertion failure, not a defaulting point.
      - {!is_in_context_force} records on [None], [Never_virtual] and [Effectively_constant] --
        unless the node is a slice alias, which answers [false] ahead of the match and records
        nothing whatever its state.

      The settled placements -- [Virtual], [Local], [On_device] -- answer and record nothing, in
      every query. And {!get} answers from the {e effective} state, so a settled placement is either
      this lineage's table decision or the tnode's declared {!field-memory_mode_intent}: a node
      minted [On_device] at construction, set [Virtual] by [Train.set_virtual], or declared [Local]
      outright, silences every later query without ever acquiring a table entry.

      So a query site's tag is the reason the node got defaulted {e at that query}, not a claim that
      the query is load-bearing: most call sites never mint their literal.

      The goldens bear this out. [c_syntax.ml] alone carries twelve query sites, each with its own
      tag; a repository-wide search of the committed [.expected] files finds fourteen distinct tags
      in printed placements, every one minted at graph construction ([tensor.ml], [train.ml]) or by
      the virtualizer in [low_level.ml] -- not one from a codegen query. *)
  let is_virtual_force p tn provenance =
    match get p tn with
    | Some (Virtual, _) -> true
    | None when is_observable tn ->
        (* The undecided-observable guard of {!default_to_most_local}: no tracked computation to
           recompute from, so materialize rather than claim virtual. *)
        Hashtbl.set p.table ~key:tn ~data:(On_device, provenance);
        false
    | None | Some (Effectively_constant, _) ->
        Hashtbl.set p.table ~key:tn ~data:(Virtual, provenance);
        true
    | _ -> false

  let rec is_materialized_force p tn provenance =
    match get p tn with
    | None -> assert false
    | Some ((Virtual | Local), _) -> false
    | Some (On_device, _) -> true
    | Some ((Never_virtual | Effectively_constant), _) ->
        default_to_most_local p tn provenance;
        is_materialized_force p tn provenance

  let rec is_in_context_force p tn provenance =
    (* See {!is_in_context_force} on tnodes: a slice-alias view owns no buffer. *)
    if is_alias tn then false
    else
      match get p tn with
      | Some ((Virtual | Local), _) -> false
      | Some (On_device, _) -> true
      | None | Some ((Effectively_constant | Never_virtual), _) ->
          default_to_most_local p tn provenance;
          is_in_context_force p tn provenance

  (** Pure counterpart of {!is_materialized_force}; see {!Tnode.is_materialized_peek}. *)
  let is_materialized_peek p tn =
    match get p tn with
    | None -> assert false
    | Some ((Virtual | Local), _) -> false
    | Some (On_device, _) -> true
    | Some (Effectively_constant, _) -> false
    | Some (Never_virtual, _) -> (
        is_observable tn
        || match most_local_materialized_mode tn with On_device -> true | _ -> false)

  let known_not_materialized p tn =
    match get p tn with Some ((Virtual | Local), _) -> true | _ -> false

  let known_constant p tn =
    match get p tn with Some (Effectively_constant, _) -> true | _ -> false

  let known_non_virtual p tn =
    match get p tn with None | Some ((Virtual | Effectively_constant), _) -> false | _ -> true

  let known_virtual p tn = match get p tn with Some (Virtual, _) -> true | _ -> false

  let mode_is_unspecified p tn =
    match get p tn with
    | None | Some ((Never_virtual | Effectively_constant), _) -> true
    | _ -> false
end

type t_set = Set.M(Comp).t

let sexp_of_t_set s = [%sexp_of: t Sequence.t] @@ Set.to_sequence s

type 'a t_map = (t, 'a, comparator_witness) Base.Map.t

let sexp_of_t_map sexp_of_v m =
  Sequence.sexp_of_t (fun (k, v) -> sexp_of_list Fn.id [ sexp_of_t k; sexp_of_v v ])
  @@ Map.to_sequence m

let dims_to_string ?(with_axis_numbers = false) arr =
  let dims_s =
    if Lazy.is_val arr.dims then
      let padding = Option.map ~f:fst (Lazy.force arr.padding) in
      Nd.int_dims_to_string ~with_axis_numbers ?padding @@ Lazy.force arr.dims
    else "<not-in-yet>"
  in
  Ops.prec_string (Lazy.force arr.storage_prec) ^ " prec " ^ dims_s

let no_grad_ident_label tn =
  let components = List.filter tn.label ~f:(fun i -> is_alphanum_ i) in
  let digits, components = List.partition_tf components ~f:(fun i -> Char.is_digit i.[0]) in
  let has_grad, result =
    match components @ digits with
    | [] -> (false, None)
    | [ "grad" ] -> (true, None)
    | "grad" :: components -> (true, Some (String.concat ~sep:"_" components))
    | components -> (false, Some (String.concat ~sep:"_" components))
  in
  ( has_grad,
    Option.find_map result ~f:(fun s -> if Char.is_digit s.[0] then Some ("_" ^ s) else Some s) )

let styled_ident ~repeating_nograd_idents ~repeating_grad_idents style arr =
  let n = id arr in
  match style with
  | `Name_only -> n
  | `Name_and_label ->
      let label = label arr in
      if String.is_empty label then n else [%string "%{n}_%{label}"]
  | `Heuristic_ocannl grad_sep -> (
      let is_grad, ident = no_grad_ident_label arr in
      let opt_grad =
        match (grad_sep, is_grad) with
        | `Dot_grad, true -> ".grad"
        | `Under_grad, true -> "_grad"
        | (`Dot_grad | `Under_grad), _ -> ""
      in
      let n_id = if is_grad then arr.id - 1 else arr.id in
      let np = ident_prefix arr.namespace in
      match ident with
      | Some ident ->
          if Hashtbl.mem (if is_grad then repeating_grad_idents else repeating_nograd_idents) ident
          then [%string "%{np}%{n_id#Int}_%{ident}%{opt_grad}"]
          else [%string "%{ident}%{opt_grad}"]
      | None when is_grad -> [%string "%{np}%{n_id#Int}%{opt_grad}"]
      | None -> n)

let update_code_name tn ident =
  match tn.code_name with
  | None -> tn.code_name <- Some ident
  | Some old_name ->
      if
        String.length ident > String.length old_name
        && not (String.is_prefix ~prefix:(id tn) old_name)
        || String.is_prefix ~prefix:(id tn) ident
      then tn.code_name <- Some ident

let get_style ?(arg_name = "ll_ident_style") ?(no_dots = false) () =
  match Utils.get_global_arg ~arg_name ~default:"heuristic" with
  | "heuristic" -> `Heuristic_ocannl (if no_dots then `Under_grad else `Dot_grad)
  | "name_and_label" -> `Name_and_label
  | "name_only" -> `Name_only
  | _ ->
      invalid_arg @@ "Wrong " ^ arg_name ^ ", must be one of: heuristic, name_and_label, name_only"

let header tn =
  let debug = Utils.settings.log_level > 0 in
  let mem_size =
    if Lazy.is_val tn.size_in_bytes then Int.to_string_hum @@ Lazy.force tn.size_in_bytes
    else "<not-in-yet>"
  in
  let repeating_nograd_idents = Hashtbl.create ~size:1 (module String) in
  let repeating_grad_idents = Hashtbl.create ~size:1 (module String) in
  [%string
    {|%{id tn} %{label tn} as %{
      styled_ident ~repeating_nograd_idents ~repeating_grad_idents (`Heuristic_ocannl `Dot_grad) tn
    }: %{debug_memory_mode tn.memory_mode_intent}; %{dims_to_string tn}; mem in bytes: %{mem_size}%{
    if debug then "; debug: " ^ Sexp.to_string_hum tn.backend_info else ""}|}]

module Registry = Stdlib.Weak.Make (struct
  type nonrec t = t

  (* Deliberately keyed by the presentational ([namespace], [id]) pair, not [uid]: [find ~id]
     queries with a mock tnode carrying only the target pair (e.g. to resolve "n42" from debug
     output back to a live node). Note the id-reuse caveat on {!find}. *)
  let equal a1 a2 = equal_int a1.id a2.id && equal_string a1.namespace a2.namespace
  let hash nd = Int.hash nd.id lxor String.hash nd.namespace
end)

let registry = Registry.create 16

let prec_of_dalayed tn =
  match tn.delayed_prec_unsafe with Default prec | Specified prec | Inferred (lazy prec) -> prec

(* docs/proposals/signed-index-precision.md: index arithmetic is signed int32 unless [large_models]
   selects int64. Int32 overflow is excluded by contract, not by widening: every index intermediate
   is bounded by some node's extent by projection construction (axis indices by their dims, conv
   affine forms by the padded input dim, flat offsets by numel), so validating the padded element
   count once per node -- here, where dims are forced -- makes every consumer inherit the guarantee.
   Violation is a hard error naming the node: auto-setting the global flag would be inconsistent
   across already-compiled routines. *)
let validate_padded_numel_contract ~id ~label (dims : int array) =
  if not Utils.settings.large_models then
    let n = Array.fold dims ~init:1 ~f:( * ) in
    if n > 2147483647 then
      raise
      @@ Utils.User_error
           [%string
             "Tensor node %{get_debug_name ~id ~label ()}: padded element count %{n#Int} exceeds \
              the int32 index range; set large_models=true to use 64-bit indices"]

let create ?namespace delayed_prec ~id ~label ~unpadded_dims ~padding () =
  let namespace =
    match namespace with
    | Some ns ->
        validate_namespace ns;
        ns
    | None -> !current_namespace
  in
  (* Compute padded dimensions: tn.dims stores buffer-inclusive (padded) dimensions *)
  let dims =
    lazy
      (let unpadded = Lazy.force unpadded_dims in
       let padded =
         match Lazy.force padding with
         | None -> unpadded
         | Some (padding_arr, _) ->
             Array.map2_exn unpadded padding_arr ~f:(fun d Ops.{ left; right } -> d + left + right)
       in
       validate_padded_numel_contract ~id ~label padded;
       padded)
  in
  let rec size_in_bytes =
    lazy
      (let n = num_elems tn in
       n * Ops.prec_in_bytes (Lazy.force tn.storage_prec))
  and tn =
    {
      delayed_prec_unsafe = delayed_prec;
      bounds = Bounds_unknown;
      storage_prec = lazy (prec_of_dalayed tn);
      dims;
      padding;
      size_in_bytes;
      id;
      namespace;
      uid = fresh_uid ();
      label;
      memory_mode_intent = None;
      observable = false;
      host_constant = false;
      alias_of = None;
      slice_of = None;
      backend_info = Sexp.List [];
      code_name = None;
    }
  in
  (* Note: if tensor nodes get non-trivial finalizers, remember to either add an is_finalized flag
     that is checked in the find function, or to convert it to a find_exn function that should never
     be called on potentially GCed nodes. *)
  Registry.add registry tn;
  tn

(* [create_from_padded] and [create_with_reshape] return the tensor node together with a lazy host
   buffer holding the node's initialization data. The buffer is no longer stored on the node; the
   caller (e.g. [Tensor.term], [Persistence.load]) records it in the computation's [host_inits] map
   so each [Context.compile] can upload it into its own context. The laziness is preserved so the
   buffer is shaped only after shape inference completes. *)
let create_from_padded ?namespace ~id ~label ~ndarray ~padding () =
  let namespace =
    match namespace with
    | Some ns ->
        validate_namespace ns;
        ns
    | None -> !current_namespace
  in
  let dims_val = Nd.dims ndarray in
  validate_padded_numel_contract ~id ~label dims_val;
  let prec_val = Nd.get_prec ndarray in
  let size_in_bytes = lazy (Nd.size_in_bytes ndarray) in
  let rec tn =
    {
      delayed_prec_unsafe = Specified prec_val;
      bounds = Bounds_unknown;
      storage_prec = lazy (prec_of_dalayed tn);
      dims = lazy dims_val;
      padding = lazy padding;
      size_in_bytes;
      id;
      namespace;
      uid = fresh_uid ();
      label;
      memory_mode_intent = Some (On_device, Site "49:ndarray-backed");
      observable = false;
      host_constant = false;
      alias_of = None;
      slice_of = None;
      backend_info = Sexp.List [];
      code_name = None;
    }
  in
  Registry.add registry tn;
  (tn, lazy ndarray)

let create_with_reshape ~id ~label ~base_ndarray ~unpadded_dims ~padding ~from_padded () =
  let namespace = !current_namespace in
  let debug = "Host array for " ^ get_debug_name ~namespace ~id ~label () in
  let prec_val = Nd.get_prec base_ndarray in
  (* Compute padded dimensions: tn.dims stores buffer-inclusive (padded) dimensions *)
  let dims =
    lazy
      (let unpadded = Lazy.force unpadded_dims in
       let padded =
         match Lazy.force padding with
         | None -> unpadded
         | Some (padding_arr, _) ->
             Array.map2_exn unpadded padding_arr ~f:(fun d Ops.{ left; right } -> d + left + right)
       in
       validate_padded_numel_contract ~id ~label padded;
       padded)
  in
  let rec init_buffer =
    lazy
      (let semantic_dims = Lazy.force unpadded_dims in
       let padded_dims = Lazy.force dims in
       let target_padding = Lazy.force padding in
       match (target_padding, from_padded) with
       | None, _ | _, true ->
           (* Use reshape to conform to inferred dims. [Nd.reshape] rather than a bare
              [Bigarray.reshape]: the view shares the source's bytes, and only the former keeps the
              source -- and hence its host-memory accounting -- alive for the view's lifetime. *)
           (* [fold ~init:1] rather than [reduce_exn]: rank-0 tensors (empty dims) hold 1
              element. *)
           let total_elems = Array.fold padded_dims ~init:1 ~f:( * ) in
           let source_total = Array.fold (Nd.dims base_ndarray) ~init:1 ~f:( * ) in
           if total_elems <> source_total then
             raise
             @@ Utils.User_error
                  [%string
                    "create_with_reshape: target dims %{Nd.int_dims_to_string padded_dims} require \
                     %{total_elems#Int} elements but source has %{source_total#Int} elements"];
           Nd.reshape base_ndarray padded_dims
       | Some _, false ->
           (* Create new bigarray with padding and copy source into non-padding parts. semantic_dims
              are the data area dimensions (without padding). *)
           let target = Nd.create_array ~debug prec_val ~dims:padded_dims ~padding:target_padding in
           let source_dims = Nd.dims base_ndarray in
           (* Check total elements match, allowing shape differences *)
           let source_total = Array.fold source_dims ~init:1 ~f:( * ) in
           let data_total = Array.fold semantic_dims ~init:1 ~f:( * ) in
           if source_total <> data_total then
             invalid_arg
               [%string
                 "create_with_reshape: source has %{source_total#Int} elements but target data \
                  area has %{data_total#Int} elements"];
           (* Use C function for efficient copying *)
           let source = Nd.reshape base_ndarray semantic_dims in
           Nd.copy_with_padding ~source ~target ~padding:(Option.value_exn target_padding |> fst);
           target)
  and size_in_bytes =
    lazy
      (let n = num_elems tn in
       n * Ops.prec_in_bytes (Lazy.force tn.storage_prec))
  and tn =
    {
      delayed_prec_unsafe = Specified prec_val;
      bounds = Bounds_unknown;
      storage_prec = lazy (prec_of_dalayed tn);
      dims;
      padding;
      size_in_bytes;
      id;
      namespace;
      uid = fresh_uid ();
      label;
      memory_mode_intent = Some (On_device, Site "49:ndarray-backed");
      observable = false;
      host_constant = false;
      alias_of = None;
      slice_of = None;
      backend_info = Sexp.List [];
      code_name = None;
    }
  in
  Registry.add registry tn;
  (tn, init_buffer)

let initial_default_prec =
  Ops.prec_of_string (Utils.get_global_arg ~default:"single" ~arg_name:"default_prec")

let find_namespaced =
  let mock =
    {
      storage_prec = lazy initial_default_prec;
      delayed_prec_unsafe = Specified initial_default_prec;
      bounds = Bounds_unknown;
      dims = lazy [||];
      padding = lazy None;
      size_in_bytes = lazy 0;
      id = -1;
      namespace = default_namespace;
      uid = -1;
      label = [];
      memory_mode_intent = None;
      observable = false;
      host_constant = false;
      alias_of = None;
      slice_of = None;
      backend_info = Sexp.List [];
      code_name = None;
    }
  in
  (* Caveat: ([namespace], [id]) pairs are reused across a [Tensor.unsafe_reinitialize] into the
     same namespace, so if a pre-reinitialization node with this pair is still reachable, lookups
     may return it instead of the current session's node. Debug-resolution only; identity-sensitive
     code must hold the tnode itself. *)
  fun ~namespace ~id -> Registry.find_opt registry { mock with id; namespace }

(* Session-internal resolution: the ambient {!current_namespace} is what these callers mean. *)
let find ~id = find_namespaced ~namespace:!current_namespace ~id

(** {2 Accessors}

    Host-side value access ([get_value], [set_value], [get_values], [set_values], [points_1d],
    [points_2d]) now lives in {!Context}: it requires an explicit context and performs an on-demand
    device-to-host transfer. There is no host copy stored on the tensor node. *)

(* gh-560: cleanups to run before an accessibility snapshot ([print_accessible_headers] /
   [log_accessible_headers]). The snapshot's subject is user-code liveness, so caches above this
   module in the dependency order (the [Low_level] analysis cache) register themselves here to be
   dropped first — nodes retained only by a compiler cache must not report as accessible. *)
let before_accessibility_snapshot : (unit -> unit) list ref = ref []

let print_accessible_headers ?(pred = fun _ -> true) () =
  Stdio.printf "Tnode: collecting accessible arrays...%!\n";
  List.iter !before_accessibility_snapshot ~f:(fun f -> f ());
  Stdlib.Gc.full_major ();
  let results =
    Registry.fold
      (fun arr acc -> if pred arr then ((arr.namespace, arr.id), header arr) :: acc else acc)
      registry []
  in
  List.sort results ~compare:(fun ((ns_a, a), _) ((ns_b, b), _) ->
      let c = compare_string ns_a ns_b in
      if c <> 0 then c else compare_int a b)
  |> List.iter ~f:(fun (_, header) -> Stdio.print_endline header);
  Stdio.printf "Tnode: Finished printing headers.%!\n"

let%debug_sexp log_accessible_headers ?(pred = fun _ -> true) () =
  List.iter !before_accessibility_snapshot ~f:(fun f -> f ());
  Stdlib.Gc.full_major ();
  let results =
    Registry.fold
      (fun arr acc -> if pred arr then ((arr.namespace, arr.id), header arr) :: acc else acc)
      registry []
  in
  List.sort results ~compare:(fun ((ns_a, a), _) ((ns_b, b), _) ->
      let c = compare_string ns_a ns_b in
      if c <> 0 then c else compare_int a b)
  |> List.iter ~f:(fun (_, _header) -> [%log _header])

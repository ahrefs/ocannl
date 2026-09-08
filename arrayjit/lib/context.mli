(** Simplified context-based interface for backend operations *)

open Base

module Backends = Backends
(** The backend registry and raw backend entry points. This is the supported path for consumers that
    genuinely need backend-level operations, such as [Parallel] and allocator/backend tests.
    Backend-independent interfaces should name their types through {!Ir} instead. *)

module Cc_backend = Cc_backend
(** The cc backend's own module, re-exported because the library's interface module is this one and
    a test outside [arrayjit] can otherwise not name it. What needs naming is
    {!Cc_backend.compiler_command}: the generated-kernel census
    ([test/operations/cc_march_census.ml], gh-ocannl-650) compiles emitted sources under foreign
    [-march] flags, and it must do so with the SAME toolchain that builds them for real -- a census
    against a different compiler would describe guarded arms nothing here selects. The interface is
    [cc_backend.mli]'s, so this exposes no more than that file already publishes. *)

module Builtins_cc = Builtins_cc
(** The cc backend's builtins table, re-exported for the same reason as {!Cc_backend} above: it is
    the enumeration of the whole-vector storage bridges [C_syntax.vec_bridge] renders calls to
    ([OCANNL_VEC_WIDEN_BFLOAT16] and its siblings), and the generated-kernel census
    ([test/operations/cc_march_census.ml], gh-ocannl-752) claims that every one of them reaches an
    emitted kernel. Deriving that set from the table is what makes the claim fail when a bridge is
    added and the fixture does not grow to cover it; a list of names in the test would instead be a
    copy asserting that it still says what it says. Each entry is a (key, definition, dependency
    keys) triple. *)

type t [@@deriving sexp_of]
(** Execution context managing device, compilation, and buffers *)

type task_handle
(** The routine's executable schedule (its backend task). Abstract on purpose: dispatch goes through
    {!run}, which validates the lineage and launch bindings before running and records the execution
    in the ledger afterwards — a directly runnable task would be a one-field-access bypass of those
    checks. *)

type routine = private {
  context : t;
      (** The context the routine was compiled from, carrying the compilation frontier advanced past
          this routine (see {!section:execution_deps}). *)
  task : task_handle;
  bindings : Ir.Indexing.lowered_bindings;
  name : string;  (** The name of the routine, derived from the backend compilation. *)
  inputs : Set.M(Ir.Tnode).t;
      (** The externally required initialization set — what {!run}'s initialization check requires
          to be initialized: the materialized nodes the routine reads before writing them, if at all
          (read-only nodes included), {i minus} those whose initialization the routine itself
          carries — the computation's embedded nodes, and nodes with registered host initialization
          data, which self-initialize at link time from [Ir.Host_inits] (gh-ocannl-333). Not the
          backend's raw read set: the execution-dependency frontier is built from the unfiltered
          backend inputs before these exclusions. *)
  outputs : Set.M(Ir.Tnode).t;
      (** The materialized nodes the routine writes — what {!run} marks initialized. *)
  routine_id : int;
      (** A unique integer identifying the routine within its root context's lifetime. *)
  execution_deps : Set.M(Int).t;
      (** The routine IDs that must execute before this routine, derived from RAW, WAR, and WAW
          hazards on ordinary tensor-node accesses plus the transfer routine supplying any
          merge-buffer read. An empty set means the routine is independent of all previously
          compiled or transferred routines in its lineage. *)
  mma : Ir.C_syntax.mma_summary;
      (** How this routine's [Tile_mma] statements actually rendered (gh-ocannl-626): the
          {!Ir.C_syntax.mma_census} of this compile, collected by {!compile} itself and summarized
          into {!Ir.C_syntax.tensorization} — [Not_requested] when codegen emitted no [Tile_mma],
          [Scalar_fallback] when every one of them declined to the lane-0 scalar path, [Tensorized]
          when at least one rendered to a tensor-core / SIMD-register-tile emission.

          It is a field of the routine, and not something a caller collects around the compile,
          because a timing labeled "tensorized" that measured the scalar fallback is a false perf
          number, and the census being opt-in made that the default for every new timing harness
          (the defect this closes). Fissioned segments compile inside the same bracket, so their
          kernels are summarized together. *)
  peel : Ir.C_syntax.peel_summary;
      (** What the reduction peel DECIDED while emitting this routine (gh-ocannl-733): per
          accumulating site, whether [Low_level.peel_accum_nest] localized it — with how many levels
          it peeled and which verdict each peeled guard earned — ceded that width to the
          warp-shuffle tree or the SIMD accumulator grid (gh-ocannl-754), or why it did not.

          A field of the routine, beside {!mma} and collected in the same bracket, because the
          emitted form does not determine the decision: a nest whose accumulated cell is free of the
          enclosing index peels BOTH levels under a confined guard, while one whose cell mentions it
          peels the inner level only, under a lane-private guard the cell separates — and the two
          render the same localized kernel. A test classifying emitted code is therefore green over
          either; this field is what lets it say which code path actually ran. *)
  volatility : Ir.C_syntax.volatility_summary;
      (** Which of this routine's serial accumulations carry the Metal compiler-bug workaround
          (gh-ocannl-782/820), in which of its two forms, and how many stayed plain because the
          backend did not request it or the accumulating loop emitted no materialized device read.

          A field of the routine, beside {!mma} and {!peel}, because a performance question about a
          Metal reduction starts by asking how many of its accumulation reads carry the workaround,
          and a residency test needs to know what the compile DECIDED rather than re-deriving it
          from the backend's name. On a backend that requests no workaround the accumulator sites
          are still reported, as {!Ir.C_syntax.Plain_accumulator}, with
          {!Ir.C_syntax.volatility_summary.requested} [= false]. *)
}
(** A compiled computational routine ready for execution. The record is [private]: only {!compile}
    constructs routines — the ledger's identity and dependency tracking rely on that — while every
    field stays readable, so tests can assert on the [inputs]/[outputs] the link actually computed
    instead of re-deriving them (gh-ocannl-590). *)

(** {2 Context creation} *)

val cuda : ?ordinal:int -> unit -> t
(** Create a CUDA context on the backend's device [ordinal] (default 0). *)

val hip : ?ordinal:int -> unit -> t
(** Create an AMD HIP (ROCm) context on the backend's device [ordinal] (default 0). *)

val metal : ?ordinal:int -> unit -> t
(** Create a Metal context on the backend's device [ordinal] (default 0). *)

val cpu : ?threads:int -> unit -> t
(** Create a CPU context. [threads] > 1 selects the [multidev_cc] backend (multiple worker-domain
    CPU devices, for debugging parallel workflows); otherwise the [cc] backend. Kernel-level CPU
    parallelism is automatic either way. *)

val auto : unit -> t
(** Automatically select the best available backend: the [backend] setting if configured, otherwise
    the first of metal, cuda, hip, cc whose device discovery succeeds. Only
    {!Ir.Backend_intf.Backend_unavailable} moves on to the next backend — a driver that is present
    but fails to initialize, an interrupt, or an assertion failure propagates rather than silently
    downgrading the run (gh-ocannl-536). *)

val advances_to_next_backend : exn -> bool
(** The selection policy of {!auto}, exposed so it can be pinned without a device: [true] exactly
    for the failures that mean "this backend is not available on this machine". *)

(** {2 Core operations} *)

val compile :
  ?name:string ->
  ?lowered_transform:(Ir.Low_level.optimized -> Ir.Low_level.optimized list) ->
  ?prelowered:Ir.Low_level.optimized ->
  t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  t * routine
(** Compile assignments into an executable routine. Returns updated context and the compiled
    routine. The returned context carries the updated compilation frontier for dependency tracking;
    the input context is unchanged (see {!section:execution_deps}). [name] names the routine and its
    compilation artifacts (generated source files, kernel functions); if omitted, the name is
    derived from the comp's {!Ir.Assignments.Block_comment} labels via
    {!Ir.Assignments.get_name_exn}, which raises if the comp contains no block comment.
    [lowered_transform] rewrites the optimized lowered code before backend compilation — the seam
    for schedule transforms and for hand-annotating hardware axis types in tests
    (docs/proposals/axis-types-for-loops.md). It returns the routine's kernel segments, so a
    whole-routine transform returns a singleton ([fun o -> [ f o ]]) and a transform that splits the
    routine into several kernels (fission) returns one element per segment; the segments run
    back-to-back on the routine's stream with device-side events at the boundaries, like
    {!Ir.Schedule.maybe_default_schedules}' segments. It must return a non-empty list.

    [prelowered] (gh-ocannl-562) is a test seam: it replaces this compile's lowering of [comp] with
    the given optimized code, which then drives codegen AND the analysis layer (I/O classification,
    liveness planning, the context-buffer delta) alike — so a hand-built {!Ir.Low_level.optimized}
    can be seeded with {!set_values}, run, and read back with {!get_values}, rather than only
    analyzed. Pass [~name] and {!Ir.Assignments.empty_comp} unless the routine's nodes must settle
    against a prior context. The record's [optimize_ctx] becomes the returned context's lineage
    state, so the caller owns its provenance; production code should not use this parameter. *)

val compile_outcome :
  ?name:string ->
  ?lowered_transform:(Ir.Low_level.optimized -> Ir.Low_level.optimized list) ->
  ?prelowered:Ir.Low_level.optimized ->
  provenance:Ir.Schedule_outcome.provenance ->
  ?candidate:string ->
  t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  (t * routine) Ir.Schedule_outcome.outcome
(** Internal containment-aware form of {!compile}. The caller supplies the schedule provenance;
    public user code should continue to use {!compile}. *)

val lowered_for_decisions :
  ?name:string ->
  ?materialized:Ir.Tnode.t list ->
  ?inline:Ir.Tnode.t list ->
  t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Ir.Low_level.optimized
(** The analyze-only entry point behind {!decision_surface} (gh-560), generalized over placement
    decisions (gh-ocannl-514, the placement-space search): the routine's optimized lowering as
    {!compile} would produce it from this context with [materialized] decided {!decide_materialized}
    and [inline] preferred {!decide_inline} — computed by lowering and optimization alone: no
    backend codegen, no linking, and no effect on the context (the lineage state is forked like a
    compile's, the decisions are recorded in the fork, and the compilation frontier and ledger are
    untouched). Sibling calls for the same routine share the underlying analysis via the analysis
    cache, so each costs only the specialization replay. Used by [Autotune.placement_surface] to
    read the all-materialized specialization of the decision surface, the form
    {!Ir.Cost_model.completion_floor} bounds partial placement vectors on. *)

val decision_surface :
  ?name:string ->
  t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Ir.Low_level.flip_candidate list
(** gh-560: the analyze-only entry point — the routine's searchable inlining decision dimensions
    ({!Ir.Low_level.field-flip_candidates}, most expensive first) as {!compile} would report them
    from this context, computed by lowering and optimization alone: no backend codegen, no linking,
    and no effect on the context (the lineage state is forked like a compile's, and the compilation
    frontier and ledger are untouched). Sibling compiles of the same routine share the underlying
    analysis via the analysis cache, so after a compile this costs only the specialization replay.
    Used by [Train.tune_placements]' flip refinement to read the decision surface. *)

val run : t -> routine -> t
(** Execute a compiled routine. Mutates buffers in-place. Returns updated context with newly
    initialized nodes tracked. Raises [Failure] if execution dependencies are not satisfied.

    The routine's static-index bindings are read HERE, not when the device gets to the work: the
    dispatch carries the values they hold at this call, so a caller may rebind them for the next run
    the moment this returns, without syncing, on every backend. That is what makes the bind/run/
    rebind loop — {!Train.sequential_loop}, and every training loop — mean the same thing under an
    asynchronous scheduler as under a synchronous one. Execution itself may still be asynchronous;
    {!sync}, or a host read, is what waits for it. *)

val check_runnable : t -> routine -> unit
(** {!run}'s pre-dispatch validation on its own — poisoned lineage, uninitialized inputs,
    unsatisfied execution dependencies, out-of-range bindings — raising exactly what [run] would.
    All of it precedes the dispatch, so a failure here proves nothing was executed and no device
    buffer was written. That is the point (gh-ocannl-550): a caller running a routine inside a
    launch-tagged failure boundary validates through this rather than letting an unattributable
    failure at [Launch] condemn the lineage for a mistake nothing had yet acted on (gh-ocannl-564).
    Equivalent to {!check_lineage_runnable} followed by {!check_launch_bindings}; a caller that must
    treat the two halves differently calls them directly. *)

val check_lineage_runnable : t -> routine -> unit
(** The lineage-wide half of {!check_runnable}: poisoned lineage, uninitialized inputs, unsatisfied
    execution dependencies. Every one is a property of the context and the computation rather than
    of any particular compiled form, so a genuine failure here fails every candidate an autotuner
    could compile in this lineage, and is the caller's to fix and retry.

    A search must therefore let this one out (gh-ocannl-569). Contained as a per-candidate decline
    it is silent: on a backend whose serial baseline is not dispatched — every GPU backend,
    gh-ocannl-532 — every candidate declines for this one reason, nothing is timed, and
    [Train.tune_placements] ships the untuned default out of an unusable lineage under a report that
    says the search completed. *)

val check_launch_bindings : routine -> unit
(** The per-candidate half of {!check_runnable}: bind-time validation of the launch parameters the
    caller just wrote (non-negative, within the declared static range, within the index width).
    Candidates differ in their static ranges, so one can fail this while its siblings time cleanly —
    which is what makes it the half a search contains as a decline (gh-ocannl-564). *)

val sync : t -> unit
(** Blocks until the context's device is idle. Host reads ({!to_host}, {!get_values}) synchronize on
    their own; explicit [sync] is for timing runs (e.g. the autotuner) and for fencing against
    out-of-band observation. *)

val failure_classifier :
  t -> Ir.Schedule_outcome.phase -> exn -> Ir.Schedule_outcome.classified_cause option
(** The backend's own failure classifier, for callers that wrap {!run} / {!sync} in
    {!Ir.Schedule_outcome.protect}. {!compile_outcome} passes it in itself; launch and sync are
    raising APIs, so the autotuner obtains it here. Passing a classifier that always answers [None]
    (as the timing loop did before gh-ocannl-536) makes every launch failure fatal by phase default,
    with no way for a backend to declare one its candidate's fault. *)

val rollback_execution : t -> int -> unit
(** Undoes {!run}'s execution marking for the given routine id. {!run} marks a routine executed
    before the later {!sync} can report an asynchronous failure, so a contained launch/sync
    rejection has to withdraw that claim — otherwise the next routine compiled in this lineage waits
    on a dependency that never completed. Only sound when the failure is known not to have written
    device buffers; otherwise use {!poison_lineage}. *)

val poison_lineage : t -> routine_name:string -> exn -> unit
(** Marks this execution lineage unusable because a failure may have left device buffers partially
    written. Every subsequent {!run}, {!sync}, {!to_host} and {!from_host} on any context sharing
    the lineage raises, naming the routine and the original failure. There is deliberately no
    restore: recovering would mean rebuilding inputs and parameters, which the current
    [timing_ctx]-shaped API cannot express (gh-ocannl-536). *)

val poisoned_failure : t -> exn option
(** The failure that poisoned this execution lineage, if any — the exception every entrypoint on it
    now raises. Lets a caller that would otherwise start fresh work on the lineage see that it
    cannot run: [Train.tune_placements] checks it before searching the second placement arm, since
    the arms share a lineage and a poisoned one refuses every timing run (gh-ocannl-550). *)

val static_properties : t -> Sexp.t
(** The backend's own dump of the properties of all its devices — names, ordinals, and the queried
    attributes each backend chooses to surface, including the launch-dimension limits
    {!hardware_limits} derives from. Not redundant with {!hardware_limits}: what a gate compares
    against may be a device query on one backend and an architectural constant on another, so the
    raw props do not tell you what the gate uses and the derived limits do not tell you whether the
    underlying query answered. Printed together by [bin/device_props.ml] (gh-ocannl-684).

    One [Sexp.message]-shaped [device] entry per device under a group atom, the same shape on every
    backend: see {!Ir.Backend_intf.parse_static_properties}, which states the contract and is the
    single reader of it (gh-ocannl-710). *)

val hardware_limits : t -> Ir.Backend_intf.hardware_limits
(** The backend's conservative per-workgroup device limits (all-[None] on backends that do not bind
    hardware axes). Chiefly for schedule transforms and the autotuner. *)

val codegen_capabilities : t -> Ir.Backend_intf.codegen_capabilities
(** Stable facts from the selected backend's C-syntax configuration. Use the compiled routine's
    censuses for decisions that depend on a particular lowering or rendering. *)

(** {2 Execution dependency tracking}

    Execution dependencies mirror compilation dependencies: they record which routines must execute
    before which others based on tensor-node read/write hazards (RAW, WAR, WAW). A merge-buffer
    input depends on the transfer routine that filled the transient merge slab, not on a writer of
    the associated node's ordinary context buffer; the slab is not an ordinary input requiring
    initialization.

    Dependencies are scoped to compilation lineage: two routines compiled from the {i same}
    [Context.t] are independent siblings, even if they access the same nodes. Only routines compiled
    from the {i returned} (child) context of a prior [compile] call can depend on that prior
    routine. This matches how [compile] advances backend state only in the returned context.

    The per-routine data — [routine_id], [name], [execution_deps], [inputs], [outputs] — is read
    directly off the {!routine} record. *)

val can_run : t -> routine -> bool
(** Whether all execution dependencies of the routine have been satisfied (i.e., all prerequisite
    routine IDs have been executed). *)

(** {2 Data operations} *)

(** Note: These operations work with backend-specific buffer types hidden behind the context
    abstraction. *)

val copy : ?into_merge_buffer:Ir.Backend_intf.merge_buffer_use -> src:t -> dst:t -> Ir.Tnode.t -> t
(** Copies the node's device buffer from [src] into [dst] (default [~into_merge_buffer:No]), or into
    [dst]'s stream's merge buffer for [~into_merge_buffer:Copy], returning the updated destination
    context. When both contexts come from the same backend the copy stays on-device via the
    backend's [device_to_device] transfer machinery (for [Copy], the source node's latest compiled
    writer must have an effective executed or externally written value, and the returned context
    carries both the merge-buffer node and the completed transfer's execution-ledger entry against
    which the next [compile] of merge-consuming code is verified). A later [Copy] refuses while a
    consumer of the current transfer is pending; after those consumers execute, replacing the slab
    invalidates their old transfer dependency so they cannot silently re-run against new bytes. A
    cross-backend copy falls back to a host round-trip ([Copy] raises). Nodes absent from [src]'s
    device buffers fall back to the host round-trip as well, serving host-init literals and
    for-print proxies. *)

(** {2 On-demand host access}

    After [gh-ocannl-333] no tensor data is stored on the host side of a tensor node. All CPU-side
    value access is an {b on-demand, context-mediated} device-to-host (or host-to-device) transfer
    through a temporary host buffer. There is no cache: every call performs a fresh transfer, which
    is {b expensive on non-unified-memory backends} — prefer batching over polling.

    Which nodes are observable is determined by the compilation lineage's placement resolution
    ({!placements}; the tnode's {!Ir.Tnode.field-memory_mode_intent} only records declared intent):
    - [On_device] (materialized) nodes have a context buffer; {!to_host}/{!get_values} read it
      directly.
    - [Virtual] nodes have no buffer anywhere, but they remain observable: their defining
      computation is tracked, so their value can be recomputed on demand — [Train.printf] does this
      via the for-print proxy mechanism ({!register_for_print}); raw {!to_host} on them raises
      unless a proxy or host-init data exists. Observability is inductive: it holds only when every
      node the tracked computation reads is itself observable — a [Virtual] node depending (even
      transitively through other [Virtual] nodes) on a [Local] node inherits its unobservability.
    - [Local] nodes are routine-scoped scratch and {b unobservable}: their computation is not
      tracked, and they are stored (to whatever degree the optimizer decides on) only within a
      single routine invocation. This is a deliberate opt-out from the observability guarantee that
      licenses backend optimizations (e.g. stack allocation). The mode is only ever assigned by the
      compiler, to nodes never read outside their defining routine; to prevent it, request
      materialization (e.g. [Train.set_materialized]) before the first routine using the node is
      compiled. Host access to a node this lineage placed [Local] {b raises}
      ({!Ir.Utils.User_error}) in both directions (gh-ocannl-599): the routines never read the
      buffer a host write would allocate, and never write one for a host read, so a permitted
      transfer would report the uploaded copy back as if it were a computed value. A for-print proxy
      is still read, since it is a separate materialized node. *)

val mem : t -> Ir.Tnode.t -> bool
(** Whether the node has a device buffer allocated in this context. *)

val register_for_print : src:Ir.Tnode.t -> proxy:Ir.Tnode.t -> unit
(** Registers [proxy] as a for-print copy of [src] (gh-ocannl-333 AC 5): when [src] is not present
    in a context, {!to_host}/{!get_values} on [src] read through [proxy] instead. Used by
    [Train.printf] to render the value of a tensor that is not directly materialized in the context.
*)

val to_host : t -> Ir.Tnode.t -> Ir.Ndarray.t
(** Transfers the node's device buffer into a fresh host [Ndarray] and returns it. Raises if the
    node is not present in the context (and has no host-init data or for-print proxy), and if this
    lineage placed the node [Local] -- routine-scoped scratch, whose buffer no routine writes (see
    the on-demand host access section above). *)

val from_host : t -> Ir.Tnode.t -> Ir.Ndarray.t -> t
(** Uploads the host buffer into the node's device buffer (allocating it if needed) and returns a
    context in which the node is marked initialized and this explicit write supersedes any compiled
    reader/writer frontier entry for the node. Raises if this lineage placed the node [Local]: no
    routine reads the uploaded buffer. *)

val get_values : t -> Ir.Tnode.t -> float array
(** Retrieves all (unpadded) values of the node via an on-demand device-to-host transfer. *)

val set_values : t -> Ir.Tnode.t -> float array -> t
(** Sets all (unpadded) values of the node via an on-demand host-to-device transfer, returning a
    context in which the node is marked initialized. *)

val get_value : t -> Ir.Tnode.t -> int array -> float
(** Retrieves a single value at the given index via an on-demand device-to-host transfer. *)

val set_value : t -> Ir.Tnode.t -> int array -> float -> t
(** Sets a single value at the given index, preserving the other elements. Returns a context in
    which the node is marked initialized. *)

val points_1d : ?from_axis:int -> xdim:int -> t -> Ir.Tnode.t -> float array
(** Like {!get_values} but extracts a 1d slice of points for plotting. *)

val points_2d : ?from_axis:int -> xdim:int -> ydim:int -> t -> Ir.Tnode.t -> (float * float) array
(** Like {!get_values} but extracts a 2d slice of points for plotting. *)

(** {2 Node tracking operations} *)

val is_initialized : t -> Ir.Tnode.t -> bool
(** Check if a node is initialized. *)

(** {2 Debug operations} *)

val backend_name : t -> string
(** Get the name of the backend. *)

val ordinal : t -> int
(** The backend's device ordinal this context runs on -- what {!cuda}/{!hip}/{!metal} was given.
    This is NOT {!Ir.Backend_intf.device.device_id}, which counts device instances process-globally
    across all backends; two contexts on different backends share ordinal 0. *)

val get_used_memory : t -> int
(** (An upper bound of) the memory used for arrays on the context's device, in bytes. Device-wide:
    covers all contexts sharing the device. Useful for asserting the footprint effect of the
    liveness memory planner (config [buffer_aliasing], gh-ocannl-489). *)

val release : t -> unit
(** Eagerly frees the device buffers this context owns — the pools holding nodes it allocated that
    its parent does not have, and that are not per-device constants. Idempotent (a second call is a
    no-op), and safe to call on a context derived from a still-live parent: sibling contexts never
    share a working pool, since each [compile] mints its own pool ids.

    {b Precondition, not checked}: the context must have no live descendants. A context compiled
    {e from} this one inherits its buffer locations while keeping it as their backend parent, so
    releasing an ancestor leaves the descendant resolving a dropped pool id (or reading a freed
    pointer). Release leaves, not interior nodes. This is the pre-existing contract of the
    underlying {!Backends.finalize} — what is new is that it is reachable from here — and it is
    deliberately left as a precondition rather than enforced: tracking live descendants would mean
    refcounting persistent context values, whose whole point is that a child can be derived at any
    time and outlive the expression that made it. The one caller in-tree ({!Autotune.tune}) releases
    only leaf siblings of one search context, which is the shape this is for.

    A context whose buffers were released must not be run or read again; it is a dead handle,
    exactly as after the finalizer had reclaimed it. Nothing in the context is invalidated for
    {e reading metadata} (placements, names, the ledger), only for touching buffers.

    What it does {e not} free: per-device constants
    ({!Ir.Backend_intf.field-device.constant_buffer_cache} entries), which are shared across
    contexts by design and outlive them. That is a real bound on what this can do for a schedule
    search — a hoisted [Stage] candidate mints a fresh packed constant per application, so those
    accumulate whatever the caller does here (gh-ocannl-565).

    This exists because releasing is otherwise not something a finalizer can be relied on to do
    (gh-ocannl-550): the backends' pool tables hold a strong reference to every slab they allocated,
    so device memory is invisible to the OCaml GC and a process under no host-heap pressure never
    reclaims it. Callers that know an exact lifetime should say so — [Autotune.tune] does, per
    candidate. Everyone else can keep ignoring this: nothing regresses by not calling it, the
    process just holds more device memory. *)

val placements : t -> Ir.Tnode.Placements.t
(** The context's compilation lineage's memory-mode resolution
    (docs/proposals/context-scoped-memory-modes.md): which nodes this lineage decided to inline
    ([Virtual]), keep as routine-scoped scratch ([Local]), or give a device buffer ([On_device]).
    Reads are side-effect free; chiefly for tests and diagnostics. *)

val decide_materialized : t -> Ir.Tnode.t list -> t
(** A child context whose compilation lineage additionally decides [On_device] placement for the
    given nodes: subsequent compiles from the returned context materialize them. This is the
    functional, context-scoped counterpart of strengthening tnode-level intent
    ([Train.set_materialized]) — the nodes' declared intent is untouched and the argument context
    (with its other descendants) is unaffected, so a default-placement sibling and a materialize-all
    sibling can coexist (e.g. the placement-A/B arms of [Train.tune_placements]). Nodes the lineage
    or intent already constrains away from plain materialization ([Virtual], [Local], or constant)
    are skipped. *)

val decide_inline : t -> Ir.Tnode.t list -> t
(** A child context whose compilation lineage additionally prefers the given nodes inline (gh-555):
    subsequent compiles exempt them from the heuristic virtualization caps ([virtualize_max_visits],
    [virtualize_max_inline_reduction], [virtualize_max_inline_fanin]) — the caps are priors of the
    default placement policy, not legality. Legality still applies: a preferred node the virtualizer
    rejects (escaping symbols, non-injective producer indices, opaque effects, ...) materializes as
    before, and the observability pessimizations (read-only, read-before-write) are unaffected. The
    preference steers placements the lineage has {e not yet decided}: a node already materialized by
    an earlier compile in this lineage keeps that decision (decisions are final within a lineage —
    compiled routines depend on them), so apply the preference to a pre-compile sibling of the
    routines that set the node, as [Train.tune_placements] does. Together with
    {!decide_materialized} this spans the per-node inlining decision vector: [Inline] here,
    [Materialize] there, the default heuristics elsewhere. Hermetic like {!decide_materialized}: the
    argument context and its other descendants are unaffected. *)

(** {1 A collection of the execution backends} *)

open Base
module Schedulers = Schedulers

val plan_pool_segments :
  cap:int ->
  what:string ->
  debug_name:(int -> string) ->
  (int * int) list ->
  (int * int) list * int list
(** gh-ocannl-344 pool-allocator planner. Lays out [(size, alignment)] allocations (in order) into
    pools so no pool's bumped extent exceeds [cap] bytes (the uint32 4 GB per-pool ceiling when
    [large_models = false]). Returns each item's [(segment_index, byte_offset)] and the byte size of
    each segment. Raises {!Ir.Utils.User_error} (naming [what] and [debug_name i]) when a single
    item exceeds [cap]. Exposed for unit testing the segmenting/cap behavior with synthetic sizes.
*)

val plan_arena_offsets :
  cap:int -> (int * int * string * (int * int) option) list -> (int list * int) option
(** gh-ocannl-489 liveness-aware pool layout: lays out
    [(size, alignment, precision_class, live_span)] allocations into ONE pool where two allocations
    may overlap iff both carry a live span, the spans are disjoint (closed intervals) and the
    precision classes are equal. A [None] span means always-live (conflicts with everything). Greedy
    by decreasing size, deterministic. Returns per-item byte offsets (in input order) and the pool's
    total size, or [None] when the layout exceeds [cap] (callers fall back to
    {!plan_pool_segments}). Exposed for unit testing the coloring with synthetic sizes. *)

val score_footprint :
  backend_name:string ->
  limits:Ir.Backend_intf.hardware_limits ->
  static_indices:Ir.Indexing.static_symbol list ->
  Ir.Low_level.optimized ->
  Ir.Low_level.footprint
(** gh-ocannl-498: score the peak footprint of a lowered routine under its own placements, with the
    same machinery the allocator uses — the default schedule and fission, the gh-ocannl-489 live
    spans, and {!plan_arena_offsets} over the working group. This is the cost side
    {!Ir.Low_level.field-flip_candidates} does not carry: the recompute-cost bound says what
    inlining a node costs, this says what it saves. Meaningful only with config [buffer_aliasing]
    on; without it every node is always-live and the score degenerates to bump packing.

    Scored over the routine's whole in-context node set rather than a context's allocation delta, so
    the number depends only on the code and the placements — the precondition for a deterministic
    selector ([Memory_budget.fit]). It is therefore a {e model} of the peak, not a prediction of
    {!Context.get_used_memory}: the real allocator skips nodes a prior context already holds, and
    the driver page-rounds pool bases. Enumeration is canonical (by {!Ir.Tnode.uid}) so the greedy
    coloring is reproducible across processes. *)

val with_segments_census : (unit -> 'a) -> 'a * Ir.Low_level.optimized list
(** gh-ocannl-1006: runs [f] and returns, beside its result, the kernel segments the backend compile
    inside it shipped — one element per kernel, in launch order, exactly what the compile's
    [lowered_transform] (or, without one, {!Ir.Schedule.maybe_default_schedules}) returned; [[]]
    when [f] reached no backend compile. Brackets like {!Ir.C_syntax.with_census}: the previous
    reading is restored on exit, exception or not. [Context.compile_outcome] is the one caller, and
    stores the result on the routine as its [segments]. *)

val finalize :
  'dev 'runner 'event.
  (module Ir.Backend_intf.Backend
     with type dev = 'dev
      and type event = 'event
      and type runner = 'runner) ->
  ('dev, 'runner, 'event) Ir.Backend_intf.context ->
  unit
(** Frees the pools that are specific to the context -- not contained in the parent context, and not
    per-device constants. Idempotent (guarded by the context's [finalized] flag), and freeing is
    per-[pool_id] rather than per-node: one pool holds several nodes. If cleanup raises after a
    strict prefix of pools, retry skips that prefix and frees only the remainder.

    This is NOT optional if the memory matters (gh-ocannl-550, correcting the note that used to
    stand here). A pool's backend buffer is never garbage-collected, because the backend's private
    [(device_id, pool_id) -> base] table is a module-level strong root holding every slab it
    allocated; dropping that entry -- which only this function does -- is what makes the buffer
    collectable at all (and eagerly frees it where the backend has a raw deallocator). Measured: a
    schedule search accumulated one unfreed pool per candidate to 12 GB while the OCaml GC ran
    normally throughout, unloading code modules (which sit behind no such table) as it went.

    {!Context.release} is the front-end entry point; prefer it over calling this directly. *)

val lower_assignments :
  Ir.Low_level.optimize_ctx ->
  ?name:string ->
  'a Ir.Indexing.bindings ->
  Ir.Assignments.t ->
  string * Ir.Low_level.optimized
(** The shared lowering front half of backend [compile]: forks the lineage state
    ([Low_level.copy_optimize_ctx], so the compile's decisions stay hermetic), derives the routine
    name, wires the debug-file callbacks, and runs [Assignments.lower]. Exposed for analyze-only
    consumers (gh-560: {!Context.decision_surface}) that read the optimized code's decision surface
    without backend codegen. *)

(** {2 The implemented backends}

    Each backend is instantiated once per process, so its context type is nameable and two
    independently-created contexts on the same backend unify -- the precondition for [Context.copy]
    dispatching to the backend's [device_to_device] via {!wrapped_context}. Instantiation touches no
    driver or hardware: device discovery stays lazy inside the backends (forced at first
    [get_device], where [Context.auto]'s fallback can catch an unusable driver/device per call), and
    on platforms without the corresponding library the dune-[select]ed missing stub is what gets
    instantiated -- harmless at init, raising on use. Backend caches consequently persist across
    [Tensor.unsafe_reinitialize]; that is safe because tnode identity ([Tnode.uid]) is never reused.
*)

module Cc_b : Ir.Backend_intf.Backend
module Multidev_cc_b : Ir.Backend_intf.Backend
module Cuda_b : Ir.Backend_intf.Backend
module Hip_b : Ir.Backend_intf.Backend
module Metal_b : Ir.Backend_intf.Backend

(** The implemented backends. Constructors statically imply the corresponding singleton module ([Cc]
    -> {!Cc_b}, ...). *)
type backend = Cc | Multidev_cc | Cuda | Hip | Metal [@@deriving sexp, equal, enumerate]
(** [enumerate] derives [all_of_backend], every backend OCANNL has, so that a check wanting "the set
    of backends" reads it off the type instead of restating it. The vocabulary of the
    [; ocannl-backend:] dune marker is such a check: [test/operations/marker_backend_vocabulary]
    holds it equal to [List.map all_of_backend ~f:backend_name] plus ["none"], which is what stops
    it agreeing with a list that has stopped being this one (gh-ocannl-689). *)

val get_backend : ?backend_name:string -> unit -> backend
(** The backend corresponding to [backend_name], or if omitted, selected via the global [backend]
    setting. Non-generative: the same backend value (hence the same singleton state) each call. *)

val backend_name : backend -> string
(** Inverse of {!get_backend}'s name parsing. *)

val backend_module : backend -> (module Ir.Backend_intf.Backend)
(** The singleton module as an existentially-packed first-class module, for generic consumers that
    thread a single backend through (the raw-API tests, [Parallel]). Code that must re-correlate two
    contexts later (e.g. [Context.copy]) should use {!wrapped_context} instead: this projection
    erases the type components. *)

(** {2 Contexts wrapped with their backend}

    A closed disjunction over the implemented backends' context types: matching two values on the
    same constructor recovers type equality directly, which is what lets [Context.copy] fall onto
    the backend-specific [device_to_device] when both contexts come from the same backend. Matching
    it by hand is rarely necessary — {!unwrap}, {!pair_contexts}, {!query} and {!with_backend}
    dispatch through the one match each direction needs. *)

type ('dev, 'runner, 'event) backend_module =
  (module Ir.Backend_intf.Backend
     with type dev = 'dev
      and type runner = 'runner
      and type event = 'event)
(** A backend singleton's module at known type components -- the package type the generic
    dispatchers here are written against, and the annotation a [ctx_op]/[ctx_query] closure needs to
    unpack its module argument. *)

type wrapped_context =
  | Cc_ctx of Cc_b.context
  | Multidev_cc_ctx of Multidev_cc_b.context
  | Cuda_ctx of Cuda_b.context
  | Hip_ctx of Hip_b.context
  | Metal_ctx of Metal_b.context

type ('dev, 'runner, 'event) backend_impl = {
  bi_backend : backend;
  bi_module : ('dev, 'runner, 'event) backend_module;
  bi_wrap : ('dev, 'runner, 'event) Ir.Backend_intf.context -> wrapped_context;
}
(** Everything a {!wrapped_context} constructor statically implies: which backend it is, its
    singleton module at that constructor's type components, and the constructor itself (to rebuild
    the wrapper around a derived context). One record per backend, so the correspondence
    [Cc_ctx <-> Cc <-> Cc_b] is written once rather than once per dispatcher. *)

type packed_impl = Packed_impl : ('dev, 'runner, 'event) backend_impl -> packed_impl

(** A wrapped context with its type components recovered as locally abstract types. *)
type unwrapped =
  | Unwrapped :
      ('dev, 'runner, 'event) backend_impl * ('dev, 'runner, 'event) Ir.Backend_intf.context
      -> unwrapped

(** Two wrapped contexts correlated: [Same_backend] recovers the type equality that lets a
    same-backend transfer dispatch to the backend's [device_to_device], while [Cross_backend] is
    every mismatched pair. *)
type paired =
  | Same_backend :
      ('dev, 'runner, 'event) backend_impl
      * ('dev, 'runner, 'event) Ir.Backend_intf.context
      * ('dev, 'runner, 'event) Ir.Backend_intf.context
      -> paired
  | Cross_backend

val impl_of_backend : backend -> packed_impl
(** The {!backend_impl} the backend constructor implies. *)

val unwrap : wrapped_context -> unwrapped
(** The {!backend_impl} and backend context a wrapped context carries. *)

val pair_contexts : wrapped_context -> wrapped_context -> paired
(** Correlate two wrapped contexts, recovering their type equality when they come from the same
    backend. [Context.copy] is the caller: a same-backend pair can dispatch to [device_to_device], a
    cross-backend one must go through a host round-trip. *)

val wrapped_backend : wrapped_context -> backend

val make_context : ?ordinal:int -> backend -> wrapped_context
(** A fresh root context (empty [optimize_ctx]) on the backend's device [ordinal] (default 0).
    Raises when the backend's hardware or library is unavailable. *)

type 'a ctx_op = {
  f :
    'dev 'runner 'event.
    ('dev, 'runner, 'event) backend_module ->
    ('dev, 'runner, 'event) Ir.Backend_intf.context ->
    ('dev, 'runner, 'event) Ir.Backend_intf.context * 'a;
}
(** A context-transforming backend operation, polymorphic over the backend's type components so
    {!with_backend} can rebuild the same {!wrapped_context} constructor around the result. *)

val with_backend : wrapped_context -> 'a ctx_op -> wrapped_context * 'a

type 'a ctx_query = {
  q :
    'dev 'runner 'event.
    ('dev, 'runner, 'event) backend_module -> ('dev, 'runner, 'event) Ir.Backend_intf.context -> 'a;
}
(** A read-only backend operation; like {!ctx_op} but leaves the context untouched. *)

val query : wrapped_context -> 'a ctx_query -> 'a

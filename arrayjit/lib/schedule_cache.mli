(** {1 Process-independent schedule identities and the schedule disk cache}

    Support for persisting and replaying {!Schedule.schedule} values (docs/proposals: the autotune
    companion of schedule-ir-optops.md). Schedules embed {!Indexing.symbol}s and {!Tnode.t}s whose
    identities are process-local (global counters), so a schedule value is only meaningful against
    the one lowering it was built for — and every backend [compile] lowers afresh. This module gives
    both a {e canonical}, structural identity:

    - loop symbols are numbered by the preorder position of their binding [For_loop] in the
      optimized code ([Base]),
    - static-index symbols by their position in the [static_indices] list ([Static]) — they occur
      free in the code, so the traversal cannot discover them and they are pre-seeded,
    - symbols minted by schedule ops themselves (e.g. [Split]'s fresh loops) by the index of the op
      that mints them ([Minted]) — replay re-mints them through the {!Schedule} builders,
    - tensor nodes by first occurrence in the same traversal.

    The same traversal renders the code into a canonical string and digests it. The digest is the
    safety guarantee: a schedule saved against a digest is only ever replayed onto code with an
    equal digest, which makes the canonical numbering total and unambiguous by construction —
    nondeterministic lowering degrades to cache misses, never to a schedule applied to the wrong
    loop.

    Schedule identity pins numerics (gh-ocannl-484): ops holding the reduction-reassociation license
    ([Split_reduce] — whose fixed combine tree is a function of [num_blocks] — [Swap] and
    [Vectorized] retypes over accumulations, [Tensorize]) make the computed values a function of the
    schedule. Replaying a cached schedule reproduces results bitwise; retuning, clearing the cache,
    or a digest change may select a different schedule and change low-order bits of reduction
    results. *)

open Base

(** Which fresh symbol of a schedule op a [Minted] reference names. *)
type mint_role =
  | Split_outer
  | Split_inner
  | Expand_axis of int  (** The [i]-th fresh symbol of an [Expand_zero]. *)
  | Tensorize_lane
  | Partition_seg of int  (** The [i]-th segment symbol of a [Partition]. *)
  | Split_reduce_block
  | Split_reduce_inner
  | Split_reduce_combine of int  (** The [i]-th combine symbol of a [Split_reduce]. *)
[@@deriving sexp, compare, equal]

(** A process-independent name for a symbol occurring in a schedule. [Base i] is the [i]-th
    [For_loop] binder in preorder of the optimized code the schedule applies to; [Static k] the
    [k]-th static index; [Minted (op, role)] the fresh symbol in [role] of the [op]-th (0-based) op
    of the same saved schedule. *)
type sym_ref = Base of int | Static of int | Minted of int * mint_role
[@@deriving sexp, compare, equal]

(** {!Schedule.optop} with symbols replaced by {!sym_ref}s and tensor nodes by their canonical
    first-occurrence index. *)
type saved_optop =
  | Split of {
      axis : sym_ref;
      factor : int;
      outer : Low_level.axis_type;
      inner : Low_level.axis_type;
    }
  | Swap of { outer : sym_ref; inner : sym_ref }
  | Retype of { axis : sym_ref; ty : Low_level.axis_type }
  | Unroll of { axis : sym_ref; materialize : bool }
  | Partition of { axis : sym_ref; breakpoints : int list }
  | Pad of { axis : sym_ref; to_multiple_of : int }
  | Stage of {
      source : int;
      tile_loops : sym_ref list;
      shared : bool;
      cooperative : int option;
      hoisted : bool;
      swizzle : Low_level.swizzle_kind option; [@sexp.option]
          (** Serialized only when set ([@sexp.option]), so pre-swizzle cache files parse. *)
      pad_stride : int option; [@sexp.option]
          (** Likewise omitted when unset, so pre-gh-481 cache files parse. *)
      pipeline_depth : int option; [@sexp.option]
          (** [None] encodes depth 1 and is omitted, so pre-gh-487 cache files parse. *)
      tile_prec : Ops.prec option; [@sexp.option]
          (** The staged tile's storage precision override (gh-ocannl-575); omitted when unset, so
              pre-gh-575 cache files parse. *)
    }
  | Privatize of { target : int; over : sym_ref }
  | Expand_zero of { tn : int }
  | Tensorize of {
      i : sym_ref;
      j : sym_ref;
      k : sym_ref;
      simd_width : int;
      tile : Register_tile.t option; [@sexp.option]
          (** The requested C-tile geometry (gh-ocannl-619); omitted when the renderer chooses, so
              pre-geometry entries stay readable without an [entry_version] bump. *)
    }
  | Fuse_epilogue of { target : int; shared : bool }
  | Split_reduce of { axis : sym_ref; target : int; num_blocks : int }
[@@deriving sexp, compare, equal]

type saved_schedule = saved_optop list [@@deriving sexp, compare, equal]

type canonical
(** The canonical identity of one [Low_level.optimized] value: the digest, the loop-binder and
    static-symbol numbering, and the tensor-node numbering. *)

val canonicalize :
  ?static_indices:Indexing.static_symbol list ->
  ?with_placements:bool ->
  Low_level.optimized ->
  canonical
(** Walks the optimized code once in preorder, numbering [For_loop] binders, first-occurrence tensor
    nodes (their dims, precision, hoisted-packing eligibility [Schedule.hoistable_constant] —
    schedule validity depends on operand constancy, gh-ocannl-470 — and effective placement class
    from the compile's {!Ir.Low_level.optimize_ctx} — identical code over [Local] scratch vs an
    [On_device] buffer generates different kernels, so same-code different-placement programs must
    not share cache keys — all enter the digest), and rendering the canonical form.
    [with_placements = false] omits the placement classes, giving the {e structural} identity:
    placement classes can render differently across compilation lineages on byte-identical code, so
    per-segment schedule matching in fissioned replays keys on structure only. The
    binder/tensor-node numbering is identical either way. [static_indices] must be the same list the
    code was lowered with ({!Indexing.bound_symbols} of the compile's bindings). *)

val digest : canonical -> string
(** Hex digest of the canonical rendering. Equal digests mean structurally identical code, hence
    interchangeable canonical numberings. *)

val complete : canonical -> bool
(** [false] when the code contains constructs the canonical rendering cannot capture
    ([Staged_compilation] closures, unbound or shadowed loop symbols). Incomplete canonical forms
    must not be used as {b disk}-cache keys (distinct programs could collide); within one process
    they still support {!to_saved}/{!of_saved} round-trips. *)

val tn_of_ref : canonical -> int -> Tnode.t
(** The tensor node at a canonical index. Raises [Invalid_argument] on out-of-range. *)

(** {2 Symbol resolution registries}

    A [registry] resolves the symbols of a particular compile's code to [sym_ref]s: base and static
    symbols through its {!canonical}, schedule-minted symbols through entries recorded by
    {!to_saved} / {!of_saved}. Use it to translate loops of {e transformed} code (base code with a
    schedule prefix applied) into references a schedule extension can persist. *)

type registry

val base_registry : canonical -> registry
val resolve : registry -> Indexing.symbol -> sym_ref option
val resolve_tn : registry -> Tnode.t -> int option

val to_saved : registry -> Schedule.schedule -> saved_schedule * registry
(** Serializes a schedule built against the registry's compile (e.g. by {!Schedule.default_gpu}),
    recording each op's minted symbols in the returned registry (op indices continue from the number
    of ops already recorded in the input registry, so extensions of replayed prefixes stay
    consistent). Raises [Invalid_argument] when an op references a symbol or tensor node the
    registry cannot resolve. *)

val of_saved : canonical -> saved_schedule -> Schedule.schedule * registry
(** Replays a saved schedule against a (fresh) compile's canonical form: base and static references
    resolve through [canonical], minting ops go through the {!Schedule} builders and their fresh
    symbols are recorded for later references. Raises [Invalid_argument] on dangling references
    (canonical mismatch — always guard with {!digest} equality first). *)

(** {2 The disk cache} *)

val numerics_tag : unit -> string
(** A filename-safe short digest of the current {!Ir.Numerics} policy ({!Ir.Numerics.get}, so it
    tracks {!Ir.Numerics.set_policy}). The policy is not a property of the code — it is consulted at
    codegen and by the autotune tile-shape choice — so it cannot enter {!digest}, yet a schedule
    tuned under one policy must never replay under another (gh-ocannl-568: a default-flags run
    replaying a tf32-tuned tensorized winner measured 5.9x slower than not tuning at all, its mma
    rendering degraded to the scalar fallback). Hence it enters {!cache_key} and {!entry}. *)

val codegen_tag : limits:Backend_intf.hardware_limits -> unit -> string
(** A filename-safe short digest of the codegen environment (gh-ocannl-572): everything consulted
    when a kernel is {e rendered, compiled or dispatched}, which happens after the lowered code that
    {!digest} names — so, exactly like {!numerics_tag}, these are invisible to the digest while
    changing the kernel or what a timing measures, and a winner crowned under one such regime must
    not replay under another. Three layers: the process-wide gates (the index and pool-slot width
    [large_models], [buffer_aliasing]'s [restrict] suppression, and the {e effective}
    routine-logging predicate — which includes the [log_level > 1] threshold, so a verbosity bump
    alone never churns keys — together with the settings that only matter once logging reaches the
    kernel ([prefer_backend_uniformity]'s logging spelling, the stream-log routing); the whole
    [limits] record, which describes the device candidates are generated, rendered and timed
    against; and, inside it, the backend's own {!Ir.Backend_intf.hardware_limits.codegen_tag}. *)

type entry = {
  version : int;
  backend : string;
  numerics : string;
      (** {!numerics_tag} of the policy the search ran under; redundant with the key, which carries
          the same tag, so it is a self-description of the file and a guard for a hand-moved entry.
      *)
  codegen : string option; [@sexp.option]
      (** {!codegen_tag} of the codegen configuration the search ran under (gh-ocannl-572): the same
          self-description as [numerics]. Optional so entries written before this field existed stay
          readable. *)
  objective : string option; [@sexp.option]
      (** The autotuner's timing objective ([autotune_timing]) the search ran under (gh-ocannl-755):
          the same self-description as [numerics] and [codegen], beside the key's own ["timing"]
          component. Optional so entries written before this field existed stay readable — they key
          differently, so nothing looks them up. *)
  source_digest : string;
  saved : saved_schedule;
  segments : (string * saved_schedule) list option; [@sexp.option]
      (** A fissioned winner (docs: per-fission-segment tuning): per-segment schedules keyed by the
          {e pre-schedule} segment's canonical digest — replay routes each of
          {!Schedule.fission_scheduled}'s [`Normal] segments through this association (unmatched
          segments degrade to the empty schedule). [None] for whole-routine schedules. With
          [segments] present, [saved] is empty except for a split-reduce winner (gh-ocannl-484 task
          3), where it holds the whole-routine prelude — resolved against the {e base} canonical
          form and applied before fission, the segment keys then addressing the {e post-prelude}
          segmentation. *)
  finer_fission : bool option; [@sexp.option]
      (** [Some true]: the [segments] keys address {!Schedule.fission_scheduled}'s [arity_cuts]
          (finer) segmentation (gh-ocannl-574); replay must re-segment under the same mode or the
          keys miss wholesale. Omitted when false, so entries stay byte-stable and pre-gh-574
          entries parse without an [entry_version] bump. *)
  best_ms : float;  (** The winning candidate's measured time, for diagnostics. *)
  baseline_ms : float;
      (** The unscheduled baseline's measured time, for diagnostics; [infinity] on GPU backends,
          where the unparallelized baseline is not dispatched (gh-ocannl-532). *)
  default_ms : float option; [@sexp.option]
      (** The untuned default pipeline's measured time from the search that wrote the entry, for
          diagnostics (gh-ocannl-552). [None] when the default seed was not timed, or for entries
          written before this field existed — optional so such entries stay readable without an
          [entry_version] bump. *)
  mma_best_ms : float option; [@sexp.option]
      (** The best TIMED tensorized candidate of the search that wrote the entry (gh-ocannl-579),
          structural rather than label-keyed, and absent when it timed none — or for entries written
          before this field existed, which therefore replay as "nothing is known". A measurement of
          the PROGRAM, like [best_ms] and [baseline_ms] and under the same key regime, which is what
          makes it replayable: the flip chain's profitability term reads it, so without it a warm
          cache would rank the decision surface differently from the cold run that measured it. *)
  default_fingerprint : string option; [@sexp.option]
      (** {!Schedule.default_schedule_fingerprint} at store time, present iff [default_ms] is: the
          cache key covers only the source digest and the backend, so a config change can redefine
          what "the default pipeline" means without missing the cache. A replaying process compares
          fingerprints and drops a stale [default_ms] (the schedule itself stays valid — only this
          diagnostic is config-relative). *)
}
[@@deriving sexp]

val entry_version : int
(** Bumped when the canonical rendering or the saved-schedule format changes; stale entries are
    ignored by {!lookup}. *)

val objective_tag : unit -> string
(** The autotuner's configured timing objective ([autotune_timing]), normalized to the spelling a
    cache key carries (gh-ocannl-755). What {!cache_key} uses when its caller supplies none — so a
    caller with no resolved mode of its own keys against the one a search in this process would use,
    rather than restating a default that could drift. An unknown spelling passes through: the
    setting is validated where it is acted on, and this only has to keep unlike regimes apart. *)

val key_components : string list
(** The named components a {!cache_key} is built from, in order: ["digest"], ["backend"],
    ["numerics"], ["codegen"], ["pool"], ["timing"]. The list drives {!cache_key} rather than
    describing it, so it is a complete and current enumeration of the cache's identity — which is
    what the digest-completeness registry classifies configuration keys against (gh-ocannl-572,
    [test/operations/digest_completeness]). *)

val cache_key :
  ?objective:string -> limits:Backend_intf.hardware_limits -> canonical -> backend:string -> string
(** Filename-safe cache key: the digest, the backend name, {!numerics_tag} of the current numerics
    policy, {!codegen_tag} of the codegen configuration (including [limits.codegen_tag], the
    compiling backend's own contribution), the worker-pool signature ([limits.worker_pool_tag],
    gh-ocannl-530: CPU crowns do not transfer across pools), and the autotuner's timing objective
    ([objective], defaulting to {!objective_tag}).

    The objective is a key component because the two objectives crown DIFFERENT candidates
    (gh-ocannl-755, measured): an entry crowned under isolated timing is not the answer to a search
    asking about queued timing, and the times it stores are readings of a different quantity, which
    a replay would copy into the reading process's report under its own label. CUDA/HIP [queued]
    keys carry policy generation 2 (gh-ocannl-892), invalidating winners measured under the old
    depth-200/short-batch regime without invalidating the unchanged cc/Metal regimes. [objective] is
    for the caller that resolved a mode explicitly rather than from configuration
    ({!Autotune.tune}'s [?timing]); everyone else wants the default. The backend-supplied components
    arrive as the whole [limits] record rather than one optional argument each, so a component added
    there reaches every call site instead of defaulting to absent at the ones that were not updated
    (gh-ocannl-572). Callers time kernels on a concrete device, so include anything else that
    distinguishes performance environments in [backend] (e.g. a device id) if needed. *)

val cache_regime_version : int
(** Version of the filename-key regime recorded once per cache directory (gh-ocannl-835). Bump it
    when the schema of {!key_components} changes. This is independent of {!entry_version}, which
    versions the payload stored at one key. *)

val regime_stamp_filename : string
(** The directory-local file containing {!cache_regime_version}. Exposed as part of the on-disk
    cache format, including for tools and synthetic cache-directory tests; callers should not edit a
    live cache's stamp. *)

val regime_lock_filename : string
(** The permanent record-lock file shared by cache-opening processes. Exposed as part of the on-disk
    cache format for synthetic cache-directory tests; callers must never unlink a live cache's lock
    file. *)

val store : dir:string -> key:string -> entry -> unit
(** Writes the entry to [dir]/[key].sexp, creating [dir] (and parents) if missing. Publication goes
    through {!Utils.Atomic_file}, so concurrent writers tolerate each other (last write wins) and a
    failed write or commit removes its own staging artifact and leaves an earlier complete entry
    intact. Before writing, cache-open serializes participating processes on a permanent lock file:
    an older or absent regime stamp sweeps every [.sexp] entry and is atomically replaced by the
    current stamp; a malformed or newer stamp refuses the operation without changing its stamp or
    entries. Holding that lock through the write prevents a concurrent regime transition from
    deleting the new entry. A filesystem refusal is not propagated: the cache is an optimization,
    and an entry that could not be written is a future miss rather than a failed run. *)

val lookup : dir:string -> key:string -> entry option
(** [None] on missing file, unparsable content, version/digest mismatch, or a refused cache-open.

    Together with {!store} this participates in the same permanent record lock and regime-stamp
    protocol described there, and sweeps the cache directory's crash-stale staging files
    ({!Utils.Atomic_file.cleanup_stale_once}) once per directory per process. Every participating
    reader and writer holds the lock through its entry I/O, so same-version processes race benignly
    and an older participating binary refuses after a newer one advances the stamp. The OS releases
    the record lock on process death; the lock file is never unlinked, avoiding an inode-replacement
    race. A binary predating this protocol does not participate and must not share a live directory
    during an upgrade. *)

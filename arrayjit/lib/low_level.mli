(** {1 A for-loop-based array language and backend-agnostic optimization} *)

open Base

(** {2 Global references} *)

module Scope_id : sig
  type t = { tn : Tnode.t; scope_id : int } [@@deriving sexp_of, equal, hash, compare]
  type comparator_witness

  val comparator : (t, comparator_witness) Base.Comparator.t
end

type scope_id = Scope_id.t = { tn : Tnode.t; scope_id : int }
[@@deriving sexp_of, equal, hash, compare]

val get_scope : Tnode.t -> scope_id

(** {2 Low-level representation} *)

(** How a loop's iterations map to hardware; see docs/proposals/axis-types-for-loops.md. [Serial] is
    an ordinary for-loop. [Grid] / [Workgroup] bind the loop index to a GPU grid / workgroup (block,
    threadgroup) hardware index instead of looping; [Workgroup_reduce] is a [Workgroup] axis
    participating in a workgroup-cooperative reduction (see the contract below). [Unrolled] is
    emitted as the repeated body with substituted constants. [Vectorized] renders eligible bodies as
    explicit SIMD code — elementwise statements via vector extensions / packed loads (gh-ocannl-164
    / gh-ocannl-463), a single recognized accumulation as independent accumulator chains with a
    horizontal reduce at exit on CPU backends (gh-ocannl-468) — and everything else as a serial loop
    annotated with the backend's vectorization pragmas when it provides them (a plain, un-annotated
    serial loop for accumulating bodies, whose loop-carried dependency the pragmas would deny) —
    like the hardware kinds, the annotating pass asserts iteration independence or, for a recognized
    accumulation, licenses reassociating it. Hardware slots are positional: among a kernel's loops
    of one kind, the innermost binds [.x], then [.y], [.z]. Annotated loops must have [from_ = 0]
    and iterations with no cross-iteration dependencies ([Vectorized] accumulations again excepted).
    [Workgroup_reduce] is the labelled exception; its body must either stage its communication
    explicitly through workgroup-shared nodes and barriers (rendered by binding the index like
    [Workgroup]), or be a single accumulation statement [acc = op(acc, contrib)] over an
    associative-commutative [op] with the accumulator's indices free of the loop index — the
    renderer then owns the communication: warp/simdgroup shuffles on GPU backends (gh-ocannl-462),
    the plain serial loop on CPU backends. Like [Vectorized], the annotation licenses reassociating
    the (floating-point) reduction. *)
type axis_type = Serial | Grid | Workgroup | Workgroup_reduce | Unrolled | Vectorized
[@@deriving sexp, compare, equal]

val axis_type_label : axis_type -> string
(** Loop keyword used by the human-readable printers: plain ["for"] for [Serial], ["for@<axis>"]
    otherwise. *)

(** Which pass minted a [Local_scope] (gh-ocannl-687). The construct has two producers, and a
    consumer that walks the IR looking for schedulable structure means only one of them:

    - [Inlined_computation] -- virtualization's inline of a virtual node's computation at a read
      site, and the rewrites that carry those scopes along. The loops inside such a body are the
      inlined node's own iteration space, re-instantiated per use site; no [Schedule] op has ever
      targeted them.
    - [Schedule_minted] -- the accumulator localization built by [Schedule]'s materializing [Unroll]
      and by [Partition] (gh-ocannl-639), and by [C_syntax.try_localize_serial_reduce]: a running
      value for a MATERIALIZED cell, whose body holds the very per-step / per-segment loops
      [Schedule.rewrite_loop] retargets.

    This is the fact [Autotune.collect_loops] needs: it enumerates loops inside [Schedule_minted]
    scopes only, so the action menu does not spend its per-unit budget proposing splits, swaps,
    unrolls and vectorize retypes for inlined interpolation and reduction loops that no schedule op
    has ever been able to reach.

    Deliberately NOT the mechanism behind the scope-target contract {!input_scope_ids} serves: that
    one asks whether a scope was in the program a given {!optimize} call was HANDED, which is a
    per-call fact -- a virtualizer-minted scope handed back into a second [optimize] still carries
    [Inlined_computation] and still is not that call's to retract -- and hand-built IR has no honest
    way to spell "not mine". *)
type scope_mint = Inlined_computation | Schedule_minted [@@deriving sexp, compare, equal]

(** The iteration order of a {!t.Scan_loop} (gh-ocannl-696): [Forward] runs the index from [from_]
    up to [to_], [Backward] from [to_] down to [from_]. Part of the construct from day one because
    every adjoint of a forward scan is a backward scan over the same range. *)
type scan_direction = Forward | Backward [@@deriving sexp, compare, equal]

(** Cases: [t] -- code, [scalar_t] -- single number at some precision. *)
type t =
  | Noop
  | Comment of string
  | Staged_compilation of (unit -> PPrint.document)
  | Seq of t * t
  | For_loop of { index : Indexing.symbol; from_ : int; to_ : int; body : t; axis : axis_type }
  | Scan_loop of {
      index : Indexing.symbol;
      from_ : int;
      to_ : int;
      direction : scan_direction;
      carried : carried list;
      body : t;
    }
      (** A loop with declared loop-carried scalar state (gh-ocannl-696): the minimal recurrence
          construct behind cumulative ops, online softmax and top-k. Semantics, for [carried] =
          [c_1 .. c_n]: each [c.prev] is set to [c.init] once, before the first iteration; then for
          every value of [index] in [from_ .. to_] taken in [direction], [body] runs reading the
          previous iteration's state through [Get_local c.prev] and producing the next through
          [Set_local c.next]; after the body, every [c.prev] takes its [c.next] simultaneously
          (phi-style rotation, so a body may read any old value after any new one is written). A
          dead range ([to_ < from_]) is malformed and refused by the validator: a producer with no
          iterations emits [Noop], so no walker needs a dead-scan convention. The final state is not
          readable after the loop: a body that wants a trajectory or a final value writes it to a
          tensor node itself.

          Contract, enforced by {!validate_scan_loops} at both ends of the pipeline: [c.prev] and
          [c.next] are ids, pairwise distinct across [carried], over one node DECLARED virtual
          ([c.prev.tn == c.next.tn]), which names and types the state and is never a buffer;
          [c.next] is written exactly once, as a top-level statement of [body] (not under a guard or
          a nested loop), and read only by statements after that write; nothing writes [c.prev];
          [c.init] reads no carried local and does not mention [index]. The state is scalars only,
          of unbounded arity -- small fixed extents unroll into it; there is no dynamic indexing
          into state.

          Placement contract: a virtualization candidate whose captured computation contains a scan
          -- one written inside the body, one fed by a scan through a scope local, or one merely
          enclosing a sibling scan -- is refused ([Non_virtual 148]): a cell's value depends on the
          whole prefix through state the index does not parameterize, so per-cell recomputation at a
          read site is unbounded and wrong. Reads inside the body inline as usual (an independent
          producer replays soundly at any position of the scan). Schedule transforms neither target
          the scan's own index nor reach the loops inside its body (the loop is opaque to
          [Schedule.find_loops_env] / [rewrite_loop], so an op naming either symbol declines with
          the usual no-such-loop [Invalid_argument]); enclosing loops keep their full menu, since
          the state is per-iteration-of-the-enclosing-loop local scratch. The index is an ordinary
          affine loop symbol for footprint purposes ({!loop_bounds}, {!affine_accesses}, interval
          analysis) -- a scan's accesses stay affine and dense even though its values are serial. *)
  | Zero_out of Tnode.t
  | Set of {
      tn : Tnode.t;
      idcs : Indexing.axis_index array;
      llsc : scalar_t;
      mutable debug : string;
    }
  | Set_dynamic of {
      tn : Tnode.t;
      idcs : Indexing.axis_index array;
          (** Static everywhere except [dyn_axis] (a [Fixed_idx 0] placeholder there). *)
      dyn_axis : int;  (** Which [idcs] slot is replaced by [dyn_value] at codegen time. *)
      dyn_value : scalar_arg;
          (** Integer-valued index spliced into the row-major offset at [dyn_axis]. *)
      llsc : scalar_t;
      mutable debug : string;
    }
      (** A scatter: like [Set] but the write lands at a runtime row of axis [dyn_axis] — the write
          counterpart of {!scalar_t.Get_dynamic}. gh-466: produced only by
          {!rewrite_one_hot_reductions} (transposed one-hot pattern, the embedding-table gradient);
          never constructed by [Assignments] lowering. Schedule analyses must treat this write as
          statically unknown: loops whose index reaches [dyn_value] carry a cross-iteration write
          dependency and must stay serial (the deterministic no-atomics invariant). *)
  | Set_from_vec of {
      tn : Tnode.t;
      idcs : Indexing.axis_index array;
      length : int;
      vec_unop : Ops.vec_unop;
      arg : scalar_arg;
      mutable debug : string;
    }
  | Set_local of scope_id * scalar_t
  | Declare_local of { id : scope_id; needs_init : bool }
  | Workgroup_barrier
      (** Workgroup-scoped synchronization ([__syncthreads()] / [threadgroup_barrier]). An opaque
          effectful statement: no CSE, hoisting, or code motion across it. Grid-scoped
          synchronization is deliberately not representable. *)
  | If of { cond : scalar_arg; body : t }
      (** Guarded statement: [body] executes iff [cond] is nonzero (renders as
          [if (cond != 0) { body }]). Introduced by launch-extent guards on hardware-annotated loops
          (docs/proposals/axis-types-for-loops.md §2); [simplify_llc] erases a guard whose condition
          an interval proves, and simplifies a surviving guard's body under the bounds the condition
          implies. A conditional write is never a definite write; virtualization treats guarded
          computations as non-inlineable in v1. *)
  | Tile_mma of {
      d : Tnode.t * Indexing.axis_index array;  (** Accumulator block base. *)
      a : Tnode.t * Indexing.axis_index array;
      b : Tnode.t * Indexing.axis_index array;
      ta : bool;
          (** [a] is stored transposed: its tile axes are [k, i]-major rather than [i, k]. *)
      tb : bool;
          (** [b] is stored transposed: its tile axes are [j, k]-major rather than [k, j]. *)
      m : int;
      n : int;
      k : int;  (** Covered block extents (multiples of the backend's intrinsic tile). *)
      ldd : int;
      lda : int;
      ldb : int;
          (** Leading-dimension strides in elements, recorded by {!Schedule.optop.Tensorize}: the
              tnode's minor dim in the plain last-two-axes case, larger when interior batch axes sit
              between the tile roles (gh-ocannl-528). *)
      lane : Indexing.symbol;  (** The cooperating [Workgroup] axis (extent = SIMD width). *)
      tile : Register_tile.t option;
          (** The C-tile geometry the register-tiled CPU rendering must use (gh-ocannl-619), carried
              from {!Schedule.optop.Tensorize}; [None] lets the renderer choose. Ignored by the
              intrinsic (tensor-core) emissions. *)
      fallback : t;  (** Semantically equivalent scalar micro-kernel over fresh serial symbols. *)
    }
      (** Cooperative tile multiply-accumulate (docs/proposals/tensorize-mma.md):
          [d[i,j] += Σ_{l<k} a[i,l] * b[l,j]] for [i < m], [j < n], relative to the operands' base
          index vectors, executed jointly by the threads of the [lane] axis (tensor cores /
          [simdgroup_matrix]). Each operand's tile is a 2-D slice: minor tile axis on the tnode's
          last axis (stride 1), major tile axis at the recorded leading-dimension stride (with
          [ta]/[tb] the stored layout is the role's transpose and emissions use the hardware
          transpose flag); the base indices must not mention [lane]. Constructed by schedule
          transforms only ({!Schedule.optop.Tensorize}), after the optimization pipeline. Backends
          without an MMA hook render [fallback] under an [if (lane == 0)] guard. Validates like
          {!Workgroup_barrier} plus a write of [d] for the coverage rule; see {!validate_parallel}.
      *)
[@@deriving sexp_of, equal]

and scalar_t =
  | Local_scope of {
      id : scope_id;
      body : t;
      orig_indices : Indexing.axis_index array;
      mint : scope_mint;  (** Which pass built this scope; see {!scope_mint}. *)
    }
      (** An inlined sub-computation whose value is [body]'s final [Set_local] of [id].

          {b Scope purity} (gh-ocannl-584): [body]'s only effect is on the locals it owns — its own
          [id], plus ids [Declare_local]d lexically within it. Never a tensor node, never a sibling
          or enclosing scope's local, and no barrier or staged callback. A scope body does not
          execute where it is written: codegen hoists it ahead of the enclosing statement ordered by
          [scope_id] rather than by the operand's syntactic position, [simplify_llc] collapses a
          single-assignment scope into the expression (moving its reads the other way), and
          [hoist_cross_statement_cse] lifts a body shared by sibling statements out of the statement
          entirely, to run once ahead of the first user. Purity is what makes all three placements
          unobservable from outside the body — the same assumption {!Affine.path_before} makes when
          it declines to order sibling [Arg] positions. (It governs a body's effects, not its
          inputs: the hoist additionally needs the body's reads — tensor nodes and scope locals
          alike — untouched across the statements it is lifted over, which is its own hazard check's
          obligation.) Enforced by {!validate_scope_bodies}; the optimization pipeline satisfies it
          by construction.

          {b Scope target} (gh-ocannl-681): [id.tn] must be VIRTUAL for as long as [optimize] is
          looking at the scope, since the scope denotes that node's inlined computation. A scope
          over a materialized node is REJECTED, never rewritten — the optimizer would have to
          discard the body and read the buffer, which is what used to happen silently. Mind that a
          node with no setter is decided non-virtual, so a hand-built scope over a freshly created
          node is rejected too unless the node is declared virtual. AFTER [optimize] the shape is
          legal and means the opposite: [Schedule]'s materializing [Unroll] and [Partition] mints
          and [C_syntax.try_localize_serial_reduce] localize a materialized accumulator this way,
          and codegen renders it. Localization is codegen's business alone (gh-ocannl-693); IR
          already in that form reaches a backend through [Context.compile ?prelowered], never
          through [optimize]. Only a scope [optimize] MINTED may be retracted to a [Get], and only
          by itself, when a later refusal materializes the node it had inlined — which happens
          (gh-ocannl-704): the virtualization walk is in source order, so a statement reached after
          a read that already minted can materialize the node under the scope. *)
  | Get_local of scope_id
  | Get of Tnode.t * Indexing.axis_index array
  | Get_dynamic of {
      tn : Tnode.t;  (** The gathered table; treated as a read of [tn], like [Get]. *)
      idcs : Indexing.axis_index array;  (** Static everywhere except [dyn_axis]. *)
      dyn_axis : int;  (** Which [idcs] slot is replaced by [dyn_value] at codegen time. *)
      dyn_value : scalar_arg;
          (** Integer-valued index spliced into the row-major offset at [dyn_axis]. gh-343: produced
              only by {!rewrite_one_hot_reductions}; never escapes low-level / backend codegen. *)
    }
  | Get_merge_buffer of Tnode.t * Indexing.axis_index array
  | Ternop of Ops.ternop * scalar_arg * scalar_arg * scalar_arg
  | Binop of Ops.binop * scalar_arg * scalar_arg
  | Unop of Ops.unop * scalar_arg
  | Constant of float
  | Constant_bits of int64  (** Direct bit representation, primarily for uint4x32 *)
  | Embed_index of Indexing.axis_index
[@@deriving sexp_of, equal, compare]

and scalar_arg = scalar_t * Ops.prec [@@deriving sexp_of, equal, compare]
(** The argument precision is preserved in heterogeneous precision operation arguments, and is
    ignored (overridden) in homogeneous precision operations. *)

and carried = { prev : scope_id; next : scope_id; init : scalar_t }
[@@deriving sexp_of, equal, compare]
(** One loop-carried scalar of a {!t.Scan_loop}: read as [prev], written as [next], rotated
    [prev := next] after each iteration, [prev := init] before the first. Both ids name the same
    virtual node, whose precision is the state's precision. *)

module Canonical_render : sig
  (** gh-563: the one canonical rendering of lowered code, shared by both digest consumers —
      {!analysis_cache_stats}' cache (keyed inside [optimize]) and [Schedule_cache.canonicalize]
      (schedule replay across sessions).

      The walk is the same for both: index / scalar / statement emission, loop-binder tokens,
      local-scope alpha renaming, comment skipping, opaque-statement handling. What deliberately
      differs is the identity {!policy} — the analysis cache keys tensor nodes and static symbols by
      identity (a hit reuses the stored code verbatim), the schedule cache alpha-renames everything
      (a hit replays a schedule onto a different-but-isomorphic lowering).

      Both digests are correctness-critical, so keep the split honest: a new {!t} / {!scalar_t}
      construct is rendered in the walk and only there (the matches are exhaustive, so omitting it
      breaks the build); a new digest-relevant {i fact} enters the walk if it belongs to the code
      itself, or exactly one {!policy} field / one consumer preamble if it is an identity choice or
      a consumer-specific companion. *)

  (** How [Tile_mma] enters the rendering. *)
  type mma_policy =
    | Opaque_mma
        (** Mark the rendering incomplete and emit a placeholder — the consumer's guarantees do not
            extend to the construct. *)
    | Structural_mma  (** Render operands, extents, lane and fallback body. *)

  type policy = {
    emit_tn : Tnode.t -> unit;  (** Render a tensor node reference. *)
    emit_free_sym : Indexing.symbol -> unit;
        (** Render a symbol that neither an enclosing loop binder nor {!initial_tokens} bound. *)
    on_bind_loop : Indexing.symbol -> id:int -> shadowed:bool -> unit;
        (** Called when a [For_loop] binder mints the token ["b<id>"]. [shadowed] iff the symbol
            already had a token: a duplicated binder makes symbol references ambiguous. *)
    mark_incomplete : unit -> unit;
        (** Called when an opaque construct makes the rendering an unfaithful summary of the code.
        *)
    mma : mma_policy;
    initial_tokens : (Indexing.symbol * string) list;
        (** Symbols pre-bound to a rendering token before the walk — the static indices, for the
            consumer that renders them positionally. *)
  }

  val emit : buf:Buffer.t -> policy -> t -> unit
  (** Appends the canonical rendering of the code to the buffer. Deterministic: the caller digests
      the buffer, usually after its own preamble and companion sections. *)
end

val scalar_precision : scalar_t -> Ops.prec
val apply_op : Ops.op -> scalar_t array -> scalar_t
val flat_lines : t list -> t list
val unflat_lines : t list -> t
val loop_over_dims : int array -> body:(Indexing.axis_index array -> t) -> t
val unroll_dims : int array -> body:(Indexing.axis_index array -> offset:int -> t) -> t

val loop_over_padding_region :
  dims:int array -> padding:Ops.axis_padding array -> body:(Indexing.axis_index array -> t) -> t
(** Generate loops that iterate only over the padding margins of a tensor. For dimensions with
    padding, generates separate loops for left margin, middle (recursing), and right margin. The
    middle region continues recursing to find padding in other dimensions. *)

val has_accumulation : t -> bool
(** Whether the tree carries a read-modify-write accumulation: some [Set] (resp. [Set_local]) reads
    its own target — a loop-carried dependency through memory when the written cell does not vary
    with an enclosing loop. Conservative: [Local_scope] contents count as reading anything, and
    [Tile_mma] and (gh-466) [Set_dynamic] accumulate by construction. Used by the autotune menu and
    by codegen fallbacks that must not assert iteration independence (e.g. vectorization pragmas)
    over an accumulating body (gh-ocannl-468). *)

val scalar_touches_tn : Tnode.t -> scalar_t -> bool
(** Whether the scalar reads the node anywhere (gh-ocannl-639) — certifies that an accumulation's
    contribution is free of the accumulator's node, which licenses holding the accumulator out of
    memory across the reduction. *)

val code_touches_tn : Tnode.t -> t -> bool
(** The statement-body counterpart of {!scalar_touches_tn}: any read or write of the node. *)

val accum_update_parts :
  tn:Tnode.t -> idcs:Indexing.axis_index array -> scalar_t -> (Ops.binop * scalar_t) option
(** The accumulation-update statement shape [tn[idcs] = op(tn[idcs], contrib)] (or its FMA form)
    over an associative-commutative [op], with [contrib] free of [tn]; returns [(op, contrib)]. The
    single source of truth shared by [C_syntax]'s widened renderings and
    [Schedule.Unroll ~materialize:true]'s scope-form unrolling (gh-ocannl-639), so "what counts as
    an accumulation" cannot drift between the schedule transform and the emission. *)

val subst_accum_read :
  tn:Tnode.t -> idcs:Indexing.axis_index array -> id:scope_id -> scalar_t -> scalar_t
(** Retarget an {!accum_update_parts}-shaped update's read of [tn[idcs]] to the scope local [id].
    Raises on any other shape. *)

val scalar_reads_scope : id:scope_id -> scalar_t -> bool
(** Whether the scalar reads the scope local [id], descending into nested scope bodies — the
    scope-local counterpart of {!scalar_touches_tn}. Distinguishes a scope-opening init (a
    [Set_local] whose value is free of the local) from a self-referential update. *)

val accum_local_update_op : id:scope_id -> scalar_t -> Ops.binop option
(** The reduction operator of a scope-local update in either spelling: the plain
    {!accum_local_update_parts} form, or virtualization's guarded-read form
    [Where (index-only cond, update, Get_local id)] (possibly nested per guard condition). The
    residency classifier's recognizer (gh-ocannl-663): the guarded form is a reduction for
    accumulator-width purposes but does NOT decompose into an unguarded [(op, contrib)], so it is
    deliberately not part of {!accum_local_update_parts} or of {!scope_updates_reduce_op}'s hoist
    license. *)

val accum_local_update_parts : id:scope_id -> scalar_t -> (Ops.binop * scalar_t) option
(** The reduce-shaped update of a scope LOCAL, [local = op(local, contrib)] (or its FMA form) with
    [contrib] free of the local — [subst_accum_read]'s output shape; returns [(op, contrib)]. The
    SIMD reduction rendering uses it to fold vector chains into a widened accumulator's scope local
    (gh-ocannl-639), and {!peel_accum_nest}'s scope-form validation is built on it. *)

type thread_storage = [ `Device | `Shared | `Thread ]
(** How a node's storage is shared across the threads a hardware binding creates: device-resident
    (every thread of the launch), workgroup-shared (the threads of one block), or thread-private (a
    per-thread local array, which no binding can race). The tags are spelled as the tile-MMA address
    space [C_syntax.mma_space] spells them, so the fragment-operand sites answer the same question
    by coercing this classifier rather than repeating it. *)

type thread_slot = [ `Grid | `Workgroup ] * int
(** One coordinate of thread identity: the hardware dimension a bound loop occupies
    ({!hardware_axis_info}'s kind and slot). *)

val unseparated_thread_write :
  active:thread_slot list ->
  thread:(Indexing.symbol -> thread_slot option) ->
  deferred:(Indexing.symbol -> bool) ->
  storage:(Tnode.t -> thread_storage) ->
  t ->
  (Tnode.t * Indexing.axis_index array * string) option
(** The legality of a hardware binding, as a query on the written cell (gh-ocannl-959). [active] is
    the launch's thread identity, one coordinate per slot some register-bound loop of extent above
    one occupies; [thread] names every bound loop and its slot — the register-bound ones and cc's
    pool-parallel outermost [Grid] loops, a per-loop binding; [deferred] the bound loops whose
    separation another call judges (a reduce lane left to its own rendering: it still covers its
    slot). The first store in the kernel to storage the threads share that two threads may own, with
    the reason: an active slot with no bound loop enclosing the store (every coordinate of the slot
    executes it), or a cell whose index map does not {!Affine.separates} the enclosing thread
    symbols over every loop symbol in scope — mentioning a bound axis is not separating it
    ([acc[i + j]] under two bound axes, [acc[i + k]] under a bound [i] and a serial [k]). Guards
    narrow the environment as gh-ocannl-566's simplifier narrows it: [If (i < 16)] shrinks the range
    the radix argument sees and [If (i == 0)] pins [i] to one thread along that axis alone. A [Grid]
    axis is not a thread of workgroup-shared storage (one block, one copy). What is judged is the
    store's own threads: whether other statements' threads touch the cell across a barrier is
    dependence analysis, the obligation of whoever mints the pin or the binding, as it was for every
    store before this rule. Reads play no part: a store every thread performs to one cell is a
    write-write race whatever the values. A dead level and a false guard execute nothing; a
    [Set_dynamic] is judged by its static slots, a [Set_from_vec] by its aligned run blocks, a
    [Zero_out] as a store of every cell, a [Tile_mma] through its [fallback] with its own [lane]
    excused (gh-ocannl-960). The whole kernel is walked. The renderer asks this at every binding it
    emits: for the kernel's [Grid]/[Workgroup] axes before rendering, and for a [Workgroup_reduce]
    lane the warp shuffle cannot own, before falling through to the plain binding. *)

val has_accumulating_cell : t -> bool
(** Whether the tree holds a SELF-RECURRENCE: some [Set] whose value reads the very cell it writes
    (gh-ocannl-733). This is what makes the localizing peel a live question at a level, and it is
    the predicate the peel census gates on. Narrower than {!has_accumulation} in two ways, both
    deliberate: the recurrence is on the CELL rather than on the node, and a [Local_scope] counts
    only when its body actually reads that cell — {!has_accumulation} must count every scope
    conservatively, being the predicate that decides whether iteration independence may be asserted,
    and censusing on it recorded every non-reduction virtualized scope as a declined reduction site.
    A dynamic gather may recur unless unequal fixed indices outside its runtime axis prove the cells
    disjoint; symbolic slots remain potentially aliasing. The selector's reads count too. *)

(** What {!peel_accum_nest} decided about one [If] it peeled through (gh-ocannl-733). Two shapes
    that render identically can earn different verdicts here, which is the whole point of reporting
    them: a test pinning the rendered FORM cannot tell "the guard was confined to the peeled levels"
    from "the guard mentioned an enclosing lane and the cell separated it". *)
type peel_guard_verdict =
  | Guard_confined  (** [Affine.Confined_to_peel]: the guard mentions no enclosing loop symbol. *)
  | Guard_lane_private
      (** [Affine.Lane_private_if_separated], ADMITTED: the guard mentions enclosing loop symbols
          and the accumulated cell was shown to separate them (gh-ocannl-721). Appears only in a
          report that reached a base — separation is checked there, so nothing before it can earn
          this. *)
  | Guard_lane_private_unresolved
      (** The same guard in a report that REFUSED: the peel stopped before the base could settle
          whether the cell separates the enclosing symbols — because the cell shares them
          ([Refused_cell_shared]) or because the peel never reached a base at all. Distinct from
          {!Guard_lane_private} so that a refusing report cannot read as an admitted guard beside
          its own refusal. *)
[@@deriving sexp, equal, compare]

(** Why {!peel_accum_nest} stopped, when it did (gh-ocannl-733). *)
type peel_refusal =
  | Refused_not_a_nest
      (** The level's body is neither a single peelable loop, nor a pure-index-guarded [If], nor an
          accumulation base: a sibling statement, a data-dependent guard, a non-accumulating [Set].
      *)
  | Refused_dead_level  (** A level with [to_ < from_], which performs no accesses. *)
  | Refused_guard_fixed of string
      (** [Affine.Not_peelable] with its reason: the guard's truth is fixed for the whole nest. *)
  | Refused_cell_varies
      (** The accumulation base's cell mentions a peeled level, so it is not one cell across them.
      *)
  | Refused_cell_shared
      (** An admitted [Affine.Lane_private_if_separated] guard whose cell does not separate the
          enclosing symbols, or does not stay inside the node's box over their full ranges. *)
[@@deriving sexp, equal, compare]

type peel_report = {
  levels : int;  (** Loop levels peeled before the outcome. *)
  guards : peel_guard_verdict list;
      (** The verdict of each peeled [If], outermost first. A report carrying a [refusal] never
          carries an ADMITTED lane-private verdict: those resolve to
          {!Guard_lane_private_unresolved}, since separation is decided at the base the peel did not
          reach. *)
  refusal : peel_refusal option;
      (** [None] exactly when the peel reached an accumulation base, i.e. when {!peel_accum_nest}
          returned [Some]. *)
}
[@@deriving sexp_of]
(** What one {!peel_accum_nest} call decided, beyond whether it succeeded (gh-ocannl-733). Reported
    to the optional [~report] callback on every call, success or refusal; [C_syntax] turns it into
    the per-routine peel census, so that a test can pin which DECISION produced a kernel and not
    only which form was rendered. *)

val peel_accum_nest :
  ?extra_level:(Indexing.symbol -> axis_type -> bool) ->
  ?report:(peel_report -> unit) ->
  loop_bounds:(Indexing.symbol * (int * int)) list ->
  free_of:Indexing.symbol list ->
  t ->
  (Tnode.t
  * Indexing.axis_index array
  * [ `Update of scalar_t | `Scope of scope_id * t list ]
  * string
  * (t -> t))
  option
(** Peel a single-statement reduction nest down to its accumulation base (gh-ocannl-639). Levels are
    Serial/[Unrolled]/[Vectorized] loops, loops [extra_level] vouches for (codegen passes a
    predicate accepting hardware-annotated reduction loops its backend serializes — the schedule
    mints pass nothing, since wrapping a hardware-annotated loop in a scope at transform time would
    break the schedule on backends that do bind the hardware dimension), and pure-index-guarded
    [If]s (the gh-490 [If (i < s)] shape and its constant-bound sibling — data-dependent guards stay
    opaque), each containing nothing else; the base is a raw {!accum_update_parts}-shaped update
    ([`Update]) or the scope form a previous rewrite minted ([`Scope]: the scope id and the update
    statements after the opening init, validated to carry ONE reduction operator — mixed-operator
    sequences are not reductions and keep their per-iteration narrowing), with the accumulated cell
    invariant across the peeled levels ([free_of] seeds the invariance check with the caller's own
    loop). [rebuild] re-wraps a replacement base statement in the peeled levels. The ONE definition
    shared by [C_syntax]'s localizing serial fallback and the schedule mints, so transform and
    emission cannot drift.

    A DEAD level ([to_ < from_]) is never peeled, and that refusal is load-bearing rather than
    tidiness: the body of a dead loop performs no accesses at all — the routine-interface walk
    propagates liveness as [live && to_ >= from_], so a node reached only under one is absent from
    the parameters and need not be allocated — while every form this peel licenses reads and writes
    the accumulated cell OUTSIDE the levels, unconditionally. Peeling a dead level would invent
    accesses the program does not make, possibly on an identifier the interface never declared.

    Whether a GUARD may join the peeled levels is [Affine.peel_guard]'s and [Affine.separates]'s
    answer, not this function's (gh-ocannl-722) — [rebuild] keeps the guard around the accumulating
    update only, so the localized form loads and stores outside it, and the engine is where the two
    hazards that creates are stated: a guard mentioning no peeled symbol is fixed for the whole
    nest, so the hoist invents both accesses, and one mentioning an enclosing loop symbol selects
    among that level's iterations, so lanes the guard excludes would write their unchanged local
    back over the accumulating lane's result. The second is admitted when the accumulated cell
    SEPARATES those enclosing symbols — each instance then owning a distinct cell makes the invented
    load/store pair private and idempotent (gh-ocannl-721) — which is why the cell reaches the
    decision and the peel defers it to the base.

    [loop_bounds] is {!loop_bounds} of the enclosing program, and supplies both the classification
    and the ranges: a guard symbol in it that is not peeled is an enclosing level's index, while one
    outside it is a static index parameter or a runtime extent and is harmless — which is what keeps
    gh-490's runtime-extent guard ([Assignments.extent_guard]'s [i < s], whose bound is a static
    symbol rather than a constant) peelable. Required rather than defaulted, and derived from the
    program rather than certified by the caller, so that no call site can forget it: the mints of
    [Schedule] need it as much as codegen does, since a refused mint there turns segment seams into
    narrowing points instead of merely declining an optimization.

    [report], when given, is called EXACTLY ONCE per call with the {!peel_report} of what was
    decided — how many levels were peeled, which verdict each peeled guard earned, and the refusal
    where there was one (gh-ocannl-733). The result alone cannot answer that: two nests differing
    only in whether the accumulated cell mentions the enclosing index peel a different number of
    levels under a different guard verdict and render the same localized kernel. Codegen passes a
    reporter that accumulates the per-routine peel census; the schedule mints pass none. *)

(** {2 Hardware axis analyses}

    Phase B of docs/proposals/axis-types-for-loops.md. Hardware slot assignment is positional, not
    stored in the IR: among a kernel's annotated loops of one kind, the innermost binds [.x] (slot
    0), the next [.y], then [.z]; [Workgroup] and [Workgroup_reduce] share the block/threadgroup
    slot space. [Grid] slots [>= 2] all share the hardware [.z] dimension by folding
    (gh-ocannl-643): the launch's [.z] extent is their per-slot maxima multiplied out
    ({!launch_dims}) and each such loop binds [(z / stride) % cap] ({!grid_fold}); [Workgroup] slots
    stay capped at 3. *)

type launch_dims = { grid : int array; block : int array } [@@deriving sexp_of, equal]
(** Arrays of length 3 ([.x], [.y], [.z]); all-1s for all-[Serial] code. *)

type hardware_axis_info = {
  ha_index : Indexing.symbol;
  ha_kind : [ `Grid | `Workgroup ];
  ha_axis : axis_type;
      (** The loop's own annotation: [Workgroup] and [Workgroup_reduce] share the kind (and the slot
          space) but not the rendering, and the binding-legality check reads the difference. *)
  ha_slot : int;  (** Positional: the innermost same-kind loop binds [.x] = slot 0. *)
  ha_from_ : int;
  ha_extent : int;  (** [to_ - from_ + 1]. *)
}

val hardware_axes : t -> hardware_axis_info list
(** All hardware-annotated loops in pre-order, with their positional slots. *)

val launch_dims : t -> launch_dims
(** Per-slot maximum extents over the kernel's annotated loops; [grid.(2)] is the product of the
    per-slot maxima of grid slots [>= 2] (the [.z] fold, see the section comment). *)

val grid_fold : hardware_axis_info list -> slot:int -> int * int option
(** The binding arithmetic of a [Grid] loop at [slot >= 2] under the [.z] fold: [(stride, cap)] such
    that the loop binds [(z / stride) % cap] — [stride] the product of the per-slot maxima of grid
    slots in [\[2, slot)], [cap = None] (no modulo) when no higher grid slot exists in the kernel.
    [(1, None)] — the bare [.z] register — for the common single-slot-2 case. *)

val scope_purity_violation : t -> string option
(** gh-ocannl-584: the predicate form of {!validate_scope_bodies} — [None] when every [Local_scope]
    body in [llc] is pure, else a description of the first violation. [hoist_cross_statement_cse]
    uses it to guard its own precondition (it is the one pass that can move an effect out of a
    scope, so an impure body reaching it would be laundered past the codegen gate); tests use it to
    assert that a body was left where it was. *)

val validate_scope_bodies : t -> unit
(** gh-ocannl-584: enforces the scope-purity contract stated at {!scalar_t.Local_scope} — a scope
    body's only effect is on the locals it owns (its own scope id, plus ids [Declare_local]d
    lexically within it). Raises [Invalid_argument] on anything else inside a [Local_scope] body at
    any nesting depth: a tensor-node write ([Set], [Set_from_vec], [Set_dynamic], [Zero_out],
    [Tile_mma]), a [Set_local] of a sibling or enclosing scope's local, a [Workgroup_barrier], or a
    [Staged_compilation]. Applied at both ends of the pipeline — {!optimize} on the way in (before a
    pass can launder a violation out of any body) and [C_syntax.compile_proc] on the way out. The
    raw analysis entry points {!analyze_proc} / {!specialize_proc} deliberately do not validate,
    being the probes that must stay conservative on IR they may not trust. The optimization pipeline
    satisfies the contract by construction. *)

val validate_scan_loops : Tnode.Placements.t -> t -> unit
(** gh-ocannl-696: the well-formedness contract of {!t.Scan_loop} -- one state node DECLARED virtual
    per carried pair, ids pairwise distinct across the list, inits free of carried state and of the
    scan's own index, each [next] written exactly once as a top-level body statement and read only
    by later statements, no write of a [prev], no binder or reference of a carried id beyond its
    scan, no opaque callback inside, and a non-empty range. Raises [Invalid_argument] naming the
    scan and the clause. Like {!validate_scope_bodies} it runs at both ends of the pipeline:
    {!optimize} on the way in, backend codegen on the way out. *)

val validate_parallel : Tnode.Placements.t -> t -> unit
(** Backend-independent well-formedness of hardware annotations (axis-types proposal §2); a no-op
    for all-[Serial] code. Raises [Invalid_argument] on structural violations: nonzero [from_], more
    than 3 slots per kind, annotated loops inside [Local_scope] bodies, barriers under divergent
    extents or [If] guards, writes to materialized nodes not nested under annotated loops covering
    {e every} active (non-unit) hardware dimension — launch dimensions are global to the kernel, so
    an uncovered dimension executes the write once per hardware index — and whole-node [Zero_out] of
    materialized nodes in multi-threaded kernels (nesting never distributes it). Does not prove
    iteration independence: an annotating pass proves it for the annotations it mints, and the
    renderer asks {!unseparated_thread_write} of every binding it actually emits. *)

val validate_parallel_classified : Tnode.Placements.t -> t -> unit
(** Internal backend-facing variant of {!validate_parallel}; transports a validation
    [Invalid_argument] as a typed {!Schedule_outcome.Illegal_schedule}. *)

val guard_annotated_extents : should_guard:([ `Grid | `Workgroup ] -> bool) -> t -> t
(** Wraps bodies of annotated loops whose extent is below their slot's launch dimension in
    [If (index < extent)] guards, for the kinds the backend binds in hardware. *)

(** {2 Optimization} *)

type virtualize_settings = {
  mutable enable_device_only : bool;
  mutable max_visits : int;
      (** Per-cell read multiplicity cap for inlining: a node with a cell read more than this many
          times (as bounded by the affine access relations, gh-554) is never virtualized unless the
          computation is simple or a one-hot selector producer. *)
  mutable max_inline_reduction : int;
      (** Recompute-cost cap for inlining: a node whose setters have enclosing reduction loops
          (loops not appearing in the setter's indices) with a trip-count product exceeding this
          value is never virtualized. Negative values disable the cap. *)
  mutable max_inline_fanin : int;
      (** Transitive fan-in cap for inlining (gh-573): a node whose fully-inlined computation would
          read more than this many distinct materialized nodes — accumulated through chains of
          virtual producers, per setter — is materialized instead. Bounds the per-consumer
          recomputation of accumulation chains (e.g. a transformer's residual stream), which the
          per-node visit and reduction caps cannot see. Negative values disable the cap. *)
  mutable inline_scalar_constexprs : bool;
  mutable inline_simple_computations : bool;
  mutable inline_complex_computations : bool;
}

val virtualize_settings : virtualize_settings

type traced_array = {
  tn : Tnode.t;
  mutable has_assignment : bool;
      (** The code contains a [Set] or [Set_from_vec] of the node ([Zero_out] is tracked separately
          as [zeroed_out]). Structural replacement (gh-554) for the retired concrete-index tracer's
          per-cell assignment table; per-cell facts are answered by affine queries over the access
          relations instead. *)
  mutable zero_initialized_by_code : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;
      (** The node is read before it is written (i.e. it is recurrent): its entry values are
          consumed, so it is an input of the routine ([input_and_output_nodes]) and not eligible for
          buffer aliasing. For a node that owns a buffer, the verdict is strict of the
          read-modify-write exemption (gh-ocannl-618): a read at its enclosing statement's write
          position still consumes the entry value unless a prior definite write covers its cells —
          the exemption applies to the visit-counting placement heuristics, not to the interface.
          The strict classification closes over the SETTLED placements (in [reconcile_traced_store],
          since placement and legality decisions after [decide_placements] can still flip a
          candidate non-virtual), and a flipped node is promoted [On_device] — an entry-consuming
          node must own a persistent buffer, not [Local] scratch; a node that stays virtual is
          exempt by construction — it has no interface, and the virtualizer's partial-write
          producers depend on that freedom. *)
  mutable read_only : bool;
      (** Surprisingly, the notions of read-only and of constant memory mode come apart: small
          hosted constants are not read-only because they are initialized on devices by being
          assigned to; and a volatile memory mode is read-only from the devices' perspective. *)
  mutable is_scalar_constexpr : bool;
      (** True only if the tensor node has all axes of dimension 1, is either zeroed-out or assigned
          before accessed, is assigned at most once, and from an expression involving only constants
          or tensor nodes that were at the time is_scalar_constexpr. *)
  mutable is_accessing : bool;
      (** False only if the tensor node is built from index embeddings and scalar constant
          expressions. *)
  mutable is_complex : bool;
      (** True only if the tensor node is built from a genuinely complex scalar computation (one
          that accesses other non-constexpr computations). Sharing a loop symbol with another tensor
          does not, by itself, make a node complex (see #134). *)
  mutable prefers_virtual_one_hot : bool;
      (** True when at least one setter for this tensor is a one-hot selector assignment, i.e. a
          [Cmpeq] between the embedded range iterator and a loop-variable-free expression. When
          [has_non_one_hot_setter] is false this tensor is exempt from the visit-count
          [Never_virtual] rule (task-73617488). *)
  mutable has_non_one_hot_setter : bool;
      (** True when at least one setter is NOT a one-hot selector (including [Set_from_vec]). A
          tensor with [prefers_virtual_one_hot && not has_non_one_hot_setter] is the candidate for
          the one-hot virtualizer exemption. *)
  mutable is_range_producer : bool;
      (** True when at least one [Set] assigns this tensor from a bare [Embed_index] scalar, i.e.
          the tensor is a [Range_over_offsets] producer. Used by the indirect arm of
          [is_one_hot_selector_assignment] to prove that a [Get(rtn, [k])] will inline to
          [Embed_index k] rather than arbitrary values (task-73617488). *)
  mutable inline_reduction_extent : int;
      (** The largest product of trip counts of loops that enclose one of the node's setters without
          appearing in its indices (i.e. reduction loops). Inlining the computation replays these
          loops at every read site; compared against [virtualize_settings.max_inline_reduction]. *)
  mutable read_by_other : bool;
      (** True when some statement other than the node's own setters reads the node. Unlike the
          read-multiplicity metric, same-cell reads count, while a setter's own read-modify-write
          does not. Gates the recompute-cost guard: a node never read in the routine has no inlining
          cost, so it must stay eligible for virtual dead-code elimination. *)
  mutable setter_reads : Set.M(Tnode).t list;
      (** Per setter statement ([Set]/[Set_from_vec]), the tensor nodes its right-hand side reads —
          including reads inside [Local_scope] bodies, excluding the node's own read-modify-write
          self-reads. Analysis fact behind the transitive inline-fanin guard (gh-573), which takes
          the per-setter maximum (a read of one cell executes one setter's computation). *)
  mutable inline_fanin : int;
      (** The transitive inline fan-in [decide_placements] computed for this node under the current
          placements (at least 1); multiplies into [fc_recompute_cost]. *)
}
[@@deriving sexp_of]

val get_node : (Tnode.t, traced_array) Base.Hashtbl.t -> Tnode.t -> traced_array
val optimize_integer_pow : bool ref

type traced_store = (Tnode.t, traced_array) Base.Hashtbl.t [@@deriving sexp_of]

type optimize_ctx = {
  computations : (Tnode.t, (Indexing.axis_index array option * t) list) Base.Hashtbl.t;
      (** The computations (of the tensor node) are retrieved for optimization just as they are
          populated, so that the inlined code corresponds precisely to the changes to the arrays
          that would happen up till that point. Within the code blocks paired with an index tuple,
          all assignments and accesses must happen via the index tuple; if this is not the case for
          some assignment, the node cannot be virtual. Currently, we only allow for-loop symbols in
          assignment indices of virtual nodes.

          A stored computation is a NAMED COMPUTATION, NOT A SNAPSHOT (gh-617, decided as
          recompute-at-read): inlining evaluates it at the consumption site with whatever its
          materialized inputs hold at that moment, so a write to one of its leaves between the
          deferring routine and a consuming read is observed by the splice — whereas the
          materialized reading of the same program snapshots the leaf at the deferring routine's
          execution point. The recompute-vs-materialize semantics is deliberately not fixed at this
          level; users select the reading via the memory-mode intent and via routine boundaries
          (routine execution is manual). See "Recompute-at-read" in docs/lowering_and_inlining.md.
      *)
  placements : Tnode.Placements.t;
      (** Per-compilation-lineage memory-mode resolution
          (docs/proposals/context-scoped-memory-modes.md): the pipeline's placement decisions
          (Virtual / Local / On_device) land here, seeded by and never written back to the tnodes'
          declared intent ({!Tnode.field-memory_mode_intent}). *)
  alias_candidates : Hash_set.M(Tnode).t;
      (** gh-ocannl-489 liveness-based buffer aliasing: nodes the memory planner may place at
          overlapping byte ranges within the routine's working pool (decided per compile, before
          codegen). Codegen must not emit the [restrict] qualifier for these parameters — whether a
          candidate pair actually shares bytes is settled only at link time, and an aliased
          [restrict] pair is a miscompile. *)
  inline_preferences : Hash_set.M(Tnode).t;
      (** gh-555: the [Inline] half of the per-lineage inlining decision vector. A node recorded
          here is exempt from the heuristic virtualization caps ([virtualize_max_visits],
          [virtualize_max_inline_reduction], [virtualize_max_inline_fanin]) — the caps are priors of
          the default decision policy, not legality; the legality rejections and observability
          pessimizations still apply. The [Materialize] half of the vector is a pre-seeded
          [On_device] decision in [placements] (see [Context.decide_materialized] /
          [Context.decide_inline]). *)
}
[@@deriving sexp_of]

val empty_optimize_ctx : unit -> optimize_ctx

val copy_optimize_ctx : optimize_ctx -> optimize_ctx
(** A shallow-copy fork of the lineage state ([computations] and [placements] tables): the copy sees
    everything decided so far; its later mutations are invisible to the original and to sibling
    copies. Backend [compile] forks the incoming context's [optimize_ctx] through this, so sibling
    candidate compiles from one frontier are hermetic. *)

val decide_materialized : ?provenance:int -> optimize_ctx -> Tnode.t list -> unit
(** Records an [On_device] decision for each node this lineage has not already resolved otherwise —
    the "materialize this node" move of the placement lattice. Nodes already resolved to [Virtual] /
    [Local] / [Effectively_constant] keep their resolution: decisions are final within a lineage.

    [Context.decide_materialized] is the context-level form (it forks the lineage first) and is what
    ordinary [Assignments] callers want. This raw form serves the paths holding an [optimize_ctx]
    directly: the analyze-only entry points, and hand-built [optimize] calls in tests — for which no
    context-level form can work, since the [?prelowered] seam replaces the context's lineage state
    with the optimized record's own [optimize_ctx]. *)

(** Granularity of the XOR remap applied to a swizzled node's minor axis (gh-ocannl-481 item 3, D1).
    Both flavors are per-row bijections of the minor axis, so the IR-level semantics are identical;
    they differ in the unit the XOR permutes and therefore in which access pattern they de-conflict.
*)
type swizzle_kind =
  | Swizzle_elem
      (** Element-granularity XOR: [P*C + col] renders as [P*C + (col lxor (P land (C-1)))]. Spreads
          same-column scalar reads of consecutive rows across banks; the flavor the scalar and
          register-blocktiled staged kernels want. *)
  | Swizzle_b128
      (** 16-byte-unit XOR: the column's 16-byte-unit index is XORed with the low bits of the row
          prefix, leaving the offset within the unit alone. This is the CUTLASS-style layout
          [ldmatrix] wants — its 8 per-phase row addresses are 16-byte-aligned, so only a remap that
          keeps 16-byte units intact can both de-conflict them and stay loadable. Requires the row's
          byte length to be a multiple of 16 and a power of two in 16-byte units. *)
[@@deriving sexp, compare, equal]

type flip_candidate = {
  fc_tn : Tnode.t;
  fc_flip : [ `Materialize | `Inline ];
  fc_recompute_cost : int;
}
[@@deriving sexp_of]
(** gh-555: one searchable inlining decision dimension of a compile — a node whose placement the
    default policy decided, together with the flip a search can try and the recompute-cost bound of
    the virtual placement (reduction extent × per-cell read multiplicity × transitive inline
    fan-in). [`Materialize] flips a node the policy left virtual (via
    [Context.decide_materialized]); [`Inline] flips a node materialized by the heuristic caps (never
    by legality or observability), via [Context.decide_inline]. An [`Inline] flip's legality is
    settled only when the virtualizer replays: a rejected flip reproduces the materialized
    placement. *)

type pipelined_tile = { pt_depth : int; pt_rotor : Indexing.symbol } [@@deriving sexp_of]
(** gh-487: a software-pipelined (double-buffered) staged tile — codegen allocates [pt_depth]
    rotating copies of the tile and renders every access with a buffer-selection term rotated by the
    [pt_rotor] loop counter: reads select copy [rotor mod depth], writes copy
    [(rotor + 1) mod depth] (the schedule emits the loads one iteration ahead), and writes outside
    the rotor loop (the prologue load) select copy 0. The IR keeps the tile's single-copy dims and
    indices — the rotation is a physical-layout choice like {!type-swizzle_kind}, invisible to
    IR-level semantics — so the pipelined rendering is bitwise identical to the unpipelined one. *)

type optimized = {
  traced_store : traced_store;
  optimize_ctx : optimize_ctx;
  llc : t;
  merge_node : Tnode.t option;
  workgroup_shared : Set.M(Tnode).t;
      (** [Local]-memory-mode nodes to be placed in workgroup-shared memory ([__shared__] /
          [threadgroup]) instead of kernel-local arrays. Populated by schedule transforms; empty for
          unscheduled code. See docs/proposals/axis-types-for-loops.md. *)
  simdgroup_fragments : Set.M(Tnode).t;
      (** [Local]-memory-mode accumulator tiles whose init-load, serial reduction and store-back
          form one per-simdgroup fragment lifetime. Backends without a fragment rendering ignore the
          marking and use the ordinary local-array code; Metal maps the marked region to a
          persistent [simdgroup_matrix] array. *)
  swizzled : swizzle_kind Map.M(Tnode).t;
      (** Nodes stored in an XOR-swizzled layout (docs/proposals/tensorize-mma.md, "Swizzled
          staging"), keyed by the remap's granularity ({!type-swizzle_kind}): codegen remaps every
          element access [flat = P*C + col] (with [C] the minor dim, [P] the linearized prefix) to a
          per-row permutation of [col] — a bijection on the buffer, so the IR-level semantics are
          unchanged; only the physical layout differs, spreading same-column accesses across
          shared-memory banks. Populated by [Schedule.Stage ~swizzle]. Renderings that assume a
          row-major layout must decline swizzled nodes; the tile-MMA intrinsic arms decline
          [Swizzle_elem] and may consume [Swizzle_b128] through [ldmatrix]-style loads. *)
  pipelined : pipelined_tile Map.M(Tnode).t;
      (** gh-487: workgroup-shared staged tiles rendered as [pt_depth] rotating buffer copies (see
          {!type-pipelined_tile}). Populated by [Schedule.Stage ~pipeline_depth] with depth > 1;
          codegen multiplies the tile's allocation by the depth and rotates a buffer-selection
          offset with the [pt_rotor] loop counter. Renderings that assume single-copy storage
          (vectorized/contiguous multi-element accesses) must decline pipelined nodes. *)
  zero_fringe : Set.M(Tnode).t;
      (** Schedule-minted staged tiles whose whole index space is safe to read: slots outside the
          staged source region (edge tiles of a non-dividing or padded staging, gh-ocannl-485) hold
          0 — the add-reduce accumulation identity — written by the load nest's [Where]-form edge
          guards or by the host-side constant packing. [Schedule.Tensorize] consults this to
          discharge pad guards on the intrinsic path. *)
  flip_candidates : flip_candidate list;
      (** gh-555: the searchable inlining decision dimensions of this compile, most expensive first,
          as decided at the whole-routine specialization (schedule-transform copies inherit the
          whole-routine list). Excluded: nodes never assigned or never read, scalar constexprs and
          pure one-hot selector producers, and nodes placed by legality, intent or observability
          rather than the heuristic policy. *)
  spliced_rbw : Set.M(Tnode).t;
      (** gh-610 review round 6: the nodes whose [read_before_write] was set by the FINAL-code
          reconciliation (a spliced read preceding, or not definitely covered by, the routine's own
          writes) — as opposed to the raw analysis' uncovered-read classification, which also flags
          every pure input. [Backends]' prior-context demand keys on this set: a reconcile-flipped
          node's entry value must already live in the linked context, while raw-classified inputs
          keep the assignments layer's curated exclusions. *)
}
[@@deriving sexp_of]

type footprint = {
  fp_total : int;  (** [fp_working + fp_constants]: the number a memory budget is compared to. *)
  fp_working : int;
      (** Bytes of the working (non-constant) pool as the arena planner would lay it out. Equals
          [fp_dedicated] when there is no liveness plan (config [buffer_aliasing] off, code opaque
          to the liveness fold, or a layout over the per-pool cap). *)
  fp_constants : int;  (** Bytes of the constant / read-only pool, always bump-packed. *)
  fp_dedicated : int;  (** What [fp_working] would be with every node on its own bytes. *)
  fp_planned : int;  (** How many working nodes carried a live span, i.e. were arena-eligible. *)
  fp_nodes : int;  (** In-context nodes scored (working + constants). *)
}
[@@deriving sexp_of, equal]
(** gh-ocannl-498: the backend-agnostic byte footprint implied by an {!optimized} routine's
    placement vector. The allocator-side scorer lives in [Backends.score_footprint]. *)

val optimize :
  optimize_ctx ->
  unoptim_ll_source:(PPrint.document -> unit) option ->
  ll_source:(PPrint.document -> unit) option ->
  name:string ->
  Indexing.static_symbol list ->
  t ->
  optimized

type analysis
(** Decision-independent analysis of a lowered routine (gh-555 step 1): the structural per-node
    facts and the lazily-materialized affine access metrics — everything the optimization pipeline
    consumes that does not depend on the lineage's placement decisions. *)

val analyze_proc : Indexing.static_symbol list -> t -> analysis
(** Compute the analysis once for a routine. [optimize] is [analyze_proc] followed by
    [specialize_proc] (plus the pretty-printing callbacks). *)

val specialize_proc : optimize_ctx -> analysis -> optimized
(** The decision-dependent tail of the pipeline: placement decisions ([decide_placements] under the
    given lineage's placements and inline preferences), virtualization, cleanup, simplification and
    CSE. Cheap to replay per candidate over one shared [analysis] (gh-555): sibling calls with
    hermetic [optimize_ctx] forks (see [copy_optimize_ctx]) produce hermetic [optimized] results —
    the traced store is record-copied per call. *)

val validate_virtualization_decision_coverage : Tnode.Placements.t -> t -> unit
(** The gh-ocannl-805 seam assertion run after virtualization and before cleanup: every
    tensor-buffer read ([Get] / [Get_dynamic]) in a statement cleanup will keep, including reads
    inside a kept inlined [Local_scope] body, must name a node with a placement decision. Reads in a
    virtual candidate's setter are excluded because cleanup drops the whole setter without visiting
    its right-hand side. Exposed so hand-built-IR tests can inject the otherwise-unreachable
    negative control directly at this boundary. *)

val analysis_cache_stats : unit -> int * int
(** gh-560: [(hits, misses)] of the process-global analysis cache consulted by [optimize]: sibling
    candidate compiles of one routine share one [analyze_proc] result — keyed by a canonical digest
    of the raw lowered code and the static indices (tensor nodes and static symbols by identity,
    loop binders and local-scope ids alpha-renamed) — and replay only [specialize_proc]. Cumulative
    counters, for tests and diagnostics. *)

val clear_analysis_cache : unit -> unit
(** Drops the analysis cache's entries (the stats persist). Entries retain their routines' lowered
    code, hence their tensor nodes; the cache clears itself before accessibility snapshots
    ({!Tnode.print_accessible_headers}) and callers that tear down a session
    ([Tensor.unsafe_reinitialize]) clear it to release the nodes promptly. Never needed for
    correctness: entries keyed by stale nodes cannot alias fresh ones ([Tnode.uid] is never reused).
*)

val reads_scope_before_set : scope_id -> t -> bool
(** [reads_scope_before_set id body] returns [true] if [id] is read (via [Get_local]) before the
    first definitely-executed [Set_local id] in [body]. Use this at code-generation time to decide
    whether a [Local_scope] or [Declare_local] declaration needs a zero initializer. *)

val simplify_llc : Indexing.static_symbol list -> t -> t
(** Top-down algebraic simplification with interval-driven comparison folding (in particular, it
    erases [If] guards whose conditions the loop extents prove). The interval environment is
    narrowed by every enclosing [If] condition that is a conjunction of integer-affine index
    comparisons (gh-ocannl-566), so a guard the statement guard proves folds too — what is
    simplified under a condition is valid only where that condition holds. Called internally by
    [optimize]; exposed for [Schedule.apply], whose transforms construct guards after the pipeline's
    simplify already ran (docs/proposals/schedule-ir-optops.md §2), and for testing. Pure and
    idempotent. *)

val rewrite_one_hot_reductions : ?static_indices:Indexing.static_symbol list -> t -> t
(** gh-343: rewrites the narrow one-hot embedding pattern -- an [Add] reduction over a loop variable
    [k] whose body selects an embedding-table row via [k == index_expr] (a logical one-hot) -- into
    a guarded dynamic gather ({!Get_dynamic}) that reads the table row at [index_expr] directly,
    with an in-range guard returning 0 out of [\[0, vocab_size)] to preserve the one-hot semantics.
    The guard is constructed generically and interval analysis
    (docs/proposals/interval-analysis-scalar-t.md) erases the conjuncts it can prove -- from the
    index precision's machine range, loop extents seeded from [static_indices], and settled
    per-tensor bounds ({!Tnode.bounds_state}).

    gh-466: also rewrites the {e transposed} one-hot pattern -- the embedding-table gradient
    [for k in \[0, V): tn[.., k, ..] += (k == index_expr) * g] where the loop variable indexes the
    written tensor itself -- into a guarded dynamic scatter-accumulate ({!Set_dynamic}):
    [if in_range(index_expr): tn[.., index_expr, ..] += g], dropping the O(V) per-position work
    (llm.c's deterministic encoder backward, docs/research/llmc-lessons.md B5). The enclosing
    position loops keep their original serial order and the schedule analyses never parallelize over
    a dynamically-written node, preserving determinism without atomics.

    Unmatched or unsupported reductions are left unchanged. Called internally by [optimize] between
    [simplify_llc] and [eliminate_common_subexpressions]; exposed for testing. *)

val eliminate_common_subexpressions : t -> t
(** Eliminates common subexpressions within each statement's scalar expression tree. Replaces
    duplicate [Local_scope] nodes (structurally identical modulo [scope_id]) with [Get_local]
    references to the first occurrence. Called internally by [optimize]; exposed for testing. *)

val hoist_cross_statement_cse : t -> t
(** Hoists shared [Local_scope] computations from sibling statements to the enclosing scope. When
    two or more sibling statements share an alpha-equivalent [Local_scope] node, the computation is
    extracted as a [Declare_local] + body preceding the first user, and all occurrences are replaced
    with [Get_local]. *)

val input_and_output_nodes : optimized -> (Set.M(Tnode).t * Set.M(Tnode).t) * Tnode.t option
(** Inputs are the materialized read-only and read-before-write (within the code) non-constant
    non-merge nodes. They are inputs in a broad sense, as they could be recurrent nodes or
    parameters. Outputs are all the materialized nodes written-to by the code. The last returned
    component is the input merge node, if used in the code. Reads entering the code only through a
    cross-routine inlined computation count: the traced store is completed from the final optimized
    code (gh-610). *)

val loop_bounds : t -> (Indexing.symbol * (int * int)) list
(** All [For_loop] bindings within the code (loop symbols are unique within a routine), with
    inclusive iteration bounds — the box environment for {!Affine} queries, and what
    {!peel_accum_nest} classifies its guard symbols against: one absent from this list is bound
    outside every loop (a static index parameter, a runtime extent) and cannot select among any
    level's iterations. *)

val scope_value_syms : t -> (int, Indexing.symbol list) Base.Hashtbl.t
(** The value-dependence symbols of statement-level scalar scope-locals, whole-code: per scope id,
    the union of the symbols its statement-level assignments depend on, transitively through
    [Get_local] references (such a local may be assigned in one statement and read in another).
    Assignments inside [Local_scope] bodies are not recorded — a scope id is re-instantiated at
    every use site with per-site loop symbols, and scope-internal flow is covered lexically by the
    value scans. Consumed by the setter value scans so a value routed through a scope-local is not
    laundered of its symbols (gh-494 per-thread value-variance). *)

val scalar_value_syms :
  locals:(int, Indexing.symbol list) Base.Hashtbl.t -> scalar_t -> Indexing.symbol list
(** Loop symbols a scalar expression's value depends on syntactically — index symbols of reads,
    embedded indices, dynamic-index sub-expressions — resolving scope-locals through [locals] (from
    {!scope_value_syms}). *)

val affine_accesses : t -> Tnode.t Affine.access list
(** gh-494 waypoint 1: the routine's tensor-node accesses as explicit affine relations
    ({!Affine.access}), extracted from (typically optimized) code, in program order (a statement's
    right-hand-side reads precede its write; [Local_scope] bodies are descended into at their use
    site; [Tile_mma] is traversed through its scalar [fallback]). Not represented: scope-locals,
    merge-buffer reads, and opaque [Staged_compilation] — callers needing exhaustiveness must check
    for the latter separately. *)

val buffer_access_spans : stmt_serial:bool -> t list -> (Tnode.t, int * int) Base.Hashtbl.t option
(** gh-ocannl-489 liveness-based buffer aliasing: per-tnode access span over the final
    (post-schedule, post-fission) code of a routine, as a closed interval of positions; the input is
    the routine's kernels in execution order (a singleton when not fissioned). With
    [stmt_serial:true] every top-level statement gets its own position — sound only for backends
    where consecutive top-level statements of one compiled procedure are fully synchronized (the C
    backends); with [stmt_serial:false] all statements of a segment share one position, since GPU
    kernels have no grid-wide synchronization between top-level statements. Returns [None] when the
    code contains [Staged_compilation] (opaque accesses: no aliasing plan can be trusted). *)

val sink_zero_outs : t -> t
(** gh-ocannl-489 follow-up: sinks each top-level [Zero_out] to just before the first later
    top-level statement accessing the zeroed node ([Train.grad_update]'s up-front [zero_grads] block
    otherwise starts every gradient's live span at that block, nesting the backprop chain's
    intervals and defeating the arena planner). Sound: a [Zero_out] commutes with statements not
    accessing the node; it never crosses such an access, a [Staged_compilation], or a
    [Workgroup_barrier]. Apply to whole-routine code BEFORE scheduling/fission. *)

(** {2 Printing}

    Both dumps render a [Constant] through {!Utils.decimal_float_literal}: a floating literal that
    parses back to exactly the double it names, so the dump distinguishes [-0.0] from [+0.0] and a
    constant whose 17th significant digit matters from one whose does not (gh-ocannl-713). The dumps
    are the surface a constant bug is chased on, so normalizing the values away there costs a
    session before anyone suspects the view. The C-dialect spellings — [INFINITY], the hexadecimal
    literal for an f32 tie — belong to {!C_syntax.c_float_literal} and are deliberately not carried
    here: a dump is not C. *)

val code_hum_margin : int ref

val function_header_doc :
  ?name:string -> ?static_indices:Indexing.static_symbol list -> unit -> PPrint.document

val get_ident_within_code : ?no_dots:bool -> ?blacklist:string list -> t array -> Tnode.t -> string

val to_doc_cstyle :
  ?name:string -> ?static_indices:Indexing.static_symbol list -> unit -> t -> PPrint.document
(** Adheres more to the C syntax, outputs implicit type casts. *)

val to_doc :
  ?name:string -> ?static_indices:Indexing.static_symbol list -> unit -> t -> PPrint.document
(** Adheres to the %cd syntax. *)

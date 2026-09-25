(** Affine legality queries over access index vectors (gh-ocannl-494, manifesto §6 waypoints 1-2).

    Loop nests are generated from einsum projections, so every access in optimized [Low_level.t]
    code carries its affine index map ([Indexing.axis_index array]) natively. This module is the
    query engine over those maps: the procedural schedule-legality analyses (the default annotator's
    per-nest hazard agreement rule, the cross-nest alignment rule, [C_syntax.parallel_grid_safe]'s
    shared rule, the covering checks) are conservative special cases of the decision procedures
    here, and are being re-derived as queries — one implementation of "is this access pattern
    race-free" instead of three.

    Scope: linear-integer reasoning over box domains — per-axis linear Diophantine equations with
    gcd/interval infeasibility and forced-equality derivation via the mixed-radix injectivity
    criterion (the same criterion as {!Indexing.affine_injective}). No full Presburger machinery:
    every query form needed so far is decided (or conservatively declined) at this level. Any
    component the engine cannot interpret ([Sub_axis], [Concat], dynamic indices) contributes no
    information, which errs on the side of declining — soundness is preserved by construction. *)

(** {2 The pair-conflict query}

    Can two accesses touch a common cell from different "threads"? Thread identity is a tuple of
    parallel loop indices; the query works over two copies of the iteration space (the [`Left] and
    [`Right] access's enclosing loops), sharing the symbols that are equal across concurrently
    executing threads (static indices, loops enclosing the parallel region — a join separates their
    iterations). *)

type verdict =
  | Disjoint  (** The two accesses never touch a common cell at all. *)
  | Same_thread
      (** Common cells occur only when every paired parallel symbol is equal — conflicts are
          confined to a single thread, where program order applies. *)
  | Cross_thread of string
      (** A cross-thread conflict is possible, or the engine cannot rule one out; the payload is the
          witness/explanation (the axis or symbol pair that failed). *)
[@@deriving sexp_of]

val pair_conflict :
  range:(Indexing.symbol -> (int * int) option) ->
  dup_left:(Indexing.symbol -> bool) ->
  dup_right:(Indexing.symbol -> bool) ->
  pairs:(Indexing.symbol * Indexing.symbol) list ->
  left:Indexing.axis_index array ->
  right:Indexing.axis_index array ->
  verdict
(** [pair_conflict ~range ~dup_left ~dup_right ~pairs ~left ~right]: verdict on whether the accesses
    with index vectors [left] and [right] (over the same tensor node; rank-padded with
    [Fixed_idx 0]) can touch a common cell from different threads. [range s] gives the inclusive
    iteration bounds of loop symbol [s] ([None] for static/unknown symbols). [dup_left]/[dup_right]
    select the symbols iterated independently by each side (its enclosing loops within the analyzed
    parallel region); other symbols are shared — equal across concurrently executing threads.
    [pairs] is the thread identity: the parallel symbols of the left copy paired with those of the
    right copy (for same-nest analyses, pairs of the form [(p, p)]).

    Sound and conservative: [Disjoint] and [Same_thread] are proven; everything else is
    [Cross_thread]. *)

val separates :
  range:(Indexing.symbol -> (int * int) option) ->
  concurrent:(Indexing.symbol -> bool) ->
  syms:Indexing.symbol list ->
  idcs:Indexing.axis_index array ->
  bool
(** The separation query: does an index vector tell apart the iterations of a set of loop symbols?
    Where {!pair_conflict} asks whether two accesses of DIFFERENT program positions can collide,
    this asks the same engine about ONE access taken twice — the instance-vs-instance form of the
    question: two instances of the same statement, iterating [concurrent] symbols independently, can
    address a common cell of [idcs] only if they agree on every symbol of [syms].

    [concurrent] must cover every symbol whose value may differ between the two instances, not only
    those of [syms]: with [idcs = acc[w1 + w2]] and [syms = [w1]], holding [w2] equal would "prove"
    that a common cell forces [w1] equal, while instances [(0, 1)] and [(1, 0)] share [acc[1]].
    [syms] is then the subset the caller needs told apart. *)

val axis_index_to_string : Indexing.axis_index -> string
(** The rendering of one index component the engine's witnesses use, for messages that quote a cell
    beside a witness in one spelling. *)

val separation_failure :
  range:(Indexing.symbol -> (int * int) option) ->
  concurrent:(Indexing.symbol -> bool) ->
  syms:Indexing.symbol list ->
  idcs:Indexing.axis_index array ->
  string option
(** {!separates} with the engine's witness: [None] where the vector separates [syms], otherwise the
    explanation {!pair_conflict} gives for the cross-thread conflict — the symbol a common cell does
    not force equal — for a refusal message that names what failed. *)

val within_box :
  range:(Indexing.symbol -> (int * int) option) ->
  dims:int array ->
  Indexing.axis_index array ->
  bool
(** [within_box ~range ~dims idcs]: does the index vector address a cell INSIDE the [dims] box for
    every valuation of its symbols within their ranges? The interval companion of {!covers_box},
    which asks about a bijection onto the box; this asks only that nothing leaves it.

    Access validity, as distinct from the distinctness {!separates} proves. A symbol with no range
    (a static index parameter) and a component the engine cannot interpret both answer [false]: an
    unknown value can be anywhere, and this query is only ever used to license moving an access to
    where a guard no longer covers it. *)

(** {2 The peel-guard legality query}

    Whether a guard may join the levels that [Low_level.peel_accum_nest] peels down to the
    accumulation base — the one place the dead-level and lane-sharing hazards of hoisting the
    accumulated cell's open/close outside a guard are decided (gh-ocannl-722). The full hazard
    analysis, the [Lane_private_if_separated] escape and its required {!separates}/{!within_box}
    side conditions are documented in the implementation. *)

type peel_guard =
  | Confined_to_peel  (** Every symbol the guard mentions is peeled or bound outside every loop. *)
  | Lane_private_if_separated of Indexing.symbol list
      (** Legal exactly if the accumulated cell {!separates} these enclosing loop symbols. *)
  | Not_peelable of string  (** With the reason, for a decline log. *)

val peel_guard :
  loop_bound:(Indexing.symbol -> bool) ->
  peeled:(Indexing.symbol -> bool) ->
  guard_syms:Indexing.symbol list ->
  peel_guard

(** {2 The covering query} *)

val covers_box :
  range:(Indexing.symbol -> (int * int) option) ->
  dims:int array ->
  Indexing.axis_index array ->
  bool
(** [covers_box ~range ~dims idcs]: whether the index vector [idcs], as its symbols range over their
    (loop) bounds, enumerates every cell of the [dims] box exactly once — a bijection onto the box.
    This is the write-dominance building block: a covering unguarded write rewrites the whole array.
    Requirements: each symbol used at most once across the vector; per axis, a zero-based
    full-extent iterator, a mixed-radix affine combination of zero-based symbols whose radix chain
    exactly composes to the axis dimension, or [Fixed_idx 0] on a unit axis. Generalizes (and is
    checked against) the procedural per-axis rule of [C_syntax.first_access_standalone_covering]. *)

(** {2 Counting} *)

val fiber_cardinality :
  domain:(Indexing.symbol * int) list ->
  Indexing.axis_index array ->
  [ `Exact of int | `At_least of int ]
(** [fiber_cardinality ~domain idcs]: how many points of the loop box [domain] (symbol, width pairs)
    map to one given cell in the image of the access map [idcs] — the per-cell visit count of a read
    access, and the recompute cost per read site of inlining a setter. Domain symbols absent from
    the map contribute the product of their widths; when the map is injective on its mentioned
    symbols ({!Indexing.affine_injective}) that product is the exact fiber size of every image cell
    (cells outside the image have zero), otherwise it is a lower bound. *)

val fiber_cardinality_ub :
  domain:(Indexing.symbol * int) list ->
  Indexing.axis_index array ->
  [ `Exact of int | `At_most of int ]
(** Upper-bound companion of {!fiber_cardinality}: at most how many points of the loop box [domain]
    map to any single cell of the image of [idcs]. Domain symbols absent from the map contribute the
    product of their widths exactly; when the map is injective on its mentioned symbols that is the
    whole fiber and the bound is exact. Otherwise the mentioned symbols' contribution is bounded per
    component (see the implementation), and the smallest component-wise bound is taken. *)

(** {2 Projection-level predicates}

    Queries about the affine LHS map of a projection: {!is_surjective} decides whether every LHS
    position is written — used to elide zero-initialization before assignments; {!is_injective}
    whether no LHS position is written twice — used with {!is_surjective} to elide initialization
    entirely. *)

val is_surjective : Indexing.projections -> bool
(** Proves exact dense coverage of the row-major LHS buffer from the product-component extents.
    Handles signed affine sums, shared symbols, [Sub_axis] strides and independent complete
    concatenations. Returns [false] for holes, unknown/runtime extents, unsupported
    concat-coordinate mixtures or arithmetic outside the proof's safe integer range. When RHS maps
    are supplied, every target block must have a viable producer (segment-choice proof capped at
    1024 combinations); an empty RHS array asks only about the mathematical LHS map. *)

val is_injective : Indexing.projections -> bool

(** {2 Access records}

    The extraction target for [Low_level.affine_accesses] (gh-494 waypoint 1): each tensor-node
    access as an explicit affine relation — the enclosing loop box, the index map into the node's
    cells, and the program placement. ['tn] abstracts the tensor-node type to keep this module below
    [Tnode] in the dependency order. *)

(** gh-561: one component of an access's program position ({!field-a_path}). [Stmt] components are
    per-[Seq] statement indices; the other constructors encode {e intra-statement} order, which bare
    statement positions cannot express. Constructor order is execution order, so the derived compare
    makes lexicographic path comparison program order: a statement's [Cond] ([If] condition)
    evaluates before its [Body], and a [Set]-family statement's [Rhs] (right-hand side and
    dynamic-index reads, including [Local_scope] bodies inlined there) executes before its [Write].
    Every access path ends in [Cond], [Rhs] or [Write], and nothing extends a path past [Write]
    (writes have no interior), so a write's path is never a proper prefix of a read's. *)
type path_comp =
  | Stmt of int  (** Statement index at one [Seq] nesting level. *)
  | Arg of int
      (** A [Local_scope] occurrence's per-statement evaluation position: two scope bodies inlined
          into one statement's scalar tree extend {e distinct} bases, so their interior components
          never interleave. Sibling positions are deliberately {e incomparable} in the visibility
          rule: evaluation order among one statement's operands is not modeled, so no cross-operand
          ordering is claimed. *)
  | Cond  (** Inside an [If] statement's condition. *)
  | Body  (** Inside an [If] statement's guarded body. *)
  | Rhs  (** Inside a [Set]-family statement's right-hand side (or a [Set_local]'s). *)
  | Write  (** The [Set]-family or [Zero_out] statement's own write. *)
[@@deriving sexp_of, compare, equal]

val same_statement : path_comp list -> path_comp list -> bool
(** Whether two accesses sit in the same [Set]-family statement — their paths agree above their
    final component (the write's [Write] against the statement's own direct [Rhs]/[Cond] reads;
    reads nested deeper, e.g. in a [Local_scope] body's statements, have longer paths and do not
    match). This is the path-level counterpart of the [a_stmt_write] subordination. *)

val stmt_head : path_comp list -> int
(** The top-level statement index of a path, [-1] when the whole routine is a single statement (its
    accesses' paths start with an intra-statement component). *)

type 'tn access = {
  a_tn : 'tn;
  a_map : Indexing.axis_index array;
      (** The affine map from the loop box into the node's cells. Empty and standing for every cell
          when [a_whole]. *)
  a_write : bool;
  a_dynamic : bool;
      (** The effective cell is not statically known (dynamic gather/scatter): the map has a
          placeholder component, so queries must not interpret it. *)
  a_whole : bool;  (** A whole-node access ([Zero_out]). *)
  a_vec_last : bool;
      (** A vectorized write ([Set_from_vec]): the last map component is the base of a run along the
          minor axis, not a single cell — queries must treat that component as opaque. *)
  a_vec_len : int;
      (** The run length of a vectorized write along the minor axis; [0] unless [a_vec_last]. *)
  a_guarded : bool;  (** Under an [If] guard: executes conditionally, never a definite write. *)
  a_gated : bool;
      (** Under a SCALAR gate (gh-ocannl-1016): a read inside a [Where] arm or a gated operand
          ({!Ops.binop_conditionality} [Gated_second], {!Ops.ternop_conditionality}
          [Cond_and_one_arm]) — evaluated only when the gate selects it, a conditionality inside one
          statement that [a_guarded] (the [If] statement) does not carry. A [Local_scope] body
          clears it: scope definitions are hoisted out of the expression and execute
          unconditionally. Always [false] for a write (statements are never gated). *)
  a_rmw : bool;
      (** The statement also reads [a_tn] on its right-hand side (an accumulation): the write
          carries a reduction dependence — an order-sensitive legality dimension (the determinism
          contract): a loop carrying only reduction edges may be reassociated (vectorized) under an
          explicit license, but never parallelized. *)
  a_val_syms : Indexing.symbol list;
      (** Writes only: loop symbols the written value depends on syntactically (index symbols of rhs
          reads, embedded indices, dynamic-index sub-expressions). Direct dependence only — a chain
          through another node's cells is not tracked. *)
  a_stmt_write : Indexing.axis_index array option;
      (** Reads only: the index map of the enclosing [Set]/[Set_from_vec]/[Set_dynamic] statement's
          write when the read occurs in that statement's right-hand side; [None] elsewhere ([If]
          conditions and [Local_scope] inner statements carry their own statements' writes). The
          subject of the read-modify-write exemption ([Low_level.rmw_exempt]): matching by statement
          subordination rather than by program path, so a guarded body's write cannot alias its [If]
          condition's read (they share a path). *)
  a_loops : (Indexing.symbol * (int * int)) list;
      (** Enclosing loops, outermost first, with inclusive iteration bounds. *)
  a_path : path_comp list;
      (** Lexicographic program-order position: statement indices per [Seq] nesting level,
          interleaved with intra-statement components ({!path_comp}) at each statement the traversal
          descends into. *)
}
[@@deriving sexp_of]

(** {2 Statement effects}

    gh-ocannl-1016: the sibling view of {!type-access}, from the same walk
    ([Low_level.affine_relations]) — what a statement does that no tensor-node access carries. A
    "between two points, what else runs?" question is a query over the two lists: a tensor write is
    an access, everything else that can make code motion observable is a row here. A consumer
    classifies the kinds with an exhaustive match, so a new kind forces each consumer's decision. *)

type 'tn effect_kind =
  | Local_write
      (** A [Set_local], or a [Scan_loop]'s carried-state initialization or per-iteration rotation:
          a scope local changes value. *)
  | Local_declare  (** A [Declare_local]: a statement-level local comes into scope. *)
  | Scope_body
      (** A [Local_scope] occurrence: its body's statements run at this position, hoisted ahead of
          the statement whose expression holds it, and unconditionally (whatever scalar gate the
          occurrence sits under). The body's own accesses and effects are rows too, under the path
          this row names ({!in_scope_body}). *)
  | Barrier  (** A [Workgroup_barrier]. *)
  | Staged  (** [Staged_compilation]: opaque code, whose accesses are not enumerated at all. *)
  | Mma
      (** A [Tile_mma]: its [fallback]'s accesses stand for its footprint, but the construct itself
          is emitted as one tensor-core operation — a code-motion barrier, not the loop nest the
          fallback spells. *)
  | Merge_read of 'tn
      (** A read of the merge buffer sourced from the node ([Get_merge_buffer]): a separate,
          transient input — not an access of the node's own buffer, which is why it is no
          {!type-access}. *)
[@@deriving sexp_of]

type 'tn statement_effect = {
  e_kind : 'tn effect_kind;
  e_loops : (Indexing.symbol * (int * int)) list;
      (** Enclosing loops, outermost first, as {!field-a_loops}. *)
  e_guarded : bool;  (** Under an [If] guard, as {!field-a_guarded}. *)
  e_gated : bool;
      (** Under a scalar gate, as {!field-a_gated}: only a [Merge_read] can be; a [Scope_body] is
          not, its body being hoisted out of the gate. *)
  e_path : path_comp list;
      (** The program position, in the components of {!field-a_path}: a statement-shaped effect
          (every kind but [Scope_body] and [Merge_read]) sits at its statement's [Write] — a
          [Set_local]'s right-hand-side reads at the same statement's [Rhs] order before it; a
          [Merge_read] at its reading expression's position, like a read access; a [Scope_body] at
          the base its body's rows extend (the [Arg] component). *)
}
[@@deriving sexp_of]

val loops_live : (Indexing.symbol * (int * int)) list -> bool
(** Whether every enclosing loop of an access or effect executes: a row under a dead loop
    ([to_ < from_]) never happens. *)

val in_scope_body : path_comp list -> bool
(** Whether a path lies inside a [Local_scope] body (it crosses an [Arg] component). *)

val within_statement : write:path_comp list -> path_comp list -> bool
(** [within_statement ~write path]: whether [path] lies inside the [Set]-family statement whose own
    write sits at [write] — its right-hand side, the [Local_scope] bodies inlined there, or the
    write itself. [false] when [write] is not a write position. *)

val may_touch_same_cell :
  ?static_range:(Indexing.symbol -> (int * int) option) -> 'tn access -> 'tn access -> bool
(** Whether two accesses (of the same node) can touch a common cell, each access taken over its
    whole loop box — the two sides' iterations paired independently, including iterations of loops
    the sides share (the accesses need not be simultaneous, so a shared loop's symbol varies
    independently between one side's visit and the other's). Symbols bound by neither side's loops
    (static indices) are shared parameters, equal on both sides, bounded by [static_range] when
    known. Conservative: [false] only when {!pair_conflict} proves disjointness; uninterpretable
    access kinds (dynamic, whole-node, vectorized) count as overlapping. *)

val vec_runs_disjoint : minor_dim:int -> 'tn access -> bool
(** Whether the runs of a vectorized access ([a_vec_last]) are pairwise disjoint in the node's flat
    cell space — the access then touches exactly [base image * a_vec_len] distinct cells
    (gh-ocannl-578). [minor_dim] is the node's minor-axis extent. Conservative: [false] when any of
    the sufficient conditions (documented in the implementation) is not proved. *)

val read_covered_before :
  ?thread:(Indexing.symbol -> bool) ->
  ?static_range:(Indexing.symbol -> (int * int) option) ->
  read:'tn access ->
  writes:'tn access list ->
  unit ->
  [ `Covered | `Unknown of string ]
(** The containment query. [read_covered_before ~read ~writes ()]: is every cell the [read] access
    can touch necessarily written before the read executes — the dominance side of dependence
    analysis, and the fifth decision procedure (gh-494 waypoint 2). Unlike the ∃-flavored
    {!pair_conflict} (negated to prove disjointness), containment is a ∀∃ query — for every read
    instance there must exist a covering write instance. Visibility is same-common-iteration program
    order over {!path_comp} paths; loop-carried coverage is declined, conservatively. With [?thread]
    naming the parallel (thread-identity) symbols, [`Covered] proves the cell side of the
    per-thread-copy transform. The full variable treatment, the cross-statement value side
    condition, and the union rule for partial covers are documented in the implementation.

    Guarded writes are the caller's choice: include them to mirror guards-taken analyses
    ([Low_level.trace_node_facts] and the coverage queries take guards unconditionally), pre-filter
    [a_guarded] for execution-accurate coverage. [writes] must be accesses of the same node as
    [read]. *)

(** {2 Crosscheck}

    Config [legality_crosscheck]: when enabled, the call sites swapped onto the queries also run the
    legacy procedural analysis and compare. A query stricter than the procedural answer raises —
    either a query precision regression or a latent unsoundness of the procedural rule, both needing
    eyes. A query more permissive than the procedural answer is the expected precision gain, logged
    to stderr for review. *)

val crosscheck_enabled : bool lazy_t

val crosscheck :
  site:string ->
  context:string ->
  procedural_safe:(unit -> bool) ->
  query_safe:bool ->
  witness:string ->
  unit

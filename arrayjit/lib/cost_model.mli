(** Analytic cost model, the extraction half (gh-ocannl-491 task 1): per-kernel compulsory memory
    footprints, arithmetic-op counts, arithmetic intensity, and the roofline lower-bound time under
    advisory envelope constants ({!Backend_intf.hardware_limits}'s [peak_flops] /
    [peak_memory_bandwidth]).

    The analysis is pure and backend-free: it reads a (typically optimized) {!Low_level.t} through
    the affine-access artifacts of gh-ocannl-494 ({!Low_level.affine_accesses},
    {!Affine.fiber_cardinality}) and returns numbers; envelope constants are plain floats supplied
    by the caller. The model ranks candidate schedules — it does not predict runtimes.

    Approximation contract (each direction is explicit, all biases point the same way):

    - Byte counts are upper bounds on compulsory traffic (distinct cells touched, perfect-cache
      assumption): per-access image cardinalities are exact for injective interpretable maps, and
      for vectorized runs whose bases are provably non-overlapping ({!Affine.vec_runs_disjoint});
      non-injective maps, guarded ([If]) accesses (counted guards-taken), other vectorized runs, and
      uninterpretable components ([Sub_axis]/[Concat]/dynamic indices — whole-node fallback) only
      over-count, as does summing multiple same-direction accesses of one node (a union bound,
      capped by the node's size) — except that a direction whose accesses are all exact and pairwise
      provably disjoint ({!Affine.may_touch_same_cell}) sums exactly (gh-ocannl-578). Conditional
      evaluation also over-counts: a read the renderers may skip — a [Where] arm's or a gated right
      operand's ([&&]/[||]/a gate) inline read — or any access under a dead loop keeps its direction
      approximate, since the image can exceed what executes; a read inside a [Local_scope] body
      under such an operand is certain, the body being hoisted out of the conditional
      (gh-ocannl-637). [fp_approx] is [false] only when the count is exact.
    - The op count is an upper bound in the same guards-taken sense, and counts every scalar
      [Unop]/[Binop]/[Ternop] evaluation as one "FLOP" regardless of precision or integerness —
      except the two-operation ternaries [FMA]/[Mul3], which count two (matching [peak_flops]'
      FMA-counted-as-two convention, so an FMA-form kernel scores the same as its mul+add form);
      selection ops [Arg1]/[Arg2]/[Identity] count zero; [Tile_mma] counts its [2*m*n*k]
      multiply-adds once per tile, not per cooperating lane.
    - The one under-counting escape hatch is opaque code — [Staged_compilation] statements and
      merge-buffer reads are invisible to the analysis — and it is flagged: when [opaque] is set,
      the upper-bound reading of the byte and op counts no longer holds. *)

type node_footprint = {
  fp_read_bytes : int;  (** Distinct cells read (including accumulation reads) times byte width. *)
  fp_write_bytes : int;  (** Distinct cells written times byte width. *)
  fp_rmw_bytes : int;
      (** The subset of [fp_write_bytes] written by read-modify-write accumulations ([a_rmw]). *)
  fp_approx : bool;
      (** [true] when any contributing count is an upper bound rather than exact (see the module
          contract for the causes). *)
}
[@@deriving sexp_of]

type summary = {
  per_node : (Tnode.t * node_footprint) list;  (** In first-access program order. *)
  read_bytes : int;  (** Sum of [fp_read_bytes] over [per_node]. *)
  write_bytes : int;  (** Sum of [fp_write_bytes] over [per_node]. *)
  flops : int;  (** Whole-kernel arithmetic-op count (loop extents times per-statement ops). *)
  flops_approx : bool;
      (** [true] when guarded ([If]) code contributed (guards-taken bound), or a short-circuiting
          form was charged in full while executing only in part — a [Where] whose arms carry any
          inline cost (only one arm's expression executes), a gated right operand with nonzero
          inline cost (gh-ocannl-578). A cost residing in a hoisted [Local_scope] body executes
          unconditionally — every renderer emits scope definitions before the statement's expression
          — so it is charged exactly and never flags (gh-ocannl-637). *)
  opaque : bool;
      (** [true] when the code contains [Staged_compilation] or merge-buffer reads: some traffic and
          ops are invisible to the analysis, so counts may UNDER-estimate. *)
}
[@@deriving sexp_of]

val analyze : Low_level.t -> summary
(** The whole-kernel extraction: footprints from {!Low_level.affine_accesses} image cardinalities,
    op counts from a loop-nest walk. Input is typically post-[optimize] code — the analysis charges
    whatever materialization the code actually performs, so virtual/inlined nodes never appear and
    recomputation is charged as ops, not bytes. *)

val total_bytes : summary -> int
(** [read_bytes + write_bytes]: the "bytes moved" denominator of arithmetic intensity — an
    accumulation's cells are charged once in each direction. *)

val arithmetic_intensity : summary -> float
(** FLOPs per byte moved, [flops / max 1 (total_bytes)]. *)

val footprint_approximate : summary -> bool
(** [true] when any per-node footprint is an upper bound rather than exact ([fp_approx]) — the byte
    counts over-approximate while the op count may still be exact. *)

type floor = { fr_flops : int; fr_bytes : int; fr_exact : bool } [@@deriving sexp_of]
(** Lower bounds on arithmetic work and compulsory traffic — the dual extraction (gh-ocannl-514
    phase 3), for bounding {e every completion} of a partial placement vector where {!analyze}
    upper-bounds one concrete candidate. [fr_exact] is [false] when some flooring-to-zero occurred
    (a guarded body, a non-exact access image, opaque code): the floor is then sound but not tight.
*)

val completion_floor : ?open_placement:(Tnode.t -> bool) -> Low_level.t -> floor
(** The floor of the roofline legs over placement completions, with every approximation biased
    {e down} — the exact dual of {!analyze}'s contract:

    - [fr_flops]: guarded ([If]) bodies count zero (guards-never-taken, dual to guards-taken); the
      short-circuiting forms count only their certain part — [Where] its condition plus the cheaper
      arm's inline work (rendered as [?:]), [And]/[Or] the left operand's (rendered as [&&]/[||]),
      both sides' hoisted scope bodies in either case (they execute regardless, gh-ocannl-637), the
      [Arg1]/[Arg2] projections only the selected operand (the discarded one is never rendered,
      hoisted definitions included); opaque code counts zero (an under-count is sound in this
      direction); [Tile_mma] keeps the lane-cooperative attribution, exact when the lane binding is
      in scope. Statements producing an [open_placement] node — the [Set] family and [Tile_mma] with
      an open accumulator — count zero: an inline completion instantiates the producer only at
      surviving consumer sites, possibly {e fewer} cells than the setter loop covers, so
      "recomputation only adds ops" does not hold and the producer's whole effect attributes to the
      open placement. Call this on the {e all-materialized} specialization of the decision surface,
      where every open node's work sits in its own producer statement.
    - [fr_bytes]: per node and direction, the sum of the exact images when they are pairwise
      provably disjoint (a disjoint union attains its sum — both extractions then agree,
      gh-ocannl-578), otherwise the largest exact image (a union is at least its largest member —
      dual to the upper extraction's capped sum; a second nonzero contribution then marks the floor
      loose); guarded, non-exact, dead-loop-enclosed, conditionally-evaluated (a [Where] arm's or an
      [And]/[Or] right operand's inline read — not a hoisted scope body's) and open-producer-operand
      accesses contribute zero — their execution is not certain in every completion. Nodes with
      [open_placement] contribute zero.

    {e Committing} a placement decision is re-evaluation with the narrowed [open_placement]: the
    suppression sets only shrink, so the floor is monotone in refinement — the property that lets a
    bound prune as commitments accumulate. (An incremental per-node delta cannot be sound in
    isolation: a Materialize commitment also makes its producer's operations and operand reads
    certain, not just the node's own traffic.) Committing to Inline tightens nothing for now: a
    nonzero recompute floor needs lower-bound multiplicity metrics, deferred until the driver proves
    the need.

    A [max (fr_flops/peak_flops) (fr_bytes/peak_bandwidth)] roofline over these floors
    ({!roofline_seconds}) lower-bounds every completion — the two legs' minima need not be
    simultaneously achievable for the max of the two to be sound. *)

val approximate : summary -> bool
(** [flops_approx || footprint_approximate]: some count is an upper bound rather than exact. Such
    counts are still upper bounds (unlike under [opaque]), but a candidate whose guards mostly fail
    can carry counts far above the work it performs — quantities derived from an approximate count
    (achieved throughput, the roofline bound vs. a measurement) are not evidence about the hardware.
    Exactness is per leg: op counts ([flops_approx]) and byte counts ([footprint_approximate]) go
    approximate independently. *)

val roofline_seconds :
  ?peak_flops:float ->
  ?peak_memory_bandwidth:float ->
  flops:int ->
  bytes:int ->
  unit ->
  float option
(** The roofline lower-bound time: [max (flops/peak_flops) (bytes/peak_memory_bandwidth)] over the
    envelope constants present ([peak_flops] in FLOP/s, [peak_memory_bandwidth] in bytes/s — the
    advisory {!Backend_intf.hardware_limits} fields); [None] when neither is given. Monotone:
    raising either constant never increases the bound. A lower bound only up to the model's
    upper-bound byte/op counts — rank with it, do not predict. *)

(** {2 The cost of one inlined computation (gh-ocannl-637 Part 2)}

    The cost model's account of what a virtual node costs to recompute at ONE read site — the flops
    and bytes of one instantiation of its computation, with the exactness the extraction tracks. The
    reader's multiplicity stays outside, as in the virtualizer's traced proxy
    ([inline_reduction_extent × read multiplicity × inline_fanin]); these queries replace the extent
    × fan-in factor with the modeled arithmetic, and {!Low_level.recompute_pricer} feeds them to the
    flip-candidate ordering (gh-ocannl-555), the memory-budget planner and, through
    [fc_recompute_cost], footprint-scoped materialization (gh-ocannl-616). Priced per Part 1: a
    hoisted scope body under a [Where] counts as what executes. *)

type recompute = {
  rc_flops : int;  (** Operations of one instantiation. *)
  rc_bytes : int;
      (** Bytes one instantiation reads from nodes other than the computed one — its own cells are a
          scope local once inlined. Distinct cells within the instantiation's own body; an expanded
          producer's traffic ({!recompute_cost}) multiplies by the enclosing trip count. *)
  rc_approx : bool;  (** Either count is an upper bound rather than exact ({!analyze}'s flags). *)
  rc_opaque : bool;  (** Opaque code: the counts may under-estimate. *)
}
[@@deriving sexp_of]

val template_cost : self:Tnode.t -> ?at:Indexing.axis_index array -> Low_level.t -> recompute
(** One stored template body ([optimize_ctx.computations]' [(at, body)] entry) as one instantiation:
    sibling setters (a shared-loop template) are dropped as instantiation drops them, the loops
    binding a bare iterator position of [at] — the ones the ordinary point read substitutes away —
    collapse to a single iteration, so a reduction loop is the only trip count left, and the passes
    the emitted code receives after virtualization run over the result — the simplifier and the
    scalar CSE, which a stored template predates: a single-assignment scope under a [Where] arm
    collapses into the arm's expression (conditional, hence a bound), two alpha-equivalent scope
    bodies execute once. What the query cannot see is the substitution a reader applies, and where
    it changes the count the result is only a bound ([rc_approx]): a symbol occurring inside an
    affine position of [at], or repeated across bare positions (a diagonal producer's consistency
    guards), may be bound or left free with its loop range-guarded or guarded, depending on the
    reader's index; a packed-uniform producer ([Set_from_vec]) inlines as the lane-extract form
    rather than its vector store; and a scope local read without its definition in the priced code
    stands for hoisted work the price cannot see. A consumer instantiating over a sub-image that
    collapses a further loop (gh-ocannl-616) applies that correction itself. Reads of other virtual
    nodes count as reads here; {!recompute_cost} expands them. *)

val recompute_cost : Low_level.optimize_ctx -> Tnode.t -> recompute option
(** The transitive cost of one inlined computation of the node, summed over its stored templates
    (every component of a multi-setter node replays at a read site, guarded — and its body hoists,
    so all execute): each template's {!template_cost}, plus, for every read of a producer with a
    stored template that is not committed non-virtual in the lineage, that producer's own recompute
    per enclosing iteration (its read cells then are not traffic). [None] when the node has no
    stored computation — a materialized node prices through {!producer_cost}. Memoized per lineage:
    partially apply to the context once per compile. Cycles (a node reached through its own
    producers) are cut at the node, priced once. *)

val producer_cost : self:Tnode.t -> Low_level.t -> recompute option
(** The per-cell cost of a materialized producer in optimized code — the twin of {!recompute_cost}
    for a node a heuristic cap materialized, whose computation was never stored: per setter
    statement of the node, {!analyze} over the code pruned to that setter (the loops enclosing it
    survive, everything else is dropped) divided (rounding up) by the distinct cells it writes,
    summed over the setters — re-inlining a multi-setter node replays every component at a read
    site, guards selecting the value while the hoisted bodies all execute. Its virtual producers are
    already inlined there, so the count is transitive by construction; a packed-uniform setter, or a
    scope local whose hoisted definition the pruning left behind, makes it a bound, as in
    {!template_cost}. [None] when the code sets the node nowhere. *)

module Calibration : sig
  (** The calibration TSV schema (config [autotune_calibration_file], gh-ocannl-491 task 4) and the
      envelope fitter over it (gh-ocannl-514 phase 0): one row per timed candidate, the model's
      inputs and roofline score next to the measured time. This module is the schema's single owner
      — rows are emitted through {!to_line} (by [Autotune]) and read back through {!of_line} (by
      [tools/fit_envelope.exe]), so writer and reader cannot drift apart. *)

  type row = {
    backend : string;
    digest : string;  (** The candidate's digest tag, already shortened at emission. *)
    routine : string;
        (** The tuned computation's name (gh-ocannl-635) — the very name [Autotune.tune] gives its
            candidate compiles (its own [?name], or the comp's block comment when no name was
            passed), so it also names the candidate's generated sources. It is READ from there
            rather than derived a second time alongside them (gh-ocannl-669): the two agreed only
            while no compile of a search was ever given a name, and a divergence would have been
            silent — every row naming the block comment while the kernels it measured carried the
            other name. Without the column a row (and every fit witness quoting one) identifies the
            candidate but not the kernel it measured, which is what forced
            [tools/calibrate_bandwidth.exe] to reconstruct per-kernel rates outside the schema.
            Empty for rows recorded before the column existed; see {!of_line}. *)
    label : string;
    measured_ms : float;
    model_ms : float option;
        (** The roofline bound under the envelope constants in force at recording time; [None] when
            the model had no coverage (opaque code or no envelope constants). *)
    kernels : int;
    flops : int;  (** Aggregated over the candidate's kernels, like [bytes]. *)
    bytes : int;
    flops_approx : bool;
        (** The op count is an upper bound rather than exact (guards-taken, any kernel): the compute
            leg of the fit skips this row; likewise [bytes_approx] for the memory leg
            ({!footprint_approximate}). Approximate legs are recorded for divergence analysis but
            excluded from envelope fitting. *)
    bytes_approx : bool;
    opaque : bool;
  }
  [@@deriving sexp_of]

  val to_line : row -> string
  (** Tab-separated, no trailing newline. [of_line (to_line r)] recovers [r] up to float formatting:
      [measured_ms] and [model_ms] record 6 decimals, {e floored} rather than rounded — a stored
      time never exceeds the true measurement, so constants fit from a file remain conservative with
      respect to the original in-process measurement (round-to-nearest could overstate a 5 us
      kernel's time by a fitting-relevant 1e-4 relative). Tabs and newlines in the free-text
      [routine] and [label] columns become spaces: a name carrying one would otherwise split its row
      into fragments no reader can parse. *)

  val of_line : string -> row option
  (** [None] on malformed lines (wrong column count, unparseable numbers). Rows in the 11-column
      schema that predates {!field-routine} (gh-ocannl-635) still parse, with an empty [routine]:
      they carry every exactness flag the fit needs, so a file accumulated across builds keeps its
      old rows — they merely cannot name their computation. *)

  val qualified : routine:string -> label:string -> string
  (** How a row names itself in a report: ["routine/label"], or just the label when the routine is
      unknown (a legacy row). *)

  val row_name : row -> string
  (** {!qualified} of the row's own fields — the naming used by the {!fit} witnesses. *)

  type fit = {
    fit_backend : string;
    fit_rows : int;  (** Non-opaque, positively timed rows; each leg uses its exact subset. *)
    fit_opaque : int;  (** Opaque rows, excluded — the model never scores them. *)
    fit_flops_approx : int;
        (** Among [fit_rows]: rows the compute leg skips (approximate op count — guards-taken
            over-counting can fake a throughput above any hardware peak, and one such row would
            inflate the envelope machine-wide); likewise [fit_bytes_approx] for the memory leg.
            Exactness is per leg, so a row with an exact op count but a multi-read (approximate)
            footprint still feeds the compute leg. *)
    fit_bytes_approx : int;
    fit_multi_kernel : int;  (** Among [fit_rows]; see {!fit} for the aggregate caveat. *)
    fit_violations : int;
        (** Fully-exact rows whose recorded [model_ms] exceeds [measured_ms]: the envelope in force
            when they were recorded understated this machine's peaks. Rows with an approximate leg
            are not counted — their exceedance may reflect over-counting instead, mirroring the
            runtime warning's gating. *)
    fit_fission_slack : (float * string) option;
        (** The uniform factor (> 1) applied to both legs so the aggregate sufficient condition
            holds on every multi-kernel row (see {!fit}), and the row forcing it; [None] when the
            per-leg maxima already suffice (or a leg is absent). *)
    fit_peak_flops : (float * string) option;
        (** The fitted constant (fission slack included) and the binding row's
            [routine/label (digest)] ({!row_name}); [None] when no scoreable row has a positive
            count for the leg. *)
    fit_peak_memory_bandwidth : (float * string) option;
  }
  [@@deriving sexp_of]

  val fit : row list -> fit list
  (** Grouped by backend in order of first appearance. The fitted constants are the tightest
      envelope under which the roofline bound respects every row it can be audited on: per row,
      [bound <= measured] requires [peak >= counts/measured] on each leg, so each leg starts as the
      maximum achieved [counts/time] over the rows where {e that leg's} counts are exact — the ratio
      is then a throughput the machine demonstrably reached. Multi-kernel rows aggregate per-kernel
      counts, making those maxima necessary for them but not sufficient ([Autotune]'s bound sums
      per-kernel max-of-legs, which can approach twice the aggregate legs on a compute-bound +
      bandwidth-bound mix), so both legs are then raised uniformly by the smallest
      {!field-fit_fission_slack} enforcing the aggregate sufficient condition
      [flops/peak_flops + bytes/peak_memory_bandwidth <= time] on every fully-exact multi-kernel row
      — after which the recomputed bound respects every row it was fit from (an over-counted leg is
      barred from forcing slack: it would inflate both constants). Raising a peak only weakens
      pruning (the bound stays a lower bound); understating one breaks fathoming.

      Sound on the data, not certified for the machine: fitted peaks are floors a kernel
      demonstrably reached, and a future candidate can achieve more. Between refits such a
      candidate's bound can exceed its would-be measured time — caught by the continuous agreement
      check when it is timed, but under [autotune_keep_fraction < 1] it may be model-pre-filtered
      before timing, where the check cannot see it. [tools/fit_envelope.exe]'s [--margin] trades
      pruning strength for headroom against exactly that. *)

  val report : fit -> string
  (** Config-pasteable: [model_peak_*=...] lines under [#] comment lines naming the binding rows and
      any fission slack (config comments must be whole lines). Printed constants carry a ~2e-6
      relative bump so that 7-significant-digit truncation cannot land below the implied minimum. *)
end

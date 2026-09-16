(** The online-softmax attention rewrite (gh-ocannl-483): the first member of the algebraic-rewrite
    tier over lowered code ({!Rewrites}), a pattern-directed substitution that CHANGES the
    computation where the schedule transforms only rearrange it.

    The composed attention [softmax (q * k^T) * v] lowers to a max-reduction and a sum-reduction
    over the key axis with two elementwise nests between them, then a reduction over the key axis
    against [v]. Two of its intermediates are [seq^2]-shaped and get materialized: the scores are
    read by both reductions, and the probabilities are read once per value-width iteration of the
    final reduction. This pass rewrites two shapes, both gated by the config key [online_softmax]
    (default off, on in the [approximate] profile):

    - The (max, sum-of-exp-shifted-by-max) reduction pair over one axis becomes ONE
      {!Ir.Low_level.t.Scan_loop} per row carrying the running max and the rescaled running sum --
      the online-softmax recurrence, {!Ir.Low_level.t.Scan_loop}'s founding use case -- writing both
      trajectories to the original nodes so every downstream reader is unaffected. The elementwise
      nests in between stay as the definitions of their nodes. This reassociates the normalizer's
      summation, which is why the pass is a numerics policy rather than an optimizer decision:
      results move within rounding.
    - A reduction consuming the normalized probabilities (through elementwise nests only) has the
      loops the probabilities do not index moved innermost, behind one read of the probability cell
      into a scope local. Bitwise exact: every moved loop indexes the reduction's target, so no
      per-cell summation order changes; the read multiplicity that forced the probabilities into a
      [seq^2] buffer is gone, and the virtualizer inlines their whole chain into that one read.

    What stays as it was: the score reduction [q * k^T] keeps its own placement decision -- the
    recompute cap [virtualize_max_inline_reduction] decides whether it is replayed at its two read
    sites (the scan and the hoisted read) or stored once. Recomputing is flash attention's memory
    trade (no [seq, seq] buffer at all); storing measured faster on both cc and Metal at every
    length in benchmarks/report-gh483-online-softmax.md, and a training step's backward reads the
    scores again through cross-routine splicing.

    The pass runs at lowering, ahead of the analyses ({!Rewrites.apply}, from [Assignments.lower],
    over the raw lowered code), so the traced store and the placements see the rewritten routine and
    no analysis digest needs the knob: the code itself carries the decision ([Code_borne]). It is
    idempotent, as the tier's fixpoint requires: its output has no max-reduction nest left to match,
    and a hoisted reduction is no longer a single nest. *)

val enabled : unit -> bool
(** Whether the rewrite applies: the programmatic override if one is set, else the config key
    [online_softmax]. *)

val set_enabled : bool option -> unit
(** Programmatic override of the config key, for experiments and tests; [None] restores the key.
    Takes effect at the next lowering. *)

val reset : unit -> unit
(** Drops the memoized scope-local nodes (the tier's session-reset hook; also run ahead of an
    accessibility snapshot). Sibling lowerings of one program mint the same nodes, which is what
    keeps them on one analysis-cache key; a reset only makes the next lowering mint afresh. *)

val rewrite : Low_level.t -> Low_level.t
(** The pass over raw lowered code (no gate: the tier consults {!enabled}). Every normalizer pattern
    in the routine is rewritten; a routine without one is returned as is. The minted scope-local
    nodes carry memory-mode provenance 483. *)

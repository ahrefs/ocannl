(** The online-softmax attention rewrite (gh-ocannl-483): the first member of the algebraic-rewrite
    tier over lowered code ({!Rewrites}), a pattern-directed substitution that CHANGES the
    computation where the schedule transforms only rearrange it.

    The composed attention [softmax (q * k^T) * v] lowers to a max-reduction and a sum-reduction
    over the key axis with two elementwise nests between them, then a reduction over the key axis
    against [v]. Two of its intermediates are [seq^2]-shaped and get materialized: the scores are
    read by both reductions, and the probabilities are read once per value-width iteration of the
    final reduction. This pass rewrites three shapes, all gated by the config key [online_softmax]
    (default off, on in the [approximate] profile), the third by a second key as well:

    - The (max, sum-of-exp-shifted-by-max) reduction pair over one axis becomes ONE
      {!Ir.Low_level.t.Scan_loop} per row carrying the running max and the rescaled running sum --
      the online-softmax recurrence, {!Ir.Low_level.t.Scan_loop}'s founding use case -- writing both
      trajectories to the original nodes so every downstream reader is unaffected. The chain is
      recognized only at one precision, the scores', and the carried pair lives at that precision
      widened to f32 (f64 stays f64): under narrow scores the state does not round per step, which
      is the one difference from the composed form beyond summation order. The elementwise nests in
      between stay as the definitions of their nodes. This reassociates the normalizer's summation,
      which is why the pass is a numerics policy rather than an optimizer decision: results move
      within rounding.
    - A reduction consuming the normalized probabilities (through elementwise nests only) has the
      loops the probabilities do not index moved innermost, behind one read of the probability cell
      into a scope local. Bitwise exact: every moved loop indexes the reduction's target, so no
      per-cell summation order changes; the read multiplicity that forced the probabilities into a
      [seq^2] buffer is gone, and the virtualizer inlines their whole chain into that one read.

    - The composed backward of that attention, in a routine that also holds the rewritten forward
      (the training step of [Train.grad_update]), gated separately by [online_softmax_backward]
      (default off, on in [approximate]; gh-ocannl-1002). Anchored on the normalizer and the value
      pass the first two shapes matched -- [O] and [v]; [dO] from the reduction [dP += dO * v]; the
      chain from [dP] through [e.grad], [l.grad] and [n.grad = e.grad * e] to the score gradient
      [dS] the two contractions against the scores' operands [q] and [k] read -- it replaces every
      [seq, seq] gradient buffer with a per-row [D = sum (dO * O)] into a minted node and two nests
      recomputing each pair's [p], [dp] and [ds = chain (p * (dp - D))] into scope locals (the
      recovered elementwise chain -- the mask's [where], the scale -- applies to [ds] only, so a
      finite mask fill keeps its probabilities and their [dV]): one over the query rows accumulating
      [q.grad], one over the keys accumulating [k.grad] and [v.grad], each owning what it writes.
      The per-cell summation orders of the three gradients are the composed ones; [ds] reassociates
      the composed [(dP / l + dl) * e], hence the numerics gate. It declines whole -- the backward
      stays composed -- on anything it cannot prove: no normalizer in the routine, a composed max
      gradient, another reader of a consumed gradient node, a requested (materialized) intermediate,
      code the census cannot see in the span, a dead or partial loop, mixed precisions along either
      chain, or an elementwise step between [e / l] and the value pass (active dropout).

    What stays as it was: the score reduction [q * k^T] keeps its own placement decision -- the
    recompute cap [virtualize_max_inline_reduction] decides whether it is replayed at its two read
    sites (the scan and the hoisted read) or stored once. Recomputing is flash attention's memory
    trade (no [seq, seq] buffer at all); storing was faster at every cell measured on cc and Metal
    (benchmarks/report-gh483-online-softmax.md, seq 128 to 1024), and a training step's backward
    reads the scores again through cross-routine splicing.

    The pass runs at lowering, ahead of the analyses ({!Rewrites.apply}, from [Assignments.lower],
    over the raw lowered code), so the traced store and the placements see the rewritten routine and
    no analysis digest needs the knob: the code itself carries the decision ([Code_borne]). It is
    idempotent, as the tier's fixpoint requires: its output has no max-reduction nest left to match
    -- so no normalizer to anchor a backward on either -- and a hoisted reduction is no longer a
    single nest. *)

val enabled : unit -> bool
(** Whether the rewrite applies: the programmatic override if one is set, else the config key
    [online_softmax]. *)

val set_enabled : bool option -> unit
(** Programmatic override of the config key, for experiments and tests; [None] restores the key.
    Takes effect at the next lowering. *)

val backward_enabled : unit -> bool
(** Whether the fused backward applies, where the rewrite does: the programmatic override if one is
    set, else the config key [online_softmax_backward]. Inert while {!enabled} is false. *)

val set_backward_enabled : bool option -> unit
(** Programmatic override of [online_softmax_backward], like {!set_enabled}. *)

val reset : unit -> unit
(** Drops the memoized scope-local nodes (the tier's session-reset hook; also run ahead of an
    accessibility snapshot). Sibling lowerings of one program mint the same nodes, which is what
    keeps them on one analysis-cache key; a reset only makes the next lowering mint afresh. *)

val rewrite : Low_level.t -> Low_level.t
(** The pass over raw lowered code (no gate: the tier consults {!enabled}). Every normalizer pattern
    in the routine is rewritten; a routine without one is returned as is. The minted scope-local
    nodes carry memory-mode provenance ["483:online-softmax-state"], the fused backward's per-row
    [D] (a stored node: [Never_virtual]) ["1002:fused-backward-row-dot"]. *)

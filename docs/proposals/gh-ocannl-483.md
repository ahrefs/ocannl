# The online-softmax attention rewrite

Issue: [#483](https://github.com/ahrefs/ocannl/issues/483) (task 1, the loop-carried recurrence,
landed as [#696](gh-ocannl-696.md))

**Date**: 2026-09-16
**Status**: **landed** — the tier `Ir.Rewrites` and its seam in `Assignments.lower`, the pass
`Ir.Online_softmax`, the config key `online_softmax` (in the `approximate` profile), and
`test/operations/online_softmax.ml`.

## What the rewrite is

The schedule IR's transforms are semantics-preserving rearrangements of a fixed computation. The
attention gap against torch/tinygrad needs a rewrite that *changes* the computation: the composed
`softmax (q * k^T) * v` materializes two `[seq, seq]` intermediates, and flash attention's
online-softmax reformulation is a reassociation no loop transform can reach. This is the first
member of the algebraic-rewrite tier over lowered code — the same species as the one-hot gather
and scatter rewrites (gh-343, gh-466), a larger pattern, and numerics-gated because it moves
results within rounding.

## Where it sits, and why there

**On the raw lowered code, ahead of the analyses.** `Assignments.lower` applies the tier
`Ir.Rewrites` between `to_low_level` and `Low_level.optimize`, so the traced store, the placements
and both digests see the rewritten routine. The tier is generic: a static list of members, each
with its own gate key, run in order to a fixpoint (structural equality) under a round cap that
refuses a member that keeps rewriting its own output. Members feed each other — the probability
hoist already depends on the normalizer having fired, and a rewrite that emits a softmax-shaped
nest would want the normalizer after it — which is what the fixpoint buys and what the member
contract (idempotent; every application consumes a pattern instance) keeps terminating. Three
reasons the tier lives at this level rather than on `Assignments` or after virtualization:

- Every `Accum_op` lowers to one top-level nest with concrete index arrays, so the pattern is a
  deterministic structure of nests — no projections to interpret, and no placement decisions yet
  to vary the shape (the same routine looks different after virtualization depending on the caps).
- It removes nodes' definitions (the max- and sum-reduction pair becomes a scan), which the traced
  store must not have registered as writes; `rewrite_one_hot_reductions`, by contrast, keeps its
  nodes and only changes how a value is read, which is why it can run late.
- The knob then needs no digest: the code itself carries the decision (`Code_borne`), so no
  schedule-cache entry is split for a routine the rewrite never touched. That is also why the key
  is not a `Numerics.t` field: the numerics record is consulted at codegen and by tile-shape
  seeding, never in lowered code, and its fingerprint enters the cache key (gh-ocannl-568).

## The two shapes

Recognition relates nests through the AXES of the nodes they share (a per-node signature of
row/reduced/fixed roles, fixed by the max-reduction's own loops), never through loop symbols,
which every nest mints afresh.

1. **The online normalizer.** `m := max over t of x; n := x - m; e := exp n; l := sum over t of e`
   (with `m`'s neutral fill and `l`'s zeroing) becomes one `Scan_loop` per row carrying `(m, l)`,
   each at its own node's precision widened to f32 — `m' = max(m, x)`, `l' = l * exp(m - m') + exp(x - m')` —
   writing both trajectories to the original `m` and `l` nodes, so every downstream reader is
   unaffected. A prefix of masked keys (`-inf` scores) keeps `m' = -inf`, where the rescaling
   would be `exp(-inf - -inf) = nan`; the rescaling reads both maxima floored at the format's
   lowest finite value, so a masked prefix rescales by `exp 0` against a zero normalizer, the
   first live score rescales the empty prefix by `exp(lowest - x) = 0`, a NaN score poisons as in
   the composed form, and the update emits no comparison (a C compiler's finite-math licence has
   nothing to fold that a finite result depends on). The stored trajectory selects the composed
   form's NaN on `m' = -inf`, the one comparison, deciding only what a row with no live score yet
   stores — so a fully masked row reads NaN in both forms. (The exact `l' + (m' - m')` was tried
   first; the simplifier reassociates it into `(l' + m') - m'`, which cancels catastrophically —
   a live instance of gh-ocannl-998.) The pointwise nests stay as their nodes' definitions.
2. **Hoisting the probabilities.** A reduction `o[.., e] += w[rows, t] * v[..]` whose `w` is
   defined elementwise from a rewritten normalizer has the loops `w` does not index moved
   innermost, behind one read of `w` into a scope local. Bitwise exact — every moved loop indexes
   the target, so no per-cell summation order changes — and it removes the read multiplicity
   (once per value-width iteration) that materialized the probabilities. The virtualizer then
   inlines the whole chain into that one read.

Two-pass rather than the single-pass "rescale the output row inside the scan" form: the same
flops (one score recompute versus a per-step `o *= alpha`), but the value pass stays an ordinary
loop nest — schedulable, and the shape a fused backward recomputes from `(m, l)`.

## What is deliberately left to others

- **The score matrix's placement.** `q * k^T` keeps its own decision: the recompute cap
  `virtualize_max_inline_reduction` decides whether it is replayed at its two read sites (flash
  attention's memory trade: no `[seq, seq]` buffer at all) or stored once; storing measured faster
  on both backends at every length in the report, so recompute is for when the buffer itself is
  the constraint. The rewrite does not
  force it virtual: a lineage decision binds every later routine, and the composed backward reads
  the scores at several sites with value-width multiplicity, so a forced-virtual score chain would
  be replayed `d_v`-fold there. The test pins both readings.
- **A fused backward.** The composed backward reads the forward's intermediates through
  cross-routine splicing (parameter gradients agree, pinned), but its own `[seq, seq]` gradient
  buffers stay. Flash attention's backward recomputes `p = exp(s - (m + log l))` per block from
  exactly the `(m, l)` this forward saves.
- **Block tiling with tensor cores.** The scan is opaque to the schedule ops (gh-ocannl-696), so
  the rewritten attention runs one row per thread with a serial key loop: memory-optimal, not
  compute-optimal. Blocking the key axis and materializing a probability tile for `Tile_mma` is
  the follow-up that would let "the schedule layer still tiles the rewritten loops" hold.
- **Dropout between the probabilities and the value reduction** is handled (the chain from `l` to
  the reduction's operand may pass through any elementwise nests); a mask applied *after* the
  normalization is not recognized as such and simply leaves the shape alone.

## Acceptance criteria

- [x] Recognition over the lowered attention pattern and emission of the online-softmax scan,
      policy-gated (`online_softmax`, off by default, on in `approximate`).
- [x] Parity tests against the composed form (two-sided tolerance claims, device floats off the
      golden): causal, masked key prefixes, two stacked blocks, and training through the composed
      backward; structural pins that the probabilities are never written, that no `[seq, seq]` node
      is written under a cap admitting the head width, and that the scores are the one left above
      it.
- [x] Measured on `gpt2_mini` and on the long-context legs `gpt2_mini_s512` / `gpt2_mini_s1024`
      (cc and Metal, `benchmarks/report-gh483-online-softmax.md`): neutral to +28% on cc, -8% to
      +18% on Metal with the crossover between seq 512 and 1024; storing the scores beats
      recomputing them on both backends.

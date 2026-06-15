# Interval (min/max) analysis over scalar_t

**Date**: 2026-06-12
**Status**: Stub — seeded by the tinygrad deep dive
([a-range-is-not-its-shape](../blog/a-range-is-not-its-shape.md), port area 3). Judged
there the best effort-to-payoff item of the six ports; no blocking dependency.

## Goal

An interval lattice over `Low_level.scalar_t` (and the integer analogue over
`Indexing.axis_index`), in the spirit of tinygrad's `vmin`/`vmax` derived property
(`tinygrad/uop/ops.py`, `_min_max`): `Constant` exact, `Embed_index` bounded by its
loop's statically-known extent, `Get` bounded by dtype range, arithmetic by interval
rules (add endpoints; extremal products; careful div/mod-by-constant cases). Exposed as
an analysis `simplify_llc`'s rewrite arms can consult to discharge validity masks,
prove indices in-bounds, and fold comparisons.

## Key points from the article's analysis

- Four existing efforts independently approximate this one analysis: stage-B
  injectivity in [#133](https://github.com/ahrefs/ocannl/issues/133) (range arithmetic
  over loop extents), read-before-write tracking in
  [#340](https://github.com/ahrefs/ocannl/issues/340), matcher side-conditions in
  [#343](https://github.com/ahrefs/ocannl/issues/343), and the landed surjectivity
  reasoning of [#420](https://github.com/ahrefs/ocannl/issues/420). Intervals are the
  unifying upgrade.
- Sequencing: before the schedule layer
  ([schedule-ir-optops](schedule-ir-optops.md)) — Padto is affordable in tinygrad
  *because* interval reasoning discharges most of the masks it introduces.
- Receiving site exists: `interval_of : scalar_t -> bounds` slots into `simplify_llc`'s
  world; loop extents are statically known from projections.

## Relations

[#133](https://github.com/ahrefs/ocannl/issues/133),
[#340](https://github.com/ahrefs/ocannl/issues/340),
[#343](https://github.com/ahrefs/ocannl/issues/343), landed #420;
[schedule-ir-optops](schedule-ir-optops.md) (downstream consumer).

## Acceptance criteria (for the elaborated proposal)

- [ ] Lattice and rules specified (float vs. index-integer variants; widening not
      needed — loop extents are finite and static).
- [ ] At least one existing approximation (#133's range arithmetic or #343's
      side-conditions) re-expressed through the shared analysis.
- [ ] Caching strategy decided (per-node memo during a `simplify_llc` run).

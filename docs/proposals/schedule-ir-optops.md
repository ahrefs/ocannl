# Schedule IR: OptOps-style loop transforms as values

**Date**: 2026-06-12
**Status**: Stub — seeded by the tinygrad deep dive
([a-range-is-not-its-shape](../blog/a-range-is-not-its-shape.md), port area 1); to be
elaborated when v0.8 tiling work ([#412](https://github.com/ahrefs/ocannl/issues/412))
needs the transform layer.

## Goal

A schedule layer for `Low_level.t`: loop-nest transforms (Split/tile, Swap/interchange,
Unroll, Upcast/vectorize, Padto) represented as *values* — a `(optop, axis_ref, arg)
list` — applied as a pure `Low_level.t -> Low_level.t` pass after virtualization,
Halide-style, rather than tinygrad's destructive mid-pipeline rewrite. A schedule is
then searchable (BEAM over schedule prefixes with on-device timing), cacheable, and
testable independently of the kernel it acts on.

## Key points from the article's analysis

- Rangeify-era tinygrad reduces the whole schedule language to: Split-with-retype,
  Swap, Padto, Nolocals, TC (`spec/tinyspec.tex`; impl `codegen/opt/__init__.py`,
  applied by `apply_opt` in `codegen/opt/postrange.py`, searched by `beam_search` in
  `codegen/opt/search.py` with on-device timing).
- OCANNL seeds exist: `unroll_dims` (unrolled nest generation), `Set_from_vec`/`vec_unop`
  (vector-store primitive), `loop_over_padding_region` (the strip/masking machinery
  Padto needs).
- Prerequisites: axis-type annotations ([axis-types-for-loops](axis-types-for-loops.md))
  so Split has a type to retype to; interval analysis
  ([interval-analysis-scalar-t](interval-analysis-scalar-t.md)) so Padto's masks
  discharge.
- The open design problem to settle in elaboration: **pass ordering against
  virtualization** — inlining changes which loops exist; Padto can flip whether a node
  should inline. Proposed default: virtualize → schedule → simplify, with a measured,
  bounded re-virtualization iteration rather than fixpoint rewriting.
- Contexts-as-values are the autotuner substrate: candidate timings are sibling
  compiles from one frontier, no global device state to isolate.

## Relations

[#412](https://github.com/ahrefs/ocannl/issues/412) (GPU tiling — the transforms),
[watch-ocannl-README-md-347818d3](watch-ocannl-README-md-347818d3.md) (CPU matmul),
[gh-ocannl-267](gh-ocannl-267.md) (Tiramisu — thesis is this layer),
[gh-ocannl-242](gh-ocannl-242.md) (TVM/Ansor lineage),
[gh-ocannl-261](gh-ocannl-261.md) (search/cost functions).

## Acceptance criteria (for the elaborated proposal)

- [ ] `optop` type and schedule application pass specified against `Low_level.t`.
- [ ] Pass-ordering decision (virtualization interaction) recorded with rationale.
- [ ] Search harness scoped (BEAM first; cost models deferred to #261 follow-ups).

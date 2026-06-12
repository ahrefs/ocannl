# Tinygrad Deep Dive: Architecture Comparison and Porting Prospects

**Date**: 2026-06-12
**Status**: Draft proposal — deliverable is a blog article (plus optional appendix), not code.
**GitHub issue**: none, by design — unlike the study family (#242 TVM, #267 Tiramisu,
#261 superoptimizers), tinygrad is a long-standing inspiration source and Łukasz
already did a personal deep dive back in the ShapeTracker era; this proposal is the
post-rangeify revisit, with a publication deliverable.

## Goal

Write a blog article comparing OCANNL and tinygrad, prospecting which tinygrad solutions
are worth porting to OCANNL and in what form. The README lists tinygrad among OCANNL's
inspirations, and ROADMAP v0.9 frames OCANNL's scheduling direction explicitly against
it ("instead of dynamic scheduling as in tinygrad, we can schedule statically by program
search"). The comparison targets *rangeify-era* tinygrad — the representational
convergence (loop-nest IR rather than ShapeTracker stride stacks) is precisely what
makes porting tractable now, and the article should say so: against old tinygrad there
was nothing to port onto.

**Deliverables**:
1. A blog article under `docs/blog/` (cross-posted to lukstafi.github.io), audience:
   compiler-curious ML practitioners; tone and depth comparable to the existing
   shape-semantics series.
2. Optionally, an appendix with additional technical detail (per-port design sketches,
   code pointers on both sides) — either a second part under `docs/blog/` or a
   companion note under `docs/` if it gets reference-manual-ish.
3. Follow-up proposal stubs for the gaps identified below, where the article's analysis
   firms them up.

## The port analysis (article skeleton)

Six areas, in rough order of value-per-difficulty (from a prior analysis session,
2026-06; the article elaborates each):

1. **The OptOps schedule layer** — the big one. Rangeify-era OptOps are loop-nest
   transforms: Split is tiling, Swap is interchange, UNROLL exists (`unroll_dims`),
   UPCAST is vectorization (`Set_from_vec`/`vec_unop` as the seed), Padto is
   pad-to-multiple with validity masks (`loop_over_padding_region` shows the masking
   machinery). These act on `For_loop` nests in `Low_level.t` directly. Missing but
   bounded: (a) axis-type annotations richer than `trace_it`
   (GLOBAL/LOCAL/REDUCE/UPCAST); (b) a transform language kept *separate* from the IR,
   Halide-style — a schedule as a value, `(optop, axis, arg) list`, applied as a
   `Low_level.t -> Low_level.t` pass after virtualization — rather than tinygrad's
   destructive-rewrite style; (c) the search harness (the easy part): BEAM over
   schedule prefixes with on-device timing — and contexts-as-values are a *better*
   autotuner substrate than tinygrad's global device state, since candidate timings are
   sibling compiles from one frontier. Key design interaction to think through in the
   article: pass ordering against virtualization (inlining changes which loops exist,
   but Padto can change whether a node should inline; tinygrad dodges this with
   fixpoint rewriting on one graph, OCANNL must pick an order and occasionally iterate).
2. **AxisTypes** — the prerequisite, needed anyway. Workgroup-level GROUP_REDUCE +
   Barrier + LOCAL-addrspace buffers are the achievable 80% of "thread
   synchronization" (shared-memory reductions, tiled matmuls) — well short of
   grid-level persistent-thread megakernels, which tinygrad's spec does not solve
   either (Barrier is workgroup-scoped; cross-kernel sequencing remains command
   buffers). Making loop-to-hardware mapping explicit in the IR instead of a backend
   convention is one change that unlocks OptOps, shared-memory reductions, and
   eventually the megakernel direction.
3. **min/max interval derivation** — cheapest, pays everywhere. tinygrad threads
   [min, max] through every UOp to discharge validity masks, prove in-bounds, fold
   comparisons. An interval lattice over `scalar_t` (constants exact, `Embed_index`
   bounded by loop range, arithmetic by interval rules) slots into `simplify_llc`'s
   world. Probably the single best effort-to-payoff item.
4. **The shard axis as a derived property** — port the *property*, not the device
   tuples. The good part of tinygrad's multi-device design is the bookkeeping: shard
   axis propagated by local per-op rules (Reshape remaps, Permute follows, Reduce on
   the shard axis annihilates), so the system always knows where data is split. OCANNL
   version: a per-tnode placement annotation threaded through shape
   inference/projections, lowering at boundaries to merge-buffer transfers between
   contexts. Honest port = the propagation algebra plus an explicit placement *policy
   hook* — heterogeneity (a Mac Studio + a MacBook) breaks tinygrad's
   interchangeable-n-tuple assumption, so "which context combines" is a real decision
   the annotation can't make alone.
5. **The Function/Call layer** — mostly maps onto what exists. OCANNL doesn't need
   trace-capture (the DSL builds graphs natively) and has routines + the hazard sets
   for composition. The narrower suggestion: *joint compilation of a lineage* — several
   comps with known hazard edges compiled as one program — plus the Memory-Plan stage
   (lifetime-based reuse of GLOBAL/LOCAL/REG buffers), of which OCANNL's local-scope
   handling is already the REG end.
6. **A declarative rewrite engine** — port the discipline, not the machinery.
   PatternMatcher exists because Python lacks pattern matching; OCaml's `match` in
   `simplify_llc` is already exhaustive, typed, fast. Worth stealing: rules as data
   with provenance, run to fixpoint — relevant precisely as the gather/one-hot collapse
   and its relatives multiply. An e-graph over `scalar_t` is a possible eventual shape;
   lowest urgency.

## Mapping to existing proposals (the article should cite these)

| Port area | Existing proposals | Status / relation |
|---|---|---|
| OptOps schedule layer | [gh-ocannl-412.md](gh-ocannl-412.md) + [watch-ocannl-README-md-347818d3.md](watch-ocannl-README-md-347818d3.md) (v0.8 tiling, GPU+CPU); [gh-ocannl-242.md](gh-ocannl-242.md) (TVM/Ansor — completed write-up, notes the tinygrad-TVM lineage and that schedule search starts mattering at v0.8 tiling); [gh-ocannl-267.md](gh-ocannl-267.md) (Tiramisu — thesis *is* the missing schedule layer); [gh-ocannl-261.md](gh-ocannl-261.md) (search/cost functions) | **Gap**: no proposal for the schedule IR itself |
| AxisTypes | [gh-ocannl-412.md](gh-ocannl-412.md) (single-threaded kernels today; first site of the GLOBAL/LOCAL decision); [gh-ocannl-195.md](gh-ocannl-195.md) (addrspace cousin at the const end); #318 megakernel write-up (landed); [gh-ocannl-263.md](gh-ocannl-263.md) (Flash attention — eventual consumer) | **Gap**: no proposal for axis-type annotations / barriers / LOCAL buffers |
| Interval derivation | [gh-ocannl-133.md](gh-ocannl-133.md) (stage-B injectivity = range arithmetic over loop extents); [gh-ocannl-340.md](gh-ocannl-340.md) (read-before-write scan); [gh-ocannl-343.md](gh-ocannl-343.md) (matcher side-conditions); landed #420 surjectivity reasoning | **Gap**: no proposal; four passes independently approximate it — intervals are the unifying upgrade |
| Shard axis | [gh-ocannl-293.md](gh-ocannl-293.md) cluster: [task-a2c331e9.md](task-a2c331e9.md) (recommends *explicit* `shard_along`/`gather`/`grad_sync` — the propagation-algebra idea is direct input to its **open verdict**; the two compose: annotation as sugar over the explicit primitives); [task-e4003e5f.md](task-e4003e5f.md) (copy-free sharding needs slice-as-alias); [task-2445dd1c.md](task-2445dd1c.md); [concise-merge-buffer-transfers.md](concise-merge-buffer-transfers.md); [distro-feasibility-study.md](distro-feasibility-study.md) (heterogeneity/multi-node end) | Cluster live; verdict pending |
| Function/Call layer | [execution-dependency-tracking.md](execution-dependency-tracking.md) (the hazard substrate; v1.0); [gh-ocannl-344.md](gh-ocannl-344.md) (Memory-Plan half via pool lifetimes) + [gh-ocannl-340.md](gh-ocannl-340.md) (REG end); landed #288 static merge-buffer verification | **Gap**: no proposal for joint lineage compilation |
| Rewrite engine | [gh-ocannl-261.md](gh-ocannl-261.md) (equality saturation/e-graphs already surveyed as the v0.9 "broader rewriting" candidate); [gh-ocannl-296.md](gh-ocannl-296.md) (the audit/doc prerequisite for rules-as-data); motivating rewrites in #343/#133/#134; [cse-alpha-equivalence-soundness.md](cse-alpha-equivalence-soundness.md) (landed — cautionary tale for rule provenance) | Discipline-level port |

## Acceptance Criteria

- [ ] Blog article in `docs/blog/` covering: the representational convergence
      (ShapeTracker → rangeify vs. OCANNL's einsum-derived loop nests), the six port
      areas with honest value-per-difficulty ordering, and what OCANNL gets *for free*
      that tinygrad reconstructs (typed IR layering, contexts as values, static shape
      inference with row polymorphism).
- [ ] Each port area cites the relevant existing proposals (table above) so the article
      doubles as a roadmap index; claims about OCANNL internals verified against
      current code (post broadcast-order reversal — use join/refines vocabulary).
- [ ] The sharding section's conclusion (propagation algebra + policy hook vs. explicit
      primitives) is recorded into [task-a2c331e9.md](task-a2c331e9.md)'s open verdict
      rather than left as prose.
- [ ] Follow-up proposal stubs created for the three gaps (schedule IR / OptOps port;
      AxisTypes; interval analysis) — or an explicit decision not to.
- [ ] Optional appendix: per-port design sketches with code pointers on both sides
      (tinygrad file/class names pinned to a commit; OCANNL functions by name not line).
- [ ] Cross-posted; in-repo `docs/blog/` copy is canonical (per the
      broadcast-aware-shape-inference precedent).

## Scope

In scope: the comparison and porting analysis; reading rangeify-era tinygrad sources
deeply enough to pin claims to commits; seeding follow-up proposals.

Out of scope: implementing any of the ports; benchmarking OCANNL vs tinygrad
performance (a separate, harder artifact); re-litigating closed directions (#341
multi-streaming, #186 dynamic indexing — though the article may cite their
resolutions as design-philosophy data points).

## Risks / precedent

The deep-dive family has mixed survival: TVM (#242) completed; Tiramisu (#267) and
superoptimizers (#261) live; Petalisp/Caten (#306) and IREE (#301) were closed
not-planned. This one is differently situated: a prior personal deep dive
(ShapeTracker era) already happened, so the unknown is what *changed* with rangeify,
not what tinygrad is; and it has a publication deliverable plus direct coupling to
open decisions (a2c331e9 verdict, v0.8 tiling design). Scope discipline matters: the
article is the deliverable; the ports are prospects, not commitments.

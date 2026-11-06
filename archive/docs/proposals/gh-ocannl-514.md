# Schedule inference as branch-and-bound: search in constraint space

**Date**: 2026-08-12
**Status**: Phase 0 implemented (envelope fitting `Ir.Cost_model.Calibration` +
`tools/fit_envelope.exe`, the continuous bound-vs-measured agreement check in `Autotune.tune`,
and this design; staging PR #320, hardened by four review rounds: per-leg count exactness,
fission slack for multi-kernel aggregate rows, floored serialization). Phase 1 implemented:
`Ir.Schedule_space` (placement levels including `Pl_stage_at`; the lazy refinement tree with
labelled, commitment-dependent choices) and the matmul family factored into it —
`Autotune.matmul_seed_params` {e is} `Schedule_space.leaves` of `matmul_sketch_tree`, pinned
list-for-list against the pre-factoring golden by test/operations/sketch_family_tree.ml; the
epilogue-twin level factored in as the tree's root (gh-613), and the conv family factored the
same way as its own tree under the same root (gh-697, `conv_sketch_tree`; `sketch_seed_params` is
the claiming family's leaves). Phase 2
implemented: legality over partial shapes for the factored family — tree children carry
verdicts decided at parent construction, mirroring `op_legality`'s three values quantified
over completions (`Refuted` with the violated constraint as witness — a decline explanation
produced before any compilation, gh-479; `Unknown` never fathoms — reserved for genuinely
compile-settled questions, e.g. permissive site detection) plus a fourth, `Excluded`, keeping
the gh-555
policy/legality separation visible in the space itself (a driver may lift a policy exclusion
by re-proposing; it must never re-propose a refutation). Every silent filter in the matmul
tree now carries its witness; `refutations`/`exclusions`/`unknowns` collect them per shape.
The parametric-op extension of `op_legality` itself (judging retype/split moves with open
factors, for the beam/preset dimension) rides with phase 4's beam subsumption, where its
consumer lives. The tree judges what the seeding layer owns — site classification, caps,
advertised capabilities, explicit configuration; builder-settled analyses (companion coverage,
auto-probed pool rendering) stay candidate-build concerns by design, their failures being
classified declines rather than mislabeled timings. Refined by gh-577 after the phase-6
evaluation measured the doctrine's cost (the lifted lattice pricing spaces whose every member
died at candidate build): the *statically-decidable components* of builder-settled rules — those
whose verdict is a function of the site classification and the lowering alone, invariant across
tile completions — lift into construction-time refutations at the pipeline level, above the
geometry menus and the lattice. Lifted so far: the zero-expansion row-axis rule, and the GPU
companion-coverage rule per fusion flavor (`companion_geometry`'s verdict never depends on the
geometry annotator, so one trivial-annotator query settles a flavor's whole family; the fused
flavor is judged separately because skipping the epilogue tail can make coverage pass where the
unfused form fails). The flavor-indexed verdicts first forced a second tree build per flavor;
gh-613 then factored the fusion choice in as the tree's root level — `Choice "fusion"
[unfused; fused]`, the fused child refuted with the recognizer's own reason
(`Schedule.fuse_epilogue_witness`) where no fusable tail exists — so each flavor's
preconditions are ordinary verdicts of its branch, the twins enumerate after all unfused
leaves exactly as before, and the unfused coverage verdict is shared (it implies the fused
one: the fused demand is a subset and the alignment analysis ignores the skip). The builders'
raise sites remain the safety net for parameters replayed
against a different lowering (fission recombination), and genuinely build-settled analyses stay
candidate-build declines. Recorded follow-ups: an A-orientation
site classifier (the `m_tb` analogue for forms reading A in place); the conv-family factoring
landed with gh-697. Phase 3 implemented: the dual (floor) extraction —
`Cost_model.completion_floor` — with every approximation biased down, the exact mirror of
`analyze`'s contract: guarded work counts guards-never-taken, the short-circuiting forms count
only their certain part (`Where`'s cheaper arm, `And`/`Or`'s left operand, `Arg1`/`Arg2`'s
selected operand), a node's multi-access union floors to its largest exact image (dual to the
capped sum, flagged loose), non-exact images, dead code and conditionally-evaluated reads floor
to zero, and opaque code (which breaks the upper contract) merely loosens a floor. Open
placement levels contribute zero on both legs — a certainty pre-pass attributes an open
producer's whole effect (its ops and operand reads) to the open level, since an inline
completion may instantiate fewer cells than the setter loop covers. Committing a placement is
re-evaluation with the narrowed open set: suppression only shrinks, so the floor is monotone in
refinement — the property that lets the bound prune (a per-node incremental delta cannot be
sound in isolation and was removed in review). The Inline recompute floor ships as zero (sound;
a nonzero floor needs lower-bound multiplicity metrics, deferred until the driver proves the
need). Phase 4a implemented: the branch-and-bound driver — `Schedule_space.search`, depth-first
in emission order (preserving the flat enumerations' first-best tie behavior), threading a
tightening incumbent with strict-improvement displacement, fathoming `Child` subtrees whose
optimistic bound meets the threshold while `Unknown` children are never fathomed and
`Refuted`/`Excluded` are the construction-time fathoms — with the fathomed-vs-scored
`search_stats` ledger (phase 6's evaluation data). `model_default`'s sketch selection now walks
the factored family through it, whole-routine and per-fission-segment, with the untuned
default's score as incumbent and the schedule-invariant `completion_floor` roofline as the
uniform bound (sketch completions share the base program's semantics, so the floor bounds them
all; it fathoms the family exactly when the incumbent already achieves it — the memory-bound
kernels where the default preset is optimal). Selections are unchanged (goldens byte-identical);
the conv family competed through a flat path at the tightened threshold until gh-697 factored it
into its own tree under the same root, so the walk covers every site. The epilogue twins
competed that way too until gh-613 made fusion the tree's root level, which put them inside the
walk — under its bound and on its stats ledger — after the unfused leaves, preserving the
seeds-then-twins order. Phase 4b (first half) implemented: bound pruning against the measured incumbent in `tune` —
an enumerative sketch candidate (whole-routine/per-segment sketches, split-reduce seeds) whose
schedule-invariant roofline floor meets the best measured time so far is skipped before
compile, the admissible-direction pruning of the tuned regime; config-gated
(`autotune_bound_pruning`, default off — it changes which candidates get timed, and soundness
rests on honest envelope constants, which the continuous agreement check guards but an
understated envelope over-prunes). Presets, saved schedules and the baseline are never pruned
(their reporting and cache-replay roles outlive winnability), and the report's `bound_pruned`
counts the fathoms apart from the keep-fraction pre-filter's. Remaining: the placement-space
search where phase 3's non-uniform floors differentiate subtrees (the joint inline × tile ×
fission space of the issue's routing comments); phase 5 makes the family bounds non-uniform
via symbolic tile extents; phase 6 is the evaluation over `benchmarks/`.
**Issue**: [ahrefs/ocannl#514](https://github.com/ahrefs/ocannl/issues/514), split off #494
(waypoint 4). Prerequisites landed: #494 waypoints 1–3 (`Ir.Affine`, the legality decision
procedures, `Schedule.op_legality`), #491 (`Ir.Cost_model`, roofline envelope, `model_default`),
#554/#555 (abstract tracer, `analyze_proc`/`specialize_proc` split, the inline decision vector,
`optimized.flip_candidates`, greedy `tune_inline_flips`), #560/#563 (analysis cache, canonical
identity in `Low_level.Canonical_render`).

## Goal

Autotuning today searches in op-list space: sketch families enumerate concrete schedules, each
candidate is compiled and timed, and legality arrives by construction-then-validation. This
proposal inverts it: the tuner proposes *shapes* of schedules — partial commitments with tile
sizes, axis assignments, placement levels, and staging decisions still open — and the solver
instantiates or refutes them before anything is compiled. Schedule inference, as the counterpart
to shape inference. With the analytic cost model as the objective, the formulation is literally
branch-and-bound:

- **Node** = a partial decision vector (below). The hand-written sketch families are
  enumerations of what should be B&B subtrees.
- **Fathoming by infeasibility** = the landed legality queries, lifted to partial shapes: a
  reduction-dependence edge refutes "parallelize k" for *every* completion of the subtree,
  before a single tile size is named. Each fathom carries a witness — the violated constraint —
  which is #479's declines-as-explanations produced *before* compilation instead of after.
- **Bounding** = the roofline envelope, admissible by construction:
  `max(FLOPs/peak-compute, compulsory-bytes/peak-bandwidth)` lower-bounds every completion by
  definition of peak. Model omissions only weaken pruning; the single failure mode is
  understated peak constants — hence phase 0.

Telamon (Beaugnon et al.) is the architectural precedent — candidates as sets of open choices,
optimistic analytic bounds, B&B to the leaves. Its acknowledged weak point, the hand-authored
per-target legality constraint set, is the part OCANNL gets natively: the polyhedral description
is never recovered, so legality is derived rather than authored. The second differentiator is
determinism as a first-class constraint: reduction order is a legality dimension inside the same
algebra (the `rmw` markers on access relations), so the search can *prune on* determinism — a
dimension classical polyhedral work rarely models and e-graph reasoning erases.

## The node: a partial decision vector

A node is a vector of decision levels, each either committed or open. The dimensions, grounded
in what already exists as data:

- **Placement level per policy-decided node**: `Inline | Stage_at | Materialize`. The Inline
  half exists (`Context.decide_inline`, `optimize_ctx.inline_preferences`); the surface is
  reported per compile as `optimized.flip_candidates` / `Context.decision_surface` with
  per-node recompute-cost bounds (reduction extent × per-cell read multiplicity, from the
  gh-554 affine metrics). `Stage_at` — the compute-at middle of the spectrum, unifying the
  vector with the `Stage` optop — is designed into the vector now so it is a decision level,
  not a bolt-on: a partial commitment over placement levels is precisely the B&B node.
- **Sketch family and its parameters**: which composed-optop pipeline (matmul sketches, conv
  flavors, split-reduce sites), with tile/block sizes open until committed. Factoring each
  family into a refinement subtree replaces its `sketch_params` enumeration.
- **Fission/coverage choices**: segment boundaries and per-segment schedules — the dimension
  where the measured wins live (the gh-528/531/569 attribution: mma share ~7% CUDA / ~0% HIP;
  tuning wins are scheduling, and the top lever, gh-569's companion-arity rule, is a coverage
  rule).

Refinement commits one open level (or splits its domain); leaves are concrete schedules, exactly
today's candidates. Candidate replay stays hermetic and cheap: one shared `analyze_proc` result,
`specialize_proc` forks per vector (gh-555), and every new dimension registers its canonical
identity once, in `Low_level.Canonical_render` (gh-563).

Two invariants bound the space, carried from the landed design:

- **Legality and observability stay outside the vector.** Preferences suppress heuristic caps
  only; `check_and_store_virtual` and the observability contract are non-negotiable (gh-555).
  The search varies cost decisions exclusively, so every node's subtree contains only
  semantics-preserving completions and fathoming is purely about cost/feasibility of *schedules*.
- **Bitwise-per-schedule discipline**: numerics-policy fields are uniform across the whole
  search and part of cache identity (gh-568); a candidate may vary only what a schedule may vary.

Default-off, bit-for-bit: with the search disabled, behavior is unchanged.

## Fathoming

1. **By infeasibility**: `Schedule.op_legality` extended from "is this concrete op valid" to
   "is any completion of this partial shape valid" — three-valued as today. `Op_illegal` over a
   partial shape fathoms the subtree and its witness is the explanation; `Op_unknown` never
   fathoms (it forces refinement or measurement, preserving the advisory contract).
   `hardware_limits` caps fathom the same way (a tile-size interval whose minimum footprint
   exceeds shared memory refutes the whole box).
2. **By bound**: the optimistic roofline of the partial shape meets or exceeds the incumbent.
   In the untuned regime the incumbent is the best model score seen (exact argmin search); in
   the tuned regime it is a *measured* time, and pruning happens only in the admissible
   direction — the optimistic bound already exceeds the incumbent's measurement.

## The bound, and its soundness statement

**Statement.** A fathom on bound is sound iff for every node `v`,
`bound(v) ≤ min { true_time(c) : c a completion of v }`. Two ingredients can break it, and they
are audited separately:

- **Counts.** Today's `Cost_model.analyze` computes *upper* bounds on a concrete candidate's
  compulsory work — the right direction for ranking and calibration, the wrong direction for
  partial-shape fathoming, because completions of one shape differ in materialized traffic (a
  `Materialize` commit adds bytes that an `Inline` completion never moves). The partial-shape
  bound needs a second, *lower*-bound counting mode: semantic FLOPs without recomputation, and
  compulsory live-in/live-out traffic minimized over the open placement levels. This is phase
  3's deliverable, and it is why phase 0 does not attempt a bound over partial shapes — the
  extraction contract must be dualized first, not reused optimistically.
- **Envelope.** The peak constants must be honest peaks: `bound ≤ measured` requires
  `peak ≥ counts/measured` on each leg for every candidate ever timed. Overstated peaks are
  safe (the bound stays a lower bound, pruning weakens); understated peaks are the one failure
  mode that prunes true winners.

**The agreement invariant (implemented, phase 0).** Per the gh-498 lesson — an invariant
between a scorer and reality is checked continuously against every sample, never unit-tested on
each side — every candidate timed by `Autotune.tune` now checks its roofline bound against its
measured time whenever envelope constants are present, independent of logging and calibration
settings. A violation warns unconditionally on stderr with the witness (candidate label, digest,
the implied per-leg minima) and prompts a refit. Under the future search this same check runs on
every survivor measurement, so a broken fathom cannot stay silent.

**Envelope fitting (implemented, phase 0).** `Ir.Cost_model.Calibration` owns the calibration
TSV schema (one writer, `Autotune`; one reader, `tools/fit_envelope.exe`; they share the code)
and the fitter: per backend, each leg starts at its tightest necessary constant — the maximum
achieved `counts/time` over the rows where *that leg's* counts are exact (exactness is per leg:
guards-taken op counting and union/multi-read footprints go approximate independently, so a row
with an exact op count still feeds the compute leg past an approximate footprint). Approximate
legs are excluded from fitting because guards-taken over-counting can fake a throughput above
any hardware peak, and one mostly-failing-guards candidate would inflate the envelope
machine-wide; they stay recorded for divergence analysis, and a bound-exceedance attributable
only to an approximate leg is an `autotune_log` diagnostic rather than an unconditional warning,
since it may indict the counts, not the envelope (an *exact* aggregate leg exceeding the
measurement indicts the envelope regardless of the other leg, and warns naming only the
configured, exactly-counted legs). Serialized milliseconds are floored, not rounded, at
the 6th decimal, so a stored time never exceeds the true measurement and file-fitted constants
stay conservative with respect to it. Multi-kernel rows aggregate per-kernel counts, so for
them the per-leg maxima are necessary but not
sufficient (the bound sums per-kernel max-of-legs, which a compute-bound + bandwidth-bound mix
pushes toward twice either aggregate leg); the fitter therefore raises both legs uniformly by
the smallest *fission slack* enforcing the aggregate sufficient condition
`flops/peak_flops + bytes/peak_bandwidth ≤ time` on every multi-kernel row, after which the
recomputed bound respects every row it was fit from. Raising peaks is the safe direction —
the bound stays a lower bound, only pruning weakens. Opaque rows are excluded — the model
never scores them.

The fit is sound on the data, not certified for the machine: fitted peaks are floors a kernel
demonstrably reached, and a future candidate can achieve more. When it is *timed*, the
agreement check catches the violation and the fit tightens; the blind spot is
`autotune_keep_fraction < 1`, where a mis-bounded candidate can be model-pre-filtered before
timing — no measurement, nothing for the check to see. That risk is inherent to any envelope
not certified from above (class constants have it too, in both directions); the mitigations
are the refit loop, the pre-filter's existing no-coverage exemption, and the fitter's
`--margin` option, which trades pruning strength for headroom explicitly.

**Envelope resolution order.** Per-machine fitted config values (`model_peak_flops`,
`model_peak_memory_bandwidth`) beat the backend's class-level advisory constants
(`hardware_limits`). Between them belongs a third source, designed for but not yet bound:
per-device driver-derived peaks (SM count × clock for compute, memory clock × bus width for
bandwidth; the occupancy/attribute queries being bound in lukstafi/ocaml-cudajit#20). The seam
is `hardware_limits` itself — a backend that can query populates better `peak_*` values, and
everything downstream (envelope, fitter, agreement check) is source-agnostic. Degradation
without the queries is the current behavior: class constants, or none (cc), in which case there
is no bound and fathoming is by legality alone — the model stays advisory at every source
quality.

## Incumbents and the two regimes

1. **Untuned-default path** (first driver target): B&B computes the model-argmin exactly, with
   no measurement. `model_default` is the 0-dimensional version of this regime — score a fixed
   menu, ties to the default, any failure falls back. The search generalizes the menu to the
   full vector space while keeping every advisory guarantee: candidates without model coverage
   are never fathomed, ties go to the default, and the pick is validated before it replaces the
   default pipeline. The floor rises from "default preset" to "model-optimal schedule shape"
   for zero measurement cost, and the result is immediately checkable against the current
   default — lowest-risk first deployment.

   **Implemented** — the family levels in phase 4a, the placement levels under config
   `model_default_placements` (= N > 0): before the compile, B&B over the top-N ranked flip
   candidates, one keep/flip choice per level, each leaf a placement vector whose hermetic
   lowering (`Context.lowered_for_decisions`, riding the gh-560 analysis cache) is priced by
   the same selection that scores the pipelines — so the search is joint over placements and
   sketch families. The all-keep leaf is visited first, making the default placements' score
   the incumbent (ties stay with the default), and `Schedule_space.search`'s bound receives
   the decision path, so the partial-vector floor tightens per subtree — this is exactly where
   phase 3's floors differentiate, the family levels' floor being schedule-invariant. The
   picked vector applies through the same context-level decisions as the tuned chain's flips
   and stays advisory: a classified failure under the picked placements falls back to
   selecting from the caller's own placements.
2. **Tuned path**: prune against a measured incumbent only when the optimistic bound exceeds
   the incumbent's measured time; return the top-K leaves for measurement. The greedy
   `tune_inline_flips` chain — the minimal search over the placement dimension, one flip at a
   time, no bounds — is both the incumbent-provider and the baseline any B&B must beat. The
   model never overrides a measured result, and every survivor timing is a calibration row.

   **Implemented** as the chain upgraded in place, keeping the hierarchy (placements outer,
   tiling/fission searched by the nested `Autotune.tune` per measured flip — the joint
   inline × tile × fission space): the chain walks `Autotune.placement_surface`'s
   enablement-first ranking (below), and under `autotune_bound_pruning` a `Materialize` flip
   whose partial-vector floor — monotone in the chain's accumulated commitments (accepted
   materializations, kept default-materialized nodes) — already meets the best measured time
   is fathomed without spending budget, phase 4b's admissible rule one level up. The budget
   counts measured flips, so fathomed candidates let deeper ones in.

## Node ordering: enablement, not marginal cost

The gh-558 reduced-scope check (benchmarks/report-gh558-hip-flips.md) is the recorded lesson:
recompute-cost bounds mis-rank *enablement* flips. On mlp_wide/hip/bf16 the greedy chain's
most-expensive-first ordering put four `Inline` flips that seed nothing (cost 2048/1024) above
the `Materialize` twin flips (512/256) whose acceptance unlocked an entire candidate family
(tensorized seeding 0 → seeded+timed, −37%); a budget-5 chain reported nothing five times while
budget 18 found it. The moral for B&B: neither the branching order nor any bound may be a
function of the flipped node's own cost alone, because a placement decision's value includes
which sketch families become *expressible* under it — the twin flip changes the feasible set,
not just the objective. That is naturally a B&B concern (subtree value, not node cost), and a
cheap enablement prior exists: whether materializing the node changes the precision triple at
an mma-eligible site is computable from the seeders' own site classification, before any
compile. The gh-558 data is the ready-made test case for whatever ordering replaces cost-only
ranking.

**Implemented.** `Autotune.placement_enablement` computes the prior from the seeders' own site
classification, before any compile: the mma-eligible matmul sites (whole-routine and
per-fission-segment, the granularity the seeders detect at) of the all-materialized
specialization that have no eligible counterpart under default placements. A `Materialize` flip
of such a site's operand or destination is family-unlocking and ranks first; an `Inline` flip
of any eligible site's node can only move away from the family and ranks last; recompute cost
orders within each class (`Autotune.rank_flip_candidates`). Config `tune_flip_ordering` selects
it; `cost` keeps the legacy ranking as the evaluation baseline, and `enablement` selects this
pure prior — which is what phase 6's cells C were measured under, and what to pin to reproduce
them.

The shipped default is neither: phase 6 found that the prior models expressibility and not
profitability, and on metal/f16 promoting a family that loses tenfold displaced the winning
cheap flip out of a budget-5 chain (follow-up gh-ocannl-579). `tune_flip_ordering=profitable`,
the default since, weighs the prior against what the placement A/B's own arms MEASURED about
that family (`Autotune.family_profit_of_reports`: `mma_best_ms` against `best_ms`, compared to
`tune_flip_profit_margin`) — it IS `enablement` when the family is competitive or was never
timed, and degenerates to `cost` when it is measured to lose here.

## Phases

0. **This PR**: the design; envelope fitting (`Calibration` schema + fitter + tool); the
   continuous agreement check. No search.
1. **Partial-vector representation**: the decision-level types, the refinement relation, and
   the factoring of one sketch family into a subtree (the others follow mechanically).
   `Stage_at` enters the vector here.
2. **Legality over partial shapes**: `op_legality` quantified over completions; witnesses as
   pre-compile decline explanations.
3. **Lower-bound counting mode** in `Cost_model`: semantic FLOPs and compulsory traffic
   minimized over open placement levels; the envelope constants audited by the (by then
   accumulated) calibration data.
4. **The driver**: B&B for the untuned-default path, checked against `model_default` and the
   current default; then the tuned path with measured incumbents and top-K leaf measurement,
   with the greedy flip chain as the baseline to beat. Delivered in two steps: the family
   levels (4a) with tuned bound pruning (4b), then the placement levels in both regimes — the
   enablement ordering, the partial-vector floors, `model_default_placements`, and the
   bound-pruned enablement-ranked chain (see the implemented notes above).
5. **Interval bounding over symbolic tile parameters** (gated on #490): footprint and occupancy
   are monotone in tile sizes, so whole boxes of the divisor lattice bound by interval
   arithmetic — the regime where B&B strictly dominates the beam.

   **Implemented.** The tile parameters go symbolic as interval boxes in the family tree: the
   staged mma geometry level gains a lattice branch — every intrinsic-tile multiple of the row
   block crossed with every staged depth block, as a binary interval refinement
   (`Autotune.interval_axis`) whose half-boxes carry corner-judged verdicts (the
   workgroup-memory floor at the most favorable corner refutes the whole box pre-expansion, the
   fathom-by-infeasibility of §Fathoming) and whose certain staging traffic prices the bound:
   `sketch_path_traffic_floor` reads the committed staging decisions off the decision path — a
   committed staged geometry contributes its operand tiles' distinct-cell footprints exactly as
   `analyze` charges them on every leaf (only traffic that is *additional* counts: a hoisted CPU
   panel replaces the original operand's reads and contributes zero), a box its smallest-tiles
   corner priced at the same per-format intrinsic tile the lattice is built from — so the family bound
   is finally **non-uniform** across the tree, monotone in refinement, and whole boxes fathom
   without expansion (the schedule-invariant floor still differentiates only placements; the
   increments differentiate the sketch-geometry subtrees). The lattice hides behind an
   `Excluded` branch (`geometry_lattice_witness`), so `leaves` — the tuner's seed lists, and
   everything measured — are unchanged by its existence; config
   `model_default_geometry_lattice` lifts it into `model_default`'s family search
   (`lift_geometry_lattice`, laziness-preserving, lifted branches still subject to the box
   refutations). On the 64³ pinned test site the lifted walk scores 13 of the 80 leaves with 26
   subtrees fathomed — the logarithmic-effective slice. Bounding over symbolic *program*
   extents (a site whose m/n/k are gh-490 `Row.Sym` dims) remains out of scope: site detection
   is concrete-extent, and `set_test_bindings` already pins symbolic extents to their upper
   bound for timing.
6. **Evaluation as research output**: nodes fathomed vs. candidates compiled, model-vs-measured
   divergence per backend (the calibration TSV is the ledger), and the untuned-default quality
   delta on `benchmarks/` — honest reporting, including where the beam remains competitive
   (short parameter enumerations without symbolic extents).

## Relations

Composes with #479 (fathoming witnesses are decline explanations produced pre-compilation) and
rides on #560's analysis cache (one `analyze_proc` shared across a search's specializations —
already landed, and load-bearing once searches multiply candidate counts). Disjoint from #261
(superoptimization proposes new programs; this searches schedules of a given program). #267
resolved as a study; its conclusions (incumbent pruning, decision-vector space) are subsumed
here.

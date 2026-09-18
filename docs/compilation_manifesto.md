# A Manifesto for OCANNL's Compilation Side

*Living document, started 2026-07-16.*

OCANNL's public identity has been its frontend: shape and projections inference as a
constraint system, the generalized einsum notation, the concise `%op`/`%cd` DSL. This
document is about the other half — the compilation stack under `arrayjit/` — and makes
three kinds of claims about it: what is **deliberate** (decided, and worth defending),
what is **emergent** (grown out of the project's design culture rather than decreed, and
worth naming so it can be cultivated), and what is **aspirational** (the direction that
would make the compilation side a research identity rather than an implementation). It
also names what is deliberately boring, because the identity claims are only credible if
the table-stakes engineering underneath is unremarkable and correct.

For a dated, issue-linked account of the remaining work and its milestone homes, see
[What remains for OCANNL](blog/2026-09-14-what-remains-for-ocannl.md).

## 1. Schedule the step, not the kernel *(deliberate)*

The unit of compilation is the whole training step. Forward, backward, and the optimizer
update lower into **one** `Low_level` program; kernel boundaries are then *discovered* by
fission at materialized cross-nest edges, and selectively erased again by lossless
cross-nest merging. This inverts the field's default decomposition: operator-graph
frameworks start from many small kernels and face the *fusion* problem — which ops to
merge, an under-constrained search over a graph whose dataflow they see only locally.
OCANNL starts from the fused whole, where the entire step's dataflow is in view, and
faces the better-posed *fission* problem: where must a single program be split.

Consequences that fall out rather than being engineered:

- Cross-forward/backward optimization is the default, not a pass: gradients inline into
  forward loops when virtualization allows; multiple tensors can share a loop nest.
- Fission segments are stable, digest-addressable units — which is why per-segment
  autotuning (`F_sketch`, per-segment schedule caches keyed by pre-schedule digests)
  exists here and has no direct analogue elsewhere.
- Re-fusing (aligned cross-nest parallelism, fused segment encoding) is a measured
  choice against alternatives, not a heuristic commitment.

The paired decision is **explicit compilation and context threading**. There is no
implicit global session or graph: `Context.compile` and routine execution thread
contexts explicitly, and placement decisions are forked per compile (`optimize_ctx`).
Compilation is a function — (program, config, context) → routine — whose result is a
value you hold. Schedule caches, digest guards, autotune persistence, and multi-device
correctness are all downstream of compilation being referentially transparent enough to
cache and replay.

## 2. Virtual-first: memory is an optimization, not a birthright *(deliberate)*

The default state of a tensor node is *virtual*: no bytes exist anywhere; consumers
recompute the defining expression inline, subject to aggressive inlining with hoisting
and common-subexpression elimination. Materialization is something a node *earns* —
visit-count thresholds, reduction-cost guards, hosting requirements — or the user
forces. Mainstream frameworks materialize by default and treat fusion as the
optimization; OCANNL inverts the polarity.

The observability contract (the goal state; the implementation is in flux): **virtuality
does not forfeit observability**. A virtual node is observable *by recomputation* — the
framework can always reconstruct its values on demand. The only placement that trades
observability away is `Local` — routine-scoped scratch — and a node must be *uniquely
unobservable* for the compiler to choose it. So the lattice is not "materialized =
visible, virtual = gone": everything is in principle visible, and the compiler
negotiates the *how* (bytes versus recomputation), never the *whether*.

Placement decisions live on the compilation context, not the tensor: a node's
`memory_mode` is intent; `Placements` on the optimization context is decision, forked
per compile. The same tensor can be virtual in one compiled routine and on-device in
another, and both routines remain valid simultaneously.

The boundary with the schedule layer (§3) is a contract worth stating, because placement
began as a normalization-like pre-pass and became searchable without becoming a schedule
op — deliberately. Placement is a *specialization parameter*: lineage-scoped, deciding
which loop nests exist at all, and part of the program identity schedules are keyed by
(the schedule cache's digest includes placement classes). A schedule is a routine-scoped
mapping of the code a placement produced. The contract between them: **the schedule layer
may create memory — the tile-namespace nodes it mints, staged tiles, split-reduction
partials, privatized scratch — but never re-decides the placement of a node it did not
mint**; the single sanctioned exception is the `Local`-to-`On_device` promotion
(`Placements.promote_local_to_device`), which fission uses both reactively — a live range
crossing a kernel cut must be promoted for the program to stay correct — and proactively
on GPU paths (`promote_locals:true` promotes statement-crossing `Local`s before
segmentation, so better cuts and hardware annotations become discoverable). Either way it
is the one door, a single named upward-only move. (Under this reading `Privatize` is not a
placement decision at all: it preserves the target's placement, `On_device` or `Local`,
and adds a caching access pattern for it.) Search reaches both layers, but
hierarchically — placement A/B and flip refinement outside, schedule search within each
candidate — because their invalidation costs differ by orders of magnitude: a placement
flip changes the digest and replays specialization, a schedule candidate transforms its
output over shared analysis. The gh-514 decision-vector reformulation would eventually
make both coordinate kinds of one searched vector; until then the two persist as two
stores in one directory: schedule winners keyed by the placement-aware digest of the
decided code, placement decisions keyed by the structural identity of the decision
problem — the raw program plus what the lineage already decided, the decision itself
outside the key (gh-786, `Schedule_cache.canonicalize_source`).

A corollary worth stating: **there is no layout problem**. Einsum projections are the
sole loop-nest generator — there is no reshape, no NCHW-vs-NHWC propagation pass, no
transpose insertion. Layout adaptation is either index arithmetic (free) or an explicit
packing `Stage` (a scheduled, measured copy). An entire subsystem that consumes
enormous machinery in XLA/Inductor-class compilers is absent by construction.

## 3. Total schedules: legality in the IR, performance at render *(emergent)*

Every schedule transform is total: it either applies exactly, preserving semantics by
construction, or it is rejected up front — and every optimized *rendering* declines
gracefully to a correct fallback. `Tile_mma` carries its scalar fallback as a subtree of
the statement itself; a tensorized schedule on a backend without tensor cores, or on a
shape the intrinsics reject, renders slower but never renders wrong. Contrast TVM's
`tensorize`, which hard-fails on mismatch, or Triton, where the tiled kernel is simply
yours to get right. In OCANNL, *any* schedule on *any* backend is correct; performance
is the only thing being negotiated.

The honest cost of this design is silent declines — "tensorized" candidates that
quietly render scalar (gh-474, gh-479). The remedy is diagnostics and pre-filters, not
abandoning totality: "correct by construction, fast by negotiation" is the right
default for a system whose schedules are increasingly machine-proposed.

Credit for this thread goes less to a decision than to the project's materials:
**algebraic-data-type IRs where loops are proper tree nodes** make "a statement that
carries an alternative rendering of itself" just another constructor field, and make
transforms structural recursions that are hard to write half-correctly. And the
**exact-`.expected` snapshot discipline** pins every rendering path in golden files —
a decline or a fallback is a visible diff in a reviewed artifact, not a mystery in a
profile.

## 4. The determinism contract *(emergent, to be made deliberate)*

Bitwise parity is treated as a compiler invariant, not a test tolerance:

- The register-tiled `Tile_mma` rendering must be *bitwise equal* to the scalar
  fallback — it declines the plain-add accumulation form precisely because a vector
  twin could not promise that equality for maybe-contracted arithmetic.
- Parallel rewrites hold to a **deterministic no-atomics** rule; where parallelization
  would race or reorder a reduction, the code stays serial rather than approximately
  parallel.
- The cross-framework benchmark suite is parity-gated: performance numbers are only
  reported for computations verified equivalent.
- Numerics relaxations are *named policies*, not silent defaults: tf32 is an explicit
  pending decision (gh-478), and where bitwise equality is genuinely impossible
  (tensor-core tile reassociation) the relaxation is localized and documented as the
  exception.

The field shrugged at run-to-run nondeterminism years ago; deterministic modes in
mainstream frameworks are opt-in and slow. The aspiration here is the inverse claim:
**bitwise-reproducible training at competitive speed**, with every departure
policy-gated. The concrete enabler is deterministic *two-pass split reductions* (partials
plus a fixed-shape tree combine) in place of atomics — parallelism for reductions,
scatter-backward, and split-K without surrendering the contract.

## 5. One vocabulary from AVX2 to tensor cores *(emergent from the functorized design)*

This fell out of pushing hard for commonality across backends — the `C_syntax` functor
with minimal per-backend overrides — rather than from a portability doctrine, and it is
worth defending now that it exists. Hardware axes are *retypings of existing loops*
(`Grid`, `Workgroup`), not a separate GPU programming model: the same `Grid` loop
renders as a `dispatch_apply`/OpenMP pool on CPU and a block index on GPU. `Tile_mma`
is one IR statement with four renderings — a tinyBLAS-style register tile on cc, Metal
simdgroup intrinsics, CUDA wmma plus inline-PTX fp8, HIP rocWMMA — under a shared
decline-to-fallback semantics. A schedule is backend-portable by construction, and
autotuning is per-backend *measurement* over a shared candidate *language*. No
mainstream framework spans a laptop's SIMD register file and a datacenter GPU's matrix
units with one schedule vocabulary.

## 6. The aspirational core: constraint solving for schedules *(in progress)*

The frontend solves shapes and projections with a constraint system — row variables,
dimension inequalities, staged solving. The scheduler, meanwhile, re-derives affine
facts *procedurally*: `validate_parallel`'s coverage and uniformity rules,
`parallel_grid_safe`'s write-dominance and footprint analyses, `detect_matmul`'s
structural pattern match. These are the same class of facts the projections algebra
already carries — the loops being analyzed were *generated from* those projections.

The direction: derive schedule legality — and eventually schedule *inference* — from
the same solver. **Shape inference for schedules.**

Positioning against the research landscape:

- **Equational / e-graph rewriting** (equality saturation; the egg/Tensat lineage) is
  the popular current direction. We should draw on it for the algebraic-rewrite tier —
  reassociations like online softmax, algorithm substitutions like Winograd — without
  adopting term graphs as the core representation: OCANNL's object is a program with
  loops and state, and its legality questions are affine and order-sensitive
  (determinism!), not purely equational. Rewriting proposes; the constraint layer
  disposes.
- **Polyhedral compilation** (the Feautrier line; isl, Pluto, Tiramisu — gh-267) is the
  classical home of affine legality, and the angle worth exploring deeply. The twist
  that makes OCANNL's position fresh: polyhedral compilers must *recover* affine
  structure from loop nests a human wrote, and lose the fight at the first non-affine
  wrinkle. OCANNL's loop nests are *generated from* einsum projections — the polyhedral
  description is native, never lost, never reconstructed. Row variables are parametric
  dimensions; convolutions arrive as affine index maps by construction; concat axes are
  piecewise-affine unions. A polyhedral layer here starts from the semantic object,
  not the syntactic one.
- **Determinism as a scheduling constraint**: reduction-order preservation can be a
  legality dimension *inside* the same algebra — a constraint classical polyhedral
  work rarely models, and a differentiator worth owning.

Rough waypoints:

1. Express projections as explicit affine relations (presburger-style sets/maps) rather
   than only as index-expression emitters. *(Landed: `Ir.Affine` and
   `Low_level.affine_accesses` — access maps with loop boxes, program-order paths, and
   reduction-dependence markers.)*
2. Re-express the procedural checks (`validate_parallel`, `parallel_grid_safe`,
   micro-kernel recognition) as emptiness/subset queries over those relations —
   equivalence with the procedural answers is the regression suite. *(Landed for
   the legality checks: the shared-memory rules as pair-conflict queries, and the
   order-sensitive per-thread-scratch rule via the containment query
   (`Affine.read_covered_before`) — a ∀∃ procedure with box-union coverage, whose
   crosschecking exposed and fixed a real annotator gap: scratch whose written
   value depends on a chain symbol without pinning the cell diverged from serial
   semantics under per-thread copies. Micro-kernel recognition —
   `detect_matmul`/`detect_conv` in the autotuner — remains a procedural
   structural matcher and has not been migrated to relation queries.)*
3. A legality *oracle* for schedule ops: given a schedule, decide validity by query
   instead of by construction-then-validation. *(First slice landed:
   `Schedule.op_legality` — three-valued verdicts sound in both proven directions,
   with the reduction-reassociation license — and the autotuner prunes
   proven-illegal menu proposals before compiling them. Remaining: modeling the
   staging/tensorization ops, and re-deriving the default annotators as
   search-plus-oracle.)*
4. Inference: search in constraint space rather than op-list space — the autotuner
   proposes shapes of schedules, the solver prunes cheaply before anything compiles.
   *(Tracked as gh-514, which carries this waypoint on its own: the first three are an
   engineering program with a regression criterion, this one is a search architecture
   with its own prerequisites and its own evaluation criteria.)*

Status, mid-2026: waypoints 1–2 landed as the `Ir.Affine` query engine — pair-conflict
(disjointness by gcd/interval infeasibility, thread-confinement by forced equalities
under the mixed-radix injectivity criterion), covering, and counting (fiber
cardinality), each verified against brute-force enumeration oracles and soaked under
the permanent `legality_crosscheck` flag, which runs the legacy procedural analyses
alongside the queries and makes any disagreement loud. The crosscheck's first runs
already earned it: one engine gap caught on real kernels, two precision gains where
queries merge kernels the procedural rules could only cut, and oracle confirmation
that a syntactically-agreeing but non-injective access pair the old rules accept is a
genuine race. The tooling question resolved against isl: `axis_index` is structurally
linear — no divisions, no quantifier alternation — so every query form so far decides
with native linear reasoning; revisit only if waypoint-4 inference needs genuine
existential elimination.

Waypoint 4 (gh-514) extends naturally to an *optimizer*: with the analytic cost model
(gh-491) as the objective, schedule search becomes branch-and-bound over partial
schedule shapes. Legality queries fathom infeasible subtrees wholesale — a reduction
edge refutes "parallelize k" for every completion, before any tile size is named —
and peak-envelope roofline bounds are admissible by construction: max(FLOPs / peak
compute, compulsory bytes / peak bandwidth) lower-bounds every completion's runtime,
model omissions only weaken pruning, and only overstated "peak" constants can
mis-prune, so the envelope numbers must be honest peaks. Two regimes keep the model
advisory: the untuned-default path takes the model-argmin exactly and cheaply; the
tuned path prunes only in the admissible direction against measured incumbents and
returns top-K leaves for measurement. The step past beam search comes when tile
parameters go symbolic (gh-490): footprint and occupancy become monotone functions of
tile sizes, so whole boxes of the divisor lattice bound by interval arithmetic.
Telamon (Beaugnon et al.) is the precedent for exactly this architecture — candidates
as sets of open choices, optimistic analytic bounds, branch-and-bound to the leaves —
and its acknowledged weak point, a hand-modeled legality constraint set, is the part
OCANNL gets natively.

## 7. Deliberately boring *(and proudly so)*

GEBP/BLIS-style packing and register tiling, wmma/simdgroup/MFMA emissions, beam search
with sketch seeding (the Ansor lineage), streams/events/memory pools, snapshot testing,
fission at materialization boundaries. These reproduce established engineering because
they are the known-good substrate; novelty here would be a liability. The identity lives
in §§1–6; everything else should look reassuringly familiar to someone who has read the
BLIS papers, the Halide/TVM schedule literature, and a CUDA best-practices guide.

## North star

A framework where the whole training step is one program; where memory and parallelism
are negotiated by a solver over the same algebra that inferred the shapes; where every
optimization is bitwise-faithful or policy-named; and where one schedule vocabulary
spans a laptop's SIMD lanes and a datacenter GPU's tensor cores.

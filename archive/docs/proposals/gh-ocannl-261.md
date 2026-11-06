# Superoptimizers for tensor programs: deep dive and follow-up plan

**Issue:** [ahrefs/ocannl#261](https://github.com/ahrefs/ocannl/issues/261)
**Status:** Deep dive **done in this revision (2026-07-07)** — the reading, literature
re-selection, and technique-to-seam mapping below replace the earlier scouting plan.
Detailed evidence base (full paper extractions, verified survey with citations,
cross-cutting implementation notes): [docs/research/superoptimizers.md](../../../docs/research/superoptimizers.md).
Remaining deliverable: file the follow-up issues and comment on #261.
**Milestone:** v0.9 (Program search with execution-based cost functions; due Aug 24, 2026 — ICFP week)

## Status update (2026-07-07) — supersedes the 2026-06-12 update

The repo moved out from under the previous revision of this document. Every
"known constraint" that shaped the old plan has flipped:

- **The schedule layer exists.** `arrayjit/lib/schedule.ml` (2198 lines) implements
  Halide-style loop-nest transforms as values — `Split`, `Swap`, `Retype`, `Unroll`
  (with IR-materializing mode), `Stage` (shared-memory tiles with cooperative loads
  and lane-aware mode), `Privatize` (accumulator privatization), `Expand_zero`,
  `Tensorize`/`Tile_mma` (tensor-core MMA) — applied as a pure
  `Low_level.optimized -> Low_level.optimized` pass at the `?lowered_transform` seam
  (`backend_intf.ml:221`, applied at `backends.ml:498-505`). See
  [schedule-ir-optops.md](../../../docs/design/schedule-ir-optops.md) (Phases S1–S5 implemented).
- **Kernels are no longer single-threaded.** The default GPU annotator
  (`automatic_gpu_schedule`, on by default for cuda/metal) parallelizes to
  Grid×Workgroup launches; hand schedules reach ~200 GFLOP/s f32 matmul on
  Apple-silicon Metal (register tiling + `Privatize`), 1800–3000× over the old 1×1
  baseline. Kernel fission splits routines at materialized cross-nest edges;
  `validate_parallel` (`low_level.ml:2660`) plus
  `Schedule.check_hardware_limits` gate legality at compile time.
- **BEAM search is in progress** (parallel work, modeled after tinygrad): beam over
  schedule prefixes with on-device timing, candidates as sibling `Context.compile`
  calls from one frontier, executed parity against the unscheduled twin as the
  correctness gate (scoped in [schedule-ir-optops.md](../../../docs/design/schedule-ir-optops.md) §8).
- **Consequence for #261:** the issue no longer owns "build a search space" — the
  schedule IR is the search space and BEAM is the search loop. What #261 owns now is
  what the papers below are actually about: (a) making the search converge fast
  (candidate construction, pruning, cost models vs. timing), (b) rewriting *above*
  the schedule layer (algebraic/graph-level rules), and (c) trusting rewrites
  (equivalence oracles).
- Refreshed code pointers (the old ones are all stale): the optimization pipeline
  entry is `Low_level.optimize` (`low_level.mli:279`), composing
  `cleanup_virtual_llc` (`low_level.ml:1450`) → `simplify_llc` (1869) →
  `eliminate_common_subexpressions` (2240) → `hoist_shared_locals` (2811) →
  `hoist_cross_statement_cse` (2908). Lowering: `Assignments.to_low_level`
  (`assignments.ml:201`), `Assignments.lower` (~1098). Codegen: `c_syntax.ml` is
  now ~1974 lines with hardware-axis rendering, SIMD vector rendering
  (gh-ocannl-164), and `mma_syntax` hooks.
- Issue states re-verified: #261 OPEN (v0.9), #267 (Tiramisu) OPEN, #412 (matmul
  tiling) OPEN with its >10× criterion already met by parallelization alone.
  v0.7 shipped 2026-07-03; the workshop-paper deadline pressure of the old
  "Known Constraints" section has passed.

## Goal (revised)

Deliver the deep dive (done — this document) and derive concrete follow-up tasks
for v0.9's program-search milestone. Success criterion unchanged: a maintainer can
pick a follow-up and start work without re-reading the papers — every follow-up
names a specific technique, a specific OCANNL seam, and a prerequisite chain.

## Literature selection: verdict

**Is Mirage + Tensat the ideal pair? Mostly yes as anchors — but no longer
sufficient.** A 2026-07 scouting sweep (arXiv/venue pages verified, not from
memory) found the lineage alive and two gaps in the original selection:

1. Nothing has displaced **Mirage** (OSDI'25) as the state of the art in joint
   algebraic + schedule superoptimization. Its lineage continues: **MPK / Mirage
   Persistent Kernel** (arXiv 2512.22219, accepted OSDI'26) compiles whole LLM
   inference into one megakernel — the opposite pole of OCANNL's kernel fission —
   and **Axon** (arXiv 2606.26344, June 2026 preprint, unreviewed) synthesizes
   transformations with SMT verification over unbounded tensors, replacing exactly
   the probabilistic-verifier part of Mirage. Watch Axon; don't anchor on it yet.
2. **Tensat** (MLSys'21) remains the canonical tensor-graph equality-saturation
   paper. Its documented weaknesses (multi-pattern blowup, ILP-extraction wall)
   now have a modern companion fix: **Hartmann, He & Yoneki** (PACT'24, arXiv
   2410.05534) — MCTS-guided rewrite selection + fast extraction. Read paired.
3. **Gap 1 — verification.** **TensorRight** (POPL'25, ADAPT/UIUC) verifies tensor
   graph rewrites for *arbitrary rank and size* via "aggregated axes" — a formalism
   that is conceptually cousin to OCANNL's row variables (`..d..`). It proves
   115/175 XLA rewrite rules in full generality. Promoted to the anchor set as the
   principled alternative to Mirage's probabilistic verifier.
4. **Gap 2 — the BEAM track.** The original selection predates OCANNL having a
   BEAM search to tune. The highest-value reading for making a beam converge:
   **ROLLER** (OSDI'22 — constructive, hardware-aligned candidate generation
   instead of pruning a huge space; seconds instead of hours, no learned model)
   and **Pruner** (ASPLOS'25, arXiv 2402.02361 — "draft-then-verify": a cheap
   *analytical* hardware-fitness model drafts a small candidate set, expensive
   measurement verifies; ~4× tuning-time reduction). Both compose directly with a
   beam over schedule prefixes. Promoted to the anchor set.

**Revised reading list:**

| Tier | Work | Why |
|---|---|---|
| Anchor | Mirage (OSDI'25, 2405.05751) | search-space architecture, pruning, verifier, where the wins actually came from |
| Anchor | Tensat (MLSys'21, 2101.01332) + Hartmann et al. (PACT'24, 2410.05534) | e-graph rewriting: what pays, what blows up, the modern extraction fix |
| Anchor | TensorRight (POPL'25) | verified rewrites at arbitrary rank; aggregated axes ↔ row variables |
| Anchor | Pruner (ASPLOS'25, 2402.02361) + ROLLER (OSDI'22) | making BEAM converge: analytical draft filter + hardware-aligned candidates |
| Secondary | EinNet (OSDI'23) | derivation-based rewriting over einsum-like expressions — closest ancestor to OCANNL's IR |
| Secondary | Heron (ASPLOS'23) | constraint-based search for tensor-core schedules (relevant post-`Tensorize`) |
| Secondary | MPK (OSDI'26) | megakernel pole of the fission/fusion axis; grid-sync execution strategy |
| Secondary | Minotaur (OOPSLA'24) | SIMD peephole superoptimization; cut granularity ≈ OCANNL's vectorized loop bodies |
| Secondary | egglog (PLDI'23); Halide autoscheduler (2019); Ansor (OSDI'20); KernelBench (2502.10517) | ecosystem/background; KernelBench's `fast_p` metric for evaluating search output |
| Watch | Axon (2606.26344); LLM-guided kernel gen (AlphaEvolve, etc.) | re-check in 6 months; not actionable for a small OCaml project today |
| Skip | Felix (ASPLOS'24), Hydride (ASPLOS'24), TVM MetaSchedule, TLP | need differentiable schedules / backend synthesis / learned-model infra OCANNL doesn't have |

## Deep dive I: Mirage (OSDI'25)

### What it does

A μGraph is a hierarchy mirroring the CUDA hierarchy: **kernel graph** (nodes are
kernels — pre-defined cuDNN/cuBLAS ops or *graph-defined* operators whose semantics
are a nested **block graph**), block graph (one thread block; intermediates always
in shared memory; grid dims ≤3 with per-input `imap` — grid dim ↦ data dim or
replicate — per-output `omap` — disjoint concatenation only — and a for-loop body
with input iterators, **for-loop accumulators**, and `fmap`), and **thread graph**
(registers; built by greedy fusion only, not searched). Search: exhaustive
enumeration at kernel+block levels in canonical form (ops added in increasing rank
— no duplicate graphs generated), with per-prefix **abstract-expression pruning**:
each tensor is abstracted to a FOL term over integer arithmetic + uninterpreted
functions (crucially keeping reduction *sizes*: `sum(k, mul(E(X),E(Y)))` for
matmul), and a Z3 query checks whether the prefix's expression is still derivable
as a subexpression of the goal under axiom sets A_eq (associativity,
distributivity, sum-collapsing, exp(x)·exp(y)=exp(x+y)) and A_sub. Cancellation
axioms are deliberately excluded — with them "everything is a subexpression of
everything" and pruning nulls out. Theorem 4.1: no equivalent μGraph reachable
via A_eq is pruned. The ablation is stark: RMSNorm search with pruning takes 11 s
at 5 block-ops and 28 s at 11; without it, 768 s at 5 ops and >10 h at ≥7 — and
the winning RMSNorm kernel needs 11.

Correctness: a **probabilistic equivalence verifier** for the *Lax* fragment
(multi-linear ops + division + at most one exponentiation per path). Outputs have
a closed form whose zero-testing generalizes Schwartz–Zippel polynomial identity
testing over two linked finite fields (Z_q inside exponents, Z_p outside, q | p−1;
in practice p=227, q=113 so a test fits in 16 bits and runs on GPU). No false
negatives; false-positive probability bounded and shrinkable by repetition — though
the deployed system runs a *single* random test and has "not observed" false
positives. ReLU is unsupported (SiLU was added as a first-class op with its own
axioms); layouts/scheduling/memory-planning are deferred to a *post-verification*
ILP + DP optimizer, since they don't affect correctness.

Where the up-to-3.3× actually came from (§8): **grid-dimension and
parallelization-axis choice** (GQA: FlashDecoding parallelizes over sample/head/
KV-seq, Mirage picks per-scenario among sample/heads/query-seq/KV-seq — up to 7×
less device-memory traffic; using TensorRT-LLM's fixed grid dims costs 18%);
**running two accumulators in one for-loop** (RMSNorm's Σx² alongside the matmul's
Σ, division commuted past the matmul); **fusing tiny kernels** (LoRA's
(W‖B)×(X‖(A×X)) identity — which required *manually adding* a 4-input concat-matmul
operator to the vocabulary); and QKNorm-style fusion of normalizations into
attention. The one structural loss: nTrans vs TensorRT, because graph-defined
kernels *always* stage tensors through shared memory — a hard-wired memory-level
policy that dominates light-compute kernels.

### What transfers to OCANNL

- **The search-space factorization is already OCANNL's.** μGraph's
  grid/fmap/imap ≈ which loops get `Retype`d to `Grid`/`Workgroup` and how `Split`
  factors them; for-loop accumulators ≈ `Privatize` + `Workgroup_reduce`; thread
  graphs ≈ register tiles via materializing `Unroll`; graph-defined operators ≈
  what kernel fission already decides at materialized cross-nest edges. Mirage
  validates the *decomposition*; OCANNL needn't adopt μGraphs as an IR.
- **The single biggest transferable fact: parallelization-axis choice is where
  the money was.** OCANNL's default annotator currently picks a fixed
  Grid×Workgroup shape. The BEAM action space should include *which* axes (batch
  vs. output vs. reduction — split-K via `Workgroup_reduce`) map to hardware dims,
  not just tile sizes. `validate_parallel` already gates legality.
- **Feasibility pruning vs. cost pruning.** Mirage prunes by *derivability toward
  the goal* (SMT), not by cost — cost appears only in post-verification layout ILP
  and final profiling. For OCANNL's schedule-space BEAM (transforms are
  semantics-preserving by construction) the analog is cheap *legality/fitness*
  filtering before timing — which is exactly Pruner's draft-then-verify (Part III).
  The SMT machinery itself is only needed if OCANNL ever searches over *algebraic*
  rewrites generatively; skip until then.
- **The verifier idea scales down.** OCANNL's planned per-candidate gate is
  executed float parity vs. the unscheduled twin. Mirage's lesson: randomized
  testing with a principled story is cheap (16-bit finite fields, GPU-executed,
  reusing the optimizer's own kernels). A scaled-down oracle — random inputs,
  tolerance-aware comparison, and exact integer/finite-field mode for the linear
  fragment — is the right trust layer for future *graph-level* rewrite rules,
  where "preserving by construction" no longer holds.
- **The nTrans warning.** Don't hard-wire staging policy: OCANNL's `Stage` is an
  explicit optop rather than an IR invariant — keep it that way (a schedule
  *without* `Stage` must remain reachable for light-compute kernels).
- **Vocabulary bounds the search.** Mirage's LoRA win needed a hand-added
  operator. OCANNL's einsum-with-concat syntax (`a^b`, landed v0.7 #49) expresses
  the concat/split algebra *natively* — an unusual structural advantage for the
  sharing-rewrite family (Part II).

## Deep dive II: Tensat (MLSys'21) and the e-graph question

### What it does

Encodes TASO's operator language as e-graph terms (parameters as child nodes;
multi-output `split` as tuple + projections; fixed-arity `concat_n`; one `noop`
root), reuses TASO's 743 synthesized-and-verified rewrite rules unchanged, runs
`egg` to (bounded) saturation, and extracts with ILP. Contributions beyond plumbing:
(a) **multi-pattern rewrites** (Algorithm 1: canonicalize patterns, single-pattern
e-match each, Cartesian-product compatible matches) — with the honest finding that
these grow the e-graph *double-exponentially*, so they're capped at k_multi
iterations (default **1**); (b) **cycle filtering** during exploration (descendants
map + post-iteration DFS onto a filter list), because acyclicity constraints make
the extraction ILP 10–1000× slower or infeasible — with them, NasRNN extraction is
1116 s vs. 0.32 s without; at k_multi=2 everything with cycle constraints times
out at 1 h; (c) the ILP itself: binary per e-node, minimize Σ cost, one pick at
root, demand propagation to children, filter-list zeroing.

Results: 9.5–379× (avg ~48×) faster optimization than TASO's backtracking search,
finding graphs up to 16% faster than TASO's (NasRNN; ~6.6% average). E-graph capped
at 50k nodes, 15 iterations, total optimizer time 0.2–11 s at k_multi=1.

The load-bearing detail: **every showcased win is a *sharing* rewrite** — k
matmuls/convs sharing an operand merged via concat+matmul+split, or conv weights
concatenated and precomputed. These are precisely the rules that (a) blow up the
e-graph, (b) create cycles, (c) defeat greedy extraction (greedy *pessimizes*
NasNet-A, 22.5 ms vs 17.8 original, because the merged kernel pays off only if
both consumers pick their `split_i` — a cross-e-class decision). And the cost
model is sum-of-measured-per-op-runtimes, whose admitted failure (SqueezeNet gets
*slower* as k_multi grows while the model claims wins) previews any cost-model
fidelity gap. ResNet-50: zero speedup — the rule set, not the search, binds.

### What transfers to OCANNL

- **The minimal viable deployment is small.** Single-pattern algebraic rules to
  saturation + *one* iteration of the concat/split sharing family + non-greedy
  extraction. OCANNL doesn't need 743 rules or a saturation engine to capture the
  demonstrated value: the sharing family is a handful of rule schemas, and
  OCANNL's concat axes make them expressible as targeted `Assignments.t`-level
  rewrites (`Accum_op` trees with einsum specs) — appliable destructively under a
  measured accept/reject gate (the BEAM cost function reused at the graph level)
  rather than via an e-graph. Phase-ordering pain is real but only bites once
  rules number in the dozens and interact; Tensat itself shows k_multi=1 suffices
  for the wins.
- **The OCaml e-graph situation (re-verified 2026-07):** the one OCaml library,
  **`ego`** ([opam](https://opam.ocaml.org/packages/ego/), verse-lab), is an egg
  port — but latest release 0.0.6 (Nov 2021), ~35 commits, effectively
  unmaintained 4+ years, and **GPL-3.0+** (license concern for vendoring).
  egglog (PLDI'23) is the ecosystem center of gravity but means FFI or shelling
  out to Rust. A from-scratch minimal e-graph is ~1–2 kLOC of core. All three
  remain heavy relative to demonstrated need → **defer** (see follow-up F5), with
  the PACT'24 MCTS-extraction paper as the map for when that day comes.
- **Cost-model humility.** Tensat's per-op-sum model failed exactly where kernels
  interact. OCANNL's execution-based cost function (time the real compiled
  routine) sidesteps this class of error — a genuine architectural advantage of
  the BEAM-with-timing design worth preserving even for graph-level rewrites:
  measure the rewritten graph end-to-end, don't sum per-op estimates.

## Deep dive III: making BEAM converge (ROLLER, Pruner, Heron, tinygrad)

The in-progress BEAM search (tinygrad-style: beam over optop prefixes, on-device
timing, winner cache) will face the classic budget problem: compile+time per
candidate is tens of milliseconds to seconds, and the schedule space is
combinatorial in tile factors × axis assignments × staging choices. What the
literature says, in decreasing order of leverage:

1. **Construct hardware-aligned candidates instead of enumerating and pruning**
   (ROLLER). rTiles are tile shapes aligned to memory-transaction granularity,
   tensor-core shapes, and bank widths; construction walks the memory hierarchy
   expanding tiles greedily by a *static* throughput model. OCANNL already has the
   inputs: `hardware_limits` (max threads, shared-mem capacity) on
   `Backend_device_common`, `mma_simd_width`, per-node numel bounds, and
   `launch_dims`/`workgroup_shared` computed at compile time. Constraining `Split`
   factors to divisors aligned with these (and to extents — v1 shared `Stage`
   requires divisibility anyway) shrinks the beam's branching factor by orders of
   magnitude before any timing happens.
2. **Draft-then-verify ordering** (Pruner). Score every frontier extension with a
   cheap analytical fitness (occupancy proxy, shared-mem footprint vs. capacity,
   arithmetic intensity of the staged tile, divergence of remainder guards —
   all computable from the transformed `Low_level.t` without compiling); time only
   the top-k drafts. This is the "cost functions" half of the v0.9 milestone in
   its cheapest viable form — an *analytical filter* rather than a learned model.
   A learned model (Halide 2019, TLP) is not worth its training-set cost at
   OCANNL's scale; revisit only if the analytical filter's hit-rate stalls.
3. **Constraint-based generation for tensor-core schedules** (Heron). Post-
   `Tensorize`, the valid-schedule manifold is thin (lane loops, fragment shapes,
   `Stage` divisibility, barrier uniformity). Encoding validity as constraints and
   enumerating within them beats generate-and-reject; OCANNL's
   `validate_parallel` failures are already *named* — the same predicates can run
   forward as generators instead of backward as rejectors.
4. **Timing-harness robustness** (the Sakana "AI CUDA Engineer" fiasco, distilled):
   any execution-based cost function will be exploited by its search — not just by
   LLMs. Guard the harness: fixed warm-up/repeat protocol, parity gate *before*
   admission to the cache, cache keyed by canonicalized kernel hash + device
   identity (already scoped in schedule-ir-optops §8), and KernelBench's `fast_p`
   (fraction of workloads both correct and >p× faster) as the reporting metric for
   the search as a whole.

tinygrad remains the engineering reference (hand-coded action space, beam width
~4-8, persistent winner cache) but contributes no literature beyond what the
deep-dive blog article already extracted.

## Deep dive IV: trusting rewrites (TensorRight, Mirage's verifier, Axon)

Three trust models for rewrite rules, in increasing strength and cost:

- **Randomized testing** (Mirage §5): cheap, GPU-executable, principled error
  bounds within the Lax fragment; degrades gracefully to tolerance-aware float
  testing outside it. Right size for OCANNL *today* as the gate on hand-written
  graph-level rules and on BEAM candidates (follow-up F3).
- **Automated verification at arbitrary rank** (TensorRight, POPL'25): a DSL with
  *aggregated axes* — bundles of axes quantified as a unit — proving rewrites for
  unbounded rank/size; 115/175 XLA rules verified in full generality (vs. 18 for
  bounded verification). The formalism is strikingly close to OCANNL's row
  variables: a rule stated with `..b..` row variables *is* an aggregated-axes
  statement. Not an implementation target for v0.9 (Haskell/SMT toolchain, and
  OCANNL has no rule catalogue yet to amortize it), but the *research affinity* is
  worth a note in any workshop-paper follow-up: OCANNL's einsum specs with row
  variables are a rule language TensorRight-style verification could consume.
- **SMT over unbounded tensors inside the superoptimizer** (Axon, 2606.26344):
  the June 2026 preprint replacing probabilistic verification with sound synthesis.
  Unreviewed; watch, re-check at the 6-month mark.

## Technique-to-seam map

| Paper technique | OCANNL seam | Today | Change | Effort | Verdict |
|---|---|---|---|---|---|
| Mirage: parallelization-axis / grid-dim search | `Schedule` annotator + BEAM action space; `validate_parallel` | fixed Grid×Workgroup preset | axis-assignment actions (incl. split-K via `Workgroup_reduce`) | M | **File (F2)** |
| Pruner: draft-then-verify; ROLLER: aligned tiles | BEAM harness; `hardware_limits`, `launch_dims`, `workgroup_shared` | timing every candidate (planned) | analytical fitness filter + divisor-aligned `Split` factors | M | **File (F1)** |
| Mirage: randomized equivalence testing | parity harness (`hardware_axes_parity` generalized) | float parity vs. unscheduled twin | reusable oracle: seeded random inputs, tolerance policy, rounding-free randomized modes | S–M | **File (F3)** |
| Tensat: concat/split sharing rewrites | `Assignments.t` rewrites; einsum concat axes (`a^b`) | no graph-level rewriting | targeted rule schemas, measured accept/reject, no e-graph | M–L | **File (F4)** |
| Tensat/egg(log): e-graph infrastructure | new library or FFI | `ego` unmaintained+GPL; egglog is Rust | defer until rule interactions demonstrably bite | L | **Defer (F5 records decision)** |
| Mirage: μGraph enumeration + SMT pruning | would be a new generative search layer | schedule BEAM covers the schedule half | not needed while rewrites are hand-curated | L | **Skip** |
| Mirage: full finite-field verifier machinery | — | — | adopt the *idea* (F3), not the Lax-fragment apparatus | — | **Skip** |
| TensorRight: aggregated-axes verification | row-variable einsum specs as rule language | rules trusted by testing | research note / workshop-paper angle, not v0.9 code | L | **Skip for v0.9, note affinity** |
| Heron: constraint-based tensor-core search | `Tensorize` composition rules in `Schedule` | generate-and-reject via `validate_parallel` | run validity predicates forward as generators | M | Fold into F1/F2, not separate |
| Minotaur: SIMD peephole superoptimization | vectorized loop bodies (`c_syntax` SIMD rendering) | rule-based vectorization (gh-164) | offline synthesis of missed peepholes | L | **Skip** (revisit post-v0.9 if SIMD underperforms) |
| MPK: megakernel / grid-sync execution | kernel-fission segmentation | event chains at boundaries | CUDA cooperative-launch over same segmentation | L | Already noted in schedule-ir-optops §7; no new issue |
| LLM-guided kernel search | — | — | — | — | **Watch only** |

## Recommended follow-ups (to file on ahrefs/ocannl)

**F1 — BEAM candidate construction: hardware-aligned actions + analytical draft
filter** (Pruner + ROLLER + Heron). Constrain `Split` factors to
divisor/alignment sets derived from `hardware_limits`, `mma_simd_width`, and
extents; score frontier extensions with a static fitness (occupancy proxy,
shared-mem footprint, remainder-guard divergence) computed from the transformed
IR; time only top-k. *Seam:* the BEAM harness + `schedule.ml` preset machinery.
*Effort:* medium. *Prereq:* BEAM v1 lands (in progress). This is the core
"cost functions" deliverable of v0.9.

**F2 — Parallelization-axis choice in the schedule search space** (Mirage's GQA
result). Add axis-assignment actions: which loops retype to `Grid`/`Workgroup`,
reduction-axis parallelization via `Workgroup_reduce` + `Privatize` (split-K),
per-shape rather than fixed. *Seam:* `Schedule` optops (exist), default annotator,
BEAM action space. *Effort:* medium. *Prereq:* F1 (otherwise the branching factor
explodes the beam).

**F3 — Randomized equivalence oracle** (Mirage §5, scaled down). A reusable
harness: seeded random inputs, per-precision tolerance policy, rounding-free
integer/finite-field modes (bounded false-positive probability for the linear
fragment; probabilistic-only for max-plus — see the research notes §4.2); gates
BEAM candidates
(cheap re-gate on cache hits) and any future `Assignments.t` rewrite rule.
*Seam:* generalize `test/operations/hardware_axes_parity.ml` into a library
function. *Effort:* small–medium. *Prereq:* none. File first — F4 depends on it.

**F4 — Sharing rewrites over `Assignments.t` via concat axes** (Tensat's winning
family + Mirage's LoRA identity). Rule schemas: k matmuls sharing an operand →
concat+matmul+split; weight-concat precompute; (W‖B)×(X‖(A×X)) for LoRA-shaped
graphs. Expressed with einsum concat specs, applied destructively under a
measured accept/reject gate — no e-graph. *Seam:* `assignments.ml` (`Accum_op`),
`tensor/operation.ml` einsum builders. *Effort:* medium–large. *Prereq:* F3;
the rules-as-data audit (#296) for provenance discipline. Honest framing: this
is the "broader rewriting rules" half of v0.9 and may slip to v0.9.x — the
demonstrated wins (Tensat: multi-branch inference graphs; Mirage: LoRA) are
narrower than OCANNL's current training-loop benchmarks.

**F5 — Decision record: defer e-graph infrastructure.** Not an implementation
issue; a short issue recording *why* (ego unmaintained + GPL, egglog = Rust FFI,
from-scratch ~1–2 kLOC core unjustified below ~dozens of interacting rules),
the trigger for revisiting (F4 rules start needing phase-ordering machinery /
a second rule family lands), and the reading map for that day (Tensat + PACT'24
MCTS extraction + egglog). *Effort:* small (writing only).

**Effort plausibility vs. v0.9 (due 2026-08-24):** F1+F2+F3 fit the milestone and
constitute its program-search core; F4 is the stretch item — file it in v0.9,
expect completion in v0.9.x; F5 is an afternoon. This is a triage, not a wishlist:
if the BEAM v1 slips past July, drop F2 from v0.9 before dropping F1.

## Skipped techniques (explicit, per AC)

1. **Mirage's generative μGraph enumeration + SMT-based abstract-expression
   pruning.** OCANNL's search is over semantics-preserving schedule transforms of
   a *given* program; nothing is generated whose derivability toward a goal needs
   proving. Adopting it would mean a bounded operator vocabulary (Mirage hand-added
   an op to win LoRA), a Z3 dependency, and 4-hour-class search budgets — against
   OCANNL's einsum front-end which already expresses the algebra Mirage searches
   for. Revisit only if hand-curated rewrite rules (F4) prove too narrow.
2. **Mirage's full probabilistic-verifier apparatus** (Lax fragment, linked
   finite fields, error-bound theorems). The *idea* ships in F3; the apparatus is
   insurance OCANNL's deterministic, locally-provable rewrites don't need yet.
3. **Equality saturation as infrastructure** — deferred with a written decision
   record (F5) rather than silently dropped.
4. **TensorRight-style automated rule verification** for v0.9 — no rule catalogue
   to amortize it yet; the row-variable affinity is recorded as a research note.
5. **Learned cost models** (Halide 2019, TLP) — training-set cost unjustified at
   OCANNL's kernel diversity; the analytical filter (F1) is the right first rung.
6. **LLM-guided kernel optimization** — watch-only; the transferable artifact is
   the Sakana cautionary tale, folded into F1's harness-robustness requirements.

## Acceptance criteria (revised)

- [x] **Write-up exists** — two-tier: this document is the decision record (kept
  at `docs/proposals/gh-ocannl-261.md` with the rest of the deep-dive family),
  and `docs/research/superoptimizers.md` (the originally planned AC path) holds
  the detailed extractions and survey. Technique-by-technique mapping with named
  files/passes: see the seam map above.
- [ ] **≥3 follow-up tasks filed** — F1–F5 above are ready to transcribe into
  GitHub issues (F3 first, then F1, F2, F4, F5), each cross-referenced from a
  comment on #261.
- [x] **≥1 explicit skip decision** — six, with reasoning (previous section).
- [x] **Effort plausibility** — stated with a triage order (drop F2 before F1 if
  BEAM v1 slips).
- [ ] **Issue #261 comment** — post after filing: link this write-up, list F1–F5,
  list the skips, recommend closing #261 as the meta-tracker once the fan-out
  exists.
- [x] **No implementation** — no code changes under `arrayjit/`, `tensor/`, `lib/`.

## Known constraints (refreshed)

- **BEAM v1 is in flight, elsewhere.** F1/F2 must land as extensions of that
  harness, not competitors; coordinate before filing so the issues name the
  actual module.
- **OCaml e-graph gap persists** — but is now a recorded deferral (F5) with
  verified facts (ego 0.0.6/Nov-2021/GPL-3; egglog = Rust), not a blocker.
- **Measured-gate costs compound.** F4's accept/reject and F3's oracle both run
  real compiles; they inherit F1's harness-robustness requirements (seeded
  inputs, warm-up protocol, canonicalized cache keys).
- **The schedule layer's v1 restrictions bound the search space**: shared `Stage`
  requires tile-divides-extent; `Swap` requires perfect nesting; barrier placement
  rejects divergent control flow. These are search-space *constraints to encode*
  (Heron-style), not bugs to fix under #261.

## Notes

- Original two-paper scope, phasing, and the 2026-06-12 status snapshot are
  superseded by this revision; consult git history for the scouting-era text.
- Estimated remaining effort: small — transcribe F1–F5 into issues (~half a day,
  after coordinating F1/F2 wording with the BEAM branch), post the #261 comment.
- `start_confidence` rationale updated: the judgment calls are now made and
  documented; what needs user review before filing is the F1/F2 split against
  the in-progress BEAM work, and whether F4 targets v0.9 or v0.9.x.

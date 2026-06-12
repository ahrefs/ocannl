# Superoptimizers for tensor programs: literature review and follow-up plan

**Issue:** [ahrefs/ocannl#261](https://github.com/ahrefs/ocannl/issues/261)
**Status:** Draft proposal — research / scouting task
**Milestone:** v0.9 (Program search with execution-based cost functions)

## Status update (2026-06-12)

- Issue [#261](https://github.com/ahrefs/ocannl/issues/261) is **OPEN**, milestone **v0.9** (GH milestone due-date 2026-05-30 is stale; per ROADMAP.md, v0.9 targets Aug 24, 2026 — ICFP week).
- The deliverable (`docs/research/superoptimizers.md` write-up + follow-up issues) has **not** been produced; `docs/research/` still does not exist (the sibling deep-dive write-ups live as proposals under `docs/proposals/`, not under `docs/research/` — adjust the AC path or create the directory when executing).
- Sibling-task states corrected: #242 (TVM) CLOSED/completed; #301 (IREE) and #306 (Petalisp/Caten) CLOSED as **not planned** (no `docs/research/` write-ups landed for them). #267 (Tiramisu) and #265 (Candle) remain OPEN as draft proposals.
- Line-number drift in `arrayjit/lib/low_level.ml` (re-verified 2026-06-12): `optimize_proc` now at line 1619; `simplify_llc` at 1014; `eliminate_common_subexpressions` at 1317; `hoist_shared_locals` at 1504; `hoist_cross_statement_cse` at 1586. `Assignments.to_low_level` (line 190) and `Assignments.lower` (line 983) unchanged. `c_syntax.ml` is now ~981 lines, `tensor/tensor.ml` ~1105 lines.
- The CUDA single-threaded baseline still holds: kernels launch with `grid_dim_x:1, block_dim_x:1` in `cuda_backend.ml`.
- Repo-wide changes since April 2026 that new text should respect: broadcast-order reversal (LUB→GLB, meet→join, "⊑" reads "refines"), dimension "label"→"basis" rename, "invalid"→"discardable" rename, and `device_to_device` now returning a transfer routine with static merge-buffer verification (`merge_buffer_use = No | Copy`). None of these invalidate the design content here, but rewrite-rule work over `Assignments.t` should use the post-reversal vocabulary.
- Verdict: still actionable as written (research + issue fan-out); nothing has been implemented or filed yet.

## Goal

This is a **research / scouting task**, not an implementation task. The
deliverable is a literature-anchored set of *concrete follow-up tasks* for
OCANNL's compilation pipeline, derived from two papers:

- *A Multi-Level Superoptimizer for Tensor Programs* — "Mirage", arxiv
  [2405.05751](https://arxiv.org/abs/2405.05751), OSDI'25. Multi-level
  superoptimizer over GPU kernel / thread-block / thread tiers, unified by a
  μGraphs IR, with abstraction-based pruning and a probabilistic equivalence
  verifier. Reported up to 3.3× over heavily-tuned baselines on DNNs.
- *Equality Saturation for Tensor Graph Superoptimization* — arxiv
  [2101.01332](https://arxiv.org/abs/2101.01332). Applies *all* available
  rewrite substitutions simultaneously via e-graphs to escape the
  ordering sensitivity of sequential graph rewriters; reports up to 16%
  speedup over (TASO-class) state of the art at ~48× lower optimisation
  cost. Widely associated with the `egg` / Rust e-graph library.

Success looks like: an OCANNL maintainer reading the resulting follow-up
tasks should be able to pick one and start work without re-reading the
papers — every follow-up names a specific paper technique, a specific
OCANNL file, and a specific prerequisite chain.

The format mirrors the closed deep-dive scouting tasks
gh-ocannl-242 (TVM), gh-ocannl-301 (IREE/MLIR), and gh-ocannl-306
(Petalisp / Caten), and the in-flight gh-ocannl-267 (Tiramisu) — each of
which produces a write-up plus a fan-out of issues. This proposal explicitly
inherits that pattern.

## Acceptance Criteria

- [ ] **Write-up exists.** A Markdown document at
  `docs/research/superoptimizers.md` (create the directory if absent —
  see gh-ocannl-267 which uses the same path) summarises both papers,
  with a one-paragraph technique-by-technique mapping to OCANNL's
  compilation seams (named files, named functions / passes).
- [ ] **≥3 follow-up tasks filed.** At least three concrete follow-up
  GitHub issues exist on `ahrefs/ocannl`, each cross-referenced from
  the write-up and from a comment on issue #261. Each follow-up:
  - Names a *specific* paper technique (e.g. "Mirage's μGraphs
    kernel-level rewrites", "Tensat-style e-graph extraction
    over `Assignments.t`", "abstraction-based pruning of the
    schedule search space").
  - Names a *specific* OCANNL file or pass it would touch
    (e.g. `arrayjit/lib/low_level.ml`'s `optimize_proc` pipeline,
    `arrayjit/lib/assignments.ml` `to_low_level`, the schedule layer
    proposed in gh-ocannl-267).
  - Carries an effort estimate (small / medium / large) backed by a
    paragraph that names its prerequisites, including which existing
    OCANNL issues / proposals must land first.
- [ ] **≥1 explicit *skip* decision.** At least one paper technique
  is evaluated and *ruled out* in the write-up, with reasoning
  (cost, OCaml ecosystem fit, scope mismatch with v0.9 goals, etc.).
  This locks the rule that the deliverable is honest evaluation, not
  rubber-stamped task creation.
- [ ] **Effort plausibility.** The total estimated effort across all
  filed follow-ups is ≤ "the v0.9 milestone budget can absorb this",
  i.e. the worker has not produced a wishlist that obviously overflows
  the milestone. If the honest answer is "this is more than v0.9 can
  hold", that fact is stated and a triage recommendation is given.
- [ ] **Issue 261 comment.** A comment on
  [ahrefs/ocannl#261](https://github.com/ahrefs/ocannl/issues/261)
  links to the write-up and to each filed follow-up issue, lists
  the techniques explicitly skipped, and recommends whether to close
  #261 (it is the meta-tracker — closing on completion is appropriate
  if all follow-ups are filed).
- [ ] **No implementation.** No code changes are made under
  `arrayjit/`, `tensor/`, or `lib/`. The proposal scope is research +
  issue filing only. (Failure of this criterion would mean the worker
  silently turned an explore task into an implementation task.)

## Context

### Why now

The v0.9 milestone description on GitHub reads: *"Program search with
execution-based per-backend or aggregate-of-backends cost functions.
Starting with augmenting the tiling and layout mechanisms from v0.8 with
cost functions, progressing to a broader range of code graph rewriting
rules."* Both papers in scope are direct prior art for that milestone:

- Mirage's multi-level search is a candidate architecture for the
  "program search" half.
- Equality saturation is a candidate architecture for the "broader range
  of code graph rewriting rules" half.

Filing follow-ups now lets v0.9 work plug into one of these designs
deliberately rather than re-deriving the design space mid-implementation.

### What OCANNL already does at each level the papers target

The Tentative Design in `tasks/gh-ocannl-261.md` (dated 2026-02-08) is
partially stale and should be re-checked by the worker. Verified state
of the world as of 2026-04-30:

- **Lowering boundary:**
  `arrayjit/lib/assignments.ml` → `Assignments.to_low_level`
  (around line 190) and `Assignments.lower` (around line 982) — the
  bridge from the high-level `Assignments.t` IR to `Low_level.t`.
- **Optimisation pipeline** (`arrayjit/lib/low_level.ml`):
  `optimize_proc` at line ~1619 *(line numbers re-verified 2026-06-12)* composes
  `cleanup_virtual_llc → simplify_llc → eliminate_common_subexpressions
   → hoist_cross_statement_cse`. `simplify_llc` itself is at line 1014
  (the Tentative Design's pointer to lines 1007–1192 is approximately
  correct — the body now extends to ~1180).
- **Already landed (stale claims in the task file):**
  - gh-ocannl-351 (CSE after inlining) — **CLOSED**. CSE is
    `eliminate_common_subexpressions` at line 1317.
  - gh-ocannl-350 (loop hoisting / loop-invariant code motion) —
    **CLOSED**. Hoisting is `hoist_cross_statement_cse` /
    `hoist_shared_locals` at lines 1504–1618.
  - gh-ocannl-25 (loop fusion exploration) — **CLOSED**.
  - gh-ocannl-131 (single product_space iteration for grouped
    accumulations) — **CLOSED**.
  - gh-ocannl-242 (TVM deep dive), gh-ocannl-301 (IREE), gh-ocannl-306
    (Petalisp / Caten) — **CLOSED**. *(Update 2026-06-12: only #242 was
    closed as completed; #301 and #306 were closed as not planned, and
    no `docs/research/` write-ups exist — the deep-dive material lives
    in the corresponding `docs/proposals/gh-ocannl-NNN.md` files.)*
- **Still open and adjacent:**
  - gh-ocannl-267 (Tiramisu deep dive) — proposal already at
    `docs/proposals/gh-ocannl-267.md`. Identifies the *missing
    schedule layer* between `Assignments.to_low_level` and
    `optimize_proc` as the central structural gap. The
    superoptimizer follow-ups must cross-reference this — both
    papers operate above any reasonable schedule layer and a schedule
    layer is a likely prerequisite.
  - gh-ocannl-265 (Candle scouting), gh-ocannl-261 (this).
  - watch-ocannl-README-md-1c953381 (program search infrastructure,
    `large`, blocked) and watch-ocannl-README-md-d7a63af1
    (reproduce tinygrad / Halide search, depends on the former) —
    these are the milestone tasks the follow-ups should slot into.
- **Codegen:** `arrayjit/lib/c_syntax.ml` is a shared C-like emitter
  used by CC / CUDA / Metal backends. Instruction-level rewrites
  (vectorisation, register tiling) would land here or earlier. CUDA
  kernels currently run with `grid_dim=1, block_dim=1`
  (`kernel_prep_line` in `cuda_backend.ml`) — the entire v0.8 GPU
  performance milestone starts from that baseline, which is relevant
  for any "Mirage GPU thread-block tier" follow-up.

### Quality-audit pause

Per Mag memory, the user is currently doing a hands-on OCANNL quality
audit; autonomous OCANNL work is paused. This proposal is being drafted
so that follow-ups can be queued for after the audit; the task itself
should remain `deferred_launch: true` and not auto-start.

## Approach

*Suggested methodology — the worker may deviate if a better path
emerges. The structure mirrors gh-ocannl-242 / gh-ocannl-267.*

### Phase 1 — Read

1. Read both abstracts + introductions + the experimental sections.
   For Mirage (2405.05751), focus on §3 (μGraphs) and §4 (search /
   pruning) and the equivalence-verification section. For the
   equality-saturation paper (2101.01332), focus on §3 (e-graph
   construction over tensor graphs) and the rewrite-rule catalogue.
2. Skim the closed sibling write-ups (`docs/research/tvm.md`,
   `docs/research/iree.md`, `docs/research/petalisp-caten.md` if those
   are the actual paths used by gh-ocannl-242 / 301 / 306 — verify) and
   the in-flight Tiramisu plan at `docs/proposals/gh-ocannl-267.md`,
   to inherit the table format and to avoid duplicating discussion of
   schedule layers / cost models.

### Phase 2 — Map techniques to OCANNL seams

For each technique extracted in Phase 1, fill in a row in a table with
columns:

| Paper technique | Closest OCANNL seam | What OCANNL does today | What the change would look like | Effort | Prerequisites |

Seams to consider, with concrete file pointers:

- **Graph-level rewrites** → `arrayjit/lib/assignments.ml` (`Accum_op`,
  `Set_vec_unop`, `Fetch`, `sequential` / `sequence` builders);
  `tensor/tensor.ml` (operator definitions, ~1105 lines). Equality
  saturation would build an e-graph over `Assignments.t` or a
  pre-`Assignments` form; this is the most natural fit for the
  e-graph paper.
- **Schedule-level rewrites** → currently *missing* (per gh-ocannl-267).
  Mirage's μGraphs sit at this level. A follow-up proposing a schedule
  layer is a likely prerequisite for the Mirage-derived work and
  should be cross-linked with gh-ocannl-267 rather than duplicating it.
- **Loop-level rewrites** → `arrayjit/lib/low_level.ml`'s
  `optimize_proc` pipeline (line 1595). Note that
  `simplify_llc`, `eliminate_common_subexpressions`, and
  `hoist_cross_statement_cse` *already exist* — Mirage-style
  thread-block / loop-tile rewrites would compose with these, not
  replace them. Identify the natural insertion point in the pipeline.
- **Instruction-level rewrites** → `arrayjit/lib/c_syntax.ml`
  (~981 lines) and the per-backend emitters
  (`cc_backend.ml`, `cuda_backend.ml`, `metal_backend.ml`). Mirage's
  thread-tier rewrites correspond to choices made here.

### Phase 3 — Filter and file

For each row, decide one of:
- **File.** Write a GitHub issue on `ahrefs/ocannl` with title, body,
  effort estimate, and a prerequisite-chain paragraph. Cross-reference
  from the write-up and from the comment on #261.
- **Skip.** Record the reason in a "Skipped techniques" section of the
  write-up. At least one skip is required (AC).
- **Defer.** A "Future work past v0.9" subsection captures techniques
  that are interesting but out of scope for this milestone (e.g.
  probabilistic equivalence checking — likely defer; OCaml e-graph
  bindings — likely defer).

Likely-promising leads the worker should test against the seams above
(non-binding — these are starting hypotheses, not conclusions):

1. **Equality saturation over `Assignments.t`.** Smaller and more
   targeted than reimplementing the full TASO substitution catalogue.
   Would need an e-graph; see Known Constraints.
2. **Mirage-style multi-level search at the loop level only.** Skip
   the GPU thread / thread-block tiers initially (CUDA is single-threaded
   today), keep μGraphs-style multi-level representation as a design
   target for when a schedule layer exists.
3. **Abstraction-based pruning** (Mirage §4) as a *technique*
   independent of μGraphs: applicable to any future search loop in
   `optimize_proc` or in the v0.9 program-search infrastructure
   (watch-ocannl-README-md-1c953381).
4. **Probabilistic equivalence verification** — almost certainly
   *skip / defer*: OCANNL's algebraic rewrites are currently
   deterministic and locally provable, and a probabilistic verifier
   is a heavy piece of infrastructure to introduce purely as
   insurance. Worth at least documenting the skip.

### Phase 4 — Comment + close-or-leave

Post a single comment on issue #261 listing: write-up link, filed
follow-ups (with one-line summary each), explicitly skipped techniques.
Recommend whether to close #261 (yes, if all follow-ups are filed —
it is a meta-tracker and the deliverable is the fan-out).

## Known Constraints

These constraints should be surfaced explicitly in the write-up (they
are part of what makes some techniques skip / defer rather than file):

- **No OCaml e-graph library.** The dominant e-graph implementation
  is `egg` (Rust). Adopting equality saturation in OCANNL means one of:
  (a) write OCaml bindings to `egg` (FFI complexity, build-system
  pain, deployment cost — non-trivial for an academic OCaml
  project); (b) reimplement an e-graph library in OCaml (substantial
  upfront cost, narrows it to a hobbyist/research effort); (c) shell
  out to a Rust binary at compile time (deployment friction). All
  three are heavy lifts. *Any equality-saturation follow-up must
  carry an explicit prerequisite for resolving this.*
- **No schedule layer yet.** Mirage's μGraphs presuppose schedule-level
  control over kernel / block / thread structure. OCANNL's loop nest
  is fixed by `product_space` order at lowering time. A
  Mirage-derived follow-up is plausibly *blocked by* the
  schedule-layer work proposed in gh-ocannl-267.
- **CUDA single-threaded baseline.** All CUDA kernels currently run
  with `grid_dim=1, block_dim=1` (Mag memory; `kernel_prep_line` in
  `cuda_backend.ml`). Mirage's GPU thread-block / thread-tier
  rewrites are not directly applicable until v0.8 GPU-parallelism
  work lands. This is a v0.8 → v0.9 ordering constraint, not a
  showstopper, but should be noted.
- **OCANNL workshop-paper deadlines.** Per Mag memory, the project
  is targeting OCaml Workshop / FProPer 2026 (May–June deadlines). A
  follow-up that claims to be tractable inside v0.9 must be honest
  about that calendar.

## Scope

**In scope:**
- The `docs/research/superoptimizers.md` write-up (worker may rename if
  a different convention is in force — verify against existing
  `docs/research/` contents).
- A comment on issue #261.
- Filing follow-up issues on `ahrefs/ocannl`, cross-referenced.

**Out of scope:**
- Implementing any superoptimizer technique. Each follow-up issue is
  its own task once filed.
- Building or running Mirage / Tensat. The published abstracts plus
  paper bodies (and the open-source repos for spot-checks) suffice.
- Re-deriving the schedule-layer discussion already in gh-ocannl-267 —
  cross-reference it.

**Dependencies:**
- Closed sibling write-ups: gh-ocannl-242 (TVM), gh-ocannl-301 (IREE),
  gh-ocannl-306 (Petalisp/Caten). Format and depth target.
- In-flight: gh-ocannl-267 (Tiramisu) — overlaps on schedule-layer
  discussion; cross-reference.
- v0.9 milestone tasks: watch-ocannl-README-md-1c953381 (program
  search) and watch-ocannl-README-md-d7a63af1 (reproduce tinygrad /
  Halide search). The filed follow-ups should slot into one of these
  or be filed as siblings.

## Notes

- Estimated effort: medium (3–5 days), primarily reading + writing.
  The worker should not try to compress this into a single session
  if the paper bodies are dense; phase 1 (reading) and phase 3 (filing)
  benefit from a sleep in between.
- This proposal does not auto-start: `start_confidence: low` reflects
  that the deliverable is a judgment call about which paper techniques
  matter for OCANNL, and that judgment should be reviewed by the user
  before issues are filed against `ahrefs/ocannl`.

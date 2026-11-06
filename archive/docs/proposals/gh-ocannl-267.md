# Tiramisu Polyhedral Compiler: Architecture Comparison and Transferable Ideas for OCANNL

**Issue:** [ahrefs/ocannl#267](https://github.com/ahrefs/ocannl/issues/267)
**Status:** Draft proposal

## Status update (2026-06-12)

- Issue [#267](https://github.com/ahrefs/ocannl/issues/267) is **OPEN**, milestone **v0.9** (GH milestone due-date 2026-05-30 is stale; per ROADMAP.md, v0.9 targets Aug 24, 2026 — ICFP week, matching the "ICFP milestone" framing below).
- The deliverable (`docs/research/tiramisu.md` + issue comment) has **not** been produced; `docs/research/` does not exist yet (harness task status: deferred).
- Code claims re-verified: `Indexing.axis_index`'s `Affine` variant, the `projections` record (`product_space` / `product_iterators`), `Low_level.t`'s `For_loop`, and the memory-mode classification (Virtual / Materialized / Device_only / Hosted) all match current code; CUDA kernels still launch with `grid_dim_x:1, block_dim_x:1`. The central claim — no schedule layer, no split/tile/reorder/fuse pass — still holds.
- `Low_level.optimize_proc` has grown since drafting: the pipeline now also runs common-subexpression elimination (`eliminate_common_subexpressions`, #351) and cross-statement CSE with hoisting (`hoist_cross_statement_cse` / `hoist_shared_locals`); it remains free of loop-restructuring transformations.
- Referenced-issue drift: #242 (TVM) is now CLOSED/completed; #320 (Metal private storage mode) is CLOSED/completed (landed in commit 1cf9a95b) — area 5's "Metal private-mode optimisation" is no longer pending work; #350 (loop hoisting) was closed as not planned (hoisting landed via cross-statement CSE instead); #344 (pool allocator) and #412 (matmul/tensor cores) remain OPEN; #278 (DisTrO) remains OPEN on v1.1.
- Communication-layer drift: `Backend.device_to_device` now **returns a transfer routine** (with static merge-buffer verification; `merge_buffer_use = No | Copy`, the streaming variant is gone) — transfers are closer to "declared" first-class objects than the "currently imperative" description in area 6/Context suggests. Multi-streaming cleanup also removed cross-stream automatic coherence (multiple streams per device remain).
- New-text vocabulary note: since April 2026 the broadcast order was reversed (LUB→GLB, meet→join, "⊑" reads "refines") and dimension "label" was renamed to "basis" — use the new terms in the eventual write-up.
- Verdict: still actionable as written; the schedule-layer gap analysis is unchanged and remains the central question for v0.9.

## Goal

[Tiramisu](https://github.com/Tiramisu-Compiler/tiramisu) is a polyhedral
compiler framework (C++, built on ISL + LLVM + Halide) that targets multicores,
GPUs, FPGAs, and distributed clusters via MPI. Its central design idea is a
**four-level intermediate representation** that fully separates: (1) the
algorithm, (2) loop transformations / schedule, (3) data layout, and (4)
communication. The reference paper is [arxiv 1804.10694](https://arxiv.org/abs/1804.10694).

OCANNL is fundamentally an affine-loop compiler — it already represents
iteration spaces as products of integer ranges and indexes them with affine
expressions — but it has no schedule layer separate from its lowering, no
explicit data-layout abstraction, and no polyhedral dependence analysis.
Tiramisu is one of the cleanest published references for what a fully
separated schedule layer looks like, and the v0.9 ICFP milestone (program
search, Halide-inspired) is exactly the point at which that separation
becomes load-bearing.

This is a research / scouting task. The deliverable is a comparison document
(this proposal serves as its outline) and a GitHub issue comment summarising
actionable findings, mirroring the format used for the TVM deep dive
(gh-ocannl-242) and reusing its structure where it overlaps.

## Acceptance Criteria

- [ ] Tiramisu's four-level IR (algorithm, schedule, data layout,
  communication) is summarised in a way that maps each level to its OCANNL
  counterpart (or notes its absence).
- [ ] Tiramisu's scheduling primitives are catalogued and compared against
  what OCANNL already exposes; primitives that would be valuable for the v0.8
  tiling milestone or the v0.9 program-search milestone are flagged with an
  effort/value assessment.
- [ ] Tiramisu's use of the polyhedral model (ISL iteration domains,
  dependence analysis, scheduling via affine transformations) is assessed for
  applicability to OCANNL's existing affine projections, with an explicit
  judgement on whether OCANNL should adopt ISL-style polyhedral analysis,
  build a lighter-weight equivalent, or stay with its current approach.
- [ ] Tiramisu's auto-scheduler is compared with OCANNL's planned program
  search (v0.9). The comparison identifies what OCANNL can borrow versus
  where its differing scope (no graph IR, OCaml type-safe DSL, smaller
  primitive set) calls for a different approach.
- [ ] The findings are posted as a comment on
  [ahrefs/ocannl#267](https://github.com/ahrefs/ocannl/issues/267), and the
  issue is closed when the write-up is merged.
- [ ] Any concrete transferable ideas that warrant implementation are filed
  as separate GitHub issues (and linked from the comment); this task does
  not implement any of them.

## Context

### OCANNL's current loop / index machinery

Key types live in `arrayjit/lib/indexing.ml` and `arrayjit/lib/low_level.ml`:

- `Indexing.axis_index` — the `Affine { symbols; offset }` variant captures
  index expressions as an integer offset plus a list of `(coefficient, symbol)`
  terms. This is already the polyhedral notion of an affine access function,
  one access at a time.
- `Indexing.projections` — record bundling `product_space : int list array`
  (the iteration domain, one bound list per axis) and
  `product_iterators : symbol list array` (the iterator symbols per axis),
  together with per-tensor `project_lhs` / `project_rhs` projections. This
  is a single iteration space with affine projections to each operand —
  essentially Tiramisu's `computation` plus its access relations, but with
  no separate schedule.
- `Low_level.t` — C-like imperative IR with `For_loop { index; from_; to_;
  body; trace_it }`, `Set`, `Get`, scalar ops. Loop nesting order is fixed
  at lowering time from the order of `product_space` axes.
- `Low_level.optimize_proc` — the post-lowering optimisation pipeline:
  tracing-based visit counting, virtualisation (inlining of single-access
  tensors), cleanup, and algebraic simplification. *(Update 2026-06-12: the
  pipeline now also includes common-subexpression elimination and
  cross-statement CSE with hoisting.)* There is no schedule
  transformation pass (no split, tile, reorder, fuse) at this layer.
- `arrayjit/lib/assignments.ml` — high-level `Accum_op`, `Set_vec_unop`,
  `Fetch`. `Assignments.to_low_level` is the lowering boundary.
- `arrayjit/lib/c_syntax.ml` — shared C-like emitter parameterised by
  `C_syntax_config`; CUDA / Metal / CC backends plug in here. CUDA kernels
  currently run with `grid_dim=1, block_dim=1` (see `kernel_prep_line`
  in `cuda_backend.ml`), so all GPU parallelism is latent.

### Tiramisu at a glance

Tiramisu's four-level IR (per the paper):

1. **Algorithm** — pure mathematical expression of *what* is computed,
   independent of order, storage, or communication. Closest OCANNL analogue:
   the `%op` / `%cd` syntax extensions plus `Assignments.t`.
2. **Schedule** — loop transformations applied to the algorithm to produce
   a concrete loop nest. Primitives include `split`, `tile`, `interchange`,
   `fuse`, `unroll`, `vectorize`, `parallelize`, `gpu_tile`,
   `compute_at`, `storage_fold`. Implemented as ISL schedule trees.
   No OCANNL analogue — loop structure is fixed by `product_space` order
   and `optimize_proc` only does virtualisation / simplification (plus,
   as of 2026, CSE and hoisting — still no loop restructuring).
3. **Data layout** — buffer allocation, storage mappings, dimension
   permutations, padding. OCANNL's analogue is the `memory_mode`
   classification (Virtual / Materialized / Device_only / Hosted) plus the
   planned Universal Pool Allocator (#344); there is no explicit storage
   permutation or padding spec.
4. **Communication** — explicit MPI send/recv, GPU host↔device transfers,
   distributed schedules. OCANNL's analogue is the merge-buffer machinery
   and backend-specific copy operations; communication is currently
   imperative rather than declared. *(Update 2026-06-12: partially shifted —
   `device_to_device` now returns a transfer routine, a first-class
   schedulable object, with static merge-buffer verification.)*

Tiramisu uses ISL for: representing iteration domains as Presburger sets,
checking dependences via Pluto-style schedule construction, and emitting
loops via `isl_ast_build`.

### Sibling research tasks already drafted

- gh-ocannl-242 (TVM deep dive, closed as completed) — proposal exists at
  `docs/proposals/gh-ocannl-242.md`. Establishes the table-of-primitives
  format and the "schedule primitives vs. OCANNL" mapping. The Tiramisu
  write-up should explicitly cross-reference it rather than re-derive the
  parts that overlap (notably the loop-transformation table).
- gh-ocannl-301 (IREE / MLIR), gh-ocannl-261 (superoptimizers),
  gh-ocannl-306 (Petalisp/Caten) — same family of scouting tasks; the
  Tiramisu write-up should note overlap.
- gh-ocannl-350 (loop hoisting) *(Update 2026-06-12: closed as not planned —
  hoisting landed via cross-statement CSE instead)*, gh-ocannl-412
  (matmul / tensor cores, still open) —
  v0.7.2 / v0.8 work where Tiramisu primitives become directly actionable
  (tile / interchange / vectorize / gpu_tile).

## Approach

*Suggested approach — agents may deviate if they find a better path.*

The agent should produce a single Markdown document at
`docs/research/tiramisu.md` (parallel to where future deep-dive write-ups
will live; create the directory if needed) covering the seven sub-areas
below, then post a condensed summary as a comment on issue #267. The
proposal file itself can be updated in place to point at the final write-up.

The seven areas to cover, in order:

1. **Four-level IR mapping.** Tabulate Tiramisu's four levels against
   OCANNL's machinery (algorithm ↔ `%op`/`Assignments.t`; schedule ↔
   *missing*; data layout ↔ `memory_mode` + pool allocator; communication
   ↔ merge buffers + backend copies). Identify the schedule layer as the
   biggest structural gap.
2. **Schedule primitives table.** Produce a TVM-style table (`split`,
   `tile`, `reorder` / `interchange`, `fuse`, `unroll`, `vectorize`,
   `parallel`, `gpu_tile`, `compute_at`, `storage_fold`) with three
   columns: Tiramisu primitive → current OCANNL equivalent → v0.8/v0.9
   relevance. Reuse the gh-ocannl-242 table where the entries match TVM
   exactly; flag Tiramisu-distinctive primitives (`storage_fold`,
   distributed schedules).
3. **Polyhedral model fit.** Concrete question: do OCANNL's
   `product_space` × `Affine` indices already form a Presburger
   representation suitable for ISL-style analysis, or do quirks
   (multi-symbol affine terms, projection ordering, virtual tensors)
   make a direct translation lossy? Recommend one of: (a) integrate ISL
   via FFI for dependence analysis only, (b) implement a lightweight
   in-house polyhedral analyser tailored to OCANNL's affine subset,
   (c) skip polyhedral analysis and rely on syntactic schedule rewrites.
4. **Auto-scheduler vs. OCANNL program search.** Compare Tiramisu's
   auto-scheduler (beam search over schedule sketches with a learned
   cost model) against the v0.9 plan. Note that OCANNL's per-backend
   execution-based cost functions (ROADMAP v0.9) align with Tiramisu's
   measurement-driven approach. Flag the design decision: schedule
   *templates* (AutoTVM-style, user supplies knobs) vs. *sketches*
   (Ansor/Tiramisu-style, generated from the algorithm) vs. pure
   rewrite-rule search.
5. **Data layout & memory.** Tiramisu separates buffer allocation and
   storage mappings from computation. Assess whether OCANNL's planned
   Universal Pool Allocator (#344) plus memory modes already cover this,
   or whether explicit dimension-permutation / padding annotations
   (à la `storage_fold` / `buffer.reshape`) would be valuable for the
   v0.8 tiling work and Metal private-mode optimisation (#320 — *Update
   2026-06-12: completed; Metal now uses private storage mode for
   GPU-only buffers*).
6. **Communication layer.** OCANNL is single-machine (no MPI); Tiramisu's
   communication layer is mostly out of scope. Briefly note that the
   merge-buffer mechanism is OCANNL's analogue and that adopting an
   explicit communication-as-IR view could be revisited if distributed
   training (#278 DisTrO) becomes a target post-1.0.
7. **Verdict & filed follow-ups.** Concrete recommendations:
   - Whether to add a schedule-tree IR layer between
     `Assignments.to_low_level` and the optimisation pipeline (likely
     yes, scoped to v0.9).
   - Whether to depend on ISL (likely no — too heavy a dependency for an
     OCaml project; build a small affine-analysis module instead).
   - Specific primitives to prototype during v0.8 tiling work, with a
     parameterisable API so v0.9 search can plug in.
   Each concrete recommendation that warrants implementation should be
   filed as a separate issue and linked from the GitHub comment.

The write-up should be roughly the same length and depth as the TVM
proposal (gh-ocannl-242).

## Scope

**In scope:**
- The seven-area write-up at `docs/research/tiramisu.md` (or equivalent
  location chosen by the agent — update the proposal frontmatter / Notes if
  the path differs).
- A GitHub issue comment on #267 summarising the findings.
- Filing follow-up issues for any concrete transferable ideas, linking
  them from the comment.

**Out of scope:**
- Implementing any Tiramisu-inspired feature (schedule IR, tile primitives,
  ISL integration, etc.). Each of those becomes its own task once filed.
- Building Tiramisu, running benchmarks, or producing OCaml bindings.
- Reading Tiramisu's full C++ source — the published paper plus the public
  documentation should suffice; consult the source only when the paper
  underspecifies a detail relevant to the OCANNL comparison.

**Dependencies:**
- Strongly related scouting tasks: gh-ocannl-242 (TVM, proposal already
  drafted), gh-ocannl-261 (superoptimizers), gh-ocannl-301 (IREE),
  gh-ocannl-306 (Petalisp/Caten). The Tiramisu write-up should
  cross-reference rather than duplicate gh-ocannl-242.
- Implementation milestones where findings would land: v0.8 (tiling,
  matmul, Metal — gh-ocannl-350, gh-ocannl-412) and v0.9 (program search).

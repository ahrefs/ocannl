# Lean Attention (Flash Attention + softmax-as-reduce): Feasibility Study

**Task**: gh-ocannl-263
**Issue**: https://github.com/ahrefs/ocannl/issues/263
**Milestone**: v0.9 (program search / performance explorations)
**Status**: Research / feasibility — write-up deliverable

## Goal

Assess whether and how OCANNL should support [Lean Attention](https://arxiv.org/abs/2405.10480)
(Flash Attention extended with softmax-as-reduce and stream-K tile distribution
for the decoder-only decode phase). The deliverable is a written feasibility
assessment that determines the prerequisites OCANNL needs before any concrete
implementation can land, picks an implementation strategy from the available
options (external fused kernel vs. IR-level expression vs. program-search
discovery), and produces actionable follow-up tasks. This is an `explore` task:
no production code changes are required by this proposal.

## Acceptance Criteria

The deliverable is a write-up (extending or replacing this proposal) plus a
summarizing comment on issue #263. The write-up must cover:

- [x] **Algorithm summary**: a precise description of online softmax, the
      Flash Attention forward pass, and the additions Lean Attention makes
      (stream-K-style work distribution across attention tiles, decode-phase
      specialization where Q has one row but K/V span the full context).
      *See `## Algorithm summary` below.*
- [x] **Feasibility assessment for OCANNL**: identify which of OCANNL's existing
      mechanisms can already express each piece (e.g. axis-aligned reductions,
      fused composition of primitives, einsum-style projections) and which
      cannot. *See `## Current OCANNL capabilities` and `## Feasibility
      assessment` below.*
- [x] **Prerequisites enumeration**: a concrete, ordered list of OCANNL
      capabilities that must exist before Lean Attention can be implemented in
      any meaningful form. At minimum the assessment must cover: tiling
      (v0.8 / #412), kernel fusion / megakernels (#318), shared memory and
      synchronization in generated CUDA, and multi-value reductions (online
      softmax keeps `(m, l, o)` — running max, running sum-of-exp, running
      output — and merges them with a non-trivial associative combiner).
      *See `## Prerequisites (ordered by dependency)` below.*
- [x] **Strategy recommendation**: an explicit recommendation among
      Option A (external handcrafted / cuDNN-fused kernel bound as a custom op),
      Option B (extend the low-level IR with the missing primitives so OCANNL
      generates the kernel), and Option C (define attention naively and rely on
      v0.9 program search to discover an efficient schedule). The recommendation
      must justify the choice in light of OCANNL's stated mission (a
      compiler / DSL, not a hand-tuned kernel library) and the timeline
      (v0.9 = Aug 2026, ICFP week). *Recommendation: Option B; see
      `## Strategy recommendation` below.*
- [x] **Follow-up tasks**: a list of harness/GitHub issues that should be
      filed to track the prerequisites and the eventual implementation, with
      explicit dependencies (e.g. "blocked by #412 tiling", "depends on
      decision in this write-up"). *See `## Follow-up tasks` below.*
- [x] **Decode-phase scope**: the write-up notes whether the recommendation
      addresses decode-only (Lean Attention's target regime, batch=1,
      `seq_q=1`, large KV cache) or also prefill / training, and whether
      prefill or training Flash Attention is in scope as a stepping stone.
      *Recommendation: decode-only; prefill/training out of scope. See
      `## Scope decision` below.*
- [x] **Out-of-scope clarity**: the write-up explicitly states what is
      *not* answered (e.g. backwards pass, multi-query / grouped-query
      attention specifics, paged KV cache integration, cross-backend story
      for Metal). *See `## Out of scope` below.*
- [x] A comment summarizing the findings is posted on
      [issue #263](https://github.com/ahrefs/ocannl/issues/263). *Posted at
      [issue #263 comment 4351844753](https://github.com/ahrefs/ocannl/issues/263#issuecomment-4351844753);
      the staged text is preserved under `## Proposed issue comment` below
      for reference.*

## Context

### How attention currently works in OCANNL

Multi-head attention is composed from primitives in
`lib/nn_blocks.ml`:

- `softmax ~spec` (around the `let%op softmax` definition) is built from
  `max-subtraction (@^^)`, `exp`, and `sum-reduce (++)` — i.e. the
  numerically-stable but unfused three-pass softmax.
- `multi_head_attention` (same file) projects Q/K/V, computes scores via
  einsum, applies `softmax` along the key axis, then einsums with V. There is
  no fused attention kernel and no online softmax.
- `cross_attention` and the masked variants reuse the same `softmax` helper.

Reductions and einsum are expressed via `Tensor.operation` and lowered to
`arrayjit/lib/low_level.ml`'s `For_loop` / `Set` / `Get` IR. Code generation
flows through `arrayjit/lib/c_syntax.ml` and the backends
(`arrayjit/lib/cuda_backend.ml`, `arrayjit/lib/metal_backend.ml`,
`arrayjit/lib/cc_backend.ml`).

### What's missing for Lean Attention

A single kernel that streams a Q tile, scans K/V tiles, and updates running
`(m, l, o)` triples in registers / shared memory requires:

1. **Loop tiling as an IR pass** — splitting an iteration into outer (tile)
   and inner (element) loops with controllable tile sizes. Tracked by #412.
   Today every CUDA kernel runs with `grid_dim=1, block_dim=1` (the
   `kernel_prep_line` guard in `cuda_backend.ml` returns early on every
   thread except `(0,0)`). Lean Attention is meaningless without removing
   this guard and mapping outer tile loops to `blockIdx` / inner loops to
   `threadIdx`.
2. **Kernel fusion / megakernels** — combining the three softmax passes plus
   the `S = QK^T` and `O = SV` einsums into one kernel. Tracked by #318.
3. **Shared memory and barriers** — the IR/codegen need a way to declare
   `__shared__` buffers and emit `__syncthreads()` (CUDA) /
   `threadgroup` + `threadgroup_barrier` (Metal). Today neither is emitted.
4. **Multi-value reductions** — online softmax's reduction operator merges
   triples `(m_a, l_a, o_a) ⊕ (m_b, l_b, o_b)` with a numerically-stable
   correction. OCANNL's reductions today are scalar-valued
   (`Accum_op` with a single accumulator over a single LHS). Either the IR
   gains tuple-valued reductions, or the writer manually splits the state
   across three tensors with hand-written cross-tile bookkeeping.
5. **Stream-K work distribution** — Lean Attention's distinguishing feature
   (vs. plain Flash Attention) is partitioning attention tiles across
   streaming multiprocessors so all SMs stay busy when Q has one row.
   Expressing this requires either explicit grid-index control surfaced
   to the frontend, or a scheduler smart enough to discover it. Neither
   exists today.
6. **A way to bind external code (Option A)** — if OCANNL chooses to call
   cuDNN's fused attention or a hand-written CUDA kernel, the compiler
   needs an FFI for "this assignment's RHS is implemented by this opaque
   kernel". That mechanism does not currently exist as a first-class
   construct; backends emit C/CUDA from `low_level.ml` and link the
   resulting `.so`/`.metallib` — there is no path for a precompiled blob.

### Related work in the repository

- **#412 — Tiling and multi-threaded GPU kernels**: the v0.8 milestone. A
  proposal already exists (`archive/docs/proposals/gh-ocannl-412.md`). Lean
  Attention is downstream of this.
- **#318 — Megakernels / kernel fusion**: prerequisite for any fused
  attention kernel.
- **#242 — TVM deep dive** (`archive/docs/proposals/gh-ocannl-242.md`): TVM's
  schedule primitives and tensorization story are directly relevant; in
  particular TVM's tensorize and split / reorder / cache_read primitives
  are what an Option B IR would need to subsume.
- **#261 — Superoptimizers / program search for tensor programs**: the
  v0.9 vehicle. Option C lives here: define attention naively and let
  search discover Flash/Lean.
- **OCANNL mission alignment**: the README and ROADMAP frame OCANNL as a
  compiler / DSL inspired by tinygrad, Luminal, TVM — not as a curated
  library of hand-tuned kernels. Option A (external fused kernel) is
  therefore in tension with the project's stated direction; Option B and
  Option C are more aligned but more expensive to deliver.

### Decode-phase specifics

Lean Attention's gains come from the decode regime: `seq_q = 1`, `seq_kv`
in the tens to hundreds of thousands, batch typically small. This is the
inference / autoregressive-generation case, not training. For OCANNL,
this matters because:

- Training-time Flash Attention has different tile shapes and a backward
  pass; conflating the two scopes makes the write-up unfocused.
- OCANNL's transformer examples (#65, #66, #74) are training/inference
  toys; a real LLM-inference integration (paged KV cache, GQA, etc.) is
  not on the roadmap until after v1.0.
- Whether this task should produce a decode-only Lean Attention
  implementation, a training-time Flash Attention implementation, or
  both, is one of the strategic questions the write-up must answer.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

The deliverable is a research write-up, not a code change. A reasonable
sequence:

1. **Read and summarize the paper** — Lean Attention (arxiv 2405.10480),
   plus Flash Attention 1 and 2 (arxiv 2205.14135, 2307.08691) for
   background, plus the online softmax derivation.
2. **Map each algorithmic primitive to OCANNL** — which can be expressed
   today (e.g. einsum projections, axis-aligned reductions); which need
   IR extensions (multi-value reductions, shared memory); which require
   scheduler / search support (stream-K tile distribution).
3. **Cross-reference the existing proposals** for #412, #318, #261, #242
   so that the prerequisite list reuses already-tracked work rather
   than inventing parallel issues.
4. **Compare strategies (A/B/C)** along axes: time-to-first-result,
   alignment with OCANNL's compiler-not-library mission, maintenance
   cost, generality (does the chosen path also help non-attention
   workloads?), and dependence on third-party kernels.
5. **Write the assessment** by extending this file in place. The final
   document replaces the "this is a study" framing with a concrete
   recommendation and a list of follow-up tasks.
6. **Post a summary comment** on issue #263 linking back to the
   committed write-up.
7. **Optionally file follow-up GitHub issues** for any prerequisites that
   are not already tracked, with `blocked_by` set to the relevant
   existing issues.

The agent should not attempt to implement Flash or Lean Attention as
part of this task; doing so prematurely (before tiling, megakernels,
and shared-memory codegen exist) would necessarily fall back to
Option A (an external kernel), which preempts the strategic decision
this study is meant to inform.

## Scope

**In scope**:
- A written feasibility assessment in this file.
- Strategy recommendation (A vs. B vs. C).
- Enumeration of prerequisites, cross-referenced against existing
  ROADMAP items and GitHub issues.
- A summarizing comment on issue #263.
- Optional: filing follow-up GitHub issues for any new prerequisites
  not already tracked.

**Out of scope**:
- Any implementation of Flash Attention, Lean Attention, or online
  softmax. These wait until the prerequisites land (v0.8 tiling /
  megakernels at the earliest).
- Backwards pass / training-time Flash Attention details beyond noting
  whether they are in or out of scope for the eventual implementation.
- Multi-query / grouped-query attention, paged KV cache, speculative
  decoding integration. These are layered concerns that can be added
  after a baseline Flash/Lean Attention exists.
- Metal-specific or CPU-backend implementation details (the assessment
  may note these as future work but is not required to design them).
- Changes to `lib/nn_blocks.ml`, `arrayjit/lib/low_level.ml`,
  `arrayjit/lib/cuda_backend.ml`, or any other source file. This task
  produces documentation only.

**Dependencies**:
- Logical dependency on #412 (tiling) and #318 (megakernels) for any
  eventual implementation; the present *study* does not block on them.
- Related to #242 (TVM deep dive) and #261 (superoptimizers) — the
  study's strategy comparison should reference whatever conclusions
  those tasks reach if they have landed first.

---

# Feasibility assessment

The sections below are the deliverable. They are written against the
state of the repository at the time of this commit (branch
`ludics/gh-ocannl-263-s6/root`, top of file
`docs/research/lean-attention-feasibility.md` plus the live source under `lib/`,
`arrayjit/lib/`, and `ROADMAP.md`).

## Executive summary

1. **Do not implement Lean Attention now.** The compiler infrastructure
   it requires is not yet in tree; trying to land an implementation
   today would force Option A (an external/handcrafted CUDA kernel
   bound as an opaque op), which is misaligned with OCANNL's stated
   compiler/DSL mission.
2. **Recommended long-term target: Option B** — extend OCANNL's
   low-level IR (tiling, shared memory + barriers, multi-value
   reductions, schedulable grid-index control) so the compiler
   *generates* a fused, tiled, decode-only Lean-style attention kernel
   from a near-naive frontend description. The same infrastructure
   pays for itself in layernorm, RMSNorm, fused softmax,
   convolutions, and any other fused-reduction-heavy kernel — Lean
   Attention is the forcing function, not the only beneficiary.
3. **Option A** (cuDNN fused attention or a handcrafted CUDA kernel,
   bound as a custom op) is acceptable only as a deliberate, time-boxed
   strategic compromise — for example, if a near-term inference
   benchmark or workshop demo requires real Flash/Lean numbers before
   the v0.8 / v0.9 infrastructure has landed. It is not the default.
4. **Option C** (let v0.9 program search discover Flash/Lean Attention
   from a naive specification) is **not** a cheaper alternative to
   Option B. The schedule-transformation space program search would
   need to explore — tiling, shared-memory placement, fused multi-value
   reductions, stream-K work distribution — does not yet exist in
   `Low_level`. Option C therefore *layers on top of* Option B's
   infrastructure rather than replacing it.
5. **Scope when implementation does start: decode-only.**
   Lean Attention's contribution over Flash Attention 2 is largest in
   the decode regime (`seq_q = 1`, very long KV). Prefill / training
   Flash Attention may become a useful intermediate target later but
   is out of scope for this issue. Backward pass, multi-query /
   grouped-query attention, paged KV cache, and Metal/CPU parity are
   all out of scope as well.

## Algorithm summary

This section is intentionally self-contained for a reader who has not
read the paper or the issue thread.

### Online softmax

Standard numerically-stable softmax over a vector `x_1, …, x_n`
computes `m = max_i x_i`, then `l = sum_i exp(x_i − m)`, then
`y_i = exp(x_i − m) / l`. This is three passes over the input.

The *online softmax* (Milakov & Gimelshein, 2018) keeps a running pair
`(m, l)` and updates it one element at a time:

```
m_new = max(m_old, x_new)
l_new = l_old · exp(m_old − m_new)  +  exp(x_new − m_new)
```

This reduces the three passes to one and, crucially, the update is
*associative* in the right way: two partial states `(m_a, l_a)` and
`(m_b, l_b)` can be merged into one state `(m, l)` via the same
formula, so a parallel reduction over chunks produces the same answer
as a sequential scan. That associativity is what makes "softmax as a
reduction" workable.

### Flash Attention forward pass

For one query row `q ∈ R^d` against `K, V ∈ R^{n × d}`:

```
S = q K^T               (n scalar logits)
P = softmax(S)          (n probabilities)
o = P V                 (one output vector in R^d)
```

Flash Attention computes this in tiles of `K` and `V` of size `B_c`,
maintaining a loop-carried triple `(m, l, o)` — running max, running
sum-of-exp, running output:

```
for each KV tile (K_j, V_j):
    s_j      = q · K_j^T                           // tile of logits
    m_new    = max(m, max(s_j))
    α        = exp(m − m_new)                       // rescale factor
    p_j      = exp(s_j − m_new)                     // tile probs (stable)
    l_new    = α · l + sum(p_j)
    o_new    = α · o + p_j · V_j                    // accumulate output
    (m, l, o) = (m_new, l_new, o_new)
return o / l                                       // one final divide
```

Two properties matter for OCANNL:

1. **`(m, l, o)` is a single reduction state.** The merge of two
   partial states is associative — the same correction factor `α`
   that combines a tile with the running state combines two tiles
   with each other. Online softmax's associativity extends to the
   triple.
2. **The whole forward pass fits in one kernel.** Score, normalisation,
   and value-mix share `q` and the running state in registers / shared
   memory; spilling intermediates to global memory (as the unfused
   `softmax(QK^T) V` formulation does) is the cost Flash Attention
   eliminates.

Flash Attention 2 (Dao, 2023) keeps the same forward kernel skeleton
but improves thread-block-level work distribution and adds a
recomputation-based backward pass; the structural points above carry
over.

### What Lean Attention adds over Flash Attention 2

Lean Attention (arxiv 2405.10480) is specifically the **decode-phase**
case where Q has only one row (`seq_q = 1`) but `seq_kv` is in the
tens to hundreds of thousands. Three contributions:

1. **Softmax-as-reduce framing.** The whole attention kernel is one
   tiled associative reduction over the `seq_kv` axis with state
   `(m, l, o)`. This is essentially the Flash forward pass restated as
   "fold an associative combiner over KV tiles," which makes the work
   distribution across threads/blocks the only remaining design knob.
2. **Stream-K-style work distribution.** Plain Flash Attention assigns
   one Q tile per thread block. With `seq_q = 1` there is one Q tile,
   so plain FA leaves most of the GPU idle. Stream-K partitions the
   KV-axis reduction itself across SMs — each SM owns a slice of KV
   tiles, computes a partial `(m, l, o)`, and a final pass merges
   the partials with the same associative combiner. Idle SMs are the
   problem; stream-K is the fix.
3. **Decode-phase specialisation.** Q is a single row, so the
   per-thread-block working set is dominated by KV. Tile shapes,
   register/shared-memory budgets, and merge-tree depth are tuned for
   that asymmetry. Reported gains: 2.6× average, up to 8.33× at
   512k context.

References:
- *Lean Attention*, arxiv 2405.10480.
- *FlashAttention*, Dao et al., arxiv 2205.14135.
- *FlashAttention-2*, Dao, arxiv 2307.08691.
- *Online Normalizer Calculation for Softmax*, Milakov & Gimelshein,
  arxiv 1805.02867.

## Current OCANNL capabilities

Citations are to live source on this branch.

1. **Softmax is multi-pass and unfused.**
   `lib/nn_blocks.ml` lines 107–113 define `softmax` as
   `max-subtraction → exp → divide-by-sum-reduce`. The implementation
   is numerically stable but allocates intermediates and reduces
   independently from the surrounding einsums.
2. **Multi-head attention is einsum-composed.**
   `lib/nn_blocks.ml` lines 181–209 build `multi_head_attention` from
   separate Q/K/V linear projections, an einsum to form scores
   (`q +* k " ... s | h d; ... t | h d => ... s | t -> h"`), the same
   `softmax` along the key axis, an einsum against `V`, and a final
   `w_o` projection. There is no fused attention kernel and no online
   softmax.
3. **Reductions are scalar-accumulator.**
   `arrayjit/lib/assignments.ml` lines 67–74 define
   `Accum_op { initialize_neutral; accum : Ops.binop; lhs : Tn.t;
   rhs : accum_rhs; projections; … }`. There is exactly one binary
   accumulator and one scalar LHS slot; tuple/state reductions are not
   first-class.
4. **Reduction lowering is ordinary nested loops.**
   `arrayjit/lib/assignments.ml` lines 282–425 lower reductions to
   nested `For_loop`s over `projections.product_space` with a scalar
   neutral initialiser (`Ops.neutral_elem accum`) and a scalar update
   in the loop body.
5. **Low-level IR has no parallel/tile/barrier/shared-memory
   constructs.** `arrayjit/lib/low_level.ml` `Low_level.t` carries
   `For_loop`, `Set`, `Get`, `Set_local`, `Declare_local`,
   `Comment`, etc. There is no parallel-loop annotation, no tuple
   accumulator, no barrier node, and no shared/threadgroup memory
   construct.
6. **CUDA kernels are launched single-threaded.**
   `arrayjit/lib/cuda_backend.ml` line 317 emits the kernel prelude
   `"/* FIXME: single-threaded for now. */if (threadIdx.x != 0 ||
   blockIdx.x != 0) { return; }"`; line 968 launches kernels with
   `~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0`. Every CUDA
   kernel in the current backend runs on a single thread of a single
   block.
7. **No shared-memory or barrier emission anywhere in codegen.**
   A repo-wide search for `__syncthreads`, `threadgroup_barrier`,
   `__shared__`, and `threadgroup ` across `arrayjit/lib/cuda_backend.ml`,
   `arrayjit/lib/c_syntax.ml`, `arrayjit/lib/low_level.ml`, and
   `arrayjit/lib/metal_backend.ml` returns no matches.
8. **No KV-cache / paged-attention / GQA / MQA / Flash / online-softmax
   primitives in tree.** Repo-wide search confirms decode-phase and
   KV-cache-specific infrastructure is not present today.
9. **Roadmap placement is consistent with "infrastructure first."**
   `ROADMAP.md` lists "Lean Attention / Flash Attention (#263)" under
   *Post-1.0 Considerations / Performance explorations* (line 190),
   downstream of v0.8 (tiling, megakernels, Metal optimisations) and
   v0.9 (program search and code-graph rewriting).

## Feasibility assessment

Each algorithmic piece classified as *expressible today*, *partially
expressible but inefficient/unfused*, or *blocked on missing
infrastructure*:

1. **Q/K/V projections, score einsum, output einsum** — *expressible
   today* via the einsum spec already used by `multi_head_attention`.
2. **Numerically-stable softmax along an axis** — *partially
   expressible*: stable, but multi-pass and unfused; cannot share
   loop-carried state with the surrounding einsums.
3. **Tile-loop over KV, mapped to thread blocks** — *blocked* on
   tiling / multi-threaded GPU kernels. Today's IR has no
   parallel-loop or tile annotation; today's CUDA backend hardcodes
   `grid_dim_x:1 block_dim_x:1` and short-circuits all but
   `(threadIdx.x = 0, blockIdx.x = 0)`.
4. **Online-softmax `(m, l, o)` carry across tiles** — *blocked* on a
   first-class multi-value reduction or, as a fallback, a manual
   three-tensor encoding with hand-written merge logic. `Accum_op`
   today carries one scalar accumulator.
5. **Shared memory for tile staging, registers for `(m, l, o)`** —
   *blocked*: codegen emits no `__shared__` / `threadgroup` buffers and
   no synchronisation intrinsics. Without these, the inner loop must
   round-trip through global memory and the fused-kernel performance
   argument collapses.
6. **Stream-K work distribution across SMs** — *blocked*: requires
   either explicit grid-index control surfaced to the frontend or a
   scheduler/search layer (#261) capable of discovering it. Neither
   exists today.
7. **External-kernel binding (only required for Option A)** —
   *blocked*: there is no first-class FFI in `Assignments` /
   `Low_level` for "this assignment's RHS is implemented by an opaque
   precompiled kernel." Backends today emit C/CUDA from `low_level.ml`
   and link the resulting `.so`/`.metallib`; adding a
   precompiled-blob path is non-trivial.

## Prerequisites (ordered by dependency)

The list below is what must land before a meaningful Lean Attention
implementation along the recommended Option B path is possible. Each
item is keyed to existing tracked work where it exists.

1. **Multi-threaded / tiled GPU kernels** — remove the single-thread
   guard at `arrayjit/lib/cuda_backend.ml:317-318`, surface tile-size
   knobs in the IR, and lower outer loops to `blockIdx` and inner
   loops to `threadIdx`. Tracked by **#412**
   (`archive/docs/proposals/gh-ocannl-412.md`); aligns with the v0.8 theme
   "GPU-style performance" in `ROADMAP.md` lines 118–134.
2. **Shared memory + barriers in codegen** — IR/codegen support for
   declaring `__shared__` (CUDA) / `threadgroup` (Metal) buffers and
   emitting `__syncthreads()` / `threadgroup_barrier`. **Not currently
   tracked locally**; this write-up's recommendation is to either file
   a follow-up issue or annex the work to #412. Adjacent to the
   "warp-shuffle reductions" line item recorded in
   `archive/docs/proposals/gh-ocannl-253.md:158-164`.
3. **Kernel fusion / megakernels** — the ability to fuse score,
   softmax, and value-mix into one kernel. Tracked upstream by
   **#318**; *the local proposal file `docs/proposals/gh-ocannl-318.md`
   is not present in this checkout*, so this write-up cites the issue
   number rather than a local document.
4. **Multi-value (tuple) reductions** — represent `(m, l, o)` as a
   single associative reducer rather than three loosely-coupled
   scalar reductions. The natural shape is to generalise `Accum_op` so
   that `accum` can be a small algebraic operator parameterised by
   `(init, lift, combine)` over a record type, with the combine
   guaranteed associative by construction. **Not tracked locally.**
   Manual three-tensor encoding is a fallback if the IR extension is
   too costly, but it pushes the cross-tile bookkeeping into user
   code, which defeats the "compiler/DSL" framing.
5. **Stream-K (or equivalent) work distribution** — explicit
   grid-index control surfaced to the frontend, or a scheduler/search
   layer. Logically downstream of **#261** (superoptimizers / program
   search; *local proposal file not in this checkout*) and #412.
   Aligns with v0.9's "Program search and optimization" theme
   (`ROADMAP.md` lines 139–155).
6. **External-kernel binding** — only required for **Option A**.
   A non-trivial extension to `Assignments`/`Low_level` and the loader
   path in `cuda_backend.ml` for declaring "this assignment's RHS is
   an opaque precompiled kernel." **Explicitly not** a prerequisite
   for the recommended Option B path.

## Strategy recommendation

Three options, evaluated against five axes: time-to-first-result,
mission alignment, maintenance cost, generality (does the same work
help non-attention workloads?), and dependence on third-party kernels.

### Option A — External fused kernel (cuDNN or handcrafted CUDA)

Bind cuDNN's `cudnnFlashAttention` family or a handcrafted CUDA
kernel into OCANNL as a custom op via a new external-kernel FFI
(prerequisite #6).

1. **Time-to-first-result**: shortest. Once the FFI lands,
   integrating cuDNN or a known-good Flash kernel is days, not weeks.
2. **Mission alignment**: weakest. OCANNL is positioned as a
   compiler/DSL inspired by tinygrad, Luminal, and TVM (see
   `README.md` and `ROADMAP.md`'s v0.8/v0.9 themes), not as a curated
   library of hand-tuned kernels. A cuDNN binding makes attention
   "fast" without making OCANNL more capable.
3. **Maintenance**: moderate. cuDNN bindings break across CUDA
   versions; handcrafted kernels are additionally tied to specific
   GPU architectures.
4. **Generality**: low. The FFI itself is general, but the
   fused-attention kernel does nothing for layernorm, RMSNorm, fused
   softmax, or any other workload.
5. **Third-party dependence**: high (cuDNN) or specialist-time
   intensive (handcrafted kernel).

**Verdict**: acceptable only as a deliberate, time-boxed strategic
compromise — for example, if a workshop demo or near-term inference
benchmark needs real Flash/Lean numbers before the Option B
infrastructure lands. Not the default.

### Option B — Extend the IR so OCANNL generates the kernel (recommended)

Land prerequisites #1–#5: tiling, shared memory + barriers, kernel
fusion, multi-value reductions, and frontend-surfaced grid-index
control (or stream-K-aware scheduling). Express attention naively (or
with light annotation) at the frontend; let the compiler emit a
fused, tiled, single-kernel implementation.

1. **Time-to-first-result**: longest. Each prerequisite is itself a
   substantial milestone (cf. v0.8 themes in `ROADMAP.md`).
2. **Mission alignment**: strongest. Every prerequisite is a
   compiler-level capability that the project already wants to build
   for other reasons.
3. **Maintenance**: lowest *per workload*. The cost is amortised
   across every fused-reduction-heavy kernel, not just attention.
4. **Generality**: highest. Tiling, shared memory, fused reductions,
   and multi-value accumulators directly help layernorm, RMSNorm,
   convolution, and any reduction-heavy primitive — not just
   attention. Stream-K-style work distribution is reusable for any
   reduction with a "few rows × many columns" asymmetry.
5. **Third-party dependence**: none.

**Verdict**: this is the recommended path. Lean Attention becomes the
*forcing function* for capabilities OCANNL already needs.

### Option C — Program search discovers Lean Attention

Define attention naively at the frontend and let v0.9 program search
find an efficient schedule.

1. **Time-to-first-result**: longest of the three. Search is a
   research problem (cf. v0.9's "Program search and optimization"
   theme). Search alone is not enough — the *space* program search
   would explore (tilings, shared-memory placements, fused-reduction
   patterns, stream-K work distributions) does not exist in
   `Low_level` today.
2. **Mission alignment**: strong (matches v0.9's stated direction),
   but *contingent* on Option B's infrastructure.
3. **Maintenance**: deferred. Cost shifts from per-workload kernel
   tuning to scheduler/search-system maintenance.
4. **Generality**: highest in principle — the same search applies to
   any tensor program — but only after the underlying space is
   modelled.
5. **Third-party dependence**: none.

**Verdict**: Option C is *not* a substitute for Option B; it is a
*successor*. It becomes credible only after the schedule space exists,
i.e., after Option B lands. Treat C as the v0.9+ direction once the
v0.8 / shared-memory / fused-reduction infrastructure has been built.

### Recommendation

1. **Mainline: Option B**, gated on prerequisites #1–#5 above.
2. **Fallback / escape hatch: Option A**, only if the project
   explicitly chooses near-term inference performance over compiler
   purity. Document the choice as a strategic compromise in the
   issue/PR that lands it.
3. **Future: Option C** once the schedule space modelled by Option B
   exists.
4. Do **not** start an implementation in this round. The deliverable
   for issue #263 is this write-up plus the summary comment; the
   actual implementation is a separate task that should be filed once
   the prerequisites are in flight.

## Scope decision

1. **Decode-only Lean Attention** (`seq_q = 1`, large `seq_kv`) is the
   recommended scope when implementation does start. This is the
   regime where Lean Attention's contribution over Flash Attention 2
   is largest, and it is the regime named in the paper.
2. **Prefill / training-time Flash Attention** is *not required* to
   answer this issue. It may become a useful intermediate target —
   the same fused-kernel infrastructure underpins both — but it is
   out of scope here. A separate task should track it if and when it
   becomes relevant.
3. **Backward pass** is out of scope for the first implementation
   regardless of which option is taken.
4. **Multi-query / grouped-query attention, paged KV cache,
   speculative decoding integration** are layered concerns that can
   be added once a baseline decode-only implementation exists.
5. **Metal / CPU backend parity** is a downstream concern that will
   inherit whatever shared-memory/barrier IR is added. It is not
   designed for in this write-up.

## Out of scope

Explicit list of items this write-up does *not* answer:

1. Backward pass for Flash or Lean Attention.
2. Multi-query / grouped-query attention specifics.
3. Paged KV cache integration.
4. Speculative-decoding integration.
5. Metal- or CPU-backend Lean Attention design.
6. Any source-code change in this round. This task ships
   documentation only.

## Follow-up tasks

Tracked work to lean on (in dependency order):

1. **#412** — Tiling and multi-threaded GPU kernels. Local proposal
   present at `archive/docs/proposals/gh-ocannl-412.md`. Hard prerequisite for
   Option B.
2. **#318** — Megakernels / kernel fusion. *Local proposal file not in
   this checkout.* Hard prerequisite for Option B; cited by issue
   number.
3. **#261** — Superoptimizers / program search. *Local proposal file
   not in this checkout.* Required for Option C; useful but not
   required for Option B.
4. **#242** — TVM deep dive. Local proposal present at
   `archive/docs/proposals/gh-ocannl-242.md`. Reference for the
   schedule/transformation layer Option B implies.
5. **#253** — Lessons from llm.c. Local proposal present at
   `archive/docs/proposals/gh-ocannl-253.md`; lines 158–164 specifically call
   out warp-shuffle reductions and fused reduction codegen as future
   work that benefits softmax/layernorm-like kernels.

New issue candidates this write-up recommends filing (or annexing to
existing issues), each marked *subject to per-issue confirmation* —
filing is optional per the proposal's Scope and is left to the
reviewer/operator decision:

6. **"Shared memory + barriers in IR/codegen"** —
   `blocked_by: #412`. Either a new issue or annexed to #412.
7. **"Multi-value / tuple reductions in `Accum_op`"** —
   `blocked_by: <none for the IR work itself, but only useful once
   tiling and fusion land>`.
8. **"Lean Attention (decode-only) implementation"** —
   `blocked_by: #412, #318, the shared-memory issue, the
   tuple-reduction issue`. Files when the prerequisites are in flight
   or landed.

## Proposed issue comment

**Posted** at
https://github.com/ahrefs/ocannl/issues/263#issuecomment-4351844753 on
2026-04-30. The text below is preserved verbatim for reference; any
future edits should be made on GitHub directly.

> Feasibility study landed in `docs/research/lean-attention-feasibility.md`.
>
> **Recommendation: do not implement Lean Attention now.** OCANNL
> today launches every CUDA kernel single-threaded
> (`arrayjit/lib/cuda_backend.ml:317`, `:968`), has no shared-memory
> or barrier emission, no tuple/online-softmax-style reductions
> (`arrayjit/lib/assignments.ml:67`), and no schedule-transformation
> layer. The pieces Lean Attention needs are exactly the pieces
> already on the v0.8 / v0.9 roadmap.
>
> **Mainline recommendation: Option B** — extend the low-level IR
> (tiling #412, shared memory + barriers, kernel fusion #318,
> multi-value reductions, stream-K-style grid control / search #261)
> so OCANNL *generates* a fused, tiled, decode-only Lean-style
> attention kernel from a near-naive frontend. The same infrastructure
> pays for itself in layernorm, RMSNorm, fused softmax, conv, and any
> reduction-heavy fused kernel.
>
> **Option A** (cuDNN or handcrafted CUDA bound as a custom op) is
> acceptable only as a deliberate, time-boxed strategic compromise.
> **Option C** (program search discovers Lean Attention) is a
> successor to Option B, not a substitute — the schedule space search
> would explore does not yet exist.
>
> **Scope when implementation starts: decode-only.** Backward pass,
> MQA/GQA, paged KV cache, and Metal/CPU parity are out of scope for
> the first attempt. Full prerequisite list and per-axis comparison
> in the write-up.

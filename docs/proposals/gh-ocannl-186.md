# Dynamic Indexing — Reintroduction Disposition Memo

## Resolution (2026-06-12): retired — subsumed by #343

Per Łukasz: **#186 is retired as subsumed by
[gh-ocannl-343](gh-ocannl-343.md)** (virtual one-hot / embedding
optimization). The disposition this memo was meant to produce is hereby
recorded:

- **Dynamic indexing is not re-introduced** as a first-class
  `axis_index`/shape-inference feature. The complexity/benefit ratio is too
  high: it touches shape inference, projections, every backend's codegen,
  and the gradient story, while the dominant in-tree need (embedding
  lookup / gather) is served by #343's compiler rewrite, whose
  `Get_dynamic`-style construct lives only in the low-level IR.
- **#343 owns the dynamic-index lowering construct** (the `indexing.ml` /
  `low_level.ml` design space sketched in its Phase 3) — no rival index
  representation should be designed under this issue.
- **Re-opening triggers**: a concrete demand for *scatter* (dynamic write
  indices), top-k routing, or KV-cache-style in-place updates — the cases
  #343's gather-only rewrite structurally cannot express. #377 (GPT-2
  inference) confirmed it does not need them (single-token one-hot
  suffices). If such a need lands, reopen with this memo as the starting
  point rather than designing from scratch.
- Practical effect: the GH issue can be closed as "not planned (subsumed by
  #343)" with a comment along the lines of the above; the v0.7.1 milestone
  move is moot. This document stays as the historical record.

The memo content below is retained for reference.

## Status update (2026-06-12)

- Issue #186 is OPEN, label still `explore`, but its milestone was moved
  **v1.1 → v0.7.1 on 2026-05-11** (GH milestone events). The "issue metadata
  says non-blocking for v0.7.1" tension described below has been resolved in
  the direction of promotion — effectively disposition (A)/(C)'s
  implementation half is now scheduled for v0.7.1, alongside #377.
  ROADMAP.md's v0.7.1 section does not list #186 explicitly yet.
- Issue #377 (transformer inference demo) is still OPEN at v0.7.1. The
  v0.7.1 milestone (originally mid-March 2026) is running late: the repo is
  still at version 0.6.3 and v0.6.4-themed work is what has been landing.
- **Dynamic indexing remains unimplemented**: `arrayjit/lib/indexing.ml`'s
  `axis_index` still has only `Fixed_idx | Iterator | Affine | Sub_axis |
  Concat`; the one-hot workarounds in `lib/nn_blocks.ml`
  (`one_hot_of_int_list`), `test/training/mlp_names.ml`
  (`fill_ctx_one_hot`), and `test/training/fsm_transformer.ml`
  (`seqs_to_flat_one_hot`) are all still in place, as is `Batch_slice` /
  `@|` (`tensor/operation.ml:630`). README's caveat is still at line 18.
- `CHANGES.md` line references drifted: the removal note is now at line 527
  and the prior-prototype behavior notes at lines ~596–602.
- Related-context drift: #271 (low-bit optimizers) was closed NOT_PLANNED;
  #341 (multicore_cc non-determinism) was closed COMPLETED. RoPE/position
  embeddings (#398), tensor stacking + block-literal `%op` syntax
  (`58bfd6e5`), and the broadcast-order reversal (LUB→GLB) and
  "label"→"basis" renames all landed — none change the DI design space,
  though any new shape-inference text for the design memo should use the
  post-reversal vocabulary ("refines", join semilattice).
- Remains to do: the design-memo content itself (API choice, per-backend
  codegen story, gradient story) and then the implementation now implied by
  the v0.7.1 milestone. The single-threaded CUDA constraint cited below is
  still accurate (`cuda_backend.ml:317` `kernel_prep_line`; launch forces
  `grid_dim_x:1, block_dim_x:1` at line 970).

## Goal

Address [gh-ocannl-186](https://github.com/ahrefs/ocannl/issues/186) —
"Consider re-introducing dynamic indexing" — by producing a written
disposition document that resolves the priority/milestone question and, if
the user picks the design-memo path, a self-contained design memo that
surfaces the API and codegen choices without committing to an
implementation.

This proposal is **artifact-only**. It does not implement dynamic indexing,
does not pick among the backend codegen approaches, and does not unilaterally
re-prioritise the issue. It surfaces three dispositions and recommends one;
the user's hands-on audit picks the final answer.

## Context

### What was removed

The "giant refactor" referenced in the issue body did happen. Confirmed by:

- `CHANGES.md` line 527 *(line updated 2026-06-12)* (release notes from the
  refactor): *"Dynamic indexing
  is not supported anymore (to reduce complexity). It might be reintroduced
  if needed."*
- `CHANGES.md` lines 596–602 *(lines updated 2026-06-12)* describe the prior prototype's behavior
  ("skips over parallel axes", "can produce virtual nodes") — that machinery
  is gone.
- `README.md` line 18 records the current contract publicly: *"OCANNL does
  not have dynamic indexing (using the last axis of one tensor as indices
  into another tensor). If it's needed, it can be added (we had a prototype
  once, removed to reduce complexity). Then it would also be integrated with
  shape inference."*
- Source-level: `arrayjit/lib/indexing.ml`'s `axis_index` type has only
  `Fixed_idx`, `Iterator`, `Affine`, `Sub_axis`, `Concat` — no
  `Dynamic_recipient` or equivalent. `arrayjit/lib/low_level.ml`'s `Get` and
  `Set` operations take `Indexing.axis_index array`, so the path from a
  runtime-tensor-valued index to a `Get` argument simply does not exist.
- A grep for `Dynamic_recipient`, `Dyn_*`, or `recipient` across `arrayjit/`
  and `tensor/` returns nothing — the excision was clean.

### What the codebase does instead today

Embedding lookup is implemented as **one-hot @ matmul**, on the host side:

- `lib/nn_blocks.ml` `one_hot_of_int_list` materializes a `[len; num_classes]`
  one-hot tensor from a host-side int list and `rebatch`es it as input data.
- `test/training/mlp_names.ml` (Karpathy "makemore Part 2" / Bengio MLP)
  uses `fill_ctx_one_hot` to fill a per-batch
  `batch_size * block_size * vocab_size` flat one-hot buffer at training
  time, and embeds via `({ w_embed; o = [ embed_dim ] } * input)` inside the
  model.
- `test/training/fsm_transformer.ml` (decoder-only transformer, single
  block, FSM language) does the same: inputs are
  `[batch_size; eff_seq_len; num_states]` one-hot tensors filled host-side
  via `seqs_to_flat_one_hot`; the model embeds with
  `({ w_embed; o = [ d_model ] } * input) + { pos_encoding }`.

Cost of the workaround: O(vocab_size) memory per token slot (host-resident
one-hot buffer, also resident on device once uploaded) and O(vocab_size *
embed_dim) flops for the embedding matmul instead of O(embed_dim) for a
true gather. For `num_states = 8` (FSM) this is fine. For
`vocab_size = 27` (Names) this is fine. For real GPT-2 inference
(`vocab_size ≈ 50257`) this is a 50257-wide one-hot per token — the
embedding matmul does 50257 × 768 ≈ 39M multiply-adds **per token** instead
of the 768 ops a true gather needs. Tractable for an educational demo;
nowhere near competitive.

There is also a static slicing primitive — `tensor/operation.ml`'s
`Batch_slice` / `@|` — but its index is a `Idx.static_symbol`, bound at
compile/launch time, not a runtime tensor value. It is not a substitute
for dynamic indexing.

For autoregressive sampling at inference time, the host-roundtrip
escape hatch is: run the model on a one-hot input, host-read the logits,
host-sample a next token (argmax / top-k / multinomial), one-hot it back,
re-launch. This is N device→host syncs for an N-token generation.
Acceptable for an educational demo, painful for any wall-clock-sensitive
use.

### Why this issue is plausibly on the critical path

`ROADMAP.md` v0.7.1 (target mid-March 2026, milestone #30) includes
**"Transformer inference demo (#377)"** — *"Inference for a small
open-weights model (GPT-2, LLaMA, or Gemma)"*. Issue #377 is open,
unlabeled, and confirms the scope as huggingface-tokenizers bindings plus
inference for one of the three models.

GPT-2's vocab is 50257; LLaMA's is 32000; Gemma's is 256000. Without
dynamic indexing, all three demos will:

1. Embed via vocab-wide one-hot @ embedding-matrix matmul (slow but
   functional).
2. Sample autoregressively via host-roundtrip per token (slow but
   functional).
3. Have no good answer for sliced kv-cache lookup if the demo grows beyond
   a "feed the whole prefix every step" naive loop, and no good answer for
   variable-length attention or causal-mask-with-real-eos handling that
   doesn't pad to fixed sequence length.

So the AC's claim that DI is needed for "embedding lookup, gather/scatter,
top-k sampling, variable-length attention" is correct *in production*. The
question this disposition resolves is whether the v0.7.1 demo needs
production-shaped DI, or whether the demo is allowed to be slow/naive and
DI is reintroduced post-v0.7.0 on its own merits.

### Issue metadata

- `explore` label, milestone v1.1 (post-v1.0), priority B, effort large
  (6–10 days). *(Update 2026-06-12: the GH milestone was moved to v0.7.1 on
  2026-05-11; the label remains `explore`. See the status update above.)*
- The `explore` label is documented as *"Priority below 'enhancement',
  non-blocking for milestones"* — i.e. the current metadata says DI is not
  blocking v0.7.1.
- This metadata predates the explicit listing of #377 in v0.7.1 in
  `ROADMAP.md`. There is real tension here that the user's audit needs to
  resolve.

### Related repository context

- Harness memory records that the user has paused autonomous OCANNL work
  pending a hands-on quality audit.
- v0.6.4 / v0.7.0 work (RoPE, transformer toy, persistence, context
  finalization) is in flight; v0.7.1 also includes tokenizer bindings (a
  separate prerequisite for #377 that is independent of DI).
- gh-ocannl-271 (Adam optimizer / quantization) and other "explore"-labeled
  issues form a parallel queue. *(Update 2026-06-12: #271 has since been
  closed as not planned.)*
- Prior closely-shaped proposals: `gh-ocannl-161` (fork-backend disposition
  memo) and `gh-ocannl-278` (DisTrO feasibility study) followed the same
  artifact-only-disposition pattern; this proposal mirrors that style.

## Acceptance Criteria

This is a disposition document. The acceptance criteria describe the
**proposal-output**, not the implementation of dynamic indexing.

- [ ] The document records the current state of indexing in OCANNL with
      concrete pointers (`indexing.ml` `axis_index` variants, absence of
      `Dynamic_recipient`, the one-hot @ matmul workaround in
      `nn_blocks.ml` / `mlp_names.ml` / `fsm_transformer.ml`,
      `Batch_slice`'s static-symbol contract) so a future reader does not
      have to re-do the archaeology.
- [ ] The document quantifies the cost of the current workaround at GPT-2
      scale (vocab × embed flops, N host-roundtrips for N-token
      generation) so the priority decision is made on numbers, not vibes.
- [ ] The document enumerates the three dispositions (A, B, C below) and
      recommends one, with the choice surfaced explicitly to the user as a
      question rather than picked silently.
- [ ] If the user picks disposition (A) — re-prioritise to v0.7.1: the
      document records what the **narrowest landing** would look like
      (embedding-lookup-only, deferring gather/scatter/atomic-scatter to a
      follow-up).
- [ ] If the user picks disposition (B) — defer DI to v1.1 as planned: the
      document records the explicit acceptance that GPT-2 inference will
      ship with vocab-wide one-hot embedding and host-roundtrip sampling,
      and that this is fine for an educational demo. README's existing
      caveat (line 18) stands.
- [ ] If the user picks disposition (C) — the document **already
      contains** the design-memo skeleton: API options for the user-facing
      surface, the codegen story for CC / CUDA / Metal at the level of
      "what changes in `low_level.ml`, what changes in `cc_backend.ml` /
      `cuda_backend.ml` / `metal_backend.ml`", the gradient story, and
      the open questions list.
- [ ] The document is committed to the OCANNL repo so it survives outside
      the harness; the GH issue is updated to link to it.

**Out of scope** (explicitly):

- Implementing any aspect of dynamic indexing — no `Dynamic_recipient`
  variant, no backend changes, no shape-inference changes, no operation
  added to `tensor/operation.ml`.
- Choosing between the API options (special axis-type vs `Gather`/`Scatter`
  ops) — that is a design question for the actual implementation phase.
- Choosing between the three backend codegen sketches in the design-memo
  skeleton — those need code-level verification against current backend
  internals.
- Re-prioritising the GitHub issue, changing its milestone, or changing
  the harness `priority`/`milestone` fields — those are controller actions
  that follow from the user's disposition decision.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

This proposal is the document. The "implementation" is writing it. The
recommended structure for the document content is below.

### Disposition options to enumerate

**(A) Hard prerequisite — promote to v0.7.1.**
Re-priority from B to A, move milestone v1.1 → v0.7.1, narrow the AC to
**embedding-lookup-only** (true gather over the rightmost vocab axis) for
the first landing. Defer general gather/scatter, scatter-add backward,
atomic-scatter, and top-k support to a follow-up issue. Rationale: GPT-2
inference is genuinely on v0.7.1's critical path and the vocab-wide
one-hot workaround is not just slow, it's embarrassing for a paper-ready
release.

Cost: 6–10 days of concentrated DI work *now*, slipping v0.7.0 paper-ready
work and v0.6.4 RoPE/transformer work. Risk: design churn during
implementation may force #377 (transformer inference) to slip too, since
the embedding API ends up coupled to it.

**(B) Workaround-acceptable — defer DI to v1.1 as planned.**
Accept that v0.7.1's GPT-2 inference demo ships with vocab-wide one-hot
embedding (≈50257 × 768 ≈ 39M flops per token spent on the embedding
matmul) and host-roundtrip sampling. README's existing line-18 caveat
stands. The paper venue is workshop-level (OCaml Workshop / FProPer), not
NeurIPS — slow-but-clean is acceptable framing.

Cost: a likely-honest paragraph in the demo's README explaining the
slowdown, plus a release-note line. Risk: a reviewer / reader benchmarks
the demo against a comparable framework and the comparison embarrasses
the project; or a downstream user expects the educational demo to scale
to "actually run a model" and is disappointed.

**(C) Hybrid — design memo now, implement after v0.7.0 ships.**
Land this disposition document, plus the design-memo skeleton below, now
— before the user's audit ends. Implementation happens after gh-ocannl-271
(Adam) lands and v0.7.0 ships, so it does not contend with paper-ready
work. v0.7.1's #377 demo ships under the (B) workaround. The design memo
becomes the input to the implementation task that follows.

Cost: the design memo time (this proposal + a follow-up that fills the
backend codegen sketches with code-level verification). Risk: low. The
design memo de-risks the eventual implementation by surfacing the API
choice early; (B)'s workaround embarrassment lasts only one release.

### Recommendation

**(C) Hybrid.** Reasoning:

- (A)'s "DI before paper" framing assumes the paper venue cares about
  inference wall-clock. The roadmap's paper venue is OCaml Workshop /
  FProPer; both reward clean abstractions and reproducible numerics over
  raw throughput. Slow-but-correct GPT-2 inference is fine for that.
- (B)'s "defer to v1.1" reading assumes the issue is genuinely
  non-blocking. It mostly is — the demo runs without DI — but indefinite
  deferral leaves the API-choice question unresolved and means the next
  person who hits a DI requirement (kv-cache slicing for fast inference,
  proper variable-length attention, top-k sampling without host
  roundtrip) re-discovers everything.
- (C) splits the resolvable-now part (write down the design space) from
  the implementation-when-ready part. The design memo costs ≤2 days; the
  implementation slot it bookmarks (post-v0.7.0, parallel to v0.7.2
  optimization work) is a natural fit.

This is a recommendation, not a decision. The user's audit picks (A), (B),
or (C) — the document is the artifact regardless.

### Design memo skeleton (only filled if disposition (C) is picked)

The design memo is **scope for this proposal's eventual artifact**, not
this proposal. The skeleton enumerates what the memo must cover; the
worker writing the memo fills each section with code-level verification
against current backend internals.

#### API options for the user-facing surface

1. **Special axis-index variant.** Add `Dynamic of Tn.t * axis_index array`
   (or similar) to `Indexing.axis_index`. Pros: minimal surface area,
   integrates with existing projections / shape inference path. Cons: the
   `Tn.t` reference inside `axis_index` cross-cuts a previously
   value-typed indexing module; serialization/comparison/hashing all need
   care.
2. **First-class `Gather` / `Scatter` ops in `tensor/operation.ml`.**
   Pros: clean separation, easy to teach in the einsum DSL, easy
   gradient story (gather backward = scatter-add, scatter backward =
   gather). Cons: doesn't compose with the existing einsum/projection
   machinery — you get a parallel surface that's its own thing.
3. **Hybrid: `Gather`/`Scatter` ops at the tensor layer, lowered to a
   special `axis_index` variant at `low_level`.** Pros: each layer keeps
   its idioms. Cons: more lowering machinery to maintain.

Memo must record the trade-offs and **recommend** one (with verification
against current `tensor/operation.ml` and `arrayjit/lib/low_level.ml`).

#### Codegen story per backend

For each of CC, CUDA, Metal, the memo must answer:

- How does `Get` lower today (point at `arrayjit/lib/c_syntax.ml` /
  `cuda_backend.ml` / `metal_backend.ml`)?
- What changes when one of the index slots is a runtime-tensor-valued
  load instead of a compile-time-known iterator/affine combination?
- Bounds checking strategy: clamp, modulo, or trap? (CHANGES.md line 588
  records that the prior prototype produced "virtual nodes" — does the
  new design need to support virtual gather, or only materialized?)
- Atomic scatter on each backend: CC has `__atomic_fetch_add`; CUDA has
  `atomicAdd`; Metal has `atomic_fetch_add_explicit`. Each has float-type
  caveats.
- Single-thread CUDA constraint: `cuda_backend.ml`'s `kernel_prep_line`
  forces `grid_dim=1, block_dim=1` (per harness memory). The memo must
  confirm this does not block the gather/scatter codegen — it
  shouldn't, single-threaded gather is just a load — but the atomic
  scatter requirement may need a note about future parallelism plans.

#### Gradient story

- gather forward → scatter-add backward (with atomic accumulation if
  duplicate indices possible).
- scatter forward → gather backward (read-only, no atomics).
- Interaction with the existing `grad_asn` / `op_asn` plumbing in
  `tensor/operation.ml` — point at a representative existing op (e.g.
  `slice`) for the pattern.

#### Open questions for the memo to surface

- Out-of-bounds policy: clamp / modulo / undefined-behavior / runtime
  trap?
- Duplicate-index policy on scatter: atomic-add (slow but correct) or
  declared-unique-only (fast, contract-based)?
- Interaction with virtual tensors (the prototype supported virtual
  dynamic-indexing per CHANGES.md line 588 — does the new design need
  this on day one)?
- Shape-inference contract: when the output dim depends on the index
  tensor's shape, does shape inference need a new "row variable that
  resolves from a tensor's runtime shape" concept, or do we restrict to
  cases where the index tensor's shape is known at compile time?

## Scope

- **In scope:** Writing `docs/proposals/gh-ocannl-186.md` (this file's
  eventual full content if the disposition skeleton is filled), committing
  it to the OCANNL repo, linking from the GitHub issue.
- **Out of scope:** Implementing dynamic indexing in any form. Picking
  among the three API options. Picking among the three backend codegen
  approaches. Re-prioritising the issue's milestone or harness priority.

**Dependencies:**

- None for the disposition document itself.
- If disposition (A) is picked: the implementation task that follows
  blocks #377 (transformer inference demo) in v0.7.1.
- If disposition (B) or (C) is picked: independent of #377.

**Surfaced question for the user (load-bearing):**

*(Update 2026-06-12: effectively answered — the issue's milestone was moved
to v0.7.1 on 2026-05-11, i.e. the promotion path was chosen; the question
below is kept for the historical record.)*

> Issue #186 is currently labeled `explore` / milestone v1.1 / priority B.
> Issue #377 (GPT-2 inference demo, milestone v0.7.1) is the most
> prominent downstream consumer of dynamic indexing. Without DI, #377
> ships with vocab-wide one-hot embedding (≈39M flops per token spent on
> embedding alone for GPT-2's 50257 vocab) and host-roundtrip sampling
> per generated token. Pick a disposition: (A) promote DI to v0.7.1 and
> narrow first landing to embedding-only, (B) accept the workaround for
> v0.7.1 and keep DI at v1.1 as planned, or (C) write the design memo
> now and implement post-v0.7.0. Recommendation: (C).

## Design review (2026-06-12)

**Verdict: sound-with-changes.** The disposition framework was the right
artifact and its claims check out against the code (verified today:
`indexing.ml:104-120` variant list; `Batch_slice`/`@|` at
`operation.ml:617-630`; `one_hot_of_int_list` at `nn_blocks.ml:61`;
README line 18; CHANGES line 527; `cuda_backend.ml:317` and the
`grid_dim_x:1, block_dim_x:1` launch at line 970). But the milestone move
answered the question the memo poses, so the document's remaining value is
the design-memo skeleton — and that skeleton has one structural gap: it does
not position dynamic indexing against #343's virtual one-hot, which is the
nearer-term solution to the same headline use case.

**Reading the milestone move.** v1.1 → v0.7.1 (2026-05-11) means "implement
DI in the v0.7.1 timeframe", i.e. (A)/(C)-accelerated. But note the #377
proposal (`docs/proposals/gh-ocannl-377.md`, "single-token one-hot is
feasible (~200KB) ... avoids the need for" dynamic indexing) concludes the
inference demo does **not** need DI. So the promotion should be read as
"same release, not blocking": keep the narrow first landing
(embedding-lookup/gather only) and do not let #377 wait on DI.

**#186 vs #343 — complementary or competing?** Both, explicitly:

- *Short term: complementary, different layers.* #343 (milestone v0.8) keeps
  all indices static and recovers gather codegen via an `Equality_with_index`
  fetch op plus a `simplify_llc` pattern — no shape-inference changes, safe
  fallback when the pattern misses. #186 makes runtime-tensor-valued indices
  first-class (API + shape inference + codegen) and covers what #343
  structurally cannot: scatter, top-k sampling without host roundtrip,
  kv-cache writes, variable-length attention.
- *Long term: competing on the embedding case only.* If #186 lands fully, a
  user-facing gather subsumes #343's embedding-lookup win; #343's simplifier
  pattern then survives as an optimization for einsum-with-one-hot code.
- *Shared substrate — the actionable point:* #343 Phase 3 already proposes a
  minimal `axis_index` extension (`Dynamic_lookup of { tn; idcs }`, rendering
  as `(int)(buffer[offset])`). The #186 design memo should adopt that variant
  as the common IR substrate — i.e. API option 3 ("hybrid": tensor-layer
  `Gather`/`Scatter` lowering to a special `axis_index`) lowering to the
  *same* variant #343's simplifier emits. Decide which issue owns the
  `indexing.ml` change and record it in both documents, otherwise two rival
  index-variant designs will accrete. (Also: `gh-ocannl-377.md` line 93
  mislabels #343 as "dynamic indexing" — fix that cross-reference.)

**Recommendations:**

1. Re-scope the acceptance criteria: the "enumerate three dispositions, user
   picks" ACs are moot post-milestone-move. The deliverable now is filling
   the design-memo skeleton (API choice, per-backend codegen, gradient
   story) plus the #343 division-of-labor statement above.
2. Resolve two of the memo's open questions now rather than deferring:
   out-of-bounds policy — clamp as the default (cheap, deterministic,
   matches #343's safe-fallback spirit; optional trap under a debug flag);
   virtual gather — not needed day one, #343's virtual one-hot covers the
   inlining-shaped uses.
3. Sequence the first landing as gather-only (embedding lookup over the
   rightmost axis), with scatter/scatter-add deferred. The single-threaded
   CUDA/Metal execution model makes gather trivial codegen today, but
   atomic-scatter design should be written *after* the v0.8
   tiling/parallelism work changes the kernel launch story — note that
   dependency in the memo.
4. Restrict shape inference initially to index tensors with
   compile-time-known shapes (no "row variable resolved from runtime shape"
   concept) — this keeps the first landing inside the existing
   constraint-solver vocabulary (post-GLB-reversal: "refines", join
   semilattice).

**Open decision points for Łukasz:**

- Confirm the milestone-move reading: is DI genuinely intended *in* v0.7.1
  (even though #377 doesn't need it), or should the issue ride along to
  v0.7.2/v0.8 next to #343?
- Which issue owns the `axis_index` extension — #343 (as its Phase 3) with
  #186 building on it, or #186 with #343's simplifier retargeted to it?
- Out-of-bounds policy sign-off: clamp (recommended) vs trap vs documented
  undefined behavior.

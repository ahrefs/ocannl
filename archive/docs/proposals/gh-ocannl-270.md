# Imbue 70B Infrastructure — Lessons Memo

## Goal

Address [gh-ocannl-270](https://github.com/ahrefs/ocannl/issues/270) — "Any
lessons from Imbue for training-in-the-large?" — by producing a written
disposition memo that extracts transferable lessons from
[Imbue's 70B infrastructure guide](https://imbue.com/research/70b-infrastructure/)
and routes each lesson either to an OCANNL milestone, an existing task, a new
backlog issue, or an explicit "not applicable / not now" bucket.

The issue is labelled `explore`, milestoned `v1.1` (post-v1.0), effort `small`.
Imbue's guide is overwhelmingly about *external* infrastructure — bare-metal
cluster bring-up, host-OS provisioning, InfiniBand fabric health monitoring,
fleet-level fault tolerance, evaluation harnesses — i.e. the layer *around* a
trainer, not the trainer itself. Two facts of OCANNL's current architecture
sharply constrain what transfers cleanly:

1. **Multi-streaming was deliberately removed** in
   [gh-ocannl-341](https://github.com/ahrefs/ocannl/issues/341) ("Not planned",
   2026-02-07). OCANNL has no in-process notion of multiple GPUs / multiple
   streams as parallel devices, no AllReduce / NCCL / MPI integration, and no
   cross-process coordination. Imbue's distributed-training lessons therefore
   live one architectural layer above OCANNL today — they belong to a future
   external infrastructure layer (alongside [gh-ocannl-278](https://github.com/ahrefs/ocannl/issues/278)
   DisTrO and [gh-ocannl-293](https://github.com/ahrefs/ocannl/issues/293)
   sharding) — and adding multi-streaming back is explicitly *not* a goal of
   this memo.

2. **CUDA kernels run single-threaded today** (`grid_dim=1, block_dim=1`,
   enforced by the `kernel_prep_line` guard in `arrayjit/lib/cuda_backend.ml`).
   The whole v0.8 GPU performance milestone starts from this baseline. Any
   kernel-level lessons from neighbour projects feed v0.8, not now.

A second co-cited input — Karpathy's
[llm.c GPT-2 1.6B training discussion](https://github.com/karpathy/llm.c/discussions/677)
— overlaps almost completely with the existing
[gh-ocannl-253](https://github.com/ahrefs/ocannl/issues/253) "Study and
incorporate Karpathy's `llm.c` lessons" task, whose proposal already covers
kernel-level (warp shuffles, online softmax, `float4` loads), memory
(contiguous parameter buffers, in-place ops), and training-loop patterns
(AdamW, cosine schedule, gradient clipping) in detail. Re-doing that analysis
here would duplicate work and split the discussion across two issues. This
memo therefore **scopes #270 to the Imbue portion only and defers the llm.c
lessons to #253**, with cross-links in both directions.

The right output, given all of the above, is a *disposition memo* that
future-us can read once prerequisites land — not a study to pull lessons into
v0.7.x or v0.8 cycles.

## Acceptance Criteria

The deliverable is a single document committed to `docs/` (an "Imbue lessons"
memo, file location TBD by the implementer — see Approach below) that
satisfies the GH issue's implicit ACs in their cheapest faithful form:

- [ ] **AC1 — Imbue lessons extracted and categorised**: the document
      summarizes Imbue's 70B guide at a level sufficient for future planners
      to recognize which lessons apply, organized into the buckets *Compiler /
      runtime concerns OCANNL touches today*, *Future external-infrastructure
      layer (post-v1.0)*, and *Not applicable to OCANNL's scope* (with a
      one-line rationale per item in the third bucket). Each lesson cites the
      relevant section of the Imbue guide.
- [ ] **AC2 — Headline empirical claims preserved verbatim**: the document
      records Imbue's load-bearing numbers as stated, with a "as of June 2024"
      caveat — at minimum (a) ~3% of machines break per week on newest
      hardware, (b) most training failures trace to InfiniBand links or GPU
      hardware (not software), (c) checkpointing and evaluation dominate
      training-time slowdowns. These are the operational facts a future infra
      layer must design around.
- [ ] **AC3 — Routing decision per lesson**: every lesson in bucket 1
      (compiler / runtime) is mapped to one of: an existing OCANNL task /
      proposal, a new backlog item filed during this work (with the GH issue
      number recorded in the memo), or an explicit "v0.8 input — included by
      reference in [gh-ocannl-253](https://github.com/ahrefs/ocannl/issues/253)
      / `docs/megakernel-deep-dive.md`" pointer. Bucket 2 lessons are simply
      noted with the milestone label `v1.1+` and cross-referenced to
      [gh-ocannl-278](https://github.com/ahrefs/ocannl/issues/278) (DisTrO)
      and [gh-ocannl-293](https://github.com/ahrefs/ocannl/issues/293)
      (sharding) where relevant. Bucket 3 lessons need only the rationale.
- [ ] **AC4 — #253 boundary stated explicitly**: the memo opens with a one-
      paragraph note that llm.c lessons are *out of scope* for this document
      and live in `docs/proposals/gh-ocannl-253.md` / the future
      `docs/llm-c-analysis.md`, with bidirectional cross-links. (The
      `gh-ocannl-253.md` proposal should be amended in the same PR to mention
      this memo as the Imbue counterpart, but this is a one-line edit and not
      a separate AC.)
- [ ] **AC5 — Workshop-paper relevance honestly scoped**: the memo includes
      a short "paper relevance" subsection that states plainly that Imbue's
      infra lessons do *not* materially feed the OCaml Workshop / FProPer
      2026 paper (whose angle is generalized einsum, row variables, and
      shape inference) and should not be cited there to inflate scope.
- [ ] **AC6 — Memo is committed and the issue is updated**: the document
      ships in the OCANNL repo so it survives outside the harness; GH issue
      #270 is updated to link to it; any new backlog issues filed under AC3
      are linked in both directions.

Out of scope (explicitly):

- Re-deriving llm.c lessons (those belong in #253).
- Proposing any addition of multi-streaming, multi-device coordination,
  AllReduce, NCCL/MPI, or process-level fault tolerance to OCANNL itself —
  these are the future external-infrastructure layer, not OCANNL.
- Implementing any of the bucket-1 lessons (those become their own
  proposals / tasks per AC3).
- Producing benchmark numbers, reproducing any of Imbue's hardware-failure
  measurements, or recommending hardware procurement.
- Updating the workshop paper (AC5 is a *negative* boundary — keep this
  memo out of the paper's bibliography).

## Context

### What Imbue's 70B guide is

Imbue (formerly Generally Intelligent) released an end-to-end public guide
covering the path from "rented bare-metal H100 cluster" to "70B model trained
end-to-end". Its sections are (paraphrased): cluster bring-up and host-OS
provisioning; InfiniBand fabric setup, monitoring, and link-failure recovery;
NCCL configuration; pre-flight diagnostics; checkpointing strategy; evaluation
harness; fault-tolerance and automated error recovery; and post-mortems on
real failures.

The unifying observation is that at the 256-512 GPU scale, hardware fails
*often enough that software design has to assume it*: roughly 3% of machines
fail per week on the newest hardware, the majority of training-loop
interruptions trace to InfiniBand link flaps or GPU hardware faults, and
checkpoint-and-resume dominates real wall-clock training time alongside
evaluation. None of these are kernel-generation concerns; almost all of them
are concerns of a fleet-management layer that lives *above* the trainer.

### Why this is overwhelmingly external infrastructure

OCANNL today is a single-process, single-device tensor compiler with backends
for C (CPU), CUDA, and Metal. There is no notion of:

- multiple processes coordinating on the same training run,
- multiple devices participating in one logical computation
  (the multi-streaming abstraction was removed in
  [gh-ocannl-341](https://github.com/ahrefs/ocannl/issues/341)),
- inter-node communication primitives,
- a process supervisor / orchestrator that restarts crashed trainers,
- a fabric-health monitor.

Building any of those would be a v1.1+ "external infrastructure layer"
project, in the same family as
[gh-ocannl-278 (DisTrO)](https://github.com/ahrefs/ocannl/issues/278) and
[gh-ocannl-293 (sharding)](https://github.com/ahrefs/ocannl/issues/293), and
ahead of any of those is the v0.7.x → v0.8 chain (RoPE → transformer →
tokenizers → GPT-2 inference → GPU performance baseline). The Imbue lessons
that *are* potentially close to OCANNL's current scope are a small minority,
and they overlap heavily with existing tasks:

- **Pre-allocated, contiguous parameter buffers** — already on the path via
  [gh-ocannl-344 (Universal Pool Allocator)](https://github.com/ahrefs/ocannl/issues/344)
  and llm.c-derived recommendations in `docs/proposals/gh-ocannl-253.md`.
- **Tensor checkpoint save/load** — shipped in
  [gh-ocannl-373](https://github.com/ahrefs/ocannl/issues/373)
  (`docs/proposals/tensor-persistence.md`); Imbue's "checkpointing dominates
  training time" finding is a tuning input for that subsystem, not a new
  proposal.
- **Bitwise-deterministic execution** — Imbue and llm.c both rely on this for
  debugging. OCANNL has
  [#341 (multicore_cc non-determinism)](https://github.com/ahrefs/ocannl/issues/341)
  in flight; further determinism work is captured under #253.

### Why this is a memo, not a study cycle

The combination of: (a) `explore` label and v1.1 milestone, (b) effort
`small`, (c) almost-total overlap of the kernel-level material with #253, and
(d) almost-total external-infrastructure character of the rest, means the
right unit of work is a *categorisation memo* with explicit routing — not a
study that lands changes in v0.7.x or v0.8.

### Code pointers (for the routing decisions in AC3)

- `arrayjit/lib/cuda_backend.ml` — kernel generation; `kernel_prep_line` is
  the single-threaded-kernel guard. v0.8 is where any GPU-side lessons (from
  llm.c / Imbue) would land, via #253 / #412 / megakernel work.
- `arrayjit/lib/builtins_cuda.ml` — CUDA-specific builtins (warp shuffles
  absent today; tracked under #253).
- `docs/proposals/gh-ocannl-253.md` — existing comprehensive llm.c gap
  analysis; the canonical home for llm.c-derived recommendations.
- `docs/megakernel-deep-dive.md` — existing megakernel analysis; relevant
  for v0.8 kernel-level decisions.
- `docs/proposals/gh-ocannl-344.md` — Universal Pool Allocator proposal;
  relevant for "pre-allocated contiguous buffers" routing.
- `docs/proposals/tensor-persistence.md` — completed tensor save/load
  (#373); relevant for "checkpointing dominates time" routing.
- `docs/proposals/distro-feasibility-study.md` — sibling defer-with-doc
  memo for #278; this proposal's structural template.
- `ROADMAP.md` — milestone definitions referenced in AC3.

## Approach

*Suggested approach — the implementer may deviate if they find a better
shape, but this is a docs-only deliverable and the structure is largely
determined by the ACs.*

1. **File location.** Create a single Markdown memo. Suggested filename:
   `docs/imbue-infrastructure-lessons.md` (lives next to
   `docs/megakernel-deep-dive.md` and the workshop docs, parallel to where
   `docs/llm-c-analysis.md` is planned to land for #253). An alternative is
   `docs/proposals/gh-ocannl-270-memo.md` if the implementer prefers
   keeping the disposition documents co-located. Either is fine; pick one
   and link from the GH issue.
2. **Memo structure.**
   - *§1 Scope and what this memo is not* — the AC4 disclaimer, plus the
     paper-relevance note for AC5.
   - *§2 Imbue's guide at a glance* — one-paragraph summary per Imbue
     section, with the AC2 numbers verbatim in a callout block.
   - *§3 Bucket A — Compiler / runtime concerns* (lessons close to
     OCANNL's current scope): pre-allocated parameter buffers,
     deterministic execution, checkpoint format efficiency. Each entry
     ends with a "Routing:" line per AC3.
   - *§4 Bucket B — Future external-infrastructure layer*: cluster
     bring-up, InfiniBand health, NCCL config, fault-tolerant supervisors,
     fleet-level evaluation. Each entry ends with a "v1.1+, see #278 /
     #293 / new issue #NNN" pointer.
   - *§5 Bucket C — Not applicable*: items genuinely orthogonal to
     OCANNL's scope (host-OS provisioning, hardware procurement, RDMA
     driver troubleshooting). One-line rationale each.
   - *§6 Backlog items filed* — table of any new GH issues opened during
     this work (likely zero or very few), per AC3.
3. **Cross-linking.** In the same commit, append a one-line note to the
   top of `docs/proposals/gh-ocannl-253.md` pointing at this memo as the
   Imbue counterpart, satisfying AC4's bidirectional cross-link.
4. **Issue update.** After the commit lands, post a comment on GH #270
   linking to the memo and to any backlog issues opened.

The memo is expected to be short — single-digit pages — because most lessons
route to existing tasks or to the v1.1+ infrastructure layer rather than
generating new work.

## Scope

**In scope**: producing the categorisation memo described above; opening
backlog issues for any genuinely new bucket-A lessons that don't already have
an OCANNL task; the one-line cross-link edit to `gh-ocannl-253.md`; updating
the GH issue #270 with a pointer to the memo.

**Out of scope**: everything in the "Out of scope (explicitly)" list under
Acceptance Criteria. In particular, this proposal must not become a vehicle
for re-introducing multi-streaming, for kernel-level v0.8 work, or for
re-doing #253's analysis.

**Dependencies**: none blocking. The memo can be written and committed
without changes to OCANNL source code. It should be written *before*
[gh-ocannl-253](https://github.com/ahrefs/ocannl/issues/253) is started in
earnest only if the implementer of #253 wants the Imbue boundary settled
first; otherwise the two are independent.

## Related Tasks

- [gh-ocannl-253](https://github.com/ahrefs/ocannl/issues/253) — Karpathy
  llm.c study; *all* llm.c lessons live there (per AC4).
- [gh-ocannl-278](https://github.com/ahrefs/ocannl/issues/278) — DisTrO
  distributed training; sibling defer-with-doc memo, structural template
  for this proposal.
- [gh-ocannl-293](https://github.com/ahrefs/ocannl/issues/293) — sharding
  and slicing with minimal copying; receives bucket-B routing pointers.
- [gh-ocannl-344](https://github.com/ahrefs/ocannl/issues/344) — Universal
  Pool Allocator; receives bucket-A "pre-allocated buffers" routing.
- [gh-ocannl-373](https://github.com/ahrefs/ocannl/issues/373) — tensor
  persistence (done); the checkpointing-cost finding from Imbue is tuning
  input for this subsystem.
- [gh-ocannl-341](https://github.com/ahrefs/ocannl/issues/341) —
  multi-streaming removal ("Not planned"); the architectural boundary that
  forces Imbue's distributed lessons into bucket B.

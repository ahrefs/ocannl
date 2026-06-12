# DisTrO/DeMo Distributed Training — Feasibility Study

## Status update (2026-06-12)

- gh-ocannl-278 is still OPEN, milestone v1.1. ROADMAP.md's post-1.0 section now lists "DisTrO distributed training (#278, see docs/proposals/distro-feasibility-study.md)" — i.e. Approach step 2 (ROADMAP pointing at this doc) is done. Remaining: comment on the issue with the disposition question (Approach step 3).
- Issue-reference corrections (the original text mischaracterized several):
  - gh-ocannl-341 is actually titled "Resolve non-determinism of multicore_cc and restore it as the primary testing target" and is CLOSED/COMPLETED (not "Not planned"). The multi-stream cleanup landed via commit 692d8c9d: cross-stream *automatic coherence* was removed, but multiple streams per device (and cross-device `device_to_device` transfers, which now return transfer routines) remain.
  - gh-ocannl-271 is "Support quantization for optimizers: low-bit optimizers", CLOSED/NOT_PLANNED — it is *not* the AdamW issue. No open issue tracks AdamW as of 2026-06-12; an implementation arc would need to file one.
  - gh-ocannl-270 (Imbue training-in-the-large lessons) is CLOSED/COMPLETED.
  - gh-ocannl-293 (sharding/slicing with minimal copying) is OPEN; ROADMAP places it under v1.0 "Feature completeness" (the GH milestone field lags).
- Per ROADMAP (the authority on milestones), v1.0 targets end of October 2026; the "v0.8 GPU performance milestone consumes available cycles" framing is aging but the cycle-scarcity argument still holds.
- The prerequisite chain and the recommendation of disposition (a) remain valid; "execution is single-device" should be read as "no multi-process/multi-host execution" — multi-stream and multi-device execution exist at the backend level.

## Goal

Address [gh-ocannl-278](https://github.com/ahrefs/ocannl/issues/278) — "Example
training loop using DisTrO low-communication distributed data parallelism" — by
producing a written feasibility document that makes the prerequisite chain
concrete, rather than building any distributed-training infrastructure.

The issue is labelled `explore`, milestoned `v1.1` (post-v1.0), and the task
effort is `large`. OCANNL currently has no distributed training capability:
the deprecated multi-stream "virtual GPU" data-parallelism infrastructure was
deliberately removed under
[gh-ocannl-341](https://github.com/ahrefs/ocannl/issues/341) *(Update
2026-06-12: closed as completed, not "Not planned"; multiple streams per
device remain, but cross-stream automatic coherence is gone)*,
there is no AllReduce/NCCL/MPI integration, and execution is
single-process/single-host.
The v0.8 GPU performance milestone (single-thread CUDA baseline) and the
ICFP 2026 OCaml Workshop / FProPer paper deadlines (May–June 2026) consume
the available cycles, so net-new distributed infra is not affordable now.

The right output for this issue, given those constraints, is a feasibility
memo that future-us can read once prerequisites land — not a training loop.

## Acceptance Criteria

The deliverable is a single document committed to `docs/` (this file plus any
follow-up edits) that satisfies the GH issue's two ACs in their cheapest
faithful form:

- [ ] **AC1 — "Study DisTrO and assess feasibility"**: the document summarizes
      DisTrO/DeMo at a level sufficient to plan a future implementation,
      including the DCT-compressed momentum mechanism and the order-of-magnitude
      claims, with citations to the DeMo paper (arxiv 2411.19870) and the
      NousResearch/DisTrO and bloc97/DeMo repositories.
- [ ] **AC2 — "Create an example training loop OR document the path to one"**:
      the document chooses the second branch and enumerates the prerequisite
      chain explicitly, marking each prerequisite as either *blocking*
      (no DisTrO is possible without it) or *missing-but-substitutable*
      (a workable but degraded implementation is possible without it).
- [ ] The document recommends a disposition for gh-ocannl-278 itself
      (defer-with-doc / thin study task / gated implementation task) and
      surfaces the choice as a user-facing question rather than picking
      unilaterally.
- [ ] The document is committed to the OCANNL repo so it survives outside
      the harness; the GH issue is updated to link to it.

Out of scope (explicitly):

- Implementing any prerequisite (multi-process execution, comm primitives,
  DCT op, AdamW).
- Choosing between MPI / NCCL / TCP as the communication substrate.
- Producing benchmark numbers or a working training loop.

## Context

### Current OCANNL state relevant to distributed training

- Single-process execution. No multi-process or multi-host coordination layer
  exists.
- Multi-streaming-as-data-parallelism (using GPU streams or CPU threads as
  parallel "virtual GPUs") was removed; gh-ocannl-341 closed as completed
  *(Update 2026-06-12: corrected from "Not planned"; streams themselves remain,
  the removed part is cross-stream automatic coherence and the data-parallel
  `Train.parallel_update` machinery)*. Any future cross-device
  parallelism will be built from scratch on a different model.
- Optimizer surface: `Train.sgd_one` / `Train.sgd_update` in `lib/train.ml`,
  identifiable by the `"param sgd step"` block comment. This is the touchpoint
  a future AdamW landing (gh-ocannl-271) would extend, and the natural place
  a future DeMo optimizer would slot in.
- No DCT / FFT op. The ops surface lives under `arrayjit/` and `tensor/`;
  any DCT would either be implemented directly or built on top of an FFT
  primitive that does not yet exist.

### What DisTrO / DeMo is (one-paragraph summary for the doc)

DeMo (Decoupled Momentum Optimization, [arxiv 2411.19870](https://arxiv.org/abs/2411.19870),
reference impl [bloc97/DeMo](https://github.com/bloc97/DeMo)) is a fused
optimizer + data-parallel algorithm. Per-step: each worker updates local
momentum, extracts the *fast* components via discrete cosine transform,
chunks/projects/top-k-sparsifies them, and AllReduces only the resulting
small tensor. Reported communication reduction: 3–4 orders of magnitude
versus naive gradient AllReduce. NousResearch's DisTrO is the production
deployment — they trained Consilience 15B/40B with it, demonstrating that
the technique works at scale, not just at toy sizes. The interesting property
for OCANNL specifically is that low-bandwidth distributed training is
exactly the regime a small team on home-network interconnect would care
about; high-bandwidth datacenter setups are not the audience.

### Prerequisite chain (the substance of the deliverable)

These are the dependencies between today's OCANNL and a working DisTrO
training loop, ordered roughly bottom-up:

1. **Multi-process execution model.** *Blocking.* OCANNL must be able to run
   the same program on N processes that share parameter shape but hold
   independent shards of data. Today there is no such concept. This is
   the largest piece and is a design decision in its own right (one
   process per device? per host? how does the JIT pipeline see them?).
2. **Inter-process communication primitive.** *Blocking.* Some mechanism
   to AllReduce a tensor between processes. Candidates: ocamlmpi (Xavier
   Leroy's MPI binding), an NCCL binding (none exists for OCaml today),
   or a TCP-based AllReduce written from scratch. Selection is out of
   scope here, but the absence is blocking.
3. **AdamW optimizer.** *Blocking for DeMo, not for distributed-training-in-general.*
   DeMo is built on top of AdamW-style adaptive moments; SGD-with-momentum
   is not a drop-in substitute. *(Update 2026-06-12: AdamW is currently
   untracked — gh-ocannl-271, previously cited here, is actually the low-bit
   optimizer quantization issue, closed as not-planned.)* Without AdamW, one
   could ship distributed SGD but not DeMo.
4. **DCT (or FFT) operation.** *Missing-but-substitutable.* DeMo's compression
   step is a DCT followed by chunking/top-k. A from-scratch DCT over
   power-of-two sizes is a few hundred lines; an FFT-based implementation
   reuses more theory. A degraded variant of DeMo could in principle skip
   the DCT and rely on top-k of raw momentum — the paper does not test
   that, so it would be a research detour, not a shortcut.
5. **Top-k / sparsification ops over a tensor.** *Missing-but-substitutable.*
   Currently no top-k op; can be expressed via primitives but inefficiently.
6. **Gradient sharding / data sharding plumbing.** *Missing.* Each worker
   needs to consume a different data shard each step. Tracked partly in
   gh-ocannl-293.

Items (1) and (2) are the dominant cost. Items (3)–(6) are weeks-of-work,
not months.

### Disposition options for gh-ocannl-278 itself

These are the three plausible ways to close out the issue:

- **(a) Defer with the feasibility doc.** Land this proposal as the
  feasibility doc, mark gh-ocannl-278 as completed-by-doc, and create no
  follow-up implementation task until prerequisites (1)–(3) at minimum
  exist. This is the cheapest option and is consistent with the
  "post-v1.0, v1.1 milestone" framing in the issue and the v0.8 GPU
  performance focus.
- **(b) Thin study task that produces the feasibility memo.** Treat the
  proposal here as scaffolding, then run a 1–2 day "study" session that
  produces a more thorough doc — read the DeMo paper end-to-end, read the
  bloc97/DeMo reference impl, sketch what the OCANNL surface would look
  like for a hypothetical `Train.demo_step`. Close the issue against that
  doc.
- **(c) Real implementation task gated on prerequisites.** Convert
  gh-ocannl-278 into a tracking issue whose checklist is the prerequisite
  chain above; close it only when all prerequisites land plus a real
  example training loop runs. This is the most ambitious framing and
  implies committing to distributed training as a v1.x roadmap line.

The proposal recommends **(a)** as the default given current cycle scarcity,
with **(b)** as the alternative if the user wants a richer artifact than
this proposal itself, and **(c)** explicitly *not* recommended for v1.1
because it commits to a multi-month implementation arc that competes
directly with the v0.8 perf milestone and the ICFP paper.

### Code pointers (by symbol, not line number)

- `Train.sgd_one`, `Train.sgd_update` in `lib/train.ml` — the optimizer
  surface a future DeMo would join.
- `Tensor.diff`, `Tensor.params` — gradient and parameter handles a
  distributed optimizer would need to enumerate.
- `arrayjit/` — JIT/op surface, where a DCT primitive (if added) would land.
- ROADMAP.md's post-v1.0 section already lists DisTrO; this proposal
  argues for keeping it there rather than promoting it.

## Approach

*Suggested approach — the assigned worker may deviate.*

Given that this proposal already contains most of the substantive content
that AC1 and AC2 demand, the cheapest faithful execution is:

1. Move/expand this proposal into a standalone `docs/distro-feasibility.md`
   (or keep it under `docs/proposals/` and link from `ROADMAP.md`),
   touching up the DeMo summary if reading the paper surfaces corrections.
2. Update `ROADMAP.md` post-v1.0 entry for DisTrO to point at the doc.
3. Comment on gh-ocannl-278 with a link to the doc and the recommended
   disposition (a/b/c), letting the user pick before closing.
4. *No code changes.*

If the user picks (b), the same doc is the seed for a deeper study session.
If the user picks (c), the same doc becomes the design appendix for a
tracking issue. The doc is useful in all three branches, which is why
writing it first is the no-regret move.

## Scope

**In scope**:

- Writing the feasibility memo (this doc, possibly relocated/expanded).
- Updating the GH issue and ROADMAP.md to reference it.
- Surfacing the disposition question (a/b/c) to the user.

**Out of scope**:

- Implementing multi-process execution, comm primitives, DCT, top-k,
  data sharding, or AdamW.
- Choosing between MPI / NCCL / TCP.
- Benchmarking, paper reproduction, or any working training loop.

**Dependencies**:

- AdamW — not a prerequisite for the doc, but a prerequisite for any future
  implementation arc the doc describes. *(Update 2026-06-12: no tracking
  issue exists; gh-ocannl-271, formerly cited here, is the low-bit optimizer
  quantization issue, closed not-planned.)*
- gh-ocannl-270 (Imbue/llm.c distributed context) — sibling exploration;
  closed completed.
- gh-ocannl-293 (sharding) — sibling prerequisite for any future arc; open,
  under v1.0 per ROADMAP.
- gh-ocannl-341 (deprecated multi-stream infrastructure removed; closed
  completed) — cited as why the distributed model has to be built fresh.

# Fork-Based Backend — Disposition Memo

## Goal

Address [gh-ocannl-161](https://github.com/ahrefs/ocannl/issues/161) —
"Implement a fork-based backend as stepping stone to distributing across
machines" — by producing a written disposition document, not by building
a fork-based backend.

The issue was originally closed in 2023-09-27 with "Not helpful, `Domain`-based
devices already implemented." It was reopened in 2024-07-29 to keep more
angles on the design space alive (multi-machine stepping stone, GC isolation
from per-process GCs, design exploration). The issue is labelled
`enhancement`, milestoned `v1.1` (post-v1.0), task effort `large`.

The reopen rationale rested on a premise that **no longer holds**: at reopen
time OCANNL had a Domain-backed *parallel devices* abstraction that a
fork-backend could meaningfully sit alongside as a sibling. That abstraction
has since been deliberately removed (see Context). What remains under
`arrayjit/lib/schedulers.ml` is a single-device, single-stream `Multicore`
scheduler that uses one `Domain.spawn` to host the worker thread for one
device — not a parallel-devices-via-Domains abstraction. And even that
residual is on a v0.7.0 deletion list (ROADMAP.md, "Deprecated streams
cleanup").

So the load-bearing question for this issue is no longer *"how should the
fork-backend be designed?"* but *"is this issue still alive at all, given
that its parent abstraction is gone and its sibling concern (gh-ocannl-278,
DisTrO distributed training) has already been routed through a feasibility
memo?"* The right output here, given the v0.8 GPU perf focus and the
ICFP 2026 paper deadlines (May–June 2026), is a disposition document, not
a backend design.

This proposal recommends a disposition and surfaces the choice as a user
question rather than picking unilaterally.

## Acceptance Criteria

- [ ] The document records the current state of parallelism in OCANNL with
      concrete pointers (Context API, removed multi-streaming, residual
      `Multicore` scheduler scheduled for cleanup) so that a future reader
      doesn't have to re-do the archaeology.
- [ ] The document enumerates the disposition options for gh-ocannl-161 and
      recommends one, with the choice surfaced explicitly to the user as
      a question rather than picked silently.
- [ ] The document, if disposition (a) or (c) is chosen, contains enough
      to justify closing or restructuring gh-ocannl-161 without further
      design work.
- [ ] The document, if disposition (b) is chosen, contains a *minimal*
      sketch of what a stepping-stone fork-backend would look like at the
      level of "what files / what one IPC primitive / what the deliverable
      shape is" — not a full design.
- [ ] The document is committed to the OCANNL repo so it survives outside
      the harness; the GH issue is updated to link to it.

Out of scope (explicitly):

- Implementing a fork-based backend, a worker pool, or any IPC plumbing.
- Choosing between Parany / pipes / Unix domain sockets / shared memory /
  ocamlmpi as the IPC substrate.
- Building or benchmarking anything.
- Re-litigating the removal of multi-streaming (gh-ocannl-341 is settled).

## Context

### Current state of parallelism in OCANNL (verified in code)

Anyone reading the original 2023-09-27 close comment ("Domain-based devices
already implemented") needs to know that the world it described is gone.
What is true today, in `~/ocannl-staging` at the time of writing:

- **Stream-based parallelism was removed in 0.6.1** (CHANGES.md:
  *"Removed stream-based parallelism in favor of simpler Context API"*,
  *"Context API as simplified backend interface replacing stream-based
  parallelism"*). [gh-ocannl-341](https://github.com/ahrefs/ocannl/issues/341)
  closed as *Not planned*. There is no longer a "GPU streams as parallel
  GPUs" or "CPU threads as parallel GPUs" abstraction.
- **Residual `Multicore` scheduler exists but is on the chopping block.**
  `arrayjit/lib/schedulers.ml` defines a `Multicore` functor that uses
  `Domain.spawn` to run one worker thread for one device, communicating
  via a `Saturn_lockfree.Single_prod_single_cons_queue`. This is a
  *single-device single-stream* model — `assert (… get_device …: only
  device 0 exists)` is wired into the implementation. It is NOT a
  parallel-devices abstraction. It is registered in `backends.ml` as the
  default backend (`multicore_cc`), but ROADMAP.md v0.7.0 explicitly
  lists *"Deprecated streams cleanup — Remove legacy streams functionality."*
  Companion document: `streams-cleanup.md`.
- **Available backends:** `multicore_cc`, `sync_cc`, `cuda`, `metal`. None
  is multi-device; CUDA kernels run with `grid_dim=1, block_dim=1` (the
  v0.8 baseline).
- **Context API is the user-facing parallelism story now**, and there isn't
  one beyond single-device. There is no parallel-devices abstraction for a
  fork-backend to be a sibling of.
- **Distributed training is routed through gh-ocannl-278** and dispositioned
  in `docs/proposals/distro-feasibility-study.md` as defer-with-doc. The
  feasibility study explicitly notes "any future cross-device parallelism
  will be built from scratch on a different model."

### Why the 2023 close reasoning has decayed

The 2023 reasoning was: *fork-based backend is unnecessary because we
already have Domain-based devices, which already give us GC isolation and
parallelism on a single machine, and the IPC patterns for multi-machine
distribution can be deferred until they're actually needed.*

Step by step, today:

- **"Domain-based devices already implemented"** — false today. The
  Domain machinery is single-device. See above.
- **GC isolation** — moot in the absence of multiple devices. With a
  single-device backend there is one OCaml runtime and one GC.
- **Multi-machine stepping stone (the 2024 reopen reason)** — partially
  superseded by the DisTrO feasibility study, which already addresses
  the multi-machine distributed-training direction and concludes
  cross-device parallelism will be built fresh. A fork-backend could
  still in principle be a stepping stone *to* whatever distributed
  abstraction lands later, but its value is now contingent on a future
  abstraction that doesn't exist and isn't on the roadmap before v1.1.
- **Design exploration angle (the other 2024 reopen reason)** — the
  cheapest version of this is a written design memo that does not require
  building anything. Which is what this document is.

### Strategic context (why net-new work here is unaffordable)

Per the harness memory and ROADMAP.md:

- v0.8 (mid-June 2026) is single-thread CUDA performance work — tiling,
  megakernels, llm.c lessons. CUDA kernels currently run `grid_dim=1,
  block_dim=1`. This is the dominant cycle sink through mid-2026.
- ICFP 2026 OCaml Workshop / FProPer paper deadlines are May–June 2026.
- Autonomous OCANNL work is paused pending a hands-on quality audit by
  the user; proposals are artifact-only.
- Distributed/multi-process work is post-v1.0 (v1.1+) per ROADMAP.md.

A fork-backend implementation does not advance any of those, and the
sibling distributed-training task already has a deferred disposition.

### Disposition options for gh-ocannl-161

These are the four plausible ways to close out the issue:

- **(a) Close as "not planned," aligned with gh-ocannl-341.** Land this
  memo, comment on the issue with a link, close the issue. Rationale: the
  parent abstraction is gone, the multi-machine direction is covered by
  gh-ocannl-278's feasibility doc, and the GC-isolation argument is moot
  in a single-device runtime. This is the cheapest option and the one
  most consistent with the v0.8 / ICFP focus.
- **(b) Defer with a thin "design memo" follow-up task.** Land this doc as
  scaffolding, then run a 1–2 day "study" session that produces a richer
  design memo: which IPC primitive (Parany worker pool? fork-and-pickle?
  pipes-with-marshal? ocamlmpi?), what the minimal stepping-stone
  deliverable would be, what the type signature of a hypothetical
  `Fork_scheduler` functor in `arrayjit/lib/schedulers.ml` would look
  like by analogy with the residual `Multicore` functor. Close the issue
  against that memo. Useful only if the user wants a record of *how* a
  fork-backend would be built when prerequisites land, distinct from the
  *what-and-why* in this document.
- **(c) Convert to a tracking issue for "fork-as-stepping-stone-to-distributed."**
  Restructure gh-ocannl-161 around the prerequisite chain shared with
  gh-ocannl-278 (multi-process execution model + IPC primitive) and let
  it be closed only when those prerequisites land. This is the most
  ambitious framing and effectively merges 161 into the distributed-
  training arc; it's not really a "fork backend" anymore at that point.
- **(d) Keep open, do nothing, revisit at v1.1.** Status quo. Costs nothing
  but adds noise to the issue tracker and risks the same archaeology
  having to be re-done by whoever picks the issue up cold.

The proposal recommends **(a)** as the default. The 2024 reopen rationale
("more angles on the design space") is preserved by *this document*, which
is a more useful artifact than a perpetually-open issue. **(b)** is the
right alternative if the user wants a more concrete "what would we build"
record. **(c)** is technically the most coherent if distributed-training
is going to be pursued post-v1.0, but it duplicates gh-ocannl-278's role.
**(d)** is dispreferred because it leaves stale context in the tracker.

### Code pointers (by symbol, not line number)

- `Schedulers.Multicore` in `arrayjit/lib/schedulers.ml` — the residual
  Domain-based scheduler. Single-device. The natural place a hypothetical
  `Schedulers.Fork` functor would sit by analogy if disposition (b) is
  taken.
- `Schedulers.Sync` in the same file — the simpler synchronous reference
  scheduler.
- `Make_device_backend_from_lowered` in `arrayjit/lib/backends.ml` and the
  `fresh_backend` dispatcher (default `multicore_cc`) — the registration
  point where a fork backend would be wired up.
- `Backend_intf` in `arrayjit/lib/backend_intf.ml` — the interface a
  fork-backend would implement. Note its `stream` / `device` types are
  themselves on the streams-cleanup chopping block; any fork-backend
  design should be specified against the *post-cleanup* interface, not
  the current one.
- `streams-cleanup.md` — the v0.7.0 cleanup plan; relevant because
  the residual stream infrastructure is what a 2024 fork-backend would
  have plugged into.
- `docs/proposals/distro-feasibility-study.md` — the sibling v1.1
  feasibility memo for gh-ocannl-278. Sets the precedent for memo-as-
  disposition for post-v1.0 distributed-execution issues.
- `ROADMAP.md` post-v1.0 section — currently lists DisTrO but does not
  list gh-ocannl-161; disposition (a) makes that absence intentional,
  disposition (b/c) would add an entry pointing here.

## Approach

*Suggested approach — the assigned worker may deviate.*

This document is itself the deliverable for disposition (a). For (b), it
is the seed for a deeper study session. For (c), it is the design appendix
for a tracking issue. The doc-first move is no-regret in all three
branches.

Concrete steps (only step 4 varies by disposition):

1. Land this proposal at `docs/proposals/gh-ocannl-161.md`.
2. Comment on gh-ocannl-161 with a link to this doc and a summary of the
   recommended disposition (a/b/c/d), letting the user pick.
3. Update task frontmatter (`proposal: docs/proposals/gh-ocannl-161.md`).
4. Branch by user's pick:
   - **(a):** close gh-ocannl-161 with a link to this doc; no roadmap
     change needed (issue is already absent from ROADMAP.md).
   - **(b):** create a follow-up task for the 1–2 day design memo;
     leave 161 open until that memo lands.
   - **(c):** rewrite the issue body around the prerequisite chain;
     consider whether to merge into gh-ocannl-278's tracking arc.
   - **(d):** re-tag the issue with a "v1.1-explore" marker and link
     this doc as the standing context.
5. *No code changes in any branch.*

## Scope

**In scope**:

- Writing this disposition memo.
- Updating the GH issue to reference it.
- Surfacing the disposition question (a/b/c/d) to the user.

**Out of scope**:

- Implementing a fork-backend, worker pool, or IPC primitive.
- Choosing between Parany / pipes / Unix sockets / ocamlmpi.
- Reviving multi-streaming (gh-ocannl-341 is settled).
- Touching `Schedulers.Multicore` — that lives under the streams-cleanup
  task, not here.

**Dependencies**:

- gh-ocannl-341 (multi-streaming removed) — closed; cited as why the
  parent abstraction is gone.
- gh-ocannl-278 (DisTrO distributed training) — sibling v1.1 issue,
  already dispositioned via `docs/proposals/distro-feasibility-study.md`.
  Disposition (c) for 161 effectively merges into 278's arc.
- v0.7.0 streams-cleanup (`streams-cleanup.md`) — removes the
  residual stream infrastructure a fork-backend would otherwise sit
  alongside; any future fork-backend design must target the post-cleanup
  interface.

# Proposal: Experiment Tracking Story — Minimal Metrics API + CSV + Narrative Doc

**Issue**: [#122](https://github.com/ahrefs/ocannl/issues/122)
**Milestone**: v1.0

## Status update (2026-06-12)

- Issue #122 is still **OPEN**, milestone v1.0 (GH milestone due date 2026-06-30; ROADMAP.md, the milestone authority, targets v1.0 for end of October 2026).
- **No implementation has landed**: there is still no `lib/metrics.ml`, no `Metrics` re-export in `lib/ocannl.ml`, and no `docs/experiment_tracking.md`.
- `Train.example_train_result` is still defined (now at `lib/train.ml:183`) and still referenced nowhere else in the tree — the delete-or-wire-up decision remains open.
- `test/training/moons_demo.ml` still hand-accumulates `losses` / `log_losses` / `learning_rates` into ref lists (lines 37-39) and feeds them to `PrintBox_utils.plot` — the retrofit target is unchanged.
- `val get_used_memory : device -> int` is still in `arrayjit/lib/backend_intf.ml` (now line 259). Note `backend_intf.ml` has changed elsewhere since this proposal (e.g. `device_to_device` now returns a transfer routine; `merge_buffer_use = No | Copy`), but none of that affects this proposal's claims.
- No repo-wide renames (basis/refines etc.) affect this proposal.
- The full scope — Metrics module, CSV export, demo retrofit, snapshot, narrative doc, follow-up issues — remains to do.

## Goal

Issue #122 asks OCANNL to "have a story about / support for experiment
tracking: graphs of observables e.g. loss, device health". The user explicitly
wrote "This needn't be the core library though." Comments on the issue mention
W&B as the gold standard, ppx_minidebug numerical logging as one possible
direction, and Ketrew / labml / Aim as references.

This is a v1.0 *story* task. The deliverable is not a sprawling experiment
tracker — it is a narrative answer to the question "if I want to track my
training runs in OCANNL, what do I do?" plus a tiny, falsifiable demonstration
that the answer works today.

The proposal therefore narrows scope to:

1. A small in-tree metrics-logging API that captures what training demos and
   tutorials currently re-implement by hand (ref-list accumulation).
2. One CSV export so a run produces a portable artefact — the foundation for
   any external graphing pipeline (gnuplot, pandas, W&B importer, …).
3. One existing demo retrofitted to use the new API, snapshotted via the
   existing `.expected` testing pattern so the story is verifiable, not
   aspirational.
4. A short narrative document `docs/experiment_tracking.md` that names the
   options (in-tree CSV, ppx_minidebug numerical logging, external systems
   like W&B / Aim / Ketrew) and points users at the recommended path for v1.0.

Integrations with external systems (W&B REST, ppx_minidebug numerical logging,
real-time dashboards) are explicitly *out of scope* for this task and are
deferred to follow-up issues. The point of v1.0 is to have a coherent story,
not every chapter of the book.

## Acceptance Criteria

- [ ] **Metrics API exists** in a new module — recommended location
      `lib/metrics.ml` exposed via `Ocannl.Metrics` (or co-located with
      `Train` if the implementer prefers a flatter surface). The API
      minimally provides:
    - a logger value/handle constructed from a destination (file path or
      out-channel),
    - a function to record a `(step, name, value)` triple where `step` is
      `int` and `value` is `float`,
    - a function to flush/close the logger.
  The exact names are an implementation choice — the proposal does not
  prescribe `Metrics.log` vs `Metrics.record` etc. What matters is that
  the API is small (≲ 5 public values) and obviously sufficient to replace
  the ref-list pattern in `moons_demo.ml`.
- [ ] **CSV export** is supported as the default destination format. The
      file is human-readable, has a header row, and one row per recorded
      observation. Whatever schema is chosen (long-form `step,name,value`
      vs. wide-form `step,loss,lr,…`) is documented in the narrative doc
      and consistent across the demo.
- [ ] **One demo retrofit.** `test/training/moons_demo.ml` is the
      recommended target because it already accumulates `losses`,
      `log_losses`, and `learning_rates` into ref lists (current code at
      `Stdio.printf "Epoch=…"` and the trailing `PrintBox_utils.plot`
      calls). After the retrofit, the demo records the same three series
      via the new API and writes a CSV to a deterministic path (a temp
      file or `_build`-relative path is fine).
- [ ] **Falsifiable snapshot.** The CSV (or a deterministic excerpt of it
      — e.g. first N rows, or a downsampled summary) is captured in an
      `.expected` snapshot under `test/training/`. Either extend
      `moons_demo.expected` to include the CSV head/tail, or add a small
      dedicated test (e.g. `metrics_csv_test.ml`) that exercises the API
      directly with a tiny synthetic training loop. Running `dune runtest`
      must compare against the snapshot, so a regression in the CSV
      schema or numerical output fails CI rather than silently passing.
- [ ] **Existing PrintBox plotting still works.** The retrofit must not
      remove the in-terminal plot in `moons_demo`. The metrics logger is
      additive: callers can still feed accumulated values into
      `PrintBox_utils.plot`, either by reading them back from the logger
      or by tee-ing into the existing ref lists. (The point is that the
      CSV is *another* sink, not a replacement for terminal plotting.)
- [ ] **Device health is at least mentioned.** The API should be capable
      of logging integer-valued or float-cast device metrics — at minimum
      `Backend.get_used_memory` (which already exists per
      `arrayjit/lib/backend_intf.ml`'s `val get_used_memory : device -> int`)
      can be logged via the same interface, even if no demo actually does so.
      The narrative doc shows how. A separate demo for device health is
      *not* required.
- [ ] **Narrative doc** `docs/experiment_tracking.md` exists and covers,
      in roughly this order:
    1. What problem this solves and why it lives outside the core
       computation graph (echoing the issue's "needn't be the core library").
    2. The recommended path for v1.0: the new `Metrics` module + CSV +
       offline plotting (gnuplot/pandas/printbox).
    3. Pointers to alternative paths users may prefer, *as named options
       only, not as supported integrations*: ppx_minidebug numerical
       logging (lukstafi 2024-07-15 comment), external trackers (W&B,
       Aim, labml, Ketrew per the 2024-07-24 comment).
    4. A short "how to log a custom metric" example using the new API.
    5. An explicit "what's not here yet" section pointing at follow-up
       issues for W&B integration and ppx_minidebug numerical logging.
- [ ] **Follow-up issues filed (or referenced).** At least two GitHub
      issues are linked from the doc (created as part of this PR or
      already existing) covering: (a) ppx_minidebug numerical logging
      hook-up, (b) W&B / external-tracker integration. These can be
      stubs — the goal is that the story has named "next chapters" so
      v1.0 doesn't pretend the topic is closed.
- [ ] **`example_train_result` cleanup decided.** `lib/train.ml` defines
      `type example_train_result = { … rev_batch_losses; rev_epoch_losses;
      learning_rates; used_memory; … }` which is unused anywhere in the
      tree (verified: the only occurrence is the definition itself). The
      implementer must either (a) delete it as dead code, or (b) wire it
      to the new `Metrics` module as the canonical "summary of one
      training run" type. Either choice is acceptable; doing nothing is
      not, because leaving an unused stub adjacent to the new API is
      misleading.
- [ ] `dune build` and `dune runtest` succeed; new snapshot(s) match.

## Context

### Current state of metrics in the codebase

- **No centralized metrics module exists.** `lib/ocannl.ml` re-exports
  `Train`, `Nn_blocks`, `Persistence`, `Tensor`, `Operation`, `Shape`,
  `Row`, `PrintBox_utils` — there is no `Metrics`.
- **`Train.example_train_result` is an unused placeholder.** It is defined
  in `lib/train.ml` (around the `type example_train_result = { inputs;
  outputs; model_result; infer_callback; rev_batch_losses;
  rev_epoch_losses; learning_rates; used_memory }` declaration, just
  before `run_once`) and is referenced nowhere else in the repo. It looks
  like a stub from a prior iteration that intended to standardise per-run
  bookkeeping.
- **Training demos accumulate by hand.** `test/training/moons_demo.ml`
  is the canonical example: it declares
  ```
  let losses = ref [] in
  let log_losses = ref [] in
  let learning_rates = ref [] in
  ```
  inside its training function, mutates them inside the inner loop next
  to its `Stdio.printf "Epoch=%d, step=%d, batch=%d, lr=%.3g, epoch
  loss=%.2f"` call, and finally feeds them into `PrintBox_utils.plot`
  with `Line_plot { points = Array.of_list_rev !losses; … }`. Other
  demos (`mnist_conv.ml`, `bigram.ml`, `cifar_conv.ml`, etc.) print
  losses but do not retain them in lists — they are even less structured.
- **Plotting** lives in `tensor/PrintBox_utils.ml` — the `plot` function
  takes a list of `Line_plot { points; content }` specs and produces an
  ASCII chart. Any new metrics module should leave this alone; it is the
  consumer side of the existing flow.
- **Device memory** is exposed by backends. `arrayjit/lib/backend_intf.ml`
  declares `val get_used_memory : device -> int`, and the CPU/CUDA/Metal
  backends each implement it (e.g. `arrayjit/lib/backend_impl.ml` keeps
  an `Atomic.make 0` `used_memory` counter). A metrics logger can pull
  from this without any new backend work.
- **No existing CSV / numerical logger.** A grep for `csv`, `numerical`,
  `log_value`, `graph_log` in `lib/` and `arrayjit/lib/` turns up
  nothing. ppx_minidebug is used inside the lib (`%track3_sexp`,
  `%debug7_sexp`) for *structured* debugging logs but does not currently
  emit numerical time series.
- **Test infrastructure.** `test/training/dune` runs each demo as a
  `(test (name …))` stanza with a corresponding `<name>.expected` file.
  `dune runtest` diffs stdout against the expected file (see
  `moons_demo.expected` — it captures the per-epoch printf lines and
  the rendered PrintBox plots). This is the obvious place to anchor a
  CSV snapshot.

### Why "minimal" is the point

The issue is from 2024 ("This needn't be the core library though") and
spawned two distinct user comments — one nominating ppx_minidebug
numerical logging, the other naming W&B/Ketrew/labml/Aim. The user has
not chosen between these directions. Building any one of them as the
"OCANNL way" risks contradicting the user's later choice. A minimal CSV
sink is neutral: it composes with all of them (W&B can ingest CSV;
ppx_minidebug numerical logging, when it lands, can write CSV-shaped
output; Aim/labml have CSV importers), and it gives v1.0 a concrete
artefact to show off without committing the project to a long-term
integration surface.

## Approach (suggested — agents may deviate)

*Suggested approach — agents may deviate if they find a better path.*

A reasonable shape for the implementation is:

1. **`lib/metrics.ml` (+ `lib/metrics.mli`).** Define an opaque `t`, a
   `create : ?destination:[`Csv of string | `Channel of Out_channel.t] ->
   unit -> t` constructor (or simpler — `create_csv : string -> t`), a
   `log : t -> step:int -> name:string -> float -> unit`, and a
   `close : t -> unit`. Long-form CSV (`step,name,value`) is the path of
   least resistance and trivially extends to new series without schema
   changes. Buffer writes per-record and flush on `close`.
2. **Re-export** as `Ocannl.Metrics` in `lib/ocannl.ml`.
3. **Retrofit `moons_demo.ml`.** Replace the three `ref []` accumulators
   with `Metrics.log` calls, keeping a parallel ref list only if the
   PrintBox plot still needs one (acceptable — the goal is that the CSV
   becomes a parallel artefact, not that ref lists vanish).
4. **Snapshot.** Either include a `head -n 5` of the CSV in the demo's
   `.expected` (via a small `Stdio.printf` of the first few lines) or add
   a tiny `metrics_csv_test.ml` that runs a 3-step toy loop and dumps the
   full CSV to stdout. The latter is preferable because it isolates the
   metrics module from training-loop nondeterminism.
5. **Decide `example_train_result`.** Recommended: delete it. If the
   implementer wants to keep it, it should become a return type
   constructed from a `Metrics.t` snapshot (e.g. a `Metrics.summary`
   function) rather than remain an isolated record. The choice should be
   documented in the PR description.
6. **Doc.** Write `docs/experiment_tracking.md` last, after the API
   shape has settled, so the narrative reflects what was actually built.
   Aim for ≲ 200 lines.

The implementer should *not*:

- Add a W&B HTTP client. Out of scope.
- Wire up ppx_minidebug numerical logging. Out of scope; mention it in
  the doc as a future direction.
- Build a real-time dashboard, web UI, or background uploader. Out of
  scope.
- Generalise the metrics module to support arbitrary value types
  (vectors, histograms, images). `float` is enough for v1.0. Note in
  the doc that this is a deliberate limitation.

## Scope

**In scope:**

- New `Metrics` module with a small CSV-backed API.
- Retrofit of one existing training demo (`moons_demo.ml` recommended).
- One `.expected` snapshot covering CSV output.
- Narrative doc `docs/experiment_tracking.md`.
- Decision on `Train.example_train_result` (delete or wire up).
- Stub follow-up issues for W&B and ppx_minidebug numerical logging.

**Out of scope (deferred to follow-ups):**

- W&B / Aim / labml / Ketrew integrations (REST clients, run-grouping,
  authentication, run resumption, …).
- ppx_minidebug numerical logging — discussed in the doc only.
- Live/streaming dashboards.
- Multi-process / distributed-training-aware metric aggregation.
- Tensor-valued metrics (gradient histograms, weight distributions,
  attention maps).
- Retrofitting demos other than the chosen one.

**Dependencies:** none. This task does not depend on any other open
proposal and does not block downstream v1.0 work — the CSV is purely
additive.

**Effort:** the elaboration claimed "medium (3-5 days)". The narrowed
scope here is closer to **small/medium**: ~150 lines of metrics code,
~30 lines of demo retrofit, ~150 lines of doc, one snapshot. A
focused implementer should finish in 1-2 days.

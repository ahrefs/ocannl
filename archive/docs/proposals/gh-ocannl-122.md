# Experiment tracking story

Issue: [#122](https://github.com/ahrefs/ocannl/issues/122)
ROADMAP: v1.0 completeness

## Current state

Training examples print metrics ad hoc; `moons_demo` additionally retains
lists for terminal plots. OCANNL already exposes `Context.get_used_memory`, the
`Ir.Alloc_census` counters, and `ppx_minidebug` for computation debugging, but
it has no run-level metrics format or guidance for external trackers.

## Goal

Give users a small, honest answer to “how do I record and inspect a run?”
without turning the core library into a tracking service.

## Direction

1. Document a recommended baseline: emit long-form
   `step,name,value` records, keep configuration and run metadata beside them,
   and analyze or import the result with the user's tool of choice.
2. Add one deterministic example or tiny utility that writes such records and
   still feeds the same values to `PrintBox_utils.plot`.
3. Explain how to include a memory metric, and say which question each one
   answers. `Ir.Alloc_census.peak_pool_bytes`, bracketed by `reset_peak`, is a
   high-water mark over the pools recorded at the shared allocator seam:
   process-global and summed across devices, excluding the reserved merge-buffer
   slab, the loaded code modules and host `Ndarray` arrays. So it is an
   allocator pool footprint, reproducible across runs and comparable between
   them -- not a device-memory total and not a per-device figure
   (gh-ocannl-1006). Actual device occupancy stays with the backend's
   `Context.get_used_memory`, recorded with the caveat that it is a current
   gauge which on `metal` and `cc` decrements from a GC finalizer, so when it is
   read matters. Distinguish both from `ppx_minidebug` traces.
4. Mention external systems as consumers of the portable data, not as
   supported integrations.

A public `Metrics` module is optional. Add it only if at least two examples can
share a genuinely useful API; otherwise a documented CSV sink in an example
is the smaller and more truthful deliverable. Do not couple this task to live
dashboards, W&B authentication, or cleanup of unrelated dead types.

## Completion criteria

- `docs/experiment_tracking.md` describes the baseline workflow, run metadata,
  device metrics, debugging distinction, and extension points.
- A test produces and checks a small deterministic metrics artifact.
- At least one training example demonstrates teeing a value to both the
  artifact and an existing terminal plot or summary.
- The output format is documented and stable enough to import with common
  data tools.

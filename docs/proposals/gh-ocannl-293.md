# Umbrella: Implement sharding and slicing with minimal copying

GitHub issue: [ahrefs/ocannl#293](https://github.com/ahrefs/ocannl/issues/293)
**Task**: gh-ocannl-293 (container, `leaf: false`)
**Date**: 2026-06-12
**Status**: Open umbrella — tracks three split subtasks; no implementation work happens under this id directly.

## Context

Issue #293 (2024-08-22, label `important`, title-only) asks for sharding and slicing with
minimal copying. During the 2026-04 proposal phase it was split into three weakly-coupled
subtasks, because the original three-phase sketch (strided/offset views → sharding
primitives → training integration) conflated concerns with very different prerequisites,
and because the multi-streaming cleanup
([#341](https://github.com/ahrefs/ocannl/issues/341), closed completed) invalidated the
original Phase 2 design assumptions.

## Subtasks

| Subtask | Proposal | Scope | Status (2026-06-12) |
|---------|----------|-------|---------------------|
| 293a `task-e4003e5f` | [task-e4003e5f.md](task-e4003e5f.md) | Slice/sub-tensor as alias view: convert `Fetch.Slice` from a materializing copy loop to buffer aliasing | **Blocked on [#344](https://github.com/ahrefs/ocannl/issues/344)** (universal pool allocator, still open, milestone v0.7) |
| 293b `task-a2c331e9` | [task-a2c331e9.md](task-a2c331e9.md) | Re-elaborate sharding primitives post-#341; elaboration-only verdict task | Ready; recommendation drafted (per-shard backend contexts) awaiting user verdict |
| 293c `task-2445dd1c` | [task-2445dd1c.md](task-2445dd1c.md) | Training-loop integration of sharding (data/pipeline parallelism) | Blocked on 293b's verdict |

Natural sequencing: 293a → 293b verdict → 293c, though 293b (a research/decision task)
can proceed independently of 293a.

## Milestone discrepancy (needs a decision)

- The GitHub issue still carries milestone **v0.8** (due 2026-02-28, already past).
- `ROADMAP.md` lists "Sharding and slicing with minimal copying (#293)" under
  **v1.0 — End of October 2026** ("Feature completeness").
- 293a additionally depends on #344, which is **v0.7** work still open.

The ROADMAP placement is the operative one; the GH issue milestone should be moved to
v1.0 to match (or the ROADMAP revised, but v0.8's actual scope is tiling/megakernel/Metal
optimizations, so the slip looks intentional).

## What changed since the original issue sketch

- **#341 removed implicit cross-stream coherence**, not multi-stream support: devices
  still hold multiple streams (`backend_intf.ml` `device_ref.streams`), `new_stream` is
  in the backend interface, and the per-stream merge buffer with
  `device_to_device ~into_merge_buffer:Copy` survived. What is gone is automatic
  "stream B waits for stream A's write" tracking, `Shared_cross_streams` sharing, and
  the `round_robin` training driver. Sharding must therefore be redesigned around
  *explicit* per-shard contexts with host-orchestrated synchronization — see
  [task-a2c331e9.md](task-a2c331e9.md).
- **`device_to_device` now returns a transfer routine** (merge-buffer static
  verification work, 2026-05): transfers compose with compute routines instead of being
  fire-and-forget scheduling calls. This is friendlier to a sharding design — gather and
  grad-sync steps become routines in the same execution model as everything else.
- **#344 (pool allocator) remains open**: `buffer` is still
  `{ ptr : 'buffer_ptr; size_in_bytes : int }` with no offset, so cheap alias views
  (293a) stay blocked on it.

## Completion criterion

This umbrella completes when all three subtasks land or close (293b may legitimately
close 293c as not-planned; see its decision tree).

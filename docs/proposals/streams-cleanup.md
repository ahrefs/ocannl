# Cleanup of Deprecated Streams Infrastructure

## Motivation

OCANNL originally supported "multi-streaming" -- using multiple GPU streams or CPU threads as parallel execution contexts on the same device for data-parallel training. This was removed as a user-facing feature in commit `77b7d5a9` (2025-09-08), and GitHub issue [#341](https://github.com/ahrefs/ocannl/issues/341) was closed as "Not planned." The Context API now serves as the simplified backend interface (see CHANGES.md).

However, the **internal backend infrastructure** for multi-streaming still permeates the codebase: type definitions, cross-stream synchronization logic, per-stream tracking hashtables, and round-robin scheduling functions. This dead code adds complexity to every backend implementation and makes future changes harder. Cleaning it up is a v0.7.0 milestone item (ROADMAP.md lines 74-75).

Related: [#320](https://github.com/ahrefs/ocannl/issues/320) (cross-stream sharing) is likely obsoleted by this cleanup.

## Current State

The stream infrastructure spans 8+ files with ~478 total occurrences of "stream" across `.ml` files. Key areas:

**Types in `arrayjit/lib/backend_intf.ml`:**
- `config` type (line 39): `Only_devices_parallel | For_parallel_copying | Most_parallel_streams` -- selects how many streams a backend creates. Only `For_parallel_copying` is used as the default in `fresh_backend`.
- `device_ref` type (lines 91-104): Carries 6 stream-tracking hashtables (`cross_stream_candidates`, `owner_stream`, `shared_writer_streams`, `host_reading_streams`, `host_writing_streams`, `streams`).
- `stream_ref` type (lines 106-116): Per-stream state including `stream_id`, `updating_for` event tracking, `reader_streams` cross-stream coordination.
- Duplicated `device` and `stream` type aliases (lines 122-181) with extensive doc comments about cross-stream sharing semantics.
- `merge_buffer_use` type (line 42): `Streaming_for` variant is stream-specific.
- `suggested_num_streams` in the `Backend_device_common` signature (line 301).

**Cross-stream synchronization in `arrayjit/lib/backends.ml`:**
- `wait_for_all` (line 28): Waits for events from other streams before proceeding.
- `update_writer_event` (lines 60-90): Records which streams have written to which nodes, with cross-stream event tracking via `shared_writer_streams`.
- `alloc_if_needed` (lines 516-559): Cross-stream candidate tracking, owner stream assignment, `Per_stream` vs `Shared_cross_streams` allocation decisions.
- `finalize` (line 627): Checks `cross_stream_candidates` before freeing.
- `fresh_backend` (line 643): Accepts `config` parameter.

**Sharing type in `arrayjit/lib/tnode.ml`:**
- `sharing` type (lines 25-33): `Unset | Per_stream | Shared_cross_streams`.
- Cross-stream algorithm comment (lines 12-24).
- Functions: `known_shared_cross_streams`, `known_non_cross_stream`, `potentially_cross_stream`.

**Backend implementations:**
- `cuda_backend.ml` (line 246): `suggested_num_streams` uses `config` to compute stream counts from GPU properties.
- `metal_backend.ml` (line 247): `suggested_num_streams` always returns 1 regardless of config.
- `schedulers.ml` (lines 192, 277): Multicore returns `recommended_domain_count() - 2`; Sync uses a mutable ref.

**Training utilities in `lib/train.ml`:**
- `round_robin` (lines 135-157) and `round_robin_dry_run` (lines 159-176): Distribute work across streams. Only used by the already-deleted `moons_demo_parallel.ml`.

**Context API in `arrayjit/lib/context.ml`:**
- Already creates a single stream per device (line 52: `Backend.new_stream device`).
- Names the old `Backends` module as `Backends_deprecated` (line 6), signaling intent.

## Proposed Change

Remove all multi-streaming infrastructure so that each device has exactly one execution context (runner), not an array of streams. After cleanup:

- **`config` type**: Remove entirely. `fresh_backend` no longer takes a `~config` parameter.
- **`device_ref`**: Keep `dev`, `ordinal`, `device_id`. Remove all 6 stream-tracking hashtables. Add a single `runner` field (or keep the stream concept but with a single mandatory instance).
- **`stream_ref`**: Simplify or merge into device. Keep `device`, `runner`, `merge_buffer`, `allocated_buffer`. Remove `stream_id`, `updating_for`, `updating_for_merge_buffer`, `reader_streams`.
- **`sharing` type in `tnode.ml`**: Remove `Per_stream` and `Shared_cross_streams` variants, the algorithm comment, and the three query functions. Arrays are simply per-device.
- **`backends.ml` synchronization**: Remove `wait_for_all` cross-stream waiting, simplify `update_writer_event` to single-context tracking, simplify `alloc_if_needed` to remove cross-stream candidate logic.
- **`merge_buffer_use`**: Remove `Streaming_for` variant.
- **`suggested_num_streams`**: Remove from all backend signatures and implementations.
- **`round_robin` / `round_robin_dry_run`**: Remove from `train.ml`.
- **Backend implementations**: Simplify CUDA, Metal, and CC backends to single-stream-per-device. Remove per-stream creation and tracking.
- **`event` type parameter**: Likely survives -- still useful for async GPU operations on a single stream. But cross-stream event coordination (tracking which streams finished writing) should be removed.
- **All tests pass** after cleanup.

### Acceptance Criteria

1. The `config` type and `suggested_num_streams` are gone.
2. `device_ref` has no stream-tracking hashtables.
3. `stream_ref` has no cross-stream coordination fields.
4. `sharing` type in `tnode.ml` has no `Per_stream`/`Shared_cross_streams` variants.
5. Cross-stream synchronization logic in `backends.ml` is removed.
6. `round_robin` and `round_robin_dry_run` are removed from `train.ml`.
7. CUDA, Metal, and CC backends are simplified to single-stream-per-device.
8. All existing tests pass.

### Edge Cases

- **Async GPU operations**: Even with a single stream, CUDA and Metal operations can be asynchronous. The cleanup must preserve async operation capability while removing multi-stream coordination.
- **Multi-device setups**: Device-to-device copies still need coordination at the device level. Some tracking currently on `device_ref` (like `host_reading_streams`, `host_writing_streams`) may need to survive in simplified form as device-level (not stream-level) tracking.
- **`event` type parameter cascade**: If the `event` type parameter is removed, it cascades through many signatures. It should likely be kept for async operations.
- **External consumers**: Any code depending on `device_ref` or `stream_ref` field names will break. The Context API (`context.ml`) already wraps these, so downstream impact should be contained.

## Scope

**In scope:**
- All files listed above: `backend_intf.ml`, `backends.ml`, `tnode.ml`, `train.ml`, `cuda_backend.ml`, `metal_backend.ml`, `schedulers.ml`, `context.ml`, `lowered_backend_missing.ml`
- Removing documentation comments about cross-stream sharing
- Updating CHANGES.md with the cleanup

**Out of scope:**
- Redesigning the Context API beyond stream removal
- Performance optimizations to the single-stream model
- Changes to the `event` mechanism for async operations (keep as-is)
- The "hosted tensor" migration (separate task: watch-ocannl-README-md-9e031df7)

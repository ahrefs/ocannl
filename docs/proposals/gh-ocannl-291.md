# Audit manually specifying sharing in tensor node memory modes

GitHub issue: [ahrefs/ocannl#291](https://github.com/ahrefs/ocannl/issues/291)

## Goal

Audit the correctness of manually specifying the `sharing` property (`Per_stream` vs `Shared_cross_streams`) on tensor node memory modes, identify bugs or inconsistencies between manual specification and the inference/allocation machinery, and either fix issues or document limitations. Since multi-streaming was removed (#341), the "shared cross-stream but updated from multiple streams" scenario from the original issue is largely moot -- the audit focuses on whether manual sharing specification works correctly in the current single-stream-per-device context and whether dead sharing code should be cleaned up.

## Acceptance Criteria

- [ ] **Manual sharing specification paths verified**: Confirm that calling `Tn.update_memory_mode` with `On_device Shared_cross_streams` or `On_device Per_stream` before compilation correctly influences `alloc_if_needed` allocation behavior in `backends.ml`.
- [ ] **update_memory_sharing consistency audit**: Verify all case arms in `update_memory_sharing` (tnode.ml lines 357-388) are reachable and correct; identify any arms that are dead code after multi-streaming removal.
- [ ] **alloc_if_needed sharing logic audit**: Verify the three-way branch in `alloc_if_needed` (backends.ml lines 520-559) -- read-only/constant path, `known_shared_cross_streams` writable path, and per-stream fallback -- handles all manual specification cases correctly.
- [ ] **Conflict detection audit**: Verify that manually specifying sharing as `Shared_cross_streams` on a node that is then written to by different streams produces a clear error (not silent corruption).
- [ ] **Dead sharing code identified**: List sharing-related code that is unreachable after multi-streaming removal (cross-stream candidates, owner_stream tracking, shared_writer_streams synchronization in `backends.ml`) and file a cleanup issue if substantial.
- [ ] **Test coverage assessed**: Determine whether existing tests exercise manual sharing specification; if not, add a minimal test or document why testing is impractical in the single-stream context.
- [ ] **Documentation of current behavior**: Update code comments or docs to clarify the post-multi-streaming-removal semantics of `sharing` -- what `Shared_cross_streams` means when there is only one stream per device.

## Context

### Memory mode and sharing types (tnode.ml)

The `sharing` type has three variants:
- `Unset` -- not yet determined, will be inferred as `Per_stream` or `Shared_cross_streams`
- `Per_stream` -- separate buffer per stream (each stream gets its own allocation)
- `Shared_cross_streams` -- single buffer per device, reused across streams

The `sharing` property appears embedded in:
- `On_device of sharing` -- device-resident tensor with specified sharing
- `Hosted (Changed_on_devices of sharing)` -- hosted tensor whose device copy has specified sharing

Users can manually set sharing via:
- `Tn.update_memory_mode tn (On_device Shared_cross_streams) provenance` -- directly on tnode
- `Tn.update_memory_sharing tn Shared_cross_streams provenance` -- updates sharing while preserving the memory mode category
- `Train.set_materialized` -- sets `Materialized` (sharing-agnostic, resolved later)
- `Train.set_on_host` / `Train.set_hosted` -- sets hosted modes (sharing resolved via `Changed_on_devices Unset`)

### Sharing inference in alloc_if_needed (backends.ml lines 495-559)

The allocation logic in `alloc_if_needed` determines sharing at link time:
1. **Read-only/constant nodes**: If not `known_non_cross_stream`, uses `cross_stream_candidates` hashtable to share a single buffer; calls `update_memory_sharing Shared_cross_streams 39`.
2. **Writable + `known_shared_cross_streams`**: Requires an owner stream; looks up shared buffer from `cross_stream_candidates`; errors if written from multiple streams.
3. **Writable + not shared**: Falls through to `update_memory_sharing Per_stream 410` and allocates a fresh buffer.

Key concern from the original issue: If a user manually marks a node as `Shared_cross_streams` but the node is writable and appears in multiple streams, the code at line 543-550 raises a `User_error`. This is the intended safety check, but since multi-streaming is removed, the scenario cannot arise in practice.

### Synchronization infrastructure (backends.ml)

The `shared_writer_streams`, `host_reading_streams`, and related event-tracking machinery in `Add_buffer_retrieval_and_syncing` was designed for multi-stream synchronization. Key functions:
- `wait_for_all` / `wait_for_ready` -- synchronize across streams before transfers
- `to_host` -- checks `potentially_cross_stream` before host transfer
- `update_writer_event` -- tracks writer events for cross-stream nodes
- `sync_routine.pre` -- waits for cross-stream writer events before routine execution

With single-stream-per-device, most of this synchronization code is technically unnecessary but harmless. The `potentially_cross_stream` predicate still fires for nodes without explicit `Per_stream` marking.

### Related tasks and issues

- **gh-ocannl-333**: Remove hosted memory mode entirely (would subsume much of the sharing complexity)
- **gh-ocannl-341**: Multi-streaming removed (closed, "Not planned") -- makes cross-stream sharing moot
- **gh-ocannl-296**: Low-level audit (adjacent concern, memory mode inference in `cleanup_virtual_llc`)

### Code pointers

- **Sharing types**: `arrayjit/lib/tnode.ml` lines 25-34
- **Memory mode types**: `arrayjit/lib/tnode.ml` lines 47-65
- **update_memory_mode**: `arrayjit/lib/tnode.ml` lines 306-352 (~46 case arms)
- **update_memory_sharing**: `arrayjit/lib/tnode.ml` lines 357-388 (~12 case arms)
- **Sharing predicates**: `known_shared_cross_streams`, `known_non_cross_stream`, `potentially_cross_stream` (tnode.ml lines 285-299)
- **alloc_if_needed**: `arrayjit/lib/backends.ml` lines 495-559
- **Synchronization**: `arrayjit/lib/backends.ml` lines 38-90 (`wait_for_all`, `to_host`, `update_writer_event`)
- **Device ref type**: `arrayjit/lib/backend_intf.ml` lines 91-100 (`cross_stream_candidates`, `owner_stream`, `shared_writer_streams`)
- **Metal backend sharing**: `arrayjit/lib/metal_backend.ml` line 764 (uses `cross_stream_candidates` for constant buffers)
- **Train helpers**: `lib/train.ml` lines 63-182 (`set_on_host`, `set_materialized`, `set_hosted`, `set_virtual`)

## Approach

### Phase 1: Static analysis of sharing code paths

1. Trace all call sites of `update_memory_sharing` (only 2: backends.ml lines 541 and 557) and `update_memory_mode` with sharing-carrying modes to build a complete map of how sharing gets set.
2. For each case arm in `update_memory_sharing`, determine: (a) is it reachable in single-stream context? (b) is the behavior correct?
3. Analyze the `alloc_if_needed` three-way branch for all combinations of manual sharing specification vs inference.

### Phase 2: Identify dead code and simplification opportunities

1. Catalog sharing-related infrastructure that is unreachable with single-stream-per-device: `owner_stream` tracking, `shared_writer_streams` synchronization, multi-stream `wait_for_all` patterns.
2. Determine what simplifications are safe vs what should be preserved for potential future multi-stream reintroduction.
3. Consider interaction with gh-ocannl-333 (hosted mode removal) -- if that lands first, much of the sharing complexity disappears.

### Phase 3: Verify or add test coverage

1. Check whether any existing test exercises manual sharing specification (search for explicit `On_device Shared_cross_streams` or `Per_stream` in test code -- current search shows none).
2. Write a minimal test that manually sets sharing before compilation and verifies correct allocation behavior, or document why this is impractical in the current architecture.

### Phase 4: Document findings and clean up

1. Update code comments on `sharing` type and `update_memory_sharing` to reflect post-multi-streaming semantics.
2. File cleanup issue for dead sharing infrastructure if warranted.
3. Update task notes with complete audit findings.

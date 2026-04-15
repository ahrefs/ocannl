# Proposal: Remove hosted memory mode and Ndarray dependency from Tnode

GitHub issue: [ahrefs/ocannl#333](https://github.com/ahrefs/ocannl/issues/333)

## Goal

Eliminate the dual host/device memory model by removing the `array` field from `Tnode.t`, the `Hosted` variant from `memory_mode`, and the `memory_type` type entirely. All tensor data lives exclusively on devices; CPU-side value access (printing, saving, inspection) happens via on-demand device-to-host transfers through a context. The `Ndarray` module is retained as a slimmed-down utility for temporary host buffers but is no longer stored inside tensor nodes.

## Acceptance Criteria

1. **`array` field removed from `Tnode.t`**: The `array : Nd.t option Lazy.t` field (tnode.ml line 74) is deleted. No host-side copy is stored in the tensor node.

2. **`Hosted` variant and `memory_type` removed**: The `Hosted of memory_type` variant is deleted from `memory_mode`. The `memory_type` type (`Unset_hosted`, `Constant`, `Nonconstant`, `Changed_on_devices`, `Volatile`) is deleted entirely. `Materialized` collapses to mean `On_device`.

3. **`devices_not_lagging_host` removed**: The host/device sync tracking field (tnode.ml line 94) and all `prepare_read`/`prepare_write` callback machinery are removed.

4. **Context-based value access**: `get_value`, `set_value`, `get_values`, `set_values` (tnode.ml lines 810-844) are replaced with context-aware versions that perform on-demand device-to-host transfers. An optional `mutable host_cache : Nd.t option` field on `Tnode.t` allows evictable caching (mutable, not lazy, per the issue author's comment about future buffer eviction).

5. **On-demand tensor printing**: `Tensor.print` works without hosted arrays. The `[%cd "for_print" =: t_to_print]` trick creates a temporary device-to-host copy. A cache of for-print tensor nodes avoids recompilation. This is off by default for `Tensor.print`, on by default for `Train.printf`.

6. **Train.ml cleanup**: `set_on_host`, `set_hosted`, `every_non_literal_on_host` are deleted. The `?(hosted = true)` parameter is removed from `to_routine`, `init_params`, `run_once`, `forward_once`. `forward` and `grad_update` stop calling `set_hosted`.

7. **Parameter initialization works**: Parameters are initialized by writing to a temporary host buffer then copying to device via `from_host`, rather than the current pattern of writing to the hosted array.

8. **`.@` operators updated**: The `Operation.At` operators (`.@{}` and `.@{}<-`) are migrated to context-aware access. Since these appear in only ~7 call sites (operation.ml, two test files, docs), the migration scope is contained.

9. **Backend interface preserved**: `from_host` and `to_host` backend operations continue to work. `to_host` becomes the primary retrieval path. `use_host_memory` flag for Apple Silicon unified memory stays.

10. **`Ndarray` module minimized**: Functions only used through the hosted array pattern are removed. Core functions needed for temporary host buffers are retained: `create_array`, `render_array`, `retrieve_flat_values`, `set_flat_values`, precision handling.

11. **No regression in existing tests**: All tests pass after the refactoring.

## Context

### Current architecture

`Tnode.t` maintains a dual-memory model where tensor nodes can have both a host-side `Ndarray` (the `array` field) and device-side buffers managed by backends. The `Hosted of memory_type` memory mode controls synchronization between host and device copies, with five `memory_type` sub-variants driving a complex state machine in `update_memory_mode` (tnode.ml lines 306-352). (Note: `update_memory_sharing` was removed in the streams cleanup.)

Key code locations:
- **Tnode.t type**: arrayjit/lib/tnode.ml lines 73-98 (record with `array`, `devices_not_lagging_host`)
- **memory_mode/memory_type**: arrayjit/lib/tnode.ml lines 36-65
- **Value access**: arrayjit/lib/tnode.ml lines 809-844 (`get_value`, `set_value`, `get_values`, `set_values`)
- **Mode transitions**: arrayjit/lib/tnode.ml lines 306-388 (~49 `Hosted` references in tnode.ml)
- **Backend transfers**: arrayjit/lib/backends.ml (`to_host`, `from_host`)
- **Train helpers**: lib/train.ml (`set_on_host`, `set_hosted`, `every_non_literal_on_host`)
- **Low-level compilation**: arrayjit/lib/low_level.ml (2 `Hosted` references in mode assignment)
- **Ndarray module**: arrayjit/lib/ndarray.ml (~772 lines)

### Design rationale (from issue author)

The author's comments clarify the progression of thinking:
1. Initially: make everything materialized also hosted on-demand, with mutable (not lazy) caching for future eviction
2. Final position: "we will make **nothing** hosted, printing, saving, and accessing selected values will perform on-demand retrieving from the devices"
3. The printing trick `[%cd "for_print" =: t_to_print]` with a cache of for-print nodes avoids the `set_materialized` problem

### Scope boundaries

**In scope**: Removing `array` field, `Hosted` variant, `memory_type` type, host/device sync tracking, Train.ml hosted helpers, updating value access to require context, implementing on-demand printing, updating all tests and examples, minimizing Ndarray.

**Out of scope**: Full Ndarray module removal (still needed for precision-polymorphic host buffers), buffer eviction policy design (future work enabled by this change), unified memory optimization (orthogonal), `from_host`/`to_host` backend signature changes.

### Impact estimate

This is a large refactoring touching ~50+ locations across tnode.ml, low_level.ml, backends.ml, train.ml, tensor.ml, operation.ml, and test files. The API change (adding `~ctx` to value access) propagates to all downstream callers. The `update_memory_mode` state machine shrinks significantly with `Hosted` removal.

### Edge cases

- **Printing without context**: Before compilation, tensor printing shows shape/metadata only (no values). A "print context" pattern or lazy printing can address interactive use.
- **Parameter serialization**: The disabled `save_params`/`restore_params` in train.ml needs redesign around context-based device-to-host-to-disk path.
- **Virtual/Local tnodes**: These have no device buffers. Printing requires materializing first.
- **C backend**: cc_backend.ml uses `Lazy.force tn.array` for compiled C function parameter binding -- must switch to context array map.

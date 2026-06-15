# Proposal: Remove hosted memory mode and Ndarray dependency from Tnode

GitHub issue: [ahrefs/ocannl#333](https://github.com/ahrefs/ocannl/issues/333)

## Status update (2026-06-12)

- Issue #333 is OPEN, GH milestone v0.7. ROADMAP.md lists it under v0.7.0 ("Remove hosted tensor mode", part of the context-handling finalization theme); that milestone's nominal date (end Feb 2026) has slipped, but the task remains on the roadmap.
- Not started: `Tnode.t` still has the `array : Nd.t option Lazy.t` field, `Hosted of memory_type`, `memory_type`, `devices_not_lagging_host`, and the `prepare_read`/`prepare_write` machinery; `lib/train.ml` still has `set_on_host`, `set_hosted`, `every_non_literal_on_host`, and `?(hosted = true)` parameters.
- Line numbers in this proposal have drifted (corrected in place below): the `array` field is now `tnode.ml:49`, `memory_type`/`memory_mode` are lines 12-40, `devices_not_lagging_host` is line 69, value access is lines 727-761, `update_memory_mode` starts at line 261. `Hosted` reference count in `tnode.ml` is now ~35 (was ~49); `ndarray.ml` is now ~1036 lines.
- Landed since this was written, easing or reshaping parts of the plan: tensor persistence (`lib/persistence.ml`, #373) — the "parameter serialization" edge case should now route through `Persistence` rather than reviving `Train.save_params` (still commented out); deprecated multi-stream infrastructure removed from the backend layer (cross-stream automatic coherence is gone; multiple streams per device remain); `device_to_device` now returns a transfer routine with static merge-buffer verification (`backend_intf.ml:307`), and `merge_buffer_use` is `No | Copy` (Streaming_for is gone); Metal now uses private storage mode for GPU-only buffers (commit 1cf9a95b), making the `use_host_memory` unified-memory note Metal-specific to shared-mode buffers.
- `use_host_memory` lives in `arrayjit/lib/backend_impl.ml` (lines 20, 188), not `backend_intf.ml`.
- The design itself (acceptance criteria 1-11, the on-demand printing trick, the context-based value access) is not invalidated by any landed work; it remains the v0.7.0 roadmap plan.

## Goal

Eliminate the dual host/device memory model by removing the `array` field from `Tnode.t`, the `Hosted` variant from `memory_mode`, and the `memory_type` type entirely. All tensor data lives exclusively on devices; CPU-side value access (printing, saving, inspection) happens via on-demand device-to-host transfers through a context. The `Ndarray` module is retained as a slimmed-down utility for temporary host buffers but is no longer stored inside tensor nodes.

## Acceptance Criteria

1. **`array` field removed from `Tnode.t`**: The `array : Nd.t option Lazy.t` field (tnode.ml line 49) is deleted. No host-side copy is stored in the tensor node.

2. **`Hosted` variant and `memory_type` removed**: The `Hosted of memory_type` variant is deleted from `memory_mode`. The `memory_type` type (`Unset_hosted`, `Constant`, `Nonconstant`, `Changed_on_devices`, `Volatile`) is deleted entirely. `Materialized` collapses to mean `On_device`.

3. **`devices_not_lagging_host` removed**: The host/device sync tracking field (tnode.ml line 69) and all `prepare_read`/`prepare_write` callback machinery are removed.

4. **Context-based value access**: `get_value`, `set_value`, `get_values`, `set_values` (tnode.ml lines 727-761) are replaced with context-aware versions that perform on-demand device-to-host transfers. **No `host_cache` field is added to `Tnode.t` and `tnode.ml` gains no `ndarray.ml` dependency** (decision 2026-06-15, see Decisions §1): there is no persistent host cache or staging buffer — each access transfers fresh through the context. The API documents loudly that host access is expensive on non-unified-memory backends. On unified-memory backends the backend returns the stored bigarray with no copy (Decisions §2). Any host-data association that later proves necessary lives in a side table in `context.ml`, never in the tensor node.

5. **On-demand tensor printing**: `Tensor.print` works without hosted arrays and **takes an explicit context argument** (decision 2026-06-15, Decisions §3 — no ambient context registry). The `[%cd "for_print" =: t_to_print]` trick creates a temporary device-to-host copy. A cache of for-print tensor nodes avoids recompilation. This is off by default for `Tensor.print`, on by default for `Train.printf` (which already has a context at its call sites).

6. **Train.ml cleanup**: `set_on_host`, `set_hosted`, `every_non_literal_on_host` are deleted. The `?(hosted = true)` parameter is removed from `to_routine`, `init_params`, `run_once`, `forward_once`. `forward` and `grad_update` stop calling `set_hosted`.

7. **Parameter initialization works**: Parameters are initialized by writing to a temporary host buffer then copying to device via `from_host`, rather than the current pattern of writing to the hosted array.

8. **`.@` operators updated**: The `Operation.At` operators (`.@{}` and `.@{}<-`) are migrated to context-aware access. Since these appear in only ~7 call sites (operation.ml, two test files, docs), the migration scope is contained.

9. **Backend interface preserved**: `from_host` and `to_host` backend operations continue to work. `to_host` becomes the primary retrieval path. `use_host_memory` flag for Apple Silicon unified memory stays.

10. **`Ndarray` module minimized**: Functions only used through the hosted array pattern are removed. Core functions needed for temporary host buffers are retained: `create_array`, `render_array`, `retrieve_flat_values`, `set_flat_values`, precision handling.

11. **No regression in existing tests**: All tests pass after the refactoring.

## Context

### Current architecture

`Tnode.t` maintains a dual-memory model where tensor nodes can have both a host-side `Ndarray` (the `array` field) and device-side buffers managed by backends. The `Hosted of memory_type` memory mode controls synchronization between host and device copies, with five `memory_type` sub-variants driving a complex state machine in `update_memory_mode` (tnode.ml lines 261-306). (Note: `update_memory_sharing` was removed in the streams cleanup.)

Key code locations *(line numbers refreshed 2026-06-12)*:
- **Tnode.t type**: arrayjit/lib/tnode.ml lines 48-73 (record with `array`, `devices_not_lagging_host`)
- **memory_mode/memory_type**: arrayjit/lib/tnode.ml lines 12-40
- **Value access**: arrayjit/lib/tnode.ml lines 727-761 (`get_value`, `set_value`, `get_values`, `set_values`)
- **Mode transitions**: arrayjit/lib/tnode.ml lines 261-306 (~35 `Hosted` references in tnode.ml)
- **Backend transfers**: arrayjit/lib/backends.ml (`to_host`, `from_host`)
- **Train helpers**: lib/train.ml (`set_on_host`, `set_hosted`, `every_non_literal_on_host`)
- **Low-level compilation**: arrayjit/lib/low_level.ml (2 `Hosted` references in mode assignment)
- **Ndarray module**: arrayjit/lib/ndarray.ml (~1036 lines)

### Design rationale (from issue author)

The author's comments clarify the progression of thinking:
1. Initially: make everything materialized also hosted on-demand, with mutable (not lazy) caching for future eviction
2. Final position: "we will make **nothing** hosted, printing, saving, and accessing selected values will perform on-demand retrieving from the devices"
3. The printing trick `[%cd "for_print" =: t_to_print]` with a cache of for-print nodes avoids the `set_materialized` problem

### Scope boundaries

**In scope**: Removing `array` field, `Hosted` variant, `memory_type` type, host/device sync tracking, Train.ml hosted helpers, updating value access to require context, implementing on-demand printing, updating all tests and examples, minimizing Ndarray.

**Out of scope**: Full Ndarray module removal (still needed for precision-polymorphic host buffers), buffer eviction policy design (future work enabled by this change), unified memory optimization (orthogonal), `from_host`/`to_host` backend signature changes. *(Update 2026-06-12: "no signature changes" is only true of the implementation-facing API — `No_buffer_retrieval_or_syncing.from_host`/`to_host` already take an explicit `Ndarray.t`. The user-facing wrappers in `backends.ml` (`Add_buffer_retrieval_and_syncing`) pattern-match on `tn.array = lazy (Some hosted)` to source/sink host data, so they must change: they **take an `Ndarray.t` argument** supplied by the caller. There is no `host_cache` staging buffer to read/write (Decisions §1, 2026-06-15); on unified memory the backend may return the stored bigarray directly (Decisions §2).)*

### Impact estimate

This is a large refactoring touching ~50+ locations across tnode.ml, low_level.ml, backends.ml, train.ml, tensor.ml, operation.ml, and test files. *(Update 2026-06-12: add `lib/persistence.ml` to this list — `save`/`restore` call `Tn.do_read` and read/write `tn.array` directly, and `load` creates hosted tnodes via `Tn.create_from_padded` before any context exists; its API needs a `~ctx` parameter (or staged host data), not just internal edits.)* The API change (adding `~ctx` to value access) propagates to all downstream callers. The `update_memory_mode` state machine shrinks significantly with `Hosted` removal.

### Edge cases

- **Printing without context**: Before compilation, tensor printing shows shape/metadata only (no values). A "print context" pattern or lazy printing can address interactive use.
- **Parameter serialization**: The disabled `save_params`/`restore_params` in train.ml needs redesign around context-based device-to-host-to-disk path. *(Update 2026-06-12: tensor persistence landed as `lib/persistence.ml` (#373); the redesign should target `Persistence` rather than reviving the train.ml stubs.)* *(Decision 2026-06-15, Decisions §4: with no `host_cache`, `Persistence.load` takes a `Context.t` and returns a new context with the loaded tnode added — the same shape as running parameter initialization; `save`/`restore` become context-mediated.)*
- **Virtual/Local tnodes**: These have no device buffers. Printing requires materializing first.
- **C backend**: cc_backend.ml uses `Lazy.force tn.array` for compiled C function parameter binding -- must switch to context array map.

## Design review (2026-06-12)

**Verdict: sound-with-changes.** The end state (nothing hosted, context-based on-demand access) matches the issue author's final position and is the right target. The plan underspecifies three load-bearing mechanisms: the staging lifecycle for host data that exists *before* any context, the fate of the `use_host_memory` zero-copy path, and the Persistence API change. Land **before** #344.

**Recommendations:**

> **Status (2026-06-15):** Recommendations 1, 2, 3, and 5 below posed open questions that are now **resolved in the Decisions section** — read them as historical framing, not open work. **Recommendation 4 (dead-code cleanup) is the only one that remains live implementation work.**

1. **[SUPERSEDED — see Decisions §1: no staging buffer, no host cache.] Design `host_cache` as a staging buffer with an explicit lifecycle, not just a read cache.** Three flows need host data before a device buffer exists: tensor literals (`create_from_padded`/`create_with_reshape`, today `Hosted Unset_hosted`), `Persistence.load` (creates tnodes from file payloads pre-context), and AC 7 parameter init. Define one mechanism: *pending host data, uploaded at first link (replacing `alloc_if_needed`'s `will_copy_from_host` path), evictable read-cache thereafter*. Otherwise AC 7 and the persistence edge case will be solved twice, differently.
2. **[RESOLVED — see Decisions §1: no `host_cache` field; side table in `Context` only if ever needed.] Resolve the Tnode/Ndarray contradiction.** AC 4 puts `mutable host_cache : Nd.t option` on `Tnode.t`, but the proposal's title removes the Ndarray dependency from Tnode. Either accept the (slimmed) dependency and retitle, or keep `host_cache` in a side table (e.g. weak `Hashtbl.M(Tnode)` owned by an Ndarray-aware layer such as `backends.ml`/`context.ml`). The side table is cleaner layering and keeps `tnode.ml` free of `ndarray.ml`; the field is simpler. Pick one explicitly.
3. **[RESOLVED — see Decisions §2: unified-memory backends return the stored bigarray (safe); no pointer-wrap, no pinned staging.] Decide the zero-copy constants question (biggest hidden cost).** Today on cc and Metal-shared, hosted constants' device buffers *wrap the host array pointer* (`alloc_if_needed`'s `use_host_memory` branch; Metal dispatch fallback at `metal_backend.ml:791-799`). With `tn.array` gone and `host_cache` evictable, wrapping an evictable pointer is unsafe. Options: (a) pin constants' staging buffers (zero-copy preserved, but it's hosted-mode-for-constants by another name); (b) always allocate device-side + copy (simpler, but transient 2x memory and an extra copy on CPU backends where host arrays currently *are* the buffers). Whichever is chosen, on unified-memory backends `get_value`/`get_values` should read through the context's buffer pointer with no copy.
4. **[LIVE — the only open recommendation.] Add missing cleanup to AC 6:** the `automatic_host_transfers` setting, `sync_routine`'s hosted pre/post hooks (`backends.ml:240-260`), and the Metal dispatch constant-wrap fallback all become dead code; `Hosted Unset_hosted` assignment in `low_level.ml:443` needs a replacement default.
5. **[RESOLVED — see Decisions §3: `Tensor.print` takes an explicit context argument; no ambient registry.] AC 5 needs a context-plumbing decision for printing** (see below) — the `for_print` trick compiles and runs code, which requires a `Context.t`, but `Tensor.print` takes none today.

## Decisions (resolved 2026-06-15 by Łukasz)

The four open decision points above are now settled. **The unifying decision: there is no persistent staging buffer and no host cache after this refactoring.** All CPU-side value access is an on-demand device-to-host transfer mediated by an explicit context. Staging buffers may be reintroduced later in a well-motivated way; this task cleans up the design first.

1. **`host_cache` location (rec 2) → side table in `Context`, and no cache at all for now.** There is **no `host_cache` field on `Tnode.t`** and **no `ndarray.ml` dependency in `tnode.ml`** — this cleanly satisfies the proposal title ("remove the Ndarray dependency from Tnode"). If any host-data association is ever needed it belongs in a side table inside `context.ml` / `Context`, not in the tensor node. But the preferred direction is **no cache**: rather than caching, the API should **loudly document that host access is expensive on non-unified-memory systems**, so callers feel the cost instead of silently paying it through a cache.

2. **Zero-copy constants on unified memory (rec 3) → safe path: backend returns the stored bigarray.** The zero-copy optimization is handled *directly in the backends* on unified-memory systems by **returning the stored bigarray** (safe), **not** by wrapping a raw pointer in a freshly-constructed bigarray value (unsafe). No pinned staging buffer is introduced for constants.

3. **Print context (DP3) → `Tensor.print` requires an explicit context argument.** No implicit/ambient "most recent context" registry is added. An ambient "most-recent-context" registry is **rejected as bad design** — it is edge-case-unsafe (the implicit context can silently be the wrong one) and trades a clear explicit dependency for hidden global state; it is not a planned follow-up. Value access is always through an explicit context. `Train.printf` already has a context at its call sites. (If print ergonomics ever need improvement, it will be approached some other way, decided against real user code — not via an ambient registry.)

4. **Persistence (DP4) → `load` takes a context and returns a new context.** With no `host_cache`, `Persistence.load` takes a `Context.t` and **returns a new context** with the loaded tnode added — structurally the same flow as running parameter initialization. (`save`/`restore` likewise become context-mediated.)

These decisions **supersede recommendation 1** (the "design `host_cache` as a staging buffer with an explicit lifecycle" recommendation): no such persistent staging lifecycle is built. AC 4 below is updated accordingly (no `host_cache` field).

**Ordering note:** land #333 before #344. Removing `Hosted` collapses Metal's storage-mode segregation to "private pools + small shared staging" and deletes the `use_host_memory` pointer-wrapping entries — exactly the special cases that would otherwise force a `Wrapped of buffer_ptr` escape hatch into #344's `buffer_offset` type, only to be ripped out again.

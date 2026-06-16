# Proposal: Implement a Universal Pool Allocator across backends

GitHub issue: [ahrefs/ocannl#344](https://github.com/ahrefs/ocannl/issues/344)

## Status update (2026-06-12)

- Issue is OPEN. The GitHub milestone reads v0.7, but ROADMAP.md (the authority) lists #344 under v0.7.2 "Memory management" (was due mid-April 2026, now past due).
- Not started: `ctx_arrays` and `buffer_ptr` still exist unchanged (`backend_intf.ml:7-8`), no pool allocator module exists, and the setting is still `big_models` (`utils.ml:30`).
- Rename landed (#356, commit 9262ab44): kernel params are now `kparams` and `Kparam_ptr` (formerly `params`/`Param_ptr`); references below have been updated.
- `device_to_device` now *returns* a transfer routine, with static merge-buffer verification (commit 1162646d; `backend_intf.ml:307`), and `merge_buffer_use` is now `No | Copy` (`Streaming_for` is gone). AC 9 must account for merge-buffer transfers being compiled routines.
- Deprecated multi-stream infrastructure was removed from the backend layer (commit 272c0880) and cross-stream automatic coherence was dropped in the #341 cleanup; multiple streams per device remain, so per-stream vs per-device pool ownership is still a live design question.
- Metal private storage mode for GPU-only buffers landed (commit 1cf9a95b) and merge buffers are now memory-mode aware (399177b0): `alloc_buffer`/`alloc_array` now take a `?mode` argument and Metal selects storage mode per tnode via `storage_mode_for_memory_mode` (`metal_backend.ml:75-90`). Pools must therefore be segregated by storage mode (private vs shared), not assigned one mode arbitrarily — this refines the gh-ocannl-320 note below.
- Tensor persistence (`lib/persistence.ml`, #373) landed; it restores into *hosted* buffers, orthogonal to device pooling.
- Cited line numbers drifted; refreshed in place below as of 2026-06-12. The core design remains valid and is still the prerequisite for the alias-view work (task-e4003e5f) noted under "Relationship to other work".

## Status update (2026-06-16): scope split — foundation extracted

A design deep-dive re-sequenced this work. The behavior-preserving **foundation** — folding
`stream` into `device`, replacing `buffer_ptr` with `buffer_loc = { pool_id; offset }`, moving
the backend pointer into a device-private `pool_id → 'base` table (dropping the `'buffer_ptr`
type parameter from the shared types), and introducing a shared **allocator seam** with a
trivial one-pool-per-tnode policy — is extracted into a new proposal:
[backend-buffer-addressing.md](backend-buffer-addressing.md). **#344 now depends on it.**

This proposal is **trimmed to the pooling capability**: the bump-arena allocation policy and the
Metal 31-binding fix, landing as a *policy swap behind the seam* that
backend-buffer-addressing.md establishes. Key reframings from the deep-dive:

- **The contract type and rename moved out.** AC 2 (`buffer_offset` type — now `buffer_loc`) and
  AC 3 (`ctx_arrays → ctx_buffers`) are realized in backend-buffer-addressing.md. AC 1 splits:
  the seam + trivial policy land there; only the **bump-arena policy body** stays here.
- **Bump-only scope, no reuse (parity with today).** The allocator is a non-reuse bump
  allocator: peak pool size = Σ(all materialized tnodes in a context), the same as today's
  per-tnode allocation. This is *not* a memory-saving change. Liveness-based offset reuse and
  eviction (the "thick allocator") are deferred — the forcing function is "Σ-tnodes exceeds
  device memory while the working set would fit."
- **Lifecycle collapses under bump + context-tied lifetimes.** No free-list, no slab
  refcounting, no runtime exhaustion *policy* beyond "current slab full → append another." The
  pool classes are: per-**context** working arena (freed at `finalize`, the only release point
  under bump; nested via the parent-context chain) + per-**device** constant pool (dedup across
  contexts) + merge/staging kept out of pools. Note this is per-*context*, not per-*stream*:
  with one compute stream per device, a per-stream working pool would never free until device
  teardown.
- **Codegen change is Metal-only (AC 5 over-scoped).** Only Metal has a binding limit. CUDA/C
  keep per-tnode pointer kernel params computed host-side as `pool_base + offset` at dispatch;
  their generated `.c`/`.cu` stays byte-identical. Only Metal emits in-shader `pool_base +
  offset`, binds pools once, and passes offsets as runtime args (`set_bytes` / offsets-table
  buffer — never source constants, since `code` objects are reused across contexts).
- **Deferred and decoupled.** AC 7 (`large_models` / uint64): cap each pool at ≤4 GB, keep
  uint32 offsets, open another pool past that; only a single >4 GB tnode needs uint64 — defer
  it, error clearly meanwhile. AC 9: keep merge buffers **out** of pools (pointer stability),
  not "integrated."
- **Retained Metal design wisdom (still in scope here):** untracked hazard mode for pool
  buffers, storage-mode pool segregation (private vs shared), offsets-as-runtime-args.

The acceptance criteria and design review below are kept for that retained capability; read AC
1/2/3 and recommendation framing through the lens of this split.

## Goal

Replace per-tensor-node individual buffer allocations with a pool allocator that consolidates tensors into a small number of large pool buffers shared across all backends (Metal, CUDA, C). Each tensor is addressed as a `(pool_id, byte_offset)` pair within a pool buffer, rather than holding its own device pointer. This solves Metal's ~31 argument buffer binding limit, reduces allocation overhead, improves memory locality, and provides a natural place to introduce configurable 32-bit vs 64-bit index arithmetic via the `large_models` setting.

## Acceptance Criteria

1. **Pool allocator module**: A new `Pool_allocator` module (or extension of `backend_intf.ml`) provides a backend-agnostic pool allocation interface. Each pool has a backend-specific base pointer, a total size, and a bump/free-list allocator that returns `{ pool_id; offset; size_in_bytes }` records with configurable alignment.

2. **`buffer_ptr` replaced by `buffer_offset`**: The `'buffer_ptr buffer` type in `backend_intf.ml` (line 7) is replaced by a `buffer_offset` record containing `pool_id : int`, `offset : int`, and `size_in_bytes : int`. The per-tnode map `ctx_arrays` (line 8) maps tnodes to `buffer_offset` values instead of raw backend pointers.

3. **`ctx_arrays` renamed to `ctx_buffers`**: All occurrences of `ctx_arrays` across `backend_intf.ml`, `backend_impl.ml`, `metal_backend.ml`, `cuda_backend.ml`, `cc_backend.ml`, and all callers are renamed to `ctx_buffers`.

4. **Pool-based allocation in all backends**: Each backend's `Alloc_buffer` implementation allocates from pools rather than making individual `mem_alloc` / `Buffer.on_device` / `allocate_n` calls:
   - **Metal** (`metal_backend.ml` lines 97-130): `Me.Buffer.on_device` creates pool buffers, not per-tnode buffers. *(Update 2026-06-12: allocation is now memory-mode aware via `resource_options_for_mode`; pools must be segregated by storage mode.)*
   - **CUDA** (`cuda_backend.ml` lines 58-75): `Cu.Deviceptr.mem_alloc` creates pool buffers.
   - **C** (`backend_impl.ml` lines 44-72): `Ctypes.allocate_n` creates pool buffers.

5. **Code generation emits pool base + offset access**: `c_syntax.ml` parameter generation (lines 847-881) emits pool base pointers as kernel arguments and offset constants (or additional arguments) instead of per-tnode pointer arguments. Generated C/Metal/CUDA code casts `(pool_base + offset)` to the appropriate typed pointer for each tensor access.

6. **Metal binding count reduced**: Metal kernel dispatch (`metal_backend.ml` lines 785-815) binds O(num_pools) buffers via `set_buffer` instead of O(num_tnodes). Offsets are passed as kernel constants or a small argument buffer. This keeps total bindings well within Metal's 31-slot limit for models with hundreds of tensor nodes. *(Update 2026-06-12: offsets cannot be source-level "kernel constants" — compilation (`compile`/`compile_batch`) happens before linking and allocation, and `code`/`code_batch` objects are reusable across contexts, so offsets are only known at link time. They must be runtime arguments: `set_bytes` per offset, or a single small offsets-table buffer. Link-time source specialization would forfeit code reuse.)*

7. **`large_models` setting controls index width**: The existing `big_models` setting in `Utils.settings` is renamed (or aliased) to `large_models`. When `large_models = true`, pool offsets and index arithmetic use `uint64_t` (supporting pools > 4 GB). When `false` (default), `uint32_t` is used. This affects `arg_int_prefix` and `loop_index_type` in `c_syntax.ml` (lines 94-95) and offset-related types in generated code.

8. **Pool lifecycle management**: Pools are created at device/stream initialization with a configurable initial size. When a pool is exhausted, a new pool is allocated (pools are never resized, since that would invalidate offsets). Pools are freed when the owning device/stream is finalized. The `allocated_buffer` reuse pattern in `stream_ref` (line 108 of `backend_intf.ml`) is replaced by pool-level management.

9. **Merge buffer integration**: The `merge_buffer` field on `stream_ref` (`backend_intf.ml` line 106) works with the pool allocator. Merge buffers can be allocated from a dedicated temporary pool or from the main pool with fast reclaim. *(Update 2026-06-12: `device_to_device` now returns a compiled transfer routine with static merge-buffer verification, and `merge_buffer_use` is `No | Copy` — pool integration must keep merge-buffer pointers stable across routine invocations.)*

10. **No regression in existing tests**: All existing tests pass after the refactoring.

## Context

### Why pool allocation is necessary

Metal compute shaders have a hard limit of ~31 argument buffer bindings per compute command encoder. The current architecture binds one Metal buffer per tensor node (`metal_backend.ml` line 790: `Me.ComputeCommandEncoder.set_buffer encoder ~index buffer`), which fails for models with more than ~30 materialized tensors in a single kernel. A pool allocator consolidates all tensors into 1-3 large pool buffers, each bound once, with individual tensors addressed by offset arithmetic within the shader.

### Current allocation architecture

Each backend allocates buffers independently per tensor node. The `Alloc_buffer` module type (`backend_intf.ml`) defines `alloc_buffer`, `alloc_array`, `alloc_zeros`, and `free_buffer`; `alloc_buffer`/`alloc_array` now also take a `?mode` (memory mode) argument. The `ctx_arrays` field (`backend_intf.ml` line 157) in the context type maps each tnode to a backend-specific `buffer_ptr`. At link time, each `Kparam_ptr tn` (renamed from `Param_ptr` in #356) in the kernel parameter list causes a separate buffer binding.

Key code paths affected:
- **Parameter list construction**: `c_syntax.ml` lines 847-862 -- each in-context tnode becomes a `Kparam_ptr tn` entry in `kparams`
- **Metal dispatch**: `metal_backend.ml` lines 785-815 -- iterates `Kparam_ptr` params, calls `set_buffer` per tnode
- **CUDA dispatch**: `cuda_backend.ml` line 940 -- iterates `Kparam_ptr` params, pushes device pointers as kernel args
- **C dispatch**: `cc_backend.ml` line 402 -- iterates `Kparam_ptr` params, passes pointers to compiled function
- **Buffer type**: `backend_intf.ml` line 7 -- `type 'buffer_ptr buffer = { ptr : 'buffer_ptr; size_in_bytes : int }`

### Alignment requirements

Pool sub-allocations must respect backend-specific alignment:
- **Metal**: 256 bytes (page alignment preferred for private storage mode)
- **CUDA**: 256 bytes (memory coalescing alignment)
- **C/host**: 64 bytes (cache line alignment)

### Relationship to other work

- **gh-ocannl-333 (remove hosted memory)**: Pool allocation changes buffer management; synergistic with removing the host-side `array` field since device buffers become the sole storage.
- **gh-ocannl-320 (Metal private mode)**: Entire pools get one storage mode, simplifying private mode adoption. *(Update 2026-06-12: private storage mode for GPU-only buffers has since landed — commit `1cf9a95b` — so pools should default to private mode for non-hosted tnodes.)*
- **gh-ocannl-340 (Local_scope init)**: Local tnodes with short lifetimes benefit from arena-style allocation within pools.
- **task-e4003e5f (slice as alias view, subtask 293a of #293)**: blocked on this work. It
  will enter *alias* entries into `ctx_buffers` — a tnode resolving to another tnode's
  `(pool_id, offset + delta)` without owning storage. **Design constraint added
  2026-06-12**: the pool reclaim/free pass must be alias-aware — never reclaim a slab
  while alias entries reference it, and skip alias entries themselves when freeing. See
  [task-e4003e5f.md](task-e4003e5f.md).

### Naming conventions

The issue requests renaming for consistency:
- `ctx_arrays` -> `ctx_buffers` (these are device buffers, not arrays)
- `buffer_ptr` -> `buffer_offset` (after pooling, individual tensors are offsets into pools)
- `big_models` -> `large_models` (consistent with the issue's terminology)

## Design review (2026-06-12)

**Verdict: sound-with-changes.** Pooling is the right long-term direction (it is also what the buffer migration/eviction endgame and the alias-view work need), but the proposal as written over-scopes the codegen change, misses two Metal-specific runtime hazards, and should explicitly land **after** #333.

**Recommendations:**

1. **Land after #333.** With `Hosted` removed, Metal storage-mode segregation collapses to "private pools + a small shared staging pool", and the `use_host_memory` host-pointer-wrapping entries (`alloc_if_needed`, `metal_backend.ml:791-799`) disappear. Doing #344 first would force a `Wrapped of buffer_ptr` escape variant into `buffer_offset` plus memory-`type`-driven pool segregation, both deleted again by #333. (ROADMAP already agrees: #333 in v0.7.0, #344 in v0.7.2.)
2. **Phase the codegen change; AC 5 is over-scoped for phase 1.** Only Metal has a binding limit. CUDA and C can keep per-tnode pointer kernel parameters with the pointer computed host-side at link/dispatch as `pool_base + offset` — generated `.c`/`.cu` stays byte-identical, confining risk and `.expected`-file churn to the allocator plus the Metal backend. Emit in-shader `pool_base + offset` for all backends only when migration/eviction actually requires it.
3. **Use untracked hazards for pool buffers on Metal.** Pool buffers allocated with `hazard_tracking_mode_tracked` (current default in `metal_backend.ml:69-77`) would make Metal serialize kernels that touch disjoint tensors in the same pool — false dependencies at pool granularity. Pools should use `hazard_tracking_mode_untracked` and rely on OCANNL's own event machinery (`updating_for`, writer events), which already expresses the true dependencies.
4. **Decide pool ownership and integrate `constant_buffer_cache` and `finalize`.** Contexts/ctx_buffers are per-stream, but `constant_buffer_cache` is per-device (shared across streams), so a per-device constant pool (with locking, since multiple streams per device remain) is needed even if working pools are per-stream. `constant_buffer_cache`'s value type changes to `buffer_offset` too. `Backends.finalize` currently frees per-tnode buffers; under pooling it must decrement slab refcounts / return sub-blocks to the free list — and skip alias entries (consistent with the task-e4003e5f constraint already noted above).
5. **Refine AC 7: offset width caps per-pool size, not total memory.** With `large_models=false` (uint32), cap each pool at 4 GB and open additional pools; total memory stays unbounded and only single tensors > 4 GB genuinely require uint64. Decouple this from `loop_index_type`, which is about element indexing within one tensor.
6. **Keep merge buffers out of pools in phase 1.** They are per-stream, reused via the existing `alloc_buffer ?old_buffer` grow-only pattern, and the new transfer-routine design requires pointer stability across invocations. A dedicated (non-pooled or single-tenant-pool) merge buffer is the simplest thing that preserves that invariant.

**Open decision points for Łukasz:**

- Pool ownership: per-stream working pools + per-device constant pool, or everything per-device with locking?
- Is pooling actually the chosen fix for the 31-binding limit, or is a Metal argument buffer / GPU-address pointer table (one binding containing per-tnode addresses) worth doing first as a small unblocker? Pooling retains independent value (eviction, alias views, allocation overhead, locality) either way, but the pointer-table is an order of magnitude smaller if Metal models > 30 tnodes is the urgent pain.
- Pool exhaustion policy: initial pool size default, growth factor for subsequent pools, and whether `Local`-mode tnodes get an arena-style sub-pool (the gh-ocannl-340 synergy) in phase 1 or later.
- Offsets at dispatch: one offsets-table buffer (1 extra binding, stable layout per routine) vs. `set_bytes` per tensor (no extra buffer, more encoder calls)?

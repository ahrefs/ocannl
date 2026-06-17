# Proposal: Universal Pool Allocator — one-pool-per-context-delta + Metal binding fix

GitHub issue: [ahrefs/ocannl#344](https://github.com/ahrefs/ocannl/issues/344)

**Status**: Open, scheduled under v0.7.2 "Memory management" in ROADMAP.md (the GitHub milestone
still reads v0.7). The behavior-preserving **foundation** this work was built on has already
landed (see "What is already in place"), so what remains is the allocation *policy* and the Metal
codegen change — both confined behind an existing seam.

## Summary

The addressing contract, interface simplification, and allocator seam that the original #344
called for are **done**, landed as the behavior-preserving refactor in
[backend-buffer-addressing.md](backend-buffer-addressing.md) (commits `7d189e75`, `d1c139f1`,
`afa4d26f`). Buffers are already `buffer_loc = { pool_id; offset }`, `ctx_arrays` is already
`ctx_buffers`, and a shared `allocate` seam (`backends.ml:61`) already mints deterministic
per-device pool ids through the backend slab API (`alloc_pool` / `free_pool` / `memset_zero`).
The current policy behind that seam is **one pool per tnode at offset 0** — byte-for-byte
equivalent to the old per-tnode allocation.

This proposal is therefore trimmed to the **pooling capability**: swap the one-pool-per-tnode
policy for a **one-pool-per-context-delta policy** (a context's new tnodes packed into a single
slab), and add the **Metal-only codegen change** that makes this collapse Metal's per-tnode buffer
bindings to O(num_pools). Neither touches the `buffer_loc` contract, the slab API, nor CUDA/C
generated code.

The design choices below are deliberate and narrow:

- **Bump-only, no reuse (parity with today).** Peak pool size = Σ(all materialized tnodes in a
  context) — the same total as today's per-tnode allocation. This is *not* a memory-saving
  change; it consolidates many allocations into few, fixing Metal's binding limit and reducing
  allocation overhead. Liveness-based offset reuse and eviction (the "thick allocator") are
  deferred; their forcing function is "Σ-tnodes exceeds device memory while the working set would
  fit," which has not yet bitten.
- **One pool per context delta.** A `pool_id` *is* a slab — one contiguous backend allocation with
  one base pointer (see "Pools and slabs" below). When a child context adds tensors beyond its
  parent, *all* of that delta's non-constant tensors go into a **single pool** sized exactly to the
  delta, owned by that context. The delta is known atomically at link time, so the slab is sized in
  one shot — no free-list, no refcounting, no "slab full → append" logic. The only release point is
  `finalize`, which frees exactly the pool(s) that context minted. The pool classes are:
  - per-**context-delta** working pool (one slab, freed at the context's `finalize`),
  - per-**device** constant pool (dedup across contexts, mirroring today's
    `constant_buffer_cache`; freed at device teardown, not context `finalize`),
  - the reserved merge pool, kept **out** of any working pool.
  This is per-*context*, not per-*stream*: with one compute stream per device, a per-stream
  working pool would never free until device teardown.
- **Uniform always-copy semantics, single storage mode (no segregation).** All backends copy on
  `to_host`/`from_host` — no device buffer ever aliases a host array. The dead `use_host_memory`
  host-aliasing hook (already a no-op after #333) is deleted, making always-copy a cross-backend
  invariant and giving identical, reproducible buffer behavior on CPU, CUDA, and Metal. Metal uses
  one storage mode for every pool — **shared**, chosen so host transfers need no staging — with no
  per-tnode segregation and no mode config. This is what collapses what would otherwise force
  multiple working pools per delta back to one.
- **Codegen change is Metal-only.** Only Metal has a binding limit. CUDA and C keep their
  per-tnode pointer kernel params, computed host-side as `pool_base + offset` at dispatch — their
  generated `.c`/`.cu` stays byte-identical, confining `.expected`-file churn to Metal. Only
  Metal emits in-shader `pool_base + offset`, binds pools once, and passes offsets as runtime
  args.
- **Offsets are runtime args, never source constants.** `compile`/`compile_batch` run before
  linking and allocation, and `code`/`code_batch` objects are reused across contexts, so offsets
  are only known at link time. They must be passed as `set_bytes` per offset or via a single
  small offsets-table buffer. Specializing shader source per link would forfeit code reuse.

## Goal

Replace per-tensor-node individual buffer allocations with a pool allocator that packs tensors
into a small number of large pool buffers across all backends (Metal, CUDA, C). Each tensor is
already addressed as `{ pool_id; offset }`; this work makes a pool actually hold many tensors
rather than one. The immediate driver is Metal's ~31 argument-buffer binding limit; the change
also reduces allocation overhead, improves memory locality, and gives a natural home for
configurable 32-bit vs 64-bit index arithmetic via the `large_models` setting.

## What is already in place (foundation, landed)

The following were #344 acceptance criteria in the original draft and are now realized — do not
re-do them:

- **`buffer_loc` addressing.** `type buffer_loc = { pool_id : int; offset : int }`
  (`backend_intf.ml:14`); `ctx_buffers : buffer_loc Map.M(Tnode).t` (`backend_intf.ml:16`). The
  `'buffer_ptr` type parameter is gone from the shared interface; the backend keeps a private
  `pool_id → 'base` table.
- **`ctx_arrays → ctx_buffers` rename**, across the interface and all backends.
- **Slab API** replacing the per-tnode `Alloc_buffer` module type: `alloc_pool` / `free_pool`
  (optional, `None` for GC backends) / `memset_zero`, int-in / int-out (`backend_intf.ml:23-40`).
- **Shared allocator seam** `allocate` (`backends.ml:61`), advancing `device.next_pool_id` in
  deterministic tnode order. Current policy: one pool per tnode, offset 0.
- **Merge buffer as a reserved pool** (`merge_buffer_pool_id = 0`, `backend_intf.ml:125`);
  `constant_buffer_cache` stores `buffer_loc` (`backend_intf.ml:114`).
- **`stream` folded into `device`**; `?mode` (memory mode) carried on `alloc_pool`. *(This work
  retires `?mode`'s role in selecting Metal storage mode — Metal commits to shared, see below — so
  the argument becomes vestigial for storage selection.)*
- **Adjacent landings that simplified the ground:** hosted memory removed (#333, `8f949e3f`) — so
  every in-context node, including constants, is allocated in `ctx_buffers` with no host-pointer
  wrapping (and `use_host_memory` is left a no-op — see "Uniform always-copy semantics" below);
  Metal private storage mode (#320, `1cf9a95b`) via `storage_mode_for_memory_mode` /
  `resource_options_for_mode` (`metal_backend.ml:86-92`); `device_to_device` returning a compiled
  transfer routine with static merge-buffer verification (`1162646d`), and `merge_buffer_use = No
  | Copy` (`backend_intf.ml:43`).

## Design

### Pools and slabs

Under the addressing contract `buffer_loc = { pool_id; offset }`, the backend resolves an access as
`base_of(pool_id) + offset` against a **single** base pointer (its private `pool_id → 'base`
table). So a `pool_id` maps to exactly one **slab** — one contiguous backend allocation
(`MTLBuffer` / `CUdeviceptr` / C `malloc`) with one base and one size; many tnodes share a pool
only by taking different offsets into that one slab. A pool cannot span two slabs (offset is
meaningless across non-contiguous bases), so "grow the arena" means "mint another `pool_id`," never
"add a slab to a pool." Throughout this document **pool ≡ slab**.

### One-pool-per-context-delta allocation policy (the seam swap)

The change lives entirely in the shared `link`/`allocate` path (`backends.ml:61`,
`alloc_if_needed` fold at `backends.ml:532`) plus a small per-device pool-state table
(`pool_id → { total_size; bump_offset }`). Today `allocate` mints a fresh `pool_id` and a full
slab for every tnode. Under the new policy, `link` allocates a context's delta in **two passes**:

1. **Enumerate and size.** Fold over `code.lowered.traced_store` to collect the delta — the
   in-context tnodes not already present in the parent's `ctx_buffers` and not constants — and sum
   `align(size, alignment)` over them. This is the whole delta, known atomically before any
   allocation.
2. **Allocate one pool, bump-assign.** `alloc_pool` a single slab of that total, then assign each
   delta tnode `{ pool_id = slab; offset = running_bump }`. No "slab full → append" path: the slab
   is sized correctly the first time.

- **Constant / read-only** nodes (`node.read_only || Tn.known_constant key`, `backends.ml:517`)
  bypass the delta pool: they dedup into the per-**device** constant pool via
  `constant_buffer_cache` exactly as today (bump-packed), because they are shared across contexts
  and outlive any single context.
- **Merge/staging** nodes stay on the reserved merge pool — never a working pool (see below).
- **4 GB cap** (`large_models = false`, uint32 offsets): if a delta's total exceeds 4 GB, split it
  across as many pools as needed. This is the *only* multi-pool-per-delta case, and it is
  deterministic — not exhaustion-driven.

`finalize` (`backends.ml:599`) frees exactly the working pool(s) the context minted. Since a
delta-pool holds only nodes new to that context, the existing skip-by-membership logic (skip
entries owned by a parent context or the constant cache) generalizes cleanly to pool granularity:
free each distinct working `pool_id` whose nodes are all this context's. There is no other release
point.

**Determinism is preserved.** The delta is enumerated in `traced_store`'s existing deterministic
order, so pool ids and offsets remain reproducible across runs — the debuggability property the
foundation established stays intact.

### Metal binding fix (the only codegen change)

Today Metal dispatch iterates `kparams` and calls `set_buffer` once per `Kparam_ptr tn`
(`metal_backend.ml:805-809`) — O(num_tnodes) bindings, which blows the ~31-slot limit for models
with more than ~30 materialized tensors in one kernel. After this work, Metal:

- binds each **pool** once via `set_buffer` — the handful of pools a kernel actually reads from:
  the context's delta-pool chain (its own delta plus any ancestor-context deltas it touches), the
  constant pool, and the merge pool. A few bindings regardless of tnode count,
- emits in-shader `pool_base + offset` casts to the typed pointer for each tensor access,
- passes all per-tensor offsets through **one small offsets-table buffer** (one extra binding,
  stable layout per routine), indexed in-shader. Source-level constants are ruled out because
  `code` objects outlive any single context's allocation, and per-tensor `set_bytes` is ruled out
  because each `set_bytes` consumes a buffer argument index — re-introducing the O(num_tnodes)
  binding blowup pooling exists to remove.

CUDA and C are untouched: they keep `Kparam_ptr tn` kernel params and compute the pointer
host-side as `pool_base + offset` at dispatch (CUDA `cuda_backend.ml:961`, C `cc_backend.ml:403`).
Their generated code stays byte-identical, so `.expected` churn is confined to the Metal backend
and the allocator. In-shader `pool_base + offset` for *all* backends is deferred until
migration/eviction actually requires it.

**Untracked hazards for pool buffers.** Pools currently allocate with
`hazard_tracking_mode_tracked` (`metal_backend.ml:71,77`). At pool granularity that makes Metal
serialize kernels touching *disjoint* tensors in the same pool — false dependencies. Pool buffers
should use `hazard_tracking_mode_untracked` and rely on OCANNL's own event machinery
(`updating_for`, writer events), which already expresses the true dependencies.

### Uniform always-copy semantics; Metal storage mode is always shared

Two coupled simplifications, chosen for reproducibility and to avoid storage-mode bookkeeping:

- **Always-copy everywhere, no host aliasing.** `to_host`/`from_host` always copy on every
  backend; a device buffer never aliases a host array. This is already the de-facto behavior after
  #333 — `use_host_memory` (`backend_impl.ml:40,61`, `metal_backend.ml:161-165`,
  `c_syntax.ml:23,80`) is ignored by `is_in_context_force` (`tnode.ml:182-185`) and its function
  form is never applied. Make it an explicit invariant by **deleting `use_host_memory`** (the
  parameter, the three backend signature fields, the CPU `Some (fun … ptr -> ptr)`, and Metal's
  unused `get_buffer_for_ptr`). The payoff is cross-backend symmetry: CPU, CUDA, and Metal all
  expose buffers through the same copy path, so behavior and `.expected` output are identical and
  reproducible across backends.
- **Metal pools are always shared storage.** Every Metal pool uses `storage_mode_shared`; the
  per-tnode `storage_mode_for_memory_mode` / `resource_options_for_mode` selection
  (`metal_backend.ml:86-92`) is retired, with no replacement config. Shared is chosen as the
  less-hassle mode: on unified memory it matches private's compute and bandwidth, and it lets host
  copies be a direct CPU `memcpy` with **no staging blit or sync** — the round-trip that private
  storage would force on every `to_host`/`from_host`. (`cpu_cache_mode` still tunes upload vs
  readback within shared; that is a separate, later knob.)

Together these are what make one-pool-per-context-delta hold: with a single device-wide storage
mode there is nothing to segregate, so a delta never splits across storage modes. Allocation,
addressing, and the working/constant/merge pool structure are storage-mode-independent.

### `large_models` setting controls index width

Rename (or alias) the existing `big_models` setting (`utils.ml:23,38,529`) to `large_models` for
consistency with the issue's terminology. It already drives `arg_int_prefix` / `loop_index_type`
in `c_syntax.ml:94-95` (uint32 vs uint64); extend it to pool-offset arithmetic.

Offset width caps **per-pool size**, not total memory. With `large_models = false` (uint32),
cap each pool at 4 GB and open additional pools past that — total device memory stays unbounded.
Only a single tnode larger than 4 GB genuinely needs uint64; that case can be deferred with a
clear error meanwhile. Keep this decoupled from `loop_index_type`, which is about element indexing
*within* one tensor, not pool offsets.

### Merge buffers stay out of working pools

The reserved merge pool (`merge_buffer_pool_id = 0`) is left out of any working pool. The
transfer-routine design (`device_to_device` returning a compiled routine) requires the merge
buffer's pointer to stay stable across routine invocations, which a working pool that may relocate
or reuse offsets cannot guarantee. A single-tenant reserved pool is the simplest thing that
preserves that invariant, and the foundation already addresses it uniformly as
`{ pool_id = 0; offset = 0 }`.

## Acceptance Criteria

1. **One-pool-per-context-delta policy behind the seam.** `link` allocates a context's delta as a
   single slab sized to the delta (two passes: enumerate+size, then `alloc_pool` once and
   bump-assign offsets), instead of one slab per tnode. Peak total allocation equals today's
   (non-reuse bump). Pool ids and offsets remain deterministic in `traced_store` order.
2. **Pool classes.** Per-context-delta working pool (freed at that context's `finalize`),
   per-device constant pool (dedup via `constant_buffer_cache`, freed at device teardown), and the
   reserved merge pool kept out of working pools.
3. **Uniform always-copy semantics; Metal always shared.** `use_host_memory` is deleted (the
   parameter and all backend bindings), making always-copy `to_host`/`from_host` a cross-backend
   invariant — no device buffer aliases a host array. Every Metal pool uses `storage_mode_shared`;
   the per-tnode `storage_mode_for_memory_mode` selection is retired with no replacement config.
4. **Metal binding count reduced.** Metal dispatch binds O(num_pools) buffers via `set_buffer`
   instead of O(num_tnodes), with all offsets passed through a single offsets-table buffer. Total
   bindings stay within the ~31-slot limit for models with hundreds of tensor nodes. Pool buffers
   use `hazard_tracking_mode_untracked`.
5. **CUDA/C generated code unchanged.** CUDA and C keep per-tnode `Kparam_ptr` params with the
   pointer computed host-side as `pool_base + offset`; their `.c`/`.cu` output is byte-identical.
6. **`large_models` setting.** `big_models` is renamed (or aliased) to `large_models`; pool
   offsets use uint32 with each pool capped at ≤ 4 GB (open another pool past the cap), uint64
   reserved for the single-tnode-over-4 GB case (may be deferred with a clear error). Decoupled
   from `loop_index_type`.
7. **No regression in existing tests.** All existing tests pass; `.expected` changes are confined
   to the Metal backend and allocator-level output.

## Context

### Why pooling is necessary

Metal compute shaders have a hard limit of ~31 argument-buffer bindings per compute command
encoder. Today the Metal backend binds one buffer per tensor node
(`metal_backend.ml:805-809`), which fails for models with more than ~30 materialized tensors in a
single kernel. Packing tensors into 1–3 large pools, each bound once and addressed by offset
arithmetic in-shader, removes the limit. Pooling retains independent value — reduced allocation
overhead, better locality, and the substrate for future eviction and alias views — regardless of
the Metal driver.

### Alignment requirements

Pool sub-allocations must respect backend-specific alignment:

- **Metal**: 256 bytes (page alignment preferred)
- **CUDA**: 256 bytes (memory-coalescing alignment)
- **C/host**: 64 bytes (cache-line alignment)

### Relationship to other work

- **[backend-buffer-addressing.md](backend-buffer-addressing.md)** — *prerequisite, landed.*
  Established the `buffer_loc` contract, the slab API, the allocator seam, and the stream→device
  fold. This proposal is the policy swap behind that seam.
- **gh-ocannl-333 (remove hosted memory)** — *landed (`8f949e3f`).* Removed the host-side `array`
  field and host-pointer wrapping, so every in-context node is a plain device buffer and
  `use_host_memory` became a no-op. This work finishes the job by deleting `use_host_memory`
  outright — a small #333 follow-through that belongs to this change because it removes the last
  storage-mode ambiguity the pool policy would otherwise have to reason about.
- **gh-ocannl-320 (Metal private mode)** — *landed (`1cf9a95b`).* Provided per-tnode private
  storage. This work supersedes that selection: Metal commits to `storage_mode_shared` for all
  pools, so the `storage_mode_for_memory_mode` mechanism is retired.
- **[task-e4003e5f.md](task-e4003e5f.md) (slice as alias view, subtask 293a of #293)** — unblocked
  by the foundation's `buffer_loc` contract (an alias is `{ pool_id; offset + delta }`), not by
  this pooling work. **Constraint on this work:** the pool free/reclaim path must be
  alias-aware — never free a slab while alias entries reference it, and skip alias entries
  themselves when freeing. `finalize`'s existing parent-shared / constant-cache skip logic is the
  natural place to extend.

## Open decisions

- **Pointer-table alternative.** If "Metal models > 30 tnodes" is the only urgent pain, a Metal
  argument buffer / GPU-address pointer table (one binding holding per-tnode addresses) is an
  order of magnitude smaller to build and would unblock it sooner. Pooling still has independent
  value (eviction, alias views, allocation overhead, locality), so this is a sequencing question,
  not an either/or.

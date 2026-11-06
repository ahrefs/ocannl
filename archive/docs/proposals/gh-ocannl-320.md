# Use private storage mode in Metal for GPU-only buffers

## Goal

Reduce CPU-cache-coherency overhead and let Metal apply GPU-specific layout
optimizations by allocating GPU-only buffers with `MTLStorageModePrivate`
instead of unconditionally using `MTLStorageModeShared`. Buffers that the CPU
needs to read or write continue to use shared mode.

Tracks ahrefs/ocannl#320 (milestone v0.8). The original issue title mentions an
exception "when tnode mode is hosted-sharing-cross-streams"; that exception is
no longer relevant because multi-streaming was removed in gh-ocannl-341, so the
decision reduces to "does the CPU need to read/write this buffer?".

## Acceptance Criteria

- Buffers backing tnodes with `Device_only` or `On_device` memory mode are
  allocated with `MTLStorageModePrivate`.
- Buffers backing tnodes with `Local` memory mode are allocated with
  `MTLStorageModePrivate`.
- Buffers backing tnodes with `Hosted _`, `Materialized`, or
  `Effectively_constant` memory mode continue to be allocated with
  `MTLStorageModeShared` (CPU access is needed for at least one of host
  initialization, host read-back, or `use_host_memory` wrapping).
- `from_host` and `to_host` correctly transfer data into and out of
  private-mode buffers (the existing blit-encoder path that bridges via a
  shared temporary buffer suffices — see Context).
- `alloc_zeros`'s blit-fill continues to work for both modes (it is a
  GPU-side fill and is mode-agnostic).
- `alloc_buffer`'s old-buffer reuse path does not hand back a buffer whose
  storage mode differs from the requested mode (i.e., a shared buffer must
  not be reused when private was requested, and vice versa).
- All existing tests under `test/` and `arrayjit/test/` that exercise the
  Metal backend pass on Apple Silicon (M1/M2/M3+) with no new failures.

## Context

OCANNL's Metal backend currently uses a single resource-options constant for
every buffer it allocates. Apple's guidance is to choose private mode when the
CPU never accesses the resource: even on the unified-memory architecture of
Apple Silicon, private buffers avoid CPU cache snooping and let Metal pick a
GPU-friendly layout (compression, tiling). On discrete-GPU Macs the win is
much larger because private buffers live in VRAM.

### Current allocation surface (`arrayjit/lib/metal_backend.ml`)

- A single module-level constant `resource_options` is defined as
  `storage_mode_shared + cpu_cache_mode_write_combined + hazard_tracking_mode_tracked`.
- It is passed to `Me.Buffer.on_device` in `Alloc_buffer.alloc_buffer`,
  `alloc_array`, and `alloc_zeros`.
- `alloc_buffer` reuses a previously allocated `buffer` when its
  `size_in_bytes` is large enough for the new request — no current check on
  storage mode, since today there is only one mode.
- `alloc_zeros` initializes via `Me.BlitCommandEncoder.fill_buffer`, a
  GPU-side operation that works on private buffers.

### Host transfer surface

- `from_host` and `to_host` (around the `(* --- Copy Operations --- *)`
  banner) already use the staging-buffer pattern, but built around
  `Me.Buffer.on_device_with_bytes_no_copy`: that wraps the host pointer as a
  shared Metal buffer, then a `BlitCommandEncoder.copy_from_buffer` performs a
  GPU-side copy to (or from) the destination buffer. Because the blit is
  GPU-to-GPU, this code path already works correctly when the
  destination/source buffer is private — no rewrite is required.
- The fast path that skips the blit when `dst_fatptr = host_ptr` only fires
  on shared/host-aliased buffers (those produced by
  `get_buffer_for_ptr` / `use_host_memory`); it is dead for private-mode
  buffers.

### Where buffers get bound to tnodes

- `Backends.alloc_if_needed` (`arrayjit/lib/backends.ml`) is the entry point
  the framework calls when materializing a tnode into a context. It already
  has the `Tnode.t key` in scope when it calls `alloc_array` / `alloc_zeros`,
  so threading the tnode's `memory_mode` to the backend is mechanical.
- Constant-buffer caching (`stream.device.constant_buffer_cache`) and
  `use_host_memory` in `metal_backend.ml` allocate buffers that *wrap* host
  memory; these stay shared.

### Tnode memory modes (`arrayjit/lib/tnode.ml`)

The `memory_mode` variant has: `Effectively_constant`, `Virtual`,
`Never_virtual`, `Local`, `Device_only`, `On_device`, `Materialized`, and
`Hosted of memory_type`. CPU-access requirement by variant:

- No CPU access: `Local`, `Device_only`, `On_device` — candidates for private.
- CPU access required: `Hosted _`, `Materialized`, `Effectively_constant`
  — must stay shared.
- `Virtual` is inlined and has no buffer; `Never_virtual` is a partially
  resolved state and should map to shared (conservative default).

### Backend interface impact (`arrayjit/lib/backend_intf.ml`)

`Alloc_buffer` is a cross-backend module type:
```
val alloc_buffer : ?old_buffer:buffer -> size_in_bytes:int -> stream -> buffer
val alloc_array : Ops.prec -> dims:int array -> stream -> buffer_ptr
val alloc_zeros : Ops.prec -> dims:int array -> stream -> buffer_ptr
```
Threading per-tnode mode into these signatures touches `metal_backend.ml`,
`cuda_backend.ml`, `backend_impl.ml`, `schedulers.ml`,
`no_device_backend_missing.ml`, and `lowered_backend_missing.ml`. Non-Metal
backends can simply ignore the new argument.

## Approach (suggested)

*Suggested approach — agents may deviate if they find a better path.*

1. **Resource-option constants in `metal_backend.ml`.** Rename the existing
   `resource_options` to `shared_resource_options` and add
   `private_resource_options`:
   ```ocaml
   let shared_resource_options =
     Me.ResourceOptions.(
       storage_mode_shared + cpu_cache_mode_write_combined + hazard_tracking_mode_tracked)
   let private_resource_options =
     Me.ResourceOptions.(storage_mode_private + hazard_tracking_mode_tracked)
   ```
   `cpu_cache_mode_write_combined` is meaningless for private buffers so it
   is omitted; `hazard_tracking_mode_tracked` applies to both modes.

2. **Mode classifier.** Add
   `let resource_options_for_mode (m : Tnode.memory_mode) = ...` mapping
   `Local | Device_only | On_device` to `private_resource_options` and
   everything else (including `Never_virtual`, `Effectively_constant`,
   `Materialized`, `Hosted _`) to `shared_resource_options`. `Virtual`
   shouldn't reach the allocator; map it to shared defensively.

3. **Thread mode through allocators.** Add an optional `?mode:Tnode.memory_mode`
   parameter to `alloc_buffer`, `alloc_array`, `alloc_zeros` in the
   `Alloc_buffer` module type. Default to a value that produces shared options
   (so no other backend changes behavior). Update Metal's implementations to
   pick options via `resource_options_for_mode`. Update callers in
   `backends.ml` (`alloc_if_needed`, lines around the existing `alloc_array`/
   `alloc_zeros` calls and the merge-buffer allocation) to pass
   `~mode:(fst (Option.value_exn key.memory_mode))` (or equivalent — the
   tnode is in scope as `key`).

   For backends that don't care (CUDA, CPU, missing-backend stubs), accept
   and ignore the parameter.

4. **Buffer-reuse storage-mode guard.** In Metal's `alloc_buffer`, when the
   `?old_buffer` candidate is large enough, additionally check that its
   `Me.Resource.get_storage_mode` (or equivalent — the implementation should
   confirm the binding exposes this) matches the requested mode before
   reusing. On mismatch, fall through to fresh allocation. Because
   `merge_buffer` reuse passes through this same path, this guard prevents
   handing a previously-shared buffer back when a private-mode tnode now
   wants its slot.

5. **Host-transfer paths.** No code changes are required in `from_host` or
   `to_host`: the existing blit pattern works for both modes. Verify this
   during testing — specifically, exercise a `Device_only` tensor that is
   later read back with `to_host`.

6. **Constant-cache and `use_host_memory`.** Leave alone:
   `get_buffer_for_ptr` wraps host memory and must stay shared, and the
   constant-buffer cache uses that same path.

### Out of scope (deferred to a follow-up)

- Promoting `Effectively_constant` and `Hosted Constant` tnodes to private
  with a one-shot blit copy from a staging buffer. The decision matrix in
  the task elaboration flags these as a possible second-pass optimization;
  this proposal explicitly leaves them on shared mode.
- Performance benchmarking — the change targets correctness of mode
  selection. A follow-up issue/task can measure the actual speedup on
  matmul / large tensor workloads.

## Scope

**In scope.** Mode-dependent buffer allocation in `metal_backend.ml`, mode
threading through the cross-backend `Alloc_buffer` interface, buffer-reuse
storage-mode guard, regression testing on Apple Silicon.

**Out of scope.** Private mode for `Effectively_constant` /
`Hosted Constant` (deferred); benchmarking; cleanup of vestigial
`cross_stream_candidates` / `owner_stream` / `Shared_cross_streams` left over
from #341 (separate task); changes to non-Metal backends beyond accepting
and ignoring the new optional parameter.

**Hardware requirement.** Implementation and verification require an Apple
Silicon machine (M1/M2/M3+) with Metal — running the existing Metal backend
tests is mandatory before merging.

**Dependencies.** None blocking. gh-ocannl-341 (multi-streaming removal) is
already done and simplifies this work.

# Proposal: Implement a Universal Pool Allocator across backends

GitHub issue: [ahrefs/ocannl#344](https://github.com/ahrefs/ocannl/issues/344)

## Goal

Replace per-tensor-node individual buffer allocations with a pool allocator that consolidates tensors into a small number of large pool buffers shared across all backends (Metal, CUDA, C). Each tensor is addressed as a `(pool_id, byte_offset)` pair within a pool buffer, rather than holding its own device pointer. This solves Metal's ~31 argument buffer binding limit, reduces allocation overhead, improves memory locality, and provides a natural place to introduce configurable 32-bit vs 64-bit index arithmetic via the `large_models` setting.

## Acceptance Criteria

1. **Pool allocator module**: A new `Pool_allocator` module (or extension of `backend_intf.ml`) provides a backend-agnostic pool allocation interface. Each pool has a backend-specific base pointer, a total size, and a bump/free-list allocator that returns `{ pool_id; offset; size_in_bytes }` records with configurable alignment.

2. **`buffer_ptr` replaced by `buffer_offset`**: The `'buffer_ptr buffer` type in `backend_intf.ml` (line 7) is replaced by a `buffer_offset` record containing `pool_id : int`, `offset : int`, and `size_in_bytes : int`. The per-tnode map `ctx_arrays` (line 8) maps tnodes to `buffer_offset` values instead of raw backend pointers.

3. **`ctx_arrays` renamed to `ctx_buffers`**: All occurrences of `ctx_arrays` across `backend_intf.ml`, `backend_impl.ml`, `metal_backend.ml`, `cuda_backend.ml`, `cc_backend.ml`, and all callers are renamed to `ctx_buffers`.

4. **Pool-based allocation in all backends**: Each backend's `Alloc_buffer` implementation allocates from pools rather than making individual `mem_alloc` / `Buffer.on_device` / `allocate_n` calls:
   - **Metal** (`metal_backend.ml` lines 75-83): `Me.Buffer.on_device` creates pool buffers, not per-tnode buffers.
   - **CUDA** (`cuda_backend.ml` lines 62-71): `Cu.Deviceptr.mem_alloc` creates pool buffers.
   - **C** (`backend_impl.ml` lines 44-72): `Ctypes.allocate_n` creates pool buffers.

5. **Code generation emits pool base + offset access**: `c_syntax.ml` parameter generation (lines 760-800) emits pool base pointers as kernel arguments and offset constants (or additional arguments) instead of per-tnode pointer arguments. Generated C/Metal/CUDA code casts `(pool_base + offset)` to the appropriate typed pointer for each tensor access.

6. **Metal binding count reduced**: Metal kernel dispatch (`metal_backend.ml` lines 757-785) binds O(num_pools) buffers via `set_buffer` instead of O(num_tnodes). Offsets are passed as kernel constants or a small argument buffer. This keeps total bindings well within Metal's 31-slot limit for models with hundreds of tensor nodes.

7. **`large_models` setting controls index width**: The existing `big_models` setting in `Utils.settings` is renamed (or aliased) to `large_models`. When `large_models = true`, pool offsets and index arithmetic use `uint64_t` (supporting pools > 4 GB). When `false` (default), `uint32_t` is used. This affects `arg_int_prefix` and `loop_index_type` in `c_syntax.ml` (lines 94-95) and offset-related types in generated code.

8. **Pool lifecycle management**: Pools are created at device/stream initialization with a configurable initial size. When a pool is exhausted, a new pool is allocated (pools are never resized, since that would invalidate offsets). Pools are freed when the owning device/stream is finalized. The `allocated_buffer` reuse pattern in `stream_ref` (line 111 of `backend_intf.ml`) is replaced by pool-level management.

9. **Merge buffer integration**: The `merge_buffer` field on `stream_ref` (`backend_intf.ml` line 109) works with the pool allocator. Merge buffers can be allocated from a dedicated temporary pool or from the main pool with fast reclaim.

10. **No regression in existing tests**: All existing tests pass after the refactoring.

## Context

### Why pool allocation is necessary

Metal compute shaders have a hard limit of ~31 argument buffer bindings per compute command encoder. The current architecture binds one Metal buffer per tensor node (`metal_backend.ml` line 761: `Me.ComputeCommandEncoder.set_buffer encoder ~index buffer`), which fails for models with more than ~30 materialized tensors in a single kernel. A pool allocator consolidates all tensors into 1-3 large pool buffers, each bound once, with individual tensors addressed by offset arithmetic within the shader.

### Current allocation architecture

Each backend allocates buffers independently per tensor node. The `Alloc_buffer` module type (`backend_intf.ml` lines 26-35) defines `alloc_buffer`, `alloc_array`, `alloc_zeros`, and `free_buffer`. The `ctx_arrays` field (`backend_intf.ml` line 188) in the context type maps each tnode to a backend-specific `buffer_ptr`. At link time, each `Param_ptr tn` in the kernel parameter list causes a separate buffer binding.

Key code paths affected:
- **Parameter list construction**: `c_syntax.ml` line 773 -- each in-context tnode becomes a `Param_ptr tn` entry
- **Metal dispatch**: `metal_backend.ml` lines 757-785 -- iterates `Param_ptr` params, calls `set_buffer` per tnode
- **CUDA dispatch**: `cuda_backend.ml` line 949 -- iterates `Param_ptr` params, pushes device pointers as kernel args
- **C dispatch**: `cc_backend.ml` line 400 -- iterates `Param_ptr` params, passes pointers to compiled function
- **Buffer type**: `backend_intf.ml` line 7 -- `type 'buffer_ptr buffer = { ptr : 'buffer_ptr; size_in_bytes : int }`

### Alignment requirements

Pool sub-allocations must respect backend-specific alignment:
- **Metal**: 256 bytes (page alignment preferred for private storage mode)
- **CUDA**: 256 bytes (memory coalescing alignment)
- **C/host**: 64 bytes (cache line alignment)

### Relationship to other work

- **gh-ocannl-333 (remove hosted memory)**: Pool allocation changes buffer management; synergistic with removing the host-side `array` field since device buffers become the sole storage.
- **gh-ocannl-320 (Metal private mode)**: Entire pools get one storage mode, simplifying private mode adoption.
- **gh-ocannl-340 (Local_scope init)**: Local tnodes with short lifetimes benefit from arena-style allocation within pools.

### Naming conventions

The issue requests renaming for consistency:
- `ctx_arrays` -> `ctx_buffers` (these are device buffers, not arrays)
- `buffer_ptr` -> `buffer_offset` (after pooling, individual tensors are offsets into pools)
- `big_models` -> `large_models` (consistent with the issue's terminology)

# Proposal: Re-introduce CUDA `__constant__` Memory for Qualifying Tensors

**Task**: gh-ocannl-195
**Issue**: https://github.com/ahrefs/ocannl/issues/195

## Goal

Selectively place small, read-only (`Effectively_constant` / `Hosted Constant`) tensors into CUDA `__constant__` memory to exploit the hardware constant cache. This is a targeted optimization for data that all threads in a warp read from the same address simultaneously (broadcast access pattern), which becomes relevant once multi-threaded kernels (gh-ocannl-412) are implemented.

## Acceptance Criteria

- [ ] Tensors marked `Effectively_constant` or `Hosted Constant` that are (a) small enough to fit within a per-kernel constant memory budget (configurable, default 16KB out of 64KB total) and (b) not mutated between kernel launches are emitted as `__constant__` device variables instead of kernel pointer parameters
- [ ] The CUDA code generator in `c_syntax.ml` / `cuda_backend.ml` emits `__constant__ <type> <name>[<size>];` declarations for qualifying tensors, and references them directly in kernel bodies (no pointer parameter needed)
- [ ] At link time, `Cu.Module.get_global` (already bound in ocaml-cudajit) is used to obtain the device pointer to each `__constant__` variable, and `Cu.Stream.memcpy_H_to_D` copies the host data into it once
- [ ] A cumulative constant memory tracker prevents exceeding 64KB per CUDA module; tensors that would overflow the budget fall back to regular global memory (kernel pointer parameter)
- [ ] The constant memory budget is configurable via `Utils.settings` or a global arg (`cuda_constant_memory_budget`, default 16384 bytes)
- [ ] All existing CUDA tests pass with no regression
- [ ] A benchmark comparing constant memory vs global memory for a workload with small constant tensors (e.g., number literals, small lookup tables, positional encoding tables) shows the expected cache benefit under multi-threaded execution (or documents that single-threaded mode shows no measurable difference, confirming the optimization is forward-looking)

## Context

### CUDA Constant Memory Characteristics

CUDA constant memory is a 64KB region of read-only device memory with a dedicated cache. Key properties:

- **Broadcast**: When all threads in a warp read the same address, the constant cache serves the request in a single cycle -- as fast as reading a register
- **Serialization**: When different threads read different addresses, accesses are serialized -- potentially 32x slower than global memory with L1/L2 cache
- **Per-module**: Each `CUmodule` has its own 64KB constant memory space. Variables are declared with `__constant__` in CUDA source and populated via `cuModuleGetGlobal` + `cuMemcpyHtoD` after module load
- **Static lifetime**: Constant memory persists for the lifetime of the module; it cannot be dynamically resized

### Current Architecture

**Tensors are passed as kernel pointer parameters.** In `c_syntax.ml` `compile_proc` (line 756), every materialized or in-context tensor becomes a pointer parameter to the kernel function. The `link_proc` function in `cuda_backend.ml` (line 929) maps each `Param_ptr tn` to a `Cu.Stream.Tensor` kernel argument.

**`Effectively_constant` is already tracked.** The `Tnode.memory_mode` type (in `tnode.ml` line 48) includes `Effectively_constant`, set for number literals, fixed parameters, and constant-filled tensors (`tensor.ml` lines 585, 601, 639). The `known_constant` function (`tnode.ml` line 273) returns true for these. Backends currently do not special-case this -- they are allocated as regular device memory.

**`__constant__` is already used for builtins.** The ThreeFry PRNG in `builtins_cuda.ml` (line 297) uses `__device__ __constant__ unsigned int THREEFRY_C240 = ...` and `THREEFRY_ROTATION[8][4]`. These are hardcoded strings embedded in the CUDA source. This confirms the compilation pipeline (NVRTC + module loading) handles `__constant__` declarations correctly.

**`cuModuleGetGlobal` is bound.** The ocaml-cudajit binding (`cuda.ml` line 1792) wraps `cuModuleGetGlobal_v2`, returning a `Deviceptr.t` and size. This is the API needed to locate `__constant__` variables after module loading.

**Kernels run single-threaded today.** The `kernel_prep_line` guard (`cuda_backend.ml` line 329) forces `grid_dim=1, block_dim=1`. Constant memory's broadcast advantage requires multiple threads reading the same address in a warp. Until gh-ocannl-412 (multi-threaded kernels), the performance benefit is negligible.

### Relevant Code Locations

| Component | File | Lines | Relevance |
|-----------|------|-------|-----------|
| Memory mode type | `arrayjit/lib/tnode.ml` | 47-55 | `Effectively_constant` definition |
| `known_constant` | `arrayjit/lib/tnode.ml` | 273-276 | Predicate for constant tensors |
| Kernel param assembly | `arrayjit/lib/c_syntax.ml` | 756-810 | Where tensor params are collected; needs branching for constant tensors |
| Kernel launch | `arrayjit/lib/cuda_backend.ml` | 929-987 | `link_proc`: maps params to kernel args; constant tensors bypass this |
| Module loading | `arrayjit/lib/cuda_backend.ml` | 989-1002 | `link`: loads PTX module; constant data copy goes here |
| Existing `__constant__` | `arrayjit/lib/builtins_cuda.ml` | 297-309 | ThreeFry constants pattern |
| `cuModuleGetGlobal` | `ocaml-cudajit/src/cuda.ml` | 1789-1794 | OCaml binding already available |
| Tensor constant marking | `tensor/tensor.ml` | 585, 601, 639 | Where `Effectively_constant` is set |

### Why Not Wait for Multi-Threaded Kernels?

The implementation is low-risk and can be done incrementally:

1. **The plumbing is straightforward**: The code generator needs to emit `__constant__` declarations instead of params, and the linker needs one `Cu.Module.get_global` + memcpy per constant tensor. These are small, self-contained changes.
2. **It reduces kernel parameter count**: Even in single-threaded mode, moving constants out of the kernel parameter list simplifies the launch interface. CUDA has a 4KB limit on kernel parameters; for models with many small constant tensors, this overhead matters.
3. **It's testable now**: Correctness can be verified immediately. Performance testing waits for gh-ocannl-412.
4. **Forward compatibility**: When multi-threaded kernels land, constant memory is automatically exploited without further changes.

However, if the team prefers to focus v0.8 effort on gh-ocannl-412 first, this task can be deferred to a post-412 follow-up with no architectural cost.

## Approach

### Phase 1: Code Generation for `__constant__` Declarations

**Modify `c_syntax.ml` `compile_proc`** to partition tensors into constant-memory vs regular params:

1. After the current `Hashtbl.fold traced_store` that builds the `params` list, add a separate pass that identifies qualifying constant tensors: `Tn.known_constant tn && not (Tn.known_virtual tn) && size_in_bytes <= budget`.
2. For qualifying tensors, emit a `__constant__ <type> <name>[<num_elements>];` declaration before the kernel function (as a module-level declaration, similar to builtins).
3. Remove these tensors from the `params` list so they are not kernel arguments.
4. Kernel body references to these tensors remain unchanged (array indexing works the same whether the array is a parameter pointer or a module-level `__constant__` variable).

This requires a new hook in `C_syntax_config`: `constant_declarations : (Tn.t * string) list -> PPrint.document` that is non-empty only for the CUDA backend.

**Track cumulative usage**: A mutable counter accumulates the size of all `__constant__` tensors per compilation unit. When the budget is exhausted, remaining qualifying tensors fall back to regular params.

### Phase 2: Data Population at Link Time

**Modify `cuda_backend.ml` `link` and `link_batch`** to populate constant memory after module loading:

1. After `Cu.Module.load_data_ex`, iterate over the constant tensors recorded during code generation.
2. For each, call `Cu.Module.get_global run_module ~name:<ident>` to get the device pointer.
3. Copy the host data with `Cu.Stream.memcpy_H_to_D` (or synchronous `Cu.Deviceptr.memcpy_H_to_D` since constant memory must be set before any kernel launch using it).

The `code` record in `cuda_backend.ml` (line 295) needs a new field: `constant_tensors : Tn.t list` listing the tensors placed in constant memory (so `link_proc` knows which tensors are NOT in the params list).

### Phase 3: Configuration and Safety

- Add `cuda_constant_memory_budget` to `Utils.settings` or as a global arg (default 16384).
- Add a diagnostic log (at log level 3+) listing which tensors were placed in constant memory and the total constant memory usage.
- Ensure `Effectively_constant` tensors that might change between training steps (like learning rate schedules) are excluded. The `known_constant` predicate in `tnode.ml` already distinguishes `Hosted Constant` from `Hosted Volatile`; only the former qualifies.

### Phase 4: Testing and Benchmarking

- Run existing CUDA test suite to verify no regression.
- Add a targeted test: create a model with several small constant tensors (scalar multipliers, bias terms), verify that the generated CUDA source contains `__constant__` declarations and that the kernel parameter list is shorter.
- Benchmark under single-threaded mode to confirm no regression (expected: neutral performance, possibly slight improvement from reduced param parsing).
- Document that significant performance gains are expected only after gh-ocannl-412 enables multi-threaded execution with warp-level broadcast patterns.

### Estimated Effort

Small-medium (2-3 days). The core changes are in `c_syntax.ml` and `cuda_backend.ml`, with no changes needed to the IR, tensor system, or ocaml-cudajit bindings.

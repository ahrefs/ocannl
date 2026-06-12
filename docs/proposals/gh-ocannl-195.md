# Proposal: Re-introduce CUDA `__constant__` Memory for Qualifying Tensors

**Task**: gh-ocannl-195
**Issue**: https://github.com/ahrefs/ocannl/issues/195

## Status update (2026-06-12)

- Issue #195 is OPEN, label `enhancement`, milestone v0.8 ("GPU tiling and
  related optimizations"; per ROADMAP.md v0.8 targets mid-June 2026, though
  the schedule is running late — the repo is still at version 0.6.3).
  Issue #412 (multi-threaded kernels / tensor cores) is also OPEN at v0.8,
  so the "performance benefit waits for #412" framing still holds.
- **Nothing has landed** for this proposal: no `__constant__` emission or
  constant-memory budget exists in `c_syntax.ml` / `cuda_backend.ml`; the
  only `__constant__` uses remain the ThreeFry builtins in
  `builtins_cuda.ml` (still line 297).
- Kernel-parameter identifiers were renamed (#356, commit `9262ab44`):
  `Param_ptr` is now `Kparam_ptr`, and the `params` list / `code.params`
  field are now `kparams` (`cuda_backend.ml:284–292`, `link_proc` at
  `cuda_backend.ml:920`). Phase 1/2 text below should read accordingly.
- Line numbers drifted (fixed in the table below): `compile_proc` is at
  `c_syntax.ml:842`; `Effectively_constant` at `tnode.ml:23`;
  `known_constant` at `tnode.ml:244`; constant marking at `tensor.ml:593,
  609, 647`; `kernel_prep_line` at `cuda_backend.ml:317` (the launch still
  forces `grid_dim_x:1, block_dim_x:1`, now at `cuda_backend.ml:970`);
  ocaml-cudajit's `get_global` at `cuda.ml:1788–1793`.
- Related backend work that landed since (Metal private storage mode for
  GPU-only buffers, commit `1cf9a95b`; merge-buffer/`device_to_device`
  reshape in `backend_intf.ml`) does not touch the CUDA constant-memory
  path; the design here remains applicable.
- Remains to do: all four phases. Still a reasonable v0.8 follow-up to or
  companion of #412.

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

**Tensors are passed as kernel pointer parameters.** In `c_syntax.ml` `compile_proc` (line 842), every materialized or in-context tensor becomes a pointer parameter to the kernel function. The `link_proc` function in `cuda_backend.ml` (line 920) maps each `Kparam_ptr tn` *(renamed from `Param_ptr` by #356)* to a `Cu.Stream.Tensor` kernel argument.

**`Effectively_constant` is already tracked.** The `Tnode.memory_mode` type (in `tnode.ml` line 23) includes `Effectively_constant`, set for number literals, fixed parameters, and constant-filled tensors (`tensor.ml` lines 593, 609, 647). The `known_constant` function (`tnode.ml` line 244) returns true for these. Backends currently do not special-case this -- they are allocated as regular device memory.

**`__constant__` is already used for builtins.** The ThreeFry PRNG in `builtins_cuda.ml` (line 297) uses `__device__ __constant__ unsigned int THREEFRY_C240 = ...` and `THREEFRY_ROTATION[8][4]`. These are hardcoded strings embedded in the CUDA source. This confirms the compilation pipeline (NVRTC + module loading) handles `__constant__` declarations correctly.

**`cuModuleGetGlobal` is bound.** The ocaml-cudajit binding (`cuda.ml` line 1792) wraps `cuModuleGetGlobal_v2`, returning a `Deviceptr.t` and size. This is the API needed to locate `__constant__` variables after module loading.

**Kernels run single-threaded today.** The `kernel_prep_line` guard (`cuda_backend.ml` line 317) plus the `grid_dim_x:1, block_dim_x:1` launch (line 970) force single-threaded execution. Constant memory's broadcast advantage requires multiple threads reading the same address in a warp. Until gh-ocannl-412 (multi-threaded kernels), the performance benefit is negligible.

### Relevant Code Locations

| Component | File | Lines | Relevance |
|-----------|------|-------|-----------|
| Memory mode type | `arrayjit/lib/tnode.ml` | 23 | `Effectively_constant` definition |
| `known_constant` | `arrayjit/lib/tnode.ml` | 244-247 | Predicate for constant tensors |
| Kernel param assembly | `arrayjit/lib/c_syntax.ml` | 842+ | `compile_proc`: where tensor kparams are collected; needs branching for constant tensors |
| Kernel launch | `arrayjit/lib/cuda_backend.ml` | 920-987 | `link_proc`: maps kparams to kernel args; constant tensors bypass this |
| Module loading | `arrayjit/lib/cuda_backend.ml` | 988-1012 | `link`: loads PTX module; constant data copy goes here |
| Existing `__constant__` | `arrayjit/lib/builtins_cuda.ml` | 297-309 | ThreeFry constants pattern |
| `cuModuleGetGlobal` | `ocaml-cudajit/src/cuda.ml` | 1788-1793 | OCaml binding already available |
| Tensor constant marking | `tensor/tensor.ml` | 593, 609, 647 | Where `Effectively_constant` is set |

*(Update 2026-06-12: line numbers refreshed; "params"/"Param_ptr" became
"kparams"/"Kparam_ptr" via #356.)*

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

The `code` record in `cuda_backend.ml` (line 284) needs a new field: `constant_tensors : Tn.t list` listing the tensors placed in constant memory (so `link_proc` knows which tensors are NOT in the `kparams` list).

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

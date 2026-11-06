# Proposal: Tiling and Multi-Threaded GPU Kernels (Matrix Multiplication Optimizations)

**Task**: gh-ocannl-412
**Issue**: https://github.com/ahrefs/ocannl/issues/412

## Status update (2026-07-04) — infrastructure landed; transform layer is the remaining work

The [axis-types-for-loops](../../../docs/design/axis-types-for-loops.md) proposal landed in full (Phase A in
PR #84 on 2026-07-03; Phases B+C plus the `validate_parallel` write-coverage hardening
on 2026-07-04), together with [interval-analysis-scalar-t](../../../docs/design/interval-analysis-scalar-t.md)
and the [signed-index-precision](../../../docs/design/signed-index-precision.md) core migration (both PR #88).
This delivers, in different form, most of what Phases 2–3 below planned as
infrastructure:

- **The single-threaded baseline is gone as a hard limit.** `kernel_prep_line` is empty
  (`cuda_backend.ml:362`); CUDA launches real grid/block dims from
  `Low_level.launch_dims` (`cuda_backend.ml:1062`), Metal dispatches real
  `threadgroups_per_grid`/`threads_per_threadgroup` and validates the block product at
  pipeline creation (`metal_backend.ml:854,929-933`). All-`Serial` kernels still launch
  1×1 — which is every kernel today, because nothing *produces* annotations yet.
- **Phase 2's "Parallel_loop variant / parallel_for_syntax hook" exists** as the `axis`
  field on `For_loop` (`axis_type = Serial | Grid | Workgroup | Workgroup_reduce |
  Unrolled`, `low_level.ml:42,59`) and the `C_syntax_config` hooks
  `hardware_index`/`barrier_syntax`/`shared_decl_prefix` (`c_syntax.ml:73-84`), with
  serial fallback on cc exactly as §Phase 2's option 1 wanted.
- **Phase 3's IR constructs exist** in the "aligned with existing architecture" variant
  this proposal recommended: no `Shared_decl`/`Cooperative_load` vocabulary — instead
  `Workgroup_barrier` (`low_level.ml:79`), shared placement as
  `optimized.workgroup_shared : Set.M(Tnode).t` over `Local`-mode nodes
  (`low_level.ml:227`), and cooperative loads as ordinary annotated
  `For_loop`/`Set`/`Get` code. Executed on Metal/CUDA via
  `test/operations/hardware_workgroup_reduce.ml` (hand-built GROUP_REDUCE).
- **Boundary handling has a mechanism**: the new `If` guard statement
  (`low_level.ml:83`) plus interval folding in `simplify_llc` — guards provable from
  loop extents fold away (construct-then-fold). `compile_proc` auto-injects
  extent-mismatch guards for sibling nests (`guard_annotated_extents`).
- **Structural safety**: `validate_parallel` (`low_level.ml:2574`) rejects slot
  overflow, barriers under divergence, and materialized writes not covering every
  active hardware dimension.

**Two decisions in this proposal are superseded:**

1. *Integration point* (Phase 1): tiling is **not** applied in `assignments.ml`'s
   `loop_accum`. The landed seam is `?lowered_transform` on `Context.compile`
   (`backend_intf.ml:198`, `backends.ml:492`): a pure
   `Low_level.optimized -> Low_level.optimized` applied after `optimize_proc`
   (virtualize → simplify → CSE → hoisting) and before backend validation/rendering.
   Projection information is not needed there: parallel (output) vs. reduction
   (contracted) loops are recoverable from the lowered IR — a loop is parallelizable
   iff its index appears in the index vector of every materialized `Set` beneath it,
   the same property `validate_parallel` enforces. Scheduling post-optimization also
   means inlined elementwise epilogues get tiled together with the contraction for
   free.
2. *Phase ordering across backends*: implementation order is **Metal first**, then cc's
   serial fallback, then CUDA (CI-only here) — the CUDA-centric phase list below
   predates that decision; treat it as a catalogue of transforms and benchmark targets,
   not a sequence.

**Remaining work** — the transform layer, now specified in
[schedule-ir-optops](../../../docs/design/schedule-ir-optops.md) (elaborated 2026-07-04), which supersedes
Phases 1–5 as the implementation plan; this document remains the requirements and
benchmark-criteria reference:

- Split/Swap/Retype/Unroll transforms + the default GPU annotator preset (subsumes
  Phases 1–2; the annotator is what makes *every* kernel stop running single-threaded).
- `Stage` (shared-tile staging with cooperative loads) — subsumes Phase 3.
- Register blocktiling via Split + IR-level Unroll + existing CSE — subsumes Phase 4.
- CPU tiling/packing presets — subsumes Phase 5 (OpenMP/`Cpu_parallel` still deferred).
- Tiling-parameter configuration and the size-threshold fallback (unchanged ACs below).

## Status update (2026-06-12)

- Issue #412 is OPEN, milestone v0.8 ("GPU tiling and related optimizations in the polyhedral style"). ROADMAP.md still targets v0.8 for mid-June 2026 with tiling as the lead item — this proposal remains the plan of record.
- None of the five phases has started: the single-thread guard (`kernel_prep_line`, now `cuda_backend.ml:317-318`) and the hardcoded `grid_dim_x:1, block_dim_x:1` launch (now `cuda_backend.ml:970`) are still in place; Metal still dispatches one threadgroup (`metal_backend.ml:818-823`).
- The soft dependencies have resolved favorably: #351 (CSE after inlining) is CLOSED COMPLETED — cross-statement CSE with hoisting to a common ancestor scope landed in `low_level.ml`; #350 (loop-invariant hoisting) is CLOSED NOT_PLANNED (largely subsumed by the CSE hoisting work); #311 (`-march=native`) is CLOSED COMPLETED, so the CPU performance floor exists.
- #318 (megakernel deep dive) is CLOSED COMPLETED as an exploration; #411 (HIP backend) remains OPEN.
- Kernel parameters were renamed `params`→`kparams` / `Param_ptr`→`Kparam_ptr` (#356), which touches the `compile_proc`/launch plumbing this proposal extends for launch-configuration carrying.
- Line numbers in the Key Code Pointers table have been refreshed to the current tree (notable drift: `optimize_proc` is now at `low_level.ml:1619`, `pp_ll` at `c_syntax.ml:331`, `compile_proc` at `c_syntax.ml:842`); `assignments.ml`'s `loop_accum` (line 282) and `to_low_level` (line 190) are unchanged.
- Remaining work: all of Phases 1–5 plus the deferred follow-ups.

## Goal

Transform OCANNL from single-threaded GPU kernels into properly parallelized, tiled execution across CUDA, Metal, and C backends. This is the core of the v0.8 milestone ("GPU-style performance"). The work introduces loop tiling as an IR transformation, maps tiled loops to GPU thread/block indices for parallel execution, and adds shared memory (SMEM) cooperation and register blocktiling for memory hierarchy exploitation. The end state is that matrix multiplication (and general einsum operations with contraction dimensions) achieves substantial speedups by utilizing GPU parallelism and memory hierarchy -- moving from the current baseline of `grid_dim=1, block_dim=1` to properly configured multi-threaded kernels.

## Acceptance Criteria

- [ ] A loop tiling IR pass exists in `low_level.ml` that can split `For_loop` nodes into outer (tile) + inner (element) loop pairs, parameterized by tile size
- [ ] CUDA kernels launch with non-trivial `grid_dim` and `block_dim` computed from tiling parameters and tensor dimensions (the `kernel_prep_line` single-thread guard is removed or conditioned)
- [ ] For parallelizable (non-contracted) dimensions, outer tile loops are mapped to `blockIdx` and inner elements to `threadIdx`, with correct index arithmetic in the generated code
- [ ] CUDA code generation can emit `__shared__` memory declarations and `__syncthreads()` barriers for cooperative tile loading
- [ ] Register blocktiling: each GPU thread computes a small TM x TN tile of the output, accumulating in register-allocated arrays, with inner loops reading from shared memory
- [ ] C backend emits tiled loops with configurable tile sizes for cache-friendly execution of contraction operations
- [ ] Matrix multiplication performance improves measurably vs the single-threaded baseline (target: >10x for matrices of size >= 256x256)
- [ ] Small tensors (below a configurable threshold) continue to use the current single-threaded path, avoiding tiling overhead
- [ ] No regression in existing tests (the tiling pass is opt-in or automatically bypassed for small/non-tileable operations)
- [ ] The tiling parameters (BM, BN, BK, TM, TN) are configurable via `Utils.settings` or a dedicated tiling config, with sensible defaults

## Context

### Current Architecture

**The single-threaded baseline.** *(Superseded 2026-07-04: the guard and hardcoded 1×1
launches are gone — launch dims now come from `Low_level.launch_dims` on all backends.
Kernels still run single-threaded in practice only because no pass annotates loops yet.)*
Original state for reference:
- `cuda_backend.ml`: `kernel_prep_line = "/* FIXME: single-threaded for now. */if (threadIdx.x != 0 || blockIdx.x != 0) { return; }"` (now `""`, line 362)
- `cuda_backend.ml`: `S.launch_kernel func ~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0 stream.runner args` (now launch-dims-driven, line 1062)
- `metal_backend.ml`: dispatched with `threadgroups_per_grid: {1,1,1}` and `width = min max_threads 1` (now real dispatch, lines 929-933)

**The IR-to-code pipeline.** Operations flow through:
1. **Shape system** (`shape.ml`): einsum specs with batch/output/input dimensions. "Input" dims are the contracted (reduction) dimensions in an einsum; "batch" and "output" are non-contracted.
2. **Projections** (`indexing.ml` lines 136-157): `projections.product_space` and `product_iterators` define the iteration space. `project_lhs` and `project_rhs` map product iterators to tensor indices.
3. **Assignments** (`assignments.ml`): `Accum_op` with accumulator (`accum: Ops.binop`), LHS, RHS, and projections. `to_low_level` (line 190) converts to nested `For_loop` nodes by iterating over `product_space` dimensions.
4. **Low-level IR** (`low_level.ml`): `For_loop { index; from_; to_; body }` nodes, `Set`/`Get`/`Local_scope` for scalar computation. `optimize_proc` (line 1619) runs virtualization, simplification, and CSE *(Update 2026-06-12: CSE now includes cross-statement hoisting to a common ancestor scope, from #351)*.
5. **C syntax emission** (`c_syntax.ml`): `pp_ll` (line 331) converts `For_loop` to C `for` loops. `compile_proc` (line 842) assembles the full kernel function.
6. **Backend launch** (`cuda_backend.ml`): NVRTC compilation, module loading, kernel launch.

**Key observation for tiling.** The `product_space` in projections already separates the iteration dimensions. In a matrix multiply `C[i,j] += A[i,k] * B[k,j]`:
- Batch dims: none (or outer batch axes)
- Output dims: `i`, `j` (appear in `project_lhs`, do NOT reduce)
- Input (contracted) dims: `k` (appears in `project_rhs` for both A and B, reduces via accumulation)

The tiling strategy maps directly:
- Output dims (`i`, `j`) -> tiled and parallelized across `blockIdx` / `threadIdx`
- Contracted dims (`k`) -> tiled sequentially within each thread, with cooperative SMEM loading

### Key Code Pointers

*(Refreshed 2026-07-04.)*

| Location | Description |
|----------|-------------|
| `arrayjit/lib/low_level.ml:42,59` | `axis_type` and `For_loop` with the `axis` field |
| `arrayjit/lib/low_level.ml:79,83` | `Workgroup_barrier` and the `If` guard statement |
| `arrayjit/lib/low_level.ml:222-227` | `optimized` record type, incl. `workgroup_shared` |
| `arrayjit/lib/low_level.ml:2498-2672` | `hardware_axes`, `launch_dims`, `validate_parallel`, `guard_annotated_extents` |
| `arrayjit/lib/low_level.ml:3109` | `optimize_proc` -- virtualize → simplify → CSE → `hoist_cross_statement_cse` |
| `arrayjit/lib/low_level.ml:3448,3467` | `loop_over_dims` / `unroll_dims` |
| `arrayjit/lib/assignments.ml:311,490` | `loop_accum` / `loop_accum_rev` -- projection-driven lowering |
| `arrayjit/lib/indexing.ml:139-146,172-189` | `axis_index` (incl. `Affine` -- Split's substitution target); `projections` |
| `arrayjit/lib/c_syntax.ml:73-84` | `C_syntax_config` hooks: `hardware_index`, `barrier_syntax`, `shared_decl_prefix` |
| `arrayjit/lib/c_syntax.ml:475,484` | `pp_ll`; its `For_loop` case branches on `axis` (hardware binding at 527) |
| `arrayjit/lib/c_syntax.ml:1134-1354` | `compile_proc` -- validates, injects guards, returns launch dims |
| `arrayjit/lib/backend_intf.ml:198`, `backends.ml:486-492` | the `?lowered_transform` seam -- where the tiling/schedule pass plugs in |
| `arrayjit/lib/cuda_backend.ml:334-390,1062` | `Cuda_syntax_config` (incl. `hardware_index` at 372); launch-dims-driven launch |
| `arrayjit/lib/builtins_cuda.ml` | CUDA builtins: vector types (`float4_t`, `half8_t`), FMA, conversions |
| `arrayjit/lib/metal_backend.ml:445-454,854,929-933` | `gid`/`lid` extra args + `hardware_index`; block-product validation; real dispatch |
| `test/operations/hardware_axes_parity.ml`, `hardware_workgroup_reduce.ml` | parity-test templates for annotated kernels |
| `ROADMAP.md` lines 118-150 | v0.8 milestone definition |

### Related Work

- **gh-ocannl-351** (CSE after inlining): Should ideally run before tiling to simplify the IR. *(Update 2026-06-12: landed — closed COMPLETED; cross-statement CSE with hoisting is in `low_level.ml`.)*
- **gh-ocannl-350** (Loop invariant hoisting): Same soft dependency as CSE. *(Update 2026-06-12: closed NOT_PLANNED — largely subsumed by the CSE hoisting work.)*
- **gh-ocannl-318** (Megakernels): Complementary -- megakernels fuse multiple operations; tiling optimizes within a single operation. Both are v0.8 targets. *(Update 2026-06-12: the #318 exploration is closed COMPLETED.)*
- **gh-ocannl-411** (HIP backend): HIP tiling follows the same patterns as CUDA tiling. *(Still open.)*
- **gh-ocannl-311** (Add `-march=native`): CPU performance floor for the C backend. *(Update 2026-06-12: landed — closed COMPLETED.)*

### Reference Articles

The optimization techniques are well-documented in the articles collected in issue #412:
- [Böhm: Fast CPU MMM](https://siboehm.com/articles/22/Fast-MMM-on-CPU) -- tiling, register blocking, cache hierarchy for CPU
- [Böhm: CUDA MMM](https://siboehm.com/articles/22/CUDA-MMM) -- global memory coalescing, SMEM tiling, 2D register blocktiling, vectorized loads, warptiling
- [Seb-v: Fast GPU MatMul](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html) -- CUDA optimization walkthrough
- [ArmbR: Tensor Cores from Scratch](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) -- WMMA/MMA intrinsics
- [CUDAForFun: Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) -- advanced tiling + tensor cores
- [gau-nernst: Blackwell tcgen05](https://gau-nernst.github.io/tcgen05/) -- next-gen tensor core architecture

## Approach

> **Superseded as a plan (2026-07-04).** The five phases below are kept as the
> catalogue of transforms and benchmark targets; the implementation plan is now
> [schedule-ir-optops](../../../docs/design/schedule-ir-optops.md) (Phases S1–S4), operating at the
> `?lowered_transform` seam with Metal-first backend ordering. Mapping: Phase 1 → S1
> `Split` (as a schedule transform, not a `loop_accum` change); Phase 2 → landed
> infrastructure + S1's default annotator preset; Phase 3 → landed IR constructs + S2
> `Stage`; Phase 4 → S3 (Split + materializing Unroll + existing CSE); Phase 5 → S4.

The implementation is structured in five phases, each building on the previous and producing a testable, committable increment. Metal backend parallelism and tensor core support are deferred to follow-up work.

### Phase 1: IR-Level Loop Tiling Transform

**Goal:** Add a tiling pass to `low_level.ml` that structurally transforms `For_loop` nodes into tiled (outer + inner) loop pairs.

**Changes:**
- Add a new IR node variant or use existing `For_loop` with a tiling annotation. The simplest approach is to keep `For_loop` unchanged and have a separate `tile_loops` pass that rewrites the IR:
  ```
  For_loop { index=i; from_=0; to_=N-1; body }
    -->
  For_loop { index=tile_i; from_=0; to_=(N-1)/TILE; body =
    For_loop { index=i; from_=tile_i*TILE; to_=min((tile_i+1)*TILE-1, N-1); body } }
  ```
- The inner loop index `i` becomes an `Affine { symbols = [(1, tile_i); (1, inner_i)]; offset = 0 }` or the body's index references are rewritten to use `tile_i * TILE + inner_i`.
- Add tiling configuration to `Utils.settings` or a new `Tiling_config` module: `tile_sizes: (int * int * int)` for (BM, BN, BK), `register_tile: (int * int)` for (TM, TN), and `min_tile_threshold: int`.
- The `tile_loops` pass receives tiling annotations from the caller (based on projections analysis) specifying which loops to tile and with what parameters.
- Insert this pass into `optimize_proc` after existing simplification passes, or apply it in `assignments.ml` at `to_low_level` time when projection information is still available.

**Integration point decision:** Apply tiling in `assignments.ml`'s `loop_accum` function (line 282) where projection information is directly available, rather than as a post-hoc pass on the IR. This avoids the need to reverse-engineer which loops correspond to output vs contracted dimensions. The `product_space` array and `product_iterators` already distinguish parallelizable from reduction dimensions. The tiled loop structure can be generated directly:
- Outer tile loops for output dimensions -> will become block indices
- Inner tile loops for output dimensions -> will become thread indices
- Outer tile loop for contracted dimension -> sequential, drives SMEM tile loading
- Inner contracted loop -> sequential within each thread's register tile

**Testability:** Generate tiled C code for the C backend, verify correctness against non-tiled version.

### Phase 2: Multi-Threaded CUDA Kernels

**Goal:** Remove the single-thread guard and launch CUDA kernels with proper grid/block dimensions.

**Changes:**
- Modify `kernel_prep_line` in `Cuda_syntax_config` to be empty (or conditional on a per-kernel flag).
- Add a `Parallel_loop` variant to the IR (or annotate `For_loop` with a `parallel: parallel_kind option` field where `parallel_kind = Block_x | Block_y | Thread_x | Thread_y`). The code generator in `c_syntax.ml` maps these to:
  - `Block_x` -> loop variable replaced by `blockIdx.x`, no for-loop emitted
  - `Thread_x` -> loop variable replaced by `threadIdx.x`, no for-loop emitted
  - Sequential (no annotation) -> normal for-loop
- Compute `grid_dim_x`, `grid_dim_y`, `block_dim_x`, `block_dim_y` from the tiling parameters and pass them to `S.launch_kernel`. This requires extending the `code` record in `cuda_backend.ml` (or the `optimized` record) to carry launch configuration.
- The `launch_kernel` call in `cuda_backend.ml` line 979 changes from hardcoded `1, 1, 0` to values derived from tiling parameters.
- Element-wise operations (no contraction, all output dims) get a simple parallelization: one thread per output element (or a 1D tiled scheme).
- Operations with contraction dimensions get the full tiled scheme from Phase 1.
- **Fallback:** Operations below the `min_tile_threshold` use `grid_dim_x:1, block_dim_x:1` with the original single-threaded guard.

**Key concern:** The existing `pp_ll` in `c_syntax.ml` emits the same code for all backends. Parallel loop annotations need to emit different code for CUDA vs C. Options:
1. Add a `parallel_for_syntax` hook to `C_syntax_config` (preferred -- keeps the design modular).
2. Check backend type inside `pp_ll` (breaks the clean parameterization).

**Testability:** Launch simple element-wise operations with multiple threads, verify correctness. Launch tiled matmul kernels, verify correctness against CPU reference.

### Phase 3: Shared Memory Tiling for CUDA

**Goal:** Cooperative loading of input tiles into shared memory before computation.

**Changes:**
- Add IR constructs for shared memory:
  - `Shared_decl of { name: string; prec: Ops.prec; size: int }` -- declare a `__shared__` array
  - `Barrier` -- emit `__syncthreads()` (CUDA) / `threadgroup_barrier(...)` (Metal)
  - `Cooperative_load of { shared: string; global_tn: Tn.t; ... }` -- each thread loads its portion
  
  Alternatively, handle SMEM as a code generation concern rather than an IR concern: the tiling pass in `assignments.ml` generates the cooperative load pattern using existing `For_loop` / `Set` / `Get` nodes, with SMEM arrays treated as special `Tnode.t` instances with a `Shared` memory mode (extending the existing `Local` mode). This is more aligned with OCANNL's existing architecture.

- For a tiled matmul with tile size BM x BN x BK:
  1. Each thread block loads a BM x BK tile of A and a BK x BN tile of B into SMEM
  2. Threads cooperate on loading: each thread loads `(BM * BK) / (block_dim_x * block_dim_y)` elements
  3. `__syncthreads()` after loading
  4. Inner loop reads from SMEM instead of global memory
  5. `__syncthreads()` before loading the next K-tile

- Boundary handling: the last tile in each dimension may be smaller than the tile size. Guard global memory loads with bounds checks.

**Testability:** Compare SMEM-tiled matmul output against CPU reference for various sizes including non-power-of-2.

### Phase 4: Register Blocktiling

**Goal:** Each thread computes a TM x TN tile of output, accumulating in registers.

**Changes:**
- Within the innermost K-loop iteration, each thread:
  1. Loads TM elements from column of A's SMEM tile into a register array
  2. Loads TN elements from row of B's SMEM tile into a register array
  3. Computes the TM x TN outer product and accumulates in a TM x TN register array
- After the K-loop completes, writes the TM x TN register tile back to global memory.

- In the IR, this manifests as:
  - The thread-level inner loops (over TM, TN) are fully unrolled (using `unroll_dims` or generated as flat sequences)
  - Register arrays are `Local_scope` accumulators -- OCANNL already has this pattern
  - The block dimension is `(BM/TM) * (BN/TN)` threads per block

- Use FMA intrinsics (`fmaf` / `__fmaf_rn`) for the multiply-accumulate, leveraging existing builtins in `builtins_cuda.ml`.

**Testability:** Measure FLOPS for tiled+register-blocked matmul vs Phase 3 (SMEM only) vs Phase 2 (parallel only) vs baseline.

### Phase 5: C Backend Tiling and OpenMP

**Goal:** Cache-friendly blocked loops for the CPU backend, optionally with OpenMP parallelism.

**Changes:**
- The tiling IR from Phase 1 is emitted as nested C for-loops (already happens naturally via `pp_ll`).
- Tile sizes chosen to fit working sets in L1/L2 cache (BM = BN = BK = 64 or similar, tunable).
- Optionally emit `#pragma omp parallel for` on outer tile loops. This requires a new `C_syntax_config` hook or a backend-specific annotation on outer loops.
- The existing `-O3 -march=native` flags (from gh-ocannl-311) ensure the compiler auto-vectorizes the inner loops.

**Testability:** Benchmark tiled CPU matmul against non-tiled baseline for various sizes.

### Deferred (Follow-up Tasks)

- **Metal backend parallelism** (Phase 6 in task elaboration): Map tiled loops to `threadgroup_position_in_grid` and `thread_position_in_threadgroup`. Use `threadgroup` memory for SMEM equivalent. This is straightforward once the IR supports parallel annotations from Phase 2.
- **Tensor cores / WMMA** (Phase 7 in task elaboration): Requires detecting suitable matmul shapes, emitting `nvcuda::wmma` intrinsics, and SMEM swizzling. Significant additional complexity.
- **Autotuning** (v0.9 scope): Compile multiple kernel variants with different tile sizes, benchmark, cache winners. OCANNL's runtime compilation is naturally suited for this.
- **Vectorized memory access** (`float4` loads): Can be added after register blocktiling for additional throughput.
- **Warptiling**: Explicit warp-level tiling hierarchy for further optimization.

### Risk Mitigation

- **Correctness of reductions**: Tiled reductions must correctly initialize accumulators and handle partial tiles. Use the existing `Local_scope` accumulator pattern for register tiles. Add assertion checks comparing tiled vs non-tiled results for a test suite of operation shapes.
- **Small tensor regression**: Use a size threshold (configurable, default: total elements < 1024 or largest dim < 32) to fall back to the single-threaded path.
- **Non-matmul operations**: Element-wise operations get simple parallelization (Phase 2) without SMEM tiling. Reductions without contraction dimensions get a parallel reduction pattern. Only operations with contraction dimensions (detected via projections) get the full SMEM + register tiling.
- **SMEM limits**: Query `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK` at device initialization. Tile sizes must respect `2 * BM * BK * sizeof(float)` plus `2 * BK * BN * sizeof(float)` <= SMEM limit. Clamp tile sizes if necessary.
- **Register pressure**: Large TM * TN register tiles reduce GPU occupancy. Default TM = TN = 8 is a good balance. The v0.9 autotuner can search this space.
- **Interaction with CSE/hoisting**: Tiling should be applied after virtualization/inlining but can precede or follow CSE. The tiling pass in `assignments.ml` runs before `optimize_proc`, so CSE will deduplicate within the tiled structure.

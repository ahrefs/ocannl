# Deep Dive: The Megakernel Approach and Lessons for OCANNL

GitHub issue: https://github.com/ahrefs/ocannl/issues/318

## Motivation

OCANNL's compilation model maps one `Assignments.comp` (computation graph) to one routine to one GPU kernel. As lukstafi noted on the issue: "megakernel = routine in OCANNL terminology." Kernel splitting into smaller kernels is not implemented and "might never be for some hardware targets" (ROADMAP.md). This means OCANNL is **inherently megakernel by default**.

Recent work from Hazy Research (Stanford) and CMU/NVIDIA (Mirage project) demonstrates that the megakernel approach -- fusing entire model forward passes into single persistent GPU kernels -- is a winning strategy for LLM inference, achieving 1.5-3.5x latency improvements over separate-kernel baselines.

This document synthesizes the state of the art in megakernel design and maps the findings to OCANNL's architecture, identifying what OCANNL already does well, what it lacks, and which ideas are most transferable.

## State of the Art

### Hazy Research: Tokasaurus Megakernel

Hazy Research demonstrated fusing the entire Llama-1B forward pass into a single GPU kernel, achieving 78% memory bandwidth utilization on H100 (vs. ~50% for vLLM/SGLang) and sub-1ms forward passes.

**Architecture: On-GPU Interpreter.** The megakernel receives a pre-compiled instruction sequence via an "instruction tensor" in GPU global memory. Each streaming multiprocessor (SM) fetches and executes instructions from this sequence. Seven fused operation types cover a full transformer layer: RMS norm + QKV projection + RoPE, attention, attention reduction, output projection + residual, RMS norm + gated MLP + SiLU, down-projection + residual, and final norm + LM head.

**Shared memory paging.** The H100's 213KB shared memory per SM is divided into 13 pages of 16KiB each. Instructions explicitly request and release pages; the interpreter reassigns freed pages to subsequent instructions. This enables weight loading for instruction N to begin as soon as instruction N-1 frees pages, eliminating inter-instruction delays.

**Counter-based synchronization.** A global memory counter array tracks instruction completion. Each instruction increments its counter upon finishing; dependent instructions poll until counters reach target values. This replaces CUDA's coarse-grained inter-kernel barriers with fine-grained, within-kernel dependency tracking. For example, MLP intermediate states are produced and consumed in four chunks with independent counters, so the down-projection begins as soon as its specific input chunk completes -- not when all hidden states are ready.

**Cross-operator pipelining.** Weight loading for instruction N overlaps with instruction N-1's computation and stores. On B200, this reduces per-instruction transition gaps from 10.2us to 3.4us.

**Multi-GPU scaling.** The tensor-parallel variant for Llama-70B uses dedicated storer threads for inter-GPU communication via direct remote memory writes, hiding network latency behind computation. A global work queue dynamically assigns instructions via atomic increments, adapting to runtime jitter. This achieved >22% throughput improvement over SGLang on ShareGPT benchmarks.

**Performance summary:**
- H100: <1ms per Llama-1B forward pass (batch size 1)
- B200: <680us per forward pass (~1,666 passes/second; theoretical limit ~3,000)
- Gap attributed to activation load latencies that cannot be hidden behind weight pipelining

### Mirage Persistent Kernel (MPK)

MPK is a compiler and runtime from CMU/UW/Berkeley/NVIDIA that automatically transforms multi-GPU LLM inference into megakernels. It achieves 1.0-1.7x speedup over vLLM/SGLang on A100/H100/B200 GPUs.

**SM-level task graph (ttGraph).** Instead of representing computation as a graph of kernels, MPK decomposes it into tasks assigned to individual SMs. Each task denotes a unit of computation or communication on a single SM. Dependencies between tasks are represented by lightweight events. This granularity exposes parallelism impossible in kernel-per-operator models.

**Event-driven synchronization.** Events activate when all prerequisite tasks have completed. The compiler performs event fusion (successor-set and predecessor-set fusion) to eliminate redundant synchronization. After fusion, a normalization pass ensures each task has exactly one dependent/triggering event.

**Compiler pipeline:**
1. Operator decomposition: partition operator outputs into per-SM tasks
2. Dependency analysis: enumerate task pairs, introduce events for overlapping data regions
3. Event fusion: eliminate redundant synchronization
4. ttGraph normalization: one event per task
5. Linearization: breadth-first ordering for compact fan-out encoding

**In-kernel runtime.** SMs partition into workers and schedulers:
- Workers: loop of dequeue-execute-notify
- Schedulers: maintain event queues, dispatch tasks when dependencies are satisfied
- Allocation is static at kernel launch, matching physical SM count

**Hybrid JIT/AOT dispatch.** The compiler classifies operators by execution predictability. Data-dependent operations (e.g., attention with variable sequence lengths) use JIT dispatch for dynamic load balancing. Predictable operations use AOT pre-enqueuing for lower dispatch latency.

**Paged shared memory.** Shared memory is partitioned into fixed-size pages. Tasks acquire and release pages on demand, enabling cross-task software pipelining: the next task's data prefetching begins while the current task computes.

**Multi-GPU via NVSHMEM.** AllReduce decomposes into inter-GPU data-transfer tasks (via `nvshmem_signal_wait_until`) and local reduction tasks, transforming synchronous collective communication into fully asynchronous tasks overlapped with computation.

**Performance:** 1.0-1.7x single-GPU speedup; 1.1-1.4x on 8-GPU tensor parallelism. Cross-task pipelining alone yields 1.2-1.3x; compute-communication overlap provides 1.1x.

## OCANNL's Current Compilation Model

OCANNL's pipeline from computation graph to GPU kernel:

```
Assignments.comp (high-level)
  --> Assignments.to_low_level (lowering)
  --> Low_level.optimize_proc (visit → virtualize → cleanup → simplify)
  --> C_syntax.compile_proc (C/CUDA/Metal code generation)
  --> Backend.compile / compile_batch (compilation to PTX/object)
  --> Backend.link / link_batch (loading onto device)
  --> Task.t (executable unit scheduled on a stream)
```

Key files:
- `arrayjit/lib/assignments.ml` -- `comp` type (lines 85-90), `to_low_level` (line 190)
- `arrayjit/lib/low_level.ml` -- IR types (lines 33-65), optimization pipeline `optimize_proc` (line 1238)
- `arrayjit/lib/c_syntax.ml` -- `compile_proc` generates a single C/CUDA/Metal function per routine
- `arrayjit/lib/cuda_backend.ml` -- kernel launch at `grid_dim_x:1, block_dim_x:1` (line 979), `kernel_prep_line` guard (line 328-329)
- `arrayjit/lib/metal_backend.ml` -- dispatch with `threadgroups_per_grid: {1,1,1}` (line 793)
- `arrayjit/lib/backend_intf.ml` -- `routine` type (lines 51-61)

**What OCANNL already does (megakernel-aligned):**

1. **One comp = one kernel.** Each `Assignments.comp` becomes a single routine/kernel. All operations within the computation graph execute without returning to the host. This IS the megakernel paradigm at the level of individual computation subgraphs.

2. **Virtualization (fusion).** The `virtual_llc` pass inlines tensor computations accessed exactly once, eliminating intermediate memory allocations. This is analogous to operator fusion in megakernel systems, though at scalar/element granularity rather than tile granularity.

3. **Single-function code generation.** `C_syntax.compile_proc` generates one function containing all loop nests and computations. There is no kernel-splitting pass.

**What OCANNL lacks compared to state-of-the-art megakernels:**

1. **Single-threaded execution.** CUDA kernels launch with `grid_dim=1, block_dim=1` (line 979). The `kernel_prep_line` guard (line 328-329) reads: `"if (threadIdx.x != 0 || blockIdx.x != 0) { return; }"`. Only one GPU thread executes the entire kernel. All megakernel benefits (within-kernel parallelism, SM-level task distribution, pipelining) require multi-threaded execution as a prerequisite.

2. **No within-kernel synchronization.** Operations execute linearly within a routine. There are no barriers, counters, or events for intra-kernel coordination. All synchronization is host-mediated via stream events (see `backends.ml` synchronization section, `backend_intf.ml` lines 91-116).

3. **No cross-operator pipelining.** Independent operations within a single routine cannot overlap. Layer N's GEMM cannot overlap with layer N-1's activation because execution is strictly sequential.

4. **No paged shared memory.** Shared memory is not used (since only one thread runs). There is no dynamic allocation or page recycling.

5. **No SM-level scheduling.** There is no mechanism to distribute work across SMs within a single kernel. The entire kernel runs on one SM.

6. **No instruction-level dispatch.** There is no on-GPU interpreter or instruction tensor mechanism. The kernel is a static sequence of operations.

## Detailed Comparison

| Aspect | OCANNL (current) | Hazy Research | Mirage MPK |
|---|---|---|---|
| **Fusion granularity** | One `comp` = one kernel | Entire forward pass = one kernel | Entire multi-GPU inference = one kernel/GPU |
| **Thread parallelism** | 1 thread, 1 block | All SMs, warp-specialized | All SMs, worker/scheduler split |
| **Synchronization** | Host-mediated (stream events) | Global counter arrays | Event-driven with fusion |
| **Scheduling** | Sequential within routine | Instruction interpreter | ttGraph + JIT/AOT hybrid |
| **Shared memory** | Not used | Paged (13x16KiB on H100) | Paged with recycling |
| **Operator fusion** | Scalar virtualization | Fused instruction types | SM-level task decomposition |
| **Pipelining** | None | Weight load overlaps compute | Cross-task prefetching |
| **Multi-GPU** | Not supported | Direct remote writes | NVSHMEM async tasks |
| **Code generation** | Ahead-of-time C/CUDA | Hand-optimized CUDA | Compiler + Mirage superoptimizer |
| **Backend targets** | CUDA, Metal, C | CUDA (H100/B200) | CUDA (A100/H100/B200) |

## Transferable Ideas for OCANNL

### Tier 1: Prerequisites (must come first)

**P1. Multi-threaded kernel execution.** The single most critical prerequisite. All megakernel optimizations require parallel threads. This means:
- Remove the `kernel_prep_line` single-thread guard
- Generate parallel loop structures: map outer loops to thread blocks and threads
- Handle thread-safe accumulation (atomic adds or reduction trees for shared accumulators)
- This is largely the domain of the tiling/parallelization work planned for v0.8 (ROADMAP issue #412)

Without P1, none of the megakernel techniques apply. The v0.8 GPU tiling milestone is therefore the natural entry point.

**P2. Thread block synchronization primitives.** Once kernels use multiple threads, OCANNL's code generator needs to emit:
- `__syncthreads()` barriers (CUDA) / `threadgroup_barrier(mem_flags::mem_threadgroup)` (Metal)
- Atomic operations for shared accumulators
- These are standard and well-understood; the code generation in `c_syntax.ml` needs new `Low_level.t` constructs to represent barriers and atomics

### Tier 2: High-value optimizations (aligned with v0.8-v0.9)

**T1. Counter-based intra-kernel synchronization.** After P1-P2, add support for global memory counters that enable fine-grained dependency tracking within a single kernel. This would allow independent operations (e.g., different attention heads, or MLP down-projection on completed chunks) to proceed without waiting for unrelated operations. Implementation:
- Add a `Sync_counter` construct to `Low_level.t` (increment, wait-until)
- Dependency analysis in the lowering phase identifies independent operations
- Code generation emits atomic counter operations
- CUDA: `atomicAdd` + spin-wait on global memory
- Metal: `atomic_fetch_add_explicit` + spin on device memory

**T2. Shared memory utilization.** Shared memory (CUDA) / threadgroup memory (Metal) is the key to performance. Two levels:
- **Static allocation:** For tiled matrix multiplications, allocate shared memory tiles per thread block. This is standard and aligns with the v0.8 tiling work.
- **Paged allocation:** For larger routines with multiple phases, divide shared memory into pages that are acquired/released as operations proceed. This enables the next operation's data loading to overlap with the current operation's computation.

Implementation in OCANNL:
- Add `Shared_mem_alloc` and `Shared_mem_free` constructs to the IR (or manage pages implicitly during code generation)
- The `Local` memory mode in `Tnode` (`memory_mode = Local`) could be extended to distinguish "local to thread" vs. "shared across threadgroup"
- Metal: use `threadgroup` address space; CUDA: use `__shared__` declarations

**T3. Cross-operator pipelining within a routine.** When a routine contains multiple independent operations (e.g., different layers, independent projections), overlap their execution:
- **Weight pipelining:** Begin loading weights for the next operation while the current operation computes. Requires asynchronous memory copies (`cp.async` on CUDA, explicit threadgroup loads on Metal).
- **Computation overlap:** If operation B does not depend on operation A, interleave their execution across different thread groups.
- Implementation: dependency analysis in `optimize_proc` identifies independent subgraphs within a routine; code generation interleaves their loop nests

### Tier 3: Advanced optimizations (v0.9+ / post-1.0)

**A1. SM-level task graph.** For very large routines (full transformer layers), decompose into tasks assigned to specific SMs. Each SM independently processes its task queue. This is Mirage MPK's core contribution.
- Requires a new IR representation (task graph) above the current `Low_level.t`
- Each task maps to a portion of the computation assigned to specific thread blocks
- An in-kernel dispatcher (simple loop with task queue) replaces static code flow
- This is a major architectural change and should only be pursued after T1-T3 prove valuable

**A2. Persistent kernel for training loops.** Instead of launching a routine's kernel once per invocation, launch it as a persistent kernel that loops, accepting work items from a host-side queue. This amortizes kernel launch overhead across iterations.
- Implementation: wrap the generated kernel body in a `while (true)` loop that reads from a command buffer
- The host enqueues work items (parameter pointers, iteration counts) via mapped memory
- Exit via a sentinel command
- This is most valuable for training where the same kernel runs thousands of times

**A3. On-GPU instruction interpreter.** Instead of compiling a static kernel, generate an interpreter that reads an instruction sequence from global memory. This enables dynamic computation graphs and runtime recompilation without kernel relaunch.
- This is Hazy Research's core approach
- Very powerful but architecturally different from OCANNL's ahead-of-time compilation model
- May conflict with OCANNL's design philosophy of compile-time optimization
- Worth investigating as an optional execution mode for inference workloads

## Implications for OCANNL's Roadmap

### The megakernel approach validates OCANNL's design

OCANNL's "one comp = one kernel" model is aligned with the direction the GPU computing community is moving. The state of the art says: **do not split into many kernels; instead, make one big kernel smarter.** This validates OCANNL's decision to not implement kernel splitting and suggests that effort should go toward making the single kernel more parallel and better at utilizing GPU resources.

### Recommended priority ordering

| Priority | Item | Milestone | Rationale |
|---|---|---|---|
| 1 | Multi-threaded kernels (P1) | v0.8 | Prerequisite for everything else; already planned as tiling work |
| 2 | Thread synchronization (P2) | v0.8 | Required companion to P1 |
| 3 | Shared memory tiling (T2, static) | v0.8 | Standard optimization with large payoff |
| 4 | Counter-based sync (T1) | v0.9 | Enables intra-kernel parallelism beyond tiling |
| 5 | Paged shared memory (T2, dynamic) | v0.9 | Enables cross-operation pipelining |
| 6 | Cross-operator pipelining (T3) | v0.9 | Requires T1 + T2 |
| 7 | Persistent kernel (A2) | Post-1.0 | Amortizes launch overhead for training |
| 8 | SM-level task graph (A1) | Post-1.0 | Major architectural change |
| 9 | On-GPU interpreter (A3) | Post-1.0 | Alternative execution model |

### Kernel splitting: deprioritize

The megakernel literature strongly suggests that kernel splitting should be deprioritized in favor of making single kernels more efficient. The only scenario where splitting is clearly beneficial is when register pressure causes severe spilling -- and even then, the megakernel approach addresses this through SM-level task decomposition (where each task uses fewer registers) rather than host-level kernel splitting. OCANNL should monitor register usage in compiled kernels (via `--ptxas-options=-v` in NVRTC) and only consider splitting if register spilling is measured to be a bottleneck.

### Metal-specific considerations

Metal's compute model differs from CUDA in ways that affect megakernel applicability:
- **No cooperative groups:** Metal uses `threadgroup_barrier` for within-threadgroup synchronization but lacks cross-threadgroup synchronization within a single dispatch. Counter-based synchronization (T1) works via device-memory atomics, but is less well-supported.
- **Tile memory:** Metal 3 (Apple Silicon M3+) introduces tile functions and tile memory, which provide a different mechanism for shared memory management. This could serve as Metal's analog to CUDA's paged shared memory.
- **SIMD groups:** Metal's SIMD groups (analogous to CUDA warps) support shuffle and reduction operations. These are useful for intra-warp communication in tiled kernels.
- **No persistent kernels:** Metal does not support persistent kernels in the same way as CUDA. The closest equivalent is repeated `dispatchThreadgroups` calls within a single command buffer, which still incurs per-dispatch overhead.

### Relationship to other planned work

- **Loop hoisting (#350):** Complementary. LICM optimizes the IR; megakernel techniques optimize GPU execution. Both improve the same kernel.
- **CSE (#351):** Complementary. Reducing redundant computation within a megakernel makes each thread's work more efficient.
- **Pool allocator (#344):** The paged shared memory concept is an in-kernel analog of pool allocation. Design insights may transfer.
- **Program search (#140, v0.9):** Within-kernel scheduling decisions (which thread block executes which operation, how shared memory pages are allocated) are search problems. The v0.9 program search infrastructure could optimize these parameters.
- **Tiling (#412):** Direct prerequisite. The tiling work for v0.8 should be designed with megakernel evolution in mind -- e.g., using shared memory paging from the start rather than static allocation.

## Conclusion

OCANNL is architecturally aligned with the megakernel trend: one computation graph compiles to one kernel. The gap is in GPU utilization -- OCANNL runs one thread where megakernels run all SMs with fine-grained synchronization and pipelining. The v0.8 tiling milestone is the natural bridge: it introduces multi-threaded kernels and shared memory usage, which are the prerequisites for all megakernel optimizations.

The key insight from this research is that OCANNL should **not** pursue kernel splitting. Instead, the path forward is: multi-threaded kernels (v0.8) -> within-kernel synchronization and pipelining (v0.9) -> SM-level task graphs and persistent kernels (post-1.0). Each step builds on the previous one and moves OCANNL closer to state-of-the-art GPU utilization while preserving its core "one comp = one kernel" design.

## References

- [Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) -- Hazy Research, May 2025
- [We Bought the Whole GPU, So We're Damn Well Going to Use the Whole GPU](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main) -- Hazy Research, September 2025 (tensor-parallel Llama-70B megakernel)
- [Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs](https://arxiv.org/abs/2512.22219) -- arXiv:2512.22219 (MPK paper)
- [Mirage project](https://github.com/mirage-project/mirage) -- GitHub repository

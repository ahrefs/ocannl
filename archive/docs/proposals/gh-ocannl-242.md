# TVM Deep Dive: Architecture Comparison and Transferable Ideas for OCANNL

**Issue:** [ahrefs/ocannl#242](https://github.com/ahrefs/ocannl/issues/242)
**Status:** Draft proposal

## Motivation

TVM (Apache TVM) is listed in the OCANNL README as one of the inspirational projects alongside tinygrad and Luminal. It is an end-to-end deep learning compiler stack with automatic optimization via schedule search (AutoTVM/Ansor). A systematic comparison of TVM's approach to OCANNL's can identify transferable ideas, validate existing design decisions, and guide the v0.8 GPU-performance milestone.

GeoHot mentioned TVM inspired some TinyGrad solutions. Since OCANNL also draws from tinygrad, understanding TVM's original ideas is valuable.

## Current State

### OCANNL's compilation pipeline

OCANNL compiles tensor computations through a well-defined pipeline:

1. **Tensor expressions** (`%op` syntax extension) define differentiable computations with shape inference.
2. **Assignments** (`Assignments.t`) provide high-level operations: `Accum_op`, `Set_vec_unop`, `Fetch`.
3. **Lowering** (`Assignments.to_low_level`) translates to `Low_level.t`, a C-like imperative IR with `For_loop`, `Set`, `Get`, scalar operations, and affine index expressions.
4. **Optimization** (`optimize_proc`): tracing-based visit counting, virtualization (inlining), cleanup, and algebraic simplification.
5. **Backend code generation**: `c_syntax.ml` provides a shared C-like code emitter parameterized by backend-specific configuration (`C_syntax_config`). CUDA, Metal, and CC backends plug into this.

Key files: `arrayjit/lib/low_level.ml` (IR), `arrayjit/lib/assignments.ml` (high-level ops), `arrayjit/lib/c_syntax.ml` (shared codegen), `arrayjit/lib/cuda_backend.ml`, `arrayjit/lib/metal_backend.ml`, `arrayjit/lib/cc_backend.ml`.

### TVM's compilation pipeline

TVM compiles through a parallel but more layered pipeline:

1. **Relay/Relax** (graph-level IR): whole-model representation with operator fusion, constant folding, and layout optimization.
2. **Tensor Expression (TE)**: compute definitions analogous to einsum -- `C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))`.
3. **Schedule**: a sequence of transformations (split, tile, reorder, vectorize, unroll, bind, compute_at, compute_inline) applied to the TE's loop nest, producing the final loop structure.
4. **TIR (Tensor IR)**: the lowered, loop-level IR -- structurally similar to OCANNL's `Low_level.t`.
5. **Code generation**: TIR is compiled to CUDA, Metal, LLVM IR, WebGPU SPIR-V, etc.

## Proposed Change

This is a research task. The deliverable is a comparison document (this proposal serves as its outline) and a GitHub issue comment summarizing actionable findings. The research should cover the following areas, producing a final write-up that replaces or extends this proposal.

### 1. Schedule Primitives: TVM vs. OCANNL

TVM's schedule primitives operate on loop nests derived from the TE compute definition:

| TVM Primitive | Purpose | OCANNL Equivalent |
|---|---|---|
| `split(axis, factor)` | Divide a loop into outer/inner | No direct equivalent; loop structure is fixed at lowering time from `projections.product_space` |
| `tile(x, y, x_f, y_f)` | 2D split (blocking) | Not supported; planned for v0.8 (CPU tiling from siboehm's guide) |
| `reorder(axes...)` | Change loop nesting order | Loop order determined by `product_space` list order; no post-hoc reordering pass |
| `vectorize(axis)` | Map inner loop to SIMD | `Set_from_vec` with `vec_unop` provides vector operations, but no automatic vectorization pass |
| `unroll(axis)` | Unroll a loop | `trace_it` flag triggers partial unrolling during tracing; no general unroll transform |
| `parallel(axis)` | Map loop to threads | CUDA backend maps to grid/block dims (currently forced to `grid_dim=1, block_dim=1`) |
| `bind(axis, thread)` | Bind loop to GPU thread axis | Not supported; all CUDA kernels are single-threaded |
| `compute_at(tensor, axis)` | Place computation inside consumer's loop | Virtualization (inlining) achieves this for single-access tensors |
| `compute_inline` | Fully inline a computation | Virtual memory mode -- OCANNL's most developed scheduling mechanism |
| `cache_read/write` | Insert explicit staging buffers | No equivalent; planned Universal Pool Allocator (#344) is related |

**Key gap:** OCANNL has no post-lowering schedule transformation layer. Loop structure is determined at lowering time and then optimized only through virtualization and simplification. TVM's schedule primitives enable search over different loop structures for the same computation, which is the foundation for auto-tuning.

### 2. Auto-Tuning: AutoTVM and Ansor

TVM's auto-tuning searches the schedule space to find optimal configurations for each operator on each target hardware:

- **AutoTVM**: user writes schedule templates with tunable knobs (e.g., tile sizes); a cost model + evolutionary search explores the space.
- **Ansor (auto-scheduler)**: generates schedules automatically from the TE compute definition without user-written templates. Uses sketch-based generation and evolutionary search with a learned cost model.

OCANNL's README mentions "instead of dynamic scheduling as in tinygrad, we can schedule statically by program search." This aligns with Ansor's philosophy, but OCANNL currently has no search infrastructure. The v0.8 milestone introduces tiling, which would be the first point where schedule search becomes relevant.

**Transferable idea:** When OCANNL adds tiling (v0.8), it should design the tiling API to be parameterizable by tile sizes, making it amenable to later auto-tuning. This is cheaper to do upfront than to retrofit.

### 3. Tensor Expression IR vs. OCANNL Einsum

TVM's TE and OCANNL's einsum both define tensor computations declaratively:

- **TVM TE**: Python lambdas over index variables, with explicit reduction axes. Close to mathematical notation. Separate from the schedule.
- **OCANNL einsum**: OCaml syntax extensions (`%op`, `%cd`) with shape inference, axis kinds (batch/input/output), and dimension labels. The computation definition is inseparable from its lowering strategy.

TVM's separation of compute definition from schedule is a deliberate design choice enabling schedule search. OCANNL's tighter coupling trades search flexibility for simplicity and OCaml type safety.

### 4. Cross-Backend Code Generation

TVM's code generation uses TIR as the common lowered representation, with target-specific passes and code printers for each backend. OCANNL's `c_syntax.ml` similarly provides a shared C-syntax emitter parameterized by `C_syntax_config`.

The approaches are structurally similar. TVM has more backend targets but OCANNL's approach is adequate for its current scope (C, CUDA, Metal). TVM's target-specific intrinsics handling (e.g., `tvm.intrin` for hardware-specific operations like tensor cores) is something OCANNL would need when targeting specialized hardware units.

### 5. Graph-Level IR (Relay/Relax)

TVM's graph-level IR enables whole-model optimizations: operator fusion, layout transformation, constant folding, and quantization. OCANNL composes computations through OCaml's type system without an explicit graph IR.

**Assessment:** Adding a graph-level IR is a large architectural change with unclear benefit for OCANNL's current scale. OCANNL's compilation-unit model (user explicitly defines routine scope) already enables megakernel-style fusion (#318). A graph IR would be valuable only if OCANNL needs automatic fusion decisions, which is not on the current roadmap.

### 6. Memory Planning

TVM performs memory planning across the operator graph: buffer reuse, workspace allocation, and memory scope assignment (global, shared, local for GPU). OCANNL's memory modes (Virtual, Materialized, Device_only, Hosted) handle per-tensor decisions, and the planned Universal Pool Allocator (#344) would add cross-tensor buffer reuse.

**Transferable idea:** TVM's memory scope assignment (mapping tensors to GPU shared memory vs. global memory) is relevant for OCANNL's v0.8 GPU performance work. When adding tiling, shared memory staging is essential for achieving good performance.

### 7. TVM as a Backend Target

Could OCANNL target TVM's TE IR instead of generating C/CUDA directly?

**Assessment:** Technically possible but undesirable. OCANNL would need to serialize its `Low_level.t` to TVM's Python-based TE API (or the TVM C++ FFI), losing OCaml type safety and adding a large runtime dependency. The benefit (access to TVM's auto-tuning) is achievable more directly by adding search capabilities to OCANNL's own pipeline. The OCANNL-to-TVM translation would also be lossy -- OCANNL's affine index expressions and virtualization decisions don't map cleanly to TVM's TE.

## Scope

**In scope:**
- Architecture comparison covering the six areas above
- Identification of transferable ideas with assessment of effort and value
- Posting findings as a GitHub issue comment

**Out of scope:**
- Implementing any TVM-inspired changes (those should be tracked as separate issues)
- Benchmarking TVM against OCANNL
- Building TVM OCaml bindings

**Dependencies:**
- Related exploration tasks: gh-ocannl-261 (superoptimizers), gh-ocannl-267 (Tiramisu), gh-ocannl-306 (Petalisp/Caten)
- The v0.8 tiling milestone (#350 loop hoisting, CPU/CUDA tiling) is where TVM ideas become most actionable

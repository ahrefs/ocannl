# Petalisp and Caten Deep Dive: Transferable Ideas for OCANNL

**Issue:** [ahrefs/ocannl#306](https://github.com/ahrefs/ocannl/issues/306)
**Status:** Draft proposal

## Motivation

Petalisp and Caten are Common Lisp array/tensor compilers -- the closest peers to OCANNL's OCaml-based approach. Both share the "functional language as compiler host" philosophy and face similar challenges around optimization, code generation, and competing with mainstream frameworks. A systematic comparison can identify concrete transferable ideas, validate existing design choices, and inform OCANNL's v0.9 scheduling and optimization roadmap.

This continues the exploration series alongside the TVM deep dive (#242, completed) and DumPy/torchdim study (#316, proposal drafted). Where TVM represents a production-grade compiler stack and DumPy/torchdim represent frontend design alternatives, Petalisp and Caten represent **peer-scale research projects** building tensor compilers in non-mainstream functional languages -- making the comparison especially direct.

## Background: The Three Systems

### Petalisp -- Lazy Evaluation and Global Optimization

[Petalisp](https://github.com/marcoheisig/Petalisp) is a JIT-compiling array system by Marco Heisig (FAU Erlangen-Nurnberg), under development since 2016. 515 GitHub stars, AGPL-3.0, 1452+ commits. x86-64 CPUs only.

**Evaluation model:** Fully lazy. Users build a DAG of array operations using five primitives, and the system analyzes, optimizes, and JIT-compiles the entire graph only when results are demanded. This gives the compiler full visibility over the computation before code generation.

**Five primitive operations:**
1. `lazy` (application) -- pointwise function application across arrays
2. `lazy-reshape` (reference) -- affine index transformations: `(transform m n to n m)` for transpose, slicing, padding
3. `lazy-reduce` (reduction) -- aggregation along dimensions
4. `lazy-fuse` (fusion) -- combine multiple arrays into a unified kernel
5. `lazy-rep` (repetition) -- broadcasting by replication

All data-parallel computations compose from these five operations on **strided arrays** -- arrays defined by (start, end, step) triples per dimension. Reshaping is zero-copy: it produces a new view with a different affine index mapping.

**Optimization pipeline:**
1. Lazy DAG construction (no computation yet)
2. Graph analysis: identify connected components, shared subexpressions
3. Fusion: merge compatible operations into single kernels, eliminating intermediate buffers
4. Temporal blocking: for iterative algorithms (Jacobi, stencils), fuse multiple time steps to improve cache utilization
5. C code generation with SIMD intrinsics (AVX/AVX2/FMA via sb-simd)
6. JIT compilation via SBCL's native compiler

**Performance:** Single-threaded daxpy at 45-90% of likwid-bench; Jacobi 2D stencil at 36-89% of optimized C++; skinny matrix multiply outperforms OpenBLAS. Square matmul is a known weakness (naive reduction needs m*n*k/8 auxiliary storage). Multi-threaded execution experimental (6-thread daxpy at 32-70% of peak).

**Academic output:** Six publications at European Lisp Symposium (2018-2024) and ARRAY workshop at PLDI (2018), covering strided arrays, JIT compilation, SIMD, and loop optimization.

### Caten -- RISC Primitives and Polyhedral Compilation

[Caten](https://github.com/hikettei/Caten) (Compile+AbstracTENsor) is an experimental deep learning compiler by hikettei, started July 2024. 237 GitHub stars. **Development on hold** ("until I secure good sponsors or enough time"). Last commit January 2026.

**Design philosophy:** "As simple as tinygrad yet as flexible as TVM." Everything reduces to exactly 26 composable primitive operations (RISC-inspired), and the compiler applies polyhedral optimization via ISL before generating backend-specific code.

**The 26 AASM primitives:**
- UnaryOps (8): NEG, RECIP, SIN, EXP2, LOG2, SQRT, NOT, CAST
- BinaryOps (10): ADD, MUL, MOD, IDIV, AND, OR, XOR, MOVE, MAX, GCD
- TernaryOps (4): !=, <, WHERE, WMMA (fused matrix multiply-add)
- Buffer (4): ALLOCATE, LOAD, STORE, VIEW
- Indexing (1): INDEX-COMPONENTS
- JIT (1): SPACE (GPU thread/block mapping)

**6-layer compilation pipeline:**
1. **API Layer:** Lazy tensor ops with forward/backward/lower methods
2. **AIR Layer:** Graph-based DAG representation
3. **AASM Layer:** 26-op instruction set
4. **Codegen Layer:** Shape inference, rewriting rules, scheduling (BFS + fusion), blueprint (loops), polyhedral optimization via ISL, code generation
5. **Runtime Layer:** Graph execution, buffer management
6. **Backend Layer:** Clang JIT (most mature), Metal (Apple GPU), planned CUDA/LLVM/WebGPU

**ISL integration:** ~30 files of CFFI bindings to the Integer Set Library for polyhedral analysis -- domain/access/schedule construction, dependency analysis, AST generation. This is the most complete ISL binding from a high-level language outside C/C++/Python.

**Schedule-level fusion:** The scheduler groups compatible operations into fused kernels before code generation. One Schedule-Item = one GPU kernel -- a clean architectural invariant. Fusion examples: `LOAD->ADD->SIN->SIN` becomes `FUSED_SIN_SIN_ADD_LOAD_LOAD`.

**Working models:** GPT-2 inference, MobileNetV2, ResNet18/34/50. Supports ONNX import and GGUF quantized model loading.

### OCANNL -- Eager Compilation with Einsum-Based IR

OCANNL compiles tensor computations through:
1. **Tensor expressions** (`%op` syntax) with shape inference (`tensor/row.ml`, `tensor/shape.ml`)
2. **Assignments** (`arrayjit/lib/assignments.ml`): `Accum_op`, `Set_vec_unop`, `Fetch`
3. **Lowering** to `Low_level.t` (`arrayjit/lib/low_level.ml`): C-like imperative IR with `For_loop`, `Set`, `Get`, scalar ops, affine indices
4. **Optimization**: tracing-based visit counting, virtualization (inlining), cleanup, algebraic simplification (`optimize_proc` in `low_level.ml`)
5. **Code generation** via `c_syntax.ml` parameterized by backend config, targeting C (gcc/clang), CUDA, Metal

**Key characteristics:** Eager evaluation (only projections use `Lazy.t`); einsum-based tensor definition; fusion via virtualization/inlining at scalar granularity; affine indexing but no polyhedral scheduling; CUDA kernels currently single-threaded (`grid_dim=1, block_dim=1`).

## Architecture Comparison

| Aspect | OCANNL | Petalisp | Caten |
|---|---|---|---|
| **Language** | OCaml | Common Lisp (SBCL) | Common Lisp |
| **Evaluation** | Eager (lazy projections only) | Fully lazy | Fully lazy with checkpoints |
| **Primitive ops** | Einsum-based (variable) | 5 array operations | 26 scalar/buffer ops |
| **IR layers** | 3 (Assignments -> Low_level -> C_syntax) | Strided arrays -> kernel IR -> C | 6 (API -> AIR -> AASM -> Codegen -> Runtime -> Backend) |
| **Optimization** | Tracing -> virtualization -> inlining -> simplification | Global graph analysis -> fusion -> temporal blocking | Shape inference -> rewriting -> scheduling -> polyhedral (ISL) |
| **Polyhedral** | Affine indexing only (no scheduling synthesis) | Implicit (affine transforms, strided ranges) | Full ISL integration |
| **Scheduling** | Implicit in projections (baked at lowering) | Runtime (JIT, lazy) | Polyhedral + BFS fusion |
| **Fusion** | Virtual node inlining (scalar level) | Automatic graph-level fusion + temporal blocking | Schedule-level kernel fusion |
| **Autodiff** | Yes (forward + backward) | Yes (differentiator function) | Yes (forward/backward/lower methods) |
| **GPU targets** | CUDA, Metal | None (CPU only) | Metal (CUDA planned) |
| **CPU targets** | C via gcc/clang | C + SIMD (AVX/AVX2/FMA) | Clang JIT |
| **Training** | Yes (SGD optimizer) | Possible (autodiff exists) | Experimental |
| **Maturity** | Active, pre-1.0, ~1 developer | Research, 10 years, ~1 developer | Experimental, on hold, ~1 developer |

## Proposed Research Areas

### 1. Petalisp's Lazy Evaluation and Graph-Level Fusion

**What to study:** How Petalisp's fully lazy model gives the compiler global visibility over the computation graph before any code is generated. Specifically: how does it decide which operations to fuse? What cost model drives fusion decisions? How does it handle the tradeoff between fusion (fewer kernel launches, better locality) and code explosion (fused kernels become too large)?

**Relevance to OCANNL:** OCANNL's eager evaluation means each operation is compiled independently. The `compile_batch` / `link_batch` functions in `backends.ml` provide a limited form of batch compilation, but there is no global graph analysis. Petalisp demonstrates that lazy evaluation + global fusion can achieve significant performance gains, especially for iterative algorithms.

**Transferable insight:** OCANNL does not need to become fully lazy. A "deferred compilation" mode -- where the user builds a computation graph eagerly but defers lowering until a batch is complete -- would give the compiler the same global visibility. This is architecturally compatible with OCANNL's existing `compile_batch` infrastructure and aligns with the v0.8-v0.9 optimization roadmap.

### 2. Petalisp's Temporal Blocking

**What to study:** How Petalisp fuses multiple iterations of iterative algorithms (stencil computations, Jacobi iteration) to improve cache utilization. The README demonstrates that scheduling multiple daxpy runs in succession allows Petalisp to apply temporal blocking, outperforming single-pass kernels.

**Relevance to OCANNL:** Training loops in deep learning are inherently iterative. Temporal blocking across training steps (or across layers within a forward pass) could significantly improve cache utilization. OCANNL currently handles one operation at a time during lowering.

**Transferable insight:** Temporal blocking requires the compiler to see multiple time steps simultaneously. This reinforces the case for deferred compilation (Area 1). For OCANNL's v0.9 program search milestone, temporal blocking would be a high-value optimization target, especially for CPU backends where cache effects dominate performance.

### 3. Petalisp's Strided Array Model

**What to study:** The strided array abstraction -- arrays defined by (start, end, step) triples per dimension -- and how `lazy-reshape` creates zero-copy views via affine index transformations. Petalisp's five primitives all operate on this single abstraction.

**Relevance to OCANNL:** OCANNL's `solved_dim` + projections system serves a similar purpose but is more complex. The `indexing.ml` module supports `Affine { symbols; offset }` indices, which correspond to Petalisp's affine transformations. However, OCANNL's projections carry more structure (axis kinds, concat specs, row variables) that Petalisp's simpler model avoids.

**Transferable insight:** The strided array model could inform simplifications to OCANNL's internal representation. In particular, Petalisp's clean separation between "the array's logical shape" and "the affine mapping to physical storage" is worth studying for potential simplification of OCANNL's projection machinery. However, OCANNL's richer projection structure supports features (concatenation, axis kinds) that Petalisp lacks, so this is more about learning from the design than directly adopting it.

### 4. Petalisp's SIMD Code Generation

**What to study:** How Petalisp generates AVX/AVX2/FMA intrinsics via sb-simd and the Loopus loop optimizer, achieving single-core performance competitive with hand-optimized C.

**Relevance to OCANNL:** OCANNL generates C code and relies on the C compiler for auto-vectorization. The `Set_from_vec` construct with `vec_unop` in `low_level.ml` provides some vector operations, but there is no systematic SIMD code generation. The `-march=native` flag (#311) would help the C compiler's auto-vectorizer, but explicit SIMD intrinsics (or pragmas guiding vectorization) could go further.

**Transferable insight:** For v0.8 CPU performance, OCANNL could generate C code with `#pragma omp simd` annotations or `__attribute__((vector_size(...)))` hints rather than full intrinsics. This is less effort than a full SIMD backend and lets the C compiler handle register allocation. Petalisp's experience with SIMD can guide which loops benefit most from vectorization hints.

### 5. Caten's ISL Integration

**What to study:** Caten's ~30 CFFI binding files to ISL. How does it construct polyhedral domains, access relations, and schedules from the AASM IR? What ISL functions are actually used (domain construction, dependency analysis, schedule computation, AST generation)?

**Relevance to OCANNL:** ISL integration is the most concrete path to polyhedral optimization for OCANNL. OCaml has excellent C FFI via `ctypes`, which is comparable to Common Lisp's CFFI. Caten demonstrates that a single developer can bind enough of ISL to support a working compiler, making this more feasible than it might appear.

**Key question:** What subset of ISL does Caten actually use? If the working set is small (e.g., 50-100 ISL functions), an OCaml binding via `ctypes` is tractable. If it requires the full ISL API, the effort may be prohibitive.

**Transferable insight:** The feasibility assessment should focus on: (a) which ISL functions Caten uses, (b) whether OCaml `ctypes` can handle ISL's C API (including its memory management model via `isl_ctx`), and (c) whether a lighter-weight polyhedral analysis (e.g., hand-coded affine scheduling without ISL) would suffice for OCANNL's needs. The ISL dependency adds GMP and ISL itself to OCANNL's build, which is significant complexity for a project that currently has minimal C dependencies.

### 6. Caten's RISC Primitive Set and Schedule-Level Fusion

**What to study:** How Caten's 26-op bottleneck makes optimization tractable. Every tensor operation must be expressed as a sequence of these primitives, which means the optimizer only needs to handle 26 cases. The scheduler then groups compatible primitives into fused kernels, with the invariant that one Schedule-Item = one GPU kernel.

**Relevance to OCANNL:** OCANNL's operations are more varied. The `assignments.ml` types (`Accum_op` with various `accum_rhs` variants, `Set_vec_unop`, `Fetch`) don't form a minimal set -- they carry high-level semantic information through to lowering. The `Low_level.t` IR is closer to a RISC-like representation (scalar operations, loops, gets/sets) but the optimization passes (`optimize_proc`) operate on this richer structure.

**Transferable insight:** A canonicalization pass that normalizes OCANNL's operations to a smaller set before optimization could enable more systematic pattern-based rewrites. This doesn't require redesigning the IR -- it could be an additional normalization step between `assignments.ml` lowering and the existing `optimize_proc`. The key lesson from Caten is that a narrow IR bottleneck pays dividends in optimizer simplicity.

### 7. Caten's Compilation Pipeline Transparency

**What to study:** Caten's `proceed` checkpoints for transparent lazy evaluation, and its REPL-driven development model where the entire compilation pipeline is explorable interactively.

**Relevance to OCANNL:** OCANNL has some IR inspection capabilities (debug logging via `ppx_minidebug`, `sexp_of` on IR types), but there is no systematic way to dump the IR at each compilation stage or visualize the computation graph. Both Petalisp and Caten leverage the Lisp REPL for interactive exploration of the compilation pipeline.

**Transferable insight:** Adding better IR inspection tools -- dump Low_level.t at each optimization pass, visualize the tensor DAG before and after fusion, show the generated C code alongside the IR -- would significantly aid development and debugging. This is a developer-experience improvement that could be implemented incrementally. OCaml's `utop` REPL provides a similar interactive environment, though it lacks Lisp's macro-based introspection.

## Mapping to OCANNL Roadmap

| Transferable Idea | OCANNL Roadmap Item | Timeline | Value |
|---|---|---|---|
| Deferred compilation / batch graph analysis | v0.8-v0.9: compile_batch improvements | Medium-term | High |
| Temporal blocking for iterative algorithms | v0.9: Program search for scheduling (#140) | Post-v0.9 | High |
| SIMD hints in generated C code | v0.8: CPU performance (#311) | Medium-term | Medium |
| ISL integration feasibility assessment | v0.9: Polyhedral scheduling (#133, #267) | Post-v0.9 | High (if feasible) |
| Schedule-level kernel fusion | v0.9: Code graph rewriting | Post-v0.9 | Medium |
| IR canonicalization pass | Not explicitly planned | Could inform v0.9 | Medium |
| IR inspection / visualization tools | v0.7: Developer experience | Near-term | Medium |
| Strided array model simplification | Not planned | Informational | Low |

## Scope

**In scope:**
- Deep study of Petalisp's lazy evaluation, fusion, temporal blocking, and SIMD code generation
- Deep study of Caten's 26-op architecture, ISL bindings, scheduler, and codegen pipeline
- Architecture comparison with OCANNL across all seven research areas
- ISL integration feasibility assessment for OCANNL
- Posting findings as a GitHub issue comment

**Out of scope:**
- Implementing any changes (tracked as separate issues)
- Benchmarking Petalisp or Caten against OCANNL
- Building ISL OCaml bindings (that would be a separate task if deemed feasible)
- Studying Petalisp's experimental multi-threaded execution in detail

**Dependencies:**
- Related exploration tasks: gh-ocannl-242 (TVM, completed), gh-ocannl-267 (Tiramisu), gh-ocannl-301 (IREE), gh-ocannl-316 (DumPy/torchdim)
- The v0.9 milestone (#140 program search, #133 affine index extension) is where Petalisp/Caten ideas become most actionable
- ISL feasibility informs #267 (Tiramisu) and #133 (multiple non-static symbols in affine index)

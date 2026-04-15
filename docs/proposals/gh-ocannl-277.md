# Proposal: Study krnl and autograph for Lessons Applicable to OCANNL

**Task**: gh-ocannl-277
**Issue**: https://github.com/ahrefs/ocannl/issues/277
**Milestone**: v0.8

## Goal

Produce a structured comparative analysis of [krnl](https://github.com/charles-r-earp/krnl) (a Rust GPGPU kernel framework using Vulkan/SPIR-V) and [autograph](https://github.com/charles-r-earp/autograph) (an ML library built on krnl), evaluating their design decisions against OCANNL's architecture and identifying any transferable lessons. The analysis also includes a comparison with [Luminal](https://github.com/jafioti/luminal) (a graph-compiler ML framework in Rust) as requested in the original issue. The deliverable is a written analysis document, not code changes.

## Acceptance Criteria

- [ ] A written analysis document (`docs/krnl-autograph-analysis.md`) covering the subsystems listed in the Context section below
- [ ] krnl's Vulkan/SPIR-V kernel model is assessed for portability lessons relevant to OCANNL's multi-backend architecture (CUDA, Metal, C)
- [ ] autograph's tensor serialization and model composition patterns are compared against OCANNL's existing and planned capabilities (especially gh-ocannl-373 tensor persistence)
- [ ] A three-way architectural comparison: krnl/autograph (eager-mode, Vulkan) vs Luminal (graph compiler, CUDA/Metal) vs OCANNL (einsum IR, code generation) -- identifying where each approach has advantages and disadvantages
- [ ] Each identified technique or pattern is classified as: (a) already addressed by OCANNL or an existing task, (b) applicable and recommended with target milestone, or (c) not applicable with rationale
- [ ] For category (b) items, concrete recommendations specify which OCANNL module/file would be affected and at what level of effort
- [ ] A summary table mapping krnl/autograph features to OCANNL equivalents or gaps

## Context

### What krnl Is

krnl is a Rust GPGPU framework that uses **Vulkan 1.2** as its compute backend, compiling kernels to **SPIR-V** via `spirv-builder` (using `rust-gpu`). Key characteristics:
- Cross-platform GPU compute through Vulkan (Windows, Linux, macOS via MoltenVK)
- Iterator-based kernel dispatch model: kernels are defined as Rust functions operating on slices/iterators, then JIT-compiled to SPIR-V
- Device management abstracted behind a `Device` API
- Supports `u8` through `f32` scalar types; no half-precision
- Small project (last significant activity ~2024), primarily a proof of concept

### What autograph Is

autograph is an ML library built on top of krnl:
- Tensor type emulating `ndarray::Array` API semantics
- Autograd (automatic differentiation) with a tape-based system
- `#[derive(Layer)]` and `#[derive(Forward)]` proc macros for model composition
- `serde` serialization for tensors, models, and optimizer state
- Layer types: Dense, Conv2d, with ReLU activation
- Optimizer: SGD (with learning rate parameter)
- Small scope: MNIST-level examples, proof-of-concept quality

### What Luminal Is

Luminal is a Rust ML framework with a graph-compiler approach:
- Computation defined as a lazy graph, then compiled and optimized before execution
- Backend plugins: CUDA, Metal, with optimization passes (kernel fusion, memory planning)
- Graph rewriting for optimizations (similar in spirit to OCANNL's IR optimization)
- Metal backend with shader compilation
- More mature than krnl/autograph but still experimental

### Subsystems to Analyze

**1. Vulkan/SPIR-V as a portable GPU backend**
- krnl's approach: single API (Vulkan) targeting all GPUs vs OCANNL's native CUDA + Metal
- Portability vs performance tradeoff
- Relevance: OCANNL already has multiple native backends; could Vulkan serve as a universal fallback?
- Related: gh-ocannl-301 (IREE deep dive, which also has a Vulkan backend)

**2. Kernel dispatch model**
- krnl's iterator-based kernel definition vs OCANNL's einsum-to-loop-nest code generation
- How krnl maps Rust iterators to GPU threads
- Luminal's graph-level kernel fusion vs OCANNL's megakernel approach (gh-ocannl-318)

**3. Serialization and persistence**
- autograph's serde-based tensor/model/optimizer serialization
- How this compares to OCANNL's planned tensor persistence (gh-ocannl-373)
- Whether serde-style derive macros have an OCaml equivalent pattern (ppx)

**4. Model composition patterns**
- autograph's derive macros for Layer/Forward composition
- Luminal's graph-based model definition
- Comparison with OCANNL's functional composition in `nn_blocks.ml`

**5. Memory management**
- krnl's Vulkan buffer management vs OCANNL's per-backend allocation
- Luminal's memory planning pass
- Relevance to OCANNL's pool allocator plans (gh-ocannl-344)

### OCANNL Architecture Constraints

Key constraints affecting which lessons transfer:

1. **Generated kernels, not hand-written.** OCANNL generates C/CUDA/Metal from the `Low_level.t` IR. Any technique must be expressible as an IR transformation or code generation pattern.

2. **Einsum-based computation model.** Operations are specified as einsum contractions. krnl's iterator model and Luminal's graph nodes are different abstraction levels.

3. **Single-threaded kernels (current state).** All CUDA kernels run with `grid_dim=1, block_dim=1`. The tiling proposal (gh-ocannl-412) is the prerequisite for GPU parallelism.

4. **Multi-backend from day one.** OCANNL already targets C, CUDA, and Metal through a parameterized `C_syntax` module. This is architecturally closer to krnl's cross-platform goal than to Luminal's backend-specific plugins.

### Related Tasks and Documents

| Document/Task | Relevance |
|---------------|-----------|
| `docs/proposals/gh-ocannl-253.md` | llm.c study -- same "study external project" format |
| `docs/proposals/gh-ocannl-412.md` | Tiling proposal -- krnl's parallelism model is relevant context |
| gh-ocannl-373 | Tensor persistence -- autograph's serde patterns are directly relevant |
| gh-ocannl-301 | IREE deep dive -- IREE also uses Vulkan/SPIR-V |
| gh-ocannl-265 | Candle study -- another Rust ML framework comparison |
| gh-ocannl-318 | Megakernels -- Luminal's kernel fusion is a comparison point |
| `docs/megakernel-deep-dive.md` | Existing megakernel analysis |

## Approach

### Phase 1: krnl Source Study (half day)

Read and analyze the krnl repository:
- `krnl/` core: Device abstraction, buffer management, kernel dispatch
- `krnl-macros/`: The `#[module]` proc macro that compiles Rust to SPIR-V
- `spirv-builder` integration: How Rust code becomes GPU kernels
- Performance characteristics: What does Vulkan cost vs native CUDA?

Document:
- The kernel dispatch model and how it maps to GPU execution
- Vulkan buffer management patterns
- Limitations (no half-precision, limited kernel expressiveness)

### Phase 2: autograph Source Study (half day)

Read and analyze the autograph repository:
- Tensor implementation: How it wraps krnl buffers with ndarray-like semantics
- Autograd tape: How gradients are tracked and computed
- Serialization: The serde derive patterns for tensors, models, optimizers
- Layer composition: The `#[derive(Layer)]` and `#[derive(Forward)]` macros

Focus on:
- Serialization patterns that could inform gh-ocannl-373
- Whether the model composition approach has advantages over OCANNL's functional style

### Phase 3: Luminal Comparison (half day)

Read and analyze Luminal's architecture:
- Graph definition and compilation pipeline
- Optimization passes (fusion, memory planning)
- CUDA and Metal backend implementations
- How it compares to OCANNL's IR-based approach

### Phase 4: Write Analysis Document (half day)

Produce `docs/krnl-autograph-analysis.md` with:
- Executive summary
- Per-subsystem analysis sections (one per subsystem from Context above)
- Three-way comparison table: krnl/autograph vs Luminal vs OCANNL
- Recommendations table: technique, applicability, target milestone, effort
- Architectural insights: what OCANNL's approach handles better, and where other approaches reveal gaps

### Expected Findings (Preliminary Assessment)

Based on the task elaboration, the likely high-value findings are:

1. **Vulkan as universal fallback** (v1.0+, exploratory): krnl demonstrates that Vulkan can serve as a single GPU compute API. However, the performance gap vs native CUDA is significant, and OCANNL already has native backends. The main lesson is whether a Vulkan/SPIR-V path could replace the Metal backend for broader portability. Likely conclusion: not worth the complexity given OCANNL's existing multi-backend architecture, but the IREE deep dive (gh-ocannl-301) may revisit this.

2. **Serialization patterns for tensor persistence** (v0.7.0, relevant): autograph's serde-based approach is clean and composable. The OCaml equivalent would be ppx-derived serializers. This directly informs gh-ocannl-373's design.

3. **Graph-level fusion vs IR-level megakernels** (v0.8, context): Luminal's graph rewriting approach is conceptually similar to OCANNL's megakernel plans but operates at a higher abstraction level. Comparing the two can clarify where OCANNL should fuse (IR level) vs where graph-level reasoning is needed.

4. **Limitations as lessons** (general): Both krnl and autograph are small, proof-of-concept projects that stopped at MNIST-level capability. Understanding why they didn't scale is itself a lesson -- likely related to the overhead of Vulkan abstraction and limited kernel expressiveness. OCANNL's code generation approach avoids these limitations.

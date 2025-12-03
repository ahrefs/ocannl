# OCANNL Roadmap to v1.0

**Target: ICFP 2026 (August 24, 2026)**

This roadmap outlines the development plan for OCANNL from the current state to version 1.0, incorporating an academic paper milestone for ICFP 2026. Dates indicate **end of period** targets.

---

## Late 2025: Foundation Stabilization

### v0.6.2 — End of November 2025
**Theme: Parser fixes and shape error detection**

- **Menhir einsum parser stabilization**
  - Fix any remaining issues with the recently migrated Menhir-based parser
  - Ensure proper error messages for malformed einsum specifications

- **Missing hidden dimensions detection** ✓ (implemented)
  - Verify the existing implementation works as expected
  - Fix/validate the failing test case

### v0.6.3 — Mid-December 2025
**Theme: Convolution padding**

- **Padding inference for convolutions** (#354, #386)
  - Integrate padding inference into shape inference pipeline
  - Refactor `use_padding` from global setting to `Conv_input` constructor field
  - Synthetic toy CNN example: counting

- **Example: Sokoban RL** (stretch goal)
  - Policy gradient example with CNN architecture

- **HIP backend** (#411) — parallel / background task, can slip to v0.6.4
  - Standalone bindings package for HIP (AMD hardware)
  - Backend implementation

---

## Q1 2026: Frontend Maturity and ICFP Paper

### v0.6.4 — End of December 2025
**Theme: Shape concatenation and position embeddings**

- **Axis concatenation in einsum** (#49)
  - Implement `^` syntax for tensor stacking/concatenation
  - Handle shifting as special case: `1^i=>i` for left shift
  - Handle padding as special case: `i=>1^i` for left padding

- **Dimension units vs axis labels** (#298)
  - Clarify design decisions
  - Document rationale

- **RoPE and position embeddings** (#398)
  - Rotary Position Embeddings implementation
  - Other non-learned position embedding variants

- **Transformer toy example** (#57)
  - Fully working decoder-only autoregressive transformer
  - Names dataset language model

### v0.7.0 — End of January 2026
**Theme: Frontend finalization (before ICFP deadline)**

This is the "paper-ready" release with mature frontend API.

- **Context handling finalization**
  - Migrate from "hosted tensor" idea to always requiring context
  - Clean, consistent API for tensor access and device handling

- **Remove hosted tensor mode** (#333)
  - Get rid of `array` field of `Tnode.t`
  - Remove or minimize `Ndarray` module

- **Deprecated streams cleanup**
  - Remove legacy streams functionality

- **Tensor persistence** (#373)
  - Tensor saving, loading, and restoring

- **Documentation**
  - Context API slides (README item 3)

**⚠️ ICFP 2026 Deadline: February 19, 2026**

Paper should use v0.7.0 examples demonstrating the mature frontend.

---

## Q2 2026: Examples and Optimizations

### v0.7.1 — Mid-March 2026
**Theme: Real-world examples and backend polish**

- **Tokenizer bindings**
  - Bindings to tokenizer from llama.cpp or equivalent

- **Transformer inference demo** (#377)
  - Inference for a small open-weights model (GPT-2, LLaMA, or Gemma)

- **CNN training examples**
  - MNIST training setup
  - CIFAR-10 training setup

- **Backend improvements**
  - Resolve multicore_cc non-determinism (#341)
  - MSVC support for Windows C backend (#313)
  - Add `-march=native` optimization (#311)

### v0.7.2 — Mid-April 2026
**Theme: Compiler optimizations**

- **Optimizations**
  - Loop invariant hoisting (#350)
  - Common subexpression elimination (#351)
  - Local scope initialization tracking (#340)

- **Memory management**
  - Universal Pool Allocator (#344)

### v0.8 — Mid-June 2026
**Theme: GPU-style performance**

This is a substantial milestone requiring ~2 months.

- **Tiling optimizations** (inspired by #412)
  - Fast matrix multiplication patterns from Böhm's CPU article
  - CUDA matmul optimizations from Böhm's CUDA article
  - Lessons from llm.c

- **Megakernel exploration** (#318)
  - Investigate megakernel approach alignment with OCANNL design
  - May require splitting routines into multiple kernels

- **Metal optimizations** (#320)
  - Use private mode appropriately

---

## Summer 2026: Search and Completion

### v0.9 — August 24, 2026 (ICFP)
**Theme: Program search and optimization**

This is a research-heavy milestone requiring ~2.5 months.

- **Static scheduling via program search**
  - Alternative to tinygrad's dynamic scheduling
  - Halide-inspired search strategies

- **Cost functions**
  - Per-backend execution-based metrics
  - Aggregate cost functions across backends

- **Code graph rewriting**
  - Broader range of rewriting rules
  - Tiling and layout mechanism augmentation

### v1.0 — End of October 2026
**Theme: Documentation, completeness, ergonomics**

- **Documentation completeness**
  - Lowering and inlining documentation (#296)
  - einops.rocks comparison documentation (#413)

- **Feature completeness**
  - Address select "explore" issues to demonstrate capability
  - Sharding and slicing with minimal copying (#293)

- **Ergonomics**
  - Concise syntax for merge buffer transfers
  - Execution dependency tracking (mirroring compilation)
  - Improve configuration handling (#409)

- **Safety**
  - Merge buffer static verification (#288)
  - Memory mode sharing audit (#291)

---

## Post-1.0 Considerations (v1.1+)

- **Shape inference enhancements**
  - Axis labels (as opposed to dimension units)
  - Shape schemes for tensor functions (#404)

- **Advanced examples**
  - BERT/ModernBERT implementation (#297)
  - LLM101n replication (#275)
  - DisTrO distributed training (#278)

- **Performance explorations**
  - Lean Attention / Flash Attention (#263)
  - Quantization for optimizers (#271)
  - XLA backend (#300)

---

## Key Milestones Summary

| Version | Target | Duration | Key Deliverables |
|---------|--------|----------|------------------|
| v0.6.2  | End Nov 2025 | now | Menhir parser fixes, hidden dimension errors |
| v0.6.3  | Mid-Dec 2025 | 2.5 weeks | Padding inference |
| v0.6.4  | End Dec 2025 | 2 weeks | Shape concatenation |
| v0.6.5  | Mid-Jan 2026 | 2 weeks | RoPE, transformer toy example |
| v0.7.0  | End Jan 2026 | 2 weeks | **Frontend finalization (ICFP: Feb 19)** |
| v0.7.1  | Mid-Mar 2026 | 6 weeks | Real-world examples, backend polish |
| v0.7.2  | Mid-Apr 2026 | 4 weeks | Compiler optimizations, pool allocator |
| v0.8    | Mid-Jun 2026 | 2 months | GPU tiling, megakernels |
| v0.9    | Aug 24, 2026 | 2.5 months | Program search **(ICFP conference)** |
| v1.0    | End Oct 2026 | 2 months | Documentation, completeness, safety |

---

## ICFP 2026 Paper Plan

**Deadline: February 19, 2026**

### Proposed Title
*"Generalized Einsum with Row Variables: Shape Inference for Deep Learning in OCaml"*

### Key Contributions
1. **Generalized einsum notation** with convolutions, strided iteration, and concatenation
2. **Row variables** for flexible axis handling ("principle of least commitment")
3. **Constraint-based shape inference** with provenance tracking for error messages
4. **Dimension units** design rationale (vs axis labels)
5. **Integration with OCaml's type system** via syntax extensions

### Related Work to Address
- einops (#413)
- torchdim/DumPy (#316)
- Named tensors in PyTorch/JAX
- Dependent types for tensor shapes

### Development Timeline
- **Nov 2025**: Release v0.6.2
- **Dec 2025**: Finish v0.6.3; outline paper, literature review
- **Jan 2026**: Finish v0.6.4–0.6.5 (concatenation, RoPE, transformer); first draft
- **Early Feb 2026**: Finish v0.7.0 (frontend finalization); update paper examples
- **Feb 1-18, 2026**: Paper revision with v0.7.0 examples
- **Feb 19, 2026**: **Submission deadline**

### Why v0.7.0 Before Paper
The paper needs working examples with OCANNL's mature frontend:
- Clean context-based API (no hosted tensors)
- Shape concatenation syntax (`^`)
- Complete transformer example with RoPE
- Consistent, documented API surface

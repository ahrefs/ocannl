# Generalized Einsum with Row Variables: Shape Inference for Deep Learning in OCaml

Proposal for OCaml Workshop 2026 / FProPer (collocated with ICFP 2026)

## Motivation

Tensor shape mismatches are among the most common and frustrating bugs in deep learning code. Existing frameworks address this differently:

- **NumPy/PyTorch einsum** provides compact notation but is limited to contraction and summation — no convolutions, strided iteration, or concatenation. Shapes are checked at runtime.
- **Named tensors** (PyTorch, JAX) let users label axes, but labels are verbose in einsum-style notation and introduce inference ambiguity when multiple axes share the same semantic unit (e.g., two spatial dimensions both representing "position").
- **Dependent types** (dex-lang, Futhark) offer compile-time guarantees but demand heavy type annotations and don't integrate naturally with mainstream ML ecosystems.
- **einops** provides a readable reshaping language but operates at a different level — it transforms shapes rather than specifying computations.

OCANNL takes a different approach: a **generalized einsum notation** with **row variables** and **constraint-based shape inference**, integrated into OCaml via PPX syntax extensions. This gives shape safety without verbosity.

## Current State

OCANNL's shape system is implemented and used in production examples (multi-head attention, transformer encoder/decoder, convolutions). The key components:

**Three axis kinds.** Every tensor shape is organized as `batch | input -> output`, reflecting deep learning semantics: batch dimensions are preserved across operations, input dimensions are consumed (like function arguments), and output dimensions are produced.

**Generalized einsum syntax.** Beyond standard einsum contraction, OCANNL's notation supports:
- Strided iteration: `stride * output + dilation * kernel` for convolutions
- Concatenation: `a ^ b` for axis concatenation
- Row variables: `...` (anonymous) or `..name..` (named) for flexible axis tails
- Multi-operand specs: `rhs1 ; rhs2 => lhs` with dimension capture

**Row variables.** Borrowed from row polymorphism in type theory, row variables let users write specs without committing to the exact number of axes. A spec like `"... s | h d; ... t | h d => ... s | t -> h"` works regardless of how many batch dimensions precede the named axes. This is the "principle of least commitment."

**Constraint-based inference.** Shape resolution uses a 7-stage pipeline: online unification during construction, then 6 on-demand stages that progressively resolve dimension variables, row variables, broadcastable dimensions, and projections. Provenance tracking maps shape errors back to the source-level einsum spec.

**Dimension units (not axis labels).** Axes have optional semantic annotations ("basis") that constrain matching — axes with different units cannot be unified — but are identified positionally, not by name. This avoids the verbosity and inference ambiguity of named axes while preserving semantic checking.

Key files:
- Einsum types: `tensor/einsum_types.ml` — `axis_spec` variants (Label, Fixed_index, Affine_spec, Concat_spec)
- Row/dim types: `tensor/row.ml` — `dim`, `row`, `row_var`, `dim_var` types; `unify_dim` constraint solver
- Shape inference: `tensor/shape.ml` — constraint generation from einsum specs
- PPX syntax: `tensor/ppx_cd.ml` — `%cd` and `%op` extensions
- Multi-head attention: `lib/nn_blocks.ml` — flagship example using einsum with row variables

Existing documentation: `shape_inference.md` (357 lines), `syntax_extensions.md` (752 lines), `slides-shapes_and_einsum.md` (565 lines).

## Proposed Change

Write and submit a workshop paper (extended abstract or short paper, per CFP format) with these contributions:

1. **Generalized einsum notation.** Formalize the extended syntax covering contraction, strided iteration, convolution, and concatenation in a unified framework. Show how the three-axis-kind model (batch/input/output) naturally captures deep learning computation patterns.

2. **Row variables for shape inference.** Describe the constraint-based inference algorithm: how row variables generate constraints, how the 7-stage pipeline resolves them, and how provenance tracking produces informative error messages. Compare expressiveness with fixed-arity approaches.

3. **Dimension units vs axis labels.** Articulate OCANNL's design rationale:
   - Positional representation builds on mathematical tradition (linear algebra, Einstein notation)
   - Optional units provide semantic safety without requiring unique names
   - Row variable inference is ambiguity-free with positional axes but problematic with named axes (which axis gets which name when two have the same unit?)
   - Concrete comparison: the same model (multi-head attention) in OCANNL einsum vs PyTorch named tensors vs einops

4. **OCaml integration.** Show how PPX syntax extensions (`%op`, `%cd`) provide an ergonomic DSL while leveraging OCaml's type system and module system.

5. **Examples.** Multi-head attention, transformer encoder/decoder, and convolution — all drawn from working OCANNL code.

**Target venues** (in priority order):
- OCaml Workshop 2026 (primary — co-located with ICFP, shorter format)
- FProPer (if it runs — practice-oriented)
- ML Workshop (backup — ML-family languages focus)
- ARRAY 2026 (backup — array programming focus)

## Scope

**In scope:**
- Paper writing, revision, and submission
- Code examples from current OCANNL (v0.6.x or v0.7.0 if available)
- Literature review: einops, torchdim/DumPy, PyTorch named tensors, dex-lang, Futhark
- Comparison examples (same model in different frameworks)

**Out of scope:**
- New feature implementation (paper uses existing features; RoPE, concatenation etc. are separate tasks)
- Formal type-theoretic proofs (workshop paper, not conference — practical demonstration over formal results)
- Benchmarking or performance evaluation (the paper is about notation and shape inference, not runtime performance)

**Dependencies:**
- gh-ocannl-298 (rename "label" → "basis") — paper terminology should be consistent with codebase. Coordinate to use the same term.
- The paper can proceed with current v0.6.x API; v0.7.0 improvements are desirable but not blocking.

**Related work tasks** that inform paper sections:
- gh-ocannl-316 (DumPy/torchdim deep dive) → related work
- gh-ocannl-413 (einops comparison) → related work
- gh-ocannl-404 (shape schemes) → future work section

# DumPy and Torchdim Deep Dive: Named Dimensions vs OCANNL's Positional Design

**Issue:** [ahrefs/ocannl#316](https://github.com/ahrefs/ocannl/issues/316)
**Status:** Draft proposal
**Date:** 2026-03-25

## Sources Examined

- **DumPy**: Blog article at [dynomight.net/dumpy](https://dynomight.net/dumpy/) (accessed 2026-03-25). PyPI package `dumpy-numpy`. ~700-line proof-of-concept built on JAX. The author explicitly states: "please do not attempt to use it for 'real work'."
- **Torchdim**: GitHub repository at [facebookresearch/torchdim](https://github.com/facebookresearch/torchdim) (archived 2024-08-01, accessed 2026-03-25). Upstreamed into PyTorch as `functorch.dim`. The recommended import for modern PyTorch is `from functorch.dim import dims`. By Zachary DeVito (Meta AI).
- **PyTorch 2 paper**: Ansel et al., "PyTorch 2: Faster machine learning through dynamic Python bytecode transformation and graph compilation," ASPLOS 2024. Covers the `functorch.dim` integration.
- **OCANNL**: Local codebase, primarily `tensor/row.ml`, `tensor/shape.ml`, `tensor/einsum_types.ml`, and `docs/shape_inference.md`.

Terminology note: throughout this document, "torchdim" refers to the concept and library (including its upstream form `functorch.dim`). The archived GitHub repo and the PyTorch 2 paper describe the same system.

## OCANNL's Current Design

OCANNL uses **positional axes grouped by kind** with **optional semantic annotations** on dimensions. This section restates the design from local sources.

### Axis identification: kind + position

Every tensor shape is organized as `batch | input -> output`. Axes are identified by their kind (batch, input, or output) and their position within that kind's row. There are no axis names.

### Dimension labels (dimension units / basis)

From `tensor/row.ml` line 63:
```ocaml
type solved_dim = { d : int; label : string option; proj_id : proj_id option }
```

Labels are optional semantic annotations on individual dimensions, not axis identifiers. From `docs/shape_inference.md` (line 49):

> "OCANNL has labeled dimensions, but not labeled axes. [...] the label is a specification of the semantics of an axis that is more fine-grained than, but of similar nature as, the number of dimensions."

**Label checking** (`tensor/row.ml` lines 1599-1602): When `unify_dim` encounters two solved dimensions that both have labels, the labels must be identical or a `Shape_error` is raised. If only one has a label, the label propagates to the other.

**Label propagation** (`tensor/row.ml` lines 408, 418, 434): Uses `Option.first_some` to merge labels during dimension construction (affine, concat operations).

The codebase currently uses the term "label"; the workshop paper uses "dimension units"; issue #298 proposes renaming to "basis". All three refer to the same concept.

### Einsum pseudo-labels

From `tensor/einsum_types.ml` lines 17-30:
```ocaml
type axis_spec =
  | Label of string
  | Fixed_index of int
  | Affine_spec of { stride : string; over_label : string; conv : conv_spec option; stride_offset : int }
  | Concat_spec of string list
```

The `Label` variant in einsum specs represents local pseudo-labels used to align axes within a single specification. A spec like `"i,j;j,k=>i,k"` uses `i`, `j`, `k` as local variables — they are not stored on tensors and have no meaning outside the spec. Each einsum spec has its own `dim_var_env` (`tensor/shape.ml` lines 256-276).

### Row variables

From `tensor/row.ml` line 175:
```ocaml
type bcast = Row_var of { v : row_var; beg_dims : dim list } | Broadcastable
```

Row variables represent unknown numbers of axes, enabling the "principle of least commitment." A spec like `"... s | h d; ... t | h d => ... s | t -> h"` works regardless of how many batch dimensions precede the named axes.

## DumPy Analysis

### Core idea

DumPy replaces NumPy's implicit broadcasting with explicit named-index notation that compiles to `jax.vmap`. Its three principles:

1. Bring back explicit loop syntax and indices — make intent readable
2. Don't actually execute loops — compile to vectorized operations
3. Remove confusing features (broadcasting, fancy indexing, axis parameters)

### Syntax

**Named index notation:**
```python
# Matrix multiply with explicit indices
Z['i','j'] = Y['j',:] @ dp.linalg.solve(A['i','j',:,:], X['i',:])

# Broadcasting made explicit
C['i','j','k'] = A['i','j'] * B['j','k']
```

**Range context manager** (loop-like syntax):
```python
Z = dp.Slot()
with dp.Range(X.shape[0]) as i:
    with dp.Range(Y.shape[0]) as j:
        Z[i,j] = Y[j,:] @ dp.linalg.solve(A[i,j,:,:], X[i,:])
```

### Compilation to `jax.vmap`

Behind the scenes, DumPy maps array dimensions to string labels via `map_axes()`, automatically vectorizes operations matching labeled dimensions, and unmaps results back to concrete positions. The equivalent JAX code would require nested `vmap` calls with explicit `in_axes` — DumPy abstracts this away.

### Usability evidence

The author scores six problems on a "thinking required" scale (10 = trivial, 1 = painful):

| Method | Mean Score |
|--------|-----------|
| Loops | 9.8 |
| DumPy | 9.5 |
| JAX/vmap | 6.8 |
| NumPy | 4.3 |

DumPy achieves near-loop clarity with GPU performance. Multi-head attention scores 10/10 in DumPy vs 1/10 in NumPy.

### Design tradeoffs

**Strengths:**
- Explicit intent: every dimension is labeled at the operation site
- No implicit broadcasting eliminates a class of silent shape bugs
- Loop-like readability with vectorized execution
- External (temporary) labels: applied per operation, not stored on tensors

**Limitations:**
- Prototype only (~700 lines, not maintained)
- All functions accept maximum 2D inputs; higher dims need explicit indices
- No partial indexing (e.g., `A[2]` shorthand not allowed)
- Only one non-scalar array index per operation
- No performance benchmarks provided

### Relevance to OCANNL

DumPy's temporary external labels are conceptually close to OCANNL's einsum pseudo-labels. Both systems apply names at operation sites and discard them afterward, rather than permanently labeling axes. This validates OCANNL's approach. The key difference: DumPy replaces implicit broadcasting entirely, while OCANNL retains it (controlled by axis kinds and row variables).

## Torchdim Analysis

### Core idea

Torchdim extends PyTorch with first-class dimension objects — Python objects, not strings or integers — that unify named tensors, einsum, automatic batching, and loop-style indexing under a single abstraction.

### Three fundamental rules

**Rule 1 — Implicit batching:** Operations batch over the union of first-class dimensions in their inputs.
```python
batch, channel = dims(2)
input = torch.rand(128, 32)[batch, channel]
bias = torch.rand(32)[channel]
result = input + bias  # result.dims == (batch, channel)
```

**Rule 2 — Dims as dimension specifiers:** Wherever an integer specifies a dimension, a `Dim` object works too.
```python
batch, channel, width, height = dims(4)
input = torch.rand(2, 3, 224, 224)[batch, channel, width, height]
avg_pixel_color = input.mean((width, height))  # dims: (batch, channel)
```

**Rule 3 — Dims are tensors:** A dimension `d` acts as `[0, 1, ..., d.size-1]`, enabling index arithmetic.
```python
i, j = dims(sizes=[4, 4])
upper_tri = where(i <= j, 1, 0).order(i, j)  # upper triangular mask
```

### Multiply-then-sum pattern matching

Torchdim pattern-matches the multiply-then-sum idiom and dispatches to optimized matmul kernels:
```python
i, j, k = dims(3)
r = (A[i, k] * B[k, j]).sum(k)  # recognized as matmul, not elementwise * then sum
```

This appears throughout examples: attention scores, Gram matrices, relative positional embeddings.

### Tuple-based reshape

Tuples of dimensions enable splitting and flattening:
```python
i, j, k = dims(3)
j.size = 2
a = A[(i, j), k]       # split dim 0 into i and j
r = a.order(i, (j, k)) # flatten j and k
```

### Advantages over string-named dimensions (PyTorch Named Tensors)

- **No naming conflicts**: Two `i = dims(1)` in different scopes are distinct objects
- **Dims carry metadata**: `.size` property, can be passed as function arguments
- **Rule 3 is impossible with strings**: Strings can't act as tensors
- **Reuses existing operators**: No new vocabulary needed

### Performance and status

- The README notes "there are known places where performance can be improved" (preview release)
- Torchdim is not a compiler — it's an ergonomic layer over eager PyTorch
- Repository archived 2024-08-01; upstreamed as `functorch.dim`
- Preview release for API feedback

### Relevance to OCANNL

Torchdim's Rule 3 (dims-as-tensors) is the most significant insight for OCANNL. It enables index-component operations (`eye`, `triu`, `diag`, sequence masks) that OCANNL currently cannot express. The multiply-then-sum pattern matching validates einsum-style contraction but uses a more verbose syntax. Torchdim's implicit batching (Rule 1) overlaps with OCANNL's batch axis kind.

## Three-Way Comparison

| Aspect | OCANNL | DumPy | Torchdim |
|--------|--------|-------|----------|
| **Axis identification** | Kind (B/I/O) + position | External string labels | First-class Dim objects |
| **Label scope** | Local to einsum spec | Local to expression (temporary) | Object identity (scope of variable) |
| **Permanence** | Labels on dims, not axes | Temporary (per operation) | Semi-permanent (until `.order()`) |
| **Broadcasting** | Implicit via kind matching + row vars | Explicit (forbidden implicitly) | Implicit over union of dims |
| **Einsum** | String spec: `"i,j;j,k=>i,k"` | Standard `@` + named indices | Multiply + sum (pattern-matched) |
| **Index arithmetic** | Affine indices in einsum: `stride*i+k` | Loop variables as values | Dims as tensors (Rule 3) |
| **Reshape/split** | Concat spec: `a^b` in einsum | Not directly supported | Tuple indexing: `A[(i,j), k]` |
| **Unknown axes** | Row variables (`...`) | Not supported | Not supported (dims have fixed size) |
| **Semantic checking** | Label mismatch -> error | Type checking at operation site | Identity-based (objects) |
| **Conflict avoidance** | N/A (positional) | Unique string generation via Range | Object identity (no collisions) |
| **GPU execution** | C/CUDA/Metal code gen | `jax.vmap` | PyTorch eager |
| **Compilation** | Ahead-of-time (JIT to C) | `jax.jit` | None (eager with pattern matching) |

## Design Argument: Why Positional + Units for Einsum-Based Systems

**Scope of the claim**: Positional axes with optional dimension units are preferable specifically for systems built around einsum-based notation with constraint-based shape inference and row variables. This is NOT a blanket argument against named dimensions — imperative tensor programming genuinely benefits from named axes.

### Arguments for positional + units

**1. Einsum notation is inherently positional.** Einstein summation uses local index variables over positional axes: $A_{ij} B_{jk} = C_{ik}$ doesn't name the axes of A, B, C. OCANNL's einsum specs (`"i,j;j,k=>i,k"`) mirror this mathematical tradition directly.

**2. Inference ambiguity with named axes.** When two axes of the same tensor share the same semantic meaning (e.g., two spatial dimensions both representing "position"), named-axis systems must disambiguate. Positional systems don't have this problem — axes are distinguished by position within their kind. Example: a 2D convolution kernel has two spatial axes with identical semantics but different roles (height vs. width). Position distinguishes them naturally.

**3. Row variables require positional structure.** Row variables represent "some unknown number of axes." Named-axis systems cannot express "some unknown axes with unknown names" — the names would need to be generated or omitted, falling back to positional structure anyway. OCANNL's positional axes compose naturally with row variables, enabling the "principle of least commitment."

**4. Orthogonal concerns.** Axis *identification* (which axis am I operating on?) and axis *semantics* (what does this axis mean?) are separate concerns. OCANNL separates them cleanly: position handles identification, dimension units handle semantics. Named-axis systems conflate these — the name serves both as identifier and semantic annotation.

**5. Conciseness.** The same matrix multiplication:
- OCANNL: `"i,j;j,k=>i,k"` (14 characters)
- Torchdim: `(A[i,k] * B[k,j]).sum(k).order(i,j)` (38 characters)
- DumPy: `C['i','j','k'] = A['i','k'] * B['k','j']` (with sum implicit in `@`)

Einsum specs are more compact for contraction-heavy patterns.

### Counterarguments: where named dims are genuinely stronger

**1. Self-documenting code.** `input[batch, channel, height, width]` is immediately readable. Positional `input` with axes 0,1,2,3 requires context.

**2. Transposition safety.** Transposing a positional tensor silently changes axis semantics. Named axes catch this class of error.

**3. Implicit batching is elegant.** Torchdim's Rule 1 (batch over union of dims) eliminates explicit batch-axis management. OCANNL's batch axis kind achieves the same effect but requires explicit `batch |` in einsum specs.

**4. Index arithmetic.** Torchdim's Rule 3 (`where(i <= j, 1, 0)` for upper triangular masks) is more natural than any current OCANNL approach. This is a real gap.

### Synthesis

Named dimensions have real advantages for imperative tensor programming. But for einsum-based declarative systems with constraint-based shape inference, positional axes with optional semantic annotations are cleaner: they avoid inference ambiguity, compose with row variables, and are more concise. The two approaches are not mutually exclusive — OCANNL could adopt specific ideas (especially dims-as-tensors) orthogonally.

## Transferable Ideas

### Worth pursuing

**1. Dims-as-tensors / arange primitive** (from torchdim Rule 3)
- **What**: A dimension acts as its index range `[0, 1, ..., n-1]`. This enables `eye`, `triu`, `diag`, sequence masks.
- **Why**: OCANNL currently cannot express index-component operations. This was identified as a gap in the Tensor Puzzles analysis (#308).
- **How**: Add an `arange`-like primitive or allow dimension variables to be used as tensor values within einsum or `%op` expressions.
- **Priority**: High. Addresses a concrete expressiveness gap.
- **Maps to**: gh-ocannl-308 (tensor puzzles), potentially a new issue for the primitive itself.

**2. Strict broadcasting mode** (from DumPy)
- **What**: An optional mode where implicit broadcasting is forbidden and all axis alignment must be specified in einsum notation.
- **Why**: DumPy demonstrates that forbidding implicit broadcasting eliminates a class of silent shape bugs. OCANNL could offer this as an opt-in safety feature.
- **How**: A configuration flag or einsum modifier that requires all dimensions to be explicitly matched.
- **Priority**: Medium. Safety feature, not blocking.

### Worth documenting only

**3. Multiply-then-sum pattern recognition** (from torchdim)
- Torchdim's `(A[i,k] * B[k,j]).sum(k)` being pattern-matched to matmul is elegant, but OCANNL's einsum spec `"i,j;j,k=>i,k"` already compiles directly to the right kernel. No action needed — but the comparison is useful for the paper.

**4. Loop mental model for documentation** (from both DumPy and torchdim)
- Both systems use "think of it as a loop" as the primary teaching tool. OCANNL's einsum notation is more compact but less immediately readable to newcomers. Documentation and teaching materials should emphasize loop-equivalent readings of einsum specs.
- **Priority**: Medium for documentation, not for implementation.

### Not aligned with OCANNL

**5. Permanent axis naming as primary identification.** Both torchdim and DumPy (in its Range variant) treat names/objects as the primary way to identify axes. This conflicts with OCANNL's positional + kind structure and would require fundamental architectural changes for unclear benefit in an einsum-based system.

**6. Implicit batching via union of dims.** Torchdim's Rule 1 is elegant but conflicts with OCANNL's explicit axis-kind structure where batch dimensions are a distinct kind, not just "whatever dims are shared."

## Impact on Related Tasks

| Insight | Task | Action |
|---------|------|--------|
| Three-way comparison for related work | gh-ocannl-299 (workshop paper) | Feeds §7 Related Work |
| Core design argument for dim units | gh-ocannl-299 (workshop paper) | Feeds §4 Dimension Units |
| Dims-as-tensors / arange primitive | gh-ocannl-308 (tensor puzzles) | Consider adding index-component primitive |
| Terminology consistency | gh-ocannl-298 (rename label -> basis) | Use "basis" or "dimension units" consistently |
| Semantic checking model | gh-ocannl-255 (dimension label audit) | Audit informed by torchdim's identity-based checking |
| Complementary notation study | gh-ocannl-413 (einops comparison) | Can reference this comparison |

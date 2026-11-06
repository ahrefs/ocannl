# DumPy and Torchdim Deep Dive: Dimension Naming Strategies Compared with OCANNL

**Issue:** [ahrefs/ocannl#316](https://github.com/ahrefs/ocannl/issues/316)
**Status:** Draft proposal
**Date:** 2026-03-25

## Sources

- **DumPy**: Blog article at [dynomight.net/dumpy](https://dynomight.net/dumpy/) (accessed 2026-03-25). PyPI package `dumpy-numpy`. ~700-line proof-of-concept built on JAX. The author explicitly states: "please do not attempt to use it for 'real work'."
- **Torchdim**: GitHub repository at [facebookresearch/torchdim](https://github.com/facebookresearch/torchdim) (archived 2024-08-01, accessed 2026-03-25). Upstreamed into PyTorch as `functorch.dim`. By Zachary DeVito (Meta AI).
- **PyTorch 2 paper**: Ansel et al., "PyTorch 2: Faster machine learning through dynamic Python bytecode transformation and graph compilation," ASPLOS 2024. Covers the `functorch.dim` integration.
- **OCANNL**: Local codebase, primarily `tensor/row.ml`, `tensor/shape.ml`, `tensor/einsum_types.ml`, and `docs/shape_inference.md`.

Terminology note: throughout this document, "torchdim" refers to the concept and library (including its upstream form `functorch.dim`). The archived GitHub repo and the PyTorch 2 paper describe the same system.

## Motivation

OCANNL uses positional axes with optional semantic labels ("dimension units" / "basis"), while DumPy and torchdim represent two distinct approaches to named dimensions. Understanding these alternatives is directly relevant to:

- **Workshop paper** (OCaml Workshop / FProPer, deadline May--June 2026): Section 5 "Dimension Units vs Axis Labels" and Section 8 "Related Work" need a rigorous comparison.
- **The `label` to `basis` rename** (#298): the terminology must be grounded in a clear design argument distinguishing OCANNL's approach from named-axis systems.
- **Missing primitives** (#308 Tensor Puzzles): torchdim's "dims as tensors" exposes a gap in OCANNL's expressiveness.

## Background: The Three Systems

### DumPy — External Named Indices via `jax.vmap`

[DumPy](https://dynomight.net/dumpy/) is a ~700-line proof-of-concept built on JAX. Its design philosophy is: "loops were better" for readability, so provide loop-like syntax that compiles to vectorized GPU code.

**Core mechanism:** Indexing an array with string labels creates a "mapped" array that appears to have fewer dimensions. When mapped arrays enter a function, JAX's `vmap` vectorizes over the mapped dimensions automatically.

```python
# Matrix multiply with explicit named indices
Z['i','j'] = Y['j',:] @ dp.linalg.solve(A['i','j',:,:], X['i',:])

# Context manager variant
with dp.Range(N) as i:
    with dp.Range(M) as j:
        Z[i,j] = A[i,:] @ B[:,j]
```

**Key design decisions:**
1. **No implicit broadcasting.** `A * B` only works if shapes are identical or one is scalar. Broadcasting must be spelled out via named indices.
2. **External dimensions.** Labels are temporary (applied per operation, not stored on tensors). This avoids the ambiguity of permanent names in linear algebra (e.g., what dimensions should `A^T A` have?).
3. **Functions restricted to 2D inputs.** Batching is handled entirely by the mapping layer, not by functions themselves.

**Limitations:** Prototype quality ("please do not attempt to use it for 'real work'"); no partial indexing; no multi-array fancy indexing; functions limited to 2D.

### Torchdim — First-Class Dimension Objects

[Torchdim](https://github.com/facebookresearch/torchdim) by Zachary DeVito (Meta AI) extends PyTorch with dimension objects as first-class Python values. Now archived; upstreamed as `functorch.dim`.

**Three fundamental rules:**

1. **Implicit batching:** Operations batch over the union of first-class dimensions in their inputs.
2. **Dims as dimension specifiers:** Wherever an integer specifies a dimension, a `Dim` object works too.
3. **Dims are tensors:** A dim `d` acts as `[0, 1, ..., d.size-1]`, enabling index arithmetic.

```python
# Binding positional to named
batch, channel = dims(2)
input_fc = input_positional[batch, channel]
bias_fc = bias_positional[channel]
result = input_fc + bias_fc  # Rule 1: batches over {batch, channel}

# Einsum as multiply-then-sum (pattern-matched to matmul)
i, j, k = dims(3)
r = (A[i, k] * B[k, j]).sum(k)  # matrix multiply

# Rule 3: dims as tensors
i, j = dims(sizes=[4, 4])
upper_tri = where(i <= j, 1, 0).order(i, j)

# Reshape via tuples
a = A[(i, j), k]          # split dim 0 into i and j
r = a.order(i, (j, k))    # flatten j and k
```

**Key advantages over string names:** No naming conflicts (object identity, not string equality); dims carry `.size` metadata; dims act as tensors (Rule 3 impossible with strings); reuses existing operators.

**Performance:** ~2 microseconds overhead per operation on top of PyTorch's ~8 microseconds. Not a compiler -- an ergonomic layer on top of PyTorch's eager execution.

### OCANNL — Positional Axes with Optional Semantic Labels

OCANNL identifies axes by **kind** (`Batch` / `Input` / `Output`) and **position** within that kind's row. Dimension labels are optional semantic annotations, not axis identifiers.

```ocaml
type solved_dim = { d : int; label : string option; proj_id : proj_id option }
(* row.ml:63 *)
```

From `shape_inference.md`: "OCANNL has labeled dimensions, but not labeled axes... The label is a specification of the semantics of an axis."

Einsum notation uses **local pseudo-labels** that match axes by position within their kind:

```
"m,n ; n,k => m,k"
```

The labels `m`, `n`, `k` are local to this einsum specification. Row variables (`...`) represent unknown numbers of axes, enabling the "principle of least commitment."

**Label checking** (`row.ml:1599`): Two solved dims with different labels raise `Shape_error`. One labeled + one unlabeled: label propagates. No global enforcement that the same label implies the same size.

## Three-Way Comparison

| Aspect | OCANNL | DumPy | Torchdim |
|---|---|---|---|
| **Axis identification** | Kind (B/I/O) + position | External string labels | First-class `Dim` objects |
| **Label scope** | Local to einsum spec | Local to expression (temporary) | Object identity (scope of variable) |
| **Permanence** | Labels on dims, not axes | Temporary (applied per op) | Semi-permanent (until `.order()`) |
| **Broadcasting** | Implicit via kind matching + row vars | Forbidden implicitly; explicit only | Implicit over union of dims |
| **Einsum** | String spec: `"i,j;j,k=>i,k"` | Standard `@` + named indices | Multiply + sum (pattern-matched) |
| **Index arithmetic** | Affine indices: `stride*i+k` | Loop variables as values | Dims as tensors (Rule 3) |
| **Reshape/split** | Concat spec: `a^b` in einsum | Not directly supported | Tuple indexing: `A[(i,j), k]` |
| **Unknown axes** | Row variables (`...`) | Not supported | Not directly (dims have fixed size) |
| **Semantic checking** | Label mismatch raises error | Type checking at operation site | Identity-based (objects, not strings) |
| **Conflict avoidance** | N/A (positional) | Generated strings via `Range` | Object identity (no name collisions) |
| **GPU execution** | C/CUDA/Metal code gen | `jax.vmap` | PyTorch eager |
| **Compilation** | Ahead-of-time (JIT to C) | `jax.jit` | None (eager with pattern matching) |

## The Core Design Argument

**OCANNL's thesis:** Positional axes with optional semantic labels ("dimension units" / "basis") are superior to named axes for an einsum-based system with constraint-based shape inference.

### Arguments for OCANNL's approach

1. **Einsum notation is inherently positional.** Einstein summation notation uses positional index variables: `A_{ij} B_{jk} = C_{ik}` doesn't name the axes of A, B, C. OCANNL's einsum specs (`"i,j;j,k=>i,k"`) mirror this directly. Both DumPy and torchdim reintroduce the positional-to-named binding step that einsum notation was designed to avoid.

2. **Row variables need positional structure.** OCANNL's row variables (`...` in einsum specs) represent unknown numbers of axes with unknown semantics. Named axes cannot easily express "some unknown number of axes with unknown names." This enables OCANNL's "principle of least commitment" in shape inference -- a tensor can be defined without knowing its full rank, and shape inference will deduce it from context.

3. **Inference ambiguity with named axes.** When two axes share the same semantic name (e.g., two spatial dimensions both meaning "position"), named-axis systems must disambiguate. Positional systems don't have this problem. Torchdim solves this with object identity (two `i = dims(1)` calls create distinct objects), but this means the "name" is really a memory address, not a semantic label.

4. **Orthogonal concerns.** Axis *identification* (which axis am I operating on?) and axis *semantics* (what does this axis mean?) are separate. OCANNL separates them cleanly: position handles identification, labels handle semantics. In torchdim and DumPy, the name serves both roles, which creates tension (e.g., DumPy explicitly avoids permanent names because of the `A^T A` ambiguity).

5. **Compactness.** Named axes require naming every axis in every operation. Torchdim: `(A[i,k] * B[k,j]).sum(k).order(i,j)`. OCANNL einsum: `"i,k;k,j=>i,j"`. The positional spec is more compact and closer to mathematical convention.

6. **Constraint-based inference.** OCANNL's shape inference maintains an environment of dimension and row variable constraints, solved through unification and subtyping. This machinery works naturally with positional axes and row variables. Adapting it to named axes would require a fundamentally different approach -- name-based unification rather than position-based -- with less clear semantics for broadcasting and unknown-rank tensors.

### Arguments against (from DumPy/torchdim perspective)

1. **Positional is error-prone.** Transposing a positional tensor silently changes which axis is which. Named axes catch this at the operation site.

2. **Self-documenting.** `input[batch, channel, height, width]` is clearer than positional `input` with axes 0,1,2,3. OCANNL's axis kinds (B/I/O) provide partial self-documentation, but within a kind, axes are distinguished only by position.

3. **Implicit batching is powerful.** Torchdim's Rule 1 (batch over union of dims) is elegant and eliminates explicit batch-axis management. OCANNL handles batching via the batch axis kind in einsum specs, which is comparable in power but requires the user to know the batch structure.

4. **Index arithmetic as tensor operations.** Torchdim's Rule 3 -- a dimension acts as `[0, 1, ..., n-1]` -- enables `where(i <= j, 1, 0)` for upper triangular masks. OCANNL currently has no equivalent (see "Transferable Ideas" below).

### Synthesis for the workshop paper

The paper should argue that **named dimensions and positional dimensions are complementary strategies optimized for different programming models:**

- **Named dimensions** (DumPy, torchdim) excel in *imperative* tensor programming where operations are composed step-by-step and each tensor's role must be clear at every point.
- **Positional dimensions** (OCANNL) excel in *declarative, einsum-based* systems where operations are specified as index-contraction patterns and shape inference handles the plumbing.

OCANNL's optional semantic labels ("basis") occupy a principled middle ground: they provide semantic checking (label mismatch raises `Shape_error`) without conflating identification with semantics. The label `"rgb"` on a dimension of size 3 means "this axis represents color channels" -- it doesn't identify the axis (position does that) or participate in matching (einsum pseudo-labels do that).

## Transferable Ideas

### From torchdim (high value)

**1. Dims-as-tensors / index-component primitive.**

Torchdim's Rule 3 is the most significant gap relative to OCANNL. A dimension `d` acting as `[0, 1, ..., d.size-1]` enables:
- `eye(n)` as `where(i == j, 1, 0)`
- `triu(A)` as `where(i <= j, A, 0)`
- `diag(v)` as `where(i == j, v, 0)`
- Attention masks, causal masks, relative position encodings

OCANNL currently cannot express these without explicit array initialization. The Tensor Puzzles analysis (#308) identified this as a gap.

**Proposed approach:** Add an `Iota` (or `Arange`) terminal type that creates a 1D tensor whose values are `[0, 1, ..., n-1]` where `n` is the axis dimension. This integrates naturally with OCANNL's existing `Fixed_index` mechanism in einsum specs but operates on *values* rather than *indices*. It could be exposed as:

```ocaml
(* New terminal: generates index values along one axis *)
let iota ~d = (* tensor with shape [d] and values 0..d-1 *)
```

Combined with OCANNL's existing element-wise operations and einsum, this would unlock the operations listed above. This is a narrower, more compositional approach than torchdim's Rule 3 (which overloads the meaning of dimension objects themselves).

**Effort:** Small-medium. The terminal itself is trivial; the main work is ensuring it participates correctly in shape inference and code generation. Could be tracked as an extension to #308.

**2. Tuple-based reshape syntax.**

Torchdim's `A[(i,j), k]` for splitting and `a.order(i, (j,k))` for flattening is more general than OCANNL's concat spec (`a^b`). However, OCANNL's concat spec already covers the primary use cases (concatenation, splitting, shifting, padding), and its integration with shape inference is mature. The generalization to arbitrary nested tuples would add complexity without clear benefit at this stage.

**Assessment:** Low priority. OCANNL's `^` syntax in einsum is adequate.

### From DumPy (medium value)

**3. Strict broadcasting mode.**

DumPy forbids implicit broadcasting entirely, requiring explicit index annotation. OCANNL currently uses implicit broadcasting (a dim-1 axis broadcasts to any size). A "strict broadcasting" mode that requires all broadcasting to be specified in einsum notation would catch subtle shape bugs.

**Proposed approach:** A per-operation or per-module flag that causes shape inference to use equations instead of inequalities for all axis matching (not just einsum). This is already the behavior for einsum operations -- it would extend the same discipline to `Pointwise_bin` and `Compose` operations.

**Effort:** Small. The constraint generation in `get_inequalities` (`shape.ml`) already distinguishes between einsum (equations) and pointwise/compose (inequalities). A flag to force equations globally would be straightforward.

**4. External dimension labeling validation.**

DumPy's temporary labels validate OCANNL's approach: labels applied per-operation (DumPy) are conceptually equivalent to labels local to an einsum spec (OCANNL). Both systems keep names external to the tensor's identity. This is a design validation, not a transferable feature.

### From both (conceptual value)

**5. Loop mental model for documentation.**

Both DumPy and torchdim use "think of it as a loop" as the primary mental model for understanding vectorized operations. The DumPy author scores six problems on a "thinking required" scale (10 = trivial, 1 = painful):

| Method | Mean Score |
|--------|-----------|
| Loops | 9.8 |
| DumPy | 9.5 |
| JAX/vmap | 6.8 |
| NumPy | 4.3 |

DumPy achieves near-loop clarity with GPU performance. Multi-head attention scores 10/10 in DumPy vs 1/10 in NumPy.

OCANNL's einsum notation is more compact but less immediately readable to newcomers. Documentation and tutorials should emphasize the loop-equivalent reading of einsum specs:

```
"i,k ; k,j => i,j"    reads as:    for i, for j: sum over k of A[i,k] * B[k,j]
```

This framing would help users coming from NumPy/PyTorch backgrounds.

### Not aligned with OCANNL

**6. Permanent axis naming as primary identification.** Both torchdim and DumPy (in its Range variant) treat names/objects as the primary way to identify axes. This conflicts with OCANNL's positional + kind structure and would require fundamental architectural changes for unclear benefit in an einsum-based system.

**7. Implicit batching via union of dims.** Torchdim's Rule 1 is elegant but conflicts with OCANNL's explicit axis-kind structure where batch dimensions are a distinct kind, not just "whatever dims are shared."

## Mapping to OCANNL Tasks and Paper

| Insight | Relevant Task | Action |
|---|---|---|
| Named dims in related work | gh-ocannl-299 (paper) | Section 8: Compare torchdim, DumPy, PyTorch Named Tensors |
| Positional vs named argument | gh-ocannl-299 (paper) | Section 5: Core design argument with formal comparison |
| Dims-as-tensors (`iota`/`arange`) | gh-ocannl-308 or new issue | Add index-component primitive |
| Label to basis rename | gh-ocannl-298 | Use "basis" consistently, grounded in this comparison |
| Strict broadcasting mode | New issue (low priority) | Optional equations-only mode for debugging |
| Loop mental model | Documentation | Emphasize loop-equivalent readings in tutorials |

## Scope

**In scope:**
- Three-way comparison of dimension-naming strategies (this document)
- Design argument for the workshop paper
- Identification of transferable ideas with effort assessment
- Posting findings as a GitHub issue comment

**Out of scope:**
- Implementing any changes (tracked as separate issues)
- Benchmarking against DumPy or torchdim
- Studying PyTorch Named Tensors or xarray in detail (related but distinct systems)

**Dependencies:**
- gh-ocannl-299: Workshop paper (this analysis feeds Sections 5 and 8)
- gh-ocannl-298: Label to basis rename (terminology depends on this analysis)
- gh-ocannl-308: Tensor Puzzles (dims-as-tensors idea directly relevant)
- gh-ocannl-413: einops comparison (related notation study)

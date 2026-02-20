# Concatenate Along Axis: Einsum Extension Completion

## Motivation

The `^` concatenation operator in OCANNL's einsum system is largely implemented and working
for 2-way concatenation, shifting, and fixed-index operations. However, two gaps remain:

1. **3-way+ backpropagation bug**: When three components of different sizes are concatenated
   and one has dimension 1, the backward pass raises "Ambiguous indices in concatenation:
   multiple blocks viable for same position." This blocks correct gradient computation for
   compound concatenation patterns.

2. **Block tensor literal syntax**: Documented in `syntax_extensions.md` but not implemented.
   The syntax `[ta; tb]`, `(ta, tb)`, `[|ta; tb|]` would allow constructing block matrices
   and stacked tensors declaratively.

Completing this task unblocks gh-ocannl-421 (block tensor projection refactoring), which
in turn unblocks gh-ocannl-398 (RoPE position embeddings) — a prerequisite for v0.6.4
and LLM inference support.

See: https://github.com/ahrefs/ocannl/issues/49

## Current State

### Working concat infrastructure

The `^` operator is fully integrated with the parser, shape inference, and lowering:

- **Parser**: `CARET` token and `Concat_spec` type in `parser.mly` (lines 78, 195-196)
- **Operation**: `concat_sum` / `++^` operator in `operation.ml` (lines 448-478) using `Asgns.Block`
- **Lowering**: Block lowering with offset computation in `assignments.ml` (lines 301-365)
- **Gradient**: `Rev_sides` semantics in `assignments.ml` (lines 57-60) — reverses LHS/RHS for backprop
- **Docs**: Comprehensive documentation in `syntax_extensions.md` (lines 503-549)
- **Tests**: `test/operations/test_concat_graph.ml` — forward and gradient tests

Patterns like `a; b => a^b` (concatenate), `a^b => a` (extract prefix), `3^a^5 => a`
(extract middle), and integer-constant shifting all work correctly.

### The 3-way backprop bug

In `assignments.ml` line 392, the `allow_by_concat` filter (lines 375-380) checks whether
each RHS block's `Concat` symbol matches the current block iteration value. When a component
has dimension 1, its projection uses `Fixed_idx 0`, which always passes the symbol check
regardless of which component is "active." This causes multiple RHS blocks to be selected
for the same LHS position, and `apply_op` (a `Unop`) receives 2 values instead of 1.

The test at `test_concat_graph.expected` line 244 captures this as a known limitation.

### Block tensor syntax gap

`syntax_extensions.md` lines 552-566 documents the planned syntax:

| Syntax | Axis kind | Meaning |
|--------|-----------|---------|
| `[ta; tb]` | Output | Stack along new output axis |
| `(ta, tb)` | Input | Stack along new input axis |
| `[|ta; tb|]` | Batch | Stack along new batch axis |

Translation: introduce a size-1 leading axis per component, then concatenate via `++^`.
Not yet implemented in the PPX.

## Proposed Change

### Fix the ambiguous indices bug

The `allow_by_concat` filter needs to be tightened so that exactly one RHS block is selected
per LHS position, even when a component has dimension 1. The fix is localized to the
block lowering in `assignments.ml` (around lines 366-396).

### Implement block tensor literal syntax

Extend the PPX to recognize `[ta; tb]`, `(ta, tb)`, and `[|ta; tb|]` as block tensor
construction. Each translates to appropriate `++^` calls with generated axis variables.

### Acceptance criteria

- 3-way+ concatenation with unit-dimension components passes forward and backward correctly
- Known-limitation test output updated to show correct results
- Block tensor literals produce correct results for simple cases (stacked vectors, 2x2 block matrix)
- All existing 2-way concat, shifting, and fixed-index operations continue working
- Stretch: block-diagonal tensor construction (`a->b;c->d=>a^c->b^d`)

### Edge cases

- **Unit-dimension `Fixed_idx 0` aliasing**: The core bug — filter must disambiguate based on
  offset ranges, not just symbol presence
- **Syntax ambiguity**: `[ta; tb]` is also a valid OCaml list. The PPX needs a disambiguation
  strategy (e.g., `[%block ta; tb]` or context-based detection within `%op` blocks)
- **Nested concatenation**: Concatenating results of concatenations — each level has its own
  Concat symbols
- **Gradient through block tensors**: `Rev_sides` must handle the introduced leading axis correctly

## Scope

**In scope**:
- Fix 3-way+ concatenation backprop bug (`assignments.ml`)
- Implement block tensor literal syntax (PPX extension)
- Update tests and documentation

**Out of scope**:
- Block-diagonal tensors (stretch goal, may be deferred)
- Multi-argument `%cd` syntax redesign (separate concern, gh-ocannl-348)
- Projection type refactoring to `axis_index array array` (gh-ocannl-421, depends on this)

**Dependencies**:
- Blocks gh-ocannl-421 (projection refactoring) → blocks gh-ocannl-398 (RoPE) → v0.6.4 release

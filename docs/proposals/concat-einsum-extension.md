# Fix 3-way+ Concatenation Backpropagation Bug

## Motivation

The `^` concatenation operator in OCANNL's einsum system is largely implemented and working
for 2-way concatenation, shifting, and fixed-index operations. However, a bug in the backward
pass blocks correct gradient computation for 3-way+ concatenation when one component has
dimension 1:

**The bug**: When three components of different sizes are concatenated and one has dimension 1,
the backward pass raises "Ambiguous indices in concatenation: multiple blocks viable for same
position." This prevents correct gradient flow through compound concatenation patterns like
`"a; b; c => a^b^c"` where `b` has dimension 1.

Fixing this unblocks gh-ocannl-421 (block tensor projection refactoring), which in turn
unblocks gh-ocannl-398 (RoPE position embeddings) — a prerequisite for v0.6.4 and LLM
inference support.

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

#### Bug trigger flow

1. Three tensors of dimensions 2, 1, 2 are concatenated via `"a; b; c => a^b^c"`
2. The lowering generates Concat symbols for the product space — one symbol per component
3. For each LHS index position, the basecase at lines 366-383 iterates over all RHS blocks
4. `allow_by_concat` (line 380) checks if the current Concat symbol values are consistent
   with the RHS block
5. When component `b` has dimension 1, its `Fixed_idx 0` always passes the check regardless
   of which Concat component is "active"
6. Both RHS block 1 (the unit-dim tensor) and RHS block 2 contribute for certain positions
7. `apply_op` receives an array of 2 RHS values but expects 1, raising `Invalid_argument`

## Proposed Change

### Fix the ambiguous indices bug

The `allow_by_concat` filter needs to be tightened so that exactly one RHS block is selected
per LHS position, even when a component has dimension 1. The fix is localized to the
block lowering in `assignments.ml` (around lines 366-396).

The fix needs to either:
- Make the Concat symbol filtering aware of unit dimensions (ensure exactly one block is
  selected per position)
- Or handle the case where a unit-dim component's `Fixed_idx 0` is always valid by
  prioritizing the correct block based on offset ranges

### Acceptance criteria

- 3-way+ concatenation with unit-dimension components passes forward and backward correctly
- Known-limitation test output updated to show correct results
- All existing 2-way concat, shifting, and fixed-index operations continue working
- No regression in existing tests

### Edge cases

- **Unit-dimension `Fixed_idx 0` aliasing**: The core bug — filter must disambiguate based on
  offset ranges, not just symbol presence
- **Nested concatenation**: Concatenating results of concatenations — each level has its own
  Concat symbols
- **Gradient through unit-dim component**: `Rev_sides` must handle the introduced `Fixed_idx 0`
  correctly in reverse direction

## Scope

**In scope**:
- Fix 3-way+ concatenation backprop bug (`assignments.ml`)
- Update tests and test expectations

**Out of scope**:
- Block tensor literal syntax (`[ta; tb]`, `(ta, tb)`, `[|ta; tb|]`) — tracked separately
  in task-fe1c593d (blocked by this fix)
- Block-diagonal tensors (`a->b;c->d=>a^c->b^d`) — deferred
- Multi-argument `%cd` syntax redesign — separate concern (gh-ocannl-348)
- Projection type refactoring to `axis_index array array` — gh-ocannl-421 (depends on this)

## Dependencies

- Blocks gh-ocannl-421 (projection refactoring) -> blocks gh-ocannl-398 (RoPE) -> v0.6.4 release
- task-fe1c593d (block tensor literal syntax) is blocked by this fix

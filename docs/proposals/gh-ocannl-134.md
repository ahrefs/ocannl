# Allow Multiple Virtual Tensors to Share a Loop

Tracked by: https://github.com/ahrefs/ocannl/issues/134

## Status

Issue #134 is still open. The feature has not landed.

`arrayjit/lib/low_level.ml` still uses `reverse_node_map` as a single-owner map from loop
symbols to one `Tnode.t`. When another tensor uses the same symbol, `track_symbol` marks
both tensors as `is_complex`. That is intended to prevent unsafe shared-loop virtualization,
but it is incomplete and too coarse.

This proposal is a bookkeeping and cleanup correctness fix for virtual tensors that share
loop symbols. It should land before broad #133 integration work to keep concerns separated,
but it is not a hard prerequisite for isolated #133 diagonal-index tests.

## Problem

The current virtualization pipeline assumes that a traced loop symbol belongs to one tensor
node:

```text
visit_llc -> virtual_llc -> cleanup_virtual_llc -> simplify_llc
         -> eliminate_common_subexpressions -> hoist_cross_statement_cse
```

The single-owner assumption creates two problems.

First, shared loop symbols make `visit_llc` mark participating tensors as `is_complex`. That
suppresses the "simple computation" shortcut but does not guarantee materialization. Low-use
tensors can still be selected for virtualization.

Second, later phases still treat the loop as owned by whichever tensor won the
`reverse_node_map` entry. `cleanup_virtual_llc` can then drop the whole loop for that tensor,
deleting sibling setters that should have remained.

The fix must also avoid storing irrelevant sibling work as if it contributed to the tensor
being virtualized. A loop body may contain several setters, but the stored computation for
one tensor must reduce to the component that produces that tensor's value. Existing
read-before-write and memory-mode checks should continue to decide whether a read is a valid
virtual read or must remain materialized.

## Goal

Allow tensors computed in the same loop nest to virtualize when each tensor is individually
eligible, while preserving loops or tensor setters that still have non-virtual work.

This is both:

- an optimization unlock for loops that compute multiple independent temporary tensors; and
- a correctness fix for the current single-owner cleanup behavior.

## Non-Goals

- Changing index admissibility rules. Repeated symbols and affine inlining are #133 work.
- Making all shared-loop virtualization profitable. This proposal should keep the existing
  visit-count heuristics. Add a new cap only if measurements show code-size growth.
- Replacing virtual inlining's existing read-before-write and memory-mode safety rules with a
  new sibling-dependency analysis.

## Design

### Track Multiple Owners

Change loop-symbol ownership from one tensor to many tensors:

```text
Symbol.t -> Tnode.t set
```

A list is also acceptable if call sites preserve deterministic ordering and deduplicate
entries. The important change is that symbol sharing alone must not set `is_complex`.

Keep `is_complex` for actual computation complexity. This proposal only removes symbol
sharing as a source of complexity.

`track_symbol` must continue to see symbols from:

- `Iterator` indices;
- symbols inside `Affine` indices;
- symbols inside `Concat` indices.

`Concat` support here means bookkeeping only; concat index virtualization remains governed
by the existing index checks.

### Store One Candidate Computation Per Tensor

When `virtual_llc` reaches a `For_loop`, look up all candidate tnodes associated with that
loop symbol. For each candidate that is not already known non-virtual, call
`check_and_store_virtual` with the loop body as the candidate computation boundary.

This is compatible with `inline_computation`: it already filters the stored body to the
target tensor's own `Set`, `Set_from_vec`, `Zero_out`, and recursive `Get` operations.

The stored computation must still be target-specific. If the full loop body contains sibling
setters, `inline_computation` should keep only the statements that contribute to the target
tensor's value and recursively rewrite valid virtual `Get`s in those statements. Sibling
setters that do not contribute must not be replayed inside the target's inlined computation.

Thread the change through `optimize_ctx` deliberately:

- `computations_table` is persisted and can store the same source loop under multiple tnode
  keys.
- `process_for` should prevent recursive self-inlining, but it should not suppress valid
  inlining of sibling providers merely because their setters share the same loop.

### Rely on Existing Virtual-Read Safety

Do not add a blanket sibling-read rejection rule.

For forward provider reads:

```text
A[i] = f(i)
B[i] = g(A[i])
```

the desired result is that `A` can be stored as a virtual computation and the surviving `B`
setter can be rewritten to:

```text
B[i] = g(f(i))
```

For reverse-order or loop-carried reads:

```text
B[i] = g(A[i])
A[i] = f(i)
```

the read of `A` is read-before-write relative to that producer and must not be rewritten to
the later value. Existing read-before-write and `known_non_virtual` handling are the safety
mechanism for this case.

The implementation should preserve that discipline: inline sibling `Get`s when the provider
is a valid virtual value at that point, and leave them materialized when the existing analysis
marks the provider non-virtual.

### Cleanup by Recursing

`cleanup_virtual_llc` should not drop a whole `For_loop` merely because its index maps to a
virtual tensor. Instead, recurse into the loop body.

The per-statement cleanup already has the right shape:

- drop `Set`, `Set_from_vec`, and `Zero_out` for tensors that became virtual;
- keep statements for known non-virtual tensors;
- elide a loop when its cleaned body is empty.

This moves memory-mode forcing to the tensor operation sites rather than the loop ownership
site, and avoids deleting sibling work.

### Code-Size Guardrail

Storing the same full loop body under several tensor keys can duplicate generated code:

```text
N virtual siblings * K reads -> roughly N * K filtered loop bodies
```

Cross-statement CSE is not expected to recombine sibling replays because the filtered bodies
target different tensors. Keep the existing visit-count controls. Do not add a new
sibling-count or loop-body-size cap in the first landing; measure Block/concat lowering
paths and revisit only if code-size growth appears.

## Acceptance Criteria

- Two independent tensors written in the same loop can both virtualize and inline at their
  use sites.
- A shared-loop case that currently drops a sibling setter is covered by a regression test.
- `reverse_node_map` or its replacement records all candidate tnodes for a symbol rather than
  one winner.
- Shared symbol ownership no longer marks tensors `is_complex`; actual computation
  complexity still does.
- `virtual_llc` stores one target-specific computation per candidate tnode from a shared
  loop.
- Valid sibling provider reads are inlined in surviving sibling setters.
- Reverse-order or read-before-write sibling reads remain protected by existing non-virtual
  handling.
- `cleanup_virtual_llc` recurses into shared loops and preserves non-virtual residual work.
- Existing virtual tensor tests still pass.
- Compile-time and generated-code size are checked on at least one Block/concat-style case
  that motivated shared-symbol tracking.

## Tests

Add focused cases for:

- two independent sibling tensors set in one loop and read downstream;
- a mixed loop where one sibling virtualizes and another remains materialized;
- a forward sibling-provider read, where the provider is virtualized and inlined into the
  surviving reader setter;
- a reverse-order/read-before-write sibling read, where the later setter is not used to
  rewrite the earlier read;
- cleanup preserving non-virtual sibling setters after removing virtual setters;
- `is_complex` still being set by genuine complex scalar computation, not by symbol sharing;
- interaction with cross-statement CSE and hoisting after cleanup.

After #134 and #133 both land, add a joint regression where a diagonal tensor and an
independent element-wise tensor are produced in the same loop and both are consumed
downstream.

## Relationship to #133

Recommended landing order:

```text
#134 -> #133 Stage A -> #133 Stage B
```

This is a complexity-ordering recommendation, not a hard prerequisite for isolated #133
target tests. #134 fixes shared-loop ownership and cleanup under the existing index rules.
#133 then changes which index patterns are admissible for virtualization. Keeping that order
separates loop ownership concerns from index-substitution concerns.

If #133 lands first, #134 remains possible, but debugging becomes more entangled: failures
may come from shared-loop bookkeeping, repeated-symbol guards, affine injectivity, or their
interaction.

## Resolved Decisions

- Use a multi-owner `reverse_node_map` for the first landing. Direct candidate discovery from
  loop bodies may be cleaner later, but it is a larger rewrite.
- Do not add a new sibling-count or loop-body-size cap initially. Measure first.
- Do not add special sibling-dependency rejection. Let existing read-before-write and
  memory-mode analysis guard virtual-read soundness.

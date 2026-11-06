# Proposal: Optimize One-Hot Reductions Into Gathers

**Issue:** https://github.com/ahrefs/ocannl/issues/343

## Status

Verified on 2026-06-19 against local HEAD `00f8ae3f`, the live GitHub issue, and a local
`~/tinygrad` checkout at `d7b10c69b`.

- Issue #343 is open, titled "Alternatives to dynamic indexing? Generically optimize one-hot
  encoding pattern for embeddings in low_level.ml", and assigned to milestone `v0.7`.
- The current tree has no low-level dynamic gather scalar. `arrayjit/lib/indexing.ml` `axis_index`
  is still static: `Fixed_idx`, `Iterator`, `Affine`, `Sub_axis`, and `Concat`.
- The previous proposal shape was wrong: this should be an optimization of existing tensor
  expressions, not a new tensor-level representation. Do not add `Equality_with_index`,
  `one_hot_virtual`, a new `fetch_op`, or a new shape rule.
- Existing `Range_over_offsets`, `range`, `eq`, `where`, and einsum-style reduction are enough to
  express a virtual one-hot at the tensor level.
- The old motivating file `test/training/bigram_mlp.ml` is gone. Current in-tree dense-one-hot
  users include `test/training/mlp_names.ml`, `test/training/bigram.ml`, and
  `test/training/fsm_transformer.ml`. `Nn_blocks.one_hot_of_int_list` should be retired or
  rewritten so it no longer produces a dense one-hot Bigarray by default.
- #186 is retired as subsumed by this proposal. Dynamic indexing should not be reintroduced as a
  general `axis_index` or shape-inference feature unless a later scatter, top-k, or KV-cache use
  case reopens it.

## Summary

OCANNL currently uses host-materialized dense one-hot inputs for embedding lookup. For batch token
slots `B`, vocabulary size `V`, and embedding size `D`, that costs `B * V` input storage and
transfer, plus `B * V * D` embedding work.

The optimization target is the existing mathematical pattern:

```text
sum k. (k == token_id[batch]) ? table[k, dim] : 0
```

or equivalently:

```text
sum k. (k == token_id[batch]) * table[k, dim]
```

The compiler should rewrite that to a guarded gather:

```text
if 0 <= token_id[batch] < V then table[token_id[batch], dim] else 0
```

Only the optimized low-level IR needs dynamic indexing. The frontend and shape/projection system
should continue to see ordinary tensor expressions over static axes.

## Tinygrad Reference

tinygrad uses the same idea without introducing a public embedding representation.

Its embedding forward is written as one-hot equality plus reduction:

```python
arange = Tensor.arange(weight.shape[0])
return (arange == idx.unsqueeze(-1)).unsqueeze(-1).where(weight, 0).sum(-2)
```

The codegen rewrite matches the reduction over a range:

```python
((idx != range).where(0, expr)).reduce(range, ADD)
```

and substitutes the range with the runtime index under an in-range guard. The generated kernel then
contains a direct guarded load such as:

```c
(-1 < idx && idx < vocab) ? weight[idx * embed_dim + dim] : 0
```

That is the model OCANNL should follow: use high-level one-hot math as the source program, then
introduce dynamic indexing only as an internal optimized form.

## Decision

Make #343 a **low-level optimization**.

Do not add:

- `Equality_with_index` to `Assignments.fetch_op`;
- `one_hot_virtual` to `Operation` or `Nn_blocks`;
- any `Shape.transpose_type`, terminal type, or shape-inference rule;
- any data-dependent variant to `Indexing.axis_index`.

Add only a low-level gather representation, after shape inference, projection inference, lowering,
virtualization, and scalar simplification have already exposed the pattern.

A concrete low-level shape is:

```ocaml
| Get_dynamic of {
    tn : Tn.t;
    idcs : Indexing.axis_index array;
    dyn_axis : int;
    dyn_value : scalar_arg;
  }
```

`idcs` remains static except at `dyn_axis`, where codegen splices an integer conversion of
`dyn_value` into the row-major offset calculation. This keeps runtime indexing out of the static
index system.

## Source Pattern

The frontend expression should be built from existing operations. Schematically:

```ocaml
let classes = range ~axis_basis:"class" vocab_size
let one_hot = classes = token_ids
let embedded = { w_embed; o = [ embed_dim ] } * one_hot
```

Exact `%op` syntax may need a focused example, but the important constraint is semantic: one-hot is
represented as equality between an existing `range` tensor and an index tensor, then consumed by the
ordinary embedding-table einsum. A small helper is acceptable only if it is pure sugar over existing
operations and introduces no new IR, terminal, fetch, or shape case.

The old dense `Nn_blocks.one_hot_of_int_list` path cannot be optimized reliably by the compiler,
because a materialized Bigarray carries no proof that it is one-hot. Tests and examples that want
this optimization must feed token-id tensors and express one-hot logically.

## API Migration

`Nn_blocks.one_hot_of_int_list` should stop being a dense data producer.

Preferred shape:

- `class_ids_of_int_list` or equivalent creates a compact tensor of class IDs with shape `[ len ]`
  or another caller-specified index shape.
- `one_hot_of_ids ~num_classes ids` builds the logical one-hot from existing operations:
  `range num_classes == ids`, with the needed reshapes/broadcasts.
- `one_hot_of_int_list ~num_classes lst`, if kept for compatibility, should compose those two
  helpers and return the logical one-hot expression, not allocate `[ len; num_classes ]` host data.

If a dense host one-hot is still needed for a test fixture or interop path, give it an explicit name
such as `dense_one_hot_of_int_list`. The default helper should preserve the proof that the tensor is
one-hot by leaving it in the expression graph.

This broadens #343 beyond embeddings in one useful way: labels and other one-hot inputs stop paying
host-side `B * V` allocation and transfer. They may still compute a dense logical one-hot on device
until separate sparse-loss rewrites exist, but the data path is no longer dense by construction.

## Implementation Plan

### 1. Add `Get_dynamic` to `Low_level`

Add the `Low_level.scalar_t` variant and update mechanical consumers:

- precision inference;
- equality / CSE;
- scalar traversals used by CSE and hoisting;
- read collection;
- human pretty-printers;
- `c_syntax.ml` scalar codegen and debug formatting.

Codegen should reuse the existing row-major offset logic where practical. The dynamic component
must render as a concrete integer expression. For a first landing, support only exact integer-valued
indices in a precision that can represent the vocabulary range.

### 2. Rewrite one-hot reductions after simplification

Add a pass between `simplify_llc` and `eliminate_common_subexpressions`:

```ocaml
hoist_cross_statement_cse
@@ eliminate_common_subexpressions
@@ rewrite_one_hot_reductions
@@ simplify_llc
@@ cleanup_virtual_llc ...
```

This mirrors tinygrad's placement: collapse the math pattern before later codegen expansion and CSE
obscure it.

The pass should match reductions with these side conditions:

- the operation is an `Add` reduction over loop variable `k`;
- the reduced expression is either `Where (Cmpeq (Embed_index (Iterator k), index_expr), table_get, 0)`
  or the equivalent multiply-by-comparison form;
- `table_get` is `Get (table, table_idcs)`;
- `k` appears exactly once in `table_idcs`, and only as a pure `Iterator k`;
- `index_expr` does not mention `k`;
- the loop range spans the full selected table axis;
- the body has no side effects beyond the matched accumulation.

The replacement is a guarded `Get_dynamic`:

```text
where (0 <= index_expr && index_expr < class_count)
  (Get_dynamic table table_idcs dyn_axis index_expr)
  0
```

Start narrow. Do not handle `Affine`, `Concat`, convolution-like projections, tropical reductions,
partial ranges, or non-add reductions in the first pass.

### 3. Support both accumulator shapes

The matcher should handle the two common lowered shapes:

- scalar-local form from virtualized reductions:
  `Local_scope` / `Set_local` accumulator;
- materialized form:
  `Zero_out lhs; For k { Set lhs (Get lhs + ...) }`.

If either shape is not cleanly recognizable, leave the original loop intact.

### 4. Retire dense one-hot helpers and convert examples

Update one focused embedding test first. Then update or add variants for the training examples.

For `mlp_names.ml`, the input buffer should become token IDs with shape roughly
`[ batch_size; block_size ]`, not a flat one-hot buffer with shape
`[ batch_size; block_size; vocab_size ]`. The model should derive the one-hot expression from
existing `range` and equality operations before multiplying by the embedding table.

Update `Nn_blocks.one_hot_of_int_list` itself, or replace its call sites with the new compact-ID
and logical-one-hot helpers. Do not leave the old dense helper as the default path for examples.

Targets used for cross-entropy can be represented as logical one-hot too. Optimizing sparse
cross-entropy is a separate task, but the helper should still avoid host-side dense data.

## Out-of-Bounds Semantics

The optimized form must preserve the logical one-hot expression. If an index is outside
`[0, vocab_size)`, the equality reduction contributes zero. Therefore the gather rewrite must be
guarded and return zero out of range.

Do not use an unchecked gather as the canonical rewrite. It is faster but changes semantics.

## Acceptance Criteria

- No new high-level representation is introduced: no `Equality_with_index`, no `one_hot_virtual`,
  no new tensor operation, no new fetch op, and no shape-inference rule.
- `Low_level` gains a dynamic gather representation local to the optimized codegen path.
- The optimizer rewrites the narrow one-hot/equality/add-reduction pattern to guarded
  `Get_dynamic`.
- The optimizer leaves unmatched or unsupported reductions unchanged.
- A focused embedding test demonstrates equivalent results between dense one-hot embedding and the
  logical range/equality expression.
- Generated C for the optimized focused test contains a guarded dynamic table read and no reduction
  loop over the vocabulary axis.
- `Nn_blocks.one_hot_of_int_list` is retired, renamed to an explicit dense helper, or rewritten to
  produce compact class-ID data plus a logical one-hot expression.
- At least one training example can feed token IDs instead of dense context one-hot buffers for
  embedding lookup.
- Existing tests pass with `OCANNL_BACKEND=sync_cc dune runtest`.

## Suggested Tests

1. Forward equivalence: compare dense one-hot matmul with the existing-operation logical one-hot
   expression for a tiny embedding table.
2. Optimization observability: assert the optimized low-level form contains `Get_dynamic`, or assert
   generated C contains a guarded table read and no vocabulary loop.
3. OOB preservation: an out-of-range token ID returns a zero embedding row in the optimized path.
4. Fallback safety: an affine or otherwise unsupported table index keeps the original equality loop
   and still computes correctly.
5. Regression example: update a small version of `mlp_names.ml` to feed token IDs for context
   embeddings and preserve loss within float32 tolerance.
6. Helper migration: assert `one_hot_of_int_list` no longer allocates or uploads
   `len * num_classes` values on the default path.

## Non-Goals

- General dynamic indexing in `Indexing.axis_index`.
- A public gather or embedding primitive.
- Dynamic writes, scatter-add, top-k routing, or KV-cache update semantics.
- Sparse gradients for the embedding table.
- Sparse cross-entropy optimization. Logical one-hot labels are in scope; a fused sparse loss is not.

## Validation Notes

- `Range_over_offsets` already gives a device-side class index through existing lowering.
- `%op` and `%cd` already expose equality and `where`.
- `simplify_llc` already collapses simple local scopes, which makes a post-simplification matcher
  practical.
- tinygrad validates the same architecture: write one-hot/equality math in the frontend, collapse
  it to a guarded dynamic load in codegen.
- The only unavoidable new representation is the low-level gather itself; it should not escape
  `Low_level` and backend codegen.

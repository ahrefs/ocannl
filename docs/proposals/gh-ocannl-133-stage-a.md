# Stage A: Virtualize Repeated-Symbol Producers

Parent: [gh-ocannl-133.md](gh-ocannl-133.md)

Tracked by: https://github.com/ahrefs/ocannl/issues/133

## Goal

Allow virtual node inlining for producers whose index vector repeats a non-static symbol.
This covers diagonal and partially diagonal tensors such as:

```text
[| Iterator i; Iterator i |]
[| Iterator i; Iterator j; Iterator i |]
```

The motivating case is an einsum-style diagonal projection such as `i => ii`. Materializing
that tensor stores mostly zero cells. Inlining should compute the producer value for diagonal
reads and preserve the zero/init value for off-diagonal reads.

## Non-Goals

- Multi-symbol affine producer positions such as `stride * oh + kh`. Those move to
  [Stage B](gh-ocannl-133-stage-b.md).
- `Concat` virtualization. `Concat` must still be eliminated before this pass and should
  continue to raise the existing non-virtual path.
- A new conditional statement form. Existing scalar `Where`, `Cmpeq`, and `Embed_index`
  are enough for this stage.

## Current Failure

`check_idcs` rejects repeated symbols with `Non_virtual 5`. The old check effectively
requires each non-static symbol to appear at exactly one index position.

If that check is simply deleted, `make_subst` still builds one `(symbol, axis_index)` pair
per producer position. A diagonal producer then generates duplicate keys and can crash via
`Map.of_alist_exn`, or it can bind one occurrence while ignoring the consistency condition
for the rest.

## Design

Relax `check_idcs` so repeated symbols are valid. Do not just remove the old uniqueness
check. Replace it with a coverage check that counts non-static symbols across both
`Iterator` positions and supported single-symbol `Affine` positions. This also fixes the
current defect where a single-symbol affine setter can pass the affine branch but still fail
because `num_syms` only counts `Iterator` positions.

Rework `make_subst` environment construction:

1. Collect `(symbol, call_site_index)` pairs from every producer index position.
2. Group pairs by symbol.
3. Use the first binding as the substitution.
4. For later bindings of the same symbol, emit a consistency guard comparing the
   substituted producer index with the call-site index.

Guard form:

```text
Where (Cmpeq (Embed_index lhs_index_after_subst, Embed_index rhs_index),
       inlined_value,
       Get_local id)
```

The existing zero/init local supplies the off-diagonal value. If future init elision means no
zero/init statement is present and a guard is introduced, explicitly initialize the local
before guarded updates.

Add a simplification rule so syntactically identical guards fold away:

```text
Where (Cmpeq (Embed_index a, Embed_index a), then_value, else_value) -> then_value
```

## Implementation Notes

- `subst` already handles nested affine substitutions and is not the main blocker.
- `Fixed_idx` and `Sub_axis` positions should continue to use structural equality.
- Repeated positions with unequal static indices can fold to the init value rather than
  emitting a live guard, but this is a nice-to-have.
- Preserve the early `Concat` rejection. Stage A should not make concat lowering depend on
  virtual inlining.

## Acceptance Criteria

- Diagonal `i => ii` producers virtualize.
- Partially diagonal producers such as `ij => iji` virtualize.
- Generic consumers that read diagonal producers with distinct symbols get guarded inline
  code and numerically match materialized execution.
- Calls where repeated positions are already syntactically equal simplify to unguarded inline
  code.
- Duplicate substitution bindings do not raise `Map.of_alist_exn`.
- Existing single-symbol virtualization tests still pass.
- `Concat` remains rejected before virtualization.

## Tests

Good starting locations:

- `test/einsum/test_surjectivity.ml`
- `test/einsum/test_accumulation_semantics.ml`

Test groups:

- Diagonal producer: `i => ii`.
- Partial diagonal producer: `ij => iji`.
- Generic consumer of a diagonal producer, requiring an equality guard.
- Equal-index consumer of a diagonal producer, requiring simplification to remove the guard.
- Regression for current single-symbol inlining behavior.
- Regression that `Concat` still raises the existing non-virtual path.

Standalone `.expected` files must include the standard two-line config lookup banner.

## Relationship to #134

#134 is not required for isolated Stage A target tests. It is recommended first to keep loop
ownership and cleanup changes separate from repeated-symbol substitution changes.

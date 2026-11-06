# Fix Inline tensor printer crash on concat/row-variable shapes

## Goal

The `~style:`Inline` tensor printer raises `Invalid_argument("index out of
bounds")` on shapes carrying non-empty `beg_dims` (concat / row-variable-expanded
shapes, e.g. block-tensor stacking results or hand-written `einsum1`+`concat`).
This is a pre-existing arrayjit/tensor bug, surfaced during `task-c088bda1`
(tensor stacking), where the stacking fixture (`test/operations/test_block_tensor.ml`)
was forced to use `~style:`Default` purely to dodge the crash. Both the coder and
reviewer recommended keeping it visible as a follow-up.

The fix makes `~style:`Inline` render beg-anchored shapes correctly, so the
stacking fixture (and any future inline print of a concat result) no longer has
to avoid it.

## Acceptance Criteria

- Printing a tensor whose shape carries non-empty `beg_dims` with `~style:`Inline`
  renders without raising `Invalid_argument("index out of bounds")`.
- The three per-row axis-count bindings in `Tensor.to_doc` count the row's full
  physical rank — `List.length (sh.<kind>.beg_dims @ sh.<kind>.dims)` for each of
  `batch`, `input`, `output` — so that `num_all_axes` in
  `Ir.Ndarray.to_doc_inline` equals `Array.length (Nd.dims arr)`.
- A regression test prints a tensor with non-empty `beg_dims` (a block-tensor /
  stacking / concat result) using `~style:`Inline` and asserts it renders without
  raising. The test's expected output is committed (cram-style `.expected`, per
  the project's convention for tests of new/fixed behavior).
- `~style:`Default` rendering of the same shapes is unchanged.
- `dune build` and `dune runtest` pass.

## Context

The crash originates in `tensor/tensor.ml`, in the `%debug5_sexp to_doc` printer.
The per-row axis counts are derived as:

```
let num_batch_axes  = List.length sh.batch.dims in
let num_input_axes  = List.length sh.input.dims in
let num_output_axes = List.length sh.output.dims in
```

These omit `beg_dims`. A `Row.t` is `{ beg_dims; dims; bcast; prov }`, and the
**physical** rank of a row is `List.length (beg_dims @ dims)`. Concat /
row-variable-expanded shapes (block-tensor stacking specs, hand-written
`einsum1`+`concat`) land a fresh outer-anchored axis in `beg_dims`, so
`List.length sh.<kind>.dims` undercounts the physical axes. (Note: a `Concat`
dim such as `bt0^bt1` is **not** the mismatch — `Shape.row_to_dims` maps a
`Concat` to a single physical axis of summed size; the defect is purely the
omission of `beg_dims`.)

The two `~style:`Inline` call sites (tensor value and gradient) pass these three
counts to `Nd.to_doc_inline` (`arrayjit/lib/ndarray.ml`). There,
`num_all_axes = num_batch_axes + num_output_axes + num_input_axes` gates the
recursion depth: `ind = Array.copy dims` is sized to the full physical rank, but
`loop` descends only `num_all_axes` levels before calling `get_as_float arr ind`.
When `num_all_axes < Array.length dims`, the trailing `ind` entries still hold
their initial `dims.(k)` values (axis *sizes*, which are out-of-bounds indices),
producing the `Invalid_argument`.

In-repo precedent: `Shape.default_display_indices` (`tensor/shape.ml`) already
counts `num_input_axes = List.length (sh.input.beg_dims @ sh.input.dims)`, and
the `~style:`Default` path (`Nd.to_doc` / `render_array`) iterates the physical
`Nd.dims arr` directly — which is why `Default` is immune. The inline counts
should follow the same `beg_dims @ dims` convention.

Key pointers (by symbol):
- `tensor/tensor.ml` — `to_doc`: the three `let num_*_axes = List.length
  sh.<kind>.dims` bindings, feeding both `Nd.to_doc_inline` call sites (value at
  the `\`Inline` value branch, gradient at the `\`Inline` gradient branch).
- `arrayjit/lib/ndarray.ml` — `to_doc_inline`: `num_all_axes` and the `loop`
  recursion that crashes; no change needed if the caller is fixed.
- `tensor/shape.ml` — `default_display_indices` (`beg_dims @ dims` precedent),
  `row_to_dims` / `to_dims_impl` (physical ndarray rank).
- `tensor/row.ml` — `type t = { beg_dims; dims; bcast; prov }`.
- `test/operations/test_block_tensor.ml` — the stacking fixture that forced
  `~style:`Default`; the natural home for the regression test (its `stacked`
  tensor is a beg-anchored output-axis concat).

## Approach

*Suggested approach — agents may deviate if they find a better path.*

1. In `tensor/tensor.ml` `to_doc`, change the three bindings to count physical
   rank:
   ```
   let num_batch_axes  = List.length (sh.batch.beg_dims  @ sh.batch.dims)  in
   let num_input_axes  = List.length (sh.input.beg_dims  @ sh.input.dims)  in
   let num_output_axes = List.length (sh.output.beg_dims @ sh.output.dims) in
   ```
2. Add the regression test in `test/operations/test_block_tensor.ml` (or a
   sibling) that prints a beg-anchored shape (e.g. the existing `stacked` tensor)
   with `~style:`Inline` and commits the resulting `.expected` output. This both
   guards against the crash and exercises the corrected per-axis bracketing
   (`[| … |]` / `[ … ]` / `( … )`) for beg-flank axes.

Verify the inline `axes_spec` string and the delimiter boundaries (which key off
`num_batch_axes` and `num_batch_axes + num_output_axes`, now including beg-flank
axes) read sensibly in the committed expected output.

## Scope

In scope: the three axis-count bindings in `Tensor.to_doc` and a regression test.
Out of scope: any change to `Nd.to_doc_inline` itself (the crash site is correct
given correct counts), and any change to `~style:`Default` rendering.

Relates to `task-c088bda1` (tensor stacking), from whose retrospective this was
surfaced. No blocking dependencies.

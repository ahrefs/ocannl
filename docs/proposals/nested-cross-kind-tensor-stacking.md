# General nested cross-kind tensor stacking

## Goal

OCANNL's block-tensor stacking syntax (`[a; b]`, `[|a; b|]`, `(a, b)`) currently
stacks only a **single** level: each desugars to one `stack` call that unsqueezes
a fresh leading axis and concatenates along it. Nesting fails. A literal like
`[[a; b]; [c; d]]` desugars correctly at the PPX level to

```
stack `Output [| stack `Output [| a; b |]; stack `Output [| c; d |] |]
```

but the **shape solver** cannot type it: the outer stack's `einsum1` unsqueeze
introduces a fresh `Dim 1` leading axis *ahead of* the inner stacked tensor's
existing leading `Concat` axis, and the solver has no arm that unifies a fresh
non-`Concat` leading axis against a `Concat` axis. It currently falls through to a
direct `Dim_eq` that equates `Dim 1` with a multi-component `Concat`, which is a
size contradiction.

The fix is **purely in the shape solver** (`tensor/row.ml`). The PPX
(`translate_block_tensor`) and `Operation.stack` already recurse correctly and
need no change. The goal is for nested stacking to **reproduce tensor-literal
construction at arbitrary depth across all three axis kinds** — the governing
invariant being: *a nested stacking expression produces the identical shape the
corresponding nested ndarray literal would.* If literal syntax didn't exist,
stacking would reproduce it.

Surfaced by the retrospective of `task-c088bda1` (single-level stacking, landed via
PR #29 on `lukstafi/ocannl-staging` master, commits `58bfd6e5` + `f313106c`), which
deliberately scoped out shape-solver changes and removed the seed PR #21 Test 5
nested fixture, documenting the gap as a known limitation. Relates to the
einsum-concat shape machinery (`gh-ocannl-49`).

## Acceptance Criteria

1. **Arbitrary-depth, three-way cross-kind nesting types correctly.** Nested
   stacking supports arbitrary nesting depth and the three axis kinds together,
   with the delimiter→axis-kind mapping already fixed by the PPX:
   - `( ... )` tuple → **input** axes (innermost / trailing, after `->`),
   - `[ ... ]` list → **output** axes (middle),
   - `[| ... |]` array → **batch** axes (outermost / leading, before `|`).

   The kinds nest in OCANNL's canonical `batch | output -> input` order. The
   verifiable invariant: a nested stacking expression produces the **identical
   shape** (batch/output/input dims, in order) that the corresponding nested
   ndarray *literal* would produce. Concretely, a same-kind 2-level output nest
   `[[a; b]; [c; d]]` over `output_dims:[3]` operands yields output axes `2; 2; 3`
   (two new size-2 stack axes ahead of the operand's `3`); a cross-kind nest such
   as `[ (a, b) ; (c, d) ]` introduces a new output axis (outer list) and a new
   input axis (inner tuple) at their respective kinds, not both on the same kind.

2. **Mixed-rank sibling rows are reconciled by inference, not rejected or
   auto-unsqueezed.** A literal like `[[a; b]; c]` is **well-formed**: shape
   inference forces the under-specified sibling to acquire the stacked rank of its
   siblings — `c` is inferred as `[d; e]` (it gains the same leading stack axis its
   sibling row carries), rather than the solver raising a shape error or silently
   auto-unsqueezing `c`. (This supersedes the Tentative Design's earlier "clean
   shape error" recommendation.)

3. **Regression fixtures restored and extended.** In
   `test/operations/test_block_tensor.ml`:
   - The seed PR #21 **Test 5** same-kind 2-level nest `[[a; b]; [c; d]]` is
     restored (replacing the current "removed / known limitation" comment block) as
     a forward regression with re-baselined `.expected`.
   - A **deeper-nest** fixture (≥3 levels, same kind) is added.
   - A **cross-kind nest** fixture (mixing at least two of input/output/batch
     delimiters across levels, e.g. `[ (a, b) ; (c, d) ]`) is added.
   - A **mixed-rank** fixture asserting AC #2 (the under-specified sibling's rank is
     inferred up; the literal does not error).

4. **Nested gradient flow composes correctly.** The Test 8 "grad of 3-way stacked"
   precedent is extended to a nested case: backward through a nested stack
   propagates the correct gradient to each leaf operand (asserted via the existing
   `sin`-then-sum-loss pattern and re-baselined `.expected`).

5. **The `docs/syntax_extensions.md` known-limitation note is removed.** The
   bullet "**Known limitation:** nested block matrices `[[ta; tb]; [tc; td]]` …
   are not yet supported by the shape solver; only single-level stacking works
   currently." is deleted (and the surrounding prose updated if it implies nesting
   is unsupported).

6. **No `Operation`/PPX surface change.** The fix lives in `tensor/row.ml` only.
   `Operation.stack`, `einsum1`/`concat`, and `ppx_op.ml`'s `translate_block_tensor`
   are not modified (beyond what re-baselining tests requires). The existing
   single-level tests (Tests 1–4, 6–15) continue to pass.

7. **Tests pass with no unrelated regressions.** The `test_block_tensor` suite and
   the einsum / operations suites pass; any fixtures whose generated codegen shifts
   (e.g. `rope_test.expected`) are re-baselined deliberately, not masked.

## Context

How things work now:

- **`tensor/operation.ml` — `stack (axis : stack_axis) ?grad_spec rhses`.** Already
  generic over operand count and axis kind. Per kind it builds an `unsqueeze_spec`
  (`einsum1` RHS `0` mints a fresh size-1 axis on the chosen kind) and a
  `concat_spec` (`bt0^bt1^…` along that axis), maps `einsum1 unsqueeze_spec` over
  each operand, then `concat`s. The result's new axis is a `Concat [Var bt0; …]`.
  No change expected here.

- **`tensor/ppx_op.ml` — `translate_block_tensor`.** Maps the recursive `loop`
  (= `translate`) over each element *before* emitting the `stack` call, so nested
  literals desugar to nested `stack` calls cleanly. Dispatch arms route list →
  ``stack `Output``, array → ``stack `Batch``, top-level tuple → ``stack `Input``,
  with `is_ndarray_constant_expr` first-leaf guard routing numeric literals to
  `ndarray`. Confirmed recursion-correct; no change expected.

- **`tensor/row.ml` — the fix site.** Key arms in the dim-level solver
  (`solve_dim_ineq` family, around the `match (res, opnd)` that handles `Concat`):
  - `Concat dims1, Concat dims2 when List.length dims1 = List.length dims2` —
    element-wise unification; works when **both** sides are already `Concat` of
    equal length (this landed on master independently and is *not* the gap).
  - `_, Concat dims when (count discardable Var components) >= length - 1` —
    preserves the inequality so the solver can infer the concat-component vars.
  - **`Concat _, _ | _, Concat _` (final fallthrough)** — defers to
    `Dim_eq { d1 = res; d2 = opnd }`, *directly equating a non-`Concat` dim with a
    `Concat`*. This is the break point: the outer unsqueeze's fresh `Dim 1` leading
    axis is paired here against the inner stack's leading `Concat`, and `Dim 1 =
    Concat [components summing to >1]` is a size contradiction.
  - Row-level pairing in `solve_row_ineq`: leading-flank axes are paired
    positionally — `List.take res.beg_dims beg_dims_l` against
    `List.take opnd.beg_dims beg_dims_l` via `Dim_ineq` — which is exactly where the
    fresh-leading-vs-`Concat` pairing is generated and routed into the dim arms
    above.
  - `apply_dim_constraint`: the `Concat _dims, At_least_dim _d_min` arm is a known
    `FIXME: reconsider if we can make progress` no-op — it does not propagate a
    lower-bound constraint into `Concat` components. Relevant if the nested case
    routes a rank/size lower bound through a `Concat`.
  - `Concat _ -> failwith "NOT IMPLEMENTED YET"` sites in `_lift_row_constraint`
    and `s_dim_one_in_row_constr` — if the nested case routes through row-constraint
    reduction (e.g. `Total_elems`), one of these may be the actual exception that
    surfaces. The implementer should first confirm *which* error the nested literal
    raises today (a `failwith` vs. a `Shape_error` from the `Dim_eq` contradiction),
    since that pins the precise arm to fix.

- **`test/operations/test_block_tensor.ml`** — 15 single-level cases pass today;
  Test 5 is a removed-with-comment placeholder; Test 8 is the 3-way grad precedent.

- **`docs/syntax_extensions.md`** — documents the delimiter→axis-kind mapping and
  carries the known-limitation bullet to be removed.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

The implementer should **first reproduce and pinpoint the failure**: build the
nested literal `[[a; b]; [c; d]]` in a scratch test and capture the exact exception
(which arm/`failwith`/`Shape_error` fires). That determines whether the fix is (a)
teaching the dim-level solver to unify a fresh non-`Concat` leading axis against a
`Concat` (the most likely site: replacing the contradiction-inducing `Dim_eq` in
the `Concat _, _ | _, Concat _` fallthrough with logic that shifts/absorbs the
fresh size-1 axis), and/or (b) one of the `NOT IMPLEMENTED YET` `Concat` arms in
the row-constraint reduction path. The invariant — *nested stacking reproduces the
nested-literal shape* — is the oracle for every fixture: write the equivalent
nested ndarray literal (or compute its shape by hand) and assert the stacked result
matches.

**One task, internally phased — not a split.** The three sub-capabilities
(same-kind arbitrary depth / cross-kind / inference-driven sibling-rank
unification) all converge on the **same** `Concat`-handling arms in `tensor/row.ml`
and the **same** regression fixture file, and none can land on main independently
in a coherent way: cross-kind nesting is the same unsqueeze-vs-`Concat` machinery
applied at a different axis kind, and sibling-rank inference is a property of the
same unification step. Splitting would fragment one solver change across PRs that
each leave the fixture file half-restored. The natural *implementation* ordering is
incremental — get same-kind depth green first (restores Test 5), then verify/extend
cross-kind, then the mixed-rank inference fixture — but it ships as one merge.

## Scope

In scope:
- `tensor/row.ml` shape-solver changes for fresh-leading-axis-vs-`Concat`
  unification at arbitrary depth and across axis kinds, plus sibling-rank
  inference for mixed-rank rows.
- Restored + new fixtures in `test/operations/test_block_tensor.{ml,expected}`
  (Test 5, deeper nest, cross-kind nest, mixed-rank, nested grad) and any
  re-baselined downstream `.expected` (e.g. `rope_test.expected`).
- Removal of the `docs/syntax_extensions.md` known-limitation note.

Out of scope:
- Any change to `Operation.stack`, `einsum1`/`concat`, or `ppx_op.ml`'s
  `translate_block_tensor` (the recursion already works).
- New stacking syntax or a new public operation surface.
- The `gh-ocannl-49` `^` concat-operator surface (aligned but separate).

Dependencies: builds on `task-c088bda1` (single-level stacking, landed on
`lukstafi/ocannl-staging` master). The PR opens against `lukstafi/ocannl-staging`
(pass `--repo lukstafi/ocannl-staging` explicitly; the fork stays a fork).

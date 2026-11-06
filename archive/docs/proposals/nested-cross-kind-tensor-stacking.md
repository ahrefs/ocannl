# Arithmetic Concat equality for nested cross-kind tensor stacking

## Goal

OCANNL's dim unifier is missing general arithmetic equality for `Concat`
dimensions. At the shape-solving level, `Concat` should be interpreted as a sum
of component dimensions, not as a structural list that only matches equal-arity
lists pointwise. That gap currently surfaces through block-tensor stacking syntax
(`[a; b]`, `[|a; b|]`, `(a, b)`), where single-level stacks work but nested stacks
fail. A literal like `[[a; b]; [c; d]]` desugars correctly at the PPX level to

```
stack `Output [| stack `Output [| a; b |]; stack `Output [| c; d |] |]
```

but the **shape solver** cannot type it because the outer stack has to prove that
the two already-stacked inner operands have the same remaining shape. Each inner
stack creates an independent concat axis, so strict unification reaches a dim
equality of the form:

```
Concat [a; b] = Concat [c; d]
```

Today `unify_dim` has no `Concat` equality normalization, so this case falls
through to the generic `"solved dimensions for axis: mismatch"` error. This is a
missing general `Concat` arithmetic-equality branch, not a stacking-specific
operation bug.

The intended shape-solving semantics are arithmetic:

- `Concat xs = Var v` or `Var v = Concat xs`: bind `v` to `Concat xs`, subject to
  the existing occurs check and variable-constraint machinery.
- `Concat [x]`: unwrap to `x` before continuing.
- `Concat xs = Dim n`: remove solved `Dim` components from `xs`, subtract their
  sizes from `n`, and continue with the residual equation. Fail if the residual
  solved size would be negative or if non-neutral basis labels conflict.
- `Concat xs = Concat ys`: cancel variables that repeat on both sides and
  normalize solved `Dim` components so that only one side carries the remaining
  solved size. The residual equation should then proceed through the existing
  variable binding, dim solving, or mismatch paths.
- Basis labels on `Concat` components use neutral-label semantics: `default` and
  `bcast_if_1` are neutral, while all non-neutral solved components in a reduced
  concat must agree on the same basis. When reducing solved `Dim` components, the
  agreed non-neutral basis survives onto the residual solved `Dim`; distinct
  non-neutral bases conflict. If only neutral basis components contribute to the
  residual solved `Dim`, use `default` for the result basis.

The shape-typing fix belongs in the shape solver (`tensor/row.ml`). The PPX
(`translate_block_tensor`) and `Operation.stack` already recurse correctly and
need no change. Once shape inference succeeds, projection inference may expose a
second layer of concat handling to generalize. The goal is for nested stacking to
**reproduce tensor-literal construction at arbitrary depth across all three axis
kinds** — the governing invariant being: *a nested stacking expression produces
the identical shape the corresponding nested ndarray literal would.* If literal
syntax didn't exist, stacking would reproduce it.

Surfaced by the retrospective of `task-c088bda1` (single-level stacking, landed via
PR #29 on `lukstafi/ocannl-staging` master, commits `58bfd6e5` + `f313106c`), which
deliberately scoped out shape-solver changes and removed the seed PR #21 Test 5
nested fixture, documenting the gap as a known limitation. Relates to the
einsum-concat shape machinery (`gh-ocannl-49`).

## Acceptance Criteria

1. **Arbitrary-depth, three-way cross-kind nesting types correctly.** Nested
   stacking supports arbitrary nesting depth and the three axis kinds together,
   with the delimiter→axis-kind mapping already fixed by the PPX. The nesting
   reproduces the **lowered (memory) axis order**, which is *input-last* —
   `batch @ output @ input` (inputs are trailing because they are reduced over in
   the inner loop) — so the **outermost** delimiter introduces the **leading**
   axis and the **innermost** the **trailing** axis:
   - `[| ... |]` array → **batch** axes — outermost / leading in memory (left of `|` in textual syntax),
   - `[ ... ]` list → **output** axes — middle (right of `->` in textual syntax),
   - `( ... )` tuple → **input** axes — innermost / trailing in memory (left of `->` in textual syntax).

   (Note the textual einsum/printing syntax `batch | input -> output` follows the
   type-systems convention with inputs *left* of `->`; it differs from the memory
   order, where inputs are last. The nesting follows the memory order.) The
   verifiable invariant: a nested stacking expression produces the **identical
   shape** (batch/output/input dims, in memory order) that the corresponding nested
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

7. **Projection inference is not forgotten.** Once shape inference succeeds,
   projection inference will also need to handle the resulting concat arithmetic.
   The proposal deliberately does not prescribe that algorithm: projection
   inference cannot simply use the shape solver's arithmetic interpretation, so
   the implementer should generalize the projection machinery according to the
   concrete normalized shapes and failing projection tests they expose.

8. **Tests pass with no unrelated regressions.** The `test_block_tensor` suite and
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

- **`tensor/row.ml` — the shape-inference fix site.** Key arms in the dim-level
  solver:
  - `solve_dim_ineq` has some concat-specific inequality handling, but strict
    equality in `unify_dim` lacks general arithmetic normalization. When a
    `Dim_eq` reaches `Concat [a; b] = Concat [c; d]`, it falls through to the
    generic mismatch. This is the core gap exposed by nested stacking.
  - `unify_dim` already has the symmetric `Var v, dim | dim, Var v` arm, so
    `Concat xs = Var v` should continue to work by binding `v` to `Concat xs`;
    the new normalization should not interfere with that path.
  - Single-component concat should be erased (`Concat [x] -> x`) before solving.
  - `Concat = Dim` should move solved `Dim` components out of the concat side and
    subtract them from the solved dim side, leaving one residual solved `Dim`
    component at most.
  - `Concat = Concat` should cancel repeated variables from both sides and
    consolidate solved `Dim` components so that only one side carries the solved
    residual.
  - Concat basis reduction should treat `default` and `bcast_if_1` as neutral
    labels. A concat may combine neutral solved components with any one
    non-neutral basis; distinct non-neutral bases conflict. When solved
    components are subtracted away, the surviving non-neutral basis, if any,
    should be preserved on the residual solved dimension; if no non-neutral basis
    survives, the residual solved dimension should use `default`.
  - Do **not** rewrite affine convolution dimensions to `Concat [over; kernel]`.
    For `use_padding = false`, the existing dimension formula is
    `stride * (over - 1) + effective_kernel_span`; with `stride = 1` and
    `dilation = 1`, this is `over + kernel - 1`, not `over + kernel`.
  - Row-level unification is the route by which nested stacking exposes the
    missing branch: the outer stack compares the shapes of its already-stacked
    operands, and the inner operands' leading axes are independent `Concat`
    dimensions that must be equated arithmetically.
  - `apply_dim_constraint`: the `Concat _dims, At_least_dim _d_min` arm is a known
    `FIXME: reconsider if we can make progress` no-op — it does not propagate a
    lower-bound constraint into `Concat` components. Relevant if the nested case
    routes a rank/size lower bound through a `Concat`.
  - `Concat _ -> failwith "NOT IMPLEMENTED YET"` sites in `_lift_row_constraint`
    and `s_dim_one_in_row_constr` — these are separate unimplemented concat cases
    that may surface for row-constraint reduction (e.g. `Total_elems`), but they
    are not expected for the basic nested-stack reproducer above.

- **`test/operations/test_block_tensor.ml`** — 15 single-level cases pass today;
  Test 5 is a removed-with-comment placeholder; Test 8 is the 3-way grad precedent.

- **`docs/syntax_extensions.md`** — documents the delimiter→axis-kind mapping and
  carries the known-limitation bullet to be removed.

- **Projection inference** — once the shape solver accepts nested concat
  arithmetic, projection inference may fail next. This proposal intentionally
  flags that work without prescribing a mechanical translation of the arithmetic
  rules above. Shape inference may solve by cancellation and numeric
  normalization; projection inference still has to decide how projected slices
  flow through concrete concat axes.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

The implementer should **first reproduce and pinpoint the failure**: build the
nested literal `[[a; b]; [c; d]]` in a scratch test and capture the exact exception
(which arm/`failwith`/`Shape_error` fires). The expected current failure is a
`Shape_error "solved dimensions for axis: mismatch"` from `unify_dim` on
`Concat [...] = Concat [...]`. If a different case surfaces, handle that evidence,
but do not special-case stacking.

The primary implementation is general arithmetic normalization for `Concat`
strict equality in `unify_dim`, not structural list equality. Normalize before
falling through to existing equality cases:

1. Unwrap single-component concats.
2. Preserve `Concat = Var` variable binding through the existing variable path.
3. For `Concat = Dim`, subtract solved `Dim` components from the solved side and
   continue with the residual equation.
4. For `Concat = Concat`, cancel repeated variables on both sides and normalize
   solved `Dim` components so only one side carries a solved residual.

For basis compatibility, treat the special labels `default` and `bcast_if_1` as
neutral. All non-neutral solved components in a reduced concat must share the
same basis. When subtracting solved `Dim` components from a concat, neutral bases
drop out, a surviving non-neutral basis remains attached to the residual solved
dimension, and distinct non-neutral bases are an error. If only neutral basis
components contribute to a residual solved dimension, use `default` as its basis.
The purpose is to express valid dimension arithmetic, not to make every concat
equation solvable.

After shape inference succeeds, expect a second layer of work in projection
inference. Do not assume the projection algorithm can reuse the arithmetic
interpretation directly; use the newly passing shape cases to expose the failing
projection cases and generalize projection handling there.

The invariant — *nested stacking reproduces the nested-literal shape* — is the
oracle for every fixture: write the equivalent nested ndarray literal (or compute
its shape by hand) and assert the stacked result matches.

**One task, internally phased — not a split.** The three sub-capabilities
(same-kind arbitrary depth / cross-kind / inference-driven sibling-rank
unification) all converge on the **same** `Concat` arithmetic / row-unification
behavior in `tensor/row.ml` and the **same** regression fixture file, and none can
land on main independently in a coherent way: cross-kind nesting is the same
concat-axis arithmetic problem at a different axis kind, and sibling-rank inference
is a property of the same unification step. Splitting would fragment one solver
change across PRs that each leave the fixture file half-restored. The natural
*implementation* ordering is incremental — get same-kind depth green first
(restores Test 5), then verify/extend cross-kind, then the mixed-rank inference
fixture — but it ships as one merge.

## Scope

In scope:
- `tensor/row.ml` shape-solver changes for general arithmetic `Concat` strict
  equality: single-component unwrapping, `Concat = Var` binding through the
  existing variable path, `Concat = Dim` subtraction, neutral/non-neutral basis
  handling, and `Concat = Concat` cancellation/solved-dim normalization.
- Any additional row/dim solver changes needed for arbitrary-depth and cross-kind
  nesting, plus sibling-rank inference for mixed-rank rows, if the regression
  fixtures expose more than the missing equality branch.
- Projection-inference generalization required once shape inference succeeds,
  with the details left to implementation based on the concrete projection
  failures that surface.
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

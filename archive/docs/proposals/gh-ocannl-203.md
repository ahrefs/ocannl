# Re-introduce within-shape scaling constraints (`Shape.set_scale`)

## Goal

Issue: https://github.com/ahrefs/ocannl/issues/203

Express *non-convolutional* multiplicative shape relationships at the user-facing
API. The motivating use cases are hourglass-style networks: MLP autoencoders with
information-bottleneck layers, MLPs with hidden-layer expansion for feature
extraction, and similar architectures where one dimension is a constant
multiple of another and both should adapt to data shape together.

The convolutional case is already handled by `Affine { conv = Some _; _ }` in
`tensor/row.ml`. What is still missing is a user-facing entrypoint that lets
authors of `nn_blocks`-style code state "this dimension equals N times that
dimension" without dropping into the einsum / row-constraint machinery
directly.

This proposal deliberately follows the narrowed scope confirmed by the user on
2026-05-05: add a single API entrypoint analogous to `Shape.set_equal`, reuse
the existing `Affine` data type (no datatype changes), and stay out of the
equational solver.

## Acceptance Criteria

- [ ] **API exists.** `Shape.set_scale` is exposed in `tensor/shape.mli` with
      a signature mirroring `Shape.set_equal`, taking the two `delayed_var_ref`
      arguments and one extra `int` parameter for the multiplicative factor
      (final argument naming/order is an implementation detail; the signature
      should read naturally as "the larger ref equals factor times the smaller
      ref").
- [ ] **Documentation.** The `.mli` carries a docstring explaining the
      relationship being asserted, mirroring the style of the `set_equal`
      docstring directly above it.
- [ ] **Constraint emission.** When neither variable is yet solved and both
      are dimension variables, `set_scale` emits a `Row.Dim_eq` constraint
      whose RHS is `Affine { stride = factor; over = Var other; conv = None;
      stride_offset = 0 }` (the same shape the solver already produces
      internally at `tensor/row.ml` around line 832 in the
      "Use Affine with stride and no convolution" branch). The `origin` field
      is populated with an `operation = Some "Shape.set_scale ..."` tag, in
      the same style as the existing `set_equal` operations.
- [ ] **Solved-side propagation.** When one ref is already solved with
      dimension `d`, `set_scale` propagates the implied dimension to the other
      ref via `set_dim`, applying the factor in the appropriate direction.
      When both are solved, it raises `Row.Shape_error` if the values do not
      satisfy the scaling relation (mirroring the both-solved branch of
      `set_equal`).
- [ ] **Conv path untouched.** No regression in the existing convolutional
      `Affine` handling: no production `Affine` value with `conv = Some _`
      changes shape, and the existing `test/einsum/test_conv_*` and
      `test/einsum/test_max_pool2d` expectation files are unchanged.
- [ ] **Test coverage.** A test under `test/einsum/` (e.g.
      `test_set_scale.ml`) exercises a hourglass-style relation between two
      otherwise-unconstrained dimension variables — for example, declaring a
      hidden layer's width to be `2 * input_width`, supplying a concrete
      `set_dim` for the input, and verifying the hidden width resolves to the
      expected value (and the symmetric case where the hidden width is set
      first and the input is inferred). The test ships with a `.expected`
      file and is wired into `test/einsum/dune`.
- [ ] **No solver edits.** `tensor/row.ml` is not modified for this task. If
      a missing solver case is discovered while writing the test, surface it
      as a follow-up rather than expanding the scope here.
- [ ] **Issue tracking.** After the implementation PR merges, post a comment
      on https://github.com/ahrefs/ocannl/issues/203 linking the merged PR
      and noting that the narrow scope (hourglass relations via
      `Shape.set_scale`) is what shipped, with the broader constraint-system
      work explicitly out of scope. Then close the issue.

## Context

How things work now:

- `tensor/shape.mli` exposes `set_equal : delayed_var_ref -> delayed_var_ref
  -> unit` (with a fully-set-up docstring) plus `set_dim : delayed_var_ref ->
  int -> unit`. These are the user-facing knobs for asserting equalities into
  the shape inference engine from `%op` / DSL code.
- `tensor/shape.ml` implements `set_equal` by case-splitting on whether each
  ref's `solved_dim` is already populated. The solved/unsolved cases delegate
  to `set_dim`; the both-unsolved case appends a `Row.Dim_eq` (or `Row_eq`,
  or a `Total_elems` `Rows_constr` for mixed dim/row) onto
  `active_constraints`, with an `origin` record tagged
  `operation = Some "Shape.set_equal ..."`.
- `tensor/row.ml` already defines the `Affine` constructor with a stride and
  an optional `convolution`. The `conv = None` form is exactly the shape
  needed for non-convolutional multiplicative scaling, and the solver
  already produces it internally — see the branch in `apply_total_elems`
  (around the comment `"Use Affine with stride and no convolution"`) where
  it emits `Affine { stride = coefficient; over = Var single_var;
  conv = None; stride_offset = 0 }`.
- The solver handles `Affine { conv = None; _ }` against integer dimensions
  in the `Dim_eq` reduction rules in `row.ml` (search for `Affine { stride;
  over = Dim s; conv = None; ... }`). So a user-emitted `Affine` with
  `conv = None` rides the existing code paths.
- Existing call sites for `set_equal` live in `lib/nn_blocks.ml` (none
  currently — only `set_dim` is used there) and in `test/ppx/test_ppx_name_conflict*.ml`.
  The hourglass use cases will be added by users in their own `%op` code;
  this task adds the API, not new `nn_blocks` exports.

Key files and symbols (no line numbers — use grep):

- `tensor/shape.mli` — `val set_equal`, `val set_dim`, `val
  get_variable_ref`, `type delayed_var_ref`.
- `tensor/shape.ml` — `let set_equal`, `let set_dim`,
  `active_constraints` ref, `Row.Dim_eq` / `Row.Rows_constr` /
  `Row.Row_eq` constraint emission.
- `tensor/row.ml` — `type dim` (the `Affine` constructor), the
  `apply_total_elems` "Use Affine with stride and no convolution" branch
  (template for the constraint we want to emit), `dim_to_string` for the
  `Affine { conv = None; _ }` printing.
- `test/einsum/dune` and existing `test/einsum/test_*` files — pattern for
  adding a new test that runs under the einsum test executable and has a
  matching `.expected` snapshot.
- `test/ppx/test_ppx_name_conflict.ml` — minimal example of `Shape.set_equal`
  being called from `%op` code.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

1. Add `val set_scale : delayed_var_ref -> delayed_var_ref -> int -> unit`
   to `tensor/shape.mli`, with a docstring placed adjacent to `set_equal`.
   (Argument convention suggestion: `set_scale ~factor large small` such
   that `large = factor * small`. Pick the convention that reads cleanest;
   document it.)
2. Implement `set_scale` in `tensor/shape.ml` by adapting `set_equal`:
   - Both solved → check `dim_large = factor * dim_small`; raise
     `Row.Shape_error` on mismatch.
   - One solved, one unsolved → call `set_dim` on the unsolved side with
     the value implied by the factor (with a clean error if the implied
     value is non-integral).
   - Both `Dim` vars → emit `Row.Dim_eq { d1 = Var large_var; d2 = Affine
     { stride = factor; over = Var small_var; conv = None; stride_offset =
     0 }; origin = [...] }` with `operation = Some "Shape.set_scale Dim-Dim"`.
   - Row-row, dim-row, row-dim cases: out of scope for this task — raise
     a clear `Shape_error` saying `set_scale` requires two dimension
     variables. Document the limitation in the docstring; broader support
     can be added later if a use case appears.
3. Add `test/einsum/test_set_scale.ml` (and `.expected`, plus dune wiring)
   exercising:
   - Hidden = 2 * input, then `set_dim input 8`, expect hidden resolves to 16.
   - The reverse: `set_dim hidden 16`, expect input resolves to 8.
   - The both-solved-and-mismatched error case.
4. Run the test and the existing einsum/ppx suites to confirm no
   regressions in the conv / pool snapshots.

## Scope

In scope:
- A new `Shape.set_scale` API in `tensor/shape.ml` / `tensor/shape.mli`
  for the dim-var × dim-var × integer-factor case, emitting the existing
  `Affine { conv = None; _ }` constraint shape.
- One new test under `test/einsum/`.
- A docstring update on the new function.
- A comment + close on GH issue #203 once the PR merges.

Out of scope (explicitly):
- Convolutional scaling (already shipped via `Affine.conv` — do not
  touch).
- Solver changes in `tensor/row.ml`. If the test surfaces a missing
  reduction, file a follow-up issue and either choose a slightly different
  test setup that the solver already handles, or stop and ask.
- Row-variable / mixed dim/row variants of `set_scale`.
- Rational / float strides — only integer factors.
- New `nn_blocks` helpers built on top of `set_scale` — those can come
  later, in their own task.
- Broader "constraint language" extensions or syntax (e.g., einsum-level
  scaling notation).

Dependencies: none — this rides on already-merged convolution support
(`Affine` constructor in `row.ml`).

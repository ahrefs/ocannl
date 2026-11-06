# Prohibit `Compose` shape logic for `/` and `**`

> **Rescoped 2026-06-12 (per Łukasz)**: the original draft prohibited `Compose` for
> *all* non-multiply binary ops; that was too aggressive. Only `/` (Div) and `**`
> (ToPowOf) are the confusing cases (they read like matrix inverse / matrix power but
> are nothing of the sort). Other non-multiply combinations stay permitted — pairing
> `Compose` contraction structure with other ops is semiring-style territory (cf. the
> tropical `@^+` einsum operator, max-reduce with add) and may be intentional.

## Status update (2026-06-12)

- Issue #192 is OPEN, label `bug`, milestone v1.0 (per ROADMAP.md, v1.0
  targets end of October 2026; the GH milestone due date lags).
- **Not yet implemented**: `tensor/ppx_cd.ml` still maps `~logic:"@"` to
  `Shape.Compose` for *any* binary op with no guard (the binary mapping is
  now at `ppx_cd.ml:1568`).
- Structural drift since this was written: the binary `~logic` translation is
  now **one shared body for three or-patterns** (tupled, curried-flat, and
  curried-nested application forms, `ppx_cd.ml` ~1543–1573), not "two
  duplicated patterns". The proposed guard therefore needs only a single
  insertion point. The ternary (`Compose_accumulate` / `fma`) branch follows
  immediately after, unchanged in role.
- Other code pointers re-verified as current: `binary_ops` with the `*` and
  `/` "no default compose type" ppx errors in `tensor/ppx_shared.ml`
  (~lines 430–445); `compose_op_of_spec` in `tensor/operation.ml:42` still
  produces only `Pointwise_bin` / `Einsum`; the `Broadcast (Compose, sh1,
  sh2)` inference branch is at `tensor/shape.ml:554`. Issue #305 (ternary
  einsum) is still open and still referenced in the neighbouring error
  message.
- In-tree audit (re-run today): no uses of `~logic:"@"` with a non-multiply
  binary op anywhere in `tensor/`, `lib/`, `test/`, `bin/`; the
  `docs/slides-*.md` snippets all pair `~logic:"@"` with `*`.
- Repo-wide changes since April 2026 (broadcast-order reversal LUB→GLB,
  "label"→"basis" rename) do not affect this proposal's design.
- Remains to do: the whole change — the ppx guard, positive/negative tests,
  and the `docs/syntax_extensions.md` note (now scoped to `/` and `**` only,
  per the rescope note above).

## Goal

Resolve [issue #192](https://github.com/ahrefs/ocannl/issues/192): division
(`/`) and to-power-of (`**`) combined with the `Compose` composition type
(`~logic:"@"`) currently compile successfully via the `%cd` ppx but produce
mathematically meaningless results — silently. Either the semantics is
broken (e.g., `=:+ a / b ~logic:"@"` computes `lhs[i,j] := sum_k a[i,k] / b[k,j]`,
not matrix inverse) or the combination should be prohibited. Per the issue
body, expanding as matrix pseudo-inverse is overkill, so we prohibit.

## Acceptance Criteria

- [ ] Using `~logic:"@"` (i.e. `Compose`) with `/` / `div` (`Div`) or
      `**` / `pow` (`ToPowOf`) produces a clear compile-time error from the
      `%cd` ppx. The error names the offending op and points the user at
      pointwise (`~logic:"."`) or einsum notation as alternatives.
- [ ] Using `~logic:"@"` with `*` / `mul` continues to work as before
      (matrix multiply / tensor contraction), and other non-multiply binary
      ops also remain accepted (semiring-style combinations are deliberate).
      No regression in existing tests, examples, or `docs/slides-*.md`
      snippets that use `=:+ a * b ~logic:"@"`.
- [ ] At least one positive test (or expect-test) demonstrates that
      `=:+ a * b ~logic:"@"` still compiles.
- [ ] At least one negative test demonstrates that
      `=:+ a / b ~logic:"@"` and `=:+ a ** b ~logic:"@"` produce a
      compile-time error with a useful message. (Negative tests can use
      a `dune` `cram` test, an expect-test capturing a ppx error, or a
      manually-checked `.expected` showing the error — pick whichever
      style the project already uses for ppx error tests.) Optionally a
      positive compile-only check that some other non-multiply op with
      `~logic:"@"` is still accepted.
- [ ] The ternary `Compose_accumulate` path (used by `fma`) is left
      untouched — it is a real, intentional use of compose-style
      contraction.
- [ ] If existing code in `tensor/`, `lib/`, `bin/`, `test/`, or the
      example slides uses `~logic:"@"` with `/` or `**`, it is either fixed
      (switching to `~logic:"."` or einsum) or, if intentional and
      meaningful, surfaced for user decision before the prohibition lands.
      (Audit re-run 2026-06-12: no such uses exist in-tree.)

## Context

### Where `Compose` enters from user code

The `%cd` ppx (`tensor/ppx_cd.ml`) parses an expression of the form
`accu_op lhs (bin_op rhs1 rhs2 ~logic:"<spec>")` and turns the spec into
a `Shape.compose_type`:

```ocaml
if String.equal spec "." then [%expr Shape.Pointwise_bin]
else if String.equal spec "@" then [%expr Shape.Compose]
else [%expr Shape.Einsum ([%e logic], [])]
```

This branch is reached from two duplicated patterns (curried and tupled
`bin_op` application forms) — both should get the new check. *(Update
2026-06-12: the patterns have since been merged into three or-patterns
sharing a single body, so one check suffices.)* The lookup
`binary_op bin_op` resolves `bin_op` to a primitive `Ir.Ops.*` constructor
via the `binary_ops` table in `tensor/ppx_shared.ml`. We need to inspect
which primitive op is being used and reject `Compose` for `Div` and
`ToPowOf`.

### Why the bug is silent today

`tensor/operation.ml`'s top-level helpers `pointpow` (`**.`) and `pointdiv`
(`/.`) are hardwired to `compose_op_of_spec`, which only produces
`Pointwise_bin` or `Einsum` — so the public OCaml-level operators cannot
reach `Compose`. The hole is the `%cd` DSL, where `~logic:"@"` is accepted
for *any* entry in `binary_ops`, including `/`, `**`, `<`, `||`, `max`,
`min`, etc.

Once the `Compose` shape is constructed, `Shape.get_inequalities` (the
`Broadcast (Compose, sh1, sh2)` branch in `tensor/shape.ml`) emits the
matrix-multiply-style row inequalities — pairing `sh1.input` with
`sh2.output` for contraction — and `arrayjit/lib/assignments.ml` lowers
the resulting projections by iterating over `product_space` and applying
the chosen op. With `=:+` (initialize-neutral, `Add` accumulator) and
`Div`, the lowering computes `sum_k a[i,k] / b[k,j]`. There is no
warning, no error, no documentation; it just runs.

There is precedent for this kind of compile-time guard: the `*` entry in
`binary_ops` already raises a ppx error if no `~logic` is given
("No default compose type for binary `*`, try ..."), and the `/` entry
does the same. The proposed check extends the same style of guard to the
case where `~logic:"@"` is supplied for `/` or `**` — the two ops whose
`Compose` reading (matrix inverse, matrix power) does not match what the
lowering computes. Other non-multiply ops are deliberately not guarded:
contraction structure over other semiring-style op pairs can be meaningful
(the tropical `@^+` einsum operator is the in-tree precedent).

### Key code pointers

- `tensor/ppx_cd.ml` — the two `Pexp_constant (Pconst_string (spec, ...))`
  branches that translate `~logic:"@"` to `Shape.Compose` for binary ops
  (search for `Shape.Compose`). The ternary branch right below them
  handles `fma` / `Compose_accumulate` and should be left alone.
- `tensor/ppx_shared.ml` — `binary_ops` table; the `Ir.Ops.*` constructor
  associated with each ident is what the check needs to inspect.
- `tensor/shape.ml` / `tensor/shape.mli` — `compose_type` definition and
  `Broadcast (Compose, ...)` shape-inference branch.
- `tensor/operation.ml` — `pointpow`, `pointdiv`, `pointmul`, `matmul`;
  shows that the user-facing OCaml operators already restrict themselves
  to pointwise/einsum and never reach `Compose` from this path.
- `arrayjit/lib/assignments.ml` — `loop_accum` etc.; consumes the
  projections produced by shape inference and applies the op blindly.

### Related

- Issue #305 (tracked in the existing ppx error message about ternary
  einsum notation) is unrelated but lives next to the code we touch.
- The existing `*` and `/` "no default compose" errors in
  `tensor/ppx_shared.ml` are the model for the new error wording.

## Approach (optional)

*Suggested approach — agents may deviate if they find a better path.*

In `tensor/ppx_cd.ml`, in the binary branch that maps `~logic:"@"` to
`Shape.Compose`, emit a compile-time error when the op is division or
power. The simplest implementation is a small deny-list match on the
`bin_op` *string* (before calling `binary_op`), since at ppx time we have
the source identifier directly and don't need to inspect the AST of the
resolved expression. Denied: `/`, `div`, `**`, `pow` (i.e. `Ir.Ops.Div`
and `Ir.Ops.ToPowOf`, per the `binary_ops` table in
`tensor/ppx_shared.ml:438-447`). Everything else with `~logic:"@"` stays
accepted. Error wording like:

> ppx_ocannl %cd: `~logic:"@"` (Compose) with `<op>` looks like matrix
> inverse/power but computes neither; use `~logic:"."` for pointwise
> `<op>`, or einsum notation for a custom contraction.

Apply the same check in both duplicated patterns (curried and tupled
forms) *(Update 2026-06-12: now a single shared body for three
or-patterns — one check suffices)*. Leave the einsum (`else`) branch
alone — users who know what they want via einsum can express it.

A defensive, lower-priority improvement: also raise from
`Shape.get_inequalities` if a `Broadcast (Compose, _, _)` is paired with
a non-`Mul` accumulator op at the assignments level. This catches code
paths that bypass the ppx (none currently exist in-tree, but it
future-proofs library users who construct shapes by hand). This is
optional and can be deferred.

## Scope

In scope:
- Compile-time guard in `tensor/ppx_cd.ml` for binary `~logic:"@"`.
- Tests covering the positive (multiply) and negative (div, pow, and at
  least one other non-multiply op) cases.
- A short note in `docs/syntax_extensions.md` clarifying that `~logic:"@"`
  is reserved for multiplication.
- Audit of in-tree uses of `~logic:"@"` (a single `grep` will do) to
  confirm no fix-ups are needed elsewhere.

Out of scope:
- Implementing matrix inverse / pseudo-inverse for `/` with `Compose`.
- Reworking the `compose_type` algebra or adding new shape logics.
- Touching the ternary `Compose_accumulate` path.
- Changing the public `pointdiv` / `pointpow` operators in
  `tensor/operation.ml` — they already correctly avoid `Compose`.

Dependencies: none. This is a self-contained ppx + tests change.

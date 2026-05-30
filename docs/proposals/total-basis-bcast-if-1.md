# Replace heuristic basis compatibility with a total basis (reserved `bcast_if_1` bottom)

## Goal

Today a dimension's `basis : string option = None` doubles as both the
*claim-free broadcastable unit* (`1_∅`, what scalars and rank-broadening
synthesize) and the *unspecified-but-real axis* (an axis of size > 1 the user
wrote without naming a basis). Conflating the two makes `None` behave as a
wildcard that matches any concrete basis of the same size. The consequence is
that the `⊑` relation computed in `Row.solve_dim_ineq` is **non-transitive**:
`3_rgb ⊑ 3_∅` and `3_∅ ⊑ 3_xyz` both hold, yet `3_rgb ⊑ 3_xyz` does not. Since
`propagate_shapes` runs more than once, solver order can change the verdict —
the same defect class as the row-side tripartite-split bug fixed in
`task-71e28eb1` (PR #458), one level down on dimensions. There is also a latent
propagation leak: the inequality comparison clause *checks* basis compatibility
but never *records* the named-basis side back onto the `None` side, so a later
comparison sees `None` again and a second conflicting basis can slip through.

This proposal makes `basis` **total** — every dimension carries a tag — and
represents the claim-free bottom as a reserved tag string `bcast_if_1` rather
than as `None`. `⊑` becomes a flat partial order with a single bottom and
every other dimension an atom; transitivity is then free. The change is a
**complete best-effort refactoring** (companion to `task-71e28eb1`), and it
includes a **frontend syntax change**: remove the char-literal basis-labelling
syntax (`'q' 2.0`) and add a syntax for labelling axes inside tensor literals.

This is a formalization-driven cleanup, not an issue-tracked bug. The
authoritative motivation and constraints live in the task's **§Design Brief**
(`task-4eb929b2`), which this proposal does not re-litigate.

## Acceptance Criteria

These are intent-level. Implementation/test specifics belong to the plan phase.
They mirror the six "technical issues that must be respected" in the brief.

1. **Basis is total.** `Row.solved_dim`'s `basis` field is `string` (not
   `string option`), per **Option A** — every constructor must supply a tag, so
   the compiler surfaces every minting site. No `None` reaches
   `Row.solve_dim_ineq` or `Row.unify_dim`. Verifiable: `grep -n 'basis = None\|basis : string option\|?basis'`
   over `tensor/row.ml`, `tensor/row.mli`, `tensor/shape.ml`, `tensor/tensor.ml`,
   `tensor/ppx_op.ml` returns no live `option`-typed basis (only historical
   comments, if any, may mention the old shape).

2. **`⊑` is a flat partial order with `1_(bcast_if_1)` as bottom.** The
   comparison clause in `Row.solve_dim_ineq` admits `d₁ ⊑ d₂` iff `d₁` and `d₂`
   are equal as dims (same size *and* same tag) or `d₁ = 1_(bcast_if_1)`. The
   `Option.map2 … ~default:true` wildcard branch is gone. Every dimension other
   than the bottom is incomparable above the bottom.

3. **Bottom and atom-unit are distinct and order oppositely.**
   `1_(bcast_if_1)` (bottom) satisfies `1_(bcast_if_1) ⊑ d` for every `d`;
   `1_default` (an atom) satisfies `1_default ⊑ d` only when `d = 1_default`.
   Tests pin **both** directions (broadcast accepts / pointwise rejects). They
   are not collapsed anywhere.

4. **Provenance decides the tag at construction.** Scalar / learning-rate /
   rank-broadening sites mint `bcast_if_1`; user-spec axes that the user left
   unnamed mint `default`. The split happens where the dimension is minted, not
   patched up later.

5. **Frontend `default` refuses to match named bases.** An unannotated user
   axis is `default` and is incompatible with any named basis: a spec carrying
   an implicit `3_default` and an explicit `3_rgb` no longer silently fuses —
   it surfaces as a shape error. (This is the intended stricter semantics, not a
   regression.)

6. **Frontend syntax change.** The char-literal basis-labelling syntax
   (`'q' 2.0`) is removed, and a new syntax for labelling axes inside tensor
   literals is added (surface form chosen during the plan/duo phase — see
   *Approach* below). Every example and test that used the old syntax
   (`test/ppx/test_ppx_op.ml`, `test/operations/hello_world_dim1x1.ml`,
   `test/operations/hello_world_op.ml`, and their `.expected` companions) is
   migrated to the new syntax and still compiles/runs.

7. **The `bcast_if_1` tag is advertisable.** A user who writes the
   `bcast_if_1` tag deliberately gets the documented behavior: the axis
   broadcasts when its size is 1 and is an ordinary fixed atom when its size is
   > 1 (`5_(bcast_if_1)` is a legal, harmless atom — not an error). A test pins
   both cases.

8. **Inequality propagation is stated explicitly.** A code comment in
   `solve_dim_ineq` and a test state what the inequality path now records about
   basis, closing the leak that was latent under `None`. (Under totality there
   is no `None`-on-a-real-axis to propagate on the inequality path; the comment
   makes that explicit by construction rather than by luck.)

9. **LUB-merge demotion preserved.** The conflicting-basis (or conflicting-size)
   demotion in `solve_dim_ineq`'s LUB-maintenance clause and in the row-level
   `meet_dim` still demotes to a broadcast unit `1` — now minted as
   `1_(bcast_if_1)` — as intentional broadcast generalization, the same as today
   minus the `None`-bridge cases. This must NOT be tightened to raise
   `Shape_error`.

10. **Transitivity regression test.** A test pins that any chain `a ⊑ b ⊑ c` is
    rejected unless `a ⊑ c` holds directly: the previously-accepted
    `3_rgb ⊑ 3_default ⊑ 3_xyz` (or analogous) chain now rejects.

11. **Suite green; expected churn documented.** The existing test suite passes.
    Any test that depended on the old wildcard behavior is updated, and the PR
    description notes that newly-failing wildcard-dependent programs are an
    expected stricter-semantics change, not a regression. `.expected` churn from
    the reserved tag now printing where `None` printed blank (e.g. `row_to_bases`
    returns the tag string instead of `""`, and `dim_to_string` no longer has a
    `None` arm) is called out as such.

### AC verification reachability

All AC-named paths (`tensor/row.ml`, `tensor/shape.ml`, `tensor/tensor.ml`,
`tensor/ppx_op.ml`, `test/**`) live inside `git -C /Users/lukstafi/ocannl-staging`'s
introspection reach, so their evidence is ordinary in-repo `grep`/build/test
runs plus the merge commit SHA. No path here sits outside the project's git
context, so the find/grep-over-SHA caveat does not apply.

## Context

How the basis machinery works today, by symbol (not line number — these drift):

- **Type.** `Row.solved_dim = { d : int; basis : string option; proj_id : … }`
  in `tensor/row.ml` (mirrored in `tensor/row.mli`). `dim` wraps it as
  `Dim of solved_dim`. `Row.get_dim ~d ?basis ?proj_id ()` is the sole
  smart-constructor; callers that omit `?basis` mint `None`.

- **Comparison clause (the bug).** In `Row.solve_dim_ineq` (the `match (cur, subr)`
  big match), the two relevant arms are:
  ```ocaml
  | Dim { basis = Some b1; _ }, Dim { basis = Some b2; _ } when not (String.equal b1 b2) ->
      raise @@ Shape_error ("dimension comparison for axis: different bases", …)
  | Dim { d = d1; basis = b1; _ }, Dim { d = d2; basis = b2; _ }
    when d1 = d2 && Option.value ~default:true (Option.map2 ~f:String.equal b1 b2) ->
      ([], env)
  | _, Dim { d = 1; basis = None; _ } -> ([], env)
  | (Dim { d = 1; basis = None; _ } as cur), _ -> ([ Dim_eq { … } ], env)
  ```
  The `Option.value ~default:true (Option.map2 …)` is the wildcard; the two
  `{ d = 1; basis = None }` arms are the bottom rule (currently keyed on
  `None`, i.e. the overloaded bottom). The clause returns `([], env)` — it
  checks but records nothing (the propagation leak).

- **Equality clause (`Row.unify_dim`).** Same `Option.value ~default:true …`
  wildcard guard, but it *does* propagate: when one side is `None` and the other
  carries a basis, it walks `upgrade_var` to rewrite the variable that resolved
  to the unbased dim so the basis persists. Under totality there is no `None`
  side to upgrade; the brief asks this asymmetry be stated explicitly after the
  change.

- **LUB-merge demotion (`Row.solve_dim_ineq` lub maintenance).** The
  `Bounds_dim … lub = Some lub2` arm: matching `(cur, lub2)` it either keeps the
  more-specified basis or, on conflicting size/basis, mints
  `get_dim ~d:1 ~proj_id:47 ()` and forces `Dim_eq`. The comment there
  ("Intentional broadcast semantics … NOT a bug") must survive. The minted
  `d=1` currently has `None` basis; it must become `bcast_if_1`.

- **Row-level `meet_dim` (in `unify_row`'s `Bounds_row` lub maintenance).**
  Several `get_dim ~d:1 ~proj_id:{48,49,50,51,52,63} ()` synthesizers plus arms
  that case on `{ d = 1; basis = None }` vs `{ d = 1; basis = Some _ }`. These
  encode the "unbased d=1 is most general, conflicting bases demote to d=1"
  policy and must be ported to the `bcast_if_1` bottom.

- **Bottom synthesizers (rank-broadening fill et al.).** `get_dim ~d:1 …` is
  minted in many places in `row.ml` (the `proj_id:42–62` family) to pad
  ranks / resolve free variables to the broadcast unit. The brief's "verify this
  is the only synthesizer of the bottom" should be read as: *every* site that
  synthesizes the broadcast unit `1` must mint `bcast_if_1`, and the plan must
  enumerate them (the `proj_id`-tagged `get_dim ~d:1` family is the index). This
  is the single most error-prone part of the refactor — a missed site re-injects
  a tag-less unit.

- **Scalar / helper constructors.** `Tensor.number` and `Tensor.bits` in
  `tensor/tensor.ml` take `?axis_basis` and, when absent, emit `output_dims:[1]`
  → `get_dim ~d:1 ()` (currently `None`). These are the canonical scalar /
  learning-rate sites and must mint `bcast_if_1`.

- **Frontend named bases.**
  - Einsum / shape-spec parsing: `Shape.shape_spec_to_dims_bio` maps an
    unannotated `Label`/`Fixed_index` to `Row.get_dim ~d ()` (→ `None` today →
    `default` after). `Shape.make`'s `make_axes` maps `(basis, d)` spec tuples to
    `get_dim ~d ~basis ()`; `make`'s `f` synthesizes a `debug_name ^ "_output"`
    basis for a lone size-1 output axis.
  - Char-literal basis syntax (the removal target): `tensor/ppx_op.ml` has three
    arms matching `Pconst_char ch` followed by a float / int / int64 literal,
    each lowering to `TDSL.number`/`TDSL.bits … ~axis_basis:<char-as-string>`.
    These are exactly the `'q' 2.0` / `'p' 1.0` forms. Used in
    `test/ppx/test_ppx_op.ml`, `test/operations/hello_world_dim1x1.ml`,
    `test/operations/hello_world_op.ml`.
  - Tensor-literal axis labelling (the addition target): tensor literals are
    parsed in `tensor/ppx_shared.ml` (`Pexp_array`/list/tuple backbone for
    batch/output/input axes) and `tensor/ppx_op.ml`'s `ndarray_op`. There is
    currently no per-axis label syntax there — this is net-new surface.

- **Printing / sexp (the `.expected` churn).** `solved_dim` derives
  `sexp`, so the `basis` field's shape change rewrites every auto-derived sexp
  that prints a dim (the hidden data-shape consumer learning from PR #458).
  `Row.dim_to_string` / `solved_dim_to_string` have explicit `None`/`Some`
  arms (and a `Only_bases` style that prints `_` for `None`); `Row.row_to_bases`
  maps `None → ""`. After totality, the bottom prints as `bcast_if_1` and
  unannotated axes as `default` where they printed blank/`_` before. Expect
  `.expected` churn comparable in shape to PR #458 across
  `test/operations/*.expected` and `test/einsum/*.expected`.

- **Tests.** `test/einsum/test_dimension_labels.{ml,expected}` is the primary
  basis-behavior suite (helpers `based_tensor` / `unbased_tensor`,
  `get_var_basis`, `row_to_bases`); `test/einsum/test_print_style.*` exercises
  the print styles. New transitivity / bottom-asymmetry / advertisable-affordance
  tests belong alongside these.

Explicitly **out of scope** (rejected in the brief — do not introduce):
- **Basis variables** (a second unifiable sort; `None` as a metavariable).
- **A `Dim_bottom` constructor** (distinct variant for the bottom).

## Approach

*Suggested approach — agents may deviate if they find a better path.*

The type-level and solver-level work is **straightforward and pinned** (Option A
totality; replace the wildcard guard with the flat-order guard; key the bottom
on the `bcast_if_1` tag; port every `get_dim ~d:1` synthesizer to mint
`bcast_if_1`; thread `default` through the unannotated frontend mints; reserve
the two tag literals). The compiler will drive most of it once `basis` becomes
`string` — make the field non-optional first and follow the type errors.

The **new axis-labelling syntax for tensor literals is a creative choice and is
deliberately NOT pinned here** — route it to **duo mode**. The brief lists
identifiers, string literals, and an OCaml-type-annotation form (`(2.0 : rgb)`)
as candidates, with char literals ruled out (they can't carry multi-character
tags like `rgb`). Two independent implementations should compete on the surface
form; the constraints they must satisfy are: multi-character tags work; the form
composes with the existing array/list/tuple tensor-literal backbone in
`ppx_shared.ml`; and the three removed char-literal arms in `ppx_op.ml` have a
clean replacement so the migrated examples read naturally.

The plan must also discharge brief **§Technical issue 3**: confirm that nothing
relies *implicitly* on an explicit size-1 axis stretching (broadcasting is
rank-based at the marker; a user `1_default` meeting a larger size is a genuine
pointwise mismatch). If any site does rely on that, surface it — it is in tension
with this design.

Per **§Technical issue 4** (and Resolved Q4): no upfront corpus sweep for
`default`-incompatibility breakage. Fix breakage as it surfaces during
implementation and note in the PR that newly-failing wildcard-dependent programs
are expected stricter semantics, not regressions.

## Scope

In scope: the `dim`/`solved_dim` type and `get_dim`; both comparison clauses
(`unify_dim` equality, `solve_dim_ineq` inequality) and both LUB/meet
maintenance clauses; every broadcast-unit synthesizer in `row.ml`; scalar/helper
constructors (`Tensor.number`, `Tensor.bits`); frontend mints of unannotated
axes (`Shape.shape_spec_to_dims_bio`, `Shape.make`); reserving the `bcast_if_1`
and `default` tag literals; removal of the char-literal basis syntax in
`ppx_op.ml`; addition of tensor-literal axis labelling; printing/sexp updates and
the resulting `.expected` churn; and the new tests (transitivity, bottom
asymmetry, frontend strictness, advertisable affordance, syntax migration).

Out of scope: basis variables; a `Dim_bottom` constructor; any change to the
rank-based broadcasting model itself beyond tagging the synthesized unit.

Dependencies: companion to `task-71e28eb1` (PR #458, merged) — that PR's commits
are the precedent for moving a field across constructor arms and for the
auto-derived-sexp `.expected` churn pattern; no hard ordering dependency remains
since #458 is on `master`.

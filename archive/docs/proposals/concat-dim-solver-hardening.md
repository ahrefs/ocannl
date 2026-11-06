# Concat dim-solver hardening: broadcast GLB for Concat + generalize `unify_dim` `Concat = Concat`

## Goal

Two related correctness gaps in the dimension solver (`tensor/row.ml`),
surfaced by task-0088ec20 (nested cross-kind tensor stacking, merged as
ocannl-staging PR #64):

1. **A reachable `assert false` crash.** In `solve_dim_ineq`'s broadcast
   greatest-lower-bound (GLB) merge, the `Affine` and `Concat` arms are
   `assert false`. A pointwise op (e.g. `add`) between two concat-axis
   (stacked / `++^`-concatenated) tensors produces a *broadcast inequality*
   between two `Concat` bounds that routes through this GLB merge and crashes,
   rather than producing a correct broadcast or a shape error. This is a
   pre-existing latent bug (present on `master`, not introduced by PR #64); the
   shipped stacking feature exercises only the `unify_dim` *equality* path and
   never hits this arm, but it is a real reachable crash for a plausible user
   program.

2. **An over-narrow `unify_dim` `Concat = Concat` guard.** PR #64's Round-4 fix
   (commit `0d13ed77`, rebased `e02d26f4`) narrowed the variable-pairing arm of
   `unify_dim`'s `Concat xs, Concat ys` branch so it fires only when *every*
   residual component on *both* sides is a bare unsolved `Var` *and* the two
   sides have *equal arity*. That arity dependence is too restrictive: nested
   stacking shapes whose two sides differ in component count (but where one side
   is variables-only) are left linked only by deferral, which fails to make
   progress and hangs the solver. The guard should be generalized so the branch
   stays load-bearing for nested stacking at any arity.

The user resolved the design fully on 2026-06-21 (see the task file's
"Questions — Resolved" section). This proposal distills that resolved intent
into verifiable acceptance criteria; the design is not re-opened.

## Acceptance Criteria

### AC1 — `solve_dim_ineq` GLB merge: replace `assert false` with equality-attempt-first

In `solve_dim_ineq` (`tensor/row.ml`), inside the `_, Var opnd_v` case's
existing-bound branch (`Bounds_dim { glb = Some glb2; ... }`), the
`let glb, glb_forcing = match (res, glb2) with ...` computation currently has:

```
| Var _, _ | _, Var _ -> assert false
| Affine _, _ | _, Affine _ -> assert false
| Concat _, _ | _, Concat _ -> assert false
```

The `Affine` and `Concat` arms must be replaced by the resolved
**equality-attempt-first** recipe: attempt to unify the two bounds (`res`
against `glb2`) via `unify_dim`, and:

- **If `unify_dim` fails** → demote to broadcast-top, exactly as the existing
  `Dim _, Dim _` (different size / basis) arm two arms above does:
  `let glb = get_bcast_dim ~d:1 ~proj_id:47 () in (glb, [ Dim_eq { d1 = opnd; d2 = glb; origin } ])`.
- **If `unify_dim` succeeds**:
  - **below stage 4** (`not (is_stage4_up stage)`) → **postpone**: keep /
    re-defer the processed inequality so the solver revisits once more is
    known, using the same deferral idiom as the existing `Dim_ineq { res; opnd;
    from_ = Sexp.List []; origin }` re-emission near the `discardable_vars`
    arm of `unify_dim`. The two bounds must remain un-demoted so they can still
    resolve equal-as-sizes later.
  - **stage 4 or up** (`is_stage4_up stage`) → **return the unification
    result**: set the merged GLB to `glb2` (i.e. `res = glb2`) combined with the
    existing GLB-forcing `res = opnd` equation (`Dim_eq { d1 = opnd; d2 = glb;
    origin }`). The constraints produced by the successful `unify_dim` are
    threaded through alongside.

- **Verifiable behaviour:** a pointwise add of two stacked (`++^`) tensors whose
  new axes are `Concat` bounds **no longer crashes**. A regression fixture
  exercising the pointwise-add-of-two-stacked-tensors path is added and passes.

  **Item-1 reachability is to be verified honestly.** The fixture author must
  trace which constraint kind the surface program actually emits — a broadcast
  *inequality* (`Dim_ineq`) between two `Concat` bounds reaching
  `solve_dim_ineq`'s GLB merge, **not** an equality reaching `unify_dim`. If no
  determinate `%op`/DSL fixture is found that demonstrably routes through the
  GLB merge with two `Concat` bounds, this must be **flagged explicitly in the
  implementation notes** rather than asserted; in that case the regression may
  fall back to a direct solver-level test (constructing the `Dim_ineq` and
  driving `solve_dim_ineq` at stage ≥ 4) that proves the arm no longer raises.

### AC2 — `solve_dim_ineq` equal-length concat arm: emit a single `Concat = Concat` equation

The arm in `solve_dim_ineq`'s top-level `match (res, opnd)`:

```
| Concat dims1, Concat dims2 when List.length dims1 = List.length dims2 ->
    (* Element-wise unification of concatenated dimensions *)
    let eqs = List.map2_exn dims1 dims2 ~f:(fun d1 d2 -> Dim_eq { d1; d2; origin }) in
    (eqs, env)
```

must stop forcing **pointwise** element equalities and instead emit a **single**
`Concat dims1 = Concat dims2` equation:
`([ Dim_eq { d1 = Concat dims1; d2 = Concat dims2; origin } ], env)`.

This routes the two concats through `unify_dim`'s `Concat = Concat` branch
(which performs structural cancellation and the generalized variable pairing of
AC3) rather than over-constraining positionally.

### AC3 — `unify_dim` `Concat = Concat` branch: generalize the variable-pairing guard

In `unify_dim`'s `Concat xs, Concat ys ->` branch (`tensor/row.ml`), the
variable-pairing arm narrowed by PR #64:

```
| _, _
  when List.is_empty solved_x && List.is_empty solved_y
       && List.length rest_x = List.length rest_y
       && List.for_all (rest_x @ rest_y) ~f:(function Var _ -> true | _ -> false) ->
    List.fold (List.zip_exn rest_x rest_y) ...
```

must be replaced by a generalized arm with the resolved guard:

- **fire when** the solver is at **stage 4 or later** (`is_stage4_up stage`)
  **and one side's residual components are all `Var`** (variables-only on at
  least one side). The **same-arity restriction is dropped entirely**; the
  **`solved_x`/`solved_y` empty restriction is dropped** — the guard is "one
  side all-`Var`", *not* "overlap all-`Var`", so this arm fires and **takes
  precedence even when the other side carries a solved `Dim`**.
- **takes precedence over** the trailing `| _, _ ->` arithmetic-cancellation
  arm: when one side is variables-only this arm must fire and **must not fall
  through** to cancellation (that arm has nothing to do with a variables-only
  side). Concats where **neither** side is all-`Var` continue through the
  arithmetic-cancellation arm unchanged.
- **action when matched:** equate the **selected oldest variables** of the two
  sides (oldest = lowest `Dim_var.id`, the existing variable-age ordering;
  `compare_dim_var` orders purely by `id`). This yields a new env /
  substitution; then **re-run `unify_dim` on the two original concats under
  that new env**. There is no leftover-variable bookkeeping, no manual
  reduced-sides `Concat = Concat` construction, and no empty-concat / size-0
  special case — the re-run (with `cancel_common_dims` now seeing the freshly
  equated components as common) handles unequal arity and any further progress.

- **Verifiable behaviour:** all existing nested-stacking fixtures
  (task-0088ec20 Test 5 `[[x1;x2];[x1;x2]]` et al. in
  `test/operations/test_block_tensor.ml`) still resolve; the full suite stays
  green; and a **new fixture covering an unequal-arity, variables-only
  `Concat = Concat`** is added (the case the old arity guard rejected).

### AC4 — Tests and build green

- New / updated fixtures live in
  `test/operations/test_block_tensor.{ml,expected}`.
- Full test run is green: `OCANNL_BACKEND=sync_cc dune runtest`.
- Type-checks: `dune build @check`.
- All pre-existing nested-stacking fixtures pass unchanged.

### AC verification reachability

All target paths live inside `git -C ~/ocannl-staging` (a normal worktree on
`origin/master`), so SHA-/grep-based verification against the project tree is
directly reachable. Use symbol pointers (function and arm names) rather than
line numbers — `tensor/row.ml` drifts as other PRs merge.

## Context

All work is in **`tensor/row.ml`** (the dimension/row constraint solver) plus
the test fixtures. The two scope items are genuinely orthogonal code paths that
share only the `Concat` constructor:

**Scope item 1 (AC1, AC2) — `solve_dim_ineq` (broadcast GLB / lower-bound
merge).** `solve_dim_ineq ~stage origin ~res ~opnd env` enforces `res ⊑ opnd`.
The crash arm lives in its `_, Var opnd_v` case, existing-bound branch
(`Bounds_dim { glb = Some glb2; ... }`), inside
`let glb, glb_forcing = match (res, glb2) with ...`. The `Dim _, Dim _`
(different) arm immediately above already demonstrates the demote idiom
(`get_bcast_dim ~d:1 ~proj_id:47 ()` plus `Dim_eq { d1 = opnd; d2 = glb }`).
Crucially `solve_dim_ineq` receives `~stage` and `unify_dim` is in module scope,
so both the `is_stage4_up stage` predicate and a recursive equality attempt are
directly available. The equal-length concat arm (AC2) is the
`Concat dims1, Concat dims2 when List.length dims1 = List.length dims2 ->` arm
in `solve_dim_ineq`'s top-level `match (res, opnd)`. The postpone/deferral
idiom to mirror is the `Dim_ineq { res; opnd; from_ = Sexp.List []; origin }`
re-emission used for the `discardable_vars` concat case.

**Scope item 2 (AC3) — `unify_dim` `Concat = Concat` (strict equality).**
`unify_dim ~stage origin (eq : dim * dim) env` is the recursive equality
solver. Its `Concat xs, Concat ys ->` branch first runs
`cancel_common_dims xs ys` (structural cancellation; two distinct unsolved
`Var`s never cancel, so they survive into the residual), then
`partition_solved_dims` into `solved_*` / `rest_*`, and matches on
`(rest_x, rest_y)`. The variable-pairing arm to generalize is the
`when List.is_empty solved_x && ... List.for_all ... Var ->` arm; the trailing
`| _, _ ->` arm is the arithmetic-cancellation remainder (cancels
`Int.min sum_x sum_y`, defers one `Dim_eq` between the with-residual sides).

**Shared helpers / predicates:**
- `is_stage4_up = function Stage1 | Stage2 | Stage3 -> false | _ -> true` —
  the exact "stage 4 or later" predicate, already used to gate GLB closure
  elsewhere in this file.
- `Dim_var.t.id` / `compare_dim_var` — variable-age ordering. Ids come from a
  single monotonic counter, never reused: **lower `id` = older**. Selecting
  oldest-against-oldest keeps unification deterministic across fixpoint
  re-entries.
- `cancel_common_dims`, `partition_solved_dims`, `reduce_solved_basis`,
  `merge_derived_basis`, `get_bcast_dim`, `get_dim`.
- Stage driver: `tensor/shape.ml` (`finish_inference` / `derive_projections`)
  invokes `Row.solve_inequalities ~stage:Stage1 … Stage4 … Stage7` in strict
  ascending order, threading `unsolved` / `env` forward.

**Fixtures:** `test/operations/test_block_tensor.{ml,expected}`. Test 5
`[[x1;x2];[x1;x2]]` is present and passing on `origin/master` (restored by
PR #64); the "Test 5 hangs" note in the retrospective describes the *historical*
pre-fix state. The unequal-arity variables-only `Concat = Concat` case is not
yet covered.

**Provenance:** target code is on `origin/master @ da787d44` (the PR #64 merge).

## Approach

*Suggested approach — agents may deviate if they find a better path.* The
design is fully resolved by the user; the ACs above already specify the exact
arms, guards, and actions, so this section restates the mechanical shape:

- **AC1:** factor a small helper (or inline) that calls
  `unify_dim ~stage origin (res, glb2) env` guarded by a `try`/result on
  `Shape_error`; branch on success/failure and on `is_stage4_up stage` per the
  recipe. Thread the resulting constraints through `glb_forcing`. Apply the same
  recipe to both the `Affine` and the `Concat` arms (the `Var _` arm stays
  `assert false` — unsolved vars should not appear as a resolved `res`/`glb2`
  here).
- **AC2:** one-line change — replace the `List.map2_exn` pointwise expansion
  with a single `Dim_eq { d1 = Concat dims1; d2 = Concat dims2; origin }`.
- **AC3:** relax the `when` guard (add `is_stage4_up stage &&`, drop the
  arity-equality and `solved_*`-empty conjuncts, relax `List.for_all` to "one
  side all-`Var`"); replace the positional `List.zip_exn` fold body with:
  select the oldest `Var` of each side, emit a `Dim_eq` equating them, extend
  the env, and `unify_dim ~stage origin (Concat xs, Concat ys) env` again under
  the new env. Termination relies on each re-entry strictly shrinking the
  residual via `cancel_common_dims`.

## Scope

**In scope:** `tensor/row.ml` (`solve_dim_ineq` AC1/AC2 arms, `unify_dim`
`Concat = Concat` arm AC3) and `test/operations/test_block_tensor.{ml,expected}`
(regression for AC1, fixture for AC3). The change is local to one module plus
its test.

**Out of scope:** the `Var _` GLB arm (stays `assert false`); broader solver
refactors; any change to the staging mechanism or stage driver; the
`Concat`-vs-non-concat deferral arms beyond what AC1/AC2 touch.

**Dependencies:** follow-up from task-0088ec20 (PR #64, merged). No blocking
dependencies.

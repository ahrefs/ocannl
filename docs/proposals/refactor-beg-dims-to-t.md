# Refactor: Move `beg_dims` from `Row_var` to `t`; fix leading-flank alignment in `solve_row_ineq`

**Target:** OCANNL v0.7.x (pre-workshop-submission, deadline June 24)

## Why

OCANNL's row representation puts `beg_dims` inside the `Row_var` arm of `bcast`:

```ocaml
type bcast =
  | Row_var of { v : row_var; beg_dims : dim list }
  | Broadcastable
type t = { dims : dim list; bcast : bcast; prov : provenance }
```

This means only open rows can carry pinned leading axes. Closed rows have no representation for "the leading edge is anchored at these axes," which interacts badly with two things:

1. **Substitution silently loses leading axes.** `s_row_one` plugging a closed value into an open row with non-empty `beg_dims` returns `{ dims = beg_dims @ more_dims @ dims; bcast = Broadcastable }`, collapsing the leading flank into the unmarked `dims`. The structural fact that those axes were leading-pinned is gone.

2. **`solve_row_ineq`'s leading-flank alignment anchors at the hole, not the outer edge.** The current per-axis emission uses `take_from_end` on both flanks:

   ```ocaml
   zip Dim_ineq (take_from_end cur_beg_dims beg_dims_l)
                (take_from_end subr_beg_dims beg_dims_l)
   ```

   This routes outer-leading excess into the LUB-vs-drop ambiguity that the LUB cannot faithfully represent. The LUB residue uses `List.drop` (drop from front) on `cur_beg_dims` — opposite end from the per-axis match — and the combination is unsound for `0 < |subr.beg_dims| < |cur.beg_dims|`. A minimal trace: `cur = [a₀,a₁]·⟨σ⟩·[b₀]`, `subr = [c₀]·⟨ρ⟩·[d₀]` matches `a₁ ↔ c₀` (hole-adjacent) and banks `lub(ρ) = [a₁]·⟨σ⟩`, duplicating `a₁` into ρ's value at close while never checking `c₀` against the actually-aligned `a₀`.

**The fix:** every row carries `beg_dims` directly (closed and open alike); broadcasting always happens in the middle (between two outer-anchored flanks); leading-flank alignment is outer-left throughout; the LUB on a row variable is itself a row carrying the inner residue.

## Design

```ocaml
type bcast =
  | Row_var of row_var
  | Broadcastable
type t = { beg_dims : dim list; dims : dim list; bcast : bcast; prov : provenance }
```

- **Open row:** `{ beg_dims = l; dims = r; bcast = Row_var v }` — pinned leading flank `l`, pinned trailing flank `r`, row variable `v` absorbing the gap between them.
- **Closed row:** `{ beg_dims = l; dims = r; bcast = Broadcastable }` — pinned leading flank `l`, pinned trailing flank `r`, no rank-broadening slack. Smaller-rank rows can still align inside via per-axis broadcasting (dim-1 broadcasts to any size), but no axes are inserted between the flanks.

Both flanks anchor at the outer edges of the row. The previous notion "leading-pinned vs broadcastable" is just "is `beg_dims` non-empty?" — a structural read, no mode bit, no separate lattice.

## Changes

### `tensor/row.mli` and `tensor/row.ml`

**Type definitions.** Move `beg_dims` from inside `Row_var` to the top of `t`. Audit and update all constructor helpers (`get_row_for_var`, `row_of_var`, and the literal layouts in `unify_row` / `solve_row_ineq`).

**`s_row_one`** (substitution). Replace the case split with a uniform composition:

```ocaml
let s_row_one v ~value ~in_ =
  match in_ with
  | { beg_dims = l1; dims = r1; bcast = Row_var v'; prov } when equal_row_var v v' ->
      { beg_dims = l1 @ value.beg_dims;
        dims     = value.dims @ r1;
        bcast    = value.bcast;
        prov }
  | _ -> in_
```

The closed-value branch is the soundness fix: `beg_dims` is faithfully composed regardless of whether `value` is closed or open. The result's `beg_dims = l1 @ value.beg_dims` is non-empty whenever `l1` was; the type system now refuses the previously-broken "collapse to unmarked closed" branch.

**`solve_row_ineq`** (broadcast inequality). Two fixes:

- *Leading-flank alignment.* Replace `take_from_end` with prefix-take on the leading flank. Both flanks now use the outer-anchor convention:

  ```ocaml
  zip Dim_ineq (List.take cur.beg_dims beg_dims_l)
               (List.take subr.beg_dims beg_dims_l)
  @ zip Dim_ineq (take_from_end cur.dims dims_l)
                 (take_from_end subr.dims dims_l)
  ```

- *LUB residue.* `r_cur` is the inner residue of `cur` after `subr`'s flanks have been matched off at the outer edges:

  ```ocaml
  let r_cur = {
    beg_dims = List.drop cur.beg_dims beg_dims_l;      (* drop matched outer prefix *)
    dims     = drop_from_end cur.dims dims_l;          (* drop matched outer suffix *)
    bcast    = cur.bcast;
    prov     = cur.prov;
  }
  ```

  Both edges drop the *matched outer* portion; the inner residue is what ρ broadcasts past or absorbs.

**LUB merge.** Make the merge symmetric across both flanks. The current right-flank dimensionwise meet (line ~2848 of `row.ml`) handles conflicts by demoting to broadcast-1; mirror this on the leading flank, **left-aligned within `beg_dims`** (since both flanks anchor outer). The merged LUB has the shorter of the two leading-flank lengths and the shorter of the two trailing-flank lengths ("prefer generality"). The existing comment near line 2846 — *"we lose connection here if both have row variables"* — becomes obsolete; the leading flank now merges correctly.

**`unify_row`.** The structural cases need their type-shape updated (literal layout under the new `t`). The alignment in `unify_row` is already outer-anchored on both flanks: it uses `List.rev … take_from_end` for leading-flank alignment, which is equivalent to outer-left prefix-take. Sanity-check this during the refactor; any branch that happens to anchor at the hole should be mirrored to the `solve_row_ineq` fix.

**Closing rules** (in `finish_inference` and related). Audit: when an open row closes — to LUB at terminals (Stage 3), or to no-further-axes at non-terminals (Stage 6) — `beg_dims` must be preserved. Closing changes `bcast` (`Row_var v` → `Broadcastable`) but must not silently drop the leading flank. A leading-pinned open row closes to a leading-pinned closed row; an unanchored open row closes to a plain broadcastable closed row.

### `tensor/shape.ml`

Pattern matches on `bcast` and row constructions are widespread; the type system will surface every site. Pay particular attention to:

- `einsum_slot_spec_to_dims_bio` — sites that build a row with leading axes from spec parsing now construct them directly into `t.beg_dims`.
- `get_inequalities` — row constructions used as inequality endpoints.
- Any destructuring of `Row_var { v; beg_dims }`.

### Einsum parser (`tensor/einsum_parse.ml` or wherever spec → row happens)

Wherever the parser constructs a row with leading axes, ensure the new layout is `{ beg_dims = …; bcast = Row_var v; … }` rather than `{ bcast = Row_var { v; beg_dims = … }; … }`.

## Tests

Add the following, in roughly increasing order of integration:

1. **Leading-flank alignment (the original bug).** With
   `cur = { beg_dims = [Dim 5; Dim 2]; dims = [Dim 4]; bcast = Broadcastable }`
   and
   `subr = { beg_dims = [Dim 2]; dims = [Dim 4]; bcast = Row_var ρ }`,
   the inequality `cur ≥ subr` should **fail**: outer-left alignment matches `subr`'s leading `Dim 2` against `cur`'s leading `Dim 5`, which is incompatible (size 2 cannot broadcast to size 5). Under the current code this incorrectly succeeds (the leading 2 aligns hole-adjacently with cur's second leading axis, also of size 2).

2. **Substitution preserves `beg_dims`.** With
   `in_ = { beg_dims = [Dim 3]; dims = []; bcast = Row_var ρ }`
   and
   `value = { beg_dims = []; dims = [Dim 4]; bcast = Broadcastable }`,
   `s_row_one ρ ~value ~in_` must yield
   `{ beg_dims = [Dim 3]; dims = [Dim 4]; bcast = Broadcastable }`,
   not `{ dims = [Dim 3; Dim 4]; bcast = Broadcastable }` (the old bug).

3. **Closing preserves `beg_dims`.** A leading-pinned open row reaching Stage 6 closes to a leading-pinned closed row: `bcast` flips to `Broadcastable`, `beg_dims` is unchanged.

4. **LUB merge symmetric.** Two upper bounds on a row variable, both with non-empty leading flanks, merge dimensionwise on both ends. Conflicts on either flank demote to broadcast-1. The merged LUB has the shorter of the two on each side.

5. **Monotonicity via re-firing.** Solve `b ⊑ c` with
   `b = { beg_dims = []; dims = []; bcast = Row_var β }`
   and
   `c = { beg_dims = [Dim 4]; dims = [Dim 7]; bcast = Broadcastable }`,
   recording `c` as `β`'s LUB. Subsequently substitute
   `β := { beg_dims = [Dim 4]; dims = []; bcast = Row_var β' }`.
   The leading `Dim 4` of the substituted `β` aligns outer-left against the leading `Dim 4` of the LUB and the inequality holds. No retraction of previously banked facts; new per-axis inequalities are added, never removed.

6. **Regression.** Run the existing OCANNL test suite. All passing tests should continue to pass; any test relying on the broken alignment (passing under the bug but inconsistent with the intended semantics) should be updated and noted.

## Out of scope

The following were considered as alternative designs during the design discussion and **explicitly rejected** in favor of this structural refactor:

- **Mode variables on row terms.** A separate two-element lattice (`Open ⊑ Fixed`) for "leading edge committed vs open." Not needed once `beg_dims` is on `t`: the structural fact is carried by the row, no lattice required.
- **`max_seen_delta` or any banked-scalar witness machinery.** Was intended to recover monotonicity under mode propagation. Unnecessary under structural `beg_dims`: the LUB itself carries the witness, and substitution rewrites it via the existing `s_row_one_in_entry` plumbing.
- **Restructuring stages 1–7 of the solver.** The stage discipline is unchanged. The closing rules are audited (above) but not redesigned.

If implementation surfaces a corner that seems to need one of these, **pause and discuss** before committing. The design intent is structural and first-order; any departure should be confirmed.

## Risks and notes

- **Refactor surface.** `bcast` patterns are widespread; the type system will guide most rewrites, but printers, sexp serializers, and hash functions may match by structure and need manual attention.
- **Closing-time silent drop is the main soundness risk.** A closing site that resets `beg_dims = []` would silently fix-by-erasure things that should error. Cover with explicit tests for case (3).
- **LUB merge interaction with existing tests.** Tests that happen to pass under the lossy left-flank merge may need updated expectations. Distinguish "expected behavior change" from "regression" carefully; flag any non-obvious cases for review.
- **Provenance combination.** Ensure substitution and merges combine `prov` correctly when both sides have non-trivial leading flanks — this may have been handled only on the trailing side before.
- **Symmetry as a check.** A useful internal invariant: `solve_row_ineq` and `unify_row` should use the *same* alignment convention on both flanks (outer-anchor). If they ever disagree, that's the bug shape we just fixed reasserting itself.

## Verification

After implementation:

- The original bug case (test 1) fails with an explicit shape error rather than silently succeeding.
- All tests 2–5 pass.
- The regression suite passes; expected behavior changes are documented.
- A sanity sweep confirms that `solve_row_ineq` and `unify_row` agree on alignment, and that `s_row_one`'s closed-value branch faithfully composes `beg_dims`.

# Proposal: Cycle detection (occurs check) for row variables

**Issue:** [ahrefs/ocannl#247](https://github.com/ahrefs/ocannl/issues/247)
**Milestone:** v1.0
**Status:** Implemented (June 2026) — see status update below

## Status update (2026-06-12)

- GH issue #247 is still OPEN with milestone v1.0, but the work has **landed in
  this repo** via PR #30 (`rank-cycle-row-vars`): commits `c66082c5` (detect
  rank cycles among row variables, Def. 4.5 rank-cycle check), `dbf67fcb`
  (reject self-referential row equations with shifted flanks), `3f80d706`
  (rotational row constraints — deferral into closing). The issue can likely
  be closed once this reaches upstream.
- The landed design **differs from this proposal's approach**: instead of a
  post-substitution syntactic `row_var_occurs_in_row` occurs check at binding
  sites, the implementation maintains a persistent rank-relation graph
  (`global_rank_edges` with `closes_positive_rank_cycle` / `add_rank_edge`,
  `tensor/row.ml` ~2020–2055). Each entailed fact `rank v >= rank w + k` is
  recorded as an edge; an edge closing a positive-total-weight cycle raises
  `Shape_error ("Infinite number of axes by rank cycle among row variables", ...)`.
  This catches transitive (indirect) cycles — AC2 — which the in-flight
  constraint reduction would otherwise diverge on by minting ever-fresh
  template variables.
- Tests landed as `test/einsum/test_row_rank_cycle.ml` + `.expected`, covering
  two- and three-variable rank cycles via `Row.solve_inequalities` (not
  `unify_row` as sketched here). The direct self-reference case (AC1) is
  rejected too; per `dbf67fcb`'s commit message, reviewing this proposal's AC1
  test surfaced the shifted-flank hole that commit fixes.
- The error message says "rank cycle among row variables" rather than
  "occurs check" — the AC wording below ("occurs check" or "row variable")
  is satisfied via the latter alternative.
- Structural drift invalidates the code sketches below: `beg_dims` moved from
  `Row_var` to `Row.t` (commit `ff8c6846`), so `Row_var { v; beg_dims = [d] }`
  no longer type-checks. Line numbers drifted: `dim_var_occurs_in_dim` is now
  at `row.ml:1681`, its use in `unify_dim` at `row.ml:1941`, `subst_row` at
  `row.ml:1348`, the depth-16 caps at `row.ml:1386`/`1493`, the depth-4 cap at
  `row.ml:3285`, and the "Infinite number of axes by self-reference" sites at
  `row.ml:1364`, `2047`, `2093`, `2167`, `2723`.
- Repo-wide renames since this was written (broadcast-order reversal LUB→GLB,
  dimension "label"→"basis") do not affect the substance of this proposal.
- Remaining: nothing implementation-side; only upstream issue hygiene
  (close #247 / link the landed commits and test).

## Goal

Add a row-variable occurs check to shape unification so that any cyclic
constraint -- a row variable bound (directly or transitively) to a row that
contains itself in its `bcast` field -- is detected and reported as a
`Shape_error` with a clear message, rather than risking infinite recursion or
stack overflow inside `subst_row` / `unify_row`. This mirrors the existing
dim-variable occurs check (`row.ml:1611`, `row.ml:1893`) *(Update 2026-06-12:
now `row.ml:1681`, `row.ml:1941`)* and brings shape
inference closer to the standard "occurs check" required for sound first-order
unification.

The original issue notes the problem is "needed for completeness but unlikely
to manifest" -- current source-language patterns don't drive unification into
this corner. The change is therefore a robustness/diagnostics improvement, not
a fix for a user-visible bug.

## Acceptance Criteria

- [ ] **Direct self-reference is detected.** A unit test that constructs a row
      `r1 = { dims=[]; bcast = Row_var { v; beg_dims=[d] } }` and unifies it
      with `r2 = { dims=[d]; bcast = Row_var { v; beg_dims=[] } }` (i.e.
      forces the binding `v |-> [d | v]`) raises `Row.Shape_error` whose
      message contains either `"occurs check"` or `"row variable"` and names
      the cyclic variable. *Mutation falsifier:* deleting the new occurs check
      makes the test hang or stack-overflow inside `subst_row`.
- [ ] **Indirect cycles via a second row variable are detected.** A test that
      first records `s |-> [d | r]` (via `unify_row`) and then attempts to
      bind `r |-> [e | s]` raises `Shape_error` with a message identifying
      one of the variables involved in the cycle. *Mutation falsifier:*
      restricting the check to direct self-reference only (i.e. only checking
      whether the bound RHS literally contains `v` syntactically, not after
      substitution chain) leaves this test hanging or stack-overflowing.
- [ ] **No regressions in existing shape inference tests.** All tests under
      `test/einsum/` and `test/operations/` still pass (`dune runtest
      test/einsum test/operations`). *Mutation falsifier:* an over-eager
      check that flags any RHS mentioning the same `row_var` -- including the
      legitimate `v = v` no-op handled at `row.ml:1298-1300` -- breaks the
      existing suite.
- [ ] **Error message names the offending row variable.** The raised message
      includes the `sexp` of the cyclic `row_var` (or its provenance), so
      users can trace which axis of which tensor produced the cycle.
      *Mutation falsifier:* a generic `"shape error"` message would not
      contain the variable identifier and the assertion would fail.
- [ ] **The new check fires before any potentially non-terminating
      operation.** Specifically, `subst_row` for a `Row_var v` whose env
      entry is a `Solved_row` mentioning `v` (transitively) must not recurse
      indefinitely. A focused property test (or a simple manually constructed
      env) verifies that calling `subst_row env r` on a deliberately
      ill-formed env returns within a small bound or raises `Shape_error`.

## Context

### OCANNL audit pause

Per harness memory, **autonomous OCANNL work is paused** for the user's
hands-on quality audit (see `project_ocannl_quality_audit.md`). This
proposal will be filed and the task left in `deferred_launch` state -- no
agent slot will auto-start it. The intent is for the user (or a future
audit-cleanup pass) to pick this up alongside other v1.0 polish items.

The work itself is well-bounded: a single auxiliary recursion plus 2-3 call
sites in `row.ml`, and a new test file. There is little ambiguity in the
implementation, but two small design choices (below) are worth flagging.

### Where the relevant code lives today

The seeded design pointer in the task file references `~/ocannl/tensor/shape.ml`,
but row-variable handling has long since been factored out into a dedicated
module:

- **`tensor/row.ml`** -- the entire row/dim unification engine, including
  `unify_row`, `subst_row`, `s_row_one`, and the env types `dim_entry`/`row_entry`.
- **`tensor/row.mli`** -- public surface, including `unify_row`,
  `subst_row`, `Shape_error`, and the `row_var` abstract type.
- **`tensor/shape.ml`** -- higher-level shape API; calls into `Row` for
  unification but does not itself implement it.

### Existing partial cycle checks

*(Update 2026-06-12: the line numbers in the table below predate the landed
rank-cycle work and the `beg_dims` move to `Row.t`; they no longer match.
The transitive check now lives in `add_rank_edge` / `closes_positive_rank_cycle`
near `row.ml:2027`.)*

Several `Shape_error "Infinite number of axes by self-reference"` sites
*already* catch specific symptoms of cyclic row-var bindings:

| Site | Case caught |
|---|---|
| `row.ml:1298-1304` (`subst_row`) | `v` resolves to a `Solved_row` whose `bcast` is `Row_var v2` with `equal_row_var v v2` and length mismatch -- the resolved row directly references itself. |
| `row.ml:1999-2001` (`unify_row`) | Two rows share the same `row_var` head but their non-row dims have incompatible lengths. |
| `row.ml:2034-2044` (`unify_row`) | Unifying `Row_var v` with another row where the residual `dims` after suffix matching is non-empty and the tail variable is `v` again. |
| `row.ml:2546-2551` (`solve_row_ineq`) | The inequality form of the same self-reference case. |

These are all **shallow / one-step** detectors. They correctly handle the
case where the cycle arises in a single unification step and the same
`row_var` appears explicitly on both sides. They do **not** catch:

1. A multi-step indirect cycle (`v |-> [d | s]`, then `s |-> [e | v]`),
   where each individual binding is well-formed but the composition is
   cyclic.
2. A self-cycle that becomes apparent only after substituting through one
   or more `Solved_row` entries already in the env (`subst_row` assumes the
   env is "idempotent (solved wrt. equalities)" -- a comment at
   `row.ml:1306` -- but nothing currently enforces that invariant).

### Existing precedent: dim-variable occurs check

`row.ml:1611-1618` defines `dim_var_occurs_in_dim`, and `row.ml:1893-1897`
uses it to short-circuit `unify_dim` when a dim variable would be bound to
a term containing itself:

```ocaml
| Var v, dim2 | dim2, Var v ->
    if dim_var_occurs_in_dim v dim2 then
      raise
      @@ Shape_error
           ( "occurs check failed: dimension variable occurs in its own definition",
             [ Dim_mismatch [ Var v; dim2 ] ] );
    ...
```

The row-variable check should follow this exact shape: a syntactic
`row_var_occurs_in_row` predicate, applied at every site that is about to
record `v |-> value` in the env.

### Termination safeguards elsewhere

`apply_rows_constraint` and `apply_row_constraint` (`row.ml:1321-1323`,
`1424-1426`) cap recursion at `depth > 16`, and `eliminate_rows_constraint`
(`row.ml:3040-3042`) caps at `depth > 4`. These guards apply to constraint
*reduction*, not to `subst_row`/`unify_row` directly. So the practical
worst case today is: an ill-formed env makes `subst_row` recurse via the
`Solved_row` lookup at `row.ml:1296-1315` until the OCaml stack overflows.
The depth-16 cap at the constraint layer would not help because the
recursion is happening one level lower.

### Test convention

Existing shape-error tests (e.g. `test/operations/test_param_shape_error.ml`,
`test/einsum/test_einsum_capture.ml`) follow the pattern:

```ocaml
try
  ... force the offending unification ...
with Row.Shape_error (msg, _) ->
  Stdio.printf "Got expected error: %s\n" msg
```

with an `.expected` file containing the matching `printf` output. The new
tests should follow this convention.

## Approach

### 1. Add a syntactic occurs predicate in `row.ml`

Place near `dim_var_occurs_in_dim` (~line 1611):

```ocaml
let row_var_occurs_in_row (v : row_var) (r : t) : bool =
  match r.bcast with
  | Broadcastable -> false
  | Row_var { v = v'; _ } -> equal_row_var v v'
```

This is a **direct syntactic** check on a single row -- it does *not*
itself walk through env entries, exactly mirroring how
`dim_var_occurs_in_dim` is purely syntactic. The "indirect cycle" case is
handled by calling this predicate **after `subst_row env value`** at every
binding site, since `subst_row` already chases through `Solved_row`
entries. So a multi-step cycle reduces to a syntactic self-reference once
the substitution is fully applied.

### 2. Apply the check at every `v |-> value` binding site in `unify_row`

The single binding point is `row.ml:2068-2104` (the `f`/`result` closures
that ultimately call `add_row row_env ~key:v ~data:(Solved_row value)`).
Insert, immediately before constructing `value` is committed to the env:

```ocaml
let value = subst_row env value in (* if not already substituted *)
if row_var_occurs_in_row v value then
  raise
  @@ Shape_error
       ( Printf.sprintf "occurs check failed: row variable %s occurs in its own definition"
           (Sexp.to_string (sexp_of_row_var v)),
         [ Row_mismatch [ row_of_var v value.prov; value ] ] );
```

The `row_of_var` helper is already in scope (used at e.g. `row.ml:2112`).

### 3. (Defensive) Add the same check inside `subst_row` to catch ill-formed envs

At `row.ml:1305` (the `Some (Solved_row { dims = more_dims; bcast; _ })`
branch -- the recursion-via-substitution path), apply the predicate to the
fully-resolved continuation before returning the constructed row. If the
result still mentions `v` in its `bcast`, raise the same error rather than
returning a row that would re-trigger the recursion on the next call.

This is belt-and-braces: with step 2 in place, the env should never reach
this state, but the guard prevents pathological behaviour if `unify_row`
is bypassed (e.g. by a future caller that builds an env directly).

### 4. Tests

New test file: `test/einsum/test_row_var_occurs.ml` (with corresponding
`.expected` and a dune stanza in `test/einsum/dune`). The file exposes
`unify_row` directly via `Row` -- this works because `Row.unify_row` is
already in the public `.mli`.

Three test cases:

```ocaml
let test_direct_self_reference () =
  Stdio.printf "Test direct self-reference\n";
  let v = Row.get_row_var () in
  let d = Row.get_dim ~d:3 () in
  let r1 = { Row.dims = []; bcast = Row_var { v; beg_dims = [ d ] }; prov = ... } in
  let r2 = { Row.dims = [ d ]; bcast = Row_var { v; beg_dims = [] }; prov = ... } in
  try
    let _ = Row.unify_row ~stage:Stage1 [] (r1, r2) Row.empty_env in
    Stdio.printf "FAIL: unify_row did not raise\n"
  with Row.Shape_error (msg, _) ->
    Stdio.printf "Got expected error: %s\n" msg

let test_indirect_cycle_via_two_vars () = ... (* v |-> [d|s], s |-> [e|v] *)

let test_legitimate_v_eq_v_still_works () = ... (* v = v must not raise *)
```

The third test pins down the falsifier for an over-eager check: the
identity binding `v = v` (handled at `row.ml:1298-1300`) must continue to
short-circuit, not raise.

### 5. Optional: error trace with constraint chain

The `error_trace` extensible variant already exists; the proposal raises
`Row_mismatch [ row_of_var v _; value ]` which is the standard pattern.
If the user wants better debugging UX (the cycle's full expansion chain),
the trace can be extended, but the simpler form is sufficient for v1.0.

## Ambiguities for user input

These are minor; defaults are noted, the user may overrule.

1. **Eager or lazy occurs check?** Standard practice (and the existing
   dim-var check) is **eager** -- check at every binding. Lazy ("only when
   we detect divergence") complicates the code without gain. *Default:
   eager.*

2. **Indirect cycles in scope, or only direct self-reference?** Direct-only
   leaves a real (if unlikely) hole. *Default: include indirect cycles*,
   handled cheaply by applying `subst_row` before the syntactic predicate
   (step 2 above). The cost is one extra `subst_row` call per row-var
   binding, which is negligible compared to the unification work itself.

3. **Error message verbosity.** Two options:
   (a) Just name the offending `row_var` (simple, matches dim-var
       precedent).
   (b) Include the full chain of intermediate bindings that closed the
       cycle (better debug UX, requires threading more state through the
       check).
   *Default: (a)*, with the existing `error_trace` mechanism providing the
   structured context. (b) can be added later if real-world cycle
   diagnostics prove insufficient.

## Out of Scope

- Refactoring the four existing "Infinite number of axes by self-reference"
  sites. They catch length-mismatch corner cases that are *not* occurs-check
  failures (they're symptom-level guards), and conflating them with the new
  check would muddy diagnostics. Leave them in place.
- Adding occurs checks to `proj_id` or other variable kinds -- separate
  concern, separate completeness gap.
- Performance-tuning `subst_row` (the env-idempotency assumption could be
  formalised, but that's a larger refactor).

## Estimated Effort

**Small** -- 0.5-1 day. One auxiliary function, ~3 call-site edits, one
new test file with 3 cases. The bulk of the work is verifying that the
check fires at every binding site without breaking the legitimate `v = v`
short-circuit.

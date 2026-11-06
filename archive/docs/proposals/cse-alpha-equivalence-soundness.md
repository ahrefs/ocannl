# Fix Unsound Alpha-Equivalence in CSE (`cse_equal_scalar`) and a Hoist Hazard Gap

## Goal

Repair a soundness bug in the CSE comparator `cse_equal_scalar` in
`arrayjit/lib/low_level.ml`. The comparator's variable renaming is
one-directional (a function, not a bijection), so it can judge two scalar
trees alpha-equivalent when they are not. Both CSE passes consume a positive
result by *replacing* one computation with a `Get_local` reference to another,
so a false positive is a miscompilation — a read silently rewired to a
different value — not merely a missed optimization.

Two adjacent defects of the same class are addressed in the same pass: a
hazard gap in `hoist_shared_locals` where `writes_of_stmt` ignores writes
performed inside sibling `For_loop`s, and a consistency bypass in the
`Local_scope` arm of the comparator that records its scope-id mapping with a
raw `Hashtbl.set` instead of the checked path.

This change should land **before** virtualization scope is extended under
gh-ocannl-133: the diagonal and repeated-symbol patterns #133 enables are
exactly the inputs that trigger the primary bug, so landing them in the wrong
order turns a latent bug into a live shape/indexing miscompile that is hard to
localize.

Surfaced during the gh-ocannl-296 audit. Related: gh-ocannl-351 (the CSE pass
itself), gh-ocannl-133 (diagonal / repeated-symbol virtualization).

## Acceptance Criteria

- `cse_equal_scalar` treats both its symbol renaming (`sym_renaming`) and its
  scope-id renaming (`scope_renaming`) as partial **bijections**: on first
  encounter of a pair, the binding is recorded only if neither the source is
  already mapped nor the target is already claimed by a different source. A
  target already claimed by a different source makes the comparison return
  `false`.
- The comparator is **symmetric**: `cse_equal_scalar a b` and
  `cse_equal_scalar b a` agree for all inputs. (The current implementation does
  not — see Bug 1.)
- A regression test demonstrates that two `Local_scope`s (or two statements)
  whose only structural difference is a distinct-vs-repeated index — one
  accessing `t[i; j]`, the other `t[i; i]`, with otherwise identical bodies —
  are **not** judged equal/CSE'd, in **both** orderings (the more-distinct
  expression seen first, and the diagonal expression seen first).
- A regression test confirms that legitimate alpha-equivalent pairs (same
  structure, consistently renamed fresh symbols) are still judged equal and
  still CSE'd, so the fix does not silently disable the pass.
- The `For_loop` hazard gap in `hoist_shared_locals` is either fixed
  (`writes_of_stmt` accounts for writes inside `For_loop` bodies) or shown
  unreachable by a documented invariant, with a test or code comment recording
  which resolution was chosen and why.
- The `Local_scope` arm of the comparator records its scope-id mapping through
  the consistency-checking path (`ids_equal` or equivalent) rather than an
  unconditional `Hashtbl.set`.
- Existing CSE and virtualization tests pass without regression
  (`arrayjit/test/test_cross_cse`, `test/einsum/test_cse`, and the broader
  test suite).

## Context

### The comparator and how its result is consumed

`cse_equal_scalar` (function `cse_equal_scalar` in
`arrayjit/lib/low_level.ml`) decides alpha-equivalence of two scalar trees,
treating loop-iterator symbols and `Local_scope` scope ids as bound variables
matched up to renaming. It maintains two hashtables — `sym_renaming :
Symbol.t -> Symbol.t` and `scope_renaming : int -> int` — built as it descends,
with nested helpers `ids_equal`, `sym_equal`, `idx_equal`, and the mutually
recursive `equal_t` / `equal_scalar` / `equal_arg`.

Both CSE passes consume a positive result by rewriting one occurrence into a
reference to another:

- `eliminate_common_subexpressions`: when a `Local_scope` matches a previously
  seen one, the current node is replaced with `Get_local existing_id`, pointing
  at the *earlier* one.
- `hoist_cross_statement_cse` → `hoist_shared_locals` →
  `replace_local_scope_in_scalar`: occurrences matching a group representative
  are replaced with `Get_local canonical_id`, pointing at the representative's
  hoisted body.

In both, a *false* positive does not waste an optimization opportunity — it
makes a read return a different computation's value. This is why the bug is
severe rather than cosmetic.

### Bug 1 — non-injective renaming (unsound; primary)

`sym_equal` and `ids_equal` check only the forward direction. Current
`sym_equal`:

```ocaml
let sym_equal (s1 : Indexing.symbol) (s2 : Indexing.symbol) =
  match Hashtbl.find sym_renaming s1 with
  | Some mapped -> Indexing.equal_symbol mapped s2
  | None ->
      Hashtbl.set sym_renaming ~key:s1 ~data:s2;
      true
```

This enforces that the renaming is a *function* (each `s1` maps to one `s2`)
but not that it is *injective* (two distinct `s1`, `s1'` may both map to the
same `s2`). Alpha-equivalence requires a bijection; only half is checked.
`ids_equal` has the identical defect for scope ids.

**Minimal counterexample.** Compare a tree reading `Get(t, [Iterator a;
Iterator b])` (a ≠ b) against one reading `Get(t, [Iterator c; Iterator c])`:
position 0 records `a → c` (true); position 1 records `b → c` (true). The two
are judged equal, but `t[a, b]` ranges over a rectangle and `t[c, c]` over the
diagonal — different computations. The real trigger is via `orig_indices` in
the `Local_scope` arm, where two accesses of the same virtual node at `[i; j]`
versus `[i; i]` compare their index arrays with this same buggy `idx_equal`,
pass, and then their structurally identical bodies pass too.

**Asymmetry.** Feeding the diagonal tree first — comparing `[c; c]` against
`[a; b]` — position 1 does `sym_equal c b`, finds `c` already mapped to `a`,
checks `a = b`, fails — correctly rejected. So the truth value depends on
argument order; the order-dependence is the signature of the missing reverse
map. Because the rewrite replaces a later occurrence with a reference to the
earlier-seen one, the false positive fires precisely when the more-distinct
expression is recorded first.

**Interaction with #133.** Today the virtualizer rejects repeated-symbol
diagonal accesses (`Non_virtual 5`), so diagonal `Local_scope`s with `[i; i]`
orig_indices rarely reach CSE. #133 deliberately mints exactly those
repeated-symbol expressions and feeds them into CSE.

### Bug 2 — `writes_of_stmt` ignores `For_loop` (hoist hazard gap; reachability unconfirmed)

`writes_of_stmt` has a catch-all `_ -> Set.empty (module Tn)`, so a `For_loop`
statement contributes **no** writes:

```ocaml
let writes_of_stmt (stmt : t) : Set.M(Tn).t =
  match stmt with
  | Set { tn; _ } -> Set.singleton (module Tn) tn
  | Set_from_vec { tn; _ } -> Set.singleton (module Tn) tn
  | Zero_out tn -> Set.singleton (module Tn) tn
  | _ -> Set.empty (module Tn)
```

`hoist_shared_locals` builds its hazard set by unioning `writes_of_stmt` over
the sibling statements from the first to the last user of a hoisted
`Local_scope`, and inserts the hoisted body before the first user only if the
body's reads do not intersect that hazard set. `flat_lines` flattens `Seq` but
keeps `For_loop` opaque, so any tensor write performed *inside* a sibling
`For_loop` between two users is invisible to the check, permitting an unsound
hoist above it (later users would read the pre-loop value).

Whether this is reachable depends on an unconfirmed invariant: whether a
`Local_scope` body at this pipeline stage can read a materialized tnode that a
sibling `For_loop` writes. The invariant must be verified before deciding the
resolution.

### Bug 3 — raw `Hashtbl.set` in the `Local_scope` arm (minor)

The `Local_scope` case of `equal_scalar` does `Hashtbl.set scope_renaming
~key:id1.scope_id ~data:id2.scope_id` directly, instead of going through the
consistency-checking `ids_equal`. It overwrites any existing mapping for that
scope id without verifying consistency — the same class of defect as Bug 1.
This is probably benign (`get_scope` ids are globally unique and a binder
precedes its uses), but a sound implementation would not have the bypass.

### Test infrastructure

`cse_equal_scalar` is **not** exported in `arrayjit/lib/low_level.mli` (only
`eliminate_common_subexpressions` and `hoist_cross_statement_cse` are public).
The existing `arrayjit/test/test_cross_cse.ml` exercises the comparator
*indirectly* by constructing `Local_scope`/`Set` nodes directly (the
`scope_id` record and the `t` / `scalar_t` constructors are exposed in the
`.mli`) and running them through `hoist_cross_statement_cse`, then inspecting
the printed result. New regression tests should follow that pattern: build a
diagonal (`[i; i]`) and an off-diagonal (`[i; j]`) `Local_scope` pair, run the
public pass in both statement orderings, and assert the diagonal case is *not*
merged into a `Get_local` reference. A genuine alpha-equivalent pair (fresh but
consistently-renamed symbols) should still merge. `test/einsum/test_cse.ml`
provides expectation-test fixtures driven from the einsum frontend, including
diagonal patterns, and is a natural place for an end-to-end legitimate-CSE
regression check.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

1. **Make the renaming bijective.** Add reverse companions
   `sym_renaming_rev` and `scope_renaming_rev` (or a single checked-update
   helper), and rewrite `sym_equal` / `ids_equal` along these lines:

   ```ocaml
   let sym_equal s1 s2 =
     match Hashtbl.find sym_renaming s1, Hashtbl.find sym_renaming_rev s2 with
     | Some mapped, _ -> Indexing.equal_symbol mapped s2
     | None, Some _ -> false                  (* s2 already claimed by another symbol *)
     | None, None ->
         Hashtbl.set sym_renaming ~key:s1 ~data:s2;
         Hashtbl.set sym_renaming_rev ~key:s2 ~data:s1;
         true
   ```

   The fix is local to `cse_equal_scalar`; the two calling passes are
   unchanged. Verify binder/shadowing handling in the `For_loop` and
   `Local_scope` arms under the new maps so a legitimately-shadowing binder is
   not over-rejected.

2. **Route Bug 3 through the checked path.** Once `ids_equal` is bijective,
   have the `Local_scope` arm record its scope-id mapping through `ids_equal`
   (or an equivalent checked update) rather than the raw `Hashtbl.set`.

3. **Add regression tests** for: (a) the diagonal-vs-off-diagonal pair rejected
   in both orderings; (b) symmetry on a small battery of pairs; (c) a genuine
   alpha-equivalent pair still equated and still CSE'd. Construct nodes
   directly as in `test_cross_cse.ml` (the comparator is not exported, so test
   through the public `hoist_cross_statement_cse` /
   `eliminate_common_subexpressions` passes). Einsum-based fixtures analogous
   to the `test/einsum` diagonal patterns are a natural source for the
   end-to-end legitimate case.

4. **Resolve Bug 2.** Confirm or refute the invariant about what a
   `Local_scope` body can read relative to sibling `For_loop` writes. If it can
   happen, fix `writes_of_stmt` to recurse into `For_loop` bodies (and any
   nested `Set` / `Set_from_vec` / `Zero_out`). If it provably cannot, record
   the invariant as a comment at the catch-all so the gap is not reintroduced.

## Risk Notes

- The fix should *narrow* what CSE accepts, never widen it: an over-conservative
  fix yields a missed optimization (safe), while leaving the bug yields a wrong
  answer. Bias toward conservative.
- After the bijection fix, re-run any benchmark/test that exercises heavy CSE
  to confirm the pass still fires on legitimate cases. A too-aggressive reverse
  check (e.g. failing to reset the reverse map on a binder that legitimately
  shadows) could over-reject.
- Sequencing with #133 is the project-level risk: this task gates #133. If #133
  is in flight, the comparator fix must land first, and the diagonal regression
  test should be added to any suite both touch.

## Scope

**In scope:** changes to `cse_equal_scalar` (bijective renaming, Bug 3 fix) and
`writes_of_stmt` / `hoist_shared_locals` (Bug 2 resolution) in
`arrayjit/lib/low_level.ml`; regression tests in `arrayjit/test/` and/or
`test/einsum/`.

**Out of scope:** the #133 virtualization-scope extension itself; broader
redesign of the CSE passes (#351); any change to the `Local_scope` /
`Get_local` IR representation.

**Dependencies:** blocks gh-ocannl-133 (must land first). Relates to
gh-ocannl-296 (audit origin) and gh-ocannl-351 (the CSE pass).

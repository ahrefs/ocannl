# Handoff: reverse the shape-order direction (code + docs)

## Summary

Align the OCANNL codebase and docs with a reversed orientation of the broadcast/shape
order. The claim-free unit `1∅` (size-1, no-basis, the `bcast_if_1` axis) becomes the
**top** of the order ("no claim"); the broadcast **error** becomes the **bottom** `⊥`;
broadcasting is the **meet**; `⊑` reads "refines / is at least as specific as." The five
blog articles have already been reworded to this convention; this task brings the code
and docs into line with them.

## The one invariant that matters

**This is a behavior-preserving relabeling.** The solver already computes the right thing.
Reversing an order swaps the *names* (top↔bottom, meet↔join, upper↔lower, LUB↔GLB,
sub↔super) while the computed values, the solve/defer decisions, the inferred shapes, the
derived projections, and the errors all stay **identical**. The article edits were a pure
relabeling and so is this.

Therefore:

- **Do not flip any comparison operator, branch condition, or "which bound solves" logic.**
  Renaming `upper_bound` to `lower_bound` is fine (it now holds what we *call* a lower
  bound); changing a `<=` to `>=` in the actual computation is a bug.
- The full test suite must pass with **identical inferred shapes and projections**. The
  only test expectations that may change are *wording* in printed/diagnostic strings
  (e.g. a message that literally said "bottom" or "least upper bound").
- If you find a case where the new framing seems to imply *different* behavior (an error
  the old code silently broadcast, or vice versa), **stop and flag it** — that is a
  semantic question for Łukasz, not part of this rename.

## The convention (authoritative; use to self-check)

- `1∅` (claim-free, size-1, `bcast_if_1` tag) = **⊤ top**, "no claim", identity of meet.
  Everything refines it: `d ⊑ 1∅`. The empty row is the **top** of the row order.
- **⊥ bottom = contradiction / broadcast error** (a name for the clash/failure state that
  already exists; you need not introduce a value unless you want it in diagnostics).
- `⊑` = "refines / at least as specific as"; concrete sizes are an **antichain** of atoms
  in the middle.
- **Broadcasting = meet.** Combining two ground axes is their meet; a true clash is
  `meet → ⊥` (error). A variable pushed by incompatible broadcast caps generalizes,
  `join → ⊤` (becomes `1∅`, no error) — the deliberate-broadcast case.
- **Solver asymmetry (names flip, behavior identical):** a bound that **pins a variable to
  a concrete atom solves immediately**; a **permissive cap defers**; incompatible caps
  **join to the top `1∅`**. (Old wording: "lower bound solves, upper bound defers,
  incompatible uppers meet to bottom.")
- **Closing (names flip, behavior identical):** a **leaf** closes **downward** to the most
  specific shape its bounds permit (its **GLB**, formerly LUB); an unforced **interior**
  closes **upward** to `1∅` / the empty row (the **top**). Hole fallback is still
  dim-1 / no-more-axes (which is now the top).
- The completed order is **non-distributive** (`M₃`): three or more size atoms between the
  shared top `1∅` and bottom `⊥`. Documentation point only; no code consequence.

## Rename map (identifiers)

These are the concrete renames. Confirm exact spellings by grep first (this list is from
description, not from reading the source).

| Old | New | Notes |
|-----|-----|-------|
| `subr` (sub-row / sub-tensor side) | `opnd` | the broadcastable **operand** side |
| `cur` (super / current side) | `res` | the **result** side |
| "subtensor" / "sub" (in names, comments) | "operand" | role name, direction-neutral |
| "supertensor" / "super" | "result" | role name |
| `lub` / `least_upper_bound` / `LUB` | `glb` / `greatest_lower_bound` / `GLB` | same computed value; the order reversed, so old-LUB *is* new-GLB |
| `upper_bound` / `ub` (accumulated solver bounds) | `lower_bound` / `lb` | **review + verify by tests**; names swap, values unchanged |
| `lower_bound` / `lb` | `upper_bound` / `ub` | the dual of the above |

The inequality orientation in the solver: the broadcast constraint is now read
`R_result ⊑ R_operand` (result more specific, below; operand more permissive, above).
This is the same constraint, re-read; rename the two sides per the table, do not re-solve
them differently.

## Comment / docstring / diagnostic text reorientations

Search comments, docstrings, and any user-facing strings and flip the *vocabulary*:

- `1∅` "bottom" → "top"; "claim-free bottom" → "claim-free top".
- empty row "least element" → "greatest element"; "below everything" → "above everything".
- "meet-semilattice with bottom (and no joins)" → "join-semilattice with top, completed
  by an error bottom `⊥` to a (non-distributive) bounded lattice".
- broadcasting described as a join / upper bound → meet / greatest-lower-bound.
- the `bcast_if_1` basis tag comment: "the claim-free broadcast **bottom**" →
  "the claim-free broadcast **top**". **Keep the identifier** `bcast_if_1` /
  `broadcastable_when_1`: the rule "broadcastable when size 1" is direction-neutral.
- closing: "leaves … upward … least upper bound" → "leaves … downward … greatest lower
  bound"; "interiors … downward to the bottom (`1∅` / empty)" → "interiors … upward to the
  top (`1∅` / empty)".
- size-0 coproduct unit: "two bottoms" → "two units". Make clear that size-0 is the
  coproduct unit, sits **off** the broadcast order (it is the bottom of the *additive*
  order the coproduct induces), and is **distinct from** the error bottom `⊥`.

## What NOT to touch

- **Surface syntax is unaffected.** The einsum notation, `batch|input->output` printing,
  the `^` concatenation operator, `+*` / `++` / `++^`, `%op` / `%cd` — none depend on the
  order direction. Leave them alone.
- The arithmetic of the solver, projections inference, padding fixpoint, concatenation
  hypergraph — unchanged.

## Files to sweep

- Code: the `Row` and `Shape` modules (the constraint solver and closing live here), and
  any projections / `Indexing` code or printers that mention bounds or poles.
- Docs (in priority order):
  - `shape_inference.md` — at least two known spots: the `bcast_if_1` type comment
    ("…broadcast bottom…") and the terminal-substitution line ("…substituted by their
    **least upper bounds** if any…" → "…**greatest lower bounds**…"). Grep the rest.
  - `syntax_extensions.md`, `slides-shapes_and_einsum.md`, `migration_guide.md` — mostly
    surface syntax; scan for stray "bottom"/"upper bound"/"meet"/"two bottoms" only.

## Grep checklist

```
bottom        top           lub           LUB            least upper
upper bound   lower bound   subr          \bcur\b        \bsub\b
super         subtensor     supertensor   semilattice    meet
join          two bottoms   bcast_if_1    broadcastable_when_1
descend.*bottom             claim-free
```

After the pass, remaining hits should be either intentional (new "top"/"GLB"/"meet"
wording) or genuinely direction-neutral (e.g. the `bcast_if_1` identifier).

## Verification

1. `dune build` is clean.
2. Full test suite passes with **identical** inferred shapes and projections. Any change
   in a numeric/shape expectation means behavior changed — revert and investigate.
3. Update only wording-level test expectations (printed messages that named a pole/bound).
4. Re-run the grep checklist; confirm no stale directional vocabulary remains.

## Why this direction (for a design comment, if you want one)

Broadcasting is the **unique idempotent** of the three operations on sizes, so it alone
carries a meet order, with its unit `1∅` at the top. The size **product** (`×`) and the
**coproduct** (`+`, concatenation) are non-idempotent, so each admits only its divisibility
order, in which the unit sits at the *bottom* (`1` divides everything; `0` is a summand of
everything). That is why `1∅` lifts to the top of the broadcast order while the coproduct
unit `0` stays at the bottom of the additive order — and why `0` must not be conflated with
the broadcast error `⊥`. The reversed orientation also matches the subtyping convention
(`⊤` = no constraint, `⊥` = contradiction) and makes broadcasting the meet in agreement
with Bachurski et al.'s Star, at the cost of the divisibility ("broadcasts-to") reading the
old orientation had. See the articles `broadcast-aware-shape-inference.md` (the order
definition and the direction-decision paragraph) and `a-shape-is-not-its-index.md` (the
design-space and principality remarks) for the full rationale.

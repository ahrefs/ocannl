# Allow Multiple Virtual Tensors to Share the Same For Loop

## Status update (2026-06-12)

- Issue #134 is still **OPEN**. Its GH milestone is v0.7 (due 2026-01-30, now past); ROADMAP.md — the milestone authority — does not list #134 explicitly, so it has effectively slipped past the v0.7.x window (which closed mid-April 2026).
- **The feature has not landed**: `reverse_node_map` is still `(Symbol.t, Tnode.t) Hashtbl.t` and `track_symbol` (in `visit_llc`, lines ~315-324) still marks both tnodes `is_complex <- true` when a second tnode uses the same symbol. The `TODO(#134)` comment is still there (now at line ~313).
- `arrayjit/lib/low_level.ml` was heavily reworked by the cross-statement CSE landing (#351; commits e48ec84f, cdc14726, 64685b24), so all cited line numbers drifted — updated in place below. Structural descriptions remain accurate.
- New since this proposal: `track_symbol` is now also applied to symbols inside `Affine` indices and the new `Concat` indices (from the tensor-stacking work, commit 58bfd6e5). The in-code comment now names Block/concat lowering as a concrete producer of symbol-sharing across tensors — strengthening the motivation for this issue.
- The pipeline gained a stage: it is now `visit_llc -> virtual_llc -> cleanup_virtual_llc -> simplify_llc -> eliminate_common_subexpressions -> hoist_cross_statement_cse`. CSE-with-hoisting may interact with any new partial-loop-stripping logic and should be regression-tested together.
- `virtual_llc` now takes a persisted computations table as its first argument (`input_ctx.computations` from `optimize_ctx`), and `inline_computation` likewise — the bookkeeping changes proposed here must thread through that context.
- All acceptance criteria remain to do.

## Goal

Enable multiple virtual (inlined) tensors that are computed within the same for loop to all be virtualized, rather than forcing them to materialize. Currently, when two tensors share a loop iterator symbol, both are marked `is_complex = true` in `visit_llc`, which prevents their virtualization. This limits optimization opportunities in cases like element-wise operations where several intermediate tensors are computed in the same loop.

Tracked by: https://github.com/ahrefs/ocannl/issues/134

## Acceptance Criteria

- Multiple virtual tensors within a single for loop are each inlined correctly at their use sites.
- `reverse_node_map` is changed from mapping each symbol to a single `Tnode.t` to mapping each symbol to a set of tnodes, so that shared symbols do not force `is_complex = true`.
- `virtual_llc` handles for loops that contribute computations for multiple virtual tensors: when the loop iterator maps to several tnodes, each virtual tnode's computation is extracted and stored via `check_and_store_virtual`.
- `cleanup_virtual_llc` removes for loops where all contributing tensors have been virtualized, and preserves loops that still have non-virtual residual operations.
- `inline_computation` correctly handles inlining a computation that was extracted from a multi-tensor for loop body (the inlined fragment should contain only the operations for the target tnode, not sibling operations).
- Existing virtual tensor tests continue to pass.
- No performance regression for programs that do not exercise the multi-virtual-tensor-per-loop pattern.

## Context

### The problem

In `visit_llc` (lines ~315-324 of `arrayjit/lib/low_level.ml`), a `track_symbol` helper maps each loop iterator symbol to the tnode it belongs to via `reverse_node_map : (Symbol.t, Tnode.t) Hashtbl.t`. When a second tnode uses the same symbol, both are marked `is_complex <- true`. The comment (now at line ~313) explicitly notes: "See TODO(#134): this prevents multiple virtual arrays from sharing for loops." *(Update 2026-06-12: `track_symbol` is now applied not just to `Iterator` symbols but also to symbols inside `Affine` and `Concat` indices.)*

The `is_complex` flag blocks virtualization at line 431: `virtualize_settings.inline_simple_computations && (not traced.is_complex)` -- a complex tensor with too many accesses becomes `Never_virtual`.

### Current pipeline

```
visit_llc -> virtual_llc -> cleanup_virtual_llc -> simplify_llc -> eliminate_common_subexpressions -> hoist_cross_statement_cse
```

*(Line numbers below updated 2026-06-12 after the CSE rework of low_level.ml; the final `hoist_cross_statement_cse` stage is new, from #351.)*

1. **`visit_llc`** (line 257): traces tensor accesses, builds `reverse_node_map` (symbol -> single tnode), marks `is_complex`.
2. **`virtual_llc`** (line 780): walks the LLC tree. For `For_loop` nodes whose index maps to a tnode in `reverse_node_map`, it calls `check_and_store_virtual` to record the entire loop as a computation for that single tnode. Currently only one tnode per loop is handled. It now takes the computations table (persisted in `optimize_ctx`) as its first argument.
3. **`check_and_store_virtual`** (line 460): validates that a computation block is suitable for inlining (no escaping variables, consistent indices, has setters), then stores it in `computations_table`.
4. **`inline_computation`** (line 627): at each `Get` site for a virtual tnode, substitutes the stored computation inline, filtering to only the operations on the target tnode (via `Tn.equal` checks on `Set`/`Set_from_vec`/`Zero_out`/`Get`).
5. **`cleanup_virtual_llc`** (line 866): removes operations on now-virtual tnodes from the original locations -- for loops whose iterator maps to a virtual tnode are entirely removed.

### Key insight

`inline_computation` already filters the stored computation body to extract only operations for the target tnode (now lines ~707-721: `Set`/`Set_from_vec`/`Zero_out`/`Get` with `Tn.equal tn traced.tn`). This means even if the full loop body contains operations for multiple tensors, inlining will correctly select only the relevant subset. The main changes needed are in the bookkeeping layer, not the inlining logic itself.

### Approach sketch

1. **Change `reverse_node_map`** from `(Symbol.t, Tnode.t) Hashtbl.t` to `(Symbol.t, Tnode.t list) Hashtbl.t` (or a set). Remove the `is_complex <- true` marking when multiple tnodes share a symbol.

2. **Update `virtual_llc`**: when a for loop's index maps to multiple tnodes, call `check_and_store_virtual` for each virtual tnode in the list, passing the same loop body. Each tnode gets the full loop body stored as its computation.

3. **Update `cleanup_virtual_llc`**: when a for loop's index maps to multiple tnodes, only remove the loop if *all* mapped tnodes are virtual. If some are virtual and some are not, keep the loop but strip out the operations for the virtual tnodes (they have been inlined elsewhere).

4. **Partial loop cleanup**: add logic to `cleanup_virtual_llc` to filter out `Set`/`Zero_out` operations for virtual tnodes within a kept loop body, while preserving operations for non-virtual tnodes. *(Update 2026-06-12: this filtering already exists — `cleanup_virtual_llc`'s `Set`/`Set_from_vec`/`Zero_out` branches drop ops on not-known-non-virtual tnodes statement-by-statement, and a `For_loop` whose body filters to `None` disappears via the `Option.map`. The only change needed is in the `For_loop` dispatch: recurse into the body instead of wholesale-dropping the loop when the index maps to a virtual tnode.)*

### Key files

- **`arrayjit/lib/low_level.ml`** -- all functions listed above: `visit_llc` (line 257), `check_and_store_virtual` (line 460), `inline_computation` (line 627), `virtual_llc` (line 780), `cleanup_virtual_llc` (line 866), `optimize_proc` (line 1619) *(line numbers as of 2026-06-12)*
- **`arrayjit/lib/tnode.ml`** -- `known_non_virtual`, `update_memory_mode`, `is_complex` usage
- **`arrayjit/lib/low_level.mli`** -- public interface for `optimize_proc` and related types

### Risks

- **Circular dependencies**: two virtual tensors in the same loop that reference each other. `check_and_store_virtual` already detects escaping variables and recurrence, which should catch most cases. The `process_for` set in `virtual_llc` prevents infinite recursion. *(Update 2026-06-12: this is too optimistic — even a one-directional sibling read (B's setter reads A, both set in the shared loop) passes all `check_and_store_virtual` checks, because the shared loop index is in `env_dom` (no escape) and A ≠ top_tn (no recursion check). The raw `Get(A, …)` then survives inside B's stored computation (A is in `process_for` so it is kept as a "recursive" get), gets inlined at B's use sites, and `cleanup_virtual_llc` collides: `Virtual` forced at A's definition vs `Never_virtual 17` at the surviving `Get` — a memory-mode contradiction. Sibling reads must be detected and excluded explicitly.)*
- **Partial loop stripping correctness**: removing virtual-tnode operations from a mixed loop body while preserving iteration structure requires care. Operations may have ordering dependencies (e.g., a non-virtual tensor reads from a virtual one within the same loop iteration).
- **`is_complex` serves dual purpose**: it also reflects genuinely complex computations (via `is_complex_comp`), not just symbol sharing. The fix must only remove the symbol-sharing source of complexity, not the computation-complexity source.

## Design review (2026-06-12)

**Verdict: sound-with-changes** — the bookkeeping direction is right and the "key insight" (inlining already filters per-tnode) checks out against the code, but the proposal misses that the current safety story is incomplete, overstates the cleanup work needed, and understates the sibling-dependency hazard.

**Recommendations:**

1. **Treat this as a correctness fix, not just an optimization unlock.** `is_complex = true` does NOT force `Never_virtual` — it only disables the `skip_simple` exemption from the max-visits check (`visit_llc` line ~431), so two low-access tensors sharing a loop can both proceed to virtualization *today*. When that happens, `cleanup_virtual_llc` drops the whole loop keyed on whichever tnode was tracked first in `reverse_node_map`, deleting the sibling's setters; and a sibling read produces the `Virtual 15` / `Never_virtual 17` memory-mode contradiction described in the annotated risk above. Add a regression test for the shared-loop case *before* the rework, to characterize current behavior.
2. **Simplify the cleanup plan: recurse instead of strip.** Per the annotation on step 4, the per-statement filtering and empty-loop elision already exist. The `For_loop` branch of `cleanup_virtual_llc` should stop consulting `reverse_node_map` for wholesale dropping and simply recurse; virtual tnodes get their `Virtual` mode forced at their `Set`/`Zero_out` sites (provenance 151/152) rather than at the loop (15). This makes acceptance criteria 3-4 nearly free and removes the FIXME(#296)-flavored mode-forcing at the loop level.
3. **Handle sibling reads explicitly in `virtual_llc`, conservatively at first.** In the `For_loop` branch, before storing computations, compute the set of tnodes written in the loop (`writes_of_stmt`, reusable from the CSE landing) and each candidate's reads (`reads_of_body`). If any candidate's computation reads a sibling that is also written in the loop, keep *both* non-virtual (exactly today's intent for that pair) and relax only independent siblings. Inlining-the-sibling-into-the-reader is a sensible follow-up, but it requires a second storing pass per loop (chicken-and-egg: both computations come from the same `For_loop` node) — out of scope for the first landing.
4. **Store the full loop body per tnode, as proposed — but flag the duplication cost.** Each virtual tnode's use site replays the whole (filtered) loop; sibling replays are *not* alpha-equivalent, so neither `eliminate_common_subexpressions` nor `hoist_cross_statement_cse` recombines them. For a loop computing N inlinable tensors each used K times, generated-code size grows ~N·K loop bodies where the materialized version ran one loop. The "no performance regression" acceptance criterion should be made concrete: measure compile-time/code-size on the Block/concat-lowering paths that motivated the `track_symbol` extension.
5. **Thread the change through `optimize_ctx` consciously.** `computations_table` is persisted across routines; storing a shared loop under multiple tnode keys is compatible with the append semantics in `check_and_store_virtual`, but the `process_for` guard in `virtual_llc`'s `For_loop` branch must add *all* of the loop's candidate tnodes before recursing, or the same loop gets stored twice for the second tnode.

**Open decision points for Łukasz:**

- Conservative sibling-read handling (keep both non-virtual) vs. nested inlining (inline A into B's stored computation) in the first landing? Recommend conservative; nested inlining as follow-up.
- Is `Tnode.t list` in `reverse_node_map` acceptable, or should the loop-level candidate discovery move out of `visit_llc` entirely (compute written-tnode sets directly in `virtual_llc` from the loop body, making `reverse_node_map` redundant for this purpose)? The latter is arguably cleaner but a bigger diff.
- Should a cap (e.g. on loop-body size or sibling count) gate multi-tensor virtualization to bound the code-duplication cost, analogous to `max_visits`?

**Pairing with #133 (landing order):** land **#134 first**, as separate changes. The two touch disjoint machinery (#134: `visit_llc`/`virtual_llc`/`cleanup_virtual_llc` bookkeeping; #133: `check_idcs`/`make_subst` index reasoning) and share no code, so merging them buys nothing. But #134 changes *which* tensors reach `check_idcs`/`make_subst` (removing the `is_complex` symbol-sharing marking), so #133's tests are only stable after #134 settles; and #134 doubles as a latent-bug fix, making it higher priority. After both land, add a joint test: a diagonal tensor and an independent element-wise tensor computed in the same loop, both consumed downstream.

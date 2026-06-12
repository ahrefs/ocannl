# Collapse Repeated Identifiers in debug_name

## Status update (2026-06-12)

- Issue #281 is still OPEN, milestone v0.7. Harness status: deferred. Not implemented — `get_debug_name` still joins components verbatim with no collapsing.
- Motivating issue #210 (runtime argument labels) is CLOSED as completed (back in v0.4), so the label repetition this proposal addresses can occur today; the motivation stands.
- Line numbers drifted: `get_debug_name` is now at `arrayjit/lib/tnode.ml` lines 96-114 (was 121-139). The structure described (code_name short-circuit, `is_alphanum_` filter, `"grad"` strip, `String.concat ~sep:"_"`, `.grad` suffix) is unchanged and the proposed insertion point (between grad-strip and concat) remains valid.
- One detail not in the original write-up: the current code also has fallback branches for empty `components` (`"n<id>"` / `"n<id-1>.grad"`), which the collapsing helper does not affect.
- The approach (self-contained `collapse_consecutive` helper in `tnode.ml`, no API changes) is still accurate; nothing else in the proposal is invalidated.

## Goal

When a tensor's `label` list contains consecutive identical identifiers (e.g. `["attention"; "attention"; "attention"]`), `get_debug_name` should collapse them into a count-suffixed form (e.g. `"attention3"`) instead of producing verbose names like `"attention_attention_attention"`. This keeps debug output readable as issue #210 (incorporating runtime argument labels) introduces label repetition.

Tracked by: https://github.com/ahrefs/ocannl/issues/281

## Acceptance Criteria

- Consecutive identical components in the filtered label list are collapsed with a numeric suffix: N identical copies of `"foo"` become `"fooN"` (for N >= 2).
- A single (non-repeated) component is left unchanged (no `"foo1"` suffix).
- Non-consecutive duplicates are not collapsed: `["a"; "b"; "a"]` remains `"a_b_a"`.
- Mixed sequences work correctly: `["encoder"; "attention"; "attention"; "output"]` produces `"encoder_attention2_output"`.
- The collapsing applies only to the alphanumeric-filtered components, before the `_`-join and before the grad suffix logic.
- Existing tests pass (expected outputs in `test/operations/*.expected` may need updating if any current labels exhibit consecutive repetition).

## Context

### Current implementation

`get_debug_name` in `arrayjit/lib/tnode.ml` (lines 96-114 as of 2026-06-12):

1. If `code_name` is set, uses it directly (with `.grad` suffix handling).
2. Otherwise, filters `label` through `is_alphanum_` to keep only alphanumeric components.
3. Strips a leading `"grad"` component and tracks `is_grad`.
4. Joins remaining components with `"_"` via `String.concat ~sep:"_"`.
5. Appends `.grad` if applicable.

The collapsing step should be inserted between step 3 (grad stripping) and step 4 (concatenation), operating on the `components` list.

### Approach

Add a helper function (e.g. `collapse_consecutive`) that folds over the string list, grouping consecutive equal elements and emitting `ident` for count=1 or `identN` for count>=2. Apply it to `components` before the `String.concat` call. This is a self-contained change to `tnode.ml` with no API changes.

### Related

- **#210**: Incorporate runtime argument labels -- the feature that motivates this collapsing. *(Update 2026-06-12: #210 is closed/completed, so the repetition it produces is live in current debug output.)*

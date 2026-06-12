# Proposal: Remove remaining unnecessary zeroing-out in backend code

**Issue**: https://github.com/ahrefs/ocannl/issues/382

## Status update (2026-06-12)

- Issue #382 is still OPEN on GitHub (milestone v0.7), but the work proposed here has effectively landed under the closely related issue #420 ("Optimize away zeroing-out before non-reducing accumulating assignment", CLOSED as completed, milestone v0.7.2).
- The exact approach proposed below was implemented in `c_syntax.ml` (commits 5c075df7 and c9e69816, May 2026, gh-ocannl-420): `zero_out_loop_redundant` consults `current_traced_store` and elides the `Zero_out` loop — but only for the *first-touch* occurrence at function scope (tracked via `zero_out_seen`, with an `~in_loop` parameter threaded through `pp_ll`), so genuine re-zeros (`Zero_out; Set; Zero_out`, or `Zero_out` inside a loop body) still emit their loops. See `c_syntax.ml:305-369`.
- The bijective-projection criterion is covered: `needs_init` in `assignments.ml` (now around line 422) skips zeroing when the projection is surjective and injective, and commit 46ec60d4 fixed `is_surjective` for trivial-dim projections.
- A direct-codegen regression test exists: `arrayjit/test/test_zero_out_codegen.ml` (with `.expected`).
- The introductory slides (`docs/slides-basics_backprop_training_codegen.md`) no longer show the dual zeroing of `n35` — only the `= {0}` declaration remains, satisfying the "visibly cleaner" criterion.
- Remaining: nothing substantive; issue #382 should likely be closed as completed (or repurposed for any leftover audit of `initialize_neutral` call sites). Note also that since this proposal was written, `Zero_out` elision interacts with the new cross-statement CSE hoisting pass in `low_level.ml`, which did not exist when the line numbers below were recorded. *(Update 2026-06-12: "nothing substantive" needs qualification — generated code today still contains one visible redundant zeroing pattern, the `Local_scope` scalar double zero-init; it is in scope of the issue's broad title but is tracked by open issue #340, not by this proposal. See the Design review section below for empirical evidence and analysis.)*
- Line-number drift in the references below: the `Zero_out` codegen is now at `c_syntax.ml:358-369`, the `= {0}` declaration emission around `c_syntax.ml:960`, and the `backends.ml` allocation decision around line 505.

## Goal

Eliminate redundant zeroing in generated C code. Currently, local arrays that need zero-initialization get zeroed twice: once via the C declaration initializer (`= {0}`) and again via an explicit loop generated from the `Zero_out` IR node. Only one mechanism should fire for any given array, and in cases where the accumulation projection is bijective (surjective + injective), no zeroing should be emitted at all.

## Acceptance Criteria

- No generated C code has both a `= {0}` declaration initializer AND an explicit zeroing loop for the same array.
- When accumulation projection is both surjective and injective, no zeroing is emitted (no `= {0}`, no loop).
- No regression in existing tests -- zeroing still occurs when genuinely needed (partial writes, re-zeroing between iterations, non-bijective projections).
- Generated code from examples (e.g., introductory slides matmul) is visibly cleaner.

## Context

### Root Cause

The dual-zeroing arises from two independent mechanisms triggered by the same `Zero_out` IR node:

1. **Tracing** (`low_level.ml:289-296`): When `Zero_out tn` is traced and the array has no prior assignments or accesses, `zero_initialized_by_code` is set to `true`. This causes the C declaration to include `= {0}` (`c_syntax.ml:872`).

2. **Code generation** (`c_syntax.ml:332-335`): `Zero_out tn` is unconditionally expanded into an explicit `loop_over_dims` that sets every element to `(float)(0)`. *(Update 2026-06-12: no longer unconditional — the first-touch, function-scope `Zero_out` is now elided; see Status update.)*

Both mechanisms fire for the same array, producing the redundant code shown in the issue.

### Key Code Paths

- `assignments.ml:417-424` -- `needs_init` check: already correctly skips zeroing when projection is surjective AND injective. This is not the problem site.
- `assignments.ml:718-719` -- `Fetch { fetch_op = Constant 0.0 }` becomes `Zero_out`.
- `low_level.ml:289-296` -- Tracing sets `zero_initialized_by_code` on first-touch `Zero_out`.
- `c_syntax.ml:332-335` -- Code gen expands `Zero_out` into explicit zeroing loop.
- `c_syntax.ml:871-873` -- Declaration emits `= {0}` when `zero_initialized_by_code` is true.
- `backends.ml:485-487` -- Device allocation uses `alloc_array` (skipping `alloc_zeros`) when `zero_initialized_by_code` is true.

### The Three Zeroing Levels

| Level | Mechanism | When needed |
|-------|-----------|-------------|
| Declaration `= {0}` | C local array init | First-touch zeroing of stack-allocated arrays |
| Explicit loop | `Zero_out` IR expansion | Re-zeroing between iterations (not first touch) |
| `alloc_zeros` | Device buffer allocation | Device buffers without `zero_initialized_by_code` |

The fix must distinguish first-touch zeroing (declaration suffices) from re-zeroing (loop required).

### Approach

**In `c_syntax.ml` code generation for `Zero_out`**: The `traced_store` is already available in the compilation context (it is part of the `optimized` record passed to the compilation function). When generating code for `Zero_out tn`:

- Look up `tn` in `traced_store`.
- If `zero_initialized_by_code` is true for that node, emit `Noop` instead of the zeroing loop -- the declaration `= {0}` already handles it.
- Otherwise (re-zeroing, or array not declaration-zeroed), emit the loop as before.

This is the minimal, safe fix. The `zero_initialized_by_code` flag is only set when `Zero_out` is the first operation on the array (no prior assignments or accesses), which exactly matches the condition where declaration initialization suffices.

**Verification**: The `needs_init` logic (`assignments.ml:422-424`) already prevents `Zero_out` from being emitted when projection is bijective. The dual-zeroing only occurs for arrays that genuinely need initialization but get it twice. No change to `needs_init` or `initialize_neutral` is required.

### Edge Cases

- **Re-zeroing between iterations**: When `Zero_out` fires on an array that already has prior assignments, `zero_initialized_by_code` remains `false` (the tracing condition at `low_level.ml:291` checks `Hash_set.is_empty traced.assignments`). The explicit loop is correctly preserved.
- **Device buffers**: Unaffected -- they use `alloc_zeros` / `alloc_array` path in `backends.ml`, not the C declaration mechanism.
- **Merge buffers**: Follow the same tracing logic; the fix applies uniformly.
- **Relation to #340**: That issue concerns scalar `Local_scope` variables (`c_syntax.ml:514-528`; now around `c_syntax.ml:597`), which is a separate code path. #340 remains OPEN.

## Design review (2026-06-12)

**Verdict: done as scoped** — #420 implemented exactly this proposal's approach (first-touch `Zero_out` elision keyed on `zero_initialized_by_code`, guarded by `zero_out_seen` and `~in_loop`), with the bijective-projection criterion covered by `needs_init` plus the `is_surjective` fix (46ec60d4) and a direct codegen regression test (`arrayjit/test/test_zero_out_codegen.ml`). The freshness audit's "LOOKS-DONE via #420" is confirmed empirically, with one important qualification about residual zeroing that belongs to #340.

**Empirical verification (2026-06-12).** Ran the issue's own example (`test/training/moons_demo.exe` with `--ocannl_output_debug_files_in_build_directory=true`) on current master and inspected `build_files/scalar_loss_forward_and_gradient_then_gd_update.c` and `moons_infer.c`:

- No array anywhere gets both `= {0}` and an explicit zeroing loop — the issue's reported redundancy (the `n35` dual zeroing) is gone. All acceptance criteria of this proposal hold.
- The only remaining per-call zeroing of a materialized buffer (`mlp_point_mlp_result[0] = (float)(0);` in `moons_infer.c`) is semantically required (it seeds a per-invocation accumulation, and allocation skips zeroing exactly because `zero_initialized_by_code` is set — `backends.ml:505-507`).
- **Residual redundancy found**: every virtualized accumulation block emits a scalar double zero-init:
  ```c
  float v27_n35 = (float)(0);   /* Local_scope declaration, unconditional init (TODO(#340), c_syntax.ml:602-603) */
  v27_n35 = (float)(0);         /* the virtualized Zero_out, inlined as Set_local(id, 0) */
  ```
  This appears 2x per matmul in the moons forward pass and again throughout the backward pass. It is literally "remaining unnecessary zeroing-out in backend code", but it is the declared scope of open issue **#340** (milestone v0.8; proposal `docs/proposals/gh-ocannl-340.md`, already symlinked in `docs/in-progress/`).
- **Non-obvious but verified**: #340's design (populate `needs_init` from `traced_array.read_before_write`) does fix this accumulator case, not just the non-accumulating case. Because `Zero_out` sets `traced.zeroed_out`, the accumulator's `Get`s are classified with `is_assigned = true` (`low_level.ml:398`) and are never `Recurrent`, so `read_before_write` stays `false` → the declaration loses its `= 0` and the body's single explicit zero remains. No #420-style second mechanism is needed at the `Local_scope` level.

**Recommendations:**

1. **Close #382 as completed**, citing commits 5c075df7, c9e69816, 46ec60d4 and the `test_zero_out_codegen` regression test, and explicitly hand the residual scalar double-init off to #340 in the closing comment so the trail is discoverable.
2. **Raise #340's priority** — it is the actual carrier of the still-visible zeroing clutter, it is past due by its own status note (ROADMAP places it in v0.7.2, due mid-April 2026), and its design is small and unblocked. If the high-priority flag on #382 reflects dissatisfaction with current generated code, #340 is the work item that addresses it.
3. **Do not extend this proposal with new mechanism.** The temptation to add a #420-style "elide leading `Set_local(id, 0)` when the declaration zero-inits" guard in `c_syntax.ml` should be resisted: per the analysis above it would duplicate what #340 achieves more cleanly at the IR level, and #340 must also cover the `Declare_local` hoisted-declaration init site, which a codegen-local guard would miss.
4. Optional small audit before closing: confirm no other backend-specific zeroing redundancy exists (`alloc_zeros` in `backends.ml` is already skipped when `zero_initialized_by_code`; the remaining conservatism — zeroing at allocation for nodes that are `zeroed_out` in code but not first-touch — is a one-time-per-allocation cost not worth new mechanism).

**Decision points for Łukasz:**

- Close #382 now and track the residual under #340 (recommended), or keep #382 open as an umbrella until #340 lands? If the latter, restore the `docs/in-progress/gh-ocannl-382.md` symlink; note this double-tracks #340, which already has an in-progress symlink.
- Should #340 be pulled forward from its v0.8 GitHub milestone to match ROADMAP's v0.7.2 placement, given it now carries the user-visible remainder of #382?

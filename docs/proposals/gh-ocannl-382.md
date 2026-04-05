# Proposal: Remove remaining unnecessary zeroing-out in backend code

**Issue**: https://github.com/ahrefs/ocannl/issues/382

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

2. **Code generation** (`c_syntax.ml:332-335`): `Zero_out tn` is unconditionally expanded into an explicit `loop_over_dims` that sets every element to `(float)(0)`.

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
- **Relation to #340**: That issue concerns scalar `Local_scope` variables (`c_syntax.ml:514-528`), which is a separate code path.

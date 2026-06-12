# Proposal: Restore `low_level.Fill` for dedicated backend fill operations

**Issue**: https://github.com/ahrefs/ocannl/issues/151

## Status update (2026-06-12)

- Issue #151 is still OPEN, milestone v0.8 (ROADMAP now targets mid-June 2026 for v0.8; the GH milestone due date of Feb 2026 lags).
- Not yet started: `Zero_out` is still the only bulk-init IR node (`low_level.ml:39`); no `Fill` variant exists; non-zero constants still lower to element loops.
- gh-ocannl-420 has since landed and CLOSED (commits `5c075df7`, `c9e69816`): `c_syntax.ml` now elides the *first-touch* `Zero_out` at function scope when the array declaration's `= {0}` already covers it (see `zero_out_seen` / `zero_out_loop_redundant`, around `c_syntax.ml:306-370`). Re-zeros still expand to element-by-element loops, so #151's payoff (bulk memset/blit for re-zeros and non-zero fills) remains.
- The `Zero_out` codegen handler moved from `c_syntax.ml:332-335` to ~line 358 and is now conditional; Step 4 must preserve the first-touch elision logic when generalizing to `Fill`.
- Line numbers in "Key code paths" have drifted slightly (assignments.ml lowering is now at 719-724; tracing at low_level.ml:291; virtualization at ~799/889; backends.ml allocation check at ~505; metal alloc_zeros at ~126-136). Identifiers are unchanged.
- gh-ocannl-350 (loop hoisting) was CLOSED NOT_PLANNED; cross-statement CSE with hoisting landed instead (`e48ec84f`). gh-ocannl-382 remains OPEN.
- Correction to History: issue #341 ("Resolve non-determinism of multicore_cc") is closed; the multi-streaming *cleanup* removed cross-stream automatic coherence and the deprecated multi-stream backend infrastructure (commit `272c0880`), but multiple streams per device remain. The load-bearing claim stands: `Parallel`/`Task_id` no longer exist in the IR (verified by grep), so the original removal rationale for `Fill` is moot.

## Goal

Restore a `Fill` instruction in the low-level IR so that filling an array with a constant value can be mapped to dedicated bulk operations on each backend (`memset` on CC, `cuMemsetD*` on CUDA, `MTLBlitCommandEncoder.fill` on Metal) instead of being expanded into element-by-element loops. This is a performance optimization for the v0.8 milestone.

## Acceptance Criteria

- A `Fill` variant exists in `Low_level.t` that takes a tensor node and a float value.
- `Fetch { fetch_op = Constant c }` in `assignments.ml` lowers to `Fill` for all constant values (not just 0.0).
- `Zero_out` becomes a special case of `Fill` (with value 0.0) or is replaced by it.
- The CC backend (`c_syntax.ml`) emits `memset` for `Fill` with value 0.0, and a typed `memset`-style loop or bulk fill for non-zero values where byte-level memset is not applicable.
- The CUDA backend emits `cuMemsetD8`/`cuMemsetD16`/`cuMemsetD32` for `Fill` where the value is representable at the appropriate granularity, falling back to a kernel loop otherwise.
- The Metal backend emits `MTLBlitCommandEncoder.fill` for zero fills, and a compute-based fill for non-zero values.
- The tracing/optimization pass (`low_level.ml`) handles `Fill` correctly: sets `zero_initialized_by_code` for `Fill 0.0`, tracks assignments, and virtualizes `Fill` nodes identically to `Zero_out`.
- The `Fill` instruction interacts correctly with padding regions (padding fill happens before the bulk fill, as with `Zero_out` today).
- All existing tests pass; `.expected` test files are updated to reflect the new IR node.
- The interpreter (if any) handles `Fill` correctly.

## Context

### History

`Fill` was removed in commit `5e808c79` (2023-05-09) with the message: "Remove `low_level.Fill`, it has tricky semantics -- It was naively and wrongly ignoring `Parallel` / `Task_id`." At that time, the IR had `Parallel` dimension markers and `Task_id` indices for multi-threaded execution. The old `Fill` simply called a whole-array fill, ignoring that in a parallel context each thread should only fill its portion.

Since then, the `Parallel`/`Task_id` machinery has been removed from the codebase (parallelism is now expressed via streams; the deprecated multi-stream backend infrastructure was further cleaned up in commit `272c0880`, though multiple streams per device remain). The tricky semantics that motivated the removal are no longer relevant. *(Update 2026-06-12: an earlier draft attributed the removal to gh-ocannl-341, which actually tracks multicore_cc non-determinism; corrected.)*

### Current state

Today, constant filling takes two paths through the IR:

1. **`Fetch { Constant 0.0 }`** lowers to `Zero_out tn` (`assignments.ml:719-720`). In the tracing pass, `Zero_out` sets `zero_initialized_by_code = true` and `zeroed_out = true`. At **code generation** time in `c_syntax.ml` (~line 358), `Zero_out` is expanded into an element-by-element zeroing loop via `loop_over_dims`. No `memset` is used in the generated C/CUDA code. *(Update 2026-06-12: since gh-ocannl-420 landed, the first-touch function-scope `Zero_out` is elided when the declaration's `= {0}` already covers it; re-zeros still emit the element loop.)*

2. **`Fetch { Constant c }` (non-zero)** lowers directly to `loop_over_dims` + `Set` at the assignments level (`assignments.ml:721-724`), producing an explicit for-loop in the low-level IR. There is no opportunity for the backend to use bulk operations.

3. **`Fetch { Constant neutral_value }`** is also emitted for `initialize_neutral` before accumulating assignments (`assignments.ml:447-450`). The neutral element can be 0.0 (Add/Sub), 1.0 (Mul/Div), infinity (Min), neg_infinity (Max), etc.

Meanwhile, the backends already have bulk fill capabilities that are only used at **allocation** time:
- CUDA: `Cu.Stream.memset_d8` in `alloc_zeros` (`cuda_backend.ml:84`)
- Metal: `Me.BlitCommandEncoder.fill_buffer` in `alloc_zeros` (`metal_backend.ml:100`)
- CC: relies on the OS for zero-initialized allocation

### Key code paths

*(Line numbers re-verified 2026-06-12 at HEAD `d9de22f0`.)*

- **Low-level IR type**: `arrayjit/lib/low_level.ml:33-50` -- `t` variant type, `Zero_out` at line 39
- **Assignments lowering**: `arrayjit/lib/assignments.ml:719-724` -- `Constant 0.0` to `Zero_out`, other constants to loops
- **Neutral element init**: `arrayjit/lib/assignments.ml:447-450` -- `Fetch { Constant neutral_value }` before accumulation
- **C syntax codegen**: `arrayjit/lib/c_syntax.ml:358-369` -- `Zero_out` expanded to loop (not memset), with first-touch elision (gh-ocannl-420) when the declaration already zeroes
- **CUDA alloc_zeros**: `arrayjit/lib/cuda_backend.ml:78-85` -- uses `memset_d8` at allocation only
- **Metal alloc_zeros**: `arrayjit/lib/metal_backend.ml:126-136` -- uses blit fill at allocation only
- **Tracing**: `arrayjit/lib/low_level.ml:291-297` -- `Zero_out` tracing sets `zero_initialized_by_code`
- **Virtualization**: `arrayjit/lib/low_level.ml:799`, `889` -- `Zero_out` virtualization
- **CSE/optimization**: `low_level.ml:1005,1025,1199,1283,1351,1378` -- `Zero_out` cases in various passes
- **Neutral elements**: `arrayjit/lib/ops.ml:447` -- `neutral_elem` function (Add->0, Mul->1, Max->neg_inf, etc.)

### Related issues

- **gh-ocannl-420**: Optimize away unnecessary zeroing-out (addresses the `is_surjective` bug causing spurious `Zero_out`). Complementary: #420 reduces the number of fills, #151 makes the remaining fills faster. *(Update 2026-06-12: CLOSED/completed — first-touch `Zero_out` elision now lives in `c_syntax.ml`.)*
- **gh-ocannl-382**: Remove unnecessary zeroing-out (broader scope). *(Still OPEN as of 2026-06-12.)*
- **gh-ocannl-350**: Loop hoisting + CSE (orthogonal optimization). *(Update 2026-06-12: CLOSED not-planned; cross-statement CSE with hoisting landed separately.)*

## Approach

### Step 1: Add `Fill` to the low-level IR

In `low_level.ml` and `low_level.mli`, add a new variant:

```ocaml
| Fill of { tn : Tnode.t; value : float }
```

This replaces `Zero_out of Tnode.t`. Keep `Zero_out` as a deprecated alias or remove it outright (replacing all occurrences with `Fill { tn; value = 0.0 }`). Removing `Zero_out` is cleaner since the number of match cases is manageable (~15 locations in `low_level.ml` plus backends).

### Step 2: Update tracing and optimization passes

In every match case that currently handles `Zero_out tn`, handle `Fill { tn; value }` instead:
- Tracing (`low_level.ml:291`): set `zero_initialized_by_code` when `value = 0.0`, and add a new `fill_initialized` flag or generalize to track the fill value.
- Virtualization: `Fill` of the virtual node inlines as `Set_local (id, Constant value)`.
- CSE, structural equality, printing: straightforward replacements.

### Step 3: Update assignments lowering

In `assignments.ml`, change:
- Lines 719-720: `Fetch { Constant 0.0 }` -> `Fill { tn = array; value = 0.0 }` (currently `Zero_out`)
- Lines 721-724: `Fetch { Constant c }` -> `Fill { tn = array; value = c }` (currently `loop_over_dims`)

### Step 4: Backend code generation

**C syntax (`c_syntax.ml`)**: Replace the `Zero_out` handler (~line 358) with a `Fill` handler, preserving the gh-ocannl-420 first-touch elision (`zero_out_seen` / `zero_out_loop_redundant`) for `value = 0.0`:
- For `value = 0.0`: emit `memset(tn, 0, size_in_bytes)` instead of an element-by-element loop.
- For non-zero values: emit a typed fill loop. (A `memset` only works for byte-replicable values. For arbitrary float values, a simple loop is still needed, but the loop is generated at code-gen time rather than polluting the IR.)

**CUDA backend (`cuda_backend.ml`)**: The C syntax module is shared. Additionally, for device-side code, `memset` cannot be called from a kernel. For the CUDA backend, `Fill` should be compiled to a host-side `cuMemsetD*` call injected via `Staged_compilation`, or the loop fallback for non-zero values.

**Metal backend (`metal_backend.ml`)**: Similarly, use `fill_buffer` for zero fills and a compute shader dispatch for non-zero fills.

### Step 5: Allocation optimization

In `backends.ml` (~line 505), the allocation path already distinguishes `zero_initialized_by_code` to skip `alloc_zeros`. With `Fill`, generalize: if the node's first operation is `Fill { value = 0.0 }` and we already used `alloc_zeros`, the `Fill` at code-gen time can be skipped (it's redundant). This avoids double-zeroing (the secondary issue from gh-ocannl-420).

### Step 6: Update tests

Update all `.expected` test files to reflect `Fill` instead of `Zero_out` in IR dumps. The `%cd` pretty-printer should show `fill tn value;` instead of `zero_out tn;`.

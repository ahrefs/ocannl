# Proposal: Restore `low_level.Fill` for dedicated backend fill operations

**Issue**: https://github.com/ahrefs/ocannl/issues/151

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

Since then, multi-streaming has been removed from OCANNL (gh-ocannl-341), and the `Parallel`/`Task_id` machinery no longer exists in the codebase. The tricky semantics that motivated the removal are no longer relevant.

### Current state

Today, constant filling takes two paths through the IR:

1. **`Fetch { Constant 0.0 }`** lowers to `Zero_out tn` (`assignments.ml:718-719`). In the tracing pass, `Zero_out` sets `zero_initialized_by_code = true` and `zeroed_out = true`. But at **code generation** time in `c_syntax.ml:332-335`, `Zero_out` is expanded into an element-by-element zeroing loop via `loop_over_dims`. No `memset` is used in the generated C/CUDA code.

2. **`Fetch { Constant c }` (non-zero)** lowers directly to `loop_over_dims` + `Set` at the assignments level (`assignments.ml:720-723`), producing an explicit for-loop in the low-level IR. There is no opportunity for the backend to use bulk operations.

3. **`Fetch { Constant neutral_value }`** is also emitted for `initialize_neutral` before accumulating assignments (`assignments.ml:447-450`). The neutral element can be 0.0 (Add/Sub), 1.0 (Mul/Div), infinity (Min), neg_infinity (Max), etc.

Meanwhile, the backends already have bulk fill capabilities that are only used at **allocation** time:
- CUDA: `Cu.Stream.memset_d8` in `alloc_zeros` (`cuda_backend.ml:84`)
- Metal: `Me.BlitCommandEncoder.fill_buffer` in `alloc_zeros` (`metal_backend.ml:100`)
- CC: relies on the OS for zero-initialized allocation

### Key code paths

- **Low-level IR type**: `arrayjit/lib/low_level.ml:33-50` -- `t` variant type, `Zero_out` at line 39
- **Assignments lowering**: `arrayjit/lib/assignments.ml:718-723` -- `Constant 0.0` to `Zero_out`, other constants to loops
- **Neutral element init**: `arrayjit/lib/assignments.ml:447-450` -- `Fetch { Constant neutral_value }` before accumulation
- **C syntax codegen**: `arrayjit/lib/c_syntax.ml:332-335` -- `Zero_out` expanded to loop (not memset)
- **CUDA alloc_zeros**: `arrayjit/lib/cuda_backend.ml:78-85` -- uses `memset_d8` at allocation only
- **Metal alloc_zeros**: `arrayjit/lib/metal_backend.ml:92-106` -- uses blit fill at allocation only
- **Tracing**: `arrayjit/lib/low_level.ml:289-296` -- `Zero_out` tracing sets `zero_initialized_by_code`
- **Virtualization**: `arrayjit/lib/low_level.ml:794-797`, `883-886` -- `Zero_out` virtualization
- **CSE/optimization**: `low_level.ml:998,1017,1190,1261,1331,1357` -- `Zero_out` cases in various passes
- **Neutral elements**: `arrayjit/lib/ops.ml:447` -- `neutral_elem` function (Add->0, Mul->1, Max->neg_inf, etc.)

### Related issues

- **gh-ocannl-420**: Optimize away unnecessary zeroing-out (addresses the `is_surjective` bug causing spurious `Zero_out`). Complementary: #420 reduces the number of fills, #151 makes the remaining fills faster.
- **gh-ocannl-382**: Remove unnecessary zeroing-out (broader scope).
- **gh-ocannl-350**: Loop hoisting + CSE (orthogonal optimization).

## Approach

### Step 1: Add `Fill` to the low-level IR

In `low_level.ml` and `low_level.mli`, add a new variant:

```ocaml
| Fill of { tn : Tnode.t; value : float }
```

This replaces `Zero_out of Tnode.t`. Keep `Zero_out` as a deprecated alias or remove it outright (replacing all occurrences with `Fill { tn; value = 0.0 }`). Removing `Zero_out` is cleaner since the number of match cases is manageable (~15 locations in `low_level.ml` plus backends).

### Step 2: Update tracing and optimization passes

In every match case that currently handles `Zero_out tn`, handle `Fill { tn; value }` instead:
- Tracing (`low_level.ml:289`): set `zero_initialized_by_code` when `value = 0.0`, and add a new `fill_initialized` flag or generalize to track the fill value.
- Virtualization: `Fill` of the virtual node inlines as `Set_local (id, Constant value)`.
- CSE, structural equality, printing: straightforward replacements.

### Step 3: Update assignments lowering

In `assignments.ml`, change:
- Line 718-719: `Fetch { Constant 0.0 }` -> `Fill { tn = array; value = 0.0 }` (currently `Zero_out`)
- Line 720-723: `Fetch { Constant c }` -> `Fill { tn = array; value = c }` (currently `loop_over_dims`)

### Step 4: Backend code generation

**C syntax (`c_syntax.ml`)**: Replace the `Zero_out` handler (lines 332-335) with a `Fill` handler:
- For `value = 0.0`: emit `memset(tn, 0, size_in_bytes)` instead of an element-by-element loop.
- For non-zero values: emit a typed fill loop. (A `memset` only works for byte-replicable values. For arbitrary float values, a simple loop is still needed, but the loop is generated at code-gen time rather than polluting the IR.)

**CUDA backend (`cuda_backend.ml`)**: The C syntax module is shared. Additionally, for device-side code, `memset` cannot be called from a kernel. For the CUDA backend, `Fill` should be compiled to a host-side `cuMemsetD*` call injected via `Staged_compilation`, or the loop fallback for non-zero values.

**Metal backend (`metal_backend.ml`)**: Similarly, use `fill_buffer` for zero fills and a compute shader dispatch for non-zero fills.

### Step 5: Allocation optimization

In `backends.ml:485-487`, the allocation path already distinguishes `zero_initialized_by_code` to skip `alloc_zeros`. With `Fill`, generalize: if the node's first operation is `Fill { value = 0.0 }` and we already used `alloc_zeros`, the `Fill` at code-gen time can be skipped (it's redundant). This avoids double-zeroing (the secondary issue from gh-ocannl-420).

### Step 6: Update tests

Update all `.expected` test files to reflect `Fill` instead of `Zero_out` in IR dumps. The `%cd` pretty-printer should show `fill tn value;` instead of `zero_out tn;`.

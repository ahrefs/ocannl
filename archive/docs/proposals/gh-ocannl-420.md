# Proposal: Optimize away zeroing-out before non-reducing accumulating assignment

**Issue**: https://github.com/ahrefs/ocannl/issues/420

## Goal

Eliminate unnecessary `Zero_out` IR nodes for non-reducing accumulating assignments. Currently, operations using `=:+` (accumulating assignment with `initialize_neutral = true`) emit a `Zero_out` before the actual assignment even when the projection is bijective (surjective + injective), meaning every output element is written exactly once. This wastes one full memory pass. Additionally, when `Zero_out` IS legitimately emitted, the generated C code can double-zero: once via declaration `= {0}` and again via an explicit zeroing loop. Both issues should be fixed.

## Acceptance Criteria

- Non-reducing accumulating assignments with bijective projections do not emit `Zero_out` in the lowered IR.
- The `top_down_prec.c.expected` test no longer has the redundant `d[0] = single_to_bfloat16((float)0);` line.
- The `top_down_prec-unoptimized.ll.expected` test no longer has `zero_out d<bfloat16>;` (and `zero_out n6<half>;` is also removed if applicable).
- Reducing accumulating assignments (genuine reductions where projection is not injective) continue to zero correctly.
- Non-surjective cases (e.g., diagonal tensor `i=>ii`) continue to zero correctly.
- When `Zero_out` is legitimately emitted AND the node has `zero_initialized_by_code = true`, the explicit zeroing loop in code generation is skipped (declaration `= {0}` suffices).
- Update all affected `.expected` test files.
- No regression in existing tests.

## Context

### Root Cause: `is_surjective` bug for trivial-dim projections

The primary issue is a bug in `indexing.ml:is_surjective`. For scalar arrays (or arrays where all dimensions are 1), the projection uses `Fixed_idx 0` for each axis (since `opt_symbol` returns `None` when dim <= 1). The surjectivity check at line 241:

```ocaml
Set.length lhs_symbol_set >= Array.length proj.project_lhs
```

fails because `lhs_symbol_set` is empty (no symbols from `Fixed_idx` entries), while `project_lhs` has one or more axes. So `0 >= 1 = false`, and `is_surjective` incorrectly returns `false`.

This causes `needs_init` at `assignments.ml:422-424` to be `true` even though the projection is trivially surjective (there is only one position per axis, and `Fixed_idx 0` covers it). The result is an unnecessary `Fetch { Constant 0.0 }` which lowers to `Zero_out`.

**Concrete example**: In `top_down_prec.ml`, `%op d = ({ a = [ 2 ] } + { b = [ 2 ] }) *. { c = [ 2 ] }` generates a pointwise multiplication with `initialize_neutral = true`. All tensors are scalar. The projection is identity (`Fixed_idx 0` for each axis). The assignment at `assignments.ml:394` correctly uses `set` (not `+=`) via `can_skip_accumulation`. But `needs_init` is incorrectly `true`, so `Zero_out` is emitted.

Note: `is_injective` does NOT have this bug -- for the scalar case, `product_iterator_sets = [[]]` (one empty combination), the empty set is a subset of the (empty) `lhs_symbol_set`, so `good` is non-empty and injectivity correctly returns `true`.

### Secondary Issue: dual zeroing in code generation

When `Zero_out` IS legitimately needed (e.g., non-surjective projections like diagonal tensors), two independent mechanisms fire:

1. **Tracing** (`low_level.ml:289-296`): Sets `zero_initialized_by_code = true` for first-touch `Zero_out`, causing `= {0}` in the C declaration (`c_syntax.ml:872`).
2. **Code generation** (`c_syntax.ml:332-335`): Unconditionally expands `Zero_out` to an explicit zeroing loop.

Both fire for the same array when it's the first operation. This is tracked more broadly in issue #382 but should be addressed here for completeness.

### Key Code Paths

- `indexing.ml:164-241` -- `is_surjective`: the buggy check (line 241)
- `indexing.ml:243-283` -- `is_injective`: correct for this case
- `assignments.ml:422-424` -- `needs_init` check using `is_surjective && is_injective`
- `assignments.ml:394` -- basecase correctly uses `set` (not `+=`) when `can_skip_accumulation`
- `assignments.ml:447-450` -- emits `Fetch { Constant 0.0 }` when `needs_init = true`
- `assignments.ml:718-719` -- `Fetch { Constant 0.0 }` lowered to `Zero_out`
- `low_level.ml:289-296` -- tracing sets `zero_initialized_by_code` on first-touch
- `c_syntax.ml:332-335` -- `Zero_out` unconditionally expanded to zeroing loop
- `c_syntax.ml:872` -- declaration emits `= {0}` when `zero_initialized_by_code`
- `backends.ml:485-487` -- allocation uses `alloc_array` when `zero_initialized_by_code`

### Test Files

- `test/operations/top_down_prec.c.expected` -- line 95: `d[0] = single_to_bfloat16((float)0);` (the redundant zeroing)
- `test/operations/top_down_prec-unoptimized.ll.expected` -- lines 3, 5: `zero_out` nodes
- `test/operations/top_down_prec.cu.expected` -- no zeroing (CUDA inlines differently)
- `test/operations/top_down_prec.metal.expected` -- no zeroing (Metal inlines differently)
- `test/einsum/test_accumulation_semantics.ml` -- tests that genuinely need zeroing (diagonal tensor, reduction)

## Approach

### Step 1: Fix `is_surjective` for trivial-dimension projections

In `indexing.ml`, the `is_surjective` function's final check should not count `Fixed_idx 0` axes where `dim <= 1`, since those positions are trivially covered:

```ocaml
(* Count how many LHS axes actually need coverage by iterator symbols *)
let non_trivial_lhs_count =
  Array.count2_exn proj.project_lhs proj.lhs_dims ~f:(fun idx dim ->
    match idx with
    | Fixed_idx 0 when dim <= 1 -> false  (* trivially covered *)
    | _ -> true)
in
Set.length lhs_symbol_set >= non_trivial_lhs_count
```

This replaces the current `Set.length lhs_symbol_set >= Array.length proj.project_lhs` on line 241. The same fix should be applied in the `has_affine` branch (line 237) for consistency.

This is the minimal fix for issue #420. After this change, `needs_init` will correctly be `false` for scalar pointwise operations and other all-dim-1 cases, and no `Zero_out` will be emitted for them.

### Step 2: Fix dual zeroing in code generation (addresses #382 partially)

In `c_syntax.ml`, when generating code for `Zero_out tn`, check the `traced_store` (already available in the compilation closure) to see if `zero_initialized_by_code` is true. If so, skip the explicit zeroing loop since the declaration `= {0}` already handles it:

```ocaml
| Zero_out tn ->
    let traced = Hashtbl.find_exn traced_store tn in
    if traced.Low_level.zero_initialized_by_code then
      (* Declaration already zeroed this array with = {0} *)
      empty
    else
      pp_ll
        (Low_level.loop_over_dims (Lazy.force tn.dims) ~body:(fun idcs ->
             Set { tn; idcs; llsc = Constant 0.0; debug = get_ident tn ^ " := 0" }))
```

This eliminates the double zeroing for first-touch `Zero_out` while preserving the explicit loop for re-zeroing between iterations (where `zero_initialized_by_code` is `false` because the tracing condition at `low_level.ml:291` requires no prior assignments).

### Step 3: Update test expectations

After both fixes:

1. Regenerate `test/operations/top_down_prec.c.expected` -- the `d[0] = single_to_bfloat16((float)0);` line should disappear.
2. Regenerate `test/operations/top_down_prec-unoptimized.ll.expected` -- `zero_out d<bfloat16>;` and possibly `zero_out n6<half>;` should disappear.
3. Check and update any other affected `.expected` files (scan for `zero_out` and zeroing patterns).
4. Run the full test suite to confirm no regressions, especially:
   - `test/einsum/test_accumulation_semantics` (tests diagonal tensor and reduction that genuinely need zeroing)
   - All other einsum and operation tests

### Edge Cases

- **Broadcasting**: When operands have different broadcast shapes, the projection may be non-injective (multiple RHS elements map to same LHS position). `is_surjective` already handles this correctly via the iterator symbol analysis -- the fix only changes the `Fixed_idx 0` trivial case.
- **Multi-dimensional arrays where some dims are 1**: The fix correctly counts only non-trivial axes. An array with dims `[3, 1, 5]` would have `Fixed_idx 0` for the middle axis, and the check would require 2 unique symbols for the other 2 axes.
- **Padding**: The fix is orthogonal to padding handling (`assignments.ml:426-441`). Padding resets use a separate code path.
- **Re-zeroing between iterations**: Protected by the `zero_initialized_by_code` flag only being set on first-touch (line 291 checks `Hash_set.is_empty traced.assignments`).
- **Backend uniformity**: C, CUDA, and Metal all use `c_syntax.ml` via the `C_syntax_config` module type, so the Step 2 fix applies uniformly.
- **Device buffer allocation**: Unaffected -- `backends.ml:485-487` already skips `alloc_zeros` when `zero_initialized_by_code` is true.

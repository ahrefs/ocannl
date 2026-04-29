# Write / Flesh-Out lowering_and_inlining.md and Audit low_level.ml

Tracked by: https://github.com/ahrefs/ocannl/issues/296

## Goal

Complete the documentation in `docs/lowering_and_inlining.md` to accurately reflect the current state of `low_level.ml`, and audit `low_level.ml` for correctness issues, dead code, stale FIXMEs, and documentation gaps. The doc is already substantial (316 lines) but has drifted from the code; the audit resolves the `FIXME(#296)` markers in `low_level.ml` and captures any findings as follow-up issues.

## Acceptance Criteria

### Documentation (lowering_and_inlining.md)

- [ ] **Accuracy pass**: Every code snippet, line-number reference, and behavioral claim in the doc matches the current source (1840 lines in `low_level.ml`).
- [ ] **Missing sections added**:
  - `Set_from_vec` / vector operations: the doc currently omits `Set_from_vec` handling in tracing, virtualization, inlining, and cleanup.
  - `eliminate_common_subexpressions` (CSE pass): added in the pipeline after `simplify_llc`, not yet documented as a pipeline phase.
  - `cse_equal_scalar`: alpha-equivalence comparison infrastructure (lines 1216-1292).
  - `substitute_float` / `substitute_proc`: scalar substitution helpers used in `simplify_llc` (lines 969-1004).
  - `loop_over_dims`, `unroll_dims`, `loop_over_padding_region`: utility functions for generating loop nests and padding-region iteration (lines 1695-1840).
  - `input_and_output_nodes`: how the optimizer determines which tensors are inputs vs outputs (lines 1367-1386).
  - `optimize` entry point vs `optimize_proc`: the `optimize` wrapper that hooks up pretty-printing callbacks (lines 1688-1693).
  - `Scope_id` module and `get_scope` UID generation.
  - `Get_merge_buffer` and merge-buffer single-buffer constraint.
  - `optimize_ctx` type and its role carrying `computations` across compilation calls.
- [ ] **Pipeline diagram updated**: Current doc shows `visit_llc -> virtual_llc -> cleanup_virtual_llc -> simplify_llc`. Actual pipeline is `visit_llc -> virtual_llc -> cleanup_virtual_llc -> simplify_llc -> eliminate_common_subexpressions`.
- [ ] **Settings section updated**: `inline_complex_computations` default changed from `false` to `true` in the current code (line 134). The doc still says "default: false".
- [ ] **Non_virtual exit codes updated**: Code 52 (Concat in check_idcs) and 140 (vec ops) are in the code but 52 is missing from the doc's list.
- [ ] **FIXME(#296) markers documented**: Each of the 5 `FIXME(#296)` sites (lines 874, 885, 891, 900, 928 in `cleanup_virtual_llc`) should be explained in the doc -- they mark places where the cleanup phase forces Virtual memory mode with specific provenance codes (15, 151, 152) for tensors not yet proven non-virtual, and one TODO about asserting Never_virtual.

### Audit (low_level.ml)

- [ ] **Resolve FIXME(#296) markers**: For each of the 5 `FIXME(#296)` comments, determine whether the current behavior is correct or needs a code change. If correct, replace the FIXME with an explanatory comment. If incorrect, file a follow-up issue.
- [ ] **TODO(#296) at line 928**: Evaluate whether the `Get` case in `cleanup_virtual_llc.loop_scalar` can indeed assert `Never_virtual` instead of calling `update_memory_mode`. If safe, convert to assert; otherwise document why not.
- [ ] **Dead code check**: Identify any unreachable match arms or functions in `low_level.ml`.
- [ ] **Edge cases**: Verify behavior when `dims` is empty (relevant to `Set_from_vec` tracing at line 341), when `computations` list is empty at inline time, and when `check_and_store_virtual` encounters `Concat` indices.
- [ ] **Audit findings documented**: All findings recorded in the task notes section, with follow-up issues filed for any bugs found.

## Context

### Current State of the Documentation

`docs/lowering_and_inlining.md` (316 lines) already covers:
- Low-level representation types (`t`, `scalar_t`, `scalar_arg`)
- Index types (`axis_index` including `Affine` and `Concat`)
- Translation from `Assignments` (projections to loops, symbol freshening, concatenation conversion)
- Optimization pipeline phases (tracing, virtualization, inlining, cleanup, simplification)
- Memory mode management
- Virtualize settings
- Limitations (affine index restriction #133, CSE gap #351)
- Code generation integration

The doc is high-quality but has fallen behind the code: the CSE pass was added but not documented, `Set_from_vec` support was added across the pipeline, and several defaults changed.

### Key Source Files

- **`arrayjit/lib/low_level.ml`** (1840 lines) -- IR types, optimization pipeline, pretty-printing
- **`arrayjit/lib/assignments.ml`** (988 lines) -- High-level assignment representation, `to_low_level` translation
- **`docs/lowering_and_inlining.md`** (316 lines) -- Existing documentation

### FIXME(#296) Sites in low_level.ml

All 5 markers are in `cleanup_virtual_llc` (lines 860-967):

1. **Line 874** (`For_loop` case): When a for-loop's symbol maps to a tnode in `reverse_node_map` that is not known non-virtual, forces `Virtual 15` and removes the loop. FIXME suggests this may be too aggressive.
2. **Line 885** (`Zero_out` case): Forces `Virtual 151` for non-virtual tnodes. Same concern.
3. **Line 891** (`Set` case): Forces `Virtual 152` for non-virtual tnodes. Same concern.
4. **Line 900** (`Set_from_vec` case): Forces `Virtual 152` for non-virtual tnodes. Same concern.
5. **Line 928** (`Get` in `loop_scalar`): Sets `Never_virtual 17`. TODO suggests this should already be `Never_virtual` by this point and could be an assert.

### Related Issues

- **#133**: Affine index virtualization (multiple non-static symbols) -- doc mentions this limitation
- **#134**: Virtual tensors sharing for-loops -- affects `reverse_node_map` shared-symbol tracking
- **#351**: CSE after inlining -- the CSE pass is now implemented but doc doesn't reflect it

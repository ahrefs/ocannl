# Common Subexpression Elimination After Inlining

## Motivation

OCANNL's virtualization pipeline inlines tensor computations into their consumers, eliminating intermediate memory allocations. However, when a tensor is accessed multiple times, each access duplicates the entire computation tree. This redundancy is the reason `inline_complex_computations` defaults to `false` ([`low_level.ml` line 133-135](../arrayjit/lib/low_level.ml)), leaving many tensors materialized unnecessarily.

CSE after inlining would give the best of both worlds: no intermediate allocations (from inlining) and no redundant computation (from deduplication). This unblocks enabling `inline_complex_computations = true` by default.

Tracked by: https://github.com/ahrefs/ocannl/issues/351

## Current State

### Compilation pipeline (in `optimize_proc`, line ~1238)

```
visit_llc -> virtual_llc -> cleanup_virtual_llc -> simplify_llc
```

The `simplify_llc` pass (lines 1007-1192) performs constant folding, algebraic simplification, local-scope elimination, FMA detection, and integer power unrolling. It does **not** detect or eliminate common subexpressions.

### How duplication arises

`inline_computation` (lines 624-761) creates a fresh `Local_scope` with a unique `scope_id` for each access site. Two accesses to the same tensor produce two `Local_scope` nodes with identical bodies but different `scope_id` values. The generated code computes the same expression twice:

```c
float v1 = 0; v1 = x[i] + y[i];  // first access
float v2 = 0; v2 = x[i] + y[i];  // second access (identical)
result[i] = v1 * v2;
```

### Equality infrastructure

`scalar_t` and `scalar_arg` already derive `equal` and `compare`. `scope_id` derives `equal`, `hash`, and `compare` via `Scope_id` module (line 13-22). The key subtlety: CSE must compare `Local_scope` bodies while ignoring their `scope_id`, since each inlining creates a fresh one.

### Key files

- **`arrayjit/lib/low_level.ml`** -- IR types (lines 33-65), inlining (lines 624-761), simplification (lines 1007-1192), pipeline (line 1238-1257)
- **`arrayjit/lib/c_syntax.ml`** -- C code generation for `Local_scope` (lines ~514-528), shows redundant declarations
- **`test/einsum/inline_permuted_view.ml`** -- test with `FIXME(#351)` that manually enables `inline_complex_computations`
- **`docs/lowering_and_inlining.md`** -- documents the optimization pipeline and CSE gap (lines 298-303)

## Proposed Change

Add a CSE pass that runs after `simplify_llc` in the `optimize_proc` pipeline. The pass should:

1. **Walk the IR in program order**, maintaining a table mapping expression structure to the `scope_id` where it was first computed.

2. **Compare `Local_scope` bodies structurally**, ignoring `scope_id` values but including `orig_indices` -- two `Local_scope` nodes are CSE-equivalent when their bodies and index patterns are identical after recursive CSE of their sub-expressions.

3. **Replace duplicates with `Get_local` references** to the first occurrence's `scope_id`. This is safe because program-order traversal guarantees the original `Local_scope` dominates the replacement site.

4. **Respect loop scoping**: expressions computed inside a `For_loop` must not be reused outside it. The CSE table should be scoped per loop level (save/restore on entry/exit).

5. **Check precision as part of the key**: identical expressions at different precisions are not equivalent.

After CSE is working:

- Flip `inline_complex_computations` default to `true` (line 134)
- Remove the `TODO(#351)` comment
- Update `FIXME(#351)` in `test/einsum/inline_permuted_view.ml` (line 15)

### Acceptance criteria

- Structurally identical scalar expressions after inlining are computed once and reused via `Get_local`
- CSE integrates into the pipeline after simplification, before code generation
- `inline_complex_computations = true` becomes the default
- `TODO(#351)` and `FIXME(#351)` markers are resolved
- No regression in existing tests
- Measurable reduction in generated code size for models with shared tensor accesses

## Scope

**In scope:**
- Expression-level CSE: deduplicating entire `Local_scope` nodes that arise from repeated inlining of the same tensor
- Scope-aware CSE table with loop boundary handling
- Pipeline integration and default-flag change
- Updating tests and TODO/FIXME markers

**Out of scope:**
- Subexpression-level CSE within different computations (e.g., shared `a + b` inside `(a+b)*c` and `(a+b)*d`) -- can be a follow-up
- Commutativity normalization (`a + b` vs `b + a`) -- can be a follow-up
- Interaction with loop-invariant code motion (gh-ocannl-350) -- that task can build on CSE independently
- Hash-consing or global value numbering -- overkill for the current IR structure

**Dependencies:**
- No blocking dependencies. gh-ocannl-350 (loop hoisting) is related but independent.
- gh-ocannl-340 (Local_scope init tracking) may benefit from fewer Local_scope nodes after CSE.

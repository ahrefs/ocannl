# Enable Inlining for Virtual Nodes with Non-Linear Index Symbols

## Goal

Allow virtual node (inlining) optimization to handle tensors whose setter indices involve non-linear symbol usage: (1) multiple non-static symbols in a single `Affine` index position, and (2) the same symbol appearing at multiple index positions (diagonal tensors). Currently, these patterns trigger `Non_virtual 51` and `Non_virtual 5` respectively, forcing materialization. Diagonal tensors created via einsum (e.g., `i=>ii`) are the primary motivating case -- without inlining, they waste storage on mostly-zero elements.

Tracked by: https://github.com/ahrefs/ocannl/issues/133

## Acceptance Criteria

- `check_idcs` in `check_and_store_virtual` accepts multi-symbol `Affine` indices (removing the `Non_virtual 51` restriction), collecting all non-static symbols from across the affine term list rather than requiring exactly one per position.
- `check_idcs` accepts repeated symbols across index positions (removing the `Non_virtual 5` restriction), so that diagonal patterns like `[| Iterator s; Iterator s |]` are valid for virtualization.
- `make_subst` in `inline_computation` constructs the substitution environment for multi-symbol affine indices. When a definition-site index is `Affine { symbols = [(c1, s1); (c2, s2)]; offset }` and the call-site index is also an `Affine` with the same structure, the substitution maps each `s_i` to the corresponding call-site symbol (or solves the affine relationship).
- `make_subst` handles the diagonal case: when the same symbol `s` appears at multiple definition-site positions (e.g., positions 0 and 1 both have `Iterator s`), the call-site indices at those positions must be consistent -- if they are both `Iterator t`, the substitution `s -> Iterator t` is recorded once; if they differ, the inlining must insert a conditional guard (an `if pos0_idx = pos1_idx` branch) or the node is marked non-virtual.
- `Fixed_idx` and `Task_id` indices in setter positions are handled: `Fixed_idx i` in a definition-site index matches the same `Fixed_idx i` at the call site (no substitution needed, already works via the `equal_axis_index` fallthrough), and the same for `Sub_axis`.
- Existing single-symbol inlining tests pass without regression.
- New tests cover: (a) a diagonal tensor `i=>ii` being virtualized, (b) a partially diagonal tensor where some axes share a symbol, (c) a multi-symbol affine index (e.g., convolution-style `stride*i + j`).

## Context

### Two blocking checks

**Block 1 -- `Non_virtual 51` (line 491):** In `check_idcs`, each `Affine` index position is required to contribute at most one non-static symbol. Multi-symbol affine indices like `Affine { symbols = [(stride, i); (dilation, j)]; offset = -padding }` (convolution patterns) are rejected. The fix is to collect all symbols from all terms across the affine expression and add them to the `syms` set.

**Block 2 -- `Non_virtual 5` (line 499):** After collecting symbols, `check_idcs` verifies that the count of `Iterator` positions with non-static symbols equals the set size. This is a uniqueness check: each symbol must appear at exactly one `Iterator` position. For diagonal tensors (`i=>ii`), the same symbol `s` appears at two `Iterator` positions, so `num_syms=2` but `Set.length syms=1`. The fix is to remove or relax this uniqueness assertion -- the key invariant to preserve is that all accesses to the tensor use *structurally identical* index arrays (enforced by `Non_virtual 4`).

### Substitution changes in `make_subst`

Currently `make_subst` (line 632) handles only `Iterator lhs_s` at each position -- one symbol per position, one substitution entry per position. For non-linear cases:

1. **Multi-symbol affine:** When `lhs_ind` is `Affine { symbols = [(c1,s1); (c2,s2)]; offset }` and `rhs_ind` is `Affine { symbols = [(c1,t1); (c2,t2)]; offset' }`, the substitution should map `s1 -> Iterator t1` and `s2 -> Iterator t2` (assuming matching coefficients). If the call-site structure differs, solving is needed or the node stays non-virtual.

2. **Repeated symbol (diagonal):** When position 0 has `Iterator s` and position 1 also has `Iterator s`, `make_subst` at position 0 produces `(s, rhs0)` and at position 1 produces `(s, rhs1)`. If `rhs0 = rhs1`, the substitution is consistent. If `rhs0 != rhs1`, the inlining must wrap the computation in a guard `if rhs0 = rhs1 then <computation> else 0.0` -- this is the "partial inlining" case described in the issue.

### Key code locations

- **Validation:** `arrayjit/lib/low_level.ml`, `check_and_store_virtual` / `check_idcs` (lines 457-500)
- **Inlining:** `arrayjit/lib/low_level.ml`, `inline_computation` / `make_subst` (lines 623-710)
- **Substitution engine:** `arrayjit/lib/low_level.ml`, `subst` function (lines 655-686) -- already handles nested `Affine` expansion correctly
- **Index types:** `arrayjit/lib/indexing.ml`, `axis_index` type (line 104)
- **Diagonal tensor test patterns:** `test/einsum/test_surjectivity.ml`, `test/einsum/test_accumulation_semantics.ml`
- **Documentation:** `docs/lowering_and_inlining.md` (lines 282-297 describe the current limitation)

### Approach sketch

1. **Relax `check_idcs`:** Replace the `Non_virtual 51` branch with symbol collection from all affine terms. Remove the `Non_virtual 5` uniqueness check (the structural equality check at `Non_virtual 4` is sufficient to ensure consistency across accesses).

2. **Extend `make_subst` for multi-symbol affine:** When both LHS and RHS are `Affine` with matching coefficient structure, produce one substitution entry per symbol pair. When structures don't match, raise `Non_virtual 13`.

3. **Extend `make_subst` for repeated symbols:** Return a list of `(symbol, axis_index)` pairs per position. After collecting all pairs, check consistency: if the same symbol maps to different call-site indices, either (a) insert an equality guard in the inlined code (`if idx_a = idx_b then ... else 0.0`), or (b) reject as non-virtual. Option (a) is the full solution for partial diagonals; option (b) is a conservative first step that still handles the case where the caller also uses the same symbol at both positions.

4. **Add guard insertion to `inline_computation`:** For the diagonal case with inconsistent call-site indices, wrap the inlined `Set_local` in a conditional. This requires adding a `Cond`-like construct at the LLC scalar level, or using the existing `Binop` with appropriate semantics.

5. **Tests:** Add einsum-based tests for `i=>ii` (diagonal), `ij=>iji` (partial diagonal), and a multi-symbol affine case.

### Risk notes

- The `subst` function (line 655) already handles nested `Affine` expansion, so the substitution engine itself needs no changes -- only `make_subst` (which builds the environment) and `check_idcs` (which gates entry) need modification.
- The guard-insertion path for inconsistent diagonal indices is the most complex part. A conservative first implementation could restrict to cases where call-site indices are consistent (same symbol at both positions), deferring guard insertion to a follow-up.
- Interaction with gh-ocannl-134 (multiple virtual tensors sharing loops): orthogonal but should be tested together once both land.

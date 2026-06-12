# Enable Inlining for Virtual Nodes with Non-Linear Index Symbols

## Status update (2026-06-12)

- Issue #133 is still **OPEN**. Its GH milestone is v0.7 (due 2026-01-30, now past); ROADMAP.md — the milestone authority — does not list #133 explicitly, so it has effectively slipped past the v0.7.x window (which closed mid-April 2026).
- **The core feature has not landed**: `check_idcs` still raises `Non_virtual 51` for multi-symbol affine indices and `Non_virtual 5` for repeated symbols; `make_subst` still handles only single `Iterator` per position (raises `Non_virtual 13` otherwise).
- `arrayjit/lib/low_level.ml` was heavily reworked by the cross-statement CSE landing (#351; commits e48ec84f, cdc14726, 64685b24), so all cited line numbers drifted — updated in place below. Structural descriptions remain accurate.
- The code now carries explicit `TODO(#133)` comments at both blocking sites in `check_idcs`, and the affine branch already collects non-static symbols from affine terms, accepting the single-symbol case — only the multi-symbol case still raises `Non_virtual 51`.
- New since this proposal: a `Concat` variant was added to `Indexing.axis_index` (tensor stacking, commit 58bfd6e5); `check_idcs` rejects it with `Non_virtual 52` (concat indices must be eliminated before virtualization). Any relaxation work must keep that rejection. `Task_id` no longer exists as an `axis_index` variant.
- The optimization pipeline gained `eliminate_common_subexpressions` + `hoist_cross_statement_cse` stages after `simplify_llc`, and the computations table is now persisted in an `optimize_ctx` passed into `virtual_llc` / `inline_computation`.
- `docs/lowering_and_inlining.md` section "Affine Index Limitations (Issue #133)" (still ~lines 282-297) remains the documented statement of this limitation.
- All acceptance criteria remain to do.
- Decision (Łukasz, 2026-06-12): pursue stage B (multi-symbol affine) rather than closing on stage A + documented limitation — see "Stage B plan (2026-06-12)" below for the injectivity criterion, the substitution construction, and the impossibility findings.

## Goal

Allow virtual node (inlining) optimization to handle tensors whose setter indices involve non-linear symbol usage: (1) multiple non-static symbols in a single `Affine` index position, and (2) the same symbol appearing at multiple index positions (diagonal tensors). Currently, these patterns trigger `Non_virtual 51` and `Non_virtual 5` respectively, forcing materialization. Diagonal tensors created via einsum (e.g., `i=>ii`) are the primary motivating case -- without inlining, they waste storage on mostly-zero elements.

Tracked by: https://github.com/ahrefs/ocannl/issues/133

## Acceptance Criteria

- `check_idcs` in `check_and_store_virtual` accepts multi-symbol `Affine` indices (removing the `Non_virtual 51` restriction), collecting all non-static symbols from across the affine term list rather than requiring exactly one per position.
- `check_idcs` accepts repeated symbols across index positions (removing the `Non_virtual 5` restriction), so that diagonal patterns like `[| Iterator s; Iterator s |]` are valid for virtualization.
- `make_subst` in `inline_computation` constructs the substitution environment for multi-symbol affine indices. When a definition-site index is `Affine { symbols = [(c1, s1); (c2, s2)]; offset }` and the call-site index is also an `Affine` with the same structure, the substitution maps each `s_i` to the corresponding call-site symbol (or solves the affine relationship). *(Update 2026-06-12: positional symbol mapping is only sound when the affine map is injective over the setter's loop ranges (mixed-radix/strided patterns, e.g. `4*s1 + s2` with `s2 < 4`). Otherwise multiple iterations write the same cell — and the lowering emits accumulation-with-init for non-injective LHS (`assignments.ml` lines ~417-423, `Indexing.is_injective` treats any multi-symbol affine LHS as non-injective) — so substitution recovers only one of several contributing writes. An injectivity check using `For_loop` bounds must gate this criterion.)*
- `make_subst` handles the diagonal case: when the same symbol `s` appears at multiple definition-site positions (e.g., positions 0 and 1 both have `Iterator s`), the call-site indices at those positions must be consistent -- if they are both `Iterator t`, the substitution `s -> Iterator t` is recorded once; if they differ, the inlining must insert a conditional guard (an `if pos0_idx = pos1_idx` branch) or the node is marked non-virtual.
- `Fixed_idx` indices in setter positions are handled: `Fixed_idx i` in a definition-site index matches the same `Fixed_idx i` at the call site (no substitution needed, already works via the `equal_axis_index` fallthrough), and the same for `Sub_axis`. *(Update 2026-06-12: the originally-mentioned `Task_id` variant no longer exists; `Concat` indices are rejected with `Non_virtual 52` and stay out of scope here.)*
- Existing single-symbol inlining tests pass without regression.
- New tests cover: (a) a diagonal tensor `i=>ii` being virtualized, (b) a partially diagonal tensor where some axes share a symbol, (c) a multi-symbol affine index (e.g., convolution-style `stride*i + j`).

## Context

### Two blocking checks

**Block 1 -- `Non_virtual 51` (line 494 as of 2026-06-12):** In `check_idcs`, each `Affine` index position is required to contribute at most one non-static symbol. Multi-symbol affine indices like `Affine { symbols = [(stride, i); (dilation, j)]; offset = -padding }` (convolution patterns) are rejected. The fix is to collect all symbols from all terms across the affine expression and add them to the `syms` set. *(Update 2026-06-12: the affine branch already filters non-static symbols out of the term list and accepts the single-symbol case; only lists of 2+ non-static symbols still raise `Non_virtual 51`.)*

**Block 2 -- `Non_virtual 5` (line 502 as of 2026-06-12):** After collecting symbols, `check_idcs` verifies that the count of `Iterator` positions with non-static symbols equals the set size. This is a uniqueness check: each symbol must appear at exactly one `Iterator` position. For diagonal tensors (`i=>ii`), the same symbol `s` appears at two `Iterator` positions, so `num_syms=2` but `Set.length syms=1`. The fix is to remove or relax this uniqueness assertion -- the key invariant to preserve is that all accesses to the tensor use *structurally identical* index arrays (enforced by `Non_virtual 4`). *(Update 2026-06-12: note that `num_syms` counts only `Iterator` positions, so check 5 as written also rejects setters whose only non-static symbols sit in `Affine` positions (syms non-empty, num_syms=0) — single-symbol affine setters do NOT currently virtualize despite passing check 51. The relaxation must redefine symbol coverage across both `Iterator` and `Affine` positions, not merely delete the assertion.)*

### Substitution changes in `make_subst`

Currently `make_subst` (line 636 as of 2026-06-12) handles only `Iterator lhs_s` at each position -- one symbol per position, one substitution entry per position. For non-linear cases:

1. **Multi-symbol affine:** When `lhs_ind` is `Affine { symbols = [(c1,s1); (c2,s2)]; offset }` and `rhs_ind` is `Affine { symbols = [(c1,t1); (c2,t2)]; offset' }`, the substitution should map `s1 -> Iterator t1` and `s2 -> Iterator t2` (assuming matching coefficients). If the call-site structure differs, solving is needed or the node stays non-virtual.

2. **Repeated symbol (diagonal):** When position 0 has `Iterator s` and position 1 also has `Iterator s`, `make_subst` at position 0 produces `(s, rhs0)` and at position 1 produces `(s, rhs1)`. If `rhs0 = rhs1`, the substitution is consistent. If `rhs0 != rhs1`, the inlining must wrap the computation in a guard `if rhs0 = rhs1 then <computation> else 0.0` -- this is the "partial inlining" case described in the issue.

### Key code locations

*(Line numbers below updated 2026-06-12 after the CSE rework of low_level.ml.)*

- **Validation:** `arrayjit/lib/low_level.ml`, `check_and_store_virtual` (line 460) / `check_idcs` (lines 469-502)
- **Inlining:** `arrayjit/lib/low_level.ml`, `inline_computation` / `make_subst` (lines 627-730)
- **Substitution engine:** `arrayjit/lib/low_level.ml`, `subst` function (lines 659-690) -- already handles nested `Affine` expansion correctly (including `Concat` rejection)
- **Index types:** `arrayjit/lib/indexing.ml`, `axis_index` type (line 104; variants are now `Fixed_idx`, `Iterator`, `Affine`, `Sub_axis`, `Concat` — there is no `Task_id`)
- **Diagonal tensor test patterns:** `test/einsum/test_surjectivity.ml`, `test/einsum/test_accumulation_semantics.ml`
- **Documentation:** `docs/lowering_and_inlining.md` (lines 282-297 describe the current limitation)

### Approach sketch

1. **Relax `check_idcs`:** Replace the `Non_virtual 51` branch with symbol collection from all affine terms. Remove the `Non_virtual 5` uniqueness check (the structural equality check at `Non_virtual 4` is sufficient to ensure consistency across accesses).

2. **Extend `make_subst` for multi-symbol affine:** When both LHS and RHS are `Affine` with matching coefficient structure, produce one substitution entry per symbol pair. When structures don't match, raise `Non_virtual 13`.

3. **Extend `make_subst` for repeated symbols:** Return a list of `(symbol, axis_index)` pairs per position. After collecting all pairs, check consistency: if the same symbol maps to different call-site indices, either (a) insert an equality guard in the inlined code (`if idx_a = idx_b then ... else 0.0`), or (b) reject as non-virtual. Option (a) is the full solution for partial diagonals; option (b) is a conservative first step that still handles the case where the caller also uses the same symbol at both positions.

4. **Add guard insertion to `inline_computation`:** For the diagonal case with inconsistent call-site indices, wrap the inlined `Set_local` in a conditional. This requires adding a `Cond`-like construct at the LLC scalar level, or using the existing `Binop` with appropriate semantics. *(Update 2026-06-12: no new construct is needed — `Ops.Where` (ternop), `Ops.Cmpeq` (binop), and `Embed_index` already exist, so the guard is `Set_local (id, Ternop (Where, Binop (Cmpeq, Embed_index rhs0, Embed_index rhs1), <inlined>, Get_local id))`, with the `Set_local (id, Constant 0.0)` from the lowering's zero-init entry providing the off-diagonal value.)*

5. **Tests:** Add einsum-based tests for `i=>ii` (diagonal), `ij=>iji` (partial diagonal), and a multi-symbol affine case.

### Risk notes

- The `subst` function (line 659) already handles nested `Affine` expansion, so the substitution engine itself needs no changes -- only `make_subst` (which builds the environment) and `check_idcs` (which gates entry) need modification.
- The guard-insertion path for inconsistent diagonal indices is the most complex part. A conservative first implementation could restrict to cases where call-site indices are consistent (same symbol at both positions), deferring guard insertion to a follow-up.
- Interaction with gh-ocannl-134 (multiple virtual tensors sharing loops): orthogonal but should be tested together once both land.

## Design review (2026-06-12)

**Verdict: sound-with-changes** — the diagonal half is well-conceived and tractable; the multi-symbol-affine half understates a soundness requirement (injectivity) and should be split out or deferred.

**Recommendations:**

1. **Split the proposal into two stages and re-scope the acceptance criteria accordingly.** Stage A (diagonal / repeated symbols, with `Where` guards) is the motivating case from the issue ("diagonal and partially diagonal tensors") and is implementable now. Stage B (multi-symbol affine) is only sound when the affine map is injective over the setter's loop ranges; `Indexing.is_injective` already conservatively classifies every multi-symbol affine LHS as non-injective, so the lowering emits accumulation-with-init for exactly these setters — substitution-based inlining would silently drop all but one contributing write. Stage B needs a mixed-radix injectivity check over `For_loop` bounds (e.g. sorted `|c_k| >= 1 + Σ_{j>k} |c_j|·(range_j − 1)`) threaded into `check_idcs`/`make_subst` before it can land. The current AC bullet "or solves the affine relationship" hides this entire sub-project.
2. **Do guard insertion in stage A, not as a follow-up.** Without guards, the only inlinable diagonal consumers are those reading with the *same* symbol at both positions (`ii=>i`-style traces) — a narrow case; generic consumers (`d * x`) read `d[j,k]` with distinct symbols. The machinery is cheap: substitute `s -> rhs.(first position)`, emit one `Where(Cmpeq(Embed_index rhs_i, Embed_index rhs_0), …)` per additional position, and add a `simplify_llc` rule folding `Where(Cmpeq(a, a), t, e) → t` for syntactically equal `Embed_index` args so the consistent case costs nothing. The conservative-first plan in the Risk notes delivers too little value to justify a separate landing.
3. **Rework `make_subst`'s env construction before relaxing check 5.** `Map.of_alist_exn` (line ~655) raises on duplicate keys, so a diagonal def-site produces an uncaught exception (not `Non_virtual`) the moment check 5 is removed. Collect `(symbol, axis_index)` pairs, group by symbol, take the first binding, and record guard conditions for the rest.
4. **Relax check 5, don't delete it.** As annotated above, `num_syms` counts only `Iterator` positions; the correct relaxation counts coverage across `Iterator` and `Affine` positions and permits repeats, while still rejecting (for stage A) any position whose affine term has 2+ non-static symbols. Wholesale deletion shifts all rejection to inline time (`Non_virtual 13`), which works mechanically (late `Never_virtual` is handled by `cleanup_virtual_llc`'s `Local_scope` fallback) but loses the early-out and makes failures harder to attribute.
5. **Add a zero-init acceptance criterion.** Off-diagonal reads are only non-`Recurrent` in `visit_llc` because the non-surjective lowering emits zero-initialization (`zeroed_out`); otherwise the node is forced `Materialized 36` before `check_idcs` is ever consulted. A test should pin this: diagonal inlining must keep working with `is_total`-style init elision logic in `assignments.ml` evolving.

**CSE interaction:** benign. Guarded inlined bodies are ordinary `Local_scope`s; `cse_equal_scalar` alpha-equivalence covers them, `Embed_index` contributes no reads to `hoist_shared_locals`' hazard analysis. No changes needed to the new passes.

**Pairing with #134 (landing order):** land after #134, as a separate change. The two touch disjoint machinery (here: `check_idcs`/`make_subst`; #134: `visit_llc`/`virtual_llc`/`cleanup_virtual_llc`), but #134 removes the `is_complex` symbol-sharing marking and thus changes which tensors reach the code paths modified here — this proposal's tests are only stable once #134 settles. See #134's review for the joint regression test.

**Open decision points for Łukasz:**

- Is stage B (multi-symbol affine) worth pursuing at all, given that injective-affine setters (unflatten/reshape-style scatter) seem rare in practice and the analysis cost is real? Closing #133 on stage A + documented limitation may be the better trade.
- Guard representation: `Where`-wrapped `Set_local` (scalar-level, no IR change, recommended) vs. a real conditional statement in `t` (more general, helps future partial-inlining work, more backend surface).
- When call-site indices at repeated positions are both static/`Fixed_idx` and unequal, should inlining constant-fold to the zero-init value instead of emitting a dead guard (nice-to-have; `simplify_llc` could also handle it)?

## Stage B plan (2026-06-12)

**Decision (Łukasz, 2026-06-12): pursue stage B — "let's find out what's impossible."** Stage A (diagonal/repeated symbols, with `Where` guards as per review recommendation 2) still lands first; stage B reuses its guard machinery and lands after it, in the sub-stages below. This section turns the review's "unsound as proposed" verdict into a concrete soundness criterion, a construction for the sound cases, and a record of what is genuinely impossible versus merely expensive.

### B0 — A real injectivity test in `indexing.ml`

`Indexing.is_injective` (indexing.ml line 253) currently classifies *every* `Affine` position with 2+ product-iterator symbols as non-injective (line 276, "If more than one product iterator in this Affine index, not injective"). The loop extents needed to do better are already at hand: `projections.product_space : int list array` is paired index-for-index with `product_iterators`, exactly the way `is_surjective` (line 222) already builds its `symbol_dims` map. Nothing about the timing changes — projections are fully solved when `is_injective` runs.

**Per-position criterion.** Consider one index position `idx = Σ_{i=1..m} c_i·s_i + off` where each `s_i` iterates a range of width `n_i`. Normalize first: coalesce repeated symbols by summing coefficients (the projection derivation in row.ml lines 4164–4180 already emits this canonical form — coalesced, zero-coefficients dropped, sorted — but `subst`'s nested-affine expansion does not re-coalesce, so normalize defensively), drop zero-coefficient terms, and treat any symbol with `n_i = 1` as static. Sort the remaining terms by `|c_i|` ascending. The position is **injective on its own symbols** if:

    for every k ≥ 2:   |c_k| ≥ 1 + Σ_{i<k} |c_i| · (n_i − 1)

*Proof sketch:* injectivity over the box fails iff some nonzero integer vector `d` with `|d_i| ≤ n_i − 1` satisfies `Σ c_i·d_i = 0`. Suppose such a `d` exists; let `k` be the largest index with `d_k ≠ 0`. Then `|c_k| ≤ |c_k·d_k| = |Σ_{i<k} c_i·d_i| ≤ Σ_{i<k} |c_i|·(n_i − 1) ≤ |c_k| − 1`, a contradiction. (For `k = 1`, `c_1·d_1 ≠ 0` directly since `c_1 ≠ 0` after normalization.) Absolute values make the condition sign-correct; offsets are irrelevant since only differences matter.

This is the mixed-radix condition: it holds with equality-tightness for exact radix systems (`idx = n_2·s_1 + s_2`, `s_2 ∈ [0, n_2)`), for strided pooling scatter with stride ≥ window (`2·oh + wh`, `wh`-range 2), and for conv patterns `stride·o + dilation·k` whenever `dilation·(K−1) < stride` covers the window. It is **sufficient, not necessary**: `idx = 3·a + 4·b` with `a`-range 3, `b`-range 2 has image `{0,3,6,4,7,10}` — injective — yet fails the test (`4 < 1 + 3·2`). The exact test is a small bounded integer-feasibility problem (does a nonzero `d` with `|d_i| ≤ n_i−1` zero the sum — solvable by DP over the bounded sum range) and can be slotted in later behind the same interface; the projection deriver only generates stride/dilation radix forms (`Conv_input` flattening in row.ml), so the mixed-radix test covers everything that occurs in practice.

**Whole-LHS criterion (pinning fixpoint).** The multi-axis map is injective if every non-static, range>1 symbol occurring in `project_lhs` is *pinned* by some position. Compute as a fixpoint: start with `pinned` = static ∪ range-1 symbols; repeatedly, for each position, restrict its affine form to unpinned symbols (pinned terms act as offset) and if the residual passes the per-position test, add its symbols to `pinned`. Accept iff all symbols end up pinned. `Iterator s` positions are the trivial `m=1, c=1` case; positions that never pin anything (e.g. a second occurrence of an already-pinned symbol, or a non-injective combination) impose consistency constraints only — they are automatically satisfied at the definition site and become `Where` guards at inline time (this is exactly stage A's diagonal guard, generalized). The fixpoint accepts triangular cases like `(s1, s1+s2)` that no single-pass per-position check accepts, and its pinning order is precisely the solving order `make_subst` will use in B1.

**Independent payoff.** Upgrading `is_injective` pays for itself before any inlining work: `can_skip_accumulation` (assignments.ml line 101) becomes true for genuinely injective scatters, so the lowering emits a plain `=` setter instead of neutral-init + read-modify-write. Example: max_pool2d backward with stride = window — input-grad index `2·oh + wh` is injective and surjective, so both the init pass and the `+=` self-read disappear. This is also a **soundness prerequisite** for B1: substitution-based inlining must only ever encounter non-accumulating setters, and after B0 the lowering guarantees exactly that for the setters B1 admits.

### B1 — Substitution inlining for injective affine setters

**Why injectivity is the precise soundness line.** `inline_computation` drops every `For_loop` whose index is bound in the substitution env (line 699) and keeps the rest. Reduction loops — symbols absent from the LHS — are kept and replay correctly today (an accumulating reduction `r[i] += x[i,j]` already inlines soundly: the `j` loop is preserved, `Zero_out` becomes `Set_local (id, 0.0)`, and the self-`Get` becomes `Get_local id`). The invariant required of the *dropped* loops is: for fixed values of the bound symbols, the definition writes the requested cell in exactly one iteration of the dropped loops. That is injectivity of the LHS map restricted to the bound symbols. Non-injective maps make the lowering emit accumulation over the dropped symbols, and a single env instance replays exactly one summand — which is the review's unsoundness, now stated as the boundary rather than a blanket rejection.

**`check_idcs` changes (low_level.ml line 469).** Loop extents are available here too: `check_and_store_virtual`'s traversal sees `For_loop { from_; to_; _ }` with static ints (the lowering emits `from_ = 0, to_ = d−1`; padded setters fold the left-pad into the `Affine` offset, not the bounds). Upgrade `env_dom` from `Set` of symbols to a `Map` from symbol to range width `to_ − from_ + 1`, and pass it to `check_idcs`. Then:

- Replace the `Non_virtual 51` raise: a multi-symbol affine position is accepted iff the whole-setter pinning fixpoint succeeds with the mapped widths. Keep raising 51 (or a new provenance code, for attribution) when it fails — this is the early-out the review's recommendation 4 asks to preserve.
- Relax check 5 into the coverage check from the fixpoint, counting symbol coverage across `Iterator` *and* `Affine` positions (fixing the annotated defect where single-symbol affine setters pass 51 but die on 5), while permitting repeated symbols (stage A).
- `Concat` stays rejected (`Non_virtual 52`); getters of `top_tn` need no injectivity (consumer-side multi-symbol affine reads are the already-working direction) but check 4's structural-equality requirement keeps def/use idcs within one computation aligned as before.

A check-time acceptance does not guarantee every consumer can be inlined — `make_subst` may still fail per call site below — and that is mechanically fine: a late `Non_virtual 13` lands in `cleanup_virtual_llc`'s `Local_scope` fallback as today. The gap should be small in practice; provenance codes should distinguish the two failure layers.

**`make_subst` construction (low_level.ml line 636).** Per review recommendation 3, first rework env construction to collect `(symbol, axis_index)` pairs, group by symbol, bind the first, and turn the rest into guard conditions. Then per definition-site position with call arg `rhs_ind`, in pinning-fixpoint order:

1. `Iterator s` — existing behavior, plus stage A's repeated-symbol grouping and guards.
2. **Structural match:** lhs `Affine { symbols = [(c_1,s_1); …]; offset }`, rhs `Affine` with the *same* canonical coefficient list and *equal* offset → bind `s_i → Iterator t_i` pairwise. Sound because injectivity makes the decomposition of any image point unique, and shape inference equates the dims on both sides so ranges agree. Offset mismatches (e.g. differing padding adjustments) are not solved — fall through.
3. **Unit-coefficient solving** (no IR change): if after substituting already-pinned symbols the residual of the position has exactly one unbound symbol and its coefficient is ±1 — e.g. lhs `stride·oh + kh` read at `Iterator t` with `oh` handled elsewhere or kept (see B2) — bind `kh → Affine { symbols = (1, t) :: [(−stride, oh')]; offset = −off }`. The existing `subst` already supports symbol→`Affine` bindings including nested expansion (lines 662–689). A **range-validity guard** is mandatory: `Where (Binop (Cmplt, Embed_index kh', Embed_index (Fixed_idx K)) /\ kh' ≥ 0, …)` with `K` taken from the dropped `For_loop`'s bounds at the drop site (line 699–705 has them in hand).
4. Otherwise → `Non_virtual 13`.

Consistency guards for non-pinning positions: `Where (Cmpeq (Embed_index (subst env lhs_ind_p), Embed_index rhs_ind_p), body, Get_local id)` — stage A's guard is the degenerate case. Pair with the `simplify_llc` folding rule from review recommendation 2 so syntactically-equal cases cost nothing.

**Zero-init interplay** (extends review recommendation 5): injective ∧ surjective setters get no init from the lowering, so whenever inlining introduces any guard (range or consistency), prepend an explicit `Set_local (id, Constant 0.0)` — do not rely on `c_syntax.ml`'s implicit local zero-init, which TODO(#340) intends to elide. Injective ∧ non-surjective setters (gaps, e.g. stride > window) keep their `Zero_out` (the `Fetch (Constant 0.0)` path, assignments.ml line 719), which `inline_computation` already turns into `Set_local (id, 0.0)`; gap-cell reads then correctly yield 0 through failed guards. Non-zero neutral elements (e.g. max-reduce scatter, neutral −∞) lower their init via `loop_over_dims` with plain-iterator idcs, which fails check 4 against the affine setter and stays non-virtual automatically — document, don't fix.

### B2 — Non-injective setters and mismatched structures: the fallback ladder

1. **Substitution (loop-dropping) inlining of a non-injective setter is impossible in principle**, not merely unimplemented: the stored cell value is a sum (or other fold) over the fiber of the index map, and one substitution instance reconstructs exactly one fold element. No choice of bindings recovers the rest — the information is only available by re-iterating the dropped loops.
2. **But inlining is not only substitution.** A fully general *guarded replay* exists in today's IR: preserve all setter loops (bind nothing), and guard the `Set_local` with `Cmpeq (Embed_index lhs_affine_p, Embed_index rhs_ind_p)` conjoined over positions. This replays the producer's whole loop nest filtered to the requested cell — accumulation included, since nothing is dropped and the local accumulates through `Get_local`/`Set_local`. `Local_scope` bodies admit `For_loop`s, and `Where`/`Cmpeq`/`Embed_index` all exist, so no IR change is needed. The cost is the product of the preserved loop ranges *per read* — the scatter→gather transposition with a linear scan standing in for index arithmetic.
3. **O(1) inversion requires an IR extension.** For an injective map in mixed-radix form, the inverse is a floor-div/mod chain (sorted by descending `|c|`: `s_1 = (t − off) / c_1`, `r = (t − off) mod c_1`, `s_2 = r / c_2`, … with divisibility and range guards; the dominance condition from B0 is exactly what makes greedy division correct, and gap cells of non-surjective maps fail the guards → 0). `axis_index` has no division, so this is **impossible without extending the index language** — e.g. `| Quot of axis_index * int | Rem of axis_index * int`. Ripple: derived `equal`/`compare`/`sexp`, `subst`, `simplify_llc` index rules, escape analysis in `check_and_store_virtual`, and `c_syntax.ml` index rendering (`/`, `%`, guarded non-negative since C truncates toward zero; CUDA/Metal inherit through the functor). Deferred — recorded as the only path to cost-free generality, to be revisited if B1's structural-match plus unit-coefficient solving leaves important consumers uncovered.
4. **Plan of record for non-injective setters: keep `Non_virtual`** — now produced by a principled test instead of "any multi-symbol affine" — with a fresh provenance code so the reason is attributable. Guarded replay (2) is viable but usually net-negative (it multiplies producer work by consumer visit count); if a use case appears, gate it behind a `virtualize_settings` flag plus a heuristic (visit count ≤ 1 and small preserved-loop product, both already computable from `visit_llc` data and `For_loop` bounds).

### What is actually impossible (findings)

- **Impossible in principle:** loop-dropping substitution for non-injective setters (drops fold elements; the canonical case is conv input-grad with stride < window, e.g. stride 1 / kernel 3, where every cell receives up to 3 contributions).
- **Impossible in the current IR:** loop-free (O(1)) inlining of injective setters whose inverse needs division — i.e. all coefficients non-unit and call site not structurally matching. Needs the `Quot`/`Rem` extension; nothing else blocks it.
- **Possible at a cost:** everything else. Guarded replay handles even non-injective accumulation correctly at O(producer loop nest) per read; unit-coefficient solving covers the stride·o + k family at no extra read cost beyond a guard.
- **Upstream blocker for direct einsum tests:** LHS conv-specs in einsum (e.g. `"ij=>i+3*j"`; see the disabled `_test_stride_gap` in `test/einsum/test_surjectivity.ml`, FIXME(#354)) don't reach projections yet, so stage-B exercise paths come from conv/pool *backprop* projections (e.g. `lib/nn_blocks.ml` `max_pool2d`'s `stride*oh< + wh` spec, `conv2d`'s `stride*oh+kh`) until #354 lands.

### Acceptance criteria (stage B)

- **B0 unit tests** (new standalone test, default location `test/operations/` with `.expected` file): `Indexing.is_injective` returns true for `2·oh + wh` (wh-range 2), `K·i + k` (k-range ≤ K), `3·i + j` (j-range ≤ 3), and the triangular `(s1, s1+s2)` two-position case; false for `i + j` (both ranges > 1) and `stride·o + k` with k-range > stride. The documented-incomplete case `3·a + 4·b` (ranges 3, 2) may return false.
- **B0 lowering:** max_pool2d backward with stride = window emits a plain `=` input-grad scatter with no neutral-init pass — pin with an `.ll.expected` snapshot using the rule pattern of `inline_permuted_view-c_fwd.ll.expected` in `test/einsum/dune`, extending `test/einsum/test_max_pool2d.ml` with a backward pass.
- **B1 structural match:** an intermediate produced by an injective strided scatter and consumed through the same affine form virtualizes — `.ll.expected` shows no buffer for the intermediate, and outputs match a materialized-baseline run in the same executable (toggling `virtualize_settings` as `inline_permuted_view.ml`'s `inline_view` does).
- **B1 unit-coefficient solving:** a `2·oh + wh` setter consumed at plain `Iterator t` (the pool-backward chain shape) inlines with range guards; numeric equality against the materialized baseline on `sync_cc` and at least one GPU backend (`OCANNL_BACKEND=cuda` or `metal`).
- **Soundness pin for non-injective:** conv input-grad with stride 1 / kernel 3 (the `test/training/circles_conv.ml` shape) remains non-virtual under the relaxed checks — assert the memory mode/provenance in a test — and `circles_conv.ml` training results are unchanged.
- **No regressions:** existing single-symbol inlining tests and stage-A diagonal tests (`test/einsum/test_surjectivity.ml`, `test_accumulation_semantics.ml`) pass; the `Concat` rejection (`Non_virtual 52`) is preserved.

### Landing order

#134 → stage A (with guards) → B0 (separately landable, immediate lowering win) → B1 → B2 stays an investigation note unless B1's coverage proves insufficient on real conv/pool models; the `Quot`/`Rem` IR extension is the designated escape hatch, not part of the initial stage-B landing.

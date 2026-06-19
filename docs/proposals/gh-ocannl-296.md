# Write / Flesh-Out lowering_and_inlining.md and Audit low_level.ml

Tracked by: https://github.com/ahrefs/ocannl/issues/296

## Status update (2026-06-12)

- Issue #296 is still OPEN, milestone v0.7. Harness status: deferred. Neither the doc accuracy pass nor the audit has been done: all 5 `FIXME(#296)`/`TODO(#296)` markers remain, and `docs/lowering_and_inlining.md` (315 lines) still claims `inline_complex_computations` defaults to `false`.
- The gap has WIDENED since this proposal was written. `low_level.ml` grew from 1840 to 2074 lines, mainly from **cross-statement CSE with hoisting** (`hoist_cross_statement_cse`, commits e48ec84f, cdc14726, 64685b24): the simplification pipeline is now `simplify_llc` -> `eliminate_common_subexpressions` -> `hoist_cross_statement_cse` (`low_level.ml:1632`). The doc covers none of the CSE phases; this proposal's pipeline-diagram criterion predates the hoisting pass and should include it.
- Line numbers cited below have drifted; key current locations: FIXME(#296) at lines 880, 891, 897, 906; TODO(#296) at 935; `cse_equal_scalar` at 1228; `substitute_float`/`substitute_proc` at 976-1014; `hoist_cross_statement_cse` at 1586; `input_and_output_nodes` at 1598; `optimize_proc` at 1619; `optimize` wrapper at 1922-1927; `loop_over_dims` at 1929, `unroll_dims` at 1947, `loop_over_padding_region` at 1977. Exit code 52 raised at line 497, code 140 at line 714.
- #351 (CSE after inlining) is CLOSED as completed — the doc's "pending CSE implementation" framing (doc line 266, 300) is doubly stale. #133 and #134 remain OPEN (v0.7), so those limitation sections stay relevant.
- Repo-wide renames since April 2026 (broadcast-order LUB->GLB reversal, dimension "label" -> "basis", "invalid" -> "discardable") may also have touched doc prose; the accuracy pass should re-verify terminology, using post-reversal vocabulary.
- Verdict: the task remains to-do and has grown — add the two CSE phases (including hoisting and the alpha-equivalence comparator fix in 64685b24) to the documentation scope.

## Goal

Complete the documentation in `docs/lowering_and_inlining.md` to accurately reflect the current state of `low_level.ml`, and audit `low_level.ml` for correctness issues, dead code, stale FIXMEs, and documentation gaps. The doc is already substantial (316 lines) but has drifted from the code; the audit resolves the `FIXME(#296)` markers in `low_level.ml` and captures any findings as follow-up issues.

## Acceptance Criteria

### Documentation (lowering_and_inlining.md)

- [ ] **Accuracy pass**: Every code snippet and behavioral claim in the doc matches the current source. **The doc must contain no source line numbers** (resolved 2026-06-19, Q2) — reference functions/phases/constructors by name, not line. Line-number drift is the exact failure this issue exists to fix, so the accuracy pass *removes* the existing `line 471`-style references rather than refreshing them.
- [ ] **Missing sections added**:
  - `Set_from_vec` / vector operations: the doc currently omits `Set_from_vec` handling in tracing, virtualization, inlining, and cleanup.
  - `eliminate_common_subexpressions` (CSE pass): added in the pipeline after `simplify_llc`, not yet documented as a pipeline phase.
  - `hoist_cross_statement_cse` (cross-statement CSE with hoisting to common ancestor scope, line 1586): added after `eliminate_common_subexpressions`, not yet documented. *(Update 2026-06-12: new since this proposal was written.)*
  - `cse_equal_scalar`: alpha-equivalence comparison infrastructure (line 1228; soundness fix in commit 64685b24).
  - `substitute_float` / `substitute_proc`: scalar substitution helpers used in `simplify_llc` (lines 976-1014).
  - `loop_over_dims`, `unroll_dims`, `loop_over_padding_region`: utility functions for generating loop nests and padding-region iteration (lines 1929-2074).
  - `input_and_output_nodes`: how the optimizer determines which tensors are inputs vs outputs (lines 1598-1617).
  - `optimize` entry point vs `optimize_proc`: the `optimize` wrapper that hooks up pretty-printing callbacks (lines 1922-1927).
  - `Scope_id` module and `get_scope` UID generation.
  - `Get_merge_buffer` and merge-buffer single-buffer constraint.
  - `optimize_ctx` type and its role carrying `computations` across compilation calls.
  - `Declare_local` IR constructor *(Update 2026-06-12: added by the hoisting work; the doc's `type t` listing omits it entirely — it must be added to the type listing, not just mentioned in the hoisting section)*.
- [ ] **Pipeline diagram updated**: Current doc shows `visit_llc -> virtual_llc -> cleanup_virtual_llc -> simplify_llc`. Actual pipeline is `visit_llc -> virtual_llc -> cleanup_virtual_llc -> simplify_llc -> eliminate_common_subexpressions -> hoist_cross_statement_cse` *(Update 2026-06-12: the hoisting phase landed after this proposal was written; see `low_level.ml:1632`)*.
- [ ] **Settings section updated**: `inline_complex_computations` default changed from `false` to `true` in the current code (line 134). The doc still says "default: false".
- [ ] **Non_virtual exit codes updated**: Code 52 (Concat in check_idcs) and 140 (vec ops) are in the code but 52 is missing from the doc's list. *(Update 2026-06-12: code 19 — `Declare_local` encountered during `check_and_store_virtual`, `low_level.ml:549` — is also in the code and missing from both the doc's list and this proposal.)*
- [ ] **FIXME(#296) markers documented**: Each of the 5 `FIXME(#296)` sites (lines 880, 891, 897, 906, 935 in `cleanup_virtual_llc` as of 2026-06-12) should be explained in the doc -- they mark places where the cleanup phase forces Virtual memory mode with specific provenance codes (15, 151, 152) for tensors not yet proven non-virtual, and one TODO about asserting Never_virtual.

### Audit (low_level.ml)

- [ ] **Resolve FIXME(#296) markers**: The intended resolution is **explain and keep the behavior** (resolved 2026-06-19, Q4): the cleanup phase's policy of defaulting undecided tnodes to `Virtual` (provenance 15/151/152) is accepted as correct. For each of the 5 `FIXME(#296)` sites, replace the FIXME with an explanatory comment stating *why* cleanup defaults the undecided node to `Virtual` (and drops the loop where applicable). This is a document-and-retain pass, **not** an investigation into whether forcing `Virtual 15` on a not-yet-decided node could discard needed computation — that deeper question is explicitly out of scope. File a follow-up issue only if the audit incidentally surfaces a concrete bug.
- [ ] **TODO(#296) at line 935** *(was 928)*: Evaluate whether the `Get` case in `cleanup_virtual_llc.loop_scalar` can indeed assert `Never_virtual` instead of calling `update_memory_mode`. If safe, convert to assert; otherwise document why not.
- [ ] **Dead code check**: Identify any unreachable match arms or functions in `low_level.ml`.
- [ ] **Edge cases**: Verify behavior when `dims` is empty (relevant to `Set_from_vec` tracing at line 331), when `computations` list is empty at inline time, and when `check_and_store_virtual` encounters `Concat` indices.
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

- **`arrayjit/lib/low_level.ml`** (2074 lines as of 2026-06-12) -- IR types, optimization pipeline, pretty-printing
- **`arrayjit/lib/assignments.ml`** (989 lines) -- High-level assignment representation, `to_low_level` translation
- **`docs/lowering_and_inlining.md`** (315 lines) -- Existing documentation

### FIXME(#296) Sites in low_level.ml

All 5 markers are in `cleanup_virtual_llc` (starts at line 866; line numbers as of 2026-06-12):

1. **Line 880** (`For_loop` case): When a for-loop's symbol maps to a tnode in `reverse_node_map` that is not known non-virtual, forces `Virtual 15` and removes the loop. FIXME suggests this may be too aggressive.
2. **Line 891** (`Zero_out` case): Forces `Virtual 151` for non-virtual tnodes. Same concern.
3. **Line 897** (`Set` case): Forces `Virtual 152` for non-virtual tnodes. Same concern.
4. **Line 906** (`Set_from_vec` case): Forces `Virtual 152` for non-virtual tnodes. Same concern.
5. **Line 935** (`Get` in `loop_scalar`): Sets `Never_virtual 17`. TODO suggests this should already be `Never_virtual` by this point and could be an assert.

### Related Issues

- **#133**: Affine index virtualization (multiple non-static symbols) -- doc mentions this limitation
- **#134**: Virtual tensors sharing for-loops -- affects `reverse_node_map` shared-symbol tracking
- **#351**: CSE after inlining -- the CSE pass is now implemented (issue closed as completed) but doc doesn't reflect it; cross-statement CSE with hoisting also landed since (commits e48ec84f, cdc14726, 64685b24)

## Design review (2026-06-12)

**Verdict: sound-with-changes.** The two-track plan (doc accuracy pass + `low_level.ml` audit) matches the issue's intent, and the missing-sections inventory is accurate (verified against `low_level.ml` at 2074 lines: pipeline order `simplify_llc -> eliminate_common_subexpressions -> hoist_cross_statement_cse` at line 1632, `inline_complex_computations` default now `true` at line 134, all 5 FIXME/TODO(#296) markers present at lines 880/891/897/906/935). But the plan has a sequencing flaw and bakes in a staleness mechanism.

**Strongest recommendations:**

1. **Ban source line numbers from the doc; change the acceptance criterion accordingly.** The criterion "every line-number reference matches the current source" institutionalizes the exact drift that caused this issue: the doc's "line 471" (mentioned twice) already points at the wrong code (the affine-symbol restriction now sits at ~484–494). The accuracy pass should *remove* line references in favor of function/phase names, and the criterion should read "the doc contains no source line numbers". Otherwise this doc is stale again within a month — `low_level.ml` grew 234 lines in the last quarter alone.

2. **Sequence the audit before the documentation, not in parallel.** As written, one criterion says "FIXME(#296) markers documented" (explain each marker in the doc) while the audit says "resolve FIXME(#296) markers" (replace them with explanatory comments or file issues). Doing the doc first means writing prose about comments the audit then deletes. Do the audit first; then document the *post-audit semantics* of `cleanup_virtual_llc` (why cleanup defaults undecided tnodes to `Virtual` with provenance 15/151/152), not the markers.

3. **Give the dead-code audit a concrete starting target: the `Declare_local` arms.** `Declare_local` is produced only by `hoist_cross_statement_cse`, which runs *last* in the pipeline, and fresh lowering from `Assignments.to_low_level` never emits it. Yet `cleanup_virtual_llc` handles it (`Declare_local _ -> Some llc`, line 926) and `check_and_store_virtual` raises `Non_virtual 19` on it (line 549). Whether these arms are reachable (e.g. via `optimize_ctx.computations` carried across compilation calls — are stored computations captured pre- or post-hoisting?) is exactly the kind of question the audit should answer and the doc should then state.

4. **Decide doc altitude before writing: conceptual overview vs. source mirror.** The missing-sections list includes genuinely conceptual items (`hoist_cross_statement_cse`, `cse_equal_scalar` alpha-equivalence, `optimize_ctx`) but also plumbing (`loop_over_dims`, `unroll_dims`, `loop_over_padding_region`, the `optimize` wrapper's pretty-printing hookup). The latter belong in `low_level.mli` doc-comments, not in `lowering_and_inlining.md` — a doc that mirrors every utility function is a second copy of the source that will drift. Suggested split: pipeline phases, IR constructors (incl. `Declare_local`), memory-mode interaction, and settings go in the doc; loop-generation utilities get `.mli` comments and at most one sentence in the doc.

5. **Resolve the milestone contradiction.** The GitHub issue sits in milestone v0.7 (due 2026-01-30, overdue); `ROADMAP.md` schedules #296 under v1.0 (end of October 2026, "Documentation completeness"). These can't both be right. The natural split: the *audit* half (FIXME resolution — it guards correctness of the cleanup phase's default-to-Virtual policy) is v0.7-flavored work that shouldn't wait until October; the *doc* half fits v1.0. Either split the issue or retarget the milestone.

**Decision points for Łukasz:**

- Split #296 into audit-now (v0.7.x) + doc-later (v1.0), or keep combined and retarget the GitHub milestone to v1.0 to match ROADMAP?
- Accept the "no line numbers in the doc" rule (recommendation 1)?
- Doc altitude (recommendation 4): should `loop_over_dims`/`unroll_dims`/`loop_over_padding_region` be documented in the doc or only in `low_level.mli`?
- For the FIXME sites: is the intended resolution "explain and keep the behavior" (cleanup defaults undecided nodes to Virtual) or is there appetite to investigate whether forcing `Virtual 15` on a not-yet-decided node whose loop is then dropped can ever discard needed computation? The audit should state which question it is answering.

## Decisions resolved (2026-06-19, Łukasz)

All four decision points are now answered; `has_questions` cleared and the task moved to `ready`.

1. **Scope / milestone — keep combined, do it now.** #296 is specifically a doc-and-audit update *at the current stage to prepare for work on v0.8*; treat the whole task (doc accuracy pass + audit) as current-cycle work, **not** split into a v1.0 doc-later half. The "later doc drift" staleness worry behind recommendation 5 is moot — that drift has not happened yet, so it is not a reason to defer the doc half. Milestone stays as-is (v0.7); no GitHub retarget, no issue split.
2. **No line numbers in the doc — accepted.** The accuracy pass removes existing source line-number references in favor of function/phase/constructor names (recommendation 1; folded into the Accuracy-pass acceptance criterion above).
3. **Doc altitude — medium.** Explain the pipeline and utilities *algorithmically but at a conceptual level*: describe what each phase/utility does and how it works in algorithmic terms, without mirroring the source line-by-line. This sits between "conceptual overview only" and "source mirror" — `loop_over_dims`/`unroll_dims`/`loop_over_padding_region` get a conceptual algorithmic description in the doc (not relegated solely to `.mli` comments, not reproduced verbatim).
4. **FIXME sites — explain and keep the behavior.** Document *why* cleanup defaults undecided tnodes to `Virtual` (provenance 15/151/152) and retain that behavior; do not open the deeper "could this discard needed computation?" investigation (folded into the Resolve-FIXME acceptance criterion above).

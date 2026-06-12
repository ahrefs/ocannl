# Simply & NanoDO Deep Dive: Comparison and Implications for OCANNL's `lib/`

**Issue:** [ahrefs/ocannl#435](https://github.com/ahrefs/ocannl/issues/435)
**Status:** Draft proposal

## Status update (2026-06-12)

- Issue #435 is OPEN (label `explore`), milestone v1.0 (end of October 2026). No comments have been posted on the issue and `docs/explore/simply-comparison.md` does not exist — the deliverable has not been produced; the task has not started.
- The OCANNL comparison target has grown since elaboration: `lib/nn_blocks.ml` is now 579 lines (the "~440" figure below is stale) — additions include `batch_norm1d`/`batch_norm2d` (#59) and RoPE/sinusoidal position embeddings (#398, closed; PoPE deferred to #444). `lib/train.ml` is 253 lines.
- The OCANNL column gained substantial new evidence: `test/training/transformer_names.ml` (decoder-only autoregressive transformer on the Names dataset, #57 closed COMPLETED), `fsm_transformer.ml` (#116), and the makemore tutorials (`mlp_names.ml`, `mlp_bn_names.ml`). The *position encoding* and *normalization* axes of the eleven-axis table now have real OCANNL entries.
- `lib/persistence.ml`/`.mli` (tensor persistence, #373) is in place as anticipated below.
- Sister explore task #318 (megakernel deep dive) is closed COMPLETED; #242/#306/#316 remain complementary context.
- The precondition note still holds: local clones of Simply and NanoDO are absent from `/Users/lukstafi/simply/` and `/Users/lukstafi/nanodo/` — the implementing worker must re-clone both.
- Remaining work: all eight acceptance criteria (read Simply/NanoDO, build the table, write `docs/explore/simply-comparison.md`, post the GH comment).

## Goal

Produce a structured deep-dive comparison of two reference Python/JAX neural-network frameworks against OCANNL's user-facing library code, and post the findings as a comment on issue #435. The two references are:

- **Simply** ([google-deepmind/simply](https://github.com/google-deepmind/simply)) — Google DeepMind's "minimal-yet-production" LLM research framework. ~4000-line `model_lib.py` covers LayerNorm/RMSNorm, gated/MoE FFN, multi-head/GQA attention with RoPE/flash/sliding-window/KV-cache, full TransformerLM, plus a separate `rl_lib.py` (~1466 lines) for REINFORCE/GRPO. Built on a custom `SimplyModule` base class with explicit `init(prng_key) -> PyTree` / `apply(params, x)` separation.
- **NanoDO** ([google-deepmind/nanodo](https://github.com/google-deepmind/nanodo)) — a ~166-line decoder-only transformer using Flax `nn.Module`. Serves as the "minimum viable transformer" baseline.

The comparison target on the OCANNL side is `lib/` — `nn_blocks.ml` (~579 lines of building blocks as of 2026-06), `train.ml` (~253 lines of training utilities), `ocannl.ml` (re-exports), and (added since elaboration) `persistence.ml`. The deliverable is a write-up that informs OCANNL's `lib/` evolution toward LLM workloads, not code changes in this round.

This is exploratory research. No public API of OCANNL is touched.

## Acceptance Criteria

The artefact is a single GitHub issue comment posted on #435, backed by a write-up file kept in the repo at `docs/explore/simply-comparison.md` so the comment can link back to the durable record. Each criterion below names a falsifier — what a deficient deliverable would look like.

1. **The eleven-axis comparison table is present and complete.** The write-up contains a Markdown table with one row per axis from the elaboration's Tentative Design table and one column for each of `Simply`, `NanoDO`, `OCANNL`. The eleven axes are: *module pattern*, *parameters*, *einsum/linear primitive*, *shape inference*, *sharding*, *mixed precision*, *attention variants*, *FFN variants*, *normalization*, *position encoding*, *RL training*. Falsifier: a row missing or a cell left as "TBD" / unaddressed.

2. **Each comparison row cites concrete code pointers (symbol or file:section, not line numbers) for the cells that are non-trivially "present" on each side.** "Not implemented" is an acceptable cell value; "implemented" without a pointer is not. Falsifier: any non-"absent" cell whose claim cannot be traced to an identified function/module/dataclass/PPX construct.

3. **The Simply `SimplyModule` vs OCANNL PPX-DSL tradeoff is analysed in a dedicated section** that addresses, at minimum: (a) where parameters live (PyTree handed in vs `{ w }` auto-lifted in `%op`), (b) how composition works (dataclass `setup()` + `apply()` vs `%op` closures returning tensors), (c) shape handling (manual dims vs row-variable inference), (d) AI-agent readability (the design goal Simply explicitly cites). Falsifier: any of (a)–(d) absent, or stated as a one-sided claim without identifying the corresponding mechanism on the other side.

4. **A "Simply features absent from OCANNL `lib/`" list is produced and prioritised.** Each entry names the missing capability, links to the Simply file/symbol that implements it, and assigns one of three levels: `roadmap-aligned` (already in OCANNL's milestones — link the milestone or issue), `worth-considering` (no current issue but plausible), `out-of-scope` (intentionally not pursued — say why). Falsifier: any entry missing the link, the level, or — for `out-of-scope` — the rationale.

5. **NanoDO is contrasted as the "minimal transformer" baseline in its own subsection.** The contrast covers what NanoDO chooses to *omit* relative to Simply (e.g., RoPE, GQA, MoE) and what its 166-line Flax `nn.Module` style would look like translated into OCANNL `nn_blocks.ml` idiom — at the level of "this Flax submodule maps to this `%op` shape", not full code. Falsifier: NanoDO discussed only as a third column in the table, with no synthesis paragraph.

6. **The "Implications for OCANNL's `lib/` design" closing section names at least three distinct, actionable directions** the comparison surfaces — each phrased as a candidate follow-up task (one-sentence title + 2–3 sentence rationale + estimated effort tier `tiny`/`small`/`medium`/`large`/`huge`). Falsifier: a closing section that is purely descriptive ("Simply has more features"), or directions that lack the title/rationale/effort tuple, or fewer than three.

7. **The GitHub-issue-comment artefact exists on issue #435** and is the comment that closes the exploration thread. The comment must (a) link to the write-up file at its committed path on `main` of the staging fork (anchor link in the comment body), (b) include the eleven-axis table inline (so the comment is readable without clicking through), and (c) end with a bulleted summary of the directions from criterion 6. Falsifier: the comment is missing any of (a), (b), (c); or the comment merely says "see the file" without inlining the table.

8. **The write-up file is committed to `docs/explore/simply-comparison.md`** on the staging fork's main branch (or a feature branch that lands in `main` via PR before the comment is posted). Falsifier: the file referenced from the GH comment does not resolve at the cited path on `main`.

## Context

### OCANNL `lib/` (the comparison target on our side)

- `lib/nn_blocks.ml` — building blocks built on the PPX-based einsum DSL (`%op`, `%cd`). Key symbols: `box_muller`, `kaiming_impl`, `xavier_impl`, `normal`/`normal1`, `kaiming`/`xavier` (extended via `%%extend_dsls`), one-hot conversion, MLP, dropout, softmax, multi-head attention, layer norm, transformer encoder/decoder, conv2d, depthwise-separable conv, ResNet blocks, LeNet, VGG blocks. Design principles documented in the file header (principle of least commitment, single-char einsum mode, kernel-axis intent, inline-param auto-lifting). Files have grown since elaboration — recheck against current tree.
- `lib/train.ml` — SGD optimizer, forward/backprop orchestration, parameter initialization, context management, round-robin parallelism.
- `lib/persistence.ml`, `lib/persistence.mli` — added since elaboration (March 31). Worth a sentence in the write-up since it touches the "missing pieces" theme (loading/saving model state).
- `lib/ocannl.ml` — re-exports of `Tensor`, `Shape`, `Row`, `Operation`, `Train`, `Nn_blocks`.
- `tensor/operation.ml`, `tensor/tensor.ml`, `tensor/shape.ml` — the underlying DSL implementation that `nn_blocks.ml` consumes; relevant to the `EinsumLinear` vs `%op` discussion.

### Simply (the production-scale reference)

Local clone (per task elaboration) was at `/Users/lukstafi/simply/`. **Precondition note:** the clone is currently absent from that path; the implementing worker must re-clone from `https://github.com/google-deepmind/simply` before starting Phase 1. The implementing worker should record the commit SHA they read against in the write-up so the comparison is reproducible.

Files to read (per elaboration):
- `simply/utils/module.py` — `SimplyModule`, `EinsumLinear`, `EmbeddingLinear`. The DSL anchor.
- `simply/model_lib.py` — `LayerNorm` / `PerDimScale` (~lines 110–200), `Attention` (~line 1167), `FeedForward` and `MoEFeedForward` (~lines 474–700), `TransformerBlock` (~lines 1559–1750), `TransformerLM` (~lines 1852–2050). Line numbers are starting points; navigate by symbol since Simply's tree may have moved.
- `simply/config_lib.py` — config-driven model construction (Gemma2 etc.).
- `simply/rl_lib.py` — REINFORCE/GRPO; skim only, just enough to characterise the post-training surface.
- `simply/utils/optimizers.py`, `simply/utils/position_encoding.py`, `simply/utils/sharding.py` — optimizer impls, RoPE, sharding annotations.

### NanoDO (the minimal-transformer baseline)

Local clone (per task elaboration) was at `/Users/lukstafi/nanodo/` — also currently absent; re-clone from `https://github.com/google-deepmind/nanodo`. Files to read: `nanodo/model.py` (the 166-line model), `nanodo/train.py`, `nanodo/loss.py`, `nanodo/optimizer.py`.

### Why this matters

OCANNL's `nn_blocks.ml` follows a similar philosophy to Simply (composable building blocks, einsum-centric, no hidden module state) but at a much smaller scale. The comparison is valuable because:

1. Simply demonstrates how a minimal-abstraction framework scales to production LLM features (MoE, GQA, flash attention, KV cache) without losing readability.
2. The `SimplyModule` pattern (PyTree-explicit init/apply) is a different point in the design space from OCANNL's `Tensor.t`-with-embedded-params; the tradeoffs around AI-agent readability and config-driven construction are concrete and worth surfacing.
3. NanoDO is a yardstick: how much of its 166 lines does OCANNL's analog occupy? Where is OCANNL's overhead going?
4. RL post-training (Simply's `rl_lib.py`) is an area OCANNL hasn't addressed; the comparison surfaces the API shape OCANNL would need.
5. The exploration is sister to other "compare-and-learn" tasks (#242 TVM, #428 TorchLean, #306 Petalisp/Caten, #316 DumPy/torchdim, #318 megakernel) — those covered low-level compilation; this covers high-level API/abstraction.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

The elaboration already specifies a 4-phase plan (~5h total). Reproduce here only what changes given the proposal's discipline:

1. **Phase 1 (~2h) — Read Simply.** Anchor on symbols, not line numbers. Capture per-axis evidence (one note per row of the eleven-axis table) into the working write-up as you go, so cell content is justified by quotes/pointers from the start, not retrofitted.
2. **Phase 2 (~30min) — Read NanoDO.** Same disciple. End the phase by drafting the "Simply omissions in NanoDO" paragraph (criterion 5) — this synthesis becomes harder if deferred to write-up time.
3. **Phase 3 (~1.5h) — OCANNL re-read in light of Simply/NanoDO.** Walk `nn_blocks.ml`, `train.ml`, and `persistence.ml` symbol-by-symbol and fill in the OCANNL column of the table. Then build the "missing features" list (criterion 4) — for each candidate, run `gh issue list --search` (or grep this proposal's sibling tasks) to label it `roadmap-aligned`/`worth-considering`/`out-of-scope`.
4. **Phase 4 (~1h) — Write-up + GH comment.** Commit `docs/explore/simply-comparison.md` to the staging fork; open a PR if working on a branch (or commit to `main` directly per project convention). Then post the GH comment per criterion 7. Keep the comment self-contained: inline the table, link to the file.

The write-up file is the durable artefact; the GH comment is the announcement. Optimise the file for re-reading by future contributors, the comment for someone scanning the issue.

## Scope

**In scope:**
- All eight acceptance criteria above
- A committed write-up file at `docs/explore/simply-comparison.md` on the staging fork
- A GitHub issue comment on `ahrefs/ocannl#435`
- Re-cloning Simply and NanoDO if absent (the elaboration's local-clone paths are stale)

**Out of scope:**
- Implementing any of the "missing features" surfaced by criterion 4 — those become follow-up tasks (criterion 6 lists candidates, but only as titles + rationales)
- Benchmarking OCANNL against Simply or NanoDO
- Refactoring `nn_blocks.ml`/`train.ml`/`persistence.ml` based on findings
- Comparison with other Python/JAX frameworks (Flax core, Haiku, Equinox) — those are separate explore tasks if useful
- Closing issue #435 (the user closes it; this task only posts the comment)

**Dependencies:**
- No code dependencies. Sister explore tasks (#242, #316, #318, #306, #428) are complementary, not blocking.
- The implementing worker needs `gh` auth for posting the comment, which is the standard project setup.

# Proposal: Migrate slipshow presentations to v0.10.0 with Mermaid

## Status update (2026-06-12)

- Issue #425 is OPEN, milestone v0.7 (that milestone's due date has long passed; this is a small background task that simply hasn't been picked up).
- Nothing has landed: `.github/workflows/gh-pages-docs.yml:20` still downloads slipshow v0.6.0, and no Mermaid diagrams have been added to any deck.
- The proposal's "v0.10.0 (the latest release)" is stale: slipshow v0.11.0 ("Brazlip") was released 2026-05-24. The migration target should be v0.11.0 — re-verify the `slipshow-linux-x86_64.tar` asset name and the v0.11.0 changelog for additional warnings/breaking changes before implementing.
- The three slide decks are unchanged in length (871 / 592 / 565 lines), so the Mermaid candidate locations listed below remain valid; `dune-project:140` still lists `(slipshow :with-doc)`.
- The workflow now compiles all three decks (workflow lines 25, 28, 45) — all must compile cleanly under the new version.
- Remaining work: the entire proposal (version bump, warning cleanup, Mermaid diagrams, deployment check), retargeted at v0.11.0.

## Goal

Upgrade the slipshow version used in the GH Pages workflow from v0.6.0 to v0.10.0 *(Update 2026-06-12: now v0.11.0, the latest release)*, enabling Mermaid diagram support in slide decks, and add Mermaid diagrams where they improve clarity.

## Acceptance Criteria

- [ ] Update slipshow download URL in `.github/workflows/gh-pages-docs.yml` from `v0.6.0` to `v0.10.0`
- [ ] All three slide decks compile without errors under the new version:
  - `docs/slides-basics_backprop_training_codegen.md`
  - `docs/slides-RL-REINFORCE.md`
  - `docs/slides-shapes_and_einsum.md`
- [ ] Address any new compiler warnings introduced by v0.10.0 (it added compile-time warnings for missing IDs, action parse failures, etc.)
- [ ] Add Mermaid diagrams in at least two candidate locations (see Context below)
- [ ] GH Pages deployment succeeds with the new version

## Context

### Version gap: v0.6.0 to v0.10.0

Key changes across the upgrade path:

- **v0.8.x**: Base improvements (details sparse in release notes)
- **v0.9.0**: Added MermaidJS support (#205), syntax highlighting for all highlight.js languages, MathJax extensions, KaTeX support, carousel-fixed-size, removed confusing auto-generated IDs
- **v0.10.0**: Added compile-time warnings (missing ID, duplicated ID, action parse failures, unknown attributes, missing files, frontmatter parse errors). Fixed drawing and keyboard shortcut issues.

The tar asset naming convention (`slipshow-linux-x86_64.tar`) is unchanged, so the `--strip-components=1` extraction in the workflow should continue to work.

### Release asset verification

The v0.10.0 release publishes `slipshow-linux-x86_64.tar` (same filename pattern as v0.6.0), confirming the download URL only needs the version string updated.

### Mermaid diagram candidates

After reviewing all three slide decks, the following locations are strong candidates for Mermaid diagrams:

1. **Backprop computation flow** (basics slides, around line 463-470): The forward/backward pass explanation describes a computation chain `x -> y(x) -> f(y(x))` and the reverse gradient flow `df -> df/dy -> df/dx`. A Mermaid flowchart would visualize this clearly.

2. **RL agent-environment loop** (RL slides, around line 14-25): The RL framework lists Agent, Environment, Actions, States, Rewards as bullet points. A Mermaid diagram showing the cyclic interaction (Agent --action--> Environment --state,reward--> Agent) is a classic RL visualization.

3. **REINFORCE to GRPO evolution** (RL slides, around line 568-574): The numbered progression from REINFORCE through clipping, KL penalty, group baselines to GRPO could be a flowchart showing the additive enhancements.

4. **Three axis kinds** (shapes slides, around line 49-55): The batch | input -> output tensor layout could be visualized as a diagram showing how the three axis kinds relate in matrix operations.

### Risk: breaking changes

The removal of auto-generated IDs in v0.9.0 could affect slides that implicitly relied on them. The new compile-time warnings in v0.10.0 will flag any such issues, making them easy to find and fix.

### Code pointers

- `.github/workflows/gh-pages-docs.yml:20` -- slipshow download URL (change `v0.6.0` to `v0.10.0`)
- `dune-project:140` -- slipshow listed as doc dependency (version constraint may need update)
- `docs/slides-basics_backprop_training_codegen.md` -- 871 lines, main tutorial
- `docs/slides-RL-REINFORCE.md` -- 592 lines, RL tutorial
- `docs/slides-shapes_and_einsum.md` -- 565 lines, shapes/einsum tutorial

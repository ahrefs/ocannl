# Agent notes: distilled cross-session knowledge

Promoted from coding agents' local session memory so that fresh clones inherit the design history
and trap knowledge that is not derivable from the code alone. Scope discipline: machine-specific
facts (which backends are installed, local paths, remote-benchmarking setups) deliberately stay out
— each machine's agents keep those locally. When a note disagrees with the code, the interfaces, or
the primary docs, those win; treat entries as leads, and verify the named symbols still exist.
Workflow rules live in AGENTS.md; these files are subsystem lore.

Cross-subsystem review checklist: release, cleanup and decline code is accepted by fault-injected
error paths, with a negative control and exact ownership assertions; happy-path evidence is only
supporting evidence. Labels and reports must be derived from the artifacts actually emitted,
timed, shipped or released, never from the transform or branch that intended to create them.

**This file is an index.** The notes live in `docs/agent-notes/`, one file per subsystem, so that
looking one trap up costs one file rather than all of them. Read the line below that matches what
you are about to touch, then open that file. When promoting new knowledge, append to the matching
file (2–6 lines, with file pointers) rather than here — the index only needs a new line if a file
grows a topic its hooks do not already name. The structure below is checked, by
`agent_notes_structure` (gh-ocannl-691): a bullet ends in punctuation and its continuations sit two
spaces in, every row of the table is one physical line, every backticked hook occurs in the file its
row links, and every file under `docs/agent-notes/` is linked from exactly one row — so a new file
does need its row the day it appears. The dialect it reads is deliberately small: headings, prose
paragraphs, `- ` bullets one level deep, and the table above. Anything else — a fenced block, an
ordered or `*` item, a block quote whether or not its marker carries a space, an HTML comment, a
heading underlined instead of written with hashes — is REPORTED rather than guessed at, because a
construct the scan cannot read is text no rule checks. Two of those readings are Markdown's rather
than the obvious one, so write around them: a quote marker does not need its space, so a line whose
first visible column is `>=` is a quote however arithmetic it reads — rewrap the comparison to keep
its operator off the first column, or write it inside a code span; and a line of nothing but `-` or
`=` directly below a paragraph underlines it into a heading, however short the run. That is the contract to argue
with if you want one of them: teach the scan what its lines mean, and it becomes part of the
dialect.

| File | What it covers |
| --- | --- |
| [shape-inference.md](agent-notes/shape-inference.md) | The row/dim solver and its staging: einsum operators and specs, fixed-index axes, `Row.Concat` arithmetic, `Shape_row` finalization, deferral and livelock rules, safe-to-guess / `resolve_at_use`, symbolic extents, padded (`=`-mode) windows. |
| [syntax-extensions.md](agent-notes/syntax-extensions.md) | `%op` / `%cd` scoping: block-tensor delimiters, inline-record init expressions and generativity, comp splicing, reading `p.grad`. |
| [graph-and-autodiff.md](agent-notes/graph-and-autodiff.md) | Forward-code ownership and `consume_forward_code`, fragment ordering, `*_pspace` product-space gradients, the numpy-oracle recipe for silent numeric divergence. |
| [lowering-and-analysis.md](agent-notes/lowering-and-analysis.md) | What `analyze_proc` establishes and what `specialize_proc` reconciles: operand conditionality, the canonical render shared by both digests, the analysis cache, the traced store as node registry, config-key classification, typed `Affine` paths, signed `index_prec`, hosted constant inits, slice aliases, `promote_prec`. |
| [virtualization-and-inlining.md](agent-notes/virtualization-and-inlining.md) | Where a candidate is refused and by which phase (provenance codes), guard- and loop-context capture, `Local_scope` purity, the visit / reduction / fan-in caps, and how to test a value-rewriting pass on hand-built IR (`ll_test`, `?prelowered`). |
| [scheduling-and-autotune.md](agent-notes/scheduling-and-autotune.md) | Companion coverage and `aligned_chains`, fission, batch-`Grid` folding, `Tile_mma` barriers and `cp.async`, asynchronous dispatch and where launch parameters are read, failure classification and arm containment, and the four things a search result is not: seeded ≠ timed ≠ tensorized ≠ crowned ≠ shipped. |
| [backend-memory.md](agent-notes/backend-memory.md) | Pool tables and why device buffers are not GC-reclaimable, `Context.release` and what it cannot reach, allocation seams, GPU graph capture, HIP scratch budgets, `get_used_memory`, rematerialization and footprint scoring. |
| [backend-precision-and-simd.md](agent-notes/backend-precision-and-simd.md) | Storage vs compute precision, accumulator residency and narrowing points, the fp8/fp16/bf16 software codecs and vendor disagreements, fast math and device-side finiteness tests, `_Float16` probing, FMA rounding, vector splats and signed zero, SIMD width ranking, the fp16-not-bf16 traffic verdict, and the arity rule for `builtins.c`'s OCaml-facing stubs. |
| [backend-dialects-and-idents.md](agent-notes/backend-dialects-and-idents.md) | Per-dialect hazards (MSL `select`, `bfloat` builtins, untyped literals, CUDA/HIP half literals and overloads), the Metal RMW miscompile and pooled binding, the macOS `dispatch_apply` trap behind pool-backed `Grid` rendering, the `ident_blacklist`, and `C_syntax_config`'s include-time binding. |
| [training-and-performance.md](agent-notes/training-and-performance.md) | `params` vs `trainable_params`, training-loop utilities, the Metal training recipe, cost-model calibration and envelope fitting, the cross-framework benchmark suite, and A/B measurement protocol. |
| [build-and-test.md](agent-notes/build-and-test.md) | Dune mechanics behind the workflow rules: the `@fmt` gate and the format-test-promote order, scanning checks, `PIN THE RELATIONSHIP, NOT THE RESTATEMENT`, `agent_notes_structure` over these very files, `tools/test-run.sh`, `copy_files` and env-var tracking, `@check` not linking, worktree roots, plus what CI actually covers. |
| [conventions.md](agent-notes/conventions.md) | Release tags, `ocannl_config.reference` and configuration spellings, `bin/` argument parsing, stdout-belongs-to-the-program, git worktree and `gh pr merge` mechanics, stacked-PR retargeting, atomic file publication through `Utils.Atomic_file`, and the honesty rules for skipped legs, references and reports. |

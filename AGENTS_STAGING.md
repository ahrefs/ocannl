# Agent Learnings (Staging)

This file collects agent-discovered learnings for later curation into CLAUDE.md / AGENTS.md.

<!-- Entry: march-native-coder | 2026-02-22 -->
### Docs/Runtime Validation Mismatch

Repository docs suggest `dune exec bin/hello_world.exe`, but this worktree currently only has `bin/compilation_speed.ml`. For CC-backend runtime sanity checks, use a known executable target such as `test/operations/micrograd_demo_logging.exe`.

`OCANNL_BACKEND=sync_cc dune runtest` may fail due to unrelated existing test breakages in this branch state. For scoped changes, rely on `dune build @check` plus targeted runtime tests for the touched area.

<!-- End entry -->
<!-- Entry: gh-ocannl-299-coder | 2026-03-02T12:40:17+0100 -->
### Pre-existing fixes may only need expected-file updates

When a proposal describes a bug, check git history for prior fix commits before implementing a new fix. The bug fix may already exist in the codebase (e.g. from a different task branch or contributor), with only the test expected files left stale. Use `git log -- <relevant-files>` to find prior fix commits. The `dune build @runtest-<test_name>` alias is the reliable way to run a single standalone test and see its diff output.

### test_max_pool2d has a pre-existing failure

As of commit e6ed2f21, `test/einsum/test_max_pool2d.ml` fails with `Map.of_alist_exn: duplicate key (Symbol 42)` in `shape.ml:derive_projections`. This is unrelated to concat work and should not block other PRs.

<!-- End entry -->
<!-- Entry: workshop-paper-coder | 2026-03-03T08:59:53+0100 -->
### LaTeX compilation for workshop papers

- The ACM `acmart` class requires `\country{}` in `\affiliation{}` or it errors. Use `[acmsmall,nonacm,review]` options for workshop drafts.
- The `lstlisting` `escapeinside` option (e.g., `(@}{@)`) conflicts with `@` characters in OCaml operator names like `@^+`. Use `\texttt{}` instead of `\lstinline` for inline code containing `@`.
- The `acmart` class already loads `amsmath`/`amssymb`; adding them again causes a `\Bbbk already defined` error.

### Key source files for OCANNL shape system documentation

- The actual multi-head attention implementation is at `lib/nn_blocks.ml:115-134` — this is the canonical example for papers/talks.
- Shape inference documentation is split across three files: `docs/shape_inference.md` (internals, 357 lines), `docs/syntax_extensions.md` (user-facing syntax, 752 lines), `docs/slides-shapes_and_einsum.md` (presentation, 565 lines). All three are needed for a complete picture.
- The inference pipeline is described as 7 stages in `shape_inference.md` but the stage table shows stages 1-7 (the doc mentions 8 stages in prose due to a fractional stage split).

<!-- End entry -->

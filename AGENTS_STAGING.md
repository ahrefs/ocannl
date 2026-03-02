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

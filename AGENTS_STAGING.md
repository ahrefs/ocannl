# Agent Learnings (Staging)

This file collects agent-discovered learnings for later curation into CLAUDE.md / AGENTS.md.

<!-- Entry: march-native-coder | 2026-02-22 -->
### Docs/Runtime Validation Mismatch

Repository docs suggest `dune exec bin/hello_world.exe`, but this worktree currently only has `bin/compilation_speed.ml`. For CC-backend runtime sanity checks, use a known executable target such as `test/operations/micrograd_demo_logging.exe`.

`OCANNL_BACKEND=sync_cc dune runtest` may fail due to unrelated existing test breakages in this branch state. For scoped changes, rely on `dune build @check` plus targeted runtime tests for the touched area.

<!-- End entry -->

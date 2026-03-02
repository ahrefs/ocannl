# Agent Learnings (Staging)

This file collects agent-discovered learnings for later curation into CLAUDE.md / AGENTS.md.

<!-- Entry: march-native-coder | 2026-02-22 -->
### Docs/Runtime Validation Mismatch

Repository docs suggest `dune exec bin/hello_world.exe`, but this worktree currently only has `bin/compilation_speed.ml`. For CC-backend runtime sanity checks, use a known executable target such as `test/operations/micrograd_demo_logging.exe`.

`OCANNL_BACKEND=sync_cc dune runtest` may fail due to unrelated existing test breakages in this branch state. For scoped changes, rely on `dune build @check` plus targeted runtime tests for the touched area.

<!-- End entry -->

<!-- Entry: gh-ocannl-49-coder | 2026-03-02T12:20:00-0500 -->
### Pair Session Branch Naming Assumption

The pair preflight helper currently assumes a `main` branch for change-footprint reporting and warns on repositories that use `master`. Treat that warning as tooling noise unless other checks fail; use `origin/HEAD` or the actual default branch when validating branch diffs.

<!-- End entry -->

<!-- Entry: gh-ocannl-49-coder | 2026-03-02T12:17:00-0500 -->
### Stale Expected Output Can Mask Fixed Behavior

In this repo, some `.expected` files may still contain historical "known limitation" error blocks after a bug is fixed. If a targeted runtest fails with a diff that only removes obsolete error text, treat it as expectation drift: update the `.expected` file and re-run the same scoped alias (for example `@runtest-test_concat_graph`) rather than broad test suites.

<!-- End entry -->

## The project: OCANNL

Read `AGENTS.md` in your worktree, then the notes matching your subsystem from
`docs/agent-notes.md` (your task section names the relevant notes). Those files own project rules
and backend facts; this part supplies OCANNL's commands to the shipped native-Claude common brief.
The common brief owns reservations, execution requests, waiting, review and hand-back. A refused
or failed batch is not readiness to commit or merge; follow its recovery path before continuing.

### Startup addition

Alongside the common brief's startup checks, run `command -v dune ocaml`, `dune --version` and
`ocaml -version`, and record the results in `@@SCR@@/startup.txt`. Check the versions against
`dune-project`'s Dune language and package dependency requirements, plus any toolchain constraint
in your task section. If the tools are missing from PATH, use `eval "$(opam env)"` in each shell
command and repeat the checks. Do not change the shared switch to satisfy a task: follow
AGENTS.md's dependency-pin guidance.

### Verification

- Treat an issue's premise as a claim to test at your base, particularly one filed from reading.
  Start with a reproducer; a finding that does not reproduce is a result to report with evidence.
  Follow AGENTS.md's testing rules for `Verdict`, discriminating producers and test placement.
- Scope runs with `tools/test-run.sh affected --since origin/@@MAIN@@`; `--referenced-only` is for
  mid-change iteration apart from the explicit first-tier prepush case below. Read the
  *Aliases and goldens* section of `docs/agent-notes/build-and-test.md` for the reach boundary,
  UNMAPPED/OFF-BUILD handling and variant aliases.
- Use the common brief's slot-and-wait wrapper for each batch. The runner command is
  `tools/test-run.sh run --targets-file <F> build -j 4`, or for one test,
  `tools/test-run.sh run build @test/operations/runtest-<name> -j 4`. Keep targets files in
  `@@SCR@@/`. Use `--cpu` on the slot only when the batch holds no GPU; AGENTS.md and the
  *GPU boxes: job caps and runtime refusals* notes own width limits and device-health refusals.
- Before pushing, run `tools/test-run.sh prepush --since origin/@@MAIN@@ build -j 4`, then
  `dune build @check -j 4`, each slot-wrapped. If the default list is CROSS-CUTTING, use
  `tools/test-run.sh prepush --referenced-only --since origin/@@MAIN@@ build -j 4` for its first
  tier **only if narrow**, plus tests exercising the change; leave the full list to PR CI. If the
  first tier is itself CROSS-CUTTING, run the tests exercising the change and name them in the
  PR body. Run applicable `slow-` aliases named from the full list: CI does not run `@slow`.
  Follow the notes for `train-` members and shell harnesses; a long batch needs the common
  brief's execution request.
- Select backend legs using AGENTS.md and the *Skip coverage* and *Codegen text inventory*
  sections of `docs/agent-notes/build-and-test.md`. A golden with a backend-skip branch needs
  a non-CPU leg before merge; value tests that run on every backend need those backend legs.
  Before changing code generation run the inventory alias named in AGENTS.md and follow both
  its text-golden and kernel-execution routes. Use the task's assigned legs or request missing
  ones through the common brief; do not infer hardware coverage from a uniform golden.
  Pin the backend with `env OCANNL_BACKEND=<backend> tools/test-run.sh run --expect-backend
  <backend> ...` and record the backend and evidence for each leg in the PR body.
- Run `tools/test-run.sh run build @test/operations/scans -j 4` for the edit triggers listed in
  AGENTS.md. Keep one Dune process per worktree and library sources unchanged during a run.
  For measurements, read `benchmarks/README.md`'s *Measurement boxes* gate before requesting one.

### Formatting and committing

Follow AGENTS.md's *Format before the first push*: `dune fmt` as its own completed, slot-wrapped
step, then tests, then promotion using the project's promotion rules, and **last**
`tools/fmt-check.sh` in a CPU slot. Commit only after that batch's published status is zero, with
no edit between the check and commit, using the common brief's commit-gate form. Never pipe the
check or chain the commit after it with `;`. A failed gate requires a fix and another check.

AGENTS.md owns commit scope, per-commit compilation, co-author credit, ignored ppx expectations
and the changelog policy. For this native-Claude template the trailer is
`Co-Authored-By: <your own model name> <noreply@anthropic.com>`; credit your model, not the
coordinator's. Keep verification evidence in the PR body and issue comments as the common brief
requires.

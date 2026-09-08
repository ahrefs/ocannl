# OCANNL Agent Guide

This file guides coding agents working in this repository.

## Project Overview

OCANNL (OCaml Compiles Algorithms for Neural Networks Learning) is a from-scratch compiled Deep Learning framework with an optimizing compiler. The project consists of two main packages:

- `arrayjit`: The low-level optimizing compiler with multiple backends (CPU, CUDA, Metal, HIP)
- `neural_nets_lib`: The high-level deep learning framework with syntax extensions, shape inference, and backpropagation

## Structure and Ownership

- `lib/`: user-facing recipes (training utilities, nn blocks, re-exports).
- `tensor/`: core framework internals (Tensor, Shape, Operation, ppx_%op/%cd).
- `arrayjit/`: compiler + backends (`assignments.ml`, `low_level.ml`, `indexing.ml`, `schedule.ml`, `context.ml`, and the `*_backend.ml` implementations).
- `bin/`: runnable benchmarks and demos.
- `test/`: tutorials and tests (ppx_expect and standalone `.expected` tests).
- `docs/`: slides and reference docs.
- `build_files/` and `log_files/`: generated artifacts when debug settings are enabled.

Key reference files:

- `docs/syntax_extensions.md` (authoritative for %op/%cd)
- `docs/shape_inference.md` (shape/projection inference pipeline)
- `arrayjit/lib/context.mli` (context-based runtime API)
- `ocannl_config.reference` (all configuration keys and defaults)
- `docs/agent-notes.md` (index) and `docs/agent-notes/` (distilled cross-session agent knowledge)

## Build Commands

The project uses Dune for building and testing:

```bash
# Build all packages; this triggers running executables for cram-style tests
dune build

# Only compile -- do not run any executable
dune build @check

# Build specific package
dune build -p neural_nets_lib
dune build -p arrayjit

# Run tests
dune runtest

# Run tests for a specific backend (bash syntax)
OCANNL_BACKEND=cuda dune runtest

# Install dependencies (OCaml >= 5.3). --with-dev-setup adds ocamlformat (pinned to the
# .ocamlformat version) and ocaml-lsp-server, which --deps-only alone never installs
opam install . --deps-only --with-test --with-dev-setup

# Install with optional backends  
opam install cudajit  # for CUDA backend
opam install hipjit   # for AMD HIP backend
```

**Worktrees**: nested ones (`.claude/worktrees/`, the Claude Code default) need a `dune-workspace` at their root, or dune builds the PARENT checkout instead; the SessionStart hook writes it, so after a mid-session worktree switch run `scripts/setup-ocaml-env.sh` by hand. See docs/agent-notes/build-and-test.md for why.

**Windows shells**: use **Git Bash** (MSYS) — not a Cygwin bash — and source `tools/opam-env.sh` before building: `opam env` emits cygwin-style paths that leave an MSYS session with a half-working toolchain (dune found, linking broken) until the script rewrites them. Route dune through `tools/dune-quiet.sh`, which filters the benign binutils warnings that flood link stderr on Windows while preserving dune's exit status. How to tell the two bashes apart, and why Cygwin's is refused (gh-ocannl-662), is in docs/agent-notes/build-and-test.md.

**Format before the first push** (gh-ocannl-938): CI's `fmt` job runs `dune build @fmt` on every PR and is red on any unformatted `.ml`, `.mli` or dune file, so run `dune fmt` before pushing — a formatting-only fix push costs a CI round AND a review round, since reviews fire on every push. Order matters where a file carries `~here` goldens: `dune fmt`, then the test run, then promote, because a reformat shifts the `file:line` those goldens embed. Prose re-wrapped by hand in a review-fix commit is the usual way an unformatted hunk slips in; `dune fmt` after editing prose too. Master is always formatted, so `dune fmt` from a branch touches only your own files. New ppx-expectation files (`test/ppx/*_expected.ml`, compared against pretty-printed ppx output) must stay unformatted — add them to `.ocamlformat-ignore` (`test/operations/ocamlformat_ignore_scan` enforces it), or `@fmt` reformats them and their test promotes them back, forever. The formatter is pinned by `.ocamlformat` and installed locally by `opam install . --deps-only --with-dev-setup`.

## Architecture Overview

**Before working on a subsystem, read the matching file under `docs/agent-notes/`** (the index is
`docs/agent-notes.md`) — distilled
cross-session agent knowledge (solver/backend traps, known bugs with workarounds, debug recipes,
design history) that is not derivable from the code alone.

### Key Concepts

1. **Dual Syntax Extensions**:
   - `%cd` ("code"): For assignment computations (`Assignments.comp`)
   - `%op` ("operation"): For tensor expressions (`Tensor.t`)
   - Inline declarations lift to unit parameter `()` scope, enabling parameter reuse

2. **Shape Inference** (`docs/shape_inference.md` is the authoritative reference):
   - Three axis kinds — batch | input -> output (matrix convention: input rightmost) — with row variables (`..d..`) for broadcasting and generalized einsum notation for convolutions, reductions, and arbitrary permutations
   - "Principle of least commitment": use row variables where axis count doesn't matter
   - Shape inference completion is forced by lowering: via `Context.compile`, or wrappers such as `Train.to_routine`, `Train.run_once` or `Train.forward_once`; `finish_inference` closes still-unsolved dims (GLB where known, otherwise 1/broadcast)
   - Operations in `Operation`, `TDSL`, `NTDSL` return functions with `Tensor.op_fun` type, so that shapes can be specified at call sites if needed
   -  Operations in `TDSL.O` (opened for `%op`), `NTDSL.O` (opened for `%cd`) hide this so that shapes have to be inferred
   
3. **Backend Architecture**: Unified interface supporting CPU (multicore), CUDA, HIP, and Metal backends

4. **Memory Management**: Tensor node memory modes are `Virtual` (inlined computations), `Local`, and `On_device`.
   CPU-side reads and writes are explicit, context-mediated operations
   (`Context.to_host`/`from_host`, `get_values`/`set_values`).

## Development Workflow

### Testing

- Tests are either inline `%expect` expectations (`ppx_expect`) or standalone executables whose stdout is compared against an `.expected` golden via Dune's `test` stanza; the two are exclusive within one test
- `.expected` tests are easier to debug — use them for new features. Tutorial `%expect` files in `test/` double as documentation and integration tests; use them only when the outputs are illustrative

**Running tests** — the dune mechanics, traps and authoring recipes behind these rules live in `docs/agent-notes/build-and-test.md`; read it before authoring a new scan, alias family, slow test or golden format:
- `dune runtest` runs everything except the training integrations (`@train`) and the slow runs (`@slow`), which have their own aliases below; `dune runtest test/operations/` scopes to one directory
- Avoid `dune exec test/.../test_name.exe` for standalone tests: the config search walks up from the invocation cwd and the root `ocannl_config` is deliberately gitignored, so the test finds no config and dies partway, or `Context.auto` silently picks a GPU backend. Run one test by its alias instead — `dune build @test/operations/runtest-<name>` — which uses dune's cwd AND applies the `.expected` diff (promotable; a misspelled name exits 1 rather than silently doing nothing), leaving `_build/default/<dir>/<name>.exe.output` to inspect. For `bin/` executables (same cwd trap), pin `OCANNL_BACKEND=...` explicitly
- A test written as an `(executable)` plus a golden-diff `(rule)` — the scanning tests, codegen snapshots, config-precedence rules, ppx-output diffs — carries a hand-written `(alias runtest-<name>)` of its own (gh-ocannl-726), aggregated back into the directory's `runtest`; the authoring rules (alias naming, aggregation, the ambient gate) are enforced by `test/operations/env_var_deps` and spelled out in the agent note. Dune's own generated `runtest-<name>` aliases need dune >= 3.20, the project's declared floor
- The repo-wide scans — the rules that parse the repository itself (the authoritative list is the `(name scans)` stanza in `test/operations/dune`) — share `dune build @test/operations/scans`, which runs in seconds (gh-ocannl-703). Run the family before pushing a change to a config key, a dune stanza, a printed claim, an agent note, or any new script or source file a scan's glob would pick up. Two focused aggregates sit beside it (gh-ocannl-783): `dune build @metal-codegen` (the Metal-pinned tests) and `dune build @lifecycle` (the resource-lifecycle probes), each spanning both `test/operations` and `arrayjit/test`
- Pinning a variable on a run (`OCANNL_BACKEND=cuda dune build @test/operations/runtest-<name>`) reaches it only for variables the stanza DECLARES — dune tracks no environment variable it was not told about, and an undeclared one serves the previous run's result as a pass. Every test stanza that can select a backend declares `(env_var OCANNL_BACKEND)` (one spelling, uppercase; setting the dropped lowercase form is a fatal startup error on case-sensitive platforms — native Windows's case-insensitive environment reads it as the same variable — gh-ocannl-652); a stanza that names its backend, or links none, instead carries a marker comment inside its own parentheses — `; ocannl-backend: <none|cc|multidev_cc|cuda|hip|metal> -- <reason>`, comma-separated where a stanza honestly names two — exactly one of the two, never neither or both (gh-ocannl-659). For an `(executable)` plus a `(rule)`, the marker and the deps go on the rule. For any other key, add `OCANNL_<KEY>` to the stanza's `(deps ...)`. `env_var_deps` enforces all of this, including the per-module tracing-gate declarations (gh-ocannl-628), ambient-environment guards (gh-ocannl-749), `Test_utils.Generated.init` callers (gh-ocannl-723), and the per-directory `env_spelling_gate`
- Config startup chatter (the welcome message, the `log_config_sourcing` trace, the profile banner) goes to stderr, so stdout stays a clean data channel and `.expected` goldens never see it; pass `--ocannl_log_config_sourcing=true` to trace where each setting came from. A backend-uniform golden cannot confirm which backend a probe ran on — read the run's stderr, or have the test print the backend (gh-ocannl-622)
- Every `(test)`/`(tests)` stanza, every `(library)` with `(inline_tests)`, and every `(rule)` that runs a test executable must list `ocannl_config` in its `(deps ...)` — nothing is sandboxed, so a missing dep makes the run order-dependent instead of failing (gh-ocannl-586); `test/operations/config_dep_completeness` enforces it over every dune file in the repository — a rule that runs something reading no configuration goes on that test's named exemption list instead
- A `(test)` stanza automatically diffs the `<name>.expected` sitting next to it on `dune runtest`; the explicit rule-plus-diff pattern is only for tests no `(test)` stanza runs, such as the `@slow` rules
- Run suites through `tools/test-run.sh`, never hand-rolled shell around dune: `tools/test-run.sh run runtest test/operations`, `tools/test-run.sh run build @slow`. It runs dune unpiped (piping masks promotion diffs), capped (default 3600s, `--cap N`), logged to a file ending in an `exit: N` sentinel, and exits with dune's status — except that an invocation dune's own CLI refused (unknown option or subcommand, malformed operand: `dune: …` then `Usage: dune …`, nothing ran) reports `INVOCATION REFUSED` and exits 2, the script's usage code, while the sentinel keeps dune's 1; the full vocabulary and exit-code table are in `docs/agent-notes/build-and-test.md`. Never write a sleep/`pgrep` waiter loop around a run — launch through the harness's background execution and act on the completion notification; only for a run that must outlive the session, use `start`, then `status last` or `wait last`
- From PowerShell, QUOTE dune alias targets (`dune build "@runtest" "@slow"`): an unquoted `@word` splats to nothing and the command degrades to a false-green plain build (bash/cmd are unaffected)
- Scope test runs to the code and configuration paths a change can reach: start with directory aliases (`dune build @test/operations/runtest`, `@test/einsum/runtest`, `@test/ppx/runtest`, `@arrayjit/runtest`, `@test/training/runtest`), then `dune build @check`. Before broad testing of a config-gated path, grep whether any test or test config enables the gate; reserve the full regular/slow suites for cross-cutting changes
- Inline tests are part of library modules and run via `dune runtest`, not `dune exec`
- Keep library sources unchanged while Dune is running; edits invalidate in-flight rules and can repeat expensive work

**Training integration runs (the `train` alias)**:
- The toy training integrations (`mlp_names`, `mlp_bn_names`, `circles_conv`, `fsm_transformer`, `transformer_names`) are small by intent but serialize on the training lock, so they live on the `train` alias off the `runtest` path: `dune build @train`, one at a time via `@test/training/train-<name>`, everything via `dune build @runtest @train`. Per-PR CI runs the tier as a macOS-only shard and the daily sweep covers every backend; after touching training dynamics, `Train.*` plumbing, or the autotuner's fission path, still run the affected members locally before pushing

**Slow training tests (the `slow` alias)**:
- Excluded from `dune runtest`; run on demand with `dune build @slow`, one at a time via `@test/training/slow-<name>` (e.g. `OCANNL_BACKEND=cc dune build @test/training/slow-cifar_conv`), or everything via `dune build @runtest @slow`. They are ordinary executables — `dune build @check` compiles them, so they cannot bit-rot; `dune promote` accepts golden changes
- Regular and `@slow` training actions share the `ocannl_training_test` Dune lock, so their process-local OpenMP pools never run concurrently; compilation remains parallel. Preserve the lock on new regular training tests and `@slow` rules. `test/operations/cpu_parallel` intentionally stays unlocked as concurrency stress
- The recipe for gating a new slow test (executable + `slow-<name>` rule + aggregate + `no-infer`) is in the agent note; `test/training/dune` is the pattern

**Test types and authoring**:
- Inline tests are files in a library's `modules` field with an `inline_tests` stanza (e.g. `test_threefry4x32.ml`); standalone tests pair a `test` stanza with an `.expected` golden
- `dune promote` accepts golden changes. On Windows, and on EVERY platform during a merge, promote through `tools/promote.sh`: mid-merge, a golden promoted after its `git add` gets committed with the pre-promotion content and only CI fails — the script stages what it promoted and says so; on Windows it also strips CRLF
- **A test that decides its own verdict reports it through `Verdict`** (`test/support/verdict.ml`; add `arrayjit.verdict` to the stanza's `(libraries ...)` — the bare private name `verdict` links fine in a dev build but is unresolvable under the `dune build -p <pkg>` that each `.opam`'s with-test command runs, which disables the stanza owning it). Prefer `open Verdict.Claims`, the lightweight claim surface shared by arrayjit-only and OCANNL-linked tests: it provides `p`, `pass_fail`, `fail`, `skipped ~backend`, and the quantified combinators without per-file aliases. A skip whose gate belongs to the host or configuration rather than the selected backend (a compiler target, preprocessing flag or filesystem feature) passes ``~aggregation:`Environment``, so its human announcement remains but is aggregated across the declared measurement-box matrix rather than reported as universal backend non-coverage. Only a claim whose execution is owned by a separately checked matrix passes ``~aggregation:`Outside_sweep``; it stays machine-record-validated and human-visible but is excluded from both fleet dimensions. A claim quantified over a collection goes through the quantified combinators — `p_all name xs ~f`, `p_none name xs ~f` and `p_exists name xs ~f` take the collection positionally, `p_empty name ~over xs` takes the derived collection positionally and the source it was derived from as `~over`, and `p_all2 name got want ~f` is the pairwise form — never `p` applied to a `List.for_all`, vacuously true on an empty collection and indistinguishable in the golden (gh-ocannl-729); pairwise distinctness goes through `p_pairwise_distinct`, which also refuses fewer than two source values and reports the first collision. The unguarded spelling stays only where emptiness is the passing case, said so at the site. A failed claim exits the process 1, which is what keeps a failure from being `dune promote`d into the golden (gh-ocannl-601). Phrase every claim so `true` is the passing reading — in a golden a blessed regression and a designed negative are the same line — and keep purely descriptive output on plain `printf`, which is not an assertion channel. `test/operations/verdict_ratchet` enforces the claim shapes (a descriptive print that trips it takes a named exemption there)
- **A float that a device reduction produced does not belong in a stdout golden at a fixed precision** — lowering the precision only moves the rounding tie across backends. Print the exact digits to stderr tagged `(not part of the golden)`, and put a `Verdict` claim on stdout about what the number was showing — one that FAILS on a wrong value, and TWO-SIDED wherever the digits were (an upper bound alone admits a sign error). Floats exact by construction (threshold constants, power-of-two loss scales, closed-form schedules, small dyadic sums) stay on stdout. The audit rule and the classified sites are in the agent note (gh-ocannl-725)
- A few `%expect_test` blocks capture exception backtraces that hard-code `file:line`; any edit that shifts lines in such a file forces a benign line-number-only re-promote — do it in the same shell as the failing run (the `.ml.corrected` can vanish between runs)
- For optimizer passes that change *what value a cell holds* (virtualization guards, index solving, accumulation/init elision), a structural test on the emitted op tree is necessary but NOT sufficient — also assert executed output against a materialized/reference run (`Context.compile`/`run`/`get_values`), and make every producer DISCRIMINATE: vary with every symbol of its iteration and stay clear of the init value (the `tick`/`tag` helpers in `test/operations/virtual_diagonal.ml`; the failure modes each non-discriminating shape hides are in the agent note)
- **A test that asserts on generated code reads it through `Test_utils.Generated`** (`test/support/generated.ml`), never by opening `build_files/<routine>.<ext>` itself — artifacts outlive the run and same-named routines overwrite each other, so an unprovenanced read can keep asserting on a kernel that is no longer emitted. Call `Generated.init ~backend_name` before the first compile (leave `build_files_prefix` at its default; declare `(env_var OCANNL_BUILD_FILES_PREFIX)` on the stanza dune runs it under), then `assert_emits`/`read`; call `arm` before each compile in a loop reusing a routine name; gate a leg the backend cannot evaluate with `Verdict.skipped` before it reaches the read
- **Pin the relationship, not the restatement**: a check that needs a set another part of the system owns derives it, or asserts the two equal from where the link cost is already paid — never a second copy asserted to still say what it says, which is a test that cannot fail. The shapes, exemplars, and the exceptions (judgment lists, deliberately independent constants) are in the agent note
- Backend codegen snapshots (e.g. `.cu.expected` files) go stale when codegen changes land without that backend's hardware available — expect to re-promote them when the hardware next runs the suite
- **Before changing code generation**, run `dune build @test/operations/runtest-codegen_text_inventory`: its golden enumerates every file that pins the TEXT of emitted code — goldens in BOTH `test/` and `arrayjit/test/`, plus sources asserting on emitted text from string literals, which fail a plain `dune build` (gh-ocannl-712). Reach emitters through a qualifier, never an `open` — the scan refuses the file otherwise (gh-ocannl-748)
- **Test placement**: always under a `test/` subdirectory — default `test/operations`, complex einsum specs in `test/einsum`, training loops in `test/training`. Give the stanza `ocannl_config` in its `(deps ...)` plus the env-var declaration or backend marker per above, and add an (initially empty) `.expected`. Tests that hand-build `Ir.Low_level.t` share the `ll_test` support library (`test/support/ll_test.ml`; it links `ocannl`, unlike `test_utils` which stays on `arrayjit.ir` alone). Debug artifacts go to per-executable `build_files/<exe-name>/` subdirectories, so concurrent tests cannot clash (only the flat legacy layout `build_files_prefix=.` shared across processes can still race) — but a same-named routine WITHIN one executable silently overwrites the earlier one's artifacts, so keep routine names distinct (the `af_`/`ops_`/`smem_` prefix conventions)

**Windows portability for `.expected` tests**: `.gitattributes` pins `*.expected` (and `test/ppx/*_expected.ml`) to LF; promote through `tools/promote.sh` and edit goldens with bash tools, since PowerShell `Set-Content`/`Out-File` write CRLF. Format floats destined for goldens with the `test_utils` portable printers (`print_float`/`print_floats`/`hex_float`, plus `set_binary_stdout` for byte-exact echoes) or `Ir.Ndarray.concise_float ~prec`, not `%g`/`%e` — the Windows C runtime prints 3-digit exponents, which those normalize. They do NOT absorb decimal-tie rounding (Windows rounds representable ties away from zero, glibc to even): avoid tie values in test data, or print with `hex_float`/OCaml's `%h`, which sidesteps decimal rounding entirely (details in the agent note)

**Module Paths and Common APIs**:

- **For files outside OCANNL implementation (tests, examples, user code), start with `open Ocannl.Nn_blocks.DSL_modules`** - it re-emits the DSLs of `Ocannl.Operation.DSL_modules` (defined near the end of `tensor/operation.ml`) extended with the non-uniform initializers `normal`, `normal1`, `normal_at`, `normal_at1`, `kaiming`, `xavier`, `kaiming_at`, `xavier_at` (`lib/nn_blocks.ml`), which is what building a model needs. `open Ocannl.Operation.DSL_modules` is the narrower open, for a file that uses no initializer
- Available modules after either open:
  - `Ir` - Low-level IR types and operations (Ndarray, Ops, Tnode, etc.)
  - `Row` - Row variables for shape inference; exported by `Operation.DSL_modules` only, so under the `Nn_blocks` open it comes from `open Ocannl` (`lib/ocannl.ml`), which such files generally have anyway
  - `Shape` - Shape inference and einsum notation
  - `Tensor` - Core tensor type and operations
  - `TDSL` - Tensor DSL with automatic differentiation (grad_spec: If_needed)
  - `NTDSL` - No-gradient tensor DSL (grad_spec: Prohibit_grad)
- There is no `PDSL` (Require_grad DSL). To build a differentiable leaf tensor with concrete values (e.g. in tests), pass the grad spec explicitly: `Operation.init ~l ~prec ~b ~o ~f ~grad_spec:Tensor.Require_grad ()` or `Tensor.term_init values ~grad_spec:Require_grad ()` (1-D); see `test/training/fused_classifier.ml`, `test/operations/primitive_ops.ml`
- Precision values: `Ir.Ops.single`, `Ir.Ops.double`, `Ir.Ops.half` (lowercase)
- Tensor printing in expect tests: `Tensor.print ~here:[%here] ~force:false ~with_code:false ~with_grad:false \`Inline tensor`
- Library sets in `dune` are per stanza, declared to match what the module actually references: a test that computes and prints typically wants `(libraries base ocannl stdio)`, but that is a starting point, not a fixed triple — a module with no `open Base` that prints nothing declares `(libraries ocannl)` alone, and a test that asserts a verdict adds `arrayjit.verdict` (required by the testing section above). An unused `open` is a fatal warning here, and unused libraries are noise

### Pull Requests

- A **PR accomplishes a goal**: one thing that is true about the system afterwards and was not before, stated in its title. Scope it generously — carry the goal to its natural completion (the change, the tests that pin it, the docs it justifies, the follow-on cleanups it exposes) rather than stopping at the smallest reviewable increment and opening three PRs. Velocity and clarity comes from finishing a goal in one pass
- A **commit is one move toward the goal**, not one slice of the work by artifact type. Whatever a move needs — the logic change, the tests that pin it, its `.expected` goldens, the doc or agent-note it justifies — belongs in that one commit. Splitting a change away from its own tests or documentation makes the series harder to read, not easier
- A goal usually takes several moves, so a series of topical commits is the norm — merged with a merge commit that preserves the series
- **Do NOT touch `CHANGES.md` in feature work** (gh-ocannl-807): the changelog is written in editorial passes — at release prep, or an occasional explicitly-requested batch catch-up — from the durable records the work already leaves (merge commits, PR bodies, issue closing comments); a per-PR entry duplicates the PR body, and the shared `## [Unreleased]` anchor made every concurrent PR conflict there. The editorial-pass conventions are in `docs/agent-notes/conventions.md`
- When you notice unrelated code smells or design problems, file separate issues
- Follow-up fixing commits are fine, and test-expectation promotions that span several topics can land in a final tests/promotions commit
- When creating commits, include the work summary in the commit message and credit yourself as a co-author
- Each commit should at least compile: loop `git checkout <rev> && dune build @check` over `git rev-list --reverse master..HEAD` (interactive rebase is typically unavailable in agent harnesses)
- **Bring the base in before opening the PR; a clean merge does not restart verification**: GitHub builds the pull request's MERGE commit, so a check that scans the whole repository can be red on the tree CI builds while green on your branch. Fetch the STAGING remote (resolve which name points there per the next bullet — it need not be `origin`) and rebase onto its `master`, or merge it in where the branch is shared and rewriting is not yours to do — `docs/agent-notes/build-and-test.md` explains the mechanism and how to validate against the merged tree. The merge gate itself is one green full-matrix run for the PR's current head (the roll-forward policy, gh-ocannl-861): however far the base has moved, a clean merge proceeds on that run; any commit that moves the head — a conflict resolution, a rebase, a merge of the base — waits for its own green run; a diff CI's `docs/**` path filter ignores gets no run, and merges on that absence; and the base's advance having touched the PR's own files is the one case that still warrants bringing it in again — the note gives the merge-base-anchored diff that answers this, since an endpoint diff of base against head cannot. Master's CI after the merge is not the merger's to watch — the CI-red triage routine owns it (the agent note's CI section)
- **Two repositories, and remote names are not the contract**: development — branches, PRs, and the `master` they land on — happens in `lukstafi/ocannl-staging`, while `ahrefs/ocannl` is the public repo that owns the ISSUES this codebase cites as `gh-ocannl-NNN`, the milestones and the GitHub releases, and receives release-relevant changes. Which remote name points where is local: a clone has whatever names it was given, a clone of the public repo calls IT `origin`, and a fresh clone has no second remote at all. So check `git remote -v` before trusting a name, add the other repo explicitly when you need it (`git remote add upstream https://github.com/ahrefs/ocannl.git`), and pass `--repo <owner>/<name>` to every `gh` command rather than letting it infer: issues to `ahrefs/ocannl`, PRs to `lukstafi/ocannl-staging`

### Configuration

- See `ocannl_config.reference` for documentation of all settings. It ships with every setting COMMENTED OUT (`#key=…`, no space; prose comments use `# `), so copying it verbatim states nothing
- Key configs: backend selection, debug logging, optimization levels
- **Adding a config key touches two places**, enforced by `test/operations/test_config_consistency`: document it in `ocannl_config.reference` and register it in `Utils.known_config_keys`. Spell the key as a string literal at the call site (`~arg_name:"the_key"`) — a helper taking the key as a parameter hides every key routed through it, and the scan fails non-literal uses outside the named lookup functions. New source files need no registration and owe no promote round: the scans glob every directory that can read configuration, and their goldens hold root names and floors, not counts (gh-ocannl-592, gh-ocannl-701). A key read only from a test is deliberately out of scope: tests are not user-facing configuration
- **Classify the key too**: `test/operations/digest_completeness` fails on a key with no entry in `Utils.config_key_classification` (gh-ocannl-572) — say whether it reaches the schedule cache's identity, and which component
- **A new post-lowering module still needs registering** in that test's `codegen_stage_modules` list — a new backend, or anything else reading configuration after lowering. The globs scan it either way, but that list is what marks its reads as happening downstream of the canonical digest, so a `Code_borne` misclassification of one of its keys goes unnoticed without it. It is a judgment call about the module, which is why it stays hand-written where the file list no longer is

**Configuration Methods** (in order of precedence):
1. Command-line flags: `--ocannl_<option>=<value>` (e.g., `--ocannl_backend=cuda`)
2. Environment variables: `OCANNL_<OPTION>=<value>` (e.g., `OCANNL_BACKEND=cuda`)
3. Config file: `ocannl_config` in current or ancestor directories

**Config profiles** (gh-ocannl-559): `profile=reproducible|performance|approximate` picks a preset bundle
whose payload (an embedded partial config file in `arrayjit/lib/utils.ml`) applies at the sublevel
just below the explicit keys of whichever source picked it — so explicit keys beat a profile of
equal immediacy, and a CLI-picked profile beats an exhaustive config file. Payload keys must be
known and documented, and the reference file's verbatim quote of each payload is checked, both by
`test/operations/test_config_consistency`. `approximate` (gh-ocannl-719) is `performance` plus every
numerics-changing knob, gated in the benchmarks at its own `PARITY_TOL_APPROX` envelope; a
numerics-changing rewrite gate lands in that payload, and in the schedule cache's numerics digest,
in the PR that adds its key (`test/operations/config_profiles` pins approximate ⊇ performance).

**Testing with Different Configurations**:

- Dune re-runs a test for an environment variable only where the stanza declares it as an `(env_var OCANNL_<KEY>)` dependency — so for a key a test should react to, add the declaration (see the testing section above), rather than working around the stale run
- **Warning**: `dune test --force` does NOT re-run expect tests (only rules with alias fields)
- For a one-off configuration change no stanza declares (notably for inline `%expect` tests): modify `test/config/ocannl_config` directly, run `dune clean`, or touch the affected test sources

**Important Debug Settings**:
- `output_debug_files_in_build_directory=true` - enables `build_files/` generation; files go to `build_files/<exe-name>/` (per-executable subdirectory, override with `build_files_prefix`; `build_files_prefix=.` for a flat layout)
- `debug_log_from_routines=true` - enables runtime logging from kernels aka. routines
- `debug_log_to_stream_files=true` - writes logs from kernels/routines to `log_files/<exe-name>/<backend>-<device>-<stream>.log`
- `clean_up_build_files_on_startup=false` and `clean_up_log_files_on_startup=false` - preserve debug files between runs
- CUDA routine logs may require `Utils.capture_stdout_logs` (see README)

**Available Backends**:
- `cc` (the default) combines the implementation cc_backend.ml with the scheduler `Sync` in schedulers.ml; kernel-level CPU parallelism is automatic (pool-rendered Grid loops)
- `multidev_cc` combines cc_backend.ml with the scheduler `Multidev`: multiple worker-domain CPU devices, for debugging multi-device parallel workflows ("sync_cc"/"multicore_cc" are accepted as deprecated aliases of cc/multidev_cc)
- `cuda` with implementation in cuda_backend.ml
- `hip` (AMD ROCm/HIP) with implementation in hip_backend.ml, mirroring the CUDA backend
- `metal` with implementation in metal_backend.ml

Backends are process-wide singletons: use `Backends.get_backend ()` or the Context API (`arrayjit/lib/context.mli`); `fresh_backend` is retired. Merge buffers (`.merge`) support stream-to-stream reductions in `%cd`.

### Backend Development

- Backends implement stream-based execution with FIFO queuing, events, and synchronization between streams/devices, generating code from `Low_level.t`
- Code generation: `c_syntax.ml` is a functor with default C patterns that each backend overrides for its own syntax; the touch-lists — including the per-backend builtins modules and the `convert_precision` obligation — are in the `extending-ocannl` skill (`.claude/skills/extending-ocannl/SKILL.md`)

### Syntax Extensions

`docs/syntax_extensions.md` is the authoritative reference for `%op`/`%cd` — the record syntax and its shorthand fields, inline-declaration scoping, einsum specs and dimension capture, projection slots. Orientation and traps:

- `%cd` requires `NTDSL` in scope, `%op` requires `TDSL` (both provided by the `DSL_modules` opens above)
- Record syntax for inline tensor declarations: `{ tensor_name }`, or `{ tensor_name = init_expr }` — initialization expressions are `%op`-only, for model parameters; they run forward-only, then `TDSL.param` adds the final parameter gradient

**Einsum notation** — binary `t1 +* "spec1; spec2 => result_spec" t2`, unary `t ++ "spec => result_spec"`; a trailing string list captures dimension/row variables (constrain them with `Shape.set_dim`):
- Operators -- binary: `+*` (`einsum`, add-reduce with multiply), `@^+` (`tropical`, max-reduce with add), `+++` (`outer_sum`, add-reduce with add); unary: `++` (`einsum1`, add-reduce), `@^^` (`einmax1`, max-reduce)
- Concatenation: `a^b` in specs creates concatenated axis (for slicing, block tensors)

**Common gotchas and idioms**:
- `*` is tensor/matrix multiply, `*.` is pointwise multiply (no `/`, use `/.` for pointwise division)
- `**.` is pointwise power with a numeric exponent (specialized gradients)
- Use `_rhs1`/`_rhs2`/`_lhs` suffixes in `%cd` for intermediate tensors when projection slots matter
- `stretch 1.0` creates a shape-inferred constant 1 whose shape resolves at the use site; `1.0` alone is a fixed scalar. Operation results otherwise close down to their arguments' shapes — a use site broadcasts them in but cannot widen them (gh-544; the old `0.5 + 0.5` idiom relied on the pre-544 widening default)
- Einsum spec must be a literal string when capturing dimensions: `x ++ "ab => a" ["b"]` works, `let s = "ab => a" in x ++ s ["b"]` fails
- Single-char vs multi-char mode: `"abc"` = 3 axes; `"abc,"` = 1 axis named `abc` (comma triggers multi-char)
- `{ param }` in `%op` creates learnable parameters; same syntax in `%cd` creates non-differentiable tensors
- Default param init is a centered scaled `uniform ()` over `[-0.25, 0.25)`, configurable via the reference `TDSL.default_param_init`
- Sub-modules with `()` must be bound before input: `let layer = make_layer () in fun x -> layer x`
- No reshape/flatten—use multi-axis operations or row variables instead

## Common Development Tasks

Touch-lists for adding a primitive operation, extending a backend, extending shape
inference, and diagnosing backend output discrepancies live in the `extending-ocannl`
skill (`.claude/skills/extending-ocannl/SKILL.md`). Debug-artifact and ppx_minidebug
tracing recipes live in the `ocannl-debug-tracing` skill
(`.claude/skills/ocannl-debug-tracing/SKILL.md`). Agents without skill support read
the files directly. In brief:

- New primitive ops: `arrayjit/lib/ops.ml` (+ `Ir.Ops`), wired into `tensor/operation.ml`
- New tensor convenience functions: `tensor/operation.ml` (use `%cd` for forward/backprop)
- Shape/projection changes: `tensor/shape.ml`, `tensor/row.ml`, `arrayjit/lib/indexing.ml`

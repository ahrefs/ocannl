# OCANNL Agent Guide

This file guides coding agents working in this repository. It is a rulebook with pointers:
the mechanism, the failure story and the enforcement behind each rule live in `docs/agent-notes/`
(index: `docs/agent-notes.md`). Keep it under 32 KiB — Claude Code truncates larger imports —
which `test/operations/agents_md_size` enforces; grow the notes, not this file.

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

# Install dependencies (OCaml >= 5.3); --with-dev-setup adds the pinned ocamlformat and ocaml-lsp-server
opam install . --deps-only --with-test --with-dev-setup

# Install with optional backends
opam install cudajit  # for CUDA backend
opam install hipjit   # for AMD HIP backend
```

**Worktrees**: nested ones (`.claude/worktrees/`) need a `dune-workspace` at their root or dune builds the PARENT checkout; the SessionStart hook writes it — after a mid-session worktree switch run `scripts/setup-ocaml-env.sh` by hand (docs/agent-notes/build-and-test.md).

**Windows shells**: use **Git Bash** (MSYS), never a Cygwin bash, and source `tools/opam-env.sh` before building (`opam env` emits cygwin-style paths that break linking). Route dune through `tools/dune-quiet.sh`, which filters the benign binutils link warnings while preserving dune's exit status (gh-ocannl-662; the agent note tells the two bashes apart).

**Windows verification placement**: GitHub-hosted Windows CI runs on its schedule, not per PR, and a merge never waits on the scheduled sweep. `rog-nv-win` and `minix-amd-win` boot Ubuntu; one the user reboots into Windows answers in minutes, where remote Windows CI takes one to three hours. So when a change needs Windows signal that matters (it fixes a Windows failure, or changes behavior only Windows exercises: line endings, float formatting in goldens, the mingw toolchain), first ask the user to boot a box into Windows, naming the box, commit and check; under a wave coordinator, hand the request to it, and it batches requests and picks the box with less queued work. With no answer in 30 minutes, withdraw the request by telling the user (through the same channel) that CI is taking the check, so a late reboot is not wasted, and only then dispatch `ci.yml` with `windows_only: true` and the full `expected_sha` and cite the run. Never dispatch while the request is open: a dispatched run's checks gate the head, and cancelling one leaves a no-verdict the merge refuses. An issue whose core development centers on Windows waits for a box rather than iterating through dispatches. The build-and-test note says how to run on a booted box.

**Format before the first push** (gh-ocannl-938): CI's `fmt` job runs `dune build @fmt` on every PR, and a formatting-only fix push costs a CI round AND a review round. Order: `dune fmt`, then the test run, then promote, since reformatting shifts the `file:line` that `~here` goldens embed. Master is always formatted. New ppx-expectation files (`test/ppx/*_expected.ml`) must stay unformatted — add them to `.ocamlformat-ignore` (`ocamlformat_ignore_scan` enforces it).

## Architecture Overview

**Before working on a subsystem, read the matching file under `docs/agent-notes/`** (index
`docs/agent-notes.md`) — distilled cross-session knowledge (solver/backend traps, known bugs with
workarounds, debug recipes, design history) not derivable from the code alone.

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
   - Operations in `TDSL.O` (opened for `%op`), `NTDSL.O` (opened for `%cd`) hide this so that shapes have to be inferred

3. **Backend Architecture**: Unified interface supporting CPU (multicore), CUDA, HIP, and Metal backends

4. **Memory Management**: Tensor node memory modes are `Virtual` (inlined computations), `Local`, and `On_device`.
   CPU-side reads and writes are explicit, context-mediated operations
   (`Context.to_host`/`from_host`, `get_values`/`set_values`).

## Development Workflow

### Testing

The dune mechanics, traps and authoring recipes behind every rule here are in
`docs/agent-notes/build-and-test.md`; read it before authoring a new scan, alias family, slow test
or golden format.

- Tests are either inline `%expect` expectations (`ppx_expect`) or standalone executables whose stdout is compared against an `.expected` golden via Dune's `test` stanza; the two are exclusive within one test
- `.expected` tests are easier to debug — use them for new features. Tutorial `%expect` files in `test/` double as documentation and integration tests; use them only when the outputs are illustrative

**Running tests**:
- `dune runtest` runs everything except the training integrations (`@train`) and the slow runs (`@slow`); `dune runtest test/operations/` scopes to one directory
- Avoid `dune exec test/.../test_name.exe` for standalone tests: the config search walks up from the cwd and the root `ocannl_config` is gitignored, so the test finds no config or `Context.auto` silently picks a GPU. Run one test by its alias — `dune build @test/operations/runtest-<name>` — which applies the `.expected` diff (promotable; a misspelled name exits 1) and leaves `_build/default/<dir>/<name>.exe.output` to inspect. For `bin/` executables pin `OCANNL_BACKEND=...` explicitly
- A test written as an `(executable)` plus a golden-diff `(rule)` carries a hand-written `(alias runtest-<name>)` of its own (gh-ocannl-726), aggregated back into the directory's `runtest`; `test/operations/env_var_deps` enforces the alias naming, aggregation and the ambient gate. Dune >= 3.20 is the declared floor
- The repo-wide scans (authoritative list: the `(name scans)` stanza in `test/operations/dune`) share `dune build @test/operations/scans`, which runs in seconds (gh-ocannl-703). Run it before pushing a change to a config key, a dune stanza, a printed claim, an agent note, or any new script or source file. Two focused aggregates sit beside it (gh-ocannl-783): `dune build @metal-codegen` and `dune build @lifecycle`, each spanning `test/operations` and `arrayjit/test`
- Pinning a variable on a run (`OCANNL_BACKEND=cuda dune build @.../runtest-<name>`) reaches it only for variables the stanza DECLARES; an undeclared one serves the previous run's result as a pass. Every stanza that can select a backend declares `(env_var OCANNL_BACKEND)` (uppercase only, gh-ocannl-652); a stanza that names its backend, or links none, instead carries `; ocannl-backend: <none|cc|multidev_cc|cuda|hip|metal> -- <reason>` inside its own parentheses, comma-separated where a stanza honestly names two — exactly one of the two forms (gh-ocannl-659); for an `(executable)` plus a `(rule)`, both go on the rule. Any other key goes into `(deps ...)` as `OCANNL_<KEY>`. `env_var_deps` enforces all of this, plus tracing-gate declarations (gh-ocannl-628), ambient-environment guards (gh-ocannl-749), `Test_utils.Generated.init` callers (gh-ocannl-723) and the per-directory `env_spelling_gate`
- Config startup chatter goes to stderr, so stdout stays a clean data channel and `.expected` goldens never see it; `--ocannl_log_config_sourcing=true` traces where each setting came from. A backend-uniform golden cannot confirm which backend a probe ran on — read stderr, or have the test print the backend (gh-ocannl-622)
- Every `(test)`/`(tests)` stanza, every `(library)` with `(inline_tests)`, and every `(rule)` that runs a test executable lists `ocannl_config` in its `(deps ...)` — nothing is sandboxed, so a missing dep makes the run order-dependent (gh-ocannl-586); `test/operations/config_dep_completeness` enforces it, with a named exemption list for rules that run something reading no configuration
- A `(test)` stanza automatically diffs the `<name>.expected` beside it; the explicit rule-plus-diff pattern is only for tests no `(test)` stanza runs, such as the `@slow` rules
- Run suites through `tools/test-run.sh`, never hand-rolled shell around dune: `tools/test-run.sh run runtest test/operations`, `tools/test-run.sh run build @slow`. It runs dune unpiped (piping masks promotion diffs), capped (`--cap N`), logged to a file ending in an `exit: N` sentinel, and exits with dune's status. Never write a sleep/`pgrep` waiter loop — launch through the harness's background execution and act on the completion notification; only for a run that must outlive the session use `start`, then `status last` or `wait last`. To genuinely re-execute an unchanged `.expected` test N times (sampling a timing-dependent golden), use `repeat N build @<dir>/runtest-<name>`: each iteration runs in a freshly cleaned, cache-disabled build directory and the outputs are diffed pairwise
- **A GPU suite on a WSL2 box runs at `-j 2`** (`rog-nv-wsl`, `minix-amd-wsl`): the `/dev/dxg` bridge overflows at dune's default width and the suite comes back red in exactly the stanzas a real backend regression lands in. `tools/test-run.sh run` injects the cap when it sees `/dev/dxg` and a GPU `OCANNL_BACKEND` (a backend named only in `ocannl_config` is invisible to it — pass `-j 2` yourself). On a native boot it injects the per-slot width the same way: `-j 4` for hip on minix's small SDMA pool, `-j 8` for cuda on rog (gh-ocannl-1033)
- From PowerShell, QUOTE dune alias targets (`dune build "@runtest" "@slow"`): an unquoted `@word` splats to nothing and the command degrades to a false-green plain build
- Scope test runs to what a change can reach: directory aliases first (`dune build @test/operations/runtest`, `@test/einsum/runtest`, `@test/ppx/runtest`, `@arrayjit/runtest`, `@test/training/runtest`), then `dune build @check`. Before broad testing of a config-gated path, grep whether any test or test config enables the gate; reserve the full regular/slow suites for cross-cutting changes
- Inline tests are part of library modules and run via `dune runtest`, not `dune exec`
- Keep library sources unchanged while Dune is running; edits invalidate in-flight rules

**Training integration runs (the `train` alias)**: the toy integrations (`bigram`, `mlp_names`, `mlp_bn_names`, `circles_conv`, `fsm_transformer`, `transformer_names`) serialize on the training lock, so they live off the `runtest` path: `dune build @train`, one at a time via `@test/training/train-<name>`. Per-PR CI runs them as a macOS-only shard and the daily sweep covers every backend; after touching training dynamics, `Train.*` plumbing or the autotuner's fission path, run the affected members locally before pushing

**Slow training tests (the `slow` alias)**: excluded from `dune runtest`; `dune build @slow`, one at a time via `@test/training/slow-<name>`, everything via `dune build @runtest @slow`. `dune build @check` compiles them, so they cannot bit-rot. Regular and `@slow` training actions share the `ocannl_training_test` Dune lock so their OpenMP pools never overlap — preserve it on new training tests and `@slow` rules (`test/operations/cpu_parallel` intentionally stays unlocked). The gating recipe is in the agent note; `test/training/dune` is the pattern

**Test types and authoring**:
- Inline tests are files in a library's `modules` field with an `inline_tests` stanza; standalone tests pair a `test` stanza with an `.expected` golden
- `dune promote` accepts golden changes. On Windows, and on EVERY platform during a merge, promote through `tools/promote.sh`: mid-merge, a golden promoted after its `git add` gets committed with the pre-promotion content — the script stages what it promoted; on Windows it also strips CRLF
- **A test that decides its own verdict reports it through `Verdict`** (`test/support/verdict.ml`; add `arrayjit.verdict` to `(libraries ...)` — the bare name `verdict` fails under `dune build -p`). Prefer `open Verdict.Claims`: `p`, `pass_fail`, `fail`, `skipped ~backend`, and the quantified combinators `p_all`/`p_none`/`p_exists`/`p_empty ~over`/`p_all2`/`p_pairwise_distinct` — never `p` applied to a `List.for_all`, vacuously true on an empty collection (gh-ocannl-729), except where emptiness is the passing case, said so at the site. A skip gated by the host rather than the backend passes ``~aggregation:`Environment``; a claim owned by a separately checked matrix passes ``~aggregation:`Outside_sweep``. A failed claim exits 1, so it cannot be `dune promote`d into the golden (gh-ocannl-601). Phrase every claim so `true` passes; descriptive output stays on plain `printf`. `test/operations/verdict_ratchet` enforces the shapes
- **A float that a device reduction produced does not belong in a stdout golden at a fixed precision** — lowering the precision only moves the rounding tie across backends. Print the exact digits to stderr tagged `(not part of the golden)`, and put a `Verdict` claim on stdout that FAILS on a wrong value and is TWO-SIDED (an upper bound alone admits a sign error). Floats exact by construction (thresholds, power-of-two loss scales, closed-form schedules, small dyadic sums) stay on stdout (gh-ocannl-725)
- A few `%expect_test` blocks capture backtraces that hard-code `file:line`; an edit that shifts lines there forces a benign re-promote — do it in the same shell as the failing run
- For optimizer passes that change *what value a cell holds* (virtualization guards, index solving, accumulation/init elision), a structural test on the op tree is necessary but NOT sufficient — also assert executed output against a materialized/reference run, and make every producer DISCRIMINATE: vary with every symbol of its iteration and stay clear of the init value (the `tick`/`tag` helpers in `test/operations/virtual_diagonal.ml`)
- **A test that asserts on generated code reads it through `Test_utils.Generated`** (`test/support/generated.ml`), never by opening `build_files/<routine>.<ext>` itself — artifacts outlive the run and same-named routines overwrite each other. Call `Generated.init ~backend_name` before the first compile (declare `(env_var OCANNL_BUILD_FILES_PREFIX)` on the stanza), then `assert_emits`/`read`; `arm` before each compile in a loop reusing a routine name; gate a leg the backend cannot evaluate with `Verdict.skipped`
- **Pin the relationship, not the restatement**: a check that needs a set another part of the system owns derives it, or asserts the two equal from where the link cost is already paid — never a second copy asserted to still say what it says. Shapes, exemplars and exceptions (judgment lists, deliberately independent constants) are in the agent note
- Backend codegen snapshots (`.cu.expected` etc.) go stale when codegen changes land without that hardware — expect to re-promote them when the hardware next runs the suite
- **Before changing code generation**, run `dune build @test/operations/runtest-codegen_text_inventory`: its golden enumerates every file that pins the TEXT of emitted code, in both `test/` and `arrayjit/test/` (gh-ocannl-712). Reach emitters through a qualifier, never an `open` (gh-ocannl-748)
- **Test placement**: always under `test/` — default `test/operations`, complex einsum specs in `test/einsum`, training loops in `test/training`. Give the stanza `ocannl_config` in `(deps ...)`, the env-var declaration or backend marker, and an (initially empty) `.expected`. Tests that hand-build `Ir.Low_level.t` share `test/support/ll_test.ml`. Debug artifacts go to per-executable `build_files/<exe-name>/`, but a same-named routine WITHIN one executable overwrites the earlier one's artifacts — keep routine names distinct (the `af_`/`ops_`/`smem_` prefixes)

**Windows portability for `.expected` tests**: `.gitattributes` pins `*.expected` (and `test/ppx/*_expected.ml`) to LF; promote through `tools/promote.sh` and edit goldens with bash tools, since PowerShell `Set-Content`/`Out-File` write CRLF. Format floats destined for goldens with the `test_utils` portable printers (`print_float`/`print_floats`/`hex_float`, plus `set_binary_stdout`) or `Ir.Ndarray.concise_float ~prec`, not `%g`/`%e`. Those do NOT absorb decimal-tie rounding (Windows rounds ties away from zero, glibc to even): avoid tie values, or print with `hex_float`/`%h`

**Module Paths and Common APIs**:

- **For files outside OCANNL implementation (tests, examples, user code), start with `open Ocannl.Nn_blocks.DSL_modules`** — the DSLs of `Ocannl.Operation.DSL_modules` (end of `tensor/operation.ml`) extended with the non-uniform initializers `normal`, `normal1`, `normal_at`, `normal_at1`, `kaiming`, `xavier`, `kaiming_at`, `xavier_at` (`lib/nn_blocks.ml`). `open Ocannl.Operation.DSL_modules` is the narrower open for a file that uses no initializer
- Available modules after either open:
  - `Ir` - Low-level IR types and operations (Ndarray, Ops, Tnode, etc.)
  - `Row` - Row variables for shape inference; exported by `Operation.DSL_modules` only, so under the `Nn_blocks` open it comes from `open Ocannl` (`lib/ocannl.ml`)
  - `Shape` - Shape inference and einsum notation
  - `Tensor` - Core tensor type and operations
  - `TDSL` - Tensor DSL with automatic differentiation (grad_spec: If_needed)
  - `NTDSL` - No-gradient tensor DSL (grad_spec: Prohibit_grad)
- There is no `PDSL` (Require_grad DSL). To build a differentiable leaf tensor with concrete values, pass the grad spec explicitly: `Operation.init ~l ~prec ~b ~o ~f ~grad_spec:Tensor.Require_grad ()` or `Tensor.term_init values ~grad_spec:Require_grad ()` (1-D); see `test/training/fused_classifier.ml`, `test/operations/primitive_ops.ml`
- Precision values: `Ir.Ops.single`, `Ir.Ops.double`, `Ir.Ops.half` (lowercase)
- Tensor printing in expect tests: `Tensor.print ~here:[%here] ~force:false ~with_code:false ~with_grad:false \`Inline tensor`
- Library sets in `dune` are per stanza, matching what the module references: a test that computes and prints typically wants `(libraries base ocannl stdio)`; a module with no `open Base` that prints nothing declares `(libraries ocannl)` alone; a test that asserts a verdict adds `arrayjit.verdict`. An unused `open` is a fatal warning here, and unused libraries are noise

### Pull Requests

- A **PR accomplishes a goal**: one thing that is true about the system afterwards and was not before, stated in its title. Scope it generously — carry the goal to its natural completion (the change, the tests that pin it, the docs it justifies, the follow-on cleanups it exposes) rather than opening three PRs
- A **commit is one move toward the goal**, not one slice by artifact type: the logic change, its tests, its `.expected` goldens and the doc or agent-note it justifies belong in one commit. A goal usually takes several moves, merged with a merge commit that preserves the series
- **Do NOT touch `CHANGES.md` in feature work** (gh-ocannl-807): the changelog is written in editorial passes from the durable records the work leaves (merge commits, PR bodies, issue closing comments); conventions in `docs/agent-notes/conventions.md`
- When you notice unrelated code smells or design problems, file separate issues
- Follow-up fixing commits are fine, and test-expectation promotions that span several topics can land in a final tests/promotions commit
- When creating commits, include the work summary in the commit message and credit yourself as a co-author
- Each commit should at least compile: loop `git checkout <rev> && dune build @check` over `git rev-list --reverse master..HEAD`
- **Bring the base in before opening the PR; a clean merge does not restart verification**: GitHub builds the PR's MERGE commit, so a repository-wide scan can be red there while green on your branch — rebase onto the staging `master` (or merge it in where the branch is shared). The merge gate is one green full PR-matrix run (Linux/macOS) for the PR's current head (roll-forward policy, gh-ocannl-861): any commit that moves the head waits for its own run; a diff CI's `docs/**` filter ignores merges without one; only the base having touched the PR's own files warrants bringing it in again (the note gives the diff). Master's CI after the merge belongs to the CI-red triage routine, not the merger
- **Two repositories, and remote names are not the contract**: development (branches, PRs, `master`) happens in `lukstafi/ocannl-staging`; `ahrefs/ocannl` is the public repo that owns the ISSUES cited as `gh-ocannl-NNN`, the milestones and the releases. Check `git remote -v` before trusting a name, add the other repo explicitly when needed, and pass `--repo <owner>/<name>` to every `gh` command: issues to `ahrefs/ocannl`, PRs to `lukstafi/ocannl-staging`

### Configuration

- See `ocannl_config.reference` for all settings. It ships with every setting COMMENTED OUT (`#key=…`, no space; prose comments use `# `), so copying it verbatim states nothing
- **Adding a config key touches two places**, enforced by `test/operations/test_config_consistency`: document it in `ocannl_config.reference` and register it in `Utils.known_config_keys`. Spell the key as a string literal at the call site (`~arg_name:"the_key"`); the scan fails non-literal uses outside the named lookup functions. New source files need no registration: the scans glob every directory that can read configuration (gh-ocannl-592, gh-ocannl-701). A key read only from a test is out of scope
- **Classify the key too**: `test/operations/digest_completeness` fails on a key with no entry in `Utils.config_key_classification` (gh-ocannl-572) — whether it reaches the schedule cache's identity, and which component
- **A new post-lowering module still needs registering** in that test's `codegen_stage_modules` list (a new backend, or anything reading configuration after lowering), so a `Code_borne` misclassification of its keys is noticed

**Configuration Methods** (in order of precedence):
1. Command-line flags: `--ocannl_<option>=<value>` (e.g., `--ocannl_backend=cuda`)
2. Environment variables: `OCANNL_<OPTION>=<value>` (e.g., `OCANNL_BACKEND=cuda`)
3. Config file: `ocannl_config` in current or ancestor directories

**Config profiles** (gh-ocannl-559): `profile=reproducible|performance|approximate` applies a preset bundle (embedded in `arrayjit/lib/utils.ml`) just below the explicit keys of the source that picked it, so explicit keys beat a profile of equal immediacy and a CLI-picked profile beats a config file; `test_config_consistency` checks the payloads and the reference file's quote of them. `approximate` (gh-ocannl-719) is `performance` plus every numerics-changing knob, gated in the benchmarks at `PARITY_TOL_APPROX`; a new numerics-changing gate lands in that payload, pinned at its default in `reproducible`, and in the schedule cache's identity in the PR that adds its key — the numerics digest for a codegen-time gate (`Keyed "numerics"`), the code digest for a lowering-time rewrite gate (`Code_borne`, gh-ocannl-483: the rewritten code carries the decision, and a numerics field would split cache entries for routines the rewrite never touches) (`test/operations/config_profiles` pins approximate ⊇ performance).

**Testing with Different Configurations**:

- Dune re-runs a test for an environment variable only where the stanza declares it as an `(env_var OCANNL_<KEY>)` dependency — add the declaration rather than working around the stale run
- **Warning**: `dune test --force` does NOT re-run expect tests (only rules with alias fields)
- For a one-off configuration change no stanza declares (notably for inline `%expect` tests): modify `test/config/ocannl_config` directly, run `dune clean`, or touch the affected test sources

**Important Debug Settings**:
- `output_debug_files_in_build_directory=true` - enables `build_files/` generation; files go to `build_files/<exe-name>/` (override with `build_files_prefix`; `build_files_prefix=.` for a flat layout)
- `debug_log_from_routines=true` - enables runtime logging from kernels aka. routines
- `debug_log_to_stream_files=true` - writes logs from kernels/routines to `log_files/<exe-name>/<backend>-<device>-<stream>.log`
- `clean_up_build_files_on_startup=false` and `clean_up_log_files_on_startup=false` - preserve debug files between runs
- CUDA routine logs may require `Utils.capture_stdout_logs` (see README)

**Available Backends**:
- `cc` (the default) combines cc_backend.ml with the scheduler `Sync` in schedulers.ml; kernel-level CPU parallelism is automatic (pool-rendered Grid loops)
- `multidev_cc` combines cc_backend.ml with the scheduler `Multidev`: multiple worker-domain CPU devices, for debugging multi-device parallel workflows ("sync_cc"/"multicore_cc" are deprecated aliases of cc/multidev_cc)
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
- `stretch 1.0` creates a shape-inferred constant 1 whose shape resolves at the use site; `1.0` alone is a fixed scalar. Operation results otherwise close down to their arguments' shapes — a use site broadcasts them in but cannot widen them (gh-544)
- Einsum spec must be a literal string when capturing dimensions: `x ++ "ab => a" ["b"]` works, `let s = "ab => a" in x ++ s ["b"]` fails
- Single-char vs multi-char mode: `"abc"` = 3 axes; `"abc,"` = 1 axis named `abc` (comma triggers multi-char)
- `{ param }` in `%op` creates learnable parameters; same syntax in `%cd` creates non-differentiable tensors
- Default param init is a centered scaled `uniform ()` over `[-0.25, 0.25)`, configurable via the reference `TDSL.default_param_init`
- Sub-modules with `()` must be bound before input: `let layer = make_layer () in fun x -> layer x`
- No reshape/flatten—use multi-axis operations or row variables instead

## Common Development Tasks

Touch-lists for adding a primitive operation, extending a backend, extending shape inference, and
diagnosing backend output discrepancies live in the `extending-ocannl` skill
(`.claude/skills/extending-ocannl/SKILL.md`). Debug-artifact and ppx_minidebug tracing recipes live
in the `ocannl-debug-tracing` skill (`.claude/skills/ocannl-debug-tracing/SKILL.md`). Agents
without skill support read the files directly. In brief:

- New primitive ops: `arrayjit/lib/ops.ml` (+ `Ir.Ops`), wired into `tensor/operation.ml`
- New tensor convenience functions: `tensor/operation.ml` (use `%cd` for forward/backprop)
- Shape/projection changes: `tensor/shape.ml`, `tensor/row.ml`, `arrayjit/lib/indexing.ml`
- Algebraic rewrites over raw lowered code (pattern-directed substitutions that change the computation, each behind its own config key): a member of the `arrayjit/lib/rewrites.ml` tier, which `Assignments.lower` runs to a fixpoint ahead of the analyses; `online_softmax.ml` is the exemplar, design record `docs/proposals/gh-ocannl-483.md`

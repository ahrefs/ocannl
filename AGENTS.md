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
- `docs/`: slides, reference docs and `docs/design/` records.
- `archive/`: frozen proposals, not guidance.
- `build_files/` and `log_files/`: generated artifacts when debug settings are enabled.

Key reference files: `docs/syntax_extensions.md` (%op/%cd), `docs/shape_inference.md` (shape/projection inference), `arrayjit/lib/context.mli` (the context-based runtime API), `ocannl_config.reference` (every configuration key and default).

## Build Commands

```bash
# Build all packages; this also runs the cram-style test executables
dune build

# Only compile -- do not run any executable
dune build @check

# Build specific package
dune build -p neural_nets_lib
dune build -p arrayjit

# Run tests
dune runtest

# Run tests on a specific backend (bash syntax)
OCANNL_BACKEND=cuda dune runtest

# Install dependencies (OCaml >= 5.3); --with-dev-setup adds the pinned ocamlformat and ocaml-lsp-server
opam install . --deps-only --with-test --with-dev-setup

# Install with optional backends
opam install cudajit  # for CUDA backend
opam install hipjit   # for AMD HIP backend
```

**Dependency pins**: before changing a shared switch, test a scratch install against master and the consumer branch (build-and-test.md#testing-a-dependency-candidate-without-changing-the-switch).

**Worktrees**: nested ones (`.claude/worktrees/`) need a `dune-workspace` at their root (else dune builds the PARENT checkout), which Claude Code’s setup hooks write (without hooks, run `scripts/setup-ocaml-env.sh --switch` in the worktree); to retire one after its merge, ship-pr's `post-merge-cleanup.sh` takes `--regenerable _build` (far-side cleanup set too: build-and-test.md#dune-mechanics).

**Windows shells**: use **Git Bash** (MSYS), never a Cygwin bash, and source `tools/opam-env.sh` before building (`opam env`'s cygwin-style paths break linking). Route dune through `tools/dune-quiet.sh`, which filters benign binutils link warnings and keeps dune's exit status (gh-ocannl-662; build-and-test.md#windows-ci tells the two bashes apart).

**Windows verification placement**: hosted Windows CI runs on schedule, not per PR; a merge never waits on that sweep (a dispatched run does gate the head). When a change needs Windows signal that matters (a Windows fix, or behavior only Windows exercises: line endings, golden float formatting, mingw), first reboot rog into Windows yourself, under your own exclusive `measurement` reservation, and always back (exit 3, `NEEDS A PERSON`, goes to the user at once); failing that, ask the user to boot `rog-nv-win` or `minix-amd-win`, naming the box, commit and check (a wave coordinator takes both steps). After 30 unanswered minutes, tell the user through the same channel that CI is taking the check, and only then dispatch `ci.yml` with `windows_only: true` and the full `expected_sha`, citing the run — never while the request is open. An issue centered on Windows waits for a box rather than iterating through dispatches. Why, the `wake-lab.sh` reboot and running on a booted box: build-and-test.md#windows-ci.

**Format before the first push** (gh-ocannl-938): CI's `fmt` job runs `tools/fmt-check.sh` (`@fmt` + invalid odoc). Order: `dune fmt` (its own step, finished before the test batch and outside its command), tests, promote (reformatting shifts `~here` goldens' `file:line`), last `tools/fmt-check.sh` (read its exit status or final `fmt-check:` verdict line, never a status piped through `tail`). New ppx-expectation files (`test/ppx/*_expected.ml`) stay unformatted — list them in `.ocamlformat-ignore` (`ocamlformat_ignore_scan` enforces it).

## Architecture Overview

**Before working on a subsystem, read the matching file under `docs/agent-notes/`** (index
`docs/agent-notes.md`) — traps, known bugs with workarounds, debug recipes and design history the
code alone does not show.

### Key Concepts

1. **Dual Syntax Extensions**:
   - `%cd` ("code"): For assignment computations (`Assignments.comp`)
   - `%op` ("operation"): For tensor expressions (`Tensor.t`)
   - Inline declarations lift to unit parameter `()` scope, enabling parameter reuse

2. **Shape Inference** (`docs/shape_inference.md` is the authoritative reference):
   - Three axis kinds — batch | input -> output (matrix convention: input rightmost) — with row variables (`..d..`) for broadcasting and generalized einsum notation for convolutions, reductions, and arbitrary permutations
   - "Principle of least commitment": use row variables where axis count doesn't matter
   - Shape inference completion is forced by lowering: via `Context.compile`, or wrappers such as `Train.to_routine`, `Train.run_once` or `Train.forward_once`; `finish_inference` closes still-unsolved dims (GLB where known, otherwise 1/broadcast)
   - Operations in `Operation`, `TDSL`, `NTDSL` return `Tensor.op_fun`s, so call sites can specify shapes; `TDSL.O` (opened for `%op`) and `NTDSL.O` (for `%cd`) hide this, so shapes are inferred

3. **Memory Management**: Tensor node memory modes are `Virtual` (inlined computations), `Local`, and `On_device`.
   CPU-side reads and writes are explicit, context-mediated operations
   (`Context.to_host`/`from_host`, `get_values`/`set_values`).

## Development Workflow

### Testing

The mechanics, traps and authoring recipes behind every rule here are in `docs/agent-notes/build-and-test.md`; read its matching `##` section before authoring a new scan, alias family, slow test or golden format.

- Tests are either inline `%expect` expectations (`ppx_expect`) or standalone executables whose stdout is compared against an `.expected` golden via Dune's `test` stanza; the two are exclusive within one test
- `.expected` tests are easier to debug — use them for new features. Tutorial `%expect` files in `test/` double as documentation and integration tests; use them only when the outputs are illustrative

**Running tests**:
- `dune runtest` runs everything except the training integrations (`@train`) and the slow runs (`@slow`); `dune runtest test/operations/` scopes to one directory
- Avoid `dune exec test/.../test_name.exe` for standalone tests: it finds no config or `Context.auto` silently picks a GPU (build-and-test.md#dune-mechanics). Run one test, or one inline `%expect` module (gh-ocannl-1037), by its alias — `dune build @test/operations/runtest-<name>` — which applies the `.expected` diff (promotable; a misspelled name exits 1) and leaves `_build/default/<dir>/<name>.exe.output` to inspect. For `bin/` executables pin `OCANNL_BACKEND=...` explicitly
- A test written as an `(executable)` plus a golden-diff `(rule)` carries a hand-written `(alias runtest-<name>)` of its own (gh-ocannl-726), aggregated back into the directory's `runtest`; `env_var_deps` enforces the naming, aggregation and the ambient gate
- The repo-wide scans (the `(name scans)` stanza in `test/operations/dune`) share `dune build @test/operations/scans`, which `tools/fmt-check.sh` builds after the formatter (`OCANNL_TOOL_FMT_CHECK_SCANS=0` skips). Mid-change, run it alone after touching a config key, a dune stanza, a printed claim, an agent note, a new script or source file, an emitted-text pin or a reference to a library value or type, test-only edits included; focused aggregates: `@metal-codegen`, `@lifecycle` (build-and-test.md#repository-scans). For shell changes also run `tools/test-harnesses.sh --affected --since BASE`, the command `prepush` closes with and does not run: the harness commands `affected --harnesses` prints; the full set, `tools/test-harnesses.sh` alone, is CI's. Either is a test batch like any other, taking its `--cpu` slot itself, foreground or detached (build-and-test.md#the-harness-shells-and-processes)
- The ambient `OCANNL_*` variables reach every rule through its `ocannl_config` dep, so `OCANNL_<KEY>=… dune build @…` reruns what it changes; a directory checking in its own config also lists `ocannl_ambient` (`config_dep_completeness` enforces both), and `env_var_deps` refuses an `(env_var OCANNL_<KEY>)` that record carries (build-and-test.md#dune-mechanics). `OCANNL_BACKEND` is the exception: a stanza that can select a backend declares `(env_var OCANNL_BACKEND)` (uppercase only); one that names its backend, or links none, instead carries `; ocannl-backend: <backend>[,<backend>…] -- <reason>` inside its own parentheses (several only where it honestly names them) — exactly one of the two (gh-ocannl-659), on the rule of an `(executable)` plus `(rule)` pair; `env_var_deps` enforces this and its neighbours (grammar: build-and-test.md#in-place-markers)
- Config startup chatter goes to stderr, never into stdout or `.expected` goldens; `--ocannl_log_config_sourcing=true` traces where each setting came from. A backend-uniform golden cannot confirm which backend a probe ran on — read stderr, or have the test print the backend (gh-ocannl-622)
- Every `(test)`/`(tests)` stanza, every `(library)` with `(inline_tests)`, and every `(rule)` that runs a test executable lists `ocannl_config` in its `(deps ...)` — nothing is sandboxed, so a missing one makes the run order-dependent (gh-ocannl-586); `config_dep_completeness` enforces it, except for its named exemption list of rules whose program reads no configuration — extend that list rather than adding a meaningless dep
- A `(test)` stanza automatically diffs the `<name>.expected` beside it; the explicit rule-plus-diff pattern is only for tests no `(test)` stanza runs, such as the `@slow` rules
- Run suites through `tools/test-run.sh`, never hand-rolled shell around dune (build-and-test.md#the-harness-shells-and-processes): it runs dune unpiped, capped and logged, exiting with dune's status (65 is `BUILD FAILED`); a GPU batch also caps each test (not `@slow`; `--test-cap N`, `0` lifts it) and never starts on a hung device query (`DEVICE UNHEALTHY`, exit 69; build-and-test.md#gpu-boxes-job-caps-and-runtime-refusals). Never write a sleep/`pgrep` waiter loop: launch through the harness's background execution and act on the notification. A native wave worker follows its brief's `execution slot --bg` plus `bg-run.sh wait` form; `wait last` is only for a run launched without `--bg`. Use `start` then `wait last` only for a run that must outlive the session, and a run is found by `list`/`paths run last`, never by position in `~/.ocannl-test-runs`. `repeat --direct`, `fresh` and `bisect GOOD BAD` are its other verbs (build-and-test.md#dune-mechanics, build-and-test.md#bisecting-a-regression)
- **A GPU suite runs width-capped**, per box, boot and backend, cc too: wider, it comes back red where a real regression would (the widths: build-and-test.md#gpu-boxes-job-caps-and-runtime-refusals). `tools/test-run.sh run` injects the tightest cap of any backend the batch can hold, not just `OCANNL_BACKEND`'s (`plan <args>` shows it). On a fleet box `run`/`start` takes the fleet's run-time slot itself, `--cpu` only when none of those backends holds a GPU; `SLOT REFUSED` (exit 75) is the slot's verdict, not the suite's
- From PowerShell, QUOTE dune alias targets (`dune build "@runtest" "@slow"`): an unquoted `@word` splats to nothing and the command degrades to a false-green plain build
- Scope test runs to what a change can reach (build-and-test.md#aliases-and-goldens): `tools/test-run.sh affected --since <base-branch>` lists the per-test aliases the changed files reach, link-only ones last; `--referenced-only` is otherwise only for iterating mid-change. Before pushing run `tools/test-run.sh prepush --since <base-branch> [--start] [build -j 4]` as one launch -- the default list without its `slow-`/`train-` members: CI's `@train` shards run the `train-` ones, no CI job runs `@slow`, so run the `slow-` ones it names that the change exercises -- then `dune build @check`. When `affected` prints CROSS-CUTTING, `prepush` refuses: the PR's CI run owns the list but its `slow-` aliases, and the local run is `prepush --referenced-only --since <base-branch> [--start] [build -j 4]` (only the first tier, if narrow), the full list's applicable `slow-` aliases it names, and tests exercising the change, named in the PR body. Before broad testing of a config-gated path, grep whether any test or test config enables the gate; the full regular/slow suites are for cross-cutting changes
- Keep library sources unchanged while Dune is running; edits invalidate in-flight rules

**Training integration runs (the `train` alias)**: the toy integrations (`bigram`, `mlp_names`, `circles_conv`, `fsm_transformer`, ...) serialize on the training lock, so they live off the `runtest` path: `dune build @train`, one at a time via `@test/training/train-<name>`. Per-PR CI runs them as two ubuntu shards (macOS runs only `main` there; build-and-test.md#what-ci-actually-covers), the daily sweep on every backend; after touching training dynamics, `Train.*` plumbing or the autotuner's fission path, run the affected members locally before pushing

**Slow training tests (the `slow` alias)**: excluded from `dune runtest`; `dune build @slow`, one at a time via `@test/training/slow-<name>`. `dune build @check` compiles them. Regular and `@slow` training actions share the global `/ocannl_training_test` Dune lock so their OpenMP pools never overlap — keep it on new ones. Gating recipe: build-and-test.md#aliases-and-goldens; `test/training/dune` is the pattern

**Test types and authoring**:
- `dune promote` accepts golden changes. On Windows, and on EVERY platform during a merge, promote through `tools/promote.sh`, which stages what it promoted and on Windows strips CRLF (build-and-test.md#promotion-and-formatting)
- **A test that decides its own verdict reports it through `Verdict`** (`test/support/verdict.ml`; add `arrayjit.verdict` to `(libraries ...)` — the bare name fails under `dune build -p`), preferably via `open Verdict.Claims`. Over a collection use the quantified combinators (`p_all`, `p_none`, `p_exists`, ...), never `p` over a `List.for_all` (vacuously true when empty, gh-ocannl-729) — unless emptiness is the passing case, said so at the site. Phrase every claim so `true` passes (a failed claim exits 1, so it cannot be promoted into the golden). Claims only some backends can evaluate are written once, in a leg (`Verdict.leg`, `in_leg`), never as a skip list restating them (gh-ocannl-1703). Skip aggregation: build-and-test.md#skip-coverage; `verdict_ratchet`'s shapes: build-and-test.md#verdict-claims
- **A float a device reduction produced stays out of stdout goldens**, at any precision (build-and-test.md#floats-and-measurements-in-goldens). Print its digits to stderr tagged `(not part of the golden)` and put a TWO-SIDED `Verdict` claim on stdout (gh-ocannl-725). Floats exact by construction (thresholds, power-of-two scales, closed forms, dyadic sums) stay on stdout
- A few `%expect_test` blocks capture backtraces that hard-code `file:line`; an edit that shifts lines there forces a benign re-promote — do it in the same shell as the failing run
- For optimizer passes that change *what value a cell holds* (virtualization guards, index solving, init elision), a structural test is necessary but NOT sufficient — also assert executed output against a materialized run, with every producer DISCRIMINATING: varying with every symbol of its iteration, clear of the init value (`tick`/`tag` in `test/operations/virtual_diagonal.ml`); a zero transform's test prefills what it zeroes (`Ll_test.run_linked`'s default), and under `test/config`'s `debug_fill_fresh_outputs` a red naming a NaN is a missing write or a read of a cell no writer covers (build-and-test.md#hand-built-ir-tests)
- **A test that asserts on generated code reads it through `Test_utils.Generated`** (`test/support/generated.ml`), never by opening `build_files/` itself (why: build-and-test.md#codegen-text-inventory). `Generated.init ~backend_name` before the first compile, then `assert_emits`/`read`; `arm` before each compile that reuses a routine name; gate a leg the backend cannot evaluate with `Verdict.skipped`
- **Pin the relationship, not the restatement**: a check that needs a set another part of the system owns derives it, or asserts the two equal from where the link cost is already paid — never a second copy asserted to still say what it says. Shapes, exemplars and exceptions (judgment lists, deliberately independent constants): build-and-test.md#pinning-the-relationship
- Backend codegen snapshots (`.cu.expected` etc.) go stale when codegen changes land without that hardware — expect to re-promote them when the hardware next runs the suite
- **Before changing code generation**, run `dune build @test/operations/runtest-codegen_text_inventory`: its golden lists every file in `test/` and `arrayjit/test/` pinning the TEXT of emitted code with the aliases re-running them, then the tests compiling kernel text they built themselves: on each GPU backend run both the applicable text-golden routes and the kernel-execution list. The GPU alias section lists only kernel-execution routes; it does not replace the earlier golden routes (for example, `micrograd_demo_logging`). Before changing a backend's builtins prelude or compiler flags, run that backend's list of tests calling its compiler from the same golden. Reach emitters through a qualifier, never an `open` (gh-ocannl-748; build-and-test.md#codegen-text-inventory)
- **Test placement**: always under `test/` — default `test/operations`, complex einsum specs in `test/einsum`, training loops in `test/training` — with `ocannl_config` in `(deps ...)`, the backend declaration or marker, and an (initially empty) `.expected`. Tests that hand-build `Ir.Low_level.t` share `test/support/ll_test.ml`. Within one executable, keep routine names distinct (the `af_`/`ops_`/`smem_` prefixes): a same-named routine overwrites the earlier one's `build_files/` artifacts

**Windows portability for `.expected` tests**: goldens are LF (`.gitattributes`); edit them with bash tools, never PowerShell `Set-Content`/`Out-File` (CRLF). Print golden floats with the `test_utils` portable printers (plus `set_binary_stdout`) or `Ir.Ndarray.concise_float ~prec`, not `%g`/`%e` — and avoid decimal ties (Windows and glibc round them differently; build-and-test.md#windows-portability), or print with `hex_float`/`%h`

**Module Paths and Common APIs**:

- **For files outside OCANNL implementation (tests, examples, user code), start with `open Ocannl.Nn_blocks.DSL_modules`** — the DSLs of `Ocannl.Operation.DSL_modules` (end of `tensor/operation.ml`) extended with the non-uniform initializers (`normal`, `kaiming`, `xavier`, their `_at` variants, ...; `lib/nn_blocks.ml`). `open Ocannl.Operation.DSL_modules` is the narrower open for a file that uses no initializer
- Available after either open: `Ir` (low-level IR: Ndarray, Ops, Tnode, ...), `Shape` (shape inference, einsum notation), `Tensor`, `TDSL` (autodiff; grad_spec If_needed), `NTDSL` (no gradients; Prohibit_grad). `Row` (row variables) is exported by `Operation.DSL_modules` only, so under the `Nn_blocks` open it comes from `open Ocannl` (`lib/ocannl.ml`)
- There is no `PDSL` (Require_grad DSL). To build a differentiable leaf tensor with concrete values, pass the grad spec explicitly: `Operation.init ~l ~prec ~b ~o ~f ~grad_spec:Tensor.Require_grad ()` or `Tensor.term_init values ~grad_spec:Require_grad ()` (1-D); see `test/training/fused_classifier.ml`, `test/operations/primitive_ops.ml`
- Precision values: `Ir.Ops.single`, `Ir.Ops.double`, `Ir.Ops.half` (lowercase)
- Tensor printing in expect tests: `Tensor.print ~here:[%here] ~force:false ~with_code:false ~with_grad:false \`Inline tensor`
- Library sets in `dune` are per stanza, matching what the module references (typically `(libraries base ocannl stdio)`). An unused `open` is a fatal warning here, and unused libraries are noise

### Pull Requests

- A **PR accomplishes a goal**: one thing that is true about the system afterwards and was not before, stated in its title. Scope it generously — carry the goal to its natural completion (the change, the tests that pin it, the docs it justifies, the follow-on cleanups it exposes) rather than opening three PRs
- A **commit is one move toward the goal**, not one slice by artifact type: the logic change, its tests, its `.expected` goldens and the doc or agent-note it justifies belong in one commit. A goal usually takes several moves, merged with a merge commit that preserves the series
- **Do NOT touch `CHANGES.md` in feature work** (gh-ocannl-807): the changelog is written in editorial passes from the durable records the work leaves (merge commits, PR bodies, issue comments); conventions in `docs/agent-notes/conventions.md`
- When you notice unrelated code smells or design problems, file separate issues
- Follow-up fixing commits are fine, and test-expectation promotions that span several topics can land in a final tests/promotions commit
- When creating commits, include the work summary in the commit message and credit yourself as a co-author
- Each commit should at least compile: loop `git checkout <rev> && dune build @check` over `git rev-list --reverse master..HEAD`
- **Bring the base in before opening the PR; after that, a green head merges as it is**: rebase onto the staging `master` (or merge it in where the branch is shared) while no run exists yet, since GitHub builds the PR's MERGE commit. The merge gate is one green PR-matrix run for the PR's current head (gh-ocannl-861; the macOS leg is advisory, `.github/ship-pr-advisory-checks`): a commit that moves the head waits for its own run, documentation edits included; afterwards move the head only for a textual conflict (build-and-test.md#bringing-the-base-in). Master's CI after the merge belongs to the CI-red triage routine, not the merger (build-and-test.md#what-ci-actually-covers)
- **Codex reviews every staging PR unrequested**: it starts on the PR's opening (usually on each later push too), so observe the round with ship-pr's `watch` rather than asking for one
- **Two repositories, and remote names are not the contract**: development (branches, PRs, `master`) happens in `lukstafi/ocannl-staging`; `ahrefs/ocannl` owns the ISSUES cited as `gh-ocannl-NNN`, the milestones and the releases. Check `git remote -v` before trusting a name, and pass `--repo` to every `gh` command: issues to `ahrefs/ocannl`, PRs to `lukstafi/ocannl-staging`

### Configuration

- See `ocannl_config.reference` for all settings. It ships with every setting COMMENTED OUT (`#key=…`, no space; prose comments use `# `), so copying it verbatim states nothing
- **Adding a config key**: document it in `ocannl_config.reference` and register it in `Utils.known_config_keys` (`test/operations/test_config_consistency`; a key read only from a test is out of scope), spelled as a string literal at the call site (`~arg_name:"the_key"`; the scan fails non-literal uses outside the named lookup functions). Classify it in `Utils.config_key_classification` — whether it reaches the schedule cache's identity, and which component (`test/operations/digest_completeness`, gh-ocannl-572) — and register a new post-lowering module (a new backend, or anything reading configuration after lowering) in that test's `codegen_stage_modules`, which notices a `Code_borne` misclassification of its keys

**Configuration Methods** (in order of precedence):
1. Command-line flags: `--ocannl_<option>=<value>`
2. Environment variables: `OCANNL_<OPTION>=<value>`
3. Config file: `ocannl_config` in current or ancestor directories

**Config profiles** (gh-ocannl-559): `profile=reproducible|performance|approximate` applies a preset bundle (`arrayjit/lib/utils.ml`) just below the explicit keys of the source that picked it (precedence: the head of `ocannl_config.reference`). A new numerics-changing gate lands in the `approximate` payload (gh-ocannl-719), pinned at its default in `reproducible`, and in the schedule cache's identity in the PR that adds its key — which digest, and the checks that pin the payloads: backend-precision-and-simd.md#numerics-policies-and-cache-identity.

**Testing with Different Configurations**:

- **Warning**: `dune test --force` does NOT re-run expect tests (only rules with alias fields)
- For a one-off configuration change, set `OCANNL_<KEY>` on the run (it reruns every reader, inline `%expect` tests included) or edit `test/config/ocannl_config`. A key a module-level lazy holds keeps its first value: state it outside the process (build-and-test.md#process-lifetime-settings)

**Important Debug Settings**:
- `output_debug_files_in_build_directory=true` - enables `build_files/` generation; files go to `build_files/<exe-name>/` (override with `build_files_prefix`; `build_files_prefix=.` for a flat layout)
- `debug_log_from_routines=true` - enables runtime logging from kernels aka. routines; emits nothing unless `log_level > 1`, and needs `debug_log_to_stream_files=true` in a test with a stdout golden: without it a default build drops the lines on cc/Metal and CUDA/HIP print them to stdout
- `debug_log_to_stream_files=true` - writes logs from kernels/routines to `log_files/<exe-name>/<backend>-<device>-<stream>.log`
- `clean_up_build_files_on_startup=false` and `clean_up_log_files_on_startup=false` - preserve debug files between runs
- CUDA/HIP routine logs need `Utils.capture_stdout_logs` unless `debug_log_to_stream_files=true` (see README)

**Available Backends**:
- `cc` (the default) combines cc_backend.ml with the scheduler `Sync` in schedulers.ml; kernel-level CPU parallelism is automatic (pool-rendered Grid loops)
- `multidev_cc` combines cc_backend.ml with the scheduler `Multidev`: multiple worker-domain CPU devices, for debugging multi-device parallel workflows ("sync_cc"/"multicore_cc" are deprecated aliases of cc/multidev_cc)
- `cuda`, `hip` (AMD ROCm/HIP, mirroring CUDA) and `metal`: cuda_backend.ml, hip_backend.ml, metal_backend.ml

Backends are process-wide singletons: use `Backends.get_backend ()` or the Context API (`arrayjit/lib/context.mli`); `fresh_backend` is retired. Merge buffers (`.merge`) support stream-to-stream reductions in `%cd`.

### Backend Development

- Backends implement stream-based execution with FIFO queuing, events, and synchronization between streams/devices, generating code from `Low_level.t`
- Code generation: `c_syntax.ml` is a functor with default C patterns that each backend overrides for its own syntax; the touch-lists (builtins modules, the `convert_precision` obligation) are in the `extending-ocannl` skill

### Syntax Extensions

`docs/syntax_extensions.md` is the authoritative reference for `%op`/`%cd` (record syntax, scoping, einsum specs and capture, projection slots). Orientation and traps:

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

Touch-lists (a primitive operation, a backend, shape inference, backend output discrepancies) live
in the `extending-ocannl` skill (`.claude/skills/extending-ocannl/SKILL.md`), debug-artifact and
ppx_minidebug tracing recipes in `ocannl-debug-tracing` (`.claude/skills/ocannl-debug-tracing/SKILL.md`);
agents without skill support read the files directly. In brief:

- New primitive ops: `arrayjit/lib/ops.ml` (+ `Ir.Ops`), wired into `tensor/operation.ml`
- New tensor convenience functions: `tensor/operation.ml` (use `%cd` for forward/backprop)
- Shape/projection changes: `tensor/shape.ml`, `tensor/row.ml`, `arrayjit/lib/indexing.ml`
- Algebraic rewrites over raw lowered code (pattern-directed substitutions that change the computation, each behind its own config key): a member of the `arrayjit/lib/rewrites.ml` tier (lowering-and-analysis.md); `online_softmax.ml` is the exemplar, design record `docs/design/gh-ocannl-483.md`

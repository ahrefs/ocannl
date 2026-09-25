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

**Windows verification placement**: hosted Windows CI runs on its schedule, not per PR, and a merge never waits on that scheduled sweep (a dispatched run is another matter: it gates the head). When a change needs Windows signal that matters (it fixes a Windows failure, or changes behavior only Windows exercises: line endings, float formatting in goldens, the mingw toolchain), first ask the user to boot `rog-nv-win` or `minix-amd-win` into Windows, naming the box, commit and check (under a wave coordinator, hand the request to it). With no answer in 30 minutes, tell the user through the same channel that CI is taking the check, and only then dispatch `ci.yml` with `windows_only: true` and the full `expected_sha`, citing the run — never while the request is open. An issue centered on Windows waits for a box rather than iterating through dispatches. Why, and how to run on a booted box: the build-and-test note.

**Format before the first push** (gh-ocannl-938): CI's `fmt` job runs `dune build @fmt` on every PR; a formatting-only fix push costs a CI round AND a review round. Order: `dune fmt`, then the test run, then promote (reformatting shifts the `file:line` that `~here` goldens embed). New ppx-expectation files (`test/ppx/*_expected.ml`) stay unformatted — list them in `.ocamlformat-ignore` (`ocamlformat_ignore_scan` enforces it).

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
- Avoid `dune exec test/.../test_name.exe` for standalone tests: it finds no config (the root `ocannl_config` is gitignored) or `Context.auto` silently picks a GPU. Run one test by its alias — `dune build @test/operations/runtest-<name>` — which applies the `.expected` diff (promotable; a misspelled name exits 1) and leaves `_build/default/<dir>/<name>.exe.output` to inspect. For `bin/` executables pin `OCANNL_BACKEND=...` explicitly
- A test written as an `(executable)` plus a golden-diff `(rule)` carries a hand-written `(alias runtest-<name>)` of its own (gh-ocannl-726), aggregated back into the directory's `runtest`; `test/operations/env_var_deps` enforces the alias naming, aggregation and the ambient gate. Dune >= 3.20 is the declared floor
- The repo-wide scans (list: the `(name scans)` stanza in `test/operations/dune`) share `dune build @test/operations/scans`, which runs in seconds. Run it before pushing a change to a config key, a dune stanza, a printed claim, an agent note, or any new script or source file. Focused aggregates beside it: `dune build @metal-codegen`, `dune build @lifecycle`
- Pinning a variable on a run (`OCANNL_BACKEND=cuda dune build @.../runtest-<name>`) reaches only stanzas that DECLARE it; an undeclared one serves the previous run's result as a pass. A stanza that can select a backend declares `(env_var OCANNL_BACKEND)` (uppercase only); one that names its backend, or links none, instead carries `; ocannl-backend: <none|cc|multidev_cc|cuda|hip|metal>[,<backend>…] -- <reason>` inside its own parentheses (comma-separated where it honestly names several) — exactly one of the two (gh-ocannl-659), on the rule of an `(executable)` plus `(rule)` pair. Other keys go into `(deps ...)` as `(env_var OCANNL_<KEY>)`. `env_var_deps` enforces this and its neighbours (tracing gates, ambient guards, `Generated.init` callers); the build-and-test note has the grammar
- Config startup chatter goes to stderr, so stdout stays a clean data channel and `.expected` goldens never see it; `--ocannl_log_config_sourcing=true` traces where each setting came from. A backend-uniform golden cannot confirm which backend a probe ran on — read stderr, or have the test print the backend (gh-ocannl-622)
- Every `(test)`/`(tests)` stanza, every `(library)` with `(inline_tests)`, and every `(rule)` that runs a test executable lists `ocannl_config` in its `(deps ...)` — nothing is sandboxed, so a missing dep makes the run order-dependent (gh-ocannl-586); `config_dep_completeness` enforces it, except for its named exemption list of rules whose program reads no configuration — extend that list rather than adding a meaningless dep
- A `(test)` stanza automatically diffs the `<name>.expected` beside it; the explicit rule-plus-diff pattern is only for tests no `(test)` stanza runs, such as the `@slow` rules
- Run suites through `tools/test-run.sh`, never hand-rolled shell around dune: `tools/test-run.sh run runtest test/operations`, `tools/test-run.sh run build @slow`. It runs dune unpiped (piping masks promotion diffs), capped, logged with an `exit: N` sentinel, and exits with dune's status. Never write a sleep/`pgrep` waiter loop — launch through the harness's background execution and act on the notification (`start`, then `wait last`, only for a run that must outlive the session). `repeat N build @<dir>/runtest-<name>` genuinely re-executes an unchanged `.expected` test N times, to sample a timing-dependent golden
- **A GPU suite runs width-capped**: `-j 2` on a WSL2 box (`rog-nv-wsl`, `minix-amd-wsl`); on a native boot, per slot, `-j 4` for hip on minix, `-j 8` for hip on tuf and cuda on rog (gh-ocannl-1033). At dune's default width the device bridge or copy-engine pool overflows and the suite comes back red in exactly the stanzas a real backend regression lands in. `tools/test-run.sh run` injects the cap from a GPU `OCANNL_BACKEND` in the environment; with the backend only in `ocannl_config`, pass `-j` yourself
- From PowerShell, QUOTE dune alias targets (`dune build "@runtest" "@slow"`): an unquoted `@word` splats to nothing and the command degrades to a false-green plain build
- Scope test runs to what a change can reach: directory aliases first (`@test/operations/runtest`, `@test/einsum/runtest`, `@arrayjit/runtest`, ...), then `dune build @check`. Before broad testing of a config-gated path, grep whether any test or test config enables the gate; reserve the full regular/slow suites for cross-cutting changes
- Keep library sources unchanged while Dune is running; edits invalidate in-flight rules

**Training integration runs (the `train` alias)**: the toy integrations (`bigram`, `mlp_names`, `circles_conv`, `fsm_transformer`, ...) serialize on the training lock, so they live off the `runtest` path: `dune build @train`, one at a time via `@test/training/train-<name>`. Per-PR CI runs them only on macOS, the daily sweep on every backend; after touching training dynamics, `Train.*` plumbing or the autotuner's fission path, run the affected members locally before pushing

**Slow training tests (the `slow` alias)**: excluded from `dune runtest`; `dune build @slow`, one at a time via `@test/training/slow-<name>`. `dune build @check` compiles them. Regular and `@slow` training actions share the `ocannl_training_test` Dune lock so their OpenMP pools never overlap — keep it on new ones. The gating recipe is in the agent note; `test/training/dune` is the pattern

**Test types and authoring**:
- Inline tests are files in a library's `modules` field with an `inline_tests` stanza, run by `dune runtest`, never `dune exec`
- `dune promote` accepts golden changes. On Windows, and on EVERY platform during a merge, promote through `tools/promote.sh`: mid-merge, a golden promoted after its `git add` gets committed with the pre-promotion content — the script stages what it promoted; on Windows it also strips CRLF
- **A test that decides its own verdict reports it through `Verdict`** (`test/support/verdict.ml`; add `arrayjit.verdict` to `(libraries ...)` — the bare name fails under `dune build -p`), preferably via `open Verdict.Claims`. Over a collection use the quantified combinators (`p_all`, `p_none`, `p_exists`, ...), never `p` applied to a `List.for_all`, vacuously true when empty (gh-ocannl-729) — unless emptiness is the passing case, said so at the site. Phrase every claim so `true` passes; a failed claim exits 1, so it cannot be promoted into the golden. Skip aggregations and the shapes `verdict_ratchet` enforces: the build-and-test note
- **A float a device reduction produced stays out of stdout goldens** — lowering the precision only moves the rounding tie across backends. Print its digits to stderr tagged `(not part of the golden)` and put a TWO-SIDED `Verdict` claim on stdout (gh-ocannl-725). Floats exact by construction (thresholds, power-of-two scales, closed-form schedules, small dyadic sums) stay on stdout
- A few `%expect_test` blocks capture backtraces that hard-code `file:line`; an edit that shifts lines there forces a benign re-promote — do it in the same shell as the failing run
- For optimizer passes that change *what value a cell holds* (virtualization guards, index solving, init elision), a structural test is necessary but NOT sufficient — also assert executed output against a materialized run, with every producer DISCRIMINATING: varying with every symbol of its iteration, clear of the init value (`tick`/`tag` in `test/operations/virtual_diagonal.ml`)
- **A test that asserts on generated code reads it through `Test_utils.Generated`** (`test/support/generated.ml`), never by opening `build_files/` itself — artifacts outlive the run and same-named routines overwrite each other. `Generated.init ~backend_name` before the first compile (declare `(env_var OCANNL_BUILD_FILES_PREFIX)`), then `assert_emits`/`read`; `arm` before each compile that reuses a routine name; gate a leg the backend cannot evaluate with `Verdict.skipped`
- **Pin the relationship, not the restatement**: a check that needs a set another part of the system owns derives it, or asserts the two equal from where the link cost is already paid — never a second copy asserted to still say what it says. Shapes, exemplars and exceptions (judgment lists, deliberately independent constants) are in the agent note
- Backend codegen snapshots (`.cu.expected` etc.) go stale when codegen changes land without that hardware — expect to re-promote them when the hardware next runs the suite
- **Before changing code generation**, run `dune build @test/operations/runtest-codegen_text_inventory`: its golden enumerates every file that pins the TEXT of emitted code, in both `test/` and `arrayjit/test/` (gh-ocannl-712). Reach emitters through a qualifier, never an `open` (gh-ocannl-748)
- **Test placement**: always under `test/` — default `test/operations`, complex einsum specs in `test/einsum`, training loops in `test/training` — with `ocannl_config` in `(deps ...)`, the env-var declaration or backend marker, and an (initially empty) `.expected`. Tests that hand-build `Ir.Low_level.t` share `test/support/ll_test.ml`. Within one executable, keep routine names distinct (the `af_`/`ops_`/`smem_` prefixes): a same-named routine overwrites the earlier one's `build_files/<exe-name>/` artifacts

**Windows portability for `.expected` tests**: goldens are LF (`.gitattributes`); promote through `tools/promote.sh` and edit goldens with bash tools, never PowerShell `Set-Content`/`Out-File` (CRLF). Print golden floats with the `test_utils` portable printers (plus `set_binary_stdout`) or `Ir.Ndarray.concise_float ~prec`, not `%g`/`%e` — and avoid decimal ties, which Windows rounds away from zero and glibc to even, or print with `hex_float`/`%h`

**Module Paths and Common APIs**:

- **For files outside OCANNL implementation (tests, examples, user code), start with `open Ocannl.Nn_blocks.DSL_modules`** — the DSLs of `Ocannl.Operation.DSL_modules` (end of `tensor/operation.ml`) extended with the non-uniform initializers (`normal`, `kaiming`, `xavier`, their `_at` variants, ...; `lib/nn_blocks.ml`). `open Ocannl.Operation.DSL_modules` is the narrower open for a file that uses no initializer
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
- Library sets in `dune` are per stanza, matching what the module references (typically `(libraries base ocannl stdio)`, plus `arrayjit.verdict` for a verdict). An unused `open` is a fatal warning here, and unused libraries are noise

### Pull Requests

- A **PR accomplishes a goal**: one thing that is true about the system afterwards and was not before, stated in its title. Scope it generously — carry the goal to its natural completion (the change, the tests that pin it, the docs it justifies, the follow-on cleanups it exposes) rather than opening three PRs
- A **commit is one move toward the goal**, not one slice by artifact type: the logic change, its tests, its `.expected` goldens and the doc or agent-note it justifies belong in one commit. A goal usually takes several moves, merged with a merge commit that preserves the series
- **Do NOT touch `CHANGES.md` in feature work** (gh-ocannl-807): the changelog is written in editorial passes from the durable records the work leaves (merge commits, PR bodies, issue closing comments); conventions in `docs/agent-notes/conventions.md`
- When you notice unrelated code smells or design problems, file separate issues
- Follow-up fixing commits are fine, and test-expectation promotions that span several topics can land in a final tests/promotions commit
- When creating commits, include the work summary in the commit message and credit yourself as a co-author
- Each commit should at least compile: loop `git checkout <rev> && dune build @check` over `git rev-list --reverse master..HEAD`
- **Bring the base in before opening the PR; a clean merge does not restart verification**: GitHub builds the PR's MERGE commit, so rebase onto the staging `master` (or merge it in where the branch is shared). The merge gate is one green full PR-matrix run (Linux/macOS) for the PR's current head (gh-ocannl-861): a commit that moves the head waits for its own run, unless CI's `docs/**` filter ignores the diff; bring the base in again only if it touched the PR's own files (the build-and-test note gives the diff). Master's CI after the merge belongs to the CI-red triage routine, not the merger
- **Two repositories, and remote names are not the contract**: development (branches, PRs, `master`) happens in `lukstafi/ocannl-staging`; `ahrefs/ocannl` owns the ISSUES cited as `gh-ocannl-NNN`, the milestones and the releases. Check `git remote -v` before trusting a name, and pass `--repo` to every `gh` command: issues to `ahrefs/ocannl`, PRs to `lukstafi/ocannl-staging`

### Configuration

- See `ocannl_config.reference` for all settings. It ships with every setting COMMENTED OUT (`#key=…`, no space; prose comments use `# `), so copying it verbatim states nothing
- **Adding a config key touches two places**, enforced by `test/operations/test_config_consistency`: document it in `ocannl_config.reference` and register it in `Utils.known_config_keys`. Spell the key as a string literal at the call site (`~arg_name:"the_key"`); the scan fails non-literal uses outside the named lookup functions. A key read only from a test is out of scope
- **Classify the key too**: `test/operations/digest_completeness` fails on a key with no entry in `Utils.config_key_classification` (gh-ocannl-572) — whether it reaches the schedule cache's identity, and which component
- **A new post-lowering module still needs registering** in that test's `codegen_stage_modules` list (a new backend, or anything reading configuration after lowering), so a `Code_borne` misclassification of its keys is noticed

**Configuration Methods** (in order of precedence):
1. Command-line flags: `--ocannl_<option>=<value>` (e.g., `--ocannl_backend=cuda`)
2. Environment variables: `OCANNL_<OPTION>=<value>` (e.g., `OCANNL_BACKEND=cuda`)
3. Config file: `ocannl_config` in current or ancestor directories

**Config profiles** (gh-ocannl-559): `profile=reproducible|performance|approximate` applies a preset bundle (embedded in `arrayjit/lib/utils.ml`) just below the explicit keys of the source that picked it: explicit keys beat a profile of equal immediacy, and a CLI-picked profile beats a config file. A new numerics-changing gate lands in the `approximate` payload (gh-ocannl-719), pinned at its default in `reproducible`, and in the schedule cache's identity in the PR that adds its key — which digest, and the checks that pin the payloads: the backend-precision note.

**Testing with Different Configurations**:

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

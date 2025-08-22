# Repository Guidelines

## Project Structure & Module Organization
- lib/: Core OCANNL library (tensors, shape inference, ppx helpers).
- arrayjit/: Optimizing compiler subpackage (low-level backends, lowering).
- bin/: Executable examples and demos (e.g., `hello_world.ml`, `moons_benchmark.ml`).
- test/: Expect and inline tests grouped by topic (`einsum/`, `operations/`, `training/`, `ppx/`).
- docs/: Slides and reference docs; datasets/: helper data; build_files/ and log_files/: generated artifacts.
- Key config: copy `ocannl_config.example` to `ocannl_config` and adjust backend.

## Build, Test, and Development Commands
- opam deps: `opam install . --deps-only` (OCaml ≥ 5.3 per `dune-project`).
- Build: `dune build` or `dune build @all`.
- Run examples: `dune exec bin/hello_world.exe` (see more in `bin/dune`).
- Test all: `OCANNL_BACKEND=sync_cc dune runtest` (valid backends include `sync_cc`, `multicore_cc`, `cuda`, `metal`). Tests read `test/config/ocannl_config` and if configured for it, generate files under `build_files/` and `log_files/`.
- Format: `dune fmt` (uses `.ocamlformat`).

## Coding Style & Naming Conventions
- OCaml formatting enforced by `.ocamlformat` (margin 100, parse/wrap docstrings). Run `dune fmt` before pushing.
- Overall preference for snake_case (e.g. files `my_module.ml`); OCaml enforces capitalized modules and constructors (`My_module`, `My_variant`).
- Prefer small, composable functions; avoid needless global state. PPX usage (`%op`, `%cd`) is described in `lib/syntax_extensions.md`.

## Testing Guidelines
- Frameworks: `ppx_expect` for inline `%expect` tests, and Dune `test` stanzas for tests with output targets in `.expected` files. Tests live under `test/<area>/*.ml` with paired `*.expected` where applicable.
- Run subset: `dune runtest test/einsum`.
- Tests may diff emitted `.ll` (low-level intermediate representation), `.c`, `.cu`, `.metal` files.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject; reference issues when relevant (e.g., “Fixes #358: …”). Include scope if helpful (einsum/ops/train).
- PRs: clear description, linked issues, reproduction or `dune runtest` output, and mention backend(s) exercised. Include any new example commands.

## Configuration & Backends
- Backend selection and runtime options are read from `ocannl_config` and `OCANNL_BACKEND`. See `ocannl_config.example` for available keys (debug logging, device, precision).
- For CUDA/Metal specifics and debugging, consult README “Using the tracing debugger” and adjust config accordingly.


# Repository Guidelines

## Project Structure & Module Organization
- lib/: Core OCANNL library (tensors, shape inference, ppx helpers).
- arrayjit/: Optimizing compiler subpackage (low-level backends, lowering).
- bin/: Executable examples and demos (e.g., `hello_world.ml`, `moons_benchmark.ml`).
- test/: Expect and inline tests grouped by topic (`einsum/`, `operations/`, `training/`, `ppx/`).
- docs/: Slides and reference docs; datasets/: helper data; build_files/ and log_files/: generated artifacts.
- Global configuration explained in `ocannl_config.example`.

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
- Backend selection and runtime options are read from the file `ocannl_config` in the current directory (or from test/config/ocannl_config for tests), from environment variables e.g. `OCANNL_BACKEND=sync_cc` (but it is not reliable for tests other than env var `OCANNL_BACKEND` which has dedicated support), commandline arguments e.g. `--ocannl_backend=sync_cc` (but this doesn't work with `dune test` which runs multiple tests). See `ocannl_config.example` for available keys (debug logging, device, precision).

**Developer Cheatsheet**
- **Packages:** `arrayjit` (compiler/backends) and `neural_nets_lib` (DL framework). Build high-level tensors in `lib/`, lower/compile in `arrayjit/`.
- **Execution Model:** Express computations as tensors → derive forward/backprop → infer shapes/projections → lower `Assignments` → compile/link per-backend → run on streams (CPU cores/CUDA streams).
- **Key Types:** `Tensor.t` (value/grad nodes), `Tnode.t` (node-level arrays), `Assignments.comp` (accumulating statements), `Indexing.projections` (loop derivation), `Ndarray.t` (host/device buffers).
- **Backends:** `sync_cc`/`multicore_cc` (C via schedulers), `gccjit`, `cuda`, `metal` (if built). Use `Backends.fresh_backend ()` in examples/tests.

**Syntax Extensions**
- **`%op` (operations):** Builds differentiable tensors using `Operation.TDSL`.
  - **Inline params:** `{ w; o = [ dims ] }` creates parameters; with initialization requires `Operation.PDSL` in scope.
  - **Convenience:** Regular OCaml works for many tensor expressions; `%op` mainly improves labels and inline decls.
- **`%cd` (code):** Builds `Assignments.comp` for forward/backward code via `Operation.NTDSL` (non‑diff tensors inside).
  - **Accum ops:** Infix assignment operators pick accumulation: `=+`, `=-`, `=*`, `=/`, `=**`, and variants `=:+` etc.
  - **Projections:** Provide `~projections` or rely on mnemonics (`v`, `v1`, `v2`, `g`, `g1`, `g2`, `lhs`, `rhs1`, `rhs2`) to select slots.
  - **Array refs:** `.v` for value node, `.grad` for gradient node, `.merge` for stream merge buffers.
  - **Embedded tensors:** `%cd` auto‑inserts forward code for created tensors and tracks `embedded_nodes` to avoid recompute.
  - **Pow operator:** Use `**.` for pointwise power with numeric exponent; gradients are specialized (fast path for p=1,2).
- **Generalized einsum:** Use `~logic:"...=>..."` for concise projections; shapes use `batch|input->output` notation.

**Shape & Projection Inference**
- **Pipeline:** `propagate_shapes` during build; `finish_inference` before jitting closes shapes (LUB or 1/broadcastable); then `derive_projections` freshens projection ids to avoid cross‑op contamination.
- **Monomorphic now:** Existential `row`/`dim` vars; future polymorphism could reuse `%op ~config` functions with abstract namespaces.
- **Rows:** Three rows per tensor: batch | input -> output; broadcasting can happen “in the middle” with fixed head/tail axes.
- **Indexing:** Projections unify per‑assignment instances (union‑find), yield iterators for product dims; dim=1 maps to `Fixed_idx 0`.
- **Convolutions:** Low‑level buffers include padding in `dims`; high‑level shapes exclude it—padding becomes observable after forcing dims.

**Backend Anatomy**
- **Frontend modules:** `Task`, `Ops`, `Ndarray`, `Tnode` (per‑device arrays, can be virtual), `Indexing`, `Assignments`, `Low_level`.
- **Interfaces:** `Backend_intf` (records parametric in `'buffer_ptr`, `'dev`, `'runner`, `'event`); `Backend_impl` for implementations; `C_syntax` helpers.
- **Implementations:** `Cc_backend`, `Gcc_backend`, `Cuda_backend`, `Metal_backend` plus `Schedulers` for CPU parallelism.
- **Lifting:** `Backends.Add_device` + `Schedulers` → CPU backends; `Raise_backend` maps `Low_level` to `Assignments` and adds buffer retrieval + syncing.
- **Lifecycle:** Compile routines in batches; link per‑stream context; free arrays with `Backends.finalize`.

**Scheduling, Streams, Transfers**
- **Streams:** Loose notion (CPU core/CUDA stream). Linking binds compiled code to a stream; scheduling preserves W→R order via events.
- **Transfers:** `from_host`, `to_host`, `device_to_device` are scheduled like compute; destination waits non‑blocking on source.
- **Merge buffers:** One per stream; use `.merge` in `%cd` (e.g., `[%cd p.grad =+ p.grad.merge]`). Modes: `Streaming_for` (source ptr, may fall back to `Copy` across devices) and `Copy` (physical buffer grown as needed).
- **Auto host transfers:** If `automatic_host_transfers`:
  - `Tnode.do_read/do_write` perform sync and schedule `to_host`/sync; fields: `prepare_read`, `prepare_write`, `devices_not_lagging_host`.
  - `Raise_backend.sync_routine` pre‑schedules `from_host` for untagged inputs; `update_writer_event` tags writers and sets `to_host`.
  - `Raise_backend.alloc_if_needed` schedules `from_host` for constants and tags device.

**Debugging & Tracing**
- **Logs:** Enable tracing in config; `%cd` supports block comments to annotate generated files; debug prints/plots appear in logs.
- **PPX tips:** Keep `%op` parameters non‑nested when labels matter; avoid capturing inner function params for labels.
- **Shape issues:** Inspect `Tensor.shape` after `finish_inference`; watch for padding effects when dims are forced.
- **Streams/merges:** Mismatch of expected vs. scheduled merge node is detected at scheduling; check `.merge` usage and stream contexts.
- **Backend checks:** Start with `sync_cc` for clarity; move to `multicore_cc`/`cuda` once semantics are validated.

**Adding Features (Guidelines)**
- **New op:** Define in `arrayjit/lib/ops.ml` + `Ir.Ops`; add infix if needed; implement forward/backprop with `%cd` (use `~projections`).
- **Tensor API:** Prefer small composable helpers in `lib/operation.ml`; mirror `%op` conveniences when useful.
- **Shape rules:** Add constraints in `lib/shape.ml` and rows in `lib/row.ml`; ensure `propagate_shapes` derives intended LUBs; update `derive_projections` if new projection forms.
- **Backend codegen:** Prefer `Low_level` lowering hooks; reuse `C_syntax`; keep kernel/routine boundaries stable for batching.
- **Docs/tests:** Add `%expect` examples under `test/` showing shapes, projections, and generated code snippets.

**Testing & Validation**
- **Unit slices:** Run subsets like `dune runtest test/einsum` or `test/operations` to iterate quickly.
- **Golden files:** Many tests diff emitted `.ll/.c/.cu/.metal`; update expected outputs only when semantics are intended.
- **Backends in CI:** Use `OCANNL_BACKEND=sync_cc` locally first; selectively exercise `cuda`/`metal` if available.

**Research Tips**
- **Read paths:** `lib/operation.ml` (ops), `lib/tensor.ml` (graph), `lib/shape.ml`/`lib/row.ml` (inference), `arrayjit/lib/*backend*.ml` (runtimes), `arrayjit/lib/indexing.ml` (projections), `arrayjit/lib/low_level.ml` (loops).
- **Compare designs:** Multi‑stream + merge buffers vs. typical single‑stream AD frameworks; generalized einsum for projections vs. manual loops.
- **Trace small models:** Use `bin/micrograd_demo*.ml` and `bin/moons_demo*.ml` with `%cd` comments and higher log level to understand pipeline.
- **Experiment knobs:** Toggle `automatic_host_transfers`, switch backends, vary precision, inspect shapes before/after jitting.

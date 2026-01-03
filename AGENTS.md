# OCANNL Agent Guide

OCANNL (OCaml Compiles Algorithms for Neural Networks Learning) is a from-scratch compiled deep
learning framework with an optimizing compiler. The repo contains two main packages:
- arrayjit: low-level IR, lowering, and backend codegen (CPU/CUDA/Metal).
- neural_nets_lib: high-level tensor DSL, shape inference, backprop, and user-facing blocks.

## Structure and Ownership
- lib/: user-facing recipes (training utilities, nn blocks, re-exports).
- tensor/: core framework internals (Tensor, Shape, Operation, ppx_%op/%cd).
- arrayjit/: compiler + backends (IR, indexing, assignments, backends, schedulers).
- bin/: runnable examples and demos.
- test/: tutorials and tests (ppx_expect and standalone .expected tests).
- docs/: slides and reference docs; ocannl_config.example is the configuration source of truth.
- build_files/ and log_files/: generated artifacts when debug settings are enabled.

Key reference files:
- docs/syntax_extensions.md (authoritative for %op/%cd)
- docs/shape_inference.md (shape/projection inference pipeline)
- arrayjit/lib/context.mli (context-based runtime API)
- ocannl_config.example (all configuration keys and defaults)

## Conceptual Map (How It Fits Together)
- Tensor expressions (%op, Tensor.t) build a graph with shape inference and backprop rules.
- Assignments (%cd, Assignments.comp) express low-level compute and are compiled by arrayjit.
- Shape inference runs during construction and is finalized by finish_inference before jitting.
- Projection inference is re-derived per operation to avoid cross-op contamination.

## Build, Run, Test
- Install deps: `opam install . --deps-only` (OCaml >= 5.3).
- Build: `dune build` (runs cram-style tests) or `dune build @check` (compile only).
- Run an example: `dune exec bin/hello_world.exe`.
- Run tests: `OCANNL_BACKEND=sync_cc dune runtest` (recommended default backend).
- Workflow note: individual tests can be run via `dune exec <test path>.exe`, or using Dune aliases
  like `dune build @runtest-<test name>` when available.
- Format: `dune fmt` (uses .ocamlformat, margin 100).

Testing notes:
- Inline tests use ppx_expect within library modules.
- Standalone tests use Dune test stanzas with .expected files; use `dune promote` to accept changes.
- `OCANNL_BACKEND` is special-cased by tests; other env vars may not retrigger tests without
  touching sources or cleaning.
- Tests read `test/config/ocannl_config` and can emit .ll/.c/.cu/.metal into build_files/.

## Coding Conventions
- Prefer small, composable functions; avoid unneeded global state.
- snake_case for files and functions; modules and constructors are capitalized by OCaml.
- Default to ASCII; don’t introduce Unicode unless file already uses it.

## DSL Usage (%op and %cd)
For code outside the core implementation (tests/examples/user code), start with:
`open Ocannl.Operation.DSL_modules`
This brings in Tensor, Shape, TDSL/NTDSL/PDSL, and Ir.

Key points:
- %op builds Tensor.t; %cd builds Assignments.comp.
- %op requires TDSL in scope; %cd requires NTDSL in scope; inline parameter init in %op
  requires PDSL in scope.
- Inline params: `{ w }` or `{ w = init }`; dims via `o`/`i`/`b` fields.
- `%op` uses a unit-parameter `()` boundary to lift parameter creation; bind layers at `()`
  before applying to inputs to avoid mis-scoped parameters.
- `**.` is pointwise power with numeric exponent (specialized gradients).

## Idioms & Gotchas
- `*` is matrix/compose; `*.` is pointwise. Use `/.` for pointwise division.
- `%op` inline params without brackets use shape inference; brackets `[...]` fix shape and values.
- Einsum capture requires a literal string: `x ++ "a,b" ["a"]` works; `let s = ... in x ++ s ["a"]` does not.
- Einsum labels: `"abc"` means 3 axes; `"abc,"` means a single axis named `abc` (comma = multi-char mode).
- `0.5 + 0.5` creates an inferred-shape constant that adapts to usage (LUB when known, otherwise
  guessed minimal); a lone `1.0` is a fixed scalar dimension and won’t grow with context.
- Use `_rhs1/_rhs2/_lhs` suffixes in %cd for intermediate tensors when projection slots matter.

## Shape & Projection Inference
- Shapes have three rows: batch | input -> output (input is rightmost in underlying arrays).
- Broadcasting can occur with fixed head/tail axes (row variables).
- finish_inference closes unsolved dims (LUB or 1/broadcast); derive_projections re-solves with
  fresh projection ids per op to avoid contamination.
- Generalized einsum `~logic:"...=>..."` supports convolutions, striding, and concatenation `^`.

## Backends, Contexts, and Transfers
- Backends: sync_cc, multicore_cc, cuda, metal (if built).
- Use `Backends.fresh_backend ()` in examples/tests or `Context` API (arrayjit/lib/context.mli).
- Automatic host transfers are controlled by ocannl_config (automatic_host_transfers).
- Merge buffers (`.merge`) support stream-to-stream reductions in %cd.

## Adding Features
- New primitive ops: arrayjit/lib/ops.ml (+ Ir.Ops) and wire into tensor/operation.ml.
- New tensor convenience functions: tensor/operation.ml (use %cd for forward/backprop).
- Shape/projection changes: tensor/shape.ml, tensor/row.ml, arrayjit/lib/indexing.ml.
- Add tests under test/ (einsum/operations/training/ppx as appropriate).
- When creating commits, include the work summary in the commit message and credit yourself as a co-author.

## Debugging & Logs
- Enable `output_debug_files_in_build_directory=true` to emit .ll/.c/.cu/.metal.
- Enable `debug_log_from_routines=true` for kernel logging; see ocannl_config.example.
- CUDA routine logs may require `Utils.capture_stdout_logs` (see README).

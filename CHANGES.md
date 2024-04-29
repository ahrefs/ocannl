## [0.4.0] -- 2024-04-30

### Added

- TODO: API improvements for mixed precision computations.
- TODO: A very naive first stab at Cuda parallelism.

### Changed

- Terminology in the API: Renamed almost all uses of "jit" into uses of "compile" and / or "link".
- Split the compile-to-ptx phase from the build-module and build-kernel-launcher phase.
- Migrated the Cuda backend to ppx_minidebug-based execution tracing.
- TODO: Fixes for mixed precision computations.

## [0.3.3] -- 2024-04-24

### Added

- GitHub workflow for continuous integration and API docs.
- Randomness plug-ins via global config `randomness_lib`: currently only `stdlib` and `for_tests`.

### Fixed

- A bit of code rot in the Cuda backend mock `cuda_backend.missing.ml`.
- NPY: Compatibility with OCaml 5.2.0.
- Renamed the main package name from `ocannl` to `neural_nets_lib`, to prevent the opam linter from complaining about a confusing name.

## [0.3.2] -- 2024-04-22

### Added

- `let%cd _ =` (and `let%op _ =`?) do not affect root tracking (intended for adding shape constraints).
- More expressive shape constraints: allowing row variables to be sandwiched between leftmost axes `beg_dims` and rightmost axes `dims`.
- Einsum notation support for leftmost axes.

### Changed

- Cleaned up "user-facing" API by moving `IDX` and `CDSL` to `Train`, and `Tensor.O` to more precise `Operation.At`.
- Added interface `Tensor.mli` to reduce "the user learning surface".
- Improved documentation and layout of `Shape.mli`.
- A more reasonable syntax for labels specifications and einsum notation. In particular, whitespace insensitive (except whitespace not allowed inside identifiers).
- Vendored the `npy` package while we wait for a PR.

### Fixed

- Moved `cudajit` to `depopts`.
- Slice shape inference is now complete, by using leftmost axes `beg_dims` in constraints.

## [0.3.1] -- 2024-04-15

### Added

- Tensor parameters saving and restoring, Ndarray saving and restoring.
- An operation `outer_sum`: like `einsum` but simpler, addition everywhere.

### Changed

- Tweaks to make the project usable as a package (external library).
- Sanitizing code inclusion via code roots management: `Tensor.consume_forward_code` and `consume_backprop_code`, (optionally but by default) used from `Train`.

### Fixed

- Shape inference in presence of non-0 fixed indexing inside einsums was broken (because actually not implemented).
- Incompleteness of shape inference for slicing was leading to inferring shapes with no axes: constraint generation was intended to raise a shape error instead. Proper fix coming in 0.3.2 will make slice shape inference complete.

## [0.3.0] -- 2024-03-31

Major rewrite. Abandoning the design choices of 0.1 and 0.2.

### Added

- Optionally, inferring or checking tensor (batch) sizes from data (e.g. file) sizes.
- Static indexing. A "slice" operator to select individual batches.
- Established the backends API with first-class modules.
- The `Train` module as an optimization "frontend".
- Parallel optimization across devices.
- Global settings configurable via config files, environment variables, and commandline flags.
- Integration of backend logging with `ppx_minidebug` (the `debug_log_from_routines` setting).

### Changed

- The Cuda backend is not supported for now. It is (optionally) buildable to reduce code rot.
- Dynamic indexing is not supported anymore (to reduce complexity). It might be reintroduced if needed.
- Factored out the `arrayjit` library / package containing compilation (former Ndarray, Node, Code).
- Renamed `Formula` -> `Tensor`
- No more "form vs. non-form" formulas / tensors.
  - Formula/tensor roots are split into forward roots and backprop roots.
- No more `%nn_rs`, `%nn_dt` syntaxes and `Synthetic` fetch primitive.
- Renamed `%nn_op` to `%op` and `%nn_cd` to `%cd`.
- Migrated `gccjit` into a separate repository.
- Migrated `cudajit` into a separate repository.
- Massive rewrite of shape inference in a declarative style.
- Generalize `zero_out` to `initialize_neutral` to prepare arbitrary accumulation operation.
- Renamed `Node` -> `Lazy_array` -> `Tnode` (tensor node).

## [0.2.1] -- 2023-07-19

### Added

- The Cuda backend.
  - The Cudajit interface based on Nvrtc and the Cuda driver API.
  - A naive `Exec_as_cuda` backend where the dedicated `Task_id` axis parallelizes over blocks, and a new dedicated `Sample_num` axis parallelizes over threads in a block.
  - When outputting debug files, stores the source `.cu` code and the assembly `.ptx` code.
  - Supports thread-only tensors, tensors with thread-local "replicated" working copies, constant tensors, and globally updated tensors.
  - The backend uses atomic adds for shared updates, and within-block synchronization to minimize update races and parameter staleness.
  - Debugging: full trace (for thread 0) by logging assignments with the assigned value and indices for the LHS tensor and the RHS tensors, the expression used to compute the assigned value, of values of subexpressions.
- Cuda FFI for retrieving GPU specs and for getting and setting limits.
- `Zero_out` low-level-code primitive using `memset`.
- `Staged_compilation` low-level-code primitive: a (stateful) callback for use by backends.
- When outputting debug files, also stores the high-level code.
- Saving and restoring tensor content to `.npz` (`.npy` archive) files (untested).
- Low-level code based optimizations:
  - unrolls `ToPowOf` with integer exponent,
  - simplifies local computations that are just expressions,
  - some arithmetic simplifications.

### Changed

- Monomorphic `axis_index`, simplified the axes-related types.
- Splits `'a low_level` into monomorphic `unit_low_level` and `float_low_level`.
- Removes integer bigarray types.
- Refactors `Node` + `NodeUI` into `Ndarray` + `Node`.
- Tensor printouts include whether a tensor contains `NaN` or `infinity`.
- Simplifies the `Task_id` functionality: removes `If_task_id_is` and `Global Task_id`; emoves parallelism from `interpret_code`; removes `task_id_func` vs `unit_func` duplication.

### Fixed

- "Non-diff" code inclusion.
- Ensures unique indices/symbols also for the `task_id` and `sample_num` bindings.
- Removes endlines from `PrintBox_utils` benchmark tables cells.

## [0.2.0] -- 2023-06-03

### Added

- The Gccjit backend operates using "on device" copies of tensors, where the "device memory" is the stack of the C function. This is intended to improve cache locality and reduce cache contention.
  - Three / four synchronization heuristics:
    - "parallel": a slice of the tensor is copied host-to-device at the beginning and device-to-host at the end, without interference because each task has a different slice.
    - "update on host": the tensor is copied host-to-device at the beginning; each write is an update, it reads the old value from host to update it on the host. Thus each write is a synchronization point.
    - "replicated": the tensor is copied host-to-device at the beginning; only task 0 copies device-to-host.
    - "device-only": no copying to/from host.
- On-device-only tensors that are not materialized on the OCaml side.
- A new category of axis dimensions is introduced: `Frozen`. It is analogous to the `Parallel` axis category in that a single task execution / "device call" only processes a 1D slice of the axis.
  - Currently, for tensors processed in parallel, we only support processing of a contiguous tensor slice (copied "to device" using `memcpy`).
- A new syntax `%nn_rs` ("postprocess results" variant of `%nn_dt`) for computations that should happen at the end of task execution / refresh step. It's meant to prepare the data to be copied back to the host.

### Changed

- Got rid of backend-agnostic synchronization. It was not worth the complexity / implementation effort at this point.
  - Keeping the `Rebalance` constructor around, but it is not playing any role.
- Got rid of `debug_virtual_nodes`, was tricky to maintain.
- Dynamic indexing now skips over parallel axes: when there is a `Parallel` axis on the left, it is preserved in the resulting tensor (slice), and the next-right axis is indexed into instead.
  - Removed the "indexing axes from-right" functionality for now (fails as not implemented).
- Dynamic indexing now can produce virtual nodes.

### Fixed

- Dynamic indexing fixes.

## [0.1.2] -- 2023-05-12

### Added

- Thread-local parameter `task_id` for automated iteration over a dimension `Parallel`.
  - This implements multicore SGD.
  - Rebalancing of computations that don't use `Parallel`, and synchronization in the `Gccjit` backend, are left as future work.
  - Already provides significant speedups in the interpreter (6-7x for me), but that's a moot point.
  - Giving up further work this approach for now, because the bottleneck is the memory access with `Gccjit`.
  - Keeping the new representation capability around, maybe it will be a stepping stone to other things.
- Monolithic step update with "macrobatch" (multiple steps within one backend call).

### Changed

- Streamlined the source code, e.g. removed the `OCaml` backend.
- Better syntax for `%nn_dt` and `%nn_op` shape specification, allows identifiers.
- Improved virtual node and scalar constant inlining.
- Better debugging, e.g. an option to "trace" `Gccjit` execution by printing the comments.

## [0.1.1] -- 2023-05-06

### Added

- An _inline constants_ optimization that compile-time computes scalar constant subexpressions and inlines the values.

### Changed

- Improved debuggability.

### Fixed

- A last-minute breaking bug (would be nice to have a pre-release or a pre-publish hook to run tests!).
- The virtual nodes optimization is more robust, correct even with aggressive inlining settings (e.g. escaping variables check).

## [0.1.0] -- 2023-05-04

### Added

- The first changes-tracking release. Earlier development history is still somewhat documented via closed issues.
- Supports single and double precision floats, more precisions in the future.
- Generates a monolithic step update routine executed by `refresh_session ()`, but can generate arbitrary additional routines at arbitrary times to be executed at arbitrary other times within a session.
- An `Interpreter` backend that can for example log all individual tensor modifications.
- A `Gccjit` backend that can sometimes be 400x faster than the `Interpreter` backend (without any debug work/output).
- A _virtual nodes (tensors)_ optimization that inlines computation of a cell in lieu of tensor accesses, can sometimes reduce memory consumption by 1/3.

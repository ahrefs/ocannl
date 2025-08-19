## [0.6.0]  -- 2025-08-19

### Added

- Support for Brain float aka. bfloat16 aka. BF16, and for FP8.
- Support for convolution via affine indexing expressions in: projections, einsum notation, shape inference.
- MNIST and CIFAR10 datasets (borrowed from Raven).
- Names dataset with bigram use-case helper.
- Half-moons synthetic dataset.
- New precision `Uint4x32` that piggybacks on the `Complex.t` type for the `Bigarray` backing.
- New precision `Int64` for integer operations.
- New operation `Threefry4x32`, which is unusually and hopefully uniquely coarse-grained (requiring nontrivial implementation code for each backend that should conform to a common algorithm).
  - This way we avoid introducing multiple operations on bits.
- Support of counter-based randomness via the `Threefry4x32` operation and random seed tracking.
  - The cascade of splits uses the Tnode id, the train step and the tensor cell position.
- Added a new operation `Uint4x32_to_prec_uniform` that converts the 128-bit random values to floating point uniform distributions efficiently.
- Vector operations support with `Set_from_vec` in low-level IR for efficient vectorized assignments.
- Added a field `params` to `Tensor.t` since we need to track parameters to properly initialize computations (see below).
- `Embed_self_id` operation for positional embeddings.
- Bidirectional precision inference (both top-down and bottom-up).
- Enhanced `%cd` syntax with support for `.forward`, `.backprop`, `.zero_grads` and automatic comment generation.
- Inline tensor declarations in `%cd` syntax for standalone expressions.
- `Train.init_params` for streamlined parameter initialization.
- Better configurability with `inline_complex_computations` setting.

### Changed

- Removed the ndarray initialization logic. Some of its functionality is now incorporated into `fetch_op`.
- Refactored `init_op` and the badly named `global_identifier` from `ops.ml` into `dedicated_access` in `low_level.ml` and a bigger `fetch_op` in `assignments.ml` (more meaningful file locations).
  - Also renamed the badly named `Get_global` to `Access`.
- Initialization now needs to be handled via running the corresponding code explicitly. In particular `Tensor.init_params` will run the forward code of tensors from the `params` field.
- Virtual nodes and inlining now also work across routines. This required changing the API to pass the `optimize_ctx` optimization context.
- Made ppx_minidebug logging per-file opt-in at compile time for better control.
- Refactored Tensor API to reduce boilerplate and share parameter signatures.
- Renamed `float_t` to `scalar_t` throughout the codebase for consistency.
- Migrated from heap-local allocation to on-stack allocation by default.
- Improved shape inference with better Total_elems constraint handling and LUB (Least Upper Bound) support.
- Enhanced projections inference with better slot selection heuristics.
- More defensive handling of empty dimensions and zero-dimension scalars.

### Fixed

- Memory leak in builtins.c.
- Context handling for constants initialized on devices.
- Zero-initialization that wasn't being performed on Linux (MacOS zero-initializes by default).
- Surjectivity and bijectivity checking in indexing operations.
- CUDA backend regressions and missing constructs.
- Duplicate Shape_rows constraints elimination.
- Precision inference issues with premature forcing.
- Bus error on large datasets.
- Session-level bugs that appeared only in specific backends.
- Identifier generation to not start with digits.
- Host-device synchronization issues with `devices_not_lagging_host` semantics.
- Shape inference corner cases with Total_elems and row constraints.
- Various issues with convolution and strided iteration support.
- Moved away from using statically loaded builtins.c from routines (kernels), all backends now prepend their builtins textually.
- Emulating _Float16 aka. half on systems with C compilers that don't support it.

## [0.5.3] -- 2025-05-24

### Added

- The Metal framework backend (Apple Silicon).
- Setting `debug_log_to_stream_files` to neatly keep logs from routine execution in their separate files.
- Settings `clean_up_artifacts_on_startup`, `prefer_backend_uniformity`.
- Tools directory and the `minised` tool: regexp replacement file rewrite.
- Directory arrayjit/bin and executable `read_config` for extracting OCANNL configuration into txt files.

### Changed

- Removed `initialize` and `is_initialized` from the backend API; instead, backends should be initialized on functor application. The functors now take `config` as argument.
- More descriptive identifier names in C-syntax code in case of name conflicts.
- Changed the backend config name `cc` to `multicore_cc` for consistency.
- Migrated out of `Stdlib.Format` to `PPrint` for all structured formatting.
- Migrated stdout capture to thread-based (domain-based actually); for Windows compatibility but also much more robust for large logs.

### Fixed

- Avoid conflicts with C math function names like `fma`.
- Satur01_gate had wrong semantics.

## [0.5.2] -- 2025-04-07

### Added

- Lots of new primitive ops:
  - Unary: Satur01 | Exp | Log | Exp2 | Log2 | Sin | Cos | Sqrt | Recip | Recip_sqrt | Neg | Tanh_approx | Not
  - Binary: Satur01_gate | Max | Min | Mod | Cmplt | Cmpeq | Cmpne
  - Ternary: Where | FMA (non-accumulating)
- Ternary tensor operations.
  - A differentiable `where` operation.
- More flexible gradient construction via the `%cd` syntax (better projections inference).
- CC backend piggy-backing on OCaml's C compiler (consistent across OSes).

### Changed

- Updated to printbox 0.12, with upstreamed graphing.
- `-pthread` -> `-lpthread` in `c_library_flags` in `dune` files.
- Removed Numpy support for easier compatibility on native Windows.
- Unary (primitive) ops and relu are now named, not operator syntax.
- Refactored `%cd` parsing of primitive ops.
- `%cd` and `%op` support both curried and uncurried operator application syntax.
- Updated to ppx_minidebug 2.2.0 with support for cross-run diffing.

### Fixed

- Numbers text rendering (consistent across OSes).
- Moved closing row variables to stage 3, because stage 2 may need to process inequalities generating more LUBs.
- Don't unnecessarily prevent bytecode-only build targets.

## [0.5.1] -- 2025-01-01

## Added

- Automatic transfers to host from the context that most recently updated a node.
- Automatic transfers of routine's inputs from host to routine's context if the host array modification was not yet transfered.

## Fixed

- Added `#` as alternative to `~~` for comment lines in `ocannl_config` files, and fixed a bug in their parsing.

## [0.5.0] -- 2024-12-18

### Added

- Interface files for `Backends` and `Low_level`.
- Fixed #245: tracking of used memory. But there's room for improvement.
- Stream-to-stream synchronization functionality, with lazy per-tensor-node synchronization.

### Changed

- Migrated to cudajit 0.6.1.
- Verifying that code is linked with the right contexts, by tracking `embedded_nodes` with assignments.
- Renaming: (virtual) `device` -> `stream`, `physical_device` -> `device`.
- New files: split out `backend_intf.ml`, `backend_impl.ml`, `schedulers.ml` from `backends.ml`; moved `Tnode.task` to `task.ml`; renamed `backend_utils.ml` to `c_syntax.ml`.
- Removed half-static verification of merge buffer nodes inside `device_to_device`.
- Fixed #286: cross-stream-sharing incorporated into `Tnode.memory_mode`.
- Moved the multicore backend from a `device = stream` model to a single device model.
- Got rid of `unsafe_cleanup`.
- Rename `subordinal` to `stream_id`.
- Removed dependency on `core`, broke up dependency on `ppx_jane`.
- Huge refactoring of backend internal interfaces and API (not repeating same code).
- Built per-tensor-node stream-to-stream synchronization into copying functions.
- Re-introduced whole-device blocking synchronization, which now is just a slight optimization as it also cleans up event book-keeping.
- Simplifications: no more explicit compilation postponing; no more hard-coded pointers (all non-local arrays are passed by parameter).
- Fresh backends are now fresh modules to structurally prevent any potential cache leaking.

### Fixed

- Validating merge nodes for the CUDA backend.
- Checking `is_released` on weak array retrieval.

## [0.4.1] -- 2024-09-17

### Added

- Implemented the previously-mocked support for half precision (FP16).
  - We work around the missing Ctypes coverage by not using `Ctypes.bigarray_start`.
  - We check FP16 constants for overflow.
  - We output half precision specific code from the CUDA backend.
- Finally proper support for mixed precision! Lazy precision defaults and delayed precision setting via `Tnode.update_prec`.
- A placeholder `nn_blocks.ml` hinting at an intended design pattern for model components.
- A memory model for the multiple virtual devices per physical device setup, implemented in the CUDA backend. It fixes the CUDA backend behavior in the data parallelism benchmark.
- Slides for the Fun OCaml meetup: [docs/Fun OCaml](docs/OCANNL-slides-basics_backprop_training_loop_codegen.pdf).
- New syntax: inline tensor declarations with a literal float as initial value.

### Changed

- Removed the `pipes_cc, pipes_gccjit` backends (`Pipes_multicore_backend`) -- I had fixed `Pipes_multicore_backend` by using the `poll` library instead of `Unix.select`, but it turns out to be very very slow.
- Changed the `%cd` block comment syntax `~~` to allow detailed structuring. Rewrote `Train.grad_update` to use the `%cd` syntax.
- Made `Train.sgd_one` slightly more thrifty: `p =- learning_rate *. sgd_delta` --> `p =- learning_rate * sgd_delta ~logic:"."` without the inline tensor expression.

### Fixed

- Log levels related de-confusion:
  - Critical bug: logging of computation traces was not properly converted to ppx_minidebug 2.0.
  - Properly restore `log_level` and inform about its setting.
  - By default do not log from tests.
  - `debug_log_from_routines` should only happen when `log_level > 1`.
- Bugs in `Multicore_backend`: `await` was not checking queue emptiness, `worker`'s `Condition.broadcast` was non-atomically guarded (doesn't need to be), possible deadloop due to the lockfree queue -- now replaced with `saturn_lockfree`.
- Reduced busy-waiting inside `c_compile_and_load`, propagating compilation errors now instead of infinite loop on error.
- Fixed loss of significant digits for small numbers when outputting files.
- Added missing mixed-precision conversions in the `C_syntax` backend builder.
- Restored the functionality of debug logging from the cuda backend.
- Always reinitialize global state at the beginning of `let%expect_test`, to make them more deterministic.

## [0.4.0] -- 2024-09-04

### Added

- A new backend "cc": C based on a configurable C compiler command, defaulting to `cc`.
- Merge buffers representational abstraction (one per virtual device):
  - backends just need to support device-to-device transfers,
  - merging gets implemented in "user space".
- CUDA streaming multiprocessor parallelism via streams <-> virtual devices.
- Support for `cuda-gdb` and `compute-sanitizer` (pass the right arguments to cudajit).
- Inline declarations for (non-differentiable) tensors in the `%cd` syntax.
- A minimal wrapper `Sync_backend` creating CPU backends with a single device only, where all calls are synchronous. (It's a baseline and helps debugging.)
- In progress: proper (condition variables based) scheduler. The legacy scheduler (pipes based) kept for now as baseline and to help debugging.
- Documentation for the syntax extensions.
- `%op` syntax: when under a `~config` parameter, refine the inline declared params' labels with `config.label`.
- `%op` syntax: incorporate the input tensor's (if any) label in the resulting tensor's label.
- Comments in config files using the line prefix `~~`.

### Changed

- Terminology in the API: Renamed almost all uses of "jit" into uses of "compile" and / or "link".
- Split the compile-to-ptx phase from the build-module and build-kernel-launcher phase.
- Migrated the CUDA backend to ppx_minidebug-based execution tracing.
- Fixes for mixed precision computations.
- Further terminology refactoring: Renamed `Low_level.compile` to `Low_level.lower`;
  - and `Low_level.compiled` to `Low_level.optimized`, making it a record.
- Further refactoring of the `Backends` API:
  - split the `device` type into virtual `device` and `physical_device`,
  - removed the direct support for `merge`, instead relying on merge buffers.
- Updated to cudajit 0.4.
- A template for C-syntax backends, refactoring CC and CUDA backends.
- Improvements to handling of tensor node labels, and to the `Tnode.debug_name` function.
- Output files generated by backends, and files generated by logging, in separate subdirectories.
- C-syntax logging: also output the pre-assignment value when logging an assignment.
- Migrated to ppx_minidebug 2.0 with the benefits it brings: no runtime passing, `Utils.settings.log_level` unified with ppx_minidebug's log levels.

### Fixed

- Allow verifying that non-embedded tensor nodes of the tensor(s) associated with a linked code are already in the context passed to `link` (resp. `link_batch`), since they won't get introduced into the context. It is the responsibility of helper functions (such as those in `Train`) to ensure the check.
- Fixed both known and newly discovered shortcomings of the syntax extensions.
- In particular, `%op` syntax: lift `~config` applications out of (tensor) functions.
- Multiple other tiny fixes.

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

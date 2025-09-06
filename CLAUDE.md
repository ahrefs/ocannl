# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCANNL (OCaml Compiles Algorithms for Neural Networks Learning) is a from-scratch compiled Deep Learning framework with an optimizing compiler. The project consists of two main packages:

- `arrayjit`: The low-level optimizing compiler with multiple backends (CPU, CUDA, Metal)
- `neural_nets_lib`: The high-level deep learning framework with syntax extensions, shape inference, and backpropagation

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

# Install dependencies
opam install . --deps-only

# Install with optional backends  
opam install cudajit  # for CUDA backend
```

## Architecture Overview

### Core Directory Structure

- `lib/`: High-level neural networks library
  - `tensor.ml/mli`: Main tensor type and operations
  - `shape.ml/mli`: Shape inference system (see detailed docs there for einsum notation)
  - `operation.ml`: Tensor operations and DSL modules
  - `train.ml`: Training utilities and optimizers
  - `nn_blocks.ml`: Basic neural network building blocks (transformers, attention, etc.)
  - `syntax_extensions.md`: Comprehensive guide to `%op` and `%cd` syntax
  - `ppx_*.ml`: Syntax extension implementations

- `arrayjit/`: Low-level array compilation framework
  - `lib/`: Core IR and backend implementations
    - `backend_intf.ml`: Backend interface definitions
    - `assignments.ml`: High-level assignment-based IR
    - `low_level.ml`: Low-level for-loop based IR
    - `tnode.ml`: Tensor node representation
    - `indexing.ml`: Array indexing and projections
    - `*_backend.ml`: Device-specific backend implementations

- `test/`: Integration tests and tutorials
- `bin/`: Command-line utilities

### Key Concepts

1. **Dual Syntax Extensions**:
   - `%cd` ("code"): For assignment computations (`Assignments.comp`)
   - `%op` ("operation"): For tensor expressions (`Tensor.t`)
   - Inline declarations lift to unit parameter `()` scope, enabling parameter reuse

2. **Shape Inference**: 
   - Three axis kinds: batch | input -> output (matrix convention: input rightmost)
   - Row variables (`..d..`) enable flexible axis handling and broadcasting
   - Einsum notation supports convolutions, reductions, and arbitrary permutations
   - "Principle of least commitment": use row variables where axis count doesn't matter

3. **Backend Architecture**: Unified interface supporting CPU (multicore), CUDA, and Metal backends

4. **Memory Management**: Sophisticated tensor node memory modes (Virtual, Local, On_device, Hosted) with automatic host transfers

## Development Workflow

### Testing

- Tests are implemented either as inline expectations using `ppx_expect`; or as cram-style tests using Dune's `test` stanza where an `.ml` file is compiled, executed, and its output compared against an `.expected` file
- The two approaches are exclusive: a test using using `.expected` file target cannot also use `%expect` inline expectations
- `.expected` tests are easier to debug, `%expect` tests should only be used when the outputs are illustrative
- Tutorial files, i.e. `%expect` tests, in `test/` serve as both documentation and integration tests

**Running Tests**:
- `dune runtest` - runs all tests including inline tests and cram-style tests
- `dune runtest test/operations/` - runs all tests in operations directory
- `dune exec test/operations/test_name.exe` - ONLY works for standalone tests with `test` stanza and `.expected` files
- Inline tests (like those in `test_threefry4x32.ml`) are part of library modules and run via `dune runtest`, not `dune exec`

**Test Types**:
- **Inline tests**: Files included in library `modules` field with `inline_tests` stanza (e.g., `test_threefry4x32.ml` in `operations_tutorials` library)
- **Standalone tests**: Files with dedicated `test` stanza and corresponding `.expected` files (e.g., `threefry4x32_demo`)
- Use `dune promote` to accept test output changes
- **Test Placement Guidelines**:
  * Always add tests under one of the test subdirectories
  * Default location is `test/operations`
  * Use `test/einsum` for tests involving complex einsum specifications
  * Use `test/training` for tests involving training loops
  * When adding a test, update the corresponding test stanza
  * For standalone tests, add an `.expected` file for test results (can initially be empty)

**Module Paths and Common APIs**:

- **For files outside OCANNL implementation (tests, examples, user code), always start with `open Ocannl.Operation.DSL_modules`** - this brings all DSL modules into scope (defined in `lib/operation.ml` lines 720-737)
- Available modules after `open Ocannl.Operation.DSL_modules`:
  - `Ir` - Low-level IR types and operations (Ndarray, Ops, Tnode, etc.)
  - `Shape` - Shape inference and einsum notation
  - `Tensor` - Core tensor type and operations
  - `TDSL` - Tensor DSL with automatic differentiation (grad_spec: If_needed)
  - `NTDSL` - No-gradient tensor DSL (grad_spec: Prohibit_grad)
  - `PDSL` - Parameter/gradient-required DSL (grad_spec: Require_grad)
- Precision values: `Ir.Ops.single`, `Ir.Ops.double`, `Ir.Ops.half` (lowercase)
- Tensor printing in expect tests: `Tensor.print ~here:[%here] ~force:false ~with_code:false ~with_grad:false \`Inline tensor`
- For simple test executables, use `(libraries base ocannl stdio)` in dune file

### Configuration

- See `ocannl_config.example` for documentation of all settings
- Key configs: backend selection, debug logging, optimization levels

**Configuration Methods** (in order of precedence):
1. Command-line flags: `--ocannl_<option>=<value>` (e.g., `--ocannl_backend=cuda`)
2. Environment variables: `OCANNL_<OPTION>=<value>` (e.g., `OCANNL_BACKEND=cuda`)
3. Config file: `ocannl_config` in current or ancestor directories

**Testing with Different Configurations**:

- When using environment variables for test configuration other than OCANNL_BACKEND, Dune won't detect changes and may skip tests
- **Warning**: `dune test --force` does NOT re-run expect tests (only rules with alias fields)
- Reliable ways to ensure tests run with new configuration:
  1. Modify `test/config/ocannl_config` directly
  2. Run `dune clean` before testing
  3. Touch/modify test source files
  4. OCANNL_BACKEND environment variable is an exception (explicit dependency)

**Important Debug Settings**:
- `output_debug_files_in_build_directory=true` - enables `build_files/` generation
- `debug_log_from_routines=true` - enables runtime logging from kernels aka. routines
- `debug_log_to_stream_files=true` - writes logs from kernels/routines to `log_files/<backend>-<device>-<stream>.log`
- `clean_up_artifacts_on_startup=false` - preserves debug files between runs

**Available Backends**:
- `sync_cc` combines the implementation cc_backend.ml with the scheduler `Sync` in schedulers.ml
- `multicore_cc` combines the implementation cc_backend.ml with the scheduler `Multicore` in schedulers.ml
- `cuda` with implementation in cuda_backend.ml
- `metal` with implementation in metal_backend.ml

### Backend Development

- Backends must implement stream-based execution with FIFO queuing
- Support for events and synchronization between streams/devices  
- Code generation through `Low_level.t` to backend-specific representations

**Backend Code Generation Architecture**:
- `c_syntax.ml` provides a functor with default C code generation patterns
- `cc_backend.ml` uses defaults from `c_syntax.ml` with minimal overrides
- `cuda_backend.ml` overrides more functions for CUDA-specific syntax (e.g., `__float2half`)
- `metal_backend.ml` overrides using MSL-specific syntax
- Backends must provide `convert_precision` for type conversions
- Builtin functions (e.g., type conversions) must be implemented in:
  - `builtins.c` for C backends
  - `builtins_cuda.ml` for CUDA backend, `builtins_metal.ml` form Metal backend
- When adding new precision types, ensure conversion functions exist in all backend builtins

### Syntax Extensions

- `%cd` requires `NTDSL` module in scope (from `Operation.NTDSL`)
- `%op` requires `TDSL` module in scope (from `Operation.TDSL`)
- Record syntax for inline tensor declarations: `{ tensor_name }` or `{ tensor_name = init_expr }`
- Generalized einsum notation for complex tensor operations

**Key differences between %op and %cd**:
- `%op` allows initialization expressions (`{ x = uniform () }`), used for model parameters
- `%cd` is self-referential only (`{ x }`), used in computation graphs where tensors are defined by operations
- See `docs/syntax_extensions.md` for comprehensive documentation

**Record syntax features**:
- OCaml punning: `{ x }` expands to default initialization (uniform() for parameters in %op)
- Shorthand field names: `o` → `output_dims`, `i` → `input_dims`, `b` → `batch_dims`
- Additional fields map to labeled arguments of tensor creation functions
- Dimension specification: lists `[...]` for output, tuples `(...)` for input, arrays `[|...|]` for batch

**Einsum notation**:
- Binary form: `tensor1 +* "spec1; spec2 => result_spec" tensor2`
- Unary form: `tensor ++ "spec => result_spec"`
- Capture dimensions: `+* "spec" ["var1"; "var2"]` binds dimension variables
- Use `Shape.set_dim var value` to constrain captured dimensions
- Special operators -- binary: `+*` (`einsum`, add-reduce with multiply), `@^+` (`tropical`, max-reduce with add), `+++` (`outer_sum`, add-reduce with add); unary: `++` (`einsum1`, add-reduce), `@^^` (`einmax1`, max-reduce)

## Common Development Tasks

### Adding New Primitive Operations

1. Add primitive operation to `arrayjit/lib/ops.ml`
2. Implement interpretation in the same file
3. Add syntax support in `lib/ppx_*.ml` if needed
4. Add high-level wrappers in `lib/operation.ml`
5. For neural network blocks, see `lib/nn_blocks.ml` for patterns

### Debugging Backend Discrepancies

When outputs differ between backends:

1. Compare runtime logs in `<backend>-<device>-<stream>.log` files (might require minimizing test tensors)
2. Check generated code in `build_files/*.c` vs `*.cu` / `*.metal` for differences
3. Common issues:
   - Incorrect type conversion in `convert_precision` overrides
   - Different numerical precision between CPU and GPU operations

### Backend Extensions

1. Implement device-specific module following `Backend_impl` signatures
2. Add compilation logic in `arrayjit/lib/backends.ml`
3. Handle memory management and synchronization
4. Add configuration options in `ocannl_config.example`

### Shape Inference Extensions

1. Modify projection logic in `arrayjit/lib/indexing.ml`
2. Update shape constraint generation in `lib/shape.ml`
3. Test with various einsum patterns in e.g. `test/einsum_trivia.ml`

## Debugging and Logging

- Set `debug_log_from_routines=true` in config for kernel/routine-level debugging
- Use `log_level=2` for verbose ppx_minidebug output
- CUDA debugging requires `Utils.capture_stdout_logs` wrapper
- Debug files generated in `log_files/` directory (cleaned on startup by default)
- Runtime logs from kernel execution are written to `<backend>-<device>-<stream>.log` (e.g., `cuda-0-0.log`)
- Generated code files in `build_files/` show high-level `.cd`, intermediate `.ll`, and backend-specific `.c`/`.cu` files

## Performance Considerations

- Virtual nodes are inlined automatically (controlled by `virtualize_max_visits`)
- Scalar constants can be inlined via `inline_scalar_constexprs=true`
- Memory sharing optimizations through cross-stream tensor nodes

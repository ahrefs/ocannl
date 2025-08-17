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

# Only compile -- do not link nor run any executable
dune build @check

# Build specific package
dune build -p neural_nets_lib
dune build -p arrayjit

# Run tests
dune runtest

# Run tests for specific package
dune runtest -p neural_nets_lib

# Install dependencies
opam install . --deps-only

# Install with optional backends  
opam install cudajit  # for CUDA backend
```

## Architecture Overview

### Core Directory Structure

- `lib/`: High-level neural networks library
  - `tensor.ml/mli`: Main tensor type and operations
  - `shape.ml/mli`: Shape inference system
  - `operation.ml`: Tensor operations and DSL modules
  - `train.ml`: Training utilities and optimizers
  - `nn_blocks.ml`: Neural network building blocks
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

2. **Shape Inference**: Comprehensive axis tracking with batch/input/output classification and optional dimension labels

3. **Backend Architecture**: Unified interface supporting CPU (multicore), CUDA, and Metal backends

4. **Memory Management**: Sophisticated tensor node memory modes (Virtual, Local, On_device, Hosted) with automatic host transfers

## Development Workflow

### Testing

- Tests are implemented either as inline expectations using `ppx_expect`; or as cram-style tests where an `.ml` file is compiled, executed, and its output compared against an `.expected` file
- Tutorial files in `test/` serve as both documentation and integration tests
- Use `dune promote` to accept test output changes
- **Test Placement Guidelines**:
  * Always add tests under one of the test subdirectories
  * Default location is `test/operations`
  * Use `test/einsum` for tests involving complex einsum specifications
  * Use `test/training` for tests involving training loops
  * When adding a test, update the corresponding test stanza
  * Add an `.expected` file for test results (can initially be empty)

### Configuration

- Copy `ocannl_config.example` to `ocannl_config` to customize settings
- Key configs: backend selection, debug logging, optimization levels
- Config is searched in current and ancestor directories

**Configuration Methods** (in order of precedence):
1. Command-line flags: `--ocannl_<option>=<value>` (e.g., `--ocannl_backend=cuda`)
2. Environment variables: `OCANNL_<OPTION>=<value>` (e.g., `OCANNL_BACKEND=cuda`)
3. Config file: `ocannl_config` in current or ancestor directories

**Important Debug Settings**:
- `output_debug_files_in_build_directory=true` - enables `build_files/` generation
- `debug_log_from_routines=true` - enables runtime logging
- `debug_log_to_stream_files=true` - writes logs to `log_files/<backend>-<stream>-<stream>.log`
- `clean_up_artifacts_on_startup=false` - preserves debug files between runs

### Backend Development

- Backends must implement stream-based execution with FIFO queuing
- Support for events and synchronization between streams/devices  
- Code generation through `Low_level.t` to backend-specific representations

**Backend Code Generation Architecture**:
- `c_syntax.ml` provides a functor with default C code generation patterns
- `cc_backend.ml` uses defaults from `c_syntax.ml` with minimal overrides
- `cuda_backend.ml` overrides more functions for CUDA-specific syntax (e.g., `__float2half`)
- Both backends must provide `convert_precision` for type conversions
- Builtin functions (e.g., type conversions) must be implemented in:
  - `builtins.c` for C backends
  - `builtins_cuda_small.ml` for CUDA backend
- When adding new precision types, ensure conversion functions exist in all backend builtins

### Syntax Extensions

- `%cd` requires `NTDSL` module in scope (from `Operation.NTDSL`)
- `%op` requires `TDSL` module in scope (from `Operation.TDSL`)
- Inline tensor declarations using string literals
- Generalized einsum notation for complex tensor operations

## Common Development Tasks

### Adding New Operations

1. Add primitive operation to `arrayjit/lib/ops.ml`
2. Implement interpretation in the same file
3. Add syntax support in `lib/ppx_*.ml` if needed
4. Add high-level wrappers in `lib/operation.ml`

### Debugging Backend Discrepancies

When outputs differ between backends:
1. Compare runtime logs in `<backend>-<stream>-<stream>.log` files
2. Check generated code in `build_files/*.c` vs `*.cu` for differences
3. Common issues:
   - Missing builtin function implementations in one backend
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
3. Test with various einsum patterns in `test/einsum_trivia.ml`

## Debugging and Logging

- Set `debug_log_from_routines=true` in config for routine-level debugging
- Use `log_level=2` for verbose ppx_minidebug output
- CUDA debugging requires `Utils.capture_stdout_logs` wrapper
- Debug files generated in `log_files/` directory (cleaned on startup by default)
- Runtime logs from execution are written to `<backend>-<stream>-<stream>.log` (e.g., `cuda-0-0.log`)
- Generated code files in `build_files/` show high-level `.cd`, intermediate `.ll`, and backend-specific `.c`/`.cu` files

## Performance Considerations

- Virtual nodes are inlined automatically (controlled by `virtualize_max_visits`)
- Scalar constants can be inlined via `inline_scalar_constexprs=true`
- Memory sharing optimizations through cross-stream tensor nodes
- Backend-specific optimization levels configurable per backend
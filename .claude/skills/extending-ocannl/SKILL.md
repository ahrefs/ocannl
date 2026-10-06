---
name: extending-ocannl
description: Touch-lists for common OCANNL extension tasks: adding a primitive operation, adding or extending a backend, extending shape inference, and diagnosing output differences between backends. Use when adding an op to ops.ml, implementing a new backend, changing projection/constraint logic, or when the same computation gives different results on cc vs cuda vs metal.
---

# Extending OCANNL

## Adding New Primitive Operations

1. Add primitive operation to `arrayjit/lib/ops.ml`
2. Implement interpretation in the same file
3. Add syntax support in `tensor/ppx_*.ml` if needed
4. Add high-level wrappers in `tensor/operation.ml`
5. For neural network blocks, see `lib/nn_blocks.ml` for patterns

## Debugging Backend Discrepancies

When outputs differ between backends:

1. Compare runtime logs in `<backend>-<device>-<stream>.log` files (might require minimizing test tensors)
2. Check generated code in `build_files/<exe-name>/*.c` vs `*.cu` / `*.metal` for differences
3. Common issues:
   - Incorrect type conversion in `convert_precision` overrides
   - Different numerical precision between CPU and GPU operations

## Backend Extensions

1. Implement device-specific module following `Backend_impl` signatures; backends implement stream-based execution with FIFO queuing, plus events and synchronization between streams/devices
2. Add compilation logic in `arrayjit/lib/backends.ml`
3. Handle memory management and synchronization
4. Add configuration options in `ocannl_config.reference`

### Code generation architecture

- `c_syntax.ml` provides a functor with default C code generation patterns from `Low_level.t`
- `cc_backend.ml` uses the defaults with minimal overrides; `cuda_backend.ml` overrides more functions for CUDA-specific syntax (e.g. `__float2half`); `metal_backend.ml` overrides using MSL-specific syntax; `hip_backend.ml` mirrors the CUDA backend
- Backends must provide `convert_precision` for type conversions
- Builtin functions (e.g. type conversions) must be implemented in the per-backend builtin modules prepended to generated code: `builtins_cc.ml` for the C backends, `builtins_cuda.ml` (CUDA), `builtins_hip.ml` (HIP), `builtins_metal.ml` (Metal). `builtins.c` provides the host-side FFI stubs compiled into the library
- When adding new precision types, ensure conversion functions exist in all backend builtins
- A change to a codegen default moves emitted text outside `test/` too: `dune build @test/operations/runtest-codegen_text_inventory` lists every pinning file. A change to `Register_tile.default`, for one, re-promotes `arrayjit/test/test_vectorized_codegen`, which a `test/operations`-only batch never reaches

## Shape Inference Extensions

1. Modify projection logic in `arrayjit/lib/indexing.ml`
2. Update shape constraint generation in `tensor/shape.ml`
3. Test with various einsum patterns in e.g. `test/einsum_trivia.ml`

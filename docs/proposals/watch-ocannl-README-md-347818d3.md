# Proposal: CPU matrix multiplication optimizations from siboehm's article

Task: watch-ocannl-README-md-347818d3
Issue: https://github.com/ahrefs/ocannl/issues/412

## Goal

Transform OCANNL's naive triple-nested-loop CPU matrix multiplication into a tiled, SIMD-vectorized kernel inspired by Simon Boehm's "Fast Multidimensional Matrix Multiplication on CPU from Scratch." The goal is to achieve an order-of-magnitude speedup on representative matrix sizes (512x512+) by applying cache-aware tiling, loop reordering, SIMD FMA intrinsics, register tiling, and operand packing --- all integrated into OCANNL's existing compilation pipeline. The tiling infrastructure should be general enough to benefit any einsum contraction, not just matmul.

## Acceptance Criteria

- Matrix multiplication on CPU uses cache-aware tiling with block sizes tuned for L1/L2 cache hierarchy.
- The innermost micro-kernel uses platform-appropriate SIMD intrinsics: AVX/AVX2 FMA on x86-64, NEON FMA on ARM/Apple Silicon.
- Register tiling: a small tile of the output matrix (e.g. 4xSIMD-width) is held in SIMD registers during the inner loop, with operands streamed through.
- Operand packing: before the tiled computation, operand blocks are repacked into contiguous memory matching the access pattern of the micro-kernel.
- Loop ordering within tiles is optimized for cache locality (ikj or equivalent tiled variant).
- Performance is measurably improved on representative matmul sizes (at minimum 512x512 and 1024x1024 single-precision). A benchmark comparing naive vs optimized is included.
- Small matrices (dimensions below a configurable threshold) bypass tiling to avoid overhead.
- Non-square matrices and dimensions that are not multiples of the tile size are handled correctly (remainder/edge tiles).
- The tiling infrastructure is general: it can apply to arbitrary einsum contractions expressed via `Shape.Compose`, not only rank-2 matmul.
- Tiling parameters (block sizes, register tile dimensions) are configurable, with sensible defaults.
- All existing tests pass with no regression.
- Parallelization (multi-threading) is out of scope; the optimizations target single-threaded throughput.

## Context

### Current compilation pipeline

OCANNL compiles tensor operations through a multi-stage pipeline. For matrix multiplication:

1. **High-level**: `matmul` uses `Shape.Compose` to define the contraction.
2. **Assignment compilation**: `assignments.ml` `loop_accum` (line 282) generates nested `For_loop` nodes over output dimensions and contraction dimension.
3. **Low-level IR**: `low_level.ml` defines `For_loop { index; from_; to_; body; trace_it }` (line 38). `loop_over_dims` (line 1695) creates simple nested loops. `unroll_dims` (line 1713) handles compile-time unrolling for small arrays.
4. **Optimization**: `optimize_proc` (line 1388) runs virtualization, cleanup, CSE, and simplification --- but no tiling or loop reordering.
5. **C code generation**: `c_syntax.ml` `pp_ll` (line 305) emits plain C `for` loops for `For_loop` nodes. `Set_from_vec` (line 404) handles existing vector operations via `vec_unop_syntax`.

The result is a **naive triple-nested loop** with no tiling, blocking, SIMD FMA, or cache optimization.

### Key code locations (verified in ocannl-staging)

| Component | File | Line(s) | Notes |
|-----------|------|---------|-------|
| `For_loop` IR node | `arrayjit/lib/low_level.ml` | 38 | `{ index; from_; to_; body; trace_it }` |
| `loop_over_dims` | `arrayjit/lib/low_level.ml` | 1695 | Creates nested `For_loop` from dimension array |
| `unroll_dims` | `arrayjit/lib/low_level.ml` | 1713 | Compile-time unrolling for small dims |
| `optimize_proc` | `arrayjit/lib/low_level.ml` | 1388 | Pipeline: trace -> virtualize -> cleanup -> CSE -> simplify |
| `loop_accum` | `arrayjit/lib/assignments.ml` | 282 | Generates loops for accumulation (matmul contraction) |
| `pp_ll` (For_loop codegen) | `arrayjit/lib/c_syntax.ml` | 314 | Emits C `for` loops |
| `Set_from_vec` codegen | `arrayjit/lib/c_syntax.ml` | 404 | Existing vector op code generation |
| `vec_unop_syntax` | `arrayjit/lib/c_syntax.ml` | 417 | Platform-specific vector op syntax |
| C/CC backend | `arrayjit/lib/cc_backend.ml` | - | Compiles and runs generated C code |

### Existing vector operation support

The `Set_from_vec` IR node and associated code generation in `c_syntax.ml` handle simple element-wise vector operations (unary ops applied across a vector width). This provides a starting point for SIMD code generation, but does not cover the FMA (fused multiply-add) patterns needed for matmul micro-kernels. The micro-kernel will require new code generation patterns: broadcast-load-FMA sequences for the inner loop.

### Interaction with other tasks

- **gh-ocannl-350** (loop hoisting / CSE): Should land first. Tiling runs after hoisting and CSE in the optimization pipeline, since tiling restructures loops and would interfere with simpler analysis passes.
- **watch-ocannl-README-md-d5eb2b05** (CUDA matmul): This task is blocked by the current one. The tiling infrastructure designed here (loop splitting, tile-size configuration) should be reusable for GPU tiling with shared memory, though GPU code generation will differ substantially.
- **gh-ocannl-134** (sharing for-loops between virtual arrays): Related loop optimization that could interact with tiling --- tiling should be designed to compose with loop sharing.

### Platform considerations

- **x86-64**: AVX2 `_mm256_fmadd_ps` (8 floats), with optional AVX-512 `_mm512_fmadd_ps` (16 floats) behind a feature flag.
- **ARM/Apple Silicon**: NEON `vfmaq_f32` (4 floats). Apple Silicon has wide execution units that benefit from register tiling even with narrower SIMD.
- Code generation in `c_syntax.ml` must select intrinsics based on the target platform. This can be done via C preprocessor guards (`#ifdef __AVX2__`, `#ifdef __ARM_NEON`) in the generated code, or via a compile-time platform detection in the OCaml code generator.

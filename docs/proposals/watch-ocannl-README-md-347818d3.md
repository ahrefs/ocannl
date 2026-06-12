# Proposal: CPU matrix multiplication optimizations from siboehm's article

Task: watch-ocannl-README-md-347818d3
Issue: https://github.com/ahrefs/ocannl/issues/412

## Status update (2026-06-12)

- Issue #412 is OPEN. Its GH milestone is v0.8 (due date 2026-02-28 has lagged); ROADMAP.md
  is authoritative and targets **v0.8 mid-June 2026** with "Tiling optimizations (inspired
  by #412)" as the lead item — this work is current, not stale.
- **No tiling/SIMD-FMA work has landed**: `low_level.ml` still has no loop
  splitting/reordering pass and `c_syntax.ml` still emits plain nested `for` loops for
  `For_loop`; the matmul kernel remains the naive triple loop.
- The pipeline-order dependency has resolved: #350 (loop-invariant hoisting) is CLOSED
  (not planned as a separate pass) — instead, cross-statement CSE with hoisting to a common
  ancestor scope landed in `low_level.ml` (`hoist_cross_statement_cse`, line 1586, commit
  `e48ec84f`, plus soundness fixes `6695f445`/`64685b24`). Tiling should be designed to run
  after this CSE-hoisting pass, per the original reasoning.
- Line drift in the cited files (identifiers unchanged): `loop_over_dims` now
  `low_level.ml:1929` (was 1695), `unroll_dims` 1947 (was 1713), `optimize_proc` 1619
  (was 1388); `pp_ll` now `c_syntax.ml:331` with the `For_loop` case at 340 (was 305/314),
  `Set_from_vec` codegen at 438 (was 404), `vec_unop_syntax` implementation at 197 (was
  417). `For_loop` is still `low_level.ml:38` and `loop_accum` still `assignments.ml:282`.
- The GPU side of #412 has its own fresher proposal, `docs/proposals/gh-ocannl-412.md`
  (tiling + multi-threaded GPU kernels); this file remains the plan for the CPU article.
  The two should share the tile-size/loop-splitting infrastructure.
- Other landed work to be aware of when implementing: kernel parameters were renamed
  params→kparams / `Param_ptr`→`Kparam_ptr` (#356), and redundant `Zero_out` elision
  landed in `c_syntax.ml` (gh-ocannl-420).

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
3. **Low-level IR**: `low_level.ml` defines `For_loop { index; from_; to_; body; trace_it }` (line 38). `loop_over_dims` (line 1929) creates simple nested loops. `unroll_dims` (line 1947) handles compile-time unrolling for small arrays.
4. **Optimization**: `optimize_proc` (line 1619) runs virtualization, cleanup, CSE (now including cross-statement hoisting), and simplification --- but no tiling or loop reordering.
5. **C code generation**: `c_syntax.ml` `pp_ll` (line 331) emits plain C `for` loops for `For_loop` nodes. `Set_from_vec` (line 438) handles existing vector operations via `vec_unop_syntax`.

The result is a **naive triple-nested loop** with no tiling, blocking, SIMD FMA, or cache optimization.

### Key code locations (verified in ocannl-staging)

| Component | File | Line(s) | Notes |
|-----------|------|---------|-------|
| `For_loop` IR node | `arrayjit/lib/low_level.ml` | 38 | `{ index; from_; to_; body; trace_it }` |
| `loop_over_dims` | `arrayjit/lib/low_level.ml` | 1929 | Creates nested `For_loop` from dimension array |
| `unroll_dims` | `arrayjit/lib/low_level.ml` | 1947 | Compile-time unrolling for small dims |
| `optimize_proc` | `arrayjit/lib/low_level.ml` | 1619 | Pipeline: trace -> virtualize -> cleanup -> CSE (incl. cross-statement hoisting) -> simplify |
| `loop_accum` | `arrayjit/lib/assignments.ml` | 282 | Generates loops for accumulation (matmul contraction) |
| `pp_ll` (For_loop codegen) | `arrayjit/lib/c_syntax.ml` | 331 | Emits C `for` loops (`For_loop` case at 340) |
| `Set_from_vec` codegen | `arrayjit/lib/c_syntax.ml` | 438 | Existing vector op code generation |
| `vec_unop_syntax` | `arrayjit/lib/c_syntax.ml` | 197 | Platform-specific vector op syntax |
| C/CC backend | `arrayjit/lib/cc_backend.ml` | - | Compiles and runs generated C code |

*(Update 2026-06-12: line numbers refreshed against the current tree.)*

### Existing vector operation support

The `Set_from_vec` IR node and associated code generation in `c_syntax.ml` handle simple element-wise vector operations (unary ops applied across a vector width). This provides a starting point for SIMD code generation, but does not cover the FMA (fused multiply-add) patterns needed for matmul micro-kernels. The micro-kernel will require new code generation patterns: broadcast-load-FMA sequences for the inner loop.

### Interaction with other tasks

- **gh-ocannl-350** (loop hoisting / CSE): Should land first. Tiling runs after hoisting and CSE in the optimization pipeline, since tiling restructures loops and would interfere with simpler analysis passes. *(Update 2026-06-12: resolved — #350 is closed; cross-statement CSE with hoisting landed in `low_level.ml` (`hoist_cross_statement_cse`), so this precondition is satisfied.)*
- **watch-ocannl-README-md-d5eb2b05** (CUDA matmul): This task is blocked by the current one. The tiling infrastructure designed here (loop splitting, tile-size configuration) should be reusable for GPU tiling with shared memory, though GPU code generation will differ substantially.
- **gh-ocannl-134** (sharing for-loops between virtual arrays): Related loop optimization that could interact with tiling --- tiling should be designed to compose with loop sharing.

### Platform considerations

- **x86-64**: AVX2 `_mm256_fmadd_ps` (8 floats), with optional AVX-512 `_mm512_fmadd_ps` (16 floats) behind a feature flag.
- **ARM/Apple Silicon**: NEON `vfmaq_f32` (4 floats). Apple Silicon has wide execution units that benefit from register tiling even with narrower SIMD.
- Code generation in `c_syntax.ml` must select intrinsics based on the target platform. This can be done via C preprocessor guards (`#ifdef __AVX2__`, `#ifdef __ARM_NEON`) in the generated code, or via a compile-time platform detection in the OCaml code generator.

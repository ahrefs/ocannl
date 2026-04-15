# Proposal: Add AVX/AVX2 intrinsics to the C backend

Task: gh-ocannl-164
Issue: https://github.com/ahrefs/ocannl/issues/164

## Goal

Enable the C backend to leverage AVX/AVX2 SIMD instructions for vectorizable inner loops, with NEON as the ARM/Apple Silicon equivalent. The primary mechanism is compiler auto-vectorization via appropriate flags and code patterns, supplemented by structured loop generation that the compiler can reliably vectorize. This provides a foundation for the explicit SIMD micro-kernels needed by the tiling task (watch-ocannl-README-md-347818d3 / gh-ocannl-412).

## Acceptance Criteria

- The C backend compiles generated code with `-mavx2 -mfma` on x86-64 (in addition to the existing `-march=native`) and verifies that the compiler supports these flags.
- Memory allocated for tensor buffers is 32-byte aligned (AVX requirement), using `aligned_alloc` or `posix_memalign` instead of the current `Ctypes.allocate_n`.
- Generated C code uses `__attribute__((aligned(32)))` for stack-allocated local arrays in compiled kernels.
- The innermost dimension of contiguous-stride loops is emitted with a pattern the compiler can auto-vectorize: simple stride-1 access, no aliasing (use `restrict` on pointer parameters), and loop trip counts visible to the optimizer.
- A new `builtins_cc.ml` entry provides platform-detection macros (`OCANNL_HAS_AVX2`, `OCANNL_HAS_NEON`) via `#ifdef __AVX2__` / `#ifdef __ARM_NEON` guards.
- Optional pragma hints (`#pragma GCC ivdep` or `#pragma clang loop vectorize(enable)`) are emitted before inner loops to encourage auto-vectorization.
- Performance improvement is measurable: at least 2x speedup on a float32 element-wise operation or reduction over arrays of size >= 1024, verified with a benchmark.
- Graceful fallback: code compiles and runs correctly on architectures without AVX2 (the flags are conditional on platform detection; the generated C code itself uses no explicit intrinsics in this phase).
- All existing tests pass with no regression.

## Context

### Current compilation pipeline (C backend)

1. **Low-level IR** (`arrayjit/lib/low_level.ml`): `For_loop { index; from_; to_; body; trace_it }` (line 38). `loop_over_dims` (line 1695) generates nested for-loops from dimension arrays. Loops iterate from `from_` to `to_` inclusive.

2. **C code generation** (`arrayjit/lib/c_syntax.ml`): `pp_ll` (line 305) emits C `for` loops. The loop index type is `uint32_t` (or `uint64_t` for big models). Array accesses use computed offsets via `pp_array_offset` (line 275).

3. **CC backend** (`arrayjit/lib/cc_backend.ml`): Compiles generated `.c` files with GCC/Clang. Compiler flags (line 86-96): `-O3 -march=native` by default. The `arch_flags` setting (line 20) defaults to `-march=native` which already enables AVX2 on machines that support it, but the generated code is not structured for auto-vectorization.

4. **Memory allocation** (`arrayjit/lib/backend_impl.ml`): Uses `Ctypes.allocate_n int8_t ~count:size_in_bytes` (line 48), which calls `malloc` -- no alignment guarantee beyond default (typically 16 bytes on 64-bit systems, insufficient for AVX's 32-byte requirement).

5. **Precision types** (`arrayjit/lib/ops.ml`): `Single_prec` -> `float` (4 bytes), `Double_prec` -> `double` (8 bytes). The most common computation precision is `float` (single). AVX2 processes 8 floats or 4 doubles per instruction.

6. **Existing vector support**: `Set_from_vec` IR node (line 41) handles `Uint4x32_to_prec_uniform` conversion -- a fixed-width vector unop. The `c_vec_typ_of_prec` function (ops.ml line 339) defines struct-based vector types (`float4_t`, `double2_t`). These are portable but not SIMD-width.

### Key code locations

| Component | File | Line(s) | Relevance |
|-----------|------|---------|-----------|
| `For_loop` codegen | `arrayjit/lib/c_syntax.ml` | 314-331 | Where loop headers are emitted; add pragma hints here |
| `compile_proc` | `arrayjit/lib/c_syntax.ml` | 756-892 | Function generation; add `restrict` to pointer params |
| Compiler flags | `arrayjit/lib/cc_backend.ml` | 86-96 | Add conditional AVX2/FMA flags |
| `arch_flags` setting | `arrayjit/lib/cc_backend.ml` | 20 | Default `-march=native` |
| Memory allocation | `arrayjit/lib/backend_impl.ml` | 44-51 | Replace with aligned allocation |
| Local array declarations | `arrayjit/lib/c_syntax.ml` | 860-877 | Add alignment attribute |
| Includes / builtins | `arrayjit/lib/builtins_cc.ml` | 1-10 | Add platform detection macros |
| Precision types | `arrayjit/lib/ops.ml` | 324-337 | C type names for SIMD width computation |
| `C_syntax_config` | `arrayjit/lib/c_syntax.ml` | 16-75 | Module type; may need `restrict` or alignment config |

### Relationship to tiling (gh-ocannl-412 / watch-ocannl-README-md-347818d3)

The tiling proposal explicitly depends on SIMD support for its micro-kernel. It calls for explicit AVX2 `_mm256_fmadd_ps` / NEON `vfmaq_f32` intrinsics in the tiled inner loop. This task provides the **foundation**: aligned memory, correct compiler flags, auto-vectorization-friendly loop patterns, and platform detection macros. The tiling task will then add explicit intrinsic emission for the micro-kernel. This separation keeps the two tasks independent and testable.

## Approach

### Phase 1: Aligned memory allocation

**File: `arrayjit/lib/backend_impl.ml`**

Replace `Ctypes.allocate_n int8_t ~count:size_in_bytes` with aligned allocation:

```ocaml
let aligned_alloc ~alignment ~size_in_bytes =
  let ptr = Ctypes.(to_voidp @@ coerce (ptr void) (ptr int8_t)
    (Foreign.foreign "aligned_alloc" Ctypes.(size_t @-> size_t @-> returning (ptr void))
       (Unsigned.Size_t.of_int alignment) (Unsigned.Size_t.of_int size_in_bytes))) in
  ...
```

Use 32-byte alignment (sufficient for AVX/AVX2; AVX-512 would need 64). Fall back to `posix_memalign` on platforms where `aligned_alloc` is unavailable. The size must be rounded up to a multiple of the alignment.

Note: `aligned_alloc` requires the size to be a multiple of alignment. Add a rounding helper.

### Phase 2: Compiler flags and platform detection

**File: `arrayjit/lib/cc_backend.ml`**

The existing `arch_flags` default of `-march=native` already enables AVX2 on capable hardware. Add explicit flag validation: attempt compilation with `-mavx2 -mfma` and cache whether the compiler/platform supports them. If supported, also pass `-ftree-vectorize` (usually on by default at `-O3`, but making it explicit helps).

**File: `arrayjit/lib/builtins_cc.ml`**

Add platform detection macros to the `includes` string:

```c
#ifdef __AVX2__
  #define OCANNL_HAS_AVX2 1
  #include <immintrin.h>
#else
  #define OCANNL_HAS_AVX2 0
#endif

#ifdef __ARM_NEON
  #define OCANNL_HAS_NEON 1
  #include <arm_neon.h>
#else
  #define OCANNL_HAS_NEON 0
#endif
```

These macros are needed by the tiling task for explicit intrinsics, and immediately useful for any conditional SIMD code paths.

### Phase 3: Auto-vectorization-friendly code generation

**File: `arrayjit/lib/c_syntax.ml`**

3a. **`restrict` qualifiers on pointer parameters** (in `compile_proc`, around line 773):

Change parameter declarations from `float *arr` to `float * restrict arr`. This tells the compiler that pointers don't alias, which is a prerequisite for auto-vectorization. OCANNL's memory model supports this: each tensor has its own allocation.

3b. **Vectorization pragma hints** (in `pp_ll`, around line 314):

Before the innermost `for` loop (detected by checking whether the loop body contains no nested `For_loop`), emit:

```c
#if defined(__GNUC__)
#pragma GCC ivdep
#elif defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#endif
```

Detection of "innermost loop" requires a simple check: scan the body for `For_loop` nodes. If none found, this is an inner loop candidate for the pragma.

3c. **Alignment attribute on local arrays** (in `compile_proc`, around line 868-874):

Change local array declarations from:
```c
float arr[N];
```
to:
```c
float arr[N] __attribute__((aligned(32)));
```

This ensures stack-allocated arrays used in inner loops are aligned for SIMD access.

### Phase 4: Verification and benchmarking

Add a benchmark (in `test/` or `bin/`) that:
1. Creates a float32 array of size 4096+
2. Runs an element-wise operation (e.g., `c[i] = a[i] + b[i]`) and a reduction (e.g., dot product)
3. Measures throughput with and without the vectorization-friendly changes
4. Verifies numerical correctness

The benchmark can use the existing `Utils.get_global_flag` mechanism to toggle the new features, comparing performance.

### What this task does NOT do

- **No explicit SIMD intrinsics** (`_mm256_fmadd_ps`, etc.) in generated code -- that belongs to the tiling task (watch-ocannl-README-md-347818d3).
- **No loop tiling or reordering** -- that is gh-ocannl-412.
- **No multi-threading** -- out of scope.
- **No AVX-512** -- AVX2 is the baseline; AVX-512 can be added later behind a feature flag.

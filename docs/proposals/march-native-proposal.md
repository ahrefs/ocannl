# Add `-march=native` C Compiler Parameter

GitHub issue: https://github.com/ahrefs/ocannl/issues/311

## Motivation

OCANNL's C backend compiles generated C code with `-O3` but no architecture-specific flags. This means the compiler generates generic x86-64 code that misses significant optimization opportunities:

- **No AVX/AVX2**: 256-bit SIMD operations (8 floats per instruction) are not used
- **No hardware FMA**: OCANNL generates `fmaf()` calls in C (`c_syntax.ml`), but without `-march=native` the compiler may not emit hardware FMA instructions
- **No architecture-specific scheduling**: Instructions are not ordered for the specific CPU's pipeline

The linked article ([Fast-MMM-on-CPU](https://siboehm.com/articles/22/Fast-MMM-on-CPU)) demonstrates that adding `-march=native` + `-ffast-math` to `-O3` reduced matrix multiplication runtime from 4.4s to 1.6s (2.75x speedup) by enabling AVX2/FMA on Haswell.

For OCANNL's generated code — nested loops over tensor dimensions with scalar float operations and `fmaf()` calls — `-march=native` enables the C compiler to auto-vectorize loops and use hardware FMA, which are the two biggest wins.

## Current State

The compiler invocation in `arrayjit/lib/cc_backend.ml` (line 84-86):

```ocaml
let cmdline : string =
  Printf.sprintf "%s %s -O%d -o %s %s > %s 2>&1" (compiler_command ()) f_path
    (optimization_level ()) libname kernel_link_flags temp_log
```

**Current flags:**
- `-O3` (configurable via `cc_backend_optimization_level`, default `"3"`)
- Link flags: `-bundle -undefined dynamic_lookup` (macOS) or `-shared -fPIC` (Linux)
- **No architecture flags** — no `-march`, `-mtune`, `-mavx`, `-mfma`, or `-ffast-math`

**Configuration system** (`utils.ml`): Multi-tier priority — command-line args > environment variables > config file > hardcoded defaults. Pattern: `Utils.get_global_arg ~default:"..." ~arg_name:"cc_backend_..."`.

Key files:
- `arrayjit/lib/cc_backend.ml` — compiler invocation (lines 59-100), `compiler_command()` (lines 20-38), `optimization_level()` (lines 17-18)
- `arrayjit/lib/utils.ml` — `get_global_arg` configuration system
- `arrayjit/lib/c_syntax.ml` — C code generation, including `fmaf()` calls
- `ocannl_config` — root-level config file
- `ocannl_config.example` — example config (documents available options)

## Proposed Change

Add `-march=native` to the C compiler invocation by default, with a configurable override for cross-compilation or portability.

**Acceptance criteria:**
- `-march=native` included in compiler command by default
- Architecture flags configurable via `cc_backend_arch_flags` (same multi-tier config system as other options)
- Optional `-ffast-math` flag via `cc_backend_fast_math` (opt-in, default off — changes numerical results)
- Works on macOS (Apple clang, both x86 and ARM) and Linux (gcc/clang)
- No regression in existing tests
- New config options documented in `ocannl_config.example`

**Edge cases:**
- **Apple Silicon**: Apple clang accepts `-march=native` on ARM and produces good code. The default should work without platform-specific logic.
- **Cross-compilation**: Users deploying to different machines must override with e.g. `cc_backend_arch_flags=-march=x86-64-v3` or `cc_backend_arch_flags=""`.
- **`-ffast-math` tradeoffs**: Enables FP associativity, reciprocal approximations, no NaN/Inf handling. Acceptable for ML training (inherently noisy) but should be opt-in.
- **Older CPUs**: `-march=native` on pre-AVX CPUs still generates the best possible code for that CPU, just without wide SIMD.

## Scope

**In scope:**
- Add `cc_backend_arch_flags` config parameter (default: `-march=native`)
- Add `cc_backend_fast_math` config parameter (default: `false`)
- Include flags in the `cmdline` string in `c_compile_and_load`
- Update `ocannl_config.example` with new options
- Test on available platform(s)

**Out of scope:**
- Explicit SIMD intrinsics (#164 — AVX/AVX2 intrinsics are a separate, much larger task)
- Benchmarking infrastructure (nice-to-have but not required)
- CPU tiling or loop reordering optimizations (v0.8 milestone)

**Related tasks:**
- gh-ocannl-164: Add AVX/AVX2 intrinsics — explicit SIMD, complementary to auto-vectorization from `-march=native`
- gh-ocannl-350: Loop hoisting — IR-level optimization; `-march=native` optimizes the generated machine code

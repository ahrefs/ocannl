# Proposal: Add AVX/AVX2 intrinsics to the C backend

Task: gh-ocannl-164
Issue: https://github.com/ahrefs/ocannl/issues/164

## Scope broadened: the CPU-improvements bundle (2026-07-05)

With the schedule layer landed (PRs #90/#91), this task's consumers are concrete: the S4
packed matmul sits at 3.8× / 9.0 GFLOP/s single-threaded on cc (256³), and after
`Privatize` the profiles on every backend point at the *loads* — exactly what this
bundle accelerates. Scope is broadened from "AVX floor" to the CPU-improvements bundle,
with `restrict` kept in scope and ordered **first**:

### 1. `restrict` — early, and cross-backend

Not CPU-only: `restrict` on cc kernel parameters, `__restrict__` on CUDA, `__restrict`
on Metal's pooled per-node pointers (Clang-based MSL; the derived pointers address
disjoint slab sub-ranges, which is all restrict asserts).

**Soundness vs. copy-less slices — analyzed 2026-07-05, safe by construction.** The
worry: a kernel receiving both a parent buffer and a zero-copy slice view of it would
make `restrict` a lie. This cannot happen: alias resolution runs at assignments→low-level
lowering (`resolve_alias` in `assignments.ml:226-240`) — every read/write of an alias
view is rewritten on the spot to a parent access with the batch index prepended
(recursively over chains), and an eligible slice's `Fetch Slice` lowers to `Noop`. Alias
tnodes therefore never appear in `Low_level.t` index vectors, hence never in
`traced_store`, hence never in `compile_proc`'s `ptr_params`: kernel parameter lists
contain only buffer-owning roots, and the parent+view parameter pair is unrepresentable.
Ineligible slices (precision mismatch etc.) materialize genuine copies — also fine.

Belt-and-braces, part of this task:

- **Assert the invariant where `ptr_params` is built**: `not (Tn.is_alias tn)` for every
  parameter. The invariant is enforced by assignments lowering, but the schedule layer
  and tests hand-build `Low_level.t` directly — a hand-built `Get` of an alias tn would
  silently mint an aliased parameter, and with `restrict` that is a miscompile instead
  of a redundant pointer. Fail loudly at compile.
- **Leave `restrict` off the merge-buffer parameter** (it is already `const`, which
  captures most of the benefit), or gate it on the `Copy` transfer mode after auditing
  whether any streaming/zero-copy merge mode can point the merge buffer at a live
  same-device buffer.
- Note: after `Privatize` (PR #91) the accumulator no longer needs `restrict` — a
  routine-local tile cannot alias kernel pointers — so the payoff here is on the loads.

### 2. Aligned allocation (Phase 1 below, unchanged)

CPU-only (drivers already align device buffers); also the prerequisite for aligned
vector loads and eventually `Padto`-style lane-multiple tiles.

### 3. `Vectorized` axis-type rendering as pragma hints

The reserved `Vectorized` axis type gets its first semantics here, schedule-driven
rather than heuristic: a `C_syntax_config` hook (in the spirit of `hardware_index`)
renders a `Vectorized`-typed loop as a pragma-annotated serial loop on cc
(`#pragma clang loop vectorize(enable)` / `#pragma GCC ivdep`) and as a plain serial
loop on backends without the hook — the same legal-fallback discipline as
`Grid`/`Workgroup`→serial. This supersedes Phase 3b's structural innermost-loop
detection: schedules (or the default annotator) decide which loops get hints, and the
rendering upgrades in place when explicit intrinsic codegen lands. The axis type itself
is backend-neutral — on CPU the payoff is compute (wide FMA), on GPU it is memory
transactions (float4/packed loads, notably for `Stage`'s cooperative loads — a
follow-up, not this task); the `Set_from_vec`/`vec_unop` machinery is the shared growth
path.

### 4. Explicit SIMD FMA micro-kernel foundations (Phases 2–3 below, unchanged)

Arch flags, platform-detection macros, alignment attributes — the floor for the
explicit-intrinsics follow-up to S4's packed matmul.

### Explicitly not in this bundle: CPU parallelism

**`Cpu_parallel` as an axis type is retired.** Within-routine CPU threading will bind
`Grid` axes to a task pool in the C backend's rendering (`dispatch_apply` on macOS,
avoiding Apple-clang's missing libomp; OpenMP or a small pthread pool elsewhere), with
`Workgroup` axes serial inside a task and barriers rejected in v1 (barrier support under
serialized workgroups requires loop fission at barrier boundaries — later, if needed).
`Grid`'s contract — independent iterations, `from_ = 0`, `validate_parallel` coverage —
is exactly the task-pool contract, so the default GPU annotator's output parallelizes
CPU with no new analysis; pool-specific knobs (chunk size, pinning) are renderer/launch
parameters, not axis semantics. Separate work item, as is the possible multicore_cc →
"CPU multi-device" repurposing (data-parallel debugging and GPU-less CI for the
multi-device machinery).

**Status update (2026-07-06): pool-backed Grid rendering implemented.** Eligible
outermost `Grid` loops render as contiguous chunks over `dispatch_apply` (macOS) or
`#pragma omp parallel for` (elsewhere; config `cc_parallel_grid`, "auto" probes the
compiler) — both process-global pools, so no pool state lives in the kernel `.so` and
the OCaml runtime is never involved. Eligibility (`C_syntax.collect_parallel_grid`)
keeps a loop serial under barriers, opaque statements, or runtime kernel logging, and
analyzes kernel-scope locals (per-thread on GPU, shared under the C serialization):
a local written under the loop must either touch disjoint cells per iteration (every
access mentions the grid index, with agreement on the mentioning components) or be
privatizable to per-chunk block-scope storage (all accesses inside the body and each
iteration's first access a covering write — `Privatize` accumulators and gh-ocannl-469
packed tiles; combined footprint capped for the pool workers' stacks). Under
`dispatch_apply`, blocks cannot capture array declarations, so non-privatized locals
accessed in the body are declared behind a `const` pointer alias. The new
`Schedule.default_cpu` preset (config `automatic_cpu_schedule`, and a threshold now named
`cpu_schedule_min_work`, gh-ocannl-1458) reuses the GPU annotator's analysis and just retypes each
nest's outermost chain loop to `Grid`. Workgroup loops stay serial inside a chunk;
barrier support via loop fission remains future work, as does the multi-device
repurposing. Coverage: `test/operations/cpu_parallel.ml`.

## Status update (2026-07-05): implemented

The CPU-improvements bundle landed on this branch; all four items plus verification:

1. **`restrict`, cross-backend**: `C_syntax_config` gains `restrict_keyword` — `restrict` on cc
   kernel pointer params (`Pure_C_config` default), `__restrict__` on CUDA, `__restrict` on
   Metal's pooled per-node derived pointers. The merge-buffer param stays unqualified (it is
   `const`). The belt-and-braces alias assert lives where `ptr_params` is built in
   `compile_proc`: a `Tn.is_alias` parameter raises `Invalid_argument` naming the restrict
   miscompile hazard (covered by `arrayjit/test/test_vectorized_codegen.ml`).
2. **Aligned allocation**: `Ops.buffer_alignment = 32` (single parameterized constant);
   `alloc_pool_raw` over-allocates via `Ctypes.allocate_n` and advances to the boundary —
   pointer arithmetic preserves the ctypes managed root, so calloc zeroing and GC lifetime
   semantics are unchanged. Within-pool offsets in `Backends.allocate_delta` pad to the same
   constant, so every node (not just pool bases) is aligned. Covered by
   `arrayjit/test/test_aligned_alloc.ml`; `test_buffer_loc.expected` shows the padded offsets.
3. **`Vectorized` axis type**: added to `Low_level.axis_type` (label `for@vectorized`,
   `hardware_kind_of_axis = None`, `Retype` accepts it with no `from_` restriction, `Stage`
   treats it like `Serial` tile loops). Rendering via `C_syntax_config.vectorize_pragma`:
   `Pure_C_config` emits `#pragma clang loop vectorize(enable) interleave(enable)` /
   `#pragma GCC ivdep` (compiler-guarded, `__clang__` first); CUDA/Metal override to `[]` →
   plain serial loop. Executed end-to-end in `test/operations/schedule_ops.ml` (`vec_inner`).
4. **Flags, macros, local alignment**: `cc_backend_simd_flags` (default `auto`) probes once per
   process — stage 1 checks the `arch_flags` target already defines `__AVX2__`/`__FMA__` (so the
   explicit flags never escalate the ISA: graceful fallback on non-AVX2 x86 and ARM), stage 2
   appends `-mavx2 -mfma -ftree-vectorize` or the accepted subset. `builtins_cc.ml` includes
   define `OCANNL_HAS_AVX2`/`OCANNL_HAS_NEON` and include `immintrin.h`/`arm_neon.h`.
   `aligned_local_attr` puts `__attribute__((aligned(32)))` on plain stack arrays (not
   workgroup-shared placements; the constant tracks `Ops.buffer_alignment`).
5. **Benchmark**: `bin/cpu_vectorization_bench.ml` — on Apple-Silicon NEON the compute-bound
   elementwise polynomial measures **2.0x** (36.5 vs 18.2 GFLOP/s) against
   `-fno-vectorize -fno-slp-vectorize` at the same `-O3 -march=native`; the strict-FP dot
   reduction is documented as needing `cc_backend_fast_math` for reassociation. Full test suite
   green on `sync_cc` and `metal`.

## Status update (2026-07-04)

- **Still not started**: no `restrict` qualifiers on kernel parameters, no
  vectorization pragmas, no `OCANNL_HAS_AVX2`/`OCANNL_HAS_NEON` macros in
  `builtins_cc.ml`, and `backend_impl.ml:79` still allocates via unaligned
  `Ctypes.allocate_n` (line moved from 48). `arch_flags` is unchanged
  (`cc_backend.ml:20`, flags assembled ~89).
- **Rebase targets moved** with the axis-types-for-loops landing (Phases B+C,
  2026-07-04): `compile_proc` is now `c_syntax.ml:1134-1354` and returns launch
  dimensions in addition to params and the function doc; Phase 3a's `restrict` change
  still lands in its `kparams`/`Kparam_ptr` construction. The local-declaration pass
  (Phase 3c's alignment attribute) now has a workgroup-shared branch
  (`c_syntax.ml:1328`, `shared_decl_prefix`) — the `__attribute__((aligned(32)))`
  applies to the plain stack-array branch only. `pp_ll` is at `c_syntax.ml:475` and
  its `For_loop` case (Phase 3b's pragma site) at 484 now branches on the loop's
  `axis` field — emit pragmas in the `Serial` rendering path (which also covers cc's
  serial fallback for `Grid`/`Workgroup` annotations, where the pragma remains valid).
- **Priority note**: the schedule layer ([schedule-ir-optops](schedule-ir-optops.md),
  elaborated 2026-07-04) plans CPU tiling/packing as its Phase S4 and explicit SIMD
  micro-kernels after that; this task is the auto-vectorization floor beneath both and
  is independently landable — nothing here depends on the schedule work.
- All Acceptance Criteria and the phased Approach remain valid as written.

## Status update (2026-06-12)

- Issue #164 is still OPEN, milestone v0.8 (ROADMAP targets mid-June 2026 for v0.8); gh-ocannl-412 (tiling) is also still OPEN on v0.8.
- Not yet started: no `restrict` qualifiers, vectorization pragmas, or `OCANNL_HAS_AVX2`/`OCANNL_HAS_NEON` macros exist anywhere in `c_syntax.ml`, `cc_backend.ml`, or `builtins_cc.ml`; `backend_impl.ml:48` still allocates via unaligned `Ctypes.allocate_n`.
- Identifiers re-verified at HEAD `d9de22f0`; several line numbers drifted (table below updated): `pp_ll` is now at `c_syntax.ml:331`, `For_loop` codegen at ~340-357, `compile_proc` at ~842-981, `loop_over_dims` at `low_level.ml:1929`. `For_loop` (line 38), `Set_from_vec` (line 41), `pp_array_offset` (line 275), `arch_flags` (cc_backend.ml:20), and `c_vec_typ_of_prec` (ops.ml:339) are unchanged.
- Kernel parameters were renamed from `params` to `kparams` (commit `9262ab44`, #356): Phase 3a's `restrict` change now lands in `compile_proc`'s `kparams` construction (`Kparam_ptr` entries, ~c_syntax.ml:847-881).
- A first-touch `Zero_out` elision landed in `c_syntax.ml` (gh-ocannl-420, `zero_out_seen` hash set cleared in `compile_proc`) — orthogonal to this proposal but nearby in the same functions; rebase Phase 3 around it.
- All Acceptance Criteria and the phased Approach remain valid as written.

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

1. **Low-level IR** (`arrayjit/lib/low_level.ml`): `For_loop { index; from_; to_; body; trace_it }` (line 38). `loop_over_dims` (line 1929) generates nested for-loops from dimension arrays. Loops iterate from `from_` to `to_` inclusive.

2. **C code generation** (`arrayjit/lib/c_syntax.ml`): `pp_ll` (line 331) emits C `for` loops. The loop index type is `uint32_t` (or `uint64_t` for big models). Array accesses use computed offsets via `pp_array_offset` (line 275).

3. **CC backend** (`arrayjit/lib/cc_backend.ml`): Compiles generated `.c` files with GCC/Clang. Compiler flags (lines 87-97): `-O3 -march=native` by default. The `arch_flags` setting (line 20) defaults to `-march=native` which already enables AVX2 on machines that support it, but the generated code is not structured for auto-vectorization.

4. **Memory allocation** (`arrayjit/lib/backend_impl.ml`): Uses `Ctypes.allocate_n int8_t ~count:size_in_bytes` (line 48), which calls `malloc` -- no alignment guarantee beyond default (typically 16 bytes on 64-bit systems, insufficient for AVX's 32-byte requirement).

5. **Precision types** (`arrayjit/lib/ops.ml`): `Single_prec` -> `float` (4 bytes), `Double_prec` -> `double` (8 bytes). The most common computation precision is `float` (single). AVX2 processes 8 floats or 4 doubles per instruction.

6. **Existing vector support**: `Set_from_vec` IR node (line 41) handles `Uint4x32_to_prec_uniform` conversion -- a fixed-width vector unop. The `c_vec_typ_of_prec` function (ops.ml line 339) defines struct-based vector types (`float4_t`, `double2_t`). These are portable but not SIMD-width.

### Key code locations

*(Line numbers re-verified 2026-06-12 at HEAD `d9de22f0`.)*

| Component | File | Line(s) | Relevance |
|-----------|------|---------|-----------|
| `For_loop` codegen | `arrayjit/lib/c_syntax.ml` | 340-357 | Where loop headers are emitted; add pragma hints here |
| `compile_proc` | `arrayjit/lib/c_syntax.ml` | 842-981 | Function generation; add `restrict` to pointer params (now called `kparams`, built via `Kparam_ptr`) |
| Compiler flags | `arrayjit/lib/cc_backend.ml` | 87-97 | Add conditional AVX2/FMA flags |
| `arch_flags` setting | `arrayjit/lib/cc_backend.ml` | 20 | Default `-march=native` |
| Memory allocation | `arrayjit/lib/backend_impl.ml` | 44-51 | Replace with aligned allocation (`Ctypes.allocate_n` at line 48) |
| Local array declarations | `arrayjit/lib/c_syntax.ml` | ~949-966 | `local_decls` in `compile_proc`; add alignment attribute |
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

3a. **`restrict` qualifiers on pointer parameters** (in `compile_proc`, around lines 847-881 where `kparams` / `Kparam_ptr` entries are built):

Change parameter declarations from `float *arr` to `float * restrict arr`. This tells the compiler that pointers don't alias, which is a prerequisite for auto-vectorization. OCANNL's memory model supports this: each tensor has its own allocation.

3b. **Vectorization pragma hints** (in `pp_ll`, around line 340):

Before the innermost `for` loop (detected by checking whether the loop body contains no nested `For_loop`), emit:

```c
#if defined(__GNUC__)
#pragma GCC ivdep
#elif defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#endif
```

Detection of "innermost loop" requires a simple check: scan the body for `For_loop` nodes. If none found, this is an inner loop candidate for the pragma.

3c. **Alignment attribute on local arrays** (in `compile_proc`'s `local_decls`, around lines 949-966):

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

*(Revised 2026-07-05 for the broadened scope; the original bullets predated the landed
schedule layer.)*

- **No explicit SIMD intrinsics** (`_mm256_fmadd_ps` / `vfmaq_f32`) in generated code.
  This bundle provides the floor — alignment, `restrict`, flags, detection macros, and
  `Vectorized`-as-pragma rendering — but real `Vectorized` codegen (intrinsic or vector-
  type emission through the `Set_from_vec`/`vec_unop` growth path) is the follow-up:
  the SIMD FMA micro-kernel under
  [watch-ocannl-README-md-347818d3](watch-ocannl-README-md-347818d3.md)'s remaining
  scope, composed with the S4 packed-tile schedules. *(Landed 2026-07-06: explicit
  vector-extension emission for eligible `Vectorized` bodies, config
  `cc_vector_bytes`; see that proposal's status note.)*
- **No loop transformations** — tiling, reordering, packing, and accumulator
  privatization landed with the schedule layer
  ([schedule-ir-optops](schedule-ir-optops.md), PRs #90/#91); this task changes only
  codegen and allocation beneath them. In particular it does not add or change optops,
  the default annotator, or tile-size presets.
- **No multi-threading** — but no longer "out of scope" of the roadmap, just of this
  bundle: within-routine CPU parallelism is pool-backed `Grid` rendering in the C
  backend (the `Cpu_parallel` axis type is retired, see the 2026-07-05 scope section),
  and the possible multicore_cc → CPU-multi-device repurposing is likewise separate
  work. Neither blocks nor is blocked by this task; `restrict` and alignment benefit
  the pool-rendered kernels unchanged.
- **No alignment-driven shape padding** — aligned allocation and aligned stack/local
  arrays are in scope, but padding tile extents to lane multiples so vector loads never
  straddle an edge is `Padto` territory (deferred in the schedule vocabulary; masking
  is representable today via `If` + interval discharge).
- **No AVX-512** — AVX2/NEON is the baseline; AVX-512 can be added later behind a
  feature flag (and would raise the alignment constant from 32 to 64 in Phase 1's
  helper — worth parameterizing now, hardcoding nothing).

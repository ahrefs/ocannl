# Efficiency lessons from ggml — research notes

Deliverable for [gh-ocannl-163](../proposals/gh-ocannl-163.md)
([ahrefs/ocannl#163](https://github.com/ahrefs/ocannl/issues/163)): what OCANNL can
learn from [ggml](https://github.com/ggml-org/ggml) for CPU efficiency, converted
into per-technique verdicts and follow-up drafts.

**ggml is not a backend candidate.** The user judged it "not flexible enough" in
2024-09 (see the discussion on
[#163](https://github.com/ahrefs/ocannl/issues/163)); this note extracts *lessons*,
not adoption plans. ggml is hand-written kernels + a fixed-order graph walker;
OCANNL is a compiler that generates C/CUDA/MSL from a shape-inferred einsum IR with
a schedule-transformation layer and autotuner. Many ggml techniques therefore map
not to "port this code" but to "make the compiler emit / the runtime provide the
equivalent".

**Provenance.** Written 2026-07-07. ggml was read from its actively developed copy
inside `ggml-org/llama.cpp`, branch `master`, commit
[`c198af4d`](https://github.com/ggml-org/llama.cpp/tree/c198af4dc24f8e0ab8a569a60f931e03a192fd79)
("spec : fix naming, spacing (#25410)", 2026-07-07) — files under `ggml/src/` plus
the loader under `src/`. All ggml claims cite file + function/struct names at that
commit. OCANNL surfaces were verified at HEAD `605d565a` of this repository; all
cited OCaml symbols were grepped at that commit.

**Quantization constraint.** [gh-ocannl-137](../proposals/gh-ocannl-137.md)
(quantization) was **closed NOT-PLANNED** (disposition memo recommendation (A):
nothing on the roadmap before v1.1 needs quantization). Quantization verdicts below
are therefore `future` / `not applicable` / `file follow-up issue` — never "covered
by 137".

**Driver workloads.** The framing target is CPU inference for the just-promoted
driver tasks: GPT-2 Small ([#377](https://github.com/ahrefs/ocannl/issues/377),
proposal [gh-ocannl-377](../proposals/gh-ocannl-377.md)) and a Gemma-class
follow-up (RoPE landed via #398; #377's issue title is "inference for one of:
Llama, GPT-2, Gemma"). This is exactly ggml's home turf — single-machine CPU
transformer inference — which is why the comparison is worth doing now.

---

## 0. The ground shifted: OCANNL state as of `605d565a`

The task proposal's 2026-06-12 status update is stale. The following landed since
and materially change the verdicts (each verified in this repo):

| Landed | Where (verified) |
|---|---|
| Aligned allocation: 32-byte via over-allocate + advance | `arrayjit/lib/backend_impl.ml` `alloc_pool_raw`, `Ops.buffer_alignment = 32` (`ops.ml`); per-node offsets padded to alignment in `Backends.allocate_delta` |
| `restrict`, vectorize pragmas, alignment attributes | `arrayjit/lib/c_syntax.ml` `restrict_keyword`, `vectorize_pragma`, `aligned_local_attr` in the `C_syntax_config` |
| Explicit SIMD codegen: `Vectorized` loops render as GCC/Clang vector extensions | `c_syntax.ml` `try_vectorize` — `__attribute__((vector_size(N)))` typedefs, vector load/arith/store + serial remainder; FMA via `__builtin_elementwise_fma` (guard `OCANNL_HAS_ELEMENTWISE_FMA`); width from config `cc_vector_bytes` (`cc_backend.ml` `vector_bytes_setting`: auto = 32 if the AVX2 probe passes, else 16) |
| Eligibility contract: only `Add/Sub/Mul/Div/Neg` + fused `FMA`, chosen so vector results match the serial path bit-for-bit | `c_syntax.ml` `try_vectorize` doc comments |
| Schedule IR: `Split`, `Retype`, `Unroll`, `Stage` (packing), `Privatize`, `Tensorize` (→ `Tile_mma` blocks); CPU preset `default_cpu` (config `automatic_cpu_schedule`, `cpu_schedule_min_work`); kernel fission (`schedule_fission`) | `arrayjit/lib/schedule.ml` |
| Autotuner with measured candidate selection, schedule cache, matmul sketch seeds | `arrayjit/lib/autotune.ml` `tune`, `sketch_candidates`; `arrayjit/lib/schedule_cache.ml` |
| CPU intra-kernel parallelism: eligible outermost `Grid` loops render over process-global pools — `dispatch_apply` (macOS) or `#pragma omp parallel for` | `c_syntax.ml` `parallel_grid_safe`, `parallel_grid_syntax` (`` `Dispatch``/`` `Openmp``), configs `cc_parallel_grid`, `cc_parallel_chunks` |
| Interval analysis (guard folding, bounds) | `arrayjit/lib/interval.ml`; `low_level.ml` `interval_of_*` |
| Signed `int32`/`int64` index precision | `ops.ml` `index_prec` |
| Tensor persistence (checkpoint save/load) | `lib/persistence.ml`; binary payload I/O in `arrayjit/lib/ndarray.ml` |
| Merge-buffer static verification, `merge_buffer_use = No \| Copy` | `arrayjit/lib/backend_intf.ml` |

Still-open gaps confirmed at `605d565a` (these are where several verdicts land):

- **No SIMD reductions**: `try_vectorize` handles elementwise chains only; there is
  no horizontal-reduction emission, and the `Vectorized` doc comment notes the
  auto-vectorizer cannot reassociate strict-FP reductions (the retype "carries that
  permission" — unexercised).
- **No vector matmul micro-kernel**: `Tensorize`→`Tile_mma` names the tiles and the
  autotune sketch seeds candidates, but the cc renderer does not emit a
  register-tiled vector FMA kernel for `Tile_mma` bodies.
- **Backends at this commit** are `sync_cc` and `multicore_cc` (default
  `multicore_cc`; `arrayjit/lib/backends.ml` `type backend = Sync_cc |
  Multicore_cc | Cuda | Metal`). `schedulers.ml` `Multicore` remains one worker
  `Domain` per stream with an SPSC queue — *inter*-routine threading; the
  *intra*-kernel parallelism above is separate and new. (A `cc`/`multidev_cc`
  rename is in flight on another branch; not present at this commit.)
- **Persistence loads by copying**, not mmap: `Ndarray.read_payload_from_channel`
  does `Stdlib.really_input` into a fresh buffer (`ndarray.ml`); no
  `Unix.map_file` anywhere in `persistence.ml`/`ndarray.ml`/`tnode.ml`.
- **No int8/int4 types**: `ops.ml` precisions are `Void, Byte (uint8), Uint16,
  Int32, Uint32, Int64, Uint64, Uint4x32, Half, Bfloat16, Fp8, Single, Double` —
  no signed int8, no 4-bit weight type.
- **GPT-2 driver still missing** GeLU, the weight-conversion script, and the
  inference executable ([gh-ocannl-377](../proposals/gh-ocannl-377.md) status).

[gh-ocannl-164](../proposals/gh-ocannl-164.md) (AVX/AVX2) is the load-bearing
"already covered" target for intrinsics-level items: its CPU-improvements bundle
(restrict, alignment, pragmas, flags/macros) landed 2026-07-05/06 per its status
updates, with explicit vector-extension emission noted as landed there too.

---

## 1. Block quantization with shared scales

### What ggml does

Weights are stored in fixed-size blocks that share quantization metadata. Formats
(struct layouts in
[`ggml/src/ggml-common.h`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/ggml-common.h),
each with a compile-time `static_assert` on its byte size):

- **`block_q4_0`** (`QK4_0 = 32`): one fp16 scale `d` + 16 bytes of nibbles = 18
  bytes / 32 weights = **4.5 bits per weight**. Decode is `w = d * (q - 8)`.
  Nibbles are split-half packed (byte *j* holds elements *j* and *j*+16), so
  unpacking is one mask + one shift, no shuffles.
- **`block_q8_0`** (`QK8_0 = 32`): fp16 `d` + 32 int8 = 34 bytes = 8.5 bpw.
- **`block_q4_K`** (`QK_K = 256` super-block): two fp16 super-scales (`d`,
  `dmin`) + twelve bytes packing 8×6-bit sub-scales and 8×6-bit sub-mins + 128
  quant bytes = 144 bytes / 256 = **4.5 bpw** with per-32 asymmetric adaptivity
  (the k-quants design, [PR #1684](https://github.com/ggml-org/llama.cpp/pull/1684)).
- **`block_q8_K`** — the *transient activation* format: fp32 scale + 256 int8 +
  precomputed per-16 `bsums`; "only used for intermediate quantization and dot
  products" (comment in `ggml-common.h`), never stored to disk.

The matmul over these formats never dequantizes to float arrays. Activations are
quantized on the fly, once per row, to the weight type's `vec_dot_type`
(`type_traits_cpu[]` in
[`ggml-cpu.c`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/ggml-cpu/ggml-cpu.c):
Q4_0 → `.vec_dot = ggml_vec_dot_q4_0_q8_0, .vec_dot_type = GGML_TYPE_Q8_0`), then
the inner loop is pure integer MAC. AVX2 path of `ggml_vec_dot_q4_0_q8_0`
([`arch/x86/quants.c`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/ggml-cpu/arch/x86/quants.c)):
nibble expand, `_mm256_sub_epi8(·, 8)`, then the signed×unsigned trick
(`_mm256_sign_epi8` twice + `_mm256_maddubs_epi16`, collapsing to
`_mm256_dpbssd_epi32` on VNNI-INT8), and exactly **one float FMA per 32-element
block**: `acc = _mm256_fmadd_ps(d, q, acc)` with `d = fp16(x.d) * fp16(y.d)`
hoisted. NEON uses chained `vdotq_s32` (SDOT), and on ARMv8.6-i8mm the kernels
consume two rows at once via `vmmlaq_s32` (`.nrows = 2`). K-quants go further: the
6-bit sub-scales are applied by *integer* `_mm256_madd_epi16` while still in the
int32 domain, and the asymmetric min-offset term is computed against Q8_K's
precomputed `bsums` without touching quants at all.

Why fp16 scales: metadata is pure overhead — fp32 scales would make q4_0 5.0 bpw
instead of 4.5 (the `iq2_xxs` comment in `ggml-common.h` calls this out
explicitly), fp16's precision dwarfs 4-bit quantization noise, and the scale is
converted and fused once per block so the fp16 decode cost is O(blocks). Where a
scale *does* need range — the never-persisted Q8_K activation format — ggml spends
fp32 on it. Importance-matrix quantization (`quantize_q4_0(..., quant_weights)` in
`ggml-quants.c`) changes only the *encoder* (weighted scale search); files and
kernels are bit-identical — encoding quality is decoupled from decoding speed.

### Relevance to OCANNL

For memory-bandwidth-bound CPU decode (one token at a time), weight bytes are the
whole game: 4.5 bpw vs 32 bpw is a ~7× reduction in the traffic that dominates
per-token latency. This is the single biggest reason llama.cpp is usable on
laptops. For the *current* drivers it matters less: GPT-2 Small is 124M params
(~0.5 GB fp32 — fits everywhere; #377's target is 50 tokens < 60 s, not
bandwidth-saturated decode), and even Gemma-2B in bf16 is ~5 GB. Quantization
becomes decisive at the 7B+ scale, which is past the current roadmap horizon —
consistent with #137's not-planned closure.

### Mapping to OCANNL surface

If/when revisited: a block-quantized tensor is a *layout*, not just a precision —
one logical axis maps to (blocks × metadata + payload) physically. The natural
seams: a new `prec` in `arrayjit/lib/ops.ml` is *not* sufficient (unlike `Fp8`,
the element size is fractional and metadata is interleaved); it would need either
a composite-node representation (scales tensor + payload tensor, the way
`Uint4x32` already packs 4 lanes) or a layout descriptor on `tnode.ml`.
Dequantize/dot kernels would land as `builtins_cc.ml` helpers plus `c_syntax.ml`
rendering, with `Tnode.Placements`-style compile-time knowledge that a tensor is
read-only quantized (the constants path in `Backends.allocate_delta` already
segregates read-only/constant nodes into per-device constant pools —
`constant_buffer_cache`). The ggml design principles worth keeping verbatim:
block size 32 = one AVX2 register; scales factor out (`Σ d_x q_x d_y q_y = d_x
d_y Σ q_x q_y`); richer transient activation format than storage format;
super-blocks amortize metadata to 0.125 bpw.

### Verdict: `future`

Revisit post-v1.1, gated on a driver model whose weight traffic exceeds the
memory budget in the intended precision (≥7B class, or Gemma-scale on small
machines). #137's closure is recorded here as the controlling decision; no new
issue is filed now to avoid re-opening it by the back door. When it is revisited,
this section plus §7 (repack) is the design seed. Sub-verdicts: on-the-fly
activation quantization and int8-domain dot kernels — `future` (same gate);
imatrix-weighted encoding — `not applicable` until a quantizer exists (it is
offline tooling, decoupled from kernels by design).

---

## 2. SIMD intrinsics and micro-kernel organization

### What ggml does

One generic loop body, written once, over a macro vocabulary
(`GGML_F32_VEC`, `_LOAD`, `_STORE`, `_FMA`, `_REDUCE`, `_STEP`, `_EPR`,
`_ARR`), with a per-ISA `#if` chain defining the vocabulary for SVE, NEON,
AVX-512, AVX/AVX2, POWER9, WASM, SSE3, LoongArch, s390x, RISC-V
([`simd-mappings.h`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/ggml-cpu/simd-mappings.h):
"we define a common set of C macros which map to specific intrinsics based on the
current architecture"). `ggml_vec_dot_f32`
([`vec.cpp`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/ggml-cpu/vec.cpp))
keeps `GGML_F32_ARR` (= 4 on AVX2/NEON) **independent vector accumulators** in a
constant-bound inner loop the compiler fully unrolls — this is how both unrolling
and FMA-latency hiding are expressed — then a tree `REDUCE` and a scalar tail.
Scalar accumulation is `double` (`ggml_float`). FP16 is capability-tiered: native
fp16 arithmetic where available (NEON `vfmaq_f16`, AVX512FP16 — accepting fp16
accumulation range risk), hardware convert-to-fp32-lanes next (F16C
`_mm256_cvtph_ps`), a 64K lookup table (`ggml_table_f32_f16`) as the floor.

Dispatch is two-level: compile-time `#ifdef` inside a build, and — for binary
distribution — `GGML_CPU_ALL_VARIANTS` builds one shared object per microarch
level (x64 `sse42`…`sapphirerapids`, ARM `armv8.0`…`armv9.2`), each exporting a
CPUID self-score, and `ggml_backend_load_best`
([`ggml-backend-reg.cpp`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/ggml-backend-reg.cpp))
dlopens the max-scoring variant. A rename header
([`arch-fallback.h`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/ggml-cpu/arch-fallback.h))
aliases `_generic` C implementations onto any kernel symbol an arch lacks.

### Relevance to OCANNL

This is the area where OCANNL's architecture already answers ggml's problem
differently and mostly better. ggml needs the macro vocabulary and fat-binary
dispatch because it ships *pre-compiled hand-written kernels*. OCANNL JIT-compiles
generated C on the host with `-march=native` (`cc_backend.ml` `arch_flags`) plus
the once-per-process probed `-mavx2 -mfma -ftree-vectorize` subset (config
`cc_backend_simd_flags`, per the gh-164 bundle), so per-host ISA specialization
is free and runtime dispatch is a non-problem. The GCC/Clang vector-extension
emission that landed (`try_vectorize`) is precisely the "write the loop once, let
the vocabulary map per arch" idea — with the compiler, rather than a macro table,
as the vocabulary.

What ggml still teaches, because the landed vectorizer doesn't do it yet:

1. **Reductions.** ggml's dot kernels exist *because* strict-FP reductions don't
   auto-vectorize; the multi-accumulator + horizontal-reduce pattern is the whole
   trick. OCANNL's `Vectorized` retype explicitly carries the reassociation
   permission (doc comment in `c_syntax.ml`) but the renderer never uses it — a
   reduction body falls back to the serial/pragma path, and the gh-164 benchmark
   documents the dot product needing `cc_backend_fast_math`. Vector-extension code
   can express ggml's exact pattern portably: N accumulator vectors, one
   fused-loop, lane-sum at exit.
2. **Independent accumulator chains** (ARR > 1) as the unroll mechanism — maps
   directly onto `Unroll` (schedule.ml) composed with `Vectorized`, but the
   renderer must keep the accumulators in distinct vector variables for the
   latency-hiding to materialize.
3. **FP16 tiers.** OCANNL already has the conversion tiers in `builtins_cc.ml`
   (`HAS_NATIVE_FLOAT16` fast path, `HALF_TO_FLOAT`/`FLOAT_TO_HALF` emulation) —
   the storage story is covered. ggml's further step (fp16 *arithmetic* on NEON,
   fp16 lanes in vectors — `c_vec_typ_of_prec` already names `half8_t`) is unexercised.

### Mapping to OCANNL surface

`arrayjit/lib/c_syntax.ml` `try_vectorize` (extend eligibility to accumulation
patterns; emit multi-accumulator + horizontal reduce under the `Vectorized`
reassociation permission); `arrayjit/lib/schedule.ml` `Unroll`/`Retype`;
`arrayjit/lib/builtins_cc.ml` (`OCANNL_HAS_AVX2`/`OCANNL_HAS_NEON` exist; add
lane-sum helpers if vector extensions' missing horizontal ops need per-ISA
snippets); `ops.ml` `c_vec_typ_of_prec` for fp16/bf16 lanes.

### Verdict: `already covered by gh-ocannl-164` (landed bundle + vector-extension emission), **with two follow-up issues for the open remnants**

The 164 proposal's acceptance criteria — flags, alignment, restrict, pragmas,
detection macros, ≥2× elementwise benchmark — are all landed per its status
updates and verified in-repo (§0). The remnants get drafts F2 (SIMD reductions)
and F3 (register-tiled micro-kernel, §3) below. Multi-variant fat-binary dispatch
and `arch-fallback.h` renaming: `not applicable` — JIT-on-host makes them moot
(OCANNL's analog of the fallback floor is the pragma/serial rendering path, which
already exists).

---

## 3. Matmul strategy: two regimes, one seam

### What ggml does

`ggml_compute_forward_mul_mat`
([`ggml-cpu.c`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/ggml-cpu/ggml-cpu.c))
has a decision tree: (1) try **llamafile/tinyBLAS** `llamafile_sgemm`
([`llamafile/sgemm.cpp`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/ggml-cpu/llamafile/sgemm.cpp),
Justine Tunney's contribution, header: "designed to have excellent performance for
matrices that fit in the CPU cache without imposing any overhead such as cache
filling or malloc calls"); it refuses `n < 2` ("only enable sgemm for prompt
processing"); (2) quantize src1 rows into the work buffer (`from_float` →
`wdata`), barrier, retry sgemm on the quantized operands; (3) fall back to the
chunked **vec_dot loop**: one dot product per output element, 16×16 block tiling,
chunks self-served via one relaxed `atomic_fetch_add` on
`threadpool->current_chunk`.

tinyBLAS is a textbook register-blocked GEMM made type-generic by tiny overload
sets (`madd`, `load<V>`, `hsum`): template `tinyBLAS<KN, D, V, TA, TB, TC>`,
micro-kernel `gemm_bloc<RM,RN>` holding an **RM×RN tile of C in vector registers
for the entire k-loop** — 4×6 on 32-register ISAs (NEON/AVX512), 4×3 on 16 — so
each k-step does RM+RN loads for RM×RN FMAs, with 24 independent accumulator
chains hiding FMA latency, zero branches and zero C-traffic inside the loop. Edge
tiles are separately instantiated template kernels (compile-time `mnpack`
descent), not masks or branches. So: **decode (n=1, memory-bound) → vec_dot;
prefill (n≥2, compute-bound) → register-tiled GEMM** — and the fast path is an
*optional accelerator* that returns `false` into a correct fallback.

### Relevance to OCANNL

This is the profile-identified gap: the gh-164 status note records the S4 packed
matmul at 3.8×/9.0 GFLOP/s single-threaded on cc with loads dominating after
`Privatize`. OCANNL has the *schedule* half of tinyBLAS (`Split`, `Stage`
packing, `Privatize` accumulators, `Tensorize`→`Tile_mma` naming the micro-tile,
autotune `sketch_candidates` seeding matmul tile candidates) but not the
*rendering* half: no register-tiled vector-FMA emission for `Tile_mma` bodies,
and `try_vectorize` can't express the k-loop reduction (§2). The two regimes map
cleanly onto OCANNL's autotune: decode and prefill are different shapes, and
`Autotune.tune` already keys measured schedules per compile (schedule-cache
digests), so regime-specific schedules fall out — *provided* GEMV-shaped
schedules are in the candidate space and the KV-cached decode graph is actually
compiled separately from prefill (a #377 pipeline decision).

### Mapping to OCANNL surface

`arrayjit/lib/schedule.ml` (`Tensorize`, `Tile_mma`), `arrayjit/lib/autotune.ml`
(`sketch_candidates` — currently rank-2 v1), `arrayjit/lib/c_syntax.ml` (a
`Tile_mma` rendering path that keeps the RM×RN accumulator tile in vector
variables; the `docs/proposals/tensorize-mma.md` seam). The ggml numbers to steal:
micro-tile sized to fill the register file (RM×RN + RM + RN vectors ≤ 32/16
registers); accumulate-in-registers across the whole k-loop; peel edges into
separate kernels.

### Verdict: `file follow-up issue` (draft F3)

The vec_dot-loop fallback regime itself (chunked, block-tiled) is
`already covered` in spirit by `default_cpu` + pool-backed `Grid` + `Split`
tiling; the decode/prefill regime split is `already covered` by shape-keyed
autotuning with a pipeline caveat recorded in F5's comment draft (no new
mechanism needed).

---

## 4. Threading model

### What ggml does

A **persistent thread pool** (`struct ggml_threadpool`,
[`ggml-cpu.c`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/ggml-cpu/ggml-cpu.c);
persistent since [PR #8672](https://github.com/ggml-org/llama.cpp/pull/8672),
2024-08): workers created once, then BSP execution — every thread walks the same
topological node list, splits each op's work by `(ith, nth)` or the atomic chunk
counter, and meets at `ggml_barrier` after every non-trivial node. The barrier is
a sense-reversing counter with pure spin (`_mm_pause`/`yield`, relaxed loads in
the spin, seq-cst fences at the edges); the three hot atomics are each
cache-line-aligned. Between graphs, a hybrid poll(`131072 × poll` relax
iterations)-then-condvar wait. Work splitting in mul_mat is **static start +
dynamic stealing**: thread *i* starts at chunk *i*, the shared counter is seeded
at `nth`, threads `fetch_add` for more — degrading to a pure static split when
chunks are scarce, and *forced* static under NUMA for locality. NUMA support:
node enumeration, `pthread_setaffinity_np` strategies, first-touch placement,
and mmap prefetch disabled under NUMA (§5). Under `GGML_USE_OPENMP` the pool is
replaced by one `#pragma omp parallel` region and `#pragma omp barrier` — OpenMP
is only the launcher; work division stays explicit.

### Relevance to OCANNL

The task elaboration's "work-stealing CPU thread pool" gap **closed** while the
proposal was in flight: eligible outermost `Grid` loops now render over
process-global pools — `dispatch_apply` on macOS, `#pragma omp parallel for`
elsewhere (`parallel_grid_safe` / `parallel_grid_syntax` in `c_syntax.ml`;
`Schedule.default_cpu` retypes each nest's outermost chain loop to `Grid`,
config-gated by `automatic_cpu_schedule`). This is the same design point ggml's
OpenMP mode occupies: delegate pool lifecycle to a mature runtime, keep the
splitting semantics yourself. libdispatch does dynamic chunk balancing
internally; the ggml lesson that remains is small: (a) the OpenMP rendering
should consider `schedule(dynamic, chunk)` for skewed iterations (ggml's
stealing counter is exactly that), and (b) barrier-free execution *within* a
routine matters more than barrier micro-optimization — OCANNL's kernel fission +
event chains already avoid ggml's barrier-per-node structure on GPU backends,
and the C serialization of `Workgroup` loops avoids intra-kernel barriers on CPU
by construction. NUMA/affinity has no OCANNL story, but none of the driver
workloads (desktop/laptop CPU inference) need one.

### Mapping to OCANNL surface

`arrayjit/lib/c_syntax.ml` (`parallel_grid_syntax`, `parallel_grid_chunks`),
`arrayjit/lib/cc_backend.ml` (configs `cc_parallel_grid`, `cc_parallel_chunks`),
`arrayjit/lib/schedule.ml` (`default_cpu`). Inter-routine layer unchanged:
`schedulers.ml` `Multicore` = one `Domain` per stream (SPSC queue), `Sync` =
main-thread; default backend `multicore_cc` (`backends.ml`).

### Verdict: `already covered` (pool-backed `Grid` rendering, landed under the gh-ocannl-164 bundle's scope note)

Sub-verdicts: spin-barrier/poll engineering — `not applicable` (delegated to
libdispatch/OpenMP runtimes; OCANNL emits no hand-rolled pool);
atomic-chunk-stealing — `already covered` via libdispatch's internal balancing,
with the OpenMP `schedule(dynamic)` option recorded as a one-line residue inside
F2/F3 benchmarking rather than its own issue; NUMA + affinity — `future`
(revisit if a many-socket serving milestone ever appears; prerequisite: a
benchmark machine where it is measurable).

---

## 5. Memory strategy: mmap, arenas, planned reuse

### What ggml does

**The file is the allocation.** GGUF stores tensor payloads at offsets padded to
`GGUF_DEFAULT_ALIGNMENT = 32` (overridable via the `general.alignment` KV;
validated power-of-two —
[`gguf.cpp`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/gguf.cpp)).
`llama_mmap`
([`llama-mmap.cpp`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/src/llama-mmap.cpp))
maps the whole file (`MAP_SHARED` + `MAP_POPULATE` prefetch,
`posix_madvise(WILLNEED)`; prefetch disabled under NUMA in favor of first-touch;
`munmap` of the metadata range after load; optional incremental `mlock`), and the
model loader sets `tensor->data` **directly into the mapping**
(`ggml_backend_tensor_alloc(buf_mmap, cur, mapping->addr() + offs)` in
`llama-model-loader.cpp`, with the mapping range wrapped as a backend buffer via
`ggml_backend_dev_buffer_from_ptr` — the same path that lets Apple-silicon Metal
use the file mapping as a GPU buffer). Zero copies for host-resident weights;
"loading" is page-cache population; weights are shared across processes.
Graceful degradation: no-mmap platforms fall back to buffered reads, discrete-GPU
targets to rotating pinned staging buffers, keyed off backend capability bits.

**No allocation on the hot path.** Tensor/graph metadata comes from a bump-arena
(`ggml_new_object`); per-op scratch is one work buffer sized to the **max** over
nodes (not the sum) at plan time, padded per-thread by a cache line. Activation
memory goes through a **measure-then-replay** graph allocator
([`ggml-alloc.c`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/ggml-alloc.c)):
a virtual best-fit allocator with free-block coalescing runs the graph once,
freeing each tensor at its last use (refcount walk) and reusing in-place where
legal (`ggml_op_can_inplace` + single-consumer check); real buffers are then
sized to the virtual high-water mark and offsets replayed — steady-state decoding
does zero allocator work.

### Relevance to OCANNL

Three distinct lessons, at three different distances:

1. **mmap weights — the confirmed gap, now with a concrete consumer.**
   `Persistence.load` reads payloads via `Ndarray.read_payload_from_channel` =
   `Stdlib.really_input` into a fresh buffer (verified §0), then `from_host`
   copies host→context buffer at link time — two copies of every weight byte.
   For GPT-2 Small this is ~0.5 GB of avoidable copying per run of the demo
   (startup latency, not throughput); for a Gemma-class model it is the
   difference between instant and tens-of-seconds startup, plus page-cache
   sharing across runs — directly serving #377's "50 tokens < 60 s" budget,
   whose clock includes load. OCANNL's checkpoint format (s-exp header + binary
   payload) needs GGUF's one design courtesy: payload offsets padded to a declared
   alignment (32 for SIMD; page-size for mapping the region a Bigarray can wrap).
   `Unix.map_file` yields exactly the `Bigarray.Genarray` that hosted `Ndarray`s
   already are. Full zero-copy (context buffers pointing into the mapping, ggml's
   `buffer_from_host_ptr`) is a second step gated on the constants path: read-only
   nodes already flow into per-device constant pools (`constant_buffer_cache` in
   `backends.ml` `allocate_delta`), which is where a "wrap this host pointer"
   variant would land for CPU backends.
2. **Arenas / no per-op malloc — already OCANNL's shape.** Tnodes pack into
   bump-offset pool slabs planned by the pure `plan_pool_segments`
   (`backends.ml`), 32-byte-aligned per node; pools are keyed and freed
   wholesale (`Make_slab` in `backend_impl.ml`); all planning is compile/link
   time, nothing allocates per run. ggml's "measure then replay" is what
   OCANNL's static compilation does by construction.
3. **Liveness-based reuse — the one arena feature OCANNL lacks.**
   `allocate_delta` bump-assigns every in-context tnode a *disjoint* region for
   the context's lifetime; ggml's gallocr overlaps tensors with disjoint
   live-ranges. OCANNL mitigates differently (virtual nodes vanish at lowering;
   `Local` nodes are routine-scoped stack/scratch), so the residue is only
   *materialized unobservable intermediates* — e.g. fission-segment boundaries —
   which are small for the driver models.

### Mapping to OCANNL surface

`lib/persistence.ml` (`save`/`load` — add aligned payload layout + mmap-backed
load), `arrayjit/lib/ndarray.ml` (`read_payload_from_channel` sibling that wraps
`Unix.map_file`), `arrayjit/lib/tnode.ml` (memory modes are
`Effectively_constant | Virtual | Never_virtual | Local | On_device` — file-backing
is a property of the *hosted array*, so no new mode is strictly required for step
1), `arrayjit/lib/backends.ml` (`constant_buffer_cache` for the zero-copy step 2),
`plan_pool_segments` (liveness intervals, if ever).

### Verdict: `file follow-up issue` for mmap weight loading (draft F1)

Arena/no-malloc and measure-then-replay: `already covered`
(`backend_impl.ml`/`backends.ml` pool planning). Liveness-based buffer reuse:
`future` — prerequisite: an inference or training graph where materialized
intermediates dominate memory (not GPT-2/Gemma-CPU scale); the seam
(`plan_pool_segments` is a pure planner that could take live-ranges) is noted so
the option stays cheap.

---

## 6. Graph execution: the no-search baseline

### What ggml does

`ggml_cgraph` is flat arrays in DFS post-order (`ggml_visit_parents_graph`,
[`ggml.c`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/ggml.c));
execution is strictly that order, one barrier per node, no scheduling search of
any kind. The only dynamism is runtime op-fusion peephole
(`ggml_cpu_try_fuse_ops`) and the multi-backend splitter
(`ggml_backend_sched`: greedy "follow the weights" assignment, flood-fill
expansion, auto-inserted copies with events — a placement heuristic, not a
search). Op implementations are one big hand-written `switch` (480
`ggml_compute_forward_*` in `ops.cpp`).

### Relevance to OCANNL / Verdict: `not applicable` (inverse lesson)

This is the architectural fork: ggml demonstrates that for one well-understood
workload family, a fixed executor + heroic kernels beats generality — but every
new op/ISA/format costs hand-written C, which is precisely the cost OCANNL's
compiler+autotuner architecture exists to avoid (and #163's premise: extract
lessons, don't adopt). OCANNL's equivalents are all compile-time: virtual-node
inlining subsumes runtime op fusion; kernel fission + event chains subsume the
barrier-per-node structure; `ggml_backend_sched`'s "run ops where the weights
live" placement echoes OCANNL's context/stream model with static merge-buffer
verification (`backend_intf.ml` `merge_buffer_use = No | Copy`). The one portable
observation: ggml sizes per-op scratch as a **max** over the graph, exploiting
sequential execution — OCANNL's per-routine `Local` scratch is per-kernel;
under kernel fission a shared max-sized scratch across segments would be the
analogous trick if scratch ever shows up in profiles (recorded as a residue, not
an issue).

---

## 7. Cache-aware layouts: repack weights once, at load time

### What ggml does

The CPU backend exposes an "extra buffer type" (`CPU_REPACK`,
[`repack.cpp`](https://github.com/ggml-org/llama.cpp/blob/c198af4dc24f8e0ab8a569a60f931e03a192fd79/ggml/src/ggml-cpu/repack.cpp)):
when weights are uploaded at model load, its `set_tensor` hook rewrites them from
canonical Q4_0/Q4_K blocks into ISA-shaped interleaved tiles —
`block_q4_0x4`/`x8` etc. interleave N *consecutive rows'* blocks so one vector
load pulls corresponding bytes of N rows (`make_block_q4_0x8`), chosen by CPU
feature detection (`ggml_repack_get_optimal_repack_type`: AVX2 → 8×8, NEON+i8mm
→ 4×8, NEON+dotprod → 4×4…). Two costs are pre-paid at repack time: the layout
transform itself, and even the nibble bias (quants XORed with `0x88` so the
kernel skips the `sub 8` — "saves subtract operations during unpacking"). The
file on disk stays canonical; matmuls on repacked tensors route to true
`gemv`/`gemm` micro-kernels computing N×4 output tiles. The general principle:
**layout specialization is a load-time, hardware-dispatched concern — pay per
model-load, never per matmul, and never in the file format.**

### Relevance to OCANNL

OCANNL's `Stage` transform packs operand tiles into scratch *inside* the routine
— paid per kernel invocation. That is the right call for activations (they
change every call) but wrong for inference weights, which are constant across
thousands of forward passes: a GPT-2 decode step re-packs the same fp32 weight
tiles on every token. OCANNL has all the pieces to hoist this: constancy is
already compile-time knowledge (`Tnode.Placements.known_constant`, the
read-only/constant partition in `allocate_delta`, per-device
`constant_buffer_cache` that outlives contexts), and a packing pass is just an
OCANNL-compiled routine run once at link time — the compiler-generates-C analog
of ggml's `set_tensor` hook, no hand-written repack kernels needed. Composed
with F3, this is what turns the S4 packed matmul from "3.8× but loads dominate"
into a tinyBLAS-shaped kernel: packed weights resident in the constant pool,
register-tiled consumption.

### Mapping to OCANNL surface

`arrayjit/lib/schedule.ml` (`Stage` gains an out-of-routine placement for
operands whose tnode is `known_constant`), `arrayjit/lib/backends.ml`
(`constant_buffer_cache` holds the packed copy; the packing routine runs at
link/first-use like `Host_inits` uploads), `arrayjit/lib/autotune.ml` (the
sketch should be allowed to propose hoisted packing so the measured comparison
is fair).

### Verdict: `file follow-up issue` (draft F4)

---

## 8. Verdict summary

The five techniques enumerated by the task, plus the extras found while reading:

| # | Technique | Verdict |
|---|---|---|
| 1 | Quantization (int4/int8/fp16 hardware kernels) | `future` (post-v1.1; #137 closed not-planned; fp16 storage tiers already covered in `builtins_cc.ml`) |
| 2 | SIMD intrinsics (AVX2/NEON) | `already covered by gh-ocannl-164` (landed bundle + vector-extension codegen); remnants → issues F2, F3 |
| 3 | Memory-mapped models | `file follow-up issue` (F1) |
| 4 | Work-stealing CPU thread pool | `already covered` (pool-backed `Grid` rendering via dispatch/OpenMP, landed) |
| 5 | Block quantization (shared scales) | `future` (same gate as #1; this note is the design seed) |
| — | Register-tiled matmul micro-kernel (tinyBLAS) | `file follow-up issue` (F3) |
| — | SIMD reductions (multi-accumulator + horizontal reduce) | `file follow-up issue` (F2) |
| — | Load-time weight repacking | `file follow-up issue` (F4) |
| — | Multi-variant fat-binary dispatch; `arch-fallback` renaming; spin-barrier engineering; no-search graph executor | `not applicable` (JIT-on-host / delegated runtimes / inverse lesson) |
| — | Arena allocation, measure-then-replay planning; runtime op fusion; decode-vs-prefill schedule split | `already covered` (pool planner; virtual-node inlining; shape-keyed autotune) |
| — | Liveness-based buffer reuse; NUMA/affinity; on-the-fly activation quantization; imatrix | `future` / `not applicable` (named gates in §§4–5, §1) |

Tally: **already covered 6** (incl. the two headline items landed since the
proposal), **file follow-up issue 4** (F1–F4), **not applicable 5**, **future 5**.

---

## 9. Prioritized shortlist: top 5 lessons for CPU GPT-2/Gemma inference

1. **mmap the weights (F1).** Cheapest win with the broadest effect: model load
   goes from two full copies to page-cache population; directly inside #377's
   end-to-end time budget; prerequisite for any larger-model demo. Requires only
   an aligned-payload revision of the checkpoint format plus a `Unix.map_file`
   load path.
2. **Register-tiled matmul micro-kernel via `Tile_mma` rendering (F3).** The
   profiled bottleneck (S4: loads dominate). ggml's numbers give the target
   shape: C-tile resident in registers across the whole k-loop, RM×RN sized to
   the register file, edges peeled — expressible in the already-landed vector
   extensions.
3. **SIMD reductions with independent accumulator chains (F2).** Decode is
   GEMV/dot-shaped; without vector reductions the n=1 path — the path a chat
   demo actually sits in — stays scalar. Also unblocks softmax/layer-norm
   reductions in the same rendering.
4. **Pack constant weights once at load (F4).** Hoist `Stage` packing of
   `known_constant` operands out of the per-token routine into the per-device
   constant pool; ggml pre-pays even the nibble-bias XOR at repack time — the
   "move every possible instruction from per-token to per-load" discipline.
5. **Keep quantization future, but design formats not just dtypes (§1).** When a
   ≥Gemma-scale-on-small-RAM demand arrives, the ggml lesson is that the win
   comes from the *block layout contract* (32-element blocks = one register,
   scales factored out of the integer loop, richer transient activation format)
   — not from adding an int4 dtype. Record this so #137's eventual successor
   starts from layouts.

---

## 10. Draft GitHub comment for issue #163

> Deep dive delivered: `docs/research/ggml-lessons.md` (ggml read at
> llama.cpp `c198af4d`, 2026-07-07; OCANNL surfaces verified at `605d565a`).
> Verdict summary for the five techniques in the task:
>
> - **SIMD intrinsics** — already covered: the #164 CPU bundle landed
>   (32-byte-aligned pools, `restrict`, pragmas, `-mavx2 -mfma` probing,
>   `OCANNL_HAS_AVX2`/`NEON`), plus explicit vector-extension codegen for
>   `Vectorized` loops (`cc_vector_bytes`). Two remnants become issues:
>   SIMD *reductions* (multi-accumulator + horizontal reduce — ggml's
>   `GGML_F32_ARR` pattern) and a register-tiled matmul micro-kernel
>   (tinyBLAS's 4×6 C-tile-in-registers, via `Tensorize`/`Tile_mma` rendering).
> - **CPU thread pool** — already covered since the schedule layer: eligible
>   `Grid` loops render over `dispatch_apply`/OpenMP process-global pools
>   (`cc_parallel_grid`, `Schedule.default_cpu`). ggml's spin-barrier/NUMA
>   engineering is delegated-runtime territory; not applicable.
> - **Memory-mapped models** — confirmed gap: `Persistence.load` copies twice
>   (`really_input` + host→context upload). Filing an issue: aligned checkpoint
>   payloads (GGUF pads tensor offsets to 32) + `Unix.map_file`-backed hosted
>   arrays, with optional zero-copy into the CPU constant pools later.
> - **Quantization / block quantization** — `future` per #137's not-planned
>   closure; the note records ggml's format design (Q4_0/Q4_K layouts, fp16
>   shared scales, int8-domain dot kernels, on-the-fly Q8 activations) as the
>   design seed for a post-v1.1 revisit. No issue filed.
> - **Bonus lesson** (issue filed): ggml repacks weights into ISA-shaped
>   interleaved tiles *once at model load* (`CPU_REPACK` buffer type). OCANNL's
>   `Stage` packs per kernel call — hoisting packing of `known_constant`
>   operands into the per-device constant pool is the compiler-native analog,
>   and composes with the matmul micro-kernel work.
>
> Priority order for the GPT-2 (#377)/Gemma CPU drivers: mmap → register-tiled
> matmul → SIMD reductions → pack-once weights. Follow-up issues: F1–F4 in the
> note's §11.

---

## 11. Draft follow-up issues (to file against `ahrefs/ocannl`, label `enhancement`)

**F1 — "mmap-backed checkpoint loading: aligned payloads + zero-copy hosted
arrays"** (milestone v0.8/v0.9, pairs with #377).
`Persistence.load` currently reads every payload with `Stdlib.really_input`
(`Ndarray.read_payload_from_channel`) and then copies host→context at link time.
Revise the checkpoint format to pad payload offsets to a declared alignment (32
bytes for SIMD, page size for mapping — cf. GGUF's `general.alignment`, default
32) and add a load path that wraps `Unix.map_file` regions as the hosted
Bigarrays. Step 2 (separate PR): let CPU backends' constant pools
(`Backends.allocate_delta` / `constant_buffer_cache`) reference the mapping
directly for read-only nodes, eliminating the second copy — ggml's
`buffer_from_host_ptr` capability, which on unified-memory Metal can extend to
GPU.

**F2 — "SIMD reductions in the Vectorized renderer: independent accumulator
chains + horizontal reduce"** (milestone v0.8; successor to #164's remnants).
`c_syntax.ml` `try_vectorize` covers elementwise `Add/Sub/Mul/Div/Neg/FMA` only;
reduction loops fall back to pragmas, which cannot reassociate strict FP — the
permission the `Vectorized` retype exists to carry. Emit ggml's `ggml_vec_dot_f32`
pattern in vector extensions: N independent accumulator vectors (N from `Unroll`
or a renderer default of 4), fused k-loop, lane-sum tree at exit, scalar tail.
This is the n=1 decode path (GEMV, softmax, layer-norm) for the #377 demo.

**F3 — "Register-tiled CPU matmul micro-kernel: vectorized `Tile_mma`
rendering"** (milestone v0.8/v0.9; composes with the autotune matmul sketch).
`Tensorize`→`Tile_mma` names the micro-tile and `Autotune.tune`'s
`sketch_candidates` seeds tile sizes, but cc renders the tile body as scalar
loops. Render `Tile_mma` with the C-tile held in vector-extension variables
across the entire k-loop, tinyBLAS-style (RM×RN + RM + RN vector registers
budgeted to the ISA's register file; 4×6 on NEON/AVX-512-class, 4×3 on AVX2;
edge tiles peeled, not masked). Target: close the gap from S4's 9 GFLOP/s
single-threaded toward compiler-BLAS territory on the GPT-2 decode/prefill
matmuls.

**F4 — "Pack constant operands once at load: hoist `Stage` packing out of the
routine for `known_constant` tnodes"** (milestone v0.9).
`Stage` packs operand tiles into scratch per kernel invocation; for inference
weights this re-packs identical data every token. When the staged operand is
compile-time constant (`Tnode.Placements.known_constant`, the read-only
partition in `allocate_delta`), materialize the packed layout once into the
per-device constant pool via a link-time packing routine (the compiler-native
analog of ggml's `CPU_REPACK` `set_tensor` hook, which even pre-pays the nibble
bias XOR at repack time). The autotuner should be able to propose hoisted vs
in-kernel packing so the choice stays measured.

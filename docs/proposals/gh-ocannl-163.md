# Proposal: Research note — efficiency lessons from ggml

Task: gh-ocannl-163
Issue: https://github.com/ahrefs/ocannl/issues/163

## Status update (2026-06-12)

- Issue #163 is still OPEN, milestone v0.8 (ROADMAP targets mid-June 2026 for v0.8).
- Not yet started: `docs/research/` does not exist and no ggml-lessons note has been written.
- **Major invalidation: gh-ocannl-137 (quantization) was CLOSED as NOT_PLANNED.** The verdict option `already covered by gh-ocannl-137` is no longer available; quantization-related techniques (including block quantization with shared scales) must get verdict `file follow-up issue`, `not applicable`, or `future` — and the note should record 137's closure rather than cite it as a live follow-up.
- gh-ocannl-164 (AVX/AVX2) is still OPEN, milestone v0.8, with its proposal at `docs/proposals/gh-ocannl-164.md` unimplemented — the `already covered by gh-ocannl-164` verdict remains available.
- Cited OCANNL surfaces re-verified at HEAD `d9de22f0`: `ops.ml`, `c_syntax.ml`, `cc_backend.ml`, `builtins_cc.ml`, `tnode.ml`, `backend_impl.ml`, `schedulers.ml` all exist; `backend_impl.ml` still uses unaligned `Ctypes.allocate_n`; `schedulers.ml` still runs one OCaml `Domain` worker per stream with no intra-kernel parallelism.
- Backend-layer changes since April 2026 worth reflecting in the note: `device_to_device` now returns a transfer routine with static merge-buffer verification (`merge_buffer_use = No | Copy`; `Streaming_for` is gone), Metal uses private storage mode for GPU-only buffers (`1cf9a95b`), and commit `272c0880` removed the deprecated multi-stream backend infrastructure (cross-stream automatic coherence) while multiple streams per device remain.

## Goal

Produce a focused research note that distills what OCANNL can learn from
[ggml](https://github.com/ggerganov/ggml) for CPU efficiency, and converts
the findings into actionable follow-up work. ggml itself is not a viable
backend candidate for OCANNL (the user judged it "not flexible enough" in
2024-09), so the deliverable is *lessons*, not adoption.

The note bounds an otherwise open-ended exploration: a fixed list of ~5
ggml techniques, each examined through the same template, each closing
with a single explicit verdict. Where ggml-inspired work is already tracked
by `gh-ocannl-137` (quantization) or `gh-ocannl-164` (AVX/AVX2 intrinsics),
the note records the mapping rather than duplicating scope. *(Update
2026-06-12: #137 has since been closed as not-planned, so only the #164
mapping remains live; see Status update.)* Where ggml
techniques are not yet tracked (the obvious candidates at draft time are
memory-mapped weights, an intra-kernel thread pool, and block-quantization
scale grouping), the worker files new GitHub issues against `ahrefs/ocannl`.

## Acceptance Criteria

- A research note exists at `docs/research/ggml-lessons.md` (the
  `docs/research/` directory is created as part of this task; see Scope).
- The note covers the five techniques enumerated in the task elaboration,
  each in its own section: quantization (int4/int8/fp16 with hardware-
  specific kernels), SIMD intrinsics (AVX/AVX2/AVX-512/NEON for quantized
  matmul), memory-mapped models, work-stealing CPU thread pool, and block
  quantization (groups sharing scale factors).
- Each technique section follows a fixed template with these subheadings,
  in this order:
  1. **What ggml does** — 2-5 sentences describing the technique as
     implemented in ggml. No source-code archaeology; cite ggml docs or
     the well-known shape of the technique.
  2. **Relevance to OCANNL** — why this matters (or doesn't) for OCANNL's
     intended workloads (transformer inference, LLM serving, training).
  3. **Mapping to OCANNL surface** — concrete OCaml file/function/type
     names where the change would land. Use names from
     `arrayjit/lib/ops.ml`, `arrayjit/lib/c_syntax.ml`,
     `arrayjit/lib/cc_backend.ml`, `arrayjit/lib/builtins_cc.ml`,
     `arrayjit/lib/tnode.ml`, `arrayjit/lib/backend_impl.ml`,
     `arrayjit/lib/schedulers.ml`. Line numbers are forbidden (they drift);
     use symbol names and short distinctive code quotes when needed.
  4. **Verdict** — exactly one of:
     - `already covered by gh-ocannl-137` (quantization) *(Update
       2026-06-12: no longer available — #137 closed as not-planned)*
     - `already covered by gh-ocannl-164` (AVX/AVX2 intrinsics)
     - `file follow-up issue` (with proposed issue title and 1-line scope)
     - `not applicable` (with reason)
     - `future` (worth revisiting after a named milestone, e.g., post-v0.8)
- For every section whose verdict is `file follow-up issue`, an issue is
  filed against `ahrefs/ocannl` with `gh issue create`, labeled
  `enhancement` and milestoned `v0.8` or later as appropriate. The note
  cross-references the new issue numbers.
- For every section whose verdict references `gh-ocannl-137` or
  `gh-ocannl-164`, the cross-reference quotes the specific acceptance
  criterion or proposal section that subsumes the ggml technique. If no
  such criterion exists yet, the verdict is `file follow-up issue`
  instead, *not* a silent claim of coverage.
- The note's introduction includes one paragraph stating ggml is not a
  backend candidate (with a link to the user's 2024-09 comment) and that
  the note's purpose is lesson extraction.
- The note's conclusion contains a short table summarizing the verdict
  for each of the five techniques.
- All cited OCANNL surface symbols are real (i.e., `grep` finds them in
  the named files at the time of writing); the note records the commit
  SHA at which symbols were verified.

## Context

### Existing OCANNL surfaces relevant to ggml lessons

| Subject | File | Notes |
|---------|------|-------|
| Precision types | `arrayjit/lib/ops.ml` | `precision`/`prec` GADTs already include `Half`, `Bfloat16`, `Fp8`, `Uint4x32`, `Single`, `Double`. FP8 conversion builtins exist in `builtins_cc.ml`. No int4/int8 quantization-aware types yet. |
| CPU code generation | `arrayjit/lib/c_syntax.ml` | Emits C from Low_level IR. `pp_ll` walks the IR; `compile_proc` emits function bodies. This is where pragma hints, `restrict` qualifiers, and SIMD-friendly loop patterns land. |
| CPU compile/run | `arrayjit/lib/cc_backend.ml` | Invokes GCC/Clang on generated C. Compiler flag selection (`arch_flags`) is here. |
| C builtins/includes | `arrayjit/lib/builtins_cc.ml` | Where SIMD platform-detection macros (`__AVX2__`, `__ARM_NEON`) and intrinsic helpers belong. |
| Tensor node model | `arrayjit/lib/tnode.ml` | `memory_mode` / `Hosted` distinguish where tensor data lives; relevant for memory-mapped weight loading. |
| Allocation | `arrayjit/lib/backend_impl.ml` | Currently uses `Ctypes.allocate_n int8_t` (no alignment guarantee). The 164 proposal already plans to switch to aligned allocation; mmap-backed allocation would integrate here. |
| CPU scheduling | `arrayjit/lib/schedulers.ml` | Per-stream OCaml `Domain` worker. Commit `272c0880` removed the *deprecated* multi-stream backend infrastructure (cross-stream automatic coherence); multiple streams per device remain. Intra-kernel work-stealing is a *separate* concept (parallelizing the body of a single kernel across cores) and is not currently present. |

Note: there is no `arrayjit/lib/cpu_backend.ml`; the CPU code path is
`c_syntax.ml` (emit C) + `cc_backend.ml` (compile and run via system C
compiler). The task elaboration mentions `cpu_backend.ml`; the worker
should correct that pointer in the research note.

### Existing scope of related tasks

**`gh-ocannl-137` (quantization, milestone v1.1, effort large).** *(Update
2026-06-12: CLOSED as not-planned; the description below is kept as
historical context, but the note must not claim coverage by this issue.)*
Acceptance criteria as of writing: tensors quantizable to int8/fp8/int4 with
scale/zero-point, mixed-precision computation, at least one quantization
scheme implemented, quantized inference within tolerance. The tentative
design notes that OCANNL already has `Half`, `Bfloat16`, `Fp8`, `Uint4x32`
precision types, and that what's missing for "practical quantization" is
QAT, PTQ with calibration, mixed-precision inference, GGUF/GPTQ weight
loading, and quantized dynamic indexing. **No proposal file yet** — only
the task with tentative design.

**`gh-ocannl-164` (AVX/AVX2 intrinsics, milestone v0.8, effort medium).**
Has a written proposal (`docs/proposals/gh-ocannl-164.md`). Phases:
aligned allocation (32-byte), `-mavx2 -mfma` flags with platform
detection, `OCANNL_HAS_AVX2`/`OCANNL_HAS_NEON` macros, `restrict`
qualifiers on pointers in `compile_proc`, `#pragma GCC ivdep` /
`#pragma clang loop vectorize(enable)` on innermost loops,
`__attribute__((aligned(32)))` on local arrays, benchmark validating
≥2x speedup. Explicitly *out of scope*: explicit intrinsic emission
(`_mm256_fmadd_ps`), tiling, multi-threading, AVX-512.

These two tasks together cover most of ggml's CPU efficiency story.
The research note should treat them as the load-bearing follow-ups and
focus its new-issue output on the gaps.

### Likely gaps (worker should confirm in the note)

1. **Memory-mapped model weights.** No current OCANNL task tracks
   `mmap`-loading of large weight files. ggml's `mmap` story enables
   instant model load and OS-managed paging for models that exceed RAM.
   Mapping point: a new constructor in `tnode.ml`'s `memory_mode` (e.g.,
   `Hosted Mmapped`) plus a loader that returns a Bigarray over the
   mapped region.

2. **Intra-kernel work-stealing thread pool.** `schedulers.ml` is
   per-stream; it does not parallelize the body of a single kernel
   across cores. A pthread-style thread pool that splits a `For_loop`'s
   range across N workers is the ggml model. Mapping point: an
   extension to `c_syntax.ml` to emit OpenMP `#pragma omp parallel for`
   on outer loops (simplest path), or a hand-written pool in
   `builtins_cc.ml` (more flexible).

3. **Block quantization with shared scale factors.** Distinct from
   "quantization" in the abstract: ggml's Q4_0/Q4_1/Q8_0 formats group
   32 values and share one `fp16` scale (and optionally one offset) per
   group. This is an *encoding* decision orthogonal to QAT/PTQ. It
   plausibly fell under `gh-ocannl-137`, but that issue is now closed
   as not-planned *(Update 2026-06-12)*, so if this technique is judged
   worthwhile it needs a fresh follow-up issue. Mapping point: a layout
   descriptor in `ops.ml` or `tnode.ml`, plus dequantize/matmul kernels
   in `builtins_cc.ml`.

The worker should validate these gap claims against the latest state of
137/164 before writing the note, and adjust the verdict per technique
accordingly.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

1. **Verify code pointers.** Run `grep`/`rg` for each cited symbol in
   the file it claims to live in. Record the OCANNL commit SHA at the
   top of the research note.
2. **Re-read 137 and 164.** Confirm the gap analysis above is still
   accurate. If 137 has been refined to include block quantization, or
   164 has expanded to include threading, update the verdicts.
3. **Write `docs/research/ggml-lessons.md`** following the template in
   the Acceptance Criteria. Hard cap: ~600 lines. Each section ≤120
   lines. No code samples longer than ~10 lines.
4. **File follow-up issues.** For each `file follow-up issue` verdict,
   draft and submit via `gh issue create --repo ahrefs/ocannl --label
   enhancement`. Record the issue number in the note before committing.
5. **Commit.** Single commit titled
   `research: efficiency lessons from ggml (gh-ocannl-163)` containing
   the new directory, the note, and (if it doesn't already exist) an
   index entry in `docs/CLAUDE.md` or wherever the doc index lives.

## Scope

**In scope**

- The research note at `docs/research/ggml-lessons.md`.
- Creation of the `docs/research/` directory (no other research notes
  exist there yet — this task establishes the convention).
- One follow-up GitHub issue per `file follow-up issue` verdict,
  expected to be 1-3 issues (mmap weights, thread pool, block-quantization
  layout).
- A short conclusion table.

**Out of scope**

- Any code change in `arrayjit/lib/` or elsewhere outside `docs/`.
- Deep ggml source-code archaeology (citing function names, line
  numbers, or commit SHAs from the ggml repo). High-level descriptions
  of techniques are sufficient.
- Performance benchmarking or prototype ports.
- Discussing techniques outside the five enumerated above (e.g., ggml's
  graph allocator, custom file format details, RPC backend). If the
  worker finds an obvious sixth technique while exploring, they may
  note it in a single "other observations" appendix — but it does not
  get a full template section and does not gate acceptance.
- Modifying the scope of `gh-ocannl-137` or `gh-ocannl-164`. The note
  *maps* ggml techniques onto those tasks; if the mapping reveals a
  scope gap, that becomes a new issue, not an edit to the existing
  tasks.

**Dependencies**

- None hard. The note assumes 137 and 164 exist with their current
  acceptance criteria. If either is closed/superseded before this task
  runs, the verdicts must reflect that.

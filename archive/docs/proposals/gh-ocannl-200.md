# LLVM / clang Backend — Disposition Memo

## Goal

Address [gh-ocannl-200](https://github.com/ahrefs/ocannl/issues/200) —
"Add an LLVM / clang backend" — by producing a written disposition document,
not by building an LLVM backend.

The issue is from 2024-01-10 (with a 2024-07-15 follow-up about NVPTX
targeting via GASS). It is labelled `enhancement`, milestoned `v1.1`
(post-v1.0), task effort `large` (10–15 days). The user has paused
autonomous OCANNL work pending a hands-on audit, so this proposal is
artifact-only.

The load-bearing question is *not* "how should an LLVM backend be designed?"
but rather *"after two years of churn in OCANNL's backend landscape, does
LLVM still offer enough marginal value over the existing C-via-system-clang
+ NVRTC + Metal triple to justify a large milestone item?"* The right
output is a disposition document that surfaces the choice as a user
question, in the same spirit as the gh-ocannl-161 fork-backend memo.

This proposal recommends a disposition and surfaces the choice rather than
picking unilaterally.

## Acceptance Criteria

- [ ] The document records the actual current shape of the OCANNL backend
      stack with concrete pointers (cc_backend.ml uses *system clang/gcc
      via `Sys.command`*, not libgccjit; CUDA backend uses NVRTC at the
      CUDA-C source level; Metal backend uses MSL) so a future reader does
      not have to re-do the verification.
- [ ] The document records the current opam status of the `llvm` OCaml
      bindings as of 2026-05.
- [ ] The document enumerates the disposition options for gh-ocannl-200
      and recommends one, with the choice surfaced explicitly to the user
      as a question rather than picked silently.
- [ ] The document, if disposition (C) is chosen, contains enough to
      justify closing or restructuring the issue without further design
      work.
- [ ] The document, if disposition (A) or (B) is chosen, contains a
      *minimal* sketch of the deliverable shape (which files, which IR
      mapping, what the gate criterion is) — not a full design.
- [ ] The document is committed to the OCANNL repo so it survives outside
      the harness; the GH issue can be updated to link to it.

Out of scope (explicitly):

- Implementing any LLVM backend, IR mapping, or LLJIT plumbing.
- Choosing between opam-llvm and a custom ctypes FFI (premature without
  disposition).
- Benchmarking auto-vec wins (this would be the *gate* of disposition (A),
  not part of this memo).
- Re-litigating gh-ocannl-164 (AVX intrinsics) on its own merits — only
  the overlap with this issue is in scope.

## Context

### Verification of the brief's premises (in code)

The orchestrator brief restated several premises from the original task
elaboration. Two of them turn out to be wrong, which materially changes
the disposition.

**Premise 1 (false): "CC backend uses libgccjit (already a JIT path)."**

`arrayjit/lib/cc_backend.ml` does **not** use libgccjit at all.
`c_compile_and_load` writes the generated C source to a temp file, then:

```
Printf.sprintf "%s %s %s -o %s %s > %s 2>&1"
  (compiler_command ()) f_path compiler_flags libname kernel_link_flags temp_log
```

— shells out to the compiler reported by `ocamlc -config | c_compiler:`,
and `dlopen`s the resulting `.so` / `.dll` / `.bundle`. The default flags
are `-O3 -march=native` (with optional `-ffast-math`). On macOS that
compiler is **clang**; on Linux it is typically gcc but may be clang.
There is no in-process JIT — the grand "third JIT path" framing in the
brief collapses: OCANNL has *one* JIT-shaped path (NVRTC, in-process
PTX-from-CUDA-C), and its CPU path is just AOT compile + dlopen.

This matters because the strongest a-priori argument for LLVM
("auto-vec wins libgccjit lacks") is now moot: clang's auto-vec is
**already** what the CC backend gets at `-O3 -march=native`. The marginal
value of LLVM-IR + LLJIT over "clang + .so + dlopen" is:
in-process JIT latency (no fork/exec, no fs round-trip), control over the
optimization pipeline, and codegen at a lower level than C source. None
of those are obviously load-bearing for OCANNL's actual workloads.

**Premise 2 (true): "CUDA backend invokes NVRTC at the C-syntax level."**

`cuda_backend.ml` passes a CUDA-C source string to
`Nvrtc.compile_to_ptx` (line ~178). LLVM's NVPTX target would substitute
at the LLVM-IR-to-PTX level, replacing both NVRTC and the CUDA-C step.
This *is* a real architectural alternative — but it competes with NVRTC,
which is already a working in-process path, not with the absence of one.
And `cuda_backend.ml` runs `grid_dim=1, block_dim=1` (the
`kernel_prep_line` early-return in line ~328); the first GPU optimization
work is cracking that single-threaded baseline, not migrating away from
NVRTC.

**Premise 3 (true): "OCaml `llvm` bindings are on opam."**

`opam show llvm` reports versions up through **`19-static` and
`19-shared`** (LLVM 19, current as of 2026), maintained by Alan
(`ahulambda@gmail.com`), depending on `conf-llvm-static` build wrapper.
The bindings have *not* bit-rotted. So the worst-case framing
("workload becomes maintain LLVM bindings on top of add a backend") does
not apply. Good news for disposition (A) or (B).

There is no mention of LLVM in `arrayjit/lib/dune` (no commented-out
hint), `CHANGES.md`, or `ROADMAP.md` post-v1.0. The closest analogue in
the v1.1 list is `XLA backend (#300)` under "Performance explorations" —
which is itself a higher-level codegen alternative.

### What gh-ocannl-200 actually competes with

The 2024-01-10 issue was filed when libgccjit-without-AVX2 was a real
constraint and a separate JIT path mattered. Two years on, that constraint
no longer exists in the code. The actual relevant alternatives today are:

| Concern                    | Current path                          | LLVM alternative                         | Marginal value of LLVM |
|----------------------------|---------------------------------------|------------------------------------------|------------------------|
| CPU codegen + auto-vec     | clang/gcc `-O3 -march=native` + .so + dlopen | LLVM-IR + LLJIT in-process            | low (clang is LLVM)     |
| CPU SIMD intrinsics        | gh-ocannl-164 (proposed; auto-vec + alignment) | LLVM intrinsics directly             | low (AVX2 already auto-vec'd) |
| GPU codegen (CUDA)         | CUDA-C → NVRTC → PTX                  | LLVM-IR → NVPTX → PTX                    | medium (low-level control, but kernel_prep_line is the actual blocker) |
| GPU portability (non-NV)   | (nothing today; Metal is separate)    | LLVM SPIR-V / AMDGPU                     | medium-to-high (no current path) |
| Higher-level optimizations | XLA backend (#300), megakernels (#318)| (overlaps with XLA via MLIR)             | low (wrong layer)       |

The cells where LLVM offers real marginal value (NVPTX unification,
non-NVIDIA GPU portability) are not what gh-ocannl-200's title or 2024
body asked for; they are sub-features that, if pursued, deserve their own
narrowly-scoped issues.

### Sibling tasks

- **gh-ocannl-164 (AVX/AVX2 intrinsics, milestone v0.8)** — has its own
  proposal at `docs/proposals/gh-ocannl-164.md`, which already chose
  *"Option A: compiler auto-vectorization via flags + alignment + restrict +
  pragmas"*, supplemented later by explicit intrinsics in the tiling
  micro-kernel (gh-ocannl-412). That proposal is exactly the
  "do this with the existing clang we already shell out to" path. **It is
  not subsumed by an LLVM backend** — it is the *direct alternative* to
  one. If gh-ocannl-164 lands and delivers measurable wins, gh-ocannl-200's
  CPU rationale evaporates. There is no reason to merge the two.

- **watch-ocannl-README-md-347818d3 (CPU tiling, v0.8)** is explicitly a
  C-codegen task: it generates AVX2/NEON intrinsics into the CC backend's
  C source. LLVM is not in scope.

- **watch-ocannl-README-md-d5eb2b05 (CUDA tiling, v0.8)** is explicitly
  CUDA-C-to-NVRTC; the milestone-blocker is the single-threaded
  `kernel_prep_line` guard, not the codegen layer.

- **gh-ocannl-300 (XLA backend, v1.1+)** is the closest sibling in the
  "alternative codegen layer" space and overlaps philosophically. Worth
  flagging but not in this memo's disposition.

### Disposition options

**(A) Promote and narrow.** Re-priority to v0.8, scope down to LLVM-IR +
LLJIT for CPU only (skip NVPTX), and gate on a measured auto-vec /
codegen-quality win against `clang -O3 -march=native` on representative
OCANNL kernels (matmul, elementwise, reduction). If the win is < ~10%,
fall back to gh-ocannl-164 + tiling. Deliverable shape: a new
`arrayjit/lib/llvm_backend.ml`, mirroring `cc_backend.ml`'s `compile`
+ `compile_batch` interface, lowering `Low_level.optimized` directly to
LLVM IR (skipping the C-syntax intermediate); plus a new dune library
parallel to `cuda_backend` / `metal_backend` (optional, gated on the
`llvm` opam package). **Risk:** the gate is unlikely to show a real
win — clang *is* LLVM, and the difference between "clang from .c on disk"
and "LLJIT from in-memory IR" is mostly latency, not codegen quality.
Likely-to-fail-the-gate, which is itself useful information.

**(B) Defer with design memo.** Keep gh-ocannl-200 open at v1.1, and
attach this memo as the design artifact. Implementation queued post-v0.8
once tiling + megakernel work has clarified what's CPU-bound and where
NVRTC's limits actually bite. Useful if the user wants to keep the
option alive but does not want to commit before v0.8 lands.

**(C) Close as not-planned.** clang already provides LLVM-quality CPU
codegen via the `Sys.command` path; NVRTC already provides in-process
GPU JIT; gh-ocannl-164 covers the explicit-SIMD case. The marginal value
of an LLVM-IR backend is small enough at the OCANNL design's *current*
shape that v1.1 is the wrong place. If a specific narrow LLVM-only
feature (e.g., NVPTX unification, AMDGPU support, MLIR / IREE bridging)
becomes load-bearing, refile a narrowly-scoped issue then. **This is the
recommended disposition.** It is consistent with how gh-ocannl-341
(multi-streaming) was retired — by recognising that the architectural
context that motivated the issue had moved on.

**(D) Sibling-merge with gh-ocannl-164.** *Rejected.* gh-ocannl-164 is
already proposed as auto-vec-first with intrinsics in the tiling
micro-kernel; merging in "build an LLVM backend instead" would expand
scope rather than focus it. The two issues address overlapping *goals*
(faster CPU kernels) but at incompatible *layers*; they should not share
a milestone item.

### Recommended disposition: (C)

The premise that motivated the original issue ("libgccjit lacks AVX2,
LLVM bindings just landed on opam") has been overtaken by events: the CC
backend never used libgccjit, and clang's own auto-vec already runs.
gh-ocannl-164 covers the SIMD-intrinsics case at the C layer. NVRTC
covers the GPU JIT case. v1.1 has crowded competition (#263 Lean
Attention, #271 quantization, #300 XLA, #297 BERT, #275 LLM101n, #278
DisTrO) all with clearer motivation.

If the user disagrees and wants to keep the option open, (B) is a
zero-cost fallback — the document already exists.

If the user is curious whether LLJIT-in-process actually beats
clang+dlopen (a legitimate empirical question), (A) is the way, with the
gate being the win-vs-clang measurement, not the existence of an LLVM
backend.

## Question for the user

Which disposition for gh-ocannl-200?

- **(A)** Promote + narrow + measure (CPU LLJIT vs clang gate).
- **(B)** Defer with this memo attached; revisit post-v0.8.
- **(C)** Close as not-planned; refile narrower issues if NVPTX or
  non-NVIDIA-GPU portability ever becomes load-bearing. *(recommended)*

A fourth implicit option is "rewrite the issue body to scope to NVPTX
unification only" — call it (C′) — which would close-and-refile in one
step; mentionable if the user wants the NVPTX angle preserved without
the CPU-LLVM framing.

## Scope

- **In scope:** disposition memo committed to OCANNL repo.
- **Out of scope:** any code changes; the GH issue update (link this memo,
  optionally close) is a follow-up Mag action after the user picks a
  disposition.
- **Dependencies:** none. This memo can land independently of v0.8 work.
- **Related:** gh-ocannl-164, gh-ocannl-300 (XLA), gh-ocannl-161
  (precedent — disposition memo format), gh-ocannl-278 (precedent —
  feasibility study).

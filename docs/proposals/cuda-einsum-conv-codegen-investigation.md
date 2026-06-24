# Proposal: Investigate + fix CUDA codegen wrong results on virtual-diagonal / einsum3 / conv-padding

Task: task-04f97340 (OCANNL)

## Goal

Identify and fix the CUDA-only codegen defect that makes three *deterministic*
computations — the virtual-diagonal projection, the 3-way einsum reduce, and the
padded convolution — produce structurally-wrong results on-device under
`OCANNL_BACKEND=cuda`, while the identical inputs pass on CPU (`sync_cc`) and Metal.
The deliverable is a **working fix**: `test_virtual_diagonal`, `test_einsum3`, and
`test_conv_padding` green under CUDA, matching the CPU/Metal reference outputs. The
root cause is not pre-identified — static analysis has narrowed it but cannot locate
it; it must be found by **on-device investigation on the sole nvidia host
(minipc-wsl)**, not by further static inspection. A diagnosis alone is not the
target; a report on falsified hypotheses is only the fallback if a fix proves
unreachable.

## Acceptance Criteria

Verified by running, on the nvidia host (minipc-wsl):

```sh
export PATH="/usr/lib/wsl/lib:$PATH"; eval $(opam env)
OCANNL_BACKEND=cuda dune runtest test/operations/test_virtual_diagonal.exe
OCANNL_BACKEND=cuda dune runtest test/operations/test_einsum3.exe
OCANNL_BACKEND=cuda dune runtest test/einsum/test_conv_padding.exe
```

- [ ] `test_virtual_diagonal` passes under `OCANNL_BACKEND=cuda`, producing the
      diagonal `[1;2;3;4;5]` matching the CPU/Metal reference output.
- [ ] `test_einsum3` passes under `OCANNL_BACKEND=cuda`, the reduce branch giving
      `[12 43]` (not `[20 40]`), matching the CPU/Metal reference.
- [ ] `test_conv_padding` passes under `OCANNL_BACKEND=cuda`, the padding-edge first
      row matching the CPU/Metal reference (`5.30 10.5 4.57 4.77`).
- [ ] The fix targets the **identified root cause**, found on-device. A single shared
      root cause is preferred; per-case fixes are acceptable only if the cases prove
      genuinely independent during bisection.
- [ ] **No regression**: the CC (`sync_cc`) and Metal backends remain correct on these
      and all currently-passing tests, and the currently-passing CUDA kernels
      (general matmul, training-to-threshold, plain reduction, backprop) still pass.
      Confirm with a regression sweep of the CUDA fast suite
      (`OCANNL_BACKEND=cuda dune runtest test/operations test/einsum`, plus a passing
      training/matmul kernel).
- [ ] **Fallback (not the target):** if, after exhausting the method below, no fix is
      reachable, the deliverable becomes a documented root-cause report containing the
      minimal on-device reproducer (the diagonal case), the `.cu`-vs-`.c` diff, and the
      explicitly falsified hypotheses with evidence. This is recorded only as a last
      resort, not delivered in lieu of a fix that is within reach.

## Context

Backported from a CUDA bug report that the bring-up task **task-bfc7c7b5** *filed
instead of fixing* once the CUDA backend was first made to build and run on real
hardware (RTX 3050 Ti, WSL CUDA). The in-repo source doc is
`docs/in-progress/cuda-einsum-conv-codegen-bug.md` (on `master`):

> A few specific einsum / projection / convolution patterns compute **wrong values
> on CUDA** with deterministic inputs, while passing under `sync_cc`. […] Since
> `sync_cc` is correct and CUDA is wrong on identical deterministic inputs, the
> divergence is CUDA-specific index/loop emission for these patterns, not the
> high-level shape inference.

This is **not** the pool-allocator addressing issue that task-bfc7c7b5 already fixed
(`ebca9212`, `aed1e3a6`): the inputs are deterministic (no PRNG), general
matmul/reduction/backprop pass on CUDA, and `cuda_pool_offset` round-trips correctly.

**What elaboration established (static analysis — narrows, does not locate):**

- The offset/index/loop/guard/padding/zeroing emission is **shared and textually
  identical** across CUDA / CC / Metal: row-major flat offset (`c_syntax.ml`
  `pp_array_offset`), axis-index emission (`indexing.ml` `pp_axis_index`), the
  inclusive `for` loop, the `reset_padding_regions` neutral-fill IR
  (`assignments.ml`), and `Zero_out` (`c_syntax.ml`). Convolution's negative padding
  offset is converted to non-negative buffer space *before* emission
  (`assignments.ml apply_padding_offset`), so the naive "unsigned wraps `i-1`" theory
  is **ruled out**.
- The CUDA-only emission overrides (loop-index type, precision conversion, binop
  syntax, the single-thread kernel guard) were audited and **none apply to the
  single-precision-float path** these tests exercise — `Single→Single` conversion is a
  no-op and `Add/Mul` emit identical `+`/`*`. Buffers are allocated **un-zeroed on
  every backend**, so correctness relies on the *shared emitted* `Zero_out` /
  `reset_padding` IR, not on alloc-time zeroing — same for CUDA and CC. So "CUDA
  leaves buffers uninitialized while CPU doesn't" is **not** a static mechanism.
- The three failures are most likely **one shared root cause** surfacing only under
  nvcc: the signatures are *structurally shifted, not random* (diagonal reads a
  neighboring cell, the reduce sums the wrong selected elements, the conv's
  padding-edge row is wrong). All three exercise the **guarded / virtual / padded
  lowering** that ordinary CUDA-passing kernels (matmul, training, plain reduction)
  skip. Whether it is one cause or several cannot be decided from static reading;
  **on-device bisection is required.**

## User's resolved direction

- **Deliverable = things working again** (the three tests green under CUDA). A report
  on falsified hypotheses is the explicit **fallback**, only if a fix can't be reached
  — don't stop at a diagnosis.
- **Q1 (one cause or several?)** → Find and fix the **single shared root cause**; fall
  back to per-case fixes only if they prove genuinely independent. Do **not** require
  all three confirmed-independent before fixing.
- **Q2 (where / what?)** → Subordinate to "make it work": fix wherever the true root
  cause lives (shared IR **or** CUDA emitter); **no** byte-identical-codegen
  constraint imposed up front.
- **User's lead hypothesis to test FIRST (flagged a "wild guess"):** memory/buffer
  **initialization happening on CPU and Metal but not effectively on CUDA**. *"If it's
  not that, keep debugging."*

## Approach (a method, not a known patch)

The root cause is unidentified by design; this is an investigate-then-fix effort.
Ordered method, on minipc-wsl:

1. **Reproduce** all three failures on-device and capture exact wrong-vs-expected
   values, confirming the static narrowing holds on the real device (CC/Metal correct,
   CUDA wrong on identical deterministic inputs).

2. **First probe — the user's init-effectiveness hypothesis, reframed.** The
   elaboration statically ruled out *naive* init-divergence (buffers are un-zeroed on
   **every** backend; correctness depends on the **shared emitted** `Zero_out` /
   `reset_padding` IR, not on alloc-time zeroing). **But static "the IR is emitted" ≠
   runtime "nvcc actually ran it."** So the concrete first on-device probe is: verify
   whether that shared `Zero_out` / `reset_padding_regions` IR **actually executes on
   CUDA** for these specific virtual / guarded / padded nodes — i.e. is it
   emitted-but-nvcc-elided, skipped for virtual producers, or running against an
   off-by-one / wrong-extent region — versus the CC/Metal builds where it evidently
   takes effect. If real, the gap is **"emitted-but-not-effective on CUDA,"** not
   "never emitted," and the fix lands there. Probe by inspecting the generated `.cu`
   for the zeroing/padding loops, and instrument the buffer pre-state on-device (e.g.
   read back the virtual/padded buffer before the producing loop) to see whether the
   neutral fill took effect.

3. **Minimal-reproducer bisection on the diagonal case** (if the first probe comes up
   empty). Reduce `test_virtual_diagonal` to the smallest failing kernel, then **diff
   the generated `.cu` against the CC `.c`** for that case and continue bisecting the
   Tentative Design's ranked hypotheses — guard-index (the `Where(Cmpeq(idx,idx),…)`
   equality guard for repeated symbols), accumulator/`Local_scope` init, the
   `Embed_index` index→float cast, and kernel argument / buffer binding — looking for
   where nvcc's integer-promotion, evaluation-order, `extern "C"` arg-binding, or
   CUDA-only UB diverges from gcc/clang on the *shared* source.

4. **Fix at the true location, then test the single-cause hypothesis.** Once the root
   cause is found, fix it where it actually lives (shared IR or CUDA emitter), then
   check whether the same fix also resolves `test_einsum3` and `test_conv_padding`
   (the diagonal and conv share the projection/guard machinery; the reduce shares the
   accumulator/`Local_scope` machinery). If it does, ship the single shared fix; if a
   case proves genuinely independent, fix it on its own.

5. **Re-run + regression sweep.** Re-run all three target tests under CUDA, then sweep
   the currently-passing CUDA kernels (matmul, training-to-threshold, plain reduction,
   backprop) and confirm CC/Metal are unchanged.

Keep the change scoped to the root cause; avoid speculative cross-backend rewrites.

## Verification host note (operational constraints)

CUDA build + run is possible **only on minipc-wsl** (sole nvidia host). The build
requires `PATH="/usr/lib/wsl/lib:$PATH"` and `eval $(opam env)`; cudajit is
GitHub-pinned via `pin-depends` until a new release. Remote orchestration of this host
carries known caveats (lukstafi/ludics#579 controller-path leakage, #580 stale-local
auto-resume) — sync the worker harness before launching and quote remote paths. All
acceptance verification must be run on this device.

# Fix CUDA threefry/PRNG distribution: restore bit-spreading in CUDA `convert_precision`

**Task:** task-7d2ed931 · **Project:** OCANNL · **Priority:** A · **Effort:** medium

## Goal

On the CUDA backend, counter-based (threefry) random output is statistically broken: under
`OCANNL_BACKEND=cuda`, `test/operations/test_random_histograms.ml` fails its statistical assertions
(chi-square ≈ 475 vs critical ≈ 30, normal distribution with no tails, Kaiming mean ≈ 10× off),
while the same tests pass on the CPU/`sync_cc` backends. The cause is the CUDA backend's local
reimplementation of `convert_precision`, which drops the bit-spreading on the `uint → Uint4x32`
counter conversion: consecutive per-element threefry counters are emitted un-spread into `v[0]`,
so 2-round light threefry yields correlated outputs. The fix makes the CUDA `convert_precision`
mirror the shared `Ops.c_convert_precision` for all `Uint4x32` cases, so on-device randoms match
the reference distribution.

## Context

Filed as an in-repo bug doc — `docs/in-progress/cuda-prng-threefry-distribution-bug.md` (on
`master`) — by the CUDA bring-up task **task-bfc7c7b5**, which made the CUDA backend build/run on
real hardware (RTX 3050 Ti, WSL CUDA) but deferred the PRNG fix rather than addressing it (PR
[lukstafi/ocannl-staging#69](https://github.com/lukstafi/ocannl-staging/pull/69), commit
`791d9aef`). The bug is pre-existing: the CUDA backend never built before, so the PRNG path was
never exercised on-device. General compute, reductions, matmuls, and training all pass on CUDA —
this is isolated to random generation.

From the doc:

> The counter-based (threefry) PRNG produces a **statistically broken** distribution on the CUDA
> device. … Uniform: `Chi-square statistic: 475.26 (df=19, critical value at 0.05: ~30.14)` …
> Normal: `Within 2 std dev %% (expected: ~95.4 …): FAIL (got 100.0000)` — no tails …
> Kaiming: `Mean: 1.12 (expected: ~0.12)` — off by ~10×.
> The clustering/no-tails pattern suggests the per-element threefry counter is not varying
> correctly across output elements on CUDA (values come out correlated).

**Validated root cause** (confirmed against the current tree on branch `codex/no-grad-inits`;
relevant files identical to `origin/master`): the CUDA backend **reimplements** `convert_precision`
(`arrayjit/lib/cuda_backend.ml:820–848`) instead of using the shared `Ops.c_convert_precision`
(`arrayjit/lib/ops.ml:798–863`, the CPU/`sync_cc` path). Two conversion arms that the random
pipeline relies on are wrong:

1. **PRIMARY — missing `uint32/uint64 → Uint4x32` spread arm.** The per-element threefry counter is
   the loop offset, of precision `Ops.index_prec ()` = `uint32` (or `uint64` under `large_models`),
   coerced to the `Uint4x32` binop result precision (binops are homogeneous, `ops.ml:967`). The CUDA
   override has **no** `Uint32_prec → Uint4x32` (nor `Uint64_prec → Uint4x32`) case, so the counter
   falls through to the catch-all at `cuda_backend.ml:847`:
   `| _, Uint4x32_prec _ -> ("{(unsigned int)(", "), 0, 0, 0}")`. This lands the counter in `v[0]`
   only, with `v[1..3] = 0` and **no bit-spreading**. The shared path emits
   `uint32_to_uint4x32(offset)` (`ops.ml:861`), which XOR-spreads across all four words
   (golden-ratio / MMIX mixing). The builtins doc-comment warns this matters explicitly:
   *"Without this, consecutive counter values produce nearly identical v[0] outputs from 2-round
   threefry, causing periodicity"* (`builtins_cuda.ml:226–228`). Un-spread consecutive offsets fed
   into 2-round **light** threefry → correlated output → exactly the observed symptom. This single
   gap accounts for the whole statistical failure.

2. **SECONDARY — `Uint4x32 → _` raw extract.** `cuda_backend.ml:841`,
   `| Uint4x32_prec _, _ -> ("", ".v[0]")`, converts a `Uint4x32` value to a scalar precision by
   raw-extracting `.v[0]` instead of calling `uint4x32_to_<prec>(...)` (shared: `ops.ml:860`). The
   primary random→float path uses the explicit `Uint4x32_to_prec_uniform[_vec]` vec op
   (`c_syntax.ml:512`), which is correct, so this arm may not drive the histogram failure — but it
   diverges from CPU semantics and would corrupt any incidental `uint4x32 → prec` read, so it should
   be fixed for parity.

**Ruled out (do not touch):** the `builtins_cuda.ml` threefry core is correct and byte-equivalent to
the CPU reference — the `threefry_round` y↔w swap, the `THREEFRY_ROTATION` table, `THREEFRY_C240`,
the 2-round-light / 20-round-crypto schedules, key injection, `rotl32` via `__funnelshift_l`, the
`uint32_to_uint4x32` / `uint64_to_uint4x32` bit-spreading helpers themselves
(`builtins_cuda.ml:246–290`), and the `uint4x32_to_*_uniform` float mappings are all validated as
matching CPU. The shared op codegen (offset counter threading, `Range_over_offsets`, the
`Set_from_vec` store loop) is backend-agnostic, and the single-threaded CUDA kernel
(`grid_dim=1, block_dim=1`) runs the sequential `For_loop` so the offset varies correctly. The bug
is solely the `convert_precision` override dropping the spread.

## Acceptance Criteria

Verification is by running the tests below on the nvidia host (minipc-wsl), not by manual eyeballing.

- [ ] **Histogram statistics pass on CUDA.** `OCANNL_BACKEND=cuda dune runtest
      test/operations/test_random_histograms.exe` passes its statistical assertions — uniform
      chi-square below the critical value (≈ 30, vs the current ≈ 475), normal distribution tails
      present (≈ 95.4% within 2σ, within tolerance, vs current 100%), and Kaiming mean ≈ expected
      (≈ 0.12, vs current ≈ 1.12) — matching the CPU/`sync_cc` reference distribution.
- [ ] **Downstream PRNG casualties no longer fail for random-init reasons.** Under
      `OCANNL_BACKEND=cuda`, `test_uniform1`, `threefry4x32_demo`, `test_block_tensor`,
      `test_param_shape_error`, `moons_demo_variant`, and `transformer_names` no longer fail due to
      corrupted random initialization. Bit-exact tests either match the CPU `.expected` or receive a
      justified `.cu.expected` fixture once the distribution is correct; `transformer_names` again
      reaches `loss below threshold = true` at the reference epochs.
- [ ] **Both index-precision widths covered.** The fix produces correctly spread `Uint4x32`
      counters for both the default `uint32` index precision and the `uint64` width used under
      `large_models` / `Ops.index_prec () = uint64` — neither falls through to the no-spread struct
      literal.
- [ ] **CPU/`sync_cc` parity preserved.** The CUDA `convert_precision` for `Uint4x32` cases produces
      the same conversion calls as `Ops.c_convert_precision`; the existing CPU/`sync_cc` test suite
      (`dune runtest test/operations test/einsum`) remains green.

## Approach

The fix is localized to `convert_precision` in `arrayjit/lib/cuda_backend.ml:820–848`. The cleanest
framing is *stop overriding what the shared `Ops.c_convert_precision` already does correctly for
`Uint4x32`* — keep only the genuinely CUDA-native arms (`__double2half`, `__float2half`,
`__ushort2half_rn`, …) and delegate the rest to the shared conversion. Sketch:

- **Close the `… → Uint4x32` gap.** Add explicit `Uint32_prec → Uint4x32` and
  `Uint64_prec → Uint4x32` arms emitting `("uint32_to_uint4x32(", ")")` / `("uint64_to_uint4x32(",
  ")")` (the device helpers already exist at `builtins_cuda.ml:246–290`). Audit every remaining
  `… → Uint4x32` path so none silently falls through to the `cuda_backend.ml:847` no-spread
  `{(unsigned int)(…), 0, 0, 0}` catch-all — the spread arms must precede it. Note the CUDA override
  also lacks `Uint32_prec, Uint32_prec` / `Uint64_prec, Uint64_prec` identity arms present in the
  shared version, which is part of why the offset path falls through; mirroring the shared match
  closes this.
- **Fix the `Uint4x32 → _` arm.** Replace `cuda_backend.ml:841`'s `("", ".v[0]")` with the shared
  `("uint4x32_to_" ^ prec_string to_ ^ "(", ")")` form, matching `ops.ml:860`.
- **Prefer delegation over arm-by-arm patching where possible.** Consider routing all `Uint4x32`
  cases (and any other non-CUDA-native conversion) straight to `Ops.c_convert_precision ~from ~to_`,
  retaining the local match only for the CUDA-specific half-precision intrinsics. This minimizes
  future drift between the two implementations — the very drift that caused this bug. The reviewer
  should weigh full delegation vs. targeted arms against any CUDA-native conversions that must stay
  local.
- **No changes to `builtins_cuda.ml`, the threefry core, or the `uint4x32_to_*_uniform` mappings** —
  they are already correct.

## Verification host note

CUDA tests build and run **only on minipc-wsl** — the federation's sole nvidia host (RTX 3050 Ti,
WSL2 CUDA 12.8). Reproduction:

```sh
export PATH="/usr/lib/wsl/lib:$PATH"   # exposes nvidia-smi / libcuda
eval $(opam env)
OCANNL_BACKEND=cuda dune build test/operations/test_random_histograms.exe
(cd test/operations && OCANNL_BACKEND=cuda ../../_build/default/test/operations/test_random_histograms.exe)
# fuller fast suite:
OCANNL_BACKEND=cuda dune runtest test/operations test/einsum
```

Operational constraints (not part of the fix): remote-orchestration lifecycle on minipc-wsl carries
the known caveats from gh-ludics-579 / gh-ludics-580 (controller-path / stale-harness traps), and
the CUDA toolchain depends on the GitHub-pinned `cudajit` (`pin-depends`). Ensure the worker harness
is synced and the deploy is current before launching, and quote remote `-p '~/path'` arguments.

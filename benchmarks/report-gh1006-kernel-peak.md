# gh-ocannl-1006: the `kernel %peak` column on Metal, cc and CUDA

The exhibit behind the report's second gh-ocannl-1006 column — %-of-peak for each cell's dominant
kernel — showing it populated on Metal, cc and CUDA (no HIP cell was run) with its no-value cases
in place. **This is a rendering claim, not a measurement campaign**: the `mlp_wide` fixture here was generated for a
smoke run on mac-studio (`gen_fixtures.py --out-dir`, recorded nowhere, measured under
`--no-fixture-digest-check`), so each section names bytes no origin records and its numbers compare
with nothing outside this file. Each section below is `orchestrate.py`'s own `report.md`, verbatim,
ambient-environment header included; `--only ocannl`, so parity reads `NO-REF`.

What the cells show (CUDA: the rog-nv-linux section at the end):

- **Metal** (`metal class constant` ceilings, 5e12 FLOP/s and 1e12 B/s): both the default and the
  tuned cell print a number, `9.0%` and `9.9% f32 compute`. The tuned cell's dominant kernel is one
  of the kernels its searched winner shipped (26 kernels against the default pipeline's 15), which
  is what `Context.routine.segments` exists to reach. Its routine is `tensorized` in the `mma`
  column, but the kernel that dominates is not one of the tensorized ones, so it keeps the `f32`
  ceiling; a GPU tensor-core kernel would read `no ceiling`.
- **cc, no configuration**: `no ceiling`, the designed no-value case — the C backends carry no class
  constant, and a roofline with one leg missing would only lower-bound the attainment.
- **cc, with `model_peak_*` set** (second section; the ambient header shows the two variables): a
  number. The constants are the M4 Max's spec-sheet CPU peaks, not a calibration: 2.2e12 FLOP/s is
  12 P-cores x 4.5 GHz x 32 f32 FLOP/cycle (four 128-bit FMA pipes) plus the E-cores, rounded up;
  5.46e11 B/s is the SoC's memory bandwidth. Both are at least what the machine can sustain, which
  is what the `model_peak_*` contract asks, so the `0.1%` is not flattering: the default cc
  pipeline's dominant kernel (`k2/7`, 191 ms of a 238 ms step, one fused segment writing the
  weight gradients) runs 543 MFLOP at under 3 GFLOP/s.

## mac-studio (Apple M4 Max): Metal and cc

Run as `orchestrate.py --workloads mlp_wide --only ocannl --tuned --no-fixture-digest-check`.

### Benchmark results

platform: macOS-26.6.2-arm64-arm-64bit arm64 | ocannl commit: 339d3db72 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager; reduced precisions get their own envelope: bf16 0.004, f16 0.002; the approximate regime 0.01)

measurement boxes declared by `fixtures/DIGESTS.txt`: m4-max, minix, rog-nv

ambient OCANNL_* environment: none


#### mlp_wide

measured on `mlp_wide.safetensors`, sha256 `b5e848d31e60d4feeecf8f4d1ab655e8623be29bf34c1d604d3641097a9a9d0a`, bytes no origin records

**MISSING DIGEST RECORD:** declared measurement box(es) with no entry for `mlp_wide.safetensors`: m4-max

`pass` says which process produced a searching cell's step times. For the OCANNL tuned cell, whose protocol splits them: `replay` is the fresh pass-2 process replaying the cached winner, **`SEARCH PASS`** is the searching process itself — whose accumulated modules and buffers inflate every launch, so those numbers are not comparable with the others, and `no search` is a tuned cell that searched nothing and replayed nothing (autotune_search=false), so it shipped the untuned default. For a tinygrad `beam` or a `torch.compile` cell, which search in the timing process by protocol: `same-process` searched here, `cached` replayed its framework's own cache (so its `compile s` is a replay cost). A tuned cell's `compile s` carries the same statement about the search pass it came from: `(cached)` for one that replayed the schedule cache, `(no search)` for one that searched nothing at all.

`peak MiB` is the peak device footprint over the cell's TIMED STEPS (gh-ocannl-1006) -- bracketed there rather than read at process exit, so a tuned cell's schedule search, which allocates a candidate buffer per arm, is not in it. `—` is a cell whose framework exposes no device counter on this backend (a pytorch `cpu` row, say): not a zero, and no host-RSS figure is substituted for it. The counters are NOT one quantity, so every measured row names its own after the number: `ocannl-seam` = OCANNL allocator seam high-water (requested bytes, all backends). A high-water counter is exact over the window; one sampled at step boundaries is a lower bound on it, blind to an allocation made and given back within a step. Requested bytes off an allocator (OCANNL's seam, `torch.cuda.max_memory_allocated`) are the same quantity as each other and are NOT the device-wide figure a driver reports -- so rank rows within a counter, and read across counters only as the orders of magnitude they are.

`kernel %peak` is the attainment of the cell's DOMINANT KERNEL (gh-ocannl-1006): of the kernels the cell shipped -- a tuned cell's searched winner included -- the one that takes longest when each is timed on its own (min of 20 runs, launch and device sync included, after the timed steps; chosen by measured time, not by the cost model's own bound, which the column exists partly to check). `kN/M` is its launch position among M kernels, then its time and the nodes it writes. The number is its roofline lower bound over that time -- max(ops / peak FLOP/s, bytes / peak bandwidth) on the cost model's counts -- and `compute` / `memory` names the leg that binds. Ceilings, matched to the kernel's precision and mma status: `f32` = metal class constant: 5e+12 FLOP/s, 1e+12 B/s. `f32` is the backend's single-precision scalar constant (FMA counted as two); `f16-native` is twice it, for a kernel whose arithmetic is all 16-bit on a target where that is native. These are CLASS constants (or a machine's `model_peak_*` override), not this device's measured peak: the column ranks before/after on one box and certifies nothing about the device -- a number above 100% says the constant is below this card, not that the kernel beat physics. A small kernel's launch overhead counts against it. Printed only on an exact count: `approx` is a kernel whose BINDING leg's op or byte count is an upper bound (the calibration fit's per-leg rule; an upper bound on the leg that does not bind cannot overtake the exact one, so it leaves the number exact), `opaque` one with code the cost model cannot see, `no ceiling` one with nothing to score against (a C backend without `model_peak_*`, or a GPU tensor-core kernel -- no class constant exists for the mma unit, and the scalar f32 peak would read above 100%), `no kernel` a cell none of whose kernels could be timed alone; `+N untimed` counts kernels that could not, so the dominant one is dominant among the rest. `—` is a cell that did not run the instrument: the Python frameworks expose no per-kernel counts to score.

`mma` says what the timed artifact's kernels actually emitted, which is not what its schedule asked for: `tensorized` is at least one tensor-core / SIMD-tile emission, **`SCALAR FALLBACK`** is a schedule carrying a `Tensorize` whose every `Tile_mma` declined at codegen to the lane-0 scalar loop, **`NO MMA EMITTED`** is one that carries a `Tensorize` and emitted no `Tile_mma` at all, and `—` is an artifact that never asked for tensor cores. The two shouted verdicts mean the row is a scalar timing: quoting it as a tensor-core number is the error this column exists to stop.

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | kernel %peak | pass | mma | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ocannl | metal | tuned | f32 | 3.846 | 3.722 | 4.005 | 2.768 | 67.06 | 21.3 ocannl-seam | 9.9% f32 compute · k16/26 1.083 ms n20_relu.grad | replay | tensorized | NO-REF |
| ocannl | metal | default | f32 | 5.270 | 5.208 | 5.413 | 4.664 | 0.02 | 21.3 ocannl-seam | 9.0% f32 compute · k12/15 2.382 ms w2.grad n20_relu.grad | — | — | NO-REF |
| ocannl | cc | tuned | f32 | 33.254 | 32.504 | 34.049 | 33.349 | 454.13 | 26.4 ocannl-seam | no ceiling · k21/24 18.974 ms w2.grad n18.grad n20_relu.grad | replay | tensorized | NO-REF |
| ocannl | cc | default | f32 | 240.281 | 231.350 | 247.054 | 240.396 | 0.47 | 21.3 ocannl-seam | no ceiling · k2/7 199.183 ms w3.grad b3.grad w2.grad +13 | — | — | NO-REF |

## mac-studio: cc with `model_peak_*` set

Run as `OCANNL_MODEL_PEAK_FLOPS=2.2e12 OCANNL_MODEL_PEAK_MEMORY_BANDWIDTH=5.46e11 orchestrate.py --workloads mlp_wide --only ocannl --gpu none --no-fixture-digest-check`.

### Benchmark results

platform: macOS-26.6.2-arm64-arm-64bit arm64 | ocannl commit: 339d3db72 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager; reduced precisions get their own envelope: bf16 0.004, f16 0.002; the approximate regime 0.01)

measurement boxes declared by `fixtures/DIGESTS.txt`: m4-max, minix, rog-nv

ambient OCANNL_* environment: {"OCANNL_MODEL_PEAK_FLOPS": "2.2e12", "OCANNL_MODEL_PEAK_MEMORY_BANDWIDTH": "5.46e11"}


#### mlp_wide

measured on `mlp_wide.safetensors`, sha256 `b5e848d31e60d4feeecf8f4d1ab655e8623be29bf34c1d604d3641097a9a9d0a`, bytes no origin records

**MISSING DIGEST RECORD:** declared measurement box(es) with no entry for `mlp_wide.safetensors`: m4-max

`peak MiB` is the peak device footprint over the cell's TIMED STEPS (gh-ocannl-1006) -- bracketed there rather than read at process exit, so a tuned cell's schedule search, which allocates a candidate buffer per arm, is not in it. `—` is a cell whose framework exposes no device counter on this backend (a pytorch `cpu` row, say): not a zero, and no host-RSS figure is substituted for it. The counters are NOT one quantity, so every measured row names its own after the number: `ocannl-seam` = OCANNL allocator seam high-water (requested bytes, all backends). A high-water counter is exact over the window; one sampled at step boundaries is a lower bound on it, blind to an allocation made and given back within a step. Requested bytes off an allocator (OCANNL's seam, `torch.cuda.max_memory_allocated`) are the same quantity as each other and are NOT the device-wide figure a driver reports -- so rank rows within a counter, and read across counters only as the orders of magnitude they are.

`kernel %peak` is the attainment of the cell's DOMINANT KERNEL (gh-ocannl-1006): of the kernels the cell shipped -- a tuned cell's searched winner included -- the one that takes longest when each is timed on its own (min of 20 runs, launch and device sync included, after the timed steps; chosen by measured time, not by the cost model's own bound, which the column exists partly to check). `kN/M` is its launch position among M kernels, then its time and the nodes it writes. The number is its roofline lower bound over that time -- max(ops / peak FLOP/s, bytes / peak bandwidth) on the cost model's counts -- and `compute` / `memory` names the leg that binds. Ceilings, matched to the kernel's precision and mma status: `f32` = flops: model_peak_flops config; bandwidth: model_peak_memory_bandwidth config: 2.2e+12 FLOP/s, 5.46e+11 B/s. `f32` is the backend's single-precision scalar constant (FMA counted as two); `f16-native` is twice it, for a kernel whose arithmetic is all 16-bit on a target where that is native. These are CLASS constants (or a machine's `model_peak_*` override), not this device's measured peak: the column ranks before/after on one box and certifies nothing about the device -- a number above 100% says the constant is below this card, not that the kernel beat physics. A small kernel's launch overhead counts against it. Printed only on an exact count: `approx` is a kernel whose BINDING leg's op or byte count is an upper bound (the calibration fit's per-leg rule; an upper bound on the leg that does not bind cannot overtake the exact one, so it leaves the number exact), `opaque` one with code the cost model cannot see, `no ceiling` one with nothing to score against (a C backend without `model_peak_*`, or a GPU tensor-core kernel -- no class constant exists for the mma unit, and the scalar f32 peak would read above 100%), `no kernel` a cell none of whose kernels could be timed alone; `+N untimed` counts kernels that could not, so the dominant one is dominant among the rest. `—` is a cell that did not run the instrument: the Python frameworks expose no per-kernel counts to score.

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | kernel %peak | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ocannl | cc | default | f32 | 237.679 | 224.438 | 248.028 | 227.544 | 0.61 | 21.3 ocannl-seam | 0.1% f32 compute · k2/7 191.202 ms w3.grad b3.grad w2.grad +13 | NO-REF |

## rog-nv-linux (native Ubuntu, CUDA): CUDA and cc

Run as a `correctness` request through `tools/machine-verify.sh rog-nv-linux ... --expect-lib cudajit`
at 924573874 (the harness proved the `cudajit` arm was compiled and selected, `backend=cuda`), with
the same smoke `mlp_wide` generated on the box (`gen_fixtures.py --out-dir`; the bytes match
mac-studio's digest), as `orchestrate.py --workloads mlp_wide --only ocannl --gpu cuda
--no-fixture-digest-check`. The CUDA cell prints a number against the cuda class constants
(1.5e13 FLOP/s, 8e12 B/s): `1.5% f32 compute` on its dominant kernel, a forward matmul (`k3/15`,
537 MFLOP in 2.35 ms, exact op count binding over an upper-bound byte count). No kernel here is
tensorized (an untuned cell), so the `no ceiling` a tensor-core kernel would get by design does
not arise; the cc row prints `no ceiling`, as without `model_peak_*` it must.

### Benchmark results

platform: Linux-7.0.0-31-generic-x86_64-with-glibc2.43 x86_64 | ocannl commit: 92457387 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager; reduced precisions get their own envelope: bf16 0.004, f16 0.002; the approximate regime 0.01)

measurement boxes declared by `fixtures/DIGESTS.txt`: m4-max, minix, rog-nv

ambient OCANNL_* environment: none


#### mlp_wide

measured on `mlp_wide.safetensors`, sha256 `b5e848d31e60d4feeecf8f4d1ab655e8623be29bf34c1d604d3641097a9a9d0a`, bytes no origin records

**MISSING DIGEST RECORD:** declared measurement box(es) with no entry for `mlp_wide.safetensors`: m4-max

`peak MiB` is the peak device footprint over the cell's TIMED STEPS (gh-ocannl-1006) -- bracketed there rather than read at process exit, so a tuned cell's schedule search, which allocates a candidate buffer per arm, is not in it. `—` is a cell whose framework exposes no device counter on this backend (a pytorch `cpu` row, say): not a zero, and no host-RSS figure is substituted for it. The counters are NOT one quantity, so every measured row names its own after the number: `ocannl-seam` = OCANNL allocator seam high-water (requested bytes, all backends). A high-water counter is exact over the window; one sampled at step boundaries is a lower bound on it, blind to an allocation made and given back within a step. Requested bytes off an allocator (OCANNL's seam, `torch.cuda.max_memory_allocated`) are the same quantity as each other and are NOT the device-wide figure a driver reports -- so rank rows within a counter, and read across counters only as the orders of magnitude they are.

`kernel %peak` is the attainment of the cell's DOMINANT KERNEL (gh-ocannl-1006): of the kernels the cell shipped -- a tuned cell's searched winner included -- the one that takes longest when each is timed on its own (min of 20 runs, launch and device sync included, after the timed steps; chosen by measured time, not by the cost model's own bound, which the column exists partly to check). `kN/M` is its launch position among M kernels, then its time and the nodes it writes. The number is its roofline lower bound over that time -- max(ops / peak FLOP/s, bytes / peak bandwidth) on the cost model's counts -- and `compute` / `memory` names the leg that binds. Ceilings, matched to the kernel's precision and mma status: `f32` = cuda class constant: 1.5e+13 FLOP/s, 8e+12 B/s. `f32` is the backend's single-precision scalar constant (FMA counted as two); `f16-native` is twice it, for a kernel whose arithmetic is all 16-bit on a target where that is native. These are CLASS constants (or a machine's `model_peak_*` override), not this device's measured peak: the column ranks before/after on one box and certifies nothing about the device -- a number above 100% says the constant is below this card, not that the kernel beat physics. A small kernel's launch overhead counts against it. Printed only on an exact count: `approx` is a kernel whose BINDING leg's op or byte count is an upper bound (the calibration fit's per-leg rule; an upper bound on the leg that does not bind cannot overtake the exact one, so it leaves the number exact), `opaque` one with code the cost model cannot see, `no ceiling` one with nothing to score against (a C backend without `model_peak_*`, or a GPU tensor-core kernel -- no class constant exists for the mma unit, and the scalar f32 peak would read above 100%), `no kernel` a cell none of whose kernels could be timed alone; `+N untimed` counts kernels that could not, so the dominant one is dominant among the rest. `—` is a cell that did not run the instrument: the Python frameworks expose no per-kernel counts to score.

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | kernel %peak | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ocannl | cuda | default | f32 | 3.904 | 3.895 | 3.913 | 3.900 | 0.18 | 21.3 ocannl-seam | 1.5% f32 compute · k3/15 2.349 ms n22 n26_relu | NO-REF |
| ocannl | cc | default | f32 | 218.110 | 211.145 | 229.085 | 219.604 | 0.40 | 21.3 ocannl-seam | no ceiling · k2/7 186.386 ms w3.grad b3.grad w2.grad +13 | NO-REF |

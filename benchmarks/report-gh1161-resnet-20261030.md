# Benchmark results — cifar_resnet re-run with tuned cells (gh-ocannl-1161, gh-ocannl-1401)

> **Measured 2026-10-30 at staging `1fc57811`** on rog-nv-linux (CUDA), mac-studio (Metal) and
> minix-amd-linux (HIP, unified memory). Each box ran in the same exclusive measurement window as
> the [GPT-2 re-run](report-tagline-gpt2-20261030.md), after its GPT-2 part (requests
> `gh1181r2-{rog-nv-linux,mac-studio,minix-amd-linux}-measure-2`, under `fleet-worker.sh
> execution hold`), with `BENCH_KERNEL_TABLE=1`. Every quoted cell passed the parity gate.
> The baseline is [report-gh1161-resnet.md](report-gh1161-resnet.md), measured 2026-10-16 at
> `e075a2ca` (rog-nv, m4-max) and `578a381e` (minix).
>
> The fixtures are the baseline's bytes, copied from each box's main checkout, never regenerated:
> `cifar_resnet` content-v1 `065d246d…f9c4` and `cifar_resnet_b1` `2bda4d0d…c68a`, both
> MATCH on every box.

## The answers

The baseline left two questions: whether OCANNL's GPU step was still 50-140x behind PyTorch's,
and whether OCANNL's GPU step at batch 1 was still slower than its own cc. "x" means how many
times slower OCANNL is.

**Is OCANNL GPU still 50-140x behind PyTorch GPU?** At batch 64, no, except on Metal. At batch 1
untuned, yes on every box. Tuned, OCANNL is 4-20x behind.

| cifar_resnet exact, OCANNL GPU vs torch GPU | untuned, Oct 16 | untuned, now | tuned, now |
|---|---|---|---|
| batch 64, rog-nv (CUDA) | 106x | **11.7x** | **11.3x** |
| batch 64, minix (HIP) | 22x | **6.1x** | **5.2x** |
| batch 64, m4-max (Metal) | 141x | **94.5x** | **20.2x** |
| batch 1, rog-nv (CUDA) | 50x | 50.8x | **4.1x** |
| batch 1, minix (HIP) | 85x | 85.4x | **9.3x** |
| batch 1, m4-max (Metal) | 291x | 230x | **17.5x** |

- The batch-64 drop on CUDA and HIP is in the **untuned** column: rog went from 184.0 to 20.4 ms
  and minix from 146.0 to 33.9 ms.
  - Metal's untuned GPU column did not move, which is consistent with the cause being
    gh-ocannl-1324's seeded untuned default. That default is CUDA/HIP only, and it landed between
    the two runs. This report did not bisect it.
  - Untuned batch 1 is flat on every GPU, so the seeding does not move those steps.
- **Caveat, Metal.** The untuned Metal column still measures the old default:
  gh-ocannl-1324's `seeded_default` is not enabled on Metal while ahrefs/ocannl#1404 is open.
  Read "94.5x untuned on Metal" as that old default.
- **Metal's PyTorch moved too.** torch MPS reads 1.45x slower than on Oct 16 at batch 64 (2.77 to
  4.01 ms) and 1.27x slower at batch 1. The box moved to macOS 27.0.1 in between, and the GPT-2
  report sees the same drift.
  - So Metal's improvement against PyTorch (141x to 94.5x) is mostly PyTorch getting slower: the
    OCANNL Metal step is flat (391.6 to 379.2 ms).
  - Against tinygrad's Metal JIT, which did not move, untuned Metal is 61x both then and now.

**Is OCANNL GPU at batch 1 still slower than its own cc?** Yes, on every box, untuned and tuned.

| cifar_resnet_b1 exact | cc untuned | GPU untuned | GPU / cc | cc tuned | GPU tuned | GPU / cc, tuned |
|---|---|---|---|---|---|---|
| rog-nv (CUDA) | 3.01 | 64.53 | 21x | 2.00 | 5.19 | 2.6x |
| minix (HIP) | 2.71 | 181.93 | 67x | 2.13 | 19.75 | 9.3x |
| m4-max (Metal) | 3.98 | 549.19 | 138x | 2.12 | 41.68 | 19.7x |

Tuning closes most of the gap (the GPU step falls 9-13x), but the tuned GPU step is still 2.6x
(CUDA) to 20x (Metal) slower than cc. At batch 64 the GPU is ahead of cc on CUDA and HIP, by
3-5x tuned and 3-8x untuned. On Metal untuned it is 4.2x behind, and tuned 1.2x behind
(80.7 against 65.4 ms).

## Step cells, untuned (step p50 ms)

The baseline report's table, re-measured. tinygrad runs one regime, its own defaults, and is filed
under `approximate`; the baseline's minix sweep filed it under `exact`.

| workload | regime | box | OCANNL cc | OCANNL GPU | torch CPU | torch GPU | tinygrad CPU | tinygrad GPU |
|---|---|---|---|---|---|---|---|---|
| cifar_resnet | exact | rog-nv (CUDA, RTX 5070 Ti) | 155.33 | 20.39 | 10.35 (REF) | 1.74 | — | — |
| cifar_resnet | exact | m4-max (Metal) | 90.52 | 379.22 | 98.33 (REF) | 4.01 | — | — |
| cifar_resnet | exact | minix (HIP gfx1151, unified) | 99.49 | 33.93 | 9.81 (REF) | 5.53 | — | — |
| cifar_resnet | approximate | rog-nv (CUDA, RTX 5070 Ti) | 143.54 | 17.25 | 10.49 | 1.85 | 300.44 | 5.17 |
| cifar_resnet | approximate | m4-max (Metal) | 80.02 | 382.15 | 98.07 | 4.05 | 50.42 | 6.21 |
| cifar_resnet | approximate | minix (HIP gfx1151, unified) | 240.06 | 25.02 | 8.32 | 5.93 | 302.67 | 6.07 |
| cifar_resnet_b1 | exact | rog-nv (CUDA, RTX 5070 Ti) | 3.01 | 64.53 | 1.48 (REF) | 1.27 | — | — |
| cifar_resnet_b1 | exact | m4-max (Metal) | 3.98 | 549.19 | 1.92 (REF) | 2.39 | — | — |
| cifar_resnet_b1 | exact | minix (HIP gfx1151, unified) | 2.71 | 181.93 | 1.57 (REF) | 2.13 | — | — |
| cifar_resnet_b1 | approximate | rog-nv (CUDA, RTX 5070 Ti) | 2.74 | 63.84 | 1.45 | 1.30 | 7.20 | 0.70 |
| cifar_resnet_b1 | approximate | m4-max (Metal) | 2.76 | 553.66 | 2.01 | 2.35 | 6.57 | 1.04 |
| cifar_resnet_b1 | approximate | minix (HIP gfx1151, unified) | 2.68 | 182.00 | 1.64 | 2.13 | 8.09 | 0.79 |

**Caveat, minix cc.** minix's parallel cc cells carry 2-5x p90/p10 tails (OpenMP fork/join
stragglers, ahrefs/ocannl#1611).
- The worst case here is `cifar_resnet/cc/default/approximate`: p10 140.1, p50 240.1, p90 430.0 ms.
  Its p50 against the exact cell's 99.5 ms is that tail, not the regime.
- Read minix's cc column, and every minix cc ratio below, as a noisy reading.
- The torch GPU cell that segfaulted on Oct 16 (`cifar_resnet` approximate) ran this time, on the
  new torch 2.14.1+rocm7.2 wheel.

## Change since Oct 16, per cell

Oct 16 value to now (step p50 ms). "Flat" is within 5%.

| workload | regime | box | OCANNL cc | OCANNL GPU | torch CPU | torch GPU | tinygrad CPU | tinygrad GPU |
|---|---|---|---|---|---|---|---|---|
| cifar_resnet | exact | rog-nv | 520.68 → 155.33 (3.35x faster) | 184.01 → 20.39 (9.02x faster) | 11.17 → 10.35 (1.08x faster) | 1.74 → 1.74 (flat) | — | — |
| cifar_resnet | exact | m4-max | 406.67 → 90.52 (4.49x faster) | 391.61 → 379.22 (flat) | 99.49 → 98.33 (flat) | 2.77 → 4.01 (1.45x slower) | — | — |
| cifar_resnet | exact | minix | 524.88 → 99.49 (5.28x faster) | 145.96 → 33.93 (4.30x faster) | 9.29 → 9.81 (1.06x slower) | 6.54 → 5.53 (1.18x faster) | — | — |
| cifar_resnet | approximate | rog-nv | 518.51 → 143.54 (3.61x faster) | 184.81 → 17.25 (10.72x faster) | 11.68 → 10.49 (1.11x faster) | 1.71 → 1.85 (1.08x slower) | 301.17 → 300.44 (flat) | 5.18 → 5.17 (flat) |
| cifar_resnet | approximate | m4-max | 404.16 → 80.02 (5.05x faster) | 389.98 → 382.15 (flat) | 99.75 → 98.07 (flat) | 2.81 → 4.05 (1.44x slower) | 51.19 → 50.42 (flat) | 6.24 → 6.21 (flat) |
| cifar_resnet | approximate | minix | 522.64 → 240.06 (2.18x faster; #1611 tail) | 145.91 → 25.02 (5.83x faster) | 9.62 → 8.32 (1.16x faster) | segfault → 5.93 | 305.40 → 302.67 (flat) | 6.08 → 6.07 (flat) |
| cifar_resnet_b1 | exact | rog-nv | 7.95 → 3.01 (2.65x faster) | 63.94 → 64.53 (flat) | 1.49 → 1.48 (flat) | 1.29 → 1.27 (flat) | — | — |
| cifar_resnet_b1 | exact | m4-max | 6.38 → 3.98 (1.60x faster) | 547.99 → 549.19 (flat) | 1.92 → 1.92 (flat) | 1.88 → 2.39 (1.27x slower) | — | — |
| cifar_resnet_b1 | exact | minix | 8.17 → 2.71 (3.01x faster) | 181.59 → 181.93 (flat) | 1.56 → 1.57 (flat) | 2.13 → 2.13 (flat) | — | — |
| cifar_resnet_b1 | approximate | rog-nv | 7.93 → 2.74 (2.90x faster) | 64.09 → 63.84 (flat) | 1.46 → 1.45 (flat) | 1.30 → 1.30 (flat) | 7.22 → 7.20 (flat) | 0.70 → 0.70 (flat) |
| cifar_resnet_b1 | approximate | m4-max | 6.34 → 2.76 (2.30x faster) | 546.29 → 553.66 (flat) | 1.89 → 2.01 (1.06x slower) | 1.87 → 2.35 (1.26x slower) | 6.20 → 6.57 (1.06x slower) | 1.01 → 1.04 (flat) |
| cifar_resnet_b1 | approximate | minix | 8.15 → 2.68 (3.05x faster) | 181.99 → 182.00 (flat) | 1.64 → 1.64 (flat) | 2.14 → 2.13 (flat) | 8.15 → 8.09 (flat) | 0.79 → 0.79 (flat) |

- **The OCANNL cc step is 1.6-5.3x faster on every box and both workloads.** The other frameworks'
  CPU cells did not move. The 343 first-parent merges between the two runs are not attributed
  here.
- **The OCANNL GPU step is 4.3-10.7x faster at batch 64 on CUDA and HIP, and flat everywhere
  else.**
- The other frameworks are flat on Linux. On the Mac, torch MPS is 1.26-1.45x slower; see the
  first section.

## Tuned cells

New since the baseline, where no tuned row passed the provenance gate before ahrefs/ocannl#1384.

**Protocol:**
- **One orchestrate sweep per tuned cell.** Every other OCANNL cell of the sweep was left out
  with `--skip-cell`. The pytorch cells stayed in, because `pytorch/cpu/eager` exact is the
  parity reference.
- **A fresh cache per sweep.** Each sweep searched into an absolute `OCANNL_AUTOTUNE_CACHE_DIR`
  of its own, `~/.local/state/gh1181-r2/resnet/<sweep>/autotune_cache`, created empty for that
  sweep. Nothing came from the shared `benchmarks/autotune_cache/` or from a prep smoke.
- **Search cost.** `search s` is that search's compile time, from empty.
- **Provenance.** Every quoted row is pass 2, a fresh process that replayed the whole cache
  (`REPLAY` after a `SEARCHED` pass 1), and passed parity.
- **CUDA's own cache.** rog's `~/.nv/ComputeCache` (the CUDA driver's JIT cache) was left alone
  and survives the wipe. So rog's search costs are lower bounds: a candidate kernel compiled in an
  earlier run is not recompiled.

| workload | regime | box | OCANNL cell | step p50 ms | p10-p90 | search s (fresh cache) | pass | parity | untuned now | tuned / untuned |
|---|---|---|---|---|---|---|---|---|---|---|
| cifar_resnet | exact | rog-nv (CUDA) | cuda/tuned | **19.85** | 19.80-19.89 | 190 | REPLAY | PASS (2.1e-07) | 20.39 | 0.97 |
| cifar_resnet | exact | rog-nv (CUDA) | cuda/tuned, `schedule_fission=false` | **21.30** | 21.27-21.36 | 552 | REPLAY | PASS (2.1e-07) | 20.39 | 1.04 |
| cifar_resnet | approximate | rog-nv (CUDA) | cuda/tuned | **16.21** | 16.16-16.27 | 744 | REPLAY | PASS (9.4e-06) | 17.25 | 0.94 |
| cifar_resnet | exact | rog-nv (CUDA) | cc/tuned | **103.87** | 98.97-111.73 | 863 | REPLAY | PASS (1.2e-06) | 155.33 | 0.67 |
| cifar_resnet | exact | m4-max (Metal) | metal/tuned | **80.69** | 80.29-81.27 | 3,276 | REPLAY | PASS (1.0e-06) | 379.22 | 0.21 |
| cifar_resnet | approximate | m4-max (Metal) | metal/tuned | **missing** | - | >3,600 (cap) | search timed out | - | 382.15 | - |
| cifar_resnet | exact | m4-max (Metal) | cc/tuned | **65.39** | 64.41-66.44 | 711 | REPLAY | PASS (8.3e-07) | 90.52 | 0.72 |
| cifar_resnet | exact | minix (HIP) | hip/tuned | **30.72** | 29.87-31.62 | 247 | REPLAY | PASS (1.7e-06) | 33.93 | 0.91 |
| cifar_resnet | approximate | minix (HIP) | hip/tuned | **22.69** | 22.03-23.74 | 410 | REPLAY | PASS (1.7e-06) | 25.02 | 0.91 |
| cifar_resnet | exact | minix (HIP) | cc/tuned | **101.42** | 68.43-229.66 | 859 | REPLAY | PASS (1.7e-06) | 99.49 | 1.02 (#1611 tail) |
| cifar_resnet_b1 | exact | rog-nv (CUDA) | cuda/tuned | **5.19** | 5.18-5.19 | 43 | REPLAY | PASS (1.1e-07) | 64.53 | 0.08 |
| cifar_resnet_b1 | approximate | rog-nv (CUDA) | cuda/tuned | **5.36** | 5.36-5.37 | 352 | REPLAY | PASS (1.1e-07) | 63.84 | 0.08 |
| cifar_resnet_b1 | exact | rog-nv (CUDA) | cc/tuned | **2.00** | 1.97-3.69 | 127 | REPLAY | PASS (1.1e-07) | 3.01 | 0.67 |
| cifar_resnet_b1 | exact | m4-max (Metal) | metal/tuned | **41.68** | 41.42-42.26 | 238 | REPLAY | PASS (1.1e-07) | 549.19 | 0.08 |
| cifar_resnet_b1 | approximate | m4-max (Metal) | metal/tuned | **46.22** | 43.27-46.85 | 545 | REPLAY | PASS (1.1e-07) | 553.66 | 0.08 |
| cifar_resnet_b1 | exact | m4-max (Metal) | cc/tuned | **2.12** | 2.00-2.25 | 99 | REPLAY | PASS (1.1e-07) | 3.98 | 0.53 |
| cifar_resnet_b1 | exact | minix (HIP) | hip/tuned | **19.75** | 19.31-20.17 | 59 | REPLAY | PASS (1.1e-07) | 181.93 | 0.11 |
| cifar_resnet_b1 | approximate | minix (HIP) | hip/tuned | **19.96** | 19.46-20.42 | 523 | REPLAY | PASS (1.1e-07) | 182.00 | 0.11 |
| cifar_resnet_b1 | exact | minix (HIP) | cc/tuned | **2.13** | 2.12-2.14 | 94 | REPLAY | PASS (2.0e-07) | 2.71 | 0.79 |

- **Batch 1 is where tuning pays.** The GPU step falls 9-13x (CUDA 64.5 to 5.2 ms, HIP 181.9 to
  19.8, Metal 549.2 to 41.7). The untuned batch-1 GPU default is the cell gh-ocannl-1324's seeding
  did not move.
- **At batch 64 on CUDA and HIP, the tuner adds 3-9% over the seeded untuned default.** On Metal,
  where that default is not enabled, tuning is worth 4.7x (379.2 to 80.7 ms).
- **gh-ocannl-503's cell, fission off.** `cifar_resnet/cuda/tuned/exact` with
  `--cell-flag cifar_resnet/cuda/tuned:schedule_fission=false` replays at 21.30 ms, against 19.85
  ms with fission on. Fission is worth 7% on this cell, and its search was 2.9x cheaper (190
  against 552 s).
- **The one missing cell is Metal batch-64 approximate.** Its search ran into the 3,600 s cell cap.
  - The exact one finished at 3,276 s, the only tuned search within 10% of the cap.
  - The cap was not raised: the budget kept each tuned cell under 3,600 s.
- **minix sweep exit status.** The `tuned-cc-cifar_resnet-exact` sweep exited 1 because its
  `pytorch/cuda(hip)/eager` cell segfaulted (exit -11, the ROCm torch crash the baseline also
  saw). The OCANNL cc cell and the CPU reference in the same sweep are unaffected.

### OCANNL GPU against PyTorch GPU and tinygrad GPU

"x" is how many times slower OCANNL is. Each untuned ratio pairs cells of one sweep. Each tuned
ratio uses the torch GPU cell of its own sweep, and the untuned sweep's tinygrad GPU JIT cell
(tinygrad has no tuned arm here). Oct 16's ratio is in parentheses where the baseline had both
cells.

| workload | regime | box | untuned vs torch GPU | tuned vs torch GPU | untuned vs tinygrad GPU | tuned vs tinygrad GPU |
|---|---|---|---|---|---|---|
| cifar_resnet | exact | rog-nv | 11.7x (106x) | 11.3x | 3.9x | 3.8x |
| cifar_resnet | exact | m4-max | 94.5x (141x) | 20.2x | 61.0x | 13.0x |
| cifar_resnet | exact | minix | 6.1x (22.3x) | 5.2x | 5.6x (24.1x) | 5.1x |
| cifar_resnet | approximate | rog-nv | 9.3x (108x) | 9.5x | 3.3x (35.7x) | 3.1x |
| cifar_resnet | approximate | m4-max | 94.2x (139x) | — | 61.5x (62.5x) | — |
| cifar_resnet | approximate | minix | 4.2x (—) | 4.1x | 4.1x (24.0x) | 3.7x |
| cifar_resnet_b1 | exact | rog-nv | 50.8x (49.6x) | 4.1x | 92.1x | 7.4x |
| cifar_resnet_b1 | exact | m4-max | 230x (291x) | 17.5x | 528x | 40.1x |
| cifar_resnet_b1 | exact | minix | 85.4x (85.3x) | 9.3x | 231x (230x) | 25.1x |
| cifar_resnet_b1 | approximate | rog-nv | 49.1x (49.3x) | 4.0x | 91.1x (91.6x) | 7.7x |
| cifar_resnet_b1 | approximate | m4-max | 235x (292x) | 20.4x | 533x (541x) | 44.5x |
| cifar_resnet_b1 | approximate | minix | 85.3x (85.0x) | 9.8x | 231x (230x) | 25.3x |

The exact-regime tinygrad ratios use the approximate tinygrad cell, since tinygrad has no exact
pin. The Metal torch ratios carry the MPS drift described in the first section.

## gh-ocannl-1401 coverage

- **Requested and recorded**, each tuned row a pass-2 replay from a fresh absolute cache:
  - `cifar_resnet{,_b1}/{cc,cuda,metal,hip}/tuned/exact`: rog cc and cuda, m4-max cc and metal,
    minix cc and hip.
  - `cifar_resnet/cuda/tuned/exact` with `schedule_fission=false`.
  - The minix tuned redo.
- **Recorded beyond the issue:** the tuned approximate GPU cells (all but Metal batch 64).
- **Not covered, as the issue says:** the HIP and Metal per-site cells, which wait on
  ahrefs/ocannl#1387.

## Reproducing a sweep

One sweep per tuned cell, from a checkout with the venv linked and the fixtures copied in, as in
the baseline report:

```bash
d=<absolute-results-dir>; mkdir -p $d
BENCH_KERNEL_TABLE=1 BENCH_RESULTS_DIR=$d BENCH_CELL_LOG_DIR=$d/cells \
  OCANNL_AUTOTUNE_CACHE_DIR=$d/autotune_cache \
  benchmarks/.venv/bin/python benchmarks/orchestrate.py --box <box> --gpu <cuda|hip|metal> \
  --workloads cifar_resnet --profile exact --only ocannl pytorch --cell-timeout 3600 --tuned \
  --skip-cell cifar_resnet/cc/default --skip-cell cifar_resnet/cc/tuned \
  --skip-cell cifar_resnet/<gpu>/default
```

The untuned sweeps drop `--tuned` and the `--skip-cell`s. They add `tinygrad` to `--only` in
the approximate regime. The fission-off cell adds
`--cell-flag cifar_resnet/cuda/tuned:schedule_fission=false`. The sweep reports, the
`results.jsonl` files and the cell logs of this run are not checked in.

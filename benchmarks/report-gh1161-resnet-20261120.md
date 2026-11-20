# Benchmark results: cifar_resnet at the 1.2 tag, with a same-window October 30 anchor (gh-ocannl-2008)

> **Measured 2026-11-20 at the `1.2` tag, staging `c3e5975a`,** on rog-nv-linux (CUDA, discrete
> VRAM), minix-amd-linux (HIP gfx1151, unified memory) and mac-studio (Metal, Apple M4 Max,
> unified memory).
>
> - **Windows.** Each box ran in the same exclusive window as the
>   [GPT-2 re-run](report-tagline-gpt2-20261120.md), after its GPT-2 part, with
>   `BENCH_KERNEL_TABLE=1`. The requests were `wave-20261119-ocannl-t2008-<box>-1`.
> - **Driver.** The harness was lukstafi/ocannl-staging PR #2243's branch (`a98c641b` on Linux,
>   `479ac9ab` on mac-studio), measuring the tag's runners through `BENCH_OCANNL_TREE`.
> - **Gates.** Every quoted cell passed the parity gate.
> - **Baseline.** The [October 30 re-run](report-gh1161-resnet-20261030.md) at `1fc57811`, and
>   new here, `1fc57811`'s untuned cells re-measured in this window.
>
> **The box stack changed on 2026-11-06.** These windows ran rog and minix on kernel
> **7.0.0-43**, not the -42 that ahrefs/ocannl#2004 recorded. rog ran NVIDIA **620.70.06**
> (CUDA 13.5); mac-studio builds against the **macOS 27.2** SDK. So only the same-window anchor
> columns compare cleanly with October 30.
>
> **Fixtures.** The baseline's bytes, copied from each box's main checkout and never
> regenerated: `cifar_resnet` content-v1 `065d246d…f9c4` and `cifar_resnet_b1` `2bda4d0d…c68a`,
> both MATCH on every box.

## The answers

"x" means how many times slower OCANNL is.

**Is OCANNL's GPU step still far behind PyTorch's?**
- **Batch 64:** 9-11x on CUDA, 3.5-5.7x on HIP, and 100x untuned or 21.6x tuned on Metal.
- **Batch 1, untuned:** 48-88x on CUDA and HIP, 242x on Metal.
- **Tuned:** 3.5-11x on CUDA and HIP, 19-22x on Metal.
- Since October 30 the GPU step moved only on HIP, untuned: 1.06x at batch 64 exact, 1.21x at
  batch 64 approximate and 1.03x at batch 1. The untuned CUDA and Metal GPU steps are flat.

| cifar_resnet, OCANNL GPU vs torch GPU | untuned, `1fc57811` (this window) | untuned, 1.2 tag | tuned, 1.2 tag | tuned, October 30 |
|---|---|---|---|---|
| batch 64 exact, rog-nv (CUDA) | 11.7x | **9.8x** (see note) | **11.3x** | 11.3x |
| batch 64 exact, minix (HIP) | 5.7x | **5.4x** (torch cell crashed; tuned sweep's torch cell) | **5.4x** | 5.2x |
| batch 1 exact, rog-nv (CUDA) | 51.1x | **51.0x** | **4.1x** | 4.1x |
| batch 1 exact, minix (HIP) | 85.4x | **83.2x** | **9.2x** | 9.3x |
| batch 64 exact, m4-max (Metal) | 104x | **100x** | **21.6x** | 20.2x |
| batch 1 exact, m4-max (Metal) | 242x | **242x** | **18.8x** | 17.5x |

**Note, rog torch GPU.** In the tag's two untuned batch-64 sweeps, the torch CUDA cell read 2.08 ms
(exact) and 2.01 ms (approximate). Those were the first sweeps after the GPT-2 part. In every
later sweep of the window it read 1.69-1.76 ms, so the 9.8x flatters OCANNL. The tuned row's
11.3x uses a 1.76 ms torch cell; against that cell the untuned tag's 20.45 ms is 11.6x.

**Is OCANNL's GPU step at batch 1 still slower than its own cc?** Yes, on every box:
- untuned by 32x (CUDA), 81x (HIP) and 265x (Metal), against 21x, 67x and 138x on October 30,
  because the cc step got faster (below);
- tuned by 2.6x (CUDA), 9.3x (HIP) and 21x (Metal), against 2.6x, 9.3x and 19.7x.

At batch 64 the GPU is 5.2-5.3x ahead of cc on CUDA and 2.1-2.6x ahead on HIP. On Metal it is
behind: 5.8x untuned and 1.24x tuned (79.79 against 64.35 ms).

## What moved since October 30, and which change moved it

Same window, untuned, the tag against `1fc57811` (step p50 ms; "flat" is within 5%):

| workload | regime | box | OCANNL cc: `1fc57811` → tag | OCANNL GPU: `1fc57811` → tag |
|---|---|---|---|---|
| cifar_resnet | exact | rog-nv | 154.52 → **108.39 (1.43x faster)** | 20.44 → 20.45 (flat) |
| cifar_resnet | approximate | rog-nv | 141.02 → 141.85 (flat) | 17.32 → 17.32 (flat) |
| cifar_resnet_b1 | exact | rog-nv | 3.01 → **2.02 (1.49x faster)** | 64.59 → 64.36 (flat) |
| cifar_resnet_b1 | approximate | rog-nv | 2.77 → 2.90 (flat; p90 3.46) | 63.84 → 63.63 (flat) |
| cifar_resnet | exact | minix | 113.86 → **81.54 (1.40x faster; #1611 tails)** | 33.76 → **31.78 (1.06x faster)** |
| cifar_resnet | approximate | minix | 244.86 → 219.00 (#1611 tails, unreadable) | 25.16 → **20.86 (1.21x faster)** |
| cifar_resnet_b1 | exact | minix | 2.72 → **2.19 (1.25x faster)** | 182.01 → 177.49 (flat, 1.03x) |
| cifar_resnet_b1 | approximate | minix | 2.70 → 2.76 (flat) | 182.66 → 177.09 (flat, 1.03x) |
| cifar_resnet | exact | m4-max | 90.93 → **68.89 (1.32x faster)** | 385.27 → 397.12 (flat, 1.03x slower) |
| cifar_resnet | approximate | m4-max | 80.84 → 81.42 (flat) | 385.11 → 396.42 (flat, 1.03x slower) |
| cifar_resnet_b1 | exact | m4-max | 3.83 → **2.08 (1.84x faster)** | 547.59 → 550.24 (flat) |
| cifar_resnet_b1 | approximate | m4-max | 2.79 → 2.73 (flat) | 545.56 → 545.75 (flat) |

- **gh-ocannl-1630 (per-channel vector accumulators on untuned cc convs) moved the exact cc
  cells: 1.25-1.84x faster on all three boxes.**
  - It is the one listed change that acts on untuned cc convs. The size matches its own A/B: 0.753
    on an M4 Max and 0.778 on a Zen 5 box at batch 64, against 0.70 on rog, 0.72 on minix and
    0.76 on mac-studio here.
  - **The approximate cc cells did not move**, on any box or either batch. On rog the tag's
    approximate cc step (141.85 ms) is now 1.31x *slower* than its exact one (108.39 ms). That
    reads as gh-ocannl-1630's accumulators not engaging under `--ocannl_profile=approximate`,
    though this window does not show why; proposed as a follow-up.
- **The untuned GPU cells are flat on CUDA and Metal and moved on HIP**: 1.21x at batch 64 approximate and
  1.06x exact. Of the listed changes,
  - gh-ocannl-1697's seeded calibration and gh-ocannl-1661's staging copy act on tuned searches
    and transfers;
  - gh-ocannl-1736's pad-masked registers act on the GPU kernels themselves.
  
  The window does not separate them, and none of their records names an untuned HIP conv cell.
  So the HIP move is not attributed.
- **The tuned cells are flat across windows** (next section):
  - Every tuned GPU cell is within 4% of October 30, except minix's batch-64 approximate (22.69 to
    20.58 ms, 1.10x).
  - The tuned cc cells are flat, except minix's batch-64 cell, whose October 30 #1611 tail is gone.
  - The tuned cells had no same-window anchor (it would have doubled each window), so these are
    cross-window comparisons.
  - Both the anchor's untuned cells and the other frameworks stayed flat across the windows,
    which is what makes those comparisons readable at all.
- **gh-ocannl-1866/1975** act on reduced-precision storage, and every cell here is f32, so they
  are not measured. **`Schedule_cache.entry_version` 23** forces a re-search, which these cold
  caches did anyway.

**Caveat, minix cc.** minix's parallel cc cells still carry 1.5-4x p10-p90 tails (OpenMP
fork/join stragglers, ahrefs/ocannl#1611). The batch-64 cc cells span 77-151 ms (exact) and
116-390 ms (approximate) at the tag. Read minix's batch-64 cc column as noisy; the batch-1 cc
cells are tight.

## Step cells, untuned, at the 1.2 tag (step p50 ms)

tinygrad runs one regime, its own defaults, and is filed under `approximate`.

| workload | regime | box | OCANNL cc | OCANNL GPU | torch CPU | torch GPU | tinygrad CPU | tinygrad GPU |
|---|---|---|---|---|---|---|---|---|
| cifar_resnet | exact | rog-nv (CUDA, RTX 5070 Ti) | 108.39 | 20.45 | 11.68 (REF) | 2.08 | — | — |
| cifar_resnet | exact | minix (HIP gfx1151, unified) | 81.54 | 31.78 | 10.46 (REF) | crashed (exit -11) | — | — |
| cifar_resnet | approximate | rog-nv (CUDA, RTX 5070 Ti) | 141.85 | 17.32 | 13.76 | 2.01 | 300.97 | 5.17 |
| cifar_resnet | approximate | minix (HIP gfx1151, unified) | 219.00 | 20.86 | 9.24 | 5.90 | 305.96 | 6.05 |
| cifar_resnet_b1 | exact | rog-nv (CUDA, RTX 5070 Ti) | 2.02 | 64.36 | 1.44 (REF) | 1.26 | — | — |
| cifar_resnet_b1 | exact | minix (HIP gfx1151, unified) | 2.19 | 177.49 | 1.64 (REF) | 2.13 | — | — |
| cifar_resnet_b1 | approximate | rog-nv (CUDA, RTX 5070 Ti) | 2.90 | 63.63 | 1.53 | 1.32 | 7.28 | 0.70 |
| cifar_resnet_b1 | approximate | minix (HIP gfx1151, unified) | 2.76 | 177.09 | 1.64 | 2.13 | 8.17 | 0.79 |
| cifar_resnet | exact | m4-max (Metal) | 68.89 | 397.12 | 98.05 (REF) | 3.97 | — | — |
| cifar_resnet | approximate | m4-max (Metal) | 81.42 | 396.42 | 98.21 | 3.98 | 56.33 | 6.24 |
| cifar_resnet_b1 | exact | m4-max (Metal) | 2.08 | 550.24 | 2.00 (REF) | 2.28 | — | — |
| cifar_resnet_b1 | approximate | m4-max (Metal) | 2.73 | 545.75 | 1.91 | 2.26 | 6.13 | 1.06 |

- **minix torch GPU crash.** The `pytorch/cuda(hip)/eager` cell of the untuned batch-64 exact
  sweep segfaulted (exit -11). This is the ROCm torch crash that the October 16 baseline and the
  October 30 re-run each also hit once.
  - It is that sweep's only failure, and the reason the minix leg exited 1.
  - Every other torch HIP cell of the window ran: batch 64 exact 5.85-5.90 ms in the anchor and
    tuned sweeps.
- The other frameworks are flat against October 30 on all three boxes; on mac-studio torch MPS reads 3.97 against 4.01 ms, so the October 30 macOS drift did not recur. On rog, torch GPU is flat
  outside the two cells the note above names.

## Tuned cells, at the 1.2 tag

**Protocol.** As on October 30:
- **Sweeps.** One orchestrate sweep per tuned cell, with every other OCANNL cell left out by
  `--skip-cell`. The pytorch cells stayed in, since `pytorch/cpu/eager` exact is the parity
  reference.
- **Cold caches.** Each sweep searched into an empty absolute `OCANNL_AUTOTUNE_CACHE_DIR` of its
  own, `~/.local/state/gh2008/window/resnet/<sweep>/autotune_cache`. Nothing came from
  `benchmarks/autotune_cache/` or a prep smoke.
- **CUDA.** The driver's `~/.nv/ComputeCache` was left alone, so rog's search costs are lower
  bounds.
- **Provenance.** Every quoted row is pass 2: a fresh process that replayed the whole cache
  (`REPLAY` after a `SEARCHED` pass 1), and it passed parity.

| workload | regime | box | OCANNL cell | step p50 ms | p10-p90 | search s (fresh cache) | parity | October 30 | tuned / untuned now |
|---|---|---|---|---|---|---|---|---|---|
| cifar_resnet | exact | rog-nv (CUDA) | cuda/tuned | **19.82** | 19.79-19.86 | 140 | PASS (2.1e-07) | 19.85 | 0.97 |
| cifar_resnet | exact | rog-nv (CUDA) | cuda/tuned, `schedule_fission=false` | **21.32** | 21.29-21.38 | 546 | PASS (2.1e-07) | 21.30 | 1.04 |
| cifar_resnet | approximate | rog-nv (CUDA) | cuda/tuned | **15.76** | 15.73-15.82 | 518 | PASS (8.3e-07) | 16.21 | 0.91 |
| cifar_resnet | exact | rog-nv (CUDA) | cc/tuned | **102.22** | 96.88-108.46 | 709 | PASS (2.1e-07) | 103.87 | 0.94 |
| cifar_resnet | exact | minix (HIP) | hip/tuned | **31.35** | 28.81-32.26 | 214 | PASS (1.7e-06) | 30.72 | 0.99 |
| cifar_resnet | approximate | minix (HIP) | hip/tuned | **20.58** | 19.92-23.26 | 355 | PASS (1.7e-06) | 22.69 | 0.99 |
| cifar_resnet | exact | minix (HIP) | cc/tuned | **66.75** | 66.28-67.16 | 641 | PASS (1.8e-06) | 101.42 (#1611 tail) | 0.82 |
| cifar_resnet_b1 | exact | rog-nv (CUDA) | cuda/tuned | **5.21** | 5.20-5.21 | 49 | PASS (1.1e-07) | 5.19 | 0.08 |
| cifar_resnet_b1 | approximate | rog-nv (CUDA) | cuda/tuned | **5.38** | 5.37-5.39 | 361 | PASS (1.1e-07) | 5.36 | 0.08 |
| cifar_resnet_b1 | exact | rog-nv (CUDA) | cc/tuned | **1.98** | 1.94-2.27 | 113 | PASS (1.1e-07) | 2.00 | 0.98 |
| cifar_resnet_b1 | exact | minix (HIP) | hip/tuned | **19.27** | 18.81-19.72 | 109 | PASS (1.1e-07) | 19.75 | 0.11 |
| cifar_resnet_b1 | approximate | minix (HIP) | hip/tuned | **19.45** | 18.95-19.80 | 508 | PASS (1.1e-07) | 19.96 | 0.11 |
| cifar_resnet_b1 | exact | minix (HIP) | cc/tuned | **2.06** | 2.06-2.07 | 78 | PASS (2.0e-07) | 2.13 | 0.94 |
| cifar_resnet | exact | m4-max (Metal) | metal/tuned | **79.79** | 79.57-81.98 | 3,155 | PASS (1.0e-06) | 80.69 | 0.20 |
| cifar_resnet | approximate | m4-max (Metal) | metal/tuned | **missing** | - | >3,600 (cap) | search timed out | missing (same cap) | - |
| cifar_resnet | exact | m4-max (Metal) | cc/tuned | **64.35** | 63.53-65.54 | 629 | PASS (8.3e-07) | 65.39 | 0.93 |
| cifar_resnet_b1 | exact | m4-max (Metal) | metal/tuned | **41.79** | 41.51-42.05 | 217 | PASS (1.1e-07) | 41.68 | 0.08 |
| cifar_resnet_b1 | approximate | m4-max (Metal) | metal/tuned | **43.71** | 43.37-44.18 | 507 | PASS (1.1e-07) | 46.22 | 0.08 |
| cifar_resnet_b1 | exact | m4-max (Metal) | cc/tuned | **1.99** | 1.93-2.07 | 106 | PASS (1.1e-07) | 2.12 | 0.96 |

- **Tuning now adds little over untuned cc on the exact cells.**
  - Batch 64: 0.94 on rog and 0.82 on minix, against 0.67 and 1.02 (a #1611 tail) on October 30.
  - Batch 1: 0.98 on rog and 0.94 on minix.
  - The untuned cc default caught up with the tuned step (gh-ocannl-1630). The tuned cc steps
    themselves are flat on rog and on minix's batch-1 cell. minix's batch-64 cell, 101.42 to
    66.75 ms, is the October 30 #1611 tail gone, not a measured speedup: its p10-p90 is
    66.3-67.2 ms now.
- **Fission is still worth 7% on `cifar_resnet/cuda/tuned/exact`** (21.32 off against 19.82 ms on),
  as on October 30. With fission on, the search was 3.9x cheaper (140 against 546 s).
- **Search costs** are lower than October 30's on most cells (rog batch-64 exact 140 against
  190 s, approximate 518 against 744 s; minix 214 against 247 s and 355 against 410 s). The
  exceptions are the batch-1 GPU exact searches: rog's took 49 against 43 s, and minix's 109
  against 59 s.

## mac-studio (Metal): the window

- **Machine.** Apple M4 Max, macOS 27.2.1. The window `wave-20261119-ocannl-t2008-mac-studio-1`
  ran its ResNet part from 09:20 to 13:07 CET. mac-studio is also the maintainer's desktop.
- **Co-tenants.** The hold recorded 49 GPU co-tenant processes at open and 46 at close, as
  cumulative GPU time; macOS gives no per-window share. Among them were WindowServer and the
  Claude, Codex, VS Code and Edge GPU helpers. The load average was 3.5 at the start, and no cell
  is pinned.
- **The one missing cell is again Metal batch-64 approximate.**
  - Its search pass ran into the 3,600 s cell cap, exactly as on October 30, and was killed. That
    is the one failure of the window and the reason its leg exited 1.
  - The exact search took 3,155 s against 3,276 s on October 30. The cap was not raised.
- **Tuning is worth 5.0x on Metal at batch 64** (397.1 to 79.8 ms) and 13x at batch 1 (550.2 to
  41.8 ms). Metal's untuned default is still not seeded (ahrefs/ocannl#1404).


## Reproducing a sweep

One sweep per cell, from a checkout of PR #2243's head or later with the venv linked and the
fixtures copied in, and the measured tree in another worktree:

```bash
d=<absolute-results-dir>; mkdir -p $d
BENCH_OCANNL_TREE=<tag-worktree> BENCH_KERNEL_TABLE=1 BENCH_RESULTS_DIR=$d \
  BENCH_CELL_LOG_DIR=$d/cells OCANNL_AUTOTUNE_CACHE_DIR=$d/autotune_cache \
  benchmarks/.venv/bin/python benchmarks/orchestrate.py --box <box> --gpu <cuda|hip|metal> \
  --workloads cifar_resnet --profile exact --only ocannl pytorch --cell-timeout 3600 --tuned \
  --skip-cell cifar_resnet/cc/default --skip-cell cifar_resnet/cc/tuned \
  --skip-cell cifar_resnet/<gpu>/default
```

- **Untuned sweeps:** drop `--tuned` and the `--skip-cell`s, and add `tinygrad` to `--only` in
  the approximate regime.
- **Anchor sweeps:** name the `1fc57811` worktree in `BENCH_OCANNL_TREE`.
- **Fission off:** add `--cell-flag cifar_resnet/cuda/tuned:schedule_fission=false`.

**Instrument cost.** The per-kernel table is the tag's own `bench_harness.ml` instrument, since
the runners are the tag's. At the tag it makes each OCANNL cell take 8-25x longer than at
`1fc57811`, for example 574 against 22 s for rog's untuned batch-1 GPU cell in the prep. This is window
time, not step time: the table runs after the timed steps.

The sweep reports, the `results.jsonl` files and the cell logs are not checked in; they stay on
each box under `~/.local/state/gh2008/window/resnet/`.

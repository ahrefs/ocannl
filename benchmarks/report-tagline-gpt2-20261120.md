# The 1.2 tag's tagline: GPT-2-mini against PyTorch and tinygrad, with a same-window October 30 anchor (gh-ocannl-2008)

**Measured 2026-11-20 on rog-nv-linux (CUDA, discrete VRAM), minix-amd-linux (HIP, unified
memory) and mac-studio (Metal, Apple M4 Max, unified memory), each box in its own exclusive
timing window.**

- **Trees.** OCANNL is the `1.2` tag, staging `c3e5975a` (PR #1742, 2026-11-06). The same-window
  anchor is `1fc57811`, the tree the [October 30 report](report-tagline-gpt2-20261030.md)
  measured as its master (1.1.3 plus one test-only merge). Both were rebuilt and re-measured on
  the same box in the same window.
- **Driver.** The driver ran from lukstafi/ocannl-staging PR #2243's branch: `a98c641b` on the
  Linux boxes, `479ac9ab` on mac-studio, with no harness difference between the two that touches
  this matrix. It measured the tag's runners through `BENCH_OCANNL_TREE`, and the anchor arm ran
  `1fc57811`'s own `bench_gpt.exe`. Every run records the SHAs of both trees and of the driver
  (`env.json`).
- **Ratios.** They are *how many times slower OCANNL is*: OCANNL's step time over the other
  framework's, both medians over six repeats of synced per-step p50s.

**The box stack changed on 2026-11-06.** That day all three Linux GPU boxes moved to new kernels,
and rog to NVIDIA 620 (ahrefs/ocannl#2004). These windows ran:
- rog and minix: kernel **7.0.0-43**, not the -42 that #2004 recorded;
- rog: NVIDIA **620.70.06**, CUDA 13.5;
- mac-studio: builds against the **macOS 27.2** SDK (October 30 ran macOS 27.0.1).

Numbers here compare with October's only through the same-window anchor.

## Taglines

"Exact" pairs OCANNL's default numerics with PyTorch pinned to match it; "approximate" pairs
`--ocannl_profile=approximate` with PyTorch's own defaults. Each paragraph quotes PyTorch's best
arm (`torch.compile`) and tinygrad's best arm (BEAM=2) first, with the fallback arms in
parentheses. "Since October 30" compares the tag with `1fc57811`, both measured in this window
on the same box, so setup drift cancels out of the change.

**rog-nv-linux, CUDA, RTX 5070 Ti Laptop (discrete VRAM).**
GPT-2-mini is **2.18x slower than PyTorch** (`torch.compile`; 1.17x slower than eager) and **1.37x slower than tinygrad** (BEAM=2; 2.48x *faster* than its JIT), exact f32 on both sides.
In approximate numerics on both sides (tf32, SDPA) it is 3.74x slower than PyTorch and 1.33x slower than tinygrad.
Since October 30: OCANNL's exact step went from 3.279 to 2.151 ms, **1.52x faster** (same window).

**minix-amd-linux, HIP, Radeon 8060S (gfx1151, unified memory).**
GPT-2-mini is **1.29x slower than PyTorch** (`torch.compile`; 1.15x *faster* than eager) and **1.27x slower than tinygrad** (BEAM=2; 1.40x faster than its JIT), exact.
In approximate numerics it is 1.74x slower than PyTorch and 1.78x slower than tinygrad.
Since October 30: OCANNL's exact step went from 4.311 to 4.082 ms, **1.06x faster** (same window). The approximate step reads 5.697 ms against October 30's 4.409 ms: **1.29x slower**, across windows (see the minix section).

**mac-studio, Metal, Apple M4 Max (unified memory).**
GPT-2-mini is **2.73x slower than PyTorch** (`torch.compile` on MPS; 1.10x *faster* than eager) and **3.30x slower than tinygrad** (BEAM=2; 1.94x slower than its JIT), exact.
In approximate numerics it is 2.75x slower than PyTorch and 2.89x slower than tinygrad.
Since October 30: OCANNL's exact step went from 6.905 to 6.739 ms (**1.02x faster**, same window, about the size of the window's spread).

**tuf-amd-linux, HIP, Radeon RX 7700S (gfx1102, discrete VRAM): not run.** The box was not
among this wave's measurement legs and no window was granted for it. So the tag has no
discrete-memory HIP number; October 30's tuf row describes 1.1.3.

## Headline set

"x" is how many times slower OCANNL is. PyTorch means `torch.compile`: exact-pinned in the exact
column, torch defaults in the approximate one. tinygrad means BEAM=2 in both columns.

| box | device, memory | OCANNL exact ms | exact: vs PyTorch / vs tinygrad | approximate: vs PyTorch / vs tinygrad | `1fc57811` in this window: exact vs PyTorch / vs tinygrad | OCANNL since October 30 (same window) |
|---|---|---|---|---|---|---|
| rog-nv-linux | CUDA RTX 5070 Ti Laptop, discrete | 2.151 | **2.18x / 1.37x** | 3.74x / 1.33x | 3.33x / 2.10x | 1.52x faster (3.279 to 2.151 ms) |
| minix-amd-linux | HIP gfx1151, unified | 4.082 | **1.29x / 1.27x** | 1.74x / 1.78x | 1.36x / 1.34x | 1.06x faster (4.311 to 4.082 ms) |
| mac-studio | Metal M4 Max, unified | 6.739 | **2.73x / 3.30x** | 2.75x / 2.89x | 2.80x / 3.38x | 1.02x faster (6.905 to 6.739 ms) |
| tuf-amd-linux | HIP gfx1102, discrete | not run | - | - | - | - |

**The anchor did not move across the discontinuity on the Linux boxes.** `1fc57811` reads 3.279
ms on rog against 3.283 ms on October 30, and 4.311 ms on minix against 4.310 ms. PyTorch and
tinygrad moved by under 4% on both boxes: `torch.compile` exact went from 0.993 to 0.985 ms on
rog and from 3.178 to 3.175 ms on minix, and BEAM=2 went from 1.511 to 1.565 ms and from 3.101 to
3.208 ms. On these two boxes this window's ratios line up with October 30's too. On mac-studio the anchor reads 6.905 ms against 7.002 ms on October 30 (1.4% faster), with
`torch.compile` exact at 2.469 against 2.483 ms and BEAM=2 at 2.044 against 2.065 ms. The macOS
27.0.1 to 27.2.1 move left this box's ratios where they were too.

## Which 1.2 changes moved these cells

The tag and the anchor are two points 222 first-parent merges apart. Every OCANNL GPT-2 arm is
tuned, and the matrix has no per-change arm, so it **cannot tell apart** the changes the issue
lists:

- **Cannot move this table:**
  - gh-ocannl-1630 changes untuned cc convs; GPT-2 has none (the
    [ResNet report](report-gh1161-resnet-20261120.md) shows that change).
  - gh-ocannl-1866/1975 act on reduced-precision storage, and every cell here is f32.
- **Moved the CUDA and HIP exact steps jointly:** gh-ocannl-1697 (seeded calibration on CUDA and
  HIP), gh-ocannl-1661 (the partitioned staging copy) and gh-ocannl-1736 (pad-masked registers).
  - These, plus whatever else landed in the 222 merges, account for rog's 1.52x and minix's
    1.06x. This window does not split them.
  - gh-ocannl-1661's own site measurement was GPT-2's output projection on rog, so it is a
    plausible contributor there, but no window here isolates it. 
  - On Metal none of the three applies: seeded calibration is CUDA/HIP only, and the staging copy
    and the pad-masked registers are GPU-staging and register changes measured on CUDA and HIP.
    The Metal step moved by 1.02x, which is about as small as anything this window can resolve.
- **Moved nothing by itself:** `Schedule_cache.entry_version` 23 only forces a re-search, and both
  trees searched from cold caches here.

## Method

The [October 30 report's method](report-tagline-gpt2-20261030.md#method) applies unchanged,
except for the trees.

- **Driver.** [`gh1181_cells.py`](gh1181_cells.py) ran with `--repeats 7 --search-once
  --cell-timeout 3600 --total-timeout 18000`; rog's cell cap was 1,800 s, as on October 30.
  - The trees were set with `BENCH_OCANNL_TREE=<tag worktree>` and `--master-commit
    c3e5975aa4fb811ffa663e2bee8404a54bbb24b1 --aug <1fc57811 worktree> --anchor-commit
    1fc57811b09e8f534174c2e7b443da2da906c3c2`.
  - The "master" arms are the tag. The arm still named `ocannl-aug-exact` is `1fc57811`.
  - The master arm's cells run in the driver's `benchmarks/`, so they read its `ocannl_config`.
    The file is the same blob at the tag, the anchor and the driver.
- **Cold caches.** Each tuned arm searched once, in repeat 0, into an empty cache of its own
  under the run's `--out`. Nothing came from `benchmarks/autotune_cache/` or a prep smoke.
  - Every timing came from a fresh process that replayed that cache: only `REPLAY` rows after a
    `SEARCHED` pass are quoted.
  - As on October 30, the CUDA driver's `~/.nv/ComputeCache` and Metal's shader cache were left
    alone, so search costs on those two backends are lower bounds.
- **Gates.** The October 30 report's: parity in every repeat against `pytorch/cpu/eager` exact
  (2e-3 exact, 1e-2 approximate), plus the regime, backend, provenance and tensorization gates.
  Every quoted row passed.
- **Windows.** One exclusive measurement reservation per box,
  `wave-20261119-ocannl-t2008-<box>-1`, run under `fleet-worker.sh execution leg --deadline` on
  Linux and `execution hold` on mac-studio.
  - Before it, a correctness leg per box built both trees, ran `benchmarks/check-box.sh` (READY
    on every box) and smoke-ran every arm once.
  - The fixture was copied from each box's main checkout, never regenerated:
    `gpt2_mini.safetensors`, content digest `c322a00b…8de8`, October's bytes, MATCH on every box.
  - The [ResNet cells](report-gh1161-resnet-20261120.md) followed in the same window on each box.

## rog-nv-linux: CUDA, RTX 5070 Ti Laptop (12 GiB discrete)

Intel Core Ultra 9 275HX, native Ubuntu, kernel 7.0.0-43, NVIDIA 620.70.06 (CUDA 13.5); torch
2.14.0+cu130, tinygrad 0.14.0 (device `CUDA`), Python 3.14.4. Window
`wave-20261119-ocannl-t2008-rog-nv-linux-1`, opened 00:24 CET; its GPT-2 part took about 74
minutes, 30 of them one hung BEAM=2 cell. No GPU co-tenant at open or at close.

| arm | regime | median step p50 ms | min-max | n | search / compile s | parity (worst rel) | provenance | tensorization |
|---|---|---|---|---|---|---|---|---|
| OCANNL tuned, 1.2 tag, exact | exact | **2.151** | 2.126-2.161 | 6 | 259 (search, once) | PASS (6.8e-08) | REPLAY | NOT-REQUESTED |
| OCANNL tuned, 1.2 tag, approximate | approximate | **2.089** | 2.079-2.105 | 6 | 433 (search, once) | PASS (3.9e-06) | REPLAY | TENSORIZED |
| OCANNL tuned, `1fc57811` (October 30), exact | exact | **3.279** | 3.267-3.287 | 6 | 300 (search, once) | PASS (1.3e-07) | REPLAY | NOT-REQUESTED |
| PyTorch eager, exact-pinned | exact | **1.846** | 1.812-1.859 | 6 | 0.5 | PASS (1.3e-07) | no search | - |
| PyTorch `torch.compile`, exact-pinned | exact | **0.985** | 0.984-0.988 | 6 | 2.7 | PASS (6.7e-08) | SAME-PROCESS | - |
| PyTorch eager, torch defaults | approximate | **0.892** | 0.891-0.895 | 6 | 0.5 | PASS (2.0e-06) | no search | - |
| PyTorch `torch.compile`, torch defaults | approximate | **0.559** | 0.558-0.560 | 6 | 2.1 | PASS (2.7e-06) | SAME-PROCESS | - |
| tinygrad JIT | approximate (no exact pin) | **5.344** | 5.195-5.351 | 6 | 1.2 | PASS (1.3e-07) | no search | - |
| tinygrad BEAM=2 | approximate (no exact pin) | **1.565** | 1.463-1.670 | **5** | 197.1-334.1 | PASS (5.4e-07) | SAME-PROCESS | - |

| OCANNL is ... slower | than `torch.compile` | than torch eager | than tinygrad BEAM=2 | than tinygrad JIT |
|---|---|---|---|---|
| 1.2 tag, exact, vs exact-pinned torch | **2.18x** | 1.17x | **1.37x** | 0.40x (2.48x faster) |
| 1.2 tag, approximate, vs torch defaults | **3.74x** | 2.34x | **1.33x** | 0.39x (2.56x faster) |
| `1fc57811`, exact, vs exact-pinned torch | 3.33x | 1.78x | 2.10x | 0.61x |
| *October 30 report* | *3.31x* | *1.77x* | *2.17x* | *0.61x* |

- **The exact step gained 1.52x (3.279 to 2.151 ms), and its search was cheaper**, at 259 s
  against the anchor's 300 s.
- **The approximate profile now gains only 3% over exact**: 2.089 against 2.151 ms.
  - On October 30 it gained 28% (2.364 against 3.283 ms). Exact caught up, and approximate
    improved too (2.364 to 2.089 ms, 1.13x).
  - That 1.13x is across windows: the matrix has no approximate anchor arm.
- **The tinygrad BEAM=2 arm has 5 repeats, not 6.**
  - Its repeat-3 cell spun until the 1,800 s cap and was killed. This is the rog BEAM=2 hang the
    October reports also recorded.
  - It was the window's only failure, and the reason the leg exited 1.
  - The other five finished in 197-334 s. Their median is quoted; every other arm has six repeats.

## minix-amd-linux: HIP, Radeon 8060S (gfx1151, unified memory)

AMD Ryzen AI MAX+ 395, unified memory with a 64 GiB GPU carve-out, Ubuntu kernel 7.0.0-43; system
ROCm HIP (hipcc 7.1); torch 2.14.1+rocm7.2 (HIP 7.2.53211), tinygrad 0.14.0 (device `AMD`),
Python 3.14.4. Window `wave-20261119-ocannl-t2008-minix-amd-linux-1`, opened 05:01 CET; the GPT-2
part took about 37 minutes. No GPU co-tenant at open or at close. Every one of its 54 timing rows
passed every gate.

| arm | regime | median step p50 ms | min-max | n | search / compile s | parity (worst rel) | provenance | tensorization |
|---|---|---|---|---|---|---|---|---|
| OCANNL tuned, 1.2 tag, exact | exact | **4.082** | 4.074-4.104 | 6 | 244 (search, once) | PASS (1.3e-07) | REPLAY | NOT-REQUESTED |
| OCANNL tuned, 1.2 tag, approximate | approximate | **5.697** | 5.682-5.766 | 6 | 396 (search, once) | PASS (6.8e-08) | REPLAY | NOT-REQUESTED |
| OCANNL tuned, `1fc57811` (October 30), exact | exact | **4.311** | 4.306-4.330 | 6 | 389 (search, once) | PASS (1.3e-07) | REPLAY | NOT-REQUESTED |
| PyTorch eager, exact-pinned | exact | **4.691** | 4.673-4.720 | 6 | 0.1 | PASS (6.7e-08) | no search | - |
| PyTorch `torch.compile`, exact-pinned | exact | **3.175** | 3.166-3.184 | 6 | 3.2-3.3 | PASS (6.7e-08) | SAME-PROCESS | - |
| PyTorch eager, torch defaults | approximate | **4.137** | 4.123-4.157 | 6 | 0.1 | PASS (6.7e-08) | no search | - |
| PyTorch `torch.compile`, torch defaults | approximate | **3.272** | 3.251-3.298 | 6 | 2.6-2.7 | PASS (1.3e-07) | SAME-PROCESS | - |
| tinygrad JIT | approximate (no exact pin) | **5.714** | 5.704-5.797 | 6 | 1.6 | PASS (1.3e-07) | no search | - |
| tinygrad BEAM=2 | approximate (no exact pin) | **3.208** | 2.947-3.440 | 6 | 92.4-163.4 | PASS (5.4e-07) | SAME-PROCESS | - |

| OCANNL is ... slower | than `torch.compile` | than torch eager | than tinygrad BEAM=2 | than tinygrad JIT |
|---|---|---|---|---|
| 1.2 tag, exact, vs exact-pinned torch | **1.29x** | 0.87x (1.15x faster) | **1.27x** | 0.71x (1.40x faster) |
| 1.2 tag, approximate, vs torch defaults | **1.74x** | 1.38x | **1.78x** | 1.00x |
| `1fc57811`, exact, vs exact-pinned torch | 1.36x | 0.92x | 1.34x | 0.75x |
| *October 30 report* | *1.36x* | *0.92x* | *1.39x* | *0.75x* |

- **The exact step gained 1.06x (4.311 to 4.082 ms)**, with a cheaper search (244 against 389 s).
- **The approximate step is slower than on October 30: 5.697 against 4.409 ms, 1.29x.**
  - This compares two windows, since the anchor arm is exact only. But the anchor and the other
    frameworks did not move between them, and the shared-slot prep smoke read about 6.0 ms too.
  - So the slowdown is most likely in the code between `1fc57811` and the tag, but this window
    does not prove it. A same-window approximate anchor and a bisect are proposed as a follow-up.
  - Approximate is now 1.40x slower than exact on this box; on October 30 it was 1.02x slower.
- **The gh-ocannl-1697 batch-8 base move (15.64 to 13.48 ms).** The issue asked to attribute this
  `gpt2_mini_train` move between two of that issue's A/B windows.
  - The coordinator dropped the attribution cell from this window as a separate question.
  - The prep's dry run *indicates* code, not window drift. It re-measured the two A/B bases,
    `9cf1d185` and `cf9adedd`, side by side in one shared slot at a 30 s tuner budget: 15.66
    against 13.84 ms, paired ratio 0.884 in 2 of 2 rounds.
  - That puts the move in the 120 first-parent merges between the two bases. The run is not
    timing-grade, and the merge is not bisected; that is proposed as a follow-up.

## mac-studio: Metal, Apple M4 Max (unified memory)

Apple M4 Max (40-core GPU), **macOS 27.2.1** (SDK 27.2); torch 2.13.0 (`mps`), tinygrad 0.13.0
(device `METAL`), Python 3.12.14.
- Window `wave-20261119-ocannl-t2008-mac-studio-1`, opened 08:50 CET. The GPT-2 part took about
  30 minutes.
- **Co-tenants.** mac-studio is also the maintainer's desktop, and no cell is pinned. At the
  window's start the load average was 3.5. The hold recorded 49 GPU co-tenant processes at open,
  as cumulative GPU time; macOS gives no per-window share. Among them were WindowServer and the
  Claude, Codex, VS Code and Edge GPU helpers.
- Every one of its 54 timing rows passed every gate.

| arm | regime | median step p50 ms | min-max | n | search / compile s | parity (worst rel) | provenance | tensorization |
|---|---|---|---|---|---|---|---|---|
| OCANNL tuned, 1.2 tag, exact | exact | **6.739** | 6.671-6.770 | 6 | 448 (search, once) | PASS (8.7e-07) | REPLAY | TENSORIZED |
| OCANNL tuned, 1.2 tag, approximate | approximate | **5.913** | 5.898-5.923 | 6 | 324 (search, once) | PASS (8.1e-07) | REPLAY | TENSORIZED |
| OCANNL tuned, `1fc57811` (October 30), exact | exact | **6.905** | 6.836-6.972 | 6 | 494 (search, once) | PASS (8.7e-07) | REPLAY | TENSORIZED |
| PyTorch eager, exact-pinned | exact | **7.433** | 7.226-7.532 | 6 | 0.0-0.4 | PASS (1.3e-07) | no search | - |
| PyTorch `torch.compile`, exact-pinned | exact | **2.469** | 2.441-2.494 | 6 | 1.4-1.5 | PASS (6.7e-08) | SAME-PROCESS | - |
| PyTorch eager, torch defaults | approximate | **3.733** | 3.628-3.753 | 6 | 0.1-0.2 | PASS (1.3e-07) | no search | - |
| PyTorch `torch.compile`, torch defaults | approximate | **2.148** | 2.117-2.180 | 6 | 1.1-1.2 | PASS (1.3e-07) | SAME-PROCESS | - |
| tinygrad JIT | approximate (no exact pin) | **3.481** | 3.164-3.672 | 6 | 0.8 | PASS (1.3e-07) | no search | - |
| tinygrad BEAM=2 | approximate (no exact pin) | **2.044** | 1.886-2.146 | 6 | 51.1-63.3 | PASS (8.0e-07) | SAME-PROCESS | - |

| OCANNL is ... slower | than `torch.compile` | than torch eager | than tinygrad BEAM=2 | than tinygrad JIT |
|---|---|---|---|---|
| 1.2 tag, exact, vs exact-pinned torch | **2.73x** | 0.91x (1.10x faster) | **3.30x** | 1.94x |
| 1.2 tag, approximate, vs torch defaults | **2.75x** | 1.58x | **2.89x** | 1.70x |
| `1fc57811`, exact, vs exact-pinned torch | 2.80x | 0.93x | 3.38x | 1.98x |
| *October 30 report* | *2.82x* | *0.95x* | *3.39x* | *1.91x* |

- **The Metal step barely moved: 6.905 to 6.739 ms, 1.02x.** The tag's search took 448 s, against
  the anchor's 494 s.
- **The approximate arm reads 5.913 ms against October 30's 5.693 ms, 4% slower, across windows.**
  That is within what this box's desktop load moves between windows, so it is not read as a
  change. It is still 1.14x faster than exact.
- **The macOS update did not shift this box this time.** October 30's PyTorch and anchor figures
  reproduce within 1.4% under macOS 27.2.1. October 30 had seen PyTorch 1.2-1.3x slower after the
  26.6.2 to 27.0.1 update.
- All three OCANNL arms ship tensorized (simdgroup-matrix) artifacts, as on October 30.

## Reproducing

From a staging checkout at PR #2243's head or later, with the tag and `1fc57811` in two more
worktrees, both at their commits with `bench_gpt.exe` built, the venv linked at
`benchmarks/.venv` and the fixture copied in:

```bash
cd benchmarks
BENCH_OCANNL_TREE=<tag-worktree> .venv/bin/python gh1181_cells.py --backend <cuda|hip|metal> \
  --box <box> --master-commit c3e5975aa4fb811ffa663e2bee8404a54bbb24b1 \
  --aug <anchor-worktree> --anchor-commit 1fc57811b09e8f534174c2e7b443da2da906c3c2 \
  --out <out-dir> --workloads gpt2_mini --repeats 7 --search-once \
  --cell-timeout 3600 --total-timeout 18000
.venv/bin/python gh1181_cells.py --summarize --out <out-dir>
```

The per-box `summary.md`, `timings.jsonl` and `env.json` of these windows are not checked in; they
stay on each box under `~/.local/state/gh2008/window/gpt2/`. The tables above are those files'
quotable rows.

# gh-ocannl-1181 re-run: GPT-2-mini against PyTorch and tinygrad on four boxes, with a same-window October anchor

**Measured 2026-10-30 on rog-nv-linux (CUDA), minix-amd-linux (HIP, unified memory),
tuf-amd-linux (HIP, discrete VRAM) and mac-studio (Metal), each box in its own exclusive timing
window.** OCANNL is staging `1fc57811` (master `35c7c043` plus the driver's `--anchor-commit`
flag, a benchmark-only change) against the October tagline commit `a723dc5b`, both rebuilt and
re-measured on the same box in the same window. This is the
[October 3–4 report](report-tagline-gpt2.md)'s protocol unchanged, with its master as the new
anchor in place of the August commit. The ratios are *how many times slower OCANNL is*: OCANNL's step time over the other
framework's, both medians over six repeats of synced per-step p50s. Every one of the 245
timing rows across the four windows passed every gate and is quoted.

## Taglines

"Exact" pairs OCANNL's default numerics with PyTorch pinned to match it; "approximate" pairs
`--ocannl_profile=approximate` with PyTorch's own defaults. Each paragraph quotes PyTorch's best
arm (`torch.compile`) and tinygrad's best arm (BEAM=2) first, with the fallback arms in
parentheses. "October" is `a723dc5b` measured in this window on the same box, against this
window's PyTorch and tinygrad, so setup drift cancels out of the change.

**rog-nv-linux, CUDA, RTX 5070 Ti Laptop (discrete VRAM).**
GPT-2-mini is **3.3x slower than PyTorch** (`torch.compile`; 1.8x slower than eager) and **2.2x slower than tinygrad** (BEAM=2; 1.6x *faster* than its JIT), exact f32 on both sides.
In approximate numerics on both sides (tf32, SDPA) it is 4.2x slower than PyTorch and 1.6x slower than tinygrad; in October those were 8.0x and 2.8x.
October: the October commit re-run in this window gives 3.5x and 2.3x; OCANNL's exact step went from 3.52 to 3.28 ms (1.07x), and the approximate one from 4.49 ms (October's reading) to 2.36 ms (1.90x).

**mac-studio, Metal, Apple M4 Max (unified memory).**
GPT-2-mini is **2.8x slower than PyTorch** (`torch.compile` on MPS; 1.05x *faster* than eager) and **3.4x slower than tinygrad** (BEAM=2; 1.9x slower than its JIT), exact.
In approximate numerics it is 2.6x slower than PyTorch and 2.8x slower than tinygrad.
October: the October commit re-run in this window gives 3.9x and 4.7x; OCANNL's step went from 9.63 to 7.00 ms (1.38x). The PyTorch ratio flatters the change: PyTorch on this box reads 1.2-1.3x slower than on October 3-4, and tinygrad does not (see the mac-studio section).

**tuf-amd-linux, HIP, Radeon RX 7700S (gfx1102, discrete VRAM).**
For the first time on this box there is a quotable master figure: every master timing process replayed its whole cache. GPT-2-mini is **1.11x slower than PyTorch** (`torch.compile`; 1.33x *faster* than eager) and **1.36x slower than tinygrad** (BEAM=2; 1.12x faster than its JIT), exact.
In approximate numerics it is 1.19x slower than PyTorch and 1.45x slower than tinygrad.
October: no quotable change, for two reasons. October's master figure on this box was not quotable, and the October commit's tuner still re-searches on every replay here, so it was left out of this window (the tuf section has the details).

**minix-amd-linux, HIP, Radeon 8060S (gfx1151, unified memory).**
For the first time there is a PyTorch figure on this box: the torch 2.14.1+rocm7.2 wheel runs here, where 2.13.0+rocm7.1 segfaulted. GPT-2-mini is **1.36x slower than PyTorch** (`torch.compile`; 1.09x *faster* than eager) and **1.39x slower than tinygrad** (BEAM=2; 1.33x faster than its JIT), exact.
In approximate numerics it is 1.35x slower than PyTorch and 1.42x slower than tinygrad.
October: the October commit re-run in this window gives 1.52x slower than tinygrad BEAM=2. OCANNL's step went from 4.70 to 4.31 ms (1.09x).

## Headline set

"x" is how many times slower OCANNL is. PyTorch means `torch.compile`: exact-pinned in the exact
column, torch defaults in the approximate one. tinygrad means BEAM=2 in both columns.

| box | device, memory | OCANNL exact ms | exact: vs PyTorch / vs tinygrad | approximate: vs PyTorch / vs tinygrad | October report (Oct 3-4): exact vs PyTorch / vs tinygrad | OCANNL since October (same window) |
|---|---|---|---|---|---|---|
| rog-nv-linux | CUDA RTX 5070 Ti Laptop, discrete | 3.28 | **3.31x / 2.17x** | 4.24x / 1.56x | 3.41x / 2.18x | 1.07x faster (3.52 to 3.28 ms) |
| mac-studio | Metal M4 Max, unified | 7.00 | **2.82x / 3.39x** | 2.60x / 2.76x | 4.40x / 4.04x | 1.38x faster (9.63 to 7.00 ms) |
| tuf-amd-linux | HIP gfx1102, discrete | 5.49 | **1.11x / 1.36x** | 1.19x / 1.45x | not quotable | not measured (anchor not cacheable here) |
| minix-amd-linux | HIP gfx1151, unified | 4.31 | **1.36x / 1.39x** | 1.35x / 1.42x | n/a (no PyTorch) / 1.54x | 1.09x faster (4.70 to 4.31 ms) |

The "since October" column compares the two commits measured in the same window, so it
excludes machine and framework drift. On rog and minix the October commit reads within 1.2% of
its own October 3-4 figure (rog 3.504 then, 3.520 now; minix 4.757 then, 4.702 now), so this
window's ratios line up with the October report's too. On mac-studio it does not; see that
section.

## What changed in the setup since October 3-4

- **Anchor.** The October tagline commit `a723dc5b` replaces the August `7014dc44`;
  `gh1181_cells.py --anchor-commit` names it, and `env.json` records it. The arm keeps its name,
  `ocannl-aug-exact`.
- **Boxes.**
  - rog and tuf: kernel 7.0.0-38 (was 7.0.0-31).
  - minix: kernel 7.0.0-38 (was 7.0.0-34) and torch **2.14.1+rocm7.2** (was 2.13.0+rocm7.1, which
    segfaulted on the first host-to-device copy).
  - mac-studio: **macOS 27.0.1** (was 26.6.2).
  - Otherwise every framework version is October's: rog torch 2.14.0+cu130, tuf torch
    2.13.0+rocm7.1, mac-studio torch 2.13.0 and tinygrad 0.13.0, tinygrad 0.14.0 on the Linux
    boxes, and Python 3.14.4 on Linux, 3.12.14 on the Mac.
  - rog now has `python3.14-dev`, so the October `C_INCLUDE_PATH` workaround for inductor is gone.
- **Fixture.** The same bytes on every box as in October: `gpt2_mini.safetensors`, content digest
  `c322a00b72df143612eafafedf7c468e9c05f3e4bc1e7eb6d77ada651ef98de8` (raw sha256
  `89702197…d49f`, 13,871,384 bytes), copied from each box's main checkout, never regenerated.
  The measured tree's `fixtures/DIGESTS.txt` predates gh-ocannl-1357's re-record, so the rows
  stamp the origin `m4-max,tuf`. Since that PR, rog-nv and minix record the same content-v1
  digest.
- **rog cell cap.** On rog the cell cap was 1,800 s rather than 3,600 s. The prep smoke's BEAM=2
  cell spun until the cap, the flake October also hit once. No cell reached either cap in this
  window.

## Method

The [October report's method](report-tagline-gpt2.md#method) applies unchanged, except where
noted below.

- **Driver.** [`gh1181_cells.py`](gh1181_cells.py) ran with `--repeats 7 --search-once
  --cell-timeout 3600 --total-timeout 18000` (1,800 s cell cap on rog).
  - Seven repeats, with repeat 0 a discarded warm-up.
  - The nine arms run one process (pair) at a time. Their order rotates by one arm per repeat and
    is reversed on odd repeats.
  - Linux cells are pinned with `taskset -c 0-15`.
- **Tuned OCANNL.** Two-pass, gh-ocannl-644:
  - Each tuned arm searches once, in repeat 0, into a fresh cache of its own.
  - Every timing comes from a fresh process that replays that cache. Only rows whose
    `search_provenance` is `REPLAY`, after a `SEARCHED` pass, are quoted.
  - The search cost is in the `search / compile s` column.
- **PyTorch and tinygrad.**
  - PyTorch eager and `torch.compile` run under `--regime exact` and under `--regime approximate`
    (torch's defaults).
  - tinygrad runs `--jit 1`, and BEAM=2 with `PARALLEL=0`.
  - Each arm gets fresh compile caches per repeat.
- **Gates.**
  - Parity runs in every repeat, against `pytorch/cpu/eager` exact: 2e-3 for exact rows, 1e-2 for
    approximate ones.
  - The regime, backend and provenance gates are the October report's.
  - Every row passed; the worst drift is 2.8e-06.
- **Windows.** One exclusive measurement reservation per box, `gh1181r2-<box>-measure-2`, run
  under `fleet-worker.sh execution hold`.
  - Before it, a correctness reservation per box (`gh1181r2-<box>-prep-1`) built both trees and
    smoke-ran every arm once.
  - The `measure-1` reservations were concluded `cancelled`: a launcher bug shifted the driver's
    arguments, it refused at argument parsing, and nothing ran.
  - The four windows started together at 12:51 CET. GPT-2 took 2,038-3,439 s per box, then the
    [ResNet re-run](report-gh1161-resnet-20261030.md) followed in the same windows on rog, minix
    and mac-studio.
  - The holds recorded no GPU co-tenant on the three Linux boxes, at open or at close.

## rog-nv-linux: CUDA, RTX 5070 Ti Laptop (12 GiB discrete)

Intel Core Ultra 9 275HX, native Ubuntu, kernel 7.0.0-38; torch 2.14.0+cu130, tinygrad 0.14.0
(device `CUDA`), Python 3.14.4. Window `gh1181r2-rog-nv-linux-measure-2`; the GPT-2 part took
2,784 s.

| arm | regime | median step p50 ms | min-max | n | search / compile s | parity (worst rel) | provenance | tensorization |
|---|---|---|---|---|---|---|---|---|
| OCANNL tuned, master, exact | exact | **3.283** | 3.278-3.288 | 6 | 297 (search, once) | PASS (1.3e-07) | REPLAY | NOT-REQUESTED |
| OCANNL tuned, master, approximate | approximate | **2.364** | 2.362-2.366 | 6 | 577 (search, once) | PASS (2.8e-06) | REPLAY | TENSORIZED |
| OCANNL tuned, `a723dc5b` (October), exact | exact | **3.520** | 3.515-3.530 | 6 | 242 (search, once) | PASS (8.7e-07) | REPLAY | NOT-REQUESTED |
| PyTorch eager, exact-pinned | exact | **1.850** | 1.793-2.000 | 6 | 0.3-3.0 | PASS (1.3e-07) | no search | - |
| PyTorch `torch.compile`, exact-pinned | exact | **0.993** | 0.987-1.037 | 6 | 2.6-2.6 | PASS (6.7e-08) | SAME-PROCESS | - |
| PyTorch eager, torch defaults | approximate | **0.892** | 0.891-0.892 | 6 | 0.3-6.0 | PASS (2.0e-06) | no search | - |
| PyTorch `torch.compile`, torch defaults | approximate | **0.558** | 0.558-0.559 | 6 | 2.0-2.0 | PASS (2.7e-06) | SAME-PROCESS | - |
| tinygrad JIT | approximate (no exact pin) | **5.349** | 5.202-5.354 | 6 | 1.2-1.8 | PASS (1.3e-07) | no search | - |
| tinygrad BEAM=2 | approximate (no exact pin) | **1.511** | 1.482-1.645 | 6 | 169.4-233.2 | PASS (5.4e-07) | SAME-PROCESS | - |

| OCANNL is ... slower | than `torch.compile` | than torch eager | than tinygrad BEAM=2 | than tinygrad JIT |
|---|---|---|---|---|
| exact, vs exact-pinned torch | **3.31x** | 1.77x | **2.17x** | 0.61x (1.63x faster) |
| approximate, vs torch defaults | **4.24x** | 2.65x | **1.56x** | 0.44x (2.26x faster) |
| October `a723dc5b`, exact, vs exact-pinned torch | 3.54x | 1.90x | 2.33x | 0.66x |
| *October report (Oct 3-4)* | *3.41x* | *1.91x* | *2.18x* | *0.65x* |

- **The approximate profile now pays off on CUDA.**
  - In October it made OCANNL slower: 4.49 ms against 3.50 ms exact, although its winner shipped
    tf32 mma.
  - It now reads 2.36 ms against 3.28 ms exact, a 1.90x gain on October's approximate figure.
  - The October commit's approximate arm is not in this matrix (the anchor arm is exact only), so
    the 1.90x compares two windows rather than one.
- **The exact step gained 7% (3.52 to 3.28 ms).** PyTorch and tinygrad did not move: torch.compile
  1.03 then, 0.99 ms now; BEAM=2 1.61 then, 1.51 ms now.
- The master searches took 297 s (exact) and 577 s (approximate), against 467 s and 804 s in
  October.
- Every BEAM=2 cell finished, in 169-233 s.

## tuf-amd-linux: HIP, Radeon RX 7700S (gfx1102, discrete 8 GiB VRAM)

AMD Ryzen 7 7435HS laptop; the same GPU drives the laptop's displays. Ubuntu kernel 7.0.0-38;
system ROCm HIP; torch 2.13.0+rocm7.1, tinygrad 0.14.0 (device `AMD`), Python 3.14.4. Window
`gh1181r2-tuf-amd-linux-measure-2`, 3,439 s, run with
`--no-anchor "a723dc5b re-searches on every replay on tuf (prep smoke 2026-10-30)"`.

| arm | regime | median step p50 ms | min-max | n | search / compile s | parity (worst rel) | provenance | tensorization |
|---|---|---|---|---|---|---|---|---|
| OCANNL tuned, master, exact | exact | **5.487** | 5.454-5.527 | 6 | 529 (search, once) | PASS (6.7e-08) | REPLAY | NOT-REQUESTED |
| OCANNL tuned, master, approximate | approximate | **5.874** | 5.833-5.905 | 6 | 692 (search, once) | PASS (6.7e-08) | REPLAY | NOT-REQUESTED |
| PyTorch eager, exact-pinned | exact | **7.271** | 7.235-7.293 | 6 | 0.2-0.2 | PASS (6.7e-08) | no search | - |
| PyTorch `torch.compile`, exact-pinned | exact | **4.958** | 4.877-5.061 | 6 | 6.9-7.1 | PASS (1.3e-07) | SAME-PROCESS | - |
| PyTorch eager, torch defaults | approximate | **7.379** | 7.294-7.434 | 6 | 0.2-0.2 | PASS (6.7e-08) | no search | - |
| PyTorch `torch.compile`, torch defaults | approximate | **4.916** | 4.873-5.019 | 6 | 6.9-7.0 | PASS (1.3e-07) | SAME-PROCESS | - |
| tinygrad JIT | approximate (no exact pin) | **6.144** | 6.078-6.682 | 6 | 2.5-2.5 | PASS (1.3e-07) | no search | - |
| tinygrad BEAM=2 | approximate (no exact pin) | **4.044** | 3.729-5.655 | 6 | 148.1-241.2 | PASS (8.0e-07) | SAME-PROCESS | - |

| OCANNL is ... slower | than `torch.compile` | than torch eager | than tinygrad BEAM=2 | than tinygrad JIT |
|---|---|---|---|---|
| exact, vs exact-pinned torch | **1.11x** | 0.75x (1.33x faster) | **1.36x** | 0.89x (1.12x faster) |
| approximate, vs torch defaults | **1.19x** | 0.80x (1.26x faster) | **1.45x** | 0.96x (1.05x faster) |

- **The master tuner caches every arm here now.**
  - In October every master process on this box re-searched an arm that the tuner refused to
    cache, `Autotune.search_measurements_cacheable` judging its timings contended, so no master
    row was quotable.
  - In this window both master arms replayed cleanly. The replays compiled in 2.0-2.5 s, against
    529-692 s for the searches.
  - This is the condition the brief set for including tuf.
- **The October commit is the one that cannot be cached here.**
  - In the prep smoke, `a723dc5b`'s arm re-searched in every replay: 414-418 s each, three
    passes, never completing its cache.
  - That is the October master's own tuner, so it shows the same veto October reported; the
    August commit, which October anchored on, had no veto.
  - Its rows could never pass the provenance gate, so the window left it out (`--no-anchor`),
    which saved about 20 minutes of re-search per repeat.
  - The October report's master readings on this box (6.685 ms exact, 5.887 ms approximate) were
    marked not quotable, and they remain so. Read only as diagnostics, they put master's exact
    step at 1.22x faster (6.69 to 5.49 ms) and approximate at flat (5.89 to 5.87 ms), across
    windows.
- **BEAM=2 did not wedge the GPU this time.** In October it took three resets. Here all six
  repeats completed. One reading, 5.655 ms (repeat 2), is far from the other five (3.73-4.39 ms);
  the median does not depend on it.
- The approximate profile is slower than exact here (5.87 against 5.49 ms). Neither OCANNL arm
  tensorized on gfx1102.

## minix-amd-linux: HIP, Radeon 8060S (gfx1151, unified memory)

AMD Ryzen AI MAX+ 395, unified memory with a 64 GiB GPU carve-out, Ubuntu kernel 7.0.0-38;
system ROCm HIP; torch **2.14.1+rocm7.2** (HIP 7.2.53211), tinygrad 0.14.0 (device `AMD`), Python
3.14.4. Window `gh1181r2-minix-amd-linux-measure-2`; the GPT-2 part took 2,756 s.

| arm | regime | median step p50 ms | min-max | n | search / compile s | parity (worst rel) | provenance | tensorization |
|---|---|---|---|---|---|---|---|---|
| OCANNL tuned, master, exact | exact | **4.310** | 4.294-4.315 | 6 | 389 (search, once) | PASS (1.3e-07) | REPLAY | NOT-REQUESTED |
| OCANNL tuned, master, approximate | approximate | **4.409** | 4.395-4.422 | 6 | 405 (search, once) | PASS (6.7e-08) | REPLAY | NOT-REQUESTED |
| OCANNL tuned, `a723dc5b` (October), exact | exact | **4.702** | 4.678-4.720 | 6 | 513 (search, once) | PASS (8.7e-07) | REPLAY | NOT-REQUESTED |
| PyTorch eager, exact-pinned | exact | **4.679** | 4.663-4.715 | 6 | 0.1-0.1 | PASS (6.7e-08) | no search | - |
| PyTorch `torch.compile`, exact-pinned | exact | **3.178** | 3.174-3.239 | 6 | 3.1-3.2 | PASS (6.7e-08) | SAME-PROCESS | - |
| PyTorch eager, torch defaults | approximate | **4.133** | 4.124-4.142 | 6 | 0.1-0.1 | PASS (6.7e-08) | no search | - |
| PyTorch `torch.compile`, torch defaults | approximate | **3.269** | 3.252-3.296 | 6 | 2.6-2.7 | PASS (1.3e-07) | SAME-PROCESS | - |
| tinygrad JIT | approximate (no exact pin) | **5.743** | 5.726-5.827 | 6 | 1.6-1.6 | PASS (1.3e-07) | no search | - |
| tinygrad BEAM=2 | approximate (no exact pin) | **3.101** | 3.040-3.807 | 6 | 126.2-177.3 | PASS (5.4e-07) | SAME-PROCESS | - |

| OCANNL is ... slower | than `torch.compile` | than torch eager | than tinygrad BEAM=2 | than tinygrad JIT |
|---|---|---|---|---|
| exact, vs exact-pinned torch | **1.36x** | 0.92x (1.09x faster) | **1.39x** | 0.75x (1.33x faster) |
| approximate, vs torch defaults | **1.35x** | 1.07x | **1.42x** | 0.77x (1.30x faster) |
| October `a723dc5b`, exact, vs exact-pinned torch | 1.48x | 1.00x | 1.52x | 0.82x |
| *October report (Oct 3-4)* | *n/a* | *n/a* | *1.54x* | *0.83x* |

- **PyTorch runs on this box now.** The October report had no PyTorch figure here because
  2.13.0+rocm7.1 segfaulted on its first host-to-device copy. 2.14.1+rocm7.2 ran all 28 torch
  cells of this window.
  - In the same windows, one ResNet cell on that wheel still crashed (exit -11, in the companion
    report). The GPT-2 cells did not.
- OCANNL's exact step gained 1.09x (4.70 to 4.31 ms). tinygrad did not move (BEAM=2 3.09 then,
  3.10 ms now). The minix GPT-2 cells show no sign of the cc tails tracked in ahrefs/ocannl#1611
  (every OCANNL min-max here is within 1%); those tails are a cc-backend effect, and they do show in the
  ResNet report.

## mac-studio: Metal, Apple M4 Max (unified memory)

Apple M4 Max (40-core GPU), **macOS 27.0.1**; torch 2.13.0 (`mps`), tinygrad 0.13.0 (device
`METAL`), Python 3.12.14. Window `gh1181r2-mac-studio-measure-2`; the GPT-2 part took 2,038 s.
mac-studio is also the maintainer's desktop. At the window's start the load average was 3.3, and
the hold recorded WindowServer, the Claude, Edge, VS Code, Codex and Discord GPU helpers as GPU
co-tenants (cumulative GPU time; macOS gives no per-window share). No cell is pinned.

| arm | regime | median step p50 ms | min-max | n | search / compile s | parity (worst rel) | provenance | tensorization |
|---|---|---|---|---|---|---|---|---|
| OCANNL tuned, master, exact | exact | **7.002** | 6.941-7.035 | 6 | 499 (search, once) | PASS (8.7e-07) | REPLAY | TENSORIZED |
| OCANNL tuned, master, approximate | approximate | **5.693** | 5.682-5.714 | 6 | 371 (search, once) | PASS (8.1e-07) | REPLAY | TENSORIZED |
| OCANNL tuned, `a723dc5b` (October), exact | exact | **9.630** | 9.583-9.752 | 6 | 529 (search, once) | PASS (8.1e-07) | REPLAY | TENSORIZED |
| PyTorch eager, exact-pinned | exact | **7.365** | 6.737-7.609 | 6 | 0.1-0.5 | PASS (1.3e-07) | no search | - |
| PyTorch `torch.compile`, exact-pinned | exact | **2.483** | 2.414-2.530 | 6 | 1.5-1.7 | PASS (6.7e-08) | SAME-PROCESS | - |
| PyTorch eager, torch defaults | approximate | **3.769** | 3.611-3.807 | 6 | 0.3-0.3 | PASS (1.3e-07) | no search | - |
| PyTorch `torch.compile`, torch defaults | approximate | **2.191** | 2.025-2.207 | 6 | 1.2-1.3 | PASS (1.3e-07) | SAME-PROCESS | - |
| tinygrad JIT | approximate (no exact pin) | **3.658** | 3.276-3.691 | 6 | 0.8-0.9 | PASS (1.3e-07) | no search | - |
| tinygrad BEAM=2 | approximate (no exact pin) | **2.065** | 1.827-2.151 | 6 | 50.8-77.8 | PASS (8.0e-07) | SAME-PROCESS | - |

| OCANNL is ... slower | than `torch.compile` | than torch eager | than tinygrad BEAM=2 | than tinygrad JIT |
|---|---|---|---|---|
| exact, vs exact-pinned torch | **2.82x** | 0.95x (1.05x faster) | **3.39x** | 1.91x |
| approximate, vs torch defaults | **2.60x** | 1.51x | **2.76x** | 1.56x |
| October `a723dc5b`, exact, vs exact-pinned torch | 3.88x | 1.31x | 4.66x | 2.63x |
| *October report (Oct 3-4)* | *4.40x* | *1.40x* | *4.04x* | *2.27x* |

- **OCANNL gained 1.38x here (9.63 to 7.00 ms, same window), more than on any other box.** The
  approximate arm gained too, 1.31x on October's figure (7.46 to 5.69 ms).
- **Ratios across windows on this box are not clean.**
  - Between October 3-4 and this window, mac-studio moved from macOS 26.6.2 to 27.0.1.
  - The October commit reads 9.63 ms now against 8.33 ms then: 1.16x slower, same code.
  - PyTorch's arms read 1.2-1.3x slower too: torch.compile exact 1.89 then, 2.48 ms now; eager
    exact 5.96 then, 7.37 ms now.
  - tinygrad's did not move: JIT 3.67 then, 3.66 ms now; BEAM=2 2.06 then, 2.07 ms now.
  - So against PyTorch the move from 4.40x to 2.82x overstates OCANNL's progress. The same-window
    anchor ratio (1.38x) and the tinygrad ratio (4.04x to 3.39x) are the comparisons to quote.
  - Neither the OS update nor the desktop load was isolated in this window, so this report does
    not say which of them slowed the October commit and PyTorch.
- All three OCANNL arms ship tensorized (simdgroup-matrix) artifacts here, as in October.

## Found on the way

1. **tuf's master tuner now caches its arms.** October's gh-ocannl-1181 blocker on tuf is gone for
   master: both master arms replay cleanly. The October commit's tuner still does not cache them
   on gfx1102.
2. **minix runs PyTorch since the ROCm 7.2 wheel.** That ends the October "no PyTorch on minix"
   gap for GPT-2. On ResNet the wheel still crashes in one cell (companion report).
3. **mac-studio's environment drifted since October.** It moved to macOS 27.0.1, and both the
   October commit and PyTorch read slower while tinygrad does not. Cross-window Metal comparisons
   need a same-window anchor.
4. **The prep smoke's BEAM=2 hang on rog recurred**: the cell spun at 100% CPU until the 3,600 s
   cap, the same symptom as October's repeat 2. It did not recur in the window, with a 1,800 s
   cap.

## Reproducing

From a staging checkout at `1fc57811` (or later), with the anchor in a second worktree at
`a723dc5b` and the venv linked at `benchmarks/.venv`:

```bash
cd benchmarks
benchmarks/.venv/bin/python gh1181_cells.py --backend <cuda|hip|metal> --box <box> \
  --aug <anchor-worktree> --anchor-commit a723dc5b23a815c670bb4047f0b023c6cbd3b14f \
  --out <out-dir> --workloads gpt2_mini --repeats 7 --search-once \
  --cell-timeout 3600 --total-timeout 18000
benchmarks/.venv/bin/python gh1181_cells.py --summarize --out <out-dir> ...
```

On tuf, add `--no-anchor "<why>"`. The per-box `summary.md`, `timings.jsonl` and `env.json` of
this window are not checked in. The tables above are those files' quotable rows, as the driver
summarized them.

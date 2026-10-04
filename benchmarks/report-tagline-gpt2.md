# gh-ocannl-1181: GPT-2-mini against PyTorch and tinygrad on four boxes, with a same-night August anchor

**Measured 2026-10-03/04 on rog-nv-linux (CUDA), minix-amd-linux (HIP, unified memory),
tuf-amd-linux (HIP, discrete VRAM) and mac-studio (Metal), each box in its own exclusive timing
window.** OCANNL is master `a723dc5b` against the August commit `7014dc44`, both rebuilt and
re-measured on the same box in the same window. The ratios below are *how many times slower OCANNL
is*: step time of OCANNL over step time of the other framework, both medians of synced per-step
p50s over six repeats (four on tuf, whose window ran into its cap).

## Taglines

One paragraph per box. "Exact" pairs OCANNL's default numerics with PyTorch pinned to match it;
"approximate" pairs `--ocannl_profile=approximate` with PyTorch's own defaults. The PyTorch figure
quoted first is its best arm (`torch.compile`), the tinygrad figure its best arm (BEAM=2); the
fallback arms follow in parentheses. "August" is `7014dc44` measured tonight on the same box
against tonight's PyTorch and tinygrad, so setup drift cancels out of it.

**rog-nv-linux, CUDA, RTX 5070 Ti Laptop (discrete VRAM).**
GPT-2-mini is **3.4x slower than PyTorch** (`torch.compile`; 1.9x slower than eager) and **2.2x slower than tinygrad** (BEAM=2; 1.5x *faster* than its JIT), OCANNL and PyTorch both in exact f32.
In approximate numerics on both sides (tf32, SDPA) it is 8.0x slower than PyTorch and 2.8x slower than tinygrad.
August: 6.4x and 4.6x in the August report (WSL2, torch 2.13, tinygrad 0.13); the August commit re-run here tonight gives 6.9x and 4.4x, and OCANNL's own step time halved, 7.09 to 3.50 ms.

**mac-studio, Metal, Apple M4 Max (unified memory).**
GPT-2-mini is **4.4x slower than PyTorch** (`torch.compile` on MPS; 1.4x slower than eager) and **4.0x slower than tinygrad** (BEAM=2; 2.3x slower than its JIT), exact.
In approximate numerics on both sides it is 4.3x slower than PyTorch and 3.6x slower than tinygrad.
August: no August report covers Metal; the August commit re-run here tonight gives 5.8x and 5.3x, and OCANNL's step time fell 10.95 to 8.33 ms (1.31x).

**tuf-amd-linux, HIP, Radeon RX 7700S (gfx1102, discrete VRAM).**
GPT-2-mini is **at most 1.6x slower than PyTorch** (`torch.compile`; about level with eager) and **at most 1.6x slower than tinygrad** (BEAM=2; 1.3x slower than its JIT), exact: those are the August commit's clean figures, and master is faster.
No clean master figure exists on this box: every master timing process re-searched a tuner arm the tuner refuses to cache here, which the protocol does not quote (the tuf section has the diagnostic readings and why).
August: the August commit re-run here tonight is that 1.59x and 1.55x; there is no August report for this box.

**minix-amd-linux, HIP, Radeon 8060S (gfx1151, unified memory).**
No PyTorch figure: the box's PyTorch ROCm wheel segfaults on this GPU. GPT-2-mini is **1.5x slower than tinygrad** (BEAM=2; 1.2x *faster* than its JIT), exact.
In approximate numerics it is likewise 1.5x slower than tinygrad BEAM=2 (1.2x faster than its JIT).
August: the August commit re-run here tonight gives 2.6x slower than tinygrad BEAM=2, and OCANNL's step time fell 7.89 to 4.76 ms (1.66x).

## Headline set

The scope comment's headline set is rog, Metal and one HIP row. No HIP box gives a complete clean
row: minix (unified) has clean OCANNL figures and no PyTorch, tuf (discrete) has PyTorch and no
clean OCANNL master figure, only the August commit's, which bounds master from above. Both are
listed. "x" is how many times slower OCANNL is.

| box | device, memory | OCANNL exact ms | exact: vs PyTorch / vs tinygrad | approximate: vs PyTorch / vs tinygrad | August commit, tonight: vs PyTorch / vs tinygrad | OCANNL since August |
|---|---|---|---|---|---|---|
| rog-nv-linux | CUDA RTX 5070 Ti Laptop, discrete | 3.50 | **3.41x / 2.18x** | 8.04x / 2.80x | 6.90x / 4.41x (Aug. report: 6.4x / 4.6x) | 2.02x faster |
| mac-studio | Metal M4 Max, unified | 8.33 | **4.40x / 4.04x** | 4.30x / 3.61x | 5.78x / 5.31x | 1.31x faster |
| tuf-amd-linux | HIP gfx1102, discrete | not quotable | **at most 1.59x / 1.55x** (the August commit's) | not quotable | 1.59x / 1.55x | faster, by an unquotable amount |
| minix-amd-linux | HIP gfx1151, unified | 4.76 | n/a / **1.54x** | n/a / 1.52x | n/a / 2.55x | 1.66x faster |

PyTorch is `torch.compile` (exact-pinned for the exact column, torch defaults for the
approximate one), tinygrad is BEAM=2 in both.

## What the comparison is, and what changed since August

- **Workload.** `gpt2_mini` (`benchmarks/workloads/gpt2_mini.json`): a 4-layer GPT-2-style
  transformer, d_model 256, 8 heads, d_ff 1024, vocabulary 1024, batch 8 x sequence 128 (1,024
  tokens per step), f32, forward plus cross-entropy loss (`mode: infer`). The fixture protocol is
  8 parity steps, 5 warm-up steps, 20 timed steps; the quoted number is the p50 of the 20 steps,
  each synchronized with the device.
- **Fixture.** Every box ran the same bytes: `gpt2_mini.safetensors`, content digest
  `c322a00b72df143612eafafedf7c468e9c05f3e4bc1e7eb6d77ada651ef98de8` (raw sha256
  `89702197c593596901865aceffa2c99cfd9640d046de04b9cb99c95c5dd5d49f`, 13,871,384 bytes), the
  `m4-max,tuf` entry of [`fixtures/DIGESTS.txt`](fixtures/DIGESTS.txt), copied from each box's main
  checkout and never regenerated. This is **not** the fixture August measured
  (`043c1ea8…`, the `rog-nv` raw-v1 entry, which is no longer on rog): same shapes and
  hyperparameters, a different random draw. Step times do not depend on the weights, and the anchor
  ran on tonight's bytes, so each box's before/after compares like with like.
- **Arms** (per box, identical everywhere; [`gh1181_cells.py`](gh1181_cells.py)):
  - OCANNL `tuned` (`BENCH_TUNE=1`, the autotuner's search) at master, exact and approximate, and at
    `7014dc44`, exact. The August runner is the August tree's own `bench_gpt.exe`, built unchanged
    with today's toolchain; it built on all four boxes, so every box has an anchor.
  - PyTorch eager and `torch.compile` (default mode), each under the exact pin (`--regime exact`:
    `float32_matmul_precision("highest")`, cudnn tf32 off, composed attention, which is August's
    arm) and under torch's defaults (`--regime approximate`: `"high"`, SDPA, `cudnn.benchmark`,
    the staging#661 arm).
  - tinygrad `--jit 1`, and BEAM=2 with the fleet's pinned `PARALLEL=0` (`DEFAULT_BEAM_PARALLEL`).
    tinygrad has no exact pin, so both rows stand in the approximate regime and are gated at its
    envelope; they would pass the exact one too (worst drift 8.7e-07).
- **Gates.** Parity is `orchestrate.parity_check` against `pytorch/cpu/eager` exact, run in every
  repeat: 2e-3 for exact rows, 1e-2 for approximate ones. Every quoted row passed, and none drifted
  beyond 2.8e-06. A row is quoted only if it and its search pass passed parity, its runner reports
  the regime it was dispatched in (`regime_check`), it ran on the backend the box was asked for
  (OCANNL's resolved `Context.backend_name`, torch's `cuda` / `cuda(hip)` / `mps`, tinygrad's
  device), and its provenance is what the arm promises: a tuned timing comes from a process that
  replayed every searched arm (`search_provenance == "REPLAY"`, after a `SEARCHED` pass), a
  compiled or BEAM row searched in its own process (`SAME-PROCESS`), eager and JIT rows searched
  nothing.

**Setup changes since August**, for the rog numbers that sit beside the August report's
([`report-gh675-cuda.md`](report-gh675-cuda.md), RTX 5070 Ti Laptop under WSL2, 2026-08-23/24):

| | August (gh-ocannl-675) | tonight |
|---|---|---|
| OS | WSL2, kernel 6.18.33.2-microsoft | native Ubuntu, kernel 7.0.0-31 |
| driver / toolkit | 610.62 / CUDA 13.3 | 615.71.09 / CUDA 13.4 |
| PyTorch | 2.13.0+cu130 | 2.14.0+cu130 |
| tinygrad | 0.13.0 (editable, `62273d50f`) | 0.14.0 |
| Python | 3.12.13 | 3.14.4 |
| fixture | `043c1ea8…` (rog-nv raw-v1) | `c322a00b…` (m4-max,tuf content-v1), same shapes |
| OCANNL tuned protocol | one search per repeat, fixed arm order | one search per arm, six fresh replays, rotated order |

## Method

- **One exclusive measurement reservation per box** (`gh1181-<box>-measure-1` in the fleet
  execution registry), run under `fleet-worker.sh execution hold --request <id>`, which drains
  every correctness slot on the box before the driver starts. Before it, a correctness
  reservation per box (`gh1181-<box>-prep-1`) set up the checkouts, built both trees, captured the
  environment and smoke-ran every arm once; rog took a second one (`prep-2`) for the
  `torch.compile` fix below. The four windows ran in parallel; each had a stated 18,000 s cap.
- **Order.** Seven repeats per box, repeat 0 a discarded warm-up. Within a repeat the nine arms run
  one process (pair) at a time, the order rotated by one arm per repeat and reversed on odd ones,
  so no arm always follows the same neighbour or sits at the same position. The CPU reference runs
  first in each repeat. Linux cells are pinned with `taskset -c 0-15`, as August's were.
- **Two-pass OCANNL.** Each tuned arm searches once, in repeat 0, into a cache of its own
  (`OCANNL_AUTOTUNE_CACHE_DIR`); the timing in every repeat comes from a fresh process that replays
  that cache (orchestrate's protocol, gh-ocannl-644). The search cost is in the `search / compile s` column. This
  departs from August, which searched in every repeat: a `gpt2_mini` search costs 8-16 minutes per
  arm on these boxes (`--search-once` exists for that budget). The tuner caches nothing from a
  search whose timings it judged contended (`Autotune.search_measurements_cacheable`), so a
  "replay" that finds an arm missing searches it again; the driver records such a pass as a cache
  completion and runs a fresh process, at most twice, and only a process that replayed every arm is
  quoted. That rule bites on tuf (below).
- **Single-process PyTorch and tinygrad.** `torch.compile` and BEAM search in the process that then
  times the steps, as orchestrate runs them (gh-ocannl-675 measured that cost, and it does not keep
  its sign across boxes). Every arm gets fresh caches per repeat (`TORCHINDUCTOR_CACHE_DIR`,
  `TRITON_CACHE_DIR`, tinygrad's `CACHEDB`), so every compiled and BEAM row searched.
- **Medians.** A cell's figure is the median over repeats 1-6 (1-4 on tuf) of its quotable rows'
  p50; min-max is the spread of those per-repeat p50s. One search per arm means one search outcome
  per window: comparing each window's replays with its prep smoke's independent search and replay (shared
  box), the step times differ by 1-6% on rog, minix and mac-studio, in either direction, and by
  up to 12% on tuf (in its section).

## rog-nv-linux: CUDA, RTX 5070 Ti Laptop (12 GiB discrete)

Intel Core Ultra 9 275HX (24 cores, no SMT), native Ubuntu, kernel 7.0.0-31; NVIDIA driver
615.71.09, CUDA 13.4; torch 2.14.0+cu130, tinygrad 0.14.0 (device `CUDA`), Python 3.14.4.
Window `gh1181-rog-nv-linux-measure-1`, 01:15-03:12, 7,009 s.

| arm | regime | median step p50 ms | min-max | n | search / compile s | parity (worst rel) | provenance |
|---|---|---|---|---|---|---|---|
| OCANNL tuned, master, exact | exact | **3.504** | 3.500-3.509 | 6 | 467 (search, once) | PASS (8.7e-07) | REPLAY |
| OCANNL tuned, master, approximate | approximate | **4.493** | 4.434-4.500 | 6 | 804 (search, once) | PASS (2.8e-06) | REPLAY |
| OCANNL tuned, `7014dc44` (August), exact | exact | **7.090** | 7.080-7.119 | 6 | 594 (search, once) | PASS (8.7e-07) | REPLAY |
| PyTorch eager, exact-pinned | exact | **1.833** | 1.768-1.865 | 6 | 0.3-0.8 | PASS (1.3e-07) | no search |
| PyTorch `torch.compile`, exact-pinned | exact | **1.027** | 0.988-1.074 | 6 | 2.6-2.6 | PASS (6.7e-08) | SAME-PROCESS |
| PyTorch eager, torch defaults | approximate | **0.892** | 0.891-0.896 | 6 | 0.3-0.3 | PASS (2.0e-06) | no search |
| PyTorch `torch.compile`, torch defaults | approximate | **0.559** | 0.557-0.559 | 6 | 2.0-4.9 | PASS (2.7e-06) | SAME-PROCESS |
| tinygrad JIT | approximate (no exact pin) | **5.352** | 5.345-5.355 | 6 | 1.2-1.8 | PASS (1.3e-07) | no search |
| tinygrad BEAM=2 | approximate (no exact pin) | **1.607** | 1.523-1.617 | 5 | 163.2-220.4 | PASS (5.4e-07) | SAME-PROCESS |

| OCANNL is ... slower | than `torch.compile` | than torch eager | than tinygrad BEAM=2 | than tinygrad JIT |
|---|---|---|---|---|
| exact, vs exact-pinned torch | **3.41x** | 1.91x | **2.18x** | 0.65x (1.53x faster) |
| approximate, vs torch defaults | **8.04x** | 5.04x | **2.80x** | 0.84x (1.19x faster) |
| August `7014dc44`, exact, vs exact-pinned torch | 6.90x | 3.87x | 4.41x | 1.32x |
| *August report (WSL2, Aug 23/24, other fixture bytes)* | *6.4x (7.4 / 1.15)* | *3.1x (7.4 / 2.40)* | *4.6x (7.4 / 1.60)* | *1.4x (7.4 / 5.36)* |

- **OCANNL halved its step time since August: 7.09 ms at `7014dc44` against 3.50 ms at master, the
  same box, the same night.** The August report's anchor read 7.4-9.7 ms under WSL2 (a protocol
  experiment's anchor row, one search per repeat); tonight's re-run of the same commit natively, at
  7.09 ms, sits at the low end of that range, so most of the halving is the code, not the move off
  WSL.
- PyTorch and tinygrad barely moved: torch.compile 1.15 to 1.03 ms, eager 2.40 to 1.83 ms (2.14
  and native Linux), tinygrad JIT 5.36 to 5.35 ms, BEAM=2 1.60 to 1.61 ms.
- **The approximate profile makes OCANNL slower here, 4.49 ms against 3.50 ms exact,** although it is
  the only OCANNL arm on rog whose shipped artifact tensorized (its arm B ships tf32 mma,
  `tensorization: TENSORIZED`; both exact arms ship untensorized arm A).
  torch's defaults gain 2x from tf32 and SDPA (0.89 ms eager, 0.56 ms compiled), so the
  approximate pairing is the less flattering one: 8.0x.
- One BEAM=2 cell (repeat 2) spun at 100% CPU with the GPU idle and an empty log until the
  3,600 s cell cap killed its process group; its other five repeats took 163-220 s. BEAM=2 is n=5.
- `torch.compile` failed in the prep smoke: inductor's triton launcher build needs `Python.h`, and
  this box has no `python3.14-dev`. The matching Ubuntu packages (`python3.14-dev`,
  `libpython3.14-dev` 3.14.4-1ubuntu0.2, the installed interpreter's exact version) were unpacked
  without root into `~/.local/opt/py314-dev` and named through `C_INCLUDE_PATH` for the whole
  window (recorded in `env.json`); a second prep smoke then passed both compiled arms.

## tuf-amd-linux: HIP, Radeon RX 7700S (gfx1102, discrete 8 GiB VRAM)

AMD Ryzen 7 7435HS (8 cores / 16 threads) laptop, discrete VRAM (8,573,157,376 bytes), the same GPU
driving the laptop's displays; Ubuntu kernel 7.0.0-31; HIP from the system ROCm
(`HIP_PATH=/usr`); torch 2.13.0+rocm7.1 (HIP 7.1.52802), tinygrad 0.14.0 (device `AMD`), Python
3.14.4. Window `gh1181-tuf-amd-linux-measure-1`, 00:16-05:16, which **ran into its 18,000 s cap**:
repeats 1-4 are complete (repeat 4 without the approximate arm), 5 and 6 never ran.

| arm | regime | median step p50 ms | min-max | n | search / compile s | parity (worst rel) | provenance |
|---|---|---|---|---|---|---|---|
| OCANNL tuned, master, exact | exact | *6.685* (not quotable) | 6.684-6.701 | 0 of 4 | - | PASS | **SEARCH-PASS** |
| OCANNL tuned, master, approximate | approximate | *5.887* (not quotable) | 5.876-5.904 | 0 of 3 | - | PASS | **SEARCH-PASS** |
| OCANNL tuned, `7014dc44` (August), exact | exact | **7.822** | 7.798-7.849 | 4 | 931 (search, once) | PASS (9.4e-07) | REPLAY |
| PyTorch eager, exact-pinned | exact | **7.292** | 7.256-7.348 | 4 | 0.2-0.2 | PASS (6.7e-08) | no search |
| PyTorch `torch.compile`, exact-pinned | exact | **4.923** | 4.890-5.012 | 4 | 6.9-7.0 | PASS (1.3e-07) | SAME-PROCESS |
| PyTorch eager, torch defaults | approximate | **7.393** | 7.357-7.446 | 4 | 0.2-0.2 | PASS (6.7e-08) | no search |
| PyTorch `torch.compile`, torch defaults | approximate | **4.887** | 4.830-4.936 | 4 | 6.9-7.0 | PASS (1.3e-07) | SAME-PROCESS |
| tinygrad JIT | approximate (no exact pin) | **6.176** | 6.096-6.324 | 4 | 2.5-2.5 | PASS (1.3e-07) | no search |
| tinygrad BEAM=2 | approximate (no exact pin) | **5.050** | 3.779-5.176 | 3 | 204.6-237.1 | PASS (8.7e-07) | SAME-PROCESS |

**Both master tuned rows fail the provenance gate, and are shown in italics for that reason, not
for parity** (every row on this box passed parity). On gfx1102 the tuner's second arm (B,
materialize-all), and in the approximate profile also its third, come back with 3-9 contended
timing windows out of 102-123 in every search of them, and the tuner caches nothing from such a search
(`Autotune.search_measurements_cacheable`, see [Found on the way](#found-on-the-way)). So every
master process, the replay included, searches arm B again before timing the arm-A winner it did
replay from the cache, and the driver's two completion passes per repeat could never complete the
cache; that cost 7-10 minutes per pass and is why the window ran out. The August tuner has no such
veto, so its rows replay cleanly. How much the in-process search moves the timing is not
measured on this box directly, but everything that bears on it says little: on the other three
boxes the search pass and the clean replays of the same artifact agree within -1.1% to +3.3%; here,
the master exact search pass (which searched both arms) read 6.658 ms against 6.685 ms for the
passes that searched only arm B, and the August arm's clean replays sit within 0.1% of its own
search pass. Those readings are diagnostics: the taglines and the headline do not use them, and
quote for this box only the August commit's clean figures, which bound master from above.

| OCANNL is ... slower | than `torch.compile` | than torch eager | than tinygrad BEAM=2 | than tinygrad JIT |
|---|---|---|---|---|
| August `7014dc44`, exact, vs exact-pinned torch (clean) | **1.59x** | 1.07x | **1.55x** | 1.27x |
| *diagnostic, not quotable:* master exact, vs exact-pinned torch | *1.36x* | *0.92x* | *1.32x* | *1.08x* |
| *diagnostic, not quotable:* master approximate, vs torch defaults | *1.20x* | *0.80x* | *1.17x* | *0.95x* |

- This is the closest race of the four boxes: on a small discrete RDNA3 part neither PyTorch nor
  tinygrad gets far from OCANNL; even the August commit is level with torch eager.
- On the diagnostic readings OCANNL improved least here, 1.17x (7.82 to 6.69 ms), and the window's
  one search per arm matters:
  the prep smoke's independent search had found a 5.98 ms exact schedule (shared box, not
  quotable), against the window's 6.69 ms, 12% slower (the approximate arm went the other way,
  6.34 ms in the smoke against 5.89 ms). On the other boxes the smoke and the window differ by 1-6%.
- On the diagnostic readings the approximate profile helps here (5.89 against 6.69 ms).
- **tinygrad's BEAM search wedged this GPU three times** (`amdgpu ... device wedged, but recovered
  through reset` at 01:53, 02:15 and 03:24): two of those cells died (`HW fault ...
  memory_lost=1`, `MMU fault`), and the 02:15 one finished across the reset with the outlying
  3.78 ms (the other two repeats: 5.05 and 5.18 ms). BEAM=2 is n=3; its median does not depend on
  the outlier. A fourth BEAM cell was cut by the window's cap.

## minix-amd-linux: HIP, Radeon 8060S (gfx1151, unified memory)

AMD Ryzen AI MAX+ 395 (16 cores / 32 threads), unified memory with a 64 GiB GPU carve-out,
Ubuntu kernel 7.0.0-34; HIP from the system ROCm (`HIP_PATH=/usr`); torch 2.13.0+rocm7.1 (HIP
7.1.52802), tinygrad 0.14.0 (device `AMD`), Python 3.14.4. Window
`gh1181-minix-amd-linux-measure-1`, 00:12-01:10, 3,503 s.

| arm | regime | median step p50 ms | min-max | n | search / compile s | parity (worst rel) | provenance |
|---|---|---|---|---|---|---|---|
| OCANNL tuned, master, exact | exact | **4.757** | 4.742-4.772 | 6 | 512 (search, once) | PASS (8.7e-07) | REPLAY |
| OCANNL tuned, master, approximate | approximate | **4.692** | 4.683-4.704 | 6 | 649 (search, once) | PASS (8.7e-07) | REPLAY |
| OCANNL tuned, `7014dc44` (August), exact | exact | **7.888** | 7.861-7.906 | 6 | 653 (search, once) | PASS (8.7e-07) | REPLAY |
| PyTorch eager, exact-pinned | exact | missing | - | 0 | - | - | exit -11 |
| PyTorch `torch.compile`, exact-pinned | exact | missing | - | 0 | - | - | exit -11 |
| PyTorch eager, torch defaults | approximate | missing | - | 0 | - | - | exit -11 |
| PyTorch `torch.compile`, torch defaults | approximate | missing | - | 0 | - | - | exit -11 |
| tinygrad JIT | approximate (no exact pin) | **5.751** | 5.719-5.804 | 6 | 1.6-1.7 | PASS (1.3e-07) | no search |
| tinygrad BEAM=2 | approximate (no exact pin) | **3.093** | 2.797-3.145 | 6 | 115.9-178.8 | PASS (5.4e-07) | SAME-PROCESS |

| OCANNL is ... slower | than tinygrad BEAM=2 | than tinygrad JIT |
|---|---|---|
| exact | **1.54x** | 0.83x (1.21x faster) |
| approximate | **1.52x** | 0.82x (1.23x faster) |
| August `7014dc44`, exact | 2.55x | 1.37x |

- **No PyTorch arm runs on this box, so it has no PyTorch ratio.** torch 2.13.0+rocm7.1 segfaults
  on its first host-to-device copy, inside the wheel's bundled `librocprofiler-sdk.so` (the
  faulthandler trace: `at::native::copy_` -> `c10::cuda::memcpy_and_sync` -> rocprofiler-sdk ->
  `libamdhip64.so`), for a bare `torch.ones(4).cuda()` as much as for the runner, with or without
  `taskset`, under `HSA_ENABLE_SDMA=0`, and with the profiler hooks disabled. All 28 torch cells
  of the window failed the same way (`exit -11`), and the same probe passes on tuf with the same
  wheel. The four arms are missing, not substituted.
- OCANNL gained 1.66x here: 7.89 ms at the August commit, 4.76 ms at master. It now beats
  tinygrad's JIT and trails its BEAM search by half.

## mac-studio: Metal, Apple M4 Max (unified memory)

Apple M4 Max (40-core GPU), macOS 26.6.2; torch 2.13.0 (`mps`), tinygrad 0.13.0 (device
`METAL`), Python 3.12.14. These are the versions in mac-studio's bench venv, one minor release
behind the Linux boxes on both frameworks. Window `gh1181-mac-studio-measure-1`, 01:19-02:10,
3,047 s. mac-studio is also the maintainer's desktop: at the window's start the load average was
3.1, with WindowServer, a browser and the ChatGPT app the busiest processes. No cell is pinned
(macOS has no `taskset`).

| arm | regime | median step p50 ms | min-max | n | search / compile s | parity (worst rel) | provenance |
|---|---|---|---|---|---|---|---|
| OCANNL tuned, master, exact | exact | **8.334** | 8.301-8.401 | 6 | 858 (search, once) | PASS (8.1e-07) | REPLAY |
| OCANNL tuned, master, approximate | approximate | **7.457** | 7.368-7.481 | 6 | 660 (search, once) | PASS (8.7e-07) | REPLAY |
| OCANNL tuned, `7014dc44` (August), exact | exact | **10.953** | 10.769-11.061 | 6 | 885 (search, once) | PASS (2.0e-07) | REPLAY |
| PyTorch eager, exact-pinned | exact | **5.960** | 5.852-6.317 | 6 | 0.2-0.5 | PASS (1.3e-07) | no search |
| PyTorch `torch.compile`, exact-pinned | exact | **1.893** | 1.809-1.919 | 6 | 1.4-1.5 | PASS (6.7e-08) | SAME-PROCESS |
| PyTorch eager, torch defaults | approximate | **3.097** | 3.076-3.125 | 6 | 0.1-0.3 | PASS (1.3e-07) | no search |
| PyTorch `torch.compile`, torch defaults | approximate | **1.732** | 1.694-1.745 | 6 | 1.1-1.2 | PASS (1.3e-07) | SAME-PROCESS |
| tinygrad JIT | approximate (no exact pin) | **3.665** | 3.300-3.770 | 6 | 0.9-0.9 | PASS (1.3e-07) | no search |
| tinygrad BEAM=2 | approximate (no exact pin) | **2.063** | 1.990-2.210 | 6 | 46.7-70.1 | PASS (4.7e-07) | SAME-PROCESS |

| OCANNL is ... slower | than `torch.compile` | than torch eager | than tinygrad BEAM=2 | than tinygrad JIT |
|---|---|---|---|---|
| exact, vs exact-pinned torch | **4.40x** | 1.40x | **4.04x** | 2.27x |
| approximate, vs torch defaults | **4.30x** | 2.41x | **3.61x** | 2.03x |
| August `7014dc44`, exact, vs exact-pinned torch | 5.78x | 1.84x | 5.31x | 2.99x |

- OCANNL improved 1.31x since August (10.95 to 8.33 ms), less than on rog and minix, and in the
  exact pairing Metal is where it is furthest behind: 4x behind both torch.compile and tinygrad
  BEAM.
- The approximate profile helps on Metal (7.46 against 8.33 ms), unlike on CUDA. All three OCANNL
  arms ship tensorized (simdgroup-matrix) artifacts here, the August one included.
- Every one of the window's 54 timing rows was quotable.

## `gpt2_mini_train`: smoke only, not measured

The scope comment allowed the training workload "where a box's budget allows". No box's did, for
three separate reasons, so no training ratio is quoted:

- **Search cost.** A tuned training search is two to four times an inference search: 1,512 s
  (rog, exact), 1,283 s and 1,299 s (mac-studio, exact and approximate), and rog's approximate one
  hit the 3,600 s cell cap. On minix one arm's seed phase alone ran past 27 minutes. Three tuned
  arms per box would not fit a 5-hour window beside the inference matrix.
- **No Metal anchor.** The August runner's training search on Metal died in the Metal compiler
  service (`newComputePipelineStateWithFunction ... XPC_ERROR_CONNECTION_INTERRUPTED`, after
  retries).
- **No minix torch** (minix section).

What the smoke runs measured, as single shared-box repeats (one process each, the box otherwise
running only the smoke), all parity-passing:

| box | OCANNL tuned exact | OCANNL tuned approx. | torch eager (exact) | torch.compile (exact / defaults) | tinygrad JIT |
|---|---|---|---|---|---|
| rog-nv-linux | 23.79 ms | (search timed out) | 5.38 ms | (not smoked) | 15.50 ms |
| mac-studio | 59.61 ms | 54.40 ms | 12.85 ms | 5.05 / 4.75 ms | 9.37 ms |

## Found on the way

1. **On tuf, some tuned arms can never be cached, so no tuned master timing there is a clean
   replay.** `Autotune` refuses to cache a search in which any candidate's timing window was
   contended (most of its samples stalled past the contention floor, `window_result`), so that "a
   later idle process" retries it (`search_measurements_cacheable`). On gfx1102 the retry never
   succeeds: of the window's 45 master searches that timed anything, the three with no contended
   window were arm A's first search in each profile and one abandoned search that timed four
   candidates; the other 42, every search of arm B and of the approximate profile's third arm,
   each came back with 3-9 contended windows out of 102-123 timed candidates. So every master process searches those arms again (7-10 minutes) before
   timing arm A's replayed winner. rog, minix and mac-studio record zero contended windows in all
   of their searches. A contention verdict that the same candidates reproduce on an exclusive box
   is a property of those candidates, not of the box, and should not veto the cache.
2. **The approximate profile slows tuned `gpt2_mini` down on CUDA, the one backend with tf32**:
   rog 4.49 against 3.50 ms exact, although that arm ships tf32 mma; it helps on tuf (diagnostic:
   5.89 against 6.69) and Metal (7.46 against 8.33) and is neutral on minix (4.69 against 4.76). torch's
   defaults gain 2x on rog. For a v1.1 whose goal is performance in the approximate profile on
   transformer workloads, the rog row is the one to look at first.
3. **minix's bench venv cannot run PyTorch**: torch 2.13.0+rocm7.1 segfaults in its bundled
   rocprofiler-sdk on the first device copy on gfx1151 (details in the minix section). The venv
   was provisioned on 2026-10-03 (lukstafi/ludics-lite#537) and this was its first GPU use.
4. **rog has no `python3.14-dev`**, so `torch.compile` cannot build its triton launcher without
   the workaround above; installing the package retires it.
5. **tinygrad 0.14.0's BEAM search can spin indefinitely on CUDA** even with `PARALLEL=0` (one
   cell in six on rog, 100% CPU, GPU idle, until the cap), and **wedged tuf's GPU three times**
   (`HW fault ... memory_lost=1`, `MMU fault`; the kernel log's `amdgpu ... device wedged, but
   recovered through reset`). orchestrate's cell cap and process-group kill contained all of it,
   and the driver went on to the next cell.

## Reproduction

Per box, from a checkout of master with `benchmarks/.venv` linked to the bench venv, the fixture
copied in, and a second checkout of `7014dc44` with its `bench_gpt.exe` built:

```sh
cd <master>/benchmarks
.venv/bin/python gh1181_cells.py --backend <cuda|hip|metal> --box <box> \
  --aug <7014dc44 checkout> --out <results dir> \
  --workloads gpt2_mini --repeats 7 --search-once --cell-timeout 3600 --total-timeout 18000
.venv/bin/python gh1181_cells.py --summarize --out <results dir>
```

On rog, `C_INCLUDE_PATH` pointed at the extracted headers (rog section). The raw records (every row,
every cell log, the order, the failures, `env.json`) are in `~/.local/state/gh1181/measure/` on
each box. The driver that produced them is recorded by sha256 in each `env.json` (`9676a065…`, the
version at the PR's third commit). Review fixes since then change only bookkeeping that nothing
quoted here reads: completion passes' searches now add to the reported search cost (no completion
pass ever produced a clean replay), the last allowed completion pass is no longer also written
once more to `checked.jsonl` (tuf's file has those duplicates; `timings.jsonl` does not),
`--summarize` keeps the measured `wall_s` and failure count, `--repeats 0` is refused, and a
wrong-backend row is kept, marked, in `raw.jsonl` (none occurred).

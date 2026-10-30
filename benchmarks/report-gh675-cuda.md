# gh-ocannl-675, CUDA leg: does a process that searched time its steps more slowly?

> **2026-10-30 (gh-ocannl-1357): measured on the replaced raw-v1 files, not comparable byte-for-byte
> with later content-v1 reports.** Its `gpt2_mini` is rog-nv's raw-v1 file (`043c1ea8…`). The files on those boxes have since been replaced by the
> m4-max content-v1 bytes, which `fixtures/DIGESTS.txt` now records for rog-nv and minix.

**Box `rog-nv-wsl` (RTX 5070 Ti Laptop, WSL2), 2026-08-23/24.** The sibling ROCm leg
(gfx1151) is the comment on gh-ocannl-675 dated 2026-08-23; this file is the NVIDIA half and the
one that settles the split/no-split call, because it ran second.

**Headline.** On this GPU the only *non-OCANNL* searching cell whose timing process is measurably
slower is tinygrad's `--beam N` -- by **+6.4%** on `mlp_small` and **+2.3%** on `gpt2_mini`, small
but carried by *every* paired repeat (9/9 and 4/4). (OCANNL's own tuned cell is slower too behind
a long search, +10.3%; it is stated separately just below because it is the anchor this whole
question is asked against, not because it is exempt.) `torch.compile` goes the other way: its
searching process is **faster** than the fresh process that replays inductor's cache
(-12.0% / -4.0%), so splitting its passes would penalise it, not credit it. OCANNL's own tuned
cell -- the one the two-pass protocol exists for -- shows a residue only after a *long* search
(+10.3% / +8.4% behind a 16 s search) and none at all behind a 4 s one. **The 2.5-3.5x that
`README.md` attributes to small CUDA kernels does not reproduce here, on the hardware class it
names.**

## Environment

| | |
|---|---|
| box | ASUS ROG, NVIDIA GeForce RTX 5070 Ti Laptop GPU (compute capability 12.0, 12227 MiB), WSL2, kernel 6.18.33.2-microsoft-standard-WSL2, 24 logical cores |
| driver / toolkit | driver 610.62 (CUDA API 13.3), CUDA toolkit 13.3 (`nvcc` V13.3.73) at `/usr/local/cuda` |
| PyTorch | `2.13.0+cu130` (stock Linux wheel), device `cuda` |
| tinygrad | 0.13.0, editable install from `~/tinygrad` at `62273d50f`, device **`CUDA`** (the system nvrtc works here; the README's `LD_LIBRARY_PATH` caveat applies only to a driver older than the toolkit, which this box no longer has) |
| other | python 3.12.13, numpy 2.5.1, safetensors 0.8.0 |
| ocannl | worktree of `origin/master` `7014dc44`, backend `cuda`, `BENCH_PRECISION=f32` |
| fixtures | `mlp_small.safetensors` (51384 B) sha256 `f09de9504262eb184672c9f58da821c47e0ee9f6311bd69207700c33298c44ca`; `gpt2_mini.safetensors` (13871360 B) sha256 `043c1ea8b42817fca8b713ac0eadae6e8b7b7326cb5ff7ea0fd527487ca2009e` |

Those are written in full deliberately: the raw records live on the measurement box, so this
checked-in file is the only place a reader can check whether a regenerated fixture is still the
workload these numbers are about.

The fixture digests differ from the ROCm leg's for the same workload names (same sizes, different
bytes: `gen_fixtures.py` draws from a numpy version that does not promise stream stability).
Absolute step times are therefore **not** comparable across the two legs -- the *ratios* are what
both legs report, and a ratio is taken within one box, one fixture, one pair of processes.

> **Resolved by gh-ocannl-759.** This report asked which of the two boxes' bytes becomes canonical
> and said it was a decision for the suite, not for one leg's report. The suite's answer is
> *neither*: `fixtures/DIGESTS.txt` now records digests per origin
> (`<sha256>  <bytes>  <name>  <origin>`), so both boxes' bytes are pinned, a fixture passes the
> gate if it is some recorded box's, and every row and report section names whose. The two digests
> above are the checked-in `rog-nv` entries; minix's are checked in beside them. The
> non-comparability the paragraph above describes is therefore now a recorded fact rather than a
> caveat one report happens to carry.

## Method

- Driver: `benchmarks/gh675_cells.py` (checked in with this report). One arm = one pair of
  processes; arms alternate across repeats rather than running an arm to completion, so a drifting
  box moves both halves of a pair together. Each run is pinned with `taskset -c 0-15` behind an
  idle gate (load average and GPU utilisation).
- **X is the median of the paired per-repeat ratios** (pass-1 `step_ms` p50 over pass-2 `step_ms`
  p50, minus 1), not a ratio of arm-level medians. Repeat 0 of every arm is a discarded warm-up.
- Caches whose warmth *is* the experiment: tinygrad's kernel cache is redirected per arm with
  `CACHEDB=`, and deleted before each pass 1 of the COLD arms (both beam widths and the
  cold-compile control) -- but deliberately NOT for the warm/warm control, whose claim is that both
  its passes replay: that arm is prewarmed once per invocation and its cache is preserved across
  the pair. Inductor gets a fresh `TORCHINDUCTOR_CACHE_DIR` per repeat with
  `TORCHINDUCTOR_FX_GRAPH_CACHE=1`; OCANNL's autotune cache (kept under `$GH675_OUT`, not in the
  checkout) is wiped before each pass 1.
- Provenance is verified, not assumed, and the expectation differs by arm. For the SEARCHING arms
  (both beam widths, both torch.compile modes, the OCANNL tuned cell) every pass-1 row reports
  `searched: true` and every pass-2 row `searched: false`. For the three CONTROLS -- whose whole
  claim is that they do not search -- both passes report `searched: false`, which is equally part
  of what makes them controls. The driver states the expectation per arm and rejects a pair that
  violates it. Reading it at all needed the gh-ocannl-751 patch (see "Two things found on the
  way").
- Parity is the orchestrator's gate recomputed offline against `pytorch/cpu/eager`, tolerance
  2e-3: **every cell passes**, worst 8.7e-07, so all rows are comparable.
- **Known limitation of THIS run's data.** Arms alternated within each repeat but in a FIXED
  order, so every arm sat at the same position in every repeat. Since the process-wide CUDA JIT
  cache warms across a session -- documented below, and worth 16 s vs 4 s of OCANNL search time --
  arm identity is partly confounded with accumulated compiler state. X itself is a within-pair
  ratio between two adjacent processes, which is what makes it robust to that drift; what the
  confound can reach is the comparison BETWEEN arms, and the anchor's two regimes in particular.
  The driver now rotates the arm order per repeat (`gh675_cells.py`), which the numbers below
  predate.

## Results

`X` = (pass-1 step p50) / (pass-2 step p50) - 1, median over paired repeats. `pos/n` is how many
of those paired ratios were positive -- the statistic that separates a small real effect from
noise. `blk2` is the retime control: pass-1's SECOND timed block over pass-2's SECOND timed block
(see below -- matched block positions, so only the process differs). Step times are the
sync-after-every-step `step_ms` p50, i.e. the number the report publishes.

| workload | cell | pass-1 p50 (ms) | pass-2 p50 (ms) | n | **X** | pos/n | X IQR | `blk2` | pass-1 `compile_s` |
|---|---|---|---|---|---|---|---|---|---|
| mlp_small | tinygrad/CUDA/beam, BEAM=2 | 0.1138 | 0.1047 | 9 | **+6.4%** | 9/9 | +4.4 .. +8.8 | 1.07 | 13.5-20.8 |
| mlp_small | tinygrad/CUDA/beam, BEAM=8 | 0.1116 | 0.1030 | 2 | **+8.3%** | 2/2 | +7.8 .. +8.9 | 1.11 | 53.0-57.1 |
| mlp_small | pytorch/cuda/compiled (default) | 0.5455 | 0.6154 | 9 | **-12.0%** | 1/9 | -17.6 .. -5.1 | 0.83 | 1.9-2.6 |
| mlp_small | pytorch/cuda/compiled, `max-autotune` | 0.4398 | 0.4657 | 9 | **-2.0%** | 1/9 | -9.4 .. -0.6 | 0.92 | 4.0-4.4 |
| mlp_small | *control* tinygrad/CUDA/jit, warm both passes | 0.1217 | 0.1217 | 9 | **-1.5%** | 3/9 | -6.9 .. +0.8 | 1.00 | 0.5 |
| mlp_small | *control* tinygrad/CUDA/jit, **cold compile, no search** | 0.1206 | 0.1208 | 9 | **-0.1%** | 4/9 | -0.8 .. +2.3 | 0.99 | 0.6-1.2 |
| mlp_small | *control* pytorch/cuda/eager | 0.5811 | 0.6030 | 9 | **-1.6%** | 3/9 | -3.9 .. +3.2 | 0.99 | 0.2 |
| mlp_small | *anchor* ocannl/cuda/tuned | 0.0501 | 0.0501 | 9 | **-0.0%** | 4/9 | -3.9 .. +2.3 | -- | 3.6-16.4 |
| gpt2_mini | tinygrad/CUDA/beam, BEAM=2 | 1.5989 | 1.5446 | 4 | **+2.3%** | 4/4 | +2.0 .. +5.1 | 1.02 | 58.6-68.4 |
| gpt2_mini | pytorch/cuda/compiled (default) | 1.1461 | 1.1855 | 5 | **-4.0%** | 0/5 | -4.0 .. -0.8 | 0.99 | 3.7-4.6 |
| gpt2_mini | pytorch/cuda/compiled, `max-autotune` | 1.4710 | 1.4873 | 3 | **-1.1%** | 1/3 | -5.6 .. +2.2 | 1.03 | 10.0-11.4 |
| gpt2_mini | *control* tinygrad/CUDA/jit, warm both passes | 5.3569 | 5.4764 | 3 | **-2.5%** | 1/3 | -2.7 .. +0.5 | 1.00 | 1.0-1.1 |
| gpt2_mini | *control* tinygrad/CUDA/jit, **cold compile, no search** | 5.4763 | 5.3677 | 3 | **+0.1%** | 2/3 | -0.2 .. +2.6 | 1.00 | 1.1-1.8 |
| gpt2_mini | *control* pytorch/cuda/eager | 2.3969 | 2.3914 | 5 | **+0.6%** | 3/5 | -0.0 .. +1.9 | 0.99 | 0.2 |
| gpt2_mini | *anchor* ocannl/cuda/tuned | 9.6509 | 7.4012 | 3 | **+0.5%** | 2/3 | -0.4 .. +47.7 | -- | 206-613 |

**Per-cell X on this hardware:** tinygrad `beam` **+6.4%** (mlp_small) / **+2.3%** (gpt2_mini);
`torch.compile` default **-12.0%** / **-4.0%**; `torch.compile max-autotune` **-2.0%** / **-1.1%**;
OCANNL `tuned` **-0.0%** / **+0.5%** pooled (but see the anchor section -- pooling hides a split).

## What the controls say

1. **Process-to-process spread.** The three non-searching controls sit at |X| <= 2.5%
   at the median, and their paired ratios are a coin flip (3/9, 4/9, 3/9 positive on `mlp_small`).
   A single repeat of the 0.1 ms cell still swings +/-10%, which is why the beam row is n=9 and why
   the sign count, not the range, is the statistic.
2. **Compilation is not the mechanism.** A tinygrad `--jit 1` cell run with a *deleted* kernel
   cache compiles every kernel in the timing process and comes out at -0.1% / +0.1% against its own
   warm replay, 4/9 and 2/3 positive. The only difference between that control and the beam arm is
   `BEAM=N`.
3. **It is not first-block warmup.** A `--retime` flag (added for this probe) times a *second*
   block of `timed_steps` inside the same process, ~400 steps in -- and BOTH passes record one, so
   the comparison is pass-1 block 2 over pass-2 block 2: same block position, same step index. What
   it does NOT match is elapsed process lifetime -- pass 1 reaches its blocks after 13.5-68.4 s of
   searching, pass 2 after a replay of a second or two -- and that cannot be matched, because it is
   the treatment: a pass 2 aged equivalently would have had to do something equally expensive, at
   which point it is not a replay. So this control excludes block position and step index as the
   explanation, and does not separate "slower because it searched" from "slower because it is an
   older, fuller process that searched" -- which is the same claim the README's stated mechanism
   makes (accumulated modules and buffers), not a competing one. The searching process stays slow:
   **1.07** (mlp_small BEAM=2), **1.11** (BEAM=8), **1.02** (gpt2_mini), against 0.99-1.00 for the
   three controls and 0.83 for `torch.compile`, whose block 2 confirms it is genuinely the *faster*
   process. (An earlier revision of this report divided pass-1's second block by pass-2's FIRST
   one, which left block position confounded with process identity; the matched comparison is the
   one tabulated, and it separates the beam cell from the controls more sharply than the
   confounded one did -- 1.07 vs 0.99-1.00, where the confounded numbers were 1.055 vs 0.97-0.98.)
4. **Beam width is a weak dose-response knob.** BEAM=8 costs 3.4x the search (55 s vs 16 s) and
   moves X from +6.4% to +8.3% -- present, but far from proportional.
5. **The effect is small and one-sided, not large and noisy.** 9/9 + 4/4 + 2/2 positive paired
   ratios across the three beam rows is what makes a ~6% median statable at all; by contrast
   `torch.compile`'s 1/9 and 0/5 make its *negative* sign equally real.

## The OCANNL anchor, and why the 2.5-3.5x is not reproduced

`ocannl/cuda/tuned` on `mlp_small` splits cleanly in two by how long its own search took:

| regime | repeats | pass-1 `compile_s` | X |
|---|---|---|---|
| long search | 1, 2 (and the discarded warm-up 0) | 16.4 s | **+10.3%**, **+8.4%** (warm-up: +34.6%) |
| short search | 3-9 | 3.6-5.2 s | **-0.2%** median, 3/7 positive |

Every one of those ten passes genuinely searched (`searched: true`, `searches: 2`, `replays: 0`
in the `tune` object) -- the provenance field rules out the obvious explanation that the later
runs replayed a cache. What changed is the *cost* of the search: the CUDA driver's JIT cache
(`~/.nv/ComputeCache`, 323 MB with 17,306 files written during this session) is warmed by every
preceding cell, so a "from-scratch" OCANNL search stops being from-scratch after the first few
repeats even with `autotune_cache/` wiped. The residue tracks the length of the search that
preceded the timing, which is consistent with the README's stated mechanism (a searching process
accumulates modules and buffers) -- and it means **`compile_s` in any published CUDA report is a
warm-JIT number unless `~/.nv/ComputeCache` was cleared too.**

Either way the headline number in `README.md` does not survive: the largest OCANNL residue
measured here -- and OCANNL's tuned cell is what that figure describes -- is **+10.3%** on the
smallest kernels, against a documented **2.5-3.5x**; on
`gpt2_mini` it is +0.5%; and the ROCm leg measured -0.2% / -0.1%. Two boxes, two backends, no
reproduction.

## The call: keep every NON-OCANNL cell single-pass

Reading both legs against the brief's rule (split a searching cell whose `mlp_small` X exceeds
~10%) -- and stating up front the one thing the brief did not specify, since two boxes now
disagree: **the threshold has to be cleared on every measured box, not on any one of them.** That
is a decision, made here and defensible on its own terms rather than derived from the rule, so it
is worth the argument: the two-pass protocol is a property of the published MATRIX, not of a
machine. Reports are read across boxes -- this file's whole reason for existing is that the ROCm
leg's table and this one are compared -- and a cell that is two-pass on gfx1151 and single-pass on
an RTX 5070 Ti produces two numbers that are not comparable with each other, which is the exact
harm the protocol exists to prevent. The alternative, a per-box protocol, was rejected on that
ground and not on the results.

Applied per-box instead, the verdicts would be: tinygrad beam SPLITS on ROCm (+14.9%) and not on
CUDA (+6.4%); `max-autotune` SPLITS on ROCm (+12.3%) and not on CUDA (-2.0%); `torch.compile`
splits on neither by the mlp_small rule. A reader who prefers per-box protocols should take that
reading -- the numbers are the same either way, and both are in the table below.

| cell | CUDA (this leg) | ROCm (gfx1151) | verdict |
|---|---|---|---|
| tinygrad `--beam N` | +6.4% / +2.3% | +14.9% / +9.0% | **no split** -- over the line on one box, well under on the other |
| `torch.compile` default | -12.0% / -4.0% | +7.2% / +21.2% | **no split** -- the sign itself is box-dependent |
| `torch.compile max-autotune` | -2.0% / -1.1% | +12.3% / +33.9% | **no split**; add the variant single-pass if it is added |
| OCANNL `tuned` (anchor) | +10.3% behind a 16 s search, ~0 behind a 4 s one | -0.2% / -0.1% | protocol kept, rationale re-stated |

No cell clears the line on both boxes, and no cell clears it on the box whose hardware the
README's own rationale names.

**No mechanism is established by these two legs, and this report should not be read as offering
one.** Two candidate stories both fail against the measured rows, which is worth recording because
each is the obvious thing to conclude from one leg alone:

- *"An expensive search leaves a residue."* Falsified here: OCANNL on `gpt2_mini` searched
  **206-613 s**, by far the most expensive search measured in either leg, and shows **+0.5%**.
- *"...and only where the step is short enough for a per-launch cost to show."* That fits this
  leg -- every CUDA cell's residue is larger on `mlp_small` (0.05-0.6 ms steps) than on
  `gpt2_mini` (1.1-9.7 ms) -- and is falsified by the ROCm leg, where the two torch cells run the
  other way round: **+21.2%** and **+33.9%** on `gpt2_mini` against +7.2% and +12.3% on
  `mlp_small`. Big steps do not swamp the residue there.

What survives is narrower and per-cell: OCANNL's own residue on this box tracks the cost of the
search that preceded it *within one workload* (+10.3% behind 16 s, ~0 behind 4 s on `mlp_small`),
and every positive CUDA cell is larger on the small-step workload. Those are observations about
this leg, not a law -- and the matrix rule below does not rest on one. It rests on the measured X
per cell per box, which is what the issue asked for; explaining WHY a given framework's process
carries a cost would need per-framework instrumentation neither leg ran. So the honest matrix rule is
"single pass, measured at <=X% per cell per box", with X recorded -- which is a result, and it is
what makes the asymmetry defensible instead of merely documented.

The two-pass protocol stays on the OCANNL tuned cell: it is cheap there (the cell is already
two processes), it is the only cell whose `compile_s` is *defined* as a from-scratch search cost,
and it is the only one gated on provenance. It is now justified by a measured <=10.3% rather than
by an unreproduced 2.5-3.5x.

## Two things found on the way (not part of the measurement)

1. **The runner leaves `runners/` on `sys.path`, which wedges every parallel beam search.**
   `runners/tinygrad/run.py` inserts its parent directory on `sys.path` to import `bench_common`
 -- after importing tinygrad, precisely so the `runners/tinygrad/` directory cannot shadow the
   real package. But tinygrad's beam search compiles its candidates in a **`spawn` pool**, and a
   spawned worker re-executes the module top-level with the *parent's* `sys.path`, where the
   `import tinygrad` at line 39 now runs with `runners/` still in front. The path scan finds
   `runners/tinygrad/` as a namespace portion, the editable install's meta-path finder (appended
   *after* `PathFinder`) never gets a look, and every worker dies with
   `ImportError: cannot import name 'Tensor' from 'tinygrad' (unknown location)`. The pool
   respawns them forever, so the search **hangs instead of failing**: the first attempt here sat
   35 minutes at 3% CPU with the GPU idle and had to be killed. Fixed in this branch by dropping
   the path entry again after the import; a cold BEAM=2 search then finishes in 28 s. This affects
   any device whose beam search parallelises (`CUDA`, `AMD`, `NV`, `METAL`, `HIP`) with an editable
   tinygrad -- i.e. exactly the canonical `orchestrate.py --beam N` run on this box.
2. **The `searched` provenance probe is stale for tinygrad >= 0.13** (gh-ocannl-751, filed by the
   ROCm leg with the patch): `instrument_tinygrad_beam` binds `tinygrad.engine.search`, which
   0.13 replaced with `tinygrad.codegen.opt.search`, so every beam cell reports `searched: null`
   and the report's `pass` column prints `?`. The patch from that issue is applied in this branch;
   without it none of the beam rows above could have stated which pass produced them.

Also observed, and **not** explained: one `mlp_small` BEAM=8 pass-1 and one `gpt2_mini` BEAM=2
pass-1 wedged the same way as the `sys.path` bug did (minutes at ~1% CPU, GPU idle) *after* that
bug was fixed, against 55 s and 63 s for the same searches in their other repeats. Both were
killed by the driver's cap and their pairs dropped (hence n=2 and n=4 in those rows). The ROCm
leg reports one wedge of the same shape. Something in tinygrad's parallel beam pool deadlocks
intermittently; it is worth its own issue (gh-ocannl-760).

Raw per-run records (every arm, every repeat, losses, `queued_step_ms`, `retime_step_ms`,
`searched`, the OCANNL `tune` objects) are kept on the box at `$GH675_OUT/records.jsonl`
(99 pairs); say the word and I will attach them.

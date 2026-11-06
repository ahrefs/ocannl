# gh-ocannl-481 on CUDA: `ldmatrix` renders, and changes nothing measurable — yet

Measurement-only report. The subject is the `ldmatrix`-over-swizzled-staging work of
[gh-ocannl-481](../archive/docs/proposals/gh-ocannl-481.md) (staging PR #285), whose design resolution is
[gh-ocannl-481-item3-ldmatrix](../archive/docs/proposals/gh-ocannl-481-item3-ldmatrix.md). That work's own
acceptance rule is T4's: *driven by benchmarks, not speculation*. This is the benchmark.

**The verdict is neutral, and the interesting part is how little the headline A/B could say about
it.** The mechanism works end to end — twins are seeded, compile, time, and render `ldmatrix`. The
same-schedule paired comparison puts the layout's effect at **+0.20% median (n=48, stdev 0.50%)**,
i.e. nothing. The whole-cell before/after, run the way the predecessor reports run it, showed a
*24% "improvement"* that its own negative control proves is search noise.

Hardware: NVIDIA GeForce RTX 5070 Ti Laptop GPU (sm_120), driver 610.62 / CUDA API 13.3, toolkit
13.3, under WSL2; Intel Core Ultra 9 275HX. Same machine as
[report-gh537-cuda.md](report-gh537-cuda.md) and [report-gh484-cuda.md](report-gh484-cuda.md).

## A/B pair

`NEW` = the branch. `OLD` = the same tree with CUDA's `mma_staged_layouts` entry emptied, so no
swizzled twin is ever seeded and no `ldmatrix` can render. The library delta is that one list
literal and nothing else; the `NEW` binary was rebuilt after restoring the source and is
byte-identical to the one measured.

That isolation is also the *whole* branch's effect on these workloads: the other items (fp8
`ta`/`tb`, `pad_stride`, the arch marker) are inert for a bf16 MLP.

Two hygiene rules, both learned the hard way inside this session:

- **The schedule disk cache must be wiped before every arm.** It is keyed by the base code's
  canonical digest, which `OLD` and `NEW` share, so the second arm otherwise replays the first
  arm's crowned schedule and the comparison is vacuous. Measured: a warm-cache cell reports
  `compile_s` 1.76 s against 29 s cold, and the "winner" is whatever the other arm found.
- **Arm order must be balanced.** Rounds 1–2 ran `OLD` first, rounds 3–4 ran `NEW` first. With a
  fixed order, arm and position-in-session are perfectly confounded — see the control below.

Four replicates per side per cell, interleaved in a single session, no foreign GPU process during
any cell. The `ocannl/cc` column was skipped for this leg (the subject is confined to
`cuda_backend.ml`, so the CPU cells cannot differ by construction, and `mlp_wide`'s cost ~107 s
each). All 32 OCANNL cells passed the parity gate against the PyTorch CPU reference.

## The headline A/B, and why it is worthless here

`mlp_wide` (256→1024→1024→10, batch 256), `ocannl/cuda/tuned`, p50 ms per round:

| cell | OLD | NEW | naive read |
|---|---|---|---|
| **bf16** (the route under test) | 1.288, 1.418, 0.999, 0.977 | 0.979, 0.975, 0.981, 1.145 | "NEW 14% faster" |
| **f32** (negative control) | 1.505, 1.521, 1.271, 1.277 | 1.298, 1.295, 1.273, 1.270 | "NEW 8% faster" |

The f32 row is the control: `mma_staged_layouts` advertises only the uniform-bf16 triple, so **no
twin is seeded at f32 and the two binaries execute identical code there** — verified directly, the
string `swz-b128` occurs 36 times in each bf16 search log and **zero** times in any f32 log. Yet
that cell moved 8%, and spans 1.271–1.521 ms pooled: a **19.8% spread on code that cannot
differ.** On `mlp_small` the same control spans **40.6%**.

So ~20–40% is the resolution floor of a tuned-cell A/B on this box, and the bf16 "14%" sits well
inside it. Reporting it as an improvement would have been the gh-479 mistake in a new costume:
a number that is real as a number and false as a claim.

The mechanism of the noise is identifiable, not mysterious — the search does not always crown the
same family:

| run | best candidate found | ms |
|---|---|---|
| OLD-r1 | `F_sketch[gpu 64x64x8/4x4, gpu 64x64x8/4x4]` (not tensorized) | 1.190 |
| OLD-r2 | `F_sketch[gpu 32x32x8/4x4, gpu 64x64x8/4x4]` (not tensorized) | 1.190 |
| OLD-r3 | `F_sketch[mma-gpu 16x32x0 ep, mma-gpu 32x32x0]` | 0.948 |
| OLD-r4 | `F_sketch[mma-gpu 16x32x0, mma-gpu 16x32x0]` | 0.928 |
| NEW-r1..r4 | `F_sketch[mma-gpu …x0, …]` (all four) | 0.928–0.955 |

`OLD` failed to find the tensorized family in two of four searches; `NEW` found it in four of four.
That is luck, not the change — nothing in the branch touches the unstaged seeds or the beam.

## The measurement that does resolve it

The autotuner times each swizzled twin **immediately after its plain sibling, in the same process,
in the same search**, with identical tile sizes and an identical rest-of-pipeline. Only the tile
layout differs. No cross-run comparison, no clock drift, no winner lottery.

`mlp_wide` bf16, all four `NEW` rounds, pairing each `swz-b128` timing with the plain seed it
directly follows:

| staged seed | n | plain (median ms) | twin (median ms) | delta |
|---|---|---|---|---|
| `mma-gpu 16x32x32` | 8 | 2.4337 | 2.4377 | +0.16% |
| `mma-gpu 16x32x32 ep` | 8 | 2.4539 | 2.4530 | −0.03% |
| `mma-gpu 32x32x16` | 8 | 2.5937 | 2.5931 | −0.02% |
| `mma-gpu 32x32x16 ep` | 8 | 2.6087 | 2.6131 | +0.17% |
| `mma-gpu 32x32x32` | 8 | 2.3965 | 2.3926 | −0.16% |
| `mma-gpu 32x32x32 ep` | 8 | 2.4135 | 2.4223 | +0.36% |

**Overall: n = 48, median +0.20%, mean +0.16%, stdev 0.50%, range [−1.1%, +1.3%], twin faster in
17/48.** `ldmatrix` over a b128-swizzled tile is, on this workload, indistinguishable from per-lane
gathers over a plain one — a hair slower if anything.

(`mlp_small` gives n = 48, median +1.47%, stdev **4.68%**, range [−10.1%, +16.4%]: at 0.06 ms/step
it is pure overhead and measures nothing. It is retained only as the "already reaches tensor cores"
control the issue asked for.)

## The mechanism did work — the evidence it did

Neutral-because-it-did-nothing and neutral-because-it-silently-declined are different findings, and
this is exactly the confusion gh-479 exists to prevent. Per `mlp_wide` bf16 search:

- **12 distinct swizzled twins seeded**, and only for staged seeds — the unstaged `…x0` seeds have
  no shared tile and are correctly never twinned.
- **12 timed** (the fissioned forms). 12 failed at `Transform`, all of them the *whole-routine*
  forms hitting gh-521's companion-coverage precondition — their 16 plain siblings fail the same
  way, so nothing swizzle-specific fails.
- **0 `Tile_mma` statements rendered the lane-0 scalar fallback** — no `NOTE n/m` line in any cell.
- **0 declines mentioning a swizzled layout**, with `schedule_log_declines=true` on a dedicated
  re-run. A swizzled operand the arm could not consume would have logged one and fallen back.

Since `mma_syntax` accepting a `Swizzled_b128` operand *is* the promise that it was read through a
swizzle-aware load (and is what makes the caller record `Mma_intrinsics_ldmatrix`), acceptance with
zero declines and zero fallbacks means the `ldmatrix` path rendered. The emission itself is proven
separately and directly by `test/operations/schedule_ldmatrix_matmul.ml`, whose `bf_sketch` leg
builds its schedule through `Autotune.sketch_schedule` — the same pipeline the tuner seeds — and
pins `ldmatrix.sync.aligned.m8n8.x4.shared.b16` plus `.x2.trans` in the emitted `.cu` with bitwise
parity against the serial twin, on this GPU.

## Why it cannot help these workloads yet

Structural, and visible in the table above: **the crowned candidate is always the *unstaged*
tensorized family** (`mma-gpu …x0`, i.e. `sk_bk = 0` — one full-K `Tile_mma` block streaming
operands from device memory) at 0.93–0.95 ms. Swizzled twins exist only in the *staged* family,
whose best member times 1.54–1.56 ms — and whose plain siblings time the same 1.55. The staged
family loses by ~65%, so the layout of its shared tiles is a question about a schedule that does
not ship.

`ldmatrix` cannot reach the artifact until staged beats unstaged on some shape. Candidates:

- Shapes where operands do not fit the unstaged streaming pattern — larger `k`, or a reduction that
  must be blocked for residency.
- gh-480's accumulator residency making the staged form cheaper per k-block.
- A workload with enough arithmetic intensity that shared-tile bandwidth, rather than launch and
  epilogue overhead, is the constraint. `mlp_wide`'s whole step is ~1 ms across ~27 kernels; the
  tensorized segment is a fraction of it, so even a real GEMM win arrives diluted.

## What this leg changes about the branch's claims

Nothing in the code, and one line in the docs: `gh-ocannl-481.md`'s "measurement is the open half"
now has its answer for two workloads — *rendered, correct, neutral* — instead of being open. The
honest summary for the PR is that the T4 emission exists and is verified, and the ceiling it was
chasing is not reachable from the schedules these benchmarks crown.

Reproducing: `NEW`/`OLD` binaries as above, then per arm `rm -rf benchmarks/autotune_cache` and
`BENCH_CELL_LOG_DIR=… OCANNL_AUTOTUNE_LOG=true python3 benchmarks/orchestrate.py --workloads
mlp_small mlp_wide --tuned --precision bf16 --only ocannl pytorch --gpu cuda --skip-build`,
alternating which arm runs first.

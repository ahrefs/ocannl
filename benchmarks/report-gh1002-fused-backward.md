# gh-ocannl-1002: the fused attention backward in training, on Metal and cc

Measurement report for PR 2 of ahrefs/ocannl#1002 — the training measurement the design record
([docs/proposals/gh-ocannl-1002-1003.md](../docs/proposals/gh-ocannl-1002-1003.md), "Measurement
plan and completion gates") asks for. PR 1 (lukstafi/ocannl-staging#885) made the fused backward
the third shape of `Ir.Online_softmax`, behind `online_softmax_backward` (default off, in the
`approximate` profile): the training step recomputes the probabilities from the saved row max and
normalizer and never writes the probability-gradient and score-gradient `[seq, seq]` buffers.
This report measures what that buys and what it costs in a complete training step.

**Verdict.** The fused backward is a **memory win at every length and on both backends, and a time
loss on Metal from seq 512 on**, which is the outcome the design record anticipated ("a memory
improvement can coexist with a slower portable implementation; report that explicitly and keep
the default off"). With the scores stored (D1), the step's peak requested memory drops from 689
to 305 MiB at seq 1024 (0.44x of the composed step's), 430 to 238 MiB at seq 512 and 236 to
188 MiB at seq 128; with the scores recomputed as well (D2) it drops to 172-177 MiB at every
length, and the batch-1 sequence sweep shows why: the memory left quadratic in `seq` goes from
four `[seq, seq]` buffers per head per layer (composed) to one (D1, the stored scores) to none but
the user's mask (D2). On Metal the price is time against the best composed-backward form (B, the
online forward with the composed backward): 1.02x at seq 128, **1.19x at seq 512 and 1.53x at seq
1024** (166 -> 254 ms). The per-kernel attribution puts all of it in one kernel: the fused dK and
dV nests share a kernel of 1024-2048 threads (33 ms per layer at seq 1024, against 3.8 ms for B's
lane-scheduled `v.grad` plus its `k.grad`). On cc the fused backward is **faster** than B at long
context (0.94x at seq 512, 0.89x at seq 1024) and neutral at seq 128. Every treatment reproduces
the composed loss trajectory to 2.1e-7 relative. The default stays off; the Metal time is
gh-ocannl-1124's (dQ/dK lanes, and dV's lanes lost to the kernel it shares with dK).

## Protocol

Hardware: Apple M4 Max (40-core GPU), macOS 26.6.2, host `mac-studio`, under the fleet's exclusive
`measurement` reservation `gh1002-pr2-mac-studio-1` (15:39Z-17:04Z on 2026-09-28; host snapshots at
every phase boundary show desktop activity only, plus one foreign test executable,
`online_softmax_block.exe`, at 8.9% CPU in the snapshot that opened the Metal phase and in no later
one). Tree: staging `6edf081d9`, i.e. master `36bfdc139` plus this PR's first three commits (through
"Per-kernel attribution of the shipped step, and the gh-ocannl-1002 driver"; the branch was
rebased before merging, so that sha is the pre-rebase one). Those commits change only
`bench_gpt`, the harness's diagnostics and the workloads; the PR's later commits touch the
driver's untimed artifact pass, a test and documentation.

Every cell is one `bench_gpt` process, untuned default pipeline, f32, the treatment as flags on
one build, run by [gh1002_cells.py](gh1002_cells.py) from `benchmarks/` with every `OCANNL_*`,
`BENCH_*` and `OMP_*` variable cleared and the schedule cache disabled; `gh1002_cells.py run` then
`summarize` reproduces every table below. The training step is `Train.grad_update` plus SGD in
one routine; the protocol is the fixture's: 6 parity steps (the loss trajectory), 3 warmup, 10
per-step-synced timed steps (the p10/p50/p90 below), 10 queued. Repeats are order-balanced: repeat
*r* runs the treatments forward for even *r*, reversed for odd *r* — three repeats on Metal, two
on cc, two on the Metal sweep, one on the cc sweep. The spread column is the widest p90/p10 of
any repeat of the cell.

| Treatment | Forward | Backward | Flags |
|---|---|---|---|
| A | composed | composed | `online_softmax=false`, `online_softmax_backward=false` |
| B | online, scores stored (default cap) | composed | `online_softmax=true` |
| C | online, scores recomputed (cap raised) | composed | B + `virtualize_max_inline_reduction=32` |
| D1 | online, scores stored | **fused** | `online_softmax=true`, `online_softmax_backward=true` |
| D2 | online, scores recomputed | **fused** | D1 + `virtualize_max_inline_reduction=32` |

Head width is 32, above the default recompute cap of 16, so B and D1 store the scores as one
`[seq, seq]` buffer per layer and C and D2 recompute them. E and F (the block-tiled forward) are
gh-ocannl-1003's.

Workloads: the `gpt2_mini_train` recipe (4 layers, d 256, 8 heads, vocab 1024, SGD lr 0.01) at
`gpt2_mini_train` (8 x 128), `gpt2_mini_train_s512` (2 x 512) and `gpt2_mini_train_s1024`
(1 x 1024) — 1024 tokens per step — and, for the sequence sweep at fixed batch,
`gpt2_mini_train_b1_s128/_s256/_s512` with `_s1024`. Fixtures are the m4-max `content-v1` entries
of `fixtures/DIGESTS.txt` (checked by the driver before any cell ran).

Memory is the result line's `peak_memory_bytes`: `Alloc_census.reset_peak` after the warmup,
`peak_pool_bytes` after the timed steps — the requested bytes at the allocator seam over the
workload's steady-state window (everything held at the window's start plus anything the steps
allocate), never a `get_used_memory` reading and never a tuning candidate. It was identical across
the repeats of every cell. Attribution is the dominant-kernel instrument's per-kernel table
(`BENCH_KERNEL_TABLE=1`, Metal only): every kernel the step shipped, compiled from its shipped IR
and timed alone min-of-20 with a sync per run after the timed steps — so each kernel carries a
~0.15 ms launch floor and the kernel sums exceed step times. The attention forward block is the
kernels from the query projection through the value pass; the backward block runs from the kernel
writing `w_o.grad` through the one writing `w_q.grad` (w_o's gradient shares a kernel with `dP` or
dQ, and the q/k/v projection gradients are the same work in every treatment). No cell requested a
tensorized contraction (untuned default: every kernel's MMA census is `not-requested`).

## Step times, Metal

| fixture | treatment | p50 per repeat (ms) | median p50 | vs A | vs B | tokens/s | p10..p90 spread |
|---|---|---|---|---|---|---|---|
| `gpt2_mini_train` (8 x 128) | A | 206.8, 206.5, 206.3 | 206.5 | 1.000x | 1.065x | 4958 | up to 1.007x |
| | B | 194.8, 193.5, 194.0 | 194.0 | 0.939x | 1.000x | 5279 | up to 1.009x |
| | C | 192.9, 192.4, 192.6 | 192.6 | 0.932x | 0.993x | 5318 | up to 1.012x |
| | D1 | 198.8, 197.3, 197.0 | 197.3 | 0.955x | 1.017x | 5190 | up to 1.013x |
| | D2 | 213.6, 214.1, 213.5 | 213.6 | 1.034x | **1.101x** | 4795 | up to 1.007x |
| `gpt2_mini_train_s512` (2 x 512) | A | 226.5, 226.6, 226.7 | 226.6 | 1.000x | 1.133x | 4518 | up to 1.003x |
| | B | 200.0, 200.0, 200.0 | 200.0 | 0.883x | 1.000x | 5119 | up to 1.003x |
| | C | 190.4, 190.4, 190.5 | 190.4 | 0.840x | 0.952x | 5379 | up to 1.004x |
| | D1 | 238.7, 238.1, 237.9 | 238.1 | 1.051x | **1.190x** | 4301 | up to 1.008x |
| | D2 | 269.5, 269.5, 269.4 | 269.5 | 1.189x | **1.347x** | 3799 | up to 1.001x |
| `gpt2_mini_train_s1024` (1 x 1024) | A | 536.2, 536.3, 536.1 | 536.2 | 1.000x | 3.237x | 1910 | up to 1.002x |
| | B | 165.5, 165.9, 165.7 | 165.7 | 0.309x | 1.000x | 6181 | up to 1.008x |
| | C | 158.5, 158.9, 158.5 | 158.5 | 0.296x | 0.957x | 6461 | up to 1.020x |
| | D1 | 253.6, 253.7, 253.7 | 253.7 | 0.473x | **1.531x** | 4037 | up to 1.003x |
| | D2 | 308.1, 308.1, 308.2 | 308.1 | 0.575x | **1.860x** | 3323 | up to 1.002x |

Bold marks a fused-backward ratio against B outside 5%. Every Metal cell's repeats agree to 0.8%
and every spread is under 2.1%, so each bold ratio is far outside run noise; D1 at seq 128
(1.017x) is within it. Compile times are 0.9-1.4 s in every cell. Kernels per step: A 159/159/180,
B 149/149/170, C 145/145/166, D1 141/141/162, D2 137/137/158 (seq 128/512/1024).

## Peak requested memory (identical on Metal and cc to 0.1 MiB)

| fixture | A | B | C | D1 | D2 |
|---|---|---|---|---|---|
| `gpt2_mini_train` (8 x 128) | 236.2 MiB | 236.2 (1.000x) | 204.2 (0.865x) | **188.1 (0.796x)** | **172.1 (0.729x)** |
| `gpt2_mini_train_s512` (2 x 512) | 429.9 | 429.9 (1.000x) | 301.9 (0.702x) | **237.8 (0.553x)** | **173.8 (0.404x)** |
| `gpt2_mini_train_s1024` (1 x 1024) | 688.9 | 688.9 (1.000x) | 432.9 (0.628x) | **304.8 (0.442x)** | **176.8 (0.257x)** |

- **The online forward alone saves no training memory** (B = A in every cell): the composed
  backward still reads the probabilities, so `P` stays stored, and `dP` and `dS` are written as
  before. Raising the recompute cap (C) drops the scores and `P` but keeps `dP` and `dS`.
- **The fused backward removes what the record promised.** Each `[seq, seq]` buffer is
  `seq^2 x 8 heads x 4 B` per layer — 32 MiB at seq 1024, 128 MiB over four layers. A - D1 is three
  of them per layer (384.1 MiB: `P`, `dP`, `dS`), D1 - D2 the stored scores (128.0 MiB), and D2 is
  flat in sequence length at fixed tokens (172-177 MiB).

## Sequence sweep at fixed batch 1

Peak requested memory (MiB) / p50 (ms); the quadratic term is a least-squares `a + b s + c s^2`
fit over the four lengths, in units of one `[seq, seq]` f32 buffer per head per layer
(`c / (4 B x 8 heads x 4 layers)`).

| backend | treatment | seq 128 | seq 256 | seq 512 | seq 1024 | quadratic buffers per head per layer |
|---|---|---|---|---|---|---|
| Metal | A | 56.0 / 44 | 98.0 / 82 | 230.4 / 187 | 688.9 / 536 | 4.03 |
| | B | 56.0 / 32 | 98.0 / 45 | 230.4 / 76 | 688.9 / 166 | 4.03 |
| | C | 52.0 / 28 | 82.0 / 42 | 166.4 / 72 | 432.9 / 158 | 2.03 |
| | D1 | 50.0 / 30 | 74.0 / 47 | 134.4 / 94 | 304.8 / 254 | 1.03 |
| | D2 | 48.0 / 31 | 66.0 / 51 | 102.4 / 111 | 176.8 / 308 | **0.03** |
| cc | A | 56.0 / 294 | 98.0 / 782 | 230.4 / 2107 | 688.8 / 6542 | 4.03 |
| | B | 56.0 / 280 | 98.0 / 600 | 230.4 / 1326 | 688.8 / 3368 | 4.03 |
| | C | 52.0 / 283 | 82.0 / 578 | 166.4 / 1247 | 432.8 / 2946 | 2.03 |
| | D1 | 49.9 / 280 | 73.9 / 573 | 134.3 / 1254 | 304.7 / 2992 | 1.03 |
| | D2 | 47.9 / 292 | 65.9 / 623 | 102.3 / 1433 | 176.7 / 3764 | **0.03** |

The fit separates quadratic from linear state cleanly: the composed step keeps four
`[seq, seq]` buffers per head per layer (the scores, `P`, `dP`, `dS`), C two, D1 one (the stored
scores), and D2's remaining 0.03 is exactly the user-supplied causal mask (one `[seq, seq]` f32
constant, 4 B x seq^2, i.e. 1/32 of a per-head-per-layer buffer), which is not attention scratch.
Everything else D2 holds is linear in `seq` (activations saved for the backward, the `(m, l)` row
state, the minted per-row `D`) or independent of it (parameters and their gradients). Time on the
Metal sweep: D1 against B is 0.92x at seq 128, 1.05x at 256, 1.23x at 512 and 1.53x at 1024 — the
fused kernels' cost grows with `seq` faster than the composed kernels'. (B's seq-128 cell is the
one noisy cell of the run: repeats 35.2 and 29.1 ms, spread 1.30x; the other sweep cells are
within 6.4%.)

## Where the Metal time goes (attribution)

Attention kernels summed over the four layers, ms, median over repeats:

| fixture | treatment | all kernels | attention forward | attention backward |
|---|---|---|---|---|
| `gpt2_mini_train` | A | 223.7 | 40.8 | 93.6 |
| | B | 209.2 | 35.6 | 85.1 |
| | C | 210.9 | 36.4 | 86.2 |
| | D1 | 214.0 | 34.8 | 90.4 |
| | D2 | 233.8 | 47.7 | 97.2 |
| `gpt2_mini_train_s512` | A | 242.8 | 26.7 | 89.0 |
| | B | 216.3 | 19.7 | 69.4 |
| | C | 208.5 | 21.4 | 60.2 |
| | D1 | 254.4 | 21.5 | **106.8** |
| | D2 | 287.5 | 28.7 | **132.2** |
| `gpt2_mini_train_s1024` | A | 545.7 | 40.1 | 435.7 |
| | B | 182.8 | 23.8 | 89.5 |
| | C | 176.6 | 27.7 | 80.0 |
| | D1 | 268.3 | 27.6 | **172.3** |
| | D2 | 324.2 | 33.9 | **222.0** |

Layer 0's attention backward, kernel by kernel (repeat 0; launch geometry as grid x block):

| fixture | B (composed backward) | D1 (fused backward) |
|---|---|---|
| seq 128 | `w_o.grad`+`dP`+`dO` 10.77 ms (256 x 128); `v.grad` 0.29 ms (lanes: 128x8x8 x 32); `w_v.grad`+`q.grad`+`dS`.. 4.96 ms; `k.grad` 2.29 ms; `w_q`/`w_k.grad` 3.01 ms — **21.3 ms** | `w_o.grad`+`dO`+`D`+dQ 11.32 ms (256 x 128); **dK+dV 7.17 ms (8 x 128)**; `w_q`/`w_k`/`w_v.grad` 4.37 ms — **22.9 ms** |
| seq 512 | 8.37; `v.grad` 0.60 (lanes); 4.16; 1.85; 2.38 — **17.4 ms** | 6.71 (512 x 8); **dK+dV 16.49 (8 x 256)**; 3.59 — **26.8 ms** |
| seq 1024 | 11.16; `v.grad` 1.38 (lanes); 5.37; 2.40; 2.26 — **22.6 ms** | 6.51 (1024 x 8); **dK+dV 33.10 (8 x 256)**; 3.47 — **43.1 ms** |

- **The whole Metal time cost is the dK+dV kernel.** The dQ side is no slower than the composed
  `dP` it replaces (at seq 512-1024 it is faster: 6.5-6.7 ms against 8.4-11.2 ms, since `dP` is no
  longer written). The dK and dV nests are conflict-free, so fission puts them in one kernel; the
  stage-1 lane geometry declines any kernel that mixes a lane nest (dV, whose preamble is `p`
  alone) with a plain one (dK, whose preamble holds `dp`'s value-width loop), and the presets give
  the pair two chain loops — 1024 threads at seq 128, 2048 at seq 512 and 1024 — each thread
  walking every query row and, per row, the key and value widths plus `dp`'s reduction, serially.
  The composed form runs `v.grad` on lanes. At seq 1024 the kernel is 33.1 ms against B's 3.8 ms
  for `v.grad` plus `k.grad`; over four layers the difference (~82 ms) is the step's (88 ms).
- **Recomputing the scores costs the fused backward more than the composed one.** D2's dQ, dK and
  dV nests each re-derive `q . k` per pair (the dK+dV kernel is 44.5 ms at seq 1024 against D1's
  33.1), so D2 trades 128 MiB for 54 ms at seq 1024, where C gains time over B.
- **Not a serial kernel.** gh-ocannl-1124 point 1 reported the fused backward in an all-serial
  Metal kernel; that was the `online_softmax` test's toy size (seq 7), where the kernel's largest
  parallel chain is below `gpu_schedule_min_parallel` and the preset declines the whole kernel.
  At every size measured here each backward nest carries geometry, and
  `test/operations/gpu_serial_lanes` leg 6 now pins it at seq 64. What the kernels lack is lanes
  and thread count, which is #1124's remaining point.
- **The composed step's seq-1024 pathology is not the quadratic buffers.** A's 536 ms at seq 1024
  is `v.grad`: the composed `v.grad` nest shares a kernel with `w_v.grad` at 256 threads (86 ms per
  layer). The online forward's hoist hands that nest to the stage-1 lanes (B: 1.4 ms), which is why
  every online treatment is 2-3x faster than A there. Filed separately as gh-ocannl-1126.

## cc

| fixture | treatment | p50 per repeat (ms) | median | vs A | vs B | p10..p90 spread |
|---|---|---|---|---|---|---|
| `gpt2_mini_train` | A | 3074, 3263 | 3168 | 1.000x | 1.050x | up to 1.062x |
| | B | 3076, 2961 | 3019 | 0.953x | 1.000x | up to 1.107x |
| | C | 3076, 3009 | 3043 | 0.960x | 1.008x | up to 1.050x |
| | D1 | 3166, 2935 | 3050 | 0.963x | 1.010x | up to 1.033x |
| | D2 | 3050, 3050 | 3050 | 0.963x | 1.010x | up to 1.004x |
| `gpt2_mini_train_s512` | A | 6135, 5083 | 5609 | 1.000x | 1.584x | up to 1.293x |
| | B | 3556, 3527 | 3542 | 0.631x | 1.000x | up to 1.187x |
| | C | 3616, 3604 | 3610 | 0.644x | 1.019x | up to 1.009x |
| | D1 | 3333, 3332 | 3333 | 0.594x | **0.941x** | up to 1.010x |
| | D2 | 3730, 3912 | 3821 | 0.681x | 1.079x | up to 1.009x |
| `gpt2_mini_train_s1024` | A | 6429, 6655 | 6542 | 1.000x | 1.942x | up to 1.068x |
| | B | 3316, 3420 | 3368 | 0.515x | 1.000x | up to 1.091x |
| | C | 2933, 2959 | 2946 | 0.450x | 0.875x | up to 1.015x |
| | D1 | 2985, 3000 | 2993 | 0.457x | **0.889x** | up to 1.092x |
| | D2 | 3801, 3726 | 3764 | 0.575x | 1.117x | up to 1.039x |

On cc the fused backward with stored scores is the fastest form at seq 512 and ties C for it at
seq 1024 (the 1.6% between them is inside D1's 9% spread); at seq 128 all four online treatments
are within the cell spreads of each other. A's seq-512 repeats disagree by 20% (one disturbed
window, spread 1.29x), so A's ratios there are indicative only; nothing else depends on them. The
CPU preset parallelizes one outermost loop per nest across the pool, which the fused nests' row
and key owners fill, and not writing and re-reading `dP`/`dS` is a bandwidth saving the cores
feel. The batch-1 sweep agrees (D1 vs B: 0.998x, 0.956x, 0.946x, 0.889x at seq 128-1024).
Recomputing the scores (D2) loses on cc as on Metal. Compile times: 4-13 s per cell, the fused
treatments' slightly shorter (fewer kernels).

## Parity

Worst relative difference of the six-step loss trajectory against treatment A of the same backend
and repeat, over every fixture: B and C 1.36e-7, D1 and D2 **2.03e-7** (cc, `gpt2_mini_train`;
1.4e-7 or less everywhere else). Treatment A against the torch CPU runner on the same fixture:
1.3e-6 at seq 128, 6.7e-7 at 512, 8.1e-7 at 1024, 2.0-4.0e-7 on the sweep. Gradient-level parity
(finite differences, masked prefixes, ties, the NaN row) is pinned by
`test/operations/online_softmax` legs 9-11, and the executed fused-vs-composed gradients at seq 64
on cc and Metal by `gpu_serial_lanes` leg 6.

## What changed between treatments (Metal, seq 1024, layer 0)

- A -> B: the forward's normalizer becomes the `(m, l)` scan, and the value pass is hoisted
  (`p` per key, lanes over the value width). The composed backward is untouched, but its `v.grad`
  nest inherits the hoisted, lane-eligible shape: 86 -> 1.4 ms.
- B -> C: the scores are recomputed instead of stored (with `P`, nothing reads them through a node
  any more): -256 MiB over the four layers at seq 1024; the `w_v.grad`/`q.grad`/`dS` kernel grows
  (5.4 -> 11.1 ms) where it recomputes `q . k`, the `w_o.grad` kernel shrinks (11.2 -> 3.0 ms).
- B -> D1: the composed backward's nests are replaced by `D`, dQ, dK and dV; `P`, `dP`, `dS` are
  never written (-384 MiB); the backward goes from five kernels per layer to three.
- D1 -> D2: the stored scores go too (-128 MiB); every fused nest recomputes `q . k` per pair.

## Not measured here

- **CUDA and HIP.** The GPU boxes were held by an issue wave for the whole window, and the design
  record's rule applies: CUDA/HIP need their own measured rows before any claim about their
  performance. The driver's cells are `bench_gpt` invocations that run there with the backend
  flag changed; the fused kernels' geometry is the shared GPU preset's, so the Metal attribution is
  the expectation to check, not a result.
- **The tuned arm**, reduced precisions, and the block-tiled forward (E and F, gh-ocannl-1003).
- **dQ/dK lanes, and dV kept out of dK's kernel** (gh-ocannl-1124): the measured schedule is the
  landed default. The attribution says what the follow-up is worth — at seq 1024 the dK+dV kernel
  is 33 ms of D1's 43 ms per layer of attention backward.
- **cc attribution.** The kernel table compiles every kernel alone; on cc that is one C compile
  per kernel, so cc's time split is not measured (its step times are).
- The composed `v.grad` pathology at seq 1024 is reported, not fixed.

## Reproducing

`dune build benchmarks/runners/ocannl/bench_gpt.exe`, then `python3 benchmarks/gh1002_cells.py run
--out DIR` (85 minutes on this box for both backends, the sweeps and the artifact pass) and
`python3 benchmarks/gh1002_cells.py summarize --out DIR`. Phases run separately with
`--phases metal,metal-sweep`, and a rerun resumes at the first cell without a result line (from
the same revision and fixture bytes only); `summarize` refuses an incomplete matrix unless
`--partial`. The
artifact pass keeps each Metal cell's generated sources under `DIR/artifacts/` (the published
run's pass, from an earlier driver revision, covered the main matrix's fixtures only).

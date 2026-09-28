# gh-ocannl-1003: the block-tiled online attention fold on Metal and cc

Measurement report for the closing PR of ahrefs/ocannl#1003 (design record
[docs/proposals/gh-ocannl-1002-1003.md](../docs/proposals/gh-ocannl-1002-1003.md), "1003: a
single-pass, block-tiled forward fold"). The fold is the online-softmax rewrite's fourth shape
(`Ir.Online_softmax.find_fold`, lukstafi/ocannl-staging#892), behind the key
`online_softmax_block` (a key-block size `B`; 0 keeps the two-pass form): per query row one scan
over the key blocks carrying the row's running max and sum, the block's scores in a tile, the
output numerator rescaled and accumulated in a second tile, the scores read once and never
stored. On a GPU whose matrix units take f32, the default schedule renders the fold cooperatively
(`Schedule.Fold_mma`, lukstafi/ocannl-staging#905): lanes are query rows, the two tiles live in
workgroup-shared memory, and the score slice and the value update are each one block `Tile_mma`.
This report measures the fold against the composed attention and the two-pass online form that
[report-gh1003-stage1.md](report-gh1003-stage1.md) left as the best online arm, and decides the
`approximate` profile's value of the key.

**Verdict.** On Metal the fold is the fastest attention form at every length measured, and the
gap widens with the sequence: with `B = 16` the inference step is **0.821x of composed at seq 128,
0.790x at seq 512 and 0.483x at seq 1024** (0.889x, 0.846x and 0.627x of the two-pass form), with
every cell's three repeats agreeing to 1% and all of them on the same side. Per layer, the
attention that the composed form spends three kernels and 3.2-7.7 ms on, and the two-pass form
two kernels and 2.1-4.5 ms on, is one kernel of 0.36 ms at seq 128 and 0.82 ms at seq 1024, and the
`seg` phase's census reads that kernel as `Mma_intrinsics x2` in every layer of every fold cell:
both contractions ran on the matrix units. The fold also removes the `[seq, seq]` buffer at any
recompute cap: the inference step's peak requested memory is 91-96 MiB at every length, against
124-353 MiB composed and 108-225 MiB two-pass. The key block sizes 8, 16 and 32 are within the
cells' spreads of each other at seq 128 and 512; at seq 1024 `B = 16` wins by 3.7% over 8 and 6.5%
over 32, outside the spreads, so 16 is the value the profile takes. On cc the fold is the scalar
single pass and ties the two-pass form within 1.3% at seq 128 and 512 -- no regression. In
training, the fold under the fused backward (treatment F) is 0.976x of the two-pass forward under
the same backward (D) at seq 1024 and 0.986x at seq 512: the step is backward-bound
(gh-ocannl-1124) and the fold's forward win is a few percent of it, with the peak memory unchanged
from D's because the backward still reads the stored scores. Every cell reproduces the composed
form's losses to 1.3e-7 relative. CUDA and HIP have not run the fold.

## Protocol

Hardware: Apple M4 Max (40-core GPU), macOS 26.6.2 (Darwin 25.6.0), host `mac-studio`, under an
exclusive fleet `measurement` reservation held for this run (2026-09-28, 21:06Z-22:05Z); the
driver's host snapshot at the start shows load 3.84 with the busiest processes at 2% CPU (desktop
agents), and no other fleet batch on the box. Tree: staging master `e9fe65e0e` (the merge of
lukstafi/ocannl-staging#905, the head of the arc), clean -- the driver refuses a dirty tree -- in a
checkout of its own. Fixtures `gpt2_mini` (8 x 128), `gpt2_mini_s512` (2 x 512), `gpt2_mini_s1024`
(1 x 1024), `gpt2_mini_train_s512` (2 x 512) and `gpt2_mini_train_s1024` (1 x 1024), 1024 tokens
per step each, matched at run time against the m4-max `content-v1` rows of `fixtures/DIGESTS.txt`.

The whole run is one invocation of the driver that landed with the scalar fold, from the
checkout's `benchmarks/`:

```
REPEATS=3 FOLD_BLOCKS="8 16 32" FIXTURE_DIR=<fixtures> \
  benchmarks/gh1003_fold_cells.sh <out> 900 build provenance dry metal seg cc train summary
```

Every cell is one `bench_gpt` process on the fixture named, untuned default pipeline
(`--ocannl_autotune_search=false`), f32, schedule fission on, the default GPU schedule, no debug
artifacts, with `OCANNL_*`, `BENCH_*` and the OpenMP variables cleared and the treatment pinned on
the command line:

| treatment | flags | record |
|---|---|---|
| composed | the defaults | A |
| two-pass | `--ocannl_online_softmax=true` (head width 32 exceeds the recompute cap 16, so the scores stay one stored `[seq, seq]` buffer per layer, read by the scan and the value pass) | B |
| fold-B | two-pass plus `--ocannl_online_softmax_block=B`, `B` in 8, 16, 32 | E (inference) |
| two-pass-bwd | two-pass plus `--ocannl_online_softmax_backward=true` | D1 |
| fold-B-bwd | fold-B plus `--ocannl_online_softmax_backward=true` | F |

Steps: `dry` (one Metal cell per treatment on `gpt2_mini`, a smoke of the matrix, not summarized),
`metal` (three passes over the 3 x 5 cells, forward / reversed / forward, so an order effect
would split the repeats of one cell), `seg` (one `bench_gpt_diag` cell per fixture x treatment
with `BENCH_SEG_TIMES=1`: every fission segment compiled as its own routine and timed min-of-20
with a sync per run, so each segment carries a ~0.15-0.20 ms launch floor and the segment sums
exceed the step times), `cc` (two passes, forward then reversed, over `gpt2_mini` and
`gpt2_mini_s512`), `train` (two passes over the two training fixtures x composed, two-pass-bwd
and the three fold-B-bwd). The timed window per cell is the fixture's: 8 parity steps and 20
per-step-synced timed steps at seq 128, 4 and 10 on the long legs, 6 and 10 in training. Each
cell's p50 is its own; the table's median is over the cell's repeats; the spread column is the
widest p90/p10 of any repeat of the cell; the ratios are between the medians. The raw cells,
`driver.log` and `summary.md` are the run's artifacts (kept beside the checkout that ran them).

## Step times, Metal inference

| fixture | treatment | p50 per repeat (ms) | median p50 | vs composed | vs two-pass | tokens/s | peak MiB | kernels | spread |
|---|---|---|---|---|---|---|---|---|---|
| `gpt2_mini` (8 x 128) | composed | 62.6, 62.5, 62.4 | 62.5 | 1.000x | 1.083x | 16384 | 123.7 | 117 | up to 1.013x |
| | two-pass | 57.5, 57.7, 57.8 | 57.7 | 0.923x | 1.000x | 17747 | 107.7 | 107 | up to 1.020x |
| | fold-8 | 51.3, 51.3, 51.4 | 51.3 | **0.821x** | **0.890x** | 19961 | 91.4 | 95 | up to 1.021x |
| | fold-16 | 51.3, 51.3, 51.3 | 51.3 | **0.821x** | **0.889x** | 19961 | 91.4 | 95 | up to 1.026x |
| | fold-32 | 51.5, 51.5, 51.5 | 51.5 | **0.824x** | **0.893x** | 19883 | 91.4 | 95 | up to 1.017x |
| `gpt2_mini_s512` (2 x 512) | composed | 62.0, 62.1, 62.1 | 62.1 | 1.000x | 1.070x | 16490 | 220.9 | 117 | up to 1.005x |
| | two-pass | 58.0, 57.9, 58.0 | 58.0 | 0.935x | 1.000x | 17655 | 156.9 | 107 | up to 1.007x |
| | fold-8 | 49.3, 49.3, 49.3 | 49.3 | **0.794x** | **0.850x** | 20771 | 92.7 | 95 | up to 1.008x |
| | fold-16 | 49.1, 48.9, 49.0 | 49.0 | **0.790x** | **0.846x** | 20898 | 92.7 | 95 | up to 1.014x |
| | fold-32 | 49.7, 49.8, 49.7 | 49.7 | **0.802x** | **0.858x** | 20604 | 92.7 | 95 | up to 1.014x |
| `gpt2_mini_s1024` (1 x 1024) | composed | 50.6, 50.9, 50.8 | 50.8 | 1.000x | 1.300x | 20157 | 352.5 | 128 | up to 1.008x |
| | two-pass | 39.2, 39.1, 39.0 | 39.1 | 0.769x | 1.000x | 26189 | 224.5 | 118 | up to 1.009x |
| | fold-8 | 25.4, 25.4, 25.3 | 25.4 | **0.499x** | **0.649x** | 40315 | 96.2 | 106 | up to 1.016x |
| | fold-16 | 24.7, 24.5, 24.5 | 24.5 | **0.483x** | **0.627x** | 41796 | 96.2 | 106 | up to 1.017x |
| | fold-32 | 26.0, 26.2, 26.1 | 26.1 | **0.514x** | **0.668x** | 39234 | 96.2 | 106 | up to 1.010x |

Bold marks a fold ratio outside 5% of its reference; every bold ratio is far outside its cell's
spread, with all three repeats on the same side. `tokens/s` is 1024 tokens over the median p50;
`peak MiB` is the result line's `peak_memory_bytes` (`Alloc_census.reset_peak` after the warmup,
`peak_pool_bytes` after the timed steps: the requested bytes at the allocator seam over the timed
window), identical across the repeats of every cell; `kernels` is the fission segment count of
the step. Compile times are 0.13-0.99 s in every cell.

- **The composed and two-pass rows reproduce the stage-1 report** (62.4 / 62.0 / 51.0 ms composed,
  57.6 / 58.2 / 39.3 ms stored-scores online, on that report's landed tree): the baseline has not
  moved between the two runs, so the fold's ratios are against the same target.
- **The fold's win grows with the sequence** because what it removes grows with it: at seq 128 the
  attention is a small part of a step dominated by the projections and the logits (the
  `dominant_kernel` of every seq-128 and seq-512 cell is the 8.2-24.4 ms logits kernel, untouched
  by any treatment), at seq 1024 the composed form's `[seq, seq]` value pass alone is 4.9 ms per
  layer.
- **The `[seq, seq]` buffer is gone at any recompute cap.** The fold's peak is 91.4, 92.7 and
  96.2 MiB at seq 128, 512 and 1024 -- flat in the sequence at fixed tokens -- where the two-pass
  form still holds one stored score buffer per layer (107.7 / 156.9 / 224.5 MiB) and the composed
  form two (123.7 / 220.9 / 352.5 MiB). The kernel count drops by 22 from composed -- per layer
  the three attention kernels and their three zero-init launches become one kernel with none
  (20), and the gate rewrites the cross-entropy loss's softmax too (2, shared with two-pass) --
  and by 12 from two-pass (per layer four kernels to one).

## Where the time goes: per-segment attribution, Metal

`seg` phase, layer 0 of four (the other layers agree to the launch floor unless noted), ms per
segment, min of 20 with a sync per run:

| fixture | form | scores (+ max / + scan) | softmax | value pass | fold (both contractions) | attention kernels per layer | zero-init segments per layer | all segments, sum |
|---|---|---|---|---|---|---|---|---|
| `gpt2_mini` (8 x 128) | composed | 1.38 | 0.72 | 1.12 | -- | 3.22 (3 kernels) | 3 | 75.5 |
| | two-pass | 1.38 | -- | 0.71 | -- | 2.09 (2 kernels) | 2 | 70.5 |
| | fold-8 | -- | -- | -- | 0.36 | 0.36 (1 kernel) | 0 | 62.5 |
| | fold-16 | -- | -- | -- | **0.36** | 0.36 | 0 | 62.2 |
| | fold-32 | -- | -- | -- | 0.45 | 0.45 | 0 | 62.6 |
| `gpt2_mini_s512` (2 x 512) | composed | 1.42 | 0.59 | 1.74 | -- | 3.75 | 3 | 81.5 |
| | two-pass | 1.44 | -- | 1.33 | -- | 2.77 | 2 | 75.6 |
| | fold-8 | -- | -- | -- | 0.57 (layers 1-3: 0.58-0.59) | 0.57 | 0 | 65.8 |
| | fold-16 | -- | -- | -- | **0.52** (layers 1-3: 0.79-0.82) | 0.52 | 0 | 65.7 |
| | fold-32 | -- | -- | -- | 0.74 (layers 1-3: 0.73-0.77) | 0.74 | 0 | 63.8 |
| `gpt2_mini_s1024` (1 x 1024) | composed | 2.10 | 0.66 | 4.90 | -- | 7.67 | 3 | 68.5 |
| | two-pass | 2.15 | -- | 2.37 | -- | 4.52 | 2 | 55.1 |
| | fold-8 | -- | -- | -- | 0.96 (layers 1-3: 0.92-0.96) | 0.96 | 0 | 38.8 |
| | fold-16 | -- | -- | -- | **0.82** (layers 1-3: 0.80-0.82) | 0.82 | 0 | 38.3 |
| | fold-32 | -- | -- | -- | 1.26 (layers 1-3: 1.21-1.24) | 1.26 | 0 | 39.7 |

- **The fold replaces the attention's two or three kernels by one, and that kernel is cheaper than
  any of them.** At seq 128 the composed scores-plus-max kernel alone is 1.38 ms; the whole fold is
  0.36 ms. At seq 1024 the two-pass form's scores-plus-scan kernel (one row per thread, the serial
  key loop the stage-1 report named as the next cost) is 2.15 ms and its lane-scheduled value pass
  2.37 ms; the fold is 0.82 ms. The zero-init segments go with them: the fold's tiles are
  workgroup-shared scratch the kernel initializes itself, where the composed form zeroed the
  scores, the normalizer and the value-pass output in a launch each, and the two-pass form the
  scores and the value-pass output.
- **The rest of the step is the same work in every treatment.** The segment sums differ between
  forms by the attention kernels' four layers plus the removed zero-init launches; at seq 1024
  the fold's remaining segments are the projections, the FFN (the `dominant_kernel` of the fold
  cells there is the 1.43 ms GELU-fused FFN kernel) and the logits.
- **Segment times are not step times.** Every segment carries the launch floor (the zero-init
  segments read 0.15-0.20 ms for nothing), so the sums exceed the step medians by 10-40 ms; and a
  single `seg` cell is one measurement per segment (min of 20 runs of that segment), which is why
  `fold-32` at seq 512 shows the lowest sum and the highest step, and why fold-16's layer-0 segment
  at seq 512 (0.52 ms) reads below its other three layers (0.79-0.82 ms): the step table, three
  repeats over the whole step, is the timing-grade comparison; the segment table says where the
  difference sits.

## Both contractions on the matrix units

The `seg` phase reads each segment's MMA census off the compiled routine
(`Context.routine.mma`). In every fold cell -- three fixtures x `B` 8, 16, 32 -- exactly four
segments read `mma:tensorized: Mma_intrinsics x2`, one per layer, each writing the layer's
attention output and the two minted tiles (`rewrite__n<k>_block_scores_where`, the score tile
with its scale/mask chain applied, and `rewrite__n<k>_block_numerator`, the output numerator):
two `Tile_mma` statements per kernel, both rendered as intrinsics, no scalar fallback. Every other
segment of those cells, and every segment of the composed and two-pass cells (117 / 128 and
107 / 118 segments), reads `not-requested`: no other kernel of the untuned pipeline asks for a
tile, so the fold's kernel is the only tensorized one in the step and the census cannot be
mistaken for another site's. This is the same evidence `test/operations/online_softmax_block_mma`
pins at seq 64 and 96 (the per-statement census and `simdgroup_multiply_accumulate` in the
generated source, through `Test_utils.Generated`), now read on the shipped step at seq 128-1024.

The `shipped mma` column of the driver's `summary.md` is empty in every row, by construction
rather than by absence: the summary reads the runner's `tune.shipped_mma`, which `bench_gpt`
emits only when a search ran (`BENCH_TUNE=1`), and the driver pins every cell untuned; the step
cells' `dominant_kernel.tensorization` names the step's dominant kernel only (the logits or the
FFN, `not-requested` in every cell, correctly). The per-contraction status of an untuned matrix is
the `seg` phase's, as above; the driver's header now says so.

## Block size: the per-backend sweep gh-ocannl-1123 asked for

| backend | fixture | fold-8 / fold-16 | fold-32 / fold-16 | cell spreads | fold segment, ms (8 / 16 / 32) |
|---|---|---|---|---|---|
| Metal | `gpt2_mini` (8 x 128) | 1.000x | 1.004x | up to 1.026x | 0.36 / 0.36 / 0.45 |
| Metal | `gpt2_mini_s512` (2 x 512) | 1.006x | 1.014x | up to 1.014x | 0.57 / 0.52 / 0.74 |
| Metal | `gpt2_mini_s1024` (1 x 1024) | **1.037x** | **1.065x** | up to 1.017x | 0.96 / 0.82 / 1.26 |
| cc | `gpt2_mini` (8 x 128) | 1.001x | 1.000x | up to 1.007x | scalar fold |
| cc | `gpt2_mini_s512` (2 x 512) | 1.000x | 0.996x | up to 1.020x | scalar fold |

At seq 128 and 512 the three block sizes are within the cells' spreads of each other on both
backends. At seq 1024 on Metal `B = 16` is the winner outside the spreads (all three repeats of
each cell on the same side: 25.3-25.4, 24.5-24.7, 26.0-26.2 ms), and the fold segment says why:
`B = 32` pays for a wider score tile per key block (1.26 ms against 0.82), `B = 8` for twice the
key blocks (0.96). The crowned geometry is the same on the two backends measured, and a rewrite-time
key expresses it, so this run does not supply the justification #1123 asks for (a crowned block
geometry that differs across backends or shapes by more than the key can express); CUDA and HIP,
where the fold has not run, are the open part of that question.

## cc: the scalar fold, no-regression leg

| fixture | treatment | p50 per repeat (ms) | median p50 | vs composed | vs two-pass | peak MiB | spread |
|---|---|---|---|---|---|---|---|
| `gpt2_mini` (8 x 128) | composed | 2122.7, 2121.8 | 2122.2 | 1.000x | 1.024x | 123.5 | up to 1.005x |
| | two-pass | 2074.3, 2070.2 | 2072.3 | 0.976x | 1.000x | 107.5 | up to 1.006x |
| | fold-8 | 2068.6, 2066.5 | 2067.6 | 0.974x | 0.998x | 91.3 | up to 1.007x |
| | fold-16 | 2066.3, 2065.4 | 2065.8 | 0.973x | 0.997x | 91.3 | up to 1.004x |
| | fold-32 | 2064.8, 2064.7 | 2064.8 | 0.973x | 0.996x | 91.3 | up to 1.003x |
| `gpt2_mini_s512` (2 x 512) | composed | 2483.2, 2480.1 | 2481.7 | 1.000x | 1.137x | 220.8 | up to 1.004x |
| | two-pass | 2185.4, 2180.6 | 2183.0 | 0.880x | 1.000x | 156.8 | up to 1.008x |
| | fold-8 | 2164.6, 2163.0 | 2163.8 | 0.872x | 0.991x | 92.6 | up to 1.003x |
| | fold-16 | 2171.0, 2155.9 | 2163.4 | 0.872x | 0.991x | 92.6 | up to 1.020x |
| | fold-32 | 2154.3, 2155.1 | 2154.7 | 0.868x | 0.987x | 92.6 | up to 1.005x |

On cc the fold runs as scalar code (the cooperative rendering is GPU-only: a `Workgroup` loop
enclosing barriers has no serial rendering), and it ties the two-pass form within 1.3% at both
lengths -- inside or at the edge of the cells' spreads, and every fold cell on the faster side. The
win over composed here is the two-pass form's (the gh-483 report's cc result), not the fold's; what
the fold adds on cc is the memory column, the same flat 91-93 MiB as on Metal. Compile times are
3.7-4.2 s at seq 128 and 1.6-2.5 s at seq 512 (the rewritten forms' shorter, fewer kernels). The
`gpt2_mini_s1024` fixture is not in the driver's cc leg.

## Training: treatment F, Metal

| fixture | treatment | p50 per repeat (ms) | median p50 | vs composed | vs two-pass-bwd | peak MiB | kernels | spread |
|---|---|---|---|---|---|---|---|---|
| `gpt2_mini_train_s512` (2 x 512) | composed (A) | 226.5, 226.8 | 226.6 | 1.000x | 0.953x | 429.9 | 159 | up to 1.003x |
| | two-pass-bwd (D1) | 237.7, 237.7 | 237.7 | 1.049x | 1.000x | 237.8 | 141 | up to 1.005x |
| | fold-8-bwd (F) | 234.8, 234.7 | 234.7 | 1.036x | 0.988x | 237.8 | 137 | up to 1.004x |
| | fold-16-bwd (F) | 234.2, 234.4 | 234.3 | 1.034x | 0.986x | 237.8 | 137 | up to 1.002x |
| | fold-32-bwd (F) | 235.2, 235.0 | 235.1 | 1.037x | 0.989x | 237.8 | 137 | up to 1.002x |
| `gpt2_mini_train_s1024` (1 x 1024) | composed (A) | 536.4, 535.3 | 535.8 | 1.000x | 2.115x | 688.9 | 180 | up to 1.002x |
| | two-pass-bwd (D1) | 253.6, 253.0 | 253.3 | 0.473x | 1.000x | 304.8 | 162 | up to 1.003x |
| | fold-8-bwd (F) | 247.9, 248.0 | 248.0 | 0.463x | 0.979x | 304.8 | 158 | up to 1.002x |
| | fold-16-bwd (F) | 247.2, 247.0 | 247.1 | 0.461x | 0.976x | 304.8 | 158 | up to 1.003x |
| | fold-32-bwd (F) | 248.7, 248.8 | 248.7 | 0.464x | 0.982x | 304.8 | 158 | up to 1.003x |

- **The fold's forward saves a few percent of a step the backward owns.** F is 1.4% faster than
  D1 at seq 512 and 2.4% at seq 1024 (outside the 0.2-0.5% spreads, both repeats on the same
  side), and the composed and D1 rows reproduce the gh-1002 report's (226.6 / 238.1 and
  536.2 / 253.7 ms). At seq 512 the composed step is still the fastest of the five, 3.4% ahead of
  F, because the fused backward costs more on Metal than the composed one at that length
  (1.19x of B in the gh-1002 report; the dK+dV kernel, gh-ocannl-1124): the fold cannot recover
  what the backward spends. At seq 1024 the 2.1x over composed is the composed `v.grad` pathology
  (gh-ocannl-1126), which every online treatment avoids; the fold's share of it is the 2.4%.
- **No memory change from D1**, because the training step still reads the scores: the fused
  backward recomputes the probabilities from the row state but reads the stored score buffer
  (D1, the default cap), so the fold keeps the score chain the inference step drops (a
  definition something later reads moves behind the fold; one nothing reads goes with it). F's
  peak is D1's (237.8 / 304.8 MiB) to the byte, and its kernel count is four fewer (the four
  layers' scan-plus-value-pass pairs folded into one kernel each). Dropping the stored scores in
  training is the fused backward's recompute form (D2), which the gh-1002 report measured as a
  time loss on Metal; the fold does not change that trade.
- Compile times 1.0-2.9 s; the per-kernel table (`BENCH_KERNEL_TABLE`) was not part of this run,
  so the training attribution is the gh-1002 report's plus the inference `seg` phase above.

## Parity

Worst relative difference of the parity-step loss trajectory against the composed cell of the
same backend and fixture, over every repeat pairing: **1.3e-7** (two-pass and every fold-B, on both
backends, inference and training; the exact figures are 6.8e-8, 1.3e-7 or 0 per cell). Same-block
cells agree on losses across repeats exactly. Gradient-level and special-value parity (masked
prefixes, fully masked rows, lowest-finite / NaN / +inf fills, `B` 1..32 over dividing, tailed and
one-block sequences) is pinned by `test/operations/online_softmax_block` on cc and Metal, also
under `cc_backend_fast_math`, and the tensorized kernel's parity with the composed form by
`online_softmax_block_mma`.

## The `approximate` profile: `online_softmax_block=16`

The design record left the payload's value of the key to this report. The evidence for entering:
the fold is a second reassociation under the numerics gate the online softmax already holds (the
value contraction as `(sum p * v) / l` instead of `sum (p / l) * v`, documented on the key), it
changes losses by 1.3e-7 relative -- the same figure as the two-pass form the payload already
flips -- and on the backends measured it is never slower: 0.63-0.89x of the two-pass form on Metal
inference, 0.99x on cc, 0.98-0.99x of D1 in Metal training, and it removes the `[seq, seq]` buffer
from the inference step at every recompute cap. `B = 16` is the median winner of the sweep and the
only value outside the spreads at seq 1024. This PR adds `online_softmax_block=16` to the
`approximate` payload (pinned at 0 in `reproducible`, as before). The caveat, in gh-ocannl-1109's
terms: the approximate payload's CUDA row is already the one nothing has isolated knob by knob,
and this key is untested there -- on CUDA the fold would run as the scalar single pass (the
cooperative rendering requires an f32 intrinsic tile, which CUDA advertises only as tf32), a
form this report measured on cc only. Benchmark rows report the key's resolution under
`regime_knobs` with the payload's other keys, so an ablation can single it out.

## Against the original target

The gh-483 report measured the online form on Metal at 1.033x of composed at seq 128 and 1.080x
at seq 512 (composed 125 and 437 ms) and 0.827x at seq 1024: the short-context loss #1003 set out
to remove. gh-ocannl-995 then rebaselined composed to 62.4 and 62.0 ms without removing the loss
(1.073x, 1.025x on the stage-1 report's base tree), stage 1's lanes turned it into a win
(0.923x, 0.939x, 0.771x), and the fold takes the same three cells to 0.821x, 0.790x and 0.483x --
against today's composed form, which is the target the record says to evaluate once the old one
no longer reproduces. The seq-1024 cell is checked for regression and is the largest win.

## Not measured here

- **CUDA and HIP.** The fold has not run on either: the GPU boxes were held by an issue wave for
  the whole window. On CUDA the default schedule keeps the scalar fold (the cooperative rendering
  needs an exact-f32 intrinsic tile; CUDA's f32 tile is tf32 under its own numerics key), so its
  row is a question about the scalar single pass on a GPU, which this report has no analogue for.
  On HIP the backend's f32 intrinsic advertisement decides which form ships. The driver's cells
  run there with the backend flag changed; the design record's rule stands: no claim about
  their performance before their own rows.
- **cc at seq 1024, cc training, and a training per-kernel attribution** -- the driver's cc leg
  covers the two shorter inference fixtures, its `train` leg is Metal, and the kernel table was
  not requested; the gh-1002 report's cc training rows and Metal kernel tables are the reference
  for the backward's cost.
- **The tuned arm and reduced precisions.** Every cell is the untuned default pipeline at f32;
  the fold's condition is f32 throughout (the probabilities are newly computed f32 data, never
  narrowed to tensorize).
- **Block sizes other than 8, 16, 32**, and **the fragment-resident rescale**: `U` is stored and
  reloaded around each key block by the intrinsic (the record's valid first path); the
  fragment-resident form is a separately measurable step the record leaves open.

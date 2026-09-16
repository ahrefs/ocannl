# gh-ocannl-483: the online-softmax rewrite across sequence lengths, on cc and Metal

Measurement report for the rewrite that lands with this issue (`online_softmax=true`; design
record `docs/proposals/gh-ocannl-483.md`). The question the gh-531 profile left open was scale:
at `gpt2_mini`'s seq 128 the attention-adjacent `seq^2` traffic was 5.4% of a CUDA step, so the
rewrite's prize had to be measured where the `[seq, seq]` intermediates grow quadratically while
the matmul work grows linearly. The two long-context legs `gpt2_mini_s512` and `gpt2_mini_s1024`
(same architecture, batch shrunk to keep 1024 tokens per step) are the trigger workloads the
issue asked for; they land with this report.

**Verdict.** The rewrite is what it was designed to be — memory-optimal, compute-naive — and the
crossover is where the arithmetic said: on the CPU backend it is 3% faster at seq 128, 11% at seq
512 and **30% faster at seq 1024**; on Metal it costs 3–8% at seq 128–512 (one row per thread
with a serial key loop under-occupies the GPU while the composed form's `seq^2` kernels are
still cheap) and is **17% faster at seq 1024**, where the composed form's two `[seq, seq]`
intermediates dominate. Leaving the score matrix to the recompute cap (stored, the default) is
the right default on both backends: recomputing it loses to storing it at every cell measured,
by 2–13%. Every arm reproduces the composed losses to 1.4e-7 relative.

Hardware: Apple M4 Max (40-core GPU), macOS 26.6.2. Tree: this branch at `1e0d9f926` (staging
master `5baf6b297` plus the rewrite after nine review rounds; the commits since changed only the
recognizer's acceptance conditions, not the emitted recurrence — the hot loop measured here is
the one that ships). Every number is `bench_gpt` on the fixture named, untuned default pipeline
(no `BENCH_TUNE`, no `model_default_schedule`), f32, one process per cell, arms dispatched as
command-line flags:

- *composed*: the flags' defaults;
- *online, scores stored*: `--ocannl_online_softmax=true` — the scan and the hoisted probability
  read; the score matrix `q * k^T` stays one stored `[seq, seq]` buffer per layer, since the
  head width 32 exceeds the recompute cap `virtualize_max_inline_reduction=16`;
- *online, scores recomputed*: the above plus `--ocannl_virtualize_max_inline_reduction=32` —
  the flash-attention form, no `[seq, seq]` buffer at all, the scores replayed at their two read
  sites.

Repeats: cc twice (arm order forward, then reversed), Metal three times (forward, reversed,
forward), so an order effect would show as a split between the repeats of one cell. Parity
steps / warmup / timed steps are the specs' (8/5/20 at seq 128, 4/3/10 on the long legs). The
per-repeat numbers are each repeat's synced p50; the spread column is the widest p90/p10 of any
repeat of that cell.

## Step times

### cc

| fixture (batch x seq) | arm | p50 per repeat (ms) | median p50 | vs composed | p10..p90 spread |
|---|---|---|---|---|---|
| `gpt2_mini` (8 x 128) | composed | 2149, 2169 | 2159.1 | 1.000x | up to 1.040x |
| `gpt2_mini` (8 x 128) | online, scores stored | 2073, 2093 | 2083.4 | 0.965x | up to 1.051x |
| `gpt2_mini` (8 x 128) | online, scores recomputed | 2113, 2137 | 2125.0 | 0.984x | up to 1.041x |
| `gpt2_mini_s512` (2 x 512) | composed | 2522, 2527 | 2524.6 | 1.000x | up to 1.032x |
| `gpt2_mini_s512` (2 x 512) | online, scores stored | 2246, 2226 | 2236.4 | **0.886x** | up to 1.029x |
| `gpt2_mini_s512` (2 x 512) | online, scores recomputed | 2480, 2356 | 2418.2 | 0.958x | up to 1.028x |
| `gpt2_mini_s1024` (1 x 1024) | composed | 2403, 2355 | 2379.0 | 1.000x | up to 1.044x |
| `gpt2_mini_s1024` (1 x 1024) | online, scores stored | 1639, 1672 | 1655.4 | **0.696x** | up to 1.037x |
| `gpt2_mini_s1024` (1 x 1024) | online, scores recomputed | 1867, 1926 | 1896.4 | **0.797x** | up to 1.015x |

### Metal

| fixture (batch x seq) | arm | p50 per repeat (ms) | median p50 | vs composed | p10..p90 spread |
|---|---|---|---|---|---|
| `gpt2_mini` (8 x 128) | composed | 125, 125, 128 | 125.0 | 1.000x | up to 1.017x |
| `gpt2_mini` (8 x 128) | online, scores stored | 129, 129, 129 | 129.2 | 1.033x | up to 1.004x |
| `gpt2_mini` (8 x 128) | online, scores recomputed | 131, 131, 131 | 130.6 | 1.044x | up to 1.044x |
| `gpt2_mini_s512` (2 x 512) | composed | 437, 450, 436 | 436.8 | 1.000x | up to 1.025x |
| `gpt2_mini_s512` (2 x 512) | online, scores stored | 472, 472, 471 | 471.6 | **1.080x** | up to 1.045x |
| `gpt2_mini_s512` (2 x 512) | online, scores recomputed | 504, 505, 504 | 504.1 | **1.154x** | up to 1.004x |
| `gpt2_mini_s1024` (1 x 1024) | composed | 53.5, 55.3, 53.7 | 53.7 | 1.000x | up to 1.015x |
| `gpt2_mini_s1024` (1 x 1024) | online, scores stored | 44.4, 45.5, 44.4 | 44.4 | **0.827x** | up to 1.014x |
| `gpt2_mini_s1024` (1 x 1024) | online, scores recomputed | 46.1, 49.4, 45.9 | 46.1 | **0.859x** | up to 1.044x |

Bold marks a ratio outside 5% of the composed arm. The cc `gpt2_mini` cells sit within the
spread of a single arm (up to 5%), so the 0.965x there is a small win at best, not a
measurement of one.

## Parity

The worst relative difference of any parity-step loss against the composed arm's, across every
arm and repeat: 6.8e-8 on cc (all three fixtures), 1.4e-7 on Metal (`gpt2_mini`, `gpt2_mini_s1024`)
and exactly zero on `gpt2_mini_s512`. The rewrite reassociates the normalizer's summation — that
is why it is a numerics policy — and at these shapes the reassociation is invisible at the 7th
digit of the loss.

## Reading the curve

- **cc.** Attention is a third of the step's flops at seq 1024 on this fixture (1024 tokens,
  d_head 32) and the `[seq, seq]` traffic is what the rewrite removes, so the 30% at seq 1024
  is most of the attention bin. Storing the scores beats recomputing them by 13% there and by
  8% at seq 512: the recompute doubles the `q * k^T` dot products, and on the CPU those are the
  expensive part once the buffers are gone.
- **Metal.** The rewritten attention runs one row per thread with a serial loop over the keys
  (the scan is opaque to the schedule ops, gh-ocannl-696), so its occupancy is the row count —
  8192 rows for every fixture here — and its per-row work grows with `seq`. The composed
  form's kernels are `(batch, s, t, head)`-parallel and fast while the `seq^2` buffers are
  small; at seq 1024 those buffers are 32 MB per intermediate per layer and the composed form
  loses 17%. Recomputing the scores costs another 1–7% on top of storing them at every length:
  the serial dot product per (row, key) is exactly what the GPU does not want to do twice.
- **Where the crossover moves.** Both curves are steep in `seq` and flat in batch, so a longer
  context moves the rewrite further ahead and a wider batch does not help the composed form.
  Real flash attention gets its GPU speed from blocking the key axis and running the two
  contractions on tensor cores inside the block; that is the follow-up the design record names,
  and it is the one that would make the rewrite win on Metal at short contexts too.
- **The `gpt2_mini_s512` Metal cells are pathological in every arm** (437 ms for the same token
  count that takes 54 ms at seq 1024 and 125 ms at seq 128): the default GPU schedule is starved
  when the leading batch axis is small, filed as ahrefs/ocannl#995 and independent of the
  rewrite — though the online arm pays more for it (1.08x against 1.03x at seq 128).

## What changed since the first measurement

The first pass of this matrix ran at `a7eca5919`, before review. The recurrence's hot loop then
changed twice: the masking guards became a comparison-free update (both maxima floored at the
format's lowest finite value where they are subtracted) with the composed NaN for an all-masked
row selected at the store, and `cc_backend_fast_math` took `-ffinite-math-only` back (the arms
here run without fast-math, so that change does not reach these numbers). The re-measured
ratios moved by 1–4 points in the rewrite's favor on cc and by under a point on Metal; the
crossover and the storing-versus-recomputing verdict are unchanged, and the one cell the first
pass called inconclusive (cc at seq 512, stored versus recomputed) now separates by 8%.

## Not measured here

- **CUDA and HIP.** The issue's original 5.4% figure is a CUDA profile; this box has neither. The
  arms above are one flag each on the same `bench_gpt`, so the GPU boxes can reproduce the matrix
  as is — with the gh-531 bucket method (`benchmarks/gpt2_bucket.py`, which needs `nsys` or
  `rocprofv3`) for the per-bucket attribution the step-level curve here cannot give.
- **Training.** The rewrite changes the forward only; the composed backward keeps its own
  `[seq, seq]` gradient buffers and reads the forward's intermediates through cross-routine
  splicing (`test/operations/online_softmax.ml` leg 6 pins the gradients agree). A training-mode
  long-context leg would measure that unchanged backward; a fused backward is the follow-up.
- **The autotuned variants, and the `approximate` profile.** Every cell is the untuned default
  pipeline at the numerics defaults. The rewritten attention's value pass is an ordinary loop
  nest the schedule search can tile; the scan's enclosing row loops keep their full menu; the
  scan body itself does not. Under `approximate`, `cc_backend_fast_math` now excludes
  finite-math-only; `test/operations/online_softmax_fast_math` pins the rewrite's claims under
  that flag, but its timings are not in this matrix.

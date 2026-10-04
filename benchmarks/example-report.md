# Benchmark results

> **Fixture provenance: unverifiable** (gh-ocannl-919). Original measured files for this historical Apple report
> were not recovered. Later `content-v1` M4 baselines do not certify these numbers.
> See the [historical provenance audit](README.md#historical-metal-fixture-provenance).

platform: macOS-26.5.2-arm64-arm-64bit-Mach-O arm64 | ocannl commit: e687da82 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager; reduced precisions get their own envelope: bf16 0.004, f16 0.002)
> Checked-in example output (`results/` itself is generated and gitignored): the macOS/Metal leg of
> the gh-ocannl-538 re-measurement sweep, run from scratch (wiped `autotune_cache`) at commit
> `e687da82`, after gh-521, gh-527, gh-532, gh-537, gh-539, gh-540 and gh-541 landed. Apple M4 Max,
> 64 GB, otherwise idle, all cells serial. **99 cells, parity gate green on every one**; 9 further
> cells produced no result and are documented under "The gpt2_mini reduced-precision leg does not
> run" below. Tuned cells use the two-pass protocol (pass 1 = the search, reported as `compile_s`;
> a fresh pass-2 process replays the cached winner for step timings). Regenerate the matrix with
> `benchmarks/.venv/bin/python benchmarks/orchestrate.py --tuned --materialized --precision bf16 f16`
> plus `--no-skip-cells` (see below); the `f16-static` / `f16-gatedN` rows are manual runner legs,
> `BENCH_PRECISION=f16` with `BENCH_STATIC_SCALE=1` / `BENCH_GATE_INTERVAL=N`.
>
> **This table is not comparable cell-by-cell with the version it replaces.** That one predates
> gh-521 (every GPU tensorized candidate failed to compile, so families won by elimination) and
> gh-532 (the unparallelized serial baseline was still dispatched and timed, so it sat in the
> candidate pool). Per gh-ocannl-538's reporting contract, no before/after delta is claimed against
> it; the attribution below rests on same-process comparisons instead.
>
> Searches ran with `autotune_log=true`, so the candidate-level counts reported here come from the
> sweep's own search passes rather than a second round of searches. That inflates the tuned cells'
> `compile_s` slightly (stderr writes during the search); step times come from the pass-2 replay and
> are unaffected. Every search logs `baseline: NOT DISPATCHED, binds no hardware dimension on metal`.
>
> `cifar_conv metal/tuned` was run with `--no-skip-cells`: see "The cifar_conv metal/tuned skip is
> stale" below. Its `SKIP_CELLS` entry has been removed.

## cifar_conv

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | mps | eager | f32 | 2.160 | 2.132 | 2.255 | 1.837 | 0.23 | PASS (1.2e-04) |
| tinygrad | METAL | jit | f32 | 4.341 | 4.323 | 4.362 | 4.059 | 1.37 | PASS (1.2e-04) |
| pytorch | cpu | eager | f32 | 35.564 | 35.167 | 36.086 | 35.391 | 0.07 | REF |
| tinygrad | CPU | jit | f32 | 62.891 | 58.629 | 69.147 | 63.194 | 1.43 | PASS (1.2e-04) |
| ocannl | cc | tuned | f32 | 189.570 | 186.742 | 192.946 | 190.133 | 483.52 | PASS (1.2e-04) |
| ocannl | metal | tuned | f32 | 277.928 | 275.875 | 279.977 | 276.448 | 238.80 | PASS (1.2e-04) |
| ocannl | metal | materialized | f32 | 327.895 | 325.789 | 331.729 | 327.279 | 1.55 | PASS (1.2e-04) |
| ocannl | cc | materialized | f32 | 705.924 | 705.117 | 707.088 | 706.335 | 3.25 | PASS (1.2e-04) |
| ocannl | cc | default | f32 | 785.773 | 784.827 | 786.828 | 785.657 | 2.47 | PASS (1.2e-04) |
| ocannl | metal | default | f32 | 1319.380 | 1315.360 | 1322.370 | 1319.060 | 1.29 | PASS (1.2e-04) |

## cifar_stride

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | mps | eager | f32 | 1.131 | 1.109 | 1.174 | 0.772 | 0.10 | PASS (2.5e-06) |
| tinygrad | METAL | jit | f32 | 1.541 | 1.488 | 1.589 | 1.245 | 1.36 | PASS (3.9e-05) |
| pytorch | cpu | eager | f32 | 20.023 | 19.747 | 20.383 | 20.219 | 0.03 | REF |
| tinygrad | CPU | jit | f32 | 24.953 | 24.787 | 25.397 | 25.016 | 1.39 | PASS (3.9e-05) |
| ocannl | cc | tuned | f32 | 54.568 | 53.977 | 55.397 | 54.714 | 284.54 | PASS (3.9e-05) |
| ocannl | metal | tuned | f32 | 95.954 | 95.238 | 96.578 | 94.753 | 141.22 | PASS (3.9e-05) |
| ocannl | metal | materialized | f32 | 117.873 | 117.047 | 118.488 | 116.535 | 1.47 | PASS (3.9e-05) |
| ocannl | cc | materialized | f32 | 186.883 | 186.445 | 187.484 | 187.023 | 3.00 | PASS (3.9e-05) |
| ocannl | cc | default | f32 | 201.837 | 201.396 | 202.409 | 201.736 | 2.59 | PASS (3.9e-05) |
| ocannl | metal | default | f32 | 258.323 | 256.701 | 260.742 | 257.413 | 1.23 | PASS (3.9e-05) |

## gpt2_mini

Rows are grouped by precision (f32 first), p50-ascending within each group.

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| tinygrad | METAL | jit | f32 | 3.161 | 3.116 | 3.206 | 2.910 | 0.72 | PASS (1.3e-07) | 323,918 |
| pytorch | mps | eager | f32 | 4.552 | 4.474 | 4.613 | 4.324 | 0.06 | PASS (1.3e-07) | 224,950 |
| pytorch | cpu | eager | f32 | 13.391 | 13.195 | 13.654 | 13.408 | 0.03 | REF | 76,467 |
| tinygrad | CPU | jit | f32 | 44.360 | 44.094 | 44.637 | 44.483 | 0.81 | PASS (8.0e-07) | 23,084 |
| ocannl | metal | tuned | f32 | 94.776 | 94.455 | 95.084 | 89.804 | 177.93 | PASS (2.0e-07) | 10,804 |
| ocannl | cc | tuned | f32 | 342.372 | 340.657 | 344.190 | 342.928 | 520.17 | PASS (8.1e-07) | 2,991 |
| ocannl | metal | default | f32 | 372.899 | 370.198 | 373.841 | 367.898 | 0.47 | PASS (2.0e-07) | 2,746 |
| ocannl | metal | materialized | f32 | 404.366 | 400.400 | 405.265 | 398.610 | 0.55 | PASS (8.1e-07) | 2,532 |
| ocannl | cc | default | f32 | 2086.770 | 2082.830 | 2089.540 | 2088.120 | 3.78 | PASS (8.1e-07) | 491 |
| ocannl | cc | materialized | f32 | 2262.640 | 2259.510 | 2264.610 | 2262.940 | 15.31 | PASS (8.1e-07) | 453 |
| ocannl | cc | tuned | bf16 | 2258.420 | 2255.360 | 2260.970 | 2257.470 | 1543.66 | PASS (9.9e-04) | 453 |
| ocannl | cc | default | bf16 | 14421.900 | 14415.500 | 14426.500 | 14420.800 | 2.35 | PASS (9.6e-04) | 71 |
| ocannl | cc | materialized | bf16 | 15506.000 | 15502.900 | 15512.200 | 15510.900 | 10.86 | PASS (1.0e-03) | 66 |

## lenet

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| tinygrad | METAL | jit | f32 | 0.780 | 0.748 | 0.791 | 0.562 | 1.31 | PASS (2.1e-07) |
| pytorch | mps | eager | f32 | 5.090 | 5.065 | 5.130 | 4.575 | 0.10 | PASS (1.0e-07) |
| tinygrad | CPU | jit | f32 | 5.710 | 5.561 | 5.817 | 5.673 | 1.33 | PASS (3.1e-07) |
| ocannl | cc | tuned | f32 | 9.769 | 9.296 | 10.311 | 9.873 | 186.12 | PASS (2.1e-07) |
| pytorch | cpu | eager | f32 | 12.457 | 12.151 | 12.755 | 12.438 | 0.02 | REF |
| ocannl | cc | materialized | f32 | 14.368 | 14.240 | 14.489 | 14.385 | 2.09 | PASS (2.1e-07) |
| ocannl | cc | default | f32 | 19.028 | 18.928 | 19.143 | 19.018 | 1.83 | PASS (2.1e-07) |
| ocannl | metal | tuned | f32 | 33.668 | 33.630 | 33.765 | 32.413 | 120.86 | PASS (2.1e-07) |
| ocannl | metal | materialized | f32 | 36.706 | 36.644 | 36.857 | 35.536 | 1.36 | PASS (2.1e-07) |
| ocannl | metal | default | f32 | 250.435 | 250.377 | 250.504 | 249.580 | 1.08 | PASS (2.1e-07) |

## mlp_small

Rows are grouped by precision (f32 first), p50-ascending within each group.

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| ocannl | cc | tuned | f32 | 0.125 | 0.113 | 0.155 | 0.100 | 37.30 | PASS (2.2e-07) |
| ocannl | cc | materialized | f32 | 0.177 | 0.157 | 0.242 | 0.140 | 0.43 | PASS (2.2e-07) |
| pytorch | cpu | eager | f32 | 0.181 | 0.159 | 0.225 | 0.179 | 0.02 | REF |
| ocannl | cc | default | f32 | 0.182 | 0.158 | 0.222 | 0.141 | 0.40 | PASS (2.2e-07) |
| tinygrad | METAL | jit | f32 | 0.291 | 0.265 | 0.314 | 0.244 | 0.36 | PASS (2.4e-07) |
| tinygrad | CPU | jit | f32 | 0.301 | 0.294 | 0.320 | 0.306 | 0.33 | PASS (2.8e-07) |
| pytorch | mps | eager | f32 | 0.535 | 0.511 | 0.597 | 0.275 | 0.08 | PASS (1.2e-07) |
| ocannl | metal | tuned | f32 | 0.688 | 0.662 | 0.793 | 0.458 | 10.15 | PASS (3.2e-07) |
| ocannl | metal | default | f32 | 0.843 | 0.789 | 0.889 | 0.481 | 0.01 | PASS (3.2e-07) |
| ocannl | metal | materialized | f32 | 0.846 | 0.798 | 0.891 | 0.485 | 0.02 | PASS (3.2e-07) |
| ocannl | cc | tuned | bf16 | 0.265 | 0.237 | 0.320 | 0.215 | 26.66 | PASS (7.5e-04) |
| ocannl | metal | tuned | bf16 | 0.683 | 0.652 | 0.788 | 0.479 | 10.49 | PASS (5.4e-04) |
| ocannl | metal | materialized | bf16 | 0.691 | 0.663 | 0.794 | 0.468 | 0.12 | PASS (5.2e-04) |
| ocannl | metal | default | bf16 | 0.694 | 0.663 | 0.797 | 0.473 | 0.14 | PASS (5.4e-04) |
| ocannl | cc | materialized | bf16 | 0.766 | 0.761 | 0.855 | 0.765 | 0.43 | PASS (5.2e-04) |
| ocannl | cc | default | bf16 | 0.779 | 0.773 | 0.869 | 0.779 | 0.43 | PASS (5.4e-04) |
| ocannl | cc | tuned | f16 | 0.267 | 0.217 | 0.341 | 0.220 | 26.50 | PASS (1.5e-04) |
| ocannl | cc | materialized | f16 | 0.328 | 0.296 | 0.405 | 0.271 | 0.82 | PASS (1.5e-04) |
| ocannl | cc | default | f16 | 0.331 | 0.301 | 0.406 | 0.276 | 0.80 | PASS (1.4e-04) |
| ocannl | metal | tuned | f16 | 1.239 | 1.187 | 1.319 | 1.167 | 18.64 | PASS (1.6e-04) |
| ocannl | metal | materialized | f16 | 1.240 | 1.212 | 1.313 | 1.156 | 0.13 | PASS (1.3e-04) |
| ocannl | metal | default | f16 | 1.240 | 1.211 | 1.305 | 1.158 | 0.17 | PASS (1.6e-04) |
| ocannl | cc | default | f16-static | 0.326 | 0.295 | 0.391 | 0.272 | 0.43 | PASS (1.4e-04) |
| ocannl | cc | default | f16-gated32 | 0.326 | 0.299 | 0.399 | 0.274 | 0.47 | PASS (1.4e-04) |
| ocannl | cc | default | f16-gated8 | 0.330 | 0.299 | 0.404 | 0.274 | 0.46 | PASS (1.4e-04) |
| ocannl | metal | default | f16-static | 0.876 | 0.819 | 0.945 | 0.510 | 0.17 | PASS (1.6e-04) |
| ocannl | metal | default | f16-gated8 | 1.068 | 1.041 | 1.207 | 0.683 | 0.19 | PASS (1.6e-04) |
| ocannl | metal | default | f16-gated32 | 1.246 | 1.189 | 1.306 | 0.613 | 0.02 | PASS (1.6e-04) |

## mlp_wide

Rows are grouped by precision (f32 first), p50-ascending within each group.

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | mps | eager | f32 | 0.619 | 0.598 | 0.628 | 0.339 | 0.08 | PASS (2.1e-07) |
| tinygrad | METAL | jit | f32 | 0.763 | 0.694 | 0.806 | 0.713 | 0.53 | PASS (2.1e-07) |
| pytorch | cpu | eager | f32 | 1.601 | 1.574 | 1.631 | 1.597 | 0.01 | REF |
| ocannl | metal | tuned | f32 | 3.632 | 3.598 | 3.668 | 3.052 | 21.27 | PASS (5.2e-07) |
| ocannl | metal | materialized | f32 | 5.414 | 5.370 | 5.478 | 4.826 | 0.06 | PASS (5.2e-07) |
| ocannl | metal | default | f32 | 5.700 | 5.665 | 5.756 | 5.131 | 0.07 | PASS (5.2e-07) |
| tinygrad | CPU | jit | f32 | 13.911 | 13.120 | 14.918 | 14.580 | 0.50 | PASS (7.3e-07) |
| ocannl | cc | tuned | f32 | 34.205 | 33.467 | 34.904 | 34.166 | 82.22 | PASS (5.2e-07) |
| ocannl | cc | materialized | f32 | 219.129 | 218.380 | 219.810 | 219.147 | 0.54 | PASS (6.2e-07) |
| ocannl | cc | default | f32 | 219.148 | 218.408 | 219.898 | 219.354 | 0.51 | PASS (6.2e-07) |
| ocannl | metal | tuned | bf16 | 3.414 | 3.384 | 3.449 | 2.824 | 25.42 | PASS (2.7e-04) |
| ocannl | metal | materialized | bf16 | 5.207 | 5.154 | 5.263 | 4.616 | 0.18 | PASS (2.5e-04) |
| ocannl | metal | default | bf16 | 5.220 | 5.167 | 5.271 | 4.635 | 0.19 | PASS (2.7e-04) |
| ocannl | cc | tuned | bf16 | 71.667 | 70.924 | 72.447 | 71.730 | 172.40 | PASS (2.6e-04) |
| ocannl | cc | materialized | bf16 | 1184.420 | 1183.250 | 1186.470 | 1185.100 | 0.49 | PASS (2.6e-04) |
| ocannl | cc | default | bf16 | 1185.230 | 1183.920 | 1187.020 | 1185.560 | 0.47 | PASS (2.7e-04) |
| ocannl | metal | tuned | f16 | 7.038 | 6.977 | 7.139 | 6.918 | 59.36 | PASS (2.0e-05) |
| ocannl | cc | tuned | f16 | 45.757 | 44.916 | 46.502 | 45.751 | 137.46 | PASS (2.0e-05) |
| ocannl | metal | materialized | f16 | 79.709 | 79.630 | 79.822 | 79.590 | 0.21 | PASS (2.0e-05) |
| ocannl | metal | default | f16 | 79.760 | 79.662 | 79.863 | 79.629 | 0.23 | PASS (2.1e-05) |
| ocannl | cc | materialized | f16 | 592.528 | 591.665 | 593.854 | 592.792 | 0.88 | PASS (2.0e-05) |
| ocannl | cc | default | f16 | 592.893 | 591.731 | 594.318 | 593.021 | 0.85 | PASS (2.0e-05) |
| ocannl | metal | default | f16-static | 5.262 | 5.218 | 5.315 | 4.636 | 0.20 | PASS (2.0e-05) |
| ocannl | metal | default | f16-gated32 | 79.697 | 79.612 | 79.811 | 79.069 | 0.03 | PASS (2.0e-05) |
| ocannl | metal | default | f16-gated8 | 79.703 | 79.617 | 79.912 | 79.163 | 0.23 | PASS (2.0e-05) |
| ocannl | cc | default | f16-static | 592.015 | 591.066 | 593.325 | 592.223 | 0.48 | PASS (2.0e-05) |
| ocannl | cc | default | f16-gated32 | 592.698 | 591.693 | 594.136 | 592.873 | 0.52 | PASS (2.0e-05) |
| ocannl | cc | default | f16-gated8 | 592.777 | 591.779 | 594.349 | 592.959 | 0.53 | PASS (2.0e-05) |

## Findings

Conventions are gh-ocannl-538's reporting contract: every segment share names the placement it is a
share of; a seg-time sum is never presented as a step time; before/after claims are same-session and
same-process; tensor-core reachability is reported as seeded *and* timed counts, never a yes/no.

### Tensor cores on Metal: seeded, timed, crowned once, shipped never

The headline change since the gh-ocannl-476 sweep, where **zero** tensorized candidates were timed
anywhere on Metal because every one failed at candidate compile. They now compile and time. Counts
are per placement arm, from the sweep's own searches, `seeded → timed`:

| workload | metal arm A | metal arm B | cc arm A | cc arm B |
|---|---|---|---|---|
| lenet | 12 → 5 | 18 → 6 | 13 → 6 | 49 → 23 |
| cifar_stride | 12 → 5 | 18 → 6 | 20 → 13 | 55 → 29 |
| cifar_conv | 12 → 5 | 18 → 6 | 24 → 17 | 59 → 33 |
| mlp_small | 19 → 14 | 29 → 17 | 7 → 0 | 47 → 24 |
| mlp_wide | 29 → 11 | 37 → 21 | 27 → 8 | 85 → 13 |
| gpt2_mini | 6 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |

Metal reaches mma at f32 with no precision policy, as expected. The distinction the contract asks
for — a candidate that never compiles versus one carrying an `mma-*` label that renders its scalar
fallback — is clean in almost every cell: **zero** scalar-fallback NOTEs at f32 anywhere, so every
timed count above is a genuinely tensorized pipeline. The exception is Metal at reduced precision:
`mlp_wide metal bf16` and `f16`, arm A, seed 15 and time 5, and **all 5 log the lane-0 scalar
fallback**. Read as "5 timed" those cells would be false; the true count of tensorized pipelines
timed there is 0. Arm B of the same cells (37 → 21) carries no NOTEs.

Being timed is not being chosen. Across the whole matrix a `Tensorize` is crowned:

- on **cc**, in `mlp_wide` f32, both arms — `F_sketch[mma-cpu 16x0x16, mma-cpu 16x0x16, mma-cpu
  64x0x256]` at 33.27 ms wins arm B, and that arm ships (the cell measures 34.205 ms). This is the
  one shipping tensorized artifact on this machine.
- on **Metal**, exactly once in 99 cells: `mlp_small` f16 arm B crowns `F_sketch[mma-gpu 16x32x32
  ep]` at 1.6484 ms — and arm A wins the placement A/B at 0.9915 ms, so it does not ship.

So on Metal the gh-521 blocker is gone and the remaining gap is *ranking*, not reachability: the
tensorized candidates compile, run, and — with that one `mlp_small` f16 exception, which wins its
own arm but loses the placement A/B — lose. That is a different and more tractable problem than the
one the previous sweep recorded. Note the shape of the exception: the one Metal arm a `Tensorize`
wins is a *reduced-precision* one, which is the direction gh-ocannl-539 made expressible.

The blockers behind the still-failing candidates, by count (Metal, arm B, conv workloads):
12x `Autotune sketch: tensorized GPU matmul companion coverage (gh-521)` — the cross-nest race
analysis bails on the routine, so the 59 companion nests cannot be given aligned geometry. On `cc`
arm B the dominant blocker is instead 14x `Schedule.Stage: source w_fc2 is written in the routine`,
plus `Fuse_epilogue` guarded-write and `validate_parallel` rejections.

### Split reduction: 10 sites on lenet, all seeded candidates time, and it is decisive on arm A

Sites are capped at `autotune_split_reduce_max_sites=8` and the overflow is logged in the decline
census, so detected = 8 + evictions:

| workload | sites detected | evicted (logged) |
|---|---|---|
| lenet | **10** | `b_logits.grad red64 out10 cost640 swap1`, `cross_entropy red64 out1 cost64` |
| cifar_conv, cifar_stride | 9 | `cross_entropy red64 out1 cost64` |
| mlp_small | 10 | 2 |
| mlp_wide | 8 | 0 |
| gpt2_mini | **21** | 13 |

lenet's 10-at-cap-8-with-two-evictions is exactly gh-541's expected value. Seeded is not timed, so:
every split-reduce candidate seeded was timed, in every cell — **0 FAILED, 0 dedup'd** (9 candidates
per arm on Metal = 8 per-site plus the composite; 17 on `cc`, which tries two `num_blocks` per site).

The attribution below is the contract-compliant form: one process, one search, same code, only the
schedule differs — the best timed split-reduce candidate against the best timed candidate of any
other family in the same arm.

| workload/backend | arm A (default placements) | arm B (materialize-all — **ships**) |
|---|---|---|
| lenet / metal | 44.44 vs 249.04 ms — **−82.2%** | 33.63 vs 35.33 ms — **−4.8%** |
| cifar_stride / metal | 123.40 vs 239.26 ms — **−48.4%** | 116.61 vs 95.40 ms — +22.2% |
| cifar_conv / metal | 392.11 vs 1271.97 ms — **−69.2%** | 324.34 vs 276.49 ms — +17.3% |
| lenet / cc | 10.26 vs 19.14 ms — **−46.4%** | 9.41 vs 14.10 ms — **−33.3%** |
| cifar_stride / cc | 54.04 vs 152.65 ms — **−64.6%** | 52.97 vs 85.31 ms — **−37.9%** |
| cifar_conv / cc | 191.22 vs 788.90 ms — **−75.8%** | 187.13 vs 271.04 ms — **−31.0%** |

This replicates the gh-537 finding and sharpens it with gh-541's ranking in place: the split is
decisive in the default-placement arm everywhere, and in the arm that ships it wins on every `cc`
cell and on Metal only where the site is narrow (lenet's `bias_conv1.grad`, 6 output cells). On the
two Metal cifar cells the split is measured and *declined* — a preset wins — which is why those
cells crown `F_preset[bs=64 priv]` / `F_preset[bs=cfg priv cfg-thresh]` while lenet crowns
`F_split[bias_conv1.grad red64 out6 b32 swap3]` in both arms.

gh-541's per-site `num_blocks` ranking is visible on `cc`, where all three conv workloads crown the
8-site composite with *mixed* block counts (`b8` on the narrow sites, `b32` on `kernel_conv*.grad`) —
a schedule the pre-541 uniform ranking could not express.

**One expected value this leg cannot confirm.** gh-ocannl-538 expects ~−17.4% on lenet's shipping
artifact against pre-gh-537. That is a cumulative gh-537 + gh-541 figure measured on CUDA, and
confirming it here would need a paired same-session A/B against a binary with gh-541's ranking
reverted, which this leg did not build. The non-compliant observation, recorded as such: the gh-537
Metal leg reported 33.227 ms for this artifact at `ad46ab21` (post-537, pre-541) and it measures
33.668 ms here — i.e. consistent with gh-541 being neutral-to-slightly-negative on the Metal lenet
artifact, against a clearly positive CUDA composite. Two different sessions and two different
commits, so per contract item 3 that comparison is indicative only. If the number matters, the
paired A/B is the way to get it.

### Per-segment attribution, both placements

`BENCH_SEG_TIMES=1` with `BENCH_STEPS=1` in the same process, so each row's control step time is
measured alongside its segment times and the instrument's error is observed rather than assumed.
Shares are **within** their placement.

| workload | placement | top segment | ms | share | seg-time sum | control step | instrument error |
|---|---|---|---|---|---|---|---|
| lenet / metal | default | seg22 `bias_conv1.grad n65.grad` | 214.49 | 83.7% | 256.36 | 250.40 | +2.4% |
| lenet / metal | **materialized (ships)** | seg33 `kernel_conv1.grad bias_conv1.grad` + SGD | 21.89 | 48.6% | 45.08 | 36.80 | **+22.5%** |
| cifar_stride / metal | default | seg22 `bias_conv1.grad n65.grad` | 139.89 | 54.1% | 258.57 | 257.10 | +0.6% |
| cifar_stride / metal | **materialized** | seg33 `kernel_conv1.grad bias_conv1.grad` | 66.49 | 53.3% | 124.79 | 117.20 | +6.5% |
| cifar_conv / metal | default | seg22 `bias_conv1.grad n65.grad` | 960.15 | 73.4% | 1308.59 | 1318.80 | −0.8% |
| cifar_conv / metal | **materialized** | seg33 `kernel_conv1.grad bias_conv1.grad` | 180.80 | 54.1% | 333.95 | 325.50 | +2.6% |
| gpt2_mini / metal | default | seg111 `logits max_logits` | 32.55 | **8.9%** | 364.73 | 373.50 | −2.3% |

The instrument's regime dependence reproduces the gh-537 Metal measurement almost exactly: +2.4% on
lenet's default placement against **+22.5%** on its materialized one (that leg measured +2.4% and
+21.9%). The materialized arm has more and smaller segments, so per-routine dispatch overhead is a
larger fraction. Shares within a placement are sound; the sums above are *not* step times, and on
the arm that ships a lenet sum would overstate by more than a fifth.

`gpt2_mini` is the structural opposite of the conv workloads and worth stating plainly: 117 segments
on the default placement with the largest at **8.9%**, so there is no dominant segment to attack —
the step is spread across per-layer FFN projections (`n589 n617_gelu`, `n728 n756_gelu`, `n450
n478_gelu` at ~24 ms each) and the `logits max_logits` head. Its materialized-placement attribution
could not be measured; see the `tanh` shadowing bug below.

On `cc` every conv and gpt graph is a single segment (100%, instrument error ≤0.4%), so the
instrument yields no attribution there — unchanged from previous sweeps.

### Reduced precision: f16's cost is the loss-scaling gate, not f16

The four-variant requirement is what makes this readable, and the answer is unambiguous.
`mlp_wide`, default placement:

| variant | metal | cc |
|---|---|---|
| f32 | 5.700 ms | 219.148 ms |
| **f16-static** (fixed scale, no gate, no host read) | **5.262 ms** | 592.015 ms |
| f16 (dynamic gate, per-step host read) | 79.760 ms | 592.893 ms |
| f16-gated8 (fused on-device gate, host samples every 8) | 79.703 ms | 592.777 ms |
| f16-gated32 (…every 32) | 79.697 ms | 592.698 ms |

On Metal, f16 *compute* is **faster than f32** (5.262 vs 5.700 ms) and the entire 15x gap is the
dynamic loss-scaling gate. This is the discriminating experiment gh-ocannl-535 was waiting for, and
it settles the attribution: nothing about f16 arithmetic is slow here.

The surprise is the second row group. **The fused on-device gate gh-ocannl-250 shipped recovers none
of it on Metal**: `gated8` and `gated32` are 79.70 ms, indistinguishable from the per-step host-read
gate and 15x the static leg. Sampling 4x less often changes nothing, so the cost is not the host
read's *frequency* — it is structural, in the gate's own per-step work. On `mlp_small/metal` the
same legs read 0.876 (static) / 1.240 (dynamic) / 1.068 (gated8) / 1.246 (gated32) ms, where gating
does recover a little but not consistently in interval order. On `cc` the gate is free at both
intervals (592.0 / 592.9 / 592.8 / 592.7 ms), so this is Metal-specific.

Tuning largely recovers it — `mlp_wide metal tuned/f16` is 7.038 ms against 79.760 untuned — so the
gate's cost is schedule-dependent, which is consistent with it being structural work the scheduler
can hoist rather than an unavoidable synchronisation.

bf16 needs no gate and is a small genuine win on Metal (`mlp_wide` 5.220 vs 5.700 ms default, 3.414
vs 3.632 tuned). On `cc` it remains ~5.4x slower than f32 (1185.230 vs 219.148 ms), consistent with
the C backend having no native bf16 arithmetic.

### Trajectory drift, and what it says about the gh-523 constants

Max relative drift vs the pytorch/cpu reference over the parity window, across all variants of each
cell:

| workload | backend | bf16 | f16 |
|---|---|---|---|
| mlp_small | cc | 5.22e-04 … 7.48e-04 | 1.35e-04 … 1.55e-04 |
| mlp_small | metal | 5.22e-04 … 5.40e-04 | 1.34e-04 … 1.64e-04 |
| mlp_wide | cc | 2.60e-04 … 2.67e-04 | 2.03e-05 |
| mlp_wide | metal | 2.47e-04 … 2.67e-04 | 1.97e-05 … 2.05e-05 |
| **gpt2_mini** | **cc** | **9.57e-04 … 1.03e-03** | *(does not run)* |

The `gpt2_mini` bf16 leg — never previously run on a fixture — drifts **2.4x more than any mlp
cell**. Against `PARITY_TOL_PRECISION` bf16 = 4e-3 that leaves **3.9x headroom**, not the ~10x the
gh-476 sweep projected from mlp-only data when it recommended these constants. The recommendation
for gh-523 is therefore the opposite of "tighten further": 4e-3 is about right and bf16 should not be
reduced on mlp evidence alone. f16 keeps ~12x headroom against 2e-3 and has room, but the same
caution applies until a workload with an attention stack can be measured at f16 at all.

Metal and `cc` agree to 2-3 significant figures on every shared cell (e.g. mlp_wide bf16 2.47e-04
vs 2.60e-04, f16 1.97e-05 vs 2.03e-05), so no backend-specific numerical divergence shows up here.

### The gpt2_mini reduced-precision leg does not run

Cell 2 of gh-ocannl-538 asks for the four precision variants on `gpt2_mini`, whose leg had never run
on a fixture. It runs now, and 9 of its 12 OCANNL cells fail to compile, for two unrelated reasons.
Both are recorded rather than worked around.

1. **f16, both backends (6 cells).** `Utils.User_error("Constant -inf is too big for FP16 aka. half
   precision, risk of overflow; increase precision of tensor node max_vals")`. `max_vals` is the
   accumulator of the softmax max-reduction and its neutral element is −inf; the storage policy
   demotes it to f16 and the precision guard rejects the initializer. −inf *is* representable in
   FP16, so this is a policy question — either the guard should special-case the infinities, or
   reduction neutral elements should be excluded from demotion the way the loss head already is.
   As it stands f16 storage is unusable for any model with a softmax.
2. **bf16, Metal only (3 cells).** ~50 instances of `error: assigning to 'bfloat' from incompatible
   type 'float'`, from `sqrt` (9 sites, `Nn_blocks.layer_norm`) and `fmax` (4 sites, the softmax
   max). MSL has no `bfloat` overload for these builtins, so they return `float`, and unlike C it
   rejects the implicit narrowing back to a `bfloat` lvalue. `cc` bf16 runs the same graph fine and
   the Metal *mlp* bf16 leg passes — the same shape as the earlier Metal bf16 `Relu` rendering fix,
   different operators. **Fixed** (gh-ocannl-549): the bridge now covers the whole math-builtin
   family, not one operator at a time, and all three Metal cells run — default 328.7, materialized
   361.2, tuned 71.7 ms/step, losses agreeing across placements. Numbers from a later run than this
   sweep's, so they are not comparable with the tables above.

`gpt2_mini cc/bf16` does run, and is the source of the drift row above. It is also 6.9x slower than
f32 untuned (14421.900 vs 2086.770 ms) and 6.6x tuned (2258.420 vs 342.372) — the same missing
native bf16 arithmetic as `mlp_wide/cc`.

### A tensor node named `tanh` shadows the builtin `tanh()`

Found while measuring the materialized-placement attribution for `gpt2_mini`, which is why that row
is missing above. The per-segment hermetic compile emits

```
device float* __restrict tanh = (device float*)(__pools[__pool_slots[22]] + __pool_slots[23]);
...
tanh[((i1519) * 128 + i1520) * 1024 + i1521] = tanh(n325[...]);
```

and fails with `called object type 'device float *' is not a function or function pointer`. Node
debug names are used verbatim as identifiers, so any node whose label collides with a backend
builtin shadows it. Most generated names are safe because they carry an `nNNN_` prefix
(`n119_log`, `n234_sqrt_std_dev`); the exposed ones are block-supplied labels that land bare
(`tanh`, `gelu`, `layer_norm`). It surfaces only once the node becomes a named on-device buffer,
which is why the ordinary `gpt2_mini metal/materialized` benchmark cell passes while the
per-segment compile of the same graph does not.

### The cifar_conv metal/tuned skip is stale

`SKIP_CELLS` carried `("cifar_conv", "metal", "tuned", None)` for a post-tune re-init hang. Both
gh-537 legs suspected it was stale; this sweep ran the decisive check — the byte-for-byte command
`orchestrate` issues for that cell (`bench_conv.exe --ocannl_backend=metal` under `BENCH_TUNE=1`),
with the autotune cache redirected to a scratch dir so the sweep proper still searched from scratch.

It completed in ~4 minutes wall with no hang, `compile_s` 230.3 s against the 2069 s a standalone
search of the same cell took in the gh-476 sweep, and emitted its result line: p50 **277.915 ms**.
The full sweep then reproduced it independently at **277.928 ms** under `--no-skip-cells`, 0.005%
apart. The entry has been removed, and the Metal column gains a tuned `cifar_conv` cell it has never
had.

### Cross-framework caveats

No cell on this machine carries a correctness caveat: torch `mps` and tinygrad `METAL` both produce
valid GPU references, and the parity gate passed on all 99 cells. OCANNL remains far behind both on
this hardware — 43x behind torch/mps on `lenet` (33.668 vs 5.090 ms) after tuning, 129x on
`cifar_conv` — and ahead of torch CPU only on `mlp_small` and `lenet` (`cc/tuned`). The GPU gap is
the standing result; what changed this round is that the schedule search now has tensorized and
split-reduce candidates in its pool on Metal, and the split-reduce family is worth 46–82% on the
default-placement arm (48–82% on Metal alone).

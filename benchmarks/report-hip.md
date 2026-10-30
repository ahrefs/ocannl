# Benchmark results

> **2026-10-30 (gh-ocannl-1357): measured on the replaced raw-v1 files, not comparable byte-for-byte
> with later content-v1 reports.** Its `gpt2_mini` is minix's raw-v1 file (`10839bd1…`), and its `lenet` predates any recorded digest. The files on those boxes have since been replaced by the
> m4-max content-v1 bytes, which `fixtures/DIGESTS.txt` now records for rog-nv and minix.

platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 x86_64 | ocannl commit: e687da82 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager; reduced precisions get their own envelope: bf16 0.004, f16 0.002, f16-static 0.002, f16-gated16 0.002)
> The HIP/ROCm leg of the **gh-ocannl-538** re-measurement sweep, run from scratch on minix-pc:
> AMD Ryzen AI Max+ 395 (Strix Halo, Zen 5, 16C/32T) with the Radeon 8060S iGPU (gfx1151,
> RDNA3.5), ROCm 7.14, under WSL2. One commit for the whole leg — staging `e687da82`, which
> contains gh-ocannl-521, -527, -532, -533, -537, -539, -540 and -541 — and a **wiped
> `autotune_cache`** before the first tuned cell (contract item 7). torch 2.13.0+rocm7.1
> (HIP 7.1.52802), tinygrad 0.13.0. Every recorded cell PASSES the parity gate and every loss
> trajectory moved — no `FAIL`, no `loss stationary`, anywhere in this run.
>
> **Fixture provenance** (added retroactively, gh-ocannl-759): these numbers are on **minix's**
> fixture bytes. Two of the six workloads are pinned in `fixtures/DIGESTS.txt` under origin
> `minix` — `mlp_small` sha256 `2c48b1d7…c20ec0c1` and `gpt2_mini` sha256 `10839bd1…6269defb`,
> the two whose digests that box published. rog-nv holds **different bytes at identical sizes**
> for both (different venv, different numpy `Generator` stream), so absolute step times here are
> **not comparable** with `report-gh675-cuda.md`'s; the ratios each leg reports are, since a ratio
> is taken within one box on one fixture.
>
> **The other four are not yet pinned.** `cifar_conv`, `cifar_stride`, `lenet` and `mlp_wide` are
> recorded for `rog-nv` only, and nothing establishes that minix's copies of them are those bytes
> — given the two that *were* compared both differ, the likeliest reading is that they differ too.
> So the majority of the cells below still rest on bytes no artifact names. Closing that takes one
> command **on minix**: `python3 benchmarks/fixture_digest.py --record --origin minix`, which
> appends its four entries without regenerating anything and leaves rog-nv's alone. Until then,
> treat this report as internally consistent (one box, one sweep, one set of files) but externally
> unpinned for those four workloads.
>
> **Cell accounting**, so an absent row is never ambiguous between "failed" and "not attempted":
>
> | | full product | attempted | recorded | failed | not attempted |
> |---|---|---|---|---|---|
> | OCANNL f32 (6 workloads × 2 backends × 3 variants) | 36 | 36 | 35 | 1 | 0 |
> | OCANNL reduced precision (mlp × 4 precisions, gpt × 2, × 2 backends × 3 variants) | 60 | 50 | 43 | 7 | **10** |
> | pytorch + tinygrad (6 workloads × 2 devices × 2 frameworks) | 24 | 24 | 24 | 0 | 0 |
> | **total** | **120** | **110** | **102** | **8** | **10** |
>
> All 8 failures are `gpt2_mini`, for three distinct reasons, all diagnosed under Findings. The 10
> not-attempted cells are **every `cc/tuned` cell at a reduced precision**, dropped deliberately:
> `cc` has no native reduced-precision arithmetic and no reduced-precision mma tile, so those
> searches answer no question this leg asks, and they are the most expensive cells in the matrix
> (untuned `cc` runs at 1.82 s/step on `mlp_wide`/bf16 and 20.2 s/step on `gpt2_mini`/bf16, so a
> search that times dozens of candidates four times each is hours per cell). Their f32 siblings and
> their untuned reduced-precision siblings are all present, so the `cc` column is complete on both
> axes separately, just not on their product.
>
> Tuned cells use the two-pass protocol (pass 1 = the search, reported as `compile_s`; a fresh
> pass-2 process replays the cached winner for the step timings).
>
> Regenerate with, from `benchmarks/`:
>
> ```bash
> # cell 1: the f32 matrix, all workloads, all three frameworks
> taskset -c 0-15 .venv/bin/python orchestrate.py --tuned --materialized --gpu hip
> # cell 2: the reduced-precision legs — f16-static/f16-gatedN are bench_mlp env knobs that
> # orchestrate's --precision axis does not cover, so they are driven per cell:
> #   BENCH_PRECISION=bf16|f16 [BENCH_STATIC_SCALE=1 | BENCH_GATE_INTERVAL=16] BENCH_TUNE=0|1 …
> ```
>
> **All cells ran under `taskset -c 0-15`** (16 physical cores, SMT halves excluded): this machine
> hard-froze during an uncapped all-core `cc` autotune search in an earlier session, and the cap is
> the agreed mitigation. The `cc` column therefore reflects 16 threads, not the machine's 16C/32T.
>
> **Reading the numbers, per gh-ocannl-538's reporting contract.** Every segment share below names
> the placement it is a share of; the *shipping* arm is primary and the default pipeline is the
> tuning story. Per-segment sums are never presented as step times — the sum's error against an
> independently measured step is stated per regime, and it ranges from −5.6% to +22.4% on this
> machine. Tensor-core reachability is reported as seeded/timed counts per cell, never as a yes/no.
> No before/after claim in this report is made against a different commit or a different report;
> where a delta is quoted it is between two candidates timed **in the same process, in the same
> search**, and it is labelled as such.
>
> **Headline: this is the first report in which a rocWMMA candidate is seeded, timed and crowned.**
> `mlp_wide`/hip/bf16 crowns `F_sketch[mma-gpu 16x32x32 ep]`, −12.8% against the best scalar
> candidate the same search timed — and it is still 12% behind the f32 artifact, and carries a 10×
> drift increase. All three of those facts are load-bearing; see Findings.
>
> **`gpt2_mini` at a reduced precision barely runs** — 3 of the 10 attempted cells produce a result,
> all of them bf16 (`hip/materialized`, `cc/default`, `cc/materialized`). The rest of the bf16 leg
> fails HIPRTC compilation and the whole f16 leg fails a precision guard, both diagnosed under
> Findings. This is the first time that leg has been pointed at a fixture at all, so these are new
> findings rather than regressions.

## cifar_conv

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda(hip) | eager | f32 | 3.690 | 3.608 | 3.864 | 3.390 | 0.35 | PASS (1.0e-07) |
| tinygrad | HIP | jit | f32 | 6.744 | 6.526 | 7.041 | 6.226 | 1.81 | PASS (2.1e-07) |
| ocannl | hip | tuned | f32 | 63.992 | 63.404 | 64.778 | 64.150 | 173.50 | PASS (3.1e-07) |
| tinygrad | CPU | jit | f32 | 73.115 | 70.132 | 78.215 | 74.239 | 1.82 | PASS (3.1e-07) |
| ocannl | hip | materialized | f32 | 96.817 | 95.839 | 97.847 | 96.766 | 2.51 | PASS (3.1e-07) |
| ocannl | cc | tuned | f32 | 241.794 | 226.878 | 262.207 | 243.379 | 620.93 | PASS (2.8e-05) |
| pytorch | cpu | eager | f32 | 464.006 | 463.896 | 468.009 | 465.160 | 0.96 | REF |
| ocannl | hip | default | f32 | 756.249 | 753.133 | 760.117 | 756.236 | 2.41 | PASS (4.1e-07) |
| ocannl | cc | materialized | f32 | 1190.750 | 1185.210 | 1200.980 | 1192.000 | 3.06 | PASS (3.1e-07) |
| ocannl | cc | default | f32 | 1317.100 | 1311.060 | 1326.430 | 1319.200 | 2.33 | PASS (3.1e-07) |

## cifar_stride

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| tinygrad | HIP | jit | f32 | 1.863 | 1.819 | 1.984 | 1.605 | 1.81 | PASS (3.8e-05) |
| pytorch | cuda(hip) | eager | f32 | 2.692 | 2.606 | 2.870 | 2.197 | 0.29 | PASS (3.2e-05) |
| pytorch | cpu | eager | f32 | 4.045 | 3.544 | 5.310 | 4.017 | 1.09 | REF |
| ocannl | hip | tuned | f32 | 17.739 | 17.542 | 17.927 | 17.726 | 130.22 | PASS (3.8e-05) |
| ocannl | hip | materialized | f32 | 20.597 | 20.398 | 20.843 | 20.534 | 1.39 | PASS (3.8e-05) |
| tinygrad | CPU | jit | f32 | 38.418 | 37.031 | 41.239 | 38.647 | 1.68 | PASS (3.7e-05) |
| ocannl | cc | tuned | f32 | 62.574 | 58.707 | 67.935 | 71.551 | 347.25 | PASS (3.8e-05) |
| ocannl | hip | default | f32 | 63.041 | 62.504 | 64.732 | 63.702 | 1.37 | PASS (3.8e-05) |
| ocannl | cc | materialized | f32 | 316.270 | 314.366 | 321.159 | 316.006 | 2.57 | PASS (3.8e-05) |
| ocannl | cc | default | f32 | 349.897 | 347.193 | 356.040 | 348.900 | 2.20 | PASS (3.8e-05) |

## gpt2_mini

Rows are grouped by precision (f32 first), p50-ascending within each group.

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda(hip) | eager | f32 | 6.013 | 5.907 | 6.167 | 5.857 | 0.26 | PASS (6.7e-08) | 170,304 |
| tinygrad | HIP | jit | f32 | 6.909 | 6.725 | 7.227 | 6.381 | 0.78 | PASS (1.3e-07) | 148,220 |
| pytorch | cpu | eager | f32 | 25.143 | 23.007 | 26.529 | 35.488 | 0.03 | REF | 40,727 |
| ocannl | hip | default | f32 | 70.342 | 68.429 | 73.193 | 72.636 | 0.43 | PASS (2.0e-07) | 14,557 |
| ocannl | hip | materialized | f32 | 74.961 | 72.641 | 76.856 | 73.908 | 0.39 | PASS (8.7e-07) | 13,660 |
| tinygrad | CPU | jit | f32 | 100.527 | 98.249 | 105.190 | 100.821 | 0.99 | PASS (8.7e-07) | 10,186 |
| ocannl | cc | tuned | f32 | 526.213 | 504.756 | 566.489 | 517.726 | 507.33 | PASS (8.0e-07) | 1,946 |
| ocannl | cc | default | f32 | 2337.780 | 2328.770 | 2349.320 | 2328.660 | 1.91 | PASS (8.7e-07) | 438 |
| ocannl | cc | materialized | f32 | 2481.270 | 2469.310 | 2500.840 | 2485.050 | 5.97 | PASS (8.0e-07) | 413 |
| ocannl | hip | materialized | bf16 | 92.798 | 92.299 | 93.249 | 91.945 | 3.80 | PASS (4.5e-04) | 11,035 |
| ocannl | cc | default | bf16 | 20248.000 | 20220.600 | 20294.100 | 20278.900 | 2.20 | PASS (9.6e-04) | 51 |
| ocannl | cc | materialized | bf16 | 21397.100 | 21361.700 | 21437.800 | 21345.600 | 4.11 | PASS (1.0e-03) | 48 |

## lenet

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| tinygrad | HIP | jit | f32 | 0.844 | 0.813 | 0.896 | 0.613 | 1.67 | PASS (2.1e-07) |
| pytorch | cpu | eager | f32 | 1.469 | 1.284 | 1.852 | 1.535 | 0.56 | REF |
| pytorch | cuda(hip) | eager | f32 | 1.781 | 1.681 | 1.943 | 1.244 | 0.27 | PASS (2.1e-07) |
| ocannl | cc | tuned | f32 | 5.566 | 4.969 | 6.565 | 5.660 | 284.94 | PASS (2.1e-07) |
| ocannl | hip | tuned | f32 | 6.454 | 6.288 | 6.655 | 6.429 | 130.93 | PASS (2.1e-07) |
| ocannl | hip | materialized | f32 | 6.915 | 6.525 | 7.370 | 6.892 | 1.26 | PASS (3.1e-07) |
| ocannl | cc | materialized | f32 | 21.810 | 21.509 | 22.672 | 22.291 | 2.50 | PASS (3.1e-07) |
| tinygrad | CPU | jit | f32 | 24.440 | 23.953 | 25.321 | 24.687 | 1.60 | PASS (3.1e-07) |
| ocannl | cc | default | f32 | 32.646 | 31.952 | 33.836 | 32.839 | 2.11 | PASS (3.1e-07) |
| ocannl | hip | default | f32 | 80.991 | 80.574 | 83.171 | 81.024 | 1.26 | PASS (2.1e-07) |

## mlp_small

Rows are grouped by precision (f32 first), p50-ascending within each group.

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| ocannl | cc | tuned | f32 | 0.127 | 0.123 | 0.143 | 0.158 | 46.06 | PASS (2.8e-07) |
| ocannl | cc | materialized | f32 | 0.131 | 0.124 | 0.148 | 0.135 | 0.32 | PASS (2.8e-07) |
| ocannl | cc | default | f32 | 0.143 | 0.138 | 0.160 | 0.148 | 0.32 | PASS (2.8e-07) |
| ocannl | hip | materialized | f32 | 0.302 | 0.279 | 0.610 | 0.235 | 0.03 | PASS (3.8e-07) |
| ocannl | hip | default | f32 | 0.316 | 0.280 | 0.561 | 0.227 | 0.03 | PASS (3.0e-07) |
| ocannl | hip | tuned | f32 | 0.381 | 0.299 | 0.740 | 0.248 | 14.76 | PASS (3.0e-07) |
| pytorch | cpu | eager | f32 | 0.446 | 0.367 | 0.581 | 0.411 | 0.77 | REF |
| tinygrad | HIP | jit | f32 | 0.450 | 0.421 | 0.536 | 0.385 | 0.47 | PASS (2.0e-07) |
| pytorch | cuda(hip) | eager | f32 | 0.645 | 0.581 | 1.032 | 0.595 | 0.21 | PASS (1.7e-07) |
| tinygrad | CPU | jit | f32 | 0.782 | 0.724 | 1.002 | 0.841 | 0.43 | PASS (2.4e-07) |
| ocannl | hip | tuned | bf16 | 0.350 | 0.305 | 0.534 | 0.257 | 42.62 | PASS (5.9e-04) |
| ocannl | hip | materialized | bf16 | 0.372 | 0.309 | 0.589 | 0.270 | 0.30 | PASS (6.6e-04) |
| ocannl | hip | default | bf16 | 0.379 | 0.312 | 0.798 | 0.245 | 0.32 | PASS (5.6e-04) |
| ocannl | cc | materialized | bf16 | 4.363 | 4.303 | 4.537 | 4.398 | 0.36 | PASS (5.2e-04) |
| ocannl | cc | default | bf16 | 4.499 | 4.450 | 4.780 | 4.601 | 0.33 | PASS (5.4e-04) |
| ocannl | hip | tuned | f16 | 0.661 | 0.596 | 0.773 | 0.587 | 51.45 | PASS (1.5e-04) |
| ocannl | hip | default | f16 | 0.669 | 0.603 | 0.829 | 0.669 | 0.47 | PASS (1.5e-04) |
| ocannl | hip | materialized | f16 | 0.685 | 0.636 | 0.813 | 0.662 | 0.48 | PASS (1.4e-04) |
| ocannl | cc | materialized | f16 | 1.235 | 1.217 | 1.275 | 1.245 | 0.56 | PASS (1.5e-04) |
| ocannl | cc | default | f16 | 1.243 | 1.219 | 1.285 | 1.254 | 0.59 | PASS (1.4e-04) |
| ocannl | hip | materialized | f16-static | 0.335 | 0.307 | 0.517 | 0.198 | 0.28 | PASS (1.4e-04) |
| ocannl | hip | tuned | f16-static | 0.338 | 0.297 | 0.486 | 0.244 | 42.42 | PASS (1.7e-04) |
| ocannl | hip | default | f16-static | 0.381 | 0.314 | 0.774 | 0.266 | 0.29 | PASS (1.5e-04) |
| ocannl | cc | materialized | f16-static | 1.225 | 1.210 | 1.265 | 1.236 | 0.31 | PASS (1.5e-04) |
| ocannl | cc | default | f16-static | 1.232 | 1.217 | 1.276 | 1.239 | 0.32 | PASS (1.4e-04) |
| ocannl | hip | materialized | f16-gated16 | 0.424 | 0.369 | 1.330 | 0.351 | 0.33 | PASS (1.4e-04) |
| ocannl | hip | tuned | f16-gated16 | 0.442 | 0.385 | 0.591 | 0.345 | 52.44 | PASS (1.5e-04) |
| ocannl | hip | default | f16-gated16 | 0.456 | 0.387 | 1.088 | 0.384 | 0.34 | PASS (1.5e-04) |
| ocannl | cc | materialized | f16-gated16 | 1.233 | 1.218 | 1.288 | 1.247 | 0.43 | PASS (1.5e-04) |
| ocannl | cc | default | f16-gated16 | 1.235 | 1.217 | 1.284 | 1.259 | 0.44 | PASS (1.4e-04) |

## mlp_wide

Rows are grouped by precision (f32 first), p50-ascending within each group.

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| tinygrad | HIP | jit | f32 | 1.063 | 1.019 | 1.148 | 0.844 | 0.59 | PASS (2.1e-07) |
| pytorch | cuda(hip) | eager | f32 | 1.106 | 1.068 | 1.189 | 0.917 | 0.20 | PASS (2.1e-07) |
| ocannl | hip | tuned | f32 | 1.758 | 1.685 | 1.883 | 1.654 | 24.79 | PASS (3.1e-07) |
| ocannl | hip | materialized | f32 | 1.966 | 1.894 | 2.075 | 1.913 | 0.05 | PASS (4.2e-07) |
| ocannl | hip | default | f32 | 1.976 | 1.916 | 2.076 | 1.940 | 0.05 | PASS (4.2e-07) |
| ocannl | cc | tuned | f32 | 18.252 | 16.225 | 20.559 | 19.871 | 93.50 | PASS (5.2e-07) |
| tinygrad | CPU | jit | f32 | 37.101 | 34.582 | 41.440 | 37.280 | 0.65 | PASS (6.2e-07) |
| ocannl | cc | materialized | f32 | 217.627 | 213.095 | 228.134 | 220.402 | 0.51 | PASS (5.2e-07) |
| ocannl | cc | default | f32 | 218.825 | 214.206 | 231.185 | 222.417 | 0.49 | PASS (5.2e-07) |
| pytorch | cpu | eager | f32 | 240.005 | 239.900 | 243.986 | 240.621 | 0.61 | REF |
| ocannl | hip | tuned | bf16 | 1.967 | 1.888 | 2.138 | 1.926 | 75.69 | PASS (2.6e-03) |
| ocannl | hip | default | bf16 | 3.645 | 3.507 | 3.819 | 3.539 | 0.41 | PASS (2.7e-04) |
| ocannl | hip | materialized | bf16 | 3.673 | 3.549 | 3.829 | 3.583 | 0.39 | PASS (2.6e-04) |
| ocannl | cc | materialized | bf16 | 1806.930 | 1791.330 | 1828.110 | 1809.460 | 0.47 | PASS (2.6e-04) |
| ocannl | cc | default | bf16 | 1821.540 | 1807.300 | 1840.650 | 1822.640 | 0.47 | PASS (2.7e-04) |
| ocannl | hip | tuned | f16 | 2.367 | 2.261 | 2.578 | 2.284 | 78.00 | PASS (2.6e-05) |
| ocannl | hip | materialized | f16 | 11.743 | 11.348 | 12.102 | 11.783 | 0.57 | PASS (2.0e-05) |
| ocannl | hip | default | f16 | 11.784 | 11.334 | 12.146 | 11.701 | 0.56 | PASS (2.0e-05) |
| ocannl | cc | materialized | f16 | 838.999 | 831.477 | 851.371 | 843.665 | 0.85 | PASS (2.0e-05) |
| ocannl | cc | default | f16 | 840.551 | 831.508 | 854.037 | 842.776 | 0.83 | PASS (2.0e-05) |
| ocannl | hip | tuned | f16-static | 1.665 | 1.606 | 1.801 | 1.638 | 63.90 | PASS (2.0e-05) |
| ocannl | hip | default | f16-static | 2.010 | 1.938 | 2.135 | 2.049 | 0.35 | PASS (2.0e-05) |
| ocannl | hip | materialized | f16-static | 2.016 | 1.918 | 2.117 | 1.938 | 0.35 | PASS (2.0e-05) |
| ocannl | cc | materialized | f16-static | 841.169 | 833.419 | 854.243 | 847.065 | 0.47 | PASS (2.0e-05) |
| ocannl | cc | default | f16-static | 842.083 | 834.433 | 858.000 | 845.374 | 0.48 | PASS (2.0e-05) |
| ocannl | hip | tuned | f16-gated16 | 2.097 | 1.965 | 2.312 | 2.054 | 82.03 | PASS (1.7e-05) |
| ocannl | hip | materialized | f16-gated16 | 11.533 | 11.178 | 11.949 | 11.505 | 0.41 | PASS (2.1e-05) |
| ocannl | hip | default | f16-gated16 | 11.553 | 11.176 | 11.948 | 11.529 | 0.39 | PASS (2.0e-05) |
| ocannl | cc | default | f16-gated16 | 840.839 | 832.750 | 853.538 | 842.202 | 0.60 | PASS (2.0e-05) |
| ocannl | cc | materialized | f16-gated16 | 842.568 | 833.939 | 854.763 | 844.364 | 0.62 | PASS (2.1e-05) |

## Findings

### Tensor cores on HIP are reached, timed, and — once — crowned

The previous revision of this report recorded a structural zero: no tuned HIP cell could contain a
rocWMMA intrinsic, because f32 seeds nothing and `bench_mlp` refused `BENCH_TUNE` together with
`BENCH_PRECISION`. gh-ocannl-529 removed the runner guard and gh-ocannl-539 made *(scheduling
variant, precision)* a product, so the cell that was inexpressible is now measurable. Counts are per
search, summed over both placement arms, from `BENCH_TUNE_REPORT=1` against a fresh cache dir:

| workload | backend | precision | mma **seeded** | mma **timed** | best mma candidate | crowned? |
|---|---|---|---|---|---|---|
| mlp_small | hip | f32 | 0 | 0 | — | — |
| mlp_small | hip | bf16 | 29 | **17** | 0.4162 ms `F_sketch[mma-gpu 16x32x32 ep]` | no |
| mlp_small | hip | f16 | 29 | **17** | 0.4676 ms `F_sketch[mma-gpu 16x32x32 ep]` | no |
| mlp_wide | hip | f32 | 0 | 0 | — | — |
| mlp_wide | hip | bf16 | 37 | **21** | 1.7849 ms `F_sketch[mma-gpu 16x32x32 ep]` | **yes** |
| mlp_wide | hip | f16 | 37 | **21** | 10.7305 ms `F_sketch[mma-gpu 16x32x32 ep,mma-gpu 32x32x16 ep]` | no |
| mlp_small | cc | f32 | 54 | 24 | — | no |
| mlp_wide | cc | f32 | 112 | 21 | — | no |

**The 0 at f32 is the correct answer, not a failure.** RDNA3.5 WMMA has only f16×f16 and bf16×bf16
input shapes, so at uniform f32 nothing is proposed — `mma_candidates = 0`, no declines to explain.
That is a different state from the pre-gh-521 GPU picture, where candidates were seeded in bulk and
none was ever timed.

**17 timed on `mlp_small` at bf16** is exactly the value gh-ocannl-538 predicted. The seeded/timed
gap is fully accounted for and has a single cause: of the 29 seeds, 6 are whole-routine
(`W_sketch[mma-gpu …]`) and 6 are per-fission-segment (`F_sketch[mma-gpu …]`), and **all 12 decline
at `autotune_sketch_companion_coverage`** — "the cross-nest race analysis bails on this routine, so
the 99 companion nest(s) … cannot be given aligned geometry". The remaining 17 compile and time. So
on this backend the tensorization arc is no longer blocked by the three gh-521 families (`Stage`,
`Fuse_epilogue`, `validate_parallel`); what is left is one analysis that gives up on wide graphs.

**Contract item 4's second failure mode does not occur here.** `NOTE tensorized candidate emitted no
Tile_mma statement` appears **zero** times across all eight censuses: every candidate carrying an
`mma-*` label really rendered a `Tile_mma`. There are no scalar-fallback impostors in these counts.

**The payoff, measured in one process.** On `mlp_wide`/hip/bf16, arm B (materialize-all):

| candidate | ms |
|---|---|
| **`F_sketch[mma-gpu 16x32x32 ep]`** (crowned, and wins the arm A/B comparison) | **1.7849** |
| `F_sketch[mma-gpu 32x32x16 ep]` | 1.7868 |
| `F_sketch[mma-gpu 32x32x32]` | 1.8059 |
| best candidate carrying no `mma` label — `F_sketch[gpu 16x16x8/2x2]` | 2.0473 |

**−12.8% from tensorization**, same process, same search, against the best scalar candidate the same
search timed. This is the first quantified rocWMMA payoff in this suite.

It does not yet make bf16 worth using: the replayed `mlp_wide` hip artifact is 1.967 ms at bf16
against **1.758 ms at f32** (table above), so the format change costs more than the tensor cores
return. And it is not free numerically — see the drift section.

`cc` keeps reaching its CPU mma tiles at f32 (54 and 112 seeded, 24 and 21 timed) and crowning
non-mma candidates; unchanged from the previous revision.

### Where the step goes, in each placement

`BENCH_SEG_TIMES=1` on `bench_conv_diag` / `bench_gpt_diag`, min of 20 runs per segment, two
replicates per cell. **Shares are within a placement and are quoted with the placement named**
(contract item 1).

**The materialized placement** (`BENCH_MATERIALIZE=1`) — the arm the tuner crowns on `lenet`,
`cifar_conv` and `gpt2_mini`. (On `cifar_stride` the tuner now crowns the *default*-placement arm
instead; see the split-reduce section. And in all four cases the shipped artifact is this placement
with a schedule applied on top, so these are the costs the tuner is choosing between, not the
artifact's own segment times.)

| workload | top segment | ms | share |
|---|---|---|---|
| lenet | seg33 `kernel_conv1 kernel_conv1.grad bias_conv1 bias_conv1.grad` + SGD | 3.645 | **43.1%** of 8.464 |
| cifar_conv | seg31 `kernel_conv2.grad bias_conv2.grad n69_relu.grad …` | 58.689 | **61.7%** of 95.138 |
| cifar_stride | seg33 `kernel_conv1.grad bias_conv1.grad` | 7.278 | **34.0%** of 21.399 |
| gpt2_mini | seg38 (an FFN block, one of 130 segments) | 5.413 | 7.6% of 70.796 |

**The default pipeline** — the tuning story, not what users get:

| workload | top segment | ms | share |
|---|---|---|---|
| lenet | seg22 `bias_conv1.grad n65.grad` | 72.344 | **89.2%** of 81.137 |
| cifar_conv | seg22 `bias_conv1.grad n65.grad` | 668.651 | **93.5%** of 715.061 |
| cifar_stride | seg22 `bias_conv1.grad n65.grad` | 49.864 | **78.8%** of 63.245 |

The two placements disagree about which segment matters and by an order of magnitude about how
much, which is the whole point of contract item 1. In the default pipeline one fully serial
64×28×28×6 nest producing 6 output cells is ~90% of the step; in the arm that ships, the
kernel-gradient/SGD fusion is 34–62% and the bias-gradient nest is folded into it.

**Per-segment sums are not step times** (contract item 2). The sum's error against the
independently measured `hip` step p50 from the matrix above is regime-dependent, and on this machine
it is *not* small everywhere:

| workload | placement | Σ segment minima | measured step p50 | error |
|---|---|---|---|---|
| lenet | default | 81.137 ms | 80.991 ms | **+0.18%** |
| lenet | materialized | 8.464 ms | 6.915 ms | **+22.4%** |
| cifar_conv | default | 715.061 ms | 756.249 ms | −5.4% |
| cifar_conv | materialized | 95.138 ms | 96.817 ms | −1.7% |
| cifar_stride | default | 63.245 ms | 63.041 ms | +0.32% |
| cifar_stride | materialized | 21.399 ms | 20.597 ms | +3.9% |
| gpt2_mini | materialized | 70.796 ms | 74.961 ms | −5.6% |

`lenet`/materialized at **+22.4%** reproduces the Metal leg's +21.9% on the same regime almost
exactly — two independent backends now agree that the materialized regime is where this instrument
is least trustworthy in absolute terms, while the default regime is sound to a few tenths of a
percent. Replicate-to-replicate spread of the sums was 0.30% (lenet default), 0.30% (cifar_conv
default), 0.35% (cifar_stride default), 0.44% (cifar_conv materialized), 1.0% (lenet materialized)
and 2.9% (cifar_stride materialized) — smaller than the regime error in every case, so the error is
systematic, not noise.

`gpt2_mini`'s default placement could not be instrumented: compiling `cross_entropy_loss_fwd` as its
own hermetic routine requests 163,856 bytes of scratch per work-item and hits the gh-ocannl-533
pre-validator. The materialized placement instruments fine, and both ordinary benchmark cells run,
so this is a limitation of the per-segment instrument on this device, not of the workload.

### Split-reduce after gh-541: 10 sites on lenet, cap 8, two evictions — and the shipping arm can now flip

`BENCH_SR_SITES=1`, and the seeding census from `--ocannl_autotune_log=true`:

| workload | sites detected | seeded | evicted by `autotune_split_reduce_max_sites=8` |
|---|---|---|---|
| lenet | **10** | 8 | `b_logits.grad` (cost 640), `cross_entropy` (cost 64) |
| cifar_conv | 9 | 8 | `cross_entropy` (cost 64) |
| cifar_stride | 9 | 8 | `cross_entropy` (cost 64) |

**Exactly the predicted 10 / cap 8 / two evictions on `lenet`.** The ranking is gh-541's estimated
segment cost, and it evicts the two *cheapest* sites — the previous `sr_red / sr_out` ranking sent
every weight gradient to ratio 0 and excluded them instead. Both weight gradients now seed on
`lenet` (`kernel_conv2.grad` cost 15,360,000 via 12 swaps; `kernel_conv1.grad` cost 7,526,400 via 9
swaps), which is precisely what the CUDA leg asked for and could not test.

Per-workload attribution, each block one search in one process (so no cross-binary comparison is
involved). "control" is the untuned-default in-process control the search itself times:

| lenet | arm A (default placements) | arm B (materialize-all) |
|---|---|---|
| untuned-default control | 80.757 ms | 6.4440 ms |
| best `F_preset` | 80.458 ms | 6.4874 ms |
| best single split site | 10.164 ms `bias_conv1.grad swap3` | **6.2326 ms `n105`** |
| 8-site composite | **9.6501 ms** | 7.4205 ms |
| arm best | 9.6501 ms | 6.2326 ms |
| winner | | **arm B at 6.2326 ms (−3.3% vs its control)** |

| cifar_conv | arm A | arm B |
|---|---|---|
| untuned-default control | 757.111 ms | 95.747 ms |
| best `F_preset` | 752.403 ms | 92.870 ms |
| arm best | 95.857 ms `bias_conv1.grad out32 swap3` | **63.536 ms `bias_conv2.grad out64 swap3`** |
| winner | | **arm B at 63.536 ms (−33.6% vs its control)** |

| cifar_stride | arm A | arm B |
|---|---|---|
| untuned-default control | 62.667 ms | 20.755 ms |
| best `F_preset` | 61.654 ms | 20.542 ms |
| arm best | **17.427 ms `bias_conv1.grad out32 swap3`** | 20.344 ms |
| winner | **arm A at 17.427 ms** | |

Three things worth recording:

1. **On `cifar_stride` the split-reduce candidate is now good enough to flip which arm ships.** Arm
   A — the virtual/default-placement graph, previously hopeless at 62.7 ms — wins at 17.427 ms
   against arm B's 20.344. The replayed artifact is 17.739 ms against 20.597 for the materialized
   cell, so the tuner's choice survives replay. The CUDA leg reported `cifar_stride` neutral; on
   this backend it is the biggest conv win in the sweep.
2. **`cifar_conv` is not neutral here either**: −31.6% against the best preset in the arm that
   ships, on `bias_conv2.grad` (64 cells) rather than `bias_conv1.grad`. The CUDA leg found the
   32-cell `bias_conv1.grad` split *actively worse* in arm B and the tuner declining it; the same
   candidate is timed here at 92.507 ms and also loses — to a different split site, not to a preset.
3. **`lenet`'s shipping-arm margin is small on this backend**: −3.9% against the best preset,
   −3.3% against the untuned control, and the site that wins is `n105` (the classifier head gh-484
   already reached), not `bias_conv1.grad`. The 8-site composite is *worse* than the single best
   site in arm B (7.4205 vs 6.2326) — splitting everything is not free.

**No cross-commit A/B was run on this leg**, so gh-ocannl-538's "~−17.4% on lenet's shipping
artifact against pre-#537" is neither confirmed nor refuted here. What is quoted above is the
in-search attribution: crowned split candidate against the best non-split candidate the same
search timed in the same process. Since before gh-537/gh-541 none of these sites was detected at
all, the best non-split candidate is what the older tree would have crowned — but that is an
argument, not a measurement, and it is labelled as one.

### Mixed precision: the loss-scaling gate's cost is a scheduling problem, not a host-sync problem

The four variants gh-ocannl-538 asks for, on `mlp_wide`/hip (ms/step, p50):

| variant | f32 | bf16 | f16 (per-step host gate) | f16-static (no gate) | f16-gated16 (fused, sampled) |
|---|---|---|---|---|---|
| default | 1.976 | 3.645 | **11.784** | 2.010 | **11.553** |
| materialized | 1.966 | 3.673 | 11.743 | 2.016 | 11.533 |
| tuned | 1.758 | 1.967 | 2.367 | **1.665** | 2.097 |

and the same on `cc`, as the control (no `tuned` row at a reduced precision: those 10 cells were
deliberately not attempted — see the cell accounting in the header):

| variant | f32 | bf16 | f16 | f16-static | f16-gated16 |
|---|---|---|---|---|---|
| default | 218.825 | 1821.540 | 840.551 | 842.083 | 840.839 |
| materialized | 217.627 | 1806.930 | 838.999 | 841.169 | 842.568 |
| tuned | 18.252 | — | — | — | — |

**Without the static leg this would have been unreadable, which is why the contract asks for it.**
At the default placement on HIP the gate costs **9.77 ms** — 5.9× the entire f32 step — and the
fused on-device gate at interval 16 does not help at all (11.553 vs 11.784). On `cc` the same gate
costs **nothing measurable** (840.6 vs 842.1, inside the spread). A per-step host sync cannot behave
like that.

The autotune log says what it actually is. At f16 the crowned candidate in both arms is a
split-reduce composite whose first-named site is `grad_checksum` — the gate's whole-gradient
inf/nan scan — and the isolated candidates price it:

```
F_split[grad_checksum red1024 out1 b512]:  3.6221 ms
F_split[grad_checksum red1024 out1 b128]:  3.6941 ms
F_split[grad_checksum red1024 out1 b32]:   4.1146 ms
F_split[n40 …, grad_checksum red1024 out1 b512, n18_cast.grad …]:  2.0877 ms   <-- crowned
```

So `grad_checksum` lowers to one serial reduction over the whole gradient set, the default GPU
schedule leaves it unparallelized, and split-reduce is what removes it. `cc` never sees the problem
because the pool parallelizes the same loop. **This is the answer to gh-ocannl-535's open question:
f16's overhead is not the dynamic gate's host round-trip; it is one reduction that only the tuner
currently fixes.** Two consequences: the fused on-device gate (gh-ocannl-250) buys nothing until
that reduction is scheduled, and the default GPU pipeline should be seeding a parallel reduction
for it without needing a search.

**And with the gate scheduled, reduced precision beats f32 for the first time in this suite**:
`mlp_wide` hip tuned f16-static at **1.665 ms** against f32 tuned 1.758 ms (−5.3%) and f32 default
1.976 ms (−15.7%). It is one cell, and it is the cell with no loss scaling at all — every gated f16
cell and every bf16 cell is still slower than its f32 counterpart, and on `mlp_small` f32 still wins
outright (best f32 0.302 ms vs best reduced 0.335 ms). gh-535's headline is therefore no longer
universally true, but the qualification is large.

On `cc` the previous picture is unchanged and worse than before: bf16 is **8.3×** f32 on `mlp_wide`
(1821.5 vs 218.8) and f16 **3.8×** (840.6) — the C backend has no native reduced-precision
arithmetic, and gh-ocannl-540's cast-twin fix does not change that.

### Reduced-precision drift, for gh-ocannl-523's constants

Max relative loss deviation vs the pytorch/cpu reference over the parity window, worst variant per
cell:

| workload | backend | bf16 | f16 | f16-static | f16-gated16 |
|---|---|---|---|---|---|
| mlp_small | cc | 5.40e-04 | 1.54e-04 | 1.54e-04 | 1.54e-04 |
| mlp_small | hip | 6.56e-04 | 1.53e-04 | 1.73e-04 | 1.53e-04 |
| mlp_wide | cc | 2.67e-04 | 2.03e-05 | 2.03e-05 | 2.09e-05 |
| mlp_wide | hip | 2.67e-04 (untuned) / **2.64e-03 (tuned)** | 2.57e-05 | 2.03e-05 | 2.14e-05 |
| gpt2_mini | cc | 1.03e-03 | — | — | — |
| gpt2_mini | hip | 4.54e-04 | — | — | — |

Two readings.

**Where no tensor core is involved, hip and cc agree to three significant figures** — `mlp_wide`
bf16 2.67e-04 on both, f16 2.0e-05 on both — which is the same cross-backend agreement macOS and
CUDA reported, and the signal that no backend is quietly computing something else. The gate variants
do not move drift, as expected: static, dynamic and fused-sampled scaling all reach the same losses.

**Where a tensor core *is* involved, drift jumps 10×.** `mlp_wide`/hip/**tuned**/bf16 — the one cell
in this sweep whose search crowns a rocWMMA candidate — drifts **2.64e-03** against 2.60–2.67e-04
for every other bf16 cell of the same workload on either backend. It passes the 4e-3 bf16 gate with
only 1.5× headroom. Tuning by itself is not what does this, and the same workload supplies the
control: on `mlp_wide`/hip, tuning moves drift from 4.16e-07 to 3.09e-07 at f32, from 2.02e-05 to
2.57e-05 at f16, and from 2.02e-05 to 2.02e-05 at f16-static — none of those searches crowns an mma
candidate. Only the one that does moves an order of magnitude. **This is a direct argument against
the tightening the previous
revision of this report proposed** (bf16 → 1e-3): that constant would now reject a correct
tensorized schedule. The f16 constant still has room (worst 1.73e-04 against 2e-3, 11.5×), but the
bf16 one should stay at 4e-3 until a tensorized bf16 cell has been characterised on more than one
backend.

### `gpt2_mini` at reduced precision: three distinct blockers, found on its first fixture run

gh-ocannl-538 notes this leg had never run on a fixture. It now has, and 3 of 10 cells survive —
`hip/materialized/bf16`, `cc/default/bf16`, `cc/materialized/bf16`, all of which pass parity. The
other seven fail for two distinct reasons:

- **bf16 fails HIPRTC compilation.** `cross_entropy_loss_fwd__seg.hip:362:68: error: use of
  overloaded operator '+' is ambiguous (with operand types '__hip_bfloat16' and 'float')`, and the
  same for `'/'` with `('float', '__hip_bfloat16')` — 12 errors, all mixed `__hip_bfloat16`/`float`
  binary expressions that the HIP headers give no unambiguous overload for. It is
  **placement-dependent**: `hip/materialized/bf16` compiles and runs cleanly (92.798 ms, 11,035
  tok/s, drift 4.5e-04), while `hip/default/bf16` and `hip/tuned/bf16` both die, because only the
  fissioned graphs emit the mixed expression. **Fixed** (gh-ocannl-549) — but not where this
  predicted: neither `convert_precision` nor the binop emission introduces the float. The op
  table's own `expf`/`logf`/`sqrtf` arms *return* float at bf16, and only inlining moves that
  result from an assignment (which `__hip_bfloat16` accepts) into a binary operand (which is
  ambiguous). The fix is bf16 arms for the float-returning unops, mirroring the `ToPowOf` arm
  already there. `gpt2_mini hip/default/bf16` now runs at 86.3 ms/step.
- **f16 trips `Low_level.check_constant` on both backends, before any codegen — on two constants
  that are not the same problem.** The guard is `Ops.exceeds_fp16_cutoff`, a *configurable magnitude
  cutoff* (`check_half_prec_constants_cutoff`, default 16384.0 = 2^14) and deliberately well below
  f16's 65504 max finite, so it is a headroom guard against overflow during arithmetic rather than a
  representability check. The two rejections it produces here need separate fixes:
  - `Constant -1000000000. is too big for FP16 … tensor node where` — the causal mask's `-1e9`.
    **A genuine overflow**: it is 4 orders of magnitude past f16's largest finite value, so the
    guard is doing exactly its job. The model has no way to say "this sentinel is the storage
    format's own min", so either the mask constant becomes precision-aware or `where` joins the
    layer norms and the CE head in the f32 pin list.
  - `Constant -inf is too big for FP16 … tensor node max_vals` — softmax's max-subtraction
    sentinel. **Not an overflow.** `-inf` is exactly representable in IEEE binary16, and the code
    generator already emits it deliberately (`c_syntax.ml` renders it as `(-INFINITY)`, and the HIP
    backend routes it through `__float2half`). `abs (-inf) >= cutoff` is trivially true for any
    finite cutoff, so the guard rejects an infinity it has no reason to reject. **The cutoff needs
    an infinity exemption**; pinning `max_vals` to f32 would work around a guard defect rather than
    fix a numerical problem.

**On this backend `gpt2_mini` has no tuned cell at any precision** — at f32 the search dies on the
gh-533 scratch pre-validator (below), at bf16 on the HIPRTC error above, at f16 on the constant
guard. This is HIP-specific: `cc/tuned` at f32 completes normally (526.213 ms, search 507.33 s), and
`cc` at a reduced precision was not tuned by choice (see the note on skipped cells above).

### gh-ocannl-533: the scratch abort is now a clean diagnosis, but it is still fatal

`gpt2_mini` hip/tuned f32 exits 2 with

```
HIP: kernel cross_entropy_loss_fwd needs 163856 bytes of private (scratch) memory per work-item,
above the 104832 bytes this device can back at full occupancy (2048 work-items x 20 CUs against a
4294967296-byte scratch allocation). Launching it would abort the queue rather than fail cleanly
(gh-ocannl-533)
```

That is a real improvement over the previous revision — the HSA queue is no longer aborted and the
device is not left disturbed — but the outcome for the sweep is identical: the cell produces no
result, because the pre-validator raises out of `Autotune.tune` instead of recording a decline and
moving to the next candidate. An unsatisfiable-scratch candidate is exactly the kind of thing the
decline census exists for. Same escape path breaks `bench_gpt_diag`'s default-placement segment
timing.

### gh-ocannl-532 is confirmed fixed, in the logs of every GPU search

Every HIP search in this sweep opens with

```
baseline: NOT DISPATCHED, binds no hardware dimension on hip -- the whole routine would run in one
work-item (gh-ocannl-532) (digest …)
```

and no cell in the matrix needed a watchdog. `mlp_wide`, `cifar_conv` and `cifar_stride` hip/tuned —
the three cells the previous revision could not produce — all complete: search costs 24.79 s,
173.50 s and 130.22 s respectively, with the display intact throughout. `SKIP_CELLS` now contains no
HIP entry.

### Cross-framework

**The torch GPU column is valid this session.** All six `pytorch/cuda(hip)` cells pass the parity
gate (1.0e-07 to 3.2e-05) with moving losses. The non-deterministic reduction garbage that made
five of six cells unusable during the gh-476 sweep did not reproduce anywhere in ~4.5 hours of
continuous load here. That is the third independent observation supporting the machine-state explanation and against
the retracted PyTorch-defect claim. **This leg therefore has two valid GPU references, not one** —
an update to gh-538's contract item 6, which assumed tinygrad/HIP would be the only one.

Best OCANNL `hip` cell against the two GPU references:

| workload | ocannl/hip best | tinygrad/HIP | pytorch/cuda(hip) | gap vs tinygrad |
|---|---|---|---|---|
| mlp_small | 0.302 (materialized) | 0.450 | 0.645 | **0.67× — ahead** |
| mlp_wide | 1.758 (tuned) | 1.063 | 1.106 | 1.65× |
| lenet | 6.454 (tuned) | 0.844 | 1.781 | 7.6× |
| cifar_stride | 17.739 (tuned) | 1.863 | 2.692 | 9.5× |
| cifar_conv | 63.992 (tuned) | 6.744 | 3.690 | 9.5× |
| gpt2_mini | 70.342 (default) | 6.909 | 6.013 | 10.2× |

The conv and GPT gaps are still close to an order of magnitude, and the shipping-arm attribution
above says where the remaining time is: the fused kernel-gradient/SGD segment, which no split-reduce
candidate currently touches. (These are within-run ratios; per contract item 3 they are not compared
against the ratios in the previous revision of this file, which was recorded at a different commit.)

**The torch CPU column inverted again, in the opposite direction.** Same machine, same
`taskset -c 0-15`, same 16 OpenMP threads on MKL 2024.2:

| workload | this run | previous revision |
|---|---|---|
| mlp_small | 0.446 ms | 112.003 ms |
| mlp_wide | 240.005 ms | 6.338 ms |
| lenet | 1.469 ms | 339.967 ms |
| cifar_conv | 464.006 ms | 17.034 ms |

Every cell moved by one to two orders of magnitude, and the specific pathology gh-538's contract
item 6 names — "`mlp_small` 18× slower than the much larger `mlp_wide`" — **does not reproduce**;
the inversion now runs the other way. The conclusion is unchanged and if anything stronger: **torch
CPU on this machine is not a usable performance baseline in either direction**, and any
OCANNL-vs-torch CPU ratio taken from either revision is an artifact of the reference. It remains
sound as the *parity* oracle — it is the reference every PASS above is measured against.
tinygrad/CPU is the sound CPU comparison here; against it, OCANNL `cc` tuned wins `mlp_small`
(0.127 vs 0.782), `lenet` (5.566 vs 24.440) and `mlp_wide` (18.252 vs 37.101), and loses
`cifar_conv` (241.794 vs 73.115) and `gpt2_mini` (526.213 vs 100.527).

### Still open

- **The tensor-core measurement this cluster exists for is now made on HIP, and it is thin.** One
  crowned cell, −12.8% in-search against the best scalar candidate, still 12% behind the f32
  artifact. The next question is not "can it be reached" but **why the companion-coverage analysis
  bails**: it declines 12 of 29 mma seeds on `mlp_small` and 16 of 37 on `mlp_wide`, including every
  whole-routine seed, and it is the single remaining blocker family on this backend.
- **The bf16 parity constant must not be tightened to 1e-3** until the 2.64e-03 tensorized-bf16
  drift is understood and reproduced on a second backend.
- **`gpt2_mini` has no tuned `hip` cell at any precision, and only one working reduced-precision
  `hip` cell** — three separate blockers (the gh-533 scratch validator at f32, the HIPRTC overload
  ambiguity at bf16, the constant cutoff at f16), none of them shared with the mlp workloads and
  none of them affecting `cc`, whose f32 tuned cell completes normally.
- **The f16 constant cutoff should exempt infinities.** `-inf` is exactly representable in binary16
  and the code generator emits it on purpose; rejecting it is a guard defect independent of the
  genuine `-1e9` mask overflow beside it.
- The gh-533 pre-validator should decline the candidate rather than raise out of the search.
- The `grad_checksum` reduction should be parallel in the default GPU pipeline, without a search.
- `cifar_conv`'s `cc/tuned` search is now the most expensive cell in the suite at 620.93 s (against
  241.794 ms/step); `gpt2_mini`'s is 507.33 s. Both are dominated by gh-541's larger site pool.
- `SKIP_CELLS` audit: the only remaining entry is `cifar_conv metal/tuned`, untestable here (no
  Metal hardware). No HIP entry remains.

### Reproducing

All from `benchmarks/`, with `LD_LIBRARY_PATH=/opt/rocm/lib/rocm_sysdeps/lib` and the wheel's
`libhsa-runtime64.so` replaced by `/opt/rocm/lib/libhsa-runtime64.so.1.21.0` (see the README's WSL
notes).

```bash
# the mma seeded/timed census, per (workload, backend, precision)
rm -rf /tmp/probe
BENCH_FIXTURE=fixtures/mlp_wide.safetensors BENCH_TUNE=1 BENCH_TUNE_REPORT=1 BENCH_PRECISION=bf16 \
  taskset -c 0-15 ../_build/default/benchmarks/runners/ocannl/bench_mlp.exe --ocannl_backend=hip \
  --ocannl_autotune_cache_dir=/tmp/probe --ocannl_autotune_log=true 2>&1 | grep -E 'tune arm:|mma-gpu'

# which split-reduce sites, through which interchange, and what the cap evicts
BENCH_FIXTURE=fixtures/lenet.safetensors BENCH_SR_SITES=1 \
  taskset -c 0-15 ../_build/default/benchmarks/runners/ocannl/bench_conv_diag.exe --ocannl_backend=hip

# the per-site attribution, one process per workload
rm -rf /tmp/probe
BENCH_FIXTURE=fixtures/lenet.safetensors BENCH_TUNE=1 \
  taskset -c 0-15 ../_build/default/benchmarks/runners/ocannl/bench_conv.exe --ocannl_backend=hip \
  --ocannl_autotune_cache_dir=/tmp/probe --ocannl_autotune_log=true 2>&1 \
  | grep -E 'F_split\[|F_preset\[|control|arm |winner'

# per-segment attribution; prefix BENCH_MATERIALIZE=1 for the arm that ships
BENCH_FIXTURE=fixtures/lenet.safetensors BENCH_SEG_TIMES=1 \
  taskset -c 0-15 ../_build/default/benchmarks/runners/ocannl/bench_conv_diag.exe --ocannl_backend=hip

# the gate-cost legs (f16 only, mutually exclusive)
BENCH_FIXTURE=fixtures/mlp_wide.safetensors BENCH_PRECISION=f16 BENCH_STATIC_SCALE=1  … bench_mlp.exe
BENCH_FIXTURE=fixtures/mlp_wide.safetensors BENCH_PRECISION=f16 BENCH_GATE_INTERVAL=16 … bench_mlp.exe
```

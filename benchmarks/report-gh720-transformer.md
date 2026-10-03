# gh-ocannl-720: v1.1 transformer benchmark legs

The four transformer legs below share one measurement campaign. The suite driver's two
HIP `report.md` outputs are embedded verbatim under [HIP tables](#hip-generated-reports),
including fixture digests, origin warnings, regime labels, envelope verdicts and divergence
notes. No cell table is transcribed or repeated in the analysis. Failed parity timings remain
visible in the generated evidence but are diagnostic only; they support no speedup comparison.
The [cc/Metal section](#cc-and-metal) is reserved for the exclusive mac-studio measurement.

## HIP analysis by leg

### Leg 1: exact and approximate regimes

Every f32 cell passes. Approximate increases OCANNL HIP p50 on all eight endpoints with
the default schedule, despite reducing several activation footprints. Base inference is
20.024 ms approximate versus 17.695 ms exact; torch HIP eager is 7.409 versus 7.312 ms.
This reports the regime column from staging#661 without promising a profile speedup.
These are whole-profile comparisons rather than isolated rewrite ablations; gfx1102 supplies
no CUDA TF32 evidence for gh-ocannl-719. Exact and approximate cells use different math
settings on both frameworks, as stated in each generated workload section.

### Leg 2: sequence scaling

The `gpt2_mini` / `_s512` / `_s1024` rows and their training counterparts hold tokens per
step at 1024 (seq 128/512/1024, batch 8/2/1). OCANNL exact f32 inference falls from
57,870 to 40,365 tokens/s across the endpoints; approximate falls from 51,138 to 17,326.
Exact training falls from 23,853 to 13,448 tokens/s; approximate from 21,116 to 9,469.
Shrinking the leading batch axis also changes scheduling, so this does not isolate sequence
alone. Seq1024 is the bounded v1.1 endpoint, reusing the recorded fixture rather than the
originally proposed seq2048. The attention and fused-backward gates are profiled together.

### Leg 4: batch scaling

At seq128, batch-1 exact f32 inference takes 2.962 ms versus 17.695 ms at batch8;
torch HIP eager takes 2.558 ms at batch1. This comparison includes framework submission
cost and does not establish a kernel-only efficiency advantage; graph capture was left at
its configured default, without an observed capture-success receipt.

OCANNL's own exact f32 training throughput **drops from 23,853 tokens/s at batch8 to
7,800 at batch256: 0.327x throughput, or 3.06x slower per token**. At batch256 torch HIP
eager reaches 50,543 tokens/s (6.48x OCANNL); OCANNL is only 1.48x the torch CPU reference's
5,280 tokens/s. CPU thread counts were not pinned equal, so the latter is an observed
framework ratio, not a controlled CPU fairness comparison. The batch256 allocation seam
reports 6,624.5 MiB (6.469 GiB), equivalent to **81.0%** of the device's 7.984 GiB
VRAM capacity; this requested-byte allocator counter is not a measured device-wide VRAM
occupancy. Approximate reports 5,084.5 MiB, but throughput drops further to 7,440 tokens/s.
The slowdown and its cause are tracked in [gh-ocannl-1183](https://github.com/ahrefs/ocannl/issues/1183).

Batch256 reuses one data batch while batch8 cycles four; the trajectories therefore use
different data. Small fixture files store weights and inputs rather than the much larger
activations. Discrete VRAM makes memory pressure relevant, but this sweep does not prove
whether memory pressure or schedule scaling caused the slowdown.

### Leg 7: reduced precision

All bf16 cells pass. Exact f16 exceeds the 0.002 envelope at seq1024 inference
(max relative difference 0.0044960326636339955) and training (0.004093540358374387), and
base training is non-finite at step 0. Approximate f16 passes its 0.01 envelope on every
endpoint, **but base training is beyond the exact f16 envelope** (0.0050481126702888205).
The driver prints `PASS (5.0e-03, beyond exact envelope)` for that row. This control belongs
with the exact failures in [gh-ocannl-1182](https://github.com/ahrefs/ocannl/issues/1182);
it does not establish the cause of the failures. The complete exact-f16 base-training
parity trajectory (`null` encodes a non-finite float; later losses are finite) is:

```json
[null, 7.10226059, 7.12176037, 7.08063602, 7.09992838, 7.10012341]
```

Batch256 passes every format/regime: exact bf16 uses 3,320.9 MiB and takes 4504.490 ms;
exact f16 uses 3,382.0 MiB and takes 3879.630 ms, versus f32 6,624.5 MiB / 4200.920 ms.
Reduced storage helps memory here without a uniform throughput improvement. These are
OCANNL formats against an f32 torch oracle, not reduced-precision torch performance arms;
untuned storage changes do not demonstrate tensorized schedules.

## Measurement protocol and provenance

HIP was measured 2026-10-03 on `tuf-amd-linux`, gfx1102, **discrete** VRAM
(8,573,157,376 bytes), under exclusive request `wave1003-720-tuf-amd-linux-2`.
Measured source: `a4a425454cb9bf12c4c24b2c0d5bb6183b1c64ff`, before the PR rebase.
Linux 7.0.0-31-generic, x86_64, glibc 2.43; torch `2.13.0+rocm7.1`.
All fixtures are unchanged copies of the mac-studio bench venv's recorded m4-max stream,
matching `m4-max,tuf` content-v1 entries in [DIGESTS.txt](fixtures/DIGESTS.txt).
They were never regenerated on TUF. Missing minix/rog-nv entries remain explicit below.

One full-protocol sweep, not repeated independent trials: OCANNL default/untuned
(`searched=false`), torch eager, no torch.compile or tinygrad, no dominant-kernel diagnostics.
No ambient `OCANNL_*` variables. Graph capture remained configured `gpu_graph_capture=true`,
without a capture-success receipt. Synced p10/p50/p90 and queued mean answer different timing
questions; compilation is separate. Peak counters cover timed steps. The allocator seam and
torch HIP allocation counter are distinct counters; rank within a counter and treat
cross-counter comparisons as approximate. Torch CPU rows are not OCANNL cc results.

The complete sweep yielded 80 rows: 69 PASS, 8 REF, 2 FAIL, 1 DIVERGED, no runner timeouts
or failures. Exit 1 is from parity. All f32, bf16, approximate and batch256 rows pass their
own regime gates; the one approximate row outside its exact envelope is identified above.

Fixture protocols (P/W/T = parity/warmup/timed; T runs once synced and once queued):
base inference 8/5/20, batch-1 inference 4/5/20, long-context inference 4/3/10, all training
6/3/10. Training uses plain SGD lr 0.01. These counts are the fixture metadata, not overrides.

Build/check/scans passed at `/home/lukstafi/.ocannl-test-runs/20261003T170038Z-2942453`;
correctness preparation was at `/tmp/wave1003/720/tuf-correctness`. The exclusive timing
window used these two invocations, with a 6900-second outer cap and separate result directories:

```sh
BENCH_VENV_PY=$HOME/.venvs/ocannl-bench/bin/python BENCH_DOMINANT_KERNEL=0 \
BENCH_RESULTS_DIR=/tmp/wave1003/720/tuf-measurement/main \
$HOME/.venvs/ocannl-bench/bin/python benchmarks/orchestrate.py --skip-build \
  --gpu hip --only ocannl pytorch --profile exact approximate --precision bf16 f16 \
  --workloads gpt2_mini gpt2_mini_s512 gpt2_mini_s1024 gpt2_mini_train \
    gpt2_mini_train_s512 gpt2_mini_train_s1024 gpt2_mini_b1 --cell-timeout 30 \
  --skip-cell gpt2_mini/cc/default --skip-cell gpt2_mini_s512/cc/default \
  --skip-cell gpt2_mini_s1024/cc/default --skip-cell gpt2_mini_train/cc/default \
  --skip-cell gpt2_mini_train_s512/cc/default --skip-cell gpt2_mini_train_s1024/cc/default \
  --skip-cell gpt2_mini_b1/cc/default

BENCH_VENV_PY=$HOME/.venvs/ocannl-bench/bin/python BENCH_DOMINANT_KERNEL=0 \
BENCH_RESULTS_DIR=/tmp/wave1003/720/tuf-measurement/large-batch \
$HOME/.venvs/ocannl-bench/bin/python benchmarks/orchestrate.py --skip-build \
  --gpu hip --only ocannl pytorch --profile exact approximate --precision bf16 f16 \
  --workloads gpt2_mini_train_b256 --cell-timeout 450 \
  --skip-cell gpt2_mini_train_b256/cc/default
```

The generated reports below are the durable measurement record; the original artifacts were
`/tmp/wave1003/720/tuf-measurement/{main,large-batch}/` and the adjacent measurement log.

## cc and Metal

The exclusive mac-studio measurement is pending. Its results will be appended here with
per-leg analysis, measured source revision, caps, skipped cells and generated reports.
cc reduced-format sequence/large-batch cells were deliberately left out of the one-hour
request; they remain pending. Any timed-out cells will also remain pending, not become
parity failures. Thus this report currently provides partial acceptance evidence for #720.

## HIP generated reports

### Ordinary endpoints — suite output, verbatim

# Benchmark results

platform: Linux-7.0.0-31-generic-x86_64-with-glibc2.43 x86_64 | ocannl commit: a4a425454 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager; reduced precisions get their own envelope: bf16 0.004, f16 0.002; the approximate regime 0.01)

measurement boxes declared by `fixtures/DIGESTS.txt`: m4-max, minix, rog-nv, tuf

ambient OCANNL_* environment: none


## gpt2_mini

measured on `gpt2_mini.safetensors`, sha256 `c322a00b72df143612eafafedf7c468e9c05f3e4bc1e7eb6d77ada651ef98de8`, m4-max,tuf's bytes

Rows are grouped by precision (f32 first), p50-ascending within each group.

`regime` is the numerics regime the row was measured and gated in (gh-ocannl-719). `exact` rows are gated at the exact envelope. `approximate` rows ran OCANNL under `--ocannl_profile=approximate` (tf32 matmuls, fp16 arithmetic, the C compiler's fast-math and contraction licence, the inlining refinement, and the algebraic-rewrite gates as they land) or torch under its own defaults (`high` matmul precision, scaled_dot_product_attention, cudnn.benchmark); they are gated at the approximate envelope 0.01, and their `parity` says whether they ALSO passed the exact envelope -- a rewrite that changes nothing measurable is a finding, not a pass. tinygrad has no exact pin, so its row stands in the approximate regime whenever the sweep has one. Exact rows come first within each precision. **`REGIME MISMATCH`** is a row whose runner reports having run under flags its regime does not own (an ambient OCANNL_PROFILE or OCANNL_TF32_MATMULS reaching an exact cell, say -- the regime is the resolution of the profile's keys, not the profile's name, and the row names the key and its source): its number belongs to neither column and is not comparable with anything -- the sweep fails on it, and the row is kept so the failure is visible where the numbers are read.

`peak MiB` is the peak device footprint over the cell's TIMED STEPS (gh-ocannl-1006) -- bracketed there rather than read at process exit, so a tuned cell's schedule search, which allocates a candidate buffer per arm, is not in it. `—` is a cell whose framework exposes no device counter on this backend (a pytorch `cpu` row, say): not a zero, and no host-RSS figure is substituted for it. The counters are NOT one quantity, so every measured row names its own after the number: `cuda-hw` = torch.cuda.max_memory_allocated (requested bytes, high-water); `ocannl-seam` = OCANNL allocator seam high-water (requested bytes, all backends). A high-water counter is exact over the window; one sampled at step boundaries is a lower bound on it, blind to an allocation made and given back within a step. Requested bytes off an allocator (OCANNL's seam, `torch.cuda.max_memory_allocated`) are the same quantity as each other and are NOT the device-wide figure a driver reports -- so rank rows within a counter, and read across counters only as the orders of magnitude they are.

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda(hip) | eager | f32 | exact | 7.312 | 7.272 | 7.337 | 7.170 | 0.24 | 105.2 cuda-hw | PASS (6.7e-08) | 140,044 |
| ocannl | hip | default | f32 | exact | 17.695 | 17.187 | 18.394 | 16.634 | 0.96 | 123.7 ocannl-seam | PASS (8.7e-07) | 57,870 |
| pytorch | cpu | eager | f32 | exact | 24.039 | 23.896 | 24.470 | 24.217 | 0.03 | — | REF | 42,597 |
| pytorch | cuda(hip) | eager | f32 | approximate | 7.409 | 7.392 | 7.453 | 7.259 | 0.24 | 101.2 cuda-hw | PASS (6.7e-08, within exact envelope) | 138,213 |
| ocannl | hip | default | f32 | approximate | 20.024 | 19.079 | 20.820 | 18.859 | 1.09 | 91.4 ocannl-seam | PASS (8.7e-07, within exact envelope) | 51,138 |
| pytorch | cpu | eager | f32 | approximate | 20.425 | 20.371 | 20.573 | 20.515 | 0.02 | — | PASS (1.3e-07, within exact envelope) | 50,135 |
| ocannl | hip | default | bf16 | exact | 17.480 | 16.968 | 17.852 | 16.456 | 1.64 | 61.9 ocannl-seam | PASS (4.5e-05) | 58,581 |
| ocannl | hip | default | bf16 | approximate | 20.786 | 20.163 | 21.606 | 19.540 | 2.19 | 45.8 ocannl-seam | PASS (1.0e-03, within exact envelope) | 49,264 |
| ocannl | hip | default | f16 | exact | 12.946 | 12.689 | 13.259 | 12.352 | 1.16 | 61.9 ocannl-seam | PASS (5.7e-05) | 79,096 |
| ocannl | hip | default | f16 | approximate | 13.774 | 13.572 | 13.950 | 13.097 | 1.27 | 45.8 ocannl-seam | PASS (1.0e-04, within exact envelope) | 74,343 |

## gpt2_mini_b1

measured on `gpt2_mini_b1.safetensors`, sha256 `5431f8c2956caa72acef8028ec37b98e433ebc9466c3a5727abda140078af922`, m4-max,tuf's bytes

**MISSING DIGEST RECORD:** declared measurement box(es) with no entry for `gpt2_mini_b1.safetensors`: minix, rog-nv

Rows are grouped by precision (f32 first), p50-ascending within each group.

`regime` is the numerics regime the row was measured and gated in (gh-ocannl-719). `exact` rows are gated at the exact envelope. `approximate` rows ran OCANNL under `--ocannl_profile=approximate` (tf32 matmuls, fp16 arithmetic, the C compiler's fast-math and contraction licence, the inlining refinement, and the algebraic-rewrite gates as they land) or torch under its own defaults (`high` matmul precision, scaled_dot_product_attention, cudnn.benchmark); they are gated at the approximate envelope 0.01, and their `parity` says whether they ALSO passed the exact envelope -- a rewrite that changes nothing measurable is a finding, not a pass. tinygrad has no exact pin, so its row stands in the approximate regime whenever the sweep has one. Exact rows come first within each precision. **`REGIME MISMATCH`** is a row whose runner reports having run under flags its regime does not own (an ambient OCANNL_PROFILE or OCANNL_TF32_MATMULS reaching an exact cell, say -- the regime is the resolution of the profile's keys, not the profile's name, and the row names the key and its source): its number belongs to neither column and is not comparable with anything -- the sweep fails on it, and the row is kept so the failure is visible where the numbers are read.

`peak MiB` is the peak device footprint over the cell's TIMED STEPS (gh-ocannl-1006) -- bracketed there rather than read at process exit, so a tuned cell's schedule search, which allocates a candidate buffer per arm, is not in it. `—` is a cell whose framework exposes no device counter on this backend (a pytorch `cpu` row, say): not a zero, and no host-RSS figure is substituted for it. The counters are NOT one quantity, so every measured row names its own after the number: `cuda-hw` = torch.cuda.max_memory_allocated (requested bytes, high-water); `ocannl-seam` = OCANNL allocator seam high-water (requested bytes, all backends). A high-water counter is exact over the window; one sampled at step boundaries is a lower bound on it, blind to an allocation made and given back within a step. Requested bytes off an allocator (OCANNL's seam, `torch.cuda.max_memory_allocated`) are the same quantity as each other and are NOT the device-wide figure a driver reports -- so rank rows within a counter, and read across counters only as the orders of magnitude they are.

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda(hip) | eager | f32 | exact | 2.558 | 2.544 | 2.593 | 2.471 | 0.23 | 51.6 cuda-hw | PASS (6.8e-08) | 50,032 |
| ocannl | hip | default | f32 | exact | 2.962 | 2.841 | 3.164 | 3.020 | 0.97 | 30.5 ocannl-seam | PASS (2.7e-07) | 43,218 |
| pytorch | cpu | eager | f32 | exact | 5.271 | 5.217 | 5.304 | 5.314 | 0.01 | — | REF | 24,285 |
| pytorch | cuda(hip) | eager | f32 | approximate | 2.379 | 2.370 | 2.398 | 2.280 | 0.24 | 51.1 cuda-hw | PASS (6.7e-08, within exact envelope) | 53,799 |
| pytorch | cpu | eager | f32 | approximate | 2.880 | 2.846 | 2.941 | 2.908 | 0.00 | — | PASS (6.7e-08, within exact envelope) | 44,451 |
| ocannl | hip | default | f32 | approximate | 3.310 | 3.219 | 3.362 | 3.336 | 1.34 | 26.5 ocannl-seam | PASS (2.7e-07, within exact envelope) | 38,670 |
| ocannl | hip | default | bf16 | exact | 2.248 | 2.216 | 2.290 | 2.279 | 1.65 | 15.3 ocannl-seam | PASS (1.0e-04) | 56,927 |
| ocannl | hip | default | bf16 | approximate | 3.343 | 3.304 | 3.390 | 3.432 | 2.15 | 13.3 ocannl-seam | PASS (1.4e-03, within exact envelope) | 38,290 |
| ocannl | hip | default | f16 | exact | 1.927 | 1.890 | 1.947 | 1.943 | 1.19 | 15.3 ocannl-seam | PASS (1.6e-04) | 66,409 |
| ocannl | hip | default | f16 | approximate | 2.240 | 2.129 | 2.281 | 2.287 | 1.25 | 13.3 ocannl-seam | PASS (7.5e-05, within exact envelope) | 57,155 |

## gpt2_mini_s1024

measured on `gpt2_mini_s1024.safetensors`, sha256 `57073d6f98aa7d310ab9bfbba6636c394afd8800cf1816fcd5109b52f1c363db`, m4-max,tuf's bytes

**MISSING DIGEST RECORD:** declared measurement box(es) with no entry for `gpt2_mini_s1024.safetensors`: minix, rog-nv

Rows are grouped by precision (f32 first), p50-ascending within each group.

`regime` is the numerics regime the row was measured and gated in (gh-ocannl-719). `exact` rows are gated at the exact envelope. `approximate` rows ran OCANNL under `--ocannl_profile=approximate` (tf32 matmuls, fp16 arithmetic, the C compiler's fast-math and contraction licence, the inlining refinement, and the algebraic-rewrite gates as they land) or torch under its own defaults (`high` matmul precision, scaled_dot_product_attention, cudnn.benchmark); they are gated at the approximate envelope 0.01, and their `parity` says whether they ALSO passed the exact envelope -- a rewrite that changes nothing measurable is a finding, not a pass. tinygrad has no exact pin, so its row stands in the approximate regime whenever the sweep has one. Exact rows come first within each precision. **`REGIME MISMATCH`** is a row whose runner reports having run under flags its regime does not own (an ambient OCANNL_PROFILE or OCANNL_TF32_MATMULS reaching an exact cell, say -- the regime is the resolution of the profile's keys, not the profile's name, and the row names the key and its source): its number belongs to neither column and is not comparable with anything -- the sweep fails on it, and the row is kept so the failure is visible where the numbers are read.

`peak MiB` is the peak device footprint over the cell's TIMED STEPS (gh-ocannl-1006) -- bracketed there rather than read at process exit, so a tuned cell's schedule search, which allocates a candidate buffer per arm, is not in it. `—` is a cell whose framework exposes no device counter on this backend (a pytorch `cpu` row, say): not a zero, and no host-RSS figure is substituted for it. The counters are NOT one quantity, so every measured row names its own after the number: `cuda-hw` = torch.cuda.max_memory_allocated (requested bytes, high-water); `ocannl-seam` = OCANNL allocator seam high-water (requested bytes, all backends). A high-water counter is exact over the window; one sampled at step boundaries is a lower bound on it, blind to an allocation made and given back within a step. Requested bytes off an allocator (OCANNL's seam, `torch.cuda.max_memory_allocated`) are the same quantity as each other and are NOT the device-wide figure a driver reports -- so rank rows within a counter, and read across counters only as the orders of magnitude they are.

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda(hip) | eager | f32 | exact | 20.442 | 20.187 | 20.567 | 20.144 | 0.24 | 167.1 cuda-hw | PASS (0.0e+00) | 50,093 |
| ocannl | hip | default | f32 | exact | 25.369 | 24.587 | 25.797 | 24.418 | 1.01 | 352.5 ocannl-seam | PASS (4.0e-07) | 40,365 |
| pytorch | cpu | eager | f32 | exact | 154.482 | 153.792 | 155.170 | 154.698 | 0.15 | — | REF | 6,629 |
| pytorch | cuda(hip) | eager | f32 | approximate | 14.302 | 14.260 | 14.366 | 14.084 | 0.23 | 148.1 cuda-hw | PASS (6.7e-08, within exact envelope) | 71,598 |
| pytorch | cpu | eager | f32 | approximate | 24.949 | 24.914 | 25.052 | 25.096 | 0.03 | — | PASS (0.0e+00, within exact envelope) | 41,044 |
| ocannl | hip | default | f32 | approximate | 59.104 | 57.786 | 60.703 | 57.935 | 1.32 | 96.2 ocannl-seam | PASS (3.4e-07, within exact envelope) | 17,326 |
| ocannl | hip | default | bf16 | exact | 20.829 | 20.451 | 21.127 | 20.120 | 1.65 | 178.3 ocannl-seam | PASS (4.2e-05) | 49,161 |
| ocannl | hip | default | bf16 | approximate | 42.491 | 41.572 | 43.562 | 39.709 | 2.16 | 50.1 ocannl-seam | PASS (1.0e-03, within exact envelope) | 24,099 |
| ocannl | hip | default | f16 | exact | 16.879 | 16.747 | 16.916 | 16.438 | 1.19 | 178.3 ocannl-seam | FAIL (4.5e-03) | 60,666 |
| ocannl | hip | default | f16 | approximate | 33.848 | 32.931 | 34.335 | 31.838 | 1.26 | 50.1 ocannl-seam | PASS (3.6e-05, within exact envelope) | 30,253 |

## gpt2_mini_s512

measured on `gpt2_mini_s512.safetensors`, sha256 `dfbb26d9d3907f7d9cec3169715ca582982619b27d2455d40647593ac0b61172`, m4-max,tuf's bytes

**MISSING DIGEST RECORD:** declared measurement box(es) with no entry for `gpt2_mini_s512.safetensors`: minix, rog-nv

Rows are grouped by precision (f32 first), p50-ascending within each group.

`regime` is the numerics regime the row was measured and gated in (gh-ocannl-719). `exact` rows are gated at the exact envelope. `approximate` rows ran OCANNL under `--ocannl_profile=approximate` (tf32 matmuls, fp16 arithmetic, the C compiler's fast-math and contraction licence, the inlining refinement, and the algebraic-rewrite gates as they land) or torch under its own defaults (`high` matmul precision, scaled_dot_product_attention, cudnn.benchmark); they are gated at the approximate envelope 0.01, and their `parity` says whether they ALSO passed the exact envelope -- a rewrite that changes nothing measurable is a finding, not a pass. tinygrad has no exact pin, so its row stands in the approximate regime whenever the sweep has one. Exact rows come first within each precision. **`REGIME MISMATCH`** is a row whose runner reports having run under flags its regime does not own (an ambient OCANNL_PROFILE or OCANNL_TF32_MATMULS reaching an exact cell, say -- the regime is the resolution of the profile's keys, not the profile's name, and the row names the key and its source): its number belongs to neither column and is not comparable with anything -- the sweep fails on it, and the row is kept so the failure is visible where the numbers are read.

`peak MiB` is the peak device footprint over the cell's TIMED STEPS (gh-ocannl-1006) -- bracketed there rather than read at process exit, so a tuned cell's schedule search, which allocates a candidate buffer per arm, is not in it. `—` is a cell whose framework exposes no device counter on this backend (a pytorch `cpu` row, say): not a zero, and no host-RSS figure is substituted for it. The counters are NOT one quantity, so every measured row names its own after the number: `cuda-hw` = torch.cuda.max_memory_allocated (requested bytes, high-water); `ocannl-seam` = OCANNL allocator seam high-water (requested bytes, all backends). A high-water counter is exact over the window; one sampled at step boundaries is a lower bound on it, blind to an allocation made and given back within a step. Requested bytes off an allocator (OCANNL's seam, `torch.cuda.max_memory_allocated`) are the same quantity as each other and are NOT the device-wide figure a driver reports -- so rank rows within a counter, and read across counters only as the orders of magnitude they are.

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda(hip) | eager | f32 | exact | 11.413 | 11.372 | 11.444 | 11.295 | 0.24 | 117.8 cuda-hw | PASS (6.7e-08) | 89,725 |
| ocannl | hip | default | f32 | exact | 22.489 | 22.137 | 22.866 | 21.648 | 1.00 | 221.0 ocannl-seam | PASS (1.2e-06) | 45,533 |
| pytorch | cpu | eager | f32 | exact | 52.389 | 52.269 | 52.612 | 52.658 | 0.05 | — | REF | 19,546 |
| pytorch | cuda(hip) | eager | f32 | approximate | 10.748 | 10.706 | 10.796 | 10.621 | 0.23 | 107.8 cuda-hw | PASS (6.7e-08, within exact envelope) | 95,278 |
| pytorch | cpu | eager | f32 | approximate | 22.788 | 22.692 | 22.816 | 22.798 | 0.03 | — | PASS (0.0e+00, within exact envelope) | 44,936 |
| ocannl | hip | default | f32 | approximate | 38.174 | 37.577 | 39.379 | 36.019 | 1.16 | 92.7 ocannl-seam | PASS (1.1e-06, within exact envelope) | 26,825 |
| ocannl | hip | default | bf16 | exact | 20.016 | 19.767 | 20.093 | 19.159 | 1.65 | 111.0 ocannl-seam | PASS (5.9e-05) | 51,160 |
| ocannl | hip | default | bf16 | approximate | 33.097 | 32.579 | 34.121 | 31.224 | 2.19 | 46.9 ocannl-seam | PASS (1.0e-03, within exact envelope) | 30,939 |
| ocannl | hip | default | f16 | exact | 15.629 | 15.546 | 15.825 | 15.322 | 1.20 | 111.0 ocannl-seam | PASS (4.2e-05) | 65,520 |
| ocannl | hip | default | f16 | approximate | 25.786 | 25.326 | 26.183 | 24.503 | 1.31 | 46.9 ocannl-seam | PASS (1.0e-04, within exact envelope) | 39,712 |

## gpt2_mini_train

measured on `gpt2_mini_train.safetensors`, sha256 `2c2458fd4f8fcb8073929bf8b6bedf74c7e5d1dbb77839b632ea216207ade50d`, m4-max,tuf's bytes

**MISSING DIGEST RECORD:** declared measurement box(es) with no entry for `gpt2_mini_train.safetensors`: minix, rog-nv

Rows are grouped by precision (f32 first), p50-ascending within each group.

`regime` is the numerics regime the row was measured and gated in (gh-ocannl-719). `exact` rows are gated at the exact envelope. `approximate` rows ran OCANNL under `--ocannl_profile=approximate` (tf32 matmuls, fp16 arithmetic, the C compiler's fast-math and contraction licence, the inlining refinement, and the algebraic-rewrite gates as they land) or torch under its own defaults (`high` matmul precision, scaled_dot_product_attention, cudnn.benchmark); they are gated at the approximate envelope 0.01, and their `parity` says whether they ALSO passed the exact envelope -- a rewrite that changes nothing measurable is a finding, not a pass. tinygrad has no exact pin, so its row stands in the approximate regime whenever the sweep has one. Exact rows come first within each precision. **`REGIME MISMATCH`** is a row whose runner reports having run under flags its regime does not own (an ambient OCANNL_PROFILE or OCANNL_TF32_MATMULS reaching an exact cell, say -- the regime is the resolution of the profile's keys, not the profile's name, and the row names the key and its source): its number belongs to neither column and is not comparable with anything -- the sweep fails on it, and the row is kept so the failure is visible where the numbers are read.

`peak MiB` is the peak device footprint over the cell's TIMED STEPS (gh-ocannl-1006) -- bracketed there rather than read at process exit, so a tuned cell's schedule search, which allocates a candidate buffer per arm, is not in it. `—` is a cell whose framework exposes no device counter on this backend (a pytorch `cpu` row, say): not a zero, and no host-RSS figure is substituted for it. The counters are NOT one quantity, so every measured row names its own after the number: `cuda-hw` = torch.cuda.max_memory_allocated (requested bytes, high-water); `ocannl-seam` = OCANNL allocator seam high-water (requested bytes, all backends). A high-water counter is exact over the window; one sampled at step boundaries is a lower bound on it, blind to an allocation made and given back within a step. Requested bytes off an allocator (OCANNL's seam, `torch.cuda.max_memory_allocated`) are the same quantity as each other and are NOT the device-wide figure a driver reports -- so rank rows within a counter, and read across counters only as the orders of magnitude they are.

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda(hip) | eager | f32 | exact | 21.040 | 20.859 | 21.291 | 20.349 | 0.68 | 292.2 cuda-hw | PASS (6.8e-08) | 48,670 |
| ocannl | hip | default | f32 | exact | 42.930 | 41.643 | 44.113 | 40.649 | 5.32 | 243.0 ocannl-seam | PASS (1.1e-06) | 23,853 |
| pytorch | cpu | eager | f32 | exact | 76.199 | 74.434 | 77.997 | 74.474 | 0.42 | — | REF | 13,439 |
| pytorch | cuda(hip) | eager | f32 | approximate | 20.463 | 20.311 | 20.625 | 19.778 | 0.68 | 276.2 cuda-hw | PASS (6.8e-08, within exact envelope) | 50,040 |
| ocannl | hip | default | f32 | approximate | 48.493 | 47.329 | 49.467 | 45.876 | 5.38 | 194.9 ocannl-seam | PASS (1.0e-06, within exact envelope) | 21,116 |
| pytorch | cpu | eager | f32 | approximate | 68.104 | 67.598 | 69.848 | 68.404 | 0.42 | — | PASS (6.7e-08, within exact envelope) | 15,036 |
| ocannl | hip | default | bf16 | exact | 56.447 | 54.377 | 56.688 | 52.539 | 7.19 | 129.5 ocannl-seam | PASS (3.7e-04) | 18,141 |
| ocannl | hip | default | bf16 | approximate | 71.838 | 70.066 | 74.592 | 69.371 | 15.00 | 105.6 ocannl-seam | PASS (1.2e-03, within exact envelope) | 14,254 |
| ocannl | hip | default | f16 | exact | 73.005 | 71.852 | 73.775 | 72.602 | 9.05 | 144.2 ocannl-seam | DIVERGED (loss non-finite from step 0) | 14,027 |
| ocannl | hip | default | f16 | approximate | 76.854 | 76.170 | 77.791 | 76.510 | 9.02 | 120.2 ocannl-seam | PASS (5.0e-03, beyond exact envelope) | 13,324 |

## gpt2_mini_train_s1024

measured on `gpt2_mini_train_s1024.safetensors`, sha256 `bcaab56e4359c5cdba46b3953d955f8dfc2736029f485f8d0ff8709ced844f6e`, m4-max,tuf's bytes

**MISSING DIGEST RECORD:** declared measurement box(es) with no entry for `gpt2_mini_train_s1024.safetensors`: minix, rog-nv

Rows are grouped by precision (f32 first), p50-ascending within each group.

`regime` is the numerics regime the row was measured and gated in (gh-ocannl-719). `exact` rows are gated at the exact envelope. `approximate` rows ran OCANNL under `--ocannl_profile=approximate` (tf32 matmuls, fp16 arithmetic, the C compiler's fast-math and contraction licence, the inlining refinement, and the algebraic-rewrite gates as they land) or torch under its own defaults (`high` matmul precision, scaled_dot_product_attention, cudnn.benchmark); they are gated at the approximate envelope 0.01, and their `parity` says whether they ALSO passed the exact envelope -- a rewrite that changes nothing measurable is a finding, not a pass. tinygrad has no exact pin, so its row stands in the approximate regime whenever the sweep has one. Exact rows come first within each precision. **`REGIME MISMATCH`** is a row whose runner reports having run under flags its regime does not own (an ambient OCANNL_PROFILE or OCANNL_TF32_MATMULS reaching an exact cell, say -- the regime is the resolution of the profile's keys, not the profile's name, and the row names the key and its source): its number belongs to neither column and is not comparable with anything -- the sweep fails on it, and the row is kept so the failure is visible where the numbers are read.

`peak MiB` is the peak device footprint over the cell's TIMED STEPS (gh-ocannl-1006) -- bracketed there rather than read at process exit, so a tuned cell's schedule search, which allocates a candidate buffer per arm, is not in it. `—` is a cell whose framework exposes no device counter on this backend (a pytorch `cpu` row, say): not a zero, and no host-RSS figure is substituted for it. The counters are NOT one quantity, so every measured row names its own after the number: `cuda-hw` = torch.cuda.max_memory_allocated (requested bytes, high-water); `ocannl-seam` = OCANNL allocator seam high-water (requested bytes, all backends). A high-water counter is exact over the window; one sampled at step boundaries is a lower bound on it, blind to an allocation made and given back within a step. Requested bytes off an allocator (OCANNL's seam, `torch.cuda.max_memory_allocated`) are the same quantity as each other and are NOT the device-wide figure a driver reports -- so rank rows within a counter, and read across counters only as the orders of magnitude they are.

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda(hip) | eager | f32 | exact | 68.153 | 67.976 | 68.823 | 68.050 | 0.69 | 475.4 cuda-hw | PASS (1.4e-07) | 15,025 |
| ocannl | hip | default | f32 | exact | 76.148 | 75.719 | 77.704 | 75.676 | 3.65 | 694.9 ocannl-seam | PASS (8.1e-07) | 13,448 |
| pytorch | cpu | eager | f32 | exact | 448.415 | 403.747 | 476.320 | 447.709 | 0.86 | — | REF | 2,284 |
| pytorch | cuda(hip) | eager | f32 | approximate | 34.372 | 33.931 | 34.901 | 33.250 | 0.72 | 443.4 cuda-hw | PASS (1.4e-07, within exact envelope) | 29,792 |
| pytorch | cpu | eager | f32 | approximate | 86.773 | 84.261 | 87.390 | 86.079 | 0.43 | — | PASS (1.3e-07, within exact envelope) | 11,801 |
| ocannl | hip | default | f32 | approximate | 108.148 | 107.629 | 108.656 | 108.275 | 4.02 | 310.8 ocannl-seam | PASS (8.1e-07, within exact envelope) | 9,469 |
| ocannl | hip | default | bf16 | exact | 75.907 | 74.000 | 78.044 | 73.347 | 5.52 | 357.4 ocannl-seam | PASS (8.3e-04) | 13,490 |
| ocannl | hip | default | bf16 | approximate | 115.331 | 113.705 | 117.346 | 114.888 | 13.28 | 165.5 ocannl-seam | PASS (1.4e-03, within exact envelope) | 8,879 |
| ocannl | hip | default | f16 | exact | 98.952 | 97.923 | 99.586 | 98.370 | 7.03 | 373.0 ocannl-seam | FAIL (4.1e-03) | 10,348 |
| ocannl | hip | default | f16 | approximate | 117.593 | 116.668 | 118.189 | 117.984 | 7.15 | 181.0 ocannl-seam | PASS (2.7e-04, within exact envelope) | 8,708 |

## gpt2_mini_train_s512

measured on `gpt2_mini_train_s512.safetensors`, sha256 `3045392f435637f3b46e48a21d23e48053f75a02209281fb20853eede2622875`, m4-max,tuf's bytes

**MISSING DIGEST RECORD:** declared measurement box(es) with no entry for `gpt2_mini_train_s512.safetensors`: minix, rog-nv

Rows are grouped by precision (f32 first), p50-ascending within each group.

`regime` is the numerics regime the row was measured and gated in (gh-ocannl-719). `exact` rows are gated at the exact envelope. `approximate` rows ran OCANNL under `--ocannl_profile=approximate` (tf32 matmuls, fp16 arithmetic, the C compiler's fast-math and contraction licence, the inlining refinement, and the algebraic-rewrite gates as they land) or torch under its own defaults (`high` matmul precision, scaled_dot_product_attention, cudnn.benchmark); they are gated at the approximate envelope 0.01, and their `parity` says whether they ALSO passed the exact envelope -- a rewrite that changes nothing measurable is a finding, not a pass. tinygrad has no exact pin, so its row stands in the approximate regime whenever the sweep has one. Exact rows come first within each precision. **`REGIME MISMATCH`** is a row whose runner reports having run under flags its regime does not own (an ambient OCANNL_PROFILE or OCANNL_TF32_MATMULS reaching an exact cell, say -- the regime is the resolution of the profile's keys, not the profile's name, and the row names the key and its source): its number belongs to neither column and is not comparable with anything -- the sweep fails on it, and the row is kept so the failure is visible where the numbers are read.

`peak MiB` is the peak device footprint over the cell's TIMED STEPS (gh-ocannl-1006) -- bracketed there rather than read at process exit, so a tuned cell's schedule search, which allocates a candidate buffer per arm, is not in it. `—` is a cell whose framework exposes no device counter on this backend (a pytorch `cpu` row, say): not a zero, and no host-RSS figure is substituted for it. The counters are NOT one quantity, so every measured row names its own after the number: `cuda-hw` = torch.cuda.max_memory_allocated (requested bytes, high-water); `ocannl-seam` = OCANNL allocator seam high-water (requested bytes, all backends). A high-water counter is exact over the window; one sampled at step boundaries is a lower bound on it, blind to an allocation made and given back within a step. Requested bytes off an allocator (OCANNL's seam, `torch.cuda.max_memory_allocated`) are the same quantity as each other and are NOT the device-wide figure a driver reports -- so rank rows within a counter, and read across counters only as the orders of magnitude they are.

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda(hip) | eager | f32 | exact | 30.295 | 30.088 | 31.006 | 29.394 | 0.73 | 394.1 cuda-hw | PASS (1.3e-07) | 33,800 |
| ocannl | hip | default | f32 | exact | 51.946 | 50.612 | 53.108 | 48.878 | 4.18 | 436.4 ocannl-seam | PASS (6.7e-07) | 19,713 |
| pytorch | cpu | eager | f32 | exact | 149.806 | 149.114 | 164.128 | 151.037 | 0.53 | — | REF | 6,836 |
| pytorch | cuda(hip) | eager | f32 | approximate | 27.884 | 27.357 | 28.324 | 26.859 | 0.68 | 330.1 cuda-hw | PASS (1.3e-07, within exact envelope) | 36,724 |
| ocannl | hip | default | f32 | approximate | 73.555 | 72.534 | 75.757 | 71.985 | 4.33 | 244.3 ocannl-seam | PASS (6.0e-07, within exact envelope) | 13,922 |
| pytorch | cpu | eager | f32 | approximate | 76.621 | 74.799 | 78.517 | 78.728 | 0.42 | — | PASS (0.0e+00, within exact envelope) | 13,364 |
| ocannl | hip | default | bf16 | exact | 63.861 | 61.750 | 66.172 | 60.227 | 5.86 | 226.7 ocannl-seam | PASS (5.7e-04) | 16,035 |
| ocannl | hip | default | bf16 | approximate | 94.827 | 91.967 | 96.192 | 92.248 | 13.72 | 130.8 ocannl-seam | PASS (1.1e-03, within exact envelope) | 10,799 |
| ocannl | hip | default | f16 | exact | 79.427 | 78.529 | 80.509 | 78.717 | 7.67 | 241.7 ocannl-seam | PASS (1.5e-04) | 12,892 |
| ocannl | hip | default | f16 | approximate | 92.630 | 92.443 | 93.479 | 92.856 | 7.83 | 145.8 ocannl-seam | PASS (6.8e-04, within exact envelope) | 11,055 |

## Cells skipped

Cells this sweep did not run. Their absence above is a choice, not a measurement and not a failure.

| cell | why |
|---|---|
| gpt2_mini ocannl/cc/default | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini ocannl/cc/default [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini ocannl/cc/default/bf16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini ocannl/cc/default/bf16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini ocannl/cc/default/f16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini ocannl/cc/default/f16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_b1 ocannl/cc/default | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_b1 ocannl/cc/default [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_b1 ocannl/cc/default/bf16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_b1 ocannl/cc/default/bf16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_b1 ocannl/cc/default/f16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_b1 ocannl/cc/default/f16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_s1024 ocannl/cc/default | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_s1024 ocannl/cc/default [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_s1024 ocannl/cc/default/bf16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_s1024 ocannl/cc/default/bf16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_s1024 ocannl/cc/default/f16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_s1024 ocannl/cc/default/f16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_s512 ocannl/cc/default | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_s512 ocannl/cc/default [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_s512 ocannl/cc/default/bf16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_s512 ocannl/cc/default/bf16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_s512 ocannl/cc/default/f16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_s512 ocannl/cc/default/f16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train ocannl/cc/default | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train ocannl/cc/default [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train ocannl/cc/default/bf16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train ocannl/cc/default/bf16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train ocannl/cc/default/f16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train ocannl/cc/default/f16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_s1024 ocannl/cc/default | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_s1024 ocannl/cc/default [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_s1024 ocannl/cc/default/bf16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_s1024 ocannl/cc/default/bf16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_s1024 ocannl/cc/default/f16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_s1024 ocannl/cc/default/f16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_s512 ocannl/cc/default | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_s512 ocannl/cc/default [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_s512 ocannl/cc/default/bf16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_s512 ocannl/cc/default/bf16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_s512 ocannl/cc/default/f16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_s512 ocannl/cc/default/f16 [approximate] | --skip-cell (the operator left it out of this sweep) |

### Batch256 — suite output, verbatim

# Benchmark results

platform: Linux-7.0.0-31-generic-x86_64-with-glibc2.43 x86_64 | ocannl commit: a4a425454 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager; reduced precisions get their own envelope: bf16 0.004, f16 0.002; the approximate regime 0.01)

measurement boxes declared by `fixtures/DIGESTS.txt`: m4-max, minix, rog-nv, tuf

ambient OCANNL_* environment: none


## gpt2_mini_train_b256

measured on `gpt2_mini_train_b256.safetensors`, sha256 `61cc0aff416cbfc330d025b1fdff8b161e0ff86546343c1e578c179ff28c97ec`, m4-max,tuf's bytes

**MISSING DIGEST RECORD:** declared measurement box(es) with no entry for `gpt2_mini_train_b256.safetensors`: minix, rog-nv

Rows are grouped by precision (f32 first), p50-ascending within each group.

`regime` is the numerics regime the row was measured and gated in (gh-ocannl-719). `exact` rows are gated at the exact envelope. `approximate` rows ran OCANNL under `--ocannl_profile=approximate` (tf32 matmuls, fp16 arithmetic, the C compiler's fast-math and contraction licence, the inlining refinement, and the algebraic-rewrite gates as they land) or torch under its own defaults (`high` matmul precision, scaled_dot_product_attention, cudnn.benchmark); they are gated at the approximate envelope 0.01, and their `parity` says whether they ALSO passed the exact envelope -- a rewrite that changes nothing measurable is a finding, not a pass. tinygrad has no exact pin, so its row stands in the approximate regime whenever the sweep has one. Exact rows come first within each precision. **`REGIME MISMATCH`** is a row whose runner reports having run under flags its regime does not own (an ambient OCANNL_PROFILE or OCANNL_TF32_MATMULS reaching an exact cell, say -- the regime is the resolution of the profile's keys, not the profile's name, and the row names the key and its source): its number belongs to neither column and is not comparable with anything -- the sweep fails on it, and the row is kept so the failure is visible where the numbers are read.

`peak MiB` is the peak device footprint over the cell's TIMED STEPS (gh-ocannl-1006) -- bracketed there rather than read at process exit, so a tuned cell's schedule search, which allocates a candidate buffer per arm, is not in it. `—` is a cell whose framework exposes no device counter on this backend (a pytorch `cpu` row, say): not a zero, and no host-RSS figure is substituted for it. The counters are NOT one quantity, so every measured row names its own after the number: `cuda-hw` = torch.cuda.max_memory_allocated (requested bytes, high-water); `ocannl-seam` = OCANNL allocator seam high-water (requested bytes, all backends). A high-water counter is exact over the window; one sampled at step boundaries is a lower bound on it, blind to an allocation made and given back within a step. Requested bytes off an allocator (OCANNL's seam, `torch.cuda.max_memory_allocated`) are the same quantity as each other and are NOT the device-wide figure a driver reports -- so rank rows within a counter, and read across counters only as the orders of magnitude they are.

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda(hip) | eager | f32 | exact | 648.320 | 621.963 | 652.790 | 652.580 | 1.33 | 6,574.6 cuda-hw | PASS (6.7e-08) | 50,543 |
| ocannl | hip | default | f32 | exact | 4200.920 | 4131.580 | 4240.900 | 4250.820 | 4.11 | 6,624.5 ocannl-seam | PASS (2.2e-06) | 7,800 |
| pytorch | cpu | eager | f32 | exact | 6206.550 | 6187.967 | 6229.418 | 6196.977 | 6.91 | — | REF | 5,280 |
| pytorch | cuda(hip) | eager | f32 | approximate | 634.044 | 610.018 | 636.025 | 636.927 | 1.28 | 6,062.6 cuda-hw | PASS (6.7e-08, within exact envelope) | 51,681 |
| ocannl | hip | default | f32 | approximate | 4404.050 | 4385.360 | 4434.820 | 4418.990 | 4.12 | 5,084.5 ocannl-seam | PASS (2.2e-06, within exact envelope) | 7,440 |
| pytorch | cpu | eager | f32 | approximate | 4888.302 | 4873.965 | 4915.561 | 4907.947 | 5.33 | — | PASS (6.7e-08, within exact envelope) | 6,703 |
| ocannl | hip | default | bf16 | exact | 4504.490 | 4421.440 | 4554.480 | 4485.670 | 5.92 | 3,320.9 ocannl-seam | PASS (2.7e-04) | 7,275 |
| ocannl | hip | default | bf16 | approximate | 5030.660 | 5000.640 | 5059.640 | 5047.430 | 13.61 | 2,552.9 ocannl-seam | PASS (1.1e-03, within exact envelope) | 6,514 |
| ocannl | hip | default | f16 | exact | 3879.630 | 3845.500 | 3898.380 | 3889.830 | 7.74 | 3,382.0 ocannl-seam | PASS (1.4e-03) | 8,446 |
| ocannl | hip | default | f16 | approximate | 3963.110 | 3948.130 | 4020.260 | 3964.710 | 7.77 | 2,614.0 ocannl-seam | PASS (6.2e-04, within exact envelope) | 8,268 |

## Cells skipped

Cells this sweep did not run. Their absence above is a choice, not a measurement and not a failure.

| cell | why |
|---|---|
| gpt2_mini_train_b256 ocannl/cc/default | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_b256 ocannl/cc/default [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_b256 ocannl/cc/default/bf16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_b256 ocannl/cc/default/bf16 [approximate] | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_b256 ocannl/cc/default/f16 | --skip-cell (the operator left it out of this sweep) |
| gpt2_mini_train_b256 ocannl/cc/default/f16 [approximate] | --skip-cell (the operator left it out of this sweep) |

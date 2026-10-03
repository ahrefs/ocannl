# gh-ocannl-720 leg 1: Exact and approximate regimes on HIP

All f32 cells pass. Approximate increases OCANNL HIP p50 on every endpoint in this
default schedule sweep, despite reducing several activation footprints. For base inference
it is 20.024 ms versus 17.695 ms exact; torch eager is 7.409 versus 7.312 ms.
This reports the regime column landed in staging#661, without promising a profile speedup.

Measured 2026-10-03 under exclusive request `wave1003-720-tuf-amd-linux-2` on
`tuf-amd-linux`, HIP gfx1102, **discrete** VRAM (8,573,157,376 bytes).
Source revision: `a4a425454cb9bf12c4c24b2c0d5bb6183b1c64ff` (before the PR rebase).
Linux 7.0.0-31-generic, x86_64, glibc 2.43; PyTorch `2.13.0+rocm7.1`.
This is one sweep with the fixture's full protocol, not repeated independent trials.
OCANNL uses the **default, untuned** schedule (`searched=false`); PyTorch is eager.
No torch.compile or tinygrad arm, schedule search, or dominant-kernel diagnostics was run.
The ambient `OCANNL_*` environment was empty. Graph capture was left at its configured
default (`gpu_graph_capture=true`); no capture-success receipt was collected.
cc and Metal rows are pending, so this is partial evidence for gh-ocannl-720.

Each cell names its regime: `exact` uses no OCANNL profile, while `approximate` uses
`--ocannl_profile=approximate`. Torch exact uses `highest` matmul precision, disables
TF32 and cudnn benchmarking, and composes attention; approximate uses `high`,
scaled_dot_product_attention and cudnn benchmarking. These are whole-profile comparisons,
not isolated rewrite ablations, and gfx1102 does not supply CUDA TF32 evidence for gh-ocannl-719.
Parity compares the fixture's first losses to the exact torch CPU oracle: tolerances f32/f16
0.002, bf16 0.004, approximate 0.01 (maximum relative loss difference).
An approximate PASS below also meets its precision's exact envelope.

Per-step p10/p50/p90 synchronize each step; queued ms uses one final synchronization.
Tokens/s = tokens per step / p50 seconds. Compile seconds are separate.
Peak MiB names its counter: `ocannl-seam` is OCANNL's requested-byte allocator high-water;
`cuda-hw` is torch.cuda.max_memory_allocated on HIP. Both cover timed steps, not search;
rank within a counter and treat cross-counter comparisons as approximate. CPU has no device
counter. Failed parity cells have their performance columns suppressed below.

The complete sweep produced 80 rows: 69 PASS, 8 REF, 2 FAIL, 1 DIVERGED, with no runner
failures or timeouts. Exit 1 is from parity. All f32, bf16, approximate, and batch-256 rows pass.
The exact-f16 failures are detailed in [leg 7](report-gh720-leg7-hip.md).
The torch CPU rows are reference-framework timings, not OCANNL cc results.

Reproduce with the suite driver and the identical fixture bytes, after building under a
correctness reservation. This timing window ran two invocations of the following matrix:

```sh
BENCH_VENV_PY=$HOME/.venvs/ocannl-bench/bin/python BENCH_DOMINANT_KERNEL=0 \
  $HOME/.venvs/ocannl-bench/bin/python benchmarks/orchestrate.py --skip-build \
  --gpu hip --only ocannl pytorch --profile exact approximate --precision bf16 f16 \
  --workloads <workloads> --cell-timeout <cap> --skip-cell <workload>/cc/default ...
```

`main` included `gpt2_mini`, `gpt2_mini_s512`, `gpt2_mini_s1024`, `gpt2_mini_train`,
`gpt2_mini_train_s512`, `gpt2_mini_train_s1024`, and `gpt2_mini_b1`, at 30 seconds/cell; `large-batch`
included only `gpt2_mini_train_b256`, at 450 seconds/cell. Each workload's cc/default
cell was explicitly skipped in both regimes and all precisions. The outer cap was 6900 seconds.
Build/check/scans had passed at `/home/lukstafi/.ocannl-test-runs/20261003T170038Z-2942453`;
arm and replay preparations were at `/tmp/wave1003/720/tuf-correctness`.
Original driver artifacts: `/tmp/wave1003/720/tuf-measurement/{main,large-batch}/` on TUF;
log: `/tmp/wave1003/720/tuf-measurement.log`. The tables here transcribe their `results.jsonl`.

## Fixtures and protocol

All fixtures match `m4-max,tuf` content-v1 origins; they were copied to TUF, never
regenerated there. The declared measuring boxes are m4-max, minix, rog-nv, tuf.
Full digests below identify every cell by workload. P/W/T are parity / warmup / timed
steps; timed steps run twice (synced then queued). Training uses SGD lr 0.01.

| workload | SHA-256 (content-v1) | bytes | seq × batch | tokens/step | P/W/T | missing declared origins |
|---|---|---|---|---|---|---|
| gpt2_mini | `c322a00b72df143612eafafedf7c468e9c05f3e4bc1e7eb6d77ada651ef98de8` | 13871384 | 128 × 8 | 1024 | 8/5/20 | none |
| gpt2_mini_s512 | `dfbb26d9d3907f7d9cec3169715ca582982619b27d2455d40647593ac0b61172` | 14231832 | 512 × 2 | 1024 | 4/3/10 | minix, rog-nv |
| gpt2_mini_s1024 | `57073d6f98aa7d310ab9bfbba6636c394afd8800cf1816fcd5109b52f1c363db` | 14756128 | 1024 × 1 | 1024 | 4/3/10 | minix, rog-nv |
| gpt2_mini_train | `2c2458fd4f8fcb8073929bf8b6bedf74c7e5d1dbb77839b632ea216207ade50d` | 13838624 | 128 × 8 | 1024 | 6/3/10 | minix, rog-nv |
| gpt2_mini_train_s512 | `3045392f435637f3b46e48a21d23e48053f75a02209281fb20853eede2622875` | 14231840 | 512 × 2 | 1024 | 6/3/10 | minix, rog-nv |
| gpt2_mini_train_s1024 | `bcaab56e4359c5cdba46b3953d955f8dfc2736029f485f8d0ff8709ced844f6e` | 14756136 | 1024 × 1 | 1024 | 6/3/10 | minix, rog-nv |
| gpt2_mini_b1 | `5431f8c2956caa72acef8028ec37b98e433ebc9466c3a5727abda140078af922` | 13809936 | 128 × 1 | 128 | 4/5/20 | minix, rog-nv |
| gpt2_mini_train_b256 | `61cc0aff416cbfc330d025b1fdff8b161e0ff86546343c1e578c179ff28c97ec` | 14068008 | 128 × 256 | 32768 | 6/3/10 | minix, rog-nv |

## Cells

| workload | framework/backend/variant | precision | regime | p10 / p50 / p90 ms | queued ms | compile s | peak MiB / counter | parity (max rel) | tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| gpt2_mini | ocannl/hip/default | f32 | exact | 17.187 / 17.695 / 18.394 | 16.634 | 0.956 | 123.7 / ocannl-seam | PASS (8.72258e-07) | 57,870 |
| gpt2_mini | pytorch/cpu/eager | f32 | exact | 23.896 / 24.039 / 24.470 | 24.217 | 0.026 | — | REF (0) | 42,597 |
| gpt2_mini | pytorch/cuda(hip)/eager | f32 | exact | 7.272 / 7.312 / 7.337 | 7.170 | 0.236 | 105.2 / cuda-hw | PASS (6.71796e-08) | 140,044 |
| gpt2_mini | ocannl/hip/default | f32 | approximate | 19.079 / 20.024 / 20.820 | 18.859 | 1.093 | 91.4 / ocannl-seam | PASS (8.72258e-07); within exact | 51,138 |
| gpt2_mini | pytorch/cpu/eager | f32 | approximate | 20.371 / 20.425 / 20.573 | 20.515 | 0.022 | — | PASS (1.34085e-07); within exact | 50,135 |
| gpt2_mini | pytorch/cuda(hip)/eager | f32 | approximate | 7.392 / 7.409 / 7.453 | 7.259 | 0.239 | 101.2 / cuda-hw | PASS (6.73391e-08); within exact | 138,213 |
| gpt2_mini_s512 | ocannl/hip/default | f32 | exact | 22.137 / 22.489 / 22.866 | 21.648 | 0.999 | 221.0 / ocannl-seam | PASS (1.20968e-06) | 45,533 |
| gpt2_mini_s512 | pytorch/cpu/eager | f32 | exact | 52.269 / 52.389 / 52.612 | 52.658 | 0.054 | — | REF (0) | 19,546 |
| gpt2_mini_s512 | pytorch/cuda(hip)/eager | f32 | exact | 11.372 / 11.413 / 11.444 | 11.295 | 0.237 | 117.8 / cuda-hw | PASS (6.71807e-08) | 89,725 |
| gpt2_mini_s512 | ocannl/hip/default | f32 | approximate | 37.577 / 38.174 / 39.379 | 36.019 | 1.158 | 92.7 / ocannl-seam | PASS (1.14205e-06); within exact | 26,825 |
| gpt2_mini_s512 | pytorch/cpu/eager | f32 | approximate | 22.692 / 22.788 / 22.816 | 22.798 | 0.026 | — | PASS (0); within exact | 44,936 |
| gpt2_mini_s512 | pytorch/cuda(hip)/eager | f32 | approximate | 10.706 / 10.748 / 10.796 | 10.621 | 0.235 | 107.8 / cuda-hw | PASS (6.71807e-08); within exact | 95,278 |
| gpt2_mini_s1024 | ocannl/hip/default | f32 | exact | 24.587 / 25.369 / 25.797 | 24.418 | 1.008 | 352.5 / ocannl-seam | PASS (4.04093e-07) | 40,365 |
| gpt2_mini_s1024 | pytorch/cpu/eager | f32 | exact | 153.792 / 154.482 / 155.170 | 154.698 | 0.155 | — | REF (0) | 6,629 |
| gpt2_mini_s1024 | pytorch/cuda(hip)/eager | f32 | exact | 20.187 / 20.442 / 20.567 | 20.144 | 0.242 | 167.1 / cuda-hw | PASS (0) | 50,093 |
| gpt2_mini_s1024 | ocannl/hip/default | f32 | approximate | 57.786 / 59.104 / 60.703 | 57.935 | 1.316 | 96.2 / ocannl-seam | PASS (3.36367e-07); within exact | 17,326 |
| gpt2_mini_s1024 | pytorch/cpu/eager | f32 | approximate | 24.914 / 24.949 / 25.052 | 25.096 | 0.034 | — | PASS (0); within exact | 41,044 |
| gpt2_mini_s1024 | pytorch/cuda(hip)/eager | f32 | approximate | 14.260 / 14.302 / 14.366 | 14.084 | 0.235 | 148.1 / cuda-hw | PASS (6.71133e-08); within exact | 71,598 |
| gpt2_mini_train | ocannl/hip/default | f32 | exact | 41.643 / 42.930 / 44.113 | 40.649 | 5.316 | 243.0 / ocannl-seam | PASS (1.14857e-06) | 23,853 |
| gpt2_mini_train | pytorch/cpu/eager | f32 | exact | 74.434 / 76.199 / 77.997 | 74.474 | 0.420 | — | REF (0) | 13,439 |
| gpt2_mini_train | pytorch/cuda(hip)/eager | f32 | exact | 20.859 / 21.040 / 21.291 | 20.349 | 0.678 | 292.2 / cuda-hw | PASS (6.75832e-08) | 48,670 |
| gpt2_mini_train | ocannl/hip/default | f32 | approximate | 47.329 / 48.493 / 49.467 | 45.876 | 5.383 | 194.9 / ocannl-seam | PASS (1.01393e-06); within exact | 21,116 |
| gpt2_mini_train | pytorch/cpu/eager | f32 | approximate | 67.598 / 68.104 / 69.848 | 68.404 | 0.425 | — | PASS (6.69681e-08); within exact | 15,036 |
| gpt2_mini_train | pytorch/cuda(hip)/eager | f32 | approximate | 20.311 / 20.463 / 20.625 | 19.778 | 0.679 | 276.2 / cuda-hw | PASS (6.75832e-08); within exact | 50,040 |
| gpt2_mini_train_s512 | ocannl/hip/default | f32 | exact | 50.612 / 51.946 / 53.108 | 48.878 | 4.176 | 436.4 / ocannl-seam | PASS (6.70514e-07) | 19,713 |
| gpt2_mini_train_s512 | pytorch/cpu/eager | f32 | exact | 149.114 / 149.806 / 164.128 | 151.037 | 0.534 | — | REF (0) | 6,836 |
| gpt2_mini_train_s512 | pytorch/cuda(hip)/eager | f32 | exact | 30.088 / 30.295 / 31.006 | 29.394 | 0.731 | 394.1 / cuda-hw | PASS (1.34243e-07) | 33,800 |
| gpt2_mini_train_s512 | ocannl/hip/default | f32 | approximate | 72.534 / 73.555 / 75.757 | 71.985 | 4.326 | 244.3 / ocannl-seam | PASS (6.04381e-07); within exact | 13,922 |
| gpt2_mini_train_s512 | pytorch/cpu/eager | f32 | approximate | 74.799 / 76.621 / 78.517 | 78.728 | 0.422 | — | PASS (0); within exact | 13,364 |
| gpt2_mini_train_s512 | pytorch/cuda(hip)/eager | f32 | approximate | 27.357 / 27.884 / 28.324 | 26.859 | 0.678 | 330.1 / cuda-hw | PASS (1.34243e-07); within exact | 36,724 |
| gpt2_mini_train_s1024 | ocannl/hip/default | f32 | exact | 75.719 / 76.148 / 77.704 | 75.676 | 3.654 | 694.9 / ocannl-seam | PASS (8.1089e-07) | 13,448 |
| gpt2_mini_train_s1024 | pytorch/cpu/eager | f32 | exact | 403.747 / 448.415 / 476.320 | 447.709 | 0.856 | — | REF (0) | 2,284 |
| gpt2_mini_train_s1024 | pytorch/cuda(hip)/eager | f32 | exact | 67.976 / 68.153 / 68.823 | 68.050 | 0.689 | 475.4 / cuda-hw | PASS (1.35192e-07) | 15,025 |
| gpt2_mini_train_s1024 | ocannl/hip/default | f32 | approximate | 107.629 / 108.148 / 108.656 | 108.275 | 4.017 | 310.8 / ocannl-seam | PASS (8.1089e-07); within exact | 9,469 |
| gpt2_mini_train_s1024 | pytorch/cpu/eager | f32 | approximate | 84.261 / 86.773 / 87.390 | 86.079 | 0.428 | — | PASS (1.34639e-07); within exact | 11,801 |
| gpt2_mini_train_s1024 | pytorch/cuda(hip)/eager | f32 | approximate | 33.931 / 34.372 / 34.901 | 33.250 | 0.719 | 443.4 / cuda-hw | PASS (1.35192e-07); within exact | 29,792 |
| gpt2_mini_b1 | ocannl/hip/default | f32 | exact | 2.841 / 2.962 / 3.164 | 3.020 | 0.969 | 30.5 / ocannl-seam | PASS (2.68774e-07) | 43,218 |
| gpt2_mini_b1 | pytorch/cpu/eager | f32 | exact | 5.217 / 5.271 / 5.304 | 5.314 | 0.009 | — | REF (0) | 24,285 |
| gpt2_mini_b1 | pytorch/cuda(hip)/eager | f32 | exact | 2.544 / 2.558 / 2.593 | 2.471 | 0.230 | 51.6 / cuda-hw | PASS (6.75172e-08) | 50,032 |
| gpt2_mini_b1 | ocannl/hip/default | f32 | approximate | 3.219 / 3.310 / 3.362 | 3.336 | 1.336 | 26.5 / ocannl-seam | PASS (2.69398e-07); within exact | 38,670 |
| gpt2_mini_b1 | pytorch/cpu/eager | f32 | approximate | 2.846 / 2.880 / 2.941 | 2.908 | 0.004 | — | PASS (6.71728e-08); within exact | 44,451 |
| gpt2_mini_b1 | pytorch/cuda(hip)/eager | f32 | approximate | 2.370 / 2.379 / 2.398 | 2.280 | 0.240 | 51.1 / cuda-hw | PASS (6.66016e-08); within exact | 53,799 |
| gpt2_mini_train_b256 | ocannl/hip/default | f32 | exact | 4131.580 / 4200.920 / 4240.900 | 4250.820 | 4.113 | 6624.5 / ocannl-seam | PASS (2.15117e-06) | 7,800 |
| gpt2_mini_train_b256 | pytorch/cpu/eager | f32 | exact | 6187.967 / 6206.550 / 6229.418 | 6196.977 | 6.910 | — | REF (0) | 5,280 |
| gpt2_mini_train_b256 | pytorch/cuda(hip)/eager | f32 | exact | 621.963 / 648.320 / 652.790 | 652.580 | 1.335 | 6574.6 / cuda-hw | PASS (6.72053e-08) | 50,543 |
| gpt2_mini_train_b256 | ocannl/hip/default | f32 | approximate | 4385.360 / 4404.050 / 4434.820 | 4418.990 | 4.119 | 5084.5 / ocannl-seam | PASS (2.15117e-06); within exact | 7,440 |
| gpt2_mini_train_b256 | pytorch/cpu/eager | f32 | approximate | 4873.965 / 4888.302 / 4915.561 | 4907.947 | 5.331 | — | PASS (6.72053e-08); within exact | 6,703 |
| gpt2_mini_train_b256 | pytorch/cuda(hip)/eager | f32 | approximate | 610.018 / 634.044 / 636.025 | 636.927 | 1.283 | 6062.6 / cuda-hw | PASS (6.72053e-08); within exact | 51,681 |

Other legs: [regimes](report-gh720-leg1-hip.md), [sequence](report-gh720-leg2-hip.md),
[batch](report-gh720-leg4-hip.md), [precision](report-gh720-leg7-hip.md).

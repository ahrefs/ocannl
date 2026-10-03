# gh-ocannl-720 leg 4: Batch scaling on HIP

At seq128, batch-1 exact f32 inference takes 2.962 ms, versus 17.695 ms at batch8;
torch eager takes 2.558 ms at batch1. This latency comparison includes whole-framework
submission overhead and does not establish a kernel-efficiency advantage. Batch256 exact
f32 training reaches 7,800 tokens/s versus torch HIP eager 50,543 (6.48x throughput).
Its OCANNL allocation seam is 6624.5 MiB exact and 5084.5 MiB approximate, on discrete
VRAM. The small fixture stores weights/data, not the much larger activations.
Batch256 reuses one batch and batch8 cycles four, so the trajectories have different data.

Measured 2026-10-03 on `tuf-amd-linux`, HIP gfx1102 with **discrete** 8 GB VRAM,
under exclusive request `wave1003-720-tuf-amd-linux-2`, source
`a4a425454cb9bf12c4c24b2c0d5bb6183b1c64ff` (before the PR rebase).
OCANNL is default/untuned, PyTorch `2.13.0+rocm7.1` is eager; one full-protocol sweep.
cc/Metal remain pending. The [leg 1 protocol and reproduction details](report-gh720-leg1-hip.md)
apply to every cell below: exact/approximate whole-profile regimes, no ambient OCANNL
settings, configured graph capture (without a capture-success receipt), per-step synced
percentiles and queued mean, separate compile time, timed-window memory counters, and
exact torch CPU loss oracle. No failed parity timing is used in comparisons.

`ocannl-seam` and `cuda-hw` are distinct requested-byte high-water counters; rank within
one counter and read cross-counter differences as approximate. `—` is unavailable.
An approximate PASS also meets its precision's exact envelope (f32/f16 0.002, bf16 0.004);
the approximate envelope is 0.01. Tokens/s use p50. Failed parity timings are suppressed.

## Fixtures and protocol

All fixtures match `m4-max,tuf` content-v1 origins; they were copied to TUF, never
regenerated there. The declared measuring boxes are m4-max, minix, rog-nv, tuf.
Full digests below identify every cell by workload. P/W/T are parity / warmup / timed
steps; timed steps run twice (synced then queued). Training uses SGD lr 0.01.

| workload | SHA-256 (content-v1) | bytes | seq × batch | tokens/step | P/W/T | missing declared origins |
|---|---|---|---|---|---|---|
| gpt2_mini | `c322a00b72df143612eafafedf7c468e9c05f3e4bc1e7eb6d77ada651ef98de8` | 13871384 | 128 × 8 | 1024 | 8/5/20 | none |
| gpt2_mini_b1 | `5431f8c2956caa72acef8028ec37b98e433ebc9466c3a5727abda140078af922` | 13809936 | 128 × 1 | 128 | 4/5/20 | minix, rog-nv |
| gpt2_mini_train | `2c2458fd4f8fcb8073929bf8b6bedf74c7e5d1dbb77839b632ea216207ade50d` | 13838624 | 128 × 8 | 1024 | 6/3/10 | minix, rog-nv |
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
| gpt2_mini_b1 | ocannl/hip/default | f32 | exact | 2.841 / 2.962 / 3.164 | 3.020 | 0.969 | 30.5 / ocannl-seam | PASS (2.68774e-07) | 43,218 |
| gpt2_mini_b1 | pytorch/cpu/eager | f32 | exact | 5.217 / 5.271 / 5.304 | 5.314 | 0.009 | — | REF (0) | 24,285 |
| gpt2_mini_b1 | pytorch/cuda(hip)/eager | f32 | exact | 2.544 / 2.558 / 2.593 | 2.471 | 0.230 | 51.6 / cuda-hw | PASS (6.75172e-08) | 50,032 |
| gpt2_mini_b1 | ocannl/hip/default | f32 | approximate | 3.219 / 3.310 / 3.362 | 3.336 | 1.336 | 26.5 / ocannl-seam | PASS (2.69398e-07); within exact | 38,670 |
| gpt2_mini_b1 | pytorch/cpu/eager | f32 | approximate | 2.846 / 2.880 / 2.941 | 2.908 | 0.004 | — | PASS (6.71728e-08); within exact | 44,451 |
| gpt2_mini_b1 | pytorch/cuda(hip)/eager | f32 | approximate | 2.370 / 2.379 / 2.398 | 2.280 | 0.240 | 51.1 / cuda-hw | PASS (6.66016e-08); within exact | 53,799 |
| gpt2_mini_train | ocannl/hip/default | f32 | exact | 41.643 / 42.930 / 44.113 | 40.649 | 5.316 | 243.0 / ocannl-seam | PASS (1.14857e-06) | 23,853 |
| gpt2_mini_train | pytorch/cpu/eager | f32 | exact | 74.434 / 76.199 / 77.997 | 74.474 | 0.420 | — | REF (0) | 13,439 |
| gpt2_mini_train | pytorch/cuda(hip)/eager | f32 | exact | 20.859 / 21.040 / 21.291 | 20.349 | 0.678 | 292.2 / cuda-hw | PASS (6.75832e-08) | 48,670 |
| gpt2_mini_train | ocannl/hip/default | f32 | approximate | 47.329 / 48.493 / 49.467 | 45.876 | 5.383 | 194.9 / ocannl-seam | PASS (1.01393e-06); within exact | 21,116 |
| gpt2_mini_train | pytorch/cpu/eager | f32 | approximate | 67.598 / 68.104 / 69.848 | 68.404 | 0.425 | — | PASS (6.69681e-08); within exact | 15,036 |
| gpt2_mini_train | pytorch/cuda(hip)/eager | f32 | approximate | 20.311 / 20.463 / 20.625 | 19.778 | 0.679 | 276.2 / cuda-hw | PASS (6.75832e-08); within exact | 50,040 |
| gpt2_mini_train_b256 | ocannl/hip/default | f32 | exact | 4131.580 / 4200.920 / 4240.900 | 4250.820 | 4.113 | 6624.5 / ocannl-seam | PASS (2.15117e-06) | 7,800 |
| gpt2_mini_train_b256 | pytorch/cpu/eager | f32 | exact | 6187.967 / 6206.550 / 6229.418 | 6196.977 | 6.910 | — | REF (0) | 5,280 |
| gpt2_mini_train_b256 | pytorch/cuda(hip)/eager | f32 | exact | 621.963 / 648.320 / 652.790 | 652.580 | 1.335 | 6574.6 / cuda-hw | PASS (6.72053e-08) | 50,543 |
| gpt2_mini_train_b256 | ocannl/hip/default | f32 | approximate | 4385.360 / 4404.050 / 4434.820 | 4418.990 | 4.119 | 5084.5 / ocannl-seam | PASS (2.15117e-06); within exact | 7,440 |
| gpt2_mini_train_b256 | pytorch/cpu/eager | f32 | approximate | 4873.965 / 4888.302 / 4915.561 | 4907.947 | 5.331 | — | PASS (6.72053e-08); within exact | 6,703 |
| gpt2_mini_train_b256 | pytorch/cuda(hip)/eager | f32 | approximate | 610.018 / 634.044 / 636.025 | 636.927 | 1.283 | 6062.6 / cuda-hw | PASS (6.72053e-08); within exact | 51,681 |

Other legs: [regimes](report-gh720-leg1-hip.md), [sequence](report-gh720-leg2-hip.md),
[batch](report-gh720-leg4-hip.md), [precision](report-gh720-leg7-hip.md).

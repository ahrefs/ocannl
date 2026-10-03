# gh-ocannl-720 leg 2: Sequence scaling on HIP

At constant 1024 tokens/step, OCANNL exact f32 inference falls from 57,870 tokens/s
at seq128 to 40,365 at seq1024; approximate falls from 51,138 to 17,326.
Exact training falls from 23,853 to 13,448 tokens/s; approximate from 21,116 to 9,469.
The batch axis shrinks 8/2/1 as sequence grows, so this does not isolate sequence alone.
Seq1024 reuses the bounded v1.1 fixture instead of the originally proposed seq2048.
The attention and fused-backward gates are profiled together, not independently ablated.

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
| gpt2_mini_s512 | `dfbb26d9d3907f7d9cec3169715ca582982619b27d2455d40647593ac0b61172` | 14231832 | 512 × 2 | 1024 | 4/3/10 | minix, rog-nv |
| gpt2_mini_s1024 | `57073d6f98aa7d310ab9bfbba6636c394afd8800cf1816fcd5109b52f1c363db` | 14756128 | 1024 × 1 | 1024 | 4/3/10 | minix, rog-nv |
| gpt2_mini_train | `2c2458fd4f8fcb8073929bf8b6bedf74c7e5d1dbb77839b632ea216207ade50d` | 13838624 | 128 × 8 | 1024 | 6/3/10 | minix, rog-nv |
| gpt2_mini_train_s512 | `3045392f435637f3b46e48a21d23e48053f75a02209281fb20853eede2622875` | 14231840 | 512 × 2 | 1024 | 6/3/10 | minix, rog-nv |
| gpt2_mini_train_s1024 | `bcaab56e4359c5cdba46b3953d955f8dfc2736029f485f8d0ff8709ced844f6e` | 14756136 | 1024 × 1 | 1024 | 6/3/10 | minix, rog-nv |

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

Other legs: [regimes](report-gh720-leg1-hip.md), [sequence](report-gh720-leg2-hip.md),
[batch](report-gh720-leg4-hip.md), [precision](report-gh720-leg7-hip.md).

# gh-ocannl-720 leg 7: Reduced precision on HIP

All bf16 cells pass. Exact f16 fails parity at seq1024 inference and training, and
base training diverges at step 0. Approximate f16 passes on all endpoints, but that does
not establish the cause of the exact failures. No invalid row is used for a speedup claim.
Batch256 passes every format/regime: exact bf16 uses 3320.9 MiB and takes 4504.490 ms;
exact f16 uses 3382.0 MiB and takes 3879.630 ms, versus f32 6624.5 MiB / 4200.920 ms.
Reduced storage therefore helps memory here without a uniform throughput improvement.
These are OCANNL storage-format legs against f32 torch oracles, not reduced-precision
torch comparisons; untuned storage changes do not demonstrate tensorized schedules.

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
| gpt2_mini_b1 | `5431f8c2956caa72acef8028ec37b98e433ebc9466c3a5727abda140078af922` | 13809936 | 128 × 1 | 128 | 4/5/20 | minix, rog-nv |
| gpt2_mini_train_b256 | `61cc0aff416cbfc330d025b1fdff8b161e0ff86546343c1e578c179ff28c97ec` | 14068008 | 128 × 256 | 32768 | 6/3/10 | minix, rog-nv |

## Cells

| workload | framework/backend/variant | precision | regime | p10 / p50 / p90 ms | queued ms | compile s | peak MiB / counter | parity (max rel) | tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| gpt2_mini | ocannl/hip/default | f32 | exact | 17.187 / 17.695 / 18.394 | 16.634 | 0.956 | 123.7 / ocannl-seam | PASS (8.72258e-07) | 57,870 |
| gpt2_mini | ocannl/hip/default | f32 | approximate | 19.079 / 20.024 / 20.820 | 18.859 | 1.093 | 91.4 / ocannl-seam | PASS (8.72258e-07); within exact | 51,138 |
| gpt2_mini | ocannl/hip/default | bf16 | exact | 16.968 / 17.480 / 17.852 | 16.456 | 1.643 | 61.9 / ocannl-seam | PASS (4.51203e-05) | 58,581 |
| gpt2_mini | ocannl/hip/default | bf16 | approximate | 20.163 / 20.786 / 21.606 | 19.540 | 2.187 | 45.8 / ocannl-seam | PASS (0.00100373); within exact | 49,264 |
| gpt2_mini | ocannl/hip/default | f16 | exact | 12.689 / 12.946 / 13.259 | 12.352 | 1.161 | 61.9 / ocannl-seam | PASS (5.68842e-05) | 79,096 |
| gpt2_mini | ocannl/hip/default | f16 | approximate | 13.572 / 13.774 / 13.950 | 13.097 | 1.273 | 45.8 / ocannl-seam | PASS (0.000102962); within exact | 74,343 |
| gpt2_mini_s512 | ocannl/hip/default | f32 | exact | 22.137 / 22.489 / 22.866 | 21.648 | 0.999 | 221.0 / ocannl-seam | PASS (1.20968e-06) | 45,533 |
| gpt2_mini_s512 | ocannl/hip/default | f32 | approximate | 37.577 / 38.174 / 39.379 | 36.019 | 1.158 | 92.7 / ocannl-seam | PASS (1.14205e-06); within exact | 26,825 |
| gpt2_mini_s512 | ocannl/hip/default | bf16 | exact | 19.767 / 20.016 / 20.093 | 19.159 | 1.654 | 111.0 / ocannl-seam | PASS (5.88973e-05) | 51,160 |
| gpt2_mini_s512 | ocannl/hip/default | bf16 | approximate | 32.579 / 33.097 / 34.121 | 31.224 | 2.189 | 46.9 / ocannl-seam | PASS (0.00102424); within exact | 30,939 |
| gpt2_mini_s512 | ocannl/hip/default | f16 | exact | 15.546 / 15.629 / 15.825 | 15.322 | 1.201 | 111.0 / ocannl-seam | PASS (4.20684e-05) | 65,520 |
| gpt2_mini_s512 | ocannl/hip/default | f16 | approximate | 25.326 / 25.786 / 26.183 | 24.503 | 1.309 | 46.9 / ocannl-seam | PASS (0.000104939); within exact | 39,712 |
| gpt2_mini_s1024 | ocannl/hip/default | f32 | exact | 24.587 / 25.369 / 25.797 | 24.418 | 1.008 | 352.5 / ocannl-seam | PASS (4.04093e-07) | 40,365 |
| gpt2_mini_s1024 | ocannl/hip/default | f32 | approximate | 57.786 / 59.104 / 60.703 | 57.935 | 1.316 | 96.2 / ocannl-seam | PASS (3.36367e-07); within exact | 17,326 |
| gpt2_mini_s1024 | ocannl/hip/default | bf16 | exact | 20.451 / 20.829 / 21.127 | 20.120 | 1.645 | 178.3 / ocannl-seam | PASS (4.15284e-05) | 49,161 |
| gpt2_mini_s1024 | ocannl/hip/default | bf16 | approximate | 41.572 / 42.491 / 43.562 | 39.709 | 2.164 | 50.1 / ocannl-seam | PASS (0.00104806); within exact | 24,099 |
| gpt2_mini_s1024 | ocannl/hip/default | f16 | exact | — | — | 1.192 | 178.3 / ocannl-seam | FAIL (0.00449603) | — |
| gpt2_mini_s1024 | ocannl/hip/default | f16 | approximate | 32.931 / 33.848 / 34.335 | 31.838 | 1.258 | 50.1 / ocannl-seam | PASS (3.60356e-05); within exact | 30,253 |
| gpt2_mini_train | ocannl/hip/default | f32 | exact | 41.643 / 42.930 / 44.113 | 40.649 | 5.316 | 243.0 / ocannl-seam | PASS (1.14857e-06) | 23,853 |
| gpt2_mini_train | ocannl/hip/default | f32 | approximate | 47.329 / 48.493 / 49.467 | 45.876 | 5.383 | 194.9 / ocannl-seam | PASS (1.01393e-06); within exact | 21,116 |
| gpt2_mini_train | ocannl/hip/default | bf16 | exact | 54.377 / 56.447 / 56.688 | 52.539 | 7.187 | 129.5 / ocannl-seam | PASS (0.000366842) | 18,141 |
| gpt2_mini_train | ocannl/hip/default | bf16 | approximate | 70.066 / 71.838 / 74.592 | 69.371 | 14.999 | 105.6 / ocannl-seam | PASS (0.00115027); within exact | 14,254 |
| gpt2_mini_train | ocannl/hip/default | f16 | exact | — | — | 9.054 | 144.2 / ocannl-seam | DIVERGED | — |
| gpt2_mini_train | ocannl/hip/default | f16 | approximate | 76.170 / 76.854 / 77.791 | 76.510 | 9.023 | 120.2 / ocannl-seam | PASS (0.00504811); within exact | 13,324 |
| gpt2_mini_train_s512 | ocannl/hip/default | f32 | exact | 50.612 / 51.946 / 53.108 | 48.878 | 4.176 | 436.4 / ocannl-seam | PASS (6.70514e-07) | 19,713 |
| gpt2_mini_train_s512 | ocannl/hip/default | f32 | approximate | 72.534 / 73.555 / 75.757 | 71.985 | 4.326 | 244.3 / ocannl-seam | PASS (6.04381e-07); within exact | 13,922 |
| gpt2_mini_train_s512 | ocannl/hip/default | bf16 | exact | 61.750 / 63.861 / 66.172 | 60.227 | 5.858 | 226.7 / ocannl-seam | PASS (0.000571664) | 16,035 |
| gpt2_mini_train_s512 | ocannl/hip/default | bf16 | approximate | 91.967 / 94.827 / 96.192 | 92.248 | 13.721 | 130.8 / ocannl-seam | PASS (0.00105659); within exact | 10,799 |
| gpt2_mini_train_s512 | ocannl/hip/default | f16 | exact | 78.529 / 79.427 / 80.509 | 78.717 | 7.670 | 241.7 / ocannl-seam | PASS (0.000149841) | 12,892 |
| gpt2_mini_train_s512 | ocannl/hip/default | f16 | approximate | 92.443 / 92.630 / 93.479 | 92.856 | 7.827 | 145.8 / ocannl-seam | PASS (0.000683335); within exact | 11,055 |
| gpt2_mini_train_s1024 | ocannl/hip/default | f32 | exact | 75.719 / 76.148 / 77.704 | 75.676 | 3.654 | 694.9 / ocannl-seam | PASS (8.1089e-07) | 13,448 |
| gpt2_mini_train_s1024 | ocannl/hip/default | f32 | approximate | 107.629 / 108.148 / 108.656 | 108.275 | 4.017 | 310.8 / ocannl-seam | PASS (8.1089e-07); within exact | 9,469 |
| gpt2_mini_train_s1024 | ocannl/hip/default | bf16 | exact | 74.000 / 75.907 / 78.044 | 73.347 | 5.521 | 357.4 / ocannl-seam | PASS (0.000827579) | 13,490 |
| gpt2_mini_train_s1024 | ocannl/hip/default | bf16 | approximate | 113.705 / 115.331 / 117.346 | 114.888 | 13.283 | 165.5 / ocannl-seam | PASS (0.00144723); within exact | 8,879 |
| gpt2_mini_train_s1024 | ocannl/hip/default | f16 | exact | — | — | 7.028 | 373.0 / ocannl-seam | FAIL (0.00409354) | — |
| gpt2_mini_train_s1024 | ocannl/hip/default | f16 | approximate | 116.668 / 117.593 / 118.189 | 117.984 | 7.151 | 181.0 / ocannl-seam | PASS (0.000274238); within exact | 8,708 |
| gpt2_mini_b1 | ocannl/hip/default | f32 | exact | 2.841 / 2.962 / 3.164 | 3.020 | 0.969 | 30.5 / ocannl-seam | PASS (2.68774e-07) | 43,218 |
| gpt2_mini_b1 | ocannl/hip/default | f32 | approximate | 3.219 / 3.310 / 3.362 | 3.336 | 1.336 | 26.5 / ocannl-seam | PASS (2.69398e-07); within exact | 38,670 |
| gpt2_mini_b1 | ocannl/hip/default | bf16 | exact | 2.216 / 2.248 / 2.290 | 2.279 | 1.647 | 15.3 / ocannl-seam | PASS (0.000104449) | 56,927 |
| gpt2_mini_b1 | ocannl/hip/default | bf16 | approximate | 3.304 / 3.343 / 3.390 | 3.432 | 2.151 | 13.3 / ocannl-seam | PASS (0.00140263); within exact | 38,290 |
| gpt2_mini_b1 | ocannl/hip/default | f16 | exact | 1.890 / 1.927 / 1.947 | 1.943 | 1.185 | 15.3 / ocannl-seam | PASS (0.000158733) | 66,409 |
| gpt2_mini_b1 | ocannl/hip/default | f16 | approximate | 2.129 / 2.240 / 2.281 | 2.287 | 1.247 | 13.3 / ocannl-seam | PASS (7.46602e-05); within exact | 57,155 |
| gpt2_mini_train_b256 | ocannl/hip/default | f32 | exact | 4131.580 / 4200.920 / 4240.900 | 4250.820 | 4.113 | 6624.5 / ocannl-seam | PASS (2.15117e-06) | 7,800 |
| gpt2_mini_train_b256 | ocannl/hip/default | f32 | approximate | 4385.360 / 4404.050 / 4434.820 | 4418.990 | 4.119 | 5084.5 / ocannl-seam | PASS (2.15117e-06); within exact | 7,440 |
| gpt2_mini_train_b256 | ocannl/hip/default | bf16 | exact | 4421.440 / 4504.490 / 4554.480 | 4485.670 | 5.922 | 3320.9 / ocannl-seam | PASS (0.000267344) | 7,275 |
| gpt2_mini_train_b256 | ocannl/hip/default | bf16 | approximate | 5000.640 / 5030.660 / 5059.640 | 5047.430 | 13.613 | 2552.9 / ocannl-seam | PASS (0.00110084); within exact | 6,514 |
| gpt2_mini_train_b256 | ocannl/hip/default | f16 | exact | 3845.500 / 3879.630 / 3898.380 | 3889.830 | 7.743 | 3382.0 / ocannl-seam | PASS (0.00142229) | 8,446 |
| gpt2_mini_train_b256 | ocannl/hip/default | f16 | approximate | 3948.130 / 3963.110 / 4020.260 | 3964.710 | 7.768 | 2614.0 / ocannl-seam | PASS (0.000618649); within exact | 8,268 |

## Exact-f16 parity findings

The tolerance is 0.002. Seq1024 inference has max relative loss difference
0.0044960326636339955; seq1024 training has 0.004093540358374387. Both are FAIL.
Base training is DIVERGED at step 0: the result emitted `null` for the first loss
(the emitter maps a non-finite float to null), then finite losses. Its complete
parity trajectory was:

```json
[null, 7.10226059, 7.12176037, 7.08063602, 7.09992838, 7.10012341]
```

These are correctness findings on gfx1102 with discrete memory, not timeouts.
Reproduction uses the same driver matrix above with any of the three named workloads.
The passing f32/bf16 and approximate controls narrow a follow-up investigation; they
do not prove whether reduced arithmetic, masking/reductions, or loss scaling is at fault.
An open-issue search for f16/parity/s1024 found no issue specifically covering these
failures on 2026-10-03; propose one focused follow-up after this PR, rather than changing
the compiler in this reporting-only leg.

Other legs: [regimes](report-gh720-leg1-hip.md), [sequence](report-gh720-leg2-hip.md),
[batch](report-gh720-leg4-hip.md), [precision](report-gh720-leg7-hip.md).

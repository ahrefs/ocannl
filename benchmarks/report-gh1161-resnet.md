# Benchmark results — cifar_resnet baseline (gh-ocannl-1161)

> **2026-10-30 (gh-ocannl-1401): re-measured, with tuned cells.** [report-gh1161-resnet-20261030.md](report-gh1161-resnet-20261030.md) repeats this table at `1fc57811` on the same three boxes and fixtures, and adds the tuned cells this baseline could not quote: every `cifar_resnet{,_b1}/{cc,cuda,metal,hip}/tuned/exact` cell, the fission-off CUDA cell, and the minix redo, each a pass-2 replay from a fresh absolute cache. The tables below remain the Oct 16 baseline.

> Checked-in baseline record for the reduced ResNet-style conv leg (`cifar_resnet` at batch 64,
> `cifar_resnet_b1` at batch 1; the workload and the cell names are in [README.md](README.md#cifar_resnet-cell-names)).
> These are the **untuned** (`default`) cells, exact and approximate, that the v1.1.2 conv issues
> (gh-ocannl-505, 1147, 1148, 697, 503) measure their before/after against. They were taken at
> staging `e075a2ca` (rog-nv, m4-max) and `578a381e` (minix) — the wave base `0922ddbe6` plus this
> leg's benchmark-only commits, so before gh-ocannl-1324 reshapes untuned contractions; the runner
> and model code are identical at both. Each box ran under a measurement hold (requests
> `wave-20261016-ocannl-1161-{rog-nv-linux,mac-studio,minix-amd-linux}-1`), with
> `BENCH_KERNEL_TABLE=1`, and every cell passed the parity gate.
>
> All three boxes measured one workload: the fixtures were generated once on m4-max and copied
> byte for byte (`cifar_resnet` sha256 `065d246d…f9c4`, `cifar_resnet_b1` sha256 `2bda4d0d…c68a`,
> content-v1, recorded for m4-max, minix and rog-nv in `fixtures/DIGESTS.txt`).

**What is not here, and why.**

- **Tuned cells.** A tuned cifar_resnet row cannot pass the provenance gate yet: every padded conv
  operand lowers into nests that re-bind a loop symbol, so the schedule cache never keys the step
  and pass 2 searches again (tuned: not cacheable, see ahrefs/ocannl#1384; `bench_conv_diag` with
  `BENCH_CANON=1` shows it). The minix sweep did run tuned cells; they are search-pass timings and
  are left out below.
- **Site cells on HIP and Metal.** The kernel-table instrument recompiles each shipped kernel on
  its own; two of those standalone kernels declare a parameter twice (`relu_mlp_layer_grad`,
  `max_pool2d_grad`). NVRTC's refusal is a classified decline, so CUDA times 99 of 100 / 124 of 126
  kernels; hiprtc's and Metal's are fatal and end the table, so `gh1161_sites.py` prints those
  tables as INCOMPLETE and attributes nothing. On cc the step is one kernel, so the site cell there
  is the step cell.
- `pytorch/cuda(hip)` on `cifar_resnet` in the approximate regime segfaulted on minix (exit -11,
  ROCm torch with `cudnn.benchmark`); it is a runner failure, not a row.

## Step cells (untuned, step p50 ms)

| workload | regime | box | OCANNL cc | OCANNL GPU | torch CPU | torch GPU | tinygrad CPU | tinygrad GPU |
|---|---|---|---|---|---|---|---|---|
| cifar_resnet | exact | rog-nv (CUDA, RTX 5070 Ti) | 520.68 | 184.01 | 11.17 (REF) | 1.74 | — | — |
| cifar_resnet | exact | m4-max (Metal) | 406.67 | 391.61 | 99.49 (REF) | 2.77 | — | — |
| cifar_resnet | exact | minix (HIP gfx1151, unified) | 524.88 | 145.96 | 9.29 (REF) | 6.54 | 301.05 | 6.05 |
| cifar_resnet | approximate | rog-nv (CUDA, RTX 5070 Ti) | 518.51 | 184.81 | 11.68 | 1.71 | 301.17 | 5.18 |
| cifar_resnet | approximate | m4-max (Metal) | 404.16 | 389.98 | 99.75 | 2.81 | 51.19 | 6.24 |
| cifar_resnet | approximate | minix (HIP gfx1151, unified) | 522.64 | 145.91 | 9.62 | — | 305.40 | 6.08 |
| cifar_resnet_b1 | exact | rog-nv (CUDA, RTX 5070 Ti) | 7.95 | 63.94 | 1.49 (REF) | 1.29 | — | — |
| cifar_resnet_b1 | exact | m4-max (Metal) | 6.38 | 547.99 | 1.92 (REF) | 1.88 | — | — |
| cifar_resnet_b1 | exact | minix (HIP gfx1151, unified) | 8.17 | 181.59 | 1.56 (REF) | 2.13 | 8.07 | 0.79 |
| cifar_resnet_b1 | approximate | rog-nv (CUDA, RTX 5070 Ti) | 7.93 | 64.09 | 1.46 | 1.30 | 7.22 | 0.70 |
| cifar_resnet_b1 | approximate | m4-max (Metal) | 6.34 | 546.29 | 1.89 | 1.87 | 6.20 | 1.01 |
| cifar_resnet_b1 | approximate | minix (HIP gfx1151, unified) | 8.15 | 181.99 | 1.64 | 2.14 | 8.15 | 0.79 |

tinygrad runs one regime (its own defaults). A sweep dispatching both regimes files it under
`approximate`, as on rog-nv and m4-max; minix's exact-only sweep filed it under `exact`. The torch
GPU `—` in the approximate minix `cifar_resnet` row is the segfault above.

## Site cells (rog-nv, CUDA, untuned)

From `gh1161_sites.py` over the sweep's cell logs: per family, the summed min-of-20 times
(launch and sync included) of the kernels the shipped step writes that family's names from. A
fused kernel counts toward every family it serves, so the family columns do not sum to the step;
`unattributed` is kernels writing only unnamed nodes. The `kernels timed` shortfall is the
duplicate-parameter declines described above.

| cell | step p50 ms | kernels timed | sum of minima ms | pad3x3 | resadd | sc1x1s2 | unattributed |
|---|---|---|---|---|---|---|---|
| cifar_resnet/cc/default/approximate | 518.514 | 1/1 | 510.247 | 510.247 | 510.247 | 510.247 | 0.000 |
| cifar_resnet/cc/default/exact | 520.679 | 1/1 | 513.801 | 513.801 | 513.801 | 513.801 | 0.000 |
| cifar_resnet/cuda/default/approximate | 184.810 | 99/100 | 162.127 | 143.531 | 11.747 | 11.383 | 7.427 |
| cifar_resnet/cuda/default/exact | 184.015 | 99/100 | 161.675 | 143.141 | 11.696 | 11.333 | 7.414 |
| cifar_resnet_b1/cc/default/approximate | 7.929 | 1/1 | 7.759 | 7.759 | 7.759 | 7.759 | 0.000 |
| cifar_resnet_b1/cc/default/exact | 7.946 | 1/1 | 7.781 | 7.781 | 7.781 | 7.781 | 0.000 |
| cifar_resnet_b1/cuda/default/approximate | 64.086 | 124/126 | 64.505 | 48.139 | 0.961 | 0.410 | 15.584 |
| cifar_resnet_b1/cuda/default/exact | 63.940 | 124/126 | 64.354 | 48.015 | 0.959 | 0.410 | 15.558 |

Per site (same rule, one column per site):

| cell | step p50 ms | kernels timed | sum of minima ms | pad3x3 | resadd | sc1x1s2 | unattributed | stem | s1c1 | s1c2 | s1res | s2c1 | s2c2 | s2res | s2sc | s3c1 | s3c2 | s3sc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar_resnet/cc/default/approximate | 518.514 | 1/1 | 510.247 | 510.247 | 510.247 | 510.247 | 0.000 | 510.247 | 510.247 | 510.247 | 510.247 | 510.247 | 510.247 | 510.247 | 510.247 | 510.247 | 510.247 | 510.247 |
| cifar_resnet/cc/default/exact | 520.679 | 1/1 | 513.801 | 513.801 | 513.801 | 513.801 | 0.000 | 513.801 | 513.801 | 513.801 | 513.801 | 513.801 | 513.801 | 513.801 | 513.801 | 513.801 | 513.801 | 513.801 |
| cifar_resnet/cuda/default/approximate | 184.810 | 99/100 | 162.127 | 143.531 | 11.747 | 11.383 | 7.427 | 82.520 | 47.199 | 30.152 | 10.317 | 11.841 | 12.590 | 1.429 | 10.089 | 3.153 | 2.662 | 1.304 |
| cifar_resnet/cuda/default/exact | 184.015 | 99/100 | 161.675 | 143.141 | 11.696 | 11.333 | 7.414 | 82.514 | 47.014 | 29.957 | 10.267 | 11.837 | 12.585 | 1.429 | 10.039 | 3.156 | 2.664 | 1.304 |
| cifar_resnet_b1/cc/default/approximate | 7.929 | 1/1 | 7.759 | 7.759 | 7.759 | 7.759 | 0.000 | 7.759 | 7.759 | 7.759 | 7.759 | 7.759 | 7.759 | 7.759 | 7.759 | 7.759 | 7.759 | 7.759 |
| cifar_resnet_b1/cc/default/exact | 7.946 | 1/1 | 7.781 | 7.781 | 7.781 | 7.781 | 0.000 | 7.781 | 7.781 | 7.781 | 7.781 | 7.781 | 7.781 | 7.781 | 7.781 | 7.781 | 7.781 | 7.781 |
| cifar_resnet_b1/cuda/default/approximate | 64.086 | 124/126 | 64.505 | 48.139 | 0.961 | 0.410 | 15.584 | 16.825 | 15.486 | 0.081 | 0.677 | 14.961 | 0.193 | 0.284 | 0.329 | 0.790 | 0.065 | 0.081 |
| cifar_resnet_b1/cuda/default/exact | 63.940 | 124/126 | 64.354 | 48.015 | 0.959 | 0.410 | 15.558 | 16.780 | 15.450 | 0.081 | 0.675 | 14.919 | 0.192 | 0.284 | 0.329 | 0.789 | 0.066 | 0.081 |

## Generated tables

Each box's `orchestrate.py` report, reduced to its platform line, fixture stamps and rows (the legends are the generated reports' own; see [README.md](README.md)).

### rog-nv

platform: Linux-7.0.0-38-generic-x86_64-with-glibc2.43 x86_64 | ocannl commit: e075a2ca0

#### cifar_resnet

measured on `cifar_resnet.safetensors`, sha256 `065d246d181aeadd2417c23d0e0fa9fb604b77d3b95a089e412d350ba0f9f9c4`, m4-max,minix,rog-nv's bytes

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | kernel %peak | mma | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | f32 | exact | 1.739 | 1.701 | 1.790 | 1.502 | 9.85 | 112.6 cuda-hw | — | — | PASS (1.8e-06) |
| pytorch | cpu | eager | f32 | exact | 11.170 | 10.397 | 12.923 | 14.207 | 0.13 | — | — | — | REF |
| ocannl | cuda | default | f32 | exact | 184.015 | 183.745 | 184.237 | 184.633 | 1.89 | 102.4 ocannl-seam | approx · k82/100 61.765 ms bias_stem.grad (+1 untimed) | — | PASS (3.1e-07) |
| ocannl | cc | default | f32 | exact | 520.679 | 514.202 | 521.266 | 520.527 | 3.15 | 102.4 ocannl-seam | no ceiling · k0/1 513.801 ms batch_x kernel_stem kernel_stem.grad +96 | — | PASS (2.1e-07) |
| pytorch | cuda | eager | f32 | approximate | 1.715 | 1.660 | 1.770 | 1.408 | 2.33 | 112.7 cuda-hw | — | — | PASS (5.0e-06, within exact envelope) |
| tinygrad | CUDA | jit | f32 | approximate | 5.176 | 5.166 | 5.183 | 5.105 | 3.59 | 38.9 tg-mem-used | — | — | PASS (2.1e-07, within exact envelope) |
| pytorch | cpu | eager | f32 | approximate | 11.678 | 11.329 | 12.163 | 11.110 | 0.19 | — | — | — | PASS (0.0e+00, within exact envelope) |
| ocannl | cuda | default | f32 | approximate | 184.810 | 184.571 | 185.047 | 184.911 | 1.95 | 102.4 ocannl-seam | approx · k82/100 61.768 ms bias_stem.grad (+1 untimed) | — | PASS (3.1e-07, within exact envelope) |
| tinygrad | CPU | jit | f32 | approximate | 301.173 | 300.763 | 301.854 | 300.472 | 3.03 | 38.9 tg-mem-used | — | — | PASS (3.1e-07, within exact envelope) |
| ocannl | cc | default | f32 | approximate | 518.514 | 518.016 | 520.377 | 518.432 | 3.19 | 102.4 ocannl-seam | no ceiling · k0/1 510.247 ms batch_x kernel_stem kernel_stem.grad +97 | — | PASS (1.0e-05, within exact envelope) |

#### cifar_resnet_b1

measured on `cifar_resnet_b1.safetensors`, sha256 `2bda4d0d75b8e7b58932de0f1ccbe17f7984fa837b08e6c27a9094adc74bc68a`, m4-max,minix,rog-nv's bytes

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | kernel %peak | mma | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | f32 | exact | 1.289 | 1.138 | 1.331 | 1.254 | 0.91 | 67.7 cuda-hw | — | — | PASS (1.1e-07) |
| pytorch | cpu | eager | f32 | exact | 1.494 | 1.415 | 4.417 | 1.526 | 0.15 | — | — | — | REF |
| ocannl | cc | default | f32 | exact | 7.946 | 7.912 | 8.073 | 7.955 | 2.56 | 3.4 ocannl-seam | no ceiling · k0/1 7.781 ms batch_x kernel_stem kernel_stem.grad +92 | — | PASS (2.1e-07) |
| ocannl | cuda | default | f32 | exact | 63.940 | 63.728 | 64.117 | 64.015 | 1.17 | 3.5 ocannl-seam | 0.0% f32 compute · k99/126 15.075 ms relu_s1c1out.grad (+2 untimed) | — | PASS (1.1e-07) |
| tinygrad | CUDA | jit | f32 | approximate | 0.703 | 0.701 | 0.706 | 0.636 | 2.77 | 1.9 tg-mem-used | — | — | PASS (1.1e-07, within exact envelope) |
| pytorch | cuda | eager | f32 | approximate | 1.300 | 1.153 | 1.354 | 1.247 | 0.94 | 66.4 cuda-hw | — | — | PASS (1.8e-05, within exact envelope) |
| pytorch | cpu | eager | f32 | approximate | 1.464 | 1.400 | 1.556 | 11.728 | 0.19 | — | — | — | PASS (0.0e+00, within exact envelope) |
| tinygrad | CPU | jit | f32 | approximate | 7.215 | 7.195 | 7.246 | 7.211 | 2.12 | 1.9 tg-mem-used | — | — | PASS (1.1e-07, within exact envelope) |
| ocannl | cc | default | f32 | approximate | 7.929 | 7.891 | 7.969 | 7.946 | 2.56 | 3.4 ocannl-seam | no ceiling · k0/1 7.759 ms batch_x kernel_stem kernel_stem.grad +92 | — | PASS (9.1e-04, within exact envelope) |
| ocannl | cuda | default | f32 | approximate | 64.086 | 63.940 | 64.180 | 64.026 | 1.23 | 3.5 ocannl-seam | 0.0% f32 compute · k99/126 15.107 ms relu_s1c1out.grad (+2 untimed) | — | PASS (1.1e-07, within exact envelope) |
### m4-max

platform: macOS-26.6.2-arm64-arm-64bit arm64 | ocannl commit: e075a2ca0

#### cifar_resnet

measured on `cifar_resnet.safetensors`, sha256 `065d246d181aeadd2417c23d0e0fa9fb604b77d3b95a089e412d350ba0f9f9c4`, m4-max,minix,rog-nv's bytes

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | kernel %peak | mma | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | mps | eager | f32 | exact | 2.768 | 2.406 | 2.903 | 2.126 | 1.96 | 1,048.9 mps-driver | — | — | PASS (8.3e-07) |
| pytorch | cpu | eager | f32 | exact | 99.494 | 98.539 | 100.586 | 100.126 | 0.13 | — | — | — | REF |
| ocannl | metal | default | f32 | exact | 391.614 | 388.044 | 397.809 | 386.353 | 0.96 | 102.4 ocannl-seam | no kernel | — | PASS (1.0e-06) |
| ocannl | cc | default | f32 | exact | 406.670 | 404.676 | 419.301 | 407.794 | 2.95 | 102.4 ocannl-seam | no ceiling · k0/1 404.251 ms batch_x kernel_stem kernel_stem.grad +96 | — | PASS (1.1e-06) |
| pytorch | mps | eager | f32 | approximate | 2.811 | 2.451 | 3.072 | 2.130 | 0.17 | 1,049.0 mps-driver | — | — | PASS (8.3e-07, within exact envelope) |
| tinygrad | METAL | jit | f32 | approximate | 6.243 | 6.087 | 6.490 | 5.536 | 5.56 | 38.9 tg-mem-used | — | — | PASS (9.3e-07, within exact envelope) |
| tinygrad | CPU | jit | f32 | approximate | 51.187 | 50.763 | 54.449 | 51.443 | 6.44 | 38.9 tg-mem-used | — | — | PASS (1.0e-06, within exact envelope) |
| pytorch | cpu | eager | f32 | approximate | 99.748 | 98.927 | 101.440 | 100.070 | 0.11 | — | — | — | PASS (0.0e+00, within exact envelope) |
| ocannl | metal | default | f32 | approximate | 389.980 | 386.058 | 395.827 | 387.606 | 0.63 | 102.4 ocannl-seam | no kernel | — | PASS (1.0e-06, within exact envelope) |
| ocannl | cc | default | f32 | approximate | 404.160 | 402.772 | 409.393 | 405.992 | 2.88 | 102.4 ocannl-seam | no ceiling · k0/1 403.760 ms batch_x kernel_stem kernel_stem.grad +97 | — | PASS (1.6e-05, within exact envelope) |

#### cifar_resnet_b1

measured on `cifar_resnet_b1.safetensors`, sha256 `2bda4d0d75b8e7b58932de0f1ccbe17f7984fa837b08e6c27a9094adc74bc68a`, m4-max,minix,rog-nv's bytes

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | kernel %peak | mma | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | mps | eager | f32 | exact | 1.880 | 1.826 | 1.974 | 1.030 | 1.75 | 10.9 mps-driver | — | — | PASS (1.1e-07) |
| pytorch | cpu | eager | f32 | exact | 1.921 | 1.785 | 2.010 | 1.917 | 0.01 | — | — | — | REF |
| ocannl | cc | default | f32 | exact | 6.379 | 6.324 | 6.457 | 6.475 | 1.88 | 3.4 ocannl-seam | no ceiling · k0/1 5.964 ms batch_x kernel_stem kernel_stem.grad +92 | — | PASS (2.1e-07) |
| ocannl | metal | default | f32 | exact | 547.991 | 547.193 | 548.364 | 541.592 | 0.87 | 3.5 ocannl-seam | no kernel | — | PASS (2.1e-07) |
| tinygrad | METAL | jit | f32 | approximate | 1.012 | 0.958 | 1.534 | 0.761 | 4.12 | 1.9 tg-mem-used | — | — | PASS (1.1e-07, within exact envelope) |
| pytorch | mps | eager | f32 | approximate | 1.868 | 1.799 | 1.972 | 1.056 | 0.17 | 11.0 mps-driver | — | — | PASS (1.1e-07, within exact envelope) |
| pytorch | cpu | eager | f32 | approximate | 1.894 | 1.787 | 2.040 | 1.902 | 0.01 | — | — | — | PASS (0.0e+00, within exact envelope) |
| tinygrad | CPU | jit | f32 | approximate | 6.198 | 6.059 | 6.314 | 6.191 | 4.90 | 1.9 tg-mem-used | — | — | PASS (1.1e-07, within exact envelope) |
| ocannl | cc | default | f32 | approximate | 6.335 | 6.290 | 6.396 | 6.440 | 1.87 | 3.4 ocannl-seam | no ceiling · k0/1 5.961 ms batch_x kernel_stem kernel_stem.grad +92 | — | PASS (1.1e-07, within exact envelope) |
| ocannl | metal | default | f32 | approximate | 546.294 | 545.589 | 546.741 | 539.925 | 0.54 | 3.5 ocannl-seam | no kernel | — | PASS (2.1e-07, within exact envelope) |
### minix, exact (tuned search-pass rows omitted)

platform: Linux-7.0.0-38-generic-x86_64-with-glibc2.43 x86_64 | ocannl commit: 578a381e4

#### cifar_resnet

measured on `cifar_resnet.safetensors`, sha256 `065d246d181aeadd2417c23d0e0fa9fb604b77d3b95a089e412d350ba0f9f9c4`, m4-max,minix,rog-nv's bytes

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | kernel %peak | pass | mma | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tinygrad | AMD | jit | f32 | 6.048 | 5.974 | 6.121 | 5.922 | 3.46 | 38.9 tg-mem-used | — | — | — | PASS (1.8e-06) |
| pytorch | cuda(hip) | eager | f32 | 6.539 | 6.438 | 6.635 | 6.489 | 5.29 | 103.1 cuda-hw | — | — | — | PASS (1.7e-06) |
| pytorch | cpu | eager | f32 | 9.292 | 8.542 | 10.080 | 9.301 | 0.50 | — | — | — | — | REF |
| ocannl | hip | default | f32 | 145.955 | 144.562 | 146.912 | 146.092 | 2.00 | 102.4 ocannl-seam | no kernel | — | — | PASS (1.8e-06) |
| tinygrad | CPU | jit | f32 | 301.054 | 300.477 | 301.942 | 300.405 | 3.69 | 38.9 tg-mem-used | — | — | — | PASS (1.7e-06) |
| ocannl | cc | default | f32 | 524.879 | 524.605 | 525.284 | 524.409 | 3.00 | 102.4 ocannl-seam | no ceiling · k0/1 524.613 ms batch_x kernel_stem kernel_stem.grad +96 | — | — | PASS (1.8e-06) |

#### cifar_resnet_b1

measured on `cifar_resnet_b1.safetensors`, sha256 `2bda4d0d75b8e7b58932de0f1ccbe17f7984fa837b08e6c27a9094adc74bc68a`, m4-max,minix,rog-nv's bytes

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | kernel %peak | pass | mma | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tinygrad | AMD | jit | f32 | 0.786 | 0.783 | 0.789 | 0.646 | 2.71 | 1.9 tg-mem-used | — | — | — | PASS (2.1e-07) |
| pytorch | cpu | eager | f32 | 1.559 | 1.541 | 1.605 | 1.561 | 0.43 | — | — | — | — | REF |
| pytorch | cuda(hip) | eager | f32 | 2.134 | 2.117 | 2.149 | 2.134 | 0.33 | 66.4 cuda-hw | — | — | — | PASS (1.1e-07) |
| tinygrad | CPU | jit | f32 | 8.070 | 8.057 | 8.085 | 8.059 | 2.56 | 1.9 tg-mem-used | — | — | — | PASS (1.1e-07) |
| ocannl | cc | default | f32 | 8.169 | 8.142 | 8.195 | 8.198 | 2.47 | 3.4 ocannl-seam | no ceiling · k0/1 7.984 ms batch_x kernel_stem kernel_stem.grad +92 | — | — | PASS (1.1e-07) |
| ocannl | hip | default | f32 | 181.589 | 176.548 | 185.670 | 182.789 | 1.59 | 3.5 ocannl-seam | no kernel | — | — | PASS (1.1e-07) |
### minix, approximate

platform: Linux-7.0.0-38-generic-x86_64-with-glibc2.43 x86_64 | ocannl commit: 578a381e4

#### cifar_resnet

measured on `cifar_resnet.safetensors`, sha256 `065d246d181aeadd2417c23d0e0fa9fb604b77d3b95a089e412d350ba0f9f9c4`, m4-max,minix,rog-nv's bytes

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | kernel %peak | mma | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cpu | eager | f32 | exact | 8.912 | 8.382 | 9.164 | 7.694 | 0.39 | — | — | — | REF |
| tinygrad | AMD | jit | f32 | approximate | 6.081 | 6.009 | 6.147 | 5.934 | 3.09 | 38.9 tg-mem-used | — | — | PASS (1.8e-06, within exact envelope) |
| pytorch | cpu | eager | f32 | approximate | 9.620 | 8.743 | 10.597 | 9.782 | 0.24 | — | — | — | PASS (0.0e+00, within exact envelope) |
| ocannl | hip | default | f32 | approximate | 145.907 | 144.525 | 146.897 | 145.758 | 2.06 | 102.4 ocannl-seam | no kernel | — | PASS (1.7e-06, within exact envelope) |
| tinygrad | CPU | jit | f32 | approximate | 305.401 | 305.085 | 305.685 | 304.967 | 3.43 | 38.9 tg-mem-used | — | — | PASS (1.7e-06, within exact envelope) |
| ocannl | cc | default | f32 | approximate | 522.637 | 522.437 | 522.864 | 522.453 | 3.14 | 102.4 ocannl-seam | no ceiling · k0/1 522.047 ms batch_x kernel_stem kernel_stem.grad +97 | — | PASS (1.1e-05, within exact envelope) |

#### cifar_resnet_b1

measured on `cifar_resnet_b1.safetensors`, sha256 `2bda4d0d75b8e7b58932de0f1ccbe17f7984fa837b08e6c27a9094adc74bc68a`, m4-max,minix,rog-nv's bytes

| framework | backend | variant | precision | regime | step p50 ms | p10 | p90 | queued ms | compile s | peak MiB | kernel %peak | mma | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cpu | eager | f32 | exact | 1.628 | 1.616 | 1.675 | 1.634 | 0.37 | — | — | — | REF |
| tinygrad | AMD | jit | f32 | approximate | 0.787 | 0.783 | 0.822 | 0.647 | 2.45 | 1.9 tg-mem-used | — | — | PASS (2.1e-07, within exact envelope) |
| pytorch | cpu | eager | f32 | approximate | 1.642 | 1.628 | 1.655 | 1.652 | 0.23 | — | — | — | PASS (0.0e+00, within exact envelope) |
| pytorch | cuda(hip) | eager | f32 | approximate | 2.141 | 1.790 | 2.344 | 2.162 | 1.81 | 66.4 cuda-hw | — | — | PASS (1.1e-07, within exact envelope) |
| ocannl | cc | default | f32 | approximate | 8.152 | 8.125 | 8.182 | 8.182 | 2.45 | 3.4 ocannl-seam | no ceiling · k0/1 7.977 ms batch_x kernel_stem kernel_stem.grad +92 | — | PASS (9.1e-04, within exact envelope) |
| tinygrad | CPU | jit | f32 | approximate | 8.153 | 8.137 | 8.169 | 8.148 | 2.40 | 1.9 tg-mem-used | — | — | PASS (1.1e-07, within exact envelope) |
| ocannl | hip | default | f32 | approximate | 181.986 | 178.445 | 186.003 | 182.052 | 1.63 | 3.5 ocannl-seam | no kernel | — | PASS (1.1e-07, within exact envelope) |

#### Runner failures

| cell | why | saved losses vs reference (diagnostic, never acceptance) |
|---|---|---|
| cifar_resnet pytorch/cuda/eager [approximate] | exit -11 | no checkpoint |
## Reproducing a cell

```bash
cd <checkout> && ln -sfn ~/.venvs/ocannl-bench benchmarks/.venv   # fixtures copied into benchmarks/fixtures/
d=$PWD/benchmarks/results/<dir>; mkdir -p $d
BENCH_KERNEL_TABLE=1 BENCH_RESULTS_DIR=$d BENCH_CELL_LOG_DIR=$d/cells \
  OCANNL_AUTOTUNE_CACHE_DIR=$d/autotune_cache \
  benchmarks/.venv/bin/python benchmarks/orchestrate.py --box <box> --gpu <cuda|hip|metal> \
  --workloads cifar_resnet cifar_resnet_b1 --profile exact approximate
benchmarks/gh1161_sites.py --per-site $d/cells
```

`OCANNL_AUTOTUNE_CACHE_DIR` must be absolute: the cells run with `benchmarks/` as their working
directory, and a relative path there names a directory that does not exist, so a tuned cell's
search would store nothing even once ahrefs/ocannl#1384 is fixed.

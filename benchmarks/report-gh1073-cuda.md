# CUDA staged fp8 accumulator residency (gh-ocannl-1073)

On rog-nv-linux's RTX 5070 Ti Laptop GPU (sm_120), holding the m16n8k32
f32 accumulator across 128 outer reduction blocks cuts kernel time by 28.95%
in a standalone reproduction of the backend's plain fragment and staging loops.
The generated-kernel confirmation cuts kernel time by 37.22% on the staged
schedule below. That supports extending the register scope: it also lets every
inline-PTX shape share one destination-boundary mechanism.

## Standalone mechanism probe

Source: [mma_register_scope_probe.cu](runners/ocannl/mma_register_scope_probe.cu).
The per-block arm loads/stores D at every k_o block and keeps its leading
barrier. The resident arm loads/stores once, unrolls its fragment loops, and
moves the leading barrier outside the reduction. The measured difference
includes all three changes; it does not isolate memory traffic alone.

- Shape: 512 x 512 x 4096; workgroup tile: 16 x 32 x 32; 128 k_o blocks.
- Build: CUDA 13.4, nvcc -O3, compute_89 PTX forward-JIT to sm_120,
  matching the CUDA backend's fp8 architecture floor.
- Both arms verified against every exact host product cell before timing.
- Exclusive fleet measurement, nine paired batches, 100 launches per arm,
  alternating A/B order; CUDA events exclude setup and warmup.
- Run: `/home/lukstafi/.ocannl-test-runs/20261003T145215Z-1391202`.

| Statistic | Per-block D | Resident D | Resident / per-block |
| --- | ---: | ---: | ---: |
| Median kernel time | 0.423660 ms | 0.301085 ms | 0.710513 (median paired ratio) |
| Paired ratio range | | | 0.710016–0.711465 |

This is a mechanism measurement, not an application speedup. The shape is
chosen to expose repeated D traffic across many k_o blocks.

## Generated-kernel confirmation protocol

[bench_mma_register_scope_emit.ml](runners/ocannl/bench_mma_register_scope_emit.ml)
emits and validates the actual plain staged fp8 kernel for 8192 x 32 x 4096,
with a 16 x 32 x 32 workgroup tile, 512 workgroups and 128 k_o blocks.
The dimensions fit the existing staged schedule's lane-wide column extent;
the standalone probe uses the same workgroup geometry but tiles a wider N.
The exporter checks every output cell against an exact host product, then
reads the fresh source through `Test_utils.Generated`.

Export the baseline from commit `c6603b2b25fc68a0e9ffb6ecdfe5b899f9f8648f`
(before the register-scope extension), using the exporter added by this change.
Export the resident arm after the extension. The opt-in
`@benchmarks/runners/ocannl/mma-register-scope-emit` alias exports the current
kernel to `generated-mma-register-scope.cu` in its build directory; direct
`dune exec` also accepts an output path as its first argument.

The same CUDA driver can include both emitted sources, renaming only the
kernel symbol. Build it before timing:

```bash
nvcc -O3 -gencode arch=compute_89,code=compute_89 \
  '-DMMA_GENERATED_BASELINE="/absolute/baseline.cu"' \
  '-DMMA_GENERATED_RESIDENT="/absolute/resident.cu"' \
  benchmarks/runners/ocannl/mma_register_scope_probe.cu -o /absolute/generated-probe
/absolute/generated-probe --dry-run
```

Run the precompiled binary without `--dry-run` only in an exclusive timing
window. It performs the same nine alternating-order paired CUDA-event batches;
there is no build or host-product oracle in that invocation. OCANNL's emitted
source is compiled with nvcc for this confirmation, and dispatch bypasses the
OCANNL runtime, so the number remains a kernel measurement.

## Generated-kernel result

The exclusive confirmation ran both precompiled emitted sources on the same
sm_120 device, with nine paired batches of 100 launches per arm and alternating
order. The dry run checked every cell against the exact host product and
checked the resident replay before the measurement window.

| Statistic | Per-block D | Resident D | Resident / per-block |
| --- | ---: | ---: | ---: |
| Median kernel time | 0.338762 ms | 0.212565 ms | 0.627809 (median paired ratio) |
| Paired ratio range | | | 0.626295–0.628337 |

The median paired reduction is **37.22%**. This confirms the gain in the actual
emitted plain staged kernel; the capability-derived residency tests also check
the swizzled twin, whose timing is not measured here. The measurement includes
register residency, fragment-loop unrolling and moving the leading barrier,
and remains a kernel result rather than an application speedup.

- Measurement revision: `603396fa8486906feb0cdec526d26eec85f9dc86`.
- Run: `/home/lukstafi/.ocannl-test-runs/20261003T153353Z-1420901`.
- Baseline emitted-source SHA-256:
  `f745cc346712c9b95776899a6ffd2bf2c2242811c17eff23fe5390b527734129`.
- Resident emitted-source SHA-256:
  `2842e85da1712b1fc6f4e7769273ee31c9b3fc3b90c496fa0af9b99aaa2c303d`.

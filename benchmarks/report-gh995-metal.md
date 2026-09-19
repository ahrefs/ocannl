# Small leading axes in the default Metal schedule

On Apple M4 Max, the untuned GPT inference step for batch 2 × sequence 512 fell from
438.132 ms to 62.352 ms (7.03×), and batch 8 × sequence 128 from 125.305 ms to
62.549 ms (2.00×). This closes the large launch-geometry cliff reported in
[ahrefs/ocannl#995](https://github.com/ahrefs/ocannl/issues/995). All reported parity
losses match the corresponding baseline exactly; requested peak allocations are unchanged.

There is one exclusive baseline run and one run of each of two changed revisions per cell,
not a balanced repeated experiment. Each runner performs
its fixture's parity and warmup windows, followed by synchronized timings and a queued window.
The sequence-1024 row measured slightly faster too, but its small difference is not a demonstrated
stable speedup. The three rows still differ: this change removes the severe small-batch cliff,
not every shape-dependent performance difference.

## Measurement

Baseline: `21f1d495c633e41c032721234b232558b72a7b1e`.
Changed (final policy): `59d996724f5d52f720f1412803e6c6ab2c767591`.
Initial candidate: `b869fa926426c281c137e4155a107d2ec155aabe`.
The measured final policy was subsequently rebased over an unrelated research-document change;
its published implementation is `ea3cbf99a`, with identical scheduler and benchmark sources.
Host: mac-studio, Apple M4 Max, macOS; 2026-09-19 UTC.
Backend explicitly `metal`, f32, default placements and default schedule, no autotuning,
`online_softmax=false`. No unrelated numerics settings changed.

| Fixture | Batch × sequence | Baseline p10 / p50 / p90 ms | Changed p10 / p50 / p90 ms | Baseline / changed queued ms |
| --- | --- | --- | --- | --- |
| gpt2_mini | 8 × 128 | 125.012 / 125.305 / 125.597 | 62.105 / 62.549 / 62.843 | 120.562 / 58.098 |
| gpt2_mini_s512 | 2 × 512 | 437.788 / 438.132 / 438.320 | 62.202 / 62.352 / 62.506 | 433.920 / 58.040 |
| gpt2_mini_s1024 | 1 × 1024 | 53.596 / 54.231 / 54.448 | 51.104 / 51.439 / 51.733 | 49.411 / 46.890 |

The initial candidate measured p50 62.480, 62.393, and 51.576 ms respectively. After review
strengthened the post-alignment fallback, all three final `cross_entropy_loss_fwd__seg.metal`
files were byte-identical to that candidate's files. The final rows above were nevertheless
remeasured on the final policy revision.

The first row uses 20 timed steps per window; the other rows use 10. All three have 1,024 tokens
per step. Peak requested allocations are respectively 129660932, 231667716, and 369596420 bytes
on both revisions. Eight reported parity losses match for the first row, four for each other row.
This is a before/after correctness comparison, not a new cross-framework parity claim.

The coordinator reserved each entire batch exclusively: no other wave builds, tests, or
measurements were active. Ordinary desktop activity remained, including WindowServer and animated
wallpaper. The changed-batch host snapshot showed approximately 15.6% CPU for WindowServer and
4.8% for the wallpaper extension. No runner contention refusal occurred.

## What the emitted kernels show

In `cross_entropy_loss_fwd__seg.metal`, embedding compute segment 1 previously bound the batch
axis to `gid.x` and the sequence axis to `lid.x`. For batch 2 / sequence 512, two threadgroups
each executed a two-iteration serial sequence chunk and a 256-iteration serial channel loop.
For batch 1, lowering removes the singleton batch loop from compute, exposing sequence and
channel as the first two parallel axes: 1,024 threadgroups × 256 threads.

| Embedding compute | Baseline grid × block | Changed grid × block | Changed serial leading iterations |
| --- | --- | --- | --- |
| batch 8 / sequence 128 | 8 × 128 | 128 × 256 | 8 |
| batch 2 / sequence 512 | 2 × 256 | 512 × 256 | 2 |
| batch 1 / sequence 1024 | 1024 × 256 | 1024 × 256 | singleton removed |

The actual changed sequence-512 source binds sequence to `gid.x` and channel to `lid.x` inside
the serial batch loop. Whole-node zero expansion previously retained the leading singleton even
for sequence 1024; its new geometry follows the same selection policy as compute.

The policy looks past a leading parallel extent below `gpu_schedule_min_parallel` (64 by default)
when a later pair provides no fewer grid groups and a larger grid-times-clamped-block product.
Skipped axes remain serial. Crucially, selection runs **before** the existing ownership and
cross-nest alignment analysis: the proof checks exactly the selected coordinates. A declined
selection falls back to analyzing the original outermost pair; so does an aligned result that
loses group count, active lanes, or the launch threshold relative to the original. The GPU
default fingerprint includes a policy version, so historical default timings are discarded
without invalidating an otherwise valid cached winner. Every scheduled nest still has
one Grid and one Workgroup dimension; no new reduction reassociation or hardware dimension is
introduced. CPU default scheduling is unchanged.

This shared GPU policy also changes eligible CUDA/HIP schedules. Tonight's hardware validation is
Metal only; CUDA/HIP correctness and performance runs remain explicit residuals. No performance
claim is made for those backends or for tuned workloads.

## Fixtures and reproduction

All fixtures use the fresh `m4-max` content-v1 entries introduced by gh-ocannl-1007 and pass
`fixture_digest.py --check` against `benchmarks/fixtures/DIGESTS.txt`. Provenance:
Python 3.12.14, NumPy 2.5.1, safetensors 0.8.0; copied unchanged from the wave's
`1007-generation-c/benchmarks/fixtures` directory.

| Fixture | Canonical SHA-256 |
| --- | --- |
| gpt2_mini | c322a00b72df143612eafafedf7c468e9c05f3e4bc1e7eb6d77ada651ef98de8 |
| gpt2_mini_s512 | dfbb26d9d3907f7d9cec3169715ca582982619b27d2455d40647593ac0b61172 |
| gpt2_mini_s1024 | 57073d6f98aa7d310ab9bfbba6636c394afd8800cf1816fcd5109b52f1c363db |

After building `benchmarks/runners/ocannl/bench_gpt.exe`, run from the checkout root (no root
`ocannl_config` was present), once for each fixture:

```bash
env -u BENCH_TUNE BENCH_FIXTURE="$PWD/benchmarks/fixtures/gpt2_mini.safetensors" \
  tools/test-run.sh run --cap 600 exec --no-build benchmarks/runners/ocannl/bench_gpt.exe -- \
  --ocannl_backend=metal --ocannl_online_softmax=false \
  --ocannl_output_debug_files_in_build_directory=true \
  --ocannl_build_files_prefix=995-gpt2_mini
```

The original invocations used absolute scratch paths as prefix values; this configuration sanitizes
them into subdirectory names beneath `build_files`, rather than interpreting them as paths. The
resulting source directories were copied intact into wave scratch after each batch.

Local evidence root: `/Users/lukstafi/.ocannl-test-runs/`.

| Fixture | Baseline run | Initial candidate run | Final policy run |
| --- | --- | --- | --- |
| gpt2_mini | 20260919T222334Z-47490 | 20260919T223328Z-73399 | 20260919T230011Z-40081 |
| gpt2_mini_s512 | 20260919T222358Z-49291 | 20260919T223334Z-75192 | 20260919T230017Z-41943 |
| gpt2_mini_s1024 | 20260919T222428Z-51176 | 20260919T223339Z-76963 | 20260919T230021Z-43798 |

All nine runners exited 0. Preserved source directories are `995-baseline-<fixture>`,
`995-changed-<fixture>`, and `995-final-<fixture>` under
`/Users/lukstafi/.local/state/issue-wave/wave-20260919-ocannl-limited/`.

The executed regression `gpu_small_leading_axis` identifies every coordinate across batch-2,
batch-8, and batch-plus-head producer/consumer nests, checks conservative fallback for an
incompatible traversal, restores nontrivial original geometry after suffix trimming or rejection,
and clears a nonzero initialized tensor through expanded zeros. Legacy fingerprint invalidation
is exercised through an actual cache hit in `slow-autotune_fission_sketch`. The
Metal `schedule_ops`, `fission_schedule`, and `fission_equivalence` regressions also pass, alongside
repository scans, `@check`, and `@fmt`.

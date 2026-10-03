# Register-tile C traffic does not distinguish column geometries

Investigation of [gh-ocannl-1099](https://github.com/ahrefs/ocannl/issues/1099),
2026-10-03. Decision: retain `Register_tile.cost` without a C-traffic term and
leave the tie-break order unchanged in this investigation. The proposed
per-pass C-load/store term cannot distinguish `rn` at a fixed vector width.
The tail-bearing tie rule has a separate unresolved measurement question,
tracked in [gh-ocannl-1180](https://github.com/ahrefs/ocannl/issues/1180).
No k argument or new register-tile seeds are added.

## The traffic term cancels

`arrayjit/lib/register_tile.ml` ranks an `m x n` site by an exact integer price,
scaled by the default's row count `rm`:

```text
V = ceil(n / lanes)
P = ceil(n / (rn * lanes))
score = V * (rm + 1) + rm * P
```

The first term counts the FMA issues and B loads; the second counts A splats.
At one lane width every candidate shares `V` and `rm`, so equal pass counts
reach the documented tie-break: fewer column-tail vectors, then larger `rn`.

The issue proposed adding `2 * rm * rn / k` vector moves per pass for the C-tile
loads and stores. This is the correct count for a **full** pass. It must also be
summed over the passes and use the smaller grid for the column tail:

```text
q = floor(n / (rn * lanes))
t = ceil((n mod (rn * lanes)) / lanes)
C moves per rm-row band = 2 * rm * (q * rn + t)
                       = 2 * rm * ceil(n / lanes)
```

`Register_tile.coverage` produces the tail's vector columns, and
`C_syntax.try_register_tile` renders them through `pass ~cols:tail_cols`. The
accumulator grid and its loads/stores have `List.length cols` columns; the tail
is not padded to `rn`. Each partial last vector counts as one vector issue in
this model. Partial memory accesses may take more instructions, but at a fixed
lane width their valid width is also independent of `rn`.

The row band uses the same emitter with fewer rows. Summing over all row bands
gives `2 * m * V` C-vector moves for the site. Dividing by `m * k` gives `2 * V / k`,
which is independent of `rn`. In the existing integer scale, adding actual
C traffic means ranking by `k * score + 2 * rm * V`: it cannot change the column
geometry's order at any k. This argument applies at each fixed lane width; it
does not claim that the term is identical across different lane widths.

At the AVX2 n=28 site (`rm=4`, `lanes=8`), 4x2 emits two two-column passes;
4x3 emits one three-column pass and a one-column tail. Both perform 32 C-vector
moves per four-row band. Both have score 28. At n=512 the C moves also agree
(512 per band), while the existing scores already prefer 4x3: 408 versus 448.

Charging `2 * rm * rn / k` just once against the whole-site price would be a
tile-footprint heuristic, not the rendered traffic count. It would break an
`rn` tie toward the smaller tile at every finite k. Charging a full `rn` for
the tail would count moves the emitter never performs. Neither is adopted.

## Exclusive AVX2 measurement

The original evidence is in
[gh-ocannl-947's AVX2 report](https://github.com/ahrefs/ocannl/issues/947#issuecomment-5853633227).
It compared tiny sites with k=n against n=512 with k=256, changing the column
extent and row block along with k. This investigation adds a comparison at
**fixed n=512 and bm=64**, changing only the packed k block between 32 and 256
within each storage-precision/geometry arm.

The run used rog-nv-linux (Core Ultra 9 275HX), backend `cc`, at clean revision
`f110746b7f4fe743b15932d25ad6650f54861804`. `/proc/cpuinfo` contained `avx2`
and no `avx512f`; this is an AVX2 result, not AVX-512. The benchmark was built
before the exclusive measurement reservation. Five interleaved paired rounds
requested `--tile=4,2,8` and `--tile=4,3,8` from the same build, reversing the
order within each pair on even rounds.

All 60 benchmark invocations passed. Every packed variant rendered
`Mma_register_tiled x1` (120 lines total), and values agreed with their
references. f16 used narrow storage with f32 compute and
`--ocannl_fp16_arithmetic=false`.

The six cases were:

| Storage | n | bm | bk (micro-k) | Repeats per invocation |
| --- | --- | --- | --- | --- |
| f32 | 28 | 4 | 28 | 10000 |
| f32 | 32 | 16 | 32 | 10000 |
| f32 | 512 | 64 | 32 | 20 |
| f32 | 512 | 64 | 256 | 20 |
| f16 | 512 | 64 | 32 | 20 |
| f16 | 512 | 64 | 256 | 20 |

Tiny sites use more repetitions than the benchmark's default of 20 to
accumulate timed work. This repeats the earlier comparison's geometry and
interleaved protocol, rather than claiming identical sampling conditions.
For each case the prebuilt executable invocation was:

```text
OCANNL_BACKEND=cc _build/default/bin/narrow_gebp_bench.exe \
  <storage> <n> <repeats> <bm> <bk> --tile=4,<rn>,8
```

Configuration sourcing was traced to stderr with
`OCANNL_LOG_CONFIG_SOURCING=true`. The f16 invocations appended the arithmetic
flag above. The driver required both packed variants to report register-tiled
emission and every benchmark process to exit successfully. A reproduction
requires its own exclusive measurement reservation and a prebuilt executable;
the recorded run used the capped project runner with `exec --no-build`.

Build run: `/home/lukstafi/.ocannl-test-runs/20261003T143436Z-1376965`.
Measurement run: `/home/lukstafi/.ocannl-test-runs/20261003T143520Z-1378184`
(exit 0, clean source). The measurement request was
`wave1003-1099-rog-nv-linux-2`.

Throughputs below are GFLOP/s ranges over five rounds. The percentage is the
median of the **paired** throughput ratios `4x2 / 4x3 - 1`; a positive value
favors 4x2. A tie means equal throughput at the benchmark's printed precision.

| Storage | n | k | Variant | 4x2 range | 4x3 range | Median 4x2 vs 4x3 | 4x2 wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | 28 | 28 | serial | 48.46–50.92 | 45.34–47.34 | +6.70% | 5/5 |
| f32 | 32 | 32 | serial | 66.46–91.79 | 65.50–68.07 | +2.37% | 3/5, one tie |
| f32 | 512 | 32 | serial | 144.54–145.05 | 149.57–150.58 | -3.52% | 0/5 |
| f32 | 512 | 256 | serial | 119.16–132.91 | 146.39–147.86 | -17.95% | 0/5 |
| f16 | 512 | 32 | serial | 129.79–130.77 | 133.63–134.71 | -3.23% | 0/5 |
| f16 | 512 | 256 | serial | 116.57–128.09 | 141.57–142.21 | -17.47% | 0/5 |
| f32 | 28 | 28 | parallel | 3.51–3.63 | 3.59–3.68 | -1.37% | 0/5 |
| f32 | 32 | 32 | parallel | 5.21–5.51 | 5.37–5.57 | -2.33% | 1/5 |
| f32 | 512 | 32 | parallel | 255.79–343.60 | 236.65–335.85 | +3.93% | 5/5 |
| f32 | 512 | 256 | parallel | 104.57–116.97 | 125.75–128.67 | -15.12% | 0/5 |
| f16 | 512 | 32 | parallel | 235.85–309.97 | 270.14–315.15 | -8.63% | 1/5 |
| f16 | 512 | 256 | parallel | 93.84–103.17 | 108.49–110.57 | -12.98% | 0/5 |

## Decision and limits

The n=28 serial comparison reproduces the smaller tile's advantage. The n=32
comparison is mixed and contains a +37.91% outlier favoring 4x2. At n=512, 4x3
wins every serial pair at both k=32 and k=256, in both storage precisions. Small
k alone therefore does not make the smaller tile preferable. The parallel
results disagree with serial on some sites and are kept separate.

The current issue-slot model cannot distinguish the n=28 pair before its
tie-break. Preferring a tail-free tile has a structural rationale: it emits
one column-tile body instead of a full body and a tail body. When both tied
candidates carry tails, however, both emit two bodies, and equal cost at equal
lanes means the same total A splats. The remaining preferences for fewer tail
vectors and then larger `rn` have no additional rationale in the issue-slot
model on such a tie. This investigation does not remove the measured serial
disadvantage of the current n=28 pick, nor establish its cause.

A zero-parameter alternative is to prefer a tail-free tile, then smaller
`rn`, at equal cost and lane width. It would choose 4x2 at n=28 and preserve
the already tail-free 4x2 choice at n=32 on AVX2. This is a tie-rule question,
separate from adding a setup penalty or a mis-summed C-traffic term.
[gh-ocannl-1180](https://github.com/ahrefs/ocannl/issues/1180) owns validating
that rule on NEON and AVX2 before changing it. The earlier
[gh-ocannl-947 NEON report](https://github.com/ahrefs/ocannl/issues/947#issuecomment-5852345656)
already measured a tail-bearing tie at f32 n=56, bm=8: rn6 with a two-vector
tail versus rn5 with a four-vector tail was neutral within noise. That one
comparison does not establish the proposed rule across NEON sites; the
follow-up needs targeted measurement coverage rather than assuming none
exists.

Keep the model and seeding unchanged here and leave the tie key to gh-ocannl-1180's
measurement window. No fresh NEON or AVX-512 measurements are needed to reject
the C-traffic term: no new heuristic is applied to either target, and the
traffic cancellation follows from the shared emitter. This makes no new
performance claim about those targets. A later setup or tie-rule investigation
should vary n, bm and k independently.

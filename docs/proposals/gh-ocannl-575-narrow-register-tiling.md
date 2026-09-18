# gh-ocannl-575: Tile_mma register tiling for narrow (16-bit) operands

Re-scoped out of gh-ocannl-530 (originally gh-ocannl-516 task 3, gh-ocannl-517's packing remark,
and gh-ocannl-516 task 4). The landed state of the two parents: narrow storage computes in f32 on
the CPU backends through the `C_syntax_config.compute_prec` seam (`Ir.Numerics.narrow_compute_f32`,
default on), with vectorized convert-on-load/store making the `Vectorized` renderings reachable —
but `try_register_tile`, the CPU `Tile_mma` rendering, still gated on f32/f64 *storage* and
declined every 16-bit GEMM to the scalar fallback.

## Design: finish the seam, don't special-case it

`try_register_tile` was the one explicit-SIMD rendering that never went through the gh-517
storage/compute split — `try_vectorize` and `try_vectorize_reduce` both key their lane geometry
off `comp_prec` and bridge at the memory boundary. The change makes the register tiling do exactly
the same, uniformly:

1. **Compute-precision-keyed gates.** The precision gate becomes
   `B.vector_prec_ok (comp_prec d_storage)` — f32/f64 everywhere, fp16 where arithmetic is native
   — and the uniformity gate compares the three operands' *compute* precisions (their storages may
   differ: a narrow `d` over f32-packed panels is uniform-f32 compute). Lane count, `vec_ext_typ`,
   and the accumulator registers all follow the compute precision.
2. **Memory-boundary bridges.** The B-row vector loads and the C-tile moves go through the
   existing `vec_bridge` (identity memcpy when storage = compute, so the f32 emission is unchanged
   modulo one whitespace split); the A splats and the scalar peel convert through the same
   `convert_precision` spellings the scalar fallback renders with. fp8 storage rides the bridge's
   per-lane fallback arm for free.
3. **Accumulator residency.** The C-tile accumulates at compute precision across the statement's
   whole k extent and narrows once at the store — the same once-per-cell narrowing as
   `try_vectorize_reduce`'s epilogue and the CUDA bf16 `mma.sync` arm (gh-ocannl-545): strictly
   better rounding than the fallback's per-k-step narrowing, never bitwise on inexact values, so
   parity tests pick narrow-exact inputs. Where storage = compute (pure-fp16 included) the
   rendering stays BITWISE equal to the fallback, per-element fused chains in serial k order.
4. **Pure-fp16 tiling** (f16 accumulators, doubled lanes) is not a separate mode: it is what the
   seam resolves to when `Ir.Numerics.fp16_arithmetic` is on AND the target's arithmetic is
   genuinely native (`hardware_limits.native_fp16_arithmetic`; on merely-promoted targets the
   policy is deliberately ignored). The scalar peel's FMA gained the `OCANNL_HALF_FMA` arm so peel
   and vector body round identically (the gh-516 single-vs-double-rounding trap).

### The packing seam: `Stage { tile_prec }`

gh-517's "free lunch": fold the widening into the GEBP packing copy so the conversion happens once
per element at pack time instead of once per read inside the micro-kernel. `Schedule.Stage` gained
`tile_prec : Ops.prec option` — mint the staged tile at this storage precision instead of the
source's. Only exact widenings are accepted (equal, or narrow-float source to f32/f64): a widening
is value-preserving and its round-trip is the identity, so the transform is unconditionally
numerics-preserving under either policy setting, and the packing copy's cross-precision `Set`
renders the conversion through the ordinary `Get` boundary. The hoisted (link-time, constant-pool)
packing converts on the host in `pack_constant_tile` via the exact `get_as_float`/`set_from_float`
round-trip; the same-precision path keeps its raw-copy fast path byte-for-byte.

### Seeding moves with emission (the gh-545 lesson)

The compute-precision resolution now has a single source of truth, `Numerics.cpu_compute_prec
~native_fp16_arithmetic`, shared verbatim by `Cc_backend.compute_prec` (emission) and the autotune
sketch pre-filters (seeding) — so a candidate is never timed under a tensorized label its
rendering must decline. Concretely, in `sketch_families.ml`:

- the matmul and conv CPU branches replace `uniform_f32_64` with compute-precision uniformity +
  vector-capability (fp16 requires `limits.native_fp16_arithmetic`), and key lane counts and the
  packed-tile footprint caps off the compute element size (an f32 panel over fp16 storage is twice
  the storage bytes; a pure-fp16 panel is half the f32 one);
- `sketch_params` gained `sk_pack_prec : Ops.prec option` — the site's compute precision, recorded
  at seed time (schedule construction has no `hardware_limits`) and threaded into the packing
  `Stage`s' `tile_prec`, normalized to `None` per operand already storing at it. Candidate labels
  carry it as ` pack<prec>`.

Pinned by `test/operations/sketch_family_tree.ml`'s half-precision scenarios: default policy seeds
11 candidates with `packprec:single`; `narrow_compute_f32=false` refutes the tensorized branch;
native-fp16 limits + the `fp16_arithmetic` policy seed pure-f16 (no pack precision, doubled
lanes); the policy on a merely-promoted target stays f32-compute.

### Tile width follows the extent, not the register cap

> **Superseded in part by gh-ocannl-620.** The scalar peel this section prices no longer exists:
> the columns the width does not cover render as a narrower register tile (whole vectors plus one
> partial vector), so `Register_tile.default` ranks by operand reuse alone and the fitted peel
> weight is gone. The measurements below stand as the record of why the peel had to go; the model
> they fit is retired.

The C-tile is `rm` rows of `rn` vectors, and `rn` is chosen against the *actual* column extent rather
than pinned at the register-pressure cap. The columns `bw = rn * lanes` does not cover are peeled to
the scalar fallback, and a peeled column costs roughly a whole vector slot — so a cap that leaves a
fat remainder loses far more than the extra A-reuse it buys. This is invisible at f32 lane counts and
brutal at doubled ones: at n = 512 a pure-fp16 `bw = 48` peels 32 columns, 6.25% of the work at
scalar speed, and that alone turned the NEON measurement below upside down (37 vs 133 GFLOP/s).
Power-of-two extents — what deep learning actually runs — are never multiples of 48.

The ranking model: per unit of `m*k`, a tile pass issues one vector FMA per lane-column plus the B
row loads (1/`rm` per FMA) and the A splats (1/`rn`), while each peeled column costs a flat 10
lane-slots. The peel weight deliberately does *not* scale with the lane count — the peel loop is the
same scalar code at either width — and the fits agree: ~8 from the 8-lane sweep below, ~10 from the
4-lane one, ~20 from the n = 2048 pair. It only has to *rank* candidates, not predict times; it
reproduces the measured order at n = 512 within a few percent across `rn = 2..6`, and where several
widths divide the extent evenly it lands on the largest affordable one. Measured on the M4 Max (10
repeats, GFLOP/s, `packmma`):

| rn (bw at 4/8 lanes) | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| pure-f16, n=512 | 92.0 | 78.7 | **132.6** | 36.9 | 36.9 |
| pure-f16, n=1024 | 136.0 | 96.7 | **203.4** | 89.0 | 111.2 |
| f16→f32, n=512 | 55.4 | 47.5 | **75.3** | 48.3 | 55.7 |
| f16→f32, n=1024 | 66.4 | 65.9 | **100.9** | 80.6 | 62.0 |

The cap itself was never the problem: isolated at extents both widths divide (n = 576, 768),
`rn = 4` and `rn = 6` land within ±10% of each other. The peel is the whole story. The fixed cap had
been the geometry since gh-ocannl-469, so this lifts the f32 register tiling too (n = 512: 55.9 →
74.3 GFLOP/s standalone) — 16-bit operands are what made it visible, not what made it true.

Erring *low* on the peel weight is what costs choices, and it costs them quietly: at n = 2048 a
`lanes`-scaled weight scored the peeling `rn = 6` just under the peel-free `rn = 4`, which measures
1.15x faster (98.4 vs 85.5 GFLOP/s at f32) — the same cliff, one notch smaller. Extents whose only
peel-free widths are narrow are where the model earns its keep, so it must not be tuned to the
extents that were easy to measure.

### Cost model

Nothing to change structurally: per-node footprint widths already come off each node's own
`storage_prec` (so narrow nodes and f32-minted packed tiles are both exact), and FLOP counts are
precision-blind. The one implicit width assumption — `hardware_limits.peak_flops` being a
single-precision ceiling that a native-fp16 kernel doubles — is documented on the field; it cannot
reorder a site's candidates because they all share one policy-resolved compute precision.

## Measurement, x86 (ROG: Core Ultra 9 275HX, WSL2, gcc 15.2, cc backend at -O3, n = 512)

`bin/narrow_gebp_bench.exe` (naive serial vs packed GEBP serial vs grid-outermost per-chunk
variant; exact inputs; readbacks outside the timed region — re-measured after the Codex round-1
fix moved the spot-check readback out of the timed interval). Re-measured at the extent-adapted tile
width and after the gh-ocannl-614 fix to the accumulator update, which superseded the numbers this
section first carried (the note below says by how much):

| storage → compute | naive | packmma | packmma_par |
|---|---|---|---|
| f32 → f32 | 21.8 GFLOP/s | 116.6 | 97.7 |
| bf16 → f32 (widened panels) | 0.39 | **105.6 (271x)** | 55.4 |
| f16 → f32 (widened panels) | 0.65 | **100.5 (156x)** | 63.9 |
| f16 → f16 forced on promoted HW | 0.65 | 2.55 | 2.49 |

- The headline: **f32-GEBP-over-narrow-storage is a ~156-271x win over the scalar narrow
  rendering** on x86, and now lands within ~10% of the f32 GEBP — the storage seam pays for itself,
  and the packing copy reading half the bytes nearly closes the gap.
- Both corrections since the first table were emission bugs of the same kind — the rendering was
  right, its C spelling was not — and each was worth more than the seam itself:
  - the **extent-adapted width** (above) stopped peeling 8 of 512 columns to scalar code;
  - **gh-ocannl-614**: the per-lane `fmaf` accumulator update made gcc -O3 spill the whole C-tile,
    ~9x at *every* storage precision (f32 12.6, f16 10.4 GFLOP/s here) — the original claim that
    only the f16 d-bridge shape was affected and that `packmma_par` dodged it did not survive the
    width change. A target whole-vector FMA builtin fixed it, and the `-O3`/`-O2` gap is now within
    run-to-run noise (f32 packmma: 120.2 / 126.1 at -O3 against 121.4 / 130.5 at -O2 over two
    interleaved pairs, against 12.6 vs 112.4 before); see the agent-note for why not
    `dst = a * b + dst`.
- The forced pure-f16 row is the negative control for the gating: computing in f16 where the
  compiler promotes is a ~40x loss against f32-compute — exactly why `fp16_arithmetic` is ignored
  off-native and why the pure-f16 seeds fire only where the probe reports native. The direction is
  not in doubt: the promoted arm does f32 arithmetic with extra per-value conversions, so it cannot
  come out ahead of computing in f32 directly. gh-ocannl-621 left this row alone deliberately —
  its cost is the promotion, and the machine it was measured on is the one where the promotion is
  unavoidable.

### gh-ocannl-621: gcc's fp16 FMA now rounds once

`OCANNL_HALF_FMA` — shared by the scalar rendering, the register tiling's scalar peel and the
per-lane arm of the vector rendering, so the three cannot round differently — had only clang's
single-rounding `__builtin_elementwise_fma` and a promoting `FLOAT_TO_HALF(fmaf(...))` fallback. On
a target with genuine fp16 arithmetic that meant gcc alone rounded twice, disagreeing with clang
and with every GPU backend's `__hfma` / `fma(half, …)`. It now reaches `__builtin_fmaf16` wherever
the ISA has the instruction (`__AVX512FP16__` or `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` — exactly
the condition `cc_backend`'s fp16 probe calls `Native`); on a promoted target nothing changes.

- The two spellings really do disagree: 42039 of 4.1e8 fp16 triples searched, and 3393 of 9.9e7
  with all three operands normal — not a subnormal-corner effect, and not something `float`'s
  `2p + 2` bits retire, since an FMA's exact `a*b + c` can need far more than 24 of them.
- It is also the larger of the two speedups this seam had left at fp16: the promoting arm widens
  and narrows every lane *inside* the k-loop, costing 5–10 instructions per FMA on both AVX512-FP16
  and ARMv8.2-FP16 against under 2 with the native one.
- Guarded on the ISA macro, never on `__has_builtin`, which always answers yes for
  `__builtin_fmaf16`: without the instruction gcc emits a call to `fmaf16()`, which a glibc need
  not export at all.

### gh-ocannl-621: the widths gh-614 left on the per-lane arm

gh-614 covered the AVX/AVX2 widths and listed three residuals: AVX-512 lanes, native-fp16 lanes,
and gcc on aarch64 (unmeasured in either direction). None could be *run* where they were written —
no AVX-512 or AVX512-FP16 hardware, QEMU's TCG implements neither, and no ARM execution — so they
were measured as a compile-time property instead: a census of the innermost FMA-carrying loop of
the emitted micro-kernel, in instructions / vector FMAs / scalar FMAs / stack references per k
step, at `-O2` and `-O3` (the fp16 rows' per-lane figures are with the promoting
`OCANNL_HALF_FMA`, which is what those widths emitted before the section above; the vector-FMA
counts there are doubled because each fp16 lane pair promotes to an f32 vector op). That census is
validated by a positive control — the pre-gh-614 kernel, rebuilt at 97e7d286, reads 18 / 8 / 0 / 0 at `-O2` and 199 / 52 / 6 at `-O3`, and measures 12.57
against ~128 GFLOP/s end to end — so a width reading like the second row is a width that would
lose about an order of magnitude.

| compute × lanes | target | per-lane `-O3` | whole-vector arm |
|---|---|---|---|
| f32 × 8 | x86 AVX2 (shipped) | 200 / 4 vec + 48 scalar / 8 | 18 / 8 / 0 |
| f32 × 4 | x86 SSE (shipped) | 281 / 8 + 48 / 64 | 33 / 16 / 10 |
| f64 × 2 | x86 SSE (shipped) | 99 / **0 + 32** / 18 (also at `-O2`) | 33 / 16 / 10 |
| f32 × 16 | x86 AVX-512 | 31 / 16 / 0 — *no degradation* | 28 / 16 / 0 |
| f64 × 8 | x86 AVX-512 | 417 / **8 + 96** / 0 | 28 / 16 / 0 |
| f16 × 16 | x86 AVX512-FP16 | 81 / 8 / 0 with the promoting macro | 18 / 8 / 0 |
| f16 × 8, × 32 | x86 AVX512-FP16 | 96–165 / 16–32 / 0, likewise | 28 / 16 / 0 |
| f32 × 4 | gcc aarch64 | 294 / **8 + 48** / 58 | 26 / 16 / 0 |
| f64 × 2 | gcc aarch64 | 77 / **0 + 32** / 0 (also at `-O2`) | 26 / 16 / 0 |
| f16 × 8 | gcc aarch64 | 238 / 32 / 45 with the promoting macro | 26 / 16 / 0 |

- **The gap that mattered was gcc on aarch64**, the one the issue called a possible non-issue: it
  is the worst measured case of all, and it is the default on every Linux ARM box (Apple clang
  takes the `__builtin_elementwise_fma` arm and never sees it).
- **Spilling is not the only failure.** Several widths *scalarize* instead — SLP declines to
  reassemble the lane loop, so a lanes-wide FMA becomes `lanes` scalar ones with no stack traffic
  to give it away. f64 × 2 does this on both x86 and aarch64 at both `-O2` and `-O3`, so
  `cc_backend_optimization_level=2` does not diagnose it the way it diagnoses a spill.
- **f32 × 16 under AVX-512 is the one width gcc handles**, which is also the width the issue
  guessed was most at risk. Its row is robustness, not a measured win.
- **At fp16 the vector arm is not where the cost was** — the scalar macro above is. Once
  `OCANNL_HALF_FMA` rounds once, the per-lane fp16 loop censuses at parity with the explicit
  builtin (SLP emits the same `vfmaddph` / `fmla .8h`), so the fp16 rows are robustness rather than
  a measured win. Their guards are the same target question as the macro's, and deliberately so: a
  vector body rounding once against a scalar peel rounding twice would break the bitwise-equality
  promise the rendering makes.

## The NEON decision: pure-f16 wins (M4 Max, Apple clang, `-O3 -mcpu=native`, cc)

The issue's reserved decision point — pure-f16 GEBP vs f32-GEBP-over-narrow-storage on a target
whose fp16 arithmetic is genuinely native (the probe resolves ARMv8.2-FP16 to `Native` here).
Sustained `packmma` throughput, medians of seven back-to-back runs at 20 repeats (single-threaded,
so a cold machine's first run turbos ~40% above the sustained figure — the numbers below are the
warm ones):

| storage → compute | naive | packmma, n=512 | packmma, n=1024 |
|---|---|---|---|
| f32 → f32 | 3.02 | 83.2 | 99.8 |
| bf16 → f32 (widened panels) | 0.48 | 82.9 | 102.5 |
| f16 → f32 (widened panels) | 1.03 | 82.9 | 101.6 |
| f16 → f16, native arithmetic | 3.06 | **132.5 (1.60x)** | **181.6 (1.79x)** |

**Pure-f16 GEBP is worth having: 1.6-1.8x over f32-GEBP-over-narrow-storage**, close to the 2x the
doubled lane count allows, and the gap grows with n. The emission is what the design intends — both
arms resolve to a 4x4 C-tile here, and the k-loop body of each is 16 vector FMAs in the by-element
form (`fmla.8h` against `fmla.4s`) plus its B loads and A splats, with no spills: the same
instruction count over twice the lanes. So the seeds this issue gates on
`native_fp16_arithmetic` earn their place, and the gating is exactly right in both directions: the
same policy forced on promoted x86 hardware loses ~18x (the negative control above).

The measurement is only true at the adapted tile width. At the fixed cap, the same comparison at
n = 512 read 37.0 vs 62.1 GFLOP/s — pure-f16 *losing* 1.7x — entirely because a 48-wide tile peels
32 of 512 columns to scalar code. The honest first measurement therefore answered a second question
it was not asked, and the answer changed the first one; see the tile-width section above.

The seam's x86 verdict — narrow storage pays for itself, landing ~30% off the f32 GEBP — holds on
ARM with the gap closed entirely: the widened-panel arms match plain f32 at n = 512 and edge past it
at n = 1024, because the packing copy reads half the bytes for the same micro-kernel.

### The AVX-512 rows, executed (gh-ocannl-621 follow-up)

The census above was taken where no AVX-512 exists — an Arrow Lake-HX part, where the hybrid
topology fuses AVX-512 off entirely. The fleet's CPU-benchmarking machine is Zen 5, which has it,
so both AVX-512 rows have since run for real (gcc 15: no `__builtin_elementwise_fma`, so the `#elif`
chain does reach them). Nothing in the census needed revising; what changed is the width that
reaches it.

`vector_bytes_setting` capped the auto width at 32 — a 512-bit machine ran the codegen at half its
register file — so the auto width is now 64 where the target has AVX-512 F/BW/VL. `bin/narrow_gebp_bench`
at n = 512, packed GEBP / pool-parallel, GFLOP/s, `taskset -c 0-15`:

| storage | 32 bytes | 64 bytes |
|---|---|---|
| f32 | 130.5 / 119.2 | **225.7 / 188.3** |
| f16 | 108.1 / 85.3 | **189.6 / 130.0** |
| bf16 | 95.1 / 48.6 | **140.6 / 57.9** |

Checksums identical at both widths. The f64 × 8 row has no f64 path through the bench; a
stand-in micro-kernel of the same tile shape measures the arm at 2.27x the per-lane loop
(112.6 vs 49.5 GFLOP/s, bitwise-equal sums, per-lane censusing 192 instructions / 4 zmm FMAs /
146 stack references against 290 / 16 / 0) — real hardware, but not the emitted kernel, so it
corroborates the census rather than replacing it.

Widening a width cannot be done by widening one number. Every rendering here declines below one
full vector, so 64 bytes alone would drop f32 extents of 8..15 — loop and micro-kernel column
extent alike — to the paths 32 bytes vectorizes; and an extent of 40 peels 8 columns at 16 lanes
where 8 lanes divide it, which is how `schedule_mma_matmul`'s extent-adapted leg caught the naive
version of this change. The width is therefore ranked over a ladder
(`Backend_intf.simd_lane_ladder`, halving to a floor of `min vector_bytes 32`): loops minimize
trips (vector steps plus scalar remainder iterations), and the register tiling extends its
peel-cost search from `rn` alone to `(lanes, rn)`. The register-pressure cap stays keyed on the
machine's width, so the wider machine still wins at n = 40 — 118.0 GFLOP/s at a 4x5 tile of 8-lane
vectors against 103.3 at the 4x1 a 32-byte machine's cap allows. Seeding calls the same function,
so it cannot be stricter than the renderer.

The AVX512-FP16 rows stay compile-checked: no AMD part implements AVX512-FP16, and this fleet's
only native-fp16 hardware is ARM, which takes the NEON rows.

## Executed coverage

- `test/operations/tile_mma_narrow.ml`: bf16 in-kernel widened packing (f32 panels,
  `narrow storage bridged: d:bfloat16` only), half hoisted host-converting pack, and a pure-f16
  leg (probe forced native via `OCANNL_CC_FP16_ARITHMETIC`; per-op rounding is unchanged under
  promotion, so parity stays bitwise) — all bitwise against serial twins, with emission pins.
- `arrayjit/test/test_vectorized_codegen.ml`: the whole-vector FMA arms at every (compute
  precision, lane count) the emission can reach (gh-ocannl-621) — which widths have an arm at all,
  the guard each sits behind, the operand order, and the AVX-512 forms' mask and rounding
  arguments. A kernel pins only the width its own `cc_vector_bytes` selects, and the setting is
  read once per process, so the table is printed directly.
- `test/operations/schedule_mma_matmul.ml`: a full-mantissa f32 leg pins that the accumulator
  update stays fused (gh-ocannl-614) — the only bitwise leg whose products are inexact, hence the
  only one that can tell a fused update from a multiply and an add; the half/bf16/fp8 whole-triple
  legs now register-tile
  on the C backends (bitwise, narrow-exact inputs); transposed-B legs still pin the decline; an
  8x40 leg pins the extent-adapted width (`full blocks 8x40 of 8x40` — peel-free at NEON's 4 f32
  lanes and at AVX2's 8, where the old cap took 24 on both), bitwise against its serial twin.
- `test/operations/sketch_family_tree.ml`: the seeding scenarios above.

## Relations

- [tensorize-mma](tensorize-mma.md): the CPU register tiling this extends (gh-ocannl-469), and the
  gh-545 accumulator-precision precedent the residency rule follows.
- gh-ocannl-516/517: the parent issues; their landed seams are the substrate.
- [gh-ocannl-530-pool-uniformity](gh-ocannl-530-pool-uniformity.md): supplies
  `native_fp16_arithmetic` and the uniform pool the seeds tune for.

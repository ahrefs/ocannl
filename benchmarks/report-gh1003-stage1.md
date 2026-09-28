# gh-ocannl-1003 stages 0 and 1: where the online attention's time goes on Metal, and lanes for its value pass

Measurement report for the first two stages of ahrefs/ocannl#1003 (plan in the issue's
2026-09-28 analysis comment). Stage 0 re-measures the gh-483 matrix
([report-gh483-online-softmax.md](report-gh483-online-softmax.md)) after gh-ocannl-995 changed the
default Metal schedule, and attributes the online arm's time per fission segment. Stage 1 is a
default-schedule change: the default GPU annotator now maps the online-softmax hoist's value pass
to `Grid (b, s, h) -> Serial t -> Workgroup e` ("lanes"; `Schedule.default_gpu`, pinned by
`test/operations/gpu_serial_lanes`).

**Verdict.** The value pass the rewrite hoists IS the dominant cost of the online arm at short
context -- but not for the reason the analysis gave. The default annotator gives the composed and
the hoisted value pass the same two chain loops and the same thread count (1024 threads at seq
128); what the hoist changed is the order inside the thread: the composed pass reduces over `t`
innermost in a register, the hoisted one walks `h, t, e` with `e` innermost and read-modify-writes
`O` in device memory once per `(t, e)`, through the Metal volatile-read workaround. At seq 128 that
kernel costs 3.02 ms per layer against the composed value pass's 1.12 ms -- more than the whole
online-vs-composed step gap over four layers. Lanes put it on 262144 threads, one lane per `O`
cell: the online arm with stored scores (`--ocannl_online_softmax=true`) becomes **14% faster at
seq 128, 8% at seq 512 and 7% at seq 1024** than on master, and now beats the composed form at
every length measured (0.923x, 0.939x, 0.771x of it, against 1.073x, 1.025x, 0.825x before). The
tuned arm replays at half its master step time (18.9 -> 9.7 ms at seq 128; caveat under *Tuned*).
The measured commit also gave lanes to the recomputed-scores form, where every lane recomputes the
inlined `q . k` in the preamble; that won 11% at seq 128 and lost 12% at seq 512 and **51% at seq
1024**, so the landed rule requires a loop-free preamble and the recomputed form's emitted Metal is
master's again (its rows below are master's). cc is untouched: the CPU preset is unchanged and the
generated C is byte-identical for all nine fixture x arm cells. Every arm reproduces master's
losses to 1.35e-7 relative.

## Protocol

Hardware: Apple M4 Max (40-core GPU), macOS 26.6.2, host `mac-studio`, under the fleet's exclusive
`measurement` reservation `gh1003-stage01-mac-studio-1` (held 12:43Z-13:19Z on 2026-09-28; no other
fleet batch on the box; host snapshots between phases show desktop activity only -- Spotlight,
XProtect, the Metal shader compiler -- and one foreign test executable in the final snapshot at
13:19Z, at the end of the cc leg, see there). Trees: *base* = staging master `1fcef8bee`; *fix* =
this branch at `03fc65e50` (the lane geometry before the loop-free-preamble rule). The landed head
adds `77ba0cc11`; its Metal sources, compared with both trees for 3 fixtures x 3 arms, are
byte-identical to *fix* for the composed and stored arms and to *base* for the recomputed arm.
The *fix* composed rows are therefore an A/A control (the composed form has no hoisted pass and
the same code in both trees).

Every Metal cell is `bench_gpt` on the fixture named, untuned default pipeline, f32, one process
per cell, arms as flags exactly as in the gh-483 report: *composed* (defaults), *stored*
(`--ocannl_online_softmax=true`; head width 32 exceeds the recompute cap 16, so the scores stay one
stored `[seq, seq]` buffer), *recomputed* (plus `--ocannl_virtualize_max_inline_reduction=32`).
Repeats: three on Metal -- arms forward with base before fix, then reversed with fix before base,
then forward again -- so an order effect would split the repeats of one cell. The spread column is
the widest p90/p10 of any repeat of the cell. Fixtures `gpt2_mini` (8 x 128), `gpt2_mini_s512`
(2 x 512), `gpt2_mini_s1024` (1 x 1024) are the m4-max `content-v1` entries of
`fixtures/DIGESTS.txt` (matched at run time). The driver ran every command from each tree's
`benchmarks/` with `OCANNL_*`, `BENCH_*` and OpenMP variables cleared. The per-segment attribution
is `bench_gpt_diag` with `BENCH_SEG_TIMES=1`: every fission segment compiled as its own routine and
timed min-of-20 with a sync per run, so each segment time carries a ~0.15 ms launch floor and the
segment sums exceed step times.

## Step times, Metal

| fixture | arm | tree | p50 per repeat (ms) | median p50 | vs base composed | vs base, same arm | p10..p90 spread |
|---|---|---|---|---|---|---|---|
| `gpt2_mini` (8 x 128) | composed | base | 62.4, 62.3, 62.9 | 62.4 | 1.000x | 1.000x | up to 1.029x |
| `gpt2_mini` (8 x 128) | composed | fix | 62.5, 62.4, 62.5 | 62.5 | 1.000x | 1.000x | up to 1.015x |
| `gpt2_mini` (8 x 128) | stored | base | 67.0, 67.1, 66.9 | 67.0 | 1.073x | 1.000x | up to 1.028x |
| `gpt2_mini` (8 x 128) | stored | fix | 57.6, 57.6, 57.9 | 57.6 | **0.923x** | **0.860x** | up to 1.025x |
| `gpt2_mini` (8 x 128) | recomputed | base = landed | 68.2, 68.3, 68.2 | 68.2 | 1.092x | 1.000x | up to 1.013x |
| `gpt2_mini` (8 x 128) | recomputed | fix at `03fc65e50` | 60.5, 60.7, 60.5 | 60.5 | 0.969x | **0.887x** | up to 1.014x |
| `gpt2_mini_s512` (2 x 512) | composed | base | 61.9, 62.0, 62.0 | 62.0 | 1.000x | 1.000x | up to 1.009x |
| `gpt2_mini_s512` (2 x 512) | composed | fix | 62.4, 62.0, 62.0 | 62.0 | 1.001x | 1.001x | up to 1.024x |
| `gpt2_mini_s512` (2 x 512) | stored | base | 63.5, 63.6, 63.3 | 63.5 | 1.025x | 1.000x | up to 1.025x |
| `gpt2_mini_s512` (2 x 512) | stored | fix | 58.1, 58.3, 58.2 | 58.2 | **0.939x** | **0.916x** | up to 1.030x |
| `gpt2_mini_s512` (2 x 512) | recomputed | base = landed | 64.1, 63.9, 63.8 | 63.9 | 1.031x | 1.000x | up to 1.015x |
| `gpt2_mini_s512` (2 x 512) | recomputed | fix at `03fc65e50` | 71.7, 71.9, 71.7 | 71.7 | 1.157x | **1.122x** | up to 1.015x |
| `gpt2_mini_s1024` (1 x 1024) | composed | base | 51.0, 51.1, 50.9 | 51.0 | 1.000x | 1.000x | up to 1.037x |
| `gpt2_mini_s1024` (1 x 1024) | composed | fix | 51.2, 50.9, 51.2 | 51.2 | 1.003x | 1.003x | up to 1.009x |
| `gpt2_mini_s1024` (1 x 1024) | stored | base | 42.1, 42.5, 41.9 | 42.1 | **0.825x** | 1.000x | up to 1.030x |
| `gpt2_mini_s1024` (1 x 1024) | stored | fix | 39.3, 39.2, 39.3 | 39.3 | **0.771x** | **0.934x** | up to 1.026x |
| `gpt2_mini_s1024` (1 x 1024) | recomputed | base = landed | 45.7, 45.4, 45.5 | 45.5 | 0.893x | 1.000x | up to 1.024x |
| `gpt2_mini_s1024` (1 x 1024) | recomputed | fix at `03fc65e50` | 68.7, 68.7, 69.0 | 68.7 | 1.348x | **1.510x** | up to 1.030x |

Bold marks a ratio outside 5% of its reference. Every bold ratio lies outside its cell's spread,
with all three repeats of the cell on the same side; the composed A/A pairs agree to 0.3%, inside
their spreads. Against the gh-483 report, gh-ocannl-995 halved the composed step at seq 128
(125 -> 62.4 ms) and cut seq 512 by 7x (437 -> 62.0 ms), but the online arm's relative loss stayed
(3.3% -> 7.3% at seq 128, 8.0% -> 2.5% at seq 512; seq 1024 still wins 17.5%) -- the short-context
loss was not an artifact of the gh-995 cliff, and stage 1 is what removes it.

## Stage 0: where the online arm's time goes (per segment, Metal, ms per layer, layer 0)

Base tree, min-of-20 per segment:

| fixture | form | scores (+ max / scan) | softmax | value pass | value-pass threads | attention total |
|---|---|---|---|---|---|---|
| `gpt2_mini` | composed | 1.39 | 0.71 | 1.12 | 1024 | 3.21 |
| `gpt2_mini` | stored | 1.37 (scores + scan) | -- | **3.02** | 1024 | 4.38 |
| `gpt2_mini` | recomputed | 1.19 (scan) | -- | 3.72 | 1024 | 4.91 |
| `gpt2_mini_s512` | composed | 1.36 | 0.60 | 1.82 | 4096 | 3.78 |
| `gpt2_mini_s512` | stored | 1.39 | -- | **2.60** | 4096 | 3.99 |
| `gpt2_mini_s1024` | composed | 2.11 | 0.65 | 4.91 | 8192 | 7.67 |
| `gpt2_mini_s1024` | stored | 2.15 | -- | 2.88 | 8192 | 5.03 |

- **The value pass is the online arm's dominant cost at short context.** At seq 128 it costs
  3.02 ms against the composed pass's 1.12 ms: +7.6 ms over four layers, while the online arm's
  whole step is +4.6 ms over composed. The scores-plus-scan kernel costs what the composed
  scores-plus-max kernel did (1.37 vs 1.39 ms), and the scan kernel of the recomputed form alone is
  1.19 ms: nothing to fix in the scan at these lengths.
- **The thread count did not change; the inner loop did.** The value-pass threads column is the
  same for both forms at every length -- the default GPU annotator takes two chain loops
  (`Grid b x Workgroup s`, or the gh-995 suffix pair) whatever the loop order -- so the analysis
  comment's premise that the hoist demoted the pass from `(b, s, h, e)`- to `(b, s, h)`-parallel is
  wrong for the untuned pipeline. What the hoist moved is `e` inside the serial key loop: the
  composed pass reduces over `t` innermost into a register accumulator ("vol:1 of 1
  accumulation(s)"), the hoisted pass accumulates into `O` in device memory once per `(t, e)`, with
  both of its reads, `O` and `V`, through the Metal volatile-read workaround (gh-ocannl-782; "1
  volatile rmw read(s)"). Each thread still walks the full `h x t x e` space.
- **At seq 1024 the comparison inverts:** the composed pass's `[seq, seq]` reads dominate
  (4.91 ms) and the hoisted pass is already cheaper (2.88 ms) -- the regime where the gh-483 report
  measured the rewrite's win.
- **What the next stages should expect.** Lanes remove the per-thread serial walk, not the
  device-memory accumulation: after stage 1 the pass still does one read-modify-write of `O` per
  key step, now coalesced across lanes. A register accumulator for `O` needs the lane's `t` loop
  innermost again -- the per-block `P_blk V_blk` contraction stage 2 of the plan emits. The seq-1024
  rows show the scores-plus-scan kernel (2.15 ms, 8192 threads, one row per thread) becoming the
  larger half of the attention once the value pass is fixed: that is the cost the block-tiled scan
  (stages 2-3) is for.

### The tuned arm (gpt2_mini, stored scores): is the score matmul tensorized?

One search per tree (`BENCH_TUNE=1`, `autotune_log=true`, fresh cache directories), **crowned on a
shared box** before the reservation and replayed under it with `--ocannl_autotune_search=false`:

| tree | replay p50 per repeat (ms) | median | tuned search's own best (arm A) |
|---|---|---|---|
| base | 19.1, 18.9, 18.9 | 18.9 | 18.49 ms |
| fix | 9.64, 9.67, 9.74 | 9.67 | 6.89 ms |

The replays are timing-grade (spread under 3%); which winner each search crowned is not, since the
searches shared the box. What the search log answers, for either tree:

- **The score matmul is seeded as tensorized, and no tensorized candidate for it ever compiles.**
  It shares its fission segment with the online-softmax scan (which writes `n257_max_vals` and the
  normalizer), and every sketch over that segment -- the tensorized `mma-gpu` families and the
  scalar register-tile `gpu` families alike, with and without `bgrid` -- fails at
  `Low_level.validate_parallel`: "write to materialized node n257_max_vals is not nested under
  annotated loops covering all active hardware dimensions". The segment's companion coverage
  (`Schedule.aligned_chains`) has no chain for a `Scan_loop` nest.
- `report.mma_timed` counts 114 of 162 seeded tensorized candidates as timed: those are the other
  sites' (projections, FFN, the lm_head). The crowned winner's 21 tensorized statements do not
  include the scores -- the replayed kernel writing `n248` is a scalar 4x4 register tile with the
  scan split into its own kernel.
- The base winner's dominant kernel is the hoisted value pass (2.99 ms of the 37.3 ms segment sum):
  the search leaves it on the presets' geometry, being an axpy nest no sketch family targets. With
  lanes it drops out of the top five.

Reported, not fixed here: the coverage decline is a stage-2 prerequisite (the score contraction
must leave the scan's segment, or the scan must get geometry from `aligned_chains`, before
`Tensorize` can take it).

## Stage 1: the value pass under lanes (per segment, Metal, ms per layer, layer 0)

| fixture | form | value pass, base | value pass, lanes | threads, base -> lanes |
|---|---|---|---|---|
| `gpt2_mini` | stored | 3.02 | **0.71** | 1024 -> 262144 |
| `gpt2_mini_s512` | stored | 2.60 | **1.31** | 4096 -> 262144 |
| `gpt2_mini_s1024` | stored | 2.88 | **2.32** | 8192 -> 262144 |
| `gpt2_mini` | recomputed (`03fc65e50` only) | 3.72 | 1.40 | 1024 -> 262144 |
| `gpt2_mini_s512` | recomputed (`03fc65e50` only) | 3.43 | 5.08 | 4096 -> 262144 |
| `gpt2_mini_s1024` | recomputed (`03fc65e50` only) | 4.35 | 9.93 | 8192 -> 262144 |

The emitted kernel (`gpt2_mini`, stored):

```
gid.z -> b, gid.y -> s, gid.x -> h
for t in 0..127 { p := exp(x[b,s,h,t] - m[b,s,h]) / l[b,s,h];   // lane-uniform, per lane
                  e := lid.x; O[b,s,h,e] = fma(p, V[b,t,h,e], O[b,s,h,e]) }
```

The gain shrinks with sequence length because the presets' geometry already grows with `seq`
(1024 -> 8192 threads) while the lanes' does not. The recomputed form shows the trade the landed
rule avoids: every lane recomputes the preamble, so lanes multiply its work by the lane width
(32); with the score `q . k` inlined there, that is 32 redundant 32-step dot products per key step.
It pays at seq 128, where the presets' 1024 threads leave the GPU idle, and loses from seq 512 on,
where the presets' 4096-8192 threads already fill it. A thread-count threshold could keep the
seq-128 win, but its crossover depends on the device and the preamble's cost, and the recomputed
form is not the default (it loses to storing at every cell here, as in gh-483). So the rule
declines any preamble holding a loop; `test/operations/gpu_serial_lanes` pins both sides, on a
hand-built nest and in the real pipeline.

## cc: no-regression leg

The CPU preset is unchanged, and the generated C for all nine fixture x arm cells is byte-identical
between the trees (`bench_gpt_diag` with debug files, compared file by file). The timed leg, the
stored arm, two repeats, base first then fix first:

| fixture | tree | p50 per repeat (ms) | median | vs base | p10..p90 spread |
|---|---|---|---|---|---|
| `gpt2_mini` | base | 2105, 2386 | 2246 | 1.000x | up to 1.442x |
| `gpt2_mini` | fix | 2083, 2117 | 2100 | 0.935x | up to 1.086x |
| `gpt2_mini_s512` | base | 2263, 2284 | 2274 | 1.000x | up to 1.036x |
| `gpt2_mini_s512` | fix | 2263, 2263 | 2263 | 0.995x | up to 1.056x |
| `gpt2_mini_s1024` | base | 1658, 1759 | 1708 | 1.000x | up to 1.072x |
| `gpt2_mini_s1024` | fix | 1716, 1726 | 1721 | 1.008x | up to 1.031x |

These ratios are noise around identical code, and the spreads say so: the `gpt2_mini` base repeat
2 (p90 3061 ms, spread 1.44x) is a disturbed window -- the host snapshot that closed the leg shows
another checkout's test executable running -- and its 0.935x is not a measurement. The other two
fixtures agree to 1%.

## Not measured here

- **CUDA and HIP.** The lane geometry is the shared GPU preset's, so it changes eligible CUDA/HIP
  schedules too; those boxes have not run it. The same flags on `bench_gpt` reproduce this matrix
  there.
- **Training.** `bench_gpt` measures the forward. The fused backward of gh-ocannl-1002 emits nests
  of the same hoisted shape and gets lanes under the same conditions -- except that a nest whose
  preamble precedes two sibling channel loops (dK and dV under one preamble) is not reached: the
  path must continue into exactly one loop.
- **A thread-count-aware rule for loop-carrying preambles**, which would recover the recomputed
  form's seq-128 win (0.887x) without its long-context losses, needs a device-parameterized
  crossover and was left out (see above).

# gh-ocannl-514 phase 6: the branch-and-bound schedule search, evaluated

> **Fixture provenance: unverifiable** (gh-ocannl-919). Original measured files for the historical Metal leg
> were not recovered. Later `content-v1` M4 baselines do not certify these numbers.
> See the [historical provenance audit](README.md#historical-metal-fixture-provenance).

Evaluation-as-research-output for the gh-514 arc (schedule inference as branch-and-bound,
phases 0–5): how much of the schedule space the verdicts and bounds dispatch without pricing or
timing, what the fitted envelopes and the continuous agreement check say about model-vs-measured
divergence per backend, what the enablement ordering buys the tuned flip chain at a sane budget,
and what the untuned regime's model-argmin (family, placements, tile lattice) delivers against
the plain default — including where the beam remains competitive and where the new machinery
loses. Measurement-only: the one code change is the `model_pruned`/`bound_pruned` fields added
to `bench_mlp`'s `BENCH_TUNE_REPORT` dump (the gh-558 instrument pattern).

**Verdict in one paragraph.** The construction-time fathoms carry the untuned regime: on
`gpt2_mini`, 25 of 30 per-segment family searches fathom **at the root** — the whole family
dispatched by one bound evaluation against the default's score, zero pricing — and every
untuned pick honored ties-to-default with zero step-time regression (and zero gain: the
default preset is the model-argmin on every cell measured, the honest null). The enablement
ordering does exactly what gh-558 asked: on **both** GPU boxes a budget-5 flip chain reaches a
family-unlocking materialize flip at position ≤ 2 and seeds 16 tensorized candidates where the
legacy cost ordering seeds **zero in five of five** flips — but tensorized loses on all three
substrates at these precisions, so what ships never changes on the GPU boxes, and on Metal the
promotion actively costs the chain its best flip (the negative result below). Measured-incumbent
bound pruning at same-run-fitted envelopes is admissible in practice (winner parity everywhere)
but bites rarely (0–1 candidates on the A/B searches), and the one search where it bit hard
over-pruned under an envelope the very same run had outrun — with the phase-0 agreement check
firing 33 warnings, exactly as designed. The tile lattice's box walk is cheap and its bounds
are monotone as pinned, but on the one segment where it fired every lattice completion died at
candidate build (the phase-2 boundary doctrine's cost, now measurable).

## Provenance

Tree `b65e0a7f` (staging master `33d3a963` + the dump instrument), all phases 0–5 merged
(staging PRs #320–#327). Three machines, one process per cell, fresh `autotune_cache_dir` per
cell (one `Train.tune_placements` invocation — the searches within a cell share it, which is
harmless: cache identity includes the placement vector, so distinct arms and flips never
replay each other's entries), `~rounds:0` as `bench_mlp` always passes, `--ocannl_autotune_log=true` throughout:

- **metal**: Apple M4 Max (Metal), `BENCH_PRECISION=f16` for the tuned cells, f32 for the
  untuned reruns (see the harness-gap note).
- **cuda**: ROG NUC, RTX 5070 Ti Laptop GPU (WSL2), f16 tuned / f32 untuned.
- **hip**: Minix, Ryzen AI MAX+ 395 w/ Radeon 8060S, gfx1151 (WSL2), bf16 throughout —
  rocWMMA's only route on RDNA3.5, and bf16 training keeps `Plain_step` (only f16 is
  loss-scale-gated), so the untuned cells run at the precision under study; `taskset -c 0-15`
  on every run (the gh-530 freeze mitigation).

Cells per box, in order (driver checked in as `benchmarks/gh514_cells.sh`): **A** tuned `mlp_wide`
placement A/B with `autotune_calibration_file`; **fit** `tools/fit_envelope.exe` over A's rows,
all later cells pinning the fitted `model_peak_*`; **B** tuned A/B with
`autotune_bound_pruning` off vs on; **C** the flip chain at budget 5, `tune_flip_ordering=cost`
vs `enablement`, two replicates each, plus enablement+pruning; **D** untuned `mlp_wide` (f32 on
the f16 boxes, bf16 on hip)
and `gpt2_mini` (f16/bf16 forward): plain default vs `model_default_schedule` vs
`+model_default_placements=5` vs `+model_default_geometry_lattice=true`.

## Calibration and the envelopes (cells A/fit)

Per-leg exactness in practice, on `mlp_wide` training steps:

| box | rows | approx-flops | approx-bytes | fitted peak_flops | binding row |
|---|---|---|---|---|---|
| metal f16 | 102 | 38 | **102** | 2.459e11 | `F_split[...]` |
| cuda f16 | 102 | 38 | **102** | 4.459e11 | `F_split[...]` |
| hip bf16 | 82 | 37 | **82** | 1.486e12 | `F_sketch[mma-gpu 32x32x0, ...]` |

Two structural facts with consequences downstream:

1. **The memory leg is unfittable from this workload alone**: every row's byte count is a
   guards-taken/union upper bound (`bytes_approx`), so per the per-leg exactness rule no
   bandwidth constant fits — the **fit** constrains only the compute leg. The **envelope in
   force** is not compute-only: `Autotune.envelope` falls back to the backend's class-level
   advisory `peak_memory_bandwidth` when no override is configured, and all three backends
   provide one, so every search in this report ran under fitted-compute + class-bandwidth
   legs. The distinction matters for honesty in both directions: the compute constants are
   demonstrated floors, while the bandwidth constants are unaudited class advisories (Metal's
   is a known understatement on this machine — the phase-0 notes record 2e11 advisory against
   ~4e11 achievable). Fitting the memory leg needs rows with exact footprints
   (elementwise/streaming kernels), i.e. calibration diversity, not more of the same
   workload.
2. On hip the compute-leg binding row is an unstaged **rocWMMA candidate at 1.49 TFLOP/s
   achieved** — the fitted peak is a demonstrated floor, and it is 3.3× cuda's and 6× metal's
   fitted constants on these cells (fitted peaks are per-machine *and* per-what-was-measured;
   they are not hardware spec sheets).

## Tuned bound pruning (cells B), and the blind spot demonstrated (cell C-enab-bp/cuda)

With the fitted envelopes pinned, `autotune_bound_pruning=true` on the `mlp_wide` A/B:

| box | off: timed (A+B) | on: timed | bound_pruned | best off → on |
|---|---|---|---|---|
| metal | 41+61 | 41+61 | 0 | 7.62 → 7.46 ms (spread) |
| cuda | 41+61 | 41+61 | 0 | 4.10 → 4.05 ms (spread) |
| hip | 31+51 | 31+50 | **1** | 1.272 → 1.258 ms (spread) |

The admissible direction holds — winners are within run-to-run spread everywhere — but the
floors (fitted compute leg maxed with the class-bandwidth traffic leg) sit well below the
measured incumbents on these kernels, so the gate almost never fires on the A/B searches. Where it did fire hard, it demonstrated the documented
blind spot: in cuda's C-enab-bp cell, one nested flip search pruned **16** candidates and
finished at 6.59 ms where its unpruned analogues reached ~5.0 ms — the envelope had been
fitted from cell A, this later search produced candidates *faster than anything the fit ever
saw*, and the same run logged **33 BOUND VIOLATION** warnings (metal logged 2, hip 0). That is
the phase-0 continuous agreement check working exactly as specified: fitted peaks are
demonstrated floors, not certified maxima; a violation is a refit prompt, and
`fit_envelope --margin` is the explicit headroom knob. The chain-level outcome was unaffected
(the refinement lost to arm B regardless), but the episode is the concrete argument for why
`autotune_bound_pruning` defaults to off and why the check is unconditional.

## The enablement ordering at budget 5 (cells C) — the gh-558 replication, and its price

The decision surface and its classification, identical across replicates:

| box | candidates | enablement-promoted | promoted flips (rank order) |
|---|---|---|---|
| metal f16 | 50 | 3 | `materialize batch_x`, `materialize n22_cast`, (+1) |
| cuda f16 | 50 | 3 | same shape |
| hip bf16 | 66 | 3 | same shape (gh-558's surface had 66 too) |

**On hip — the cell gh-558 measured — the mechanism closes.** Cost ordering spends all five
flips on `batch_x` + four `inline` flips that seed nothing (**0 mma seeded, five of five, both
replicates** — byte-for-byte gh-558's budget-5 failure); enablement ordering reaches
`materialize n22_cast` at flip 2, seeds **16 tensorized candidates**, the chain accepts it, and
subsequent flips keep the family expressible (10–16 seeded each). What gh-558 needed budget 18
to find, the comparator finds at budget 5 — pre-compile, from the seeders' own classification.
The chain's own best also improves (1.96–2.05 ms vs 2.16–2.20 ms cost-ordered). It still does
not ship: arm B (materialize-all) wins at 1.26–1.30 ms on this cell in every search, exactly
gh-558's conclusion — whole-graph materialization exposes a two-site tensorization the
targeted flips cannot reach.

**On cuda the same reachability holds** (flip 2 seeds 16 under enablement, zero under cost;
tensorized loses at 10.2–12.3 ms vs 4.4 ms — consistent with the gh-528 attribution that cuda
tuning wins are scheduling, not tensorization) and arm B ships regardless.

**On metal the promotion has a price, and it loses.** Arm A wins the A/B on metal, the chain
ships its refinement in all five C cells — and the *cost-ordered* chains ship better results
(6.55/6.64 ms) than the enablement-ordered ones (7.03–7.14 ms): the two promoted twins occupy
budget slots 1–2, seed a family whose candidates lose catastrophically here (mma_best 79–92 ms
against 7.5 ms — Metal f16 simdgroup on this cell), and push the actually-winning cheap
`inline n32_relu.grad` flip (cost 1024, rank 5 under cost ordering) out of the budget. The
enablement prior models expressibility, not profitability — on a backend where the family it
unlocks is hopeless, promotion is pure opportunity cost. This is why `tune_flip_ordering=cost`
stays available, and it is a real argument for a future profitability term (e.g. consulting
the measured `mma_best_ms` of prior searches in the cache) before promotion.

A composition detail worth recording: a single twin does not always enable. In metal's
C-enab-1 the chain accepted `batch_x` first and flip 2 (`n22_cast`) then seeded 16; in
C-enab-2 `batch_x` was rejected on its timing draw and the same `n22_cast` flip from the
un-flipped base seeded **0** — the site's triple needs both operand sides reduced.
Enablement is a property of placement *sets*; the greedy chain reaches set members only
through accepted prefixes, which is exactly the limitation a wider placement B&B (more than
one path through the product space) would lift.

## The untuned regime (cells D, f32; ledgers under `autotune_log`)

The advisory contract held everywhere: every cell chose the default (ties-to-default on equal
scores), every step time matched the plain default within spread, and no pick ever regressed:

| box | mlp_wide p50 default → model → +plc → +lattice |
|---|---|
| metal f32 | 5.97 → 5.97 → 5.99 → 5.96 ms |
| cuda f32 | 4.13 → 4.13 → 4.13 → 4.10 ms |
| hip bf16 | 3.56 → 3.57 → 3.58 → 3.58 ms |

(An earlier supplemental hip f32 battery agreed on every conclusion — chose default, ties,
step parity at 1.50–1.52 ms — but is not tabulated: it carried the box's bf16-fitted compute
peak into f32 compiles, the very cross-precision provenance bounded below, and the driver
deliberately produces no such combination.) The hip bf16 row is the sharper null: at the precision under study the mma family is genuinely
in play, and the ledger shows why the pick still stays default — under the untuned default
placements the cast twins are virtual, the site reads f32 masters, and the tensorized branch
refutes (the very premise the tuned cells measure), so the lattice behind it is unreachable;
the placement walk then prices all 32 leaves (375 family evaluations, 375 unbuildable sketch
attempts) without surfacing a single buildable tensorized candidate. Reaching the family
untuned needs the placement leaves' lowerings to make the twins' materialization pay off in
the *model* — and under the envelope in force (fitted compute + class-bandwidth legs) the
twins' traffic deltas are the same microseconds-against-a-0.1-ms-slack mismatch as the
placement walk's, so no leaf can displace the default on price while the family it would
unlock stays unbuildable.

That the model-argmin *is* the default on these cells is the honest null: `gpt2_mini`'s
segments are memory-bound (where the default preset is provably optimal at the floor —
phase 4a's design case) and `mlp_wide`'s sketch candidates all fail validation on these
backends' untuned pipelines. The value shown is in the **ledger** — what was dispatched
without pricing:

- **Wholesale family fathoming**: on `gpt2_mini`, 25 of 30 per-segment family searches
  fathomed at the root (both GPU boxes, identical counts) (`0 expanded … 1 fathomed`): one floor evaluation against the default
  incumbent dispatched the entire family. The remaining segments expanded 8 and rejected
  their candidates at validation (`unscored (rejected)` — the split the phase-4a review
  demanded, so cost-model gaps are not misclassified).
- **The placement walk** (`model_default_placements=5`): 31 expanded / 32 scored /
  **0 fathomed** on all three boxes — the full 2⁵ product walked, ~0.4–0.6 s of compile-time
  at N=5, ties to the default. The zero is a scale mismatch, not a missing leg: the traffic
  leg was present (class bandwidth constants — see the calibration note), but on these cells
  both the floor and the incumbent are dominated by the compute leg (metal: family bound
  7.71 ms against the default's 7.82 ms score; cuda: 4.25 vs 4.31), and a placement
  commitment's certain-traffic delta — a few MB of materialized activations over a
  ~10¹¹ B/s class constant — is tens of microseconds against a ~0.1 ms incumbent-floor
  slack. Placement fathoming becomes real where the materialization deltas are commensurate
  with that slack (bigger footprints, or tighter incumbents), and where the bandwidth
  constant is *fitted* rather than advisory, so the fathom is trustworthy in the admissible
  direction. (The regression test pins the fathoming mechanics under a bandwidth-dominant
  envelope; this is a deployment observation, not a code gap.)
- **The tile lattice** (`model_default_geometry_lattice=true`): on `mlp_wide`/f32 the mma
  branch refutes (no f32 tile at these capabilities), the lattice is unreachable, and the
  ledger is bit-identical to the non-lattice cell — the excluded branch costs nothing, as
  designed. On `gpt2_mini`'s one mma-eligible segment the lifted walk expanded 123 (cuda) /
  131 (hip) nodes — and every lattice completion was **rejected at candidate build** (241/255
  unbuildable vs 69 without the lattice), scoring nothing. The boxes' corner verdicts and the
  traffic bounds worked (nothing needed fathoming — the incumbent was far above the floors);
  what killed the leaves is builder-settled analysis (companion coverage and kin), which the
  phase-2 boundary doctrine deliberately keeps *out* of the tree's verdicts. The lattice's
  search-side machinery is sound and cheap (~0.6–0.8 s for the whole walk), but on segments
  like this it prices a space whose members cannot build — lifting statically-decidable
  builder preconditions into tree verdicts is the natural follow-up if the lattice is to pay
  off beyond synthetic sites.

**A precision mismatch in the retained metal/cuda untuned cells, and its bound**: those D
batteries ran at f32 while carrying the compute peaks fitted from the f16 cell A — a
cross-precision application of achieved-throughput constants that could in principle shift the
roofline balance. It could not have changed these picks: in every such cell the alternatives
lost by *buildability* (all sketch candidates rejected at validation — `15 unbuildable` against
`11 scored`), which no envelope constant can alter, and the placement walk's zero-fathom result
only becomes more conservative under an overstated compute peak. The checked-in driver now
withholds cross-precision fits from the D cells (they then run under the class advisories —
also the deployment default for untuned compiles); hip's bf16 battery is precision-consistent
throughout.

**A harness gap found and worked around**: at f16 `bench_mlp`'s training step is
loss-scale-gated (`Host_gated`/`Device_gated`), and those arms of
`Bench_harness.compile_train_step` bypass the `model_default` gate — only `Plain_step` routes
through it (bf16 keeps `Plain_step`, so hip is unaffected). The f16 D cells therefore measured
nothing about the model (their variants are identical executions; metal's late-campaign f16
"D" cells also drifted thermally), and the metal/cuda untuned comparisons above are f32
reruns. The gap is now closed — `compile_train_step` routes *every* untuned compile of a step
through `Autotune.model_default`, the `Host_gated` gradient and optimizer routines and the
`Device_gated` fused routine alongside `Plain_step`'s. That the optimizer routine is in scope
matters for reading a future f16 D cell: the leg's step time is both routines, so both compiles'
model-selected schedules are part of the treatment (only the *tuned* arm keeps that routine on a
plain compile, since timing it is not what the leg measures). The driver's D cells accordingly
run at the requested precision instead of dropping to f32 (see Reproduction — a `f16:f32`
precision argument restores the workaround). The f32 reruns reported above stand as measured;
f16 model-vs-default cells are measurable from here on without it.

## Where the beam remains competitive

Stated plainly, per the issue's demand for honest reporting: on every cell measured, the
curated menus' beam already contained everything that could win. The B&B's measured value in
this evaluation is **economy and reachability** — wholesale fathoming of families the default
already beats, pre-compile refutation witnesses, an enablement comparator that reaches
family-unlocking flips at one third of the budget gh-558 needed — not shipped-time
improvements, which on these workloads are still decided by the placement A/B and the
fissioned preset sweeps. The regimes where the bound machinery should strictly dominate
remain: (a) placement search on cells whose materialization deltas are commensurate with the
incumbent slack, under a *fitted* memory leg (needs calibration diversity — the class
advisories in force here are unaudited), (b) the lattice on sites whose staged candidates
actually build (needs the builder-precondition lift), and (c) workloads where the default is
*not* the model-argmin. None of those were
reachable on these cells, and this report records that rather than sampling around it.

## Reproduction

One driver per box, and the driver is the reproduction — it pins every benchmark-mode
variable, tuner knob, numerics-policy key, and pipeline gate the cells depend on (ambient
configuration cannot change a treatment), generates missing fixtures, and fails loudly on any
cell:

```bash
benchmarks/gh514_cells.sh hip bf16 ~/ocannl-staging taskset -c 0-15
```

(`cuda f16` and `metal f16` analogously; results land in `~/gh514-eval-results-<backend>-<precision>/`.)
To reproduce a single cell — say the hip budget-5 enablement headline — run the driver and read
that cell's `.out`/`.err`, or lift the cell's exact, fully-pinned argv from the script: the
cells are the commands, and the script is deliberately the single place they are maintained.

The D rows of the f16 boxes were measured with the untuned mlp cells dropped to f32 (the
harness gap above). The driver no longer drops them — with the gate reaching the gated step
shapes, `cuda f16` now measures the f16 model-vs-default comparison the tables could not — so
rerunning the f32 D cells takes the split-precision argument that restores the workaround
(results land in a `-df32`-suffixed directory of their own):

```bash
benchmarks/gh514_cells.sh cuda f16:f32 ~/ocannl-staging
```

That rerun is not bit-for-bit the tabulated cells and is not meant to be: it runs them under
the class-advisory envelope, because the driver deliberately withholds cross-precision fits (the
precision-mismatch note above, which also bounds why the mismatch could not have moved those
picks). Restoring the mismatch is not on offer — a mode that fed an f16 fit to an f32 compile
would reproduce a defect rather than a result.

Runs behind this report: per box, 8 tuned cells (A, the two B arms of the pruning A/B, five
budget-5 chains) — counted in the implementation's unit that is **41 `Autotune.tune` searches
each** (every cell runs both placement arms, and each of the five chains measures its five
flips as full searches) — plus the untuned compiles (7 per box; an untabulated supplemental hip f32 battery — see the untuned section — adds 4 more there) and the fit; serial per box, all three boxes
in parallel, ~8 min (cuda) to ~25 min (metal) wall each. The checked-in driver additionally
pins every cell's treatment explicitly — the tuned cells enable the search and zero the flip
budget where it is not the treatment, the control arms disable their gates, and the gpt cells pin
`BENCH_TUNE=0` — so an ambient config cannot contaminate the matrix; the original runs predate
those pins but were executed with all gates at their defaults (off), so the driver reproduces
exactly what is tabulated above — except for the two documented deviations, both deliberate and
both bounded where they are described: the f16 boxes' D cells (the `f16:f32` argument) and the
class-advisory envelope those cells now run under.

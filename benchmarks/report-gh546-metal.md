# gh-ocannl-546 on Metal: the placement A/B is right, and it is not where tensorization is decided

> **Fixture provenance: unverifiable** (gh-ocannl-919). Original measured files for this historical Metal report
> were not recovered. Later `content-v1` M4 baselines do not certify these numbers.
> See the [historical provenance audit](README.md#historical-metal-fixture-provenance).

Measurement-only report. The subject is the cell [example-report.md](example-report.md) singled out:
`mlp_small` / metal / tuned / f16, where arm B crowned `F_sketch[mma-gpu 16x32x32 ep]` at 1.6484 ms
and arm A won the placement A/B at 0.9915 ms, so no Metal shipping artifact contained a `Tensorize`.

The issue asks three things: whether arm B is genuinely slower or the A/B disadvantages it, whether
the A/B should be precision-aware, and whether "crowned in an arm that did not ship" deserves a
report-level signal. All three are answered below from repeated searches of that exact cell.

Two of the answers correct the premise:

- **The A/B is doing its job.** Arm B is slower end to end by 57% (f16) and 95% (f32) — a margin
  4–13x the per-arm timing spread. Nothing about the comparison is unfair to B.
- **The one arm a `Tensorize` can win at f16 is the only arm where one *exists*.** Arm A seeds
  **zero** tensorized candidates at f16, deterministically, in every run. This is a seeding gate, not
  a ranking outcome, and it is fixable at arm A's cost.
- And the crowning that started the issue is not a stable property: across five identical searches
  arm B crowns a tensorized candidate 3 times out of 5, and at f32 **arm A — the arm that ships —
  crowns one 2 times out of 5**. So a tensorized artifact does ship on Metal on this machine; the
  original "shipped never" was one draw of a near-coin-flip.

Hardware: Apple M4 Max (40-core GPU), macOS 26.5.2. Tree: staging master `478e6773` plus this
issue's branch. Every number below comes from `bench_mlp` on `fixtures/mlp_small.safetensors`
(2→64→64→2 MLP, batch 64, 32 batches), `BENCH_TUNE=1`, `OCANNL_AUTOTUNE_LOG=true`, one process per
run, **cold autotune cache per run** (a warm cache replays instead of searching), five runs per
configuration.

## Headline

| | f32 | f16 | f16, twins materialized |
|---|---|---|---|
| arm A best (5 runs) | 0.637–0.721 ms | 1.042–1.089 ms | 1.069–1.105 ms |
| arm B best (5 runs) | 1.292–1.389 ms | 1.590–1.739 ms | 1.509–1.665 ms |
| arm B / arm A, means | **1.95x** | **1.57x** | 1.46x |
| kernels per step, arm A | 12–14 | 14–16 | 15–17 |
| kernels per step, arm B | 26–28 | 28–30 | 28–30 |
| untuned default, arm A | 0.779–0.874 ms | 1.073–1.628 ms | 1.232–1.493 ms |
| untuned default, arm B | 1.353–1.570 ms | 1.691–2.347 ms | 1.572–1.958 ms |
| mma candidates seeded → timed, arm A | 19 → 14 | **0 → 0** | **19 → 14** |
| mma candidates seeded → timed, arm B | 29 → 17 | 29 → 17 | 29 → 17 |
| arm A crowns a `Tensorize` | **2 of 5 runs** | impossible (none seeded) | 0 of 5 runs |
| arm B crowns a `Tensorize` | 2 of 5 runs | 3 of 5 runs | 1 of 5 runs |

Arm A wins the A/B in 15 of 15 runs.

## 1. Why arm A wins: materialize-all is a whole-graph decision, and it is priced as one

Arm B is not slower because a tensorized candidate was measured under bad conditions. It is slower
before any candidate is scheduled at all. The **untuned default pipeline** — the same program, only
the placements differ — is 1.35–1.57 ms in arm B against 0.78–0.87 ms in arm A at f32, and the
kernel count of every timed candidate doubles (12–14 → 26–28). Materializing every embedded node
turns intermediates that were recomputed inside one kernel into device-resident buffers with their
own producer kernels, and at this workload's size (~1.8 MFLOP per step) launch and traffic are the
whole cost.

The best tensorized candidate recovers part of arm B's self-inflicted overhead and never approaches
arm A. One f16 search, one process, only the schedule differing between lines:

```
arm A  F_preset[bs=cfg cfg-thresh]:      1.1763 ms   (14 kernels)   <- untuned default of this arm
arm A  F_sketch[gpu 16x16x8/2x2]:        1.0470 ms   (14 kernels)   <- crowned, ships
arm B  F_preset[bs=cfg cfg-thresh]:      1.9058 ms   (28 kernels)   <- untuned default of this arm
arm B  F_sketch[mma-gpu 16x32x32]:       2.0580 ms   (28 kernels)
arm B  F_sketch[gpu 16x16x8/2x2]:        1.5565 ms   (28 kernels)   <- crowned, discarded
```

So: **arm B's artifact is genuinely slower end to end**, and the A/B — both arms timed on the same
device, min-of-N, same repeats — is comparing them fairly. The margin (57% at f16, 95% at f32)
dwarfs the spread within either arm.

### The comparison *is* asymmetric, but not in timing

What differs between the arms is not only how candidates rank. It is which candidates exist:

| | arm A | arm B |
|---|---|---|
| f32 | 19 seeded, 14 timed | 29 seeded, 17 timed |
| f16 | **0 seeded** | 29 seeded, 17 timed |

Zero, in all five runs — the seeding is deterministic given the code and backend, so this is not
sampling. The mechanism is `Autotune.mma_tile_for_precisions`: a tensorized candidate is seeded only
when the site's two operand precisions and its destination precision resolve to a tile the backend
advertises, and Metal's `simdgroup_matrix` table is *uniform* (f32×f32→f32, f16×f16→f16,
bf16×bf16→bf16 — there is no mixed-precision multiply-accumulate). Under the mixed-precision recipe
the reduced-precision cast twins of the weights are virtual in the default-placement arm, so the
matmul site reads the **f32 masters** into an **f16 destination**: a mixed triple, no tile matches,
no seed. Materialize-all makes the twins real f16 nodes, the triple becomes uniform, and 29 seeds
fire.

That is the honest answer to the issue's first question: the tensorized candidate is not losing the
placement A/B. At f16 it is only *expressible* in the arm that pays a whole-graph materialization
tax, and the tax is bigger than the prize.

## 2. Should the A/B be precision-aware? No — the coupling is one level down

The premise to check is whether the arms' relative merit differs by precision. On this cell it does
not: arm A wins at f32 (0.69 vs 1.35 ms, mean) and at f16 (1.07 vs 1.67 ms), by comparable margins,
in every run. Both arms are already searched under the precision the cell actually runs, which is
the only precision-awareness a timing A/B can have; a precision-conditioned *choice of arm* would be
a rule overriding measurements that already point the same way.

What is genuinely precision-dependent is seeding reachability. The productive change is therefore
not a precision-aware arm selection but a **targeted placement**: materialize the handful of nodes a
tensorized candidate needs, instead of the whole graph. The knob exists
(`Mixed_prec.Twin_materialized`, exposed by the runner as `BENCH_TWIN_PLACEMENT=materialized`), and
materializing exactly the three weight cast twins does the job:

| f16 arm A | default twins | twins materialized |
|---|---|---|
| mma seeded → timed | 0 → 0 | **19 → 14** |
| best (5 runs) | 1.042–1.089 ms | 1.069–1.105 ms |
| best tensorized candidate | none | 1.096–1.207 ms |
| crowned | never tensorized (4 distinct families) | never tensorized |

Three small casts of weight matrices (2×64, 64×64, 64×2) buy the entire tensorized candidate family
in the arm that ships, at a cost inside the run-to-run spread. The tensorized candidates then lose
*on the merits* — but by 0.5%, 0.7% and 1.2% in three of the five runs, against 40%-class margins
when the only way to reach them was arm B. That converts an unreachability problem into a ranking
problem, which is the tractable kind.

Not proposed as a default here: on this cell it does not make the shipping artifact faster, and
changing a placement default on a no-win measurement would be speculative. What it establishes is
where the fix belongs — a placement targeted at the site, not an arm selected by precision.

## 3. The report-level signal: yes, and as a margin, not a flag

The measurement that forces this is the reproducibility of the crown. Five identical searches, same
binary, same machine, cold cache each time:

| f16 | crowned in arm A | crowned in arm B |
|---|---|---|
| run 1 | `F_sketch[gpu 16x16x8/4x4]` 1.0571 | **`F_sketch[mma-gpu 16x32x0]` 1.5904** |
| run 2 | `F_split[n32_relu.grad …]` 1.0736 | `F_sketch[gpu 64x64x8/4x4]` 1.7394 |
| run 3 | `F_preset[bs=cfg cfg-thresh]` 1.0789 | **`F_sketch[mma-gpu 32x32x32]` 1.6688** |
| run 4 | `F_preset[bs=cfg priv cfg-thresh]` 1.0893 | **`F_sketch[mma-gpu 32x32x16]` 1.7012** |
| run 5 | `F_sketch[gpu 16x16x8/4x4]` 1.0425 | `F_sketch[gpu 32x32x16/2x2]` 1.6721 |

Four different families crown arm A across five runs whose best times span 4.5%. The arm's *time* is
stable; the arm's *winner* is a lottery among candidates separated by less than the noise. The same
holds at f32 in the arm that ships, where the best tensorized and best untensorized candidates are
1.1%, 2.5%, 2.7%, 1.7% apart in four of five runs (and 19% in the fifth) — and the tensorized one
wins twice:

| f32 arm A | crowned | best tensorized | best untensorized |
|---|---|---|---|
| run 1 | `F_preset[bs=cfg priv cfg-thresh]` | 0.7294 | **0.7214** |
| run 2 | `F_sketch[gpu 16x16x8/2x2]` | 0.7222 | **0.7045** |
| run 3 | **`F_sketch[mma-gpu 16x32x0]`** | **0.7003** | 0.7193 |
| run 4 | `F_sketch[gpu 32x32x16/2x2]` | 0.7610 | **0.6370** |
| run 5 | **`F_sketch[mma-gpu 16x32x0 ep]`** | **0.6947** | 0.7068 |

So "a `Tensorize` is crowned on Metal exactly once in 99 cells, and it does not ship" reads a single
draw as a property. Two runs of this one cell ship a tensorized artifact. A report that records only
which candidate won, once, over-reads noise in both directions.

The signal implemented for this issue is therefore per-arm and quantitative, in
`Autotune.report`: `best_label` (which candidate won), `best_tensorized` (read off the winner's
schedule, not off its label's promise), the winner's `Tile_mma` statement count with how many
rendered as the lane-0 scalar fallback (the gh-ocannl-545 distinction, applied to the artifact
rather than to the candidate pool), and `mma_best_ms` — the best *timed* tensorized candidate, whose
margin against `best_ms` is what separates "uncompetitive" from "lost inside the noise".
`Train.tune_placements` states the cross-arm conclusion on the same trace:

```
tune_placements: winner: arm A (A 1.1716 ms vs B 1.7571 ms)
tune_placements: NOTE arm B crowned a tensorized candidate (F_sketch[mma-gpu 32x32x32 ep] at
  1.7571 ms) and did NOT ship: arm A wins the placement A/B at 1.1716 ms (no tensorized candidate
  was timed in the shipping arm)
```

and a tuned benchmark cell now carries both arms in its result line, so the sweep's `results.jsonl`
holds the evidence instead of a discarded stderr stream:

```json
"tune":{"shipped":"A","arms":[
  {"arm":"A","best_ms":1.17158,"best_label":"F_sketch[gpu 32x32x16/2x2]","tensorized":false,
   "mma_scalar_fallbacks":0,"mma_seeded":0,"mma_timed":0,"mma_best_ms":null},
  {"arm":"B","best_ms":1.75708,"best_label":"F_sketch[mma-gpu 32x32x32 ep]","tensorized":true,
   "mma_scalar_fallbacks":0,"mma_seeded":29,"mma_timed":17,"mma_best_ms":1.75708}]}
```

The `mma_seeded: 0` on arm A is the finding of section 1 in machine-readable form: reported once per
arm, a sweep can tell "tensorization lost" from "tensorization was never proposed" without anyone
grepping a log.

## Measurement hygiene and what this leg does not establish

- Every configuration is five cold-cache runs in one session on an otherwise idle machine; single
  numbers are quoted only from a named run, ranges otherwise. No cross-session comparison is made.
- The workload is deliberately the smallest one in the suite, because it is the cell the issue names.
  At ~1.8 MFLOP/step it is launch-bound, which is exactly why the candidate spread is wide and the
  arm gap is not. Nothing here generalizes to the GEMM-dominated cells; `mlp_wide` and `gpt2_mini`
  would need their own leg.
- The `example-report.md` sweep ran on this same machine at `e687da82`. Its arm B figure (1.6484 ms)
  falls inside this leg's arm B range; its arm A figure (0.9915 ms) sits ~5% below this leg's arm A
  minimum. Different session, different commit, so the comparison is recorded and not leaned on —
  which is itself the section-3 point: at this scale a single number carries a session's worth of
  drift, while the arm gap (57%) survives both.
- Not established: whether targeted materialization pays on a larger workload. On `mlp_small` it is
  neutral-to-slightly-negative on the shipping artifact while unlocking the candidate family — the
  interesting test is a cell where the tensorized candidate has enough work to win by more than the
  materialization costs (`mlp_wide`, `gpt2_mini_train`). That is gh-ocannl-558, which also carries
  the mechanical obstacle: by the time site detection runs, the twin whose placement needs changing
  has already been inlined out of the program.

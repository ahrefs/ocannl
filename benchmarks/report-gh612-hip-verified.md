# gh-ocannl-612 on HIP, verified: every ratio on an artifact that shipped and was executed

> **2026-10-30 (gh-ocannl-1357): measured on the replaced raw-v1 files, not comparable byte-for-byte
> with later content-v1 reports.** Its `gpt2_mini` is minix's 13,871,360-byte raw-v1 file (`10839bd1…`; md5 `5b3dfff8…` below). The files on those boxes have since been replaced by the
> m4-max content-v1 bytes, which `fixtures/DIGESTS.txt` now records for rog-nv and minix.

Measurement report, and the closing half of [`report-gh612-hip.md`](report-gh612-hip.md). That
report measured [#573](https://github.com/ahrefs/ocannl/issues/573) and
[#574](https://github.com/ahrefs/ocannl/issues/574) on the Radeon 8060S and stated one limitation in
its verdict: every headline ratio was computed on the **default-placement arm A**, while in three of
its four cells the search shipped **arm B** — so those arm A routines were compiled, dispatched and
timed on the real lineage but never executed against a reference. `AGENTS.md` requires an executed
output check for passes that change cell values, and nothing in the search supplies one. Closing it
needed a way to ship a chosen arm; there was none, so it was filed as
[#638](https://github.com/ahrefs/ocannl/issues/638) rather than bodged.

#638 landed (config `tune_ship_arm`), and this is the session it exists for: **the same cells, the
same box, the same fixture, with arm A forced in every one of them.** 18 cold cells, each searched,
shipped, executed, replayed in a fresh process and output-verified.

**Verdict: the limitation is closed and every conclusion of the earlier report survives — the
coverage table is now `yes` in every row, the acceptance fingerprints reproduce to within 0.5%, and
the parity gate tightens from 5 loss sequences at 14 ulp to 3 sequences at 2 ulp. Two claims the
earlier report could not make are now available, both because the arm is held fixed: gh-573's
end-to-end step ratio (1.28x, non-overlapping, where it was 1.12x inside the noise floor and
explicitly not claimed), and the cap-default comparison on a uniform arm A (cap 4 beats the default
cap 8 by 7.1% end to end and 9.4% on the per-kernel instrument, both non-overlapping).**

What forcing does *not* change is as important: the arm A artifacts measured here are the same
artifacts. `base574A` profiles at 32.4 ms over 117 kernels against the earlier report's 32.33 ms over
117; the tied lm_head is 8.034 ms against 8.036; the `cap −1 → cap 8` analytic traffic is 528.2 →
472.2 MB against 528.2 → 472.1; the four newly materialized nodes are the same four by name. Forcing
selects which arm's routine the process keeps — it does not change what either arm compiles.

## Provenance

- Box: AMD Ryzen AI Max+ 395 ("Strix Halo", Zen 5, 16C/32T) with the **Radeon 8060S iGPU — gfx1151,
  RDNA3.5**, under **WSL2**, kernel 6.18.33.2-microsoft-standard-WSL2, ROCm/HIP **7.14.60850**. The
  same box at the same ROCm version as [`report-gh569-hip.md`](report-gh569-hip.md) and
  [`report-gh612-hip.md`](report-gh612-hip.md), so the numbers are directly comparable rather than
  merely analogous. The box was verified quiet before the first cell (the earlier report's hard-won
  rule: a foreign build on this APU inflated identical replays from ~19 ms to 47–64 ms), and the
  session waited for an in-flight `dune runtest` to finish rather than measuring through it.
- Workload: `benchmarks/fixtures/gpt2_mini.safetensors` — 4 layers, d=256, 8 heads, seq 128, vocab
  1024, batch 8, forward-only, 1024 tokens/step, f32. The fixture is gitignored, so the artifact is
  pinned by digest: **md5 `5b3dfff860fc8c54af2a7d440f4cf202`**, 13 871 360 bytes — the same file the
  earlier session measured, symlinked into all three trees, and every cell refuses to run against
  anything else.
- Trees, at `../wt-gh612v-{base,feat,master}` relative to the checkout — which is where the
  Reproduction block creates them and what `gh612v_session.sh` defaults to (`BASE`/`FEAT`/`MASTER`
  override it; this session ran them from `/home/lukstafi/`). Each is the earlier report's commit
  plus a **verbatim backport of the #638 arm selector**, committed so the checkout is clean and
  nameable:

  | tree | base commit | with #638 | gh-574 `arity_cuts` | gh-573 fanin guard |
  |---|---|---|---|---|
  | BASE | `6d14f401` | `ca4db3bf` | absent | absent |
  | FEAT | `76f50dcd` | `e6b7b415` | present | absent |
  | master | `5d0c86d8` | `4d1ebb11` | present | present |

  **Why the backport is not a confound, stated as checkable facts rather than as a claim of intent.**
  The five files it copies (`lib/train.ml` and the four benchmark runners) were **byte-identical** in
  all three trees and on master before it was applied, so the same patch applies verbatim to each; it
  touches the shipping decision and the shipped-arm attribution, and no lowering, scheduling or
  codegen path. `arrayjit/lib/utils.ml` is patched rather than copied — its base content legitimately
  differs between these commits — and the patch only registers and classifies the key.

  What "verbatim" means here is checked before each cell rather than argued afterwards, in two
  steps because either alone is insufficient. First the backport commit's **whole changed-path set**
  must equal the backport's own seven paths — a commit that also touched, say,
  `arrayjit/lib/schedule.ml` would satisfy any number of per-file digests while changing what the
  arms compile. Then, within that set, the five copied files must match these digests exactly:

  | file | sha256 (all three trees) |
  |---|---|
  | `lib/train.ml` | `5d98ac818c547662…52195fcd` |
  | `benchmarks/runners/ocannl/bench_gpt.ml` | `de3cb8b4d56cac22…b548a1e` |
  | `benchmarks/runners/ocannl/bench_harness.ml` | `f51ef550c7146425…aaa7d2c1` |
  | `benchmarks/runners/ocannl/bench_conv.ml` | `77dea492448dd317…53dfe0f3` |
  | `benchmarks/runners/ocannl/bench_mlp.ml` | `328c1dcc51a016ee…eb2df44b` |

  **Which revision of the selector, precisely.** That `lib/train.ml` is the selector as it first
  landed (staging commit `ffc428d2`), *before* two review-round fixes made on the PR branch
  afterwards. Both fixes are on failure paths this session never took: releasing the shipped context
  when a caller's `on_ship` callback raises (this session's callback is the benchmark harness's
  recorder, which did not raise), and which arm's failure propagates when *both* arms fail (no cell
  here had a failing arm — all 18 shipped a successful arm A). Neither changes what an arm compiles
  or which arm ships on a successful run, which is why the measurements stand as taken; stated
  because "verbatim backport of the selector" would otherwise read as "of the selector as merged".

  The measured artifacts corroborate the whole argument: the per-kernel profiles, kernel counts,
  crowned-candidate traffic and named node sets reproduce the pre-#638 session's (Part 6).
- Cache discipline: one **cold** search into a fresh `--ocannl_autotune_cache_dir` per cell per rep,
  never shared — the [`report-gh481-cuda.md`](report-gh481-cuda.md) rule, since a warm cache makes an
  A/B vacuous by replaying the other arm's crowned schedule.
- Arm order balanced across reps: gh-574 ran BASE→FEAT, FEAT→BASE, BASE→FEAT; gh-573 cap8→cap−1,
  cap−1→cap8, cap8→cap−1; the cap block cap8→cap4, cap4→cap8, cap8→cap4 as its own r4–r6 block. All
  work under `taskset -c 0-15`.
- **Every quoted step p50 is a pass-2 number**, per [`benchmarks/README.md`](README.md): pass 1
  (`search`) is the cold search, then a **fresh process** (`replay`) replays the cached winner and
  provides the timings. All 18 cells have an accepted pass-2 replay, and the gate covers those
  replays as well as the searches.
- Driver: [`gh612_cells.sh`](gh612_cells.sh) unchanged, driven by
  [`gh612v_session.sh`](gh612v_session.sh), which pins **what certification means** rather than
  inheriting it — the workload and its digest, the parity threshold (64 ulp) and the expected sample
  count are set by the wrapper, so an exported `PARITY_MAX_ULP=1e12` or an emptied `FIXTURE_MD5`
  cannot weaken the gate while every other check stays green (only `OUT_ROOT` and the tree paths,
  which say *where* to run, are inherited) — and which validates each tree before dispatching a cell —
  pinned base commit, at most the backport commit on top, clean worktree, the selector present in
  `lib/train.ml`, and a `bench_gpt.exe` not older than the sources the backport touches (a
  half-backported tree, sources patched and binary stale, is the quiet failure: it would ignore the
  flag while every downstream gate still passed) — and which declares each cell's treatment **once**
  so that `search`, `snap`, `replay` and `profile` cannot drift apart on the arm flag. That is not
  tidiness: a `replay` that omitted `--ocannl_tune_ship_arm=a` would still replay from the cache,
  still emit a `step_ms` record and still pass the driver's two-cache-hit gate — while shipping arm
  B, i.e. producing an arm B timing under an arm A label, which is the exact confusion this session
  exists to remove.
- One incidental observation, recorded because it is large and not understood: cold searches ran at
  **56–82 s** here against **331–536 s** in the earlier session on the same box, same trees, same
  workload (one earlier cell did run at 70 s). Nothing in this report depends on compile time; a
  warm ROCm/comgr compilation cache is the obvious suspect, and it is left as an observation rather
  than a finding.

## Part 1 — what forcing changed, per cell

`tune_ship_arm=a` was set in all 18 cells, including the ones that would have shipped arm A anyway:
the point is that the treatment is uniform, so no cross-cell comparison is conditioned on which arm
each individual search happened to prefer. The override is loud by construction — every cell's
stderr carries both announcements, at resolution and at the decision.

| cell | tree / flag | reps | arm A best (ms) | arm B best (ms) | would have shipped | shipped |
|---|---|---:|---|---|---|---|
| `base574A` | BASE | 3 | 28.31–28.41 | 25.53–27.38 | **B B B** | A A A |
| `feat574A` | FEAT | 3 | 23.88–25.01 | 19.96–21.27 | **B B B** | A A A |
| `capoffA` | master, cap −1 | 3 | 23.70–24.45 | 20.00–21.00 | **B B B** | A A A |
| `cap8A` | master, default | 6 | 19.37–20.41 | 20.20–22.67 | A ×6 | A ×6 |
| `cap4A` | master, cap 4 | 3 | 18.05–20.06 | 20.79–21.83 | A A A | A A A |

Nine of eighteen cells shipped an arm the timing comparison would have discarded. In the earlier
session those nine cells' arm A routines are precisely the ones that were profiled and never run.

## Part 2 — the correctness gate, which is the point of the session

`gh612v_session.sh gate` over exactly this session's cells:

```
36 records (18 search + 18 pass-2 replay), 3 distinct loss sequences
per-step agreement across ALL runs:
  step 0: 1 distinct, span 0.000e+00 =   0.0 f32 ulp (rel 0.00e+00)
  step 1: 1 distinct, span 0.000e+00 =   0.0 f32 ulp (rel 0.00e+00)
  step 2: 2 distinct, span 4.800e-07 =   1.0 f32 ulp (rel 6.74e-08)
  step 3: 2 distinct, span 4.700e-07 =   1.0 f32 ulp (rel 6.62e-08)
  step 4: 1 distinct, span 0.000e+00 =   0.0 f32 ulp (rel 0.00e+00)
  step 5: 2 distinct, span 9.600e-07 =   2.0 f32 ulp (rel 1.36e-07)
  step 6: 1 distinct, span 0.000e+00 =   0.0 f32 ulp (rel 0.00e+00)
  step 7: 2 distinct, span 4.700e-07 =   1.0 f32 ulp (rel 6.63e-08)
WORST: 2 f32 ulp (threshold 64). NOT bit-identity -- state it as
       agreement to within that many ulp.
parity gate PASSED
all 18 cells have an accepted pass-2 replay
```

The gate's **first** check is provenance: `gate` is documented as independently runnable, and an
artifact directory does not otherwise record what produced it — a stale `OUT_ROOT` populated from
another checkout would pass every gate below under these labels, because all of them read artifacts
and none of them knows their origin. So a search now writes the validated role, tree, HEAD and
backport digests into the cell — and so do `snap`, `profile` and `replay`, each regenerating
claim-bearing artifacts long after the search (the emitted arm A source, the per-kernel CSVs, and
the pass-2 timings and losses every table here quotes), which a search manifest would go on matching
— and the gate requires each manifest to match the tree it validates now. **These 18 cells' manifests were backfilled**, honestly and visibly: the check was added during
review, after the session had run, so each manifest carries a `.backfilled` note and the gate counts
them separately (`18 backfilled`). Two artifacts are the exception, recorded rather than attested
because they were regenerated through the wrapper after the check existed: `cap4A/r4`'s snapshot and
profile (Part 5's instrument, re-run for exactly this reason) and `cap8A/r1`'s pass-2 replay, which a
verification probe of the driver's own manifest-retraction consumed and which was therefore re-run —
moving that cell's p50 from 18.656 to 18.704 ms, inside the 2.6% reproducibility the instrument table
quotes, and changing no ratio. What they attest rather than record is that the cells came from
this session's driver against these roles' trees, which have validated unchanged before, during and
after the run; a future session records them at search time and prints no such note.

Then the session's premise, which neither of the driver's own gates can see:
`parity` compares loss vectors and `replays` checks timings and cache hits, and both are satisfied by
a perfectly good arm B session — so a stale binary in one worktree or a dropped flag in one
subcommand would produce a green gate over exactly the artifacts this session exists to replace. So
each expected record is parsed for the shipped arm directly, over both passes:

```
arm-A premise: 36 records (search + pass-2) all shipped arm A
```

The coverage table the earlier report had to print with three `no` rows:

| artifact | shipped? | covered by the loss gate |
|---|---|---|
| `cap8A` arm A — the denominator and both fingerprints | yes (6/6) | **yes** |
| `base574A` / `feat574A` arm A — the gh-574 ratio | yes (3/3 each) | **yes** |
| `capoffA` arm A — the gh-573 ratio | yes (3/3) | **yes** |
| `cap4A` arm A — the cap-default claim | yes (3/3) | **yes** |

Two things this gate does and does not show, stated as the earlier report learned to state them.
It shows that 18 arm A artifacts — across three trees, three caps and **fourteen distinct crowned schedules**
— compute the same function to within **2 f32 ulp**, in both the searching process and a fresh
replaying one. It does **not** show bitwise agreement: there are 3 distinct full-precision sequences,
which is the expected signature of reassociating one f32 reduction across different schedules.

**And a within-session gate cannot cross-check arm A against arm B, because every cell here is arm
A.** That comparison is available anyway, from the earlier session's artifacts: its cells shipped
arm B where these ship arm A, on the same commits and the same fixture. Across both sessions —
**forced arm A here against independently executed arm B there** — the losses agree to within
**14 f32 ulp**, the same bound the earlier session reported internally. So the two arms are now
checked against each other by execution, which is what "the discarded arm was never run" used to
prevent.

## Part 3 — the gh-574 arm: BASE `6d14f401` vs FEAT `76f50dcd`

Not config-gated, so two built trees; arm A forced in both.

| | BASE | FEAT | |
|---|---:|---:|---|
| **arm A per-kernel profile** (3 harness runs) | **32.42 ms** / 117 kernels | **25.97 ms** / 135 kernels | **1.25x** |
| **step p50, 3 reps** (pass 2, order-balanced) | 28.333 / 28.366 / 28.593 | 23.769 / 24.050 / 24.652 | **1.18x**, non-overlapping |
| untuned-default pipeline, 3 reps | 64.79 / 65.56 / 65.66 | 65.29 / 65.56 / 65.75 | 1.00x |
| crowned arm A analytic traffic | 441.5 MB | 528.3 MB | — |

The untuned row is the same internal control the earlier report used, and it comes out the same way:
`arity_cuts` is a candidate-*generation* mode and must not move the non-searched pipeline, and it
does not — the two medians are 65.558 and 65.558 ms.

**Acceptance fingerprints** (`gh612v_session.sh structure`, per-kernel medians over 3 harness runs):

| | BASE | FEAT |
|---|---:|---:|
| lm_head / CE tail | 8.162 ms over 5 kernels | 0.510 ms over 6 |
| — of which the fused `logits, max_logits, n794_layer_norm, wte` kernel | **8.034 ms** | cut apart |
| lm_head / CE **whole chain** | 8.134 ms over 4 | 0.482 ms over 5 — **16.9x** |
| QKᵀ **whole chain** | 3.654 ms over 8 | 2.417 ms over 16 — **1.51x** |
| the five kernels (4× FFN GEMM1 + lm_head) | 8.986 ms = 27.7% | 1.315 ms = 5.1% |

The single largest kernel on this device measures **8.034 ms** here against the earlier report's
8.036 ms — the artifact is the same one, now executed. `diff base574A 1 feat574A 1`: **14 exclusive
signatures (14.692 ms) on BASE against 32 (7.912 ms) on FEAT**, zero differing multiplicities — the
same 14-vs-32 the earlier report pinned. The four QKᵀ sites are freed here as they were on CUDA, and
measured over the whole chain rather than the fragment that keeps the name they contribute the
smaller share, as the earlier report found.

## Part 4 — the gh-573 arm: `virtualize_max_inline_fanin` −1 vs 8

One tree, a config flip, arm A forced on both sides.

| | cap −1 (before) | cap 8 (after) | |
|---|---:|---:|---|
| **arm A per-kernel profile** | **24.93 ms** / 135 kernels | **18.51 ms** / 136 kernels | **1.35x** |
| **step p50, 3 reps** (pass 2, order-balanced) | 23.009 / 23.782 / 24.256 | 18.495 / 18.579 / 18.704 | **1.28x**, non-overlapping |
| untuned-default pipeline, 3 reps | 65.52 / 65.70 / 65.98 | 60.94 / 61.25 / 61.68 | **1.072x** |
| layernorm / elementwise bucket | 3.586 ms (14.4%) | **0.800 ms (4.3%)** | 4.48x |
| LayerNorm sites (9) | 3.586 ms | 0.800 ms | — |
| crowned arm A analytic traffic | 528.2 MB | 472.2 MB | **−10.6%** |

**The end-to-end row is the new claim.** The earlier report measured 1.12x here with overlapping
ranges and explicitly declined to claim it, for an identified reason: without the guard the search
shipped arm B — materialize-all, the crude form of the same transform — so the comparison was
guard-vs-materialize-all rather than guard-vs-no-guard. With the arm held fixed the ranges do not
overlap at all (worst cap 8 rep 18.704 < best cap −1 rep 23.009) and the ratio is **1.28x**, in
agreement with the per-kernel instrument's 1.35x. That is the same mechanism the earlier report
described, now measured instead of inferred.

`diff capoffA 1 cap8A 1`: **16 exclusive signatures (8.516 ms) at cap −1 against 17 (1.711 ms) at
cap 8** — the earlier report's 16-vs-17 and 8.495 → 1.702 ms, reproduced to 0.5% — and the newly
materialized nodes are **+4: `centered`, `n446`, `n792`, `x1`**, the same four by name.

**The triangle, and its truncation**, still visible in the emitted parameter lists: at cap −1 each
LayerNorm site carries an accumulated `l*_ffn_b2` prefix that ramps `0,0,1,1,2,2,3,3,4` with depth,
and the per-site cost ramps with it (0.036 → 1.136 ms across the nine sites on BASE). At cap 8 the
prefix resets and the whole bucket is 0.800 ms.

**Negative control.** `feat574A` and `capoffA` are the same configuration reached two ways
(`arity_cuts` without the guard, from a tree that lacks it and from a tree that disables it).
`diff feat574A 1 capoffA 1` reports **IDENTICAL kernel sets** — zero exclusive signatures on both
sides, zero differing multiplicities, 155 materialized nodes each — and their pass-2 step medians
agree to **1.1%** (24.050 vs 23.782 ms). The earlier report had the structural half of this control;
the timing half is new, and it is only available because both sides now ship the same arm.

## Part 5 — the cap default: is 8 the right trade on gfx1151?

**No, and the answer no longer depends on which arm each cap shipped.** One order-balanced block
(r4–r6, cap8→cap4, cap4→cap8, cap8→cap4), arm A forced throughout.

| | cap 8 (default) | cap 4 | |
|---|---:|---:|---|
| **arm A per-kernel profile** | 18.51 ms / 136 kernels | **16.76 ms / 137 kernels** | **1.10x** (−9.4%) |
| **step p50** (pass 2, the balanced block) | 18.577 / 18.636 / 18.755 | 17.204 / 17.322 / 17.393 | **7.1%**, non-overlapping |
| untuned-default pipeline | 60.91 / 60.91 / 61.05 | 60.42 / 60.46 / 60.47 | 0.8% |
| layernorm / elementwise | 0.800 ms | **0.313 ms** | 2.6x |
| crowned arm A analytic traffic | 472.2 MB | 454.3 MB | −3.8% |

`diff cap8A 1 cap4A 4`: 21 exclusive signatures (3.100 ms) against 22 (1.928 ms); materialized nodes
159 → 164, with +7 at cap 4 and +2 at cap 8 (the two are not nested — lowering a cap changes which
consumers reset, not only how many nodes are forced).

The earlier report measured 5.7% for the same comparison and declined to propose a default change;
this session measures **7.1%** on a uniform arm-A basis with a second, deterministic instrument
agreeing at 9.4%. **The recommendation is unchanged: do not change the default on this evidence.**
One fixture, one depth, one device — `gpt2_mini` at 4 layers is exactly the workload whose residual
fan-in is small enough for a tighter cap to be free, and the cap is a global policy prior. What has
changed is the quality of the evidence, not its breadth: the claim is now about two artifacts that
both shipped and were both executed, rather than about two searches that happened to prefer
different arms.

## Part 6 — what reproduces from the pre-#638 session

Every load-bearing number, on artifacts that are now output-verified:

| quantity | `report-gh612-hip.md` | here | |
|---|---:|---:|---|
| BASE arm A profile | 32.33 ms / 117 kernels | 32.42 ms / 117 | +0.3% |
| FEAT arm A profile | 24.79 ms / 135 | 25.97 ms / 135 | +4.8% |
| cap −1 arm A profile | 24.82 ms / 135 | 24.93 ms / 135 | +0.4% |
| cap 8 arm A profile | 18.88 ms / 136 | 18.51 ms / 136 | −2.0% |
| cap 4 arm A kernels | 137 | 137 | = |
| tied lm_head (BASE) | 8.036 ms | 8.034 ms | −0.02% |
| gh-573 triangle | 8.495 → 1.702 ms | 8.516 → 1.711 ms | +0.3% |
| gh-574 exclusive signatures | 14 vs 32 | 14 vs 32 | = |
| gh-573 exclusive signatures | 16 vs 17 | 16 vs 17 | = |
| gh-573 newly materialized nodes | 4 (`centered`, `n446`, `n792`, `x1`) | the same 4 | = |
| cap −1 → cap 8 traffic | 528.2 → 472.1 MB | 528.2 → 472.2 MB | = |
| untuned-default, cap −1 / cap 8 | 65.48–65.63 / 60.86–61.31 | 65.52–65.98 / 60.91–61.68 | ≤0.6% |

The ratios move slightly because the crowned schedules differ run to run (gh-481's family lottery,
which is why the per-kernel instrument and three reps exist): gh-574 is 1.25x here against 1.30x,
gh-573 1.35x against 1.31x, composed **1.75x against 1.71x**. The FEAT cell is the one outlier
above 1%, and it is on the noisier side of the same lottery — its arm A best_ms spans 23.88–25.01
across three reps.

## Limitations

- **One device, one workload, one precision**, as before. Nothing here extrapolates to another
  fixture depth, another backend, or reduced precision.
- **The loss gate is a same-function check to 2 ulp, not a proof of correctness.** It compares
  executed outputs across schedules and arms; it cannot see an error that every schedule of this
  graph makes identically. What it now covers, and did not before, is every artifact this report
  quotes a number for.
- **`cap4A` was profiled at r4 only** (one cell, three harness runs), like the earlier report's
  single-cell caps. Its step timings are the full balanced block. That profile was re-run through
  the session wrapper during review — an earlier revision produced it by calling `gh612_cells.sh`
  directly, which skipped tree validation and left it without provenance. The re-run emitted the
  *same* arm A source (identical snapshot digest) and moved the per-kernel total by 0.2% (16.80 →
  16.76 ms), which is this instrument's own reproducibility; the structural results are unchanged
  (21-vs-22 exclusive signatures, 159 → 164 materialized nodes).
- **The backport is pinned by digest and corroborated by reproduction, not proven inert.** The
  digests establish *which* code ran, not that it cannot matter; the argument that it cannot is the
  file-identity one in Provenance, and the evidence for it is Part 6 — fifteen quantities across
  three trees reproduce, including named node sets and exact signature counts.
- The compile-time discrepancy in Provenance is unexplained. It does not enter any claim.

## Reproduction

```bash
# three trees at the earlier report's commits, each with the #638 selector backported verbatim,
# one shared fixture. See Provenance for the resulting commit ids.
cd ocannl-staging
for w in master:5d0c86d8 base:6d14f401 feat:76f50dcd; do
  d=../wt-gh612v-${w%%:*}; want=${w##*:}
  [ -e "$d" ] || git worktree add --detach "$d" "$want" || exit 1
  # (1) the five files, copied verbatim -- these are what the digests in gh612v_session.sh pin
  # ffc428d2 is the selector as it first landed -- the revision these trees carry (see Provenance)
  git -C "$d" checkout ffc428d2 -- lib/train.ml benchmarks/runners/ocannl/bench_harness.ml \
    benchmarks/runners/ocannl/bench_gpt.ml benchmarks/runners/ocannl/bench_conv.ml \
    benchmarks/runners/ocannl/bench_mlp.ml || exit 1
  # (2) utils.ml is PATCHED, not copied: its base content differs between these commits, so the key
  # is registered and classified in place. Both edits are required and the exact text matters --
  # validate_tree pins the resulting digest per role. This recipe was checked against both pins.
  awk '{ print }
       /^      "tune_flip_ordering";$/ && !r { print "      \"tune_ship_arm\";"; r = 1 }
       /^      \[ "tune_inline_flips"; "tune_flip_ordering" \] \);$/ && !c {
         print "    ( Search_shaping,"
         print "      \"it decides which of the two searched placement arms ships, overriding the measured \\"
         print "       comparison rather than changing either arm (gh-ocannl-638)\","
         print "      [ \"tune_ship_arm\" ] );"
         c = 1
       }' "$d/arrayjit/lib/utils.ml" > "$d/utils.patched" \
    && mv "$d/utils.patched" "$d/arrayjit/lib/utils.ml" || exit 1
  # (3) the reference file, which the seven-path check expects the backport to touch
  printf '%s\n' '' \
    '# gh-ocannl-638 (backported onto this measurement tree): which placement arm' \
    '# `Train.tune_placements` ships -- auto (the measured winner), a (default placements),' \
    '# b (materialize-all). Measurement-only; announces itself on stderr.' \
    '#tune_ship_arm=auto' >> "$d/ocannl_config.reference" || exit 1
  mkdir -p "$d/benchmarks/fixtures"
  ln -sf "$PWD/benchmarks/fixtures/gpt2_mini.safetensors" "$d/benchmarks/fixtures/gpt2_mini.safetensors"
  (cd "$d" && git add -A && git commit -m "Backport the gh-ocannl-638 arm selector" && dune build @check bin/ benchmarks/) || exit 1
done
md5sum "$(readlink -f benchmarks/fixtures/gpt2_mini.safetensors)"  # 5b3dfff860fc8c54af2a7d440f4cf202
```

```bash
# the session: each block is independent and resumable; every cell forces arm A.
S=benchmarks/gh612v_session.sh
bash $S trees                     # print the three trees and their commits, measure nothing
bash $S gh573                     # cap8A vs capoffA, 3 reps, order alternated
bash $S gh574                     # base574A vs feat574A, 3 reps, order alternated
bash $S caps                      # cap8A vs cap4A, reps 4-6, order alternated
bash $S replays                   # pass 2: the step timings every table above quotes
bash $S structure                 # snap + profile + finger on r1, then the three claim-bearing diffs
bash $S capprofile                # Part 5's per-kernel profile of cap 4, and the cap-8-vs-cap-4 diff
bash $S gate                      # LAST: provenance, the arm-A premise, parity, the artifact gates
```

`gate` runs last because it *requires* the artifacts the two blocks before it produce: the four
structural cells' and cap-4's snapshots and profiles are what Parts 3–5 quote, so a missing one is a
failure rather than something to skip. Verified by removing `cap4A/r4`'s first CSV, which makes the
gate name it and refuse.

A cell's stderr carries the two `Train.tune_placements:` announcements that make the treatment
auditable after the fact — the resolution line and the decision line, the latter naming both the arm
that shipped and the arm the timings preferred:

```
Train.tune_placements: tune_ship_arm selects arm A (default placements), which will ship whatever
  the timings say. This is a measurement-only setting; a normal run ships the measured winner.
Train.tune_placements: shipping arm A by tune_ship_arm; the measured winner is arm B
  (A 23.6990 ms vs B 20.0014 ms), so the override changed what ships.
```

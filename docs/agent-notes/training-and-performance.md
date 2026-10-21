# Training and performance

Parameter sets, training-loop utilities, calibration, and benchmark methodology.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- `t.params` contains only `Tensor.param`-registered tensors (`{ w }` in `%op`, `TDSL.param`,
  `reshape_param`/`wrap_param`) — a leaf built with `Operation.init ~grad_spec:Require_grad` (the
  deterministic-values test idiom) gets a gradient but does NOT join it; for deterministic real
  params use `TDSL.reshape_param ~l ~i ~o ndarray ()` / `TDSL.param ~values l ()`. `params` is the
  *state* set (what `init_params` initializes and `Persistence` saves — a frozen backbone behind
  `stop_gradient` included), NOT the trained set: the optimizer-side helpers (`Train.sgd_update`,
  `grad_l2_norm`/`clip_by_global_norm`, `grad_checksum`, `zero_params_grads`) operate on
  `Train.trainable_params` — the params whose gradient the backprop writes
  (gh-ocannl-673: a frozen parameter takes no step, in particular no weight decay) — and raise
  `Session_error` when that set is empty rather than silently compiling empty routines
  (gh-ocannl-670: the gh-465 test initially "passed" parameter parity because no optimizer step
  ran on either side). `?params` on each helper overrides the derivation.
- Training-loop utilities (gh-465, `lib/train.ml`): `Lr_schedule` (host floats; feed via
  `scheduled_learning_rate`'s data-backed `host_scalar` — a fetch-defined constant would be
  re-fetched each step, undoing `set_values`), `grad_l2_norm`/`clip_by_global_norm` +
  `sgd_update ~grad_scale` (scale folded into the gradient read; buffers untouched — the norm
  can be recomputed on the host from the buffers to cross-check), `grad_update ~accum_steps` +
  `zero_params_grads` (param zeroing is filtered OUT of the micro-step's `zero_grads` tree by
  matching the params' grad `Fetch`es; intermediate grads must keep their per-micro-step zeroing
  since backprop `=+` relies on it), `Outlier_detector` (z-score vs sliding window; nan during
  warmup, and a constant window gives std 0 → infinite z). Executed parity: `loop_utils.ml`.
- Optimizer state that survives across invocations of a routine must be MATERIALIZED, or lowering
  fails with `Stale optimize_ctx: No computations found for #N: <node>` — the virtualizer treats an
  undetermined read-before-write node as an inlining candidate, and the computation defining it is
  in the previous invocation, not this one. `sgd_one` materializes its `sgd_momentum` buffer for
  exactly this reason (gh-ocannl-772); a new optimizer with its own buffers owes the same. The
  option matrix has executed coverage in `test/operations/sgd_variants.ml`, which compares a
  multi-step trajectory against a host simulation and sends non-default momentum through all four
  construction paths (`sgd_one`, `sgd_update`, `Mixed_prec.scaled_sgd_update`,
  `Mixed_prec.gated_scaled_update`) on `cc`; each wrapper's non-default result must also differ from
  its own zero-momentum run. The higher-level `Parallel.data_parallel` forwarder is exercised in
  `test/training/data_parallel.ml`: two-step `cc` trajectories separately send non-default momentum
  and weight decay through the wrapper, match an independent closed-form host recurrence, and
  differ from the same default trajectory. Reach for that shape rather than a structural check,
  since `~momentum` had been dead on arrival with a green suite behind it.
- `Train.to_routine` returns `Context.t * Context.routine` (gh-ocannl-772), like `Context.compile`
  and `Train.run_once`. `routine.Context.context` is the same value, so both spellings work; prefer
  chaining the returned context.
  Doc-check `note_training_context`: `let compile ctx bindings comp = let open Ocannl.Nn_blocks.DSL_modules in let ctx, routine = Ocannl.Train.to_routine ctx bindings comp in ctx, routine.Context.context`.
- Metal training recipe: `Train.every_non_literal_materialized loss` (kernel fission then cuts
  every cross-nest edge) + `Autotune.tune ~rounds:0 ~timing_ctx:scratch`. `~rounds:0` keeps
  .expected files schedule-invariant (preset seeds preserve reduction order); `?timing_ctx` on a
  scratch lineage is MANDATORY for training comps fed via `set_values` — timing on all-zero
  inputs poisons params through log(0)→NaN, and reinit-after-tune is not airtight (ledger trips,
  Metal device-side races).
- `Autotune.report.candidates_timed` is NOT comparable across a CPU and a GPU backend, so an
  assertion on it recorded from `cc` is a portability trap (gh-ocannl-543 —
  `autotune_fission_sketch` timed 19 candidates on cc and 1 on HIP for the same chain). The GPU
  rule of gh-ocannl-532 refuses every candidate that binds no hardware dimension, and on a
  two-nest chain that is most of the space: the whole-routine presets dedup to the unscheduled
  base, and the beam's `Split`/`Swap`/`Retype-Vectorized` moves off that base cannot introduce a
  `Grid`/`Workgroup` loop (only `Tensorize` and placement retypes can), so the fissioned preset is
  the only thing left to measure. The refusals now carry a typed census key,
  `Not_dispatched_key of origin` (`baseline` | `candidate` | `beam_move`) — read it (or
  `BENCH_TUNE_REPORT=1`) before concluding a GPU search "found nothing".
- The calibration TSV schema (`autotune_calibration_file`) is owned by
  `Ir.Cost_model.Calibration` — writer (`Autotune.emit_calibration`) and reader
  (`tools/fit_envelope.exe`) share `to_line`/`of_line`, so change it in one place only. A row
  names both the tuned computation (the routine name derived from its block comment, as the
  candidate's generated sources are named) and the candidate, and every report quotes them as
  `routine/label (digest)` — a fitted constant has to say which kernel demonstrated it. Evolving
  the schema means keeping the older column counts parseable: rows accumulate in one file across
  builds, so `of_line` still accepts the 11-column rows that predate the routine column (they fit,
  unnamed), while the 9-column rows predating the approx flags cannot prove exactness and the
  fitter skips them with a warning. The
  fitter's constants are the tightest envelope respecting every row a leg can be audited on:
  per-leg max achieved counts/time (per backend), where each leg uses the rows whose counts are
  exact FOR THAT LEG (flops-approx and bytes-approx are independent flags — a multi-read
  footprint doesn't disqualify an exact op count; approx legs and opaque rows are excluded, or
  one mostly-failing-guards candidate's fake throughput would inflate the envelope
  machine-wide), then a uniform "fission slack" on both legs so the aggregate sufficient
  condition (flops/pf + bytes/pb <= t) holds on fully-exact multi-kernel rows — per-leg maxima
  alone are necessary but NOT sufficient there (the bound sums per-kernel max-of-legs).
  Serialized milliseconds are FLOORED at the 6th decimal (round-to-nearest could store a 5 us
  kernel's time high by 1e-4 relative and break file-fit conservatism). Fitted peaks are
  demonstrated floors, not certified maxima; under autotune_keep_fraction < 1 a
  faster-than-observed candidate can be pre-filtered before the agreement check can see it
  (fit_envelope's --margin is the headroom knob). The bound-agreement invariant (gh-ocannl-514
  phase 0) runs on EVERY candidate `Autotune.tune` times whenever envelope constants are
  present — not gated by `autotune_log` or the calibration file — warning unconditionally on
  stderr only for exact-count candidates (approx exceedances log under autotune_log as possible
  over-counting). The class-level `hardware_limits` bandwidth advisories are class CEILINGS for
  exactly this reason (gh-ocannl-578): a machine sustaining more than its backend's advisory
  would otherwise print `BOUND VIOLATION` on every exact streaming row. If such warnings appear
  anyway, that is the invariant working — a new hardware tier has outgrown the ceiling, or a
  configured `model_peak_*` override is stale: refit from calibration data rather than silencing
  the check. The warning prints once per (backend, device, digest) per process —
  a candidate timed again (two tuning arms, a re-tune) warns once — but every timing still writes
  its own calibration row, so the fitter sees all of them.
- The envelope's memory leg is unfittable from matmul-family tuning data alone (gh-ocannl-578):
  those rows are compute-bound — even exactly counted, their bytes/time understates achievable
  bandwidth, and an understated `model_peak_memory_bandwidth` is the unsound direction (a
  bound-driven search over-prunes) — and they usually carry approximate byte counts anyway.
  `Ocannl.Calibrate.stream` (CLI: `tools/calibrate_bandwidth.exe`) is the STREAM-style calibration
  pass whose kernels are bytes-exact by construction: one access site per node and direction, no
  guards, tuned through `Autotune.tune ~rounds:0 ~cache_dir:""` so rows go through the ordinary
  emission path. It verifies after the fact that exact rows actually appeared (a failure means the
  schedule introduced guards — keep `--elems` a power of two so every workgroup size divides the
  extent) and refits the whole file. Cost-model exactness is correspondingly wider (same issue):
  same-direction accesses that are individually exact, unconditionally evaluated (no `Where` arm
  or gated-operand reads — `Cost_model.access_uncertainty`, shared with the floor), and pairwise
  provably disjoint (`Affine.may_touch_same_cell`) sum exactly instead of tripping the
  multi-access union bound, and
  a vectorized run whose minor-axis base spacing is at least the run length
  (`Affine.vec_runs_disjoint`, coefficient-gcd argument) counts exactly — so scheduled streaming
  candidates stay fit-eligible, and the floor agrees with the upper extraction on disjoint unions.
- The cost model keeps TWO scores apart (gh-ocannl-1253). `Cost_model.analyze`'s distinct-cell,
  perfect-cache roofline is the lower bound that `tune`'s bound pruning, the keep-fraction
  pre-filter, the agreement warning and the calibration rows read; it scores an untiled contraction
  exactly at its floor however often blocks re-read the operands, so on it no reuse-tiled sketch
  can ever win. `Cost_model.ranking_seconds` is the untimed choosers' score: global reads
  requested per block (`reuse_traffic`: `access_cells ~fixed` over the non-`Grid` loop box, times
  the blocks), charged in two tiers — at DRAM bandwidth once per-block bytes times resident blocks
  exceed the cache capacity, at the cache bandwidth within it (free re-reads tied the untiled and
  seeded plans at batch 8 on the HIP boxes, where seeded measured faster) — and every memory leg
  at least `analyze`'s distinct bytes at DRAM bandwidth, so it is never below any bound, which
  `test/operations/cost_model_reuse` claims over every leaf of a sketch tree. Never feed the
  ranking score to a bound: a kernel that hits in cache would then trip `BOUND VIOLATION`.
- Exact ranking ties go to fewer tile-MMA operand loads (gh-ocannl-1254), never a priced term.
  The count describes emitted code only through the backend's own arm description, never a
  cost-model copy of it: `mma_operand_loads` (which operands use `ldmatrix`; CUDA's
  `Cuda_mma.operand_loads`, which the census reads too) and `mma_operand_fragments` (the arm's own
  tile, not the descriptor's, whether its loop nest reloads each operand per tile of the other's
  free axis — inline-PTX yes, wmma no — and whether a fragment loads whole, wmma's
  `load_matrix_sync`), both per emission scope; operand layouts go through
  `Low_level.mma_operand_layout`, the renderer's own decline; and `mma_accepts`, the
  backend's own `mma_syntax` hook probed without addresses, decides both whether a statement counts
  at all and in which scope, in the renderer's order: a fragment-marked statement through its
  region (`Low_level.fragment_region`, the renderer's own recognition, finds the backing target), in the
  fragment scope where `mma_fragment_syntax` accepts the target and the update-only call, else as
  statements against the target; an unmarked one per statement. A
  backend that describes no arm (CPU register tiles, Metal, HIP) gets `Undescribed`, code with no
  `Tile_mma` `No_tile_mma` (`Cost_model.load_count`), and a keyless leaf neither wins nor loses a tie: each review round of staging#1216 that restated an
  emission detail in the cost model found the next arm it got wrong. `Schedule_space.search
  ~tie_break` lets a tied keyed leaf displace a keyed best, never the incumbent, and enters a
  subtree bounded at the threshold only under a keyed best. `Autotune.model_family_search` is the
  walk `model_default` runs, exposed so `test/operations/model_load_tiebreak` drives it on cc with a
  CUDA-shaped record and capability; `Autotune.model_explain` lists what the model sees of each
  whole-routine candidate and, since gh-ocannl-1438, of each segment of its per-segment
  substitution (`x_segments`: the preset first, then the leaves labeled as `F_sketch` entries). The swizzled staged fp8 twin beats its plain twin, but at the
  gh-ocannl-1190 shape neither is the argmin: an unstaged tensorized leaf ranks strictly below
  every staged one (the staging copies add ops and the per-block global reads are the same; until
  gh-ocannl-1442 the shared tiles also added distinct-cell bytes at the DRAM rate). Every
  tensorized leaf there is compute-bound at `peak_flops`, so the score cannot see load paths at
  all: making `model_default` pick a staged twin needs the model to price what staging buys, not a
  tie-break.
- The ranking score's `cache_capacity_bytes` and `cache_bandwidth` are calibrated, not looked up
  (gh-ocannl-1253): the working set sums per-block footprints that co-resident blocks share, so
  the boundary sits above any L2. `tools/calibrate_cache_capacity.exe`
  (`Ocannl.Calibrate.working_set`/`fit_capacity`) sweeps the untiled weight-gradient contraction
  over reduction extents on the plain default pipeline and splits the requested-bytes rate against
  the STREAM-calibrated DRAM bandwidth (pass it: the advisory is a class ceiling): the boundary is
  the capacity, the cached plateau's median the cache bandwidth. Drop launch-bound points first
  (the fit does: below the fastest point's working set), and print no capacity rather than invent
  one when no point reaches DRAM — a lower bound is neither `none` (unlimited) nor a value; minix's
  unified gfx1151 stayed at 1.7 TB/s requested to 10.8 GB, and its class constant is a separate,
  documented per-class decision. `bench_wgrad`'s `wgrad-model:` lines print each plan's per-kernel rank beside its
  distinct-cell bound; `benchmarks/gh1253_cells.sh` runs the whole calibration on one box.
- The ranking score divides by its own rates, never the bound's ceilings (gh-ocannl-1442):
  `issue_rate` (scalar operations, the effective rate of a register-blocktiled kernel with its
  staging) and `dram_bandwidth` (the STREAM rate), `Untimed` fields overridden by
  `model_issue_rate` / `model_dram_bandwidth`, each read as at most its ceiling and as the ceiling
  when unset. Before them the two tiers were inert under default config: the DRAM tier divided by
  the advisory class ceiling (8e12 on CUDA, 6e12 on HIP), and a cache bandwidth below it reads as
  it. The two rates land together or not at all: an issue rate under the ceiling's DRAM makes the
  untiled preset compute-bound and the `auto` gate stops seeding (`cost_model_reuse` pins it).
  `Tile_mma` work stays at `peak_flops` only where the backend's own `mma_accepts` hook accepts
  it (`Cost_model.accepted_tile_mma_flops`, the operand-load count's walk); declined or unreached
  tile-MMA work runs its scalar fallback and is priced as scalar. Under the per-block model the
  chip-resident nodes (`ru_scratch`) leave the DRAM leg, their cost carried by the fitted issue
  rate for the geometry it was fitted on: an approximation, not a model of copies, barriers or
  shared gathers. The staffed start's gate tag names both rates, since its per-segment decisions
  rest on them.
- What the rates do and do not fix, measured (`benchmarks/gh1253_cells.sh`, 2026-10-21, rog CUDA,
  minix HIP unified, tuf HIP discrete). The seeded weight-gradient kernel is predicted within
  0.69-1.35x at b8 and b256 on all three (3-12x too fast at the class peak), and at b1 everywhere
  but tuf, whose small kernels sit on an erratic 0.1-0.4 ms floor; the issue-rate sweep's
  intercept does not capture that floor, so a latency term fitted from it would not either. Two
  orderings it does NOT fix, read off `bench_rank_site` + `benchmarks/gh1442_agreement.py`: among
  scalar sketches of one site the compute leg ties geometries that measure up to 2x apart (an
  occupancy effect no leg models; the argmin was a 2x-slower 16x16x8/4x4 on rog and tuf), and at
  the gh-ocannl-1190 fp8 site every tensorized leaf ties at `peak_flops` while staged leaves ran
  2.3x faster than unstaged and the fastest was the plain 32x32x32 twin, not a swizzled one. The
  operand-load count orders that top pair wrongly, so pricing it is not the staging term either.
- **The score is blind to how a scalar blocktile's work is LAUNCHED** (gh-ocannl-1278, measured at
  gpt2_mini's q/k/v and out projections, b8 and b256, on rog CUDA and minix HIP unified; RANK_ONLY
  `gh1253_cells.sh` + `bench_rank_site` `qkv`/`outproj`). Every scalar leaf of a site scores the
  same compute leg whatever its geometry, batch-grid flavor or coalescing: a plain leaf (batch loops
  serial in the kernel) ties its `bgrid` twin and measures 3-36x slower, and a coalesced leaf ties
  its uncoalesced twin while the fastest leaf at q/k/v was `64x64x8/4x4 bgrid coal-n` (1.25-1.75x
  under the seeded plan). The seeded plan scores slightly ABOVE the tie (its zero-fill kernel's
  ops), so `model_default` under seeding (the `performance`/`approximate` profiles) displaces it
  with the first leaf of the walk, a plain one: `bench_rank_site` `BENCH_PICK=1` reads
  `32x32x8/4x4` at q/k/v (8.8-35x slower than seeded) and `64x64x8/4x4` at the out projection
  (2.6-11x), under gate 32 and `auto` alike. So coalesced leaves stay out of `model_default` until
  the score prices launch parallelism; they ARE numerics-preserving (`coalesce_numerics`: every
  `coal-n`/`coal-k` seed computes the scalar family's values bitwise on order-sensitive inputs, on
  CUDA, HIP and Metal), so admission is a score question only.
- benchmarks/ is the cross-framework parity+timing suite (self-describing safetensors fixtures,
  one-JSON-line runners, loss-trajectory parity gate ~1e-7 fp32 vs pytorch/cpu). The gate
  doubles as a gradient oracle. tinygrad: realize the loss BEFORE `opt.step()` or it recomputes
  from updated weights. The autotune schedule cache persists across processes; compiler changes
  invalidate it by digest.
- A torch row's matmul policy (`regime_settings`, torch's tf32 getters) says nothing about what its
  tensors ARE: a weight cast on load or an autocast context leaves every policy getter unchanged.
  The runner therefore reads `param_dtypes`/`input_dtypes` from the tensors and `autocast` from
  torch's per-device getters INSIDE the step (an autocast entered around the runner is invisible
  from `main`), and `precision_check` fails a row whose label they contradict (gh-ocannl-1208).
  Note that `torch.get_autocast_dtype` answers a default dtype even when autocast is off, so the
  dtype only means something beside `enabled`.
- **Every Metal number recorded before gh-ocannl-693 carries a ~4x accumulator tax and must not be
  compared against one taken after.** Until then a serial reduction at f32 accumulated in the output
  node's global memory, and `volatile_serial_accumulation` shadowed each step's read-modify-write with a
  `device volatile` alias — which defeats every cache and every reassociation, on every unmatched
  contraction, not merely on loss reductions. Localizing the accumulator took gpt2_mini forward on
  Metal (M4 Max, default schedule) from a step p50 of 367.1 ms to 93.9 ms, -74.4%, with the emitted
  fused kernel's volatile shadow count going 124 -> 0 and the batch losses bitwise identical in 7 of
  8 batches (the eighth differs by 5e-7 relative: the qualifier had also been blocking FMA
  contraction). Per segment the win is in the CONTRACTIONS -- lm_head 33.0 -> 6.5 ms, each FFN
  matmul 24.6 -> 5.5 ms, each attention projection 6.2 -> 1.5 ms -- while the cross-entropy loss
  reduction, only 0.21 ms to begin with, moved -5.4%. Two consequences: the published Metal cells in
  `benchmarks/example-report.md` are stale by that factor, and any tuned-vs-default gap measured on
  Metal before this was partly the search buying its way out of the tax (`Privatize` "sidesteps the
  volatile-RMW workaround tax", `autotune.ml`), so those gaps should be expected to shrink.
  gh-ocannl-820 later changed the remaining workaround's form from a volatile accumulator to
  expression-level volatile device reads inside the accumulating loop: on the standalone
  scalar-loss shape the ratio fell from 4.08x to 1.03x while its reproducer matrix stayed green
  row-for-row.
- GPT segment diagnostics follow the fixture's mode (gh-ocannl-1170): `gpt2_mini_train` in
  `bench_gpt_diag` uses `Bench_gpt_model` and `Bench_harness.compile_step`, the same model,
  injected parameters and backprop + SGD as `bench_gpt`. Census/timing in both modes walks the compiled
  `Context.routine.segments`, through the same shipped-IR compilation seam as `dominant_kernel`;
  rebuilding fission from a forward graph omits the very gradient segment the table is for.
  `BENCH_STEPS=1` runs BEFORE `BENCH_SEG_TIMES=1`: isolated backward/SGD runs mutate gradients and
  weights, so controls taken afterwards do not follow the fixture trajectory. Each isolated time
  includes a launch + sync floor, and their sum is not step latency. `gh1133_cells.sh`'s `trainseg`
  step appends those tables to `summary` beside the step matrix and refuses an older forward-only
  diagnostic. The driver records per-cell exit statuses; its shared validator excludes forward-only
  and failed/partial cells from the report. A census inventories both compiled host-gated routines
  before any execution; the benchmark-window filter is only for the dominant-kernel column.
  `BENCH_PROMOTE=0` is still a forward-only reconstructed census experiment.
- **An isolated per-kernel time is repeatable but is no attribution of step time** (gh-ocannl-1347).
  At a fixed revision a kernel's isolated min-of-20 repeats within about 1% across processes. What
  moves it between rows and runs is code: per-layer kernels with identical geometry inline
  different-length recomputations of the Virtual residual stream (`l0_ffn_b2.grad` re-derives
  every later layer's layer-norm backward), so 4.4 ms vs 18.5 ms between layers is work, and a
  change between two revisions is a code change. Do not read "same grid, different time" as noise,
  and diff `build_files/` kernels before blaming the instrument. What it does NOT give is a
  kernel's share of the step. The in-step device durations (`gh1133_cells.sh ... kprof`,
  `--cuda-graph-trace=node`: the step replays one CUDA graph, so without it nsys sees no
  segments) sum to the step; read step shares from them only. A shared-box co-tenant can double a
  whole isolated window, since 20 back-to-back runs fit in under a second; GPU counters (`ncu`)
  need admin on rog. The laptop clock hops between 1.8 and 2.8 GHz under the power cap, so b8
  kernels below 1 ms carry about 10% step-to-step spread, and a per-kernel difference below the
  `SPREAD` it shows is not one.
- **What separates an isolated time from the in-step one on rog, and what does not**
  (gh-ocannl-1391). Not the buffers: the isolated compile is a child of the step's context and
  links against the step's own buffers (a probe of every kernel of gpt2_mini_train_b256 found
  none fresh and no non-finite input; a node new to a context is zero-filled at link anyway). Not
  state drift: running each kernel 21 times in census order compounds the accumulators later
  kernels read (gradients reach ~1e5), yet restoring every kernel's outputs after its window moved
  no row, including gh-ocannl-1347's 2.3x `gamma_ln2_l1.grad` at its own revision `d90f69c2b`.
  That row belonged to the single-kernel layer-norm parameter gradient that gh-ocannl-1325's split
  retired; since then the same gradient's kernel reads within 10% of in-step (0.98 at `3985d045`,
  1.08 at `d061aeb59`). What remains is the clock: a b256 step runs at the SW power cap (reason
  0x4, ~114 W) and in-step every kernel of >= 0.5 ms runs a common factor slower than its isolated
  minimum (F = 1.064-1.066 in the exclusive A/A, per-step 1.05-1.08); the per-step slowdowns of
  kernels adjacent in time correlate (median 0.84-0.96, random pairs about 0-0.27),
  and re-running the whole step right before each isolated window leaves the factor in place, so
  it is the step's aggregate draw, not a warm-up state. `kernel_attribution.py` prints that factor
  per cell and flags `OWN` only what isolated times the factor still misses (18 b256 kernels, 6.6%
  of the step, `wte.grad` at 0.62 the largest). A row that declines
  to compile alone empties the attribution: a kernel declares only the store entries its code
  touches (`C_syntax.derive_kparams`), because parameter names are disambiguated over the code and
  an untouched entry -- a node whose zero `gpu_elide_covered_zeros` dropped -- otherwise shared
  its name with another (ahrefs/ocannl#1387; `shipped_kernels_alone` pins it).
- **HIP in-step attribution: `BACKEND=hip gh1133_cells.sh ... kprof`** (gh-ocannl-1392) traces
  with `rocprofv3 --kernel-trace`, which sees the replayed hipGraph's nodes without a flag, and
  states each cell's memory model (unified on minix's gfx1151, discrete on tuf's gfx1102) from the
  backend's own `integrated` query -- `unified_addressing` reads false on both, so it says nothing
  about the memory model. The boxes' Ubuntu ROCm ships no rocprofv3, and its HSA runtime is built
  without rocprofiler-register, so no rocprofiler-sdk loaded into it sees a kernel (a run under
  `HSA_TOOLS_LIB` exits 0 with no trace). `benchmarks/rocprofv3-install.sh` unpacks AMD's
  profiler plus AMD's HSA runtime alone, rootless; rocprofv3 puts that lib/ on LD_LIBRARY_PATH,
  so the step keeps the system's HIP and hiprtc. AMD's 7.1.0 HSA runtime segfaults in the first
  `hipStreamCreateWithPriority` under Ubuntu's 7.1 HIP (also under AMD's own 7.1 HIP); 7.2.4's
  works. Two non-exclusive runs per box (correctness legs, one repeat each, not measurements):
  every cell mapped; the in-step medians summed to 89-93% of the inference step p50 and 77-108%
  of the training one -- above 100% on tuf's training cell, so before quoting a HIP share as
  precise, check on an exclusive box how rocprofv3's dispatch stamps add up against the step. On
  tuf the inference cell's isolated minima summed to 1.63x their in-step medians in one run (56
  kernels ISO, single kernels 2.7-3.1x off identical twins) and 1.02x in the next (none ISO),
  while the in-step medians moved under 5%: on a shared box an isolated HIP window is as exposed
  to a co-tenant as CUDA's (above).
- That digest covers the compiled result, NOT the diagnostics. When re-measuring a search's
  decline census after changing only an error message or a log line, `rm benchmarks/autotune_cache/*.sexp`
  first — otherwise the second run reports `state=cache-replay`, `timed=0`, `declines=[]` and every
  counter reads zero, which looks exactly like "the problem went away". Fixtures are also not in a
  fresh checkout (`benchmarks/fixtures/*.safetensors` is gitignored and generated), and they are
  only valid while `gen_fixtures.py`, `benchmarks/workloads/` **and the generating numpy** are
  unchanged.
- **Take a GPU schedule verdict at the workload's size, never at a test model's.** A probe of
  the `online_softmax` test's seq-7 model saw the whole fused attention backward in one all-serial
  Metal kernel (gh-ocannl-1124 point 1), but that is `gpu_schedule_min_parallel` (64) declining
  a kernel whose largest chain is 32 threads; at gpt2_mini's sizes every nest carries geometry
  (`gpu_serial_lanes` leg 6 pins it at seq 64). The per-kernel table of the step as shipped
  (`BENCH_KERNEL_TABLE=1`) answers the question directly, geometry included. Two more traps from
  the same measurement (`report-gh1002-fused-backward.md`): **the online-softmax forward alone
  saves no training memory** -- the composed backward still reads `P` and writes `dP`/`dS`, so
  only the fused backward (or a raised recompute cap, for the scores and `P`) moves the peak; and
  `--ocannl_build_files_prefix` names a SUBDIRECTORY of `./build_files` -- an absolute path is
  mangled into one name there, not written to.
- **Do not run `gen_fixtures.py` to get past the digest gate.** It is the reflex the refusal
  message used to invite, and it is wrong: regenerating draws a NEW workload from this box's
  numpy, which retires every published number on the old bytes and does nothing for the other
  measuring boxes. If your copies are merely unrecorded, `python3 benchmarks/fixture_digest.py
  --record` pins them as they are (stdlib-only, no venv, and it leaves other origins alone);
  `--check` reports disk against record, and `--check --expect-origin <box>` answers a driver
  pinned to one box's records by its exit status -- never grep the report for the box, since its
  origin field joins every agreeing box (`m4-max,tuf's bytes`). `--check` names each fixture's
  declared boxes with no entry, and `--expect-recorded <box>` fails on any, so confirming a
  newly recorded box needs no `python -c` over `missing_origins`. Regeneration is a cross-box
  event to be coordinated across every origin in `DIGESTS.txt`'s `# measurement-boxes:` header
  field at once (gh-ocannl-759, gh-ocannl-850). That list is independent of the entry rows, so
  `divergent_origins` and generated reports name a declared box even when its fixture entry is
  absent; deriving the box set from rows would recreate the silence the field exists to remove.
  Rebuilding a fixture that VANISHED is not that regeneration when the rebuild reproduces this
  box's recorded digest: `gen_fixtures.py --out-dir` into scratch (or copy from another tree),
  `fixture_digest.py --check --expect-origin <box>` the candidate, and copy it in only when that
  exits 0 (recipe and the `raw-v1` caveat in `benchmarks/README.md`).
- The boxes are **not** on the same fixture bytes, and the digest file says so per origin
  (`<sha256>  <bytes>  <name>  <origin>`, gh-ocannl-759). `mlp_small` and `gpt2_mini` hash
  differently on minix and rog-nv at identical sizes — two venvs, two numpy `Generator` streams,
  one spec — so `report-hip.md` and `report-gh675-cuda.md` are **not cross-box comparable for
  those two workloads**; each is self-consistent within its box, so within-box session-to-session
  comparisons stand. A fixture MATCHes if it is *some* recorded box's bytes, and every row
  (`fixture_origin`) and report section names whose. Before quoting one box's number against
  another's, check that the two sections name the same origin.
- **A measurement session starts with `benchmarks/check-box.sh --box <id> --fixture <name>...`**
  from the checkout it measures in, and measures only on exit 0 (gh-ocannl-1062). Five waves found
  the gap only once a session had begun: a native boot with no venv, fresh worktrees with no
  fixtures, a ROCm torch that passed `is_available()` and segfaulted on its first copy, drivers in
  a new worktree finding no `benchmarks/.venv` beside a perfectly good shared venv. It checks and
  never provisions; its items, repairs and the per-box state table are in `benchmarks/README.md`
  (Measurement boxes), and the sweep runs it per box without letting a gap touch a unit's row.
  The drivers enforce it: every fixture-loading driver takes `--box <id>` and refuses to start on
  a GAP (`benchmarks/box-gate.sh`, `box_gate.py`); `--skip-box-check` is the deliberate partial
  run, and a new driver that loads a fixture without the gate fails `test_orchestrate.py`'s
  inventory.
- Because of that, no *comparable* cell runs without a Python ML venv -- but the MEASUREMENT path
  does: `bench_mlp --self-test` / `Bench_harness.run_self_test` fabricates a tiny model in memory
  and drives the whole protocol and emitter (gh-ocannl-702), and `test/operations/bench_self_test`
  runs it in CI on `OCANNL_BACKEND`'s backend. Reach for it when changing `measure_and_emit`,
  `Bench_json.result_line` or the step machinery in `Bench_harness`, and when bringing up a new
  backend -- the alternative first signal is a wrong number in a report from a GPU box, days later.
  It sidesteps the fixture contract rather than weakening it: `selftest-tiny` is not a workload,
  its numbers compare to nothing, and the emitted record's `workload`/`variant` say so.
- A benchmark cell reports what it measured through ONE JSON line, so anything that line cannot
  express is a measurement thrown away (gh-ocannl-676). Non-finite numbers are the whole class:
  OCaml's `%g` spells them `nan`/`inf`, `json.dumps` writes `NaN`, JSON has neither, and
  `orchestrate.py` drops a cell whose line does not parse — so a DIVERGED run, the thing the parity
  gate exists to catch, was reported as a broken runner. All three runners emit `null` there
  (`Bench_json.num`/`fixed`/`nums`, `bench_common.json_safe`) and the orchestrator reads it as the
  DIVERGED verdict. The general trap: `json.loads` accepts `NaN` and `Infinity` (and reads `1e999`
  as infinity), so the Python runners' cells survived while OCANNL's vanished, and the report read
  as an OCANNL runner bug rather than a numerics one — when a sweep loses only one framework's rows,
  suspect the emitter's spelling before the framework. Every Python reader of a runner's result or
  checkpoint line therefore parses it with `benchmarks/strict_json.py`'s `loads`, which refuses all
  three spellings (`NonFinite`, a `ValueError`), so a regressed emitter fails its cell by name instead
  of reaching only the Python consumers. A new reader uses it too — a module a driver runs by path
  finds it beside itself, and a script run under `python3 -I` puts its own directory on `sys.path`
  first (`gh834_session.py`) — and test_orchestrate's inventory scan fails any other `json.loads`
  in the drivers that is not exempted with a reason.
- A shell driver reads that line by PARSING it (stdlib `python3`), never with a `sed`/`grep`
  pattern over its text: a scan from `"arm":"A"` to the first `}` read a nested object placed
  before `best_ms` as an arm that timed nothing, and pinned the runner's field order to suit it --
  and `step_ms` is a `{p10,p50,p90}` object, not a number. For one field,
  `python3 benchmarks/strict_json.py field CELL.out step_ms.p50` (or `tune.shipped`) prints it off
  the parsed last result line and exits 1 naming why there is none; `ShellFieldGrepInventoryTest`
  fails a driver line that greps a quoted JSON key (gh-ocannl-1355). For a whole verdict,
  `benchmarks/gh834_session.py` is the shape; its harness `benchmarks/test-gh834-cells.sh` renders
  `Bench_json`'s own templates, so a renamed field fails there rather than in a box's session.
  A shell driver's result reader -- a summary matrix, a gate, a protocol check -- lives in a module
  the driver runs by path, not in an inline `python3 -` heredoc that a test can reach only by
  extracting it from the script's text (gh-ocannl-1365: `gh612_parity.py`, `gh728_protocol.py`,
  `gh1003_summary.py`): the test imports or runs it directly, and drives the driver's step where it
  needs no device. `gh1133_summary.py` owns reading the numbered cells (`read_cells`), which
  `gh1003_summary.py` shares, so a change to how a result line is read or refused lands once.
  `gh1003_fold_cells.sh OUT CAP summary` over an earlier OUT re-renders its matrix under the fold
  blocks its driver.log recorded -- the one invocation a non-empty OUT is accepted for.
- Every report-writing script in `benchmarks/` -- `orchestrate.py`, the `gh*_cells.py`/`_summary.py`
  drivers, `gpt2_bucket.py`, `kernel_attribution.py`, and the shell drivers' inline Python --
  writes each Markdown table row through `benchmarks/display_safe.py`'s `table_row`/`table_lines`,
  which refuse (`UnsafeTableRow`) a cell holding a raw `|` or line break when the row is rendered,
  and each run-log line naming an outside value through `log_line`/`log_lines`, which refuse
  (`UnsafeLogLine`) a raw line break (gh-ocannl-1355). A string the operator or the environment
  supplies -- a fixture's file name, a cache path, a knob value, a driver's label or treatment
  argument -- reaches a cell or a line through `display_value` (`code_span` inside backticks),
  which escapes exactly what the builders refuse; the records keep the raw value. Text a runner or
  the host printed and the script's own regex matched (a census, a kernel's writes, a CPU name)
  stays raw: the render-time refusal is its guard, and quoting it would only make every report
  noisier. `TableRowTest` scans every shipped `*.py` and every `*.sh` heredoc fed to `python3` for a
  row literal (`| ...`, `|---`, `| ---`) written outside the builders, so a new table cannot bypass
  them. orchestrate's `!!!` lines print inside a cell's failure handling, where a refusal would
  lose the sweep, and stay unchecked; the end-of-run RUNNER FAILURES lines check the same notes.

- Before its later stages a cell CHECKPOINTS what it completed (gh-ocannl-1209):
  `measure_and_emit` writes a `bench: checkpoint {...}` line to stderr, flushed, after every
  parity step and again after the timed steps, before the dominant-kernel instrument -- which
  compiles and times every shipped kernel alone and can outlast the workload (a TUF
  `gpt2_mini_train_s1024` cell lost six finite losses to a 90 s cap expiring inside it). The
  instrument's own wall-clock budget, `BENCH_DOMINANT_KERNEL_BUDGET_S`, bounds that stage: past it
  the column is `no-kernel` and the cell keeps its result (gh-ocannl-1260,
  `Bench_harness.with_deadline`); the checkpoint covers a cap shorter than the budget. A
  checkpoint carries the losses so far, stage statuses, the step count and the cell's identity, and
  NO timing: its `result` stage is always `pending` and `accepted` is `false`. `orchestrate.run_cell`
  folds the last one into a failed cell's note and hands it to `on_checkpoint`;
  `orchestrate.run_cell_kept` returns it beside the note, and `orchestrate.failure_record` is the
  one record every sweep driver writes of a failed cell (gh-ocannl-1234: orchestrate's
  `partial-failures.jsonl`, gh1181's `failures.jsonl`, gh675's failed passes in `records.jsonl`,
  and gh1002's `failures.jsonl`, which reads the checkpoint off the cell's own `.err`). It keeps
  the losses -- beside the failure, never among the rows -- with the dispatched `regime`, the
  inherited OCANNL_* environment, and under `provenance` the fixture stamp, the source revision
  and the runner binary's sha256, since the checkpoint names its fixture and runner only by paths
  a later sweep reuses. An interrupted cell's checkpoint is recorded before the cancellation
  propagates.
  The report's "Runner failures" table shows those saved losses' drift from the reference beside
  the envelope the cell would have been gated at (`saved_loss_parity`) -- a DIAGNOSTIC of whether
  the work before the failure was on course, worded outside the parity column's vocabulary; the
  cell still fails the sweep. The envelope's regime comes from the checkpoint's own `profile`
  (the one its process resolved, as on the result line), not from what the sweep threads down
  from the dispatch site: a forgotten argument there used to fall back to the exact envelope. The
  `partial-failures.jsonl` record carries the dispatched `regime` beside it, and the diagnostic
  names a disagreement between the two. A profile name cannot show an ambient knob with no profile
  behind it (`OCANNL_TF32_MATMULS` on an exact cell reads `profile: null`), so the checkpoint also
  carries the result line's `regime_knobs`, and the diagnostic names a knob outside the envelope's
  regime in `regime_knob_leaks`' words, a declared `--cell-flag` excepted (gh-ocannl-1297).
  Do not give the line a `{` prefix or move it to stdout: drivers take a cell's result from the
  last `{`-line of its combined output, and the checkpoint must never be one.

- A benchmark cell can WEDGE, and a wedged cell is a failure rather than a slow one
  (gh-ocannl-760). tinygrad's parallel beam search deadlocks intermittently — its candidate-compile
  pool is `spawn`-based with `maxtasksperchild`, and a worker lost between `imap_unordered` chunks
  leaves the parent in `futex_do_wait` forever, at ~1% CPU with the GPU idle, on searches that take
  under two minutes in their other repeats. Seen on both the CUDA and the HIP box; the root cause is
  upstream and unchased. It remains in tinygrad 0.14.0: on minix, one of five cold-cache
  `gpt2_mini` BEAM=2 repeats exceeded a 300 s cap against three healthy 37–51 s searches (the
  fifth completed after 157 s). Therefore `orchestrate.py` defaults `--beam-parallel` to `0` on
  every measurement box — the only setting that removes the spawn pool — while an explicit
  positive value opts back into the pool with the cap as backstop (gh-ocannl-843).
  `cell_group.py` gives benchmark cells in `orchestrate.py` (whose `run_cell` gh675 and gh1181
  run theirs through), `gh998_cells.py` and `gh1002_cells.py` (and the latter's preflight Dune
  build, which forks compilers; gh-ocannl-1200) one shared group/job, TERM-to-KILL,
  output-preserving reap discipline — the drivers start no other child but short `git`/`ps`/`sysctl`
  probes (`test_sweep_drivers_have_no_unmanaged_spawn_site`, over every `benchmarks/*.py` that
  imports `cell_group` directly or through another script, so a new driver is scanned unlisted);
  `orchestrate.py` puts cells under `--cell-timeout` (default 1800 s) and kills the GROUP on expiry:
  the pool workers hold the cell's
  stdout pipe, so killing the direct child alone moves the hang into the sweep's own
  `communicate()` — the trap to remember whenever a runner is bounded from outside. Two facts
  outlive the fix: `timeout(1)` cannot be the mechanism here (uutils' `-k` misses the process
  group), and a search killed midway leaves tinygrad's single `cache.db` partial, so the next run
  over it reports a `searched` verdict nobody wrote — the kill renames it aside (with the sqlite
  sidecars UNDER the quarantined name, since `<database>-wal` is the only place sqlite looks for
  them), and any hand-killed wedge must have the same done to it before its retry means anything.
  OCANNL's `autotune_cache/` is not torn by a kill — `Utils.Atomic_file` commits entries by rename —
  but a retry over it is still not a from-scratch search: the finished arms replay and the rest are
  searched, and a pass reports SEARCHED whenever ANY arm searched, so the mixed retry's compile cost
  wears a from-scratch label. Wipe it before quoting a search timing.

- Killing a runner from outside: the escalation to SIGKILL is owed to the process GROUP, never to
  the child's pipes (`cell_group.terminate`; gh-ocannl-842 unified the two hand-written versions
  in `gh675_cells.py` and `orchestrate.py`). A descendant that ignores SIGTERM but does not hold the
  cell's stdout lets `communicate` return promptly, so a kill path that escalates only when its
  read blocks skips the SIGKILL exactly where it was needed — and the survivor keeps the GPU while
  the sweep stamps every later cell valid. The reverse trap is in the same function: a descendant
  that DOES hold the pipe makes the parent's read the new hang. Both are answered by
  `os.killpg(pgid, 0)` polled while reaping, and by carrying the pgid rather than looking it up
  after the leader is gone. The escalation must also be spelled as a boolean rather than a signal
  number if the code runs on Windows: `signal.SIGKILL` does not exist there, so a
  "SIGKILL if posix else SIGTERM" argument silently picks the CTRL_BREAK branch again instead of
  a Windows Job Object. The child is born suspended, assigned to a kill-on-close Job, and only
  then resumed; the Job retains descendants after their leader exits and supplies the active
  process count that `taskkill /F /T` could not. And `start_new_session` cuts both ways — it is
  what lets a cap reach the
  descendants, and it is why a SIGTERM to the DRIVER no longer reaches them, so a driver that
  spawns cells this way owes itself a SIGTERM handler that takes the running cell with it.

- Two traps in that handler, both paid for in gh-ocannl-760's review. A cancellation must be
  DEFERRED across the spawn and across the kill — between `_execute_child` and `Popen` returning
  there is no name for the new process, and a signal raised inside an `except` clause that is
  doing the killing cannot be caught by a sibling `except BaseException`, so the escalation stops
  halfway — but deferring must not be spelled `pthread_sigmask`: the mask is INHERITED across
  fork/exec, so every child starts with SIGTERM blocked, its graceful phase does nothing, and
  every kill costs the full grace before SIGKILL (measured here: 1.0 s → 11.5 s per killed cell).
  Deliver a held cancellation from the deferral's `finally`, not after it: the latter is skipped
  when cleanup is already unwinding with another exception, leaving SIGTERM pending until a later
  child or forever when this was the last one (gh-ocannl-842).
  Defer with a flag the handler checks, and re-raise on the way out. The other one is about what
  a survivor means: a cell that ran to completion while something it spawned outlived SIGKILL is
  a FAILED cell, not a successful one with a warning — the survivor holds the device, so the
  row's own timing and every later row of that run were measured against it.

- A detached Windows supervisor can give even a trivial Python cell a live `conhost.exe` after
  the Python leader exits (gh-ocannl-974). The Job accounting is correct: killing that member
  means `on_incomplete(killed=True)`. A test requiring an ordinary exit with no remaining member
  must suppress its console with `DETACHED_PROCESS`, as `test_orchestrate.py` does; changing the
  production observer or adding a settling sleep would change the cleanup policy to fit a fixture.

- A `bin/` bench's correctness guard is a position-weighted checksum of the WHOLE output, and its
  position dependence is the whole of it: a residue of the FLATTENED offset `t = i*n + j` loses its
  row dependence exactly when the modulus divides the row stride, so `1 + (t mod 251)` gives every
  row the identical weight at n = 251, 502, 753 — a row permutation, which is what a misplaced edge
  peel produces, then leaves the checksum unchanged while the interior spot cell is blind to other
  rows at the same time. The same collapse hits operand data drawn as `(t mod p)`: at `p | stride`
  every row is identical and a schedule substituting the wrong row computes the right answer. Key on
  the (row, column) PAIR through `Bench_checksum` (`bin/bench_checksum.ml`, gh-ocannl-711) — shared
  by `schedule_bench` and `narrow_gebp_bench` precisely because the fixed copy and the degenerate one
  had sat one file apart. Keep the weights capped below 256 so the products of exact-in-binary
  operands stay exact in the accumulator and variants summing in different orders compare BITWISE;
  and keep the checksum outside the timed region. But do not let the checksum be the ASSERTION: it is
  a linear functional of the output, so a row swap survives it whenever the value difference is
  orthogonal to the weight difference — by the weights colliding (a capped weight puts a row's
  weight vector in `cap ^ row_stride` values, so at stride 2 rows 9 and 363 share one) or by plain
  cancellation (at n = 2 the generated rows 355 and 2891 cancel in BOTH streams). No
  bounded-weight scalar escapes that class, and more streams only shrink it. What decides is
  `Bench_checksum.first_difference`, an elementwise comparison against the first variant to
  complete; the checksum is what the line PRINTS, a fingerprint for reading a table and comparing
  runs. Make that reference the UNSCHEDULED computation rather than "whichever variant ran first":
  the two coincide while the naive leg runs and diverge exactly where it matters — under
  `schedule_bench`'s `naive_repeats = 0` the naive leg is skipped, and the first scheduled variant
  would then be labelled the oracle without anything having validated it. Skipping an expensive
  reference TIMING should cost the timing only; materialize the oracle with one untimed run. And
  make a failed comparison EXIT NONZERO once every variant has been reported: a guard that only
  prints leaves an automated run free to keep the speedup of a kernel already known to be wrong —
  the same hazard `Verdict` exists for on the test side. Where a bench has legs that may
  legitimately round differently, put the predicate saying so in ONE binding used by both the
  runtime note and the exit status, and point the comparisons that stay required at each other
  rather than at the leg the note excuses (in `narrow_gebp_bench`, `packmma_par` is compared against
  `packmma` — they narrow at the same k-block boundaries, so nothing the note excuses can separate
  them, and comparing both against naive would bury a par-only defect in expected rounding).
- Two data-side blindnesses sit behind that guard, where no output check can help. A producer value
  that can BE the accumulator's init hides a dropped producer: a mixed operand row is all-zero with
  probability `levels ^ -row_stride`, likely at narrow extents, which a flat form's marching values
  could not do — so the multiplicand whose row spans the reduction is minted strictly positive
  (`Bench_checksum.positive_level`). And two identical operand rows make a wrong-row schedule
  compute the right answer: how many rows a generator keeps distinct is bounded by
  `levels ^ row_stride` whatever it does, and keying on the (row, column) pair does not move that
  bound — only the LEVEL COUNT does. `schedule_bench`'s `ma` is 48 levels of 1/16 over (0, 3] since
  gh-ocannl-738, up from 12 of 1/4, which at the narrowest reduction it accepts (k = 2) takes the
  bound from 144 to 2304 and the measured first repeat from row 11 to row 33; the table over k is
  the golden of `test/operations/bench_checksum_discrimination`, and it is the point of that test
  rather than noise in it.
- What caps that level count is the MMA OPERAND FORMAT, not f32. A bench whose legs tensorize
  compares them against an unscheduled oracle that carries no `Tile_mma` and so rounds nothing,
  while the tensorized legs round both operands to the mma input format — so `= reference` on an
  mma line is a claim that the operands survive that rounding exactly, and the operand generator's
  granularity is the thing that has to fit. CUDA's uniform-f32 arm is gated on `tf32_matmuls`
  (which `schedule_bench` sets), and tf32 carries an 11-bit significand, as does f16; a multiple of
  1/16 below 3 needs six significant bits and mb's integers in -8..8 need four, so the budget has
  about five bits of headroom. Measured on an RTX 5070 Ti under CUDA 13.3 at 256³, 512³, 128x128x1024
  and 320x192x64: `= reference` on every leg, `mma_pd1`/`mma_pd2` confirmed `Mma_intrinsics` by the
  census. The check is not vacuous — reminting `ma` at 1/4096 (twelve significant bits) turns
  exactly the two mma legs red and leaves `parallel`/`smem`/`regtile` green, which is the negative
  control to re-run before raising the granularity again.
- A mixing function that folds its state DOWN to its output width must not do it linearly. The
  aperiodic mix here folded a 40-bit product into 24 bits with one xor-shift, which is GF(2)-linear,
  so two rows' outputs differed by a value depending on neither the column nor the salt: row pairs
  existed (5977 and 10232 the first of eight below 20000) that were identical in EVERY derived
  stream at EVERY salt, which no number of streams could repair. Masking the state to the output
  width FIRST fixes it structurally rather than statistically — the multiplier is odd so the state
  is injective in each index, and every later step (xor-shift, multiply by an odd constant) is a
  bijection on that width, so distinct indices below the width differ at every column. Check a hash
  meant to separate indices for this before trusting a sweep of it.

- A benchmark leg belongs to a WORKLOAD, not to a runner (gh-ocannl-551). `BENCH_STATIC_SCALE` /
  `BENCH_GATE_INTERVAL` lived in `bench_mlp` alone, so the gate-cost contract silently had no
  answer for `gpt2_mini` — and a forward-only workload has no loss scale to gate in the first
  place. The resolution to copy when adding a leg: put the flag parsing and step shapes in
  `Bench_harness` (shared), dispatch the step shape on the fixture's `mode` (as the Python
  runners do), add a training workload when the question needs one (`gpt2_mini_train`), and make
  orchestrate report an inexpressible cell as NOT APPLICABLE with its reason instead of omitting
  it. In OCANNL specifics: a parameter has `batch_dims = []` pinned by `Tensor.param`, so a
  positional-embedding table must be output-axis-shaped and placed with an einsum-add. The gpt
  f16 cells additionally needed gh-ocannl-548 (`-inf` mask fill) and gh-ocannl-547 (reduction
  identities out of the fp16 constant guard's scope); the runner-side workaround they replaced —
  pinning the softmax at f32 and materializing the masked scores — is gone, so do not
  reintroduce a pin for a constant the library now keeps representable.
- Before comparing two vector renderings, check that neither is measuring its SCALAR PEEL
  (gh-ocannl-575; the peel is gone since gh-ocannl-620 — the remainder is a vector tile now — but the
  reading trap stands). `try_register_tile` covered `n - (n mod bw)` columns and peeled the rest to scalar
  code, and a peeled column costs roughly a whole vector slot — so at a width that does not divide
  the extent, the peel can be most of the runtime. This bit the pure-fp16 vs f32-compute comparison
  exactly backwards: doubling the lane count doubles `bw` too, and at n = 512 the wider tile peeled
  32 columns where the f32 one peeled 8, reading as "fp16 arithmetic is 1.5x slower" when it is
  1.6-1.8x faster. The tile width now adapts to the extent, but the general trap outlives the fix:
  when a change moves lane counts, tile widths, or blocking factors, benchmark at extents ALL arms
  divide evenly (or sweep the geometry) before attributing a difference to arithmetic. Power-of-two
  extents are not automatically safe — they are multiples of no odd width.
- Single-threaded microbenchmarks on Apple Silicon read ~40% high on the first run of a cold
  machine and settle once runs are back-to-back (core-cluster placement and clock behavior). Take
  the median of several warm repetitions; a lone run after an idle gap is not comparable to one
  taken mid-sweep.

- The A/B protocol that makes a small effect legible: alternate the arms RUN BY RUN (a fixed order
  confounds the treatment with position in the session — worth ~1.4pp of an apparent 7.1% once),
  wipe the schedule cache between arms (it is keyed by base-code digest and shared across binaries,
  so one arm otherwise replays the other's winner), and keep as a drift control a variant whose
  EMITTED kernel is byte-identical across the arms in the same process. Diff the generated
  `.metal`/`.cu` per variant before trusting a control: gh-ocannl-567 learned that way that two of
  its intended controls had also changed, and that the surviving control's own spread (±3–7% on
  Metal kernel timings) was larger than the effect under test. After `Train.tune_placements` the
  debug source left on disk is arm B's — the returned already-compiled arm — so grep the artifact you
  actually mean.
- An effect smaller than the variance between two independent searches (on `gpt2_mini` that
  variance is several percent of a step, gh-ocannl-1165) is measured by SPLICING, not by comparing
  two searched schedules: both arms replay the control's winner except at the target segments
  (`bench_gpt`'s `BENCH_SPLICE_*` mode, `benchmarks/runners/ocannl/bench_splice.ml`, gh-ocannl-1277;
  the benchmarks README has the recipe). Pick the targets from its `BENCH_SPLICE_LIST=1` digest
  groups, and run both arms on the treatment's build.
- Benchmark drivers should START hermetic rather than be hardened into it: unset ambient
  `OCANNL_*`/`BENCH_*` wholesale, pin every treatment on argv (the command line outranks every other
  config source), content-stamp the fixtures, fail loudly per cell — and name the DRIVER, not a
  hand-typed command, as the reproduction. `benchmarks/gh514_cells.sh` is the model, and it looks
  like that because a ten-round review was a long tail of "pin one more ambient knob". Note that
  `Utils.config_file_args`' `find_up` takes the NEAREST `ocannl_config`, so running from
  `benchmarks/` shadows a personal root config — a feature to rely on, and a trap when reproducing
  someone else's numbers from a different cwd.
- Cleanup, release and decline paths are acceptance-tested by fault INJECTION, not by the happy path:
  of the review findings on the gh-ocannl-550 release work, 14 of 14 lived on error paths
  (decline / raise / abort / cache-hit / callback-failure), where a flattened memory curve over three
  clean replicates is almost no evidence — the code exists for the error moments.
  `Autotune.on_candidate_attempt` / `on_candidate_preflight` are the injection seams.
- When a change re-promotes a WAVE of goldens (an init-stream change, a reduction-order change), grep
  the full new outputs for `nan`/`inf`, not just the loss lines. A promoted golden records whatever it
  is given: a trailing dim-1 projection bug once filled conv kernels with NaN behind numbers that
  still looked plausible line by line.

# Scheduling and autotune

Schedule legality and coverage, sketch families, and how to read what a search actually did.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

## GPU mapping and companion coverage
- Small leading GPU axes can starve the default schedule even when later axes have ample work
  (gh-ocannl-995). `gpu_parallel_suffix` chooses a better-populated pair before
  `analyze_parallel_chains` proves ownership, with the original selection as a conservative
  fallback; `zero_expansion` shares the policy. Never select a subset after proving a larger
  thread-coordinate tuple. Metal GPT measurements and the explicit CUDA/HIP residual are in
  `benchmarks/report-gh995-metal.md`; `gpu_small_leading_axis` executes dependent-nest and
  zero-initialization oracles. Since gh-ocannl-1133 that pair is the FALLBACK of the lane plans
  below, not the default.
- The default GPU schedule maps every loop of a proved chain (gh-ocannl-1133, "lane plans"):
  `analyze_parallel_chains` runs uncapped, `Schedule.plan_nest` puts the leading loops on `Grid`
  and the innermost on the `Workgroup` lane (split `Grid` outer past the block size, never
  `Serial`), widening the workgroup upward while it holds fewer than `gpu_schedule_workgroup_fill`
  threads. The constraint mapping more loops adds is COVERAGE, not ownership: every
  chain-carrying nest of a kernel needs the same `Grid` and `Workgroup` counts, or
  `validate_parallel` rejects the nest short of a slot. `unify_candidates` narrows workgroups or splits
  a short nest's lane into a one-block `Grid` slot, else declines; `lane_plans_gain` then keeps
  the two-loop presets unless no nest loses groups or active threads and one gains. The launch
  takes each slot's maximum ACROSS nests, so `plan_chains` also judges the union: workgroup
  product within the block size (a 32 x 8 beside a 256 x 1 launched 2048 threads) and at most
  twice the largest nest's own allocation (a `(v, d)` weight gradient beside a `(b, s, d)` one
  launched 1024 x 128 groups; on HIP that segment became the training step's dominant kernel).
  Per-nest checks cannot see either. Dependent
  nests have pointwise-equal chains, so they get identical plans at every unify step -- keep any
  new unify rule a function of the chain alone, or the positional thread identity breaks. The
  slot arithmetic has one owner, `Schedule.launch_geometry_of_nests` (per-slot maxima BEFORE the
  `.z` fold, overflow = refusal), which `Sketch_families.predicted_launch_geometry` forwards to.
  The lm_head was untouched by this: its segment carried `max_logits`, whose chain `(b, s)` trims
  the logits nest's `(b, s, v)` to `(b, s)` -- an alignment trim fission's no-loss guard let through
  comparing at a `max_chain` of 2; the schedule-aware merge rule below cuts it. `test/operations/gpu_parallel_prefix`;
  measured by `benchmarks/gh1133_cells.sh`; the tables are on lukstafi/ocannl-staging#909 and ahrefs/ocannl#1133.
- Fission's merge decision is schedule-aware on GPU backends (gh-ocannl-1126, `Schedule.keeps_mapping`,
  config `gpu_fission_keep_mapping`): a merge the race analysis admits -- an aligned dependent merge,
  and a conflict-free one, which it admits unconditionally -- is still refused when some statement
  gets less of its OWN mapping merged than alone (fewer groups or active threads of its own loops,
  read off `default_gpu`'s ops by `statement_mappings`). Judge per statement, never by the kernel's
  largest thread count: the merged kernel's launch can be wide while one nest in it runs on 256
  threads -- the composed `v.grad` trimmed to `(h, e)` by `w_v.grad` at batch 1 (86 ms per layer on
  Metal), the fused dV denied its lanes by dK, the logits trimmed by their row max. The probes run
  under `Indexing.discarding_symbols`: `split` mints symbols, and a discarded probe's would shift every
  later minted name. Autotuner fission call sites pass `Schedule.fission_keep_mapping` too, or their
  segmentation stops being the untuned default's (`fission_equivalence`): they go through
  `Autotune.tuner_fission`, which carries it with the placements and zero policy (gh-ocannl-1483). The segmentation's inputs
  beyond the code (the gate, `gpu_schedule_block_size`, `_min_parallel`, `_workgroup_fill`) stay
  out of the schedule-cache key: a fissioned winner replays its persisted segmentation, never a
  re-derived one (gh-ocannl-1164, "Cache regimes and placement replay" below). A cut needs no retest after
  scope-local resolution: a cut that resolution merges back is serial either way (the reason is in
  the comment on `keeps_mapping`). `test/operations/gpu_fission_mapping`.
- A `Zeros` segment is held together by the same rule (gh-ocannl-1169, `zeros_keep_mapping`): one
  launch serves every node, so a run of zeros with no common lane-plan topology sends ALL of them to
  the two-loop presets -- the gpt2_mini step's 147 gradient zeros did (`plan_chains` found no
  topology), the only mixed-shape fallback left in that step once `keeps_mapping` cut the
  `wte.grad` + final-LN-grad union. Zero runs are stably sorted by shape in `collect_units` (they
  commute; units stay a function of the code), else the cuts fall at every alternation of a
  backward pass's shapes. Case 4 of `gpu_fission_mapping`. A parity check of the sort must not
  share it: compare against the unfissioned routine (no `lowered_transform`, original order), and
  prefill the zeroed buffers with nonzero values, since fresh ones are already zero and hide a zero
  that never runs (gh-ocannl-1250, `zero_run_reorder`). Reordering zeros WITHIN a run is
  value-neutral (that is the soundness argument), so no executed check can see it; the hazards a
  value check catches are a dropped zero and a sort that crosses the run's boundary (an
  overlapping slice write, an accumulation). The fold pairs a zero with its neighbour AFTER the
  sort, so the sort can enable a fold the statement order did not have.
- Scratch (`Local`, a function-scope array on the C backends) is where the CPU preset and fission
  part from the GPU reading (gh-ocannl-1343). A whole-node `Zero_out` of scratch conflicts with
  nothing, so greedy grouping used to take it into the PRECEDING segment and cut before its
  accumulation; the node then crossed the boundary, `promote_crossing` made it `On_device`, and the
  promoted zero -- a bare materialized write -- bailed BOTH segments, which coalesced serial. In
  mlp_bn_names the batch-norm statistics' zeros serialized the first layer's 1000x200x30 matmul that
  way. `group_units` now lets such a zero join a segment only when the unit accumulating into its
  node would join too; a `Solo` next unit that may touch the node (opaque units always may)
  splits it off as well, so the zero shares the serial unit's fate (review round 2). Separately, `default_cpu` runs `analyze_parallel_chains ~shared_locals:true`:
  a nest writing only scratch takes its chain from those writes (on GPU it runs whole per thread
  over a per-thread copy, so the empty chain is right there; on cc a serial nest runs once, on one
  core), falling back to the plain analysis when linking it bails OR costs the kernel its
  `min_parallel` eligibility (a `y[i, j]` reading `s[i]` is trimmed to `[i]`; compare
  threshold eligibility, never chain sizes -- the preset retypes only the outermost loop, which a
  trim keeps, and a size comparison left the scratch nest serial in exactly the case it serves:
  review round 1; the test's stats and rowstats legs pin both directions) -- `C_syntax.parallel_grid_safe`
  stays the arbiter of whether a loop over a local renders in parallel. Mind the size: at the
  default `stack_threshold_in_bytes` (16 KB) a scratch node holds at most 4096 cells, below the
  default `cpu_schedule_min_parallel`, so the chain change bites in the tuner's `min_parallel:1`
  presets or beside a large nest; the zero rule bites at any size. `test/operations/cpu_scratch_matmul_grid`.
- A parallel loop under a serial loop is reachable only past lane-uniform scalar work
  (gh-ocannl-1003). The presets' chain is the single-child loop path, which stops at the
  online-softmax hoist's preamble (`for t { p := P[s, t]; for e { O[s, e] += p * V[t, e] } }`);
  `default_gpu`'s lane geometry reads the path through it (`path_loops ~lanes`) and emits
  `Grid (every chain loop above) -> Serial t -> Workgroup e`. It applies only when every
  chain-carrying nest of the kernel is such a lane nest with the same Grid arity (positional slot
  coverage), so a kernel mixing it with a plain nest keeps the presets. The preamble must be loop-free
  (each lane recomputes it: the recomputed-scores form's inlined `q . k` cost 1.5x at seq 1024 on
  Metal under lanes) and must precede exactly ONE loop: a nest with two sibling channel loops under
  one preamble (a fused dK+dV) is not reached. `test/operations/gpu_serial_lanes`; measured in `benchmarks/report-gh1003-stage1.md`.
- One loop IS admitted into a lane preamble (gh-ocannl-1124): a `preamble_reduction`, a serial loop
  whose body is ONE loop-free accumulation into the fused backward's own `dp` local
  (`dp = sum_e dO . v` ahead of dK's and dQ's channel loop), per `gpu_lane_preamble_reduction`.
  Provenance, not shape, admits it (`Online_softmax.reassociable_local`, in the schedule and the
  renderer alike): the all-reduce reassociates, and the license is `online_softmax_backward`'s,
  an approximate-tier gate. A same-shaped ordinary local keeps its plain plan and, hand-retyped
  `Workgroup_reduce`, the hardware binding a staged reduction relies on. An
  inlined reduction inside an expression (a `Local_scope` whose body loops) stays refused in every
  mode. `cooperative` retypes a reduction whose extent is the lane's unsplit workgroup
  `Workgroup_reduce` — the lane's own `.x` slot, no lane axis nested inside the output lanes — and
  `C_syntax.try_lane_all_reduce` owns EVERY local-target `Workgroup_reduce`: the xor butterfly
  (an all-reduce: the total lands in every lane, no shared scratch, no barrier) at exactly one
  simdgroup, else the serial loop in every lane. Never the `Workgroup` binding: each lane would
  keep its own term, a wrong value rather than a race. Past one simdgroup (gh-ocannl-1168) the
  per-simdgroup totals go through a workgroup-shared slot array between TWO barriers per pair:
  the second keeps the next pair's stores off a slot some lane has not read. Every lane reads all
  slots in slot order, so all hold the bitwise-same total. Every lane must reach both barriers,
  and whole-`.x` coverage alone does not prove it: a guard reading a lane-dependent scope local
  (`flag := lane < 32; if flag { ... }`) parts the lanes. `Low_level.lane_all_reduce_sites` is the
  one predicate: whole `.x`, a one-dimensional workgroup, no enclosing workgroup loop, and every
  enclosing condition built from constants and non-lane index symbols (a local or a memory read
  disqualifies). The renderer and `Schedule.workgroup_memory_bytes` (which counts the slots
  through the `lane_all_reduce_workgroup_bytes` capability) both read it. The same walk finds
  `try_warp_reduce`'s multi-warp tree into a cell, whose two barriers have NO barrier-free fallback
  (gh-ocannl-1238): `Low_level.inserted_collectives ~warp_size` lists every statement whose
  rendering inserts a lane collective the IR lacks, `validate_parallel ~warp_size` refuses a
  non-uniform entry before rendering (typed `Illegal_schedule`), and the renderer emits the tree
  only for a listed uniform entry. Inserted collectives answer to uniformity over the lanes they
  span, the IR's own barriers to the stricter no-`If` rule. The single-warp forms insert no barrier
  but still shuffle, and a lane missing from a shuffle is undefined on every GPU backend (gh-ocannl-1334):
  they answer to WARP uniformity, where an enclosing workgroup loop (always above slot 0) and a
  guard on its symbol are uniform, since both renderings span the whole `.x` dimension in whole
  warps. The single-warp tree into a cell is refused under divergent control; the butterfly
  declines to the serial loop. The schedule retypes at `k` simdgroups only up to
  `gpu_lane_all_reduce_simdgroups` (`auto` = `hardware_limits.lane_all_reduce_simdgroups`: 4 on
  Metal, where heads of 64 and 128 train 3-31% faster on it, and on CUDA (RTX, 16-63% faster,
  gh-ocannl-1256); 1 until a device measures the barriers paying, HIP included).
  Measured on Metal (D1 training, lukstafi/ocannl-staging PR for gh-ocannl-1124): duplicated is a
  1.07-1.45x step REGRESSION (every lane pays the value width per pair), cooperative a 0.94-0.98x
  win, and it lanes dQ too -- fission then cuts dQ from the row dot `D`, whose merge would now cost
  dQ its mapping. CUDA agrees (0.91-0.98x); HIP (gfx1151) LOSES 1.07-1.20x, and not from workgroup
  width (widening the lanes to `gpu_schedule_workgroup_fill` was measured neutral on Metal and CUDA,
  worse on HIP, and reverted): every lane recomputes the pair's scalar preamble (`p`, `ds`), which
  the plain plan pays once per thread over its channel loop. So the default `auto` resolves per
  device from `hardware_limits.lane_scalar_recompute_cheap` (Metal, CUDA true; HIP, cc and anything
  unmeasured false, i.e. refused) -- a device fact on the limits seam, never a backend name. `test/operations/gpu_lane_reduction`, and leg 6 of `gpu_serial_lanes` pins dK's
  own nest.
- The same field decides whether lanes are worth a kernel BOUNDARY (gh-ocannl-1167, config
  `gpu_serial_lanes`: `cut` where true, `admitted` where false, or `off`). On gfx1151 the stage-1
  lanes split by nest: the forward value pass, alone in its kernel, wins on lanes (0.74 vs 1.12 ms
  per layer at s1024; its plain plan walks a whole row per thread), while the fused dV loses -- not
  on its own, but because `keeps_mapping` cut it from dK to keep its lanes (dK 1.03 + dV 2.30 ms,
  against 1.72 ms for the two merged on the plain plan). So `admitted` takes lanes wherever a
  kernel admits them and has `fission_keep_mapping` judge merges between nests WITH a lane path
  on the plain plans (`mapping_probe`'s `mp_lane_trade`): no boundary is kept for lanes between
  them, so dV joins dK. The evidence is that dK/dV trade only, so a merged kernel that would also
  hold a loop nest with NO lane path (`Schedule.laneless_nest`: a row statistic sharing the value
  pass's plain chain) must pass the lane-inclusive probe too -- a segment's statement may keep the
  mapping it already has there, so a nest the trade already took off its lanes loses nothing more
  (gh-ocannl-1239; per-nest lanes in a mixed kernel was the costlier alternative: the coverage
  rule would need a guarded lane slot for the plain nest).
  Gating lanes wholesale was measured and rejected (it threw away the forward win). Measured on
  gfx1151 and gfx1102: training 4-10% over `cut`, inference unchanged. Leg 7 of `gpu_serial_lanes`
  pins both halves, the training step's own forward included; leg 8 the row-statistic cut. `cut`
  keeps a boundary only where a merge would cost a nest its lanes, never "dV runs apart": where lanes
  are cheap dK takes them too (the cooperative preamble), on dV's topology, so dQ, dK and dV share
  one lane kernel on Metal and CUDA (leg 7's executed-step claim).
- **A contraction inside a scan body is tensorized by rewriting the whole scan's owner, not by
  `Tensorize`** (gh-ocannl-1003, `Schedule.Fold_mma`): `rewrite_loop` does not enter a scan, and
  a lane loop minted inside the body would take a second `Workgroup` slot under the row loop. The
  online-softmax fold's shape is lane = query row: the row loop around the scan becomes the
  `Workgroup` lane loop, each lane runs its row's scan in lockstep (carried state stays per-lane
  scalars), and each contraction becomes one block `Tile_mma` whose `lane` is that outer loop,
  between explicit barriers (a `Tile_mma`'s leading bracket is form-dependent, see below).
  `validate_parallel` and the renderer take this unchanged; what they refuse is a guard around a
  barrier, so query and key tails keep the scalar form. It is GPU-only: a `Workgroup` loop
  enclosing barriers has no serial rendering (cc would finish lane 0's scan before lane 1 starts).
  Read the per-statement census (`Context.routine.mma` renderings), not the aggregate, to show both
  contractions tensorized (`test/operations/online_softmax_block_mma`). Measured in
  `benchmarks/report-gh1003-block-fold.md`: on Metal the attention's per-layer kernels go from
  3.2-7.7 ms (composed, seq 128-1024) to one kernel of 0.36-0.82 ms, both contractions
  `Mma_intrinsics` in every layer; the key block 8/16/32 is within noise at seq 128-512 and 16
  wins at seq 1024. `approximate` selects `online_softmax_block=auto` (gh-ocannl-1171):
  the device limits' `online_softmax_auto_block` keeps 16 on Metal and CPU,
  and 0 (the two-pass rewrite) on CUDA, HIP and unmeasured targets. CUDA's block fold loses
  on the tuned f32 transformer even with tf32 (gh-ocannl-1194); the two-pass form retains the
  tensorized matmul gain ([ablation report](../../benchmarks/report-gh1194-cuda-approximate.md)).
  Clean cached three-arm confirmation (gh-ocannl-1184, runtime `3858b8a5`) measured the
  tuned f32 forward on
  gfx1151 **unified** memory at 7.07 ms with 16 against 4.76 ms with auto (exact 4.72 ms),
  and gfx1102 **discrete** at 9.56 ms against 6.30 ms (exact 6.12 ms). Full unpruned
  searches and separate cache audits had no timing refusals; three Latin-square replay
  sweeps per device confirmed that 16 loses on both. Explicit
  integers and `set_block` force their size. The limits travel through backend compilation
  AND analyze-only lowering before the rewrites; the resolved code retains the existing
  `Code_borne` cache classification. Backend-free lowering conservatively resolves auto to 0.
  In training the step is backward-bound
  (gh-ocannl-1124): the fold moves it by 1-2%.
- GPU fission folds a reduction's whole-node zero into the reduction's kernel (gh-ocannl-1175) as a
  pass over the FINISHED segmentation (`group_units`'s `fold_zeros`): a singleton zero segment
  folds, expanded by the zero policy's `Expand_zero` (GPU, above its size threshold), into the
  following segment only when that whole segment re-admits every unit with the per-cell zero at
  its head. The fold must change nothing else. Expanding before grouping (staging#934, reverted)
  let the zero join the PRECEDING segment and bridge sibling reductions its whole-node form had
  kept apart: each GPT2 layer's q and k projections merged into one kernel with two matmul sites,
  which no sketch reaches, and the tuned CUDA step doubled. `qkv_zero_init` pins the segmentation
  as the unexpanded one with zero segments folded forward; seed validity alone did not catch it.
  Sketch families give the folded zero the site's geometry through `companion_geometry` when the
  target is materialized; companion coverage counts only materialized writes, so a target the
  tensorized fused twin places in workgroup-shared storage (`epilogue_shared`, an intermediate whose
  only consumer is the fused tail) is annotated by the pipeline itself (`shared_folded_zero`,
  gh-ocannl-1453) — bare, it failed gh-ocannl-959's binding legality on every such twin. A test of
  the folded path takes the tuner's own segmentation, `Ll_test.tuner_segments` (over
  `Autotune.sketch_segmentation`, gh-ocannl-1483; `tuner_transform` compiles through it as a
  per-segment candidate does: derived on a copy, replayed with the test's schedules), never a
  hand-rolled `fission_scheduled`: `qkv_zero_init`'s old copy promoted no locals, left that fold a
  whole-node `Zero_out`, which takes the whole-node expansion, and passed on the unfixed code. A
  test-supplied schedule is a REPLAY of the derived plan, as in the tuner: a per-segment `preset`
  in one pass can change the cuts, since coalescing reads which segments the preset annotated. The
  tuner and model selector explore a singleton segment whose pre-schedule digest the fold changed,
  since whole-routine sketches see the original whole-node zero. The fold is OPT-IN
  (`fission_scheduled ~fold_zeros`): only the sketch candidates and their seeding pass it. The
  untuned default keeps the separate zero kernel because an untiled accumulation with its zero
  register-forwarded measured 7% slower on HIP gfx1151 (unified memory), although its emitted
  source differs only in that forwarding.
- A whole-routine sketch expands its own site's whole-node zero with the site's geometry
  (`zero_geometry` for matmuls; `conv_zero_geometry` for the GPU conv leg, gh-ocannl-503), which is
  the nest and geometry the fold above gives the per-cell zero. So where one fission segment holds
  the whole routine, its sketch candidates compile to their whole-routine twins' digests and dedup
  (`schedule_conv_gemm`'s `cvu` pins `fiss_sketch_timed = 0` on metal): count the whole-routine
  candidates before reading a zero per-segment timing count as a regression. The dedup holds only
  while the twin's digest is in the dedup set: a window refused for any cause, cut as hopeless or
  stopped by the budget drops it, so on a loaded host the per-segment twin is retried and timed
  (gh-ocannl-1484). A claim that counts dedups gates on the report (`timings_contended`,
  `timings_hopeless`, `budget_cut` all clear) and skips as `Environment` otherwise; the
  forced-refusal control is `with_contended_test_windows` around the tune. The expanded zero's
  loops follow the output's AXIS order (`Expand_zero`), the site's chain its NEST order; the conv
  expansion reorders the zero nest through `c_d_syms` before `annotate`, since thread identity
  across nests is positional.
- A GPU schedule must cover EVERY materialized-writing nest of the routine, not only the one the
  pipeline builds. Launch dimensions are kernel-global, so `Low_level.validate_parallel` rejects any
  companion write (a bias/relu tail; the elementwise statements an aligned-merged fission segment
  carries) not nested under loops covering every active `(kind, slot)` pair — and on GPU there is no
  all-serial fallback, so the whole candidate fails to compile. Do not relax the rule: an uncovered
  dimension means every hardware index executes that write. Annotate instead, from
  `Schedule.aligned_chains` — the default annotators' cross-nest analysis exposed as data (which
  loops may carry geometry, already trimmed so that chain position *k* denotes the same thread
  coordinate in every linked nest), which lets a pipeline supply its own per-position geometry while
  alignment stays `schedule.ml`'s rule. A pipeline that instead leaves companions bare and counts on
  `Fuse_epilogue` absorbing them has NO surviving form when the fusion declines; that cascade
  (gh-ocannl-521) had every GPU backend seeding tensorized candidates in bulk and timing none of
  them. Residual, shared with the shipped zeroing geometry: a tensorized nest's workgroup slot is the
  opaque `Tensorize` lane, so a per-lane companion reads cells other lanes of the same simdgroup
  produced. These exchanges rely on the intrinsic's trailing workgroup barrier and the lane-0
  fallback's bracketing barriers, including device-memory ordering for distributed zeroing before
  store-back. Metal's `barrier_syntax` fences both device and threadgroup memory (gh-ocannl-963),
  as CUDA/HIP's `__syncthreads()` does; one simdgroup alone is not a memory-ordering guarantee.
  Cross-statement dependence analysis over barrier regions remains unimplemented (gh-ocannl-963),
  so this reliance does not establish legality for arbitrary hand-built or `Retype`d schedules.
  Query the analysis at the SITE'S arity (`aligned_chains ?max_chain`,
  default 2 = the presets' Grid+Workgroup shape): a batched matmul's chain is batch loops + row +
  column, and under the default cap a rank-3+ site can never match its full chain, so every seed for
  such a site declines on companion coverage — that single decline held gpt2_mini's five FFN-class
  kernels at a 1024-thread launch, 70% of the CUDA step at 1.3% of fp32 peak (gh-ocannl-569). A
  companion that reduces OVER the site's minor axis (the lm_head's max-logits row) trims the common
  prefix below the site's arity and correctly still declines — that one needs fission, not coverage:
  `fission_scheduled ~arity_cuts:true` (gh-ocannl-574) cuts it apart — measured on HIP/gfx1151 at
  1.30x on gpt2_mini (`report-gh612-hip.md`). **Size a fission's payoff over the WHOLE CHAIN, never
  over the fragment that keeps the site's name**: post-fission the mask, row-max and softmax work runs
  in separate downstream kernels, so dividing a standalone QKᵀ by a fused QKᵀ+mask+row-max reports a
  meaningless 5.2x. Like-for-like, summing every fragment on both sides: the lm_head/CE chain goes 4
  kernels / 8.136 ms → 5 / 0.357 ms (22.8x) and the four QKᵀ chains 8 kernels / 3.666 ms → 16 /
  2.038 ms (1.80x), so the QKᵀ sites are ~17% of what the two freed line items give against the
  lm_head's ~83%. Anchor the CE chain on `logits`, NOT on `wte`: the input token-embedding gather
  reads `wte` too (the embedding is `wte * onehot_x`) and is not part of the head. **The finer fission also COSTS +2.57 ms in the FFN bucket** -- it splits
  residual adds into more separately launched kernels, each re-deriving the running sum -- which the gh-573 fanin guard is what
  recovers, so the two must be measured together or each is mis-attributed. Why such a
  pair merges in the first place: the fission pass's no-parallelism-loss guard compares chains under the presets'
  `max_chain=2` cap, so trimming a rank-3 GEMM's minor axis reads as lossless; and a max-reduce is
  the shape that hits it because its `-inf` init is a `Set` nest, not a `Zero_out` — a sum-reduce's
  `Zero_out` already separates the statements. The arity_cuts mode analyzes uncapped AND requires
  merged nests to share one extent list (the init nest is conflict-free with the GEMM, so a pure
  no-loss rule would still merge it and companion coverage would still decline on it). It is a
  candidate-generation mode — the autotuner seeds fine-flagged per-segment sketches when the finer
  segmentation mints new digests, and a fine winner's cached segmentation is the fine one
  (gh-ocannl-1164) — never the default pipeline, which would pay the extra launches
  unconditionally; the default cuts only where a merge costs a nest its mapping (gh-ocannl-1126, above). Since gh-ocannl-577 the
  coverage verdict is also a construction-time refutation in the matmul family tree
  (`matmul_coverage_witness`): this is sound because `companion_geometry`'s Ok/Error never depends
  on the geometry its `annotate` callback emits — only on the lowering, the site chain, the fused
  flavor's `skip` and the zeroing expansion. If you ever make the verdict consult the emitted
  geometry, the static witness goes stale and the tree will refute families whose candidates
  would build (or vice versa) — keep the invariant, or re-derive the witness. The conv family's
  GPU staged leg asks the same verdict (`conv_coverage_witness`, gh-ocannl-1406) where it used to
  cap the lowering at two real statements besides the site's zero: the cap refused every
  aligned-merged conv segment (conv + relu/bias/pool, 3-6 statements) although the pipeline
  annotated each companion, and admitted single companions it could not cover. Not yet
  tree-factored, the conv leg seeds when EITHER flavor is covered (its fused twins are flag-flipped
  from every seed), so where only the fused flavor is, the unfused seeds decline at build and the
  leg's notes say so; `conv_merged_gpu_seeds` executes every seed of the merged shape. The fused
  (`Fuse_epilogue`) flavor is judged separately: skipping the epilogue tail can empty the
  coverage demand before the alignment analysis is consulted, so twins can survive a routine the
  unfused family is refuted on (a `Local` product whose consumer shares its kernel). Until
  gh-ocannl-1325 only the TENSORIZED twins compiled on GPU: `Fuse_epilogue` refused the scalar
  block-tiling pipeline's `Privatize` store-back, which sits under the register split's
  `Workgroup` loops and, at a padded geometry, under its pad masks, so every `gpu …/TMxTN ep`
  candidate was a wasted compile and such a segment had no sketch at all on CUDA at the default
  `tf32_matmuls=false` or on HIP for f32. The store-back fusion now admits `Workgroup` write-path
  loops (the tail runs per element after the thread's own store) and per-axis range guards in
  `guard_axis`'s form (the tail lands inside them); `epilogue_fusion_mma_seeds` executes every
  scalar twin, and `autotune_sketch_placements`' differ leg claims the first twin's single on
  every GPU backend. Since gh-ocannl-613 the flavor is the family tree's ROOT level
  (`fusion = unfused | fused`, `matmul_family_tree`): the fused child is refuted with the
  recognizer's own reason (`Schedule.fuse_epilogue_witness`) on a site with no fusable tail —
  so a tree's `refutations` always carries one entry more than its pipelines produce, and a
  test asserting "every refutation is X" must scope to the `Family_decision.Fusion `Unfused`` path — and
  the unfused coverage verdict is shared with the fused branch because it implies it (the fused
  demand is a subset, and `aligned_chains` ignores `skip`): `aligned_chains` runs twice only
  where the unfused flavor is refuted. `sketch_seed_params` is the tree's `leaves`, twins
  included, and `model_default` has no separate twins step. The conv family is a tree of its own
  under the same root since gh-ocannl-697 (`conv_sketch_tree`: `conv-leg = cpu | gpu`, the CPU
  `conv-shape` menu, the GPU `conv-staging` above `conv-rows`, so the leaves keep the flat list's
  plain-geometries-then-depth-twins order for any depth list); `sketch_family_tree` is whichever
  family claims the lowering, under a `family = matmul | conv` root since gh-ocannl-1488 (each
  family judged by its own detector, the other refuted with `no_matmul_site_witness` /
  `no_conv_site_witness`; the per-family trees and `family_seeding`'s trees carry no such level,
  so a path read off the combined view starts one level deeper), and its census lines are the
  tree's refutations, path by path, where the conv legs used to keep free-text notes. `model_family_search` walks it too, so a conv
  site gets the family bound, the stats ledger and the operand-load tie-break: where `model_default`
  used to scan conv seeds flat (`best_flat`, first strict improvement), an exact score tie between
  two conv leaves with tile-MMA load counts can now crown the second one. The conv traffic-floor
  increment (gh-ocannl-1489, `conv_path_traffic_floor`) prices both staged operand tiles, written
  and read, once the row geometry commits (`conv-shape` / `conv-rows`; every prefix above prices 0,
  so monotonicity needs no copy of the menus' feasibility): each `Stage` mints a FRESH tile node
  that the load nest fills and `Tile_mma`'s fallback reads in full, and `analyze` charges a node at
  least its distinct cells, so the charge is additional to the base nodes. Widths are the tiles' own
  (`cpu_conv_pack_prec`, shared with the tree, on the CPU leg; storage widths on the GPU leg, whose
  reduction and column extents pad to the format tile); a pipelined tile keeps its logical dims, so
  one copy. `conv_family_tree` checks it against every leaf's scheduled `analyze` (fresh nodes, the
  composed floor at every prefix, the bound against the ranking score): on every fixture the two
  `Stage` tiles are charged EXACTLY the increment, and the fresh-node slack is `Tensorize`'s
  accumulator fragment (`rows x noc_p` at the output's width, written and read), which the floor
  leaves unpriced — a tightening once its certainty is argued. **Its pruning power is
  capped by the base floor, not the increment**: `completion_floor` prices an overlapping-window
  input read at 0 (the image is not exact), so a real leaf's traffic sits about the input's size
  above floor + increment — more than two leaves' A tiles differ by — and with the incumbent at a
  real score nothing below the root fathoms; the test's pruning claim therefore puts the incumbent
  between the floor and the largest raised bound. A sound lower bound on a windowed image (an
  affine sum's interval) is what would let conv subtrees fathom at real incumbents. **Consequence for diagnostics: since gh-577 a companion-coverage
  DECLINE CENSUS is empty, and that is not evidence the rule stopped firing** — a refuted family is
  never seeded, so it never reaches the decline log. gh-569's Part 3 read 25 coverage declines out of
  `schedule_log_declines`; the same workload on current master logs zero
  (`report-gh612-hip.md`). Ask the emitted source and the launch geometry instead.
- **The same rule binds the CPU tensorized flavors that bind a hardware dimension** (gh-ocannl-1561):
  the whole-triple `Tile_mma` form's `Workgroup` lane (its zeroing's column loop at the full column
  extent, so always active; plus a `Grid` row split at `sk_bm > 0`) and the packed Grid shapes'
  row-block loop. No CPU builder gives a companion geometry, so the tree refutes those flavors
  wherever the companion demand (`companion_demand`, fused tail skipped) is non-empty
  (`cpu_mma_companion_writes`), and the all-serial shapes -- packed Serial/Hoisted, the scalar
  pipeline -- carry the search there. A label's `0x0x0` is the whole-triple form (unsplit row), not
  a zero-size tile: it builds wherever fission leaves the matmul alone (`F_sketch`), and failed
  only in whole-routine searches beside a materialized companion (tune_footprint_default's
  materialize-all arm). Untried alternatives that would keep the form: aligned companion geometry
  as on GPU, or a unit lane (`simd_width:1`, as the packed and conv forms use) for `bm = 0`.
  `test/operations/cpu_mma_companion_coverage`.
## Family decisions and launch geometry
- **The family tree's decisions are DATA, and reading one back means matching on it — never
  re-parsing a label** (gh-ocannl-591). `Ir.Schedule_space` is parameterized over the decision type
  (`('l, 'a) tree`, paths `(string * 'l) list`); the matmul family instantiates it at
  `Autotune.Family_decision.t`, one constructor per level carrying the geometry, the lattice
  interval, the pipeline depth, the packing shape. `Family_decision.level` derives the level name
  from the decision and `to_label`/`render_path` are the rendering, so renaming a level or
  rewording a label is a display change with no consumer behind it. Before this, levels minted
  labels with `sprintf` and `sketch_path_traffic_floor` read them back with `sscanf` under
  `try … with _ -> None`: any reword made every arm fall through, so the certain-traffic increment
  was `0` on every path — a SOUND lower bound, so nothing raised, no golden moved, and the family
  bound silently degraded to the schedule-invariant floor. If you add a level, add its constructor;
  if a consumer needs to know what was committed, match the datum. The same shape applies to
  `model_default`'s placement tree, whose children carry `(flip_candidate, `Keep | `Flip reading)`.
- **A test that prices decision paths must walk them out of the tree, not write them down.**
  `sketch_family_tree.ml` used to call the traffic floor with literal paths, which pinned its own
  parser and let the tree mint anything. It now enumerates the real tree and asserts that every
  leaf's increment equals the traffic the LEAF'S OWN `sketch_params` imply (independent of the
  path) and that the increment is monotone along every prefix — so a commitment a consumer stops
  reading is a mismatch instead of a uniform zero, and the golden's leaf-path counts and level
  inventory move when a level is added or removed. Verified both ways: rewording three labels moves
  190 rendered lines and no number; deleting the `twin` level moves the counts and fails.
- Batch loops of a GPU matmul sketch are no longer unconditionally `Serial` (gh-ocannl-643, the
  rank-4 q/k/v residue of gh-569: `36.7%` of the gpt2_mini step at 10% of sgemm peak because a
  `(batch, head, seq, head_dim)` site launched 4 blocks with batch and head serial inside). Two
  mechanisms, layered: (1) `Grid` slots >= 2 are legal and FOLD onto the hardware `.z` dimension —
  `Low_level.launch_dims` multiplies their per-slot maxima into `grid.(2)` and each folded loop
  binds `(z / stride) % cap` (`Low_level.grid_fold`; rendered in `C_syntax.hardware_binding`,
  degenerating to the bare `.z` register for a lone slot-2 loop, so pre-existing kernels emit
  byte-identical source); the 3-slot cap now applies to `Workgroup` only, and cc's serial fallback
  is untouched. (2) The GPU matmul sketch families seed each geometry in two batch flavors
  (`sk_batch_grid` twins, a "batch" tree level above "geometry"): batch positions `Retype`d to
  `Grid` vs. the historical `Serial` — TWINS, not a replacement, because the device block-count
  curve is non-monotone (gh-569's probe), so the tuner measures both. Three traps encoded in the
  implementation: the zero nest and every companion nest must carry the SAME per-position batch
  annotation with interior (`m_bi`) batch loops hoisted identically (`companion_role_ops`,
  `zero_geometry ~batch_grid`) or positional slot order diverges between nests and a thread zeroes
  cells another thread accumulates; `companion_geometry`'s `annotate` callback therefore takes the
  companion's whole chain (the hoist `Swap`s name the companion's own symbols) — its Ok/Error
  verdict still never depends on what `annotate` emits, so the gh-577 static witness stays sound;
  and batch products beyond the device's `.z` cap are never seeded — one dimension of the launch
  predicate below, not a filter of its own.
  The pre-driver gate (`Schedule.check_hardware_limits_classified`)
  covers BOTH 16-bit grid dimensions against the single `hardware_limits.max_grid_yz`: `grid.(2)`
  (the fold) and `grid.(1)` (the row-block count, which overflows on m-extent alone — no batch axis
  involved). One limit field, two typed resources (`Grid_y_extent` / `Grid_z_extent`), since the
  rejection key is what an autotune search groups declines by and the fixes differ.
  Pinned end-to-end by `test/operations/schedule_batch_grid.ml` (structure everywhere, execution
  and emitted-source fold on GPU backends).
  A third flavor, `bgrid-in` (`sk_batch_inner`, the `batch-grid-inner` decision of the family tree's
  batch level, gh-ocannl-728), exists only on sites with INTERIOR batch loops (`m_bi`, the q/k/v projections'
  heads): the interior loops hoist above the row's in-block remainder instead of above `m_i`, so
  the grid nest is `m_bo; row blocks; m_bi; column blocks` — heads on `.y` beside the column
  blocks, row blocks folded with `m_bo` onto `.z`, i.e. the launch order of the heads-merged
  layout. One layout type (`batch_layout`) drives the site, zero and companion nests; the row
  geometry comes in two halves (`row_parts`: block split / remainder / register split) so the
  hoist lands between them. Trap: a NON-dividing `Split` wraps the remainder loop's body in its
  guard, which `Swap` cannot pass, so the zero and companion nests `Pad` the row loop first under
  this flavor (the site nest is padded by the pipeline at such geometries anyway). A menu change
  like this one bumps `Schedule_cache.entry_version`, appending its history line: a stored crown
  is only the best of the menu that searched it.
  The gate covers the WORKGROUP's dimensions the same way (gh-ocannl-679):
  `hardware_limits.max_workgroup_dims` is an `(int * int * int) option` of per-dimension caps
  beside — not instead of — `max_threads_per_workgroup`, which caps only the thread PRODUCT.
  A tuple, not an array: every GPU backend memoizes its `hardware_limits` behind a `lazy` and
  `Context.hardware_limits` returns that record itself, so ONE mutable cell anywhere in the record
  would let a caller deriving tighter limits write through into the process-wide singleton. Keep
  the record free of mutable cells when adding fields — it is what makes handing out the memoized
  value safe, and `max_workgroup_dims` was the only field that ever broke it. The two are different hardware
  facts: CUDA's `maxThreadsDim` is `(1024, 1024, 64)`, so a `2 x 2 x 128` workgroup is a legal
  512-thread product and an invalid launch configuration. `Workgroup` slots cap at 3 and the
  innermost binds `.x`, so the outermost annotated loop's extent lands on `.z` directly; no fold is
  involved. Filled by all three GPU backends (CUDA queries `max_block_dim_{x,y,z}`, HIP the
  `max_threads_dim` triple, Metal all three components of `maxThreadsPerThreadgroup` — it used to
  read `width` alone), `None` on the C backends. `Schedule.default_gpu` and
  `Schedule.zero_expansion` clamp their block size against the `.x` entry too, so the gate is a
  backstop; a lane plan widened past one `Workgroup` loop (`gpu_schedule_workgroup_fill`) checks
  the `.y`/`.z` entries it reaches itself (`plan_nest`), which is why no in-tree annotator
  reaches the `.z` cliff and why `test/operations/launch_dim_gate.ml` builds that geometry by
  hand.
  **The gate is now one table, five rows** (block `.x`/`.y`/`.z`, grid `.y`/`.z`), not a
  hand-written `Option.iter` per bound — each bound used to be a copy of its neighbour, which is
  how `gridDim.y` went ungated for a release and how the workgroup dimensions went ungated
  entirely. `grid.(0)` is the deliberate sixth absence: 2^31-scale wherever hardware axes bind.
  Adding a cap means adding a row.

- **One predicate for the launch caps, consulted by the gate AND by seeding** (gh-ocannl-709).
  Those five rows live in `Schedule.launch_geometry_excess : limits:hardware_limits ->
  launch_geometry -> launch_excess option`, not in the gate. A `launch_geometry` is the five capped
  dimensions as `int option`s (`None` = "this caller does not predict it", which EXEMPTS the
  dimension rather than refusing on it); `launch_excess` carries the typed
  `Schedule_outcome.resource`, the requested extent, the limit, and `lx_phrase` — the verb phrase
  both callers render, so a gate `detail` and a seeding refutation witness are the same sentence
  about the same candidate. The gate fills all five from the lowered code
  (`launch_geometry_of_dims (Low_level.launch_dims llc)`); autotune's matmul family predicts them
  from the parameters (`Autotune.matmul_launch_geometry`) and refutes at the leaf. The workgroup
  thread PRODUCT is deliberately NOT a row: it is not a per-dimension geometry question, and only
  the gate asks it.
  Why it matters: before this, seeding pre-filtered exactly ONE of the five (the `.z` fold, in
  `batch_grid_twin_ok`'s own copy of `max_grid_yz`) and a search learned the other four one wasted
  GPU compile at a time — while the one it did filter was a second encoding of a cap the gate
  already held. `batch_grid_twin_ok` is now structural only ("are there batch loops worth a decision
  level"), so an over-cap fold refutes at the leaf **with a reason** instead of the twin level
  silently vanishing.
  Two traps when adding a family or a dimension. (1) A prediction must be a LOWER bound: it
  describes the site's own nest, while `launch_dims` maxes over the zeroing and companion nests too
  — an under-prediction costs one compile the gate then declines, an OVER-prediction silently
  withholds a legal candidate, which is the worse failure. `test/operations/launch_predicate_parity.ml`
  therefore cross-checks the prediction against every GPU seed's applied `launch_dims`, per
  dimension, alongside the seed/gate parity claims (each with its at-the-cap negative control).
  (2) Slot assignment is positional from the INSIDE out and is encoded once, in
  `Sketch_families.predicted_launch_geometry ~grid ~block` (extents in nest order, outermost first):
  the innermost same-kind loop binds `.x`, the next `.y`, and `Grid` loops beyond the second fold
  their PRODUCT onto `.z`. For the GPU matmul pipelines that makes the column blocks `.x`, the row
  blocks `.y` and the batch fold `.z`; the blocktile's two `Workgroup` splits put `bn/tn` on `.x`
  and `bm/tm` on `.y`, while the mma pipeline's lone tensorization lane is `.x` alone.
  Seeding saturates an unadvertised `max_grid_yz` to the conservative `max_grid_fold_extent`
  (65535) so the seed set does not swing with the machine; the GATE does not — there, an
  unadvertised cap is genuinely no cap.
  The CONV family is wired too (gh-ocannl-739): `conv_launch_geometry` describes the outer output
  `Grid` loops followed by the blocked flavor's row-block loop (the innermost grid coordinate), plus
  the tensorization lane as the sole `Workgroup` loop. Every conv leaf passes that lower-bound
  prediction through the same predicate (`launch_geometry_refutation`), so since gh-ocannl-697 an
  over-cap conv geometry is refuted at its own path with the gate's sentence instead of filtered. `launch_predicate_parity` derives a real conv site behind
  `fission_scheduled`, applies every GPU seed, and proves no predicted dimension exceeds
  `launch_dims`. Its negative controls share one conv whose outer grid loops are a 65536 batch
  and a non-row spatial 2: blocked, the row blocks bind `.x`, the spatial extent `.y`, and the
  batch folds ALONE onto `.z`; unblocked, the batch lands on `.y` and nothing folds. Each claim
  names the dimension, the requested extent and the cap (gh-ocannl-939) -- a resource-only claim
  ("some seed is refused on `.z`") holds for a wrong reading of the geometry too -- and each
  flavor is the other's control: the `.z` claim must fail on the `.y` refusals and vice versa,
  and on the matmul twin's `.z` refusal at another extent.

  The GPU backends' `static_properties` dumps list the queried launch-dimension limits next to
  `max_threads_per_block` — HIP `max_grid_size` and `max_threads_dim`, CUDA `max_block_dim` and
  `max_grid_dim`, Metal the `max_threads_per_threadgroup` triple — so a run on hardware can read
  back what those gates compare against; without them the only evidence a query is not degenerate
  is that no kernel got rejected, which is also what a query returning 0 would produce.
  **`bin/device_props` is the supported way to read them** (gh-ocannl-684): it prints both
  `static_properties` and the derived `hardware_limits` for the selected backend, one
  `path = value` line per fact, and compiles no routine. Do NOT reach it through `dune exec` (the
  `bin/` cwd trap): `dune build bin/device_props.exe`, then run
  `_build/default/bin/device_props.exe --ocannl_backend=<name>`, pinning the backend explicitly —
  with none configured `Context.auto` walks metal -> cuda -> hip -> cc and would report a device
  other than the one being asked about. Local readings: Metal on an M4 Max reports
  `max_threads_per_threadgroup = 1024 1024 1024`, so Apple parts cannot exercise the per-dimension
  cliff either. On gfx1151/ROCm/WSL2 the HIP values read `(2147483647 65535 65535)` and
  `(1024 1024 1024)`, i.e. `max_grid_yz = 65535` and a `max_workgroup_dims` that equals the product
  cap — that device cannot exercise the per-dimension cliff; CUDA's `.z` of 64 is the one that
  can.
## Contraction detection and strided axes
- `detect_conv`'s boundary is SINGLETON axes, not rank (gh-ocannl-912). 3-D, 1-D, batchless and
  multi-batch convs are detected and seeded on both legs; what it refuses is any extent-1 axis,
  because lowering drops the loop and indexes the axis at `Fixed_idx 0` while the matcher wants
  plain iterators on the output and kernel: batch 1, output extent 1, one input channel (lenet's
  conv1), a k-by-1 window. A 1x1 window is a GEMM and goes to the matmul family. A conv fixture
  that "is not detected" usually has a singleton somewhere — keep every axis at extent >= 2 unless
  the singleton is the point. `test/operations/conv_detection_boundary` pins each class with its
  reason, derived from the lowered maps.
- **A strided 1x1 conv's GEMM row is the taller of `b` and `ow`** (gh-ocannl-1076, gh-ocannl-1148,
  gh-ocannl-1397): `classify_matmul` has two row candidates, the deepest write loop A reads at unit
  coefficient (`Sketch_families.unit_axis`; the batch axis of a stride-2 1x1 conv) and the deepest
  one it reads as a single strided term (`ow`), and picks the strided one where no unit candidate
  exists (batch 1) or where it is strictly taller (batch 2-8 under a 16-wide `ow`); a tie keeps the
  unit row. With the batch row (cifar_resnet's b64), `oh, ow` are interior batch loops (`m_bi`) and
  every seed stages dense tiles from a strided source address. With the strided row
  (`m_row_stride = 2`, the batch axis and `oh` in `m_bo`) every pipeline must pack A through a
  compacting `Stage` (`Schedule.stage_compaction_factor`: the gcd of the tile part's
  coefficients, so the blocktile's twice-split row `2*tm*i_w + 2*i_t` compacts too), and the
  family tree refutes the shapes that read A in place (unstaged full-K mma, whole-triple,
  hoisted-grid) or pack it at link time (`site_hoistable`: hoisted staging refuses compaction).
  A short batch row divides few block sizes: before the extent rule cifar_resnet's stage-2
  shortcut at batch 2 seeded one whole-routine geometry on cc, three with the strided row. Inside the full block (at 64 channels; at 16 the shortcut's conv is virtualized
  into its norm and is no site) cc does not fission it, so its whole-routine seeds are the
  shortcut's only sketches there; GPU fissions it and the segment's seeds run.
  `test/operations/schedule_strided_1x1` uses the public `resnet_block` with explicit
  `out_channels` (gh-ocannl-1146) and PINS the role structure on both sides of the extent rule
  (batch 1 and 2: strided; batch 8 over an 8-wide row: the tie, batch) and every seed's parity
  (or typed decline, in the block, except cc's whole-routine seeds, which must run); which seeds
  decline where it only reports on stderr, and the 16-channel virtualization is a manual
  observation (conv2d into batch_norm2d, 16 channels) that no test exercises.
  `benchmarks/runners/ocannl/bench_strided_1x1` times the site alone (roles, seed count, untuned
  and tuned arms, exact oracle), a build on each side of a classifier change making the A/B.
- **`Stage` stages only the reads that vary with its tile loops** (gh-ocannl-1147). A whole-routine
  matmul seed stages its A operand while another nest reads the same node at another index vector
  (the 3x3 conv or max pool beside a strided 1x1 shortcut, the K and V projections beside Q).
  `apply_stage` remaps only the reads whose index vector mentions a tile loop, which must still
  agree on one vector (v1), and leaves every other read on the source: the source is never written
  in the routine, so staging is a copy, not a move. On cc, which does not fission such a routine,
  a refused second reader declines every whole-routine seed and the tuner times no sketch for the
  site. Per-reader staging (a tile per distinct vector) is the unbuilt alternative, for when a
  second reader is itself hot.
- **`Privatize`'s look-alike v1 refusal stays, because nothing reaches it** (gh-ocannl-1395). It
  scans only its own loop's subtree, where a sketch's perfectly nested accumulation and the
  privatized preset's self-reading `Set` leave one index vector; two censuses (tuned benchmark
  cells, the tuning test corpus) never saw it fire. Its target IS written under the loop, so the
  gh-ocannl-1147 move does not carry over: a second accessor would need proved disjointness or a
  remap into the tile. A search's privatized presets report what they drop as
  `report.privatize_drops` (gh-ocannl-1438), aggregated by cause like `declines` but kept out of it:
  a drop costs a proposal inside a candidate that still compiles, and counting it there would break
  `declines`' sum to `candidates_failed`. `Autotune.privatize_extension` returns the drops for
  segments no search compiles, and `privatize_census` is the tripwire that says when a workload
  needs the relaxation. Nothing in the corpus reaches a real refusal on cc, so
  `test/operations/autotune_report_drops` injects one through `Autotune.on_privatize_proposal`,
  which raises inside the extension's refusal boundary.
- **A per-segment candidate's executed-parity reference is its own segmentation on the default
  preset, not the unscheduled form, wherever the values are inexact** (gh-ocannl-1217). The
  candidate schedules every OTHER segment by the default preset too, so against the unscheduled
  form it is also charged with that preset's reduction orders. `resnet_block`'s inference-mode batch
  norm divides by `sqrt` of the stored variance, so the block's outputs reach ~2e5, where an f32 ulp
  (1/64) is above an "exact" 1e-3 tolerance; on HIP every shortcut seed then "diverged" identically by
  5 ulps. The signature to recognize is a divergence that does not depend on the seed.
  `schedule_strided_1x1` compares each seed exactly against `run_default_fission` and checks that
  reference against the unscheduled form up to f32 rounding. Exact-integer fixtures (its bare
  einsum legs) can keep the unscheduled reference.
- **Backprop's contractions reach the matmul family only through the enabling interchange**
  (gh-ocannl-1183, `Sketch_families.detect_matmul_canonical`). A weight gradient lowers as
  `for b, s, o, i: dw[o,i] += dy[b,s,o] * x[b,s,i]` (contraction loops outermost) and a data
  gradient with its contraction loop between write loops, while `classify_matmul` reads the
  contraction nest off the innermost end: before the interchange, not one of the gpt2_mini training
  step's backward contractions was a site, so the search never seeded a tile for the kernels that
  are 76% of the batch-256 step. The canonical form sinks the contraction loops below the write
  loops by `Op_legal` adjacent `Swap`s (the gh-537 split-reduce precedent), re-detects, and every
  schedule built from the site carries the chain as its PREFIX, so it applies to the original code
  and replays from a fresh lowering. The chain also sinks the write loop owning the output's minor
  axis below the other write loops where backprop left it above them (gh-ocannl-1325): the tied
  lm_head's `wte.grad[e,v]` and attention's `v.grad[b,t,h,d]` keep the forward product's write
  order, so the sink alone left them unclassified (in gpt2_mini's GPU segmentation, 78 of the 116
  product accumulations were their segment's site before, 83 after). A plain site keeps an empty prefix, and a segment `detect_conv`
  claims is never interchanged, so every seed that existed before is byte-identical. Every entry point that
  reads a site off `opt` must go through the canonical form (seeding, the family tree, the traffic
  floor, `mma_eligible_sites`, `sketch_schedule`); one that calls `detect_matmul` directly silently
  sees no backward site. A forward einsum spelled like a gradient (`"bso;bsi=>oi"`) lowers with
  the output loops first and is a plain site: reproduce backprop's order with an actual backprop
  (`benchmarks/runners/ocannl/bench_wgrad.ml`) or hand-built IR (`test/operations/matmul_interchange`).
  The untuned default reached these sites in gh-ocannl-1324 (next bullet). Bare `Context.compile`
  still renders every contraction with no operand reuse.
- **The seeded untuned default replays the DEFAULT segmentation, never the tuner's**
  (gh-ocannl-1324, `Autotune.seeded_default`). The tuner's per-segment sketches use a different
  segmentation: they fold each covering zero into its reduction's segment, and their other segments
  get the `min_parallel:1` presets. The seeded default instead derives `maybe_default_schedules`'
  own plan and swaps the preset for a sketch only on segments that seed. It compares that plan with
  its `arity_cuts` refinement block by block (`Autotune.refine_segments`, gh-ocannl-1413): a block
  is a maximal run of units whose two ends are boundaries in both, and it takes the finer segments
  where more of them seed. That is how a weight gradient fused with its bias gradient, or a
  projection's `w_k.grad` with its input's data gradients and `w_q.grad`, is reached (the shared
  segment's own seeds all decline on companion coverage). A block rather than a segment, because
  the finer cuts can move a boundary: attention's score product shares its default segment with
  the row max's `-inf` initialization, which the finer plan keeps with the max instead, so before
  gh-1413 gpt2_mini_train's four score products were never seeded (95 → 99 of its 99 matmul sites,
  same launch count). So every other kernel is byte-identical to the default's, and a routine with
  no seed IS the default compile. **A site census is not a seed census**: `bench_gpt_diag
  BENCH_SITE_CENSUS=1` lists each segment's product accumulations, its site, and the search's
  seed count, and its seeded-default line counts the sketches actually taken; config
  `autotune_log_seeding` names, per segment, why a seed was refused (gh-ocannl-1407). gh-1413 was filed
  from the site column alone ("only the first product site is seeded") while both pipelines already
  seeded every one of those products through the refinement; 17 of the step's 116 products are not
  matmuls even alone (row sums, the attention row max, loss terms). Two traps that cost measurable time on
  gpt2_mini. First, geometry must be padding-aware: attention's value product (32 columns) under a
  64-column tile wasted half its threads and made threshold 128 slower than 256. With padding at
  most 1/8 preferred, threshold 32 gave b8 15.4 ms (the tuned step was 15.3) and b256 565 ms
  (unseeded 1725). Second, seeding is refused where `accum_prec` widens the output's storage
  precision. A sketch's `Privatize` keeps a wide accumulator across the whole reduction, while the
  default rounds back to storage after each outer contraction step, so a narrow-storage weight
  gradient would change its result. Losses stayed bitwise identical on f32 (pinned by
  `test/operations/seeded_default`).
- **The seeded untuned default also splits long, narrow reductions, before segmenting, on CUDA
  and HIP** (gh-ocannl-1325, `Autotune.seeded_split_sites`, config
  `gpu_default_split_reduce_min_serial`; where it runs is `Autotune.seeded_split_backend`).
  Measured on gpt2_mini_train b256: CUDA 543 → 391 ms. HIP (gfx1151, unified memory) went
  885 → 1256 ms, so the split was CUDA-only until gh-ocannl-1412 gave its first pass coalesced
  lanes (the next bullet). It applies the search's split-reduce detection and enabling interchange as a whole-routine
  prelude (`Autotune.split_reduce_prelude`, the `F_split` seeds' own), to every site whose serial
  chain per output cell reaches the threshold. On gpt2_mini_train b256 these are the 26 bias and
  layer-norm parameter gradients and the loss sum, at 32768 rows each. Each split launches
  32768 first-pass threads, where the default ran one thread per parameter. The b8 step's
  parameter gradients reduce over `b, s` with `b = 8` outermost; the next bullet is how they became
  sites (`bench_gpt_diag` `BENCH_SR_SITES=1` lists the sites and the verdicts). Two traps. The prelude must run
  on a copy of the traced store only, with the compile's own placement table (as the search's
  `rebase` does); on a fully hermetic copy the partials nodes never reach the placements the
  link reads, and the compile dies in `Tnode.Placements.is_materialized_force`. And unlike the
  seeded sketches the split changes results: it reassociates those sums into a fixed balanced tree
  over blocks chosen from the routine alone, as gh-ocannl-1124's cooperative all-reduces do in the
  default plan and the tuner's own split-reduce candidates do under the default
  `autotune_search=true` (`test/operations/seeded_split_reduce` holds it to an f64 reference). The
  `reproducible` profile pins the key at 0, keeping its sums serial.
- **A split's first pass runs its block loop outside the cell loops** (gh-ocannl-1412,
  `sr_block_hoist`, appended to `Autotune.split_reduce_prelude`). `Split_reduce` puts the block
  loop where the reduction loop was, BELOW the cell loops its pinning rule requires:
  `for o: for blk: { init; for inner: for s: .. }`. The GPU preset maps the outer loop of that chain
  to the grid and the inner one to the lanes, so a wavefront's lanes walked the blocks, `chunk`
  rows apart, each loading its own cache line on every row. The suspected register-promotion cause
  was wrong: the HIP ISA of a split first pass shows the partials cell localized (one load, a
  register accumulation, one store). The kernel table showed the real cost instead: on minix
  (gfx1151, unified memory) the b256 first pass of `l1_ffn_b2.grad`, fused by fission with the
  `n512.grad` nest it recomputes inline, took 197 ms where the unsplit bias-gradient kernel took 15.
  Detection now records the cell loops when hoisting the block loop above them is `Op_legal` on the
  split code (static sites only), and the prelude appends those `Swap`s, so the lanes walk the cell
  axis as in the elementwise nests that produce the summed rows. Each cell's summation order is
  unchanged (one block accumulates one partials row in the original order), so the result is
  bitwise the unhoisted split's (`test/operations/split_reduce_block_hoist`, which also pins the
  lanes under the device's preset beside a control site without the hoist). Diagnostic runs on
  minix (b256, same binary, kernel table on): split unhoisted 1285 ms against 939 unsplit; hoisted
  630 against 945, with that first pass at 4.4 ms and the layer-norm first passes from 4-23 ms down
  to 0.3-1.1 ms. The plan digest counts each site's hoisted loops (`hoist<n>`), so a search crowned
  under an unhoisted seed is a different trajectory.
- **A two-loop reduction band neither of whose loops splits alone is split as a pair**
  (gh-ocannl-1414, `Autotune.coalesce_reduction_pair`, field `sr_coalesce`). At b8 a parameter
  gradient `for b, s, o: g[o] += dy[b,s,o]` has `b` below `sr_red_min`, and `s` fails the split's
  pinning rule because `b`, a reduction loop, encloses it. So where no single loop of a site
  splits, `split_reduce_sites` hoists the cell loops above the two outermost reduction loops,
  merges them with `Sched.Coalesce`, and re-probes the split on the merged loop; the prelude
  replays the swaps, the `Coalesce` (under the merged symbol detection minted), then the split. A
  site one loop already splits keeps that split, so b256 keeps its 27 sites and block counts. The same
  path admits a small-output contraction over two axes (the out projection's `(h, e)`, gh-1279's
  pair), pinned by `test/operations/split_reduce_pair`. Trap: the first two layers' gradients read
  the positional table `wpe[s, o]`, inlined into the residual stream, and that read has no
  flattened address, so `Coalesce` declines 7 of the 26 b8 sites. For those the pair is
  interchanged instead, `Swap (b, s)` appended to the chain, and `s` is split with `b` running
  inside each block. Second trap, latent before this: `Sketch_families.fma_form` took an access
  path's head `Stmt` as an index into the segment's top-level statements, but loops add no path
  component, so a segment that is one loop around a sequence (a split's first pass,
  `for blk { init; for inner { .. } }`) indexed past the end and raised in the seeded default's
  matmul detection; it now follows the path (`stmt_at`). These 1024-row chains are why the
  threshold's default dropped from 4096 to 1024, measured in one exclusive rog window
  (`benchmarks/gh1133_cells.sh` treatments `sr4096`/`sr1024`, same binary): the CUDA b8 step went
  15.16 → 12.59 ms, b2 s512 19.88 → 15.89, b1 s1024 29.03 → 26.29, and b256 stayed at 395 ms
  (its sites and block counts do not change). In b8's nsys traces the kernels writing a split
  target or its partials fell 4.19 → 1.89 ms, most of the 2.2 ms the step's kernels lost; the
  vocabulary-wide softmax reductions that 1024 also admits held under 0.15 ms either way. The
  block count also stays within the square root of the chain: the menu's "most threads up to
  32768" rule, tuned on 32768-row chains, would have split b8's 1024-row loss sum into 512 blocks
  of 2 rows and a 512-term combine.
- **The seeded default sinks the reduction loops of the segments it does not seed**
  (gh-ocannl-1348, `Autotune.reduction_sink`). Backprop lowers a batch reduction into a small
  output as `for b, s, o: db[o] += dy[b,s,o]`, and the preset maps `o` below the serial `b`, `s`
  loops, so the code generator cannot localize the cell: localization only peels reduction loops
  BELOW the cell's own. The sink reuses the enabling interchange's `contraction_sink_swaps`
  statement by statement, each chain `Op_legal` after the ones kept before it, and the preset then
  runs on the interchanged code. Bitwise identical because only accumulators whose write loops each
  own an axis as a plain `Iterator` qualify (one cell, one tuple of write-loop values), and a
  narrow-storage cell the backend widens is skipped, as for seeding. Its machine-level effect is
  HIP's alone (see the backend-precision note on reading the PTX/ISA first): the layer-norm
  gamma/beta nests stop round-tripping their cells, while NVRTC had already promoted every one.
  Not in `maybe_default_schedules`: the tuner's presets and the fission probes are unchanged; the
  search meets the sink only through the seeded plan's seed (next bullet;
  `gpu_default_reduction_sink`, `Search_shaping` since then).
- **The search seeds the seeded untuned plan FIRST** (gh-ocannl-1388, `Autotune.seeded_plan`,
  flavor `F_seeded`). Before it, a GPU search's pool started from the plain default pipeline, so a
  search that timed only its first candidate, or refused every later window, crowned a step the
  untuned compile had already beaten (gpt2_mini_train b256 on rog: 1720 ms against 563 ms). The plan
  is a value (`seeded_plan`: the split sites, then the segmentation and one `seeded_segment` per
  segment -- preset, sketch or sunk -- with the structural digest it was decided on), decided once on
  the base lowering and replayed in the candidate compile through `seeded_replay`, the same function
  the untuned compile replays through, so the candidate's code is the shipped step's digest for
  digest (`test/operations/autotune_seeded_seed` holds that, the crown of a search cut to its first
  candidate, and bitwise gradients). Two traps. The plan must be derived from the base lowering AS
  THE TRANSFORM SAW IT: codegen later forces the undecided placements into the table the captured
  `base_opt` references, and the decisions key on placement-sensitive digests, so `tune` keeps a
  pristine `scratch_of` copy taken inside the base compile's transform (GPU only). And a split
  needs a long OUTERMOST reduction loop: a fixture with `b = 4` outermost proposes no split at all
  (the b8 case above), so `autotune_seeded_seed` lays out 1024 x 4 rows. A plan equal to the default
  pipeline (another backend, nothing seeds, splits or sinks) is not seeded, so CPU searches and
  unseeded GPU ones keep their candidate order. The floor is the step the untuned compile ships
  only where that compile IS `seeded_default`: under `model_default_schedule=true` (the
  `performance` and `approximate` profiles) it is `model_default`'s pick, whose placement pre-search
  changes the program, so no candidate reproduces it (the model starts from this plan, next
  bullet but one). A refused window does not lose the floor: a contended digest leaves
  `seen` for an equivalent later seed to measure, and since no later seed reproduces the plan, the
  seed pass ends with one retry of it. `placement_entry_version` 2 retired the recorded arm
  decisions that compared searches without the floor. The three `gpu_default_*` keys are `Search_shaping`
  (registered defaults `auto`, 1024, true), the report's `seeded_ms` is the seed's time beside
  `default_ms` (not stored, so `None` on a cache hit), and `entry_version` 16 retired the crowns
  searched without this floor. `seeded_plan_digest` names the plan from its content alone (split
  sites by target name and extents, not by node id or symbol), and `tune` appends it to
  `trajectory.search_shape` (`|seed=<digest>`, `|seed=none` without a plan, never the cache key): the
  plan reads `Untimed` fields under `auto`, so a crown searched under a different seed is a
  different trajectory and never stops this search early. Deriving it costs one detection pass over
  the pristine program, forced lazily (the search, or a trajectory-matched warm hit, forces it; the
  `seeded untuned plan derived in` log line times it). `limits_field_classes` sanctions two paths
  to `Untimed` fields from `tune`: `seeded_plan`, pinning that a definition `tune` reaches through
  it records the digest, and `seeded_transform`, but only where the search-less fallback spells it
  (gh-ocannl-1491): the local binding every `not search` branch of `tune` calls, found by the scan
  rather than named, so a search path calling `seeded_transform` or a wrapper of it (`seeded_default`)
  fails the scan. A new search-less branch must call that same fallback, or the sanction empties
  and the scan refuses; one that legitimately ships something else extends the scan's forms.
- **Every untuned fallback in `tune` ships the seeded plan, not the plain default** (gh-ocannl-1441).
  The post-search fallbacks (nothing timed, a budget cut that left the floor unmeasured, a failed
  winner replay) replay the plan the search seeded; the paths that never search (search off, a
  refused lineage, a deferred replay that failed) compile through `seeded_transform`, as
  `seeded_default` does, and recompile the plain default only when that fails classified. Before,
  those paths shipped the plain default the recipes had already beaten. A seed changes no cache
  key, so a crown searched under seed A still replays under seed B, but its `best_steps` and an
  abandonment recorded under A do not (`autotune_seeded_seed` runs both seeds; only the
  `Autotune.on_seeded_plan` seam moves the seed alone, since every key that steers the plan also
  renders in the trajectory's settings).
- **`gpu_default_sketch_min_reduction=auto` (the default) replaces the threshold by the ranking
  score** (gh-ocannl-1253, `Autotune.Ranked`): a segment seeds when `seeded_params`' sketch ranks
  strictly below the same segment's preset, scored alone under the device's capacity and cache
  bandwidth — so it seeds where the cross-block re-reads the sketch removes cost more than its
  staging adds; without the cache tier, never within capacity, where under the distinct-cell
  roofline a staged sketch only adds traffic. Geometry and the numerics-preserving candidate set are
  `seeded_params`' either way (`scalar_blocktile`: no mma, conv, epilogue, coalesce, batch-inner,
  swizzle or pipelining; `accumulates_at_storage`: the site's accumulator precision IS its storage
  precision), and a device publishing neither constant (Metal, cc, an uncalibrated GPU) falls back
  to threshold 32 — integers stay the fallback. `tune`'s seeded candidate is the same resolved plan
  (its digest in the trajectory identity, previous bullet). `model_default` starts from it wherever
  the untuned compile seeds (CUDA and HIP, seeding on: the same `seeded_settings` decides both):
  its incumbent is the seeded plan (`mc_label` `"seeded"`), its candidates compete on the program
  after the plan's split prelude, and its sketch candidates are restricted to the same
  numerics-preserving forms (counted as `mc_excluded`), so the model's pick never changes a value
  the untuned default computes. Elsewhere — Metal, cc, or CUDA/HIP with the seeding off — the
  incumbent stays the default pipeline and the whole family competes. The census claim
  (`test/operations/seeded_default`) pins capacity 0, no cache tier, against threshold 32 and is
  CUDA/HIP only; on cc the seeded default is off altogether.
## Dispatch bindings and device properties
- **A dispatch's launch parameters are read on the HOST, at `Context.run`, and carried to the
  device** — never re-read from the caller's refs when the device gets around to the task. Only
  `Schedulers.Multidev` defers a task at all (`Sync.schedule_task` is `Task.run`, and the GPU
  backends enqueue into their stream from inside the task, on the host thread), so it is the one
  scheduler where the distinction is observable, and the observation is a wrong answer rather than a
  slowdown: `Indexing.apply` dereferences the static indices inside the kernel task, while a caller's
  loop — `Train.sequential_loop`, and every training loop — has rebound them for the next step by
  then. Sharing one ref made `multidev_cc` launch on whichever batch the host had raced ahead to, so
  batches were skipped and repeated and the learning rate keyed on the step counter was read at the
  wrong point; the training trajectory diverged from `cc` systematically AND moved run to run.
  `Backends.Add_device` therefore hands the caller refs of its own and snapshots them per dispatch,
  the task copying the snapshot into the kernel's refs on the worker, in queue order, just before
  the launch reads them (`Task.enschedule ?snapshot`). `Context.check_launch_bindings` validating the
  caller's values at dispatch is the same contract stated from the validation side. Extending this:
  any launch input a HOST loop can mutate between dispatches belongs in that snapshot; a merge
  buffer does not, because its writer is itself a task on the same queue and FIFO order covers it.
  `test/operations/async_launch_bindings` is the probe — sweep a static index with nothing between
  the dispatch and the next rebind, accumulate two moments of the bound value on device, sync once.
- **A launch parameter starts UNASSIGNED, and a run refuses it** (gh-ocannl-940): every backend's
  link mints its cells with `Indexing.fresh_launch_param` (an out-of-band `Int.min_value`, so the
  cell stays an `int ref` and `find_exn bindings sym := v` stays the API), and
  `Indexing.validate_bound_value` names an unassigned one — at `Context.check_launch_bindings` before
  dispatch, and again at each backend's launch-side read (cc's `work`, the GPU `Static_idx` arms) for
  a task scheduled around `Context.run`. Library code that runs a routine nobody outside can bind
  writes 0 itself (`Train.init_params`/`run_once`, `Parallel.schedule`, `Autotune.set_test_bindings`
  for an unranged one); a new such site must too. `test/operations/unassigned_launch_param` pins it.
- **`static_properties` has one shape across the backends, and it is a contract**
  (gh-ocannl-710): `(<backend>_devices (device (key value) ...) (device ...) ...)` — a group atom
  naming the dump, then exactly one `Sexp.message`-shaped entry per device, in ordinal order, each
  carrying at least `device_name` and `device_ordinal`, all carrying the same keys. The device
  COUNT is never a child of its own — it is the number of entries — and neither is any other
  backend-level fact; a backend with no devices to describe (an unlinked one,
  `lowered_backend_missing`) names its group something that does not end in `_devices`, so a reader
  tells the two apart without guessing. `Backend_intf.parse_static_properties` is the single reader
  of the contract: `bin/device_props` and `test/operations/static_properties_contract` both go
  through it, so the tool and the test cannot drift apart about what a device entry is. The test's
  negative controls are the shapes this replaced — Multidev dumped
  `(multidev_cc_devices (device_name CPU) (num_devices 16))`, no device entries at all, which a
  generic reader indexed as two devices that do not exist, on the one backend whose whole purpose
  is multi-device debugging; Metal and cc wrapped their pairs one nesting level deeper than
  CUDA/HIP. To surface a new per-device fact, add a key to every entry of that backend's dump; do
  not add a backend-level child, and do not restate anything the entries already determine.
## Staged barriers and asynchronous copies
- "`Tile_mma` is a barrier" is only half true, and the half that fails is the one barrier elision
  wants. Every rendering form ENDS the intrinsic block with a workgroup barrier, so a staging
  barrier that follows one is always redundant (`Schedule.elide_staged_barriers` drops it, and the
  after-loop one, at any pipeline depth). The LEADING bracket is form-dependent: the fragment-scope
  form (`render_mma_fragment_scope`, the crowned Metal/CUDA shape) emits it ONCE, on the scope
  wrapping the whole anchor loop, not per iteration — so a barrier may never be elided against a
  *following* `Tile_mma`. That is why a depth-1 staged k-block keeps exactly one explicit barrier
  (between its loads and its compute) while the pipelined form keeps none: the pipelined prefetch
  writes the *next* iteration's buffer copy, so the previous iteration's trailing bracket is what
  separates it from its reads. Elision is a synchronization-only transform, so the test that pins it
  is an executed BITWISE comparison against the same schedule with barriers re-inserted
  (`schedule_pipelined_matmul`) — adding barriers is always conservative, which makes that reference
  sound no matter what the elision does.
- The CUDA `cp.async` arm (gh-ocannl-487 phase 2, `C_syntax_config.async_copy`) keeps that
  discipline by never touching the intrinsic's brackets: an async copy is only complete for the
  issuing thread after a wait, and only visible to the workgroup after a barrier that FOLLOWS the
  wait — so the rotor loop's body is uniformly prefixed with `ocannl_cp_async_wait_all();
  __syncthreads();`, re-inserting for the async arm exactly the phase opener
  `elide_staged_barriers` drops for synchronous stores (those are published by the previous
  iteration's trailing bracket; an async copy waited AFTER a barrier is published to no one).
  Wait-all (PTX `cp.async.wait_all` = commit_group + wait_group 0) instead of commit/wait-group
  bookkeeping is what makes the emission per-`Set` opportunistic and safe: any staging statement
  the arm declines (precision conversion, surviving fringe ternary, non-global source, elements
  outside 4/8 bytes — sub-4-byte has no cp.async size, and 16-byte needs a destination alignment
  plain shared declarations don't guarantee) falls back to a plain store published by the same
  barrier,
  and correctness never depends on which statements were accepted. It is also why depth stays 2:
  deeper lookahead needs per-group waits. Eligibility is per tile in `compile_proc`
  (`current_async_tiles`; kernel logging disables it — a logged `Set` reads back what an
  in-flight copy cannot provide). Measured on the RTX 5070 Ti (paired in-process pd1/pd2, 9
  replicates, tf32 fragment-scope form): 512³ f32 pd2/pd1 = 0.97 median within a ~14% spread;
  deep-K 256×256×2048 = 0.92 with all 9 replicates in 0.906–0.946 against ≤4.4% arm spread — the
  overlap genuinely pays where the k_o loop dominates, reversing the portable form's Metal
  ~1.4–1.5× cost and HIP's null.
## Renderer failures and fault injection
- A schedule can pass `Schedule.apply`'s validation and still be one the RENDERER cannot express:
  the pipelined-tile checks in `c_syntax.ml` (a read reached outside its rotor loop; a rotor loop no
  longer `Serial`) are positional facts about the final IR, which schedule application does not
  re-derive. Such a check must raise `Schedule_outcome.Cause_at (Backend_codegen, Unsupported …)`,
  not `invalid_arg` — an untyped exception at a compile-side phase is `Fatal` under
  `strict_failure_classification`, so one composed candidate ends the whole search (seen on Metal
  searches over `tile_acr_ma`). `raise_cause` re-renders the same `Invalid_argument` at the public
  `Context.compile` boundary, so typing one costs nothing for hand-written schedules. To probe a
  renderer precondition in a test, do surgery on the applied `optimized` (re-point `pt_rotor`)
  rather than retyping loops — a retyped anchor trips `Low_level.validate_parallel` first and never
  reaches the check under test.
- The same holds one step later, for generated code the backend's COMPILER rejects: every backend
  types it as `Backend_rejected` at stage `"compiler"` (cc and Metal raise `Cause_at` at the compile
  call, CUDA and HIP answer `NVRTC_`/`HIPRTC_ERROR_COMPILATION` in `classify_failure`), so it is a
  decline of that candidate or kernel. Only the compile status is typed: an environment fault fails
  every candidate alike and must stay loud, which is why Metal types a failure only when its report
  carries a `program_source:L:C: error:` diagnostic, and why every backend leaves a header the
  compiler could not open untyped (`C_syntax.missing_include`) although its toolchain reports it
  under that same status (`compile_failure_declines`, ahrefs/ocannl#1387).
- The two pipelined-tile facts are also stated upstream, over the applied IR, as
  `Schedule.pipeline_violation` (gh-ocannl-1231): `op_legality` refuses any op leaving one — a
  `Split` or materializing `Unroll` of the rotor (its binder vanishes, so the reads fall outside),
  an in-place `Unroll` or retype of it — so the beam menu, seeding's `schedule_legality` and the
  renderer agree, and the renderer's declines are the backstop. The renderer builds those declines
  from the same type (`Schedule.pipeline_violation_cause`), so each fact has one wording and one
  decline feature. A new op whose output can move a tile read or rewrite a loop needs nothing
  extra; a new pipelined-rendering precondition is a new constructor, raised by the renderer
  through that function, plus a fixture producing it: `pipelined_move_legality` derives the
  oracle's reasons from the type (ppx_variants_conv) and fails until the fixtures exercise every
  one, and on GPU backends it relates each renderer decline to the oracle's violations of the
  very IR it was compiled from, in both directions.
- The simdgroup fragment scope has the same treatment, with one difference (gh-ocannl-1342). The
  renderer maps a marked fragment's lane-0 load, reduction and store-back to persistent intrinsics
  only when `Low_level.fragment_region` (shared with the cost model) matches: exactly one
  `Tile_mma` between the transfers, the bracket at the HEAD of the statement sequence `pp_ll` tries
  (a recognized scope's trailing statements render one by one). Any other `Tile_mma` into the
  fragment silently takes the lane-0 scalar fallback, timed 2-3x slower: a materializing `Unroll`
  of a loop inside the scope (k_o, or a Base contraction axis around it) replicates the tile, one
  of a loop around the scope leaves a second scope trailing the first, and a `Vectorized` retype of
  the transfers' copy loop (a GPU menu move) unbrackets it. `op_legality` refuses an op that leaves
  MORE `fragment_scope_violations` than it found, not any IR with one, because the fallback still
  renders; `fragment_scope_move_legality` relates each reason to the census's fallback count on a
  GPU. A trailing scope beside a hook-rendered first copy used to omit the fragment's local-array
  declaration (an untyped Metal compile failure); `unscoped_simdgroup_fragments` now keeps it.
- The `Local_scope` grammar is the third scope with this treatment, and the only one APPLICATION
  enforces (gh-ocannl-1467): a scope body is hoisted out of its position, so it holds no
  hardware-annotated loop (`Low_level.scope_hardware_loop`, the predicate `validate_parallel` gates
  on) and no effect beyond its own locals (`Low_level.scope_purity_violation`). The mints never
  break it, but a LATER op can: a hardware `Retype`/`Split` of a loop the scope a materializing
  `Unroll`/`Partition` minted holds, a `Privatize` over such a loop. A `Stage` keeps itself out
  (gh-ocannl-1534): its loads go at its anchor (the deepest loop an outer-part index varies with),
  so an anchor the scope binds is refused by name before a tile is minted, even for a tile loop
  OUTSIDE the scope. Hoisting those loads ahead of the scope is unsound at that tile shape (the
  anchor is unbound there); the scope's whole footprint is the larger stage with the scope-bound
  loops added to `tile_loops`, which the op never infers, since budgets read the tile size off
  `tile_loops`. An unanchored shared stage wraps the statement holding the scope instead of the
  scope-nested tile loop (`rewrite_loop_holder` in `schedule.ml`). `stage_scope_anchor`.
  `Schedule.apply_opt_op` checks the program every op leaves and refuses, typed and naming the op,
  a violation the op ADDED (a malformed entry is the compile gates' to refuse, not the next op's);
  the race queries alone called the `Grid` retype and the `Stage` legal. `scope_verdict` asks the
  same of the single op, from one unchecked probe shared with the other two scopes, only where a
  scope body holds a loop. The same post-op check re-judges every `Privatize` tile of the
  application: no hardware-typed loop may sit inside its per-thread lifetime (between its
  init-load and store-back) — the mint-time refusals, for the ops after the mint. That needs the
  application's tile record, so a bare `op_legality` of one op does not see it (`Op_unknown`: the
  tile is `Local` scratch, which `loops_independent` cannot prove illegal); `schedule_legality`
  does. Compile gates stay as backstops. `privatize_recheck`.
- Autotune fault injection keyed by ATTEMPT INDEX (`Autotune.on_candidate_attempt`) is
  backend-dependent and silently vacuous: how many attempts precede an arm's first *timed* candidate
  varies. On Metal a small matmul's materialize-all arm has a baseline binding no hardware dimension
  (gh-532), and its whole `W_preset` block then dedups against that same digest — six attempts, none
  timed. Counting timing runs with `Autotune.on_candidate_preflight` is the same trap one level down
  (gh-ocannl-898): a preflight fires before the window's verdict, and under the queued objective the
  window can be refused as contended (gh-855) without growing `candidates_timed` — on a loaded CUDA
  device an arm's first two windows were both refused, so a preflight-counted "has timed candidates
  of its own" precondition fired the injection on an arm whose report said it timed nothing. Count
  admitted timings with `Autotune.on_candidate_timed`, which fires exactly where `candidates_timed`
  grows, and inject relative to that. Relatedly, hoisted (link-time packed) `Stage` candidates are a CPU family only —
  `matmul_seed_params` proposes `sk_hoist` from its `is_cpu` branch — so any test precondition about
  packed-constant pools is false on GPU backends and has to be stated as an equivalence.
- The search's test seams (`on_candidate_attempt`, `_preflight`, `_timed`, `_measured`,
  `_callback`, and the `with_uncontended_test_windows`/`with_contended_test_windows` scopes, one
  seam) are `Autotune.Seam.t` values, one per
  DOMAIN: scope one with `Seam.with_value` (restores on exit, nests), set one for the domain's life
  with `Seam.set`, read it with `Seam.get`. A search reaches only its own domain's values, and a
  freshly spawned domain starts with the defaults — so a fixture that searches from a spawned domain
  sets its seams inside that domain. Setting REPLACES (a harness that also wants the enclosing value
  reads it first and calls it); only the timing-call observers chain. A new seam is a `Seam.t`, never
  a process-wide ref: two domains' searches would otherwise overwrite and observe each other
  (`autotune_seam_domains` pins the isolation). The resource fault-injection callback follows the
  same per-domain rule; its placement constraint is in backend-memory.md.
- **Searches in different domains take turns; they never overlap** (gh-ocannl-1352). `Autotune.tune`
  runs inside `Autotune.exclusive_search`, and `Train.tune_placements` takes it for the whole call,
  so another domain cannot slip a search in between two arms; a search started while another
  domain holds it raises `Autotune.Concurrent_search` before resolving a setting, calling a seam or
  emitting a report. The holder is a domain with a per-domain depth, so nesting in the holding
  domain proceeds. Refused rather than made safe because the compile path's process-wide state
  (`Tnode.next_uid`, `Low_level.analysis_cache`, `Context`'s print proxies, a device's
  `constant_buffer_cache`, Base hash tables throughout) was never audited for two domains at once;
  the per-domain seams, streams and observers serve turn-taking searches. A test that overlaps two
  searches on purpose blocks one inside its own `on_candidate_attempt` until the other has tried
  (`autotune_concurrent_refusal`, whose refusal claims fail with the gate removed).
- Scripted search fixtures opt into `Autotune.with_uncontended_test_windows` (gh-ocannl-1156)
  around `on_candidate_measured`: the measurement callback alone only sees windows the host already
  admitted, so a loaded runner could time nothing and leave cache controls vacuous. The scoped seam
  clears candidate contention only; calibration, invalid readings and unresolved batching keep
  their admission rules. `flip_abandonment` pins production refusal, nesting and failure restoration
  with a majority-stalled window, then asserts search, placement and cache branches without skips.
- **A refused window names one cause** (gh-ocannl-1517): `Autotune.timing_admission` is the one
  classifier (the gate, the `NOT TIMED` lines, the report's counts and each timing call's
  `admission` all read it), in the order unbatched, unreadable reading, contention. The reading
  precedes contention because a zero minimum is also the dispersion floor, against which every
  positive sample reads as a stall: before the split, a clock that resolved nothing was counted
  and logged as a loaded host. `report.timings_contended` still counts EVERY refusal (every cache
  and completeness gate reads it unchanged); `timings_unbatched` and `timings_unreadable` are its
  named subsets and `Autotune.contention_refusals` the rest, and `baseline_refusal` names the
  baseline's cause. A skip or retry excused by a refused window gates on the cause it expects —
  under `with_uncontended_test_windows` that is `timings_unreadable`, the one refusal an `Isolated`
  search has left (`autotune_callback_release`) — and its reason names the cause it met
  (`autotune_replay_refusal`). Drive each cause on any host: a degenerate reading by
  `on_candidate_measured` returning NaN inside the uncontended scope (outside it, a host contention
  refusal can pre-empt the scripted one), contention by `with_contended_test_windows`; both are
  pinned in `autotune_refusal_causes`.
- **Dispersion is a fact; on a CPU backend contention is an attribution** (gh-ocannl-1558). The
  2x-majority rule (`window_result`) sets `dispersed`, and records `spread` (median raw wall over
  the floor). On GPU backends a dispersed window is still refused (`Refuse_dispersed`). On
  `Sched.backend_is_cpu` backends `time_routine` reads `Utils.Host_load` (machine busy CPU-s from
  `/proc/stat` or mach ticks, minus this whole process's `Unix.times`, over the wall) from before
  the calibration to after the window, and refuses a dispersed window only on `Cotenant` (at least a
  core of foreign load plus a two-tick allowance over the span; a dispersed window shorter than
  `Host_load.min_span_s` waits out the rest idle and is charged the padded span's whole foreign
  load, so padding never dilutes a co-tenant); `Quiet` admits it, and
  `Unavailable` (Windows, or a counter that went backwards) keeps the refusal. A co-tenant does not
  refuse a tight window: uniform delay stays a valid reading. Why: a 32-thread cc routine's own
  spread met the 2x rule on a quiet minix, so no search there cached. Consequences: own-process load
  in other domains is never a co-tenant; a claim derived from `median <= 2 * minimum` must bypass
  `dispersed`, not only `contended` (`autotune_timing_modes`' executed legs); the report counts
  `timings_dispersed` and the cache entry stores `best_spread`. Drive it with
  `Autotune.host_load_reading` (scripted readings) and `Autotune.on_batch_wall` (scripted clock),
  as `autotune_dispersion_attribution` does.
- **A hopeless window ends early, against what its reading is RANKED by, not against the search's
  best** (gh-ocannl-1527). `Autotune.hopeless_window` cuts a window once 3+ samples have a minimum
  over 4x the reference and are not dispersed (noise only adds time, so the full window could rank
  under the reference only if every sample so far were inflated past twice the stall ratio; a
  uniform co-tenant delay over the prefix is the stated boundary). `time_routine ?hopeless_against`
  applies it to the calibration's singles (gated at `queued_batch_ms`, where cc/Metal settle at
  depth 1) and to every timed window, raising `Timing_hopeless`; the search releases the candidate
  unranked (`timings_hopeless`; not a refusal, so still cacheable; not an admitted timing, so
  `candidates_timed` and `best_steps` skip it). The trap the issue walked into: the per-segment
  singles that dominate a gpt2_mini arm-B search (~131-144 ms on minix around an 18 ms seeded
  incumbent, probe of staging#1327) are timed around an all-preset backdrop and ranked only
  against their own key's singles at the 0.5% playoff margin, so cutting them against the
  incumbent would delete the recombination composite. Only whole-routine seeds (against the beam:
  its head with `rounds = 0`, its slowest entry once full otherwise) and round candidates (against
  the round's slowest once full) are cut; singles, composites, playoff alternates and floors never.
  On that probe the sound cut reaches the ~9 slow `F_preset` seeds (~3 s each of 556 s timing), not
  the 166 singles: their cost is the 16-sample floor plus the depth-2 probe, a different lever.
  Nothing is cut under a scripted `on_candidate_measured` (the physical window says nothing about
  a scripted rank). `autotune_hopeless_window` drives it with `on_batch_wall`.
- A new scripted-ranking test takes the harness in `test/support/scripted_search.ml` (library
  `scripted_search`) rather than hand-rolling it: `Scripted_search.scripted` is the scope (the
  measurement script with each window's ordinal, label, digest and reading; the uncontended windows;
  the attempts and admitted windows recorded; settings stated for the scope and restored after it;
  `~attempt_clock` for a budget cut at a chosen attempt), and `search` is one `Autotune.tune` in it
  with the cache I/O recorded and an output read; `with_stderr_captured` and `clear_cache_dir`
  sit beside them. It sets only the attempt, measurement and budget-clock seams, and calls the
  attempt hook in force when it starts, so the test scopes its other injections (`_preflight`,
  `_timed`, `on_staffed_start`, an attempt-time raise) around the call. A test whose windows must
  keep PRODUCTION admission (`autotune_measured_refusal`, `autotune_callback_release`,
  `placement_store`'s ranking) stays hand-rolled: the harness's scope is uncontended.
## Render-time typed declines
- This section is the inventory of every render-time typed decline (gh-ocannl-1351): a
  `Schedule_outcome.Cause_at` constructed at phase `Backend_codegen`, one candidate's decline
  raised while its code is generated, before any backend compiler runs. Each entry opens with the
  decline's key in backticks — the literal the cause's record names it by (an `Illegal_schedule`'s
  `check`, an `Unsupported`'s `feature`), or the oracle constructor it is built from — and is classified against
  `Schedule.op_legality`. "paired (`Schedule.X`)" means the renderer builds the decline from the
  oracle's own reason `X` through a `Schedule.*_cause` function, so the oracle refuses the moves
  that produce it and the renderer is the backstop. "render-only, because" says why no op verdict
  can state the fact. `render_decline_inventory` (in `@scans`) parses every `arrayjit/lib` source
  and this section, and fails on a decline no entry classifies, an entry no site raises, a "paired"
  entry over a literal site or a "render-only" one over an oracle-built site, a paired constructor
  no code `op_legality` reaches in `schedule.ml` (by name, transitively) constructs any more, a
  cause it cannot read, a `Cause_at` anywhere whose phase is not a spelled-out constructor, a module
  bound under the oracle's name `Schedule` (which would make a `Schedule.*_cause` call name
  something else), an exception rebinding `Cause_at` under another name, and a `C_syntax` `Cause_at`
  at any other phase. So a new typed decline lands with its entry here, and a new render-time
  precondition the oracle can state lands as an oracle constructor raised through its `*_cause`
  function, the gh-ocannl-1231 pattern above.
- `Read_outside_rotor` (`C_syntax.pp_pipelined_rotation`): paired (`Schedule.Read_outside_rotor`).
  A read of a pipelined tile with no rotor value in scope; `pipelined_move_legality` relates the
  renderer's decline to the oracle's violation on the IR it compiled.
- `Rotor_not_serial` (`C_syntax.compile_proc`): paired (`Schedule.Rotor_not_serial`). The rotor
  loop retyped, or unrolled in place, so the rotation would freeze at copy 0.
- `hardware_binding_race` (`C_syntax.refuse_unseparated_thread_write`, asked at `compile_proc` and
  in `try_warp_reduce`'s decline arm): render-only, because the thread identity it judges is the
  BACKEND's binding, and `op_legality` takes no backend. Which loops bind a register is
  `bound_register` (cc binds none: a `Workgroup` loop iterates there, so the same IR is legal on cc
  and refused on a GPU); which outermost `Grid` loops cc's pool runs as threads is
  `collect_parallel_grid`, decided while rendering; which `Workgroup_reduce` lanes are deferred to
  the warp shuffle follows the backend's `warp_size`; and the storage class (`thread_storage`)
  forces placements at render. The backend-independent shadow of the fact, a hardware retype whose
  threads the affine queries prove share a written cell, is already `access_legality`'s
  `Op_illegal`; everything else it answers `Op_unknown`, compile and see. Pairing the rest needs a
  backend binding descriptor in the oracle's signature.
- `Low_level.validate_parallel` (`Low_level.validate_parallel_classified`, the first check in
  `C_syntax.compile_proc`): render-only, because it is a fact about the finished schedule, not
  about one op. Its coverage rule (a materialized write sits under bound loops of every active
  slot) and its barrier rules (equal workgroup extents, no barrier under a guard or non-uniform
  control) are kernel-global, and a legal chain passes through states that break them: retyping
  one nest of a two-nest kernel leaves the other nest's materialized writes uncovered until the
  next op retypes it. The search asks it of finished schedules wherever it ranks without compiling
  (`Autotune.validate_segments_for_model`: model ranking, seeded replay); the renderer's call is
  the backstop, before a line is emitted.
- Outside the inventory: the renderer's silent fallbacks (`declinef`, logged under
  `schedule_log_declines`), where the candidate still renders, of which the fragment scope's
  lane-0 fallback has its oracle counterpart (`Schedule.fragment_scope_violations`,
  gh-ocannl-1342, above); and `C_syntax`'s untyped `invalid_arg`s, which are not declines but
  fatal under `strict_failure_classification`.
## Search evidence and tensorization
- "Seeded" is not "timed". An autotune family can be enumerated in bulk and rejected in bulk at
  candidate compile, and a count of proposals then reads as coverage it does not have — assert on
  the *timed* counter (`report.mma_timed`, `fiss_sketch_timed`, `split_reduce_timed`), and follow it
  with an executed value check, since a candidate that compiles is not yet one that computes.
- The `N segs` of `F_saved[N segs]` counts every segment of the persisted segmentation, one kernel
  each (gh-ocannl-1164). Reports written before then quote `F_saved[fine N segs]`, whose N counted
  only the `Normal` segments' entries: `fine 77 segs` emitted 136 `__global__`s (gpt2_mini on HIP,
  `report-gh612-hip.md`), and `[58 segs]` 117 — wrong by ~1.8x as a kernel count. Take kernel counts from
  the launch log (`schedule_log_launches`, whose `seg i/N` names the real fission width) or from the
  emitted source. The launch
  log's FIRST fissioned `seg 0/N` (skipping the `N=1` whole-routine probe) is arm A, the next is
  arm B — which is also how to pick the right file out of a content-polling snapshot of
  `<routine>__seg.hip`. The watcher can catch a partially written file, and the kernel count alone
  does NOT identify a usable capture: a torn file can already carry every `__global__` line while its
  last body is incomplete, and glob order is hash order. Require balanced braces AND a clean `hipcc`
  compile before accepting a snapshot (`benchmarks/gh612_cells.sh pick_armA`).
- A label names one candidate (gh-ocannl-1230). A beam-round candidate prints as
  `F_saved[N segs from <tag>]+seg<i>:<move>` (`W_saved[N ops from <tag>]+<move>` whole-routine):
  `<tag>` is the parent's digest tag, the one the `round k:` beam line and the parent's timing line
  print, and `<move>` the appended op's s-expression; a cache replay prints no suffix, while the
  winner rebuilt into the caller's context (`?timing_ctx`) compiles under the label it was timed
  under, so a failed winner replay names the round candidate. An `F_split` entry renders every
  `sr_site` field (target `name#id`, axis, `red`/`out`/`cost`, `b`, `dyn`, `swap:o/i+…`). An `F_sketch`
  entry ends in ` @<tag>:u<units>` of its segment key: the structural digest's tag, then the keyed
  segments' first units joined by `+u`, since two keys can share the digest (gh-ocannl-1243). A sketch label renders every `sketch_params` field, and `autotune_labels` reads the field list off the type's
  derived s-expression, so a new field fails that test until `sketch_params_label` renders it; the same
  test pins `sr_site`'s fields to `split_sites_label`.
- A per-kernel profile's sum may only be validated against the step time of **the compile it came
  from**. Each search rep crowns a different artifact with different tile sizes, so holding one rep's
  profile against another rep's step p50 measures the search lottery, not the reconstruction — on
  gpt2_mini/HIP that turns a genuine 0.9% agreement into an apparent 2.3% disagreement, and the error
  is invisible because both numbers are real. Quote the paired p50 from the same cell's **pass-2**
  `replay2.out` and nothing else: `search.out`'s p50 is a pass-1 timing carrying the search process's
  own overhead, which `benchmarks/README.md`'s two-pass protocol excludes, and `snap`'s `replay.out`
  is a debug-file run rather than a clean timing pass.
- "Timed" is not "tensorized" either, and that failure is worse: a declined `Tile_mma` renders its
  scalar fallback, which compiles and runs, so the candidate is timed, ranked and possibly crowned
  under an `mma-*` label (gh-ocannl-545: 20 of 20 timed bf16 candidates on CUDA were scalar). The
  emission is the source of truth, and since gh-ocannl-626 it is carried, not fetched: every
  compiled routine has `Context.routine.mma`, an `Ir.C_syntax.mma_summary` whose `tensorization`
  field is `Tensorized` (at least one tensor-core / SIMD-register-tile emission), `Scalar_fallback`
  (statements emitted, every one declined) or `Not_requested` (no `Tile_mma` emitted at all), with
  the statement and fallback counts beside it. Read that; do NOT bracket `mma_census_enabled`
  yourself (`C_syntax.with_census` is the bracket, it nests additively, and it is what
  `Context.compile` calls). `Autotune.report.best_tensorization` is the crowned candidate's label,
  `None` when nothing was crowned — and the pair to read is `best_tensorized` (what the SCHEDULE
  asked) against `best_tensorization` (what the EMISSION delivered): a `true` beside anything but
  `Tensorized` is a scalar timing under a tensorized label. `schedule_log_declines=true`
  names the rule that fired. When seeding and emission can disagree, fix the seeding side too, or the
  measurement budget keeps going to schedules that never tensorize: `mma_format_tiles` is keyed on
  the whole `(a, b, accumulator)` format triple, with per-entry arch floors, precisely so that a
  combination a backend supports at one accumulator width but not the other cannot be seeded.
  The GPU descriptor/resolver cross-check in `schedule_mma_matmul` compiles every f32/f16/bf16/fp8
  storage triple under default, wide and tf32 policies (gh-ocannl-1058). Its expectation uses
  `Backend_intf.advertises_mma_format_in_scope`, a pure descriptor judgment with explicit policy
  and accumulator scope; tile divisibility is asserted separately. Missing MMA capability skips
  this both-sides census leg, including HIP's device/header conjunction. `mma_format_scope` covers
  absent wide scopes and policy combinations a concrete GPU cannot vary. The descriptor's
  correspondence with the arm table, for every backend, needs no device at all:
  `mma_format_arm_tables` (gh-ocannl-1246) sweeps `Numerics.all` x emission scopes x storage
  triples through `Autotune.mma_tile_for_precisions_in_scope` against each backend's `mma_arm`,
  plus reachability of every advertised key, over CUDA's `Ir.Cuda_mma`, Metal's `Ir.Metal_mma`,
  HIP's `Ir.Hip_mma` (the device-free tables each backend installs once its device set qualifies)
  and cc's live tables. Put a new backend's mma tables in such a module and add a row; what the
  census leg alone still pins is that the hooks render the arm the table names. The arch-floor axis no
  fleet GPU can vary either (every box sits above CUDA's 70/80/89 cutoffs) is `cuda_mma_cutoffs`'s
  (gh-ocannl-1214): CUDA's descriptor and arm resolver live in `Ir.Cuda_mma`, outside the
  cudajit-gated backend, as functions of the compute capability, and the test evaluates both just
  below and at every cutoff either owner has, both cutoff sets derived (the descriptor's by
  sweeping its record, the resolver's from `mma_arm` floors), never listed. Since gh-ocannl-1232
  both take their floors from `Cuda_mma.arm_floor` (descriptor entries are named by the arm they
  seed), so what the test still decides is the PAIRING of format keys and scopes with arms. A descriptor field
  whose cutoff no arm checks is caught by the cutoff-set comparison, not by the admission one.
  Where a timing is REPORTED the label is now printed, so a mismatch is legible without re-deriving
  anything: the `autotune_log` NOTE lines lead with it, `Train.tune_placements`' arm lines read
  `[tensorized/<label>]`, `bin/schedule_bench` and `bin/narrow_gebp_bench` print
  `C_syntax.mma_summary_string` on EVERY timing line (not only when something declined), the
  benchmark harness prints it per segment in the per-kernel table, and the result line's `tune`
  arms carry `tensorization` + `mma_statements`, which `orchestrate.py` renders in the report's
  `mma` column (`SCALAR FALLBACK` / `NO MMA EMITTED` shouted) plus a `TENSORIZATION NOTICE`. What
  that column reads is top-level `shipped_mma` for tuned and untuned cells, falling back to
  `tune.shipped_mma` only in older records where the top-level key is absent (gh-ocannl-1132).
  A present null census stays unknown. The census merges `compiled_step_routines`, including
  conditional host-gated SGD before execution, without extra timing; the dominant-kernel filter
  alone describes work observed in the measured window. It comes from compiled routines, not the
  arm named as shipped — a crowned arm candidate is not always the shipped artifact: a gh-555 flip
  refinement ships under `shipped: "flip"` and is not an arm, and the `timing_ctx` path can fall
  back to the untuned default after crowning a winner. Same rule as "crowned is not shipped", one
  level down.
  `mma_staged_layouts` (gh-ocannl-481) is keyed the same way for the same reason: the swizzled
  staged twin is seeded only where the emission can actually read that layout. CUDA advertises
  uniform bf16 and fp8 x fp8 -> f32 (gh-ocannl-1073): bf16 uses `ldmatrix` for both operands;
  fp8's row-major staged A uses `ldmatrix`, while B's four strided bytes per register gather
  through the swizzle map. Eligibility remains per operand and orientation. The census
  distinguishes `Mma_intrinsics_ldmatrix` from `Mma_intrinsics` using the actual load choice,
  so "tensorized" and "fed at rate" are separable in a sweep.
- **A test decides how a `Tile_mma` rendered from the census, never the source** (gh-ocannl-1316).
  The lane-0 fallback's `if (lane == 0)` guard also wraps the C backends' register tiling, so
  `"== 0)"` never told the two apart; "no intrinsic spelling" says nothing about a statement that
  declined for another reason; and a decline diagnostic goes to stderr, so a check that the source
  lacks it always passes. `Ll_test` holds the statement-level forms over
  `Ll_test.renderings routine.Context.mma`: `all_rendered`, `all_tensorized`, `all_declined`,
  `none_emitted` (for GPU legs of a CPU-only `Tensorize` schedule) and
  `census_agrees_with_capability` (tensorized where `Ll_test.advertised` says the backend has the
  tile for the leg's storage precisions, declined elsewhere — its stanza then declares
  `OCANNL_TF32_MATMULS` and `OCANNL_CC_VECTOR_BYTES` beside the backend). Keep a text pin only
  for what the text is about (a dialect's instruction, a register-tile geometry, a fence's
  position, a swizzle's XOR), and say so at the site.
## Register-tile search and ranking
- **The register-tile geometry is a schedule decision, not a renderer constant** (gh-ocannl-619).
  `Schedule.Tensorize` carries `tile : Register_tile.t option` (`{rm; rn; lanes}`) into
  `Low_level.Tile_mma`; `C_syntax.try_register_tile` honours a request EXACTLY or declines it to
  the scalar fallback with the violated rule (`Register_tile.check`: a lane count on the file's
  ladder that `n` fills, `rm <= m`, `rn * lanes <= n`, `rm*rn + rm + rn` within the 19/34-register
  budget) — never substitutes, so a candidate timed under a geometry label ran that geometry or
  ran scalar, and the census says which. `None` is the renderer's ranking model,
  `Register_tile.default` — since gh-ocannl-620 a reuse-only ranking in vector-issue slots over
  `Register_tile.coverage` (the full passes plus the column tail as a narrower tile), with no
  fitted constant; the gh-575 peel weight of 10 went with the scalar peel — which the seeding
  consults through the same module: every CPU tensorized leaf gets a `register-tile` level of
  `auto` plus `Register_tile.alternatives` (the LARGEST tail-free `rn >= 2` at the widest fitting
  width — the smaller tail-free widths are dominated on the model's own terms — plus the budget
  cap when its column tail is at most one vector, plus the largest tile tying a tail-bearing
  default), only where at least one exists. The default is
  usually the cap now, so on a non-dividing site the twin is the notch below it: one question per
  leaf (gh-614's register pressure), where the pre-620 rule asked none on a site whose cap peeled
  fat (`tile_mma_declines` 22 -> 32 seeds, `sketch_family_tree`'s AVX2 tree 23 -> 34). The emitted header appends
  `; geometry from the schedule` on a request and nothing on a default, so pre-619 codegen goldens
  stand. Sweep a width by seeding it or by handing `?tile` to `Sched.tensorize` — never by
  patching the renderer again. The cache saves the field as `[@sexp.option]`, so pre-619 entries
  parse. Summed over a site the model's price is `ceil(n/lanes)*(1+1/rm) + ceil(n/(rn*lanes))`,
  so at one width it ranks by pass count alone and exact ties are common; it is computed as an
  integer (scaled by `rm`) because the float form broke those ties by rounding at rm = 3 and 1
  (gh-ocannl-947). The two-row twin `Register_tile.rm_twin` (rm = 2, the widest the budget admits)
  joins the level only under `autotune_register_tile_rm_twin`, off until a timing shows it winning.
  The CPU conv leaves carry the same level since gh-ocannl-697 (micro-kernel `m` = the GEMM row
  or the panel's `sk_bm` rows, `n` = the out-channels; `cpu_conv_sketch_schedule` hands `sk_tile`
  to both of its `Sched.tensorize` calls), pinned by `schedule_conv_gemm`'s `cvr` leg (every twin
  renders its header's geometry and matches its `auto` sibling bitwise). Small-channel convs get
  no twin: at 8 out-channels one or two vector columns leave no alternative at any width, which is
  why the conv seed counts there did not move. Not done: whether a partial vector's masked copies
  deserve a term of their own.
- **C-tile traffic cannot rank `rn` at a fixed lane width** (gh-ocannl-1099).
  Full passes plus the narrower column tail move `2 * rm * ceil(n/lanes)` C vectors per row
  band, independent of `rn`; charging `2 * rm * rn / k` at full `rn` on the tail pass, or once
  per site, miscounts it. The exclusive AVX2 A/B confirmed a 4x2 serial win at n=28, but 4x3
  won at fixed n=512 with k=32 as well as k=256. Derivation, paired measurements and
  reproduction protocol: [gh-ocannl-1099](../research/gh-1099-register-tile-c-traffic.md).
- **A price tie at one width goes tail-free, then to the SMALLER `rn`** (gh-ocannl-1180). Equal
  price at one width is an equal pass count, and two tail-free tiles never share one, so the `rn`
  key only ranks two tail-bearing tiles: two bodies each, equal A splats, equal C traffic. Only
  register pressure (`live_registers`) still differs, and the smallest `rn` reaching the pass
  count is the most even split of the vector columns. The old "larger `rn` for more A-reuse" key
  claimed reuse the price shows is not there. The larger tile is not lost: `alternatives` seeds
  the largest tile tying a tail-bearing default, so the tuner times both sides of every tie. The
  rule moves picks on AVX2 only at f32 n = 25..31, but on many NEON and AVX-512 sites
  (`gh1180_tie_ab.sh` in `benchmarks/` times a pair from one build; the A/B table is on
  lukstafi/ocannl-staging's PR for the issue).
## Crowned and shipped artifacts
- "Crowned" is not "shipped", and neither is reproducible on a small routine. `Train.tune_placements`
  runs two searches and keeps one artifact, so a family can win the arm that is then discarded whole
  — read `report.best_label` / `best_tensorized` / `best_tensorization` / `mma_best_ms` per arm (the A/B calls `?report`
  for arm A first and ships the smaller `best_ms`), never the fact that some search crowned it.
  Since gh-ocannl-638, do not re-derive WHICH arm shipped from the reports' times either: config
  `tune_ship_arm=a|b` overrides the comparison, and `?on_ship` is the callback that says what
  actually shipped: a `Train.shipped` record (gh-ocannl-1438) with the arm (`"A"` / `"B"` /
  `"flip"`), whether the placement was a placement-store replay, the returned routine's name and
  the report of the search whose compile shipped (its `source_digest` is the schedule-cache entry a
  splice A/B must match; never infer it from the arm's position). That knob is what a measurement needs: the discarded
  arm is never executed against anything (`?report` carries timing metadata, `winner replay ok` is
  a dispatchability check, and a per-kernel harness times kernels on synthetic buffers without
  checking results), so profiling arm A while arm B ships leaves the profiled routine
  output-unverified — the limitation `benchmarks/report-gh612-hip.md` states in its verdict and
  `report-gh612-hip-verified.md` closes by forcing the arm. Forcing does not skip the other arm's
  search, so the A-vs-B numbers stay quotable; it does suppress the flip refinement.
  Below GEMM-dominated sizes the crown is a lottery: on `mlp_small`/metal five identical cold-cache
  searches crowned four different families in one arm with a 4.5% spread of best times, while the
  arm gap stayed at 57–95% (gh-ocannl-546, benchmarks/report-gh546-metal.md). Conclusions of the
  form "family X wins/never wins here" need repeats; the arm-level verdict does not.
  `Autotune.report` says what a call did about searching in ONE field, `outcome` (gh-ocannl-677):
  `Searched` | `Search_died of terminal_failure` | `Cache_replay` | `Search_disabled` |
  `Pre_search_failure of terminal_failure`. Match it; do not re-derive it. In particular **"this
  process searched" is not `not cache_hit`** — under `autotune_search=false` (the `reproducible`
  profile) and on every pre-search failure a call reports neither having searched nor having
  replayed, and ships the untuned default. That mis-derivation, made twice in one PR, is what the
  variant replaced four independent booleans to stop; the benchmark JSON carries the state by name
  (`arms[].state`) and counts the third bucket (`tune.no_searches`) so the sweep reads it instead
  of recovering it from two zeroed counters.
  The arms are independent experiments and are contained as such since gh-ocannl-550: an arm whose
  search raises is a LOSING arm (ranked `infinity`), the other arm's winner ships and stays cached,
  and the failed arm's report still arrives in position as `Autotune.Search_died` — read the
  outcome (or the `Autotune.terminal_failure` projection over it) before `best_ms`, because a
  failed arm's best is a time whose routine was never compiled. Before that fix, one arm's late failure destroyed the other arm's finished work
  in-process (the cache entry survived, since `SC.store` precedes the winner replay — so a warm
  cache could still replay it; five of five tf32 `gpt2_mini` runs lost arm A this way). Note where
  it escaped: NOT at the failing candidate — candidate-grade protection absorbed those OOMs as
  `Backend_link` declines — but after the search concluded, when the exhausted device defeated both
  the winner replay and the untuned-default fallback compile behind it. Containment tests do not
  need a device that can fail: `Autotune.on_candidate_attempt` injects one
  (`test/operations/autotune_arm_containment`). An arm whose base compile the device refuses is
  refused before its search, typed, as a `Pre_search_failure` (gh-ocannl-1326; backend-memory.md
  has the mechanism), and the placement store records no decision over a failed arm.
  **No arm's routine is resident while another arm searches** (gh-ocannl-1408): the arms and
  flips search through `Autotune.tune_deferred`, which reports, releases everything it compiled
  and hands back the compile of what it ships; `tune_placements` compiles only the placement that
  ships, after every search, falling back to the next completed result if that compile fails.
  Before, arm A's winner was compiled as its search ended, and at b256 on 12 GiB its 6.6 GiB
  refused arm B's base compile, so every such A/B was single-arm. Measure residency where the
  next search begins (`Autotune.on_candidate_attempt`'s second `"baseline"`), not at the previous
  search's report: a deferred report precedes the compile, so report time cannot tell the two
  orders apart (`test/operations/autotune_device_memory` scenario 4).
- **A crown is a time on the timing context's VALUES** (gh-ocannl-1557). The default-placements
  arm inlines the ReLU and max-pool backward selects (`(max_pool == relu(..)) != 0 ? grad : 0`)
  into the nests that consume them, where clang compiles them to `fcmp` + conditional branches, so
  that arm's step time follows the activation sparsity; the materialize-all arm evaluates each
  select once per element and barely moves. On cifar_resnet/cc (M4 Max) the same arm-A routine ran
  min 88–96 ms on the `Train.init_params` draw (78k exactly-zero parameter-grad entries of 220k)
  and 109–119 ms on the fixture's weights (50k), interleaved in one process; arm B moved 3%
  (88.7 vs 91.1). Scaling the weights by 4 restored the zeros and the speed, and dividing the init
  draw by 4 removed both, so it is the sparsity rather than the draw; no parameter or grad was
  subnormal. That was most of the "arm A ships 18–28% above its crown" gap: the benchmark runners
  searched on an init-draw scratch lineage and stepped on fixture weights. They now seed the scratch
  with the fixture (`Bench_harness.tuning_scratch`; `BENCH_TUNE_SCRATCH_WEIGHTS=init` restores the
  old search, `benchmarks/gh1557_cells.sh` runs the A/B), which took arm A's step p10 from 28% over
  its crown to 4% on minix (Zen 5, a measurement window; 9% on a shared M4 Max) and left arm B
  within 3%. Those arm-A steps were timed in the searching process, which only inflates launches,
  so they bound the fixed gap from above; the driver now times both arms in replay processes. The
  rest is the search's own drift: every timed run is a
  training step on the scratch lineage at one fixed batch, so after a 200 s search its weights had
  61k zero grad entries against the fixture's 50k, and the shipped routine ran 100 ms on them
  against 112 ms on the fixture. Late candidates are therefore timed at a sparser state than early
  ones. Nothing restores the timing lineage between candidates; the library cannot seed it for the
  caller either (a `?timing_ctx` is the caller's lineage, and a training run's weights keep
  moving), so on data-dependent code a crown is exact only for the sparsity it was timed at. The
  init draw also mis-ranks the arms: arm A crowned 87.5 ms against B's 84.9 on it, and 102–104
  against 86 on the fixture. To attribute a crown/step gap, time the shipped routine under several
  weight sets interleaved in one process (`Context.get_values`/`set_values` on the params; the
  issue's comments hold the diagnostic).
## Failure phases and lineage containment
- A timing failure's *phase* decides whether the lineage is condemned, so pre-dispatch validation
  needs its own. `Context.run` validates (poisoned lineage, uninitialized inputs, unsatisfied
  execution dependencies, out-of-range static bindings) before dispatching; inside a `Launch`-tagged
  boundary those failures were unattributable — `classify_failure` returns `None` on every C
  backend — so `classify_raw` made them `Fatal` and the handler poisoned the lineage. A one-line
  user mistake (a `timing_ctx` scratch context missing one of the caller's initializations, which
  its own docs warn about) thus condemned the search *and* the context, with no restore
  (gh-ocannl-564; gh-ocannl-536 for why there is no restore). `Schedule_outcome.Preflight` now tags that
  region and classifies as a contained `No_device_writes` decline **without consulting the
  backend** — host-side validation, so a classifier guessing `Writes_may_have_occurred` would
  escalate a failure that provably wrote nothing. Rule for any new boundary around `Context.run`:
  tag what precedes `Ir.Task.run` as `Preflight`, or a fixable mistake reads as device damage —
  **but only the per-candidate half of it may be contained** (gh-ocannl-569, found on HIP). The
  split is `Context.check_lineage_runnable` (poisoned lineage, uninitialized inputs, unexecuted
  dependencies) against `Context.check_launch_bindings` (out-of-range static bindings), and it is
  the difference between a condition that belongs to the *lineage* and one that belongs to *this
  candidate's* bindings. Only the second can fail one candidate while its siblings time cleanly;
  the first fails every candidate of every arm identically, so containing it is silent —
  a search whose serial baseline is not dispatched (**every GPU search**, gh-ocannl-532) then
  declines every candidate for the one reason, times nothing, and `tune_placements` *returns
  normally* shipping the untuned default out of an unusable lineage, with no exception and no
  `terminal_failure`. On the C backends the dispatched serial baseline hits the condition first and
  takes the arm down with the caller's message, which is why every golden encodes the CPU shape and
  CI never saw it. So: **contain the per-candidate half, raise the lineage-wide half outside the
  boundary** — at the candidate site before `Outcome.protect`, and inside the baseline's match
  scrutinee so `condemn` still reads it as pre-dispatch and leaves the lineage usable. Tag the
  hoisted raise `Preflight` anyway: it carries no boundary there, but `search`'s fallback handler
  would otherwise report a validation error under its `Transform` default.
  These causes also resist injection: they belong to the lineage and the bindings, not to a
  candidate, so a genuine one fails *every* candidate at once, which is why
  `Autotune.on_candidate_preflight` exists — though since the lineage-wide half now escapes the
  region, injecting one of *its* exceptions through that hook exercises the containment machinery
  with a realistic payload rather than mirroring where a real one is raised.
- A typed cause is not automatically a containable one. `Schedule_outcome.uncontainable` names the
  causes `protect` makes `Fatal` although typed, the fatal record keeping them in `cause`: today the
  cc backend's `dlopen`, `artifact_missing`, and `codesign` rejections at `Backend_link`
  (gh-ocannl-1077, gh-ocannl-1142). In the `dlopen` case the object compiled and the
  loader found a symbol nothing supplies, an OCANNL link bug; contained, it declines exactly the
  candidates whose code reaches the symbol (gh-ocannl-1045's libmvec: the vectorized ones), and
  the search quietly ships a slower winner. Before, it escaped as a raw `Dl.DL_error`, contained
  under permissive classification. A JIT rejecting one candidate's PTX stays a counted decline.
  `test/operations/cc_dlopen_cause` manufactures one via the compiler command. Missing artifacts
  after a successful compiler exit and signing failures are also fatal: they violate the
  host/toolchain contract, rather than establish that a schedule is unsuitable. `Compiler_bug`
  here identifies that broken backend contract; the actual cause may be an external tool or
  filesystem. A search must surface it instead of quietly falling back. The
  `cc_dlopen_cause` modes provoke all three paths with permissive classification on Unix;
  Windows cannot run the shell fixtures or defer undefined symbols to dlopen. The
  `test_schedule_outcome` unit test pins all three stages across provenance and strictness on
  every platform.
- **A schedule op refuses typed; any other raise inside `apply` is a transform bug** (gh-ocannl-1323).
  `Schedule.apply_classified` used to turn every `Invalid_argument` escaping application into an
  `Illegal_schedule` decline, so an array bound or a `List.zip_exn` in a transform read as one more
  declined candidate. Now an op precondition raises `Schedule.refuse` (the typed cause) at its own
  site, the classified path passes only those through, and the public `Schedule.apply` re-renders
  them as `Invalid_argument`. Internal probes (`op_legality`, `schedule_legality`,
  `fuse_epilogue_witness`, the hermetic swap/coalesce probes) catch the typed cause, not
  `Invalid_argument`. A new precondition written as a bare `invalid_arg` ends a strict search, on
  GPU-only paths only on a GPU sweep: `schedule_refusal_scan` requires every untyped raise bound
  before `apply_typed` to carry its own `(* invariant: ... *)` comment.
  `Schedule.with_transform_fault` injects one for tests (`schedule_refusal_classification`).
- **The same holds one level up, for the sketch builder and the advisory fallbacks** (gh-ocannl-1390).
  `sketch_schedule` no longer converts every `Invalid_argument` escaping the builder into
  `Illegal_schedule "Autotune.sketch"`: a builder precondition (no site of the family's kind, Grid
  companions that do not align, no coalescible pair) raises `Sketch_families.refuse` — that same
  cause and census key — at its own site, and `schedule_refusal_scan` holds `sketch_families.ml`
  to the marker rule up to `sketch_schedule`. The catches around builder and apply calls in the
  seeding probes (`seeded_ranker`, `seeded_segment_judgment`, the sink probe, the staffed weights)
  take `Cause_at` only, and `model_default`'s placement pick falls back to the default placements
  through `compile_advisory ~on_unrecovered`, which hands it the CLASSIFIED cause alone; it used to
  catch `Invalid_argument`/`User_error` out of the whole compile, which is also what a fatal
  failure re-raises as. A probe that needs "did not apply" catches `Cause_at`, never
  `Invalid_argument`. Any builder call `tune` makes before a candidate's failure boundary needs the
  same catch: the `keep_fraction < 1` model pre-filter scores every sketch by building it, and an
  uncaught refusal there ended the search (gh-ocannl-1437); it now records the refusal in the
  decline census, as the compile would have, and drops the candidate. Seams: `Autotune.with_sketch_builder_fault`; `model_default_placements`
  locates its injected fault by a clean run's op-application count (an elementwise graph: one whose
  consumers are scalar reductions applies no default-pipeline op on cc).
## Placement and dispatchable baselines
- Placement decides which tensorized candidates *exist*, not just how they rank, because
  `mma_tile_for_precisions` keys on the storage precisions of the nodes the site actually reads.
  Under the mixed-precision recipe on a uniform-format backend (Metal's `simdgroup_matrix`: no mixed
  multiply-accumulate) the default-placement arm seeds **zero** mma candidates — the reduced-precision
  cast twins are virtual, so the site reads f32 masters into an f16 destination and no advertised
  tile matches. `Mixed_prec.Twin_materialized` (three small weight casts) restores the whole family
  at the default arm's cost, whereas materialize-all buys it by doubling the kernel count. If a
  reduced-precision cell reports `mma_candidates = 0`, look at the twins before the seeding rules.
- Supplying a `?lowered_transform` bypasses the default annotator entirely (`backends.ml` `compile`
  only calls `Schedule.maybe_default_schedules` in the `None` arm), so **any** code that goes
  through that seam is the unscheduled serial form unless it schedules itself. The autotuner's base
  compile is exactly that, which is why its baseline candidate binds no hardware dimension on GPU:
  the whole routine in one work-item. Such a dispatch is unbounded in cost and uninterruptible on a
  device shared with the display — measured 6.9 s/run on Metal for LeNet (winner 35.7 ms) and hours
  on gfx1151, with driver timeouts and a lost display (gh-ocannl-532). `Autotune.tune` therefore
  does not dispatch an unparallelized candidate on a GPU backend at all: `dispatchable` gates both
  the baseline and every candidate, `baseline_ms` is `infinity` there, and if nothing at all gets
  timed the search stores no cache entry and returns the untuned default compile rather than the
  serial incumbent. On CPU backends the serial form runs at full single-core speed and is still
  timed. When timing anything else through this seam, price the serial form before dispatching it.
- **What a search refuses or dedups costs no codegen** (gh-ocannl-1340): `compile_candidate_gated`
  decides both verdicts inside the lowered transform, on the forms it is about to return (the
  digest is already computed there, before codegen forces placements), and unwinds with a typed
  `Not_dispatched` cause; the base compile does the same with origin `"baseline"`. Before, the
  never-dispatched GPU baseline cost 303 s of hiprtc per arm on gfx1102 and `W_preset[bs=cfg]`
  relinked the base's digest. The callers keep a post-codegen check, on the routine's SHIPPED
  segments (`shipped_dispatchable`, `routine.segments`), since zero-only segments are elided
  after the transform. Consequence: on GPU `baseline_declined` is always false (the gh-533 scratch
  decline is unreachable there, `hip_scratch_tune_survives` is vacuous), and a refused replay
  reports `cache_replay_refusal = Not_dispatched "cache_replay"`. To assert "no compile was
  paid for and thrown away", charge each `on_candidate_attempt` the `Link_after_delta` hits and
  `on_candidate_preflight` runs that follow it, under `tune_deferred` so the shipping compile falls
  after the report (`autotune_serial_baseline`).
- **What a candidate's compile costs is on the record, per compile** (gh-ocannl-1314): every
  compile a `tune` call makes (the baseline, each candidate attempt, a replay, the winner, an
  untuned fallback) writes one `compiled` progress event and one `Autotune.compile_record` to the
  observers `BENCH_TIMING_TRACE` prints, with its verdict (`compiled|dedup|refused|failed`) and its
  phases from `Ir.Compile_cost`: lowering, transform, codegen, the vendor compiler (`device_s`;
  `none` on Metal, which compiles at link) and link, plus kernels rendered and compiled; the
  report sums them in `report.compile_cost`. Read cost questions off these, not off CPU load: after
  gh-1340 the full `gpt2_mini_train` b8 step on gfx1102 (materialize-all graph) timed its first
  candidate 16.9 s in, and a candidate compiled in 6.5 s, 5.8 s of it hiprtc on all 261 kernels
  although a sketch single changes 1-12 of them, against a 4.4 s timing window. The ~300 s
  candidates the issue reported were serial whole-routine compiles gh-1340 now refuses. The HIP
  kernel code cache (backend-memory.md#hip-kernel-code-cache) answers the rest: a candidate compiles
  only the kernels it changed.
- The base compile staying unscheduled is a settled decision, not an oversight (gh-ocannl-552): the
  default pipeline is `maybe_default_schedules` — fission then per-segment annotation, so several
  kernels in general, not one `optimized` to rebase candidates on; every candidate family assumes
  the serial zero point; and annotation reads `hardware_limits`, which would bake per-device
  decisions into the cache's `source_digest`. The "did tuning beat the shipped default?" reference
  is `report.default_ms` instead — the config-thresholds fissioned seed reproduces the untuned
  pipeline exactly and its time is attributed by digest (so a seed that dedups against a timed twin,
  the CPU serial baseline included, still reports).
## Placement flips and search depth
- The flip chain's **enablement prior prices expressibility, and the profitability term is what
  keeps that from costing the run** (gh-ocannl-514 → gh-ocannl-579). The prior promotes a
  `Materialize` flip because materializing that node makes a tensorized family *reachable*, which
  the per-node recompute-cost bound has no term for; it says nothing about whether the family, once
  reachable, is fast. On gh-514's metal/f16 `mlp_wide` cell the two promoted cast twins took budget
  slots 1–2 of a budget-5 chain, the family they unlocked timed 79–92 ms against the arm's 7.5 ms,
  and the cheap `inline` flip that actually won (cost-rank 5) fell out of the budget: enablement
  chains shipped 7.03–7.14 ms where cost-ordered ones shipped 6.55/6.64. **The evidence that settles
  it is already paid for at the decision point** — `Train.tune_placements` searches arm B, the
  all-materialized specialization `placement_enablement` derives its enablement set *from*, before
  the chain walks, so `report.mma_best_ms` against `report.best_ms` prices the promotion for free,
  on this device, on this computation, this session. `Autotune.family_profit_of_reports` reads both
  arms' reports; `tune_flip_ordering=profitable` (the default) resolves to `enablement` when the
  family is competitive or was never timed and to `cost` when it lost by more than
  `tune_flip_profit_margin`. Two things about the shape of that rule are deliberate: **both** of the
  prior's classes go at once (promoting family-unlocking flips and demoting family-breaking ones are
  the same bet on the same family), and **the absence of a confirmation is not evidence against** —
  an arm that seeded tensorized candidates and timed none (the gh-ocannl-521 state) measured nothing
  about the family, so the prior stands and gh-558's budget-5 reachability closure is untouched. Read
  "was one timed" off `mma_best_ms` being finite, **never** off `mma_timed`: those are deliberately
  different populations — `mma_timed` counts candidates whose LABEL promised a tensorized pipeline,
  while a beam round appending a `Tensorize` to a saved or preset incumbent promises nothing in its
  label and is exactly as tensorized, and can win. Keying the guard on the label lets a family that
  lost tenfold read as unmeasured. For the same reason the cache entry carries `mma_best_ms` (an
  optional field, so pre-gh-579 entries stay readable and simply claim nothing): the replay report's
  COUNTERS describe the call and are all zero, but its TIMES are the storing search's, exactly as
  `best_ms` and `baseline_ms` already were — without it the same workload ranks its surface by cost
  on the cold run that measured the family and by enablement on every warm run after it, which is a
  policy that depends on cache state. The
  granularity limit is worth knowing: `mma_best_ms` is per *search*, not per site, so the term
  cannot demote one site's promotion while keeping another's; per-site attribution would need the
  search to report per-site tensorized bests. Alternatives that were weighed and rejected: reading
  prior searches back out of the schedule cache (it persists no tensorized margin, its key is a
  per-program digest so an entry answers about a *different* program, and the in-process arm report
  is strictly better evidence anyway); and pricing the *displaced* flip instead of the promoted one
  (its gain is unknown until measured, which is exactly the budget the promotion consumes).
  `model_default`'s placement walk hands over no evidence, so it gets the prior and is unchanged —
  and the derivation happens inside `placement_surface`, on the `profitable` path only, so the
  ordering of a run pinned to `cost` or `enablement` never reads `tune_flip_profit_margin`
  (`ps_profit` is `None` there, which is what the log line reports instead of a verdict nothing
  consulted; the flip chain's abandonment rule, next entry, does read it). A malformed
  margin is a `Utils.User_error` and must reach the caller: `tune_placements`' containment around the
  decision-surface lowering names the classes it does NOT absorb, because swallowing that one skips
  the refinement the configuration asked for and ships the A/B winner as though the setting had been
  honored.
- **A hopeless flip is abandoned at EQUAL search depth, never against the incumbent's final best**
  (gh-ocannl-1110): `Autotune.tune ?abandon` stops once its best after `beam_width` admitted timings
  trails the incumbent's `report.best_steps` at that depth by more than the margin squared, raising
  `Search_abandoned`. On gh-719's cuda gpt2_mini cell arm A sat at 11.7x its final 6.862 ms for 207 of
  209 timed candidates (the recombination composites delivered the rest), so a final-best rule would
  abandon every flip. `best_steps` is cached like `mma_best_ms`, keyed by every `Search_shaping`
  key whose resolved value is not its default (`Utils.config_class_fingerprint`, `SC.trajectory`),
  so a replayed incumbent still has one; a failed one abandons nothing. A clean abandoned prefix is also persisted
  (gh-ocannl-1136), under the unchanged schedule key with an `abandonment-` key prefix.
  A replay requires the same search shape and re-evaluates the current incumbent and ratio
  against those timings; without a qualifying rule it searches normally. It reports
  `Abandonment_replay` and raises `Search_abandoned` with zero search counters, so a warm
  flip chain does not re-search its losing flips or mislabel the harness's tuned row. **The
  trajectory identity holds only what a single search reads**: the keys that steer
  `Train.tune_placements` above it (`tune_inline_flips`, `tune_flip_ordering`,
  `tune_flip_profit_margin`, `tune_ship_arm`, `tune_placement_store`) are their own
  `Placement_shaping` class, outside the fingerprint (gh-ocannl-1192), and so are the
  `model_default_*` keys, read only by the zero-timing `Autotune.model_default`, as the
  `Untimed_model` class (gh-ocannl-1241). While they were
  `Search_shaping`, merely STATING one changed the identity, since the fingerprint then rendered
  every key some source sets: on rog's approximate CUDA gpt2_mini cell a warm chain run under
  `tune_placement_store=false` (gh-1020's way to compare the arms afresh) replayed both arms yet
  searched each of the eight flips abandoned cold in full (775 s and 310 attempts apiece where the
  cold abandonment took 19 s), because the replayed incumbent had no record under that shape and no
  abandonment record matched. Two traps for reproducing that row: a warm run with the placement
  store on replays the recorded decision as ONE search and never walks the chain; and under a
  commandline-picked profile (`--ocannl_profile=approximate`, whose payload sets 2) the flip budget
  cannot be raised by `OCANNL_TUNE_INLINE_FLIPS`, since the profile outranks the environment --
  pass `--ocannl_tune_inline_flips=N`, which `benchmarks/orchestrate.py` does per cell with
  `--cell-flag tune_inline_flips=N`. **The fingerprint compares values, against the built-in
  default** (gh-ocannl-1272): a `Search_shaping` key renders only when its resolved value, parsed
  as its read sites parse it (`Utils.config_key_defaults` registers each default with its kind),
  differs from that default, so `autotune_keep_fraction=1` against the read site's `"1."` keys
  nothing, while a profile's non-default value renders exactly as a statement of it would. An
  `auto` lane mode is resolved per device (`Schedule.resolve_search_shaping_value`, passed by
  `tune` once `limits` is known), so an explicit mode equal to `auto`'s pick is also the default.
  A new `Search_shaping` key needs a `config_key_defaults` row (the `config_class_fingerprint`
  test fails without one) and is read only through its kind's typed getter (`Utils.registered_int`
  and siblings, gh-ocannl-1332): the getter owns the default and refuses text the kind does not
  parse with `Utils.Invalid_config_value` naming the key (`autotune_rounds=two` used to run as 2),
  `test_config_consistency` fails any other read, and a keyword's modes are its read site's
  `~parse`. The key must also be read within `Autotune.tune`'s static reach (`digest_completeness`
  check 4, gh-ocannl-1331); a key only some backends read goes in
  `Schedule.search_shaping_key_read` (or `Autotune.search_shaping_key_read` for the seeded plan's),
  so `gpu_*` stated on cc keys nothing -- `flip_abandonment` states every key the predicate calls
  unread on cc OFF its default and requires the warm chain to replay its abandonment record. It
  states valid values, not garbage, and so pins the identity, not an absence of reads. The absence
  of reads at the placement level above the search is `placement_surface`'s claim: its mma sites
  are segmented as the backend's own seeders segment (`Sched.fission_keep_mapping ~is_gpu`), so cc
  no longer reads `gpu_fission_keep_mapping` or `gpu_serial_lanes` there (gh-ocannl-1430).
- **A structural twin of a lost flip is skipped without a budget slot** (gh-ocannl-1274). The
  identity is `placement_surface`'s `ps_twin_identity`: every top-level statement of the raw
  program (`optimized.source`, where an inlined node is still present) that sets or reads the
  flipped node, rendered by the schedule cache's canonical walk (`SC.canonicalize_code`), each node
  tagged with the facts its doc LISTS (lineage and decided placements, the chain's reading, the
  timing lineage's tag and decided placement, observability, enablement class, bounds candidate),
  then the readings and recompute cost. It refuses (no identity) where the source cannot be the
  whole local reading: an opaque construct anywhere, an unassigned flipped node, a node with a
  computation inherited from an earlier routine (in the caller's lineage, or in the timing lineage
  via `Train.tune_placements`' timing tag), a statement-level local no statement defines. The
  neighbourhood closes over the locals it reads, but no `Assignments` lowering puts a local's
  definition in a top-level statement apart from its reads (online softmax emits its locals inside
  the reading nest; probed on 2-layer attention, forward, fused backward and block mode), so the
  closure and that refusal fire only on hand-built code: `flip_twins` reaches them through
  `Autotune.flip_twin_identity`. To isolate the inheritance refusal, inline BOTH twins' operands
  into one earlier routine: every rendered tag stays symmetric, so only the refusal separates them.
  **Keep it a stated list, not a promise**: the
  review of staging#1124 ran five rounds, each naming one more fact a flip's outcome can depend
  on. What ended the loop was a stated boundary. Whole-routine context (live ranges) is
  deliberately out, since per-layer twins differ in exactly that, and a false twin costs one
  unmeasured flip, never a wrong value. Add a fact when it is per-node and cheap. Rebut it when
  it is global. Only CLEAN losses anchor: an abandonment or a completed search with no refused
  window and a finite best no faster than the incumbent. The exception is a search-disabled
  report, which is what lets a search-off chain (`autotune_search=false`, where every flip and
  the incumbent sit at `infinity`) list twin classes cheaply. No twin record is persisted. The
  rule is a function of what the chain measured, and a warm chain's anchors replay, so it skips
  the same twins. On cc's approximate gpt2_mini search-off listing (budget 16), the q/k/v and
  gelu flips of layers 2-4 twinned onto layer 1's, plus one more pair: 13 skips, 16 distinct
  classes measured. The per-layer layer_norm flips did NOT twin: their recompute costs differ by
  layer (7168 down to 2304), since what reads them differs.
- **A per-segment single is timed against a backdrop, so near-tie singles rank by noise**
  (gh-ocannl-1166). An `F_sketch` single times the WHOLE routine with every other segment on its
  untuned preset: on gpt2_mini/CUDA tf32 a q/k/v single measured ~23.2 ms where the composite it
  staffs ran at 2.4 ms. Two replicate searches put the same single 0.016-0.031 ms apart, as wide
  as the gap between that site's top four geometries, and the replicates crowned different ones
  (`mma 32x32x0 bgrid` vs `mma 16x32x0 bgrid-in`) — the "step search crowns weaker geometries
  than the per-site search" symptom. Exact mode, whose gaps are larger, crowned the per-site best
  in both. `bench_gpt` tunes with `~rounds:0`, so no beam round refines these crowns (next bullet).
  The composite playoff re-times each key's singles within `autotune_composite_playoff_margin`
  (0.005, i.e. 0.5%) of its best, at most `autotune_composite_playoff_width` (2; 0 = off,
  gh-ocannl-1259) per key, inside the faster timed composite, as coordinate descent over the
  keys; its windows count in
  `report.fiss_sketch_playoff_timed`, not in `fiss_sketch_timed`. It cost ~14 candidates and
  5-26% of search wall per arm. With it both tf32 replicates crowned `mma 16x32x0 bgrid-in`; the alternates
  it beat by ~1.3% inside the composite had tied it within 0.1% as singles, and the replayed step
  moved 2.474 -> 2.427 ms (replicate means) with the spread narrowed 16x. Before reading a per-key
  crown as a ranking, compare the singles' gap with their backdrop's replicate spread; the
  `autotune_log` lines carry both.
- **Staffed singles time a key's sketches inside the running composite** (gh-ocannl-1138, config
  `autotune_staffed_singles`, **default off: measured, not yet worth it**). With the seeded-rule start
  (below) on the gh-ocannl-1109 cell (gpt2_mini, approximate, rog/CUDA, one pair) the first 1% gain came at
  12 s of a search instead of 455 s, but the best came within 5% of final only at 55%/75% of the two
  placement arms' timings (bare: 96%), at +29% search compile time, crown and step p50 unchanged,
  against the plan's bar of under half. The pre-sweep seeds (presets, whole-routine sketches) still
  come first and late keys still lower the best; flip the default only on a re-measurement that
  clears it (entry_version bump with it).
  **Not for HIP either** (gh-ocannl-1528). On minix (gfx1151, **unified**, gpt2_mini_train b8, exact,
  cold caches, `74f53479`), the off cell finished. The on cell finished arm A and was cut by the
  window 4 keys into arm B's sweep. (1) **The staffed backdrop is ~30 ms, not the seeded composite's
  15.5.** Staffing cuts the singles' backdrop from 102-131 ms (preset) to 21-32 ms, but the start
  lacks `F_seeded`'s 30 split-reduce choices and its sunk node. (2) **The backdrop moves only
  near-ties.** Arm A, 22 keys: each key's increments over its best correlate at r = 0.98 across
  backdrops (arm B's 4 keys: 0.91). Kendall tau has a median of 0.56, and 14 of 22 crowns differ, but
  every differing staffed crown sat within 0.62% (<= 0.78 ms) of the bare crown under the preset
  backdrop. Inside the composite the bare crown loses by 0.2-1.8%. (3) **No per-segment composite
  beats `F_seeded` in either mode.** Both arms of both cells crowned the very first candidate (A
  15.57/15.53 ms, B 17.74/17.74 ms). The best composite of singles reached 21.2 ms staffed and
  21.6 ms bare in A, and 18.4 ms bare in B. So the singles' ranking decides nothing shipped on this
  workload. (4) **Staffed singles cost ~3x the compile.** On HIP a staffed single compiles about 67
  of 183 kernels (median 4.9 s; arm A total 1510 s), where a bare single compiles 4 of 178 (1.7 s;
  total 550 s). The staff's sketched kernels miss the HIP kernel code cache, though they are
  unchanged across a key's sweep. Arm A's wall was nonetheless only +6% (2055 vs 1934 s), because
  the shorter windows time faster. What would make the singles matter is a backdrop that IS the
  seeded composite (the seeded plan with one key replaced), not a different setting of this flag.
  The coarse keys are swept one at a time, each single's `staff` holding
  the composite's choice for every OTHER key (label `F_sketch[p @k | staff ...]`; tests read the part
  before ` | staff `, `Fissioned_chain.focus`), and a key's fastest joins the composite after its
  sweep, heaviest key first (segment count times its best sketch's roofline score, logged as
  `staffed: segment <tag> weighs <w>`; unscored keys last, in enumeration order). Only times within
  one key's sweep are comparable, so per-key minima stay valid but a cross-key comparison does not; the composite's own choice is swept first, at its known time and
  without a compile. **The start is the seeded plan's decision RULE, not its recorded decisions**:
  `seeded_params` applied to each key's own segment. The recorded `Seeded_sketch` digests were taken
  on the split routine under the untuned segmentation without the search's zero fold, so they equal
  no key's digest: at `381bbc93` the composite started empty on gpt2_mini/CUDA, its best sat at the
  seeded floor (3.70 ms) until two late keys left on presets were swept, and it came within 5% of its
  final best only at 77% of its timings (bare search: 96%; ABBA, crowns and step p50 unchanged).
  A refused (not contended) staffed single is retried bare once (`fiss_sketch_backdrop_fallbacks`),
  until a staffed single of the key is admitted (the staff is then known compatible, so later
  refusals are the seeds' own). A key's incumbent is its own choice as timed in the sweep, or the
  running composite whole (for a key with no choice, the key on its preset): a single joins only by
  beating it. A key whose staffed singles ALL decline cannot join that
  staff (the composite holding both is the refused code), so its fastest bare single, the composite
  of that key alone, replaces the running composite when faster as a whole routine. The last key's
  winning window IS the coarse composite, so `fs_composite` is set from it at admission (and stays
  `Timed` if a bare replacement later shrinks the composite; the split flags likewise), and the
  playoff skips a key whose sweep ran inside the incumbent.
  The fine segmentation (GPU, gh-ocannl-574) keeps its own composite: a fine key the finer cuts left
  unchanged IS the coarse key and carries the coarse choice, a new one starts from the seeded rule's,
  and only the new keys are swept (`test/operations/autotune_staffed_fine`, skipped off GPU).
  The split-reduce sites keep one more (`F_split`'s `staff`, the other sites' block counts, starting
  empty): staffing a site's prelude with the others' splits forms the multi-site composite during
  the seeds, but every segment of a split candidate still runs its preset, so the split singles'
  backdrop does not shrink the way a sketch single's does, and the seeded plan's split is no start
  worth its approximate site matching.
  `test/operations/autotune_staffed_singles` scripts an additive landscape and seeds the start through
  the `Autotune.on_staffed_start` seam, which only a seeded plan fills in production.
- **Beam rounds after the step search's composites bought nothing measured on gpt2_mini, at +150% search**
  (gh-ocannl-1137; a capped correctness-grade probe on a shared rog box, CUDA, approximate profile).
  A round is adopted only when its best beats the incumbent by `min_progress` (1%); a sub-1% best is
  timed and dropped, and the winner stays the beam head. Arm A finished round 1 (635 candidates,
  +1002 s on 674 s of seeds) 0.50% ahead and shipped its rounds:0 winner. Arm B was cut off 97
  candidates into a 1325-candidate round (~1.9 s each, ~+480% projected on 528 s of seeds), 0.31%
  ahead so far. Hence every benchmark runner keeps `~rounds:0` and records it as each arm's
  `rounds_run`/`beam_width` on the result line. That probe's rounds expanded every beam entry in
  full, and after the composite playoff the beam is the composite plus its near-copies (1-4 of
  122-150 segments apart), so a round re-timed one entry's menu ~4x; the next bullet is the fix.
- **A round expands the beam's head in full and every other entry only where it differs from the
  head** (gh-ocannl-1229, `autotune_round_expansion=differing`, the default). A fissioned routine's
  time is the sum of its segments', and every non-head entry trails the head, so on a segment the two
  share, the entry's move is the head's same move over a backdrop already measured slower: that child
  is dominated before it is compiled. On the entry's differing segments its moves start from a
  geometry the head lacks (the playoff's near-tie alternate), which is the one thing a second
  starting point buys; a differently segmented or whole-routine entry shares no unit and expands in
  full. `full` is the old rule; `head` (the head alone) is the single-entry control, which loses
  exactly the gains that start from an alternate geometry. **The duplication is in the expansion,
  not the admission**: deduping the beam by `units_differing`, or admitting only the playoff
  alternates that swapped, frees the slots for the next-fastest candidates, which on gpt2_mini are
  the per-segment singles and split-reduce seeds at ~29.4 ms against the composites' ~2.44 ms --
  expanded in full, so the round keeps its size and spends three quarters of it on hopeless parents.
  `test/operations/autotune_round_expansion` scripts an additive landscape over the playoff chain
  and pins that `differing` times exactly `full`'s round minus the near-copy's shared-segment moves
  and crowns `full`'s round best. Its searches, like the playoff test's, run on
  `test/support/fissioned_chain.ml`'s small menu (the roofline pre-filter at 0.2 with a synthetic
  envelope, about five singles per segment), and unlike the playoff's its claims depend on WHICH
  alternate seats as the beam's near-copy: its differing segment must offer round moves.
  Tensorized and fully register-tiled alternates offer none (their rounds move only the shared
  segments), and WHICH alternates those are is the cc vector width's: 16 (NEON), 32 (AVX2) and 64
  (AVX-512) bytes keep different singles through the pre-filter, so a seat picked by attempt
  position passed on ARM and on AVX-512 runners and failed six claims on the AVX2 ones
  (ahrefs/ocannl#1363; the scripted crown at 10.00 ms was the round falling short, not a timing
  tie). The test now pins `cc_vector_bytes=32` and seats by the property, read off the report:
  `Autotune.report.round_moves` (gh-ocannl-1360) records each round's beam entries with their
  per-unit `unit_expanded`/`unit_moves`, before the round attempts anything, so a probe search
  with a beam wide enough for every alternate, stopped by raising from `on_candidate_attempt` at
  the first round candidate, says which alternates offer moves on their differing segment without
  compiling one; a setup control claims the positional seat offers none at that width. Reproduce a
  CPU-menu difference between boxes with `--ocannl_cc_vector_bytes=<16|32|64>` on the built exe. `autotune_sketch_placements` runs on the same small menu on CPU backends
  (its claims ask which segments a single sketched, so they need one single per key, not particular
  ones). Before reaching for it elsewhere, count the search's `F_sketch`/`W_sketch` labels in its
  autotune log or progress lines: `autotune_split_reduce`, `autotune_labels` and `flip_abandonment`
  search matvec and pointwise chains that seed no sketch at all, so the pre-filter prunes nothing
  there and their wall time is presets, split-reduce seeds and round moves. The `autotune_log` line `round N: K candidate(s) from a beam of
  W, expansion <rule>: ...` gives each entry's `units differing` from the head and how many of its
  units the rule expanded. Measured on rog (gpt2_mini, CUDA, approximate profile, beam 4, a
  correctness-grade probe on a shared box, two legs at an equal 2100 s cap): arm A's round 1 took
  332 candidates (302 + 7 + 4 + 19) and ~406 s under `differing`, 1204 and ~1490 s under `full`, and
  both reached the same best, 2.2047 ms, 0.2-0.35% under the seed winner, so not adopted. That best
  came from a near-copy's differing segment, which `head` would not have expanded. With the time
  saved, `differing` reached arm B's round (624 = 524 + 48 + 4 + 48) before the cap; `full` did not
  start it.
## Action enumeration and budget sharing
- The action menu's loop enumeration is provenance-aimed **by action category**, not by loop
  (gh-ocannl-687). `Local_scope` has two producers — virtualization's inline at a read site, and the
  accumulator localization `Schedule`'s materializing `Unroll` / `Partition` and
  `C_syntax.try_localize_serial_reduce` mint over a MATERIALIZED cell — and `Low_level.scope_mint` on
  the node tells them apart. `Autotune.collect_loops` descends both and tags each descriptor
  (`ld_inlined`); a loop reached through an inline draws the `Vectorized` retype and nothing else.
  Two things this is NOT about. Not reachability: `Schedule.rewrite_loop` descends every
  `Local_scope`, so a proposal naming an inlined loop applies. And not "which loops exist": the
  first attempt at this dropped them from the enumeration wholesale, which **destroys** the
  candidate rather than moving it outward — `C_syntax`'s elementwise vectorizer bails on any
  `Local_scope` in the body, and an accumulating bailout falls back to a plain serial loop, so the
  enclosing loop's retype renders exactly like the baseline, while the inlined reduction one level
  down is precisely what `try_vectorize_reduce` was built for (gh-639). `contains_loop` therefore
  stays provenance-blind: innermost-ness decides which loop gets the retype, and the renderer
  answers that structurally. What the exclusion buys is the other three categories — up to eight
  descriptors per loop, no evidence any pays on a per-use-site inline, each costing a candidate
  compile and displacing one for the main nest. **When narrowing a search space, check whether the
  thing you are dropping has a renderer the alternative lacks**; "propose fewer things" and "propose
  the same things elsewhere" are different changes. A flag on the node is the durable form of this
  fact; contrast `input_scope_ids` (gh-ocannl-681), which answers the per-call question of whether a
  scope was in the program a given `optimize` was HANDED, and must stay id-set-based: a mint is
  claimable, and hand-built IR has no honest way to spell "not mine".
- The per-unit action cap is shared round-robin across the menu's categories, not spent as a prefix
  over their concatenation (`Autotune.share_cap`, gh-ocannl-685). The menu list is category-ordered
  and UNRANKED, so a prefix over it is arbitrary — a unit whose tensorizes alone reached 48 offered
  the search no split, swap, unroll or vectorize at all, and those are exactly the categories a unit
  needs when its tensorizes turn out `Op_illegal`. Contrast `List.take surface.ps_candidates
  placement_budget`, a prefix over a RANKED list where top-N is the intended semantics; that one is
  fine as it stands. When capping anything else in this search, check which kind of list you have.
  Survivors keep category order, so an under-cap menu is byte-identical to before; the `menu:` log
  now also reports what the cap DROPPED (it used to print only the per-category counts taken before
  the take, so a truncated menu logged the same numbers as an untruncated one) and what the
  provenance filter withheld.
  **A cap must also sit at the right altitude, not just be shared fairly.** `menu`'s `?admits` runs
  ahead of the cap so the budget is spent on moves the caller can use: the beam's GPU rule — an
  incumbent binding no hardware dimension can only be expanded through a move that binds one — used
  to filter *after* `menu` had capped, so a tensorize-rich unit got its share of five categories and
  kept only a fraction of the one category the beam could use. The old plain prefix happened to hand
  all 48 to the tensorizes, so sharing without moving the filter would have been a regression
  exactly where gh-ocannl-685 meant to help. When adding a consumer-side filter over a capped list, ask
  whether the cap should see it.
## Contraction nests and padded tiles
- A site contracting over SEVERAL axes is a matmul site whose k-loop lowering has already split
  (gh-ocannl-683): the matcher's contraction nest is the maximal innermost suffix of loops absent
  from the accumulator's index map — `m_k` the innermost, the rest `m_ko` — and every pipeline
  names "the k-block loop" through `Sketch_families.k_blocks` (the outer contraction loops, then
  its own k-split's outer loop). Before, `classify_matmul` took the single innermost loop as `k`
  and demanded every other loop own an accumulator axis, so attention's out projection
  `{ w_o } * attn` (weight input axes `(head, head_dim)`) was refused and never seeded — a miss
  invisible to the decline census, exactly like the gh-577 refutations: the emitted source
  (no `__shared__`, contraction loops serial inside an 8-block launch) was the only evidence.
  The rule that makes admitting nests safe: a tile-role symbol must be the SOLE symbol of the
  component it owns (`sole_axis`). Without it a convolution's `(ky, kx, ic)` suffix classifies as
  a matmul — `ic` as `k`, the `oy + ky` window axis as `i` — and since `sketch_seed_params` tries
  the matmul family FIRST, the conv family silently stops being seeded; `schedule_conv_gemm` is the
  test that catches it (11 claims), so run it whenever the matmul classifier is relaxed.
  Two things the generalization does NOT do: the family does not coalesce the nest into one loop,
  so within it a tile's k-extent is judged against the innermost contraction extent alone (the
  coalesced branch's contraction layout merges the pair instead, gh-ocannl-1279, below); and the
  whole-`m_k` forms (the unstaged `bk = 0` tensorize, the CPU
  whole-triple) keep the outer contraction loops above the block statement. Accumulator contraction
  then promotes one logical fragment around that whole loop chain, so this shape is
  **fragment-scoped for capability purposes**: a wide-f16 backend that advertises only
  per-statement MMA must not seed even `bk = 0` for a multi-axis contraction. Conversely, a
  staged split whose padded block count is one has no nontrivial enclosing reduction and remains
  per-statement. Pinned by
  `test/operations/schedule_contraction_nest.ml` (detection, every family's construction, GPU
  blocktile execution, CPU-family execution on cc) and the scope negative control in
  `test/operations/sketch_family_tree.ml`.
- Non-multiple extents PAD in both GPU matmul pipelines, not only the tensorized one
  (gh-ocannl-730). `Sketch_families.pad_composition_ok` is the shared judgment both consult, from
  their seeding gate and from their `pad_to` triple alike: a geometry may replace a divisibility
  gate with a pad exactly when every operand is read through a zero-fringe staged tile (`Stage`'s
  per-axis `Where` guards store 0 out of range), which both GPU pipelines satisfy whenever their
  k-block is staged. What made the blocktile family look different was never the staging — it
  stages both operands at every geometry — but what a pad leaves BEHIND: with no `Tensorize` to
  absorb the masks the leaf keeps its `If`, and `Schedule.Privatize` used to reject any guard
  mentioning a symbol bound inside the accumulation loop, so a padded candidate died at
  construction with "varies across the accumulation" rather than at the gate. Privatize now
  classifies: an iteration-invariant condition still gates the init-load and store-back (staging#91's
  lane-restriction rule, unchanged); a condition free of hardware-typed loop symbols is an
  iteration mask over one thread's own accumulator, kept on the update alone; and a condition that
  IS the target's own index compared against a bound no larger than that axis's dimension is a
  row/column pad mask whose transfers already carry the identical edge guard. Everything else is
  still rejected, and that line is load-bearing: a guard mixing a thread-selecting symbol into a
  varying condition would leave non-updating lanes storing back a stale accumulator.
  Measured on Metal (M4 Max, f32, attention out projection at head_dim 12, `b=8 s=512 j=768`,
  `bk` 8 and 16 both padding 12 to 16): all ten padded candidates match the serial reference
  BITWISE — the padded k slots contribute exact zeros, so a pad is not an approximation — and the
  best costs 13.15 ms against 12.84 ms for the same geometry at head_dim 16, i.e. the pad costs
  the padded arithmetic plus ~2% and no new class of overhead, replacing a 1615 ms untiled kernel.
  One piece of residue: the CPU blocktile pipeline (in-kernel and
  hoisted packing) keeps the gate on purpose. Its stack-scratch tiles are zero-fringe too, so the
  pad is sound, and it is free: padded candidates were bitwise-exact and no slower than the
  dividing control. But the family loses to the CPU tensorized family even where it divides
  (gh-ocannl-741, out projection b8 s512 j768 h4, best blocktile vs best tensorized: at the
  dividing head_dim 16, 18.0 vs 13.5 ms on mac-studio and 21.2 vs 17.3 ms on AVX2 rog; padded at
  head_dim 12, 16.1 vs 13.1 and 20.8 vs 16.9 ms). Padded leaves would only lengthen the cc
  search, so the gate stays, and it still renders the gh-ocannl-683 k-extent label that
  `schedule_contraction_nest` reads.
  Revisit only if the family starts winning where it divides.
- A hand-written packed GEBP at extents its blocking does not divide pads BEFORE it splits, as the
  tuner's builders do (gh-ocannl-627): `bin/bench_gebp.ml` is the builder of both
  `narrow_gebp_bench` and `schedule_bench`'s packed variants (gh-ocannl-1562), executed by
  `gebp_padded_parity` and, as the shipped benches, by `narrow_gebp_bench_padded` and
  `schedule_bench_padded`. Only a ROW
  remainder needs the pad — `Split`'s guard on `i_i`'s body stops the `Swap` that sinks `k_o`; a
  k remainder's guard lands inside `k_i`, where `Tensorize` reads it as a mask. Every pad needs
  BOTH operands staged: `Tensorize` refuses a pad guard over an operand read in place ("requires
  operand ma to be a zero-fringe staged tile"), so `schedule_bench`'s `pm_hoist` (A in place) keeps
  the divisibility gate, by `Bench_gebp.in_place_a_refusal`. A hoisted (link-time packed) B~ panel
  pads fine: its edge-tile pad slots are zero-filled on the host. A row mask forces
  `Tensorize`'s accumulator contraction, and in the serial flavor (`k_o { i_o { .. } }`) no loop
  qualifies, so the transfers land at the statement site: a storage-precision fragment round-trip
  per row block and k block, ~11-15% beyond the pad waste on AVX2 rog (gh-ocannl-627 cells,
  gh-ocannl-1562). Since gh-ocannl-1563 `contract_tensorized_accumulator` splits the block loop at
  its last fully valid block (`full_block_split`): full blocks keep the `Tile_mma` on the output,
  only the edge block contracts, and the trailing simplify folds the full segment's tile edge
  guards. It applies only under a reduction loop (some enclosing loop of extent > 1 leaves the
  accumulator unindexed) on a `Serial` block loop whose symbol is every mask's one outer term, with
  no GPU shared fragment. A whole-K site keeps the single contraction ON PURPOSE: its store-back
  runs once per output tile, which is what `Fuse_epilogue` relocates a tail to, and
  `fuse_at_fragment` neither descends a segment starting above 0 nor fuses two sites; under a
  k-block loop it refuses either form on exactly-once, so splitting there costs nothing. The
  `Grid` flavors are untouched (a `Grid` loop is not split; `Grid_outermost` contracts at `k_o`).
  The census then reports two register-tiled `Tile_mma`s for one padded serial GEBP;
  `gebp_padded_parity` pins where each row block accumulates.
- What a pad MAY do is not what it SHOULD: `Sketch_families.judge_pad_waste` (gh-ocannl-740)
  excludes, per geometry menu, a geometry whose padded work is at least `autotune_pad_waste_max`
  (2) times that of the menu's least-padded PROPOSED geometry, witness naming both ratios. Relative
  on purpose: on a 6-row site every row block pads 2.67x and that one was Metal's best sketch, while
  catching a 64-row block on a 32-row site (2x, timed 34x off) needs a gate as low as 2x. Its blast
  radius is the CPU packed menu's 128/256-deep blocks, which pad every site below them — a test
  whose claims need the old menu lifts it (`Autotune.lift_pad_waste`, or `inf` as
  `Fissioned_chain.small_menu` states it). The conv family's menus are judged too since
  gh-ocannl-1393 (the CPU `conv-shape` menu on the row, the GPU `conv-rows` menus on row,
  out-channel and reduction), and exclude NOTHING at the default 2x by construction: a row block
  needs at least two blocks, so its row waste is under 2x, and the intrinsic-tile pads of the
  out-channel and reduction extents are site-wide (every conv geometry pads them alike), so they
  cancel in the relative ratio. A small-channel conv (3 input channels on an 8-deep tile) is never
  excluded for its channel pads: the menu has no less-padded sibling to replace it with, and an
  absolute gate would empty the family. The judgment is a witness channel and a lever for a
  lowered threshold. Trap for a test lowering the threshold: the trees are lazy and read the key
  where a menu is built, so walk the tree while the setting is in force.
- **The coalesced layout is a search BRANCH over another lowering, not a level of the family tree**
  (gh-ocannl-1165). The q/k/v site `d[b,s,h,e] += w[h,e,k] * x[b,s,k]` caps its column tile at one
  head; `Sched.Coalesce` merges the adjacent `(h, e)` pair into one loop whose accesses read
  `Sub_axis; Iterator f`. `Sketch_families.coalesced` builds the structural prefix (the site's
  `Coalesce (last m_bi, m_j)`, preceded where `d`'s `Zero_out` shares the routine by its
  `Expand_zero` with the zero nest's last two loops coalesced alike, and where a covering per-cell
  zero nest of `d` does — the folded zero of a GPU fission segment, gh-ocannl-1175 — by that nest's
  own coalescing) and applies it to a hermetic
  copy; `coalesced_seed_params` enumerates the ordinary family over THAT code, so every refutation
  (launch caps, companion coverage, divisibility) is judged on the site the schedule builds, and
  `sketch_schedule` on an `sk_coalesce` seed re-derives the prefix and prepends it. The expanded
  zero nest is then an ordinary companion of the merged chain; an unexpanded `Zero_out` would keep
  the uncoalesced chain. Winners persist as plain saved schedules — the `Coalesce` op names its
  loops structurally (`Base`/`Minted` refs, `Coalesce_merged` mint role) — and the menu change
  bumped `entry_version` to 13. v1 boundary of this (columns) layout, each a decline rather than
  a wrong value: unfused GPU scalar blocktile leaves only (its OUTPUT is flattened, and both
  `Tensorize`'s fragment geometry and `Fuse_epilogue`'s per-axis tail relocation decline a
  `Sub_axis` accumulator index); a companion nest over the
  uncoalesced axes (an elementwise tail) trims the merged chain and refutes the whole branch; one
  pair per site. Seeding discards its probe's minted symbols (`Indexing.discarding_symbols`); the
  prefix a schedule carries keeps its own. `test/operations/schedule_coalesce`; the scalar
  pipeline's flattened operands are why `Stage`/`Privatize` read `Affine.axis_extents`
  (lowering-and-analysis.md).
- **The branch has two layouts, named by `sk_coalesce : coalesce_layout option`** (gh-ocannl-1279):
  `Coalesce_columns` above (label `coal-n`), and `Coalesce_contraction` (label `coal-k`) for the
  attention out projection `d[b,s,j] += w[j,h,e] * x[b,s,h,e]`, whose contraction nest
  (`m_ko = [h]`, `m_k = e`) capped the k-tile at one head. Its prefix is `Coalesce (last m_ko, m_k)`
  alone, prepared by `coalesce_reduction_pair` (moved into `Sketch_families`, the one preparation
  gh-ocannl-1414's split-reduce pair sites use too); output coordinates are untouched, so the zeroing is not
  coalesced and stays an ordinary companion. The prepared code is RE-DETECTED and the prefix kept
  only when the new site is the expected one (the merged loop as `m_k` over `h*e`, the pair gone from
  `m_ko`, output roles unchanged): a weight stored `w[h, j, e]`, a per-head-only operand or a padded
  inner axis declines at the op. The contraction layout enumerates its own bounded K menu
  (`blocktile_geometries`): depths 256/128/64/32 over 16x16 of 2x2, 32x32 of 4x4 and (gh-ocannl-1459)
  64x64 of 4x4 — 256 threads, the block CUDA's uncoalesced winner uses — never the family's 8/16
  (those are the uncoalesced site's own tiles); a depth whose half already covers K is refuted
  (`k_menu_gate`: only pads). Mind the memory floor: 16x16x256 f32 is exactly Metal's 32 KiB and is
  admitted (executed bitwise there); 32x32x256 is 64 KiB and refuted on every current GPU; 64x64
  needs 32 KiB at depth 64 (CUDA admits 64 and 32, HIP 128 too). So a test executing seeds must
  enumerate them under the device's limits, not no-limits. The row role is `s` (the batch is a
  batch loop), so pad waste excludes the 64-row block wherever `s` is below 64 — a fixture pinning it
  needs 64 rows of `s`, not of `b*s`. Because this layout flattens only the OPERANDS, it also admits
  (gh-ocannl-1459) the STAGED tensorized leaves and the `Fuse_epilogue` twins of both pipelines:
  staging rewrites the operand reads into dense cooperative tiles before `Tensorize` looks, and the
  accumulator and store-back site are the site's own. Its mma menu is a staged K menu (16 or 32 rows,
  `bn` the lane width, the same depths) with no unstaged `bk = 0` geometry, whose in-place
  flattened reads `Tensorize` declines (`schedule_coalesce`'s `ckt` legs pin both sides). Hoisted
  packing is not reached: the branch is GPU-only, and hoisted `Stage` refuses a flattened source. `Stage`, `Privatize` and `Split_reduce` needed no change for the
  merged k (the flattened arm of gh-ocannl-1148/gh-ocannl-1147 already covered it); `schedule_coalesce`'s hand-built
  contraction legs pin that before any seed relies on it. The k-split of the merged loop is not a sketch twin: it stays
  gh-ocannl-1414's `F_split` candidate (`sr_coalesce = Some (h, e)`, label `coalesce:h*e`), seeded only
  below `sr_out_max = 4096` output cells, so the production out projection (262144 cells) gets
  serial-K tiles and no split, while its merged-k `Split_reduce` is still `Op_legal`. Both routes
  are pinned in `split_reduce_pair`: the split site records exactly the pair the contraction prefix
  coalesces, and on the prefix-prepared code the merged k is an ordinary uncoalesced site whose
  prelude, behind the prefix, gives the same exact sums. Measured (`benchmarks/gh1279_cells.sh`, isolated out projection,
  j = 256, (h, e) = (8, 32); best coal-k finalist against best uncoalesced one, fwd/rev legs):
  HIP gfx1151 1.23x on both production sites with depth 32, the search crowning a coal-k seed;
  CUDA RTX 5070 Ti 0.90x there, because the uncoalesced winner is a 64x64 block (256 threads)
  the K menu does not offer, and the search crowns uncoalesced refinements; Metal neutral within
  its desktop co-tenant noise. Depth 256 won only on CUDA's 16-row small-output site (1.43x), where
  the search's crown was the split of the merged k (partials plus combine, 0.0096 ms) instead.
## Timing objectives and calibration
- **What the tuner's numbers are a measurement OF is a config choice, and the two objectives do not
  crown the same candidate** (gh-ocannl-755). `Autotune.time_routine` takes a `timing_mode`, from
  config `autotune_timing`: `isolated` is the historical one launch plus one host sync — a lone
  dispatch's latency — and `queued` (the default since gh-755) dispatches a calibrated batch back
  to back, syncs once and divides, which is what a kernel sustains inside a stream that already has
  work in it, i.e. what a training step presents to every kernel of a layer. Measured at the
  gpt2_mini out-projection shapes on gfx1151, over the ten seeded blocktile geometries, four
  interleaved runs (`bin/projection_shape_bench.exe 200 8 d {fwd,rev} seeds`, whose closing table is
  this comparison): the isolated crown differed from the independently measured batched crown in
  2 of 8 site-runs and the queued crown in 0 of 8; discordant candidate pairs against the batched
  ranking were 17/220 isolated against 5/220 queued. The mechanism is that the round trip is ~50-60
  us while the fastest candidates run in 60-70 us, and the offset is NOT a constant — within one
  run it spans 39-86 us across candidates, which is larger than the 5-8 us separating the top two.
  Two consequences. A `best_ms` is only comparable to another taken under the same setting, and
  under `isolated` it is not a throughput number at all (the gh-ocannl-728 arc compared one to a
  batched harness figure and the agreement was a coincidence of two instrument errors). And the
  batch depth is the thing to check when a queued search behaves like an isolated one:
  `Autotune.queued_batch_depth` targets ~10 ms of wall per batch and floors at 1. CUDA/HIP cap at
  2048; cc and Metal retain the historical 200 cap (Metal's measured ~59-launch target is below
  both, while raising cc's cap only multiplied CPU-suite cost). A routine slower than 10 ms per
  launch is measured identically in both modes by construction —
  `test/operations/autotune_timing_modes.ml` pins the policy and, via a `n[0] += 1` routine that
  counts its own launches, that the reading is per launch rather than per batch.
  On Metal that 10 ms target is now a measured safety boundary too (gh-ocannl-828). On an M4 Max,
  macOS 26.6.2 build 25G83, current-master `schedule_bench` showed no superlinear queue cost from
  256³ through 512³: its one-thread kernel rose from 210 ms to 2.32 s, and two queued runs stayed
  within 11% per launch at every size. The OCANNL-free
  `benchmarks/runners/ocannl/metal_queue_probe.ml` separates four submission shapes over the same
  long kernel. Raw back-to-back command buffers still overlapped at a 1.586 s single-kernel time.
  The arm the gh-828 session read as "the exact `Metal_backend` SharedEvent shape" (kernel,
  signal, wait+kernel, signal) overlapped through 1.198 s and serialized by 1.213 s — but that arm
  had encoded its wait AFTER the second kernel's compute pass, where `encodeWaitForEvent` orders
  nothing, so it measured two effectively unordered buffers, and the transition it found is the
  driver serializing long unordered command buffers, not a response to the SharedEvent shape
  (gh-ocannl-909). The backend's actual shape — the wait encoded before the compute pass, as
  `Metal_backend.link_proc` does — is the probe's `event-chain` arm since gh-909 and serializes at
  every kernel length (1.001x of a synced pair at 150 ms; `test/operations/back_to_back_runs`
  pins it executed at ~5 ms); the misplaced-wait shape is kept as the `wait-after-kernel` arm
  (0.501x at 150 ms) as the record of the artifact. Neither unordered arm reproduced the
  historical ~30x penalty: wait-after-kernel changed from ~1x to ~2x single-kernel wall. What
  survives for the autotuner is unchanged: `queued_batch_ms` sits about 120x below the unordered
  regime's transition, any estimate at or above 10 ms already selects depth 1, and the backend's
  own launches are ordered regardless. Do not add a per-repeat host sync to the timing path on
  this evidence; rerun the standalone probe after an OS/driver change if the superlinear symptom
  returns. The probe takes a duration target it feedback-corrects to within
  2% (it prints the requested and achieved single-kernel ms and the iteration count it settled on)
  or an exact `--iterations=N`, so a narrow threshold rerun pins the count from the previous
  report instead of re-adjusting the target by hand.
  On CUDA/HIP a synchronized single dispatch includes the round trip batching removes, so it selects
  only a provisional depth; a queued probe at that depth separates fixed synchronization cost from
  marginal launch cost, and that affine wall model selects the final depth. This matters even when
  the provisional batch is
  shallow: dividing its wall by depth would charge part of the fixed synchronization to every
  launch and leave the final batch below the contention scale. The selected depth is validated by
  up to four further batch probes; each short probe refits the affine model. The first probe that
  reaches the target is first interpolated back inside a measured below/above bracket when it
  overshoots, then confirmed at a 25% deeper depth (clamped to and measured at the cap) and retained
  only when the pair's inferred fixed component is below the target and no more negative than a
  quarter of the larger of the target and the base's wall (the noise tolerance matching that
  confirmation step; relative since gh-ocannl-1098, because the absolute 2.5 ms was under 1% of a
  slow candidate's batch and its slightly superlinear pairs never fit), so one fixed stall
  or a physically invalid negative fit cannot select a shallow final depth while ordinary
  submit/sync overhead remains in the wall model. When a clean pair's fixed component alone fills
  the wall target, its positive marginal slope selects a depth carrying ~10 ms of launch work: the
  fixed stall does not select a shallow base, and a genuinely slow kernel does not jump to a
  many-second cap batch. A confirmation more than 2x its supported target-sized base
  is itself treated as a contention outlier and retried once before an unresolved pair selects the
  cap, so one transient stall cannot inflate both the timed batch and its later refusal threshold.
  **Every unresolved outcome is then wall-bounded** (gh-ocannl-1096): the cap bounds launches, not
  wall, and a ~61 ms gfx1151 candidate whose slightly superlinear batches never fit was timed in
  126 s batches, 2016 s for one call. A
  NaN-wall outcome of `calibrate_and_time` now settles no deeper than the deepest depth it MEASURED
  within the target (depth 1 when none): slow candidates fall back to the isolated reading, a fast
  one whose deeper probes stalled keeps its deepest clean batch, and one whose single launch owed
  it a batch but that measured none within the target first takes ONE rescue probe at the
  shallowest over-target batch projected linearly to the target (`rescue_depth` in
  `autotune.ml`; any cost whose per-launch average does not fall with depth reads within the
  target there, so a genuine queue threshold below the provisional depth is timed and cacheable),
  and is REFUSED only if that also reads over — as `unbatched`, not `contended`, counted in
  `report.timings_unbatched` (a subset of `timings_contended`, like `timings_unreadable`, so every
  cache and completeness
  gate is unchanged), because its depth-1 reading would be the isolated objective. `unbatched`
  names what was measured, not a cause: host load stalling every batched probe reads the same, so
  consumers (the benchmark JSON's per-arm `timings_unbatched`, `gh834_cells.sh`) keep treating it
  as an incomplete measurement; one that repeats on an idle rerun is the threshold. A sampled
  depth-2 batch with a resolved deeper confirmation whose marginal work fits the target stays at
  depth 2 when synchronized singles owed batching and the fixed term is below the target
  (gh-ocannl-1184): on gfx1102, pairs near
  `(2, 19.14 ms)` / `(3, 28.28 ms)` fitted a one-launch wall just above 10 ms despite marginal
  work below it, then refused that isolated settle and vetoed every cache. Keep the directly
  measured batch, never an unmeasured depth 2 projected from deeper points; unresolved or
  over-target marginal work still refuses and suppresses the whole comparison's cache. A sampled
  shallower crossing is refitted against the batch above it and never settles past that batch: a
  fixed-dominated refit projects far deeper, unmeasured, where a queue cost may jump. Every other
  settle is capped at `Autotune.queue_depth_projection_factor` (2) times the deepest batch probed
  (gh-ocannl-1100), one chokepoint after the branches rather than a fix per exit: the last
  validation's affine projection, a linear scale from a below-target confirmation and a fit wanting
  the cap all used to settle unmeasured (a pair (2, 12.25) / (3, 12.5) wants depth 40). A bound in
  depth, not wall, spending no extra probe after validation ends. A fit wanting an unmeasured cap
  spends its remaining validations on measured doublings toward it (gh-ocannl-1144), including an
  initial cap projection; settling immediately left sub-5 us kernels below the target despite
  unused probes. Each doubling is charged to the existing wall and count budgets, and a measured
  cap retains the same fit and fallback checks. `autotune_timing_modes` pins convergence, probe
  exhaustion and a stalled doubling that spends the wall budget. Do not replace
  this with a bound
  extrapolated through a per-launch cost (least `wall / depth`): the readings that leave the fits
  unresolved cannot tell a host stall from a cost that jumps past a queue threshold, and two review
  rounds on staging#846 each built a threshold device that defeated such a bound (400 and 600 ms
  timed batches); monotonicity of wall in depth is the only premise that survives both.
  **The probes themselves are wall-budgeted** (gh-ocannl-1098), because the fallback bounds only
  the settled depth: the doubling retries still spent 760 launches (~49 s) on an unresolved 64 ms
  candidate and ~1600 s on a threshold device. A probe stops at three minima once its own wall
  passes two target-sized probes' (a 64 ms step's depth-2 confirmation costs 3 batches, not 12),
  and once the probes' summed wall reaches `Autotune.queue_calibration_wall_ms` (eight
  target-sized probes) nothing but the rescue starts and the calibration ends unresolved. These are
  WALL budgets, not per-launch bounds, so the staging#846 threshold devices have no extrapolation to
  defeat; they are the regression fixtures. A converging calibration's probes are target-sized and
  never reach either budget (pinned: the clean fast device still takes twelve minima per probe).
  A resolved fit between two over-target batches projects to the fit's first target crossing and
  samples it rather than keeping its base, which could be any length (a doubling that reached
  depth 4 of a 64 ms kernel would keep 256 ms batches). None of this changes the objective (the depth picks the scale;
  an entry timed at an older depth is an accurate, merely expensive, reading), so no cache-key
  generation bump. `autotune_timing_modes` pins each change on the injected clock with a claim
  that fails without it. Its probe claims read the call record's `calibration_probes`
  (gh-ocannl-1119), one entry per probe tagged with the branch that started it, never the `batch`
  calls: runs of same-depth batches merge a stall retry into its confirmation, so the retry escaped
  the budget claim, and chunking them counts probes only as a lower bound. A new probe site passes
  its own `~role`; the test's witness claim holds every reported probe, and the timed loop after
  them, against the device's batch log.
  An unresolved first pair retries at double depth, and the next fit uses the two batch observations so an
  inflated synchronized-single window cannot force the cap. If the last bounded probe first reaches
  the target, the interpolated target depth is still sampled and checked against the measured
  overshoot. A non-monotone confirmation scales from the deeper measured batch instead of returning
  to the earlier suspect target crossing. After four noisy but resolved misses the latest projected
  depth wins rather than jumping
  to a 20--30 ms cap batch, because such an overlong batch would blunt the 2x contention threshold.
  Metal and cc retain the historical single-estimate path and 200 cap, so this repair adds no probe
  or timed work there; Metal's measured ~59 depth is unchanged. On faster CUDA/HIP kernels
  calibration grows the batch toward the same wall target. A CUDA/HIP synchronized-single estimate
  at or above the target is likewise checked at depth 2: a genuinely slow routine retains depth 1,
  while a transient single-window stall cannot bypass batch validation.
  CUDA/HIP cache keys spell the current queued policy as `queued-v3`
  (`Schedule_cache.queued_objective_version`, held to the `cache_version_history` rule: a bump
  appends its history line) while entries and user-facing configuration still say `queued`. Generation 2 separated the old depth-200 scale (gh-ocannl-892);
  generation 3 separates the short cap-bounded windows repaired by gh-ocannl-1144, so an old winner
  cannot bypass the new calibration. cc/Metal keep the unversioned `queued` key because their policy
  did not change.
  Since gh-ocannl-855 the top-up budget accumulates PER-LAUNCH samples, never queued-batch wall.
  The jitter-sensitive synchronized-single calibration and every timed window have a 16-sample
  floor; the CUDA/HIP-only, already-millisecond batch probes use twelve minima because they choose
  scale rather than rank a candidate. A host stall can no longer spend the timed budget and collapse
  a min-of-N to three samples. `Autotune.timing_result` also marks a
  window contended when at least half its raw wall samples exceed their minimum by 2x. Queued mode
  tests that dispersion on BATCH wall before dividing the ranked and budgeted samples by depth, so
  the division cannot hide a fixed host stall. The search neither ranks a contended timing nor
  caches any winner from a search with one or more contention refusals; `report.timings_contended`
  records every refusal so the incomplete candidate set is diagnosable and a later cache-cold call
  retries it. Falling back to depth 1 would silently change the queued objective back into the
  isolated one.
  **Metal queued contention gets one fresh window before it becomes a candidate refusal**
  (gh-ocannl-1060). On a loaded M4 Max the refused batches already had 10--14 ms median walls
  at depths well below 200, so raising the queue cap did not address the observed dispersion.
  `time_routine` instead enables one retry at the same calibrated depth, with the same sample
  budget and 2x-majority rule. The windows are never pooled: the first is discarded and the
  retry's own verdict reaches ranking and refusal accounting. Persistent contention still refuses
  once and prevents caching; a recovered window leaves the candidate measured.
  `report.timings_retried` and the closing log count discarded first windows even when their
  retries recover (gh-ocannl-1191); final refusals alone still govern cache admission. Retries
  count as they start, so partial/fatal reports retain them too; diagnostic control timing is
  excluded. A depth-one retry takes fresh singles instead of resuming the refused calibration window. Only finite
  positive contention readings qualify, not an unresolved clock.
  The call record's `discarded_window` accounts for the discarded window's extra dispatches, while
  its `window` describes only the returned one; `autotune_measured_refusal` labels the
  retries separately from final host refusals and prints each returned depth and median wall.
  cc, CUDA/HIP and isolated timing keep their existing dispatch counts and policy. This adds
  sampling of the same objective, so cache-key generations stay unchanged.
  **That 2x-majority rule is a statement about a ~10 ms batch, and applying it anywhere else is a
  scale error** (gh-ocannl-888). The queued calibration samples ONE dispatch plus one host sync;
  on a GPU the dispersion of that quantity is the round trip's own heavy tail, and a majority above
  2x the minimum is ordinary rather than evidence of a host stall — a 08-31 CUDA probe read
  0.143874 / 0.052629 / 0.016898 ms. When `queued_batch_depth` refused on that verdict, the refusal
  was returned as the candidate's timing, so every search over microsecond kernels timed NOTHING on
  both GPU backends while `cc` (no round trip to disperse, and the only backend per-PR CI runs)
  stayed green. So `Autotune.queued_batch_depth` is **total** and never reads `contended`: a depth
  is a scale estimate whose error is bounded both ways (floor 1, cap 2048), and a deeper batch is
  the REMEDY for dispatch dispersion. Refusal happens once, downstream, on the timed loop's own
  window, which is a batch. Correspondingly `sample_min`'s `contended` is dispersion only: a
  non-positive or non-finite minimum is a clock that resolved nothing, refused by
  `Autotune.admitted_timing_ms` on the number itself, and such an estimate batches at the cap
  because sub-resolution readings are exactly what queueing exists to resolve.
  Symptom to recognize: the call record's depth (printed by `autotune_timing_modes` to stderr)
  reporting depth 1 where a batch was expected, alongside `NOT TIMED` for every candidate. Note
  that `autotune_timing_modes`' own device-facing claims are all written `... || contended`, so a
  total refusal reads there as a pass — the tuner tests are what catch it. Downstream,
  `family_profit_of_report` treats such a partial report as `Unmeasured`, and benchmark JSON carries
  the refusal count on each tune arm so an artifact never presents it as a complete measurement.
  **The objective is a cache-key component** (`Schedule_cache.key_components`' `timing`, classified
  `Keyed "timing"`), which is the one place this differs from its `autotune_*` neighbours: those
  change how carefully the SAME quantity is measured, so either process's winner answers the
  other's question, whereas an isolated-crowned entry answers a question a queued search did not
  ask — and, left unkeyed, a warm cache would replay isolated winners forever and defeat the
  default outright. Practical consequences: changing `autotune_timing` re-tunes rather than
  replaying, pre-gh-755 entries are never replayed, and a `Cache_replay` report's times are
  therefore known to be this call's objective, which is what lets `Autotune.report.timing` (and the
  benchmark JSON's `timing` field) be filled in on a replay at all. That field is not an option and
  the JSON's spelling is never `null`: `tune` resolves the objective from `?timing` or the config
  before it constructs anything, so a report with no objective is not a state it can be in. The
  template every report starts from, `Autotune.no_search_report`, therefore takes `~timing` — it is
  a function of the objective rather than a constant, which is the only reason the field can be
  plain (the option it briefly had existed solely to let that constant exist).
- **Queued timing's search cost is dominated by candidates SLOWER than the batch target, not fast
  ones** (gh-ocannl-834). A gpt2_mini tuned search on gfx1151 took 794 s isolated against 1370 s
  queued: equal timed loops, but 587 s of calibration against 14 s, because 215 of 220 calls settle
  at depth 1 after ~40 calibration launches (sixteen singles plus the depth-2 confirmation) and
  then time exactly what `isolated` would. `BENCH_TIMING_TRACE=1` in the benchmark runners splits a
  session's wall this way; `benchmarks/gh834_cells.sh` is the per-box driver. Its per-call
  `decision` line (gh-ocannl-1199) is the `decision` of the call's `Autotune.timing_call` record,
  built from the calibration's own metadata: each exit of `calibrate_and_time` returns its
  `batch_settle`, and `fit_queued_batch_depth_between_with_cap` returns the `affine_fit` (fixed,
  marginal, verdict) behind every depth it chooses — so a new calibration branch must name its
  settle and verdict, and a diagnosis reads them instead of re-deriving the policy from probe
  readings. **Everything a timing call did reaches observers as that ONE record, when the call
  returns** (gh-ocannl-1263): a search's own accounting takes it through `time_routine`'s per-call
  `~on_call` (that is how `report.timing_decisions`, the settle × admission counts each benchmark
  arm carries in `results.jsonl`, is counted, so neither a trace nor a search in another domain
  moves it), a trace through the domain-local `Autotune.install_timing_call_observer` /
  `with_timing_call_observer`. Add a new per-call fact as a field of the record, never as a seam
  fired mid-call: the four such seams it replaced (depth decision, probe, retry, window) made the
  harness pair them up by arrival order (`depth_at`, `preflight_at`), the class of bug the trace
  kept patching. A call that raises returns no record; the trace counts those as pre-dispatch
  validations started less tagged records returned. A cold schedule cache
  is not a cold session: the backend's own compiled-code cache persists across processes, and in
  gh-ocannl-834's CUDA pair the second session's compile-and-bookkeeping time was 64 s against the
  first's 406 s from the PTX ComputeCache alone — the driver gives each CUDA session an empty
  `CUDA_CACHE_PATH`; on backends whose cache it cannot redirect, warm it with a discarded session
  first (a fresh OUT does not reset it), then run the modes ABBA. Since gh-ocannl-1074 a depth-1
  settle does not time that window again: the calibration's singles are depth-1 batches taken under
  the timed loop's own stopping rule, so `sample_window ~prior` resumes them as the window and the
  loop only tops up past the caller's `repeats` floor. That is not a change of
  objective (no cache-key generation bump): the reading is still a min-of-N synchronized singles
  judged whole by the 2x-majority rule — only the redundant second window is gone. The trace's
  `timed` share at depth 1 therefore drops to ~0 while `calib` is unchanged; each call line says
  how many batches it `reused`. `Autotune.calibrate_and_time` is the whole policy behind an
  injected `batch` function, which is how `autotune_timing_modes` counts a call's launches exactly
  without a device or a machine-dependent slow routine. The gh-755 offset
  had shrunk to 0-6 us on the same site by 2026-09-27, yet the isolated crown still moved in 3 of 4
  site-runs (gh-ocannl-833); it moved in 3 of 4 on M4 Max Metal and 0 of 4 on CUDA sm_120, and
  queued crowned the batched winner everywhere. On CUDA the time inside `time_routine` was 1.35x
  isolated under the post-gh-ocannl-1074 policy, but a CUDA session pair's whole-search wall is
  confounded by run order: the driver's PTX ComputeCache (`~/.nv/ComputeCache`) serves the second
  session's kernels warm, and it persists across runs, so an ABBA order still charges the cold
  start to the first arm alone. Clear it before EACH arm, or discard a cold warm-up run and compare
  only warm ones, before comparing CUDA search walls.
- **A batched per-launch reading is not comparable to a synchronized round trip, on any constant**
  (gh-ocannl-994). `autotune_timing_modes` bracketed its `Queued` reading from below at
  `floor_ms / 16`, where `floor_ms` is a minimum over one-launch-plus-one-sync round trips. Those
  are two different physical quantities — the round trip includes exactly the host synchronization
  queueing exists to amortize away — so the fraction between them is the backend's sync cost over
  its launch cost, not a noise allowance. Measured over 534 runs of that instrument on five hosts,
  four backends and loads from idle to 6x oversubscription, it spans **0.0077 to 1.2**: on Linux
  `multidev_cc` the worker-domain round trip is 20-57 us against a 0.4-0.5 us amortized launch,
  while on `cc` the round trip *is* about one launch. That 160x spread is wider than the factor of
  `depth` such a check has to resolve, so the divisor had no admissible value — it refused 23% of
  the sweep's non-contended readings, all on Linux `multidev_cc` (30 of 30 on one box), and stayed
  green only because those readings were usually flagged `contended` and took the claim's escape
  hatch. A bound on a batched reading belongs on the SAME quantity, and specifically on the WINDOW
  the reading is a minimum over: the timed batches and their summed wall, which `time_routine`
  reports as its `Autotune.timing_call` record's `window` (`samples` — counted by the loop, not
  restated from its result, so a test can hold the two against each other — `reused`, the window's
  batches taken from the calibration and already counted there, `wall_ms` and `median_wall_ms`). Not the whole call — its wall also holds the warmup
  and the calibration's synchronized singles, and on a backend whose round trip is two orders of
  magnitude above an amortized launch those few dozen singles are ~40% of the call against tens of
  thousands of timed dispatches, so a whole-call mean is diluted by construction and a stall in the
  untimed part moves it without moving the reading. And within that window, the MEDIAN batch
  rather than the mean: `contended` is declared on a MAJORITY of the window's batches exceeding
  twice its floor, so the regime a claim must survive is exactly the one a median is unmoved over,
  while one arbitrarily long batch among 64 moves the mean without limit (measured on the same 342
  runs: the minimum sat at 0.19 of its window's median at worst, against 0.064 of its mean).
  Against either the ratio is at most 1 by construction (min ≤ the middle of the same samples),
  which makes the refusal of a twice-divided reading structural rather than calibrated: it cannot
  exceed `median / depth`, so a bound at `2 * median / depth` refuses it at every depth with
  nothing measured, and a per-batch reading overshoots a `2 * median` upper side by `depth / 2`.
  The trade to know: a statistic with a longer tail above the minimum refuses a division by the RUN
  count more often (the mean's tail reaches 15x, the median's 1.35x) and false-fails on precisely
  the stalls the bound exists to survive — the guaranteed refusal is the depth one, and a
  shallow-depth regime where an error is inside the envelope's own factor is a skip, not a pass —
  and that skip is gated on the DEPTH, never on a quantity derived from the reading under test,
  which decides whether to check using the very number in question (it passed a per-batch reading
  at depth 2 in review). The gate is DERIVED, not measured: `sample_min` declares `contended` when
  at least half a window's batches exceed twice its minimum, so in any window these claims judge
  the median is at most TWICE the minimum. A per-batch reading is that minimum, so an
  `f * median / depth` upper side refuses it once `depth > f * (median / minimum)` — `depth > 4` at
  `f = 2` — and the low side admits a correct reading from `depth >= 4` by the same bound. Each side
  therefore gates at its own threshold, 5 and 4 rather than one shared 5, and both are structural on every window not already bypassed. A fleet-measured gate
  stood here for one round and was 32, which would have left depths 5 to 31 unchecked; read the
  sweep's tail back through the invariant instead — a window measured below half its median was
  necessarily contended. `Isolated` keeps ONE side, against the round trip (`>= floor/3`): at depth
  1 a window-anchored upper side is a theorem, and a round-trip one compares two separately sampled
  windows, which a uniformly delayed window does not report as contended. The error it would have
  refused — a window summed instead of minimized — belongs on the injected clock, where it needs no
  device and no second window. The round trip is the side that gets contended in a suite: under
  `-j 8` sibling CUDA processes time-slice the GPU and hold a WHOLE 16-sample anchor window at 2-5
  ms against an idle 5.6 us, so all three anchors can read ms while the isolated window between
  them still has a fast majority (2026-10-22 rog-nv cuda sweep: 0.006 ms against 1.88 ms). A
  reading under `floor/3` therefore resamples the anchor first (bounded: 32 windows or 2 s) — a
  minimum over more samples only falls toward the true round trip, never below it, so no reading a
  clean anchor refuses is admitted. Such a failure is not evidence that the instrument skipped its
  sync: compare the isolated reading against a solo run's round trip before bisecting.
  The dispersion argument that rejected the mean for `Isolated` (a stall-cut 6-dispatch call read
  22x its own minimum, gh-ocannl-851) does not transfer to `Queued`: a stall lands inside one batch
  of `depth` dispatches among `samples` batches, so it moves the mean by a fraction of itself.
  Recipe for re-deriving any such constant: the instrument prints its raw readings to stderr as
  "(not part of the golden)" including both contention flags, so running the built exe in a loop
  against a spinner-generated load ladder (`_build/default/test/operations/`, `OCANNL_BACKEND=...`
  pinned) yields the distribution directly — and CPU-backend depths sit at the 200 cap, Metal's at
  31-80, HIP's at 200-750 and CUDA's at ~1700, so one box cannot stand in for the fleet.
## Cache regimes and placement replay
- The schedule-cache directory carries one key-regime stamp, independent of the serialized entry's
  `entry_version` (gh-ocannl-835). Bump `Schedule_cache.cache_regime_version` whenever
  `key_components` changes: the next STORE's cache-open deletes every `.sexp` entry under an
  older or absent stamp, then atomically publishes the current stamp; there is deliberately no
  migration arm per historical regime. A lookup never deletes (gh-ocannl-1439): it reads a
  superseded generation as misses (not refusals) and leaves entries, stamp and staging files in
  place, so reading another binary's cache cannot destroy it. Every lookup and store takes the same permanent-file `lockf` plus an
  in-process mutex through the entry I/O -- a store exclusive, a lookup shared -- so participating
  processes cannot write a current entry under a sweep, and a binary seeing a newer or malformed stamp refuses cache I/O without changing
  the directory. The lock file stays in place to avoid the unlink/recreate inode race and the OS
  releases its record lock on process death. Only a store creates the lock file (gh-ocannl-1492): a
  lookup opens it read-only and reads a missing one as unlocked (no writer exists that has not
  created it), so a read-only cache directory replays rather than refusing; a lookup creating the
  file read every directory it could not write as a refusal. A process killed before stamp publication leaves the
  old stamp and the next opener retries; power-loss durability is the filesystem's, not an fsync
  guarantee. Pre-gh-835 binaries do not take the lock and must not share a live cache directory
  during an upgrade.
- **An entry's filename is a digest of its key, never the key** (gh-ocannl-1228):
  `Schedule_cache.cache_file` names it by the key's 32-hex MD5, the entry stores the full key
  beside its payload, and a lookup whose stored key differs reads as a miss. A key-spelling name
  grew with every key component and outgrew Windows' 260-character MAX_PATH (rog's staging paths
  were 268-279, hosted CI's ~263), so every store was refused there. A test or tool that needs the
  keys present reads them with `Schedule_cache.entry_keys`, and derives a path with `cache_file`,
  never by parsing or spelling a filename; a shell script reads the key off the entry's first line,
  `((key <key>)` (`entry_key` in `benchmarks/gh612_cells.sh`), and prints that rather than the
  opaque `basename`.
- **A version bump appends its own history line.** Each version constant in `schedule_cache.ml`
  (`entry_version`, `placement_entry_version`, `cache_regime_version`) sits under a verbatim
  `(*= <name> history` comment, one `<value>: gh-ocannl-<N> -- <what changed>` line per value,
  and `test/operations/cache_version_history` (in `@scans`) refuses a constant its last line does
  not name. Two parallel bumps used to write the same `let entry_version = N` line and merge
  silently onto one number (the October 2026 wave did it twice); with the history line they
  conflict in git. Resolve by renumbering your line AND the constant to the next free value —
  keeping both lines fails the strictly-increasing rule.
- **A REQUIRED field in the saved form is the scoped generation bump** (gh-ocannl-1116). Optional
  (`[@sexp.option]`) fields keep old entries readable, which is right when the old meaning is the
  default; when old entries were timed under a rendering the fix removes, give the new field no
  default instead: exactly the entries carrying that op fail to decode, `lookup` swallows the
  failure as a miss, the routine re-tunes and the store overwrites the file, while entries without
  the op stay valid — no `entry_version` or `cache_regime_version` bump sweeping every winner.
  `Privatize.acc_prec` is the instance; `autotune_privatize` pins that the pre-fix spelling does not
  decode.
- **A fissioned winner persists its segmentation, and replay applies it** (gh-ocannl-1164). The
  entry's `segments` list every segment in order — kind, length in units (`Schedule.segmentation`:
  a unit is one top-level statement with its leading comments, so units are a function of the
  code), structural pre-schedule digest, and schedule, `Zeros` expansions and `Solo`'s empty one
  included — and `Autotune.compile_candidate` replays through `Schedule.fission_segmented
  ~replay`, which cuts there with no grouping, coalescing or `keep_mapping` probe. So no
  segmentation-policy input belongs in the key: the policy shapes which winner a search finds, not
  what a found one replays as — `Search_shaping`. **Match saved segments by POSITION, never by
  digest**: the structural digest leaves placements out, and two segments of one structure can
  differ in placements, hence in kind (`Solo` vs `Normal`) and schedule — a first-match lookup hands
  the later one a schedule the search never timed (`fission_replay_segmentation` leg 5; the same
  holds for `extend_spec`, whose units name their segment index, and for the per-segment sketch
  candidates, gh-ocannl-1243: an `F_sketch` replays the segmentation its seeds were enumerated
  under, its `replay` callback answering `None` for a segment left to its kind's preset, and keys
  its segments by first unit plus structural digest, `Autotune.seg_key`). Segments of one structure
  share a sketch key only when their placement-aware digests agree too, so repeated layers still
  share one sketch. A GPU's restored `Local` products are what split such twins
  (`autotune_sketch_placements`); on the C backends a `Local` product's zero opens its own segment
  (gh-ocannl-1343, above), so the twins differ in structure there instead.
  The replay-validity check rides
  on the position: segment `i`'s digest must equal the saved `i`th, and a recorded segmentation
  that does not partition the routine's units, or separates a scope-local definition from a use,
  is refused by `fission_segmented`. **A candidate transform's failure must be a typed cause**
  (`Schedule_outcome.Cause_at`, here `Illegal_schedule`), never a bare `invalid_arg`: under the
  default `strict_failure_classification` an untyped exception is FATAL, ending `tune` instead of
  declining the candidate (a cache replay into a re-search). Two compiles of one schedule in one
  process differ in the process-wide `i<n>`/`v<n>_` counters, so a same-process comparison of
  emitted code renumbers those stems first (the test's `alpha_normalize`).
- **A replay rebuilds what its recording compile ran: the head zero's fold is recorded, never
  inferred** (gh-ocannl-1480). `segments_of_plan` expands a `` `Normal `` segment's head
  whole-node zero when a fold is provable, since only `group_units`' fold puts a `Zeros` unit at a
  `Normal` segment's head. `fission_segmented`'s single-segment fallback (the routine collapsed to
  one kernel: one group, or everything coalesced) used to record `` `Normal `` whatever its head
  was, so a single kernel opening with an unfolded zero (every preset, which never folds; a sketch
  whose fold was declined) replayed as code the search never timed, and the replay's digest guard
  refused it: a searched winner shipped the untuned default, its cache entry re-searched on every
  warm call, and an `F_sketch` single segment silently fell to its preset. It now records
  `` `Unfolded `` there (`Schedule.segment_kind`; `segment_class` maps it to `Normal` for every
  consumer that classifies segments). No version bump: a `Normal` entry keeps its meaning, and the
  broken ones already refused and re-search once. It looked like "arm A's crown stops replaying
  once arm B has searched" only because `tune_placements` compiles what ships after every search
  (gh-ocannl-1408); the warm call's cache replay failed before any arm B ran. Rule: a new
  segmentation shape a replay must reconstruct gets its own recorded spelling -- if `fallback` or
  `group_units` can produce a segment the plan's `(kind, units)` reads back differently,
  `fission_replay_segmentation` section 7's derive-then-replay digest equality is the check to
  extend. A searched winner whose replay fails now warns on stderr unconditionally, and
  `Train.shipped.ship_substituted` says when no measured schedule shipped
  (`test/operations/tune_arm_a_replay`); the benchmark harness carries it into the result line's
  `tune.ship_substituted` / `substituted_from`, and `orchestrate.py` marks such a row
  `SHIPPED UNTUNED` (gh-ocannl-1559, `test/operations/bench_ship_substituted`).
- **What a replay rebuilds must not read process state that a link changes** (gh-ocannl-1480, the
  CUDA shape). A split-reduce prelude applies whole-routine before fission, and `Schedule.apply`'s
  trailing `simplify_llc` folds gather guards against each node's bounds candidate, which a
  `Host_inits` upload proposes at link. On CUDA and HIP the search's first candidate (the seeded
  plan) compiles before anything has linked, so its prelude kept a guard the winner's replay then
  folded: gpt2_mini_train's arm A crown (295 segments) refused at segment 1 and shipped the untuned
  default. The search's and the seeded plan's preludes now apply under
  `Low_level.without_node_bounds`; per-segment schedules still fold against the live bounds,
  since only the pre-schedule segments are digest-checked. A new whole-routine transform ahead of
  fission applies through `Autotune.apply_before_fission`, which owns the scope. Less folding is
  always sound.
- **The pre-fission audit (gh-ocannl-1560): only the split prelude read the bounds.** What runs
  between the base lowering and the digest check, and whether it reads a node's bounds candidate
  (`Low_level.interval_of_node`, reached through every `ienv` whose `node_bounds` is set:
  `simplify_llc`, `rewrite_one_hot_reductions`' `guard_conjuncts`): (1) the base lowering
  (`optimize`: inlining, the one-hot rewrite, `simplify_llc`) reads them, but `tune` captures it
  once per call and every candidate and the in-call winner replay rebase on it, so it cannot
  diverge within a call; across calls its canonical digest IS the cache key (`source_digest`),
  so a base lowered after a link folds into a different key -- a miss and a fresh search, never a
  mis-replay; (2) the split-reduce prelude (`F_split`, `F_split_saved`, `F_seeded`, and
  `seeded_split` deriving the plan) read them through `Schedule.apply`'s trailing simplify -- the
  gh-ocannl-1480 fix; (3) inside `Schedule.fission_segmented` before the cut, the Local
  promotions touch placements only (the replay digest is structural), `collect_units`,
  `group_units`, `resolve_scope_crossings` and `segments_of_plan` are structural, and a folded
  zero's `expand` (`probe_opt_op` of `Expand_zero`) builds a zero-writing loop nest and folds
  nothing; (4)
  `segment_optimized` filters metadata and simplifies nothing. After the cut the per-segment
  schedules fold against the live bounds, a Whole form's replay checks no digest, and a post-
  schedule digest only deduplicates, so a later fold there drops a run-time guard and changes no
  verdict. `autotune_strict_replay=true` checks this at run time: each fissioned candidate is
  rebuilt from its saved form after its own link, as a replay would, and a refusal is fatal
  (`test/operations/tune_strict_replay`, which also runs the CUDA/HIP prelude case end to end --
  its search must be the first thing in the process to touch the ids, since a device run before it
  uploads them and the BASE lowering then folds the guard, leaving nothing for a replay to fold).
- **What a cache entry says is input, so its refusals are typed** (gh-ocannl-1242). The rule above
  holds for every raise a replay can reach: `SC.of_saved`'s dangling symbol reference and
  `SC.tn_of_ref`'s out-of-range node index raise `Cause_at (Transform, Illegal_schedule
  { check = "Schedule_cache.of_saved" | "Schedule_cache.tn_of_ref" })`, so a stale or corrupt
  `W_saved` entry or split-reduce prelude declines into a re-search instead of ending `tune`. The
  report names the cause in `cache_replay_refusal` (outside `declines`, which counts candidates);
  `autotune_replay_refusal` plants both entries. The line between converting and keeping: a raise
  whose trigger is the entry (or the program) is a refusal; one that only a bug of the search's own
  builders can reach stays untyped and fatal — `SC.to_saved` (it serializes a schedule built
  against the very canonical form it resolves through), the `map2_exn`/`zip_exn` pairings of one
  fission's outputs, "the transform was not invoked". A replay failure that is NOT typed is still
  fatal under strict classification, on `Cache_replay` provenance too, so a new decodable-but-wrong
  field must be checked where it is read, typed.
- **`Train.tune_placements` persists its decision beside the schedule entries** (gh-ocannl-786,
  `Schedule_cache.store_placements` / `lookup_placements`, same directory, lock, regime stamp and
  key components). Placement stays outside the schedule value — a schedule is keyed by the
  placement-aware digest, so folding placements in would be circular — hence a second store. Its
  key is the DECISION PROBLEM's identity, `Schedule_cache.canonicalize_source`: the raw lowered code
  (`Low_level.optimized.source`, where a node the policy inlines away is still a statement, so
  every flip candidate has a structural position) plus, per node, the placement the lineage brings
  (`Context.placements` of the caller's context — prior decisions, or the intent the lookup falls
  back to) and its inline/footprint preferences; the decision itself is what the entry records, so
  it is outside the key. The entry holds `Default` / `Materialize_all` / `Refined flips` and an
  `outcome_digest` — the placement-aware digest of the lowering the decision produces. It is
  RECORDED from the shipped search's own `Autotune.report.source_digest` (gh-ocannl-1022: what was
  tuned, the key its schedule entry lives under) and RECOMPUTED at replay through
  `Train.decision_lowering_digest` (`Context.lowered_for_decisions` in the SEARCH lineage —
  `timing_ctx` when given, since every candidate and the shipped winner derive from that lineage's
  base lowering): an entry whose decision no longer reproduces its program (a cap moved, a lineage
  inherits differently, a flip names a node the problem lacks) is re-tuned and overwritten, never
  applied. That recording and recomputation agree is lowering determinism between a compile's
  `lowered_transform` and the analysis-only path; the test asserts it as an equality, so a
  divergence fails there instead of surfacing as a store that re-tunes every warm run. A hit runs ONE search from the
  replayed context (normally a schedule-cache replay), so `?report` sees one report there and the
  positional two-arm contract holds only on cold runs — attribute by `on_ship`, which the harness's
  `tune_json` now uses to name a lone report. Recorded only from clean evidence (every observed
  search completed, uncontended, the shipped one timed something), never under `tune_ship_arm`,
  which also never consults it. `test/operations/placement_store.ml` pins all of this; the
  directory is `Autotune.resolve_cache_dir`'s, so `autotune_search=false` with an unchosen
  directory disables both stores together. To bypass the placement store ALONE — a warm-cache
  placement A/B (both arms reporting, both schedules replaying), or a test sharing one warm
  directory across two-arm scenarios — set `tune_placement_store=false` (or pass
  `~placement_store:false`; gh-ocannl-1020) rather than deleting the `placements-` entries:
  `autotune_arm_containment`'s rule passes it on the command line.
- **A test claiming that a store or a replay happened waives on the cache's REPORTED refusal, not
  on a reference context** (gh-ocannl-1040/1043). Every store and lookup absorbs a filesystem
  refusal by design: the lock, the regime stamp, the write, or a Windows commit that outlives
  `Atomic_file`'s one-second retry (Defender holding the staging file is the suspect on CI). So a
  claim like "a clean cold run records its decision" can fail for a reason its timing predicates
  never see. Wrap the run in `Schedule_cache.recording_cache_io` and waive on a `refusal` of the
  operations the claim rests on (a record claim on its placement stores, a replay on its lookups),
  as an `` `Environment `` skip (`Verdict.gated`), never as a silent `true`. Then add one claim that
  is NEVER waived by a refusal: of the stores the process attempted under real key names, per
  cache directory, not all were refused. A deterministic refusal (a rejected filename, a commit that always fails) refuses
  every store, and without that claim it turns the test into a permanent skip that hosted CI
  aggregates nowhere. Gate on the timing identity of the run's OWN context, the one the store keys
  on, never a reference context's. All three are shared in `Test_utils.Cache_plant`: the waiver
  `unless_refused` (`refused_in` for a hand-written branch), the identity gate `unless_unkeyed`
  (`unless_unkeyed_or_refused` for both at once), and the unwaived claim
  `not_every_store_refused`; `placement_store`'s `persistence` record is the exemplar. A test planting an entry by hand builds it through
  `Cache_plant.entry` (synthetic) or `Cache_plant.plant` (keyed and self-described as
  `Autotune.tune` stores one, returning the store's I/O for the waiver), never as a record literal,
  so a new entry field reaches every planted entry through one default. `schedule_cache_regime`
  pins the record itself: committed and admitted I/O, the regime, lock, read and commit refusals,
  keyless calls, and nesting.
- **A claim that rests on a recording retries the recording, then skips with a reason** (gh-ocannl-1086).
  A recording run whose evidence is unclean records nothing by design, and every "the replay
  ships the recorded X" claim written as `(not stored) || X` then passes vacuously — about half
  the runs on a loaded box, with no trace. `placement_store` re-runs each recording (run 1, run 6,
  the cross-process recorder) in a FRESH directory, up to three attempts (the same directory would
  replay the first attempt's clean searches), and gates the dependent claims through
  ``Verdict.gated ~aggregation:`Environment`` on why no entry exists, so the sweep counts the skip.
  The test's "clean" predicate is `Train.tune_placements`' own recording rule — uncontended, no
  terminal failure, a finite best — under which an abandoned flip is a measured loser, not spoiled
  evidence; a stricter test-side copy (requiring every report `Searched`/`Cache_replay`) disagrees
  with the recorder exactly when a flip is abandoned.
- **A timing lineage's schedule compiles into the caller only where the caller lowers the same
  program** (gh-ocannl-1421, found as gh-ocannl-1386's flake). `compile_candidate`'s `rebase`
  applies the base lowering under the compiling lineage's placements, which is drift-immune within
  one lineage but wrong across two: with `?timing_ctx` materializing a node the caller leaves
  virtual, a whole-routine winner emitted code reading an undeclared buffer (a classified decline
  on cc, an unclassified `Failure` on Metal) and a fissioned one raised
  `promote_local_to_device: … is virtual`, fatal under strict classification. Before the cache and
  the search, `tune` lowers the routine analysis-only (`Context.lowered_for_decisions`) in BOTH the
  caller's and the timing context and compares their `Autotune.lineage_identity`: the code with
  loop binders canonically numbered and every tensor node anchored to its actual id
  (`Schedule_cache.canonicalize_code` with an id `node_tag`). The canonical form alone
  alpha-renames nodes (`u := x` against `u := y`, `x - y` against `y - x`), and an access set
  misses `x + y` against `x - y` (review rounds 5-7). A mismatch reports `Lineage_refused`, searches nothing, warns on
  stderr unconditionally and ships the untuned default at `best_ms = infinity`; a lowering that
  fails is a pre-search failure. Together, because lowering is stateful in compile order: the base
  compile settles an index tensor's value bounds, so a lowering taken after it folds a gather guard
  the base kept -- comparing the base itself with the caller's lowering refused every arm A of
  `bench_gpt`'s `gpt2_mini_train` (cc, CUDA, HIP). Up front, not at the winner compile: a refusal
  there left the arm ranked, and its decision recorded, by the crown it never shipped.
  `Train.placement_evidence_clean` lets such an arm stand beside a recording;
  `Train.placement_problem` tags each node with both lineages' `Autotune.lineage_accesses`
  (canonically numbered, and drift-stable, since the key is taken before any compile settles
  bounds) -- coarser than the refusal, so two lineage pairs differing only in inherited arithmetic
  share a key (a perf-only residual). No compile-time backstop: it would compare a post-timing
  lowering, where drift is the rule. It is conservative: a timing
  lineage that inlines a computation the caller holds materialized (`flip_twins`' inheriting
  lineage) runs fine but is refused too. `placement_store` run 6 ranks fissioned candidates first
  through `Autotune.on_candidate_measured`, which reproduced the old failure deterministically.
- **A budget-cut crown is stored, but only a budgeted call replays it** (gh-ocannl-1409). With
  `autotune_budget_seconds`, `Autotune.tune` checks the budget as each attempt and each beam round
  starts (`budget_allows`), skips everything later except the untuned floors (the seeded plan and
  the default seed), and crowns its incumbent -- unless no floor was measured (no candidate
  reproduces the default under `schedule_fission=false`), when it ships the untuned default and
  stores nothing. The crown is stored with `Schedule_cache.entry`'s
  `budget_cut`, and a call without a budget declines it and searches in full, the way a contended
  search is kept out of the cache entirely (`search_measurements_cacheable`): an incomplete
  measurement set must not answer every later call. The placement store refuses cut evidence for
  the same reason (`Train.tune_placements`' `persist`). The budget is `Search_shaping`, so a cut
  `best_steps` serves early stopping only under the same budget value. A test places a cut with the
  `Autotune.budget_elapsed` seam's virtual clock, never with a wall clock
  (`test/operations/autotune_budget.ml`). **The budget bounds wall time, not only the last start**
  (gh-ocannl-1556): checked only as attempts started, a 120 s budget ran 452 s on tuf, the slow
  candidate in flight finishing its whole compile and timing. Now an attempt also does not start
  when the elapsed time plus the costliest candidate compile so far (`costliest_compile_s`, on the
  same clock; the base compile is not evidence, so the first candidate compile is unprojected)
  would pass it, and a non-floor candidate's timing is abandoned before its first batch past it
  (`time_routine ?stop`, raising `Timing_stopped`, caught inside the candidate's failure boundary
  as an unranked `Ok None`, not a decline). The stop is strict (`elapsed > b`) and the projection
  admits an exact fit, so a virtual clock that charges a whole attempt as it starts (the scripted
  harness's attempt clock) cuts exactly where it did. `test/operations/autotune_budget_overrun.ml`
  charges compiles through the compile observer and timing batches as budget reads after
  `on_candidate_preflight`; the stated tolerance is in `ocannl_config.reference`. On a small cc chain the default seed dedups against the
  baseline, so check the floor's TIME (`default_ms`), not its label. Every report carries the
  budget its call resolved (`budget_s`, gh-ocannl-1438) -- the call's, where a replay's
  `budget_cut` is the crown's -- and benchmark arms carry it as `budget_s`. It is deliberately not
  a `regime_knobs` key: those are numerics settings the orchestrator's regime gate holds to the
  cell's regime, and a budgeted exact cell would read as a leak.
- **A `tune_placements` call budget is split, not shared first-come** (gh-ocannl-1440).
  `tune_call_budget_seconds` runs one clock for the call; each search gets what is left over the
  searches still planned (`Autotune.call_budget`, reported as `call_budget_s`/`call_share_s` beside
  the per-search `budget_s`), and is cut at the tighter of its share and its own budget by the same
  `budget_allows`. A first-come deadline let arm A spend everything and compared arm B on its floors
  alone. Both arms always search (the positional `?report` contract, and they are the call's
  floor); the flip chain stops before any alternative once the budget is spent, mid-group included
  (an unmeasured alternative leaves its node open, like a pruned one), and a decision reached with
  flips left unsearched is not recorded in the placement store, like one over a cut search. The
  key is
  `Placement_shaping`: a share follows the wall clock, so it stays out of the trajectory identity,
  and a cut trajectory is a prefix of the same order, which early stopping reads conservatively.
  `test/operations/tune_call_budget.ml` places cuts with two virtual clocks counted in attempts
  (`Autotune.call_budget_elapsed` for the call, `budget_elapsed` per search, a search's first
  attempt being its `baseline`); per-search, no non-floor attempt may start at an index at or past
  the cut.
- **A deferred cache hit ships its validated code relinked** (gh-ocannl-1440).
  `Context.compile_outcome ?retain` keeps a successful compile's code, censuses and pre-link
  context (`Context.retained`, no working pools: those are a link's), and
  `Context.relink_outcome` links it again on that context with a fresh delta and routine id. The
  deferred hit releases the validating replay's context and relinks at `compile_deferred`, so the
  cached schedule compiles once; a relink the device refuses ships the untuned default, as the
  second compile's failure did. Metal compiles its library at LINK, so a relink alone would compile
  the source again: the library is memoized on the code value (`Metal_backend`'s `library_memo`),
  and the link-time compile is counted with the others. The counters: `Context.backend_compiles`
  (per domain, `Utils.count_backend_compile`: every code generation plus every link-time device
  compile), `report.compiles`/`replay_compiles`, `Autotune.deferred_compiles`, and
  `Train.shipped.ship_compiles`, which together account for every compile of a call
  (`test/operations/deferred_hit_compiles.ml`). A searched winner still recompiles from its saved
  form: its candidate was compiled in the search lineage, which a `timing_ctx` makes another
  context. A backend-level code cache (the HIP code-object cache of gh-ocannl-1314) composes:
  the relink is the backend's own `link`.
## Scan-loop scheduling

- **A `Scan_loop` is opaque to the schedule ops in both directions and transparent to the
  annotator** (gh-ocannl-696, `test/operations/scan_loop.ml` leg 7): `find_loops_env` and
  `rewrite_loop` do not enter it, so an op naming the scan's own index or a loop nested in its body
  declines with the standard "no For_loop with index" refusal — the gh-ocannl-668 law holds because
  neither walk descends. `Pad` guards the scan whole (a guard inside its body would make the carried
  update conditional); `Stage`, `Privatize` and `Split_reduce` refuse a scan in their region by name.
  The default annotator DOES descend, registering the body's accesses like a serial loop's, so an
  enclosing loop the accesses prove independent keeps its `Grid` mapping — the carried state is
  per-iteration scratch of that loop (leg 4, Serial vs Grid parity on cc and Metal). For footprint
  queries the scan index is an ordinary loop symbol: `loop_ranges`, interval analysis, and
  `affine_accesses`, where the inits sit at path `Stmt 0` and the body at `Stmt 1` so program order
  is preserved.
- **A scan nest in a GPU sketch's kernel is a companion like any other** (gh-ocannl-1134):
  `companion_geometry` walks into `Scan_loop` bodies for its coverage demand, as
  `validate_parallel` does, so a scan's materialized row writes need covering. Where the scan's
  enclosing loops align with the site's full chain, they take the site's per-position geometry and
  the scan runs serially per thread. The two-pass attention's scan runs over the score site's column
  axis, so its chain trims below the site's arity and the family is refuted naming the scan's
  nodes. Fission makes the cut there: the default schedule-aware `keep_mapping` probe already keeps
  the scores' kernel apart, and `arity_cuts` does so under the legality rules alone. The walk used
  to stop at the scan: the witness passed and every seed died at compile, which is how gh-1003's
  stage-1 tuned arm lost its tensorized scores. The sketch's demand walks (`materialized_writes`,
  `nest_loop_syms`, `epilogue_tail_loop_syms`) are now `Low_level.write_demand`, which enters scans
  (gh-ocannl-1451): ask it for a kernel's writes and loop binders rather than writing another walk;
  `nest_loop_syms` keeps the `For_loop` binders beneath a scan but not the scan's own (no schedule
  op targets it). `test/operations/scan_companion_coverage` executes every seed of both shapes.
## Timing identity

- Timed cache evidence uses `Context.timing_identity` separately from conservative construction
  limits (gh-ocannl-594): schedule AND placement keys include concrete device capabilities. CUDA/HIP
  key model, architecture, compute resources and static memory/clock properties, not device ordinal
  or backend-wide minima. CPU retains model/host/compiler metadata; macOS CPU/Metal additionally
  query hardware UUID and OS build, and Metal retains device registry identity. Toolchain metadata
  is optional and explicitly partial: CUDA driver, selected OpenMP libraries and HIP/rocWMMA headers
  remain preexisting provenance gaps (gh-ocannl-1026), not reasons to disable supported caches. Outer `None` means
  concrete device discovery failed and bypasses all shared cache I/O. Execution/tuning remain
  available. `schedule_cache_device` pins device/metadata separation, bypass, and real-backend replay.
## Capabilities and search diagnostics
- **HIP's tensor-core capability is gated on the HOST, not only on the device** (gh-ocannl-1032):
  `mma_supported` is `all_rdna_wave32 && rocwmma_include_dir`, and both `hardware_limits.mma` and
  `mma_syntax` consult it, so a gfx11/gfx12 wave32 box whose filesystem has no rocWMMA headers
  advertises no MMA, seeds no tensorized candidate and renders the lane-0 scalar fallback — and is
  right to. The WSL ROCm SDK bundled rocWMMA and native Ubuntu 26.04 does not, which is how
  `schedule_mma_matmul` came to print 11 bare `false`s on a native AMD box with nothing naming the
  cause; the native stack's codegen was never the difference, and with a full header tree on
  `ROCWMMA_PATH` that same ROCm 7.1.0 / gfx1151 box passes the suite unchanged. Two rules follow. A
  test arm expecting a rocWMMA emission DERIVES it from the advertised capability and asserts the
  fallback rendering otherwise, as the tf32 gate and the CUDA `Fp16_wide` arm already do; only a
  claim whose subject is the fragment scope itself skips, and as an ORDINARY backend skip rather
  than `` `Environment ``, however host-shaped the reason looks on this fleet. That is the second
  rule, and it is about the conjunction: `hardware_limits` reports only the AND, so from a test
  neither half is visible, and on a CDNA gfx9 wave64 part the leg is withdrawn by the DEVICE --
  coverage that hardware can never have, not a host condition a sweep should aggregate away.
  Separating them would mean restating the device predicate the backend owns, to buy a label;
  `Backend` is the conservative reading (Verdict already lets an `` `Environment `` mark elsewhere
  carry same-key `Backend` skips), and the host half belongs in the human diagnostic, which for the
  same reason names BOTH halves and qualifies the rocWMMA remedy by device eligibility rather than
  prescribing it to a gfx9 box no header tree can help. And the probe requires
  `rocwmma/internal/types.hpp` beside `rocwmma/rocwmma.hpp`: Ubuntu's librocwmma-dev 7.1.0 installs
  the umbrella headers without `rocwmma/internal/`, so a one-file probe accepts a tree on which
  every tensorized kernel then fails inside hiprtc — the capability decline exists precisely to
  keep that unreachable. The header half searches `hip_sdk_include_dir`'s tree too, which probes
  `HIP_PATH`, then `/opt/rocm`, then the distro `/usr` (gh-ocannl-1070): before `/usr` joined, a
  native box reached its headers only through the login shell's `HIP_PATH=/usr`, so a hermetic
  environment (machine-verify, a bare ssh command) silently lost the capability. Each HIP device's
  `static_properties` entry carries `tile_mma_eligible`, the device half read through the gate's
  own predicate, so a readback (`bin/device_props`) can tell the two halves apart without
  restating it. The header half is in `Hip_backend.host_capabilities` (`rocwmma_include`), which
  the test configuration's host record carries, so a tree appearing or vanishing reruns the tests
  whose verdicts read it (gh-ocannl-1038; build-and-test.md#dune-mechanics).
- A search's COST is read from `autotune_progress` lines, not `autotune_log` (gh-ocannl-1061):
  the latter is per-candidate and times an extra untuned-default control, so it moves the cost it
  would record; the former adds a clock read per candidate, is flushed line by line (a search
  killed at a benchmark cap keeps its record), and splits a search's `elapsed_s` into candidate
  `compile_s` and `timing_s`. First reading, gpt2_mini cc on mac-studio: 79 s before the first
  seed (base compile, analyses, the ~2 s baseline's timing window), then timing windows dominate
  compiles about 9:1 on second-scale candidates — so on cc a candidate's cost is its sample count
  times its step, not its compile. The format is the interface's (`Autotune.progressf`). A test
  turns either stream on for one call with `~log`/`~progress` on `Autotune.tune` or
  `Train.tune_placements` (domain-local, inherited by nested searches, restored on return) and
  reads it from captured stderr; `autotune_composite_playoff` claims both.
- **Seeding refusals are invisible to the decline census; `autotune_log_seeding=true` shows them**
  (gh-ocannl-1407). A refused seed never becomes a candidate, so neither the search report's
  `declines` nor `schedule_log_declines` (render and compile declines) can hold it. The census
  (`autotune-seeding:` lines on stderr, `Autotune.seeding_census` for a test) is printed for every
  lowering `tune` and `model_default` seed — the whole routine and each coarse and fine fission
  segment — and once per segment of the plan the CUDA/HIP seeded default chooses. Each line names the
  family that claimed the lowering, its site and seed count, then its refusals: the matmul tree's
  refutations grouped by witness, or each conv leg's seed count or EVERY failing gate (companion
  coverage, no mma tile, the vector file, launch caps, workgroup memory), and the
  lowering's conv nests whatever family claimed it. The conv gates record their own reasons
  (`Sketch_families.conv_seed_judgment`), so the census cannot drift from the seeder. Use it
  before citing a seeding gate as evidence: the decline census can never show a seeding refusal,
  and on real training graphs the binding conv gate is usually the matmul family claiming the
  whole routine first, not the gate an issue names.
- A test claims a tensorized seed's PRESENCE wherever the backend advertises the capability, and
  gates on that capability, never on the seed list its claims are about (gh-ocannl-1115). Gated on
  the seeds, a seeding regression — or a claim asking for a seed the seeder excludes by
  construction, like the bf16 pipelined-staged shape below the 4-byte async floor — skips on every
  backend, and skip coverage notices only once every backend skips. The capability is the seeder's
  own judgment, `Autotune.tensorized_capability_refutation`: the family tree refutes its tensorized
  branch through it (format tile, lane width and routine logging on GPU; vector file, lanes,
  precision uniformity and routine logging on CPU), and `Ll_test.tensorized_capability` gates on it,
  deriving a withheld capability's skip aggregation (`Environment` when `tf32_matmuls` on or routine
  logging off would lift it) instead of keying on a backend name. Review found the piecemeal
  re-derivation's gaps one condition per round (logging on GPU, then CPU, then the vector width),
  which is why the predicate now has one owner. A gated test declares the configuration the judgment
  reads (`OCANNL_TF32_MATMULS`, `OCANNL_PROFILE`, the two logging keys, `OCANNL_CC_VECTOR_BYTES`).
  For an f32 site that leaves Metal as the only GPU evaluating those claims at default config (CUDA
  needs tf32, HIP has no f32 shape), and CI's macOS job runs cc, not Metal.

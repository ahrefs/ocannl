# Scheduling and autotune

Schedule legality and coverage, sketch families, and how to read what a search actually did.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

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
  segmentation stops being the untuned default's (`fission_equivalence`). The segmentation's inputs
  beyond the code -- the gate, `gpu_schedule_block_size`, `_min_parallel`, `_workgroup_fill` -- are
  the schedule cache's `fission` key component: a fissioned winner replays by re-segmenting. A cut needs no retest after
  scope-local resolution: a cut that resolution merges back is serial either way (the reason is in
  the comment on `keeps_mapping`). `test/operations/gpu_fission_mapping`.
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
  keep its own term, a wrong value rather than a race. Multi-simdgroup widths decline in v1.
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
  segmentation mints new digests, and a fine winner records `finer_fission` in its cache entry so
  replay re-segments identically — never the default pipeline, which would pay the extra launches
  unconditionally; the default cuts only where a merge costs a nest its mapping (gh-ocannl-1126, above). Since gh-ocannl-577 the
  coverage verdict is also a construction-time refutation in the matmul family tree
  (`matmul_coverage_witness`): this is sound because `companion_geometry`'s Ok/Error never depends
  on the geometry its `annotate` callback emits — only on the lowering, the site chain, the fused
  flavor's `skip` and the zeroing expansion. If you ever make the verdict consult the emitted
  geometry, the static witness goes stale and the tree will refute families whose candidates
  would build (or vice versa) — keep the invariant, or re-derive the witness. The fused
  (`Fuse_epilogue`) flavor is judged separately: skipping the epilogue tail can empty the
  coverage demand before the alignment analysis is consulted, so twins can survive a routine the
  unfused family is refuted on. Since gh-ocannl-613 the flavor is the family tree's ROOT level
  (`fusion = unfused | fused`, `matmul_family_tree`): the fused child is refuted with the
  recognizer's own reason (`Schedule.fuse_epilogue_witness`) on a site with no fusable tail —
  so a tree's `refutations` always carries one entry more than its pipelines produce, and a
  test asserting "every refutation is X" must scope to the `Family_decision.Fusion `Unfused`` path — and
  the unfused coverage verdict is shared with the fused branch because it implies it (the fused
  demand is a subset, and `aligned_chains` ignores `skip`): `aligned_chains` runs twice only
  where the unfused flavor is refuted. `sketch_seed_params` is the tree's `leaves`, twins
  included, and `model_default` has no separate twins step. **Consequence for diagnostics: since gh-577 a companion-coverage
  DECLINE CENSUS is empty, and that is not evidence the rule stopped firing** — a refuted family is
  never seeded, so it never reaches the decline log. gh-569's Part 3 read 25 coverage declines out of
  `schedule_log_declines`; the same workload on current master logs zero
  (`report-gh612-hip.md`). Ask the emitted source and the launch geometry instead.
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
  like this one bumps `Schedule_cache.entry_version`: a stored crown is only the best of the menu
  that searched it.
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
  the tensorization lane as the sole `Workgroup` loop. `conv_seed_params` filters that lower-bound
  prediction through the same predicate. `launch_predicate_parity` derives a real conv site behind
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
- `detect_conv`'s boundary is SINGLETON axes, not rank (gh-ocannl-912). 3-D, 1-D, batchless and
  multi-batch convs are detected and seeded on both legs; what it refuses is any extent-1 axis,
  because lowering drops the loop and indexes the axis at `Fixed_idx 0` while the matcher wants
  plain iterators on the output and kernel: batch 1, output extent 1, one input channel (lenet's
  conv1), a k-by-1 window. A 1x1 window is a GEMM and goes to the matmul family. A conv fixture
  that "is not detected" usually has a singleton somewhere — keep every axis at extent >= 2 unless
  the singleton is the point. `test/operations/conv_detection_boundary` pins each class with its
  reason, derived from the lowered maps.
- **The matmul family never tiles or packs along a strided axis** (gh-ocannl-1076): a role needs
  unit coefficient in the operand (`Sketch_families.unit_axis`), so a stride-2 1x1 conv
  (`resnet_block`'s downsample) classifies with the BATCH axis as the GEMM row and `oh, ow` as
  interior batch loops (`m_bi`) — every seed stages dense tiles from a strided source address and
  executes correctly on cc and HIP, but the row is only `b` rows tall, and at batch 1 no family
  seeds the site at all. Inside the full block (at 64 channels; at 16 the shortcut's conv is
  virtualized into its norm and is no site) cc does not fission it and every whole-routine seed
  declines — the operand `Stage` meets the 3x3 conv's second read of `x` — so the tuner times no
  sketch for the shortcut there; GPU fissions it and the segment's seeds run. `resnet_block` itself
  does not compile (no out-channel knob). `test/operations/schedule_strided_1x1` PINS the role
  structure, the batch-1 refusal and every seed's parity (or typed decline, in the block); which
  seeds decline where it only reports on stderr, and the 16-channel virtualization is a manual
  observation (conv2d into batch_norm2d, 16 channels) that no test exercises.
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
- "Seeded" is not "timed". An autotune family can be enumerated in bulk and rejected in bulk at
  candidate compile, and a count of proposals then reads as coverage it does not have — assert on
  the *timed* counter (`report.mma_timed`, `fiss_sketch_timed`, `split_reduce_timed`), and follow it
  with an executed value check, since a candidate that compiles is not yet one that computes.
- The `N segs` in an autotune label such as `F_saved[fine 77 segs]` counts the SAVED PER-SEGMENT
  PLACEMENT ENTRIES, not kernels: the arm that reports `fine 77 segs` emitted 136 `__global__`s
  (gpt2_mini on HIP, `report-gh612-hip.md`), and `[58 segs]` emitted 117. Take kernel counts from
  the launch log (`schedule_log_launches`, whose `seg i/N` names the real fission width) or from the
  emitted source; a report that quotes the label as a kernel count is wrong by ~1.8x. The launch
  log's FIRST fissioned `seg 0/N` (skipping the `N=1` whole-routine probe) is arm A, the next is
  arm B — which is also how to pick the right file out of a content-polling snapshot of
  `<routine>__seg.hip`. The watcher can catch a partially written file, and the kernel count alone
  does NOT identify a usable capture: a torn file can already carry every `__global__` line while its
  last body is incomplete, and glob order is hash order. Require balanced braces AND a clean `hipcc`
  compile before accepting a snapshot (`benchmarks/gh612_cells.sh pick_armA`).
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
  Where a timing is REPORTED the label is now printed, so a mismatch is legible without re-deriving
  anything: the `autotune_log` NOTE lines lead with it, `Train.tune_placements`' arm lines read
  `[tensorized/<label>]`, `bin/schedule_bench` and `bin/narrow_gebp_bench` print
  `C_syntax.mma_summary_string` on EVERY timing line (not only when something declined), the
  benchmark harness prints it per segment in the per-kernel table, and the result line's `tune`
  arms carry `tensorization` + `mma_statements`, which `orchestrate.py` renders in the report's
  `mma` column (`SCALAR FALLBACK` / `NO MMA EMITTED` shouted) plus a `TENSORIZATION NOTICE`. What
  that column reads is `tune.shipped_mma`, the census of the routine that was TIMED, not the arm
  named as shipped — a crowned arm candidate is not always the shipped artifact: a gh-555 flip
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
  cap when its column tail is at most one vector), only where at least one exists. The default is
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
  Not done: the conv family (not tree-factored); whether a partial vector's masked copies deserve
  a term of their own.
- **C-tile traffic cannot rank `rn` at a fixed lane width** (gh-ocannl-1099).
  Full passes plus the narrower column tail move `2 * rm * ceil(n/lanes)` C vectors per row
  band, independent of `rn`; charging `2 * rm * rn / k` at full `rn` on the tail pass, or once
  per site, miscounts it. The exclusive AVX2 A/B confirmed a 4x2 serial win at n=28, but 4x3
  won at fixed n=512 with k=32 as well as k=256. A tail-free tie preference avoids a second
  tile body; when both candidates have tails, both emit two bodies and equal cost means equal
  A splats, so the remaining tie keys have no additional issue-slot rationale. Keep them
  unchanged here; [gh-ocannl-1180](https://github.com/ahrefs/ocannl/issues/1180) owns validating
  a tail-free-then-smaller-`rn` rule on NEON and AVX2. gh-ocannl-947's NEON n=56 tail-bearing tie was
  neutral; more targeted coverage is needed. Derivation, paired measurements and reproduction
  protocol: [gh-ocannl-1099](../research/gh-1099-register-tile-c-traffic.md).
- "Crowned" is not "shipped", and neither is reproducible on a small routine. `Train.tune_placements`
  runs two searches and keeps one artifact, so a family can win the arm that is then discarded whole
  — read `report.best_label` / `best_tensorized` / `best_tensorization` / `mma_best_ms` per arm (the A/B calls `?report`
  for arm A first and ships the smaller `best_ms`), never the fact that some search crowned it.
  Since gh-ocannl-638, do not re-derive WHICH arm shipped from the reports' times either: config
  `tune_ship_arm=a|b` overrides the comparison, and `?on_ship` (`"A"` / `"B"` / `"flip"`) is the
  callback that says what actually shipped. That knob is what a measurement needs: the discarded
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
  (`test/operations/autotune_arm_containment`).
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
- The base compile staying unscheduled is a settled decision, not an oversight (gh-ocannl-552): the
  default pipeline is `maybe_default_schedules` — fission then per-segment annotation, so several
  kernels in general, not one `optimized` to rebase candidates on; every candidate family assumes
  the serial zero point; and annotation reads `hardware_limits`, which would bake per-device
  decisions into the cache's `source_digest`. The "did tuning beat the shipped default?" reference
  is `report.default_ms` instead — the config-thresholds fissioned seed reproduces the untuned
  pipeline exactly and its time is attributed by digest (so a seed that dedups against a timed twin,
  the CPU serial baseline included, still reports).
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
  key's value (`Utils.config_class_fingerprint`, `SC.trajectory`), so a replayed incumbent still
  has one; a failed one abandons nothing. A clean abandoned prefix is also persisted
  (gh-ocannl-1136), under the unchanged schedule key with an `abandonment-` filename prefix.
  A replay requires the same search shape and re-evaluates the current incumbent and ratio
  against those timings; without a qualifying rule it searches normally. It reports
  `Abandonment_replay` and raises `Search_abandoned` with zero search counters, so a warm
  flip chain does not re-search its losing flips or mislabel the harness's tuned row.
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
  Two things the generalization does NOT do: it cannot coalesce the nest into one loop (the
  per-axis index maps cannot express `f / M, f mod M`), so a tile's k-extent is judged against the
  innermost contraction extent alone and `bk` values above it are refuted by the ordinary
  divisibility gates; and the whole-`m_k` forms (the unstaged `bk = 0` tensorize, the CPU
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
  Two pieces of residue. Nothing prunes a wasteful pad: a 20-row site now seeds a 64-row block
  tile whose padded work is 3x the real work, and only the tuner's timing rejects it (the
  tensorized family has always behaved this way). And the CPU blocktile and packed-scalar
  pipelines stage into stack scratch, so the same argument would let them pad, but they keep the
  gate — which is what still renders the gh-ocannl-683 k-extent label, and where
  `schedule_contraction_nest` reads it off.
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
  `report.timings_unbatched` (a subset of `timings_contended`, so every cache and completeness
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
  depth, not wall, spending no probe; ultra-fast kernels whose fit wanted the cap now batch shorter
  than the target. Do not replace
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
  that fails without it. Its probe claims read `Autotune.on_calibration_probe` (gh-ocannl-1119),
  one record per probe tagged with the branch that started it, never the `batch` calls: runs of
  same-depth batches merge a stall retry into its confirmation, so the retry escaped the budget
  claim, and chunking them counts probes only as a lower bound. A new probe site passes its own
  `~role`; the test's witness claim holds every report against the device's batch log.
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
  CUDA/HIP cache keys spell this changed queued policy as `queued-v2` while entries and user-facing
  configuration still say `queued`; otherwise a depth-200 winner already on disk would replay
  without running any of this calibration. cc/Metal keep the unversioned `queued` key because their
  policy did not change.
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
  `on_timing_retry` accounts for the discarded window's extra dispatches, while
  `on_timed_window` describes only the returned window; `autotune_measured_refusal` labels the
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
  Symptom to recognize: the `on_batch_depth` seam (printed by `autotune_timing_modes` to stderr)
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
  session's wall this way; `benchmarks/gh834_cells.sh` is the per-box driver. A cold schedule cache
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
  reports through `Autotune.on_timed_window` (payload: `~samples` — counted by the loop, not
  restated from its result, so a test can hold the two against each other — `~reused`, the window's
  batches taken from the calibration and already counted there, `~wall_ms` and `~median_wall_ms`). Not the whole call — its wall also holds the warmup
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
  device and no second window.
  The dispersion argument that rejected the mean for `Isolated` (a stall-cut 6-dispatch call read
  22x its own minimum, gh-ocannl-851) does not transfer to `Queued`: a stall lands inside one batch
  of `depth` dispatches among `samples` batches, so it moves the mean by a fraction of itself.
  Recipe for re-deriving any such constant: the instrument prints its raw readings to stderr as
  "(not part of the golden)" including both contention flags, so running the built exe in a loop
  against a spinner-generated load ladder (`_build/default/test/operations/`, `OCANNL_BACKEND=...`
  pinned) yields the distribution directly — and CPU-backend depths sit at the 200 cap, Metal's at
  31-80, HIP's at 200-750 and CUDA's at ~1700, so one box cannot stand in for the fleet.
- The schedule-cache directory carries one key-regime stamp, independent of the serialized entry's
  `entry_version` (gh-ocannl-835). Bump `Schedule_cache.cache_regime_version` whenever
  `key_components` changes: the next cache-open deletes every `.sexp` entry under an older or
  absent stamp, then atomically publishes the current stamp; there is deliberately no migration
  arm per historical regime. Every lookup and store takes the same permanent-file `lockf` plus an
  in-process mutex through the entry I/O, so participating processes cannot write a current entry
  under a sweep, and a binary seeing a newer or malformed stamp refuses cache I/O without changing
  the directory. The lock file stays in place to avoid the unlink/recreate inode race and the OS
  releases its record lock on process death. A process killed before stamp publication leaves the
  old stamp and the next opener retries; power-loss durability is the filesystem's, not an fsync
  guarantee. Pre-gh-835 binaries do not take the lock and must not share a live cache directory
  during an upgrade.
- **A REQUIRED field in the saved form is the scoped generation bump** (gh-ocannl-1116). Optional
  (`[@sexp.option]`) fields keep old entries readable, which is right when the old meaning is the
  default; when old entries were timed under a rendering the fix removes, give the new field no
  default instead: exactly the entries carrying that op fail to decode, `lookup` swallows the
  failure as a miss, the routine re-tunes and the store overwrites the file, while entries without
  the op stay valid — no `entry_version` or `cache_regime_version` bump sweeping every winner.
  `Privatize.acc_prec` is the instance; `autotune_privatize` pins that the pre-fix spelling does not
  decode.
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
  `~placement_store:false`; gh-ocannl-1020) rather than deleting `placements-*.sexp` entries:
  `autotune_arm_containment`'s rule passes it on the command line.
- **A `Scan_loop` is opaque to the schedule ops in both directions and transparent to the
  annotator** (gh-ocannl-696, `test/operations/scan_loop.ml` leg 7): `find_loops_env` and
  `rewrite_loop` do not enter it, so an op naming the scan's own index or a loop nested in its body
  declines with the standard "no For_loop with index" refusal — the gh-ocannl-668 law holds because
  neither walk descends. `Pad` guards the scan whole (a guard inside its body would make the carried
  update conditional); `Stage`, `Privatize` and `Split_reduce` refuse a scan in their region by name.
  The default annotator DOES descend, registering the body's accesses like a serial loop's, so an
  enclosing loop the accesses prove independent keeps its `Grid` mapping — the carried state is
  per-iteration scratch of that loop (leg 4, Serial vs Grid parity on cc and Metal). For footprint
  queries the scan index is an ordinary loop symbol: `loop_bounds`, interval analysis, and
  `affine_accesses`, where the inits sit at path `Stmt 0` and the body at `Stmt 1` so program order
  is preserved.

- Timed cache evidence uses `Context.timing_identity` separately from conservative construction
  limits (gh-ocannl-594): schedule AND placement keys include concrete device capabilities. CUDA/HIP
  key model, architecture, compute resources and static memory/clock properties, not device ordinal
  or backend-wide minima. CPU retains model/host/compiler metadata; macOS CPU/Metal additionally
  query hardware UUID and OS build, and Metal retains device registry identity. Toolchain metadata
  is optional and explicitly partial: CUDA driver, selected OpenMP libraries and HIP/rocWMMA headers
  remain preexisting provenance gaps (gh-ocannl-1026), not reasons to disable supported caches. Outer `None` means
  concrete device discovery failed and bypasses all shared cache I/O. Execution/tuning remain
  available. `schedule_cache_device` pins device/metadata separation, bypass, and real-backend replay.

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
  restating it.
- A search's COST is read from `autotune_progress` lines, not `autotune_log` (gh-ocannl-1061):
  the latter is per-candidate and times an extra untuned-default control, so it moves the cost it
  would record; the former adds a clock read per candidate, is flushed line by line (a search
  killed at a benchmark cap keeps its record), and splits a search's `elapsed_s` into candidate
  `compile_s` and `timing_s`. First reading, gpt2_mini cc on mac-studio: 79 s before the first
  seed (base compile, analyses, the ~2 s baseline's timing window), then timing windows dominate
  compiles about 9:1 on second-scale candidates — so on cc a candidate's cost is its sample count
  times its step, not its compile. The format is the interface's (`Autotune.progressf`).
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

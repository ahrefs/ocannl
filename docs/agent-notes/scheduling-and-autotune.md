# Scheduling and autotune

Schedule legality and coverage, sketch families, and how to read what a search actually did.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

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
  produced — safe only because the threadgroup is exactly one simd width; a cross-nest simdgroup
  barrier is the formal fix. Query the analysis at the SITE'S arity (`aligned_chains ?max_chain`,
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
  unconditionally for parallelism its 2-loop presets cannot use. Since gh-ocannl-577 the
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
  `model_default`'s placement tree, whose children carry `(flip_candidate, `Keep | `Flip)`.
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
  backstop; they emit one `Workgroup` loop per nest, which is why no in-tree annotator can reach
  the `.z` cliff and why `test/operations/launch_dim_gate.ml` builds that geometry by hand.
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
  `launch_dims`; its negative control is a blocked conv whose outer grid loops are a 65536 batch
  and a small non-row spatial axis, so the row blocks bind `.x`, the spatial extent `.y`, and the
  batch folds ALONE onto `.z` at 65536 -- one past the cap. It proves the seed exists under a
  permissive cap but is absent at the device cap. Blocking is what makes it a `.z` question:
  unblocked the site has two grid coordinates and the batch lands on `.y`.

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
  staged twin is seeded only where the emission can actually read that layout, which on CUDA is
  the uniform-bf16 combination and not fp8 (whose B side has no 16-bit `ldmatrix` form at the
  orientation the staged sketches mint). The census distinguishes `Mma_intrinsics_ldmatrix` from
  `Mma_intrinsics`, so "tensorized" and "fed at rate" are separable in a sweep.
- **The register-tile geometry is a schedule decision, not a renderer constant** (gh-ocannl-619).
  `Schedule.Tensorize` carries `tile : Register_tile.t option` (`{rm; rn; lanes}`) into
  `Low_level.Tile_mma`; `C_syntax.try_register_tile` honours a request EXACTLY or declines it to
  the scalar fallback with the violated rule (`Register_tile.check`: a lane count on the file's
  ladder that `n` fills, `rm <= m`, `rn * lanes <= n`, `rm*rn + rm + rn` within the 19/34-register
  budget) — never substitutes, so a candidate timed under a geometry label ran that geometry or
  ran scalar, and the census says which. `None` is the renderer's ranking model,
  `Register_tile.default` (the gh-575 fit: constant peel weight 10, keyed on the actual `n`), which
  the seeding consults through the same module: every CPU tensorized leaf gets a `register-tile`
  level of `auto` plus `Register_tile.alternatives` (peel-free `rn >= 2` at the widest fitting
  width, plus the budget cap when it peels at most one vector per row), only where at least one
  exists — a rule that seeded the cap everywhere doubled the CPU tensorized seed count. The emitted header appends
  `; geometry from the schedule` on a request and nothing on a default, so pre-619 codegen goldens
  stand. Sweep a width by seeding it or by handing `?tile` to `Sched.tensorize` — never by
  patching the renderer again. The cache saves the field as `[@sexp.option]`, so pre-619 entries
  parse. Not done: `rm` alternatives (seeding varies the width only), the conv family (not
  tree-factored), and a `Cost_model`-derived peel weight.
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
  and the derivation happens inside `placement_surface`, on the `profitable` path only, so a run
  pinned to `cost` or `enablement` never reads `tune_flip_profit_margin` (`ps_profit` is `None`
  there, which is what the log line reports instead of a verdict nothing consulted). A malformed
  margin is a `Utils.User_error` and must reach the caller: `tune_placements`' containment around the
  decision-surface lowering names the classes it does NOT absorb, because swallowing that one skips
  the refinement the configuration asked for and ships the A/B winner as though the setting had been
  honored.
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
  only when the pair's inferred fixed component is below the target and no more than one
  quarter-target negative (the noise tolerance matching that confirmation step), so one fixed stall
  or a physically invalid negative fit cannot select a shallow final depth while ordinary
  submit/sync overhead remains in the wall model. When a clean pair's fixed component alone fills
  the wall target, its positive marginal slope selects a depth carrying ~10 ms of launch work: the
  fixed stall does not select a shallow base, and a genuinely slow kernel does not jump to a
  many-second cap batch. A confirmation more than 2x its supported target-sized base
  is itself treated as a contention outlier and retried once before an unresolved pair selects the
  cap, so one transient stall cannot inflate both the timed batch and its later refusal threshold.
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

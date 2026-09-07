# Changelog

These notes record each release's outward-facing key achievements — features, architectural
choices, important bug fixes — not ongoing progress. Finer-grained history lives in merge
commits, PR pages (development happens in `lukstafi/ocannl-staging`), and issue threads
(`ahrefs/ocannl`, cited as gh-ocannl-NNN).

## [Unreleased]

### Added

- `fp16_arithmetic` is ternary (`auto|true|false`, default `auto`): `false` gives every f16
  reduction accumulator f32 residency. `Numerics.policy.fp16_arithmetic` is `Fp16_auto` (old
  `false`) | `Fp16_narrow` (old `true`) | `Fp16_wide` (new) (gh-ocannl-680, gh-ocannl-789).
- Merge-buffer reads are execution dependencies: a routine reading `p.merge` depends on the
  `Context.copy` that filled the slab, which refuses to overwrite a slab a compiled consumer still
  awaits (gh-ocannl-766, gh-ocannl-779).
- Dead zero stores before localized serial reductions are elided when the reduction provably
  overwrites every cell (gh-ocannl-821).
- Schedule-cache directories carry a regime stamp: an older one prunes the stale generation, a
  newer or malformed one refuses reads and writes; the library's temp-then-rename publications go
  through `Utils.Atomic_file` (gh-ocannl-835, gh-ocannl-780, gh-ocannl-803).
- Benchmark provenance and supervision: `benchmarks/fixtures/DIGESTS.txt` records which box's
  fixture bytes each published number was measured on (gh-ocannl-759); sweep children run under a
  per-cell wall-clock cap, on by default (gh-ocannl-842, gh-ocannl-760).
- `Context.codegen_capabilities` (`supports_f64`, `accum_prec`, asynchronous staging copies) and
  `Ir.Backend_intf.advertises_mma_format` answer capability questions tests used to guess from
  the backend name; `Autotune.report.fiss_mma_candidates` scopes the MMA count (gh-ocannl-822).
- `mma_capability` says at which emission lifetime wide-f16 accumulation holds (`Mma_per_statement`,
  `Mma_fragment_scope`); seeding derives the scope a site needs from its reduction extents, and CUDA
  withholds multi-statement wide-f16 schedules where HIP and Metal advertise both (gh-ocannl-836).
- Metal keeps uniform-f16 MMA candidates under `Fp16_wide`: half fragments feed a float
  `simdgroup_matrix` accumulator, narrowed once at the store boundary (gh-ocannl-837).
- `tools/test-run.sh repeat [--alone] N <dune args>` runs one target N times in a clean build
  directory and reports whether the iterations agree, so "does it repeat?" has a supported answer
  (gh-ocannl-896).
- Benchmarks: `fixtures/DIGESTS.txt` declares the measurement fleet (`# measurement-boxes:`), so a
  box with no record for a fixture is reported rather than invisible (gh-ocannl-850); beam cells
  default to tinygrad `PARALLEL=0`, with `--beam-parallel N` the opt-in (gh-ocannl-843).

### Changed

- Autotune's default `Queued` timing measures candidates under queue depth (`Isolated` stays
  selectable), withholds contended or non-finite timings from ranking, and reports
  `{ ms; contended; samples }`; `report.timing` is a `timing_mode` (gh-ocannl-755, gh-ocannl-855, gh-ocannl-888).
- Queued timing on CUDA and HIP calibrates its batch depth to the ~10 ms wall the contention rule
  was set for (queue cap 200 → 2048; cc and Metal unchanged); their schedule-cache entries carry
  the `queued-v2` policy generation, so earlier winners re-time rather than replay (gh-ocannl-892).
- Convolution sketch seeds predict their launch geometry, and over-cap candidates are withheld
  before compile as matmul seeds already were (gh-ocannl-739).
- RTC options (`Compiler_options.nvrtc`, `.metal`, `.hiprtc`) are pure, pinned state, printed by
  sweeps and appended to CUDA, HIP and Metal compile failures; Metal on macOS 15+ selects
  `mathMode=Safe` with fast float functions (gh-ocannl-784, gh-ocannl-848, gh-ocannl-849).
- Metal's serial-accumulation workaround is a volatile cast on device reads, far cheaper than
  the volatile accumulator; the capability is `volatile_serial_accumulation` and
  `Context.routine.volatility` says which sites took it (gh-ocannl-782, gh-ocannl-820).
- Retired API: `Backend.compile_batch` / `link_batch` (gh-ocannl-767); `Tensor.raw_unop` /
  `raw_binop` / `raw_ternop`, collapsed onto `Tensor.raw_accum` + `Tensor.buffer_of` (gh-ocannl-812);
  `Tensor.diff.zero_grads` is a `comp`, not an `asgns` — read `.asgns` (gh-ocannl-771).
- `Indexing.projections` carries one `components` array, not `product_space` /
  `product_iterators` (gh-ocannl-775).
- `Context.Backends_deprecated.footprint` is `Ir.Low_level.footprint` and
  `Context.Backends_deprecated` is `Context.Backends` (gh-ocannl-810).
- `Assignments`, `Indexing`, `Affine`, `Interval`, `Host_inits`, `Compiler_options` and
  `Cpu_topology` have explicit interfaces that hide their zero-reference helpers
  (`Assignments.fold_leaves`, `Affine.equal_verdict`, ...); the removed surface is under gh-ocannl-806.
- `PrintBox_utils` and `Task` have explicit interfaces: `PrintBox_utils.concise_float`, `nolines`
  and `render_group` are private, and its `sexp_of_box`, `sexp_of_dag` and `sexp_of_table_row_spec`
  converters are removed (gh-ocannl-915).
- `Shape.update_step.neutral_elem` is immutable, fixed at creation, and `Shape.product_space_shape`
  takes `~shape ~logic` rather than an `update_step`; `Tensor.op` runs `op_asn` under
  `Shape.with_construction_window`, which refuses a finalization before the op's constraints exist (gh-ocannl-830).
- `Train.to_routine` returns `Context.t * Context.routine`, like `Context.compile`; fix the type
  error with `let _, routine = ...` or chain the context (gh-ocannl-772).
- `?lowered_transform` / `?lowered_transforms` on `Context.compile`, `compile_outcome` and backend
  `compile` are one `?lowered_transform : optimized -> optimized list` (gh-ocannl-768).
- `Context.device_id` is `Context.ordinal`, `?device_id` parameters are `?ordinal`, no alias
  (gh-ocannl-776).
- The memory-budget planner is the `arrayjit.memory_budget` library: `Context.plan_memory_budget`
  → `Memory_budget.fit`, `memory_budget` → `.t`, `budget_plan` → `.plan`, `footprint` and
  `compare_relief_ratio` → `Memory_budget.*` (gh-ocannl-776).
- `Get_dynamic` gather tables decide their own placement: a table that recomputation cannot
  serve materializes (gh-ocannl-734).
- `batch_norm1d`/`batch_norm2d ?momentum` are `?_momentum` and `mobile_cnn ?width_mult` is
  `?_width_mult`: placeholders with no effect, now labelled as such (gh-ocannl-811).
- Scripts under `tools/`, `scripts/` and `benchmarks/`, dune files, workflow YAML,
  `ocannl_config` files and the docs under `docs/`, `AGENTS.md`, `README.md` are scanned against
  `Utils.known_config_keys`, so a renamed key fails the scans instead of drifting (gh-ocannl-790).

### Fixed

- `multidev_cc` launched kernels on whichever static index the host had raced ahead to, so
  `Train.sequential_loop` skipped and repeated batches on that backend alone; launches now bind
  the index at dispatch (`lukstafi/ocannl-staging` PR #592, found and fixed there).
- `Train.sgd_update ~momentum` works: the momentum buffer was left a virtualization candidate,
  failing at lowering; `test/operations/sgd_variants` pins momentum, nesterov, weight decay and
  `grad_scale` against a host simulation (gh-ocannl-772).
- The mul-add→FMA rewrite is guarded to floating-point precisions; int64 mul-add went through
  double `fma` and lost integers above 2^53 (gh-ocannl-824); the C-family renderer now also refuses
  an integer-precision `FMA` outright instead of mapping it to `fma`/`fmaf` (gh-ocannl-873).
- `Tensor.op` raises `Session_error` when a custom `op_asn` forces projections before the
  neutral element is installed, which silently miscomputed padding and guards
  (`lukstafi/ocannl-staging` PR #506).
- `Set_vec_unop` lowering refuses a launch-bound symbolic extent, a defensive guard behind the
  shape solver's existing rejection of such graphs (gh-ocannl-817).
- Reducing over a bound symbolic axis declared with maximum 1 returned its operand where the empty
  sum is 0: the axis has no iterator for the extent guard to attach to, so an extent of zero still
  ran the body once; `Accum_op` lowering now refuses it and names the remedy (gh-ocannl-878).
- Metal on macOS 26 silently took the macOS 14 legacy compile path (`fastMathEnabled=false`): the
  class-level selector query answered false for properties the options object accepts; the query
  now asks an initialized `MTLCompileOptions` (gh-ocannl-881, gh-ocannl-882).
- `debug_backend=flushing` ignored `location_format=no_location`, on both the log-file and the
  stdout-prefixed destinations (gh-ocannl-876).
- `OCANNL_LOG_LEVEL_CC_BACKEND=1` and `=3` compile again (gh-ocannl-823).

## [1.0.1] -- 2026-08-26

> Release note: theme — consolidation after 1.0: making a green result mean what it says. This is
> the release previously planned as v1.1, renumbered because version-number depth tracks release
> scope in this project (as in the 0.6.x line): the ladder is now
> `1.0 → 1.0.1 → 1.0.2 → 1.1 → 1.1.1 → 1.2` (see ROADMAP.md). Its 135 closed issues were mostly
> filed by PR review cycles, and their common shape is trust: a failing check that cannot be
> `dune promote`d into a golden, an environment variable that cannot be mistyped silently, an
> inlined computation that cannot lose its guard or its repetition loop, a reduction whose
> accumulator width cannot depend on which schedule won, and benchmark rows that say which pass
> produced them and which fixture bytes they measured. The training-loop mechanics land here too:
> LR schedules, global-norm clipping, gradient accumulation, mmap-backed checkpoint loading with
> its Windows arm measured and then pinned by a cross-platform regression test, and
> `trainable_params` as distinct from "needs initialization".
>
> Consolidation still moved the performance needle, because several soundness fixes were
> performance fixes: precision-neutral accumulator localization took the Metal `gpt2_mini` forward
> step p50 from 367.1 to 93.9 ms (−74.4%, gh-ocannl-693); batch-grid twins took the HIP step 1.72x
> (gh-ocannl-643); finer fission took the tuned CUDA step 1.43x (1.56x at tf32, gh-ocannl-574);
> and on CPU, whole-vector FMA took packed f32 GEBP from 12.6 to 127.2 GFLOP/s at the default
> flags (gh-ocannl-614), with the auto-resolved AVX-512 width then worth 130.5 → 225.7 GFLOP/s on
> a Zen 5 (gh-ocannl-621, gh-ocannl-648).
>
> Honest nulls, recorded as such: the two-pass benchmark protocol stays OCANNL-only — no other
> framework's searching cell clears ~10% on both GPU boxes, and the old 2.5–3.5x rationale is
> superseded by a measured ≤10.3% (gh-ocannl-675); the menu's descent into virtualization-inlined
> scopes measured zero loops reached across the whole suite (gh-ocannl-687); and the gh-573/gh-574
> HIP ratios were re-verified end to end after the first session's arm A routines turned out
> profiled-but-never-executed in three of four cells (gh-ocannl-612).

### Added

- **Training-loop mechanics** (gh-ocannl-465): host-side LR schedules (`Train.Lr_schedule`:
  cosine / linear / constant / WSD with warmup, via `Train.host_scalar`); global-norm gradient
  clipping (`Train.grad_l2_norm`, `Train.clip_by_global_norm`, folded into the update by
  `sgd_update ~grad_scale` — gradient buffers stay untouched, llm.c-style); gradient accumulation
  (`Train.grad_update ~accum_steps`, `Train.zero_params_grads`); and the `Train.Outlier_detector`
  sliding-window z-score monitor. Executed-parity coverage against host oracles.
- **Mapped checkpoint and safetensors loading** (gh-ocannl-467, gh-ocannl-587, gh-ocannl-588):
  payloads load as private, copy-on-write `Unix.map_file` regions instead of element-by-element
  decodes, so pages are read lazily by the OS; the checkpoint format gained an `alignment` field
  (old readers and old files keep working); unaligned payloads are decoded rather than mapped (an
  unaligned mapping is undefined behaviour); and the Windows carve-out was measured, found
  unnecessary, and retired — mapping defaults on for every platform (opt out with
  `checkpoint_load_mmap=false` or `?mmap:false`, for a filesystem that does refuse replacement
  under a live mapping), pinned by a cross-platform
  regression test. **Behavior change**: `Safetensors.to_ndarray` returns each payload's own
  precision instead of forcing F32 — pass `?prec` for a fixed target — and I8/I16/F8_E4M3 payloads
  are refused rather than reinterpreted (`to_float32` keeps its F32-only contract).
- **Frozen parameters are not trained** (gh-ocannl-670, gh-ocannl-673): `Train.trainable_params`
  derives the trained subset from the backprop code (`?params` to override) while `loss.params`
  keeps its union semantics as the state set; params-driven helpers over a loss that trains
  nothing raise at construction instead of compiling empty routines.
- **The pre-driver launch gate covers every hardware launch dimension** (gh-ocannl-679,
  gh-ocannl-643, gh-ocannl-684): per-dimension workgroup caps (`max_workgroup_dims`) and the `.y`
  grid extent join the checks, enumerated from one table rather than hand-copied per bound
  (`Backend_intf.max_grid_z` became `max_grid_yz`); new
  `bin/device_props` prints the queried device properties and derived limits the gates compare
  against.
- **Tuning ergonomics and verification**: `Autotune.tune` takes `?name` like `Context.compile`, so
  a nameable computation is a tunable one (gh-ocannl-669); `tune_ship_arm` lets a measurement ship
  a chosen placement arm so the profiled artifact is the one whose losses are reported
  (gh-ocannl-638); and the gh-573/gh-574 HIP measurement was re-verified end to end with every
  quoted routine executed — gh-573 is worth 1.28x end to end (gh-ocannl-612,
  `benchmarks/report-gh612-hip-verified.md`).
- **Cache identity is complete by construction** (gh-ocannl-572): every configuration key is
  classified against the schedule cache's identity (code-borne / keyed component / search-shaping
  / neutral), enforced by `test/operations/digest_completeness`; the cache key gains a codegen
  component covering the emission gates, each backend's codegen knobs, the whole `hardware_limits`
  record, and a cc toolchain fingerprint.
- **The virtualizer's rejection boundary is pinned row by row** (gh-ocannl-658): one minimal,
  executed IR shape per outcome, stating which phase decides it and under which provenance; and
  hand-built lowered code can be executed, not only analyzed, via `Context.compile ?prelowered`
  (gh-ocannl-562). **Behavior changes**: `Context.routine` became a `private` record — fields
  stay readable, but out-of-tree code can no longer construct or functionally update one — and
  `Context.get_values` / `set_values` on a node the
  optimizer placed `Local` now raise instead of serving an unrelated uploaded copy — host access
  requires materializing the node (gh-ocannl-599).
- **A fault-injection inventory for resource-owning seams** (gh-ocannl-571), with a GPU-free
  harness over finalize/release, pool, merge-pool, transfer and cache seams — it found a real
  `from_host` leak before the harness was done; and the cost model's **memory leg is fittable**
  from STREAM-style bandwidth calibration (gh-ocannl-578).

### Changed

- **Soundness fixes that were performance fixes**: reduction accumulators are localized without a
  widening request — an f32 reduction accumulates in a local scope instead of one global-memory
  read-modify-write per step — taking the Metal `gpt2_mini` forward step p50 from 367.1 to
  93.9 ms (gh-ocannl-693); `sk_batch_grid` twins bind the batch/head product to `blockIdx.z`,
  1.72x on the whole HIP step (gh-ocannl-643); finer fission ships the lm_head GEMM apart from its
  row-max so its output axis can spread, 1.43x on the tuned CUDA step (gh-ocannl-574); and
  accumulation chains materialize a running sum instead of re-summing at every consumer
  (`virtualize_max_inline_fanin`, default 8), removing the transformer residual stream's
  quadratic re-derivation (gh-ocannl-573).
- **Accumulator width is policy, not schedule** (gh-ocannl-639, gh-ocannl-663): every
  serial-rendered form of a reduction holds its accumulator at the backend's declared residency
  and narrows once, pinned by a 27-member executed table of schedule compositions
  (gh-ocannl-664); HIP compiles with `-fno-associative-math -fhonor-infinities` so narrow
  reductions and the `(-INFINITY)` mask sentinel survive fast math (gh-ocannl-735).
- **CPU SIMD: whole-vector FMA and the widths to use it at** (gh-ocannl-614, gh-ocannl-621,
  gh-ocannl-648): packed f32 GEBP 12.6 → 127.2 GFLOP/s at the default flags; `cc_vector_bytes`
  auto-resolves to 64 where the target asserts AVX-512; `Max`/`Min` reductions render
  compare-and-select instead of a libm call per lane; and a cross-target `-march` census keeps
  guarded arms compiling wherever a toolchain for them exists (gh-ocannl-650).
- **CPU register tiling reaches 16-bit GEMMs** (gh-ocannl-575): `try_register_tile` joined the
  storage/compute-precision seam, packing stages can mint tiles at the compute precision, and
  pure-fp16 seeds fire exactly where the probe reports native arithmetic — f32-GEBP-over-narrow
  storage reaches 51.3 GFLOP/s at bf16 against a 0.4–0.7 GFLOP/s scalar rendering.
- **fp8 everywhere it was almost** (gh-ocannl-632, gh-ocannl-647, gh-ocannl-657): Metal stores
  e5m2 as a byte with a codec bit-identical to `builtins.c` over all 2^32 floats; HIP guards the
  vendor's defective float→fp8 narrowing; the codecs' exhaustive verification lives in-tree
  (`@slow-fp8_codec_exhaustive`, plus `tools/fp8_soak.exe` as the CUDA/HIP GPU arm).
- **The flip chain's enablement promotion is weighed against measured profitability**
  (gh-ocannl-579, closing the gh-ocannl-514 arc): the new default `tune_flip_ordering=profitable`
  demotes the enablement prior when the family it promotes lost by more than
  `tune_flip_profit_margin` (default 1.25), using evidence the placement A/B already pays for;
  `=cost` and `=enablement` remain as the two unconditional baselines.
- **A "tensorized" timing can no longer be a scalar-fallback timing** (gh-ocannl-626): the
  `Tile_mma` rendering census is derived once inside `Context.compile` and carried on the routine
  (`Tensorized` / `Scalar_fallback` / `Not_requested`), printed wherever timings are reported and
  read into the benchmark reports.
- **The sketch families are a module, and their trees judge what seeding owns** (gh-ocannl-577,
  gh-ocannl-580, gh-ocannl-613, gh-ocannl-591): family construction moved out of `autotune.ml`;
  statically-decidable builder preconditions are tree verdicts, so a refuted family fathoms at
  the root instead of dying candidate-by-candidate; epilogue fusion is a root-level `Choice`; and
  decision labels are typed data. The enumeration and menu keep up with what the schedule ops
  mint (gh-ocannl-666, gh-ocannl-687, gh-ocannl-685, gh-ocannl-709), and `Autotune.report` states
  its outcome as a five-state variant (gh-ocannl-677).
- **Cross-routine splices are reconciled, and deferred computations have stated semantics**
  (gh-ocannl-610, gh-ocannl-611, gh-ocannl-617, gh-ocannl-618): spliced-in leaf reads get
  declared entries, read-before-write flips are judged by per-cell definite-write coverage, a
  routine that optimizes to nothing is a legal empty result, and recompute-at-read is the
  documented semantics of virtual nodes (`docs/lowering_and_inlining.md`).
- **`Local_scope` has a contract, enforced at both ends of the pipeline** (gh-ocannl-584,
  gh-ocannl-681, gh-ocannl-704): a scope body's only effect is on the locals it owns — two live
  miscompiles fell out — and a scope over a materialized node is rejected by name instead of
  silently collapsing to a `Get`.
- **One environment spelling, and no silent demotion** (gh-ocannl-605, gh-ocannl-628,
  gh-ocannl-652): a configuration key is read from `OCANNL_<KEY>` and nothing else; a known key in
  a spelling nothing reads is a fatal startup error naming the spelling that works (on
  case-sensitive platforms — native Windows's case-insensitive environment reads the lowercase
  spelling as the same variable); a mistyped
  variable warns by name (gh-ocannl-629); and per-directory `env_spelling_gate` rules plus
  `env_var_deps` make dune reruns track exactly the declared variables. On the commandline,
  dashes now go all the way (`--ocannl-log-level=1` is read alongside `--ocannl_log_level=1`),
  while a bare `ocannl_log_level=1` with no leading dash is no longer read at all — a positional
  argument belongs to the host application (gh-ocannl-605). `Utils.env_var_names` became
  `Utils.env_var_name`.
- **Test seams that cannot report a false pass**: a test that decides its own verdict reports it
  through `Verdict`, which exits the process nonzero so a regression cannot be `dune promote`d
  into a golden — with a ratchet against unguarded claim prints and empty-collection-safe
  quantifiers (gh-ocannl-601, gh-ocannl-668, gh-ocannl-729); generated-kernel assertions
  establish artifact provenance through `Test_utils.Generated` (gh-ocannl-655, gh-ocannl-723);
  every test stanza declares `ocannl_config` and the repository scans glob rather than trust hand
  lists, with `codegen_text_inventory` enumerating every file that pins emitted-kernel text
  (gh-ocannl-586, gh-ocannl-592, gh-ocannl-703, gh-ocannl-712); floats from device reductions
  stay out of stdout goldens (gh-ocannl-725); and each `@slow` test gets its own `slow-<name>`
  alias.
- **Benchmark trust** (gh-ocannl-675, gh-ocannl-676, gh-ocannl-634, gh-ocannl-737): a diverged
  cell reports DIVERGED naming the step instead of a runner failure; the tinygrad-BEAM and
  torch.compile cells were measured to not need OCANNL's two-pass protocol (≤10.3%, superseding
  the README's 2.5–3.5x rationale); checksums key on the (row, column) pair; `bench_mlp
  --self-test` smoke-tests the measurement path with no fixtures or Python venv; one `Bench_args`
  argv convention across `bin/`; and a green GPU sweep row records whether tests actually ran
  (`incremental-pass` vs `forced`).
- **Startup chatter is opt-in, so a warning on stderr is legible** (gh-ocannl-593, gh-ocannl-595):
  `log_config_sourcing` now defaults to false and `log_level` to 0 (raise `log_level` to restore
  the previous diagnostics), taking a default run's stderr to three lines; the startup
  streams (empty stdout, readable stderr) are pinned by test.

### Fixed

- **A guarded setter's computation is no longer inlined without its guard** (gh-ocannl-651), and a
  candidate's computation is no longer inlined without its enclosing repetition loop
  (gh-ocannl-674) — both stored the computation with the guard or loop silently dropped,
  replaying wrong values at every read site; both now reject at store time.
- **RoPE attention gradient-flow loss diverged 68% on Metal** (gh-ocannl-731): localized reduction
  accumulators join the `volatile` workaround for Metal's pooled-slot compiler-bug family.
- **Intermittent SIGBUS at startup of every arrayjit executable** (gh-ocannl-688): the `Ir.Ops`
  link-forcing block called the uint4x32 stubs with one-element arrays whose four lanes the stubs
  read blind.
- `uint32`/`uint64` ndarrays convert to and from floats as unsigned — a u32 `0xffffffff` used to
  read back as `-1.` through the signed host conversion.
- Emitted float constants are floating literals that round-trip (forced radix point, `%.17g`
  retry; gh-ocannl-623, gh-ocannl-713), and routine names cannot crash codegen — reserved-word
  and builtin collisions are mangled once per compile entry (gh-ocannl-686).

## [1.0] -- 2026-08-13

> Release note: theme — advanced compiler tiers and schedule-quality follow-through. This release
> marks the compilation side reaching the shape argued for in
> [the compilation manifesto](docs/compilation_manifesto.md): schedule inference is a search in
> constraint space, with legality and cost queries answered over *partial* schedules and every
> refusal carrying its witness (gh-ocannl-514). Inlining joined scheduling as a first-class,
> searchable decision (gh-ocannl-555) once the concrete-index tracer was retired in favor of the
> affine access relations (gh-ocannl-554), so the two decision spaces are now one surface. The
> advanced tiers filed for this milestone all landed: CUDA/HIP graph capture (gh-ocannl-488),
> software-pipelined double-buffered staging (gh-ocannl-487), budget-driven rematerialization
> (gh-ocannl-498), the CUDA tensor-core profile's remaining shapes (gh-ocannl-481), and CPU reduced
> precision at both 16-bit tiers (gh-ocannl-517, gh-ocannl-516).
>
> The end-to-end number the milestone is measured by is `gpt2_mini`. The v0.9 sweep left it 72x off
> torch CUDA; the attribution profile (gh-ocannl-531) put 70.2% of the step in five kernels declined
> by one companion-coverage rule, and fixing that rule at the site's arity (gh-ocannl-569) took the
> tuned step from 107.4 to 52.4 ms on CUDA (47.9 ms at tf32) and from 45.6 to 25.4 ms on HIP —
> ~2.05x and ~1.79x, against a predicted floor of ~48.6 ms. Batched/rank-3 matmul sites are seeded
> (gh-ocannl-528), so tensor cores are reachable on transformer workloads at all.
>
> Several results in this release are honest nulls, recorded as such rather than shipped as wins:
> `ldmatrix` over swizzled staging is emitted, crowned and worth +0.1% (gh-ocannl-481); barrier
> elision is free but not faster on a one-simdgroup Metal threadgroup (gh-ocannl-567); software
> pipelining is a ~8% win on CUDA `cp.async`, a null on HIP, and a 1.4–1.5x *cost* on Metal
> (gh-ocannl-487); site-targeted materialization works and loses to both A/B arms (gh-ocannl-558);
> and branch-and-bound's fathoming is real and cheap while its promised strict dominance is not
> reachable everywhere, with the reasons filed as gh-ocannl-577/578/579 (gh-ocannl-514,
> [report](benchmarks/report-gh514-eval.md)). The advisory contract on the cost model held in every
> measured cell: a candidate the model cannot price is never dropped.

### Added

- **Schedule inference as branch-and-bound** (gh-ocannl-514): `Ir.Schedule_space` replaces
  enumerate-then-filter with a refinement tree over *partial* schedules — legality answers typed
  verdicts with witnesses so subtrees fathom pre-compile, `Cost_model.completion_floor` is an
  admissible lower bound, and the placement walk and staged tile lattice ride the same tree. The
  three-machine evaluation is a research output in its own right
  ([benchmarks/report-gh514-eval.md](benchmarks/report-gh514-eval.md)), its nulls stated as nulls
  and its causes filed (gh-ocannl-577/578/579).
- **Inlining as a first-class, searchable schedule decision** (gh-ocannl-555): `Low_level.optimize`
  splits into analyze and specialize so one analysis replays per candidate;
  `Context.decide_inline` carries per-node preferences (legality and observability untouched);
  `optimized.flip_candidates` exposes the cost-ranked decisions; `Train.tune_placements` gains a
  greedy flip refinement (`tune_inline_flips`).
- **CUDA and HIP graph capture of the fissioned step** (gh-ocannl-488), behind
  `gpu_graph_capture`: launch-bound steps gain what launch overhead cost them and nothing else —
  `mlp_small` 0.105 → 0.055 ms p50 on CUDA, ~2x on HIP, `lenet` unchanged; loss trajectories
  bitwise identical.
- **Software-pipelined double-buffered staging** (gh-ocannl-487): `Schedule.Stage
  ~pipeline_depth`, with CUDA emitting `cp.async` on sm_80+. Verdict per substrate: CUDA ~8%
  where the k-loop dominates, HIP null, Metal a 1.4–1.5x cost; refinements are gh-ocannl-576.
- **Budget-driven rematerialization** (gh-ocannl-498): a deterministic selector trades storage for
  recompute over `Inline` flips until the liveness-arena layout fits config `memory_budget`.
- **Batched and rank-3 matmul sites are seeded** (gh-ocannl-528), so transformers reach tensor
  cores at all: on `gpt2_mini`, 16 tensorized candidates timed where there were 0, crowned
  recombination 1.77x against untuned.
- **The CUDA tensor-core profile's remaining shapes** (gh-ocannl-481): fp8 `mma.sync`, typed
  swizzle marks, `ldmatrix` over swizzled staged tiles, per-family arch markers. The measurement
  is a null on the record: +0.096% pooled over 45 paired timings.
- **Site-targeted materialization** (gh-ocannl-558): `Context.decide_materialized` pre-seeds
  individual nodes; it works and loses 0/3 measured cells to the existing A/B arms — neither
  placement arm dominates globally.
- **Config profiles `reproducible` and `performance`** (gh-ocannl-559), picked through the
  ordinary sources with picker-inherited precedence (explicit keys beat a profile of equal
  immediacy). `reproducible` is deterministic and, wherever reasonable, identical across machines
  (cross-backend reproducibility is out of its scope); `performance` is the fastest
  configuration at unchanged semantics, with one named exception — it enables
  `fp16_arithmetic`, the mantissa-for-throughput trade — while result-changing gates like
  `tf32_matmuls` stay on the orthogonal numerics axis. The reference config now ships with every
  setting commented out.
- **Honest search reporting**: `Autotune.report` gains `best_label`, `best_tensorized` (read off
  the winner's schedule, not its label) and `mma_best_ms`, and the placement A/B states when a
  tensorized winner is discarded (gh-ocannl-546, [report](benchmarks/report-gh546-metal.md));
  the tuner reports the untuned default's measured time as `default_ms`, the honest reference
  point (gh-ocannl-552); and `gpt2_mini_train` joins the benchmark suite with gate-cost precision
  legs that report `NOT APPLICABLE` rather than going silently missing (gh-ocannl-551).

### Changed

- **The concrete-index tracer is retired** (gh-ocannl-554): the affine access relations
  (gh-ocannl-494) answer exactly what the enumerating tracer approximated;
  `virtualize_max_tracing_dim` is gone. One `Low_level` analysis is shared across sibling
  candidate compiles under a canonical digest (gh-ocannl-560); affine access paths encode
  intra-statement order with typed components (gh-ocannl-561); and digesting and cache
  canonicalization share a single canonical-llc emission core (gh-ocannl-563).
- **`cc` restricts its worker pool to one core class on hybrid CPUs** (gh-ocannl-530,
  `cc_pool_core_class`): conv-sketch tuning wins that did not port across CPUs traced to pool
  *heterogeneity* — uniform subsets recover 25–34% of tuned time — and the pool signature enters
  the autotune cache key, because crowns do not transfer across pools.
- **Native fp16 arithmetic on the CPU backends** (gh-ocannl-516, config `fp16_arithmetic`,
  default **false** — opt-in, since it trades mantissa for throughput): where the hardware has
  genuine 16-bit arithmetic, fp16 computes in fp16 at twice f32's lanes —
  1.99x on an M-series compute-bound control. `cc_backend_arch_flags` defaults to `auto`
  (`-march=native` silently *downgrades* Apple clang arm64 targets; `auto` probes the family's
  right spelling).
- **16-bit storage, f32 compute on the CPU backends** (gh-ocannl-517, `narrow_compute_f32`,
  default **true**; setting it false restores the previous per-operator rounding): a
  node's precision is storage-only; `cc` computes narrow floats in f32 — strictly more accurate,
  and the precondition for vectorizing 16-bit nodes at all. Measured verdict recorded: fp16
  storage 1.97x on a bandwidth-bound add, bf16 0.91x (its narrowing costs more than the halved
  traffic saves).
- **Operation results close down; `stretch` requests use-site resolution** (gh-ocannl-544): an
  operation result's open row closes to its arguments' shapes and the use site broadcasts it in;
  use-site resolution remains the rule for leaves, parameters and init expressions, and the new
  identity `stretch` requests it by name (`stretch 1.0` replaces the `0.5 + 0.5` idiom).
- **Four explicit barriers per k-block become one** (gh-ocannl-567): load phases group at the same
  anchor and elision keys on `workgroup_shared` (config `elide_pipelined_barriers` renamed
  `elide_staged_barriers`). Free rather than faster on Metal (+0.2% against a ±3–7% noise floor),
  measured and recorded.
- **Config startup chatter goes to stderr** (gh-ocannl-581), so tools whose stdout is a data
  channel need no suppression incantation.
- **CI on OCaml 5.5; Windows off the per-PR path onto a schedule; GPU backends on a daily
  cross-machine sweep** (`tools/sweep.sh`) — CI's runners have no GPU, so Metal/CUDA/HIP were
  never covered there. The `cc` backend's toolchain probes are memoized machine-wide.
- `cudajit` and `hipjit` are pinned to their releases (0.8.0 / 0.2.0, required by graph capture).

### Fixed

- **Companion coverage is judged at the site's arity** (gh-ocannl-569) — the single largest
  end-to-end win in this release: the gh-ocannl-521 rule capped its inspected chain at two loops,
  declining five FFN-class kernels (70.2% of the CUDA `gpt2_mini` step) into naive scalar forms.
  `aligned_chains ?max_chain` lifts the cap; tuned step 107.4 → 52.4 ms CUDA, 45.6 → 25.4 ms HIP,
  values bitwise-verified.
- **A backend's tensor-core capability is keyed on the accumulator format** (gh-ocannl-545): CUDA
  advertised the bf16 multiplicand *pair* that wmma only supports against an f32 accumulator, so a
  uniformly-bf16 network's "20 timed mma candidates" were all scalar fallback under mma labels.
  The general trap — "timed" is not "tensorized" — is recorded.
- **The autotuner no longer accumulates device memory across candidates** (gh-ocannl-550):
  `Backends.finalize` had no caller anywhere; `Context.release` gives it one and `Autotune.tune`
  uses it — peak 1.9 GB against 11.9 GB unfixed on cold tf32 `gpt2_mini` runs. Also: a failed
  placement arm ranks at infinity instead of destroying the other arm's finished winner; a
  fixable pre-timing mistake is a contained `Preflight` decline rather than a poisoned lineage
  (gh-ocannl-564); and a lineage-wide validation failure propagates instead of reading as
  per-candidate declines.
- **The schedule cache key covers the numerics policy** (gh-ocannl-568): a default-flags run
  sharing a cache directory with a tf32-tuned search used to replay the tf32 winner at 5.9x
  slower than not tuning at all, silently. Pre-existing cache entries are invalidated: one cold
  search per program.
- `simplify_llc` narrows its interval environment from enclosing `If` conditions, so schedule
  guards fold where the guard decides them (gh-ocannl-566); a roofline bound violation is
  reported once per process, not once per compile.

## [0.9] -- 2026-08-03

> Release note: theme — program search and optimization. Since 0.8, the schedule system has
> gained hardware tensor-core paths, native affine legality and cost analyses, symbolic runtime
> extents, convolution sketch families, a liveness-based memory planner, deterministic split
> reductions, and a mixed-precision training recipe. Multi-round search over schedule
> compositions is implemented; cost-model-guided default and beam selection are in (config-gated,
> advisory). End-to-end benchmark validation is complete for this milestone: the cross-machine
> sweep (gh-ocannl-476, re-measured under gh-ocannl-538) was run from a wiped autotune cache on
> Metal, CUDA and HIP, and the checked-in reports under `benchmarks/` are the record. The
> practical theme of the second half of the milestone was making the search *survivable*: a
> candidate the toolchain or the device refuses is now a typed, censused decline rather than a
> fatal, a hung dispatch, or a silent omission.
>
> Reduced-precision status: bf16 and f16 training are correct and measured on every backend, but
> generally slower than f32 (gh-ocannl-535). The sweep located the f16 cost — the loss-scaling
> gate's `grad_checksum`, which lowers to one serial reduction that only split-reduce
> parallelizes; with it scheduled, HIP `mlp_wide` tuned f16-static is the first reduced-precision
> cell in the suite to beat tuned f32. Tensor cores are reached and crowned on CUDA (tf32) and HIP
> (bf16); on Metal a tensorized candidate wins an arm but the placement A/B does not ship it
> (gh-ocannl-546, v1.0). CPU reduced-precision arithmetic (gh-ocannl-516, gh-ocannl-517) is v1.0
> scope.

### Added

- **Native affine program analysis** (gh-ocannl-494): `Ir.Affine` and
  `Low_level.affine_accesses` expose loop boxes and access relations; conflict, coverage,
  fiber-cardinality and read-before-write queries drive shared-memory safety, fission, scratch
  validation, and a `Schedule.op_legality` oracle pruning proven-illegal proposals.
- **Analytic cost model** (gh-ocannl-491): footprint/FLOP extraction, roofline lower bounds,
  per-backend envelopes; `Autotune.model_default` picks untuned defaults with zero timing runs
  (`model_default_schedule`) and `tune` gains a keep-fraction pre-filter — advisory throughout:
  candidates without model coverage are never dropped, and calibration logging pairs scores with
  measured times.
- **Hardware matrix units and packed microkernels** (gh-ocannl-412): CUDA renders `Tile_mma`
  through WMMA (f16/bf16) and inline PTX (fp8); HIP through rocWMMA; CPU schedules compose
  register tiling with cache-blocked packing and pool-parallel panels. Accumulator fragments stay
  resident across the serial reduction (gh-ocannl-480); `Stage ~swizzle` marks XOR-swizzled tiles.
- **Convolution schedule families** (gh-ocannl-493, gh-ocannl-500, gh-ocannl-501): implicit-GEMM
  sketches using `Stage` as virtual im2col, blocked tile flavors, epilogue twins spanning the
  whole kernel-window chain, and `Schedule.Fuse_epilogue` folding elementwise consumers into a
  tile's store-back.
- **Pad-to-tile scheduling** (gh-ocannl-485) and **partition / index-set splitting**
  (gh-ocannl-508): tensorized paths cover arbitrary extents (a 33x65x70 matmul register-tiles
  bitwise-exactly), and partition-then-specialize erases remainder and concatenation guards.
- **Launch-time symbolic extents** (gh-ocannl-490): bounded symbolic axes lower to runtime extent
  parameters, so one compiled routine serves multiple batch or sequence lengths.
- **Liveness-based buffer aliasing** (gh-ocannl-489): the pool planner reuses non-overlapping
  working-buffer slots, with `Zero_out` sinking exposing further training-step reuse.
- **Deterministic split reductions made reachable** (gh-ocannl-537, gh-ocannl-541):
  `Swap`-hoisting composes interchange chains so conv-gradient accumulations — up to 89% of HIP
  lenet's step — reach the gh-ocannl-484 split-reduce family; detected sites are ranked by serial
  work and capped with every eviction censused (`autotune_split_reduce_max_sites`, default raised
  4 → 8).
- **Mixed-precision training recipe** (gh-ocannl-492): `Precision_policy.apply` assigns storage
  precisions by structural class; `Mixed_prec` implements torch-AMP-style master weights via cast
  twins, dynamic loss scaling with `Train.grad_checksum`, and a fused on-device loss-scaling gate
  (`Where`-selection, so skipped steps leave state untouched exactly); forward-only reduced
  precision converts at ingestion. Executed-parity coverage against f32 oracles.
- **Numerics policy with tf32 matmuls** (gh-ocannl-478): `Ir.Numerics` — user-chosen, never
  optimizer-chosen, identical across sibling candidates; opt-in `tf32_matmuls` routes CUDA
  uniform-f32 GEMMs through wmma `precision::tf32`.
- **Packed `uniform` total over shapes** (gh-ocannl-509): no more block-width divisibility
  requirement; it becomes the default parameter init, and packed uniform results can be virtual
  via lane extraction, bitwise-equal to materialized runs on every backend. **Reproducibility
  break**: this changes every default-initialized random stream even at an unchanged seed —
  `uniform1` and `default_uniform1_param_init` are deprecated but kept exactly for reproducing
  pre-0.9 streams.
- **Search survivability**: typed candidate-failure containment with per-backend classifiers and
  damage tracking — a rejection that provably wrote nothing rolls back, one that may have written
  poisons the lineage by name (gh-ocannl-536); HIP scratch-budget pre-validation, since ROCm
  aborts the queue instead of failing cleanly (gh-ocannl-533; set `hip_scratch_validation=false`,
  default true, on a device where the occupancy model is wrong); and a decline census accounting
  for every refusal, including undispatched and cap-evicted candidates (gh-ocannl-532,
  gh-ocannl-541, gh-ocannl-543).
- **Benchmark matrix and sweep**: tuning and precision are independent cell axes (gh-ocannl-539);
  the cross-machine sweep reports are checked in (gh-ocannl-476, gh-ocannl-538) — first
  seeded-timed-and-crowned rocWMMA candidate (HIP `mlp_wide` bf16), CUDA tf32 mma reached, split
  reduction worth 46–82% on the default-placement arm.

### Changed

- **An unparallelized candidate is refused, not dispatched, on GPU backends** (gh-ocannl-532):
  one-work-item kernels measured 6.9 s against a 35.7 ms winner on Metal and ran for hours on
  gfx1151; CPU backends keep the serial form as a legitimate competitor.
- **Attention masks fill with `-infinity`** instead of `-1e9` (gh-ocannl-548), and the fp16
  magnitude guard exempts infinities (gh-ocannl-547) — together unblocking every attention model
  at f16. For padding masks that can cover a whole query row (where `-inf` makes the softmax's
  max-subtraction produce NaN), every mask-taking `Nn_blocks` entry point takes `?mask_fill` to
  select a finite fill.
- `Tnode.t` renames: `memory_mode` → `memory_mode_intent` (intent-only since the context-scoped
  `Placements` migration in 0.8) and `prec` → `storage_prec` (the settled bytes-in-buffers
  precision, as opposed to the compute precisions the numerics policy governs).
- Padding layout and neutral values are committed as tensor-node identity, so padded convolutions
  lower offset-free; incompatible later padding demands fail in shape inference.
- Reserved identifiers are derived from each backend's own syntax rather than C's spellings
  (gh-ocannl-553): MSL spells `Tanh_approx` as `tanh`, and a gelu-derived node named `tanh` used
  to shadow the function.

### Fixed

- **bfloat16 math builtins on every GPU backend** (gh-ocannl-549): no GPU dialect has bfloat math
  overloads, so builtins returned promoted `float` — rejected outright by MSL, and a
  placement-dependent compile error on CUDA/HIP; all three now bridge the result back to bf16.
- **A cast twin inherited its use site's batch axes** (gh-ocannl-540): a master-weights twin read
  as a batched matmul operand materialized per-batch-row copies of the weight — invisible to the
  loss oracle, 64x the memory, and the reason every tensorized candidate declined on HIP
  (0 → 17 timed mma candidates after the fix).
- **A declined baseline no longer ends the search** (gh-ocannl-533), and companion-coverage
  rejections are typed declines rather than `invalid_arg` fatals (gh-ocannl-521) — whose fused
  epilogue twins now compile and reach timing on Metal/CUDA/HIP (route 1 of making GPU mma seeds
  reachable).
- **Non-overlapping pooling keeps the fast gradient gate** (gh-ocannl-527): the exact
  product-space gate for overlapping windows (gh-ocannl-512) cost 1.8–2.6x where windows cannot
  overlap; `?nonoverlapping` restores the input-space gate on the domain where the two agree
  bitwise.
- Model-picked untuned defaults validate their picks at the seam and skip candidates they cannot
  build (gh-ocannl-519, gh-ocannl-522).
- Assorted codegen and lowering: multi-term affine indices parenthesized, integer `Mod` rendered
  as `%` on GPU dialects, fp8 conversion/arithmetic/uniform fixes across all backends,
  `Total_elems` counting `beg_dims`, the slab-pool table synchronized across `multidev` worker
  domains, and atomic self-healing CIFAR downloads.

## [0.8] -- 2026-07-13

> Release note: theme — parallel schedules and autotuning; AMD HIP backend. Scope changes
> vs. the original plan: **tensor cores are pushed out to v0.9**; conversely **autotuning**
> was not on the original roadmap and lands here as multi-round, execution-measured beam search
> over schedule compositions. See [ROADMAP.md](ROADMAP.md).

### Added

- **Schedule IR with automatic GPU schedules**: `Low_level` loops carry an axis type
  (`Serial | Grid | Workgroup | Workgroup_reduce | Unrolled | Vectorized`); `Grid`/`Workgroup`
  loops render as hardware index bindings with launch dimensions, barriers, shared-memory
  declarations and extent guards. Schedule transforms are values (`Split`, `Swap`, `Retype`,
  `Unroll`, `Stage`, `Privatize`, `Tensorize`, …) applied at the `Context.compile
  ?lowered_transform` seam; on cuda/hip/metal, race-free kernels get the default GPU schedule
  automatically (`automatic_gpu_schedule`), clamped and validated against per-backend
  `hardware_limits` so driver launch failures become early named errors.
- **Kernel fission** (`schedule_fission`): routines split into segment kernels at cross-workgroup
  dependency edges, each with its own launch geometry; aligned cross-nest parallelism keeps
  race-free chains fused, and Metal encodes fissioned steps into one command buffer
  (circles_conv sgd step 19.7 → 7.1 ms).
- **Within-kernel CPU parallelism** (gh-ocannl-164): the default CPU schedule retypes the
  outermost parallelizable loop to `Grid`, rendered as contiguous chunks on a process-global
  native pool (`dispatch_apply` on macOS, OpenMP elsewhere); results bitwise identical to serial.
- **Explicit SIMD codegen on the C backends** (`cc_vector_bytes`): vector-extension
  loads/arithmetic/stores with serial remainders, ggml-style vector accumulator chains
  (gh-ocannl-468), and tinyBLAS-style `Tile_mma` register tiling (gh-ocannl-469).
- **Tensor-core / GPU vector rendering**: `Tensorize` recognizes the matmul micro-kernel and emits
  a cooperative `Tile_mma` (Metal `simdgroup_matrix`; CUDA draft wmma with lane-0 fallback);
  `Workgroup_reduce` renders llm.c's two-phase shuffle reduction (gh-ocannl-462); 128-bit packed
  loads/stores on CUDA and Metal (gh-ocannl-463).
- **Autotuning**: `Autotune.tune` is a drop-in for `Context.compile` — beam search over schedule
  compositions timed on the real device, with a digest-guarded structural schedule cache and disk
  persistence, per-segment schedules for fissioned routines, matmul sketch seeding, `?timing_ctx`
  scratch lineages, and `Train.tune_placements` A/B-tuning default vs materialize-all placements.
  Tuned is the fastest OCANNL variant in every benchmark cell.
- **AMD HIP backend** (gh-ocannl-411) via the independent hipjit bindings, mirroring the CUDA
  backend (hiprtc, peer copies, bf16/fp8 types, wave32/wave64-correct shuffles), including on
  Windows.
- **Cross-framework benchmark suite** under `benchmarks/`: OCANNL vs PyTorch (eager,
  `torch.compile`) vs tinygrad (default, BEAM) on shared safetensors fixtures, every cell gated
  on loss-trajectory parity before timings count. The parity gate doubled as a correctness
  oracle: its first run caught two real backward-pass bugs (see Fixed).
- **Fused cross-entropy classifier** (gh-ocannl-464): log-sum-exp cross-entropy from raw logits
  with `stop_gradient` on the row max, accumulating `probs - targets` directly with no
  `[batch, vocab]` intermediate in either pass.
- **GPT-2 with pretrained weights**: `Safetensors` HF checkpoint reader, a GPT-2 module verified
  exact against a NumPy reference, greedy-decoding tutorial, and the dataprep BPE tokenizer
  bridge.
- Embedding backward as a guarded scatter-accumulate (`Set_dynamic`, dropping O(vocab)
  per-position work, deterministic without atomics); packed constants hoisted out of routines
  into per-device constant pools (gh-ocannl-470); on-device epoch-loss accumulation
  (`Train.grad_update ?accum_loss`); interval analysis folding interval-decided guards; a
  recompute-cost guard capping virtualized reductions (`virtualize_max_inline_reduction` —
  gpt2_mini on cc: ~13,000 → 2,361 ms/step); tensor-node ID namespaces (gh-ocannl-372 —
  `Tensor.t.id` was dropped in favor of the tnode's uid, reached through `value`).

### Changed

- **Breaking notation change**: an einsum/labels row with its kind separator omitted reads as the
  context ellipsis, not an empty row — `x` means `...|...->x`, so terse specs broadcast batch and
  input axes by default; the empty-row reading is kept by writing the separator (`| ->x`, `|x`,
  `->x`), and sum-to-scalar becomes `++ "...|... => |->0"`. In multi-operand specs where one slot
  passes batch through, the other slots must now close their batch rows explicitly (e.g. the conv
  kernel slot `; |kh, kw, ..ic.. -> ..oc..`).
- **The default backend is now `cc`**; `sync_cc` renamed to `cc`, `multicore_cc` to `multidev_cc`
  (deprecated aliases kept); `Multidev` exposes multiple worker-domain CPU devices.
- **Breaking**: backends are process-wide singletons — `fresh_backend` became `get_backend`, with
  `Backends.wrapped_context` as a closed disjunction over the singleton context types; the module
  tower cleanup retired `new_stream`.
- Memory placement decisions are context-scoped: `Tnode.memory_mode` is declared intent only,
  streamlined to five constructors (`Materialized`/`Device_only` folded away; gradients'
  `Never_virtual` replaced by an `is_observable` intent); decisions land in per-lineage
  `Placements` tables on `Low_level.optimize_ctx`, forked per compile so sibling compiles are
  hermetic (the autotuning forcing function).
- Index arithmetic is signed (`Ops.index_prec` int32, int64 under `large_models`), with
  per-node element counts checked to fit.
- Generated code and logs go to per-executable subdirectories `build_files/<exe>/` and
  `log_files/<exe>/` (override with `build_files_prefix`; `.` restores the flat legacy layout);
  step timing uses the monotonic clock (Windows' `gettimeofday` ticks at
  ~1 ms); the gated training tests moved back to `runtest` via `Autotune.tune ~rounds:0` with
  materialize-all placements (bigram on Metal 345 → 25-29 s).

### Fixed

- **Two executed-backward gradient bugs found by cross-framework loss parity**: CSE and hoisting
  treated *free* iterator symbols as renameable during alpha-equivalence, deduplicating a nested
  virtual-node recomputation into a stale local; and the scalar simplifier rewrote `(a / b) / c`
  to `(a * c) / b`. Regression test `test/training/virtual_grads_parity.ml`.
- `Nn_blocks.layer_norm` computed with a sum where it needed a mean; the fix exposed shape
  fixes in `compute_row_product` (fan_in silently 1 → wrong xavier/kaiming scales) and GLB
  closing.
- Repeated random values in inferred-shape parameter init: the PRNG counter's shape is pinned to
  the result, instead of broadcasting 50 unique values over 5000 cells.
- `Nn_blocks.conv2d` inline bias is per-channel, not broadcast to the full feature map.
- Forward and backward fragment ordering is topological, not tensor-id order (GPT-2's attention v
  projection silently read zeros); fragment cycles error (gh-ocannl-461).
- Worked around an Apple Metal shader-compiler miscompile of serial accumulation at a
  loop-invariant address (volatile shadow pointer; standalone repro checked in); Metal launches
  are ordered after all previously enqueued work; Metal default-schedule pathology on gpt2_mini
  fixed (81 s → 0.3 s steps — the `gpu_schedule_min_parallel` default is lowered to 64, so
  kernels that used to stay on the serial 1x1 fallback now get the automatic GPU schedule).
- Windows: the CUDA backend restored (NVRTC arch floors for half/bf16 intrinsics), test suite
  green on cc and hip.
- 1-element `Reshape`/`rebatch` data no longer collapses to a scalar (gh-ocannl-460).

## [0.7] -- 2026-07-03

> Release note: **0.6.4 is skipped** as a tagged release (last release before 0.7 was 0.6.3). The
> work originally planned for 0.6.4/0.6.5/0.7.0 (frontend finalization, concatenation,
> position embeddings, transformer toy) and for 0.7.2 (compiler optimizations, pool
> allocator) is **consolidated into 0.7**. See [ROADMAP.md](ROADMAP.md).

### Added

- **Removed the hosted tensor mode** (gh-ocannl-333): dropped the `array` field of `Tnode.t`;
  tensor value access and printing are context-mediated, host-init nodes self-initialize at link
  time. Removed with it: the `automatic_host_transfers` setting and the `use_host_memory` hook.
- **Axis concatenation / block tensors in einsum notation** (`a^b`): concatenation
  (`a; b => a^b`), slicing (`a^b => a`), n-ary block-tensor specs, the `++^` operator, and concat
  projection unification in shape inference.
- Ternary einsum notation (`einsum3`, where-with-spec, a `Mul3` scalar op across backends)
  (gh-ocannl-305); pointwise operations optionally accept einsum/permute specs via
  `?spec`/`?capture_dims`.
- **Universal pool allocator across backends** (gh-ocannl-344): tensors addressed by
  `{ pool_id; offset }` — working tensors bump-packed into per-context-delta pools, constants
  into per-device pools, merge buffers reserved; Metal binds pool slabs plus a slot table,
  staying under its binding limit for large routines.
- Loop-invariant code motion (gh-ocannl-350) and common subexpression elimination after inlining
  (gh-ocannl-351); virtual-node inlining extended to non-scalar constants and ranges
  (gh-ocannl-142).
- Data-parallel training: `shard_along` / `gather` primitives with merge-buffer gradient
  all-reduce, and **zero-copy leading-axis slice views** (`@|` / `Fetch.Slice`) — writing through
  an alias-eligible slice now mutates the parent; ineligible slices fall back to copies; and
  host-side access (`Context.get_values` / `set_values`) of an alias view is rejected with a
  clear error — read or write the parent tensor instead (gh-ocannl-293).
- Tensor saving, loading, and restoring (gh-ocannl-373); `Uint32`/`Uint64` precisions with
  index-embedding precision selected by `large_models` (formerly `big_models`, still accepted as a
  deprecated alias) (gh-ocannl-349, gh-ocannl-177).
- Makemore-progression examples and tutorial: `mlp_names` (Bengio MLP), `mlp_bn_names`
  (+ `batch_norm1d`), `docs/makemore_tutorial.md`; Sasha Rush Tensor Puzzles in extended einsum
  notation (gh-ocannl-308).
- **Paper artifacts**: the workshop article (Markdown, LaTeX, rendered PDF), the standalone
  formal core technical report covering the shape/projection inference proof effort, and
  `docs/shape-constraint-generation.md`, with regression coverage for the formalization's
  counterexamples.

### Changed

- **Breaking**: `Backend.device_to_device` returns `context routine option` instead of scheduling
  a side-effect copy — callers run the transfer routine, and linking a merge-buffer consumer
  against its context verifies the node at link time (gh-ocannl-288).
- Default `%op` parameter initialization is a centered, scaled `uniform1` over `[-0.25, 0.25)`.
- Moved `datasets/` to the separate `dataprep` package; relaxed the required `ocannl_` prefix on
  commandline arguments with config-key validation, and renamed `ocannl_config.example` to
  `ocannl_config.reference` (gh-ocannl-409).
- Extended the identifier blacklist with C keywords and backend-reserved words (gh-ocannl-383);
  renamed kernel parameters to `kparam`/`kparams` (gh-ocannl-356); removed remaining unnecessary
  buffer zeroing (gh-ocannl-382).
- Heavy training integration tests gated behind Dune's `slow` alias.

### Fixed

- Detect rank cycles among row variables during shape inference (gh-ocannl-247).
- Concat lowering and projection fixes: cumulative offsets, unit dims, `d=1` components,
  `invalid_vars` quantifier logic and dimension-0 guessing.
- CUDA `Where` ternaries parenthesized; `Uint32`/`Uint64` to `uint4x32` PRNG-counter conversions
  spread bits rather than collapsing entropy.
- Prohibit `~logic:"@"` with `/` and `**` in `%cd`, and ternary `~logic` mapped to the right
  projection type (gh-ocannl-192); C-syntax tracing `printf` line breaks (gh-ocannl-179).

## [0.6.3] -- 2025-12-19

### Added

- Neutral element tracking during shape inference for proper padding reset
- `use_padding` syntax in einsum notation (replacing global flag)
- Circle counting dataset and MLP training test
- Cross-entropy loss and `one_hot_of_int_list` helper for classification tasks
- `out_channels` parameter to `conv2d` for explicit channel specification
- Projection slot detection by naming convention in `%cd` syntax extension
- Configurable scaling to `kaiming` and `xavier` initialization functions
- New documentation: `tensors_and_contexts.md`, affine indexing for convolutions
- Documentation for `op_fun` and `param_op_fun` types, roots, embedded nodes, and params concepts

### Changed

- Padding is now reset by tracking neutral elements through shape inference
- Changed default random initialization to `uniform1`, which doesn't impose shape constraints
- Refactored `vbs` from Map to list for order-preserving let bindings in syntax extensions
- Infer the shape of inline definitions assigned a slot for `%cd` expressions with `projections` in scope

### Fixed

- Gracefully disable inlining for convolution patterns
- Don't propagate padding across operations, even if the same tensor participates in them
- Padding margin initialization for tensors with multiple operations
- Padding initialization bug for max-pool operations
- `uniform1` periodicity by spreading bits in `*_to_uint4x32` conversions
- `tropical` (max-reduce) backprop to use input-shaped condition tensors
- `tropical` g2 gradient by using correct projection for kernel gradients
- Kernel extent calculation to depend on kernel size parity
- Shape inference for `Total_elems` constraints with `Strided_var` numerators
- `compute_row_product` to return `None` for unresolved variables
- Deferred dim variable guessing to Stage 5 for `Total_elems` propagation
- Padding offset application during lowering for correct buffer indexing
- Intermediate grads from `kaiming`, `xavier` appearing in `zero_grads`
- Random seed initialization missing in transformer test

## [0.6.2] -- 2025-11-27

### Added

- Normal distribution random number generation
- `%%extend_dsls` syntax extension for extending DSL modules
- `interleave` operation in DSL modules
- `Defined_by_cd_logic` shape inference specification for explicit shape logic in forward code
- Menhir-based einsum parser replacing Angstrom for better maintainability
- Name clash detection for inline definitions and variable captures in syntax extensions
- `is_param` flag in shape inference for improved parameter-related error messages
- Teacher forcing support in transformer implementation
- Heuristics for "missing hidden dimensions" error messages with row variables
- `Tree_map` persistent map utility with exposed tree structure in sexp serialization

### Changed

- Migrated shape environment to use `Utils.Tree_map` for ppx_minidebug v3 full-scale debugging
- Replaced explicit non-iteration tracking with improved projection constraints derivation
- Support for offset-only affine expressions in shape inference
- Renamed optional dimension variable parameter from `label` to `name`
- Row IDs replaced with provenance tracking (`Row.id` → `Row.prov`) supporting deduplication
- Tensor labels interface improved: per-operation `op_label` string with `label` list as trailing parameter
- Adapted to ppx_minidebug renaming (`entry_id` → `scope_id`)
- Prefixed block names in `lib/nn_blocks.ml` for better namespace management
- Tests reorganized: more einsum-related tests moved to `test/einsum/`

### Fixed

- Normal distribution test determinism across different machines
- Convolution/affine indexing shape inference offset adjustment by strides
- Parameter gradients not embedded after params moved earlier in processing
- Einsum parser handling of missing convolution and single-character cases
- Shape inference for `Conv_input` additional cases
- Incremental construction of tensors in `Tensor.op`
- Attention masks now have empty output dimensions for proper broadcasting to multihead attentions
- LUB (Least Upper Bound) computation in `dim_ineq`
- Axis labels distinguished from dimension units (labels) in `shape_spec_to_dims_bio`
- Shape inference for dim-1 with labels treated same as dim>1 (only dim-1 without label is different)
- Shape specification requiring LUB incorporation for non-terminal shapes
- Missing CUDA backend cases and NVRTC compatibility
- Premature guessing of dim variables as dim-1 when participating in `Total_elems` constraints
- Generic constraints ignored for unused tensors
- Missing propagation when `set_dim` happened before parsing the spec
- Guard `axis_keys_to_idcs` from un-inferred shapes
- More informative error messages for parameter shape errors
- Crash on repeated variable capture in syntax extensions
- Additional syntax support for binary einsum operators

## [0.6.1] -- 2025-09-12

### Added

- Record-based syntax for inline tensor definitions in `%op` and `%cd` expressions
- `uniform1` variants for non-vectorized random number generation (`Uint4x32_to_prec_uniform1`)
- Support for uint32/uint64 precisions and `big_models` flag for indexing arithmetic
- Created docs landing page with automatic publishing action
- Group Relative Policy Optimization (GRPO) documentation in RL slides
- Counter-based randomness with lightweight (2-round) Threefry variant as default
- Claude GitHub Actions for automated code review and PR assistance
- Heterogeneous precision support for primitive operations
- Both zero-initialized and undefined-initialization buffer creation options
- More output options for `ocannl_read_config` utility
- Added comprehensive RL/REINFORCE tutorial slides with concrete examples
- Added clear explanations of slipshow navigation semantics to CLAUDE.md
- Transformer architecture support with multi-head attention, layer normalization, and positional encodings
- CNN building blocks: conv2d, pooling operations (max/avg), and comprehensive migration guide
- Context API as simplified backend interface replacing stream-based parallelism
- Shape constraint provenance tracking for dramatically improved error messages with origins
- Dimension capture and equality constraints in einsum specifications via `set_dim` and `set_equal`
- New einsum operations: `einmax1` (unary max-reduce) and `tropical` (max-reduce with add)
- `%oc` anti-quotation syntax for improved OCaml integration in ppx extensions
- Tensor initialization operation with configurable strategies
- `offsets` convenience operation for index generation
- Comprehensive migration guide for PyTorch/TensorFlow users
- Shapes and einsum tutorial slides with slipshow presentation format
- Configurable limit on shape constraint provenance tracking
- Origin tracking in shape error messages

### Changed

- Major fix to tensor initialization handling with uniform generation across TDSL, NTDSL, PDSL
- `%op` scope delimiting changed from `~config` to unit parameters for cleaner syntax
- Renamed `zero_initialized` to `zero_initialized_by_code` for clarity
- Split Threefry4x32 into crypto (20-round) and light (2-round) variants
- Changed `Operation.range` semantics to match Python's `range` function
- Improved precision handling in low-level operations with bidirectional inference
- Enhanced record syntax with field shortcuts (`o` → `output_dims`, `i` → `input_dims`, `b` → `batch_dims`)
- More precise and thus more lenient rootness checks in `Tensor.consume_` functions
- Generalized `guess_output_nodes` to `collect_nodes_guess_output` for more reuse
- Converted documentation slides to use slipshow for better updatability and navigation
- Updated CLAUDE.md with record syntax documentation and testing guidelines
- Major reorganization: moved tensor-related modules to dedicated `tensor/` directory
- Renamed einsum operator `*+` to `+*` for better consistency
- Refactored DSL modules into `Operation.DSL_modules` for cleaner API
- Removed stream-based parallelism in favor of simpler Context API
- Improved `%op` and `%cd` syntax extensions with better function application handling
- Enhanced shape inference with proper dimension staging (no closing at stage 2)
- Migrated documentation to `docs/` directory with pandoc rendering support
- Improved `.cd` file generation with clearer rendering of special operations
- Updated ppx_minidebug integration with log pruning for better performance

### Fixed

- Unnecessary dune rules triggering issue
- Precision handling for `Uint4x32_to_prec_uniform1` in scalar computations
- CUDA, Metal, and C backend fixes for various precision and initialization issues
- Zero-dimensional Bigarray indexing
- Test dependency on `OCANNL_BACKEND` environment variable
- Build setup for `ocannl_read_config` utility needed for tests
- Missing package dependencies and assignments in dune configuration
- Critical transformer bugs: mask handling, attention dimension specifications, position encodings
- C backend INFINITY macro usage (was using invalid inf literals)
- Shape inference bugs with dimension closing and constraint generation
- Dropout pseudo-random number splitting
- Layer normalization implementation in `nn_blocks.ml`
- Dimension inference for attention layers with hidden dimensions
- Pooling operations projection inference
- Division simplification for integer precision
- Various syntax extension edge cases and error handling

## [0.6.0] -- 2025-08-19

### Added

- Support for Brain float aka. bfloat16 aka. BF16, and for FP8.
- Support for convolution via affine indexing expressions in: projections, einsum notation, shape inference.
- MNIST and CIFAR10 datasets (borrowed from Raven).
- Names dataset with bigram use-case helper.
- Half-moons synthetic dataset.
- New precision `Uint4x32` that piggybacks on the `Complex.t` type for the `Bigarray` backing.
- New precision `Int64` for integer operations.
- New operation `Threefry4x32`, which is unusually and hopefully uniquely coarse-grained (requiring nontrivial implementation code for each backend that should conform to a common algorithm).
  - This way we avoid introducing multiple operations on bits.
- Support of counter-based randomness via the `Threefry4x32` operation and random seed tracking.
  - The cascade of splits uses the Tnode id, the train step and the tensor cell position.
- Added a new operation `Uint4x32_to_prec_uniform` that converts the 128-bit random values to floating point uniform distributions efficiently.
- Vector operations support with `Set_from_vec` in low-level IR for efficient vectorized assignments.
- Added a field `params` to `Tensor.t` since we need to track parameters to properly initialize computations (see below).
- `Embed_self_id` operation for positional embeddings.
- Bidirectional precision inference (both top-down and bottom-up).
- Enhanced `%cd` syntax with support for `.forward`, `.backprop`, `.zero_grads` and automatic comment generation.
- Inline tensor declarations in `%cd` syntax for standalone expressions.
- `Train.init_params` for streamlined parameter initialization.
- Better configurability with `inline_complex_computations` setting.

### Changed

- Removed the ndarray initialization logic. Some of its functionality is now incorporated into `fetch_op`.
- Refactored `init_op` and the badly named `global_identifier` from `ops.ml` into `dedicated_access` in `low_level.ml` and a bigger `fetch_op` in `assignments.ml` (more meaningful file locations).
  - Also renamed the badly named `Get_global` to `Access`.
- Initialization now needs to be handled via running the corresponding code explicitly. In particular `Tensor.init_params` will run the forward code of tensors from the `params` field.
- Virtual nodes and inlining now also work across routines. This required changing the API to pass the `optimize_ctx` optimization context.
- Made ppx_minidebug logging per-file opt-in at compile time for better control.
- Refactored Tensor API to reduce boilerplate and share parameter signatures.
- Renamed `float_t` to `scalar_t` throughout the codebase for consistency.
- Migrated from heap-local allocation to on-stack allocation by default.
- Improved shape inference with better Total_elems constraint handling and LUB (Least Upper Bound) support.
- Enhanced projections inference with better slot selection heuristics.
- More defensive handling of empty dimensions and zero-dimension scalars.

### Fixed

- Memory leak in builtins.c.
- Context handling for constants initialized on devices.
- Zero-initialization that wasn't being performed on Linux (MacOS zero-initializes by default).
- Surjectivity and bijectivity checking in indexing operations.
- CUDA backend regressions and missing constructs.
- Duplicate Shape_rows constraints elimination.
- Precision inference issues with premature forcing.
- Bus error on large datasets.
- Session-level bugs that appeared only in specific backends.
- Identifier generation to not start with digits.
- Host-device synchronization issues with `devices_not_lagging_host` semantics.
- Shape inference corner cases with Total_elems and row constraints.
- Various issues with convolution and strided iteration support.
- Moved away from using statically loaded builtins.c from routines (kernels), all backends now prepend their builtins textually.
- Emulating _Float16 aka. half on systems with C compilers that don't support it.

## [0.5.3] -- 2025-05-24

### Added

- The Metal framework backend (Apple Silicon).
- Setting `debug_log_to_stream_files` to neatly keep logs from routine execution in their separate files.
- Settings `clean_up_artifacts_on_startup`, `prefer_backend_uniformity`.
- Tools directory and the `minised` tool: regexp replacement file rewrite.
- Directory arrayjit/bin and executable `read_config` for extracting OCANNL configuration into txt files.

### Changed

- Removed `initialize` and `is_initialized` from the backend API; instead, backends should be initialized on functor application. The functors now take `config` as argument.
- More descriptive identifier names in C-syntax code in case of name conflicts.
- Changed the backend config name `cc` to `multicore_cc` for consistency.
- Migrated out of `Stdlib.Format` to `PPrint` for all structured formatting.
- Migrated stdout capture to thread-based (domain-based actually); for Windows compatibility but also much more robust for large logs.

### Fixed

- Avoid conflicts with C math function names like `fma`.
- Satur01_gate had wrong semantics.

## [0.5.2] -- 2025-04-07

### Added

- Lots of new primitive ops:
  - Unary: Satur01 | Exp | Log | Exp2 | Log2 | Sin | Cos | Sqrt | Recip | Recip_sqrt | Neg | Tanh_approx | Not
  - Binary: Satur01_gate | Max | Min | Mod | Cmplt | Cmpeq | Cmpne
  - Ternary: Where | FMA (non-accumulating)
- Ternary tensor operations.
  - A differentiable `where` operation.
- More flexible gradient construction via the `%cd` syntax (better projections inference).
- CC backend piggy-backing on OCaml's C compiler (consistent across OSes).

### Changed

- Updated to printbox 0.12, with upstreamed graphing.
- `-pthread` -> `-lpthread` in `c_library_flags` in `dune` files.
- Removed Numpy support for easier compatibility on native Windows.
- Unary (primitive) ops and relu are now named, not operator syntax.
- Refactored `%cd` parsing of primitive ops.
- `%cd` and `%op` support both curried and uncurried operator application syntax.
- Updated to ppx_minidebug 2.2.0 with support for cross-run diffing.

### Fixed

- Numbers text rendering (consistent across OSes).
- Moved closing row variables to stage 3, because stage 2 may need to process inequalities generating more LUBs.
- Don't unnecessarily prevent bytecode-only build targets.

## [0.5.1] -- 2025-01-01

## Added

- Automatic transfers to host from the context that most recently updated a node.
- Automatic transfers of routine's inputs from host to routine's context if the host array modification was not yet transfered.

## Fixed

- Added `#` as alternative to `~~` for comment lines in `ocannl_config` files, and fixed a bug in their parsing.

## [0.5.0] -- 2024-12-18

### Added

- Interface files for `Backends` and `Low_level`.
- Fixed #245: tracking of used memory. But there's room for improvement.
- Stream-to-stream synchronization functionality, with lazy per-tensor-node synchronization.

### Changed

- Migrated to cudajit 0.6.1.
- Verifying that code is linked with the right contexts, by tracking `embedded_nodes` with assignments.
- Renaming: (virtual) `device` -> `stream`, `physical_device` -> `device`.
- New files: split out `backend_intf.ml`, `backend_impl.ml`, `schedulers.ml` from `backends.ml`; moved `Tnode.task` to `task.ml`; renamed `backend_utils.ml` to `c_syntax.ml`.
- Removed half-static verification of merge buffer nodes inside `device_to_device`.
- Fixed #286: cross-stream-sharing incorporated into `Tnode.memory_mode`.
- Moved the multicore backend from a `device = stream` model to a single device model.
- Got rid of `unsafe_cleanup`.
- Rename `subordinal` to `stream_id`.
- Removed dependency on `core`, broke up dependency on `ppx_jane`.
- Huge refactoring of backend internal interfaces and API (not repeating same code).
- Built per-tensor-node stream-to-stream synchronization into copying functions.
- Re-introduced whole-device blocking synchronization, which now is just a slight optimization as it also cleans up event book-keeping.
- Simplifications: no more explicit compilation postponing; no more hard-coded pointers (all non-local arrays are passed by parameter).
- Fresh backends are now fresh modules to structurally prevent any potential cache leaking.

### Fixed

- Validating merge nodes for the CUDA backend.
- Checking `is_released` on weak array retrieval.

## [0.4.1] -- 2024-09-17

### Added

- Implemented the previously-mocked support for half precision (FP16).
  - We work around the missing Ctypes coverage by not using `Ctypes.bigarray_start`.
  - We check FP16 constants for overflow.
  - We output half precision specific code from the CUDA backend.
- Finally proper support for mixed precision! Lazy precision defaults and delayed precision setting via `Tnode.update_prec`.
- A placeholder `nn_blocks.ml` hinting at an intended design pattern for model components.
- A memory model for the multiple virtual devices per physical device setup, implemented in the CUDA backend. It fixes the CUDA backend behavior in the data parallelism benchmark.
- Slides for the Fun OCaml meetup: [docs/Fun OCaml](docs/OCANNL-slides-basics_backprop_training_loop_codegen.pdf).
- New syntax: inline tensor declarations with a literal float as initial value.

### Changed

- Removed the `pipes_cc, pipes_gccjit` backends (`Pipes_multicore_backend`) -- I had fixed `Pipes_multicore_backend` by using the `poll` library instead of `Unix.select`, but it turns out to be very very slow.
- Changed the `%cd` block comment syntax `~~` to allow detailed structuring. Rewrote `Train.grad_update` to use the `%cd` syntax.
- Made `Train.sgd_one` slightly more thrifty: `p =- learning_rate *. sgd_delta` --> `p =- learning_rate * sgd_delta ~logic:"."` without the inline tensor expression.

### Fixed

- Log levels related de-confusion:
  - Critical bug: logging of computation traces was not properly converted to ppx_minidebug 2.0.
  - Properly restore `log_level` and inform about its setting.
  - By default do not log from tests.
  - `debug_log_from_routines` should only happen when `log_level > 1`.
- Bugs in `Multicore_backend`: `await` was not checking queue emptiness, `worker`'s `Condition.broadcast` was non-atomically guarded (doesn't need to be), possible deadloop due to the lockfree queue -- now replaced with `saturn_lockfree`.
- Reduced busy-waiting inside `c_compile_and_load`, propagating compilation errors now instead of infinite loop on error.
- Fixed loss of significant digits for small numbers when outputting files.
- Added missing mixed-precision conversions in the `C_syntax` backend builder.
- Restored the functionality of debug logging from the cuda backend.
- Always reinitialize global state at the beginning of `let%expect_test`, to make them more deterministic.

## [0.4.0] -- 2024-09-04

### Added

- A new backend "cc": C based on a configurable C compiler command, defaulting to `cc`.
- Merge buffers representational abstraction (one per virtual device):
  - backends just need to support device-to-device transfers,
  - merging gets implemented in "user space".
- CUDA streaming multiprocessor parallelism via streams <-> virtual devices.
- Support for `cuda-gdb` and `compute-sanitizer` (pass the right arguments to cudajit).
- Inline declarations for (non-differentiable) tensors in the `%cd` syntax.
- A minimal wrapper `Sync_backend` creating CPU backends with a single device only, where all calls are synchronous. (It's a baseline and helps debugging.)
- In progress: proper (condition variables based) scheduler. The legacy scheduler (pipes based) kept for now as baseline and to help debugging.
- Documentation for the syntax extensions.
- `%op` syntax: when under a `~config` parameter, refine the inline declared params' labels with `config.label`.
- `%op` syntax: incorporate the input tensor's (if any) label in the resulting tensor's label.
- Comments in config files using the line prefix `~~`.

### Changed

- Terminology in the API: Renamed almost all uses of "jit" into uses of "compile" and / or "link".
- Split the compile-to-ptx phase from the build-module and build-kernel-launcher phase.
- Migrated the CUDA backend to ppx_minidebug-based execution tracing.
- Fixes for mixed precision computations.
- Further terminology refactoring: Renamed `Low_level.compile` to `Low_level.lower`;
  - and `Low_level.compiled` to `Low_level.optimized`, making it a record.
- Further refactoring of the `Backends` API:
  - split the `device` type into virtual `device` and `physical_device`,
  - removed the direct support for `merge`, instead relying on merge buffers.
- Updated to cudajit 0.4.
- A template for C-syntax backends, refactoring CC and CUDA backends.
- Improvements to handling of tensor node labels, and to the `Tnode.debug_name` function.
- Output files generated by backends, and files generated by logging, in separate subdirectories.
- C-syntax logging: also output the pre-assignment value when logging an assignment.
- Migrated to ppx_minidebug 2.0 with the benefits it brings: no runtime passing, `Utils.settings.log_level` unified with ppx_minidebug's log levels.

### Fixed

- Allow verifying that non-embedded tensor nodes of the tensor(s) associated with a linked code are already in the context passed to `link` (resp. `link_batch`), since they won't get introduced into the context. It is the responsibility of helper functions (such as those in `Train`) to ensure the check.
- Fixed both known and newly discovered shortcomings of the syntax extensions.
- In particular, `%op` syntax: lift `~config` applications out of (tensor) functions.
- Multiple other tiny fixes.

## [0.3.3] -- 2024-04-24

### Added

- GitHub workflow for continuous integration and API docs.
- Randomness plug-ins via global config `randomness_lib`: currently only `stdlib` and `for_tests`.

### Fixed

- A bit of code rot in the Cuda backend mock `cuda_backend.missing.ml`.
- NPY: Compatibility with OCaml 5.2.0.
- Renamed the main package name from `ocannl` to `neural_nets_lib`, to prevent the opam linter from complaining about a confusing name.

## [0.3.2] -- 2024-04-22

### Added

- `let%cd _ =` (and `let%op _ =`?) do not affect root tracking (intended for adding shape constraints).
- More expressive shape constraints: allowing row variables to be sandwiched between leftmost axes `beg_dims` and rightmost axes `dims`.
- Einsum notation support for leftmost axes.

### Changed

- Cleaned up "user-facing" API by moving `IDX` and `CDSL` to `Train`, and `Tensor.O` to more precise `Operation.At`.
- Added interface `Tensor.mli` to reduce "the user learning surface".
- Improved documentation and layout of `Shape.mli`.
- A more reasonable syntax for labels specifications and einsum notation. In particular, whitespace insensitive (except whitespace not allowed inside identifiers).
- Vendored the `npy` package while we wait for a PR.

### Fixed

- Moved `cudajit` to `depopts`.
- Slice shape inference is now complete, by using leftmost axes `beg_dims` in constraints.

## [0.3.1] -- 2024-04-15

### Added

- Tensor parameters saving and restoring, Ndarray saving and restoring.
- An operation `outer_sum`: like `einsum` but simpler, addition everywhere.

### Changed

- Tweaks to make the project usable as a package (external library).
- Sanitizing code inclusion via code roots management: `Tensor.consume_forward_code` and `consume_backprop_code`, (optionally but by default) used from `Train`.

### Fixed

- Shape inference in presence of non-0 fixed indexing inside einsums was broken (because actually not implemented).
- Incompleteness of shape inference for slicing was leading to inferring shapes with no axes: constraint generation was intended to raise a shape error instead. Proper fix coming in 0.3.2 will make slice shape inference complete.

## [0.3.0] -- 2024-03-31

Major rewrite. Abandoning the design choices of 0.1 and 0.2.

### Added

- Optionally, inferring or checking tensor (batch) sizes from data (e.g. file) sizes.
- Static indexing. A "slice" operator to select individual batches.
- Established the backends API with first-class modules.
- The `Train` module as an optimization "frontend".
- Parallel optimization across devices.
- Global settings configurable via config files, environment variables, and commandline flags.
- Integration of backend logging with `ppx_minidebug` (the `debug_log_from_routines` setting).

### Changed

- The Cuda backend is not supported for now. It is (optionally) buildable to reduce code rot.
- Dynamic indexing is not supported anymore (to reduce complexity). It might be reintroduced if needed.
- Factored out the `arrayjit` library / package containing compilation (former Ndarray, Node, Code).
- Renamed `Formula` -> `Tensor`
- No more "form vs. non-form" formulas / tensors.
  - Formula/tensor roots are split into forward roots and backprop roots.
- No more `%nn_rs`, `%nn_dt` syntaxes and `Synthetic` fetch primitive.
- Renamed `%nn_op` to `%op` and `%nn_cd` to `%cd`.
- Migrated `gccjit` into a separate repository.
- Migrated `cudajit` into a separate repository.
- Massive rewrite of shape inference in a declarative style.
- Generalize `zero_out` to `initialize_neutral` to prepare arbitrary accumulation operation.
- Renamed `Node` -> `Lazy_array` -> `Tnode` (tensor node).

## [0.2.1] -- 2023-07-19

### Added

- The Cuda backend.
  - The Cudajit interface based on Nvrtc and the Cuda driver API.
  - A naive `Exec_as_cuda` backend where the dedicated `Task_id` axis parallelizes over blocks, and a new dedicated `Sample_num` axis parallelizes over threads in a block.
  - When outputting debug files, stores the source `.cu` code and the assembly `.ptx` code.
  - Supports thread-only tensors, tensors with thread-local "replicated" working copies, constant tensors, and globally updated tensors.
  - The backend uses atomic adds for shared updates, and within-block synchronization to minimize update races and parameter staleness.
  - Debugging: full trace (for thread 0) by logging assignments with the assigned value and indices for the LHS tensor and the RHS tensors, the expression used to compute the assigned value, of values of subexpressions.
- Cuda FFI for retrieving GPU specs and for getting and setting limits.
- `Zero_out` low-level-code primitive using `memset`.
- `Staged_compilation` low-level-code primitive: a (stateful) callback for use by backends.
- When outputting debug files, also stores the high-level code.
- Saving and restoring tensor content to `.npz` (`.npy` archive) files (untested).
- Low-level code based optimizations:
  - unrolls `ToPowOf` with integer exponent,
  - simplifies local computations that are just expressions,
  - some arithmetic simplifications.

### Changed

- Monomorphic `axis_index`, simplified the axes-related types.
- Splits `'a low_level` into monomorphic `unit_low_level` and `float_low_level`.
- Removes integer bigarray types.
- Refactors `Node` + `NodeUI` into `Ndarray` + `Node`.
- Tensor printouts include whether a tensor contains `NaN` or `infinity`.
- Simplifies the `Task_id` functionality: removes `If_task_id_is` and `Global Task_id`; emoves parallelism from `interpret_code`; removes `task_id_func` vs `unit_func` duplication.

### Fixed

- "Non-diff" code inclusion.
- Ensures unique indices/symbols also for the `task_id` and `sample_num` bindings.
- Removes endlines from `PrintBox_utils` benchmark tables cells.

## [0.2.0] -- 2023-06-03

### Added

- The Gccjit backend operates using "on device" copies of tensors, where the "device memory" is the stack of the C function. This is intended to improve cache locality and reduce cache contention.
  - Three / four synchronization heuristics:
    - "parallel": a slice of the tensor is copied host-to-device at the beginning and device-to-host at the end, without interference because each task has a different slice.
    - "update on host": the tensor is copied host-to-device at the beginning; each write is an update, it reads the old value from host to update it on the host. Thus each write is a synchronization point.
    - "replicated": the tensor is copied host-to-device at the beginning; only task 0 copies device-to-host.
    - "device-only": no copying to/from host.
- On-device-only tensors that are not materialized on the OCaml side.
- A new category of axis dimensions is introduced: `Frozen`. It is analogous to the `Parallel` axis category in that a single task execution / "device call" only processes a 1D slice of the axis.
  - Currently, for tensors processed in parallel, we only support processing of a contiguous tensor slice (copied "to device" using `memcpy`).
- A new syntax `%nn_rs` ("postprocess results" variant of `%nn_dt`) for computations that should happen at the end of task execution / refresh step. It's meant to prepare the data to be copied back to the host.

### Changed

- Got rid of backend-agnostic synchronization. It was not worth the complexity / implementation effort at this point.
  - Keeping the `Rebalance` constructor around, but it is not playing any role.
- Got rid of `debug_virtual_nodes`, was tricky to maintain.
- Dynamic indexing now skips over parallel axes: when there is a `Parallel` axis on the left, it is preserved in the resulting tensor (slice), and the next-right axis is indexed into instead.
  - Removed the "indexing axes from-right" functionality for now (fails as not implemented).
- Dynamic indexing now can produce virtual nodes.

### Fixed

- Dynamic indexing fixes.

## [0.1.2] -- 2023-05-12

### Added

- Thread-local parameter `task_id` for automated iteration over a dimension `Parallel`.
  - This implements multicore SGD.
  - Rebalancing of computations that don't use `Parallel`, and synchronization in the `Gccjit` backend, are left as future work.
  - Already provides significant speedups in the interpreter (6-7x for me), but that's a moot point.
  - Giving up further work this approach for now, because the bottleneck is the memory access with `Gccjit`.
  - Keeping the new representation capability around, maybe it will be a stepping stone to other things.
- Monolithic step update with "macrobatch" (multiple steps within one backend call).

### Changed

- Streamlined the source code, e.g. removed the `OCaml` backend.
- Better syntax for `%nn_dt` and `%nn_op` shape specification, allows identifiers.
- Improved virtual node and scalar constant inlining.
- Better debugging, e.g. an option to "trace" `Gccjit` execution by printing the comments.

## [0.1.1] -- 2023-05-06

### Added

- An _inline constants_ optimization that compile-time computes scalar constant subexpressions and inlines the values.

### Changed

- Improved debuggability.

### Fixed

- A last-minute breaking bug (would be nice to have a pre-release or a pre-publish hook to run tests!).
- The virtual nodes optimization is more robust, correct even with aggressive inlining settings (e.g. escaping variables check).

## [0.1.0] -- 2023-05-04

### Added

- The first changes-tracking release. Earlier development history is still somewhat documented via closed issues.
- Supports single and double precision floats, more precisions in the future.
- Generates a monolithic step update routine executed by `refresh_session ()`, but can generate arbitrary additional routines at arbitrary times to be executed at arbitrary other times within a session.
- An `Interpreter` backend that can for example log all individual tensor modifications.
- A `Gccjit` backend that can sometimes be 400x faster than the `Interpreter` backend (without any debug work/output).
- A _virtual nodes (tensors)_ optimization that inlines computation of a cell in lieu of tensor accesses, can sometimes reduce memory consumption by 1/3.

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
- benchmarks/ is the cross-framework parity+timing suite (self-describing safetensors fixtures,
  one-JSON-line runners, loss-trajectory parity gate ~1e-7 fp32 vs pytorch/cpu). The gate
  doubles as a gradient oracle. tinygrad: realize the loss BEFORE `opt.step()` or it recomputes
  from updated weights. The autotune schedule cache persists across processes; compiler changes
  invalidate it by digest.
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
- That digest covers the compiled result, NOT the diagnostics. When re-measuring a search's
  decline census after changing only an error message or a log line, `rm benchmarks/autotune_cache/*.sexp`
  first — otherwise the second run reports `state=cache-replay`, `timed=0`, `declines=[]` and every
  counter reads zero, which looks exactly like "the problem went away". Fixtures are also not in a
  fresh checkout (`benchmarks/fixtures/*.safetensors` is gitignored and generated), and they are
  only valid while `gen_fixtures.py`, `benchmarks/workloads/` **and the generating numpy** are
  unchanged.
- **Do not run `gen_fixtures.py` to get past the digest gate.** It is the reflex the refusal
  message used to invite, and it is wrong: regenerating draws a NEW workload from this box's
  numpy, which retires every published number on the old bytes and does nothing for the other
  measuring boxes. If your copies are merely unrecorded, `python3 benchmarks/fixture_digest.py
  --record` pins them as they are (stdlib-only, no venv, and it leaves other origins alone);
  `--check` reports disk against record. Regeneration is a cross-box event to be coordinated
  across every origin in `DIGESTS.txt`'s `# measurement-boxes:` header field at once
  (gh-ocannl-759, gh-ocannl-850). That list is independent of the entry rows, so
  `divergent_origins` and generated reports name a declared box even when its fixture entry is
  absent; deriving the box set from rows would recreate the silence the field exists to remove.
- The boxes are **not** on the same fixture bytes, and the digest file says so per origin
  (`<sha256>  <bytes>  <name>  <origin>`, gh-ocannl-759). `mlp_small` and `gpt2_mini` hash
  differently on minix and rog-nv at identical sizes — two venvs, two numpy `Generator` streams,
  one spec — so `report-hip.md` and `report-gh675-cuda.md` are **not cross-box comparable for
  those two workloads**; each is self-consistent within its box, so within-box session-to-session
  comparisons stand. A fixture MATCHes if it is *some* recorded box's bytes, and every row
  (`fixture_origin`) and report section names whose. Before quoting one box's number against
  another's, check that the two sections name the same origin.
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
  DIVERGED verdict. The general trap: `json.loads` accepts `NaN` and `Infinity`, so the Python
  runners' cells survived while OCANNL's vanished, and the report read as an OCANNL runner bug
  rather than a numerics one — when a sweep loses only one framework's rows, suspect the emitter's
  spelling before the framework.

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
  `cell_group.py` now gives every child spawned by `orchestrate.py` and
  `gh675_cells.py` one shared group/job, TERM-to-KILL, output-preserving reap discipline;
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
  (gh-ocannl-575). `try_register_tile` covers `n - (n mod bw)` columns and peels the rest to scalar
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

# Backend memory, allocation and limits

Pools, release, graph capture, hardware resource budgets, and the footprint planner.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- The resource-seam acceptance map is `docs/resource-fault-injection-inventory.md` (gh-ocannl-571).
  A release change is not accepted by a flat memory curve: inject after allocation, before and
  during cleanup, and at persistence commit/replay; pair every failure with an uninjected control
  and assert exact ownership (one free, no rooted table entry, retry state honest).

- **Device buffers are not GC-reclaimable, and the reason is a table, not GC pressure**
  (gh-ocannl-550). Each backend's private `Slab.pools` (`(device_id, pool_id) -> base`) holds a
  strong reference to every slab it allocated, so no finalizer on a pool can ever run; the one
  function that drops an entry, `Backends.finalize`, had **no caller anywhere in the repo** until
  `Context.release` was added. Measured consequence, from the per-candidate census of a cold tf32
  `gpt2_mini` search: +1 working pool and ~102 MiB per candidate *attempted* — dedups and
  `Backend_link` declines pay in full, because the pool is allocated at link, before the dedup check
  — monotone to 11.9 GB of a 12,227 MiB card, `pools_freed = 0` the whole way. Contrast the same
  run's code modules: 35 loaded, 31 unloaded, live count flat at 3–4. Modules sit behind no such
  table, so cudajit's `cuModuleUnload` finalizer fires fine on an ordinary host heap. So do not
  reach for `Gc.full_major` when device memory grows — it cannot help a rooted table — and do not
  assume a class leaks just because another one does. `Ir.Alloc_census` (config `autotune_log`
  prints it per candidate) separates the four classes: working pools, constant pools, contexts,
  modules.
- **A footprint number wanted over a window is a counter, not a reading** (gh-ocannl-1006). Every
  backend's `Context.get_used_memory` is a CURRENT gauge, and the backends disagree in kind:
  CUDA/HIP sum an explicit pool table (deterministic), while `metal`'s `allocated_memory` and `cc`'s
  Ctypes total are decremented from a GC finalizer, so the same run reports different numbers
  depending on when a collection happened to run. So do not sample one of them at the end of a
  window and call it a peak. `Ir.Alloc_census.peak_pool_bytes` is the high-water mark beside them —
  raised at the shared allocator seam, never lowered by a free, rebased by `reset_peak` where a
  caller brackets a window — and it is backend-uniform, which is what makes a `cc` row and a CUDA
  row the same quantity (requested bytes) and comparable with `torch.cuda.max_memory_allocated`. It
  inherits the census's coverage exactly: a device's reserved merge-buffer slab, loaded code modules
  and host-side `Ndarray` arrays are all outside it, and it is process-global rather than per
  device. A window a tuner ran in front of needs the rebase, not a filter: an unbracketed reading
  reports the search's high water (one candidate buffer per arm), which is not the workload's.
- A CAS-guarded cleanup must not commit the flag before the cleanup succeeds. `Backends.finalize`'s
  `ctx.finalized` means "the pools were freed", not "a free was attempted": `Backend.await` inside it
  can raise (a device still reporting an asynchronous error, a dead worker domain), and committing
  first made every later release of that context a silent no-op with its pools rooted for the process
  — i.e. it reinstated gh-550's growth on precisely the failure paths where callers catch a failed
  release and carry on. It resets the flag on exception instead; that is safe only because freeing is
  idempotent per `pool_id` on every backend, so a partially completed cleanup does not double-free on
  retry. Any new atomic "done" flag around fallible cleanup wants the same shape.
- **On WSL2 a device-memory bug does not look like one.** The CUDA driver there backs allocations past
  VRAM with host memory, so the same unfixed search that OOMs promptly elsewhere reached **28.8 GB
  requested** on a 12,227 MiB card while `nvidia-smi` sat pinned at 11,879 MiB and reported headroom
  throughout; the observable symptom was thrashing (candidate times 105 ms → 3,563 ms, search wall
  time 767 s vs 105 s fixed), and `CUDA_ERROR_OUT_OF_MEMORY` arrived only at the very end. Two
  consequences: an OOM's *position* in a candidate stream is a property of the box's spare host RAM,
  not of the bug (gh-550's "arm-B candidate 47" landmark reproduced at arm-B ~135 here), and
  `nvidia-smi` is the wrong instrument — `Context.get_used_memory` sums the pool table's requested
  bytes and matched the census to the decimal at every sample.
- Anything that knows an artifact's exact lifetime should call `Context.release` (idempotent, eager,
  finalizer-independent); `Autotune.tune` does it per candidate, bounding a search at
  `beam_width + 2` live candidates instead of one per attempt. Not calling it is never a
  correctness bug, only a memory one. What `release` frees is precisely the pools a context holds
  that its parent does not and that are not per-device constants — so sibling contexts are
  independent (each `compile` mints its own `pool_id`s) but a released context is a dead handle, and
  **release leaves, never interior nodes**: a context compiled from another inherits its buffer
  locations, so releasing an ancestor leaves the descendant resolving a dropped `pool_id`. Unchecked
  precondition, deliberately (refcounting persistent context values would defeat their point).
- **Two classes `release` cannot reach, so "bounded" always needs a qualifier.** (a) Per-device
  constants: it skips every `constant_buffer_cache` key by design. That is right for a shared weight
  and wrong for a hoisted `Stage` candidate, whose `apply_stage` mints a FRESH packed-constant tnode
  per application (`fresh_tile_id ()`), so a CPU search seeding `hoist` sketches grows one constant
  pool per such candidate — measured on `cc` at 1 → 109 constant pools over 181 candidates while
  working pools stayed within 2–6. Not safely fixable in place, because constants are bump-packed
  several to a pool and the first candidate's pool mixes its private tile with the shared operand
  weights later candidates reuse; a safe rule is per-pool purity, i.e. gh-ocannl-565's eviction-policy
  work. (b) A link that RAISES after `allocate_delta` — now handled (`Backends.with_delta` frees the
  delta on the way out), but the shape is worth knowing: allocation precedes backend linking, so any
  new failure point between them leaks a whole routine footprint with no context to release it
  through. When asserting a memory bound, assert on `live_working_pools`, not `live_pools`: summing the
  constant class in makes the assertion fail for a reason it is not about, and on a workload with no
  hoisted candidates it passes while proving less than it looks like.
- Four facts about the allocation seams that each cost a review round to learn, and that any further
  release work will meet again. (1) There are **two** shared allocation sites, not one:
  `Backends.allocate_delta` for a compile's delta, and the `allocate` inside
  `Add_buffer_retrieval_and_syncing` for a `from_host`/`copy` destination not yet in the context. Both
  land in the same pool tables and are freed by the same context `finalize`. (2) `allocate_delta` is
  **not atomic** — it schedules host uploads and can allocate several segments — so a guard wrapped
  around it from outside cannot see a partial delta; the unwind has to live inside, and must `await`
  before freeing because those uploads are asynchronous. (3) Constant-cache entries **point into**
  pools, so unwinding an allocation must drop the entries that allocation inserted before freeing,
  while leaving pre-existing ones (they belong to earlier compiles). (4) Retain-then-raise is the
  standing bug shape in this area: decide what ships *after* the last thing that can raise, or the one
  artifact you deliberately kept is the one nobody can reach. Corollary for reviewing such a change:
  each fix adds a container, a guard or a retention decision, i.e. a new path with the same obligation
  — re-examine the failure paths the fix itself created, not just the ones it closed.
- Fissioned-step segment batches go through the `sequence_segments` seam
  (`Backend_impl.Lowered_backend`): Metal encodes one serial-dispatch command buffer; CUDA/HIP
  stream-capture the launch loop into a graph replayed as one `cuGraphLaunch`/`hipGraphLaunch`
  per step (gh-ocannl-488, config `gpu_graph_capture`). Graph capture bakes kernel arguments, so
  instantiated graphs are cached keyed on every launch-time-varying argument: static-index
  binding values and the merge-buffer position. Two traps encoded there: the merge pool is the
  one pool that can be REALLOCATED IN PLACE (same `pool_id`, new base), so its key component must
  be pointer identity, not `buffer_loc`; and a failed capture leaves the stream in capture mode —
  always terminate via `end_capture` before falling back or re-raising. The legacy NULL stream
  cannot be captured (OCANNL streams are all non-default, so this only bites standalone repros).
  Fallback paths (logging on, capture rejected, config off) are plain per-segment launches —
  same-stream FIFO makes the generic event chain redundant on CUDA/HIP.

- HIP scratch (private segment) is budgeted **per work-item, independent of launch geometry**, and
  a kernel over budget aborts the HSA queue instead of failing cleanly (gh-ocannl-533). The
  post-link validator in `hip_backend.ml` (`validate_scratch_budget`) declines it first, as
  `Resource_exceeded Thread_scratch`. Measured on gfx1151/ROCm 7.14/WSL2: the cutoff is
  `ceil(pss/64)*64 * max_threads_per_multiprocessor * multiprocessor_count <= 4 GiB` — 104832 B
  accepted, 104848 B rejected; gh-ocannl-533's 163856 B is far over. Traps worth remembering: the 4 GiB cap
  is NOT queryable (it is enforced by the WSL WDDM thunk, `wsl::thunk::ComputeQueue::UpdateScratch`),
  `hipLimitStackSize` is 1024 and has nothing to do with it, and hipcc separately refuses frames
  over 262136 B. Disable with `ocannl_hip_scratch_validation=false` where the model doesn't hold.
  Guard: `test/operations/hip_scratch_budget.ml` (`slow` alias).
- A typed decline is only half of gh-ocannl-533: what the issue asked for is that the SEARCH
  survive it. The rejection fired on `Autotune.tune`'s own base compile — the identity-transform
  capture, historically the one compile in `tune` that raised instead of returning an outcome — so
  it bypassed `try_spec`, the decline census and the partial report, and killed the run with a
  perfectly-classified cause. Two facts worth carrying: the baseline is the one candidate not
  compiled *as* a candidate, and passing `?lowered_transform` bypasses the default annotator, so
  what gets validated there is the unscheduled serial form — the worst case for per-work-item
  scratch, and on GPU never dispatched anyway. It is now declined (`report.baseline_declined`,
  `baseline_ms = infinity`) and the scheduled candidates carry the search; fission plus
  `promote_locals` is what brings a large softmax/CE head back within budget, which is why
  `gpt2_mini hip/tuned` completes while every whole-routine preset declines. In the census it
  carries its own cause and NOT gh-ocannl-543's `Not_dispatched_key "baseline"` — a declined
  baseline is never dispatched either, but recording both would double-count it under a reason that
  is not the one. One refusal, one entry. Guard:
  `test/operations/hip_scratch_tune_survives.ml` (`slow` alias).
- Building a test kernel that actually *has* a big scratch frame takes care: write the `Local`
  array in one loop and read it back in REVERSE in another. A forward read in the same order lets
  the compiler forward each store to its load and delete the array, leaving nothing to reject.
- `Context.get_used_memory` must report OCANNL's OWN allocation (`Slab.used_memory`, or the
  backend's atomic counter) — never the driver's `total - free`. That is device-global: it counts
  other processes and moves in allocation granules, so it cannot see sub-granule effects like the
  liveness planner's arena savings. gh-ocannl-289 fixed this for CUDA; HIP kept asking the driver
  until gh-ocannl-542, where on a gfx1151 APU sharing memory with the display it made
  `buffer_aliasing` report the planner INCREASING the footprint 106496 -> 2072576 B, against
  1556896 -> 1425668 B once measured properly (cc: 1556640 -> 1425540). When one backend's
  numeric assertion inverts while its parity assertions pass, suspect the measurement API before
  the pass under test — and check whether a sibling backend already fixed the same thing.
- Rematerialization (gh-ocannl-498) is a PLANNING pass, not a search: `Memory_budget.fit` picks
  `Inline` flips from `Backends.score_footprint` (the arena layout's own bytes) versus the
  recompute cost `flip_candidates` already carries (the cost model's per-instantiation count since
  gh-ocannl-637, the traced proxy where that count is only a bound), and compiles nothing to
  decide. The trap
  it is built around: footprint relief is NOT per-node-local under aliasing. A node whose live span
  was already shared frees nothing by leaving, and inlining one node moves the others' spans — so
  both the solo pass and the cumulative prefix are scored against a real lowering, and a candidate
  with zero marginal relief is skipped rather than taken for free (the gh-ocannl-558 enablement
  lesson, in reverse). Score the layout, never the node's own size.
- The footprint scorer deliberately scores the routine's WHOLE in-context node set, not a context's
  allocation delta, and enumerates by uid rather than in `traced_store` order — otherwise the
  selector's choices would drift with how much of the graph a particular context had already
  allocated, and with hash order across processes. On `cc` the model came out equal to the measured
  `Context.get_used_memory` delta to the byte (1392772 both ways in
  `test/operations/memory_budget_planner`), but treat that as a coincidence of a single-context
  run: the real allocator skips already-held nodes and the driver page-rounds pool bases.
- `Inline`-direction candidates want the CHEAPEST recompute cost first; `flip_candidates` is sorted
  most-expensive-first because the `Materialize` chain of `Train.tune_placements` wants that end. A
  pre-filter cut that forgets to reverse keeps exactly the flips a budget would least want to pay
  for.

- Post-admission callbacks can fail before a nonwinning candidate enters the beam or round
  (`Autotune.tune`, gh-ocannl-975). The pending owner covers compile-to-admission, and the exit
  sweep includes it; an undispatched baseline is released eagerly. Replacing the best also releases
  its previous owner if a tie had evicted it from the beam. The regression
  `test/operations/autotune_callback_release.ml` injects at both callback boundaries and compares
  exact working-pool and context census deltas against ordinary completion, separately from the
  persistent constant cache. It must inject at a NONWINNING admitted candidate — the one whose
  release the fix owns; injecting at a winner proves nothing, since a winner already sits in
  `best_so_far` and the exit sweep released it before the fix — and whether a real search admits
  one is a property of the machine's timings (a short search whose samples fall monotonically
  admits only winners: rog-nv/cuda, sweep 2026-09-20). So the test pins the ranking through
  `Autotune.on_candidate_measured` (gh-ocannl-1027), the seam that replaces each ADMITTED window's
  time before anything ranks or records it: the k-th admitted window measures k ms, the second is
  the first nonwinner on every backend and run, and the claims name it and the partial report's
  incumbent exactly. A fault-injection precondition on measured wall-clock is not a stable gate —
  pin the measurement through that seam rather than re-rolling the search for a lucky ordering.

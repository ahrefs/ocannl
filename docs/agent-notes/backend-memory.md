# Backend memory, allocation and limits

Pools, release, graph capture, hardware resource budgets, and the footprint planner.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- The resource-seam acceptance map is `docs/resource-fault-injection-inventory.md` (gh-ocannl-571).
  A release change is not accepted by a flat memory curve: inject after allocation, before and
  during cleanup, and at persistence commit/replay; pair every failure with an uninjected control
  and assert exact ownership (one free, no rooted table entry, retry state honest).
  `Resource_fault_injection.with_callback` is per DOMAIN (a `Domain.DLS` key, the `Autotune.Seam`
  rule; it cannot reuse that module, which sits above `Schedule_cache` in the dependency order): a
  callback installed in one domain is all that domain's operations reach, and a fresh domain starts
  with the no-op. That holds only while every `hit` fires on the domain that CALLED the operation.
  On `multidev_cc` the worker runs the task the scheduler ENQUEUES (the backend's kernel or transfer
  task); the routine-level schedule `Task.enschedule` wraps it in only enqueues, on the caller. A
  `hit` placed in an enqueued task would fire on the worker against the no-op and vanish from every
  test. `resource_fault_injection_domains` pins both halves (two-domain isolation on `cc`; the same
  point set from the caller on `multidev_cc`).

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
  finalizer-independent); `Autotune.tune` does it per candidate at the end of its timing window,
  so at most one candidate is live (next entry). Not calling it is never a
  correctness bug, only a memory one. What `release` frees is precisely the pools a context holds
  that its parent does not and that are not per-device constants — so sibling contexts are
  independent (each `compile` mints its own `pool_id`s) but a released context is a dead handle, and
  **release leaves, never interior nodes**: a context compiled from another inherits its buffer
  locations, so releasing an ancestor leaves the descendant resolving a dropped `pool_id`. This
  discipline remains for compiled descendants (now refused, next entries, not repaired); upload
  siblings reference-count shared working pools (gh-ocannl-1173) without counting superseded values
  in a linear upload chain.
- **A search holds one candidate, and a replay never sits beside the base compile**
  (gh-ocannl-1326). gh-ocannl-550's beam bound (`beam_width + 2` routine footprints) still caps
  tuning at half the device: a b256 `gpt2_mini_train` search on a 12 GiB card timed one candidate
  (6.6 GiB) and declined every later one at link with `CUDA_ERROR_OUT_OF_MEMORY`. `Autotune.tune`
  now releases each candidate when its timing window ends, ranks incumbents by their saved forms,
  and recompiles the winner from `ctx` on both paths (one compile; the `timing_ctx` path always
  paid it). The issue read the warm-run re-search as "nothing cached"; the entry WAS stored, and
  the replay compile was refused because the base compile's 6.6 GiB was still resident, so the
  baseline is now released before any replay and relinked if the entry is stale. One level up,
  `Autotune.tune_deferred` leaves even the winner uncompiled, so a caller running several searches
  (`Train.tune_placements`' arms and flips) holds no searched routine while the next one searches
  (gh-ocannl-1408).
  `test/operations/autotune_device_memory` simulates a capacity at the `Delta_pool_allocated`
  fault-injection seam; `autotune_candidate_release` pins zero working pools between candidates.
- **A refusal's free bytes are read with the refused compile's own earlier pools still allocated**
  (gh-ocannl-1424). A group past the per-pool cap (4 GiB unless `large_models`, `pool_cap`) is
  several pools that `allocate_delta` allocates one after another before placing anything, and
  the driver's free bytes in `Device_memory_exhausted` are read at the refusal, before the unwind
  gives the earlier pools back. Read bare, that looks like memory something else holds: the b256
  `gpt2_mini_train` materialize-all arm (28.2 GB working in seven pools, by
  `Memory_budget.footprint`) was refused at its third pool on rog's 12 GiB card with its first two
  (8.35 GB) still allocated, and at its second on tuf's 8 GiB card, and the free bytes read as 8.6
  GB and 3.9 GB "resident after arm A", while arm A (6.9 GB) gives back everything it allocated. The
  refusal's detail now names the pool's position, the group's total and the bytes the earlier pools
  held. Before blaming a
  leak, score the footprint, and compare the `working_bytes` (census) and `device_free` (driver)
  fields of `Train.tune_placements`' progress lines across the arm boundaries: equal working bytes
  with a falling `device_free` is memory no pool table accounts for.
- Device allocation refusals are typed at the GPU pool allocators
  (`Schedule_outcome.Device_memory_exhausted`, the `Resource_exceeded Device_memory` decline with
  the refused bytes and the driver's free bytes), because HIP has no `classify_failure` and its
  out-of-memory error was therefore unclassified, i.e. fatal under strict classification, and
  CUDA's was a `Backend_rejected` stage string nothing could reason about. A search-wide
  refusal (`Pre_search_failure` at `Backend_link`, before the cache is consulted) asks one question:
  does the device admit the FLOOR every schedule of the base lowering must allocate,
  `Context.probe_link_floor` (gh-ocannl-1554)? Asked where the GPU baseline is refused before
  codegen (gh-ocannl-1340), and where the base link or a stale entry's baseline relink is refused
  for device memory, which then declines the baseline and nothing more. Not the base link's own
  delta (`Context.probe_link_delta`, gh-ocannl-1493, kept exact): "scheduling only adds device
  buffers" fails twice. Under `buffer_aliasing` the GPU planner's position is the kernel, so the
  one-kernel base overlaps nothing a fissioned candidate overlaps (954 KB against 791 KB for
  `link_delta_parity`'s training step on Metal), and a CPU `Split_reduce` candidate refines the
  base's statement spans. And `Fuse_epilogue`'s shared placement keeps an unsettled target off the
  device (49 KB against 25 KB). The floor (`Backends.link_floor`) is raw node bytes over the delta's
  nodes the base code touches, less `Schedule.may_keep_off_device` (an unsettled fusion target
  within `max_workgroup_memory_bytes`, which the hardware gate holds every kernel's shared bytes to,
  GPU only) and `Schedule.may_replace_by_packed_tile` (a hoisted `Stage` source); under the planner,
  the nodes failing `Backends.alias_eligible` plus the largest eligible node per precision class.
  The size window is narrow in practice: a `Never_virtual` node within `stack_threshold_in_bytes`
  is `Local` in the base anyway. Its probe asks each group's bytes in one request cut at the pool
  cap, decisive under a page-granular allocator that admits by the total of page-rounded requests,
  not against a contiguous-block limit. A floor the device admits says nothing about any
  candidate. A transform that newly takes storage off a node must extend the floor and
  `link_delta_parity`.
  `Context.link_delta` is the allocation-free half (the plan's bytes), and
  `test/operations/link_delta_parity` pins it to the census delta of a real compile. Size a
  capacity meant to refuse the search on the FLOOR, which is the unscheduled delta only where
  nothing is left out, and never on the default compile: on a GPU that needs more (8192 against
  4096 B on the test's workload under CUDA), which made a half-footprint test capacity fit the
  baseline there. The same holds for a search WINNER: the floor bounds every shipped routine from
  below, but the default compile's footprint does not, since a winner such as an epilogue-fused
  sketch need not carry fission's promotions (4096 B on Metal where the default holds 8192). A
  capacity meant to leave no room beside "the shipped routine" is sized from the floor, or it
  passes or fails by which candidate the timings crowned (gh-ocannl-1415).
- **`Context.t.context_id` names a release unit, not a value** (gh-ocannl-1203). The merge-buffer
  ledger retires a consumer's registration by the released context's id, so every value that shares
  a backend release flag (`finalized`: a linear upload chain, an upload into an existing buffer)
  must share the id, and `Context.derive` preserves it for them; a fresh id per upload strands the
  readers compiled before it and refuses every later refill. An upload fork gets its own id and a
  copy of its parent's registrations; `test_execution_deps` Test 9 pins both directions.
- **The same flag is what refuses a dead handle** (gh-ocannl-1329). `Context.check_live` reads
  `finalized` at every entry point that touches buffers (compile, run on either the run context or
  the routine's compile context, sync, the transfers, the `decide_*` recorders) and raises a typed
  `User_error`. The compile-context check is waived when the run context shares its upload lineage:
  a live upload fork reference-counts the routine's pools, so it outlives the released chain. Because the flag is shared exactly where pools are, an earlier value of a linear
  upload chain is refused with the latest one, while compile siblings and upload forks keep working.
  A released ANCESTOR is refused too (gh-ocannl-1462): a compiled descendant has a fresh flag of
  its own, so `check_live` also walks the backend context's `parent` chain (the lifecycle chain
  `make_child` records) for a set flag, naming how many levels up; no second registry. An upload
  fork keeps its source's PARENT, not the source, so a fork outliving its chain is not refused, nor
  is a child compiled from it. `release` itself is not checked: a descendant still frees its own
  pools. The `decide_*` refusal matters for the same reason: `make_child` of a dead handle would
  mint a live-looking flag over freed buffer locations. Guard:
  `test/operations/context_handle_refusals.ml` (`chr_released_ancestor`, `chr_fork_child`).
- **`Context.run`'s initialization check has two classes** (gh-ocannl-1339). A read-before-write
  input (in both `inputs` and `outputs`: a counter, a running statistic) must have its BUFFER
  written in the lineage: the ledger's `written_buffers` holds the buffer keys every run, upload and
  device copy wrote. It is keyed by buffer, not by node, so a compile sibling that seeded its own
  buffer for the same node vouches for nothing here (the run context's node-level
  `initialized_nodes` is deliberately not consulted for this class), and a caller dropping the
  context `run` returned loses nothing. `rollback_execution` withdraws exactly the keys the
  rolled-back run introduced (`fresh_writes`). Allocation-time zeros no longer count. A READ-ONLY
  input must be written the same way unless `Tnode.zero_valid` declares zero its correct value
  (gh-ocannl-1462): `Tensor` marks every gradient at creation (both sites, `op` and
  `force_param_diff`), `Tnode.mark_zero_valid` marks any other read-only accumulator, and the mark
  never excuses read-before-write state. Before, a read-only input passed once merely allocated, and
  since `compile` allocates every input a forgotten data input read zeros. A node every routine only
  reads may live in a constant pool of its root that compile siblings SHARE, so a sibling-seeding
  test makes it `context_owned` to get a buffer per sibling (`chr_read_only_buffer`). Hand-built
  tests that accumulate from zero therefore seed the accumulator (`peel_census`) or mark it; state
  meant to start from zero inside a model is an inline declaration, an embedded node excluded from
  `inputs` (`Train.sgd_one`'s momentum). An SGD step compiled without `Train.init_params` never
  reaches this check: linking already refuses a parameter the prior context lacks.
- **Constants are cached per root; only host literals are shared across roots** (gh-ocannl-1654,
  gh-ocannl-1623). Every non-`context_owned` node a compile only reads (data inputs, zero-valid
  accumulators, `Host_inits` literals: only parameters are `context_owned`) lands in a constant pool
  that outlives its contexts, cached in the device's `constant_buffer_cache` under a
  `BI.Constant_key`: a `Host_inits` literal is `Shared` (one buffer per device, uploaded once), any
  other node `Rooted` under the `constant_root` that `Device.make_context` mints and every derived
  context inherits. Compile siblings and descendants of one root therefore share a constant (a
  sibling-seeding test still needs `context_owned`), while independently created roots each bind
  their own: a `set_values`, a writer routine or a device copy in one root never reaches another,
  and a copy between roots is a real transfer. Before, every root bound one device-wide buffer for
  every constant while keeping its own ledger, so one root's write was a silent wrong value in the
  other (then refused as stale). The literal class keeps that hazard, since literals are writable
  through the API: every write into a `Shared` buffer mints a token (`Context.shared_write_tokens`,
  plus the writer ledger's `observed_tokens`); `run` refuses a `routine.shared_reads` member whose
  current token its ledger did not leave or adopt, a no-op copy adopts only from a source that is
  current, and `rollback_execution` restores both tokens while nothing wrote since. Ordering needs
  no edge: roots on one device share its single FIFO runner (concurrent host domains are outside
  this, as for all of `Context`). Not refused: `to_host`/`get_values` and real copies out of a stale
  source, since reads check no initialization. Lookups (`plan_delta`, `link_floor`,
  `owned_pool_ids`) go through `BI.find_constant`, root first; insertion through `BI.constant_key`.
  Rooted constants pack into pools of their own (`dp_new_root_constants`), and each release unit
  holds the rooted pools its `ctx_buffers` binds (`held_constant_pools`, counted in the device's
  `rooted_constant_owners` at the three places a unit is minted: `make_context`, `make_child`, an
  upload fork); the release of the last binder frees the pool and drops its cache entries, so
  fresh roots compiled and released one after another hold memory flat, as the device-wide cache
  did by reuse. A unit nobody releases keeps its pools, as for working pools. A link that fails
  after `allocate_delta` reclaims the delta's rooted pools no unit holds in `free_delta`, after the
  owned pools (an evicted pool would read as owned and be freed twice), and a failure after its
  `make_child` first gives the discarded child's holds back (`reclaim_rooted_constants`'s modes). A new
  place that mints a `finalized` flag must hold its rooted pools too, or a release frees a pool it
  binds. Guard: `context_handle_refusals` (`chr_root_constants`, `chr_root_constants_release`,
  `chr_shared_overwrite`, `chr_shared_writer`, `chr_shared_implicit`).
- **A transfer adds no lifecycle level** (gh-ocannl-1623). `device_to_device` into a buffer `dst`
  already holds returns `dst` itself, and a merge-buffer fill returns `{ dst with
  merge_buffer_node }`: neither allocates in `dst`, so both are `dst`'s release unit (releasing
  either releases both, as in a linear upload chain). A `make_child` per copy grew the `parent`
  chain `check_live` walks by one level per transfer and retained every ancestor. Compiles and the
  `decide_*` recorders are genuine children and still add a level; `Context.lifecycle_depth` reads
  it. Nor does a merge fill add a ledger record (gh-ocannl-1654): it retires its predecessor's
  `executed` entry, reader registrations and routine name, so a retired transfer is the one id
  `routine_names` forgets, and the missing-dependency refusal reports an unnamed id as a superseded
  fill; transfer ids are never reused. `Context.ledger_entries` reads the count. Guard: `context_copy`
  scenario 6 (1000 copies per path, depth and ledger size constant).
- **Read-only in one routine does not make a parameter a device-wide constant** (gh-ocannl-641).
  A host-backed parameter used only as another parameter's initializer input has no executable
  write in that routine. Its `Tnode.context_owned` construction marker keeps it in the working
  group nevertheless; both allocation and footprint scoring use `partition_layout_groups`.
  Two fresh contexts must retain independent parameter/state edits and resets. The marker changes
  allocation only, not the post-virtualization scalar recompute-price inputs.
- **Two classes `release` cannot reach, so "bounded" always needs a qualifier.** (a) Device-wide
  constants (host literals; a root's constants are freed with their last binder, above): it skips
  every `Shared` `constant_buffer_cache` key by design. That is right for a shared weight
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
- Five facts about the allocation seams that each cost a review round to learn, and that any further
  release work will meet again. (1) There are **two** shared allocation sites, not one:
  `Backends.allocate_delta` for a compile's delta, and the transfer destinations inside
  `Add_buffer_retrieval_and_syncing` for a node not yet in the context (`upload_slot` for
  both `from_host` and `init_from_device`, gh-ocannl-1172). Both land in the same pool tables and
  are freed by the same context `finalize`. (2) `allocate_delta` is
  **not atomic** — it schedules host uploads and can allocate several segments — so a guard wrapped
  around it from outside cannot see a partial delta; the unwind has to live inside, and must `await`
  before freeing because those uploads are asynchronous. (3) Constant-cache entries **point into**
  pools, so unwinding an allocation must drop the entries that allocation inserted before freeing,
  while leaving pre-existing ones (they belong to earlier compiles). (4) Retain-then-raise is the
  standing bug shape in this area: decide what ships *after* the last thing that can raise, or the one
  artifact you deliberately kept is the one nobody can reach. Corollary for reviewing such a change:
  each fix adds a container, a guard or a retention decision, i.e. a new path with the same obligation
  — re-examine the failure paths the fix itself created, not just the ones it closed. (5) A cleanup
  with no retry owner (`allocate_delta`'s unwind, `free_delta` after a failed link, the probe's give-back)
  frees through `Backends.free_pools_continuing`: every pool is attempted and the first refusal
  re-raised at the end, since nothing holds the ids afterwards and a skipped pool is leaked for good
  (gh-ocannl-1555). A context `finalize` is the opposite case: it stops at the first refusal and
  keeps `released_pool_ids` as the retry ledger. `resource_fault_injection` pins the unwind's half at
  `Delta_unwind_before_free` with a working and a constant pool.
- **Host uploads and device copies of absent nodes share upload arenas, owned by the context LIFECYCLE**
  (gh-ocannl-1125). One pool per `set_values` of a not-yet-linked node made a routine reading 18
  such parameters need 20 pools, over Metal's 16-binding budget (`build_pool_binding`), so
  `upload_slot` bump-packs them first-fit into `context.upload_arenas`: first arena exact, later
  ones doubling. The doubling sums only arenas the uploading value holds a tenant of: summing the
  whole lineage made `k` simultaneously held forks, each uploading one node, mint `2^k - 1` nodes'
  worth (gh-ocannl-1204). The arenas are shared by the values `evolve_with_buffer` derives (one `finalized`
  flag along a linear chain) and fresh in `make_child`. Uploading into an earlier value forks an
  independent release flag; `upload_arenas.pool_owners` reference-counts the pools those sibling
  leaves share (gh-ocannl-1173). Linear uploads supersede their intermediate values without retaining
  extra owners. Failed cleanup remembers retired references alongside freed pools, so retry cannot
  decrement twice. Within a lifecycle only a value holding an
  arena's latest tenant extends it, so two sibling uploads cannot reuse the same tail. The arena's
  tenant history is trimmed against surviving owner maps on retirement, allowing the survivor to
  reuse a released tail instead of stranding one pool per fork. A tenant is a LOCATION: "holds" is
  `Backend_intf.holds_tenant` (the node at the tenant's own pool and offset) in `fits`, the doubling's
  sizing and the trim alike; asking `Map.mem` of the node let a fork holding the same node elsewhere
  append into a pool only its sibling owned, freed under it at the sibling's release (gh-ocannl-1328).
  The resulting invariant -- every tenant a retained arena lists is held at its location by a
  registered owner of its pool, since only a holder of the last tenant appends, maps only grow, and
  the trim stops at a held tenant -- is `Backend_intf.upload_arena_violations`; `finalize` checks it
  after a successful cleanup under `debug_check_upload_arenas` (gh-ocannl-1462), outside the
  flag-resetting branch and after the release hooks, and `set_values_pool_coalescing` runs with it
  on.
  A reused tail keeps its `Buffer_key`, so the execution ledger may hold a released leaf's writer for
  it; harmless, because only `upload_slot` re-issues a location and its callers record an external
  write over it, and a ledger writer is always executed. A failed upload into an existing arena rolls
  its bump back and frees nothing; `finalize` removes an arena only when its last owner frees the
  slab. Release only the latest value in each linear upload chain. Guard: `test/operations/set_values_pool_coalescing.ml`, plus the arena leg of
  `resource_fault_injection`. Device copies share the same allocator and await inside its unwind
  guard before publishing ownership (gh-ocannl-1172); mixed transfers inherit the same rules.
  Metal publishes its per-routine binding budget as `hardware_limits.max_bound_pools`, and its
  link error suggests packing absent nodes before compiling.
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
- The instantiated graphs belong to the routine's context (gh-ocannl-1445): `sequence_segments`
  returns a release action beside the task, `Backends.link` adds it to the child context's
  `release_hooks`, and `Backends.finalize` runs the hooks once the pools are freed, outside the
  retryable cleanup: a failing hook neither keeps the pools rooted nor resets the `finalized` flag
  over freed pools. Hooks are idempotent and leave the routine able to recapture, since an upload
  fork shares the list and can outlive the value released. Every cache still alive at exit
  is destroyed by the backend's `at_exit` teardown, on HIP before the stream. The caches are held
  through `Backend_impl.Weak_registry`, so a routine dropped without a release is still collected,
  its graphs by cudajit's and hipjit's own finalizers. `live_graph_execs` in the backend's global
  debug info (`Context.global_debug_info`) counts what the caches hold;
  `test/operations/graph_exec_release` pins the release, `exit_stream_teardown`'s `device_graph`
  arm the exit.

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
- The compiler's frame limit is the other half, and it is gated BEFORE codegen (gh-ocannl-1313):
  LLVM AMDGPU refuses a stack frame over 262136 B, and hiprtc says so untyped, only after compiling
  the whole kernel — 52 s on the `gpt2_mini_train` step's serial base form, fatal under strict
  classification, so `tune_placements`' default-placements arm ended `pre-search-failure`. Each
  node there sat under `stack_threshold_in_bytes` (89 `Local` arrays of 4 KiB) while their sum did
  not: the threshold is per node, the limit per kernel. `Schedule.thread_private_bytes` sums the
  arrays codegen will declare through the pure `Tn.Placements.is_local_peek` (360452 B against the
  measured 360544 B frame), and `check_hardware_limits_classified` refuses it at `Hardware_limits`
  against `hardware_limits.max_thread_private_bytes` (HIP only). The base and the whole-routine
  presets then decline in about a second each, and the fissioned candidates, whose `promote_locals`
  moves the crossing scratch to device memory, compile and get searched. Guard:
  `test/operations/thread_private_gate.ml`. The sum reads `Tn.size_in_bytes_peek`, never
  `Lazy.force tn.size_in_bytes`, and so does `Schedule.workgroup_memory_bytes`: a backend-specific
  estimate leaves the node's lazies as every other backend finds them. Whatever the tests print
  must not read a lazy's forced state — `Tn.header`'s `mem in bytes` is derived from dims and
  precision (`<not-in-yet>` only while the dims are unresolved), not `Lazy.is_val` of the size
  (gh-ocannl-1369), after a HIP-only forcing turned the backend-uniform
  `test/einsum/moons_demo_variant` golden red on HIP alone.
- Building a test kernel that actually *has* a big scratch frame takes care: write the `Local`
  array in one loop and read it back in REVERSE in another. A forward read in the same order lets
  the compiler forward each store to its load and delete the array, leaving nothing to reject.
- **The cc counterpart is a stack budget, not a refusal** (gh-ocannl-1669). A `Local` array (a
  `Stage` tile, an explicit `Local` intent: nothing caps their size, unlike `Never_virtual`'s
  `stack_threshold_in_bytes`) went on the stack at any size, safe only on the 8 MiB main thread;
  libdispatch workers and macOS secondary threads (a `multidev_cc` domain) get 512 KiB. Now
  `C_syntax.heap_locals` keeps a scope's smallest arrays on the stack within
  `routine_stack_locals_bytes` (function scope) or `chunk_stack_locals_bytes` (a pool chunk's
  privatized tiles) and `malloc`s the rest, freed in the same scope (GPU backends opt out through
  `heap_local_arrays`). The 12 MiB legs of `test/operations/heap_local_arrays` segfault without it.
- **On Metal an over-limit `threadgroup` allocation is refused by pipeline creation, not by the
  post-link check** (gh-ocannl-1226): `newComputePipelineStateWithFunction` fails first ("Threadgroup
  memory size (32772) exceeds the maximum threadgroup memory allowed (32768)"), so
  `get_static_threadgroup_memory_length` never sees it. `Metal_backend.link_proc` types that
  sentence as `Resource_exceeded Workgroup_memory` at `Backend_link` with the driver's counts (a
  decline to a search); any other pipeline-creation failure stays untyped. The schedule gate is
  still the cheaper refusal, before compilation, so everything a kernel declares in `threadgroup`
  memory must reach `Schedule.workgroup_memory_bytes` (gh-ocannl-1205): staged tiles from
  `workgroup_shared`, and backend-emitted scratch through
  `codegen_capabilities.mma_kernel_workgroup_bytes`, which the emitting backend derives from the same
  predicate as its kernel-scope declaration (`C_syntax_config.mma_kernel_prologue`): Metal's
  converted-boundary coordinate table, 256 B once per kernel however many converted `Tile_mma`
  scopes (each `Unroll`/`Partition` copy is one) read it (gh-ocannl-1255; it was per scope). The
  renderer hands the prologue the triples of the emissions it accepted, the estimate those of every
  `Tile_mma`, so the estimate is the upper bound. Renderer-inserted collectives route the same way:
  the lane all-reduce's `lred_partials_` (`lane_all_reduce_workgroup_bytes`, gh-ocannl-1168) and
  the multi-warp warp-shuffle reduction into a cell's `wred_partials_`
  (`warp_reduce_workgroup_bytes` over `Low_level.warp_reduce_cell_sites`, one slot per warp at the
  cell's accumulator residency; the renderer checks its declaration against the capability,
  gh-ocannl-1452). A new `threadgroup` or `__shared__` declaration in a backend emitter needs the
  same route. Metal rounds each `threadgroup` array up
  to 4 bytes (measured, M4 Max: the driver's request for two odd `half` tiles is their rounded sum),
  so the estimate charges each array rounded to
  `codegen_capabilities.workgroup_allocation_alignment` (gh-ocannl-1330), through
  `Schedule.kernel_workgroup_bytes` — which the sketch families' workgroup-memory refutations and
  filters call too, with the compiling context's capabilities (`?capabilities`; without them they
  charge exact tile sizes and no scratch, a weaker floor). So does the untuned default's
  scalar-blocktile seeding rule (`Autotune.seeded_params`, through `seeded_plan`, its decision
  rule and every context-holding caller); its tiles are multiples of 8 elements, so Metal's
  granule changes none of its verdicts today, and
  `test/operations/seeded_params_capabilities.ml` pins the threading with a coarser stand-in
  granule (seeded at the gate's estimate, refuted one byte below). A prediction prices each staged
  tile at the precision its `Stage` declares it at: the source operand's storage precision, not the
  accumulator's (the conv family's GPU leg priced both at the output's until gh-ocannl-1452, refusing
  mixed-precision seeds that fit), and the conv leg also charges the accumulator fragment of a
  flavor with a row or column pad (a reduction pad alone masks nothing: `Tensorize` discharges its
  guard), which the masked contraction places in workgroup memory on a staged pipeline
  (`test/operations/workgroup_memory_accounting.ml`: every GPU conv seed kept at its built
  schedule's gate estimate, refused one byte below). With the gate exact, the one route to
  the link-time refusal is the test seam `Schedule.with_gate_capabilities`. Guards:
  `test/operations/schedule_mma_scope_scratch.ml` (tiles of exactly the limit compile; plus one
  converted scope they are refused at `Hardware_limits`) and
  `test/operations/metal_threadgroup_overflow.ml` (odd tiles at the limit are refused at
  `Hardware_limits`, even ones run; under the seam's exact sizes the odd ones reach `Backend_link`).
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
  for. The planner's cut counts nodes (gh-ocannl-1017), ranked by each node's cheapest
  `Inline`/`Footprint` alternative, so it cannot split a node's directions.

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

## HIP kernel code cache
- **hiprtc costs per PROGRAM and compiles one program at a time per process** (gh-ocannl-1314,
  ROCm 7.1 hiprtc). On the dumped 261-kernel source of a `gpt2_mini_train` step: one joint program
  5.29 s on gfx1102 (3.03 s on gfx1151), the same kernels as one program each 87.7 s (37.1 s),
  i.e. a fixed 0.34 s (0.14 s) per program, and 4, 8 or 16 threads changed nothing although
  hipjit's bindings release the runtime lock. So neither per-kernel modules nor in-process parallel
  compiles pay; compiling FEWER kernels in one program does.
- `Hip_backend`'s batch compile therefore keys each kernel by its standalone source
  (`Compile_driver.render_batch`'s `assemble [i]`, name included) with its minted names renamed by
  first occurrence (`Ir.Kernel_code_cache.canonical_symbols`: every lowering and every
  loop-splitting op mints fresh `i<N>` symbols and `v<N>_` locals, so without it no kernel of the
  next candidate ever matched; the renaming is the KEY's only, since tests and the volatility census
  read the emitted names) plus its complete hiprtc option
  list (`hiprtc_options`), reuses the code of every kernel an earlier compile in the process
  produced, and compiles the misses as ONE program per distinct option list, published to the cache
  only after hiprtc returns (`Ir.Kernel_code_cache`: bounded FIFO by bytes,
  `hip_kernel_cache_mib`, 0 restores one joint program per batch). Parameters, bindings and launch
  geometry always come from the current rendering; `link_batch` loads each distinct program once and
  every kernel still goes through `link_proc`'s scratch validation. Under device debugging
  (`Utils.with_runtime_debug`) the cache is bypassed: a code object's debug info names the file it
  was compiled from, which the next compile of the routine rewrites. Within a search the winner's
  compile hits the cache, so the shipped code is the timed code.
- The `.hip` build artifact stays the routine's FULL generated source whatever was compiled (tests
  read it through `Generated`); `.hsaco`/`.hip_log` belong to the program actually compiled, and
  `.hip_log` opens with `kernel code cache: compiled k of n kernels`. Guards: `kernel_code_cache` (the cache, any backend), `gpu_kernel_cache` (reuse on HIP and CUDA,
  with an `OCANNL_HIP_KERNEL_CACHE_MIB=0` / `OCANNL_CUDA_KERNEL_CACHE_MIB=0` negative control).
- **CUDA has the same cache, for a different reason** (gh-ocannl-1529, `cuda_kernel_cache_mib`,
  default 64): nvrtc and the driver's PTX JIT cost per KERNEL, not per program. Measured on rog
  (sm_120, nvrtc 13.4, `CUDA_CACHE_DISABLE=1`) on a `gpt2_mini_train` step's 269 kernels with
  `benchmarks/gh1529_nvrtc_cost.sh`: nvrtc 18 ms per program + 12.9 ms per kernel, JIT 4.6 ms per
  kernel; a cold search's candidates spent a median 1.42 s in nvrtc and 0.95 s in the JIT of a
  2.77 s compile, on all 261 kernels. Per-kernel programs cost 2.6x the joint one and nvrtc's
  threads recover only 1.7x, so the HIP design carries over: key, misses as one program per option
  list, publish after nvrtc returns. Two CUDA specifics: a `program` keeps its loaded module per
  CUDA context (`module_of`), since reloading a reused program would re-pay the JIT for every
  kernel in it; that keeps device code resident while the cache or a routine holds the program,
  hence the smaller default. And the nvrtc options key on the source WITH its conditional includes
  (`nvrtc_options`, as `cuda_to_ptx` did), where HIP keys before them. Measuring trap: with the
  driver's compute cache on (the default), nvrtc too answers an unchanged program from it (0.14 s
  instead of 3.5 s for the joint gpt2 program), so a cost probe that repeats a compile must run
  under `CUDA_CACHE_DISABLE=1` -- an A/B of two searches too, or the arm re-rendering candidates an
  earlier run compiled reads warm. Measured that way (`gh1529_nvrtc_cost.sh ab`), the cache took a
  cold gpt2 b8 search's candidate compile from 3.46 s to 1.73 s (nvrtc 1.38 to 0.40 s, JIT 0.96 to
  0.17 s, 29 of 261 kernels compiled) and fit 40 candidates in a budget that fit 26.
- **CUDA compiles a large group of misses as a few concurrent programs** (gh-ocannl-1603,
  `cuda_compile_programs` default auto = min 8 / domain count, `cuda_compile_program_min_kernels`
  default 16). A search's first compile misses every kernel; on rog (nvrtc 13.4,
  `CUDA_CACHE_DISABLE=1`) a gpt2 step's 269 kernels took 4.73 s as one program, 2.56 s as 4 and
  2.14-2.18 s as 8 contiguous programs compiled on as many threads and loaded serially, 7.22 s as
  269 per-kernel programs on 8 threads (`gh1529_nvrtc_cost.sh`'s `chunks` rows): nvrtc parallelizes
  across programs, and only its per-program cost forbids going fine. HIP stays one program (hiprtc
  serializes in-process). The machinery is `Kernel_code_cache.resolve`'s `?concurrent` hook: after
  dedup and option grouping, each group is cut into `programs_for` contiguous `chunks`; when any is
  cut, every program is `prepare`d on the calling domain (artifacts, census), compiled at once (one
  domain each, the caller's included, all joined under the compile lock), then `finish`ed and
  published in batch order, the first failure re-raised after. Traps: `Compile_cost` is
  domain-local, so `device_s` is the phase's wall time timed on the caller (`phase`), never a sum;
  the worker runs only `nvrtc_compile` (no ppx_minidebug logging, no `Utils.build_file`, whose
  artifact lock is process state). The partition is invisible to the key and to the schedule cache.
  Tests: `kernel_code_cache` (fake compiler, a start barrier that a serialized run cannot pass),
  `gpu_compile_programs` (a training step compiled whole and split in one process, bitwise parity,
  claims gated on `Kernel_code_cache.split_capable_compiles`; `OCANNL_CUDA_KERNEL_CACHE_MIB=0` is
  its negative control). Tests force the policy with `Kernel_code_cache.with_split`.
- **An unchanged kernel must key alike whatever names the rest of the batch made it take**
  (gh-ocannl-1525). Three things renamed unchanged kernels between candidates: node names are
  disambiguated over the WHOLE batch (`get_ident`: a repeated label becomes `n<id>_<label>`, and the
  `Schedule` tile nodes a sketch mints -- `tile__n<fresh>`, label `packed_<src>` -- get fresh ids
  every lowering, so two sketched kernels packing same-labelled sources change text on every
  candidate, as do the locals derived from them); segment kernels are named by position
  (`<routine>__seg<i>`), so a candidate adding or removing a segment renames every later one; and
  equal kernels of one batch (repeated layers) compiled once each. Simulated over a Metal staffed
  gpt2 search's 10.5k rendered kernels: 3852 misses with the old key, 2104 name-insensitive, about
  630 with node names renamed too. So the key is `Compile_driver.render_batch`'s `key_text`: the
  prelude verbatim, the definition through `Kernel_code_cache.canonical_kernel`, which renames the
  names the kernel BINDS -- symbols, scope locals, the identifiers `get_ident` gave its nodes (bare
  or disambiguated), and the emitter's derivations of them (`C_syntax.minted_derivations`; every
  emission site spells a derived local through `derived_name`) -- and drops its segment index (`Kernel_code_cache.kernel_token`: the routine name stays, since it can decide whether the text compiles); the
  cache stores the symbol each kernel was compiled under (`code.symbol`, `procedure.symbol`) and the
  link looks the function up by it. Sound because a declaration shadows every outer meaning of its
  name in its scope; members, attributes, preprocessor lines and literals are never renamed, and a
  node named like a free name (`routine_ident_blacklist`) disables node renaming for that kernel.
  `Kernel_code_cache.resolve` is the one orchestration (lookup unlocked, misses re-looked-up under
  the lock, one representative per key, one program per option list, publish after success),
  pinned against a fake compiler in `kernel_code_cache`; `gpu_kernel_cache` pins a fresh build with
  colliding labels compiling no kernel (and a renamed routine compiling afresh). Measured on minix (gfx1151,
  **unified**, gpt2 b8, 300 s budget, base vs head): a bare candidate's median compile 1.87 to 0.68 s
  (12 to 1 of 261 kernels), a staffed one's 4.10 to 0.69 s (68 to 1), the winner's 6.7 to 1.9 s (112
  to 6-8 of 269).
- **cc has the cache too** (gh-ocannl-1525, `cc_kernel_cache_mib`, default 256, weighted by library
  file bytes): the C compiler costs per kernel and a lot -- a `gpt2_mini_train` candidate of 16
  kernels spent 10-37 s in it on mac-studio. Misses compile as one library; the toolchain identity
  (command, resolved executable, kernel flags, link flags) is in the key. A cached library stays
  loaded while the cache holds it (under OpenMP none is ever unloaded anyway). Measured on mac-studio (gpt2
  b8 cc search, 600 s budget, cache on vs `cc_kernel_cache_mib=0`, 2 rounds): candidate compile
  7.45 s to 4.11 s median (C compiler 5.8 to 0.46 s, 128 to 1 of 128 kernels compiled; the rest is
  lowering), the winner's 7.2 s to 1.8 s with no kernel compiled; the budget there is dominated by
  the two serial base compiles (about 53 s each).
- **A large cc kernel is outlined, and a large batch compiled as several units** (gh-ocannl-1603).
  The cache cannot help a search's FIRST compile, the serial baseline: one C function of every
  statement of the routine, and clang -O3's cost grows faster than linearly in a function's size
  (`gpt2_mini_train` b8, arm B's 9,050 lines: 127 s whole, 30 s as 62 `noinline` functions in one
  unit, 8 s across 8 units compiled concurrently). So `Kernel_outline.groups` packs a kernel's
  top-level statements into parts (`cc_kernel_part_weight`, a statement weighing its IR nodes,
  about a line of C each; a kernel is outlined above sixteen parts), each an
  `ARRAYJIT_KERNEL_PART` function taking the kernel's parameters plus a pointer per function-scope
  local array, called in order; and `Cc_backend` compiles a batch whose units total at least two
  `cc_compile_unit_min_bytes` as up to `cc_compile_jobs` translation units (`-c` each, by
  concurrent processes, then one link). Traps: a cut must not fall between a zero initializer and
  the statement it may be forwarded into, nor inside a top-level scope local's reach (materialize-all
  code has them: a reduction's `Local_scope` declared at the top level); the builtins needed
  internal linkage (`ARRAYJIT_BUILTIN_LINKAGE`, `static` in kernels) because every unit carries
  them, while the host header keeps them external for the stubs that link against `builtins.c`;
  `OCANNL_`-prefixed macro names trip the environment-spelling scan; a test reading a kernel's
  statements must splice its parts back in (`Generated.main_logic`, build-and-test.md). Windows
  compiles one unit.
  Arm A's base compile went from 30.7 to 8-13 s, arm B's from 102-157 to 11.7 s (correctness-grade
  diagnostics, shared box).
- **Metal caches the library AND the pipeline states** (gh-ocannl-1605, `metal_kernel_cache_mib`,
  default 64, weighted by compiled source bytes -- a proxy, Metal reports no library or pipeline
  size). Metal compiles at LINK, and on a gpt2 b8 search the library compile was only about 0.4 s
  of a 1.8 s candidate compile: the rest was creating each kernel's compute pipeline state, which a
  library-only cache does not save. So `Metal_backend` renders through `render_batch` at compile,
  keeps `assemble`/`key_text` in the code value, and resolves at link, where the device is known:
  the key adds the rendered Metal option state (computed once per link, and the same value
  `compile_metal_source` applies) and the device (ordinal and registry id: a library and its
  pipelines belong to one `MTLDevice`). The cached value is a `program`: the library plus a table of
  per-symbol `(MTLFunction, pipeline)` created LAZILY on a kernel's first link (`pipeline_of`), so
  a kernel whose pipeline the driver refuses (the gh-1226 threadgroup overflow, a normal decline)
  costs only its own retry, never the group's library. The launch-dependent checks re-run on every
  link against the reused pipeline. The per-code `resolved` memo (gh-1440) still makes a relink
  resolve nothing, and `Utils.count_backend_compile` fires once per memo miss whether the kernels
  hit or not -- `deferred_hit_compiles`' `one_compile` is 2 on Metal and must stay so, since the
  winner's in-process recompile now hits the cache. Device debugging bypasses the cache (a GPU
  capture shows the library's source, another routine's text for a reused kernel).
  `Compile_cost.pipelines_created` counts creations; `gpu_kernel_cache` pins both levels, gated on
  the census (`reuse_capable`, `pipelines_created > 0`), with `OCANNL_METAL_KERNEL_CACHE_MIB=0` as
  the negative control. Measured on mac-studio (M-series Metal, gpt2 b8 search, 300 s budget, 2
  alternating rounds, `benchmarks/gh1525_reuse_ab.sh`, base 65aae889c vs head): a candidate's median
  compile went from 2.10 / 1.63 s (base) and 1.79 / 1.79 s (head, cache off) to 0.61 / 0.60 s, its
  link from 1.17-1.55 s to 0.27-0.29 s (median 1 of 262 kernels compiled, against all of them), so
  the budget fit 127-129 candidates where it fit 74-92; the winner's compile went from 1.9-2.0 s to
  0.50 s (0 kernels compiled). The tuned step was unchanged (64-66 ms). The base arm's library
  compiles may have been warmed by macOS's own Metal compiler cache; its link time is read from the
  trace as measured.

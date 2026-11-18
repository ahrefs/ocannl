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
  that its parent does not and that are not constants (those go with their last binder, below),
  plus the ancestors a wrapper handed it (`Context.adopt_ancestor`, below) — so sibling contexts are
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
  refusal (`Pre_search_failure` at `Backend_link`, before any cache replay) asks one question:
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
  the nodes failing `Backends.alias_eligible` plus a certified clique per precision class. In a
  loop-free program of fixed-index copies, adds and literal stores to non-scalar buffers, the
  operands and destination of a store coexist through every fission. Elsewhere the largest eligible
  node of each class is the bound: unanchored packing can move a source read to a segment's entry.
  `Seeded_search` charges shared targets where no tile exists for any pair of storage or rounded
  operand precisions at the destination precision. The public default also bounds explicit shared
  fusion with a scalar fallback. A replayable entry with a shared epilogue gets that bound.
  If replay fails, the generated search checks its own floor before trying candidates.
  The size window is narrow in practice: a `Never_virtual` node within `stack_threshold_in_bytes`
  is `Local` in the base anyway. Its probe asks each group's bytes in one request cut at the pool
  cap, decisive under a page-granular allocator that admits by the total of page-rounded requests,
  not against a contiguous-block limit. A floor the device admits says nothing about any
  candidate. A complete candidate carries its pre-transform obligations: a missing charged node at
  compile or a short working or constant group in the link plan is a fatal
  `Schedule_outcome.Link_floor_violation`. A transform that takes storage off a node must extend
  the floor and `link_delta_parity`.
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
  rolled-back run introduced (`fresh_writes`), and `release` retires the keys of every location it
  gives up (gh-ocannl-1738): `Backends.finalize`'s `on_retire` reports each freed pool and each
  trimmed upload-arena tenant as it goes, so a cleanup that raises part way retires exactly what it
  freed. Freed pool ids are never minted again, and a trimmed tail, re-issued at the same location to
  a surviving sibling's next upload, is written by that upload before anything reads it, so a stale
  key was latent rather than a live false vouch: retiring keeps the guarantee local to `release`
  instead of resting on every caller of `upload_slot` writing first. `Context.buffer_written`
  reads the record, released contexts included (`set_values_pool_coalescing`'s reused-tail case,
  `resource_fault_injection`'s partial free). Allocation-time zeros no longer count. A READ-ONLY
  input, embedded or not (gh-ocannl-2023), must be written the same way unless
  `Tnode.zero_valid` declares zero its correct value
  (gh-ocannl-1462): `Tensor` marks every gradient at creation (both sites, `op` and
  `force_param_diff`), `Tnode.mark_zero_valid` marks any other read-only accumulator, and the mark
  never excuses read-before-write state. Before, a read-only input passed once merely allocated, and
  since `compile` allocates every input a forgotten data input read zeros. A node every routine only
  reads may live in a constant pool of its root that compile siblings SHARE, so a sibling-seeding
  test makes it `context_owned` to get a buffer per sibling (`chr_read_only_buffer`). Hand-built
  tests that accumulate from zero therefore seed the accumulator (`peel_census`) or mark it; state
  meant to start from zero inside a model is an embedded output, excluded from `inputs`, and declares
  so with the mark (`{ sgd_momentum; zero_valid = true }`, gh-ocannl-1966). An SGD step compiled without `Train.init_params` never
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
  other (then refused as stale). The literal class keeps that hazard, since `Host_inits` nodes stay
  writable through device copies, and an unmarked one (a broadcast `term_init`) through routines
  (host writes to marked ndarray literals are refused, gh-ocannl-2021, and routine writes to any
  host-constant literal, gh-ocannl-2104; parameterization clears the initializer's literal marker): every write into a `Shared` buffer mints a token (`Context.shared_write_tokens`,
  plus the writer ledger's `observed_tokens`); `run` refuses a `routine.shared_reads` member whose
  current token its ledger did not leave or adopt, a no-op copy adopts only from a source that is
  current, and `rollback_execution` restores both tokens while nothing wrote since. Ordering needs
  no edge: roots on one device share its single FIFO runner (concurrent host domains are outside
  this, as for all of `Context`). Not refused: `to_host`/`get_values` and real copies out of a stale
  source, since reads check no initialization. Lookups (`plan_delta`, `link_floor`,
  `owned_pool_ids`) go through `BI.find_constant`, root first; insertion through `BI.constant_key`.
  Rooted constants pack into pools of their own (`dp_new_root_constants`), and each release unit
  holds the rooted pools its `ctx_buffers` binds (`held_constant_pools`, counted in the device's
  `constant_owners` at the three places a unit is minted: `make_context`, `make_child`, an
  upload fork); the release of the last binder frees the pool and drops its cache entries, so
  fresh roots compiled and released one after another hold memory flat, as the device-wide cache
  did by reuse. Since gh-ocannl-1762 a `Shared` literal pool is held the same way, its count
  spanning roots (`constant_owners`, `constant_pools_of` resolving either key): dropping its strong
  `Tnode.t` key is what lets a released graph's literal node, and its `Host_inits` array, be
  collected. A unit nobody releases keeps its pools, as for working pools. A link that fails
  after `allocate_delta` reclaims the delta's rooted pools no unit holds in `free_delta`, after the
  owned pools (an evicted pool would read as owned and be freed twice), and a failure after its
  `make_child` first gives the discarded child's holds back (`reclaim_constants`'s modes). A new
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
  group nevertheless; both allocation and footprint scoring use `partition_layout_groups`, which
  orders each group by the routine's raw code (`Low_level.layout_order`, gh-ocannl-2051), never by
  uid.
  Two fresh contexts must retain independent parameter/state edits and resets. The marker changes
  allocation only, not the post-virtualization scalar recompute-price inputs.
- **Two classes `release` cannot reach, so "bounded" always needs a qualifier.** (a) A constant
  pool some live unit still binds (every constant, literals included, goes with its last binder,
  above). Constants are bump-packed several to a pool, so one live binder of any entry keeps the
  whole pool: a hoisted `Stage` candidate's `apply_stage` mints a FRESH packed-constant tnode per
  application (`fresh_tile_id ()`), and a CPU search seeding `hoist` sketches was measured on `cc`
  at 1 → 109 constant pools over 181 candidates while working pools stayed within 2–6, back when no
  constant pool was ever freed; a pool mixing a released candidate's private tile with operand
  weights a live context still binds stays whole. Per-pool purity is gh-ocannl-565's
  eviction-policy work. (b) A link that RAISES after `allocate_delta` — now handled (`Backends.with_delta` frees the
  delta on the way out), but the shape is worth knowing: allocation precedes backend linking, so any
  new failure point between them leaks a whole routine footprint with no context to release it
  through. When asserting a memory bound, assert on `live_working_pools`, not `live_pools`: summing the
  constant class in makes the assertion fail for a reason it is not about, and on a workload with no
  hoisted candidates it passes while proving less than it looks like.
- **A wedged device keeps its pools: a release there refuses rather than frees** (gh-ocannl-2134).
  After a bounded wait (`Backend.await_within`) gives up, the abandoned work may still be writing
  its buffers, so `Raise_backend` marks the device and its `await` raises `Device_wedged` from then
  on. `Backends.finalize` awaits before `free_pool`, so a release on the marked device raises with
  nothing freed and the flag reset (the gh-ocannl-550 rule), and every pool of the process on that
  device leaks until exit: deliberately, since freeing memory under a running kernel is worse.
  Recovery (gh-ocannl-2133) is what would end the leak. The mark covers the wrapped `await` only:
  the lower layers' own `Device.await` calls and event-level `sync` do not see it, and on Metal the
  runner's `abandoned` latch covers those.
- **A dropped graph is held by device tables, never by the session** (gh-ocannl-1747). Since
  gh-ocannl-1745 the session lets an evaluated graph go, and the only strong holders of its nodes
  were two device tables: `updating_for` (every node a routine or upload wrote, now pruned of done
  events at each write, since its one reader treats a done event as absent) and the `Shared` keys of
  `constant_buffer_cache` (host literals, dropped with their last binder since gh-ocannl-1762). A
  held node is a host-memory pin, since `Host_inits` is an ephemeron keyed by the node. Pools are a
  separate question: the pool table holds them, nodes or not, until `release`. The analysis cache
  keeps eight compiles' nodes, bounded. To find a pinner, put weak witnesses on the nodes and
  clear candidates one at a time
  (`clear_analysis_cache`, `unsafe_reinitialize`, one more graph's writes -- which prune
  `updating_for` -- the constant cache): the recipe that found these two, and
  `dropped_graph_buffers` pins what it found. There is no device-wide event reset to call instead:
  an idle device's entries are all done, so clearing them changes no ordering and only frees early
  what the next write prunes, while resetting `updating_for_merge_buffer` would make a consumer
  scheduled after the sync fail `check_merge_buffer` against a transfer from before it. A
  device-memory claim reads the process's own accounting (`Alloc_census`, the used gauge), never
  the driver's free bytes: device-wide, so any concurrent GPU process -- the suite's own parallel
  GPU tests included -- moves them between two readings (gh-ocannl-1798); print them on stderr only.
  Host arrays have their own counted reading, `Ndarray.get_used_memory`, a gauge a finalizer gives
  back: it covers `Host_inits` data, so "the literal's host array is reclaimed" is a byte claim
  after a settle, with a memory-mapped array outside it (gh-ocannl-1859).
- **Per-routine ledger history ends at its last live release unit** (gh-ocannl-2063).
  `Context.ledger_census` separates table entry counts and reachable words; the latter overlap
  where values are shared and must not be summed as heap usage. `root_ledger` measures the
  whole heap with `Gc.stat` separately, after settling and clearing the bounded analysis cache.
  `routine_units` records each release unit's immutable routine ancestry, and `routine_holders`
  counts its holders. A compile descendant or upload fork takes its own hold; record copies in
  one release unit share it. Final release retires names, executed ids, per-run rollback records,
  gradient-buffer records, and diagnostic fragment/zeroing owners. Accumulation counts forget
  retired ids in both current state and surviving rollback snapshots. This is NOT rollback:
  values written into surviving ancestor buffers and their write tokens remain valid. Their
  runtime writer becomes `Released_writer`, a buffer-lifetime readiness marker without a routine
  id: deleting it would make a merge copy fall back to an unexecuted sibling's compile frontier
  even though the buffer holds the released writer's values (`root_ledger` pins that case).
  A partial release keeps its holds for retry; a committed release (including a later hook failure) retires
  them once. Live ancestry, buffers and the current merge fill still retain their own records.
  Hash-table backing capacity may retain its high-water mark; the bound is on repeated
  compile/release at fixed live ancestry. A third loop accumulates into a surviving ancestor's
  gradient, verifying that its values survive retirement; disabling just accumulation-count
  retirement makes that table's bounded-growth claim fail. Restoring a saved accumulation
  history filters retired routine ids and freed buffers as well, so a later restore cannot
  resurrect the records: the third loop restores a saved history after each release, while
  `never_zeroed_grad` pins restoration of a live routine's counts.
  Guards: `root_ledger`'s three 256-cycle loops, per-routine table counts after a fork/failed
  release, a dependent consumer of the fork's already-executed producer, and cleanup retry;
  `context_handle_refusals`, `test_execution_deps`, `resource_fault_injection`.
- **A context a wrapper mints and does not return is adopted by the one it returns**
  (gh-ocannl-1762). `Train.run_once`'s initialization context (the parameters' pools) and a memory
  budget's decision context (rooted constant holds) are lifecycle ancestors of the returned context
  that no caller holds, so `release` of the result never reached them and every `forward_once` in a
  fresh-graph loop left one parameter pool behind. `Context.adopt_ancestor` hands one to the result:
  a hold counted per release unit (`minted`, shared by the unit's values through record copies),
  taken again by an upload fork in `derive` -- a fork keeps the source's lifecycle parent, so it
  binds the ancestor's buffers too -- and not by a compile child, which is released first. The last
  holder's `release` releases the ancestors, deepest first; a failure there is resumed by the next
  `release` of the unit. A new wrapper that compiles from a context it minted adopts it the same way
  (`Train.adopt_minted`), skipping the caller's own unit (`Context.same_release_unit`). Guard:
  `dropped_graph_buffers` (the forward_once and fork cases; the fork case's claims fail when the
  fork does not take its hold). A wrapper owes this on its raising paths too (gh-ocannl-1859): what
  it minted is released when no caller will receive it, through `Train.release_on_raise`, which
  skips the caller's unit and never replaces the original failure. A search that decides contexts
  as it goes holds them until it has shipped, since a fallback's deferred compile still needs its
  decided base: `Train.settle_decided` then adopts those on the shipped chain and releases the rest,
  newest first, and `Autotune.model_default` does the same for its pick. A decided child holds no
  pools but takes constant holds, so a dropped one pins the root's constant pools.
  `Context.decide_readings` mints one child for its three readings, so an adopter holds the whole
  decision. Guard: `wrapper_release_paths` (arm B's case fails with the adoption alone removed).
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
- **A CUDA launch can be refused device memory with no buffer in sight** (gh-ocannl-1773). The
  driver backs a kernel's local memory (its PTX `__local_depot`: `Local` arrays, spills) for every
  thread the device keeps resident -- depot x threads per multiprocessor x multiprocessors, whatever
  the launch geometry. Measured on rog-nv (46 x 1536 threads): `schedule_strided_1x1`'s unscheduled
  resnet-block reference (s11d_ref), ONE thread with a 48 KiB depot (three 16 KiB `Local` nodes,
  each exactly at `stack_threshold_in_bytes`), reserves 3.3 GiB, and under a second GPU batch
  `cu_launch_kernel` answered `CUDA_ERROR_OUT_OF_MEMORY` (the 2026-10-02 and 10-31 flakes). The
  kernel launch, and since gh-ocannl-1872 CUDA and HIP graph instantiation (where a graph's
  kernels first reserve: a hopeless CUDA frame is refused there) and HIP's kernel launch, back off
  over 15.5 s (`launch_with_memory_backoff`; a stderr line per refusal states the reservation),
  CUDA at once re-raising when the reservation exceeds total memory. An instantiation refused
  memory runs the segments plainly that once. The exception stays the binding's own: the sweep's
  serial rerun keys on its text, and CUDA's `classify_failure` declines it `No_device_writes` in a
  search (HIP's does not classify it). The global debug info counts `launch_memory_retries` and
  `launch_memory_refusals`.
- **A refused graph LAUNCH is never retried on its exec** (gh-ocannl-1872). Measured on rog-nv
  under a neighbour's hold: after `cuGraphLaunch` of a cached exec answered
  `CUDA_ERROR_OUT_OF_MEMORY`, its retry was accepted with the memory still held, and the replay
  faulted (`CUDA_ERROR_LAUNCH_FAILED`, sticky): the driver relaunched without regrowing the
  reservation, while plain launches and instantiation kept refusing until memory was free. So
  `sequence_segments`' `launch_graph` sets the stack limit back to the baseline (re-syncing the
  driver's view of the reservation), drops and destroys the exec, runs the segments plainly that
  once, where each kernel launch backs off, and captures afresh on the next run. HIP drops the
  exec the same way, with no limit to reset. Counted as `graph_launch_memory_drops`; no
  `cu_graph_launch` or `hip_graph_launch` exception escapes for memory.
- **The driver keeps a grown local-memory reservation until the context's stack limit is set
  again** (gh-ocannl-1872). Nothing shrinks it on its own, so one s11d_ref launch used to pin its
  process at +3.3 GiB until exit, starving `-j 8` neighbours. `cuCtxSetLimit(STACK_SIZE, baseline)`
  releases it in full. Measured on rog-nv's RTX 5070 Ti (11.8 GB, 70656 resident threads, the
  48 KiB frame): the reset frees +3244 MiB, whether the kernel ran plainly or inside a graph whose
  exec stays alive (the cached replay regrows it). It takes 1.5 ms on an idle device and BLOCKS
  until the context's preceding work completes, and the next relaunch regrows the reservation for
  about 4 ms (10 launch+sync rounds: 7.5 ms plain, 75 ms resetting after each). Setting the limit
  to its current, grown value frees nothing: only lowering it releases memory.
- **So the reset is gated, and never runs mid-capture** (gh-ocannl-1872). `bound_local_memory`
  (`cuda_backend.ml`) resets after a successful submission only when depot x resident threads
  passes a sixteenth of total memory: the large-depot class, where a host wait and a few
  milliseconds buy gigabytes for every neighbour, while ordinary launches stay asynchronous. It runs
  outside the back-off's handler, so its failure is never retried as a refused launch. During a
  stream capture the kernel tasks only record their depot into a per-domain collector, and the
  graph launch resets for them. The depot is a lower bound, so a frame that is all JIT spills is not
  gated. Counted as `local_memory_resets`; the share itself is reported as
  `local_memory_reset_share`, which tests size their frames from. HIP has no counterpart: its scratch is budgeted per
  work-item at link (below), and `hipLimitStackSize` does not govern it.
- **A one-thread CUDA kernel can put its private arrays in device scratch** (gh-ocannl-2263).
  The final grid and block dimensions must each be all ones. When the ordinary private arrays'
  aligned frame exceeds 16 KiB, codegen moves the entire frame into one scratch argument, including
  arrays smaller than the budget. It preserves their logical placement and clears each slice whose
  original declaration would have initialized it. The owning stream orders allocation before the
  launch and free after it. Capturing those operations creates graph allocation and free nodes, so
  cached replay preserves the lifetime and independently linked uses can share the kernel code.
  Plain scratch allocation has its own memory back-off before kernel submission. Capture records
  the allocation with no retry inside capture. The compiler enables this ABI on a device set with
  stream-ordered memory pools. The reset and
  back-off still use the emitted PTX depot for multi-thread kernels and for other remaining local
  memory. Separate per-thread slices and total-size accounting would be needed for multi-thread
  scratch. The CUDA regression checks three live slices on repeat execution, with stale
  initialization distinguished by the captured replay. A one-thread array at the cutoff stays
  in the depot, and three smaller arrays just above the combined cutoff use scratch.
  On rog-nv-linux's RTX 5070 Ti Laptop GPU, with an exclusive hold and no compute co-tenants at
  open or close, two real `s11d_ref` submissions at each revision recorded these driver readings.
  On `028856294`, the PTX depot was 49152 B and free memory fell by 3356 MiB at both submissions,
  before the automatic reset. On `b7e817928`, the aligned source declarations requested 66304 B
  of scratch and the PTX depot was zero. The first allocation took 32 MiB of driver backing,
  the second reused it, and neither submission grew the reservation. Requested scratch counts
  the source declarations, while the base depot reflects NVRTC's optimized PTX.
  Run records on rog-nv-linux: `20261117T223304Z-2571991` (base) and
  `20261117T223402Z-2578262` (scratch). In the same runs, the 25 ms sampled process peaks fell
  from 3516 to 192 MiB for `schedule_strided_1x1` and from 1468 to 608 MiB for
  `tile_mma_narrow`. These peaks include memory beyond the individual launch's reservation.
- **Plain CUDA scratch and graph allocation nodes use separate allocators** (gh-ocannl-2323).
  Plain scratch uses the stream device's current pool, which OCANNL leaves at the device default.
  Graph backing can remain reserved after the last exec is destroyed. CUDA's
  `get_global_debug_info` reports current backing in bytes as `stream_pool_reserved_bytes` and
  `graph_pool_reserved_bytes`, each summed across visible devices or `unavailable` if a read
  fails. These driver snapshots preserve the requested-slab meaning of `get_used_memory`.
  On rog-nv-linux, two overlapping 66,304 B requests used 132,608 B and shared 32 MiB of the
  default stream pool's backing. Its 0 B release threshold let synchronization return that
  backing in the probe. Graph backing stayed at 32 MiB across cache eviction and after release
  dropped the live exec count to zero. The raw readings and exclusive run record are on
  gh-ocannl-2323. The two opt-in modes of `test/operations/cuda_launch_reservation` reproduce the
  workloads with the driver's current and high-water attributes alongside free memory.
- Guard for the depot path: `test/operations/cuda_launch_reservation` declines a hopeless two-block
  frame and returns the free memory after a two-block kernel with 48 KiB per thread, plainly and in
  a graph. It brackets the reset threshold and runs HIP's scratch frame.
  By hand, under an exclusive hold, `--contended` outwaits a child process holding the memory in device buffers (a frame reservation is now
  released at once) for a plain launch, a graph's first instantiation and a cached replay, whose
  refused exec must be dropped. On HIP (tuf, discrete, ROCm) the same hold refuses nothing: ROCr
  backs scratch for the waves actually launched, on demand, so a one-work-item frame whose
  full-occupancy scratch would be 2 GiB launched at once with 380 MB free -- HIP's launch back-off
  is in place but no run has reached it, and `--contended` pins the non-refusal. In a shared batch
  that child refused its neighbours. `schedule_strided_1x1` claims the real s11d_ref runs without a
  local-memory reset on CUDA devices that would have reset the base's depot.
- Raw-lowering test audit (gh-ocannl-2219): a `lowered_transform` that captures IR for a structural
  assertion need not launch that IR. `online_softmax_block`'s `forward` and `training`,
  `online_softmax`'s `training`, SGD `trajectory` and leaf-gradient legs, and `fused_classifier`
  capture before applying default schedules. Their value claims use scheduled execution.
  `online_softmax.training` also retains a compile-only raw routine for its unscheduled interface
  assertion: fission's promoted scratch makes the scheduled interface a different set. Its value
  routine is compiled as a sibling from the initialization context (compiling the same fragments
  from a descendant is correctly refused by the lineage guard).
  `build_order_programs.observe` snapshots the unscheduled source, placements, digests and link
  layout before compiling a separately named default-scheduled GPU routine for its nonzero,
  changed-program and cross-creation-order loss comparisons. The loss comparisons use the sibling
  executions, each retaining its build's creation order; the raw snapshots pin source and layout.
  Its CPU and explicit scheduled arms retain their original runs.
  Kept raw GPU launches: `build_order_programs.bump_pair` reads both outputs and the input to
  validate the bump allocator's layout/value relationship; `tile_mma_narrow.run_census`'s serial
  references are the independent per-operation rounding oracle for its explicitly packed/tiled
  executions (including the inexact bf16 case); and `schedule_strided_1x1.run_plain` references
  are the unscheduled numerical oracle for whole and per-segment candidates. In particular
  `s11d_ref` remains the launch-memory acceptance witness. Explicit candidate/packing schedules
  in the latter two tests are also retained: they exercise those schedules rather than merely
  capturing IR. Declined `schedule_strided_1x1.candidate` fallbacks are now compile-only:
  `run_with` returns no readback when the transform recorded a decline, and releases both contexts.
  Compile-only captures and declined-schedule probes in the other tests need no conversion.
  Specifically, the retained serial rounding routines are `nrw_bf16_serial`,
  `nrw_bf16_inexact_serial`, `nrw_half_serial` and `nrw_f16_serial`. The retained strided references
  are `s11w_ref`, `s11r_ref`, `s11t_ref`, `s11d_ref`, `s11o_ref`, `s11v_ref`, `s11h_ref`,
  `s11u_ref` and `s11p_ref`; the `s11c` unit-stride control is compile-only already.
  One-second `nvidia-smi` baseline samples on rog-nv-linux, native PTX at `-j 8`, reproduced
  peaks of 4468/3234/2444/1470/3964/3516 MiB for `online_softmax_block`/`build_order_programs`/
  `online_softmax`/`tile_mma_narrow`/`fused_classifier`/`schedule_strided_1x1`; test-run record
  `20261116T155544Z-407184`, base `608c490e7`. After the audit, on `f0e652488`, the same six
  processes peaked at 160/168/164/1470/170/3516 MiB respectively, record
  `20261116T162527Z-1163468`. Both native-target batches ran at `-j 8` under exclusive holds,
  with no GPU co-tenants at open or close. Raw samples are on rog-nv-linux under
  `/home/lukstafi/wave-scratch/wave-20261116-ocannl/t2219/{baseline-native-v3,after-native-v1}/gpu-samples.csv`.
  These are sampled process peaks, not aggregate simultaneous use or timing measurements.
  The two retained controls' peaks stayed unchanged. All six audit tests also passed on Metal on
  macbook-pro at `c2d07e7c3`, record `20261116T173946Z-58623` (cache disabled, `-j 4`).
  Do not use `dune --force` as evidence of fresh
  test actions: the first audit attempt ran no tests. Disable the shared cache and change an
  action dependency, then confirm executable actions in the trace and process samples.
- For an opt-in Linux CUDA census (gh-ocannl-2291), run `env OCANNL_BACKEND=cuda python3
  tools/gpu-memory-census.py -- --expect-backend cuda build @DIR/runtest-NAME -j 4`. Its
  `gpu-memory-census.json` sits in the foreground run record. The timing reader shares its
  allowlisted process rows before the runner deletes the raw trace. Readable polls inside an
  owned PID's traced lifetime give its sampled process peak and count. Unseen rows stay unknown,
  unread probes say UNREAD, and a cached batch says when no test ran.
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
  its negative control). Tests force the policy with `Kernel_code_cache.with_split`, and start
  cold with `Kernel_code_cache.with_cold_caches` (every cache emptied, put back on exit;
  gh-ocannl-1700).
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
- Cache identity also needs width-independent layout. The pre-transform link-floor snapshot
  added a zero-elision pass that mints extra scope ids. On CUDA, longer scope and node names
  together wrapped two accumulation expressions in `gpu_kernel_cache`'s offset build. The
  ordinary emitter's key uses the compact document so renaming starts with the same layout.
  `kernel_cache_key_wrapping` renders fixed scope ids in the pure C dialect and pins both the
  extra break in the pretty source and the equality of the two keys.
- **A fresh-build cache miss can be a free-name collision in the index namespace**
  (gh-ocannl-1932, gh-ocannl-2214). The forced node-offset probe reproduced a miss whose two
  `render_batch.key_text` dumps differed at loop symbol `i386`: it was left verbatim because
  `system_macro_idents` includes the GNU C 32-bit x86 macro, while neighbouring symbols were
  canonicalized. Earlier autotune searches consume a timing-dependent number of symbols, so
  ordinary repeats can miss this one number. `Indexing.symbol_ident (Symbol 386)` now spells
  `i0386`, still in the reserved numeric index namespace and injective. Do not remove the
  canonicalizer's free-name protection: on a target defining the macro, that declaration would
  not compile. `kernel_code_cache` pins the exact collision; `gpu_kernel_cache` executes fresh
  builds separated by node allocations and reports cc cache stats on failure. The cc cache is
  process-local: parallel test executables cannot populate or evict it. In the reproducer there
  were no evictions; compare snapshots after the first and second build to distinguish a
  future key miss from capacity pressure.
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
- **What an all-hit Metal link still pays is key building, not pipelines** (gh-ocannl-1675). Read
  it with `bench_gpt`'s `BENCH_RECOMPILE=K` (fresh all-hit compiles of the step, each census on
  stderr) and `Compile_cost.link_delta_s` (the `allocate_delta` share of `link_s`, which every
  backend pays); `sample` attributes OCaml frames poorly here (broken stacks, GC frames), so time
  phases rather than trust its tree. On gpt2_mini_train b8's default step (179 kernels,
  mac-studio, correctness-grade): link 0.117 s, of which the per-kernel `link_proc` work (pipeline
  lookup, launch checks, slot-table buffer) was about 0.003 s, the zero fills 0.026 s, and
  `key_text` about 0.089 s -- two thirds of it `prepend_builtins_to_text` scanning each kernel's
  whole assembled source a second time for conditional includes. Two fixes: the prelude reads its
  conditional includes off the joined scans of its parts (`Source_tokens.joined`, `scan_static`,
  `mentions_indexed`; codegen on every backend got cheaper too), and `allocate_delta` hands all
  of a delta's zero ranges to one `zero_ranges` call -- Metal's `memset_zero` waited a GPU round
  trip (~0.5 ms) per node. Together: link 0.117 -> 0.080 s, delta 0.023 -> 0.013 s. The rest is
  `canonical_kernel` (~0.025 s), paid at compile on the other backends.

## Host transfers and page-locked memory
- **A pinned staging pool does not pay; page-locked host arrays do, on rog only** (gh-ocannl-170,
  `benchmarks/gh170_transfer.sh` over `benchmarks/gh170_transfer_probe.c`, 15 reps, at 367043e6).
  `Cuda_backend`/`Hip_backend` `from_host`/`to_host` copy straight from the pageable bigarray. At
  256 MiB, host-to-device / device-to-host GB/s, pageable vs page-locked vs staged through one
  pinned buffer vs a two-buffer 4 MiB chunked pipeline (the bounded pool's shape) vs the bigarray
  after `cuMemHostRegister`/`hipHostRegister`: rog (RTX 5070 Ti Laptop, discrete) 16.4/13.0,
  50.7/22.5, 16.2/11.9, 12.4/11.1, 50.6/21.6; tuf (RX 7700S, discrete) 13.6/14.1, 13.6/14.1,
  8.2/4.5, 12.7/8.3, 13.6/14.2; minix (Radeon 8060S, unified) 47.4/46.4, 47.6/46.2, 16.1/15.4,
  21.6/17.7, 47.2/47.6. The 4 and 64 MiB rows give the same conclusions.
- Staging loses because the host memcpy into the pinned buffer (about 10 GB/s single-threaded on
  rog) is the same work the driver's own pageable path does, now serialized before the DMA. HIP's
  pageable path already runs at the link's rate on both AMD boxes, so pinning buys bandwidth only
  on rog (3.1x / 1.7x). The asynchrony pinning buys (the host returns from the call in about
  10 us) is invisible through `Context.from_host`/`to_host`: both await the device before and after
  the backend copy.
- Registration's one-off cost on rog is about 19 us per MiB (5.1 ms for 256 MiB, 7.3 ms with the
  unregister): about a third of one pageable transfer, so a node transferred every step repays it
  at once. HIP reported a size-independent 3 us on tuf and 1 us on minix, so treat
  `hipHostRegister`'s cost as unmeasured there (likely deferred). cudajit 0.8.0 binds neither
  `cuMemHostAlloc` nor `cuMemHostRegister`, so the registration route starts with bindings.

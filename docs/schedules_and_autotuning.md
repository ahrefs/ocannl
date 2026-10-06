# Schedules and Autotuning

OCANNL separates *what* a routine computes from *how* its loop nests execute. The "how" is a
**schedule**: a list of transform values (`Schedule.optop`) applied as a pure
`Low_level.optimized -> Low_level.optimized` pass. Schedules parallelize kernels onto hardware
axes, tile and stage operands through scratch memory, hand matmul micro-kernels to tensor cores
and SIMD register tiles, and are the objects the autotuner searches over and caches.

This is the primary documentation of the schedule layer and its search. The normative per-op
contracts live in the interface, [schedule.mli](../arrayjit/lib/schedule.mli) — this document
gives the system view: the pass-ordering contract, the transform vocabulary, the composition
recipes, the automatic presets, legality, and the autotuner. Design rationale and the decision
history are in [docs/proposals/](proposals/) (notably
[schedule-ir-optops.md](proposals/schedule-ir-optops.md) and
[tensorize-mma.md](proposals/tensorize-mma.md)); proposal files record evolution and are not kept
current — when they disagree with this document or the interfaces, the latter win. For the
big-picture direction of the compiler, see the
[compilation manifesto](compilation_manifesto.md) and the design-space essay
[Two Ways to Tame a Schedule Space](blog/tiramisu-telamon-optimization-space-pruning.md).

## The seam and the pass-ordering contract

Schedules run **after** the whole `Low_level.optimize` pipeline — virtualization, inlining,
simplification, CSE — at the `?lowered_transform` seam of backend `compile` (reachable from user
code as `Context.compile ~lowered_transform`). Consequences:

- Transforms see the final, fused code: what they split and stage is what will render.
- There is **no re-virtualization**: a schedule may synthesize new statements (staged loads,
  accumulator transfers) but never re-runs the virtualization decisions.
- Guards follow the **construct-then-fold** discipline: transforms freely emit `If` and scalar
  `Where` guards (split remainders, staging edge guards, pad masks), and `Schedule.apply`'s
  trailing `simplify_llc` interval-folds every guard the loop extents prove. When a transform
  duplicated code (materializing `Unroll`, `Partition`), `apply` also re-runs CSE and
  cross-statement hoisting.
- Schedules are **values**: they serialize (see `Schedule_cache`), replay across compiles of
  structurally identical code, and compose left to right. Loops are addressed by their index
  symbols; smart constructors (`Schedule.split`, `partition`, `tensorize`, `expand_zero`) mint
  and return the fresh symbols so subsequent ops can reference the loops they create.

When no `lowered_transform` is given, backend `compile` applies the default annotator presets
with kernel fission (below).

## The transform vocabulary

Structural rewrites:

| optop | effect |
|---|---|
| `Split { axis; factor; outer; inner }` | `i` becomes `i_o { i_i }` with `i := factor*i_o + i_i`; ceil extent plus a remainder guard when the factor does not divide; the outer/inner axis types annotate hardware axes in the same step. |
| `Swap { outer; inner }` | Interchange a perfectly nested pair (reordering licensed for the accumulation patterns lowering emits). |
| `Retype { axis; ty }` | Change a loop's axis type in place: `Serial`, `Grid`, `Workgroup`, `Workgroup_reduce`, `Unrolled`, `Vectorized`. |
| `Unroll { axis; materialize }` | Annotation flavor (codegen repeats the body) vs. IR flavor (substitute constants so simplify/CSE see the copies — register blocktiling is `Split` + materializing `Unroll` + CSE). |
| `Partition { axis; breakpoints }` | Index-set splitting (gh-ocannl-508): consecutive absolute-range segment loops, each a body copy under a fresh symbol; per-segment interval folding specializes the guards each segment decides. `Schedule.partition_breakpoints` derives breakpoints from the guards already present. |
| `Pad { axis; to_multiple_of }` | Pad-to-tile (gh-ocannl-485, PADTO): extend a Serial loop to the next tile multiple, guarding each effectful leaf statement with `If (axis < N)` — pad iterations are no-ops, downstream `Split`s divide cleanly, and `Tensorize` recognizes the guards as maskable. |

Code-synthesizing transforms:

- **`Stage { source; tile_loops; shared; cooperative; hoisted; swizzle; pad_stride; pipeline_depth }`** — stage reads of a
  tensor node through a fresh tile sized by the tile loops' extents. `shared = false` is CPU
  operand packing (a serial copy nest into `Local` scratch, normalizing layout — the tile's axes
  follow `tile_loops` order, so packing untransposes operands); `shared = true` places the tile
  in workgroup-shared memory with a cooperative load nest and barriers;
  `cooperative = Some width` is the lane-aware mode that composes with `Tensorize`;
  `hoisted = true` packs a compile-time-constant operand once at link time into the per-device
  constant pool (gh-ocannl-470); `swizzle` stores the tile XOR-swizzled against shared-memory
  bank conflicts — `Swizzle_elem` permutes single elements and is declined by every intrinsic
  path, `Swizzle_b128` permutes whole 16-byte units and is the layout CUDA's inline-PTX
  `mma.sync` arms read with `ldmatrix` (gh-ocannl-481); `pad_stride` rounds the tile's minor dim
  up to a multiple, widening the row stride (bank conflicts, and layout rules stated on the
  leading dimension) without changing the iterated index space. Edge guards are `Where`-form and store 0 — the add-reduce identity — to
  out-of-range slots, so every staged tile is safe to read over its whole index space
  (tracked in `Low_level.optimized.zero_fringe`; the padding contract of PADTO). A tile axis
  whose source index is a single strided term `c*i` (`c > 1` — a stride-2 conv's GEMM row) is
  *compacted* (gh-ocannl-502): the tile is sized by the loop extent and stored/read at
  coefficient 1, only the load's source index and its edge guard keeping the stride, so the
  packed tile is dense and satisfies `Tensorize`'s unit-coefficient index discipline. Multi-term
  tile parts keep the range-sized layout, and hoisted staging rejects compaction (v1).
  `pipeline_depth = d > 1` software-pipelines a cooperative staging (gh-ocannl-487, an int so a
  search has a dimension; `d = 2` only — both the portable form and CUDA's `cp.async` arm have
  single-step lookahead, so deeper depths await commit-group/wait-group bookkeeping in the async
  arms): the copy for k-block `k+1` issues before the compute of `k` (prologue
  copy before the serial anchor loop, one in-loop barrier instead of two, a trailing barrier;
  pipelined stages sharing the anchor group their prefetches behind that one barrier), and
  codegen allocates `d` rotating tile copies selected by the loop counter
  (`Low_level.optimized.pipelined`; shared-memory accounting multiplies by `d`). The rendering is
  bitwise identical to depth 1 — a pure prefetch-timing transform, searchable without a numerics
  gate. Autotune twins each staged mma/conv seed per depth in the backend's
  `mma_capability.mma_pipeline_depths` (Metal `[2]`, portable form; CUDA `[2]` on sm_80+, where
  the staging copies render as `cp.async` overlapping the compute; HIP empty until its LDS arm) —
  but only for staged operands of at least 4-byte storage, the async arms' element floor: a
  narrower twin could only render the portable synchronous form, paying the doubled shared-memory
  footprint with no overlap to buy back.
- **`Privatize { target; over }`** — contract a materialized accumulator's read-modify-write
  across a reduction loop into per-thread `Local` scratch with one init-load and one store-back;
  recovers for materialized nodes what virtualization gives virtual accumulators.
- **`Expand_zero { tn }`** — expand a whole-node `Zero_out` into an ordinary loop nest so the
  zeroing can be split and annotated with the same geometry as the computation (whole-node
  zeroing of materialized nodes is illegal in multi-threaded kernels).
- **`Tensorize { i; j; k; simd_width }`** — replace a perfectly nested serial `i × j × k` matmul
  micro-kernel (`d[..., i, j] += a[..., i, k] * b[..., k, j]`, plain-add or FMA form, transposed
  operand layouts recognized) by a `Low_level.Tile_mma` block statement under a fresh
  `Workgroup` lane loop, keeping the original nest as the semantically equivalent scalar
  `fallback`. A follow-up pass contracts the accumulator into a fragment tile around the
  enclosing reduction loop (init-load, reduction chain, store-back — the region Metal renders as
  persistent `simdgroup_matrix` fragments). Pad/remainder guards around the micro-kernel are
  recognized: row/column guards move to the fragment transfers as masks, reduction-axis guards
  are discharged against zero-fringe staged operands (gh-ocannl-485).
- **`Fuse_epilogue { target; shared }`** — fold the sole-consumer elementwise tail (bias add,
  activation, residual) into `target`'s store-back site — the fragment store-back, the
  `Privatize` store-back, or the plain accumulation nest — eliminating the tail's separate
  memory pass (gh-ocannl-486).
- **`Split_reduce { axis; target; num_blocks }`** — deterministic two-pass split reduction
  (gh-ocannl-484): parallel reductions without atomics. The Serial reduction loop splits into
  `num_blocks` chunks, each accumulating into its own row of a fresh
  `[num_blocks, target-dims..]` on-device partials node (tile namespace, traced-store
  registered), and a synthesized combine statement folds the partials into `target` in a fixed
  balanced-tree order. The block loop is freely annotatable — its index pins the partials row —
  and the partials producer/consumer pair is exactly the materialized cross-nest edge kernel
  fission cuts at, so under the fissioning flows the two passes (plus, for the scatter form, a
  partials-zeroing pass) compile as separate kernels with the event chain supplying the
  grid-wide synchronization the combine needs. Handles the plain rmw accumulation
  (`⊕ ∈ {Add, Max, Min, Mul}`, FMA) and the gh-ocannl-466 embedding-backward `Set_dynamic`
  scatter (per-block partial gradient rows; within a block, colliding rows stay serial — llm.c's
  deterministic encoder backward, but parallel over blocks).

Rendering of `Tile_mma` is a per-call backend decision with a decline ladder: hardware
intrinsics (Metal `simdgroup` 8×8×8; CUDA wmma 16×16×16, tf32 16×16×8 under the numerics
policy, fp8 `mma.sync`; HIP rocWMMA 16×16×16) → the C backends' register-tiled vector
micro-kernel (tinyBLAS-style C-tile held in vector registers, edges peeled) → the lane-0 scalar
fallback. Declines are always semantics-preserving and observable (`schedule_log_declines`,
`C_syntax.mma_census` — summarized on every compiled routine as `Context.routine.mma`, whose
`tensorization` is `Tensorized` / `Scalar_fallback` / `Not_requested`).

## Composition recipes

The pipelines below are exercised by tests in `test/operations/` (`schedule_*` files) and are
what the autotuner's sketch seeds parameterize:

- **GPU cooperative tensorized matmul**: `Split i, j` into `Grid × Serial` blocks (+ `Split k`),
  cooperative shared `Stage` of both operands at the k-block anchor, `Tensorize` the inner
  triple with the backend's `mma_simd_width`; zero expansion mirrors the grid geometry.
- **CPU packed GEBP**: `Split i` and `k` (optionally `j`), sink to
  `k_o { i_o { micro-kernel } }`, packing `Stage` of the B panel at `k_o` and the A tile at
  `i_o`, `Tensorize` with a unit lane — the register tiling streams contiguous cache-resident
  tiles. Flavors: hoisted (link-time) packing of constant operands, pool-parallel `Grid` row
  blocks, grid-outermost per-chunk re-packing.
- **Implicit-GEMM convolution** (gh-ocannl-493): reorder to
  `outer..; kernel-window..; row; oc; ic`, pack the input's strided-window slice and the kernel
  slice (the packing *is* im2col, one window slice at a time), `Tensorize (row, oc, ic)`; the
  accumulator fragment stays resident across the whole kernel window. GPU leg: `Grid` outer
  loops with cooperative shared staging; row-block flavors for device fill (gh-ocannl-500).
  Strided rows (stride-2 stems and downsample blocks) ride the compacting `Stage` and are seeded
  on both legs (gh-ocannl-502) — the stride only changes the packing nest's load arithmetic.
- **PADTO** (gh-ocannl-485): `Pad` each awkward extent to its tile multiple before the splits;
  staged tiles zero-fill the fringe, `Tensorize` masks the fragment transfers, and the whole
  padded block runs the intrinsic/register-tiled path — arbitrary extents tensorize at the cost
  of `(padded/valid − 1)` wasted flops, a measured tradeoff.
- **Partition specializations** (gh-ocannl-508): partition-then-split for guard-free main nests
  plus a serial epilogue; partitioning a consumer loop at `partition_breakpoints` to converge
  inlined concatenations with segmented rendering. Pad masks are `partition_breakpoints` flip
  points, so partitioning an enclosing block loop at the last fully valid block specializes the
  interior segments guard-free (the pad-vs-peel frame).
- **Clamped windows** (gh-ocannl-504): a padded (`=`-mode) max/tropical-family window spec
  demands no margins — the assignments lowering range-guards the window to the operand's valid
  region (out-of-range positions contribute the accumulation identity, i.e. are skipped; the
  backward argmax scatter gets `If`-guarded writes). The guards mention the window symbol, so
  `partition_breakpoints` bounds it by its loop range and returns both transition points:
  partitioning the output loop there folds the guards everywhere except the truncated boundary
  segments — guard-free full-window interiors, specialized edges.
- **Two-pass split reduction** (gh-ocannl-484): `Split_reduce` on the reduction loop, then the
  fission pipeline (`fission_scheduled` with the default presets, or
  `Context.compile ~lowered_transform`) — the partials edge cuts, pass 1 parallelizes over the
  block loop (and any output loops), the combine parallelizes over the target's elements. Serves
  large single-axis reductions (losses, norms, softmax denominators), the embedding-backward
  scatter, and split-K GEMMs. The parity discipline: for a fixed schedule the parallel execution
  is bitwise-equal to the serial execution of the same schedule (the combine tree is a function
  of the schedule, not of thread timing); exercised by
  `test/operations/schedule_split_reduce.ml`. The autotuner seeds this composition for
  reduction-dominated sites (task 3 of gh-ocannl-484; see Autotuning below).

## The default schedules and kernel fission

With no explicit transform, backend `compile` applies `Schedule.maybe_default_schedules`:

- **GPU preset** (`default_gpu`): for each nest whose parallelism is provable from the lowered
  code alone (every materialized write covered by the loop's index; conservative race analysis),
  annotate one `Grid` and one `Workgroup` loop, splitting by `gpu_schedule_block_size`
  (default 256, clamped to the device's `max_threads_per_workgroup`). Reduction loops stay
  serial. Below `gpu_schedule_min_parallel` (default 64) the nest stays serial. Dynamic
  (`Set_dynamic`/`Get_dynamic`) accesses of a materialized node no longer serialize the whole
  nest (gh-ocannl-484 task 2): the data-dependent component is masked from the affine conflict
  queries, so loops whose index pins a same-position static component of every access
  parallelize (the embedding-dim column of the gh-ocannl-466 scatter, `Split_reduce`'s block
  loop over partials rows), while loops driving the dynamic index are never proven confined and
  stay serial — the deterministic no-atomics invariant, now per-loop instead of per-nest.
- **CPU preset** (`default_cpu`): the same analysis; the outermost parallelizable loop is
  retyped `Grid`, which the C backend renders on a process-global thread pool
  (`dispatch_apply` / OpenMP). Threshold `cpu_schedule_min_parallel` (default 16384).
- **Kernel fission** (`schedule_fission`): top-level statements are partitioned into segments at
  cross-workgroup dependency edges (materialized producer/consumer pairs, bare writes,
  whole-node `Zero_out`s, statements opaque to the analysis); each segment is scheduled
  independently and compiled as its own kernel, run in order with a device event chained at each
  boundary. Aligned cross-nest pairs (identical annotation geometry, per-axis index agreement)
  stay in one kernel; `Local` scratch crossing a cut is promoted to `On_device`; materialized
  zero segments are expanded and annotated on GPU; adjacent unannotated segments coalesce.
- Gates: `automatic_gpu_schedule` / `automatic_cpu_schedule` (both default true), disabled while
  `debug_log_from_routines` keeps logs serial. `Schedule.check_hardware_limits` validates every
  scheduled kernel (workgroup size, shared-memory bytes) against the device limits before the
  driver sees it.

## Legality

`Schedule.op_legality` is the three-valued oracle (gh-ocannl-494): a schedule op is a
thread-pairing transform over the routine's access relations, decided by the `Ir.Affine`
queries — pairing the op's loop symbols as thread identity must leave every write-involving
access pair `Disjoint` or `Same_thread`, with a reassociation license for `Vectorized` retypes,
`Swap`s of accumulations, and `Tensorize`'s reduction. `Op_legal` and `Op_illegal` are proven;
`Op_unknown` means "compile and see" and is never a rejection — the ops' own apply-time
preconditions (which fail loudly) remain in force either way. `legality_crosscheck=true` runs
the retained procedural analyses alongside the affine engine and raises on divergence.

## Autotuning

`Autotune.tune` searches schedules empirically per routine, on-device:

- **Sketch seeding**: the compositions above are seeded directly as parameterized candidates
  (`sketch_params`), because a greedy beam cannot reach them incrementally — a bare `Tensorize`
  loses round 1 before the annotations that justify it can join. Seeds are pre-filtered by
  statically decidable decline rules (precision uniformity, lane widths, per-format intrinsic
  tiles from `hardware_limits.mma.mma_format_tiles`); since gh-ocannl-485, non-multiple extents
  seed `(pad, …, tensorize)` compositions instead of being filtered, for every pipeline that
  stages all its operands. Fused-epilogue twins and per-fission-segment sketches
  (keyed by structural digest) extend the pool.
- **Batch-grid twins** (gh-ocannl-643): a batched (rank-3+) matmul site's GPU seeds each come in
  two batch-geometry flavors — batch loops `Serial` (the historical shape: the row/column blocks
  spend grid slots 0 and 1, batch and head iterate inside every block) and batch loops `Retype`d
  to `Grid`, landing on grid slots ≥ 2 which fold onto the hardware `.z` dimension (the launch's
  `.z` extent is the batch extents' product; each loop binds `(z / stride) % cap` —
  `Low_level.grid_fold`). The zeroing nest and every companion nest carry the same per-position
  annotation with interior batch loops hoisted identically, preserving cross-nest positional
  thread identity. Twins rather than a replacement because block-count curves are non-monotone
  (gh-ocannl-569's clean-room probe peaked near 128 blocks and regressed by 1024): a rank-4
  q/k/v projection at `8×8×128×32` gains 64× the blocks, which the tuner measures against the
  occupancy it costs. Products beyond the CUDA/HIP `gridDim.z` cap (65535) are not seeded.
- **Multi-axis contractions** (gh-ocannl-683): a site contracting over several axes — attention's
  out projection `{ w_o } * attn`, whose weight carries two input axes `(head, head_dim)`, lowers
  to `d[b,s,j] += w[j,h,e] * x[b,s,h,e]`, a reduction nest of two loops — is a matmul site. The
  matcher takes the contraction nest to be the maximal innermost suffix of loops absent from the
  accumulator (lowering orders the reduction loops after the output loops), its innermost loop
  being `m_k` and the rest `m_ko`: k-loops lowering has already split. Every pipeline treats them
  as k-block loops above the one its own k-split mints (`Sketch_families.k_blocks`) — sunk below
  the output roles, the staged tiles reloaded at, the accumulator privatized over the outermost —
  so the tiling machinery is shared unchanged; the tile's k-extent is judged against the
  innermost contraction extent alone (a k-block cannot straddle two lowered loops, since the
  per-axis index maps cannot express a coalesced loop). Before this the out projection was never
  seeded: it shipped as an untiled global-accumulator nest at 8 blocks, 22% of the gpt2_mini step
  on gfx1151 at 9% of sgemm peak.
- **Split-reduce seeds** (gh-ocannl-484 task 3): `Autotune.split_reduce_sites` detects
  reduction-dominated accumulations — a target with at most a few thousand cells fed by a serial
  reduction loop of substantial extent (conv bias/weight gradients, softmax denominators, skinny
  split-K GEMMs; the gh-ocannl-476 sweep measured one such segment at 60–95% of the default conv
  training step on both Metal and CUDA), each candidate axis settled by the hermetic
  `op_legality` probe. Every site is seeded as the two-pass composition above with a few
  `num_blocks` values as the tunable, applied whole-routine *before* fission so the two passes
  land in separate kernels; per-site singles plus one composite recombining the best-timed
  `num_blocks` per site. A split winner's cache entry stores the whole-routine prelude alongside
  the post-prelude per-segment schedules.
  **The enabling interchange** (gh-ocannl-537): as filed, the family reached none of the
  conv-gradient accumulations it was aimed at. OCANNL lowers a parameter gradient with the
  accumulated channel loop *innermost* and the reduction loops (batch, y, x) outside it — the
  inverse of the static form's pinning discipline, which requires every accumulation-cell symbol
  to be bound by a loop *enclosing* the reduction loop. So every axis was rejected, on every
  parameter gradient of the network, while that one segment was 89% of HIP lenet's step. That
  cause, and only that cause, a loop interchange removes: `Schedule.split_reduce_hoist` reports
  the offending symbols from the recognizer's own probe (not from its message), seeding bubbles
  each outside the reduction loop with adjacent `Swap`s — relative order preserved, every `Swap`
  confirmed `Op_legal` against the code it acts on — and re-probes the split on the interchanged
  code. The chain is recorded on the site and replayed by the candidate's prelude ahead of the
  `Split_reduce`. `model_default` deliberately does not propose the
  family: the roofline model prices bytes and flops, not the serialization the split removes, so
  it could only ever rank the split (strictly more traffic) below the default — the family is
  reachable through measurement alone.
- **Beam search**: `autotune_rounds` rounds of width `autotune_beam_width`, each round proposing
  menu actions on the frontier (loop splits, vectorized retypes, tensorization role
  permutations, privatization extensions), compiling and timing candidates
  (`autotune_repeats`, min-of-N with a time floor). Proven `Op_illegal` proposals are pruned
  before compiling.
- **Cost model** (gh-ocannl-491): `Ir.Cost_model` computes a per-kernel roofline estimate from
  byte/op footprints and a calibrated device envelope — used to *rank, not predict*: the
  `autotune_keep_fraction` pre-filter trims the candidate pool, and `model_default_schedule`
  lets recipe-level untuned compiles (`Train`'s wrappers such as `to_routine`/`run_once`, and
  the benchmark runners, via `Autotune.model_default` — not direct `Context.compile` calls,
  which always use the ordinary presets) pick the default-schedule flavor by model, with zero
  timing runs. Counts are deliberately conservative upper bounds; exactness is tracked per
  kernel.
- **Caching**: winners persist in `autotune_cache_dir`, keyed by `Schedule_cache.canonicalize` —
  an alpha-renamed structural digest of the optimized code including dims, precisions, operand
  hoistability, and placement classes — plus the backend name and a digest of the numerics
  policy. The policy is *not* a property of the code (the backends consult `Ir.Numerics` at
  codegen), so without it in the key a default-flags run would report an ordinary cache hit on a
  tf32-tuned tensorized winner and replay it through the scalar fallback — measured at 5.9x
  slower than not tuning at all (gh-ocannl-568). Policies therefore keep separate entries and can
  share a cache directory. Cached schedules rebind their symbols onto the fresh
  lowering at each compile; a digest guard rejects stale entries.
  **Completeness is an invariant, not a habit** (gh-ocannl-572): everything else read *after*
  lowering shares that hazard, so the key carries a codegen tag as well — the backend-independent
  emission gates (`large_models`, `buffer_aliasing`'s `restrict` suppression, the *effective*
  routine-logging predicate — folding in `log_level > 1` so a verbosity bump alone never churns
  keys — and, only once logging reaches the kernel, `prefer_backend_uniformity`'s logging spelling
  and the stream-log routing) plus the compiling backend's own knobs, which it reports as
  `hardware_limits.codegen_tag` (for `cc`: the compiler command and flags, the OpenMP team,
  `cc_vector_bytes`, `cc_fp16_arithmetic`, `cc_parallel_grid`/`cc_parallel_chunks`,
  `cc_grid_private_bytes_cap`; for CUDA and HIP the graph-capture regime, which moves a fissioned
  candidate's launch overhead relative to a whole-routine one, and debug compilation, which only
  their compilers read) — and the CPU worker-pool signature (gh-ocannl-530). The tag also hashes the
  whole `hardware_limits` record, since `backend` is a backend *name*: two GPUs of one backend
  differ in compute capability, mma formats and memory limits, which is what candidate generation
  and rendering read. On `cc` the toolchain is fingerprinted by its own predefined macros under
  the configured flags (`-dM -E`), because `-mcpu=native` is a flag *spelling* — two machines
  targeting different microarchitectures, or one target through two compiler versions, would
  otherwise share a key. Backends hand
  their components to `cache_key` as the whole `hardware_limits` record, so a component added
  there reaches every call site. Every config key is classified against these components in
  `Utils.config_key_classification` — code-borne (it reaches the digest through the code), keyed
  (it must be carried), search-shaping, placement-shaping (it steers `Train.tune_placements`
  above any single search and no `Autotune.tune` call reads it directly, so not even a search's
  trajectory identity carries it), or
  execution-neutral, each with its reason —
  and `test/operations/digest_completeness` fails on a key that is not classified, with
  `digest_identity_flips` calibrating one representative of each class against a real compile
  (`flip_abandonment` pins placement-shaping's exclusion from the search-trajectory identity).
  **Schedule identity pins
  numerics** (gh-ocannl-484): a reduction-reassociating op (`Split_reduce`, `Swap`/`Vectorized`
  over accumulations, `Tensorize`) makes the computed values a function of the schedule — e.g.
  `Split_reduce`'s combine tree is fixed by `num_blocks` — so results are bitwise-reproducible
  as long as the cached schedule replays, but retuning (or clearing the cache, or a digest
  change) may select a different tree and change low-order bits. Pin the cache directory (and
  back it up) where bitwise reproducibility across environments matters.
- **Diagnostics**: `schedule_log_declines` explains why a rendering declined;
  `C_syntax.mma_census` distinguishes "the tensorized candidate lost" from "it never ran
  tensorized", and its per-routine summary (`Context.routine.mma`, surfaced as
  `Autotune.report.best_tensorization` and in every reported timing) is what keeps a tensorized
  label from standing over a scalar measurement; `schedule_log_launches` prints each compiled segment's launch geometry.

## Design position

The schedule layer sits deliberately between the two schools of the tensor-compiler literature
(the full argument, with references, is in
[Two Ways to Tame a Schedule Space](blog/tiramisu-telamon-optimization-space-pruning.md)):

- **Mechanism as a scheduling language** (the Halide/Tiramisu lineage): OCANNL grew the same
  layered separation of concerns independently — a pure, memory-free algorithm (`%op`/`%cd`
  plus shape inference), then execution order and hardware mapping (this layer's optops), then
  memory placement (memory modes, placements, staging), then communication (streams, events,
  merge buffers, explicit host transfers). Each phase leaves the later concerns fluid, which is
  exactly what keeps the transforms composable.
- **Policy over a searchable space** (the Telamon lineage): optop pipelines do not commute, so
  the autotuner treats compositions as seeds — sketches are partial assignments by another
  name — and the roofline bound is used as a filter and an explanation device rather than an
  oracle. The recorded directions on this side: admissible incumbent pruning (a true lower
  bound never discards a candidate that could win), binding-arm diagnostics ("which roofline
  inequality binds, per subspace") as compiler-development signal, and — if seeded beam search
  ever hits a wall — the decision-vector reformulation where legality constraints come from the
  affine engine and all decisions commute.

Specialization is the economic premise: the tuner adapts per shape and per digest,
cache-persistently, which is where searched kernels beat fixed library entry points — while the
last stretch on vendor-flagship shapes (texture paths, undocumented micro-architecture) is
deliberately out of scope.

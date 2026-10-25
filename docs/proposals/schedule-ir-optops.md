# Schedule IR: OptOps-style loop transforms as values

**Date**: 2026-06-12 (stub); elaborated 2026-07-04; implemented 2026-07-05
**Status**: Phases S1–S4 implemented (`arrayjit/lib/schedule.ml`). Promoted to primary
documentation: [docs/schedules_and_autotuning.md](../schedules_and_autotuning.md) — like all
proposal files, this document records the design rationale and its evolution and is not kept
current with the code. Deviations and findings:

- The vocabulary gained `Expand_zero` (not in the original §1): lowering keeps `Zero_out` as an
  opaque statement, and whole-node zeroing is rejected in multi-threaded kernels, so annotated
  matmul schedules must first expand it into a splittable loop nest.
- `Split` carries caller-minted fresh symbols (`Schedule.split` smart constructor) instead of
  minting inside `apply` — programmatic schedules need the symbols for later ops.
- `Stage`'s tile shape generalizes §5: dims derive per source axis from the range of the
  tile-loop terms, so multi-term register-tile decompositions (`TM*i_w + i_t`) stage cleanly.
  V1 shared staging requires tile sizes dividing the extents — `Split`'s whole-body remainder
  guards would put the barriers under divergent control flow.
- The §6 default annotator ships and is on by default for cuda/metal (config
  `automatic_gpu_schedule`, `gpu_schedule_block_size`, `gpu_schedule_min_parallel`); it adds a
  conservative race analysis beyond the §6 sketch (cross-nest producer/consumer bail, per-node
  index-agreement on parallel components).
- Benchmarks (`bin/schedule_bench.exe`, Apple-silicon Metal, 256³/512³ f32 matmul): parallel
  (S1 shape) ≈ 100–130 GFLOP/s, 1800–3000× over the naive 1×1 launch — the #412 >10× criterion
  is met by parallelization alone. SMEM (S2) and register tiles (S3) initially landed at parity
  with S1: every fma round-tripped the output element through global memory.
- **`Privatize` (2026-07-05, also not in the original §1)** closes that gap: it contracts the
  read-modify-write of a materialized accumulator across a reduction loop into a per-thread
  `Local` tile (init-load before, store-back after) — recovering for materialized nodes the
  scope-local form virtualization gives virtual accumulators, and sidestepping the aliasing
  obstacle since a routine-local tile cannot alias kernel pointers (no `restrict` needed for
  the accumulator; [gh-ocannl-164](gh-ocannl-164.md) still helps the loads). Composition order:
  Splits → Stages → Privatize → materializing Unrolls (which turn the tile accesses into
  constant-indexed, register-allocatable form). With it, register tiling pulls ahead of plain
  parallel: 200 vs 184 GFLOP/s at 512³, 145 vs 125 at 1024³ (~16%); cc cache tiles + packing
  improve from 3.3× to 3.8× single-threaded (9.0 GFLOP/s at 256³). The remaining headroom is
  tile-size tuning, vectorized loads, and eventually the BEAM search. Both prerequisites landed on 2026-07-04:
[axis-types-for-loops](axis-types-for-loops.md) (Phases A–C: `axis_type` on `For_loop`,
`Workgroup_barrier`, `workgroup_shared`, the `If` guard statement, hardware rendering,
launch dims, `validate_parallel`) and
[interval-analysis-scalar-t](interval-analysis-scalar-t.md) (guard folding). The
application seam also landed: `?lowered_transform` on `Context.compile`
(`backend_intf.ml:198`, applied at `backends.ml:492`). Seeded by the tinygrad deep dive
([a-range-is-not-its-shape](../blog/a-range-is-not-its-shape.md), port area 1); the
consumer is v0.8 tiling ([#412](https://github.com/ahrefs/ocannl/issues/412)).

## Goal

A schedule layer for `Low_level.t`: loop-nest transforms (Split-with-retype,
Swap/interchange, Unroll, Stage/tile-staging, later Upcast/vectorize and Padto)
represented as *values* — a list of `optop`s — applied as a pure
`Low_level.optimized -> Low_level.optimized` pass at the `?lowered_transform` seam,
Halide-style, rather than tinygrad's destructive mid-pipeline rewrite. A schedule is
then searchable (BEAM over schedule prefixes with on-device timing), cacheable, and
testable independently of the kernel it acts on.

## What the landed groundwork provides

Everything downstream of the transform now exists; only the transform itself is missing:

- **A type to retype to**: `axis_type = Serial | Grid | Workgroup | Workgroup_reduce |
  Unrolled` on `For_loop` (`low_level.ml:42,59`). Annotated kernels render as hardware
  index bindings, launch with real grid/block dims on CUDA and Metal, and fall back to
  serial loops on cc (`c_syntax.ml:484-540`, hooks at `c_syntax.ml:73-84`).
- **A guarded-write statement**: `If of { cond : scalar_arg; body : t }`
  (`low_level.ml:83`) — exactly what Split's remainder tiles and Padto's masks need.
  `simplify_llc` folds interval-decided conditions, so guards that the loop extents
  prove (the divisible-tile common case) cost nothing.
- **Shared tiles and barriers**: `optimized.workgroup_shared : Set.M(Tnode).t`
  (`low_level.ml:227`) renders as `__shared__`/`threadgroup` declarations;
  `Workgroup_barrier` renders per backend; cooperative loads are ordinary
  Workgroup-annotated `For_loop`/`Set`/`Get` code — no new IR vocabulary needed.
- **A safety net**: `compile_proc` runs `validate_parallel` (`low_level.ml:2574`) —
  slot arity, barrier divergence, materialized-write coverage of every active hardware
  dimension — then injects extent guards (`guard_annotated_extents`,
  `low_level.ml:2672`) and computes `launch_dims` (`low_level.ml:2532`). An ill-formed
  schedule fails at backend-compile time with a named error, not a race.
- **A test template**: `test/operations/hardware_axes_parity.ml` and
  `hardware_workgroup_reduce.ml` — run the transformed kernel against its `Serial`
  twin, assert structure from `build_files/` source, dispatch expectations per backend.
- **Signed, bounded indices**: hardware bindings and loop counters are `int32`
  (`int64` under `large_models`), and the per-node numel contract bounds every extent —
  Split's `Affine` index arithmetic cannot overflow the index width.

## Design

### 1. The `optop` vocabulary

```ocaml
(* new module arrayjit/lib/schedule.ml *)
type optop =
  | Split of {
      axis : Indexing.symbol;       (* the loop to split, identified by its index symbol *)
      factor : int;                 (* inner extent *)
      outer : Low_level.axis_type;  (* retype of the new outer loop (Serial = no retype) *)
      inner : Low_level.axis_type;  (* retype of the new inner loop *)
    }
  | Swap of { outer : Indexing.symbol; inner : Indexing.symbol }
      (* interchange two perfectly-nested loops *)
  | Retype of { axis : Indexing.symbol; ty : Low_level.axis_type }
  | Unroll of { axis : Indexing.symbol; materialize : bool }
      (* materialize=false: set axis=Unrolled, codegen repeats the body;
         materialize=true: unroll in the IR so simplify/CSE see the copies (§4) *)
  | Stage of {
      source : Tnode.t;             (* tensor whose accesses to stage through a tile *)
      tile_loops : Indexing.symbol list;  (* loops whose extents size the tile *)
      shared : bool;                (* true: workgroup_shared tile + cooperative load +
                                       barriers; false: per-thread/Local tile (CPU packing) *)
    }

type schedule = optop list  (* applied left to right *)

val apply : schedule -> Low_level.optimized -> Low_level.optimized
```

Axes are referenced by their index symbol: symbols are unique within one lowered
routine (`optimize_proc` keys `reverse_node_map` by symbol), and `Split` mints fresh
symbols via `Indexing.get_symbol` for the outer/inner pair, returning them to the
caller (schedules built programmatically) or recording them in a name environment
(schedules built by search, which addresses loops positionally and resolves to symbols
when applying).

Deliberately absent from v1, mirroring the axis-types enum's restraint: `Upcast`
(vectorize — waits on the `Set_from_vec`/`vec_unop` growth path and a `Vectorized`
axis type) and tensor-core ops. (`Padto` — originally deferred here — landed as
`Pad` once its profitability case, tensor-core alignment, arrived: §3c,
gh-ocannl-485.)
`Cpu_parallel` retyping is **retired** (decision 2026-07-05, recorded in
[gh-ocannl-164](gh-ocannl-164.md)): within-routine CPU threading binds `Grid` axes to a
task pool in the C backend's rendering — `Grid`'s contract is exactly the task-pool
contract, so the same schedules serve GPU and CPU. The exhaustive-match language keeps
late additions safe.

### 2. Pass ordering — the seam is *post*-optimization (decision)

The stub's open question was ordering against virtualization. The landed seam settles
it: `lowered_transform` runs after the whole `optimize_proc` pipeline (virtualize →
cleanup → `simplify_llc` → one-hot rewrite → CSE → `hoist_cross_statement_cse`,
`low_level.ml:3109-3128`) and before `compile_proc`'s validate/guard/launch steps. So
the realized order is:

```
lower → virtualize → simplify → CSE/hoist → SCHEDULE → (local re-simplify) → validate → guard → render
```

Consequences, recorded as the normative contract:

- **The schedule sees fused code.** Virtual (inlined) nodes are already folded into
  the loop nests, so tiling a matmul automatically tiles its inlined elementwise
  epilogue — fusion composes with tiling for free.
- **Transforms must fold their own guards.** `Split` with a non-dividing factor and
  `Stage`'s edge tiles emit `If` guards; since `simplify_llc` already ran,
  `Schedule.apply` finishes by re-running `simplify_llc` (and, when `Unroll
  ~materialize:true` or `Stage` created duplication, `eliminate_common_subexpressions`
  + `hoist_cross_statement_cse`) on its output. All three are pure and idempotent;
  this is the interval proposal's "introducing pass invokes the folder" rule applied
  wholesale. Cheaper and simpler than moving the seam inside `optimize_proc`.
- **No re-virtualization.** `traced_store` is fixed by the time the schedule runs; a
  schedule cannot flip an inlining decision (the bounded re-virtualization iteration
  the stub contemplated is dropped — nothing in the v1 vocabulary needs it; Padto, the
  one op that could, is deferred). Tiles created by `Stage` are fresh `Local`-mode
  nodes born after virtualization, so there is no interaction to police. Note
  `traced_store` must gain entries for staged tiles (they are materialized-per-kernel
  locals); `apply` updates the store the same way `optimize_proc`'s tracing would.

### 3. `Split` mechanics

`For_loop {index=i; from_=0; to_=N-1; body}` with factor `T` becomes

```
For_loop {index=i_o; from_=0; to_=ceil(N/T)-1; axis=outer; body=
  For_loop {index=i_i; from_=0; to_=T-1; axis=inner; body=
    [If (T*i_o + i_i < N)]   (* only when T ∤ N *)
      subst(body, i := T*i_o + i_i)}}
```

Substitution is total over the two places a symbol can occur:

- **Index vectors** (`Get`/`Set`/`Set_from_vec` `idcs`): `Iterator i` becomes
  `Affine {symbols=[(T, i_o); (1, i_i)]; offset=0}`; an existing `Affine` term
  `(c, i)` expands to `(c*T, i_o); (c, i_i)` (`indexing.ml:139-146` — the
  representation was built for this).
- **Scalar expressions**: `Embed_index` (`low_level.ml:112`) carries an `axis_index`,
  so the same rewrite applies.

The remainder guard is `Binop (Cmplt, Embed_index (Affine ...), Constant N)` wrapped
in `If` — and when `T | N`, the interval environment (`i_o ∈ [0, N/T-1]`,
`i_i ∈ [0, T-1]` ⇒ `T*i_o + i_i ∈ [0, N-1]`) folds it away; `apply`'s trailing
simplify does this. Annotated loops require `from_ = 0`, which lowering guarantees
today; `Split` asserts it.

**Legality**: splitting a serial loop is always legal (same iterations, same order
when the nest order is outer-then-inner). Retyping to `Grid`/`Workgroup` carries the
independence obligation — that is the annotating caller's responsibility exactly as
`validate_parallel`'s docs state; the structural half is checked downstream.

`Swap` requires perfect nesting (the outer loop's body is exactly the inner loop);
`apply` fails loudly otherwise. Interchanging serial loops reorders iterations —
legal for the accumulation patterns lowering emits (associative-commutative accums;
bitwise-reproducibility caveats belong to the search harness, which compares against
the unscheduled twin).

### 3b. `Partition`: index-set splitting at static breakpoints (gh-ocannl-508)

`For_loop {index=i; from_; to_; body}` (required `Serial`) with breakpoints
`b_1 < … < b_m` (each strictly inside the range) becomes the sequence of segment
loops

```
For_loop {index=s_0; from_=from_; to_=b_1-1; body=subst(body, i := s_0)};
For_loop {index=s_1; from_=b_1;   to_=b_2-1; body=subst(body, i := s_1)};
...
For_loop {index=s_m; from_=b_m;   to_=to_;   body=subst(body, i := s_m)}
```

Segment ranges stay **absolute** — no rebasing to 0 — so the substitution is a pure
rename and no index arithmetic changes; iteration order is preserved exactly, hence
the op is unconditionally legal (`op_legality`: `Op_legal`). Sibling segments
duplicate the body, so scalar-local scope ids are refreshed per copy (the
materializing-`Unroll` discipline) and `apply` runs CSE + hoisting after a schedule
containing a `Partition`. The fresh segment symbols are returned by the smart
constructor, making each segment individually addressable by subsequent ops
(per-segment scheduling/tensorization).

Specialization needs no logic of its own: each segment's narrowed range feeds the
interval environment, so `apply`'s trailing simplify folds every guard the segment
decides — statement `If`s and scalar `Where` range guards alike. This replaces the
two in-loop-guard workarounds:

- **Split remainders**: partition at the last tile-multiple first, then `Split` the
  (now dividing) main segment — clean main nest + serial epilogue, no `If` anywhere,
  instead of the construct-then-fold remainder guard.
- **Inlined concatenation** (the virtualizer's per-component `Where` range guards,
  gh-ocannl-134): partition the consumer loop at the component boundaries and every
  `Where` folds — converging the inlined rendering with the sequential per-component
  segment nests that materialized concatenation already lowers to
  (`assignments.ml`'s Concat product entries).

`Schedule.partition_breakpoints` derives the breakpoints from the guards already
present in the loop body: comparisons whose two sides differ by `k*axis + off` with
everything else constant flip truth value at one point of the axis range; the
collected in-range flip points are exactly the partition that makes every such guard
segment-decided. Downstream consumers: gh-ocannl-504's clamped-window interior
specialization, gh-ocannl-485's PADTO valid-vs-fringe masking, gh-ocannl-500's edge
peeling.

### 3c. `Pad`: pad-to-tile with masked edges (gh-ocannl-485, PADTO)

`Pad { axis; to_multiple_of }` extends a Serial loop `[0, N)` to `[0, M)` (the least
tile multiple ≥ N) and guards every effectful **leaf statement** of the body with
`If (axis < N)` — pad iterations are no-ops, so the op is unconditionally legal, and
leaf placement (rather than wrapping the body) keeps later-inserted barriers (shared
`Stage`) out of divergent control flow while leaving nests perfectly nested for
`Swap`/reorders. Downstream `Split`s by factors of `M` divide cleanly. The masking
then distributes across the transforms that already exist:

- **Operands** stage through `Stage` tiles whose dims derive from the (padded) tile
  loops; the edge guards are `Where`-form — out-of-range slots store 0, the
  add-reduce identity — so the tile's whole index space is readable (tiles are
  tracked in `optimized.zero_fringe`; hoisted constant packing already zero-fills).
- **`Tensorize`** skims the guards off the micro-kernel and classifies each by its
  micro symbol: `i`/`j` guards become masks re-emitted at the accumulator
  contraction's transfers (0-filled `Where` init-load, `If`-guarded store-back — the
  padded-scratch, valid-region-only discipline), and `k` guards are *discharged*
  against zero-fringe operands (padded contributions are exact zeros). Masks force
  the contraction; on GPU pipelines (recognized by staged shared tiles) the masked
  fragment lands in workgroup-shared memory — guarded transfers are not a
  fragment-resident scope and thread-local arrays are not simdgroup-loadable — so
  intrinsics fire per statement, bracketed by a barrier pair. The fallback keeps the
  guards and stays exactly semantics-preserving.
- **Autotune** seeds `(pad, split, stage, tensorize)` compositions instead of
  filtering non-multiple sites: the staged GPU MMA family, the CPU packed
  compositions, and the conv whole/blocked flavors drop their extent-divisibility
  gates (block sizes must still be intrinsic-tile multiples); pipelines that read an
  operand in place keep theirs. `Split`'s ceil + remainder guard composes into the
  same masked recognition, so a separate pad is only needed for unsplit axes or to
  pick the multiple.

The surviving per-element guards are exactly `partition_breakpoints` flip points: a
later `Partition` of the enclosing block loop at the last fully valid block folds
them in the interior segments (pad-vs-peel as measured alternatives — the peel leg
remains a follow-up). `Tensorize` does that split itself where it pays most
(gh-ocannl-1563): a masked contraction at the statement site under a reduction
loop (the serial CPU GEBP) would round-trip the accumulator per reduction block, so
the block loop is split at the last fully valid block, the full blocks keep the
unmasked `Tile_mma` on the target and only the edge blocks contract. A whole-K site
keeps its single contraction (the store-back `Fuse_epilogue` relocates a tail to),
and so do a `Grid` block loop and a GPU shared fragment.

### 4. `Unroll`: annotation flavor vs. IR flavor

The landed `Unrolled` axis type repeats the body at *codegen* time
(`c_syntax.ml`, per-block constant bindings) — after simplify and CSE have run. That
is right for freeing an index register, but wrong for register tiling, where the
payoff comes from the optimizer seeing the copies: constant-folding `Affine` indices
into `Fixed_idx`, and CSE deduplicating the shared-tile loads that the TM×TN
outer-product form repeats across copies. Hence the `materialize` flag: the IR flavor
substitutes constants (per the `unroll_dims` precedent, `low_level.ml:3467`) and
relies on `apply`'s trailing simplify+CSE to produce the register-blocked form. This
is the "optimized via inlining and CSE" leg of the plan: **register blocktiling is
not a bespoke transform — it is Split + materializing Unroll + the existing CSE.**

### 5. `Stage`: tile staging (the one genuinely new pass)

Split/Swap/Retype/Unroll are structural rewrites. `Stage` synthesizes code:

1. Mint a tile `Tnode.t` (`Local` memory mode — the "uniquely unobservable
   routine-scoped scratch" contract fits exactly), dims = extents of `tile_loops`,
   precision = source's.
2. If `shared`: add it to `optimized.workgroup_shared`; emit a cooperative-load nest
   (Workgroup-annotated loops striding the tile so each thread loads its portion),
   a `Workgroup_barrier`, then the consumer nest with `Get source` rewritten to
   `Get tile` under remapped (tile-local) indices, then a second barrier before the
   next outer-tile iteration. Edge tiles get `If` guards on the loads
   (construct-then-fold as usual).
3. If not `shared` (CPU packing): a plain serial copy nest into a stack tile —
   Böhm's operand packing — no barriers; profitable because the repacked layout is
   contiguous for the micro-kernel.

This is where tinygrad pays with GROUP_REDUCE/LOCAL machinery; OCANNL pays with one
transform that emits ordinary IR. The index remapping is the delicate part: within
the consumer scope, `source`'s index vector restricted to `tile_loops` becomes the
tile's index vector; accesses mixing staged and unstaged symbols in one axis are
rejected in v1.

### 6. Heuristic presets — the default annotator

The schedule layer is also how *every* kernel stops running single-threaded, not just
matmul. A preset (`Schedule.default_gpu : optimized -> schedule`) computes, on the
lowered IR:

- **Parallelizable loops**: outermost loops whose index appears in the index vector of
  every materialized `Set` beneath them — the same coverage property
  `validate_parallel` enforces, used generatively. (Projection information is *not*
  needed post-lowering: output vs. contraction structure is recoverable from where
  indices appear in writes.)
- **Elementwise / outer-parallel kernels**: `Split` the parallel loop(s) by the block
  size (default 256, clamped by the backend's max-threads query), retype outer=`Grid`,
  inner=`Workgroup`. Up to 3 of each kind (slot limit); fold or leave-serial beyond.
- **Reduction loops stay serial** in the preset (one thread per output element);
  `Workgroup_reduce` staging is the opt-in matmul/attention path.
- **Small kernels** (total iteration count below a threshold, default 1024): empty
  schedule — the all-`Serial` kernel launches 1×1 as today.

The matmul preset is the composition from
[gh-ocannl-412](gh-ocannl-412.md): `Split i (BM)` → `Split j (BN)` → `Split k (BK)` →
retype `i_o,j_o=Grid`, `i_i,j_i=Workgroup` → `Swap` k_o outward → `Stage A ~shared` +
`Stage B ~shared` → `Split i_i (TM)`/`Split j_i (TN)` + materializing `Unroll` for the
register tile. Each prefix of that schedule is independently runnable and
parity-testable — which is exactly the property BEAM search needs.

### 7. Kernel fission at cross-workgroup edges

The §6 analysis bails — whole routine serial — whenever sibling top-level nests form a
producer/consumer (or WAW/WAR) pair over a materialized node: threads of one launch have
no grid-wide synchronization, so the pair is a race once the kernel is multi-threaded.
That bail is *the* blocker for real models: any routine with a materialized intermediate
(retained activations, `virtualize_max_visits` overflow), every backward pass
(`Zero_out` of gradients + accumulation), any bare scalar update.

**Aligned cross-nest parallelism** relaxes the bail without a launch boundary: a
dependency component of nests is race-free when every member's chain trims to one common
equal-extent prefix (identical annotation geometry, so a hardware thread covers the same
index slice in every member) and every write/access pair over a shared node aligns per
axis position — plain `Iterator`s at the same chain position on both sides, or parallel
symbols on neither. Each thread then touches only its own elements in every nest and
reads back its own program-order writes, matching serial semantics with no barrier. This
keeps e.g. an elementwise/softmax chain over one batch-parallel intermediate in a single
kernel; a parallelism switch (batch-parallel producer feeding a reduce-over-batch
consumer), a `Get_dynamic` access, or an `Affine` component over a parallel symbol still
bails to fission.

Fission (`Schedule.maybe_default_schedules`, config `schedule_fission`) recovers the
synchronization from the stream instead of the launch: top-level statements are
partitioned into **segments** at the remaining dependency edges (an aligned edge merges
only when the shared kernel keeps every nest's standalone parallelism — alignment may
trim chains, and a launch boundary is the better default than trading parallelism for
one launch), each segment compiles
to its own kernel (`name__seg<i>`, via the backend's batch compile), and the routine's
task launches them in order on the routine's stream, chaining a device-side event
(`all_work` + `will_wait_for`) at each boundary. The event is load-bearing: Metal
command buffers over untracked resources may overlap in execution even on one queue
(caught by `test_random_histograms` — the parallel copy segment raced the serial
threefry segment), so queue FIFO alone is not an ordering contract there. Each segment
then receives the §6 preset
*independently*, so a wide elementwise nest and a narrow bias nest no longer share one
launch geometry, and statements the annotator can never cover (bare materialized
writes, opaque `Staged_compilation`, non-injective nests) are quarantined into serial
1×1 segments instead of poisoning their neighbors. Materialized whole-node `Zero_out`s
are isolated and — on GPU — expanded (`Expand_zero`) and annotated like ordinary nests.
A cut at a materialized edge adds **no** memory traffic (the node already round-trips
through device memory); the only cost is launch overhead, so adjacent segments that
both end up unannotated are coalesced back and an all-serial routine still compiles to
a single kernel.

Two constructs cross segment boundaries and need repair, because a kernel's locals die
at launch end:

- **Hoisted scope-locals** (`hoist_cross_statement_cse`'s top-level `Declare_local` +
  defining statements): the definitions are *replicated* at the head of every consuming
  segment — valid exactly when nothing written between the definition's original
  position and the segment start overlaps the definition's reads; otherwise the
  offending segment range merges back and runs serially (today's behavior).
- **`Local`-placed scratch** whose accesses split across segments is promoted to
  `On_device` (`Tnode.Placements.promote_local_to_device`) — sound because `Local` is
  always a compiler decision premised on single-kernel lifetime, and fission runs
  before any consumer of the decision (codegen parameter lists, context allocation)
  reads it. This in particular unlocks inference-mode chains whose unobservable hidden
  activations resolve `Local`.

CPU backends get the same treatment: the §6 CPU preset has the same cross-nest bail,
and segments there are just separate C functions called in order by the routine's task.
Segment kernels of one routine share one `lowered_bindings` (static-index refs) — the
CUDA/Metal batch link already did; the cc batch link now mints one shared assoc.
Grid-sync via cooperative launch (CUDA-only, occupancy-bound) is a possible later
execution strategy over the *same* segmentation: one kernel with `grid.sync()` at
segment boundaries when launches are numerous and small; Metal has no device-scope
barrier, so fission remains the portable mechanism.

### 8. Search and caching (scoped, deferred)

- BEAM over schedule prefixes, timing candidates on-device; contexts-as-values means
  candidate compiles are sibling `Context.compile` calls from one frontier — no global
  device state to isolate. Correctness gate per candidate: executed parity against the
  unscheduled twin (the `hardware_axes_parity` harness generalized).
- Cache winners keyed by (canonicalized kernel hash, device identity); OCANNL's
  runtime compilation makes per-shape specialization natural.
- Cost models and search policy belong to
  [gh-ocannl-261](gh-ocannl-261.md) follow-ups; v1 ships the presets of §6 plus the
  hand-written matmul schedule, no search.

## Phasing

- **Phase S1 — structural ops**: `schedule.ml` with `Split`/`Swap`/`Retype`/`Unroll`,
  substitution, guard construction, trailing simplify+CSE; unit tests at the
  `Low_level.t` level (structural) plus executed parity via `?lowered_transform`
  (Metal + cc fallback locally, CUDA in CI). Deliverable: the default GPU annotator
  preset — every elementwise kernel launches parallel.
- **Phase S2 — `Stage`**: shared tiles + cooperative loads; the SMEM matmul schedule;
  parity + a first speedup measurement vs. the S1 (parallel, un-tiled) kernel.
- **Phase S3 — register tiling**: materializing `Unroll`, the full matmul preset;
  benchmark against S2 and the naive baseline (the #412 >10× criterion lives here).
- **Phase S4 — CPU**: non-shared `Stage` (packing) + cache-sized `Split` presets for
  cc ([watch-ocannl-README-md-347818d3](watch-ocannl-README-md-347818d3.md) scope);
  vectorization/`Upcast` remains a follow-up (`Cpu_parallel` is retired — CPU
  parallelism is pool-backed `Grid` rendering; see [gh-ocannl-164](gh-ocannl-164.md)).
- **Phase S5 — kernel fission** (§7): segmentation at cross-workgroup edges, per-segment
  default schedules, scope-local replication, `Local` promotion, `Zero_out` expansion;
  routines with materialized intermediates (notably forward+backward passes) stop
  falling back to 1×1 launches. Follow-ups: parallel zero expansion on CPU,
  re-classifying promoted scratch's whole-node zeroing, CUDA cooperative-launch
  grid-sync over the same segmentation.

## Acceptance criteria

- [x] `optop` type and `Schedule.apply` implemented against `Low_level.optimized`,
      with substitution over `axis_index` and `Embed_index`, remainder guards via
      `If`, and the trailing simplify/CSE contract of §2.
- [x] Pass-ordering decision recorded (§2: post-optimization seam, transforms
      re-simplify locally, no re-virtualization) — done in this document; the
      code comments in `schedule.ml` point here.
- [x] Default GPU annotator preset: all-`Serial` elementwise kernels above the
      threshold get Grid×Workgroup schedules automatically; parity suite green on
      cc/Metal locally (CUDA via CI).
- [x] Hand-written SMEM+register matmul schedule passes executed parity
      (`test/operations/schedule_smem_matmul.ml`, `schedule_register_matmul.ml`) and
      beats the unscheduled kernel by far more than the #412 target on Metal
      (~1800–3000×; see the status note — the margin comes from parallelization,
      with SMEM/register tiles at parity pending accumulator privatization).
- [x] Device-limit-aware tile sizes: `hardware_limits` on `Backend_device_common`
      (max threads per workgroup, workgroup-shared memory capacity; min across
      devices, all-`None` on CPU backends) — the annotator clamps the configured
      `gpu_schedule_block_size` to the max-threads limit, and backend `compile`
      validates every kernel (including hand-written `?lowered_transform`
      schedules) against both limits via `Schedule.check_hardware_limits`,
      turning driver-level launch failures into early, named errors.
- [x] Search harness scoped (BEAM first; cost models deferred to #261 follow-ups) —
      §7; implementation deferred.

## Relations

[axis-types-for-loops](axis-types-for-loops.md) (**landed** — supplies the types,
barrier, shared placement, `If`, validation, rendering, launch),
[interval-analysis-scalar-t](interval-analysis-scalar-t.md) (**landed** — guard
discharge), [signed-index-precision](signed-index-precision.md) (**landed** — index
width and extent bounds),
[#412](https://github.com/ahrefs/ocannl/issues/412) (GPU tiling — the consumer;
[gh-ocannl-412](gh-ocannl-412.md) is now the requirements/benchmarks doc, this is the
mechanism doc), [watch-ocannl-README-md-347818d3](watch-ocannl-README-md-347818d3.md)
(CPU matmul — Phase S4), [gh-ocannl-164](gh-ocannl-164.md) (CPU SIMD floor),
[gh-ocannl-267](gh-ocannl-267.md) (Tiramisu — thesis is this layer),
[gh-ocannl-242](gh-ocannl-242.md) (TVM/Ansor lineage),
[gh-ocannl-261](gh-ocannl-261.md) (search/cost functions),
[gh-ocannl-263](gh-ocannl-263.md) (Flash attention — eventual `Stage` consumer).

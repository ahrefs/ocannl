# Axis-type annotations on Low_level loops

**Date**: 2026-06-12 (stub); elaborated 2026-07-03; refreshed 2026-07-04; Phases B+C
landed 2026-07-04
**Status**: **landed** — Phase A (representation) via
[PR #84](https://github.com/lukstafi/ocannl-staging/pull/84) (merged 2026-07-03), Phases
B (rendering + launch) and C (barriers + workgroup-shared) together on 2026-07-04 (see
"As landed" below for deltas from the design). Seeded by the tinygrad deep dive
([a-range-is-not-its-shape](../blog/a-range-is-not-its-shape.md), port area 2); first
site of the decision is [#412](https://github.com/ahrefs/ocannl/issues/412)'s grid/block
mapping. Two sibling landings shaped Phase B's design:
[interval-analysis-scalar-t](interval-analysis-scalar-t.md) (Phase A + B v1, PR #88) makes
the extent-mismatch guards of §2 dischargeable by construction, and
[signed-index-precision](signed-index-precision.md) (core migration, PR #88) fixes the
type of the hardware index bindings in §1/§5.

### As landed (Phases B+C, 2026-07-04)

Four deltas from the design as written below:

1. **The IR gained a guarded-statement constructor** `If of { cond : scalar_arg; body : t }`
   (`low_level.ml`), because §2's extent-mismatch guards were not otherwise representable:
   `Where` is expression-level, and a suppressed *write* (not just a value) is what a
   launch guard needs. Pass contract mirrors the barrier's: inert-descend for analyses,
   `Non_virtual 142` in virtualization, no CSE dedup, a conditional write is never a
   definite write (`reads_scope_before_set`), and `simplify_llc` folds an
   interval-decided condition (true → body, false → `Noop`) — the construct-then-fold
   discipline of §2. Guards are injected at backend-compile time by
   `Low_level.guard_annotated_extents`, only for axis kinds the backend binds in hardware
   (the serial fallback iterates the true extent and needs none).
2. **The annotation seam is `?lowered_transform`** on `Context.compile` /
   `Backend_common.compile`: an optional `Low_level.optimized -> Low_level.optimized`
   applied between lowering and backend compilation. The parity tests hand-annotate (and
   for Phase C wholesale-replace) the lowered code through it; the schedule layer will
   occupy the same seam.
3. **Slot assignment for sibling nests** is formalized as: the slot of an annotated loop
   is the maximum nesting depth of same-kind annotated loops strictly inside it, so
   sibling nests align positionally (`Low_level.hardware_axes`); `Workgroup` and
   `Workgroup_reduce` share the block-slot space. `validate_parallel` additionally
   rejects annotated loops inside `Local_scope` bodies, barriers under `If` guards or
   divergent workgroup extents, writes to materialized nodes not nested under annotated
   loops covering *every* active (non-unit) hardware dimension (launch dims are global
   to the kernel, so an uncovered dimension executes the write once per hardware index —
   per-review hardening of the initial outside-all-loops check), and whole-node
   `Zero_out` of materialized nodes in multi-threaded kernels (nesting never
   distributes it).
4. **`Unrolled` rendering** needs no substitution machinery: the body is emitted
   repeatedly with the index bound as a per-block constant
   (`{ const int32_t iN = k; ... }`).

Tests: `test/operations/hardware_axes_parity.ml` (Grid+Workgroup elementwise parity vs.
the Serial twin, unequal Grid extents exercising the `If` guard, `Unrolled` variant;
structural checks on `build_files/` source, backend-dispatched) and
`test/operations/hardware_workgroup_reduce.ml` (hand-built GROUP_REDUCE: shared tile,
barriers, tree combine — executed parity on Metal/CUDA, clean rejection pinned on cc).
Also fixed in passing: a mkdir TOCTOU race in `Utils.build_file` exposed by concurrently
running tests.

## Goal

Make the loop-to-hardware mapping explicit in the IR instead of a backend convention:
an axis-type annotation on `For_loop` in the spirit of tinygrad's `AxisType`
(GLOBAL / LOCAL / THREAD / LOOP / REDUCE / GROUP_REDUCE / UPCAST / UNROLL —
`tinygrad/uop/ops.py`), so that backends emit grid/block/thread index bindings for
annotated axes rather than loops, and so that schedule transforms
([schedule-ir-optops](schedule-ir-optops.md)) have a type to assign when splitting.

Companion representation decisions bundled here because they share the annotation's
lifetime and consumers: a workgroup-scoped barrier statement in `Low_level.t`, and a
workgroup-shared placement refinement for `Local`-mode tensor nodes (the
tinygrad-`LOCAL`-addrspace analogue).

## Acceptance criteria

- [x] `axis_type` field added to `For_loop` with `Serial` default at every construction
      site; all-`Serial` programs compile to byte-identical kernels on cc, CUDA, and
      Metal — **landed in PR #84 with zero promotions** (the removal of the CUDA
      single-thread guard moves to the launch-plumbing criterion below, where it
      belongs: it is a Phase B behavior change).
- [x] `Workgroup_barrier` statement and shared-placement refinement represented in
      `Low_level.t` / `optimized` — **landed in PR #84**; rendering on CUDA and Metal
      and the clear cc rejection are Phase B/C (today `c_syntax.ml` rejects *all*
      non-`Serial` axes, barriers, and shared placements with explicit errors:
      `c_syntax.ml:456-459,717,1043`).
- [x] Launch dimensions derived from annotations and plumbed to `launch_kernel`
      (CUDA) / `dispatch_threadgroups` (Metal); the `kernel_prep_line` single-thread
      guard is gone — **landed 2026-07-04** (`Low_level.launch_dims`, `compile_proc`
      returns launch dims, backend `code` records carry them; Metal validates the block
      product against `maxTotalThreadsPerThreadgroup` at pipeline creation).
- [x] A hand-annotated executable parity test: the same `Assignments.comp` lowered
      once with all-`Serial` loops and once with `Grid`/`Workgroup` annotations
      produces equal results (Metal locally, CUDA in CI, cc via serial fallback) —
      **landed 2026-07-04**, `test/operations/hardware_axes_parity.ml`, annotated via
      the `?lowered_transform` seam.
- [x] A structural expected-file test showing the generated `.cu`/`.metal` source for
      an annotated kernel (index bindings, `__shared__`/`threadgroup` declaration,
      barrier) — **landed 2026-07-04**: the parity tests assert on `build_files/`
      source (booleans dispatched per backend so one `.expected` holds everywhere);
      `hardware_workgroup_reduce.ml` covers the shared declaration and barrier.

## Context

### Where the code stands (updated 2026-07-04, post-Phase-A)

The representation exists; every backend still emits single-threaded kernels because
the rendering (Phase B) has not landed:

- `axis_type = Serial | Grid | Workgroup | Workgroup_reduce | Unrolled`
  (`low_level.ml:42`) and the `axis` field on `For_loop` (`low_level.ml:59-66`) are in;
  all construction sites default to `Serial`. The human-readable printers emit
  `for@grid` / `for@workgroup` / `for@workgroup_reduce` / `for@unrolled` for annotated
  loops (`axis_type_label`, `low_level.ml:47`) and plain `for` for `Serial` (legacy
  output unchanged).
- `Workgroup_barrier` is a statement constructor with per-pass semantics already
  decided and landed (see §3): inert leaf for analyses, opaque for motion,
  never-virtualizable, never CSE-deduplicated, a hoisting boundary.
- `optimized` carries `workgroup_shared : Set.M(Tnode).t` (`low_level.ml:219`), empty
  until schedule transforms exist.
- `pp_ll` (`c_syntax.ml:452-459`) renders `Serial` loops as C `for` statements and
  raises on any other axis; barriers (`c_syntax.ml:717`) and shared placements
  (`c_syntax.ml:1043`) likewise raise "not supported yet". So annotations are
  representable and printable, but not yet executable.
- CUDA guards every kernel with `kernel_prep_line = "if (threadIdx.x != 0 ||
  blockIdx.x != 0) { return; }"` (`cuda_backend.ml:359`) and launches
  `~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0` (`cuda_backend.ml:1043`).
- Metal already passes `uint3 gid [[threadgroup_position_in_grid]]` and `uint3 lid
  [[thread_position_in_threadgroup]]` to every kernel (`extra_args`,
  `metal_backend.ml:443-446`) but dispatches one threadgroup of one thread
  (`metal_backend.ml:902-908`).
- Loop counters and index kernel arguments are **signed** since the
  [signed-index-precision](signed-index-precision.md) core migration
  (`loop_index_type` = `int32_t`/`int64_t` on cc and Metal, `int`/`long long` on CUDA;
  every per-node padded element count is validated to fit int32 unless `large_models`).
  Hardware index bindings must use the same width (§5).
- `simplify_llc` now threads an interval environment over every in-scope symbol
  (`ienv_extend` reads `For_loop.{from_, to_}` and is axis-agnostic), and folds
  interval-decided comparisons/`Where` guards
  ([interval-analysis-scalar-t](interval-analysis-scalar-t.md)). This is the mechanism
  §2's extent-consistency guards should be built against.

The original question — backend convention versus IR annotation — was settled in
tinygrad's favor by PR #84: the schedule layer's Split-and-retype needs a type to
retype *to*; the kernel before and after scheduling are both printable, diffable
values; and the backend renderers stay dumb — they translate annotations, they don't
decide them.

### Key code pointers (line numbers as of 2026-07-04)

| Location | Description |
|----------|-------------|
| `arrayjit/lib/low_level.ml:42` | `axis_type` definition; `:47` the printer labels |
| `arrayjit/lib/low_level.ml:59-66` | `For_loop` with the `axis` field |
| `arrayjit/lib/low_level.ml:219` | `optimized.workgroup_shared` |
| `arrayjit/lib/low_level.ml:732` | `check_and_store_virtual`: barrier ⇒ `Non_virtual 141` |
| `arrayjit/lib/low_level.ml:1136` | `inline_computation`: barrier unreachable (`assert false`) |
| `arrayjit/lib/low_level.ml:2087-2096` | CSE alpha-equivalence includes `axis`; barrier-containing code never judged equal |
| `arrayjit/lib/low_level.ml:2402-2415` | `hoist_shared_locals` splits sibling segments at barriers |
| `arrayjit/lib/low_level.ml:3133,3182` | `loop_over_dims` / `loop_over_padding_region` — main construction sites |
| `arrayjit/lib/assignments.ml:311,490` | `loop_accum` / `loop_accum_rev` — projection-driven construction sites |
| `arrayjit/lib/c_syntax.ml:452-459` | `pp_ll` `For_loop` case — Phase B's rendering branch point (currently rejects non-`Serial`) |
| `arrayjit/lib/c_syntax.ml:717,1043` | barrier / shared-placement rejections to be replaced by rendering hooks |
| `arrayjit/lib/c_syntax.ml:1035` | `compile_proc` — kernel prologue, local declarations, future slot/launch computation |
| `arrayjit/lib/c_syntax.ml:112-113` | `arg_int_prefix` / `loop_index_type` — the (signed) index width the bindings must share |
| `arrayjit/lib/cuda_backend.ml:359,1043` | single-thread guard and hardcoded 1×1 launch |
| `arrayjit/lib/metal_backend.ml:443,902` | `gid`/`lid` extra args and 1-threadgroup dispatch |
| `arrayjit/lib/tnode.ml` (`memory_mode`) | the 5-constructor lattice (kept untouched by this proposal) |

## Design

### 1. Annotation carrier: a `For_loop` field, not symbol metadata *(landed)*

```ocaml
(* low_level.ml — as landed *)
type axis_type =
  | Serial            (* plain for-loop; the default — today's behavior *)
  | Grid              (* one grid dimension: blockIdx.* / threadgroup_position_in_grid *)
  | Workgroup         (* one block/threadgroup dimension: threadIdx.* / thread_position_in_threadgroup *)
  | Workgroup_reduce  (* a Workgroup axis participating in a shared-memory reduction *)
  | Unrolled          (* fully unrolled at codegen *)
[@@deriving sexp, compare, equal]

  | For_loop of {
      index : Indexing.symbol;
      from_ : int;
      to_ : int;
      body : t;
      trace_it : bool;
      axis : axis_type;
    }
```

Why the loop and not the symbol:

- `Indexing.symbol` is `Symbol of int` (`indexing.ml:8`) — context-free, compared and
  hashed pervasively. Attaching metadata means either changing equality semantics or
  maintaining a global side table.
- A side table is fragile exactly where it matters: virtualization inlines by
  substituting *fresh* symbols, and every renaming would have to migrate table
  entries. A record field travels for free through the
  `For_loop { for_config with body }` idiom used throughout the optimizer.
- The annotation is a property of the *loop* (how these iterations map to hardware),
  not of the index value. Body occurrences need nothing: an annotated loop renders as
  a binding `const int32_t i42 = (int32_t)blockIdx.x;` (at the **signed index width**,
  `loop_index_type` — see [signed-index-precision](signed-index-precision.md); the
  hardware registers are unsigned, but grid/block extents fit int32 by device limits
  and by the per-node numel contract), so `pp_symbol` and every `axis_index` in the
  body are untouched.
- `trace_it` is the precedent: loop-level metadata lives on the loop.

Deliberately absent from the initial enum, to be added when their codegen exists (an
exhaustive-match language makes late additions safe): `Vectorized` (UPCAST — reserved
for the `Set_from_vec`/`vec_unop` growth path), `Cpu_parallel` (THREAD — within-routine
CPU parallelism; today cc parallelism is across streams via the Multicore scheduler),
and any warp-level type. tinygrad's `REDUCE` is *not* ported: reduction-ness of a
serial loop is recoverable from the body (accumulation into a scope or node), and
carrying it would create a redundancy the optimizer must keep consistent.

**Hardware slot assignment** is positional, not stored: among a kernel's `Grid` loops,
the innermost binds `.x`, the next `.y`, then `.z` (innermost-fastest for coalescing);
same rule independently for `Workgroup`/`Workgroup_reduce` loops. More than 3 of a
kind per kernel is a validation error in v1 (the schedule layer can fold axes before
annotating, later). Keeping slots out of the IR means Split/Swap transforms never
have to renumber anything.

### 2. Semantics and well-formedness

An annotated loop means: *iterations are executed by distinct hardware threads; the
loop binds its index to the hardware index rather than iterating*. The loop's extent
(`to_ - from_ + 1`; `from_` must be 0 for annotated loops) contributes to the kernel's
launch dimensions.

Obligations on whatever pass writes annotations (initially hand-written tests and
[#412](https://github.com/ahrefs/ocannl/issues/412)'s heuristics, then the schedule
layer), collected in a `validate_parallel : t -> unit` check run at backend-compile
time:

- **Independence**: iterations of a `Grid`/`Workgroup` loop must have no
  cross-iteration dependencies. `Workgroup_reduce` is the labelled exception: its
  cross-iteration communication must be staged through shared placements and
  barriers.
- **Thread-locality across `Seq`**: a producer statement and consumer statement in
  the same kernel are legal only if the dependence is thread-local under the
  annotation (the hardware thread reads what it itself wrote — e.g. the zero-init
  nest and accumulation nest of one `Accum_op` annotated with identical extents and
  slot order), or workgroup-local and separated by a barrier. Cross-workgroup
  dependencies inside one kernel are ill-formed — there is no grid sync; that
  boundary belongs to kernel splitting (megakernel territory,
  [#318](https://github.com/ahrefs/ocannl/issues/318)).
- **Extent consistency**: sibling nests binding the same slot should have equal
  extents. Launch dims take the per-slot max; a smaller-extent binding is wrapped in
  an `if (i < extent)` guard — but a kernel containing barriers must have *equal*
  workgroup extents everywhere (a barrier under divergent control flow is UB), which
  `validate_parallel` enforces.
- **Barrier placement**: `Workgroup_barrier` must not appear lexically inside a
  guarded/divergent region or inside a `Serial` loop nested under differing-extent
  workgroup axes.

**Guard discharge via interval analysis** *(new since the sibling landed)*. The
smaller-extent guards should be constructed as ordinary IR comparisons
(`Where (Cmplt (Embed_index i, extent), body-value, ...)` or a guarded `Set` form),
not as codegen string concatenation: `simplify_llc`'s interval folding then erases
every guard whose loop extent proves it (the common equal-extent case folds to the
body with no runtime cost), exactly like the gh-343 gather guard re-derivation. The
interval environment already covers annotated loops — `ienv_extend` reads
`For_loop.{from_, to_}` regardless of `axis`, so a hardware-bound index carries its
`[0, extent)` interval like any serial index. Two provisos carried over from the
interval proposal's binding constraints: construct-then-fold is subject to the
fail-safe rule (every emitted conjunct must be correct unfolded at its precision —
trivially satisfied here since extents are non-negative ints at the signed index
width), and if guards are introduced *after* `simplify_llc` in the pipeline, the
introducing pass must invoke the folder locally (the gather rewrite's pattern) or a
post-rewrite simplify pass must be added.

Serialization is always a legal implementation of `Grid`/`Workgroup` *in the absence
of barriers* (running the iterations in order refines any parallel interleaving).
This is what keeps the cc backend sound with zero new machinery — see §5. Barriers
break serialization-by-outer-loop (thread 0 would run past the barrier before thread
1 starts), which is why cc rejects them rather than no-ops them.

### 3. Barrier and shared placement in the IR *(landed)*

**Barrier** — one statement constructor, `Workgroup_barrier`, landed with per-pass
semantics that PR #84 settled (the review-notes decisions, now the normative
contract):

- **Analyses** treat it as an inert effectful leaf (`Noop`-like arms in `visit_llc`,
  read/write collectors, the interval-analysis `pin_device_written_bounds` walker).
- **Virtualization**: a computation containing a barrier is never inlineable —
  `check_and_store_virtual` raises `Non_virtual 141` (`low_level.ml:732`), making the
  barrier arm of `inline_computation` unreachable (`assert false`,
  `low_level.ml:1136`).
- **CSE** never judges barrier-containing code alpha-equivalent
  (`low_level.ml:2094-2096`) — deduplication would delete a synchronization point.
- **Hoisting** (`hoist_shared_locals`) splits its sibling-statement list at barriers
  and hoists within each barrier-delimited segment independently
  (`low_level.ml:2402-2415`), because its write-hazard check (`writes_of_stmt`)
  cannot see a barrier (it writes no tensor).
- **Interval folding** needs no barrier awareness: its rewrites are expression-local
  (a comparison folds to a constant in place); it performs no motion across
  statements, and the symbol environment scoping follows loop structure only.

Grid-scoped synchronization is deliberately not representable — that is the honest
boundary tinygrad also draws (its `Barrier` is workgroup-scoped; kernels split at
reduction edges). If cooperative-groups grid sync ever earns its keep, it arrives as
a new constructor, not a parameter.

Note the pipeline-order simplification anticipated by the original draft holds: the
planned order is virtualize → schedule → simplify, so virtualization never sees a
barrier in practice; the `Non_virtual 141` arm is the defensive backstop for
hand-built IR (which the Phase B/C tests use).

**Shared placement** — *not* a new memory mode. The `memory_mode` lattice stays at
its five constructors; `Local`'s contract ("uniquely unobservable routine-scoped
scratch") already describes a shared tile exactly. What differs is placement within
the kernel, which is a *schedule decision* about compiler-generated tiles that never
escape the kernel — so it rides with the lowered code, not with the tensor node:
`optimized.workgroup_shared : Set.M(Tnode).t` (landed, `low_level.ml:219`, empty
until schedule transforms exist; `compile_proc` currently rejects a non-empty set,
`c_syntax.ml:1043`).

In Phase C, `compile_proc`'s local-declaration pass consults the set: a `Local` node
in `workgroup_shared` is declared with the backend's shared prefix (`__shared__ float
t[256];` / `threadgroup float t[256];`) instead of a plain stack array. Sizes are
static (dims are known at lowering), so CUDA's `shared_mem_bytes` stays 0 (static
`__shared__` doesn't use the dynamic pool). Zero-initialization of shared arrays
cannot use `= {0}` (not allowed for `__shared__`); nodes needing init get a
cooperative-store loop generated by the annotating pass, which is what it generates
for tile loads anyway. Interaction with the tensor-node bounds lifecycle
([interval-analysis-scalar-t](interval-analysis-scalar-t.md) Phase B v1) is
automatic: shared tiles are device-written, so their bounds candidates are pinned to
top at lowering like any compiled write — no special-casing.

No `Cooperative_load` construct: cooperative tile loading is ordinary
`For_loop`(Workgroup-annotated)/`Set`/`Get` code plus a barrier, generated by the
schedule pass. The IR grows two things (a field's worth of annotation and a barrier),
not a vocabulary.

### 4. Launch-dimension plumbing

A pure analysis in `low_level.ml`:

```ocaml
type launch_dims = { grid : int array; block : int array }  (* each length ≤ 3 *)
val launch_dims : t -> launch_dims  (* all-1s for all-Serial code *)
```

computed at `compile_proc` time (it also performs slot assignment, shared with
`pp_ll` via an environment). Extents are static ints — no interval machinery is
needed *here*; and the [signed-index-precision](signed-index-precision.md) per-node
element-count contract already bounds every extent within int32 (an annotated loop's
extent is some node's axis dimension by projection construction), so launch dims and
hardware index bindings cannot overflow the 32-bit index width. Each backend's
`code` record grows a `launch : launch_dims` field, consumed at link/run time:

- CUDA: `S.launch_kernel func ~grid_dim_x:g.(0) ... ~block_dim_x:b.(0) ...
  ~shared_mem_bytes:0` replacing the hardcoded 1×1 (`cuda_backend.ml:1043`);
  `kernel_prep_line` is deleted (an all-Serial kernel launches 1×1, making the guard
  redundant).
- Metal: `threadgroups_per_grid`/`threads_per_threadgroup` from the same record
  (`metal_backend.ml:902-908`, replacing the current
  `{width = min max_threads 1; ...}` placeholder). Validate the `block` product
  against `maxTotalThreadsPerThreadgroup` at pipeline creation.
- cc: after `validate_parallel` confirms no barriers/shared placements, launch dims
  are ignored (serial fallback).

Device-limit checks (max threads per block, max grid dims, shared-memory capacity)
live in the backends where device properties are queryable; `validate_parallel`
checks only backend-independent structure.

### 5. Backend mapping rules

Rendering is driven by three new `C_syntax_config` items (defaults in `c_syntax.ml`,
overridden per backend), replacing the current hard rejections at
`c_syntax.ml:456-459,717,1043`:

```ocaml
val hardware_index : kind:[ `Grid | `Workgroup ] -> slot:int -> string option
  (* None = backend cannot bind this axis in hardware *)
val barrier_syntax : string option
val shared_decl_prefix : string option
```

Bindings are emitted at the backend's `loop_index_type` — **signed** int32 (int64
under `large_models`) since the signed-index migration — with an explicit cast from
the unsigned hardware register (well-defined: the values fit by device limits and the
numel contract):

| IR construct | CUDA | Metal | cc (sync/multicore) |
|---|---|---|---|
| `Serial` loop | `for (...)` | `for (...)` | `for (...)` |
| `Grid` loop, slot k | `const int iN = (int)blockIdx.{x,y,z};` binding, no loop; extent → `grid_dim_{x,y,z}` | binding from `(int32_t)gid.{x,y,z}` (already in `extra_args`); extent → `threadgroups_per_grid` | serial `for` (legal fallback) |
| `Workgroup` loop, slot k | binding from `(int)threadIdx.{x,y,z}`; extent → `block_dim_{x,y,z}` | binding from `(int32_t)lid.{x,y,z}`; extent → `threads_per_threadgroup` | serial `for`, legal only barrier-free (validated) |
| `Workgroup_reduce` loop | as `Workgroup` (the distinction is for the schedule layer and validation, not the renderer) | as `Workgroup` | rejected |
| `Unrolled` loop | emitted as repeated body with substituted constants (per `unroll_dims` precedent) | same | same |
| `Workgroup_barrier` | `__syncthreads();` | `threadgroup_barrier(mem_flags::mem_threadgroup);` | rejected: `barrier_syntax = None` |
| shared placement | `__shared__ T name[n];` | `threadgroup T name[n];` | rejected: `shared_decl_prefix = None` |

A `None` from a hook is a compile-time error for barriers/shared, and a serial
fallback for `Grid`/`Workgroup` — matching §2's soundness argument. The schedule
layer will eventually consult a per-backend capability record before annotating; in
v1 the annotating test code just knows.

### 6. Migration sketch

Steps 1–3 (Phase A) **landed in PR #84** exactly as sketched, zero-diff on generated
code — including the two follow-through items the sketch predicted: `axis` in the CSE
alpha-equivalence comparison (`low_level.ml:2087`) and barrier match arms everywhere
the compiler exhausts `t` (analyses leaf-like, motion-refusing; the substantive
decisions are catalogued in §3). No sexp-churn promotions turned out to be needed:
`Serial` prints as legacy `for`.

Remaining (Phase B): new `C_syntax_config` hooks with cc/CUDA/Metal instances;
`pp_ll` `For_loop` case branches on `axis` (replacing the rejection at
`c_syntax.ml:456-459`); `compile_proc` computes slots + launch dims; backend `code`
records carry `launch`; `validate_parallel`; CUDA guard deleted. Extent-mismatch
guards, when the schedule layer starts producing them, follow §2's
construct-then-fold discipline.

## Approach / phasing

Implementation order across backends is **Metal first** (the primary dev box is a
Mac), then cc's serial fallback, then CUDA (CI-verified; not buildable locally). The
[#412 writeup](../../archive/docs/proposals/gh-ocannl-412.md)'s CUDA-centric phase list predates this decision and
is outdated on ordering — treat it as a catalogue of transforms, not a sequence.

- **Phase A — representation** *(landed 2026-07-03, PR #84)*. IR field + barrier
  constructor + `workgroup_shared` field, all defaults preserving today's output
  byte-for-byte. Landed with the full test suite green and no promotions.
- **Phase B — rendering and launch.** Hooks, bindings (at the signed index width),
  slot assignment, `launch_dims`, guard removal, `validate_parallel` — Metal dispatch
  first, cc serial fallback alongside, CUDA following. Tests: structural
  expected-file test of a hand-built annotated `Low_level.t` (per the established
  pattern, built directly — lowering from the high level never produces annotations
  yet); executed parity test of a `Grid`+`Workgroup`-annotated element-wise kernel
  vs. its `Serial` twin on Metal (locally), the serial fallback on cc, and CUDA in
  CI. The parity-test shape to follow is `test/operations/test_bounds_folded_gather.ml`
  (run both variants, compare values, assert structure from `build_files/` output —
  not from re-lowering, which reflects mutated global state).
- **Phase C — workgroup structure.** Shared declarations + barrier rendering (Metal
  `threadgroup` first); a hand-built shared-memory reduction (the GROUP_REDUCE
  pattern: partial sums in a shared tile, barrier, tree or serial combine) as an
  executed parity test vs. the serial reduction. The barrier-motion contract of §3 is
  already enforced in CSE/hoisting; the Phase C audit reduces to new passes added
  since (currently only interval folding, which is motion-free by construction).
- **Consumers.** [#412](https://github.com/ahrefs/ocannl/issues/412) Phase 2+
  annotates via heuristics/tiling; [schedule-ir-optops](schedule-ir-optops.md)'s
  Split-with-retype makes annotation a schedule value;
  [gh-ocannl-263](../research/lean-attention-feasibility.md) (Flash attention) eventually consumes the
  workgroup structure. The `__constant__` end of address spaces
  ([#195](https://github.com/ahrefs/ocannl/issues/195)) can reuse the
  placement-refinement pattern (§3) rather than a memory mode, but is out of scope
  here.

## Risks and non-goals

- **Not solved here**: grid-level synchronization, persistent blocks, megakernels
  ([#318](https://github.com/ahrefs/ocannl/issues/318)) — tinygrad's spec doesn't
  solve them either; this is the staircase, not the summit. Kernel splitting at
  cross-workgroup dependence edges is future work the well-formedness rules make
  nameable.
- **Validation is structural, not semantic**: `validate_parallel` cannot prove
  iteration independence; that is the annotating pass's obligation (like `trace_it`
  correctness today). An executed-parity test harness is the practical check —
  every annotated kernel in the test suite runs against its `Serial` twin.
- **Pass-audit surface for barriers** is bounded and mostly discharged: the
  CSE/hoisting treatment landed with Phase A (§3); newly added passes must keep the
  contract (the interval-analysis walkers added 2026-07-04 do — barrier arms are
  leaves, and folding never moves code). The rule of thumb for future passes: a
  barrier is an opaque effectful statement, a segment boundary for anything that
  reorders or deduplicates siblings.
- **Interaction with tensor-node bounds** (Phase B v1 of the interval proposal) is
  by-construction safe: annotated kernels' writes pin bounds like serial writes; a
  bounds-discharged guard (e.g. a folded gather guard) inside an annotated kernel
  changes nothing about thread-mapping. No joint validation is needed.

## Relations

[#412](https://github.com/ahrefs/ocannl/issues/412) (consumer and first site),
[#195](https://github.com/ahrefs/ocannl/issues/195) (addrspace cousin),
[#318](https://github.com/ahrefs/ocannl/issues/318) (megakernel exploration, landed),
[gh-ocannl-263](../research/lean-attention-feasibility.md) (Flash attention — eventual consumer),
[schedule-ir-optops](schedule-ir-optops.md) (depends on this);
[interval-analysis-scalar-t](interval-analysis-scalar-t.md) (**landed** 2026-07-04:
supplies the guard-discharge mechanism for §2's extent-mismatch guards and the
Padto/logical-padding masks downstream);
[signed-index-precision](signed-index-precision.md) (**core landed** 2026-07-04:
hardware index bindings are signed; the numel contract bounds annotated extents).

## Original acceptance criteria (from the stub)

- [x] Annotation carrier decided (`For_loop` field vs. symbol metadata) with
      migration sketch for `to_low_level`/backends — §1, §6; landed.
- [x] Barrier + LOCAL buffer representation in `Low_level.t` sketched — §3; landed.
- [x] Backend mapping rules (CUDA grid/block, Metal threadgroups, CC threads)
      drafted — §5.

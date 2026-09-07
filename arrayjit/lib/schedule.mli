(** {1 Schedule IR: loop-nest transforms as values}

    Halide-style schedules over the lowered, optimized IR: a list of {!optop}s applied as a pure
    [Low_level.optimized -> Low_level.optimized] pass at the [?lowered_transform] seam of backend
    [compile]. See docs/schedules_and_autotuning.md for the system view (composition recipes,
    default presets, fission, autotuning) and docs/proposals/schedule-ir-optops.md for the design
    rationale, including the pass-ordering contract (§2): schedules run after the whole
    [optimize_proc] pipeline (so they see fused code), {!apply} folds freshly constructed guards by
    re-running [simplify_llc] (plus CSE and hoisting when a transform duplicated code), and there is
    no re-virtualization. *)

open Base
module Tn = Tnode

type optop =
  | Split of {
      axis : Indexing.symbol;  (** The loop to split, identified by its index symbol. *)
      factor : int;  (** Extent of the new inner loop. *)
      outer : Low_level.axis_type;  (** Axis type of the new outer loop ([Serial] = no retype). *)
      inner : Low_level.axis_type;  (** Axis type of the new inner loop. *)
      outer_index : Indexing.symbol;  (** Fresh symbol for the outer loop; see {!split}. *)
      inner_index : Indexing.symbol;  (** Fresh symbol for the inner loop; see {!split}. *)
    }
      (** [For_loop i in [0, N)] becomes [i_o in [0, ceil(N/factor)) { i_i in [0, factor) }] with
          [i := factor*i_o + i_i] substituted throughout the body (index vectors and
          [Embed_index]), and — when [factor] does not divide [N] — the body wrapped in an
          [If (factor*i_o + i_i < N)] remainder guard (construct-then-fold: {!apply}'s trailing
          simplify erases it whenever the loop extents prove it). The split loop must start at 0,
          which lowering guarantees. Splitting a serial loop preserves iteration order; retyping
          the results to hardware axes carries the iteration-independence obligation exactly as
          for [Low_level.validate_parallel]. *)
  | Swap of { outer : Indexing.symbol; inner : Indexing.symbol }
      (** Interchange two perfectly nested loops (the outer loop's body must be exactly the inner
          loop; fails loudly otherwise). Reorders iterations — legal for the associative-commutative
          accumulation patterns lowering emits; bitwise reproducibility is the caller's concern. *)
  | Retype of { axis : Indexing.symbol; ty : Low_level.axis_type }
      (** Change a loop's axis type in place. Retyping to a hardware kind requires [from_ = 0] and
          iteration independence (the caller's obligation; structure is checked downstream by
          [Low_level.validate_parallel]). Retyping to [Vectorized] likewise asserts iteration
          independence: the C backends render the loop under vectorization pragmas (gh-ocannl-164),
          backends without pragmas render it as a plain serial loop. *)
  | Unroll of { axis : Indexing.symbol; materialize : bool }
      (** [materialize = false]: set the axis type to [Unrolled] — codegen repeats the body with the
          index bound as a per-block constant (after simplify/CSE have run, so the copies are opaque
          to the optimizer). [materialize = true]: unroll in the IR by substituting index constants,
          so that {!apply}'s trailing simplify + CSE see the copies — constant-folding [Affine]
          indices and deduplicating repeated loads. Register blocktiling is [Split] + materializing
          [Unroll] + the existing CSE (schedule-ir-optops §4). *)
  | Partition of {
      axis : Indexing.symbol;  (** The loop to partition, identified by its index symbol. *)
      breakpoints : int list;
          (** Segment starts: strictly increasing, each strictly inside the loop range
              ([from_ < b <= to_], so every segment is non-empty). *)
      segment_indices : Indexing.symbol list;
          (** Fresh symbols, one per segment ([length breakpoints + 1]); see {!partition}. *)
    }
      (** Index-set splitting (gh-ocannl-508): replace the [Serial] loop
          [For_loop axis in [from_, to_]] by consecutive segment loops
          [s_0 in [from_, b_1) ; s_1 in [b_1, b_2) ; ... ; s_m in [b_m, to_]], each binding a fresh
          symbol substituted for [axis] in a copy of the body (scalar-local scope ids refreshed per
          copy, as for materializing [Unroll]). Segment ranges stay absolute — no index arithmetic
          changes, iteration order is preserved exactly — and each segment's narrowed range lets
          {!apply}'s trailing simplify interval-fold the guards it decides (statement [If]s and
          scalar [Where] range guards alike). This is the segmented-rendering replacement for
          in-loop range guards: an inlined concatenation's per-component guards (partition the
          consumer at the component boundaries, e.g. from {!partition_breakpoints}) and [Split]'s
          construct-then-fold remainder guard (partition at the last tile-multiple first, then
          [Split] the dividing main segment — clean main nest plus epilogue) both specialize into
          guard-free segment nests, and the fresh symbols make each segment individually
          addressable by subsequent ops (per-segment scheduling). *)
  | Pad of {
      axis : Indexing.symbol;  (** The loop to pad, identified by its index symbol. *)
      to_multiple_of : int;  (** The padded extent is the least multiple [>=] the loop extent. *)
    }
      (** Pad-to-tile (gh-ocannl-485, PADTO): extend the [Serial] loop
          [For_loop axis in [0, N)] to [[0, M)] where [M] is the least multiple of [to_multiple_of]
          with [M >= N], and guard every effectful leaf statement of the body with [If (axis < N)]
          (leaf statements, not the whole body, so barriers inserted by a later shared {!Stage} stay
          under uniform control flow). The pad iterations are no-ops, so the op is unconditionally
          semantics-preserving ([op_legality]: [Op_legal]); when [to_multiple_of] already divides
          the extent it is the identity. Purpose: downstream [Split]s by factors of [M] divide
          cleanly (no remainder guard), {!Stage} tiles minted over the padded tile loops zero-fill
          their fringe (see {!constructor-Stage}), and {!constructor-Tensorize} recognizes the
          guards as pad masks — moving row/column masks to the accumulator transfers and
          discharging reduction masks against zero-filled staged operands — so tensorized paths
          cover arbitrary extents. A SCALAR pipeline keeps the leaf guards instead, and
          {!constructor-Privatize} classifies them rather than rejecting them (gh-ocannl-730), which
          is what lets the register-blocktiled GPU family pad too. The surviving guards are exactly
          the flip points
          {!partition_breakpoints} detects: a later {!constructor-Partition} of an enclosing block
          loop at the last fully valid block specializes them away in the interior segments. *)
  | Stage of {
      source : Tn.t;
      tile_loops : Indexing.symbol list;
      shared : bool;
      cooperative : int option;
      hoisted : bool;
      swizzle : Low_level.swizzle_kind option;
      pad_stride : int option;
      pipeline_depth : int;
      tile_prec : Ops.prec option;
    }
      (** Stage reads of [source] through a tile: a fresh [Local]-mode node registered in the traced
          store, its dims derived per source axis from the range of the index terms over
          [tile_loops] (schedule-ir-optops §5). All reads of [source] must use one index vector (v1)
          whose per-axis terms split cleanly into tile-loop terms (positive coefficients) and outer
          terms. With [shared = true] the tile is added to [workgroup_shared] and a cooperative-load
          nest plus barriers are inserted at the deepest loop carrying an outer-part symbol or a
          reused [Workgroup] tile axis: [Workgroup]-typed tile loops are reused as the cooperating
          thread indices, [Serial] tile loops are iterated under fresh symbols, per-axis edge guards
          are constructed then folded — in [Where] form storing 0 (the add-reduce accumulation
          identity) to out-of-range slots, so edge tiles of a non-dividing or padded staging are
          safe to read over their whole index space; every minted tile is recorded in
          {!Low_level.optimized.zero_fringe} accordingly (gh-ocannl-485) — and redundant loading
          along non-participating workgroup axes is restricted to one representative thread. Shared
          stages sharing an anchor compose into ONE phase per iteration of it: a later stage's load
          nest is grouped in front of the phase barrier the earlier one left, back to back with its
          loads (gh-ocannl-567 — the loads write distinct tiles and depend on nothing of each
          other's, so a barrier between them would only serialize two independent copies), and the
          phase closer is shared rather than duplicated. Grouping requires that the phase already in
          place read nothing this stage's remap turns into a read of its tile. A shared stage with
          no anchor (no outer-part symbol, no reused workgroup axis — e.g. staging a broadcast
          vector) wraps the outermost tile loop instead of the routine root, so enclosing workgroup
          axes can guard the loads; every workgroup slot active in the kernel's launch must be
          reused or bound by a loop enclosing the staging point, otherwise threads differing in the
          uncovered slot would race on the shared tile and the op raises. Note that [Split]'s
          whole-body remainder guards would place the inserted barriers under divergent control flow
          (rejected by [validate_parallel]), so v1 shared staging requires tile sizes dividing the
          extents. With [shared = false] (CPU operand packing) all tile loops must be [Serial] and a
          plain serial copy nest is inserted, no barriers. The source must not be written in the
          routine.

          [cooperative = Some w] is the lane-aware mode for composing with {!constructor-Tensorize}
          (docs/proposals/tensorize-mma.md, "Lane-aware Stage"): shared staging with all-[Serial]
          tile loops whose load nest is wrapped in a {e fresh} extent-[w] [Workgroup] lane loop —
          positionally the same slot 0 the tensorized micro-kernel's lane loop binds, with extent
          agreement enforced downstream by [Low_level.validate_parallel]'s barrier-strength
          uniformity (pass the backend's {!field:Backend_intf.mma_simd_width} to both ops). The lane
          is folded linearly into the innermost fresh copy loop: extent [<= w] replaces the loop by
          the lane index under a [lane < extent] guard (folds when equal); an extent divisible by
          [w] iterates [extent / w] chunks at [w*step + lane]; otherwise the whole nest is
          restricted to lane 0 (division is not expressible in affine indices). The lane loop covers
          workgroup slot 0 for the staging-point coverage rule by construction.

          [swizzle = Some kind] stores the tile in an XOR-swizzled layout (the
          bank-conflict-avoidance follow-up of docs/proposals/tensorize-mma.md): the tile node is
          recorded in the [optimized] record's {!field:Low_level.swizzled} map under [kind] and
          codegen remaps every element access [P*C + col] to a per-row bijection of the minor axis,
          so semantics are unchanged while same-column accesses (the classic strided read of a
          staged tile) spread across shared-memory banks. Requires [shared = true] and a tile with
          at least two axes.

          The two flavors differ in the unit the XOR permutes — hence in their minor-extent
          requirement, neither of which implies the other — and in who can consume them
          (gh-ocannl-481 item 3):

          - {!constructor-Low_level.Swizzle_elem} remaps [P*C + col] to
            [P*C + (col lxor (P land (C-1)))], requiring a power-of-two minor tile dim [> 1].
            Renderings that assume row-major storage decline it: [Tile_mma] intrinsic/register-tiled
            paths fall back to the scalar micro-kernel (which reads elementwise and stays correct),
            so do not combine it with {!constructor-Tensorize} when the intrinsics are the goal —
            this flavor is for scalar/register-blocktiled staged kernels.
          - {!constructor-Low_level.Swizzle_b128} permutes whole 16-byte units, requiring the minor
            tile dim to span a power-of-two count [> 1] of 16-byte units. This is the layout the
            CUDA inline-PTX [mma.sync] arms' [ldmatrix] loads consume, so it DOES combine with
            {!constructor-Tensorize}: the tile is both bank-de-conflicted and fragment-loadable in
            one instruction. Backends without such loads still decline it to the scalar
            micro-kernel, which stays correct.

          [pad_stride = Some p] rounds the tile's MINOR dim up to a multiple of [p] (gh-ocannl-481
          item 4). The tile's leading-dimension stride is that dim — every consumer reads it off the
          node — so this changes the stride while the iterated index space stays the unpadded
          extents. Two payoffs, both about the stride rather than the data: shared memory bank
          conflicts on a strided read of the tile (proposal §5's "correct, possibly bank-conflicted"
          v1), and layout rules stated on the stride — a fragment load's ld-multiple constraint, and
          [Swizzle_b128]'s 16-byte-unit count, which is why the two compose as "pad first, then
          check". Requires a tile with at least two axes and [p > 1].

          The padded slots hold nothing under a row-major layout: no loop reaches them, so they are
          neither written nor read, and the {!field:Low_level.zero_fringe} contract — about the
          fringe of the staged {e source} region within the iterated space — is unaffected. Under a
          swizzle they do carry data, the XOR being a bijection of the whole padded row, and reads
          go through the same map.

          [hoisted = true] packs a compile-time-constant operand once, out of the routine
          (gh-ocannl-470, the compiler-native analog of ggml's [CPU_REPACK] [set_tensor] hook):
          instead of a per-invocation scratch tile refilled by an in-kernel load nest, the packed
          layout covers the {e whole} source — one packed-buffer axis per outer coordinate
          ([outer part / tile dim], requiring outer-part coefficients and offsets divisible by the
          tile dim on tiled axes) followed by the tile axes — and the reads are remapped to it
          directly; no load nest, no barriers. The packed node is minted as a host-initialized
          constant: its buffer is computed on the host when first forced (a [Host_inits] lazy driven
          by the same affine index maps; pad slots of edge tiles are zero-filled) and uploaded once
          per device into the constant pool ([constant_buffer_cache]), so re-linking into sibling
          contexts reuses the same packed buffer. Requires [shared = false] (and hence all-[Serial]
          tile loops), a source with registered host-init data that is known constant (declared
          [Effectively_constant] intent or a constant placement), no padding on the source, and
          every outer-part symbol bound by an enclosing loop (static/dynamic indices are rejected —
          packing runs at link time with no bindings). Note: later host-side writes to the source
          (e.g. [set_values]) do NOT refresh the packed copy.

          [pipeline_depth = d] with [d > 1] software-pipelines a cooperative staging (gh-ocannl-487;
          an int rather than a bool so a search has a dimension, not a switch — [1] is the identity,
          taking exactly the unpipelined code path; [d = 2] only: both the portable form (phase 1)
          and CUDA's [cp.async] arm (phase 2) have single-step lookahead — one prefetch per
          iteration, completed behind one barrier resp. one wait-all — so deeper depths would only
          consume shared memory and are rejected until the async arms grow commit-group/wait-group
          bookkeeping). Requires [cooperative = Some _] and an anchor loop L* that is [Serial],
          starts at 0 and has at least one iteration (the rotation needs a well-defined iteration
          order and the prologue fills buffer copy 0 — a dead anchor must not execute it; the
          no-anchor broadcast case has no rotor). The staging composition then becomes: a prologue
          copy of iteration [from_] before L*, per iteration [k] a single barrier followed by the
          copy for [k + 1] under an [If (k < to_)] guard (folded away on 1-trip loops) followed by
          the compute, and a trailing barrier after L* — [2N] barriers become [N + 1], and the
          prefetch is issued before the compute whose latency hides it. Pipelined stages sharing the
          anchor compose into the {e same} phase — the later stage's prefetch is grouped behind the
          existing barrier, back to back with the earlier ones, keeping one barrier per iteration
          for all of them (a barrier between prefetches would force the earlier copy to complete
          before the compute, forfeiting its overlap). When a later {!constructor-Tensorize} leaves
          the rotor body ending in a [Tile_mma] — a barrier by the emission contract, bracketed on
          every barrier-capable backend — {!apply}'s finalization elides the pipeline's own barriers
          (the loop-leading one, and the after-loop one where still adjacent): the intrinsic's
          bracket is the phase, so the tensorized pipelined kernel pays NO explicit barrier per
          iteration, one fewer than the depth-1 form (whose k-block loads feed the SAME iteration's
          compute, so its phase barrier is load-bearing — only the intrinsic's trailing bracket is
          emitted per iteration on every rendering form; see {!apply}). A scalar pipelined compute
          keeps its explicit barriers. The tile is recorded in {!field:Low_level.pipelined}: codegen
          allocates [d] rotating copies and selects the buffer by the loop counter (reads [k mod d],
          in-loop writes [(k+1) mod d], the prologue write copy 0), so the compute reads exactly the
          values the unpipelined form reads, in the same order — the pipelined rendering is bitwise
          identical to [pipeline_depth = 1], a pure prefetch-timing transform
          ({!check_hardware_limits} accounts the tile's shared-memory bytes times [d]). [Tile_mma]
          in the compute is compatible: its bracketing barriers are uniformly reached and separate
          same-buffer phases even more strongly than required. Backends that cannot render
          workgroup-shared staging (the serial C backends) reject the composition exactly as they
          reject unpipelined shared staging.

          [tile_prec = Some p] mints the tile at storage precision [p] instead of the source's,
          folding the widening conversion into the staging copy (gh-ocannl-575, the gh-ocannl-517
          packing seam): a packed panel of a narrow-storage operand becomes e.g. f32 scratch,
          converted once per element at pack time instead of once per read at the memory boundary,
          and precision-uniformity consumers (the register-tiled [Tile_mma] micro-kernel) see the
          compute precision directly. Only widenings are accepted — [p] must equal the source's
          storage precision or be [single]/[double] over a narrow float — because a widening is
          exact and its round-trip is the identity, making the transform unconditionally
          numerics-preserving; a narrowing tile would change values and is rejected. Composes with
          [hoisted] (the host-side pack converts) and with the in-kernel copy nest; the edge-guard
          zero fill is at the tile's precision either way. *)
  | Privatize of { target : Tn.t; over : Indexing.symbol }
      (** Accumulator privatization: contract the read-modify-write accumulation of the materialized
          [target] across the (Serial) [over] loop's whole subtree into a per-thread [Local]
          accumulator tile — initialized from [target] before the loop, accumulated in place, stored
          back after, one final write per element. This recovers, for materialized nodes, the
          scope-local form virtualization gives virtual accumulators ([Local_scope]) — and because a
          routine-local tile cannot alias the kernel's device pointers, downstream compilers
          register-allocate it without waiting for [restrict] (gh-ocannl-164). Tile shape: per
          target axis, the index terms over loops nested inside [over] (required [Serial] with
          [from_ = 0]); no such terms yields a scalar accumulator. All accesses of [target] under
          [over] must use one index vector, which must not mention [over] itself, and must sit under
          one [If] guard chain, whose conditions are classified (gh-ocannl-730). An
          iteration-INVARIANT condition (no memory reads, no symbols bound inside [over]'s subtree)
          also gates the init-load and store-back, so lane-restricted accumulations (e.g. [w == 0])
          privatize correctly. A condition that varies is kept on the update alone, and only in the
          two shapes that provably cannot select threads: one mentioning no hardware-typed loop
          symbol at all (a mask over one thread's own iterations — what a reduction-axis
          {!constructor-Pad} leaves behind), and one that is literally [target]'s own index on some
          axis compared against a bound no larger than that axis's dimension (a row/column pad mask;
          the transfers already carry that same edge guard). Any other varying condition — data
          dependent, or mixing a hardware symbol into a comparison that is not [target]'s index — is
          rejected, since it could restrict which threads accumulate while the transfers write back
          an accumulator that never received the update. A [Zero_out] of [target] elsewhere is left
          in place — the init-load observes it, so semantics are preserved without a surjectivity
          analysis. Compose as: [Split]s → [Stage]s → [Privatize] → materializing [Unroll]s (the
          unrolls then turn the tile accesses into constant-indexed, register-allocatable form). *)
  | Expand_zero of { tn : Tn.t; indices : Indexing.symbol list }
      (** Expand the unique [Zero_out tn] statement into an ordinary loop nest over the supplied
          symbols (one per axis of [tn]'s padded dims; see {!expand_zero}). Whole-node [Zero_out] of
          a materialized node is rejected by [Low_level.validate_parallel] in multi-threaded kernels
          — expanding it first lets the schedule split and annotate the zeroing with the same
          hardware geometry as the computation that follows. *)
  | Tensorize of {
      i : Indexing.symbol;
      j : Indexing.symbol;
      k : Indexing.symbol;
      lane : Indexing.symbol;  (** Fresh symbol for the cooperating lane loop; see {!tensorize}. *)
      simd_width : int;
          (** Extent of the lane loop (the backend's {!field:Backend_intf.mma_simd_width}; 32 on
              Metal and CUDA). *)
      tile : Register_tile.t option;
          (** The C-tile geometry of the register-tiled CPU rendering (gh-ocannl-619): [rm] rows by
              [rn] vectors of [lanes], held across the k-loop. [None] (the default, and what every
              pre-existing schedule means) lets the renderer choose by its ranking model
              ({!Register_tile.default}); [Some] is honoured exactly, or declined to the scalar
              fallback with a diagnostic when it does not fit ({!Register_tile.check}: a width the
              file renders and the extents fill, the live-register budget) — never silently
              replaced, so a candidate timed under a geometry label ran that geometry or ran none.
              Part of the schedule's identity (the cache saves it, the canonical render digests it).
              The intrinsic (tensor-core) emissions ignore it. *)
    }
      (** Tensor-core emission (docs/proposals/tensorize-mma.md §3): replace the perfectly nested
          serial [i × j × k] matmul micro-kernel — whose body is the single accumulation
          [d[..., i, j] += a[..., i, k] * b[..., k, j]] (plain-add or FMA form; each operand's tile
          spans its last two axes with unit coefficients, transposed layouts rejected in v1) — with
          a [Low_level.Tile_mma] block statement covering the full [m×n×k] extents, wrapped in a
          fresh [Workgroup]-typed lane loop of extent [simd_width]. The original nest is kept as the
          statement's scalar [fallback]: backends without an MMA hook (or declining a particular
          precision/shape) render it once per simdgroup under an [if (lane == 0)] guard, so the op
          is always semantics-preserving. Apply after [Split]s and [Stage]s; [Stage]/[Privatize]
          must come before it. Divisibility by the intrinsic tile (8 on Metal) is a per-call
          emission concern, not checked here. *)
  | Fuse_epilogue of { target : Tn.t; shared : bool }
      (** Epilogue fusion (gh-ocannl-486): fold the sole-consumer, index-space-compatible
          elementwise tail that re-reads [target] — the typical bias add / activation / residual
          after a reduction — into [target]'s store-back site, eliminating the tail's separate
          memory pass and keeping the fused routine a single kernel/segment. Recognized sites: the
          lane-0 fragment store-back synthesized by [Tensorize]'s accumulator contraction (the tail
          becomes a fourth, lane-0-guarded statement of the marked region, rendered after the
          backend's intrinsic block; a pad-masked store-back — gh-ocannl-485's range guards on a
          non-dividing site — is recognized through its guards, which are re-imposed on the
          relocated tail, gh-ocannl-521), the whole-K [Tile_mma] writing [target] directly (the
          unstaged tensorized pipelines: the intrinsic block completes the accumulator's m x n tile,
          so the tail becomes a sibling lane-0 nest over the tile at the accumulator's base indices,
          gh-ocannl-521), the [Privatize] tile store-back (per-element), and the plain accumulation
          nest (the tail slides inside the output loops after the serial reduction loop). The tail
          must be the first real statement after the last statement writing [target]: a perfect
          Serial nest over exactly [target]'s dims, assigning a different node at the identity index
          tuple, elementwise, with every read of [target] at that tuple, and no later statement
          mentioning [target]. The store-back of [target] itself is kept. Elementwise tails never
          reorder the reduction, so on the C backends the fused values are bitwise equal to the
          two-kernel form. Apply after [Tensorize]/[Privatize].

          [shared] (GPU only, requires the fragment site): the fused tail is often [target]'s last
          consumer, so placement makes [target] routine-local — a per-thread array the fragment
          hooks cannot [simdgroup_load] from. With [shared], [target] is placed in workgroup-shared
          memory (like [Stage]'s shared tiles) unless already settled on-device, so the intrinsic
          fragment path fires against threadgroup memory. CPU backends reject shared placement. *)
  | Split_reduce of {
      axis : Indexing.symbol;  (** The Serial reduction loop to split, by its index symbol. *)
      target : Tn.t;  (** The accumulated node whose reduction over [axis] is split. *)
      num_blocks : int;  (** Number of per-block partials (the new block loop's extent). *)
      block_index : Indexing.symbol;  (** Fresh symbol for the block loop; see {!split_reduce}. *)
      inner_index : Indexing.symbol;
          (** Fresh symbol for the in-block chunk loop; see {!split_reduce}. *)
      combine_indices : Indexing.symbol list;
          (** Fresh symbols, one per [target] axis, binding the combine nest's loops; see
              {!split_reduce}. *)
    }
      (** Deterministic two-pass split reduction (gh-ocannl-484): parallel reductions without
          atomics. The [Serial] loop [axis in [0, N)] — carrying the single accumulation of
          [target] in its subtree — becomes
          [block in [0, num_blocks) { inner in [0, ceil(N/num_blocks)) }] with
          [axis := ceil(N/num_blocks)*block + inner] substituted and a construct-then-fold
          remainder guard as for {!constructor-Split}; the accumulation is redirected into a fresh
          scratch node [partials] of dims [num_blocks x target-dims] (tile namespace, placed
          [On_device], registered in the traced store) at the original cell prefixed by [block];
          and a synthesized combine statement, inserted right after the enclosing top-level
          statement, folds the partials into [target] in a {e fixed balanced-tree order} over
          [combine_indices]-bound loops: [target[c..] := target[c..] ⊕ tree(partials[0..B-1][c..])].
          The block loop is freely annotatable afterwards ([Retype]/the default presets): its index
          pins the partials row, so parallelizing it is race-free by construction, and the
          [partials] producer/consumer pair is exactly the materialized cross-nest edge kernel
          fission cuts at — under [fission_scheduled]/[maybe_default_schedules] the two passes (and
          the scatter form's zeroing) compile as separate kernels with the event chain supplying
          the grid-wide synchronization the combine needs. Never annotate the block loop to a
          hardware axis while compiling both passes into one kernel.

          Recognized accumulation forms (the [axis] subtree must contain no other access of
          [target], and the rest of the statement must not touch [target] — its combined value
          exists only after the combine statement):

          - Static: a single rmw [Set], [target[idcs] := target[idcs] ⊕ e] with
            [⊕ ∈ {Add, Max, Min, Mul}] (FMA counts as [Add]), [idcs] free of [axis]. Every loop
            enclosing [axis] within the statement must pin exactly one component of [idcs]
            (injectively: at most one symbol per component), so distinct enclosing iterations use
            distinct partial cells and the combine re-iterates exactly the written cells. Each
            block initializes its partial cell to the accumulation identity in-nest (no separate
            zeroing pass).
          - Dynamic (the gh-466 embedding-backward scatter): a single [Set_dynamic]
            add-accumulation of its own row (the [Get_dynamic] rmw form built by
            [rewrite_one_hot_reductions]), possibly under guards. The scatter is redirected to
            [partials] with the block index prepended (dynamic axis shifted by one): within a block
            colliding rows stay serial, across blocks rows land in disjoint partials slices — the
            block loop parallelizes what the scatter alone cannot. Rows are data-dependent, so
            [partials] is zeroed by a preceding whole-node [Zero_out] statement (its own fission
            segment) and the combine covers all of [target].

          Within each chunk the original serial order and rounding are preserved; across chunks the
          reduction is reassociated (the same license as {!constructor-Swap} of accumulations). The
          result is deterministic {e per schedule} — bitwise-reproducible run to run and across
          serial/parallel renderings of the same schedule — but not bitwise-equal to the unsplit
          serial reduction: schedule identity pins numerics (retuning may change [num_blocks] and
          hence the combine tree). *)
[@@deriving sexp_of]

type schedule = optop list [@@deriving sexp_of]
(** Applied left to right by {!apply}. *)

val split :
  axis:Indexing.symbol ->
  factor:int ->
  outer:Low_level.axis_type ->
  inner:Low_level.axis_type ->
  optop * Indexing.symbol * Indexing.symbol
(** Builds a {!constructor-Split} with fresh outer and inner index symbols (via
    [Indexing.get_symbol]) and returns them, so subsequent ops in a programmatically built schedule
    can reference the new loops. *)

val partition : axis:Indexing.symbol -> breakpoints:int list -> optop * Indexing.symbol list
(** Builds a {!constructor-Partition} with one fresh index symbol per segment (via
    [Indexing.get_symbol]) and returns them in segment order, so subsequent ops in a
    programmatically built schedule can address individual segments. *)

val partition_breakpoints : axis:Indexing.symbol -> Low_level.t -> int list
(** Derives candidate {!constructor-Partition} breakpoints of the [axis] loop from the guards
    already present in its body: statement [If] conditions and scalar [Where] conditions (e.g. the
    virtualizer's per-component range guards of an inlined concatenation, [Split]'s remainder guard,
    or a clamped window's range guards, gh-ocannl-504) are scanned for comparisons whose two sides
    differ by [k*axis + off]. With everything besides [axis] constant, such a comparison flips truth
    value at one point of the axis range. Non-axis symbols bound by other loops (inside the [axis]
    loop — e.g. the window symbol of a clamped-window guard — or enclosing it) are bounded by their
    loop ranges, putting [off] in an interval: the comparison is then always-true / mixed /
    always-false over the axis range, and both transition points are recorded — the mixed (boundary)
    segments are exactly delimited while the decided segments fold their guards. Returns the
    collected flip points that fall strictly inside the loop range, sorted and deduplicated
    (possibly empty — e.g. when every guard is already interval-decided); partitioning at them makes
    every such guard interval-decided within each segment (for interval-ranged comparisons: outside
    the mixed segments), so {!apply}'s trailing simplify erases them. Loops are located the way
    {!apply} rewrites them (gh-ocannl-668): inside a [Local_scope] body, where the accumulation
    mints of a materializing {!constructor-Unroll} or a {!constructor-Partition} put the segment and
    inner loops; and over EVERY loop binding [axis], not the first — a materializing
    {!constructor-Unroll} leaves one copy per unrolled step with a different constant substituted
    for the unrolled index, so one source guard's copies flip at different points and the result is
    their union (each copy filtered by its own range). Raises [Invalid_argument] when no loop binds
    [axis]. *)

val expand_zero : tn:Tn.t -> optop * Indexing.symbol list
(** Builds an {!constructor-Expand_zero} with one fresh symbol per axis of [tn] (forcing [tn]'s
    dims) and returns the symbols for subsequent [Split]/[Retype] ops. *)

val split_reduce :
  axis:Indexing.symbol ->
  target:Tn.t ->
  num_blocks:int ->
  optop * Indexing.symbol * Indexing.symbol * Indexing.symbol list
(** Builds a {!constructor-Split_reduce} with fresh block and inner index symbols plus one fresh
    combine symbol per axis of [target] (forcing [target]'s dims), and returns them
    ([op, block, inner, combine]) so subsequent ops can annotate the pass-1 block loop and the
    combine nest's loops. *)

val tensorize :
  ?tile:Register_tile.t ->
  i:Indexing.symbol ->
  j:Indexing.symbol ->
  k:Indexing.symbol ->
  simd_width:int ->
  unit ->
  optop * Indexing.symbol
(** Builds a {!constructor-Tensorize} with a fresh lane symbol (via [Indexing.get_symbol]) and
    returns it. [?tile] is the register-tile geometry request (gh-ocannl-619), absent by default. *)

val split_reduce_hoist : Low_level.optimized -> optop -> Indexing.symbol list
(** The loops that would have to enclose the reduction for this {!constructor-Split_reduce} to be
    recognized: the accumulation cell's symbols that are currently bound {e inside} the reduction
    loop (gh-ocannl-537). This is the one rejection cause a loop interchange can remove — OCANNL
    lowers conv bias/weight gradients with the accumulated channel loops innermost and the reduction
    loops (batch, y, x) outside them, the exact inverse of the static form's pinning discipline — so
    autotune seeding hoists these symbols with a {!constructor-Swap} chain and re-probes, rather
    than dropping the site. The answer comes from the recognizer itself (a hermetic probe of its own
    apply), not from a re-implementation of the discipline or a parse of its message.

    [[]] for every other outcome: a legal op, a non-[Split_reduce] op, and any rejection an
    interchange cannot remove (the cell mentioning the reduction loop itself, an enclosing loop that
    pins no component, an unrecognized accumulation form). *)

val fuse_epilogue_witness : target:Tn.t -> Low_level.optimized -> string option
(** Why {!constructor-Fuse_epilogue} on [target] would fail on this code — the recognizer's own
    rejection message, probed hermetically like {!split_reduce_hoist} — or [None] when an eligible
    elementwise tail immediately follows the reduction over [target] and the fusion applies. The
    autotune family tree judges its fused-flavor branch with it (the check runs on the pre-schedule
    code, where the plain accumulation-nest site applies), so a site without a fusable tail carries
    the recognizer's reason as a construction-time refutation rather than silently minting no twins.
*)

val can_fuse_epilogue : target:Tn.t -> Low_level.optimized -> bool
(** [Option.is_none (fuse_epilogue_witness ~target opt)]: whether the fusion applies. *)

val hoistable_constant : Tn.t -> bool
(** Whether the node is eligible as a [hoisted] {!constructor-Stage} source: declared value-constant
    ([Tnode.known_host_constant]) with registered host-init data to pack from. Shared by the
    autotune sketch and by [Schedule_cache.canonicalize], which renders it per tensor node so that
    same-shape programs differing in operand constancy do not share cached schedules. *)

val apply :
  ?static_indices:Indexing.static_symbol list ->
  schedule ->
  Low_level.optimized ->
  Low_level.optimized
(** Applies the ops left to right to the optimized code, then elides the staging barriers a later
    [Tensorize] made redundant, then re-runs [Low_level.simplify_llc] (which folds remainder and
    edge guards the loop extents prove) and, when a materializing [Unroll] duplicated code, CSE +
    cross-statement hoisting. [Stage] registers its tile in the traced store (and [workgroup_shared]
    when shared); the optimization context and merge node are never changed. Raises
    [Invalid_argument] when an op references a loop that does not exist at its point in the
    schedule, or violates an op precondition (see {!optop}). An empty schedule is the identity.

    The barrier elision (gh-ocannl-487, widened to every staged anchor by gh-ocannl-567) is a
    synchronization-only rewrite — bitwise identical output, on every backend path. It rests on the
    [Tile_mma] emission contract, whose two halves are NOT symmetric: every rendering form ENDS the
    intrinsic block with a workgroup barrier, so a staging barrier that follows one (inside a staged
    anchor's body, or right after the loop) is dropped at any pipeline depth; but only some forms
    OPEN one per iteration — the fragment-scope form opens it once, on the scope wrapping the anchor
    loop — so a barrier is never elided against a following [Tile_mma]. Hence the loop-carried arm
    (a rotor body's leading barrier, against the previous iteration's bracket) stays pipelined-only,
    and a depth-1 k-block keeps the barrier between its loads and its compute. *)

val apply_classified :
  ?static_indices:Indexing.static_symbol list ->
  schedule ->
  Low_level.optimized ->
  Low_level.optimized
(** Internal candidate-facing variant of {!apply}. An [Invalid_argument] raised inside schedule
    application is transported as a typed {!Schedule_outcome.Illegal_schedule}. *)

type op_verdict = Op_legal | Op_illegal of string | Op_unknown of string
[@@deriving sexp_of, equal]

val op_legality : Low_level.optimized -> optop -> op_verdict
(** gh-494 waypoint 3: the op-legality oracle. A schedule op is a thread-pairing transform over the
    routine's access relations; its obligation is decided by the {!Affine} queries — pairing the
    op's loop symbol(s) as thread identity must leave every write-involving access pair of every
    written node [Disjoint] or [Same_thread] (with the reduction reassociation license for
    [Vectorized] retypes and [Swap]s of accumulations). Three-valued and sound in both proven
    directions: [Op_legal] proves the annotation race-free; [Op_illegal] proves a violation (e.g. a
    materialized node's unguarded write provably independent of the retyped axis); [Op_unknown]
    means "compile and see" — the op may be valid under semantics the queries do not model
    (per-thread scratch copies, renderer fallbacks) — and must never be treated as a rejection.
    Hardware annotations interleave sibling statements' threads with no grid-wide synchronization,
    so a hardware retype/split of a loop sharing an accessed node across its statement boundary
    (with a write on either side) reports [Op_unknown]: such pairs need the default annotator's
    aligned-mapping analysis, which the single-op oracle does not model. [Vectorized] retypes and
    [Swap]s only reorder within the nest and keep the intra-nest scope.

    [Tensorize]'s role assignment is decided: the micro-kernel recognition of apply (a pure function
    of the code) is probed, so an invalid role permutation — e.g. the reduction loop assigned to
    [i]/[j], violating the discipline that the [k] role is the only rmw-carrying loop of the
    accumulation — is a proven [Op_illegal]; on structural success the affine queries decide [i]/[j]
    iteration independence under the reduction-reassociation license (the intrinsic reassociates
    [k]), with the hardware-lane cross-statement downgrade to [Op_unknown] as above. [Stage] is
    likewise probed hermetically (a raising apply is [Op_illegal]), then its implicit contract — the
    staged tile covers the reads it replaces within the staging scope — is the containment query
    [Affine.read_covered_before] over the fresh tile's accesses: a covered non-shared (packing)
    stage is [Op_legal]; shared staging (barrier placement and launch geometry are validated
    downstream) and hoisted staging (link-time host-side packing) report [Op_unknown].
    [Privatize]/[Expand_zero]/[Fuse_epilogue] report [Op_unknown]; their own apply-time
    preconditions remain in force. *)

val schedule_legality : Low_level.optimized -> schedule -> (optop * op_verdict) list
(** Per-op verdicts, each against the code with the preceding ops applied (on a hermetic copy —
    checking never mutates the argument). Stops after a proven-illegal op or a failing application;
    an op that fails to apply reports [Op_illegal] with the exception. *)

val aligned_chains :
  ?max_chain:int ->
  ?expanded_zeros:Tn.t list ->
  Low_level.optimized ->
  (Low_level.t * (Indexing.symbol * int) list) list option
(** The cross-nest analysis behind {!default_gpu} / {!default_cpu}, as data: per top-level loop nest
    (paired with the nest statement itself), the outermost loops that may carry hardware geometry —
    already trimmed to the common aligned prefix each dependency component admits, so chain position
    [k] of two linked nests denotes the same hardware thread coordinate and the two extents at
    position [k] are equal. [None] when the analysis bails, i.e. when the kernel is not safely
    parallelizable at all.

    [max_chain] caps each nest's chain length; the default 2 mirrors the presets' one-Grid +
    one-Workgroup shape. A caller supplying its own geometry per chain position passes the arity it
    needs — a batched matmul site's chain is its batch loops plus row plus column, and capping at 2
    made every rank-3+ site's companion coverage decline (gh-ocannl-569). The alignment rule itself
    is arity-independent.

    The presets consume this to emit their own Grid/Workgroup-per-nest geometry. A sketch pipeline
    (autotune) consumes it to cover the nests it does {e not} build — a tensorized accumulation nest
    binds two [Grid] slots and a [Workgroup] lane, geometry no preset emits, and its companion nests
    must be annotated with the matching slot structure or [Low_level.validate_parallel] rejects
    their writes (gh-ocannl-521). Callers supplying their own geometry must keep the positional
    pairing: the alignment argument is exactly that thread [(c0, c1, ...)] covers the same index
    slice in every linked nest.

    [expanded_zeros] names nodes whose whole-node [Zero_out] the caller's schedule will turn into a
    per-element nest ({!Expand_zero}) carrying the caller's geometry. Such a statement is skipped
    for the query instead of bailing it as a bare materialized write — the write it stands for is
    not bare by the time the code is validated. *)

val default_gpu :
  ?block_size:int ->
  ?min_parallel:int ->
  ?limits:Backend_intf.hardware_limits ->
  Low_level.optimized ->
  schedule
(** The default GPU annotator preset (schedule-ir-optops §6): for each top-level loop nest whose
    parallelism is provable from the lowered code alone, produce ops annotating exactly one [Grid]
    and one [Workgroup] loop (splitting single parallel loops by [block_size], default from config
    [gpu_schedule_block_size] = 256, clamped to [limits]'
    {!field:Backend_intf.max_threads_per_workgroup} when given — the configured block size is a
    target, the device's workgroup capacity a hard cap). A loop is parallelizable when its index
    occurs as a plain [Iterator] component in every materialized write vector beneath it — the same
    coverage property [Low_level.validate_parallel] enforces, used generatively — and the kernel
    passes a conservative race analysis (all accesses to written nodes agree on parallel-index
    components, no [Zero_out] of materialized nodes, no barriers or opaque statements; reduction
    loops stay serial). Cross-nest producer/consumer (or WAW/WAR) pairs over a written node are
    allowed only when {e aligned}: the linked nests' chains are trimmed to a common equal-extent
    prefix — identical annotation geometry, so each hardware thread covers the same index slice in
    every linked nest — and per axis position the paired accesses either both use plain [Iterator]s
    of same-chain-position parallel symbols or neither mentions one; otherwise the analysis bails.
    For non-materialized (per-thread copy) scratch the edge additionally requires value
    thread-invariance at the chosen trim: a chain symbol feeding a scratch write's value without
    pinning the written cell would leave each consumer thread's copy holding its own chunk's last
    value where the serial reference holds the last chunk's, so the trim search serializes that loop
    (gh-494; direct syntactic dependence only). Returns the empty schedule when any check fails or
    when the largest parallelizable nest has fewer than [min_parallel] iterations (default from
    config [gpu_schedule_min_parallel] = 64: a kernel launches either way, so any real parallelism
    beats the serial 1x1 fallback — a single GPU thread is 1-2 orders of magnitude slower than a CPU
    core; the remaining small threshold keeps sub-simdgroup-scale programs fully serial so their
    segments coalesce and placements stay unchanged). *)

val default_cpu : ?min_parallel:int -> Low_level.optimized -> schedule
(** The default CPU annotator preset: the same conservative analysis as {!default_gpu}, but each
    nest's outermost parallelizable loop is merely retyped to [Grid] — pool-backed Grid rendering in
    the C backend (docs/proposals/gh-ocannl-164.md) partitions that loop into contiguous chunks
    executing on a process-global native thread pool, and a [Workgroup] split would only add loop
    structure that runs serially inside a chunk. Returns the empty schedule below [min_parallel]
    (default from config [cpu_schedule_min_parallel] = 16384; task fan-out costs more than a GPU
    launch is worth on small kernels). *)

val log_launches : bool Lazy.t
(** Config [schedule_log_launches]: backend [compile] logs one stderr line per compiled segment with
    its launch grid/block dims and statement count — for diffing what two compiles of nominally
    identical code actually emit. *)

val backend_is_gpu : string -> bool
(** Whether the named backend binds hardware indices (currently: name contains ["cuda"], ["hip"] or
    ["metal"]).

    This and {!backend_is_cpu} are what a test asks to decide whether it may evaluate a GPU-only or
    CPU-only leg, so [test/operations/marker_backend_vocabulary] holds them to covering
    [Backends.all_of_backend] by exactly one each: a backend neither knows would read as CPU at
    every caller phrased [if backend_is_gpu ... else ...]. *)

val backend_is_cpu : string -> bool
(** Whether the named backend renders [Grid] loops on the CPU pool (currently: name contains
    ["cc"]). *)

val automatic_schedule_active : backend_name:string -> bool
(** Whether {!maybe_default_schedule} (and {!maybe_default_schedules}) would apply a default preset
    on this backend rather than the identity: the backend binds Grid loops, the corresponding config
    gate ([automatic_gpu_schedule] / [automatic_cpu_schedule]) is on, and runtime kernel logging
    ([debug_log_from_routines]) is off. Callers that substitute their own default-schedule choice
    (e.g. the cost-model default selection of gh-ocannl-491) must respect this gate. *)

val default_pipeline_fissions : unit -> bool
(** The config [schedule_fission] gate consulted by {!maybe_default_schedules}: whether the untuned
    default pipeline fissions into per-segment kernels (subject to {!automatic_schedule_active}), or
    applies the whole-routine {!maybe_default_schedule} instead. *)

val default_schedule_fingerprint : backend_name:string -> string
(** A stable summary of the configuration that shapes the untuned default pipeline on this backend:
    the {!automatic_schedule_active} gate (["inactive"] when it is off — the untuned default is then
    the unscheduled serial form), the [schedule_fission] gate, and the preset thresholds
    ([gpu_schedule_block_size] / [gpu_schedule_min_parallel] / [cpu_schedule_min_parallel]).
    Diagnostics recorded against one default pipeline (e.g. the autotuner's [default_ms],
    gh-ocannl-552) compare fingerprints to detect that a config change redefined the default (Codex
    P2 on PR #279). Does not cover per-device hardware limits — cache consumers already key per
    backend. *)

val maybe_default_schedule :
  backend_name:string ->
  ?limits:Backend_intf.hardware_limits ->
  static_indices:Indexing.static_symbol list ->
  Low_level.optimized ->
  Low_level.optimized
(** The implicit transform applied by backend [compile] when the caller passes no
    [?lowered_transform]: {!apply} of {!default_gpu} on GPU backends, of {!default_cpu} on CPU
    backends, the identity otherwise. [limits] (default {!Backend_intf.no_hardware_limits}) should
    be the compiling backend's {!Backend_intf.Backend_device_common.hardware_limits}. Disabled by
    config [automatic_gpu_schedule=false] / [automatic_cpu_schedule=false] respectively, and skipped
    when runtime kernel logging ([debug_log_from_routines]) is active, to keep logs serial and
    deterministic. *)

val zero_expansion :
  ?block_size:int ->
  ?min_parallel:int ->
  limits:Backend_intf.hardware_limits ->
  Tnode.t list ->
  schedule
(** The expand-and-annotate schedule {!maybe_default_schedules} applies to a fission segment of
    materialized whole-node [Zero_out]s on GPU backends: {!expand_zero} plus the same Grid/Workgroup
    geometry policy as {!default_gpu}. Below [min_parallel] (largest node) the zeros stay whole-node
    (a serial kernel renders them as [memset]). Exposed for callers (e.g. the autotuner) that
    replicate the default fission pipeline with custom per-segment schedules. *)

val fission_scheduled :
  ?promote_locals:bool ->
  ?arity_cuts:bool ->
  preset:(Low_level.optimized -> schedule) ->
  zero_sched:(Tnode.t list -> schedule) ->
  static_indices:Indexing.static_symbol list ->
  Low_level.optimized ->
  ([ `Normal | `Zeros | `Solo ] * Low_level.optimized * schedule * Low_level.optimized) list
(** The kernel-fission pipeline underlying {!maybe_default_schedules}, with caller-supplied
    per-segment schedules and a per-segment result: the routine's top-level statements are
    partitioned at cross-workgroup dependency edges exactly as described there (edges the aligned
    cross-nest rule of {!default_gpu} proves race-free without losing any nest's standalone
    parallelism do not cut), [preset] is called on each [`Normal] segment's (pre-schedule)
    [optimized] slice and [zero_sched] on each [`Zeros] segment's nodes, and each result tuple
    carries the segment kind, the pre-schedule segment, the schedule chosen for it, and the
    scheduled segment ({!apply} of the schedule). [`Solo] segments (opaque to the analysis, or
    coalesced runs of unannotated segments) get the empty schedule. When fission does not apply
    (single segment, unfissionable crossings, or everything coalesces back) the result is a single
    [`Normal] tuple over the whole routine with [preset]'s schedule. Callers compile each scheduled
    segment as its own kernel in order (the plural transform seam of backend [compile]); see
    {!maybe_default_schedules} for the synchronization contract.

    [arity_cuts] (default [false], gh-ocannl-574): segment for the {e full-arity} sketch pipelines
    instead of the default presets. The no-parallelism-loss guard normally compares chains under the
    presets' Grid+Workgroup cap, so a merge that trims a rank-3+ site's minor axis reads as lossless
    — correct for the default annotators, which never bind more than that prefix, but the GPU matmul
    sketches annotate the site's whole chain and cover every materialized-writing nest of the kernel
    with the same geometry ([companion_geometry], gh-ocannl-521/569). Under [arity_cuts] chains are
    analyzed uncapped and a merge additionally requires every materialized-writing nest to keep an
    identical extent list, so a companion that cannot follow the site's arity — a reduction {e over}
    the site's minor axis and the reduction target's initialization nest, the lm_head's max-logits
    row — is cut into its own downstream kernel (stream order supplies the synchronization) and the
    site's kernel seeds at full arity. The finer segmentation costs launches the default pipeline
    does not want to pay unconditionally, so this is a candidate-generation mode (the autotuner
    times it), never the default.

    [promote_locals] (default [false]): promote statement-crossing [Local] scratch to [On_device]
    before segmentation. A nest whose only writes land in [Local] scratch gets no parallel chain
    (the annotator's coverage property quantifies over materialized writes), and Local
    producer/consumer edges do not cut — so such producers either drag their segment down to a
    serial 1x1 launch or are redundantly re-executed by every hardware thread; small reduction
    intermediates (layer-norm statistics, softmax max/denominator) land exactly here via the [Local]
    stack threshold. Promotion lets the ordinary materialized machinery apply; promotions fission
    does not end up needing (single-kernel fallbacks, or all accesses within one segment after
    coalescing) are restored. {!maybe_default_schedules} passes [true] on GPU backends, where a
    serial nest costs orders of magnitude more than on CPU. *)

val maybe_default_schedules :
  backend_name:string ->
  ?limits:Backend_intf.hardware_limits ->
  static_indices:Indexing.static_symbol list ->
  Low_level.optimized ->
  Low_level.optimized list
(** Like {!maybe_default_schedule}, but with kernel fission: the routine's top-level statements are
    partitioned into segments at the cross-workgroup dependency edges the default annotator's
    interference analysis would otherwise reject wholesale — materialized producer/consumer (and
    WAW/WAR) pairs of sibling statements, bare materialized writes, materialized whole-node
    [Zero_out]s, statements opaque to the analysis — and each segment receives the default schedule
    independently, on its own launch geometry. The caller compiles each returned [optimized] as its
    own kernel and runs them in order on the routine's stream, chaining a device-side event at each
    boundary — that event supplies the grid-wide synchronization at each cut (queue FIFO alone does
    not order overlapping command buffers over Metal's untracked resources). Scalar scope-locals
    hoisted across statements are replicated into consuming segments when provably value-preserving
    (segments merge back and run serially otherwise); [Local]-placed scratch whose live range
    crosses a cut is promoted to [On_device] in the compile's placement fork. On GPU, segments
    consisting of materialized [Zero_out]s are expanded ({!expand_zero}) and annotated like ordinary
    nests. Adjacent segments that end up unannotated are coalesced, so an all-serial routine stays a
    single kernel. Returns a singleton equal to {!maybe_default_schedule}'s result when fission does
    not apply (non-GPU/CPU backend, automatic scheduling disabled, config [schedule_fission=false],
    kernel logging active, or nothing to split). *)

type launch_geometry = {
  lg_grid_y : int option;
  lg_grid_z : int option;
  lg_block_x : int option;
  lg_block_y : int option;
  lg_block_z : int option;
}
(** As much of a candidate's launch geometry as its caller knows (gh-ocannl-709): the two grid
    dimensions backends cap and the workgroup's three, [None] meaning "not predicted here" — which
    exempts that dimension rather than refusing on it. Five fields, not six: [grid.(0)] is
    2^31-scale on every backend that binds hardware axes, so no backend has a cap to report for it.
    {!check_hardware_limits_classified} fills all five from the lowered code
    ({!launch_geometry_of_dims}); autotune's seeding pre-filter predicts what it can from the
    parameters it is about to commit to. A prediction is a {e lower bound} — it describes the site's
    own nest, while {!Low_level.launch_dims} maxes over the kernel's zeroing and companion nests too
    — so an under-prediction costs a compile the gate then declines, and only an over-prediction
    could withhold a legal candidate. *)

val unknown_launch_geometry : launch_geometry
(** All-[None]: predicts nothing, refuses nothing. The base a family builds its prediction on. *)

val launch_geometry_of_dims : Low_level.launch_dims -> launch_geometry
(** The geometry a lowered kernel will actually launch with — every field [Some]. *)

type launch_excess = {
  lx_resource : Schedule_outcome.resource;
  lx_requested : int;
  lx_limit : int;
  lx_phrase : string;
}
(** The first dimension of a geometry the device refuses. [lx_phrase] is the verb phrase both
    callers render — e.g.
    ["requests a .z workgroup extent of 128, exceeding the device limit of 64"] — so a gate [detail]
    and a seeding refutation witness say the same thing about the same candidate. *)

val launch_geometry_excess :
  limits:Backend_intf.hardware_limits -> launch_geometry -> launch_excess option
(** The one static reading of a device's per-dimension launch caps (gh-ocannl-709), consulted both
    by {!check_hardware_limits_classified} (on the lowered code's geometry) and by autotune's
    seeding pre-filter (on a candidate's predicted geometry), so the two cannot disagree about what
    a device permits and a cap is never encoded twice. [None] = every predicted dimension fits. Rows
    are tested in the order [.x]/[.y]/[.z] workgroup, [.y] grid, [.z] grid fold; a [None] cap
    ({!field:Backend_intf.max_workgroup_dims} on the C backends, {!field:Backend_intf.max_grid_yz}
    on Metal) exempts its dimensions. The workgroup thread PRODUCT is deliberately not a row: it is
    not a per-dimension geometry question, and only the gate checks it. *)

val check_hardware_limits :
  name:string -> limits:Backend_intf.hardware_limits -> Low_level.optimized -> unit
(** Validates the scheduled kernel [name] against the device limits, raising [Utils.User_error] on
    violation: the launch's workgroup size (the product of {!Low_level.launch_dims}' block
    dimensions) against {!field:Backend_intf.max_threads_per_workgroup}, the total bytes of
    [workgroup_shared] tiles against {!field:Backend_intf.max_workgroup_memory_bytes}, and then the
    launch geometry dimension by dimension through the shared {!launch_geometry_excess} — the
    workgroup's [.x]/[.y]/[.z] extents against {!field:Backend_intf.max_workgroup_dims} (a separate
    hardware fact from the thread product: CUDA caps [maxThreadsDim.z] at 64 while the product cap
    is 1024 — gh-ocannl-679), and both 16-bit-capped grid dimensions, the [.y] extent ([grid.(1)], a
    blocktiled matmul's row-block count) and the folded [.z] extent ([grid.(2)], the product of the
    Grid slots [>= 2] — gh-ocannl-643), against {!field:Backend_intf.max_grid_yz}. [grid.(0)] is
    deliberately ungated: it is 2^31-scale wherever hardware axes bind. Backend [compile] calls this
    after any [?lowered_transform] (or the default annotator, which already respects the limits),
    turning driver-level launch failures into early, named errors. A no-op for all-[None] limits. *)

val check_hardware_limits_classified :
  name:string -> limits:Backend_intf.hardware_limits -> Low_level.optimized -> unit
(** Internal candidate-facing variant of {!check_hardware_limits}; transports excess thread,
    workgroup-memory, per-workgroup-dimension, [.y]-grid or folded-[.z]-grid requests as typed
    {!Schedule_outcome.Resource_exceeded} causes, one variant per launch dimension
    ({!Schedule_outcome.Workgroup_x_extent} .. {!Schedule_outcome.Grid_z_extent}) so an autotune
    search's declines group by the dimension that asked too much. *)

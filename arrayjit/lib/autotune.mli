(** {1 Empirical schedule search (autotuning)}

    tinygrad-style beam search over {!Ir.Schedule} transforms, timed on the real device
    (docs/proposals/schedule-ir-optops.md; the search-harness half of the OptOps port). {!tune} is a
    drop-in replacement for {!Context.compile}: it compiles candidate schedules through the
    [?lowered_transform] seam, times each on the context's device, and returns the routine of the
    fastest one. Every candidate (and the winner replay) derives from a hermetic copy of the {e one}
    base lowering captured at the start — each candidate compile's own fresh lowering is ignored,
    because timing runs settle tensor-node value bounds and later lowerings can fold guards or
    re-segment fission differently, silently corrupting digest comparisons and replays. Winning
    schedules are persisted, in the structurally-rebindable saved form of {!Ir.Schedule_cache}, to a
    disk cache keyed by the code's canonical digest and the backend, so a re-run of the same program
    skips the search (cross-process replay is guarded by digest equality against that process's own
    base lowering).

    The candidate space:

    - {b Whole-routine presets}: the serial baseline, the default annotator, and a block-size sweep
      through {!Ir.Schedule.default_gpu}. On GPU backends the serial baseline (and any candidate
      that degenerates to it) is enumerated but never dispatched: with no hardware dimension bound
      the whole routine runs in one work-item, which cannot win and whose cost is unbounded — hours
      of uninterruptible dispatch on a device shared with the display (gh-ocannl-532). It keeps its
      role as the search's starting point for menu moves and as the code every candidate derives
      from; it just carries no measurement. On CPU backends it runs at full single-core speed and is
      timed as before. Every such refusal enters the report's decline census under
      [Not_dispatched_key] (gh-ocannl-543): [candidates_timed] alone cannot distinguish a GPU search
      whose serial candidates were all refused from one whose candidate space was empty. Beam rounds
      expanding an incumbent that was never dispatched propose only the moves that can bind a
      hardware dimension ([Tensorize], placement retypes); the rest are pruned before compile and
      counted in the same census. Separately, the baseline's compile is protected like any
      candidate's (gh-ocannl-533): a typed rejection — a large softmax/cross-entropy head whose
      unscheduled form exceeds the HIP scratch budget is the case in hand — declines the baseline
      ([baseline_declined] in the report, and its own key in the census rather than a
      [Not_dispatched] refusal, which would assert a reason that is not the one) and the search
      proceeds on the scheduled candidates, which fission and placement promotion routinely bring
      back within budget. Only the base lowering itself is indispensable: a failure before it is
      captured has no search to run and propagates.
    - {b Fissioned candidates}: the kernel-fission pipeline ({!Ir.Schedule.fission_scheduled}) with
      per-segment schedules — the same preset sweep per segment, and beam rounds that extend
      {e one segment at a time}. Per-segment schedules are cached keyed by the pre-schedule
      segment's canonical digest. [`Zeros] segments keep the default zero-expansion; [`Solo]
      segments stay unscheduled. One seed uses the config-default thresholds, reproducing the
      untuned default pipeline exactly — so the winner is never worse than not tuning, even on
      launch-overhead-bound workloads where every aggressive preset loses to it; its measured time
      is surfaced as the report's [default_ms] reference (gh-ocannl-552). Each preset is
      additionally seeded in a {e privatized} variant ({!extend_with_privatize}): per segment, every
      materialized read-modify-write accumulator is contracted into a per-thread register tile
      ({!Ir.Schedule.optop.Privatize}) over its serial reduction loop where the op's preconditions
      permit — a routine-local accumulator beats a device-memory RMW, and on Metal it sidesteps the
      volatile scalar-RMW workaround.
    - {b Matmul sketches}: when a matmul micro-kernel is detected, parameterized instantiations of
      the composed pipelines pinned by the schedule tests — register blocktiling (Split + Swap +
      shared Stage + Privatize + materializing Unroll) on GPU backends, operand packing (non-shared
      Stage + Privatize) on CPU backends — with dividing tile sizes. When the backend reports an mma
      capability, additionally the {e tensorized} pipelines (docs/proposals/tensorize-mma.md): Split
      into Grid blocks + [Tensorize] targeting [simdgroup_matrix]/tensor cores, both unstaged (one
      full-reduction [Tile_mma] block) and cooperatively staged through shared tiles (lane-aware
      Stage) — Stage-only by design, [Privatize] would move the accumulator into thread-space the
      MMA loads cannot address. On the C backends the tensorized whole-triple and Grid-split-row
      forms are seeded regardless of [limits.mma] — their [Tile_mma] renders as the register-tiled
      vector micro-kernel. Seeding matters because the beam cannot reach these compositions
      incrementally: a bare [Tensorize] from the serial baseline loses its round and is discarded
      before Grid retypes could join it. The sketches are seeded whole-routine {e and} per fission
      segment: on a fissionable computation, the fission segmentation is enumerated once and the
      sketch pipelines are instantiated for each segment where a matmul site is detected (keyed by
      the segment's pre-schedule digest), the remaining segments keeping the default preset. A
      segment's site has its [Zero_out] in a separate [`Zeros] segment, so the pipelines skip the
      zero-expansion geometry there — sound because [Privatize] init-loads the accumulator tile from
      the (pre-zeroed) target and [Tile_mma] loads the accumulator fragment before the reduction. On
      GPU backends the segmentation is additionally enumerated under
      {!Ir.Schedule.fission_scheduled}'s [arity_cuts] (finer) mode (gh-ocannl-574): a segment
      carrying a companion that cannot follow its site's full arity — the lm_head GEMM with its
      max-logits row reduction — has every seed of the shared segment decline on companion coverage,
      and the finer cut frees the site into its own kernel; segments whose digest is new versus the
      coarse segmentation seed [fine]-flagged singles, one composite recombines the fine keys'
      best-timed singles (coarse-timed bests staff the digest-identical segments), and a fine winner
      records the mode in its cache entry so replay re-segments identically.
    - {b Convolution sketches} (gh-ocannl-493): when a convolution accumulation site is detected
      ({!detect_conv}), the implicit-GEMM pipeline — the packing [Stage] serving as im2col, the
      micro-kernel the ordinary [Tile_mma]. On the C backends: serial and Grid-parallel flavors, the
      latter adopting the default preset's aligned whole-segment Grid geometry on merged segments
      (lenet's conv+bias/relu+pooling). On GPU backends with an mma capability: the staged flavor —
      outer output loops Grid-typed, both slices staged through cooperative shared tiles at the
      kernel-window anchor, the accumulator fragment resident across the window (gh-ocannl-480).
      Strided rows (stride-2 stems and downsample blocks) are seeded on both legs since the
      compacting [Stage] (gh-ocannl-502) packs the strided window densely.
    - {b Split-reduce seeds} (gh-ocannl-484 task 3): when a reduction-dominated site is detected
      ({!split_reduce_sites} — an rmw accumulation, or the gh-466 [Set_dynamic] scatter, whose
      target has little output parallelism while a long serial reduction loop feeds it: bias and
      weight gradients of convolutions, softmax denominators, skinny split-K GEMMs), the
      deterministic two-pass split reduction ({!Ir.Schedule.constructor-Split_reduce}) with a few
      [num_blocks] values as the tunable. The prelude applies {e whole-routine before fission}: the
      per-block partials edge it mints is exactly the materialized cross-nest edge kernel fission
      cuts at, so the two passes compile as separate kernels with the event chain supplying the
      grid-wide synchronization the combine needs, and each segment then gets the default preset
      (the block loop parallelizes pass 1). Sites are seeded as singles plus one composite
      recombining each site's best-timed [num_blocks]. A split winner persists as prelude +
      post-prelude per-segment schedules; note the numerics pin — the combine tree is a function of
      the schedule, so retuning may change low bits (see the schedule-cache docs).
    - {b Beam-round menu actions} on the incumbents: dividing serial Splits, Swaps of perfect serial
      pairs, Unrolls, Retype-Vectorized on innermost loops (explicit SIMD on CPU including the
      reduction-chains rendering of accumulations — gh-ocannl-468 — while GPU accumulations stay
      excluded; 128-bit packed loads/stores on GPU — gh-ocannl-463), and Tensorize role permutations
      when the backend reports an mma capability — including the CPU backends, whose [Tile_mma]
      renders as the register-tiled vector micro-kernel (gh-ocannl-469). The loop enumeration
      descends into the [Local_scope] bodies the accumulation mints of [Unroll ~materialize:true] /
      [Partition] create (gh-ocannl-666, matching [Schedule.rewrite_loop]'s reach since
      gh-ocannl-639), so a materialized unroll or partition does not hide the segment/inner loops
      from later rounds; scope-nested loops are proposed only for the ops that survive scope nesting
      — [Tensorize]'s hardware lane loop does not, so its triples stay statement-level.

    Caveats (v1):

    - Timing runs execute the routine several times, mutating its outputs (and accumulators — a
      non-idempotent routine, e.g. gradient accumulation, will accumulate the timing runs).
      Initialize inputs before tuning, and tune before meaningful state exists, or re-initialize
      afterwards.
    - Timing uses wall clock around a device sync, so it includes queue overhead; times are
      min-of-N, where fast routines get extra runs beyond [repeats] until ~25 ms of accumulated
      per-launch samples — not queued-batch wall. Queued timing can consequently spend hundreds of
      milliseconds on its timed batches and calibration; {!time_routine} documents the exact
      dispatch bounds. On sub-millisecond kernels a min-of-3 is launch-jitter roulette and can crown
      the wrong candidate. Static indices are bound to the midpoint of their declared ranges during
      timing and restored afterwards.

    Implementation note: the {e structured} half of the candidate space — matmul/conv site
    detection, the composed schedule pipelines those sites parameterize, and the refinement trees
    whose leaves are the seed lists — lives in [sketch_families.ml] and is included here
    (gh-ocannl-580). This interface is unchanged by that split and remains the library's only gate;
    the family entry points below ({!sketch_params}, {!detect_conv}, {!matmul_sketch_tree},
    {!sketch_schedule}, {!sketch_path_traffic_floor}, …) are defined there, and
    {!sketch_seed_params} is the composition the search enumerates. *)

open Base

type sketch_params = {
  sk_gpu : bool;
  sk_mma : bool;
  sk_simd : int;
  sk_bm : int;
  sk_bn : int;
  sk_bk : int;
  sk_tm : int;
  sk_tn : int;
  sk_hoist : bool;
  sk_grid : bool;
  sk_pack_rest : bool;
  sk_conv : bool;
  sk_epilogue : bool;
  sk_batch_grid : bool;
  sk_swizzle : Ir.Low_level.swizzle_kind option;
  sk_depth : int;
  sk_pack_prec : Ir.Ops.prec option;
  sk_tile : Ir.Register_tile.t option;
}
(** Parameters of one matmul-sketch seed candidate; see the implementation's field docs. Exposed for
    tests (the seeding pre-filter of gh-ocannl-479 and the mixed grid-outermost shape of
    gh-ocannl-473 are asserted on directly). *)

type matmul_site = {
  m_i : Ir.Indexing.symbol;
  m_j : Ir.Indexing.symbol;
  m_k : Ir.Indexing.symbol;
      (** The innermost contraction loop — the one a pipeline's k-split divides; its extent is
          [m_nk]. *)
  m_ni : int;
  m_nj : int;
  m_nk : int;
  m_ko : (Ir.Indexing.symbol * int) list;
      (** Contraction loops enclosing [m_k], in nest order (gh-ocannl-683): a multi-axis contraction
          — attention's out projection, whose weight carries two input axes — is a k-loop lowering
          has already split, and every pipeline treats these as k-block loops above the one its own
          k-split mints. Empty on single-axis contractions. *)
  m_bo : (Ir.Indexing.symbol * int) list;
  m_bi : (Ir.Indexing.symbol * int) list;
  m_row_axis : int;
  m_d : Ir.Tnode.t;
  m_a : Ir.Tnode.t;
  m_b : Ir.Tnode.t;
  m_zeroed : bool;
  m_tb : bool option;
  m_fma : bool;
}
(** A recognized matmul accumulation site; see the implementation's field docs. Exposed for tests
    (gh-ocannl-683 asserts on the contraction nest directly). *)

val detect_matmul : Ir.Low_level.t -> matmul_site option
(** Recognize a matmul accumulation nest: a perfectly nested all-serial accumulation whose
    contraction loops are the innermost suffix absent from the accumulator's index map, every other
    loop owning a distinct accumulator axis, with the 2-D tile roles assigned per
    [classify_matmul]'s operand rules and everything else batch. Reads off the extracted access
    relations like {!detect_conv}, with the same [legality_crosscheck] soak. Exposed for tests. *)

type conv_axis = {
  cx_o : Ir.Indexing.symbol;  (** Output spatial symbol (a plain iterator of the output). *)
  cx_no : int;
  cx_k : Ir.Indexing.symbol;  (** Kernel-window symbol (read by the kernel, not the output). *)
  cx_nk : int;
  cx_stride : int;
  cx_dilation : int;
  cx_offset : int;  (** Padding offset on the input access ([<= 0] for padded convs). *)
}

type conv_site = {
  c_loops : Ir.Indexing.symbol list;
  c_outer : (Ir.Indexing.symbol * int) list;
  c_kernel : Ir.Indexing.symbol list;
  c_axes : conv_axis list;
  c_row : Ir.Indexing.symbol;
  c_nrow : int;
  c_oc : Ir.Indexing.symbol;
  c_noc : int;
  c_red : Ir.Indexing.symbol;
  c_nred : int;
  c_d : Ir.Tnode.t;
  c_a : Ir.Tnode.t;
  c_b : Ir.Tnode.t;
  c_zeroed : bool;
  c_fma : bool;
}
(** A recognized convolution accumulation site (gh-ocannl-493); see the implementation's field docs.
    Exposed for tests. *)

val detect_conv : Ir.Low_level.t -> conv_site option
(** Recognize a convolution accumulation nest: the output written at plain distinct iterators, one
    operand carrying affine components that mix an output symbol with a kernel-window symbol (the
    projections carry the strides, dilations, and padding offsets), the other operand reading the
    kernel window, exactly one out-channel and one reduction-channel symbol, with the out-channel at
    the output's last axis and a conv axis at its second-to-last (the implicit-GEMM row). Reads off
    the extracted access relations ([Ir.Low_level.affine_accesses] — the gh-494 artifact the
    op-legality oracle also consumes); under config [legality_crosscheck] the retained procedural
    matcher runs alongside and any divergence raises. Exposed for tests. *)

val matmul_launch_geometry : matmul_site -> sketch_params -> Ir.Schedule.launch_geometry
(** The launch geometry a GPU matmul seed will have, predicted from the parameters alone
    (gh-ocannl-709) — the grid's row-block [.y] extent and folded batch [.z] product, and the
    workgroup's register-split extents — so seeding can consult the same
    {!Ir.Schedule.launch_geometry_excess} the pre-driver gate does and never propose a candidate the
    device would refuse. {!Ir.Schedule.unknown_launch_geometry} for parameters that name no block
    geometry (every CPU pipeline: the C backends render annotated loops serially). Exposed for
    tests, which cross-check the prediction against an applied schedule's
    {!Ir.Low_level.launch_dims} — a prediction that drifted from what the builders emit would
    withhold legal candidates. *)

val conv_launch_geometry : conv_site -> sketch_params -> Ir.Schedule.launch_geometry
(** The launch geometry a GPU convolution seed will have (gh-ocannl-739): the site's outer output
    loops followed by the blocked flavor's row-block loop in grid nest order, and the tensorization
    lane as the sole workgroup loop. This lets convolution seeding consult
    {!Ir.Schedule.launch_geometry_excess} before compiling a candidate. The prediction is a lower
    bound on the applied schedule's geometry; tests derive a real conv site through
    {!Ir.Schedule.fission_scheduled} and cross-check every GPU seed against
    {!Ir.Low_level.launch_dims}. {!Ir.Schedule.unknown_launch_geometry} for CPU parameters. *)

val sketch_seed_params :
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  sketch_params list
(** The matmul-sketch seeds proposed for the given lowering: parameterized instantiations of the
    composed pipelines with dividing tile sizes, pre-filtered against rules that statically imply a
    declined rendering (gh-ocannl-479) — on GPU backends: the (operand, operand, accumulator) format
    tile advertised by [limits.mma.mma_format_tiles], including policy-enabled TF32 and excluding
    combinations the backend supports at one accumulator width but not the other (gh-ocannl-545:
    CUDA's bf16 has no wmma accumulator of its own); on the C backends: operand-precision uniformity
    (f32/f64), the fused accumulation form, micro-kernel column extent at least one vector of lanes
    ([limits.simd_vector_bytes]), and transposed-B storage for shapes that read B in place. Builder
    preconditions the schedule builders settle identically for every tile completion — the
    zero-expansion row-axis rule, and the GPU sketches' companion-coverage rule per fusion flavor —
    also pre-filter here, as the family tree's construction-time refutations (gh-ocannl-577): a seed
    list never proposes a candidate that statically must fail its build. For the matmul family this
    list {e is} {!Ir.Schedule_space.leaves} of {!matmul_sketch_tree}, epilogue twins included.
    Exposed for tests. *)

module Family_decision : sig
  (** {1 What a commitment on the matmul family tree is (gh-ocannl-591)}

      The family tree's levels commit to values of {!t}, not to display strings. A consumer that
      reads a decision back off a path — the certain-traffic floor {!sketch_path_traffic_floor}, the
      lattice lift {!lift_geometry_lattice}, a ranking or profitability pass over the search's
      paths, the tests — matches on the datum. {!to_label} is the rendering, used by logs, decline
      reports and goldens; nothing parses it back, so rewording a label (or renaming a level, which
      {!level} derives from the datum) changes what is printed and nothing else.

      This replaced a [Printf.sprintf] / [Scanf.sscanf] protocol whose failure mode was silent: a
      reworded geometry label made every scan arm fall through, so the traffic floor's increment was
      [0] on every path — a sound bound, so nothing raised, no golden moved, and the family bound
      quietly stopped differentiating the tree. *)

  type geometry = { g_bm : int; g_bn : int; g_bk : int; g_tm : int; g_tn : int }
  (** A committed tile geometry. Which fields are meaningful is the {!geometry_choice} constructor's
      business: [g_bm]/[g_bk] always; [g_bn] is [0] for {!Cpu_packed}'s unsplit full column extent
      and the mma lane width for {!Gpu_mma}; [g_tm]/[g_tn] are the per-thread tile of
      {!Gpu_blocktile} and [0] elsewhere. [g_bk = 0] in {!Gpu_mma} is the unstaged full-K block. *)

  type geometry_choice =
    | Gpu_blocktile of geometry
        (** The GPU scalar blocktile menu: both operand tiles staged in kernel. *)
    | Gpu_mma of geometry
        (** The GPU tensorized menu: [g_bk > 0] stages both operand tiles in kernel, [g_bk = 0] is
            the unstaged full-K block. *)
    | Cpu_blocktile of int  (** The CPU blocktile menu's single block size (bm = bn = bk). *)
    | Cpu_packed of geometry
        (** The CPU packed composition; what it costs depends on the {!Packing_shape} above it. *)
    | Lattice
        (** The staged tile-size lattice beyond the curated menu (gh-ocannl-514 phase 5), excluded
            by default policy and lifted by {!lift_geometry_lattice}; its axes commit as
            {!Lattice_box}. *)

  (** One committed decision. Each constructor belongs to exactly one level ({!level}) and carries
      the whole identity of the commitment: no consumer needs the level name, or the label, to know
      what was decided. *)
  type t =
    | Fusion of [ `Unfused | `Fused ]  (** The root: the epilogue-fusion flavor (gh-ocannl-613). *)
    | Pipeline of [ `Blocktile | `Tensorized ]  (** Which composed pipeline. *)
    | Batch of [ `Serial | `Grid ]  (** The batch-geometry twin (gh-ocannl-643), GPU only. *)
    | Packing of [ `In_kernel | `Hoisted ]
        (** The CPU blocktile pipeline's link-time packing twin (gh-ocannl-470). *)
    | Geometry of geometry_choice  (** The tile geometry, per the pipeline's own menu. *)
    | Lattice_box of { lb_axis : [ `Bm | `Bk ]; lb_lo : int; lb_hi : int }
        (** One binary interval refinement of a lattice axis: the value range still open below the
            commitment, [lb_lo = lb_hi] at a singleton. A box prices at [lb_lo], its most favorable
            corner. *)
    | Twin of [ `Plain | `Swizzled | `Depth of int ]
        (** The per-staged-geometry twins: the swizzled staged layout, the pipelined depths. *)
    | Tensorized_form of [ `Whole_triple | `Packed ]  (** The CPU tensorized composition. *)
    | Row_block of int
        (** The CPU whole-triple row block; [0] is the unsplit form, [> 0] a pool-rendered Grid
            split. *)
    | Packing_shape of
        [ `Serial | `Hoisted | `Hoisted_grid | `Hoisted_grid_pack_rest | `Grid_pack_rest | `Grid ]
        (** Which CPU packed composition: where the panels are packed (in kernel, at link time, per
            Grid chunk) — what makes a packed geometry's traffic additional or merely relocated. *)
    | Register_tile of Ir.Register_tile.t option
        (** The CPU tensorized micro-kernel's C-tile geometry (gh-ocannl-619): [None] is the
            renderer's own ranking-model choice, [Some] a schedule-carried alternative the renderer
            honours exactly. The level appears only where {!Ir.Register_tile.alternatives} offers
            one. *)

  type path = (string * t) list
  (** What {!Ir.Schedule_space.enumerate} and the [~path] of {!Ir.Schedule_space.search} carry at
      this label type: the committed vector, outermost level first. The [string] is the level's
      display name; the decision is the identity. *)

  val equal : t -> t -> bool
  val compare : t -> t -> int

  val level : t -> string
  (** The level a decision belongs to — the name {!Ir.Schedule_space.Choice} carries. Derived from
      the datum, so a node's level and its children's identities cannot drift apart. *)

  val to_label : t -> string
  (** The display rendering. Nothing reads it back. *)

  val render_path : path -> string
  (** A path as ["level=label > level=label > …"], for logs and reports. *)
end

val matmul_sketch_tree :
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree option
(** The matmul sketch family as a refinement tree (gh-ocannl-514 phase 1): decision levels —
    epilogue fusion, pipeline, packing shape, geometry, twins — whose lazily-refined choices depend
    on earlier commitments, and whose {!Ir.Schedule_space.leaves} are exactly the family's
    {!sketch_seed_params}, in enumeration order. [None] when no matmul site is detected. The root
    level is the [Fuse_epilogue] flavor (gh-ocannl-613): [unfused] first, then [fused] — refuted
    with the fusion recognizer's reason ([Schedule.fuse_epilogue_witness]) when the site's output
    feeds no fusable tail — so the twins enumerate after every unfused leaf and each flavor's
    construction-time verdicts are its own (gh-ocannl-577: companion coverage can pass fused where
    it fails unfused). Statically-decidable builder preconditions refute at the pipeline level —
    above the geometry menus and the tile lattice, so a family whose every completion must fail its
    candidate build never expands (gh-ocannl-577). The conv family factors the same way as a
    follow-up. Exposed for tests and as the phase-4 search driver's entry into the family space. *)

val geometry_lattice_witness : string
(** The exclusion witness marking the tile-size lattice branches beyond the curated geometry menus
    (gh-ocannl-514 phase 5): binary interval refinements over the staged tile sizes — every
    intrinsic-tile multiple of the row block crossed with every staged depth block — whose boxes
    carry corner-judged verdicts (a workgroup-memory floor at the most favorable corner refutes the
    whole box pre-expansion). {!Ir.Schedule_space.leaves} never enumerates an excluded branch, so
    the tuner's seed lists are unchanged by the lattice's existence. *)

val lift_geometry_lattice :
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree ->
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree
(** Lift every geometry-lattice exclusion — the branches whose decision is
    [Family_decision.Geometry Lattice], identified by that datum rather than by the exclusion's
    prose witness — preserving the laziness of everything else; lifted branches remain subject to
    legality (the box refutations), and other exclusions stay excluded. {!model_default}'s family
    search applies this under config [model_default_geometry_lattice]. Exposed for tests. *)

val sketch_path_traffic_floor :
  limits:Ir.Backend_intf.hardware_limits -> Ir.Low_level.optimized -> Family_decision.path -> int
(** The certain-traffic increment (bytes) of a family-tree decision path (gh-ocannl-514 phase 5):
    traffic every completion below the path moves beyond the schedule-invariant
    {!Ir.Cost_model.completion_floor}, read off the path's committed staging decisions as {e data}
    ({!Family_decision}, gh-ocannl-591; the decision also says which pipeline minted it, so the
    caller's backend kind is not needed alongside the path) — a committed staged geometry
    contributes its operand tiles' distinct-cell footprints exactly as {!Ir.Cost_model.analyze}
    charges them on every leaf (in-kernel GPU stages read and write; CPU packed panels only under
    in-kernel [serial] packing — a hoisted panel replaces the original operand's reads rather than
    adding traffic), and a lattice box contributes its most favorable (smallest-tiles) corner priced
    at the same per-format intrinsic tile the lattice is built from, so the increment is monotone in
    refinement. [0] when no matmul site is detected or nothing is certain. Composed with the floor's
    legs, this is what makes the family bound non-uniform across the tree — the schedule-invariant
    floor differentiates only placements (phase 3), the increments differentiate the sketch-geometry
    subtrees. Detection runs once at partial application; the returned closure is cheap per path.
    Exposed for tests. *)

val sketch_schedule : p:sketch_params -> Ir.Low_level.optimized -> Ir.Schedule.schedule
(** The composed pipeline a seed parameterizes, built against the given lowering (the site is
    re-detected). Raises [Invalid_argument] when no site is detected or the parameters do not fit
    the segment. Exposed for tests (the pad-composition seeding of gh-ocannl-485 is executed
    directly). *)

val extend_with_privatize :
  static_indices:Ir.Indexing.static_symbol list ->
  Ir.Schedule.schedule ->
  Ir.Low_level.optimized ->
  Ir.Schedule.schedule
(** The privatized preset extension used by the fissioned candidates: appends a
    [Schedule.Privatize { target; over }] for every materialized read-modify-write accumulator
    detected in the schedule's application to the segment — [over] being the outermost enclosing
    [Serial] loop whose symbol the access vector does not mention and whose subtree contains no
    hardware-typed loop. Each proposal is validated by try-applying the grown schedule against a
    hermetic copy of the segment (proposals violating the op's preconditions are dropped), so the
    result always applies cleanly where the input schedule does. Exposed for tests. *)

type sr_site = {
  sr_axis : Ir.Indexing.symbol;  (** The reduction loop to split. *)
  sr_target : Ir.Tnode.t;  (** The accumulated node. *)
  sr_red : int;  (** The reduction loop's extent. *)
  sr_out : int;  (** The target's cell count — the site's whole output parallelism. *)
  sr_cost : int;
      (** Estimated segment cost: the accumulation statement's trip count (the product of every
          enclosing loop extent) — the serial work a split could recover. The ranking key
          (gh-ocannl-541). *)
  sr_dynamic : bool;  (** The gh-466 scatter form ([Set_dynamic]). *)
  sr_swaps : (Ir.Indexing.symbol * Ir.Indexing.symbol) list;
      (** The gh-ocannl-537 enabling interchange: [(outer, inner)] [Swap]s applied {e in order}
          before the [Split_reduce], each hoisting an accumulation-cell loop outside [sr_axis].
          Empty when the site is splittable as lowered. *)
}
(** A reduction-dominated accumulation site eligible for {!Ir.Schedule.constructor-Split_reduce}
    seeding (gh-ocannl-484 task 3); see the implementation's field docs. Exposed for tests. *)

val split_reduce_sites :
  ?static_indices:Ir.Indexing.static_symbol list -> Ir.Low_level.optimized -> sr_site list
(** The split-reduce seeding sites of the given lowering: rmw accumulations (and gh-466
    [Set_dynamic] scatters) whose target has at most a few thousand cells while a serial reduction
    loop of substantial extent feeds it — little output parallelism, lots of splittable reduction
    work. Per site the largest-extent enclosing serial loop that passes the hermetic
    {!Ir.Schedule.op_legality} probe of the corresponding [Split_reduce] is chosen (the op's own
    recognizer decides the pinning discipline), so every returned site is seedable as proposed.

    Every detected site is returned, ranked by descending estimated segment cost ([sr_cost], the
    accumulation's trip count) — gh-ocannl-541: the earlier [sr_red / sr_out] integer-division ratio
    zeroed every large-output site, and the in-detection cap then silently excluded (and, once
    gh-537 made more sites reachable, evicted) exactly the sites carrying the most serial work. The
    candidate-volume cap now lives in {!tune} ([max_split_reduce_sites] / config
    [autotune_split_reduce_max_sites]), which records evicted sites in the decline census.

    A candidate rejected {e only} because the accumulation cell's loops sit inside the reduction
    loop — how OCANNL lowers conv bias/weight gradients, where nothing else in the schedule space
    reaches the dominant segment — is re-probed after the enabling loop interchange
    ({!Ir.Schedule.split_reduce_hoist} names the loops, each [Swap] confirmed [Op_legal] on the code
    it acts on); the chain is recorded in [sr_swaps] and replayed by the candidate's prelude.
    [static_indices] only reaches the interchange probe's [Sched.apply]. Exposed for tests. *)

val share_cap : cap:int -> (string * 'a list) list -> 'a list * (string * int) list
(** gh-ocannl-685: spend a cap round-robin across named categories instead of as a prefix over their
    concatenation, returning the survivors (in the original category order, so an under-cap input
    comes back unchanged) and the per-category drop counts. One proposal per category per round, a
    category that runs out stops taking turns: every non-empty category is represented before any
    gets a second, and unused share spills to the others without imposing a ranking. Used for
    {!menu}'s per-unit action cap, whose list is a category-ordered concatenation and NOT ranked --
    a plain prefix there starved every category after the first outright. Exposed for tests. *)

val menu :
  ?admits:(Ir.Schedule_cache.saved_optop -> bool) ->
  is_cpu:bool ->
  is_gpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  registry:Ir.Schedule_cache.registry ->
  Ir.Low_level.optimized ->
  Ir.Schedule_cache.saved_optop list
(** The beam-round action menu over one unit's transformed code — [registry] must resolve its loop
    binders, including symbols minted by the schedule applied so far (the search harness builds it
    with {!Ir.Schedule_cache.to_saved} over that schedule): dividing serial [Split]s, [Swap]s of
    perfect serial pairs, [Unroll]s, [Retype]-[Vectorized] on innermost loops, and [Tensorize] role
    permutations, each proposal vetted by {!Ir.Schedule.op_legality} (proven-illegal ones are pruned
    before they cost a candidate compile; [Op_unknown] proceeds to compile-and-time). The loop
    enumeration descends into accumulation-minted [Local_scope] bodies and treats binder-sharing
    mint copies as one decision (gh-ocannl-666). It descends virtualization's inlined computations
    too, but a loop reached through one draws the [Retype]-[Vectorized] proposal ALONE
    ([Ir.Low_level.scope_mint], gh-ocannl-687): that is the one category with a renderer built for
    the shape (an inlined reduction's [Set_local] accumulation is what
    [C_syntax.try_vectorize_reduce] recognizes, and the enclosing loop -- whose body holds a
    [Local_scope] -- cannot be explicitly vectorized at all), while [Split]s, [Swap]s and [Unroll]s
    there are up to eight descriptors per loop that nothing proposed before gh-666 and that displace
    proposals for the main nest.

    The per-unit action cap is shared across the categories by {!share_cap} rather than spent in
    category order (gh-ocannl-685). [admits] filters proposals BEFORE that cap, so the budget is
    shared over moves the caller can use rather than over moves it is about to discard -- the beam
    passes its GPU dispatchability rule here, since an incumbent binding no hardware dimension can
    only be expanded through a move that binds one. It may record its refusals. Exposed for tests.
*)

type decline_summary = {
  key : Ir.Schedule_outcome.rejection_key;
  count : int;
  sample_details : string list;
}
(** Aggregate of candidate declines sharing one stable key. Details retain at most the first three
    distinct diagnostics and are never part of the key. *)

type terminal_failure = {
  phase : Ir.Schedule_outcome.phase;
  candidate : string option;
  detail : string;
}

(** What the call did about searching (gh-ocannl-677). The states are mutually exclusive and each
    carries exactly its own data, so a consumer matches instead of re-deriving: "it searched" is
    [Searched | Search_died _] and nothing else — in particular it is {e not} [not cache_hit], the
    reading that costs a benchmark harness every tuned cell under the reproducible profile, where a
    call reports having neither searched nor replayed. Spelled as four independent booleans until
    gh-ocannl-677, where two consumers mis-derived the state in one PR.

    Note the arithmetic: five constructors for what reads as four outcomes, because "nothing
    searched" is two — a deliberate no-search that ships the untuned default and returns, and a
    failure before the search that reports and then raises. The old encoding could not tell them
    apart either, it just did not say so. *)
type outcome =
  | Searched
      (** A search ran and completed: candidates were proposed and compiled or timed, leaving the
          process loaded with their modules and buffers. *)
  | Search_died of terminal_failure
      (** A search ran and terminated on the carried fatal failure. The counters hold the work it
          had reached, and [best_ms] is a mid-search measurement of the {e search} context: no
          routine was compiled from the caller's context for it, so it is never shippable
          (gh-ocannl-550). *)
  | Cache_replay
      (** A cached winner replayed; no search ran in this process. The census is then empty except
          for a declined baseline: the base compile precedes the lookup, so its rejection is real
          information about this process on this device even though nothing was searched. *)
  | Search_disabled
      (** Nothing was searched and there was nothing to replay: config [autotune_search=false] (the
          reproducible profile, gh-ocannl-559) with no chosen cache, or a chosen cache that missed.
          The caller gets the untuned default compile — the same code {!Context.compile} would have
          produced. Every counter is zero and every time [infinity]; a declined baseline can still
          populate [declines]. *)
  | Pre_search_failure of terminal_failure
      (** The call failed before (or instead of) the search proper — a base compile that failed
          before its lowering was captured, a fatal baseline link, a fatal cache replay, a baseline
          timing failure, or an untuned fallback compile of a search-less call — and raised. It
          still reports (gh-ocannl-550), carrying whatever census the call had reached, so a caller
          attributing arms by arrival order (the positional [?report] of {!Train.tune_placements})
          gets a slot for it. *)

type timing_mode =
  | Isolated
      (** One launch followed by a host synchronization: the number is the latency of a lone
          dispatch, kernel plus one submit/sync round trip. What every schedule crowned before
          gh-ocannl-755 was ranked by, and the right objective for a workload that really does
          dispatch one kernel and wait for it. *)
  | Queued
      (** A calibrated number of launches dispatched back to back with one synchronization, divided
          by the count: the round trip is amortized and the number is what the kernel sustains
          inside a stream that already has work in it. The default, because that is what a training
          step presents — it queues every kernel of a layer and synchronizes at the end, so no
          kernel in it pays a round trip of its own. *)

type timing_sample = {
  per_launch_ms : float;  (** The quantity the sampling budget accumulates and the minimum ranks. *)
  contention_ms : float;
      (** Raw wall time used only for contention detection. Under {!Queued} this is the whole batch
          before division by its depth, so a fixed host stall is not divided out of the signal. *)
}

type timing_result = {
  ms : float;  (** The minimum per-launch sample. Callers admit it through {!admitted_timing_ms}. *)
  contended : bool;
      (** At least half the sample window was more than 2x slower than its minimum. This is a
          refusal signal: the window mostly measured host stalls, so the autotuner does not rank or
          cache the number (gh-ocannl-855). Dispersion only — a window whose minimum is non-positive
          or non-finite is a clock that resolved nothing, a separate fact that {!admitted_timing_ms}
          refuses on the number itself (gh-ocannl-888).

          Because they are separate, [not contended] is HALF a proof of usability: a consumer
          deciding whether to keep a number asks {!admitted_timing_ms}, never this field. Reading it
          directly is for saying something about contention itself — a diagnostic, a test that
          excuses a claim under host load. *)
  samples : int;
      (** The number of samples behind [ms] and [contended], for diagnostics and exact dispatch
          accounting. *)
}

val admitted_timing_ms : timing_result -> float option
(** The shared admission gate for the consumers that RANK: candidate selection, the calibration
    rows, the roofline consistency check, cache attribution. Returns a positive finite [ms] only for
    an uncontended result; a refused or degenerate result is [None]. Deliberately NOT the gate for
    {!queued_batch_depth}, which chooses a scale rather than consuming a measurement
    (gh-ocannl-888). *)

(** What a candidate's timing is a measurement of (gh-ocannl-755), selected by config
    [autotune_timing].

    The two are different objectives and they do not crown the same candidate. On gfx1151 the
    submit/sync round trip is ~50-60 us while the fastest candidates at the gpt2_mini out-projection
    shape run in 60-70 us, so {!Isolated} reads about 2x the steady-state cost there — and the
    offset varies from candidate to candidate with the block count and the per-launch queue work
    (39-86 us over that site's ten seeded geometries, and up to 45 us of spread within a single
    run), which is what lets two candidates 5-8 us apart in steady state swap places — measured, in
    2 of 8 runs. Consequently a [best_ms] measured under {!Isolated} is not a throughput number and
    must not be compared with a batched per-kernel figure; under {!Queued} it is, up to the batch's
    residual ~1% of round trip. *)

type report = {
  outcome : outcome;
      (** What this call did about searching. The counters below say how much work that state got
          through; they never identify it — several are zero in more than one state. *)
  candidates_timed : int;
      (** Including the serial baseline where it was dispatched — on GPU backends it is not
          (gh-ocannl-532), and neither is any other candidate that binds no hardware dimension. So
          this count is not comparable across a CPU and a GPU backend: every serial-form candidate
          the CPU backends legitimately time is refused on GPU, and the refusals are counted in
          [declines] under [Not_dispatched_key] instead (gh-ocannl-543). *)
  timings_contended : int;
      (** Baseline and candidate timing windows refused because host contention dominated their
          samples or the clock result was non-positive/non-finite (gh-ocannl-855). Zero on cache
          replay and search-disabled calls, which time nothing in this process. Lets a completed
          search distinguish transient measurement refusal from structural candidate declines and
          retry when appropriate. Counts WINDOWS: a refused digest is dropped from the dedup set so
          an equivalent seed can retry it, so one candidate refused twice contributes two. Sound
          evidence for "was this search's measurement set complete?", and for nothing narrower — see
          [candidates_contended] and [default_refused] for the per-candidate facts. *)
  candidates_contended : int;
      (** Distinct candidate digests whose timing window was refused and which no later equivalent
          seed managed to time — the population [timings_contended] over-counts (Codex P2 on PR
          #608). This is the count that composes with [candidates_timed] and the
          [Not_dispatched_key] declines into "how many distinct candidates did this search reach",
          which a sum over refused windows cannot answer. *)
  default_refused : bool;
      (** The untuned-default seed's own digest had a refused timing window. Separates the two
          reasons [default_ms] can be [None] on a completed search: the load refused the reference's
          measurement, or the default seed was never proposed or never attributed — the
          gh-ocannl-552 regression, which a report-wide refusal count cannot distinguish from the
          first. False on cache replay and search-disabled calls. *)
  candidates_failed : int;
      (** Candidates rejected by op preconditions, hardware limits, or backend compilation — the
          serial baseline included (gh-ocannl-533) — plus detected seed sites declined before
          proposal (split-reduce sites evicted by [max_split_reduce_sites], gh-ocannl-541) and
          candidates refused as unparallelized on a GPU backend (gh-ocannl-532), which the decline
          census records so a previously-proposed site — or a candidate space the backend's
          execution model empties — never stops being proposed silently. *)
  baseline_declined : bool;
      (** The serial baseline's own compile was rejected with a typed cause and the search ran on
          the scheduled candidates alone (gh-ocannl-533): [baseline_ms] is then [infinity] and the
          rejection is in [declines] like any candidate's. The HIP scratch validator declining the
          unscheduled serial form of a large softmax/cross-entropy head at [Backend_link] is the
          case this exists for — before it was contained, that one rejection ended the search. *)
  declines : decline_summary list;
      (** Candidate rejections aggregated by stable cause key. Their counts sum to
          [candidates_failed]. Cache-entry replay failures are excluded. *)
  rounds_run : int;  (** Beam-expansion rounds actually executed (0 = seeds only). *)
  sketch_candidates : int;
      (** Whole-routine matmul-sketch instantiations seeded (0 when no matmul micro-kernel was
          detected or no tile sizes divide the extents), after the model pre-filter when one is
          active ([keep_fraction < 1]). Deterministic given the computation, backend, and
          configuration. *)
  epilogue_sketch_candidates : int;
      (** Of [sketch_candidates], the fused-epilogue twins (gh-ocannl-486): seeded when the site's
          output feeds an eligible elementwise tail ([Schedule.can_fuse_epilogue]) — each sketch is
          then proposed both unfused and with [Schedule.Fuse_epilogue] appended, so the tuner
          measures the one-kernel fused form against the fissioned two-kernel form. *)
  fiss_sketch_candidates : int;
      (** Per-fission-segment sketch candidates seeded (0 when the computation does not fission, or
          no segment contains a compatible matmul site). Includes the finer-segmentation
          ([arity_cuts], gh-ocannl-574) singles on GPU backends. Deterministic given the computation
          and backend. *)
  fiss_sketch_timed : int;
      (** Of the seeded per-fission-segment sketch candidates, those that compiled and were actually
          timed (not rejected by op preconditions or hardware limits, not deduplicated by digest).
          Includes completed timing windows refused as unusable; [timings_contended] identifies
          those verdicts. *)
  fiss_sketch_composite_eligible : bool;
      (** At least two coarse fission segments supplied usable single-sketch timings for
          recombination when the search reached that step. Refused windows do not supply a winner.
          False without a search or if it died before recombination. *)
  fiss_sketch_composite_timed : bool;
      (** The coarse recombination candidate reached a timing window, including a refused window.
          Does not count the separate finer-fission composite. *)
  split_reduce_candidates : int;
      (** Split-reduce seeds (gh-ocannl-484 task 3): one candidate per {!split_reduce_sites} site
          within the [max_split_reduce_sites] cap and eligible [num_blocks] value — the two-pass
          deterministic split reduction applied whole-routine before fission, each resulting segment
          getting the default preset. Deterministic given the computation and backend; [0] when no
          reduction-dominated site is detected. Sites the cap evicts appear in [declines] under
          [Seed_evicted_key "split_reduce"]. *)
  split_reduce_timed : int;
      (** Of the split-reduce candidates (the per-site singles and the recombined multi-site
          composite), those that compiled and were actually timed. Includes completed timing windows
          refused as unusable. *)
  split_reduce_composite_eligible : bool;
      (** Whether at least two split-reduce sites supplied usable best-timed singles, making the
          multi-site composite eligible for proposal. False on cache replay and search-disabled
          calls. *)
  split_reduce_composite_timed : bool;
      (** Whether the eligible multi-site split-reduce composite compiled and reached a timing
          window. A completed window refused as unusable still counts: the field pins candidate
          reachability, independently of whether its timing verdict was usable. False when the
          composite was ineligible, on cache replay, and on search-disabled calls. *)
  mma_candidates : int;
      (** Candidates whose label promises a tensorized ([Schedule.Tensorize]) pipeline that the
          search put through candidate compile: whole-routine and per-fission-segment sketch seeds,
          the cross-segment recombination composite, and beam-expansion candidates. Counted at the
          same point as [mma_timed], so the two always describe the same population. *)
  fiss_mma_candidates : int;
      (** Of [mma_candidates], candidates built from per-fission-segment MMA sketches (including
          their recombination composites). This is counted at candidate compile, not enumeration, so
          it detects a fission MMA family that disappeared or never reached compilation without
          borrowing evidence from a whole-routine candidate. *)
  mma_timed : int;
      (** Of [mma_candidates], those that compiled and were actually timed (dedup'd duplicates
          excluded — an identical candidate was already timed). Includes completed timing windows
          refused as unusable; [timings_contended] identifies those verdicts. [mma_candidates > 0]
          with [mma_timed = 0] means the search never reached a timing window for a tensorized
          pipeline at all, the state gh-ocannl-521 recorded for every GPU backend: candidates are
          cheap to enumerate and were being rejected in bulk at candidate compile. *)
  model_scored : int;
      (** Sketch candidates the analytic cost model scored during the seed pre-filter
          (gh-ocannl-491); [0] when the pre-filter is off ([keep_fraction >= 1]) or nothing was
          scoreable (e.g. no envelope constants). *)
  model_pruned : int;
      (** Of [model_scored], the candidates dropped before compilation and timing. Candidates
          without model coverage are never counted here — they are always kept. *)
  bound_pruned : int;
      (** Candidates the measured-incumbent bound pruning skipped before compile (gh-ocannl-514
          phase 4b, config [autotune_bound_pruning]): their schedule-invariant roofline floor met
          the best measured time so far — an admissible fathom, so no pruned candidate could have
          won. Counted apart from [model_pruned] (the keep-fraction pre-filter). *)
  fissioned : bool;
      (** The winning candidate compiles as multiple fissioned kernels; [false] when nothing was
          timed. *)
  baseline_ms : float;
      (** The unscheduled serial baseline's measured time, or [infinity] when it was not dispatched:
          on a GPU backend an unparallelized candidate is never run (gh-ocannl-532 — the whole
          routine in one work-item, unbounded in cost and uninterruptible), so it has no measurement
          and cannot win. Also [infinity] when [baseline_declined]. *)
  default_ms : float option;
      (** The untuned default pipeline's measured time (gh-ocannl-552): the [config_thresholds]
          fissioned-preset seed reproduces {!Ir.Schedule.maybe_default_schedules} exactly, so this
          is the schedule the user gets without tuning — the reference [baseline_ms] cannot provide
          on GPU backends, where it is [infinity]. Attributed by digest, so it is present even when
          the seed deduplicated against an identical earlier candidate (the timed serial baseline
          included, on CPU backends whose config thresholds leave the code unparallelized). On a
          completed search with [default_ms = Some d], [best_ms <= d] by construction — the seed is
          in the pool — and the margin between them is the value tuning added (the question
          gh-ocannl-491 asks). The attribution honors the scheduling gates: with automatic
          scheduling inactive ({!Ir.Schedule.automatic_schedule_active}) the untuned default is the
          unscheduled serial form and this field reports the baseline's measurement (so [None] on
          GPU, where that form is never dispatched); with config [schedule_fission=false] no
          candidate reproduces the whole-routine config-thresholds default and this is [None]. Also
          [None] when the seed failed to compile, was refused as unparallelized on GPU
          (gh-ocannl-532), or on a cache hit whose entry predates this field or was written under a
          config that shaped a different default pipeline
          ({!Ir.Schedule.default_schedule_fingerprint} mismatch). *)
  best_ms : float;
      (** The winner's measured time, or [infinity] when nothing was timed at all — every candidate
          failed and the baseline was not dispatched (or was declined). In that case no cache entry
          is stored and the returned routine is the untuned default compile, not the serial
          baseline.

          It is a reading of {!timing_mode}, and which one is not recorded anywhere in the report or
          the cache entry, so a number carried across processes is only comparable to another taken
          under the same setting. Under the default [Queued] it is a per-launch steady-state time
          and can be compared with a batched per-kernel figure (up to the batch's residual ~1% of
          round trip). Under [Isolated] it is a lone dispatch's latency, kernel plus one submit/sync
          round trip: on gfx1151 that is a 26-105% inflation over the same candidate's steady-state
          cost at the gpt2_mini projection shapes, varying per candidate, and the number must NOT be
          read as a throughput figure (gh-ocannl-755). The same caution applies to [baseline_ms],
          [default_ms] and [mma_best_ms], which are the same instrument's readings; ratios between
          them are safe, since all four were taken under one setting within one search. *)
  timing : timing_mode;
      (** The {!timing_mode} every time in this report was measured under (gh-ocannl-755), including
          the times a [Cache_replay] carries: the objective is a cache-key component, so an entry
          this call could look up was measured under this call's objective. Never absent: {!tune}
          resolves the objective before it constructs any report, and {!no_search_report} takes it
          as an argument for the same reason.

          It is here because nothing else records it. A consumer storing a [best_ms] in an artifact,
          or comparing one across processes, otherwise has to reconstruct the objective from ambient
          configuration — which a caller's explicit [?timing] need not match, and which a later
          reader of the artifact does not have at all. *)
  best_label : string;
      (** The crowned candidate's spec label — the same string the [autotune_log] lines carry (e.g.
          ["F_sketch[mma-gpu 16x32x32 ep]"]). ["baseline"] when no candidate beat the serial
          baseline, and [""] exactly when nothing was timed — including the states that time nothing
          by construction, which say so in [outcome] rather than through this string. Which
          candidate won is otherwise recoverable only by matching [best_ms] against the log's
          per-candidate times (gh-ocannl-546). *)
  best_tensorized : bool;
      (** The crowned schedule contains a [Schedule.Tensorize] — read off [best_schedule], so this
          is what the winner {e is}, not what its label promised. This is the fact a caller needs to
          state that a search's shipping artifact uses tensor cores; per-search [mma_timed] answers
          only whether one was measured.

          It says what the schedule {e asked for}. What the emission {e delivered} is
          [best_tensorization]; the two disagreeing is exactly the false "tensorized" timing. *)
  best_tensorization : Ir.C_syntax.tensorization option;
      (** How the crowned candidate's [Tile_mma] statements actually rendered (gh-ocannl-626), read
          off its compiled routine's {!Context.routine.mma} rather than re-derived by bracketing the
          census global: [Tensorized] when at least one tensor-core / SIMD-register-tile emission
          happened, [Scalar_fallback] when every emitted [Tile_mma] declined to the lane-0 scalar
          path, [Not_requested] when codegen emitted no [Tile_mma] at all.

          [None] exactly when there is no crowned candidate to consult (nothing was timed, or the
          call never searched) — a report that consulted no census must not read as tensorized, so
          the absence is a distinct value and not a default of [Not_requested].

          Against [best_tensorized] this closes the reporting contract of gh-ocannl-545: a schedule
          that asked and got scalar code is [best_tensorized = true] with [Scalar_fallback], one
          that asked and emitted nothing is [best_tensorized = true] with [Not_requested], and a
          genuinely tensorized artifact is [best_tensorized = true] with [Tensorized]. *)
  best_mma_statements : int;
      (** [Tile_mma] statements the crowned candidate emitted, and of those, how many rendered as
          the lane-0 scalar fallback ([best_mma_scalar_fallbacks]). The pair keeps the reporting
          contract's distinction (gh-ocannl-545): [best_tensorized] with
          [best_mma_scalar_fallbacks > 0] is a schedule that carries a [Tensorize] and executes
          scalar code, and [best_tensorized] with [best_mma_statements = 0] is one that emitted no
          tensorized statement at all. A genuinely tensorized artifact is [best_tensorized] with
          [best_mma_statements > 0] and [best_mma_scalar_fallbacks = 0]. *)
  best_mma_scalar_fallbacks : int;
  mma_best_ms : float;
      (** The best {e timed} tensorized candidate's time, [infinity] when none was timed
          (gh-ocannl-546). Against [best_ms] this is the margin by which tensorization won or lost
          this search, which is the difference between "the tensorized pipeline is uncompetitive
          here" and "it lost inside measurement noise". Where this field carries a measurement at
          all — [Float.is_finite mma_best_ms], the only state in which comparing it to [best_ms] is
          meaningful — [best_tensorized] implies [mma_best_ms <= best_ms], the winner being a member
          of the population it minimizes over. An infinite [mma_best_ms] is the absence of a
          measurement and stands under any [best_tensorized]: a replay of an entry written before
          the field existed (see the cache paragraph below) reports a tensorized winner with a
          finite [best_ms] and no family time at all, and asserting the implication unconditionally
          would call that legacy replay inconsistent.

          It does {e not} imply equality, and neither does [not best_tensorized] imply
          [mma_best_ms >= best_ms] (gh-ocannl-716). A beam round is accepted only when it improves
          on the incumbent by at least [min_progress] (1%); a round that improves by less is
          rejected with the incumbent still crowned, yet its candidates were timed, so a tensorized
          one among them can have lowered [mma_best_ms] below [best_ms] by up to that band. Callers
          reading the margin as a profitability ratio ({!family_profit_of_report}) are unaffected —
          a ratio just under 1.0 means the family tied the winner inside the beam's own tolerance —
          but a caller asserting the strict inequality is asserting a coin toss, which is how it
          reached a CI failure.

          Its population is {e structural} — timed candidates whose schedule contains a
          [Schedule.Tensorize] — and therefore differs from [mma_timed]'s label-promised one, in
          both directions. A beam move can append a [Tensorize] to a saved or preset incumbent,
          producing a candidate that is tensorized while its label promises nothing (so it counts
          here and not in [mma_timed]); conversely a labeled candidate whose applied schedule
          carries no [Tensorize] counts in [mma_timed] and not here. Same choice as
          [best_tensorized], for the same reason: what shipped is a property of the schedule.

          On a cache hit it is the STORING search's measurement, exactly as [best_ms] and
          [baseline_ms] there are (gh-ocannl-579): this process timed nothing, but the counters are
          what describe this call, while the times describe the program — under a key regime that
          already makes those two replayable. Without that the flip chain's profitability term would
          rank the decision surface one way on the cold run that measured the family and the other
          way on every warm-cache run after it. [infinity] when the storing search timed no
          tensorized candidate, and for entries written before the field existed. *)
  best_schedule : Ir.Schedule_cache.saved_schedule;
      (** The winner's schedule; for a fissioned winner, the concatenation of the per-segment
          schedules (informational). Empty when nothing was timed. *)
}

val no_search_report : timing:timing_mode -> report
(** The report of a {!tune} call that never searched (config [autotune_search=false], gh-ocannl-559,
    and no cache entry to replay): [outcome = Search_disabled], every counter zero, every time
    [infinity], [best_label] empty and [best_tensorization = None]. The caller gets the untuned
    default compile. Also the base the pre-search failure reports are built on, with [outcome]
    replaced and whatever census the call had reached filled in.

    [timing] is the objective the call resolved, which is all that distinguishes one of these from
    another: it names what the (absent) times {e would} have been measured under, and keeps
    {!report.timing} a plain field rather than an option every consumer of a real report would have
    to unwrap. A caller synthesizing a report — a test building a search outcome to feed
    {!family_profit_of_reports}, say — passes whichever objective its scenario is about. *)

val outcome_name : outcome -> string
(** The stable one-word name of an outcome state — ["searched"], ["search-died"], ["cache-replay"],
    ["search-disabled"], ["pre-search-failure"] — for logs, JSON records and test goldens. *)

val terminal_failure : report -> terminal_failure option
(** The fatal failure that ended the call, from whichever of the two failing states carried it;
    [None] otherwise. A projection over {!outcome}, not a re-derivation of it: "did this call fail"
    spans two states, and every caller that ranks or attributes arms asks exactly that — an arm
    carrying one is {e never} the shipped arm, whatever its pre-failure [best_ms] says
    (gh-ocannl-550). The failure's [phase] is the one it carries — where the call actually died (at
    link, at launch, at sync), not where the report was assembled.

    {!tune} reports exactly once per call, on every path that does any work. Argument validation is
    the exception, and deliberately so: an incompatible [timing_ctx] is a precondition violation
    detected before anything happens, not an outcome of a search, and reporting it would attribute a
    phase to a call that never reached one. *)

val model_score :
  static_indices:Ir.Indexing.static_symbol list ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  Ir.Schedule.schedule ->
  float option
(** The analytic cost model's ranking score of a candidate schedule (gh-ocannl-491, the selection
    half): {!Ir.Schedule.apply} on a hermetic copy, {!Ir.Cost_model.analyze}, then the roofline
    lower-bound seconds under the envelope constants — [limits]' advisory [peak_flops] /
    [peak_memory_bandwidth], each overridable by config [model_peak_flops] /
    [model_peak_memory_bandwidth] (calibrated per-machine values beat the class constants). [None] —
    no model coverage — when the schedule fails to apply, the code is opaque to the extraction (its
    counts may under-estimate, so ranking on them could prune the true winner), or no envelope
    constant is present. A ranking score, not a runtime prediction. Exposed for tests. *)

val model_prefilter : keep_fraction:float -> ('a * float option) list -> ('a * float option) list
(** The order-preserving pre-filter over model-scored candidates: keeps every unscored ([None])
    candidate — the no-coverage exemption: never dropped, only measured — plus the best
    [ceil (keep_fraction * n)] of the [n] scored ones (at least one; ties at the cutoff are all
    kept, so the outcome is independent of enumeration order). The identity when
    [keep_fraction >= 1]. Exposed for tests. *)

val placement_enablement :
  limits:Ir.Backend_intf.hardware_limits ->
  static_indices:Ir.Indexing.static_symbol list ->
  base:Ir.Low_level.optimized ->
  allmat:Ir.Low_level.optimized ->
  Set.M(Ir.Tnode).t * Set.M(Ir.Tnode).t
(** The enablement prior of the placement levels (gh-ocannl-514, the gh-ocannl-558 lesson): given
    the default-policy lowering and its all-materialized specialization, classify the flip
    candidates by whether their placement decides a sketch family's {e expressibility} — computable
    from the seeders' own site classification, before any compile. Returns
    [(enablement, disablement)]:

    - [enablement]: operand/destination nodes of an mma-eligible matmul site (the backend advertises
      a format tile and the wide-f16 policy's required scope for the site's unstaged form,
      whole-routine or per-fission-segment — the granularity the seeders detect at) that exists in
      the all-materialized lowering but has no eligible counterpart under default placements.
      Materializing such a node is what makes the tensorized family expressible — the flip changes
      the feasible set, not just the objective, which the per-node recompute-cost bound has no term
      for (on gh-558's [mlp_wide]/hip/bf16, cost ranking buried the family-unlocking cast twins
      below four no-op [`Inline] flips and a budget-5 chain found nothing).
    - [disablement]: operand/destination nodes of sites already eligible under default placements.
      An [`Inline] flip of a node in {e either} set can only move away from an eligible site —
      {!rank_flip_candidates} demotes those.

    Empty on backends without an [mma] capability. A classified fission failure degrades to
    whole-routine detection; the classification is a ranking prior, never a legality fact. *)

type family_profit =
  | Unmeasured
      (** No completed search timed a tensorized candidate ([mma_best_ms] infinite), so nothing is
          known about the family's speed here — including the gh-ocannl-521 state (many seeded, none
          surviving candidate compile), which is a fact about candidate compilation, not about the
          family. Read from [mma_best_ms] rather than [mma_timed]: the latter counts LABEL-promised
          candidates, and a beam-appended [Tensorize] on a saved incumbent measures the family
          without promising it in a label. *)
  | Pays of float
      (** The best timed tensorized candidate ([mma_best_ms], structural rather than label-keyed)
          was within the profit margin of the search's best, at this ratio of it ([<= 1.] when the
          family won outright). *)
  | Loses of float
      (** It lost by more than the margin, at this ratio of the search's best. On gh-514's metal/f16
          [mlp_wide] cell the ratio is ~12 (mma_best 79-92 ms against 7.5 ms). *)

val flip_profit_margin_of_string : string -> float
(** Parse config [tune_flip_profit_margin] (gh-ocannl-579): a ratio of at least 1.0. Anything else —
    unparseable, non-finite, or below 1.0, which would demote a family that WON — raises
    {!Utils.User_error} rather than falling back to the default, since a run that asked for a
    profitability policy it also made impossible should not quietly get a different one. Exposed for
    tests. *)

val family_profit_of_report : ?margin:float -> report -> family_profit

val family_profit_of_reports : ?margin:float -> report list -> family_profit
(** What completed searches measured about the tensorized family's profitability on this device
    (gh-ocannl-579): [mma_best_ms] against [best_ms], compared to config [tune_flip_profit_margin].
    A report with any [timings_contended] contributes [Unmeasured]: its admitted minima cover an
    incomplete candidate set and cannot safely price the family. Over several reports the most
    favourable evidence wins — the expressibility prior is deleted only by evidence that contradicts
    it, never by the absence of a confirmation. A failing arm's report counts: its timings are
    measurements of the family even though its [best_ms] is not shippable. Exposed for tests and for
    [Train.tune_placements], which derives it from the placement A/B's two arm reports. *)

val family_profit_summary : family_profit -> string
(** A log-line phrase naming the evidence and its ratio. *)

val effective_flip_ordering :
  ordering:[ `Cost | `Enablement | `Profitable ] -> profit:family_profit -> [ `Cost | `Enablement ]
(** The ordering [`Profitable] resolves to under the given evidence: [`Enablement] when the family
    is unmeasured or competitive, [`Cost] when it is measured to lose here. Both of the prior's
    classes go at once, because the promotion of family-unlocking flips and the demotion of
    family-breaking ones are the same bet on the same family. Exposed for tests. *)

val rank_flip_candidates :
  ordering:[ `Cost | `Enablement | `Profitable ] ->
  ?profit:family_profit ->
  enablement:Set.M(Ir.Tnode).t ->
  disablement:Set.M(Ir.Tnode).t ->
  Ir.Low_level.flip_candidate list ->
  Ir.Low_level.flip_candidate list
(** Deduplicate (by [Tn.uid], keep-first) and rank the decision surface. [`Cost] is the legacy
    recompute-cost-descending order (the gh-555 chain's, kept as the evaluation baseline);
    [`Enablement] sorts family-unlocking [`Materialize] flips ([enablement] members) first and
    family-breaking [`Inline] flips (members of either set) last, cost-descending within each class;
    [`Profitable] (gh-ocannl-579) is [`Enablement] weighed against [profit] per
    {!effective_flip_ordering} — the prior models expressibility, and on a device where the family
    it unlocks is measured hopeless, promotion is pure opportunity cost that displaces the winning
    flip out of a small budget. [profit] defaults to [Unmeasured] (so a caller with no measurements,
    such as {!model_default}, gets the prior). Config [tune_flip_ordering] selects the default
    ordering. Exposed for tests. *)

type placement_surface = {
  ps_candidates : Ir.Low_level.flip_candidate list;
      (** Deduplicated, ranked per {!rank_flip_candidates} under config [tune_flip_ordering]. *)
  ps_ordering : [ `Cost | `Enablement ];
      (** The ordering [ps_candidates] actually came out in — with [tune_flip_ordering=profitable]
          (the default) this is where the measured evidence landed, so a log line or a test can say
          which prior ranked the surface rather than which one was configured. *)
  ps_profit : family_profit option;
      (** The evidence that decided [ps_ordering], and [None] when the configured ordering is
          unconditional ([cost] or [enablement]) and no evidence was consulted — which is also why
          such a run never reads [tune_flip_profit_margin] and cannot fail on a malformed one. *)
  ps_enablement : Set.M(Ir.Tnode).t;
  ps_disablement : Set.M(Ir.Tnode).t;
  ps_floor_ms : materialized:Ir.Tnode.t list -> float option;
      (** The roofline floor (ms) of a partial placement vector: every completion in which
          [materialized] holds costs at least this much — {!Ir.Cost_model.completion_floor} on the
          all-materialized specialization with the other candidates' placements open, under the same
          envelope constants as {!model_score}. Monotone in [materialized] (commitments only narrow
          [open_placement]), so it is a sound branch-and-bound fathom: in the tuned regime, a flip
          whose floor meets the best {e measured} time cannot win and is skipped without spending
          budget (the admissible direction — the bound already exceeds the incumbent's measurement).
          [None] when no envelope constant is present. *)
}
(** The placement decision surface prepared for search (gh-ocannl-514): the per-node
    inline/materialize levels of the joint placement x sketch x fission space. *)

val placement_surface :
  ?name:string ->
  ?ordering:[ `Cost | `Enablement | `Profitable ] ->
  ?evidence:report list ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  placement_surface
(** Read and rank the decision surface from [ctx] (analyze-only — two hermetic lowerings via
    {!Context.lowered_for_decisions}, sharing the gh-560 analysis cache; no backend codegen, no
    effect on [ctx]). [name] names the computation for those lowerings exactly as
    {!Context.compile}'s does, and is what makes this work for a comp carrying no
    {!Ir.Assignments.Block_comment} (gh-ocannl-669). [ordering] defaults from config
    [tune_flip_ordering] ([profitable]). [evidence] is the completed searches [`Profitable] weighs
    the enablement prior against ({!family_profit_of_reports}, reported as [ps_profit]):
    [Train.tune_placements]' flip refinement passes the placement A/B arms' reports, while
    {!model_default}'s placement search (config [model_default_placements]) passes none — it
    measures nothing, so the prior stands there. Under an unconditional ordering the evidence is not
    derived at all, so such a run never reads [tune_flip_profit_margin]. *)

type model_choice = {
  mc_label : string;
      (** ["default"], or the winning candidate's spec label (matching the [autotune_log] labels).
      *)
  mc_model_ms : float option;
      (** The winner's roofline lower bound in ms — a ranking score, not a runtime prediction;
          [None] when selection did not run. *)
  mc_scored : int;
      (** Model evaluations that produced a score (the default pipeline included; the fissioned flow
          also scores per segment). *)
  mc_skipped : int;  (** Model evaluations without coverage, excluded from the ranking. *)
  mc_rejected : int;
      (** Candidates excluded from the ranking because their scheduled form fails
          {!Ir.Low_level.validate_parallel}: the model cannot see that a schedule will not compile,
          and rates the tensorized families best, so on a backend where those are unbuildable it
          used to crown one and then degrade to the default (gh-ocannl-522). *)
}

val model_default_enabled : bool Lazy.t
(** Config [model_default_schedule]: recipe-level untuned compiles ({!Train.to_routine},
    {!Train.run_once}, the benchmark runners) route through {!model_default} instead of
    {!Context.compile}. *)

val compile_advisory :
  ?name:string ->
  ?on_fallback:(exn -> unit) ->
  ?fallback_if:(unit -> bool) ->
  (Ir.Low_level.optimized -> Ir.Low_level.optimized list) ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Context.t * Context.routine
(** {!Context.compile_outcome} with advisory provenance and the given [lowered_transform], falling
    back to a plain {!Context.compile} — the ordinary default pipeline — for a classified compiler
    rejection, including validation in backend codegen. Fatal failures propagate without retrying.
    [on_fallback] is called with the public rendering of the cause when fallback fires.
    [fallback_if] (default: always) is consulted first, for transforms that may themselves have
    degraded to the default pipeline — [false] re-raises the original exception, backtrace included,
    instead of duplicating a compile that has nothing to fall back to. For advisory transforms only:
    a failure of the default pipeline itself propagates. [name] names the routine, exactly as
    {!Context.compile}'s does, and reaches the fallback compile too. See {!model_default}
    (gh-ocannl-519). *)

val model_default :
  ?name:string ->
  ?report:(model_choice -> unit) ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Context.t * Context.routine
(** A drop-in for {!Context.compile} that raises the untuned floor (gh-ocannl-491 task 3): inside
    the compile's own transform seam, the untuned default pipeline and the sketch families
    (whole-routine, and per-fission-segment substitutions when the default fissions) are scored with
    the roofline model, and the model-argmin schedule is applied — zero measurement, one backend
    compile. Advisory by construction: a candidate without model coverage is never picked over the
    default, ties go to the default, and missing envelope constants, a disabled default annotator
    ({!Ir.Schedule.automatic_schedule_active}), or any classified scoring, application, backend
    validation, or compilation failure fall back to the ordinary default pipeline — the reported
    {!model_choice} then says ["default"]. Fatal failures propagate without retrying. Once the
    compile is on that pipeline there is nothing left to fall back to, so its failures propagate as
    they would from {!Context.compile}, without a duplicate attempt. Unlike {!tune}, nothing is
    executed and no cache is involved — results depend only on the computation, backend, and
    envelope constants.

    Being a drop-in for {!Context.compile} includes its [name] (gh-ocannl-669): it names the routine
    and every artifact of this compile — the model pick's, the fallback's, and the hermetic
    lowerings of the placement search alike — and, as there, a comp carrying no
    {!Ir.Assignments.Block_comment} requires it. *)

val set_test_bindings : Context.routine -> unit
(** Binds representative values for timing runs: ranged static indices at [range / 2], and gh-490
    symbolic extents at their upper bound [range] (the schedule-cache identity is
    extent-value-independent, so the single tuned entry is measured at the maximum). Unranged
    bindings are left at their current values. Exposed for tests and custom timing harnesses. *)

val queue_depth_cap_for_backend : string -> int
(** Queue-memory bound selected by canonical backend name: 2048 for CUDA/HIP, and the historical 200
    for cc/Metal. Exposed with the neighboring pure calibration seams so the backend scoping of
    gh-ocannl-892 is pinned without running a full search. *)

val queued_batch_depth : timing_result -> int
(** How many launches {!Queued} puts in one batch, given an estimate of what one launch plus one
    synchronization costs. Aims at ~10 ms of wall time per batch, capped at 2048 launches and
    floored at 1 — so a routine at or above the target is batched at depth 1 and measured exactly as
    {!Isolated} measures it, and a microsecond routine cannot mint an unbounded batch. A
    non-positive or NaN estimate is a clock that resolved nothing rather than a zero-cost kernel and
    batches at the cap; an infinite one floors at 1, the far end of the scale the policy is for.
    Total, over every float and every verdict: a positive subnormal estimate saturates rather than
    raising. Exposed because those two boundaries are what a regression would cross silently: a
    depth stuck at 1 turns a queued search back into an isolated one.

    [contended] is deliberately unread (gh-ocannl-888). The 2x-majority rule judges a ~10 ms batch,
    while this estimate is one dispatch plus one synchronization, whose dispersion on a GPU is the
    round trip's own tail; refusing on it starved every search on both GPU backends. A depth is not
    a measurement — an overestimate only shortens the batch, an underestimate is capped — and a
    deeper batch is the remedy for that dispersion, so the refusal belongs downstream in the timed
    loop, where the window judged IS a batch. *)

val refine_queued_batch_depth : single_ms:float -> probe_depth:int -> probe_ms:float -> int * float
(** Refines a provisional queued depth from the synchronized single-launch minimum and the wall
    minimum of a batch at [probe_depth]. Returns the final depth and its estimated whole-batch wall.
    The two observations separate fixed synchronization cost from marginal launch cost, so even a
    shallow provisional batch reaches the ~10 ms contention scale rather than counting a fraction of
    the fixed cost against every launch. A probe exactly at the target keeps its depth; one that
    overshoots is interpolated back inside the measured bracket. An unresolved pair requests a
    doubled retry and returns [nan] for the wall rather than jumping to the cap or labeling the
    shallow probe as a cap-wall estimate; repeated unresolved batch pairs eventually reach the
    2048-launch memory cap. Exposed as the deterministic policy seam for tests. *)

val refine_queued_batch_depth_between :
  base_depth:int -> base_ms:float -> probe_depth:int -> probe_ms:float -> int * float
(** The same affine refinement between two batch observations. An already-target-sized [base_ms] is
    retained only when the pair's inferred fixed component is below the target and no more than one
    quarter-target negative (the noise tolerance matching the 25% confirmation step); this is the
    depth-separated confirmation that keeps a shared fixed stall or a physically invalid fit from
    selecting a shallow final depth while preserving legitimate submit/sync overhead. Otherwise an
    unresolved pair requests a deeper retry and reports a [nan] wall. Exposed as the deterministic
    validation-policy seam for tests. *)

val sample_min : repeats:int -> sample:(unit -> timing_sample) -> timing_result
(** Pure sampling-policy seam used by calibration and the timed loop (gh-ocannl-855). Takes at least
    [max 16 repeats] samples; after that floor is met, tops up until their accumulated
    [per_launch_ms] reaches ~25 ms or 64 samples have been taken. Thus [repeats > 64] deliberately
    overrides the top-up limit. Reports [contended] when at least half the raw [contention_ms]
    samples exceed their minimum by 2x — dispersion only; a minimum that is non-positive or
    non-finite is refused by {!admitted_timing_ms} instead. Exposed so tests can inject a
    deterministic clock. *)

val search_measurements_cacheable : nothing_timed:bool -> timings_contended:int -> bool
(** Pure cache-policy seam (gh-ocannl-855). A search result is cacheable only when at least one
    candidate was timed and no timing window was refused as unusable. The current call may still
    ship its best usable candidate, but an incomplete measurement set must be retried by a later
    cache-cold search. *)

val timing_string : timing_mode -> string
(** The mode's canonical spelling ([isolated] / [queued]) — what a cache key's ["timing"] component
    carries, what a report's mode is rendered as, and the round trip of {!timing_of_setting}. *)

val timing_of_setting : string -> timing_mode
(** Parses the [autotune_timing] spelling ([isolated] / [queued], case- and space-insensitive);
    raises [Invalid_argument] on anything else. *)

val time_routine :
  ?tag_failures:bool ->
  timing:timing_mode ->
  repeats:int ->
  Context.t ->
  Context.routine ->
  timing_result
(** The tuner's own instrument, exposed so a harness can rank candidates by exactly what a search
    ranks them by (gh-ocannl-755) rather than by a re-derivation of it. Binds test values
    ({!set_test_bindings}) and restores the routine's bindings afterwards, runs one warmup, then
    minimizes the per-launch time over at least 16 and [repeats] timed runs, topping up until ~25 ms
    of per-launch samples has accumulated (up to 64 runs unless the [repeats] floor is larger) so
    that a host stall cannot collapse the min-of-N. Under [~timing:Queued] a "run" is a whole batch
    of dispatches whose depth is calibrated per candidate to ~10 ms of wall, capped at 2048 on
    CUDA/HIP and at the historical 200 on cc/Metal, and floored at 1. cc/Metal retain the historical
    synchronized-single estimate. On CUDA/HIP that estimate instead seeds a provisional depth; when
    that is above 1, a provisional queued probe separates fixed synchronization from marginal launch
    cost and selects the depth whose affine wall estimate reaches the target. The batch probes take
    the minimum of twelve runs: unlike ranked timing, they only choose scale and are already
    milliseconds long. Up to four further probes validate the depth. The first target-sized batch is
    confirmed at a 25% deeper depth (clamped to and measured at the cap) and retained only when the
    pair's inferred fixed component is below the target. A confirmation over 2x its supported base
    is retried once before an unresolved pair selects the cap. An unresolved single/probe pair
    retries at double depth, and the next affine fit uses the two batch observations so an inflated
    single window cannot force the cap. If the bounded loop first reaches the target on its last
    probe, the interpolated target depth is still sampled and checked against the measured
    overshoot. A non-monotone confirmation scales from the deeper measured batch, never the earlier
    suspect crossing. When a clean pair's fixed component already fills the target, its marginal
    slope selects a depth carrying ~10 ms of launch work instead of accepting a shallow stalled
    window or jumping to the cap. After four noisy but resolved underestimates, calibration keeps
    the latest affine projection rather than jumping to a 20--30 ms cap batch that would blunt the
    2x contention threshold. A CUDA/HIP routine slower than the target is confirmed by a depth-2
    probe, stays at depth 1, and is measured identically in both modes. The calibration always
    yields a depth; the result of the timed loop reports when most of ITS samples were stalled, and
    the tuner refuses such a candidate measurement rather than ranking and caching it
    (gh-ocannl-888). Since the budget is per-launch rather than batch wall, queued timing can spend
    up to [max 64 repeats] batches on a fast candidate; [max_timing_runs] bounds the top-up beyond
    the caller's requested floor.

    With [~tag_failures:true] the pre-dispatch validation, the launches and the synchronization are
    wrapped in their {!Ir.Schedule_outcome} phases, which is what lets a caller's
    {!Ir.Schedule_outcome.protect} attribute a failure to the phase it happened in; without it they
    propagate raw. Timing dispatches the routine repeatedly against live buffers, so an accumulating
    routine must be timed on a scratch lineage (see [tune]'s [?timing_ctx]) if its inputs matter
    afterwards. [Queued] raises how many such dispatches happen. For [repeats <= 64], the maxima are
    65 under [Isolated] and, under [Queued], 278593 on CUDA/HIP or 12865 on cc/Metal. The CUDA/HIP
    bound includes warmup, 64 single-launch calibration runs, at most seven twelve-sample
    calibration probes and 64 timed batches at the cap; cc/Metal have no batch probes. In general
    the queued bounds are [65 + 2048 * (84 + max 64 repeats)] on CUDA/HIP and
    [65 + 200 * max 64 repeats] on cc/Metal. Thus a routine whose values grow per run reaches larger
    ones. That is a fact about the scratch buffers, not about the measurement: the cap bounds each
    in-memory queue while the ~25 ms budget accumulates per-launch samples, and a candidate's time
    is not what it accumulated. *)

val on_batch_depth : (int -> calibration_samples:int -> unit) ref
(** Observation seam for the timing tests (gh-ocannl-851), called by each {!time_routine} call with
    the batch depth it settled on — after calibration, before the timed loop; {!Isolated} always
    reports 1. The negative control for a twice-divided queued reading needs the depth the call
    ACTUALLY used: the call recalibrates independently, so re-applying {!queued_batch_depth} to an
    estimate taken outside it guesses wrong exactly on the busy runners the control must survive.
    [calibration_samples] retains its historical label for source compatibility; its value is zero
    for {!Isolated} and the actual number of launches spent on single-launch, provisional-batch and
    depth-validation calibration for {!Queued}. The default is a no-op and no configuration selects
    it. *)

val on_candidate_attempt : (string -> unit) ref
(** Fault-injection seam for the containment tests (gh-ocannl-550), called with each candidate's
    label just before its compile — including the baseline's, which is a candidate (gh-ocannl-533);
    that one is called inside the base compile's transform, so a fault injected there is classified
    like any other and surfaces as the pre-search failure of a search that never started. The
    default is a no-op and no configuration selects it; raising from it terminates the search the
    way an uncontainable failure does — the [Search_died] report (carrying its failure) is emitted
    to [?report] and the exception propagates out of {!tune}, which is what [Train.tune_placements]
    must survive without losing the other arm's winner. Not a production seam: candidate failures
    that a backend {e can} attribute are contained without it (see [declines]). *)

val on_candidate_preflight : (string -> unit) ref
(** Fault-injection seam for the pre-dispatch containment tests (gh-ocannl-564), called with a
    routine's name inside the {!Ir.Schedule_outcome.Preflight} region of its timing run, just before
    {!Context.check_launch_bindings}. Raising from it classifies exactly as a real per-candidate
    validation failure does: for a candidate a contained decline under
    [Unclassified_key (Preflight, _)]; for the baseline the propagating pre-search failure a
    baseline timing failure always is.

    It exists because even the per-candidate trigger — an out-of-range static binding — needs a
    candidate whose static ranges differ from its siblings' to arise naturally, which the preset
    families do not supply. The lineage-wide triggers cannot be reached through it at all any more:
    {!Context.check_lineage_runnable} runs outside this region (gh-ocannl-569), so injecting one of
    their exceptions here exercises the containment machinery with a realistic payload but does not
    mirror where a real one is now raised. Default a no-op; no configuration selects it. *)

val on_candidate_timed : (string -> timed_so_far:int -> unit) ref
(** Observation seam for the containment tests (gh-ocannl-898), called with a routine's name each
    time a timing run's window yields an admitted measurement — the moment the search's
    [candidates_timed] accounting grows, for the dispatched baseline and candidates alike.
    [timed_so_far] is that accounting's own value after this admission — what a report cut at this
    instant would state as [candidates_timed] — passed so a precondition of the form "the arm under
    injection has timed N candidates" pins the tuner's number rather than maintaining a second
    counter that can drift from it. {!on_candidate_preflight} is upstream of the window's verdict
    and not a substitute: under the {!Queued} objective a preflighted run can still be refused (a
    contended window, a degenerate clock reading; gh-ocannl-855) and then times nothing, so counted
    in preflights such a precondition fires, on a loaded device, on an arm whose report says it
    timed nothing. Default a no-op; no configuration selects it. *)

val tune :
  ?name:string ->
  (* Names the computation, exactly as {!Context.compile}'s [name] names its single routine
     (gh-ocannl-669): every compile of this search -- each candidate, the baseline, the cache
     replay, the winner, the untuned fallbacks, the [autotune_log] control -- is named alike, and so
     is the [routine] column of the calibration rows this search emits. Required, as there, for a
     comp carrying no {!Ir.Assignments.Block_comment}; omitted, the name is derived per compile via
     {!Ir.Assignments.get_name_exn}. *)
  ?search:bool ->
  (* Whether to search at all; default from config [autotune_search] (true). With [false]
     (gh-ocannl-559: the [reproducible] profile) a committed cache entry still replays -- a pinned
     schedule is deterministic -- but nothing is timed, and a cache miss compiles the untuned
     default pipeline and reports {!no_search_report} under the resolved objective. Only a CHOSEN
     cache replays: [cache_dir] passed here, or [autotune_cache_dir] set at some config source. The
     built-in default counts as no cache, so a search-less run cannot silently pin itself to
     whatever an earlier local search left in ./autotune_cache. *)
  ?beam_width:int ->
  (* Default from config [autotune_beam_width] (2). *)
  ?rounds:int ->
  (* Maximum beam-expansion rounds beyond the seeds; default from config [autotune_rounds] (2). The
     search also stops when a round improves the incumbent by less than 1%. *)
  ?repeats:int ->
  (* Timed runs per candidate (after one warmup), min taken; default from config [autotune_repeats]
     (3). In [Queued] timing each such run is a whole batch of dispatches, so this is a floor on
     batches rather than on launches. *)
  ?timing:timing_mode ->
  (* What a candidate's time is a measurement of; default from config [autotune_timing] ([queued]).
     See {!timing_mode}. *)
  ?seed_block_sizes:int list ->
  (* Workgroup sizes swept through {!Ir.Schedule.default_gpu} as seed candidates on GPU backends
     (default [[64; 128; 256; 512]]), both whole-routine and per-fission-segment, in addition to the
     config-default preset and the serial baseline. *)
  ?cache_dir:string ->
  (* Directory of the schedule disk cache; [""] disables caching. Default from config
     [autotune_cache_dir] ([autotune_cache]). *)
  ?keep_fraction:float ->
  (* The model pre-filter of the sketch seeding (gh-ocannl-491): per candidate family (the
     whole-routine sketches; each fission segment's sketches), rank with {!model_score} and keep the
     best [keep_fraction] of the scored candidates before compiling or timing anything. Default from
     config [autotune_keep_fraction] (1 = pre-filter off). Candidates without model coverage are
     always kept — never dropped, only measured — so the pre-filter never overrides (or precludes) a
     measured result; presets, saved schedules and the baseline are never pruned. *)
  ?max_split_reduce_sites:int ->
  (* Candidate-volume cap on the split-reduce seed family: the top so-many {!split_reduce_sites}
     (ranked by estimated segment cost) are seeded; evicted sites are recorded in the decline census
     under [Seed_evicted_key "split_reduce"] (gh-ocannl-541). Default from config
     [autotune_split_reduce_max_sites] (8); [0] disables the family. *)
  ?timing_ctx:Context.t ->
  (* A scratch context lineage against which candidates are compiled and timed, so the timing runs
     never mutate [ctx]'s live buffers (parameters, accumulators — running a training step on
     scratch/zero data can even poison them with inf/NaN). It must contain the nodes the computation
     requires from a prior context, e.g. by repeating parameter initialization on a fresh root
     context, and must live on the same backend and device as the target context (raises
     [Invalid_argument] otherwise — candidates timed elsewhere do not predict this device). Only the
     winning schedule is then compiled from [ctx], exactly like a cache hit. Without it, the search
     shares [ctx]'s buffers and the caller should re-initialize mutated state afterwards. *)
  ?report:(report -> unit) ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Context.t * Context.routine
(** Like {!Context.compile}, but returns the empirically fastest of the searched schedule
    candidates. The returned context/routine come from an ordinary sibling compile of [ctx], so
    execution-dependency tracking behaves as if the winning compile were the only one. Raises like
    {!Context.run} would (e.g. uninitialized inputs) — tune in the same state you would run in.

    "Like {!Context.compile}" includes its [name] (gh-ocannl-669) — without which a comp that
    carries no {!Ir.Assignments.Block_comment} but that {!Context.compile} can name (as [Parallel]'s
    collective routines are named) could be compiled but not tuned. The schedule cache deliberately
    does not see it: a cache key is [Ir.Schedule_cache.cache_key] over the canonical lowering, the
    backend, the numerics and codegen tags, and the worker pool — and the name reaches only
    codegen's artifact and kernel naming, never the lowering. Two identical computations under
    different names therefore share one tuned entry, which is what the cache is for; naming a
    routine neither invalidates a crown nor mints a private one.

    Memory (gh-ocannl-550): every candidate this searches is {!Context.release}d as soon as it stops
    being a beam survivor, so the search's live {e working} pools and contexts are bounded by
    [beam_width] rather than by candidates attempted. The bound holds across contained failures and
    across dedups — a deduplicated candidate has still paid for a compile and a link — and the
    returned routine is the only artifact left when [tune] returns. Nothing about the returned value
    changes: it is the same live context and routine as before, from the same lineage.

    The bound is on working pools, and that qualifier is load-bearing: {!Context.release} cannot
    free per-device constants, and a hoisted [Stage] candidate mints a fresh packed constant per
    application. A search that seeds those (the CPU [hoist] sketches) therefore still grows one
    constant pool per such candidate — measured on [cc] at 1 -> 109 constant pools over 181
    candidates, against working pools held within 2-6. Bounding it needs an eviction rule inside the
    shared constant cache, which is gh-ocannl-565's subject; pinned as far as it can be by
    [test/operations/autotune_candidate_release]. *)

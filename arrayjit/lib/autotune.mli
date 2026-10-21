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
      counted in the same census. These refusals, and the search's dedup of a candidate whose code
      an earlier one already compiled to, are decided inside the lowered transform, before codegen
      (gh-ocannl-1340): a refused or duplicate candidate, the GPU baseline included, costs no
      codegen, device compile or link. Separately, the baseline's compile is protected like any
      candidate's (gh-ocannl-533) where it reaches codegen (on CPU backends): a typed rejection — a
      large softmax/cross-entropy head whose unscheduled form exceeds the HIP scratch budget is the
      case in hand — declines the baseline ([baseline_declined] in the report, and its own key in
      the census rather than a [Not_dispatched] refusal, which would assert a reason that is not the
      one) and the search proceeds on the scheduled candidates, which fission and placement
      promotion routinely bring back within budget. Only the base lowering itself is indispensable:
      a failure before it is captured has no search to run and propagates.
    - {b Fissioned candidates}: the kernel-fission pipeline ({!Ir.Schedule.fission_scheduled}) with
      per-segment schedules — the same preset sweep per segment, and beam rounds that extend
      {e one segment at a time}. [`Zeros] segments keep the default zero-expansion; [`Solo] segments
      stay unscheduled. A fissioned winner is cached with its whole segmentation — each segment's
      kind, length and pre-schedule canonical digest, and its schedule, zero expansions included
      ({!Ir.Schedule_cache.saved_segment}, gh-ocannl-1164) — and replays by cutting there
      ({!Ir.Schedule.fission_segmented}'s [replay]) rather than by re-segmenting under the current
      policy: a segmentation that no longer fits, or a segment whose digest differs from the saved
      one, fails the replay into a re-search. One seed uses the config-default thresholds,
      reproducing the untuned default pipeline exactly — so the winner is never worse than not
      tuning, even on launch-overhead-bound workloads where every aggressive preset loses to it; its
      measured time is surfaced as the report's [default_ms] reference (gh-ocannl-552). On CUDA and
      HIP, where the recipes' untuned compile ships the seeded plan ({!seeded_default}) rather than
      that pipeline, the plan is seeded too, as the very FIRST candidate ({!seeded_plan},
      gh-ocannl-1388; its time is [seeded_ms]): a search that stops after its first timing, or whose
      every later window is refused, crowns the step that ships untuned; a refused window of the
      plan's own is retried once after the seed pass. Under config [model_default_schedule=true]
      (the [performance] and [approximate] profiles) the recipes' untuned compile is
      {!model_default}'s pick instead, which no candidate reproduces (its placement pre-search
      changes the program), so there the floor is the seeded plan, not the step that ships untuned,
      until gh-ocannl-1253 starts the model's pick from this plan. Each preset is additionally
      seeded in a {e privatized} variant ({!extend_with_privatize}): per segment, every materialized
      read-modify-write accumulator is contracted into a per-thread register tile
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
      sketch pipelines are instantiated for each segment where a matmul site is detected, the
      remaining segments keeping the default preset. The candidates replay that segmentation and
      find their segments in it by position ({!seg_key}, gh-ocannl-1243): segments of one structure
      are sketched together only when their placements agree too. GPU segments can carry a covering
      per-cell zero companion, mapped with the site's geometry; [Privatize] forwards its seed when
      its scope owns the whole reduction. Whole-node zeros retained by the policy (including CPU
      zeros) remain separate, and the accumulator init-load observes them. [Tile_mma] loads its
      accumulator fragment before the reduction. On GPU backends the segmentation is additionally
      enumerated under {!Ir.Schedule.fission_scheduled}'s [arity_cuts] (finer) mode (gh-ocannl-574):
      a segment carrying a companion that cannot follow its site's full arity — the lm_head GEMM
      with its max-logits row reduction — has every seed of the shared segment decline on companion
      coverage, and the finer cut frees the site into its own kernel; segments whose key is new
      versus the coarse segmentation seed [fine]-flagged singles, one composite recombines the fine
      keys' best-timed singles (coarse-timed bests staff the segments the finer cuts leave
      unchanged); a fine winner's cached segmentation is the fine one, so its replay needs no record
      of the mode.
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
    (gh-ocannl-580). [Sketch_families]' interface bounds that include to the construction contracts
    consumed here; the family entry points below ({!sketch_params}, {!detect_conv},
    {!matmul_sketch_tree}, {!sketch_schedule}, {!sketch_path_traffic_floor}, …) are defined there,
    and {!sketch_seed_params} is the composition the search enumerates. *)

open Base

(** Which pair a coalesced seed merges; see {!Sketch_families.coalesce_layout}. *)
type coalesce_layout = Sketch_families.coalesce_layout = Coalesce_columns | Coalesce_contraction
[@@deriving sexp]

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
  sk_batch_inner : bool;
  sk_coalesce : coalesce_layout option;
  sk_swizzle : Ir.Low_level.swizzle_kind option;
  sk_depth : int;
  sk_pack_prec : Ir.Ops.prec option;
  sk_tile : Ir.Register_tile.t option;
}
[@@deriving sexp]
(** Parameters of one matmul-sketch seed candidate; see {!Sketch_families.sketch_params} for field
    documentation. Exposed for tests (the seeding pre-filter of gh-ocannl-479 and the mixed
    grid-outermost shape of gh-ocannl-473 are asserted on directly). *)

val digest_tag : string -> string
(** The short tag of a candidate's post-schedule digest (the [~digest] {!on_candidate_measured}
    receives; a fissioned candidate's is its segments' digests joined by ['+']) that the
    [autotune_log] lines print and that a beam-round candidate's label names its parent by
    ([F_saved[N segs from <tag>]+seg<i>:<move>], gh-ocannl-1230). *)

val sketch_params_label : sketch_params -> string
(** How a sketch's parameters print inside its candidate label ([W_sketch[...]], and each entry of
    [F_sketch[...]]) in the [autotune_log] lines and {!report.best_label}: every field renders, so
    two seeds that differ in any one field — a swizzled, pipelined or register-tile twin of one
    geometry — never share a label (gh-ocannl-1230). Contains no [','], the separator of a
    composite's entries. *)

type seg_key = { key_units : int list; key_digest : string }
(** Which segments of a recorded fission segmentation a per-segment sketch is for (gh-ocannl-1243):
    each one's first unit ({!Ir.Schedule.segmentation}'s units) and their common structural
    pre-schedule digest. The digest leaves placements out, so it cannot name a segment on its own:
    segments of one structure share a key only when their placement-aware digests agree too. *)

val fiss_sketch_label :
  ?staff:(seg_key * sketch_params) list -> fine:bool -> (seg_key * sketch_params) list -> string
(** The label of a per-segment sketch candidate ([F_sketch[...]]) over its [(segment key, params)]
    entries: each entry is {!sketch_params_label} followed by [" @"] and a tag of its segment key —
    a tag of the digest, then [":u"] and the keyed segments' first units joined by ["+u"] — so
    one-entry candidates of two segments that admit the same parameters print apart, structurally
    identical segments included. Entries are separated by [','] in key order; [fine] marks the finer
    segmentation. A staffed candidate (gh-ocannl-1138) appends [" | staff "] and its [?staff]
    entries rendered alike, so the part before it is the candidate's focus and reads as the bare
    single's whole label does. *)

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
  m_row_stride : int;
      (** [m_a]'s coefficient on [m_i]: [1] wherever a unit-stride row exists, greater on a strided
          row only a compacting [Stage] of [m_a] can tile (gh-ocannl-1148). *)
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

val detect_matmul_canonical :
  Ir.Low_level.optimized -> (matmul_site * Ir.Schedule.schedule * Ir.Low_level.optimized) option
(** The matmul site every seeding and schedule-construction entry point uses (gh-ocannl-1183): the
    site, the enabling-interchange prefix every schedule built from it carries, and the code it was
    detected on. The prefix is empty, and the code the argument, for a site {!detect_matmul} finds;
    otherwise it is an adjacent-[Swap] chain, each [Op_legal], sinking the contraction loops of the
    first accumulation-of-a-product statement below its write loops (the backward contractions
    backprop lowers with the contraction loops outermost or interleaved), and the code is the
    interchanged one. [None] when the plain matcher fails and {!detect_conv} succeeds: the conv
    family keeps its segments. Exposed for tests. *)

val reduction_sink :
  accum_prec:(Ir.Ops.prec -> Ir.Ops.prec) -> Ir.Low_level.optimized -> Ir.Schedule.schedule
(** The interchange {!seeded_default} gives a segment it does not seed (gh-ocannl-1348): for each
    perfectly nested serial accumulation whose reduction loops (those absent from the accumulator's
    index map) enclose a write loop, the adjacent-[Swap] chain of {!detect_matmul_canonical}'s
    contraction sink, taking the reduction loops below the write loops in their original relative
    order, so the default preset maps the write loops and code generation keeps the accumulator in a
    register instead of reading and writing device memory every step. Statement by statement, each
    chain kept only when every [Swap] is [Op_legal] after the chains kept before it. A statement
    qualifies only when every write loop owns an accumulator axis as a plain iterator (so the
    interchange keeps each cell's summation order: values are unchanged bitwise) and its storage
    precision is its own accumulator precision under [accum_prec]. Empty where nothing qualifies.
    Exposed for tests. *)

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
  c_d_syms : Ir.Indexing.symbol list;
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
    matcher runs alongside and any divergence raises. Exposed for tests.

    {b Boundary} (gh-ocannl-912): rank-generic — any number of batch axes and of spatial axes (1-D,
    2-D, 3-D) beyond the row become outer loops — but it refuses every SINGLETON axis, because
    lowering drops extent-1 loops and indexes the axis at a literal [Fixed_idx]: a batch of one or
    an output spatial extent of one (the output is no longer written at plain iterators), a single
    input channel or a k-by-1 window (the kernel is no longer read at plain iterators; with one
    input channel no reduction-channel loop remains for [Tensorize (row, oc, ic)]). A channel row of
    several axes is refused too (exactly one reduction channel). An all-singleton window (1x1, any
    stride) is a plain GEMM: {!detect_matmul} owns it. Refused sites get no matmul or conv sketch
    seeds (the preset and split-reduce candidates are unaffected);
    [test/operations/conv_detection_boundary] is the witness. *)

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

val mma_format_triples :
  a_prec:Ir.Ops.prec ->
  b_prec:Ir.Ops.prec ->
  d_prec:Ir.Ops.prec ->
  (Ir.Backend_intf.mma_input_format
  * Ir.Backend_intf.mma_input_format
  * Ir.Backend_intf.mma_input_format)
  list
(** The (operand, operand, accumulator) format triples a site with these storage precisions resolves
    to, in preference order — the key the typed seeds look up in [mma_format_tiles] and
    [mma_staged_layouts]. Single-precision operands read as tf32 first only under
    {!Ir.Numerics.t.tf32_matmuls}. Exposed for tests, which derive the storage precisions of an
    advertised triple from it rather than restating the format table. *)

val mma_tile_for_precisions_in_scope :
  Ir.Backend_intf.mma_capability ->
  scope:Ir.Backend_intf.mma_emission_scope ->
  a_prec:Ir.Ops.prec ->
  b_prec:Ir.Ops.prec ->
  d_prec:Ir.Ops.prec ->
  (int * int * int) option
(** The advertised intrinsic tile a site with these storage precisions seeds in this emission scope
    under the current policy, [None] where the descriptor withholds it (no advertised format triple,
    or a wide-accumulator lifetime the scope lists exclude). Exposed for tests: it is the
    descriptor's verdict as seeding reads it, which [cuda_mma_cutoffs] compares against the arm
    resolver's floors at every cutoff (gh-ocannl-1214). *)

val sketch_seed_params :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
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
    ([limits.simd_vector]), and transposed-B storage for shapes that read B in place. Builder
    preconditions the schedule builders settle identically for every tile completion — the
    zero-expansion row-axis rule, and the GPU sketches' companion-coverage rule per fusion flavor —
    also pre-filter here, as the family tree's construction-time refutations (gh-ocannl-577): a seed
    list never proposes a candidate that statically must fail its build. For the matmul family this
    list {e is} {!Ir.Schedule_space.leaves} of {!sketch_family_tree} — the matmul family's tree
    where a matmul site is detected, else the convolution family's (gh-ocannl-697) — epilogue twins
    included. [capabilities] is the compiling context's [Context.codegen_capabilities]: the
    workgroup-memory refutations charge the staged tiles by it as the gate does
    ([Schedule.kernel_workgroup_bytes]: allocation rounding, tile-MMA scratch, gh-ocannl-1255); the
    default charges exact tile sizes, a weaker floor. Exposed for tests. *)

val coalesce_prefix :
  layout:coalesce_layout -> Ir.Low_level.optimized -> Ir.Schedule.schedule option
(** See {!Sketch_families.coalesce_prefix} (gh-ocannl-1165, gh-ocannl-1279). Exposed for tests. *)

val coalesced_seed_params :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  sketch_params list
(** The coalesced tuner branch (gh-ocannl-1165, gh-ocannl-1279), which {!tune} seeds beside
    {!sketch_seed_params}'s candidates, whole-routine and per fission segment: see
    {!Sketch_families.coalesced_seed_params}. Its winners persist as ordinary saved schedules whose
    leading {!Ir.Schedule.constructor-Coalesce} ops name their loops structurally. *)

val tensorized_capability_refutation :
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  a_prec:Ir.Ops.prec ->
  b_prec:Ir.Ops.prec ->
  d_prec:Ir.Ops.prec ->
  string option
(** Why the backend and the configuration withhold the tensorized family from a matmul site with
    these operand and destination storage precisions, before any geometry or site structure is
    consulted — [None] when they do not (gh-ocannl-1115). {!matmul_sketch_tree} refutes its
    tensorized branch with exactly this witness. On GPU: routine logging, an mma lane wider than the
    workgroup, no advertised format tile; on CPU, the register tiling's shape-independent rules: a
    usable vector file, two lanes at the compute precision, uniform vector-capable compute
    precisions, a register file holding a four-row tile ({!Ir.Register_tile.rn_cap}), routine
    logging. Exposed so a test claims a tensorized seed's PRESENCE exactly where the seeder's own
    capability judgment admits it, never gating on the seed list under test. *)

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
    | Batch of [ `Serial | `Grid | `Grid_inner ]
        (** The batch-geometry twin (gh-ocannl-643), GPU only; [`Grid_inner] binds the interior
            batch loops inside the row blocks (gh-ocannl-728, [sk_batch_inner]). *)
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
    | Conv_leg of [ `Cpu | `Gpu ]
        (** The conv family's route (gh-ocannl-697): the CPU register-tiled implicit-GEMM pipeline
            or the GPU staged one; the leg the backend kind does not take is refuted. *)
    | Conv_shape of [ `Serial | `Grid | `Row_block of int ]
        (** The CPU conv pipeline's flavor: serial, Grid-parallel, or row panels of that many rows
            (gh-ocannl-500). *)
    | Conv_staging of [ `Plain | `Pipelined ]
        (** The GPU conv pipeline's staging: single-copy tiles, or rotating copies whose depth the
            [twin] level below each geometry commits (gh-ocannl-487). *)
    | Conv_rows of int
        (** The GPU conv geometry: the GEMM row in Grid blocks of that many rows, [0] the whole
            (intrinsic-tile padded) extent. *)

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
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
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
    candidate build never expands (gh-ocannl-577). The conv family is its own tree under the same
    root, {!conv_sketch_tree} (gh-ocannl-697). Exposed for tests and as the phase-4 search driver's
    entry into the family space. *)

val conv_sketch_tree :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree option
(** The convolution sketch family as a refinement tree (gh-ocannl-697): see
    {!Sketch_families.conv_sketch_tree}. [None] when no conv site is detected. Exposed for tests. *)

val sketch_family_tree :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree option
(** The tree of whichever family claims the lowering ({!matmul_sketch_tree} where a matmul site is
    detected, else {!conv_sketch_tree}), unforced below its root: {!sketch_seed_params} is its
    leaves, and {!model_family_search} walks it. *)

(** {2 The seeding census (gh-ocannl-1407)}

    Seeding refusals are decided before a candidate exists, so the decline report and
    [schedule_log_declines] (render and compile declines) cannot hold them. The census reads them
    off the judgment the seeds come from; config [autotune_log_seeding] prints it for every lowering
    {!tune} and {!model_default} seed, and one line per segment of the plan the untuned seeded
    default chooses. See {!Sketch_families.seeding_census} for the fields' documentation.

    The census is the families' seeding. A driver's own filters and lifts after it are not seeding
    gates and are not in it: {!tune}'s keep-fraction model pre-filter, and {!model_default}'s
    numerics restriction on CUDA/HIP and its geometry-lattice lift (which re-proposes branches the
    head line counts in [excluded=N]). *)

type family_seeding =
  | Seeding_matmul of {
      site : matmul_site;
      tree : (Family_decision.t, sketch_params) Ir.Schedule_space.tree;
    }
  | Seeding_conv of {
      site : conv_site;
      real_stmts : int;
      tree : (Family_decision.t, sketch_params) Ir.Schedule_space.tree;
    }
  | Seeding_none

val family_seeding :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  family_seeding * sketch_params list
(** {!sketch_seed_params} beside the judgment its seeds came from. *)

type coalesced_verdict = {
  cv_layout : coalesce_layout;
  cv_seeds : sketch_params list;
  cv_refusal : string option;
}

val coalesced_seeding :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  coalesced_verdict list

type seeding_census = {
  sc_stmts : int;
  sc_whole_zeros : int;
  sc_conv_nests : (Ir.Tnode.t * bool) list;
  sc_family : family_seeding;
  sc_seeds : sketch_params list;
  sc_coalesced : coalesced_verdict list option;
}

val seeding_census :
  ?coalesced:bool ->
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  seeding_census

val seeding_census_prefix : string
val seeding_census_lines : where:string -> seeding_census -> string list

val log_seeding_enabled : unit -> bool
(** Config [autotune_log_seeding], read once per process. *)

val log_seeding :
  ?coalesced:bool ->
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  where:string ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  unit
(** The census of one lowering on stderr, tagged [where]; a no-op unless {!log_seeding_enabled}. *)

val geometry_lattice_witness : string
(** The exclusion witness marking the tile-size lattice branches beyond the curated geometry menus
    (gh-ocannl-514 phase 5): binary interval refinements over the staged tile sizes — every
    intrinsic-tile multiple of the row block crossed with every staged depth block — whose boxes
    carry corner-judged verdicts (a workgroup-memory floor at the most favorable corner refutes the
    whole box pre-expansion). {!Ir.Schedule_space.leaves} never enumerates an excluded branch, so
    the tuner's seed lists are unchanged by the lattice's existence. *)

val pad_waste_max : unit -> float
(** The pad-waste threshold of the matmul family's geometry menus (gh-ocannl-740), as this process
    resolves config [autotune_pad_waste_max] (at least 1; a malformed or NaN value raises
    [Utils.Invalid_config_value], gh-ocannl-1332). A geometry's pad-waste ratio is its padded work
    over the site's: the product, over the axes it pads to its block sizes, of padded over real
    extent. A geometry whose ratio is at least this many times that of the least-padded geometry its
    own menu proposes is {!Ir.Schedule_space.child.Excluded} with a witness naming both ratios:
    valid, never proposed by default, re-proposed exactly by {!Ir.Schedule_space.lift_excluded}. The
    least-padded geometry is never excluded, and legality refutations take precedence. Judged in the
    GPU blocktile menu, the GPU tensorized curated menu and the CPU packed menu of each packing
    shape, and in the conv family's CPU shape menu and GPU row-geometry menus (gh-ocannl-1393),
    where nothing is excluded at the default threshold: a row block needs two or more blocks, so its
    waste stays under 2x, and the GPU channel pads are site-wide, so they cancel. Exposed so a test
    judges the tree against the threshold the tree was built with. *)

val lift_pad_waste :
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree ->
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree
(** Lift every {!pad_waste_max} exclusion in the family tree (gh-ocannl-740), as
    {!lift_geometry_lattice} lifts the lattice: laziness preserved, the lifted geometries still
    subject to legality, other policies' exclusions kept. Its {!Ir.Schedule_space.leaves} are what
    the menus may propose with the pad-waste economy off. *)

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

val sketch_schedule :
  accum_prec:(Ir.Ops.prec -> Ir.Ops.prec) ->
  p:sketch_params ->
  Ir.Low_level.optimized ->
  Ir.Schedule.schedule
(** The composed pipeline a seed parameterizes, built against the given lowering (the site is
    re-detected). Refuses typed when no site is detected or the parameters do not fit the segment
    ({!Sketch_families.sketch_schedule}: [Ir.Schedule_outcome.Cause_at (Transform, _)]); any other
    exception escaping the builder is a builder bug, fatal under [strict_failure_classification]
    (gh-ocannl-1390). [accum_prec] is the rendering backend's accumulator resolution
    ([(Context.codegen_capabilities ctx).accum_prec]): the scalar pipelines' [Privatize] mints its
    tile at it ({!Ir.Schedule.privatize}, gh-ocannl-1116). Exposed for tests (the pad-composition
    seeding of gh-ocannl-485 is executed directly). *)

val with_sketch_builder_fault : (sketch_params -> unit) -> (unit -> 'a) -> 'a
(** Test seam: see {!Sketch_families.with_sketch_builder_fault} (gh-ocannl-1390). *)

val extend_with_privatize :
  accum_prec:(Ir.Ops.prec -> Ir.Ops.prec) ->
  static_indices:Ir.Indexing.static_symbol list ->
  Ir.Schedule.schedule ->
  Ir.Low_level.optimized ->
  Ir.Schedule.schedule
(** The privatized preset extension used by the fissioned candidates: appends a
    [Schedule.privatize ~accum_prec ~target ~over] for every materialized read-modify-write
    accumulator detected in the schedule's application to the segment — [over] being the outermost
    enclosing [Serial] loop whose symbol the access vector does not mention and whose subtree
    contains no hardware-typed loop. Each proposal is validated by try-applying the grown schedule
    against a hermetic copy of the segment (proposals violating the op's preconditions are dropped),
    so the result always applies cleanly where the input schedule does. Exposed for tests. *)

type privatize_drop = {
  pd_target : Ir.Tnode.t;  (** The accumulator the proposal would privatize. *)
  pd_over : Ir.Indexing.symbol;  (** The serial loop it would privatize over. *)
  pd_cause : Ir.Schedule_outcome.cause;  (** The refusal that dropped it. *)
}
(** A {!extend_with_privatize} proposal the op refused. *)

val privatize_extension :
  accum_prec:(Ir.Ops.prec -> Ir.Ops.prec) ->
  static_indices:Ir.Indexing.static_symbol list ->
  Ir.Schedule.schedule ->
  Ir.Low_level.optimized ->
  Ir.Schedule.schedule * privatize_drop list
(** {!extend_with_privatize} beside the proposals it dropped, in detection order. Where a search's
    privatized presets drop proposals, they reach its report as [privatize_drops] (gh-ocannl-1438);
    this is where a census of [Privatize]'s refusals on segments no search compiles reads them
    (gh-ocannl-1395: [test/operations/privatize_census]). Empty when the input schedule itself does
    not apply. *)

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
  sr_coalesce : (Ir.Indexing.symbol * Ir.Indexing.symbol) option;
      (** gh-ocannl-1414: the reduction pair [(outer, inner)] that a
          {!Ir.Schedule.constructor-Coalesce}, applied after [sr_swaps] and before the split, merges
          into the loop [sr_axis] names (see {!coalesce_reduction_pair}); [None] when [sr_axis] is a
          loop of the lowering. *)
  sr_block_hoist : Ir.Indexing.symbol list;
      (** gh-ocannl-1412: the accumulation-cell loops enclosing [sr_axis] once [sr_swaps] and
          [sr_coalesce] are applied, outermost first. The prelude interchanges the split's block
          loop above them, so the first pass runs [for block: for cells: ..] and the GPU presets
          give its lanes the cell axes, contiguous in memory, rather than the blocks, [chunk] rows
          apart. Empty for a scatter site and where that interchange is not [Op_legal]. *)
}
[@@deriving sexp_of]
(** A reduction-dominated accumulation site eligible for {!Ir.Schedule.constructor-Split_reduce}
    seeding (gh-ocannl-484 task 3); see the implementation's field docs. Exposed for tests. *)

val split_sites_label : ?staff:(sr_site * int) list -> (sr_site * int) list -> string
(** The label of a split-reduce candidate ([F_split[...]]) over its [(site, num_blocks)] entries, in
    the [autotune_log] lines and {!report.best_label}: each entry renders every field of its site —
    the target by debug name and, where that differs, its id ([name#id]), the split axis, [red],
    [out], [cost], the block count, [dyn], the interchange chain pair by pair ([swap:o/i+o/i]) and
    the coalesced reduction pair ([coalesce:o*i]) and the cell loops the block loop is hoisted above
    ([hoist:c+c]) — so two sites that differ in any one field never share a label (gh-ocannl-1230).
    Entries are separated by [','], which no entry contains. *)

val coalesce_reduction_pair :
  ?static_indices:Ir.Indexing.static_symbol list ->
  Ir.Low_level.optimized ->
  outer:Ir.Indexing.symbol ->
  inner:Ir.Indexing.symbol ->
  (Ir.Indexing.symbol * Ir.Low_level.optimized) option
(** gh-ocannl-1414: the loops [outer] and [inner] merged by a {!Ir.Schedule.constructor-Coalesce},
    as the merged loop's fresh symbol and the coalesced code (built on a hermetic copy), or [None]
    where the op declines: the pair is not a perfect nest of two [Serial] loops, or some access does
    not read it as two adjacent plain iterators over unpadded axes. {!split_reduce_sites} prepares a
    two-loop reduction band with it, and a site it prepares records the pair in [sr_coalesce], so
    the candidate's prelude replays the same op (under the same merged symbol) before the split.
    Exposed so that a caller preparing its own reduction pair, such as a contraction over two axes,
    reaches the split through the same preparation. *)

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

    A site no single loop splits — a small-batch parameter gradient [for b, s, o: g[o] += dy[b,s,o]]
    with [b] too short to split and [s] enclosed by [b], which pins no component of the cell — is
    re-probed with its two outermost reduction loops merged (gh-ocannl-1414): the cell loops below
    the outer one are hoisted as above, the pair is merged by {!coalesce_reduction_pair}, and the
    split is probed on the merged loop, whose extent is [sr_red]. A site one loop already splits
    keeps that split. [static_indices] only reaches the interchange and coalescing probes'
    [Sched.apply]. Exposed for tests. *)

val split_reduce_prelude : (sr_site * int) list -> Ir.Schedule.schedule
(** The whole-routine prelude splitting each [(site, num_blocks)]: per site its interchange
    ([sr_swaps]), its coalescing ([sr_coalesce], minting the merged loop under the symbol [sr_axis]
    names, as detection probed it), the {!Ir.Schedule.constructor-Split_reduce}, then the [Swap]s
    hoisting the split's block loop above the cell loops [sr_block_hoist] names (gh-ocannl-1412).
    The schedule the search's [F_split] candidates and the untuned split ({!seeded_split_sites})
    both apply. Exposed for tests. *)

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

type abandonment = {
  ab_timed : int;
      (** How many admitted timings the verdict was taken after: the search's beam width, the depth
          at which its first beam is full. *)
  ab_best_ms : float;  (** This search's best after its first [ab_timed] admitted timings. *)
  ab_incumbent_ms : float;
      (** The incumbent's best after {e its} first [ab_timed] — the same depth, not its final best.
      *)
  ab_ratio : float;  (** The trailing ratio [ab_best_ms] exceeded, against [ab_incumbent_ms]. *)
}
(** Why a search was abandoned by its caller's [?abandon] rule (gh-ocannl-1110). *)

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
  | Abandonment_replay of abandonment
      (** Cached timings under this call's search shape still satisfy its current [?abandon] rule.
          No search ran, the counters describe only the base compile, and {!tune} raises
          {!Search_abandoned} without returning a routine. *)
  | Search_disabled
      (** Nothing was searched and there was nothing to replay: config [autotune_search=false] (the
          reproducible profile, gh-ocannl-559) with no chosen cache, or a chosen cache that missed.
          The caller gets the untuned default compile (see {!tune}'s untuned fallback: the seeded
          plan on CUDA and HIP, the code {!Context.compile} would have produced elsewhere). Every
          counter is zero and every time [infinity]; a declined baseline can still populate
          [declines]. *)
  | Pre_search_failure of terminal_failure
      (** The call failed before (or instead of) the search proper — a base compile that failed
          before its lowering was captured, a fatal baseline link, a fatal cache replay, a baseline
          timing failure, or an untuned fallback compile of a search-less call — and raised. It
          still reports (gh-ocannl-550), carrying whatever census the call had reached, so a caller
          attributing arms by arrival order (the positional [?report] of {!Train.tune_placements})
          gets a slot for it. *)
  | Abandoned of abandonment
      (** The caller's [?abandon] rule stopped the search once its first timed candidates trailed
          the incumbent's at the same depth by more than the rule's ratio (gh-ocannl-1110), and
          {!tune} raised {!Search_abandoned}. Not a failure: the counters hold the work it reached,
          [best_ms] is a measurement of the search context, nothing was compiled for the caller and
          clean measured prefixes are cached for a later {!Abandonment_replay}. *)
  | Lineage_refused of { caller_digest : string; timing_digest : string }
      (** gh-ocannl-1421: the [timing_ctx] lowers this routine to another program than the caller's
          context does -- the {!lineage_identity} digests of the two lowerings, taken together and
          carried here, differ, as when the timing lineage materializes a node the caller's leaves
          virtual. Every candidate derives from the timing lineage's base lowering, so no schedule
          tuned or cached there can be compiled into the caller: nothing is searched or replayed,
          and the caller gets the untuned default compile, as under {!Search_disabled}. Every time
          is [infinity], so a caller ranking searches ([Train.tune_placements]) ranks it last; it is
          no failure. Decided before the cache is consulted, from an analysis-only lowering of the
          caller's context ({!Context.lowered_for_decisions}). *)

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
          refuses on the number itself (gh-ocannl-888). Not set for {!field-unbatched}'s refusal,
          which has its own reason (gh-ocannl-1098).

          Because they are separate, [not contended] is HALF a proof of usability: a consumer
          deciding whether to keep a number asks {!admitted_timing_ms}, never this field. Reading it
          directly is for saying something about contention itself — a diagnostic, a test that
          excuses a claim under host load. *)
  unbatched : bool;
      (** A CUDA/HIP queued calibration measured no batch within its target for a candidate whose
          single launch owed it one (gh-ocannl-1096), even at the rescue probe below its shallowest
          over-target batch (gh-ocannl-1098): only an isolated depth-1 reading was left, and the
          readings cannot tell a stall from a queue threshold, so the reading is refused exactly as
          a contended one is. The flag names what calibration MEASURED, not a diagnosis: host load
          stalling every batched probe reads the same. It is kept apart from [contended] because the
          two call for different follow-ups -- a no-batch refusal that persists across idle reruns
          is a threshold's signature, which a count mixed into the contention verdict hides. A
          refusal like [contended]: {!admitted_timing_ms} admits neither. *)
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
    2 of 8 runs. Re-measured on the same device at the same site (gh-ocannl-833, 2026-09-27) the
    offset had shrunk to 0-6 us ({!Isolated} at most 1.08x {!Queued}), and the crown still moved in
    3 of 4 site-runs, {!Queued} keeping the batched-throughput winner in all 4: the offset's
    candidate-to-candidate variation, not its size, is what reorders a close pair. The same
    site-runs moved the isolated crown in 3 of 4 on M4 Max Metal (offset 42-143 us) and in 0 of 4 on
    CUDA sm_120 (a uniform 3.8-5.6 us against a 10-18% gap between the top two), and {!Queued}
    crowned the batched winner in every site-run on all three backends. Consequently a [best_ms]
    measured under {!Isolated} is not a throughput number and must not be compared with a batched
    per-kernel figure; under {!Queued} it is, up to the batch's residual ~1% of round trip. *)

(** Which exit of the calibration chose the depth, before the post-calibration bounds
    ({!batch_decision}'s [bounded_from], [fallback_from], [rescue]) adjusted it. *)
type batch_settle =
  | Isolated_objective  (** {!Isolated}: no calibration, depth 1. *)
  | Single_estimate  (** The cc/Metal policy: the synchronized singles' estimate alone. *)
  | Singles_reach_target
      (** The singles reached the target and the provisional depth-2 pair confirmed depth 1. *)
  | Measured_batch  (** A resolved fit confirmed a depth a probe (or the singles) measured. *)
  | Affine_projection  (** A resolved fit's projection to a depth no probe measured. *)
  | Wall_scaled  (** A linear scale from the deepest measured batch, the fit being unresolved. *)
  | Cap_measured  (** The cap, measured by a probe. *)
  | Cap_unresolved  (** The cap, wanted by an unresolved monotone pair, unmeasured. *)
  | Budget_exhausted
      (** The probe wall or count budget ended the calibration unresolved, at the depth in hand. *)

(** The call's admission verdict, in {!admitted_timing_ms}'s terms; a refusal names the first of
    [unbatched], [contended], an unreadable reading that holds. *)
type batch_admission = Admitted | Refused_unbatched | Refused_contended | Refused_unreadable

val batch_settle_name : batch_settle -> string
(** The settle's spelling in the benchmarks' timing trace and result line: [isolated],
    [single_estimate], [singles_reach_target], [measured_batch], [affine_projection], [wall_scaled],
    [cap_measured], [cap_unresolved], [budget_exhausted]. *)

val batch_admission_name : batch_admission -> string
(** Likewise: [admitted], [refused_unbatched], [refused_contended], [refused_unreadable]. *)

type round_unit_moves = {
  unit_segment : int option;
      (** [Some i]: the entry's [i]th fission segment, the [i] a round candidate's label names as
          [+seg<i>]; [None]: a whole-routine entry's one unit. A fissioned entry's zero-fill and
          solo segments are not units: no move extends them, and they are not listed. *)
  unit_expanded : bool;
      (** Whether the rule offered this unit to the action menu. An unexpanded unit cost no legality
          check and has [unit_moves = 0]; an expanded one with [unit_moves = 0] is a unit whose menu
          had nothing to propose -- a tensorized or fully register-tiled geometry, say. *)
  unit_moves : int;
      (** The round candidates this unit's moves became, each of which the round attempts (one
          {!on_candidate_attempt} call apiece, whose label names the entry as [from <tag>] and, on a
          fissioned entry, the unit as [+seg<i>]) unless the search stops first. An attempt can
          still end before a timing window: bound-pruned, declined, deduplicated by digest. *)
}
(** One unit of a beam entry in a beam-expansion round (gh-ocannl-1360): what the round's expansion
    rule (config [autotune_round_expansion]) did with it. *)

type round_entry_moves = {
  entry_digest_tag : string;
      (** {!digest_tag} of the entry's digest: the tag its round candidates' labels name as their
          parent ([from <tag>]), and the tag of the digest {!on_candidate_measured} saw when the
          entry was timed. *)
  entry_ms : float;  (** The time that ranked it into the beam. *)
  entry_segments_differing : int list option;
      (** The entry's segments whose schedule differs from the round's head (the first entry), in
          [unit_segment]'s indexing: [Some []] for the head itself and for an entry identical to it
          segment by segment; [None] when the two do not segment alike or are whole-routine
          programs. *)
  entry_units : round_unit_moves list;  (** Every unit of the entry, in segment order. *)
}
(** One beam entry as a round found it. *)

type round_moves = {
  round_index : int;  (** 1 for the first round, as in [rounds_run]. *)
  round_expansion_rule : string;  (** [differing], [full] or [head]. *)
  round_entries : round_entry_moves list;  (** The beam, fastest first: the head, then the rest. *)
}
(** One beam-expansion round (gh-ocannl-1360): the per-entry, per-unit move counts the round's
    [autotune_log] line [round N: ...] summarizes, recorded before the round attempts any of them. A
    test reads them to choose a beam entry with the moves it needs instead of discovering their
    absence by failing a claim. *)

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
          [candidates_contended] and [default_refused] for the per-candidate facts. Includes the
          [timings_unbatched] windows. *)
  timings_retried : int;
      (** Baseline and candidate timing calls that started a fresh contention retry (Metal queued
          timing only). Counts each discarded first window once, whether its retry was admitted,
          refused or failed. Independent of [timings_contended], which counts final refusals only.
          Zero on cache replay and search-disabled calls. Excludes diagnostic control timing. *)
  timings_unbatched : int;
      (** Of [timings_contended], the windows refused because queued calibration measured no batch
          within its target ({!field-unbatched}, gh-ocannl-1098) rather than for dispersion or a
          degenerate clock. Not a diagnosis -- a stall on every batched probe reads the same as a
          queue threshold the rescue probe could not get under -- but a count that persists across
          idle reruns is the threshold's signature, which the contention count alone hides. Every
          refusal still makes the measurement set incomplete. *)
  timing_decisions : (batch_settle * batch_admission * int) list;
      (** How this call's baseline and candidate timing calls settled their depth and were admitted
          (gh-ocannl-1263): one [(settle, admission, calls)] per pair that occurred, in constructor
          order, from each call's own {!timing_call} record — the settle-by-admission matrix that
          lets a sweep compare calibration behaviour across boxes without scraping a trace. Counted
          per call through {!time_routine}'s [on_call], so neither another observer nor a search in
          another domain moves it. Empty on cache replay and search-disabled calls, which time
          nothing; excludes diagnostic control timing, like [timings_retried]. *)
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
          case this exists for — before it was contained, that one rejection ended the search. Since
          gh-ocannl-1340 a GPU backend's baseline never reaches codegen (it binds no hardware
          dimension, so it is refused in the transform and counted under [Not_dispatched_key]), so
          this is [false] there and the flag describes CPU backends. *)
  baseline_refused : bool;
      (** The dispatched serial baseline ran its timing window and the window was refused
          (gh-ocannl-1423): contention, a queued calibration that measured no batch within its
          target, or a degenerate clock reading. [baseline_ms] is then [infinity] although the
          baseline ran, and the window counts in [timings_contended]. This is the baseline's own
          record, where [timings_contended] counts every refused window: a claim about the
          baseline's measurement gates on it, so a loaded host reads as a skip and every other
          [infinity] (a baseline never relinked, never dispatched, declined) still fails the claim.
          An identical seed may time the same code later, which [default_ms] attributes by digest;
          [baseline_ms] stays the baseline's own reading. [false] on cache replay and
          search-disabled calls, which time nothing, and on a declined or undispatched baseline. *)
  declines : decline_summary list;
      (** Candidate rejections aggregated by stable cause key. Their counts sum to
          [candidates_failed]. Cache-entry replay failures are excluded: they are
          [cache_replay_refusal]; so are the privatized presets' dropped proposals, which are
          [privatize_drops]. *)
  privatize_drops : decline_summary list;
      (** The [Privatize] proposals the privatized preset candidates ([F_preset[... priv]]) dropped
          because the op refused them ({!privatize_extension}'s drops, gh-ocannl-1438), aggregated
          by cause key as [declines] is: one count per proposal per candidate compile that made it,
          so a refusal every privatized preset meets counts once per such preset. A drop is not a
          declined candidate -- the candidate compiles without that [Privatize] -- so these counts
          are outside [candidates_failed]. Empty on cache replay and search-disabled calls, which
          compile no preset. Before gh-ocannl-1438 the search discarded them, so a census of
          [Privatize]'s refusals ([test/operations/privatize_census]) could not read them from a
          report. *)
  cache_replay_refusal : Ir.Schedule_outcome.classified_cause option;
      (** Why the cache entry stored under this call's key did not replay, when its replay was
          refused with a typed cause (gh-ocannl-1242): a stale or corrupt entry — a reference its
          saved schedule makes that this lowering's canonical form cannot resolve
          ([check = "Schedule_cache.of_saved"] or ["Schedule_cache.tn_of_ref"]), a fission segment
          that drifted from the one its schedule was saved against ([Autotune.fission_replay]), a
          transform precondition the replayed schedule violates ([Schedule.apply]). The call then
          proceeds as on a miss — it searches, and the search's store overwrites the entry, or
          compiles the untuned default under [autotune_search=false] — and every report it emits
          carries the cause. [None] when no entry was found or matched its key's self-description,
          when the found one replayed. A replay that binds no hardware dimension on a GPU backend
          (gh-ocannl-532) is refused too, typed: [Not_dispatched { origin = "cache_replay"; _ }], at
          phase [Transform] when the replayed schedule's forms were refused before codegen, and at
          [Backend_codegen] when only the kernels codegen shipped bind none (gh-ocannl-1340). An
          untyped failure of the replay is not a refusal: it ends the call as a [Pre_search_failure]
          under strict failure classification. *)
  rounds_run : int;  (** Beam-expansion rounds actually executed (0 = seeds only). *)
  round_moves : round_moves list;
      (** Every round this call's search started, in order (gh-ocannl-1360): one record per round
          counted in [rounds_run], including a round a fatal failure or an abandonment interrupted,
          whose report carries the record of the round in flight. Empty on cache replay and
          search-disabled calls, and for [~rounds:0]. *)
  budget_cut : bool;
      (** The crown comes from a search that stopped proposing candidates because its wall-clock
          budget (config [autotune_budget_seconds], gh-ocannl-1409) ran out: this call's own search,
          whose candidates timed are then a prefix of its menu plus the untuned floors (exempt from
          the budget), or for a {!Cache_replay} the search that stored the entry, as the entry's
          [budget_cut] records -- like [best_ms], a fact about the crown rather than about this
          call. [outcome] stays {!Searched} for a cut search -- it completed and shipped its crown
          -- so this is the field telling a cut search from a full one. *)
  budget_s : float option;
      (** The wall-clock budget this call resolved (config [autotune_budget_seconds],
          gh-ocannl-1409), [None] for none: what [budget_cut] was measured against, on every report
          the call emits, a cache replay's and a search-less call's included (a fact about this
          call, like [beam_width], where [budget_cut] on a replay is the stored crown's). The budget
          is also the replay rule's input: only a call with a budget replays a cut crown. Recorded
          here rather than in a benchmark result's [regime_knobs], whose keys are numerics settings
          the orchestrator's regime gate holds to the cell's regime (gh-ocannl-1438). *)
  beam_width : int;
      (** The beam width the search resolved ([?beam_width], else config [autotune_beam_width]): how
          many of the fastest candidates each round expands, and the admitted-timing depth an
          [?abandon] rule decides at. 0 in {!no_search_report}'s template, which {!tune} fills. With
          [rounds_run], what a configured [autotune_rounds] / [autotune_beam_width] actually reached
          — a caller passing [~rounds:0], as the benchmark runners do, runs no round whatever the
          configuration says (gh-ocannl-1137). *)
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
      (** Per-fission-segment sketch candidates seeded (0 when segmentation neither splits the
          computation nor exposes a changed lowering, or no segment contains a compatible site). GPU
          zero expansion can expose a sketchable singleton with a per-cell zero companion. Includes
          the finer-segmentation ([arity_cuts], gh-ocannl-574) singles on GPU backends.
          Deterministic given the computation and backend. *)
  fiss_sketch_timed : int;
      (** Of the seeded per-fission-segment sketch candidates, those that compiled and were actually
          timed (not rejected by op preconditions or hardware limits, not deduplicated by digest).
          Includes completed timing windows refused as unusable; [timings_contended] identifies
          those verdicts. Includes coarse and fine recombination windows, so it can exceed
          [fiss_sketch_candidates]. *)
  fiss_sketch_composite : [ `Ineligible | `Singles_refused | `Proposed | `Refused | `Timed ];
      (** Coarse recombination's own outcome (excluding the finer-fission composite): [`Ineligible]
          means no search, a decision not yet reached, or fewer than two usable coarse singles.
          [`Singles_refused] refines that absence when every missing segment key has a refused
          window and no admitted single. [`Proposed] means at least two usable singles staffed a
          proposal, but it did not complete a timing window; [`Refused] and [`Timed] record that
          proposal's refused or admitted window. Window outcomes are published before post-admission
          callbacks can raise, including in partial reports. With staffed singles (config
          [autotune_staffed_singles], gh-ocannl-1138) the coarse composite is the running composite,
          which its last accepted staffed window already timed: [`Timed] then records that window,
          and only a composite no window timed is proposed and timed apart. *)
  fiss_sketch_playoff_timed : int;
      (** The composite playoff's windows (gh-ocannl-1166), counted here and NOT in
          [fiss_sketch_timed]. A per-segment single is timed against every other segment's untuned
          preset, a backdrop several times the composite's time, so near-tie singles are re-ranked
          inside the faster timed composite: per keyed segment, up to config
          [autotune_composite_playoff_width] (2) singles within config
          [autotune_composite_playoff_margin] (0.005, a fraction) of the segment's best single take
          its place; width [0] turns the playoff off, and this count stays [0] (gh-ocannl-1259).
          Changing the default margin or the default width bumps [Schedule_cache.entry_version].
          Includes refused windows, like [fiss_sketch_timed]. *)
  fiss_sketch_playoff_swaps : int;
      (** Of [fiss_sketch_playoff_timed], the alternates that ran faster than the incumbent
          composite and replaced it, so later segments' alternates build on them. *)
  fiss_sketch_staffed_timed : int;
      (** Of [fiss_sketch_timed], the staffed singles' windows (gh-ocannl-1138, config
          [autotune_staffed_singles]): singles timed with the other keyed segments on the running
          composite's choices rather than on their presets. [0] with the setting off. *)
  fiss_sketch_backdrop_fallbacks : int;
      (** Staffed singles that were refused (a compile or run decline, not a contended window) and
          retried once without their staff (gh-ocannl-1138), so one invalid choice in the running
          composite cannot mask another segment's seeds. Every staffed refusal of a key is retried
          until a staffed single of that key is admitted (or the composite's own choice for it is
          reused), which shows the staff compatible: later refusals are the seeds' own and are not
          retried. *)
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
          composite was ineligible, on cache replay, and on search-disabled calls. With staffed
          singles (gh-ocannl-1138) the composite is the running split composite, and its window the
          last accepted staffed split single's. *)
  split_reduce_backdrop_fallbacks : int;
      (** The split-reduce counterpart of [fiss_sketch_backdrop_fallbacks] (gh-ocannl-1138): staffed
          split singles refused and retried with no other site split, under the same rule. *)
  mma_candidates : int;
      (** Candidates whose label promises a tensorized ([Schedule.Tensorize]) pipeline that the
          search put through candidate compile: whole-routine and per-fission-segment sketch seeds,
          the cross-segment recombination composite, composite-playoff alternates, and
          beam-expansion candidates. Counted at the same point as [mma_timed], so the two always
          describe the same population. *)
  fiss_mma_candidates : int;
      (** Of [mma_candidates], candidates built from per-fission-segment MMA sketches (including
          their recombination composites and composite-playoff alternates). This is counted at
          candidate compile, not enumeration, so it detects a fission MMA family that disappeared or
          never reached compilation without borrowing evidence from a whole-routine candidate. *)
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
          and cannot win. Also [infinity] when [baseline_declined] or [baseline_refused]. *)
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
          ({!Ir.Schedule.default_schedule_fingerprint} mismatch). On CUDA and HIP the recipes'
          untuned compile ships the seeded plan instead ({!seeded_default}), whose time is
          [seeded_ms]. *)
  seeded_ms : float option;
      (** The seeded untuned plan's measured time (gh-ocannl-1388): the step {!seeded_default} ships
          on CUDA and HIP, which the search seeds FIRST, so on a completed search with
          [seeded_ms = Some s], [best_ms <= s] by construction, and a search that timed only its
          first candidate crowned it. Attributed by digest, like [default_ms]. [None] where no plan
          was seeded (another backend, or a plan that is the default pipeline, whose time is
          [default_ms]), when its seed failed to compile or both its window and its one retry were
          refused, and on a cache hit (not stored). *)
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
          ["F_sketch[mma-gpu 16x32x32 simd32 ep @1a2b3c4d/5e6f7a8b]"]). ["baseline"] when no
          candidate beat the serial baseline, and [""] exactly when nothing was timed — including
          the states that time nothing by construction, which say so in [outcome] rather than
          through this string. Which candidate won is otherwise recoverable only by matching
          [best_ms] against the log's per-candidate times (gh-ocannl-546). *)
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
  source_digest : string;
      (** The digest of the base lowering this call tuned (gh-ocannl-1022):
          {!Ir.Schedule_cache.digest} of the placement-aware canonical form the base compile
          captured inside its [lowered_transform], in the search's lineage ([timing_ctx] when one
          was given). It is the digest the call's schedule entry is stored and looked up under, and
          what [Train.tune_placements] records as a placement decision's [outcome_digest] — what the
          shipped search actually tuned, rather than what a re-lowering says it would have.

          Carried by every report the call emits once the capture has happened: completed searches,
          cache replays, dead searches, a search-less call's cache miss, and the pre-search failures
          that follow the base compile. [""] exactly when the call never reached a base lowering: a
          search-less call with no cache directory (which skips the base compile), and a failure
          before the base compile's transform ran (a failure after it, at codegen or link, carries
          the digest of the lowering it had captured). *)
  best_steps : (int * float) list;
      (** The best-so-far as a step function of the admitted timings (gh-ocannl-1110): [(n, ms)]
          says the search's best became [ms] at its [n]-th admitted timing (the dispatched baseline
          is the first), ascending in [n] and descending in [ms], so {!best_after} reads the best at
          any depth. What {!Train.tune_placements}' flip chain abandons a hopeless flip against. A
          {!Cache_replay} carries the storing search's steps, like its times, when that search had
          this call's shape ({!Ir.Schedule_cache.trajectory}); empty when nothing was timed, for an
          entry older than the field, and for one stored under another shape. *)
  compile_cost : Ir.Compile_cost.t;
      (** The compiles this call made before delivering this report -- the baseline's, each
          candidate attempt's, a cache replay's, the winner's and any untuned fallback's -- summed
          by phase (gh-ocannl-1314), each as one {!compile_record} reports it. A deferred call's
          ({!tune_deferred}) shipping compile runs after the report and is in its own
          {!compile_record} only. Unlike the times above, it describes this call even on a
          {!Cache_replay}. [kernels_compiled < kernels] when the backend reused compiled kernels
          ([hip_kernel_cache_mib]). *)
}

val no_search_report : timing:timing_mode -> report
(** The report of a {!tune} call that never searched (config [autotune_search=false], gh-ocannl-559,
    and no cache entry to replay): [outcome = Search_disabled], every counter zero, every time
    [infinity], [best_label] empty, [best_tensorization = None] and [source_digest = ""]. The caller
    gets the untuned default compile. Also the base the pre-search failure reports are built on,
    with [outcome] replaced and whatever census the call had reached filled in.

    [timing] is the objective the call resolved, which is all that distinguishes one of these from
    another: it names what the (absent) times {e would} have been measured under, and keeps
    {!report.timing} a plain field rather than an option every consumer of a real report would have
    to unwrap. A caller synthesizing a report — a test building a search outcome to feed
    {!family_profit_of_reports}, say — passes whichever objective its scenario is about. *)

(** {2 Progress lines (gh-ocannl-1061)}

    A search's cost record, for a search that may be killed before it reports: with config
    [autotune_progress=true] (default [false]), or for one call its [~progress:true], {!tune} and
    {!Train.tune_placements} write single lines to stderr, each flushed as it is written, of the
    form

    {v autotune-progress: wall_s=<s> event=<event> <key>=<value> ... v}

    [wall_s] is seconds since the process started (this module's initialization); string values are
    OCaml-quoted ([%S]), so a line splits on spaces outside quotes, and a time that was never
    measured prints as [none]. The events, with the keys each carries beyond [wall_s] and [event]:

    - [search_start] ({!tune}): [routine], [elapsed_s], [backend], [device], [search], [beam],
      [rounds].
    - [phase] ({!tune}, one per phase with a known size): [routine], [elapsed_s], [phase] ([seeds]
      or [round<k>]), [candidates] (the phase's total), [timed], [attempts], [compile_s],
      [timing_s], [best_ms], [best].
    - [candidate] ({!tune}, as EVERY candidate attempt starts): [routine], [elapsed_s], [phase]
      ([seeds], [staffed] (gh-ocannl-1138's staffed sweep, config [autotune_staffed_singles]),
      [recombine], [playoff] or [round<k>]), [tried=<k>/<total>] (the ordinal of the attempt
      starting within its phase; [?] when the phase's total is not known up front), [timed],
      [attempts] (this one included), [compile_s], [timing_s], [best_ms], [best] (all as of the
      attempt's start), and [attempt], the starting candidate's label. A search killed inside a
      candidate is therefore inside the one its last [candidate] line names. The rate is bounded by
      the attempts themselves, not by a clock: a time bound would leave a window in which the
      candidate being worked on is unnamed.
    - [compiled] ({!tune}, as every compile it makes RETURNS, gh-ocannl-1314; see
      {!compile_record}): [routine], [stage], [attempt], [verdict], [took_s] (this compile's wall),
      [lower_s], [transform_s], [codegen_s], [device_s] ([none] where the backend compiles at link),
      [link_s], [kernels] and [kernels_compiled]. A candidate's [compiled] line follows its
      [candidate] line; a compile killed before returning writes none, so the stream's last
      [candidate] or [stage] line still names it.
    - [stage] ({!tune}): [routine], [elapsed_s], [stage], [attempts], [compile_s], [timing_s],
      written BEFORE each step outside the candidates that can block for long: [base_compile],
      [cache_replay], [baseline_timing], [seed_enumeration] (the lowerings the seeds are derived
      from), [winner_compile], [untuned_default_compile], [untuned_control] (under [autotune_log] or
      a [?log] only). From {!Train.tune_placements}, [stage] alone: [placement_store] (the decision
      problem's lowering and replay check) and [flip_surface]. Together with [candidate] and
      [arm_start] this is the rule the stream keeps: every step that can block is named by a line
      written before it, so a killed search is inside the step its last line names.
    - [search_done] ({!tune}, exactly once per call, with its report): [routine], [elapsed_s],
      [outcome] ({!outcome_name}), [timed], [contended], [timings_retried], [unbatched], [failed],
      [rounds], [budget_cut], [baseline_refused] and [privatize_dropped] (the report's
      [privatize_drops] counts summed; gh-ocannl-1438), [attempts], [compile_s], [timing_s],
      [best_ms], [best].
    - [placements_start] ({!Train.tune_placements}, once, before anything is searched or replayed):
      [working_bytes], [device_free] and [device_total] (gh-ocannl-1424). The first is the
      {!Ir.Alloc_census} live working-pool bytes (process-wide, counted at the shared allocator
      seam, so deterministic on every backend), the other two the driver's reading of the device
      ({!Context.get_free_and_total_memory}; [none] where the backend has none).
    - [arm_start] / [arm_done] ({!Train.tune_placements}, around each arm's, flip's or replayed
      placement's search): [arm]; a flip's carry [flip=<k>/<budget>]; [arm_done] adds [result]
      ([ok], [failed] or [abandoned]), [best_ms] and [elapsed_s]. An abandoned flip's [best_ms] is
      its best at the verdict's depth, followed by that depth as [after] and the incumbent's best
      there as [incumbent_ms] (gh-ocannl-1110). Both end with [working_bytes] and [device_free],
      read as for [placements_start]: an arm that starts with more working bytes than the call did
      started next to something an earlier search left in OCANNL's pools, and a [device_free] lower
      than the call's at equal working bytes is device memory no pool table accounts for.
    - [flip_twin] ({!Train.tune_placements}, gh-ocannl-1274): a flip skipped without a budget slot
      as a structural twin of one that lost: [arm], and that flip's arm label as [twin_of]. No
      [arm_start] / [arm_done] frames it, since nothing is searched.
    - [flips_start] / [flips_done] ({!Train.tune_placements}): [candidates] and [budget];
      [flips_done] adds [measured], [pruned], [abandoned] (of the [measured]), [twinned] (skipped as
      twins, not among the [measured]), [improved] (whether the chain beat the A/B winner) and
      [best_ms].

    [elapsed_s] is seconds since that search (or arm) started; [compile_s] and [timing_s] are the
    parts of it spent in candidate compiles and in candidate timing windows, [attempts] the
    candidates attempted so far (bound-pruned, declined, deduplicated and timed alike), counting one
    whose fatal failure ended the search. *)

val progress_enabled : unit -> bool
(** Whether the progress lines are on in this domain: the [?progress] of the innermost enclosing
    {!tune} or {!Train.tune_placements} call that passed one, else config [autotune_progress]. *)

val log_enabled : unit -> bool
(** Whether the [autotune:] diagnostics (and the untuned-default control they gate) are on in this
    domain: the [?log] of the innermost enclosing {!tune} or {!Train.tune_placements} call that
    passed one, else config [autotune_log]. *)

val with_streams : ?log:bool -> ?progress:bool -> (unit -> 'a) -> 'a
(** [with_streams ?log ?progress f] runs [f] with {!log_enabled} and {!progress_enabled} overridden
    by whichever of the two is given, in the calling domain only, restoring the enclosing settings
    on return or exception. It is how {!tune} and {!Train.tune_placements} scope their arguments; an
    omitted argument leaves the enclosing setting in place. *)

val progressf : ('a, unit, string, unit) format4 -> 'a
(** Writes one progress line (prefix and [wall_s] added) when {!progress_enabled}; the format
    supplies [event=...] and the rest. *)

val progress_ms : float -> string
(** A time in milliseconds as progress lines print it: [%.4f], or [none] when not finite. *)

val progress_stopwatch : unit -> unit -> float
(** Starts a clock; the returned function reads the seconds since. *)

val outcome_name : outcome -> string
(** The stable one-word name of an outcome state — ["searched"], ["search-died"], ["cache-replay"],
    ["search-disabled"], ["pre-search-failure"], ["abandoned"] — for logs, JSON records and test
    goldens. *)

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

val envelope_legs :
  limits:Ir.Backend_intf.hardware_limits ->
  (float * [ `Config | `Backend ]) option * (float * [ `Config | `Backend ]) option
(** The envelope constants {!model_score} scores against — [(peak_flops, peak_memory_bandwidth)],
    FLOP/s and bytes/s — each with where it came from: [`Config] for a [model_peak_flops] /
    [model_peak_memory_bandwidth] override, [`Backend] for [limits]' advisory class constant. [None]
    for a leg neither provides (the C backends carry no class constant). Exposed so that a report
    scoring a measured kernel against the envelope reads the very constants the model ranks with,
    and can say whose they are (gh-ocannl-1006). *)

val ranking_capacity : limits:Ir.Backend_intf.hardware_limits -> int option * int option
(** The cache capacity (bytes) and device residency (threads) the ranking score
    ({!Ir.Cost_model.ranking_seconds}, gh-ocannl-1253) uses: config [model_cache_capacity] ([none]
    turns the capacity off) over the backend's advisory [cache_capacity_bytes], and its
    [max_resident_threads]. Both are [Untimed] readings of the device description: only the untimed
    choosers and the calibration tools call this, never anything {!tune} reaches. *)

val ranking_cache_bandwidth : limits:Ir.Backend_intf.hardware_limits -> float option
(** The ranking score's within-capacity tier, in bytes/s: config [model_cache_bandwidth] ([none]
    turns it off) over the backend's advisory [cache_bandwidth]. An [Untimed] reading, like
    {!ranking_capacity}. *)

val ranking_issue_rate : limits:Ir.Backend_intf.hardware_limits -> float option
(** The ranking score's scalar issue rate, in operations/s (gh-ocannl-1442): config
    [model_issue_rate] ([none]: scalar work priced at the envelope's [peak_flops]) over the
    backend's advisory [issue_rate]. An [Untimed] reading, like {!ranking_capacity}. *)

val ranking_dram_bandwidth : limits:Ir.Backend_intf.hardware_limits -> float option
(** The ranking score's achieved DRAM rate, in bytes/s (gh-ocannl-1442): config
    [model_dram_bandwidth] ([none]: DRAM traffic priced at the envelope's [peak_memory_bandwidth])
    over the backend's advisory [dram_bandwidth]. An [Untimed] reading, like {!ranking_capacity}. *)

val ranking_ms :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized list ->
  (float * float) option
(** [(rank, bound)] in milliseconds for scheduled kernels run in sequence: the ranking score
    {!model_default} minimizes and {!model_score}'s roofline lower bound, under {!envelope_legs}'
    constants and {!ranking_capacity}, pricing at the matrix units' rate only the tile-MMA work
    [capabilities]' hook accepts (none without them, gh-ocannl-1442). [None] without model coverage.
    For reports ([benchmarks/runners/ocannl/bench_wgrad.ml], [tools/calibrate_cache_capacity.exe]):
    a ranking score, not a runtime prediction. *)

val ranking_legs_ms :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized list ->
  (float option * float option) option
(** The ranking score's [(compute, memory)] legs in milliseconds, each summed over the kernels
    ({!Ir.Cost_model.ranking_legs_of}; [None] for a leg whose envelope constant is absent), under
    the same readings as {!ranking_ms} — the issue and DRAM rates included (gh-ocannl-1442). [None]
    where {!ranking_ms} is. For reports. *)

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

(** {2 Abandoning a hopeless flip (gh-ocannl-1110)}

    [Train.tune_placements]' flip chain runs a full {!tune} per tried flip, and on gh-719's
    approximate CUDA gpt2_mini cell two flips that ended 9-15x behind the arm they refined took 4747
    s of a 9145 s search. The rule that cuts them is a comparison at EQUAL DEPTH: a flip is
    abandoned once its best after its first [k] admitted timings trails the incumbent's best after
    the incumbent's own first [k] by more than [trailing_ratio]. Against the incumbent's FINAL best
    the same rule would abandon every flip, the profitable ones included: that cell's arm A itself
    sat at 80.6 ms — 11.7x its final 6.862 ms — for 207 of its 209 timed candidates, and only the
    last two (the recombination composites) brought it down. At equal depth the two flips trailed by
    2.0x and 1.7x, and arm B (the materialize-all specialization, the other direction a flip can
    take) by 1.07x. *)

exception Search_abandoned of abandonment
(** Raised by {!tune} when its [?abandon] rule decides, including from cached evidence. Its report
    ({!Abandoned} or {!Abandonment_replay}) has already been delivered, and every candidate this
    call compiled released. *)

type abandon_rule = {
  incumbent_steps : (int * float) list;  (** The incumbent search's {!report.best_steps}. *)
  trailing_ratio : float;  (** See {!flip_abandon_ratio}. *)
}

val best_after : (int * float) list -> int -> float
(** [best_after steps k]: the best after the first [k] admitted timings of the search whose
    {!report.best_steps} is [steps] — its final best when it timed fewer than [k], [infinity] when
    it timed nothing. *)

val abandon_verdict : abandon_rule -> k:int -> steps:(int * float) list -> abandonment option
(** [Some] exactly when [best_after steps k] exceeds [trailing_ratio] times the incumbent's
    [best_after] at the same [k] — strictly, so a flip trailing by exactly the ratio keeps searching
    — and the incumbent timed something by then. {!tune} decides once, as soon as it has
    [k = beam_width] admitted timings (the depth at which its first beam is full), before its next
    attempt. Exposed for tests. *)

val flip_abandon_ratio : ?margin:float -> unit -> float
(** The trailing ratio the flip chain abandons at: config [tune_flip_profit_margin], squared (1.5625
    at the default 1.25). The margin is the gap the chain already calls a measured loss for a
    family; squared, a flip must trail by that much once more than the rest of its search could be
    granted to catch up. That separates gh-719's hopeless flips (2.0x, 1.7x at equal depth) from arm
    B's 1.07x with room on both sides. [margin] overrides the configured one; a malformed configured
    value raises {!Utils.User_error} as {!flip_profit_margin_of_string} does. *)

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
(** Deduplicate (by [Tn.uid], keep-first), drop refused alternatives
    ({!Ir.Low_level.field-fa_refused}, gh-ocannl-1093: each replays to the materialized placement)
    and the nodes left with none, and rank the decision surface. [`Cost] is the legacy
    recompute-cost-descending order (the gh-555 chain's, kept as the evaluation baseline);
    [`Enablement] sorts family-unlocking [`Materialize] flips ([enablement] members) first and
    family-breaking [`Inline] flips (members of either set) last, cost-descending within each class.
    The unit ranked is the candidate (gh-ocannl-1017): each node's alternatives are sorted by that
    order and the node ranks where its best-ranked alternative does, so a top-N prefix is N nodes.
    [`Profitable] (gh-ocannl-579) is [`Enablement] weighed against [profit] per
    {!effective_flip_ordering} — the prior models expressibility, and on a device where the family
    it unlocks is measured hopeless, promotion is pure opportunity cost that displaces the winning
    flip out of a small budget. [profit] defaults to [Unmeasured] (so a caller with no measurements,
    such as {!model_default}, gets the prior). Config [tune_flip_ordering] selects the default
    ordering. Exposed for tests. *)

val placement_floor_withheld : Ir.Low_level.flip_candidate list -> bool
(** Whether {!placement_surface} withholds [ps_floor_ms] over these candidates: when one of them is
    not materialized by default ([fc_default]) and offers no [`Materialize] alternative — a node an
    earlier routine of the lineage left virtual and this routine footprint-scoped, which nothing
    here writes (gh-ocannl-616). The all-materialized specialization the floor is read from keeps
    that node's scratch, so the floor would not lower-bound its inline completion. Decided by the
    default reading (gh-ocannl-1017), not by which alternatives exist: a cap-materialized node with
    no smaller footprint offers the same lone [`Inline] and leaves the floor sound. Exposed for
    tests. *)

type placement_surface = {
  ps_candidates : Ir.Low_level.flip_candidate list;
      (** Deduplicated, ranked per {!rank_flip_candidates} under config [tune_flip_ordering]: one
          candidate per node, its alternatives in rank order, none of them refused. *)
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
          [None] when no envelope constant is present, or when {!placement_floor_withheld}. *)
  ps_twin_identity :
    node_tag:(Ir.Tnode.t -> string option) ->
    chain:(Ir.Tnode.t -> Ir.Low_level.reading option) ->
    Ir.Low_level.flip_candidate ->
    Ir.Low_level.flip_alternative ->
    string option;
      (** gh-ocannl-1274: the structural-twin identity of one alternative of a candidate on this
          surface, given the readings the flip chain has committed so far ([chain], [None] for an
          undecided node). [Train.tune_placements] skips an alternative whose identity matches one
          already measured and lost in the same chain, without a budget slot.

          What equal identities claim is a stated list, not "everything the flip's outcome depends
          on", which no local reading can be. The neighbourhood is every top-level statement of the
          raw program that sets or reads the flipped node, closed over the statement-level locals it
          reads (a top-level statement defining one joins it), rendered with fresh-name
          normalization ({!Ir.Schedule_cache.canonicalize_code}): per node its dims, precision and
          constancy, the placement the lineage brought, the placement the specialization decided,
          the chain's reading of it, the caller's [node_tag] (the flip chain tags its timing
          lineage, since the flips are measured there; [None] from it refuses, as for an inherited
          computation in that lineage), observability (an observable buffer is never aliased), the
          surface's enablement classes and the compile-time bounds candidate; then the flipped
          node's mark, its default and target readings, recompute cost and whether it is modeled.

          [None] — never a twin — where the rendering cannot be the flip's whole local reading: an
          incomplete rendering; a raw program holding an opaque construct anywhere (a
          [Staged_compilation] touches no node the neighbourhood filter can see); a flipped node no
          statement here assigns; a rendered node with a computation an earlier routine of the
          lineage stored ({!Context.inherits_computation}), whose replayed prefix the source does
          not show; a local read that no statement defines. Out by design: whole-routine context
          (where the neighbourhood sits, hence live ranges — per-layer twins differ in exactly
          that), and anything a farther node contributes except through the recompute cost. A false
          twin costs one unmeasured flip, never a wrong value. *)
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

val flip_twin_identity :
  static_indices:Ir.Indexing.static_symbol list ->
  lineage:Ir.Tnode.Placements.t ->
  inherited:(Ir.Tnode.t -> bool) ->
  enablement:Set.M(Ir.Tnode).t ->
  disablement:Set.M(Ir.Tnode).t ->
  Ir.Low_level.optimized ->
  node_tag:(Ir.Tnode.t -> string option) ->
  chain:(Ir.Tnode.t -> Ir.Low_level.reading option) ->
  Ir.Low_level.flip_candidate ->
  Ir.Low_level.flip_alternative ->
  string option
(** The rule behind {!placement_surface}'s [ps_twin_identity], over an already-lowered routine:
    [placement_surface] applies it to its decision lowering with the context's placements as
    [lineage], {!Context.inherits_computation} as [inherited] and the surface's enablement classes.
    Exposed so a test can reach the two refusals no [Assignments] lowering can isolate
    (test/operations/flip_twins.ml): a neighbourhood whose statement-level local is defined in a
    separate top-level statement, or by none -- shapes only hand-built raw code has -- and the
    inherited-computation refusal with everything else held fixed ([inherited] swapped on the same
    lowering). *)

type model_choice = {
  mc_label : string;
      (** ["default"], ["seeded"] (the seeded untuned plan, on CUDA and HIP), or the winning
          candidate's spec label (matching the [autotune_log] labels). *)
  mc_model_ms : float option;
      (** The applied winner's roofline lower bound in ms ({!Ir.Cost_model.analyze}'s counts) — not
          a runtime prediction; [None] when selection did not run. *)
  mc_rank_ms : float option;
      (** The winner's ranking score in ms ({!Ir.Cost_model.ranking_seconds}, gh-ocannl-1253), the
          quantity the pick minimizes; at least [mc_model_ms], and [None] exactly when it is. *)
  mc_scored : int;
      (** Model evaluations that produced a score (the default pipeline included; the fissioned flow
          also scores per segment). *)
  mc_skipped : int;  (** Model evaluations without coverage, excluded from the ranking. *)
  mc_rejected : int;
      (** Candidates excluded from the ranking because their scheduled form fails
          {!Ir.Low_level.validate_parallel}: the model cannot see that a schedule will not compile,
          and rates the tensorized families best, so on a backend where those are unbuildable it
          used to crown one and then degrade to the default (gh-ocannl-522). *)
  mc_excluded : int;
      (** Sketch candidates kept out of the ranking where the model starts from the seeded plan
          (CUDA and HIP, seeding on) because they are not numerics-preserving (gh-ocannl-1253:
          tensorized, conv, coalesced and the other forms {!model_default}'s restriction leaves out,
          or any sketch at a site accumulating wider than its storage); never scored, so neither in
          [mc_skipped] nor in [mc_rejected]. *)
}

val model_default_enabled : bool Lazy.t
(** Config [model_default_schedule]: recipe-level untuned compiles ({!Train.to_routine},
    {!Train.run_once}, the benchmark runners) route through {!model_default} instead of
    {!Context.compile}. *)

val compile_advisory :
  ?name:string ->
  ?on_fallback:(exn -> unit) ->
  ?fallback_if:(unit -> bool) ->
  ?on_unrecovered:(Ir.Schedule_outcome.classified_cause -> Context.t * Context.routine) ->
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
    instead of duplicating a compile that has nothing to fall back to; given [on_unrecovered], that
    classified cause is handed to it instead of raised — as is a classified failure of the fallback
    compile itself — so a caller with a fallback of its own acts on a classified rejection alone,
    never on a fatal failure (gh-ocannl-1390). For advisory transforms only: a failure of the
    default pipeline itself propagates. [name] names the routine, exactly as {!Context.compile}'s
    does, and reaches the fallback compile too. See {!model_default} (gh-ocannl-519). *)

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
    the cost model's ranking score ({!Ir.Cost_model.ranking_seconds}, gh-ocannl-1253: the roofline,
    with operand re-reads requested per block and charged at DRAM bandwidth once the working set
    exceeds the cache capacity, at the cache bandwidth within it — {!ranking_capacity},
    {!ranking_cache_bandwidth}), and the argmin schedule is applied — zero measurement, one backend
    compile. The family search's bound stays the distinct-cell floor, below every ranking score.
    Sketch leaves whose scores tie exactly are ordered by their tile-MMA operand loads under the
    backend's [mma_operand_loads] capability ({!model_family_search}, gh-ocannl-1254). Advisory by
    construction: a candidate without model coverage is never picked over the default, ties go to
    the default, and missing envelope constants, a disabled default annotator
    ({!Ir.Schedule.automatic_schedule_active}), or any classified scoring, application, backend
    validation, or compilation failure fall back to the ordinary default pipeline — the reported
    {!model_choice} then says ["default"]. On CUDA and HIP with the seeding on (as for
    {!seeded_default}), the model starts from the seeded untuned plan instead (gh-ocannl-1253,
    {!seeded_plan} under the configured gate, [auto] included): it is the incumbent every candidate
    must strictly beat and the pick when none does ([mc_label] ["seeded"], even where no segment
    seeds and the plan is the default pipeline -- a plan the compile then does not retry on a
    backend failure, as for the default), the candidates compete on the program after the plan's
    split prelude, and the sketch candidates are restricted to the seeding's numerics-preserving
    forms — scalar block tiles at a site accumulating at its storage precision — so the pick never
    changes a value the untuned compile computes. Fatal failures propagate without retrying. Once
    the compile is on that pipeline there is nothing left to fall back to, so its failures propagate
    as they would from {!Context.compile}, without a duplicate attempt. Unlike {!tune}, nothing is
    executed and no cache is involved — results depend only on the computation, backend, and
    envelope constants.

    Being a drop-in for {!Context.compile} includes its [name] (gh-ocannl-669): it names the routine
    and every artifact of this compile — the model pick's, the fallback's, and the hermetic
    lowerings of the placement search alike — and, as there, a comp carrying no
    {!Ir.Assignments.Block_comment} requires it. *)

val model_operand_loads :
  capabilities:Ir.Backend_intf.codegen_capabilities ->
  Ir.Low_level.optimized ->
  Ir.Cost_model.load_count
(** {!Ir.Cost_model.operand_load_count} under [capabilities]' [mma_operand_loads] rule and
    [mma_operand_fragments] structure, for the statements its [mma_accepts] hook probe accepts: the
    count {!model_family_search} breaks exact ties with and {!model_explain} reports
    (gh-ocannl-1254): [Undescribed] where the backend describes no arm for some statement (every
    backend but CUDA). The walk keys only [Loads]. *)

val model_family_search :
  ?on_rejected:(unit -> unit) ->
  ?admit:(sketch_params -> bool) ->
  ?on_excluded:(unit -> unit) ->
  capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  static_indices:Ir.Indexing.static_symbol list ->
  peak_flops:float option ->
  peak_memory_bandwidth:float option ->
  score:(Ir.Low_level.optimized list -> float option) ->
  incumbent:float ->
  Ir.Low_level.optimized ->
  ((sketch_params * float) option * Ir.Schedule_space.search_stats * float option) option
(** {!model_default}'s branch-and-bound walk over the factored family of one lowering
    ({!sketch_family_tree}: the matmul family, or since gh-ocannl-697 the conv family):
    [(best, stats, bound)], where [best] is the first leaf whose [score] (of its scheduled form) is
    strictly below [incumbent], the threshold tightening as leaves improve on it, and [bound] the
    family's schedule-invariant roofline floor under the envelope. Exact ties between leaves go to
    fewer tile-MMA operand load instructions ({!model_operand_loads} under [capabilities] —
    {!model_default} passes the backend's; gh-ocannl-1254) between leaves that both have tile-MMA
    sites, then to the first; a tie never displaces [incumbent]. Sketches are scheduled at
    [capabilities]' [accum_prec]. [on_rejected] is called for each leaf whose schedule fails to
    apply; a leaf [admit] refuses is left unscored, [on_excluded] called, before it is scheduled
    ({!model_default}'s numerics-preserving restriction). [None] for a lowering neither family
    claims. A conv path carries no certain-traffic increment, so its bound is the schedule-invariant
    floor, sound for every completion. Exposed so a test can drive the walk with stand-in limits and
    load rules on any backend. *)

type seeded_choice = {
  sd_seeded : (int * sketch_params) list;
      (** Per seeded segment, in order: its first unit and the parameters seeded into it. Empty when
          nothing was seeded. *)
  sd_sunk : (int * int) list;
      (** Per unseeded segment whose accumulations' reduction loops were sunk below their write
          loops ({!reduction_sink}, gh-ocannl-1348), in order: its first unit and the number of
          [Swap]s. *)
  sd_refined : int;
      (** Default segments replaced by finer [arity_cuts] segments ({!refine_segments},
          gh-ocannl-1413): every default segment of each block whose finer segments seed more than
          its default ones -- an unseeded segment one of whose pieces seeds, a seeded one whose
          other product sites seed once cut apart, or a run of segments whose finer cuts move a
          boundary. *)
  sd_split : (Ir.Tnode.t * int) list;
      (** The reductions split before segmentation (gh-ocannl-1325), in detection order: each
          accumulated node and its block count. Empty when nothing was split. *)
  sd_fell_back : bool;
      (** The seeded pipeline failed to compile with a classified rejection and the routine was
          recompiled on the default pipeline. *)
}
(** What {!seeded_default} did to one compile. *)

val site_reduction_extent : matmul_site -> int
(** A matmul site's reduction extent: its innermost contraction loop's extent times every enclosing
    contraction loop's — for a backprop weight gradient, the batch-and-sequence product. *)

val seeded_params :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  limits:Ir.Backend_intf.hardware_limits ->
  accum_prec:(Ir.Ops.prec -> Ir.Ops.prec) ->
  min_reduction:int ->
  Ir.Low_level.optimized ->
  sketch_params option
(** The parameters {!seeded_default} seeds into one fission segment (gh-ocannl-1324): [None] unless
    the segment is a matmul site ({!detect_matmul_canonical}, so backprop's contractions count) with
    a {!site_reduction_extent} of at least [min_reduction], and whose output's storage precision is
    its own accumulator precision under [accum_prec] (the backend's resolution): the sketch keeps a
    wider accumulator across the whole reduction where the default rounds it back per contraction
    step, so a narrow-storage site stays on the default rather than change its result. Otherwise one
    leaf of the segment's own GPU {!matmul_sketch_tree} — so every refutation the search applies
    applies here — restricted to the unfused scalar blocktile pipeline without twin flavors: batch
    loops as grid blocks on a batched site where the device admits them, and the first geometry of a
    fixed preference list (64x64x8/4x4, 32x32x8/4x4, 32x32x16/2x2, 16x16x8/2x2, 16x16x8/4x4) whose
    launch has 128 blocks, else the one putting the most threads on the output — among the
    geometries padding the output by at most an eighth, where there are any. [capabilities] as for
    {!matmul_sketch_tree}: the compiling context's {!Context.codegen_capabilities}, so the tree's
    workgroup-memory refutation charges the staged tiles exactly as the gate judging the seeded
    kernel does; every caller holding a context passes them, the seeded plan's decision rule
    included. Exposed for tests. *)

val seeded_params_judged :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  limits:Ir.Backend_intf.hardware_limits ->
  accum_prec:(Ir.Ops.prec -> Ir.Ops.prec) ->
  min_reduction:int ->
  Ir.Low_level.optimized ->
  (sketch_params, string) Result.t
(** {!seeded_params} with the refusing gate named (gh-ocannl-1407): [Ok] exactly where
    {!seeded_params} is [Some]. *)

val seeded_default_backend : string -> bool
(** Whether {!seeded_default} seeds on this backend ({!Context.backend_name}): CUDA and HIP. *)

val seeded_split_backend : string -> bool
(** Whether {!seeded_default} splits long reductions on this backend: wherever it seeds, CUDA and
    HIP (see {!seeded_split_sites}). A test states what each backend promises through these, never a
    restated list. *)

val search_shaping_key_read : backend:string -> string -> bool
(** Whether a search on [backend] can read the [Utils.Search_shaping] configuration key [key]
    (gh-ocannl-1331): the seeded plan's [gpu_default_*] keys where {!seeded_default_backend} (and,
    for the split threshold, {!seeded_split_backend}) holds, [autotune_register_tile_rm_twin] where
    [Ir.Schedule.backend_is_cpu] does (only the CPU register-tiled rendering reads it), the default
    pipeline's keys per {!Ir.Schedule.search_shaping_key_read}, and every other key everywhere. A
    key a backend's search cannot read keys nothing in its trajectory identity
    ({!search_shaping_identity}). *)

val search_shaping_identity : backend:string -> limits:Ir.Backend_intf.hardware_limits -> string
(** The [Search_shaping] part of {!tune}'s search-trajectory identity on [backend] at [limits]:
    [Utils.config_class_fingerprint] of the class, over the keys {!search_shaping_key_read} says the
    backend reads, with defaults resolved per device ({!Ir.Schedule.resolve_search_shaping_value}).
*)

val seeded_split_sites :
  backend:string ->
  min_serial:int ->
  static_indices:Ir.Indexing.static_symbol list ->
  Ir.Low_level.optimized ->
  (sr_site * int) list
(** The reductions {!seeded_default} splits (gh-ocannl-1325), each with its block count: the
    {!split_reduce_sites} (the split's own legality and enabling interchange, as the search proposes
    them) whose serial chain per output cell -- [sr_cost / sr_out] -- is at least [min_serial] ([0]:
    none), scatter sites excepted; each at the block count of the search's menu ([32; 128; 512], at
    most half the split loop's extent and at most the square root of the chain, gh-ocannl-1414)
    giving the most first-pass threads up to 32768, else the smallest such; a chain too short for
    every count is not split. The choice reads no device property, so the split, and the summation
    order it fixes, depend on the routine alone. Empty where {!seeded_split_backend} is false
    ([backend] is the {!Context.backend_name}). The split was CUDA's alone until its first pass ran
    the block loop outside the cell loops ([sr_block_hoist], gh-ocannl-1412): before that, HIP's
    gpt2_mini_train b256 step measured 1.42x SLOWER split (885 to 1256 ms, gfx1151 unified memory)
    where CUDA gained 1.39x. Exposed for tests. *)

(** Which segments {!seeded_default} seeds (gh-ocannl-1253). *)
type seed_gate =
  | Min_reduction of int
      (** gh-ocannl-1324's threshold: a site whose {!site_reduction_extent} is at least this; [0]
          disables the seeding. *)
  | Ranked of {
      capacity : int option;
      cache_rate : float option;
      scalar_rate : float option;
      dram_rate : float option;
    }
      (** The [auto] gate: a segment whose {!seeded_params} sketch the ranking score
          ({!Ir.Cost_model.ranking_seconds}) puts strictly below the default preset of the same
          segment, scored alone with this cache capacity (bytes; [None]: no DRAM tier) and cache
          bandwidth (bytes/s; [None]: re-reads within capacity uncharged), and the score's own
          scalar issue rate and DRAM rate ({!ranking_issue_rate}, {!ranking_dram_bandwidth}; [None]:
          the envelope's ceilings, gh-ocannl-1442) — so it seeds exactly where the cross-block
          re-reads the sketch removes are priced. The geometry and the numerics-preserving candidate
          set are {!seeded_params}' either way. *)

val seeded_default_gate : limits:Ir.Backend_intf.hardware_limits -> seed_gate
(** Config [gpu_default_sketch_min_reduction]: an integer is [Min_reduction]; [auto] is [Ranked]
    under {!ranking_capacity}'s capacity and {!ranking_cache_bandwidth}, and [Min_reduction 32] —
    the threshold's measured default — where nothing calibrates either tier: no advisory (Metal, the
    C backends) and no [model_cache_capacity] / [model_cache_bandwidth] override. An explicit [none]
    override is a calibration and keeps [Ranked], even with both tiers off. Raises
    [Utils.Invalid_config_value] naming the key on anything else (gh-ocannl-1332). *)

(** What the seeded plan does with one segment (gh-ocannl-1388). *)
type seeded_segment =
  | Seeded_preset  (** The default preset, as the default pipeline schedules the segment. *)
  | Seeded_sketch of { digest : string; params : sketch_params }
      (** The scalar blocktile sketch [params] ({!seeded_params}), for the segment whose structural
          digest ([Schedule_cache.canonicalize ~with_placements:false]) is [digest]. *)
  | Seeded_sunk of { digest : string; swaps : int }
      (** {!reduction_sink}'s [swaps] interchanges, then the default preset, likewise matched. *)

type seeded_start = {
  decide : Ir.Low_level.optimized -> sketch_params option;
      (** The plan's per-segment decision: the sketch its gate admits on a segment, probed and
          validated there, or [None] for the preset. *)
  gate_tag : string;  (** The gate it reads, rendered for the search shape. *)
}
(** The seeded plan's decision rule, for {!tune}'s staffed start (gh-ocannl-1138, config
    [autotune_staffed_singles]), which applies it to the search's own segments: the plan's recorded
    decisions are taken on the split routine under the untuned segmentation and match no search key.
    Carried in the plan so that the device readings the gate rests on reach {!tune} only through
    {!seeded_plan}. *)

type seeded_plan = {
  sp_split : (sr_site * int) list;
      (** The reductions split whole-routine before segmenting ({!seeded_split_sites}). *)
  sp_segments : (Ir.Schedule.segmentation * seeded_segment list) option;
      (** The segmentation of the split routine and one decision per segment, by position; [None]:
          the default pipeline's own segments. *)
  sp_refined : int;
      (** Default segments replaced by their finer [arity_cuts] pieces ({!refine_segments}: a block
          takes its pieces where more of them seed). *)
  sp_start : seeded_start option;
      (** The plan's decision rule, as {!seeded_plan} derives it; [None] in the plans the untuned
          compile builds for itself. *)
}
(** The seeded untuned plan as a replayable value (gh-ocannl-1388): what {!seeded_default} decides
    for one lowering. {!tune} derives it on its base lowering and seeds it as its FIRST candidate,
    replayed the way the untuned compile replays it, so the step the recipes ship untuned is the
    search's floor by construction (where {!untuned_compile} is {!seeded_default}, i.e. without
    [model_default_schedule=true]); gh-ocannl-1138 extends it from there. *)

val seeded_plan :
  ?min_reduction:int ->
  ?seed_gate:seed_gate ->
  ?min_serial:int ->
  ?reduction_sink:bool ->
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  backend:string ->
  limits:Ir.Backend_intf.hardware_limits ->
  accum_prec:(Ir.Ops.prec -> Ir.Ops.prec) ->
  static_indices:Ir.Indexing.static_symbol list ->
  Ir.Low_level.optimized ->
  seeded_plan option
(** The plan {!seeded_default} would compile [opt] under, derived on a hermetic copy; [None] where
    that is the plain default pipeline (another backend, scheduling or fission off, the thresholds
    at 0, or nothing to split, seed or sink). The optional arguments are {!seeded_default}'s, the
    [auto] gate included: {!tune} seeds this plan first, and the plan's {!seeded_plan_digest} joins
    the search trajectory's identity, so a gate steered by the device description's [Untimed]
    capacity reaches the search only as the plan it resolved to (gh-ocannl-1253). [capabilities] are
    the compiling context's, which {!seeded_params} prices workgroup memory by — in the plan's
    segments and in its decision rule; {!tune} passes its context's, as the untuned compile does. *)

val seeded_plan_digest : seeded_plan -> string
(** The plan's identity from its content (split sites by target name, extents and block count; the
    segmentation; each segment's decision and the structural digest it was decided on), alike in
    every process deriving it from the same lowering under the same settings. *)

val seeded_plan_label : seeded_plan -> string
(** The search's label for the plan's seed:
    [F_seeded[<n> segs, <k> seeded, <s> sunk, <p> split, plan <digest prefix>]]. *)

val refine_segments :
  coarse:(int * 'k * int * 'p) list ->
  fine:(int * 'k * int * 'p) list option Lazy.t ->
  seed:('k -> 'p -> 's option) ->
  (int * 'k * int * 's option * 'p) list * int
(** The site refinement of {!seeded_default} (gh-ocannl-1413): two segmentations of one routine,
    given as [(first unit, kind, units, pre-schedule slice)] per segment, are compared block by
    block, a block being a maximal run of units whose ends are segment boundaries in both. A block
    takes its [fine] segments when more of them [seed] than of its [coarse] ones, and keeps its
    coarse segments otherwise. Returns the result with each segment's seed, and the number of coarse
    segments replaced. *)

val seeded_transform :
  ?min_reduction:int ->
  ?seed_gate:seed_gate ->
  ?min_serial:int ->
  ?reduction_sink:bool ->
  ?on_choice:(seeded_choice -> unit) ->
  Context.t ->
  Ir.Indexing.unit_bindings ->
  (Ir.Low_level.optimized -> Ir.Low_level.optimized list) option
(** The [lowered_transform] {!seeded_default} compiles through, for a caller that drives
    {!Context.compile_outcome} itself (a benchmark keeping its own failure classification): [None]
    where {!seeded_default} would compile the plain default (another backend, automatic scheduling
    inactive, fission off, a threshold of 0 with the reduction sink off). The transform calls
    [on_choice] with what it seeded and sank, and falls back to the default segments itself when the
    seeded pipeline does not apply; a failure in backend compilation is the caller's, as
    {!seeded_default}'s fallback is not part of it. *)

val seeded_default :
  ?name:string ->
  ?min_reduction:int ->
  ?seed_gate:seed_gate ->
  ?min_serial:int ->
  ?reduction_sink:bool ->
  ?report:(seeded_choice -> unit) ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Context.t * Context.routine
(** A drop-in for {!Context.compile} that gives the untuned GPU default operand reuse on its large
    contractions (gh-ocannl-1324), without timing or scoring anything: the default pipeline's own
    fission segmentation, with each segment {!seeded_params} admits scheduled by that sketch instead
    of the default preset. The default segmentation is compared with the finer [arity_cuts] one
    ({!Ir.Schedule.fission_segmented}) block by block ({!refine_segments}, gh-ocannl-1413): where a
    run of default segments, between boundaries the two share, holds fewer seeding segments than the
    finer segments covering the same units, the finer ones replace it — a weight gradient cut away
    from the bias gradient over the same rows, a projection's data gradients cut away from its
    weight gradient, attention's score product cut away from the row maximum's initialization (a
    moved boundary). Such a replacement can change the launch count; every segment outside those
    runs is exactly the default's. On CUDA and HIP only, with automatic scheduling active and
    fission on; [seed_gate] (default: {!seeded_default_gate}, config
    [gpu_default_sketch_min_reduction], [auto]: the ranked gate, or 32 on a device publishing no
    cache constants) decides which segments seed, and [min_reduction] is
    [~seed_gate:(Min_reduction n)] (passing both is an [Invalid_argument]). Every seeded segment is
    probed alone first (the sketch must apply and pass {!Ir.Low_level.validate_parallel}); a segment
    that fails keeps the preset, and a classified failure of the whole seeded compile recompiles the
    default pipeline ({!compile_advisory}). The scalar blocktile pipeline keeps each output's
    summation order, and only sites whose accumulator is at their storage precision are seeded, so
    the seeded sketches leave the routine's results bitwise the default's.

    Before segmenting (on CUDA and HIP), the reductions {!seeded_split_sites} admits at [min_serial]
    (default: config [gpu_default_split_reduce_min_serial], 1024; 0 disables) are split
    whole-routine, as the search's [F_split] seeds split them (gh-ocannl-1325): a bias or layer-norm
    parameter gradient, which the default runs as one thread per output cell walking every batch row
    in sequence, becomes a parallel pass over blocks of rows and a combine. That DOES change the
    summation order of those reductions -- a deterministic balanced tree over the blocks, chosen
    from the routine alone -- as the default plan's cooperative all-reduces (gh-ocannl-1124) do.

    Every [`Normal] segment no sketch takes gets {!reduction_sink}'s interchange, then the default
    preset of the interchanged code (gh-ocannl-1348), probed alone like a seed, and covered by the
    same fallback to the default pipeline when the compile fails classified: a batch reduction into
    a bias or layer-norm parameter gradient then keeps its accumulator in a register instead of
    reading and writing its device cell every step under the preset's in-loop guard — which HIP's
    compiler does not promote, though NVRTC does. Values stay bitwise the default's.
    [reduction_sink] (default: config [gpu_default_reduction_sink], true) turns it off; it applies
    with the seeding disabled too. Reads and writes no schedule cache. *)

type explain = {
  ex_floor_ms : float option;
      (** The candidate's roofline lower bound ({!Ir.Cost_model.analyze}'s counts), as
          [mc_model_ms]. *)
  ex_rank_ms : float option;
      (** Its ranking score ({!Ir.Cost_model.ranking_seconds}), the quantity {!model_default}
          minimizes: per kernel the larger of the two legs below, summed. *)
  ex_compute_ms : float option;  (** The compute leg, summed over the kernels. *)
  ex_memory_ms : float option;
      (** The two-tier memory leg ({!Ir.Cost_model.ranking_legs_of}), summed over the kernels. *)
  ex_spilled : bool option;
      (** Some kernel's working set exceeds the cache capacity; [None] where the ranking score was
          not computed (no envelope constants, or an opaque kernel). *)
  ex_operand_loads : Ir.Cost_model.load_count;
      (** Tile-MMA operand load instructions under the backend's load rule ({!model_operand_loads}),
          over all the candidate's kernels ({!Ir.Cost_model.combine_load_counts}: [Undescribed] if
          any kernel is): not a price, the count {!model_default}'s family search breaks exact score
          ties with (gh-ocannl-1254). *)
  ex_admitted : bool;
      (** Whether {!model_default}'s numerics-preserving restriction (in force where it starts from
          the seeded plan, CUDA and HIP) admits the candidate to its ranking; the incumbent always
          is. *)
}
(** All times in ms; the score fields are [None] without envelope constants or model coverage. *)

type segment_explain = {
  xs_segment : string;
      (** The segment's key tag ([<digest prefix>:u<first units>]), as {!tune}'s per-segment sketch
          labels ([F_sketch[...]]) and {!model_default}'s pick name it. *)
  xs_candidates : (string * explain) list;
      (** ["preset"] first -- the segment under its default preset, the incumbent its sketches must
          strictly beat -- then every sketch leaf {!model_default}'s family search visits on the
          segment, in visiting order, each labeled [<sketch parameters> @<key tag>] as an [F_sketch]
          label renders its entries ({!sketch_params_label}). *)
}

type explanation = {
  x_routine : (string * explain) list;
      (** The whole-routine candidates: the incumbent first -- ["seeded"] where {!model_default}
          starts from the seeded plan (CUDA and HIP under the seeded settings), else ["default"],
          the untuned default pipeline -- then every whole-routine sketch leaf of the program it
          searches (the seeded plan's split prelude included), labeled as {!tune}'s [autotune_log]
          labels them ([W_sketch[...]]), in the family search's visiting order. *)
  x_segments : segment_explain list;
      (** One entry per keyed segment of the per-segment substitution {!model_default} ranks beside
          the whole-routine candidates (gh-ocannl-1438), in segment order: the fold-zeros fission
          segmentation of the searched program, keyed as {!tune}'s per-segment candidates key it.
          Empty where {!model_default} makes no per-segment substitution: fission splits nothing and
          exposes no changed lowering, or the segmentation declines. *)
}

val model_explain :
  ?name:string -> Context.t -> Ir.Assignments.comp -> Ir.Indexing.unit_bindings -> explanation
(** What {!model_default} sees of the candidates it ranks for one computation, without choosing or
    compiling (gh-ocannl-1254, the interface gh-ocannl-1269 reads): the whole-routine candidates
    and, per segment, the per-segment ones (gh-ocannl-1438), each with whether the numerics
    restriction admits it. The lowering is {!Context.lowered_for_decisions}', so the context is
    untouched. The placement pre-search is not listed. Candidates whose scheduled form fails to
    apply or validate are omitted, as {!model_default} excludes them; a segment whose preset has no
    score is listed with its unscored entries, where {!model_default} skips it. Where
    {!model_default} ranks nothing — no envelope constants, or the default annotator off —
    [x_routine] is the default pipeline alone and [x_segments] is empty. *)

val untuned_compile :
  ?name:string ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Context.t * Context.routine
(** The recipes' untuned compile ({!Train.to_routine}, {!Train.run_once} and the benchmark runners'
    untuned arms): {!model_default} under config [model_default_schedule=true], {!seeded_default}
    otherwise. *)

val set_test_bindings : Context.routine -> unit
(** Binds representative values for timing runs: ranged static indices at [range / 2], and gh-490
    symbolic extents at their upper bound [range] (the schedule-cache identity is
    extent-value-independent, so the single tuned entry is measured at the maximum). Unranged
    bindings are left at their current values. Exposed for tests and custom timing harnesses. *)

val queued_batch_ms : float
(** The wall time, in milliseconds, one {!Queued} batch aims for (10 ms). Exposed so a test can
    state a batch-wall bound against the policy's own target rather than a copy of it. *)

val queue_calibration_wall_ms : float
(** The wall budget, in milliseconds, on one CUDA/HIP queued calibration's batch probes
    (gh-ocannl-1098): eight target-sized twelve-minimum probes. Once the probes' summed wall reaches
    it, no further validation, confirmation or retry starts and the calibration ends unresolved, for
    the wall-bounded fallback to decide from what was measured; only the one rescue probe may
    follow. Exposed so a test states the budget against the policy's own value. *)

val queue_batch_probe_runs : int
(** The minima one queued calibration batch probe takes when its wall stays within twice the target
    (12); an over-target probe may stop at three. Exposed so a dispatch bound is stated against the
    policy's own count. *)

val queue_calibration_max_probes : int
(** The most batch probes one CUDA/HIP queued calibration dispatches (9): the provisional probe,
    four validations, a confirmation, its stall retry, a sampled shallower crossing, and the rescue.
    Enforced alongside {!queue_calibration_wall_ms}: once all but the last have started, only the
    rescue may follow. Exposed so {!time_routine}'s dispatch maximum is stated against it. *)

val max_depth_validation_probes : int
(** The maximum number of queued depth validations (4), including measured doublings toward an
    unmeasured cap. Exposed so synthetic timing fixtures hold convergence against its probe budget.
*)

val queue_depth_projection_factor : int
(** How far past the deepest batch it probed a CUDA/HIP queued calibration may settle, as a multiple
    of that depth (2, one doubling; gh-ocannl-1100). An affine projection or a linear scale can land
    far past every measured batch, where a queue cost may jump unseen; the settled depth is capped
    here after calibration ends. A projection to an unmeasured cap in depth validation uses its
    remaining probes to walk toward it by this factor (gh-ocannl-1144). Exposed so a test states the
    bound against the policy's own factor. *)

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
    2048-launch memory cap, which {!calibrate_and_time} then bounds by the deepest depth measured
    within the target (gh-ocannl-1096). Exposed as the deterministic policy seam for tests. *)

val refine_queued_batch_depth_between :
  base_depth:int -> base_ms:float -> probe_depth:int -> probe_ms:float -> int * float
(** The same affine refinement between two batch observations. An already-target-sized [base_ms] is
    retained only when the pair's inferred fixed component is below the target and no more negative
    than a quarter of the larger of the target and [base_ms] (the noise tolerance matching the 25%
    confirmation step, relative to the base's wall since gh-ocannl-1098 so that a slow candidate's
    slightly superlinear pair still fits), and only when the base is the fit's first target
    crossing; a resolved base further over the target projects to that crossing instead, shallower
    than the base. This is the depth-separated confirmation that keeps a shared fixed stall or a
    physically invalid fit from selecting a shallow final depth while preserving legitimate
    submit/sync overhead. Otherwise an unresolved pair requests a deeper retry and reports a [nan]
    wall. Exposed as the deterministic validation-policy seam for tests. *)

val rescue_depth : observed:(int * float) list -> int option
(** The depth of the one rescue probe a CUDA/HIP calibration makes before refusing a candidate whose
    single launch owed it a batch but whose calibration settled at depth 1 (gh-ocannl-1098), from
    every [(depth, minimum wall)] it measured: the shallowest batch over the target, projected
    linearly down to the target, at least 2 and strictly shallower than that batch; [None] when no
    depth lies between. Exposed so a test can tell the rescue from the probes before it. *)

val sample_min : repeats:int -> sample:(unit -> timing_sample) -> timing_result
(** Pure sampling-policy seam used by calibration and the timed loop (gh-ocannl-855). Takes at least
    [max 16 repeats] samples; after that floor is met, tops up until their accumulated
    [per_launch_ms] reaches ~25 ms or 64 samples have been taken. Thus [repeats > 64] deliberately
    overrides the top-up limit. Reports [contended] when at least half the raw [contention_ms]
    samples exceed their minimum by 2x — dispersion only; a minimum that is non-positive or
    non-finite is refused by {!admitted_timing_ms} instead. Exposed so tests can inject a
    deterministic clock. Equal to [window_result (sample_window ~repeats ~sample ())]. *)

val sample_window :
  ?prior:timing_sample list ->
  repeats:int ->
  sample:(unit -> timing_sample) ->
  unit ->
  timing_sample list
(** {!sample_min}'s loop, returning the window itself, oldest first. [prior] (default empty) is a
    window already taken, which the loop resumes under the same stopping rule rather than restarting
    (gh-ocannl-1074): a resumed window stops exactly where one uninterrupted window over the same
    samples would, so it takes no fresh sample when [prior] already meets the floor and the top-up
    budget. *)

val window_result : timing_sample list -> timing_result
(** A window's reading: its minimum [per_launch_ms], its sample count, and the dispersion-only
    [contended] verdict {!sample_min} documents. *)

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

(** The branch of a CUDA/HIP {!Queued} calibration that started a batch probe. The longest path
    {!queue_calibration_max_probes} counts is one provisional probe, four validations, a
    confirmation, its stall retry, a sampled shallower crossing and the rescue. *)
type calibration_probe_role =
  | Provisional_probe  (** At the depth the synchronized singles' estimate provisionally picks. *)
  | Validation_probe  (** At a depth an affine fit projected, checked against its base. *)
  | Confirmation_probe
      (** One step deeper than a target-sized batch, the depth-separated confirmation. *)
  | Stall_retry_probe
      (** The confirmation repeated once at the same depth, after it read as a stall. *)
  | Crossing_probe  (** At a fit's target crossing, below a measured over-target batch. *)
  | Rescue_probe
      (** At {!rescue_depth}, before refusing a candidate that measured no batch within the target;
          charged to no budget. *)

type calibration_probe = {
  role : calibration_probe_role;
  depth : int;
  runs : int;  (** The minima taken: {!queue_batch_probe_runs}, or as few as three over-target. *)
  wall_ms : float;
      (** The probe's summed wall over its finite positive batches: what it charged to
          {!queue_calibration_wall_ms}. *)
  min_ms : float;  (** The minimum batch wall, the probe's reading. *)
}
(** One batch probe of a calibration, as it ended. *)

(** What one affine fit of a CUDA/HIP {!Queued} calibration concluded about its pair of batch
    minima: which branch of the refinement produced the depth it returned. *)
type fit_verdict =
  | Fit_unordered  (** The probe was not deeper than the base: no slope, a doubled retry. *)
  | Fit_unreadable  (** A reading was not finite and positive: a doubled retry. *)
  | Fit_nonpositive_marginal  (** The slope was not positive: unresolved, a doubled retry. *)
  | Fit_negative_fixed
      (** The fixed term was more negative than the noise tolerance allows: unresolved. *)
  | Fit_fixed_dominated
      (** The fixed term alone fills the target: a depth carrying one target of marginal work. *)
  | Fit_base_crossing  (** The base is the fit's first target crossing and is kept. *)
  | Fit_boundary_floor
      (** The crossing is shallower than the measured depth-2 base, which is kept anyway: singles
          owed a batch, the fixed term is below the target and the marginal work fits it
          (gh-ocannl-1184). *)
  | Fit_shallower_crossing  (** The crossing, shallower than an over-target base. *)
  | Fit_deeper_crossing
      (** The crossing, deeper than an under-target base (an unmeasured depth). *)

type affine_fit = {
  base_depth : int;
  probe_depth : int;
  fixed_ms : float;  (** The fitted synchronization term; [nan] when no slope was resolved. *)
  marginal_ms : float;  (** The fitted per-launch term; [nan] when the readings gave no slope. *)
  verdict : fit_verdict;
}
(** One affine fit of a calibration, as the refinement computed it. *)

type batch_decision = {
  phase : string option;
      (** The caller's [phase]: for a {!tune} candidate, the search phase as its [autotune_progress]
          lines name it ([seeds], [recombine], [playoff] for the composite playoff of
          gh-ocannl-1166, or [round<k>]); [None] for the baseline, the untuned control and any other
          caller. Carried by the call, not read from shared state, so concurrent searches never
          label each other's decisions. *)
  timing : timing_mode;
  depth : int;  (** The depth the call settled on and its timed window ran at. *)
  cap : int;  (** The call's [queue_depth_cap]. *)
  target_ms : float;  (** {!queued_batch_ms}. *)
  estimated_wall_ms : float option;
      (** The CUDA/HIP calibration's whole-batch wall estimate at [depth], [nan] when unresolved;
          [None] under {!Isolated} and the cc/Metal policy. *)
  settle : batch_settle;
  fit : affine_fit option;  (** The calibration's latest affine fit, if it made one. *)
  cap_directed_probes : int;
      (** Validation probes whose depth a fit wanting the unmeasured cap directed: measured
          doublings toward the cap (gh-ocannl-1144), as against ordinary affine projections. *)
  bounded_from : int option;
      (** The depth the exit chose, when {!queue_depth_projection_factor} times the deepest probe
          bounded it (gh-ocannl-1100). *)
  fallback_from : int option;
      (** The depth before the wall-bounded fallback shortened an unresolved outcome
          (gh-ocannl-1096). *)
  rescue : (int * float) option;
      (** The rescue probe's depth and minimum, when one ran (gh-ocannl-1098); [depth] equals it
          exactly when it read within the target. *)
  probes : int;  (** Batch probes started, against {!queue_calibration_max_probes}. *)
  probe_wall_ms : float;  (** Their summed wall, against {!queue_calibration_wall_ms}. *)
  budget_spent : bool;  (** Whether that budget was spent when the calibration ended. *)
  admission : batch_admission;
  retried : bool;  (** Whether a contended first window was discarded for one fresh retry. *)
}
(** One timing call's depth decision and admission verdict, from the policy's own metadata. *)

type timed_window = {
  samples : int;
      (** The window's batches, counted by the timed loop itself rather than restated from its
          result, so a test can hold the two against each other. *)
  reused : int;
      (** How many of [samples] the loop did not dispatch itself: a {!Queued} call that settles at
          depth 1 resumes the calibration's synchronized singles as its window (gh-ocannl-1074), and
          those launches are already counted in [calibration_launches], so the loop's own launches
          are [depth * (samples - reused)]. 0 at every other depth and under {!Isolated}. *)
  wall_ms : float;  (** The window's summed batch wall. *)
  median_wall_ms : float;
      (** The median of the same batch walls. A bound that must hold whenever the contention claim
          is not bypassed belongs on it rather than on the mean: {!sample_min} declares [contended]
          on a majority of the window exceeding twice its floor, and a median is unmoved by a
          stalled minority while one long batch moves a mean without limit. *)
}
(** A window of {!time_routine}'s timed loop (gh-ocannl-994): the window a reading is a minimum
    over, and so the one a bound on that reading belongs against. The whole call's wall also holds
    the warmup and the calibration's synchronized singles, which on a backend whose host round trip
    is two orders of magnitude above an amortized launch are a comparable share of the call to every
    timed dispatch put together, so a whole-call mean is diluted by construction. *)

type timing_call = {
  decision : batch_decision;  (** The depth decision and the admission verdict. *)
  tagged : bool;
      (** Whether the call ran with [~tag_failures:true] — a search's baseline or candidate, whose
          pre-dispatch validation fired {!on_candidate_preflight}. The diagnostic untuned control is
          untagged. *)
  calibration_singles : int;
      (** The calibration's synchronized single launches (each its own depth-1 batch); 0 under
          {!Isolated}. *)
  calibration_launches : int;
      (** Every launch the calibration dispatched: the singles plus [runs * depth] per probe. *)
  calibration_probes : calibration_probe list;
      (** The CUDA/HIP calibration's batch probes in dispatch order (gh-ocannl-1119); empty for the
          cc/Metal policy and {!Isolated}. *)
  discarded_window : timed_window option;
      (** A contention-refused first window discarded for one fresh retry (Metal queued timing,
          gh-ocannl-1060): its batches were dispatched at [decision.depth], but it is not a
          candidate refusal and the reading is [window]'s alone. *)
  window : timed_window;  (** The window the returned reading was taken over. *)
  setup_ms : float;
      (** Wall from the call's start to the depth decision: for {!time_routine}, the pre-dispatch
          validation, the warmup and the calibration. *)
  timed_ms : float;
      (** Wall from the depth decision to the record: the timed loop, a discarded window included.
      *)
}
(** Everything one timing call did, reported once as it returns (gh-ocannl-1263): replaces the
    separate seams that fired at the depth decision, at each probe, at a retry and at the window,
    which consumers had to stitch together by arrival order. The launches the call dispatched are
    [1 + calibration_launches + depth * (samples - reused)] summed over [window] and
    [discarded_window] (the 1 is {!time_routine}'s warmup; {!calibrate_and_time} has none). *)

val install_timing_call_observer : (timing_call -> unit) -> unit
(** Adds an observer of every {!time_routine} / {!calibrate_and_time} call that returns IN THE
    CALLING DOMAIN, for the rest of the domain's life, after the observers already installed there.
    Trace-only (gh-ocannl-1263): observing changes nothing measured or ranked, and a search running
    in another domain neither reaches nor disturbs it — the treatment [autotune_log] got for
    searches in different domains (see {!with_streams}). A newly spawned domain starts with none.
    How the benchmarks' [BENCH_TIMING_TRACE] is installed. No configuration selects it. *)

type compile_record = {
  compile_routine : string;
  compile_stage : string;
      (** [candidate] for a search attempt; otherwise the progress stage the compile ran in:
          [base_compile], [cache_replay], [cache_replay_compile], [baseline_relink],
          [untuned_default_compile], [untuned_control], [winner_compile]. *)
  compile_attempt : string;  (** The candidate's label; [baseline] for the base compile. *)
  compile_verdict : string;
      (** [compiled] (it reached link), [dedup] (a candidate the search already compiled, answered
          before codegen), [refused] (the typed [Not_dispatched] refusal before codegen), [failed]
          (any other failure, or an exception). *)
  compile_wall_s : float;  (** The whole compile's host wall, link included. *)
  compile_cost : Ir.Compile_cost.t;  (** Its phases, which sum to at most [compile_wall_s]. *)
}
(** One compile a {!tune} call made, and what it cost (gh-ocannl-1314): written to the progress
    stream as a [compiled] event, onto [autotune_log], into {!report.compile_cost}, and to the
    observers of {!install_compile_observer}. A host wall-clock reading, not a timing claim; it is
    never part of a candidate's measured time. *)

val install_compile_observer : (compile_record -> unit) -> unit
(** Adds an observer of every compile a {!tune} call IN THE CALLING DOMAIN makes, as
    {!install_timing_call_observer} does for timing calls: trace-only, per domain, observers called
    in installation order. How the benchmarks' [BENCH_TIMING_TRACE] prints compiles. *)

val with_timing_call_observer : (timing_call -> unit) -> (unit -> 'a) -> 'a
(** [with_timing_call_observer f body] runs [body] with [f] added as by
    {!install_timing_call_observer}, restoring the domain's observers on return or exception, so
    scopes nest. *)

val time_routine :
  ?on_retry:(unit -> unit) ->
  ?on_call:(timing_call -> unit) ->
  ?tag_failures:bool ->
  ?phase:string ->
  timing:timing_mode ->
  repeats:int ->
  Context.t ->
  Context.routine ->
  timing_result
(** [on_retry] defaults to a no-op and observes each fresh contention retry as it starts; [on_call]
    (also a no-op by default) receives the call's {!timing_call} record as it returns, before the
    domain's {!with_timing_call_observer} observers. Both are per call, which is how a search's
    accounting stays its own (gh-ocannl-1263). The tuner's own instrument, exposed so a harness can
    rank candidates by exactly what a search ranks them by (gh-ocannl-755) rather than by a
    re-derivation of it. Binds test values ({!set_test_bindings}) and restores the routine's
    bindings afterwards, runs one warmup, then minimizes the per-launch time over at least 16 and
    [repeats] timed runs, topping up until ~25 ms of per-launch samples has accumulated (up to 64
    runs unless the [repeats] floor is larger) so that a host stall cannot collapse the min-of-N.
    Under [~timing:Queued] a "run" is a whole batch of dispatches whose depth is calibrated per
    candidate to ~10 ms of wall, capped at 2048 on CUDA/HIP and at the historical 200 on cc/Metal,
    and floored at 1. cc/Metal retain the historical synchronized-single estimate. On CUDA/HIP that
    estimate instead seeds a provisional depth; when that is above 1, a provisional queued probe
    separates fixed synchronization from marginal launch cost and selects the depth whose affine
    wall estimate reaches the target. The batch probes take the minimum of twelve runs: unlike
    ranked timing, they only choose scale and are already milliseconds long. Since gh-ocannl-1098
    both are wall-budgeted: a probe stops at three minima once its own wall passes two target-sized
    probes', and once the probes' summed wall reaches {!queue_calibration_wall_ms} no further probe
    starts and the calibration ends unresolved. Up to four further probes validate the depth; a
    resolved crossing shallower than the batches it was fitted from is sampled and checked like an
    interpolation. The first target-sized batch is confirmed at a 25% deeper depth (clamped to and
    measured at the cap) and retained only when the pair's inferred fixed component is below the
    target. A confirmation over 2x its supported base is retried once before an unresolved pair
    selects the cap. Such a no-verdict fallback is then wall-bounded (gh-ocannl-1096): it never goes
    deeper than the deepest depth any calibration batch was measured at within the ~10 ms target,
    and is depth 1 when none was -- the only bound sound for any cost that grows with depth, since
    the readings that left the fits unresolved cannot tell a host stall from a queue threshold. So a
    slow candidate whose fits never resolve is timed at depth 1, not in 2048-launch batches, while a
    fast one whose deeper probes stalled still batches; one that measured no batch within the target
    at all first takes one rescue probe below its shallowest over-target batch, projected linearly
    to the target (gh-ocannl-1098) -- where a queue-threshold or superlinear cost reads within the
    target -- and is refused as [unbatched] only when that also reads over it, rather than timed as
    an isolated depth-1 reading. An unresolved single/probe pair retries at double depth, and the
    next affine fit uses the two batch observations so an inflated single window cannot force the
    cap. If the bounded loop first reaches the target on its last probe, the interpolated target
    depth is still sampled and checked against the measured overshoot. A non-monotone confirmation
    scales from the deeper measured batch, never the earlier suspect crossing. When a clean pair's
    fixed component already fills the target, its marginal slope selects a depth carrying ~10 ms of
    launch work instead of accepting a shallow stalled window or jumping to the cap. After four
    noisy but resolved underestimates, calibration keeps the latest affine projection rather than
    jumping to a 20--30 ms cap batch that would blunt the 2x contention threshold. None of these
    projections settles deeper than {!queue_depth_projection_factor} times the deepest batch the
    calibration probed (gh-ocannl-1100): past it a queue cost can jump where nothing measured. A
    CUDA/HIP routine slower than the target is confirmed by a depth-2 probe, stays at depth 1, and
    is measured identically in both modes. When singles below the target owe batching, a resolved
    pair starting at depth 2 whose marginal launch cost fits the target and whose fixed term is
    below it retains that measured batch even if its fixed term puts the fitted depth-1 wall above
    the target (gh-ocannl-1184). It neither returns to the isolated singles nor refuses an ordinary
    integer-depth boundary; unresolved or over-target marginal fits keep their refusal. Whenever a
    queued call settles at depth 1 its timed window resumes the calibration's synchronized singles
    ({!sample_window}'s [prior]) rather than timing a fresh one (gh-ocannl-1074): at depth 1 they
    are samples of the very quantity the window measures, taken under the same stopping rule, so the
    loop dispatches only what the caller's [repeats] floor asks beyond the calibration's sixteen.
    The calibration always yields a depth; the result of the timed loop reports when most of ITS
    samples were stalled, and the tuner refuses such a candidate measurement rather than ranking and
    caching it (gh-ocannl-888). Since the budget is per-launch rather than batch wall, queued timing
    can spend up to [max 64 repeats] batches on a fast candidate; [max_timing_runs] bounds the
    top-up beyond the caller's requested floor. Metal queued timing retries a contention-refused
    window once at the same depth with fresh samples (gh-ocannl-1060). The retry faces the same
    admission rule; only its outcome reaches the search's refusal accounting. Other backends and
    isolated timing take no retry.

    [phase] labels the call's {!batch_decision} (gh-ocannl-1199); none by default.

    With [~tag_failures:true] the pre-dispatch validation, the launches and the synchronization are
    wrapped in their {!Ir.Schedule_outcome} phases, which is what lets a caller's
    {!Ir.Schedule_outcome.protect} attribute a failure to the phase it happened in; without it they
    propagate raw. Timing dispatches the routine repeatedly against live buffers, so an accumulating
    routine must be timed on a scratch lineage (see [tune]'s [?timing_ctx]) if its inputs matter
    afterwards. [Queued] raises how many such dispatches happen. For [repeats <= 64], the maxima are
    65 under [Isolated] and, under [Queued], 352321 on CUDA/HIP, 12865 on cc or 25665 on Metal. The
    CUDA/HIP bound includes warmup, 64 single-launch calibration runs, at most
    {!queue_calibration_max_probes} (nine) {!queue_batch_probe_runs}-sample calibration probes (the
    provisional probe, four validations, a confirmation, its stall retry, a sampled shallower
    crossing, and the rescue) and 64 timed batches at the cap; cc/Metal have no batch probes. In
    general the queued bounds are [65 + 2048 * (108 + max 64 repeats)] on CUDA/HIP and
    [65 + 200 * max 64 repeats] on cc, twice the timed part on Metal for its one retry. Thus a
    routine whose values grow per run reaches larger ones. That is a fact about the scratch buffers,
    not about the measurement: the cap bounds each in-memory queue while the ~25 ms budget
    accumulates per-launch samples, and a candidate's time is not what it accumulated. *)

val calibrate_and_time :
  retry_contended:bool ->
  timing:timing_mode ->
  repeats:int ->
  queue_depth_cap:int ->
  batch:(int -> float) ->
  timing_result
(** {!time_routine} after its warmup — the calibration and the timed loop, the domain's
    {!with_timing_call_observer} observers included — with the device reduced to [batch depth],
    which must dispatch [depth] launches back to back, synchronize once and return the wall in
    milliseconds. [queue_depth_cap] is {!queue_depth_cap_for_backend}'s value for the backend being
    modelled; it also selects between the CUDA/HIP affine calibration and the historical cc/Metal
    single estimate. Exposed so a test can drive the whole timing policy on an injected clock and
    count its launches exactly (gh-ocannl-1074). [retry_contended] is
    {!retry_contended_window_for_backend}'s value for the backend being modelled; it enables one
    retry only under [Queued], never [Isolated]. *)

val calibrate_and_time_with_retry_observer :
  on_retry:(unit -> unit) ->
  on_call:(timing_call -> unit) ->
  phase:string option ->
  retry_contended:bool ->
  timing:timing_mode ->
  repeats:int ->
  queue_depth_cap:int ->
  batch:(int -> float) ->
  timing_result
(** {!calibrate_and_time} with the per-call observers that drive search accounting. [on_retry] runs
    once when a fresh contention retry starts, before any retry dispatch; [on_call] receives the
    call's {!timing_call} record as {!time_routine}'s does, its [setup_ms] timed from this call's
    own start and its [tagged] false. [phase] is reported as {!batch_decision}'s. The original
    helper remains a no-op-observer, phase-less wrapper for source compatibility. *)

val retry_contended_window_for_backend : string -> bool
(** The backend whose contended queued windows get one immediate retry: Metal only. *)

module Seam : sig
  type 'a t
  (** A test seam of the search: one value PER DOMAIN (the treatment the timing-call observers got,
      gh-ocannl-1263). A domain's searches call the value set in that domain; a search in another
      domain neither reaches nor disturbs it, so searches in different domains -- which take turns,
      {!Concurrent_search} -- can each be observed and injected into on their own. A newly spawned
      domain starts with the seam's default. Unlike {!install_timing_call_observer}, setting a seam
      REPLACES its value rather than chaining onto it: a caller that wants the enclosing value too
      reads it with {!get} first and calls it. *)

  val get : 'a t -> 'a
  (** The calling domain's current value. *)

  val set : 'a t -> 'a -> unit
  (** Replaces the calling domain's value for the rest of the domain's life (or until set again);
      what a long-lived harness installs. *)

  val with_value : 'a t -> 'a -> (unit -> 'b) -> 'b
  (** [with_value seam v body] runs [body] with the calling domain's value replaced by [v],
      restoring the previous value on return or exception, so scopes nest. *)
end

val on_candidate_attempt : (string -> unit) Seam.t
(** Fault-injection seam for the containment tests (gh-ocannl-550), called with each candidate's
    label just before its compile — including the baseline's, which is a candidate (gh-ocannl-533);
    that one is called inside the base compile's transform, so a fault injected there is classified
    like any other and surfaces as the pre-search failure of a search that never started. The
    default is a no-op and no configuration selects it; raising from it terminates the search the
    way an uncontainable failure does — the [Search_died] report (carrying its failure) is emitted
    to [?report] and the exception propagates out of {!tune}, which is what [Train.tune_placements]
    must survive without losing the other arm's winner. Not a production seam: candidate failures
    that a backend {e can} attribute are contained without it (see [declines]). *)

val on_staffed_start :
  (menu:(seg_key * sketch_params list) list ->
  (seg_key * sketch_params) list ->
  (seg_key * sketch_params) list)
  Seam.t
(** Test seam for the staffed sweep (gh-ocannl-1138, config [autotune_staffed_singles]), called once
    per search that sweeps with [~menu], the coarse segment keys and the parameters seeded for each,
    and the running composite the sweep would start from (the seeded untuned plan's sketches matched
    to those keys; empty where there is no seeded plan, on cc always); the sweep starts from its
    result. It is how a test gives a key a choice before that key's own sweep, which only a seeded
    plan does in production. The default is the identity and no configuration selects it. *)

val on_seeded_plan : (seeded_plan option -> seeded_plan option) Seam.t
(** Test seam for the search's seed (gh-ocannl-1441): called with the seeded untuned plan {!tune}
    derived on its base lowering ({!seeded_plan}; [None] off CUDA/HIP or where it is the default
    pipeline), at most once per call; the search seeds, falls back to and names in its trajectory
    identity what it returns. It is how a test runs two seeds against one trajectory, which no
    setting does alone: the keys that steer the plan render in the trajectory's settings themselves.
    The default is the identity and no configuration selects it. *)

val on_privatize_proposal : (target:Ir.Tnode.t -> over:Ir.Indexing.symbol -> unit) Seam.t
(** Fault-injection seam (gh-ocannl-1438): called with each proposal just before
    {!privatize_extension} validates it, inside the validation's refusal boundary, so a typed
    refusal ([Ir.Schedule_outcome.Cause_at]) raised from it drops the proposal as the op's own
    refusal would. No workload of the tuning corpus reaches a real refusal on cc, so this is how a
    test puts a dropped proposal through a real search. The default is a no-op and no configuration
    selects it. *)

val on_candidate_preflight : (string -> unit) Seam.t
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

val on_candidate_timed : (string -> timed_so_far:int -> unit) Seam.t
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

val search_setting : unit -> bool
(** Config [autotune_search] (default [true]): what {!tune} searches under when its caller passes no
    [?search]. *)

val resolve_cache_dir : ?cache_dir:string -> search:bool -> unit -> string
(** The schedule cache directory a tuning call uses, or [""] for none: [cache_dir] when given, else
    config [autotune_cache_dir] — except that with the search off only a CHOSEN directory (passed,
    or set at some config source) counts, the built-in default being treated as no cache
    (gh-ocannl-559). {!tune} resolves its directory through this, and so does
    [Train.tune_placements] for the placement-decision store it keeps in the same directory
    (gh-ocannl-786), so the two open the same directory under the same rule. *)

exception Concurrent_search of { holder : int; caller : int }
(** Raised by {!tune} and [Train.tune_placements] when another domain's search is running
    (gh-ocannl-1352): searches run one at a time per process. [holder] and [caller] are the two
    domains' ids ([Domain.self ()]). The compile path a search drives keeps process-wide state that
    was never made safe for two domains at once (tensor-node registries and uid counters, the
    lowering's analysis cache, a device's constant buffer cache, Base hash tables throughout), so a
    second search is refused before it does anything -- before it resolves a setting, compiles,
    emits a report or calls a seam -- rather than left to race the first. Searches in different
    domains taking TURNS are supported, and what the per-domain {!Seam}s, streams and observers are
    for. *)

val exclusive_search : (unit -> 'a) -> 'a
(** [exclusive_search f] runs [f] as this process's one search: raises {!Concurrent_search} when
    another domain is inside [exclusive_search], and otherwise holds the search for [f]'s duration,
    releasing it on return or exception. Nested calls in the holding domain proceed, so a search
    started inside a search of the same domain (the arms of [Train.tune_placements], a [?report]
    callback that tunes) is not refused. {!tune} runs inside it; a caller wrapping several searches
    that must not interleave with another domain's (as [Train.tune_placements] does) takes it for
    the whole span. *)

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
     search also stops when a round improves the incumbent by less than 1%. A round expands the
     beam's head in full and, on a fissioned routine, every other entry only on the segments where
     it differs from the head (config [autotune_round_expansion], gh-ocannl-1229). *)
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
     [Invalid_argument] otherwise — candidates timed elsewhere do not predict this device). The
     winning schedule is compiled from [ctx], exactly like a cache hit, with or without it. Without
     it, the search shares [ctx]'s buffers and the caller should re-initialize mutated state
     afterwards. *)
  ?abandon:abandon_rule ->
  (* gh-ocannl-1110: stop the search, raising {!Search_abandoned} after delivering an {!Abandoned}
     report, when {!abandon_verdict} decides at [k = beam_width] admitted timings. Absent (the
     default), a search always runs to completion. [Train.tune_placements] passes it for flips only,
     against the incumbent they refine. A clean abandoned prefix is saved under the unchanged
     schedule key in a separate filename space. With the same search shape, a later call
     re-evaluates this rule on that prefix and may raise without searching, reporting
     {!Abandonment_replay}; without a rule, it always searches or replays a winner. *)
  ?log:bool ->
  (* Whether this call writes the [autotune:] diagnostic lines to stderr and, after a search,
     compiles and times the untuned-default in-process control. Omitted, it inherits the [?log] of
     an enclosing [tune] or [Train.tune_placements] call in the same domain, and only without one
     falls back to config [autotune_log] (false), which is read once per process. Scoped to this
     call and to the calling domain, and restored on return or exception, so a test can turn the
     diagnostics on for one search. *)
  ?progress:bool ->
  (* Whether this call writes the [autotune-progress:] lines (see {!progressf}); inherited and
     scoped exactly as [?log] is, falling back to config [autotune_progress] (false). *)
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

    Memory (gh-ocannl-550, gh-ocannl-1326): every candidate this searches is {!Context.release}d as
    soon as its timing window is over — the incumbents the beam ranks included, which are kept as
    descriptions (saved schedules), not artifacts — so at most one candidate's {e working} pools are
    live at a time, and a step that uses over half the device can be tuned. The winner is then
    compiled from [ctx] out of its saved schedule, the one compile this costs over keeping it, and
    the returned routine is the only artifact left when [tune] returns. The bound holds across
    contained failures and across dedups — a deduplicated candidate has still paid for a compile and
    a link. A cache replay is not co-resident with the base compile either: the baseline is released
    before the replay compile.

    A device that refuses the base compile's allocation ({!Ir.Schedule_outcome.Device_memory}) has
    refused one candidate's working set, which every candidate, a replay and the untuned fallback
    need about as much of: the call reports {!Pre_search_failure} at [Backend_link] with that
    decline in its census, and raises, instead of declining every candidate in turn.

    The bound is on working pools, and that qualifier is load-bearing: {!Context.release} cannot
    free per-device constants, and a hoisted [Stage] candidate mints a fresh packed constant per
    application. A search that seeds those (the CPU [hoist] sketches) therefore still grows one
    constant pool per such candidate — measured on [cc] at 1 -> 109 constant pools over 181
    candidates, against working pools held within 2-6. Bounding it needs an eviction rule inside the
    shared constant cache, which is gh-ocannl-565's subject; pinned as far as it can be by
    [test/operations/autotune_candidate_release].

    Budget (gh-ocannl-1409, config [autotune_budget_seconds], default none): a search whose
    wall-clock time since the call started has reached the budget attempts no further candidate --
    checked as each attempt and each beam round starts, so the candidate in flight completes -- and
    crowns its incumbent as a completed search does; the untuned floors (the seeded untuned plan and
    the default-pipeline seed, {!default_seed_label}) are attempted regardless, so the crown is no
    slower than what the caller ships untuned. Where the cut left the untuned default unmeasured --
    no candidate reproduces it (automatic scheduling with [schedule_fission=false] and no seeded
    plan), or its windows were refused -- the call ships the untuned default compile, stores nothing
    and reports as a search that timed nothing ({!budget_floor_unmeasured}). The report's
    [budget_cut] says the budget stopped something. A cut search with clean timings stores its crown
    marked as cut ({!Ir.Schedule_cache.entry}'s [budget_cut]); a later call with a budget replays
    it, and one without a budget searches in full and overwrites it. The budget's value is part of
    the search trajectory's identity, so a cut search's [best_steps] serve early stopping only under
    the same budget.

    Untuned fallback (gh-ocannl-1441): every path on which {!tune} ships "the untuned default
    compile" -- search disabled with nothing to replay, a refused lineage, a deferred cache replay
    that failed, a search that timed nothing, a budget cut that left the floor unmeasured, a winner
    replay that failed -- compiles the step the recipes ship untuned without
    [model_default_schedule]: {!seeded_default}'s seeded plan wherever its settings admit one (CUDA
    and HIP, seeding on), the plain default pipeline elsewhere. After a search the plan is the one
    the search seeded, replayed as its [F_seeded] candidate; without one it is derived inside the
    fallback's own compile, as {!seeded_default} derives it. Under [model_default_schedule=true] the
    fallback is still the seeded plan, not {!model_default}'s pick: the floor the search measured. A
    seeded fallback that fails classified recompiles the plain default. *)

type deferred
(** The outcome of a {!tune_deferred} search: the compile of what it ships, not yet run. Holds no
    device resources. *)

val tune_deferred :
  ?name:string ->
  ?search:bool ->
  ?beam_width:int ->
  ?rounds:int ->
  ?repeats:int ->
  ?timing:timing_mode ->
  ?seed_block_sizes:int list ->
  ?cache_dir:string ->
  ?keep_fraction:float ->
  ?max_split_reduce_sites:int ->
  ?timing_ctx:Context.t ->
  ?abandon:abandon_rule ->
  ?log:bool ->
  ?progress:bool ->
  ?report:(report -> unit) ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  deferred
(** {!tune}, except that the routine it ships is not compiled: {!compile_deferred} compiles it
    (gh-ocannl-1408). The search, its cache store and its report happen here, with {!tune}'s
    arguments, failures and reporting contract; then nothing the call compiled is left resident --
    every candidate, the baseline and a validating cache replay are released -- so a caller running
    several searches before choosing one, as [Train.tune_placements] does with its placement arms
    and flips, holds no searched routine while the next one searches, and compiles only the one that
    ships.

    The difference from {!tune} is the order of the report and the compile: the report is delivered
    before the compile, so a failure of {!compile_deferred} raises without a report of its own -- it
    comes after the search has reported its outcome. What the compile runs is what {!tune} would
    have: the winner's saved schedule replayed from [ctx] (falling back to the untuned default where
    {!tune}'s winner replay would), the untuned default where the search shipped that, and for a
    cache hit the entry's replay -- compiled twice in all, since the first replay (the one that
    tells a stale entry from a sound one, and supplies the report's emission census) is released. *)

val lineage_identity :
  static_indices:Ir.Indexing.static_symbol list -> Ir.Low_level.optimized -> string
(** gh-ocannl-1421: what a lowering of the routine in one lineage must share with one in another for
    a schedule tuned on the first to apply to the second: its code, loop binders canonically
    numbered and every tensor node anchored to its actual identity (the canonical form alone
    alpha-renames nodes, so [u := x - y] and [u := y - x] would match). Process-local, by node ids:
    {!tune} compares the caller's and the timing context's lowerings, taken together so they see the
    same tensor-node value bounds -- a lowering taken after a compile has settled an index tensor's
    bounds folds a gather guard an earlier one kept -- and refuses on a difference
    ({!Lineage_refused}). *)

val lineage_accesses : Ir.Low_level.optimized -> Ir.Tnode.t -> string
(** How the lowering's code accesses the node: ["r"], ["w"], ["rw"], or ["-"] for not at all. Stable
    under the bounds drift, and per node, so [Train.placement_problem] tags each node with it inside
    its canonically numbered (process-independent) key. *)

val compile_deferred : deferred -> Context.t * Context.routine
(** Compiles what a {!tune_deferred} call ships, from its [ctx], under the [?log] and [?progress]
    that call ran under, and like {!tune_deferred} itself inside {!exclusive_search} -- the first
    call only, which compiles; a later one returns the memoized outcome unguarded. At most once: a
    second call returns the same context and routine, or raises the same exception. A [deferred]
    dropped without this call costs nothing on the device. *)

val shipped_substitute : deferred -> bool
(** Whether {!compile_deferred} shipped the untuned default IN PLACE of the schedule the call's
    report named -- the winner's or the cache entry's replay failed or degenerated, the fallbacks
    {!tune} takes too. The report, delivered before the compile, still describes that schedule, so a
    caller that ranked or recorded by it reads this before relying on the routine: what it got was
    never measured. [false] before the compile. *)

val budget_setting : unit -> float option
(** Config [autotune_budget_seconds] as {!tune} resolves it (gh-ocannl-1409): [Some] seconds when
    the setting is a positive number, [None] (no budget) when it is empty, zero or negative; text
    that is no number raises [Utils.Invalid_config_value] (gh-ocannl-1332). *)

val budget_floor_unmeasured :
  budget_cut:bool -> auto_sched:bool -> baseline_dispatched:bool -> floor_measured:bool -> bool
(** gh-ocannl-1409: whether a search its budget cut ships the untuned default compile, stores
    nothing and reports as a search that timed nothing, rather than crowning what it reached: the
    budget cut it ([budget_cut]), it measured neither untuned floor ([floor_measured]: the default
    seed's time, the baseline's when automatic scheduling is off, or the seeded plan's), and the
    untuned default is a form a timed candidate could be slower than -- a scheduled pipeline
    ([auto_sched]) or a dispatched serial baseline ([baseline_dispatched]), not the undispatched
    serial form of a GPU with automatic scheduling off. *)

val budget_elapsed : (float -> float) Seam.t
(** Test seam (gh-ocannl-1409): maps the call's real elapsed seconds to the seconds a budget check
    reads. A test substitutes a virtual clock -- the attempts so far, say -- to place a cut at a
    chosen candidate, which a wall clock cannot. The default is the identity and no configuration
    selects it. *)

val on_candidate_callback :
  ([ `Timed | `Calibration ] -> candidate_ms:float -> incumbent_ms:float -> unit) Seam.t
(** Default-no-op fault-injection seam immediately before each post-admission callback. The times
    are the admitted measurements the search ranks, after {!on_candidate_measured}; a strictly
    larger [candidate_ms] selects a nonwinner while an already measured incumbent exists. *)

val on_candidate_measured : (label:string -> digest:string -> float -> float) Seam.t
(** Measurement seam (gh-ocannl-1027), called once per ADMITTED timing window — the dispatched
    baseline's (label ["baseline"]) and each candidate's — with the candidate's label, its schedule
    digest and the admitted time in milliseconds. The returned time replaces the reading before the
    admission gate, so it is what the search ranks, records and reports: the beam, the best-so-far
    incumbent, {!on_candidate_callback}, [report.best_ms]/[baseline_ms], the calibration rows and
    the schedule cache's stored times. A refused window (contended, or a degenerate clock reading)
    never reaches it unless the test explicitly opts into {!with_uncontended_test_windows};
    degenerate readings still never reach it. A returned time that is not finite and positive
    refuses the window the way a degenerate reading does. It must return rather than raise: it runs
    outside the search's containment boundaries. It exists so a test about the nonwinner paths can
    pin the ranking — make the first admitted window the fastest and every later one slower —
    instead of depending on the machine's timings. The default is the identity and no configuration
    selects it. *)

val with_uncontended_test_windows : (unit -> 'a) -> 'a
(** Explicit test-only scope (gh-ocannl-1156) for scripted measurements: clear the physical window's
    contention flag before candidate admission and {!on_candidate_measured}. Invalid physical or
    scripted readings and unresolved queued batching still refuse admission. Calibration and
    dispatch remain real. Restores the previous mode on return or exception, including for nested
    scopes. No configuration selects it; normal searches retain production contention refusal. Like
    the {!Seam} test hooks it is per domain: it covers the searches the calling domain runs, and
    neither reaches nor disturbs a search in another domain. *)

val apply_measurement_seam : label:string -> digest:string -> timing_result -> timing_result
(** Candidate-window admission and scripted measurement, shared by real searches and deterministic
    opposing controls. Production refuses contended windows before calling {!on_candidate_measured};
    {!with_uncontended_test_windows} explicitly opts a scripted fixture out of that refusal only. *)

val staffed_singles_setting : unit -> bool
(** Config [autotune_staffed_singles] as {!tune} resolves it (gh-ocannl-1138, default [false]):
    whether its per-segment singles are swept against a running composite. A staffed search times
    its recombination composites inside the sweep, so a test counting the bare search's extra
    composite window reads this rather than restating the default. *)

val default_seed_label : backend_name:string -> string option
(** The label {!on_candidate_measured} receives for the candidate a search on [backend_name]
    attributes as the untuned default — the one whose refused window sets [report.default_refused]
    and whose time is [report.default_ms]. [Some "baseline"] with automatic scheduling inactive,
    where the default is the serial form — [None] there on a GPU backend, which never dispatches it
    (gh-ocannl-532); the default-reproducing [config_thresholds] seed's label with it active and
    fissioning; [None] when it is active without fission, where no candidate reproduces the default.
    {!tune}'s own attribution reads this, so the two cannot disagree. One case it cannot foresee: a
    default seed whose code equals the serial baseline's dedups against it, and is timed only under
    the label ["baseline"]. *)

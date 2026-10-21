(** {1 Structured schedule candidates}

    Site recognition, composed matmul and convolution pipelines, and the refinement trees consumed
    by {!Autotune}. Construction depends on the lowering, hardware limits and current configuration;
    this module does not compile or time candidates. {!sketch_seed_params} combines the families
    into the search's seed list. *)

(** {2 Recognizing sites and describing candidates} *)

(** Which pair of a matmul site the coalesced tuner branch ({!coalesced_seed_params}) merges into
    one loop with {!Ir.Schedule.constructor-Coalesce}. One pair per site and layout. *)
type coalesce_layout =
  | Coalesce_columns
      (** gh-ocannl-1165: the innermost interior batch loop into the column role —
          [Coalesce { outer = last m_bi; inner = m_j }], plus the zeroing nest's own last two loops
          alike — so the q/k/v projections' column tile spans several heads. *)
  | Coalesce_contraction
      (** gh-ocannl-1279: the two innermost contraction loops into one k —
          [Coalesce { outer = last m_ko; inner = m_k }] — so the attention out projection
          [d[b,s,j] += w[j,h,e] * x[b,s,h,e]] contracts over one loop of extent [h * e], and its
          k-tile can span several heads. Output coordinates are untouched, so the zeroing nest is
          not coalesced. Its seeds come from a bounded K menu of their own (depths 256 down to 32
          over 16x16, 32x32 and 64x64 blocks, gh-ocannl-1459, under the usual workgroup-memory,
          thread, pad and launch judgments), not the family's 8/16-deep tiles. *)
[@@deriving sexp]

type sketch_params = {
  sk_gpu : bool;  (** Register blocktiling with shared staging vs. CPU operand packing. *)
  sk_mma : bool;
      (** Tensorized (tile-MMA) pipeline instead of the scalar blocktiling/packing one: on GPU,
          Split → (optional cooperative shared Stage) → Tensorize targeting [simdgroup_matrix] /
          tensor cores; on cc, the whole-triple [Tile_mma] rendered register-tiled (gh-ocannl-469),
          optionally Grid-parallel over row blocks — or, with [sk_bk > 0], the cache-blocked packed
          composition (packing Stages feeding the register-tiled kernel;
          [cpu_mma_pack_sketch_schedule]), itself optionally Grid-parallel ([sk_grid]: hoisted
          packing runs Grid-outermost; in-kernel packing relies on the renderer's per-chunk tile
          privatization). Seeded directly because the greedy menu cannot reach the composition: a
          bare [Tensorize] from the serial baseline (one simdgroup, everything else serial) loses
          round 1 and the beam discards it before Grid retypes could join it. *)
  sk_simd : int;  (** MMA lane width ([hardware_limits.mma_simd_width]); 0 when [not sk_mma]. *)
  sk_bm : int;
  sk_bn : int;
  sk_bk : int;
      (** For GPU MMA sketches, [sk_bk = 0] = unstaged (one full-K [Tile_mma] block). For conv GPU
          seeds, [sk_bn]/[sk_bk] are re-purposed as the pad-to multiples of the column/reduction
          extents (gh-ocannl-485; 0 = already an intrinsic-tile multiple). *)
  sk_tm : int;
      (** Register-tile factors; unused on CPU. For conv GPU seeds, [sk_tm] is re-purposed as the
          row pad-to multiple of the unblocked flavor (gh-ocannl-485; 0 = no pad). *)
  sk_tn : int;
  sk_hoist : bool;
      (** CPU packing only: pack compile-time-constant operands out of the routine, into the
          per-device constant pool (gh-ocannl-470). Proposed alongside the in-kernel packing variant
          so the choice stays measured; applied per operand, only to hoistable (known-constant,
          host-init-backed) sources. *)
  sk_grid : bool;
      (** CPU packed composition only ([sk_mma] with [sk_bk > 0]): split [i] into pool-parallel
          [Grid] row blocks instead of Serial ones. Four shapes, keyed by [sk_hoist] and
          [sk_pack_rest]:

          - With [sk_hoist] alone, hoisted-only packing: only hoistable operands are packed (at link
            time, into the constant pool) and the rest are read in place, leaving the kernel body
            all-materialized; the Grid loop stays outermost (one dispatch spanning the whole GEBP
            triple). The typical inference GEMM: activations (in place) x constant weights.
          - With [sk_hoist] and [sk_pack_rest], the mixed grid-outermost shape (gh-ocannl-473):
            hoistable operands still pack at link time, but a non-hoistable operand gets an
            in-kernel packing Stage instead of being read in place — its tile lands inside the Grid
            body and is privatized to per-chunk block-scope storage by the renderer. For the
            inference GEMM this recovers the A~ pack the hoisted-only shape forfeits (a per-chunk
            [bm x bk] tile) while keeping the single outermost dispatch.
          - With [sk_pack_rest] alone, grid-outermost in-kernel packing (gh-ocannl-475): both
            operands pack inside the Grid body and privatize per chunk — each chunk re-packs its own
            B~ panel (redundant copies, but one dispatch instead of one per k-block). Needs the
            tiles under the renderer's per-chunk privatization cap (config
            [cc_grid_private_bytes_cap]).
          - Without [sk_hoist] or [sk_pack_rest], in-kernel packing: the per-row-block A~ packing
            Stage lands inside the Grid body — its tile is privatized to per-chunk block-scope
            storage by the renderer ([C_syntax.parallel_grid_safe]'s privatization rule) — while the
            B~ panel packs at the k-block loop outside the Grid and is read-only inside (shared
            across the row-block chunks, behind a pointer alias under the blocks extension),
            re-entering the parallel construct once per k-block.

          Proposed alongside the serial flavors so the choice stays measured. *)
  sk_pack_rest : bool;
      (** Grid-outermost packed compositions only (with [sk_grid]): give non-hoistable operands a
          non-hoisted in-kernel packing Stage instead of reading them in place, relying on the
          renderer's per-chunk tile privatization. With [sk_hoist], the mixed shape of gh-ocannl-473
          (hoisted constant panel + per-chunk pack of the rest); without [sk_hoist], the per-chunk
          B~ re-packing shape of gh-ocannl-475 — the Grid loop stays outermost (one dispatch
          spanning the GEBP triple) and every operand packs inside the Grid body. No effect on the
          serial flavors or the hoisted-only Grid flavor, whose stages are already determined. *)
  sk_conv : bool;
      (** Convolution site (gh-ocannl-493): the seed instantiates the implicit-GEMM conv pipeline
          ([cpu_conv_sketch_schedule] / [gpu_conv_sketch_schedule] via [detect_conv]) instead of a
          matmul one. The packing [Stage] serves as im2col and the micro-kernel is the ordinary
          [Tile_mma] ([sk_mma] is set so the census expectations apply). On CPU, [sk_grid]
          pool-parallelizes the outermost batch/spatial loop — on merged segments with the aligned
          whole-segment geometry of the default preset ([conv_aligned_grid]). On GPU backends with
          an mma capability ([sk_gpu] with [sk_simd] the lane width), the staged pipeline: outer
          loops [Grid]-typed, cooperative shared-tile staging, the accumulator fragment resident
          across the kernel window (gh-ocannl-480). *)
  sk_epilogue : bool;
      (** Epilogue fusion (gh-ocannl-486): append [Sched.Fuse_epilogue] on the site's output, so the
          sole-consumer elementwise tail (bias add / activation / residual) folds into the
          store-back and the whole routine is one kernel — the fused competitor to the fissioned
          two-kernel form. The matmul family tree's root level (gh-ocannl-613): the fused flavor is
          refuted with the recognizer's own reason ([Sched.fuse_epilogue_witness]) when the base
          code has no fusable tail, and otherwise enumerates after every unfused leaf; a candidate
          whose scheduled form no longer admits the fusion (e.g. materializing unrolls duplicating
          the store-back) fails its compile and is skipped like any other invalid candidate. On GPU
          the accumulator moves to workgroup-shared memory (the [shared] flag) so the Metal fragment
          intrinsics keep firing after placement makes it routine-local. *)
  sk_batch_grid : bool;
      (** GPU matmul pipelines on batched (rank-3+) sites only (gh-ocannl-643): [Retype] the site's
          batch loops — [m_bo] and the hoisted [m_bi] — to [Grid], so a batched/multi-head GEMM's
          batch and head axes launch as grid blocks (folded onto the hardware [.z] dimension, see
          [Low_level]'s hardware-axis section comment) instead of running as serial loops inside
          each block. The zeroing nest and every companion nest carry the same per-position
          annotation, with interior batch loops hoisted identically, so the cross-nest positional
          thread identity is preserved. Seeded as a {e twin} of each geometry — the serial-batch
          flavor stays measured, because block-count curves are non-monotone (gh-ocannl-569's probe
          peaked near 128 blocks and regressed by 1024): the tuner, not a heuristic, decides whether
          the extra parallelism beats the occupancy it costs. Refuted at the leaf, like every other
          launch dimension, when the batch extents' product exceeds the backend's [.z] limit
          ([Schedule.launch_geometry_excess] over [hardware_limits.max_grid_yz], with
          [max_grid_fold_extent] standing in where the backend advertises none) — the same reading
          [Schedule.check_hardware_limits_classified] enforces pre-driver for schedules that do not
          come from these seeds. *)
  sk_batch_inner : bool;
      (** With [sk_batch_grid], on sites with interior batch loops ([m_bi], the q/k/v projections'
          heads) only (gh-ocannl-728): bind the interior batch loops {e inside} the row blocks —
          grid nest order [m_bo; row blocks; m_bi; column blocks] instead of
          [m_bo; m_bi; row blocks; column blocks]. The interior batch then takes the grid slot next
          to the column blocks, and the row blocks fold with [m_bo] onto [.z]: consecutive blocks in
          launch order sweep the heads of one row block, the order the heads-merged twin of the same
          site launches in (heads on the column grid axis). Move 0 of gh-ocannl-728 measured most of
          that twin's gain at an unchanged tile and block count — i.e. from this launch order. A
          third batch flavor rather than the [sk_batch_grid] twin's order, for the same reason the
          twin is one: the tuner measures, not a heuristic. Always [false] where [sk_batch_grid] is,
          and on sites without interior batch loops, where the two orders are the same schedule. *)
  sk_coalesce : coalesce_layout option;
      (** GPU scalar blocktile pipeline only: [Some layout] when the seed was enumerated on — and
          its schedule is built over — the site with that layout's pair coalesced
          ({!coalesce_prefix}): the interior batch loop into the column role (gh-ocannl-1165), so
          the column tile spans useful columns of several heads, or the contraction pair into one k
          (gh-ocannl-1279), so the k-tile spans several heads. The schedule is that structural
          prefix followed by the ordinary pipeline over the coalesced site; every tile field of the
          record describes the coalesced site ([sk_bk] a tile of the merged k). [None] for the
          family's own seeds. *)
  sk_swizzle : Ir.Low_level.swizzle_kind option;
      (** Staged GPU mma sketches only ([sk_mma] with [sk_bk > 0]): store both cooperative operand
          tiles in this XOR layout (gh-ocannl-481 item 3, D3). Seeded as a {e twin} of each staged
          seed — same tile sizes, both operands marked — and only for format triples the backend
          advertises in {!Ir.Backend_intf.mma_capability.mma_staged_layouts}, so a twin is never
          proposed where the emission would decline it back to the scalar fallback (gh-ocannl-479).
          The tuner, not a heuristic, decides whether the bank-conflict fix beats the plain tile:
          the same "propose both, measure" pattern as hoisted packing. Unstaged seeds have no shared
          tile to swizzle and are never twinned. *)
  sk_depth : int;
      (** Staged GPU mma/conv sketches: the cooperative stages' software-pipelining depth
          ([Schedule.Stage ~pipeline_depth], gh-ocannl-487); 1 = unpipelined. Depths > 1 are seeded
          as {e twins} of each staged seed — same tile sizes, same pipeline, so a timing difference
          between the two is the prefetch overlap's (against the halved occupancy from the doubled
          shared-memory footprint), and nothing else's — for exactly the depths the backend
          advertises in {!Ir.Backend_intf.mma_capability.mma_pipeline_depths}, and only for staged
          operands of at least 4-byte storage — the async arms' element floor
          ([C_syntax_config.async_copy]); a narrower twin could only render the portable synchronous
          form, whose occupancy cost phase 1 measured. The rendering is bitwise identical to the
          plain sibling, so the tuner's choice is free of numerics concerns. Unstaged seeds have no
          cooperative copy to pipeline and are never twinned. *)
  sk_pack_prec : Ir.Ops.prec option;
      (** CPU packing compositions only: the compute precision the site's register-tiled
          micro-kernel runs at, resolved by the seeding pre-filter through
          {!Ir.Numerics.cpu_compute_prec} (gh-ocannl-575). The packing [Stage]s mint their tiles at
          this precision ([Stage.tile_prec]) where it differs from an operand's storage precision,
          folding the narrow-storage widening into the packing copy — packed panels become e.g. f32
          scratch, converted once per element at pack time instead of once per read inside the
          micro-kernel. [None] for GPU seeds and for CPU sites whose storage already is the compute
          precision. Recorded in the params (rather than re-derived at build time) because schedule
          construction has no [hardware_limits] and the instantiated schedule must reproduce the
          seed-time decision exactly. *)
  sk_tile : Ir.Register_tile.t option;
      (** CPU tensorized pipelines only: the register-tile geometry the [Tensorize] carries
          (gh-ocannl-619). [None] lets the renderer's ranking model choose
          ({!Ir.Register_tile.default}); the family tree twins each CPU tensorized leaf with the
          {!Ir.Register_tile.alternatives} of its micro-kernel extents (the "register-tile" level),
          so the width the tuner ships is measured rather than modelled. *)
}
[@@deriving sexp]

type matmul_site = {
  m_i : Ir.Indexing.symbol;
  m_j : Ir.Indexing.symbol;
  m_k : Ir.Indexing.symbol;  (** Innermost contraction loop, of extent [m_nk]. *)
  m_ni : int;
  m_nj : int;
  m_nk : int;
  m_ko : (Ir.Indexing.symbol * int) list;  (** Enclosing contraction loops in nest order. *)
  m_bo : (Ir.Indexing.symbol * int) list;  (** Batch loops outside the row loop. *)
  m_bi : (Ir.Indexing.symbol * int) list;  (** Batch loops inside the row loop. *)
  m_row_axis : int;
  m_row_stride : int;
      (** [m_a]'s coefficient on [m_i]: greater than [1] on a strided row only a compacting [Stage]
          of [m_a] can tile (gh-ocannl-1148), which the classifier picks where no unit-stride row
          exists or where it is strictly taller than the unit-stride one (gh-ocannl-1397); [1]
          otherwise. *)
  m_d : Ir.Tnode.t;
  m_a : Ir.Tnode.t;
  m_b : Ir.Tnode.t;
  m_zeroed : bool;
  m_tb : bool option;
  m_fma : bool;
}
(** The matmul site separates the innermost contraction from its enclosing contraction and batch
    loops. Extents and operand roles are read from the serial lowering. *)

val detect_matmul : Ir.Low_level.t -> matmul_site option
(** Recognize an all-serial matmul accumulation; [None] when its access relations cannot supply
    distinct row, column and contraction roles. *)

val detect_matmul_canonical :
  Ir.Low_level.optimized -> (matmul_site * Ir.Schedule.schedule * Ir.Low_level.optimized) option
(** The site with its enabling-interchange prefix and the code it was detected on (gh-ocannl-1183);
    see {!Autotune.detect_matmul_canonical}. *)

val reduction_sink :
  accum_prec:(Ir.Ops.prec -> Ir.Ops.prec) -> Ir.Low_level.optimized -> Ir.Schedule.schedule
(** The adjacent-[Swap] chain sinking each accumulation's reduction loops below its write loops
    (gh-ocannl-1348), statement by statement, every [Swap] [Op_legal]; empty where no statement
    qualifies. See {!Autotune.reduction_sink}. *)

type conv_axis = {
  cx_o : Ir.Indexing.symbol;  (** Output spatial iterator. *)
  cx_no : int;
  cx_k : Ir.Indexing.symbol;  (** Kernel-window iterator. *)
  cx_nk : int;
  cx_stride : int;
  cx_dilation : int;
  cx_offset : int;  (** Offset in the affine input access. *)
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
(** The implicit-GEMM row, output channel and reduction channel, together with outer output and
    kernel-window loops. [c_axes] retains the spatial access coefficients. *)

val detect_conv : Ir.Low_level.t -> conv_site option
(** Recognize an implicit-GEMM convolution with affine spatial accesses. Singleton axes whose loops
    lowering removed are refused; a 1x1 window belongs to {!detect_matmul}. *)

(** {2 Judging capability and launch geometry}

    These queries let the harness and tests consult the family builders' judgments before
    constructing or compiling a candidate. *)

val matmul_launch_geometry : matmul_site -> sketch_params -> Ir.Schedule.launch_geometry
(** Predict a GPU matmul seed's grid and workgroup extents from its site and parameters. CPU
    parameters return {!Ir.Schedule.unknown_launch_geometry}. *)

val conv_launch_geometry : conv_site -> sketch_params -> Ir.Schedule.launch_geometry
(** Predict the GPU convolution seed's launch geometry, as a lower bound on the applied schedule.
    CPU parameters return {!Ir.Schedule.unknown_launch_geometry}. *)

val mma_format_triples :
  a_prec:Ir.Ops.prec ->
  b_prec:Ir.Ops.prec ->
  d_prec:Ir.Ops.prec ->
  (Ir.Backend_intf.mma_input_format
  * Ir.Backend_intf.mma_input_format
  * Ir.Backend_intf.mma_input_format)
  list
(** Operand and accumulator format triples in policy preference order, for lookup in the backend's
    advertised tiles and staged layouts. *)

val tensorized_capability_refutation :
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  a_prec:Ir.Ops.prec ->
  b_prec:Ir.Ops.prec ->
  d_prec:Ir.Ops.prec ->
  string option
(** Why backend capabilities or configuration withhold tensorized candidates, before any geometry or
    site structure is considered. [None] admits the family, subject to its later site and geometry
    checks. *)

val matmul_mma_scope : matmul_site -> bk:int -> Ir.Backend_intf.mma_emission_scope
(** The accumulator lifetime of a matmul candidate: enclosing contraction loops and, for [bk > 0],
    the staged reduction blocks determine whether it spans multiple statements. *)

val mma_tile_for_precisions_in_scope :
  Ir.Backend_intf.mma_capability ->
  scope:Ir.Backend_intf.mma_emission_scope ->
  a_prec:Ir.Ops.prec ->
  b_prec:Ir.Ops.prec ->
  d_prec:Ir.Ops.prec ->
  (int * int * int) option
(** Resolve the policy-selected advertised tile, withholding unsupported wide-accumulator lifetimes.
    Used by the placement enablement analysis for its whole-contraction candidate. *)

(** {2 Enumerating and refining the candidate families}

    Decisions carry data; labels are for display. The search can refine lazy subtrees, lift geometry
    exclusions and price committed traffic without parsing labels. *)

(** Each level commits to one pipeline, packing, geometry or fusion decision. [path] records these
    commitments in outermost-first order; [to_label] and [render_path] only render them. *)
module Family_decision : sig
  type geometry = { g_bm : int; g_bn : int; g_bk : int; g_tm : int; g_tn : int }

  type geometry_choice =
    | Gpu_blocktile of geometry
    | Gpu_mma of geometry
    | Cpu_blocktile of int
    | Cpu_packed of geometry
    | Lattice

  type t =
    | Fusion of [ `Unfused | `Fused ]
    | Pipeline of [ `Blocktile | `Tensorized ]
    | Batch of [ `Serial | `Grid | `Grid_inner ]
    | Packing of [ `In_kernel | `Hoisted ]
    | Geometry of geometry_choice
    | Lattice_box of { lb_axis : [ `Bm | `Bk ]; lb_lo : int; lb_hi : int }
    | Twin of [ `Plain | `Swizzled | `Depth of int ]
    | Tensorized_form of [ `Whole_triple | `Packed ]
    | Row_block of int
    | Packing_shape of
        [ `Serial | `Hoisted | `Hoisted_grid | `Hoisted_grid_pack_rest | `Grid_pack_rest | `Grid ]
    | Register_tile of Ir.Register_tile.t option
    | Conv_leg of [ `Cpu | `Gpu ]
    | Conv_shape of [ `Serial | `Grid | `Row_block of int ]
    | Conv_staging of [ `Plain | `Pipelined ]
    | Conv_rows of int

  type path = (string * t) list

  val equal : t -> t -> bool
  val compare : t -> t -> int
  val level : t -> string
  val to_label : t -> string
  val render_path : path -> string
end

val matmul_sketch_tree :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree option
(** The lazy matmul family, including unfused and epilogue-fused flavors; [None] when no site is
    recognized. Refutations carry statically decidable builder failures. The workgroup-memory
    refutations charge the staged tiles as the gate does ([Schedule.kernel_workgroup_bytes]):
    [capabilities] (the compiling context's [Context.codegen_capabilities]) supplies the backend's
    allocation rounding and tile-MMA scratch; the default,
    {!Ir.Backend_intf.no_codegen_capabilities}, charges the exact tile sizes alone — a weaker floor,
    never a wrong one (gh-ocannl-1255). *)

val conv_sketch_tree :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree option
(** The lazy convolution family (gh-ocannl-697); [None] when {!detect_conv} recognizes no site. The
    root is the same epilogue-fusion level as {!matmul_sketch_tree}'s, each flavor judged under its
    own preconditions; below it the [conv-leg] (CPU, then GPU, the other refuted), the CPU
    [conv-shape] menu, and the GPU [conv-staging] then [conv-rows] menus with the pipelined
    geometries' depth [twin]s. Every leaf is judged against the launch caps
    ({!launch_geometry_refutation}'s witness). [capabilities] as for {!matmul_sketch_tree}. *)

val sketch_family_tree :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree option
(** The tree of whichever family claims the lowering — the matmul family where a matmul site is
    detected (directly or through the enabling interchange), else the convolution family — unforced
    below its root; [None] when neither does. {!sketch_seed_params} is its leaves. *)

val sketch_seed_params :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  sketch_params list
(** The composed seed list: {!Ir.Schedule_space.leaves} of {!sketch_family_tree}, fused twins
    included. Empty when neither family detects a site. *)

(** {2 The seeding census (gh-ocannl-1407)} *)

type family_seeding =
  | Seeding_matmul of {
      site : matmul_site;
      tree : (Family_decision.t, sketch_params) Ir.Schedule_space.tree;
    }  (** The matmul family claimed the lowering; its refusals are the tree's refutations. *)
  | Seeding_conv of {
      site : conv_site;
      real_stmts : int;
          (** The lowering's real top-level statements besides the site's own covering zero — the
              count the CPU leg's companion caps judge. The GPU leg judges companion coverage
              instead ([companion_geometry], gh-ocannl-1406). *)
      tree : (Family_decision.t, sketch_params) Ir.Schedule_space.tree;
    }  (** No matmul site, a conv site; its refusals are the tree's refutations (gh-ocannl-697). *)
  | Seeding_none  (** Neither family detects a site. *)

val family_seeding :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  family_seeding * sketch_params list
(** {!sketch_seed_params} with the family tree its seeds are the leaves of: the seed list is the
    second component. *)

type coalesced_verdict = {
  cv_layout : coalesce_layout;
  cv_seeds : sketch_params list;
  cv_refusal : string option;  (** Set exactly when the layout proposes no seed. *)
}
(** One layout of the coalesced tuner branch ({!coalesced_seed_params}): its seeds or why it
    proposes none. *)

val coalesced_seeding :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  coalesced_verdict list
(** {!coalesced_seed_params} per layout, columns then contraction, refusals named; empty off GPU.
    Hermetic like it. *)

type seeding_census = {
  sc_stmts : int;  (** Top-level statements, glue excluded. *)
  sc_whole_zeros : int;
      (** Of those, whole-node zero initializers: a [Zero_out] or its expanded covering nest
          ([Low_level.zero_initializer_target]). *)
  sc_conv_nests : (Ir.Tnode.t * bool) list;
      (** Every top-level statement {!detect_conv} recognizes, with whether one of the whole-node
          zero initializers above targets its output — listed whatever family claims the lowering.
      *)
  sc_family : family_seeding;
  sc_seeds : sketch_params list;  (** Exactly {!sketch_seed_params}' list. *)
  sc_coalesced : coalesced_verdict list option;
      (** The coalesced branch's verdicts, when the census was asked for them ([~coalesced:true], as
          {!Autotune.tune}'s seeding enumerates that branch beside the families). *)
}
(** What seeding makes of one lowering, refusals included: the evidence a decline report cannot
    hold, since a refused seed never becomes a candidate. *)

val seeding_census :
  ?coalesced:bool ->
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  seeding_census
(** Hermetic: the symbols any probe mints are rewound. *)

val seeding_census_prefix : string
(** ["autotune-seeding:"], the prefix every census line carries. *)

val seeding_census_lines : where:string -> seeding_census -> string list
(** The census rendered for stderr: a head line
    [where="…" stmts=N whole_zeros=N family=matmul|conv|none [site=…] seeds=N [refuted=N
     excluded=N]], then indented detail lines — the claiming tree's refutations grouped by witness
    with the first refuted path, after a conv site's zeroing and real-statement count — then, for a
    census taken with [~coalesced:true], one line per coalesced layout (its refusal, or its seed
    count and the leaves' distinct geometries, gh-ocannl-1459), and the lowering's conv nests. *)

val coalesce_reduction_pair :
  ?static_indices:Ir.Indexing.static_symbol list ->
  Ir.Low_level.optimized ->
  outer:Ir.Indexing.symbol ->
  inner:Ir.Indexing.symbol ->
  (Ir.Indexing.symbol * Ir.Low_level.optimized) option
(** gh-ocannl-1414: the loops [outer] and [inner] merged by a {!Ir.Schedule.constructor-Coalesce},
    as the merged loop's fresh symbol and the coalesced code (built on a hermetic copy), or [None]
    where the op declines. The one preparation of a reduction pair: the split-reduce sites of a
    two-loop band ([Autotune.split_reduce_sites]) and the contraction layout
    ([Coalesce_contraction], gh-ocannl-1279) both prepare through it. *)

val coalesce_prefix :
  layout:coalesce_layout -> Ir.Low_level.optimized -> Ir.Schedule.schedule option
(** A coalesced layout's structural prefix, when the detected matmul site has that layout's pair and
    the op accepts it. [Coalesce_columns] (gh-ocannl-1165): [Coalesce (last m_bi, m_j)], preceded,
    when the site's output is zeroed in the routine, by that [Zero_out]'s expansion with its last
    two loops coalesced alike — or, for a covering per-cell zero nest (a GPU fission segment's
    folded zero, gh-ocannl-1175), by that nest's own coalescing. [Coalesce_contraction]
    (gh-ocannl-1279): [Coalesce (last m_ko, m_k)] alone, prepared by {!coalesce_reduction_pair}, and
    kept only when the prepared code re-detects as the expected site — the merged loop as [m_k] over
    the pair's product, the pair gone from [m_ko], every output role unchanged. [None] when there is
    no such pair, or when the op declines (a per-head operand, a padded inner axis, a weight storing
    the pair apart, any access not reading the pair as adjacent plain iterators) — probed on a
    hermetic copy. Mints fresh symbols, which the returned ops carry. *)

val coalesced_seed_params :
  ?capabilities:Ir.Backend_intf.codegen_capabilities ->
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  sketch_params list
(** The coalesced tuner branch, beside {!sketch_seed_params}: per layout (columns, then
    contraction), the matmul family's unfused GPU scalar-blocktile leaves enumerated over the code
    {!coalesce_prefix} produces — every refutation judged on the coalesced site — each stamped
    [sk_coalesce = Some layout]. Empty off GPU and where no prefix applies. *)

val geometry_lattice_witness : string
(** The exclusion witness for tile-lattice alternatives beyond the curated geometry menus. *)

val pad_waste_max : unit -> float
(** The pad-waste threshold the geometry menus judge against (config [autotune_pad_waste_max]). *)

val lift_pad_waste :
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree ->
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree
(** Lift the pad-waste exclusions, preserving laziness, other exclusions and legality refutations.
*)

val lift_geometry_lattice :
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree ->
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree
(** Lift geometry-lattice exclusions by their decision, preserving laziness, other exclusions and
    legality refutations. *)

val sketch_path_traffic_floor :
  limits:Ir.Backend_intf.hardware_limits -> Ir.Low_level.optimized -> Family_decision.path -> int
(** Additional bytes every completion below a path must move beyond
    {!Ir.Cost_model.completion_floor}. Detection runs at partial application; an unrecognized site
    or an uncommitted staging decision contributes zero. *)

(** {2 Constructing schedules and inspecting reductions} *)

val sketch_schedule :
  accum_prec:(Ir.Ops.prec -> Ir.Ops.prec) ->
  p:sketch_params ->
  Ir.Low_level.optimized ->
  Ir.Schedule.schedule
(** Re-detect the site and instantiate the composed pipeline. A builder precondition that does not
    hold -- no site of the family's kind is recognized, or the parameters do not fit -- is refused
    typed, as [Ir.Schedule_outcome.Cause_at (Transform, cause)] ([Illegal_schedule] at check
    ["Autotune.sketch"], or [Unsupported] for a companion-coverage or output-rank limitation); any
    other exception escaping the builder is a builder bug and propagates untyped, fatal under
    [strict_failure_classification] (gh-ocannl-1390). [accum_prec] is the rendering backend's
    accumulator resolution, used for scalar privatization. *)

val with_sketch_builder_fault : (sketch_params -> unit) -> (unit -> 'a) -> 'a
(** Test seam (gh-ocannl-1390), the sketch-level twin of {!Ir.Schedule.with_transform_fault}:
    [with_sketch_builder_fault fault body] runs [body] with [fault] called on the parameters at the
    start of each {!sketch_schedule}, inside the builder, so a test can make the builder raise what
    a builder bug raises. The calling domain's previous fault is restored on return or exception;
    the default is a no-op. *)

val collect_gets : Ir.Low_level.scalar_t -> (Ir.Tnode.t * Ir.Indexing.axis_index array) list
(** Collect ordinary reads through scalar arithmetic, retaining their index maps. Dynamic reads
    contribute the reads of their index expression; local scopes are opaque. Used by the harness's
    reduction-site and placement analyses. *)

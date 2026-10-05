(** {1 Structured schedule candidates}

    Site recognition, composed matmul and convolution pipelines, and the refinement trees consumed
    by {!Autotune}. Construction depends on the lowering, hardware limits and current configuration;
    this module does not compile or time candidates. {!sketch_seed_params} combines the families
    into the search's seed list. *)

(** {2 Recognizing sites and describing candidates} *)

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

  type path = (string * t) list

  val equal : t -> t -> bool
  val compare : t -> t -> int
  val level : t -> string
  val to_label : t -> string
  val render_path : path -> string
end

val matmul_sketch_tree :
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  (Family_decision.t, sketch_params) Ir.Schedule_space.tree option
(** The lazy matmul family, including unfused and epilogue-fused flavors; [None] when no site is
    recognized. Refutations carry statically decidable builder failures. *)

val sketch_seed_params :
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  sketch_params list
(** The composed seed list: matmul tree leaves when a matmul is detected, otherwise convolution
    seeds followed by their eligible epilogue-fusion twins. Empty when neither site is detected. *)

val geometry_lattice_witness : string
(** The exclusion witness for tile-lattice alternatives beyond the curated geometry menus. *)

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
(** Re-detect the site and instantiate the composed pipeline. Raises [Invalid_argument] if no site
    is recognized or the parameters do not fit. [accum_prec] is the rendering backend's accumulator
    resolution, used for scalar privatization. *)

val collect_gets : Ir.Low_level.scalar_t -> (Ir.Tnode.t * Ir.Indexing.axis_index array) list
(** Collect ordinary reads through scalar arithmetic, retaining their index maps. Dynamic reads
    contribute the reads of their index expression; local scopes are opaque. Used by the harness's
    reduction-site and placement analyses. *)

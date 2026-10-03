(** {1 Structured schedule candidates}

    Site recognition, composed matmul and convolution pipelines, and the refinement trees consumed
    by {!Autotune}. Construction depends on the lowering, hardware limits and current configuration;
    this module does not compile or time candidates. {!Autotune.sketch_seed_params} combines the
    families into the search's seed list. *)

(** {2 Recognizing sites and describing candidates} *)

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
  sk_swizzle : Ir.Low_level.swizzle_kind option;
  sk_depth : int;
  sk_pack_prec : Ir.Ops.prec option;
  sk_tile : Ir.Register_tile.t option;
}
(** A complete pipeline choice: [sk_gpu]/[sk_mma]/[sk_conv] select the builder; block and register
    tile sizes set its geometry. Packing, fusion, batch placement, staged layout and depth describe
    the measured alternatives. For convolution candidates, the tile fields also encode pad-to
    multiples; see the implementation's field documentation. *)

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

val matmul_seed_params :
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  opt:Ir.Low_level.optimized ->
  matmul_site ->
  sketch_params list
(** Enumerate the matmul tree's leaves in seed order for an already recognized site. *)

val conv_seed_params :
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  (sketch_params list * Ir.Tnode.t) option
(** Detect and seed the convolution family, returning its output node so the harness can append
    epilogue-fusion twins. [None] when no site is detected. *)

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

val idcs_mention : Ir.Indexing.axis_index array -> Ir.Indexing.symbol -> bool
(** Whether a symbol occurs in a plain iterator or affine component of the index map. The harness
    uses this to distinguish output axes from reduction axes. *)

val collect_gets : Ir.Low_level.scalar_t -> (Ir.Tnode.t * Ir.Indexing.axis_index array) list
(** Collect ordinary reads through scalar arithmetic, retaining their index maps. Dynamic reads
    contribute the reads of their index expression; local scopes are opaque. Used by the harness's
    reduction-site and placement analyses. *)

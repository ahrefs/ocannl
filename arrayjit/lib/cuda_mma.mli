(** CUDA's tensor-core tables as pure functions of storage precisions, the numerics policy and the
    compute capability: the one arm variant the backend's [mma_syntax] and [mma_fragment_syntax]
    hooks dispatch on, its arch floors, markers and tiles, the capability descriptor
    ({!Backend_intf.mma_capability}) autotune seeds from, and the nvrtc arch-target policy that
    reads the floors back out of rendered source.

    They live here, outside the [cudajit]-gated [Cuda_backend], so they are tested wherever OCANNL
    builds, with no toolkit and no device ([test/operations/cuda_mma_cutoffs], gh-ocannl-1214;
    [test/operations/cuda_mma_arms] and [arrayjit/test/test_cuda_arch_flags], gh-ocannl-1232).

    {b One resolution} (gh-ocannl-1232). Every CUDA mma cutoff is written once, in {!arm_floor}.
    {!resolve_arm} is the single table from a storage triple, the policy and the emission scope to
    an arm; the hooks match on its result (so they have no dispatch order to restate), {!mma_arm}
    projects it into {!Backend_intf.codegen_capabilities}, {!capability} names each descriptor entry
    by the arm it seeds, and {!gpu_arch_options} maps each arm's rendering marker to that arm's
    floor. Compute capabilities are [major * 10 + minor] throughout. *)

(** {1 The arm variant} *)

type m16n8k16_kind =
  | M16n8k16_bf16  (** Uniform bf16 (gh-ocannl-545): the combination wmma cannot express. *)
  | M16n8k16_f16  (** Uniform f16 under [Numerics.Fp16_wide], per statement (gh-ocannl-680). *)
[@@deriving equal, sexp_of, enumerate]

(** The inline-PTX arms. Both accumulate in the architected m16n8 f32 per-lane register layout,
    which is what lets their persistent-fragment scope hold registers across an outer reduction
    (gh-ocannl-1063). *)
type inline_ptx =
  | Fp8_m16n8k32  (** [mma.sync.aligned.m16n8k32] over e5m2 x e5m2 -> f32. *)
  | M16n8k16 of m16n8k16_kind
      (** [mma.sync.aligned.m16n8k16] with 16-bit operands; one body over both element types, the
          PTX fragment layouts being shared by .f16 and .bf16. *)
[@@deriving equal, sexp_of, enumerate]

(** The [nvcuda::wmma] C++-API arms (tensorize-mma T3). *)
type wmma_kind =
  | Wmma_f16_f32
  | Wmma_f16_f16  (** Gated off under [Numerics.Fp16_wide]. *)
  | Wmma_f16_wide
      (** f16 x f16 -> f32 fragments over an f16 destination converted at the [d] boundary
          (gh-ocannl-925): only a fragment scope renders it. *)
  | Wmma_bf16
  | Wmma_tf32  (** Uniform f32 read as tf32, only when [Numerics.t.tf32_matmuls] opts in. *)
[@@deriving equal, sexp_of, enumerate]

(** Every tensor-core rendering the CUDA hooks can emit. [all_of_arm] enumerates them, so a table
    over arms (the arch-target markers, the floors' minimum) covers a new arm automatically. *)
type arm = Inline_ptx of inline_ptx | Wmma of wmma_kind [@@deriving equal, sexp_of, enumerate]

val resolve_arm :
  a_prec:Ops.prec ->
  b_prec:Ops.prec ->
  d_prec:Ops.prec ->
  scope:Backend_intf.mma_emission_scope ->
  arm option
(** The arm the hooks render a storage triple with in an emission scope, under the current numerics
    policy; [None]: no tensor-core form, the scalar fallback renders. Only the precision- and
    policy-level choice: the hooks still decline on extents, strides, address spaces, layouts and
    {!arm_floor}, and a decline is final -- no hook falls through to another arm. An update-only
    statement against a fragment array is resolved at [Mma_fragment_scope], the scope that declared
    the array. The one scope-dependent triple is uniform f16 under [Numerics.Fp16_wide]: inline-PTX
    m16n8k16 per statement, [Wmma_f16_wide] in a fragment scope; a per-statement arm never has a
    converted destination. *)

val arm_floor : arm -> int
(** The arm's arch floor, the minimum compute capability whose PTX contains its instructions: the
    only place CUDA's mma cutoffs are written. *)

val arm_marker : arm -> string
(** The marker the arm's rendering comment carries in parentheses (e.g. [wmma-tf32], [mma-bf16]),
    published as [Backend_intf.mma_arm.arm_name]; {!gpu_arch_options} reads it back. *)

val arm_tile : arm -> int * int * int
(** The intrinsic tile [(m, n, k)]: 16x8x32 for fp8, 16x8x16 for m16n8k16, 16x16x8 for tf32 and
    16x16x16 for the other wmma arms. *)

val arm_accumulator : arm -> Ops.prec
(** The precision the arm's accumulator resides at: f32 for both inline-PTX instructions whatever
    the storage, the fragment's for wmma. *)

val arm_ldmatrix_operands : arm -> ta:bool -> tb:bool -> a:bool -> b:bool -> bool * bool
(** Which operands the arm loads with the warp-cooperative [ldmatrix], given whether each is
    eligible at all (shared memory in the swizzled 16-byte-unit layout) and stored transposed. fp8
    registers are contiguous only in A's row-major and B's transposed orientations; the 16-bit arms
    use [ldmatrix] in both; wmma never. Shared by emission and its census, and the per-arm place an
    operand-load description for the cost model hangs (gh-ocannl-1254). *)

val ldmatrix_eligible : Backend_intf.mma_operand_place -> bool
(** Whether an operand can be loaded with [ldmatrix] at all (gh-ocannl-481 item 3): the swizzled
    16-byte-unit layout in shared memory, the only combination whose per-lane row addresses
    [ldmatrix] can both reach and de-conflict. The emission's eligibility test. *)

val operand_loads :
  a_prec:Ops.prec ->
  b_prec:Ops.prec ->
  d_prec:Ops.prec ->
  scope:Backend_intf.mma_emission_scope ->
  ta:bool ->
  tb:bool ->
  a:Backend_intf.mma_operand_place ->
  b:Backend_intf.mma_operand_place ->
  bool * bool
(** {!arm_ldmatrix_operands} of the scope's arm ({!resolve_arm}) over each operand's
    {!ldmatrix_eligible}: which operands an accepted call loads with [ldmatrix]. The backend's
    [C_syntax_config.mma_operand_loads], so the rendering census and
    {!Backend_intf.codegen_capabilities.mma_operand_loads} — what the cost model's load tie-break
    reads (gh-ocannl-1254) — both answer with the emission's own rule. [(false, false)] for a triple
    with no arm. *)

val arm_reloads_operands : arm -> bool
(** Whether the arm's emission reloads each operand's intrinsic fragment for every tile of the other
    operand's free axis: the inline-PTX bodies load both fragments inside their [m x n x k] tile
    loops, wmma loads each fragment once per reduction step and reuses it. *)

val operand_fragments :
  a_prec:Ops.prec ->
  b_prec:Ops.prec ->
  d_prec:Ops.prec ->
  scope:Backend_intf.mma_emission_scope ->
  Backend_intf.mma_fragments option
(** The scope's arm ({!resolve_arm}) as its emission loads operands: {!arm_tile} — the arm's own
    tile, not the descriptor's (uniform f16 under [Fp16_wide] emits m16n8k16 per statement under an
    advertised 16x16x16) — and {!arm_reloads_operands}; [None] with no arm. The backend's
    [C_syntax_config.mma_operand_fragments], published as
    {!Backend_intf.codegen_capabilities.mma_operand_fragments} for the cost model's operand-load
    count (gh-ocannl-1254). *)

(** {1 Per-arm rendering data} *)

type wmma_info = {
  wc_ab_typ : string;
      (** The [matrix_a]/[matrix_b] fragment element type: a C++ type for the 16-bit combinations,
          the tag type [nvcuda::wmma::precision::tf32] for tf32 (whose storage stays [float]). *)
  wc_acc_prec : Ops.prec;  (** The accumulator fragment's precision. *)
  wc_ab_ld_mult : int;  (** wmma stride constraint: a/b leading-dim multiple, in elements. *)
  wc_d_ld_mult : int;  (** wmma stride constraint: d leading-dim multiple, in elements. *)
  wc_cvt_tf32 : bool;
      (** Convert loaded a/b fragment elements with [__float_to_tf32] (the intrinsic requires
          already-converted inputs). *)
  wc_d_cvt : (string * string) option;
      (** The [(widen, narrow)] conversions of a destination whose storage type is not the
          accumulator fragment's ([Wmma_f16_wide]): [d] then crosses the fragment boundary element
          by element. *)
}

val wmma_info : wmma_kind -> wmma_info
(** A wmma arm's fragment types and stride constraints; its tile is {!arm_tile}. *)

val wmma_accepts :
  wmma_kind ->
  cc:int ->
  m:int ->
  n:int ->
  k:int ->
  d:[ `Resident of [ `Plain | `Swizzled_b128 ] | `Addressed of int * Backend_intf.mma_operand_place ] ->
  a:int * Backend_intf.mma_operand_place ->
  b:int * Backend_intf.mma_operand_place ->
  bool
(** Whether a wmma arm renders a statement on a device of compute capability [cc]: every layout
    plain, {!arm_tile} dividing the extents, the leading dimensions meeting {!wmma_info}'s stride
    multiples, every addressed operand in device or shared memory, and [cc] at the arm's
    {!arm_floor}. [d] is [`Resident] for an update-only statement against the fragment array a scope
    declared (its accumulator is not addressed), and [`Addressed (ldd, place)] for a self-contained
    statement's destination or a fragment scope's backing target. The one predicate both of
    [Cuda_backend]'s wmma sites decide by (gh-ocannl-1456); a self-contained statement additionally
    never has a converted destination ([wc_d_cvt]). *)

val wmma_frag_typ : wmma_kind -> string -> string -> string option -> string
(** [wmma_frag_typ kind role elt layout]: the [nvcuda::wmma::fragment] type of [role] ([matrix_a],
    [matrix_b], [accumulator]) over the arm's {!arm_tile} blocks of [elt], with [layout] for an
    operand. *)

val wmma_k_loop_lines :
  wmma_kind ->
  acc:string ->
  ta:bool ->
  tb:bool ->
  lda:int ->
  ldb:int ->
  m:int ->
  n:int ->
  k:int ->
  string list
(** The reduction loop of an accepted [m x n x k] wmma statement, as both of [Cuda_backend]'s hooks
    render it: [load_matrix_sync]s of the [a] and [b] fragments from [__mma_ap] / [__mma_bp]
    ([col_major] for transposed storage, [ta] / [tb]; tf32 then converts each element) and
    [mma_sync] into the [mt x nt] accumulator array [acc]. {!operand_fragments} describes its loads,
    which [test/operations/mma_format_arm_tables] counts. *)

type m16n8k16_spelling = {
  ms_elt_typ : string;  (** The C element type. *)
  ms_as_ushort : string;  (** The bits-as-ushort intrinsic. *)
  ms_widen : string;  (** Element to [float]. *)
  ms_narrow : string;  (** [float] to element. *)
  ms_infix : string;  (** The instruction's element infix ([bf16], [f16]). *)
}

val m16n8k16_spelling : m16n8k16_kind -> m16n8k16_spelling

val inline_ptx_d_storage : inline_ptx -> string * (string * string) option
(** An inline-PTX arm's destination element type and its [(widen, narrow)] boundary conversions
    ([None] for fp8's f32 destination, moved directly). *)

val inline_ptx_accepts :
  inline_ptx ->
  cc:int ->
  m:int ->
  n:int ->
  k:int ->
  a:Backend_intf.mma_operand_place ->
  b:Backend_intf.mma_operand_place ->
  bool
(** Whether an inline-PTX arm renders a statement's operands on a device of compute capability [cc]:
    {!arm_tile} dividing the extents, every operand plain or {!ldmatrix_eligible}, in device or
    shared memory, and [cc] at the arm's {!arm_floor}. The one operand predicate both of
    [Cuda_backend]'s inline-PTX sites decide by (gh-ocannl-1511); the destination's conditions stay
    with them. *)

val inline_ptx_statement_lines :
  inline_ptx ->
  a_prec:Ops.prec ->
  b_prec:Ops.prec ->
  resident:string option ->
  ta:bool ->
  tb:bool ->
  lda:int ->
  ldb:int ->
  ldd:int ->
  a_swz:bool ->
  b_swz:bool ->
  m:int ->
  n:int ->
  k:int ->
  string * string list
(** An accepted [m x n x k] inline-PTX statement as [Cuda_backend]'s [mma_syntax] renders it
    (gh-ocannl-1511): its block-comment title and the lines after the [__mma_dp] / [__mma_ap] /
    [__mma_bp] pointer declarations -- the [__mi x __ni x __ki] loops, each step gathering (or
    [ldmatrix]-loading) an [a] and a [b] fragment and issuing one [mma.sync]. [resident] names the
    register fragment array an update-only call accumulates into ([None]: a self-contained statement
    that loads and stores [d] at its storage type, {!inline_ptx_d_storage}). [a_swz] / [b_swz]: the
    operand is {!ldmatrix_eligible}; the arm's {!arm_ldmatrix_operands} decides which of them load
    through [ldmatrix], and the others gather per lane through the swizzled address map.
    {!operand_fragments} and {!operand_loads} describe its loads, which
    [test/operations/mma_format_arm_tables] counts. *)

val inline_ptx_d_boundary_lines :
  inline_ptx -> dir:[ `Load | `Store ] -> frag:string -> ldd:int -> mt:int -> nt:int -> string list
(** The [d] boundary of an inline-PTX register fragment scope (gh-ocannl-1063): [frag] is
    [float frag[mt][nt][4]], one m16n8 f32 accumulator tile per [(__mi, __ni)], filled from
    ([`Load]) or written back to ([`Store]) the storage at [__mma_dp] (leading dimension [ldd]),
    converted per {!inline_ptx_d_storage}. *)

(** {1 Capability} *)

val mma_arm :
  a_prec:Ops.prec ->
  b_prec:Ops.prec ->
  d_prec:Ops.prec ->
  scope:Backend_intf.mma_emission_scope ->
  Backend_intf.mma_arm option
(** {!resolve_arm} projected to its marker, accumulator and floor (gh-ocannl-1153): the backend's
    [C_syntax.C_syntax_config.mma_arm]. *)

val async_copy_floor : int
(** The [cp.async] staging arm's arch floor (gh-ocannl-487 phase 2): the backend renders it, and
    {!capability} proposes software-pipelined depths, exactly at and above it. *)

val capability : cc:int -> Backend_intf.mma_capability option
(** The tile-MMA descriptor of a device of compute capability [cc]; [None] below every arm's floor.
    Each entry is named by the arm it seeds and takes that arm's floor and tile. *)

(** {1 nvrtc architecture targets} *)

val gpu_arch_options : device_cc:int -> string -> string list
(** The [--gpu-architecture] options one nvrtc compile of the given CUDA source gets, given the
    minimum compute capability across the attached devices: a pure function of the source's arch
    markers. A FLOOR target (forward-JIT-compiled on every later GPU) is the max of the triggered
    floors -- each arm's [(<arm_marker>)] at its {!arm_floor}, [nvcuda::wmma] at the lowest wmma
    floor, [cp.async.] at {!async_copy_floor}, half/bf16 arithmetic at theirs -- raised to
    compute_75 where every device can load that (CUDA 13 dropped offline compilation below it). The
    [(mma-mxfp8)] marker instead selects a [compute_<cc>a] FAMILY target, only on a device of that
    family (gh-ocannl-481 item 3, D4). *)

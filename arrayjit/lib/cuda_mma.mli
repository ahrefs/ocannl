(** CUDA's tensor-core tables as pure functions of storage precisions, the numerics policy and the
    compute capability: the arm resolver the backend's [mma_syntax] and [mma_fragment_syntax] hooks
    dispatch on, the arch floors those arms check, and the capability descriptor
    ({!Backend_intf.mma_capability}) that autotune seeds from.

    They live here, outside the [cudajit]-gated [Cuda_backend], so the agreement between the two
    owners of the 70/80/89 cutoffs -- the descriptor's [entry ~min_cc] and the resolver's floors
    ([wc_min_cc], {!mma16_min_cc}) -- is tested on both sides of every cutoff wherever OCANNL
    builds, with no toolkit and no device ([test/operations/cuda_mma_cutoffs], gh-ocannl-1214). The
    backend consumes these definitions directly: [Cuda_backend.Impl] includes this whole interface
    into its syntax configuration, and builds [hardware_limits.mma] as [capability ~cc] at the
    attached devices' minimum compute capability. Compute capabilities are [major * 10 + minor]
    throughout. *)

(** {1 The wmma arm} *)

type wmma_combo_info = {
  wc_ab_typ : string;
      (** The [matrix_a]/[matrix_b] fragment element type: a C++ type for the 16-bit combinations,
          the tag type [nvcuda::wmma::precision::tf32] for tf32 (whose storage stays [float]). *)
  wc_acc_prec : Ops.prec;  (** The accumulator fragment's precision. *)
  wc_tm : int;
  wc_tn : int;
  wc_tk : int;
      (** The intrinsic tile shape: 16x16x16 for the 16-bit combinations, 16x16x8 for tf32. *)
  wc_ab_ld_mult : int;  (** wmma stride constraint: a/b leading-dim multiple, in elements. *)
  wc_d_ld_mult : int;  (** wmma stride constraint: d leading-dim multiple, in elements. *)
  wc_min_cc : int;  (** The arm's arch floor. *)
  wc_marker : string;
      (** Marker suffix of the rendering comment ([""], ["-bf16"], ["-tf32"], ["-f16-wide"]);
          [Cuda_backend.gpu_arch_options] greps the [-bf16] and [-tf32] ones for their floor. *)
  wc_cvt_tf32 : bool;
      (** Convert loaded a/b fragment elements with [__float_to_tf32] (the intrinsic requires
          already-converted inputs). *)
  wc_d_cvt : (string * string) option;
      (** The [(widen, narrow)] conversions of a destination whose storage type is not the
          accumulator fragment's (the wide-f16 arm, gh-ocannl-925): [d] then crosses the fragment
          boundary element by element, and only a fragment scope renders the combination. *)
}
(** One wmma-supported precision combination (tensorize-mma T3). *)

val wmma_combo : a_prec:Ops.prec -> b_prec:Ops.prec -> d_prec:Ops.prec -> wmma_combo_info option
(** The wmma combination of a storage triple under the current numerics policy, shared by
    [mma_syntax] and [mma_fragment_syntax] so that a fragment scope accepts exactly when its nested
    update-only calls would. *)

(** {1 The inline-PTX arms} *)

val mma16_spellings :
  a_prec:Ops.prec ->
  b_prec:Ops.prec ->
  d_prec:Ops.prec ->
  (string * string * string * string * string * string) option
(** The m16n8k16 arm's element spellings: the C type, the bits-as-ushort intrinsic, the widening and
    narrowing conversions, the instruction's element infix, and the arch marker. Returns [Some] for
    uniform bf16 under every policy and uniform f16 under [Numerics.Fp16_wide]; [None] otherwise. *)

val mma_fp8_marker : string
(** The marker of the fp8 m16n8k32 arm, shared by its statement and register-scope renderings. *)

val mma_fp8_combo : a_prec:Ops.prec -> b_prec:Ops.prec -> d_prec:Ops.prec -> bool
(** Whether the triple is the fp8 x fp8 -> f32 combination of the m16n8k32 arm. *)

val mma16_register_scope :
  a_prec:Ops.prec ->
  b_prec:Ops.prec ->
  d_prec:Ops.prec ->
  (string * (string * string) option * string) option
(** The persistent per-lane register scope of the inline-PTX arms: the destination's element type,
    its [(widen, narrow)] boundary conversions ([None] for an f32 destination), and the arm marker.
    [None] where the fragment scope renders through wmma instead. *)

val mma16_min_cc : a_prec:Ops.prec -> b_prec:Ops.prec -> d_prec:Ops.prec -> int
(** The inline-PTX arms' arch floor: 89 for the fp8 arm, 80 otherwise. *)

(** {1 Resolution and capability} *)

val mma_arm :
  a_prec:Ops.prec ->
  b_prec:Ops.prec ->
  d_prec:Ops.prec ->
  scope:Backend_intf.mma_emission_scope ->
  Backend_intf.mma_arm option
(** The arm each hook selects at the precision level, in the hooks' own dispatch order
    (gh-ocannl-1153); the backend's [C_syntax.C_syntax_config.mma_arm]. *)

val async_copy_floor : int
(** The [cp.async] staging arm's arch floor (gh-ocannl-487 phase 2): the backend renders it, and
    {!capability} proposes software-pipelined depths, exactly at and above it. *)

val capability : cc:int -> Backend_intf.mma_capability option
(** The tile-MMA descriptor of a device of compute capability [cc]; [None] below sm_70. Each entry
    mirrors an arm of {!mma_arm}, including its accumulator format and arch floor. *)

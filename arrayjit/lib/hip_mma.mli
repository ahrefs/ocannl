(** HIP's rocWMMA tables as pure functions of storage precisions and the numerics policy: the
    bf16-residency predicate, the one combination table both [Hip_backend] mma hooks consult, the
    arm table it publishes ({!Backend_intf.codegen_capabilities.mma_arm}), and the capability
    descriptor ({!Backend_intf.mma_capability}) autotune seeds from.

    They live here, outside the [hipjit]-gated [Hip_backend], so they are tested wherever OCANNL
    builds, with no toolkit and no device ([test/operations/mma_format_arm_tables], gh-ocannl-1246).
    The backend still gates the descriptor at the device level (gfx11/gfx12 wave32 with discoverable
    rocWMMA headers), which is why no arm names a floor. *)

val bf16_accum_wide : unit -> bool
(** Whether bf16 reduction accumulators reside in f32 on this backend: under [Numerics.Bf16_wide],
    and since gh-ocannl-1051 under [Bf16_auto] too -- only [Bf16_narrow] keeps gfx11's
    bf16-accumulate WMMA and storage-width serial accumulators. The one predicate behind the
    backend's [accum_prec] and {!mma_accumulator}, so emission and residency cannot disagree. *)

val tile : int
(** rocWMMA's square fragment extent: every arm's intrinsic tile is [tile x tile x tile]. *)

val mma_accumulator :
  a_prec:Ops.prec -> b_prec:Ops.prec -> d_prec:Ops.prec -> (Ops.prec * int * int) option
(** Per storage triple, under the current numerics policy: the precision the accumulator fragment
    resides at, and the wmma leading-dimension multiples for a/b and for d; [None]: no rocWMMA form,
    the scalar fallback renders. The accumulator and the destination storage coincide on every arm
    but the wide-f16 and wide-bf16 ones, where the backend's [mma_d_boundary_lines] converts. *)

val mma_arm :
  a_prec:Ops.prec ->
  b_prec:Ops.prec ->
  d_prec:Ops.prec ->
  scope:Backend_intf.mma_emission_scope ->
  Backend_intf.mma_arm option
(** {!mma_accumulator} projected to an arm, the same in both emission scopes (gh-ocannl-1153): the
    backend's [C_syntax.C_syntax_config.mma_arm]. *)

val capability : Backend_intf.mma_capability
(** The tile-MMA descriptor of a device set that qualifies for rocWMMA. Its [mma_format_tiles] lists
    the STORAGE triples {!mma_accumulator} has an arm for; [test/operations/mma_format_arm_tables]
    checks the correspondence both ways under every numerics policy. *)

(** Metal's [simdgroup_matrix] tables as pure functions of storage precisions and the numerics
    policy: the one combination resolver both [Metal_backend] mma hooks consult, the arm table it
    publishes ({!Backend_intf.codegen_capabilities.mma_arm}), and the capability descriptor
    ({!Backend_intf.mma_capability}) autotune seeds from.

    They live here, outside the [metal]-gated [Metal_backend], so they are tested wherever OCANNL
    builds, with no device ([test/operations/mma_format_arm_tables], gh-ocannl-1246). The backend
    still gates the descriptor at the device level (Apple7 and later), which is why no arm names a
    floor. *)

val mma_accumulator : d_prec:Ops.prec -> a_prec:Ops.prec -> b_prec:Ops.prec -> Ops.prec option
(** The precision a storage triple's accumulator fragment resides at, under the current numerics
    policy; [None]: no [simdgroup_matrix] form, the scalar fallback renders. Every fragment type
    follows from this precision, and the destination boundary converts exactly where it differs from
    the destination's storage precision. *)

val mma_arm :
  a_prec:Ops.prec ->
  b_prec:Ops.prec ->
  d_prec:Ops.prec ->
  scope:Backend_intf.mma_emission_scope ->
  Backend_intf.mma_arm option
(** {!mma_accumulator} projected to an arm, the same in both emission scopes (gh-ocannl-1153): the
    backend's [C_syntax.C_syntax_config.mma_arm]. *)

val capability : Backend_intf.mma_capability
(** The tile-MMA descriptor of an Apple7+ device. Its [mma_format_tiles] lists the STORAGE triples
    {!mma_accumulator} has an arm for; [test/operations/mma_format_arm_tables] checks the
    correspondence both ways under every numerics policy. *)

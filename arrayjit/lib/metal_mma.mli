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

(** {1 Operand loads (gh-ocannl-1456)} *)

val tile : int
(** [simdgroup_matrix]'s square extent: every arm's intrinsic tile is [tile x tile x tile]. *)

val k_loop_lines :
  a_frag:string ->
  b_frag:string ->
  acc:string ->
  ta:bool ->
  tb:bool ->
  lda:int ->
  ldb:int ->
  m:int ->
  n:int ->
  k:int ->
  string list
(** The reduction loop of an accepted [m x n x k] [Tile_mma] statement, as both of the backend's
    hooks render it: [simdgroup_load]s of the [a] and [b] tiles from [__mma_ap] / [__mma_bp] (with
    [transpose_matrix] for transposed storage, [ta] / [tb]) and [simdgroup_multiply_accumulate] into
    the [mt x nt] accumulator array [acc]. [a_frag] / [b_frag] are the operand fragment types. *)

val operand_fragments :
  a_prec:Ops.prec ->
  b_prec:Ops.prec ->
  d_prec:Ops.prec ->
  scope:Backend_intf.mma_emission_scope ->
  Backend_intf.mma_fragments option
(** How {!k_loop_lines} brings the operands of a triple with an arm ({!mma_accumulator}) into
    registers: the {!tile} cube, each fragment once per reduction step, whole. The backend's
    [C_syntax_config.mma_operand_fragments]; [test/operations/mma_format_arm_tables] counts the
    loads {!k_loop_lines} renders against it. *)

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
(** [(false, false)]: Metal has no [ldmatrix] analogue — the hooks decline a swizzled operand, and
    every load {!k_loop_lines} emits is a whole-fragment [simdgroup_load]. The backend's
    [C_syntax_config.mma_operand_loads]. *)

(** Mixed-precision training recipe (gh-ocannl-492): master weights via cast twins, and dynamic loss
    scaling. This is the torch-AMP / JAX-policy recipe translated to OCANNL's structures; it
    composes with {!Precision_policy} (storage-precision assignment over the rest of the graph) and
    with [Ir.Numerics] (compute-precision policy, gh-ocannl-478).

    The three pieces and how they fit together:

    - {b Master weights} ({!with_master_weights}): the parameter tensor node stays at the session
      default precision (typically f32) and remains the optimizer's target — the canonical value
      that SGD updates and {!Persistence} saves. The graph instead reads a reduced-precision "cast
      twin" ({!Operation.cast}) inserted by a {!Tensor.param_postprocess} hook at parameter
      construction. The twin is recomputed from the master each forward pass; whether it is a
      virtual node (recompute-on-read) or a materialized per-step copy is the [placement] choice.
      The twin rounds the master at the twin's precision whichever placement it gets, on every
      backend and under either [narrow_compute_f32], so the choice never changes what a read of the
      twin gives (gh-ocannl-1866); it can still decide which schedules are reachable, since a matmul
      reads a virtual twin through a scope. That guarantee is the cast's: a virtual narrow node
      computing {e arithmetic} elsewhere in the model keeps the compute-precision policy of
      gh-ocannl-517, and skips its store's rounding where compute is wider than storage. The twin's
      gradient flows back into the master's f32 gradient through an accumulating assignment, which
      is where the f16 -> f32 widening happens.
    - {b Loss scaling} ({!Loss_scaler}): f16's exponent range underflows on small gradients, so the
      backprop is seeded with a scale factor ([Train.grad_update ~loss_scale]) and gradients are
      unscaled in the optimizer step ([Train.sgd_update ~grad_unscale]). The dynamic schedule backs
      the scale off when non-finite gradients are detected ([Train.grad_checksum] read on the host
      between the two routines — see {!scaled_step}) and grows it after a run of good steps. bf16
      has f32's exponent range and skips loss scaling entirely.
    - {b Everything else} stays {!Precision_policy}'s job: apply a policy with [param_prec = None]
      (masters and twins already carry [Specified] precisions, so any [param_prec] would be a no-op
      on them anyway) to assign activation and gradient storage precisions, with the [except]
      predicate pinning precision-sensitive ops (softmax, norms, losses) at the default. *)

open Ocannl_tensor

(** Where a parameter's reduced-precision cast twin lives. *)
type twin_placement =
  | Twin_auto  (** Leave the twin's memory mode to the compiler (virtualization heuristics). *)
  | Twin_virtual  (** Recompute the cast at every read site. *)
  | Twin_materialized  (** A device-resident copy, refreshed once per step by the forward code. *)

val cast_param :
  ?placement:twin_placement -> ?master_prec:Ir.Ops.prec -> prec:Ir.Ops.prec -> Tensor.t -> Tensor.t
(** [cast_param ~prec p] wraps a differentiable parameter [p] with a reduced-precision cast twin and
    returns the twin; non-differentiable parameters are returned unchanged. The master [p] and its
    gradient are pinned at [master_prec] (default [Ops.single]) — pinned, not left alone, because
    parameters are [top_down_prec]: the cast op registers an [Inferred] top-down update of the
    twin's precision into the master, and only a [Specified] precision overrides it (same finding as
    {!Precision_policy}'s [except] semantics). A master whose precision is already [Specified] —
    e.g. a [wrap_param]/[reshape_param] over a non-f32 ndarray — keeps its own storage precision (it
    is safe from the top-down demotion for the same reason), so [master_prec] only applies to
    parameters whose precision is still open. *)

val with_master_weights :
  ?placement:twin_placement -> ?master_prec:Ir.Ops.prec -> prec:Ir.Ops.prec -> (unit -> 'a) -> 'a
(** [with_master_weights ~prec f] runs the model-building thunk [f] with a
    {!Tensor.param_postprocess} hook installed that gives every differentiable parameter a
    [prec]-precision cast twin (see {!cast_param}). Wrap the model {e construction} — the [()]
    application that creates the inline parameters — not just the application to inputs. The
    optimizer path is unchanged: [loss.params] still contains the f32 masters, so
    [Train.sgd_update], [Train.init_params] and {!Persistence} all keep operating on the canonical
    full-precision values. *)

(** Dynamic loss-scale state, host-managed: the scale (and its reciprocal) live in tiny
    device-resident tensors embedded in the compiled routines, and are overwritten via
    [Context.set_values] when the schedule changes the scale — no recompilation. The schedule is
    torch-AMP's: multiply the scale by [backoff_factor] whenever a step produced non-finite
    gradients (the step is skipped), and by [growth_factor] after [growth_interval] consecutive good
    steps. *)
module Loss_scaler : sig
  type t = private {
    scale : Tensor.t;  (** Pass as [Train.grad_update ~loss_scale]. *)
    unscale : Tensor.t;  (** The reciprocal; pass as [Train.sgd_update ~grad_unscale]. *)
    mutable scale_val : float;
    mutable good_steps : int;
    growth_factor : float;
    backoff_factor : float;
    growth_interval : int;
  }
  (** Read-only outside this module: the schedule below is the only writer, and the device copies of
      the scale follow it only through {!set_scale}. *)

  val create :
    ?init_scale:float ->
    ?growth_factor:float ->
    ?backoff_factor:float ->
    ?growth_interval:int ->
    unit ->
    t
  (** Defaults: [init_scale = 65536.], [growth_factor = 2.], [backoff_factor = 0.5],
      [growth_interval = 200]. *)

  val scale_value : t -> float
  (** The current scale, as the host last set it. *)

  val set_scale : t -> Context.t -> float -> Context.t
  (** Sets the scale to [v] and writes it (and its reciprocal) to the device tensors. *)

  val update : t -> Context.t -> grads_finite:bool -> Context.t
  (** One step of the schedule: backoff on non-finite gradients, growth after [growth_interval]
      consecutive good steps. *)

  val update_n : t -> Context.t -> grads_finite:bool -> steps:int -> Context.t
  (** Like {!update}, crediting [steps] steps at once — for the fused gated recipe ({!gated_step}),
      whose host only samples the checksum every [check_interval] steps: a finite sample credits the
      whole window toward growth, a non-finite one backs off once. Growth consumes [growth_interval]
      from the credited count and keeps the remainder (growing once per full interval contained in
      it), so the average growth cadence matches the per-step schedule even when the sampling
      interval does not divide [growth_interval]. *)

  val save_state : t -> string
  (** The scaler's host state -- the current scale, the good-step count toward the next growth, and
      the schedule's parameters, floats in exact hexadecimal -- as one line, for a resumable
      checkpoint to carry in [Checkpoint.progress]'s [extra] under a key of the loop's choosing
      (gh-ocannl-1726). The device copies of the scale are inputs the routines only read, so
      [Checkpoint.training_state] does not hold them, and a scaler re-created at a resume would
      restart from [init_scale] with no good steps: the resumed run would skip and grow on other
      steps than the saved one. *)

  val restore_state : t -> Context.t -> string -> Context.t
  (** [restore_state t ctx s] puts the state {!save_state} encoded in [s] back into [t], a scaler
      created afresh by the resumed program, and writes the restored scale (and its reciprocal) to
      the device tensors, as {!set_scale} does. Raises [Failure] when [s] is not such an encoding,
      or was saved under other [growth_factor], [backoff_factor] or [growth_interval] values than
      [t]'s (naming both): the schedule belongs to the program, and adopting or overriding it
      silently would not continue the saved run. *)
end

val scaled_grad_update :
  ?setup_for_parallel:bool ->
  ?accum_loss:Tensor.t ->
  Loss_scaler.t ->
  Tensor.t ->
  Tensor.t * Ir.Assignments.comp
(** [scaled_grad_update scaler loss] is {!Train.grad_update} seeded with the scaler's scale,
    followed by the {!Train.grad_checksum} reduction, as one computation; returns the checksum flag
    tensor alongside. Parameter gradients are materialized here because the optimizer half
    ({!scaled_sgd_update}) is compiled as a separate routine — the host must read the checksum in
    between — and reads them across the routine boundary.

    This module's update wrappers are a weight-training recipe: the checksum, the materialization
    and the step all cover {!Train.trainable_params}, and must agree. To include a
    [~grad_spec:Require_grad] leaf (input optimization), compose the {!Train} primitives directly:
    {!Train.grad_update}[ ~loss_scale], then {!Train.grad_checksum} and {!Train.sgd_update}
    [~grad_unscale] with the same [?params] (gh-ocannl-1743). *)

val scaled_sgd_update :
  Loss_scaler.t ->
  learning_rate:Tensor.t ->
  ?momentum:float ->
  ?weight_decay:float ->
  ?nesterov:bool ->
  Tensor.t ->
  Ir.Assignments.comp
(** {!Train.sgd_update} with the scaler's reciprocal as [grad_unscale]: gradients are unscaled in
    place before the optimizer math, so the momentum buffer and any later gradient reader see true
    gradient magnitudes. *)

val scaled_step :
  scaler:Loss_scaler.t ->
  grad_routine:Context.routine ->
  sgd_routine:Context.routine ->
  checksum:Tensor.t ->
  Context.t ->
  Context.t * bool
(** One dynamically-scaled training step: run the gradient routine (compiled from
    {!scaled_grad_update}'s computation), read the gradient checksum on the host, and either run the
    optimizer routine (compiled from {!scaled_sgd_update}'s) or — on non-finite gradients — skip it;
    then let the scaler adjust the scale. Returns whether the optimizer step ran. The per-step
    [Context.get_values] awaits the device, so this recipe trades stream overlap for the inf/nan
    gate — the same trade torch-AMP's unscale-and-check makes. *)

(** {2 The fused gated recipe (gh-ocannl-492 task 5)}

    {!scaled_step}'s per-step host read costs a full device await AND a routine split; the checksum
    reduction itself is a full sweep over every gradient. The fused recipe removes the per-step host
    round-trip: the inf/nan gate is evaluated {e on device} — the checksum flag is range-tested into
    a broadcastable 0/1 [update_gate], and every optimizer-state mutation selects through it
    ([Train.sgd_update ~update_gate], [Where] selection so skipped steps leave parameters and
    momentum buffers untouched exactly) — so the whole step is one routine and steps queue without
    synchronizing. The host only samples a {e sticky} window checksum every [check_interval] steps
    to drive the dynamic-scale schedule: [window_checksum] accumulates the per-step flag and
    non-finite values are absorbing (inf/nan never cancel back to finite), so a single overflow
    anywhere in the window is caught at the next sample and backs the scale off. Overflowing steps
    inside a window are already skipped on device — delayed sampling delays only the
    {e scale adjustment}, never poisons state.

    The finiteness test is a range comparison ([-3e38 < flag < 3e38]) rather than [x <> x] or
    [x - x = 0]: the latter fold away under the fast-math flags some GPU backends compile with,
    while an ordered hardware compare of a runtime value against a constant is IEEE-honest about inf
    and nan on every backend (nan fails both ordered compares, so it gates to 0). *)

val gated_scaled_update :
  ?setup_for_parallel:bool ->
  ?accum_loss:Tensor.t ->
  Loss_scaler.t ->
  learning_rate:Tensor.t ->
  ?momentum:float ->
  ?weight_decay:float ->
  ?nesterov:bool ->
  Tensor.t ->
  Tensor.t * Ir.Assignments.comp
(** [gated_scaled_update scaler ~learning_rate ... loss] builds the whole dynamically-scaled
    training step as ONE computation: scaled backprop, gradient checksum, the on-device gate, and
    the gated+unscaled SGD step. Returns [(window_checksum, comp)]; compile [comp] once and drive it
    with {!gated_step}. *)

val gated_step :
  scaler:Loss_scaler.t ->
  routine:Context.routine ->
  window_checksum:Tensor.t ->
  check_interval:int ->
  step:int ->
  Context.t ->
  Context.t * bool
(** One fused gated step: run the single routine; every [check_interval]-th step, read the sticky
    window checksum, reset it, and let the scaler adjust (backoff if any step of the window
    overflowed — those steps already skipped themselves on device — growth crediting the whole
    window otherwise). Returns [(ctx, window_finite)] — [window_finite] is [true] on non-sampling
    steps. [step] is 0-based. *)

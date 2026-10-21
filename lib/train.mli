(** {1 Training recipes}

    The user-facing training layer over the context API ({!Context}): building the forward, backprop
    and optimizer computations of a loss, compiling and running them, the training-loop utilities
    (learning-rate schedules, gradient clipping, outlier detection), placement autotuning, and
    printing tensors from a context. *)

open Ocannl_tensor

(** {2 Shorthands} *)

(** Precisions and debug toggles, for configuring a session from code. *)
module CDSL : sig
  val half : Ir.Ops.prec
  val single : Ir.Ops.prec
  val double : Ir.Ops.prec

  val virtualize_settings : Ir.Low_level.virtualize_settings
  (** The process-wide virtualization settings ({!Ir.Low_level.virtualize_settings}). *)

  val enable_all_debugs : ?debug_logs:bool -> ?hosted_only:bool -> unit -> unit
  (** Raises the log level to at least 2 and turns on the build-directory debug files; with
      [hosted_only] (default [true]) disables device-only virtualization, and with [debug_logs]
      (default [false]) turns on logging from routines. *)

  val disable_all_debugs : ?restore_defaults:bool -> unit -> unit
  (** Undoes {!enable_all_debugs}; [restore_defaults] (default [false]) also re-enables device-only
      virtualization. *)
end

(** Index-binding shorthands over {!Ir.Indexing}. *)
module IDX : sig
  val empty : 'a Ir.Indexing.bindings
  (** No static indices. *)

  val get_static_symbol :
    ?static_range:int ->
    (int -> 'a) Ir.Indexing.bindings ->
    Ir.Indexing.static_symbol * 'a Ir.Indexing.bindings

  val find_exn : Ir.Indexing.lowered_bindings -> Ir.Indexing.static_symbol -> int ref
end

val run : Context.t -> Context.routine -> unit
(** Runs [routine] in [ctx], discarding the resulting context. *)

(** {2 Memory-mode intent} *)

val set_materialized : Ir.Tnode.t -> unit
(** Declares the node materialized: device-resident, persisting across routine runs and readable on
    demand via the context. *)

val set_virtual : Ir.Tnode.t -> unit
(** Declares the node virtual: recomputed at every read site, never stored. *)

val every_non_literal_materialized : Tensor.t -> unit
(** Materializes every non-literal embedded tensor node of [t] (so its value is inspectable on
    demand via the context). Replaces the old [every_non_literal_on_host] now that there is no
    hosted memory mode (gh-ocannl-333). *)

(** {2 Forward, backprop and the optimizer} *)

val forward : Tensor.t -> Ir.Assignments.comp
(** Sets the tensor's value as materialized (device-resident, inspectable on demand via the
    context), and returns the tensor's forward code with a label-derived comment. *)

val loss_accumulator : ?label:string -> unit -> Tensor.t
(** A scalar non-differentiable accumulator for {!grad_update}'s [?accum_loss]: zero-initialized at
    allocation and materialized. Read it with [Context.get_values] (which awaits the device) and
    reset it with [Context.set_values ctx t.value [| 0. |]] — e.g. once per epoch. *)

val trainable_params : Tensor.t -> (Tensor.t, Tensor.comparator_witness) Base.Set.t
(** The subset of [loss.params] that [loss]'s backprop actually trains: the parameters whose
    gradient the backprop code writes ([Asgns.collect_written]). [t.params] deliberately answers a
    broader question — which parameter leaves the forward graph reads, hence what {!init_params}
    must initialize and {!Persistence} must save — so a parameter detached behind
    {!Operation.stop_gradient} (a frozen backbone) stays in [loss.params] while its gradient is
    neither zeroed nor written. Stepping it anyway would apply weight decay to weights the user
    froze (gh-ocannl-673); the optimizer-side helpers ({!sgd_update}, {!grad_l2_norm},
    {!clip_by_global_norm}, {!grad_checksum}, {!zero_params_grads}) therefore derive this set
    instead of trusting [loss.params]. Empty when [loss] is not differentiable. *)

val grad_update :
  ?setup_for_parallel:bool ->
  ?accum_steps:int ->
  ?accum_loss:Tensor.t ->
  ?loss_scale:Tensor.t ->
  Tensor.t ->
  Ir.Assignments.comp
(** Returns the tensor's forward, zeroing gradients, and backprop code wrapped with label-derived
    comments. Sets the tensor's value as materialized. If [setup_for_parallel] is true (false by
    default), sets the trained parameters' gradients as "non-local" (on-device). When [accum_loss]
    is given (see {!loss_accumulator}), the update also accumulates the loss value into it
    ([accum_loss =+ loss]): training loops can then read the loss sum once per epoch instead of once
    per step — on GPU backends a per-step [Context.get_values] awaits the whole device, serializing
    the stream, while steps that only accumulate on device queue up and overlap with host-side
    scheduling. When [loss_scale] is given (see {!Mixed_prec.Loss_scaler}), the backprop is seeded
    with the scale's value instead of 1 ([loss.grad =: loss_scale]), so all gradients come out
    multiplied by the scale — unscale them before the optimizer update (the [grad_unscale] argument
    of {!sgd_update}).

    When [accum_steps] is given (gh-ocannl-465), the returned code is a {e micro-step} of gradient
    accumulation, llm.c-style: parameter gradients are NOT zeroed here (they are materialized so
    they persist across runs, and each micro-step's backprop [=+]-accumulates into them — run
    {!zero_params_grads} at the start of each accumulation cycle instead), while intermediate
    gradients are still zeroed every micro-step; and the backprop seed is pre-scaled by
    [1/accum_steps] (folded with [loss_scale] if both are given), so after [accum_steps] runs the
    parameter gradients hold the mean of the micro-batch gradients — matching a single batch
    [accum_steps] times larger under a mean-reduced loss. Run the optimizer step once per cycle,
    after the last micro-step. *)

val zero_params_grads :
  ?params:(Tensor.t, Tensor.comparator_witness) Base.Set.t -> Tensor.t -> Ir.Assignments.comp
(** Code zeroing the trained parameters' gradients ({!trainable_params}, or [?params]), as a
    standalone computation: the gradient-accumulation counterpart of {!grad_update}[ ~accum_steps]
    (which deliberately does not zero them). Compile it as its own routine and run it at the start
    of each accumulation cycle — before the first micro-step, llm.c-style ("we're about to +=
    accumulate into them"). *)

val grad_checksum :
  ?params:(Tensor.t, Tensor.comparator_witness) Base.Set.t ->
  Tensor.t ->
  Tensor.t * Ir.Assignments.comp
(** A scalar checksum over the trained parameters' gradients ({!trainable_params}, or [?params]) of
    [loss]: returns the flag tensor and the code that resets it to 0 and accumulates the sum of
    every gradient cell into it. The sum is non-finite if and only if some gradient cell is
    non-finite (a finite sum cannot arise from non-finite cells: same-sign infinities stay infinite,
    opposite-sign infinities and NaNs produce NaN; a spurious overflow of large finite gradients
    only triggers a benign extra backoff). Sequence it after {!grad_update} in the same routine,
    read the flag with [Context.get_values] and gate the optimizer step on [Float.is_finite] — the
    dynamic loss scaling recipe ({!Mixed_prec.scaled_step}) does exactly this. *)

val sgd_one :
  learning_rate:Tensor.t ->
  ?momentum:float ->
  ?weight_decay:float ->
  ?nesterov:bool ->
  ?grad_unscale:Tensor.t ->
  ?grad_scale:Tensor.t ->
  ?update_gate:Tensor.t ->
  Tensor.t ->
  Ir.Assignments.comp
(** See: https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/optim.py

    When [grad_unscale] is given (the reciprocal of {!grad_update}'s [loss_scale]), the gradient is
    first multiplied in place by it, so the optimizer math below — including the momentum buffer —
    sees unscaled gradients, and so does any later reader of [p.grad] (e.g. gradient clipping).

    When [grad_scale] is given (a broadcastable scalar, e.g. {!field-grad_clipping.grad_scale} of
    {!clip_by_global_norm}, gh-ocannl-465), the gradient is multiplied by it {e as read} into the
    update — the gradient buffer itself is left untouched (llm.c folds its clipping scale into the
    optimizer kernel the same way): later readers, the next accumulation cycle, and logged gradient
    norms all see the unclipped values, and no extra per-parameter sweep is emitted. The scale
    applies to the gradient only, before weight decay's [p] term joins the delta.

    When [update_gate] is given (a broadcastable scalar holding 1 to apply the step and 0 to skip
    it, computed on device — see [Mixed_prec.gated_scaled_update], gh-ocannl-492 task 5), every
    optimizer-state mutation is gated by [Where] {e selection}: on a skipped step the parameter and
    the momentum buffer keep their previous values exactly. Selection, not multiplication — the
    skipped steps are the ones whose gradients hold [inf]/[nan], and [0 * inf] is [nan]. *)

val sgd_update :
  learning_rate:Tensor.t ->
  ?momentum:float ->
  ?weight_decay:float ->
  ?nesterov:bool ->
  ?grad_unscale:Tensor.t ->
  ?grad_scale:Tensor.t ->
  ?update_gate:Tensor.t ->
  ?params:(Tensor.t, Tensor.comparator_witness) Base.Set.t ->
  Tensor.t ->
  Ir.Assignments.comp
(** Maps {!sgd_one} over the parameters [loss] trains ({!trainable_params}, or [?params]): a
    parameter frozen behind {!Operation.stop_gradient} takes no step — in particular no weight decay
    (gh-ocannl-673). *)

val sequential_loop : f:(unit -> unit) -> (Ir.Indexing.static_symbol * int ref) list -> unit
(** All and only bindings with associated ranges are iterated, with the binding's initial value
    lost. Bindings without ranges remain at their initial values, as do symbolic extents (gh-490):
    an extent is a size set once by the user, not an index to iterate.

    [f] need not wait for the device: {!Context.run} reads the bindings at the dispatch, so the next
    iteration may rebind them immediately whatever the backend schedules asynchronously. *)

(** {2 Training-loop utilities (gh-ocannl-465)}

    Host-side learning-rate schedules, global-norm gradient clipping, and a loss/grad-norm outlier
    detector — ports of llm.c's loop scaffolding ([llmc/schedulers.h], [llmc/global_norm.cuh],
    [llmc/outlier_detector.h]); the gradient-accumulation piece is {!grad_update}[ ~accum_steps] +
    {!zero_params_grads} above. *)

(** Host-side learning-rate schedules: pure functions from the step number to a float, fed to the
    device via {!scheduled_learning_rate} (or any host-written scalar). All schedules start with a
    linear warmup over [warmup_steps] steps ([base_lr * (step+1) / warmup_steps]; no warmup when 0)
    and decay toward [base_lr *. final_frac] at [total_steps]. Steps beyond [total_steps] clamp to
    the final value. *)
module Lr_schedule : sig
  type kind =
    | Constant  (** [base_lr] after warmup; [final_frac] is ignored. *)
    | Cosine  (** Half-cosine from [base_lr] down to [base_lr *. final_frac]. *)
    | Linear  (** Straight line from [base_lr] down to [base_lr *. final_frac]. *)
    | Wsd of { decay_frac : float }
        (** Warmup-stable-decay (arXiv:2405.18392): hold [base_lr] until the final [decay_frac]
            fraction of [total_steps] (llm.c uses 0.2), then decay as [1 - sqrt(ratio)]. *)

  type t = {
    kind : kind;
    base_lr : float;
    warmup_steps : int;
    total_steps : int;
    final_frac : float;  (** The final learning rate as a fraction of [base_lr]. *)
  }

  val learning_rate : t -> step:int -> float
  (** The learning rate at [step] (0-based). *)
end

val host_scalar : l:string -> float -> Tensor.t
(** A device-resident, broadcastable scalar the host can overwrite with [Context.set_values].
    Data-backed on purpose: a 1-element [term_init] would come out as a [Constant] fetch, re-fetched
    by every step's forward code, silently undoing [Context.set_values]. The [bcast_if_1] axis basis
    (as in [Tensor.number]) lets the scalar broadcast into tensors of any shape. *)

val scheduled_learning_rate :
  ?label:string -> Lr_schedule.t -> Tensor.t * (Context.t -> step:int -> Context.t)
(** A learning-rate scalar driven by a host-side schedule: returns the tensor (pass it as
    {!sgd_update}'s [~learning_rate]) and a setter overwriting it with the schedule's value at
    [step] — call the setter once per step, before running the optimizer routine. The overwrite is a
    tiny host-to-device transfer, not a recompilation. *)

val grad_l2_norm :
  ?grad_unscale:Tensor.t ->
  ?label:string ->
  ?params:(Tensor.t, Tensor.comparator_witness) Base.Set.t ->
  Tensor.t ->
  Tensor.t * Ir.Assignments.comp
(** A scalar holding the global L2 norm over the gradients of the parameters [loss] trains
    ({!trainable_params}, or [?params]): returns the norm tensor (materialized and observable — read
    it with [Context.get_values] for logging or {!Outlier_detector} feeding) and the code that
    computes it: per-parameter sum-of-squares einsum reductions (deterministic by construction —
    OCANNL emits no atomics) followed by a square root. Sequence it after {!grad_update} in the same
    routine or a later one. Like {!grad_checksum}, it must be called only after the model and the
    loss are fully constructed.

    When [grad_unscale] is given (the reciprocal of a {!Mixed_prec.Loss_scaler}'s scale), the norm
    is multiplied by it after the square root, so the result is the true-magnitude norm even when
    backprop was seeded with a loss scale (the buffers themselves still hold scaled gradients at
    this point — {!sgd_one} unscales them in place later). *)

type grad_clipping = {
  grad_norm : Tensor.t;
      (** The pre-clip global L2 norm (observable — read it for logging or outlier detection). *)
  grad_scale : Tensor.t;
      (** The clipping scale, computed on device: [min(1, max_norm / grad_norm)]. Pass it as
          {!sgd_update}'s [~grad_scale] so it folds into the update. *)
  clip_comp : Ir.Assignments.comp;
      (** Sequence it after the gradient update and before the optimizer step (all three can be one
          routine — no host round-trip is involved). *)
}
(** What {!clip_by_global_norm} returns. *)

val clip_by_global_norm :
  ?grad_unscale:Tensor.t ->
  ?label:string ->
  ?params:(Tensor.t, Tensor.comparator_witness) Base.Set.t ->
  max_norm:float ->
  Tensor.t ->
  grad_clipping
(** Global-norm gradient clipping, llm.c-style ([llmc/global_norm.cuh] feeding [grad_scale] into the
    AdamW launch): the returned scale leaves gradient buffers untouched and multiplies the gradients
    as the optimizer reads them ({!sgd_update}[ ~grad_scale]).

    Behavior at the edges of the finite range, deliberately (matching llm.c's
    [grad_scale = grad_clip / grad_norm], which shares both properties): a [nan] norm fails the
    ordered comparison and selects scale 1 — clipping does not gate non-finite gradients, combine
    with {!grad_checksum} or [Mixed_prec.gated_scaled_update] for inf/nan defense. An {e infinite}
    norm — including the overflow of squaring a finite f32 gradient component above ~1.8e19 (the
    accumulator is f32 and narrow-storage gradients widen at load, so storage precision does not
    lower that threshold) — makes the ratio 0, suppressing the gradient term entirely: the step
    degrades to weight decay alone, strictly more conservative than rescaling an explosion of that
    magnitude to [max_norm], and self-recovering since the buffers are untouched. *)

(** A sliding-window z-score outlier detector for host-observed scalars (per llm.c's
    [llmc/outlier_detector.h]): feed it the per-step loss and/or {!grad_l2_norm} values and skip the
    optimizer step when the returned z-score exceeds a threshold. Pure host-side state; use one
    detector per monitored quantity. *)
module Outlier_detector : sig
  type t
  (** The sliding window and its fill state. *)

  val create : ?window_size:int -> unit -> t
  (** A detector over a window of [window_size] samples (default 128, must be positive). *)

  val update : t -> float -> float
  (** Returns [v]'s z-score against the sliding window of the {e previously} recorded values, then
      records [v] (replacing the oldest sample): [Float.nan] until the window has filled — treat
      that as "not an outlier". Three deliberate departures from llm.c's [update_detector]: the
      score is computed {e before} [v] joins the window, since self-inclusion dilutes the baseline
      and caps any finite spike's score at [sqrt (n - 1)] — for a small window that bound sits below
      reasonable thresholds; a non-finite [v] never enters the window and scores [infinity], so any
      finite threshold flags it and the update is skipped, while later samples still get a healthy
      baseline; and the moments are recomputed from the stored window on normalized values instead
      of llm.c's running [sum]/[sum_sq] — the [E[x^2] - E[x]^2] form cancels catastrophically for a
      window with a large common offset and small variance, and even a centered sum can overflow
      once samples near the float maximum have been recorded, turning later scores into 0 or [nan] —
      false passes. Normalization goes first so every intermediate is bounded:
      [mag = max_i (abs x_i)] is a pure maximum (cannot overflow), the mean is accumulated as
      [x /. mag /. n] (partial sums within [[-1, 1]]), and deviations [x/mag - mean/mag] lie in
      [[-2, 2]] — no finite window can overflow any of it; O(window) per step is free on the host
      where llm.c needed O(1) in a kernel-adjacent loop. A genuinely constant-valued window has
      standard deviation 0, making the z-score of any deviation [infinity] (and of [v = mean], 0).
  *)
end

(** {2 Compiling and running} *)

val memory_budget_setting : unit -> Memory_budget.t option
(** gh-ocannl-498 rematerialization: the configured device-memory budget for a compiled routine, or
    [None] when there is none (the default — under which the planning pass never runs and
    compilation is bit-for-bit what it was). The setting is a byte count with an optional K/M/G
    suffix (powers of 1024), the word [minimize], or 0 / off / false / none. *)

val to_routine :
  Context.t ->
  ?output_cd_file:bool ->
  ?budget:Memory_budget.t ->
  ?max_candidates:int ->
  ?budget_report:(Memory_budget.plan -> unit) ->
  Ir.Indexing.unit_bindings ->
  Ir.Assignments.comp ->
  Context.t * Context.routine
(** Compiles [comp] and returns the post-compile context together with the routine. [budget],
    [max_candidates] and [budget_report] are the gh-ocannl-498 rematerialization seam: [comp]'s
    inlining decision vector is planned by {!Memory_budget.fit} against [budget], or else against
    config [memory_budget] ({!memory_budget_setting}). With neither, nothing is planned and the
    compile is exactly what it was; [budget_report], if given, observes the plan that shipped.

    The post-compile context is returned rather than discarded (gh-ocannl-772), matching
    {!Context.compile} and {!run_once}: chain it into the next compile instead of reaching into
    [routine.context] for the same value. A caller that only wants the routine can [snd] this. *)

val init_params :
  ?reinit_all:bool -> Context.t -> Ir.Indexing.unit_bindings -> Tensor.t -> Context.t
(** [init_params] initializes the parameters of [t], via running their forward code or copying from
    the host as appropriate. If [reinit_all] is true, all parameters are reinitialized, otherwise
    only the parameters that are not in [ctx.ctx_buffers] are initialized. *)

val run_once :
  ?output_cd_file:bool ->
  ?skip_init:bool ->
  ?reinit_all:bool ->
  ?bindings:Ir.Indexing.unit_bindings ->
  ?budget:Memory_budget.t ->
  ?max_candidates:int ->
  ?budget_report:(Memory_budget.plan -> unit) ->
  f:(Tensor.t -> Ir.Assignments.comp) ->
  Context.t ->
  Tensor.t ->
  Context.t
(** [run_once] is a wrapper around {!init_params} that additionally runs code of [f t] and returns
    the context. If [skip_init] is true (false by default), no initialization is performmed. If
    [reinit_all] is true (false by default), all parameters are reinitialized, otherwise only the
    parameters that are not in [ctx.ctx_buffers] are initialized.

    If [output_cd_file] is true, the global setting [output_debug_files_in_build_directory] must be
    true, and the update code is output to a file before shape inference potentially crashes at
    [init_params]. *)

val forward_once :
  ?output_cd_file:bool ->
  ?skip_init:bool ->
  ?reinit_all:bool ->
  ?bindings:Ir.Indexing.unit_bindings ->
  ?budget:Memory_budget.t ->
  ?max_candidates:int ->
  ?budget_report:(Memory_budget.plan -> unit) ->
  Context.t ->
  Tensor.t ->
  Context.t
(** [forward_once] is a wrapper around {!run_once} that runs the forward code of [t]. *)

val update_once :
  ?output_cd_file:bool ->
  ?skip_init:bool ->
  ?reinit_all:bool ->
  ?bindings:Ir.Indexing.unit_bindings ->
  ?budget:Memory_budget.t ->
  ?max_candidates:int ->
  ?budget_report:(Memory_budget.plan -> unit) ->
  Context.t ->
  Tensor.t ->
  Context.t
(** [update_once] is a wrapper around {!run_once} that runs the gradient update code of [t]: both
    forward and backprop. *)

(** {2 Placement autotuning} *)

(** Which placement arm {!tune_placements} ships (gh-ocannl-638).

    [Measured_winner] is the default and the only setting a normal run should use: both arms are
    timed and the faster one ships. The two forcing settings are {e measurement-only}. They ship a
    chosen arm whatever the timings said, which is how a measurement gets {e executed} values out of
    the artifact it profiles rather than out of whichever artifact happened to win — the gap
    benchmarks/report-gh612-hip.md had to state in its verdict, where three of four cells shipped
    arm B while every ratio was computed on arm A's never-executed routines.

    Forcing changes what ships, not what is measured: both arms are still searched, so the A-vs-B
    comparison a report quotes stays available and the positional [?report] contract is untouched. A
    forced arm that {e failed} has no fallback — its failure propagates rather than the other arm
    shipping in its place, since the caller asked for that artifact and there is none. *)
type placement_arm = Measured_winner | Force_arm_a | Force_arm_b

val placement_arm_of_string : source:string -> string -> placement_arm
(** Parses the [tune_ship_arm] spelling of a {!placement_arm}. [source] names what is being parsed,
    for the error message. *)

type shipped = {
  ship_arm : string;
      (** ["A"] (the default placements), ["B"] (materialize-all) or ["flip"] (a flip-refined
          placement vector); on a placement-store replay, the recorded decision's
          ({!Ir.Schedule_cache.shipped_label}). *)
  ship_replayed : bool;
      (** The placement is the placement-decision store's recorded decision, replayed: the call ran
          one search, from the decision's context, instead of the arms. Whether that search's
          SCHEDULE was itself replayed is its report's [outcome] ({!Autotune.Cache_replay}). *)
  ship_routine : string;
      (** The returned routine's own name ([Context.routine]'s [name]), read off the routine
          delivered. *)
  ship_report : Autotune.report option;
      (** The report of the search whose compile shipped -- the same value [report] or [flip_report]
          received: its [source_digest] is the base lowering that search tuned and the digest its
          schedule-cache entry is stored under, its [best_label] the crown it compiled. On a
          fallback (the winner did not compile, or compiled the untuned default in its place) it is
          the fallback's. [None] only for the last resort, the caller's own untuned compile shipping
          as arm A after every completed result failed to compile. One case ships other code than
          the report's crown under it: a compile that substituted the untuned default
          ({!Autotune.shipped_substitute}) with nothing left to fall back to (a forced arm, or the
          last completed result). *)
  ship_compiles : int;
      (** The backend compiles ({!Context.backend_compiles}) the call performed outside its
          searches, whose own each search's report counts ([Autotune.report.compiles]) -- that is,
          compiling what it returned, a shipping compile that failed or was substituted before it
          included (gh-ocannl-1440). 0 when the shipped schedule was a schedule-cache hit: the
          routine is the code the search's validating replay compiled, linked again
          ({!Context.relink_outcome}). 1 for a searched winner, recompiled from its saved form in
          the caller's lineage. *)
}
(** What {!tune_placements} shipped, as its [on_ship] callback receives it (gh-ocannl-1438): the
    identity of the returned artifact, which the arm reports alone do not determine (see
    {!tune_placements}). *)

val tune_placements :
  ?name:string ->
  ?beam_width:int ->
  ?rounds:int ->
  ?repeats:int ->
  ?cache_dir:string ->
  ?timing_ctx:Context.t ->
  ?report:(Autotune.report -> unit) ->
  ?flip_report:(Autotune.report -> unit) ->
  ?inline_flips:int ->
  ?ship_arm:placement_arm ->
  ?placement_store:bool ->
  ?call_budget:float ->
  ?log:bool ->
  ?progress:bool ->
  ?on_ship:(shipped -> unit) ->
  Context.t ->
  Tensor.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Context.t * Context.routine
(** Placement A/B autotuning: {!Autotune.tune} on [comp] under the graph's current (default)
    placements — virtual intermediates plus the compiler's promotions — and again with every
    embedded node of [loss] materialized, keeping the measured winner (the arms' [best_ms] are
    min-of-N timings on the same device, so directly comparable). Under the default
    [ship_arm = Measured_winner] the result is by construction at least as fast as the better of the
    default and materialize-all placements, whichever the search would find; this generalizes the
    old "materialize everything before tuning" recipe instead of replacing one fixed placement
    policy with another. {b That guarantee is exactly what the other [ship_arm] settings give up}
    (gh-ocannl-638): a forced arm ships whether or not it was the faster one, which is the point —
    they exist so a measurement can execute the artifact it profiles — so every "keeps the winner"
    and "at least as fast" statement here is about [Measured_winner] alone. Respecting the two-level
    memory-mode split (docs/proposals/context-scoped-memory-modes.md) — tnode-level [memory_mode] is
    declared, semantics-bearing intent, while placement {e decisions} are context-level and
    functional — the B arm does not touch intent: it tunes from {!Context.decide_materialized}
    siblings of [ctx] (and of [timing_ctx]), so the arms are hermetic and [tune_placements] leaves
    no trace on the graph or on the caller's contexts beyond the returned winner. See
    test/operations/materialize_after_compile.ml. [report], when given, observes both arms' reports
    in order — arm A first, then arm B — so a consumer holding both reports can attribute every
    per-arm fact to the arm that produced it. What the reports do {e not} determine is which arm
    SHIPPED: read [on_ship] for that. The two came apart in stages — a winning flip refinement ships
    a placement vector that is neither arm (gh-ocannl-555), and [ship_arm] (gh-ocannl-638) overrides
    the time comparison outright — so "the smaller [best_ms] shipped" is no longer a rule a consumer
    can apply, and applying it is exactly the misattribution [on_ship] exists to prevent. The
    reports are measurements of two searches; [on_ship] is the identity of the returned artifact.
    That separation is what makes "a [Schedule.Tensorize] was crowned in an arm that did not ship"
    reportable (gh-ocannl-546): [best_tensorized] on the other arm's report, with [mma_best_ms]
    against [best_ms] for the margin. The same conclusion is logged here under config
    [autotune_log]. [log] and [progress] override configs [autotune_log] and [autotune_progress] for
    the whole call -- its own arm lines and every search it runs -- as {!Autotune.tune}'s do for one
    search. Other arguments are forwarded to {!Autotune.tune}; the same caveats apply (notably
    [timing_ctx] and non-idempotent routines — both arms share [timing_ctx]'s device for their
    searches). [name] included (gh-ocannl-669): it names both arms' compiles and the flip
    refinement's decision-surface lowerings, and is what lets a comp carrying no
    {!Ir.Assignments.Block_comment} — one {!Context.compile} would name at the call site — be tuned
    here at all. The whole call is one search ({!Autotune.exclusive_search}, gh-ocannl-1352): made
    while another domain searches, it raises {!Autotune.Concurrent_search} before doing anything.

    [call_budget] (config [tune_call_budget_seconds], gh-ocannl-1440; zero or less means none) is a
    wall-clock budget for the whole call, on one clock from its start, the placement-store replay
    included. Each search the call starts gets a share ({!Autotune.call_budget}): what is left of
    the budget divided among the searches still planned -- the arms not yet searched plus the flip
    slots still planned ([inline_flips], or the decision surface's alternatives if fewer) -- and is
    cut at the tighter of its share and [autotune_budget_seconds] by that key's rule. Both arms are
    always searched: they are the call's floor, a spent budget leaving arm B only its untuned
    floors, and the positional [report] contract holds. The flip refinement stops before any further
    alternative once the budget is spent. The budget is cooperative: compiles and timing windows
    cannot be cancelled, so the call can overrun it by each search's own overrun, by arm B's floors
    when arm A spent everything, and by the shipped routine's compile. Every report carries the
    call's budget and the search's share ([call_budget_s], [call_share_s]); a decision reached over
    a cut search is not recorded in the placement store.

    An arm that fails is a {e losing} arm, not a failed run (gh-ocannl-550): a search that
    terminates on a fatal failure ranks at [infinity], the other arm's completed winner ships and
    stays cached, and the failed arm's own report — [Autotune.Search_died], carrying the failure —
    still reaches [report] in position, so the failure is recorded rather than downgraded to "that
    arm merely lost". A failed arm's [best_ms] is deliberately {e not} shippable and not compared:
    its search raised, so there is no routine to compile from [ctx] for it. Only when every arm
    fails does [tune_placements] propagate, with the first failure's original backtrace — with two
    exceptions that are not the arm's to absorb and propagate at once: process-level failures
    ([Out_of_memory], [Sys.Break]) and compiler invariant violations ([Assert_failure],
    [Stack_overflow]), the same classes {!Ir.Schedule_outcome.classify_raw} refuses to classify. A
    [report] callback's own exception likewise propagates rather than counting as an arm failure,
    and so does a failure that poisoned the lineage the arms share ({!Context.poisoned_failure}):
    every timing run in the sibling would then refuse to execute, so the second arm can only burn a
    search proving it. Consumers attributing arms by arrival order should read
    {!Autotune.terminal_failure} — or match the report's [outcome] — before [best_ms], as
    benchmarks/runners/ocannl/bench_harness.ml does.

    The in-position guarantee is {!Autotune.tune}'s reporting contract, and inherits its one
    carve-out: an argument-precondition violation (an incompatible [timing_ctx]) is detected before
    the call reaches any phase, so it reports nothing and propagates. Both arms are given the same
    contexts, so both raise it; there is no surviving arm to misattribute.

    gh-ocannl-1408: every arm and flip searches through {!Autotune.tune_deferred}, so a search
    leaves nothing it compiled on the device, and only the placement that ships is compiled from
    [ctx] -- once, after every search. Arm A's routine is therefore not resident while arm B
    searches: on a step over half the device it used to be, and arm B was refused at its base
    compile, making every such A/B single-arm. A shipping compile that fails, or that ships the
    untuned default in place of the reported schedule ({!Autotune.shipped_substitute}), falls back
    to the next completed result -- the other arm, or after a flip refinement the A/B winner and
    then the other arm -- which ships unrecorded; under a forced arm a failure propagates and a
    substitute ships unrecorded. The replayed decision's one search compiles at once, so a replay
    that does not compile is a failed replay like any other.

    The arms differ in which candidates {e exist}, not only in how they rank: a tensorized candidate
    is seeded only when the matmul site's operand and destination storage precisions resolve to a
    tile the backend advertises ({!Autotune.tensorized_capability_refutation}), and placement
    decides which nodes the site reads. Under the mixed-precision recipe on a uniform-format backend
    (Metal's simdgroup matrices) that makes arm A tensorization-free: the reduced-precision cast
    twins are virtual there, so the site reads f32 masters into a reduced-precision destination — a
    mixed triple no tile matches — while materialize-all turns the twins into real reduced-precision
    nodes and the seeds fire. Materializing just the twins ([Mixed_prec.Twin_materialized]) reaches
    the same seeds at arm A's cost; see benchmarks/report-gh546-metal.md.

    gh-555: the A/B is the coarse level of the hierarchical inlining search — inlining decided
    first, tiling/scheduling within each arm by the nested {!Autotune.tune}. [inline_flips] (config
    [tune_inline_flips], default 0) adds a greedy per-node refinement level: the default-policy
    arm's compile reports its searchable decision dimensions
    ({!Ir.Low_level.field-flip_candidates}), and the candidates are tried one at a time from arm A's
    context — [Materialize] via {!Context.decide_materialized} (walking toward arm B one node at a
    time), [Inline] via {!Context.decide_inline} — each accepted flip becoming the base for the
    next, and the refined result shipping only if it beats the A/B winner. Every measured flip costs
    a full search like an arm, so the budget is explicit and defaults to zero. Flip searches report
    through [flip_report], not [report]: the positional arm-A-then-arm-B contract of [report] is
    preserved regardless of the budget.

    gh-514, the tuned placement-space search: the chain walks the surface in
    {!Autotune.placement_surface}'s ranking — family-unlocking (enablement) [Materialize] flips
    before cost, per the gh-558 lesson; config [tune_flip_ordering=cost] restores the legacy
    recompute-cost order as the evaluation baseline — weighed, under the default
    [tune_flip_ordering=profitable] (gh-ocannl-579), against what the two arms just MEASURED about
    the family that prior points at ({!Autotune.family_profit_of_reports} over both arm reports: arm
    B is the all-materialized specialization the enablement set is derived from, so its best
    tensorized time against its best time prices the promotion for free). A family measured to lose
    here by more than [tune_flip_profit_margin] voids the prior and the surface ranks by cost — the
    prior models expressibility, and on gh-514's metal/f16 cell promoting a hopeless family took
    budget slots 1-2 and pushed the winning cheap flip out of a budget-5 chain. And, under config
    [autotune_bound_pruning], fathoms a [Materialize] flip pre-search when the roofline floor of the
    chain's partial placement vector extended by it already meets the best measured time
    (admissible: the floor lower-bounds every completion, so the flip cannot win). Fathomed flips do
    not consume the budget, which counts measured flips.

    gh-ocannl-1110: a flip whose best after its first [beam_width] timed candidates trails the
    incumbent's best after the incumbent's own first as many by more than
    {!Autotune.flip_abandon_ratio} is abandoned there ({!Autotune.tune}'s [?abandon]): its report
    reaches [flip_report] as {!Autotune.Abandoned} or, on a cache replay,
    {!Autotune.Abandonment_replay}; it loses, and it counts toward the flip budget. The incumbent's
    record is its report's [best_steps], so an arm whose report has none (a cache entry older than
    the field, or stored under another search shape) leaves the flips to run in full, as does an
    incumbent that failed: its partial record is no shippable routine's.

    gh-ocannl-1274: a flip whose local reading is canonically identical
    ({!Autotune.placement_surface}'s [ps_twin_identity]) to one already measured and lost in this
    chain — abandoned, or searched and not faster than its incumbent — is skipped as [twin-of] that
    flip without a budget slot, logged beside the abandonments and recorded as an [event=flip_twin]
    progress line ([flips_done] counts them as [twinned]). It is rejected like the loss it repeats.
    A failed search and an accepted flip anchor nothing. The rule is a function of what the chain
    measured, so a warm chain whose anchors replay skips the same twins.

    gh-ocannl-638, [ship_arm] (config [tune_ship_arm], default [Measured_winner]): ship a chosen
    {!placement_arm} instead of the measured winner. It exists for measurement — a profile of arm
    A's kernels is evidence about arm A's routine, and until that routine is the one that ships,
    nothing ever executes it against a reference, so a value-changing regression inside it leaves
    every structural and timing figure plausible. Deliberately loud: a non-default setting announces
    itself on stderr regardless of [autotune_log], both when it is resolved and at the decision it
    changes, because it is the optimizer's shipping path. Forcing also skips the flip refinement
    (which walks away from the chosen arm one node at a time, so its result is neither arm), and a
    forced arm that failed propagates its failure rather than falling back to the other arm.

    [on_ship] is called exactly once with the {!shipped} record -- its [ship_arm] ["A"], ["B"] or
    ["flip"] -- on the path that returns a routine, and not at all when nothing ships. It is what a
    consumer should attribute the returned artifact by: deriving the shipped arm from the reports'
    [best_ms] is only valid while nothing can override the comparison, which [ship_arm] now can, and
    it never described a flip-refined result at all.

    gh-ocannl-786, the placement-decision store: what shipped -- the arm, or the flips the
    refinement accepted -- is recorded beside the schedule entries in [cache_dir]
    ({!Ir.Schedule_cache.store_placements}), keyed by the decision problem's structural identity
    ({!Ir.Schedule_cache.canonicalize_source}: the raw program and what the lineage already decided,
    with the decision itself outside the key) under the schedule cache's own key regime, and guarded
    by the digest of the lowering the decision produces. A later process whose problem hits replays
    the decision through {!Context.decide_materialized} / {!Context.decide_inline} /
    {!Context.decide_footprint} and runs ONE search from the resulting context -- normally itself a
    schedule-cache replay -- instead of enumerating candidates and searching both arms and every
    flip. On that path [report] observes exactly one report, the shipped placement's, so the
    positional two-arm contract above holds for cold runs and [on_ship] remains the identity of the
    artifact either way; [flip_report] observes nothing. A replayed search that fails without
    poisoning the lineage is a losing arm one level up: it is not reported, the recorded decision is
    treated as stale, and the two arms are searched and reported in position. An entry is recorded
    only from clean evidence (every observed search completed, none with contention-refused windows
    or cut by its budget, the shipped one having timed something), never under a forced [ship_arm]
    -- which also never consults the store: the override keeps precedence over a recorded decision
    exactly as over a measured one -- and is ignored, and overwritten by the cold run that follows,
    when its decision no longer reproduces the program it was measured on. The directory is resolved
    as {!Autotune.tune} resolves it ({!Autotune.resolve_cache_dir}), [cache_dir = ""] disabling both
    stores.

    gh-ocannl-1020, [placement_store] (config [tune_placement_store], default [true]): [false]
    bypasses the placement-decision store alone -- neither consulted nor recorded, the posture a
    forced [ship_arm] takes, without forcing an arm -- while the schedule cache keeps replaying. On
    a warm schedule cache that is the placement A/B with both arms' schedules replaying: both arms
    report in position and the measured winner ships, which is the comparison a store hit
    short-circuits into one search. *)

val embedded_nodes : Tensor.t -> Ir.Tnode.t list
(** The embedded nodes of [loss]: what {!tune_placements}' materialize-all arm decides
    {!Context.decide_materialized} on. *)

val placement_problem :
  ?name:string ->
  ?timing_ctx:Context.t ->
  Context.t ->
  Tensor.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Ir.Schedule_cache.canonical
(** The identity of the placement decision problem {!tune_placements} answers for [comp] from [ctx]
    (gh-ocannl-786): {!Ir.Schedule_cache.canonicalize_source} over the routine's raw lowering — the
    program, what the lineage already decided about each of its nodes — plus, per node, whether it
    is in [loss]'s embedded set, the set arm B materializes (the same computation tuned against two
    losses poses two problems, since their materialize-all arms differ), and, with a [timing_ctx],
    what THAT lineage brings to the node ({!Ir.Schedule_cache.lineage_tag}): the arms are measured
    in the timing lineage and only the winner is compiled from [ctx], so a timing context inheriting
    other decisions or preferences measures other programs. What the placement-decision store keys
    on; exposed so a test can reason about the key. *)

val placement_outcome_digest :
  ?name:string ->
  ?timing_ctx:Context.t ->
  Context.t ->
  Tensor.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Ir.Schedule_cache.placement_decision ->
  string
(** The replay guard of {!tune_placements}' placement store for [decision], from scratch: the
    decision problem ({!placement_problem}), the decision's lists against it, and the
    placement-aware digest ({!Ir.Schedule_cache.canonicalize}) of the lowering those lists produce
    in the lineage the arms are searched in -- [timing_ctx] when given, else [ctx]. Exposed so a
    test can assert the recomputation agrees with the digest the shipped search tuned
    (gh-ocannl-1022). *)

val placement_evidence_clean : Autotune.report -> bool
(** Whether a search {!tune_placements} observed is evidence a recorded placement decision may rest
    on (gh-ocannl-786's rule): it completed, with no contention-refused window, uncut by its budget
    (gh-ocannl-1409), and measured something. A {!Autotune.Lineage_refused} search measured nothing
    but spoils nothing either (gh-ocannl-1421): its placement can never ship into the caller under
    this decision problem, whose key carries how both lineages' lowerings access each node
    ({!placement_problem}), so the decision does not skip a competitor a later process could measure
    -- up to lineage pairs that differ only in inherited arithmetic, which share a key. The
    recorder's own predicate, exposed so a test judges evidence exactly as the recorder does. *)

(** {2 Printing} *)

val printf :
  ?here:Lexing.position ->
  ?with_grad:bool ->
  ?with_code:bool ->
  ?with_low_level:bool ->
  ?style:Tensor.array_print_style ->
  Context.t ->
  Tensor.t ->
  unit
(** [printf] is a wrapper around {!Tensor.print} that assumes [~force:true], and by default sets
    [~with_code:false], [~with_grad:true], and [~style:`Default]. It takes an explicit context and
    retrieves values on demand (gh-ocannl-333). If the tensor's value is not already materialized in
    [ctx], it is recomputed via the [for_print] copy trick so real values are still shown. *)

val printf_tree :
  ?here:Lexing.position ->
  ?with_value:bool ->
  ?with_grad:bool ->
  ?depth:int ->
  Context.t ->
  Tensor.t ->
  unit
(** [printf_tree] is a wrapper around {!Tensor.print_tree} that assumes [~force:true], and by
    default sets [~with_value:true], [~with_grad:true], and [~depth:9]. It takes an explicit context
    and retrieves values on demand (recomputing via [for_print] if not already materialized). *)

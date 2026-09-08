open Base
module Ops = Ir.Ops
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module Task = Ir.Task
open Ocannl_tensor.Operation.DSL_modules

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_TRAIN=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_TRAIN"]

module CDSL = struct
  let half = Ir.Ops.half
  let single = Ir.Ops.single
  let double = Ir.Ops.double
  let virtualize_settings = Ir.Low_level.virtualize_settings

  let enable_all_debugs ?(debug_logs = false) ?(hosted_only = true) () =
    Utils.set_log_level @@ max 2 @@ Utils.settings.log_level;
    Utils.settings.output_debug_files_in_build_directory <- true;
    if hosted_only then virtualize_settings.enable_device_only <- false;
    if debug_logs then Utils.settings.debug_log_from_routines <- true

  let disable_all_debugs ?(restore_defaults = false) () =
    Utils.settings.debug_log_from_routines <- false;
    Utils.set_log_level 0;
    Utils.settings.output_debug_files_in_build_directory <- false;
    if restore_defaults then virtualize_settings.enable_device_only <- true
end

module IDX = struct
  let empty = Idx.Empty
  let get_static_symbol = Idx.get_static_symbol
  let find_exn = Idx.find_exn
end

let run ctx routine = ignore (Context.run ctx routine)

(* Parameter persistence now lives in {!Persistence} (gh-ocannl-373) and is context-mediated; the
   old hosted-array-based save/restore helpers were removed with the hosted memory mode
   (gh-ocannl-333). *)

let set_materialized (a : Tn.t) = Tn.update_memory_mode a On_device 28

(** Sets the tensor's value as materialized (device-resident, inspectable on demand via the
    context), and returns the tensor's forward code with a label-derived comment. *)
let forward t =
  let fwd = Tensor.consume_forward_code t in
  set_materialized t.Tensor.value;
  let label = Tn.debug_name t.value in
  { fwd with asgns = Asgns.Block_comment (label ^ " fwd", fwd.asgns) }

(** A scalar non-differentiable accumulator for {!grad_update}'s [?accum_loss]: zero-initialized at
    allocation and materialized. Read it with [Context.get_values] (which awaits the device) and
    reset it with [Context.set_values ctx t.value [| 0. |]] — e.g. once per epoch. *)
let loss_accumulator ?(label = "loss_accum") () =
  let t = NTDSL.init ~l:label ~prec:Ir.Ops.single ~o:[ 1 ] ~f:(fun _ -> 0.) () in
  set_materialized t.Tensor.value;
  t

(** The subset of [loss.params] that [loss]'s backprop actually trains: the parameters whose
    gradient the backprop code writes ([Asgns.collect_written]). [t.params] deliberately answers a
    broader question — which parameter leaves the forward graph reads, hence what {!init_params}
    must initialize and {!Persistence} must save — so a parameter detached behind
    {!Operation.stop_gradient} (a frozen backbone) stays in [loss.params] while its gradient is
    neither zeroed nor written. Stepping it anyway would apply weight decay to weights the user
    froze (gh-ocannl-673); the optimizer-side helpers ({!sgd_update}, {!grad_l2_norm},
    {!clip_by_global_norm}, {!grad_checksum}, {!zero_params_grads}) therefore derive this set
    instead of trusting [loss.params]. Empty when [loss] is not differentiable. *)
let trainable_params loss =
  match loss.Tensor.diff with
  | None -> Set.empty (module Tensor)
  | Some diff ->
      let written = Asgns.collect_written diff.Tensor.backprop.Asgns.asgns in
      Set.filter loss.Tensor.params ~f:(fun p ->
          match p.Tensor.diff with None -> false | Some d -> Set.mem written d.Tensor.grad)

(* The parameter set an optimizer-side helper operates on: [?params] when given (the escape hatch
   for exotic flows, e.g. a gradient written outside this loss's backprop), the derived
   {!trainable_params} otherwise. An empty derived set fails loudly: a differentiable loss that
   trains no registered parameters has no legitimate use for these helpers, and the failure is
   otherwise silent at every stage — the helpers compile empty or vacuous routines that run as
   no-ops (gh-ocannl-670). *)
let params_for ~fn_name ?params loss =
  match params with
  | Some ps -> ps
  | None ->
      if Option.is_none loss.Tensor.diff then
        raise @@ Tensor.Session_error (fn_name ^ ": loss is not differentiable", Some loss);
      let ps = trainable_params loss in
      if Set.is_empty ps then
        raise
        @@ Tensor.Session_error
             ( fn_name
               ^ ": the loss trains no parameters -- no [loss.params] member's gradient is written \
                  by the backprop code. Note that only [Tensor.param]-registered tensors join \
                  [params] ([Operation.init ~grad_spec:Require_grad] leaves do not), and a \
                  parameter behind [stop_gradient] is not trained",
               Some loss );
      ps

(** Replaces the zeroing of the given gradient nodes with [Noop] inside a [zero_grads] computation
    (each per-tensor zeroing is a [Fetch] of zeros — see [Tensor.fetch_zeros]). Used by the
    gradient-accumulation variant of {!grad_update}, which must keep zeroing the {e intermediate}
    gradients every micro-step (their backprop contributions are plain [=+] accumulations relying on
    a same-routine reset) while the {e parameter} gradients accumulate across micro-steps.

    The match is by [Fetch] constructor and tnode identity, so a change to how zeroing is emitted (a
    different constructor, or a parameter gradient dropping out of the tree) would silently keep
    zeroing parameter gradients — corrupting the accumulation. A gradient that the backprop writes
    is accumulated into, so it is zeroed in the tree: we raise when such a [grads] member had
    nothing removed rather than let the drift through. Pass only the gradients backprop reaches — a
    parameter detached from the loss (behind {!Operation.stop_gradient}, say) stays in [loss.params]
    while its gradient is neither zeroed nor accumulated. *)
let filter_out_grad_zeroing ~grads (comp : Asgns.comp) =
  let removed = Hash_set.create (module Tn) in
  let rec loop = function
    | Asgns.Noop -> Asgns.Noop
    | Asgns.Seq (t1, t2) -> Asgns.Seq (loop t1, loop t2)
    | Asgns.Block_comment (s, t) -> Asgns.Block_comment (s, loop t)
    | Asgns.Fetch { array; _ } when Set.mem grads array ->
        Hash_set.add removed array;
        Asgns.Noop
    | (Asgns.Accum_op _ | Asgns.Set_vec_unop _ | Asgns.Fetch _) as t -> t
  in
  let result = loop comp.asgns in
  let missing = Set.filter grads ~f:(fun g -> not (Hash_set.mem removed g)) in
  if not (Set.is_empty missing) then
    invalid_arg @@ "Train.filter_out_grad_zeroing: no zeroing found for accumulated gradient(s): "
    ^ String.concat ~sep:", " (List.map (Set.to_list missing) ~f:Tn.debug_name)
    ^ " -- the shape of the zero_grads code changed, gradient accumulation would be corrupted";
  (* The parameter gradients stay embedded: the surrounding {!grad_update} backprop accumulates into
     them regardless, and they are materialized so they persist across micro-steps. *)
  { comp with Asgns.asgns = result }

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
let grad_update ?(setup_for_parallel = false) ?accum_steps ?accum_loss ?loss_scale loss =
  Option.iter accum_steps ~f:(fun k ->
      if k <= 0 then invalid_arg "Train.grad_update: accum_steps must be positive");
  set_materialized loss.Tensor.value;
  (* Training loops read the loss from the host; declare the intent so the liveness memory planner
     (config [buffer_aliasing], gh-ocannl-489) never aliases the loss buffer -- like param
     gradients' observation intent declared in [Tensor.param]. *)
  Tn.set_observable loss.Tensor.value;
  if setup_for_parallel || Option.is_some accum_steps then
    (* Only the trained parameters' gradients: a frozen parameter's gradient is never written, so
       declaring it materialized would demand a buffer nothing computes (gh-ocannl-673). *)
    Set.iter (trainable_params loss) ~f:(fun p ->
        set_materialized (Option.value_exn ~here:[%here] p.diff).grad);
  let zero_grads =
    match loss.Tensor.diff with
    | None ->
        raise @@ Tensor.Session_error ("Train.grad_update: loss is not differentiable", Some loss)
    | Some diff -> (
        match accum_steps with
        | None -> diff.zero_grads
        | Some _ ->
            (* Only the parameters the backprop reaches ({!trainable_params}): one detached behind
               {!Operation.stop_gradient} (a frozen backbone) is still in [loss.params], but its
               gradient is neither zeroed here nor accumulated into, so it is not ours to strip --
               and demanding a zeroing for it would reject the freezing flow outright. *)
            let grads =
              Set.filter_map
                (module Tn)
                (trainable_params loss)
                ~f:(fun p -> Option.map p.Tensor.diff ~f:(fun d -> d.Tensor.grad))
            in
            filter_out_grad_zeroing ~grads diff.zero_grads)
  in
  let inv_accum = 1. /. Float.of_int (Option.value accum_steps ~default:1) in
  (* Note: the %cd syntax for [loss.grad] does not modify roots. *)
  [%cd
    ~~(loss "forward and gradient update";
       (* In the accumulating branch, referencing [loss] embeds its forward code (the single
          consumption of it), so the one statement computes the loss and accumulates it. *)
       (match accum_loss with
       | Some acc -> acc =+ loss
       | None -> loss.forward);
       ~~(loss "zero grads and backprop";
          zero_grads;
          (match (loss_scale, accum_steps) with
          | Some scale, None -> loss.grad =: scale
          | Some scale, Some _ -> loss.grad =: scale * !.inv_accum ~logic:"."
          | None, Some _ -> loss.grad =: !.inv_accum
          | None, None -> loss.grad =: 1);
          loss.backprop))]

(** Code zeroing the trained parameters' gradients ({!trainable_params}, or [?params]), as a
    standalone computation: the gradient-accumulation counterpart of {!grad_update}[ ~accum_steps]
    (which deliberately does not zero them). Compile it as its own routine and run it at the start
    of each accumulation cycle — before the first micro-step, llm.c-style ("we're about to +=
    accumulate into them"). *)
let zero_params_grads ?params loss =
  let one_param p =
    match p.Tensor.diff with
    | None -> raise @@ Tensor.Session_error ("Train.zero_params_grads: not differentiable", Some p)
    | Some diff -> diff.zero_grads
  in
  let comp =
    Set.to_list (params_for ~fn_name:"Train.zero_params_grads" ?params loss)
    |> List.map ~f:one_param |> Asgns.sequence
  in
  { comp with asgns = Asgns.Block_comment ("zero_params_grads", comp.asgns) }

(** A scalar checksum over the trained parameters' gradients ({!trainable_params}, or [?params]) of
    [loss]: returns the flag tensor and the code that resets it to 0 and accumulates the sum of
    every gradient cell into it. The sum is non-finite if and only if some gradient cell is
    non-finite (a finite sum cannot arise from non-finite cells: same-sign infinities stay infinite,
    opposite-sign infinities and NaNs produce NaN; a spurious overflow of large finite gradients
    only triggers a benign extra backoff). Sequence it after {!grad_update} in the same routine,
    read the flag with [Context.get_values] and gate the optimizer step on [Float.is_finite] — the
    dynamic loss scaling recipe ({!Mixed_prec.step}) does exactly this. *)
let grad_checksum ?params loss =
  let params = params_for ~fn_name:"Train.grad_checksum" ?params loss in
  let flag = NTDSL.init ~l:"grad_checksum" ~prec:Ir.Ops.single ~o:[ 1 ] ~f:(fun _ -> 0.) () in
  set_materialized flag.Tensor.value;
  Tn.set_observable flag.Tensor.value;
  (* Settle shape inference for the parameters first: the total-reduce einsum below unifies each
     parameter's rows with the spec's row variables, and a parameter row still unsolved at
     settlement would then be refused the close-to-empty guess (row.ml's "You forgot to specify the
     hidden dimension(s)" — a row variable used in an einsum spec is no longer safe to guess).
     Forcing dims here closes e.g. a bias's inferred-empty input row before the spec touches it —
     which is also why [grad_checksum] must be called only after the model and the loss are fully
     constructed. *)
  Set.iter params ~f:(fun p -> ignore (Lazy.force p.Tensor.value.Tn.dims : int array));
  let one_param p =
    if Option.is_none p.Tensor.diff then
      raise @@ Tensor.Session_error ("Train.grad_checksum: not differentiable", Some p);
    [%cd flag =+ id p.grad ~logic:"...|...->... => |->0"]
  in
  let comps = Set.to_list params |> List.map ~f:one_param in
  let reset = [%cd flag =: 0] in
  let comp = Asgns.sequence (reset :: comps) in
  (flag, { comp with asgns = Asgns.Block_comment ("grad_checksum", comp.asgns) })

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
let sgd_one ~learning_rate ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) ?grad_unscale
    ?grad_scale ?update_gate p =
  if Option.is_none p.Tensor.diff then
    raise @@ Tensor.Session_error ("Train.sgd_one: not differentiable", Some p);
  (* The [%cd] payload is written once and the [update_gate] arms differ only where the gating
     actually changes the emitted assignment. Inline declarations ([{ sgd_delta }], [{ sgd_momentum
     }]) are hoisted by the ppx above the enclosing [match]/[if], so an arm that declares a tensor
     and an arm that only reads it name the same tensor. *)
  [%cd
    ~~(p "param sgd step";
       (match grad_unscale with
       (* The binary form: a unary [p.grad =* unscale] would be a Pointwise_un, which does not
          broadcast the scalar's closed rows against parameters with input axes. *)
       | Some unscale -> p.grad =: p.grad * unscale ~logic:"."
       | None -> Asgns.empty_comp);
       (* Gradients can only be read as direct operands (not inside subexpressions), hence the
          scaled read is its own statement. *)
       (match grad_scale with
       | Some scale ->
           { sgd_delta } =: p.grad * scale ~logic:".";
           sgd_delta =+ !.weight_decay *. p
       | None -> sgd_delta =: p.grad + (!.weight_decay *. p));
       if Float.(momentum > 0.0) then (
         (match update_gate with
         | None -> { sgd_momentum } =: (!.momentum *. sgd_momentum) + sgd_delta
         | Some gate ->
             sgd_momentum =: where gate ((!.momentum *. sgd_momentum) + sgd_delta) sgd_momentum);
         if nesterov then sgd_delta =+ !.momentum *. sgd_momentum else sgd_delta =: sgd_momentum);
       (* The final selection covers every path: without momentum it discards the possibly
          non-finite delta; with momentum the buffer kept its old (finite) value above, and this
          still zeroes the step so [p] is untouched. *)
       (match update_gate with
       | Some gate -> sgd_delta =: where gate sgd_delta 0
       | None -> Asgns.empty_comp);
       p =- learning_rate * sgd_delta ~logic:".")]
  |> fun comp ->
  (* The momentum buffer is optimizer state: it is read before it is written, and the value it
     carries into the next step is the whole point of it. Left undetermined it is a virtualization
     candidate, and inlining a node whose defining computation is in a PREVIOUS invocation of the
     routine has no meaning -- lowering fails outright ("No computations found for #N:
     sgd_momentum_..."), which is why [~momentum] was unusable until gh-ocannl-772 covered it. The
     label's head is the inline declaration's identifier, so this names exactly the node the [%cd]
     payload above declared. *)
  if Float.(momentum > 0.0) then
    Set.iter comp.Asgns.embedded_nodes ~f:(fun tn ->
        match tn.Tn.label with "sgd_momentum" :: _ -> set_materialized tn | _ -> ());
  comp

(** Maps {!sgd_one} over the parameters [loss] trains ({!trainable_params}, or [?params]): a
    parameter frozen behind {!Operation.stop_gradient} takes no step — in particular no weight decay
    (gh-ocannl-673). *)
let sgd_update ~learning_rate ?momentum ?weight_decay ?nesterov ?grad_unscale ?grad_scale
    ?update_gate ?params loss =
  let f =
    sgd_one ~learning_rate ?momentum ?weight_decay ?nesterov ?grad_unscale ?grad_scale ?update_gate
  in
  let comp =
    Set.to_list (params_for ~fn_name:"Train.sgd_update" ?params loss)
    |> List.map ~f |> Asgns.sequence
  in
  { comp with asgns = Asgns.Block_comment ("sgd_update", comp.asgns) }

(** All and only bindings with associated ranges are iterated, with the binding's initial value
    lost. Bindings without ranges remain at their initial values, as do symbolic extents (gh-490):
    an extent is a size set once by the user, not an index to iterate.

    [f] need not wait for the device: {!Context.run} reads the bindings at the dispatch, so the next
    iteration may rebind them immediately whatever the backend schedules asynchronously. *)
let%track3_sexp sequential_loop ~f lowered_bindings =
  let rec loop = function
    | [] -> f ()
    | ({ Idx.static_range = None; static_symbol = _; _ }, _) :: more -> loop more
    | ({ Idx.used_as_extent = true; _ }, _) :: more -> loop more
    | ({ Idx.static_range = Some range; static_symbol = _; _ }, idx) :: more ->
        let old_idx = !idx in
        for i = 0 to range - 1 do
          idx := i;
          loop more
        done;
        idx := old_idx
  in
  loop lowered_bindings

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
module Lr_schedule = struct
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

  (** The learning rate at [step] (0-based). *)
  let learning_rate t ~step =
    let clamped_frac ~from =
      let denom = t.total_steps - from in
      if denom <= 0 then 1.0
      else Float.max 0.0 @@ Float.min 1.0 @@ (Float.of_int (step - from) /. Float.of_int denom)
    in
    let min_lr = t.base_lr *. t.final_frac in
    (* The [total_steps] clamp outranks warmup: a warmup longer than the horizon (a degenerate but
       constructible config — the record is transparent and unvalidated) must not keep ramping past
       the documented endpoint, so past [total_steps] the decay branch answers (its clamped ratio is
       1 there, i.e. the final value). *)
    let with_warmup decayed =
      if step < t.warmup_steps && step < t.total_steps then
        t.base_lr *. Float.of_int (step + 1) /. Float.of_int t.warmup_steps
      else decayed ()
    in
    match t.kind with
    | Constant -> with_warmup (fun () -> t.base_lr)
    | Cosine ->
        with_warmup (fun () ->
            let ratio = clamped_frac ~from:t.warmup_steps in
            min_lr +. (0.5 *. (1.0 +. Float.cos (Float.pi *. ratio)) *. (t.base_lr -. min_lr)))
    | Linear ->
        with_warmup (fun () ->
            let ratio = clamped_frac ~from:t.warmup_steps in
            min_lr +. ((1.0 -. ratio) *. (t.base_lr -. min_lr)))
    | Wsd { decay_frac } ->
        with_warmup (fun () ->
            let decay_point = Float.to_int ((1.0 -. decay_frac) *. Float.of_int t.total_steps) in
            if step < decay_point then t.base_lr
            else
              let ratio = clamped_frac ~from:decay_point in
              min_lr +. ((1.0 -. Float.sqrt ratio) *. (t.base_lr -. min_lr)))
end

(** A device-resident, broadcastable scalar the host can overwrite with [Context.set_values].
    Data-backed on purpose: a 1-element [term_init] would come out as a [Constant] fetch, re-fetched
    by every step's forward code, silently undoing [Context.set_values]. The [bcast_if_1] axis basis
    (as in [Tensor.number]) lets the scalar broadcast into tensors of any shape. *)
let host_scalar ~l v =
  let ndarray =
    Ir.Ndarray.init_array ~debug:l Ir.Ops.single ~dims:[| 1 |] ~padding:None ~f:(fun _ -> v)
  in
  let t =
    (* Reshape rather than Keep_shape_no_padding: the latter pins the data axis at the default
       basis, which conflicts with the [bcast_if_1] tag (bases are incompatible atoms). *)
    Tensor.term ~grad_spec:Tensor.Prohibit_grad ~init_data:(Asgns.Reshape ndarray) ~label:[ l ]
      ~batch_dims:[] ~input_dims:[]
      ~output_axes:[ (Row.bcast_if_1, 1) ]
      ()
  in
  set_materialized t.Tensor.value;
  Tn.set_observable t.Tensor.value;
  t

(** A learning-rate scalar driven by a host-side schedule: returns the tensor (pass it as
    {!sgd_update}'s [~learning_rate]) and a setter overwriting it with the schedule's value at
    [step] — call the setter once per step, before running the optimizer routine. The overwrite is a
    tiny host-to-device transfer, not a recompilation. *)
let scheduled_learning_rate ?(label = "learning_rate") schedule =
  let lr = host_scalar ~l:label (Lr_schedule.learning_rate schedule ~step:0) in
  let set_step ctx ~step =
    Context.set_values ctx lr.Tensor.value [| Lr_schedule.learning_rate schedule ~step |]
  in
  (lr, set_step)

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
let grad_l2_norm ?grad_unscale ?(label = "grad_norm") ?params loss =
  let params = params_for ~fn_name:"Train.grad_l2_norm" ?params loss in
  let norm = NTDSL.init ~l:label ~prec:Ir.Ops.single ~o:[ 1 ] ~f:(fun _ -> 0.) () in
  set_materialized norm.Tensor.value;
  Tn.set_observable norm.Tensor.value;
  (* Settle shape inference for the parameters first — same reason as in {!grad_checksum}: a
     parameter row still unsolved at settlement would be refused the close-to-empty guess once the
     einsum spec's row variables touch it. *)
  Set.iter params ~f:(fun p -> ignore (Lazy.force p.Tensor.value.Tn.dims : int array));
  let one_param p =
    if Option.is_none p.Tensor.diff then
      raise @@ Tensor.Session_error ("Train.grad_l2_norm: not differentiable", Some p);
    [%cd norm =+ p.grad * p.grad ~logic:"...|...->...; ...|...->... => |->0"]
  in
  let comps = Set.to_list params |> List.map ~f:one_param in
  let reset = [%cd norm =: 0] in
  let root =
    match grad_unscale with
    | Some unscale ->
        [%cd
          norm =: sqrt norm;
          norm =: norm * unscale ~logic:"."]
    | None -> [%cd norm =: sqrt norm]
  in
  let comp = Asgns.sequence ((reset :: comps) @ [ root ]) in
  (norm, { comp with asgns = Asgns.Block_comment (label, comp.asgns) })

type grad_clipping = {
  grad_norm : Tensor.t;
      (** The pre-clip global L2 norm (observable — read it for logging or outlier detection). *)
  grad_scale : Tensor.t;
      (** The clipping scale, computed on device: [min(1, max_norm / grad_norm)]. Pass it as
          {!sgd_update}'s [~grad_scale] so it folds into the update. *)
  clip_comp : Asgns.comp;
      (** Sequence it after the gradient update and before the optimizer step (all three can be one
          routine — no host round-trip is involved). *)
}

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
let clip_by_global_norm ?grad_unscale ?(label = "grad_clip") ?params ~max_norm loss =
  (* Validated because this is where configuration-derived floats arrive: a negative threshold would
     REVERSE gradients (negative ratio), and a nan one fails the ordered comparison and silently
     disables clipping. 0 stays legal — "clip to zero" freezes the gradient term. *)
  if (not (Float.is_finite max_norm)) || Float.(max_norm < 0.) then
    invalid_arg "Train.clip_by_global_norm: max_norm must be finite and nonnegative";
  let grad_norm, norm_comp = grad_l2_norm ?grad_unscale ~label:(label ^ "_norm") ?params loss in
  let grad_scale = host_scalar ~l:(label ^ "_scale") 1. in
  let scale_comp =
    (* Ordered comparison and selection: norm = 0 selects 1 (the division's [inf] is discarded by
       the [Where] selection). The [i => j] copy bridges the axis bases — [clip_sel] inherits the
       norm's default-basis axis while [grad_scale] carries the broadcastable [bcast_if_1] axis
       (distinct spec variables keep the two from unifying — incompatible basis atoms). *)
    [%cd
      { clip_ratio } =: !.max_norm /. grad_norm;
      { clip_sel } =: where (!.max_norm < grad_norm) clip_ratio 1;
      grad_scale =: id clip_sel ~logic:"i => j"]
  in
  let comp = Asgns.sequence [ norm_comp; scale_comp ] in
  {
    grad_norm;
    grad_scale;
    clip_comp = { comp with asgns = Asgns.Block_comment (label, comp.asgns) };
  }

(** A sliding-window z-score outlier detector for host-observed scalars (per llm.c's
    [llmc/outlier_detector.h]): feed it the per-step loss and/or {!grad_l2_norm} values and skip the
    optimizer step when the returned z-score exceeds a threshold. Pure host-side state; use one
    detector per monitored quantity. *)
module Outlier_detector = struct
  type t = {
    window : float array;
    mutable count : int;  (** Number of recorded values while the window is still filling. *)
    mutable index : int;  (** Replacement position once the window is full. *)
  }

  let create ?(window_size = 128) () =
    if window_size <= 0 then
      invalid_arg "Train.Outlier_detector.create: window_size must be positive";
    { window = Array.create ~len:window_size 0.; count = 0; index = 0 }

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
  let update t v =
    if not (Float.is_finite v) then Float.infinity
    else
      let n = Array.length t.window in
      if t.count < n then (
        t.window.(t.count) <- v;
        t.count <- t.count + 1;
        Float.nan)
      else
        let nf = Float.of_int n in
        let mag = Array.fold t.window ~init:0. ~f:(fun acc x -> Float.max acc (Float.abs x)) in
        let z =
          if Float.(mag = 0.) then
            (* All-zero window: any nonzero [v] is infinitely surprising. *)
            if Float.(v = 0.) then 0. else Float.copysign Float.infinity v
          else
            let smean = Array.fold t.window ~init:0. ~f:(fun acc x -> acc +. (x /. mag /. nf)) in
            let s =
              Array.fold t.window ~init:0. ~f:(fun acc x ->
                  let d = (x /. mag) -. smean in
                  acc +. (d *. d))
            in
            let rstd = Float.sqrt (s /. nf) in
            let num = (v /. mag) -. smean in
            if Float.(rstd = 0.) then
              if Float.(num = 0.) then 0. else Float.copysign Float.infinity num
            else num /. rstd
        in
        t.window.(t.index) <- v;
        t.index <- (t.index + 1) % n;
        z
end

let set_virtual (a : Tn.t) = Tn.update_memory_mode a Virtual 29

(** Materializes every non-literal embedded tensor node of [t] (so its value is inspectable on
    demand via the context). Replaces the old [every_non_literal_on_host] now that there is no
    hosted memory mode (gh-ocannl-333). *)
let every_non_literal_materialized =
  Tensor.iter_embedded ~f:(fun a ->
      if Tn.mode_is_unspecified a && not (Tn.known_constant a) then set_materialized a)

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

(** Parses the [tune_ship_arm] spelling of a {!placement_arm}. [source] names what is being parsed,
    for the error message. *)
let placement_arm_of_string ~source s =
  match String.lowercase (String.strip s) with
  | "auto" | "measured" -> Measured_winner
  | "a" | "default" -> Force_arm_a
  | "b" | "materialize-all" | "materialize_all" -> Force_arm_b
  | other ->
      invalid_arg
        (source ^ " should be auto | a | b (aliases: measured, default, materialize-all); found: "
       ^ other)

let placement_arm_name = function
  | Measured_winner -> "the measured winner"
  | Force_arm_a -> "A (default placements)"
  | Force_arm_b -> "B (materialize-all)"

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
    [autotune_log]. Other arguments are forwarded to {!Autotune.tune}; the same caveats apply
    (notably [timing_ctx] and non-idempotent routines — both arms share [timing_ctx]'s device for
    their searches). [name] included (gh-ocannl-669): it names both arms' compiles and the flip
    refinement's decision-surface lowerings, and is what lets a comp carrying no
    {!Ir.Assignments.Block_comment} — one {!Context.compile} would name at the call site — be tuned
    here at all.

    An arm that fails is a {e losing} arm, not a failed run (gh-ocannl-550): a search that
    terminates on a fatal failure ranks at [infinity], the other arm's completed winner ships and
    stays cached, and the failed arm's own report — [Autotune.Search_died], carrying the failure —
    still reaches [report] in position, so the failure is recorded rather than downgraded to "that
    arm merely lost". A failed arm's [best_ms] is deliberately {e not} shippable and not compared:
    {!Autotune.tune} raised, so no routine was compiled from [ctx] for it. Only when every arm fails
    does [tune_placements] propagate, with the first failure's original backtrace — with two
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

    The arms differ in which candidates {e exist}, not only in how they rank: a tensorized candidate
    is seeded only when the matmul site's operand and destination storage precisions resolve to a
    tile the backend advertises ({!Autotune.mma_tile_for_precisions}), and placement decides which
    nodes the site reads. Under the mixed-precision recipe on a uniform-format backend (Metal's
    simdgroup matrices) that makes arm A tensorization-free: the reduced-precision cast twins are
    virtual there, so the site reads f32 masters into a reduced-precision destination — a mixed
    triple no tile matches — while materialize-all turns the twins into real reduced-precision nodes
    and the seeds fire. Materializing just the twins ([Mixed_prec.Twin_materialized]) reaches the
    same seeds at arm A's cost; see benchmarks/report-gh546-metal.md.

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

    gh-ocannl-638, [ship_arm] (config [tune_ship_arm], default [Measured_winner]): ship a chosen
    {!placement_arm} instead of the measured winner. It exists for measurement — a profile of arm
    A's kernels is evidence about arm A's routine, and until that routine is the one that ships,
    nothing ever executes it against a reference, so a value-changing regression inside it leaves
    every structural and timing figure plausible. Deliberately loud: a non-default setting announces
    itself on stderr regardless of [autotune_log], both when it is resolved and at the decision it
    changes, because it is the optimizer's shipping path. Forcing also skips the flip refinement
    (which walks away from the chosen arm one node at a time, so its result is neither arm), and a
    forced arm that failed propagates its failure rather than falling back to the other arm.

    [on_ship] is called exactly once with ["A"], ["B"] or ["flip"] on the path that returns a
    routine, and not at all when nothing ships. It is what a consumer should attribute the returned
    artifact by: deriving the shipped arm from the reports' [best_ms] is only valid while nothing
    can override the comparison, which [ship_arm] now can, and it never described a flip-refined
    result at all. *)
let tune_placements ?name ?beam_width ?rounds ?repeats ?cache_dir ?timing_ctx ?report ?flip_report
    ?inline_flips ?ship_arm ?on_ship ctx loss comp bindings =
  (* Arm attribution on the same stderr trace as Autotune's config [autotune_log] — winner-arm
     ambiguity misdirected the CUDA benchmark debugging on PR #140. *)
  let log_arms =
    match
      String.lowercase
        (String.strip (Utils.get_global_arg ~arg_name:"autotune_log" ~default:"false"))
    with
    | "true" | "1" -> true
    | _ -> false
  in
  let logf fmt =
    Stdlib.Printf.ksprintf (fun s -> if log_arms then Stdio.eprintf "tune_placements: %s\n%!" s) fmt
  in
  (* gh-ocannl-638. Resolved before the first search, and announced there rather than only at the
     decision it changes: a search is minutes to hours, and a measurement that set this on the wrong
     cell should learn it from the head of the log, not from the summary of a run it has finished
     paying for. Unconditional on stderr, not through [logf]: this overrides the optimizer's
     shipping path, which is not something a run should have to have enabled [autotune_log] to find
     out about. *)
  let ship_arm =
    match ship_arm with
    | Some a -> a
    | None ->
        placement_arm_of_string ~source:"ocannl_tune_ship_arm"
          (Utils.get_global_arg ~arg_name:"tune_ship_arm" ~default:"auto")
  in
  let forced = match ship_arm with Measured_winner -> false | Force_arm_a | Force_arm_b -> true in
  if forced then
    Stdio.eprintf
      "Train.tune_placements: tune_ship_arm selects arm %s, which will ship whatever the timings \
       say. This is a measurement-only setting; a normal run ships the measured winner.\n\
       %!"
      (placement_arm_name ship_arm);
  let last = ref None in
  (* The public [?report] contract is positional — arm A's report then arm B's, which consumers
     (e.g. the benchmark harness) attribute by arrival order — so flip-refinement searches report
     through the separate [?flip_report] instead ([~to_report] selects the callback).

     gh-ocannl-550: the arms are independent experiments, so one arm's terminal failure is no
     evidence about another arm's completed result — and must not destroy it. Per-candidate
     containment inside a search is {!Autotune.tune}'s job (gh-ocannl-533/536) and it holds: on the
     reproduction that motivated this (benchmarks/report-gh528-gpt2-cuda.md §3, five of five tf32
     gpt2_mini runs) the OOMing candidates were absorbed as ordinary [Backend_link] declines and the
     search ran to its end. What escaped is the aftermath: with the device exhausted, the arm's
     winner replay could not compile and neither could its untuned-default fallback, so
     [Autotune.tune] raised — and with no handler here, arm B's failure took arm A's already
     finished, already cached winner (106.389 ms, the best arm-A result of that whole leg) out of
     the process with it. Catching per arm is the whole fix: a failed search returns [Error] and
     ranks at [infinity], which is not the same as a completed search that timed nothing (which
     returns the untuned default compile and also ranks at [infinity] — hence [Result.t] rather than
     a sentinel time). *)
  (* The arms are contained; two classes of failure are not the arm's to absorb, and they are the
     same two {!Ir.Schedule_outcome.classify_raw} refuses to classify one level down — containing
     them here would re-absorb exactly what that policy exists to refuse.

     - Process-level ([Out_of_memory] on the host heap, [Sys.Break]): no evidence about this arm and
       nothing to fall back to. Swallowing an interrupt to go run the other arm would make Ctrl-C
       during a half-hour search do nothing.
     - Compiler invariant violations ([Assert_failure], [Stack_overflow]): a tuner bug, not a
       schedule that lost. Demoting one to "that arm lost" would let a process exit 0 with a shipped
       winner and the bug unmentioned outside the report. (A stale cache entry tripping an assert is
       already turned into a classified decline inside {!Autotune.tune} under [Cache_replay]
       provenance, so it never reaches here as an exception.)

     The device OOM this containment exists for is an ordinary [Invalid_argument] from the driver
     and stays contained. *)
  let must_propagate = function
    | Out_of_memory | Stdlib.Sys.Break | Stack_overflow | Assert_failure _ -> true
    | _ -> false
  in
  (* A [report] callback exception is the caller's failure, not the search's — {!Autotune.tune}
     propagates it deliberately on the completion path — so it is re-raised rather than reclassified
     as an arm failure (which would hide it, and could ship the other arm even though this search
     completed). It is marked by WRAPPING it at the raise site: a nullary exception ([Exit],
     [End_of_file]) is a singleton value, so physical identity cannot tell a callback's [Exit] from
     an arm failure's, and the tuner's fatal path deliberately swallows the callback's exception and
     raises the arm's — the very case a same-value collision would misread as "the callback failed".
     A wrapper the tuner never constructs cannot collide: whatever comes out unwrapped is the
     search's. The rendered message is carried alongside because the tuner prints the wrapper with
     the default exception printer when it swallows it, and that printer shows string fields
     only. *)
  let exception Report_callback_failed of string * exn * Stdlib.Printexc.raw_backtrace in
  let tune ?to_report arm ctx timing_ctx =
    let to_report = Option.value to_report ~default:report in
    let capture r =
      last := Some r;
      Option.iter to_report ~f:(fun f ->
          match f r with
          | () -> ()
          (* Wrapped so it can be told apart from the search's own failures — except when it is
             process-fatal, where that distinction is pointless and the wrapper would hide it from
             the tuner's own guard on the fatal path (which re-raises such exceptions rather than
             swallowing them). *)
          | exception exn when must_propagate exn -> raise exn
          | exception exn ->
              raise
                (Report_callback_failed
                   (Exn.to_string exn, exn, Stdlib.Printexc.get_raw_backtrace ())))
    in
    logf "arm %s search:" arm;
    last := None;
    let result =
      match
        Autotune.tune ?name ?beam_width ?rounds ?repeats ?cache_dir ?timing_ctx ~report:capture ctx
          comp bindings
      with
      | compiled -> Ok compiled
      (* Unwrapped, so the caller sees its own exception with its own backtrace: the wrapper is an
         internal marker, not part of this function's contract. *)
      | exception Report_callback_failed (_, callback_exn, callback_backtrace) ->
          Stdlib.Printexc.raise_with_backtrace callback_exn callback_backtrace
      | exception exn ->
          let backtrace = Stdlib.Printexc.get_raw_backtrace () in
          if must_propagate exn then Stdlib.Printexc.raise_with_backtrace exn backtrace
          else Error (exn, backtrace)
    in
    (* [?report] is positional and consumers name arms by arrival order, so a failing arm must still
       occupy its slot. It does: {!Autotune.tune} reports exactly once per call on every path that
       does any work, each pre-search failure carrying the phase it died at. *)
    let r = !last in
    let best_ms =
      match result with
      (* Not [r.best_ms]: a [Search_died] report's best is a measurement of the search context, and
         no routine was compiled from [ctx] for it. There is nothing to ship at that time. *)
      | Error _ -> Float.infinity
      | Ok _ -> Option.value_map r ~default:Float.infinity ~f:(fun r -> r.Autotune.best_ms)
    in
    (match result with
    | Error (exn, _) ->
        logf "arm %s FAILED, it loses the comparison (%s): %s" arm
          (Option.value_map r ~default:"it reported nothing" ~f:(fun r ->
               if Float.is_inf r.Autotune.best_ms then "it had timed nothing"
               else
                 Printf.sprintf "its pre-failure best of %.4f ms is not shippable"
                   r.Autotune.best_ms))
          (Exn.to_string exn)
    | Ok _ ->
        logf "arm %s best: %.4f ms (%s)" arm best_ms
          (Option.value_map r ~default:"no report" ~f:(fun r ->
               Printf.sprintf "%s%s, best tensorized %s"
                 (if String.is_empty r.Autotune.best_label then "nothing timed"
                  else r.Autotune.best_label)
                 (* What the schedule asked for, and next to it what the emission delivered
                    (gh-ocannl-626): "[tensorized]" alone has read as a tensor-core claim over a
                    kernel whose every [Tile_mma] fell back to the lane-0 scalar loop. *)
                 (if r.Autotune.best_tensorized then
                    Printf.sprintf " [tensorized/%s]"
                      (Option.value_map r.Autotune.best_tensorization ~default:"no census"
                         ~f:Ir.C_syntax.tensorization_name)
                  else "")
                 (if Float.is_inf r.Autotune.mma_best_ms then "none"
                  else Printf.sprintf "%.4f ms" r.Autotune.mma_best_ms))));
    (result, best_ms, r)
  in
  (* gh-ocannl-550: every arm and every flip that SUCCEEDS produces a compiled routine, and exactly
     one of them ships. Dropping the OCaml value does not free anything — the backend's pool table
     roots the slabs (see {!Context.release}) — so an unshipped result is a permanently rooted
     routine footprint, once per [tune_placements] call, plus one per attempted flip. Bounded per
     call, and therefore not the per-candidate growth {!Autotune.tune} fixes, but a
     repeatedly-tuning process (a benchmark sweep, a script tuning several routines) accumulates
     them without limit.

     Collected here rather than released at each decision point, because "may still ship" is not a
     local property: the flip chain bases on arm A's result, so A stays alive while the chain runs
     even when B won the A/B, and the chain's incumbent stays alive until the final comparison. One
     sweep at the single point where the answer is known avoids having to re-derive that per
     branch. *)
  let produced = ref [] in
  let record = function Ok compiled -> produced := compiled :: !produced | Error _ -> () in
  let release_unshipped ?keep () =
    List.iter !produced ~f:(fun ((cctx, _) as compiled) ->
        if not (Option.exists keep ~f:(phys_equal compiled)) then
          try Context.release cctx
          with exn when not (must_propagate exn) ->
            logf "release of an unshipped tune result failed: %s" (Exn.to_string exn))
  in
  (* The arms are independent experiments but they are not independent lineages: the B arm searches
     in a {!Context.decide_materialized} sibling, which shares the execution ledger. A failure whose
     damage the backend could not bound ([Writes_may_have_occurred], or an unattributed launch/sync
     failure) poisons that ledger, and every timing run in the sibling then refuses to execute — so
     the next arm cannot produce a result, only burn a search proving it. Propagate the failure that
     poisoned it instead, which is also the honest error: the second arm's failures would all be
     consequences of the first (gh-ocannl-550). Giving each arm its own root lineage is the other
     way out, at the cost of re-initializing the caller's parameters per arm. *)
  let search_lineage = Option.value timing_ctx ~default:ctx in
  let lineage_poisoned () = Option.is_some (Context.poisoned_failure search_lineage) in
  let propagate_if_poisoned who = function
    | Error (exn, backtrace) when lineage_poisoned () ->
        logf "arm %s poisoned the shared %s lineage, so no sibling arm can run: propagating" who
          (if Option.is_some timing_ctx then "timing" else "caller's");
        (* Nothing will ship (gh-ocannl-550): a sibling arm that already succeeded has no reader
           left. *)
        release_unshipped ();
        Stdlib.Printexc.raise_with_backtrace exn backtrace
    | _ -> ()
  in
  (* A [report] callback exception propagates by design (it is the caller's failure, not the
     search's), which means it exits [tune_placements] without reaching any of the ship paths below
     — so results already collected would be abandoned rooted. Every call after the first goes
     through this (gh-ocannl-550, round-four review); the first needs nothing, since [produced] is
     still empty. *)
  let tune_or_release ?to_report arm c t =
    match tune ?to_report arm c t with
    | r -> r
    | exception exn ->
        let backtrace = Stdlib.Printexc.get_raw_backtrace () in
        release_unshipped ();
        Stdlib.Printexc.raise_with_backtrace exn backtrace
  in
  let a, a_ms, a_report = tune "A (default placements)" ctx timing_ctx in
  record a;
  propagate_if_poisoned "A" a;
  let embedded = ref [] in
  Tensor.iter_embedded ~f:(fun tn -> embedded := tn :: !embedded) loss;
  (* [decide_materialized] skips the nodes constrained away from materialization (constants,
     declared-virtual), mirroring [every_non_literal_materialized]'s guards at the decision
     level. *)
  let materialize c = Context.decide_materialized c !embedded in
  let b, b_ms, b_report =
    tune_or_release "B (materialize-all)" (materialize ctx) (Option.map timing_ctx ~f:materialize)
  in
  record b;
  (* Both arms gone: there is no winner to ship, and the first failure is the one that has not been
     cascaded from — a device the other arm's failure exhausted or a lineage it poisoned would
     otherwise be reported as the cause. *)
  (match (a, b) with
  | Error (a_exn, a_backtrace), Error (b_exn, b_backtrace) ->
      (* gh-ocannl-638: which failure to propagate is the selector's question too. With an arm
         forced, the caller asked for THAT artifact and its failure is the answer — handing back the
         other arm's exception would report a search the caller did not select, and contradict the
         documented promise that a forced arm's failure propagates rather than being replaced. With
         no arm forced the first failure still wins, for the original reason: it is the one that has
         not been cascaded from (a device the other arm exhausted, or a lineage it poisoned, would
         otherwise be reported as the cause). *)
      let exn, backtrace =
        match ship_arm with
        | Measured_winner | Force_arm_a -> (a_exn, a_backtrace)
        | Force_arm_b -> (b_exn, b_backtrace)
      in
      logf "both arms failed, nothing to ship (A: %s; B: %s); propagating %s" (Exn.to_string a_exn)
        (Exn.to_string b_exn)
        (match ship_arm with
        | Measured_winner -> "arm A's, the failure that has not been cascaded from"
        | Force_arm_a -> "arm A's, the forced arm"
        | Force_arm_b -> "arm B's, the forced arm");
      Stdlib.Printexc.raise_with_backtrace exn backtrace
  | _ -> ());
  let measured_a_wins =
    match (a, b) with
    (* A failed arm never wins, whatever the other arm's time is — including [infinity], which a
       completed search that timed nothing legitimately reports. *)
    | Ok _, Error _ -> true
    | Error _, Ok _ -> false
    | Ok _, Ok _ | Error _, Error _ -> Float.( <= ) a_ms b_ms
  in
  (* gh-ocannl-638: the measured comparison is still computed and still logged under a forced arm —
     it is the number the measurement reports — but it no longer decides. *)
  let a_wins =
    match ship_arm with
    | Measured_winner -> measured_a_wins
    | Force_arm_a -> true
    | Force_arm_b -> false
  in
  let arm_ms r ms = match r with Error _ -> "FAILED" | Ok _ -> Printf.sprintf "%.4f ms" ms in
  logf "winner: arm %s (A %s vs B %s)"
    (if measured_a_wins then "A" else "B")
    (arm_ms a a_ms) (arm_ms b b_ms);
  if forced then (
    Stdio.eprintf
      "Train.tune_placements: shipping arm %s by tune_ship_arm; the measured winner is arm %s (A \
       %s vs B %s)%s.\n\
       %!"
      (if a_wins then "A" else "B")
      (if measured_a_wins then "A" else "B")
      (arm_ms a a_ms) (arm_ms b b_ms)
      (if Bool.equal a_wins measured_a_wins then ", so the override changed nothing"
       else ", so the override changed what ships");
    (* A forced arm has no fallback: shipping the other one would return an artifact the caller did
       not ask for, under a setting whose whole purpose is that the returned routine IS the profiled
       one. Said here because the propagation below is otherwise indistinguishable from an ordinary
       both-arms-failed run. *)
    match if a_wins then a else b with
    | Ok _ -> ()
    | Error (exn, _) ->
        Stdio.eprintf
          "Train.tune_placements: the arm tune_ship_arm selected failed, so its failure propagates \
           rather than the other arm shipping in its place: %s\n\
           %!"
          (Exn.to_string exn));
  (* gh-ocannl-546: a tensorized winner of the arm that is then discarded reaches no artifact and no
     end-to-end number, so the placement A/B is where it has to be said. Stated as the margin it
     lost by, not as a bare flag: on a small routine the arms can be separated by less than the
     candidate-level timing spread. *)
  let shipped, dropped = if a_wins then (a_report, b_report) else (b_report, a_report) in
  Option.iter dropped ~f:(fun d ->
      if d.Autotune.best_tensorized then
        logf
          "NOTE arm %s crowned a tensorized candidate (%s at %.4f ms%s) and did NOT ship: arm %s \
           %s at %.4f ms%s"
          (if a_wins then "B" else "A")
          d.Autotune.best_label d.Autotune.best_ms
          (* A failed arm's crown is mid-search: it lost the A/B by failing, not by its time. *)
          (if Option.is_some (Autotune.terminal_failure d) then ", before that arm failed" else "")
          (if a_wins then "A" else "B")
          (* Under a forced arm the shipped one need not have won anything (gh-ocannl-638). *)
          (if forced then "ships by tune_ship_arm" else "wins the placement A/B")
          (if a_wins then a_ms else b_ms)
          (Option.value_map shipped ~default:"" ~f:(fun s ->
               if s.Autotune.best_tensorized then " (which is tensorized too)"
               else if Float.is_inf s.Autotune.mma_best_ms then
                 " (no tensorized candidate was timed in the shipping arm)"
               else
                 Printf.sprintf " (its own best tensorized candidate: %.4f ms)"
                   s.Autotune.mma_best_ms)));
  let winner, winner_ms = if a_wins then (a, a_ms) else (b, b_ms) in
  let inline_flips =
    match inline_flips with
    | Some n -> n
    | None -> Int.of_string (Utils.get_global_arg ~arg_name:"tune_inline_flips" ~default:"0")
  in
  (* gh-ocannl-638: the chain walks from arm A toward arm B one node at a time, so a refined result
     is neither arm — which is exactly what a forced arm asks not to ship. Skipped rather than
     rejected: the combination arises from configuration (a config file's flip budget plus a
     commandline arm), so it is a request to resolve, not a caller error to fail. *)
  let inline_flips =
    if forced && inline_flips > 0 then (
      Stdio.eprintf
        "Train.tune_placements: tune_ship_arm forces an arm, so the %d-flip inline refinement is \
         skipped -- a refined placement vector is neither arm.\n\
         %!"
        inline_flips;
      0)
    else inline_flips
  in
  (* The unwrap is the type-level statement of an invariant already established above: both arms
     [Error] propagated, [a_wins] never picks a failed arm, and the flip chain only ever replaces
     its incumbent with a strictly faster {e completed} search ([infinity] ranks a failed one). *)
  let ship ~what = function
    | Ok compiled -> (
        (* The poisoned check comes FIRST, and the retention decision after it (gh-ocannl-550,
           round-four review): retaining [compiled] and then raising would leave the one artifact
           nobody can reach. Whether this returns or raises, exactly one of the two calls below runs. *)
        (* A later arm can poison the lineage after this one succeeded. With a [timing_ctx] that is
           the scratch lineage and the winner — compiled from [ctx] — is unaffected. WITHOUT one the
           arms search in the caller's own lineage, so the winner's first [Context.run] is
           guaranteed to raise: handing it back would report success for a routine that cannot
           execute, and blame it on whichever routine poisoned the ledger. The poisoning failure is
           the honest answer, at the point where it is known. *)
        match if Option.is_none timing_ctx then Context.poisoned_failure ctx else None with
        | None ->
            release_unshipped ~keep:compiled ();
            (* gh-ocannl-638: the one place that knows what shipped, on the one path that ships.

               A raising callback is the caller's failure and propagates, like [report]'s — but it
               propagates INSTEAD of returning [compiled], and by then [compiled] is the one result
               deliberately not released. Dropping the OCaml value frees nothing (the backend's pool
               table roots the slabs), and the caller never received a handle, so the routine would
               be permanently unreachable and unreleasable: a repeatedly-tuning process with a flaky
               callback accumulates one routine footprint per call. Release it here, then re-raise
               the caller's own exception with its original backtrace. *)
            (match on_ship with
            | None -> ()
            | Some f -> (
                try f what
                with exn ->
                  let backtrace = Stdlib.Printexc.get_raw_backtrace () in
                  (try Context.release (fst compiled)
                   with exn2 when not (must_propagate exn2) ->
                     logf "release after a failing on_ship callback failed: %s" (Exn.to_string exn2));
                  Stdlib.Printexc.raise_with_backtrace exn backtrace));
            compiled
        | Some poisoned ->
            logf "the winner cannot ship: a later arm poisoned the caller's lineage";
            (* Nothing ships, so nothing is retained -- including this arm's own winner. *)
            release_unshipped ();
            raise poisoned)
    | Error (exn, backtrace) ->
        (* Nothing ships, so nothing is retained. *)
        release_unshipped ();
        Stdlib.Printexc.raise_with_backtrace exn backtrace
  in
  let winner_arm = if a_wins then "A" else "B" in
  if inline_flips <= 0 then ship ~what:winner_arm winner
  else
    let (* gh-555: greedy per-node refinement over the inlining decision vector. The vector lives on
           the default-policy arm (arm B's placements are caller-seeded wholesale, so its compile
           reports no policy decisions to flip), so the chain refines from arm A's context — a
           Materialize chain walks from A toward B one node at a time — and the refined result ships
           only if it beats the A/B winner. The decision surface is read analyze-only (gh-560; the
           arms' compiles already populated the analysis cache, so this costs specialization
           replays).

           gh-514, the placement-space search over this chain: the surface arrives ranked
           enablement-first ({!Autotune.placement_surface} — the gh-558 ordering lesson: a flip's
           value includes which sketch families become expressible under it, so family-unlocking
           Materialize flips outrank cost), and under config [autotune_bound_pruning] a
           [`Materialize] flip whose partial-vector roofline floor
           ({!Ir.Cost_model.completion_floor}, monotone in the chain's accumulated commitments)
           already meets the chain's best measured time is fathomed without spending budget — the
           admissible direction, exactly phase 4b's rule one level up. The budget counts {e
           measured} flips, so a fathomed candidate lets the next one in. *)
      module
      LL =
      Ir.Low_level
    in
    let
        (* gh-514 follow-up gh-ocannl-579: the enablement prior prices which sketch families a flip
           makes EXPRESSIBLE, and nothing about whether they pay. The arms just searched settle
           that, for free: arm B is the all-materialized specialization the prior derives its
           enablement set from, so its report's best tensorized time against its best time says what
           the promoted flips' family is worth on this device, on this computation, in this session.
           Under [tune_flip_ordering=profitable] (the default) a family measured to lose here voids
           the prior and the surface ranks by cost — on gh-514's metal/f16 cell the promotion
           displaced the winning cheap inline flip out of a budget-5 chain. Both arms' reports are
           handed over, including a failing arm's: its timings are measurements of the family even
           though its [best_ms] is not shippable. The surface derives the verdict, and only when the
           ordering in force consults one. *)
        evidence =
      List.filter_opt [ a_report; b_report ]
    in
    let surface =
      (* Outside the tuner's failure containment; a lowering failure (the A/B searches above can
         still have crowned a winner) must skip the refinement, not fail the tune. *)
      match Autotune.placement_surface ?name ~evidence ctx comp bindings with
      | s -> Some s
      (* This containment is for a lowering that declined, and for nothing else. A malformed
         [tune_flip_profit_margin] raises {!Utils.User_error} from inside it (gh-ocannl-579): the
         configuration asked for a refinement it also made impossible, and swallowing that would
         silently skip the refinement and ship the A/B winner as though the setting had been
         honored. The two process-level classes are not this containment's either, for the same
         reasons they are not the arms'. *)
      | exception exn when match exn with Utils.User_error _ -> true | _ -> must_propagate exn ->
          let backtrace = Stdlib.Printexc.get_raw_backtrace () in
          Stdlib.Printexc.raise_with_backtrace exn backtrace
      | exception exn ->
          logf "flip refinement skipped: the decision-surface lowering failed: %s"
            (Exn.to_string exn);
          None
    in
    match surface with
    | None -> ship ~what:winner_arm winner
    | Some surface ->
        let bound_pruning =
          Utils.get_global_flag ~default:false ~arg_name:"autotune_bound_pruning"
        in
        let candidates = surface.Autotune.ps_candidates in
        logf "flip refinement: %d candidate(s), %d enablement-promoted, ranked by %s, budget %d%s"
          (List.length candidates)
          (Set.length surface.Autotune.ps_enablement)
          (let name =
             match surface.Autotune.ps_ordering with `Cost -> "cost" | `Enablement -> "enablement"
           in
           match surface.Autotune.ps_profit with
           | Some profit -> name ^ " (" ^ Autotune.family_profit_summary profit ^ ")"
           | None -> name ^ " (configured unconditionally, so no profitability evidence was read)")
          inline_flips
          (if bound_pruning then ", bound pruning on" else "");
        let chain = ref (a, a_ms, ctx, timing_ctx) in
        (* The chain's accumulated placement commitments, for the floor: an accepted [`Materialize]
           flip and a rejected [`Inline] flip both leave the node certainly materialized in every
           completion the chain can still reach; the other two outcomes leave it open (inline
           commitments never tighten the floor). *)
        let certain_mat = ref [] in
        let measured = ref 0 and pruned = ref 0 in
        let rec walk = function
          | [] -> ()
          | _ when !measured >= inline_flips -> ()
          (* Same lineage, same rule as the A/B above: a poisoned one refuses every timing run, so a
             further search can only fail. Unlike the arms, there is nothing to propagate here — the
             A/B winner is already in hand — so the refinement just stops. *)
          | _ when lineage_poisoned () ->
              logf "flip refinement stopped: the shared lineage is poisoned"
          | fc :: rest -> (
              let _, chain_ms, base_ctx, base_timing = !chain in
              let arm =
                Printf.sprintf "flip %s %s (cost %d%s)"
                  (match fc.LL.fc_flip with `Inline -> "inline" | `Materialize -> "materialize")
                  (Tn.debug_name fc.LL.fc_tn) fc.LL.fc_recompute_cost
                  (if Set.mem surface.Autotune.ps_enablement fc.LL.fc_tn then ", enablement" else "")
              in
              let floor =
                match fc.LL.fc_flip with
                | `Materialize when bound_pruning ->
                    surface.Autotune.ps_floor_ms ~materialized:(fc.LL.fc_tn :: !certain_mat)
                | `Materialize | `Inline -> None
              in
              match floor with
              | Some fl when Float.(fl >= chain_ms) ->
                  (* Fathomed: the floor lower-bounds every completion with this node materialized,
                     so no nested search from here can beat the incumbent. The node's placement
                     stays open — nothing to commit. *)
                  Int.incr pruned;
                  logf "%s bound-pruned: floor %.4f ms >= incumbent %.4f ms" arm fl chain_ms;
                  walk rest
              | _ ->
                  let apply c =
                    match fc.LL.fc_flip with
                    | `Materialize -> Context.decide_materialized c [ fc.LL.fc_tn ]
                    | `Inline -> Context.decide_inline c [ fc.LL.fc_tn ]
                  in
                  let ctx' = apply base_ctx in
                  let timing' = Option.map base_timing ~f:apply in
                  let r, ms, _rep = tune_or_release ~to_report:flip_report arm ctx' timing' in
                  record r;
                  Int.incr measured;
                  let accepted = Float.(ms < chain_ms) in
                  if accepted then chain := (r, ms, ctx', timing');
                  (match (fc.LL.fc_flip, accepted) with
                  | `Materialize, true | `Inline, false ->
                      certain_mat := fc.LL.fc_tn :: !certain_mat
                  | `Materialize, false | `Inline, true -> ());
                  walk rest)
        in
        walk candidates;
        if !pruned > 0 then
          logf "flip refinement: %d flip(s) bound-pruned, %d measured" !pruned !measured;
        let chain_result, chain_ms, _, _ = !chain in
        if Float.(chain_ms < winner_ms) then (
          logf "flip refinement ships: %.4f ms (the placement A/B winner was %.4f ms)" chain_ms
            winner_ms;
          ship ~what:"flip" chain_result)
        else (
          logf "flip refinement did not improve on the A/B winner (%.4f ms vs %.4f ms)" chain_ms
            winner_ms;
          ship ~what:winner_arm winner)

module Lazy = Utils.Lazy

(* The untuned compile of the recipes, behind the gh-ocannl-491 config gate: with
   [model_default_schedule=true], the default schedule is picked by the analytic cost model
   ({!Autotune.model_default} — zero timing runs, advisory, falls back to the ordinary default
   pipeline); otherwise plain [Context.compile]. *)
let compile_with_model_gate ?(budgeted = false) ctx comp bindings =
  (* gh-ocannl-498: a memory budget is scored against the DEFAULT schedule pipeline
     ({!Ir.Schedule.maybe_default_schedules}, what [Backends.score_footprint] runs). The model gate
     picks a different segmentation, and alias spans and arena size are functions of the final
     segmentation — so honoring the gate on a budgeted compile would link a layout nobody scored,
     and a plan reported within budget could still exhaust the device. The budget is a constraint
     and the model pick is an advisory optimization, so the constraint wins: a budgeted compile uses
     the pipeline that was scored. Documented at [memory_budget] in ocannl_config.reference. *)
  if budgeted then (
    if Lazy.force Autotune.model_default_enabled then
      Stdio.eprintf
        "Train: memory_budget is set, so this compile uses the default schedule pipeline that the \
         budget was scored against, not model_default_schedule's pick.\n\
         %!";
    Context.compile ctx comp bindings)
  else if Lazy.force Autotune.model_default_enabled then Autotune.model_default ctx comp bindings
  else Context.compile ctx comp bindings

(** gh-ocannl-498 rematerialization: the configured device-memory budget for a compiled routine, or
    [None] when there is none (the default — under which the planning pass never runs and
    compilation is bit-for-bit what it was). The setting is a byte count with an optional K/M/G
    suffix (powers of 1024), the word [minimize], or 0 / off / false / none. *)
let memory_budget_setting () =
  let raw = String.strip (Utils.get_global_arg ~arg_name:"memory_budget" ~default:"0") in
  let bad () =
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Train: ocannl_memory_budget should be a byte count (optionally suffixed K, M or G), \
             the word \"minimize\", or 0 to disable; found: %s"
            raw)
  in
  match String.lowercase raw with
  | "" | "0" | "off" | "false" | "none" -> None
  | "minimize" -> Some Memory_budget.Minimize
  | lower ->
      let lower = Option.value (String.chop_suffix lower ~suffix:"b") ~default:lower in
      let mult, digits =
        match String.chop_suffix lower ~suffix:"k" with
        | Some d -> (1024, d)
        | None -> (
            match String.chop_suffix lower ~suffix:"m" with
            | Some d -> (1024 * 1024, d)
            | None -> (
                match String.chop_suffix lower ~suffix:"g" with
                | Some d -> (1024 * 1024 * 1024, d)
                | None -> (1, lower)))
      in
      let n = match Int.of_string_opt (String.strip digits) with Some n -> n | None -> bad () in
      (* The suffix scaling is where a syntactically fine setting turns into nonsense: "5000000000G"
         parses, then wraps to a negative or tiny target that the planner would honor as an
         unreachably tight budget and rematerialize hard against. Reject instead. *)
      if n <= 0 || n > Int.max_value / mult then bad () else Some (Memory_budget.Bytes (n * mult))

(** gh-ocannl-498: plan [comp]'s inlining decision vector against a device-memory budget and return
    the context to compile it from. [budget] overrides the config key [memory_budget]; with neither,
    this is the identity on [ctx] and returns [None] — the default-off path does not lower, score or
    decide anything. See {!Memory_budget.fit} for what the planner does and what it requires (config
    [buffer_aliasing]). *)
let fit_memory_budget ?budget ?max_candidates ?name ctx comp bindings =
  match match budget with Some _ as b -> b | None -> memory_budget_setting () with
  | None -> (ctx, None)
  | Some budget ->
      let ctx, plan = Memory_budget.fit ?name ?max_candidates ~budget ctx comp bindings in
      (ctx, Some plan)

(** Dumps [comp] as a [.cd] file in the build directory, for the [?output_cd_file] argument of
    {!to_routine} and {!run_once}. [caller] names the calling function in the error raised when the
    global setting [output_debug_files_in_build_directory] is false. *)
let dump_cd_file ~caller bindings (comp : Asgns.comp) =
  let name = Asgns.get_name_exn comp.Asgns.asgns in
  if not Utils.settings.output_debug_files_in_build_directory then
    raise
    @@ Utils.User_error
         (caller ^ ": output_cd_file is true, but output_debug_files_in_build_directory is false");
  let cd_source = Utils.output_to_build_file ~fname:(name ^ "-debug.cd") in
  let static_indices = Idx.bound_symbols bindings in
  match cd_source with
  | None -> ()
  | Some callback -> callback (Asgns.to_doc ~name ~static_indices () comp.Asgns.asgns)

(** The gh-ocannl-498 rematerialization seam shared by {!to_routine} and {!run_once}: plan [comp]'s
    inlining decision vector against the budget, compile from the planned context, and only then let
    [budget_report] observe the plan.

    The report fires AFTER the compile so it observes the plan that SHIPPED — a compile or link
    failure must not have announced one. It also keeps a callback from reaching the compile it is
    reporting on: the config gates the scoring depends on ([buffer_aliasing]) are re-read at each
    compile, so a callback that flipped one would make the routine use a layout other than the one
    just scored. *)
let compile_within_budget ?budget ?max_candidates ?budget_report ctx comp bindings =
  let ctx, budget_plan = fit_memory_budget ?budget ?max_candidates ctx comp bindings in
  let budgeted = Option.is_some budget_plan in
  let ctx, routine = compile_with_model_gate ~budgeted ctx comp bindings in
  Option.iter budget_report ~f:(fun f -> Option.iter budget_plan ~f);
  (ctx, routine)

(** Compiles [comp] and returns the post-compile context together with the routine. [budget],
    [max_candidates] and [budget_report] are the gh-ocannl-498 rematerialization seam, forwarded to
    {!fit_memory_budget}: with no [budget] and no [memory_budget] config key nothing is planned and
    the compile is exactly what it was; [budget_report], if given, observes the plan that shipped.

    The post-compile context is returned rather than discarded (gh-ocannl-772), matching
    {!Context.compile} and {!run_once}: chain it into the next compile instead of reaching into
    [routine.context] for the same value. A caller that only wants the routine can [snd] this. *)
let%track7_sexp to_routine (ctx : Context.t) ?(output_cd_file = false) ?budget ?max_candidates
    ?budget_report bindings comp =
  if output_cd_file then dump_cd_file ~caller:"Train.to_routine" bindings comp;
  (* Materialize the guessed output nodes so they persist across calls and are inspectable on demand
     via the context (gh-ocannl-333). *)
  Set.iter (snd @@ Asgns.collect_nodes_guess_output comp.Asgns.asgns) ~f:set_materialized;
  (* gh-ocannl-498: budget planning goes AFTER the output-materialization intent above (those nodes
     must not be flip candidates) and BEFORE the compile whose placements it steers. Off by default:
     with no budget this is the identity and the compile below is unchanged. *)
  compile_within_budget ?budget ?max_candidates ?budget_report ctx comp bindings

(** [init_params] initializes the parameters of [t], via running their forward code or copying from
    the host as appropriate. If [reinit_all] is true, all parameters are reinitialized, otherwise
    only the parameters that are not in [ctx.ctx_buffers] are initialized. *)
let init_params ?(reinit_all = false) ctx bindings t =
  let comp =
    if reinit_all then Tensor.init_params t
    else
      (* Check which params are already initialized *)
      let skip = Map.empty (module Tn) in
      Set.fold t.Tensor.params ~init:skip ~f:(fun skip p ->
          if Context.is_initialized ctx p.Tensor.value then
            Map.set skip ~key:p.Tensor.value ~data:()
          else skip)
      |> fun skip -> Tensor.init_params ~skip t
  in
  (* Materialize the parameters being initialized so they persist and are inspectable on demand. *)
  Set.iter (snd @@ Asgns.collect_nodes_guess_output comp.Asgns.asgns) ~f:set_materialized;
  (* Compile and run the initialization. Literal/ndarray-backed embedded nodes are uploaded into the
     context automatically at link time from [Host_inits] (gh-ocannl-333); there is no longer a
     separate host-array copy step here. *)
  let ctx, routine = Context.compile ctx comp bindings in
  Context.run ctx routine

type example_train_result = {
  inputs : Tensor.t;
  outputs : Tensor.t;
  model_result : Tensor.t;  (** Do not use [model_result] for deriving gradients. *)
  infer_callback : float array -> float array;
      (** Computes the output for the given input via the [model_result] tensor. Note:
          [infer_callback] is inefficient as it is not batched. *)
  rev_batch_losses : float list;
  rev_epoch_losses : float list;
  learning_rates : float list;
  used_memory : int;
}

(** [run_once] is a wrapper around {!init_params} that additionally runs code of [f t] and returns
    the context. If [skip_init] is true (false by default), no initialization is performmed. If
    [reinit_all] is true (false by default), all parameters are reinitialized, otherwise only the
    parameters that are not in [ctx.ctx_buffers] are initialized.

    If [output_cd_file] is true, the global setting [output_debug_files_in_build_directory] must be
    true, and the update code is output to a file before shape inference potentially crashes at
    [init_params]. *)
let%track3_sexp run_once ?(output_cd_file = false) ?(skip_init = false) ?reinit_all
    ?(bindings = IDX.empty) ?budget ?max_candidates ?budget_report ~f ctx (t : Tensor.t) : Context.t
    =
  set_materialized t.Tensor.value;
  (* Compute the update early, to ensure the shape inference is done. *)
  let update = f t in
  if output_cd_file then dump_cd_file ~caller:"Train.run_once" bindings update;
  let ctx =
    if skip_init || Set.is_empty t.params then ctx else init_params ?reinit_all ctx bindings t
  in
  let ctx, routine =
    compile_within_budget ?budget ?max_candidates ?budget_report ctx update bindings
  in
  Context.run ctx routine

(** Context-based versions of training functions for the new simplified API *)

(** [forward_once] is a wrapper around {!run_once} that runs the forward code of [t]. *)
let forward_once ?output_cd_file ?(skip_init = false) ?reinit_all ?(bindings = IDX.empty) ?budget
    ?max_candidates ?budget_report ctx t =
  let ctx =
    run_once ?output_cd_file ~skip_init ?reinit_all ~bindings ?budget ?max_candidates ?budget_report
      ~f:forward ctx t
  in
  (* gh-ocannl-777's recompute-on-read path should retire this forward-root consumption cleanup. *)
  Tensor.discard_backprop_code t;
  ctx

(** [update_once] is a wrapper around {!run_once} that runs the gradient update code of [t]: both
    forward and backprop. *)
let update_once ?output_cd_file ?(skip_init = false) ?reinit_all ?(bindings = IDX.empty) ?budget
    ?max_candidates ?budget_report ctx t =
  run_once ?output_cd_file ~skip_init ?reinit_all ~bindings ?budget ?max_candidates ?budget_report
    ~f:grad_update ctx t

(* For-print materialization (gh-ocannl-333 AC 5): the [%cd "for_print" =: t] trick. When a tensor's
   value is not already materialized in the printing context, recompile a copy of it ([for_print = t
   + 0]) into a fresh device-resident node and register that node as a for-print proxy, so the
   printer reads the tensor's value through it.

   This is best-effort: it works for recomputable (e.g. virtual / fetch-defined) tensors. For a
   tensor that is materialized elsewhere but simply absent from this context, the copy cannot be
   linked (its operand has no value here) — in that case we fall back to the metadata placeholder
   rather than crash. A fresh copy is built each call because [forward_once] consumes the copy's
   forward root; the for-print node is registered as the source's proxy for subsequent reads. *)
let ensure_printable (ctx : Context.t) (t : Tensor.t) : Context.t =
  if Context.mem ctx t.Tensor.value then ctx
  else
    try
      let for_print =
        let%op for_print = t + 0 in
        for_print
      in
      let ctx = forward_once ctx for_print in
      Context.register_for_print ~src:t.Tensor.value ~proxy:for_print.Tensor.value;
      ctx
    with _ -> ctx

(** [printf] is a wrapper around {!Tensor.print} that assumes [~force:true], and by default sets
    [~with_code:false], [~with_grad:true], and [~style:`Default]. It takes an explicit context and
    retrieves values on demand (gh-ocannl-333). If the tensor's value is not already materialized in
    [ctx], it is recomputed via the [for_print] copy trick so real values are still shown. *)
let%debug7_sexp printf ?here ?(with_grad = true) ?(with_code = false) ?(with_low_level = false)
    ?(style = `Default) (ctx : Context.t) (t : Tensor.t) : unit =
  let ctx = ensure_printable ctx t in
  Tensor.print ?here ~force:true ~ctx ~with_grad ~with_code ~with_low_level style t

(** [printf_tree] is a wrapper around {!Tensor.print_tree} that assumes [~force:true], and by
    default sets [~with_value:true], [~with_grad:true], and [~depth:9]. It takes an explicit context
    and retrieves values on demand (recomputing via [for_print] if not already materialized). *)
let printf_tree ?here ?with_value ?(with_grad = true) ?(depth = 9) (ctx : Context.t) t =
  let ctx = ensure_printable ctx t in
  Tensor.print_tree ?here ~force:true ~ctx ?with_value ~with_grad ~depth t

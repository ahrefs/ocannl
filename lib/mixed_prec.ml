open Base
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Asgns = Ir.Assignments
module Tensor = Ocannl_tensor.Tensor
module Operation = Ocannl_tensor.Operation
open Ocannl_tensor.Operation.DSL_modules

type twin_placement = Twin_auto | Twin_virtual | Twin_materialized

let cast_param ?(placement = Twin_auto) ?(master_prec = Ops.single) ~prec (p : Tensor.t) =
  match p.Tensor.diff with
  | None -> p
  | Some diff ->
      (* The twin keeps its master's shape by the close-down default of gh-ocannl-544: as an
         operation result, the cast's open batch row closes down to the master's (empty) batch row
         instead of being widened by a use site. Before that default, a batched-matmul read widened
         the twin to [batch, out, in] — a per-batch-row copy of the weight, numerically invisible
         but costing [batch]x memory and cast work and carrying the row symbol into the matmul's
         weight operand, which declined every tensorized candidate (gh-ocannl-540). The
         [mixed_prec_twin_shape] test guards the twin's shape either way. *)
      let twin = Operation.cast ~grad_spec:Tensor.If_needed p () in
      Tn.update_prec twin.Tensor.value prec;
      if Option.is_none (Tn.get_specified_prec p.Tensor.value) then
        Tn.update_prec p.Tensor.value master_prec;
      if Option.is_none (Tn.get_specified_prec diff.grad) then Tn.update_prec diff.grad master_prec;
      (match placement with
      | Twin_auto -> ()
      | Twin_virtual -> Train.set_virtual twin.Tensor.value
      | Twin_materialized -> Train.set_materialized twin.Tensor.value);
      twin

let with_master_weights ?placement ?master_prec ~prec f =
  let saved = !Tensor.param_postprocess in
  (Tensor.param_postprocess := fun p -> cast_param ?placement ?master_prec ~prec (saved p));
  Exn.protect ~f ~finally:(fun () -> Tensor.param_postprocess := saved)

module Loss_scaler = struct
  type t = {
    scale : Tensor.t;
    unscale : Tensor.t;
    mutable scale_val : float;
    mutable good_steps : int;
    growth_factor : float;
    backoff_factor : float;
    growth_interval : int;
  }

  (* A device-resident, broadcastable scalar the host can overwrite — see {!Train.host_scalar}
     (moved there by gh-ocannl-465 so learning-rate feeding and clipping scales share it). *)
  let host_scalar = Train.host_scalar

  let create ?(init_scale = 65536.) ?(growth_factor = 2.) ?(backoff_factor = 0.5)
      ?(growth_interval = 200) () =
    let scale = host_scalar ~l:"loss_scale" init_scale in
    let unscale = host_scalar ~l:"grad_unscale" (1. /. init_scale) in
    {
      scale;
      unscale;
      scale_val = init_scale;
      good_steps = 0;
      growth_factor;
      backoff_factor;
      growth_interval;
    }

  let scale_value t = t.scale_val

  let set_scale t ctx v =
    t.scale_val <- v;
    let ctx = Context.set_values ctx t.scale.Tensor.value [| v |] in
    Context.set_values ctx t.unscale.Tensor.value [| 1. /. v |]

  let update t ctx ~grads_finite =
    if grads_finite then (
      t.good_steps <- t.good_steps + 1;
      if t.good_steps >= t.growth_interval then (
        t.good_steps <- 0;
        set_scale t ctx (t.scale_val *. t.growth_factor))
      else ctx)
    else (
      t.good_steps <- 0;
      set_scale t ctx (t.scale_val *. t.backoff_factor))

  let update_n t ctx ~grads_finite ~steps =
    if grads_finite then (
      t.good_steps <- t.good_steps + steps;
      let ctx = ref ctx in
      while t.good_steps >= t.growth_interval do
        t.good_steps <- t.good_steps - t.growth_interval;
        ctx := set_scale t !ctx (t.scale_val *. t.growth_factor)
      done;
      !ctx)
    else (
      t.good_steps <- 0;
      set_scale t ctx (t.scale_val *. t.backoff_factor))
end

let scaled_grad_update ?setup_for_parallel ?accum_loss (scaler : Loss_scaler.t) loss =
  Set.iter (Train.trainable_params loss) ~f:(fun p ->
      Train.set_materialized (Option.value_exn ~here:[%here] p.Tensor.diff).grad);
  let update =
    Train.grad_update ?setup_for_parallel ?accum_loss ~loss_scale:scaler.Loss_scaler.scale loss
  in
  let flag, check = Train.grad_checksum loss in
  (flag, Asgns.sequence [ update; check ])

let scaled_sgd_update (scaler : Loss_scaler.t) ~learning_rate ?momentum ?weight_decay ?nesterov loss
    =
  Train.sgd_update ~learning_rate ?momentum ?weight_decay ?nesterov
    ~grad_unscale:scaler.Loss_scaler.unscale loss

let scaled_step ~(scaler : Loss_scaler.t) ~grad_routine ~sgd_routine ~checksum ctx =
  let ctx = Context.run ctx grad_routine in
  let sum = (Context.get_values ctx checksum.Tensor.value).(0) in
  let grads_finite = Float.is_finite sum in
  let ctx = if grads_finite then Context.run ctx sgd_routine else ctx in
  let ctx = Loss_scaler.update scaler ctx ~grads_finite in
  (ctx, grads_finite)

let gated_scaled_update ?setup_for_parallel ?accum_loss (scaler : Loss_scaler.t) ~learning_rate
    ?momentum ?weight_decay ?nesterov loss =
  (* Materialized like {!scaled_grad_update}'s: overflow detection depends on gradients being STORED
     at their (reduced) precision — a virtualized gradient recomputed at the consumer could bypass
     the f16 rounding that produces the inf the gate exists to catch. *)
  Set.iter (Train.trainable_params loss) ~f:(fun p ->
      Train.set_materialized (Option.value_exn ~here:[%here] p.Tensor.diff).grad);
  let update =
    Train.grad_update ?setup_for_parallel ?accum_loss ~loss_scale:scaler.Loss_scaler.scale loss
  in
  let flag, check = Train.grad_checksum loss in
  let gate = Loss_scaler.host_scalar ~l:"update_gate" 1. in
  let wflag = NTDSL.init ~l:"window_checksum" ~prec:Ops.single ~o:[ 1 ] ~f:(fun _ -> 0.) () in
  Train.set_materialized wflag.Tensor.value;
  Tn.set_observable wflag.Tensor.value;
  let gate_comp =
    (* Ordered comparisons yield exact 0/1, and nan fails both, so the product is the gate. The [i
       => j] copy bridges the axis bases: [finite01] inherits the flag's default-basis axis while
       [gate] carries the broadcastable [bcast_if_1] axis — distinct spec variables keep the two
       from unifying (incompatible basis atoms). *)
    [%cd
      { finite01 } =: (flag < !.3.0e38) * (!.(-3.0e38) < flag) ~logic:".";
      gate =: id finite01 ~logic:"i => j";
      wflag =+ flag]
  in
  let sgd =
    Train.sgd_update ~learning_rate ?momentum ?weight_decay ?nesterov
      ~grad_unscale:scaler.Loss_scaler.unscale ~update_gate:gate loss
  in
  (wflag, Asgns.sequence [ update; check; gate_comp; sgd ])

let gated_step ~(scaler : Loss_scaler.t) ~routine ~window_checksum ~check_interval ~step ctx =
  if check_interval <= 0 then invalid_arg "Mixed_prec.gated_step: check_interval must be positive";
  let ctx = Context.run ctx routine in
  if (step + 1) % check_interval = 0 then
    let sum = (Context.get_values ctx window_checksum.Tensor.value).(0) in
    let grads_finite = Float.is_finite sum in
    let ctx = Context.set_values ctx window_checksum.Tensor.value [| 0. |] in
    (Loss_scaler.update_n scaler ctx ~grads_finite ~steps:check_interval, grads_finite)
  else (ctx, true)

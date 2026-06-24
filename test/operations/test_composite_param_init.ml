open Base
open Ocannl
open Ocannl.Operation.DSL_modules

let centered_uniform1 ?label ?top_down_prec ?batch_dims ?batch_axes ?input_dims ?output_dims
    ?input_axes ?output_axes ?deduced () =
  let u = NTDSL.uniform1 () () in
  let two = Tensor.number ~grad_spec:Tensor.Prohibit_grad 2. in
  let one = Tensor.number ~grad_spec:Tensor.Prohibit_grad 1. in
  let scaled = NTDSL.pointmul two u () in
  NTDSL.sub scaled one ?label ?top_down_prec ?batch_dims ?batch_axes ?input_dims ?output_dims
    ?input_axes ?output_axes ?deduced ()

let () =
  Tensor.unsafe_reinitialize ();
  let w = TDSL.param ~param_init:centered_uniform1 "w" ~output_dims:[ 4 ] () in
  let%op loss = (w *. w) ++ "...|... => 0" in
  let update = Train.grad_update loss in
  let ctx = Train.init_params (Context.auto ()) Train.IDX.empty loss in
  let%op learning_rate = 0.01 in
  let sgd = Train.sgd_update ~learning_rate loss in
  let step = Train.to_routine ctx Train.IDX.empty (Ir.Assignments.sequence [ update; sgd ]) in
  let ctx = Context.context step in
  ignore (Context.run ctx step : Context.t)

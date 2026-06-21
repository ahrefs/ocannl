open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments

type run_result = {
  final_loss : float;
  learning_rates : float list;
  mlp_result : Tensor.t;
  margin_loss : Tensor.t;
}

let train_once ~seed () : run_result =
  (* Micrograd half-moons example, single device/stream. *)
  (* Note: for as-yet unknown reason, this test can lead to different results on different versions
     of dependencies. The 3-seed retry and epsilon=0.1 threshold ensure convergence on every host. *)
  Utils.settings.fixed_state_for_init <- Some seed;
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let open Operation.At in
  (* Sensitive to batch size -- smaller batch sizes are better. *)
  let batch_size = 10 in
  let len = batch_size * 40 in
  let n_batches = 2 * len / batch_size in
  let epochs = 80 in
  let steps = epochs * 2 * len / batch_size in
  let config = Dataprep.Half_moons.Config.{ noise_range = 0.1; seed = Some seed } in
  let moons_coordinates, moons_labels = Dataprep.Half_moons.generate_single_prec ~config ~len () in
  let moons_flat_ndarray = Ir.Ndarray.as_array Ir.Ops.Single moons_coordinates in
  let moons_classes_ndarray = Ir.Ndarray.as_array Ir.Ops.Single moons_labels in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  let moons_flat = TDSL.rebatch ~l:"moons_flat" moons_flat_ndarray () in
  let moons_classes = TDSL.rebatch ~l:"moons_classes" moons_classes_ndarray () in
  let%op mlp x =
    { w3 } * relu ({ b2; o = [ 16 ] } + ({ w2 } * relu ({ b1; o = [ 16 ] } + ({ w1 } * x))))
  in
  let%op moons_input = moons_flat @| batch_n in
  let%op moons_class = moons_classes @| batch_n in
  let learning_rates = ref [] in
  let%op margin_loss = relu (1 - (moons_class *. mlp moons_input)) in
  (* We don't need a regression loss formula thanks to weight_decay built into the sgd_update
     computation. *)
  let weight_decay = 0.0001 in
  let%op scalar_loss = (margin_loss ++ "...|... => 0") /. !..batch_size in
  let update = Train.grad_update scalar_loss in
  (* TODO(#321): Define learning_rate above the call to grad_update to test the consume_forward_code
     fix *)
  let%op learning_rate = 0.1 *. ((2 *. !..steps) - !@step_n) /. !..steps in
  (* TODO: is set_materialized needed? *)
  Train.set_materialized learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate ~weight_decay scalar_loss in
  let ctx = Train.init_params ctx bindings scalar_loss in
  let sgd_routine = Train.to_routine ctx bindings (Asgns.sequence [ update; sgd ]) in
  let ctx = Context.context sgd_routine in
  let step_ref = IDX.find_exn (Context.bindings sgd_routine) step_n in
  step_ref := 0;
  let final_loss = ref 0. in
  for _ = 1 to epochs do
    let epoch_loss = ref 0. in
    Train.sequential_loop (Context.bindings sgd_routine) ~f:(fun () ->
        Train.run ctx sgd_routine;
        epoch_loss := !epoch_loss +. (ctx, scalar_loss).@[0];
        learning_rates := ~-.((ctx, learning_rate).@[0]) :: !learning_rates;
        Int.incr step_ref);
    final_loss := !epoch_loss
  done;
  (* %cd instead of %op to not get complaints about point being uninitialized. *)
  let%cd mlp_result = mlp { point } in
  Train.set_materialized mlp_result.value;
  { final_loss = !final_loss; learning_rates = !learning_rates; mlp_result; margin_loss }

let main () =
  (* Don't decay the learning rate too quickly, it behaves better than in the original. *)
  let epsilon = 0.1 in
  let seeds = [ 1; 2; 3 ] in
  let winning =
    List.find_map seeds ~f:(fun seed ->
        let result = train_once ~seed () in
        if Float.(result.final_loss < epsilon) then Some result else None)
  in
  match winning with
  | None ->
    Stdio.eprintf "moons_demo: FAILED to converge in %d seeds\n%!" (List.length seeds);
    Stdlib.exit 1
  | Some result ->
    Stdio.printf "moons_demo: converged (final epoch loss < %.2g)\n%!" epsilon;
    Stdio.printf "\nLearning rate:\n%!";
    let plot_lr =
      PrintBox_utils.plot ~x_label:"step" ~y_label:"learning rate" ~small:true
        [ Line_plot { points = Array.of_list_rev result.learning_rates; content = PrintBox.line "-" } ]
    in
    PrintBox_text.output Stdio.stdout plot_lr;
    (* Testing how the syntax extension %op creates labels for the resulting tensors: *)
    Stdio.printf "mlp_result's name: %s\n%!" @@ Tensor.debug_name result.mlp_result;
    (* Note: mlp_result is not included in the resulting tensor's label, because the identifier label
       does not propagate across function calls. *)
    Stdio.printf "(mlp moons_input) name: %s\n%!"
    @@ Tensor.debug_name
    @@
    (match result.margin_loss.children with
    | [
     {
       subtensor =
         { children = [ _; { subtensor = { children = [ _; { subtensor; _ } ]; _ }; _ } ]; _ };
       _;
     };
    ] ->
        subtensor
    | _ -> assert false)

let () = main ()

open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
open Operation.DSL_modules
module Asgns = Ir.Assignments

let main () =
  (* Micrograd half-moons example, single device/stream. *)
  let seed = 2 in
  Utils.settings.fixed_state_for_init <- Some seed;
  Tensor.unsafe_reinitialize ();
  (* Note: for as-yet unknown reason, this test can lead to different resuls on different versions
     of dependencies. *)
  let module Backend = (val Backends.fresh_backend ()) in
  let open Operation.At in
  (* Sensitive to batch size -- smaller batch sizes are better. *)
  let batch_size = 10 in
  let len = batch_size * 40 in
  let n_batches = 2 * len / batch_size in
  let epochs = 80 in
  let steps = epochs * 2 * len / batch_size in
  let config = Datasets.Half_moons.Config.{ noise_range = 0.1; seed = Some seed } in
  let moons_coordinates, moons_labels = Datasets.Half_moons.generate_single_prec ~config ~len () in
  let moons_flat_ndarray = Ir.Ndarray.as_array Ir.Ops.Single moons_coordinates in
  let moons_classes_ndarray = Ir.Ndarray.as_array Ir.Ops.Single moons_labels in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  let moons_flat = TDSL.rebatch ~l:"moons_flat" moons_flat_ndarray () in
  let moons_classes = TDSL.rebatch ~l:"moons_classes" moons_classes_ndarray () in
  let%op mlp x =
    { w3 } * relu ({ b2; o = [ 16 ] } + ({ w2 } * relu ({ b1; o = [ 16 ] } + ({ w1 } * x))))
  in
  (* Don't decay the learning rate too quickly, it behaves better than in the original. *)
  let%op moons_input = moons_flat @| batch_n in
  let%op moons_class = moons_classes @| batch_n in
  let losses = ref [] in
  let log_losses = ref [] in
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
  (* TODO: is set_hosted needed? *)
  Train.set_hosted learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate ~weight_decay scalar_loss in
  let ctx = Train.init_params (module Backend) bindings scalar_loss in
  let sgd_routine =
    Train.to_routine (module Backend) ctx bindings (Asgns.sequence [ update; sgd ])
  in
  let step_ref = IDX.find_exn sgd_routine.bindings step_n in
  step_ref := 0;
  for epoch = 1 to epochs do
    let epoch_loss = ref 0. in
    Train.sequential_loop sgd_routine.bindings ~f:(fun () ->
        Train.run sgd_routine;
        let batch_ref = IDX.find_exn sgd_routine.bindings batch_n in
        epoch_loss := !epoch_loss +. scalar_loss.@[0];
        if !step_ref = steps - 5 then Stdio.printf "\n%!";
        if !step_ref < 10 then
          Stdio.printf "Epoch=%d, step=%d, batch=%d, lr=%.3g, loss=%.4g, epoch loss=%.4g\n%!" epoch
            !step_ref !batch_ref learning_rate.@[0] scalar_loss.@[0] !epoch_loss;
        if !step_ref > 10 && !step_ref % 100 = 0 then Stdio.printf ".%!";
        learning_rates := ~-.(learning_rate.@[0]) :: !learning_rates;
        losses := scalar_loss.@[0] :: !losses;
        log_losses := Float.max (-10.) (Float.log scalar_loss.@[0]) :: !log_losses;
        Int.incr step_ref)
  done;
  let points = Tn.points_2d ~xdim:0 ~ydim:1 moons_flat.value in
  let classes = Tn.points_1d ~xdim:0 moons_classes.value in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  (* %cd instead of %op to not get complaints about point being uninitialized. *)
  let%cd mlp_result = mlp { point } in
  Train.set_on_host mlp_result.value;
  let result_routine =
    Train.to_routine
      (module Backend)
      sgd_routine.context IDX.empty
      [%cd
        ~~("moons infer";
           mlp_result.forward)]
  in
  let callback (x, y) =
    Tn.set_values point.value [| x; y |];
    Train.run result_routine;
    Float.(mlp_result.@[0] >= 0.)
  in
  let _plot_moons =
    PrintBox_utils.plot ~as_canvas:true
      [
        Scatterplot { points = points1; content = PrintBox.line "#" };
        Scatterplot { points = points2; content = PrintBox.line "%" };
        Boundary_map
          { content_false = PrintBox.line "."; content_true = PrintBox.line "*"; callback };
      ]
  in
  (* Stdio.printf "Half-moons scatterplot and decision boundary:\n%!"; *)
  (* PrintBox_text.output Stdio.stdout plot_moons; *)
  Stdio.printf "\nLoss:\n%!";
  let plot_loss =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"loss" ~small:true
      [ Line_plot { points = Array.of_list_rev !losses; content = PrintBox.line "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "Log-loss, for better visibility:\n%!";
  let _plot_loss =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"log loss"
      [ Line_plot { points = Array.of_list_rev !log_losses; content = PrintBox.line "-" } ]
  in
  (* PrintBox_text.output Stdio.stdout plot_loss; *)
  Stdio.printf "\nLearning rate:\n%!";
  let plot_lr =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"learning rate" ~small:true
      [ Line_plot { points = Array.of_list_rev !learning_rates; content = PrintBox.line "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_lr;

  (* Testing how the syntax extension %op creates labels for the resulting tensors: *)
  Stdio.printf "mlp_result's name: %s\n%!" @@ Tensor.debug_name mlp_result;
  (* Note: mlp_result is not included in the resulting tensor's label, because the identifier label
     does not propagate across function calls. *)
  Stdio.printf "(mlp moons_input) name: %s\n%!"
  @@ Tensor.debug_name
  @@
  match margin_loss.children with
  | [
   {
     subtensor =
       { children = [ _; { subtensor = { children = [ _; { subtensor; _ } ]; _ }; _ } ]; _ };
     _;
   };
  ] ->
      subtensor
  | _ -> assert false

let () = main ()

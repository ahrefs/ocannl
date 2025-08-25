open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL

let main () =
  (* Micrograd half-moons example, with multi-stream execution simulating multi-device. *)
  let seed = 6 in
  Utils.settings.fixed_state_for_init <- Some seed;
  Tensor.unsafe_reinitialize ();
  let hid_dim = 16 in
  (* let hid_dim = 4 in *)
  (* Sensitive to batch size -- smaller batch sizes are better but cannot be too small because of
     splitting into stream minibatches. *)
  let batch_size = 20 in
  (* let batch_size = 60 in *)
  let len = batch_size * 20 in
  let init_lr = 0.1 in
  (* let epochs = 10 in *)
  let epochs = 120 in
  (* let epochs = 1 in *)
  let moons_config = Datasets.Half_moons.Config.{ noise_range = 0.1; seed = Some seed } in
  let moons_coordinates, moons_labels =
    Datasets.Half_moons.generate_single_prec ~config:moons_config ~len ()
  in
  let moons_flat_ndarray = Ir.Ndarray.as_array Ir.Ops.Single moons_coordinates in
  let moons_classes_ndarray = Ir.Ndarray.as_array Ir.Ops.Single moons_labels in
  let moons_flat = TDSL.rebatch ~l:"moons_flat" moons_flat_ndarray () in
  let moons_classes = TDSL.rebatch ~l:"moons_classes" moons_classes_ndarray () in
  let%op mlp x = { w3 } * relu ({ b2; o = [ hid_dim ] } + ({ w2 } * relu ({ b1; o = [ hid_dim ] } + ({ w1 } * x)))) in
  (* let%op mlp x = { b } + ("w" * x) in *)
  let%op loss_fn ~output ~expectation = relu (!..1 - (expectation *. output)) in
  (* We don't need a regression loss formula thanks to weight_decay built into the sgd_update
     computation. *)
  let weight_decay = 0.0002 in
  (* So that we can inspect them. *)
  let module Backend = (val Backends.fresh_backend ()) in
  let per_batch_callback ~at_batch:_ ~at_step:_ ~learning_rate:_ ~batch_loss:_ ~epoch_loss:_ = () in
  (* Tn.print_accessible_headers (); *)
  let epoch_loss_target_limits = [| 87.; 32.; 29.; 26.; 23.; 20.; 19.; 17.; 16.; 15. |] in
  let per_epoch_callback ~at_step:_ ~at_epoch ~learning_rate:_ ~epoch_loss =
    if at_epoch < 10 then
      Stdio.printf "Epoch=%d, loss under target %g: %b%s\n%!" at_epoch
        epoch_loss_target_limits.(at_epoch)
        Float.(epoch_loss_target_limits.(at_epoch) > epoch_loss)
        (if Float.(epoch_loss_target_limits.(at_epoch) > epoch_loss) then ""
         else ", actual loss: " ^ Float.to_string epoch_loss);
    if at_epoch > 10 && at_epoch % 10 = 0 then Stdio.printf ".%!"
  in
  let {
    Train.inputs;
    outputs;
    model_result = _;
    infer_callback;
    rev_batch_losses;
    rev_epoch_losses;
    learning_rates = _;
    used_memory = _;
  } =
    Train.example_train_loop ~seed ~batch_size ~max_num_streams:(batch_size / 2) ~init_lr
      ~data_len:len ~epochs ~inputs:moons_flat ~outputs:moons_classes ~model:mlp ~loss_fn
      ~weight_decay ~per_batch_callback ~per_epoch_callback
      (module Backend)
      ()
  in
  let epoch_loss = List.hd_exn rev_epoch_losses in
  Stdio.printf "\nFinal epoch loss under target 0.002: %b%s\n%!"
    Float.(0.002 > epoch_loss)
    (if Float.(0.002 > epoch_loss) then "" else ", actual loss: " ^ Float.to_string epoch_loss);
  (* if Float.(epoch_loss < 1.5) then Stdio.printf "Success\n" else *)
  let points = Tn.points_2d ~xdim:0 ~ydim:1 inputs.value in
  let classes = Tn.points_1d ~xdim:0 outputs.value in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  let callback (x, y) = Float.((infer_callback [| x; y |]).(0) >= 0.) in
  let _plot_moons =
    PrintBox_utils.plot ~as_canvas:true
      [
        Scatterplot { points = points1; content = PrintBox.line "#" };
        Scatterplot { points = points2; content = PrintBox.line "%" };
        Boundary_map
          { content_false = PrintBox.line "."; content_true = PrintBox.line "*"; callback };
      ]
  in
  (* Stdio.printf "\nHalf-moons scatterplot and decision boundary:\n"; *)
  (* PrintBox_text.output Stdio.stdout plot_moons; *)
  (* Stdio.printf "\nBatch Log-loss:\n%!"; *)
  let _plot_loss =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"batch log loss"
      [
        Line_plot
          {
            points =
              Array.of_list_rev_map rev_batch_losses ~f:Float.(fun x -> max (log 0.00003) (log x));
            content = PrintBox.line "-";
          };
      ]
  in
  (* PrintBox_text.output Stdio.stdout plot_loss; *)
  (* Stdio.printf "\nEpoch loss:\n%!"; *)
  let _plot_loss =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"epoch loss"
      [ Line_plot { points = Array.of_list_rev rev_epoch_losses; content = PrintBox.line "-" } ]
  in
  (* PrintBox_text.output Stdio.stdout plot_loss *)
  ()

let () = main ()

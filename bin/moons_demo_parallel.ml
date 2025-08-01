open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL

let experiment ~seed ~backend_name ~config () =
  let hid_dim = 16 in
  (* let hid_dim = 4 in *)
  let batch_size = 20 in
  (* let batch_size = 60 in *)
  (* let batch_size = 20 in *)
  let len = batch_size * 20 in
  let init_lr = 0.1 in
  let epochs = 100 in
  (* let epochs = 20 in *)
  (* let epochs = 1 in *)
  let moons_config = Datasets.Half_moons.Config.{ noise_range = 0.1; seed = Some seed } in
  let moons_coordinates, moons_labels = Datasets.Half_moons.generate ~config:moons_config ~len () in
  let moons_flat_ndarray = Ir.Ndarray.as_array Ir.Ops.Double moons_coordinates in
  let moons_classes_ndarray = Ir.Ndarray.as_array Ir.Ops.Double moons_labels in
  let moons_flat = TDSL.rebatch ~l:"moons_flat" moons_flat_ndarray () in
  let moons_classes = TDSL.rebatch ~l:"moons_classes" moons_classes_ndarray () in
  let%op mlp x = "w3" * relu ("b2" hid_dim + ("w2" * relu ("b1" hid_dim + ("w1" * x)))) in
  (* let%op mlp x = ("w" * x) in *)
  let%op loss_fn ~output ~expectation = relu (!..1 - (expectation *. output)) in
  (* We don't need a regression loss formula thanks to weight_decay built into the sgd_update
     computation. *)
  let weight_decay = 0.0002 in
  (* So that we can inspect them. *)
  let module Backend = (val Backends.fresh_backend ~backend_name ~config ()) in
  let per_batch_callback ~at_batch ~at_step ~learning_rate ~batch_loss ~epoch_loss =
    if (at_batch + 1) % 20 = 0 then
      Stdio.printf "Batch=%d, step=%d, lr=%f, batch loss=%f, epoch loss=%f\n%!" at_batch at_step
        learning_rate batch_loss epoch_loss
  in
  (* Tn.print_accessible_headers (); *)
  let per_epoch_callback ~at_step ~at_epoch ~learning_rate ~epoch_loss =
    Stdio.printf "Epoch=%d, step=%d, lr=%f, epoch loss=%f\n%!" at_epoch at_step learning_rate
      epoch_loss
  in
  let {
    Train.inputs;
    outputs;
    model_result;
    infer_callback;
    rev_batch_losses;
    rev_epoch_losses;
    learning_rates;
    used_memory;
  } =
    Train.example_train_loop ~seed ~batch_size ~max_num_streams:(batch_size / 2) ~init_lr
      ~data_len:len ~epochs ~inputs:moons_flat ~outputs:moons_classes ~model:mlp ~loss_fn
      ~weight_decay ~per_batch_callback ~per_epoch_callback
      (module Backend)
      ()
  in
  let points = Tn.points_2d ~xdim:0 ~ydim:1 inputs.value in
  let classes = Tn.points_1d ~xdim:0 outputs.value in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  Stdio.print_endline "\n******** mlp_result **********";
  Train.printf_tree ~with_grad:false ~depth:9 model_result;
  Stdio.printf "\n********\nUsed memory: %d\n%!" used_memory;
  let callback (x, y) = Float.((infer_callback [| x; y |]).(0) >= 0.) in
  let plot_moons =
    PrintBox_utils.plot ~as_canvas:true
      [
        Scatterplot { points = points1; content = PrintBox.line "#" };
        Scatterplot { points = points2; content = PrintBox.line "%" };
        Boundary_map
          { content_false = PrintBox.line "."; content_true = PrintBox.line "*"; callback };
      ]
  in
  Stdio.printf "\nHalf-moons scatterplot and decision boundary:\n%!";
  PrintBox_text.output Stdio.stdout plot_moons;
  Stdio.printf "\nBatch Loss:\n%!";
  let plot_loss =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"batch loss"
      [ Line_plot { points = Array.of_list_rev rev_batch_losses; content = PrintBox.line "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nEpoch Loss:\n%!";
  let plot_loss =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"epoch loss"
      [ Line_plot { points = Array.of_list_rev rev_epoch_losses; content = PrintBox.line "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nBatch Log-loss:\n%!";
  let plot_loss =
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
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nEpoch Log-loss:\n%!";
  let plot_loss =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"epoch log loss"
      [
        Line_plot
          {
            points = Array.of_list_rev_map rev_epoch_losses ~f:Float.log;
            content = PrintBox.line "-";
          };
      ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nLearning rate:\n%!";
  let plot_lr =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"learning rate"
      [ Line_plot { points = Array.of_list_rev learning_rates; content = PrintBox.line "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_lr

let () = experiment ~seed:1 ~backend_name:"multicore_cc" ~config:Only_devices_parallel ()
let _suspended () = experiment ~seed:1 ~backend_name:"multicore_cc" ~config:Only_devices_parallel ()
let _suspended () = experiment ~seed:1 ~backend_name:"cuda" ~config:Most_parallel_streams ()

let _suspended () =
  for seed = 0 to 19 do
    Stdio.printf "\n*************** EXPERIMENT SEED %d ******************\n%!" seed;
    experiment ~seed ~backend_name:"multicore_cc" ~config:Only_devices_parallel ()
  done

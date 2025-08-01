open Base
open Ocannl
module Tn = Ir.Tnode
module Asgns = Ir.Assignments
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL

let _get_local_debug_runtime = Utils.get_local_debug_runtime

let experiment seed ~no_batch_shape_inference ~use_builtin_weight_decay () =
  let hid_dim = 16 in
  let len = 300 in
  let batch_size = 20 in
  let n_batches = 2 * len / batch_size in
  let epochs = 75 in
  let steps = epochs * n_batches in
  (* let weight_decay = 0.0002 in *)
  let moons_config = Datasets.Half_moons.Config.{ noise_range = 0.1; seed = Some seed } in
  let moons_coordinates, moons_labels = Datasets.Half_moons.generate ~config:moons_config ~len () in
  let moons_flat_ndarray = Ir.Ndarray.as_array Ir.Ops.Double moons_coordinates in
  let moons_classes_ndarray = Ir.Ndarray.as_array Ir.Ops.Double moons_labels in
  let moons_flat = TDSL.rebatch ~l:"moons_flat" moons_flat_ndarray () in
  let moons_classes = TDSL.rebatch ~l:"moons_classes" moons_classes_ndarray () in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  let%op mlp x = "w3" * relu ("b2" hid_dim + ("w2" * relu ("b1" hid_dim + ("w1" * x)))) in
  let%op moons_input = moons_flat @| batch_n in
  (* Tell shape inference to make a minibatch axis. *)
  let () =
    if no_batch_shape_inference then
      let%cd _ = moons_input =: 0 ++ "i=>2|i" in
      ()
  in
  let%op moons_class = moons_classes @| batch_n in
  let () =
    if no_batch_shape_inference then
      let%cd _ = moons_class =: 0 ++ "i=>2|i" in
      ()
  in
  let losses = ref [] in
  let log_losses = ref [] in
  let learning_rates = ref [] in
  let%op margin_loss = relu (1 - (moons_class *. mlp moons_input)) in
  (* We don't need a regression loss formula thanks to weight_decay built into the sgd_update
     computation. *)
  let scalar_loss, weight_decay =
    if use_builtin_weight_decay then
      let%op scalar_loss = (margin_loss ++ "...|... => 0") /. !..batch_size in
      (scalar_loss, 0.0002)
    else
      let%op ssq w = (w **. 2) ++ "...|...->... => 0" in
      let reg_loss = List.map ~f:ssq [ w1; w2; w3; b1; b2 ] |> List.reduce_exn ~f:TDSL.O.( + ) in
      let%op scalar_loss =
        ((margin_loss ++ "...|... => 0") /. !..batch_size) + (0.0001 *. reg_loss)
      in
      (scalar_loss, 0.0)
  in
  (* So that we can inspect them. *)
  let update = Train.grad_update scalar_loss in
  let%op learning_rate = 0.1 *. (!..steps - !@step_n) /. !..steps in
  Train.set_hosted learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate ~weight_decay scalar_loss in

  let module Backend = (val Backends.fresh_backend ~backend_name:"multicore_cc" ()) in
  let ctx = Train.init_params (module Backend) ~hosted:true IDX.empty scalar_loss in
  let routine = Train.to_routine (module Backend) ctx bindings (Asgns.sequence [ update; sgd ]) in
  (* Stdio.print_endline "\n******** scalar_loss **********"; Train.printf_tree ~with_grad:false
     ~depth:9 scalar_loss; Stdio.print_endline "\n******** learning_rate **********";
     Train.printf_tree ~with_grad:false ~depth:9 learning_rate; Stdio.printf "\n********\n%!"; *)
  let open Operation.At in
  let epoch_loss = ref 0. in
  let step_ref = IDX.find_exn routine.bindings step_n in
  let batch_ref = IDX.find_exn routine.bindings batch_n in
  step_ref := 0;
  let%track3_sexp _train_loop : unit =
    (* Tn.print_accessible_headers (); *)
    for epoch = 0 to epochs - 1 do
      for batch = 0 to n_batches - 1 do
        batch_ref := batch;
        Train.run routine;
        (* Stdio.printf "Data batch=%d, step=%d, lr=%f, batch loss=%f\n%!" !batch_ref !step_ref
           learning_rate.@[0] scalar_loss.@[0]; *)
        learning_rates := learning_rate.@[0] :: !learning_rates;
        losses := scalar_loss.@[0] :: !losses;
        epoch_loss := !epoch_loss +. scalar_loss.@[0];
        log_losses := Float.log scalar_loss.@[0] :: !log_losses;
        Int.incr step_ref
      done;
      if epoch % 1000 = 0 || epoch = epochs - 1 then
        Stdio.printf "Epoch %d, lr=%f, epoch loss=%f\n%!" epoch learning_rate.@[0] !epoch_loss;
      epoch_loss := 0.
    done
  in
  let points = Tn.points_2d ~xdim:0 ~ydim:1 moons_flat.value in
  let classes = Tn.points_1d ~xdim:0 moons_classes.value in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  let%cd mlp_result = mlp "point" in
  Train.set_on_host mlp_result.value;
  (* By using jitted.context here, we don't need to copy the parameters back to the host. *)
  let result_routine =
    Train.to_routine
      (module Backend)
      routine.context IDX.empty
      [%cd
        ~~("moons infer";
           mlp_result.forward)]
  in
  Stdio.print_endline "\n******** mlp_result **********";
  Train.printf_tree ~with_grad:false ~depth:9 mlp_result;
  Stdio.printf "\n********\n%!";
  let callback (x, y) =
    Tn.set_values point.value [| x; y |];
    (* For the gccjit backend, point is only on host, not on device. For cuda, this will be
       needed. *)
    Train.run result_routine;
    Float.(mlp_result.@[0] >= 0.)
  in
  let%track3_sexp _plotting : unit =
    let plot_moons =
      PrintBox_utils.plot ~as_canvas:true
        [
          Scatterplot { points = points1; content = PrintBox.line "#" };
          Scatterplot { points = points2; content = PrintBox.line "%" };
          Boundary_map
            { content_false = PrintBox.line "."; content_true = PrintBox.line "*"; callback };
        ]
    in
    Stdio.printf "Half-moons scatterplot and decision boundary:\n%!";
    PrintBox_text.output Stdio.stdout plot_moons
  in
  Stdio.printf "Loss:\n%!";
  let plot_loss =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"loss"
      [ Line_plot { points = Array.of_list_rev !losses; content = PrintBox.line "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;

  Stdio.printf "Log-loss, for better visibility:\n%!";
  let plot_loss =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"log loss"
      [ Line_plot { points = Array.of_list_rev !log_losses; content = PrintBox.line "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;

  Stdio.printf "\nLearning rate:\n%!";
  let plot_lr =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"learning rate"
      [ Line_plot { points = Array.of_list_rev !learning_rates; content = PrintBox.line "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_lr

let () = experiment 4 ~no_batch_shape_inference:true ~use_builtin_weight_decay:true ()
let _suspended () = experiment 4 ~no_batch_shape_inference:false ~use_builtin_weight_decay:false ()

let _suspended () =
  for seed = 0 to 19 do
    Stdio.printf "\n*************** EXPERIMENT SEED %d ******************\n%!" seed;
    experiment seed ~no_batch_shape_inference:true ~use_builtin_weight_decay:true ()
  done

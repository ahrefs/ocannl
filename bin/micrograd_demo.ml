open Base
open Ocannl
module Tn = Arrayjit.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Utils = Arrayjit.Utils
module Rand = Arrayjit.Rand.Lib

let experiment seed ~no_batch_shape_inference ~use_builtin_weight_decay () =
  Rand.init 0;
  Utils.settings.with_debug_level <- 1;

  (* Utils.settings.output_debug_files_in_run_directory <- true; *)
  (* Utils.settings.debug_log_from_routines <- true; *)
  let hid_dim = 16 in
  let len = 300 in
  let batch_size = 20 in
  let n_batches = 2 * len / batch_size in
  let epochs = 75 in
  let steps = epochs * n_batches in
  (* let weight_decay = 0.0002 in *)
  Utils.settings.fixed_state_for_init <- Some seed;
  let noise () = Rand.float_range (-0.1) 0.1 in
  let moons_flat =
    Array.concat_map (Array.create ~len ())
      ~f:
        Float.(
          fun () ->
            let i = Rand.int len in
            let v = of_int i * pi / of_int len in
            let c = cos v and s = sin v in
            [| c + noise (); s + noise (); 1.0 - c + noise (); 0.5 - s + noise () |])
  in
  let b = if no_batch_shape_inference then Some [ n_batches; batch_size ] else None in
  let moons_flat = TDSL.init_const ~l:"moons_flat" ?b ~o:[ 2 ] moons_flat in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes = TDSL.init_const ~l:"moons_classes" ?b ~o:[ 1 ] moons_classes in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  let%op mlp x = "b3" + ("w3" * ?/("b2" hid_dim + ("w2" * ?/("b1" hid_dim + ("w1" * x))))) in
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
  let%op margin_loss = ?/(1 - (moons_class *. mlp moons_input)) in
  (* We don't need a regression loss formula thanks to weight_decay built into the sgd_update computation. *)
  let scalar_loss, weight_decay =
    if use_builtin_weight_decay then
      let%op scalar_loss = (margin_loss ++ "...|... => 0") /. !..batch_size in
      (scalar_loss, 0.0002)
    else
      let%op ssq w = (w **. 2) ++ "...|...->... => 0" in
      let reg_loss = List.map ~f:ssq [ w1; w2; w3; b1; b2; b3 ] |> List.reduce_exn ~f:TDSL.O.( + ) in
      let%op scalar_loss = ((margin_loss ++ "...|... => 0") /. !..batch_size) + (0.0001 *. reg_loss) in
      (scalar_loss, 0.0)
  in
  (* So that we can inspect them. *)
  let update = Train.grad_update scalar_loss in
  let%op learning_rate = 0.1 *. (!..steps - !@step_n) /. !..steps in
  Train.set_hosted learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate ~weight_decay update in

  let module Backend = (val Train.fresh_backend ()) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let routine = Backend.(link ctx @@ compile bindings (Seq (update.fwd_bprop, sgd))) in
  Train.all_host_to_device (module Backend) routine.context scalar_loss;
  Train.all_host_to_device (module Backend) routine.context learning_rate;
  (* Stdio.print_endline "\n******** scalar_loss **********"; Tensor.print_tree ~with_id:true ~with_grad:false
     ~depth:9 scalar_loss; Stdio.print_endline "\n******** learning_rate **********"; Tensor.print_tree
     ~with_id:true ~with_grad:false ~depth:9 learning_rate; Stdio.printf "\n********\n%!"; *)
  let open Operation.At in
  let epoch_loss = ref 0. in
  let step_ref = IDX.find_exn routine.bindings step_n in
  let batch_ref = IDX.find_exn routine.bindings batch_n in
  step_ref := 0;
  (* Tn.print_accessible_headers (); *)
  for epoch = 0 to epochs - 1 do
    for batch = 0 to n_batches - 1 do
      batch_ref := batch;
      Train.run routine;
      Backend.await device;
      Backend.to_host routine.context learning_rate.value;
      Backend.to_host routine.context scalar_loss.value;
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
  done;
  let points = Tensor.value_2d_points ~xdim:0 ~ydim:1 moons_flat in
  let classes = Tensor.value_1d_points ~xdim:0 moons_classes in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  let%op mlp_result = mlp "point" in
  Train.set_on_host Volatile mlp_result.value;
  (* By using jitted.context here, we don't need to copy the parameters back to the host. *)
  let result_routine =
    Backend.(link routine.context @@ compile IDX.empty @@ Block_comment ("moons infer", mlp_result.forward))
  in
  Stdio.print_endline "\n******** mlp_result **********";
  Tensor.print_tree ~with_id:true ~with_grad:false ~depth:9 mlp_result;
  Stdio.printf "\n********\n%!";
  let callback (x, y) =
    Tensor.set_values point [| x; y |];
    (* For the gccjit backend, point is only on host, not on device. For cuda, this will be needed. *)
    Backend.from_host result_routine.context point.value;
    Train.run result_routine;
    Backend.await device;
    Backend.to_host result_routine.context mlp_result.value;
    Float.(mlp_result.@[0] >= 0.)
  in
  let plot_moons =
    let open PrintBox_utils in
    plot ~size:(120, 40) ~x_label:"ixes" ~y_label:"ygreks"
      [
        Scatterplot { points = points1; pixel = "#" };
        Scatterplot { points = points2; pixel = "%" };
        Boundary_map { pixel_false = "."; pixel_true = "*"; callback };
      ]
  in
  Stdio.printf "Half-moons scatterplot and decision boundary:\n%!";
  PrintBox_text.output Stdio.stdout plot_moons;
  Stdio.printf "Loss:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"loss"
      [ Line_plot { points = Array.of_list_rev !losses; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;

  Stdio.printf "Log-loss, for better visibility:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"log loss"
      [ Line_plot { points = Array.of_list_rev !log_losses; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;

  Stdio.printf "\nLearning rate:\n%!";
  let plot_lr =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"learning rate"
      [ Line_plot { points = Array.of_list_rev !learning_rates; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_lr

let () = experiment 4 ~no_batch_shape_inference:true ~use_builtin_weight_decay:true ()

let _suspended () =
  for seed = 0 to 19 do
    Stdio.printf "\n*************** EXPERIMENT SEED %d ******************\n%!" seed;
    experiment seed ~no_batch_shape_inference:true ~use_builtin_weight_decay:true ()
  done

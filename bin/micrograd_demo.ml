open Base
open Ocannl
module LA = Arrayjit.Lazy_array
module IDX = Arrayjit.Indexing.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Arrayjit.Low_level.CDSL

let () =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  CDSL.with_debug := true;
  CDSL.keep_files_in_run_directory := true;
  (* CDSL.debug_log_jitted := true; *)
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  CDSL.fixed_state_for_init := Some 4;
  let hid_dim = 16 in
  let len = 200 in
  let batch = 20 in
  let n_batches = 2 * len / batch in
  let epochs = 20 in
  let steps = epochs * n_batches in
  let noise () = Random.float_range (-0.1) 0.1 in
  let moons_flat =
    Array.concat_map (Array.create ~len ())
      ~f:
        Float.(
          fun () ->
            let i = Random.int len in
            let v = of_int i * pi / of_int len in
            let c = cos v and s = sin v in
            [| c + noise (); s + noise (); 1.0 - c + noise (); 0.5 - s + noise () |])
  in
  let moons_flat = TDSL.init_const ~l:"moons_flat" ~b:[ n_batches; batch ] ~o:[ 2 ] moons_flat in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes = TDSL.init_const ~l:"moons_classes" ~b:[ n_batches; batch ] ~o:[ 1 ] moons_classes in
  let step_sym, step_ref, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let%op mlp x = "b3" 1 + ("w3" * !/("b2" hid_dim + ("w2" * !/("b1" hid_dim + ("w1" * x))))) in
  let%op learning_rate = 0.1 *. (!..steps - !@step_sym) /. !..steps in
  (* FIXME: Shape.broadcast shouldn't be needed here! Aaargh... *)
  Shape.broadcast learning_rate.shape;
  let%op moons_input = moons_flat @| step_sym in
  let%op moons_class = moons_classes @| step_sym in
  let losses = ref [] in
  let log_losses = ref [] in
  let learning_rates = ref [] in
  let%op margin_loss = !/(1 - (moons_class *. mlp moons_input)) in
  (* We don't need a regression loss formula thanks to weight_decay built into the sgd_update computation. *)
  let weight_decay = 0.0001 in
  let%op scalar_loss = (margin_loss ++ "...|... => 0") /. !..batch in
  (* So that we can inspect them. *)
  Train.set_fully_on_host scalar_loss.value;
  Train.set_fully_on_host learning_rate.value;
  let update = Train.grad_update scalar_loss in
  let sgd = Train.sgd_update ~learning_rate ~weight_decay scalar_loss in
  let sgd_jitted = Backend.jit ctx ~verbose:true bindings (Seq (update, sgd)) in
  Train.all_host_to_device (module Backend) sgd_jitted.context scalar_loss;
  Train.all_host_to_device (module Backend) sgd_jitted.context learning_rate;
  (* Stdio.print_endline "\n******** scalar_loss **********";
     Tensor.print_tree ~with_id:true ~with_grad:false ~depth:9 scalar_loss;
     Stdio.print_endline "\n******** learning_rate **********";
     Tensor.print_tree ~with_id:true ~with_grad:false ~depth:9 learning_rate;
     Stdio.printf "\n********\n%!"; *)
  (* * step_ref :=0;
     sgd_jitted.run ();
     Backend.await device;
     Train.all_device_to_host (module Backend) sgd_jitted.context scalar_loss;
     Stdio.print_endline "\n******** scalar_loss **********";
     Tensor.print_tree ~with_id:true ~with_grad:false ~depth:9 scalar_loss;
     Stdio.print_endline "\n******** learning_rate **********";
     Tensor.print_tree ~with_id:true ~with_grad:false ~depth:9 learning_rate;
     Stdio.printf "\n********\n%!"
     * *)
  let open Tensor.O in
  (* Alternative flat loop:
     for step = 0 to steps - 1 do
       step_ref := step % n_batches;
       sgd_jitted.run ();
       ...
       Stdio.printf "Step=%d, lr=%f, loss=%f\n%!" step learning_rate.@[0] scalar_loss.@[0];
       ...
  *)
  let epoch_loss = ref 0. in
  for epoch = 1 to epochs do
    Train.for_loop bindings ~f:(fun () ->
        sgd_jitted.run ();
        Backend.await device;
        assert (Backend.to_host sgd_jitted.context learning_rate.value);
        assert (Backend.to_host sgd_jitted.context scalar_loss.value);
        (* Stdio.printf "Data step=%d, lr=%f, loss=%f\n%!" !step_ref learning_rate.@[0] scalar_loss.@[0]; *)
        ignore step_ref;
        learning_rates := ~-.(learning_rate.@[0]) :: !learning_rates;
        losses := scalar_loss.@[0] :: !losses;
        epoch_loss := !epoch_loss +. scalar_loss.@[0];
        log_losses := Float.log scalar_loss.@[0] :: !log_losses);
    Stdio.printf "Epoch %d, lr=%f, loss=%f\n%!" epoch learning_rate.@[0] !epoch_loss;
    epoch_loss := 0.
  done;
  let points = Tensor.value_2d_points ~xdim:0 ~ydim:1 moons_flat in
  let classes = Tensor.value_1d_points ~xdim:0 moons_classes in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  let%op point = [ 0; 0 ] in
  let mlp_result = mlp point in
  Train.set_fully_on_host point.value;
  Train.set_fully_on_host mlp_result.value;
  let result_jitted =
    Backend.jit ctx (* sgd_jitted.context *) IDX.empty @@ Block_comment ("moons infer", mlp_result.forward)
  in
  Stdio.print_endline "\n******** mlp_result **********";
  Tensor.print_tree ~with_id:true ~with_grad:false ~depth:9 mlp_result;
  Stdio.printf "\n********\n%!";
  let callback (x, y) =
    Tensor.set_values point [| x; y |];
    (* For the gccjit backend, point is only on host, not on device. For cuda, this will be needed. *)
    ignore (Backend.from_host result_jitted.context point.value : bool);
    result_jitted.run ();
    Backend.await device;
    assert (Backend.to_host result_jitted.context mlp_result.value);
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

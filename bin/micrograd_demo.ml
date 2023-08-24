open Base
open Ocannl
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Arrayjit.Low_level.CDSL


let () = SDSL.set_executor Cuda

let () =
  let open Tensor.O in
  (* SDSL.drop_all_sessions (); *)
  (* CDSL.with_debug := true; *)
  (* CDSL.keep_files_in_run_directory := true; *)
  (* Low_level.debug_verbose_trace := true; *)
  Random.init 0;
  (* The seeds 0, 6, 8 are unlucky. Seeds 2-5, 7, 9 are good. From better to worse: 4, 2, 9, 7, 1, 5, 3. *)
  CDSL.fixed_state_for_init := Some 4;
  let hid_dim = 16 in
  let len = 200 in
  let batch = 20 in
  let n_batches = 2 * len / batch in
  let epochs = 75 in
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
  let moons_flat =
    TDSL.init_const ~l:"moons_flat"
      ~b:[ n_batches; batch ]
      ~o:[ 2 ]
      moons_flat
  in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes =
    TDSL.init_const ~l:"moons_classes"
      ~b:[ n_batches; batch ]
      ~o:[ 1 ]
      moons_classes
  in
  let%nn_op mlp x = "b3" 1 + ("w3" * !/("b2" hid_dim + ("w2" * !/("b1" hid_dim + ("w1" * x))))) in
  let session_step = NTDSL.O.(NTDSL.counter !..1) in
  let%nn_op minus_lr = -0.1 *. (!..steps - session_step) /. !..steps in
  SDSL.minus_learning_rate := Some minus_lr;
  let%nn_op moons_input = moons_flat @.| session_step in
  let%nn_op moons_class = moons_classes @.| session_step in
  let losses = ref [] in
  let log_losses = ref [] in
  let learning_rates = ref [] in
  let%nn_op margin_loss = !/(1 - (moons_class *. mlp moons_input)) in
  let%nn_op ssq w = (w **. 2) ++ "...|...->... => 0" in
  let reg_loss = List.map ~f:ssq [ w1; w2; w3; b1; b2; b3 ] |> List.reduce_exn ~f:TDSL.O.( + ) in
  let%nn_op total_loss = ((margin_loss ++ "...|... => 0") /. !..batch) + (0.0001 *. reg_loss) in
  (* SDSL.everything_on_host_or_inlined (); *)
  for step = 1 to steps do
    SDSL.refresh_session ();
    if step % (len / batch) = 1 || step = steps then
      Stdio.printf "Step=%d, session_step=%f, -lr=%f, loss=%f\n%!" step session_step.@[0] minus_lr.@[0]
        total_loss.@[0]
      (* SDSL.print_tree ~with_backend_info:true ~with_grad:true ~depth:9 total_loss *);
    learning_rates := ~-.(minus_lr.@[0]) :: !learning_rates;
    losses := total_loss.@[0] :: !losses;
    log_losses := Float.log total_loss.@[0] :: !log_losses
  done;
  CDSL.with_debug := false;
  CDSL.keep_files_in_run_directory := false;
  let points = SDSL.value_2d_points ~xdim:0 ~ydim:1 moons_flat in
  let classes = SDSL.value_1d_points ~xdim:0 moons_classes in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  SDSL.close_session ();
  let%nn_op point = [ 0; 0 ] in
  let mlp_result = mlp point in
  SDSL.refresh_session ~with_backprop:false ();
  let callback (x, y) =
    SDSL.set_values point [| x; y |];
    SDSL.refresh_session ~with_backprop:false ();
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

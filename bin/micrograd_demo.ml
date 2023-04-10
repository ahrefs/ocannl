open Base
open Ocannl
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL
module CDSL = Code.CDSL

let () = Session.SDSL.set_executor OCaml

let () =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_all_sessions();
  Random.init 0;
  let len = 200 in
  let batch = 10 in
  let epochs = 40 in
  let noise() = Random.float_range (-0.1) 0.1 in
  (* TODO: convert it to [Synthetic] data. *)
  let moons_flat = Array.concat_map (Array.create ~len ()) ~f:Float.(fun () ->
    let i = Random.int len in
    let v = of_int i * pi / of_int len in
    let c = cos v and s = sin v in
    [|c + noise(); s + noise(); 1.0 - c + noise(); 0.5 - s + noise()|]) in
  let moons_flat = FDSL.term ~label:"moons_flat" ~batch_dims:[epochs; batch] ~output_dims:[2] @@
      First (Constant_fill moons_flat) in
  let moons_classes = Array.init (len*2) ~f:(fun i -> if i % 2 = 0 then 1. else (-1.)) in
  let moons_classes = FDSL.term ~label:"moons_flat" ~batch_dims:[epochs; batch] ~output_dims:[1] @@
      First (Constant_fill moons_classes) in
  let%nn_op mlp x = "b3" 1 + "w3" * !/ ("b2" 16 + "w2" * !/ ("b1" 16 + "w1" * x)) in
  let steps = epochs * 2 * len/batch in
  let session_step = FDSL.data ~label:"session_step" ~batch_dims:[] ~output_dims:[1]
      (fun ~n -> Synthetic [%nn_cd n =+ ~= 1 ~logic:"."]) in
  minus_learning_rate := Some (
      FDSL.data ~label:"minus_lr" ~batch_dims:[] ~output_dims:[1]
        (fun ~n -> Synthetic
            [%nn_cd n =: ~= (-0.1 *. (!..steps - session_step) /. !..steps) ~logic:"."]));
  let moons_input = FDSL.data ~label:"moons_input" ~batch_dims:[batch] ~output_dims:[2]
      (fun ~n -> Synthetic [%nn_cd n =: moons_flat @.. session_step ~logic:"."]) in
  let moons_class = FDSL.data ~label:"moons_class" ~batch_dims:[batch] ~output_dims:[1]
      (fun ~n -> Synthetic [%nn_cd n =: moons_classes @.. session_step ~logic:"."]) in
  let points1 = ref [] in
  let points2 = ref [] in
  let losses = ref [] in
  let log_losses = ref [] in
  let learning_rates = ref [] in
  let%nn_op margin_loss = !/ (1 - moons_class *. mlp moons_input) in
  let%nn_op ssq w = (w **. 2) ++"...|...->... => 0" in
  let reg_loss = List.map ~f:ssq [w1; w2; w3; b1; b2; b3] |> List.reduce_exn ~f:FDSL.O.(+) in
  let%nn_op total_loss = (margin_loss ++"...|... => 0") /. !..batch + 0.0001 *. reg_loss in
  for step = 1 to steps do
    refresh_session ();
    Option.value_exn !update_params ();
    let points = NodeUI.retrieve_2d_points ~xdim:0 ~ydim:1 moons_input.node.node.value in
    let classes = NodeUI.retrieve_1d_points ~xdim:0 moons_class.node.node.value in
    let npoints1, npoints2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
    points1 := npoints1 :: !points1;
    points2 := npoints2 :: !points2;
    let mlr = NodeUI.retrieve_1d_points ~xdim:0 
        (Option.value_exn !minus_learning_rate).node.node.value in
    assert (Array.length mlr = 1);
    Stdio.printf "Minus learning rate over batch for step %d: %f\n%!" step mlr.(0);
    learning_rates := ~-. (mlr.(0)) :: !learning_rates;
    let batch_loss = NodeUI.retrieve_1d_points ~xdim:0 total_loss.node.node.value in
    assert (Array.length batch_loss = 1);
    Stdio.printf "Loss over batch for step %d: %f\n%!" step batch_loss.(0);
    losses := batch_loss.(0) :: !losses;
    log_losses := Float.log batch_loss.(0) :: !log_losses;
  done;
  close_session ();
  let%nn_op point = [0; 0] in
  let mlp_result = mlp point in
  let callback (x, y) =
    Ocannl_runtime.Node.(set_from_float point.node.node.value [|0|] x;
                         set_from_float point.node.node.value [|1|] y);
    refresh_session ();
    let result = NodeUI.retrieve_1d_points ~xdim:0 mlp_result.node.node.value in
    Float.(result.(0) >= 0.) in
  let plot_moons = 
    let open PrintBox_utils in
    plot ~size:(120, 40) ~x_label:"ixes" ~y_label:"ygreks"
      [Scatterplot {points=Array.concat !points1; pixel="#"}; 
       Scatterplot {points=Array.concat !points2; pixel="%"};
       Boundary_map {pixel_false="."; pixel_true="*"; callback}] in
  Stdio.printf "Half-moons scatterplot and decision boundary:\n%!";
  PrintBox_text.output Stdio.stdout plot_moons;
  Stdio.printf "Loss curve:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"loss"
      [Line_plot {points=Array.of_list_rev !losses; pixel="-"}] in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "Log-loss, for better visibility:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"log loss"
      [Line_plot {points=Array.of_list_rev !log_losses; pixel="-"}] in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nLearning rate:\n%!";
  let plot_lr =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"learning rate"
      [Line_plot {points=Array.of_list_rev !learning_rates; pixel="-"}] in
  PrintBox_text.output Stdio.stdout plot_lr;
  Stdio.printf "\n%!"

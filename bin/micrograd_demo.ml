open Base
open Ocannl
module FDSL = Operation.FDSL

let () = Session.SDSL.set_executor OCaml

let () =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_all_sessions();
  Random.init 0;
  let len = 200 in
  let batch = 10 in
  let epochs = 200 in
  let noise() = Random.float_range (-0.1) 0.1 in
  let moons_flat = Array.concat_map (Array.create ~len ()) ~f:Float.(fun () ->
    let i = Random.int len in
    let v = of_int i * pi / of_int len in
    let c = cos v and s = sin v in
    [|c + noise(); s + noise(); 1.0 - c + noise(); 0.5 - s + noise()|]) in
  let moons_classes = Array.init (len*2) ~f:(fun i -> if i % 2 = 0 then 1. else (-1.)) in
  let moons_input = FDSL.data ~label:"moons_input" ~batch_dims:[batch] ~output_dims:[2]
      (Init_op (Fixed_constant moons_flat)) in
  let moons_class = FDSL.data ~label:"moons_class" ~batch_dims:[batch] ~output_dims:[1]
      (Init_op (Fixed_constant moons_classes)) in
  let%nn_op mlp x = "b3" 1 + "w3" * !/ ("b2" 16 + "w2" * !/ ("b1" 16 + "w1" * x)) in
  let steps = epochs * 2 * len/batch in
  minus_learning_rate := Some (
      FDSL.data ~label:"minus_lr" ~batch_dims:[] ~output_dims:[1]
        Float.(Compute_point (fun ~session_step ~dims:_ ~idcs:_ ->
            (0.999 * of_int session_step / of_int steps - 1.) / 50.)));
  let points1 = ref [] in
  let points2 = ref [] in
  let losses = ref [] in
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
    if step % 50 = 0 then
      let mlr = NodeUI.retrieve_1d_points ~xdim:0 
          (Option.value_exn !minus_learning_rate).node.node.value in
      assert (Array.length mlr = 1);
      Stdio.printf "Minus learning rate over batch for step %d: %f\n%!" step mlr.(0);
      learning_rates := ~-. (mlr.(0)) :: !learning_rates;
      let batch_loss = NodeUI.retrieve_1d_points ~xdim:0 total_loss.node.node.value in
      assert (Array.length batch_loss = 1);
      Stdio.printf "Loss over batch for step %d: %f\n%!" step batch_loss.(0);
      losses := batch_loss.(0) :: !losses;
  done;
  close_session ();
  let point = [|0.; 0.|] in
  let point_input = FDSL.data ~label:"point_input" ~batch_dims:[1] ~output_dims:[2]
      (Compute_point (fun ~session_step:_ ~dims:_ ~idcs -> point.(idcs.(1)))) in
  let mlp_result = mlp point_input in
  let callback (x, y) =
    point.(0) <- x; point.(1) <- y;
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
    plot ~size:(120, 30) ~x_label:"step/50" ~y_label:"loss"
      [Line_plot {points=Array.of_list_rev !losses; pixel="-"}] in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nLearning rate curve:\n%!";
  let plot_lr =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step/50" ~y_label:"learning rate"
      [Line_plot {points=Array.of_list_rev !learning_rates; pixel="-"}] in
  PrintBox_text.output Stdio.stdout plot_lr;
  Stdio.printf "\n%!"

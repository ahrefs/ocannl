open Base
open Ocannl
module FDSL = Operation.FDSL

let () = Session.SDSL.set_executor OCaml

let _suspended () =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_all_sessions();
  Random.init 0;
  let len = 10 in
  let batch = 2 in
  let epochs = 1 in
  let noise() = Random.float_range (-0.1) 0.1 in
  let moons_flat = Array.concat_map (Array.create ~len ()) ~f:Float.(fun () ->
    let i = Random.int len in
    let v = of_int i * pi / of_int len in
    let c = cos v and s = sin v in
    [|s + noise(); c + noise(); 0.5 - s + noise(); 1.0 - c + noise()|]) in
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
  (* Although [mlp] is not yet applied to anything, we can already compile the weight updates,
     because the parameters are already created by parameter punning. *)
  let points1 = ref [] in
  let points2 = ref [] in
  let losses = ref [] in
  let%nn_op margin_loss = !/ (1 - moons_class *. mlp moons_input) in
  let%nn_op ssq w = (w **. 2) ++"...|...->... => 0" in
  let reg_loss = List.map ~f:ssq [w1; w2; w3; b1; b2; b3] |> List.reduce_exn ~f:FDSL.O.(+) in
  let%nn_op total_loss = margin_loss + 0.0001 *. reg_loss in
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
    Stdio.printf "\nMinus learning rate over batch for step %d: %s\n%!" step
      (Array.map mlr ~f:Float.to_string |> String.concat_array ~sep:", ");
    let batch_losses = NodeUI.retrieve_1d_points ~xdim:0 total_loss.node.node.value in
    (* let batch_losses = NodeUI.retrieve_1d_points ~xdim:0 margin_loss.node.node.value in *)
    Stdio.printf "\nLosses over batch for step %d: %s\n%!" step
      (Array.map batch_losses ~f:Float.to_string |> String.concat_array ~sep:", ");
    Stdio.printf "\nTree for step %d\n%!" step;
    print_node_tree ~with_grad:true ~depth:8 total_loss.id;
    (* Stdio.printf "\nFormula for step %d\n%!" step;
    print_formula ~with_grad:true ~with_code:true `Default margin_loss; *)
    losses := batch_losses :: !losses;
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
  let plot_box = 
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"ixes" ~y_label:"ygreks"
      [Scatterplot {points=Array.concat !points1; pixel="#"}; 
       Scatterplot {points=Array.concat !points2; pixel="%"};
       Boundary_map {pixel_false="."; pixel_true="*"; callback}] in
  Stdio.printf "Half-moons scatterplot:\n%!";
  PrintBox_text.output Stdio.stdout plot_box;
  Stdio.printf "\n%!"


let () =
  let open Session.SDSL in
  drop_all_sessions();
  Random.init 0;
  let%nn_op c = "a" [-4] + "b" [2] in
  let%nn_op d = a *. b + b **. 3 in
  let%nn_op c = c + c + 1 in
  let%nn_op c = c + 1 + c + ~-a in
  let%nn_op d = d + d *. 2 + !/ (b + a) in
  let%nn_op d = d + 3 *. d + !/ (b - a) in
  let%nn_op e = c - d in
  let%nn_op f = e *. e in
  let%nn_op g = f /. 2 in
  let%nn_op g = g + 10. /. f in

  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ g;
  print_node_tree ~with_grad:true ~depth:99 g.id
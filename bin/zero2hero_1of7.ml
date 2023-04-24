open Base
open Ocannl
module CDSL = Code.CDSL
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL

let () = Session.SDSL.set_executor OCaml

let () =
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  let%nn_op n = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  refresh_session ();
  Stdio.printf "\n%!";
  print_preamble ();
  Stdio.printf "\n%!";
  print_node_tree ~with_grad:true ~depth:9 n.id;
  Stdio.printf "\n%!";
  print_formula ~with_grad:false ~with_code:true `Default n;
  Stdio.printf "\n%!"

let () =
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  let%nn_op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%nn_op f5 = f 5 in
  refresh_session ();
  print_node_tree ~with_grad:false ~depth:9 f5.id;
  Stdio.print_endline "\n";
  (* close_session / drop_session is not necessary. *)
  (* close_session (); *)
  (* drop_session (); *)
  let size = 100 in
  let xs = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  let x_flat =
    FDSL.term ~needs_gradient:true ~label:"x_flat" ~batch_dims:[ size ] ~input_dims:[] ~output_dims:[ 1 ]
      (First (Constant_fill xs))
  in
  let session_step =
    FDSL.data ~label:"session_step" ~batch_dims:[] ~output_dims:[ 1 ] (fun ~n -> Synthetic [%nn_cd n =+ 1])
  in
  let%nn_op x = x_flat @.| session_step in
  let%nn_op fx = f x in
  (* print_formula ~with_grad:true ~with_code:true ~with_low_level:true `Default fx; *)
  let ys =
    Array.map xs ~f:(fun _ ->
        refresh_session ();
        (NodeUI.retrieve_1d_points ~xdim:0 fx.node.node.value).(0))
  in
  (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
  let dys =
    Array.map xs ~f:(fun _ ->
        refresh_session ();
        let dy = NodeUI.retrieve_1d_points ~xdim:0 (Option.value_exn x.node.node.form).grad in
        if Array.is_empty dy then 70.0 else dy.(0))
  in
  print_preamble ();
  let plot_box =
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"x" ~y_label:"f(x)"
      [
        Scatterplot { points = Array.zip_exn xs ys; pixel = "#" };
        Scatterplot { points = Array.zip_exn xs dys; pixel = "*" };
        Line_plot { points = Array.create ~len:20 0.; pixel = "-" };
      ]
  in
  PrintBox_text.output Stdio.stdout plot_box

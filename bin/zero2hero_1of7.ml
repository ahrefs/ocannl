open Base
open Ocannl
module CDSL = Code.CDSL
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL

let () = Session.SDSL.set_executor Gccjit

let _suspended () =
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  let%nn_op n = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  refresh_session ();
  Stdio.printf "\n%!";
  print_node_tree ~with_id:true ~with_grad:true ~depth:9 n.id;
  Stdio.printf "\n%!"

let () =
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  let%nn_op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let size = 100 in
  let xs = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  let x_flat =
    FDSL.term ~needs_gradient:true ~label:"x_flat" ~batch_dims:[ size ] ~input_dims:[] ~output_dims:[ 1 ]
      ~init_op:(Constant_fill xs) ()
  in
  let%nn_dt session_step ~output_dims:[ 1 ] = n =+ 1 in
  let%nn_op x = x_flat @.| session_step in
  let%nn_op fx = f x in
  Stdio.print_endline "\n";
  print_node_tree ~with_id:true ~with_value:false ~with_grad:false ~depth:9 fx.id;
  Stdio.print_endline "\n";
  (* print_formula ~with_grad:true ~with_code:true ~with_low_level:true `Default fx; *)
  let ys =
    Array.map xs ~f:(fun _ ->
        refresh_session ();
        fx.@[0])
  in
  (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
  let dys =
    Array.map xs ~f:(fun _ ->
        refresh_session ();
        x.@%[0])
  in
  Stdio.print_endline "\n";
  print_preamble ();
  Stdio.print_endline "\n";
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

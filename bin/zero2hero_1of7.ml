open Base
open Ocannl
module CDSL = Code.CDSL
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL

let () = Session.SDSL.set_executor OCaml

let () =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_all_sessions();
  Random.init 0;
  let%nn_op f x = 3*.x**.2 - 4*.x + 5 in
  let%nn_op f5 = f 5 in
  refresh_session ();
  print_node_tree ~with_grad:false ~depth:9 f5.id;
  (* close_session is not necessary. *)
  close_session ();
  let xs = Array.init 100 ~f:Float.(fun i -> of_int i / 10. - 5.) in
  let x = FDSL.data ~needs_gradient:true ~label:"x" ~batch_dims:[] ~output_dims:[1] (Init_op (Fixed_constant xs)) in
  let fx = f x in
  let ys = Array.map xs ~f:(fun _ ->
    refresh_session ();
    (NodeUI.retrieve_1d_points ~xdim:0 fx.node.node.value).(0)) in
  (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
  let dys = Array.map xs ~f:(fun _ ->
    refresh_session ();
    (NodeUI.retrieve_1d_points ~xdim:0 (Option.value_exn x.node.node.form).grad).(0)) in
  let plot_box = 
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"x" ~y_label:"f(x)"
      [Scatterplot {points=Array.zip_exn xs ys; pixel="#"};
      Scatterplot {points=Array.zip_exn xs dys; pixel="*"};
      Line_plot {points=Array.create ~len:20 0.; pixel="-"}] in
  PrintBox_text.output Stdio.stdout plot_box

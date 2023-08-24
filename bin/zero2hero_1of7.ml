open Base
open Ocannl
module CDSL = Arrayjit.Low_level.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL




let _suspended () =
  (* SDSL.drop_all_sessions (); *)
  Random.init 0;
  let%nn_op v = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  SDSL.refresh_session ();
  Stdio.printf "\n%!";
  SDSL.print_tree ~with_id:true ~with_grad:true ~depth:9 v;
  Stdio.printf "\nHigh-level code:\n%!";
  SDSL.print_session_code ();
  Stdio.printf "\nCompiled code:\n%!";
  SDSL.print_session_code ~compiled:true ();
  Stdio.printf "\n%!"

let _suspended () =
  (* SDSL.drop_all_sessions (); *)
  Random.init 0;
  CDSL.enable_all_debugs ();
  CDSL.virtualize_settings.enable_device_only <- false;
  let%nn_op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%nn_op f5 = f 5 in
  SDSL.refresh_session ();
  Stdio.printf "\n%!";
  SDSL.print_tree ~with_grad:false ~depth:9 f5;
  Stdio.printf "\n%!"

let _suspended () =
  let open Tensor.O in
  (* SDSL.drop_all_sessions (); *)
  Random.init 0;
  let%nn_op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let size = 100 in
  let xs = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  let x_flat =
    TDSL.term ~grad_spec:Require_grad ~label:"x_flat"
      ~batch_dims:[ size ]
      ~input_dims:[]
      ~output_dims:[ 1 ]
      ~init_op:(Constant_fill xs) ()
  in
  let session_step = NTDSL.O.(NTDSL.counter !..1) in
  let%nn_op x = x_flat @.| session_step in
  let%nn_op fx = f x in
  Stdio.print_endline "\n";
  SDSL.print_tree ~with_id:true ~with_value:false ~with_grad:false ~depth:9 fx;
  Stdio.print_endline "\n";
  (* print_tensor ~with_grad:true ~with_code:true ~with_low_level:true `Default fx; *)
  let ys =
    Array.map xs ~f:(fun _ ->
        SDSL.refresh_session ();
        fx.@[0])
  in
  (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
  let dys =
    Array.map xs ~f:(fun _ ->
        SDSL.refresh_session ();
        x.@%[0])
  in
  Stdio.print_endline "\n";
  SDSL.print_preamble ();
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

let _suspended () =
  (* %expect_test "Graph drawing recompile" *)
  (* let open Operation.TDSL in *)
  let open Tensor.O in
  (* SDSL.drop_all_sessions (); *)
  (* CDSL.with_debug := true;
     CDSL.keep_files_in_run_directory := true; *)
  Random.init 0;
  Stdio.print_endline "\nFirst refresh:";
  let%nn_op f = (3 *. ("x" [ 5 ] **. 2)) - (4 *. x) + 5 in
  SDSL.refresh_session ();
  SDSL.print_tree ~with_grad:true ~depth:9 f;
  let xs = Array.init 10 ~f:Float.(fun i -> of_int i - 5.) in
  let ys =
    Array.map xs ~f:(fun v ->
        (* This is inefficient because it compiles the argument update inside the loop. *)
        SDSL.compile_routine [%nn_cd x =: !.v] ~name:"assign_x" ();
        SDSL.refresh_session ();
        f.@[0])
  in
  let plot_box =
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"x" ~y_label:"f(x)"
      [ Scatterplot { points = Array.zip_exn xs ys; pixel = "#" } ]
  in
  PrintBox_text.output Stdio.stdout plot_box

let () =
  (* SDSL.drop_all_sessions (); *)
  CDSL.with_debug := true;
  CDSL.keep_files_in_run_directory := true;
  Random.init 0;
  let%nn_op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%nn_op d = e + "c" [ 10 ] in
  let%nn_op l = d *. "f" [ -2 ] in
  SDSL.minus_learning_rate := Some (TDSL.init_const ~l:"minus_lr" ~o:[ 1 ] [| 0.1 |]);
  SDSL.everything_fully_on_host ();
  SDSL.refresh_session ~update_params:false ();
  Stdio.print_endline
    "\n\
     We did not update the params: all values and gradients will be at initial points,\n\
    \    which are specified in the tensor in the brackets.";
  SDSL.print_tree ~with_grad:true ~depth:9 l;
  SDSL.refresh_session ~update_params:true ();
  Stdio.print_endline
    "\n\
     Now we updated the params, but after the forward and backward passes:\n\
    \    only params values will change, compared to the above.";
  SDSL.print_tree ~with_grad:true ~depth:9 l;
  SDSL.refresh_session ~update_params:false ();
  Stdio.print_endline
    "\n\
     Now again we did not update the params, they will remain as above, but both param\n\
    \    gradients and the values and gradients of other nodes will change thanks to the forward and \
     backward passes.";
  SDSL.print_tree ~with_grad:true ~depth:9 l

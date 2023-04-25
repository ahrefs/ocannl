open Base
open Ocannl
module CDSL = Code.CDSL
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL

let () = Session.SDSL.set_executor Gccjit

let%expect_test "Graph drawing recompile" =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  let%nn_op f = (3 *. ("x" [ 5 ] **. 2)) - (4 *. x) + 5 in
  refresh_session ();
  print_node_tree ~with_grad:true ~depth:9 f.id;
  [%expect
    {|
                               [13] f <+>
                                6.00e+1
                               Gradient
                                1.00e+0
                          [12] <+>                        │[2] <5>
                           5.50e+1                        │ 5.00e+0
                          Gradient                        │<void>
                           1.00e+0                        │
           [9] <*.>        │          [11] <*.>           │
            7.50e+1        │           -2.00e+1           │
           Gradient        │          Gradient            │
            1.00e+0        │           1.00e+0            │
    [8] <3>  │  [6] <**.>  │[10] <-1> │    [4] <*.>       │
     3.00e+0 │   2.50e+1   │ -1.00e+0 │     2.00e+1       │
    <void>   │  Gradient   │<void>    │    Gradient       │
             │   3.00e+0   │          │     -1.00e+0      │
             │[1]│[5] <2>  │          │[3] <4>  │[1] <x>  │
             │   │ 2.00e+0 │          │ 4.00e+0 │ 5.00e+0 │
             │   │<void>   │          │<void>   │Gradient │
             │   │         │          │         │ 2.60e+1 │ |}];
  let xs = Array.init 10 ~f:Float.(fun i -> of_int i - 5.) in
  let ys =
    Array.map xs ~f:(fun v ->
        (* This is inefficient because it compiles the argument update inside the loop. *)
        let setval = compile_routine [%nn_cd x =: !.v] in
        setval ();
        refresh_session ();
        (NodeUI.retrieve_1d_points ~xdim:0 f.node.node.value).(0))
  in
  let plot_box =
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"x" ~y_label:"f(x)"
      [ Scatterplot { points = Array.zip_exn xs ys; pixel = "#" } ]
  in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
     1.000e+2│#
             │
             │
             │
             │
             │
             │
             │
             │
             │
             │
             │        #
             │
             │
             │
    f        │
    (        │
    x        │
    )        │
             │
             │                #
             │
             │
             │                                                                          #
             │
             │
             │
             │                        #
             │
             │                                                                 #
             │
             │
             │                                #
             │                                                         #
     4.000e+0│                                         #       #
    ─────────┼───────────────────────────────────────────────────────────────────────────
             │-5.000e+0                                                          4.000e+0
             │                                     x |}]

let%expect_test "Graph drawing fetch" =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  let%nn_op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%nn_op f5 = f 5 in
  refresh_session ();
  print_node_tree ~with_grad:false ~depth:9 f5.id;
  [%expect
    {|
                               [12] f <+>
                                6.00e+1
                          [11] <+>                        │[2] <5>
                           5.50e+1                        │ 5.00e+0
           [8] <*.>        │          [10] <*.>           │
            7.50e+1        │           -2.00e+1           │
    [7] <3>  │  [6] <**.>  │[9] <-1>  │     [4] <*.>      │
     3.00e+0 │   2.50e+1   │ -1.00e+0 │      2.00e+1      │
             │[1]│[5] <2>  │          │[3] <4>  │[1] <5>  │
             │   │ 2.00e+0 │          │ 4.00e+0 │ 5.00e+0 │ |}];
  (* close_session is not necessary. *)
  close_session ();
  let size = 100 in
  let xs = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  let x_flat =
    FDSL.term ~needs_gradient:true ~label:"x_flat" ~batch_dims:[ size ] ~input_dims:[] ~output_dims:[ 1 ]
      ~init_op:(Constant_fill xs) ()
  in
  let session_step =
    FDSL.data ~label:"session_step" ~output_dims:[ 1 ] (fun ~n -> Synthetic [%nn_cd n =+ 1])
  in
  let%nn_op x = x_flat @.| session_step in
  let%nn_op fx = f x in
  let ys =
    Array.map xs ~f:(fun _ ->
        refresh_session ();
        (NodeUI.retrieve_1d_points ~xdim:0 fx.node.node.value).(0))
  in
  (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
  let dys =
    Array.map xs ~f:(fun _ ->
        refresh_session ();
        (NodeUI.retrieve_1d_points ~xdim:0 (Option.value_exn x.node.node.form).grad).(0))
  in
  let plot_box =
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"x" ~y_label:"f(x)"
      [
        Scatterplot { points = Array.zip_exn xs ys; pixel = "#" };
        Scatterplot { points = Array.zip_exn xs dys; pixel = "*" };
        Line_plot { points = Array.create ~len:20 0.; pixel = "-" };
      ]
  in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
     9.703e+1 │                                                                          #
              │#
              │ ##
              │  #
              │   #
              │    #
              │     #
              │      #
              │       #
              │        #
              │         #
              │          ##
              │           #                                                            ##
              │            ##                                                         #
              │              #                                                       #
    f         │               ##                                                   ##
    (         │                 #                                                 ##
    x         │                 ##                                              ##
    )         │                   ##                                           ##
              │#                    ##                                       ##       ***
              │                       ##                                   ##    ******
              │                         ##                               ##  ****
              │                          ####                         ##*****
              │                             ####                   *****
              │                                ######         *****
              │                                      ####******
              │-  -   -   -   -  -   -   -   -  -   -***-   -  -   -   -   -  -   -   -
              │                                ******
              │                            *****
              │                       *****
              │                  ******
              │              ****
              │        ******
              │    *****
     -3.400e+1│****                                                                      *
    ──────────┼───────────────────────────────────────────────────────────────────────────
              │-5.000e+0                                                          4.900e+0
              │                                     x |}]

let%expect_test "Simple gradients" =
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  let%nn_op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%nn_op d = e + "c" [ 10 ] in
  let%nn_op l = d *. "f" [ -2 ] in
  minus_learning_rate :=
    Some
      (FDSL.term ~label:"minus_lr" ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ]
         ~init_op:(Constant_fill [| 0.1 |]) ());
  refresh_session ();
  print_node_tree ~with_grad:true ~depth:9 l.id;
  [%expect
    {|
                    [7] l <*.>
                     -8.00e+0
                    Gradient
                     1.00e+0
              [5] d <+>            │[6] <f>
               4.00e+0             │ -2.00e+0
              Gradient             │Gradient
               -2.00e+0            │ 4.00e+0
         [3] e <*.>     │[4] <c>   │
          -6.00e+0      │ 1.00e+1  │
         Gradient       │Gradient  │
          -2.00e+0      │ -2.00e+0 │
    [1] <a>  │[2] <b>   │          │
     2.00e+0 │ -3.00e+0 │          │
    Gradient │Gradient  │          │
     6.00e+0 │ -4.00e+0 │          │ |}];
  Option.value_exn !update_params ();
  refresh_session ();
  print_node_tree ~with_grad:true ~depth:9 l.id;
  [%expect
    {|
                    [7] l <*.>
                     3.78e+0
                    Gradient
                     1.00e+0
              [5] d <+>            │[6] <f>
               -2.36e+0            │ -1.60e+0
              Gradient             │Gradient
               -1.60e+0            │ -2.36e+0
         [3] e <*.>     │[4] <c>   │
          -1.22e+1      │ 9.80e+0  │
         Gradient       │Gradient  │
          -1.60e+0      │ -1.60e+0 │
    [1] <a>  │[2] <b>   │          │
     3.20e+0 │ -3.80e+0 │          │
    Gradient │Gradient  │          │
     6.08e+0 │ -5.12e+0 │          │ |}]

let%expect_test "tanh plot" =
  (* TODO: NOT IMPLEMENTED *)
  ()

let%expect_test "2D neuron" =
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  let%nn_op n = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  refresh_session ();
  print_node_tree ~with_grad:true ~depth:9 n.id;
  [%expect
    {|
                        [5] n <+>
                         7.00e-1
                        Gradient
                         1.00e+0
                  [4] <*>                  │[1] <b>
                   -6.00e+0                │ 6.70e+0
                  Gradient                 │Gradient
                   1.00e+0                 │ 1.00e+0
    [2] <w>            │[3] <x>            │
     -3.00e+0  1.00e+0 │ 2.00e+0  0.00e+0  │
    Gradient           │Gradient           │
     2.00e+0  0.00e+0  │ -3.00e+0  1.00e+0 │ |}]

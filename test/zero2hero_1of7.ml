open Base
open Ocannl
module CDSL = Code.CDSL
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL
module SDSL = Session.SDSL

let () = SDSL.set_executor Gccjit

let%expect_test "Graph drawing recompile" =
  (* let open Operation.FDSL in *)
  let open SDSL.O in
  SDSL.drop_all_sessions ();
  Random.init 0;
  let%nn_op f = (3 *. ("x" [ 5 ] **. 2)) - (4 *. x) + 5 in
  SDSL.refresh_session ();
  SDSL.print_node_tree ~with_grad:true ~depth:9 f.id;
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
        SDSL.compile_routine [%nn_cd x =: !.v] ();
        SDSL.refresh_session ();
        f.@[0])
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
  let open SDSL.O in
  SDSL.drop_all_sessions ();
  Random.init 0;
  let%nn_op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%nn_op f5 = f 5 in
  SDSL.refresh_session ();
  SDSL.print_node_tree ~with_grad:false ~depth:9 f5.id;
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
  SDSL.close_session ();
  let size = 100 in
  let xs = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  let x_flat =
    FDSL.term ~needs_gradient:true ~label:"x_flat" ~batch_dims:[ size ] ~input_dims:[] ~output_dims:[ 1 ]
      ~init_op:(Constant_fill xs) ()
  in
  let%nn_dt session_step ~output_dims:[ 1 ] = n =+ 1 in
  let%nn_op x = x_flat @.| session_step in
  let%nn_op fx = f x in
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
  SDSL.drop_all_sessions ();
  Random.init 0;
  let%nn_op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%nn_op d = e + "c" [ 10 ] in
  let%nn_op l = d *. "f" [ -2 ] in
  SDSL.minus_learning_rate := Some (FDSL.init_const ~l:"minus_lr" ~o:[ 1 ] [| 0.1 |]);
  SDSL.refresh_session ~update_params:false ();
  (* We did not update the params: all values and gradients will be at initial points, which are
     specified in the formula in the brackets. *)
  SDSL.print_node_tree ~with_grad:true ~depth:9 l.id;
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
  SDSL.refresh_session ~update_params:true ();
  (* Now we updated the params, but after the forward and backward passes: only params values
     will change, compared to the above. *)
  SDSL.print_node_tree ~with_grad:true ~depth:9 l.id;
  [%expect
    {|
                    [7] l <*.>
                     -8.00e+0
                    Gradient
                     1.00e+0
              [5] d <+>            │[6] <f>
               4.00e+0             │ -1.60e+0
              Gradient             │Gradient
               -2.00e+0            │ 4.00e+0
         [3] e <*.>     │[4] <c>   │
          -6.00e+0      │ 9.80e+0  │
         Gradient       │Gradient  │
          -2.00e+0      │ -2.00e+0 │
    [1] <a>  │[2] <b>   │          │
     2.60e+0 │ -3.40e+0 │          │
    Gradient │Gradient  │          │
     6.00e+0 │ -4.00e+0 │          │ |}];
  SDSL.refresh_session ~update_params:false ();
  (* Now again we did not update the params, they will remain as above, but both param gradients
     and the values and gradients of other nodes will change thanks to the forward and backward passes. *)
  SDSL.print_node_tree ~with_grad:true ~depth:9 l.id;
  [%expect
    {|
                      [7] l <*.>
                       -1.54e+0
                      Gradient
                       1.00e+0
                [5] d <+>            │[6] <f>
                 9.60e-1             │ -1.60e+0
                Gradient             │Gradient
                 -1.60e+0            │ 9.60e-1
           [3] e <*.>     │[4] <c>   │
            -8.84e+0      │ 9.80e+0  │
           Gradient       │Gradient  │
            -1.60e+0      │ -1.60e+0 │
      [1] <a>  │[2] <b>   │          │
       2.60e+0 │ -3.40e+0 │          │
      Gradient │Gradient  │          │
       5.44e+0 │ -4.16e+0 │          │ |}]

let%expect_test "tanh plot" =
  (* TODO: NOT IMPLEMENTED *)
  ()

let%expect_test "2D neuron" =
  SDSL.drop_all_sessions ();
  Random.init 0;
  let%nn_op n = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  (* No need for [~update_params:false] because we have not set [minus_learning_rate]. *)
  SDSL.refresh_session ();
  SDSL.print_node_tree ~with_grad:true ~depth:9 n.id;
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

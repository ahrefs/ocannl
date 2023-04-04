open Base
open Ocannl
module CDSL = Code.CDSL
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL

let () = Session.SDSL.set_executor OCaml

let%expect_test "Graph drawing recompile" =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_all_sessions();
  Random.init 0;
  let%nn_op f = 3*."x"[5]**.2 - 4*.x + 5 in
  refresh_session ();
  print_node_tree ~with_grad:true ~depth:9 f.id;
  [%expect {|
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
  let ys = Array.map xs ~f:(fun v ->
    (* This is very inefficient because it compiles the argument update inside the loop. *)
    let setval = compile_routine [%nn_cd x =: ~= !.v ~logic:"."] in
    setval (); refresh_session ();
    (NodeUI.retrieve_1d_points ~xdim:0 f.node.node.value).(0)) in
  let plot_box = 
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"x" ~y_label:"f(x)"
      [Scatterplot {points=Array.zip_exn xs ys; pixel="#"}] in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect {|
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
  drop_all_sessions();
  Random.init 0;
  let%nn_op f x = 3*.x**.2 - 4*.x + 5 in
  let%nn_op f5 = f 5 in
  refresh_session ();
  print_node_tree ~with_grad:false ~depth:9 f5.id;
  [%expect {|
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
  let xs = Array.init 100 ~f:Float.(fun i -> of_int i / 10. - 5.) in
  let x = FDSL.data ~needs_gradient:true ~label:"x" ~batch_dims:[] ~output_dims:[1]
      (Init_op (Fixed_constant xs)) in
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
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect {|
     1.000e+2 │#
              │#
              │ #
              │  #
              │  #
              │   ##
              │     #
              │     #
              │      ##
              │        #
              │        ##
              │          #                                                               #
              │           #                                                             #
              │            #                                                          ##
              │             ##                                                       ##
    f         │              ##                                                    ##
    (         │                ##                                                 ##
    x         │                 ##                                              ##
    )         │                   ##                                           ##
              │                    ##                                        ##          *
              │                      ##                                    ###     ******
              │                        ###                               ##    *****
              │                          ###                           ## *****
              │                             ###                     #*****
              │                                #####           #*****
              │                                     #######*****
              │-  -   -   -   -  -   -   -   -  -   - **-***-  -   -   -   -  -   -   -
              │                                  *****
              │                             *****
              │                        ******
              │                    ****
              │              ******
              │          *****
              │     *****
     -3.400e+1│*****
    ──────────┼───────────────────────────────────────────────────────────────────────────
              │-5.000e+0                                                          4.900e+0
              │                                     x |}]

let%expect_test "Simple gradients" =
  let open Session.SDSL in
  drop_all_sessions();
  Random.init 0;
  let%nn_op e = "a" [2] *. "b" [-3] in
  let%nn_op d = e + "c" [10] in
  let%nn_op l = d *. "f" [-2] in
  minus_learning_rate := Some (
      FDSL.data ~label:"minus_lr" ~batch_dims:[] ~output_dims:[1]
        (Init_op (Constant_of_value 0.1)));
  refresh_session ();
  print_node_tree ~with_grad:true ~depth:9 l.id;
  [%expect {|
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
  [%expect {|
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
  let open Session.SDSL in
  drop_all_sessions();
  Random.init 0;
  let%nn_op n = "w" [-3, 1] * "x" [2; 0] + "b" [6.7] in
  refresh_session ();
  print_node_tree ~with_grad:true ~depth:9 n.id;
  [%expect {| |}]

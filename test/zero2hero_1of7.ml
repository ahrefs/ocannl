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
                               [13] +
                                6.00e+1
                               Gradient
                                1.00e+0
                          [12] +                          │[2] 5
                           5.50e+1                        │ 5.00e+0
                          Gradient                        │<void>
                           1.00e+0                        │
           [9] *.          │          [11] *.             │
            7.50e+1        │           -2.00e+1           │
           Gradient        │          Gradient            │
            1.00e+0        │           1.00e+0            │
    [8] 3    │  [6] **.    │[10] -1   │    [4] *.         │
     3.00e+0 │   2.50e+1   │ -1.00e+0 │     2.00e+1       │
    <void>   │  Gradient   │<void>    │    Gradient       │
             │   3.00e+0   │          │     -1.00e+0      │
             │[1]│[5] 2    │          │[3] 4    │[1] x    │
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
                               [12] +
                                6.00e+1
                          [11] +                          │[2] 5
                           5.50e+1                        │ 5.00e+0
           [8] *.          │          [10] *.             │
            7.50e+1        │           -2.00e+1           │
    [7] 3    │  [6] **.    │[9] -1    │     [4] *.        │
     3.00e+0 │   2.50e+1   │ -1.00e+0 │      2.00e+1      │
             │[1]│[5] 2    │          │[3] 4    │[1] 5    │
             │   │ 2.00e+0 │          │ 4.00e+0 │ 5.00e+0 │ |}];
  (* close_session is not necessary. *)
  close_session ();
  let xs = Array.init 100 ~f:Float.(fun i -> of_int i / 10. - 5.) in
  let x = FDSL.data ~label:"x" ~batch_dims:[] ~output_dims:[1] (Init_op (Fixed_constant xs)) in
  let fx = f x in
  let ys = Array.map xs ~f:(fun _ ->
    refresh_session ();
    (NodeUI.retrieve_1d_points ~xdim:0 fx.node.node.value).(0)) in
  let plot_box = 
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"x" ~y_label:"f(x)"
      [Scatterplot {points=Array.zip_exn xs ys; pixel="#"}] in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect {|
     1.000e+2│#
             │
             │#
             │ #
             │  #
             │  #
             │   #
             │    #
             │     #
             │     #
             │      #
             │       #
             │        #
             │        #
             │         #
    f        │          #
    (        │           #                                                             ##
    x        │           #                                                            #
    )        │            #                                                          #
             │             ##                                                        #
             │              #                                                       #
             │               #                                                    ##
             │                ##                                                  #
             │                 #                                                ##
             │                  #                                              #
             │                   ##                                           ##
             │                    ##                                         #
             │                      #                                       #
             │                       #                                    ##
             │                        ##                                 #
             │                          ##                             ##
             │                            ##                         ##
             │                             ###                     ##
             │                                ###               ###
     3.670e+0│                                   ###############
    ─────────┼───────────────────────────────────────────────────────────────────────────
             │-5.000e+0                                                          4.900e+0
             │                                     x |}]

open Base
open Ocannl
module CDSL = Arrayjit.Low_level.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL




let%expect_test "Graph drawing recompile" =
  (* let open Operation.TDSL in *)
  let open Tensor.O in
  (* SDSL.drop_all_sessions (); *)
  Random.init 0;
  let%nn_op f = (3 *. ("x" [ 5 ] **. 2)) - (4 *. x) + 5 in
  Tensor.set_fully_on_host x;
  (* refresh_session (); *)
  Tensor.print_tree ~with_grad:true ~depth:9 f;
  [%expect
    {|
                          [13] f <+>
                           6.00e+1
                          Gradient
                          <void>
                       [12] <+>                    │[2] <5>
                       <void>                      │<void>
                       Gradient                    │
                       <void>                      │
         [9] <*.>      │         [11] <*.>         │
         <void>        │         <void>            │
         Gradient      │         Gradient          │
         <void>        │         <void>            │
    [8] <3>│ [6] <**.> │[10] <-1>│    [4] <*.>     │
    <void> │ <void>    │<void>   │    <void>       │
           │ Gradient  │         │    Gradient     │
           │ <void>    │         │    <void>       │
           │[1]│[5] <2>│         │[3] <4>│[1] <x>  │
           │   │<void> │         │<void> │ 5.00e+0 │
           │   │       │         │       │Gradient │
           │   │       │         │       │ 2.60e+1 │ |}];
  let xs = Array.init 10 ~f:Float.(fun i -> of_int i - 5.) in
  let ys =
    Array.map xs ~f:(fun v ->
        (* This is inefficient because it compiles the argument update inside the loop. *)
        SDSL.compile_routine [%nn_cd x =: !.v] ~name:"assign_x" ();
        (* refresh_session (); *)
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
  let open Tensor.O in
  (* SDSL.drop_all_sessions (); *)
  Random.init 0;
  CDSL.virtualize_settings.enable_device_only <- false;
  CDSL.virtualize_settings.inline_constants <- false;
  let%nn_op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%nn_op f5 = f 5 in
  (* everything_fully_on_host (); *)
  (* refresh_session (); *)
  Tensor.print_tree ~with_grad:false ~depth:9 f5;
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
    TDSL.term ~grad_spec:Require_grad ~label:"x_flat"
      ~batch_dims:[ size ]
      ~input_dims:[]
      ~output_dims:[ 1 ]
      ~init_op:(Constant_fill xs) ()
  in
  let session_step = NTDSL.O.(NTDSL.counter !..1) in
  let%nn_op x = x_flat @.| session_step in
  Tensor.set_fully_on_host x;
  let%nn_op fx = f x in
  let ys =
    Array.map xs ~f:(fun _ ->
        (* refresh_session (); *)
        fx.@[0])
  in
  (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
  let dys =
    Array.map xs ~f:(fun _ ->
        (* refresh_session (); *)
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
     1.000e+2 │                                                                          #
              │#
              │#
              │ #
              │  #
              │  ##
              │    #
              │     #
              │     ##
              │       #
              │        #
              │         #                                                               #
              │          ##                                                            #
              │           #                                                           #
              │            ##                                                       ##
    f         │              #                                                     #
    (         │               ##                                                 ##
    x         │                 #                                               #
    )         │                  ##                                           ##
              │                    #                                         #          *
              │                     ###                                   ###      *****
              │                       ###                               ###   *****
              │                          ##                           ## *****
              │                            ###                     #*****
              │                               #####           #******
              │                                    ########****
              │-  -   -   -   -  -   -   -   -  -   -***-** -  -   -   -   -  -   -   -
              │                                 ******
              │                             ****
              │                       ******
              │                   *****
              │              *****
              │         *****
              │    *****
     -3.400e+1│****                                                                      *
    ──────────┼───────────────────────────────────────────────────────────────────────────
              │-5.000e+0                                                          4.900e+0
              │                                     x |}]

let%expect_test "Simple gradients materialized" =
  (* SDSL.drop_all_sessions (); *)
  Random.init 0;
  let%nn_op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%nn_op d = e + "c" [ 10 ] in
  let%nn_op l = d *. "f" [ -2 ] in
  SDSL.minus_learning_rate := Some (TDSL.init_const ~l:"minus_lr" ~o:[ 1 ] [| 0.1 |]);
  (* everything_fully_on_host (); *)
  SDSL.refresh_session ~update_params:false ();
  (* We did not update the params: all values and gradients will be at initial points, which are
     specified in the tensor in the brackets. *)
  Tensor.print_tree ~with_grad:true ~depth:9 l;
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
     will change, compared to the above, as long as gradient tensors are materialized.
     Since virtual tensors are computed by-need, they will always be recomputed using the latest
     parameter state. When parameter gradients are virtual, this will lead to different parameter updates. *)
  Tensor.print_tree ~with_grad:true ~depth:9 l;
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
  Tensor.print_tree ~with_grad:true ~depth:9 l;
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

let%expect_test "Simple gradients virtual" =
  (* SDSL.drop_all_sessions (); *)
  Random.init 0;
  let%nn_op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%nn_op d = e + "c" [ 10 ] in
  let%nn_op l = d *. "f" [ -2 ] in
  SDSL.minus_learning_rate := Some (TDSL.init_const ~l:"minus_lr" ~o:[ 1 ] [| 0.1 |]);
  SDSL.refresh_session ~update_params:false ();
  (* We did not update the params: all values and gradients will be at initial points, which are
     specified in the tensor in the brackets. *)
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                   [7] l <*.>
                    -8.00e+0
                   Gradient
                   <void>
              [5] d <+>           │[6] <f>
              <void>              │ -2.00e+0
              Gradient            │Gradient
              <void>              │<void>
         [3] e <*.>     │[4] <c>  │
         <void>         │ 1.00e+1 │
         Gradient       │Gradient │
         <void>         │<void>   │
    [1] <a>  │[2] <b>   │         │
     2.00e+0 │ -3.00e+0 │         │
    Gradient │Gradient  │         │
    <void>   │<void>    │         │ |}];
  SDSL.refresh_session ~update_params:true ();
  (* Now we updated the params, but after the forward and backward passes: only params values
     will change, compared to the above, as long as gradient tensors are materialized.
     Since virtual tensors are computed by-need, they will always be recomputed using the latest
     parameter state. When parameter gradients are virtual, this will lead to different parameter updates. *)
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                   [7] l <*.>
                    -8.00e+0
                   Gradient
                   <void>
              [5] d <+>           │[6] <f>
              <void>              │ -1.62e+0
              Gradient            │Gradient
              <void>              │<void>
         [3] e <*.>     │[4] <c>  │
         <void>         │ 9.80e+0 │
         Gradient       │Gradient │
         <void>         │<void>   │
    [1] <a>  │[2] <b>   │         │
     2.54e+0 │ -3.32e+0 │         │
    Gradient │Gradient  │         │
    <void>   │<void>    │         │ |}];
  SDSL.refresh_session ~update_params:false ();
  (* Now again we did not update the params, they will remain as above, but both param gradients
     and the values and gradients of other nodes will change thanks to the forward and backward passes. *)
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                     [7] l <*.>
                      -2.21e+0
                     Gradient
                     <void>
                [5] d <+>           │[6] <f>
                <void>              │ -1.62e+0
                Gradient            │Gradient
                <void>              │<void>
           [3] e <*.>     │[4] <c>  │
           <void>         │ 9.80e+0 │
           Gradient       │Gradient │
           <void>         │<void>   │
      [1] <a>  │[2] <b>   │         │
       2.54e+0 │ -3.32e+0 │         │
      Gradient │Gradient  │         │
      <void>   │<void>    │         │ |}]

let%expect_test "tanh plot" =
  (* TODO: NOT IMPLEMENTED *)
  ()

let%expect_test "2D neuron materialized" =
  (* SDSL.drop_all_sessions (); *)
  Random.init 0;
  let%nn_op v = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  (* No need for [~update_params:false] because we have not set [minus_learning_rate]. *)
  (* everything_fully_on_host (); *)
  (* refresh_session (); *)
  Tensor.print_tree ~with_grad:true ~depth:9 v;
  [%expect
    {|
                        [5] v <+>
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

let%expect_test "2D neuron virtual" =
  (* SDSL.drop_all_sessions (); *)
  Random.init 0;
  let%nn_op v = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  (* No need for [~update_params:false] because we have not set [minus_learning_rate]. *)
  (* refresh_session (); *)
  Tensor.print_tree ~with_grad:true ~depth:9 v;
  [%expect
    {|
                       [5] v <+>
                        7.00e-1
                       Gradient
                       <void>
                   [4] <*>                │[1] <b>
                   <void>                 │ 6.70e+0
                   Gradient               │Gradient
                   <void>                 │<void>
    [2] <w>            │[3] <x>           │
     -3.00e+0  1.00e+0 │ 2.00e+0  0.00e+0 │
    Gradient           │Gradient          │
    <void>             │<void>            │ |}]

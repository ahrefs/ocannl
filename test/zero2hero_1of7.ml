open Base
open Ocannl
module LA = Arrayjit.Lazy_array
module IDX = Arrayjit.Indexing.IDX
module CDSL = Arrayjit.Low_level.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL

let%expect_test "Graph drawing recompile" =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let open Tensor.O in
  let%op f = (3 *. ("x" [ 5 ] **. 2)) - (4 *. x) + 5 in
  Train.set_on_host x.value;
  let f_jitted = Backend.jit ctx ~verbose:true IDX.empty @@ Train.forward f in
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
        let x_jitted = Backend.jit f_jitted.context IDX.empty ~name:"assign_x" [%cd x =: !.v] in
        x_jitted.run ();
        f_jitted.run ();
        Backend.await device;
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
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let open Tensor.O in
  CDSL.virtualize_settings.enable_device_only <- false;
  let%op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%op f5 = f 5 in
  Train.every_non_literal_on_host f5;
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
  let size = 100 in
  let xs = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  let x_flat =
    Tensor.term ~grad_spec:Require_grad ~label:[ "x_flat" ] ~batch_dims:[ size ] ~input_dims:[]
      ~output_dims:[ 1 ]
      ~init_op:(Constant_fill { values = xs; strict = true })
      ()
  in
  let step_sym, bindings = IDX.get_static_symbol ~static_range:size IDX.empty in
  let%op x = x_flat @| step_sym in
  Train.set_on_host x.value;
  let%op fx = f x in
  let fx_jitted = Backend.jit ctx ~verbose:true bindings @@ Train.grad_update fx in
  let ys =
    Array.map xs ~f:(fun _ ->
        fx_jitted.run ();
        Backend.await device;

        fx.@[0])
  in
  (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
  let dys = Array.map xs ~f:(fun _ -> (* refresh_session (); *)
                                      x.@%[0]) in
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

let%expect_test "Simple gradients hosted" =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let%op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%op d = e + "c" [ 10 ] in
  let%op l = d *. "f" [ -2 ] in
  let%op learning_rate = 0.1 in
  Train.every_non_literal_on_host l;
  Train.every_non_literal_on_host learning_rate;
  let grad = Train.grad_update l in
  let sgd = Train.sgd_update ~learning_rate l in
  let grad_jitted = Backend.jit ctx IDX.empty grad in
  let sgd_jitted = Backend.jit grad_jitted.context IDX.empty sgd in
  (* Check out the initial state without running a forward pass. *)
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect {| |}];
  (* Do not update the params: all values and gradients will be at initial points, which are
     specified in the tensor in the brackets. *)
  Train.sync_run (module Backend) grad_jitted l;
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
  (* Now we update the params, but we are not doing the forward and backward passes: only params values
     will change, compared to the above.
     Since virtual tensors are computed by-need, they will always be recomputed using the latest
     parameter state. *)
  sgd_jitted.run ();
  Backend.await device;
  Train.all_device_to_host (module Backend) sgd_jitted.context l;
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

  (* Now the params will remain as above, but both param gradients and the values and gradients
     of other nodes will change thanks to the forward and backward passes. *)
  grad_jitted.run ();
  Backend.await device;
  Train.all_device_to_host (module Backend) sgd_jitted.context l;
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
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let%op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%op d = e + "c" [ 10 ] in
  let%op l = d *. "f" [ -2 ] in
  let%op learning_rate = 0.1 in
  let grad = Train.grad_update l in
  let sgd = Train.sgd_update ~learning_rate l in
  let grad_jitted = Backend.jit ctx IDX.empty grad in
  let sgd_jitted = Backend.jit grad_jitted.context IDX.empty sgd in
  (* Check out the initial state without running a forward pass. *)
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect {| |}];
  (* Do not update the params: all values and gradients will be at initial points, which are
     specified in the tensor in the brackets. *)
  Train.sync_run (module Backend) grad_jitted l;
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
  (* Now we update the params, but are not doing the forward and backward passes: only params values
     will change, compared to the above.
     Since virtual tensors are computed by-need, they will always be recomputed using the latest
     parameter state. *)
  sgd_jitted.run ();
  Backend.await device;
  Train.all_device_to_host (module Backend) sgd_jitted.context l;
  Tensor.print_tree ~with_grad:true ~depth:9 l;
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
  (* Now the params will remain as above, but both param gradients and the values and gradients
     of other nodes will change thanks to the forward and backward passes. *)
  grad_jitted.run ();
  Backend.await device;
  Train.all_device_to_host (module Backend) sgd_jitted.context l;
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

let%expect_test "2D neuron hosted" =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let%op v = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  Train.every_non_literal_on_host v;
  let jitted = Backend.jit ctx IDX.empty @@ Train.grad_update v in
  jitted.run ();
  Backend.await device;
  Train.all_device_to_host (module Backend) jitted.context v;
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
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let%op v = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  let jitted = Backend.jit ctx IDX.empty @@ Train.grad_update v in
  jitted.run ();
  Backend.await device;
  Train.all_device_to_host (module Backend) jitted.context v;
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

open Base
open Ocannl
module Nd = Ir.Ndarray
module Asgns = Ir.Assignments
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL

module type Backend = Ir.Backend_intf.Backend

let%expect_test "Graph drawing recompile" =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let open Operation.At in
  let%op f_nd = (3 *. ("x" [ 5 ] **. 2)) - (4 *. x) + 5 in
  Train.set_hosted x.value;
  ignore (Train.forward_once backend f_nd);
  Train.printf_tree ~with_grad:true ~depth:9 f_nd;
  [%expect
    {|
                                 #15 +_f_nd
                                  6.00e+1
                                 #16 grad_+_f_nd Virt/30
                                 <void>
                             #13 - Virt/152                             │#2 5 Virt/40
                             <void>                                     │<void>
                             #14 grad_- Virt/30                         │
                             <void>                                     │
                  #11 *. Virt/152                   │#4 *. Virt/152     │
                  <void>                            │<void>             │
                  #12 grad_*. Virt/30               │#5 grad_*. Virt/30 │
                  <void>                            │<void>             │
    #10 3 Virt/40│       #7 **. Virt/152            │#3 4 Virt/40│[#0 x]│
    <void>       │       <void>                     │<void>      │      │
                 │       #8 grad_**. Virt/30        │            │      │
                 │       <void>                     │            │      │
                 │#0 x non-emb         │#6 2 Virt/40│            │      │
                 │ 5.00                │<void>      │            │      │
                 │#1 grad_x Local/26030│            │            │      │
                 │<void>               │            │            │      │
    |}];
  let%op f = (3 *. ("x" [ 5 ] **. 2)) - (4 *. x) + 5 in
  Train.every_non_literal_on_host f;
  let f_upd = Train.grad_update f in
  let ctx = Train.init_params (module Backend) ~ctx IDX.empty f in
  let f_bprop = Train.to_routine (module Backend) ctx IDX.empty f_upd in
  Train.run f_bprop;
  Train.printf_tree ~with_grad:true ~depth:9 f;
  [%expect
    {|
                                    #32 +_f
                                     6.00e+1
                                    #33 grad_+_f
                                     1.00
                              #30 -                                │#19 5 Virt/40
                               5.50e+1                             │<void>
                              #31 grad_-                           │
                               1.00                                │
                   #28 *.                    │     #21 *.          │
                    7.50e+1                  │      2.00e+1        │
                   #29 grad_*.               │     #22 grad_*.     │
                    1.00                     │      -1.00          │
    #27 3 Virt/40│       #24 **.             │#20 4 Virt/40│[#17 x]│
    <void>       │        2.50e+1            │<void>       │       │
                 │       #25 grad_**.        │             │       │
                 │        3.00               │             │       │
                 │#17 x non-emb│#23 2 Virt/40│             │       │
                 │ 5.00        │<void>       │             │       │
                 │#18 grad_x   │             │             │       │
                 │ 2.60e+1     │             │             │       │
    |}];
  let xs = Array.init 10 ~f:Float.(fun i -> of_int i - 5.) in
  let ys =
    Array.map xs ~f:(fun v ->
        (* This is inefficient because it compiles the argument update inside the loop. *)
        let assign_x =
          Train.to_routine
            (module Backend)
            f_bprop.context IDX.empty ~name:"assign_x" [%cd x =: !.v]
        in
        Train.run assign_x;
        Train.run f_bprop;
        f.@[0])
  in
  let plot_box =
    PrintBox_utils.plot ~x_label:"x" ~y_label:"f(x)"
      [ Scatterplot { points = Array.zip_exn xs ys; content = PrintBox.line "#" } ]
  in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    ┌────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ 1.00e+2│#                                                                                                   │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │           #                                                                                        │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │f       │                                                                                                    │
    │(       │                                                                                                    │
    │x       │                                                                                                    │
    │)       │                                                                                                    │
    │        │                                                                                                    │
    │        │                      #                                                                             │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                   #│
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                 #                                                                  │
    │        │                                                                                                    │
    │        │                                                                                        #           │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                            #                                                       │
    │        │                                                                             #                      │
    │        │                                                                                                    │
    │ 4.00   │                                                       #          #                                 │
    ├────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │        │-5.00                                                                                           4.00│
    │        │                                                 x                                                  │
    └────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}]

let%expect_test "Graph drawing fetch" =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let open Operation.At in
  CDSL.virtualize_settings.enable_device_only <- false;
  let%op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%op f5 = f 5 in
  Train.every_non_literal_on_host f5;
  ignore (Train.forward_once (module Backend) f5);
  Train.printf_tree ~with_grad:false ~depth:9 f5;
  [%expect
    {|
                                   #9 +_f_5
                                    6.00e+1
                            #8 -                              │#1 5 Virt/40
                             5.50e+1                          │<void>
               #7 *.                │        #3 *.            │
                7.50e+1             │         2.00e+1         │
    #6 3 Virt/40│     #5 **.        │#2 4 Virt/40│#0 5 Virt/40│
    <void>      │      2.50e+1      │<void>      │<void>      │
                │[#0 5]│#4 2 Virt/40│            │            │
                │      │<void>      │            │            │
    |}];
  let size = 100 in
  let xs = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  (* Yay, the whole shape gets inferred! *)
  let x_flat = Tensor.term_init xs ~label:[ "x_flat" ] ~grad_spec:Require_grad () in
  let step_sym, bindings = IDX.get_static_symbol ~static_range:size IDX.empty in
  let%op x = x_flat @| step_sym in
  let%op fx = f x in
  Train.set_hosted x.value;
  Train.set_hosted (Option.value_exn ~here:[%here] x.diff).grad;
  let update = Train.grad_update fx in
  let fx_routine = Train.to_routine (module Backend) ctx bindings update in
  let step_ref = IDX.find_exn fx_routine.bindings step_sym in
  let ys, dys =
    Array.unzip
    @@ Array.mapi xs ~f:(fun i _ ->
           step_ref := i;
           Train.run fx_routine;
           (fx.@[0], x.@%[0]))
  in
  (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
  let plot_box =
    PrintBox_utils.plot ~x_label:"x" ~y_label:"f(x)"
      [
        Scatterplot { points = Array.zip_exn xs ys; content = PrintBox.line "#" };
        Scatterplot { points = Array.zip_exn xs dys; content = PrintBox.line "*" };
        Line_plot { points = Array.create ~len:20 0.; content = PrintBox.line "-" };
      ]
  in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    ┌─────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ 1.00e+2 │#                                                                                                   │
    │         │#                                                                                                   │
    │         │  #                                                                                                 │
    │         │  #                                                                                                 │
    │         │    #                                                                                               │
    │         │     #                                                                                              │
    │         │     #                                                                                              │
    │         │       #                                                                                            │
    │         │       #                                                                                            │
    │         │         #                                                                                          │
    │         │          ##                                                                                        │
    │         │            #                                                                                       │
    │         │            #                                                                                       │
    │         │             # #                                                                                   #│
    │         │                #                                                                               # # │
    │         │                 #                                                                              #   │
    │         │                   #                                                                          ##    │
    │         │                    ##                                                                       #      │
    │f        │                      #                                                                    #        │
    │(        │                      # #                                                                ##         │
    │x        │                         ##                                                             #           │
    │)        │                           #                                                          #             │
    │         │                            # #                                                    ###            **│
    │         │                               ##                                                #         * ****   │
    │         │                                 # #                                           ##     * ****        │
    │         │                                    ###                                     ##   * ***              │
    │         │                                      # #                               # ##** **                   │
    │         │                                          ####                     # ###* *                         │
    │         │                                             # #### # #    ## # ####                                │
    │         │                                                       # # ** *                                     │
    │         │-    -    -    -    -    -    -    -    -    -    - * ** *  -    -    -    -    -    -    -    -    │
    │         │                                             * ****                                                 │
    │         │                                        * ****                                                      │
    │         │                                   **** *                                                           │
    │         │                            * ****                                                                  │
    │         │                      * ****                                                                        │
    │         │                 * ****                                                                             │
    │         │            ** ***                                                                                  │
    │         │     * * ***                                                                                        │
    │ -3.40e+1│* * **                                                                                              │
    ├─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │         │-5.00                                                                                           4.90│
    │         │                                                 x                                                  │
    └─────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}]

let%expect_test "Simple gradients hosted" =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let%op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%op d = e + "c" [ 10 ] in
  let%op l = d *. "f" [ -2 ] in
  (* We need to either call `grad_update` before introducing `learning_rate`, or disable the
     rootness check. *)
  let grad = Train.grad_update l in
  let%op learning_rate = 0.1 in
  Train.every_non_literal_on_host l;
  Train.every_non_literal_on_host learning_rate;
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params (module Backend) ~ctx IDX.empty l in
  let grad_routine = Train.to_routine (module Backend) ctx IDX.empty grad in
  let sgd_routine = Train.to_routine (module Backend) grad_routine.context IDX.empty sgd in
  (* Check out the initial state without running an init or forward pass. *)
  Train.printf_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                       #12 *._l
                        0.00
                       #13 grad_*._l
                        0.00
                 #8 +_d                   │#10 f non-emb
                  0.00                    │ -2.00
                 #9 grad_+_d              │#11 grad_f
                  0.00                    │ 0.00
          #4 *._e            │#6 c non-emb│
           0.00              │ 1.00e+1    │
          #5 grad_*._e       │#7 grad_c   │
           0.00              │ 0.00       │
    #0 a non-emb│#2 b non-emb│            │
     2.00       │ -3.00      │            │
    #1 grad_a   │#3 grad_b   │            │
     0.00       │ 0.00       │            │
    |}];
  (* Do not update the params: all values and gradients will be at initial points, which are
     specified in the tensor in the brackets. *)
  Train.run grad_routine;
  Train.printf_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                       #12 *._l
                        -8.00
                       #13 grad_*._l
                        1.00
                 #8 +_d                   │#10 f non-emb
                  4.00                    │ -2.00
                 #9 grad_+_d              │#11 grad_f
                  -2.00                   │ 4.00
          #4 *._e            │#6 c non-emb│
           -6.00             │ 1.00e+1    │
          #5 grad_*._e       │#7 grad_c   │
           -2.00             │ -2.00      │
    #0 a non-emb│#2 b non-emb│            │
     2.00       │ -3.00      │            │
    #1 grad_a   │#3 grad_b   │            │
     6.00       │ -4.00      │            │
    |}];
  (* Now we update the params, but we are not doing the forward and backward passes: only params
     values will change, compared to the above. The update is in the opposite direction of the
     gradient. *)
  Train.run sgd_routine;
  Train.printf_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                       #12 *._l
                        -8.00
                       #13 grad_*._l
                        1.00
                 #8 +_d                   │#10 f non-emb
                  4.00                    │ -2.40
                 #9 grad_+_d              │#11 grad_f
                  -2.00                   │ 4.00
          #4 *._e            │#6 c non-emb│
           -6.00             │ 1.02e+1    │
          #5 grad_*._e       │#7 grad_c   │
           -2.00             │ -2.00      │
    #0 a non-emb│#2 b non-emb│            │
     1.40       │ -2.60      │            │
    #1 grad_a   │#3 grad_b   │            │
     6.00       │ -4.00      │            │
    |}];

  (* Now the params will remain as above, but both param gradients and the values and gradients of
     other nodes will change thanks to the forward and backward passes. *)
  Train.run grad_routine;
  Train.printf_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                       #12 *._l
                        -1.57e+1
                       #13 grad_*._l
                        1.00
                 #8 +_d                   │#10 f non-emb
                  6.56                    │ -2.40
                 #9 grad_+_d              │#11 grad_f
                  -2.40                   │ 6.56
          #4 *._e            │#6 c non-emb│
           -3.64             │ 1.02e+1    │
          #5 grad_*._e       │#7 grad_c   │
           -2.40             │ -2.40      │
    #0 a non-emb│#2 b non-emb│            │
     1.40       │ -2.60      │            │
    #1 grad_a   │#3 grad_b   │            │
     6.24       │ -3.36      │            │
    |}]

let%expect_test "Simple gradients virtual" =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let%op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%op d = e + "c" [ 10 ] in
  let%op l = d *. "f" [ -2 ] in
  (* We pretend this is for parallel updates, to force materializing gradients, because our SGD
     update is compiled separately from our gradient update. Alternatively we could compile
     grad_update and sgd_update together.*)
  let grad = Train.grad_update ~setup_for_parallel:true l in
  let%op learning_rate = 0.1 in
  let sgd = Train.sgd_update ~learning_rate l in
  (* Check out the initial state without forcing memory modes by compilation. *)
  Tensor.print_tree ~force:false ~with_grad:true ~depth:9 l;
  [%expect
    {|
                                                       #12 *._l Host&dev/412
                                                       <not-in-yet>
                                                       #13 grad_*._l unknown
                                                       <not-in-yet>
                                        #8 +_d unknown                                          │#10 f non-emb Host-non-const/24
                                        <not-in-yet>                                            │<not-in-yet>
                                        #9 grad_+_d unknown                                     │#11 grad_f Material/28
                                        <not-in-yet>                                            │<not-in-yet>
                        #4 *._e unknown                          │#6 c non-emb Host-non-const/24│
                        <not-in-yet>                             │<not-in-yet>                  │
                        #5 grad_*._e unknown                     │#7 grad_c Material/28         │
                        <not-in-yet>                             │<not-in-yet>                  │
    #0 a non-emb Host-non-const/24│#2 b non-emb Host-non-const/24│                              │
    <not-in-yet>                  │<not-in-yet>                  │                              │
    #1 grad_a Material/28         │#3 grad_b Material/28         │                              │
    <not-in-yet>                  │<not-in-yet>                  │                              │
    |}];
  let ctx = Train.init_params (module Backend) ~ctx IDX.empty l in
  let grad_routine = Train.to_routine (module Backend) ctx IDX.empty grad in
  (* Check out the state without running a forward pass or compiling the SGD update. *)
  Train.printf_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                   #12 *._l
                    0.00
                   #13 grad_*._l Virt/40
                   <void>
             #8 +_d Local/1046            │#10 f non-emb
             <void>                       │ -2.00
             #9 grad_+_d Virt/40          │#11 grad_f
             <void>                       │ 0.00
      #4 *._e Virt/152       │#6 c non-emb│
      <void>                 │ 1.00e+1    │
      #5 grad_*._e Virt/40   │#7 grad_c   │
      <void>                 │ 0.00       │
    #0 a non-emb│#2 b non-emb│            │
     2.00       │ -3.00      │            │
    #1 grad_a   │#3 grad_b   │            │
     0.00       │ 0.00       │            │
    |}];
  (* Do not update the params: all values and gradients will be at initial points, which are
     specified in the tensor in the brackets. *)
  Train.run grad_routine;
  Train.printf_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                   #12 *._l
                    -8.00
                   #13 grad_*._l Virt/40
                   <void>
             #8 +_d Local/1046            │#10 f non-emb
             <void>                       │ -2.00
             #9 grad_+_d Virt/40          │#11 grad_f
             <void>                       │ 4.00
      #4 *._e Virt/152       │#6 c non-emb│
      <void>                 │ 1.00e+1    │
      #5 grad_*._e Virt/40   │#7 grad_c   │
      <void>                 │ -2.00      │
    #0 a non-emb│#2 b non-emb│            │
     2.00       │ -3.00      │            │
    #1 grad_a   │#3 grad_b   │            │
     6.00       │ -4.00      │            │
    |}];
  (* Only now compile the SGD update. *)
  let sgd_routine = Train.to_routine (module Backend) grad_routine.context IDX.empty sgd in
  (* Now we update the params, but are not doing the forward and backward passes: only params values
     will change, compared to the above. Since virtual tensors are computed by-need, they will
     always be recomputed using the latest parameter state. *)
  Train.run sgd_routine;
  Train.printf_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                   #12 *._l
                    -8.00
                   #13 grad_*._l Virt/40
                   <void>
             #8 +_d Local/1046            │#10 f non-emb
             <void>                       │ -2.40
             #9 grad_+_d Virt/40          │#11 grad_f
             <void>                       │ 4.00
      #4 *._e Virt/152       │#6 c non-emb│
      <void>                 │ 1.02e+1    │
      #5 grad_*._e Virt/40   │#7 grad_c   │
      <void>                 │ -2.00      │
    #0 a non-emb│#2 b non-emb│            │
     1.40       │ -2.60      │            │
    #1 grad_a   │#3 grad_b   │            │
     6.00       │ -4.00      │            │
    |}];
  (* Now the params will remain as above, but both param gradients and the values and gradients of
     other nodes will change thanks to the forward and backward passes. *)
  Train.run grad_routine;
  Train.printf_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                   #12 *._l
                    -1.57e+1
                   #13 grad_*._l Virt/40
                   <void>
             #8 +_d Local/1046            │#10 f non-emb
             <void>                       │ -2.40
             #9 grad_+_d Virt/40          │#11 grad_f
             <void>                       │ 6.56
      #4 *._e Virt/152       │#6 c non-emb│
      <void>                 │ 1.02e+1    │
      #5 grad_*._e Virt/40   │#7 grad_c   │
      <void>                 │ -2.40      │
    #0 a non-emb│#2 b non-emb│            │
     1.40       │ -2.60      │            │
    #1 grad_a   │#3 grad_b   │            │
     6.24       │ -3.36      │            │
    |}]

let%expect_test "tanh plot" =
  Tensor.unsafe_reinitialize ();
  (* TODO: NOT IMPLEMENTED *)
  ()

let%expect_test "2D neuron hosted" =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let%op v = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  Train.every_non_literal_on_host v;
  let update = Train.grad_update v in
  let ctx = Train.init_params (module Backend) IDX.empty v in
  let routine = Train.to_routine (module Backend) ctx IDX.empty update in
  Train.run routine;
  Train.printf_tree ~with_grad:true ~depth:9 v;
  [%expect
    {|
                  #8 +_v
                   7.00e-1
                  #9 grad_+_v
                   1.00
             #6 *              │#0 b non-emb
              -6.00            │ 6.70
             #7 grad_*         │#1 grad_b
              1.00             │ 1.00
    #2 w non-emb │#4 x non-emb │
     -3.00  1.00 │ 2.00  0.00  │
    #3 grad_w    │#5 grad_x    │
     2.00  0.00  │ -3.00  1.00 │
    |}]

let%expect_test "2D neuron virtual" =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let%op v = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  let update = Train.grad_update v in
  let ctx = Train.init_params (module Backend) IDX.empty v in
  let routine = Train.to_routine (module Backend) ctx IDX.empty update in
  Train.run routine;
  Train.printf_tree ~with_grad:true ~depth:9 v;
  [%expect
    {|
              #8 +_v
               7.00e-1
              #9 grad_+_v Virt/40
              <void>
         #6 * Local/1046       │#0 b non-emb
         <void>                │ 6.70
         #7 grad_* Virt/40     │#1 grad_b
         <void>                │ 1.00
    #2 w non-emb │#4 x non-emb │
     -3.00  1.00 │ 2.00  0.00  │
    #3 grad_w    │#5 grad_x    │
     2.00  0.00  │ -3.00  1.00 │
    |}]

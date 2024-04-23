open Base
open Ocannl
module Tn = Arrayjit.Tnode
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module Utils = Arrayjit.Utils
module Rand = Arrayjit.Rand.Lib

let%expect_test "Graph drawing recompile" =
  Rand.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let backend = (module Backend : Train.Backend_type with type context = Backend.context) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let open Operation.At in
  let%op f_nd = (3 *. ("x" [ 5 ] **. 2)) - (4 *. x) + 5 in
  Train.set_hosted x.value;
  Train.forward_and_forget backend ctx f_nd;
  Tensor.print_tree ~with_grad:true ~depth:9 f_nd;
  [%expect
    {|
                                              #15 +_f_nd
                                               6.00e+1
                                              #16 grad_+_f_nd <waiting>
                                              <not-in-yet>
                                       #13 - <Virtual 152>                                    │#2 5. <Virtual 40>
                                       <not-in-yet>                                           │<not-in-yet>
                                       #14 grad_- <waiting>                                   │
                                       <not-in-yet>                                           │
              #11 *. <Virtual 152>            │             #4 *. <Virtual 152>               │
              <not-in-yet>                    │             <not-in-yet>                      │
              #12 grad_*. <waiting>           │             #5 grad_*. <waiting>              │
              <not-in-yet>                    │             <not-in-yet>                      │
    #10 3. <Virtual 40>│#7 **. <Virtual 152>  │#3 4. <Virtual 40>│#0 x                        │
    <not-in-yet>       │<not-in-yet>          │<not-in-yet>      │ 5.00e+0                    │
                       │#8 grad_**. <waiting> │                  │#1 grad_x <Never_virtual 26>│
                       │<not-in-yet>          │                  │<not-in-yet>                │
                       │[0]│#6 2. <Virtual 40>│                  │                            │
                       │   │<not-in-yet>      │                  │                            │ |}];
  let%op f = (3 *. ("x" [ 5 ] **. 2)) - (4 *. x) + 5 in
  Train.every_non_literal_on_host f;
  let f_upd = Train.grad_update f in
  let f_bprop = Backend.jit ctx IDX.empty f_upd.fwd_bprop in
  Train.sync_run backend f_bprop f;
  Tensor.print_tree ~with_grad:true ~depth:9 f;
  [%expect
    {|
                                             #32 +_f
                                              6.00e+1
                                             #33 grad_+_f
                                              1.00e+0
                                    #30 -                                      │#19 5. <Virtual 40>
                                     5.50e+1                                   │<not-in-yet>
                                    #31 grad_-                                 │
                                     1.00e+0                                   │
                    #28 *.                      │         #21 *.               │
                     7.50e+1                    │          2.00e+1             │
                    #29 grad_*.                 │         #22 grad_*.          │
                     1.00e+0                    │          -1.00e+0            │
    #27 3. <Virtual 40>│      #24 **.           │#20 4. <Virtual 40>│#17 x     │
    <not-in-yet>       │       2.50e+1          │<not-in-yet>       │ 5.00e+0  │
                       │      #25 grad_**.      │                   │#18 grad_x│
                       │       3.00e+0          │                   │ 2.60e+1  │
                       │[17]│#23 2. <Virtual 40>│                   │          │
                       │    │<not-in-yet>       │                   │          │
  |}];
  let xs = Array.init 10 ~f:Float.(fun i -> of_int i - 5.) in
  let ys =
    Array.map xs ~f:(fun v ->
        (* This is inefficient because it compiles the argument update inside the loop. *)
        let assign_x = Backend.jit f_bprop.context IDX.empty ~name:"assign_x" [%cd x =: !.v] in
        Train.sync_run (module Backend) assign_x x;
        Train.sync_run (module Backend) f_bprop f;
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
  Rand.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let backend = (module Backend : Train.Backend_type with type context = Backend.context) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let open Operation.At in
  CDSL.virtualize_settings.enable_device_only <- false;
  let%op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%op f5 = f 5 in
  Train.every_non_literal_on_host f5;
  Train.forward_and_forget (module Backend) ctx f5;
  Tensor.print_tree ~with_grad:false ~depth:9 f5;
  [%expect
    {|
                                                   #54 +_f
                                                    6.00e+1
                                         #53 -                                          │#46 5. <Virtual 40>
                                          5.50e+1                                       │<not-in-yet>
                     #52 *.                     │               #48 *.                  │
                      7.50e+1                   │                2.00e+1                │
    #51 3. <Virtual 40>│       #50 **.          │#47 4. <Virtual 40>│#45 5. <Virtual 40>│
    <not-in-yet>       │        2.50e+1         │<not-in-yet>       │<not-in-yet>       │
                       │[45]│#49 2. <Virtual 40>│                   │                   │
                       │    │<not-in-yet>       │                   │                   │ |}];
  let size = 100 in
  let xs = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  (* Yay, the whole shape gets inferred! *)
  let x_flat =
    Tensor.term ~grad_spec:Require_grad ~label:[ "x_flat" ]
      ~init_op:(Constant_fill { values = xs; strict = true })
      ()
  in
  let step_sym, bindings = IDX.get_static_symbol ~static_range:size IDX.empty in
  let%op x = x_flat @| step_sym in
  let%op fx = f x in
  Train.set_hosted x.value;
  Train.set_hosted (Option.value_exn x.diff).grad;
  let update = Train.grad_update fx in
  let fx_routine = Backend.jit ctx bindings update.fwd_bprop in
  let step_ref = IDX.find_exn fx_routine.bindings step_sym in
  let ys, dys =
    Array.unzip
    @@ Array.mapi xs ~f:(fun i _ ->
           step_ref := i;
           Train.sync_run backend fx_routine fx;
           (fx.@[0], x.@%[0]))
  in
  (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
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

let%expect_test "Simple gradients hosted" =
  Rand.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let backend = (module Backend : Train.Backend_type with type context = Backend.context) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let%op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%op d = e + "c" [ 10 ] in
  let%op l = d *. "f" [ -2 ] in
  (* We need to either call `grad_update` before introducing `learning_rate`, or disable the rootness
     check. *)
  let grad = Train.grad_update l in
  let%op learning_rate = 0.1 in
  Train.every_non_literal_on_host l;
  Train.every_non_literal_on_host learning_rate;
  let sgd = Train.sgd_update ~learning_rate grad in
  let grad_routine = Backend.jit ctx IDX.empty grad.fwd_bprop in
  let sgd_routine = Backend.jit grad_routine.context IDX.empty sgd in
  (* Check out the initial state without running a forward pass. *)
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                   #87 *._l
                    0.00e+0
                   #88 grad_*._l
                    0.00e+0
              #83 +_d               │#85 f
               0.00e+0              │ -2.00e+0
              #84 grad_+_d          │#86 grad_f
               0.00e+0              │ 0.00e+0
        #79 *._e         │#81 c     │
         0.00e+0         │ 1.00e+1  │
        #80 grad_*._e    │#82 grad_c│
         0.00e+0         │ 0.00e+0  │
    #75 a     │#77 b     │          │
     2.00e+0  │ -3.00e+0 │          │
    #76 grad_a│#78 grad_b│          │
     0.00e+0  │ 0.00e+0  │          │ |}];
  (* Do not update the params: all values and gradients will be at initial points, which are specified in the
     tensor in the brackets. *)
  Train.sync_run backend grad_routine l;
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                   #87 *._l
                    -8.00e+0
                   #88 grad_*._l
                    1.00e+0
              #83 +_d               │#85 f
               4.00e+0              │ -2.00e+0
              #84 grad_+_d          │#86 grad_f
               -2.00e+0             │ 4.00e+0
        #79 *._e         │#81 c     │
         -6.00e+0        │ 1.00e+1  │
        #80 grad_*._e    │#82 grad_c│
         -2.00e+0        │ -2.00e+0 │
    #75 a     │#77 b     │          │
     2.00e+0  │ -3.00e+0 │          │
    #76 grad_a│#78 grad_b│          │
     6.00e+0  │ -4.00e+0 │          │ |}];
  (* Now we update the params, but we are not doing the forward and backward passes: only params values will
     change, compared to the above. The update is in the opposite direction of the gradient. *)
  Train.sync_run backend sgd_routine l;
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                   #87 *._l
                    -8.00e+0
                   #88 grad_*._l
                    1.00e+0
              #83 +_d               │#85 f
               4.00e+0              │ -2.40e+0
              #84 grad_+_d          │#86 grad_f
               -2.00e+0             │ 4.00e+0
        #79 *._e         │#81 c     │
         -6.00e+0        │ 1.02e+1  │
        #80 grad_*._e    │#82 grad_c│
         -2.00e+0        │ -2.00e+0 │
    #75 a     │#77 b     │          │
     1.40e+0  │ -2.60e+0 │          │
    #76 grad_a│#78 grad_b│          │
     6.00e+0  │ -4.00e+0 │          │ |}];

  (* Now the params will remain as above, but both param gradients and the values and gradients of other nodes
     will change thanks to the forward and backward passes. *)
  Train.sync_run backend grad_routine l;
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                     #87 *._l
                      -1.57e+1
                     #88 grad_*._l
                      1.00e+0
                #83 +_d               │#85 f
                 6.56e+0              │ -2.40e+0
                #84 grad_+_d          │#86 grad_f
                 -2.40e+0             │ 6.56e+0
          #79 *._e         │#81 c     │
           -3.64e+0        │ 1.02e+1  │
          #80 grad_*._e    │#82 grad_c│
           -2.40e+0        │ -2.40e+0 │
      #75 a     │#77 b     │          │
       1.40e+0  │ -2.60e+0 │          │
      #76 grad_a│#78 grad_b│          │
       6.24e+0  │ -3.36e+0 │          │ |}]

let%expect_test "Simple gradients virtual" =
  Rand.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let backend = (module Backend : Train.Backend_type with type context = Backend.context) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let%op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%op d = e + "c" [ 10 ] in
  let%op l = d *. "f" [ -2 ] in
  (* We pretend this is for parallel updates, to force materializing gradients, because our SGD update is
     compiled separately from our gradient update. Alternatively we could mark all
     [Assignments.recurrent_nodes sgd] as materialized. Or, the best non-parallel option is to compile
     grad_update and sgd_update together.*)
  let grad = Train.grad_update ~setup_for_parallel:true l in
  let%op learning_rate = 0.1 in
  let sgd = Train.sgd_update ~learning_rate grad in
  (* Check out the initial state without forcing memory modes by compilation. *)
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                                                #123 *._l <(Hosted Changed_on_devices) 41>
                                                <not-in-yet>
                                                #124 grad_*._l <waiting>
                                                <not-in-yet>
                                         #119 +_d <waiting>                                           │#121 f <(Hosted Nonconstant) 24>
                                         <not-in-yet>                                                 │<not-in-yet>
                                         #120 grad_+_d <waiting>                                      │#122 grad_f <Materialized 28>
                                         <not-in-yet>                                                 │<not-in-yet>
                        #115 *._e <waiting>                          │#117 c <(Hosted Nonconstant) 24>│
                        <not-in-yet>                                 │<not-in-yet>                    │
                        #116 grad_*._e <waiting>                     │#118 grad_c <Materialized 28>   │
                        <not-in-yet>                                 │<not-in-yet>                    │
    #111 a <(Hosted Nonconstant) 24>│#113 b <(Hosted Nonconstant) 24>│                                │
    <not-in-yet>                    │<not-in-yet>                    │                                │
    #112 grad_a <Materialized 28>   │#114 grad_b <Materialized 28>   │                                │
    <not-in-yet>                    │<not-in-yet>                    │                                │ |}];
  let grad_routine = Backend.jit ctx IDX.empty grad.fwd_bprop in
  (* Check out the state without running a forward pass or compiling the SGD update. *)
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                                            #123 *._l
                                             0.00e+0
                                            #124 grad_*._l <Virtual 40>
                                            <not-in-yet>
                               #119 +_d <Local 33>                                  │#121 f
                               <void>                                               │ -2.00e+0
                               #120 grad_+_d <Virtual 40>                           │#122 grad_f <On_device 33>
                               <not-in-yet>                                         │<void>
                 #115 *._e <Virtual 152>                 │#117 c                    │
                 <not-in-yet>                            │ 1.00e+1                  │
                 #116 grad_*._e <Virtual 40>             │#118 grad_c <On_device 33>│
                 <not-in-yet>                            │<void>                    │
    #111 a                    │#113 b                    │                          │
     2.00e+0                  │ -3.00e+0                 │                          │
    #112 grad_a <On_device 33>│#114 grad_b <On_device 33>│                          │
    <void>                    │<void>                    │                          │ |}];
  (* Do not update the params: all values and gradients will be at initial points, which are specified in the
     tensor in the brackets. *)
  Train.sync_run backend grad_routine l;
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                                            #123 *._l
                                             -8.00e+0
                                            #124 grad_*._l <Virtual 40>
                                            <not-in-yet>
                               #119 +_d <Local 33>                                  │#121 f
                               <void>                                               │ -2.00e+0
                               #120 grad_+_d <Virtual 40>                           │#122 grad_f <On_device 33>
                               <not-in-yet>                                         │<void>
                 #115 *._e <Virtual 152>                 │#117 c                    │
                 <not-in-yet>                            │ 1.00e+1                  │
                 #116 grad_*._e <Virtual 40>             │#118 grad_c <On_device 33>│
                 <not-in-yet>                            │<void>                    │
    #111 a                    │#113 b                    │                          │
     2.00e+0                  │ -3.00e+0                 │                          │
    #112 grad_a <On_device 33>│#114 grad_b <On_device 33>│                          │
    <void>                    │<void>                    │                          │ |}];
  (* Only now compile the SGD update. *)
  let sgd_routine = Backend.jit grad_routine.context IDX.empty sgd in
  (* Now we update the params, but are not doing the forward and backward passes: only params values will
     change, compared to the above. Since virtual tensors are computed by-need, they will always be recomputed
     using the latest parameter state. *)
  Train.sync_run backend sgd_routine l;
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                                            #123 *._l
                                             -8.00e+0
                                            #124 grad_*._l <Virtual 40>
                                            <not-in-yet>
                               #119 +_d <Local 33>                                  │#121 f
                               <void>                                               │ -2.40e+0
                               #120 grad_+_d <Virtual 40>                           │#122 grad_f <On_device 33>
                               <not-in-yet>                                         │<void>
                 #115 *._e <Virtual 152>                 │#117 c                    │
                 <not-in-yet>                            │ 1.02e+1                  │
                 #116 grad_*._e <Virtual 40>             │#118 grad_c <On_device 33>│
                 <not-in-yet>                            │<void>                    │
    #111 a                    │#113 b                    │                          │
     1.40e+0                  │ -2.60e+0                 │                          │
    #112 grad_a <On_device 33>│#114 grad_b <On_device 33>│                          │
    <void>                    │<void>                    │                          │ |}];
  (* Now the params will remain as above, but both param gradients and the values and gradients of other nodes
     will change thanks to the forward and backward passes. *)
  Train.sync_run backend grad_routine l;
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  [%expect
    {|
                                              #123 *._l
                                               -1.57e+1
                                              #124 grad_*._l <Virtual 40>
                                              <not-in-yet>
                                 #119 +_d <Local 33>                                  │#121 f
                                 <void>                                               │ -2.40e+0
                                 #120 grad_+_d <Virtual 40>                           │#122 grad_f <On_device 33>
                                 <not-in-yet>                                         │<void>
                   #115 *._e <Virtual 152>                 │#117 c                    │
                   <not-in-yet>                            │ 1.02e+1                  │
                   #116 grad_*._e <Virtual 40>             │#118 grad_c <On_device 33>│
                   <not-in-yet>                            │<void>                    │
      #111 a                    │#113 b                    │                          │
       1.40e+0                  │ -2.60e+0                 │                          │
      #112 grad_a <On_device 33>│#114 grad_b <On_device 33>│                          │
      <void>                    │<void>                    │                          │ |}]

let%expect_test "tanh plot" =
  (* TODO: NOT IMPLEMENTED *)
  ()

let%expect_test "2D neuron hosted" =
  Rand.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let backend = (module Backend : Train.Backend_type with type context = Backend.context) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let%op v = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  Train.every_non_literal_on_host v;
  let update = Train.grad_update v in
  let routine = Backend.jit ctx IDX.empty update.fwd_bprop in
  Train.sync_run backend routine v;
  Tensor.print_tree ~with_grad:true ~depth:9 v;
  [%expect
    {|
                       #155 +_v
                        7.00e-1
                       #156 grad_+_v
                        1.00e+0
                  #153 *                   │#147 b
                   -6.00e+0                │ 6.70e+0
                  #154 grad_*              │#148 grad_b
                   1.00e+0                 │ 1.00e+0
    #149 w             │#151 x             │
     -3.00e+0  1.00e+0 │ 2.00e+0  0.00e+0  │
    #150 grad_w        │#152 grad_x        │
     2.00e+0  0.00e+0  │ -3.00e+0  1.00e+0 │ |}]

let%expect_test "2D neuron virtual" =
  Rand.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let backend = (module Backend : Train.Backend_type with type context = Backend.context) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let%op v = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  let update = Train.grad_update v in
  let routine = Backend.jit ctx IDX.empty update.fwd_bprop in
  Train.sync_run backend routine v;
  Tensor.print_tree ~with_grad:true ~depth:9 v;
  [%expect
    {|
                         #166 +_v
                          7.00e-1
                         #167 grad_+_v <Virtual 40>
                         <not-in-yet>
              #164 * <Local 33>                  │#158 b
              <void>                             │ 6.70e+0
              #165 grad_* <Virtual 40>           │#159 grad_b <Local 33>
              <not-in-yet>                       │<void>
    #160 w                │#162 x                │
     -3.00e+0  1.00e+0    │ 2.00e+0  0.00e+0     │
    #161 grad_w <Local 33>│#163 grad_x <Local 33>│
    <void>                │<void>                │ |}]

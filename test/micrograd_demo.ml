open Base
open Ocannl
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Asgns = Arrayjit.Assignments
module Rand = Arrayjit.Rand.Lib
module Tn = Arrayjit.Tnode

module type Backend = Arrayjit.Backend_intf.Backend

let%expect_test "Micrograd README basic example" =
  Tensor.unsafe_reinitialize ();
  Rand.init 0;
  let module Backend = (val Arrayjit.Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let%op c = "a" [ -4 ] + "b" [ 2 ] in
  let%op d = (a *. b) + (b **. 3) in
  let%op c = c + c + 1 in
  let%op c = c + 1 + c + ~-a in
  let%op d = d + (d *. 2) + relu (b + a) in
  let%op d = d + (3 *. d) + relu (b - a) in
  let%op e = c - d in
  let%op f = e **. 2 in
  let%op g = f /. 2 in
  let%op g = g + (10. /. f) in
  List.iter ~f:(Option.iter ~f:(fun diff -> Train.set_hosted diff.Tensor.grad)) [ a.diff; b.diff ];
  let update = Train.grad_update g in
  let step = Train.to_routine (module Backend) ctx IDX.empty update.fwd_bprop in
  Train.run step;
  Tensor.print ~with_code:false ~with_grad:false `Default g;
  [%expect
    {|
    ┌────────────────────┐
    │[75]: +_g shape 0:1 │
    │┌┬─────────┐        │
    │││axis 0   │        │
    │├┼─────────┤        │
    │││ 2.47e+1 │        │
    │└┴─────────┘        │
    └────────────────────┘
    |}];
  Tensor.print ~with_code:false ~with_grad:true `Default a;
  [%expect
    {|
    ┌─────────────────┐
    │[0]: a shape 0:1 │
    │┌┬───────┐       │
    │││axis 0 │       │
    │├┼───────┤       │
    │││ -4.00 │       │
    │└┴───────┘       │
    └─────────────────┘
    ┌────────────────────────┐
                                                              │[0]: a shape 0:1  grad_a│
                                                              │┌┬─────────┐            │
                                                              │││axis 0   │            │
                                                              │├┼─────────┤            │
                                                              │││ 1.39e+2 │            │
                                                              │└┴─────────┘            │
                                                              └────────────────────────┘
    |}];
  Tensor.print ~with_code:false ~with_grad:true `Default b;
  [%expect
    {|
    ┌─────────────────┐
    │[2]: b shape 0:1 │
    │┌┬──────┐        │
    │││axis 0│        │
    │├┼──────┤        │
    │││ 2.00 │        │
    │└┴──────┘        │
    └─────────────────┘
    ┌────────────────────────┐
                                                              │[2]: b shape 0:1  grad_b│
                                                              │┌┬─────────┐            │
                                                              │││axis 0   │            │
                                                              │├┼─────────┤            │
                                                              │││ 6.46e+2 │            │
                                                              │└┴─────────┘            │
                                                              └────────────────────────┘
    |}]

let%expect_test "Micrograd half-moons example" =
  Tensor.unsafe_reinitialize ();
  Rand.init 5;
  (* Note: for as-yet unknown reason, this test can lead to different resuls on different versions
     of dependencies. *)
  let module Backend = (val Arrayjit.Backends.fresh_backend ~backend_name:"cc" ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let open Operation.At in
  let len = 200 in
  let batch_size = 10 in
  let n_batches = 2 * len / batch_size in
  let epochs = 10 in
  let steps = epochs * 2 * len / batch_size in
  let noise () = Rand.float_range (-0.1) 0.1 in
  let moons_flat =
    Array.concat_map (Array.create ~len ())
      ~f:
        Float.(
          fun () ->
            let i = Rand.int len in
            let v = of_int i * pi / of_int len in
            let c = cos v and s = sin v in
            [| c + noise (); s + noise (); 1.0 - c + noise (); 0.5 - s + noise () |])
  in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  (* FIXME: should also work with explicit batch shape. *)
  let moons_flat =
    TDSL.init_const ~l:"moons_flat" (* ~b:[ n_batches; batch_size ] *) ~o:[ 2 ] moons_flat
  in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  (* FIXME: should also work with explicit batch shape. *)
  let moons_classes =
    TDSL.init_const ~l:"moons_classes" (* ~b:[ n_batches; batch_size ] *) ~o:[ 1 ] moons_classes
  in
  let%op mlp x = "b3" + ("w3" * relu ("b2" 16 + ("w2" * relu ("b1" 16 + ("w1" * x))))) in
  (* Don't decay the learning rate too quickly, it behaves better than in the original. *)
  let%op moons_input = moons_flat @| batch_n in
  (* Tell shape inference to make a minibatch axis. *)
  let%cd _ = moons_input =: 0 ++ "i=>2|i" in
  let%op moons_class = moons_classes @| batch_n in
  let%cd _ = moons_class =: 0 ++ "i=>2|i" in
  let losses = ref [] in
  let log_losses = ref [] in
  let learning_rates = ref [] in
  let%op margin_loss = relu (1 - (moons_class *. mlp moons_input)) in
  (* We don't need a regression loss formula thanks to weight_decay built into the sgd_update
     computation. *)
  let weight_decay = 0.0001 in
  let%op scalar_loss = (margin_loss ++ "...|... => 0") /. !..batch_size in
  let update = Train.grad_update scalar_loss in
  let%op learning_rate = 0.1 *. ((2 *. !..steps) - !@step_n) /. !..steps in
  Train.set_hosted learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate ~weight_decay update in
  let sgd_routine =
    Train.to_routine (module Backend) ctx bindings (Asgns.sequence [ update.fwd_bprop; sgd ])
  in
  let step_ref = IDX.find_exn sgd_routine.bindings step_n in
  step_ref := 0;
  for _epoch = 1 to epochs do
    Train.sequential_loop sgd_routine.bindings ~f:(fun () ->
        Train.run sgd_routine;
        (* let batch_ref = IDX.find_exn sgd_jitted.bindings batch_n in Stdio.printf "Epoch=%d,
           step=%d, batch=%d, lr=%f, loss=%f\n%!" epoch !step_ref !batch_ref learning_rate.@[0]
           scalar_loss.@[0]; *)
        learning_rates := ~-.(learning_rate.@[0]) :: !learning_rates;
        losses := scalar_loss.@[0] :: !losses;
        log_losses := Float.max (-10.) (Float.log scalar_loss.@[0]) :: !log_losses;
        Int.incr step_ref)
  done;
  let points = Tn.points_2d ~xdim:0 ~ydim:1 moons_flat.value in
  let classes = Tn.points_1d ~xdim:0 moons_classes.value in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  let%op mlp_result = mlp "point" in
  Train.set_on_host mlp_result.value;
  let result_routine =
    Train.to_routine
      (module Backend)
      sgd_routine.context IDX.empty
      [%cd
        ~~("moons infer";
           mlp_result.forward)]
  in
  let callback (x, y) =
    Tn.set_values point.value [| x; y |];
    (* For the gccjit backend, point is only on host, not on device. For cuda, this will be
       needed. *)
    Train.run result_routine;
    Float.(mlp_result.@[0] >= 0.)
  in
  let plot_moons =
    PrintBox_utils.plot ~as_canvas:true
      [
        Scatterplot { points = points1; content = PrintBox.line "#" };
        Scatterplot { points = points2; content = PrintBox.line "%" };
        Boundary_map
          { content_false = PrintBox.line "."; content_true = PrintBox.line "*"; callback };
      ]
  in
  Stdio.printf "Half-moons scatterplot and decision boundary:\n%!";
  PrintBox_text.output Stdio.stdout plot_moons;
  [%expect
    {|
    Half-moons scatterplot and decision boundary:
    ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │********************************#*******************************************************************│
    │************************#*##*#*##*#*##*#************************************************************│
    │*********************##*#****************##*#*******************************************************│
    │*******************#************************#*##****************************************************│
    │****************##*******************************#**************************************************│
    │**************##***********************************#************************************************│
    │************##***************************************#**********************************************│
    │**********##*****************************************#*#********************************************│
    │**********#********************************************#********************************************│
    │********##***********************************************#*************************************.....│
    │*******#***************************************************#******************************..........│
    │*******#**************************************************##**************************..............│
    │*****##*****************************************************#********************...................│
    │****#*******************************************************#***************........................│
    │****#***************************%%***************************##********.............................│
    │***#****************************%*********************************................................%%│
    │***#*****************************%***************************.##....................................│
    │**#*****************************%%**********************..........................................%.│
    │**#******************************%*****************............##...................................│
    │*#*******************************%************..................#................................%%.│
    │**********************************%*******......................#................................%..│
    │##*******************************%***............................#...............................%..│
    │********************************..%%............................##..............................%...│
    │#**************************.......%..............................#..............................%...│
    │**********************.............%%............................#.............................%....│
    │#****************.................................................#............................%....│
    │************........................%%...........................#............................%.....│
    │*******.............................................................................................│
    │**....................................%.....................................................%%......│
    │.......................................%%..................................................%........│
    │.......................................%...................................................%........│
    │.........................................%...............................................%%.........│
    │...........................................%............................................%...........│
    │...........................................%.%.........................................%............│
    │.............................................%.......................................%%.............│
    │...............................................%...................................%%...............│
    │.................................................%...............................%..................│
    │...................................................%%.%........................%....................│
    │.....................................................%%.%.................%.%%......................│
    │........................................................%.%%.%%.%.%%.%.%%.%.........................│
    └────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}];
  Stdio.printf "Loss:\n%!";
  let plot_loss =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"loss"
      [ Line_plot { points = Array.of_list_rev !losses; content = PrintBox.line "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  [%expect
    {|
    Loss:
    ┌────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ 1.6│            -                                                                                       │
    │    │      - -               -                                                                           │
    │    │          -     -   -     - -       -   -   - -                                                     │
    │    │              -   -   -       - - -   -   -     -   - - - - - - - - -     - - - - -                 │
    │    │                                                  -                   - -           - - -         - │
    │    │                                                                                          - - - -   │
    │    │                                                                                                    │
    │    │    -                                                                                               │
    │    │                                                                                                    │
    │    │  -                                                                                                 │
    │    │                                                                                                    │
    │    │                                                                                                    │
    │    │                                                                                                    │
    │    │                                                                                                    │
    │    │                                                                                                    │
    │    │                                                                                                    │
    │    │                                                                                                    │
    │    │                                                                                                    │
    │l   │-                                                                                                   │
    │o   │                                                                                                    │
    │s   │--                                                                                                  │
    │s   │ -                                                                                                  │
    │    │- -                                                                                                 │
    │    │                       -                                                                            │
    │    │ -         -       -         -                                                                      │
    │    │   -           -                 -     -     -     -   -                 -                     - -  │
    │    │  -    -    -   -          -       - -   - -   - -   -   - - - - - - - -   -   - - - - - - - -     -│
    │    │    -   -           -    -     -                                             -                      │
    │    │  -  --  -           -                                                                              │
    │    │             -   -                                                                                  │
    │    │   -                          -                                                                     │
    │    │   -      -       -     - -                                                                         │
    │    │   -                        -                                                                       │
    │    │              -       -           -                                                                 │
    │    │    -                               -   - -                                                         │
    │    │                                -     -         -   -       -                                       │
    │    │    -                                         -   -     -     -   -   -                   -         │
    │    │     -                                      -         -   -     -   -   - - - - - - - - -   - - - - │
    │    │     -                                                                                              │
    │ 0  │      ----------------------------------------------------------------------------------------------│
    ├────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │    │0                                                                                              4e+02│
    │    │                                                step                                                │
    └────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}];
  Stdio.printf "Log-loss, for better visibility:\n%!";
  let plot_loss =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"log loss"
      [ Line_plot { points = Array.of_list_rev !log_losses; content = PrintBox.line "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  [%expect
    {|
    Log-loss, for better visibility:
    ┌─────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ 0.48│            -                                                                                       │
    │     │  - - - - -   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - │
    │     │                                                                                                    │
    │     │--                                                                                                  │
    │     │ ---       --  --  -   -   - -   - -   - - - -   - - - - - -   - - - - - - -   -   - - - - - - - - -│
    │     │  - ------   -   -  --   -     -     -         -             -               -   -                  │
    │     │   -      -       -     - -   -                                                                     │
    │     │   -          -       -     -     -                                                                 │
    │     │    -                               - - - -                                                         │
    │     │    -                           -             - - - -   -   -         -                             │
    │     │                                            -         -       -   -     - -   - -         -         │
    │     │     -                                                    -     -   -       -     - - - -   - - - - │
    │     │     -                                                                                              │
    │     │     -                                                                                              │
    │     │                                                                                                    │
    │     │                      -             -                                                               │
    │l    │                              -           -                                                         │
    │o    │                -       -                             -                                             │
    │g    │                                            -   -                     -         -                   │
    │     │                                                            -     -                                 │
    │l    │      -     -     -       -                                             -                           │
    │o    │               -                -     -                                                   -         │
    │s    │                                                  -     -                                           │
    │s    │                                                              -                                     │
    │     │                                                                            -                       │
    │     │                                                                                  -                 │
    │     │                                                                                      -             │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                -   │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │ -10 │      ----------------------------------------------------------------------------------------------│
    ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │     │0                                                                                              4e+02│
    │     │                                                step                                                │
    └─────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}];
  Stdio.printf "\nLearning rate:\n%!";
  let plot_lr =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"learning rate"
      [ Line_plot { points = Array.of_list_rev !learning_rates; content = PrintBox.line "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_lr;
  [%expect
    {|
    Learning rate:
    ┌─────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ -0.1│                                                                                                   -│
    │     │                                                                                                 ---│
    │     │                                                                                              ----  │
    │     │                                                                                            ---     │
    │     │                                                                                         ----       │
    │     │                                                                                       ---          │
    │     │                                                                                    ---             │
    │     │                                                                                  ---               │
    │     │                                                                               ---                  │
    │     │                                                                            ----                    │
    │     │                                                                          ---                       │
    │     │                                                                       ----                         │
    │     │                                                                     ---                            │
    │l    │                                                                  ----                              │
    │e    │                                                                ---                                 │
    │a    │                                                             ---                                    │
    │r    │                                                           ---                                      │
    │n    │                                                        ---                                         │
    │i    │                                                     ----                                           │
    │n    │                                                   ---                                              │
    │g    │                                                ----                                                │
    │     │                                              ---                                                   │
    │r    │                                           ----                                                     │
    │a    │                                         ---                                                        │
    │t    │                                      ---                                                           │
    │e    │                                    ---                                                             │
    │     │                                 ---                                                                │
    │     │                              ----                                                                  │
    │     │                            ---                                                                     │
    │     │                         ----                                                                       │
    │     │                       ---                                                                          │
    │     │                    ----                                                                            │
    │     │                  ---                                                                               │
    │     │               ---                                                                                  │
    │     │             ---                                                                                    │
    │     │          ---                                                                                       │
    │     │       ----                                                                                         │
    │     │     ---                                                                                            │
    │     │  ----                                                                                              │
    │ -0.2│---                                                                                                 │
    ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │     │0                                                                                              4e+02│
    │     │                                                step                                                │
    └─────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}];

  (* Testing how the syntax extension %op creates labels for the resulting tensors: *)
  Stdio.printf "mlp_result's name: %s\n%!" @@ Tensor.debug_name mlp_result;
  (* Note: mlp_result is not included in the resulting tensor's label, because the identifier label
     does not propagate across function calls. *)
  [%expect {| mlp_result's name: mlp_point |}];
  (Stdio.printf "(mlp moons_input) name: %s\n%!"
  @@ Tensor.debug_name
  @@
  match margin_loss.children with
  | [
   {
     subtensor =
       { children = [ _; { subtensor = { children = [ _; { subtensor; _ } ]; _ }; _ } ]; _ };
     _;
   };
  ] ->
      subtensor
  | _ -> assert false);
  [%expect {| (mlp moons_input) name: mlp_moons_input |}]

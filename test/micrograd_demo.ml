open Base
open Ocannl
module IDX = Arrayjit.Indexing.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Arrayjit.Low_level.CDSL

let%expect_test "Micrograd README basic example" =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let%op c = "a" [ -4 ] + "b" [ 2 ] in
  let%op d = (a *. b) + (b **. 3) in
  let%op c = c + c + 1 in
  let%op c = c + 1 + c + ~-a in
  let%op d = d + (d *. 2) + ?/(b + a) in
  let%op d = d + (3 *. d) + ?/(b - a) in
  let%op e = c - d in
  let%op f = e **. 2 in
  let%op g = f /. 2 in
  let%op g = g + (10. /. f) in
  List.iter ~f:(Option.iter ~f:(fun diff -> Train.set_on_host diff.Tensor.grad)) [ a.diff; b.diff ];
  let update = Train.grad_update g in
  let step = Backend.jit ctx IDX.empty update.fwd_bprop in
  Train.run step;
  Backend.await device;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ g;
  [%expect
    {|
    ┌──────────────────────┐
    │[49]: g <+> shape 0:1 │
    │┌┬─────────┐          │
    │││axis 0   │          │
    │├┼─────────┼───────── │
    │││ 2.47e+1 │          │
    │└┴─────────┘          │
    └──────────────────────┘ |}];
  Tensor.print ~with_code:false ~with_grad:true `Default @@ a;
  [%expect
    {|
    ┌───────────────────┐
    │[1]: <a> shape 0:1 │
    │┌┬──────────┐      │
    │││axis 0    │      │
    │├┼──────────┼───── │
    │││ -4.00e+0 │      │
    │└┴──────────┘      │
    └───────────────────┘
    ┌─────────────────────────────┐
    │[1]: <a> shape 0:1  Gradient │
    │┌┬─────────┐                 │
    │││axis 0   │                 │
    │├┼─────────┼──────────────── │
    │││ 1.39e+2 │                 │
    │└┴─────────┘                 │
    └─────────────────────────────┘ |}];
  Tensor.print ~with_code:false ~with_grad:true `Default @@ b;
  [%expect
    {|
    ┌───────────────────┐
    │[2]: <b> shape 0:1 │
    │┌┬─────────┐       │
    │││axis 0   │       │
    │├┼─────────┼────── │
    │││ 2.00e+0 │       │
    │└┴─────────┘       │
    └───────────────────┘
    ┌─────────────────────────────┐
    │[2]: <b> shape 0:1  Gradient │
    │┌┬─────────┐                 │
    │││axis 0   │                 │
    │├┼─────────┼──────────────── │
    │││ 6.46e+2 │                 │
    │└┴─────────┘                 │
    └─────────────────────────────┘ |}]

let%expect_test "Micrograd half-moons example" =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let open Tensor.O in
  let len = 200 in
  let batch = 10 in
  let n_batches = 2 * len / batch in
  let epochs = 50 in
  let steps = epochs * 2 * len / batch in
  let noise () = Random.float_range (-0.1) 0.1 in
  let moons_flat =
    Array.concat_map (Array.create ~len ())
      ~f:
        Float.(
          fun () ->
            let i = Random.int len in
            let v = of_int i * pi / of_int len in
            let c = cos v and s = sin v in
            [| c + noise (); s + noise (); 1.0 - c + noise (); 0.5 - s + noise () |])
  in
  let moons_flat = TDSL.init_const ~l:"moons_flat" ~b:[ epochs; batch ] ~o:[ 2 ] moons_flat in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes = TDSL.init_const ~l:"moons_classes" ~b:[ epochs; batch ] ~o:[ 1 ] moons_classes in
  let%op mlp x = "b3" 1 + ("w3" * ?/("b2" 16 + ("w2" * ?/("b1" 16 + ("w1" * x))))) in
  let step_sym, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let%op learning_rate = 0.1 *. (!..steps - !@step_sym) /. !..steps in
  let%op moons_input = moons_flat @| step_sym in
  let%op moons_class = moons_classes @| step_sym in
  let losses = ref [] in
  let log_losses = ref [] in
  let learning_rates = ref [] in
  let%op margin_loss = ?/(1 - (moons_class *. mlp moons_input)) in
  (* We don't need a regression loss formula thanks to weight_decay built into the sgd_update computation. *)
  let weight_decay = 0.0001 in
  let%op scalar_loss = (margin_loss ++ "...|... => 0") /. !..batch in
  Train.set_on_host learning_rate.value;
  let update = Train.grad_update scalar_loss in
  let sgd = Train.sgd_update ~learning_rate ~weight_decay update in
  let sgd_jitted = Backend.jit ctx bindings (Seq (update.fwd_bprop, sgd)) in
  Train.all_host_to_device (module Backend) sgd_jitted.context scalar_loss;
  Train.all_host_to_device (module Backend) sgd_jitted.context learning_rate;
  for _epoch = 1 to epochs do
    Train.sequential_loop sgd_jitted.bindings ~f:(fun () ->
        Train.run sgd_jitted;
        Backend.await device;
        assert (Backend.to_host sgd_jitted.context learning_rate.value);
        assert (Backend.to_host sgd_jitted.context scalar_loss.value);
        (* let step_ref = IDX.find_exn sgd_jitted.bindings step_sym in
           Stdio.printf "Data step=%d, lr=%f, loss=%f\n%!" !step_ref learning_rate.@[0] scalar_loss.@[0]; *)
        learning_rates := ~-.(learning_rate.@[0]) :: !learning_rates;
        losses := scalar_loss.@[0] :: !losses;
        (* epoch_loss := !epoch_loss +. scalar_loss.@[0]; *)
        log_losses := Float.log scalar_loss.@[0] :: !log_losses)
    (* Stdio.printf "Epoch %d, lr=%f, loss=%f\n%!" epoch learning_rate.@[0] !epoch_loss; *)
  done;
  let points = Tensor.value_2d_points ~xdim:0 ~ydim:1 moons_flat in
  let classes = Tensor.value_1d_points ~xdim:0 moons_classes in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  let%op point = [ 0; 0 ] in
  let mlp_result = mlp point in
  Train.set_on_host point.value;
  Train.set_on_host mlp_result.value;
  let result_jitted =
    Backend.jit sgd_jitted.context IDX.empty @@ Block_comment ("moons infer", mlp_result.forward)
  in
  let callback (x, y) =
    Tensor.set_values point [| x; y |];
    (* For the gccjit backend, point is only on host, not on device. For cuda, this will be needed. *)
    ignore (Backend.from_host result_jitted.context point.value : bool);
    Train.run result_jitted;
    Backend.await device;
    assert (Backend.to_host result_jitted.context mlp_result.value);
    Float.(mlp_result.@[0] >= 0.)
  in
  let plot_moons =
    let open PrintBox_utils in
    plot ~size:(120, 40) ~x_label:"ixes" ~y_label:"ygreks"
      [
        Scatterplot { points = points1; pixel = "#" };
        Scatterplot { points = points2; pixel = "%" };
        Boundary_map { pixel_false = "."; pixel_true = "*"; callback };
      ]
  in
  Stdio.printf "Half-moons scatterplot and decision boundary:\n%!";
  PrintBox_text.output Stdio.stdout plot_moons;
  [%expect
    {|
    Half-moons scatterplot and decision boundary:
     1.083e+0 │************************************#***********************************************************************************
              │**********************************#**#*******#*###**********************************************************************
              │*****************************************##***#**#**#*******************************************************************
              │***********************#**###**#******#*#####*#***###****##*************************************************************
              │***********************##**##***#****#*#******##********#***#***********************************************************
              │*************************#####*#***********#*#***####**#*****#*********************************************************.
              │********************#***#**************************#**#***#**#***#**************************************************....
              │**********#****#*##****#***#*******************************#**#***************************************************......
              │**************##*###*****************************************#****###*******************************************........
              │**************#*###*****************************************#*******#******************************************.........
              │***********#******************************************************##*#***************************************...........
              │********#*****#**************************....*********************#**#**##*#********************************............
              │******#***#****************************.........*****************#***#************************************..............
              │*******#***##**********************................******************#***#*******************************.........%.....
              │******#*##**#******************.......................******************##******************************..............%.
              │****##*#*#*****************.................%..........****************#***#**#***********************............%%...%
              │**#*####****************..............%..%%%............*******************#*************************................%%.
    y         │**#**#*#*#************..................%...%............******************#**#*********************.............%......
    g         │##***##***************...............%....................***************##*##*#******************......................
    r         │*******#*************..................%..%.%...............*************###**##*****************...................%%%.
    e         │*********************.................%%.%%..%...............**************####*#**************.....................%...
    k         │***#*****************..................%.%.%.%%................***********###**#**************...............%.%........
    s         │********************..................%..%.%..%.................***********#*****************................%....%%%...
              │##*#****************.....................%.%.....................***********##*************...................%.%.%%....
              │*****##************.....................%.......%..................**********#************..................%%%%%.%.%...
              │**##*#*************.....................%...%.......................*********************.......................%%......
              │#******************.........................%.........................*********#*******.......................%..%%.....
              │******************......................%..%...........................**************......................%.....%......
              │******************..........................%.%.%%%..%...................**********............................%........
              │******************.................................%........................*****........................%..%.%.........
              │*****************............................%%%%.................................................%.%...................
              │*****************.............................%.................................................%.%.%...%%%.............
              │****************..................................%%%.....%..%..............................%.%.....%%.%.%..............
              │****************.....................................%.%.%.%....................................%%.%%%..................
              │****************........................................%....%..%%..%...............%%..%...............%...............
              │***************...........................................%%.......%..%...%%...........%..%..%%%%.......................
              │***************............................................%....%...%.......%..%........%.%%....%.%.....................
              │***************..........................................%.....%%..%%.%%.%..%....%..%.%....%.%..........................
              │**************..................................................%%....%%%...%%%........%...%............................
     -5.946e-1│**************.......................................................%.......%..%.....%.................................
    ──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
              │-1.071e+0                                                                                                       2.093e+0
              │                                                          ixes |}];
  Stdio.printf "Loss:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"loss"
      [ Line_plot { points = Array.of_list_rev !losses; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  [%expect
    {|
    Loss:
     3.078e+1│-
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
             │
             │
    l        │
    o        │
    s        │
    s        │
             │
             │
             │
             │
             │-
             │
             │
             │
             │-
             │-
             │-
             │---        -
     1.312e-3│------------------------------------------------------------------------------------------------------------------------
    ─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
             │0.000e+0                                                                                                        1.999e+3
             │                                                          step |}];
  Stdio.printf "Log-loss, for better visibility:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"log loss"
      [ Line_plot { points = Array.of_list_rev !log_losses; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  [%expect
    {|
    Log-loss, for better visibility:
     3.427e+0 │-
              │
              │
              │
              │-
              │
              │-
              │-
              │- -
              │-
              │---  -     -
    l         │ --------- -- --
    o         │--- ---------------------
    g         │------------------------------
              │-- -- -------------- -----------
    l         │ ------------- ----- -- ------- ----
    o         │ - --- - -- ----------- -----------
    s         │ - ----  - -    -- ---------- --------- -  -  -  -
    s         │ --  - -- -    - -- -     --- -----     -           -     -              -
              │        -    -- -  -   -    --  ---- -  -     -  -  -  -     -  -  -        -
              │     -     --  --   --- -    - -- --- -             -     -           -        -  -     -  -  -
              │ --   -       -  --  -- - --- - ---   --  -- --- -                                   -           -  -  -  -  -  -  -  -
              │ - -   -  -  -       -   -     - --      - -  -     -  -  -     -  -                    -
              │           -        -    -    -  -- ----    -      -         -  - -      -  -  -  -
              │  - -                    -     -   --   - --     ---   - -   -     -  - -                  -
              │           --              - -   -   -    --- -     - -  --  -- - -  -                        -  -  -  -
              │    -     -             -    -  -   -  -             -                               -   -                -  -  -
              │                               -      -         -  -                                                             -
              │             -    -                            -    --------------------------------------------------------------------
     -6.637e+0│-----------------------------------------------------
    ──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
              │0.000e+0                                                                                                        1.999e+3
              │                                                          step |}];
  Stdio.printf "\nLearning rate:\n%!";
  let plot_lr =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"learning rate"
      [ Line_plot { points = Array.of_list_rev !learning_rates; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_lr;
  [%expect
    {|
    Learning rate:
     9.995e-2│-
             │-----
             │    -----
             │        -----
             │            -----
             │                -----
             │                    -----
             │                        -----
    l        │                            ------
    e        │                                 -----
    a        │                                     -----
    r        │                                         -----
    v        │                                             -----
    i        │                                                 -----
    v        │                                                     -----
    g        │                                                         -----
             │                                                              -----
    r        │                                                                  -----
    a        │                                                                      -----
    t        │                                                                          -----
    e        │                                                                              -----
             │                                                                                  -----
             │                                                                                      -----
             │                                                                                           -----
             │                                                                                               -----
             │                                                                                                   -----
             │                                                                                                       -----
             │                                                                                                           -----
             │                                                                                                               -----
     0.000e+0│                                                                                                                   -----
    ─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
             │0.000e+0                                                                                                        1.999e+3
             │                                                          step |}]

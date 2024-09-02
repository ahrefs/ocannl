open Base
open Ocannl
module Tn = Arrayjit.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Utils = Arrayjit.Utils
module Rand = Arrayjit.Rand.Lib

let%expect_test "Half-moons data parallel" =
  let seed = 1 in
  let hid_dim = 16 in
  (* let hid_dim = 4 in *)
  let batch_size = 120 in
  (* let batch_size = 60 in *)
  (* let batch_size = 20 in *)
  let len = batch_size * 20 in
  let init_lr = 0.1 in
  (* let epochs = 10 in *)
  let epochs = 20 in
  (* let epochs = 1 in *)
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
  let moons_flat ~b = TDSL.init_const ~l:"moons_flat" ~b ~o:[ 2 ] moons_flat in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes ~b = TDSL.init_const ~l:"moons_classes" ~b ~o:[ 1 ] moons_classes in
  let%op mlp x = "b3" + ("w3" * ?/("b2" hid_dim + ("w2" * ?/("b1" hid_dim + ("w1" * x))))) in
  (* let%op mlp x = "b" + ("w" * x) in *)
  let%op loss_fn ~output ~expectation = ?/(!..1 - (expectation *. output)) in
  (* We don't need a regression loss formula thanks to weight_decay built into the sgd_update
     computation. *)
  let weight_decay = 0.0002 in
  (* So that we can inspect them. *)
  let backend = Arrayjit.Backends.fresh_backend () in
  let per_batch_callback ~at_batch ~at_step ~learning_rate ~batch_loss ~epoch_loss =
    if (at_batch + 1) % 20 = 0 then
      Stdio.printf "Batch=%d, step=%d, lr=%f, batch loss=%f, epoch loss=%f\n%!" at_batch at_step
        learning_rate batch_loss epoch_loss
  in
  (* Tn.print_accessible_headers (); *)
  let per_epoch_callback ~at_step ~at_epoch ~learning_rate ~epoch_loss =
    Stdio.printf "Epoch=%d, step=%d, lr=%f, epoch loss=%f\n%!" at_epoch at_step learning_rate
      epoch_loss
  in
  let module Backend = (val backend) in
  let inputs, outputs, _model_result, infer_callback, batch_losses, epoch_losses, learning_rates =
    Train.example_train_loop ~seed ~batch_size ~max_num_devices:(batch_size / 2) ~init_lr
      ~data_len:len ~epochs ~inputs:moons_flat ~outputs:moons_classes ~model:mlp ~loss_fn
      ~weight_decay ~per_batch_callback ~per_epoch_callback
      (module Backend)
      ()
  in
  [%expect
    {|
    Batch=19, step=20, lr=0.195250, batch loss=0.263769, epoch loss=45.768608
    Epoch=0, step=20, lr=0.195250, epoch loss=45.768608
    Batch=19, step=40, lr=0.190250, batch loss=0.210844, epoch loss=5.625710
    Epoch=1, step=40, lr=0.190250, epoch loss=5.625710
    Batch=19, step=60, lr=0.185250, batch loss=0.199376, epoch loss=5.396800
    Epoch=2, step=60, lr=0.185250, epoch loss=5.396800
    Batch=19, step=80, lr=0.180250, batch loss=0.193945, epoch loss=5.220046
    Epoch=3, step=80, lr=0.180250, epoch loss=5.220046
    Batch=19, step=100, lr=0.175250, batch loss=0.189124, epoch loss=5.124980
    Epoch=4, step=100, lr=0.175250, epoch loss=5.124980
    Batch=19, step=120, lr=0.170250, batch loss=0.190997, epoch loss=5.006840
    Epoch=5, step=120, lr=0.170250, epoch loss=5.006840
    Batch=19, step=140, lr=0.165250, batch loss=0.179420, epoch loss=4.851730
    Epoch=6, step=140, lr=0.165250, epoch loss=4.851730
    Batch=19, step=160, lr=0.160250, batch loss=0.166937, epoch loss=4.694808
    Epoch=7, step=160, lr=0.160250, epoch loss=4.694808
    Batch=19, step=180, lr=0.155250, batch loss=0.155812, epoch loss=4.430248
    Epoch=8, step=180, lr=0.155250, epoch loss=4.430248
    Batch=19, step=200, lr=0.150250, batch loss=0.139330, epoch loss=4.112350
    Epoch=9, step=200, lr=0.150250, epoch loss=4.112350
    Batch=19, step=220, lr=0.145250, batch loss=0.118694, epoch loss=3.616201
    Epoch=10, step=220, lr=0.145250, epoch loss=3.616201
    Batch=19, step=240, lr=0.140250, batch loss=0.087419, epoch loss=2.948867
    Epoch=11, step=240, lr=0.140250, epoch loss=2.948867
    Batch=19, step=260, lr=0.135250, batch loss=0.054353, epoch loss=2.079585
    Epoch=12, step=260, lr=0.135250, epoch loss=2.079585
    Batch=19, step=280, lr=0.130250, batch loss=0.036618, epoch loss=1.943864
    Epoch=13, step=280, lr=0.130250, epoch loss=1.943864
    Batch=19, step=300, lr=0.125250, batch loss=0.025849, epoch loss=0.977757
    Epoch=14, step=300, lr=0.125250, epoch loss=0.977757
    Batch=19, step=320, lr=0.120250, batch loss=0.009790, epoch loss=0.637346
    Epoch=15, step=320, lr=0.120250, epoch loss=0.637346
    Batch=19, step=340, lr=0.115250, batch loss=0.006237, epoch loss=0.458649
    Epoch=16, step=340, lr=0.115250, epoch loss=0.458649
    Batch=19, step=360, lr=0.110250, batch loss=0.005388, epoch loss=0.362362
    Epoch=17, step=360, lr=0.110250, epoch loss=0.362362
    Batch=19, step=380, lr=0.105250, batch loss=0.004444, epoch loss=0.256659
    Epoch=18, step=380, lr=0.105250, epoch loss=0.256659
    Batch=19, step=400, lr=0.100250, batch loss=0.004354, epoch loss=0.211964
    Epoch=19, step=400, lr=0.100250, epoch loss=0.211964
    |}];
  let points = Tensor.value_2d_points ~xdim:0 ~ydim:1 inputs in
  let classes = Tensor.value_1d_points ~xdim:0 outputs in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  let callback (x, y) = Float.((infer_callback [| x; y |]).(0) >= 0.) in
  let plot_moons =
    let open PrintBox_utils in
    plot ~size:(120, 40) ~x_label:"ixes" ~y_label:"ygreks"
      [
        Scatterplot { points = points1; pixel = "#" };
        Scatterplot { points = points2; pixel = "%" };
        Boundary_map { pixel_false = "."; pixel_true = "*"; callback };
      ]
  in
  Stdio.printf "\nHalf-moons scatterplot and decision boundary:\n%!";
  PrintBox_text.output Stdio.stdout plot_moons;
  [%expect
    {|
    Half-moons scatterplot and decision boundary:
     1.094e+0 │***************************************#********************************************************************************
              │***************************#*#*#########*###**######********************************************************************
              │***************************######*####*#*#####*########*#***************************************************************
              │*********************#**#########**#######*###############*###**********************************************************
              │******************####*####################################*###*********************************************************
              │***************#*#*###*###*###########*#*##*#####################*******************************************************
              │************#*######**#########*##*****************##*##*########*#*****************************************************
              │*************########*#*###*#**********************#******####*######***************************************************
              │**************#######*#*##******************************#########*##*##************************************************.
              │**********#######*###*#****************************************###**###*#******************************************.....
              │********#*######**##****************....**********************#*##*####*#***************************************........
              │********###*#*#**##*************............*******************###########*#*********************************...........
              │******########**************.........%....%.%...*******************##########*******************************..........%.
              │*******#######*************...........%..........*******************##*######*****************************.......%.%..%.
              │****##########************............%%%.%%%......*****************##########***************************......%..%%%%%.
              │*****######*#************............%%%.%...........**************#*#########*************************........%.%.%%..%
              │**######*#***************............%%%%%%%%.........*****************#*##*###**********************...........%%%%%%%.
    y         │**##*#####**************..............%%%%%%%...........**************#########*********************............%%.%%%..
    g         │**########*************..............%%%%%%%%.............**************##*######*****************..............%%%%%%%.
    r         │*########**************..............%%%.%%%.%%............**************#####*******************.............%%%%%%%%%.
    e         │*########*************................%%%%%%%%%..............************###*##*#**************................%%%%%%%..
    k         │##*######************.................%%%%%%%.%...............************######*#************................%%%%%%%%..
    s         │######*##************.................%%.%%%%%%.................**********########**********..................%%%%.%%.%.
              │###*##**#***********...................%.%%%%%%%%.................*********#####*#*********..................%%%%%%%%...
              │##*#####***********.....................%%%%%%.%.%.................*******#*#*####*******...................%%.%%%%%....
              │#####*##***********.....................%.%%%%%%%%...................*****##**##********..................%%%%%%%%%%%...
              │**#*##*#**********.......................%%%.%%%%%.%...................***#####*#*****.....................%%%%%%%......
              │##****##*********.........................%%.%%%%%%%%...................****###*##***...................%%%%%%%%%%......
              │*****************.........................%%.%%%%%%%......................*********.....................%..%%.%%%.......
              │****************............................%...%%%%%.%%....................******.................%.%%%%%%%%%%.........
              │***************...............................%.%%%%%.%%%%...................***...................%%%%%%%%.%.%%........
              │***************.................................%..%%%%%...%......................................%%%%%%%%%%............
              │**************....................................%%%.%%%%%%%%..............................%%..%%%%.%%%%%.%............
              │*************....................................%%%.%%%%%%.%%...%.........................%.%%%%%%%.%%%.%..............
              │*************........................................%.%%%.%%%%%%%%%...................%.%%%%%%%%%%%%%.%.%..............
              │************..........................................%.%%%%.%%%%%%%%%.%%%%%%%%%.%.%%%%%%%%%%%%%%%%%%%.%................
              │***********.............................................%%%%%%%%%%%%%%%%%%%%%.%%%%%%%.%%%.%%%%%%%%%%....................
              │***********.................................................%%%%%%%%%%%%%%%%%.%%%%%%%%%%%%%%%%%%%.......................
              │**********......................................................%%%%%%.%%%%%%%%%%%%%%%%%%%%%%%..........................
     -5.975e-1│*********..........................................................%....%%%%%.%%..%%%%...%..............................
    ──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
              │-1.098e+0                                                                                                       2.095e+0
              │                                                          ixes
    |}];
  Stdio.printf "\nBatch Loss:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"batch loss"
      [ Line_plot { points = Array.of_list_rev batch_losses; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nEpoch Loss:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"epoch loss"
      [ Line_plot { points = Array.of_list_rev epoch_losses; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nBatch Log-loss:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"batch log loss"
      [
        Line_plot
          {
            points =
              Array.of_list_rev_map batch_losses ~f:Float.(fun x -> max (log 0.00003) (log x));
            pixel = "-";
          };
      ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nEpoch Log-loss:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"epoch log loss"
      [ Line_plot { points = Array.of_list_rev_map epoch_losses ~f:Float.log; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  [%expect
    {|
    Batch Loss:
     2.405e+1│-
             │
             │
             │
             │
             │
             │
             │
             │
             │
    b        │
    a        │
    t        │
    c        │
    h        │
             │
    l        │
    o        │
    s        │
    s        │-
             │
             │
             │
             │
             │
             │
             │-
             │
             │---
     3.181e-4│ -----------------------------------------------------------------------------------------------------------------------
    ─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
             │0.000e+0                                                                                                        3.990e+2
             │                                                          step
    Epoch Loss:
     4.577e+1│-
             │
             │
             │
             │
             │
             │
             │
             │
             │
    e        │
    p        │
    o        │
    c        │
    h        │
             │
    l        │
    o        │
    s        │
    s        │
             │
             │
             │
             │
             │
             │
             │      -     -     -     -     -
             │                                    -     -     -     -     -
             │                                                                  -     -     -
     2.120e-1│                                                                                    -     -     -     -     -     -
    ─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
             │0.000e+0                                                                                                        1.900e+1
             │                                                          step
    Batch Log-loss:
     3.180e+0 │-
              │
              │
              │-
              │
              │
              │-
              │
    b         │ -
    a         │- -
    t         │ --
    c         │   -- -  -     -     -     -                                                     -
    h         │    ----- ------------- ----- ----- ----- -- -- -- -- -  --    --
              │      - - ----- ----- ----- ----- ----- ----- ----- ----- ------- -- --      -  -
    l         │                                                     --    ---- ----- ------  -  -     -
    o         │                                                                       ---- - ------   -     --          -
    g         │                                                                            --  - - -- ----  -  -  -
              │                                                                               -  -----  -- -    -  -    --
    l         │                                                                                     --  ----- --  -     -     --    -
    o         │                                                                                        -   - - ------- --    --    - -
    s         │                                                                                          -   -     - --  ------   ---
    s         │                                                                                           -   -  -       - -   ----  --
              │                                                                                                     -  -  --         -
              │                                                                                                -    -        -  --    -
              │                                                                                                      -     -       -
              │
              │                                                                                                                  -    -
              │                                                                                                           -
              │
     -8.053e+0│                                                                                                                 -
    ──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
              │0.000e+0                                                                                                        3.990e+2
              │                                                          step
    Epoch Log-loss:
     3.824e+0 │-
              │
              │
              │
              │
              │
              │
              │
    e         │
    p         │
    o         │
    c         │
    h         │      -     -     -     -     -
              │                                    -     -     -
    l         │                                                      -     -
    o         │                                                                  -
    g         │
              │                                                                        -
    l         │                                                                              -
    o         │
    s         │
    s         │                                                                                    -
              │
              │
              │                                                                                          -
              │                                                                                                -
              │
              │                                                                                                      -
              │                                                                                                            -
     -1.551e+0│                                                                                                                  -
    ──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
              │0.000e+0                                                                                                        1.900e+1
              │                                                          step
    |}];
  Stdio.printf "\nLearning rate:\n%!";
  let plot_lr =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"learning rate"
      [ Line_plot { points = Array.of_list_rev learning_rates; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_lr;
  [%expect
    {|
    Learning rate:
     1.953e-1│-
             │
             │      -
             │
             │            -
             │                  -
             │
             │                        -
    l        │                              -
    e        │
    a        │                                    -
    r        │                                          -
    n        │
    i        │                                                -
    n        │                                                      -
    g        │
             │                                                            -
    r        │                                                                  -
    a        │
    t        │                                                                        -
    e        │                                                                              -
             │
             │                                                                                    -
             │                                                                                          -
             │
             │                                                                                                -
             │                                                                                                      -
             │
             │                                                                                                            -
     1.002e-1│                                                                                                                  -
    ─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
             │0.000e+0                                                                                                        1.900e+1
             │                                                          step
    |}]

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
    Batch=59, step=60, lr=0.203235, batch loss=3.335014, epoch loss=36.081213
    Batch=119, step=120, lr=0.202215, batch loss=0.509602, epoch loss=39.752181
    Batch=179, step=180, lr=0.201195, batch loss=1.060667, epoch loss=42.616545
    Batch=239, step=240, lr=0.200175, batch loss=0.233840, epoch loss=44.186787
    Batch=299, step=300, lr=0.199155, batch loss=0.261280, epoch loss=45.340253
    Epoch=0, step=300, lr=0.199155, epoch loss=45.340253
    Batch=59, step=360, lr=0.198135, batch loss=0.353312, epoch loss=1.084888
    Batch=119, step=420, lr=0.197115, batch loss=0.277223, epoch loss=2.225298
    Batch=179, step=480, lr=0.196095, batch loss=0.356725, epoch loss=3.399804
    Batch=239, step=540, lr=0.195075, batch loss=0.225071, epoch loss=4.650029
    Batch=299, step=600, lr=0.194055, batch loss=0.212796, epoch loss=5.596965
    Epoch=1, step=600, lr=0.194055, epoch loss=5.596965
    Batch=59, step=660, lr=0.193035, batch loss=0.332231, epoch loss=1.008910
    Batch=119, step=720, lr=0.192015, batch loss=0.268184, epoch loss=2.118301
    Batch=179, step=780, lr=0.190995, batch loss=0.350068, epoch loss=3.258327
    Batch=239, step=840, lr=0.189975, batch loss=0.210938, epoch loss=4.452267
    Batch=299, step=900, lr=0.188955, batch loss=0.207439, epoch loss=5.374862
    Epoch=2, step=900, lr=0.188955, epoch loss=5.374862
    Batch=59, step=960, lr=0.187935, batch loss=0.318277, epoch loss=0.957002
    Batch=119, step=1020, lr=0.186915, batch loss=0.256429, epoch loss=1.980713
    Batch=179, step=1080, lr=0.185895, batch loss=0.328127, epoch loss=3.092157
    Batch=239, step=1140, lr=0.184875, batch loss=0.245747, epoch loss=4.291347
    Batch=299, step=1200, lr=0.183855, batch loss=0.193343, epoch loss=5.283991
    Epoch=3, step=1200, lr=0.183855, epoch loss=5.283991
    Batch=59, step=1260, lr=0.182835, batch loss=0.310547, epoch loss=0.932372
    Batch=119, step=1320, lr=0.181815, batch loss=0.242672, epoch loss=1.919124
    Batch=179, step=1380, lr=0.180795, batch loss=0.325418, epoch loss=2.994339
    Batch=239, step=1440, lr=0.179775, batch loss=0.213394, epoch loss=4.131793
    Batch=299, step=1500, lr=0.178755, batch loss=0.188089, epoch loss=5.062602
    Epoch=4, step=1500, lr=0.178755, epoch loss=5.062602
    Batch=59, step=1560, lr=0.177735, batch loss=0.304712, epoch loss=0.931078
    Batch=119, step=1620, lr=0.176715, batch loss=0.236301, epoch loss=1.905623
    Batch=179, step=1680, lr=0.175695, batch loss=0.315893, epoch loss=2.952261
    Batch=239, step=1740, lr=0.174675, batch loss=0.204929, epoch loss=4.065377
    Batch=299, step=1800, lr=0.173655, batch loss=0.190588, epoch loss=4.981880
    Epoch=5, step=1800, lr=0.173655, epoch loss=4.981880
    Batch=59, step=1860, lr=0.172635, batch loss=0.293859, epoch loss=0.904283
    Batch=119, step=1920, lr=0.171615, batch loss=0.226019, epoch loss=1.847752
    Batch=179, step=1980, lr=0.170595, batch loss=0.308259, epoch loss=2.867408
    Batch=239, step=2040, lr=0.169575, batch loss=0.195349, epoch loss=3.937485
    Batch=299, step=2100, lr=0.168555, batch loss=0.177697, epoch loss=4.803518
    Epoch=6, step=2100, lr=0.168555, epoch loss=4.803518
    Batch=59, step=2160, lr=0.167535, batch loss=0.277328, epoch loss=0.853072
    Batch=119, step=2220, lr=0.166515, batch loss=0.224874, epoch loss=1.768610
    Batch=179, step=2280, lr=0.165495, batch loss=0.298105, epoch loss=2.773350
    Batch=239, step=2340, lr=0.164475, batch loss=0.198649, epoch loss=3.825650
    Batch=299, step=2400, lr=0.163455, batch loss=0.173268, epoch loss=4.701841
    Epoch=7, step=2400, lr=0.163455, epoch loss=4.701841
    Batch=59, step=2460, lr=0.162435, batch loss=0.268239, epoch loss=0.818936
    Batch=119, step=2520, lr=0.161415, batch loss=0.218898, epoch loss=1.714754
    Batch=179, step=2580, lr=0.160395, batch loss=0.286570, epoch loss=2.669310
    Batch=239, step=2640, lr=0.159375, batch loss=0.191241, epoch loss=3.677503
    Batch=299, step=2700, lr=0.158355, batch loss=0.165198, epoch loss=4.509585
    Epoch=8, step=2700, lr=0.158355, epoch loss=4.509585
    Batch=59, step=2760, lr=0.157335, batch loss=0.252190, epoch loss=0.774354
    Batch=119, step=2820, lr=0.156315, batch loss=0.204453, epoch loss=1.616114
    Batch=179, step=2880, lr=0.155295, batch loss=0.272133, epoch loss=2.514288
    Batch=239, step=2940, lr=0.154275, batch loss=0.193256, epoch loss=3.492436
    Batch=299, step=3000, lr=0.153255, batch loss=0.153980, epoch loss=4.250107
    Epoch=9, step=3000, lr=0.153255, epoch loss=4.250107
    Batch=59, step=3060, lr=0.152235, batch loss=0.235917, epoch loss=0.735425
    Batch=119, step=3120, lr=0.151215, batch loss=0.197940, epoch loss=1.525605
    Batch=179, step=3180, lr=0.150195, batch loss=0.251685, epoch loss=2.352272
    Batch=239, step=3240, lr=0.149175, batch loss=0.176343, epoch loss=3.246319
    Batch=299, step=3300, lr=0.148155, batch loss=0.142091, epoch loss=3.958504
    Epoch=10, step=3300, lr=0.148155, epoch loss=3.958504
    Batch=59, step=3360, lr=0.147135, batch loss=0.212882, epoch loss=0.666686
    Batch=119, step=3420, lr=0.146115, batch loss=0.164095, epoch loss=1.355471
    Batch=179, step=3480, lr=0.145095, batch loss=0.204504, epoch loss=2.053549
    Batch=239, step=3540, lr=0.144075, batch loss=0.162518, epoch loss=2.842626
    Batch=299, step=3600, lr=0.143055, batch loss=0.124966, epoch loss=3.543600
    Epoch=11, step=3600, lr=0.143055, epoch loss=3.543600
    Batch=59, step=3660, lr=0.142035, batch loss=0.178272, epoch loss=0.559618
    Batch=119, step=3720, lr=0.141015, batch loss=0.138280, epoch loss=1.110549
    Batch=179, step=3780, lr=0.139995, batch loss=0.162534, epoch loss=1.672241
    Batch=239, step=3840, lr=0.138975, batch loss=0.096877, epoch loss=2.324762
    Batch=299, step=3900, lr=0.137955, batch loss=0.088672, epoch loss=2.802225
    Epoch=12, step=3900, lr=0.137955, epoch loss=2.802225
    Batch=59, step=3960, lr=0.136935, batch loss=0.131200, epoch loss=0.488680
    Batch=119, step=4020, lr=0.135915, batch loss=0.098591, epoch loss=0.865268
    Batch=179, step=4080, lr=0.134895, batch loss=0.113244, epoch loss=1.286682
    Batch=239, step=4140, lr=0.133875, batch loss=0.048121, epoch loss=1.870983
    Batch=299, step=4200, lr=0.132855, batch loss=0.069599, epoch loss=2.239889
    Epoch=13, step=4200, lr=0.132855, epoch loss=2.239889
    Batch=59, step=4260, lr=0.131835, batch loss=0.088278, epoch loss=0.297689
    Batch=119, step=4320, lr=0.130815, batch loss=0.048423, epoch loss=0.561479
    Batch=179, step=4380, lr=0.129795, batch loss=0.095387, epoch loss=0.847892
    Batch=239, step=4440, lr=0.128775, batch loss=0.029962, epoch loss=1.183814
    Batch=299, step=4500, lr=0.127755, batch loss=0.034442, epoch loss=1.403857
    Epoch=14, step=4500, lr=0.127755, epoch loss=1.403857
    Batch=59, step=4560, lr=0.126735, batch loss=0.051376, epoch loss=0.153330
    Batch=119, step=4620, lr=0.125715, batch loss=0.025430, epoch loss=0.304768
    Batch=179, step=4680, lr=0.124695, batch loss=0.047700, epoch loss=0.466758
    Batch=239, step=4740, lr=0.123675, batch loss=0.020897, epoch loss=0.775139
    Batch=299, step=4800, lr=0.122655, batch loss=0.020777, epoch loss=0.915513
    Epoch=15, step=4800, lr=0.122655, epoch loss=0.915513
    Batch=59, step=4860, lr=0.121635, batch loss=0.033293, epoch loss=0.098308
    Batch=119, step=4920, lr=0.120615, batch loss=0.018037, epoch loss=0.178296
    Batch=179, step=4980, lr=0.119595, batch loss=0.040397, epoch loss=0.297512
    Batch=239, step=5040, lr=0.118575, batch loss=0.013933, epoch loss=0.492889
    Batch=299, step=5100, lr=0.117555, batch loss=0.009512, epoch loss=0.562148
    Epoch=16, step=5100, lr=0.117555, epoch loss=0.562148
    Batch=59, step=5160, lr=0.116535, batch loss=0.019371, epoch loss=0.070393
    Batch=119, step=5220, lr=0.115515, batch loss=0.007765, epoch loss=0.139673
    Batch=179, step=5280, lr=0.114495, batch loss=0.024870, epoch loss=0.234491
    Batch=239, step=5340, lr=0.113475, batch loss=0.013419, epoch loss=0.348124
    Batch=299, step=5400, lr=0.112455, batch loss=0.005786, epoch loss=0.386028
    Epoch=17, step=5400, lr=0.112455, epoch loss=0.386028
    Batch=59, step=5460, lr=0.111435, batch loss=0.011737, epoch loss=0.038071
    Batch=119, step=5520, lr=0.110415, batch loss=0.004301, epoch loss=0.070729
    Batch=179, step=5580, lr=0.109395, batch loss=0.026152, epoch loss=0.157258
    Batch=239, step=5640, lr=0.108375, batch loss=0.010461, epoch loss=0.258442
    Batch=299, step=5700, lr=0.107355, batch loss=0.004833, epoch loss=0.287873
    Epoch=18, step=5700, lr=0.107355, epoch loss=0.287873
    Batch=59, step=5760, lr=0.106335, batch loss=0.008844, epoch loss=0.029072
    Batch=119, step=5820, lr=0.105315, batch loss=0.002439, epoch loss=0.062771
    Batch=179, step=5880, lr=0.104295, batch loss=0.014950, epoch loss=0.122992
    Batch=239, step=5940, lr=0.103275, batch loss=0.010948, epoch loss=0.202330
    Batch=299, step=6000, lr=0.102255, batch loss=0.005021, epoch loss=0.229280
    Epoch=19, step=6000, lr=0.102255, epoch loss=0.229280
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
              │**************#######*#*##******************************#########*##*##***********************************************..
              │**********#######*###*#****************************************###**###*#******************************************.....
              │********#*######**##******************************************#*##*####*#**************************************.........
              │********###*#*#**##**************...........*******************###########*#********************************............
              │******########***************........%....%.%...*******************##########*****************************............%.
              │*******#######*************...........%...........******************##*######***************************.........%.%..%.
              │****##########*************...........%%%.%%%......*****************##########*************************........%..%%%%%.
              │*****######*#*************...........%%%.%...........**************#*#########************************.........%.%.%%..%
              │**######*#***************............%%%%%%%%..........****************#*##*###*********************............%%%%%%%.
    y         │**##*#####**************..............%%%%%%%...........**************#########********************.............%%.%%%..
    g         │**########**************.............%%%%%%%%.............**************##*######****************...............%%%%%%%.
    r         │*########**************..............%%%.%%%.%%.............*************#####******************..............%%%%%%%%%.
    e         │*########*************................%%%%%%%%%..............************###*##*#*************.................%%%%%%%..
    k         │##*######************.................%%%%%%%.%................***********######*#***********.................%%%%%%%%..
    s         │######*##************.................%%.%%%%%%.................**********########**********..................%%%%.%%.%.
              │###*##**#***********...................%.%%%%%%%%.................*********#####*#********...................%%%%%%%%...
              │##*#####***********.....................%%%%%%.%.%..................******#*#*####*******...................%%.%%%%%....
              │#####*##**********......................%.%%%%%%%%...................*****##**##*******...................%%%%%%%%%%%...
              │**#*##*#**********.......................%%%.%%%%%.%...................***#####*#*****.....................%%%%%%%......
              │##****##*********.........................%%.%%%%%%%%....................***###*##***...................%%%%%%%%%%......
              │****************..........................%%.%%%%%%%......................*********.....................%..%%.%%%.......
              │***************.............................%...%%%%%.%%....................*****..................%.%%%%%%%%%%.........
              │***************...............................%.%%%%%.%%%%...................***...................%%%%%%%%.%.%%........
              │**************..................................%..%%%%%...%......................................%%%%%%%%%%............
              │*************.....................................%%%.%%%%%%%%..............................%%..%%%%.%%%%%.%............
              │************.....................................%%%.%%%%%%.%%...%.........................%.%%%%%%%.%%%.%..............
              │************.........................................%.%%%.%%%%%%%%%...................%.%%%%%%%%%%%%%.%.%..............
              │***********...........................................%.%%%%.%%%%%%%%%.%%%%%%%%%.%.%%%%%%%%%%%%%%%%%%%.%................
              │**********..............................................%%%%%%%%%%%%%%%%%%%%%.%%%%%%%.%%%.%%%%%%%%%%....................
              │*********...................................................%%%%%%%%%%%%%%%%%.%%%%%%%%%%%%%%%%%%%.......................
              │*********.......................................................%%%%%%.%%%%%%%%%%%%%%%%%%%%%%%..........................
     -5.975e-1│********...........................................................%....%%%%%.%%..%%%%...%..............................
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
     2.074e+1│-
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
    l        │-
    o        │
    s        │
    s        │
             │
             │
             │
             │
             │
             │-
             │-
             │
             │ - -
     1.031e-3│  ----------------------------------------------------------------------------------------------------------------------
    ─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
             │0.000e+0                                                                                                        3.990e+2
             │                                                          step
    Epoch Loss:
     4.534e+1│-
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
             │                                    -     -     -     -     -     -
             │                                                                        -     -
     2.293e-1│                                                                                    -     -     -     -     -     -
    ─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
             │0.000e+0                                                                                                        1.900e+1
             │                                                          step
    Batch Log-loss:
     3.032e+0 │
              │-
              │
              │-
              │
              │
              │-
              │-
    b         │
    a         │ - -
    t         │ - -
    c         │  --
    h         │   -- -  -     -     --                                               -
              │     ---- - ----- -------- -- -- -- -- -- -- -- -- -- -  --    --                 -
    l         │    ----- ----- -----  ---- ----- ----- ----- ----- ----- ------- -- -- -  --
    o         │                                               --    -- -  ----  ---- ------ --  - -   -     -
    g         │                                                                 -     ---- ------- -  -
              │                                                                             - --  ------- - -     -
    l         │                                                                                  - ---  --  ---   --
    o         │                                                                                         ----   -  -     -     --
    s         │                                                                                        - ---  ------------   --     --
    s         │                                                                                              --- -   -- -    -     --
              │                                                                                                -   - --  ---- - -----
              │                                                                                                 -   -          - -   --
              │                                                                                                      - -  --     --
              │                                                                                                              -  -     -
              │                                                                                                           --
              │                                                                                                             -    - -  -
              │
     -6.877e+0│                                                                                                                 -
    ──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
              │0.000e+0                                                                                                        3.990e+2
              │                                                          step
    Epoch Log-loss:
     3.814e+0 │-
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
    h         │      -     -     -
              │                        -     -     -     -     -     -
    l         │                                                            -     -
    o         │
    g         │                                                                        -
              │                                                                              -
    l         │
    o         │
    s         │                                                                                    -
    s         │
              │                                                                                          -
              │
              │
              │                                                                                                -
              │
              │                                                                                                      -
              │                                                                                                            -
     -1.473e+0│                                                                                                                  -
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
     1.992e-1│-
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
     1.023e-1│                                                                                                                  -
    ─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
             │0.000e+0                                                                                                        1.900e+1
             │                                                          step
    |}]

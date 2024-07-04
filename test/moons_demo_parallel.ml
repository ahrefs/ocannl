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
  let backend = Train.fresh_backend () in
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
  let inputs, outputs, _model_result, infer_callback, batch_losses, epoch_losses, learning_rates =
    Train.example_train_loop ~seed ~batch_size ~max_num_devices:(batch_size / 2) ~init_lr
      ~data_len:len ~epochs ~inputs:moons_flat ~outputs:moons_classes ~model:mlp ~loss_fn
      ~weight_decay ~per_batch_callback ~per_epoch_callback backend ()
  in
  [%expect {|
    Batch=459, step=460, lr=0.187630, batch loss=0.302324, epoch loss=44.965767
    Batch=479, step=480, lr=0.187400, batch loss=0.242093, epoch loss=45.207860
    Epoch=0, step=480, lr=0.187400, epoch loss=45.207860
    Batch=459, step=940, lr=0.182830, batch loss=0.261239, epoch loss=5.464439
    Batch=479, step=960, lr=0.182600, batch loss=0.203043, epoch loss=5.667482
    Epoch=1, step=960, lr=0.182600, epoch loss=5.667482
    Batch=459, step=1420, lr=0.178030, batch loss=0.244262, epoch loss=5.197238
    Batch=479, step=1440, lr=0.177800, batch loss=0.211025, epoch loss=5.408263
    Epoch=2, step=1440, lr=0.177800, epoch loss=5.408263
    Batch=459, step=1900, lr=0.173230, batch loss=0.249422, epoch loss=5.030923
    Batch=479, step=1920, lr=0.173000, batch loss=0.180412, epoch loss=5.211335
    Epoch=3, step=1920, lr=0.173000, epoch loss=5.211335
    Batch=459, step=2380, lr=0.168430, batch loss=0.232375, epoch loss=4.936376
    Batch=479, step=2400, lr=0.168200, batch loss=0.180837, epoch loss=5.117212
    Epoch=4, step=2400, lr=0.168200, epoch loss=5.117212
    Batch=459, step=2860, lr=0.163630, batch loss=0.230121, epoch loss=4.853825
    Batch=479, step=2880, lr=0.163400, batch loss=0.171041, epoch loss=5.024865
    Epoch=5, step=2880, lr=0.163400, epoch loss=5.024865
    Batch=459, step=3340, lr=0.158830, batch loss=0.225951, epoch loss=4.747527
    Batch=479, step=3360, lr=0.158600, batch loss=0.193549, epoch loss=4.941076
    Epoch=6, step=3360, lr=0.158600, epoch loss=4.941076
    Batch=459, step=3820, lr=0.154030, batch loss=0.219958, epoch loss=4.613691
    Batch=479, step=3840, lr=0.153800, batch loss=0.162122, epoch loss=4.775813
    Epoch=7, step=3840, lr=0.153800, epoch loss=4.775813
    Batch=459, step=4300, lr=0.149230, batch loss=0.212796, epoch loss=4.386305
    Batch=479, step=4320, lr=0.149000, batch loss=0.155903, epoch loss=4.542209
    Epoch=8, step=4320, lr=0.149000, epoch loss=4.542209
    Batch=459, step=4780, lr=0.144430, batch loss=0.196582, epoch loss=4.205096
    Batch=479, step=4800, lr=0.144200, batch loss=0.146156, epoch loss=4.351251
    Epoch=9, step=4800, lr=0.144200, epoch loss=4.351251
    Batch=459, step=5260, lr=0.139630, batch loss=0.170325, epoch loss=3.775596
    Batch=479, step=5280, lr=0.139400, batch loss=0.131116, epoch loss=3.906712
    Epoch=10, step=5280, lr=0.139400, epoch loss=3.906712
    Batch=459, step=5740, lr=0.134830, batch loss=0.155171, epoch loss=3.374118
    Batch=479, step=5760, lr=0.134600, batch loss=0.114241, epoch loss=3.488359
    Epoch=11, step=5760, lr=0.134600, epoch loss=3.488359
    Batch=459, step=6220, lr=0.130030, batch loss=0.140047, epoch loss=3.087934
    Batch=479, step=6240, lr=0.129800, batch loss=0.088562, epoch loss=3.176496
    Epoch=12, step=6240, lr=0.129800, epoch loss=3.176496
    Batch=459, step=6700, lr=0.125230, batch loss=0.104433, epoch loss=2.675076
    Batch=479, step=6720, lr=0.125000, batch loss=0.085433, epoch loss=2.760509
    Epoch=13, step=6720, lr=0.125000, epoch loss=2.760509
    Batch=459, step=7180, lr=0.120430, batch loss=0.113704, epoch loss=1.785049
    Batch=479, step=7200, lr=0.120200, batch loss=0.050677, epoch loss=1.835725
    Epoch=14, step=7200, lr=0.120200, epoch loss=1.835725
    Batch=459, step=7660, lr=0.115630, batch loss=0.056435, epoch loss=1.233646
    Batch=479, step=7680, lr=0.115400, batch loss=0.023459, epoch loss=1.257105
    Epoch=15, step=7680, lr=0.115400, epoch loss=1.257105
    Batch=459, step=8140, lr=0.110830, batch loss=0.027698, epoch loss=0.695773
    Batch=479, step=8160, lr=0.110600, batch loss=0.012053, epoch loss=0.707827
    Epoch=16, step=8160, lr=0.110600, epoch loss=0.707827
    Batch=459, step=8620, lr=0.106030, batch loss=0.024702, epoch loss=0.471656
    Batch=479, step=8640, lr=0.105800, batch loss=0.009089, epoch loss=0.480745
    Epoch=17, step=8640, lr=0.105800, epoch loss=0.480745
    Batch=459, step=9100, lr=0.101230, batch loss=0.016628, epoch loss=0.356585
    Batch=479, step=9120, lr=0.101000, batch loss=0.007834, epoch loss=0.364419
    Epoch=18, step=9120, lr=0.101000, epoch loss=0.364419
    Batch=459, step=9580, lr=0.096430, batch loss=0.012248, epoch loss=0.270942
    Batch=479, step=9600, lr=0.096200, batch loss=0.005346, epoch loss=0.276288
    Epoch=19, step=9600, lr=0.096200, epoch loss=0.276288
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
  [%expect {|
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
              │**********#######*###*#****************************************###**###*#*******************************************....
              │********#*######**##******************************************#*##*####*#***************************************........
              │********###*#*#**##**************...........*******************###########*#**********************************..........
              │******########****************.......%....%.%...*******************##########*******************************..........%.
              │*******#######**************..........%...........******************##*######*****************************.......%.%..%.
              │****##########*************...........%%%.%%%.......****************##########***************************......%..%%%%%.
              │*****######*#*************...........%%%.%...........**************#*#########*************************........%.%.%%..%
              │**######*#***************............%%%%%%%%..........****************#*##*###**********************...........%%%%%%%.
    y         │**##*#####***************.............%%%%%%%............*************#########*********************............%%.%%%..
    g         │**########**************.............%%%%%%%%.............**************##*######*****************..............%%%%%%%.
    r         │*########**************..............%%%.%%%.%%.............*************#####*******************.............%%%%%%%%%.
    e         │*########**************...............%%%%%%%%%...............***********###*##*#**************................%%%%%%%..
    k         │##*######*************................%%%%%%%.%................***********######*#***********.................%%%%%%%%..
    s         │######*##************.................%%.%%%%%%..................*********########**********..................%%%%.%%.%.
              │###*##**#************..................%.%%%%%%%%..................********#####*#********...................%%%%%%%%...
              │##*#####************....................%%%%%%.%.%..................******#*#*####*******...................%%.%%%%%....
              │#####*##***********.....................%.%%%%%%%%....................****##**##*******...................%%%%%%%%%%%...
              │**#*##*#***********......................%%%.%%%%%.%....................**#####*#****......................%%%%%%%......
              │##****##**********........................%%.%%%%%%%%....................***###*##**....................%%%%%%%%%%......
              │*****************.........................%%.%%%%%%%.......................*******......................%..%%.%%%.......
              │*****************...........................%...%%%%%.%%.....................****..................%.%%%%%%%%%%.........
              │****************..............................%.%%%%%.%%%%....................*....................%%%%%%%%.%.%%........
              │***************.................................%..%%%%%...%......................................%%%%%%%%%%............
              │***************...................................%%%.%%%%%%%%..............................%%..%%%%.%%%%%.%............
              │**************...................................%%%.%%%%%%.%%...%.........................%.%%%%%%%.%%%.%..............
              │*************........................................%.%%%.%%%%%%%%%...................%.%%%%%%%%%%%%%.%.%..............
              │*************.........................................%.%%%%.%%%%%%%%%.%%%%%%%%%.%.%%%%%%%%%%%%%%%%%%%.%................
              │************............................................%%%%%%%%%%%%%%%%%%%%%.%%%%%%%.%%%.%%%%%%%%%%....................
              │***********.................................................%%%%%%%%%%%%%%%%%.%%%%%%%%%%%%%%%%%%%.......................
              │***********.....................................................%%%%%%.%%%%%%%%%%%%%%%%%%%%%%%..........................
     -5.975e-1│**********.........................................................%....%%%%%.%%..%%%%...%..............................
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
  [%expect {|
    Batch Loss:
     2.165e+1│-
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
             │-
             │
             │
             │-
             │ --
     9.009e-4│  ----------------------------------------------------------------------------------------------------------------------
    ─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
             │0.000e+0                                                                                                        4.190e+2
             │                                                          step
    Epoch Loss:
     4.521e+1│-
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
             │      -     -     -     -     -     -
             │                                          -     -     -     -     -
             │                                                                        -     -     -
     2.763e-1│                                                                                          -     -     -     -     -
    ─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
             │0.000e+0                                                                                                        1.900e+1
             │                                                          step
    Batch Log-loss:
     3.075e+0 │-
              │
              │
              │-
              │
              │-
              │
              │
    b         │--
    a         │ --
    t         │ --
    c         │  --
    h         │   -- - --- -  -  -  -     -     -                                                -
              │    -------------- ---------- ----- ----- ----- ----- ----- - ---        -  -
    l         │    ----  ----- ----- ----- ---   ---   --- - --- - --- --------------- -- --  -  -    -     -
    o         │        -     -  -  - --  -  ----  ---- - --- ----- ----- ----  --- ----- --------- - --
    g         │                                                              -  - -- -- -   -- -- ---- -- -- -    --    -
              │                                                                          - -- -    -- -- -  --
    l         │                                                                                -     -  ---- ---         -     -
    o         │                                                                                     -    - --- -----  -  - -         -
    s         │                                                                                        -  -   ---- --- -- ------ - ---
    s         │                                                                                               ----     - -   - -----
              │                                                                                                     --- -  -- -     ---
              │                                                                                                           -   - ---
              │                                                                                                    - - - --- -      - -
              │                                                                                                       -        ----
              │
              │                                                                                                                    - -
              │
     -7.012e+0│                                                                                                                       -
    ──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
              │0.000e+0                                                                                                        4.190e+2
              │                                                          step
    Epoch Log-loss:
     3.811e+0 │
              │-
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
    h         │      -
              │            -     -     -     -     -     -
    l         │                                                -     -     -
    o         │                                                                  -
    g         │                                                                        -     -
              │
    l         │
    o         │                                                                                    -
    s         │
    s         │                                                                                          -
              │
              │
              │                                                                                                -
              │
              │                                                                                                      -
              │
              │                                                                                                            -
     -1.286e+0│                                                                                                                  -
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
  [%expect {|
    Learning rate:
     1.874e-1│-
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
     9.620e-2│                                                                                                                  -
    ─────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
             │0.000e+0                                                                                                        1.900e+1
             │                                                          step
    |}]

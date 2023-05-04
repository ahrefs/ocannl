open Base
open Core_bench
open Ocannl
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL
module CDSL = Code.CDSL

let _suspended () =
  (* Code.CDSL.with_debug := false; *)
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  set_executor Interpreter;
  enable_all_debugs ();
  CDSL.virtualize_settings.virtualize <- true;
  drop_all_sessions ();
  Random.init 0;
  let len = 400 in
  let minibatch = 2 in
  let n_batches = 2 * len / minibatch in
  let epochs = 100 in
  let steps = epochs * n_batches in
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
  let moons_flat = FDSL.init_const ~l:"moons_flat" ~b:[ n_batches; minibatch ] ~o:[ 2 ] moons_flat in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes = FDSL.init_const ~l:"moons_classes" ~b:[ n_batches; minibatch ] ~o:[ 1 ] moons_classes in
  let%nn_op mlp x = "b2" 1 + ("w2" * !/("b1" 2 + ("w1" * x))) in
  (* let%nn_op mlp x =
       "b6" 1
       + "w6"
         * !/("b4" 4
             + "w4"
               * !/("b2" 8
                   + ("w2" * !/("b1" 16 + ("w1" * x)))
                   + "b3" 8
                   + ("w3" * !/(b2 + (w2 * !/(b1 + (w1 * x))))))
             + ("b5" 4 + ("w5" * !/(b4 + (w4 * !/(b3 + (w3 * !/(b2 + (w2 * !/(b1 + (w1 * x)))))))))))
     in *)
  (* let%nn_op mlp x =
       "b6" 1
       + "w6"
         * !/("b5" 4
             + ("w5" * !/("b4" 4 + ("w4" * !/("b3" 8 + ("w3" * !/("b2" 8 + ("w2" * !/("b1" 16 + ("w1" * x)))))))))
             )
     in *)
  let%nn_dt session_step ~output_dims:[ 1 ] = n =+ 1 in
  let%nn_dt minus_lr ~output_dims:[ 1 ] = n =: -0.0001 *. (!..steps - session_step) /. !..steps in
  minus_learning_rate := Some minus_lr;
  let%nn_op moons_input = moons_flat @.| session_step in
  let%nn_op moons_class = moons_classes @.| session_step in
  let%nn_op margin_loss = !/(1 - (moons_class *. mlp moons_input)) in
  let%nn_op ssq w = (w **. 2) ++ "...|...->... => 0" in
  let reg_loss = List.map ~f:ssq [ w1; w2; b1; b2 ] |> List.reduce_exn ~f:FDSL.O.( + ) in
  (* let reg_loss =
       List.map ~f:ssq [ w1; w2; w3; w4; w5; w6; b1; b2; b3; b4; b5; b6 ] |> List.reduce_exn ~f:FDSL.O.( + )
     in *)
  let%nn_op total_loss = ((margin_loss ++ "...|... => 0") /. !..minibatch) + (0.001 *. reg_loss) in
  refresh_session ();
  print_decimals_precision := 9;
  Stdio.print_endline "\nPreamble:\n";
  print_preamble ();
  Stdio.printf "\nStep 1: loss %f\n%!" total_loss.@[0];
  print_node_tree ~with_id:true ~with_grad:true ~depth:9 total_loss.id;
  List.iter [ w1; w2; b1; b2 ] ~f:(fun f ->
      Stdio.print_endline "\n";
      print_formula ~with_grad:true ~with_code:false `Default f);
  refresh_session ();
  Stdio.printf "\nStep 2: loss %f\n%!" total_loss.@[0];
  print_node_tree ~with_id:true ~with_grad:true ~depth:9 total_loss.id;
  List.iter [ w1; w2; b1; b2 ] ~f:(fun f ->
      Stdio.print_endline "\n";
      print_formula ~with_grad:true ~with_code:false `Default f);
  refresh_session ();
  Stdio.printf "\nStep 3: loss %f\n%!" total_loss.@[0];
  print_node_tree ~with_id:true ~with_grad:true ~depth:9 total_loss.id;
  refresh_session ();
  Stdio.printf "\nStep 4: loss %f\n%!" total_loss.@[0];
  print_node_tree ~with_id:true ~with_grad:true ~depth:9 total_loss.id

let classify_moons ~virtualize executor ~opti_level precision () =
  let epochs = 20000 in
  (* let epochs = 200 in *)
  Stdio.prerr_endline @@ "\n\n****** Benchmarking virtualized: "
  ^ Sexp.to_string_hum
      ([%sexp_of: bool * Session.backend * int * NodeUI.prec] (virtualize, executor, opti_level, precision))
  ^ " for " ^ Int.to_string epochs ^ " epochs ******";
  CDSL.debug_virtual_nodes := false;
  CDSL.virtualize_settings.virtualize <- virtualize;
  Session.SDSL.set_executor executor;
  Session.SDSL.default_value_prec := precision;
  Session.SDSL.default_grad_prec := precision;
  Exec_as_gccjit.optimization_level := opti_level;
  let open Session.SDSL in
  disable_all_debugs ();
  drop_all_sessions ();
  Random.init 0;
  (* let init_mem = Mem_usage.info () in *)
  (* let hid1 = 64 in *)
  let len = 400 in
  let minibatch = 10 in
  let n_batches = 2 * len / minibatch in
  let steps = epochs * n_batches in
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
  let moons_flat = FDSL.init_const ~l:"moons_flat" ~b:[ n_batches; minibatch ] ~o:[ 2 ] moons_flat in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes = FDSL.init_const ~l:"moons_classes" ~b:[ n_batches; minibatch ] ~o:[ 1 ] moons_classes in
  let%nn_op mlp x =
    "b6" 1
    + "w6"
      * !/("b4" 4
          + "w4"
            * !/("b2" 8
                + ("w2" * !/("b1" 16 + ("w1" * x)))
                + "b3" 8
                + ("w3" * !/(b2 + (w2 * !/(b1 + (w1 * x))))))
          + ("b5" 4 + ("w5" * !/(b4 + (w4 * !/(b3 + (w3 * !/(b2 + (w2 * !/(b1 + (w1 * x)))))))))))
  in
  (* let%nn_op mlp x =
       "b6" 1
       + "w6"
         * !/("b5" 4
             + ("w5" * !/("b4" 4 + ("w4" * !/("b3" 8 + ("w3" * !/("b2" 8 + ("w2" * !/("b1" 16 + ("w1" * x)))))))))
             )
     in *)
  (* let%nn_op mlp x = "b2" 1 + ("w2" * !/("b1" 16 + ("w1" * x))) in *)
  let%nn_dt session_step ~output_dims:[ 1 ] = n =+ 1 in
  let%nn_dt minus_lr ~output_dims:[ 1 ] = n =: -0.1 *. (!..steps - session_step) /. !..steps in
  minus_learning_rate := Some minus_lr;
  let%nn_op moons_input = moons_flat @.| session_step in
  let%nn_op moons_class = moons_classes @.| session_step in
  let points1 = ref [] in
  let points2 = ref [] in
  let minibatch_loglosses = ref [] in
  let min_loss = ref Float.infinity in
  let max_loss = ref 0.0 in
  let epoch_loss = ref 0.0 in
  let losses = ref [] in
  let log_losses = ref [] in
  let learning_rates = ref [] in
  let stop = ref false in
  let step = ref 1 in
  let%nn_op margin_loss = !/(1 - (moons_class *. mlp moons_input)) in
  let%nn_op ssq w = (w **. 2) ++ "...|...->... => 0" in
  let reg_loss =
    List.map ~f:ssq [ w1; w2; w3; w4; w5; w6; b1; b2; b3; b4; b5; b6 ] |> List.reduce_exn ~f:FDSL.O.( + )
  in
  (* let reg_loss = List.map ~f:ssq [ w1; w2; b1; b2 ] |> List.reduce_exn ~f:FDSL.O.( + ) in *)
  let%nn_op total_loss = ((margin_loss ++ "...|... => 0") /. !..minibatch) + (0.0001 *. reg_loss) in
  while not !stop do
    refresh_session ();
    (* if step = 1 then (
       print_node_tree ~with_id:true ~with_grad:true ~depth:9 total_loss.id;
       Stdio.printf "\n%!"); *)
    if !step <= n_batches then (
      let points = value_2d_points ~xdim:0 ~ydim:1 moons_input in
      let classes = value_1d_points ~xdim:0 moons_class in
      let npoints1, npoints2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
      points1 := npoints1 :: !points1;
      points2 := npoints2 :: !points2);
    let loss = total_loss.@[0] in
    epoch_loss := !epoch_loss +. loss;
    if Float.(loss > 10.0 * !min_loss && loss > (!min_loss + !max_loss) / 2.0) then stop := true;
    if Float.(loss < !min_loss) then min_loss := loss;
    if Float.(loss > !max_loss) then max_loss := loss;
    if !step % n_batches = 0 || !stop then (
      learning_rates := ~-.(minus_lr.@[0]) :: !learning_rates;
      minibatch_loglosses := Float.log loss :: !losses;
      losses := !epoch_loss :: !losses;
      log_losses := Float.log !epoch_loss :: !log_losses;
      if !step / n_batches % (epochs / 10) = 0 || !stop then (
        Stdio.printf "Minus learning rate over minibatch for step %d of %d: %f\n%!" !step steps minus_lr.@[0];
        Stdio.printf "Loss over minibatch for step %d: %f; epoch loss: %f; min loss: %f; max loss: %f\n%!"
          !step loss !epoch_loss !min_loss !max_loss));
    if !step % n_batches = 0 then epoch_loss := 0.0;
    Int.incr step;
    if !step > steps then stop := true
  done;
  (* let train_mem = Mem_usage.info () in *)
  close_session ();
  let%nn_op point = [ 0; 0 ] in
  let mlp_result = mlp point in
  refresh_session ();
  let callback (x, y) =
    set_values point [| x; y |];
    refresh_session ();
    Float.(mlp_result.@[0] >= 0.)
  in
  let plot_moons =
    let open PrintBox_utils in
    plot ~size:(120, 40) ~x_label:"ixes" ~y_label:"ygreks"
      [
        Scatterplot { points = Array.concat !points1; pixel = "#" };
        Scatterplot { points = Array.concat !points2; pixel = "%" };
        Boundary_map { pixel_false = "."; pixel_true = "*"; callback };
      ]
  in
  Stdio.printf "\nHalf-moons scatterplot and decision boundary:\n%!";
  PrintBox_text.output Stdio.stdout plot_moons;
  Stdio.printf "\nEpoch (cumulative) loss curve:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"loss"
      [ Line_plot { points = Array.of_list_rev !losses; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nLog-loss, for better visibility:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"log loss"
      [ Line_plot { points = Array.of_list_rev !log_losses; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nMinibatch log-loss:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"log loss"
      [ Line_plot { points = Array.of_list_rev !minibatch_loglosses; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nLearning rate:\n%!";
  let plot_lr =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"learning rate"
      [ Line_plot { points = Array.of_list_rev !learning_rates; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_lr;
  (* Stdio.printf "\nProcess memory delta: %d\n%!"
     (train_mem.process_physical_memory - init_mem.process_physical_memory); *)
  Exec_as_gccjit.optimization_level := 3;
  Stdio.printf "\n%!"

let benchmarks =
  [
    (* ("non-virt. Interpreter single", classify_moons ~virtualize:false Interpreter ~opti_level:3 CDSL.single); *)
    (* ("non-virt. OCaml single", classify_moons ~virtualize:false OCaml ~opti_level:3 CDSL.single); *)
    ("non-virt. gccjit O0 single", classify_moons ~virtualize:false Gccjit ~opti_level:0 CDSL.single);
    ("non-virt. gccjit O1 single", classify_moons ~virtualize:false Gccjit ~opti_level:1 CDSL.single);
    ("non-virt. gccjit O2 single", classify_moons ~virtualize:false Gccjit ~opti_level:2 CDSL.single);
    ("non-virt. gccjit O3 single", classify_moons ~virtualize:false Gccjit ~opti_level:3 CDSL.single);
    (* ("virtualized Interpreter single", classify_moons ~virtualize:true Interpreter ~opti_level:3 CDSL.single); *)
    (* ("virtualized OCaml single", classify_moons ~virtualize:true OCaml ~opti_level:3 CDSL.single); *)
    ("virtualized gccjit O0 single", classify_moons ~virtualize:true Gccjit ~opti_level:0 CDSL.single);
    ("virtualized gccjit O1 single", classify_moons ~virtualize:true Gccjit ~opti_level:1 CDSL.single);
    ("virtualized gccjit O2 single", classify_moons ~virtualize:true Gccjit ~opti_level:2 CDSL.single);
    ("virtualized gccjit O3 single", classify_moons ~virtualize:true Gccjit ~opti_level:3 CDSL.single);
    (* ("non-virt. Interpreter double", classify_moons ~virtualize:false Interpreter ~opti_level:3 CDSL.double); *)
    (* ("non-virt. OCaml double", classify_moons ~virtualize:false OCaml ~opti_level:3 CDSL.double); *)
    ("non-virt. gccjit O0 double", classify_moons ~virtualize:false Gccjit ~opti_level:0 CDSL.double);
    ("non-virt. gccjit O1 double", classify_moons ~virtualize:false Gccjit ~opti_level:1 CDSL.double);
    ("non-virt. gccjit O2 double", classify_moons ~virtualize:false Gccjit ~opti_level:2 CDSL.double);
    ("non-virt. gccjit O3 double", classify_moons ~virtualize:false Gccjit ~opti_level:3 CDSL.double);
    (* ("virtualized Interpreter double", classify_moons ~virtualize:true Interpreter ~opti_level:3 CDSL.double); *)
    (* ("virtualized OCaml double", classify_moons ~virtualize:true OCaml ~opti_level:3 CDSL.double); *)
    ("virtualized gccjit O0 double", classify_moons ~virtualize:true Gccjit ~opti_level:0 CDSL.double);
    ("virtualized gccjit O1 double", classify_moons ~virtualize:true Gccjit ~opti_level:1 CDSL.double);
    ("virtualized gccjit O2 double", classify_moons ~virtualize:true Gccjit ~opti_level:2 CDSL.double);
    ("virtualized gccjit O3 double", classify_moons ~virtualize:true Gccjit ~opti_level:3 CDSL.double);
  ]

let _suspended () = classify_moons ~virtualize:true Gccjit ~opti_level:3 CDSL.single ()

let () =
  List.map benchmarks ~f:(fun (name, test) -> Bench.Test.create ~name test)
  |> Bench.make_command |> Command_unix.run

(* Example output, 200 epochs:

   ┌────────────────────────────────┬──────────────┬─────────────┬────────────┬────────────┬────────────┐
   │ Name                           │     Time/Run │     mWd/Run │   mjWd/Run │   Prom/Run │ Percentage │
   ├────────────────────────────────┼──────────────┼─────────────┼────────────┼────────────┼────────────┤
   │ non-virt. Interpreter single   │ 109_023.97ms │ 30_436.21Mw │ 5_935.83kw │ 5_856.90kw │     87.81% │
   │ non-virt. OCaml single         │  31_138.47ms │  7_851.82Mw │   947.45kw │   830.35kw │     25.08% │
   │ non-virt. gccjit O0 single     │   1_620.33ms │      6.37Mw │   763.69kw │   755.67kw │      1.31% │
   │ non-virt. gccjit O1 single     │   1_009.66ms │      6.38Mw │   763.59kw │   755.58kw │      0.81% │
   │ non-virt. gccjit O2 single     │   1_157.94ms │      6.37Mw │   763.61kw │   755.59kw │      0.93% │
   │ non-virt. gccjit O3 single     │   3_032.70ms │      6.37Mw │   763.61kw │   755.59kw │      2.44% │
   │ virtualized Interpreter single │ 124_034.27ms │ 39_199.56Mw │ 9_344.26kw │ 9_203.45kw │     99.90% │
   │ virtualized OCaml single       │  29_818.01ms │  7_144.75Mw │ 1_120.24kw │   968.39kw │     24.02% │
   │ virtualized gccjit O0 single   │   1_373.18ms │      9.06Mw │   875.24kw │   867.23kw │      1.11% │
   │ virtualized gccjit O1 single   │     865.23ms │      9.05Mw │   875.21kw │   867.19kw │      0.70% │
   │ virtualized gccjit O2 single   │   1_017.83ms │      9.05Mw │   875.17kw │   867.15kw │      0.82% │
   │ virtualized gccjit O3 single   │   2_124.96ms │      9.06Mw │   875.27kw │   867.26kw │      1.71% │
   │ non-virt. Interpreter double   │ 114_671.32ms │ 30_436.20Mw │ 5_973.27kw │ 5_860.26kw │     92.36% │
   │ non-virt. OCaml double         │  30_803.15ms │  7_851.60Mw │   906.53kw │   806.55kw │     24.81% │
   │ non-virt. gccjit O0 double     │   1_646.47ms │      6.37Mw │   763.59kw │   755.57kw │      1.33% │
   │ non-virt. gccjit O1 double     │   1_046.90ms │      6.37Mw │   763.59kw │   755.57kw │      0.84% │
   │ non-virt. gccjit O2 double     │   1_181.91ms │      6.37Mw │   763.59kw │   755.57kw │      0.95% │
   │ non-virt. gccjit O3 double     │   3_014.77ms │      6.37Mw │   763.59kw │   755.58kw │      2.43% │
   │ virtualized Interpreter double │ 124_156.15ms │ 39_199.56Mw │ 9_342.78kw │ 9_201.58kw │    100.00% │
   │ virtualized OCaml double       │  30_173.51ms │  7_144.75Mw │ 1_120.10kw │   968.15kw │     24.30% │
   │ virtualized gccjit O0 double   │   1_617.95ms │      9.06Mw │   875.34kw │   867.33kw │      1.30% │
   │ virtualized gccjit O1 double   │     975.96ms │      9.05Mw │   875.19kw │   867.17kw │      0.79% │
   │ virtualized gccjit O2 double   │   1_083.58ms │      9.06Mw │   875.24kw │   867.23kw │      0.87% │
   │ virtualized gccjit O3 double   │   2_237.71ms │      9.06Mw │   875.33kw │   867.32kw │      1.80% │
   └────────────────────────────────┴──────────────┴─────────────┴────────────┴────────────┴────────────┘

   20000 epochs:

   ┌──────────────────────────────┬──────────┬─────────┬──────────┬──────────┬────────────┐
   │ Name                         │ Time/Run │ mWd/Run │ mjWd/Run │ Prom/Run │ Percentage │
   ├──────────────────────────────┼──────────┼─────────┼──────────┼──────────┼────────────┤
   │ non-virt. gccjit O0 single   │   96.55s │ 96.35Mw │   1.32Mw │   1.07Mw │     91.57% │
   │ non-virt. gccjit O1 single   │   27.81s │ 96.34Mw │   1.31Mw │   1.07Mw │     26.38% │
   │ non-virt. gccjit O2 single   │   18.43s │ 96.35Mw │   1.31Mw │   1.07Mw │     17.48% │
   │ non-virt. gccjit O3 single   │   16.09s │ 96.35Mw │   1.31Mw │   1.07Mw │     15.26% │
   │ virtualized gccjit O0 single │   92.85s │ 99.03Mw │   1.43Mw │   1.18Mw │     88.06% │
   │ virtualized gccjit O1 single │   24.11s │ 99.03Mw │   1.42Mw │   1.18Mw │     22.86% │
   │ virtualized gccjit O2 single │   18.44s │ 99.03Mw │   1.42Mw │   1.18Mw │     17.49% │
   │ virtualized gccjit O3 single │   15.00s │ 99.03Mw │   1.42Mw │   1.18Mw │     14.23% │
   │ non-virt. gccjit O0 double   │  105.44s │ 96.35Mw │   1.31Mw │   1.07Mw │    100.00% │
   │ non-virt. gccjit O1 double   │   28.42s │ 96.35Mw │   1.31Mw │   1.07Mw │     26.95% │
   │ non-virt. gccjit O2 double   │   21.55s │ 96.35Mw │   1.31Mw │   1.07Mw │     20.44% │
   │ non-virt. gccjit O3 double   │   17.42s │ 96.35Mw │   1.31Mw │   1.07Mw │     16.53% │
   │ virtualized gccjit O0 double │   93.55s │ 99.03Mw │   1.42Mw │   1.18Mw │     88.73% │
   │ virtualized gccjit O1 double │   24.92s │ 99.03Mw │   1.42Mw │   1.18Mw │     23.64% │
   │ virtualized gccjit O2 double │   20.29s │ 99.03Mw │   1.42Mw │   1.18Mw │     19.25% │
   │ virtualized gccjit O3 double │   17.72s │ 99.03Mw │   1.42Mw │   1.18Mw │     16.80% │
   └──────────────────────────────┴──────────┴─────────┴──────────┴──────────┴────────────┘
*)

open Base
open Ocannl
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL
module CDSL = Code.CDSL
module SDSL = Session.SDSL

let () =
  (* Code.CDSL.with_debug := false; *)
  (* let open Operation.FDSL in *)
  let open SDSL.O in
  (* SDSL.set_executor Interpreter;
     SDSL.enable_all_debugs ~trace_interpreter:true (); *)
  SDSL.set_executor Gccjit;
  (* SDSL.enable_all_debugs (); *)
  CDSL.virtualize_settings.enable_device_only <- true;
  CDSL.virtualize_settings.inline_constants <- true;
  SDSL.drop_all_sessions ();
  Random.init 0;
  let num_parallel_tasks = SDSL.num_domains - 1 in
  Stdio.printf "\n****** num_parallel_tasks = %d\n%!" num_parallel_tasks;
  SDSL.num_parallel_tasks := num_parallel_tasks;
  let minibatch = 4 in
  let refresh_batch = 2 in
  let batch = refresh_batch * num_parallel_tasks * minibatch in
  let n_batches = 2 in
  let len = n_batches * batch / 2 in
  let epochs = 100 in
  let refreshes = epochs * n_batches in
  let steps = refreshes * refresh_batch in
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
  let b = Shape.[ Frozen n_batches; Parallel; Dim refresh_batch; Dim minibatch ] in
  let moons_flat = FDSL.init_const ~l:"moons_flat" ~b ~o:Shape.[ Dim 2 ] moons_flat in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes = FDSL.init_const ~l:"moons_classes" ~b ~o:Shape.[ Dim 1 ] moons_classes in
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
  let%nn_dt session_step ~o:1 = n =+ 1 in
  let%nn_dt session_refresh ~o:1 = n =+ 1 /. !..refresh_batch in
  let%nn_dt minus_lr ~o:1 = n =: -0.0001 *. (!..steps - session_step) /. !..steps in

  SDSL.minus_learning_rate := Some minus_lr;
  let%nn_op moons_input = (moons_flat @.| session_refresh) @.| session_step in
  let%nn_op moons_class = (moons_classes @.| session_refresh) @.| session_step in
  let%nn_op margin_loss = !/(1 - (moons_class *. mlp moons_input)) in
  let%nn_op ssq w = (w **. 2) ++ "...|...->... => 0" in
  let reg_loss = List.map ~f:ssq [ w1; w2; b1; b2 ] |> List.reduce_exn ~f:FDSL.O.( + ) in
  (* let reg_loss =
       List.map ~f:ssq [ w1; w2; w3; w4; w5; w6; b1; b2; b3; b4; b5; b6 ] |> List.reduce_exn ~f:FDSL.O.( + )
     in *)
  let step_batch = num_parallel_tasks * minibatch in
  let%nn_op subsubtotal = margin_loss ++ "...|... => ...|0" in
  SDSL.set_non_virtual subsubtotal;
  let%nn_op subtotal = subsubtotal ++ "...|0 => 0" in
  SDSL.set_non_device_only subtotal;
  let%nn_op total_loss = (subtotal /. !..step_batch) + (0.001 *. reg_loss) in
  let%nn_dt epoch_loss ~o:1 = n =+ total_loss in
  let run_for_steps = refresh_batch in
  SDSL.refresh_session ~run_for_steps ();
  Stdio.print_endline "\nPreamble:\n";
  SDSL.print_preamble ();
  Stdio.print_endline "\nHigh-level code:\n";
  SDSL.print_session_code ();
  Stdio.print_endline "\nLow-level code:\n";
  SDSL.print_session_code ~compiled:true ();
  SDSL.print_decimals_precision := 9;
  Stdio.printf "Step 1: session_step: %f\n%!" session_step.@[0];
  Stdio.printf "Step 1: Minus learning rate: %f\n%!" minus_lr.@[0];
  Stdio.printf "\nStep 1: loss %f\n%!" epoch_loss.@[0];
  SDSL.print_node_tree ~with_id:true ~with_grad:true ~depth:9 total_loss.id;
  (* List.iter [ w1; w2; b1; b2 ] ~f:(fun f ->
      Stdio.print_endline "\n";
      SDSL.print_formula ~with_grad:true ~with_code:false `Default f); *)
  SDSL.refresh_session ();
  Stdio.printf "Step 2: session_step: %f\n%!" session_step.@[0];
  Stdio.printf "\nStep 2: Minus learning rate: %f\n%!" minus_lr.@[0];
  Stdio.printf "\nStep 2: loss %f\n%!" epoch_loss.@[0];
  SDSL.print_node_tree ~with_id:true ~with_grad:true ~depth:9 total_loss.id;
  (* List.iter [ w1; w2; b1; b2 ] ~f:(fun f ->
      Stdio.print_endline "\n";
      SDSL.print_formula ~with_grad:true ~with_code:false `Default f); *)
  SDSL.refresh_session ();
  Stdio.printf "Step 3: session_step: %f\n%!" session_step.@[0];
  Stdio.printf "\nStep 3: Minus learning rate: %f\n%!" minus_lr.@[0];
  Stdio.printf "\nStep 3: loss %f\n%!" epoch_loss.@[0];
  SDSL.print_node_tree ~with_id:true ~with_grad:true ~depth:9 total_loss.id;
  SDSL.refresh_session ();
  Stdio.printf "Step 4: session_step: %f\n%!" session_step.@[0];
  Stdio.printf "\nStep 4: Minus learning rate: %f\n%!" minus_lr.@[0];
  Stdio.printf "\nStep 4: loss %f\n%!" epoch_loss.@[0];
  SDSL.print_node_tree ~with_id:true ~with_grad:true ~depth:9 total_loss.id;
  Stdio.printf "\nSize in bytes: %d\n%!" (SDSL.global_size_in_bytes ())

let classify_moons ~on_device executor ~opti_level ~inlining_cutoff ?(inline_constants = true)
    ~num_parallel_tasks precision () =
  (* let epochs = 20000 in *)
  let epochs = 2000 in
  (* let epochs = 200 in *)
  (* let epochs = 20 in *)
  let bench_title =
    Sexp.to_string_hum
    @@ [%sexp_of:
         string (* * Session.backend *)
         * string
         * int
         * string
         * int (* * string *)
         * string
         * int
         * NodeUI.prec]
         ( (if on_device then "on-dev" else "non-d."),
           (* executor, *)
           "gcc-opt",
           opti_level,
           "inline",
           inlining_cutoff,
           (* (if inline_constants then "..." else "NIC"), *)
           "parallel",
           num_parallel_tasks,
           precision )
  in
  Stdio.prerr_endline @@ "\n\n****** Benchmarking virtualized: " ^ bench_title ^ " for "
  ^ Int.to_string epochs ^ " epochs ******";
  CDSL.virtualize_settings.enable_device_only <- on_device;
  CDSL.virtualize_settings.max_visits <- inlining_cutoff;
  CDSL.virtualize_settings.inline_constants <- inline_constants;
  SDSL.set_executor executor;
  SDSL.default_value_prec := precision;
  SDSL.default_grad_prec := precision;
  Exec_as_gccjit.optimization_level := opti_level;
  SDSL.num_parallel_tasks := num_parallel_tasks;
  SDSL.disable_all_debugs ();
  (* SDSL.enable_all_debugs (); *)
  SDSL.drop_all_sessions ();
  let open SDSL.O in
  Random.init 0;
  (* let init_mem = Mem_usage.info () in *)
  let hid_2_3 = Shape.Dim 8 in
  let hid_4_5 = Shape.Dim 4 in
  let init_time = Time_now.nanoseconds_since_unix_epoch () in
  let len = 4200 in
  (* let len = 420 in *)
  let minibatch = 10 in
  let batch = minibatch * num_parallel_tasks in
  let n_batches = 2 * len / batch in
  let steps = epochs * n_batches in
  (* let run_for_steps = 10000 * n_batches in *)
  let run_for_steps = 100 * n_batches in
  (* let run_for_steps = n_batches in *)
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
  let moons_flat =
    FDSL.init_const ~l:"moons_flat" ~b:[ Dim n_batches; Parallel; Dim minibatch ] ~o:[ Dim 2 ] moons_flat
  in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes =
    FDSL.init_const ~l:"moons_classes"
      ~b:[ Dim n_batches; Parallel; Dim minibatch ]
      ~o:[ Dim 1 ] moons_classes
  in
  let%nn_op mlp x =
    "b6" 1
    + "w6"
      * !/("b4" hid_4_5
          + "w4"
            * !/("b2" hid_2_3
                + ("w2" * !/("b1" 16 + ("w1" * x)))
                + "b3" hid_2_3
                + ("w3" * !/(b2 + (w2 * !/(b1 + (w1 * x))))))
          + ("b5" hid_4_5 + ("w5" * !/(b4 + (w4 * !/(b3 + (w3 * !/(b2 + (w2 * !/(b1 + (w1 * x)))))))))))
  in
  (* let%nn_op mlp x =
       "b6" 1
       + "w6"
         * !/("b5" 4
             + ("w5" * !/("b4" 4 + ("w4" * !/("b3" 8 + ("w3" * !/("b2" 8 + ("w2" * !/("b1" 16 + ("w1" * x)))))))))
             )
     in *)
  (* let%nn_op mlp x = "b2" 1 + ("w2" * !/("b1" 16 + ("w1" * x))) in *)
  let%nn_dt session_step ~o:1 = n =+ 1 in
  let%nn_dt minus_lr ~o:1 = n =: -0.01 *. (!..steps - session_step) /. !..steps in
  SDSL.minus_learning_rate := Some minus_lr;
  let%nn_op moons_input = moons_flat @.| session_step in
  let%nn_op moons_class = moons_classes @.| session_step in
  let min_loss = ref Float.infinity in
  let max_loss = ref 0.0 in
  let losses = ref [] in
  let log_losses = ref [] in
  let learning_rates = ref [] in
  let stop = ref false in
  let step = ref 0 in
  let%nn_op margin_loss = !/(1 - (moons_class *. mlp moons_input)) in
  let%nn_op ssq w = (w **. 2) ++ "...|...->... => 0" in
  let reg_loss =
    List.map ~f:ssq [ w1; w2; w3; w4; w5; w6; b1; b2; b3; b4; b5; b6 ] |> List.reduce_exn ~f:FDSL.O.( + )
  in
  (* let reg_loss = List.map ~f:ssq [ w1; w2; b1; b2 ] |> List.reduce_exn ~f:FDSL.O.( + ) in *)
  let%nn_op total_loss = ((margin_loss ++ "...|... => 0") /. !..minibatch) + (0.0001 *. reg_loss) in
  (* This is not really an epoch loss, it's a run_for_steps cumulative loss. *)
  let%nn_dt epoch_loss ~o:1 = n =+ total_loss in
  let loss = ref 0.0 in
  Stdio.printf "\n%!";
  while not !stop do
    SDSL.refresh_session ~run_for_steps ();
    step := !step + run_for_steps;
    loss := epoch_loss.@[0];
    if Float.(!loss < !min_loss) then min_loss := !loss;
    if Float.(!loss > !max_loss) then max_loss := !loss;
    learning_rates := ~-.(minus_lr.@[0]) :: !learning_rates;
    losses := !loss :: !losses;
    log_losses := Float.log !loss :: !log_losses;
    if !step / n_batches % (epochs / 10) = 0 || !stop then (
      Stdio.printf "Minus learning rate over minibatch for step %d of %d: %f\n%!" !step steps minus_lr.@[0];
      Stdio.printf "Epoch loss for step %d: %f; min loss: %f; max loss: %f\n%!" !step !loss !min_loss
        !max_loss);
    (* Stdio.printf "\nPreamble:\n%!";
       SDSL.print_preamble ();
       Stdio.printf "\nTree:\n%!";
       SDSL.print_node_tree ~with_grad:true ~depth:9 total_loss.id; *)
    epoch_loss.@[0] <- 0.0;
    if !step >= steps then stop := true
  done;
  let points = SDSL.value_2d_points ~xdim:0 ~ydim:1 moons_flat in
  let classes = SDSL.value_1d_points ~xdim:0 moons_classes in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  (* let train_mem = Mem_usage.info () in *)
  let final_time = Time_now.nanoseconds_since_unix_epoch () in
  let time_in_sec = Int63.(to_float @@ (final_time - init_time)) /. 1000_000_000. in
  Stdio.printf "\nTime in sec: %f\n%!" time_in_sec;
  let result =
    PrintBox_utils.Benchmark
      {
        bench_title;
        time_in_sec;
        total_size_in_bytes = SDSL.global_size_in_bytes ();
        result_label = "min epoch loss, last epoch loss";
        (* This is not really an epoch loss, it's a run_for_steps cumulative loss. *)
        result = [%sexp_of: float * float] (!min_loss, !loss);
      }
  in
  SDSL.close_session ();
  Stdio.print_endline "\nSession closed.";
  SDSL.num_parallel_tasks := 1;
  let%nn_op point = (* [ 0; 0 ] *) "point" 2 in
  let mlp_result = mlp point in
  SDSL.refresh_session ();
  let callback (x, y) =
    SDSL.set_values point [| x; y |];
    SDSL.refresh_session ();
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
  Stdio.printf "\n%!";
  result

let benchmark_executor = SDSL.Gccjit

let _suspended () =
  ignore
  @@ classify_moons ~on_device:true benchmark_executor ~opti_level:3 ~inlining_cutoff:3 ~num_parallel_tasks:21
       CDSL.single ()

let benchmarks =
  [
    classify_moons ~on_device:true benchmark_executor ~opti_level:3 ~inlining_cutoff:3 ~num_parallel_tasks:1
      CDSL.single;
    classify_moons ~on_device:true benchmark_executor ~opti_level:3 ~inlining_cutoff:3 ~num_parallel_tasks:2
      CDSL.single;
    classify_moons ~on_device:true benchmark_executor ~opti_level:3 ~inlining_cutoff:3 ~num_parallel_tasks:4
      CDSL.single;
    classify_moons ~on_device:true benchmark_executor ~opti_level:3 ~inlining_cutoff:3 ~num_parallel_tasks:5
      CDSL.single;
    classify_moons ~on_device:true benchmark_executor ~opti_level:3 ~inlining_cutoff:3 ~num_parallel_tasks:10
      CDSL.single;
    classify_moons ~on_device:true benchmark_executor ~opti_level:3 ~inlining_cutoff:3 ~num_parallel_tasks:12
      CDSL.single;
    classify_moons ~on_device:true benchmark_executor ~opti_level:3 ~inlining_cutoff:3 ~num_parallel_tasks:15
      CDSL.single;
    classify_moons ~on_device:true benchmark_executor ~opti_level:3 ~inlining_cutoff:3 ~num_parallel_tasks:18
      CDSL.single;
    classify_moons ~on_device:true benchmark_executor ~opti_level:3 ~inlining_cutoff:3 ~num_parallel_tasks:21
      CDSL.single;
    (* classify_moons ~on_device:true benchmark_executor ~opti_level:3 ~inlining_cutoff:3 ~num_parallel_tasks:30
       CDSL.single; *)
    (* classify_moons ~on_device:true benchmark_executor ~opti_level:3 ~inlining_cutoff:3 ~num_parallel_tasks:42
       CDSL.single; *)
  ]

let _suspended () =
  List.map benchmarks ~f:(fun bench -> bench ()) |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout

(* Early example output from back when using core_bench, 200 epochs (all are non-virtual):

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

   Parallelization turned out to be a failed sub-project, because the Gccjit-compiled code has too much
   memory contention. There is no benefit from running it in parallel unless we prepare a "working-memory"
   copy of the specific sub-tensors that each task operates on.

    ┌───────────────────────────────────────────────────┬────────────┬───────────────┬───────┬────────┬───────────────────────────────────────┐
    │Benchmarks                                         │Time in sec │Memory in bytes│Speedup│Mem gain│min epoch loss, last epoch loss        │
    ├───────────────────────────────────────────────────┼────────────┼───────────────┼───────┼────────┼───────────────────────────────────────┤
    │(virtu. gcc-opt 3 inline 3 parallel 1 Single_prec) │11.504924748│114572         │1.106  │2.941   │(60.326202392578125 60.326202392578125)│
    │(virtu. gcc-opt 3 inline 3 parallel 2 Single_prec) │8.327399981 │125692         │1.528  │2.681   │(42008.91796875 42008.91796875)        │
    │(virtu. gcc-opt 3 inline 3 parallel 4 Single_prec) │8.970155171 │147932         │1.419  │2.278   │(13.373936653137207 13.373936653137207)│
    │(virtu. gcc-opt 3 inline 3 parallel 5 Single_prec) │9.000583522 │159052         │1.414  │2.119   │(16807.171875 16807.171875)            │
    │(virtu. gcc-opt 3 inline 3 parallel 10 Single_prec)│11.491387089│214652         │1.108  │1.570   │(8406.435546875 8407.2265625)          │
    │(virtu. gcc-opt 3 inline 3 parallel 12 Single_prec)│10.743618418│236892         │1.185  │1.422   │(7006.48681640625 7008.13037109375)    │
    │(virtu. gcc-opt 3 inline 3 parallel 15 Single_prec)│11.777287131│270252         │1.081  │1.247   │(5603.9892578125 5605.5009765625)      │
    │(virtu. gcc-opt 3 inline 3 parallel 18 Single_prec)│11.590590388│302172         │1.098  │1.115   │(2.7644500732421875 3.0231761932373047)│
    │(virtu. gcc-opt 3 inline 3 parallel 21 Single_prec)│12.726874021│336972         │1.000  │1.000   │(3.4613373279571533 3.8374130725860596)│
    └───────────────────────────────────────────────────┴────────────┴───────────────┴───────┴────────┴───────────────────────────────────────┘
*)

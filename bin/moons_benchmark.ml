open Base
open Ocannl
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Arrayjit.Low_level.CDSL


let _suspended () =
  (* Session.Utils.settings.with_debug <- false; *)
  (* let open Operation.TDSL in *)
  let open Tensor.O in
  (* CDSL.enable_all_debugs (); *)

  (* CDSL.enable_all_debugs (); *)
  (*
  CDSL.virtualize_settings.enable_device_only <- true;
  *)
  Utils.settings.with_debug <- true;
  Utils.settings.keep_files_in_run_directory <- true;
  (* SDSL.drop_all_sessions (); *)
  Random.init 0;
  let parallel_dims = 1 in
  Stdio.printf "\n****** parallel_dims = %d\n%!" parallel_dims;
  (* FIXME: *)
  let refresh_batch = 1 in
  let batch = refresh_batch * parallel_dims in
  let n_batches = 2 in
  let len = n_batches * batch / 2 in
  (* let epochs = 100 in *)
  let epochs = 2 in
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
  (* FIXME: *)
  let b =
    [
      n_batches;
      parallel_dims;
      refresh_batch;
      20;
    ]
  in
  let moons_flat = TDSL.init_const ~l:"moons_flat" ~b ~o:[ 2 ] moons_flat in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes = TDSL.init_const ~l:"moons_classes" ~b ~o:[ 1 ] moons_classes in
  (* let%op mlp x = "b2" 1 + ("w2" * !/("b1" 2 + ("w1" * x))) in *)
  let%op mlp x = "b1" 1 + ("w1" * x) in

  let session_step = NTDSL.O.(NTDSL.counter !..1) in
  let session_refresh = NTDSL.O.(NTDSL.counter (!..1 /. !..refresh_batch)) in
  let%op minus_lr = -0.00001 in

  (* minus_learning_rate := Some minus_lr; *)
  let%op moons_input = (moons_flat @| session_refresh) @| session_step in
  let%op moons_class = (moons_classes @| session_refresh) @| session_step in
  let%op margin_loss = !/(1 - (moons_class *. mlp moons_input)) in
  (* let%op ssq w = (w **. 2) ++ "...|...->... => 0" in *)
  let%op ssq w = (w *. w) ++ "...|...->... => 0" in
  let reg_loss = List.map ~f:ssq [ w1; b1 ] |> List.reduce_exn ~f:TDSL.O.( + ) in
  (* let reg_loss =
       List.map ~f:ssq [ w1; w2; w3; w4; w5; w6; b1; b2; b3; b4; b5; b6 ] |> List.reduce_exn ~f:TDSL.O.( + )
     in *)
  let%op weighted_reg_loss = 0.00001 *. reg_loss in
  (* let step_batch = parallel_dims * minib in *)
  let%op batch_of_losses = margin_loss ++ "...|... => ...|0" in
  let updates_per_run = refresh_batch in
  (* every_non_literal_fully_on_host (); *)
  (* SDSL.everything_on_host_or_inlined (); *)
  SDSL.refresh_session ~updates_per_run ();
  Stdio.print_endline "\nPreamble:\n";
  (* print_preamble (); *)
  Stdio.print_endline "\nHigh-level code:\n";
  (* print_session_code (); *)
  Stdio.print_endline "\nLow-level code:\n";
  (* print_session_code ~compiled:true (); *)
  SDSL.print_decimals_precision := 9;
  Stdio.printf "Step 1: session_step: %f, session_refresh: %f, of steps %d\n%!" session_step.@[0]
    session_refresh.@[0] steps;
  Stdio.printf "Step 1: Minus learning rate: %f\n%!" minus_lr.@[0];
  Tensor.print_tree ~with_id:true ~with_grad:true ~depth:9 minus_lr;
  let step_1_loss = batch_loss.@[0] in
  Stdio.printf "\nStep 1: loss %f\n%!" step_1_loss;
  Tensor.print_tree ~with_id:true ~with_grad:true (* ~with_backend_info:true *) ~depth:9 batch_of_losses;
  (* List.iter [ w1; w2; b1; b2 ] ~f:(fun f ->
      Stdio.print_endline "\n";
      Tensor.print ~with_grad:true ~with_code:false `Default f); *)
  (* refresh_session (); *)
  Stdio.printf "Step 2: session_step: %f\n%!" session_step.@[0];
  Stdio.printf "\nStep 2: Minus learning rate: %f\n%!" minus_lr.@[0];
  let step_2_loss = batch_loss.@[0] in
  Stdio.printf "\nStep 2: cumulative loss %f, step loss %f\n%!" step_2_loss (step_2_loss -. step_1_loss);
  Tensor.print_tree ~with_id:true ~with_grad:true (* ~with_backend_info:true *) ~depth:9 batch_of_losses;
  (* List.iter [ w1; w2; b1; b2 ] ~f:(fun f ->
      Stdio.print_endline "\n";
      Tensor.print ~with_grad:true ~with_code:false `Default f); *)
  (* refresh_session (); *)
  Stdio.printf "Step 3: session_step: %f\n%!" session_step.@[0];
  Stdio.printf "\nStep 3: Minus learning rate: %f\nStep 3 weighted reg. loss:\n%!" minus_lr.@[0];
  Tensor.print_tree ~with_id:true ~with_grad:true (* ~with_backend_info:true *)
    ~depth:9 weighted_reg_loss;
  let step_3_loss = batch_loss.@[0] in
  Stdio.printf "\nStep 3: cumulative loss %f, step loss %f\n%!" step_3_loss (step_3_loss -. step_2_loss);
  Tensor.print_tree ~with_id:true ~with_grad:true (* ~with_backend_info:true *) ~depth:9 batch_of_losses;
  (* refresh_session (); *)
  Stdio.printf "Step 4: session_step: %f\n%!" session_step.@[0];
  Stdio.printf "\nStep 4: Minus learning rate: %f\n%!" minus_lr.@[0];
  Stdio.printf "\nStep 4 weighted reg. loss:\n%!";
  Tensor.print_tree ~with_id:true ~with_grad:true (* ~with_backend_info:true *)
    ~depth:9 weighted_reg_loss;
  Stdio.printf "\nStep 4 epoch loss tensor:\n%!";
  Tensor.print_tree ~with_id:true ~with_grad:true (* ~with_backend_info:true *) ~depth:9 batch_loss;
  let step_4_loss = batch_loss.@[0] in
  Stdio.printf "\nStep 4: cumulative loss %f, step loss %f\n%!" step_4_loss (step_4_loss -. step_3_loss);
  Tensor.print_tree ~with_id:true ~with_grad:true (* ~with_backend_info:true *) ~depth:9 batch_of_losses;
  Stdio.printf "\nHost size in bytes: %d\n%!" (SDSL.global_host_size_in_bytes ())

let classify_moons ~with_reg ~random_seed ~on_device executor ~inlining_cutoff
    ~parallel_dims ~minibatch_size ~updates_per_run precision () =
  (* ignore random_seed; *)
  let bench_title =
    Sexp.to_string_hum
    @@ [%sexp_of:
         string
         * int
         * string
         * string
         * Session.backend
         * string
         * int
         * string
         * int
         * string
         * int
         * string
         * int
         * Ndarray.prec]
         ( "seed",
           random_seed,
           (if with_reg then "reg.loss" else "non-reg."),
           (if on_device then "on-dev" else "non-d."),
           executor,
           "inline",
           inlining_cutoff,
           "parallel",
           parallel_dims,
           "minibatch",
           minibatch_size,
           "per-sync",
           updates_per_run,
           precision )
  in
  Stdio.printf "\n*** %s ***\n%!" bench_title;
  CDSL.virtualize_settings.enable_device_only <- on_device;
  CDSL.virtualize_settings.max_visits <- inlining_cutoff;
  Stdio.printf "Set the executor.\n%!";
  SDSL.set_executor executor;
  SDSL.default_value_prec := precision;
  SDSL.default_grad_prec := precision;
  let open Tensor.O in
  (* SDSL.drop_all_sessions (); *)
  (* Utils.settings.with_debug <- true; *)
  (* Utils.settings.keep_files_in_run_directory <- true; *)
  (* Low_level.debug_log_jitted := true; *)
  Random.init (* random_seed *) 0;
  Utils.settings.fixed_state_for_init <- Some random_seed;
  (* let hid_2_3 = 8 in
     let hid_4_5 = 4 in *)
  let hid_dim = 16 in
  let repeats = updates_per_run in
  let len = 320 in
  (* let len = 10240 in *)
  let batch_size = parallel_dims * minibatch_size in
  let n_batches = 2 * len / batch_size in
  let epochs = 100 / repeats in
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
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let init_time = Time_now.nanoseconds_since_unix_epoch () in
  let moons_flat =
    TDSL.init_const ~l:"moons_flat"
      ~b:[ n_batches; parallel_dims; minibatch_size ]
      ~o:[ 2 ]
      moons_flat
  in
  let moons_classes =
    TDSL.init_const ~l:"moons_classes"
      ~b:[ n_batches; parallel_dims; minibatch_size ]
      ~o:[ 1 ]
      moons_classes
  in
  (* *
     let%op mlp x =
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
     * *)
  let%op mlp x = "b3" 1 + ("w3" * !/("b2" hid_dim + ("w2" * !/("b1" hid_dim + ("w1" * x))))) in
  let session_step = NTDSL.O.(NTDSL.counter !..1) in
  let steps = epochs * n_batches in
  let%op minus_lr = -0.1 *. (!..steps - session_step) /. !..steps in
  (* minus_learning_rate := Some minus_lr; *)
  let%op moons_input = moons_flat @| session_step in
  let%op moons_class = moons_classes @| session_step in
  let%op margin_loss = !/(1 - (moons_class *. mlp moons_input)) in
  let total_loss =
    if with_reg then
      let%op ssq w = (w **. 2) ++ "...|...->... => 0" in
      (* let reg_loss =
           List.map ~f:ssq [ w1; w2; w3; w4; w5; w6; b1; b2; b3; b4; b5; b6 ] |> List.reduce_exn ~f:TDSL.O.( + )
         in *)
      let reg_loss = List.map ~f:ssq [ w1; w2; w3; b1; b2; b3 ] |> List.reduce_exn ~f:TDSL.O.( + ) in
      let%op total_loss = ((margin_loss ++ "...|... => 0") /. !..batch_size) + (0.0001 *. reg_loss) in
      total_loss
    else
      let%op total_loss = (margin_loss ++ "...|... => 0") /. !..batch_size in
      total_loss
  in
  (* Warmup step, with update. *)
  SDSL.refresh_session ~updates_per_run ();
  let losses = ref [] in
  let log_losses = ref [] in
  let batch_losses = ref [] in
  let batch_log_losses = ref [] in
  let learning_rates = ref [] in
  let min_loss = ref Float.infinity in
  let max_loss = ref 0.0 in
  let last_loss = ref Float.infinity in
  let start_time = Time_now.nanoseconds_since_unix_epoch () in
  for epoch = 1 to epochs do
    for batch_n = 1 to n_batches do
      SDSL.refresh_session ~updates_per_run ();
      batch_losses := total_loss.@[0] :: !batch_losses;
      batch_log_losses := Float.log total_loss.@[0] :: !batch_log_losses;
      if
        (* false
           && *)
        (epochs / 10 = 0 || epoch = epochs || epoch % (epochs / 10) = 1)
        && (n_batches / 5 = 0 || batch_n = n_batches || batch_n % (n_batches / 5) = 1)
      then
        Stdio.printf "Epoch=%d, batch=%d, session_step=%f, -lr=%f, batch loss=%f\n%!" epoch
          batch_n session_step.@[0] minus_lr.@[0] total_loss.@[0]
        (* Tensor.print_tree ~with_backend_info:true ~with_grad:true ~depth:9 total_loss; *)
    done;
    learning_rates := ~-.(minus_lr.@[0]) :: !learning_rates;
    last_loss := batch_loss.@[0];
    losses := !last_loss :: !losses;
    min_loss := Float.min !min_loss !last_loss;
    max_loss := Float.max !max_loss !last_loss;
    log_losses := Float.log !last_loss :: !log_losses
  done;
  let final_time = Time_now.nanoseconds_since_unix_epoch () in
  (* TODO: include init time in benchmarks? *)
  ignore init_time;
  let time_in_sec = Int63.(to_float @@ (final_time - start_time)) /. 1000_000_000. in
  Stdio.printf "\nTime in sec: %f\n%!" time_in_sec;
  let result =
    PrintBox_utils.Benchmark
      {
        bench_title;
        time_in_sec;
        mem_in_bytes = SDSL.global_host_size_in_bytes ();
        result_label = "min epoch loss, last epoch loss";
        (* This is not really an epoch loss, it's a run_for_steps cumulative loss. *)
        result = [%sexp_of: float * float] (!min_loss, !last_loss);
      }
  in
  Utils.settings.with_debug <- false;
  Utils.settings.keep_files_in_run_directory <- false;
  let points = Tensor.value_2d_points ~xdim:0 ~ydim:1 moons_flat in
  let classes = Tensor.value_1d_points ~xdim:0 moons_classes in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  SDSL.close_session ();
  let%op point = [ 0; 0 ] in
  let mlp_result = mlp point in
  SDSL.refresh_session ~with_backprop:false ();
  let callback (x, y) =
    SDSL.set_values point [| x; y |];
    SDSL.refresh_session ~with_backprop:false ();
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
  Stdio.printf "\nLoss, per-batch <.>:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"batch L ."
      [ Line_plot { points = Array.of_list_rev !batch_losses; pixel = "." } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nLoss, per-epoch <->:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"epoch L -"
      [ Line_plot { points = Array.of_list_rev !losses; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nLog-loss, per-batch <.>:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"batch log L ."
      [ Line_plot { points = Array.of_list_rev !batch_log_losses; pixel = "." } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nLog-loss, per-epoch:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"epoch log L -"
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
  Stdio.printf "\n\n%!";
  result

let _suspended () =
  ignore
  @@ classify_moons ~with_reg:false ~random_seed:3 ~on_device:true SDSL.Cuda ~inlining_cutoff:3
       ~parallel_dims:1 ~minibatch_size:20 ~updates_per_run:1 CDSL.single ()

let benchmarks =
  List.concat_map [ (* 0; 3; 5 *) 3 ] ~f:(fun inlining_cutoff ->
      List.concat_map [ 1; (* 2; 4; 8; 10; *) 16 (* ; 20 *) ] ~f:(fun parallel_dims ->
          List.concat_map [ (* 1; 8; *) 32; 64; 128 (* ; 256; 512; 1024 *) ] ~f:(fun minibatch_size ->
              List.concat_map [ 1 (* ; 10; 100 *) ] ~f:(fun updates_per_run ->
                  List.concat_map [ 0; 1; 2 (* ; 3; 4 *) ] ~f:(fun random_seed ->
                      List.concat_map [ SDSL.Cuda (* ; SDSL.Gccjit *) ] ~f:(fun executor ->
                          [
                            classify_moons ~random_seed ~on_device:true executor ~inlining_cutoff
                              ~parallel_dims ~minibatch_size ~updates_per_run CDSL.single;
                          ]))))))

(*
let time_of = function PrintBox_utils.Benchmark { time_in_sec; _ } -> time_in_sec
let nth_best nth bench =
  let results = List.init 5 ~f:(fun random_seed -> bench ~random_seed ()) in
  let sorted = List.sort results ~compare:(fun r1 r2 -> Float.compare (time_of r1) (time_of r2)) in
  List.nth_exn sorted (nth - 1)
*)

let fixed_seed_search random_seed =
  classify_moons ~with_reg:false ~random_seed ~on_device:true SDSL.Cuda ~inlining_cutoff:3 ~parallel_dims:1
    ~minibatch_size:32 ~updates_per_run:1 CDSL.single ()

let _suspended () =
  List.init 20 ~f:fixed_seed_search |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout

(*
let () =
  List.map benchmarks ~f:(nth_best 2) |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout
*)

let benchmark ~with_reg =
  List.map benchmarks ~f:(fun bench -> bench ~with_reg ())
  |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout

let () = benchmark ~with_reg:true

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

open Base
open Ocannl
module Nd = Ir.Ndarray
module Ops = Ir.Ops
module Tn = Ir.Tnode
module IDX = Train.IDX
open Operation.DSL_modules
module CDSL = Train.CDSL

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_MOONS_BENCHMARK=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_MOONS_BENCHMARK"]

let classify_moons ~seed ~on_device ~inlining_cutoff ~num_streams ~batch_size ~backend_name
    ~value_prec ~grad_prec () =
  [%track_sexp
    let _debug : string = "started" in
    (fun (started : unit) -> started) ()];
  (* ignore seed; *)
  let bench_title =
    [%string
      "seed %{seed#Int}, inline %{inlining_cutoff#Int}, parallel %{num_streams#Int}, batch \
       %{batch_size#Int}, backend %{backend_name}, val prec %{Ops.prec_string value_prec}, grad \
       prec %{Ops.prec_string grad_prec}"]
  in
  Stdio.printf "\n*** %s ***\n%!" bench_title;
  CDSL.virtualize_settings.enable_device_only <- on_device;
  CDSL.virtualize_settings.max_visits <- inlining_cutoff;
  Tensor.default_value_prec := value_prec;
  Tensor.default_grad_prec := grad_prec;
  let hid_dim_1 = 16 in
  let hid_dim_2 = 8 in
  let hid_dim_3 = 4 in
  (* TINY for debugging: *)
  (* let hid_dim = 2 in *)
  (* let hid_dim = 4 in *)
  let data_len = 3 * 5 * 1024 in
  (* TINY for debugging: *)
  (* let data_len = 3 * 4 in *)
  (* let data_len = 3 * 8 in *)
  let flat_len = data_len / 2 in
  (* Note: [minibatch_size = batch_size / num_streams] is the actual per-device batch used. *)
  (* let epochs = 400 in *)
  let epochs = 100 in
  (* let epochs = 50 in *)
  (* TINY for debugging: *)
  (* let epochs = 3 in *)
  (* let epochs = 2 in *)
  (* let epochs = 1 in *)
  (* let init_lr = 0.1 in *)
  let init_lr = 0.01 in
  let moons_config = Datasets.Half_moons.Config.{ noise_range = 0.1; seed = Some seed } in
  let moons_coordinates, moons_labels =
    Datasets.Half_moons.generate ~config:moons_config ~len:flat_len ()
  in
  let moons_flat_ndarray = Ir.Ndarray.as_array Ir.Ops.Double moons_coordinates in
  let moons_classes_ndarray = Ir.Ndarray.as_array Ir.Ops.Double moons_labels in
  let moons_flat = TDSL.rebatch ~l:"moons_flat" moons_flat_ndarray () in
  let moons_classes = TDSL.rebatch ~l:"moons_classes" moons_classes_ndarray () in

  let init_time = Time_now.nanoseconds_since_unix_epoch () in
  let%op mlp x =
    { w4 }
    * relu
        ({ b3; o = [ hid_dim_3 ] }
        + { w3 }
          * relu
              ({ b2; o = [ hid_dim_2 ] }
              + ({ w2 } * relu ({ b1; o = [ hid_dim_1 ] } + ({ w1 } * x)))))
  in
  (* TINY for debugging: *)
  (* let%op mlp x = { w2 } * relu({ b1; o = [ hid_dim ] } + ({ w1 } * x)) in *)
  let%op loss_fn ~output ~expectation = relu (!..1 - (expectation *. output)) in
  let start_time = ref None in
  let weight_decay = 0.0002 in
  Backends.Schedulers.sync_suggested_num_streams := num_streams;
  let module Backend =
    (val Backends.fresh_backend ~backend_name ~config:Train.BT.Most_parallel_streams ())
  in
  Stdlib.Format.printf "Initial backend global debug info: %a\n%!" Sexp.pp_hum
  @@ Backend.get_global_debug_info ();
  let per_batch_callback ~at_batch ~at_step ~learning_rate ~batch_loss ~epoch_loss =
    Stdio.printf "Batch=%d, step=%d, lr=%f, batch loss=%f, epoch loss=%f\n%!" at_batch at_step
      learning_rate batch_loss epoch_loss;
    if Option.is_none !start_time then start_time := Some (Time_now.nanoseconds_since_unix_epoch ())
  in
  (* Tn.print_accessible_headers (); *)
  let per_epoch_callback ~at_step ~at_epoch ~learning_rate ~epoch_loss =
    (* if at_epoch % 10 = 9 then *)
    Stdio.printf "Epoch=%d, step=%d, lr=%f, epoch loss=%f\n%!" at_epoch at_step learning_rate
      epoch_loss
  in

  let {
    Train.inputs;
    outputs;
    model_result;
    infer_callback;
    rev_batch_losses;
    rev_epoch_losses;
    learning_rates;
    used_memory;
  } =
    Train.example_train_loop ~seed ~batch_size ~init_lr ~max_num_streams:num_streams ~data_len
      ~epochs ~inputs:moons_flat ~outputs:moons_classes ~model:mlp ~loss_fn ~weight_decay
      ~per_batch_callback ~per_epoch_callback ~per_epoch_debug_streams:false
      (module Backend)
      ()
  in
  let points = Tn.points_2d ~xdim:0 ~ydim:1 inputs.value in
  let classes = Tn.points_1d ~xdim:0 outputs.value in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  Stdio.print_endline "\n******** mlp_result **********";
  Train.printf_tree ~with_grad:false ~depth:9 model_result;
  Stdio.printf "\n********\n%!";
  (* Ir.Tnode.print_accessible_headers (); *)
  let callback (x, y) = Float.((infer_callback [| x; y |]).(0) >= 0.) in
  let%track3_sexp plot_moons () =
    (* [%log_level 0; *)
    PrintBox_utils.plot
    (* TINY for debugging: *)
    (* ~small:true *)
      ~as_canvas:true
      [
        Scatterplot { points = points1; content = PrintBox.line "#" };
        Scatterplot { points = points2; content = PrintBox.line "%" };
        Boundary_map
          { content_false = PrintBox.line "."; content_true = PrintBox.line "*"; callback };
      ]
    (* ] *)
  in
  Stdio.printf "\nHalf-moons scatterplot and decision boundary:\n%!";
  PrintBox_text.output Stdio.stdout @@ plot_moons ();
  Stdio.printf "\nBatch Log-loss:\n%!";
  let plot_loss =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"batch log loss"
      [
        Line_plot
          {
            points =
              Array.of_list_rev_map rev_batch_losses ~f:Float.(fun x -> max (log 0.00003) (log x));
            content = PrintBox.line "-";
          };
      ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nEpoch Log-loss:\n%!";
  let plot_loss =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"epoch log loss"
      [
        Line_plot
          {
            points = Array.of_list_rev_map rev_epoch_losses ~f:Float.log;
            content = PrintBox.line "-";
          };
      ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nLearning rate:\n%!";
  let plot_lr =
    PrintBox_utils.plot ~x_label:"step" ~y_label:"learning rate"
      [ Line_plot { points = Array.of_list_rev learning_rates; content = PrintBox.line "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_lr;
  let final_time = Time_now.nanoseconds_since_unix_epoch () in
  (* TODO: include init time in benchmarks? *)
  let init_time_in_sec =
    Int63.(to_float @@ (Option.value_exn ~here:[%here] !start_time - init_time)) /. 1000_000_000.
  in
  let time_in_sec =
    Int63.(to_float @@ (final_time - Option.value_exn ~here:[%here] !start_time)) /. 1000_000_000.
  in
  Stdio.printf "\nTime in sec: %f\n%!" time_in_sec;
  let result =
    PrintBox_utils.Benchmark
      {
        bench_title;
        time_in_sec;
        mem_in_bytes = used_memory;
        result_label = "init time in sec, min loss, last loss";
        result =
          [%sexp_of: float * float * float]
            ( init_time_in_sec,
              List.reduce_exn rev_epoch_losses ~f:Float.min,
              List.hd_exn rev_epoch_losses );
      }
  in
  Stdio.printf "\n\n%!";
  (* Ir.Tnode.print_accessible_headers (); *)
  Stdlib.Format.printf "Final backend global debug info: %a\n%!" Sexp.pp_hum
  @@ Backend.get_global_debug_info ();
  result

let _suspend () =
  ignore
  @@ classify_moons ~seed:0 ~on_device:true ~inlining_cutoff:3 ~num_streams:8 ~batch_size:16
       ~backend_name:"gccjit" ~value_prec:CDSL.single ~grad_prec:CDSL.double ()

let _cuda_benchmarks =
  List.concat_map [ 1; 3; 6; 12; 16; 20 (* 32; 64 *) ] ~f:(fun num_streams ->
      List.concat_map
        [
          (* TINY for debugging: *)
          (* 3 * 2 *)
          3 * 5 * 16 (* ; 3 * 5 * 32; 3 * 5 * 64 *);
        ]
        ~f:(fun batch_size ->
          List.concat_map [ (* 0; 1; 2; *) 3 ] ~f:(fun inlining_cutoff ->
              List.concat_map [ (* 1; 3; *) 7 (* *) ] ~f:(fun seed ->
                  List.concat_map [ (* "gccjit" ; "cuda";"sync_cc" ; *) "multicore_cc" ]
                    ~f:(fun backend_name ->
                      List.concat_map [ (* CDSL.double; *) CDSL.single (* ; CDSL.half *) ]
                        ~f:(fun value_prec ->
                          [
                            classify_moons ~seed ~on_device:true ~inlining_cutoff ~num_streams
                              ~batch_size ~backend_name ~value_prec ~grad_prec:value_prec;
                          ]))))))

let _cuda_parallel_benchmarks =
  List.concat_map
    [
      (* 1; *)
      2;
      (* 3; 4; 5; 6; 8; 10; 12; 16; 20 *)
      (* 32; 64 *)
    ] ~f:(fun num_streams ->
      List.concat_map
        [
          (* TINY for debugging: *)
          3 * 4
          (* 3 * 5 * 16 *)
          (* ; 3 * 5 * 32 *);
        ]
        ~f:(fun batch_size ->
          List.concat_map [ 0 (* 1; 2; 3 *) ] ~f:(fun inlining_cutoff ->
              List.concat_map [ (* 1; 3; *) 7 (* *) ] ~f:(fun seed ->
                  List.concat_map [ (* "gccjit"; "cuda" ;"multicore_cc"; *) "sync_cc" ]
                    ~f:(fun backend_name ->
                      List.concat_map [ (* CDSL.double; *) CDSL.single (* ; CDSL.half *) ]
                        ~f:(fun value_prec ->
                          [
                            classify_moons ~seed ~on_device:true ~inlining_cutoff ~num_streams
                              ~batch_size ~backend_name ~value_prec ~grad_prec:value_prec;
                          ]))))))

let _mem_benchmarks =
  List.concat_map [ 1; 3; 6; 12; 16 (* ; 20; 32; 64 *) ] ~f:(fun num_streams ->
      List.concat_map
        [
          (* TINY for debugging: *)
          (* 3 * 2 *)
          3 * 5 * 16 (* ; 3 * 5 * 32; 3 * 5 * 64 *);
        ]
        ~f:(fun batch_size ->
          List.concat_map [ 0; (* 1; 2; *) 3 ] ~f:(fun inlining_cutoff ->
              List.concat_map [ (* 1; 3; *) 7 (* *) ] ~f:(fun seed ->
                  List.concat_map [ (* "gccjit" ; *) "multicore_cc"; "cuda" ]
                    ~f:(fun backend_name ->
                      List.concat_map [ (* CDSL.double; *) CDSL.single; CDSL.half ]
                        ~f:(fun value_prec ->
                          [
                            classify_moons ~seed ~on_device:true ~inlining_cutoff ~num_streams
                              ~batch_size ~backend_name ~value_prec ~grad_prec:value_prec;
                          ]))))))

(* let time_of = function PrintBox_utils.Benchmark { time_in_sec; _ } -> time_in_sec let nth_best
   nth bench = let results = List.init 5 ~f:(fun seed -> bench ~seed ()) in let sorted = List.sort
   results ~compare:(fun r1 r2 -> Float.compare (time_of r1) (time_of r2)) in List.nth_exn sorted
   (nth - 1) *)

let fixed_seed_search seed =
  classify_moons ~seed ~on_device:true ~inlining_cutoff:3 ~num_streams:1 ~batch_size:20
    ~backend_name:"cuda" ~value_prec:CDSL.single ~grad_prec:CDSL.single ()

let _suspended () =
  List.init 20 ~f:fixed_seed_search |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout

(* let () = List.map benchmarks ~f:(nth_best 2) |> PrintBox_utils.table |> PrintBox_text.output
   Stdio.stdout *)

let _suspended () =
  [
    classify_moons ~seed:7 ~on_device:true ~inlining_cutoff:0 ~num_streams:3 ~batch_size:240
      ~backend_name:"multicore_cc" ~value_prec:CDSL.half ~grad_prec:CDSL.half ();
  ]
  |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout

let benchmark benchmarks =
  List.map benchmarks ~f:(fun bench -> bench ())
  |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout

let _suspended () = benchmark _cuda_parallel_benchmarks
let () = benchmark _cuda_benchmarks

open Base
open Ocannl
module Nd = Arrayjit.Ndarray
module Ops = Arrayjit.Ops
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Utils = Arrayjit.Utils
module Debug_runtime = Utils.Debug_runtime

[%%global_debug_log_level Nothing]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

let classify_moons ~seed ~on_device ~inlining_cutoff ~num_devices ~batch_size ~backend_name precision () =
  [%track_sexp
    let _debug : string = "started" in
    (fun (started : unit) -> started) ()];
  (* ignore seed; *)
  let bench_title =
    [%string
      "seed %{seed#Int}, inline %{inlining_cutoff#Int}, parallel %{num_devices#Int}, batch \
       %{batch_size#Int}, backend %{backend_name}, prec %{Ops.prec_string precision}"]
  in
  Stdio.printf "\n*** %s ***\n%!" bench_title;
  CDSL.virtualize_settings.enable_device_only <- on_device;
  CDSL.virtualize_settings.max_visits <- inlining_cutoff;
  Tensor.default_value_prec := precision;
  Tensor.default_grad_prec := precision;
  Utils.settings.output_debug_files_in_run_directory <- true;
  Utils.settings.debug_log_from_routines <- true;
  Random.init (* seed *) 0;
  (* let hid_2_3 = 8 in let hid_4_5 = 4 in *)
  let hid_dim = 16 in
  (* let hid_dim = 4 in *)
  let len = batch_size * 20 in
  (* let epochs = 100 in *)
  (* let epochs = 10 in *)
  let epochs = 5 in
  let init_lr = 0.1 in
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
  let moons_flat ~b = TDSL.init_const ~l:"moons_flat" ~b ~o:[ 2 ] moons_flat in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes ~b = TDSL.init_const ~l:"moons_classes" ~b ~o:[ 1 ] moons_classes in

  let init_time = Time_now.nanoseconds_since_unix_epoch () in
  (* * let%op mlp x = "b6" 1 + "w6" * ?/("b4" hid_4_5 + "w4" * ?/("b2" hid_2_3 + ("w2" * ?/("b1" 16 + ("w1" *
     x))) + "b3" hid_2_3 + ("w3" * ?/(b2 + (w2 * ?/(b1 + (w1 * x)))))) + ("b5" hid_4_5 + ("w5" * ?/(b4 + (w4 *
     ?/(b3 + (w3 * ?/(b2 + (w2 * ?/(b1 + (w1 * x))))))))))) in * *)
  let%op mlp x = "b3" 1 + ("w3" * ?/("b2" hid_dim + ("w2" * ?/("b1" hid_dim + ("w1" * x))))) in
  let%op loss_fn ~output ~expectation = ?/(!..1 - (expectation *. output)) in
  let start_time = ref None in
  let weight_decay = 0.0002 in
  let backend = Train.fresh_backend ~backend_name () in
  let per_batch_callback ~at_batch:_ ~at_step:_ ~learning_rate:_ ~batch_loss:_ ~epoch_loss:_ =
    if Option.is_none !start_time then start_time := Some (Time_now.nanoseconds_since_unix_epoch ())
  in
  (* Tn.print_accessible_headers (); *)
  let per_epoch_callback ~at_step ~at_epoch ~learning_rate ~epoch_loss =
    Stdio.printf "Epoch=%d, step=%d, lr=%f, epoch loss=%f\n%!" at_epoch at_step learning_rate epoch_loss
  in
  let inputs, outputs, model_result, infer_callback, batch_losses, epoch_losses, learning_rates =
    Train.example_train_loop ~name:"moons" ~seed ~batch_size ~init_lr ~num_devices ~data_len:len ~epochs
      ~inputs:moons_flat ~outputs:moons_classes ~model:mlp ~loss_fn ~weight_decay ~per_batch_callback
      ~per_epoch_callback backend ()
  in
  let points = Tensor.value_2d_points ~xdim:0 ~ydim:1 inputs in
  let classes = Tensor.value_1d_points ~xdim:0 outputs in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  Stdio.print_endline "\n******** mlp_result **********";
  Tensor.print_tree ~with_id:true ~with_grad:false ~depth:9 model_result;
  Stdio.printf "\n********\n%!";
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
  Stdio.printf "\nBatch Log-loss:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"batch log loss"
      [
        Line_plot
          {
            points = Array.of_list_rev_map batch_losses ~f:Float.(fun x -> max (log 0.00003) (log x));
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
  Stdio.printf "\nLearning rate:\n%!";
  let plot_lr =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"learning rate"
      [ Line_plot { points = Array.of_list_rev learning_rates; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_lr;
  let final_time = Time_now.nanoseconds_since_unix_epoch () in
  (* TODO: include init time in benchmarks? *)
  let init_time_in_sec = Int63.(to_float @@ (Option.value_exn !start_time - init_time)) /. 1000_000_000. in
  let time_in_sec = Int63.(to_float @@ (final_time - Option.value_exn !start_time)) /. 1000_000_000. in
  Stdio.printf "\nTime in sec: %f\n%!" time_in_sec;
  let result =
    PrintBox_utils.Benchmark
      {
        bench_title;
        time_in_sec;
        (* FIXME: implement total mem assessment. *)
        mem_in_bytes = 0;
        result_label = "init time in sec, min loss, last loss";
        result =
          [%sexp_of: float * float * float]
            (init_time_in_sec, List.reduce_exn epoch_losses ~f:Float.min, List.hd_exn epoch_losses);
      }
  in
  Stdio.printf "\n\n%!";
  result

let _suspended () =
  ignore
  @@ classify_moons ~seed:0 ~on_device:true ~inlining_cutoff:3 ~num_devices:8 ~batch_size:16
       ~backend_name:"gccjit" CDSL.single ()

let benchmarks =
  List.concat_map [ 0; 1; 2; 3 ] ~f:(fun inlining_cutoff ->
      List.concat_map [ 1; 2; 5; 8; 10; 16 (* ; 20 *) ] ~f:(fun num_devices ->
          List.concat_map [ 120; 160 (* ; 320; 640; 1280 *) ] ~f:(fun batch_size ->
              List.concat_map [ 0; 1 (* ; 2; 3; 4 *) ] ~f:(fun seed ->
                  List.concat_map [ "gccjit" (* *; "cuda" *) ] ~f:(fun backend_name ->
                      [
                        classify_moons ~seed ~on_device:true ~inlining_cutoff ~num_devices ~batch_size
                          ~backend_name CDSL.single;
                      ])))))

(* let time_of = function PrintBox_utils.Benchmark { time_in_sec; _ } -> time_in_sec let nth_best nth bench =
   let results = List.init 5 ~f:(fun seed -> bench ~seed ()) in let sorted = List.sort results ~compare:(fun
   r1 r2 -> Float.compare (time_of r1) (time_of r2)) in List.nth_exn sorted (nth - 1) *)

let fixed_seed_search seed =
  classify_moons ~seed ~on_device:true ~inlining_cutoff:3 ~num_devices:1 ~batch_size:20 ~backend_name:"cuda"
    CDSL.single ()

let _suspended () =
  List.init 20 ~f:fixed_seed_search |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout

(* let () = List.map benchmarks ~f:(nth_best 2) |> PrintBox_utils.table |> PrintBox_text.output
   Stdio.stdout *)

let benchmark () =
  List.map benchmarks ~f:(fun bench -> bench ()) |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout

let () = benchmark ()

open Base
open Ocannl
module Nd = Arrayjit.Ndarray
module Ops = Arrayjit.Ops
module IDX = Arrayjit.Indexing.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Arrayjit.Low_level.CDSL
module Utils = Arrayjit.Utils
module Debug_runtime = Utils.Debug_runtime

[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

let classify_moons ~random_seed ~on_device ~inlining_cutoff ~num_devices ~batch ~backend_name precision () =
  [%track_sexp
    let _debug : string = "started" in
    (fun (started : unit) -> started) ()];
  let module Backend = (val Train.fresh_backend ~backend_name () : Arrayjit.Backends.Backend) in
  let num_devices = min num_devices @@ Backend.num_devices () in
  let devices = Array.init num_devices ~f:(fun ordinal -> Backend.get_device ~ordinal) in
  let contexts = Array.map devices ~f:Backend.init in
  (* ignore random_seed; *)
  let bench_title =
    [%string
      "seed %{random_seed#Int}, inline %{inlining_cutoff#Int}, parallel %{num_devices#Int}, batch \
       %{batch#Int}, backend %{Backend.name}, prec %{Ops.prec_string precision}"]
  in
  Stdio.printf "\n*** %s ***\n%!" bench_title;
  CDSL.virtualize_settings.enable_device_only <- on_device;
  CDSL.virtualize_settings.max_visits <- inlining_cutoff;
  Tensor.default_value_prec := precision;
  Tensor.default_grad_prec := precision;
  let open Tensor.O in
  let debug_batches = true in
  Utils.settings.output_debug_files_in_run_directory <- true;
  Utils.settings.debug_log_jitted <- true;
  Random.init (* random_seed *) 0;
  (* Utils.settings.fixed_state_for_init <- Some random_seed; *)
  (* let hid_2_3 = 8 in
     let hid_4_5 = 4 in *)
  (* let hid_dim = 16 in *)
  let hid_dim = 4 in
  (* let len = 64 * 32 in *)
  let len = 64 * 8 in
  let minibatch = batch / num_devices in
  let n_batches = 2 * len / minibatch in
  (* let epochs = 100 in *)
  (* let epochs = 10 in *)
  (* let epochs = 1000 in *)
  let epochs = 1 in
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
  let moons_flat = TDSL.init_const ~l:"moons_flat" ~b:[ n_batches; minibatch ] ~o:[ 2 ] moons_flat in
  let moons_classes = TDSL.init_const ~l:"moons_classes" ~b:[ n_batches; minibatch ] ~o:[ 1 ] moons_classes in

  (* *
     let%op mlp x =
       "b6" 1
       + "w6"
         * ?/("b4" hid_4_5
             + "w4"
               * ?/("b2" hid_2_3
                   + ("w2" * ?/("b1" 16 + ("w1" * x)))
                   + "b3" hid_2_3
                   + ("w3" * ?/(b2 + (w2 * ?/(b1 + (w1 * x))))))
             + ("b5" hid_4_5 + ("w5" * ?/(b4 + (w4 * ?/(b3 + (w3 * ?/(b2 + (w2 * ?/(b1 + (w1 * x)))))))))))
     in
     * *)
  let%op mlp x = "b3" 1 + ("w3" * ?/("b2" hid_dim + ("w2" * ?/("b1" hid_dim + ("w1" * x))))) in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  let steps = epochs * n_batches in
  let%op learning_rate = 0.003 *. (!..steps - !@step_n) /. !..steps in
  let%op moons_input = moons_flat @| batch_n in
  let%op moons_class = moons_classes @| batch_n in
  let%op margin_loss = ?/(1 - (moons_class *. mlp moons_input)) in
  let%op scalar_loss = (margin_loss ++ "...|... => 0") /. !..batch in

  [%track_sexp
    let batch_losses = ref [] in
    let epoch_losses = ref [] in
    let batch_log_losses = ref [] in
    let epoch_log_losses = ref [] in
    let learning_rates = ref [] in
    let min_loss = ref Float.infinity in
    let max_loss = ref 0.0 in
    let last_loss = ref Float.infinity in
    let start_time = Time_now.nanoseconds_since_unix_epoch () in

    Train.set_hosted learning_rate.value;
    Train.set_hosted scalar_loss.value;
    let weight_decay : float = 0.0001 in
    let update = Train.grad_update ~setup_for_parallel:true scalar_loss in
    let sgd = Train.sgd_update ~learning_rate ~weight_decay update in
    let grad_updates = Array.map contexts ~f:(fun ctx -> Backend.jit ctx bindings update.fwd_bprop) in
    let sgd_update = Backend.jit grad_updates.(0).context bindings sgd in
    Train.all_host_to_device (module Backend) sgd_update.context scalar_loss;
    Train.all_host_to_device (module Backend) sgd_update.context learning_rate;
    let epoch_loss = ref 0. in
    let batch_ref = IDX.find_exn sgd_update.bindings batch_n in
    let step_ref = IDX.find_exn sgd_update.bindings step_n in
    let update =
      Train.parallel_update
        (module Backend)
        ~grad_updates ~sgd_update update
        ~post_sync:(fun ~num_synced_devices ->
          step_ref := !step_ref + num_synced_devices;
          assert (Backend.to_host sgd_update.context learning_rate.value);
          (* scalar_loss is not in the sgd_update context. *)
          assert (Backend.to_host grad_updates.(0).context scalar_loss.value);
          let loss = scalar_loss.@[0] in
          epoch_loss := !epoch_loss +. loss;
          batch_losses := loss :: !batch_losses;
          epoch_loss := !epoch_loss +. scalar_loss.@[0];
          if debug_batches then
            Stdio.printf "Batch=%d, step=%d, lr=%f, batch loss=%f, epoch loss=%f\n%!" !batch_ref !step_ref
              learning_rate.@[0] loss !epoch_loss)
    in
    for epoch = 1 to epochs do
      epoch_loss := 0.;
      update ();
      learning_rates := learning_rate.@[0] :: !learning_rates;
      epoch_losses := !epoch_loss :: !epoch_losses;
      min_loss := Float.min !min_loss !epoch_loss;
      max_loss := Float.max !max_loss !epoch_loss;
      epoch_log_losses := Float.log !epoch_loss :: !epoch_log_losses;
      Stdio.printf "Epoch=%d, step=%d, lr=%f, epoch loss=%f\n%!" epoch !step_ref learning_rate.@[0]
        !epoch_loss
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
          (* FIXME: implement total mem assessment. *)
          mem_in_bytes = 0;
          result_label = "min epoch loss, last epoch loss";
          result = [%sexp_of: float * float] (!min_loss, !last_loss);
        }
    in
    (* Utils.settings.with_debug <- false; *)
    (* Utils.settings.output_debug_files_in_run_directory <- false; *)
    let points = Tensor.value_2d_points ~xdim:0 ~ydim:1 moons_flat in
    let classes = Tensor.value_1d_points ~xdim:0 moons_classes in
    let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
    let%op mlp_result = mlp "point" in
    Train.set_on_host Volatile mlp_result.value;
    (* By using sgd_jitted.context here, we don't need to copy the parameters back to the host. *)
    let result_jitted =
      Backend.jit sgd_update.context IDX.empty @@ Block_comment ("moons infer", mlp_result.forward)
    in
    let callback ((x : float), (y : float)) : bool =
      Tensor.set_values point [| x; y |];
      (* For the gccjit backend, point is only on host, not on device. For cuda, this will be needed. *)
      ignore (Backend.from_host result_jitted.context point.value : bool);
      Train.run result_jitted;
      Backend.await devices.(0);
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
        [ Line_plot { points = Array.of_list_rev !epoch_losses; pixel = "-" } ]
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
        [ Line_plot { points = Array.of_list_rev !epoch_log_losses; pixel = "-" } ]
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
    result]

let () =
  [%track_sexp
    let _debug : string = "entry" in
    (fun (entry : unit) -> entry) ()];
  ignore
  @@ classify_moons ~random_seed:3 ~on_device:true ~inlining_cutoff:3 ~num_devices:8 ~batch:16
       ~backend_name:"gccjit" CDSL.single ()

let benchmarks =
  List.concat_map [ (* 0; 3; 5 *) 3 ] ~f:(fun inlining_cutoff ->
      List.concat_map [ 1; (* 2; 4; 8; 10; *) 16 (* ; 20 *) ] ~f:(fun num_devices ->
          List.concat_map [ (* 1; 8; *) 32; 64; 128 (* ; 256; 512; 1024 *) ] ~f:(fun batch ->
              List.concat_map [ 0; 1; 2 (* ; 3; 4 *) ] ~f:(fun random_seed ->
                  List.concat_map [ "gccjit" (* *; "cuda" *) ] ~f:(fun backend_name ->
                      [
                        classify_moons ~random_seed ~on_device:true ~inlining_cutoff ~num_devices ~batch
                          ~backend_name CDSL.single;
                      ])))))

(*
let time_of = function PrintBox_utils.Benchmark { time_in_sec; _ } -> time_in_sec
let nth_best nth bench =
  let results = List.init 5 ~f:(fun random_seed -> bench ~random_seed ()) in
  let sorted = List.sort results ~compare:(fun r1 r2 -> Float.compare (time_of r1) (time_of r2)) in
  List.nth_exn sorted (nth - 1)
*)

let fixed_seed_search random_seed =
  classify_moons ~random_seed ~on_device:true ~inlining_cutoff:3 ~num_devices:1 ~batch:20 ~backend_name:"cuda"
    CDSL.single ()

let _suspended () =
  List.init 20 ~f:fixed_seed_search |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout

(*
let () =
  List.map benchmarks ~f:(nth_best 2) |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout
*)

let benchmark () =
  List.map benchmarks ~f:(fun bench -> bench ()) |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout

let _suspended () = benchmark ()

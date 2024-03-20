open Base
open Ocannl
module Tn = Arrayjit.Tnode
module IDX = Arrayjit.Indexing.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Arrayjit.Low_level.CDSL
module Utils = Arrayjit.Utils

let num_devices = 20
(* let num_devices = 10 *)
(* let num_devices = 1 *)

let experiment ~seed ~use_builtin_weight_decay () =
  Random.init 0;
  Utils.settings.with_debug <- true;
  Utils.settings.output_debug_files_in_run_directory <- true;
  (* Utils.settings.debug_log_jitted <- true; *)
  let hid_dim = 16 in
  (* let hid_dim = 4 in *)
  let len = 600 in
  (* let len = 30 in *)
  let init_lr = 0.1 in
  (* let batch_size = 120 in *)
  let batch_size = 20 in
  let minibatch_size = batch_size / num_devices in
  let n_batches = 2 * len / minibatch_size in
  let epochs = 20 in
  (* let epochs = 5 in *)
  let steps = epochs * n_batches in
  Utils.settings.fixed_state_for_init <- Some seed;
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
  let moons_flat = TDSL.init_const ~l:"moons_flat" ~b:[ n_batches; minibatch_size ] ~o:[ 2 ] moons_flat in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes =
    TDSL.init_const ~l:"moons_classes" ~b:[ n_batches; minibatch_size ] ~o:[ 1 ] moons_classes
  in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  let%op mlp x = "b3" + ("w3" * ?/("b2" hid_dim + ("w2" * ?/("b1" hid_dim + ("w1" * x))))) in
  (* let%op mlp x = "b" + ("w" * x) in *)
  let%op learning_rate = !.init_lr *. (!..steps - !@step_n) /. !..steps in
  let%op moons_input = moons_flat @| batch_n in
  let%op moons_class = moons_classes @| batch_n in
  let losses = ref [] in
  let log_losses = ref [] in
  let learning_rates = ref [] in
  let%op margin_loss = ?/(1 - (moons_class *. mlp moons_input)) in
  (* We don't need a regression loss formula thanks to weight_decay built into the sgd_update computation. *)
  let scalar_loss, weight_decay =
    if use_builtin_weight_decay then
      let%op scalar_loss = (margin_loss ++ "...|... => 0") /. !..batch_size in
      (scalar_loss, 0.0002)
    else
      let%op ssq w = (w **. 2) ++ "...|...->... => 0" in
      let reg_loss = List.map ~f:ssq [ w1; w2; w3; b1; b2; b3 ] |> List.reduce_exn ~f:TDSL.O.( + ) in
      (* let reg_loss = List.map ~f:ssq [ w; b ] |> List.reduce_exn ~f:TDSL.O.( + ) in *)
      let%op scalar_loss = ((margin_loss ++ "...|... => 0") /. !..batch_size) + (0.0001 *. reg_loss) in
      (scalar_loss, 0.0)
  in
  (* So that we can inspect them. *)
  Train.set_hosted learning_rate.value;
  let update = Train.grad_update ~setup_for_parallel:true scalar_loss in
  let sgd = Train.sgd_update ~learning_rate ~weight_decay update in

  let module Backend = (val Train.fresh_backend ()) in
  let num_devices = min num_devices @@ Backend.num_devices () in
  let devices = Array.init num_devices ~f:(fun ordinal -> Backend.get_device ~ordinal) in
  let contexts = Array.map devices ~f:Backend.init in
  let grad_update = Backend.prejit ~shared:true bindings update.fwd_bprop in
  let grad_updates = Array.map contexts ~f:(fun ctx -> Backend.jit ctx grad_update) in
  let sgd_update = Backend.jit_code grad_updates.(0).context bindings sgd in
  Train.all_host_to_device (module Backend) sgd_update.context scalar_loss;
  Train.all_host_to_device (module Backend) sgd_update.context learning_rate;
  let open Tensor.O in
  let epoch_loss = ref 0. in
  let step_ref = IDX.find_exn sgd_update.bindings step_n in
  (* let batch_ref = IDX.find_exn sgd_update.bindings batch_n in *)
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
        losses := loss :: !losses
        (* Stdio.printf "Batch=%d, step=%d, lr=%f, batch loss=%f, epoch loss=%f\n%!" !batch_ref !step_ref
           learning_rate.@[0] loss !epoch_loss *))
  in
  (* Tn.print_accessible_headers (); *)
  for epoch = 0 to epochs - 1 do
    epoch_loss := 0.;
    update ();
    learning_rates := learning_rate.@[0] :: !learning_rates;
    Stdio.printf "Epoch=%d, step=%d, lr=%f, epoch loss=%f\n%!" epoch !step_ref learning_rate.@[0] !epoch_loss
  done;
  let points = Tensor.value_2d_points ~xdim:0 ~ydim:1 moons_flat in
  let classes = Tensor.value_1d_points ~xdim:0 moons_classes in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  let%op mlp_result = mlp "point" in
  Train.set_on_host Volatile mlp_result.value;
  (* By using jitted.context here, we don't need to copy the parameters back to the host. *)
  let result_jitted =
    Backend.jit_code sgd_update.context IDX.empty @@ Block_comment ("moons infer", mlp_result.forward)
  in
  Stdio.print_endline "\n******** mlp_result **********";
  Tensor.print_tree ~with_id:true ~with_grad:false ~depth:9 mlp_result;
  Stdio.printf "\n********\n%!";
  let callback (x, y) =
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
  Stdio.printf "Loss:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"loss"
      [ Line_plot { points = Array.of_list_rev !losses; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;

  Stdio.printf "Log-loss, for better visibility:\n%!";
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
  PrintBox_text.output Stdio.stdout plot_lr

let () = experiment ~seed:0 ~use_builtin_weight_decay:true ()

let _suspended () =
  for seed = 0 to 19 do
    Stdio.printf "\n*************** EXPERIMENT SEED %d ******************\n%!" seed;
    experiment ~seed ~use_builtin_weight_decay:true ()
  done

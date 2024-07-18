open Base
open Ocannl
module Tn = Arrayjit.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Utils = Arrayjit.Utils
module Rand = Arrayjit.Rand.Lib
module Debug_runtime = Utils.Debug_runtime

let demo () =
  Rand.init 0;
  Utils.enable_runtime_debug ();
  (* Utils.settings.debug_log_from_routines <- true; *)
  let hid_dim = 16 in
  let len = 300 in
  let batch_size = 20 in
  let n_batches = 2 * len / batch_size in
  let epochs = 75 in
  let steps = epochs * n_batches in
  let weight_decay = 0.0002 in
  Utils.settings.fixed_state_for_init <- Some 4;
  let noise () = Rand.float_range (-0.1) 0.1 in

  let%op mlp x = "b3" + ("w3" * ?/("b2" hid_dim + ("w2" * ?/("b1" hid_dim + ("w1" * x))))) in

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
  let moons_flat = TDSL.init_const ~l:"moons_flat" ~o:[ 2 ] moons_flat in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes = TDSL.init_const ~l:"moons_classes" ~o:[ 1 ] moons_classes in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  let%op moons_input = moons_flat @| batch_n in
  let%op moons_class = moons_classes @| batch_n in
  let%op margin_loss = ?/(1 - (moons_class *. mlp moons_input)) in
  let%op scalar_loss = (margin_loss ++ "...|... => 0") /. !..batch_size in
  let update = Train.grad_update scalar_loss in
  let%op learning_rate = 0.1 *. (!..steps - !@step_n) /. !..steps in
  Train.set_hosted learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate ~weight_decay update in

  let epoch_loss = ref 0. in

  let module Backend = (val Train.fresh_backend ~backend_name:"cc" ()) in
  let device = Backend.(new_virtual_device @@ get_device ~ordinal:0) in
  let ctx = Backend.init device in
  let routine = Backend.(link ctx @@ compile bindings (Seq (update.fwd_bprop, sgd))) in
  Train.all_host_to_device (module Backend) routine.context scalar_loss;
  Train.all_host_to_device (module Backend) routine.context learning_rate;
  let open Operation.At in
  let step_ref = IDX.find_exn routine.bindings step_n in
  let batch_ref = IDX.find_exn routine.bindings batch_n in
  step_ref := 0;
  let%track_this_sexp _train_loop : unit =
    for epoch = 0 to epochs - 1 do
      for batch = 0 to n_batches - 1 do
        batch_ref := batch;
        Train.run routine;
        assert (Backend.to_host routine.context learning_rate.value);
        assert (Backend.to_host routine.context scalar_loss.value);
        Backend.await device;
        epoch_loss := !epoch_loss +. scalar_loss.@[0];
        Int.incr step_ref
      done;
      Stdio.printf "Epoch %d, lr=%f, epoch loss=%f\n%!" epoch learning_rate.@[0] !epoch_loss;
      epoch_loss := 0.
    done
  in

  let points = Tensor.value_2d_points ~xdim:0 ~ydim:1 moons_flat in
  let classes = Tensor.value_1d_points ~xdim:0 moons_classes in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  let%op mlp_result = mlp "point" in
  Train.set_on_host Changed_on_devices mlp_result.value;
  let result_routine =
    Backend.(
      link routine.context @@ compile IDX.empty @@ Block_comment ("moons infer", mlp_result.forward))
  in
  let callback (x, y) =
    Tensor.set_values point [| x; y |];
    assert (Backend.from_host result_routine.context point.value);
    Train.run result_routine;
    assert (Backend.to_host result_routine.context mlp_result.value);
    Backend.await device;
    Float.(mlp_result.@[0] >= 0.)
  in

  let%track_this_sexp _plotting : unit =
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
    PrintBox_text.output Stdio.stdout plot_moons
  in
  Backend.unsafe_cleanup ~unsafe_shutdown:true ()

let () = demo ()

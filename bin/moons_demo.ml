open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Asgns = Ir.Assignments
module Rand = Ir.Rand.Lib
(* module Half_moons = Datasets.Half_moons *)

let _get_local_debug_runtime = Utils.get_local_debug_runtime

let demo () =
  let seed = 3 in
  Rand.init seed;
  Utils.settings.fixed_state_for_init <- Some seed;
  Utils.settings.output_debug_files_in_build_directory <- true;
  (* Utils.enable_runtime_debug (); *)
  let hid_dim = 16 in
  let len = 512 in
  let batch_size = 32 in
  let epochs = 75 in
  (* Utils.settings.debug_log_from_routines <- true; *)
  (* TINY for debugging: *)
  (* let hid_dim = 2 in let len = 16 in let batch_size = 2 in let epochs = 2 in *)
  let n_batches = 2 * len / batch_size in
  let steps = epochs * n_batches in
  let weight_decay = 0.0002 in

  let%op mlp x = "b3" + ("w3" * relu ("b2" hid_dim + ("w2" * relu ("b1" hid_dim + ("w1" * x))))) in

  let config = Datasets.Half_moons.Config.{ noise_range = 0.1; seed = Some seed } in
  let moons_coordinates, moons_labels = Datasets.Half_moons.generate ~config ~len () in
  let moons_flat = TDSL.rebatch ~l:"moons_flat" (Ir.Ndarray.as_array Ir.Ops.Double moons_coordinates) in
  let moons_classes = TDSL.rebatch ~l:"moons_classes" (Ir.Ndarray.as_array Ir.Ops.Double moons_labels) in

  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  let%op moons_input = moons_flat @| batch_n in
  let%op moons_class = moons_classes @| batch_n in

  let%op margin_loss = relu (1 - (moons_class *. mlp moons_input)) in
  let%op scalar_loss = (margin_loss ++ "...|... => 0") /. !..batch_size in

  let update = Train.grad_update scalar_loss in
  let%op learning_rate = 0.1 *. (!..steps - !@step_n) /. !..steps in
  Train.set_hosted learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate ~weight_decay scalar_loss in

  let module Backend = (val Backends.fresh_backend ~backend_name:"metal" ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let init_params = Tensor.init_params scalar_loss in
  let init = Backend.link ctx @@ Backend.compile ctx.optimize_ctx IDX.empty init_params in
  let ctx = init.context in
  let routine = Train.to_routine (module Backend) ctx bindings (Asgns.sequence [ update; sgd ]) in

  let points = Tn.points_2d ~xdim:0 ~ydim:1 moons_flat.value in
  let classes = Tn.points_1d ~xdim:0 moons_classes.value in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  let plot_moons =
    PrintBox_utils.plot ~as_canvas:true
      [
        Scatterplot { points = points1; content = PrintBox.line "#" };
        Scatterplot { points = points2; content = PrintBox.line "+" };
      ]
  in
  Stdio.printf "\nHalf-moons scatterplot:\n%!";
  PrintBox_text.output Stdio.stdout plot_moons;
  Stdio.print_endline "\n";

  let open Operation.At in
  let step_ref = IDX.find_exn routine.bindings step_n in
  let batch_ref = IDX.find_exn routine.bindings batch_n in
  let epoch_loss = ref 0. in
  step_ref := 0;
  Train.run init;
  let%track_sexp _train_loop : unit =
    for epoch = 0 to epochs - 1 do
      for batch = 0 to n_batches - 1 do
        batch_ref := batch;
        Utils.capture_stdout_logs @@ fun () ->
        Train.run routine;
        epoch_loss := !epoch_loss +. scalar_loss.@[0];
        Int.incr step_ref
      done;
      Stdio.printf "Epoch %d, lr=%f, epoch loss=%f\n%!" epoch learning_rate.@[0] !epoch_loss;
      epoch_loss := 0.
    done
  in

  let%cd mlp_result = mlp "point" in
  Train.set_on_host mlp_result.value;
  let result_routine =
    Train.to_routine
      (module Backend)
      routine.context IDX.empty
      [%cd
        ~~("moons infer";
           mlp_result.forward)]
  in
  let callback (x, y) =
    Tn.set_values point.value [| x; y |];
    Utils.capture_stdout_logs @@ fun () ->
    Train.run result_routine;
    Float.(mlp_result.@[0] >= 0.)
  in

  let%track_sexp _plotting : unit =
    let plot_moons =
      PrintBox_utils.plot ~as_canvas:true
        [
          Scatterplot { points = points1; content = PrintBox.line "#" };
          Scatterplot { points = points2; content = PrintBox.line "+" };
          Boundary_map
            { content_false = PrintBox.line "."; content_true = PrintBox.line "*"; callback };
        ]
    in
    Stdio.printf "\nHalf-moons scatterplot and decision boundary:\n%!";
    PrintBox_text.output Stdio.stdout plot_moons;
    Stdio.print_endline "\n"
  in
  ()

let () = demo ()

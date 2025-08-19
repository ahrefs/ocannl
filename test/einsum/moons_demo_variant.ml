open Base
open Ocannl
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Asgns = Ir.Assignments
module Tn = Ir.Tnode

module type Backend = Ir.Backend_intf.Backend

let () =
  (* Very similar to micrograd_demo.ml, but with an einsum shape inference corner case. *)
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let open Operation.At in
  let len = 200 in
  let batch_size = 10 in
  let n_batches = 2 * len / batch_size in
  let moons_config = Datasets.Half_moons.Config.{ noise_range = 0.1; seed = Some 5 } in
  let moons_coordinates, moons_labels =
    Datasets.Half_moons.generate_single_prec ~config:moons_config ~len ()
  in
  let moons_flat_ndarray = Ir.Ndarray.as_array Ir.Ops.Single moons_coordinates in
  let moons_classes_ndarray = Ir.Ndarray.as_array Ir.Ops.Single moons_labels in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  let moons_flat = TDSL.rebatch ~l:"moons_flat" moons_flat_ndarray () in
  let moons_classes = TDSL.rebatch ~l:"moons_classes" moons_classes_ndarray () in
  (* let%op mlp x = 0.5 + ("w3" * relu ("b2" 16 + ("w2" * relu ("b1" 16 + ("w1" * x))))) in *)
  let%op mlp x = "w2" * relu ("b1" 16 + ("w1" * x)) in
  (* Don't decay the learning rate too quickly, it behaves better than in the original. *)
  let%op moons_input = moons_flat @| batch_n in
  (* THIS IS THE SPECIFIC SHAPE INFERENCE ASPECT OF THE TEST. *)
  let%cd _ = moons_input =: 0 ++ "i=>2|i" in
  let%op moons_class = moons_classes @| batch_n in
  let%cd _ = moons_class =: 0 ++ "i=>2|i" in
  let%op margin_loss = relu (1 - (moons_class *. mlp moons_input)) in
  (* We don't need a regression loss formula thanks to weight_decay built into the sgd_update
     computation. *)
  let weight_decay = 0.0001 in
  let%op scalar_loss = (margin_loss ++ "...|... => 0") /. !..batch_size in
  let update = Train.grad_update scalar_loss in
  let%op learning_rate = 0.1 *. ((2 *. !..len) - !@step_n) /. !..len in
  Train.set_hosted learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate ~weight_decay scalar_loss in
  let ctx = Train.init_params (module Backend) bindings scalar_loss in
  let sgd_routine =
    Train.to_routine (module Backend) ctx bindings (Asgns.sequence [ update; sgd ])
  in
  (* Skipping over the training loop, not needed for the test. *)
  let points = Tn.points_2d ~xdim:0 ~ydim:1 moons_flat.value in
  let classes = Tn.points_1d ~xdim:0 moons_classes.value in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  let%cd mlp_result = mlp "point" in
  let result_routine =
    Train.to_routine
      (module Backend)
      sgd_routine.context IDX.empty
      [%cd
        ~~("moons infer";
           mlp_result.forward)]
  in
  let callback (x, y) =
    Tn.set_values point.value [| x; y |];
    Train.run result_routine;
    Float.(mlp_result.@[0] >= 0.)
  in
  let _plot_moons =
    PrintBox_utils.plot ~as_canvas:true ~size:(5, 5)
      [
        Scatterplot { points = points1; content = PrintBox.line "#" };
        Scatterplot { points = points2; content = PrintBox.line "%" };
        Boundary_map
          { content_false = PrintBox.line "."; content_true = PrintBox.line "*"; callback };
      ]
  in
  (* PrintBox_text.output Stdio.stdout _plot_moons; *)
  (* Stdio.printf "Losses: %.4g, %.4g\n%!" losses.(epochs / 2) losses.(epochs - 1); *)
  Tn.print_accessible_headers ();

  (* Testing how the syntax extension %op creates labels for the resulting tensors: *)
  Stdio.printf "mlp_result's name: %s\n%!" @@ Tensor.debug_name mlp_result;
  (* Note: mlp_result is not included in the resulting tensor's label, because the identifier label
     does not propagate across function calls. *)
  Stdio.printf "(mlp moons_input) name: %s\n%!"
  @@ Tensor.debug_name
  @@
  match margin_loss.children with
  | [
   {
     subtensor =
       { children = [ _; { subtensor = { children = [ _; { subtensor; _ } ]; _ }; _ } ]; _ };
     _;
   };
  ] ->
      subtensor
  | _ -> assert false

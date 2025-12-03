(** Circle counting training test using synthetic dataset.

    This test trains a model to count circles in synthetic images.

    {2 Known Issues with conv2d in Training}

    When attempting to use [Nn_blocks.conv2d] with SGD training, several shape inference issues were
    encountered:

    1. {b max_pool2d row variable mismatch}: [max_pool2d] uses [..c..] for channel row variable,
    while [conv2d] uses [..oc..] for output channels. When composing [max_pool2d (conv2d x)], the
    shape inference fails with "incompatible stride" errors because the row variables don't unify.

    2. {b Unconstrained output channels}: The [conv2d] kernel's output channels [..oc..] are not
    automatically constrained by the network structure. This causes "You forgot to specify the
    hidden dimension(s)" errors during SGD update compilation, as the gradient tensors cannot
    determine their shapes.

    3. {b Workaround}: Use an MLP instead - OCANNL's matrix multiplication handles
    multi-dimensional inputs automatically without explicit flattening.

    These issues suggest that [conv2d] may need:
    - An explicit [out_channels] parameter to constrain output shape
    - Consistent row variable naming with pooling operations
    - Or specialized handling in [Train.grad_update] for conv kernels *)

open Base
open Ocannl
open Stdio
module Tn = Ir.Tnode
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments

let () =
  let seed = 42 in
  Utils.settings.fixed_state_for_init <- Some seed;
  Tensor.unsafe_reinitialize ();

  (* Configuration for circle dataset *)
  let image_size = 16 in
  let max_circles = 3 in
  let config =
    Datasets.Circles.Config.
      { image_size; max_radius = 4; min_radius = 2; max_circles; seed = Some seed }
  in

  (* Generate training data *)
  let batch_size = 8 in
  let total_samples = batch_size * 20 in
  let n_batches = total_samples / batch_size in

  printf "Generating %d circle counting images...\n%!" total_samples;
  let images_data, labels_data =
    Datasets.Circles.generate_single_prec ~config ~len:total_samples ()
  in
  printf "Dataset generated: images shape [%d; %d; %d; 1], labels shape [%d; 1]\n%!" total_samples
    image_size image_size total_samples;

  (* Convert to tensors *)
  let images_ndarray = Ir.Ndarray.as_array Ir.Ops.Single images_data in
  let labels_ndarray = Ir.Ndarray.as_array Ir.Ops.Single labels_data in

  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in

  let images = TDSL.rebatch ~l:"images" images_ndarray () in
  let labels = TDSL.rebatch ~l:"labels" labels_ndarray () in

  (* Simple model for regression: flatten input and apply MLP. NOTE: Using conv2d with training has
     shape inference issues: 1. max_pool2d has row variable mismatch with conv2d (..c.. vs ..oc..)
     2. conv2d output channels aren't constrained, causing "hidden dimension" errors We use a simple
     MLP as a workaround for now. *)
  let%op model x =
    (* Two hidden layers - specify input dims for w1 *)
    let h1 = relu (({ w1 } * x) + { b1 = 0.; o = [ 32 ] }) in
    let h2 = relu (({ w2 } * h1) + { b2 = 0.; o = [ 16 ] }) in
    (* Output layer for regression *)
    ({ w_out } * h2) + { b_out = 0. }
  in

  (* Batch input/output *)
  let%op batch_images = images @| batch_n in
  let%op batch_labels = labels @| batch_n in

  (* Forward pass and MSE loss *)
  let%op predictions = model batch_images in
  let%op diff = predictions - batch_labels in
  let%op mse_loss = ((diff *. diff) ++ "...|... => 0") /. !..batch_size in

  (* Training setup *)
  let epochs = 10 in
  let total_steps = epochs * n_batches in
  let update = Train.grad_update mse_loss in
  let%op learning_rate = 0.01 *. ((2 *. !..total_steps) - !@step_n) /. !..total_steps in
  Train.set_hosted learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate mse_loss in

  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings mse_loss in
  let sgd_routine = Train.to_routine ctx bindings (Asgns.sequence [ update; sgd ]) in

  let step_ref = IDX.find_exn (Context.bindings sgd_routine) step_n in
  step_ref := 0;

  printf "\nStarting training for %d epochs (%d steps)...\n%!" epochs total_steps;

  let open Operation.At in
  for epoch = 1 to epochs do
    let epoch_loss = ref 0. in
    Train.sequential_loop (Context.bindings sgd_routine) ~f:(fun () ->
        Train.run ctx sgd_routine;
        epoch_loss := !epoch_loss +. mse_loss.@[0];
        Int.incr step_ref);
    printf "Epoch %d: avg loss = %.4f\n%!" epoch (!epoch_loss /. Float.of_int n_batches)
  done;

  printf "\nTraining complete!\n%!"

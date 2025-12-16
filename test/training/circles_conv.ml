(** Circle counting training test using synthetic dataset.

    This test trains a model to classify images by the number of circles they contain. Uses
    cross-entropy loss for classification.

    {2 Known Issues with conv2d in Training}

    When attempting to use [Nn_blocks.lenet] with SGD training, a shape inference issue was
    encountered:

    1. {b max_pool2d row variable mismatch}: [max_pool2d] uses [..c..] for channel row variable,
    while [conv2d] uses [..oc..] for output channels. When composing [max_pool2d (conv2d x)], the
    shape inference fails with "incompatible stride" errors because the row variables don't unify.

    {b Workaround}: Use an MLP instead - OCANNL's matrix multiplication handles multi-dimensional
    inputs automatically without explicit flattening. *)

open Base
open Ocannl
open Stdio
module Tn = Ir.Tnode
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments

let lenet = Nn_blocks.lenet
let softmax = Nn_blocks.softmax

let () =
  let seed = 42 in
  Utils.settings.fixed_state_for_init <- Some seed;
  Tensor.unsafe_reinitialize ();

  (* Use scaled initialization to prevent activation explosion. Default uniform1() in [0,1] causes
     logits to grow to millions. *)
  TDSL.default_param_init := PDSL.xavier ~scale_sq:0.06 TDSL.O.uniform1;

  (* Configuration for circle dataset *)
  let image_size = 16 in
  let max_circles = 3 in
  let num_classes = max_circles in
  (* Classes: 1, 2, 3 circles -> indices 0, 1, 2 *)
  let config =
    Datasets.Circles.Config.
      { image_size; max_radius = 4; min_radius = 2; max_circles; seed = Some seed }
  in

  (* Generate training data *)
  let batch_size = 8 in
  let total_samples = batch_size * 20 in
  let n_batches = total_samples / batch_size in

  printf "Generating %d circle counting images (%d classes)...\n%!" total_samples num_classes;
  let images_data, labels_data =
    Datasets.Circles.generate_single_prec ~config ~len:total_samples ()
  in
  printf "Dataset generated: images shape [%d; %d; %d; 1], labels shape [%d; 1]\n%!" total_samples
    image_size image_size total_samples;

  (* Convert labels to 0-based indices for one-hot encoding *)
  let labels_array = Bigarray.array2_of_genarray labels_data in
  let labels_list =
    List.init total_samples ~f:(fun i ->
        (* Labels are 1 to max_circles, convert to 0-based *)
        Int.of_float (Bigarray.Array2.get labels_array i 0) - 1)
  in

  (* Convert to tensors *)
  let images_ndarray = Ir.Ndarray.as_array Ir.Ops.Single images_data in
  let labels_one_hot = Nn_blocks.one_hot_of_int_list ~num_classes labels_list in

  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in

  let images = TDSL.rebatch ~l:"images" images_ndarray () in

  (* Batch input/output *)
  let%op batch_images = images @| batch_n in
  let%op batch_labels = labels_one_hot @| batch_n in

  let%op logits = lenet () ~train_step:None batch_images in

  (* Softmax and cross-entropy loss *)
  let%op probs = softmax ~spec:"...|v" () logits in
  (* Sum the probability mass for the correct class *)
  let%op correct_prob = (probs *. batch_labels) ++ "...|... => ...|0" in
  (* Cross-entropy: -log(p) for each sample, then average *)
  let%op sample_loss = neg (log correct_prob) in
  let%op batch_loss = (sample_loss ++ "...|... => 0") /. !..batch_size in

  (* Training setup *)
  let epochs = 1000 in
  let total_steps = epochs * n_batches in
  let update = Train.grad_update batch_loss in
  let%op learning_rate = 0.01 *. ((1.2 *. !..total_steps) - !@step_n) /. !..total_steps in
  Train.set_hosted learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate batch_loss in

  (* Ensure we can read loss on host *)
  Train.set_hosted batch_loss.value;

  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings batch_loss in
  let sgd_routine = Train.to_routine ctx bindings (Asgns.sequence [ update; sgd ]) in

  let step_ref = IDX.find_exn (Context.bindings sgd_routine) step_n in
  step_ref := 0;

  printf "\nStarting training for %d epochs (%d steps)...\n%!" epochs total_steps;

  let open Operation.At in
  for epoch = 1 to epochs do
    let epoch_loss = ref 0. in
    Train.sequential_loop (Context.bindings sgd_routine) ~f:(fun () ->
        Train.run ctx sgd_routine;
        (* printf "batch_loss = %.4f\n%!" batch_loss.@[0]; *)
        epoch_loss := !epoch_loss +. batch_loss.@[0];
        Int.incr step_ref);
    if epoch % 10 = 0 || epoch = epochs then
      printf "Epoch %d: avg loss = %.2f\n%!" epoch (!epoch_loss /. Float.of_int n_batches)
  done;

  printf "\nTraining complete!\n%!"

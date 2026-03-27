(** CIFAR-10 image classification using LeNet-style CNN.

    Demonstrates: loading real-world RGB data, [int8] to [float] conversion, CNN training on
    3-channel images with cross-entropy loss, and held-out test evaluation.

    Uses [Conv_data.load_cifar10] which downloads the binary distribution of CIFAR-10 from U Toronto
    (the [Dataprep.Cifar10] loader is broken because it downloads the Python pickle version).

    {2 Regression mode vs full-run mode}

    By default this uses small subsets for fast regression testing via [dune runtest]. For full
    training that targets the issue acceptance criteria (>60% test accuracy):

    - Set [num_train = 50000], [num_test = 10000], [epochs = 50], [batch_size = 100]
    - Use wider channels: [out_channels1 = 32], [out_channels2 = 64]

    Full-run manual validation:
    {v OCANNL_BACKEND=sync_cc dune exec test/training/cifar_conv.exe v}

    after editing the constants below. Expect ~30-60 minutes on CPU. *)

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

  (* Xavier init -- same as circles_conv.ml and mnist_conv.ml.
     CIFAR data is centered to [-0.5, 0.5] in Conv_data.cifar_images_to_float32 so that
     all-positive Xavier-uniform weights produce zero-centered conv outputs. *)
  TDSL.default_param_init := PDSL.xavier ~scale_sq:0.06 TDSL.O.uniform1;

  (* --- Configuration ---
     Regression mode (default): fast, loose thresholds, used by dune runtest.
     Full-run mode: change these constants for issue acceptance targets (>60% accuracy). *)
  let num_train = 2000 in
  (* Full-run: 50000 *)
  let num_test = 1000 in
  (* Full-run: 10000 *)
  let batch_size = 50 in
  (* Full-run: 100 *)
  let epochs = 100 in
  (* Full-run: 50 *)
  let num_classes = 10 in
  let out_channels1 = 6 in
  (* Full-run: 32 *)
  let out_channels2 = 16 in
  (* Full-run: 64 *)

  (* --- Data Loading ---
     Conv_data.load_cifar10 downloads the binary distribution from U Toronto on first call,
     caching to ~/.cache/ocaml-dataprep/datasets/cifar-10-bin/.
     Suppress stdout messages during loading (contain machine-specific paths). *)
  printf "Loading CIFAR-10 dataset...\n%!";
  let load_quietly f =
    Stdlib.flush_all ();
    let old_stdout = Unix.dup Unix.stdout in
    let devnull = Unix.openfile "/dev/null" [ Unix.O_WRONLY ] 0o644 in
    Unix.dup2 devnull Unix.stdout;
    Unix.close devnull;
    Exn.protect ~f ~finally:(fun () ->
        Stdlib.flush_all ();
        Unix.dup2 old_stdout Unix.stdout;
        Unix.close old_stdout)
  in
  let (train_images_raw, train_labels_raw), (test_images_raw, test_labels_raw) =
    load_quietly (fun () -> Conv_data.load_cifar10 ())
  in

  let n_batches = num_train / batch_size in
  let n_test_batches = num_test / batch_size in

  (* Convert int8 images to float32 [N; 32; 32; 3], centered to [-0.5, 0.5] *)
  let train_images_f32 = Conv_data.cifar_images_to_float32 train_images_raw in
  let train_images_f32 = Conv_data.take_prefix_images ~n:num_train train_images_f32 in
  let train_labels_list = Conv_data.labels_to_int_list train_labels_raw in
  let train_labels_list = List.take train_labels_list num_train in

  let test_images_f32 = Conv_data.cifar_images_to_float32 test_images_raw in
  let test_images_f32 = Conv_data.take_prefix_images ~n:num_test test_images_f32 in
  let test_labels_list = Conv_data.labels_to_int_list test_labels_raw in
  let test_labels_list = List.take test_labels_list num_test in
  let test_labels_arr = Array.of_list test_labels_list in

  printf "Training on %d samples, testing on %d samples, batch_size=%d\n%!" num_train num_test
    batch_size;

  (* Wrap training data as OCANNL tensors *)
  let images_ndarray = Ir.Ndarray.as_array Ir.Ops.Single train_images_f32 in
  let labels_one_hot = Nn_blocks.one_hot_of_int_list ~num_classes train_labels_list in

  (* --- Training Model Setup --- *)
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in

  let images = TDSL.rebatch ~l:"images" images_ndarray () in
  let%op batch_images = images @| batch_n in
  let%op batch_labels = labels_one_hot @| batch_n in

  (* LeNet with wider channels for CIFAR's 3-channel RGB input.
     Store the model function so we can reuse it (with shared parameters) for eval. *)
  let model = lenet ~out_channels1 ~out_channels2 () in
  let%op logits = model ~train_step:None batch_images in

  (* Softmax cross-entropy loss. Epsilon prevents log(0) when softmax underflows. *)
  let%op probs = softmax ~spec:"...|v" () logits in
  let%op correct_prob = (probs *. batch_labels) ++ "...|... => ...|0" in
  let%op sample_loss = neg (log (correct_prob + 1e-7)) in
  let%op batch_loss = (sample_loss ++ "...|... => 0") /. !..batch_size in

  (* --- Training Setup --- *)
  let total_steps = epochs * n_batches in
  let update = Train.grad_update batch_loss in
  let%op learning_rate = 0.01 *. ((1.2 *. !..total_steps) - !@step_n) /. !..total_steps in
  Train.set_hosted learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate batch_loss in
  Train.set_hosted batch_loss.value;

  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings batch_loss in
  let sgd_routine = Train.to_routine ctx bindings (Asgns.sequence [ update; sgd ]) in

  let step_ref = IDX.find_exn (Context.bindings sgd_routine) step_n in
  step_ref := 0;

  printf "\nStarting training for %d epochs (%d steps)...\n%!" epochs total_steps;

  (* --- Training Loop --- *)
  let open Operation.At in
  for epoch = 1 to epochs do
    let epoch_loss = ref 0. in
    Train.sequential_loop (Context.bindings sgd_routine) ~f:(fun () ->
        Train.run ctx sgd_routine;
        epoch_loss := !epoch_loss +. batch_loss.@[0];
        Int.incr step_ref);
    if epoch % 10 = 0 then
      printf "Epoch %d: avg loss = %.2f\n%!" epoch (!epoch_loss /. Float.of_int n_batches)
  done;

  (* --- Test-Set Evaluation (forward-only) ---
     Build a separate eval graph that reuses trained lenet parameters. Following the pattern from
     moons_demo.ml and bigram_mlp.ml: apply the model to new inputs and compile a forward-only
     routine. No gradients, no backprop, no SGD updates -- parameters are not mutated. *)
  printf "\nEvaluating on %d test samples...\n%!" num_test;

  let test_images_ndarray = Ir.Ndarray.as_array Ir.Ops.Single test_images_f32 in
  let test_labels_one_hot = Nn_blocks.one_hot_of_int_list ~num_classes test_labels_list in
  let test_images_t = TDSL.rebatch ~l:"images" test_images_ndarray () in

  let eval_batch_n, eval_bindings =
    IDX.get_static_symbol ~static_range:n_test_batches IDX.empty
  in
  let%op eval_batch_images = test_images_t @| eval_batch_n in
  let _eval_batch_labels = [%op test_labels_one_hot @| eval_batch_n] in

  (* Reuse the trained lenet function -- shares parameters with training graph *)
  let%op eval_logits = model ~train_step:None eval_batch_images in
  let%op eval_probs = softmax ~spec:"...|v" () eval_logits in

  (* Test loss: cross-entropy on held-out data *)
  let%op eval_correct_prob = (eval_probs *. _eval_batch_labels) ++ "...|... => ...|0" in
  let%op eval_sample_loss = neg (log (eval_correct_prob + 1e-7)) in
  let%op eval_batch_loss = (eval_sample_loss ++ "...|... => 0") /. !..batch_size in

  Train.set_hosted eval_batch_loss.value;
  Train.set_hosted eval_probs.value;

  (* Forward-only routine via %cd .forward -- no grad_update, no sgd_update *)
  let eval_routine =
    Train.to_routine (Context.context sgd_routine) eval_bindings
      [%cd ~~("eval forward"; eval_batch_loss.forward)]
  in

  (* Compute test loss and accuracy across all test batches *)
  let test_loss = ref 0. in
  let correct = ref 0 in
  let batch_idx = ref 0 in
  Train.sequential_loop (Context.bindings eval_routine) ~f:(fun () ->
      Train.run ctx eval_routine;
      test_loss := !test_loss +. eval_batch_loss.@[0];
      (* Read all probability values for this batch as a flat array,
         then compute argmax per sample in OCaml. *)
      let flat_probs = Tn.get_values eval_probs.value in
      for s = 0 to batch_size - 1 do
        let max_c = ref 0 in
        let max_v = ref Float.neg_infinity in
        for c = 0 to num_classes - 1 do
          let v = flat_probs.(s * num_classes + c) in
          if Float.(v > !max_v) then (max_v := v; max_c := c)
        done;
        let label_idx = (!batch_idx * batch_size) + s in
        if !max_c = test_labels_arr.(label_idx) then Int.incr correct
      done;
      Int.incr batch_idx);
  let avg_test_loss = !test_loss /. Float.of_int n_test_batches in
  let accuracy = Float.of_int !correct /. Float.of_int num_test *. 100. in
  printf "Test loss = %.2f\n%!" avg_test_loss;
  printf "Test accuracy = %.1f%% (%d/%d)\n%!" accuracy !correct num_test;
  (* Regression-mode threshold checks (conservative, must pass on small subsets).
     CIFAR-10 is harder than MNIST; random chance = 10%, so 15% demonstrates some learning. *)
  printf "Test loss below 2.3 = %b\n%!" Float.(avg_test_loss < 2.3);
  printf "Test accuracy above 15%% = %b\n%!" Float.(accuracy > 15.);

  printf "\nTraining complete!\n%!"

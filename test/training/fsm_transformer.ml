open Base
open Ocannl
open Stdio
module Tn = Ir.Tnode
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments

(* === FSM Definition === *)

let num_states = 8

(* transition.(state).(input_bit) = next_state
   A fixed deterministic FSM with 8 states and binary input.  All states are reachable
   and every state has two distinct successors. *)
let transition =
  [|
    [| 3; 5 |];
    [| 6; 0 |];
    [| 7; 4 |];
    [| 1; 2 |];
    [| 5; 7 |];
    [| 0; 6 |];
    [| 4; 1 |];
    [| 2; 3 |];
  |]

let generate_sequence rng ~seq_len ~init_state =
  let states = Array.create ~len:seq_len 0 in
  states.(0) <- init_state;
  for i = 1 to seq_len - 1 do
    let bit = Random.State.int rng 2 in
    states.(i) <- transition.(states.(i - 1)).(bit)
  done;
  states

let generate_batch rng ~num_seqs ~seq_len =
  let eff_seq_len = seq_len - 1 in
  let inputs = Array.init num_seqs ~f:(fun _ -> Array.create ~len:eff_seq_len 0) in
  let targets = Array.init num_seqs ~f:(fun _ -> Array.create ~len:eff_seq_len 0) in
  for i = 0 to num_seqs - 1 do
    let init_state = Random.State.int rng num_states in
    let seq = generate_sequence rng ~seq_len ~init_state in
    for t = 0 to eff_seq_len - 1 do
      inputs.(i).(t) <- seq.(t);
      targets.(i).(t) <- seq.(t + 1)
    done
  done;
  (inputs, targets)

let seqs_to_flat_one_hot ~batch_size ~eff_seq_len (seqs : int array array) ~offset =
  let flat = Array.create ~len:(batch_size * eff_seq_len * num_states) 0. in
  for i = 0 to batch_size - 1 do
    for t = 0 to eff_seq_len - 1 do
      let base = ((i * eff_seq_len) + t) * num_states in
      flat.(base + seqs.(offset + i).(t)) <- 1.
    done
  done;
  flat

(* === Main === *)

let () =
  Utils.settings.fixed_state_for_init <- Some 3;
  Tensor.unsafe_reinitialize ();

  let seq_len = 9 in
  let eff_seq_len = seq_len - 1 in
  let batch_size = 32 in
  let num_train_seqs = 256 in
  let num_test_seqs = 32 in
  let d_model = 16 in
  let num_heads = 2 in
  let d_k = 8 in
  let d_v = 8 in
  let d_ff = 32 in
  let epochs = 40 in

  let train_rng = Random.State.make [| 42 |] in
  let train_inputs_arr, train_targets_arr =
    generate_batch train_rng ~num_seqs:num_train_seqs ~seq_len
  in
  let test_rng = Random.State.make [| 99 |] in
  let test_inputs_arr, _test_targets_arr =
    generate_batch test_rng ~num_seqs:num_test_seqs ~seq_len
  in

  let n_batches = num_train_seqs / batch_size in
  let step_n, bindings = IDX.get_static_symbol IDX.empty in
  let total_tokens = batch_size * eff_seq_len in

  let make_data_tensor label =
    let open Bigarray in
    let ga = Genarray.create Float32 c_layout [| batch_size; eff_seq_len; num_states |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:If_needed ~label:[ label ]
      ~batch_dims:[ batch_size; eff_seq_len ] ~input_dims:[] ~output_dims:[ num_states ] ()
  in
  let input_batch = make_data_tensor "input_batch" in
  let target_batch = make_data_tensor "target_batch" in

  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ eff_seq_len ] ~i:[ eff_seq_len ] ~o:[]
      ~f:(function
        | [| s; t |] -> if s >= t then 1. else 0.
        | _ -> failwith "unexpected mask indices")
      ()
  in

  (* Model: decoder-only transformer (single block, no layer_norm).
     Layer norm is omitted because with recentered weights, it normalizes values to
     unit variance and kills the gradient signal.  Residual connections preserve
     gradients for this small model. *)
  let open Nn_blocks in
  let mha = multi_head_attention ~label:[ "mha" ] ~num_heads ~d_k ~d_v () in
  let ffn = Nn_blocks.mlp ~label:[ "ffn" ] ~hid_dims:[ d_ff ] () in
  let%op build_model () =
    fun ~train_step ~mask input ->
      let embedded = ({ w_embed; o = [ d_model ] } * input) + { pos_encoding } in
      let x1 = embedded + mha ~train_step ~mask embedded in
      let x2 = x1 + ffn x1 in
      { w_out } * x2
  in
  let model = build_model () in

  (* === Training computation === *)
  let train_logits = model ~train_step:(Some step_n) ~mask input_batch in
  let%op counts = exp train_logits in
  let%op probs = counts /. (counts ++ "...|... => ...|0") in
  let%op output_probs = (probs *. target_batch) ++ "...|... => ...|0" in
  let%op loss = neg (log output_probs) in
  let%op batch_loss = (loss ++ "...|... => 0") /. !..total_tokens in

  let update = Train.grad_update batch_loss in
  let steps = epochs * n_batches in
  let%op learning_rate = 1.0 *. ((1.5 *. !..steps) - !@step_n) /. !..steps in
  let sgd = Train.sgd_update ~learning_rate batch_loss in

  (* === Inference computation (forward-only, shares trained weights) ===
     Following the bigram_mlp.ml pattern: invoke the model a second time via %cd
     with a fresh input tensor.  This creates a separate forward graph that shares
     the trained weight parameters but has its own (unconsumed) forward code. *)
  let infer_input =
    let open Bigarray in
    let ga = Genarray.create Float32 c_layout [| num_test_seqs; eff_seq_len; num_states |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:Prohibit_grad ~label:[ "infer_input" ]
      ~batch_dims:[ num_test_seqs; eff_seq_len ] ~input_dims:[] ~output_dims:[ num_states ] ()
  in
  let%cd infer_logits = model ~train_step:None ~mask infer_input in
  let%cd infer_comp =
    ~~("fsm infer";
       infer_logits.forward)
  in

  (* === Compile === *)
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings batch_loss in
  Train.set_on_host input_batch.value;
  Train.set_on_host target_batch.value;
  (* Recenter all model parameters from uniform [0,1) to [-0.25, 0.25).
     OCANNL's default uniform1 init produces all-positive weights; through the
     transformer's Q*K^T attention scores this causes extreme values and exp overflow.
     Centered initialization (e.g. xavier/normal) is standard for transformers but
     not yet available as a built-in default_param_init in OCANNL, so we recenter
     post-init. *)
  Set.iter batch_loss.Tensor.params ~f:(fun p ->
      let tn = p.Tensor.value in
      Train.set_on_host tn;
      let vals = Tn.get_values tn in
      Array.iteri vals ~f:(fun i v -> vals.(i) <- 0.5 *. (v -. 0.5));
      Tn.set_values tn vals);
  Train.set_on_host infer_logits.value;
  Train.set_on_host infer_input.value;
  (* Compile the training routine.  This adds all training nodes (including the
     shared mask constant) to the context via Context.compile. *)
  let train_comp = Asgns.sequence [ update; sgd ] in
  Set.iter (snd @@ Asgns.collect_nodes_guess_output train_comp.Asgns.asgns) ~f:Train.set_hosted;
  let ctx, sgd_step = Context.compile ctx train_comp bindings in
  (* Compile the inference routine using the context from training compilation,
     which already contains the mask constant and all model weight buffers.
     This is forward-only: no backprop, no SGD update. *)
  Set.iter (snd @@ Asgns.collect_nodes_guess_output infer_comp.Asgns.asgns) ~f:Train.set_hosted;
  (* The mask constant is embedded in the training compilation but not in the inference
     compilation (because consume_forward_code builds embedded_nodes independently for
     each tensor).  Add mask to the inference comp's embedded_nodes so Context.compile
     treats it as an embedded constant rather than an uninitialized input. *)
  let infer_comp =
    { infer_comp with
      Asgns.embedded_nodes = Set.add infer_comp.Asgns.embedded_nodes mask.value }
  in
  let ctx, infer_routine = Context.compile ctx infer_comp IDX.empty in

  let open Operation.At in
  let step_ref = IDX.find_exn (Context.bindings sgd_step) step_n in
  Train.set_on_host batch_loss.value;

  (* === Training loop ===
     Per-token random baseline: ln(8) ≈ 2.08, epoch sum ≈ 2.08 * n_batches ≈ 16.6.
     Optimal loss for binary FSM: ln(2) ≈ 0.693 per token, epoch sum ≈ 5.5. *)
  let epoch_loss_limit_first = 16.0 in
  let epoch_loss_limit_mid = 8.0 in
  let epoch_loss_limit_last = 7.0 in
  for epoch = 0 to epochs - 1 do
    let epoch_loss = ref 0. in
    for batch = 0 to n_batches - 1 do
      let offset = batch * batch_size in
      Tn.set_values input_batch.value
        (seqs_to_flat_one_hot ~batch_size ~eff_seq_len train_inputs_arr ~offset);
      Tn.set_values target_batch.value
        (seqs_to_flat_one_hot ~batch_size ~eff_seq_len train_targets_arr ~offset);
      let ctx' = Context.run ctx sgd_step in
      ignore (ctx' : Context.t);
      epoch_loss := !epoch_loss +. batch_loss.@[0];
      Int.incr step_ref
    done;
    if epoch = 0 || epoch = epochs / 2 || epoch = epochs - 1 then (
      let limit =
        if epoch = 0 then epoch_loss_limit_first
        else if epoch = epochs / 2 then epoch_loss_limit_mid
        else epoch_loss_limit_last
      in
      printf "Epoch %d, loss below threshold=%b\n%!" epoch Float.(!epoch_loss < limit))
  done;

  (* === Held-out Evaluation (inference only, no gradient update) ===
     We evaluate "valid transition accuracy": whether the argmax-predicted state is
     one of the two valid successors for the current state.  This tests whether the
     model learned the FSM transition function (which two states are reachable from
     each state).  The binary input bit is intentionally unobserved, so exact-match
     prediction of the specific successor caps at ~50%, but a model that learned
     the transition relation achieves ~100% valid-transition accuracy.
     Random baseline: 2/8 = 25%.  Threshold: >= 90%. *)
  Tn.set_values infer_input.value
    (seqs_to_flat_one_hot ~batch_size:num_test_seqs ~eff_seq_len test_inputs_arr ~offset:0);
  let _ctx = Context.run ctx infer_routine in

  let correct = ref 0 in
  let total = num_test_seqs * eff_seq_len in
  for seq = 0 to num_test_seqs - 1 do
    for t = 0 to eff_seq_len - 1 do
      let predicted = ref 0 in
      let max_logit = ref Float.neg_infinity in
      for s = 0 to num_states - 1 do
        let logit = infer_logits.@{[| seq; t; s |]} in
        if Float.(logit > !max_logit) then (
          max_logit := logit;
          predicted := s)
      done;
      let current = test_inputs_arr.(seq).(t) in
      if !predicted = transition.(current).(0) || !predicted = transition.(current).(1) then
        Int.incr correct
    done
  done;
  let accuracy = Float.of_int !correct /. Float.of_int total in
  printf "Held-out valid-transition accuracy above 0.90=%b\n%!" Float.(accuracy >= 0.90)

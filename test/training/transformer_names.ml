open Base
open Ocannl
open Stdio
module Tn = Ir.Tnode
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments

(* === Configuration === *)

let ctx_len = 16
let eff_seq_len = ctx_len - 1
let d_model = 16
let num_heads = 2
let d_k = 8
let d_v = 8
let d_ff = 32
let vocab_size = Dataprep.Names.dict_size
let batch_size = 32
let epochs = 10
let pad_char = ' '
let pad_idx = Dataprep.Names.char_index pad_char
let bos_idx = Dataprep.Names.char_index '.'

(* === Data preparation === *)

(** Convert a name to a fixed-length integer sequence.
    "emma" -> [0; 5; 13; 13; 1; 0; 1; 1; ...] where 0='.' and 1=' ' (padding).
    Returns (input_indices, target_indices) for teacher forcing. *)
let name_to_sequences name =
  let chars = '.' :: (String.to_list name @ [ '.' ]) in
  let len = List.length chars in
  let padded =
    if len >= ctx_len then List.take chars ctx_len
    else chars @ List.init (ctx_len - len) ~f:(fun _ -> pad_char)
  in
  let indices = List.map padded ~f:Dataprep.Names.char_index in
  let input_indices = List.take indices eff_seq_len in
  let target_indices = List.tl_exn (List.take indices ctx_len) in
  (Array.of_list input_indices, Array.of_list target_indices)

let prepare_dataset () =
  let names = Dataprep.Names.read_names () in
  let num_names = List.length names in
  printf "Names loaded: %d\n%!" num_names;
  let pairs = List.map names ~f:name_to_sequences in
  let inputs = Array.of_list (List.map pairs ~f:fst) in
  let targets = Array.of_list (List.map pairs ~f:snd) in
  let num_examples = Array.length inputs in
  (* Round down to multiple of batch_size *)
  let num_examples = num_examples - (num_examples % batch_size) in
  printf "Training examples: %d\n%!" num_examples;
  (inputs, targets, num_examples)

let seqs_to_flat_one_hot (seqs : int array array) ~offset =
  let flat = Array.create ~len:(batch_size * eff_seq_len * vocab_size) 0. in
  for i = 0 to batch_size - 1 do
    for t = 0 to eff_seq_len - 1 do
      let base = ((i * eff_seq_len) + t) * vocab_size in
      flat.(base + seqs.(offset + i).(t)) <- 1.
    done
  done;
  flat

(* === Main === *)

let () =
  Utils.settings.fixed_state_for_init <- Some 3;
  Tensor.unsafe_reinitialize ();

  let train_inputs, train_targets, num_examples = prepare_dataset () in
  let n_batches = num_examples / batch_size in

  let step_n, bindings = IDX.get_static_symbol IDX.empty in
  let total_tokens = batch_size * eff_seq_len in

  (* === Data tensors === *)
  let make_data_tensor label =
    let open Bigarray in
    let ga = Genarray.create Float32 c_layout [| batch_size; eff_seq_len; vocab_size |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:If_needed ~label:[ label ]
      ~batch_dims:[ batch_size; eff_seq_len ] ~input_dims:[] ~output_dims:[ vocab_size ] ()
  in
  let input_batch = make_data_tensor "input_batch" in
  let target_batch = make_data_tensor "target_batch" in

  (* === Causal mask === *)
  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ eff_seq_len ] ~i:[ eff_seq_len ] ~o:[]
      ~f:(function
        | [| s; t |] -> if s >= t then 1. else 0.
        | _ -> failwith "unexpected mask indices")
      ()
  in

  (* === Model ===
     Decoder-only transformer: masked self-attention + FFN with residual connections.
     Layer norm is omitted for this small model to keep generated code compact and
     avoid the gradient signal issue noted in fsm_transformer.ml.
     Uses multi_head_attention with ~mask for causal masking. *)
  let open Nn_blocks in
  let mha = multi_head_attention ~label:[ "mha" ] ~num_heads ~d_k ~d_v () in
  let ffn = Nn_blocks.mlp ~label:[ "ffn" ] ~hid_dims:[ d_ff ] () in
  let%op build_model () =
    fun ~train_step ~mask input ->
      let embedded = ({ tok_embed; o = [ d_model ] } * input) + { pos_encoding } in
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
  let%op learning_rate = 0.01 *. ((1.5 *. !..steps) - !@step_n) /. !..steps in
  let sgd = Train.sgd_update ~learning_rate batch_loss in

  (* === Inference computation (forward-only, shares trained weights) === *)
  let infer_input =
    let open Bigarray in
    let ga = Genarray.create Float32 c_layout [| 1; eff_seq_len; vocab_size |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:Prohibit_grad ~label:[ "infer_input" ]
      ~batch_dims:[ 1; eff_seq_len ] ~input_dims:[] ~output_dims:[ vocab_size ] ()
  in
  let counter_n, infer_bindings = IDX.get_static_symbol IDX.empty in
  let%cd infer_logits = model ~train_step:None ~mask infer_input in
  let%cd infer_comp =
    ~~("names infer";
       infer_logits.forward;
       { dice } =: uniform_at !@counter_n)
  in

  (* === Compile === *)
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings batch_loss in
  Train.set_on_host input_batch.value;
  Train.set_on_host target_batch.value;
  (* Recenter all model parameters from uniform [0,1) to [-0.25, 0.25).
     OCANNL's default uniform1 init produces all-positive weights; through the
     transformer's Q*K^T attention scores this causes extreme values and exp overflow.
     Same mitigation as fsm_transformer.ml. *)
  Set.iter batch_loss.Tensor.params ~f:(fun p ->
      let tn = p.Tensor.value in
      Train.set_on_host tn;
      let vals = Tn.get_values tn in
      Array.iteri vals ~f:(fun i v -> vals.(i) <- 0.5 *. (v -. 0.5));
      Tn.set_values tn vals);
  Train.set_on_host infer_logits.value;
  Train.set_on_host infer_input.value;

  (* Compile training routine *)
  let train_comp = Asgns.sequence [ update; sgd ] in
  Set.iter (snd @@ Asgns.collect_nodes_guess_output train_comp.Asgns.asgns) ~f:Train.set_hosted;
  let ctx, sgd_step = Context.compile ctx train_comp bindings in

  (* Compile inference routine *)
  Set.iter (snd @@ Asgns.collect_nodes_guess_output infer_comp.Asgns.asgns) ~f:Train.set_hosted;
  let infer_comp =
    { infer_comp with
      Asgns.embedded_nodes = Set.add infer_comp.Asgns.embedded_nodes mask.value
    }
  in
  let ctx, infer_routine = Context.compile ctx infer_comp infer_bindings in

  let open Operation.At in
  let step_ref = IDX.find_exn (Context.bindings sgd_step) step_n in
  let counter_ref = IDX.find_exn (Context.bindings infer_routine) counter_n in
  counter_ref := 0;
  Train.set_on_host batch_loss.value;

  (* === Training loop ===
     Random baseline: ln(28) ≈ 3.33 per token, epoch sum ≈ 3.33 * n_batches.
     We check loss at first, middle, and last epochs. *)
  let epoch_loss_limit_first = 2.0 *. Float.of_int n_batches in
  let epoch_loss_limit_mid = 1.4 *. Float.of_int n_batches in
  let epoch_loss_limit_last = 1.3 *. Float.of_int n_batches in
  for epoch = 0 to epochs - 1 do
    let epoch_loss = ref 0. in
    for batch = 0 to n_batches - 1 do
      let offset = batch * batch_size in
      Tn.set_values input_batch.value (seqs_to_flat_one_hot train_inputs ~offset);
      Tn.set_values target_batch.value (seqs_to_flat_one_hot train_targets ~offset);
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

  (* === Autoregressive generation ===
     Generate names token-by-token, sampling from the model's output distribution.
     Uses the CDF-based sampling pattern from bigram_mlp.ml. *)
  let set_one_hot_seq context =
    let flat = Array.create ~len:(1 * eff_seq_len * vocab_size) 0. in
    for t = 0 to eff_seq_len - 1 do
      let base = t * vocab_size in
      flat.(base + context.(t)) <- 1.
    done;
    Tn.set_values infer_input.value flat
  in

  let gen_name () =
    let context = Array.create ~len:eff_seq_len pad_idx in
    context.(0) <- bos_idx;
    let rec aux pos =
      if pos >= eff_seq_len then
        (* Max length reached — extract what we have *)
        let name = Buffer.create 16 in
        for i = 1 to eff_seq_len - 1 do
          let c = List.nth_exn Dataprep.Names.letters_with_dot context.(i) in
          if not (Char.equal c '.' || Char.equal c ' ') then Buffer.add_char name c
        done;
        Buffer.contents name
      else begin
        set_one_hot_seq context;
        Int.incr counter_ref;
        let _ctx = Context.run ctx infer_routine in
        let dice_value = dice.@[0] in

        (* Compute softmax probabilities at position (pos-1) in the output
           (the model predicts token at position pos given input up to pos-1). *)
        let logits = Array.init vocab_size ~f:(fun v ->
            infer_logits.@{[| 0; pos - 1; v |]}) in
        let max_logit = Array.fold logits ~init:Float.neg_infinity ~f:Float.max in
        let exp_logits = Array.map logits ~f:(fun l -> Float.exp (l -. max_logit)) in
        let sum_exp = Array.fold exp_logits ~init:0. ~f:( +. ) in
        let probs = Array.map exp_logits ~f:(fun e -> e /. sum_exp) in

        (* CDF-based sampling *)
        let max_i = vocab_size - 1 in
        let rec sample i acc =
          if i >= max_i then i
          else
            let new_acc = acc +. probs.(i) in
            if Float.(new_acc > dice_value) then i
            else sample (i + 1) new_acc
        in
        let sampled_idx = sample 0 0. in
        let sampled_char = List.nth_exn Dataprep.Names.letters_with_dot sampled_idx in

        if Char.equal sampled_char '.' && pos > 1 then begin
          (* EOS — extract name from positions 1..pos-1 *)
          let name = Buffer.create 16 in
          for i = 1 to pos - 1 do
            let c = List.nth_exn Dataprep.Names.letters_with_dot context.(i) in
            if not (Char.equal c '.' || Char.equal c ' ') then Buffer.add_char name c
          done;
          Buffer.contents name
        end
        else begin
          context.(pos) <- sampled_idx;
          aux (pos + 1)
        end
      end
    in
    aux 1
  in

  (* Generate very few names because different hardware backends diverge quickly. *)
  let names = Array.init 3 ~f:(fun _ -> gen_name ()) in
  Array.iter names ~f:print_endline

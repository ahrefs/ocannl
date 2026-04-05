open Base
open Ocannl
open Stdio
open Bigarray
module Tn = Ir.Tnode
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments

(* === Hyperparameters === *)
let ctx_len = 16
let seq_len = ctx_len - 1
let vocab_size = Dataprep.Names.dict_size
let d_model = 32
let num_heads = 4
let d_k = d_model / num_heads
let d_v = d_k
let num_layers = 2
let d_ff = 64
let batch_size = 256
let epochs = 10

(* === Local data helpers === *)
let name_to_indices name =
  let chars = String.to_list name in
  0 :: (List.map chars ~f:Dataprep.Names.char_index @ [ 0 ])

let pad_or_truncate ~len seq =
  let n = List.length seq in
  if n >= len then List.take seq len else seq @ List.init (len - n) ~f:(fun _ -> 0)

let () =
  Utils.settings.fixed_state_for_init <- Some 3;
  Tensor.unsafe_reinitialize ();

  (* === Dataset construction === *)
  let names = Dataprep.Names.read_names () in
  let n_names = List.length names in
  Stdio.printf "names: %d\n%!" n_names;

  let all_seqs =
    Array.of_list
      (List.map names ~f:(fun name ->
           let indices = name_to_indices name in
           let true_len = min (List.length indices) ctx_len in
           let padded = pad_or_truncate ~len:ctx_len indices |> Array.of_list in
           (padded, true_len)))
  in
  let n_seqs_raw = Array.length all_seqs in
  let n_seqs =
    if n_seqs_raw % batch_size = 0 then n_seqs_raw
    else n_seqs_raw + batch_size - (n_seqs_raw % batch_size)
  in
  let n_batches = n_seqs / batch_size in
  Stdio.printf "sequences: %d, seq_len: %d, n_batches: %d\n%!" n_seqs seq_len n_batches;

  (* Build data as [n_batches; batch_size; seq_len; vocab_size] *)
  let input_arr =
    Genarray.create Float32 c_layout [| n_batches; batch_size; seq_len; vocab_size |]
  in
  let target_arr =
    Genarray.create Float32 c_layout [| n_batches; batch_size; seq_len; vocab_size |]
  in
  (* Loss mask: 1 for valid target positions, 0 for padding *)
  let loss_mask_arr =
    Genarray.create Float32 c_layout [| n_batches; batch_size; seq_len |]
  in
  (* Valid token count per batch for normalization *)
  let valid_count_arr = Genarray.create Float32 c_layout [| n_batches |] in
  Genarray.fill input_arr 0.;
  Genarray.fill target_arr 0.;
  Genarray.fill loss_mask_arr 0.;
  Genarray.fill valid_count_arr 0.;

  for b = 0 to n_batches - 1 do
    let valid_count = ref 0. in
    for s = 0 to batch_size - 1 do
      let seq_idx = (b * batch_size + s) % n_seqs_raw in
      let padded, true_len = all_seqs.(seq_idx) in
      let valid_positions = true_len - 1 in
      for pos = 0 to seq_len - 1 do
        Genarray.set input_arr [| b; s; pos; padded.(pos) |] 1.;
        Genarray.set target_arr [| b; s; pos; padded.(pos + 1) |] 1.;
        if pos < valid_positions then Genarray.set loss_mask_arr [| b; s; pos |] 1.
      done;
      valid_count := !valid_count +. Float.of_int valid_positions
    done;
    Genarray.set valid_count_arr [| b |] !valid_count
  done;

  let inputs =
    TDSL.reshape ~l:"inputs" ~b:[ n_batches; batch_size; seq_len ] ~o:[ vocab_size ]
      (Ir.Ndarray.as_array Ir.Ops.Single input_arr)
      ()
  in
  let targets =
    NTDSL.reshape ~l:"targets" ~b:[ n_batches; batch_size; seq_len ] ~o:[ vocab_size ]
      (Ir.Ndarray.as_array Ir.Ops.Single target_arr)
      ()
  in
  let loss_mask =
    NTDSL.reshape ~l:"loss_mask" ~b:[ n_batches; batch_size; seq_len ] ~o:[]
      (Ir.Ndarray.as_array Ir.Ops.Single loss_mask_arr)
      ()
  in
  let valid_counts =
    NTDSL.reshape ~l:"valid_counts" ~b:[ n_batches ] ~o:[]
      (Ir.Ndarray.as_array Ir.Ops.Single valid_count_arr)
      ()
  in

  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  let steps = epochs * n_batches in

  let%op input_batch = inputs @| batch_n in
  let%op target_batch = targets @| batch_n in
  let%op loss_mask_batch = loss_mask @| batch_n in
  let%op valid_count_batch = valid_counts @| batch_n in

  (* === Causal mask === *)
  let mask =
    NTDSL.init ~l:"causal_mask" ~prec:Ir.Ops.single ~b:[ seq_len ] ~i:[ seq_len ] ~o:[]
      ~f:(function
        | [| s; t |] -> if s >= t then 1. else 0.
        | _ -> failwith "unexpected mask indices")
      ()
  in

  (* === Model: outputs raw logits === *)
  let decoder =
    Nn_blocks.decoder_only ~label:[ "names" ] ~num_layers ~num_heads ~d_k ~d_v ~d_ff ()
  in

  let%op model () =
    fun ~train_step input ->
      let embedded = { w_embed; o = [ d_model ] } * input in
      let encoded = embedded + { pos_encoding } in
      let decoded = decoder ~train_step ~mask encoded in
      { w_out = 0. } * decoded
  in
  let model = model () in

  (* === Cross-entropy loss via numerically stable log-softmax === *)
  (* log_softmax(x) = x - max(x) - log(sum(exp(x - max(x)))) *)
  (* gradient w.r.t. logits = softmax(logits) - target, bounded in [-1, 1] *)
  let%op logits = model ~train_step:(Some step_n) input_batch in
  let%op max_logits = logits @^^ "...|... => ...|0" in
  let%op shifted = logits - max_logits in
  let%op log_sum_exp = log (exp shifted ++ "...|... => ...|0") in
  let%op log_probs = shifted - log_sum_exp in
  let%op nll_per_pos = neg ((target_batch *. log_probs) ++ "...|... => ...|0") in
  let%op masked_nll = nll_per_pos *. loss_mask_batch in
  let%op batch_loss = (masked_nll ++ "...|... => 0") /. valid_count_batch in

  (* Training *)
  let update = Train.grad_update batch_loss in
  let%op learning_rate = 0.1 *. ((1.5 *. !..steps) - !@step_n) /. !..steps in
  let sgd = Train.sgd_update ~learning_rate batch_loss in

  let ctx = Context.auto () in
  Train.set_on_host mask.Tensor.value;
  let ctx = Train.init_params ctx bindings batch_loss in
  (* Explicitly initialize the causal mask so it's available for both training and inference. *)
  let ctx = Context.init_from_host_deprecated ctx mask.Tensor.value in
  (* Center parameters around zero for more stable training with SGD *)
  Set.iter batch_loss.Tensor.params ~f:(fun p ->
      let values = Tn.get_values p.Tensor.value in
      let scaled = Array.map values ~f:(fun v -> (v -. 0.5) *. 0.2) in
      Tn.set_values p.Tensor.value scaled);
  let sgd_step = Train.to_routine ctx bindings (Asgns.sequence [ update; sgd ]) in

  let open Operation.At in
  let batch_ref = IDX.find_exn (Context.bindings sgd_step) batch_n in
  let step_ref = IDX.find_exn (Context.bindings sgd_step) step_n in
  let epoch_loss_target_limits =
    [| 430.; 370.; 365.; 362.; 360.; 359.; 358.; 357.5; 357.; 356.5 |]
  in
  for epoch = 0 to epochs - 1 do
    let epoch_loss = ref 0. in
    for batch = 0 to n_batches - 1 do
      batch_ref := batch;
      Train.run ctx sgd_step;
      let loss = batch_loss.@[0] in
      if epoch = 0 && batch < 3 then Stdio.printf "  batch %d loss: %.4f\n%!" batch loss;
      epoch_loss := !epoch_loss +. loss;
      Int.incr step_ref
    done;
    let below = Float.(epoch_loss_target_limits.(epoch) > !epoch_loss) in
    Stdio.printf "Epoch %d, avg token loss: %.4f, epoch loss below %g=%b%s\n%!" epoch
      (!epoch_loss /. Float.of_int n_batches)
      epoch_loss_target_limits.(epoch) below
      (if below then "" else ", actual loss: " ^ Float.to_string !epoch_loss)
  done;

  (* === Autoregressive generation === *)
  let counter_n, infer_bindings = IDX.get_static_symbol IDX.empty in
  let model_infer = model ~train_step:None in
  let%cd infer_logits = model_infer { seq_input } in
  let%cd infer_step =
    infer_logits.forward;
    { dice } =: uniform_at !@counter_n
  in
  Train.set_on_host infer_logits.value;
  let infer_step = Train.to_routine (Context.context sgd_step) infer_bindings infer_step in
  let counter_ref = IDX.find_exn (Context.bindings infer_step) counter_n in
  counter_ref := 0;

  (* Compute softmax probabilities from logits for a given position *)
  let softmax_probs pos =
    let logits = Array.init vocab_size ~f:(fun i ->
        Tn.get_value infer_logits.value [| pos; i |]) in
    let max_l = Array.fold logits ~init:Float.neg_infinity ~f:Float.max in
    let exps = Array.map logits ~f:(fun l -> Float.exp (l -. max_l)) in
    let sum_exp = Array.fold exps ~init:0. ~f:( +. ) in
    Array.map exps ~f:(fun e -> e /. sum_exp)
  in

  let gen_name () =
    for pos = 0 to seq_len - 1 do
      for v = 0 to vocab_size - 1 do
        Tn.set_value seq_input.value [| pos; v |] 0.
      done
    done;
    Tn.set_value seq_input.value [| 0; 0 |] 1.;

    let rec aux pos name =
      if pos > seq_len then name
      else begin
        Int.incr counter_ref;
        Train.run ctx infer_step;
        let dice_value = dice.@[0] in
        let probs = softmax_probs (pos - 1) in
        let max_i = vocab_size - 1 in
        let rec sample i sum =
          if i >= max_i then '.'
          else
            let new_sum = sum +. probs.(i) in
            if Float.(new_sum > dice_value) then
              List.nth_exn Dataprep.Names.letters_with_dot i
            else sample (i + 1) new_sum
        in
        let next_char = sample 0 0. in
        if Char.equal next_char '.' || Char.equal next_char ' ' then name
        else begin
          let char_idx = Dataprep.Names.char_index next_char in
          Tn.set_value seq_input.value [| pos; char_idx |] 1.;
          aux (pos + 1) (name ^ String.make 1 next_char)
        end
      end
    in
    aux 1 ""
  in

  (* Generate very few names because different hardware backends diverge quickly. *)
  let generated_names = Array.init 3 ~f:(fun _ -> gen_name ()) in
  Array.iter generated_names ~f:print_endline

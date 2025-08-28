open Base
open Ocannl
open Stdio
open Bigarray
module Tn = Ir.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Asgns = Ir.Assignments
module Nn = Nn_blocks

module type Backend = Ir.Backend_intf.Backend

let () =
  Utils.settings.fixed_state_for_init <- Some 42;
  Tensor.unsafe_reinitialize ();
  
  (* Toy transformer hyperparameters *)
  let d_model = 32 in
  let n_heads = 4 in  
  let d_ff = 64 in
  let max_seq_len = 16 in
  let n_blocks = 2 in
  let dropout_rate = 0.1 in
  let vocab_size = Datasets.Names.dict_size in
  let batch_size = 32 in
  
  (* Get training data - sequences from the names dataset *)
  let all_names = Datasets.Names.get_all () in
  let names_subset = List.take all_names 1000 in (* Use subset for quick training *)
  
  (* Create sequences with target shifted by one *)
  let create_sequences names =
    List.concat_map names ~f:(fun name ->
      let chars = String.to_list name in
      let char_indices = List.map chars ~f:Datasets.Names.char_to_index in
      let padded = List.take (char_indices @ List.init max_seq_len ~f:(fun _ -> 0)) max_seq_len in
      let seq_len = Int.min (List.length char_indices) (max_seq_len - 1) in
      if seq_len > 1 then
        List.init (seq_len - 1) ~f:(fun i ->
          let input_seq = List.take padded (i + 1) @ List.init (max_seq_len - i - 1) ~f:(fun _ -> 0) in
          let target = List.nth_exn padded (i + 1) in
          (input_seq, target)
        )
      else []
    )
  in
  
  let sequences = create_sequences names_subset in
  Stdio.printf "Total sequences: %d\n%!" (List.length sequences);
  
  (* Pad sequences to batch size *)
  let round_up_by = batch_size - (List.length sequences % batch_size) in
  let sequences = List.take sequences round_up_by @ sequences in
  let n_batches = List.length sequences / batch_size in
  
  let input_seqs, targets = List.unzip sequences in
  
  (* Convert to tensors *)
  let input_tensor = 
    let arr = Array.of_list_map input_seqs ~f:(fun seq -> Array.of_list seq) in
    let genarray = Genarray.create Float32 c_layout [| List.length input_seqs; max_seq_len; vocab_size |] in
    (* Convert to one-hot *)
    Array.iteri arr ~f:(fun batch_idx seq ->
      Array.iteri seq ~f:(fun seq_idx char_idx ->
        if char_idx > 0 then
          Genarray.set genarray [| batch_idx; seq_idx; char_idx |] 1.
      )
    );
    TDSL.rebatch ~l:"input" (Ir.Ndarray.as_array Ir.Ops.Single genarray) ()
  in
  
  let target_tensor =
    let arr = Array.of_list targets in
    let genarray = Genarray.create Float32 c_layout [| List.length targets; vocab_size |] in
    (* Convert to one-hot *)
    Array.iteri arr ~f:(fun batch_idx char_idx ->
      Genarray.set genarray [| batch_idx; char_idx |] 1.
    );
    TDSL.rebatch ~l:"target" (Ir.Ndarray.as_array Ir.Ops.Single genarray) ()
  in
  
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  
  let%op input_batch = input_tensor @| batch_n in
  let%op target_batch = target_tensor @| batch_n in
  
  (* Build the transformer model *)
  let transformer_decoder ~label () =
    (* Embedding and positional encoding *)
    let embed = Nn.embedding ~label:(["embed"] @ label) ~vocab_size ~d_model in
    let pos_enc = Nn.positional_encoding ~label:(["pos_enc"] @ label) ~d_model ~max_seq_len () in
    
    (* Transformer blocks *)
    let blocks = 
      List.init n_blocks ~f:(fun i ->
        Nn.transformer_block 
          ~label:(["block" ^ Int.to_string i] @ label) 
          ~heads:n_heads ~d_model ~d_ff ~dropout_rate ())
    in
    
    fun () input ->
      let%op embedded = embed () input in
      let%op x = TDSL.O.(embedded + pos_enc) in
      
      (* Apply transformer blocks sequentially *)
      let x_transformed = List.fold blocks ~init:x ~f:(fun x_in block ->
        block () x_in
      ) in
      
      (* Output projection to vocabulary - average pool over sequence *)
      let%op x_pooled = TDSL.einsum1 "batch, seq, d_model => batch, d_model" x_transformed in
      let%op x_avg = TDSL.O.(x_pooled /. !..max_seq_len) in
      x_avg * { w_out = uniform (); o = [vocab_size]; i = [d_model] }
  in
  
  let%op logits = transformer_decoder ~label:["transformer"] () input_batch in
  let%op probs = softmax_last_axis logits in
  let%op loss = neg (log ((probs *. target_batch) ++ "batch, vocab => batch, 1")) in
  let%op batch_loss = (loss ++ "batch, 1 => 0") /. !..batch_size in
  
  let update = Train.grad_update batch_loss in
  let%op learning_rate = 0.01 in
  let sgd = Train.sgd_update ~learning_rate batch_loss in
  
  let module Backend = (val Backends.fresh_backend ()) in
  let ctx = Train.init_params (module Backend) bindings batch_loss in
  let sgd_step = Train.to_routine (module Backend) ctx bindings (Asgns.sequence [ update; sgd ]) in
  
  let open Operation.At in
  let batch_ref = IDX.find_exn sgd_step.bindings batch_n in
  
  (* Training loop *)
  for epoch = 0 to 5 do
    let epoch_loss = ref 0. in
    for batch = 0 to n_batches - 1 do
      batch_ref := batch;
      Train.run sgd_step;
      let loss = batch_loss.@[0] in
      epoch_loss := !epoch_loss +. loss;
      if batch % 10 = 0 then 
        Stdio.printf "Epoch %d, batch %d/%d, loss=%.4g\n%!" epoch batch n_batches loss
    done;
    Stdio.printf "Epoch %d, avg loss=%.4g\n%!" epoch (!epoch_loss /. Float.of_int n_batches)
  done;
  
  (* Generate some text *)
  let counter_n, gen_bindings = IDX.get_static_symbol IDX.empty in
  
  (* Create input for generation - start with a single character *)
  let%cd gen_input = { start_char } in
  Train.set_on_host gen_input.value;
  
  let%cd gen_logits = transformer_decoder ~label:["gen"] () gen_input in
  let%cd gen_step =
    gen_logits.forward;
    { dice } =: uniform_at !@counter_n
  in
  Train.set_on_host gen_logits.value;
  
  let gen_routine = Train.to_routine (module Backend) ctx gen_bindings gen_step in
  let counter_ref = IDX.find_exn gen_routine.bindings counter_n in
  counter_ref := 0;
  
  let generate_char input_seq =
    (* Convert input sequence to one-hot tensor *)
    let input_array = Genarray.create Float32 c_layout [| 1; max_seq_len; vocab_size |] in
    List.iteri input_seq ~f:(fun seq_idx char_idx ->
      if char_idx > 0 && seq_idx < max_seq_len then
        Genarray.set input_array [| 0; seq_idx; char_idx |] 1.
    );
    Tn.set_values gen_input.value (Ir.Ndarray.as_array Ir.Ops.Single input_array);
    Int.incr counter_ref;
    Train.run gen_routine;
    
    (* Sample from output distribution *)
    let dice_value = dice.@[0] in
    let rec sample_char i sum =
      if i >= vocab_size then vocab_size - 1
      else
        let prob = gen_logits.@{[| i |]} in
        let prob_normalized = Float.exp prob in
        let new_sum = sum +. prob_normalized in
        if Float.compare new_sum dice_value > 0 then i
        else sample_char (i + 1) new_sum
    in
    sample_char 0 0.
  in
  
  (* Generate a few names *)
  Stdio.printf "\nGenerating names:\n%!";
  for _ = 0 to 9 do
    let rec generate_name seq =
      if List.length seq >= max_seq_len then
        List.map seq ~f:Datasets.Names.index_to_char |> String.of_char_list
      else
        let next_idx = generate_char seq in
        if next_idx = 0 then (* End token *)
          List.map seq ~f:Datasets.Names.index_to_char |> String.of_char_list
        else
          generate_name (seq @ [next_idx])
    in
    let start_idx = 1 + Random.int (vocab_size - 1) in (* Random starting character *)
    let name = generate_name [start_idx] in
    Stdio.printf "  %s\n%!" name
  done
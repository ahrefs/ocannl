open! Base
open Ocannl.Nn_blocks.DSL_modules

let () =
  (* Basic transformer test with teacher forcing *)
  let ctx = Context.auto () in
  (* Test configuration *)
  let batch_size = 2 in
  let src_seq_len = 10 in
  let tgt_seq_len = 8 in
  let d_model = 64 in
  let num_heads = 4 in
  let d_ff = 128 in
  let src_vocab_size = 100 in
  let tgt_vocab_size = 100 in
  let num_encoder_layers = 1 in
  let num_decoder_layers = 1 in

  Stdio.printf "Testing transformer with teacher forcing\n";

  (* Create a simple transformer model *)
  let transformer_model =
    Ocannl.Nn_blocks.transformer ~label:[ "test_transformer" ] ~num_encoder_layers
      ~num_decoder_layers ~num_heads ~d_enc:d_model ~d_dec:d_model ~d_ff ()
  in

  (* Create input tensors *)
  let src =
    TDSL.range_of_shape ~label:[ "src" ] ~batch_dims:[ batch_size; src_seq_len ] ~input_dims:[]
      ~output_dims:[ src_vocab_size ] ()
  in

  (* For teacher forcing: create shifted versions of target sequence *)
  (* tgt_input: positions 0 to tgt_seq_len-2 (all but last) *)
  let tgt_input =
    TDSL.range_of_shape ~label:[ "tgt_input" ]
      ~batch_dims:[ batch_size; tgt_seq_len - 1 ]
      ~input_dims:[] ~output_dims:[ tgt_vocab_size ] ()
  in

  (* tgt_target: positions 1 to tgt_seq_len-1 (all but first) *)
  (* In practice, this would be shifted token IDs, here we use one-hot for simplicity *)
  let tgt_target =
    NTDSL.init ~l:"tgt_target" ~prec:Ir.Ops.single
      ~b:[ batch_size; tgt_seq_len - 1 ]
      ~i:[] ~o:[ tgt_vocab_size ]
      ~f:(function
        | [| _b; s; v |] ->
            (* Create a simple one-hot pattern for testing *)
            if v = Int.((s + 1) % tgt_vocab_size) then 1. else 0.
        | idcs -> failwith @@ "Invalid indices: " ^ Sexp.to_string_hum ([%sexp_of: int array] idcs))
      ()
  in

  (* Create a causal mask for the decoder input (shifted target sequence) *)
  (* Mask should be 0 for positions to mask out, 1 for positions to keep *)
  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single
      ~b:[ tgt_seq_len - 1 ]
      ~i:[ tgt_seq_len - 1 ]
      ~o:[]
      ~f:(function
        | [| s; t |] -> if s >= t then 1. else 0.
        | idcs ->
            failwith @@ "Invalid indices: expected [| _; s; _; t |], got "
            ^ Sexp.to_string_hum ([%sexp_of: int array] idcs))
      ()
  in

  (* Use the transformer with teacher forcing *)
  (* Create a train_step symbol for training mode *)
  let module IDX = Ocannl.Train.IDX in
  let train_step_n, bindings = IDX.get_static_symbol IDX.empty in
  let loss, logits =
    Ocannl.Nn_blocks.transformer_with_loss ~label:[ "transformer_loss" ] ~model:transformer_model ()
      ~train_step:(Some train_step_n) ~src ~tgt_input ~tgt_target ~mask
  in

  (* Forward pass to check shapes and loss; set [output_cd_file] to true for debugging. *)
  let _ctx = Ocannl.Train.forward_once ~output_cd_file:false ~bindings ctx loss in

  (* Verify shapes *)
  Stdio.printf "Loss shape:\n%s\n" (Sexp.to_string_hum ([%sexp_of: Shape.t] loss.Tensor.shape));
  Stdio.printf "Logits shape:\n%s\n%!"
    (Sexp.to_string_hum ([%sexp_of: Shape.t] logits.Tensor.shape))

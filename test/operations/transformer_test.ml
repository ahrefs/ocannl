open! Base
open Ocannl.Operation.DSL_modules

let () =
  (* Basic transformer test *)
  let ctx = Context.auto () in
  (* Test configuration *)
  let batch_size = 2 in
  let src_seq_len = 10 in
  let tgt_seq_len = 8 in
  let d_model = 64 in
  let num_heads = 6 in
  let d_ff = 128 in
  let src_vocab_size = 100 in
  let tgt_vocab_size = 100 in
  let num_encoder_layers = 2 in
  let num_decoder_layers = 2 in

  Stdio.printf "Testing basic transformer model\n";

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

  let tgt =
    TDSL.range_of_shape ~label:[ "tgt" ] ~batch_dims:[ batch_size; tgt_seq_len ] ~input_dims:[]
      ~output_dims:[ tgt_vocab_size ] ()
  in

  (* Create a causal mask for the decoder input (target sequence) *)
  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ batch_size; tgt_seq_len ] ~i:[ tgt_seq_len ]
      ~o:[ 1 ]
      ~f:(function
        | [| _; s; _; t |] -> if s <= t then 1. else 0.
        | idcs ->
            failwith @@ "Invalid indices length: expected [| _; s; _; t |], got "
            ^ Sexp.to_string_hum ([%sexp_of: int array] idcs))
      ()
  in

  (* Forward pass *)
  let output = transformer_model ~train_step:None ~src ~tgt ~mask in

  let _ctx = Ocannl.Train.forward_once ctx output in

  (* Verify output shape *)
  Stdio.printf "Output shape:\n%s\n%!"
    (Sexp.to_string_hum ([%sexp_of: Shape.t] output.Tensor.shape))

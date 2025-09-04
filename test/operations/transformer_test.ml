open! Base
open Ocannl.Operation.DSL_modules

let () =
  (* Basic transformer test *)
  let module Backend = (val Backends.fresh_backend ()) in
  (* Test configuration *)
  let batch_size = 2 in
  let src_seq_len = 10 in
  let tgt_seq_len = 8 in
  let d_model = 64 in
  let num_heads = 4 in
  let d_ff = 128 in
  let src_vocab_size = 100 in
  let tgt_vocab_size = 100 in
  let num_encoder_layers = 2 in
  let num_decoder_layers = 2 in

  Stdio.printf "Testing basic transformer model\n";

  (* Create a simple transformer model *)
  let transformer_model =
    Ocannl.Nn_blocks.transformer ~label:[ "test_transformer" ] ~num_encoder_layers
      ~num_decoder_layers ~num_heads ~d_model ~d_ff ()
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
  let output = transformer_model ~src ~tgt ~mask in

  (* Verify output shape *)
  Stdio.printf "Output shape:\n%s\n%!"
    (Sexp.to_string_hum ([%sexp_of: Shape.t] output.Tensor.shape))

(* ;

   (* Test transformer components *) Stdio.printf "\nTesting transformer components\n";

   let d_model = 32 in let num_heads = 2 in let seq_len = 5 in let batch_size = 1 in

   (* Test multi-head attention *) let%op mha = Ocannl.Nn_blocks.multi_head_attention ~label:[
   "test_mha" ] ~num_heads () in

   let input = Tensor.ndarray ~label:[ "input" ] ~grad_spec:Tensor.Prohibit_grad [| Array.init
   batch_size ~f:(fun _ -> Array.init seq_len ~f:(fun _ -> Array.init d_model ~f:(fun _ ->
   Random.float 1.0))); |] in

   let mha_output = mha input in

   (* Test layer norm *) let%op ln = Nn_blocks.layer_norm ~label:[ "test_ln" ] () in let ln_output =
   ln input in

   (* Test feed forward *) let%op ffn = Nn_blocks.feed_forward ~label:[ "test_ffn" ] ~d_model
   ~d_ff:64 () in let ffn_output = ffn input in

   (* Verify shapes *) Stdio.printf "MHA output shape: %s\n" (Sexp.to_string_hum ([%sexp_of: int
   array] (Tensor.shape mha_output).output_dims)); Stdio.printf "Layer norm output shape: %s\n"
   (Sexp.to_string_hum ([%sexp_of: int array] (Tensor.shape ln_output).output_dims)); Stdio.printf
   "FFN output shape: %s\n" (Sexp.to_string_hum ([%sexp_of: int array] (Tensor.shape
   ffn_output).output_dims));

   Stdio.printf "\nAll tests completed successfully!\n" *)

open! Base
open Ocannl.Nn_blocks.DSL_modules

let () =
  (* Basic transformer test *)
  let ctx = Context.auto () in
  (* Test configuration *)
  let batch_size = 2 in
  let tgt_seq_len = 8 in
  let d_model = 64 in
  let num_heads = 4 in
  let tgt_vocab_size = 100 in

  Stdio.printf "Testing basic multi-head attention\n";

  let attention_model =
    Ocannl.Nn_blocks.multi_head_attention ~label:[ "test_attention" ] ~num_heads ~d_k:d_model
      ~d_v:d_model ()
  in

  (* Create input tensors *)

  let seq =
    TDSL.range_of_shape ~label:[ "tgt" ] ~batch_dims:[ batch_size; tgt_seq_len ] ~input_dims:[]
      ~output_dims:[ tgt_vocab_size ] ()
  in

  (* Create a causal mask for the decoder input (target sequence) *)
  (* Mask should be 0 for positions to mask out, 1 for positions to keep *)
  (* This creates an upper triangular matrix where future positions are masked *)
  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ batch_size; tgt_seq_len ] ~i:[ tgt_seq_len ]
      ~o:[ 1 ]
      ~f:(function
        | [| _; s; _; t |] -> if s >= t then 1. else 0.
        | idcs ->
            failwith @@ "Invalid indices length: expected [| _; s; _; t |], got "
            ^ Sexp.to_string_hum ([%sexp_of: int array] idcs))
      ()
  in

  (* Forward pass *)
  let output = attention_model ~train_step:None ~mask seq in

  let _ctx = Ocannl.Train.forward_once ctx output in

  (* Verify output shape *)
  Stdio.printf "Output shape:\n%s\n%!"
    (Sexp.to_string_hum ([%sexp_of: Shape.t] output.Tensor.shape))

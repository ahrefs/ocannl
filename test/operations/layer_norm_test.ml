open! Base
open Ocannl.Nn_blocks.DSL_modules

let%op mini_decoder_block ~label ~num_heads ~d_k ~d_v ~d_ff ?(epsilon = 1e-5) () =
  let open Ocannl.Nn_blocks in
  let masked_mha = multi_head_attention ~label:("masked_mha" :: label) ~num_heads ~d_k ~d_v () in
  (* Standard 2-layer FFN: expand to d_ff then contract back to d_model *)
  let ffn = mlp ~label:("ffn" :: label) ~hid_dims:[ d_ff ] () in
  let ln1 = layer_norm ~label:("ln1" :: label) ~epsilon () in
  let ln2 = layer_norm ~label:("ln2" :: label) ~epsilon () in
  fun ~train_step target ~mask ->
    let x1 = ln1 (target + masked_mha ~train_step ~mask target) in
    ln2 (x1 + ffn x1)

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
  let d_ff = 128 in
  let tgt_vocab_size = 100 in

  Stdio.printf "Testing basic mini decoder model\n";

  (* Create a simple transformer model *)
  let mini_decoder_model =
    mini_decoder_block ~label:[ "test_mini_decoder" ] ~num_heads ~d_k:d_model ~d_v:d_model ~d_ff ()
  in

  let tgt =
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
  let output = mini_decoder_model ~train_step:None tgt ~mask in

  let _ctx = Ocannl.Train.forward_once ctx output in

  (* Verify output shape *)
  Stdio.printf "Output shape:\n%s\n%!"
    (Sexp.to_string_hum ([%sexp_of: Shape.t] output.Tensor.shape))

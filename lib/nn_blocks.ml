(** This file contains basic building blocks for neural networks, with limited functionality. Feel
    free to copy-paste and modify as needed.

    We follow "the principle of least commitment": where possible, we use row variables to remain
    agnostic to the number of axes. This flexibility often remains unused, but it makes explicit the
    architectural structure.

    The einsum specifications in this file often use the single-char mode (no commas), where the
    spaces are entirely ignored / optional, but are used copiously for readability. *)

open! Base
open Operation.DSL_modules
module Tn = Ir.Tnode

let%op mlp_layer ~label ~hid_dim () x = relu (({ w = uniform () } * x) + { b = 0.; o = [ hid_dim ] })

(** Masks and scales by 1/keep_prob to maintain expected value. When [train_step = None], the
    dropout rate is ignored and the tensor is returned unmodified. *)
let%op dropout ~rate () ~train_step x =
  match train_step with
  | Some train_step when Float.(rate > 0.0) ->
      x *. (!.rate < uniform_at !@train_step *. x) /. (1.0 - !.rate)
  | _ -> x

(** Multi-layer perceptron of depth [List.length hid_dims + 1], with a linear output layer. *)
let%op mlp ~label ~hid_dims () =
  let layers =
    List.mapi hid_dims ~f:(fun i hid_dim ->
        mlp_layer ~label:(("L" ^ Int.to_string i) :: label) ~hid_dim ())
  in
  fun x ->
    let hidden = List.fold layers ~init:x ~f:(fun x layer -> layer x) in
    { w_out } * hidden

let reduce_specified_axes spec =
  let lhs =
    if String.contains spec ',' then
      Str.global_replace (Str.regexp "[A-Za-z][A-Za-z_0-9]*") "0" spec
    else Str.global_replace (Str.regexp "[A-Za-z]") "0" spec
  in
  spec ^ " => " ^ lhs

(** Softmax across specified axes. Does not support non-default row variables. *)
let%op softmax ~spec ?(temperature = 1.0) () =
  let spec = reduce_specified_axes spec in
  fun x ->
    let x_scaled = if Float.(temperature <> 1.0) then x /. !.temperature else x in
    let max_vals = x_scaled @^^ spec in
    let exp_vals = exp (x_scaled - max_vals) in
    exp_vals /. (exp_vals ++ spec)

let%op multi_head_attention ~label ~num_heads ?temperature ?(dropout_rate = 0.0) () ~train_step
    ?mask x =
  let q = { w_q } * x in
  let k = { w_k } * x in
  let v = { w_v } * x in
  (* Works with arbitrary number of model axes via `..d..` (row variable syntax). *)
  let scores =
    (q +* " ... s | h ..d..; ... t | h ..d.. => ... s | t -> h " [ "h"; "d" ] k) /. sqrt (dim d)
  in
  Shape.set_dim h num_heads;
  (* We don't need to lift [softmax ~spec ()] because it doesn't introduce any new params. *)
  let attn_weights =
    softmax ~spec:" ... | t -> ..." ?temperature ()
      (match mask with None -> scores | Some mask -> where mask scores !.(-1e9))
  in
  let attn_weights = dropout ~rate:dropout_rate () ~train_step attn_weights in
  let attended = attn_weights +* " ... s | t -> h; ... t | h ... => ... s | h ... " v in
  { w_o } * attended

let%op layer_norm ~label ?(epsilon = 1e-5) () x =
  let mean = x ++ " ... | ..d..  => ... | 0 " [ "d" ] in
  let centered = (x - mean) /. dim d in
  let variance = (centered * centered) ++ " ... | ... => ... |  0 " in
  let std_dev = sqrt (variance + !.epsilon) in
  let normalized = centered /. std_dev in
  (* gamma and beta are learned, but initialized to good defaults *)
  ({ gamma = 1. } *. normalized) + { beta = 0. }

let%op transformer_encoder_block ~label ~num_heads ~d_ff ?(epsilon = 1e-5) () =
  let mha = multi_head_attention ~label:(label @ [ "mha" ]) ~num_heads () in
  (* Standard 2-layer FFN: expand to d_ff then contract back to d_model *)
  let ffn = mlp ~label:(label @ [ "ffn" ]) ~hid_dims:[ d_ff ] () in
  let ln1 = layer_norm ~label:(label @ [ "ln1" ]) ~epsilon () in
  let ln2 = layer_norm ~label:(label @ [ "ln2" ]) ~epsilon () in
  fun ~train_step input ->
    let attn_output = mha ~train_step input in
    let x1 = ln1 (input + attn_output) in
    let ffn_output = ffn x1 in
    ln2 (x1 + ffn_output)

let%op cross_attention ~label ~num_heads ?temperature ?(dropout_rate = 0.0) () ~train_step x
    ~enc_output =
  let q = { w_q } * x in
  let k = { w_k } * enc_output in
  let v = { w_v } * enc_output in
  let scores =
    (q +* " ... s | h ..d..; ... t | h ..d.. => ... | s t -> h " [ "h"; "d" ] k) /. sqrt (dim d)
  in
  Shape.set_dim h num_heads;
  let attn_weights = softmax ~spec:" ... | ... t -> ..." ?temperature () scores in
  let attn_weights = dropout ~rate:dropout_rate () ~train_step attn_weights in
  let attended = attn_weights +* " ... | s t -> h; ... t | h ... => ... s | h ... " v in
  { w_o } * attended

let%op transformer_decoder_block ~label ~num_heads ~d_ff ?(epsilon = 1e-5) () =
  let masked_mha = multi_head_attention ~label:(label @ [ "masked_mha" ]) ~num_heads () in
  let cross_mha = cross_attention ~label:(label @ [ "cross_mha" ]) ~num_heads () in
  (* Standard 2-layer FFN: expand to d_ff then contract back to d_model *)
  let ffn = mlp ~label:(label @ [ "ffn" ]) ~hid_dims:[ d_ff ] () in
  let ln1 = layer_norm ~label:(label @ [ "ln1" ]) ~epsilon () in
  let ln2 = layer_norm ~label:(label @ [ "ln2" ]) ~epsilon () in
  let ln3 = layer_norm ~label:(label @ [ "ln3" ]) ~epsilon () in
  fun ~train_step target ~enc_output ~mask ->
    let self_attn_output = masked_mha ~train_step ~mask target in
    let x1 = ln1 (target + self_attn_output) in
    let cross_attn_output = cross_mha ~train_step x1 ~enc_output in
    let x2 = ln2 (x1 + cross_attn_output) in
    let ffn_output = ffn x2 in
    ln3 (x2 + ffn_output)

let transformer_encoder ~label ~num_layers ~num_heads ~d_ff ?(epsilon = 1e-5) () =
  let layers =
    List.init num_layers ~f:(fun i ->
        transformer_encoder_block
          ~label:(label @ [ "layer" ^ Int.to_string i ])
          ~num_heads ~d_ff ~epsilon ())
  in
  fun ~train_step x -> List.fold layers ~init:x ~f:(fun x layer -> layer ~train_step x)

let transformer_decoder ~label ~num_layers ~num_heads ~d_ff ?(epsilon = 1e-5) () =
  let layers =
    List.init num_layers ~f:(fun i ->
        transformer_decoder_block
          ~label:(label @ [ "layer" ^ Int.to_string i ])
          ~num_heads ~d_ff ~epsilon ())
  in
  fun ~train_step target ~enc_output ~mask ->
    List.fold layers ~init:target ~f:(fun x layer -> layer ~train_step x ~enc_output ~mask)

let%op transformer ~label ~num_encoder_layers ~num_decoder_layers ~num_heads ~d_model ~d_ff
    ?(epsilon = 1e-5) () =
  let encoder =
    transformer_encoder ~label:(label @ [ "encoder" ]) ~num_layers:num_encoder_layers ~num_heads
      ~d_ff ~epsilon ()
  in
  let decoder =
    transformer_decoder ~label:(label @ [ "decoder" ]) ~num_layers:num_decoder_layers ~num_heads
      ~d_ff ~epsilon ()
  in
  (* All inline definitions, including for d, are lifted up to the unit parameter above. *)
  Shape.set_dim d d_model;
  fun ~train_step ~src ~tgt ~mask ->
    (* Learned positional encoding *)
    let enc_output =
      encoder ~train_step
        (src +* " ... s | ..v.. ; ..v.. -> d => ... s | d " [ "d" ] { src_embed } + { pos_encoding })
    in
    let tgt_embedded =
      tgt +* " ... t | ..v.. ; ..v.. -> d => ... t | d " { tgt_embed } + pos_encoding
    in
    decoder ~train_step tgt_embedded ~enc_output ~mask
    +* " ... | d; d -> ..v.. => ... | ..v.. " { w_out }

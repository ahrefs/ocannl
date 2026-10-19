(** {1 Demo models}

    Complete example architectures assembled from {!Nn_blocks}: the models the tutorials, tests and
    benchmarks train end to end. They are demonstrations of composing the blocks rather than
    reusable components -- copy one and adapt it. *)

open! Base
open Nn_blocks.DSL_modules
open Nn_blocks

(** {2 Attention and transformers} *)

(** Multi-head self-attention reduced to its core -- no label, mask, dropout or position embedding
    -- for illustration; {!Nn_blocks.multi_head_attention} is the reusable block. *)
let%op multi_head_att_workshop ~num_heads ~d_k ~d_v () x =
  let q = { w_q } * x in
  let k = { w_k } * x in
  let v = { w_v } * x in
  let scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ]) /. sqrt (dim d)
  in
  Shape.set_dim h num_heads;
  Shape.set_dim d d_k;
  Shape.set_dim e d_v;
  let attn_weights = softmax ~spec:" ... | t -> ..." () scores in
  { w_o } * (attn_weights +* v " ... s | t -> h; ... t | h e => ... s | h e" [ "e" ])

(** An encoder-decoder transformer: token embeddings, position encoding per [pos_embed], the
    {!Nn_blocks.transformer_encoder} and {!Nn_blocks.transformer_decoder} stacks, and an output
    projection. *)
let%op transformer ~label ~num_encoder_layers ~num_decoder_layers ~num_heads ~d_enc ~d_dec ~d_ff
    ?(epsilon = 1e-5) ?mask_fill ?(pos_embed = Learned_additive) () =
  let enc_att = [%oc d_enc / num_heads] in
  let dec_att = [%oc d_dec / num_heads] in
  let attn_pos_embed = match pos_embed with RoPE _ as pe -> pe | _ -> No_pos_embed in
  let encoder =
    transformer_encoder ~label:("encoder" :: label) ~num_layers:num_encoder_layers ~num_heads
      ~d_k:enc_att ~d_v:enc_att ~d_ff ~epsilon ~pos_embed:attn_pos_embed ()
  in
  let decoder =
    transformer_decoder ~label:("decoder" :: label) ~num_layers:num_decoder_layers ~num_heads
      ~d_k:dec_att ~d_v:dec_att ~d_ff ~epsilon ?mask_fill ~pos_embed:attn_pos_embed ()
  in
  (* NOTE: { pos_encoding } and { pos_encoding_tgt } are learned inline params lifted by %op. They
     are created unconditionally but only used when pos_embed = Learned_additive. *)
  let pos_encoding_tgt = if Int.(d_enc = d_dec) then pos_encoding else { pos_encoding_tgt } in
  fun ~train_step ~src ~tgt ~mask ->
    let enc_input = { src_embed; o = [ d_enc ] } * src in
    let enc_output =
      match pos_embed with
      | Learned_additive -> encoder ~train_step (enc_input + { pos_encoding })
      | Sinusoidal_additive { enc_encoding; _ } -> encoder ~train_step (enc_input + enc_encoding)
      | RoPE _ | No_pos_embed -> encoder ~train_step enc_input
    in
    let tgt_embedded_base = { tgt_embed; o = [ d_dec ] } * tgt in
    let tgt_embedded =
      match pos_embed with
      | Learned_additive -> tgt_embedded_base + pos_encoding_tgt
      | Sinusoidal_additive { dec_encoding; _ } -> tgt_embedded_base + dec_encoding
      | RoPE _ | No_pos_embed -> tgt_embedded_base
    in
    { w_out } * decoder ~train_step tgt_embedded ~enc_output ~mask

(** Transformer with teacher forcing for autoregressive training.

    TODO: Simplify once tensor shifting/slicing is better supported in shape inference. Currently
    requires pre-shifted tgt_input (all but last token) and tgt_target (all but first token). During
    training, the model learns to predict tgt_target given tgt_input. *)
let%op transformer_with_loss ~label:_ ~model () ~train_step ~src ~tgt_input ~tgt_target ~mask =
  (* Get model predictions for the input sequence *)
  let logits = model ~train_step ~src ~tgt:tgt_input ~mask in

  (* Numerically stable cross-entropy over the vocabulary dimension; tgt_target should be one-hot
     encoded or use label smoothing *)
  let loss = cross_entropy_loss ~spec:"... | v" () ~logits ~targets:tgt_target in

  (* Return both loss and logits for potential additional metrics *)
  (loss, logits)

(** {2 Convolutional networks} *)

(** LeNet-style architecture for simple image classification (e.g., MNIST). Classic architecture:
    conv -> pool -> conv -> pool -> fc layers. Output shape is inferred from training data. *)
let%op lenet ?(label = [ "lenet" ]) ?(out_channels1 = 6) ?(out_channels2 = 16) ?(use_padding = true)
    () =
  let conv1 =
    conv2d ~label:("conv1" :: label) ~kernel_size:5 ~use_padding ~out_channels:out_channels1 ()
  in
  let pool1 = max_pool2d ~stride:2 () in
  let conv2 =
    conv2d ~label:("conv2" :: label) ~kernel_size:5 ~use_padding ~out_channels:out_channels2 ()
  in
  let pool2 = max_pool2d ~stride:2 () in
  let fc1 = mlp_layer ~label:("fc1" :: label) ~hid_dim:120 () in
  let fc2 = mlp_layer ~label:("fc2" :: label) ~hid_dim:84 () in
  fun ~train_step:_ x ->
    let x = conv1 x |> relu |> pool1 |> conv2 |> relu |> pool2 |> fc1 |> fc2 in
    (* Final classification layer - output shape inferred from training data *)
    ({ w_logits } * x) + { b_logits = 0. }

(** Simple CNN for Sokoban-like grid environments. Processes grid states with multiple conv layers
    and outputs action logits. *)
let%op sokoban_cnn ~label ?(num_actions = 4) () =
  (* Process spatial features with conv layers *)
  let conv1 = conv_bn_relu ~label:("conv1" :: label) ~kernel_size:3 () in
  let conv2 = conv_bn_relu ~label:("conv2" :: label) ~kernel_size:3 () in
  let conv3 = conv_bn_relu ~label:("conv3" :: label) ~kernel_size:3 () in
  fun ~train_step ~grid_state ->
    let x = conv1 ~train_step grid_state |> conv2 ~train_step |> conv3 ~train_step in

    (* Global pooling to aggregate spatial info *)
    let x = global_avg_pool2d x in

    (* Action head *)
    let action_logits = ({ w_action } * x) + { b_action = 0.; o = [ num_actions ] } in

    (* Optional: value head for actor-critic methods *)
    let value = ({ w_value } * x) + { b_value = 0.; o = [ 1 ] } in

    (action_logits, value)

(** Modern CNN with depthwise separable convolutions for efficiency. Suitable for mobile/edge
    deployment. [_width_mult] is caller-visible as unimplemented: changing it currently has no
    effect on channel counts. *)
let%op mobile_cnn ~label ?(num_classes = 1000) ?(_width_mult = 1.0) () =
  (* TODO: implement channel width multiplier *)
  (* Initial standard conv *)
  let conv_init = conv_bn_relu ~label:("conv_init" :: label) ~kernel_size:3 ~stride:2 () in

  (* Depthwise separable blocks *)
  let dw_block1 = depthwise_separable_conv2d ~label:("dw1" :: label) ~stride:1 () in
  let dw_block2 = depthwise_separable_conv2d ~label:("dw2" :: label) ~stride:2 () in
  let dw_block3 = depthwise_separable_conv2d ~label:("dw3" :: label) ~stride:1 () in
  let dw_block4 = depthwise_separable_conv2d ~label:("dw4" :: label) ~stride:2 () in

  let bn = batch_norm2d ~label:("bn_final" :: label) () in

  fun ~train_step x ->
    let x =
      conv_init ~train_step x |> dw_block1 |> relu |> dw_block2 |> relu |> dw_block3 |> relu
      |> dw_block4 |> relu |> bn ~train_step
    in

    (* Global pooling and classification *)
    let x = global_avg_pool2d x in
    ({ w_classifier } * x) + { b_classifier = 0.; o = [ num_classes ] }

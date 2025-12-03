(** {1 Neural Network Building Blocks}

    This file contains basic building blocks for neural networks, with limited functionality. Feel
    free to copy-paste and modify as needed.

    Design principles, OCANNL fundamentals, and common patterns:
    - "Principle of least commitment": use row variables where axis count doesn't matter
    - Einsum specs here often use single-char mode (no commas) but with spaces for readability
    - Pooling uses constant kernels (0.5 + 0.5) to propagate window dimensions
    - conv2d uses convolution syntax: "stride*out+kernel," (often in multi-char mode)
    - Input axes (before →) for kernels show intent (and end up rightmost for memory locality)
    - Inline params \{ \} are always learnable and are lifted to unit parameter ()
    - Introduce inputs to a block after sub-block construction (sub-blocks have no automatic lifting
      like there is for inline definitions of params)
    - Always use literal strings with einsum operators when capturing variables
    - Avoid unnecessary variable captures in einsum operators, be mindful they can shadow other
      identifiers *)

open! Base
open Ocannl_tensor.Operation.DSL_modules
module Tn = Ir.Tnode

let%op box_muller grad_spec init_f () =
  let epsilon = [%oc Float.ldexp 1. (-24)] in
  let u1 = init_f () in
  let u2 = init_f () in
  Ocannl_tensor.Operation.pointmul ~grad_spec
    (sqrt (-2. *. log (u1 + (!.epsilon *. (1. - u1)))))
    (cos (2. *. !.Float.pi *. u2))

[%%extend_dsls
let normal () = [%oc box_muller grad_spec uniform ()]
let normal1 () = [%oc box_muller grad_spec uniform1 ()]
let normal_at counter = [%oc box_muller grad_spec (fun () -> uniform_at counter) ()]
let normal_at1 counter = [%oc box_muller grad_spec (fun () -> uniform_at1 counter) ()]]

open DSL_modules

(** Convert a list of integers to a one-hot encoded tensor.
    @param num_classes The number of classes (size of the one-hot dimension)
    @param lst List of integer class indices (0-based)
    @return A tensor of shape [len; num_classes] with one-hot encoding *)
let one_hot_of_int_list ~num_classes lst =
  let open Bigarray in
  let len = List.length lst in
  let arr = lst |> Array.of_list in
  let genarray = Genarray.create Float32 c_layout [| len; num_classes |] in
  (* Initialize to zero *)
  for i = 0 to len - 1 do
    for j = 0 to num_classes - 1 do
      Genarray.set genarray [| i; j |] 0.
    done
  done;
  (* Set one-hot positions *)
  for i = 0 to len - 1 do
    Genarray.set genarray [| i; arr.(i) |] 1.
  done;
  TDSL.rebatch ~l:"one_hot" (Ir.Ndarray.as_array Ir.Ops.Single genarray) ()

let%op mlp_layer ~label ~hid_dim () x = relu (({ w } * x) + { b = 0.; o = [ hid_dim ] })

(** Masks and scales by 1/keep_prob to maintain expected value. When [train_step = None], the
    dropout rate is ignored and the tensor is returned unmodified. *)
let%op dropout ~rate () ~train_step x =
  match train_step with
  | Some train_step when Float.(rate > 0.0) ->
      x *. (!.rate < uniform_at !@train_step) /. (1.0 - !.rate)
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

let%op multi_head_attention ~label ~num_heads ~d_k ~d_v ?temperature ?(dropout_rate = 0.0) ()
    ~train_step ?mask x =
  let q = { w_q } * x in
  let k = { w_k } * x in
  let v = { w_v } * x in
  let scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ]) /. sqrt (dim d)
  in
  Shape.set_dim h num_heads;
  (* NOTE: often d_k = d_v = d_model / num_heads, but we allow for other values. *)
  Shape.set_dim d d_k;
  Shape.set_dim e d_v;
  (* We don't need to lift [softmax ~spec ()] because it doesn't introduce any new params. *)
  let attn_weights =
    softmax ~spec:" ... | t -> ..." ?temperature ()
      (match mask with None -> scores | Some mask -> where mask scores !.(-1e9))
  in
  let attn_weights = dropout ~rate:dropout_rate () ~train_step attn_weights in
  (* w_o output shape will automatically be set to the model dimension(s) by shape inference. *)
  { w_o } * (attn_weights +* v " ... s | t -> h; ... t | h e => ... s | h e" [ "e" ])

let%op layer_norm ~label ?(epsilon = 1e-5) () x =
  let mean = x ++ " ... | ..d..  => ... | 0 " [ "d" ] in
  let centered = (x - mean) /. dim d in
  let variance = (centered *. centered) ++ " ... | ... => ... |  0 " in
  let std_dev = sqrt (variance + !.epsilon) in
  let normalized = centered /. std_dev in
  (* gamma and beta are learned, but initialized to good defaults *)
  ({ gamma = 1. } *. normalized) + { beta = 0. }

let%op transformer_encoder_block ~label ~num_heads ~d_k ~d_v ~d_ff ?(epsilon = 1e-5) () =
  let mha = multi_head_attention ~label:("mha" :: label) ~num_heads ~d_k ~d_v () in
  (* Standard 2-layer FFN: expand to d_ff then contract back to d_model *)
  let ffn = mlp ~label:("ffn" :: label) ~hid_dims:[ d_ff ] () in
  let ln1 = layer_norm ~label:("ln1" :: label) ~epsilon () in
  let ln2 = layer_norm ~label:("ln2" :: label) ~epsilon () in
  fun ~train_step input ->
    let x1 = ln1 (input + mha ~train_step input) in
    ln2 (x1 + ffn x1)

let%op cross_attention ~label ~num_heads ~d_k ~d_v ?temperature ?(dropout_rate = 0.0) () ~train_step
    x ~enc_output =
  let q = { w_q } * x in
  let k = { w_k } * enc_output in
  let v = { w_v } * enc_output in
  let scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h " [ "h"; "d" ]) /. sqrt (dim d)
  in
  Shape.set_dim h num_heads;
  Shape.set_dim d d_k;
  Shape.set_dim e d_v;
  let attn_weights = softmax ~spec:" ... | t -> ..." ?temperature () scores in
  let attn_weights = dropout ~rate:dropout_rate () ~train_step attn_weights in
  { w_o } * (attn_weights +* v " ... s | t -> h; ... t | h e => ... s | h e" [ "e" ])

let%op transformer_decoder_block ~label ~num_heads ~d_k ~d_v ~d_ff ?(epsilon = 1e-5) () =
  let masked_mha = multi_head_attention ~label:("masked_mha" :: label) ~num_heads ~d_k ~d_v () in
  let cross_mha = cross_attention ~label:("cross_mha" :: label) ~num_heads ~d_k ~d_v () in
  (* Standard 2-layer FFN: expand to d_ff then contract back to d_model *)
  let ffn = mlp ~label:("ffn" :: label) ~hid_dims:[ d_ff ] () in
  let ln1 = layer_norm ~label:("ln1" :: label) ~epsilon () in
  let ln2 = layer_norm ~label:("ln2" :: label) ~epsilon () in
  let ln3 = layer_norm ~label:("ln3" :: label) ~epsilon () in
  fun ~train_step target ~enc_output ~mask ->
    let x1 = ln1 (target + masked_mha ~train_step ~mask target) in
    let x2 = ln2 (x1 + cross_mha ~train_step x1 ~enc_output) in
    ln3 (x2 + ffn x2)

let transformer_encoder ~label ~num_layers ~num_heads ~d_k ~d_v ~d_ff ?(epsilon = 1e-5) () =
  let layers =
    List.init num_layers ~f:(fun i ->
        transformer_encoder_block
          ~label:(("layer" ^ Int.to_string i) :: label)
          ~num_heads ~d_k ~d_v ~d_ff ~epsilon ())
  in
  fun ~train_step x -> List.fold layers ~init:x ~f:(fun x layer -> layer ~train_step x)

let transformer_decoder ~label ~num_layers ~num_heads ~d_k ~d_v ~d_ff ?(epsilon = 1e-5) () =
  let layers =
    List.init num_layers ~f:(fun i ->
        transformer_decoder_block
          ~label:(("layer" ^ Int.to_string i) :: label)
          ~num_heads ~d_k ~d_v ~d_ff ~epsilon ())
  in
  fun ~train_step target ~enc_output ~mask ->
    List.fold layers ~init:target ~f:(fun x layer -> layer ~train_step x ~enc_output ~mask)

let%op transformer ~label ~num_encoder_layers ~num_decoder_layers ~num_heads ~d_enc ~d_dec ~d_ff
    ?(epsilon = 1e-5) () =
  let enc_att = [%oc d_enc / num_heads] in
  let dec_att = [%oc d_dec / num_heads] in
  let encoder =
    transformer_encoder ~label:("encoder" :: label) ~num_layers:num_encoder_layers ~num_heads
      ~d_k:enc_att ~d_v:enc_att ~d_ff ~epsilon ()
  in
  let decoder =
    transformer_decoder ~label:("decoder" :: label) ~num_layers:num_decoder_layers ~num_heads
      ~d_k:dec_att ~d_v:dec_att ~d_ff ~epsilon ()
  in
  (* All inline definitions, including for ds, dt, are lifted up to the unit parameter above. *)
  let pos_encoding_tgt = if Int.(d_enc = d_dec) then pos_encoding else { pos_encoding_tgt } in
  fun ~train_step ~src ~tgt ~mask ->
    (* Learned positional encoding *)
    let enc_output = encoder ~train_step ({ src_embed; o = [ d_enc ] } * src) + { pos_encoding } in
    let tgt_embedded = ({ tgt_embed; o = [ d_dec ] } * tgt) + pos_encoding_tgt in
    { w_out } * decoder ~train_step tgt_embedded ~enc_output ~mask

(** Transformer with teacher forcing for autoregressive training.

    TODO: Simplify once tensor shifting/slicing is better supported in shape inference. Currently
    requires pre-shifted tgt_input (all but last token) and tgt_target (all but first token). During
    training, the model learns to predict tgt_target given tgt_input. *)
let%op transformer_with_loss ~label:_ ~model () ~train_step ~src ~tgt_input ~tgt_target ~mask =
  (* Get model predictions for the input sequence *)
  let logits = model ~train_step ~src ~tgt:tgt_input ~mask in

  (* Compute cross-entropy loss between predictions and target *)
  (* softmax over vocabulary dimension *)
  let log_probs = log (softmax ~spec:"... | v" () logits) in

  (* Negative log likelihood loss: -sum(target * log_probs) *)
  (* tgt_target should be one-hot encoded or use label smoothing *)
  let loss = -(tgt_target *. log_probs) ++ "...|... => 0" in

  (* Return both loss and logits for potential additional metrics *)
  (loss, logits)

(** {2 Convolutional Neural Network Building Blocks} *)

(** 2D convolution layer with flexible padding and stride options. *)
let%op conv2d ~label ?(kernel_size = 3) ?(stride = 1) ?(use_padding = true) () x =
  (* Notation: kernel height (kh), kernel width (kw), input channels (ic), output channels (oc),
     output height (oh), output width (ow) *)
  Shape.set_dim kh kernel_size;
  Shape.set_dim kw kernel_size;
  x
  +* { kernel }
       "... | stride*oh+kh, stride*ow+kw, ..ic..; kh, kw, ..ic.. -> ..oc.. => ... | oh, ow, ..oc.."
       [ "kh"; "kw" ]
  + { bias = 0. }

(** Depthwise separable convolution - more efficient for mobile/edge devices. Consists of depthwise
    conv (spatial filtering per channel) followed by pointwise conv (1x1 conv for channel mixing) *)
let%op depthwise_separable_conv2d ~label ?(kernel_size = 3) ?(stride = 1) ?(use_padding = true) () x =
  (* Depthwise: each input channel is convolved with its own filter *)
  Shape.set_dim kh kernel_size;
  Shape.set_dim kw kernel_size;
  let depthwise =
    x
    +* { dw_kernel }
         "... | stride*oh+kh, stride*ow+kw, ..ic..; kh, kw -> ..ic.. => ... | oh, ow, ..ic.."
         [ "kh"; "kw" ]
  in
  (* Pointwise: 1x1 conv to mix channels *)
  depthwise
  +* { pw_kernel } "... | h, w, ..ic..; ..ic.. -> ..oc.. => ... | h, w, ..oc.."
  + { bias = 0. }

(** Max pooling for 2D spatial data - reduces spatial dimensions by taking maximum values. *)
let%op max_pool2d ?(stride = 2) ?(window_size = 2) () x =
  Shape.set_dim wh window_size;
  Shape.set_dim ww window_size;
  (* NOTE: projections inference runs per-assignment in a distinct phase from shape inference, so
     for it to know about the window size, we use a constant kernel = 1 to propagate the shape. We
     use a trick to create a shape-inferred constant tensor, equivalently we could write "NTDSL.term
     ~fetch_op:(Constant 1.) ()" but that's less concise. See:
     https://github.com/ahrefs/ocannl/discussions/381 *)
  x
  @^+ "... | stride*oh< + wh, stride*ow< + ww, ..c..; wh, ww => ... | oh, ow, ..c.." [ "wh"; "ww" ]
        (0.5 + 0.5)

(** Average pooling for 2D spatial data - reduces spatial dimensions by averaging values. *)
let%op avg_pool2d ?(stride = 2) ?(window_size = 2) () x =
  Shape.set_dim wh window_size;
  Shape.set_dim ww window_size;
  let sum =
    x
    +++ "... | stride*oh< + wh, stride*ow< + ww, ..c..; wh, ww => ... | oh, ow, ..c.." [ "wh"; "ww" ]
          (0.5 + 0.5)
  in
  sum /. (dim wh *. dim ww)

(** Global average pooling - reduces each feature map to a single value by averaging. Commonly used
    before final classification layer. *)
let%op global_avg_pool2d x = x ++ "... | h, w, ..c.. => ... | 0, 0, ..c.."

(** Batch normalization for CNN layers - normalizes across the batch dimension for each channel.
    Typically applied after convolutions and before activations. *)
let%op batch_norm2d ~label ?(epsilon = 1e-5) ?(momentum = 0.9) () ~train_step x =
  let _ = momentum in
  (* FIXME: implement running statistics, currently using learned params *)
  (* Compute batch statistics across batch and spatial dimensions for each channel *)
  let total_size = dim o *. dim h *. dim w in
  let mean = (x ++ "..o.. | h, w, ..c.. => 0 | 0, 0, ..c.." [ "o"; "h"; "w" ]) /. total_size in
  let centered = x - mean in
  let variance = ((centered *. centered) ++ "... | h, w, ..c.. => 0 | 0, 0, ..c..") /. total_size in
  let std_dev = sqrt (variance + !.epsilon) in
  let normalized = centered /. std_dev in
  (* Scale and shift with learnable parameters *)
  match train_step with
  | Some _ ->
      (* During training: update running statistics *)
      ({ gamma = 1. } *. normalized) + { beta = 0. }
  | None ->
      (* During inference: use running statistics (simplified for now) *)
      (gamma *. normalized) + beta

(** Conv block with conv -> batch norm -> activation pattern *)
let%op conv_bn_relu ~label ?(kernel_size = 3) ?(stride = 1) () =
  let conv = conv2d ~label:("conv" :: label) ~kernel_size ~stride () in
  let bn = batch_norm2d ~label:("bn" :: label) () in
  fun ~train_step x -> relu (bn ~train_step (conv x))

(** Residual block for ResNet-style architectures. Features skip connections that help with gradient
    flow in deep networks. *)
let%op resnet_block ~label ?(stride = 1) () =
  let conv1 = conv2d ~label:("conv1" :: label) ~kernel_size:3 ~stride () in
  let bn1 = batch_norm2d ~label:("bn1" :: label) () in
  let conv2 = conv2d ~label:("conv2" :: label) ~kernel_size:3 ~stride:1 () in
  let bn2 = batch_norm2d ~label:("bn2" :: label) () in
  let identity =
    if stride > 1 then
      (* Need to downsample the skip connection *)
      let downsample_conv = conv2d ~label:("downsample" :: label) ~kernel_size:1 ~stride () in
      let downsample_bn = batch_norm2d ~label:("downsample_bn" :: label) () in
      fun train_step x -> downsample_bn ~train_step (downsample_conv x)
    else fun _train_step x -> x
  in
  fun ~train_step x ->
    let out = conv1 x |> bn1 ~train_step |> relu |> conv2 |> bn2 ~train_step in
    relu (out + identity train_step x)

(** LeNet-style architecture for simple image classification (e.g., MNIST). Classic architecture:
    conv -> pool -> conv -> pool -> fc layers *)
let%op lenet ~label ?(num_classes = 10) () =
  let conv1 = conv2d ~label:("conv1" :: label) ~kernel_size:5 () in
  let pool1 = max_pool2d ~stride:2 () in
  let conv2 = conv2d ~label:("conv2" :: label) ~kernel_size:5 () in
  let pool2 = max_pool2d ~stride:2 () in
  let fc1 = mlp_layer ~label:("fc1" :: label) ~hid_dim:120 () in
  let fc2 = mlp_layer ~label:("fc2" :: label) ~hid_dim:84 () in
  fun ~train_step:_ x ->
    let x = conv1 x |> relu |> pool1 |> conv2 |> relu |> pool2 |> fc1 |> fc2 in
    (* Final classification layer *)
    ({ w_logits } * x) + { b_logits = 0.; o = [ num_classes ] }

(** VGG-style block - multiple convolutions with same filter count followed by pooling *)
let%op vgg_block ~label ~num_convs ?(kernel_size = 3) () =
  let convs =
    List.init num_convs ~f:(fun i ->
        conv_bn_relu ~label:(("conv" ^ Int.to_string i) :: label) ~kernel_size ())
  in
  let pool = max_pool2d ~stride:2 () in
  fun ~train_step x ->
    let x = List.fold convs ~init:x ~f:(fun x conv -> conv ~train_step x) in
    pool x

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
    deployment. *)
let%op mobile_cnn ~label ?(num_classes = 1000) ?(width_mult = 1.0) () =
  let _ = width_mult in
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

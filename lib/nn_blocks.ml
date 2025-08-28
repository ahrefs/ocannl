open! Base
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL

let%op mlp_layer ~label ~hid_dim () x = relu (({ w = uniform () } * x) + { b = 0.; o = [ hid_dim ] })

let mlp ~label ~hid_dims () =
  let layers =
    List.mapi hid_dims ~f:(fun i hid_dim ->
        mlp_layer ~label:(("L" ^ Int.to_string i) :: label) ~hid_dim ())
  in
  fun x -> List.fold layers ~init:x ~f:(fun x layer -> layer x)

(** Multi-head attention components maintaining explicit head dimension *)

(** Multi-head attention using explicit head dimension throughout computation.
    This avoids reshaping by keeping heads as a separate axis. *)
let multi_head_attention ~label ~heads ~d_model () =
  let d_k = d_model / heads in
  let d_v = d_model / heads in
  fun () x_q x_k x_v ->
    let%op query = TDSL.einsum "batch, seq, d_model; heads, d_k, d_model => batch, seq, heads, d_k" 
      x_q { w_q = uniform (); o = [heads; d_k]; i = [d_model] } in
    let%op key = TDSL.einsum "batch, seq, d_model; heads, d_k, d_model => batch, seq, heads, d_k" 
      x_k { w_k = uniform (); o = [heads; d_k]; i = [d_model] } in
    let%op value = TDSL.einsum "batch, seq, d_model; heads, d_v, d_model => batch, seq, heads, d_v" 
      x_v { w_v = uniform (); o = [heads; d_v]; i = [d_model] } in
    let dk_sqrt = TDSL.O.(sqrt (!..(d_k))) in
    let%op scores = TDSL.einsum "batch, seq_q, heads, d_k; batch, seq_kv, heads, d_k => batch, seq_q, seq_kv, heads" 
      query key in
    let%op scaled_scores = TDSL.O.(scores /. dk_sqrt) in
    let%op attention_weights = TDSL.softmax_last_axis scaled_scores in
    let%op attention_output = TDSL.einsum "batch, seq_q, seq_kv, heads; batch, seq_kv, heads, d_v => batch, seq_q, heads, d_v" 
      attention_weights value in
    TDSL.einsum "batch, seq, heads, d_v; d_model, heads, d_v => batch, seq, d_model" 
      attention_output { w_o = uniform (); o = [d_model]; i = [heads; d_v] }

(** Layer normalization *)  
let layer_norm ~label ~d_model ~epsilon () =
  fun () x ->
    let d_model_f = TDSL.O.(!..(d_model)) in
    let%op mean = TDSL.O.((TDSL.einsum1 "..., d_model => ..., 1" x) /. d_model_f) in
    let%op centered = TDSL.O.(x - mean) in
    let%op variance = TDSL.O.((TDSL.einsum1 "..., d_model => ..., 1" (centered *. centered)) /. d_model_f) in
    let%op std = TDSL.O.(sqrt (variance + !.epsilon)) in
    let%op normalized = TDSL.O.(centered /. std) in
    TDSL.O.((normalized *. { gamma = 1.; o = [d_model] }) + { beta = 0.; o = [d_model] })

(** Feed-forward network *)
let%op feed_forward ~label ~d_model ~d_ff () x =
  let%op hidden = relu ((x * { w1 = uniform (); o = [d_ff]; i = [d_model] }) + { b1 = 0.; o = [d_ff] }) in
  (hidden * { w2 = uniform (); o = [d_model]; i = [d_ff] }) + { b2 = 0.; o = [d_model] }

(** Transformer encoder/decoder block *)
let transformer_block ~label ~heads ~d_model ~d_ff ~dropout_rate () =
  let mha = multi_head_attention ~label:(["attn"] @ label) ~heads ~d_model () in
  let ln1 = layer_norm ~label:(["norm1"] @ label) ~d_model ~epsilon:1e-5 () in
  let ln2 = layer_norm ~label:(["norm2"] @ label) ~d_model ~epsilon:1e-5 () in
  let ff = feed_forward ~label:(["ff"] @ label) ~d_model ~d_ff in
  fun () x ->
    let%op attn_output = mha () x x x in
    let%op x_with_attn = TDSL.O.(x + attn_output) in
    let%op x_normed1 = ln1 () x_with_attn in
    let%op ff_output = ff () x_normed1 in
    let%op x_with_ff = TDSL.O.(x_normed1 + ff_output) in
    ln2 () x_with_ff

let positional_encoding ~label ~d_model ~max_seq_len () =
  let pos_encodings = 
    Array.init max_seq_len ~f:(fun pos ->
      Array.init d_model ~f:(fun i ->
        if i % 2 = 0 then
          Float.sin (Float.of_int pos /. Float.(10000. ** (Float.of_int i /. Float.of_int d_model)))
        else
          Float.cos (Float.of_int pos /. Float.(10000. ** (Float.of_int (i - 1) /. Float.of_int d_model)))
      )
    ) in
  TDSL.ndarray (Array.to_list pos_encodings |> List.map ~f:Array.to_list)

let%op embedding ~label ~vocab_size ~d_model () input_ids =
  input_ids * { embed_weight = uniform (); o = [d_model]; i = [vocab_size] }

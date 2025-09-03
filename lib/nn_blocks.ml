open! Base
open Operation.DSL_modules

let%op mlp_layer ~label ~hid_dim () x = relu (({ w = uniform () } * x) + { b = 0.; o = [ hid_dim ] })

let mlp ~label ~hid_dims () =
  let layers =
    List.mapi hid_dims ~f:(fun i hid_dim ->
        mlp_layer ~label:(("L" ^ Int.to_string i) :: label) ~hid_dim ())
  in
  fun x -> List.fold layers ~init:x ~f:(fun x layer -> layer x)

let%op softmax x =
  let max_vals = x @^^ "...|...t->... => ...|...0->..." in
  let exp_vals = exp (x - max_vals) in
  exp_vals /. (exp_vals ++ "...|...t->... => ...|...0->...")

let%op basic_multi_head_attention ~label ~num_heads () x =
  let q = { w_q } * x in
  let k = { w_k } * x in
  let v = { w_v } * x in
  let scores = q +* "...s|h...; ...t|h... => ...|st->h" [ "h" ] k in
  Shape.set_dim h num_heads;
  let attn_weights = softmax scores in
  let attended = attn_weights +* "...|st->h; ...t|h... => ...s|h..." v in
  { w_o } * attended

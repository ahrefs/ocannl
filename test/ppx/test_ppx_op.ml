open Base
open Ocannl
open Nn_blocks.DSL_modules

let%op y0 = (2 *. { hey1 }) + 3
let%op y1 x = ({ hey2 } * 2) + x
let%op y2 x1 x2 = (x1 *. { hey3 }) + x2
let%op a = [ (1, 2, 3); (4, 5, 6) ]
let%op b = [| [ 7; 8 ]; [ 9; 10 ] |]
let%op y = ({ hey4 } * ([ 2.0 ] : q)) + ([ 1.0 ] : p)
let%op z = (([ 2.0 ] : q) *. { hey5 }) + ({ hey6 } *. ([ 1.0 ] : q))

let stride = 2
and dilation = 3
and use_padding = true

let%op z2 = { hey7 } +* "stride*a+dilation*b,;b=>a," { hey8 }

let z3 =
  let s = 2 and d = 3 in
  [%op { hey9 } +* "i, s*a+d*bc; b => i, a, c" { hey10 }]

let%op concat_single = { hey11 } ++^ "i=>i"
let%op concat_pair = ({ hey12 }, { hey13 }) ++^ "i; j => i^j" [ "i"; "j" ]
let%op initialized_no_unit x = relu ({ init_w = kaiming ~scale_sq:2.0 normal1 () } * x)

let initialized_expr =
  [%op
    let f x = relu ({ expr_w = kaiming normal1 () } * x) in
    f]

let () = ignore (y0, y1, y2, a, b, y)
let () = ignore (z, z2, z3)
let () = ignore (concat_single, concat_pair)
let () = ignore (initialized_no_unit, initialized_expr)
let%op mlp_layer ~label ~hid_dim () ~x = relu (({ w } * x) + { b; o = [ hid_dim ] })

let%op _use_layer =
  let l1 = mlp_layer ~label:[ "L" ] ~hid_dim:3 () in
  let l2 = mlp_layer ~label:[ "L2" ] ~hid_dim:3 () in
  fun x -> l1 ~x:(l2 ~x)

let%op _config_layer ~label () =
  let l = mlp_layer ~label:(label @ [ "L" ]) ~hid_dim:3 () in
  fun x -> l ~x

let%op _three_layer_perceptron ~label ~dim1 ~dim2 ~dim3 () =
  let l1 = mlp_layer ~label:(label @ [ "L1" ]) ~hid_dim:dim1 () in
  let l2 = mlp_layer ~label:(label @ [ "L2" ]) ~hid_dim:dim2 () in
  let l3 = mlp_layer ~label:(label @ [ "L3" ]) ~hid_dim:dim3 () in
  fun x -> l3 ~x:(l2 ~x:(l1 ~x))

(* Dimension captures are bound per call, at the top of their innermost enclosing function, while
   inline declarations keep their lifting to the unit parameter or the definition
   (gh-ocannl-1507). *)
let%op _capture_no_unit x = x ++ "ab => a" [ "b" ]

let%op _capture_with_unit () x =
  let y = { cw } * x in
  (y ++ "ab => a" [ "b" ]) /. dim b

let%op _capture_siblings =
  let f x = x ++ "ab => a" [ "b" ] in
  let g x = x ++ "ab => b" [ "b" ] in
  fun x -> f x + g x

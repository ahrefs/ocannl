open Base
open Ocannl
module TDSL = Operation.TDSL

let%op y0 = (2 *. { hey1 }) + 3
let%op y1 x = ({ hey2 } * 2) + x
let%op y2 x1 x2 = (x1 *. { hey3 }) + x2
let%op a = [ (1, 2, 3); (4, 5, 6) ]
let%op b = [| [ 7; 8 ]; [ 9; 10 ] |]
let%op y = ({ hey4 } * 'q' 2.0) + 'p' 1.0
let%op z = ('q' 2.0 * { hey5 }) + ({ hey6 } * 'p' 1.0)

let stride = 2
and dilation = 3

let%op z2 = { hey7 } *+ "stride*a+dilation*b,;b=>a," { hey8 }

let z3 =
  let s = 2 and d = 3 in
  [%op { hey9 } *+ "is*a+d*bc;b=>iac" { hey10 }]

let () = ignore (y0, y1, y2, a, b, y, z, z2, z3)
let%op mlp_layer ~label ~hid_dim () x = relu (({ w } * x) + { b; o = [ hid_dim ] })

let%op _use_layer x =
  mlp_layer ~label:[ "L" ] ~hid_dim:3 () (mlp_layer ~label:[ "L2" ] ~hid_dim:3 () x)

let%op _config_layer ~config:_ x = mlp_layer ~label:[ "L" ] ~hid_dim:3 () x

let%op _three_layer_perceptron ~label ~dim1 ~dim2 ~dim3 () x =
  mlp_layer ~label:(label @ [ "L3" ]) ~hid_dim:dim3 ()
    (mlp_layer ~label:(label @ [ "L2" ]) ~hid_dim:dim2 ()
       (mlp_layer ~label:(label @ [ "L1" ]) ~hid_dim:dim1 () x))

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

type mlp_layer_config = { label : string list; hid_dim : int }

let%op mlp_layer ~config x = relu (({ w } * x) + { b; o = [ config.hid_dim ] })

let%op _use_layer x =
  mlp_layer ~config:{ label = [ "L" ]; hid_dim = 3 }
    (mlp_layer ~config:{ label = [ "L2" ]; hid_dim = 3 } x)

let%op _config_layer ~config:_ x = mlp_layer ~config:{ label = [ "L" ]; hid_dim = 3 } x

type tlp_config = { label : string list; dim1 : int; dim2 : int; dim3 : int }

let%op _three_layer_perceptron ~(config : tlp_config) x =
  mlp_layer
    ~config:{ label = "L3" :: config.label; hid_dim = config.dim3 }
    (mlp_layer
       ~config:{ label = "L2" :: config.label; hid_dim = config.dim2 }
       (mlp_layer ~config:{ label = "L1" :: config.label; hid_dim = config.dim1 } x))

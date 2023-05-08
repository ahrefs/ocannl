open Base
open Ocannl
module FDSL = Operation.FDSL

let%nn_op y0 = (2 *. "hey1") + 3
let%nn_op y1 x = ("hey2" * 2) + x
let%nn_op y2 x1 x2 = (x1 *. "hey3") + x2
let%nn_op a = [ (1, 2, 3); (4, 5, 6) ]
let%nn_op b = [| [ 7; 8 ]; [ 9; 10 ] |]
let%nn_op y = ("hey4" * 'q' 2.0) + 'p' 1.0
let%nn_op z = ('q' 2.0 * "hey5") + ("hey6" * 'p' 1.0)
let () = ignore (y0, y1, y2, a, b, y, z)

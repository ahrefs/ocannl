open Base
open Ocannl

let%nn_mo y0 = 2 * "hey1" + 3

let%nn_mo y1 x = 2 * "hey2" + x

let%nn_mo y2 x1 x2 = x1 * "hey3" + x2

let%nn_mo a = [1, 2, 3; 4, 5, 6]
let%nn_mo b = [|[7; 8]; [9; 10]|]

let%nn_mo y = 'q' 2.0 * "hey4" + 'p' 1.0

let%nn_mo z = 'q' 2.0 * "hey5" + "hey6" * 'p' 1.0

let () = ignore (y0, y1, y2, a, b, y, z)
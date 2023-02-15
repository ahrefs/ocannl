open Base
open Ocannl

let%ocannl y0 = 2 * "hey" + 3

let%ocannl y1 x = 2 * "hey" + x

let%ocannl y2 x1 x2 = x1 * "hey" + x2

let%ocannl a = [1, 2, 3; 4, 5, 6]
let%ocannl b = [|[7; 8]; [9; 10]|]

let%ocannl y = "q" 2.0 * "hey" + "p" 1.0

let () = ignore (y0, y1, y2, a, b, y)
open Base
open Ocannl

let%ocannl y0 = 2 * "hey" + 3

let%ocannl y1 x = 2 * "hey" + x

let%ocannl y2 x1 x2 = x1 * "hey" + x2

let () = ignore (y0, y1, y2)
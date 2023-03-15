open Base
open Ocannl
let y0 =
  let open! Operation.O in
    let hey1 = Operation.unconstrained_param ?init:None "hey1" in
    ((Operation.number (Float.of_int 2)) *. hey1) +
      (Operation.number (Float.of_int 3))
let y1 =
  let open! Operation.O in
    let hey2 = Operation.unconstrained_param ?init:None "hey2" in
    fun x -> (hey2 * (Operation.number (Float.of_int 2))) + x
let y2 =
  let open! Operation.O in
    let hey3 = Operation.unconstrained_param ?init:None "hey3" in
    fun x1 -> fun x2 -> (x1 *. hey3) + x2
let a =
  let open! Operation.O in
    Operation.ndarray ~batch_dims:[] ~input_dims:[3] ~output_dims:[2]
      [|(Float.of_int 1);(Float.of_int 2);(Float.of_int 3);(Float.of_int 4);(
        Float.of_int 5);(Float.of_int 6)|]
let b =
  let open! Operation.O in
    Operation.ndarray ~batch_dims:[2] ~input_dims:[] ~output_dims:[2]
      [|(Float.of_int 7);(Float.of_int 8);(Float.of_int 9);(Float.of_int 10)|]
let y =
  let open! Operation.O in
    let hey4 = Operation.unconstrained_param ?init:None "hey4" in
    (hey4 * (Operation.number ~axis_label:"q" 2.0)) +
      (Operation.number ~axis_label:"p" 1.0)
let z =
  let open! Operation.O in
    let hey5 = Operation.unconstrained_param ?init:None "hey5"
    and hey6 = Operation.unconstrained_param ?init:None "hey6" in
    ((Operation.number ~axis_label:"q" 2.0) * hey5) +
      (hey6 * (Operation.number ~axis_label:"p" 1.0))
let () = ignore (y0, y1, y2, a, b, y, z)

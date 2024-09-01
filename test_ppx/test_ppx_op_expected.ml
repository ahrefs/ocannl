open Base
open Ocannl
module TDSL = Operation.TDSL
let y0 =
  let hey1 = TDSL.param ?values:None "hey1" in
  let open! TDSL.O in
    ((+) ?label:(Some ["y0"]))
      ((( *. ) ?label:None) (TDSL.number (Float.of_int 2)) hey1)
      (TDSL.number (Float.of_int 3))
let y1 =
  let hey2 = TDSL.param ?values:None "hey2" in
  let open! TDSL.O in
    fun x ->
      ((+) ?label:(Some ["y1"]))
        ((( * ) ?label:None) hey2 (TDSL.number (Float.of_int 2))) x
let y2 =
  let hey3 = TDSL.param ?values:None "hey3" in
  let open! TDSL.O in
    fun x1 ->
      fun x2 -> ((+) ?label:(Some ["y2"])) ((( *. ) ?label:None) x1 hey3) x2
let a =
  let open! TDSL.O in
    TDSL.ndarray ?label:(Some ["a"]) ~batch_dims:[] ~input_dims:[3]
      ~output_dims:[2]
      [|(Float.of_int 1);(Float.of_int 2);(Float.of_int 3);(Float.of_int 4);(
        Float.of_int 5);(Float.of_int 6)|]
let b =
  let open! TDSL.O in
    TDSL.ndarray ?label:(Some ["b"]) ~batch_dims:[2] ~input_dims:[]
      ~output_dims:[2]
      [|(Float.of_int 7);(Float.of_int 8);(Float.of_int 9);(Float.of_int 10)|]
let y =
  let hey4 = TDSL.param ?values:None "hey4" in
  let open! TDSL.O in
    ((+) ?label:(Some ["y"]))
      ((( * ) ?label:None) hey4 (TDSL.number ?label:None ~axis_label:"q" 2.0))
      (TDSL.number ?label:None ~axis_label:"p" 1.0)
let z =
  let hey5 = TDSL.param ?values:None "hey5"
  and hey6 = TDSL.param ?values:None "hey6" in
  let open! TDSL.O in
    ((+) ?label:(Some ["z"]))
      ((( * ) ?label:None) (TDSL.number ?label:None ~axis_label:"q" 2.0) hey5)
      ((( * ) ?label:None) hey6 (TDSL.number ?label:None ~axis_label:"p" 1.0))
let () = ignore (y0, y1, y2, a, b, y, z)

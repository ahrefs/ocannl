open Base
open Ocannl
module TDSL = Operation.TDSL
let y0 =
  let open! TDSL.O in
    let hey1 = TDSL.param ?values:None "hey1" in
    ((+) ?desc_label:(Some "y0"))
      ((( *. ) ?desc_label:None) (TDSL.number (Float.of_int 2)) hey1)
      (TDSL.number (Float.of_int 3))
let y1 =
  let open! TDSL.O in
    let hey2 = TDSL.param ?values:None "hey2" in
    fun x ->
      ((+) ?desc_label:(Some "y1"))
        ((( * ) ?desc_label:None) hey2 (TDSL.number (Float.of_int 2))) x
let y2 =
  let open! TDSL.O in
    let hey3 = TDSL.param ?values:None "hey3" in
    fun x1 ->
      fun x2 ->
        ((+) ?desc_label:(Some "y2")) ((( *. ) ?desc_label:None) x1 hey3) x2
let a =
  let open! TDSL.O in
    TDSL.ndarray ?desc_label:(Some "a") ~batch_dims:[] ~input_dims:[3]
      ~output_dims:[2]
      [|(Float.of_int 1);(Float.of_int 2);(Float.of_int 3);(Float.of_int 4);(
        Float.of_int 5);(Float.of_int 6)|]
let b =
  let open! TDSL.O in
    TDSL.ndarray ?desc_label:(Some "b") ~batch_dims:[2] ~input_dims:[]
      ~output_dims:[2]
      [|(Float.of_int 7);(Float.of_int 8);(Float.of_int 9);(Float.of_int 10)|]
let y =
  let open! TDSL.O in
    let hey4 = TDSL.param ?values:None "hey4" in
    ((+) ?desc_label:(Some "y"))
      ((( * ) ?desc_label:None) hey4
         (TDSL.number ?desc_label:None ~axis_label:"q" 2.0))
      (TDSL.number ?desc_label:None ~axis_label:"p" 1.0)
let z =
  let open! TDSL.O in
    let hey5 = TDSL.param ?values:None "hey5"
    and hey6 = TDSL.param ?values:None "hey6" in
    ((+) ?desc_label:(Some "z"))
      ((( * ) ?desc_label:None)
         (TDSL.number ?desc_label:None ~axis_label:"q" 2.0) hey5)
      ((( * ) ?desc_label:None) hey6
         (TDSL.number ?desc_label:None ~axis_label:"p" 1.0))
let () = ignore (y0, y1, y2, a, b, y, z)

open Base
open Ocannl
module FDSL = Operation.FDSL
let y0 =
  let open! FDSL.O in
    let hey1 = FDSL.params ?value:None "hey1" in
    ((+) ?desc_label:(Some "y0"))
      ((( *. ) ?desc_label:None) (FDSL.number (Float.of_int 2)) hey1)
      (FDSL.number (Float.of_int 3))
let y1 =
  let open! FDSL.O in
    let hey2 = FDSL.params ?value:None "hey2" in
    fun x ->
      ((+) ?desc_label:(Some "y1"))
        ((( * ) ?desc_label:None) hey2 (FDSL.number (Float.of_int 2))) x
let y2 =
  let open! FDSL.O in
    let hey3 = FDSL.params ?value:None "hey3" in
    fun x1 ->
      fun x2 ->
        ((+) ?desc_label:(Some "y2")) ((( *. ) ?desc_label:None) x1 hey3) x2
let a =
  let open! FDSL.O in
    FDSL.ndarray ?desc_label:(Some "a") ~batch_dims:[]
      ~input_dims:[Shape.dim 3] ~output_dims:[Shape.dim 2]
      [|(Float.of_int 1);(Float.of_int 2);(Float.of_int 3);(Float.of_int 4);(
        Float.of_int 5);(Float.of_int 6)|]
let b =
  let open! FDSL.O in
    FDSL.ndarray ?desc_label:(Some "b") ~batch_dims:[Shape.dim 2]
      ~input_dims:[] ~output_dims:[Shape.dim 2]
      [|(Float.of_int 7);(Float.of_int 8);(Float.of_int 9);(Float.of_int 10)|]
let y =
  let open! FDSL.O in
    let hey4 = FDSL.params ?value:None "hey4" in
    ((+) ?desc_label:(Some "y"))
      ((( * ) ?desc_label:None) hey4
         (FDSL.number ?desc_label:None ~axis_label:"q" 2.0))
      (FDSL.number ?desc_label:None ~axis_label:"p" 1.0)
let z =
  let open! FDSL.O in
    let hey5 = FDSL.params ?value:None "hey5"
    and hey6 = FDSL.params ?value:None "hey6" in
    ((+) ?desc_label:(Some "z"))
      ((( * ) ?desc_label:None)
         (FDSL.number ?desc_label:None ~axis_label:"q" 2.0) hey5)
      ((( * ) ?desc_label:None) hey6
         (FDSL.number ?desc_label:None ~axis_label:"p" 1.0))
let () = ignore (y0, y1, y2, a, b, y, z)

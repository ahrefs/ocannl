open Base
open Ocannl
module TDSL = Operation.TDSL

let y0 =
  let open! TDSL.O in
  let hey1 = TDSL.param ?values:None "hey1" in
  (( + ) ~label:[ "y0" ])
    ((( *. ) ~label:[]) (TDSL.number (Float.of_int 2)) hey1)
    (TDSL.number (Float.of_int 3))

let y1 =
  let open! TDSL.O in
  let hey2 = TDSL.param ?values:None "hey2" in
  fun x -> (( + ) ~label:[ "y1" ]) ((( * ) ~label:[]) hey2 (TDSL.number (Float.of_int 2))) x

let y2 =
  let open! TDSL.O in
  let hey3 = TDSL.param ?values:None "hey3" in
  fun x1 x2 -> (( + ) ~label:[ "y2" ]) ((( *. ) ~label:[]) x1 hey3) x2

let a =
  let open! TDSL.O in
  TDSL.ndarray ~label:[ "a" ] ~batch_dims:[] ~input_dims:[ 3 ] ~output_dims:[ 2 ]
    [|
      Float.of_int 1; Float.of_int 2; Float.of_int 3; Float.of_int 4; Float.of_int 5; Float.of_int 6;
    |]

let b =
  let open! TDSL.O in
  TDSL.ndarray ~label:[ "b" ] ~batch_dims:[ 2 ] ~input_dims:[] ~output_dims:[ 2 ]
    [| Float.of_int 7; Float.of_int 8; Float.of_int 9; Float.of_int 10 |]

let y =
  let open! TDSL.O in
  let hey4 = TDSL.param ?values:None "hey4" in
  (( + ) ~label:[ "y" ])
    ((( * ) ~label:[]) hey4 (TDSL.number ~label:[] ~axis_label:"q" 2.0))
    (TDSL.number ~label:[] ~axis_label:"p" 1.0)

let z =
  let open! TDSL.O in
  let hey5 = TDSL.param ?values:None "hey5" and hey6 = TDSL.param ?values:None "hey6" in
  (( + ) ~label:[ "z" ])
    ((( * ) ~label:[]) (TDSL.number ~label:[] ~axis_label:"q" 2.0) hey5)
    ((( * ) ~label:[]) hey6 (TDSL.number ~label:[] ~axis_label:"p" 1.0))

let () = ignore (y0, y1, y2, a, b, y, z)

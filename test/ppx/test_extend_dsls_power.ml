open Ocannl.Nn_blocks.DSL_modules

let counter = ref 2
let exponent = ref 2.0

[%%extend_dsls
let squared x = x **. 2

let counted_power x =
  [%oc
    Stdlib.incr counter;
    x]
  **. Float.of_int !counter

let escaped_power x = x **. [%oc !exponent]

let nested_power x =
  let square = x **. 2 in
  relu (square **. [%oc !exponent])

let power_of_two () = 2.0 **. Float.of_int !counter]

(* This expansion-only probe leaves the escaped integer for OCaml's float type check. *)
let escaped_integer_initializer source = [%op { w = source **. [%oc 2] }]

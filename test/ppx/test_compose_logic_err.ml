open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

(* Negative tests: ~logic:"@" with / and ** must produce ppx compile-time errors. *)

let test_div_compose a b =
  let%cd _r = { r } =:+ a / b ~logic:"@" in
  _r

let test_pow_compose a b =
  let%cd _r = { r } =:+ a ** b ~logic:"@" in
  _r

open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

(* Negative tests: ~logic:"@" with / and ** (and their word-form aliases div/pow) must produce ppx
   compile-time errors. *)

let test_div_compose a b =
  let%cd _r = { r } =:+ a / b ~logic:"@" in
  _r

let test_pow_compose a b =
  let%cd _r = { r } =:+ a ** b ~logic:"@" in
  _r

let test_div_alias_compose a b =
  let%cd _r = { r } =:+ div a b ~logic:"@" in
  _r

let test_pow_alias_compose a b =
  let%cd _r = { r } =:+ pow a b ~logic:"@" in
  _r

(* Positive compile-only check: a non-multiply, non-banned binary op with ~logic:"@" must NOT be
   rejected (semiring-style combinations are intentional). "add" with Compose computes sum_k a[i,k]
   + b[k,j] -- unusual but not prohibited. *)
let test_add_compose_accepted a b =
  let%cd _r = { r } =:+ add a b ~logic:"@" in
  _r

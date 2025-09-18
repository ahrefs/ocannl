open Ocannl
open Operation.DSL_modules

let%op test_inline_defs x =
  let q = { w } * x in
  let k = { w } * x in
  let v = { w } * x in
  (q * k) + v

let%op test_variable_capture x =
  Shape.set_equal a b;
  x ++ "ab=>ba" [ "a"; "b"; "a" ]

let%op test_mixed x =
  Shape.set_equal a b;
  x +* { b } "ab;bc=>ac" [ "a"; "b" ]

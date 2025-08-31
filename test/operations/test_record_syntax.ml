open Ocannl
module Tensor = Tensor
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module PDSL = Operation.PDSL

(* Test %op record syntax with different initialization patterns *)

(* Basic record syntax - tensor with default initialization using punning *)
let%op _test_op_uniform = { x = uniform () }

(* Record syntax with float constant *)
let%op _test_op_float = { y = 0.5 }

(* Record syntax with integer constant *)
let%op _test_op_int = { z = 42 }

(* Record syntax with list initialization for output dimensions *)
let%op _test_op_list = { weights = [ 0.1; 0.2; 0.3 ] }

(* Record syntax with nested list initialization *)
let%op _test_op_nested = { biases = [ [ 0.0; 1.0 ]; [ 2.0; 3.0 ] ] }

(* Record syntax with extra dimension arguments using full names *)
let%op _test_op_with_dims = { w = uniform (); input_dims = [ 2; 3 ]; output_dims = [ 4 ] }

(* Record syntax with shorthand dimension names *)
let%op _test_op_shorthands = { v = uniform (); i = [ 5 ]; o = [ 6; 7 ] }

(* Test %cd record syntax - inline tensor definitions in computations *)

(* Basic inline tensor definition using punning *)
let _test_cd_computation () =
  let%cd result = { temp } =: !.2.0 in
  result

(* Inline tensor with dimension specifications using full names *)
let _test_cd_with_dims () =
  let%cd result = { temp; output_dims = [ 3; 4 ] } =: !.1.0 in
  result

(* Inline tensor with shorthand dimension names *)
let _test_cd_shorthands () =
  let%cd result = { x; o = [ 10 ] } =: !.3.0 in
  result

let () =
  Stdio.printf "Test compilation successful!\n";
  Stdio.printf "Record syntax for both %%op and %%cd extensions works correctly.\n";
  Stdio.printf "All initialization patterns and shorthand notation supported.\n"

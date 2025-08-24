open Ocannl
module Tensor = Tensor
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL

(* Test %op record syntax with different initialization patterns *)

(* Basic record syntax - tensor with default initialization *)
let%op _test_op_uniform = { x = Operation.uniform ~grad_spec:Require_grad () }

(* Record syntax with explicit value initialization *)
let%op _test_op_value = { y = Tensor.term_init ~grad_spec:Require_grad [| 0.5 |] }

(* Record syntax with extra dimension arguments *)
let%op _test_op_with_dims = 
  { z = Operation.uniform ~grad_spec:Require_grad (); input_dims = [2; 3]; output_dims = [4] }

(* Test %cd record syntax - inline tensor definitions in computations *)

(* Basic inline tensor definition *)
let _test_cd_computation () =
  let%cd result = 
    { temp = temp } =: !.2.0
  in
  result

(* Inline tensor with dimension specifications *)
let _test_cd_with_dims () =
  let%cd result = 
    { temp = temp; output_dims = [3; 4] } =: !.1.0
  in
  result

let () =
  Stdio.printf "Test compilation successful!\n";
  Stdio.printf "Record syntax for both %%op and %%cd extensions works correctly.\n"
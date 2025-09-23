open! Base
open Ocannl.Operation.DSL_modules

(* Test module-level expansion *)
[%%extend_dsls
let my_tensor x = TDSL.O.( + ) x !.2.0 + !.1.0

let my_complex_op a b c =
  let d = a * b in
  let e = d + c in
  relu e

let my_random_op () = { u1 = uniform () } - { u2 = uniform () }]

(* The above should create three modules: TDSL, NTDSL, and PDSL, each containing the functions with
   appropriate DSL operators *)

let () =
  (* Test that the functions are available in each DSL module *)
  let open! DSL_modules.TDSL.O in
  let x = uniform () in
  let result = my_complex_op (my_tensor x) (my_random_op ()) x in
  Tensor.print ~here:[%here] ~force:false ~with_code:false ~with_grad:true `Inline result

let () =
  let open! DSL_modules.NTDSL.O in
  let x = uniform () in
  let result = my_complex_op (my_tensor x) (my_random_op ()) x in
  Tensor.print ~here:[%here] ~force:false ~with_code:false ~with_grad:true `Inline result

let () =
  let open! DSL_modules.PDSL.O in
  let x = uniform () in
  let result = my_complex_op (my_tensor x) (my_random_op ()) x in
  Tensor.print ~here:[%here] ~force:false ~with_code:false ~with_grad:true `Inline result

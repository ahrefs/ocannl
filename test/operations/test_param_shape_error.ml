open Base
open Ocannl
open Ocannl.Operation.DSL_modules

let%expect_test "param without hidden dims should raise error" =
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  (try
    let _w_o = PDSL.param ~label:"w_o" () in
    print_endline "ERROR: Should have raised an exception"
  with
  | Shape.Shape_error (msg, _) ->
      Printf.printf "Got expected error: %s\n" msg);
  
  [%expect {| Got expected error: You forgot to specify the hidden dimension(s) |}]

let%expect_test "param with specified dims should work" =
  (* This should work because we specify the dimensions *)
  let _w_o = PDSL.param ~label:"w_o" ~output_dims:[256] ~input_dims:[128] () in
  (* Just check it doesn't raise an exception during construction *)
  print_endline "Parameter created successfully";
  
  [%expect {| Parameter created successfully |}]
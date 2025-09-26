open! Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

let raises_error_without_hidden_dims () =
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let w_o = PDSL.param "w_o" () in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf w_o;
    Stdio.print_endline "\nERROR: Should have raised an exception"
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

let all_dims_specified () =
  Tensor.unsafe_reinitialize ();
  (* This should work because we specify the dimensions *)
  let w_o = PDSL.param "w_o" ~output_dims:[ 256 ] ~input_dims:[ 128 ] () in
  let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
  Train.printf w_o;
  (* Just check it doesn't raise an exception during construction *)
  Stdio.print_endline "Parameter created successfully"

let () =
  raises_error_without_hidden_dims ();
  all_dims_specified ()

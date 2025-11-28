open! Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

let default_lone_param () =
  Stdio.printf "Testing default lone parameter -- corner case\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let w_o = PDSL.param "w_o" () in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf w_o;
    Stdio.print_endline "\nERROR: Should have raised an exception"
  with Row.Shape_error (msg, _) -> Stdio.printf "Got acceptable error: %s\n" msg

let lone_param_1d () =
  Stdio.printf "Testing lone parameter 1D -- corner case\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = { w_o = uniform1 () } in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf w_o;
    ()
  with Row.Shape_error (msg, _) -> Stdio.printf "Got acceptable error: %s\n" msg

let default_linear_op () =
  Stdio.printf "Testing default linear operation\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = { w = uniform1 () } * [ 3; 4; 5 ] in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf w;
    ()
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

let default_affine_op_propagated () =
  Stdio.printf "Testing default affine operation with propagated dimensions\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = [ 1; 2 ] + ({ w = uniform1 () } * [ 3; 4; 5 ]) in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf w;
    ()
  with Row.Shape_error (msg, _) -> Stdio.printf "Got unacceptable error: %s\n" msg

let default_affine_op_unknown_input () =
  Stdio.printf "Testing default affine operation with unknown input dimensions\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = [ 1; 2 ] + ({ w = uniform1 () } * { x = uniform1 () }) in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf w;
    ()
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

let default_bias_param () =
  Stdio.printf "Testing default bias parameter\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = { x } + { y } in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf w_o;
    Stdio.print_endline "\nERROR: Should have raised an exception"
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

let default_bias_param_1d () =
  Stdio.printf "Testing bias parameter 1D\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = { x = uniform1 () } + { y = uniform1 () } in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf w_o;
    ()
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

let () =
  default_lone_param ();
  lone_param_1d ();
  default_linear_op ();
  default_affine_op_propagated ();
  default_affine_op_unknown_input ();
  default_bias_param ();
  default_bias_param_1d ()

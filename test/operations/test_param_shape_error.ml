open! Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

let default_lone_param () =
  Stdio.printf "Testing default lone parameter -- corner case\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let w_o = TDSL.param "w_o" () in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w_o;
    (* Recorded outcome, not a failure: since d6121e6e ("Update hidden dims heuristic test") a lone
       parameter takes the heuristic's default shape instead of raising, and the golden pins the
       tensor it gets. The line here used to read "ERROR: Should have raised an exception" -- text
       left over from the raising cases below, which sat in a passing golden reading like a blessed
       regression (gh-ocannl-601). The sibling [lone_param_1d] records the same outcome silently. *)
    Stdio.printf "Lone parameter took the heuristic's default shape; no error\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "Got acceptable error: %s\n" msg

let lone_param_1d () =
  Stdio.printf "Testing lone parameter 1D -- corner case\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = { w_o = uniform1 () } in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w_o;
    ()
  with Row.Shape_error (msg, _) -> Stdio.printf "Got acceptable error: %s\n" msg

let default_linear_op () =
  Stdio.printf "Testing default linear operation\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = { w = uniform1 () } * [ 3; 4; 5 ] in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w;
    ()
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

let default_affine_op_propagated () =
  Stdio.printf "Testing default affine operation with propagated dimensions\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = [ 1; 2 ] + ({ w = uniform1 () } * [ 3; 4; 5 ]) in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w;
    ()
  with Row.Shape_error (msg, _) -> Stdio.printf "Got unacceptable error: %s\n" msg

let default_affine_op_unknown_input () =
  Stdio.printf "Testing default affine operation with unknown input dimensions\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = [ 1; 2 ] + ({ w = uniform1 () } * { x = uniform1 () }) in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w;
    ()
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

let default_bias_param () =
  Stdio.printf "Testing default bias parameter\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = { x } + { y } in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w_o;
    Verdict.fail "should have raised an exception"
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

let default_bias_param_1d () =
  Stdio.printf "Testing bias parameter 1D\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = { x = uniform1 () } + { y = uniform1 () } in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w_o;
    ()
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

(* gh-ocannl-1508: a parameter whose input axes are a fixed-index result of an einsum -- the shape
   [Nn_blocks.global_avg_pool2d] had before gh-ocannl-880 -- is not a forgotten hidden dimension: a
   fixed index only bounds the axis from below. The error names the spec instead. *)
let fixed_index_into_param () =
  Stdio.printf "Testing fixed-index axes consumed by a parameter\n";
  Tensor.unsafe_reinitialize ();
  try
    let x = TDSL.range_of_shape ~batch_dims:[ 2 ] ~output_dims:[ 3; 3; 4 ] () in
    let%op pooled = x ++ "... | h, w, ..c.. => ... | 0, 0, ..c.." in
    let%op logits = ({ w } * pooled) + { b = 0.; o = [ 5 ] } in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty logits in
    Verdict.fail "should have raised an exception"
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

let () =
  default_lone_param ();
  lone_param_1d ();
  default_linear_op ();
  default_affine_op_propagated ();
  default_affine_op_unknown_input ();
  default_bias_param ();
  default_bias_param_1d ();
  fixed_index_into_param ()

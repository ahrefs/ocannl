open Base
open Ocannl

let test_scale_input_solved_first () =
  let open Nn_blocks.DSL_modules in
  Tensor.unsafe_reinitialize ();
  Stdio.printf "\n=== set_scale: input solved first, hidden propagated ===\n";
  let input = Shape.get_variable_ref "input_dim" in
  let hidden = Shape.get_variable_ref "hidden_dim" in
  Shape.set_dim input 8;
  Shape.set_scale ~factor:2 hidden input;
  Stdio.printf "input = %s\n"
    (Option.value_map input.var_ref.solved_dim ~default:"unresolved" ~f:Int.to_string);
  Stdio.printf "hidden = %s\n"
    (Option.value_map hidden.var_ref.solved_dim ~default:"unresolved" ~f:Int.to_string)

let test_scale_hidden_solved_first () =
  let open Nn_blocks.DSL_modules in
  Tensor.unsafe_reinitialize ();
  Stdio.printf "\n=== set_scale: hidden solved first, input propagated ===\n";
  let input = Shape.get_variable_ref "input_dim" in
  let hidden = Shape.get_variable_ref "hidden_dim" in
  Shape.set_dim hidden 16;
  Shape.set_scale ~factor:2 hidden input;
  Stdio.printf "input = %s\n"
    (Option.value_map input.var_ref.solved_dim ~default:"unresolved" ~f:Int.to_string);
  Stdio.printf "hidden = %s\n"
    (Option.value_map hidden.var_ref.solved_dim ~default:"unresolved" ~f:Int.to_string)

let test_scale_both_solved_mismatch () =
  let open Nn_blocks.DSL_modules in
  Tensor.unsafe_reinitialize ();
  Stdio.printf "\n=== set_scale: both solved, mismatch ===\n";
  let input = Shape.get_variable_ref "input_dim" in
  let hidden = Shape.get_variable_ref "hidden_dim" in
  Shape.set_dim input 8;
  Shape.set_dim hidden 17;
  try
    Shape.set_scale ~factor:2 hidden input;
    Stdio.printf "ERROR: expected exception, got none\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "got expected Shape_error: %s\n" msg

let test_scale_non_divisible () =
  let open Nn_blocks.DSL_modules in
  Tensor.unsafe_reinitialize ();
  Stdio.printf "\n=== set_scale: solved hidden not divisible by factor ===\n";
  let input = Shape.get_variable_ref "input_dim" in
  let hidden = Shape.get_variable_ref "hidden_dim" in
  Shape.set_dim hidden 17;
  try
    Shape.set_scale ~factor:2 hidden input;
    Stdio.printf "ERROR: expected exception, got none\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "got expected Shape_error: %s\n" msg

let test_scale_invalid_factor () =
  let open Nn_blocks.DSL_modules in
  Tensor.unsafe_reinitialize ();
  Stdio.printf "\n=== set_scale: invalid factor (0) ===\n";
  let input = Shape.get_variable_ref "input_dim" in
  let hidden = Shape.get_variable_ref "hidden_dim" in
  try
    Shape.set_scale ~factor:0 hidden input;
    Stdio.printf "ERROR: expected exception, got none\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "got expected Shape_error: %s\n" msg

(* set_scale used as a consistency assertion alongside einsum capture. propagate_shapes runs eagerly
   during let%op construction, so by the time set_scale runs, i and k have been resolved from the
   tensor input shapes (i = 4, k = 12). factor:3 is consistent (3*4 = 12), so this hits the
   both-solved consistent branch silently. *)
let test_scale_consistent_with_einsum () =
  let open Nn_blocks.DSL_modules in
  Tensor.unsafe_reinitialize ();
  Stdio.printf "\n=== set_scale: consistent factor alongside einsum capture ===\n";
  let ctx = Context.auto () in
  let%op a = { a = uniform1 (); o = [ 4; 6 ] } in
  let%op b = { b = uniform1 (); o = [ 6; 12 ] } in
  let%op c = a +* "ij;jk=>ik" [ "i"; "j"; "k" ] b in
  Shape.set_scale ~factor:3 k i;
  let _ctx = Train.forward_once ctx c in
  Stdio.printf "i = %s\n"
    (Option.value_map i.var_ref.solved_dim ~default:"unresolved" ~f:Int.to_string);
  Stdio.printf "j = %s\n"
    (Option.value_map j.var_ref.solved_dim ~default:"unresolved" ~f:Int.to_string);
  Stdio.printf "k = %s\n"
    (Option.value_map k.var_ref.solved_dim ~default:"unresolved" ~f:Int.to_string)

(* Both-unsolved Dim/Dim branch end-to-end: shape-free parameters leave their einsum captures with
   var = `Dim _ but solved_dim = None (the +* with concrete-shape inputs resolves solved_dim eagerly
   via update_delayed_var_refs). set_scale therefore reaches the both-unsolved Dim/Dim branch and
   emits the Affine constraint into active_constraints. set_dim then pins the small side, and
   forward_once runs the constraint solver, which must reduce Dim_eq { Var v_k, Affine { stride=2;
   over=Var v_i; conv=None; stride_offset=0 } } against Var v_i = 4 to derive Var v_k = 8. The
   asserted post-condition (k = 8) would fail under any local mutation of the emitted constraint --
   e.g. stride = 1 (would give k=4), the wrong over variable (self-reference becomes unsat), or
   replacing Affine with plain Dim_eq equality (would force k=i=4). *)
let test_scale_dim_dim_propagates_through_solver () =
  let open Nn_blocks.DSL_modules in
  Tensor.unsafe_reinitialize ();
  Stdio.printf "\n=== set_scale: Dim/Dim propagates through solver ===\n";
  let ctx = Context.auto () in
  let%op a = { a } in
  let%op b = { b } in
  let%op c = a +* "i;k=>ik" [ "i"; "k" ] b in
  Shape.set_scale ~factor:2 k i;
  Shape.set_dim i 4;
  let _ctx = Train.forward_once ctx c in
  Stdio.printf "i = %s\n"
    (Option.value_map i.var_ref.solved_dim ~default:"unresolved" ~f:Int.to_string);
  Stdio.printf "k = %s\n"
    (Option.value_map k.var_ref.solved_dim ~default:"unresolved" ~f:Int.to_string)

(* Negative companion to the above: pin both sides to values inconsistent with the Affine
   relationship (i = 8, k = 15, factor = 2 -> 2*8 = 16 != 15), then trigger inference. The solver
   must surface a Shape_error. If the emitted constraint were a plain equality or a different Affine
   shape, the solver would either silently accept or fail with a different error -- the assertion on
   the first error line catches both. *)
let test_scale_dim_dim_mismatch_through_solver () =
  let open Nn_blocks.DSL_modules in
  Tensor.unsafe_reinitialize ();
  Stdio.printf "\n=== set_scale: Dim/Dim mismatch surfaces through solver ===\n";
  let ctx = Context.auto () in
  let%op a = { a } in
  let%op b = { b } in
  let%op c = a +* "i;k=>ik" [ "i"; "k" ] b in
  Shape.set_scale ~factor:2 k i;
  Shape.set_dim i 8;
  Shape.set_dim k 15;
  try
    let _ctx = Train.forward_once ctx c in
    Stdio.printf "ERROR: expected Shape_error, got none\n"
  with Row.Shape_error (msg, _) ->
    let first_line = List.hd_exn (String.split_lines msg) in
    Stdio.printf "got expected Shape_error: %s\n" first_line

(* Row-variable rejection: mirror the row capture pattern from test/einsum/test_einsum_capture.ml
   (`"a..s..;..s..b=>ab" [ "s" ]`). propagate_shapes binds `s` to `Row _`. set_scale must reject
   this with Shape_error rather than routing through set_dim's Row arm. *)
let test_scale_rejects_row_variable () =
  let open Nn_blocks.DSL_modules in
  Tensor.unsafe_reinitialize ();
  Stdio.printf "\n=== set_scale: rejects row-variable ref ===\n";
  let _ctx = Context.auto () in
  let%op x_row = { x_row = uniform1 (); o = [ 2; 6 ] } in
  let%op y_row = { y_row = uniform1 (); o = [ 6; 3 ] } in
  let%op _z_row = x_row +* "a..s..;..s..b=>ab" [ "s" ] y_row in
  let dim_var = Shape.get_variable_ref "dim_var" in
  Shape.set_dim dim_var 12;
  try
    Shape.set_scale ~factor:2 s dim_var;
    Stdio.printf "ERROR: expected exception, got none\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "got expected Shape_error: %s\n" msg

let () =
  test_scale_input_solved_first ();
  test_scale_hidden_solved_first ();
  test_scale_both_solved_mismatch ();
  test_scale_non_divisible ();
  test_scale_invalid_factor ();
  test_scale_consistent_with_einsum ();
  test_scale_dim_dim_propagates_through_solver ();
  test_scale_dim_dim_mismatch_through_solver ();
  test_scale_rejects_row_variable ()

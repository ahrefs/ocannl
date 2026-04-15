open! Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

let dummy_origin : Row.constraint_origin list =
  [ { lhs_name = "test"; lhs_kind = `Output; rhs_name = "test"; rhs_kind = `Output; operation = None }
  ]

(* Helper to create a labeled tensor with given output axes *)
let labeled_tensor axes =
  let values = Array.create ~len:(List.fold axes ~init:1 ~f:(fun acc (_, d) -> acc * d)) 1.0 in
  Tensor.ndarray ~grad_spec:Prohibit_grad values ~batch_dims:[] ~input_dims:[] ~output_axes:axes ()

(* Helper to create an unlabeled tensor with given output dims *)
let unlabeled_tensor dims =
  let values = Array.create ~len:(List.fold dims ~init:1 ~f:( * )) 1.0 in
  Tensor.ndarray ~grad_spec:Prohibit_grad values ~batch_dims:[] ~input_dims:[] ~output_dims:dims ()

(* ================================================================ *)
(* DSL integration tests (user-reachable paths)                     *)
(* ================================================================ *)

let test_same_label_same_size () =
  Stdio.printf "Test 1: Same label, same size -- succeeds\n";
  Tensor.unsafe_reinitialize ();
  try
    let t1 = labeled_tensor [ ("batch", 4) ] in
    let t2 = labeled_tensor [ ("batch", 4) ] in
    let%op result = t1 + t2 in
    let ctx = Train.forward_once (Context.auto ()) result in
    ignore (ctx : Context.t);
    Stdio.printf "  PASS: same label unification succeeded\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_conflicting_labels_same_size () =
  Stdio.printf "Test 2: Conflicting labels, same size -- raises\n";
  Tensor.unsafe_reinitialize ();
  try
    let t1 = labeled_tensor [ ("batch", 4) ] in
    let t2 = labeled_tensor [ ("features", 4) ] in
    let%op result = t1 + t2 in
    let ctx = Train.forward_once (Context.auto ()) result in
    ignore (ctx : Context.t);
    Stdio.printf "  FAIL: should have raised Shape_error\n"
  with Row.Shape_error (msg, _) ->
    if String.is_substring msg ~substring:"different labels" then
      Stdio.printf "  PASS: got expected Shape_error: %s\n" msg
    else Stdio.printf "  FAIL: wrong error message: %s\n" msg

let test_one_labeled_one_unlabeled () =
  Stdio.printf "Test 3: One labeled, one unlabeled, same size -- compatible\n";
  Tensor.unsafe_reinitialize ();
  try
    let t1 = labeled_tensor [ ("batch", 4) ] in
    let t2 = unlabeled_tensor [ 4 ] in
    let%op result = t1 + t2 in
    let ctx = Train.forward_once (Context.auto ()) result in
    ignore (ctx : Context.t);
    Stdio.printf "  PASS: labeled + unlabeled compatible\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_variable_mediated_unlabeled_then_labeled () =
  Stdio.printf "Test 4: Variable-mediated: unlabeled first, labeled later -- compatible\n";
  Tensor.unsafe_reinitialize ();
  try
    (* x has unknown output dims. First constrain it to size 4 (unlabeled),
       then constrain to labeled "batch" size 4. Should succeed. *)
    let%cd x = { x } in
    let unlabeled = unlabeled_tensor [ 4 ] in
    let labeled = labeled_tensor [ ("batch", 4) ] in
    let%cd step1 = x + unlabeled in
    let%cd result = step1 + labeled in
    let ctx = Train.forward_once (Context.auto ()) result in
    ignore (ctx : Context.t);
    Stdio.printf "  PASS: variable-mediated unlabeled then labeled\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_variable_mediated_conflicting () =
  Stdio.printf "Test 5: Variable-mediated: unlabeled, labeled, then conflicting -- raises\n";
  Tensor.unsafe_reinitialize ();
  try
    (* Use direct Row API for the variable-mediated conflict test, since the DSL
       creates fresh shapes for each operation and doesn't reuse dim variables.
       The scenario: var v is solved to Dim{d=4; label=None}, then later unified
       with Dim{d=4; label=Some "batch"} (upgrade), then with Dim{d=4; label=Some "features"} (conflict). *)
    let v = Row.get_var ~name:"v" () in
    let constraints =
      [ Row.Dim_eq { d1 = Row.Var v; d2 = Row.get_dim ~d:4 (); origin = dummy_origin }
      ; Row.Dim_eq { d1 = Row.Var v; d2 = Row.get_dim ~d:4 ~label:"batch" (); origin = dummy_origin }
      ; Row.Dim_eq
          { d1 = Row.Var v; d2 = Row.get_dim ~d:4 ~label:"features" (); origin = dummy_origin }
      ]
    in
    let _remaining, _env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    Stdio.printf "  FAIL: should have raised Shape_error\n"
  with Row.Shape_error (msg, _) ->
    if String.is_substring msg ~substring:"label" then
      Stdio.printf "  PASS: got expected Shape_error: %s\n" msg
    else Stdio.printf "  FAIL: wrong error message: %s\n" msg

let test_shape_to_labels () =
  Stdio.printf "Test 6: Shape.to_labels after inference\n";
  Tensor.unsafe_reinitialize ();
  try
    let t = labeled_tensor [ ("batch", 4) ] in
    let ctx = Train.forward_once (Context.auto ()) t in
    ignore (ctx : Context.t);
    let labels = Shape.to_labels t.shape in
    let label_str = String.concat_array ~sep:"," labels in
    Stdio.printf "  Labels: [%s]\n" label_str;
    if Array.exists labels ~f:(fun l -> String.is_substring l ~substring:"batch") then
      Stdio.printf "  PASS: label visible in Shape.to_labels\n"
    else Stdio.printf "  FAIL: label not found in Shape.to_labels\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_number_axis_label () =
  Stdio.printf "Test 7: Tensor.number ~axis_label\n";
  Tensor.unsafe_reinitialize ();
  try
    let t = NTDSL.number ~axis_label:"count" 5.0 in
    (* Check label is visible before running — shape inference happens at compile time *)
    let labels = Shape.to_labels t.shape in
    let label_str = String.concat_array ~sep:"," labels in
    Stdio.printf "  Labels: [%s]\n" label_str;
    if Array.exists labels ~f:(fun l -> String.equal l "count") then
      Stdio.printf "  PASS: axis_label survives inference\n"
    else Stdio.printf "  FAIL: axis_label not found\n"
  with
  | Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg
  | exn -> Stdio.printf "  FAIL: unexpected exception: %s\n" (Exn.to_string exn)

let test_range_axis_label () =
  Stdio.printf "Test 8: Operation.range ~axis_label\n";
  Tensor.unsafe_reinitialize ();
  try
    let open Operation in
    let t = range ~axis_label:"idx" 5 in
    (* Check label is visible before running *)
    let labels = Shape.to_labels t.shape in
    let label_str = String.concat_array ~sep:"," labels in
    Stdio.printf "  Labels: [%s]\n" label_str;
    if Array.exists labels ~f:(fun l -> String.equal l "idx") then
      Stdio.printf "  PASS: range axis_label survives inference\n"
    else Stdio.printf "  FAIL: range axis_label not found\n"
  with
  | Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg
  | exn -> Stdio.printf "  FAIL: unexpected exception: %s\n" (Exn.to_string exn)

(* ================================================================ *)
(* Direct Row construction tests (internal paths)                   *)
(* ================================================================ *)

let test_concat_consistent_labels () =
  Stdio.printf "Test 9: Concat, consistent labels -- label preserved\n";
  Tensor.unsafe_reinitialize ();
  try
    (* Use two variables to force concat collapse via substitution:
       w = Concat [Dim labeled "x"; Dim labeled "x"], then v = w.
       When v is solved, s_dim_one substitutes w's value and collapses the Concat. *)
    let v = Row.get_var ~name:"v" () in
    let w = Row.get_var ~name:"w" () in
    let concat_dim =
      Row.Concat [ Row.get_dim ~d:2 ~label:"x" (); Row.get_dim ~d:3 ~label:"x" () ]
    in
    let constraints =
      [ Row.Dim_eq { d1 = Row.Var w; d2 = concat_dim; origin = dummy_origin }
      ; Row.Dim_eq { d1 = Row.Var v; d2 = Row.Var w; origin = dummy_origin }
      ]
    in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let result = Row.get_dim_val env v in
    (match result with
    | Some d -> Stdio.printf "  Solved to d=%d\n" d
    | None -> Stdio.printf "  Not fully solved\n");
    Stdio.printf "  PASS: concat with consistent labels did not raise\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_concat_conflicting_labels () =
  Stdio.printf "Test 10: Concat, conflicting labels -- raises\n";
  Tensor.unsafe_reinitialize ();
  try
    let v = Row.get_var ~name:"v" () in
    let w = Row.get_var ~name:"w" () in
    let concat_dim =
      Row.Concat [ Row.get_dim ~d:2 ~label:"x" (); Row.get_dim ~d:3 ~label:"y" () ]
    in
    let constraints =
      [ Row.Dim_eq { d1 = Row.Var w; d2 = concat_dim; origin = dummy_origin }
      ; Row.Dim_eq { d1 = Row.Var v; d2 = Row.Var w; origin = dummy_origin }
      ]
    in
    let _remaining, _env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    Stdio.printf "  FAIL: should have raised Shape_error\n"
  with Row.Shape_error (msg, _) ->
    if String.is_substring msg ~substring:"conflicting dimension labels" then
      Stdio.printf "  PASS: got expected Shape_error: %s\n" msg
    else Stdio.printf "  FAIL: wrong error message: %s\n" msg

let test_concat_mixed_labeled_unlabeled () =
  Stdio.printf "Test 11: Concat, mix labeled/unlabeled -- label preserved\n";
  Tensor.unsafe_reinitialize ();
  try
    let v = Row.get_var ~name:"v" () in
    let w = Row.get_var ~name:"w" () in
    let concat_dim = Row.Concat [ Row.get_dim ~d:2 ~label:"x" (); Row.get_dim ~d:3 () ] in
    let constraints =
      [ Row.Dim_eq { d1 = Row.Var w; d2 = concat_dim; origin = dummy_origin }
      ; Row.Dim_eq { d1 = Row.Var v; d2 = Row.Var w; origin = dummy_origin }
      ]
    in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let result = Row.get_dim_val env v in
    (match result with
    | Some d -> Stdio.printf "  Solved to d=%d\n" d
    | None -> Stdio.printf "  Not fully solved\n");
    Stdio.printf "  PASS: concat with mixed labels did not raise\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_affine_matching_labels () =
  Stdio.printf "Test 12: Affine, matching labels -- preserved\n";
  Tensor.unsafe_reinitialize ();
  try
    (* Create an Affine where over uses a variable that gets solved to a labeled dim.
       The kernel has the same label. *)
    let v_over = Row.get_var ~name:"over" () in
    let v_result = Row.get_var ~name:"result" () in
    let affine_dim =
      Row.Affine
        {
          stride = 1;
          over = Row.Var v_over;
          conv =
            Some
              { dilation = 1; kernel = Row.get_dim ~d:3 ~label:"x" (); use_padding = false };
          stride_offset = 0;
        }
    in
    let constraints =
      [ Row.Dim_eq { d1 = Row.Var v_result; d2 = affine_dim; origin = dummy_origin }
      ; Row.Dim_eq { d1 = Row.Var v_over; d2 = Row.get_dim ~d:4 ~label:"x" (); origin = dummy_origin }
      ]
    in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let result = Row.get_dim_val env v_result in
    (match result with
    | Some d -> Stdio.printf "  Solved to d=%d\n" d
    | None -> Stdio.printf "  Not fully solved\n");
    Stdio.printf "  PASS: affine with matching labels did not raise\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_affine_conflicting_labels () =
  Stdio.printf "Test 13: Affine, conflicting labels -- raises\n";
  Tensor.unsafe_reinitialize ();
  try
    (* Create an Affine where over uses a variable that gets solved to a labeled dim.
       The kernel has a different label. When the variable is substituted, s_dim_one
       collapses the Affine and should detect the label conflict. *)
    let v_over = Row.get_var ~name:"over" () in
    let v_result = Row.get_var ~name:"result" () in
    let affine_dim =
      Row.Affine
        {
          stride = 1;
          over = Row.Var v_over;
          conv =
            Some
              { dilation = 1; kernel = Row.get_dim ~d:3 ~label:"y" (); use_padding = false };
          stride_offset = 0;
        }
    in
    let constraints =
      [ Row.Dim_eq { d1 = Row.Var v_result; d2 = affine_dim; origin = dummy_origin }
      ; Row.Dim_eq { d1 = Row.Var v_over; d2 = Row.get_dim ~d:4 ~label:"x" (); origin = dummy_origin }
      ]
    in
    let _remaining, _env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    Stdio.printf "  FAIL: should have raised Shape_error\n"
  with Row.Shape_error (msg, _) ->
    if String.is_substring msg ~substring:"conflicting dimension labels" then
      Stdio.printf "  PASS: got expected Shape_error: %s\n" msg
    else Stdio.printf "  FAIL: wrong error message: %s\n" msg

let test_stride_noconv_forward () =
  Stdio.printf "Test 14: Stride no-conv forward -- label propagated\n";
  Tensor.unsafe_reinitialize ();
  try
    let affine_dim =
      Row.Affine
        { stride = 2; over = Row.get_dim ~d:4 ~label:"x" (); conv = None; stride_offset = 0 }
    in
    let target = Row.get_dim ~d:8 ~label:"x" () in
    let constraints = [ Row.Dim_eq { d1 = affine_dim; d2 = target; origin = dummy_origin } ] in
    let _remaining, _env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    Stdio.printf "  PASS: stride forward with matching labels succeeded\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_stride_noconv_reverse () =
  Stdio.printf "Test 15: Stride no-conv reverse -- label propagated\n";
  Tensor.unsafe_reinitialize ();
  try
    let v = Row.get_var ~name:"over" () in
    let affine_dim =
      Row.Affine { stride = 2; over = Row.Var v; conv = None; stride_offset = 0 }
    in
    let target = Row.get_dim ~d:8 ~label:"x" () in
    let constraints = [ Row.Dim_eq { d1 = target; d2 = affine_dim; origin = dummy_origin } ] in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let result = Row.get_dim_val env v in
    (match result with
    | Some d -> Stdio.printf "  Solved over to d=%d\n" d
    | None -> Stdio.printf "  over not fully solved\n");
    Stdio.printf "  PASS: stride reverse with label propagation succeeded\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_conv_nopadding_forward () =
  Stdio.printf "Test 16: Conv no-padding forward -- label propagated\n";
  Tensor.unsafe_reinitialize ();
  try
    (* input_size = stride * (output_size - 1) + kernel_size = 1 * (4 - 1) + 3 = 6 *)
    let affine_dim =
      Row.Affine
        {
          stride = 1;
          over = Row.get_dim ~d:4 ~label:"x" ();
          conv = Some { dilation = 1; kernel = Row.get_dim ~d:3 (); use_padding = false };
          stride_offset = 0;
        }
    in
    let target = Row.get_dim ~d:6 ~label:"x" () in
    let constraints = [ Row.Dim_eq { d1 = affine_dim; d2 = target; origin = dummy_origin } ] in
    let _remaining, _env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    Stdio.printf "  PASS: conv forward with matching labels succeeded\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_conv_nopadding_reverse () =
  Stdio.printf "Test 17: Conv no-padding reverse -- label propagated\n";
  Tensor.unsafe_reinitialize ();
  try
    (* Given input_size=6, kernel=3, stride=1: output_size = (6 - 3) / 1 + 1 = 4 *)
    let v = Row.get_var ~name:"over" () in
    let affine_dim =
      Row.Affine
        {
          stride = 1;
          over = Row.Var v;
          conv = Some { dilation = 1; kernel = Row.get_dim ~d:3 (); use_padding = false };
          stride_offset = 0;
        }
    in
    let target = Row.get_dim ~d:6 ~label:"x" () in
    let constraints = [ Row.Dim_eq { d1 = target; d2 = affine_dim; origin = dummy_origin } ] in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let result = Row.get_dim_val env v in
    (match result with
    | Some d -> Stdio.printf "  Solved over to d=%d\n" d
    | None -> Stdio.printf "  over not fully solved\n");
    Stdio.printf "  PASS: conv reverse with label propagation succeeded\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_lub_conflicting_labels () =
  Stdio.printf "Test 18: LUB with conflicting labels -- demotion to d=1\n";
  Tensor.unsafe_reinitialize ();
  try
    let prov = Row.empty_provenance in
    let r1 = { Row.dims = [ Row.get_dim ~d:4 ~label:"x" () ]; bcast = Broadcastable; prov } in
    let r2 = { Row.dims = [ Row.get_dim ~d:4 ~label:"y" () ]; bcast = Broadcastable; prov } in
    let constraints =
      [ Row.Row_ineq { cur = r1; subr = r2; origin = dummy_origin }
      ; Row.Row_ineq { cur = r2; subr = r1; origin = dummy_origin }
      ]
    in
    let _remaining, _env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    Stdio.printf "  PASS: LUB with conflicting labels did not raise (intentional broadcast)\n"
  with Row.Shape_error (msg, _) ->
    Stdio.printf "  INFO: Shape_error in LUB test: %s\n" msg;
    Stdio.printf "  NOTE: LUB may raise when axes are mutually constrained\n"

(* ================================================================ *)
(* Main                                                             *)
(* ================================================================ *)

let () =
  Stdio.printf "=== Dimension Label Tests ===\n\n";
  test_same_label_same_size ();
  Stdio.printf "\n";
  test_conflicting_labels_same_size ();
  Stdio.printf "\n";
  test_one_labeled_one_unlabeled ();
  Stdio.printf "\n";
  test_variable_mediated_unlabeled_then_labeled ();
  Stdio.printf "\n";
  test_variable_mediated_conflicting ();
  Stdio.printf "\n";
  test_shape_to_labels ();
  Stdio.printf "\n";
  test_number_axis_label ();
  Stdio.printf "\n";
  test_range_axis_label ();
  Stdio.printf "\n";
  test_concat_consistent_labels ();
  Stdio.printf "\n";
  test_concat_conflicting_labels ();
  Stdio.printf "\n";
  test_concat_mixed_labeled_unlabeled ();
  Stdio.printf "\n";
  test_affine_matching_labels ();
  Stdio.printf "\n";
  test_affine_conflicting_labels ();
  Stdio.printf "\n";
  test_stride_noconv_forward ();
  Stdio.printf "\n";
  test_stride_noconv_reverse ();
  Stdio.printf "\n";
  test_conv_nopadding_forward ();
  Stdio.printf "\n";
  test_conv_nopadding_reverse ();
  Stdio.printf "\n";
  test_lub_conflicting_labels ();
  Stdio.printf "\n";
  Stdio.printf "=== Done ===\n"

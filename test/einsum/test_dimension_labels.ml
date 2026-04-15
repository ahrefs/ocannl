open! Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

let dummy_origin : Row.constraint_origin list =
  [ { lhs_name = "test"; lhs_kind = `Output; rhs_name = "test"; rhs_kind = `Output; operation = None }
  ]

(* Helper: get the label string for a dim_var from the environment using row_to_labels.
   Returns the label or "" if unlabeled. *)
let get_var_label env (v : Row.dim_var) : string =
  let prov = Row.empty_provenance in
  let row = { Row.dims = [ Row.Var v ]; bcast = Broadcastable; prov } in
  let labels = Row.row_to_labels env row in
  if Array.length labels > 0 then labels.(0) else ""

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
    let labels = Shape.to_labels result.shape in
    let label_str = String.concat_array ~sep:"," labels in
    Stdio.printf "  Result labels: [%s]\n" label_str;
    if Array.exists labels ~f:(fun l -> String.equal l "batch") then
      Stdio.printf "  PASS: label preserved in result\n"
    else Stdio.printf "  FAIL: label not preserved in result\n"
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
    let labels = Shape.to_labels result.shape in
    let label_str = String.concat_array ~sep:"," labels in
    Stdio.printf "  Result labels: [%s]\n" label_str;
    Stdio.printf "  PASS: labeled + unlabeled compatible\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_variable_mediated_unlabeled_then_labeled () =
  Stdio.printf "Test 4: Variable-mediated: unlabeled first, labeled later -- compatible\n";
  Tensor.unsafe_reinitialize ();
  try
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

let test_variable_mediated_label_upgrade () =
  Stdio.printf "Test 5b: Variable-mediated: label upgrade verified in environment\n";
  Tensor.unsafe_reinitialize ();
  try
    (* Var v is solved to unlabeled d=4, then unified with labeled d=4.
       After the second unification, the environment should show the label. *)
    let v = Row.get_var ~name:"v" () in
    let constraints =
      [ Row.Dim_eq { d1 = Row.Var v; d2 = Row.get_dim ~d:4 (); origin = dummy_origin }
      ; Row.Dim_eq { d1 = Row.Var v; d2 = Row.get_dim ~d:4 ~label:"batch" (); origin = dummy_origin }
      ]
    in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let label = get_var_label env v in
    Stdio.printf "  Var v label after upgrade: \"%s\"\n" label;
    if String.equal label "batch" then
      Stdio.printf "  PASS: variable label upgraded to \"batch\"\n"
    else Stdio.printf "  FAIL: expected label \"batch\", got \"%s\"\n" label
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

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
    (* Concat collapse happens in s_dim_one when a variable inside the Concat is substituted.
       Create a Concat with a Var component so that when the var is solved, s_dim_one
       collapses the Concat and we can verify the label on the resulting Dim. *)
    let v_result = Row.get_var ~name:"result" () in
    let v_component = Row.get_var ~name:"comp" () in
    let concat_dim =
      Row.Concat [ Row.Var v_component; Row.get_dim ~d:3 ~label:"x" () ]
    in
    let constraints =
      [ Row.Dim_eq { d1 = Row.Var v_result; d2 = concat_dim; origin = dummy_origin }
      ; Row.Dim_eq { d1 = Row.Var v_component; d2 = Row.get_dim ~d:2 ~label:"x" (); origin = dummy_origin }
      ]
    in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let d = Row.get_dim_val env v_result in
    let label = get_var_label env v_result in
    Stdio.printf "  Solved to d=%s, label=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string) label;
    if Option.equal Int.equal d (Some 5) && String.equal label "x" then
      Stdio.printf "  PASS: concat preserved label \"x\" with d=5\n"
    else Stdio.printf "  FAIL: expected d=5, label=\"x\"\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_concat_conflicting_labels () =
  Stdio.printf "Test 10: Concat, conflicting labels -- raises\n";
  Tensor.unsafe_reinitialize ();
  try
    let v_result = Row.get_var ~name:"result" () in
    let v_component = Row.get_var ~name:"comp" () in
    let concat_dim =
      Row.Concat [ Row.Var v_component; Row.get_dim ~d:3 ~label:"y" () ]
    in
    let constraints =
      [ Row.Dim_eq { d1 = Row.Var v_result; d2 = concat_dim; origin = dummy_origin }
      ; Row.Dim_eq { d1 = Row.Var v_component; d2 = Row.get_dim ~d:2 ~label:"x" (); origin = dummy_origin }
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
    (* Same as test 9 but one component is unlabeled — the label from the labeled component
       should be preserved in the collapsed Dim. *)
    let v_result = Row.get_var ~name:"result" () in
    let v_component = Row.get_var ~name:"comp" () in
    let concat_dim =
      Row.Concat [ Row.Var v_component; Row.get_dim ~d:3 () ]
    in
    let constraints =
      [ Row.Dim_eq { d1 = Row.Var v_result; d2 = concat_dim; origin = dummy_origin }
      ; Row.Dim_eq { d1 = Row.Var v_component; d2 = Row.get_dim ~d:2 ~label:"x" (); origin = dummy_origin }
      ]
    in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let d = Row.get_dim_val env v_result in
    let label = get_var_label env v_result in
    Stdio.printf "  Solved to d=%s, label=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string) label;
    if Option.equal Int.equal d (Some 5) && String.equal label "x" then
      Stdio.printf "  PASS: concat preserved label \"x\" from labeled component\n"
    else Stdio.printf "  FAIL: expected d=5, label=\"x\"\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_affine_matching_labels () =
  Stdio.printf "Test 12: Affine, matching labels -- preserved\n";
  Tensor.unsafe_reinitialize ();
  try
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
    let d = Row.get_dim_val env v_result in
    let label = get_var_label env v_result in
    (* input_size = 1 * (4 - 1) + 3 = 6, label should be "x" *)
    Stdio.printf "  Solved to d=%s, label=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string) label;
    if Option.equal Int.equal d (Some 6) && String.equal label "x" then
      Stdio.printf "  PASS: affine preserved label \"x\" with d=6\n"
    else Stdio.printf "  FAIL: expected d=6, label=\"x\"\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_affine_conflicting_labels () =
  Stdio.printf "Test 13: Affine, conflicting labels -- raises\n";
  Tensor.unsafe_reinitialize ();
  try
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
    (* Affine{stride=2; over=Dim{d=4; label="x"}} should produce d=8 with label "x" *)
    let v = Row.get_var ~name:"result" () in
    let affine_dim =
      Row.Affine
        { stride = 2; over = Row.get_dim ~d:4 ~label:"x" (); conv = None; stride_offset = 0 }
    in
    let constraints =
      [ Row.Dim_eq { d1 = Row.Var v; d2 = affine_dim; origin = dummy_origin } ] in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let d = Row.get_dim_val env v in
    let label = get_var_label env v in
    Stdio.printf "  Solved to d=%s, label=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string) label;
    if Option.equal Int.equal d (Some 8) && String.equal label "x" then
      Stdio.printf "  PASS: stride forward propagated label \"x\"\n"
    else Stdio.printf "  FAIL: expected d=8, label=\"x\"\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_stride_noconv_reverse () =
  Stdio.printf "Test 15: Stride no-conv reverse -- label propagated\n";
  Tensor.unsafe_reinitialize ();
  try
    (* Dim{d=8; label="x"} unified with Affine{stride=2; over=Var v}
       should solve v to d=4 with label "x" propagated via get_dim *)
    let v = Row.get_var ~name:"over" () in
    let affine_dim =
      Row.Affine { stride = 2; over = Row.Var v; conv = None; stride_offset = 0 }
    in
    let target = Row.get_dim ~d:8 ~label:"x" () in
    let constraints = [ Row.Dim_eq { d1 = target; d2 = affine_dim; origin = dummy_origin } ] in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let d = Row.get_dim_val env v in
    let label = get_var_label env v in
    Stdio.printf "  Solved over to d=%s, label=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string) label;
    if Option.equal Int.equal d (Some 4) && String.equal label "x" then
      Stdio.printf "  PASS: stride reverse propagated label \"x\" to over\n"
    else Stdio.printf "  FAIL: expected d=4, label=\"x\"\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_conv_nopadding_forward () =
  Stdio.printf "Test 16: Conv no-padding forward -- label propagated\n";
  Tensor.unsafe_reinitialize ();
  try
    (* input_size = stride * (output_size - 1) + kernel_size = 1 * (4 - 1) + 3 = 6 *)
    let v = Row.get_var ~name:"result" () in
    let affine_dim =
      Row.Affine
        {
          stride = 1;
          over = Row.get_dim ~d:4 ~label:"x" ();
          conv = Some { dilation = 1; kernel = Row.get_dim ~d:3 (); use_padding = false };
          stride_offset = 0;
        }
    in
    let constraints =
      [ Row.Dim_eq { d1 = Row.Var v; d2 = affine_dim; origin = dummy_origin } ] in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let d = Row.get_dim_val env v in
    let label = get_var_label env v in
    Stdio.printf "  Solved to d=%s, label=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string) label;
    if Option.equal Int.equal d (Some 6) && String.equal label "x" then
      Stdio.printf "  PASS: conv forward propagated label \"x\"\n"
    else Stdio.printf "  FAIL: expected d=6, label=\"x\"\n"
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
    let d = Row.get_dim_val env v in
    let label = get_var_label env v in
    Stdio.printf "  Solved over to d=%s, label=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string) label;
    if Option.equal Int.equal d (Some 4) && String.equal label "x" then
      Stdio.printf "  PASS: conv reverse propagated label \"x\" to over\n"
    else Stdio.printf "  FAIL: expected d=4, label=\"x\"\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_lub_conflicting_labels () =
  Stdio.printf "Test 18: LUB with conflicting labels -- strict equality raises\n";
  Tensor.unsafe_reinitialize ();
  (* The dim-level inequality solver (solve_dim_ineq) checks labels strictly at the
     Dim/Dim fast path before reaching the LUB computation. When two concrete dims
     with same size but different labels meet through mutual Row_ineq constraints,
     the strict check raises Shape_error. This is correct behavior: the LUB demotion
     to d=1 only applies within the Bounds_dim path, not when both dims are already
     solved. Verify that the error is raised. *)
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
    Stdio.printf "  FAIL: should have raised Shape_error for conflicting labels\n"
  with Row.Shape_error (msg, _) ->
    if String.is_substring msg ~substring:"different labels" then
      Stdio.printf "  PASS: conflicting labels in mutual inequality correctly raises: %s\n" msg
    else Stdio.printf "  FAIL: wrong error message: %s\n" msg

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
  test_variable_mediated_label_upgrade ();
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

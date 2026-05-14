open! Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

let dummy_origin : Row.constraint_origin list =
  [
    {
      lhs_name = "test";
      lhs_kind = `Output;
      rhs_name = "test";
      rhs_kind = `Output;
      operation = None;
    };
  ]

(* Helper: get the basis string for a dim_var from the environment using row_to_bases. Returns the
   basis or "" if unbased. *)
let get_var_basis env (v : Row.dim_var) : string =
  let prov = Row.empty_provenance in
  let row = { Row.dims = [ Row.Var v ]; bcast = Broadcastable; prov } in
  let bases = Row.row_to_bases env row in
  if Array.length bases > 0 then bases.(0) else ""

(* Helper to create a based tensor with given output axes *)
let based_tensor axes =
  let values = Array.create ~len:(List.fold axes ~init:1 ~f:(fun acc (_, d) -> acc * d)) 1.0 in
  Tensor.ndarray ~grad_spec:Prohibit_grad values ~batch_dims:[] ~input_dims:[] ~output_axes:axes ()

(* Helper to create an unbased tensor with given output dims *)
let unbased_tensor dims =
  let values = Array.create ~len:(List.fold dims ~init:1 ~f:( * )) 1.0 in
  Tensor.ndarray ~grad_spec:Prohibit_grad values ~batch_dims:[] ~input_dims:[] ~output_dims:dims ()

(* ================================================================ *)
(* DSL integration tests (user-reachable paths)                     *)
(* ================================================================ *)

let test_same_basis_same_size () =
  Stdio.printf "Test 1: Same basis, same size -- succeeds\n";
  Tensor.unsafe_reinitialize ();
  try
    let t1 = based_tensor [ ("batch", 4) ] in
    let t2 = based_tensor [ ("batch", 4) ] in
    let%op result = t1 + t2 in
    let ctx = Train.forward_once (Context.auto ()) result in
    ignore (ctx : Context.t);
    let bases = Shape.to_bases result.shape in
    let basis_str = String.concat_array ~sep:"," bases in
    Stdio.printf "  Result bases: [%s]\n" basis_str;
    if Array.exists bases ~f:(fun b -> String.equal b "batch") then
      Stdio.printf "  PASS: basis preserved in result\n"
    else Stdio.printf "  FAIL: basis not preserved in result\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_conflicting_bases_same_size () =
  Stdio.printf "Test 2: Conflicting bases, same size -- raises\n";
  Tensor.unsafe_reinitialize ();
  try
    let t1 = based_tensor [ ("batch", 4) ] in
    let t2 = based_tensor [ ("features", 4) ] in
    let%op result = t1 + t2 in
    let ctx = Train.forward_once (Context.auto ()) result in
    ignore (ctx : Context.t);
    Stdio.printf "  FAIL: should have raised Shape_error\n"
  with Row.Shape_error (msg, _) ->
    if String.is_substring msg ~substring:"different bases" then
      Stdio.printf "  PASS: got expected Shape_error: %s\n" msg
    else Stdio.printf "  FAIL: wrong error message: %s\n" msg

let test_one_based_one_unbased () =
  Stdio.printf "Test 3: One based, one unbased, same size -- compatible\n";
  Tensor.unsafe_reinitialize ();
  try
    let t1 = based_tensor [ ("batch", 4) ] in
    let t2 = unbased_tensor [ 4 ] in
    let%op result = t1 + t2 in
    let ctx = Train.forward_once (Context.auto ()) result in
    ignore (ctx : Context.t);
    let bases = Shape.to_bases result.shape in
    let basis_str = String.concat_array ~sep:"," bases in
    Stdio.printf "  Result bases: [%s]\n" basis_str;
    Stdio.printf "  PASS: based + unbased compatible\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_variable_mediated_unbased_then_based () =
  Stdio.printf "Test 4: Variable-mediated: unbased first, based later -- compatible\n";
  Tensor.unsafe_reinitialize ();
  try
    let%cd x = { x } in
    let unbased = unbased_tensor [ 4 ] in
    let based = based_tensor [ ("batch", 4) ] in
    let%cd step1 = x + unbased in
    let%cd result = step1 + based in
    let ctx = Train.forward_once (Context.auto ()) result in
    ignore (ctx : Context.t);
    Stdio.printf "  PASS: variable-mediated unbased then based\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_variable_mediated_conflicting () =
  Stdio.printf "Test 5: Variable-mediated: unbased, based, then conflicting -- raises\n";
  Tensor.unsafe_reinitialize ();
  try
    let v = Row.get_var ~name:"v" () in
    let constraints =
      [
        Row.Dim_eq { d1 = Row.Var v; d2 = Row.get_dim ~d:4 (); origin = dummy_origin };
        Row.Dim_eq
          { d1 = Row.Var v; d2 = Row.get_dim ~d:4 ~basis:"batch" (); origin = dummy_origin };
        Row.Dim_eq
          { d1 = Row.Var v; d2 = Row.get_dim ~d:4 ~basis:"features" (); origin = dummy_origin };
      ]
    in
    let _remaining, _env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    Stdio.printf "  FAIL: should have raised Shape_error\n"
  with Row.Shape_error (msg, _) ->
    if String.is_substring msg ~substring:"bases" then
      Stdio.printf "  PASS: got expected Shape_error: %s\n" msg
    else Stdio.printf "  FAIL: wrong error message: %s\n" msg

let test_variable_mediated_basis_upgrade () =
  Stdio.printf "Test 5b: Variable-mediated: basis upgrade verified in environment\n";
  Tensor.unsafe_reinitialize ();
  try
    (* Var v is solved to unbased d=4, then unified with based d=4. After the second unification,
       the environment should show the basis. *)
    let v = Row.get_var ~name:"v" () in
    let constraints =
      [
        Row.Dim_eq { d1 = Row.Var v; d2 = Row.get_dim ~d:4 (); origin = dummy_origin };
        Row.Dim_eq
          { d1 = Row.Var v; d2 = Row.get_dim ~d:4 ~basis:"batch" (); origin = dummy_origin };
      ]
    in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let basis = get_var_basis env v in
    Stdio.printf "  Var v basis after upgrade: \"%s\"\n" basis;
    if String.equal basis "batch" then Stdio.printf "  PASS: variable basis upgraded to \"batch\"\n"
    else Stdio.printf "  FAIL: expected basis \"batch\", got \"%s\"\n" basis
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_shape_to_bases () =
  Stdio.printf "Test 6: Shape.to_bases after inference\n";
  Tensor.unsafe_reinitialize ();
  try
    let t = based_tensor [ ("batch", 4) ] in
    let ctx = Train.forward_once (Context.auto ()) t in
    ignore (ctx : Context.t);
    let bases = Shape.to_bases t.shape in
    let basis_str = String.concat_array ~sep:"," bases in
    Stdio.printf "  Bases: [%s]\n" basis_str;
    if Array.exists bases ~f:(fun b -> String.is_substring b ~substring:"batch") then
      Stdio.printf "  PASS: basis visible in Shape.to_bases\n"
    else Stdio.printf "  FAIL: basis not found in Shape.to_bases\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_number_axis_basis () =
  Stdio.printf "Test 7: Tensor.number ~axis_basis\n";
  Tensor.unsafe_reinitialize ();
  try
    let t = NTDSL.number ~axis_basis:"count" 5.0 in
    let bases = Shape.to_bases t.shape in
    let basis_str = String.concat_array ~sep:"," bases in
    Stdio.printf "  Bases: [%s]\n" basis_str;
    if Array.exists bases ~f:(fun b -> String.equal b "count") then
      Stdio.printf "  PASS: axis_basis survives inference\n"
    else Stdio.printf "  FAIL: axis_basis not found\n"
  with
  | Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg
  | exn -> Stdio.printf "  FAIL: unexpected exception: %s\n" (Exn.to_string exn)

let test_bits_axis_basis () =
  Stdio.printf "Test 7b: Tensor.bits ~axis_basis\n";
  Tensor.unsafe_reinitialize ();
  try
    let t = NTDSL.bits ~axis_basis:"word" 7L in
    let bases = Shape.to_bases t.shape in
    let basis_str = String.concat_array ~sep:"," bases in
    Stdio.printf "  Bases: [%s]\n" basis_str;
    if Array.exists bases ~f:(fun b -> String.equal b "word") then
      Stdio.printf "  PASS: bits axis_basis survives inference\n"
    else Stdio.printf "  FAIL: bits axis_basis not found\n"
  with
  | Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg
  | exn -> Stdio.printf "  FAIL: unexpected exception: %s\n" (Exn.to_string exn)

let test_range_axis_basis () =
  Stdio.printf "Test 8: Operation.range ~axis_basis\n";
  Tensor.unsafe_reinitialize ();
  try
    let open Operation in
    let t = range ~axis_basis:"idx" 5 in
    let bases = Shape.to_bases t.shape in
    let basis_str = String.concat_array ~sep:"," bases in
    Stdio.printf "  Bases: [%s]\n" basis_str;
    if Array.exists bases ~f:(fun b -> String.equal b "idx") then
      Stdio.printf "  PASS: range axis_basis survives inference\n"
    else Stdio.printf "  FAIL: range axis_basis not found\n"
  with
  | Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg
  | exn -> Stdio.printf "  FAIL: unexpected exception: %s\n" (Exn.to_string exn)

let test_number_int_axis_basis () =
  Stdio.printf "Test 8b: NTDSL.number_int ~axis_basis\n";
  Tensor.unsafe_reinitialize ();
  try
    let t = NTDSL.number_int ~axis_basis:"k" 3 in
    let bases = Shape.to_bases t.shape in
    let basis_str = String.concat_array ~sep:"," bases in
    Stdio.printf "  Bases: [%s]\n" basis_str;
    if Array.exists bases ~f:(fun b -> String.equal b "k") then
      Stdio.printf "  PASS: number_int axis_basis survives inference\n"
    else Stdio.printf "  FAIL: number_int axis_basis not found\n"
  with
  | Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg
  | exn -> Stdio.printf "  FAIL: unexpected exception: %s\n" (Exn.to_string exn)

(* ================================================================ *)
(* Direct Row construction tests (internal paths)                   *)
(* ================================================================ *)

let test_concat_consistent_bases () =
  Stdio.printf "Test 9: Concat, consistent bases -- basis preserved\n";
  Tensor.unsafe_reinitialize ();
  try
    (* Concat collapse happens in s_dim_one when a variable inside the Concat is substituted. Create
       a Concat with a Var component so that when the var is solved, s_dim_one collapses the Concat
       and we can verify the basis on the resulting Dim. *)
    let v_result = Row.get_var ~name:"result" () in
    let v_component = Row.get_var ~name:"comp" () in
    let concat_dim = Row.Concat [ Row.Var v_component; Row.get_dim ~d:3 ~basis:"x" () ] in
    let constraints =
      [
        Row.Dim_eq { d1 = Row.Var v_result; d2 = concat_dim; origin = dummy_origin };
        Row.Dim_eq
          { d1 = Row.Var v_component; d2 = Row.get_dim ~d:2 ~basis:"x" (); origin = dummy_origin };
      ]
    in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let d = Row.get_dim_val env v_result in
    let basis = get_var_basis env v_result in
    Stdio.printf "  Solved to d=%s, basis=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string)
      basis;
    if Option.equal Int.equal d (Some 5) && String.equal basis "x" then
      Stdio.printf "  PASS: concat preserved basis \"x\" with d=5\n"
    else Stdio.printf "  FAIL: expected d=5, basis=\"x\"\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_concat_conflicting_bases () =
  Stdio.printf "Test 10: Concat, conflicting bases -- raises\n";
  Tensor.unsafe_reinitialize ();
  try
    let v_result = Row.get_var ~name:"result" () in
    let v_component = Row.get_var ~name:"comp" () in
    let concat_dim = Row.Concat [ Row.Var v_component; Row.get_dim ~d:3 ~basis:"y" () ] in
    let constraints =
      [
        Row.Dim_eq { d1 = Row.Var v_result; d2 = concat_dim; origin = dummy_origin };
        Row.Dim_eq
          { d1 = Row.Var v_component; d2 = Row.get_dim ~d:2 ~basis:"x" (); origin = dummy_origin };
      ]
    in
    let _remaining, _env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    Stdio.printf "  FAIL: should have raised Shape_error\n"
  with Row.Shape_error (msg, _) ->
    if String.is_substring msg ~substring:"conflicting dimension bases" then
      Stdio.printf "  PASS: got expected Shape_error: %s\n" msg
    else Stdio.printf "  FAIL: wrong error message: %s\n" msg

let test_concat_mixed_based_unbased () =
  Stdio.printf "Test 11: Concat, mix based/unbased -- basis preserved\n";
  Tensor.unsafe_reinitialize ();
  try
    (* Same as test 9 but one component is unbased — the basis from the based component should be
       preserved in the collapsed Dim. *)
    let v_result = Row.get_var ~name:"result" () in
    let v_component = Row.get_var ~name:"comp" () in
    let concat_dim = Row.Concat [ Row.Var v_component; Row.get_dim ~d:3 () ] in
    let constraints =
      [
        Row.Dim_eq { d1 = Row.Var v_result; d2 = concat_dim; origin = dummy_origin };
        Row.Dim_eq
          { d1 = Row.Var v_component; d2 = Row.get_dim ~d:2 ~basis:"x" (); origin = dummy_origin };
      ]
    in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let d = Row.get_dim_val env v_result in
    let basis = get_var_basis env v_result in
    Stdio.printf "  Solved to d=%s, basis=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string)
      basis;
    if Option.equal Int.equal d (Some 5) && String.equal basis "x" then
      Stdio.printf "  PASS: concat preserved basis \"x\" from based component\n"
    else Stdio.printf "  FAIL: expected d=5, basis=\"x\"\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_affine_matching_bases () =
  Stdio.printf "Test 12: Affine, matching bases -- preserved\n";
  Tensor.unsafe_reinitialize ();
  try
    let v_over = Row.get_var ~name:"over" () in
    let v_result = Row.get_var ~name:"result" () in
    let affine_dim =
      Row.Affine
        {
          stride = 1;
          over = Row.Var v_over;
          conv = Some { dilation = 1; kernel = Row.get_dim ~d:3 ~basis:"x" (); use_padding = false };
          stride_offset = 0;
        }
    in
    let constraints =
      [
        Row.Dim_eq { d1 = Row.Var v_result; d2 = affine_dim; origin = dummy_origin };
        Row.Dim_eq
          { d1 = Row.Var v_over; d2 = Row.get_dim ~d:4 ~basis:"x" (); origin = dummy_origin };
      ]
    in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let d = Row.get_dim_val env v_result in
    let basis = get_var_basis env v_result in
    (* input_size = 1 * (4 - 1) + 3 = 6, basis should be "x" *)
    Stdio.printf "  Solved to d=%s, basis=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string)
      basis;
    if Option.equal Int.equal d (Some 6) && String.equal basis "x" then
      Stdio.printf "  PASS: affine preserved basis \"x\" with d=6\n"
    else Stdio.printf "  FAIL: expected d=6, basis=\"x\"\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_affine_conflicting_bases () =
  Stdio.printf "Test 13: Affine, conflicting bases -- raises\n";
  Tensor.unsafe_reinitialize ();
  try
    let v_over = Row.get_var ~name:"over" () in
    let v_result = Row.get_var ~name:"result" () in
    let affine_dim =
      Row.Affine
        {
          stride = 1;
          over = Row.Var v_over;
          conv = Some { dilation = 1; kernel = Row.get_dim ~d:3 ~basis:"y" (); use_padding = false };
          stride_offset = 0;
        }
    in
    let constraints =
      [
        Row.Dim_eq { d1 = Row.Var v_result; d2 = affine_dim; origin = dummy_origin };
        Row.Dim_eq
          { d1 = Row.Var v_over; d2 = Row.get_dim ~d:4 ~basis:"x" (); origin = dummy_origin };
      ]
    in
    let _remaining, _env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    Stdio.printf "  FAIL: should have raised Shape_error\n"
  with Row.Shape_error (msg, _) ->
    if String.is_substring msg ~substring:"conflicting dimension bases" then
      Stdio.printf "  PASS: got expected Shape_error: %s\n" msg
    else Stdio.printf "  FAIL: wrong error message: %s\n" msg

let test_stride_noconv_forward () =
  Stdio.printf "Test 14: Stride no-conv forward -- basis propagated\n";
  Tensor.unsafe_reinitialize ();
  try
    (* Affine{stride=2; over=Dim{d=4; basis="x"}} should produce d=8 with basis "x" *)
    let v = Row.get_var ~name:"result" () in
    let affine_dim =
      Row.Affine
        { stride = 2; over = Row.get_dim ~d:4 ~basis:"x" (); conv = None; stride_offset = 0 }
    in
    let constraints = [ Row.Dim_eq { d1 = Row.Var v; d2 = affine_dim; origin = dummy_origin } ] in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let d = Row.get_dim_val env v in
    let basis = get_var_basis env v in
    Stdio.printf "  Solved to d=%s, basis=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string)
      basis;
    if Option.equal Int.equal d (Some 8) && String.equal basis "x" then
      Stdio.printf "  PASS: stride forward propagated basis \"x\"\n"
    else Stdio.printf "  FAIL: expected d=8, basis=\"x\"\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_stride_noconv_reverse () =
  Stdio.printf "Test 15: Stride no-conv reverse -- basis propagated\n";
  Tensor.unsafe_reinitialize ();
  try
    (* Dim{d=8; basis="x"} unified with Affine{stride=2; over=Var v} should solve v to d=4 with
       basis "x" propagated via get_dim *)
    let v = Row.get_var ~name:"over" () in
    let affine_dim = Row.Affine { stride = 2; over = Row.Var v; conv = None; stride_offset = 0 } in
    let target = Row.get_dim ~d:8 ~basis:"x" () in
    let constraints = [ Row.Dim_eq { d1 = target; d2 = affine_dim; origin = dummy_origin } ] in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let d = Row.get_dim_val env v in
    let basis = get_var_basis env v in
    Stdio.printf "  Solved over to d=%s, basis=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string)
      basis;
    if Option.equal Int.equal d (Some 4) && String.equal basis "x" then
      Stdio.printf "  PASS: stride reverse propagated basis \"x\" to over\n"
    else Stdio.printf "  FAIL: expected d=4, basis=\"x\"\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_conv_nopadding_forward () =
  Stdio.printf "Test 16: Conv no-padding forward -- basis propagated\n";
  Tensor.unsafe_reinitialize ();
  try
    (* input_size = stride * (output_size - 1) + kernel_size = 1 * (4 - 1) + 3 = 6 *)
    let v = Row.get_var ~name:"result" () in
    let affine_dim =
      Row.Affine
        {
          stride = 1;
          over = Row.get_dim ~d:4 ~basis:"x" ();
          conv = Some { dilation = 1; kernel = Row.get_dim ~d:3 (); use_padding = false };
          stride_offset = 0;
        }
    in
    let constraints = [ Row.Dim_eq { d1 = Row.Var v; d2 = affine_dim; origin = dummy_origin } ] in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let d = Row.get_dim_val env v in
    let basis = get_var_basis env v in
    Stdio.printf "  Solved to d=%s, basis=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string)
      basis;
    if Option.equal Int.equal d (Some 6) && String.equal basis "x" then
      Stdio.printf "  PASS: conv forward propagated basis \"x\"\n"
    else Stdio.printf "  FAIL: expected d=6, basis=\"x\"\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_conv_nopadding_reverse () =
  Stdio.printf "Test 17: Conv no-padding reverse -- basis propagated\n";
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
    let target = Row.get_dim ~d:6 ~basis:"x" () in
    let constraints = [ Row.Dim_eq { d1 = target; d2 = affine_dim; origin = dummy_origin } ] in
    let _remaining, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    let d = Row.get_dim_val env v in
    let basis = get_var_basis env v in
    Stdio.printf "  Solved over to d=%s, basis=\"%s\"\n"
      (Option.value_map d ~default:"?" ~f:Int.to_string)
      basis;
    if Option.equal Int.equal d (Some 4) && String.equal basis "x" then
      Stdio.printf "  PASS: conv reverse propagated basis \"x\" to over\n"
    else Stdio.printf "  FAIL: expected d=4, basis=\"x\"\n"
  with Row.Shape_error (msg, _) -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" msg

let test_lub_conflicting_bases () =
  Stdio.printf "Test 18: LUB with conflicting bases -- strict equality raises\n";
  Tensor.unsafe_reinitialize ();
  (* The dim-level inequality solver (solve_dim_ineq) checks bases strictly at the Dim/Dim fast path
     before reaching the LUB computation. When two concrete dims with same size but different bases
     meet through mutual Row_ineq constraints, the strict check raises Shape_error. This is correct
     behavior: the LUB demotion to d=1 only applies within the Bounds_dim path, not when both dims
     are already solved. Verify that the error is raised. *)
  try
    let prov = Row.empty_provenance in
    let r1 = { Row.dims = [ Row.get_dim ~d:4 ~basis:"x" () ]; bcast = Broadcastable; prov } in
    let r2 = { Row.dims = [ Row.get_dim ~d:4 ~basis:"y" () ]; bcast = Broadcastable; prov } in
    let constraints =
      [
        Row.Row_ineq { cur = r1; subr = r2; origin = dummy_origin };
        Row.Row_ineq { cur = r2; subr = r1; origin = dummy_origin };
      ]
    in
    let _remaining, _env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    Stdio.printf "  FAIL: should have raised Shape_error for conflicting bases\n"
  with Row.Shape_error (msg, _) ->
    if String.is_substring msg ~substring:"different bases" then
      Stdio.printf "  PASS: conflicting bases in mutual inequality correctly raises: %s\n" msg
    else Stdio.printf "  FAIL: wrong error message: %s\n" msg

(* ================================================================ *)
(* Main                                                             *)
(* ================================================================ *)

let () =
  Stdio.printf "=== Dimension Basis Tests ===\n\n";
  test_same_basis_same_size ();
  Stdio.printf "\n";
  test_conflicting_bases_same_size ();
  Stdio.printf "\n";
  test_one_based_one_unbased ();
  Stdio.printf "\n";
  test_variable_mediated_unbased_then_based ();
  Stdio.printf "\n";
  test_variable_mediated_conflicting ();
  Stdio.printf "\n";
  test_variable_mediated_basis_upgrade ();
  Stdio.printf "\n";
  test_shape_to_bases ();
  Stdio.printf "\n";
  test_number_axis_basis ();
  Stdio.printf "\n";
  test_bits_axis_basis ();
  Stdio.printf "\n";
  test_range_axis_basis ();
  Stdio.printf "\n";
  test_number_int_axis_basis ();
  Stdio.printf "\n";
  test_concat_consistent_bases ();
  Stdio.printf "\n";
  test_concat_conflicting_bases ();
  Stdio.printf "\n";
  test_concat_mixed_based_unbased ();
  Stdio.printf "\n";
  test_affine_matching_bases ();
  Stdio.printf "\n";
  test_affine_conflicting_bases ();
  Stdio.printf "\n";
  test_stride_noconv_forward ();
  Stdio.printf "\n";
  test_stride_noconv_reverse ();
  Stdio.printf "\n";
  test_conv_nopadding_forward ();
  Stdio.printf "\n";
  test_conv_nopadding_reverse ();
  Stdio.printf "\n";
  test_lub_conflicting_bases ();
  Stdio.printf "\n";
  Stdio.printf "=== Done ===\n"

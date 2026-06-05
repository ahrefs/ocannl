(** Regression tests for the beg_dims-on-Row.t refactor.

    See docs/proposals/refactor-beg-dims-to-t.md for the design notes. Each test exercises one of
    the five tests called out in the proposal:

    1. Outer-left alignment in solve_row_ineq.
    2. s_row_one composes beg_dims faithfully.
    3. Closing preserves beg_dims via substitution.
    4. GLB merge symmetric on both flanks.
    5. Monotonicity via re-firing. *)

open! Base
open Ocannl

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

let prov = Row.empty_provenance
let dim ?(basis = Row.default_basis) d = Row.get_dim ~d ~basis ()

(* Test 1: outer-left alignment.
   res  = { beg_dims = [Dim 5; Dim 2]; dims = [Dim 4]; bcast = Broadcastable }
   opnd = { beg_dims = [Dim 2];        dims = [Dim 4]; bcast = Row_var rho }
   res ⊑ opnd must FAIL: outer-left aligns opnd's leading Dim 2 with res's leading Dim 5,
   which is incompatible. *)
let test_1_outer_left_alignment () =
  Stdio.printf "Test 1: outer-left leading-flank alignment rejects size mismatch\n";
  let res : Row.t =
    { beg_dims = [ dim 5; dim 2 ]; dims = [ dim 4 ]; bcast = Broadcastable; prov }
  in
  let rho = Row.get_row_var () in
  let opnd : Row.t =
    { beg_dims = [ dim 2 ]; dims = [ dim 4 ]; bcast = Row_var rho; prov }
  in
  let ineq = Row.Row_ineq { res; opnd; origin = dummy_origin } in
  try
    let _remaining, _env =
      Row.solve_inequalities ~stage:Stage1 [ ineq ] Row.empty_env
    in
    Stdio.printf "  FAIL: inequality should have raised Shape_error (5 ≠ 2)\n"
  with Row.Shape_error (msg, _) ->
    Stdio.printf "  PASS: got Shape_error: %s\n" msg

(* Test 2: substitution preserves beg_dims via uniform composition.
   in_   = { beg_dims = [Dim 3]; dims = []; bcast = Row_var rho }
   value = { beg_dims = [];      dims = [Dim 4]; bcast = Broadcastable }
   After solving Row_eq { row_of_var rho; value } and then substituting into in_, the
   result must be { beg_dims = [Dim 3]; dims = [Dim 4]; Broadcastable }. *)
let test_2_substitution_preserves_beg_dims () =
  Stdio.printf "Test 2: s_row_one composes beg_dims (closed value into open row)\n";
  let rho = Row.get_row_var () in
  let in_ : Row.t =
    { beg_dims = [ dim 3 ]; dims = []; bcast = Row_var rho; prov }
  in
  let value : Row.t =
    { beg_dims = []; dims = [ dim 4 ]; bcast = Broadcastable; prov }
  in
  let rho_row : Row.t = { beg_dims = []; dims = []; bcast = Row_var rho; prov } in
  let eq = Row.Row_eq { r1 = rho_row; r2 = value; origin = dummy_origin } in
  let _remaining, env =
    Row.solve_inequalities ~stage:Stage1 [ eq ] Row.empty_env
  in
  let result = Row.subst_row env in_ in
  let expected : Row.t =
    { beg_dims = [ dim 3 ]; dims = [ dim 4 ]; bcast = Broadcastable; prov }
  in
  if Row.equal result expected then Stdio.printf "  PASS\n"
  else
    Stdio.printf "  FAIL: expected %s got %s\n"
      (Sexp.to_string_hum (Row.sexp_of_t expected))
      (Sexp.to_string_hum (Row.sexp_of_t result))

(* Test 3: closing preserves beg_dims.
   r = { beg_dims = [Dim 7]; dims = []; bcast = Row_var rho }
   Force closure at Stage7 via a Shape_row constraint (which closes row variables in stage 7).
   After substitution, r must close to { beg_dims = [Dim 7]; dims = []; Broadcastable }.

   This test guards the silent-erasure soundness risk: under the new layout, the bare row
   variable closes to empty Broadcastable, and r's beg_dims is preserved through s_row_one's
   uniform composition at substitution time. *)
let test_3_closing_preserves_beg_dims () =
  Stdio.printf "Test 3: closing preserves beg_dims (Row_var v -> Broadcastable)\n";
  let rho = Row.get_row_var () in
  let r : Row.t =
    { beg_dims = [ dim 7 ]; dims = []; bcast = Row_var rho; prov }
  in
  let constraints = [ Row.Shape_row (r, dummy_origin) ] in
  let _remaining, env =
    Row.solve_inequalities ~stage:Stage7 constraints Row.empty_env
  in
  let result = Row.subst_row env r in
  match result with
  | { beg_dims = [ Dim { d = 7; _ } ]; dims = []; bcast = Broadcastable; _ } ->
      Stdio.printf "  PASS\n"
  | _ ->
      Stdio.printf "  FAIL: got %s\n"
        (Sexp.to_string_hum (Row.sexp_of_t result))

(* Test 4: GLB merge symmetric across both flanks.
   First inequality:  res1 = { beg_dims=[Dim 3; Dim 5]; dims=[Dim 7]; Broadcastable }
                      opnd = rho_row (open)
   Second inequality: res2 = { beg_dims=[Dim 3];        dims=[Dim 8; Dim 7]; Broadcastable }
                      opnd = rho_row (same rho)
   Merged GLB should keep the shorter leading flank length (1) and shorter trailing flank length
   (1). Outer-left Dim 3 matches outer-left Dim 3 → Dim 3; outer-right Dim 7 matches outer-right
   Dim 7 → Dim 7. *)
let test_4_glb_merge_symmetric () =
  Stdio.printf "Test 4: GLB merge symmetric on both flanks (matching outer axes)\n";
  let rho = Row.get_row_var () in
  let rho_row : Row.t = { beg_dims = []; dims = []; bcast = Row_var rho; prov } in
  let res1 : Row.t =
    { beg_dims = [ dim 3; dim 5 ]; dims = [ dim 7 ]; bcast = Broadcastable; prov }
  in
  let res2 : Row.t =
    { beg_dims = [ dim 3 ]; dims = [ dim 8; dim 7 ]; bcast = Broadcastable; prov }
  in
  let ineq1 = Row.Row_ineq { res = res1; opnd = rho_row; origin = dummy_origin } in
  let ineq2 = Row.Row_ineq { res = res2; opnd = rho_row; origin = dummy_origin } in
  let _remaining, env =
    Row.solve_inequalities ~stage:Stage1 [ ineq1; ineq2 ] Row.empty_env
  in
  let _remaining, env =
    Row.solve_inequalities ~stage:Stage6 [ Row.Shape_row (rho_row, dummy_origin) ] env
  in
  let result = Row.subst_row env rho_row in
  match result with
  | {
      beg_dims = [ Dim { d = 3; _ } ];
      dims = [ Dim { d = 7; _ } ];
      bcast = Broadcastable;
      _;
    } ->
      Stdio.printf "  PASS\n"
  | _ ->
      Stdio.printf "  FAIL: expected {beg_dims=[Dim 3]; dims=[Dim 7]; Broadcastable}, got %s\n"
        (Sexp.to_string_hum (Row.sexp_of_t result))

(* Test 4b: leading-flank CONFLICT demotes to unbased Dim 1.
   Two upper bounds on the same rho with different leading-flank values (Dim 3 vs Dim 5) at the
   same outer-left position. The merge must demote to an unbased Dim 1. Trailing flank stays
   compatible (Dim 7 in both). A regression that removed the leading-flank conflict case from
   meet_dim would leave one of the original axes (Dim 3 or Dim 5) in beg_dims and this assertion
   would fail. *)
let test_4b_glb_leading_conflict_demotes_to_one () =
  Stdio.printf "Test 4b: leading-flank conflict demotes to unbased Dim 1\n";
  let rho = Row.get_row_var () in
  let rho_row : Row.t = { beg_dims = []; dims = []; bcast = Row_var rho; prov } in
  let res1 : Row.t =
    { beg_dims = [ dim 3 ]; dims = [ dim 7 ]; bcast = Broadcastable; prov }
  in
  let res2 : Row.t =
    { beg_dims = [ dim 5 ]; dims = [ dim 7 ]; bcast = Broadcastable; prov }
  in
  let ineq1 = Row.Row_ineq { res = res1; opnd = rho_row; origin = dummy_origin } in
  let ineq2 = Row.Row_ineq { res = res2; opnd = rho_row; origin = dummy_origin } in
  let _remaining, env =
    Row.solve_inequalities ~stage:Stage1 [ ineq1; ineq2 ] Row.empty_env
  in
  let _remaining, env =
    Row.solve_inequalities ~stage:Stage6 [ Row.Shape_row (rho_row, dummy_origin) ] env
  in
  let result = Row.subst_row env rho_row in
  match result with
  | {
      beg_dims = [ Dim { d = 1; basis; _ } ];
      dims = [ Dim { d = 7; _ } ];
      bcast = Broadcastable;
      _;
    }
    when String.equal basis Row.bcast_if_1 ->
      Stdio.printf "  PASS\n"
  | _ ->
      Stdio.printf
        "  FAIL: expected leading flank demoted to broadcast-top Dim 1; got %s\n"
        (Sexp.to_string_hum (Row.sexp_of_t result))

(* Test 4c: trailing-flank CONFLICT demotes to unbased Dim 1.
   Mirror of Test 4b: same leading flank (Dim 3), different trailing-flank values (Dim 7 vs
   Dim 11). The merge must demote the trailing flank to an unbased Dim 1. A regression that
   removed the trailing-flank conflict case from meet_dim would leave one of the originals
   (Dim 7 or Dim 11) and this assertion would fail. *)
let test_4c_glb_trailing_conflict_demotes_to_one () =
  Stdio.printf "Test 4c: trailing-flank conflict demotes to unbased Dim 1\n";
  let rho = Row.get_row_var () in
  let rho_row : Row.t = { beg_dims = []; dims = []; bcast = Row_var rho; prov } in
  let res1 : Row.t =
    { beg_dims = [ dim 3 ]; dims = [ dim 7 ]; bcast = Broadcastable; prov }
  in
  let res2 : Row.t =
    { beg_dims = [ dim 3 ]; dims = [ dim 11 ]; bcast = Broadcastable; prov }
  in
  let ineq1 = Row.Row_ineq { res = res1; opnd = rho_row; origin = dummy_origin } in
  let ineq2 = Row.Row_ineq { res = res2; opnd = rho_row; origin = dummy_origin } in
  let _remaining, env =
    Row.solve_inequalities ~stage:Stage1 [ ineq1; ineq2 ] Row.empty_env
  in
  let _remaining, env =
    Row.solve_inequalities ~stage:Stage6 [ Row.Shape_row (rho_row, dummy_origin) ] env
  in
  let result = Row.subst_row env rho_row in
  match result with
  | {
      beg_dims = [ Dim { d = 3; _ } ];
      dims = [ Dim { d = 1; basis; _ } ];
      bcast = Broadcastable;
      _;
    }
    when String.equal basis Row.bcast_if_1 ->
      Stdio.printf "  PASS\n"
  | _ ->
      Stdio.printf
        "  FAIL: expected trailing flank demoted to broadcast-top Dim 1; got %s\n"
        (Sexp.to_string_hum (Row.sexp_of_t result))

(* Test 5: monotonicity via re-firing (success case).
   First: solve res ⊑ b with b = rho_row, c = {beg_dims=[Dim 4]; dims=[Dim 7]; Broadcastable}.
   The GLB c is banked for rho.
   Then: substitute rho := {beg_dims=[Dim 4]; dims=[]; bcast=Row_var rho'}. The leading Dim 4 of
   the substituted row aligns outer-left against the banked Dim 4 and the inequality holds. *)
let test_5_monotonicity_via_refiring () =
  Stdio.printf "Test 5: monotonicity via re-firing — compatible substitution succeeds\n";
  let rho = Row.get_row_var () in
  let rho' = Row.get_row_var () in
  let b : Row.t = { beg_dims = []; dims = []; bcast = Row_var rho; prov } in
  let c : Row.t =
    { beg_dims = [ dim 4 ]; dims = [ dim 7 ]; bcast = Broadcastable; prov }
  in
  let ineq = Row.Row_ineq { res = c; opnd = b; origin = dummy_origin } in
  let _remaining, env =
    Row.solve_inequalities ~stage:Stage1 [ ineq ] Row.empty_env
  in
  let subst : Row.t =
    { beg_dims = [ dim 4 ]; dims = []; bcast = Row_var rho'; prov }
  in
  let rho_row : Row.t = { beg_dims = []; dims = []; bcast = Row_var rho; prov } in
  let eq = Row.Row_eq { r1 = rho_row; r2 = subst; origin = dummy_origin } in
  try
    let _remaining, _env = Row.solve_inequalities ~stage:Stage1 [ eq ] env in
    Stdio.printf "  PASS\n"
  with Row.Shape_error (msg, _) ->
    Stdio.printf "  FAIL: substitution should not retract; got Shape_error: %s\n" msg

(* Test 5b: monotonicity via re-firing (negative mutation).
   Same setup as Test 5 but the substituted leading axis CONFLICTS with the banked GLB's leading
   axis (banked Dim 4 vs substituted Dim 5). After substitution, the banked GLB must still be
   enforced — re-firing via the substituted row should expose the conflict either at substitution
   time or when the row variable is closed against its GLB. We attempt to close rho' (the new
   row variable) and check that subst_row of rho's row reflects either:
   (a) an explicit Shape_error during solving (the banked fact is retained and rejects the bad
       substitution), or
   (b) a substituted result that has not silently lost the GLB constraint — i.e., the closed row
       must contain the GLB's Dim 4 somewhere; a regression that dropped the banked fact would
       produce a clean Dim 5 with no trace of Dim 4.

   This guards against the "ignored/dropped banked GLB" regression flagged by the reviewer. *)
let test_5b_monotonicity_negative_mutation () =
  Stdio.printf "Test 5b: monotonicity — incompatible substitution preserves banked facts\n";
  let rho = Row.get_row_var () in
  let rho' = Row.get_row_var () in
  let b : Row.t = { beg_dims = []; dims = []; bcast = Row_var rho; prov } in
  let c : Row.t =
    { beg_dims = [ dim 4 ]; dims = [ dim 7 ]; bcast = Broadcastable; prov }
  in
  let ineq = Row.Row_ineq { res = c; opnd = b; origin = dummy_origin } in
  let _remaining, env =
    Row.solve_inequalities ~stage:Stage1 [ ineq ] Row.empty_env
  in
  (* Mutation: substitute rho with a row whose leading Dim 5 CONFLICTS with the banked GLB's
     leading Dim 4. *)
  let subst : Row.t =
    { beg_dims = [ dim 5 ]; dims = []; bcast = Row_var rho'; prov }
  in
  let rho_row : Row.t = { beg_dims = []; dims = []; bcast = Row_var rho; prov } in
  let eq = Row.Row_eq { r1 = rho_row; r2 = subst; origin = dummy_origin } in
  let raised_during_subst, env_after_subst =
    try
      let _remaining, env =
        Row.solve_inequalities ~stage:Stage1 [ eq ] env
      in
      (false, env)
    with Row.Shape_error _ -> (true, env)
  in
  if raised_during_subst then
    Stdio.printf "  PASS: banked fact rejected the conflict at substitution time\n"
  else
    (* Force closure and check that the result reflects the banked GLB (must contain Dim 4) or
       raises during close. *)
    let closed =
      try
        let _remaining, env_closed =
          Row.solve_inequalities ~stage:Stage6
            [ Row.Shape_row (rho_row, dummy_origin) ]
            env_after_subst
        in
        Some (Row.subst_row env_closed rho_row)
      with Row.Shape_error _ -> None
    in
    match closed with
    | None ->
        Stdio.printf "  PASS: banked fact rejected the conflict at close time\n"
    | Some result ->
        let mentions_four_or_one_demotion =
          let dims_have d =
            List.exists (result.beg_dims @ result.dims) ~f:(function
              | Row.Dim { d = d'; _ } -> d' = d
              | _ -> false)
          in
          dims_have 4 || dims_have 1
        in
        if mentions_four_or_one_demotion then
          Stdio.printf
            "  PASS: closed row preserved banked Dim 4 (or demoted to Dim 1): %s\n"
            (Sexp.to_string_hum (Row.sexp_of_t result))
        else
          Stdio.printf
            "  FAIL: closed row dropped banked fact (no Dim 4 or Dim 1 demotion present): %s\n"
            (Sexp.to_string_hum (Row.sexp_of_t result))

let () =
  Tensor.unsafe_reinitialize ();
  test_1_outer_left_alignment ();
  test_2_substitution_preserves_beg_dims ();
  test_3_closing_preserves_beg_dims ();
  test_4_glb_merge_symmetric ();
  test_4b_glb_leading_conflict_demotes_to_one ();
  test_4c_glb_trailing_conflict_demotes_to_one ();
  test_5b_monotonicity_negative_mutation ();
  test_5_monotonicity_via_refiring ()

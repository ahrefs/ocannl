(** Regression tests for the beg_dims-on-Row.t refactor.

    See docs/proposals/refactor-beg-dims-to-t.md for the design notes. Each test exercises one of
    the five tests called out in the proposal:

    1. Outer-left alignment in solve_row_ineq.
    2. s_row_one composes beg_dims faithfully.
    3. Closing preserves beg_dims via substitution.
    4. LUB merge symmetric on both flanks.
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
let dim ?basis d = Row.get_dim ~d ?basis ()

(* Test 1: outer-left alignment.
   cur  = { beg_dims = [Dim 5; Dim 2]; dims = [Dim 4]; bcast = Broadcastable }
   subr = { beg_dims = [Dim 2];        dims = [Dim 4]; bcast = Row_var rho }
   cur >= subr must FAIL: outer-left aligns subr's leading Dim 2 with cur's leading Dim 5,
   which is incompatible. *)
let test_1_outer_left_alignment () =
  Stdio.printf "Test 1: outer-left leading-flank alignment rejects size mismatch\n";
  let cur : Row.t =
    { beg_dims = [ dim 5; dim 2 ]; dims = [ dim 4 ]; bcast = Broadcastable; prov }
  in
  let rho = Row.get_row_var () in
  let subr : Row.t =
    { beg_dims = [ dim 2 ]; dims = [ dim 4 ]; bcast = Row_var rho; prov }
  in
  let ineq = Row.Row_ineq { cur; subr; origin = dummy_origin } in
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

(* Test 4: LUB merge symmetric across both flanks.
   First inequality:  cur1 = { beg_dims=[Dim 3; Dim 5]; dims=[Dim 7]; Broadcastable }
                      subr = rho_row (open)
   Second inequality: cur2 = { beg_dims=[Dim 3];        dims=[Dim 8; Dim 7]; Broadcastable }
                      subr = rho_row (same rho)
   Merged LUB should keep the shorter leading flank length (1) and shorter trailing flank length
   (1). Conflicts on either side demote to broadcast-1; here Dim 5 (only on side 1) gets
   trimmed off entirely (shorter wins), and Dim 8 likewise. *)
let test_4_lub_merge_symmetric () =
  Stdio.printf "Test 4: LUB merge symmetric on both flanks\n";
  let rho = Row.get_row_var () in
  let rho_row : Row.t = { beg_dims = []; dims = []; bcast = Row_var rho; prov } in
  let cur1 : Row.t =
    { beg_dims = [ dim 3; dim 5 ]; dims = [ dim 7 ]; bcast = Broadcastable; prov }
  in
  let cur2 : Row.t =
    { beg_dims = [ dim 3 ]; dims = [ dim 8; dim 7 ]; bcast = Broadcastable; prov }
  in
  let ineq1 = Row.Row_ineq { cur = cur1; subr = rho_row; origin = dummy_origin } in
  let ineq2 = Row.Row_ineq { cur = cur2; subr = rho_row; origin = dummy_origin } in
  let _remaining, env =
    Row.solve_inequalities ~stage:Stage1 [ ineq1; ineq2 ] Row.empty_env
  in
  (* Check the banked LUB for rho. *)
  match Row.get_row_from_env env rho with
  | Some lub ->
      Stdio.printf "  Banked LUB for rho: %s\n"
        (Sexp.to_string_hum (Row.sexp_of_t lub))
  | None -> (
      let entry = Row.unsolved_constraints env in
      Stdio.printf "  No solved row for rho; remaining constraints: %d\n"
        (List.length entry));
  Stdio.printf "  PASS (informational)\n"

(* Test 5: monotonicity via re-firing.
   First: solve b <= c with b = rho_row (open), c = {beg_dims=[Dim 4]; dims=[Dim 7]; Broadcastable}.
   The LUB c is banked for rho.
   Then: substitute rho := {beg_dims=[Dim 4]; dims=[]; bcast=Row_var rho'}. The leading Dim 4 of
   the substituted row aligns outer-left against the leading Dim 4 of the LUB and the inequality
   still holds. No retraction of banked facts. *)
let test_5_monotonicity_via_refiring () =
  Stdio.printf "Test 5: monotonicity via re-firing (substitute into banked LUB)\n";
  let rho = Row.get_row_var () in
  let rho' = Row.get_row_var () in
  let b : Row.t = { beg_dims = []; dims = []; bcast = Row_var rho; prov } in
  let c : Row.t =
    { beg_dims = [ dim 4 ]; dims = [ dim 7 ]; bcast = Broadcastable; prov }
  in
  let ineq = Row.Row_ineq { cur = c; subr = b; origin = dummy_origin } in
  let _remaining, env =
    Row.solve_inequalities ~stage:Stage1 [ ineq ] Row.empty_env
  in
  (* Now substitute rho := { beg_dims = [Dim 4]; dims = []; bcast = Row_var rho' } *)
  let subst : Row.t =
    { beg_dims = [ dim 4 ]; dims = []; bcast = Row_var rho'; prov }
  in
  let rho_row : Row.t = { beg_dims = []; dims = []; bcast = Row_var rho; prov } in
  let eq = Row.Row_eq { r1 = rho_row; r2 = subst; origin = dummy_origin } in
  try
    let _remaining, _env = Row.solve_inequalities ~stage:Stage1 [ eq ] env in
    Stdio.printf "  PASS: substitution succeeded (no retraction)\n"
  with Row.Shape_error (msg, _) ->
    Stdio.printf "  FAIL: substitution should not retract; got Shape_error: %s\n"
      msg

let () =
  Tensor.unsafe_reinitialize ();
  test_1_outer_left_alignment ();
  test_2_substitution_preserves_beg_dims ();
  test_3_closing_preserves_beg_dims ();
  test_4_lub_merge_symmetric ();
  test_5_monotonicity_via_refiring ()

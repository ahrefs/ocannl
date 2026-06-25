(** Regression test for rotational (shifted-flank) row constraints — same row variable on both
    sides, and the closed-closed base case (the gh-ocannl-247 family).

    With equal total flank lengths but shifted splits, the residue rotates through the shared
    variable's value (the word equation x ++ t = s ++ x for equality; pointwise chains through x for
    inequality). These cases used to be silently dropped: the unsatisfiable [3].<v> = <v>.[5] and
    [3].<v> <= <v>.[5] passed without a single dimension check, and so did the closed-closed
    [3].<closed> <= <closed>.[5] at equal ranks (where broadcast inserts no padding, so the
    operand's explicit dim must pin the result's position).

    The resolution is deferral into the closing policy: the constraint stays in flight; if the
    variable is solved by other constraints the substituted closed-closed check is exact; otherwise
    stage 6/7 closes the variable upward — the least-material disjunct — and the constraint reduces
    to comparing the surplus words directly (equal for =; pointwise <= for inequality, where an
    operand-side claim-free dim can still absorb). Closed-closed inequalities compare the operand's
    explicit material against the result's flat axis list from the outer edges; only
    broadcast-inserted middle positions are unconstrained.

    Each check drives the constraint through Stage1 (where rotational cases defer) and then Stage7
    (where surviving row variables are closed upward and the residue is checked). *)

open! Base
open Ocannl

let origin : Row.constraint_origin list =
  [
    {
      lhs_name = "test";
      lhs_kind = `Output;
      rhs_name = "test";
      rhs_kind = `Output;
      operation = None;
    };
  ]

let dim d = Row.get_default_dim ~d ()

let closed_row ~sh_id ~beg_dims ~dims =
  { Row.beg_dims; dims; bcast = Broadcastable; prov = Row.provenance ~sh_id ~kind:`Output }

let check name constr =
  match
    let leftover, env = Row.solve_inequalities ~stage:Stage1 [ constr ] Row.empty_env in
    Row.solve_inequalities ~stage:Stage7 leftover env
  with
  | _ -> Stdio.printf "%s: no error\n" name
  | exception Row.Shape_error (msg, _) -> Stdio.printf "%s: Shape_error: %s\n" name msg

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:1 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:2 ~kind:`Output) rho in
  check "eq, shifted flanks, conflicting dims (unsatisfiable)"
    (Row.Row_eq { r1 = { r1 with beg_dims = [ dim 3 ] }; r2 = { r2 with dims = [ dim 5 ] }; origin })

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:3 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:4 ~kind:`Output) rho in
  (* Satisfiable; resolved least-materially at closing (v |-> [], surplus words equal). *)
  check "eq, shifted flanks, conjugate dims (least-material resolution)"
    (Row.Row_eq { r1 = { r1 with beg_dims = [ dim 3 ] }; r2 = { r2 with dims = [ dim 3 ] }; origin })

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:5 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:6 ~kind:`Output) rho in
  check "eq, aligned flanks, conflicting dims"
    (Row.Row_eq { r1 = { r1 with dims = [ dim 3 ] }; r2 = { r2 with dims = [ dim 5 ] }; origin })

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:7 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:8 ~kind:`Output) rho in
  check "eq, aligned flanks, matching dims (legitimate no-op)"
    (Row.Row_eq { r1 = { r1 with dims = [ dim 3 ] }; r2 = { r2 with dims = [ dim 3 ] }; origin })

let () =
  check "ineq, closed-closed shifted, conflicting dims (unsatisfiable)"
    (Row.Row_ineq
       {
         res = closed_row ~sh_id:9 ~beg_dims:[ dim 3 ] ~dims:[];
         opnd = closed_row ~sh_id:10 ~beg_dims:[] ~dims:[ dim 5 ];
         origin;
       })

let () =
  check "ineq, closed-closed shifted, matching dims (satisfiable)"
    (Row.Row_ineq
       {
         res = closed_row ~sh_id:11 ~beg_dims:[ dim 3 ] ~dims:[];
         opnd = closed_row ~sh_id:12 ~beg_dims:[] ~dims:[ dim 3 ];
         origin;
       })

let () =
  (* The result's leading axis faces broadcast-inserted padding (rank 2 vs 1): unconstrained. *)
  check "ineq, closed-closed, result surplus faces inserted padding (satisfiable)"
    (Row.Row_ineq
       {
         res = closed_row ~sh_id:13 ~beg_dims:[ dim 3 ] ~dims:[ dim 5 ];
         opnd = closed_row ~sh_id:14 ~beg_dims:[] ~dims:[ dim 5 ];
         origin;
       })

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:15 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:16 ~kind:`Output) rho in
  check "ineq, same-var shifted, conflicting dims (unsatisfiable)"
    (Row.Row_ineq
       { res = { r1 with beg_dims = [ dim 3 ] }; opnd = { r2 with dims = [ dim 5 ] }; origin })

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:17 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:18 ~kind:`Output) rho in
  check "ineq, same-var shifted, matching dims (satisfiable)"
    (Row.Row_ineq
       { res = { r1 with beg_dims = [ dim 3 ] }; opnd = { r2 with dims = [ dim 3 ] }; origin })

(* Two-variable cross-surplus equalities (the split-surplus case of the formal core's Def. 4.3(c)):
   one side's leading flank and the other side's trailing flank are both in surplus. The flat
   residue is the two-variable word equation s.x1 = x2.t, whose solutions are the principal family
   (x2 = s.w, x1 = w.t) PLUS sporadic cross-overlap solutions (x2 a proper prefix of s). Binding
   with a fresh variable would commit the family only — exact under the former marked semantics but
   losing the sporadic solutions under the adopted flat-equivalence semantics. The implementation
   instead DEFERS (unify_row's beg_handled=false branch re-emits the residual equation): exact once
   a variable is pinned by other constraints, and the upward close at stage 6 picks the
   least-material disjunct, which here IS the sporadic solution.

   We pin the orientation [5].<rho1> ~ <rho2>.[5,3] (family: x2 = [5].w, x1 = w.[5,3]; sporadic: x2
   = [], x1 = [3]). All three cases succeed: the deferral finds the sporadic solution whether rho2 ~
   [] arrives before or after the cross equality. The mirror orientation <rho1>.[5] ~ [3,5].<rho2>
   pins the OTHER conservative deviation: once rho2 is pinned empty, the closed side's material sits
   entirely in beg_dims, and the asymmetric trailing guard (open trailing flank longer than the
   closed side's structural trailing flank) rejects a flat-satisfiable store — a placement-sensitive
   policy rejection. *)

let check_list name constrs =
  match
    let leftover, env = Row.solve_inequalities ~stage:Stage1 constrs Row.empty_env in
    Row.solve_inequalities ~stage:Stage7 leftover env
  with
  | _ -> Stdio.printf "%s: no error\n" name
  | exception Row.Shape_error (msg, _) -> Stdio.printf "%s: Shape_error: %s\n" name msg

let cross_surplus_pair ~sh_id rho1 rho2 =
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id ~kind:`Output) rho1 in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:(sh_id + 1) ~kind:`Output) rho2 in
  Row.Row_eq
    { r1 = { r1 with beg_dims = [ dim 5 ] }; r2 = { r2 with dims = [ dim 5; dim 3 ] }; origin }

let pin_empty ~sh_id rho =
  let r = Row.get_row_for_var (Row.provenance ~sh_id ~kind:`Output) rho in
  Row.Row_eq { r1 = r; r2 = closed_row ~sh_id:(sh_id + 1) ~beg_dims:[] ~dims:[]; origin }

let () =
  let rho1 = Row.get_row_var () and rho2 = Row.get_row_var () in
  check_list "eq, cross surpluses alone (satisfiable, family binding)"
    [ cross_surplus_pair ~sh_id:19 rho1 rho2 ]

let () =
  let rho1 = Row.get_row_var () and rho2 = Row.get_row_var () in
  check_list "eq, cross surpluses then rho2 ~ [] (satisfiable only sporadically)"
    [ cross_surplus_pair ~sh_id:21 rho1 rho2; pin_empty ~sh_id:23 rho2 ]

let () =
  let rho1 = Row.get_row_var () and rho2 = Row.get_row_var () in
  check_list "eq, rho2 ~ [] then cross surpluses (same store, pinned first)"
    [ pin_empty ~sh_id:25 rho2; cross_surplus_pair ~sh_id:27 rho1 rho2 ]

let mirror_cross_pair ~sh_id rho1 rho2 =
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id ~kind:`Output) rho1 in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:(sh_id + 1) ~kind:`Output) rho2 in
  Row.Row_eq
    { r1 = { r1 with dims = [ dim 5 ] }; r2 = { r2 with beg_dims = [ dim 3; dim 5 ] }; origin }

let () =
  let rho1 = Row.get_row_var () and rho2 = Row.get_row_var () in
  check_list "eq, mirror cross surpluses then rho2 ~ [] (trailing guard rejects)"
    [ mirror_cross_pair ~sh_id:29 rho1 rho2; pin_empty ~sh_id:31 rho2 ]

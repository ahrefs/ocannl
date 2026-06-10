(** Regression test for rotational (shifted-flank) row constraints — same row variable on both
    sides, and the closed-closed base case (the gh-ocannl-247 family).

    With equal total flank lengths but shifted splits, the residue rotates through the shared
    variable's value (the word equation x ++ t = s ++ x for equality; pointwise chains through x
    for inequality). These cases used to be silently dropped: the unsatisfiable
    [3].<v> = <v>.[5] and [3].<v> <= <v>.[5] passed without a single dimension check, and so did
    the closed-closed [3].<closed> <= <closed>.[5] at equal ranks (where broadcast inserts no
    padding, so the operand's explicit dim must pin the result's position).

    The resolution is deferral into the closing policy: the constraint stays in flight; if the
    variable is solved by other constraints the substituted closed-closed check is exact;
    otherwise stage 6/7 closes the variable upward — the least-material disjunct — and the
    constraint reduces to comparing the surplus words directly (equal for =; pointwise <= for
    inequality, where an operand-side claim-free dim can still absorb). Closed-closed
    inequalities compare the operand's explicit material against the result's flat axis list
    from the outer edges; only broadcast-inserted middle positions are unconstrained.

    Each check drives the constraint through Stage1 (where rotational cases defer) and then
    Stage7 (where surviving row variables are closed upward and the residue is checked). *)

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

(** Regression test for self-referential row equations (same row variable on both sides of
    [Row.unify_row]), the gh-ocannl-247 family.

    The shifted-flanks (rotational) cases used to be accepted silently: with equal total flank
    lengths the same-variable branch only unified the [min]-overlaps of each flank, which are
    empty when all surplus is leading on one side and trailing on the other — so the
    unsatisfiable [3].<v> = <v>.[5] passed without a single dimension check. The equation's
    residue is the word equation x ++ t = s ++ x (x = v's value, s/t the surplus flank words):
    unsatisfiable under marker-sensitive equality, and satisfiable under content equality only
    for conjugate s/t with cyclically periodic x — outside the solver's language, so the solver
    now rejects shifted splits conservatively (even the conjugate-satisfiable [3].<v> = <v>.[3]).
    Aligned splits are unaffected: overlaps are unified in full. *)

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

let check name (r1, r2) =
  match Row.unify_row ~stage:Stage1 origin (r1, r2) Row.empty_env with
  | _ -> Stdio.printf "%s: no error\n" name
  | exception Row.Shape_error (msg, _) -> Stdio.printf "%s: Shape_error: %s\n" name msg

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:1 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:2 ~kind:`Output) rho in
  check "shifted flanks, conflicting dims (unsatisfiable)"
    ({ r1 with beg_dims = [ dim 3 ] }, { r2 with dims = [ dim 5 ] })

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:3 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:4 ~kind:`Output) rho in
  (* Satisfiable under content equality (v cyclically all-3s), rejected conservatively. *)
  check "shifted flanks, conjugate dims (conservative rejection)"
    ({ r1 with beg_dims = [ dim 3 ] }, { r2 with dims = [ dim 3 ] })

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:5 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:6 ~kind:`Output) rho in
  check "aligned flanks, conflicting dims"
    ({ r1 with dims = [ dim 3 ] }, { r2 with dims = [ dim 5 ] })

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:7 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:8 ~kind:`Output) rho in
  check "aligned flanks, matching dims (legitimate no-op)"
    ({ r1 with dims = [ dim 3 ] }, { r2 with dims = [ dim 3 ] })

(** Solver-level regression tests for the Concat dim-solver hardening (task-887c4062).

    These drive [tensor/row.ml] directly because the targeted arms have no determinate high-level
    [%op] fixture that demonstrably routes through them (see
    [docs/proposals/concat-dim-solver-hardening.md], the AC1 reachability note): a broadcast
    inequality between two [Concat] bounds only reaches [solve_dim_ineq]'s GLB merge when an operand
    dimension variable already carries a [Concat] greatest-lower-bound, and a variables-only
    [Concat = Concat] of unequal arity only reaches [unify_dim]'s generalized pairing arm. Building
    the constraints by hand and calling [solve_inequalities] pins the exact arm.

    Before this task both AC1 GLB arms ([Affine]/[Concat]) raised [assert false]; the AC3 pairing arm
    fired only at equal arity. The tests below would fault (Assert_failure) or fail to link on the
    pre-task code. *)

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

let dim ?(basis = Row.default_basis) d = Row.get_dim ~d ~basis ()
let no_from = Sexp.List []

let ineq res v : Row.constraint_ =
  Row.Dim_ineq { res; opnd = Row.Var v; from_ = no_from; origin = dummy_origin }

(* Run a solve, classifying the outcome so an [assert false] regression (any non-[Shape_error]
   exception) is distinguished from a legitimate [Shape_error] and from success. *)
let run_solve ~stage ineqs =
  try `Ok (Row.solve_inequalities ~stage ineqs Row.empty_env) with
  | Row.Shape_error (m, _) -> `Shape m
  | e -> `Other (Exn.to_string e)

(* AC1 — GLB merge of two COMPATIBLE Concat bounds (stage 4) commits without crashing.
   First inequality banks [Concat 2^3] (size 5) as the glb of a fresh operand var; the second feeds
   [Concat 1^4] (also size 5) into the merge. unify_dim of the two bounds succeeds, so at stage >= 4
   the merge commits. Pre-task: this arm was [assert false]. *)
let test_ac1_glb_merge_compatible () =
  Stdio.printf "AC1: GLB merge of two compatible Concat bounds (stage 4) commits, no crash\n";
  let v = Row.get_var () in
  match
    run_solve ~stage:Stage4
      [ ineq (Row.Concat [ dim 2; dim 3 ]) v; ineq (Row.Concat [ dim 1; dim 4 ]) v ]
  with
  | `Ok _ -> Stdio.printf "  PASS: merge completed without raising\n"
  | `Shape m -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" m
  | `Other e -> Stdio.printf "  FAIL: regression (expected no assert false): %s\n" e

(* AC1 — GLB merge of two INCOMPATIBLE Concat bounds (stage 4) demotes to broadcast-top, swallowing
   the failed equality rather than propagating a Shape_error. [Concat 2^3] (size 5) vs [Concat 2^4]
   (size 6): unify_dim fails, so the merge must demote (broadcast), not raise. Pre-task: [assert
   false]. *)
let test_ac1_glb_merge_incompatible_demotes () =
  Stdio.printf "AC1: GLB merge of two incompatible Concat bounds (stage 4) demotes, no raise\n";
  let v = Row.get_var () in
  match
    run_solve ~stage:Stage4
      [ ineq (Row.Concat [ dim 2; dim 3 ]) v; ineq (Row.Concat [ dim 2; dim 4 ]) v ]
  with
  | `Ok _ -> Stdio.printf "  PASS: incompatible bounds demoted to broadcast-top (no raise)\n"
  | `Shape m -> Stdio.printf "  FAIL: should demote, not raise Shape_error: %s\n" m
  | `Other e -> Stdio.printf "  FAIL: regression (expected no assert false): %s\n" e

(* AC1 — below stage 4 the merge POSTPONES: it neither crashes nor demotes, re-deferring the
   inequality so the two bounds can still resolve equal later. We check a [Dim_ineq] survives in the
   returned residual at Stage1. Pre-task: [assert false]. *)
let test_ac1_glb_merge_postpone_below_stage4 () =
  Stdio.printf "AC1: GLB merge below stage 4 postpones (defers, no demote/crash)\n";
  let v = Row.get_var () in
  match
    run_solve ~stage:Stage1
      [ ineq (Row.Concat [ dim 2; dim 3 ]) v; ineq (Row.Concat [ dim 1; dim 4 ]) v ]
  with
  | `Ok (remaining, _) ->
      if List.exists remaining ~f:(function Row.Dim_ineq _ -> true | _ -> false) then
        Stdio.printf "  PASS: inequality deferred for a later stage\n"
      else Stdio.printf "  FAIL: expected a deferred Dim_ineq, got none\n"
  | `Shape m -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" m
  | `Other e -> Stdio.printf "  FAIL: regression (expected no assert false): %s\n" e

(* AC3 — variables-only [Concat = Concat] at UNEQUAL arity (3 components vs 2). The generalized
   pairing arm fires at stage >= 4 (the old guard required equal arity), equates the oldest variable
   of each side (a=d, then b=e) and re-runs under the new env, leaving c = 0 (since a+b+c = d+e with
   a=d, b=e). We verify the arithmetic (c=0) and the LINK: forcing a := 7 must surface as d = 7, and
   b := 9 as e = 9 — only possible if the oldest variables were actually equated. *)
let test_ac3_unequal_arity_all_var () =
  Stdio.printf "AC3: variables-only Concat=Concat at unequal arity (3 vs 2) links oldest vars\n";
  let a = Row.get_var () and b = Row.get_var () and c = Row.get_var () in
  let d = Row.get_var () and e = Row.get_var () in
  let eq d1 d2 : Row.constraint_ = Row.Dim_eq { d1; d2; origin = dummy_origin } in
  let lhs = Row.Concat [ Row.Var a; Row.Var b; Row.Var c ] in
  let rhs = Row.Concat [ Row.Var d; Row.Var e ] in
  match
    try
      let _rem, env = Row.solve_inequalities ~stage:Stage4 [ eq lhs rhs ] Row.empty_env in
      let c_val = Row.get_dim_val env c in
      let _rem, env = Row.solve_inequalities ~stage:Stage4 [ eq (Row.Var a) (dim 7) ] env in
      let d_val = Row.get_dim_val env d in
      let _rem, env = Row.solve_inequalities ~stage:Stage4 [ eq (Row.Var b) (dim 9) ] env in
      let e_val = Row.get_dim_val env e in
      `Ok (c_val, d_val, e_val)
    with
    | Row.Shape_error (m, _) -> `Shape m
    | e -> `Other (Exn.to_string e)
  with
  | `Ok (Some 0, Some 7, Some 9) ->
      Stdio.printf "  PASS: c=0, a=d (d read 7 after a:=7), b=e (e read 9 after b:=9)\n"
  | `Ok (c_val, d_val, e_val) ->
      let show = function Some n -> Int.to_string n | None -> "?" in
      Stdio.printf "  FAIL: c=%s d=%s e=%s (expected 0, 7, 9)\n" (show c_val) (show d_val)
        (show e_val)
  | `Shape m -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" m
  | `Other e -> Stdio.printf "  FAIL: regression (expected resolution, not exn): %s\n" e

(* AC3 — the pairing arm fires even when the OTHER side carries a solved [Dim] (the guard is "one
   side all-Var", not "overlap all-Var", and must take precedence over the arithmetic-cancellation
   arm). [Concat a^b] = [Concat (Dim 4)^e] with a,b,e variables: the all-Var left side fires the arm,
   the oldest residual variables a and e are equated, and the leftover binds b to the solved 4
   (a+b = 4+e with a=e ⟹ b=4). We verify b = 4. *)
let test_ac3_one_side_all_var_other_has_solved () =
  Stdio.printf "AC3: one side all-Var, other carries a solved Dim — pairing arm still fires\n";
  let a = Row.get_var () and b = Row.get_var () and e = Row.get_var () in
  let eq d1 d2 : Row.constraint_ = Row.Dim_eq { d1; d2; origin = dummy_origin } in
  let lhs = Row.Concat [ Row.Var a; Row.Var b ] in
  let rhs = Row.Concat [ dim 4; Row.Var e ] in
  match
    try
      let _rem, env = Row.solve_inequalities ~stage:Stage4 [ eq lhs rhs ] Row.empty_env in
      `Ok (Row.get_dim_val env b)
    with
    | Row.Shape_error (m, _) -> `Shape m
    | ex -> `Other (Exn.to_string ex)
  with
  | `Ok (Some 4) -> Stdio.printf "  PASS: leftover bound b = 4 (solved Dim crossed into all-Var side)\n"
  | `Ok other ->
      Stdio.printf "  FAIL: expected b = 4, got %s\n"
        (match other with Some n -> Int.to_string n | None -> "?")
  | `Shape m -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" m
  | `Other e -> Stdio.printf "  FAIL: regression (expected no assert false): %s\n" e

let () =
  Stdio.printf "=== Concat dim-solver hardening (solver-level) ===\n";
  Tensor.unsafe_reinitialize ();
  test_ac1_glb_merge_compatible ();
  test_ac1_glb_merge_incompatible_demotes ();
  test_ac1_glb_merge_postpone_below_stage4 ();
  test_ac3_unequal_arity_all_var ();
  test_ac3_one_side_all_var_other_has_solved ();
  Stdio.printf "=== Done ===\n"

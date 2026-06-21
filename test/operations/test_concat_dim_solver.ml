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

let eq d1 d2 : Row.constraint_ = Row.Dim_eq { d1; d2; origin = dummy_origin }

(* Run a solve over [env], classifying the outcome so an [assert false] regression (any
   non-[Shape_error] exception) is distinguished from a legitimate [Shape_error] and from success. *)
let run_solve_env ~stage cs env =
  try `Ok (Row.solve_inequalities ~stage cs env) with
  | Row.Shape_error (m, _) -> `Shape m
  | e -> `Other (Exn.to_string e)

let run_solve ~stage cs = run_solve_env ~stage cs Row.empty_env

let has_dim_ineq cs = List.exists cs ~f:(function Row.Dim_ineq _ -> true | _ -> false)

(* [v] forced through a size-[n] GLB iff equating it to [n] is consistent but to [n+1] conflicts. *)
let forced_to_size ~env v n =
  (match run_solve_env ~stage:Stage4 [ eq (Row.Var v) (dim n) ] env with `Ok _ -> true | _ -> false)
  && match run_solve_env ~stage:Stage4 [ eq (Row.Var v) (dim (n + 1)) ] env with
     | `Shape _ -> true
     | _ -> false

(* AC1 — GLB merge of two COMPATIBLE Concat bounds (stage 4) COMMITS: it forces the operand var
   through the merged GLB (and does not merely avoid raising). First inequality banks [Concat 2^3]
   (size 5) as the glb of a fresh operand var; the second feeds [Concat 1^4] (also size 5) into the
   merge, which [unify_dim] reconciles, so at stage >= 4 the merge commits. Pre-task: [assert false].

   The PASS condition pins the recipe, not just "Ok":
   - no residual [Dim_ineq] survives — a Stage4 RE-DEFER mutation would leave one (this is exactly the
     residual the postpone test below asserts IS present at Stage1);
   - the operand is forced to size 5 — equating v = 5 is consistent, v = 6 conflicts. A mutation that
     succeeds without forcing the operand through the GLB (drops the bound, or demotes to bcast-top)
     fails [forced_to_size] (under bcast-top v = 5 itself would conflict). *)
let test_ac1_glb_merge_compatible () =
  Stdio.printf "AC1: GLB merge of two compatible Concat bounds (stage 4) commits (forces operand)\n";
  let v = Row.get_var () in
  match
    run_solve ~stage:Stage4
      [ ineq (Row.Concat [ dim 2; dim 3 ]) v; ineq (Row.Concat [ dim 1; dim 4 ]) v ]
  with
  | `Ok (remaining, env) ->
      let no_residual = (not (has_dim_ineq remaining)) && not (has_dim_ineq (Row.unsolved_constraints env)) in
      let forced = forced_to_size ~env v 5 in
      if no_residual && forced then
        Stdio.printf "  PASS: operand committed to the merged size-5 GLB; no residual deferral\n"
      else
        Stdio.printf "  FAIL: expected commit (no_residual=%b forced_to_5=%b)\n" no_residual forced
  | `Shape m -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" m
  | `Other e -> Stdio.printf "  FAIL: regression (expected no assert false): %s\n" e

(* AC1 — GLB merge of two INCOMPATIBLE Concat bounds (stage 4) DEMOTES the operand to the broadcast-
   top GLB (size 1), swallowing the failed equality rather than propagating a Shape_error or
   committing to either bound. [Concat 2^3] (size 5) vs [Concat 2^4] (size 6): [unify_dim] fails.
   Pre-task: [assert false].

   The PASS condition pins demotion specifically: the operand is solved to size 1 — the broadcast-top
   that is distinct from BOTH bounds (5 and 6). A mutation that commits to either bound (5 or 6), or
   re-defers (operand unsolved), or drops the GLB yields [get_dim_val <> Some 1] and flips the
   assertion; merely "not raising" is not enough. *)
let test_ac1_glb_merge_incompatible_demotes () =
  Stdio.printf "AC1: GLB merge of two incompatible Concat bounds (stage 4) demotes to bcast-top\n";
  let v = Row.get_var () in
  match
    run_solve ~stage:Stage4
      [ ineq (Row.Concat [ dim 2; dim 3 ]) v; ineq (Row.Concat [ dim 2; dim 4 ]) v ]
  with
  | `Ok (_remaining, env) -> (
      match Row.get_dim_val env v with
      | Some 1 ->
          Stdio.printf "  PASS: operand demoted to broadcast-top (size 1), not committed to 5/6\n"
      | other ->
          Stdio.printf "  FAIL: expected demote to size 1, got %s\n"
            (match other with Some n -> Int.to_string n | None -> "<unsolved>"))
  | `Shape m -> Stdio.printf "  FAIL: should demote, not raise Shape_error: %s\n" m
  | `Other e -> Stdio.printf "  FAIL: regression (expected no assert false): %s\n" e

(* AC1 — below stage 4 the merge POSTPONES: it neither crashes, commits, nor demotes — it re-defers
   the inequality so the two bounds can still resolve equal later. PASS pins both halves: a [Dim_ineq]
   survives in the residual AND the operand is left UNSOLVED (a Stage1 demote/commit mutation would
   solve it — [get_dim_val] would be [Some _]). Pre-task: [assert false]. *)
let test_ac1_glb_merge_postpone_below_stage4 () =
  Stdio.printf "AC1: GLB merge below stage 4 postpones (defers, operand left unsolved)\n";
  let v = Row.get_var () in
  match
    run_solve ~stage:Stage1
      [ ineq (Row.Concat [ dim 2; dim 3 ]) v; ineq (Row.Concat [ dim 1; dim 4 ]) v ]
  with
  | `Ok (remaining, env) ->
      let deferred = has_dim_ineq remaining in
      let not_committed = Option.is_none (Row.get_dim_val env v) in
      if deferred && not_committed then
        Stdio.printf "  PASS: inequality deferred and operand left unsolved for a later stage\n"
      else
        Stdio.printf "  FAIL: expected postpone (deferred=%b operand_unsolved=%b)\n" deferred
          not_committed
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

(* Nested-[Concat] component (beyond the proposal, per user request): a component that is itself a
   [Concat] must be flattened (concatenation is associative — sum semantics) so the oldest-variable
   pairing aligns by flat position, not by nesting level. [Concat [Concat [a; b]; c]] =
   [Concat [d; e; f]] (all distinct vars) must link a=d, b=e, c=f. Without flattening the residual
   [Concat[a;b]; c] is not all-`Var` (its first component is a nested `Concat`), so the pairing would
   mis-align — its only top-level var c would pair with the other side's oldest var d — giving wrong
   linkage (or a deferral). We force a:=2, b:=3, c:=4 and require d, e, f to read back 2, 3, 4. *)
let test_nested_concat_flattened () =
  Stdio.printf "Nested Concat: Concat[Concat[a;b];c] = Concat[d;e;f] flattens, links a=d,b=e,c=f\n";
  let a = Row.get_var () and b = Row.get_var () and c = Row.get_var () in
  let d = Row.get_var () and e = Row.get_var () and f = Row.get_var () in
  let lhs = Row.Concat [ Row.Concat [ Row.Var a; Row.Var b ]; Row.Var c ] in
  let rhs = Row.Concat [ Row.Var d; Row.Var e; Row.Var f ] in
  match
    try
      let _rem, env = Row.solve_inequalities ~stage:Stage4 [ eq lhs rhs ] Row.empty_env in
      let _rem, env =
        Row.solve_inequalities ~stage:Stage4
          [ eq (Row.Var a) (dim 2); eq (Row.Var b) (dim 3); eq (Row.Var c) (dim 4) ]
          env
      in
      `Ok (Row.get_dim_val env d, Row.get_dim_val env e, Row.get_dim_val env f)
    with
    | Row.Shape_error (m, _) -> `Shape m
    | ex -> `Other (Exn.to_string ex)
  with
  | `Ok (Some 2, Some 3, Some 4) ->
      Stdio.printf "  PASS: nested concat flattened; d=2 e=3 f=4 (a=d, b=e, c=f)\n"
  | `Ok (dv, ev, fv) ->
      let show = function Some n -> Int.to_string n | None -> "?" in
      Stdio.printf "  FAIL: expected d,e,f = 2,3,4; got %s,%s,%s\n" (show dv) (show ev) (show fv)
  | `Shape m -> Stdio.printf "  FAIL: unexpected Shape_error: %s\n" m
  | `Other e -> Stdio.printf "  FAIL: regression: %s\n" e

(* Codex P2 regression: an all-`Var` residual on one side facing a residual that still holds a non-
   variable, non-solved component (an unresolved `Affine` from a strided axis) must NOT enter the
   pairing arm. [Concat [a; c]] = [Concat [Affine{2,w}; b]] is an underdetermined sum equality; the
   buggy "one side all-Var" guard equated the oldest variables (a=b) and forced the leftover (c) to
   the affine, over-constraining. The fix requires BOTH residuals to be variables-only, so this falls
   through to arithmetic deferral and leaves a and b INDEPENDENT. We detect the spurious a=b link: if
   it were present, forcing a:=5 then b:=7 would conflict; it must instead be consistent. *)
let test_ac3_affine_residual_not_overconstrained () =
  Stdio.printf "AC3: all-Var side vs Affine-bearing side — pairing must NOT over-constrain\n";
  let a = Row.get_var () and c = Row.get_var () and b = Row.get_var () and w = Row.get_var () in
  let aff = Row.Affine { stride = 2; over = Row.Var w; conv = None; stride_offset = 0 } in
  let lhs = Row.Concat [ Row.Var a; Row.Var c ] in
  let rhs = Row.Concat [ aff; Row.Var b ] in
  match run_solve ~stage:Stage4 [ eq lhs rhs ] with
  | `Ok (_remaining, env) -> (
      match run_solve_env ~stage:Stage4 [ eq (Row.Var a) (dim 5); eq (Row.Var b) (dim 7) ] env with
      | `Ok _ -> Stdio.printf "  PASS: a and b left independent (a:=5, b:=7 both consistent)\n"
      | `Shape m -> Stdio.printf "  FAIL: over-constrained (spurious a=b link): %s\n" m
      | `Other e -> Stdio.printf "  FAIL: unexpected exception: %s\n" e)
  | `Shape m -> Stdio.printf "  FAIL: first solve raised Shape_error: %s\n" m
  | `Other e -> Stdio.printf "  FAIL: regression: %s\n" e

let () =
  Stdio.printf "=== Concat dim-solver hardening (solver-level) ===\n";
  Tensor.unsafe_reinitialize ();
  test_ac1_glb_merge_compatible ();
  test_ac1_glb_merge_incompatible_demotes ();
  test_ac1_glb_merge_postpone_below_stage4 ();
  test_ac3_unequal_arity_all_var ();
  test_ac3_one_side_all_var_other_has_solved ();
  test_nested_concat_flattened ();
  test_ac3_affine_residual_not_overconstrained ();
  Stdio.printf "=== Done ===\n"

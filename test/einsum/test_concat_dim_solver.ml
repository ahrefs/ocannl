open! Base
open Ocannl

(* Solver-level regressions for the Concat dim-solver hardening (task-7d2ed931), following the
   [test_basis_total_order] pattern of driving [Row.solve_inequalities] on hand-built constraints.

   These exercise the two reachable defects the task fixes:

   - AC1: a broadcast inequality between two [Concat] bounds reaches [solve_dim_ineq]'s existing-GLB
     merge, whose [Affine]/[Concat] arms were [assert false]. We construct the inequality path
     directly (two [Dim_ineq]s against the same operand variable, so the first sets a [Concat] GLB
     and the second drives the merge) — the honest reachability check, since this is by construction
     a broadcast [Dim_ineq] between [Concat] bounds rather than an equality through [unify_dim].

   - AC3: an unequal-arity, variables-only [Concat = Concat] equality. The old guard required equal
     arity and left these deferred forever; the generalized stage-4+ variable-pairing arm equates the
     oldest variable of each side and re-runs, resolving deterministically at any arity.

   Crucially, every assertion checks the EXACT solver outcome, distinguishing three cases:
   [Solved] (returned with NO leftover constraints — the equation is actually solved), [Deferred n]
   (returned without error but left [n] unsolved constraints — the pre-fix behaviour for AC3), and
   [Raised] (a [Shape_error] / occurs-check / [assert false] escaped). A bare "did not raise" check
   would pass on the pre-fix [Deferred] behaviour, so it is not used. *)

let dummy_origin : Row.constraint_origin list =
  [ { lhs_name = "t"; lhs_kind = `Output; rhs_name = "t"; rhs_kind = `Output; operation = None } ]

let v () = Row.Var (Row.get_var ())

type outcome = Solved | Deferred of int | Raised

let outcome_to_string = function
  | Solved -> "solved"
  | Deferred n -> Printf.sprintf "deferred(%d)" n
  | Raised -> "raised"

let equal_outcome a b =
  match (a, b) with
  | Solved, Solved | Raised, Raised -> true
  | Deferred m, Deferred n -> m = n
  | _ -> false

(* Run the solver and classify: [Solved] iff it returns with an EMPTY leftover-constraint list (the
   equation is genuinely resolved, not merely deferred); [Deferred n] iff it returns [n] > 0 leftover
   constraints; [Raised] iff it raised. *)
let run ~stage cs : outcome =
  try
    match Row.solve_inequalities ~stage cs Row.empty_env with
    | [], _ -> Solved
    | leftover, _ -> Deferred (List.length leftover)
  with Row.Shape_error _ -> Raised

let report name ~(expect : outcome) (actual : outcome) =
  if equal_outcome expect actual then Stdio.printf "  PASS: %s [%s]\n" name (outcome_to_string actual)
  else
    Stdio.printf "  FAIL: %s (expected %s, got %s)\n" name (outcome_to_string expect)
      (outcome_to_string actual)

(* AC1 — the existing-GLB merge of two [Concat] bounds no longer crashes AND actually merges. The
   first [Dim_ineq] records [Concat [_; _]] as [vv]'s GLB; the second drives the merge at stage 4
   (the [Concat] arm that was [assert false]). Pre-fix this raised [Assert_failure]; post-fix it
   merges with no leftover ([Solved]). *)
let test_ac1_concat_glb_merge () =
  Stdio.printf "AC1: broadcast-GLB merge of two Concat bounds resolves (was assert false)\n";
  let vv = Row.get_var () in
  let cs =
    [
      Row.Dim_ineq
        { res = Row.Concat [ v (); v () ]; opnd = Row.Var vv; from_ = Sexp.List []; origin = dummy_origin };
      Row.Dim_ineq
        { res = Row.Concat [ v (); v () ]; opnd = Row.Var vv; from_ = Sexp.List []; origin = dummy_origin };
    ]
  in
  report "two Concat GLB bounds merge at stage 4" ~expect:Solved (run ~stage:Stage4 cs)

(* AC3 — unequal-arity, variables-only [Concat = Concat] is SOLVED (leftover empty) at stage 4. The
   disjoint cases are the shape the old equal-arity guard rejected (it would leave them [Deferred]);
   the shared-leading-component cases are the nested-stacking shape (a common operand axis cancels,
   leaving an unequal variables-only residual). [Solved] — not merely "did not raise" — is what
   distinguishes the fix from the pre-fix deferral. *)
let test_ac3_unequal_arity_solved () =
  Stdio.printf "AC3: unequal-arity variables-only Concat = Concat is SOLVED at stage 4\n";
  report "3-var = 2-var (disjoint)" ~expect:Solved
    (run ~stage:Stage4
       [ Row.Dim_eq { d1 = Row.Concat [ v (); v (); v () ]; d2 = Row.Concat [ v (); v () ]; origin = dummy_origin } ]);
  report "2-var = 3-var (disjoint)" ~expect:Solved
    (run ~stage:Stage4
       [ Row.Dim_eq { d1 = Row.Concat [ v (); v () ]; d2 = Row.Concat [ v (); v (); v () ]; origin = dummy_origin } ]);
  let s = v () in
  report "shared [s;a;b] = [s;c]" ~expect:Solved
    (run ~stage:Stage4
       [ Row.Dim_eq { d1 = Row.Concat [ s; v (); v () ]; d2 = Row.Concat [ s; v () ]; origin = dummy_origin } ]);
  let s2 = v () in
  report "shared [s;p] = [s;q;r]" ~expect:Solved
    (run ~stage:Stage4
       [ Row.Dim_eq { d1 = Row.Concat [ s2; v () ]; d2 = Row.Concat [ s2; v (); v () ]; origin = dummy_origin } ])

(* AC3 stage gating — below stage 4 the generalized variable-pairing arm must NOT fire, so an
   unequal-arity variables-only equation is left [Deferred] (exactly one leftover constraint here),
   NOT [Solved]. This instantiates the [is_stage4_up] guard's false branch: if the stage gate were
   removed, this would resolve to [Solved] and the assertion would fail. *)
let test_ac3_stage_gate_defers () =
  Stdio.printf "AC3: below stage 4 the equation is DEFERRED (gate holds), not solved\n";
  report "3-var = 2-var deferred at stage 1" ~expect:(Deferred 1)
    (run ~stage:Stage1
       [ Row.Dim_eq { d1 = Row.Concat [ v (); v (); v () ]; d2 = Row.Concat [ v (); v () ]; origin = dummy_origin } ])

let () =
  Stdio.printf "Concat dim-solver hardening regressions:\n\n";
  test_ac1_concat_glb_merge ();
  test_ac3_unequal_arity_solved ();
  test_ac3_stage_gate_defers ()

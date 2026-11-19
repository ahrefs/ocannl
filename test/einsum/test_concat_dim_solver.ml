open! Base
open Ocannl

(* Failures go through [Verdict], so that a regression exits nonzero instead of being `dune
   promote`d into the golden as the expected output (gh-ocannl-601). *)
let fail fmt = Printf.ksprintf Verdict.fail fmt

(* Solver-level regressions for the Concat dim-solver hardening (task-7d2ed931), following the
   [test_basis_total_order] pattern of driving [Row.solve_inequalities] on hand-built constraints.

   These exercise the two reachable defects the task fixes:

   - AC1: a broadcast inequality between two [Concat] bounds reaches [solve_dim_ineq]'s existing-GLB
   merge, whose [Affine]/[Concat] arms were [assert false]. We construct the inequality path
   directly (two [Dim_ineq]s against the same operand variable, so the first sets a [Concat] GLB and
   the second drives the merge) — the honest reachability check, since this is by construction a
   broadcast [Dim_ineq] between [Concat] bounds rather than an equality through [unify_dim].

   - AC3: an unequal-arity, variables-only [Concat = Concat] equality. A stage-4+ generalization of
   the variable pairing once equated the oldest variable of each side and re-ran, which sized a
   disjoint equation's leftover component to 0 ([[a; b; c] = [d; e]] gave [c = 0]), a size only a
   discardable component may take. The constraints decide no such pairing, so a disjoint equation
   now waits at every stage until its components are sized (gh-ocannl-2368). A shared component
   cancels, and a singleton residual side is the sum of the other ([[s; a; b] = [s; c]] solves [c =
   a + b]), so those stay solved.

   Crucially, every assertion checks the EXACT solver outcome, distinguishing three cases: [Solved]
   (returned with NO leftover constraints — the equation is actually solved), [Deferred n] (returned
   without error but left [n] unsolved constraints), and [Raised] (a [Shape_error] / occurs-check /
   [assert false] escaped). A bare "did not raise" check cannot tell a pending equation from a
   solved one, so it is not used. *)

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
   equation is genuinely resolved, not merely deferred); [Deferred n] iff it returns [n] > 0
   leftover constraints; [Raised] iff it raised. *)
let run ~stage cs : outcome =
  try
    match Row.solve_inequalities ~stage cs Row.empty_env with
    | [], _ -> Solved
    | leftover, _ -> Deferred (List.length leftover)
  with Row.Shape_error _ -> Raised

let report name ~(expect : outcome) (actual : outcome) =
  if equal_outcome expect actual then
    Stdio.printf "  PASS: %s [%s]\n" name (outcome_to_string actual)
  else fail "%s (expected %s, got %s)" name (outcome_to_string expect) (outcome_to_string actual)

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
        {
          res = Row.Concat [ v (); v () ];
          opnd = Row.Var vv;
          from_ = Sexp.List [];
          origin = dummy_origin;
        };
      Row.Dim_ineq
        {
          res = Row.Concat [ v (); v () ];
          opnd = Row.Var vv;
          from_ = Sexp.List [];
          origin = dummy_origin;
        };
    ]
  in
  report "two Concat GLB bounds merge at stage 4" ~expect:Solved (run ~stage:Stage4 cs)

(* AC3 — unequal-arity, variables-only [Concat = Concat] at stage 4. A disjoint equation waits
   ([Deferred 1]: its one equation stays pending), since pairing its variables would choose a split,
   and size a leftover to 0. A shared leading component cancels, leaving a singleton side that is
   the sum of the other, which is [Solved]. On the base the disjoint cases read [Solved], with the
   leftover sized 0. *)
let test_ac3_unequal_arity_solved () =
  Stdio.printf "AC3: unequal-arity variables-only Concat = Concat at stage 4: disjoint waits\n";
  report "3-var = 2-var (disjoint)" ~expect:(Deferred 1)
    (run ~stage:Stage4
       [
         Row.Dim_eq
           {
             d1 = Row.Concat [ v (); v (); v () ];
             d2 = Row.Concat [ v (); v () ];
             origin = dummy_origin;
           };
       ]);
  report "2-var = 3-var (disjoint)" ~expect:(Deferred 1)
    (run ~stage:Stage4
       [
         Row.Dim_eq
           {
             d1 = Row.Concat [ v (); v () ];
             d2 = Row.Concat [ v (); v (); v () ];
             origin = dummy_origin;
           };
       ]);
  let s = v () in
  report "shared [s;a;b] = [s;c]" ~expect:Solved
    (run ~stage:Stage4
       [
         Row.Dim_eq
           { d1 = Row.Concat [ s; v (); v () ]; d2 = Row.Concat [ s; v () ]; origin = dummy_origin };
       ]);
  let s2 = v () in
  report "shared [s;p] = [s;q;r]" ~expect:Solved
    (run ~stage:Stage4
       [
         Row.Dim_eq
           {
             d1 = Row.Concat [ s2; v () ];
             d2 = Row.Concat [ s2; v (); v () ];
             origin = dummy_origin;
           };
       ])

(* AC3 below stage 4 — the disjoint equation waits at stage 1 too: no stage pairs it. *)
let test_ac3_stage_gate_defers () =
  Stdio.printf "AC3: below stage 4 the equation is DEFERRED, not solved\n";
  report "3-var = 2-var deferred at stage 1" ~expect:(Deferred 1)
    (run ~stage:Stage1
       [
         Row.Dim_eq
           {
             d1 = Row.Concat [ v (); v (); v () ];
             d2 = Row.Concat [ v (); v () ];
             origin = dummy_origin;
           };
       ])

let () =
  Stdio.printf "Concat dim-solver hardening regressions:\n\n";
  test_ac1_concat_glb_merge ();
  test_ac3_unequal_arity_solved ();
  test_ac3_stage_gate_defers ()

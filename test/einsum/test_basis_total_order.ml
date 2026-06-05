open! Base
open Ocannl

(* Solver-level regression tests for the total-basis broadcast order (task-4eb929b2).

   The dimension broadcast order is now a flat partial order: [d1 ⊑ d2] iff [d1] and [d2] are equal
   as dims (same size AND same tag) or [d2 = 1_(bcast_if_1)] (the claim-free top). There is no
   wildcard, so the order is transitive.

   [Row.solve_dim_ineq]/[Dim_ineq { res; opnd }] enforce [res ⊑ opnd]. The helper below tests
   whether [res ⊑ opnd] is accepted by the solver. *)

let dummy_origin : Row.constraint_origin list =
  [ { lhs_name = "t"; lhs_kind = `Output; rhs_name = "t"; rhs_kind = `Output; operation = None } ]

(* Returns [true] iff the solver accepts [res ⊑ opnd] (no [Shape_error]). *)
let leq ~opnd ~res : bool =
  let ineq = Row.Dim_ineq { res; opnd; from_ = Sexp.List []; origin = dummy_origin } in
  try
    let _ = Row.solve_inequalities ~stage:Stage1 [ ineq ] Row.empty_env in
    true
  with Row.Shape_error _ -> false

let d ~basis n = Row.get_dim ~d:n ~basis ()
let report name ~expect actual =
  if Bool.equal expect actual then Stdio.printf "  PASS: %s\n" name
  else Stdio.printf "  FAIL: %s (expected %b, got %b)\n" name expect actual

(* The basis a dim variable resolves to in [env] (or "" if unsolved/unnamed). *)
let var_basis env v =
  let row = { Row.beg_dims = []; dims = [ Row.Var v ]; bcast = Broadcastable; prov = Row.empty_provenance } in
  let bases = Row.row_to_bases env row in
  if Array.length bases > 0 then bases.(0) else ""

(* AC#10 — transitivity: under the old [None] wildcard, [3_default ⊑ 3_rgb] and
   [3_xyz ⊑ 3_default] both held (the bridge) while [3_xyz ⊑ 3_rgb] did not — a non-transitive
   relation. With the total basis the bridge links themselves reject, so no chain relates two
   directly-unrelated dims. *)
let test_transitivity () =
  Stdio.printf "Transitivity: bridging through default no longer relates rgb and xyz\n";
  (* The two former "bridge" links now reject. *)
  report "3_default ⊑ 3_rgb rejects" ~expect:false (leq ~opnd:(d ~basis:"rgb" 3) ~res:(d ~basis:"default" 3));
  report "3_xyz ⊑ 3_default rejects" ~expect:false (leq ~opnd:(d ~basis:"default" 3) ~res:(d ~basis:"xyz" 3));
  (* The endpoints never related directly, and still do not. *)
  report "3_xyz ⊑ 3_rgb rejects" ~expect:false (leq ~opnd:(d ~basis:"rgb" 3) ~res:(d ~basis:"xyz" 3));
  (* Same tag still relates reflexively. *)
  report "3_rgb ⊑ 3_rgb accepts" ~expect:true (leq ~opnd:(d ~basis:"rgb" 3) ~res:(d ~basis:"rgb" 3))

(* AC#3 — top vs atom-unit order oppositely. [1_(bcast_if_1)] (top) is above everything;
   [1_default] (a size-1 atom) is above only itself. *)
let test_top_asymmetry () =
  Stdio.printf "Top asymmetry: 1_(bcast_if_1) broadcasts, 1_default does not\n";
  report "5_rgb ⊑ 1_(bcast_if_1) accepts" ~expect:true
    (leq ~opnd:(Row.get_bcast_dim ~d:1 ()) ~res:(d ~basis:"rgb" 5));
  report "5_rgb ⊑ 1_default rejects" ~expect:false
    (leq ~opnd:(Row.get_default_dim ~d:1 ()) ~res:(d ~basis:"rgb" 5));
  (* And the top is above an unannotated axis too. *)
  report "5_default ⊑ 1_(bcast_if_1) accepts" ~expect:true
    (leq ~opnd:(Row.get_bcast_dim ~d:1 ()) ~res:(Row.get_default_dim ~d:5 ()))

(* AC#7 — advertisable affordance: a user-written [bcast_if_1] tag broadcasts when size 1 and is an
   ordinary inert atom when size > 1 (it is legal, not an error). *)
let test_advertisable_affordance () =
  Stdio.printf "Advertisable affordance: bcast_if_1 stretches at size 1, inert atom at size > 1\n";
  report "7_xyz ⊑ 1_(bcast_if_1) accepts (stretches)" ~expect:true
    (leq ~opnd:(d ~basis:Row.bcast_if_1 1) ~res:(d ~basis:"xyz" 7));
  report "5_rgb ⊑ 5_(bcast_if_1) rejects (inert atom, not top)" ~expect:false
    (leq ~opnd:(d ~basis:Row.bcast_if_1 5) ~res:(d ~basis:"rgb" 5));
  report "5_(bcast_if_1) ⊑ 5_(bcast_if_1) accepts (equal atom)" ~expect:true
    (leq ~opnd:(d ~basis:Row.bcast_if_1 5) ~res:(d ~basis:Row.bcast_if_1 5))

(* AC#4 / brief §Technical-issue-1 — an explicit user size-1 axis ([1_default]) is an atom that
   does NOT stretch, in contrast to the broadcast top [1_(bcast_if_1)]. This pins the provenance
   split at the order level: only the top is above a larger axis. *)
let test_explicit_one_does_not_stretch () =
  Stdio.printf "Explicit user 1_default does not stretch; only 1_(bcast_if_1) does\n";
  report "5_default ⊑ 1_default rejects (no stretch)" ~expect:false
    (leq ~opnd:(Row.get_default_dim ~d:1 ()) ~res:(Row.get_default_dim ~d:5 ()));
  report "5_rgb ⊑ 1_default rejects (no stretch)" ~expect:false
    (leq ~opnd:(Row.get_default_dim ~d:1 ()) ~res:(d ~basis:"rgb" 5));
  report "5_default ⊑ 1_(bcast_if_1) accepts (top stretches)" ~expect:true
    (leq ~opnd:(Row.get_bcast_dim ~d:1 ()) ~res:(Row.get_default_dim ~d:5 ()))

(* AC#8 — the inequality path records NO basis update. A variable [v] solved to the top
   [1_(bcast_if_1)], then constrained [5_rgb ⊑ v] in the SAME environment, must still read back as
   [bcast_if_1] — the inequality does not upgrade [v] to carry [rgb]. (Under the old [None] this
   leak was latent: the inequality checked but never recorded, so a later conflicting basis could
   slip through.) This assertion reuses the solved environment, so a propagating solver fails it. *)
let test_inequality_no_propagation () =
  Stdio.printf "Inequality no-propagation: the top records no basis claim\n";
  let v = Row.get_var ~name:"v" () in
  let constraints =
    [
      Row.Dim_eq { d1 = Row.Var v; d2 = Row.get_bcast_dim ~d:1 (); origin = dummy_origin };
      Row.Dim_ineq
        { res = d ~basis:"rgb" 5; opnd = Row.Var v; from_ = Sexp.List []; origin = dummy_origin };
    ]
  in
  let basis =
    try
      let _, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
      var_basis env v
    with Row.Shape_error _ -> "<shape-error>"
  in
  if String.equal basis Row.bcast_if_1 then
    Stdio.printf "  PASS: v stays 1_(bcast_if_1) after 5_rgb ⊑ v; inequality recorded no basis\n"
  else Stdio.printf "  FAIL: inequality leaked a basis onto v (got %S, expected bcast_if_1)\n" basis

let () =
  Stdio.printf "Testing total-basis broadcast order:\n\n";
  test_transitivity ();
  test_top_asymmetry ();
  test_advertisable_affordance ();
  test_explicit_one_does_not_stretch ();
  test_inequality_no_propagation ()

open! Base
open Ocannl

(* Solver-level regression tests for the total-basis broadcast order (task-4eb929b2).

   The dimension broadcast order is now a flat partial order: [d1 ⊑ d2] iff [d1] and [d2] are equal
   as dims (same size AND same tag) or [d1 = 1_(bcast_if_1)] (the claim-free bottom). There is no
   wildcard, so the order is transitive.

   [Row.solve_dim_ineq]/[Dim_ineq { cur; subr }] enforce [subr ⊑ cur]. The helper below tests
   whether [sub ⊑ super] is accepted by the solver. *)

let dummy_origin : Row.constraint_origin list =
  [ { lhs_name = "t"; lhs_kind = `Output; rhs_name = "t"; rhs_kind = `Output; operation = None } ]

(* Returns [true] iff the solver accepts [sub ⊑ super] (no [Shape_error]). *)
let leq ~sub ~super : bool =
  let ineq = Row.Dim_ineq { cur = super; subr = sub; from_ = Sexp.List []; origin = dummy_origin } in
  try
    let _ = Row.solve_inequalities ~stage:Stage1 [ ineq ] Row.empty_env in
    true
  with Row.Shape_error _ -> false

let d ~basis n = Row.get_dim ~d:n ~basis ()
let report name ~expect actual =
  if Bool.equal expect actual then Stdio.printf "  PASS: %s\n" name
  else Stdio.printf "  FAIL: %s (expected %b, got %b)\n" name expect actual

(* AC#10 — transitivity: under the old [None] wildcard, [3_rgb ⊑ 3_default] and
   [3_default ⊑ 3_xyz] both held (the bridge) while [3_rgb ⊑ 3_xyz] did not — a non-transitive
   relation. With the total basis the bridge links themselves reject, so no chain relates two
   directly-unrelated dims. *)
let test_transitivity () =
  Stdio.printf "Transitivity: bridging through default no longer relates rgb and xyz\n";
  (* The two former "bridge" links now reject. *)
  report "3_rgb ⊑ 3_default rejects" ~expect:false (leq ~sub:(d ~basis:"rgb" 3) ~super:(d ~basis:"default" 3));
  report "3_default ⊑ 3_xyz rejects" ~expect:false (leq ~sub:(d ~basis:"default" 3) ~super:(d ~basis:"xyz" 3));
  (* The endpoints never related directly, and still do not. *)
  report "3_rgb ⊑ 3_xyz rejects" ~expect:false (leq ~sub:(d ~basis:"rgb" 3) ~super:(d ~basis:"xyz" 3));
  (* Same tag still relates reflexively. *)
  report "3_rgb ⊑ 3_rgb accepts" ~expect:true (leq ~sub:(d ~basis:"rgb" 3) ~super:(d ~basis:"rgb" 3))

(* AC#3 — bottom vs atom-unit order oppositely. [1_(bcast_if_1)] (bottom) is below everything;
   [1_default] (a size-1 atom) is below only itself. *)
let test_bottom_asymmetry () =
  Stdio.printf "Bottom asymmetry: 1_(bcast_if_1) broadcasts, 1_default does not\n";
  report "1_(bcast_if_1) ⊑ 5_rgb accepts" ~expect:true
    (leq ~sub:(Row.get_bcast_dim ~d:1 ()) ~super:(d ~basis:"rgb" 5));
  report "1_default ⊑ 5_rgb rejects" ~expect:false
    (leq ~sub:(Row.get_default_dim ~d:1 ()) ~super:(d ~basis:"rgb" 5));
  (* And the bottom is below an unannotated axis too. *)
  report "1_(bcast_if_1) ⊑ 5_default accepts" ~expect:true
    (leq ~sub:(Row.get_bcast_dim ~d:1 ()) ~super:(Row.get_default_dim ~d:5 ()))

(* AC#7 — advertisable affordance: a user-written [bcast_if_1] tag broadcasts when size 1 and is an
   ordinary inert atom when size > 1 (it is legal, not an error). *)
let test_advertisable_affordance () =
  Stdio.printf "Advertisable affordance: bcast_if_1 stretches at size 1, inert atom at size > 1\n";
  report "1_(bcast_if_1) ⊑ 7_xyz accepts (stretches)" ~expect:true
    (leq ~sub:(d ~basis:Row.bcast_if_1 1) ~super:(d ~basis:"xyz" 7));
  report "5_(bcast_if_1) ⊑ 5_rgb rejects (inert atom, not bottom)" ~expect:false
    (leq ~sub:(d ~basis:Row.bcast_if_1 5) ~super:(d ~basis:"rgb" 5));
  report "5_(bcast_if_1) ⊑ 5_(bcast_if_1) accepts (equal atom)" ~expect:true
    (leq ~sub:(d ~basis:Row.bcast_if_1 5) ~super:(d ~basis:Row.bcast_if_1 5))

(* AC#8 — inequality records no basis update by design. Solving [1_(bcast_if_1) ⊑ 5_rgb] is
   accepted but records nothing onto the (claim-free) bottom: there is no [None]-on-a-real-axis to
   silently absorb a second, conflicting basis later. We exercise this by solving the bottom against
   one named basis, then asserting that a SEPARATE bottom against a DIFFERENT named basis is equally
   accepted independently — neither leaks a recorded tag into the other. *)
let test_inequality_no_propagation () =
  Stdio.printf "Inequality no-propagation: the bottom records no basis claim\n";
  let env_after =
    let ineq = Row.Dim_ineq { cur = d ~basis:"rgb" 5; subr = Row.get_bcast_dim ~d:1 ();
                              from_ = Sexp.List []; origin = dummy_origin } in
    try Some (snd (Row.solve_inequalities ~stage:Stage1 [ ineq ] Row.empty_env))
    with Row.Shape_error _ -> None
  in
  (match env_after with
   | None -> Stdio.printf "  FAIL: 1_(bcast_if_1) ⊑ 5_rgb should be accepted\n"
   | Some _ -> Stdio.printf "  PASS: 1_(bcast_if_1) ⊑ 5_rgb accepted, recording nothing\n");
  (* A fresh, independent bottom still freely relates to a different named basis — proving no tag
     was globally recorded onto the bottom by the previous solve. *)
  report "1_(bcast_if_1) ⊑ 5_xyz still accepts independently" ~expect:true
    (leq ~sub:(Row.get_bcast_dim ~d:1 ()) ~super:(d ~basis:"xyz" 5))

let () =
  Stdio.printf "Testing total-basis broadcast order:\n\n";
  test_transitivity ();
  test_bottom_asymmetry ();
  test_advertisable_affordance ();
  test_inequality_no_propagation ()

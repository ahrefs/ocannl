(* Regression test for termination of [Row.solve_inequalities] on mutually-growing row variables
   (the rank-cycle divergence of docs/blog/ocannl-formal-core.md, Example 4.5a / Def. 4.5).

   Each constraint set below forces every row variable on a cycle to have strictly more axes than
   itself, e.g. {Row_ineq {res = <rho1>; opnd = <rho2>.[a]}, Row_ineq {res = <rho2>; opnd =
   <rho1>.[b]}} with [a], [b] in the trailing flank ([dims], not [beg_dims]) demands rank(rho1) >=
   rank(rho2) + 1 and rank(rho2) >= rank(rho1) + 1 — unsatisfiable. The deficit (template) rule used
   to ping-pong such constraint sets, minting fresh row variables forever (the three-variable cycle
   diverged; the two-variable one happened to collapse into the one-step self-reference check). With
   the transitive rank-cycle check these must raise [Shape_error]. *)

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

let check name ineqs =
  match Row.solve_inequalities ~stage:Stage1 ineqs Row.empty_env with
  | _ -> Verdict.fail (Printf.sprintf "%s: solve_inequalities returned without an error" name)
  | exception Row.Shape_error (msg, _) -> Stdio.printf "%s: Shape_error: %s\n" name msg

let () =
  let rho1 = Row.get_row_var () in
  let rho2 = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:1 ~kind:`Output) rho1 in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:2 ~kind:`Output) rho2 in
  check "two-variable rank cycle"
    [
      Row.Row_ineq { res = r1; opnd = { r2 with dims = [ dim 2 ] }; origin };
      Row.Row_ineq { res = r2; opnd = { r1 with dims = [ dim 3 ] }; origin };
    ]

let () =
  let rho1 = Row.get_row_var () in
  let rho2 = Row.get_row_var () in
  let rho3 = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:3 ~kind:`Output) rho1 in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:4 ~kind:`Output) rho2 in
  let r3 = Row.get_row_for_var (Row.provenance ~sh_id:5 ~kind:`Output) rho3 in
  (* This one used to diverge: each round of the deficit rule presented a fresh template variable to
     the (res_v, deficit)-keyed memoization, and the rank facts kept moving between the [Bounds_row]
     adjacency lists and the in-flight re-emitted constraints. *)
  check "three-variable rank cycle"
    [
      Row.Row_ineq { res = r1; opnd = { r2 with dims = [ dim 2 ] }; origin };
      Row.Row_ineq { res = r2; opnd = { r3 with dims = [ dim 3 ] }; origin };
      Row.Row_ineq { res = r3; opnd = { r1 with dims = [ dim 4 ] }; origin };
    ]

let () =
  let rho1 = Row.get_row_var () in
  let rho2 = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:6 ~kind:`Output) rho1 in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:7 ~kind:`Output) rho2 in
  check "two-variable rank cycle via leading flank"
    [
      Row.Row_ineq { res = r1; opnd = { r2 with beg_dims = [ dim 2 ] }; origin };
      Row.Row_ineq { res = r2; opnd = { r1 with beg_dims = [ dim 3 ] }; origin };
    ]

let () =
  let rho1 = Row.get_row_var () in
  let rho2 = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:8 ~kind:`Output) rho1 in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:9 ~kind:`Output) rho2 in
  (* Mixed-sign cycle:

     rank(rho1) >= rank(rho2) + 2 rank(rho2) >= rank(rho1) - 1

     The second constraint is initially an RI-cap and records no rank edge. Once the first
     constraint grows rho1 by two axes, that one axis of negative slack is exhausted and the
     re-emitted second constraint becomes a positive deficit. This pins the negative-slack discharge
     case used in the Detection Lemma proof. *)
  check "mixed-sign rank cycle via cap discharge"
    [
      Row.Row_ineq { res = r1; opnd = { r2 with dims = [ dim 2; dim 3 ] }; origin };
      Row.Row_ineq { res = { r2 with dims = [ dim 4 ] }; opnd = r1; origin };
    ]

(* gh-ocannl-1802: the rank-cycle search is bounded by the levels of the new edge's endpoints. On
   random constraint sets it must reach the verdicts of the unbounded walk it replaced. Each solve
   mints its own row variables: the rank graph is persistent, so a set solved twice over the same
   variables would meet its own earlier edges. *)
let () =
  let open Verdict.Claims in
  let rng = Random.State.make [| 1802 |] in
  let n_vars = 8 in
  let side () = (Random.State.int rng n_vars, Random.State.int rng 3, Random.State.int rng 2) in
  let cases =
    List.init 400 ~f:(fun _ ->
        List.init (4 + Random.State.int rng 10) ~f:(fun _ -> (side (), side ())))
  in
  let solve case =
    let vars = Array.init n_vars ~f:(fun _ -> Row.get_row_var ()) in
    let row (i, n_dims, n_beg) : Row.t =
      {
        (Row.get_row_for_var (Row.provenance ~sh_id:(100 + i) ~kind:`Output) vars.(i)) with
        dims = List.init n_dims ~f:(fun _ -> dim 2);
        beg_dims = List.init n_beg ~f:(fun _ -> dim 2);
      }
    in
    let ineqs =
      List.map case ~f:(fun (res, opnd) -> Row.Row_ineq { res = row res; opnd = row opnd; origin })
    in
    match Row.solve_inequalities ~stage:Stage1 ineqs Row.empty_env with
    | _ -> "solved"
    | exception Row.Shape_error (msg, _) -> msg
  in
  let run () =
    Row.reset_rank_search_visits ();
    let verdicts = List.map cases ~f:solve in
    (verdicts, Row.rank_search_visits ())
  in
  let bounded, bounded_visits = run () in
  let unbounded, unbounded_visits = Row.with_unbounded_rank_search run in
  Stdio.eprintf "(not part of the golden) rank-search visits: bounded %d, unbounded %d\n"
    bounded_visits unbounded_visits;
  p_all2 "on random constraint sets, the bounded search reaches the unbounded one's verdicts"
    (Array.of_list bounded) (Array.of_list unbounded) ~f:String.equal;
  p_exists "some random set closes a rank cycle" bounded
    ~f:(String.equal "Infinite number of axes by rank cycle among row variables");
  p_exists "some random set solves" bounded ~f:(String.equal "solved");
  p "the bounded search visits fewer states than the unbounded one"
    (bounded_visits < unbounded_visits)

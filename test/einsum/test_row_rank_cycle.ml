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

(* gh-ocannl-1891: a rank potential answers the rank-cycle check, without a search unless the edge
   breaks it. On random constraint sets it must reach the verdicts of the walk it replaced, which
   [with_unbounded_rank_search] also compares edge by edge. Each solve mints its own row variables:
   the rank graph is persistent, so a set solved twice over the same variables would meet its own
   earlier edges. *)
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
  Stdio.eprintf "(not part of the golden) rank-cycle check visits: potentials %d, walk %d\n"
    bounded_visits unbounded_visits;
  p_all2 "on random constraint sets, the rank potentials reach the walk's verdicts"
    (Array.of_list bounded) (Array.of_list unbounded) ~f:String.equal;
  p_exists "some random set closes a rank cycle" bounded
    ~f:(String.equal "Infinite number of axes by rank cycle among row variables");
  p_exists "some random set solves" bounded ~f:(String.equal "solved");
  p "the rank potentials move fewer variables than the walk visits"
    (bounded_visits < unbounded_visits)

(* gh-ocannl-1955: a terminal row is closed to its GLB by choice, so the choice yields to the rank
   facts. Both bounds below are true of rho = [], and [rho = [4; 4].<rho2>] contradicts the
   first. *)
let () =
  let rho = Row.get_row_var () and rho2 = Row.get_row_var () in
  let r = Row.get_row_for_var (Row.provenance ~sh_id:9 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:10 ~kind:`Output) rho2 in
  let constraints =
    [
      Row.Row_ineq { res = r2; opnd = r; origin };
      Row.Row_ineq { res = { r2 with beg_dims = [ dim 4; dim 4 ] }; opnd = r; origin };
      Row.Terminal_row (false, r, origin);
    ]
  in
  let solved =
    match
      List.fold [ Row.Stage1; Stage2; Stage3; Stage4; Stage5; Stage6; Stage7 ]
        ~init:(constraints, Row.empty_env) ~f:(fun (pending, env) stage ->
          Row.solve_inequalities ~stage pending env)
    with
    | _ -> "solved"
    | exception Row.Shape_error (msg, _) -> msg
  in
  Stdio.printf "a GLB the ranks refute is not closed to: %s\n" solved

(* gh-ocannl-1891: [Dim_ineq]'s cycle search on [layers] diamonds of dimension variables, each
   residual connection of a stack on unsized arguments: a chain with two ways through every layer.
   The search visits each variable once, and an edge already recorded repeats without one, which is
   what a trace's end validation does to every inequality. Closing the chain still finds the cycle,
   over the paths the visited set cuts short. *)
let () =
  let open Verdict.Claims in
  let layers = 10 in
  let var () = Row.Var (Row.get_var ()) in
  let joint_vars = Array.init (layers + 1) ~f:(fun _ -> Row.get_var ()) in
  let joints = Array.map joint_vars ~f:(fun v -> Row.Var v) in
  let ineq ~res ~opnd = Row.Dim_ineq { res; opnd; from_ = Sexp.List []; origin } in
  let diamonds =
    List.concat_map (List.range 0 layers) ~f:(fun i ->
        let left = var () and right = var () in
        [
          ineq ~res:joints.(i + 1) ~opnd:left;
          ineq ~res:joints.(i + 1) ~opnd:right;
          ineq ~res:left ~opnd:joints.(i);
          ineq ~res:right ~opnd:joints.(i);
        ])
  in
  let visits f =
    Row.reset_dim_cycle_visits ();
    let result = f () in
    (result, Row.dim_cycle_visits ())
  in
  let solve ineqs env = snd (Row.solve_inequalities ~stage:Stage1 ineqs env) in
  let env, first = visits (fun () -> solve diamonds Row.empty_env) in
  let _, repeated = visits (fun () -> solve diamonds env) in
  let _, walked =
    Row.with_unmemoized_cycle_search (fun () -> visits (fun () -> solve diamonds env))
  in
  Stdio.eprintf "(not part of the golden) cycle-search visits: first %d, repeated %d, walked %d\n"
    first repeated walked;
  p "repeating recorded edges takes no cycle search" (repeated = 0);
  p "walking every path repeats them at more than 2^layers visits" (walked > 1 lsl layers);
  (* The bottom of the chain above its top: everything is equal, whichever path the search took. *)
  let closed = solve [ ineq ~res:joints.(0) ~opnd:joints.(layers) ] env in
  let rec representative v =
    match Row.get_dim_from_env closed v with
    | Some (Row.Var u) -> representative u
    | Some _ -> -1
    | None -> Row.dim_var_id v
  in
  let representatives = Array.to_list joint_vars |> List.map ~f:representative in
  p "closing the chain equates its joints"
    (List.length (List.dedup_and_sort representatives ~compare:Int.compare) = 1
    && not (List.mem representatives (-1) ~equal:Int.equal))

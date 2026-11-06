(** What a row solution owes the entries bounded by it (gh-ocannl-1977).

    When [v := value] and [e] is bounded by [v], [unify_row] re-emits [v]'s own bounds, and the
    visit of [e] in [s_row_one_in_entry] emits past it. A value with axes must go past: rows carry a
    lower bound nowhere else, and no other test noticed its loss. A bare variable has no axes to
    carry, so it does not, which is what keeps a residual stack from storing the transitive closure
    of its bounds. Each scenario names the store and prints the closed rows, or the error.

    The sweep holds the solver to the constraints themselves and to the emission it replaced, on
    seeded random stores: every solution it reports must satisfy its store, checked here on ground
    rows, and no store the old emission solved may be refused. Differences in how a store closes are
    expected, and counted on stderr. *)

open! Base
open Ocannl
open Verdict.Claims

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
let prov () = Row.provenance ~sh_id:40 ~kind:`Output
let fresh () = Row.get_row_for_var (prov ()) (Row.get_row_var ())

let closed dims =
  { Row.beg_dims = []; dims = List.map dims ~f:dim; bcast = Broadcastable; prov = prov () }

let ineq res opnd = Row.Row_ineq { res; opnd; origin }
let eq r1 r2 = Row.Row_eq { r1; r2; origin }
let all_stages = [ Row.Stage1; Stage2; Stage3; Stage4; Stage5; Stage6; Stage7 ]

let through_stages constraints env =
  List.fold all_stages ~init:(constraints, env) ~f:(fun (constraints, env) stage ->
      Row.solve_inequalities ~stage constraints env)

(* One solve per step at stage 1, the shape of per-registration solving, then every stage. [leaves]
   are terminal rows, which close from their GLB; every row of [rows] is a shape to close. *)
let solve ~leaves ~rows steps =
  let init =
    List.map leaves ~f:(fun r -> Row.Terminal_row (false, r, origin))
    @ List.map rows ~f:(fun r -> Row.Shape_row (r, origin))
  in
  let pending, env =
    List.fold steps ~init:(init, Row.empty_env) ~f:(fun (pending, env) step ->
        Row.solve_inequalities ~stage:Stage1 (step @ pending) env)
  in
  snd (through_stages pending env)

let axes env (r : Row.t) =
  let r = Row.subst_row env r in
  List.map (r.beg_dims @ r.dims) ~f:(function Row.Dim { d; _ } -> Int.to_string d | _ -> "?")
  |> String.concat ~sep:","

let scenario name ~leaves ~rows steps =
  match solve ~leaves ~rows steps with
  | env ->
      Stdio.printf "%s: %s\n" name
        (String.concat ~sep:" " (List.map rows ~f:(fun r -> "[" ^ axes env r ^ "]")))
  | exception Row.Shape_error (msg, _) -> Stdio.printf "%s: Shape_error: %s\n" name msg

(* What must go past the entry. *)
let () =
  let v = fresh () and e = fresh () and l = fresh () in
  scenario "a value with axes reaches the far operand: v <= e <= leaf, v := [3]" ~leaves:[ l ]
    ~rows:[ v; e; l ]
    [ [ ineq v e; ineq e l ]; [ eq v (closed [ 3 ]) ] ];
  let v = fresh () and e = fresh () and u = fresh () in
  scenario "the entry's GLB reaches a bare value: [7] <= e <= v, v := u, u a leaf" ~leaves:[ u ]
    ~rows:[ v; e; u ]
    [ [ ineq (closed [ 7 ]) e; ineq e v ]; [ eq v u ] ]

(* What need not: every fact still arrives through the entry. *)
let () =
  let v = fresh () and e = fresh () and r = fresh () in
  scenario "the result side, value with axes: r <= e <= v, v := [5]" ~leaves:[ r ] ~rows:[ r; e; v ]
    [ [ ineq r e; ineq e v ]; [ eq v (closed [ 5 ]) ] ];
  let v = fresh () and e = fresh () and r = fresh () and u = fresh () in
  scenario "the result side, through an alias: r <= e <= v, v := u, u := [5]" ~leaves:[ r ]
    ~rows:[ r; e; u ]
    [ [ ineq r e; ineq e v ]; [ eq v u ]; [ eq u (closed [ 5 ]) ] ];
  let v = fresh () and e = fresh () and o = fresh () and u = fresh () in
  scenario "a refusal past an alias, operand side: v <= e <= o, v := u, o := [3], u := [4]"
    ~leaves:[] ~rows:[ u; e; o ]
    [ [ ineq v e; ineq e o ]; [ eq v u ]; [ eq o (closed [ 3 ]) ]; [ eq u (closed [ 4 ]) ] ];
  let v = fresh () and e = fresh () and r = fresh () and u = fresh () in
  scenario "a refusal past an alias, result side: r <= e <= v, v := u, r := [3], u := [4]"
    ~leaves:[] ~rows:[ r; e; u ]
    [ [ ineq r e; ineq e v ]; [ eq v u ]; [ eq r (closed [ 3 ]) ]; [ eq u (closed [ 4 ]) ] ];
  let v = fresh () and e = fresh () and o = fresh () and u = fresh () in
  scenario "the same refusal over an entry no shape closes: v <= e <= o" ~leaves:[] ~rows:[ u; o ]
    [ [ ineq v e; ineq e o ]; [ eq v u ]; [ eq o (closed [ 3 ]) ]; [ eq u (closed [ 4 ]) ] ]

(* An alias does not lengthen a lower bound's reach: it is two hops with or without one. The first
   scenario used to close its leaf to [3], through an edge the alias stored; with the equations in
   the other order it did not. *)
let () =
  let v = fresh () and e = fresh () and o = fresh () and l = fresh () and u = fresh () in
  scenario "reach past an alias: v <= e <= o <= leaf, v := u, u := [3]" ~leaves:[ l ]
    ~rows:[ u; e; o; l ]
    [ [ ineq v e; ineq e o; ineq o l ]; [ eq v u ]; [ eq u (closed [ 3 ]) ] ];
  let e = fresh () and o = fresh () and l = fresh () and u = fresh () in
  scenario "the same store written with u: u <= e <= o <= leaf, u := [3]" ~leaves:[ l ]
    ~rows:[ u; e; o; l ]
    [ [ ineq u e; ineq e o; ineq o l ]; [ eq u (closed [ 3 ]) ] ];
  let v = fresh () and e = fresh () and o = fresh () and l = fresh () and u = fresh () in
  scenario "the equations swapped: u := [3], then v := u" ~leaves:[ l ] ~rows:[ u; e; o; l ]
    [ [ ineq v e; ineq e o; ineq o l ]; [ eq u (closed [ 3 ]) ]; [ eq v u ] ];
  Row.with_transitive_row_emissions (fun () ->
      let v = fresh () and e = fresh () and o = fresh () and l = fresh () and u = fresh () in
      scenario "the old emission, reach past an alias" ~leaves:[ l ] ~rows:[ u; e; o; l ]
        [ [ ineq v e; ineq e o; ineq o l ]; [ eq v u ]; [ eq u (closed [ 3 ]) ] ];
      let v = fresh () and e = fresh () and o = fresh () and l = fresh () and u = fresh () in
      scenario "the old emission, the equations swapped" ~leaves:[ l ] ~rows:[ u; e; o; l ]
        [ [ ineq v e; ineq e o; ineq o l ]; [ eq u (closed [ 3 ]) ]; [ eq v u ] ])

(* A store over five rows, as the indices of the rows it relates. *)
type step =
  | Bound of int * int  (** [r_i ⊑ r_j] *)
  | Alias of int * int  (** [r_i = r_j] *)
  | Axes of int * int list  (** [r_i = dims] *)
  | Open of int * int * int  (** [r_i = <r_j>.[d]] *)

let num_rows = 5
let num_stores = 2000

type store = { steps : (step * Row.stage) list; leaves : int list }

let random_store state =
  let int n = Random.State.int state n in
  let pair () =
    let i = int num_rows in
    (i, (i + 1 + int (num_rows - 1)) % num_rows)
  in
  let axes () = List.nth_exn [ [ 2 ]; [ 3 ]; [ 2; 3 ]; [ 3; 2 ] ] (int 4) in
  let bounds = List.init (2 + int 5) ~f:(fun _ -> Bound (fst (pair ()), snd (pair ()))) in
  let bounds = List.filter bounds ~f:(function Bound (i, j) -> i <> j | _ -> true) in
  let equations =
    List.init
      (1 + int 3)
      ~f:(fun _ ->
        match int 7 with
        | 0 | 1 | 2 ->
            let i, j = pair () in
            Alias (i, j)
        | 3 | 4 | 5 -> Axes (int num_rows, axes ())
        | _ ->
            let i, j = pair () in
            Open (i, j, 2))
  in
  let steps =
    List.map (bounds @ equations) ~f:(fun s -> (Random.State.bits state, s))
    |> List.sort ~compare:(fun (a, _) (b, _) -> Int.compare a b)
    |> List.map ~f:snd
  in
  let at_stage1 = int (List.length steps + 1) in
  {
    steps = List.mapi steps ~f:(fun i s -> (s, if i < at_stage1 then Row.Stage1 else Row.Stage2));
    leaves = List.filter (List.range 0 num_rows) ~f:(fun _ -> int 2 = 0);
  }

(* The closed rows, or the refusal. [None] marks a row with an axis that closed to no number. *)
let run { steps; leaves } =
  let rows = Array.init num_rows ~f:(fun _ -> fresh ()) in
  let constraint_ = function
    | Bound (i, j) -> ineq rows.(i) rows.(j)
    | Alias (i, j) -> eq rows.(i) rows.(j)
    | Axes (i, ds) -> eq rows.(i) (closed ds)
    | Open (i, j, d) -> eq rows.(i) { (rows.(j)) with dims = [ dim d ] }
  in
  let init =
    List.map leaves ~f:(fun i -> Row.Terminal_row (false, rows.(i), origin))
    @ List.map (Array.to_list rows) ~f:(fun r -> Row.Shape_row (r, origin))
  in
  match
    List.fold steps ~init:(init, Row.empty_env) ~f:(fun (pending, env) (s, stage) ->
        Row.solve_inequalities ~stage (constraint_ s :: pending) env)
    |> fun (pending, env) -> through_stages pending env
  with
  | _, env ->
      let ground r =
        let r = Row.subst_row env r in
        Option.all
          (List.map (r.beg_dims @ r.dims) ~f:(function Row.Dim { d; _ } -> Some d | _ -> None))
      in
      Ok (Array.map rows ~f:ground)
  | exception Row.Shape_error (msg, _) -> Error msg

(* The order itself, on ground rows, aligned on the right as flanks-free rows are. *)
let refines res opnd =
  let excess = List.length res - List.length opnd in
  excess >= 0 && List.for_all2_exn (List.drop res excess) opnd ~f:(fun r o -> r = o || o = 1)

let holds rows (step, _) =
  let get i = rows.(i) in
  let check f = function [ Some a; Some b ] -> f a b | _ -> false in
  match step with
  | Bound (i, j) -> check refines [ get i; get j ]
  | Alias (i, j) -> check (List.equal Int.equal) [ get i; get j ]
  | Axes (i, ds) -> Option.equal (List.equal Int.equal) (get i) (Some ds)
  | Open (i, j, d) -> check (fun a b -> List.equal Int.equal a (b @ [ d ])) [ get i; get j ]

let () =
  let state = Random.State.make [| 1977 |] in
  let stores = List.init num_stores ~f:(fun _ -> random_store state) in
  let outcomes =
    List.map stores ~f:(fun store ->
        (store, run store, Row.with_transitive_row_emissions (fun () -> run store)))
  in
  let solved =
    List.filter_map outcomes ~f:(fun (s, r, _) -> Result.ok r |> Option.map ~f:(fun r -> (s, r)))
  in
  let count f = List.count outcomes ~f in
  Stdio.eprintf
    "(not part of the golden) %d stores, %d solved; against the old emission: %d close \
     differently, %d newly solved, %d newly refused\n"
    num_stores (List.length solved)
    (count (fun (_, n, o) -> match (n, o) with Ok a, Ok b -> not (Poly.equal a b) | _ -> false))
    (count (fun (_, n, o) -> Result.is_ok n && Result.is_error o))
    (count (fun (_, n, o) -> Result.is_error n && Result.is_ok o));
  Verdict.case "random stores" (fun () ->
      p "the sweep solves stores and refuses others"
        (List.length solved > 0 && List.length solved < num_stores);
      p_all "every solution satisfies its store, on ground rows" solved ~f:(fun (store, rows) ->
          List.for_all store.steps ~f:(holds rows));
      p_none "no store the old emission solved is refused" outcomes ~f:(fun (_, n, o) ->
          Result.is_error n && Result.is_ok o))

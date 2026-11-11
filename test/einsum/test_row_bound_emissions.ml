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
let all_stages = [ Row.Stage1; Stage2; Stage3; Stage4; Stage5; Stage6a; Stage6b; Stage7 ]

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

(* A lower bound travels the whole chain (gh-ocannl-2126), so the leaf closes to [3] with or without
   an alias, in either equation order and under either emission. While a row's bound reached two
   hops, these stores closed the leaf empty, except where the old emission's alias stored an edge
   that carried it a third. *)
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
      (* The old emission stored the transitive closure of the bounds; now that a bound travels the
         chain by itself, the stored pairs decide nothing. *)
      p_none "the old emission decides or closes no store differently" outcomes ~f:(fun (_, n, o) ->
          match (n, o) with
          | Ok a, Ok b -> not (Poly.equal a b)
          | Error _, Error _ -> false
          | _ -> true))

(* No guess between two rows (gh-ocannl-2039, gh-ocannl-2142). From Stage2 on, re-processing a
   recorded [r ⊑ o] whose rows were each the other's only bound used to equate them: made mid-pass
   it refused stores a solution existed for, by where the repeated pair sat in the call, and held to
   the call's fixpoint it still fired only on a repeat. It stood in for a bound's reach along a
   chain; the bound travels the chain now (gh-ocannl-2126), and a repeated bound is no event. Each
   case states a store a solution exists for, and claims it is solved with the rows that solution
   prescribes, or a store none exists for, and claims it is refused. *)

let ground env r =
  let r = Row.subst_row env r in
  Option.all (List.map (r.beg_dims @ r.dims) ~f:(function Row.Dim { d; _ } -> Some d | _ -> None))

(* Each step its own call, at its stage, then every stage: the sweep's harness. *)
let solve_steps ~rows steps =
  let init = List.map rows ~f:(fun r -> Row.Shape_row (r, origin)) in
  match
    List.fold steps ~init:(init, Row.empty_env) ~f:(fun (pending, env) (stage, step) ->
        Row.solve_inequalities ~stage (step @ pending) env)
    |> fun (pending, env) -> through_stages pending env
  with
  | _, env -> Ok (List.map rows ~f:(ground env))
  | exception Row.Shape_error (msg, _) -> Error msg

(* The store as production closes it: [recorded] solved at stage 1, then ONE call per closing stage,
   [at_stage2] all in the stage-2 call. *)
let solve_closing ~rows ~recorded at_stage2 =
  let init = List.map rows ~f:(fun r -> Row.Shape_row (r, origin)) in
  match
    let pending, env = Row.solve_inequalities ~stage:Stage1 (recorded @ init) Row.empty_env in
    let pending, env = Row.solve_inequalities ~stage:Stage2 (at_stage2 @ pending) env in
    List.fold [ Row.Stage3; Stage4; Stage5; Stage6a; Stage6b; Stage7 ] ~init:(pending, env)
      ~f:(fun (pending, env) stage -> Row.solve_inequalities ~stage pending env)
  with
  | _, env -> Ok (List.map rows ~f:(ground env))
  | exception Row.Shape_error (msg, _) -> Error msg

let solved_to want = function
  | Ok got -> List.equal (Option.equal (List.equal Int.equal)) got (List.map want ~f:Option.some)
  | Error _ -> false

let refused = Result.is_error

(* The issue's store. Its stage-2 equations arrive either in ONE call, as the closing does it, or
   one call each, as the sweep above feeds them. Both solve: while a guess was made at a stage-2
   call's fixpoint, a call after it could only find [e = u] already chosen, and the second form was
   refused. *)
let issue_store ~one_call () =
  let e = fresh () and u = fresh () and v = fresh () in
  let a = fresh () and f = fresh () and w = fresh () in
  let at_stage1 =
    List.map
      ~f:(fun c -> (Row.Stage1, [ c ]))
      [ ineq e u; ineq e v; ineq a f; ineq f w; eq w u; eq f (closed [ 2 ]) ]
  in
  let at_stage2 = [ eq v u; eq e (closed [ 3; 2 ]); eq u (closed [ 2 ]) ] in
  let at_stage2 =
    if one_call then [ (Row.Stage2, at_stage2) ]
    else List.map at_stage2 ~f:(fun c -> (Row.Stage2, [ c ]))
  in
  solve_steps ~rows:[ e; u; v; a; f; w ] (at_stage1 @ at_stage2)

let () =
  (* The witness: e = a = [3,2], u = v = f = w = [2]; [a] closes to [2] from its bound. *)
  let rows_ok = function
    | Ok [ e; u; v; a; f; w ] ->
        Option.equal (List.equal Int.equal) e (Some [ 3; 2 ])
        && List.equal
             (Option.equal (List.equal Int.equal))
             [ u; v; f; w ]
             [ Some [ 2 ]; Some [ 2 ]; Some [ 2 ]; Some [ 2 ] ]
        && Option.value_map a ~default:false ~f:(fun a -> refines a [ 2 ])
    | _ -> false
  in
  Verdict.case "the issue's store" (fun () ->
      p "e<=u, e<=v, a<=f, f<=w, w=u, f=[2], then in one stage-2 call v=u, e=[3,2], u=[2]: solved"
        (rows_ok (issue_store ~one_call:true ()));
      p "the same store under the transitive emission: solved"
        (rows_ok (Row.with_transitive_row_emissions (issue_store ~one_call:true))));
  Verdict.case "the issue's store, a call per equation" (fun () ->
      p "the issue's store, one stage-2 call per equation: solved"
        (rows_ok (issue_store ~one_call:false ())))

let () =
  let pair () =
    let r = fresh () and o = fresh () in
    (r, o)
  in
  let closing ?(recorded = fun r o -> [ ineq r o ]) at_stage2 =
    let r, o = pair () in
    solve_closing ~rows:[ r; o ] ~recorded:(recorded r o) (at_stage2 r o)
  in
  let r_32 r = eq r (closed [ 3; 2 ]) and o_2 o = eq o (closed [ 2 ]) in
  Verdict.case "a repeated bound in one stage-2 call" (fun () ->
      p "r<=o recorded, then r=[3,2], o=[2]: solved"
        (solved_to [ [ 3; 2 ]; [ 2 ] ] (closing (fun r o -> [ r_32 r; o_2 o ])));
      p "the same with the bound repeated first: solved"
        (solved_to [ [ 3; 2 ]; [ 2 ] ] (closing (fun r o -> [ ineq r o; r_32 r; o_2 o ])));
      p "the same with the bound repeated last: solved"
        (solved_to [ [ 3; 2 ]; [ 2 ] ] (closing (fun r o -> [ r_32 r; o_2 o; ineq r o ]))));
  Verdict.case "what a bound still refuses" (fun () ->
      p "r<=o repeated, r=[2], o=[3,2]: refused, the operand has more axes"
        (refused (closing (fun r o -> [ ineq r o; eq r (closed [ 2 ]); eq o (closed [ 3; 2 ]) ])));
      p "r<=o repeated, r=[3,2], o=[3]: refused, the aligned axes differ"
        (refused (closing (fun r o -> [ ineq r o; r_32 r; eq o (closed [ 3 ]) ])));
      p "r<=o and o<=r, r=[3,2], o=[2]: refused, the bounds make them equal"
        (refused
           (closing
              ~recorded:(fun r o -> [ ineq r o; ineq o r ])
              (fun r o -> [ ineq r o; r_32 r; o_2 o ])));
      p "r<=o and o<=r, r=[3,2]: solved, o with it"
        (solved_to
           [ [ 3; 2 ]; [ 3; 2 ] ]
           (closing ~recorded:(fun r o -> [ ineq r o; ineq o r ]) (fun r _ -> [ r_32 r ]))))

(* The contract over random stores: repeating a recorded bound anywhere in the stage-2 call changes
   no store's outcome, neither whether it solves nor what closes. The stage-1 steps are solved one
   per call as before; every later step goes into the one stage-2 call. *)
let run_closing ?dup { steps; leaves } =
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
  let early, late = List.partition_tf steps ~f:(function _, Row.Stage1 -> true | _ -> false) in
  let late = List.map late ~f:(fun (s, _) -> constraint_ s) in
  let late =
    match dup with
    | None -> late
    | Some (bound, at) -> List.take late at @ (constraint_ bound :: List.drop late at)
  in
  match
    let pending, env =
      List.fold early ~init:(init, Row.empty_env) ~f:(fun (pending, env) (s, _) ->
          Row.solve_inequalities ~stage:Stage1 (constraint_ s :: pending) env)
    in
    let pending, env = Row.solve_inequalities ~stage:Stage2 (late @ pending) env in
    List.fold [ Row.Stage3; Stage4; Stage5; Stage6a; Stage6b; Stage7 ] ~init:(pending, env)
      ~f:(fun (pending, env) stage -> Row.solve_inequalities ~stage pending env)
  with
  | _, env ->
      let ground r =
        let r = Row.subst_row env r in
        Option.all
          (List.map (r.beg_dims @ r.dims) ~f:(function Row.Dim { d; _ } -> Some d | _ -> None))
      in
      Ok (Array.map rows ~f:ground)
  | exception Row.Shape_error (msg, _) -> Error msg

let () =
  let state = Random.State.make [| 2039 |] in
  let stores = List.init num_stores ~f:(fun _ -> random_store state) in
  let with_dup =
    List.filter_map stores ~f:(fun store ->
        let bounds =
          List.filter_map store.steps ~f:(function (Bound _ as b), _ -> Some b | _ -> None)
        in
        let late = List.count store.steps ~f:(function _, Row.Stage1 -> false | _ -> true) in
        if List.is_empty bounds then None
        else
          let bound = List.nth_exn bounds (Random.State.int state (List.length bounds)) in
          Some (store, (bound, Random.State.int state (late + 1))))
  in
  let outcomes =
    List.map with_dup ~f:(fun (store, dup) -> (run_closing store, run_closing ~dup store))
  in
  let same (a, b) =
    match (a, b) with Ok a, Ok b -> Poly.equal a b | Error _, Error _ -> true | _ -> false
  in
  Stdio.eprintf
    "(not part of the golden) %d stores with a bound to repeat, %d solved, %d changed by the repeat\n"
    (List.length outcomes)
    (List.count outcomes ~f:(fun (a, _) -> Result.is_ok a))
    (List.count outcomes ~f:(fun o -> not (same o)));
  Stdio.eprintf
    "(not part of the golden) by the repeat: %d newly refused, %d newly solved, %d close differently\n"
    (List.count outcomes ~f:(function Ok _, Error _ -> true | _ -> false))
    (List.count outcomes ~f:(function Error _, Ok _ -> true | _ -> false))
    (List.count outcomes ~f:(function Ok a, Ok b -> not (Poly.equal a b) | _ -> false));
  Verdict.case "random stores, one stage-2 call" (fun () ->
      p_none ~min:1000 "repeating a recorded bound in the stage-2 call changes no store's outcome"
        outcomes ~f:(fun o -> not (same o)))

(* The dimension twin of the row guess (gh-ocannl-2141). From Stage2 on, re-processing a recorded
   [d1 ⊑ d2] whose variables are each the other's only bound equates them, which a solution need not
   do: [3 ⊑ 1_(bcast_if_1)] holds unequal. It matches only an edge already recorded, so it fires
   only on a repeated bound -- and the closure repeats every bound, re-solving its stage-1
   constraints at Stage2. Deleting it made [Nn_blocks.dropout]'s mask one value broadcast over its
   input (gh-ocannl-2185): its merge carries a draw's resolve-at-use mark onto the axis a result's
   template minted (gh-ocannl-2180), so it stays until that axis keeps its row's role. What does not
   depend on the repeat is claimed; what does is printed, not claimed, so the guess's removal shows
   as a diff. Each claim states a store a solution exists for, and claims it is solved with the
   dimensions, sizes and bases, that solution prescribes, or a store none exists for, and claims it
   is refused. *)

let dim_ineq res opnd = Row.Dim_ineq { res; opnd; from_ = Sexp.Atom "test"; origin }
let dim_eq d1 d2 = Row.Dim_eq { d1; d2; origin }
let dim_var () = Row.Var (Row.get_var ())
let top () = Row.get_bcast_dim ~d:1 ()
let dim_row d = { Row.beg_dims = []; dims = [ d ]; bcast = Broadcastable; prov = prov () }

(* A closed dimension as its size and basis: [get_dim_val] alone cannot tell the broadcast top from
   a default [1]. *)
let ground_dim env d =
  match (Row.subst_row env (dim_row d)).dims with
  | [ Row.Dim { d; basis; _ } ] -> Some (d, basis)
  | _ -> None

(* [recorded] solved at Stage1, then each of [calls] its own stage-2 call, then every closing stage.
   Each dimension sits in a row to close; [leaves] are terminal, closing to their use sites. *)
let solve_dims ?(leaves = []) ~dims ~recorded calls =
  let init =
    List.map leaves ~f:(fun d -> Row.Terminal_dim (false, d, origin))
    @ List.map dims ~f:(fun d -> Row.Shape_row (dim_row d, origin))
  in
  match
    let pending, env = Row.solve_inequalities ~stage:Stage1 (recorded @ init) Row.empty_env in
    let pending, env =
      List.fold calls ~init:(pending, env) ~f:(fun (pending, env) call ->
          Row.solve_inequalities ~stage:Stage2 (call @ pending) env)
    in
    List.fold [ Row.Stage3; Stage4; Stage5; Stage6a; Stage6b; Stage7 ] ~init:(pending, env)
      ~f:(fun (pending, env) stage -> Row.solve_inequalities ~stage pending env)
  with
  | _, env -> Ok (List.map dims ~f:(ground_dim env))
  | exception Row.Shape_error (msg, _) -> Error msg

let dims_to want = function
  | Ok got -> List.equal (Option.equal Poly.equal) got (List.map want ~f:Option.some)
  | Error _ -> false

let () =
  let default d = (d, Row.default_basis) and top_ = (1, Row.bcast_if_1) in
  (* [d1 ⊑ d2] recorded; [calls] builds the stage-2 calls from [d1], [d2]. *)
  let pair ?(two_way = false) calls =
    let d1 = dim_var () and d2 = dim_var () in
    let recorded = dim_ineq d1 d2 :: (if two_way then [ dim_ineq d2 d1 ] else []) in
    solve_dims ~dims:[ d1; d2 ] ~recorded (calls d1 d2)
  in
  let three d1 = dim_eq d1 (dim 3) and to_top d2 = dim_eq d2 (top ()) in
  Verdict.case "a repeated dimension bound" (fun () ->
      p "d1<=d2 recorded, then d1=3, d2=1_(bcast_if_1) in one stage-2 call: solved to 3, the top"
        (dims_to [ default 3; top_ ] (pair (fun d1 d2 -> [ [ three d1; to_top d2 ] ])));
      p "the bound repeated between the equations: solved alike"
        (dims_to
           [ default 3; top_ ]
           (pair (fun d1 d2 -> [ [ three d1; dim_ineq d1 d2; to_top d2 ] ])));
      p "the bound repeated last: solved alike"
        (dims_to
           [ default 3; top_ ]
           (pair (fun d1 d2 -> [ [ three d1; to_top d2; dim_ineq d1 d2 ] ]))));
  (* Where the repeat comes before an equation the guess equates the pair first, and the store is
     refused. A guess made at the repeat-only call's fixpoint would do the same. *)
  let show = function
    | Ok dims ->
        String.concat ~sep:" "
          (List.map dims ~f:(function Some (d, b) -> Int.to_string d ^ "_" ^ b | None -> "?"))
    | Error msg -> "Shape_error: " ^ msg
  in
  Stdio.printf "the dim twin, the bound repeated first: %s\n"
    (show (pair (fun d1 d2 -> [ [ dim_ineq d1 d2; three d1; to_top d2 ] ])));
  Stdio.printf "the dim twin, the equations swapped, the bound repeated first: %s\n"
    (show (pair (fun d1 d2 -> [ [ dim_ineq d1 d2; to_top d2; three d1 ] ])));
  Stdio.printf "the dim twin, the bound repeated alone, then a call per equation: %s\n"
    (show (pair (fun d1 d2 -> [ [ dim_ineq d1 d2 ]; [ three d1 ]; [ to_top d2 ] ])));
  Verdict.case "what a dimension bound still decides" (fun () ->
      p "d1<=d2 repeated, d1=3, d2=3: solved, equal"
        (dims_to
           [ default 3; default 3 ]
           (pair (fun d1 d2 -> [ [ dim_ineq d1 d2; three d1; dim_eq d2 (dim 3) ] ])));
      p "d1<=d2 repeated, d1=3: solved, d2 closing above it"
        (match pair (fun d1 d2 -> [ [ dim_ineq d1 d2; three d1 ] ]) with
        | Ok [ Some a; Some b ] ->
            Poly.equal a (default 3) && (Poly.equal b (default 3) || Poly.equal b top_)
        | _ -> false);
      p "d1<=d2 repeated, d1=3, d2=1_(default): refused, a default 1 is no top"
        (refused (pair (fun d1 d2 -> [ [ dim_ineq d1 d2; three d1; dim_eq d2 (dim 1) ] ])));
      p "d1<=d2 repeated, d1=3, d2=2: refused"
        (refused (pair (fun d1 d2 -> [ [ dim_ineq d1 d2; three d1; dim_eq d2 (dim 2) ] ])));
      p "d1<=d2 repeated, d1=3_(rgb), d2=3: refused, the bases differ"
        (refused
           (pair (fun d1 d2 ->
                [
                  [
                    dim_ineq d1 d2; dim_eq d1 (Row.get_dim ~d:3 ~basis:"rgb" ()); dim_eq d2 (dim 3);
                  ];
                ])));
      p "d1<=d2 and d2<=d1, d1=3, d2=1_(bcast_if_1): refused, the bounds make them equal"
        (refused (pair ~two_way:true (fun d1 d2 -> [ [ dim_ineq d1 d2; three d1; to_top d2 ] ])));
      p "d1<=d2 and d2<=d1, d1=3: solved, d2 with it"
        (dims_to
           [ default 3; default 3 ]
           (pair ~two_way:true (fun d1 d2 -> [ [ dim_ineq d1 d2; three d1 ] ]))));
  (* A chain: the lower bound reaches the leaf with or without a repeat, before or after the
     edges. *)
  let chain ~repeat ~early =
    let d1 = dim_var () and d2 = dim_var () and d3 = dim_var () in
    let edges = [ dim_ineq d1 d2; dim_ineq d2 d3 ] in
    let size = [ three d1 ] in
    let recorded = if early then size @ edges else edges in
    let call = (if repeat then edges else []) @ if early then [] else size in
    solve_dims ~leaves:[ d3 ] ~dims:[ d1; d2; d3 ] ~recorded [ call ]
  in
  Verdict.case "a dimension chain" (fun () ->
      let leaf_is_3 = function Ok [ _; _; Some l ] -> Poly.equal l (default 3) | _ -> false in
      p_all
        "d1<=d2<=leaf, d1=3: the leaf closes to 3, with or without the repeat, d1=3 early or late"
        [ (false, false); (true, false); (false, true); (true, true) ]
        ~f:(fun (repeat, early) -> leaf_is_3 (chain ~repeat ~early));
      p "and the repeat changes nothing in how it closes"
        (Poly.equal (chain ~repeat:false ~early:false) (chain ~repeat:true ~early:false)))

(* The contract over random dimension stores, the row sweep's twin. The store reads plain variables
   and closed sizes only, default-based or the broadcast top, related by bounds and equations, some
   variables leaves: no affine, concatenated or symbolic dimension, no fixed-index [At_least_dim]
   and no [Total_elems] (those have tests of their own). The stage-1 steps are solved one per call;
   the rest go into one stage-2 call, with and without a repeated bound, and with every stage-1 step
   re-solved there as the closure does it. Every solution reported is checked on ground
   dimensions. *)
type dim_step =
  | D_bound of int * int  (** [d_i ⊑ d_j] *)
  | D_alias of int * int  (** [d_i = d_j] *)
  | D_size of int * (int * string)  (** [d_i = size_basis] *)

let num_dims = 5

let random_dim_store state =
  let int n = Random.State.int state n in
  let pair () =
    let i = int num_dims in
    (i, (i + 1 + int (num_dims - 1)) % num_dims)
  in
  let sizes =
    [ (2, Row.default_basis); (3, Row.default_basis); (1, Row.default_basis); (1, Row.bcast_if_1) ]
  in
  let bounds =
    List.init
      (2 + int 5)
      ~f:(fun _ ->
        let i, j = pair () in
        D_bound (i, j))
  in
  let equations =
    List.init
      (1 + int 3)
      ~f:(fun _ ->
        match int 5 with
        | 0 | 1 ->
            let i, j = pair () in
            D_alias (i, j)
        | _ -> D_size (int num_dims, List.nth_exn sizes (int 4)))
  in
  let steps =
    List.map (bounds @ equations) ~f:(fun s -> (Random.State.bits state, s))
    |> List.sort ~compare:(fun (a, _) (b, _) -> Int.compare a b)
    |> List.map ~f:snd
  in
  let at_stage1 = int (List.length steps + 1) in
  ( List.take steps at_stage1,
    List.drop steps at_stage1,
    List.filter (List.range 0 num_dims) ~f:(fun _ -> int 2 = 0) )

let run_dims ?(late_extra = fun _ late -> late) (early, late, leaves) =
  let dims = Array.init num_dims ~f:(fun _ -> dim_var ()) in
  let constraint_ = function
    | D_bound (i, j) -> dim_ineq dims.(i) dims.(j)
    | D_alias (i, j) -> dim_eq dims.(i) dims.(j)
    | D_size (i, (d, basis)) -> dim_eq dims.(i) (Row.get_dim ~d ~basis ())
  in
  let init =
    List.map leaves ~f:(fun i -> Row.Terminal_dim (false, dims.(i), origin))
    @ List.map (Array.to_list dims) ~f:(fun d -> Row.Shape_row (dim_row d, origin))
  in
  let late = List.map ~f:constraint_ (late_extra early late) in
  match
    let pending, env =
      List.fold early ~init:(init, Row.empty_env) ~f:(fun (pending, env) s ->
          Row.solve_inequalities ~stage:Stage1 (constraint_ s :: pending) env)
    in
    let pending, env = Row.solve_inequalities ~stage:Stage2 (late @ pending) env in
    List.fold [ Row.Stage3; Stage4; Stage5; Stage6a; Stage6b; Stage7 ] ~init:(pending, env)
      ~f:(fun (pending, env) stage -> Row.solve_inequalities ~stage pending env)
  with
  | _, env -> Ok (Array.map dims ~f:(ground_dim env))
  | exception Row.Shape_error (msg, _) -> Error msg

(* The flat broadcast order on ground dimensions: equal in size and basis, or the operand the
   top. *)
let dim_holds dims step =
  let get i = dims.(i) in
  let top = (1, Row.bcast_if_1) in
  match step with
  | D_bound (i, j) -> (
      match (get i, get j) with Some r, Some o -> Poly.equal r o || Poly.equal o top | _ -> false)
  | D_alias (i, j) -> ( match (get i, get j) with Some a, Some b -> Poly.equal a b | _ -> false)
  | D_size (i, s) -> Option.equal Poly.equal (get i) (Some s)

let () =
  let state = Random.State.make [| 2141 |] in
  let stores = List.init num_stores ~f:(fun _ -> random_dim_store state) in
  let with_dup =
    List.filter_map stores ~f:(fun ((early, late, _) as store) ->
        let bounds = List.filter (early @ late) ~f:(function D_bound _ -> true | _ -> false) in
        if List.is_empty bounds then None
        else
          let bound = List.nth_exn bounds (Random.State.int state (List.length bounds)) in
          Some (store, (bound, Random.State.int state (List.length late + 1))))
  in
  let outcomes =
    List.map with_dup ~f:(fun (store, (bound, at)) ->
        let dup _ late = List.take late at @ (bound :: List.drop late at) in
        (* The closure's form: every stage-1 step re-solved in the stage-2 call. *)
        let refeed early late = early @ late in
        (store, run_dims store, run_dims ~late_extra:dup store, run_dims ~late_extra:refeed store))
  in
  let same a b =
    match (a, b) with Ok a, Ok b -> Poly.equal a b | Error _, Error _ -> true | _ -> false
  in
  let solved =
    List.filter_map outcomes ~f:(fun (s, r, _, _) -> Result.ok r |> Option.map ~f:(fun r -> (s, r)))
  in
  Stdio.eprintf
    "(not part of the golden) %d dimension stores with a bound to repeat, %d solved, %d changed by \
     the repeat, %d by re-solving stage 1\n"
    (List.length outcomes) (List.length solved)
    (List.count outcomes ~f:(fun (_, a, b, _) -> not (same a b)))
    (List.count outcomes ~f:(fun (_, a, _, c) -> not (same a c)));
  Verdict.case "random dimension stores, one stage-2 call" (fun () ->
      p "the dimension sweep solves stores and refuses others"
        (List.length solved > 0 && List.length solved < List.length outcomes);
      p_all ~min:500 "every dimension solution satisfies its store, on ground dimensions" solved
        ~f:(fun ((early, late, _), dims) -> List.for_all (early @ late) ~f:(dim_holds dims));
      (* What the guess may do: refuse a store, or close it otherwise, but every solution it gives
         still satisfies the store, and it solves nothing the single statement refused. *)
      let repeated =
        List.concat_map outcomes ~f:(fun ((early, late, _), _, b, c) ->
            List.filter_map [ b; c ] ~f:(fun r ->
                Result.ok r |> Option.map ~f:(fun dims -> (early @ late, dims))))
      in
      p_all ~min:500
        "every solution with a bound repeated satisfies its store, on ground dimensions" repeated
        ~f:(fun (steps, dims) -> List.for_all steps ~f:(dim_holds dims));
      p_none ~min:1000 "a repeated bound solves no store the single statement refused" outcomes
        ~f:(fun (_, a, b, c) -> Result.is_error a && (Result.is_ok b || Result.is_ok c)));
  (* Printed, not claimed: how many stores the repeat still decides (gh-ocannl-2141). *)
  Stdio.printf
    "the dimension sweep, stores changed by repeating a bound: %d; by re-solving stage 1 at stage \
     2: %d\n"
    (List.count outcomes ~f:(fun (_, a, b, _) -> not (same a b)))
    (List.count outcomes ~f:(fun (_, a, _, c) -> not (same a c)))

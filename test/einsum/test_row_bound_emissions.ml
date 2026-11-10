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

(* A guess waits for the call's fixpoint (gh-ocannl-2039). From Stage2 on, re-processing a recorded
   [r ⊑ o] whose rows are each the other's only bound equates them. Made mid-pass, that guess
   refused stores a solution existed for, by where the repeated pair sat in the call; it is now held
   to the fixpoint. Each case states a store a solution exists for, and claims it is solved with the
   rows that solution prescribes, or a store none exists for, and claims it is refused. *)

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
    List.fold [ Row.Stage3; Stage4; Stage5; Stage6; Stage7 ] ~init:(pending, env)
      ~f:(fun (pending, env) stage -> Row.solve_inequalities ~stage pending env)
  with
  | _, env -> Ok (List.map rows ~f:(ground env))
  | exception Row.Shape_error (msg, _) -> Error msg

let solved_to want = function
  | Ok got -> List.equal (Option.equal (List.equal Int.equal)) got (List.map want ~f:Option.some)
  | Error _ -> false

let refused = Result.is_error

(* The issue's store. Its stage-2 equations arrive either in ONE call, as the closing does it, or
   one call each, as the sweep above feeds them. The second form is outside what a closing stage
   assumes: a stage-2 call's fixpoint is where the held guess is made, so a call after it can only
   find [e = u] already chosen. *)
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
  Stdio.printf "the issue's store, one stage-2 call per equation: %s\n"
    (match issue_store ~one_call:false () with
    | Ok _ -> "solved"
    | Error msg -> "Shape_error: " ^ msg)

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

(* The dim twin in [solve_dim_ineq] keeps its singleton arm, guarded by [glb = None] and no
   [At_least_dim]: printed, not claimed, until it is decided on its own. [d1 = 3] and [d2] the
   broadcast top [1] satisfy [d1 <= d2]; whether it is refused depends on the repeat. *)
let () =
  let outcome ~repeat =
    let d1 = Row.Var (Row.get_var ()) and d2 = Row.Var (Row.get_var ()) in
    let dim_ineq res opnd = Row.Dim_ineq { res; opnd; from_ = Sexp.Atom "test"; origin } in
    let dim_eq d1 d2 = Row.Dim_eq { d1; d2; origin } in
    match
      let pending, env = Row.solve_inequalities ~stage:Stage1 [ dim_ineq d1 d2 ] Row.empty_env in
      Row.solve_inequalities ~stage:Stage2
        ((if repeat then [ dim_ineq d1 d2 ] else [])
        @ [ dim_eq d1 (dim 3); dim_eq d2 (Row.get_bcast_dim ~d:1 ()) ]
        @ pending)
        env
    with
    | _, env ->
        let show = function
          | Row.Var v -> Option.value_map (Row.get_dim_val env v) ~default:"?" ~f:Int.to_string
          | _ -> "?"
        in
        show d1 ^ " " ^ show d2
    | exception Row.Shape_error (msg, _) -> "Shape_error: " ^ msg
  in
  Stdio.printf "the dim twin: d1<=d2 recorded, then d1=3, d2=1 in one stage-2 call: %s\n"
    (outcome ~repeat:false);
  Stdio.printf "the dim twin, the bound repeated first: %s\n" (outcome ~repeat:true)

(* The contract over random stores: repeating a recorded bound anywhere in the stage-2 call refuses
   no store the call solves without it. The stage-1 steps are solved one per call as before; every
   later step goes into the one stage-2 call. What closes can still differ, and is counted on
   stderr: the guess is made only on a pair re-processed, so a repeat can be what makes it. *)
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
    List.fold [ Row.Stage3; Stage4; Stage5; Stage6; Stage7 ] ~init:(pending, env)
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
      p_none ~min:1000 "repeating a recorded bound in the stage-2 call refuses no store it solves"
        outcomes ~f:(function
        | Ok _, Error _ -> true
        | _ -> false))

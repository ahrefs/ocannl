(** gh-ocannl-2024: how far a row's GLB travels, pinned as policy. A row's GLB does not travel by
    itself: a solution with axes carries it two hops (gh-ocannl-1977), where a dimension's lower
    bound travels all the way. So a leaf three operations from a solved row closes empty, and a
    draw's sibling bounded only through a stage-6 row closes empty and repeats.

    Both ways of closing that gap, forwarding a GLB along every variable-to-variable bound and only
    into rows that resolve at their use sites, were built and compared under gh-ocannl-2024 and not
    adopted: neither changed an executed program in the suite
    (test/operations/row_glb_reach_executed pins the executed side), and both widen operation
    results above a leaf that takes the bound, against gh-544. Each store below prints its closed
    rows; the claims pin them. *)

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
let prov kind = Row.provenance ~sh_id:41 ~kind
let fresh ?(kind = `Output) () = Row.get_row_for_var (prov kind) (Row.get_row_var ())
let var_of (r : Row.t) = match r.bcast with Row_var v -> v | Broadcastable -> assert false

let closed dims =
  { Row.beg_dims = []; dims = List.map dims ~f:dim; bcast = Broadcastable; prov = prov `Output }

let ineq res opnd = Row.Row_ineq { res; opnd; origin }
let eq r1 r2 = Row.Row_eq { r1; r2; origin }
let all_stages = [ Row.Stage1; Stage2; Stage3; Stage4; Stage5; Stage6; Stage7 ]
let block = 4

(* A packed draw's round-up total over [rows], as [Uint4x32_to_prec] builds it. *)
let draw rows =
  let counter = Row.get_var () in
  ( counter,
    Row.Rows_constr
      {
        r = rows;
        constr =
          Total_elems
            {
              numerator =
                Strided_var
                  {
                    coeff = Utils.safe_lazy "test block" (fun () -> block);
                    var = counter;
                    denom = 1;
                    round_up = true;
                  };
              divided_by = [];
              keep_axis = false;
            };
        origin;
      } )

(* One stage-1 solve per step, the shape of per-registration solving, then every stage. [leaves] are
   terminal rows; every row of [rows] is a shape to close. *)
let solve ~leaves ~rows steps =
  let init =
    List.map leaves ~f:(fun r -> Row.Terminal_row (false, r, origin))
    @ List.map rows ~f:(fun r -> Row.Shape_row (r, origin))
  in
  let pending, env =
    List.fold steps ~init:(init, Row.empty_env) ~f:(fun (pending, env) step ->
        Row.solve_inequalities ~stage:Stage1 (step @ pending) env)
  in
  snd
    (List.fold all_stages ~init:(pending, env) ~f:(fun (c, env) stage ->
         Row.solve_inequalities ~stage c env))

let axes env (r : Row.t) =
  let r = Row.subst_row env r in
  let shown =
    List.map (r.beg_dims @ r.dims) ~f:(function
      | Row.Dim { d; _ } -> Int.to_string d
      | Var v -> Option.value_map (Row.get_dim_val env v) ~default:"?" ~f:Int.to_string
      | _ -> "?")
  in
  "[" ^ String.concat ~sep:"," shown ^ match r.bcast with Row_var _ -> ",..]" | _ -> "]"

(* [build] makes fresh rows (rank facts are global, so every store takes its own) and returns them
   with the store; the result is the closed rows, or the error. *)
let closes name build =
  let leaves, rows, steps, extra = build () in
  let outcome =
    match solve ~leaves ~rows steps with
    | env -> Ok (List.map rows ~f:(axes env) @ List.map extra ~f:(fun f -> f env))
    | exception Row.Shape_error (msg, _) -> Error msg
  in
  Stdio.printf "%s\n  %s\n" name
    (match outcome with
    | Ok shown -> String.concat ~sep:" " shown
    | Error msg -> "Shape_error: " ^ msg);
  fun want -> match outcome with Ok shown -> List.equal String.equal shown want | Error _ -> false

(* The issue's store: three hops from a solved row to a leaf. The same store past an alias is
   test_row_bound_emissions'. *)
let () =
  Verdict.case "a leaf three hops away" @@ fun () ->
  let shows =
    closes "u <= e <= o <= leaf, u := [3]  (rows u e o leaf)" (fun () ->
        let u = fresh () and e = fresh () and o = fresh () and l = fresh () in
        Row.add_resolve_at_use (var_of l);
        ([ l ], [ u; e; o; l ], [ [ ineq u e; ineq e o; ineq o l ]; [ eq u (closed [ 3 ]) ] ], []))
  in
  (* Taking the bound here would widen [e] and [o] with the leaf: an operand cannot have more axes
     than its result. When those rows are operation results, that is the widening gh-544 forbids. *)
  pf "a leaf three hops away closes empty" (shows [ "[3]"; "[]"; "[]"; "[]" ])

(* gh-ocannl-1961's chain: a draw's sibling row [a] bounded only by [w], a use-site row whose own
   bound closes it at stage 6. No executed program reaches it: whatever stands between a draw and a
   stage-6 row in a program is an operation result, which closes down first. *)
let () =
  Verdict.case "a draw's sibling bounded through a stage-6 row" @@ fun () ->
  let shows =
    closes "[5] <= w <= a, draw over [a; o]  (rows a o w; counter)" (fun () ->
        let a = fresh ~kind:`Batch () and ro = fresh () and w = fresh () in
        Row.add_resolve_at_use (var_of a);
        Row.add_resolve_at_use (var_of w);
        let counter, total = draw [ a; ro ] in
        ( [],
          [ a; ro; w ],
          [ [ ineq (closed [ 5 ]) w; ineq w a ]; [ total ] ],
          [
            (fun env ->
              Option.value_map (Row.get_dim_val env counter) ~default:"?" ~f:Int.to_string);
          ] ))
  in
  pf "the sibling closes empty, so the draw of one block repeats along w's 5"
    (shows [ "[]"; "[4]"; "[5]"; "1" ])

(* Two bounds meet in the leaf's GLB: incompatible ones join to the broadcast top. *)
let () =
  Verdict.case "a join" @@ fun () ->
  let shows =
    closes "u1 <= e1 <= o <= leaf, u2 <= e2 <= o, u1 := [3], u2 := [4]  (rows u1 u2 o leaf)"
      (fun () ->
        let u1 = fresh () and u2 = fresh () and e1 = fresh () and e2 = fresh () in
        let o = fresh () and l = fresh () in
        Row.add_resolve_at_use (var_of l);
        ( [ l ],
          [ u1; u2; o; l ],
          [
            [ ineq u1 e1; ineq e1 o; ineq u2 e2; ineq e2 o; ineq o l ];
            [ eq u1 (closed [ 3 ]) ];
            [ eq u2 (closed [ 4 ]) ];
          ],
          [] ))
  in
  pf "incompatible bounds join to the broadcast top" (shows [ "[3]"; "[4]"; "[1]"; "[1]" ])

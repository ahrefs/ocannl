(** gh-ocannl-2126: a row's lower bound travels the whole chain of bounds, as a dimension's does, so
    what a row closes to does not depend on how many operations stand between it and the solved row
    that bounds it. A row that resolves at its use sites closes to the bound; the operation results
    on the way take no use-site bound themselves and close after their arguments, to what those
    require (gh-544).

    While the bound reached two hops (gh-ocannl-2024), a leaf three operations from a solved row
    closed empty, and a draw's sibling bounded only through a stage-6 row closed empty and repeated.
    test/operations/row_glb_reach_executed pins the executed side. Each store below prints its
    closed rows; the claims pin them. *)

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
let all_stages = [ Row.Stage1; Stage2; Stage3; Stage4; Stage5; Stage6a; Stage6b; Stage7 ]
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

(* A chain of [n] results between a solved row and a leaf: [u <= e_1 <= ... <= e_n <= leaf]. *)
let chain n () =
  let u = fresh () and l = fresh () in
  let es = List.init n ~f:(fun _ -> fresh ()) in
  Row.add_resolve_at_use (var_of l);
  let rec bounds = function a :: (b :: _ as rest) -> ineq a b :: bounds rest | _ -> [] in
  ([ l ], (u :: es) @ [ l ], [ bounds ((u :: es) @ [ l ]); [ eq u (closed [ 3 ]) ] ], [])

(* gh-ocannl-2024's store, three hops from a solved row to a leaf, and the chain at other lengths.
   The same store past an alias is test_row_bound_emissions'. *)
let () =
  Verdict.case "a leaf at the end of a chain" @@ fun () ->
  let lengths = [ 1; 2; 3; 6 ] in
  let shown =
    List.map lengths ~f:(fun n ->
        let names = ("u" :: List.init n ~f:(fun i -> Printf.sprintf "e%d" (i + 1))) @ [ "leaf" ] in
        closes
          (Printf.sprintf "%s, u := [3]  (rows %s)" (String.concat ~sep:" <= " names)
             (String.concat ~sep:" " names))
          (chain n))
  in
  (* The results between take the leaf's axes with it: a result has at least its operand's. *)
  p_all "at every length the leaf takes the bound, and the results between it"
    (List.zip_exn lengths shown) ~f:(fun (n, shows) ->
      shows (List.init (n + 2) ~f:(fun _ -> "[3]")))

(* gh-ocannl-1961's chain: a draw's sibling row [a] bounded only by [w], a use-site row whose own
   bound closes it at stage 6. *)
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
  pf "the sibling takes w's 5, and the draw sizes its blocks for it"
    (shows [ "[5]"; "[]"; "[5]"; "2" ])

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

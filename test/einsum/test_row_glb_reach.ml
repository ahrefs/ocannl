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

(* gh-ocannl-2180: a result with an axis of its own, a numeric literal's ([u + 1]), between a leaf
   and its use site. The leaf closing to its bound took that axis itself, which closed minimal as a
   result's does; it takes an axis of its own now, bounded by the result's, and the result's follows
   from it. With two such results the GLB join keeps one axis variable, and the other result stays
   related to the leaf through its held inequality: without it that result closed minimal under a
   leaf of 3. *)
let () =
  Verdict.case "a leaf under results with axes of their own" @@ fun () ->
  let with_axis () = { (fresh ()) with Row.dims = [ Row.Var (Row.get_var ()) ] } in
  let leaf_under results () =
    let l = fresh () in
    Row.add_resolve_at_use (var_of l);
    let rs = List.init results ~f:(fun _ -> with_axis ()) in
    ( [ l ],
      rs @ [ l ],
      [ List.map rs ~f:(fun r -> ineq r l); List.map rs ~f:(fun r -> ineq (closed [ 3 ]) r) ],
      [] )
  in
  let one = closes "[3] <= e <= leaf, e = ..,d  (rows e leaf)" (leaf_under 1) in
  let two =
    closes "[3] <= e1 <= leaf, [3] <= e2 <= leaf, e1 = ..,d1, e2 = ..,d2  (rows e1 e2 leaf)"
      (leaf_under 2)
  in
  pf "one result: the leaf takes the use site's 3, and the result with it" (one [ "[3]"; "[3]" ]);
  pf "two results: the leaf takes 3, and both results with it" (two [ "[3]"; "[3]"; "[3]" ])

(* gh-ocannl-2244: a packed draw under one result, below a result row with a concrete axis and an
   open tail ([relu u *. c[|5]]: the product's row is [5].<t>). The draw's GLB, [5].<t>, has no
   variable axis; the draw waited at stage 6a on its open tail, and the result between, closing
   empty at 6b, closed the draw empty. The twin repeats the result's bound on the draw, as both
   operands of [u *. u] give it. *)
let () =
  Verdict.case "a packed draw under a result, its GLB's axes concrete" @@ fun () ->
  let store ~repeats () =
    let y = { (fresh ()) with Row.dims = [ dim 5 ] } and r = fresh () and u = fresh () in
    Row.add_resolve_at_use (var_of u);
    let counter, total = draw [ u ] in
    ( [],
      [ u; r; y ],
      [ ineq y r :: List.init repeats ~f:(fun _ -> ineq r u); [ total ] ],
      [ (fun env -> Option.value_map (Row.get_dim_val env counter) ~default:"?" ~f:Int.to_string) ]
    )
  in
  let once = closes "[5].<t> <= r <= u, draw over [u]  (rows u r y; counter)" (store ~repeats:1) in
  let twice =
    closes "[5].<t> <= r <= u twice, draw over [u]  (rows u r y; counter)" (store ~repeats:2)
  in
  pf "the draw takes the use site's 5, the result with it, and the draw sizes its blocks"
    (once [ "[5]"; "[5]"; "[5]"; "2" ]);
  pf "the repeated bound closes the same" (twice [ "[5]"; "[5]"; "[5]"; "2" ])

(* gh-ocannl-2247: [v = value] makes [value]'s axes [v]'s own, so a row that resolves at its use
   sites passes its mark to the dimension variables of the value it is solved to. A draw read
   through an einsum spec ([u ++ "ab => ba"]) is solved to the spec's axes, minted for the
   operation's result: unmarked, they closed minimal and the draw repeated
   (test/operations/row_glb_reach_executed pins the executed side). Each store is one Stage1 solve;
   the claims read the marks. *)
let () =
  Verdict.case "a solved row passes its mark to the axes it absorbs" @@ fun () ->
  let var () = Row.get_var () in
  let marked () =
    let r = fresh () in
    Row.add_resolve_at_use (var_of r);
    r
  in
  let flat beg_dims dims = { Row.beg_dims; dims; bcast = Broadcastable; prov = prov `Output } in
  let stage1 eqs = ignore (Row.solve_inequalities ~stage:Stage1 eqs Row.empty_env : _ * _) in
  let is_marked = Row.is_resolve_at_use_dim in
  (* Both flanks of the absorbed value, and the equation's sides swapped. *)
  let a = var () and b = var () in
  stage1 [ eq (marked ()) (flat [ Var a ] [ Var b ]) ];
  pf "a marked row's absorbed axes take the mark, both flanks" (is_marked a && is_marked b);
  let a = var () and b = var () in
  stage1 [ eq (flat [ Var a ] [ Var b ]) (marked ()) ];
  pf "the same with the equation's sides swapped" (is_marked a && is_marked b);
  let a = var () and b = var () in
  stage1 [ eq (fresh ()) (flat [ Var a ] [ Var b ]) ];
  pf "an unmarked row passes nothing" (not (is_marked a || is_marked b));
  (* [[a].<v> = [a; c]]: the flank solves [a := c] first, so the value [a] is [c] when stored. *)
  let a = var () and c = var () in
  stage1 [ eq { (marked ()) with dims = [ Var a ] } (flat [] [ Var a; Var c ]) ];
  pf "an axis the flank solved passes the mark to what it was solved to" (is_marked c);
  (* An open remainder shares the mark, and passes it on when it is solved in turn. *)
  let d = var () and e = var () and w = fresh () in
  stage1 [ eq (marked ()) { w with dims = [ Var d ] }; eq w (flat [] [ Var e ]) ];
  pf "the remainder takes the mark and passes it to its own axes"
    (Row.is_resolve_at_use (var_of w) && is_marked d && is_marked e);
  (* Composite axes: a stride's and a concatenation's variables. *)
  let o = var () and p = var () and q = var () in
  stage1
    [
      eq (marked ())
        (flat []
           [
             Affine { stride = 2; over = Var o; conv = None; stride_offset = 0 };
             Concat [ Var p; Var q ];
             dim 3;
           ]);
    ];
  pf "the variables of composite axes take the mark" (is_marked o && is_marked p && is_marked q);
  (* [[x].<v> = <v2>.[y]], balanced flanks: Stage1 defers it, solving neither row. *)
  let x = var () and y = var () in
  stage1 [ eq { (marked ()) with beg_dims = [ Var x ] } { (fresh ()) with dims = [ Var y ] } ];
  pf "a deferred equation marks no axis" (not (is_marked x || is_marked y));
  (* The marks are module-level inference state, which a rewind puts back. *)
  let a = var () in
  let saved = Row.capture_globals () in
  stage1 [ eq (marked ()) (flat [] [ Var a ]) ];
  let held = is_marked a in
  Row.restore_globals saved;
  pf "a rewind takes the mark back" (held && not (is_marked a))

(* What the mark does: a use site's 5 bounds the absorbed axis. Marked, the row closes to it;
   unmarked, it closes minimal, as an operation result's axis does. *)
let () =
  Verdict.case "an absorbed axis closes by the row's role" @@ fun () ->
  let store ~mark () =
    let u = fresh () and d = Row.Var (Row.get_var ()) in
    if mark then Row.add_resolve_at_use (var_of u);
    let spec = { Row.beg_dims = []; dims = [ d ]; bcast = Broadcastable; prov = prov `Output } in
    ([], [ u ], [ [ eq u spec; ineq (closed [ 5 ]) spec ] ], [])
  in
  let marked = closes "u := [d], [5] <= [d], u marked  (rows u)" (store ~mark:true) in
  let unmarked = closes "u := [d], [5] <= [d], u unmarked  (rows u)" (store ~mark:false) in
  pf "a marked row's absorbed axis closes to its use site" (marked [ "[5]" ]);
  pf "an unmarked row's closes minimal" (unmarked [ "[1]" ])

(* gh-ocannl-1961: where [Row.solve_inequalities] used to close an unresolved row by guess, driven
   stage by stage so each case can assert the state it enters and tell a correct deferral from a
   dropped constraint. A packed draw's round-up total is the constraint under test throughout:
   [draw] builds it as [Uint4x32_to_prec] does. Row variables are global, and rank facts with them,
   so every case takes fresh ones. The executed counterparts are in
   test/operations/row_closure_guess.ml. *)

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
let block = 4
let next_sh = ref 0

let prov kind =
  Int.incr next_sh;
  Row.provenance ~sh_id:!next_sh ~kind

let open_row ?(kind = `Output) () =
  let v = Row.get_row_var () in
  (v, Row.get_row_for_var (prov kind) v)

let closed dims : Row.t = { beg_dims = []; dims; bcast = Broadcastable; prov = prov `Output }

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

let shape_row r = Row.Shape_row (r, origin)
let ineq ~res ~opnd = Row.Row_ineq { res; opnd; origin }
let stages_to_6 : Row.stage list = [ Stage1; Stage2; Stage3; Stage4; Stage5; Stage6a; Stage6b ]

let run stages (ineqs, env) =
  List.fold stages ~init:(ineqs, env) ~f:(fun (ineqs, env) stage ->
      Row.solve_inequalities ~stage ineqs env)

let solve_all ineqs = run (stages_to_6 @ [ Stage7 ]) (ineqs, Row.empty_env)

(* The axes of a row with every variable solved; [None] while any is open. *)
let axes env r =
  let ({ beg_dims; dims; bcast; _ } : Row.t) = Row.subst_row env r in
  let dim_val : Row.dim -> int option = function
    | Dim { d; _ } -> Some d
    | Var v -> Row.get_dim_val env v
    | _ -> None
  in
  match bcast with
  | Row_var _ -> None
  | Broadcastable -> Option.all (List.map (beg_dims @ dims) ~f:dim_val)

let has_axes env r want = Option.equal (List.equal Int.equal) (axes env r) (Some want)
let is_open env r = Option.is_none (axes env r)

let holds_total env r ~counter ~elems =
  has_axes env r elems
  && Option.equal Int.equal (Row.get_dim_val env counter)
       (Some ((List.fold elems ~init:1 ~f:( * ) + block - 1) / block))

let refused_with substring f =
  match f () with
  | _ -> false
  | exception Row.Shape_error (msg, _) -> String.is_substring msg ~substring

let refused f = refused_with "" f

let bounds_state env v =
  match Row.find_row_entry env v with
  | Some (Bounds_row { res; glb; is_in_param; _ }) -> Some (res, glb, is_in_param)
  | _ -> None

(* Both orders: a bound's axes must not be decided by which constraint the solver met first. *)
let in_both_orders name f =
  Verdict.case (name ^ ", rows first") (fun () -> f ~order:"rows first" ~arrange:Fn.id);
  Verdict.case (name ^ ", bound first") (fun () -> f ~order:"bound first" ~arrange:List.rev)

(* (a) A bound closing only at the final stage: a fixed index's lower bound. The closed axis's
   product is what used to decide the outcome, as divisible totals took the guess. *)
let () =
  List.iter
    [ ("divisible", 4); ("not divisible", 3) ]
    ~f:(fun (kind, known) ->
      in_both_orders ("stage-7 bound, " ^ kind) (fun ~order ~arrange ->
          let v, rv = open_row () in
          let d = Row.get_var () in
          let drawn = { rv with dims = [ dim known ] } in
          let bound = closed [ Var d; dim known ] in
          let counter, total = draw [ drawn ] in
          let ineqs, env =
            run stages_to_6
              ( Row.Dim_constr { d = Var d; constr = At_least_dim { d_min = 3; sites = [] }; origin }
                :: ineq ~res:bound ~opnd:drawn
                :: arrange [ total; shape_row drawn; shape_row bound ],
                Row.empty_env )
          in
          pf "stage-7 bound, %s, %s: stage 6 leaves the row open" kind order
            (is_open env drawn
            && Option.is_none (Row.get_dim_val env counter)
            && Option.is_some (bounds_state env v));
          let left, env = Row.solve_inequalities ~stage:Stage7 ineqs env in
          pf "stage-7 bound, %s, %s: the row derives from it" kind order
            (holds_total env drawn ~counter ~elems:[ 3; known ]);
          p_empty
            (Printf.sprintf "stage-7 bound, %s, %s: nothing is left over" kind order)
            ~over:ineqs left))

(* (a) An open tail in the bound is as unresolved as a variable in a flank. *)
let () =
  in_both_orders "open tail" (fun ~order ~arrange ->
      let v, rv = open_row () in
      let w, rw = open_row () in
      Row.add_resolve_at_use v;
      Row.add_resolve_at_use w;
      let counter, total = draw [ rv ] in
      let _left, env =
        solve_all
          (ineq ~res:(closed [ dim 5 ]) ~opnd:rw
          :: ineq ~res:{ rw with dims = [ dim 3 ] } ~opnd:rv
          :: arrange [ total; shape_row rv; shape_row rw ])
      in
      pf "open tail, %s: the draw fills the bound's tail too" order
        (holds_total env rv ~counter ~elems:[ 5; 3 ]))

(* (b) A sibling with a use site still open stays open until the size arrives: a [Num_elems] total
   gives it at stage 3, or at stage 5 when another row of its own keeps it from closing earlier. *)
let leaf r = Row.Terminal_row (false, r, origin)

let sized_late rows =
  Row.Rows_constr
    {
      r = rows;
      constr = Total_elems { numerator = Num_elems 5; divided_by = []; keep_axis = false };
      origin;
    }

let () =
  in_both_orders "sibling at stage 3" (fun ~order ~arrange ->
      let a, ra = open_row ~kind:`Batch () in
      let _o, ro = open_row () in
      let _w, rw = open_row () in
      Row.add_resolve_at_use a;
      let counter, total = draw [ ra; ro ] in
      let ineqs, env =
        run [ Stage1; Stage2 ]
          ( ineq ~res:rw ~opnd:ra
            :: arrange [ total; sized_late [ rw ]; leaf rw; shape_row ra; shape_row ro ],
            Row.empty_env )
      in
      pf "sibling, %s: enters stage 3 bounded through a var-var chain only" order
        (match bounds_state env a with Some (_ :: _, None, false) -> true | _ -> false);
      let _left, env = run (List.drop stages_to_6 2 @ [ Stage7 ]) (ineqs, env) in
      pf "sibling, %s: the size arriving at stage 3 reaches it" order (has_axes env ra [ 5 ]);
      pf "sibling, %s: the total counts it" order
        (Option.equal Int.equal (Row.get_dim_val env counter) (Some ((5 + block - 1) / block))))

let () =
  in_both_orders "sibling at stage 5" (fun ~order ~arrange ->
      let a, ra = open_row ~kind:`Batch () in
      let _o, ro = open_row () in
      let _b, rb = open_row ~kind:`Batch () in
      let _w, rw = open_row () in
      Row.add_resolve_at_use a;
      let _counter, total = draw [ ra; ro ] in
      (* The bound on [rb] keeps that row open to stage 5, so [rw] is sized only then. *)
      let _left, env =
        solve_all
          (ineq ~res:rw ~opnd:ra
          :: ineq ~res:(closed [ dim 1 ]) ~opnd:rb
          :: arrange [ total; sized_late [ rb; rw ]; shape_row ra; shape_row ro ])
      in
      pf "sibling, %s: the size arriving at stage 5 reaches it" order (has_axes env ra [ 5 ]))

(* (b) A GLB is empty only when both flanks are. *)
let () =
  Verdict.case "leading axes" @@ fun () ->
  let a, ra = open_row ~kind:`Batch () in
  let _o, ro = open_row () in
  Row.add_resolve_at_use a;
  let _counter, total = draw [ ra; ro ] in
  let _left, env =
    solve_all
      [
        ineq ~res:{ (closed []) with beg_dims = [ dim 5 ] } ~opnd:ra;
        total;
        shape_row ra;
        shape_row ro;
      ]
  in
  p "sibling: a GLB with leading axes only is not empty" (has_axes env ra [ 5 ])

(* (c) A GLB past the rank facts' reach: [v] is bounded by [w], so it cannot be [w] with axes around
   it. Reads as absent on both of [process_shape_row]'s routes, and is no error. *)
let refuted_pair () =
  let v, rv = open_row () in
  let w, rw = open_row () in
  ignore (w : Row.row_var);
  (v, rv, rw, [ ineq ~res:rw ~opnd:rv ])

let () =
  Verdict.case "refuted, no constraint" @@ fun () ->
  let v, rv, rw, bounded = refuted_pair () in
  Row.add_resolve_at_use v;
  let _left, env =
    solve_all
      (bounded
      @ [ ineq ~res:{ rw with dims = [ dim 2; dim 3 ] } ~opnd:rv; shape_row rv; shape_row rw ])
  in
  p "refuted GLB, no constraint: closes minimally" (has_axes env rv [] && has_axes env rw [])

let () =
  Verdict.case "refuted, a total" @@ fun () ->
  let _v, rv, rw, bounded = refuted_pair () in
  let divisor = Row.get_var () in
  let _left, env =
    solve_all
      (bounded
      @ [
          ineq ~res:{ rw with dims = [ dim 2 ] } ~opnd:rv;
          Row.Rows_constr
            {
              r = [ rv ];
              constr =
                Total_elems { numerator = Num_elems 6; divided_by = [ divisor ]; keep_axis = false };
              origin;
            };
          shape_row rv;
          shape_row rw;
        ])
  in
  p "refuted GLB, a total: the total closes it instead"
    (is_open env rw |> not && Option.equal Int.equal (Row.get_dim_val env divisor) (Some 6))

(* What a guess is not: each of these closes as it did. *)
let () =
  Verdict.case "no bounds" @@ fun () ->
  let _v, rv = open_row () in
  let counter, total = draw [ rv ] in
  let _left, env = solve_all [ total; shape_row rv ] in
  p "control, no bounds: one block" (holds_total env rv ~counter ~elems:[ block ])

let () =
  Verdict.case "resolved bound" @@ fun () ->
  let _v, rv = open_row () in
  let counter, total = draw [ rv ] in
  let _left, env =
    solve_all [ ineq ~res:(closed [ dim 3; dim 4 ]) ~opnd:rv; total; shape_row rv ]
  in
  p "control, resolved bound: fills it" (holds_total env rv ~counter ~elems:[ 3; 4 ])

let () =
  Verdict.case "empty bound" @@ fun () ->
  let _v, rv = open_row () in
  let counter, total = draw [ rv ] in
  let _left, env = solve_all [ ineq ~res:(closed []) ~opnd:rv; total; shape_row rv ] in
  p "control, empty bound: one element" (holds_total env rv ~counter ~elems:[])

let () =
  in_both_orders "control, chain" (fun ~order ~arrange ->
      let a, ra = open_row ~kind:`Batch () in
      let _o, ro = open_row () in
      let _w, rw = open_row () in
      Row.add_resolve_at_use a;
      let counter, total = draw [ ra; ro ] in
      let _left, env =
        solve_all
          (ineq ~res:rw ~opnd:ra :: arrange [ total; shape_row ra; shape_row ro; shape_row rw ])
      in
      pf "control, a chain ending unbounded, %s: the sibling closes empty, one block" order
        (has_axes env ra [] && holds_total env ro ~counter ~elems:[ block ]))

let () =
  Verdict.case "unmarked result" @@ fun () ->
  let _v, rv = open_row () in
  let bound = closed [ dim 3 ] in
  let _left, env = solve_all [ ineq ~res:bound ~opnd:rv; shape_row rv ] in
  p "control, an unmarked operation result closes down" (has_axes env rv [])

let () =
  Verdict.case "rank cycle" @@ fun () ->
  let _v, rv = open_row () in
  let _w, rw = open_row () in
  p "control, a genuine rank cycle is still refused"
    (refused (fun () ->
         solve_all
           [
             ineq ~res:rw ~opnd:{ rv with dims = [ dim 2 ] };
             ineq ~res:rv ~opnd:{ rw with dims = [ dim 3 ] };
           ]))

(* The refusals: no derivation exists. A symbolic extent sizes no total (gh-ocannl-817). *)
let () =
  Verdict.case "symbolic bound" @@ fun () ->
  let sym, _ = Ir.Indexing.get_static_symbol ~static_range:6 Ir.Indexing.Empty in
  let _v, rv = open_row () in
  let _counter, total = draw [ { rv with dims = [ dim 4 ] } ] in
  let bound = closed [ Row.get_sym_dim sym; dim 4 ] in
  p "a symbolic bound refuses the total, naming why"
    (refused_with "not supported with symbolic dimensions" (fun () ->
         solve_all
           [
             ineq ~res:bound ~opnd:{ rv with dims = [ dim 4 ] };
             total;
             shape_row { rv with dims = [ dim 4 ] };
           ]))

(* Two draws each bounded by the other's counter: neither closes, and the leftovers say so rather
   than reading as solved. [Shape.refuse_unsolved] turns them into the error. *)
let () =
  Verdict.case "mutual draws" @@ fun () ->
  let _v1, r1 = open_row () in
  let _v2, r2 = open_row () in
  let c1, total1 = draw [ r1 ] in
  let c2, total2 = draw [ r2 ] in
  let left, _env =
    solve_all
      [
        ineq ~res:(closed [ Var c2 ]) ~opnd:r1;
        ineq ~res:(closed [ Var c1 ]) ~opnd:r2;
        total1;
        total2;
        shape_row r1;
        shape_row r2;
      ]
  in
  p "mutually bounded draws are left unsolved, not closed" (not (List.is_empty left))

(* gh-ocannl-2025: a fixed total (a reshape's [Num_elems]) closes its non-output rows empty from
   stage 3, on the reading that the output row takes the axes. A kept row bounded by a closed use
   site cannot take more than that site holds, so a smaller bound leaves the rest to the one
   sibling; closing it empty refused the store. Draws keep their own gate, above. *)
let fixed_total n rows =
  Row.Rows_constr
    {
      r = rows;
      constr = Total_elems { numerator = Num_elems n; divided_by = []; keep_axis = true };
      origin;
    }

let fixed_pair ~n ~bound =
  let _b, rb = open_row ~kind:`Batch () in
  let o, ro = open_row () in
  Row.add_resolve_at_use o;
  ( rb,
    ro,
    [ ineq ~res:(closed (List.map bound ~f:dim)) ~opnd:ro; fixed_total n [ rb; ro ] ]
    @ [ shape_row rb; shape_row ro ] )

let () =
  in_both_orders "fixed total, bounded output" (fun ~order ~arrange ->
      let rb, ro, ineqs = fixed_pair ~n:12 ~bound:[ 4 ] in
      let _left, env = solve_all (arrange ineqs) in
      pf "fixed total, %s: the output takes its bound, the sibling the rest" order
        (has_axes env ro [ 4 ] && has_axes env rb [ 3 ]))

let () =
  in_both_orders "fixed total, empty bound" (fun ~order ~arrange ->
      let rb, ro, ineqs = fixed_pair ~n:12 ~bound:[] in
      let _left, env = solve_all (arrange ineqs) in
      pf "fixed total, %s: an output bounded empty leaves the sibling everything" order
        (has_axes env ro [] && has_axes env rb [ 12 ]))

(* What is not short stays as it was: the output takes the total, the sibling closes empty. *)
let () =
  in_both_orders "fixed total, bound covering it" (fun ~order ~arrange ->
      let rb, ro, ineqs = fixed_pair ~n:12 ~bound:[ 12 ] in
      let _left, env = solve_all (arrange ineqs) in
      pf "control, fixed total, %s: a bound holding the total, the sibling closes empty" order
        (has_axes env ro [ 12 ] && has_axes env rb []));
  in_both_orders "fixed total, unbounded" (fun ~order ~arrange ->
      let _b, rb = open_row ~kind:`Batch () in
      let o, ro = open_row () in
      Row.add_resolve_at_use o;
      let _left, env =
        solve_all (arrange [ fixed_total 12 [ rb; ro ]; shape_row rb; shape_row ro ])
      in
      pf "control, fixed total, %s: no bound, the output takes it all" order
        (has_axes env ro [ 12 ] && has_axes env rb []))

(* (c) gh-ocannl-2035: what a refuted GLB still says. Its trailing flank bounds [v]'s last axes at
   any rank, so a row that must take axes takes them from it: a draw fills the flank, a fixed total
   takes it when it holds the total. One block of 4 against [..w.., 2, 2] was refused. *)
let refuted_by_flank flank extra =
  let v, rv, rw, bounded = refuted_pair () in
  Row.add_resolve_at_use v;
  let constrs, counter = extra rv in
  let _left, env =
    solve_all
      (bounded
      @ (ineq ~res:{ rw with dims = List.map flank ~f:dim } ~opnd:rv :: constrs)
      @ [ shape_row rv; shape_row rw ])
  in
  (env, rv, rw, counter)

let fixed n rv =
  ( [
      Row.Rows_constr
        {
          r = [ rv ];
          constr = Total_elems { numerator = Num_elems n; divided_by = []; keep_axis = false };
          origin;
        };
    ],
    None )

let () =
  List.iter
    [ [ 2; 2 ]; [ 3 ]; [ 2; 3 ] ]
    ~f:(fun flank ->
      let name = String.concat ~sep:"," (List.map flank ~f:Int.to_string) in
      Verdict.case ("refuted, a draw, flank " ^ name) @@ fun () ->
      let env, rv, rw, counter =
        refuted_by_flank flank (fun rv ->
            let counter, total = draw [ rv ] in
            ([ total ], Some counter))
      in
      pf "refuted GLB, a draw, flank %s: fills the flank" name
        (holds_total env rv ~counter:(Option.value_exn counter) ~elems:flank
        && has_axes env rw flank))

let () =
  Verdict.case "refuted, a fixed total its flank holds" @@ fun () ->
  let env, rv, _rw, _ = refuted_by_flank [ 2; 2 ] (fixed 4) in
  p "refuted GLB, a fixed total its flank holds: takes the flank" (has_axes env rv [ 2; 2 ])

(* No row whose last axes broadcast into [2, 2] and the rest into [w] holds 6: this refusal is
   right. *)
let () =
  Verdict.case "refuted, a fixed total no flank-bounded row holds" @@ fun () ->
  p "refuted GLB, a fixed total no row within the flank holds: refused"
    (refused (fun () -> refuted_by_flank [ 2; 2 ] (fixed 6)))

(** gh-ocannl-2024: how far a row's GLB travels, compared across [Row.row_glb_transport].

    [Two_hops] is the solver's behaviour: a row's GLB does not travel by itself, and a solution with
    axes carries it two hops (gh-ocannl-1977). [Use_site] forwards a GLB along variable-to-variable
    bounds into rows that resolve at their use sites (and parameters), and lets such a row take a
    closed use site's bound arriving at stage 6. [Transitive] forwards it everywhere, as a
    dimension's travels. Each store prints its closed rows under the three, so the golden is the
    comparison; the claims pin what [Two_hops] does today and what each alternative changes. *)

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

let transports : (string * Row.row_glb_transport) list =
  [ ("two hops", Two_hops); ("use site", Use_site); ("transitive", Transitive) ]

(* [build] makes fresh rows (rank facts are global, so every run takes its own) and returns them
   with the store; the result is each transport's closed rows, or the error. *)
let compare_store name build =
  Stdio.printf "%s\n" name;
  List.map transports ~f:(fun (tname, t) ->
      let leaves, rows, steps, extra = build () in
      let outcome =
        match Row.with_row_glb_transport t (fun () -> solve ~leaves ~rows steps) with
        | env -> Ok (List.map rows ~f:(axes env) @ List.map extra ~f:(fun f -> f env))
        | exception Row.Shape_error (msg, _) -> Error msg
      in
      Stdio.printf "  %-10s %s\n" tname
        (match outcome with
        | Ok shown -> String.concat ~sep:" " shown
        | Error msg -> "Shape_error: " ^ msg);
      (tname, outcome))

let shows outcomes tname want =
  match List.Assoc.find_exn outcomes tname ~equal:String.equal with
  | Ok shown -> List.equal String.equal shown want
  | Error _ -> false

(* The issue's store: three hops from a solved row to a leaf. *)
let () =
  Verdict.case "a leaf three hops away" @@ fun () ->
  let o =
    compare_store "u <= e <= o <= leaf, u := [3]  (rows u e o leaf)" (fun () ->
        let u = fresh () and e = fresh () and o = fresh () and l = fresh () in
        Row.add_resolve_at_use (var_of l);
        ([ l ], [ u; e; o; l ], [ [ ineq u e; ineq e o; ineq o l ]; [ eq u (closed [ 3 ]) ] ], []))
  in
  pf "two hops: the leaf closes empty" (shows o "two hops" [ "[3]"; "[]"; "[]"; "[]" ]);
  (* Once the leaf has an axis, every result between it and [u] must too: an operand cannot have
     more axes than its result. So the use-site reach widens operation results too. *)
  pf "use site: the leaf takes the bound, and the results between it widen with it"
    (shows o "use site" [ "[3]"; "[3]"; "[3]"; "[3]" ]);
  pf "transitive: the same" (shows o "transitive" [ "[3]"; "[3]"; "[3]"; "[3]" ])

let () =
  Verdict.case "past an alias" @@ fun () ->
  let o =
    compare_store "v <= e <= o <= leaf, v := u, u := [3]  (rows u e o leaf)" (fun () ->
        let v = fresh () and u = fresh () and e = fresh () and o = fresh () and l = fresh () in
        Row.add_resolve_at_use (var_of l);
        ( [ l ],
          [ u; e; o; l ],
          [ [ ineq v e; ineq e o; ineq o l ]; [ eq v u ]; [ eq u (closed [ 3 ]) ] ],
          [] ))
  in
  pf "past an alias, two hops: the leaf closes empty"
    (shows o "two hops" [ "[3]"; "[]"; "[]"; "[]" ]);
  pf "past an alias, use site: the leaf takes the bound"
    (shows o "use site" [ "[3]"; "[3]"; "[3]"; "[3]" ])

(* gh-ocannl-1961's chain: a draw's sibling row [a] bounded only by [w], a use-site row whose own
   bound closes it at stage 6. *)
let () =
  Verdict.case "a draw's sibling bounded through a stage-6 row" @@ fun () ->
  let o =
    compare_store "[5] <= w <= a, draw over [a; o]  (rows a o w; counter)" (fun () ->
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
  pf "two hops: the sibling closes empty, so the draw of one block repeats along w's 5"
    (shows o "two hops" [ "[]"; "[4]"; "[5]"; "1" ]);
  pf "use site: the sibling takes w's axis" (shows o "use site" [ "[5]"; "[]"; "[5]"; "2" ]);
  pf "transitive: the same" (shows o "transitive" [ "[5]"; "[]"; "[5]"; "2" ])

(* Two bounds meet in the leaf's GLB: incompatible ones join to the broadcast top. *)
let () =
  Verdict.case "a join" @@ fun () ->
  ignore
    (compare_store "u1 <= e1 <= o <= leaf, u2 <= e2 <= o, u1 := [3], u2 := [4]  (rows u1 u2 o leaf)"
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
      : (string * (string list, string) Result.t) list);
  p "a join, printed" true

(* The sweep of test_row_bound_emissions, run under each transport. A store [Two_hops] solves has a
   ground solution that test checks, so a store another transport refuses is a false refusal. *)
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
    |> fun (pending, env) ->
    List.fold all_stages ~init:(pending, env) ~f:(fun (c, env) stage ->
        Row.solve_inequalities ~stage c env)
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

let show_step = function
  | Bound (i, j), stage ->
      Printf.sprintf "r%d <= r%d%s" i j (if Poly.equal stage Row.Stage2 then " @2" else "")
  | Alias (i, j), stage ->
      Printf.sprintf "r%d = r%d%s" i j (if Poly.equal stage Row.Stage2 then " @2" else "")
  | Axes (i, ds), stage ->
      Printf.sprintf "r%d = [%s]%s" i
        (String.concat ~sep:"," (List.map ds ~f:Int.to_string))
        (if Poly.equal stage Row.Stage2 then " @2" else "")
  | Open (i, j, d), stage ->
      Printf.sprintf "r%d = <r%d>,%d%s" i j d (if Poly.equal stage Row.Stage2 then " @2" else "")

let () =
  let state = Random.State.make [| 1977 |] in
  let stores = List.init num_stores ~f:(fun _ -> random_store state) in
  let base =
    List.map stores ~f:(fun s -> (s, Row.with_row_glb_transport Two_hops (fun () -> run s)))
  in
  List.iter
    [ ("use site", Row.Use_site); ("transitive", Row.Transitive) ]
    ~f:(fun (name, t) ->
      let outcomes =
        List.map base ~f:(fun (s, b) -> (s, b, Row.with_row_glb_transport t (fun () -> run s)))
      in
      let count f = List.count outcomes ~f in
      let newly_refused =
        List.filter outcomes ~f:(fun (_, b, n) -> Result.is_ok b && Result.is_error n)
      in
      Stdio.printf
        "sweep, %s against two hops: %d solved, %d close differently, %d newly solved, %d newly \
         refused\n"
        name
        (count (fun (_, _, n) -> Result.is_ok n))
        (count (fun (_, b, n) ->
             match (b, n) with Ok a, Ok c -> not (Poly.equal a c) | _ -> false))
        (count (fun (_, b, n) -> Result.is_error b && Result.is_ok n))
        (List.length newly_refused);
      let by_size =
        List.sort newly_refused ~compare:(fun (a, _, _) (b, _, _) ->
            Int.compare (List.length a.steps) (List.length b.steps))
      in
      List.iteri by_size ~f:(fun k (s, _, n) ->
          if k < 2 then
            Stdio.printf "  refused: leaves %s; %s -- %s\n"
              (String.concat ~sep:"," (List.map s.leaves ~f:(Printf.sprintf "r%d")))
              (String.concat ~sep:"; " (List.map s.steps ~f:show_step))
              (match n with Error m -> m | Ok _ -> ""));
      Verdict.case ("sweep, " ^ name) (fun () ->
          p_all
            (Printf.sprintf "sweep, %s: every solution satisfies its store" name)
            (List.filter_map outcomes ~f:(fun (s, _, n) ->
                 Result.ok n |> Option.map ~f:(fun r -> (s, r))))
            ~f:(fun (store, rows) -> List.for_all store.steps ~f:(holds rows))))

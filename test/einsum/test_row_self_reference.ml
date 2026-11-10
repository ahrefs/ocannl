(** Regression test for rotational (shifted-flank) row constraints — same row variable on both
    sides, and the closed-closed base case (the gh-ocannl-247 family).

    With equal total flank lengths but shifted splits, the residue rotates through the shared
    variable's value (the word equation x ++ t = s ++ x for equality; pointwise chains through x for
    inequality). These cases used to be silently dropped: the unsatisfiable [3].<v> = <v>.[5] and
    [3].<v> <= <v>.[5] passed without a single dimension check, and so did the closed-closed
    [3].<closed> <= <closed>.[5] at equal ranks (where broadcast inserts no padding, so the
    operand's explicit dim must pin the result's position).

    The resolution is deferral into the closing policy: the constraint stays in flight; if the
    variable is solved by other constraints the substituted closed-closed check is exact; otherwise
    stage 6/7 closes the variable upward — the least-material disjunct — and the constraint reduces
    to comparing the surplus words directly (equal for =; pointwise <= for inequality, where an
    operand-side claim-free dim can still absorb). Closed-closed inequalities compare the operand's
    explicit material against the result's flat axis list from the outer edges; only
    broadcast-inserted middle positions are unconstrained.

    Each check drives the constraint through Stage1 (where rotational cases defer) and then Stage7
    (where surviving row variables are closed upward and the residue is checked). *)

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

let closed_row ~sh_id ~beg_dims ~dims =
  { Row.beg_dims; dims; bcast = Broadcastable; prov = Row.provenance ~sh_id ~kind:`Output }

let check name constr =
  match
    let leftover, env = Row.solve_inequalities ~stage:Stage1 [ constr ] Row.empty_env in
    Row.solve_inequalities ~stage:Stage7 leftover env
  with
  | _ -> Stdio.printf "%s: no error\n" name
  | exception Row.Shape_error (msg, _) -> Stdio.printf "%s: Shape_error: %s\n" name msg

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:1 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:2 ~kind:`Output) rho in
  check "eq, shifted flanks, conflicting dims (unsatisfiable)"
    (Row.Row_eq { r1 = { r1 with beg_dims = [ dim 3 ] }; r2 = { r2 with dims = [ dim 5 ] }; origin })

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:3 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:4 ~kind:`Output) rho in
  (* Satisfiable; resolved least-materially at closing (v |-> [], surplus words equal). *)
  check "eq, shifted flanks, conjugate dims (least-material resolution)"
    (Row.Row_eq { r1 = { r1 with beg_dims = [ dim 3 ] }; r2 = { r2 with dims = [ dim 3 ] }; origin })

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:5 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:6 ~kind:`Output) rho in
  check "eq, aligned flanks, conflicting dims"
    (Row.Row_eq { r1 = { r1 with dims = [ dim 3 ] }; r2 = { r2 with dims = [ dim 5 ] }; origin })

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:7 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:8 ~kind:`Output) rho in
  check "eq, aligned flanks, matching dims (legitimate no-op)"
    (Row.Row_eq { r1 = { r1 with dims = [ dim 3 ] }; r2 = { r2 with dims = [ dim 3 ] }; origin })

let () =
  check "ineq, closed-closed shifted, conflicting dims (unsatisfiable)"
    (Row.Row_ineq
       {
         res = closed_row ~sh_id:9 ~beg_dims:[ dim 3 ] ~dims:[];
         opnd = closed_row ~sh_id:10 ~beg_dims:[] ~dims:[ dim 5 ];
         origin;
       })

let () =
  check "ineq, closed-closed shifted, matching dims (satisfiable)"
    (Row.Row_ineq
       {
         res = closed_row ~sh_id:11 ~beg_dims:[ dim 3 ] ~dims:[];
         opnd = closed_row ~sh_id:12 ~beg_dims:[] ~dims:[ dim 3 ];
         origin;
       })

let () =
  (* The result's leading axis faces broadcast-inserted padding (rank 2 vs 1): unconstrained. *)
  check "ineq, closed-closed, result surplus faces inserted padding (satisfiable)"
    (Row.Row_ineq
       {
         res = closed_row ~sh_id:13 ~beg_dims:[ dim 3 ] ~dims:[ dim 5 ];
         opnd = closed_row ~sh_id:14 ~beg_dims:[] ~dims:[ dim 5 ];
         origin;
       })

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:15 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:16 ~kind:`Output) rho in
  check "ineq, same-var shifted, conflicting dims (unsatisfiable)"
    (Row.Row_ineq
       { res = { r1 with beg_dims = [ dim 3 ] }; opnd = { r2 with dims = [ dim 5 ] }; origin })

let () =
  let rho = Row.get_row_var () in
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id:17 ~kind:`Output) rho in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:18 ~kind:`Output) rho in
  check "ineq, same-var shifted, matching dims (satisfiable)"
    (Row.Row_ineq
       { res = { r1 with beg_dims = [ dim 3 ] }; opnd = { r2 with dims = [ dim 3 ] }; origin })

(* Two-variable cross-surplus equalities (the split-surplus case of the formal core's Def. 4.3(c)):
   one side's leading flank and the other side's trailing flank are both in surplus. The flat
   residue is the two-variable word equation s.x1 = x2.t, whose solutions are the principal family
   (x2 = s.w, x1 = w.t) PLUS sporadic cross-overlap solutions (x2 a proper prefix of s). Binding
   with a fresh variable would commit the family only — exact under the former marked semantics but
   losing the sporadic solutions under the adopted flat-equivalence semantics. The implementation
   instead DEFERS (unify_row's beg_handled=false branch re-emits the residual equation): exact once
   a variable is pinned by other constraints, and the upward close at stage 6 picks the
   least-material disjunct, which here IS the sporadic solution.

   We pin the orientation [5].<rho1> ~ <rho2>.[5,3] (family: x2 = [5].w, x1 = w.[5,3]; sporadic: x2
   = [], x1 = [3]). All three cases succeed: the deferral finds the sporadic solution whether rho2 ~
   [] arrives before or after the cross equality. The mirror orientation <rho1>.[5] ~ [3,5].<rho2>
   used to pin the OTHER conservative deviation: once rho2 is pinned empty, the closed side's
   material sits entirely in beg_dims, and the former asymmetric trailing guard (open trailing flank
   longer than the closed side's structural TRAILING flank only) rejected this flat-satisfiable
   store. Closed rows are now guarded by total axis count (structural flanks of a closed row are
   equivalent under flat semantics), so the store is accepted. *)

let check_list name constrs =
  match
    let leftover, env = Row.solve_inequalities ~stage:Stage1 constrs Row.empty_env in
    Row.solve_inequalities ~stage:Stage7 leftover env
  with
  | _ -> Stdio.printf "%s: no error\n" name
  | exception Row.Shape_error (msg, _) -> Stdio.printf "%s: Shape_error: %s\n" name msg

let cross_surplus_pair ~sh_id rho1 rho2 =
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id ~kind:`Output) rho1 in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:(sh_id + 1) ~kind:`Output) rho2 in
  Row.Row_eq
    { r1 = { r1 with beg_dims = [ dim 5 ] }; r2 = { r2 with dims = [ dim 5; dim 3 ] }; origin }

let pin_empty ~sh_id rho =
  let r = Row.get_row_for_var (Row.provenance ~sh_id ~kind:`Output) rho in
  Row.Row_eq { r1 = r; r2 = closed_row ~sh_id:(sh_id + 1) ~beg_dims:[] ~dims:[]; origin }

let () =
  let rho1 = Row.get_row_var () and rho2 = Row.get_row_var () in
  check_list "eq, cross surpluses alone (satisfiable, family binding)"
    [ cross_surplus_pair ~sh_id:19 rho1 rho2 ]

let () =
  let rho1 = Row.get_row_var () and rho2 = Row.get_row_var () in
  check_list "eq, cross surpluses then rho2 ~ [] (satisfiable only sporadically)"
    [ cross_surplus_pair ~sh_id:21 rho1 rho2; pin_empty ~sh_id:23 rho2 ]

let () =
  let rho1 = Row.get_row_var () and rho2 = Row.get_row_var () in
  check_list "eq, rho2 ~ [] then cross surpluses (same store, pinned first)"
    [ pin_empty ~sh_id:25 rho2; cross_surplus_pair ~sh_id:27 rho1 rho2 ]

let mirror_cross_pair ~sh_id rho1 rho2 =
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id ~kind:`Output) rho1 in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:(sh_id + 1) ~kind:`Output) rho2 in
  Row.Row_eq
    { r1 = { r1 with dims = [ dim 5 ] }; r2 = { r2 with beg_dims = [ dim 3; dim 5 ] }; origin }

let () =
  let rho1 = Row.get_row_var () and rho2 = Row.get_row_var () in
  check_list "eq, mirror cross surpluses then rho2 ~ [] (flat-satisfiable, accepted)"
    [ mirror_cross_pair ~sh_id:29 rho1 rho2; pin_empty ~sh_id:31 rho2 ]

(* Balanced flanks with conflicting rigid dims: [3].<rho1> ~ <rho2>.[5]. The empty close (k = 0) is
   refuted (3 <> 5), but the overlap-shifted principal family is satisfiable: rho2 = [3].<m>, rho1 =
   <m>.[5] (least member rho2 = [3], rho1 = [5]). The stage-6/7 close must pick the least
   non-refuted overlap rather than unconditionally closing rho2 to empty. *)
let balanced_conflict_pair ~sh_id rho1 rho2 =
  let r1 = Row.get_row_for_var (Row.provenance ~sh_id ~kind:`Output) rho1 in
  let r2 = Row.get_row_for_var (Row.provenance ~sh_id:(sh_id + 1) ~kind:`Output) rho2 in
  Row.Row_eq { r1 = { r1 with beg_dims = [ dim 3 ] }; r2 = { r2 with dims = [ dim 5 ] }; origin }

let () =
  let rho1 = Row.get_row_var () and rho2 = Row.get_row_var () in
  check_list "eq, balanced conflicting flanks (family solution, accepted)"
    [ balanced_conflict_pair ~sh_id:33 rho1 rho2 ]

(* gh-ocannl-1955: an equation waiting for the rank solves neither of its variables, so what is
   bounded through one of them must not meet the other side's remainder as that variable's value.
   Here rho1 closes to [] with rho = [], while rho >= rho0 >= rho1 read through rho1 = <rho2>.[4; 4]
   would demand two axes of an empty row. The empty row arrives in a later solve, as the store
   settles the bounds first. *)
let () =
  let row sh_id = Row.get_row_for_var (Row.provenance ~sh_id ~kind:`Output) (Row.get_row_var ()) in
  let r = row 35 and r0 = row 36 and r1 = row 37 and r2 = row 38 in
  let name = "eq waiting for the rank, then bounds through its variable (satisfiable)" in
  match
    let leftover, env =
      Row.solve_inequalities ~stage:Stage1
        [
          Row.Row_ineq { res = r; opnd = r0; origin };
          Row.Row_ineq { res = r0; opnd = r1; origin };
          Row.Row_eq
            {
              r1 = { r1 with beg_dims = [ dim 4; dim 4 ] };
              r2 = { r2 with dims = [ dim 4; dim 4 ] };
              origin;
            };
        ]
        Row.empty_env
    in
    let closed = Row.Row_eq { r1 = r; r2 = closed_row ~sh_id:39 ~beg_dims:[] ~dims:[]; origin } in
    let leftover, env = Row.solve_inequalities ~stage:Stage1 (closed :: leftover) env in
    Row.solve_inequalities ~stage:Stage7 leftover env
  with
  | _ -> Stdio.printf "%s: no error\n" name
  | exception Row.Shape_error (msg, _) -> Stdio.printf "%s: Shape_error: %s\n" name msg

(* gh-ocannl-1963: [flanked] rows and a walk through every stage, as the driver takes them, for what
   a waiting mixed-flank equation owes its bounds and what its closing commit may refuse. *)
let fresh_rows =
  let sh_id = ref 40 in
  fun () ->
    Int.incr sh_id;
    Row.get_row_for_var (Row.provenance ~sh_id:!sh_id ~kind:`Output) (Row.get_row_var ())

let flanked (r : Row.t) ~beg ~dims = { r with beg_dims = beg; dims }
let closed dims = closed_row ~sh_id:40 ~beg_dims:[] ~dims:(List.map dims ~f:dim)
let ineq res opnd = Row.Row_ineq { res; opnd; origin }
let eq r1 r2 = Row.Row_eq { r1; r2; origin }

let through_stages constraints env =
  List.fold [ Row.Stage1; Stage2; Stage3; Stage4; Stage5; Stage6; Stage7 ] ~init:(constraints, env)
    ~f:(fun (constraints, env) stage -> Row.solve_inequalities ~stage constraints env)

let show_axes env (r : Row.t) =
  let r = Row.subst_row env r in
  List.map (r.beg_dims @ r.dims) ~f:(function Row.Dim { d; _ } -> Int.to_string d | _ -> "?")
  |> String.concat ~sep:","

let decide name ~shows solve =
  match solve () with
  | env ->
      Stdio.printf "%s: no error: %s\n" name
        (String.concat ~sep:" " (List.map shows ~f:(fun r -> "[" ^ show_axes env r ^ "]")))
  | exception Row.Shape_error (msg, _) -> Stdio.printf "%s: Shape_error: %s\n" name msg

(* Solving nothing keeps every bound on the variable, so the equation's eventual solution still
   answers to them: [..] <= rho1, then [5].<rho1> = <rho2>.[5] waits, and rho2 = [5; 6] gives rho1 =
   [6; 5], which [7; 7] does not refine. *)
let waiting_eq rho1 rho2 =
  eq (flanked rho1 ~beg:[ dim 5 ] ~dims:[]) (flanked rho2 ~beg:[] ~dims:[ dim 5 ])

let () =
  let rho1 = fresh_rows () and rho2 = fresh_rows () in
  decide "bound on the waiting variable, its solution refutes it (unsatisfiable)"
    ~shows:[ rho1; rho2 ] (fun () ->
      let leftover, env =
        Row.solve_inequalities ~stage:Stage1
          [ ineq (closed [ 7; 7 ]) rho1; waiting_eq rho1 rho2 ]
          Row.empty_env
      in
      snd (through_stages (eq rho2 (closed [ 5; 6 ]) :: leftover) env))

let () =
  let r = fresh_rows () and r0 = fresh_rows () and rho1 = fresh_rows () and rho2 = fresh_rows () in
  decide "chain to the waiting variable, its solution refutes it (unsatisfiable)"
    ~shows:[ r; r0; rho1; rho2 ] (fun () ->
      let leftover, env =
        Row.solve_inequalities ~stage:Stage1
          [ ineq r r0; ineq r0 rho1; waiting_eq rho1 rho2 ]
          Row.empty_env
      in
      let leftover, env =
        Row.solve_inequalities ~stage:Stage1 (eq r (closed [ 7; 7 ]) :: leftover) env
      in
      snd (through_stages (eq rho2 (closed [ 5; 6 ]) :: leftover) env))

(* The closing commit must not outrun a bound that fixes the rank: read alone, [5].<rho1> =
   <rho2>.[7] closes at overlap 1, rho2 = [5], rho1 = [7]; with rho1 <= [6; 7] only the principal
   family is left. *)
let () =
  let rho1 = fresh_rows () and rho2 = fresh_rows () in
  decide "bound fixes the rank of the waiting variable (satisfiable)" ~shows:[ rho1; rho2 ]
    (fun () ->
      snd
        (through_stages
           [
             ineq rho1 (closed [ 6; 7 ]);
             eq (flanked rho1 ~beg:[ dim 5 ] ~dims:[]) (flanked rho2 ~beg:[] ~dims:[ dim 7 ]);
           ]
           Row.empty_env))

(* The closing commit skips an overlap on what [unify_dim] would refuse, no less: each store below
   has its solution at overlap 1, and overlap 0 must read as refuted. *)
let balanced_close name ~beg ~dims:d =
  let rho1 = fresh_rows () and rho2 = fresh_rows () in
  decide name ~shows:[ rho1; rho2 ] (fun () ->
      snd
        (through_stages
           [ eq (flanked rho1 ~beg ~dims:[]) (flanked rho2 ~beg:[] ~dims:d) ]
           Row.empty_env))

let () =
  balanced_close "balanced flanks, 3 against 1 (satisfiable at overlap 1)"
    ~beg:[ dim 3 ]
    ~dims:[ dim 1 ];
  balanced_close "balanced flanks, equal sizes of different bases (satisfiable at overlap 1)"
    ~beg:[ Row.get_dim ~d:5 ~basis:"a" () ]
    ~dims:[ Row.get_dim ~d:5 ~basis:"b" () ];
  let sym, _ = Ir.Indexing.get_static_symbol ~static_range:5 Ir.Indexing.Empty in
  balanced_close "balanced flanks, symbol against size (satisfiable at overlap 1)"
    ~beg:[ Row.get_sym_dim sym ]
    ~dims:[ dim 5 ]

(* An equation first met at the closing stage conflicts on what its peeled flank has just solved: x
   := 3 from the outer pair, so [x].<rho1>.[x] = <rho2>.[9; 3] refutes overlap 0 and sets rho2 =
   [3]. *)
let () =
  let rho1 = fresh_rows () and rho2 = fresh_rows () in
  let x = Row.Var (Row.get_var ()) in
  decide "balanced flanks, variable solved by the peeled flank (satisfiable at overlap 1)"
    ~shows:[ rho1; rho2 ] (fun () ->
      snd
        (Row.solve_inequalities ~stage:Stage7
           [
             eq (flanked rho1 ~beg:[ x ] ~dims:[ x ]) (flanked rho2 ~beg:[] ~dims:[ dim 9; dim 3 ]);
           ]
           Row.empty_env))

(* gh-ocannl-2037: the closing commit refutes an overlap on its equations together, as the solver
   does, not pair by pair. Each store below has its least solution at overlap 1, but no single pair
   of its overlap 0 conflicts: two occurrences of one variable are forced apart. The rows shown are
   the whole flanked row, then rho1, then rho2. *)
let joint_close ?(flip = false) name ~beg ~dims:d =
  let rho1 = fresh_rows () and rho2 = fresh_rows () in
  let lhs = flanked rho1 ~beg ~dims:[] and rhs = flanked rho2 ~beg:[] ~dims:d in
  decide name ~shows:[ lhs; rho1; rho2 ] (fun () ->
      snd (through_stages [ (if flip then eq rhs lhs else eq lhs rhs) ] Row.empty_env))

let () =
  let x = Row.Var (Row.get_var ()) in
  joint_close "joint close, [x; x] against [3; 5] (satisfiable at overlap 1)" ~beg:[ x; x ]
    ~dims:[ dim 3; dim 5 ];
  let x = Row.Var (Row.get_var ()) in
  joint_close ~flip:true "joint close, flipped, [x; x] against [3; 5] (satisfiable at overlap 1)"
    ~beg:[ x; x ]
    ~dims:[ dim 3; dim 5 ];
  let x = Row.Var (Row.get_var ()) in
  joint_close "joint close, [x; x] against [3; 3] (satisfiable at overlap 0)" ~beg:[ x; x ]
    ~dims:[ dim 3; dim 3 ];
  let twice () =
    Row.Affine { stride = 2; over = Var (Row.get_var ()); conv = None; stride_offset = 0 }
  in
  joint_close "joint close, 2*x against 3 (satisfiable at overlap 1)"
    ~beg:[ twice () ]
    ~dims:[ dim 3 ];
  let x2 = twice () in
  joint_close "joint close, [2*x; 2*x] against [6; 10] (satisfiable at overlap 1)" ~beg:[ x2; x2 ]
    ~dims:[ dim 6; dim 10 ];
  let xc = Row.Concat [ Var (Row.get_var ()); dim 1 ] in
  joint_close "joint close, [x^1; x^1] against [3; 5] (satisfiable at overlap 1)" ~beg:[ xc; xc ]
    ~dims:[ dim 3; dim 5 ];
  (* The concatenation's equation decides nothing until the other pairs size its components; then it
     refutes overlap 0 (2 + 5 is not 4), and the commit of overlap 0 raises. *)
  let a = Row.Var (Row.get_var ()) and b = Row.Var (Row.get_var ()) in
  joint_close "joint close, deferred [a; b; a^b] against [2; 5; 4] (satisfiable at overlap 1)"
    ~beg:[ a; b; Row.Concat [ a; b ] ]
    ~dims:[ dim 2; dim 5; dim 4 ]

(* Below stage 6 a balanced equation still waits, whatever its overlaps. *)
let () =
  let rho1 = fresh_rows () and rho2 = fresh_rows () in
  let x = Row.Var (Row.get_var ()) in
  let leftover, _env =
    List.fold
      [ Row.Stage1; Stage2; Stage3; Stage4; Stage5 ]
      ~init:
        ( [ eq (flanked rho1 ~beg:[ x; x ] ~dims:[]) (flanked rho2 ~beg:[] ~dims:[ dim 3; dim 5 ]) ],
          Row.empty_env )
      ~f:(fun (constraints, env) stage -> Row.solve_inequalities ~stage constraints env)
  in
  Stdio.printf "joint close, before stage 6: %d constraint(s) waiting\n" (List.length leftover)

(* A refused overlap leaves no inference effect behind. At overlap 0 of [a; b; a].<rho1> =
   <rho2>.[b; 3; 5], a = b shares [a]'s use-site mark with [b] before a = 3 and a = 5 conflict; at
   overlap 1 only a = 3 is required, so [b] stays open and unmarked. *)
let () =
  let rho1 = fresh_rows () and rho2 = fresh_rows () in
  let av = Row.get_var () and bv = Row.get_var () in
  let a = Row.Var av and b = Row.Var bv in
  Row.add_resolve_at_use_dim av;
  decide "joint close, refused overlap rolled back (satisfiable at overlap 1)"
    ~shows:[ closed_row ~sh_id:40 ~beg_dims:[] ~dims:[ a; b ]; rho1; rho2 ]
    (fun () ->
      let env =
        snd
          (Row.solve_inequalities ~stage:Stage6
             [
               eq
                 (flanked rho1 ~beg:[ a; b; a ] ~dims:[])
                 (flanked rho2 ~beg:[] ~dims:[ b; dim 3; dim 5 ]);
             ]
             Row.empty_env)
      in
      Verdict.p "joint close, refused overlap rolled back: b stays unmarked"
        (not (Row.is_resolve_at_use_dim bv));
      env)

(* gh-ocannl-2036: the occurs check around one row variable, by which side has the surplus. A
   broadcast result may have more axes than its operand, so [<v>.[2] >= <v>] holds at [v = []]: it
   defers like the shifted case, and the substituted check decides it once [v] is known. The
   operand's surplus, [<v> >= <v>.[2]], and an equation with unequal totals, the issue's einsum [[h;
   w].<c> = <c>] (one tensor in two slots of a spec), hold for no finite [v] and stay refused. *)
let solved constraints =
  match through_stages constraints Row.empty_env with
  | _, env -> Ok env
  | exception Row.Shape_error (msg, _) -> Error msg

let self_reference = function
  | Error msg -> String.is_prefix msg ~prefix:"Infinite number of axes by self-reference"
  | Ok _ -> false

let refused_by_dims = function
  | Error msg -> String.is_prefix msg ~prefix:"dimension comparison for axis"
  | Ok _ -> false

let axes outcome (r : Row.t) =
  match outcome with Ok env -> Some (show_axes env r) | Error _ -> None

let () =
  let rho = fresh_rows () in
  let res = flanked rho ~beg:[] ~dims:[ dim 2 ] in
  let o = solved [ ineq res rho ] in
  Verdict.p "same variable, result trailing surplus: [2] >= [] at v = []"
    (Option.equal String.equal (axes o res) (Some "2")
    && Option.equal String.equal (axes o rho) (Some ""));
  let rho = fresh_rows () in
  let h = Row.Var (Row.get_var ()) and w = Row.Var (Row.get_var ()) in
  let res = flanked rho ~beg:[ h; w ] ~dims:[] in
  Verdict.p "same variable, result leading surplus (a pointwise result over [h; w].<c> >= <c>)"
    (Result.is_ok (solved [ ineq res rho ]));
  let rho = fresh_rows () in
  let res = flanked rho ~beg:[] ~dims:[ dim 2 ] in
  let o = solved [ eq rho (closed [ 2 ]); ineq res rho ] in
  Verdict.p "same variable, result surplus, v = [2] first: [2; 2] >= [2]"
    (Option.equal String.equal (axes o res) (Some "2,2"));
  let rho = fresh_rows () in
  let res = flanked rho ~beg:[] ~dims:[ dim 2 ] in
  Verdict.p "same variable, result surplus, v = [3] later: [3; 2] >= [3] refused by its dims"
    (refused_by_dims (solved [ ineq res rho; eq rho (closed [ 3 ]) ]));
  let rho = fresh_rows () in
  let res = flanked rho ~beg:[] ~dims:[ dim 2; dim 3 ]
  and opnd = flanked rho ~beg:[] ~dims:[ dim 5 ] in
  Verdict.p "same variable, result surplus, outer overlap 3 against 5 refused by its dims"
    (refused_by_dims (solved [ ineq res opnd ]));
  let rho = fresh_rows () in
  Verdict.p "same variable, operand surplus refused by self-reference"
    (self_reference (solved [ ineq rho (flanked rho ~beg:[] ~dims:[ dim 2 ]) ]));
  let rho = fresh_rows () in
  let h = Row.Var (Row.get_var ()) and w = Row.Var (Row.get_var ()) in
  Verdict.p "same variable, equation [h; w].<c> = <c> refused by self-reference"
    (self_reference (solved [ eq (flanked rho ~beg:[ h; w ] ~dims:[]) rho ]))

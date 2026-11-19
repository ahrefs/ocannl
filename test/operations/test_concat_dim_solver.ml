(** Solver-level regression tests for the Concat dim-solver hardening (task-887c4062).

    These drive [tensor/row.ml] directly because the targeted arms have no determinate high-level
    [%op] fixture that demonstrably routes through them (see
    [archive/docs/proposals/concat-dim-solver-hardening.md], the AC1 reachability note): a broadcast
    inequality between two [Concat] bounds only reaches [solve_dim_ineq]'s GLB merge when an operand
    dimension variable already carries a [Concat] greatest-lower-bound, and a variables-only
    [Concat = Concat] of unequal arity only reaches [unify_dim]'s generalized pairing arm. Building
    the constraints by hand and calling [solve_inequalities] pins the exact arm.

    Before this task both AC1 GLB arms ([Affine]/[Concat]) raised [assert false]; the AC3 pairing
    arm fired only at equal arity. The tests below would fault (Assert_failure) or fail to link on
    the pre-task code. *)

open! Base
open Ocannl

(* Failures go through [Verdict], so that a regression exits nonzero instead of being `dune
   promote`d into the golden as the expected output (gh-ocannl-601). *)
let fail fmt = Printf.ksprintf Verdict.fail fmt

let dummy_origin : Row.constraint_origin list =
  [
    {
      lhs_name = "test";
      lhs_kind = `Output;
      rhs_name = "test";
      rhs_kind = `Output;
      operation = None;
    };
  ]

let dim ?(basis = Row.default_basis) d = Row.get_dim ~d ~basis ()
let no_from = Sexp.List []

let ineq res v : Row.constraint_ =
  Row.Dim_ineq { res; opnd = Row.Var v; from_ = no_from; origin = dummy_origin }

let eq d1 d2 : Row.constraint_ = Row.Dim_eq { d1; d2; origin = dummy_origin }

(* Run a solve over [env], classifying the outcome so an [assert false] regression (any
   non-[Shape_error] exception) is distinguished from a legitimate [Shape_error] and from
   success. *)
let run_solve_env ~stage cs env =
  try `Ok (Row.solve_inequalities ~stage cs env) with
  | Row.Shape_error (m, _) -> `Shape m
  | e -> `Other (Exn.to_string e)

let run_solve ~stage cs = run_solve_env ~stage cs Row.empty_env
let has_dim_ineq cs = List.exists cs ~f:(function Row.Dim_ineq _ -> true | _ -> false)

(* [v] forced through a size-[n] GLB iff equating it to [n] is consistent but to [n+1] conflicts. *)
let forced_to_size ~env v n =
  (match run_solve_env ~stage:Stage4 [ eq (Row.Var v) (dim n) ] env with
    | `Ok _ -> true
    | _ -> false)
  &&
  match run_solve_env ~stage:Stage4 [ eq (Row.Var v) (dim (n + 1)) ] env with
  | `Shape _ -> true
  | _ -> false

(* A shared free component keeps a bound a [Concat]: substitution sums one of concrete components
   (gh-ocannl-2019), and a [Dim] bound never reaches the arms below. *)
let bound a parts = Row.Concat (Row.Var a :: List.map parts ~f:dim)

(* AC1 — GLB merge of two COMPATIBLE Concat bounds (stage 4) COMMITS: it forces the operand var
   through the merged GLB (and does not merely avoid raising). First inequality banks [a^2^3] (size
   a + 5) as the glb of a fresh operand var; the second feeds [a^1^4] (the same size) into the
   merge, which [unify_dim] reconciles, so at stage >= 4 the merge commits. Pre-task: [assert
   false].

   The PASS condition pins the recipe, not just "Ok": - no residual [Dim_ineq] survives — a Stage4
   RE-DEFER mutation would leave one (this is exactly the residual the postpone test below asserts
   IS present at Stage1); - with a = 1, the operand is forced to size 6 — equating v = 6 is
   consistent, v = 7 conflicts. A mutation that succeeds without forcing the operand through the GLB
   (drops the bound, or demotes to bcast-top) fails [forced_to_size] (under bcast-top v = 6 itself
   would conflict). *)
let test_ac1_glb_merge_compatible () =
  Stdio.printf "AC1: GLB merge of two compatible Concat bounds (stage 4) commits (forces operand)\n";
  let v = Row.get_var () and a = Row.get_var () in
  match run_solve ~stage:Stage4 [ ineq (bound a [ 2; 3 ]) v; ineq (bound a [ 1; 4 ]) v ] with
  | `Ok (remaining, env) ->
      let no_residual =
        (not (has_dim_ineq remaining)) && not (has_dim_ineq (Row.unsolved_constraints env))
      in
      let forced =
        match run_solve_env ~stage:Stage4 [ eq (Row.Var a) (dim 1) ] env with
        | `Ok (_, env) -> forced_to_size ~env v 6
        | _ -> false
      in
      if no_residual && forced then
        Stdio.printf "  PASS: operand committed to the merged GLB; no residual deferral\n"
      else fail "expected commit (no_residual=%b forced_to_5=%b)" no_residual forced
  | `Shape m -> fail "unexpected Shape_error: %s" m
  | `Other e -> fail "regression (expected no assert false): %s" e

(* AC1 — GLB merge of two INCOMPATIBLE Concat bounds (stage 4) DEMOTES the operand to the broadcast-
   top GLB (size 1), swallowing the failed equality rather than propagating a Shape_error or
   committing to either bound. [a^2^3] vs [a^2^4] differ by 1: [unify_dim] fails. Pre-task: [assert
   false].

   The PASS condition pins demotion specifically: the operand is solved to size 1 — the
   broadcast-top that is distinct from BOTH bounds. A mutation that commits to either bound, or
   re-defers (operand unsolved), or drops the GLB yields [get_dim_val <> Some 1] and flips the
   assertion; merely "not raising" is not enough. *)
let test_ac1_glb_merge_incompatible_demotes () =
  Stdio.printf "AC1: GLB merge of two incompatible Concat bounds (stage 4) demotes to bcast-top\n";
  let v = Row.get_var () and a = Row.get_var () in
  match run_solve ~stage:Stage4 [ ineq (bound a [ 2; 3 ]) v; ineq (bound a [ 2; 4 ]) v ] with
  | `Ok (_remaining, env) -> (
      match Row.get_dim_val env v with
      | Some 1 ->
          Stdio.printf
            "  PASS: operand demoted to broadcast-top (size 1), not committed to a bound\n"
      | other ->
          fail "expected demote to size 1, got %s"
            (match other with Some n -> Int.to_string n | None -> "<unsolved>"))
  | `Shape m -> fail "should demote, not raise Shape_error: %s" m
  | `Other e -> fail "regression (expected no assert false): %s" e

(* AC1 — below stage 4 the merge POSTPONES: it neither crashes, commits, nor demotes — it re-defers
   the inequality so the two bounds can still resolve equal later. PASS pins both halves: a
   [Dim_ineq] survives in the residual AND the operand is left UNSOLVED (a Stage1 demote/commit
   mutation would solve it — [get_dim_val] would be [Some _]). Pre-task: [assert false]. *)
let test_ac1_glb_merge_postpone_below_stage4 () =
  Stdio.printf "AC1: GLB merge below stage 4 postpones (defers, operand left unsolved)\n";
  let v = Row.get_var () and a = Row.get_var () in
  match run_solve ~stage:Stage1 [ ineq (bound a [ 2; 3 ]) v; ineq (bound a [ 1; 4 ]) v ] with
  | `Ok (remaining, env) ->
      let deferred = has_dim_ineq remaining in
      let not_committed = Option.is_none (Row.get_dim_val env v) in
      if deferred && not_committed then
        Stdio.printf "  PASS: inequality deferred and operand left unsolved for a later stage\n"
      else fail "expected postpone (deferred=%b operand_unsolved=%b)" deferred not_committed
  | `Shape m -> fail "unexpected Shape_error: %s" m
  | `Other e -> fail "regression (expected no assert false): %s" e

(* gh-ocannl-2368: a variables-only [Concat = Concat] at UNEQUAL arity (3 components vs 2) waits. [a
   + b + c = d + e] implies neither [a = d] nor [c = 0]: the stage-4+ pairing that equated the
   oldest variables (a = d, then b = e) left [c = 0], a size only a discardable component may take.
   The equation stays pending, in either orientation, and links nothing; once the other components
   are sized, arithmetic sizes the last one, positive, and an inconsistent size is still refused, so
   the equation was kept, not dropped. Closed by a shape row with every component registered as a
   concatenation component (closure guesses none of them, gh-ocannl-1656), it is refused rather than
   sized. *)
let test_unequal_arity_waits () =
  Stdio.printf "Unequal arity: variables-only [a; b; c] = [d; e] waits, in either orientation\n";
  let vars () = List.init 5 ~f:(fun _ -> Row.get_var ()) in
  let equation ~flip vs =
    match List.map vs ~f:(fun v -> Row.Var v) with
    | [ a; b; c; d; e ] ->
        let lhs = Row.Concat [ a; b; c ] and rhs = Row.Concat [ d; e ] in
        if flip then eq rhs lhs else eq lhs rhs
    | _ -> assert false
  in
  let size vs sizes =
    List.filter_map (List.zip_exn vs sizes) ~f:(fun (v, n) ->
        Option.map n ~f:(fun n -> eq (Row.Var v) (dim n)))
  in
  let show env v = match Row.get_dim_val env v with Some n -> Int.to_string n | None -> "?" in
  List.iter [ false; true ] ~f:(fun flip ->
      let side = if flip then "[d; e] = [a; b; c]" else "[a; b; c] = [d; e]" in
      let vs = vars () in
      (match run_solve ~stage:Stage4 [ equation ~flip vs ] with
      | `Ok (pending, env) ->
          let a = List.nth_exn vs 0 and c = List.nth_exn vs 2 and d = List.nth_exn vs 3 in
          let linked =
            match run_solve_env ~stage:Stage4 (pending @ [ eq (Row.Var a) (dim 7) ]) env with
            | `Ok (_, env) -> Option.is_some (Row.get_dim_val env d)
            | _ -> true
          in
          if List.length pending = 1 && Option.is_none (Row.get_dim_val env c) && not linked then
            Stdio.printf "  PASS: %s: one pending equation, c unsized, a := 7 leaves d unsized\n"
              side
          else
            fail "%s: %d pending, c=%s, linked=%b (expected 1, ?, false)" side (List.length pending)
              (show env c) linked
      | `Shape m -> fail "%s: unexpected Shape_error: %s" side m
      | `Other e -> fail "%s: regression: %s" side e);
      let vs = vars () in
      let e = List.nth_exn vs 4 in
      match run_solve ~stage:Stage4 [ equation ~flip vs ] with
      | `Ok (pending, env) -> (
          (match
             run_solve_env ~stage:Stage4
               (pending @ size vs [ Some 2; Some 3; Some 4; Some 1; None ])
               env
           with
          | `Ok ([], env) when Option.equal Int.equal (Row.get_dim_val env e) (Some 8) ->
              Stdio.printf "  PASS: %s: a, b, c, d = 2, 3, 4, 1 then sizes e = 8, nothing pending\n"
                side
          | `Ok (rest, env) ->
              fail "%s: e=%s with %d pending (expected 8, none)" side (show env e)
                (List.length rest)
          | `Shape m -> fail "%s: sizing refused: %s" side m
          | `Other m -> fail "%s: regression: %s" side m);
          match
            run_solve_env ~stage:Stage4
              (pending @ size vs [ Some 2; Some 3; Some 4; Some 1; Some 7 ])
              env
          with
          | `Shape _ -> Stdio.printf "  PASS: %s: e = 7 against the same sizes is refused\n" side
          | `Ok _ -> fail "%s: e = 7 accepted: the equation was dropped" side
          | `Other m -> fail "%s: regression: %s" side m)
      | `Shape m -> fail "%s: unexpected Shape_error: %s" side m
      | `Other e -> fail "%s: regression: %s" side e);
  let vs = vars () in
  List.iter vs ~f:Row.add_concat_component;
  let close : Row.constraint_ =
    Row.Shape_row
      ( {
          beg_dims = [];
          dims = List.map vs ~f:(fun v -> Row.Var v);
          bcast = Broadcastable;
          prov = Row.provenance ~sh_id:2368 ~kind:`Output;
        },
        dummy_origin )
  in
  let result =
    List.fold
      [
        (Row.Stage1, [ equation ~flip:false vs ]);
        (Stage4, [ close ]);
        (Stage5, []);
        (Stage6a, []);
        (Stage6b, []);
        (Stage7, []);
      ]
      ~init:(`Ok ([], Row.empty_env))
      ~f:(fun acc (stage, cs) ->
        match acc with
        | `Ok (pending, env) -> run_solve_env ~stage (pending @ cs) env
        | other -> other)
  in
  match result with
  | `Shape m -> Stdio.printf "  PASS: closed as concatenation components: refused: %s\n" m
  | `Ok (_, env) ->
      fail "closed as concatenation components: accepted, c = %s" (show env (List.nth_exn vs 2))
  | `Other e -> fail "closed as concatenation components: regression: %s" e

(* gh-ocannl-2368: a variables-only side against one carrying a solved [Dim] waits too. The stage-4+
   pairing equated [a = e] in [[a; b] = [4; e]] and so chose the split [b = 4]; nothing in the
   equation forces it. Once [a] and [e] are sized, arithmetic sizes [b]. *)
let test_solved_side_waits () =
  Stdio.printf "A solved side: [a; b] = [4; e] waits; a = 2, e = 3 then size b = 5\n";
  let a = Row.get_var () and b = Row.get_var () and e = Row.get_var () in
  let lhs = Row.Concat [ Row.Var a; Row.Var b ] in
  let rhs = Row.Concat [ dim 4; Row.Var e ] in
  match run_solve ~stage:Stage4 [ eq lhs rhs ] with
  | `Ok (pending, env) -> (
      if Option.is_some (Row.get_dim_val env b) || List.length pending <> 1 then
        fail "expected one pending equation and b unsized, got %d pending, b sized"
          (List.length pending)
      else Stdio.printf "  PASS: one pending equation, b unsized\n";
      match
        run_solve_env ~stage:Stage4
          (pending @ [ eq (Row.Var a) (dim 2); eq (Row.Var e) (dim 3) ])
          env
      with
      | `Ok ([], env) when Option.equal Int.equal (Row.get_dim_val env b) (Some 5) ->
          Stdio.printf "  PASS: b = 5 from the sum, nothing pending\n"
      | `Ok (_, env) ->
          fail "expected b = 5, got %s"
            (match Row.get_dim_val env b with Some n -> Int.to_string n | None -> "?")
      | `Shape m -> fail "sizing refused: %s" m
      | `Other m -> fail "regression: %s" m)
  | `Shape m -> fail "unexpected Shape_error: %s" m
  | `Other e -> fail "regression (expected no assert false): %s" e

(* Nested-[Concat] component (beyond the proposal, per user request): a component that is itself a
   [Concat] must be flattened (concatenation is associative — sum semantics) so the oldest-variable
   pairing aligns by flat position, not by nesting level. [Concat [Concat [a; b]; c]] = [Concat [d;
   e; f]] (all distinct vars) must link a=d, b=e, c=f. Without flattening the residual [Concat[a;b];
   c] is not all-`Var` (its first component is a nested `Concat`), so the pairing would mis-align —
   its only top-level var c would pair with the other side's oldest var d — giving wrong linkage (or
   a deferral). We force a:=2, b:=3, c:=4 and require d, e, f to read back 2, 3, 4. *)
let test_nested_concat_flattened () =
  Stdio.printf "Nested Concat: Concat[Concat[a;b];c] = Concat[d;e;f] flattens, links a=d,b=e,c=f\n";
  let a = Row.get_var () and b = Row.get_var () and c = Row.get_var () in
  let d = Row.get_var () and e = Row.get_var () and f = Row.get_var () in
  let lhs = Row.Concat [ Row.Concat [ Row.Var a; Row.Var b ]; Row.Var c ] in
  let rhs = Row.Concat [ Row.Var d; Row.Var e; Row.Var f ] in
  match
    try
      let _rem, env = Row.solve_inequalities ~stage:Stage4 [ eq lhs rhs ] Row.empty_env in
      let _rem, env =
        Row.solve_inequalities ~stage:Stage4
          [ eq (Row.Var a) (dim 2); eq (Row.Var b) (dim 3); eq (Row.Var c) (dim 4) ]
          env
      in
      `Ok (Row.get_dim_val env d, Row.get_dim_val env e, Row.get_dim_val env f)
    with
    | Row.Shape_error (m, _) -> `Shape m
    | ex -> `Other (Exn.to_string ex)
  with
  | `Ok (Some 2, Some 3, Some 4) ->
      Stdio.printf "  PASS: nested concat flattened; d=2 e=3 f=4 (a=d, b=e, c=f)\n"
  | `Ok (dv, ev, fv) ->
      let show = function Some n -> Int.to_string n | None -> "?" in
      fail "expected d,e,f = 2,3,4; got %s,%s,%s" (show dv) (show ev) (show fv)
  | `Shape m -> fail "unexpected Shape_error: %s" m
  | `Other e -> fail "regression: %s" e

(* Codex P2 regression: an all-`Var` residual on one side facing a residual that still holds a non-
   variable, non-solved component (an unresolved `Affine` from a strided axis) must NOT enter the
   pairing arm. [Concat [a; c]] = [Concat [Affine{2,w}; b]] is an underdetermined sum equality; the
   buggy "one side all-Var" guard equated the oldest variables (a=b) and forced the leftover (c) to
   the affine, over-constraining. The fix requires BOTH residuals to be variables-only, so this
   falls through to arithmetic deferral and leaves a and b INDEPENDENT. We detect the spurious a=b
   link: if it were present, forcing a:=5 then b:=7 would conflict; it must instead be
   consistent. *)
let test_ac3_affine_residual_not_overconstrained () =
  Stdio.printf "AC3: all-Var side vs Affine-bearing side — pairing must NOT over-constrain\n";
  let a = Row.get_var () and c = Row.get_var () and b = Row.get_var () and w = Row.get_var () in
  let aff = Row.Affine { stride = 2; over = Row.Var w; conv = None; stride_offset = 0 } in
  let lhs = Row.Concat [ Row.Var a; Row.Var c ] in
  let rhs = Row.Concat [ aff; Row.Var b ] in
  match run_solve ~stage:Stage4 [ eq lhs rhs ] with
  | `Ok (_remaining, env) -> (
      match run_solve_env ~stage:Stage4 [ eq (Row.Var a) (dim 5); eq (Row.Var b) (dim 7) ] env with
      | `Ok _ -> Stdio.printf "  PASS: a and b left independent (a:=5, b:=7 both consistent)\n"
      | `Shape m -> fail "over-constrained (spurious a=b link): %s" m
      | `Other e -> fail "unexpected exception: %s" e)
  | `Shape m -> fail "first solve raised Shape_error: %s" m
  | `Other e -> fail "regression: %s" e

(* gh-ocannl-2019: a [Concat] of concrete components, which substitution takes for its sum, against
   what the solver held it against as parts. A variable [w] stores one. *)
let test_concrete_concat () =
  Stdio.printf "Concrete Concat: its sum, against a stored Concat and a named basis\n";
  let c23 = Row.Concat [ dim 2; dim 3 ] in
  let solves name cs ~check =
    match run_solve ~stage:Stage4 cs with
    | `Ok (_, env) ->
        if check env then Stdio.printf "  PASS: %s\n" name else fail "%s: wrong solution" name
    | `Shape m -> fail "%s: unexpected Shape_error: %s" name m
    | `Other e -> fail "%s: regression: %s" name e
  in
  let refuses name cs ~needle =
    match run_solve ~stage:Stage4 cs with
    | `Shape m when String.is_substring m ~substring:needle ->
        Stdio.printf "  PASS: %s: refused: %s\n" name m
    | `Shape m -> fail "%s: refused otherwise: %s" name m
    | `Ok _ -> fail "%s: accepted" name
    | `Other e -> fail "%s: regression: %s" name e
  in
  (let w = Row.get_var () and a = Row.get_var () in
   solves "a stored 2^3 solves a^3"
     [ eq (Row.Var w) c23; eq (Row.Concat [ Row.Var a; dim 3 ]) (Row.Var w) ]
     ~check:(fun env ->
       Option.equal Int.equal (Row.get_dim_val env w) (Some 5)
       && Option.equal Int.equal (Row.get_dim_val env a) (Some 2)));
  (let w = Row.get_var () and a = Row.get_var () and b = Row.get_var () in
   match
     run_solve ~stage:Stage4
       [ eq (Row.Var w) c23; eq (Row.Concat [ Row.Var a; Row.Var b ]) (Row.Var w) ]
   with
   | `Ok (remaining, env) when not (List.is_empty remaining) ->
       if Option.is_none (Row.get_dim_val env a) && Option.is_none (Row.get_dim_val env b) then
         Stdio.printf "  PASS: a stored 2^3 leaves a^b to the closure's refusal\n"
       else fail "a^b guessed from a stored 2^3"
   | _ -> fail "a^b against a stored 2^3 did not stay deferred");
  solves "2^3 = 5" [ eq c23 (dim 5) ] ~check:(fun _ -> true);
  refuses "2^3 = 6" [ eq c23 (dim 6) ] ~needle:"mismatch";
  solves "a named component names the sum"
    [ eq (Row.Concat [ dim ~basis:"rgb" 2; dim 3 ]) (dim ~basis:"rgb" 5) ]
    ~check:(fun _ -> true);
  (* As a [Dim] of 5 is, and as this was before the sum: all-integer components claim [default], a
     short-term rule (gh-ocannl-2093). The named component above names the sum, and a variable one
     takes the named size's claim as its binding records it; a solved sum records no basis when it
     meets a solved size, so a claim-free [2^3] taking [rgb 5] could later take [hsv 5] too, merging
     the two silently. gh-ocannl-2095 is the sound alternative (integer components as slices of the
     partner's basis). *)
  refuses "a plain 2^3 against a named 5" [ eq c23 (dim ~basis:"rgb" 5) ] ~needle:"different bases";
  refuses "a named 1^3 against a plain 4"
    [ eq (Row.Concat [ dim ~basis:"rgb" 1; dim ~basis:"rgb" 3 ]) (dim 4) ]
    ~needle:"different bases"

(* gh-ocannl-2191: a component's basis is decided as the fully solved axis is checked, so each
   system below gets one verdict whichever of its equations is solved first. *)
let test_tagged_components_any_order () =
  Stdio.printf "Tagged components: one verdict in either order\n";
  (* A system's discardable variables, set by its [mk] (only a 0 needs one, gh-ocannl-2368). *)
  let discardable = ref Row.dim_var_set_empty in
  let solve_in_turn cs =
    List.fold cs
      ~init:(`Ok ([], Row.empty_env))
      ~f:(fun acc c ->
        match acc with
        | `Ok (deferred, env) -> (
            try
              `Ok
                (Row.solve_inequalities ~stage:Stage4 ~discardable_vars:!discardable
                   (deferred @ [ c ]) env)
            with
            | Row.Shape_error (m, _) -> `Shape m
            | e -> `Other (Exn.to_string e))
        | other -> other)
  in
  let value env v =
    match Row.get_dim_from_env env v with
    | Some (Row.Dim { d; basis; _ }) -> Printf.sprintf "%d of %s" d basis
    | _ -> "unsized"
  in
  let in_both_orders name (mk : unit -> Row.dim_var * Row.constraint_ * Row.constraint_) ~want =
    let verdict first =
      discardable := Row.dim_var_set_empty;
      let u, c1, c2 = mk () in
      match solve_in_turn (if first then [ c1; c2 ] else [ c2; c1 ]) with
      | `Ok (_, env) -> "u = " ^ value env u
      | `Shape _ -> "refused"
      | `Other e -> "regression: " ^ e
    in
    let v1 = verdict true and v2 = verdict false in
    if String.equal v1 want && String.equal v2 want then
      Stdio.printf "  PASS: %s: %s in either order\n" name want
    else fail "%s: %s, then %s, want %s" name v1 v2 want
  in
  let rgb = "rgb" in
  let concat_of a u = Row.Concat [ Row.Var a; Row.Var u ] in
  in_both_orders "[a; u] = 4, a = 1 rgb: a claim against a plain size"
    (fun () ->
      let a = Row.get_var () and u = Row.get_var () in
      (u, eq (concat_of a u) (dim 4), eq (Row.Var a) (dim ~basis:rgb 1)))
    ~want:"refused";
  in_both_orders "4 = [a; u], a = 1 rgb: the other orientation"
    (fun () ->
      let a = Row.get_var () and u = Row.get_var () in
      (u, eq (dim 4) (concat_of a u), eq (Row.Var a) (dim ~basis:rgb 1)))
    ~want:"refused";
  in_both_orders "[a; u] = 4 rgb, a = 1 rgb"
    (fun () ->
      let a = Row.get_var () and u = Row.get_var () in
      (u, eq (concat_of a u) (dim ~basis:rgb 4), eq (Row.Var a) (dim ~basis:rgb 1)))
    ~want:"u = 3 of rgb";
  in_both_orders "[a; u] = 4, a = 3 bcast_if_1: an atom above size 1"
    (fun () ->
      let a = Row.get_var () and u = Row.get_var () in
      (u, eq (concat_of a u) (dim 4), eq (Row.Var a) (dim ~basis:Row.bcast_if_1 3)))
    ~want:"refused";
  in_both_orders "[a; u] = 4, a = 1 bcast_if_1: the claim-free top"
    (fun () ->
      let a = Row.get_var () and u = Row.get_var () in
      (u, eq (concat_of a u) (dim 4), eq (Row.Var a) (dim ~basis:Row.bcast_if_1 1)))
    ~want:"u = 3 of default";
  (* A 0 binds a discardable component only (gh-ocannl-2368), as a block spec's would be. *)
  in_both_orders "[a; u] = 3 rgb, a = 0 bcast_if_1, a discardable: an empty segment claims nothing"
    (fun () ->
      let a = Row.get_var () and u = Row.get_var () in
      discardable := Set.add Row.dim_var_set_empty a;
      (u, eq (concat_of a u) (dim ~basis:rgb 3), eq (Row.Var a) (dim ~basis:Row.bcast_if_1 0)))
    ~want:"u = 3 of rgb";
  in_both_orders "[u; 2] = [k; 2 rgb], k = 3: a zero residual keeps its claim"
    (fun () ->
      let u = Row.get_var () and k = Row.get_var () in
      ( u,
        eq (Row.Concat [ Row.Var u; dim 2 ]) (Row.Concat [ Row.Var k; dim ~basis:rgb 2 ]),
        eq (Row.Var k) (dim 3) ))
    ~want:"u = 3 of rgb";
  in_both_orders "[u; 2 rgb] = [k; 2 rgb], k = 3 hsv: claimed components do not cancel"
    (fun () ->
      let u = Row.get_var () and k = Row.get_var () in
      ( u,
        eq (Row.Concat [ Row.Var u; dim ~basis:rgb 2 ]) (Row.Concat [ Row.Var k; dim ~basis:rgb 2 ]),
        eq (Row.Var k) (dim ~basis:"hsv" 3) ))
    ~want:"refused"

(* gh-ocannl-2316: concatenation arithmetic never sizes a component to 0 unless it is discardable,
   and a zero residual carries no claim, since substitution erases a 0-sized component. Each system
   is solved in every registration order, one constraint per Stage 1 pass, then closed by a shape
   row as a tensor's variables would be. On the base, [k = 2_default] sized [u] to [0_a], which [u =
   0_a] stated first refuses. *)
let test_zero_component () =
  Stdio.printf "Zero components: only a discardable one is sized 0, in every order\n";
  let close vs : Row.constraint_ =
    Row.Shape_row
      ( {
          beg_dims = [];
          dims = List.map vs ~f:(fun v -> Row.Var v);
          bcast = Broadcastable;
          prov = Row.provenance ~sh_id:2316 ~kind:`Output;
        },
        dummy_origin )
  in
  let rec permutations = function
    | [] -> [ [] ]
    | xs ->
        List.concat_mapi xs ~f:(fun i x ->
            let rest = List.filteri xs ~f:(fun j _ -> j <> i) in
            List.map (permutations rest) ~f:(fun perm -> x :: perm))
  in
  let value env v =
    match Row.get_dim_from_env env v with
    | Some (Row.Dim { d; basis; _ }) -> Printf.sprintf "%d of %s" d basis
    | _ -> "unsized"
  in
  (* [discardable] names which of [u] and a third variable [w] a block spec would have marked. A run
     is solved one constraint per Stage 1 pass in [order], or all in one ([batched]). Success
     requires nothing left pending (gh-ocannl-2368). *)
  let verdict ~discardable ~batched mk order =
    let u = Row.get_var () and k = Row.get_var () and w = Row.get_var () in
    let cs = mk u k w in
    let discardable_vars =
      match discardable with
      | `None -> Row.dim_var_set_empty
      | `U -> Set.add Row.dim_var_set_empty u
      | `W -> Set.add Row.dim_var_set_empty w
    in
    let registered = List.map order ~f:(fun i -> List.nth_exn cs i) in
    let stages =
      (if batched then [ (Row.Stage1, registered) ]
       else List.map registered ~f:(fun c -> (Row.Stage1, [ c ])))
      @ [ (Row.Stage2, []); (Stage3, []); (Stage4, [ close [ u; k; w ] ]) ]
      @ List.map [ Row.Stage5; Stage6a; Stage6b; Stage7 ] ~f:(fun s -> (s, []))
    in
    match
      List.fold stages ~init:([], Row.empty_env) ~f:(fun (deferred, env) (stage, cs) ->
          Row.solve_inequalities ~stage ~discardable_vars (deferred @ cs) env)
    with
    | [], env -> Printf.sprintf "u = %s, k = %s" (value env u) (value env k)
    | pending, _ -> Printf.sprintf "%d pending" (List.length pending)
    | exception Row.Shape_error _ -> "refused"
    | exception e -> "regression: " ^ Exn.to_string e
  in
  let in_every_order ?(discardable = `None) name ~n mk ~want =
    let orders = permutations (List.init n ~f:Fn.id) in
    let runs = List.cartesian_product [ false; true ] orders in
    Verdict.Claims.p_all
      ~min:(2 * List.length orders)
      (Printf.sprintf "  %s: %s in every order, and batched" name want)
      runs
      ~f:(fun (batched, order) ->
        let got = verdict ~discardable ~batched mk order in
        if not (String.equal got want) then
          Stdio.eprintf "  %s, %s %s: %s (not part of the golden)\n" name
            (if batched then "batched" else "order")
            (String.concat ~sep:"," (List.map order ~f:Int.to_string))
            got;
        String.equal got want)
  in
  let concat u k ~basis =
    eq (Row.Concat [ Row.Var u; dim 4 ]) (Row.Concat [ Row.Var k; dim ~basis 2 ])
  in
  List.iter [ "a"; "h" ] ~f:(fun t ->
      in_every_order
        ("[u; 4] = [k; 2 " ^ t ^ "], k = 2: the zero residual is not sized")
        ~n:2
        (fun u k _ -> [ concat u k ~basis:t; eq (Row.Var k) (dim 2) ])
        ~want:"refused";
      in_every_order
        ("[u; 4] = [k; 2 " ^ t ^ "], k = 2, u = 0 " ^ t ^ ": the answer the base gave, stated")
        ~n:3
        (fun u k _ ->
          [ concat u k ~basis:t; eq (Row.Var k) (dim 2); eq (Row.Var u) (dim ~basis:t 0) ])
        ~want:"refused";
      in_every_order
        ("[u; 4] = [k; 2 " ^ t ^ "], k = 4: a positive residual takes the claim")
        ~n:2
        (fun u k _ -> [ concat u k ~basis:t; eq (Row.Var k) (dim 4) ])
        ~want:("u = 2 of " ^ t ^ ", k = 4 of default");
      in_every_order ~discardable:`U
        ("[u; 4] = [k; 2 " ^ t ^ "], k = 2, u discardable: an empty segment cannot carry the claim")
        ~n:2
        (fun u k _ -> [ concat u k ~basis:t; eq (Row.Var k) (dim 2) ])
        ~want:"refused");
  in_every_order "[u; 4] = [k; 2], k = 2: no claim, still not sized" ~n:2
    (fun u k _ -> [ concat u k ~basis:Row.default_basis; eq (Row.Var k) (dim 2) ])
    ~want:"refused";
  in_every_order ~discardable:`U "[u; 4] = [k; 2], k = 2, u discardable: sized 0" ~n:2
    (fun u k _ -> [ concat u k ~basis:Row.default_basis; eq (Row.Var k) (dim 2) ])
    ~want:"u = 0 of default, k = 2 of default";
  (* The permission to be empty is the component's: an alias it was equated with may be sized 0 by
     the sum too, whichever of the two the substitution presents. *)
  in_every_order ~discardable:`U "[u; 4] = [k; 2], k = 2, u = w, u discardable: sized 0" ~n:3
    (fun u k w ->
      [ concat u k ~basis:Row.default_basis; eq (Row.Var k) (dim 2); eq (Row.Var u) (Row.Var w) ])
    ~want:"u = 0 of default, k = 2 of default";
  (* A zero residual decides its basis before it waits: two bounds [4] and [[u; 4_a]] on one operand
     conflict, so the GLB merge demotes to the broadcast top, as on the base, and [u = 1_a] stays
     consistent with them in either order. *)
  List.iter [ true; false ] ~f:(fun concat_first ->
      let x = Row.get_var () and u = Row.get_var () in
      let bounds = [ ineq (dim 4) x; ineq (Row.Concat [ Row.Var u; dim ~basis:"a" 4 ]) x ] in
      let bounds = if concat_first then List.rev bounds else bounds in
      Verdict.Claims.p
        (Printf.sprintf "  bounds 4 and [u; 4 a] on one operand (%s first), then u = 1 a: accepted"
           (if concat_first then "the concatenation" else "4"))
        (match
           List.fold
             (List.map bounds ~f:(fun b -> [ b ]) @ [ [ eq (Row.Var u) (dim ~basis:"a" 1) ] ])
             ~init:([], Row.empty_env)
             ~f:(fun (deferred, env) cs -> Row.solve_inequalities ~stage:Stage4 (deferred @ cs) env)
         with
        | _ -> true
        | exception Row.Shape_error _ -> false));
  (* A 0 stated outright binds only a variable that may be empty (gh-ocannl-2368). For any other it
     waits while constraints register, so an alias to a discardable variable arriving later still
     binds it, and is refused at closing. On the base this store accepted [u = 0]. *)
  in_every_order "[u; 4] = [k; 2], k = 2, u = 0 stated: refused" ~n:3
    (fun u k _ ->
      [ concat u k ~basis:Row.default_basis; eq (Row.Var k) (dim 2); eq (Row.Var u) (dim 0) ])
    ~want:"refused";
  in_every_order "u = 0 stated, nothing else: refused" ~n:1
    (fun u _ _ -> [ eq (Row.Var u) (dim 0) ])
    ~want:"refused";
  in_every_order "0 = u stated, nothing else: refused" ~n:1
    (fun u _ _ -> [ eq (dim 0) (Row.Var u) ])
    ~want:"refused";
  in_every_order ~discardable:`U "u = 0 stated, u discardable: sized 0" ~n:1
    (fun u _ _ -> [ eq (Row.Var u) (dim 0) ])
    ~want:"u = 0 of default, k = 1 of bcast_if_1";
  (* The permission travels with an equation, whichever comes first: [w] discardable sized 0 and [u
     = w] in either order (the 0 [u] inherits came through substitution), or [u = 0] stated before
     the alias that licenses it. *)
  in_every_order ~discardable:`W "w = 0, u = w, w discardable: u inherits 0" ~n:2
    (fun u _ w -> [ eq (Row.Var w) (dim 0); eq (Row.Var u) (Row.Var w) ])
    ~want:"u = 0 of default, k = 1 of bcast_if_1";
  in_every_order ~discardable:`W "u = 0, u = w, w discardable: the alias licenses u" ~n:2
    (fun u _ w -> [ eq (Row.Var u) (dim 0); eq (Row.Var u) (Row.Var w) ])
    ~want:"u = 0 of default, k = 1 of bcast_if_1";
  (* A row equation substitutes [w] before its axes meet, so the 0 reaches [u] as a value, not a
     statement: it is inherited in either order. *)
  in_every_order ~discardable:`W "w = 0, [u] = [w] as rows, w discardable: u inherits 0" ~n:2
    (fun u _ w ->
      let row v : Row.t =
        {
          beg_dims = [];
          dims = [ Row.Var v ];
          bcast = Broadcastable;
          prov = Row.provenance ~sh_id:2368 ~kind:`Output;
        }
      in
      [ eq (Row.Var w) (dim 0); Row.Row_eq { r1 = row u; r2 = row w; origin = dummy_origin } ])
    ~want:"u = 0 of default, k = 1 of bcast_if_1";
  in_every_order "u = [] stated, the empty concatenation: refused" ~n:1
    (fun u _ _ -> [ eq (Row.Var u) (Row.Concat []) ])
    ~want:"refused";
  (* Equal values are no alias: [w] sized 0 licenses no other variable sized 0 by a literal. *)
  in_every_order ~discardable:`W "w = 0, u = 0, w discardable: u is refused" ~n:2
    (fun u _ w -> [ eq (Row.Var w) (dim 0); eq (Row.Var u) (dim 0) ])
    ~want:"refused";
  (* A concatenation component stated 0 is a zero-width segment, as an einsum extraction's capture
     [p^o^q => o] with [q = 0] states it (einsum specs mark no discardable component). The
     arithmetic still waits on a component's zero residual: [[u; 4] = [k; 2]] meets [u = 0] in every
     order, and alone it is refused at closure, which sizes no component. *)
  in_every_order "u a concatenation component, u = 0 stated: a zero-width segment" ~n:1
    (fun u _ _ ->
      Row.add_concat_component u;
      [ eq (Row.Var u) (dim 0) ])
    ~want:"u = 0 of default, k = 1 of bcast_if_1";
  in_every_order "[u; 4] = [k; 2], k = 2, u = 0 stated, u a component: sized 0" ~n:3
    (fun u k _ ->
      Row.add_concat_component u;
      [ concat u k ~basis:Row.default_basis; eq (Row.Var k) (dim 2); eq (Row.Var u) (dim 0) ])
    ~want:"u = 0 of default, k = 2 of default";
  in_every_order "[u; 4] = [k; 2], k = 2, u a component: the zero residual is not sized" ~n:2
    (fun u k _ ->
      Row.add_concat_component u;
      [ concat u k ~basis:Row.default_basis; eq (Row.Var k) (dim 2) ])
    ~want:"refused";
  (* A residual of one component erases to it, which passed a bare 0 to the binding: [[k; u; 2] =
     [k; 2]] cancels to [[u] = []] and sized [u] to 0 on the base. It waits as the [Concat]/[Dim]
     residual does. *)
  in_every_order "[k; u; 2] = [k; 2], u a component: refused, not sized 0" ~n:1
    (fun u k _ ->
      Row.add_concat_component u;
      [ eq (Row.Concat [ Row.Var k; Row.Var u; dim 2 ]) (Row.Concat [ Row.Var k; dim 2 ]) ])
    ~want:"refused";
  in_every_order ~discardable:`U "[k; u; 2] = [k; 2], u discardable: sized 0" ~n:1
    (fun u k _ ->
      [ eq (Row.Concat [ Row.Var k; Row.Var u; dim 2 ]) (Row.Concat [ Row.Var k; dim 2 ]) ])
    ~want:"u = 0 of default, k = 1 of bcast_if_1"

(* gh-ocannl-2368: closure's guesses read the permission to be empty through an alias. With [u]
   discardable and [u = w], closure sizes both 0 in either orientation of the equation, which
   decides whether substitution presents [u] or [w]; on the base, plain membership guessed [w] to 1
   when it represented the two. With [w] registered as a concatenation component, the guess still
   goes to 0 instead of refusing to size it (gh-ocannl-1656 exempts a component that may be
   empty). *)
let test_alias_guesses_empty () =
  Stdio.printf "Closure guesses an alias of a discardable component to 0\n";
  let solve ~component ~u_first =
    let u = Row.get_var () and w = Row.get_var () in
    if component then Row.add_concat_component w;
    let alias = if u_first then eq (Row.Var u) (Row.Var w) else eq (Row.Var w) (Row.Var u) in
    let close : Row.constraint_ =
      Row.Shape_row
        ( {
            beg_dims = [];
            dims = [ Row.Var w ];
            bcast = Broadcastable;
            prov = Row.provenance ~sh_id:2368 ~kind:`Output;
          },
          dummy_origin )
    in
    let discardable_vars = Set.add Row.dim_var_set_empty u in
    match
      List.fold
        [
          (Row.Stage1, [ alias ]);
          (Stage4, [ close ]);
          (Stage5, []);
          (Stage6a, []);
          (Stage6b, []);
          (Stage7, []);
        ]
        ~init:([], Row.empty_env)
        ~f:(fun (pending, env) (stage, cs) ->
          Row.solve_inequalities ~stage ~discardable_vars (pending @ cs) env)
    with
    | [], env -> (Row.get_dim_val env u, Row.get_dim_val env w)
    | _ -> (None, None)
    | exception Row.Shape_error _ -> (None, None)
  in
  List.iter [ false; true ] ~f:(fun component ->
      Verdict.Claims.p_all ~min:2
        (Printf.sprintf "  u = w, u discardable%s: both 0 in either orientation, nothing pending"
           (if component then ", w a concatenation component" else ""))
        [ true; false ]
        ~f:(fun u_first ->
          match solve ~component ~u_first with
          | Some 0, Some 0 -> true
          | u, w ->
              let show = function Some n -> Int.to_string n | None -> "?" in
              Stdio.eprintf
                "  alias guess (u_first=%b, component=%b): u=%s w=%s (not part of the golden)\n"
                u_first component (show u) (show w);
              false));
  (* [concat_bounded_waiting] carries a waiting component's bound onto its closing partner whether
     or not that partner may be empty: gh-ocannl-2268's store [w(s2,k3) (v) = 5_a ; [u; 2v] = [k;
     2_a] ; k >= 5] with [u] equated to a discardable [w] has the one answer [u = 3_a, k = 5], since
     [u = 0] would make [k = 2_a], which its bound refutes. On the base, plain membership refused it
     when [w] represented the two. *)
  let bounded_store ~u_first =
    let v = Row.get_var () and u = Row.get_var () and k = Row.get_var () and w = Row.get_var () in
    let window =
      Row.Affine
        {
          stride = 2;
          over = Row.Var v;
          conv = Some { dilation = 1; kernel = dim 3; use_padding = false };
          stride_offset = 0;
        }
    in
    let doubled = Row.Affine { stride = 2; over = Row.Var v; conv = None; stride_offset = 0 } in
    let cs =
      [
        eq window (dim ~basis:"a" 5);
        eq (Row.Concat [ Row.Var u; doubled ]) (Row.Concat [ Row.Var k; dim ~basis:"a" 2 ]);
        ineq (dim 5) k;
        (if u_first then eq (Row.Var u) (Row.Var w) else eq (Row.Var w) (Row.Var u));
      ]
    in
    let close : Row.constraint_ =
      Row.Shape_row
        ( {
            beg_dims = [];
            dims = List.map [ v; u; k; w ] ~f:(fun x -> Row.Var x);
            bcast = Broadcastable;
            prov = Row.provenance ~sh_id:2368 ~kind:`Output;
          },
          dummy_origin )
    in
    let discardable_vars = Set.add Row.dim_var_set_empty w in
    let show env x =
      match Row.get_dim_from_env env x with
      | Some (Row.Dim { d; basis; _ }) -> Printf.sprintf "%d of %s" d basis
      | _ -> "unsized"
    in
    match
      List.fold
        (List.map cs ~f:(fun c -> (Row.Stage1, [ c ]))
        @ [ (Row.Stage4, [ close ]); (Stage5, []); (Stage6a, []); (Stage6b, []); (Stage7, []) ])
        ~init:([], Row.empty_env)
        ~f:(fun (pending, env) (stage, cs) ->
          Row.solve_inequalities ~stage ~discardable_vars (pending @ cs) env)
    with
    | [], env -> Printf.sprintf "u = %s, k = %s" (show env u) (show env k)
    | pending, _ -> Printf.sprintf "%d pending" (List.length pending)
    | exception Row.Shape_error _ -> "refused"
  in
  let want = "u = 3 of a, k = 5 of default" in
  Verdict.Claims.p_all ~min:2
    ("  a bounded closing component aliased to a discardable one: " ^ want
   ^ " in either orientation")
    [ true; false ]
    ~f:(fun u_first ->
      let got = bounded_store ~u_first in
      if not (String.equal got want) then
        Stdio.eprintf "  bounded store aliased (u_first=%b): %s (not part of the golden)\n" u_first
          got;
      String.equal got want)

(* With a claim already on each prefix, the ordinary ground equation accepts the free components
   under their own bounds. Exercise the closing prepass without the window that sizes the prefix in
   [test_concat_order_independence]. *)
let test_plain_concat_bounds () =
  let no_extra _ _ = [] in
  let fixed_min v d : Row.constraint_ =
    Row.Dim_constr
      { d = Row.Var v; constr = At_least_dim { d_min = d; sites = [] }; origin = dummy_origin }
  in
  let stride2 v = Row.Affine { stride = 2; over = Row.Var v; conv = None; stride_offset = 0 } in
  let closed dims : Row.t =
    { beg_dims = []; dims; bcast = Broadcastable; prov = Row.provenance ~sh_id:2317 ~kind:`Output }
  in
  let settles ~reverse ~k_size ~u_expected ?bound_u ?(before = no_extra) ?(after = no_extra) () =
    let u = Row.get_var () and k = Row.get_var () in
    let larger = Row.Concat [ Row.Var u; dim ~basis:"a" 4 ]
    and smaller = Row.Concat [ Row.Var k; dim ~basis:"a" 2 ] in
    let equality = if reverse then eq smaller larger else eq larger smaller in
    let constraints =
      before u k
      @ [ equality; ineq (dim k_size) k ]
      @ Option.to_list (Option.map bound_u ~f:(fun d -> ineq (dim d) u))
      @ after u k
    in
    let close = Row.Shape_row (closed [ Row.Var u; Row.Var k ], dummy_origin) in
    let stages =
      [ (Row.Stage1, constraints); (Stage4, [ close ]); (Stage6a, []); (Stage6b, []); (Stage7, []) ]
    in
    let result =
      List.fold stages
        ~init:(`Ok ([], Row.empty_env))
        ~f:(fun acc (stage, cs) ->
          match acc with
          | `Ok (deferred, env) -> run_solve_env ~stage (deferred @ cs) env
          | (`Shape _ | `Other _) as error -> error)
    in
    match result with
    | `Ok (_, env) ->
        let has v (size, tag) =
          match Row.get_dim_from_env env v with
          | Some (Row.Dim { d; basis; _ }) -> d = size && String.equal basis tag
          | _ -> false
        in
        has u u_expected && has k (k_size, Row.default_basis)
    | `Shape m | `Other m ->
        Stdio.eprintf "plain concat bounds: %s\n" m;
        false
  in
  Verdict.Claims.p_all ~min:2
    "  matching plain concat bounds settle to both GLBs in either orientation" [ false; true ]
    ~f:(fun reverse -> settles ~reverse ~k_size:5 ~u_expected:(3, Row.default_basis) ~bound_u:3 ());
  Verdict.Claims.p_all ~min:2
    "  a plain concat bound at the minimum leaves a positive residual in either orientation"
    [ false; true ] ~f:(fun reverse -> settles ~reverse ~k_size:3 ~u_expected:(1, "a") ());
  (* The fixed index sizes the shared stride at stage 7. Its held equations give both free
     components their plain tags before the concatenation chooses a residual's tag. *)
  Verdict.Claims.p_all ~min:2 "  held size equations still supply the components at the minimum"
    [ false; true ] ~f:(fun reverse ->
      let after u k =
        let w = Row.get_var () in
        [ eq (bound u [ 3 ]) (stride2 w); eq (bound k [ 1 ]) (stride2 w); fixed_min w 2 ]
      in
      settles ~reverse ~k_size:3 ~u_expected:(1, Row.default_basis) ~after ());
  (* The deferred same-sum equation must see both plain stride-derived tags. With u bound early, it
     gives x the prefix claim before the stride can size x. *)
  Verdict.Claims.p_all ~min:2 "  matching plain bounds let competing held equations supply tags"
    [ false; true ] ~f:(fun reverse ->
      let before u _ =
        let w = Row.get_var () and x = Row.get_var () in
        let with_claim v = Row.Concat [ Row.Var v; dim ~basis:"a" 4 ] in
        [
          eq (with_claim u) (with_claim x);
          eq (bound u [ 3 ]) (stride2 w);
          eq (bound x [ 3 ]) (stride2 w);
          fixed_min w 3;
        ]
      in
      settles ~reverse ~k_size:5 ~u_expected:(3, Row.default_basis) ~bound_u:3 ~before ());
  (* A held row total can also supply the tag, when the fixed minimum gives its other axis a
     size. *)
  Verdict.Claims.p_all ~min:2 "  held row totals supply the components at the minimum"
    [ false; true ] ~f:(fun reverse ->
      let before u k =
        let w = Row.get_var () in
        let total v n : Row.constraint_ =
          Row.Rows_constr
            {
              r = [ closed [ Row.Var v; Row.Var w ] ];
              constr = Total_elems { numerator = Num_elems n; divided_by = []; keep_axis = false };
              origin = dummy_origin;
            }
        in
        [ total u 2; total k 6; fixed_min w 2 ]
      in
      settles ~reverse ~k_size:3 ~u_expected:(1, Row.default_basis) ~before ())

let () =
  Stdio.printf "=== Concat dim-solver hardening (solver-level) ===\n";
  Tensor.unsafe_reinitialize ();
  test_ac1_glb_merge_compatible ();
  test_ac1_glb_merge_incompatible_demotes ();
  test_ac1_glb_merge_postpone_below_stage4 ();
  test_unequal_arity_waits ();
  test_solved_side_waits ();
  test_nested_concat_flattened ();
  test_ac3_affine_residual_not_overconstrained ();
  test_concrete_concat ();
  test_tagged_components_any_order ();
  test_zero_component ();
  test_alias_guesses_empty ();
  Verdict.case "plain concatenation bounds" test_plain_concat_bounds;
  Stdio.printf "=== Done ===\n"

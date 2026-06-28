(** Pinning tests for the transitive-glb reading and dimension-closing counterexample of Remark
    6.3 in docs/blog/ocannl-formal-core.md.

    Phi = {3 <= alpha, alpha <= beta, 5 <= beta}, both variables terminal (leaves). Satisfiable:
    the solutions are exactly beta = 1_(bcast_if_1), alpha in {3, 1_(bcast_if_1)}.

    Naive sequential closing is order-sensitive: committing alpha first re-emits 3 <= beta, the
    conflicting ground bounds {5, 3} on beta demote to the broadcast top (DI-cap), and closing
    finds the genuine solution alpha = 3, beta = top. Committing beta |-> 5 first squeezes alpha
    into the empty interval 3 <= alpha <= 5 (atoms form an antichain below the top), failing a
    satisfiable store.

    This test drives the same constraint set through the solver with both emission orders of the
    terminals (and of the inequalities), pinning that beta closes to its transitive glb, not just
    the explicit cap 5.

    It also pins two nearby cases:
    - {3 <= alpha, beta <= alpha}: beta is not guessed to top before alpha closes; it is pinned to
      3 during re-solving.
    - {3 <= alpha, 5 <= beta, gamma <= alpha, gamma <= beta}: the store is satisfiable by raising a
      terminal to top, but deterministic downward leaf closing commits alpha = 3 and beta = 5, then
      fails when gamma is pinned below incompatible atoms. *)

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

let check name (alpha, beta, constraints) =
  match
    let stages = [ Row.Stage2; Stage3; Stage4; Stage5; Stage6; Stage7 ] in
    let leftover, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    List.fold stages ~init:(leftover, env) ~f:(fun (leftover, env) stage ->
        Row.solve_inequalities ~stage leftover env)
  with
  | _, env ->
      let value v = Option.value_map (Row.get_dim_val env v) ~default:"?" ~f:Int.to_string in
      Stdio.printf "%s: no error, alpha = %s, beta = %s\n" name (value alpha) (value beta)
  | exception Row.Shape_error (msg, _) -> Stdio.printf "%s: Shape_error: %s\n" name msg

let check3 name (alpha, beta, gamma, constraints) =
  match
    let stages = [ Row.Stage2; Stage3; Stage4; Stage5; Stage6; Stage7 ] in
    let leftover, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
    List.fold stages ~init:(leftover, env) ~f:(fun (leftover, env) stage ->
        Row.solve_inequalities ~stage leftover env)
  with
  | _, env ->
      let value v = Option.value_map (Row.get_dim_val env v) ~default:"?" ~f:Int.to_string in
      Stdio.printf "%s: no error, alpha = %s, beta = %s, gamma = %s\n" name (value alpha)
        (value beta) (value gamma)
  | exception Row.Shape_error (msg, _) -> Stdio.printf "%s: Shape_error: %s\n" name msg

let interleave_example ~terminals_order ~ineqs_order =
  let alpha_v = Row.get_var ~name:"alpha" () in
  let beta_v = Row.get_var ~name:"beta" () in
  let alpha = Row.Var alpha_v in
  let beta = Row.Var beta_v in
  let ineqs =
    [
      Row.Dim_ineq { res = dim 3; opnd = alpha; from_ = Sexp.List []; origin };
      Row.Dim_ineq { res = alpha; opnd = beta; from_ = Sexp.List []; origin };
      Row.Dim_ineq { res = dim 5; opnd = beta; from_ = Sexp.List []; origin };
    ]
  in
  let ineqs = match ineqs_order with `Forward -> ineqs | `Reverse -> List.rev ineqs in
  let terminals =
    [ Row.Terminal_dim (false, alpha, origin); Row.Terminal_dim (false, beta, origin) ]
  in
  let terminals =
    match terminals_order with `Alpha_first -> terminals | `Beta_first -> List.rev terminals
  in
  (alpha_v, beta_v, ineqs @ terminals)

let upper_bound_example ~terminals_order ~ineqs_order =
  let alpha_v = Row.get_var ~name:"alpha" () in
  let beta_v = Row.get_var ~name:"beta" () in
  let alpha = Row.Var alpha_v in
  let beta = Row.Var beta_v in
  let ineqs =
    [
      Row.Dim_ineq { res = dim 3; opnd = alpha; from_ = Sexp.List []; origin };
      Row.Dim_ineq { res = beta; opnd = alpha; from_ = Sexp.List []; origin };
    ]
  in
  let ineqs = match ineqs_order with `Forward -> ineqs | `Reverse -> List.rev ineqs in
  let terminals =
    [ Row.Terminal_dim (false, alpha, origin); Row.Terminal_dim (false, beta, origin) ]
  in
  let terminals =
    match terminals_order with `Alpha_first -> terminals | `Beta_first -> List.rev terminals
  in
  (alpha_v, beta_v, ineqs @ terminals)

let incompatible_upper_bounds_example ~terminals_order ~ineqs_order =
  let alpha_v = Row.get_var ~name:"alpha" () in
  let beta_v = Row.get_var ~name:"beta" () in
  let gamma_v = Row.get_var ~name:"gamma" () in
  let alpha = Row.Var alpha_v in
  let beta = Row.Var beta_v in
  let gamma = Row.Var gamma_v in
  let ineqs =
    [
      Row.Dim_ineq { res = dim 3; opnd = alpha; from_ = Sexp.List []; origin };
      Row.Dim_ineq { res = dim 5; opnd = beta; from_ = Sexp.List []; origin };
      Row.Dim_ineq { res = gamma; opnd = alpha; from_ = Sexp.List []; origin };
      Row.Dim_ineq { res = gamma; opnd = beta; from_ = Sexp.List []; origin };
    ]
  in
  let ineqs = match ineqs_order with `Forward -> ineqs | `Reverse -> List.rev ineqs in
  let terminals =
    [ Row.Terminal_dim (false, alpha, origin); Row.Terminal_dim (false, beta, origin) ]
  in
  let terminals =
    match terminals_order with `Alpha_first -> terminals | `Beta_first -> List.rev terminals
  in
  (alpha_v, beta_v, gamma_v, ineqs @ terminals)

let () =
  check "alpha terminal first, ineqs forward"
    (interleave_example ~terminals_order:`Alpha_first ~ineqs_order:`Forward);
  check "beta terminal first, ineqs forward"
    (interleave_example ~terminals_order:`Beta_first ~ineqs_order:`Forward);
  check "alpha terminal first, ineqs reversed"
    (interleave_example ~terminals_order:`Alpha_first ~ineqs_order:`Reverse);
  check "beta terminal first, ineqs reversed"
    (interleave_example ~terminals_order:`Beta_first ~ineqs_order:`Reverse);
  check "upper alpha terminal first, ineqs forward"
    (upper_bound_example ~terminals_order:`Alpha_first ~ineqs_order:`Forward);
  check "upper beta terminal first, ineqs forward"
    (upper_bound_example ~terminals_order:`Beta_first ~ineqs_order:`Forward);
  check "upper alpha terminal first, ineqs reversed"
    (upper_bound_example ~terminals_order:`Alpha_first ~ineqs_order:`Reverse);
  check "upper beta terminal first, ineqs reversed"
    (upper_bound_example ~terminals_order:`Beta_first ~ineqs_order:`Reverse);
  check3 "incompatible alpha terminal first, ineqs forward"
    (incompatible_upper_bounds_example ~terminals_order:`Alpha_first ~ineqs_order:`Forward);
  check3 "incompatible beta terminal first, ineqs forward"
    (incompatible_upper_bounds_example ~terminals_order:`Beta_first ~ineqs_order:`Forward);
  check3 "incompatible alpha terminal first, ineqs reversed"
    (incompatible_upper_bounds_example ~terminals_order:`Alpha_first ~ineqs_order:`Reverse);
  check3 "incompatible beta terminal first, ineqs reversed"
    (incompatible_upper_bounds_example ~terminals_order:`Beta_first ~ineqs_order:`Reverse)

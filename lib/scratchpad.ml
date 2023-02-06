(** This module is just for illustrating issue #41. *)

type _ calc =
  | Lambda: {bind: 'b ref; body: 'a calc} -> ('b -> 'a) calc
  | App: {func: ('a -> 'b) calc; arg: 'a calc} -> 'b calc
  | Var: 'a ref -> 'a calc
  | Comp: 'a arith -> 'a calc

and _ arith =
  | Add_int: int arith * int arith -> int arith
  | Add_float: float arith * float arith -> float arith
  | Const: 'a -> 'a arith

  (* [@@deriving_inline fold_sig] *)
(* START of auto-generated code *)
module type FOLD_CALC = sig
  type 'a calc_result
  type 'a arith_result
  val lambda: bind: 'b ref -> body: 'a calc_result -> ('b -> 'a) calc_result
  val app: func:('a -> 'b) calc_result -> arg: 'a calc_result -> 'b calc_result
  val var: 'a ref -> 'a calc_result
  val comp: 'a arith_result -> 'a calc_result

  val add_int: int arith_result -> int arith_result -> int arith_result
  val add_float: float arith_result -> float arith_result -> float arith_result
  val const: 'a -> 'a arith_result
end

module Fold(F: FOLD_CALC) = struct
  let fold c =
    let rec calc_loop: 'a. 'a calc -> 'a F.calc_result =
      fun (type a) (calc: a calc) ->
        match calc with
        | Lambda {bind; body} -> (F.lambda ~bind ~body:(calc_loop body) : a F.calc_result)
        | App {func; arg} -> F.app ~func:(calc_loop func) ~arg:(calc_loop arg)
        | Var v -> F.var v
        | Comp c -> F.comp (arith_loop c)
    and arith_loop: 'a. 'a arith -> 'a F.arith_result =
      fun (type a) (arith: a arith) ->
        match arith with
        | Add_int (a1, a2) -> (F.add_int (arith_loop a1) (arith_loop a2): a F.arith_result)
        | Add_float (a1, a2) -> F.add_float (arith_loop a1) (arith_loop a2)
        | Const c -> F.const c in
    calc_loop c
end
(* [@@@end] *)
(* END of auto-generated code *)

module Eval: FOLD_CALC = struct
  type 'a calc_result = 'a
  type 'a arith_result = 'a
  let lambda ~bind ~body = fun x -> bind := x; body
  let app ~func ~arg = func arg
  let var v = !v
  let comp c = c
  let add_int i1 i2 = Int.add i1 i2
  let add_float i1 i2 = Float.add i1 i2
  let const c = c
end

module EvalComp = Fold(Eval)

module PrintApTree: FOLD_CALC = struct
  type 'a calc_result = string
  type 'a arith_result = int
  let lambda ~bind:_ ~body = "(fun <some var> -> "^body^")"
  let app ~func ~arg = "("^func^" "^arg^")"
  let var _ = "<some var>"
  let comp c = "(added "^Int.to_string c^" elements)"
  let add_int i1 i2 = i1 + i2
  let add_float i1 i2 = i1 + i2
  let const _ = 1
end

module PrintApTreeComp = Fold(PrintApTree)
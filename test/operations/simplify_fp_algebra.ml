(* gh-ocannl-998: each ablation controls only its floating-point family. *)
open Base
open Verdict.Claims
open Ll_test
module LL = Ir.Low_level
module Ops = Ir.Ops

let () =
  let mk = node_factory ~first_id:99800 ~dims:[| 1 |] () in
  let input = mk "input" and output = mk "output" in
  let x = get input [| fixed 0 |] in
  let sub = binop Ops.Sub and div = binop Ops.Div in
  let cases =
    [
      ("contract", add (mul x x) x);
      ("constants", add (add x (c 2.)) (c 3.));
      ("sub", add x (sub x x));
      ("mul_div", mul x (div x x));
      ("pow", binop Ops.ToPowOf x (c 3.));
      ("identities", mul (c 0.) x);
    ]
  in
  let simplify selection rhs =
    match LL.simplify_llc ~fp_algebra:selection [] (set_at output (fixed 0) rhs) with
    | LL.Set { llsc; _ } -> llsc
    | _ -> failwith "simplification lost the assignment"
  in
  p_all "all-off preserves each floating-point expression" cases ~f:(fun (_, rhs) ->
      LL.equal_scalar_t rhs (simplify "none" rhs));
  p_all "each family changes its witness when selected" cases ~f:(fun (family, rhs) ->
      not (LL.equal_scalar_t rhs (simplify family rhs)));
  let pairs =
    List.concat_map cases ~f:(fun (family, _) ->
        List.map cases ~f:(fun (other, rhs) -> (family, other, rhs)))
  in
  p_all "each family leaves other witnesses alone" pairs ~f:(fun (family, other, rhs) ->
      String.equal family other || LL.equal_scalar_t rhs (simplify family rhs));
  p_all "default matches explicit all" cases ~f:(fun (_, rhs) ->
      let default = LL.simplify_llc [] (set_at output (fixed 0) rhs) in
      match default with
      | LL.Set { llsc; _ } -> LL.equal_scalar_t llsc (simplify "all" rhs)
      | _ -> false);
  let add_zero = add x (c 0.) in
  p "all-off retains signed-zero addition" (LL.equal_scalar_t add_zero (simplify "none" add_zero));
  p "constant folding still runs all-off"
    (LL.equal_scalar_t (c 5.) (simplify "none" (add (c 2.) (c 3.))));
  let imk = node_factory ~prec:Ops.int64 ~first_id:99900 ~dims:[| 1 |] () in
  let integer_out = imk "integer_out" and integer_input = imk "integer_input" in
  let i = get integer_input [| fixed 0 |] in
  let ibinop op a b = LL.Binop (op, (a, Ops.int64), (b, Ops.int64)) in
  let isimplify rhs =
    match LL.simplify_llc ~fp_algebra:"none" [] (set_at integer_out (fixed 0) rhs) with
    | LL.Set { llsc; _ } -> llsc
    | _ -> failwith "integer simplification lost the assignment"
  in
  p "integer zero identity still runs all-off"
    (LL.equal_scalar_t i (isimplify (ibinop Ops.Add i (c 0.))));
  let integer_constants = [ (Ops.Add, 5.); (Ops.Mul, 6.) ] in
  p_all "integer constant reassociation still runs all-off" integer_constants
    ~f:(fun (op, combined) ->
      LL.equal_scalar_t
        (isimplify (ibinop op (ibinop op i (c 2.)) (c 3.)))
        (ibinop op (c combined) i));
  p "unknown family is refused"
    (try
       ignore (simplify "typo" x);
       false
     with Utils.User_error _ -> true)

open Ocannl.Operation.DSL_modules

(* Fixtures pinning the call shapes the raw %cd accumulation path expands to. "Raw" means no
   [~projections:] is in scope, so ppx_cd's single [process_raw_accum] path constructs the shape
   logic itself and hands it to [Tensor.raw_accum]: unary arities go through [Shape.Transpose],
   ternary ones through [Shape.Broadcast_tern], and every operand -- merge buffers included --
   reaches the assignment through [Tensor.buffer_of]. The binary [Shape.Broadcast] arity is already
   pinned by [test_compose_logic_err]. *)

(* Unary arity: the un_op's own shape logic, wrapped in Shape.Transpose. *)
let test_raw_unop a =
  let%cd _r = { r } =: relu a in
  _r

(* The transpose_type that gives Shape.Transpose its name: ~logic:"T" swaps inputs and outputs. *)
let test_raw_unop_transpose a =
  let%cd _r = { r } =:+ id a ~logic:"T" in
  _r

(* An einsum spec on a unary operand becomes Shape.Permute under the same Shape.Transpose. *)
let test_raw_unop_permute a =
  let%cd _r = { r } =:+ id a ~logic:"ij=>ji" in
  _r

(* A bare assignment has no un_op at all and still routes through the unary shape logic. *)
let test_raw_identity a =
  let%cd _r = { r } =: a in
  _r

(* Ternary arity: three operand shapes under Shape.Broadcast_tern. *)
let test_raw_ternop_fma a b c =
  let%cd _r = { r } =: fma a b c in
  _r

(* An explicit ~logic:"." picks the pointwise ternary_type. *)
let test_raw_ternop_pointwise a b c =
  let%cd _r = { r } =: where a b c ~logic:"." in
  _r

(* An einsum spec on a ternary operand becomes Shape.Einsum_tern. *)
let test_raw_ternop_einsum a b c =
  let%cd _r = { r } =:+ mul3 a b c ~logic:"i;i;i=>i" in
  _r

(* Merge-buffer operand: .merge sets ~is_merge:true on that operand's Tensor.buffer_of. *)
let test_merge_value_operand a =
  let%cd _r = { r } =: a.merge in
  _r

(* Per-operand flags: .grad.merge sets both ~is_grad:true and ~is_merge:true, while the plain
   operand beside it keeps both false. *)
let test_merge_grad_operand a b =
  let%cd _r = { r } =:+ a.grad.merge * b ~logic:"." in
  _r

(* A statement sequence (gh-ocannl-1770): the first statement is let-bound before the list, so the
   statements are evaluated in source order and the first one naming a forward root embeds its code.
   Within each statement, the operands' forward code is let-bound in order before the assignment. *)
let test_statement_sequence a =
  let%cd _r =
    { p1 } =: a;
    { p2 } =: a
  in
  _r

let () =
  ignore
    ( test_raw_unop,
      test_raw_unop_transpose,
      test_raw_unop_permute,
      test_raw_identity,
      test_raw_ternop_fma,
      test_raw_ternop_pointwise,
      test_raw_ternop_einsum,
      test_merge_value_operand,
      test_merge_grad_operand,
      test_statement_sequence )

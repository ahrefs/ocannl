# Syntax extensions `%cd` and `%op`

## Notes true for bogth `%cd` and `%op`

When an extension is over a wildcard (ignore result) binding: `let%cd _ = ...` and `let%op _ = ...`, the generated code is wrapped in `Tensor.with_unchanged_roots`, to prevent it from upsetting rootness checks. The use-case for writing `%op` and `%cd` notations with ignored result is to generate additional shape inference constraints.

## Syntax extension `%cd`, standing for "code", to express assignments: `Assignments.t`

### Implementation details

The translate function returns an record. The `expr` field (filler expression) meaning depends on `typ` (filler type): for `Code`, this is an `Assignments.t` expression. For `Unknown` and `Tensor`, this is a `Tensor.t` expression. For `Array` and `Merge_value`, this is a non-optional `Tnode.t` expression, and for `Grad_of_tensor` and `Merge_grad`, it's an optional `Tnode.t` expresssion.

Next, `setup_array ~is_lhs:true` converts the filler expression into a `Tnode.t option` expression, and `setup_array ~is_lhs:false` converts the filler into an `Assignments.buffer option` expression according to `filler_typ`.

```ocaml
type expr_type =
  | Code
  | Array
  | Grad_of_tensor of expression
  | Tensor
  | Unknown
  | Merge_value
  | Merge_grad of expression

type projections_slot = LHS | RHS1 | RHS2 | Nonslot | Undet
```

## Syntax extension `%op`, standing for "operation", to express tensors: `Tensor.t`
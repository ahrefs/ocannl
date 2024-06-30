# Syntax extensions `%cd` and `%op`

## Syntax extension `%cd`, standing for "code", to express assignments: `Assignments.t`

### Implementation details

The translate function returns a triple. The first component, `filler_typ`, is an `expr_type`. The second component, `slot`, is a `projection_slot` designator. The third component, `filler` is an expression. Its meaning depends on `filler_typ`: for `Code`, this is an `Assignments.t` expression. For `Unknown` and `Tensor`, this is a `Tensor.t` expression. For `Array` and `Merge_value`, this is a non-optional `Tnode.t` expression, and for `Grad_of_tensor` and `Merge_grad`, it's an optional `Tnode.t` expresssion.

Next, `setup_array ~is_lhs:true` converts the filler into a `Tnode.t option` expression, and `setup_array ~is_lhs:false` converts the filler into an `Assignments.buffer option` expression according to `filler_typ`.

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
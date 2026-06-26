# Neural Networks Without Shape Boilerplate: An OCaml DSL Case Study

## Abstract

Modern neural network code is often cluttered with shape bookkeeping: explicit calls to reshape, unsqueeze, expand, transpose, reductions, and integer-axis arguments. OCANNL combines an embedded DSL that removes this boilerplate, with an end-to-end compiler with GPU backends. Users define tensor operations as OCaml functions using concise notation generalizing the Einstein summation convention. Concatenation is an expression over indices e.g. `i^j`, convolution is a contraction with affine operand addressing e.g. `stride*i + dilation*k`. A sequence of axes can be captured by a single variable, e.g. `..batch..` or `..activations..`, sandwiched between leading and trailing axes; with up to three such variables per shape. The paper showcases three examples: tensor expressions core multi-head attention, rank-polymorphic 2D convolution; and tensor computation for Stochastic Gradient Descent with momentum and Nesterov-inspired correction. OCANNL performs broadcast-aware global bidirectional shape inference and derives loop-index maps for code generation. We formalize the inference problem for the core calculus (excluding affine and concatenation) and show properties of OCANNL's solver (proofs in the appendix). OCANNL's compiler inlines computations to avoid materializing intermediate tensors, and performs common subexpression elimination. This paper does not showcase benchmarks nor argue for performance, leaving that to follow-up work and the Q&A.

## Shapes, Syntax Extensions and Examples

OCANNL shapes have three kinds (sequences) of axes, from outermost: batch, output, input. The kind designation is by convention (not enforced, but facilitated). In specifications, the syntax resembles programming language types: `batch | input -> output`. For example: tensor multiplication `*` reduces the input axes of the operator tensor (matrix etc.) with output axes of the operand tensor (vector or matrix etc.), while broadcasting or treating pointwise the batch axes.

OCANNL introduces two syntaxes. Extension point `%op` creates *tensor expressions* (type `Tensor.t`), or OCaml functions returning tensor expressions which we call tensor *operations*. Extension point `%cd` creates tensor *computations* (assignments, type `Assignments.comp`). Unlike computations, tensor expressions are differentiable and support separate initialization. Both extensions support inline definitions of new tensors via OCaml's record syntax. For example: `{ w1 = kaiming normal1 () }` inside `%op` introduces tensor `w1` with initialization expression `kaiming normal1 ()`; `{ sgd_momentum }` inside `%cd` below introduces a (non-differentiable, no initialization) tensor `sgd_momentum`; `{ w_q }` inside `%op` below introduces a tensor with default initialization (e.g. centered uniform distribution).

At the heart of 

```ocaml
let%op multi_head_attention ~num_heads ~d_k ~d_v () x =
  let q = { w_q } * x in
  let k = { w_k } * x in
  let v = { w_v } * x in
  let scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ])
    /. sqrt (dim d)
  in
  Shape.set_dim h num_heads;
  Shape.set_dim d d_k;
  Shape.set_dim e d_v;
  let attn_weights =
    softmax ~spec:" ... | t -> ..." () scores
  in
  { w_o } *
    (attn_weights +* v
       " ... s | t -> h; ... t | h e => ... s | h e"
       [ "e" ])
```
> **Figure 1. Core multi-head attention in OCANNL.** Axes: query position `s`, key/value position `t`, head `h`, key width `d`, and value width `e`. Batch rank, parameter shapes, score shape, contractions, and loop-index maps are inferred.

```ocaml
let%op conv2d ~label ?(kernel_size = 3) ?(stride = 1)
    ?(use_padding = true) ?out_channels () x =
  Shape.set_dim kh kernel_size;
  Shape.set_dim kw kernel_size;
  Option.iter out_channels ~f:(Shape.set_dim oc);
  x +* { kernel }
       "... | stride*oh + kh, stride*ow + kw, ..ic..;
             kh, kw, ..ic.. -> ..oc..
        => ... | oh, ow, ..oc.."
       [ "kh"; "kw"; "oc" ]
  + { bias = 0. }
```
> **Figure 2. Rank-polymorphic 2D convolution.** The input is addressed at affine spatial positions `stride*oh+kh` and `stride*ow+kw`. The kernel axes `kh` and `kw`, together with the input-channel row `..ic..`, are contracted. The context row `...` and output-channel row `..oc..` are inferred and may have arbitrary rank.

```ocaml
let sgd_one ~learning_rate ?(momentum = 0.0) ?(weight_decay = 0.0)
    ?(nesterov = false) p =
  [%cd
    { sgd_delta } =: p.grad + (!.weight_decay *. p);
    if Float.(momentum > 0.0) then (
      { sgd_momentum } =:
        (!.momentum *. sgd_momentum) + sgd_delta;
      if nesterov then sgd_delta =+ !.momentum *. sgd_momentum
      else sgd_delta =: sgd_momentum);
    p =- learning_rate * sgd_delta ~logic:"."]
```
> **Figure 3. Stochastic Gradient Descent with momentum.** Introduces intermediate state-tracking tensors in a computation. Optional Nesterov-inspired correction.

## Shape and Projections inference semantics

Definitions and theorems.

## Contexts and explicit compilation

Unified, explicitly passed (immutable) context values. Empty root contexts determine device.

## Related work

## Conclusion

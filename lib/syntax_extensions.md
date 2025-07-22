# Syntax extensions %cd and %op

- Table of contents
  - [Preliminaries](#preliminaries)
  - [Primitive operations](#primitive-operations)
  - [The syntax for %op](#the-syntax-for-op)
  - [The syntax for %cd](#the-syntax-for-cd)
  - [Numeric and N-dimensional array literals](#numeric-and-n-dimensional-array-literals)
  - [Wildcard bindings](#wildcard-bindings)
  - [Inline declarations](#inline-declarations)
  - [Using OCANNL's generalized einsum notation](#using-ocannls-generalized-einsum-notation)
    - [Syntax of the generalized einsum notation](#syntax-of-the-generalized-einsum-notation)
  - [Further features of the syntax extension %cd](#further-features-of-the-syntax-extension-cd)
    - [Referencing arrays: tensor value, tensor gradient, merge buffer of a tensor node](#referencing-arrays-tensor-value-tensor-gradient-merge-buffer-of-a-tensor-node)
    - [Block comments](#block-comments)
  - [Further features of the syntax extension %op](#further-features-of-the-syntax-extension-op)
    - [Name from binding](#name-from-binding)
    - [Label from function argument](#label-from-function-argument)
    - [Configuring inline declarations: inline output dimensions, initial values](#configuring-inline-declarations-inline-output-dimensions-initial-values)
    - [Lifting of the applications of config arguments: if an error, refactor your code](#lifting-of-the-applications-of-config-arguments-if-an-error-refactor-your-code)
  - [Implementation details](#implementation-details)
    - [The hard-coded to-the-power-of operator](#the-hard-coded-to-the-power-of-operator)
    - [Intricacies of the syntax extension %cd](#intricacies-of-the-syntax-extension-cd)
- In a nutshell
  - Syntax extension `%cd` stands for "code", to express assignments and computations: `Assignments.comp`.
  - Syntax extension `%op` stands for "operation", to express tensors: `Tensor.t`.

## Preliminaries

OCANNL, and arrayjit specifically, is built around a fixed number of numeric operations, declared in `arrayjit/ops.ml`. We assign lexical operators to the binary operations, inventing novel operators if needed. For example, Rectified Linear Unit `Relu` operation, which computes `f(x) = max(0,x)`, is called `relu`, while the ReLU-Gate `Relu_gate` operation, which computes `f(x,y) = if x > 0.0 then y else 0.0`, gets the operator `-?/` in addition to name `relu_gate`. These built-in numeric operations are used to construct assignments (`Assignments.t` packaged as `Assignments.comp`). The syntax `%cd` is needed to build assignments concisely, and the assignment operators always start with `=` (unlike in C where they end with `=`). On the other hand, while the syntax `%op` helps build tensors (`Tensor.t`), they can be expressed concisely in pure OCaml. Unlike for assignments, the building blocks for tensor expressions are easy to extend. The meaningful basic ones are provided in `lib/operation.ml`.

In OCANNL, we call a tensor that is prohibited from propagating gradients, does not have a gradient node nor backprop code, a _non-differentiable tensor_. Accordingly we can call the "plain" tensors with a gradient node _differentiable tensors_. Expressions in the `%cd` syntax will sometimes build new non-differentiable tensors as components of assignments (they will never build new differentiable tensors). The syntax extensions make the following assumption:

- `%cd` assumes that any extension point will be in the scope of a module `NTDSL` that provides at least the functionality of `Operation.NTDSL`.
- `%op` assumes that any extension point will be in the scope of a module `TDSL` that provides at least the functionality of `Operation.TDSL`.

Functions inside `Operation.NTDSL` use `~grad_spec:Prohibit_grad` when calling into `Tensor`, making the resulting tensors non-differentiable. Functions inside `Operation.TDSL` use `~grad_spec:If_needed`, which will make the tensors non-differentiable when the gradient is not needed -- except for `TDSL.param`, which internally sets `~grad_spec:Require_grad`.

The extension points open `NTDSL.O`, resp. `TDSL.O`, for the scope of the extension point, to expose the corresponding operators.

## Primitive operations

To accomodate stylistic preferences, OCANNL supports both curried and uncurried syntaxes for primitive operation application. Binary operators are associated with infix operators, in addition to having alphabetic identifiers. This stems from the following restriction: in the `%cd` syntax, the assignment is always an infix operator, and it needs to pick the accumulation operation.

The unary primitive operations:

| Identifier | Default projection | Constructor in `Ir.Ops` |
|------------|--------------------|-------------|
| `id` | pointwise | `Identity` |
| `relu` | pointwise | `Relu` |
| `sat01` | pointwise | `Satur01` |
| `exp` | pointwise | `Exp` |
| `log` | pointwise | `Log` |
| `exp2` | pointwise | `Exp2` |
| `log2` | pointwise | `Log2` |
| `sin` | pointwise | `Sin` |
| `cos` | pointwise | `Cos` |
| `sqrt` | pointwise | `Sqrt` |
| `recip` | pointwise | `Recip` |
| `recip_sqrt` | pointwise | `Recip_sqrt` |
| `neg` | pointwise | `Neg` |
| `tanh` | pointwise | `Tanh_approx` |
| `not` | pointwise | `Not` |

The binary primitive operations:

| Identifier | Infix operator | Default projection | Constructor in `Ir.Ops` | Assignments |
|------------|----------------|--------------------|-------------|-------------|
| `fst` | `-@>` | pointwise | `Arg1` | none |
| `snd` | `-/>` | pointwise | `Arg2` | `=:` |
| `add` | `+` | pointwise | `Add` | `=+`, `=:+` |
| `sub` | `-` | pointwise | `Sub` | `=-`, `=:-` |
| `mul` | `*` | none | `Mul` | `=*`, `=:*` |
| `div` | `/` | none | `Div` | `=/`, `=:/` |
| `pow` | `**` | pointwise | `ToPowOf` | `=**`, `=:**` |
| `relu_gate` | `-?/` | pointwise | `Relu_gate` | `=?/`, `=:?/` |
| `sat01_gate` | `-?^` | pointwise | `Satur01_gate` | `=?^`, `=:?^` |
| `lt` | `<` | pointwise | `Cmplt` | none |
| `eq` | `=` | pointwise | `Cmpeq` | none |
| `ne` | `<>` | pointwise | `Cmpne` | none |
| `or_` | `\|\|` | pointwise | `Or` | `=\|\|`, `=:\|\|` |
| `and_` | `&&` | pointwise | `And` | `=&&`, `=:&&` |
| `mod_` | `%` | pointwise | `Mod` | none |
| `max` | `@^` | pointwise | `Max` | `=@^`, `=:@^` |
| `min` | `@-` | pointwise | `Min` | `=@-`, `=:@-` |
| `threefry4x32` | `^^^^` | pointwise | `Threefry4x32` | `=^^^^`, `=:^^^^` |

The ternary primitive operations:

| Identifier | Default projection | Constructor in `Ir.Ops` |
|------------|--------------------|-------------|
| `where` | pointwise | `Where` |
| `fma` | compose-accumulate | `FMA` |

The interpretation functions also state the semantics:

```ocaml
let interpret_unop op v =
  let open Float in
  match op with
  | Identity -> v
  | Relu when v >= 0. -> v
  | Relu -> 0.
  | Satur01 when v <= 0. -> 0.
  | Satur01 when v >= 1. -> 1.
  | Satur01 -> v
  | Exp -> exp v
  | Log -> log v
  | Exp2 -> 2. ** v
  | Log2 -> log v / log 2.
  | Sin -> sin v
  | Cos -> cos v
  | Sqrt -> sqrt v
  | Recip -> 1. / v
  | Recip_sqrt -> 1. / sqrt v
  | Neg -> ~-.v
  | Tanh_approx -> tanh v
  | Not -> if v = 0. then 1. else 0.

let interpret_binop op v1 v2 =
  let open Float in
  match op with
  | Arg1 -> v1
  | Arg2 -> v2
  | Add -> v1 + v2
  | Sub -> v1 - v2
  | Mul -> v1 * v2
  | Div -> v1 / v2
  | ToPowOf when is_integer v2 -> int_pow v1 @@ to_int v2
  | ToPowOf -> v1 ** v2
  | Relu_gate -> if v1 > 0.0 then v2 else 0.0
  | Satur01_gate -> if v1 > 0.0 && v1 < 1.0 then v2 else 0.0
  | Max -> max v1 v2
  | Min -> min v1 v2
  | Mod -> v1 % v2
  | Cmplt -> if v1 < v2 then 1. else 0.
  | Cmpeq -> if v1 = v2 then 1. else 0.
  | Cmpne -> if v1 <> v2 then 1. else 0.
  | Or -> if v1 <> 0. || v2 <> 0. then 1. else 0.
  | And -> if v1 <> 0. && v2 <> 0. then 1. else 0.

let interpret_ternop op v1 v2 v3 =
  let open Float in
  match op with Where -> if v1 <> 0. then v2 else v3 | FMA -> (v1 * v2) + v3
```

## The syntax for %op

 The `%op` syntax is simpler than the `%cd` syntax since it relies more on regular OCaml expressions. For example, we can write without syntax extensions:

```ocaml
  let hid_dim = 8 in
  let w = TDSL.param "w" in
  let b = TDSL.param ~output_dims:[ hid_dim ] "b" in
  let layer x = TDSL.O.( relu(w * x + b) ) in
  ...
```

Since `TDSL.O` is opened for the scope of an extension point `%op`:

```ocaml
  let hid_dim = 8 in
  let w = TDSL.param "w" in
  let b = TDSL.param ~output_dims:[ hid_dim ] "b" in
  let%op layer x = relu(w * x + b) in
  ...
```

Using [inline declarations](#inline-declarations), this becomes more concise:

```ocaml
  let hid_dim = 8 in
  let%op mlp_layer x = relu("w" * x + "b" hid_dim) in
  ...
```

When there is a function directly under the `%op` extension point, like in the example above, or directly under a function taking a `~config` parameter, the function parameter must be a tensor. That's because `%op` uses this tensor's (value's) label to enrich the label of the resulting tensor.

When the declaration is followed by a literal float, the float provides the initial value to initialize the tensor. Otherwise, the tensor value cells are initialized randomly with uniform distribution.

## The syntax for %cd

The basic building blocks of the `%cd` syntax are individual assignments, separated by semicolons. The assignments, represented via `Assignments.Accum_binop` and `Assignments.Accum_unop`, are in full generality accumulating:

```ocaml
type Assignments.t =
   ...
  | Accum_binop of {
      initialize_neutral : bool;
      accum : Ops.binop;
      op : Ops.binop;
      lhs : Tnode.t;
      rhs1 : buffer;
      rhs2 : buffer;
      projections : Indexing.projections Lazy.t;
    }
  | Accum_unop of {
      initialize_neutral : bool;
      accum : Ops.binop;
      op : Ops.unop;
      lhs : Tnode.t;
      rhs : buffer;
      projections : Indexing.projections Lazy.t;
    }
```

For example the binary case in pseudocode: `if initialize_neutral then lhs = 0; lhs = lhs accum (rhs1 op rhs2)` (assuming the neutral element of `accum` is 0). The representation also has a field `projections` which determines which loops should be run and how the tensor nodes should be indexed to perform the computation.

The basic `%cd` syntax for assignments has the form: `<lhs> <asgn-op> <primitive-op-application[rhs1, rhs2?, rhs3?]>`. See [Primitive operations](#primitive-operations) for the syntax of primitive operation application, where `<rhs1>`, `<rhs2>` (for binary and ternary ops), `<rhs3>` (for ternary ops) are subexpressions. `<asgn-op>` starts with `=`, followed by `:` only if `initialize_neutral` is true, then followed by the operator syntax variant of a binary primitive operation. The fields `<lhs>`, `<rhs1>`, `<rhs2>`, `<rhs3>` will often be either special-purpose identifiers (specifically `v`, `t`, `t1`, `t2`, `t3`, `g`, `g1`, `g2`, `g3`) or identifiers bound to tensors. `<rhs1>`, `<rsh2>`, `<rsh3>` will also often be (non-differentiable) tensor expressions. The notation `<tensor>.grad` stands for the gradient node of the given tensor. For more about "slot fillers", and to learn about the operators `*+` and `++`, see the section [further features of the syntax extension %cd](#further-features-of-the-syntax-extension-cd).

How is the `projections` field determined? `projections` can be given explicitly as a labeled argument `~projections`. If they aren't but `%cd` realizes there is a `~projections` parameter in scope, it uses it -- see `lib/operation.ml` where this option is used to define tensor operations. If instead of `~projections` a `~logic` labeled argument is given, the string passed is used to determine projections. `~logic:"."` means a pointwise operation. `~logic:"@"` means an "output axes of rhs2 match input axes of rhs1" operation (matrix multiplication is a special case). `~logic:"T"` means transpose of input and output axes. The string passed to `~logic` can also use OCANNL's generalization of the einsum notation, allowing arbitrary permutations and reductions of axes. If no information is given, the default depends on the primitive operation, but it is almost always a pointwise operation.

Here we see an example of tensor multiplication -- extending matrix multiplication to arbitrary number of axes -- multiplying `a` by `b` to get `c`. In `=:+`, `=` is required to separate the assigned-to part from the computation, `:` clears-out `c` before the computation, `+` selects addition to accumulate the results.

```ocaml
c =:+ a * b ~logic:"@"
```

Compare the following two ways of updating a parameter `p`:

```ocaml
p =+ learning_rate * p.grad ~logic:"."
```

and:

```ocaml
p =+ learning_rate *. p.grad
```

In the first case, we have a binary assignment calculated pointwise. The resulting representation is `Accum_binop` where `accum` is `Add` and `op` is `Mul` (multiplication). In the second case, `*.` is not recognized as one of the built-in operators. This leaves the expression `learning_rate *. p.grad` un-transformed. Since `(*.)` is bound in `NTDSL.O` to pointwise tensor multiplication, this creates an intermediate tensor, that is then added onto p. The resulting representation is `Accum_unop` where `accum` is `Add` and `op` is `Identity`. Both variants end up with the same result, and even with the same computation, because the second variant's computation will get optimized (unless configured not to).

Advanced note: when a `~projections` parameter is in scope but no assignment-specific `~projections` argument is given -- the typical case in `lib/operation.ml` -- the actual projections field for an assignment is computed by transforming the projections parameter according to hints regarding how tensor nodes relate to the given projections. Specifically, the identifiers `rhs1`, `t1`, `v1`, `g1` are "slot RHS1" of the projections, `rhs2`, `t2`, `v2`, `g2` are "slot RHS2", `lhs,`, `t`, `v`, `g` are "slot LHS". Scalar constants are provided the projection directly, to make the automated derivation more expressive; this is supported both for literals, and (heuristically) for `!.` and `!..` embedding operators.

## Numeric and N-dimensional array literals

Both `%cd` and `%op` extensions use a shared syntax for N-dimensional array literals. `%cd` uses `NTDSL.number` and `NTDSL.ndarray` functions, while `%op` uses `TDSL.number` and `TDSL.ndarray` functions. (This is just for consistency: `TDSL.ndarray` invokes `Tensor.ndarray ~grad_spec:If_needed`, which will figure out the gradient is not needed and will make the tensor non-differentiable.)

Numbers are a special case: an array of (output) dimension 1.

N-dimensional array literals combine the list, tuple and array syntaxes to strictly distinguish between output, input and batch axes:

- The tuple syntax translates to an input axis.
- The list syntax translates to an output axis.
- The array syntax translates to a batch axis.

For example, `[ (1, 2, 3); (4, 5, 6) ]` is a mathematical matrix converting 3D vectors into 2D vectors.

OCANNL supports dimension labels. The syntax for number allows prefixing a number by a character that stands for the dimension label of the resulting output dimension 1. These labels can then propagate to specify labels of other dimensions in other tensors, via shape inference. Example: `let%op y = ("hey" * 'q' 2.0) + 'p' 1.0 in ...`

## Wildcard bindings

When an extension is over a wildcard (ignore result) binding: `let%cd _ = ...` and `let%op _ = ...`, the generated code is wrapped in `Tensor.with_unchanged_roots`, to prevent it from upsetting rootness checks. The use-case for writing `%op` and `%cd` notations with ignored result is to generate additional shape inference constraints.

## Inline declarations

Both `%cd` and `%op` syntaxes support inline declarations of tensors. For `%op` these are differentiable, for `%cd` non-differentiable tensors. A declaration site uses the string syntax, the content of the string is the is bound to the newly created tensor, and the string itself functions equivalently to using the newly introduced identifier. The scope of the binding is the full scope of the extension point, even if the declaring string appeared in the body of a function that's inside the extension point scope (except for `%op` there is a special case of `~config` labeled argument discussed below). The first element of the label of the created tensor is the string that introduced it.

For `%cd`, inline declarations are allowed both in the assigned-to position (left-hand side) of assignments and in standalone tensor expressions. When used in assignments, one of the tensors on the right-hand-side is picked to provide additional label information if possible. In particular, tensors that are function parameters inside the scope of the extension point, cannot be picked to provide label information, as they would escape their scope at the point the tensor is created. Inline declarations are still prohibited within the right-hand side of assignments to discourage over-use in locations with less label information. Example showing two tensor nodes declared inline, both of them include the label of the param `p` in their labels:

```ocaml
let sgd_one ~learning_rate ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) p =
  [%cd
    "sgd_delta" =: p.grad + (!.weight_decay *. p);
    if Float.(momentum > 0.0) then (
      "sgd_momentum" =: (!.momentum *. sgd_momentum) + sgd_delta;
      if nesterov then sgd_delta =+ !.momentum *. sgd_momentum else sgd_delta =: sgd_momentum);
    p =- learning_rate *. sgd_delta]
```

Inline declarations can also be used outside of assignments for creating non-differentiable tensors, to mimic the behavior of `%op` but without the burden of initialization that a parameter would introduce:

```ocaml
  let%cd mlp_result = mlp "point" in
  let result_routine =
    Train.to_routine (module Backend) sgd_routine.context IDX.empty
      [%cd ~~("mlp infer"; mlp_result.forward)]
  in
  let callback (x, y) =
    Tn.set_values point [| x; y |];
    Train.run result_routine;
    Float.(mlp_result.@[0] >= 0.)
  in
```

For `%op`, the declaration is allowed anywhere. If there is a `~config` function parameter used inside the extension scope, for example as `fun ~config ... -> ...` or a more specific example `let%op mlp ~config x = ...`, the scope of an inline-declared tensor is no longer the full scope of the extension point. Instead, the tensor is defined right underneath the introduction of the `~config` parameter: `fun ~config -> let <definitions of the inline-declared tensors> in ...`. The config value passed to the generated code must be a record with at least a field `label : string list`. The inline-declared tensor that's defined under a `~config` parameter is defined as `TDSL.param ~more_label:config.label ...` Example showing two param tensors declared inline, including `config.label` in their labels:

```ocaml
type mlp_layer_config = { label : string list; hid_dim : int }

let%op mlp_layer ~config x = relu ("w" * x + "b" config.hid_dim)
```

## Using OCANNL's generalized einsum notation

As we mentioned above, in the `%cd` syntax you can set up an arbitrary assignment with projections derived from a generalized einsum specification, by passing the specification as a string with the `~logic` label. However, both the `%cd` and `%op` syntaxes support built-in operators that take an einsum specification: `*+` binding to `NTDSL.einsum` resp. `TDSL.einsum`, and `++` binding to `NTDSL.einsum1` resp. `TDSL.einsum1`. `*+` is a "ternary" operator, binary wrt. tensor arguments, and `++` is a binary operator, unary postfix wrt. tensor arguments. The einsum specification string should directly follow `*+` and `++`.

Both `*+` and `++` use addition for the accumulation operation; `*+` uses multiplication. You can verify that looking at the `Operation.einsum` and `Operation.einsum1` definitions. You can find examples of `*+` and `++` behavior in the test suite [einsum_trivia.ml](test/einsum_trivia.ml). A frequent use-case for `++` is to sum out all axes of a tensor:

```ocaml
  let%op scalar_loss = (margin_loss ++ "...|... => 0") /. !..batch_size in
  ...
```

where `(!..)` converts an integer into a constant tensor.

### Syntax of the generalized einsum notation

The specification syntax has two modes:

- if there is a comma anywhere in a spec, it is the _multichar mode_: axis identifiers are comma-separated and can have multiple characters;
- otherwise, it is the _single-char mode_: each alphanumeric character corresponds to an axis.

The syntax of a generalized einsum spec has two variants:

- unary: "\<rhs\> shape spec `=>` \<lhs\> shape spec", specifies a unary assignment `<lhs> <asgn-op> <rhs>` (see [syntax for `%cd`](#the-syntax-for-cd)),
- binary: "\<rhs1\> shape spec `;` \<rhs2\> shape spec `=>` \<lhs\> shape spec", specifies a binary assignment `<lhs> <asgn-op> <rhs1> <op> <rhs2>` (see [syntax for `%cd`](#the-syntax-for-cd)).

Recall that a tensor _shape_ is composed of three _rows_, i.e. sequences of axes: batch, input and output axes. Correspondingly, a shape spec in the notation can be:

- the output row at the end of the spec, or just the output row,
- the input row to the left of `->`, if given,
- the batch row to the left of `|`, if given.

The notation for a row is composed of sequences of row specs, and an optional _row variable_ spec. A row variable tracks broadcasting. The syntax of a row:

- a sequence of axis specs: specifies the rightmost axes, with untracked broadcasting "to the left",
- a row variable spec followed a sequence of axis specs for the rightmost axes,
- leftmost axes specs, followed by a row variable, followed by rightmost axes specs.

The syntax of a row variable:

- `..`variable_id`..`: variable_id stands for the row variable identifier,
- ellipsis `...` is context dependent: in the batch row it means `..batch..`, in the input row `..input..`, in the output row `..output..`.

The syntax of an axis spec:

- Depending on the mode, either a alphabetic character or an alphanumeric identifier provides an axis variable.
- Dhe underscore `_` is a placeholder to align other axes, but does not specify anything for the given axis (it is not a variable).
- A number specifies the particular dimension within the axis,
- A `+` sign specifies a convolution input axis with the output on the left of `+` and the kernel on the right of `+`.
  - In both the output part and the kernel part you can prefix the axis variable by a constant coefficient with the `*` sign.
  - The coefficient can directly only be an integer, e.g. `"2*i+3*k"`, but under the `%op` and `%cd` syntax extensions, it can also be an identifier of an integer value, e.g. `let stride = 2 and dilation = 3 in [%op "input" *+ "stride * a + dilation * b; b=>a," "kernel"]`.
  - Note the comma above. The syntax extension's expansion of stride and dilation respects the "multichar" mode. Without the comma we are limited to single-character identifiers, e.g. `let s = 2 and d = 3 in [%op "input" *+ "is*a+d*bc;b=>iac" "kernel"]`.

Examples:

- `...|...->... => 0`: reduce all axes of the argument into a single number. Useful e.g. for reducing losses to a single number.
- `...|... => 0`, `...->... => 0`, `... => 0` do the same but will fail if the argument has axes of the kind for which the ellipsis is missing.
- `...|...->... => ...|...->...`: fully pointwise unary operation.
- `...->... => ...->...`, `...|... => ...|...`, `... => ...`: fully pointwise but will fail if the argument has axes of the kind for which the ellipsis is missing.
- `...|...->... ; ...|...->... => ...|...->...`: fully pointwise binary operation.
- `...|...->... => ...->...`: reduce the batch axes into the result.
- `2...|...->... => ...|...->...`: slice the tensor at dimension 2 of the leftmost batch axis. Note that the tensor operation `@|` implements slicing at the leftmost batch axis for arbitrary dimension.
- `...|... => ...|...2`: expand the tensor by putting the argument at leftmost output dimension 2 of the result (and reduce input axes if any). `rhs ++ "...|... => ...|...2"` will fill the other cells of the new tensor with zeroes; `[%cd lhs =:* rhs ~logic:"...|... => ...|...2"]` will fill the other cells of `lhs` with ones since it's the neutral element of the assignment (reduction) operator, here with ones.
- `ijk => kji`: reverse the three output axes, fails if the argument has any other axes.
- `ijk => ki`: as above but also reduce the second-leftmost output axis.
- `..v..|...ijk => ..v..kji`: reverse the three rightmost output axes, reduce any other output axes, pointwise for batch axes, pairing the batch axes with the leftmost output axes of the result. Fails if the argument has input axes.
- `2..v..|... => ..v..`: slice the tensor at dimension 2 of the leftmost batch axis, reduce all its output axes, preserve its other batch axes as output axes. Fails if the argument has input axes.

## Further features of the syntax extension %cd

### Referencing arrays: tensor value, tensor gradient, merge buffer of a tensor node

The `%cd` syntax uses record-style notation to point to:

- the value tensor node of a tensor `<tensor>.value`,
- the gradient tensor node of a tensor `<tensor>.grad`,
- the merge buffer of a tensor node `<tensor-node>.merge`; `<tensor>.merge` is a shorthand for `<tensor>.value.merge`.

The accessor `.value` can (almost?) always be dropped: by default, tensors in the `%cd` syntax refer to their value nodes.

For example, in a data-parallel computation, gradients of the same param `p` can be merged across devices using the code `p.grad =+ p.grad.merge`, combined with an explicit device-to-device transfer.

### Block comments

The `%cd` syntax uses the prefix operator `(~~)` in a semicolon sequence to introduce block comments:

```ocaml
type Assignments.t =
  ...
  | Block_comment of string * t
  ...
```

 Schematic example: `~~("space" "separated" "comment" "tensor p debug_name:" p; <scope of the comment>)`. The content of the comment uses application syntax, must be composed of strings, `<tensor>`, `<tensor>.value` (equivalent to `<tensor>`), `<tensor>.grad` components, where `<tensor>` is any tensor expression or tensor identifier.

## Further features of the syntax extension %op

### Name from binding

When an extension point is applied to a let-binding, e.g. `let%op mlp_layer ~config x = relu ("w" * x + "b" config.hid_dim)`, it uses the name of the binding (`mlp_layer` in the example) for the label of the primary tensor created by the extension, if any. This is why the resulting layer tensor in the example has its label starting with `"mlp_layer"`. If the extension is over a semicolon-separated sequence of expressions, the primary tensor can only be in the last component of the sequence, other syntax constructs are handled analogously.

### Label from function argument

The resulting (primary) tensor's label will also have incorporated the label of the input argument, if any. In our example, the resulting `mlp_layer` tensor will also include the label of the actually applied `x`.

Note that we do not include `config.label`, even if `config` is available, because the actually applied input argument will typically have more specific information.

### Configuring inline declarations: inline output dimensions, initial values

In the `%op` syntax, when a tuple follows an inline declaration of a tensor (i.e. a string literal), the tuple is passed to specify the output axes in the tensor definition (via the `~output_dims` argument).

When it is an integer, an identifier, or a record field dereference following an inline declaration, this expression specifies the single output axis in the tensor definition. You can see an example above in this document: `let%op mlp_layer ~config x = relu ("w" * x + "b" config.hid_dim)`.

If it is a list expression following an inline declaration, the expression is parsed as an [N-dimensional array constant](#numeric-and-n-dimensional-array-literals), and used to initialize the value tensor node of the defined tensor. A very simple example from [micrograd_demo: Micrograd README basic example](test/micrograd_demo.ml):

```ocaml
  let%op c = "a" [ -4 ] + "b" [ 2 ] in
  ...
```

### Lifting of the applications of config arguments: if an error, refactor your code

If you recall, inline declared param tensors get lifted out of functions except for the function `fun ~config ->`, where they get defined. Our example `let%op mlp_layer ~config x = relu ("w" * x + "b" config.hid_dim)` translates as:

```ocaml
let mlp_layer ~config =
  let w = TDSL.param "w" and b = TDSL.param ~output_dims:[ config.hid_dim ] "b" in
  fun x -> TDSL.O.(w * x + b)
```

For this to work properly, when employing such network blocks, their params also need to be introduced at the right moment. Therefore, the `%op` syntax ensures that this example:

```ocaml
type tlp_config = { label : string list; dim1 : int; dim2 : int; dim3 : int }

let%op three_layer_perceptron ~config x =
  mlp_layer ~config:{ label = [ "L3" ]; hid_dim = config.dim3 }
    (mlp_layer ~config:{ label = [ "L2" ]; hid_dim = config.dim2 }
       (mlp_layer ~config:{ label = [ "L1" ]; hid_dim = config.dim1 } x))
```

gets expanded to:

```ocaml
type tlp_config = { label : string list; dim1 : int; dim2 : int; dim3 : int }

let three_layer_perceptron ~config =
  let config_block__1 = mlp_layer ~config:{ label = [ "L3" ]; hid_dim = config.dim3 }
  and config_block__2 = mlp_layer ~config:{ label = [ "L2" ]; hid_dim = config.dim2 }
  and config_block__3 = mlp_layer ~config:{ label = [ "L1" ]; hid_dim = config.dim1 } in
  fun x -> config_block__1 (config_block__2 (config_block__3 x))
```

However, this raises a concern for more complex situations. Consider this code that fails to compile:

```ocaml
type mlp_config = { label : string list; hid_dims : int list }

let%op mlp ~config x =
  List.foldi config.hid_dims ~init:x ~f:(fun i x hid_dim ->
      mlp_layer ~config:{ label = [ "L" ^ Int.to_string i ]; hid_dim } x)
```

The attempted lifting breaks because of the escaping variables `i` and `hid_dim`. This reminds us to rewrite the example, ensuring the proper introduction of params:

```ocaml
type mlp_config = { label : string list; hid_dims : int list }

let mlp ~config =
  let layers =
    List.mapi config.hid_dims ~f:(fun i hid_dim ->
        mlp_layer ~config:{ label = [ "L" ^ Int.to_string i ]; hid_dim })
  in
  fun x -> List.fold layers ~init:x ~f:(fun x layer -> layer x)
```

Unfortunately, we need to be mindful to introduce params at the right times.

## Implementation details

### The hard-coded to-the-power-of operator

OCANNL has a built-in numerical binary operation to-power-of: `Ops.ToPowOf`. As part of assignments, the corresponding operator is `**`. Here is the full definition of the to-power-of tensor operation from [Operation](lib/operation.ml):

```ocaml
let rec pointpow ?(label : string list = []) ~grad_spec p t1 : Tensor.t =
  let module NTDSL = struct
    include Initial_NTDSL

    module O = struct
      include NDO_without_pow

      let ( **. ) ?label base exp = pointpow ?label ~grad_spec:Tensor.Prohibit_grad exp base
    end
  end in
  let p_t = NTDSL.number p in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 ** v2 ~projections in
  let%cd grad_asn =
    if Tensor.is_prohibit_grad grad_spec then fun ~v:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ -> Asgns.Noop
    else if Float.equal p 2.0 then fun ~v:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ p_t *. t1 * g
    else if Float.equal p 1.0 then fun ~v:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ g
    else fun ~v:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ p_t *. (t1 **. (p -. 1.)) * g
  in
  Tensor.binop ~label:("**." :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn ~grad_spec t1 p_t
```

On the `Tensor` level, this is implemented as a binary tensor operation, but it is exposed as a unary tensor operation! To avoid the complexities of propagating gradient into the exponent, `Operation.pointpow` is implemented as a function of only one tensor, the exponent is a number. We hard-code the pointwise-power-of operator `NTDSL.O.( **. )`, resp. `TDSL.O.( **. )`, in the `%cd` and `%op` syntaxes, to pass the numeric value to `pointpow` (the second argument of `**.`) without converting it to a tensor first.

### Intricacies of the syntax extension %cd

The syntax `%cd` translator needs to accomplish more than a context-free conversion of a concise notation to an `Assignments.comp` data-type. In particular:

- It needs to keep track if `~projections` is in scope, and it needs to collect the information about an assignment to properly transofm the projections from the scope into the projections valid for the particular assignment.
- Whenever the parsed notation uses tensors whose value nodes have not been computed yet, the translator needs to include the "forward" code of the tensors among the generated assignments. Typically this is required for embedded tensor expressions, which create new tensors. The translator puts the forward code in sequence just prior to the assignment that made use of the created tensor. The translator includes the forward code of tensors that are "forward roots" at the time the assigments are constructed (using `Tensor.is_fwd_root`).
- For inline declarations of tensors, the translator needs to pick the right other tensor, if any, to enrich the label information of the created tensor. Mechanisms:
  - Prefer tensors from identifiers (or field dereferences), since labels of tensor expressions (creating new tensors) will typically be overly verbose.
  - Filter out escaping variables (identifiers coming from nested function parameters).
  - When one inline declaration uses another inline declaration on its right-hand-side, recall the other declaration's label-enriching-tensor and use it directly.
- The argument slots in `Assignments.Accum_binop` and `Assignments.Accum_unop` can be either regular tensor nodes, or merge buffers of tensor nodes. The translator needs to determine that.
- When a tensor expression is used to create a new tensor, the translator lifts the expression into a let-binding, to be able to refer to the (same) tensor more than once. The created tensor is referred to at least twice: at its use site, and to include its forward code among the assignments.

### Embedded nodes

In fact, the syntax `%cd` produces `Assignments.comp` values:

```ocaml
type comp = {
  asgns : t;
  embedded_nodes : Set.M(Tnode).t;
}
```

The tensor nodes that are in `asgns` but not in `embedded_nodes`, and are on-device, must already be present in contexts with which the computation is linked. Such non-embedded nodes can be seen as inputs to the computation -- except that for `backprop` code of a tensor, they are actually the outputs! Embedded nodes are closely related to _rootness_ -- when a node has not been used in the code of another tensor, it is a root (a forward root for value nodes and a backprop root for grad nodes). `embedded_nodes` were roots the first time they were used in `asgns`. Parameters, as created by `Tensor.param`, are not embedded in the code that uses them and thus will not be in `embedded_nodes` of the forward and backprop code over the parameters; however, they will constitute the `embedded_nodes` of the `Tensor.init_params` code.

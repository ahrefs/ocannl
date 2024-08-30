# Syntax extensions `%cd` and `%op` {#syntax-extensions-cd-and-op}

- Table of contents
  - [Preliminaries](#preliminaries)
  - [The syntax for %op {#syntax-for-op}](#syntax-for-op)
  - [The syntax for %cd](#syntax-for-cd)
  - [Numeric and N-dimensional array literals](#numeric-and-n-dimensional-array-literals)
  - [Wildcard bindings](#wildcard-bindings)
  - [Inline declarations](#inline-declarations)
  - [Using OCANNL's generalized einsum notation](#using-ocannls-generalized-einsum-notation)
  - [Further features of the syntax extension %cd](#features-of-syntax-cd)
    - [Referencing arrays: tensor value, tensor gradient, merge buffer of a tensor node](#referencing-arrays-tensor-value-tensor-gradient-merge-buffer-of-a-tensor-node)
    - [Block comments](#block-comments)
  - [Further features of the syntax extension %op](#features-of-syntax-op)
    - [Name from binding](#name-from-binding)
    - [Label from function argument](#label-from-function-argument)
    - [Lifting of the applications of ~config arguments: if it's an error, refactor your code](#lifting-of-the-applications-of-config-arguments-if-its-an-error-refactor-your-code)
  - [Implementation details](#implementation-details)
    - [Syntax extension %cd](#implementation-extension-cd)
    - [Syntax extension %op](#implementation-extension-op)
- In a nutshell
  - Syntax extension `%cd` stands for "code", to express assignments: `Assignments.t`.
  - Syntax extension `%op` stands for "operation", to express tensors: `Tensor.t`.

## Preliminaries

OCANNL, and Arrayjit specifically, is built around a fixed number of numeric operations, declared in `arrayjit/ops.ml`. We assign operators to many of the operations, inventing new operators. For example, Rectified Linear Unit `Relu` operation, which computes `f(x) = max(0,x)`, gets the operator `!/`, and the ReLU-Gate `Relu_gate` operation, which computes `f(x,y) = if x > 0.0 then y else 0.0`, gets the operator `-?/`. These built-in numeric operations are used to construct assignments (`Assignments.t`). The syntax `%cd` is needed to build assignments concisely. On the other hand, while the syntax `%op` helps build tensors (`Tensor.t`), they can be expressed concisely in pure OCaml. Unlike for assignments, the building blocks for tensor expressions are easy to extend. The meaningful basic ones are provided in `lib/operation.ml`.

In OCANNL, we call a tensor that is prohibited from propagating gradients, does not have a gradient node nor backprop code, a _non-differentiable tensor_. Accordingly we can call the "plain" tensors with a gradient node _differentiable tensors_. Expressions in the `%cd` syntax will sometimes build new non-differentiable tensors as components of assignments (they will never build new differentiable tensors). The syntax extensions make the following assumption:

- `%cd` assumes that any extension point will be in the scope of a module `NTDSL` that provides at least the functionality of `Operation.NTDSL`.
- `%op` assumes that any extension point will be in the scope of a module `TDSL` that provides at least the functionality of `Operation.TDSL`.

Functions inside `Operation.NTDSL` use `~grad_spec:Prohibit_grad` when calling into `Tensor`, making the resulting tensors non-differentiable. Functions inside `Operation.TDSL` use `~grad_spec:If_needed`, which will make the tensors non-differentiable when the gradient is not needed -- except for `TDSL.param`, which internally sets `~grad_spec:Require_grad`.

The extension points open `NTDSL.O`, resp. `TDSL.O`, for the scope of the extension point, to expose the corresponding iterators.

## The syntax for `%op` {#syntax-for-op}

 The `%op` syntax is simpler than the `%cd` syntax since it relies more on regular OCaml expressions. For example, we can write without syntax extensions:

```ocaml
  let hid_dim = 8 in
  let w = Tensor.param "w" in
  let b = Tensor.param ~output_dims:[ hid_dim ] "b" in
  let layer x = TDSL.O.( !/(w * x + b) ) in
  ...
```

Since `TDSL.O` is opened for the scope of an extension point `%op`:

```ocaml
  let hid_dim = 8 in
  let w = Tensor.param "w" in
  let b = Tensor.param ~output_dims:[ hid_dim ] "b" in
  let%op layer x = !/(w * x + b) in
  ...
```

Using [inline declarations](#inline-declarations), this becomes more concise:

```ocaml
  let hid_dim = 8 in
  let%op mlp_layer x = !/("w" * x + "b" hid_dim) in
  ...
```

## The syntax for `%cd` {#syntax-for-cd}

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

The basic `%cd` syntax for binary operator assignments has the form: `<lhs> <asgn-op> <rhs1> <op> <rhs2>` (or `<lhs> <asgn-op> <op> <rhs1> <rhs2>` when `<op>` is not an operator). The binary operators in the `<rhs1> <op> <rhs2>` part have a straightfowrad syntax: `<op>` is one of `+`, `-`, `*`, `/`, `**` (to-power-of), `-?/` (ReLU-Gate). `<asgn-op>` starts with `=`, followed by `:` only if `initialize_neutral` is true, then followed by one of `+`, `-`, `*`, `/`, `**`, `?/`. The fields `<lhs>`, `<rhs1>`, `<rhs2>` will often be either special-purpose identifiers (e.g. `t`, `t1`, `t2`, `g`, `g1`, `g2`) or identifiers bound to tensors. `<rhs1>`, `<rsh2>` will also often be (non-differentiable) tensor expressions. The notation `<tensor>.grad` stands for the gradient node of the given tensor. For more about "slot fillers", and to learn about the operators `*+` and `++`, see the section [further features of the syntax extension %cd](#features-of-syntax-cd).

How is the `projections` field determined? `projections` can be given explicitly as a labeled argument `~projections`. If they aren't but `%cd` realizes there is a `~projections` parameter in scope, it uses it -- see `lib/operation.ml` where this option is used to define tensor operations. If instead of `~projections` a `~logic` labeled argument is given, the string passed is used to determine projections. `~logic:"."` means a pointwise operation. `~logic:"@"` means an "output axes of rhs2 match input axes of rhs1" operation (matrix multiplication is a special case). `~logic:"T"` means transpose of input and output axes. The string passed to `~logic` can also use OCANNL's generalization of the einsum notation, allowing arbitrary permutations and reductions of axes. If no information is given, the default is a pointwise operation.

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

Advanced note: when a `~projections` parameter is in scope but no assignment-specific `~projections` argument is given -- the typical case in `lib/operation.ml` -- the actual projections field for an assignment is computed by transforming the projections parameter according to hints regarding how tensor nodes relate to the given projections. Specifically, the identifiers `rhs1`, `t1`, `v1`, `g1` are "slot RHS1" of the projections, `rhs2`, `t2`, `v2`, `g2` are "slot RHS2", `lhs,`, `t`, `v`, `g` are "slot LHS".

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

For `%cd`, the declaration is (currently) only allowed on the left-hand-side, i.e. in the assigned-to position, of an assignment. If possible, one of the tensors on the right-hand-side is picked to provide additional label information. In particular, tensors that are function parameters inside the scope of the extension point, cannot be picked to provide label information, as they would escape their scope at the point the tensor is created. Example showing two tensor nodes declared inline, both of them include the label of the param `p` in their labels:

```ocaml
let sgd_one ~learning_rate ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) p =
  [%cd
    "sgd_delta" =: p.grad + (!.weight_decay *. p);
    if Float.(momentum > 0.0) then (
      "sgd_momentum" =: (!.momentum *. sgd_momentum) + sgd_delta;
      if nesterov then sgd_delta =+ !.momentum *. sgd_momentum else sgd_delta =: sgd_momentum);
    p =- learning_rate *. sgd_delta]
```

For `%op`, the declaration is allowed anywhere. If there is a `~config` function parameter used inside the extension scope, for example as `fun ~config ... -> ...` or a more specific example `let%op mlp ~config x = ...`, the scope of an inline-declared tensor is no longer the full scope of the extension point. Instead, the tensor is defined right underneath the introduction of the `~config` parameter: `fun ~config -> let <definitions of the inline-declared tensors> in ...`. The config value passed to the generated code must be a record with at least a field `label : string list`. The inline-declared tensor that's defined under a `~config` parameter is defined as `TDSL.param ~more_label:config.label ...` Example showing two param tensors declared inline, including `config.label` in their labels:

```ocaml
type mlp_layer_config = { label : string list; hid_dim : int }

let%op mlp_layer ~config x = !/ ("w" * x + "b" config.hid_dim)
```

## Using OCANNL's generalized einsum notation

## Further features of the syntax extension `%cd` {#features-of-syntax-cd}

### Referencing arrays: tensor value, tensor gradient, merge buffer of a tensor node

The `%cd` syntax uses record-style notation to point to:

- the value tensor node of a tensor `<tensor>.value`,
- the gradient tensor node of a tensor `<tensor>.grad`,
- the merge buffer of a tensor node `<tensor-node>.merge`; `<tensor>.merge` is a shorthand for `<tensor>.value.merge`.

The accessor `.value` can (almost?) always be dropped: by default, tensors in the `%cd` syntax refer to their value nodes.

For example, in a data-parallel computation, gradients of the same param `p` can be merged across devices using the code `p.grad =+ p.grad.merge`, combined with an explicit device-to-device transfer.

### Block comments

## Further features of the syntax extension `%op` {#features-of-syntax-op}

### Name from binding

When an extension point is applied to a let-binding, e.g. `let%op mlp_layer ~config x = !/ ("w" * x + "b" config.hid_dim)`, it uses the name of the binding (`mlp_layer` in the example) for the label of the primary tensor created by the extension, if any. This is why the resulting layer tensor in the example has its label starting with `"mlp_layer"`. If the extension is over a semicolon-separated sequence of expressions, the primary tensor can only be in the last component of the sequence, other syntax constructs are handled analogously.

### Label from function argument

The resulting (primary) tensor's label will also have incorporated the label of the input argument, if any. In our example, the resulting `mlp_layer` tensor will also include the label of the actually applied `x`.

Note that we do not include `config.label`, even if `config` is available, because the actually applied input argument will typically have more specific information.

### Lifting of the applications of `~config` arguments: if it's an error, refactor your code

If you recall, inline declared param tensors get lifted out of functions except for the function `fun ~config ->`, where they get defined. Our example `let%op mlp_layer ~config x = !/ ("w" * x + "b" config.hid_dim)` translates as:

```ocaml
let mlp_layer ~config =
  let w = Tensor.param "w" and b = Tensor.param ~output_dims:[ config.hid_dim ] in
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

### Syntax extension `%cd` {#implementation-extension-cd}

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

### Syntax extension `%op` {#implementation-extension-op}

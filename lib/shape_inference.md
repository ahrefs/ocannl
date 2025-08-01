# Shape inference and projection inference

To separate concerns, OCANNL is split into the `arrayjit` library, responsible for compilation of high-level n-D array operation sequences (`Assignments.comp`) via the gccjit and cuda backends, and the main `ocannl` library, responsible for deriving the operations computing the forward propagation and backpropagation from tensor expressions. In particular, `arrayjit` contains `Indexing`, which represents complex indexing into arrays, and the main library `ocannl` has `Row` and `Shape` modules, which do the most "heavy-lifting" in the translation from concise tensor expressions to sequences of assignments.

Shape inference broadly speaking consists in OCANNL of inferring the `Shape.t` record -- shape inference proper, and inferring the `Indexing.projections` record -- projections inference. `Shape.t` records are mutable, so that the partially inferred shapes can be observed by the user. Shape and projections inference is intended to be declarative -- independent of the order in which constraints are added. There is one aspect that is not declarative: when tensor expressions are compiled to assignments, i.e. jitted, still-unsolved shape variables in terminal nodes are substituted by their least upper bounds if any, or by dimension-1 / no-more-axes.

The bulk of the projections inference happens alongside shape inference, with the projections-relevant information stored in auxiliary fields -- this prevents subtle bugs where projection semantics deviates from shape semantics, and will simplify adding new shape/projection inference features. Shape inference happens during `propagate_shapes` calls, and then again in a `finish_inference` call, which is triggered whenever the dimensions or projections are required (i.e. typically by jitting). Finally, the projections are reconstructed in `derive_projections`. It would seem `derive_projections` could reuse the already-computed solutions constraints. But we face a problem: we must prevent contaminating projections across different operations. To illustrate: we conclude the dimensions of two axes are the same because they are reduced together in another operation -- this should not force the axes to share a projection in the processed operation. To prevent the contamination, in each `derive_projections` call, we freshen the projection ids in the (inferred) shapes, and regenerate and re-solve the constraints with the fresh projection ids.

The shape system in OCANNL is currently monomorphic: both row and dimension variables are interpreted existentially. It can in principle be made polymorphic: by abstracting over the remaining fresh variables when forming a tensor-producing function, and by replacing universally bound variables by fresh variables when applying such functions. However, this is non-trivial and would depend on introducing namespaces for tensor nodes. Then, we could perform "abstract interpretation" (aka. tracing like e.g. in JAX) by computing an OCaml function under an abstract tensor node namespace. Applying the function would not execute the OCaml code again, but instead would copy the tensors generated by "abstract interpretation"-stage execution with appropriately freshened shape variables into a concrete tensor node namespace. There is a natural context for introducing such abstraction: the special `~config` labeled functions as processed by the `%op` syntax extension -- see [syntax extensions](./syntax_extensions.md). Exploring this is left as potential future work (no earlier than OCANNL v2).

## Representing shapes and constraints

A tensor shape in OCANNL is composed of three rows of axes: batch, input and output. These are ordered input-last (`batch @ output @ input`) in the underlying n-dimensional array implementation of tensors (at least when hosted, as backends can reorder axes via a stride mechanism NOTE: NOT IMPLEMENTED YET). A tensor shape can have empty axes, which represent scalars (tensors with a single element). This allows shape inference to naturally handle corner cases where tensors hold just one cell. For printing and einsum-notation-like specifications, we use the syntax: `batch|input->output` (or `input->output`, `batch|output`, `output`), where `batch`, `input`, `output` are whitespace or comma or parenthesis separated axis entries; or the axis entries are the individual characters, if no separators are used (except if it's digits only).

A row is a sequence of axes of a single kind: batch, input, or output. The shape type incorporates information relevant to inference, in particular shape variables: both for individual axes (`dim` variables), and for extending a row with more axes (`row` variables). Currently, all rows are (independently) broadcastable: can be broadcasted to a larger number of axes. However, in OCANNL the broadcasting can happen "in the middle", with not only the given trailing axes fixed, but also with the given leading axes fixed. (TODO: clarify here the precise logic as it is implemented, I'm not sure this description is correct.)

```ocaml
type solved_dim = { d : int; label : string option; proj_id : proj_id option }
(** A single axis in a shape. *)

type dim =
  | Var of dim_var
  | Dim of solved_dim
  | Conv_input of { stride : int; output : dim; dilation : int; kernel : dim }
      (** The offset is implicit, automatically derived. If [!use_padding] is [true], the offset is
          the left part of the dimensionality-preserving symmetric padding, otherwise it is 0. If
          [!use_padding] is [true], the value stands for dimensions size [stride * output],
          otherwise for dimensions size [stride * output + dilation * kernel]. If [dilation = 0],
          the value stands for projections of strided iteration rather than convolution. *)

type bcast =
  | Row_var of row_var  (** The row can be inferred to have more axes. *)
  | Broadcastable  (** The shape does not have more axes of this kind, but is "polymorphic". *)

type row = Row.t = { dims : dim list; bcast : bcast; id : row_id }

type shape = Shape.t = {
  mutable batch : row;
  mutable input : row;
  mutable output : row;
  id : int;  (** A node that has the same shape as this shape. *)
  debug_name : string;
}
```

The actual implementation is split into the `Row` module, which handles multi-row inference, and the `Shape` module which deals with the specific axis kinds (batch, input, output), _einsum_ specifications, and the shape-relevant semantics of operations expressed via the `Shape.logic` variant type. Since broadcasting extends leading axes (preserves trailing axes), substituting a `row_var` means prepending to the `dims` of the row that has the row variable as its `bcast` field.

Labels are a part of OCANNL, but it's a topic that needs more exploration and future work. Currently, OCANNL has labeled dimensions, but not labeled axes. This means that when two axes need to agree on the number of dimensions, they also need to agree on the labels. If the dimensions of both axes have labels, the labels need to be the same, and if one doesn't have a label initially, it's inferred to have the label from the other axis. Intuitively, the label is a specification of the semantics of an axis that is more fine-grained than, but of similar nature as, the number of dimensions. Currently, there is no global check to prevent the same label be used with different numbers of dimensions (on unrelated axes). Example: a label `"rgb"` attached to dimensions size 3 to denote that an axis represents three channels "red", "green" and "blue".

### Convolution-based indexing

The `Conv_input` constructor represents convolution-style input dimensions that enable support for operations like convolutions where output indices relate to input indices through the relationship:

```
input_dimension = stride * output_iterator + dilation * kernel_iterator
```

When `use_padding` is true, the offset is chosen to preserve dimensionality (i.e., output size equals input size for stride=1). When false, the offset is 0 (no padding).

The shape and projection inference handles `Conv_input` terms differently depending on `use_padding`. If `use_padding` is false, the impact of convolution kernels is incorporated additively during shape inference, and there's nothing more to do during projection inference. If `use_padding` is true, convolution kernels don't contribute during shape inference, and padding is computed during projection inference, keyed by `proj_id`. Padding is maximal size (width) of dilated kernels as encountered in `Dim` - `Conv_input` constraints and is propagated in either direction, although in practice for CNNs `Conv_input` should only appear as `subr` of `Dim_ineq`.

Shape inference does not maintain padding for axes of individual tensor nodes, these padding values are computed and updated during projections inference.

### Inference strategy

The actual shape inference combines row polymorphism with (nominal) subtyping, as known in the type inference literature. The subtyping stems merely from the fact that a dimension-1 axis can be used in the context of any dimension due to per-axis broadcasting. Row polymorphism stems from broadcasting to more axes: for example, when unifying an unknown (shape) row with a known one, we cannot assume that the unknown row will have just the axes of the known one, because maybe the known row is meant to be broadcasted here to more axes. The combination of row polymorphism with nominal subtyping means that the constraints we are solving are inequalities, both inequalities between rows (the `Row.t` type, i.e. the `row` type above), and between axes/dimensions (the `Row.dim` type). We maintain the inequality ordering between variables in the environment to compute the transitive closure during simplification. We also maintain a least upper bound on the solution.

```ocaml
type dim_entry =
  | Solved_dim of dim
  | Bounds_dim of { cur : dim_var list; subr : dim_var list; lub : dim option; constr : dim_constraint }

type row_entry =
  | Solved_row of t
  | Bounds_row of { cur : row_var list; subr : row_var list; lub : t option; constr : row_constraint }

type dim_env = dim_entry Map.M(Dim_var).t
type row_env = row_entry Map.M(Row_var).t

type environment = { dim_env : dim_env; row_env : row_env }

type constraint_ =
  | Dim_eq of { d1 : dim; d2 : dim }
  | Row_eq of { r1 : t; r2 : t }
  | Dim_ineq of { cur : dim; subr : dim }
  | Row_ineq of { cur : t; subr : t }
  | Dim_constr of { d : dim; constr : dim_constraint }
  | Rows_constr of { r : t list; constr : row_constraint }
  | Terminal_dim of dim
  | Terminal_row of t
```

We tie the direction of inequalities with capturing information in the structure of tensor expressions: where relevant, `cur` is a part of the shape of a super-tensor, and `subr` of a sub-tensor in a tensor expression. This reflects the nature of broadcasting: it is one-directional in that the shape of a subtensor can be "smaller-than-expected" thanks to broadcasting, but the shape of a super-tensor cannot be "smaller-than-expected". So, for ground (variable-free) dimensions, _n ≥ m_ means: _either n = m, or m = 1_; and for ground (variable-free) rows, _q ≥ r_ means: _q has at least as many axes as r, and for each dimension n of q at an axis where r has dimension m, we have n ≥ m_. The least upper bound `lub` of a variable is derived from the `cur` sides of inequalities with the variable on the `subr` side. We don't need to maintain a greatest lower bound, because we can incorporate the corresponding information immediately. For rows, we can substitute the row variable by a new row consisting of variables only, and add the corresponding `dim` inequalities with the variables on the `cur` side.

The entry point to shape inference is the shape logic specification, that each operation instance needs to provide. There are shortcuts in the syntax extension `%cd` to make it painless.

```ocaml
type deduce_within_shape = Not_constrained | Input_equals_output

type compose_type =
  | Pointwise_bin
      (** NumPy-style broadcast matching batch, input and output axes, e.g. as in [s1 + s2]. *)
  | Compose
      (** Compose the outputs of the second shape with the inputs of the first shape, i.e. the shape
          of [fun x -> s1(s2(x))], or [s1 * s2] where [*] is the inner product (e.g. matrix
          multiply). *)
  | Einsum of string
      (** The binary "einsum" syntax: RHS1;RHS2=>LHS, where RHSi, LHS are labels specifications.
          Since OCANNL's extended einsum notation supports both axis variables and row variables, it
          makes other compose types redundant. The [axis_labels] use pseudo-labels local to the
          notation, to line up the axes and row variables. The symmetric difference / disjunctive
          union of RHS1 and RHS2's pseudo-labels should be equal to LHS pseudo-labels.

          Note: The "right-hand-side" is on the left! I.e. the syntax is "rhs=>lhs",
          "rhs1;rhs2=>lhs". *)

type transpose_type =
  | Transpose  (** Swaps inputs and outputs of a shape, preserves batch axes. *)
  | Pointwise_un  (** Preserves the shape. *)
  | Permute of string  (** The unary "einsum" syntax: RHS1=>LHS. *)
  | Batch_slice of Ir.Indexing.static_symbol  (** Removes the leftmost batch axis. *)
  | Uint4x32_to_prec of Ir.Ops.prec Lazy.t
      (** Converts precision in a bit-effient way, with a corresponding conversion in total number
          of elements. Currently, assumes the incoming tensor (RHS) has just a single axis to not
          force unnecessary minimum sizes on output axes. *)

(** If you miss expressivity here, leave a note on
    {{:https://github.com/ahrefs/ocannl/issues/305}issue 305}. *)
type ternary_type =
  | Pointwise_tern  (** As in the operation [Where]. *)
  | Compose_accumulate  (** As in the operation [FMA]. *)

(** Extracts any available shape information from the initialization or fetch. *)
type terminal_type = Data of Ir.Assignments.init_data | Fetch of Ir.Assignments.fetch_op
```

### Non-tensor-like constraints

The above mechanisms (excluding `dim_constraint` and `row_constraint`) are sufficient to express tensor applications such as inner and outer products, axis permutations. They cannot directly express: size constraints, fixed position indexing (except for the special case of position 0), axis concatenation and "reverse concatenation" / splitting, strides, convolutions. At present, we implement size constraints and fixed position indexing.

```ocaml
type dim_constraint = Unconstrained_dim | At_least_dim of int

type total_elems =
  | Num_elems of int
  | Strided_var of { coeff : int Lazy.t; var : dim_var }
      (** The total number of elements is the coefficient times the number of dimensions the
          variable represents. *)

type row_constraint =
  | Unconstrained
  | Total_elems of { nominator : total_elems; divided_by : dim_var_set }
      (** The rows, inclusive of the further row spec, have this many elements. *)
  | Exact of dim list  (** The concatenated rows have these axes. *)
```

During the solution process, the constraints are incorporated, or propagated, into the environment `constr` entry fields, and into further `constraint_` constraints, as needed. This provides sufficient scaffolding to implement the other complex constraints as the need arises.

## Solving the constraints

The constraints are solved by: unification of the equation constraints, unification-like simplification of the inequality constraints, propagation of the complex constraints. Simplification of an inequality, and constraint propagation, can generate more constraints, so we need to be careful to keep it terminating. The solution proceeds in stages.

* Stage 1 is online as tensors are composed, and conservatively performs unification and constraint propagation. Stages 2, 3, 4 are only performed once necessary: when projections or dimensions are requested.
* Stage 2, when solving the constraints, substitutes dim variables in terminal shapes that do not have a LUB or other constraints, by dimension-1. (This is generalized at stage 6 to all variables.) (FIXME: reconsider this, see the algo for row variables: a new LUB can still be inferred.) Forces coefficients coming from precision byte sizes.
* Stage 3, when solving the constraints, sets yet-unknown dimension and row variables in terminal shapes to their least upper bounds (if any), but for rows only if they don't have a `Total_elems 1` constraint. It substitutes row variables in terminal shapes that do not have a LUB by one axis if that's required to satisfy the variable's constraint.
* Stage 4 sets yet-unknown dimensions with >1 lower bounds from direct accesses, to their LUBs if they have any. It substitutes row variables in terminal shapes that do not have a LUB by no-further-axes. (This is generalized at stage 6 to all variables.)
* Stage 5 addresses `Total_elems` and `Exact` constraints with yet-unknown row variables. For `Total_elems` and a single row variable: if the constraint can be satisfied by assuming the row variable is no-further-axes, it sets the row variable to `Broadcastable`, otherwise it sets it to one axis of the required dimension. For multiple row variables, if one is of the Output kind, sets the other variables to no-further-axes, and retries.
* Stage 6 sets row variables in the remaining inequalities to no-further-axes values. This can unlock further between-axis inequalities because of row variables sandwiched between leftmost axes from their side of the inequality and rightmost axes from the other side of the inequality. In row constraints, this also unlocks inference for the embedded dim variables.
* Stage 7 sets all dim variables remaining in updated shapes to the lower bound if they have any, otherwise to dimension-1. It sets all row variables remaining in updated shapes to no-further-axes.

Let's explain the shape inference functions.

* `s_dim_one_in_entry` / `s_row_one_in_entry`: substitutes the given dim / row variable in one dim / row env entry. Generates new inequalities if the variable was in one of the sides of a `Bounds` entry. Updates the constraint `constr` if the variable appears in it.
* `subst_dim` / `subst_row`: substitutes out a variable in a dim / row value, if any.
* `unify_dim`: solves a single equation between two values of type `dim`, and recursively all `dim` equations that this entails, but not other constraints.
* `unify_row`: solves a single equation between two rows, and recursively all `dim` and `row` equations that this entails, but not other constraints.
  * This is a depth-first component of an otherwise breadth-first overall constraint solution process, a "simplify early" optimization.
* `apply_dim_constraint` resp. `apply_row_constraint`: if they cannot make any progress on the constraint, they return `None`. Otherwise, they return a list of derived constraints, and an updated `dim_constraint` resp. `row_constraint`.
* `solve_dim_ineq`: solves a single inequality between two values of type `dim`; returns derived equations and inequalities. It maintains the between-variable bounds and the least-upper-bound (LUB). But there can only be one LUB (a dimension > 1) without forcing the bound variable itself to a solved form (with dimension = 1).
* `solve_row_ineq`: solves a single inequality between two rows; returns derived equations and inequalities. It derives between-`dim` inequalities from the known parts of the compared rows. It maintains between-row-variable bounds (when known parts of the rows match) and the LUB. It forces the `cur` side to have at least the number of axes of the `subr` side (via a variables-only `template`). It updates the LUB by computing dimensions-wise LUBs.
* `close_dim_terminal` and `close_row_terminal`: produce the equal-to-LUB constraint when available, from `Terminal_dim` and `Terminal_row` constraints produced for shapes of leaf tensors in tensor expressions, but only when `~stage:true`.
* `solve_inequalities`: solves equations, inequalities, and row constraints, until only row constraints remain. Row constraints can "pass" if there is not enough information, rather than reflecting their effect in the environment. Calls `close_dim_terminal` and `close_row_terminal` as appropriate.

The rationale behind only closing leaf (terminal) tensor shapes to their LUBs, while closing the remaining ones to dim-1:

* the composite tensors will get their shapes forced "from below" by their components;
* the leaf tensors cannot have their shape forced as they can always be broadcasted -- the only way they acquire shape information is downstream of `close_row_terminal`.

## Projections inference

Unlike shape inference proper, constraints that participate in projections inference pertain to a single assignment, and projection equations only tie individual instances of tensor accesses in the assignments. In particular even if the same tensor repeats in the assignments, the distinct instances will have different projection ids and participate in disjoint equations, so that they can be indexed differently.


```ocaml
type proj = Var of dim_var | Proj of { proj_id : int; d : int } | Solved of axis_index
type proj_to_index = Ir.Indexing.axis_index Map.M(Int).t
type proj_classes = int Map.M(Int).t

type proj_env = {
  proj_to_index : proj_to_index;
  proj_classes : proj_classes;
  product_dim : int Map.M(Int).t;
  non_product : Set.M(Int).t;
}
```

The projection inference functions.

* `get_proj_equations inequalities proj_axis_env env` converts both equations and inequalitites to projection equations. For inequalities, it takes broadcasting into account, and equates a potentially-broadcasted dim-1 projection to `Fixed_idx 0`. `proj_axis_env` originates from the `Shape` module, holds projections from the slice operator and the einsum syntax.
* `solve_proj_equations` unifies the projection equations, using union-find to maintain a representative for equal projections. Projections that already have an `axis_index` are `non_product` (not to be iterated over). The remaining projections have a `product_dim`, and get a fresh iterator.
* `get_dim_index` gets an `axis_index` for a `dim` based on the representative of its `proj_id`; and `Fixed_idx 0` for dim=1.

### Convolutions

There is an important and intentional difference between `dims` in the `arrayjit` part of the project: tensor nodes, `Ndarray` buffers, code generation -- they include padding in the dimension sizes; and on the other hand shape types, shape inference and tensors exclude padding from the dimension sizes. There is a tension: once the delayed computations of padding, projections and dims (dimension sizes) are forced for a particular node, the padding can no longer be updated (the underlying `Ndarray` buffer might already be created). Since during inference we update the padding incrementally without variables standing in for insufficient information, this unfortunately causes observability of the during-inference and post-inference distinction for the padding of a tensor node.

## Deriving the constraints

Other important functions in the `Shape` module.

* `einsum_slot_spec_to_dims_bio ~generative` parses an einsum spec for a single shape, returns the three rows and a mapping from axis (`dim`) variables to indices where the einsum specifies fixed indexing. When `generative` is true for the kind of a row, when an axis has a fixed projection to dimension 0, the axis is not a variable added to the fixed indexing mapping, but is instead dimension-1 (solved). The "generative" rows are the ones with no initial user-provided shape information. This is just a heuristic to avoid surprises where a tensor axis with only dimension 0 populated gets inferred a bigger dimension size -- it might be revisited in the future.
* `get_inequalities` builds row inequalities by pairing the rows of the current shape (as `cur`) with the rows of sub-shapes (as `subr`). It also derives a batch row constraint for terminals initialized with `Constant_fill values`. For `Batch_slice` (the `@|` operation) it waits till the batch row variables (if any) are solved, and derives row equations (not inequalities) between the current shape and the sub-shape, with `cur_sh.batch.dims` expanded to account for the slicing / indexing. For einsum specs, it derives inequalities, roughly: _current shape ≥ lhs spec shape_, and _rhs spec shape ≥ sub-shape_.
* `propagate_shapes` gets and then solves the inequalities, using a global state for the environment. It udpates the shapes in-place with the partial solution. It is invoked twice for each `update_step`: first during the bottom-up process of building tensors, and then in reverse order from `finish_inference`.
* `finish_inference` is called right before some projections or array dimensions are required (typically, because of jitting). It performs a second round of `propagate_shapes`, and then once again attempts to solve any remaining constraints that `propagate_shapes` didn't solve. Then it "closes the shapes": substitutes out remaining shape variables by their LUBs if any, or dimension-1 / `Broadcastable` (no-more-axes). Then it resets the environment state, since the shapes are now guaranteed to not have variables.
* `derive_projections` starts by freshening the `proj_id`s in the `update_step`. Then it generates and solves shape inequalities, and then generates and solves projection equations, and constructs the `projections` record.
* `of_spec` constructs a shape record from an einsum slot spec. If `deduced = Input_equals_output`, it adds the corresponding equation to the global environment.

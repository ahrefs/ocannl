# Shape inference and projection inference

To separate concerns, OCANNL is split into the `arrayjit` library, responsible for compilation of high-level n-D array operation sequences (`Assignments.t`) via the gccjit and cuda backends, and the main `ocannl` library, responsible for deriving the operations computing the forward propagation and backpropagation from tensor expressions. In particular, `arrayjit` contains `Indexing`, which represents complex indexing into arrays, and the main library `ocannl` has `Row` and `Shape` modules, which do the most "heavy-lifting" in the translation from concise tensor expressions to sequences of assignments.

Shape inference broadly speaking consists in OCANNL of inferring the `Shape.t` record -- shape inference proper, and inferring the `Indexing.projections` record -- projections inference. `Shape.t` records are mutable, so that the partially inferred shapes can be observed by the user. Shape and projections inference is intended to be declarative -- independent of the order in which constraints are added, but it's unclear to me if it is fully declarative, and in one sense it is not declarative: it depends on when ocannl expressions are compiled to assignments. Shape inference has two phases: `is_complete:false` shape inference is performed as constraints are added; `is_complete:true` inference happens at once when the corresponding tensor expressions are jitted. We need the "completed" phase because we want to make stronger assumptions: choose appropriate constraint boundaries as the sizes of dimensions that haven't been fully solved yet. If shape variables still remain in the inferred shapes after the `is_complete:true` pass, they default to broadcastable/no-more-axes and dimension-1.

The bulk of the projections inference happens alongside shape inference, with the projections-relevant information stored in auxiliary fields -- this prevents subtle bugs where projection semantics deviates from shape semantics, and will simplify adding new shape/projection inference features. Shape inference happens during `propagate_shapes` calls. Finally, the projections are reconstructed in `derive_projections`. It would seem `derive_projections` could reuse the already-computed solutions constraints. But we face a problem: we must prevent contaminating projections across different operations. To illustrate: we conclude the dimensions of two axes are the same because they are reduced together in another operation -- this should not force the axes to share a projection in the processed operation. To prevent the contamination, in each `derive_projections` call, we freshen the projection ids in the (inferred) shapes, and regenerate and re-solve the constraints with the fresh projection ids.

## Representing shapes and constraints

A tensor shape in OCANNL is composed of three rows of axes: batch, input and output. These are ordered input-last (`batch @ output @ input`) in the underlying n-dimensional array implementation of tensors. A (fully inferred) tensor shape must have non-empty output axes; we do not use the convention where empty axes mean the tensor is a scalar -- scalars = 1-D output-only tensors. For printing and einsum-notation-like specifications, we use the syntax: `batch|input->output` (or `input->output`, or `output`), where `batch`, `input`, `output` are whitespace or comma or parenthesis separated axis entries; or the axis entries are the individual characters, if no separators are used (except if it's digits only).

The shape type incorporates information relevant to inference, in particular shape variables: both for individual axes, and for extending a row with more axes. Currently, all rows are (independently) broadcastable: can be broadcasted to a larger number of axes, except when used with an einsum specification that forbids it.

```ocaml
type dim = Var of dim_var | Dim of { d : int; label : string option; proj_id : int option }

type bcast =
  | Row_var of int  (** The row can be inferred to have more axes. *)
  | Broadcastable  (** The shape does not have more axes of this kind, but is "polymorphic". *)

type dims_constraint =
  | Unconstrained
  | Total_elems of int  (** The shape-kind, inclusive of the further row spec, has this many elements. *)

type row = Row.t = { dims : dim list; constr : dims_constraint; bcast : bcast; id : row_id }

type shape = Shape.t = {
  mutable batch : row;
  mutable input : row;
  mutable output : row;
  id : int;  (** A node that has the same shape as this shape. *)
  debug_name : string;
}
```

The actual implementation is split into the `Row` module, which handles multi-row inference, and the `Shape` module which deals with the specific axis kinds (batch, input, output), _einsum_ specifications, and the shape-relevant semantics of operations expressed via the `Shape.logic` variant type.

Labels are a part of OCANNL, but it's a topic that needs more exploration and future work. Currently, OCANNL has labeled dimensions, but not labeled axes. This means that when two axes need to agree on the number of dimensions, they also need to agree on the labels. If the dimensions of both axes have labels, the labels need to be the same, and if one doesn't have a label initially, it's inferred to have the label from the other axis. Intuitively, the label is a specification of the semantics of an axis that is more fine-grained than, but of similar nature as, the number of dimensions. Currently, there is no global check to prevent the same label be used with different numbers of dimensions (on unrelated axes). Example: a label `"rgb"` attached to dimensions size 3 to denote that an axis represents three channels "red", "green" and "blue".

The actual shape inference combines row polymorphism with (nominal) subtyping, as known in the type inference literature. The subtyping stems merely from the fact that a dimension-1 axis can be used in the context of any dimension due to per-axis broadcasting. Row polymorphism stems from broadcasting to more axes: for example, when unifying an unknown (shape) row with a known one, we cannot assume that the unknown row will have just the axes of the known one, because maybe the known row is meant to be broadcasted here to more axes. The combination of row polymorphism with nominal subtyping means that the constraints we are solving are inequalities, both inequalities between rows (the `Row.t` type, i.e. the `row` type above), and between axes/dimensions (the `Row.dim` type). We use roughly the Fourier-Motzkin approach, where one reduces inequalities into a form "pivoted" on a variable, schematically: _{rows / dims that are greater-or-equal than v} ≥ v ≥ {rows / dims that are smaller-or-equal than v}_.

```ocaml
type 'a entry = { cur : 'a list; subr : 'a list; solved : 'a option }
(** An entry implements inequalities [cur >= v >= subr] and/or an equality [v = solved]. *)

type dim_env = dim entry Map.M(Dim_var).t
type row_env = row entry Map.M(Int).t

type environment = { dim_env : dim_env; row_env : row_env }
```

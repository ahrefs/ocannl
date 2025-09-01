(** {1 Tensor shape types, shape inference, projection inference.} *)

(** {2 Labels specifications and einsum notation.}

    Definition and properties of the syntax of labels specifications and einsum notation:
    - Whitespace-insensitive except that whitespace separates identifiers.
    - Comes in two variants: single-character and multicharacter;
    - if there is a comma [','] anywhere in the initial text, the multicharacter version is used,
    - otherwise the single character version is used.
    - Currently, the only non-whitespace, non-alphanumeric characters that make sense / are allowed
      in a spec are: ['>', '|', '-', ',', '=', ';', '+', '*', '_'].
    - identifier: single alphanum character or '_' in single-char mode, a sequence of alphanum
      characters or '_' otherwise (whitespace not allowed).
    - separators: a sequence of commas and whitespaces.
    - separators_with_comma: commas and whitespaces containing at least one comma.
    - axes_spec_single_char: separators? identifier+ separators?
    - axes_spec_multichar: separators? (identifier separators_with_comma)* identifier separators?
    - conv_expression: term '+' term
    - term: (coeff '*')? identifier
    - coeff: integer -- note that syntax extensions will splice in the value of an OCaml identifier
    - ellipsis_spec: '...' <|> '..' identifier '..'
    - row_spec: axes_spec <|> ellipsis_spec axes_spec <|> axes_spec ellipsis_spec axes_spec
    - labels_spec: row_spec <|> row_spec '|' row_spec <|> row_spec '->' row_spec <|> row_spec '|'
      row_spec '->' row_spec.
    - permute_spec: labels_spec '=>' labels_spec
    - einsum_spec: labels_spec ';' labels_spec '=>' labels_spec

    If labels_spec does not contain ["|"] nor ["->"], each label is of the kind [Output]. If the
    spec doesn't contain ["|"], labels to the left of ["->"] are [Input] and to the right [Output].
    Labels to the left of ["|"] are [Batch], and between ["|"] and ["->"] are [Input].

    The labels [".."ident".."], ["..."] (where [ident] does not contain any of the special
    characters) are only allowed once for a kind. They are used to enable (in-the-middle)
    broadcasting for the axis kind in the einsum-related shape inference (like the ellipsis ["..."]
    in [numpy.einsum]), and are translated to row variables. The ellipsis ["..."] is context
    dependent: in the batch row it is the same as ["..batch.."], in the input row the same as
    ["..input.."], in the output row the same as ["..output.."]. When the same row variable is used
    in multiple rows, the corresponding broadcasted axes are matched pointwise in the resulting
    operation.

    The label ["_"] is a place-holder: it is not output to the resulting map but aligns the axes of
    other labels.

    Conv expressions have the form [stride*output+dilation*kernel] where stride and dilation are
    optional integer coefficients (defaulting to 1), and output/kernel are axis labels. This syntax
    enables convolution-style indexing where input_dimension = stride * output_iterator + dilation *
    kernel_iterator. Conv expressions automatically trigger multichar mode and are only supported in
    multichar mode.

    Note: currently, OCANNL shapes always allow broadcasting. Row variables track the broadcasted
    axes -- if there is no row variable, broadcasted axes are not tracked. In the notation case
    `row_spec` = `axes_spec`, the axes are the rightmost axes (broadcasting to the left). In the
    past, we supported preventing broadcasting, but removed that to reduce complexity. *)

(** {2 User-ish API.} *)

open Base

type padding = Row.axis_padding array option [@@deriving sexp, equal]

(** Specification for individual axes in the einsum notation. *)
type axis_spec =
  | Label of string  (** A variable axis label. *)
  | Fixed_index of int  (** A fixed index, used for projection. *)
  | Conv_spec of { stride : int; output_label : string; dilation : int; kernel_label : string }
      (** Convolution-style axis specification: stride*output + dilation*kernel. *)
[@@deriving compare, sexp]

type t = {
  mutable batch : Row.t;
  mutable input : Row.t;
  mutable output : Row.t;
  mutable batch_padding : padding;
  mutable input_padding : padding;
  mutable output_padding : padding;
  id : int;  (** A node that has the same shape as this shape, or [-1]. *)
  debug_name : string;
}
[@@deriving equal, sexp]

type deduce_within_shape = Not_constrained | Input_equals_output [@@deriving compare, sexp]

type compose_type =
  | Pointwise_bin
      (** NumPy-style broadcast matching batch, input and output axes, e.g. as in [s1 + s2]. *)
  | Compose
      (** Compose the outputs of the second shape with the inputs of the first shape, i.e. the shape
          of [fun x -> s1(s2(x))], or [s1 * s2] where [*] is the inner product (e.g. matrix
          multiply). *)
  | Einsum of string * Ir.Indexing.variable_ref list
      (** The binary "einsum" syntax: RHS1;RHS2=>LHS, where RHSi, LHS are labels specifications.
          Since OCANNL's extended einsum notation supports both axis variables and row variables, it
          makes other compose types redundant. The [axis_labels] use pseudo-labels local to the
          notation, to line up the axes and row variables. The symmetric difference / disjunctive
          union of RHS1 and RHS2's pseudo-labels should be equal to LHS pseudo-labels.

          The optional {!Ir.Indexing.variable_ref}s will capture the solutions of the dimensions
          corresponding to the specification labels equal to [ref_label] of a reference.

          Note: The "right-hand-side" is on the left! I.e. the syntax is "rhs=>lhs",
          "rhs1;rhs2=>lhs". *)
[@@deriving sexp_of, equal]

type transpose_type =
  | Transpose  (** Swaps inputs and outputs of a shape, preserves batch axes. *)
  | Pointwise_un  (** Preserves the shape. *)
  | Permute of string * Ir.Indexing.variable_ref list
      (** The unary "einsum" syntax: RHS1=>LHS.

          The optional {!Ir.Indexing.variable_ref}s will capture the solutions of the dimensions
          corresponding to the specification labels equal to [ref_label] of a reference. *)
  | Batch_slice of Ir.Indexing.static_symbol  (** Removes the leftmost batch axis. *)
  | Uint4x32_to_prec of Ir.Ops.prec Lazy.t
      (** Converts precision in a bit-effient way, with a corresponding conversion in total number
          of elements. Currently, assumes the incoming tensor (RHS) has just a single axis to not
          force unnecessary minimum sizes on output axes. *)
[@@deriving equal, sexp_of]

(** If you miss expressivity here, leave a note on
    {{:https://github.com/ahrefs/ocannl/issues/305}issue 305}. *)
type ternary_type =
  | Pointwise_tern  (** As in the operation [Where]. *)
  | Compose_accumulate  (** As in the operation [FMA]. *)
[@@deriving equal, sexp_of]

(** Extracts any available shape information from the initialization or fetch. *)
type terminal_type = Data of Ir.Assignments.init_data | Fetch of Ir.Assignments.fetch_op
[@@deriving equal, sexp_of]

val make :
  ?batch_dims:int list ->
  ?input_dims:int list ->
  ?output_dims:int list ->
  ?batch_axes:(string * int) list ->
  ?input_axes:(string * int) list ->
  ?output_axes:(string * int) list ->
  ?deduced:deduce_within_shape ->
  debug_name:string ->
  id:int ->
  unit ->
  t
(** Creates a shape. [id] should be the id the associated tensor (if any). At most one of the pairs
    [batch_dims], [batch_axes] etc. should be given: if none, the corresponding row will be
    inferred. [batch_axes] etc. provide labels for the dimensions of the corresponding axes. Note
    that these are dimensions labels and not axis labels: they need not be unique for a row, are
    inferred when provided, and must match whenever the axis sizes must match. *)

val to_string_hum : ?style:Row.print_style -> t -> string

val unsafe_reinitialize : unit -> unit
(** Bring global state to its initialization values. This invalidates any unfinished inference. *)

(** {2 Internal-ish API.} *)

(** How to propagate shape updates and do the last update of [Tensor.t.shape] when finalizing the
    tensor. Axes are broadcast-expanded on a bottom-up update to fit the incoming shape. *)
type logic =
  | Broadcast of compose_type * t * t
      (** Matches the shapes for a binary operation.

          For [Broadcast (Einsum (ls1, ls2, ls3), s1, s2)], the labels of [s1] and [s2] must match
          according to the [ls1], [ls2] lineup, and the resulting shape inherits the labels
          according to the [ls3] lineup. *)
  | Transpose of transpose_type * t
      (** Permutes the axes of a shape. One case of [Transpose] is to swap inputs with outputs of
          [s1], hence the name. *)
  | Broadcast_tern of ternary_type * t * t * t  (** Matches the shapes for a ternary operation. *)
  | Terminal of terminal_type
      (** Extracts any available shape information from the initialization. *)
[@@deriving equal, sexp_of]

type update_id [@@deriving equal, compare, hash, sexp]

val get_update_id : unit -> update_id

val logic_to_spec : logic -> string
(** Converts a shape logic to its string specification for debugging/display purposes. *)

type update_step = { shape : t; logic : logic; id : update_id } [@@deriving sexp_of]
(** Data required for a shape inference update step. Ideally, an update should be performed at least
    twice, the second time after all the other relevant updates have been performed for the first
    time. In OCANNL, this is achieved by performing updates both as the tensors are constructed, and
    via lazy callbacks as the corresponding [Ir.Indexing] dimensions and projections are first
    accessed. *)

val to_dims : t -> int array
(** Uses the matrix convention of putting the input axes last. *)

val to_padding : t -> (Ir.Ops.axis_padding array * float) option
(** Returns the padding of the shape, if any. Includes the padded value. Uses the matrix convention
    of putting the input axes last. *)

val propagate_shapes : update_step -> unit

val derive_projections : update_step -> Ir.Indexing.projections
(** Computes the indexing into subtensors given the shape information of a tensor.
    [derive_projections] should only be invoked when the shapes are fully inferred already! *)

val of_spec : ?deduced:deduce_within_shape -> debug_name:string -> id:int -> string -> t
val default_display_indices : t -> int array

val to_labels : t -> string array
(** Uses the matrix convention of putting the input axes last. *)

type 'a axis_map
type parsed_axis_labels [@@deriving sexp]

val axis_labels : parsed_axis_labels -> axis_spec axis_map
val axis_labels_of_spec : string -> parsed_axis_labels
val axis_map_to_dims_index : ?default:'a -> 'a axis_map -> 'a array

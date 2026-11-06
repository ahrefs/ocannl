(** {1 Tensor shape types, shape inference, projection inference.} *)

(** {2 Labels specifications and einsum notation.}

    Definition and properties of the syntax of labels specifications and einsum notation:
    - Whitespace-insensitive except that whitespace separates identifiers.
    - Comes in two variants: single-character and multicharacter;
    - if there is one of [',', '+', '*', '^'] anywhere in the initial text, the multicharacter
      version is used,
    - otherwise the single character version is used.
    - Currently, the only non-whitespace, non-alphanumeric characters that make sense / are allowed
      in a spec are: ['>', '|', '-', ',', '=', ';', '+', '*', '_', '^'].
    - The lexer handles whitespace.
    - identifier: single alphanum character or '_' in single-char mode, a sequence of alphanum
      characters or '_' otherwise
    - axes_spec_single_char: identifier+
    - axes_spec_multichar: (axis_spec comma)* axis_spec
    - axis_spec: identifier <|> conv_expr <|> concat_expr
    - conv_expr: term '<' '+' term <|> term '=' '+' term -- syntax extensions allow omitting
      ['<' | '='] when [use_padding] is in scope.
    - term: (coeff '*')? identifier
    - coeff: integer -- syntax extensions will splice in the value of an OCaml identifier
    - concat_expr: identifier '^' identifier ('^' identifier)*
    - ellipsis_spec: '...' <|> '..' identifier '..'
    - row_spec: axes_spec <|> ellipsis_spec axes_spec <|> axes_spec ellipsis_spec axes_spec
    - labels_spec: row_spec <|> row_spec '|' row_spec <|> row_spec '->' row_spec <|> row_spec '|'
      row_spec '->' row_spec.
    - permute_spec: labels_spec '=>' labels_spec
    - einsum_spec: labels_spec ';' labels_spec '=>' labels_spec

    If labels_spec does not contain ["|"] nor ["->"], each label is of the kind [Output]. If the
    spec doesn't contain ["|"], labels to the left of ["->"] are [Input] and to the right [Output].
    Labels to the left of ["|"] are [Batch], and between ["|"] and ["->"] are [Input].

    A row whose kind separator is omitted altogether reads as the context ellipsis ["..."]: [x] is
    equivalent to [...|...->x], so batch and input axes broadcast through terse specs by default.
    Writing the separator with an empty row spec closes the row: [| ->x] means no batch and no input
    axes, [|x] means no batch axes, [->x] means no input axes. In einsum specs the distinction
    matters wherever another row of the same kind is written ["..."]: an omitted row shares the
    context row variable with it. On the result side of [=>] this means those axes pass through
    (write e.g. [=> |->0] to reduce over them instead), and across operands it forces the rows equal
    (close the row, e.g. a [|]-prefixed kernel slot, when an operand must not receive the broadcast
    axes).

    The labels [".."ident".."], ["..."] (where [ident] does not contain any of the special
    characters) are only allowed once for a kind. They are used to enable (in-the-middle)
    broadcasting for the axis kind in the einsum-related shape inference (like the ellipsis ["..."]
    in [numpy.einsum]), and are translated to row variables. The ellipsis ["..."] is context
    dependent: it stands for a reserved row variable specific to the kind of the row it appears in
    (batch, input or output), shared by every ["..."] of that kind within one einsum spec. The
    reserved variables cannot be referred to by name, so identifiers such as [batch] remain
    available as ordinary dimension labels and ["..batch.."] is an ordinary user row variable. When
    the same row variable is used in multiple rows, the corresponding broadcasted axes are matched
    pointwise in the resulting operation.

    The label ["_"] is a place-holder: it is not output to the resulting map but aligns the axes of
    other labels.

    Conv expressions have the form [stride*output< + dilation*kernel + stride_offset] or with [=] in
    place of [<], where stride and dilation are optional integer coefficients (defaulting to 1),
    output/kernel are axis labels, and stride_offset is an optional integer offset that's smaller
    than stride. This syntax enables convolution-style indexing where input_dimension = stride *
    output_iterator + dilation * kernel_iterator. Conv expressions automatically trigger multichar
    mode. Under syntax extensions [%cd] and [%op], the integer coefficients can be spliced-in OCaml
    identifiers and the specifier [<] vs. [=] can be omitted if [use_padding] is in scope.

    Adding [<] after the output label (e.g., [stride*output<+kernel]) indicates no-padding mode,
    where indices must stay within the input bounds. In this mode, the input dimension must satisfy:
    [(input - effective_kernel_span) mod stride = 0], where
    [effective_kernel_span = 1 + (kernel - 1) * dilation]. With [=] instead, padding is applied for
    the kernel needs, and the constraint is simply [input mod stride = 0].

    Concat expressions have the form [label1 ^ label2 ^ ...], indicating that multiple axes are
    concatenated into a single axis. Concat expressions automatically trigger multichar mode.

    Note: currently, OCANNL shapes always allow broadcasting. Row variables track the broadcasted
    axes -- if there is no row variable, broadcasted axes are not tracked. In the notation case
    `row_spec` = `axes_spec`, the axes are the rightmost axes (broadcasting to the left). In the
    past, we supported preventing broadcasting, but removed that to reduce complexity. *)

(** {2 User-ish API.} *)

open Base

type padding = Row.axis_padding array option [@@deriving sexp, equal]

type t = private {
  mutable batch : Row.t;
  mutable input : Row.t;
  mutable output : Row.t;
  mutable batch_padding : padding;
  mutable input_padding : padding;
  mutable output_padding : padding;
  mutable padding_elem : float option option;
      (** The padding element for this shape's tensors. [None] means "unknown" (not yet determined),
          [Some (Some v)] means all operations use neutral element [v], [Some None] means different
          operations require different neutral elements (margin must be reset before each
          operation). *)
  mutable closed : bool; [@sexp_drop_if fun _ -> true] [@default false] [@equal.ignore]
      (** Set by a successful [finish_inference] on every shape of the steps it commits
          (gh-ocannl-1301): a closed shape's rows (modulo projection ids), paddings and
          [padding_elem] no longer change -- a write that would change them raises before writing
          anything. Not part of the shape's value: dropped from sexps, ignored by [equal]. *)
  mutable discarded_by : string option;
      [@sexp_drop_if fun _ -> true] [@default None] [@equal.ignore]
      (** Set by [Tensor.rewind_to_last_closure] on every open shape it discards, to the rewind's
          provenance: a step over a discarded shape is refused before it is registered. Not part of
          the shape's value. *)
  id : int;  (** A node that has the same shape as this shape, or [-1]. *)
  debug_name : string;
}
[@@deriving equal, sexp_of]
(** Private (gh-ocannl-1506): fields are read and matched freely, but only [Shape]'s guarded setters
    write them and only [Shape] constructs a shape, so the closed-shape guard cannot be bypassed.
    The privacy is shallow: a padding array read off a shape is still a mutable array. *)

type deduce_within_shape = Not_constrained | Input_equals_output [@@deriving compare, sexp]

val infer_equal : t -> t -> unit

type delayed_var_ref = private {
  var_ref : Ir.Indexing.variable_ref;
  mutable var : [ `Row of Row.row_var | `Dim of Row.dim_var | `Not_set_yet ];
  mutable carried_row : Row.dim list option;
      (** The axes of the row the reference was bound to when its inference closed, if it was bound
          to a row that resolved (gh-ocannl-1689), outer to inner, [Sym] axes included: across the
          closure it stands in for that row. A concatenated axis of concrete components is its sum
          (gh-ocannl-1969). [None] when the binding at the closure was a dimension, or the row held
          a concatenated or affine axis over a symbolic extent. Omitted from sexps when [None]. *)
  uid : int;  (** The reference's identity for rewind tracking; ignored by [equal] and sexps. *)
  mutable discarded_by : string option;
      (** The provenance of the rewind that discarded the reference, which every setter then refuses
          (gh-ocannl-1789); ignored by [equal] and sexps. *)
}
[@@deriving equal, sexp_of]
(** Private (gh-ocannl-1573): read and matched freely, written only by [Shape] -- through guarded
    setters that refuse a reference a rewind discarded and count the write as a session mutation --
    and built only by {!get_variable_ref}. The [Ir.Indexing.variable_ref] it carries is solved only
    by [Shape] too: [Ir.Indexing.set_solved_dim] and [set_solved_sym] take the process's one
    [Ir.Indexing.shape_writer], which [Shape] claims as it initializes (gh-ocannl-1624). *)

val get_variable_ref : string -> delayed_var_ref
(** Returns a fully unset variable reference with the given label.

    A reference passed through [~capture_dims] to several operations names one dimension, in
    whichever order inference solves them (gh-ocannl-1545): within one inference (up to the next
    [finish_inference]) the operations' dimension variables are equated, as are their row variables,
    and a dimension variable with a row variable has the row's product. Across a closure the
    reference carries its solved value: a dimension's size or symbol, and a row's axes
    ([carried_row], gh-ocannl-1689) together with their product ([solved_dim], see {!set_dim}). A
    later row capture is the same row -- [[2; 3]] does not become [[6]] or [[3; 2]] -- and a later
    dimension capture has the row's product. A reference bound to a dimension when its inference
    closed carries only the number, so a row captured from it afterwards is tied by product, as it
    would be within one inference.

    A row holding a symbolic extent has no numeric product: across its closure it can be captured
    again as a row, or related to a row reference by {!set_equal}, but capturing it as a dimension,
    or relating it by {!set_dim}, {!set_sym_dim}, a dimension {!set_equal} or {!set_scale}, raises
    [Row.Shape_error] naming the closure (gh-ocannl-1655). *)

val set_dim : delayed_var_ref -> int -> unit
(** Sets the dimension resp. total elements of the dim resp. row variable reference to the given
    value. This will propagate through shape inference.

    For row variables, this means the product of the dimensions, via the [Total_elems] constraint.
*)

val set_sym_dim : delayed_var_ref -> Ir.Indexing.static_symbol -> unit
(** Sets the dim variable reference to a symbolic extent (gh-490): during shape inference the axis
    is a rigid constant equal only to itself, and it is materialized at its declared maximum extent
    ([static_range], which is required and must be positive) for Tnode dims and projections.

    Row variables are not supported: a symbolic total-elements constraint would require symbolic
    arithmetic (raises [Row.Shape_error]). Conflicts with a prior {!set_dim}, or a different symbol
    set on the same reference, also raise [Row.Shape_error]. *)

val set_equal : delayed_var_ref -> delayed_var_ref -> unit
(** Sets the two variable references to be equal (in some sense). This will propagate through shape
    inference.

    When both references are dimension variables or both are row variables, this means they are
    precisely equal: two rows get the same axes, not only the same product, whatever either
    reference has been solved to so far (gh-ocannl-1617). When one is a dimension variable and the
    other is a row variable, this means they have the same number of total elements.

    A row reference bound by an inference that has since closed takes part in a Row-Row call with
    the row it carries ([carried_row], gh-ocannl-1689): the other row is made equal to it. A closed
    row reference that carries no row (its row held a concatenated or affine axis over a symbolic
    extent) raises [Row.Shape_error]: relate such rows before the closure.

    An unbound reference (from {!get_variable_ref}, not yet passed to [~capture_dims]) paired with a
    row reference takes or gives a number, the row's product. Binding it to a row afterwards would
    leave the two rows tied by product alone, so that binding raises [Row.Shape_error]
    (gh-ocannl-1655): bind it first to have the rows equal. Binding it to a dimension stands. *)

val set_scale : factor:int -> delayed_var_ref -> delayed_var_ref -> unit
(** Asserts a multiplicative relation between two dimension variable references: the first ref
    equals [factor] times the second. Mirrors {!set_equal} in spirit but expresses non-convolutional
    scaling -- the motivating use case is hourglass-style networks where one dimension is a constant
    multiple of another (e.g. an MLP hidden width that is twice the input width).

    Concretely, if the first ref ends up bound to dimension [d_large] and the second to [d_small],
    this asserts [d_large = factor * d_small].

    Restrictions:
    - Both refs must be (or resolve to) dimension variables. Row-variable and mixed dim/row variants
      are not currently supported and raise [Row.Shape_error] regardless of which side is solved (no
      silent routing through {!set_dim}'s row total-elements path).
    - [factor] must be a positive integer ([>= 1]). [factor = 1] is accepted and behaves exactly
      like {!set_equal}; callers should prefer {!set_equal} in that case for clarity.
    - Convolutional scaling (kernel/dilation/padding) is expressed via einsum convolution notation,
      not via this function.
    - As with {!set_equal}, calling on two fresh refs from {!get_variable_ref} (both still
      [`Not_set_yet]) raises [Row.Shape_error]; use {!set_dim} on one side first, or capture both
      refs through einsum notation (e.g. [+*]) so that shape propagation can bind them to dimension
      variables before this call.

    On a both-solved mismatch, on a one-solved-larger case where the solved dimension is not
    divisible by [factor], on insufficient information (both refs are [`Not_set_yet]), or when
    either ref resolves to a row variable, this raises [Row.Shape_error]. *)

type compose_type =
  | Pointwise_bin
      (** NumPy-style broadcast matching batch, input and output axes, e.g. as in [s1 + s2]. *)
  | Compose
      (** Compose the outputs of the second shape with the inputs of the first shape, i.e. the shape
          of [fun x -> s1(s2(x))], or [s1 * s2] where [*] is the inner product (e.g. matrix
          multiply). *)
  | Einsum of string * delayed_var_ref list
      (** The binary "einsum" syntax: RHS1;RHS2=>LHS, where RHSi, LHS are labels specifications.
          OCANNL's extended einsum notation supports both axis variables and row variables. The
          [axis_labels] use pseudo-labels local to the notation, to line up the axes and row
          variables. The symmetric difference / disjunctive union of RHS1 and RHS2's pseudo-labels
          should be equal to LHS pseudo-labels.

          Unlike [Pointwise_bin] and [Compose], einsum operations use equations only (not
          inequalities), so they do NOT permit broadcasting. This makes einsum more restrictive but
          also more precise for operations where exact shape matching is required.

          The optional {!Ir.Indexing.variable_ref}s will capture the solutions of the dimensions
          corresponding to the specification labels equal to [ref_label] of a reference.

          Note: The "right-hand-side" is on the left! I.e. the syntax is "rhs=>lhs",
          "rhs1;rhs2=>lhs". *)
  | Defined_by_cd_logic
      (** A placeholder for operations where the shape logic is defined by the %cd extension. *)
[@@deriving sexp_of, equal]

type transpose_type =
  | Transpose  (** Swaps inputs and outputs of a shape, preserves batch axes. *)
  | Pointwise_un  (** Preserves the shape. *)
  | Permute of string * delayed_var_ref list
      (** The unary "einsum" syntax: RHS1=>LHS.

          Unlike [Pointwise_un], permute operations use equations only (not inequalities), so they
          do NOT permit broadcasting. This makes permute more restrictive but also more precise for
          operations where exact shape matching is required.

          The optional {!Ir.Indexing.variable_ref}s will capture the solutions of the dimensions
          corresponding to the specification labels equal to [ref_label] of a reference. *)
  | Batch_slice of Ir.Indexing.static_symbol  (** Removes the leftmost batch axis. *)
  | Uint4x32_to_prec of Ir.Ops.prec Lazy.t
      (** Converts precision in a bit-effient way, with a corresponding conversion in total number
          of elements. Currently, assumes the incoming tensor (RHS) has just a single axis to not
          force unnecessary minimum sizes on output axes. *)
  | Defined_by_cd_logic
      (** A placeholder for operations where the shape logic is defined by the %cd extension. *)
[@@deriving equal, sexp_of]

type ternary_type =
  | Pointwise_tern  (** As in the operation [Where]. *)
  | Compose_accumulate  (** As in the operation [FMA]. *)
  | Defined_by_cd_logic
      (** A placeholder for operations where the shape logic is defined by the %cd extension. *)
  | Einsum_tern of string * delayed_var_ref list
      (** Ternary einsum contraction. The spec string has three RHS slots separated by [;] and one
          LHS slot after [=>]. Axis labels absent from the LHS become reduction axes. E.g.
          ["ij;jk;km=>im"] for a chain contraction (note: O(N⁴) — prefer binary chains for chain
          contractions; ["p;a;b=>out"] for select-before-reduce with [where]. *)
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
    inferred. [batch_axes] etc. provide bases (semantic annotations) for the dimensions of the
    corresponding axes. Note that these are dimension bases and not axis labels: they need not be
    unique for a row, are inferred when provided, and must match whenever the axis sizes must match.
*)

val to_string_hum : ?style:Row.print_style -> t -> string

val unsafe_reinitialize : unit -> unit
(** Bring global state to its initialization values. This invalidates any unfinished inference. *)

(** {2 Internal-ish API.} *)

val set_terminal : is_param:bool -> t -> unit
(** Mark the shape as terminal, so that its rows can be closed to Greatest Lower Bounds (GLBs). This
    function is only intended for parameters shapes, which would otherwise not be terminal because
    of the initialization expressions of the parameters. *)

val set_resolve_at_use : t -> unit
(** Mark the shape's rows as resolved by their use sites (gh-ocannl-544): at the final solver stages
    their row variables may close to the GLB of the use sites' rows, where operation results close
    to an empty remainder by default. Used by [Tensor.param] for parameter-initialization cones and
    by the [stretch] operation; terminal shapes (leaves, parameters) are marked automatically. *)

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
  | Terminal of { is_param : bool; logic : terminal_type }
      (** Extracts any available shape information from the initialization. The [is_param] field
          indicates if this is a parameter tensor that requires gradients. *)
  | Block of { spec : string; delayed_vars : delayed_var_ref list; rhses : t list }
      (** Block tensor / concatenation operation. Takes an n-ary einsum specification where
          concatenated axes use the [a^b^c] syntax, e.g., ["x;y;z => x^y^z"]. *)
[@@deriving equal, sexp_of]

type update_id [@@deriving equal, compare, hash, sexp]

val logic_to_spec : logic -> string
(** Converts a shape logic to its string specification for debugging/display purposes. *)

type update_step = private {
  shape : t;
  logic : logic;
  id : update_id;
  mutable unsafe_projections : Ir.Indexing.projections option;
  neutral_elem : float option;
      (** The neutral element of the operation's accumulation, fixed at creation: [Some v] when all
          assignment ops in the update step use the same neutral element [v], [None] when different
          operations have different neutral elements or when there are no accumulator operations.
          [derive_projections] reads it for the clamped-window decision (gh-504) and to commit the
          margins' neutral ([padding_elem]), so a step is only created once its operation's
          assignments are known -- see [Tensor.op]. *)
  mutable discarded_by : string option;
      (** The provenance of the rewind that discarded the step, which {!get_projections} then
          refuses (gh-ocannl-1789); left out of sexps. *)
}
[@@deriving sexp_of]
(** Data required for a shape inference update step. Ideally, an update should be performed at least
    twice, the second time after all the other relevant updates have been performed for the first
    time. In OCANNL, this is achieved by performing updates both as the tensors are constructed, and
    via lazy callbacks as the corresponding [Ir.Indexing] dimensions and projections are first
    accessed.

    Private (gh-ocannl-1573): read and matched freely; built only by {!make_update_step}, and only
    [Shape]'s projection derivation publishes [unsafe_projections], once. *)

val make_update_step : shape:t -> logic:logic -> neutral_elem:float option -> update_step
(** A fresh update step with a new id and no projections yet. It is not registered: see
    {!propagate_shapes}. *)

val to_dims : t -> int array
(** Uses the matrix convention of putting the input axes last. *)

val product_space_shape : shape:t -> logic:logic -> t
(** The product-space proxy shape of an einsum-family operation (gh-512): the result's axes followed
    by the reduced-over (contracted) axes, in the product-space order of the derived projections.
    The proxy shares the operation's shape-inference solution (its constraints are emitted on
    construction), so [infer_equal]-ing a [%cd] [*_pspace] intermediate against it gives the
    intermediate the dimensions of the operation's full product space (e.g. the (output x kernel)
    pair space of a windowed reduction). Must be called before the operation's inference completes.
    Raises [Row.Shape_error] when the product space is not expressible as a shape: a non-einsum
    operation, more than one reduced-over row variable, or a reduced-over row variable combined with
    an open result input row. *)

val to_padding : t -> (Ir.Ops.axis_padding array * float) option
(** Returns the padding of the shape, if any. The inner [float option] is the padded value: [Some v]
    when all operations use the same neutral element, [None] when different operations require
    different neutral elements (margin must be reset before each operation). Uses the matrix
    convention of putting the input axes last. *)

val with_construction_window : (unit -> 'a) -> 'a
(** Runs [f] with the construction-in-progress depth raised. While it is positive, every entry point
    that finalizes inference ([to_dims], [to_padding], [get_projections]) raises [Row.Shape_error]
    at the finalizer, before anything is committed: [Tensor.op] runs its [op_asn] under it, since
    the operation has not registered its constraints yet and a finalization there would close its
    operands without them (gh-ocannl-830). *)

val propagate_shapes : update_step -> unit
(** Registers the step and solves its constraints eagerly. Refused while the session is tainted; a
    failure after registering taints it. *)

(** {3 Session consistency (gh-ocannl-1301)}

    [finish_inference] is the session's only commit point: it marks every shape of the steps it
    commits {!field:closed}. A construction that fails after mutating the inference session (see
    [Row.mutation_serial]) taints it; while tainted, constructions, finalization and compilation
    raise [Row.Shape_error] naming the first failure. [Tensor.rewind_to_last_closure] restores a
    usable session. *)

val check_untainted : unit -> unit
(** Raises [Row.Shape_error] when the session is tainted. *)

val taint_provenance : unit -> string option
(** The first failure's provenance, when the session is tainted. *)

val taint_session : provenance:string -> unit
(** Records a taint; the first one is kept. *)

val exn_summary : exn -> string
(** The message of a [Row.Shape_error] or [Utils.User_error], else the printed exception. *)

val reserved_id_base : int
(** The base of the reserved range from which a window that is always rewound -- a shape-scheme
    trace, a definition check -- allocates its tensors' ids, above every session id
    (gh-ocannl-1751). [Tensor] allocates from it; {!error_to_string_hum} recognizes such ids. *)

val error_to_string_hum : string -> Row.error_trace list -> string
(** [error_to_string_hum msg trace] renders a [Row.Shape_error (msg, trace)] for a person
    (gh-ocannl-1799): the message, then one indented line per finding -- the conflicting sizes or
    rows (or the one axis or row the message is about), the shapes involved, the failed constraint
    (with [⊑] for "refines", as in [Tensor.Scheme.to_string]), and the tensors it relates with their
    axis kinds and the operation that related them. Variables are named by first appearance (['a],
    ..['b].., followed by [:<name>] for a named dimension variable), and a tensor of a trace or a
    check, whose id comes from the reserved range, by first appearance too ([n'1], [#'1]), so the
    text depends on neither the session's id counters nor the reserved ones; a session tensor keeps
    its id. This is what [Printexc.to_string] -- and so the toplevel -- prints for a
    [Row.Shape_error] (prefixed ["Shape_error: "]), and what its sexp converter ([Exn.to_string])
    carries, one atom per line; the setting [shape_error_full_sexp=true] restores the full
    constraint sexp to both. *)

val with_taint_on_failure : describe:(exn -> string) -> (unit -> 'a) -> 'a
(** Runs [f]; if it raises after the session was mutated, taints the session with [describe exn],
    then re-raises with the original backtrace. *)

val set_on_closure : (unit -> unit) -> unit
(** Installs the hook called at the end of every successful closure ([Tensor] records its checkpoint
    there). *)

val check_can_rewind : ?what:string -> unit -> unit
(** Raises when called within a construction window; [what] names the caller in the message. *)

val can_checkpoint : unit -> bool
(** Whether an explicit checkpoint can be taken now: outside a construction window and untainted. *)

exception Untraceable_scheme of string
(** A shape-scheme trace refused by design (gh-ocannl-1751): the function is valid, but a scheme
    cannot carry what it does -- a precision-dependent total-elements constraint (random number
    generation), [set_dim], [set_sym_dim], [set_equal] or [set_scale] on a capture reference made
    outside the trace (gh-ocannl-1791; binding one is an input), or forcing dims or projections
    during the trace. Raised by {!Scheme.trace}, prefixed ["tracing the shape scheme <label>: "]; a
    contradiction raises [Row.Shape_error] instead. *)

val tracing_in_progress : unit -> bool
(** Whether a shape-scheme trace is running. *)

val as_tracing : label:string -> (unit -> 'a) -> 'a
(** [as_tracing ~label f] runs [f] as part of the shape-scheme trace [label]: its update steps are
    not counted or timed as inline registrations. {!Scheme.trace} runs its function this way, and
    [Tensor.Scheme] makes a trace's tracers this way just before the trace starts. *)

type registration_counts = {
  checked_applications : int;
  inline_registrations : int;
  instantiated_constraints : int;
  instantiated_solutions : int;
  instantiated_boundary : int;
  inline_constraints : int;
  closures : int;
  closure_constraints : int;
  rank_search_visits : int;
  dim_cycle_visits : int;
  subst_visits : int;
}
[@@deriving sexp_of]
(** Registration statistics (gh-ocannl-1752), process-wide since the last reset: checked [%op]
    applications, and every registration outside a trace that ran [get_inequalities]. Projection
    derivation is not counted.

    The constraint counts (gh-ocannl-1752) say what each source adds to the open constraint store: a
    scheme instantiation's renamed trace constraints (C'), its renamed solution equations (E'), and
    its boundary and anchor equations -- checked instantiations included, though their window is
    rewound -- and every registration's own inequalities outside a trace. [closure_constraints] sums
    the store's size at the entry of each [finish_inference] ([closures] counts them).

    [rank_search_visits] is the rank-cycle search's work counter (gh-ocannl-1802): the states it
    visited, for every rank fact recorded since the reset, inside traces too. [dim_cycle_visits] is
    the dimension-cycle search's (gh-ocannl-1891): the variables it visited. [subst_visits] is
    solution substitution's (gh-ocannl-1904): the environment entries solutions visited. *)

type registration_times = {
  tracing_s : float;
  instantiation_s : float;
  inline_steps_s : float;
  closure_s : float;
}
[@@deriving sexp_of]
(** Where shape-registration wall time goes (gh-ocannl-1752), process-wide since the last reset, in
    seconds: tracing schemes (the body on tracers, and the rewind), instantiating them (checked
    applications and explicit instantiations), every step registered outside a trace, and
    [finish_inference]. A phase nested in another (an instantiation inside a closure built
    mid-construction, say) is charged to the inner phase alone. Construction work that is not shape
    registration (building tensors, their code) is in none of them. *)

type phase = [ `Tracing | `Instantiation | `Inline_step | `Closure ]

val timed : phase -> (unit -> 'a) -> 'a
(** Runs the function, charging its wall time, less that of nested timed phases, to the phase. *)

val registration_times : unit -> registration_times
val reset_registration_times : unit -> unit
val registration_counts : unit -> registration_counts
val reset_registration_counts : unit -> unit
val note_checked_application : unit -> unit

(** {3 Checkpoints (gh-ocannl-903)}

    The shape-inference half of [Tensor.checkpoint]: use that. A checkpoint snapshots the open
    frontier -- this module's and [Row]'s module-level session state (persistent values, captured as
    pointers), and the mutable fields of the shapes and steps of the active update steps and of
    every live capture reference -- and a rewind puts them back and discards what the window
    created. *)
module Checkpoint : sig
  type t

  val label : t -> string
  val is_closure : t -> bool

  val refusal : t -> string option
  (** Why the checkpoint can no longer be rewound to: a closure since it was taken, a rewind past it
      (or to the last closure, or a reset), a raising committed-state read of a shape open at the
      checkpoint, or a shape older than it outside its frontier. [None] when it can be. Checked
      before a rewind writes anything. *)

  val categories : string list
  (** The restore categories {!rewind} can be told to [omit]: negative controls only. *)

  val rewind : ?omit:string -> t -> provenance:string -> unit
  (** Discards the window's shapes, update steps and the capture references it made and touched
      (with [provenance]); restores the frontier's objects and the module-level state; clears the
      taint. The caller checks {!refusal} first. *)

  val differences : t -> string list
  (** The components of the session that differ from the checkpoint, by name: a diagnostic. *)

  val sizes : t -> (string * int) list
  (** The open frontier's sizes at the checkpoint: a diagnostic. *)
end

val checkpoint : label:string -> id_mark:int -> Checkpoint.t
(** An explicit checkpoint; [id_mark] is the next tensor id. Refuse it while tainted or within a
    construction window first ([Tensor.checkpoint] does). *)

val closure_checkpoint : id_mark:int -> Checkpoint.t
(** The last closure's checkpoint (empty frontier), which stays valid until the next closure. *)

val discarded_registry_sizes : unit -> (string * int) list

val with_memoized_draw_coefficient : (unit -> 'a) -> 'a
(** A negative control for tests (gh-ocannl-1890): while [f] runs, every packed uint4x32 conversion
    reuses the first coefficient registered, instead of a lazy over its own result node's precision.
*)

(** The records the session holds of what rewinds discarded, by kind: a diagnostic for tests. *)

val get_projections : update_step -> Ir.Indexing.projections
(** Returns the projections for this update step, computing them if not already done. This triggers
    [finish_inference] and then retrieves the projections from [unsafe_projections]. Use this
    instead of [derive_projections] directly. *)

val of_spec : ?deduced:deduce_within_shape -> debug_name:string -> id:int -> string -> t
val default_display_indices : t -> int array

val to_bases : t -> string array
(** The per-axis dimension bases (semantic annotations); the basis is total, so unannotated axes
    read back as ["default"] and the broadcast top as ["bcast_if_1"]. Uses the matrix convention of
    putting the input axes last. *)

val to_bases_bio : t -> string array * string array * string array
(** Per-kind dimension bases [(batch, input, output)], so callers can check which row a basis landed
    on rather than only that it appears somewhere in the flattened {!to_bases} list. *)

val parse_n5_layout : string -> int array
(** Parse a N5_layout priority string (e.g., "0,1,2") into display indices. Only supports integer
    labels (Fixed_index). *)

(** {2 Shape schemes (gh-ocannl-404)}

    The shape-inference half of [Tensor.Scheme]: use that. A scheme is the constraints a traced
    function registers, read off the session before the trace's rewind, with the solution equations
    of their variables; an instantiation renames the variables the trace minted (the generalized
    ones), keeps the older ones (shared parameters' and every other pre-existing tensor's: fixed),
    and relates the scheme's boundary rows to given argument and result shapes. It runs no function
    body and registers no update step: the next closure applies the solution to the boundary shapes
    and closes them, and finalizes the scheme's internal rows (renamed) as it finalizes an update
    step's rows. *)
module Scheme : sig
  type shape := t
  type t

  (** Negative controls for the tests: each disables one part of the mechanism. *)
  type sabotage =
    | No_generalization  (** Rename nothing: applications share the trace's variables. *)
    | Sever_shared
        (** Rename every variable and drop the anchors: no application stays tied to pre-existing
            shapes. *)
    | No_marks  (** Re-add none of the trace's policy marks ([instantiate] only). *)
    | No_internal_rows
        (** Register none of the scheme's internal rows with the closure ([instantiate] only). *)
    | No_end_validation  (** Skip the end-of-trace solve of the whole window ([trace] only). *)
    | No_ref_inputs
        (** Relate and bind none of the scheme's capture inputs ([instantiate] and [check]). *)
    | No_draw_holes
        (** Keep the trace's own draw coefficients, and their nodes, in the scheme ([trace] only).
        *)
    | No_draw_refusal  (** Instantiate a scheme with random draws ([instantiate] only). *)

  val label : t -> string
  val arity : t -> int
  val num_results : t -> int

  val to_string : t -> string
  (** See [Tensor.Scheme.to_string]. *)

  val trace :
    ?sabotage:sabotage ->
    var_mark:int ->
    label:string ->
    args:shape list ->
    (unit -> shape list) ->
    t
  (** [trace ~var_mark ~label ~args f] runs [f], which returns the result shapes, and reads the
      scheme off the session. [args] are the tracers' shapes, made after [var_mark] was read with
      [Row.var_mark] (so their variables are generalized) and before this call (so their own
      constraints are not the scheme's). Must run inside a rewound window ([Tensor.with_rewind]),
      which undoes the trace. While [f] runs, [finish_inference] raises. Refuses, with
      [Untraceable_scheme] on what a scheme cannot carry: a total-elements constraint with a lazy
      coefficient of unknown provenance, a capture reference made outside the window and set or
      related in it ([set_dim] and the like), or forcing dims or projections; and [Row.Shape_error]
      on a contradiction among the window's constraints (solved at Stage1 once more at the end, so a
      trailing [set_dim] is checked too). Both are prefixed ["tracing the shape scheme <label>: "].

      A capture reference made outside the window that [f]'s operations bind ([~capture_dims]) is an
      input of the scheme (gh-ocannl-1791): during the trace it reads as unbound and unsolved, so
      the scheme does not depend on its state then, and each application relates the variable of its
      first binding in [f] to its state at that application -- an open binding, or a solution held
      across a closure -- as the body's own first binding of it would.

      A random draw in [f] (a packed uint4x32 conversion, gh-ocannl-1890) is carried with its
      coefficient -- values per 128-bit block -- as a hole. The coefficient is a lazy over the
      conversion node's own [storage_prec], which the node's consumers set top-down
      ([update_infer_prec]) as they register, and which only Stage2 and later stages force: so it
      reads its call site's precision at closure, with no step of its own. A scheme never forces it,
      since its trace, its checks and its instances solve Stage1 only; the conversion node the body
      registers at an application fills it there. The hole holds none of the trace's nodes. *)

  val instantiate :
    ?sabotage:sabotage -> t -> application:string -> args:shape list -> results:shape list -> unit
  (** Adds the scheme's renamed constraints and the equalities of its boundary rows with [args] and
      [results] (and of its anchored pre-existing shapes with their current rows), solves them at
      Stage1, and registers the boundary shapes and the internal rows with the next closure. Every
      added constraint carries an origin naming the scheme and [application], so a closure-time
      failure names them too. Raises [Row.Shape_error] prefixed
      ["applying the shape scheme <label> to <application>: "]: before anything is written, on an
      arity mismatch, a boundary or anchored shape discarded by a rewind, or a closure since the
      trace when the scheme depends on open state no anchored shape holds; after registering
      (tainting the session, as a failed construction does), on a contradiction. The scheme's
      capture inputs are related to the references' current state and bound, to the variable of
      their last binding in the trace, so their solutions are published -- after the Stage1 solve
      and at the closure -- as the body's would be. A scheme with random draws is refused before
      anything is written: the closure forces a draw's coefficient, which only the conversion node
      of the body's own application settles (raised as [Untraceable_scheme] while a trace runs). *)

  val check : ?sabotage:sabotage -> t -> application:string -> args:shape list -> unit
  (** [instantiate] against [args] alone, relating none of the scheme's results, binding no capture
      reference (its capture inputs are only related to their current state), then puts back
      everything it wrote (the session and solver globals, the taint, the rows of the arguments and
      anchored shapes): an argument-dependent shape error raises as [instantiate]'s does, and the
      session is unchanged either way. Costs what [instantiate] does, independent of the open
      frontier. *)

  val stale : t -> bool
  (** Whether {!instantiate} would refuse the scheme for a closure since its trace (the scheme
      depends on open state no anchored shape holds): trace it again. *)
end

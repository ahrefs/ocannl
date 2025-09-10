(** {1 Construction of runtime-compiled code supporting backpropagation.} *)

open Base

type ndarray = Ir.Ndarray.t
type tn = Ir.Tnode.t
type tn_set = Set.M(Ir.Tnode).t
type asgns = Ir.Assignments.t
type comp = Ir.Assignments.comp
type fetch_op = Ir.Assignments.fetch_op
type projections = { projections_debug : string; projections : Ir.Indexing.projections Lazy.t }

type diff = {
  grad : tn;
  zero_grads : asgns;  (** Prepares for backpropagation. Beware of the "missing zero_grads" bug. *)
  backprop : comp;
      (** Backpropagates for the tensor and its descendants; which typically means adding partial
          gradients to the gradient tensor of the subtensors, then for sub-subtensors etc. *)
}

type t = {
  params : (t, comparator_witness) Base.Set.t;
      (** Parameters [t.params] are the descendants of [t] whose {!field:diff} is not [None] and
          whose {!field:forward} code is not included in [t.forward] as it is meant for
          initialization. *)
  forward : comp;
  diff : diff option;
  id : int;  (** Same as [value.id]. *)
  value : tn;
  top_down_prec : bool;  (** Whether to propagate precision bottom-up (the default) or top-down. *)
  shape : Shape.t;
      (** The eventual shape of [t.value] and [t.diff.grad], incorporating the current state of
          shape inference. *)
  children : subtensor list;
}
[@@deriving sexp_of]
(** Information needed for compositional code generation. *)

and subtensor = {
  subtensor : t;
  embedded : bool;
      (** A tensor can be an [embedded] child at most once -- that's where its [forward] computation
          ends up when used as part of a bigger computation. *)
}

and comparator_witness

val comparator : (t, comparator_witness) Base.Comparator.t

val init_params : ?skip:'a Map.M(Ir.Tnode).t -> t -> comp
(** [init_params ?skip t] collects into a single sequence the {!field:forward} code of [t.params],
    and transitively the initializations of the parameters of the parameters. If [skip] is provided,
    it is used to filter out the parameters belonging to [skip] (e.g. [skip] can be a set of
    parameters that are already initialized). NOTE: it always outputs code with a block comment,
    even if the params set is empty. *)

val is_fwd_root : t -> bool
val remove_fwd_root : t -> unit
val is_bprop_root : t -> bool
val remove_bprop_root : t -> unit
val with_unchanged_roots : f:(unit -> 'a) -> 'a

val default_value_prec : Ir.Ops.prec ref
(** The default precision for the value node of terminal (i.e. non-composite) tensors.

    Note: the precision of a node can be set arbitrarily via {!Ir.Tnode.update_prec}. The default
    precision for value nodes of composite tensors is the maximum of precisions of the value nodes
    of sub-tensors. *)

val default_grad_prec : Ir.Ops.prec ref
(** The default precision for the gradient node of terminal (i.e. non-composite) tensors.

    Note: the precision of a node can be set arbitrarily via {!Ir.Tnode.update_prec}. The default
    precision for gradient nodes of composite tensors is the maximum of precisions of the gradient
    nodes of sub-tensors. *)

exception Session_error of string * t option

val max_sublabel_length : int ref

val raw_ternop :
  initialize_neutral:bool ->
  accum:Ir.Ops.binop ->
  t:t ->
  lhs_is_grad:bool ->
  op:Ir.Ops.ternop ->
  t1:t ->
  rhs1_is_grad:bool ->
  rhs1_is_merge:bool ->
  t2:t ->
  rhs2_is_grad:bool ->
  rhs2_is_merge:bool ->
  t3:t ->
  rhs3_is_grad:bool ->
  rhs3_is_merge:bool ->
  logic:Shape.ternary_type ->
  asgns

val raw_binop :
  initialize_neutral:bool ->
  accum:Ir.Ops.binop ->
  t:t ->
  lhs_is_grad:bool ->
  op:Ir.Ops.binop ->
  t1:t ->
  rhs1_is_grad:bool ->
  rhs1_is_merge:bool ->
  t2:t ->
  rhs2_is_grad:bool ->
  rhs2_is_merge:bool ->
  logic:Shape.compose_type ->
  asgns

val raw_unop :
  initialize_neutral:bool ->
  accum:Ir.Ops.binop ->
  t:t ->
  lhs_is_grad:bool ->
  op:Ir.Ops.unop ->
  t1:t ->
  rhs_is_grad:bool ->
  rhs_is_merge:bool ->
  logic:Shape.transpose_type ->
  asgns

type grad_spec = Require_grad | Prohibit_grad | If_needed

val is_prohibit_grad : grad_spec -> bool

type param_op_fun =
  ?input_dims:int list ->
  ?output_dims:int list ->
  ?input_axes:(string * int) list ->
  ?output_axes:(string * int) list ->
  ?deduced:Shape.deduce_within_shape ->
  unit ->
  t

type op_fun =
  ?label:string list ->
  ?top_down_prec:bool ->
  ?batch_dims:int list ->
  ?batch_axes:(string * int) list ->
  param_op_fun

val binop :
  ?compose_op:Shape.compose_type ->
  op_asn:(v:tn -> t1:t -> t2:t -> projections:projections -> comp) ->
  grad_asn:(t:t -> g:tn -> t1:t -> t2:t -> projections:projections -> comp) ->
  ?grad_spec:grad_spec ->
  t ->
  t ->
  op_fun
(** The defaults are pointwise operations. The [grad_asn] function receives the non-differentiable
    variant of the tensor as an argument, which can be used to access the tensor's value in a tensor
    expression. *)

val unop :
  ?transpose_op:Shape.transpose_type ->
  op_asn:(v:tn -> t1:t -> projections:projections -> comp) ->
  grad_asn:(t:t -> g:tn -> t1:t -> projections:projections -> comp) ->
  ?grad_spec:grad_spec ->
  t ->
  op_fun
(** See {!binop} -- same comments apply. *)

val ternop :
  ?ternary_op:Shape.ternary_type ->
  op_asn:(v:tn -> t1:t -> t2:t -> t3:t -> projections:projections -> comp) ->
  grad_asn:(t:t -> g:tn -> t1:t -> t2:t -> t3:t -> projections:projections -> comp) ->
  ?grad_spec:grad_spec ->
  t ->
  t ->
  t ->
  op_fun
(** See {!binop} -- same comments apply. *)

val term :
  ?init_data:Ir.Assignments.init_data -> ?fetch_op:fetch_op -> ?grad_spec:grad_spec -> op_fun

(** A terminal: a constant, a parameter, an input of the model. The semantics of shape specification
    is the same as in {!Shape.make}, and by default the shape will be inferred. At most one of
    [init_data] or [fetch_op] should be provided. If [init_data] is provided, it is used to
    initialize the tensor's [value] node. If [fetch_op] is provided, it is used to generate the
    tensor's forward code. If [init_data] is provided, it is also used to verify the shape of the
    tensor's [value] node: [Reshape] (the default) if the data is not padded and both the tensor's
    shape and padding are inferred, [Keep_shape_no_padding] if the tensor should not be padded and
    the shape is as given by the ndarray, and [Padded] if the data is already padded as given, and
    the shape is as given by the ndarray. *)

val number : ?label:string list -> ?axis_label:string -> ?grad_spec:grad_spec -> float -> t
(** A number: a tensor with a single axis of one dimension, initialized to the given value.
    [grad_spec] is by default [Prohibit_grad]. *)

val bits : ?label:string list -> ?axis_label:string -> ?grad_spec:grad_spec -> int64 -> t
(** A number with exact bit representation: a tensor with a single axis of one dimension,
    initialized to the given int64 value. Useful for initializing uint4x32 tensors where exact bit
    patterns matter. [grad_spec] is by default [Prohibit_grad]. *)

val ndarray : ?grad_spec:grad_spec -> float array -> op_fun
(** A tensor with an explicit shape, initialized to the given values. Omitted shape rows default to
    no axes. [grad_spec] is by default [Prohibit_grad]. If [strict] is [true] (the default), the
    given values must fill the tensor's [value] node precisely; otherwise, the values will be looped
    over to populate the [value] node. *)

val param : t:op_fun -> string -> ?more_label:string list -> param_op_fun
(** For proper parameters, [t] should produce a tensor with no batch axes; input and output axes
    should by default be inferred; [grad_spec] should be [Require_grad]. [t]'s label is the passed
    string, appended by [more_label] if any, other parameters are forwarded to [t]. This function
    returns [t]'s result with the field {!field:params} replaced by a singleton set containing that
    result, and it also updates the memory modes. *)

val term_init : ?grad_spec:grad_spec -> float array -> op_fun
(** A {!term} wrapper that sets up the value node initialization (it generalizes {!ndarray} to
    tensors with inferred shapes). *)

val consume_forward_code : t -> comp
(** A forward root is a tensor that is not (currently) used to compute another tensor.
    [consume_forward_code t] ensures [t] is a forward root, removes it from forward roots, and
    checks that there are no other forward roots for tensors with children. *)

val consume_backprop_code : t -> comp
(** A backprop root is a tensor with a gradient that is not (currently) receiving gradients from
    another tensor. I.e. it is not currently used to compute a tensor with a gradient.
    [consume_backprop_code t] ensures [t] is a backprop root, removes it from backprop roots, and
    checks that there are no other backprop roots for tensors with children. It returns the backprop
    code -- note this does not include the zero_grads code. *)

val iter_embedded : f:(tn -> unit) -> t -> unit
(** [iter_embedded t] iterates over all descendant nodes that are embedded, i.e. are members of
    [t.forward.embedded_nodes] or '[t.diff.backprop.embedded_nodes]' (if any). Note: [iter_embedded]
    should only be called after shape inference finishes. *)

val unsafe_reinitialize : unit -> unit
(** Bring global state to its initialization values. This invalidates any previously defined tensors
    and tensor nodes. Also reinitializes the modules: {!Shape}, {!Ir.Tnode}.

    While this function is intended for testing, using it can prevent unintentional session state
    pollution errors. *)

val set_random_seed : ?seed:int -> unit -> unit
(** Creates the random seed tensor. If [seed] is provided, it is used to set the random seed.
    Otherwise, the seed is taken from the settings. *)

val get_random_seed : unit -> t
(** Returns a tensor with the current random seed. Lazily initialized using {!set_random_seed} and
    reset when {!unsafe_reinitialize} is called. IMPORTANT: all sites using the same global random
    seed, e.g. using [get_random_seed ()] not separated by a call to {!unsafe_reinitialize}, must
    descend from the first caller's optimization context. *)

(** {2 Printing.} *)

val header : t -> string
(** Converts ID, label and the dimensions of a node to a string. *)

val log_debug_info : from_log_level:int -> t -> unit
(** Logs debug information about the tensor on the default ppx_minidebug runtime. *)

type array_print_style =
  [ `Default
    (** The inner rectangles comprise both an input and an output axis, if available. Similarly, the
        outer rectangle comprises a second-from-end input axis and a second-from-end output axis, if
        available. At least one batch axis is output, when available. The axes that couldn't be
        output are printed at position/dimension [0]. *)
  | `N5_layout of string
    (** The string should provide exclusively non-negative integer pseudo-labels. The numbers
        [0]-[4] represent the priorities of the axes to be printed out, where the priorities
        correspond to, from highest: horizontal, vertical direction of the inner rectangle,
        horizontal, vertical direction of the outer rectangle, repetition (see also
        [Node.pp_print]). The numbers [n >= 5] stand for the actual positions [n - 5] within the
        corresponding axes. *)
  | `Label_layout of (string * int) list
    (** The association from axis labels to integers. The negative numbers [-5] to [-1] represent
        the priorities of the axes to be printed out, where the priorities correspond to, from
        highest: horizontal, vertical direction of the inner rectangle, horizontal, vertical
        direction of the outer rectangle, repetition (as above). The numbers [n >= 0] stand for the
        actual positions within the corresponding axes. Unspecified axes are printed at position
        [0]. *)
  | `Inline
    (** The tensors are printed linearly, in a bracketed manner, optionally prefixed with the labels
        specification. Note that the syntax causes ambiguity for 1-dimensional input axes
        (underscores are used for axes without explicit labels); when there is a 1-dimensional input
        axis, we output the labels specification even if there are no axis labels as a way to
        display the number of axes. The axis nesting is right-to-left (rightmost is innermost). The
        input axes are innermost and the batch axes outermost. The input axes use [,] as a separator
        and [()] as axis delimiters, but the delimiter for the outermost (i.e. leftmost) axis is
        omitted. The output axes use [;] as a separator and [[]] as axis delimiters (obligatory).
        The batch axes use [;] as a separator and [[||]] as axis delimiters (obligatory). *) ]
(** We print out up to 5 axes when printing a tensor, as a grid (outer rectangle) of (inner)
    rectangles, possibly repeated (screens). *)

val to_printbox :
  ?single_node:bool ->
  ?embedded_only:bool ->
  ?entries_per_axis:int ->
  ?with_id:bool ->
  ?force:bool ->
  ?with_shape:bool ->
  ?with_value:bool ->
  with_grad:bool ->
  depth:int ->
  t ->
  PrintBox.t

val to_doc :
  force:bool ->
  with_grad:bool ->
  with_code:bool ->
  ?with_low_level:bool ->
  array_print_style ->
  t ->
  PPrint.document

val print :
  ?here:Ppx_here_lib.position ->
  ?force:bool ->
  with_grad:bool ->
  with_code:bool ->
  ?with_low_level:bool ->
  array_print_style ->
  t ->
  unit

val print_forward_roots : with_grad:bool -> with_code:bool -> array_print_style -> unit

val print_tree :
  ?here:Ppx_here_lib.position ->
  ?force:bool ->
  ?entries_per_axis:int ->
  ?with_backend_info:bool ->
  ?with_id:bool ->
  ?with_shape:bool ->
  ?with_value:bool ->
  ?embedded_only:bool ->
  with_grad:bool ->
  depth:int ->
  t ->
  unit

val debug_name : t -> string

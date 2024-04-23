(** {1 Construction of runtime-compiled code supporting backpropagation.} *)

open Base

type tn = Arrayjit.Tnode.t
type asgns = Arrayjit.Assignments.t
type init_op = Arrayjit.Ops.init_op
type fetch_op = Arrayjit.Assignments.fetch_op
type projections = Arrayjit.Indexing.projections

type diff = {
  grad : tn;
  zero_grads : asgns;  (** Prepares for backpropagation. Always compile as: [Seq (zero_grads, backprop)]. *)
  backprop : asgns;
      (** Backpropagates for the tensor and its descendants; which typically means adding partial gradients to
          the gradient tensor of the subtensors, then for sub-subtensors etc. *)
}

type t = {
  forward : asgns;
  diff : diff option;
  id : int;  (** Same as [value.id]. *)
  value : tn;
  shape : Shape.t;
      (** The eventual shape of [t.value] and [t.diff.grad], incorporating the current state of shape
          inference. *)
  children : subtensor list;
}
[@@deriving sexp_of]
(** Information needed for compositional code generation. *)

and subtensor = { subtensor : t; embedded : bool }

type comparator_witness

val comparator : (t, comparator_witness) Base.Comparator.t
val is_fwd_root : t -> bool
val remove_fwd_root : t -> unit
val is_bprop_root : t -> bool
val remove_bprop_root : t -> unit
val with_unchanged_roots : f:(unit -> 'a) -> 'a
val default_value_prec : Arrayjit.Ops.prec ref
val default_grad_prec : Arrayjit.Ops.prec ref

exception Session_error of string * t option

val max_sublabel_length : int ref

val raw_binop :
  initialize_neutral:bool ->
  accum:Arrayjit.Ops.binop ->
  t:t ->
  lhs_is_grad:bool ->
  op:Arrayjit.Ops.binop ->
  t1:t ->
  rhs1_is_grad:bool ->
  t2:t ->
  rhs2_is_grad:bool ->
  logic:Shape.compose_type ->
  asgns

val raw_unop :
  initialize_neutral:bool ->
  accum:Arrayjit.Ops.binop ->
  t:t ->
  lhs_is_grad:bool ->
  op:Arrayjit.Ops.unop ->
  t1:t ->
  rhs_is_grad:bool ->
  logic:Shape.transpose_type ->
  asgns

type grad_spec = Require_grad | Prohibit_grad | If_needed

val is_prohibit_grad : grad_spec -> bool

val op :
  label:string list ->
  ?compose_op:Shape.compose_type ->
  ?transpose_op:Shape.transpose_type ->
  ?init_op:init_op ->
  op_asn:(v:tn -> projections:projections Lazy.t -> asgns) ->
  grad_asn:(v:tn -> g:tn -> projections:projections Lazy.t -> asgns) ->
  ?grad_spec:grad_spec ->
  (debug_name:string -> id:int -> Shape.t) ->
  t list ->
  t

val binop :
  label:string list ->
  ?compose_op:Shape.compose_type ->
  op_asn:(v:tn -> t1:t -> t2:t -> projections:projections Lazy.t -> asgns) ->
  grad_asn:(v:tn -> g:tn -> t1:t -> t2:t -> projections:projections Lazy.t -> asgns) ->
  ?grad_spec:grad_spec ->
  t ->
  t ->
  t

val unop :
  label:string list ->
  ?transpose_op:Shape.transpose_type ->
  op_asn:(v:tn -> t1:t -> projections:projections Lazy.t -> asgns) ->
  grad_asn:(v:tn -> g:tn -> t1:t -> projections:projections Lazy.t -> asgns) ->
  ?grad_spec:grad_spec ->
  t ->
  t

val term :
  label:string list ->
  grad_spec:grad_spec ->
  ?batch_dims:int list ->
  ?input_dims:int list ->
  ?output_dims:int list ->
  ?batch_axes:(string * int) list ->
  ?input_axes:(string * int) list ->
  ?output_axes:(string * int) list ->
  ?deduced:Shape.deduce_within_shape ->
  ?init_op:init_op ->
  ?fetch_op:(v:tn -> fetch_op) ->
  unit ->
  t
(** A terminal: a constant, a parameter, an input of the model. The semantics of shape specification is the
    same as in {!Shape.make}, and by default the shape will be inferred. *)

val number : ?label:string list -> ?axis_label:string -> ?grad_spec:grad_spec -> float -> t
(** A number: a tensor with a single axis of one dimension, initialized to the given value. [grad_spec] is by
    default [Prohibit_grad]. *)

val ndarray :
  ?label:string list ->
  ?grad_spec:grad_spec ->
  ?batch_dims:int list ->
  ?input_dims:int list ->
  ?output_dims:int list ->
  ?batch_axes:(string * int) list ->
  ?input_axes:(string * int) list ->
  ?output_axes:(string * int) list ->
  ?strict:bool ->
  float array ->
  t
(** A tensor with an explicit shape, initialized to the given values. Omitted shape rows default to no axes.
    [grad_spec] is by default [Prohibit_grad]. If [strict] is [true] (the default), the given values must fill
    the tensor's [value] node precisely; otherwise, the values will be looped over to populate the [value]
    node. *)

val param :
  ?input_dims:int list ->
  ?output_dims:int list ->
  ?input_axes:(string * int) list ->
  ?output_axes:(string * int) list ->
  ?deduced:Shape.deduce_within_shape ->
  ?strict:bool ->
  ?values:float array ->
  string ->
  t
(* A tensor with no batch axes; input and output axes are by default inferred. [grad_spec] is set to
   [Require_grad]. *)

val iter_embedded_arrays : f:(tn -> unit) -> t -> unit

val consume_forward_code : t -> asgns
(** A forward root is a tensor that is not (currently) used to compute another tensor.
    [consume_forward_code t] ensures [t] is a forward root, removes it from forward roots, and checks that
    there are no other forward roots for tensors with children. *)

val consume_backprop_code : t -> asgns * asgns
(** A backprop root is a tensor with a gradient that is not (currently) receiving gradients from another
    tensor. I.e. it is not currently used to compute a tensor with a gradient. [consume_backprop_code t]
    ensures [t] is a backprop root, removes it from backprop roots, and checks that there are no other
    backprop roots for tensors with children. *)

(** {2 Printing.} *)

val header : t -> string
(** Converts ID, label and the dimensions of a node to a string. *)

type array_print_style =
  [ `Default
    (** The inner rectangles comprise both an input and an output axis, if available. Similarly, the outer
        rectangle comprises a second-from-end input axis and a second-from-end output axis, if available. At
        least one batch axis is output, when available. The axes that couldn't be output are printed at
        position/dimension [0]. *)
  | `N5_layout of string
    (** The string should provide exclusively non-negative integer pseudo-labels. The numbers [0]-[4]
        represent the priorities of the axes to be printed out, where the priorities correspond to, from
        highest: horizontal, vertical direction of the inner rectangle, horizontal, vertical direction of the
        outer rectangle, repetition (see also [Node.pp_print]). The numbers [n >= 5] stand for the actual
        positions [n - 5] within the corresponding axes. *)
  | `Label_layout of (string * int) list
    (** The association from axis labels to integers. The negative numbers [-5] to [-1] represent the
        priorities of the axes to be printed out, where the priorities correspond to, from highest:
        horizontal, vertical direction of the inner rectangle, horizontal, vertical direction of the outer
        rectangle, repetition (as above). The numbers [n >= 0] stand for the actual positions within the
        corresponding axes. Unspecified axes are printed at position [0]. *)
  | `Inline
    (** The tensors are printed linearly, in a bracketed manner, optionally prefixed with the labels
        specification. Note that the syntax causes ambiguity for 1-dimensional input axes (underscores are
        used for axes without explicit labels); when there is a 1-dimensional input axis, we output the labels
        specification even if there are no axis labels as a way to display the number of axes. The axis
        nesting is right-to-left (rightmost is innermost). The input axes are innermost and the batch axes
        outermost. The input axes use [,] as a separator and [()] as axis delimiters, but the delimiter for
        the outermost (i.e. leftmost) axis is omitted. The output axes use [;] as a separator and [[]] as axis
        delimiters (obligatory). The batch axes use [;] as a separator and [[||]] as axis delimiters
        (obligatory). *) ]
(** We print out up to 5 axes when printing a tensor, as a grid (outer rectangle) of (inner) rectangles,
    possibly repeated (screens). *)

val to_printbox :
  ?single_node:bool ->
  ?entries_per_axis:int ->
  ?with_id:bool ->
  ?with_shape:bool ->
  ?with_value:bool ->
  with_grad:bool ->
  depth:int ->
  t ->
  PrintBox.t

val print :
  with_grad:bool -> with_code:bool -> ?force:bool -> ?with_low_level:bool -> array_print_style -> t -> unit

val print_forward_roots : with_grad:bool -> with_code:bool -> array_print_style -> unit

val print_tree :
  ?entries_per_axis:int ->
  ?with_backend_info:bool ->
  ?with_id:bool ->
  ?with_shape:bool ->
  ?with_value:bool ->
  with_grad:bool ->
  depth:int ->
  t ->
  unit

(** {2 Accessors.} *)

val value_1d_points : ?from_axis:int -> xdim:int -> t -> float array
val value_2d_points : ?from_axis:int -> xdim:int -> ydim:int -> t -> (float * float) array
val grad_1d_points : ?from_axis:int -> xdim:int -> t -> float array
val grad_2d_points : ?from_axis:int -> xdim:int -> ydim:int -> t -> (float * float) array
val set_value : t -> int array -> float -> unit
val get_value : t -> int array -> float
val set_grad : t -> int array -> float -> unit
val get_grad : t -> int array -> float
val set_values : t -> float array -> unit
val get_values : t -> float array

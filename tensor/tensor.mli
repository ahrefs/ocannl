(** {1 Construction of runtime-compiled code supporting backpropagation.} *)

open Base

type ndarray = Ir.Ndarray.t
type tn = Ir.Tnode.t
type tn_set = Set.M(Ir.Tnode).t
type asgns = Ir.Assignments.t
type comp = Ir.Assignments.comp
type fetch_op = Ir.Assignments.fetch_op

type projections = {
  projections_debug : string;
  projections : Ir.Indexing.projections Lazy.t;
  product_shape : Shape.t Lazy.t;
      (** The product-space proxy shape of the operation (gh-512), for [%cd] [*_pspace]
          intermediates; see {!Shape.product_space_shape}. Forcing it emits shape constraints, so it
          must be forced (if at all) while the operation's code is being built. Unlike
          [projections], it does not need the operation's shape update step, which {!op} creates
          only after [op_asn] returns: forcing [projections] from within an [op_asn] is rejected. *)
}

type handout =
  | Not_handed_out
  | Taken
      (** What became of one side of a tensor's code once it stopped being a root: [Taken] by
          {!consume_forward_code}, {!take_forward_code} (the [%cd] embedding path) or
          {!consume_backprop_code}; [Not_handed_out] while it is a root, and for a parameter or a
          subterm whose code a consumer embeds. *)

type owner_cell
(** The tensor's forward owner cell (gh-ocannl-1796): whether its forward code is owned by an
    unconsumed forward root, kept apart from the session's weak root map so that
    {!consume_forward_code}'s refusal does not depend on what the GC has reclaimed. *)

type consumption = { mutable fwd : handout; mutable bprop : handout; fwd_owner : owner_cell }
(** A mutable cell shared by every copy of the record, so the marker follows the tensor; it is what
    lets the consume functions say "already consumed" rather than "not a root". *)

type diff = {
  grad : tn;
  zero_grads : comp;
      (** Prepares for backpropagation: zeroes this tensor's gradient and, transitively, the
          gradients its backprop accumulates into. Beware of the "missing zero_grads" bug. The
          gradients are its {!Ir.Assignments.field-embedded_nodes}, so it can be compiled as a
          routine on its own (see {!Train.zero_params_grads}). *)
  backprop : comp;
      (** Backpropagates for the tensor and its descendants; which typically means adding partial
          gradients to the gradient tensor of the subtensors, then for sub-subtensors etc. *)
}

type t = {
  params : (t, comparator_witness) Base.Set.t;
      (** Parameters [t.params] are the descendants of [t] whose {!field:forward} code is not
          included in [t.forward] as it is meant for initialization. This is the {e state} set —
          what must be initialized, saved and restored for [t]'s forward pass, including
          non-differentiable running statistics — so a parameter detached behind
          [Operation.stop_gradient] (a frozen backbone) is deliberately included even though [t]'s
          backprop never writes its gradient. Optimizer-side consumers want the {e trained} subset
          instead: [Train.trainable_params] derives it from the backprop code (gh-ocannl-673). *)
  forward : comp;
  diff : diff option;
  value : tn;
      (** The session id and namespace of the tensor are [value.id] / [value.namespace]; tensor
          identity (comparison) follows [value.uid]. *)
  top_down_prec : bool;
      (** Whether to propagate precision bottom-up (the default) or top-down. A bottom-up tensor
          promotes over its operands; one none of whose operands, transitively, states a precision
          also follows its consumers, promoting and never narrowing (gh-ocannl-2028,
          [docs/precision_inference.md]). *)
  shape : Shape.t;
      (** The eventual shape of [t.value] and [t.diff.grad], incorporating the current state of
          shape inference. *)
  children : subtensor list;
      (** Hand-built records must keep every member of [params] reachable through [children]. *)
  consumption : consumption;
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
    and transitively the initializations of the parameters of the parameters. Initializers follow
    dependency order; independent fragments follow first access in [t]'s forward statements, with
    tensor operand order breaking ties. Seed literals are written before the draws; changing this
    statement order preserves the random values of a fixed graph (node counters are unchanged). If
    [skip] is provided, it is used to filter out the parameters belonging to [skip] (e.g. [skip] can
    be a set of parameters that are already initialized). The code also writes every random seed the
    selected initializers read whose own code lives elsewhere -- e.g. absorbed by a discarded draw
    -- by replaying its constant literal, whichever seed is the {!get_random_seed} singleton at the
    time (see {!with_saved_random_seed}). NOTE: it always outputs code with a block comment, even if
    the params set is empty. *)

val is_fwd_root : t -> bool
(** Whether [t]'s forward code is still unconsumed: no consumer embedded it and it was not handed
    out. The session holds its forward roots WEAKLY, checkpoints included (gh-ocannl-1796), as it
    does its backprop roots: a graph built and dropped without ever being evaluated is reclaimed by
    the GC, as is a root only the last closure's checkpoint captured. {!consume_forward_code}'s
    refusal of a tensor whose reads another forward root computes does not read the map, so it does
    not depend on the GC: a root dropped before its code was consumed keeps refusing its readers. *)

val is_bprop_root : t -> bool
(** Whether [t]'s backprop code is still unconsumed. The session holds its backprop roots WEAKLY,
    checkpoints included (gh-ocannl-1745). The entry lives while any record sharing [t]'s
    [consumption] cell is reachable from outside the session's root maps: [t] itself, a
    [{ t with ... }] copy that keeps ownership, or a tensor that holds [t] as a child. A
    differentiable graph whose forward code was handed out ({!Train.forward},
    {!Train.forward_once}), and which the caller then dropped, is reclaimed by the GC. A loss built
    over a tensor the caller still holds embeds its backprop as before. A dropped root no longer
    vetoes {!consume_backprop_code}; a dependence on its code is refused by the compilation
    lineage's requirement check ({!Context.run}).

    A reclaimed root, its backprop never consumed, can never run, so the leaves (terms and
    parameters) it absorbed, transitively, are backprop roots again (gh-ocannl-1804): the next
    consumer of such a leaf carries its gradient's zeroing. This reader, [Tensor.op] over such a
    leaf and every closure see it. Computed tensors the root absorbed are not given back.
    Reclamation is the GC's to time: a consumer dropped but not yet collected still holds the leaf,
    and [Gc.full_major ()] settles it. *)

val take_forward_code : t -> comp
(** The unchecked half of {!consume_forward_code}, for a tensor the caller has established is a
    forward root ({!is_fwd_root}): removes it from the forward roots, marks its forward code as
    taken, and returns the code. It is an EMBEDDING, not a handout: {!rewind_to_last_closure} treats
    an embedding made since the last closure as discarded work and makes the tensor a root again;
    code a caller keeps is handed out with {!consume_forward_code}. This is the handout every path
    that embeds a tensor's forward code goes through -- {!consume_forward_code} and the [%cd]
    operands' embedding alike -- so that a later {!consume_forward_code} on the same tensor returns
    the same comp. Like {!consume_forward_code}, it returns the code with the fetch of every random
    seed the code reads and does not write prepended (gh-ocannl-2157). *)

val with_unchanged_roots : f:(unit -> 'a) -> 'a
(** Runs [f] and restores the forward and backprop root maps afterwards, including the consumption
    markers of the roots that were live at entry. *)

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
val buffer_of : is_grad:bool -> is_merge:bool -> t -> Ir.Assignments.buffer

val raw_accum :
  initialize_neutral:bool ->
  accum:Ir.Ops.binop ->
  t:t ->
  lhs_is_grad:bool ->
  shape_logic:Shape.logic ->
  rhs:Ir.Assignments.accum_rhs ->
  asgns

type grad_spec = Require_grad | Prohibit_grad | If_needed [@@deriving sexp_of]

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
(** Labels are collected in tensor construction order, with more specific information first. *)

val binop :
  ?op_label:string ->
  ?compose_op:Shape.compose_type ->
  op_asn:(t:t -> t1:t -> t2:t -> projections:projections -> comp) ->
  grad_asn:(t:t -> g:tn -> t1:t -> t2:t -> projections:projections -> comp) ->
  ?grad_spec:grad_spec ->
  t ->
  t ->
  op_fun
(** The defaults are pointwise operations. The [grad_asn] function receives the non-differentiable
    variant of the tensor as an argument, which can be used to access the tensor's value in a tensor
    expression. [op_label] is prepended to [label] of the {!op_fun} signature.

    The operation's neutral element is read off the assignments [op_asn] returns, and its shape
    update step is created afterwards, so [op_asn] must not force [projections.projections], nor
    finalize inference by any other route ([Shape.to_dims]/[to_padding]): the construction is
    rejected (a [Session_error], resp. a [Row.Shape_error] raised at the finalizer, before anything
    is committed). The rejection is a bug report, not a recovery API -- shape inference is not
    transactional (gh-ocannl-903), so the session is reset rather than continued.
    [projections.product_shape] and [*_pspace] intermediates are fine. An operation that needs shape
    information uses [Shape.set_dim] or the einsum capture syntax instead; captured references are
    bound at registration, after [op_asn], so relating them ([Shape.set_equal]/[set_scale]) belongs
    at the operation-expression level. *)

val unop :
  ?op_label:string ->
  ?transpose_op:Shape.transpose_type ->
  op_asn:(t:t -> t1:t -> projections:projections -> comp) ->
  grad_asn:(t:t -> g:tn -> t1:t -> projections:projections -> comp) ->
  ?grad_spec:grad_spec ->
  t ->
  op_fun
(** See {!binop} -- same comments apply. *)

val ternop :
  ?op_label:string ->
  ?ternary_op:Shape.ternary_type ->
  op_asn:(t:t -> t1:t -> t2:t -> t3:t -> projections:projections -> comp) ->
  grad_asn:(t:t -> g:tn -> t1:t -> t2:t -> t3:t -> projections:projections -> comp) ->
  ?grad_spec:grad_spec ->
  t ->
  t ->
  t ->
  op_fun
(** See {!binop} -- same comments apply. *)

val blockop :
  ?op_label:string ->
  spec:string ->
  ?spec_source:string ->
  delayed_vars:Shape.delayed_var_ref list ->
  op_asn:(t:t -> projections:projections -> comp) ->
  grad_asn:(t:t -> g:tn -> projections:projections -> comp) ->
  ?grad_spec:grad_spec ->
  t array ->
  op_fun
(** Like {!binop}, but uses block/concatenation shape logic over an array of RHS tensors.
    [spec_source] is the spec as written when [spec] was built from it at run time
    ({!Shape.einsum_spec}). *)

val term :
  ?init_data:Ir.Assignments.init_data -> ?fetch_op:fetch_op -> ?grad_spec:grad_spec -> op_fun
(** A terminal: a constant, a parameter, an input of the model. The semantics of shape specification
    is the same as in {!Shape.make}, and by default the shape will be inferred. At most one of
    [init_data] or [fetch_op] should be provided. If [init_data] is provided, it is used to
    initialize the tensor's [value] node. If [fetch_op] is provided, it is used to generate the
    tensor's forward code. If [init_data] is provided, it is also used to verify the shape of the
    tensor's [value] node: [Broadcast] fills the inferred shape with a scalar without constraining
    its element count, [Reshape] (the default) if the data is not padded and both the tensor's shape
    and padding are inferred, [Keep_shape_no_padding] if the tensor should not be padded and the
    shape is as given by the ndarray (a layout committed at creation, so a consumer demanding
    margins is refused), and [Padded] if the data is already padded as given, and the shape is as
    given by the ndarray.

    With [~grad_spec:Require_grad] the term is a differentiable leaf that is not a parameter: it
    stays out of {!field:params} (nothing initializes it; its value is its data or its forward
    code), and its gradient carries observation intent ({!Ir.Tnode.set_observable}), as a
    parameter's does -- the host and later routines are its only readers, so it can be read with
    [Context.get_values] or stepped through the optimizers' [?params] without a materialization
    hint. Its value carries the intent too, since a recomputed gradient reads it. A leaf whose value
    comes from [fetch_op] is re-established by every forward run, so an optimized input should be
    data-backed ([init_data]) or a parameter. *)

val number : ?label:string list -> ?axis_basis:string -> ?grad_spec:grad_spec -> float -> t
(** A number: a tensor with a single axis of one dimension, initialized to the given value.
    [axis_basis] sets the dimension basis (semantic annotation) of that axis. [grad_spec] is by
    default [Prohibit_grad]. *)

val bits : ?label:string list -> ?axis_basis:string -> ?grad_spec:grad_spec -> int64 -> t
(** A number with exact bit representation: a tensor with a single axis of one dimension,
    initialized to the given int64 value. Useful for initializing uint4x32 tensors where exact bit
    patterns matter. [grad_spec] is by default [Prohibit_grad]. *)

val ndarray : ?prec:Ir.Ops.prec -> ?grad_spec:grad_spec -> float array -> op_fun
(** A tensor with an explicit shape, initialized to the given values. Omitted shape rows default to
    no axes. [prec] fixes the value precision at construction, including for data-backed literals
    above [limit_constant_fill_size]. When omitted, small literals follow precision inference and
    data-backed ones use [default_value_prec]. [grad_spec] is by default [Prohibit_grad]. *)

val param : ?require_grad:bool -> t:op_fun -> string -> ?more_label:string list -> param_op_fun
(** [param] is intended for all tensors with initialization; these typically are parameters, but
    [param] is also used for non-differentiable tensors that need to be initialized. For proper
    parameters, pass [require_grad:true]. [t] should produce a tensor with no batch axes; input and
    output axes should by default be inferred. [t]'s label is the passed string, appended by
    [more_label] if any, other parameters are forwarded to [t]. This function returns [t]'s result
    with the field {!field:params} replaced by a singleton set containing that result, and it also
    updates the memory modes. If [require_grad] is true, any gradient structure inherited from the
    initialization expression is replaced by a fresh gradient for the final parameter value only.
    The result has fresh consumption state, independent of any prior handouts of [t]’s result.

    The float nodes of the initialization expression are at least as precise as the parameter,
    whatever its precision comes from (gh-ocannl-1967): a narrower draw would quantize the initial
    values. A wider one is left alone. *)

val param_postprocess : (t -> t) ref
(** Post-processing hook applied by {!param} to each fully-constructed parameter before it is
    returned to user code; defaults to the identity. Mixed-precision recipes install a wrapper here
    that returns a reduced-precision "cast twin" consuming the parameter (gh-ocannl-492 master
    weights): the graph then reads the twin while the optimizer keeps updating the master parameter,
    which remains the sole member of the result's {!field:params}. Reset by {!unsafe_reinitialize}.
*)

val term_init : ?grad_spec:grad_spec -> float array -> op_fun
(** A {!term} wrapper that sets up the value node initialization (it generalizes {!ndarray} to
    tensors with inferred shapes). *)

val consume_forward_code : t -> comp
(** A forward root is a tensor that is not (currently) used to compute another tensor. On the first
    handout, checks that non-embedded reads are not owned by other unconsumed roots -- including a
    root the caller dropped before consuming it, whose code can then never run, so its readers must
    be rebuilt (gh-ocannl-1796) -- then removes [t] from the forward roots and returns its comp.
    Subsequent calls return the same comp, including after {!take_forward_code} (the [%cd] embedding
    path), without checking or modifying roots constructed since the first handout. This is
    equivalent to retaining the original comp; the caller still owns execution order and the
    availability of non-embedded inputs. Parameters and subtensors embedded in a tensor consumer
    remain rejected.

    Random seeds are the exception to "the caller owns the availability" (gh-ocannl-2157): the comp
    handed out carries the [Constant_bits] fetch of every seed literal ({!set_random_seed}) that
    [t]'s code reads and does not write -- a seed whose code another graph absorbed: a sibling
    compiled into another routine, an unused draw, under the current, a scoped or a replaced
    singleton -- and embeds it. So a routine reading a seed writes it, in whichever context it is
    compiled, and a seed is never a conflicting read. A seed literal is immutable and draws derive
    their counters from node ids and offsets, so the replayed fetch never changes a draw. A raw
    [t.forward] passed to [Context.compile] bypasses the handout and gets no fetch.

    The conflicting-roots check carries correctness, not just an early diagnosis (gh-ocannl-1741):
    no lineage check backs it, since a computation's [requires] names backprop fragments only, and
    the run-time initialization check accepts a read the owner never computed when the node is
    zero-valid or was written by other means ([set_values], an earlier routine). *)

val consume_backprop_code : t -> comp
(** A backprop root is a differentiable tensor not currently receiving gradients from another
    tensor. Checks non-embedded reads against other live backprop roots on the first handout and
    removes [t] from the backprop roots. Subsequent calls return the same comp without checking or
    modifying newer roots, as for {!consume_forward_code}; compiling the replayed code a second time
    in one compilation lineage is refused by [Context.compile] (gh-ocannl-1724). Non-differentiable
    tensors and embedded subtensors remain rejected. The returned code does not include [zero_grads]
    or seeding the output gradient; callers must still arrange those for each run.

    The conflicting-roots check is an early diagnostic (gh-ocannl-1741): a gradient write into
    another root's propagation is a backprop fragment the returned comp [requires], which
    [Context.run] refuses unless the routine's own lineage owns it (gh-ocannl-1724). It stays
    because it fires at the handout, naming the root to consume first, and refuses the order
    outright where the lineage check would wait for the run. *)

val iter_embedded : f:(tn -> unit) -> t -> unit
(** [iter_embedded t] iterates over all descendant nodes that are embedded, i.e. are members of
    [t.forward.embedded_nodes] or '[t.diff.backprop.embedded_nodes]' (if any). Note: [iter_embedded]
    should only be called after shape inference finishes. *)

val bump_next_id : int -> unit
(** [bump_next_id id] ensures the next allocated tensor ID will be strictly greater than [id].
    Called after loading tensors from a checkpoint file to prevent ID collisions with subsequently
    created tensors. *)

val get_next_id : unit -> int
(** [get_next_id ()] returns the next tensor ID that will be allocated by the session. *)

val rewind_to_last_closure : ?reason:string -> unit -> unit
(** Restores a usable session after a failed construction, without discarding closed or compiled
    work (gh-ocannl-1301). A construction that fails after mutating the inference session taints it:
    from then on, constructions, [Shape.finish_inference] (so forcing dims or projections), code
    handout and every [Context] compile raise, naming the first failure. Rewinding discards
    everything opened since the last successful [Shape.finish_inference] -- the failed construction
    and every construction since that closure, successful or not -- and clears the taint:

    - the solver's open state is emptied (closed shapes carry no variables, and their committed
      rows, paddings and neutral elements cannot have changed: a write that would change them is
      refused);
    - the forward and backprop root maps are restored to their state at the closure, with the
      handout markers of the restored roots reset; code handed out to callers since then stays
      handed out;
    - the tensors allocated since the closure are discarded (exactly the nodes the frontend
      allocated, by uid: a node created in that window outside the frontend is not): using one as an
      operand, handing out or compiling its code, printing it or transferring its node raises
      [Session_error] naming the rewind's provenance ("discarded by the rewind after ..."), as do
      their update steps' projections;
    - variable references the discarded work bound or set are refused by [Shape.set_dim] and the
      other reference setters, and by any later einsum they are passed to;
    - the cached random seed is restored to the closure's.

    Counters are never reset, so later tensors never collide with discarded ones. Contexts, routines
    and buffers are untouched, and nothing executed is undone. Without a taint it simply discards
    the open work ([reason] names it in the stale-handle errors). Raises when called from within an
    [op_asn]. The explicit checkpoints taken since the closure can no longer be rewound to. *)

(** {2 Explicit checkpoints (gh-ocannl-903)}

    A finer rewind than {!rewind_to_last_closure}: back to a point inside the open work -- a REPL
    phrase, a notebook cell, a tentative trace such as a shape scheme's. Not to be confused with
    [Ocannl.Checkpoint], which saves trained state to files. *)

type checkpoint
(** A snapshot of the open inference session. *)

val checkpoint : ?label:string -> unit -> checkpoint
(** Snapshots the open frontier: the solver's and the graph's session state (persistent values,
    recorded as pointers), and every mutable field of the open work's shapes, update steps and
    tensor nodes and of every live capture reference ({!Shape.get_variable_ref}). The cost is linear
    in that frontier, so take one per phrase or trace, not per operation. [label] names it in
    messages (default ["checkpoint #n"]). Raises when the session is tainted (rewind the failure
    first) and when called from within an [op_asn]. *)

val rewind : ?reason:string -> checkpoint -> unit
(** Puts the session back as it was at the checkpoint, and clears the taint:

    - every object open at the checkpoint gets its fields back -- the shapes' rows, paddings and
      neutral elements (rewritten in place by later operations), the capture references' bindings
      and solutions (a reference made before the checkpoint and solved after it is unsolved again),
      the open nodes' precision links, memory-mode intents, ownership and observability (a
      [Tensor.param] over an open initializer included);
    - the solver's environment, constraints, active steps, inference sets and rank graph, the
      graph-root maps (with the handout markers of the restored roots reset) and the random seed are
      the checkpoint's; code handed out to callers since stays handed out;
    - what the window created is discarded, as by {!rewind_to_last_closure}: its tensors (exactly
      the nodes the frontend allocated), update steps and the capture references it made and bound
      raise [Session_error] / [Shape_error] naming the rewind's provenance when used.

    Counters are never reset. Closed and compiled work is untouched, and nothing executed is undone.
    The checkpoint stays valid, so it can be rewound to again; the checkpoints taken after it can
    not. Raises [Session_error], before anything is written, when the checkpoint cannot be rewound
    to: inference closed since it was taken (a closure commits); a rewind past it,
    {!rewind_to_last_closure} or {!unsafe_reinitialize} ran since; or the window memoized state of
    the frontier that no rewind can reset -- a dims, padding or projections read of an open shape
    that raised (its lazy keeps the exception), or an open node's precision that began settling.
    {!rewind_to_last_closure} is always available. A [Tensor.take_forward_code] called directly on a
    tensor that was not a root at the checkpoint leaves its marker taken. Raises when called from
    within an [op_asn]. *)

val with_checkpoint : ?label:string -> (unit -> 'a) -> 'a
(** [with_checkpoint f] runs [f]; when [f] raises, rewinds to a checkpoint taken at entry and
    re-raises [f]'s exception with its backtrace. When that checkpoint can no longer be rewound to
    (say [f] closed inference), [f]'s exception still wins, and the session keeps any taint [f]
    left. A normal return keeps [f]'s work. *)

val with_rewind : ?label:string -> (unit -> 'a) -> 'a
(** [with_rewind f] runs [f] and always rewinds to a checkpoint taken at entry -- a trace that
    leaves nothing in the session -- returning [f]'s result. Read what you need of the open state
    inside [f]; tensors [f] creates are discarded handles afterwards, so create the ones meant to
    outlive the trace (shared parameters) before calling it. When [f] returns but its checkpoint can
    no longer be rewound to, the rewind's [Session_error] is raised; when [f] raises, [f]'s
    exception wins, as in {!with_checkpoint}. *)

(** {2 Shape schemes (gh-ocannl-404)}

    The low-level, explicit layer of shape schemes, which the [%op] integration (gh-ocannl-1751)
    wraps. A scheme is the shape constraints a tensor function imposes, inferred once by a trace and
    instantiated at each application:

    - a shape error inside the function that does not depend on its arguments' shapes is raised by
      {!Scheme.trace}, at the definition;
    - {!Scheme.instantiate} adds the scheme's constraints, renamed, relating it to given argument
      and result shapes, without running the function: the shapes close as an inline application of
      the function would have closed them.

    What is shared and what is per-application follows ML's value restriction: the dimensions of a
    tensor created before the trace -- a shared parameter, lifted to the function's unit-parameter
    scope -- stay fixed across applications, and everything the function creates (its intermediates,
    a parameter created inside it, a capture reference made inside it) is fresh per application. So
    create the shared parameters before {!Scheme.trace}. The design note is the gh-ocannl-404
    issue's reconciled-plan comment; [docs/shape_inference.md] has the summary. *)
module Scheme : sig
  type tensor := t
  type t

  val label : t -> string
  val arity : t -> int
  val num_results : t -> int

  val to_string : t -> string
  (** [to_string s] prints [s]'s boundary: [<label> : <arguments> => <results>], each shape as
      [batch|input->output] under the trace's solution, with [..v..] for a row variable and [()] for
      an empty row, and [_] for a scheme with no results. Variables are named by first appearance,
      so the text does not depend on the session's id counters: a generalized variable (renamed per
      application) as ['a], ['b], ...; a fixed one (shared with what the function closes over, such
      as a parameter lifted to its [()] scope) as ['_a], ['_b], ..., the way OCaml prints weak type
      variables. Then the indented lines: the broadcasting inequalities [r ⊑ o1, o2, ...] (the
      result side refines each operand side) among the boundary's variables and the fixed ones,
      grouped by left side; then [_ ⊑ o1, o2, ...] for the variables some eliminated intermediate
      refines together when no printed variable does (gh-ocannl-1988: arguments consumed together by
      a reduction, whose common refinement the result no longer carries); then the lower bounds of
      the boundary's variables, [b ⊑ v], the variables such a bound names related like the
      boundary's. All under the trace's final environment, with the function's intermediate
      variables eliminated along monotone chains. The order is canonical: the boundary's variables
      by their appearance in the header, a fixed one by its relations to them -- never the solver's
      traversal order. Other constraints (totals, concatenations) are not printed.
      [test/operations/reusable_tensor_programs.ml] shows one. *)

  val trace : label:string -> int -> (tensor list -> tensor list) -> t
  (** [trace ~label n f] applies [f] to [n] tracing tensors whose shapes are fresh row variables,
      inside {!with_rewind}, and returns the scheme of the constraints it registered over them and
      over its results (a flat list: flatten a structure). The trace leaves nothing in the session.
      Raises [Row.Shape_error] prefixed ["tracing the shape scheme <label>: "] on a shape error in
      [f] (including one only the end-of-trace check of [f]'s trailing [Shape.set_dim] and the like
      finds), when [f] forces dims or projections (inference cannot close during a trace), and for
      what a scheme cannot carry: [Shape.set_dim] and the like on a capture reference made outside
      [f]. Random number generation inside [f] is carried (gh-ocannl-1890): see [Shape.Scheme.trace]
      for how its coefficient reads each call site's precision. A capture reference made outside [f]
      that [f]'s operations bind is an input of the scheme (gh-ocannl-1791): each application
      relates it to the reference's state then, and {!instantiate} binds it, as the body would. *)

  val instantiate :
    ?application:string ->
    ?here:Ppx_here_lib.position ->
    t ->
    args:Shape.t list ->
    results:Shape.t list ->
    unit
  (** [instantiate s ~args ~results] adds [s]'s constraints for one application of its function to
      [args] with [results] (fresh, inferred shapes, or the shapes of tensors computing the
      results), without running the function. An argument-dependent shape error raises
      [Row.Shape_error] prefixed ["applying the shape scheme <label> to <application>: "] (with
      [~here:[%here]], ["... to <application> at <file>:<line>: "]: gh-ocannl-2007) and taints the
      session, as a failed construction does; the constraints carry the application's name (and
      [here]), so a failure at the closure names it too. The next closure applies the solution to
      [args] and [results] and closes them. It finalizes the scheme's rows but derives no
      projections for the body's steps, so projection-time refusals do not run: an [Affine] or
      [Concat] over a [Sym] that cannot be sized, window verification, and shared symbolic
      iterators. An instance can therefore close where the body's inline application is refused
      (gh-ocannl-2020). Automatic [%op] schemes are unaffected: their body always runs. Valid across
      closures, unless the scheme depends on open state no shape it used holds (then refused after a
      closure: trace again). A scheme whose function draws random numbers is refused: the size of a
      draw's counter is the precision its conversion node settles, and an instance, unlike the
      function's own application, has none. *)

  (** {3 Automatic schemes of [%op] functions (gh-ocannl-1751)}

      [%op] gives a function a scheme when every parameter of the function it returns is a tensor
      parameter ([docs/syntax_extensions.md] says which qualify). The closure built at the [()]
      application -- or, for a function without a [()] stage, its first application -- traces the
      scheme. How each application uses it is the [op_shape_schemes] setting:
      - [check] (the default): each application checks the scheme against its arguments' shapes, in
        a window that is always rewound, before the body runs and registers as it always does. The
        applications in the body of a checked application are not checked themselves: its scheme
        holds their constraints, so an argument error in one is reported by the outer check, naming
        the outer function (gh-ocannl-1814). Only a function's first application there still traces
        its presence patterns, which the outer scheme does not hold. Under an opted-out, refused or
        otherwise unchecked function they are checked as usual;
      - [off]: no schemes.

      A shape error that does not depend on the arguments raises at the trace, prefixed
      ["tracing the shape scheme <function>: "] -- or, inside an application written in [%op],
      ["tracing the shape scheme <function> in the %op application at <file>:<line>: "]
      (gh-ocannl-2164). One that does raises at the application, before the body runs, prefixed
      ["applying the shape scheme <function> to <arguments>: "] -- or, for an application evaluated
      inside an application written in [%op] code, the innermost such,
      ["applying the shape scheme <function> to <arguments> in the %op application at <file>:<line>:
       "] (gh-ocannl-2007). The expansion records that site; an application written in plain OCaml
      outside any [%op] application has none (OCaml has no caller location), so its message names
      the function and the arguments only. A trace a scheme cannot carry (setting a capture
      reference made outside the function, forcing dims, or a body that cannot run on tracers)
      leaves the function unschemed, as it was before, once its rewind succeeded, and says so on
      stderr -- once per function and reason in a process (gh-ocannl-1791). A capture reference made
      outside the function that its operations bind is a scheme input, checked against each
      application's state of it. Ids are unaffected: traces and checks allocate tensor ids, node
      uids and shape-variable ids from reserved ranges.

      Option-valued tensor parameters (gh-ocannl-1792), annotated [~(p : Tensor.t option)] or
      [?(p : Tensor.t option)]: the function gets one scheme per presence pattern -- which of them
      are [Some] -- each traced, checked and refused on its own, under the label
      [<function> [p, no q]] (just [<function>] without such parameters). The closure built at [()]
      traces every pattern; a function without a [()] stage traces every pattern at its first
      application; afterwards each pattern re-traces on its own when stale.

      An automatic scheme's results (gh-ocannl-1801) are what the traced body returns, when it is a
      tensor the body made (or a copy of its record, [{ t with forward = ... }]) or one of its
      arguments, or a tuple (or record) of such: one result per component. Any other value -- a
      function, a list, a tensor made before the body ran -- gives the scheme no results. The checks
      relate the arguments only, so the results serve {!to_string} and {!instantiate}. {!traced_by}
      returns the schemes a closure traced: the ones its applications are checked against, not a
      re-trace. *)

  type handle
  (** A function's scheme slots: one handle per closure, one slot per presence pattern. *)

  val status : handle -> string
  (** ["pending"], ["schemed"], or ["unschemed: <reason>"]; with option-valued parameters, one
      ["<label>: <status>"] per pattern, separated by ["; "]. *)

  type event =
    | Traced of t
        (** A function's scheme (a pattern's, under its label) was traced: the scheme its
            applications are checked against. *)
    | Unschemed of string * string
        (** A function (a pattern, by its label) was left unschemed, with the reason. *)
    | Checked of string  (** An application was checked against its pattern's scheme. *)

  val registration_counts : unit -> Shape.registration_counts
  (** Process-wide registration statistics since the last reset (gh-ocannl-1752's attribution): see
      {!Shape.registration_counts}. *)

  val reset_registration_counts : unit -> unit

  val registration_times : unit -> Shape.registration_times
  (** Process-wide shape-registration wall time since the last reset (gh-ocannl-1752): see
      {!Shape.registration_times}. *)

  val reset_registration_times : unit -> unit

  val traced_by : (unit -> 'a) -> 'a * t list
  (** [traced_by (fun () -> block conf ())] runs the function and returns its result with the
      automatic schemes traced while it ran, innermost first: those of the closures it built at
      their [()] stage (every presence pattern of each), of the functions without a [()] stage it
      first applied, and of the stale ones it re-traced. They are the schemes the closures use --
      their applications are checked against them -- so [to_string] prints the scheme in use, with
      no re-trace (gh-ocannl-1801). The scheme's {!label} names its function (and pattern). *)

  val check_definition : name:string -> (unit -> unit) -> t list
  (** [check_definition ~name (fun () -> ignore (block conf ()))] runs a trial application of a
      [%op] block definition inside a window that is always rewound and allocates no session ids, so
      the [()] traces of the block and of the sub-blocks it builds run, and with them any shape
      error that does not depend on the arguments, prefixed ["checking the definition <name>: "].
      Returns the schemes traced, as {!traced_by} does (every presence pattern of a closure built at
      [()] in the trial: see above), innermost first; the session is unchanged afterwards, and the
      schemes stay printable. A function without a [()] stage that the trial only builds is not
      traced. It catches errors that hold for every configuration and those of the trial
      configuration only; it costs the trial's construction and traces. Prefer it to a bare
      [block conf ()], which leaves the trial's parameters in the session. *)

  (** The runtime the [%op] expansion calls. Not for direct use. *)
  module For_ppx : sig
    val closure : name:string -> arity:int -> (tensor list -> 'r) -> handle
    (** Traces now, the body applied to tracers: the [()] stage of a [%op] function. Its value gives
        the scheme's results (see above), then is dropped. *)

    val deferred : name:string -> arity:int -> (tensor list -> 'r) -> handle
    (** Traces at the first application: a [%op] function without a [()] stage. *)

    val at_site : string * int * int * int -> (unit -> 'a) -> 'a
    (** [at_site __POS__ (fun () -> f args)]: an application written in [%op] code, run with its
        source position recorded for the checks inside it to name (gh-ocannl-2007); the outer one is
        restored afterwards. *)

    val apply : handle -> tensor list -> (unit -> 'a) -> 'a
    (** [apply h args body] (tracing [h]'s scheme first when pending or stale) checks the scheme
        against [args] -- naming the innermost {!at_site} in progress, if any -- then runs [body].
        Inside a trace, and when unschemed, only runs [body]. So it does inside the [body] of an
        application that was checked, whose scheme holds this application's constraints
        (gh-ocannl-1814), but for tracing the presence patterns of a function on its first
        application. *)

    val closure_variants :
      name:string -> optional:string option list -> (tensor option list -> 'r) -> handle
    (** {!closure} for a function with option-valued tensor parameters: [optional] names each
        position's option-valued parameter ([None]: a required tensor, passed as [Some]). Traces
        every presence pattern now. *)

    val deferred_variants :
      name:string -> optional:string option list -> (tensor option list -> 'r) -> handle
    (** {!deferred} likewise: every presence pattern is traced at the first application. *)

    val apply_variants : handle -> tensor option list -> (unit -> 'a) -> 'a
    (** {!apply} for the variant handles: the arguments' presence selects the scheme checked against
        the present arguments. *)
  end
end

(** Diagnostics and negative controls for the checkpoint machinery's tests. *)
module For_testing : sig
  val restore_categories : string list

  val trace_scheme :
    sabotage:Shape.Scheme.sabotage -> label:string -> int -> (t list -> t list) -> Scheme.t
  (** {!Scheme.trace} with one part of the mechanism disabled: negative controls. *)

  val instantiate_scheme :
    sabotage:Shape.Scheme.sabotage ->
    ?application:string ->
    ?here:Ppx_here_lib.position ->
    Scheme.t ->
    args:Shape.t list ->
    results:Shape.t list ->
    unit
  (** {!Scheme.instantiate} with one part of the mechanism disabled: negative controls. *)

  val observe_schemes : (unit -> 'a) -> on_event:(Scheme.event -> unit) -> 'a
  (** Runs the function, reporting the automatic schemes' events to [on_event]. *)

  val set_op_shape_schemes : [ `Check | `Off ] option -> unit
  (** Overrides the [op_shape_schemes] setting in-process ([None]: back to the setting). *)

  val discarded_registry_sizes : unit -> (string * int) list
  (** The records the session holds of what rewinds discarded, by kind (nodes, capture references,
      update steps). Call it after a full major collection to count only what is reachable. *)

  val with_nested_rechecks : (unit -> 'a) -> 'a
  (** Runs the function with every schemed application checked, those a checked application's body
      makes included: a negative control of gh-ocannl-1814, and the reference for what skipping them
      saves. *)

  val without_id_neutrality : (unit -> 'a) -> 'a
  (** Runs the function with the shape-scheme windows allocating session ids: a negative control of
      their id neutrality. *)

  val rewind_omitting : omit:string -> checkpoint -> unit
  (** {!rewind} skipping one restore category: to show a test's oracle notices it. *)

  val checkpoint_differences : checkpoint -> string list
  (** The components of the session that differ from the checkpoint (physically; padding arrays by
      contents), by name. *)

  val checkpoint_sizes : checkpoint -> (string * int) list
  (** The open frontier's sizes at the checkpoint. *)

  val forward_conflicts_by_cells : t -> string list
  (** The roots {!consume_forward_code} would refuse [t] over, by name, as its owner cells say
      (gh-ocannl-1796). *)

  val forward_conflicts_by_live_roots : t -> string list
  (** The same, by the pre-gh-ocannl-1796 rule over the live forward roots: equal to
      {!forward_conflicts_by_cells} while every root is held. *)

  val owner_count : t -> int
  (** The number of owners [t]'s forward owner cell records, as last compacted (the probe compacts
      nothing itself). *)

  val drop_owner_edges : t -> unit
  (** Forgets every owner of [t]'s forward owner cell: a negative control of the
      [debug_check_forward_root_owners] check, standing for a source of [embedded_nodes] that added
      no owner edge. *)
end

val update_operand_prec : t -> Ir.Ops.prec -> unit
(** Sets an operand's precision ahead of an operation that requires it; refused (raises
    [Session_error]) when it would change a closed operand's precision. *)

val update_operands_prec : t list -> Ir.Ops.prec -> unit
(** {!update_operand_prec} over several operands, all validated before any is written. *)

val check_not_discarded : what:string -> t -> unit
(** Raises [Session_error] when the tensor's value or gradient node was discarded by a rewind. *)

val unsafe_reinitialize : ?namespace:string -> unit -> unit
(** Bring global state to its initialization values. This invalidates any previously defined tensors
    and tensor nodes. Also reinitializes the modules: {!Shape}, {!Ir.Tnode}.

    [namespace] sets the namespace stamped on subsequently created tensor nodes (gh-ocannl-372);
    this is the only way to change it. Defaults to {!Ir.Tnode.default_namespace} ([ocannl]) — also
    when omitted, i.e. reinitializing never preserves a previously set namespace. Namespaces must
    match [A-Za-z_][A-Za-z0-9_]* so they can qualify identifiers in generated code and debug names
    (rendered as [ns__n42]; the default namespace is elided).

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

val random_seeds : unit -> int list
(** The distinct seeds of the random-seed singletons this session created ({!set_random_seed},
    directly or lazily through {!get_random_seed}), in the order they first appeared, without one a
    rewind discarded: every seed the session's graphs can draw from, including those built under
    {!with_saved_random_seed}. One value per seed, whether its graphs are held or not, so a session
    re-seeding with the same values grows nothing, and one choosing a new value per graph keeps one
    integer for each. *)

val with_saved_random_seed : (unit -> 'a) -> 'a
(** Runs [f], restoring the global random-seed singleton (the one {!get_random_seed} returns) to its
    prior value afterwards, even on exception. Use this to scope {!set_random_seed} mutations — e.g.
    building several graphs with distinct per-graph seeds — without perturbing the caller's seed.
    Tensors built inside [f] keep referencing the seed tensors that were current when they were
    constructed; only the {e global} pointer is restored.

    Initialization follows the captured seed (gh-ocannl-2119): {!init_params}, after [f] returned or
    raised, draws a parameter built inside [f] from the seed it captured, exactly as a program
    owning that seed would, even when the seed's own code was absorbed by another graph (a discarded
    draw) and another singleton has been restored since. The same holds after a plain
    {!set_random_seed} replaced the singleton. A seed literal is immutable, so replaying it replays
    no random stream: draws derive their counters from node ids and offsets. A {e forward} graph
    reading such a seed is handed its fetch the same way ({!consume_forward_code}, gh-ocannl-2157).
    A seed a rewind discarded is refused as discarded. *)

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
[@@deriving sexp_of]
(** We print out up to 5 axes when printing a tensor, as a grid (outer rectangle) of (inner)
    rectangles, possibly repeated (screens). *)

val to_printbox :
  ?single_node:bool ->
  ?embedded_only:bool ->
  ?entries_per_axis:int ->
  ?with_id:bool ->
  ?force:bool ->
  ?ctx:Context.t ->
  ?with_shape:bool ->
  ?with_value:bool ->
  with_grad:bool ->
  depth:int ->
  t ->
  PrintBox.t
(** [to_printbox] is the box {!print_tree} prints, laid out by [depth] the same way, with [with_id]
    and [with_value] defaulting to [false] and [true]. *)

val to_doc :
  ?ctx:Context.t ->
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
  ?ctx:Context.t ->
  with_grad:bool ->
  with_code:bool ->
  ?with_low_level:bool ->
  array_print_style ->
  t ->
  unit
(** [print] renders a tensor's metadata and, when an explicit [?ctx] is supplied, its values via an
    on-demand device-to-host transfer (gh-ocannl-333). Without [?ctx] (the default) values are shown
    as a placeholder. *)

val print_tree :
  ?here:Ppx_here_lib.position ->
  ?force:bool ->
  ?ctx:Context.t ->
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
(** [print_tree] renders the tensor's graph: its outermost [depth] levels side by side in boxes, the
    levels below them as an indented tree. [depth] shapes the layout only, so every node of the
    graph is shown. With [?ctx], the values of all the nodes it shows are read together through
    {!Context.to_host_all}, so the virtual ones are recomputed by one throwaway routine
    (gh-ocannl-1737); an unreadable node is shown as a placeholder naming why
    ({!Context.observability_tag}). [print] reads the value and gradient it shows the same way. *)

val debug_name : t -> string

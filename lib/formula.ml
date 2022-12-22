(** The compositional primitives for runtime-compiled code supporting backpropagation. *)

open Base

module AxisKey = struct
  module T = struct
    type shape_kind = 
      | Batch
      | Input
      | Output
    [@@deriving compare, sexp]
    type t = {
      in_axes: shape_kind;
      from_end: int
      (** Axes are indexed from the end, to avoid reindexing when broadcasting; starting with [1]. *)
     } [@@deriving compare, sexp]
     let to_string key = 
      (match key.in_axes with Batch -> "bch" | Input -> "inp" | Output -> "out") ^
      Int.to_string key.from_end
  end
  include T
  include Comparator.Make(T)
end

type axis_labels = string Map.M(AxisKey).t [@@deriving compare, sexp]

type dims =
| Given of int list
| Inferred of int list
| Unknown [@@deriving compare, sexp, variants]

type deduce_dims =
[ `Not_deduced
| `Preserve
| `Scale of float
] [@@deriving compare, sexp, variants]

(** Converts dimensions according to the specification. Note that scalar axes (1D) are not scaled,
    for compatibility with broadcasting. *)
let deduce_dims from: deduce_dims -> dims = function
| `Not_deduced -> Unknown
| `Preserve ->
  (match from with
  | Given dims -> Inferred dims
  | from -> from)
| `Scale sc ->
  match from with
  | Unknown -> Unknown
  | (Inferred dims | Given dims) -> Inferred (List.map dims ~f:(
      fun d -> if d = 1 then 1 else Float.(iround_exn ~dir:`Up @@ sc * of_int d)))

(** The datatype from which the actual Ndarray shapes are computed. In the future we can have
    named axes here instead of the predefined options.

    Mutability is sufficient to perform inference, since there is no need for backtracking and
    no explicit unification variables for now. [Unknown] stands for "not yet specified". *)
type shape = {
  mutable batch_shape: dims;
  mutable input_shape: dims;
  mutable output_shape: dims;
  mutable axis_labels: axis_labels;
  shape_of_node_id: int;
  deduce_output_from_input: deduce_dims;
  (** Intended for terminal node cases where both [input_shape] and [output_shape] are initially
      unknown. It makes it trivial to implement dimension-preserving hidden layers: just set
      [deduce_output_from_input=`Preserve]. *)
} [@@deriving fields, sexp]

let dims_of_kind =
  let open AxisKey in function
  | Batch -> batch_shape
  | Input -> input_shape
  | Output -> output_shape

type compose_type =
  [ `Pointwise
  (** NumPy-style broadcast matching batch, input and output axes, e.g. as in [s1 + s2]. *)
  | `Compose
  (** Compose the outputs of the second shape with the inputs of the first shape, i.e. the shape of
      [fun x -> s1(s2(x))], or [s1 * s2] where [*] is the inner product (e.g. matrix multiply). *)
  | `Einsum of axis_labels * axis_labels * axis_labels
  (** A version of the [einsum] syntax. Note that currently [`Pointwise] and [`Compose] are
      not redundant with [`Einsum], because they enable more shape inference: they do not specify
      the number of axes. The [axis_labels] use pseudo-labels local to the notation, to line up the axes.
      For [`Einsum (ls1, ls1, ls2)], the symmetric difference / disjunctive union of [ls1] and [ls2]'s
      pseudo-labels should be equal to [ls3] pseudo-labels.
      
      Currently, we support two variants of the [einsum] syntax: either all the axes are provided,
      or all input, output axes are provided but none of the batch axes. *)
  ]

type transpose_type =
  [ `Transpose
  (** Swaps inputs and outputs of a shape, preserves batch axes. *)
  | `Permute of axis_labels * axis_labels
  (** [`Permute (ls1, ls2)] is equivalent to [`Einsum (ls1, ls1, ls2)] (also to 
      [`Einsum (ls1, axis_labels.empty, ls2)] etc.). *)
  ]

(** How to propagate shape updates and do the last update of [t.shape] when finalizing the formula.
    Axes are broadcast-expanded on a bottom-up update to fit the incoming shape. *)
type shape_logic = 
  | Broadcast of compose_type * shape * shape
  (** Matches the shapes for a binary operation, allowing for broadcasting e.g. an axis of dimension 1
      does not conflict with a matching axis of a greater dimension.

     For [Broadcast (`Einsum (ls1, ls2, ls3), s1, s2)], the labels of [s1] and [s2] must match according
     to the [ls1], [ls2] lineup, and the resulting shape inherits the labels according to the [ls3] lineup.
  *)
  | Transpose_shape of transpose_type * shape
  (** Permutes the axes of a shape. The simplest [Transpose_shape] is to swap inputs with outputs of [s1],
      hence the name. *)
  | Terminal_shape

(** Data required for a shape inference update step. A step should equilibrate information, passing it both
    top-down and bottom-up. The child should be identifiable within the parent via physical equality
    (allowing that a child fills both slots of a binary parent). *)
type shape_update_step = {
  shape: shape;
  shape_logic: shape_logic;
}

exception Shape_error of string * shape * shape [@@deriving sexp]

(** Uses [code option], i.e. [None] instead of [.< () >.], to improve readability of generated code. *)
type t = {
  toplevel_forward: (unit -> unit) Codelib.code;
  (** Only apply at the root, since otherwise some computation may be elided (incorrect results). *)
  toplevel_backprop: (unit -> unit) Codelib.code;
  (** Only apply at the root! Gradients propagate from the top and are only propagated once. Zeroes
      the gradients before propagating. *)
  forward_body: unit Codelib.code option;
  init_values: unit Codelib.code;
  (** Initializes the values. Computed only once per model compilation. *)
  init_grads: unit Codelib.code;
  (** Initializes the gradient data: typically, simply creates the ndarrays.
      Gradients are zeroed separately. *)
  backprop_body: unit Codelib.code option;
  zero_grads: unit Codelib.code;
  (** Initializes the backpropagation phase. Computed once per backpropagation. *)
  node_id: int;
  comp_node: Node.t;
  (** This tracks the computation node as long as the model is not cross-compiled to a different
      process etc. *)
  node: Node.t Codelib.code;
  (** The node storing the computation results. [.!(t.node)] should equal [t.comp_node]. *)
  mutable processed: bool;
  (** [true] if [forward_body]/[backprop_body]/[zero_grads] were already included in a parent [t]. *)
  shape_logic: shape_logic;
  (** How to do the last update of [t.shape] when finalizing the formula. *)
  shape: shape;
  (** The eventual shape of [.!(t.node).value] and [.!(t.node).grad], incorporating the current state of
      shape inference. *)
  subtree_shape_updates: shape_update_step Sequence.t;
  (** We piggy-back on the code generation setup to arrange the updates. We perform each update twice
      to propagate information between all subformulas: first in postfix order while computing [t],
      second in prefix order by iterating over [t.subtree_shape_updates]. *)
}

(* The code relies on argument evaluation order. To lift the requirement, we could use
   [t Lazy.t], but that's an unnecessary obfuscation. *)
let l2r_comp_order =
  let l2r_ord = ref None in
  (fun () () ->
    match !l2r_ord with
    | Some b -> b
    | None -> assert false) (l2r_ord := Some false) (l2r_ord := Some true)

(* Design choice: tensor shapes are decided while code is constructed, although not immediately.
   Due to mutable updates during shape inference, it is not possible to reuse the same formula with
   different shapes. *)

let propagate_shapes (update: shape_update_step) =
  let pointwise_labels debug1 debug2 ls1 ls2 = Map.merge ls1 ls2 ~f:(fun ~key ->
    function
    | `Both (l1, l2) ->
      if String.equal l1 l2 then Some l1
      else
        let error = "Axis label mismatch: "^l1^" vs "^l2^" for "^
                    (Sexp.to_string_hum @@ AxisKey.sexp_of_t key) in
         raise @@ Shape_error (error, debug1, debug2)
    | `Right l | `Left l -> Some l
  ) in
  let broad_dim ?(to_broader=false) ?(to_narrower=false) debug1 debug2 axis_key label = function
    | d1, d2 when d1 = d2 -> d1
    | 1, d when not to_broader -> d
    | d, 1 when not to_narrower -> d
    | d1, d2 ->
      let opt_label = match label with None -> "" | Some l -> " ("^l^")" in
      let error = "Dimension mismatch for axis "^AxisKey.to_string axis_key^opt_label^": "^
                  Int.to_string d1^" vs. "^Int.to_string d2 in
      raise @@ Shape_error (error, debug1, debug2) in
  let broadcast_dims ?(to_broader=false) ?(to_narrower=false) debug1 debug2 kind labels =
    let rec broad_back_dims accu i = function
    | [], [] -> accu
    | [], dims when not to_broader -> List.rev_append dims accu
    | dims, [] when not to_narrower -> List.rev_append dims accu
    | [], _ | _, [] ->
      let key = AxisKey.{in_axes=kind; from_end=i} in
      let opt_label = match Map.find labels key with None -> "" | Some l -> " ("^l^")" in
      let error = "Different number of axes around from-end "^AxisKey.to_string key^opt_label in
      raise @@ Shape_error (error, debug1, debug2)
    | d1::dims1, d2::dims2 ->
      let key = AxisKey.{in_axes=kind; from_end=i} in
      broad_back_dims 
        (broad_dim ~to_broader ~to_narrower debug1 debug2 key (Map.find labels key) (d1, d2)::accu)
        (i+1) (dims1, dims2) in
    function
    | Unknown, Unknown -> Unknown
    | (Inferred dims | Given dims), Unknown | Unknown, (Inferred dims | Given dims) -> Inferred dims
    | (Inferred dims1 | Given dims1), (Inferred dims2 | Given dims2) ->
      Inferred (broad_back_dims [] 1 (List.rev dims1, List.rev dims2)) in
  let cur_sh = update.shape in
  let update_labels sh1 to_kind sh2 from_kind =
    pointwise_labels sh1 sh2 sh1.axis_labels @@
    Map.map_keys_exn (module AxisKey) ~f:(fun k -> {k with in_axes=to_kind}) @@
    Map.filter_keys sh2.axis_labels ~f:(fun k -> phys_equal k.in_axes from_kind) in
  let broadcast_into ?to_broader ?to_narrower to_sh to_kind from_sh from_kind =
    match dims_of_kind to_kind to_sh, dims_of_kind from_kind from_sh with
    | Given _ as into_dims, from_dims ->
      ignore @@
        broadcast_dims ?to_broader ?to_narrower to_sh from_sh to_kind to_sh.axis_labels
          (into_dims, from_dims);
      into_dims
    | into_dims, from_dims ->
      to_sh.axis_labels <- update_labels to_sh to_kind from_sh from_kind;
      broadcast_dims to_sh from_sh to_kind to_sh.axis_labels (into_dims, from_dims) in
  match update.shape_logic with
  | Terminal_shape -> ()
  | Transpose_shape (`Transpose, sh) ->
    cur_sh.input_shape <- broadcast_into cur_sh Input sh Output;
    cur_sh.output_shape <- broadcast_into cur_sh Output sh Input;
    cur_sh.batch_shape <- broadcast_into cur_sh Batch sh Batch;
    sh.input_shape <- broadcast_into sh Input cur_sh Output;
    sh.output_shape <- broadcast_into sh Output cur_sh Input;
    sh.batch_shape <- broadcast_into sh Batch cur_sh Batch;

  | Transpose_shape (`Permute einsum, sh) -> 
    ignore (einsum, sh); failwith "Not implemented yet"

  | Broadcast (`Pointwise, sh1, sh2) ->
    let up_labels = pointwise_labels sh1 sh2 sh1.axis_labels sh2.axis_labels in
    cur_sh.axis_labels <- up_labels;
    (if is_unknown cur_sh.input_shape then
      cur_sh.input_shape <- broadcast_dims sh1 sh2 AxisKey.Input up_labels
          (sh1.input_shape, sh2.input_shape)
    else (
      cur_sh.input_shape <- broadcast_into cur_sh Input sh1 Input;
      cur_sh.input_shape <- broadcast_into cur_sh Input sh2 Input;
    ));
    (if is_unknown cur_sh.output_shape then
      cur_sh.output_shape <- broadcast_dims sh1 sh2 AxisKey.Output up_labels
          (sh1.output_shape, sh2.output_shape)
    else (
      cur_sh.output_shape <- broadcast_into cur_sh Output sh1 Output;
      cur_sh.output_shape <- broadcast_into cur_sh Output sh2 Output;
    ));
    (if is_unknown cur_sh.batch_shape then
      cur_sh.batch_shape <- broadcast_dims sh1 sh2 AxisKey.Batch up_labels
          (sh1.batch_shape, sh2.batch_shape)
    else (
      cur_sh.batch_shape <- broadcast_into cur_sh Batch sh1 Batch;
      cur_sh.batch_shape <- broadcast_into cur_sh Batch sh2 Batch;
    ));
    
    sh1.input_shape <- broadcast_into sh1 Input cur_sh Input;
    sh1.output_shape <- broadcast_into sh1 Output cur_sh Output;
    sh1.batch_shape <- broadcast_into sh1 Batch cur_sh Batch;
    sh2.input_shape <- broadcast_into sh2 Input cur_sh Input;
    sh2.output_shape <- broadcast_into sh2 Output cur_sh Output;
    sh2.batch_shape <- broadcast_into sh2 Batch cur_sh Batch;

  | Broadcast (`Compose, sh1, sh2) ->
    (* [sh2] is the value or the function that gets applied first: [cur_sh(x) = sh1(sh2(x))].
       I.e. [cur.I = sh2.I, cur.O = sh1.O, sh2.O = sh1.I]. *)
    cur_sh.input_shape <- broadcast_into cur_sh AxisKey.Input sh2 AxisKey.Input;
    cur_sh.output_shape <- broadcast_into cur_sh AxisKey.Output sh1 AxisKey.Output;
    (if is_unknown cur_sh.batch_shape then
      let up_labels = update_labels cur_sh Batch sh1 Batch in
      cur_sh.axis_labels <- up_labels;
      let up_labels = update_labels cur_sh Batch sh2 Batch in
      cur_sh.axis_labels <- up_labels;
      cur_sh.batch_shape <- broadcast_dims sh1 sh2 AxisKey.Batch up_labels
           (sh1.batch_shape, sh2.batch_shape)
     else (
       cur_sh.batch_shape <- broadcast_into cur_sh Batch sh1 Batch;
       cur_sh.batch_shape <- broadcast_into cur_sh Batch sh2 Batch;
     ));
    
    sh1.input_shape <- broadcast_into sh1 Input sh2 Output;
    sh1.output_shape <- broadcast_into sh1 Output cur_sh Output;
    sh1.batch_shape <- broadcast_into sh1 Batch cur_sh Batch;
    sh2.input_shape <- broadcast_into sh2 Input cur_sh Input;
    sh2.output_shape <- broadcast_into sh2 Output sh1 Input;
    sh2.batch_shape <- broadcast_into sh2 Batch cur_sh Batch;

    (* Always re-derive the output shape, to have the latest information. *)
    if not @@ is_not_deduced sh1.deduce_output_from_input then
      sh1.output_shape <- deduce_dims sh2.input_shape sh1.deduce_output_from_input

  | Broadcast (`Einsum spec, sh1, sh2) ->
    ignore (spec, sh1, sh2); failwith "Not implemented yet"

let binop ~op_label ?(compose_op=`Pointwise) ~op_body ~grad_body m1 m2: t =
  let m1_l = m1.comp_node.label in
  let m1_l = if String.length m1_l > 11 then "n"^Int.to_string m1.node_id else m1_l in
  let m2_l = m2.comp_node.label in
  let m2_l = if String.length m2_l > 11 then "n"^Int.to_string m2.node_id else m2_l in
  let label = m1_l ^ op_label ^ m2_l in
  let comp_node = Node.create ~label in
  let node_id = comp_node.id in
  let axis_labels = Map.empty (module AxisKey) in
  let shape = { batch_shape=Unknown; input_shape=Unknown; output_shape=Unknown; axis_labels;
                shape_of_node_id=node_id; deduce_output_from_input=`Not_deduced } in
  let shape_logic = Broadcast (compose_op, m1.shape, m2.shape) in
  let local_shape_update = { shape; shape_logic } in
  propagate_shapes local_shape_update;
  let node = Codelib.genlet ~name:label (.< Node.get node_id >.) in
  let nv = (.< .~node.value >.) in
  let n1v = (.< .~(m1.node).value >.) in
  let n2v = (.< .~(m2.node).value >.) in
  let op_body = op_body ~nv ~n1v ~n2v in
  (* The code needs to be included in the order it was computed! *)
  let forward_body =
    match m1.processed, m1.forward_body, m2.processed, m2.forward_body with
    | true, _, true, _ | true, _, _, None | _, None, true, _ | _, None, _, None -> op_body
    | false, Some m1_body, false, Some m2_body when l2r_comp_order ->
      (.< .~m1_body; .~m2_body; .~op_body >.)
    | false, Some m1_body, false, Some m2_body ->
      (.< .~m2_body; .~m1_body; .~op_body >.) 
    | _, _, false, Some m2_body -> (.< .~m2_body; .~op_body >.)
    | false, Some m1_body, _, _ -> (.< .~m1_body; .~op_body >.)
  in
  let init_values_body = (.<
    .~node.value <- Ndarray.create (Ndarray.shape .~n1v);
  >.) in
  (* Not required, but we preserve the order, for readability. *)
  let init_values =
    if m1.processed && m2.processed then init_values_body
    else if m1.processed then (.< .~(m2.init_values); .~init_values_body >.)
    else if m2.processed then (.< .~(m1.init_values); .~init_values_body >.)
    else if l2r_comp_order then (.< .~(m1.init_values); .~(m2.init_values); .~init_values_body >.)
    else (.< .~(m2.init_values); .~(m1.init_values); .~init_values_body >.) in
  let toplevel_forward = (.< .~init_values; fun () -> .~forward_body >.) in
  let nd = (.< .~node.grad >.) in
  let n1d = (.< .~(m1.node).grad >.) in
  let n2d = (.< .~(m2.node).grad >.) in
  let zero_body = (.< Ndarray.reset_zeros .~nd >.) in
  (* The order of zeroing gradients is irrelevant and multiple zeroing is fine, but we avoid it
     and keep the backprop order for readability. *)
  let zero_grads =
    if m1.processed && m2.processed then zero_body
    else if m1.processed then (.< .~zero_body; .~(m2.zero_grads) >.)
    else if m2.processed then (.< .~zero_body; .~(m1.zero_grads) >.)
    else if l2r_comp_order then (.< .~zero_body; .~(m2.zero_grads); .~(m1.zero_grads) >.)
    else (.< .~zero_body; .~(m1.zero_grads); .~(m2.zero_grads) >.) in
  (* The code needs to be included in the reverse order to which it was computed! This guarantees
     that all ancestors of a node are backpropagated before the node is backpropagated, even for
     non-tree DAGs. *)
  let grad_body = grad_body ~n1d ~n2d ~nd ~nv ~n1v ~n2v in
  let backprop_body =
    match m1.processed, m1.backprop_body, m2.processed, m2.backprop_body with
    | true, _, true, _ | true, _, _, None | _, None, true, _ | _, None, _, None -> grad_body
    | false, Some m1_body, false, Some m2_body when l2r_comp_order ->
      (.< .~grad_body; .~m1_body; .~m2_body >.)
    | false, Some m1_body, false, Some m2_body ->
      (.< .~grad_body; .~m2_body; .~m1_body;  >.) 
    | _, _, false, Some m2_body -> (.< .~grad_body; .~m2_body  >.)
    | false, Some m1_body, _, _ -> (.< .~grad_body; .~m1_body  >.)
    in
  let init_grads_body = (.<
    .~node.grad <- Ndarray.create (Ndarray.shape .~nv);
  >.) in
  (* The order is not relevant, we keep the same order as in backprop for readability. *)
  let init_grads =
    if m1.processed && m2.processed then init_grads_body
    else if m1.processed then (.< .~init_grads_body; .~(m2.init_grads) >.)
    else if m2.processed then (.< .~init_grads_body; .~(m1.init_grads) >.)
    else if l2r_comp_order then (.< .~init_grads_body; .~(m2.init_grads); .~(m1.init_grads) >.)
    else (.< .~init_grads_body; .~(m1.init_grads); .~(m2.init_grads) >.) in
  let toplevel_backprop = (.<
    .~init_grads;
    fun () ->
      .~(m1.zero_grads);
      .~(m2.zero_grads);
      Ndarray.reset_ones .~nd;
      .~backprop_body
  >.) in
  (* The order is reverse to the order the updates were already executed for the first time. *)
  let local_shape_updates = Sequence.singleton local_shape_update in
  let subtree_shape_updates: shape_update_step Sequence.t =
    if m1.processed && m2.processed then local_shape_updates
    else if m1.processed then Sequence.append local_shape_updates m2.subtree_shape_updates
    else if m2.processed then Sequence.append local_shape_updates m1.subtree_shape_updates
    else if l2r_comp_order then 
      Sequence.(concat @@ of_list
                  [local_shape_updates; m2.subtree_shape_updates; m1.subtree_shape_updates])
    else Sequence.(concat @@ of_list
                     [local_shape_updates; m1.subtree_shape_updates; m2.subtree_shape_updates]) in

  m1.processed <- true; m2.processed <- true;
  {toplevel_forward; toplevel_backprop;
   forward_body=Some forward_body; backprop_body=Some backprop_body;
   init_values; init_grads; zero_grads;
   node_id; processed=false; comp_node; node;
   shape_logic; shape; subtree_shape_updates}

let unop ~op_label ?(transpose_op=`Transpose) ~op_body ~grad_body m: t =
  let m_l = m.comp_node.label in
  let m_l = if String.length m_l > 11 then "n"^Int.to_string m.node_id else m_l in
  let label = op_label ^ m_l in
  let comp_node = Node.create ~label in
  let node_id = comp_node.id in

  (* The default is that a transpose is its own inverse. *)
  let axis_labels = Map.empty (module AxisKey) in
  let shape = { batch_shape=Unknown; input_shape=Unknown; output_shape=Unknown; axis_labels;
                shape_of_node_id=node_id; deduce_output_from_input=`Not_deduced } in
  let shape_logic = Transpose_shape(transpose_op, m.shape) in
  (* let shape_update_step = { shape_update; shape; parent_shape; parent_shape_logic } in *)
  let local_shape_update = { shape; shape_logic } in
  propagate_shapes local_shape_update;

  let node = Codelib.genlet ~name:label (.< Node.get node_id >.) in
  let nv = (.< .~node.value >.) in
  let n1v = (.< .~(m.node).value >.) in
  let op_body = op_body ~nv ~n1v in
  (* The code needs to be included in the order it was computed! *)
  let forward_body =
    match m.processed, m.forward_body with
    | true, _ | _, None -> op_body
    | false, Some m_body -> (.< .~m_body; .~op_body >.) in
  let init_values = (.<
    .~(m.init_values);
    .~node.value <- Ndarray.create (Ndarray.shape .~n1v);
  >.) in
  let toplevel_forward = (.< .~init_values; fun () -> .~forward_body >.) in
  let nd = (.< .~node.grad >.) in
  let n1d = (.< .~(m.node).grad >.) in
  let zero_body = (.< Ndarray.reset_zeros .~nd >.) in
  (* The order of zeroing gradients is irrelevant and multiple zeroing is fine, but we avoid it
       and keep the backprop order for readability. *)
  let zero_grads =
    if m.processed then zero_body
    else (.< .~zero_body; .~(m.zero_grads) >.) in
  let grad_body = grad_body ~n1d ~nd ~nv ~n1v in
  (* The code needs to be included in the reverse order to which it was computed! *)
  let backprop_body =
    match m.processed, m.backprop_body with
    | true, _ | _, None -> grad_body
    | false, Some m_body -> (.< .~grad_body; .~m_body >.) in
  let init_grads_body = (.<
    .~node.grad <- Ndarray.create (Ndarray.shape .~nv);
  >.) in
  (* The order is not relevant, we keep the same order as in backprop for readability. *)
  let init_grads =
    if m.processed then init_grads_body
    else (.< .~init_grads_body; .~(m.init_grads) >.) in
  let toplevel_backprop = (.<
    .~init_grads;
    fun () ->
      .~(m.zero_grads);
      Ndarray.reset_ones .~nd;
      .~backprop_body
  >.) in
  let local_shape_updates = Sequence.singleton local_shape_update in
  let subtree_shape_updates: shape_update_step Sequence.t =
    if m.processed then local_shape_updates
    else Sequence.append local_shape_updates m.subtree_shape_updates in
  m.processed <- true;
  {toplevel_forward; toplevel_backprop;
   forward_body=Some forward_body; backprop_body=Some backprop_body;
   init_values; init_grads; zero_grads;
   node_id; processed=false; comp_node; node; shape_logic; shape; subtree_shape_updates}

(* ********** User API below ********** *)
type axis_labels_spec = string [@@deriving compare, sexp]

type term_spec =
[ `Unknown
(** The shape will need to be fully inferred. *)
| `Constant of int list * axis_labels_spec
(** [`Constant (output_dims, labels)]
    A constant shape has no batch nor input dimensions, only output dimensions. *)
| `Data of int list * int list * axis_labels_spec
(** [`Data (batch_dims, output_dims, labels)]
    A data shape does not have input dimensions. *)
| `Params of int list *  int list * axis_labels_spec
(** [`Params (input_dims, output_dims, labels)]
    A parameters shape with fixed dimensionality. Parameters not have batch dimensions. *)
| `Unknown_batch_data of int list * axis_labels_spec
(** [`Unknown_batch_data (output_dims, labels)]
    A data shape where the batch dimensions are left up to inference. *)
| `Deduced_params of deduce_dims
(** Parameters with inferred dimensionality. Example use cases:
    [`Deduced_params `Preserve] -- a hidden layer preserving the dimensionality.
    [`Deduced_params (`Scale 2.0)] -- an expansion hidden layer doubling the dimensionality.
    [`Deduced_params (`Scale 0.5)] -- an bottleneck hidden layer halving the dimensionality.
    Note that scalar axes (1D) are not scaled, for compatibility with broadcasting. *)
] [@@deriving compare, sexp]

(** Parses a labels specification.

    * If [spec] contains alphanumeric characters only, each character is converted into a label of
    the kind [Output].
    * If [spec] contains a substring ["->"] plus alphanumeric characters only, characters to the left
    of [->] are converted to [Input] labels, to the right of [->] to [Output] labels.
    * If [spec] contains any of: whitespace, comma, parentheses, then those characters are used
    as label name separators. *)
let axis_labels_of_spec (spec: string): axis_labels =
  let axis_labels = Map.empty (module AxisKey) in
  (* FIXME: NOT IMPLEMENTED *)
  ignore spec;
  axis_labels

  (* TODO: implement [einsum_of_spec] using a ["spec;spec=>spec"] syntax. *)
  
let shape_of_term_spec ~node_id : term_spec -> shape = function
| `Unknown ->
  { batch_shape=Unknown; input_shape=Unknown; output_shape=Unknown;
    axis_labels=Map.empty (module AxisKey);
    shape_of_node_id=node_id; deduce_output_from_input=`Not_deduced }
| `Constant (dims, labels_spec) ->
  { batch_shape=Given []; input_shape=Given []; output_shape=Given dims;
    axis_labels=axis_labels_of_spec labels_spec;
    shape_of_node_id=node_id; deduce_output_from_input=`Not_deduced }
| `Data (batch_dims, dims, labels_spec) ->
  { batch_shape=Given batch_dims; input_shape=Given []; output_shape=Given dims;
    axis_labels=axis_labels_of_spec labels_spec;
    shape_of_node_id=node_id; deduce_output_from_input=`Not_deduced }
| `Params (input_dims, output_dims, labels_spec) ->
  { batch_shape=Given []; input_shape=Given input_dims; output_shape=Given output_dims;
    axis_labels=axis_labels_of_spec labels_spec;
    shape_of_node_id=node_id; deduce_output_from_input=`Not_deduced }
| `Unknown_batch_data (dims, labels_spec) ->
  { batch_shape=Unknown; input_shape=Given []; output_shape=Given dims;
    axis_labels=axis_labels_of_spec labels_spec;
    shape_of_node_id=node_id; deduce_output_from_input=`Not_deduced }
| `Deduced_params deduce_output_from_input ->
  { batch_shape=Given []; input_shape=Unknown; output_shape=Unknown;
    axis_labels=Map.empty (module AxisKey);
    shape_of_node_id=node_id; deduce_output_from_input }

(** A terminal: a constant, a parameter, an input of the model. *)
let term ~label (spec: term_spec) ~(init_code:Ndarray.t Codelib.code) : t =
  let comp_node = Node.create ~label in
  let node_id = comp_node.id in
  let shape = shape_of_term_spec ~node_id spec in
  let shape_logic = Terminal_shape in
  (* FIXME: not much of an updatable info *)
  let local_shape_update = { shape; shape_logic } in
  propagate_shapes local_shape_update;

  let node = Codelib.genlet ~name:label (.< Node.get node_id >.) in
  let nv = (.< .~node.value >.) in
  (* Very unlikely someone will compute just the parameters. *)
  let forward_body = None in
  let init_values = (.< .~node.value <- .~init_code >.) in
  let toplevel_forward = (.< .~init_values; fun () -> () >.) in
  let nd = Codelib.genlet ~name:(label^"d") (.< .~node.grad >.) in
  let zero_grads = (.< Ndarray.reset_zeros .~nd >.) in
  let backprop_body = None in
  (* Very unlikely someone will want dw/dw. *)
  let init_grads = (.<
    .~node.grad <- Ndarray.create (Ndarray.shape .~nv);
  >.) in
  let toplevel_backprop = (.<
    .~init_grads;
    fun () -> Ndarray.reset_ones .~nd; ()
  >.) in
  let subtree_shape_updates = Sequence.singleton local_shape_update in
  {toplevel_forward; toplevel_backprop; forward_body; backprop_body;
    init_values; init_grads; zero_grads;
    node_id; processed=false; comp_node; node; shape_logic; shape; subtree_shape_updates}

let add =
  let op_body ~nv ~n1v ~n2v = (.< Ndarray.assign_add .~nv .~n1v .~n2v >.) in
  let grad_body ~n1d ~n2d ~nd ~nv:_ ~n1v:_ ~n2v:_ = (.<
    Ndarray.assign_add .~n1d .~n1d .~nd;
    Ndarray.assign_add .~n2d .~n2d .~nd
  >.) in
  binop ~compose_op:`Pointwise ~op_label:"t" ~op_body ~grad_body

let mul_pointwise =
  let op_body ~nv ~n1v ~n2v = (.< Ndarray.assign_mul .~nv .~n1v .~n2v >.) in
  let grad_body ~n1d ~n2d ~nd ~nv:_ ~n1v ~n2v = (.<
    Ndarray.assign_add .~n1d .~n1d (Ndarray.mul .~nd .~n2v);
    Ndarray.assign_add .~n2d .~n2d (Ndarray.mul .~nd .~n1v)
  >.) in
  binop ~compose_op:`Pointwise ~op_label:"" ~op_body ~grad_body

let matmul =
  let op_body ~nv ~n1v ~n2v = (.< Ndarray.assign_mul .~nv .~n1v .~n2v >.) in
  let grad_body ~n1d ~n2d ~nd ~nv:_ ~n1v ~n2v = (.<
    Ndarray.assign_add .~n1d .~n1d (Ndarray.mul .~nd .~n2v);
    Ndarray.assign_add .~n2d .~n2d (Ndarray.mul .~nd .~n1v)
  >.) in
  binop ~compose_op:`Compose ~op_label:"" ~op_body ~grad_body

let relu =
  let op_body ~nv ~n1v = (.< Ndarray.assign_relu .~nv .~n1v >.) in
  let grad_body ~n1d ~nd ~nv ~n1v:_ = (.<
    Ndarray.assign_add .~n1d .~n1d (Ndarray.relu_gate .~nv .~nd)
  >.) in
  unop ~op_label:"r" ~op_body ~grad_body

let init_zeroes shape = (.< let p = Ndarray.create shape in Ndarray.reset_zeros p; p >.)
let init_uniform shape = (.< Ndarray.get_uniform ~low:(-1.0) ~high:1.0 shape >.)

let float_to_label v = "v" ^ (
  Float.to_string v |> String.substr_replace_all ~pattern:"." ~with_:"p"
  |> String.substr_replace_all ~pattern:"-" ~with_:"m")

let number v =
  (* Note: no axis label so that we do not conflict with user labels. *)
  term ~label:(float_to_label v) (`Constant ([1], ""))
    ~init_code:(.< Ndarray.get_val v [|1|] >.)

module O = struct
  let ( * ) = matmul
  let ( *. ) = mul_pointwise
  let (+) = add
  let (!/) = relu
  let (!~) label shape = term ~label ~init_code:(init_uniform shape)
  let (!.) = number
  let (-) m1 m2 = m1 + !.(-1.) * m2
end

let sprint code =
  let closed, check = Codelib.close_code_delay_check code in
  ignore (Caml.Format.flush_str_formatter());
  Caml.Format.pp_set_margin Caml.Format.str_formatter 160;
  Codelib.format_code Caml.Format.str_formatter closed;
  let s = Caml.Format.flush_str_formatter() in
  let s = String.substr_replace_all s ~pattern:"Base." ~with_:"" in
  let s = String.substr_replace_all s ~pattern:"Ocannl." ~with_:"" in
  let s = String.substr_replace_all s ~pattern:"Ndarray." ~with_:"" in
  let s = String.substr_replace_all s ~pattern:"Node." ~with_:"" in
  s, check

(* TODO: maybe streamline [t] to enable [t_of_sexp]. *)
let sexp_of_t m =
  Sexp.message "Formula" [
    "label", String.sexp_of_t m.comp_node.label; "node_id", Int.sexp_of_t m.node_id;
    "toplevel_forward", String.sexp_of_t @@ fst @@ sprint m.toplevel_forward;
    "toplevel_backprop", String.sexp_of_t @@ fst @@ sprint m.toplevel_backprop;
  ]

include Comparator.Make(struct
    type nonrec t = t
    let compare m1 m2 = Int.compare m1.node_id m2.node_id
    let sexp_of_t = sexp_of_t
end)

module Summable = struct
  type nonrec t = t
  let (+) = add
  let zero = number 0.0
end

(*
let postprocess code =
  let closed, check = Codelib.close_code_delay_check code in
  let ast = Codelib.ast_of_code closed in
  Printast.expression
*)

(* 
~/ocannl$ dune utop

open Base
#load "_build/default/lib/ocannl.cma"
open Ocannl
module F = Formula
let d = [|3; 3|]
let nn = F.O.(!/(!~"w" d * !~"x" d + !~"b" d))
let () = Stdio.print_endline @@ fst @@ F.sprint nn.toplevel_forward
let () = Stdio.print_endline @@ fst @@ F.sprint nn.toplevel_backprop
*)

(** The compositional primitives for runtime-compiled code supporting backpropagation. *)

open Base

module AxisKey = struct
  module T = struct
    type shape_kind = 
      | Batch_shape
      | Input_shape
      | Output_shape
    [@@deriving compare, sexp]
    type t = shape_kind * int [@@deriving compare, sexp]
  end
  include T
  include Comparator.Make(T)
end

type axis_labels = string Map.M(AxisKey).t [@@deriving compare, sexp]

(** The datatype from which the actual Ndarray shapes are computed. In the future we can have
    named axes here instead of the predefined options.

    Mutability is sufficient to perform unification, since there is no need for backtracking and
    no explicit unification variables. [None] stands for "not yet specified". *)
type shape = {
  mutable batch_shape: int list option;
  mutable input_shape: int list option;
  mutable output_shape: int list option;
  mutable axis_labels: axis_labels;
  fixed: AxisKey.shape_kind list;
  (** The axes which were user-specified and should not be mutated. *)
  shape_of_node_id: int;
} [@@deriving fields, sexp]

type compose_type =
  [ `Pointwise
  (** NumPy-style broadcast matching batch, input and output axes, e.g. as in [s1 + s2]. *)
  | `Compose
  (** Compose the outputs of the second shape with the inputs of the first shape, i.e. the shape of
      [fun x -> s1(s2(x))], or [s1 * s2] where [*] is the inner product (e.g. matrix multiply). *)
  | `Einsum of axis_labels * axis_labels * axis_labels
  (** A version of the [einsum] syntax. Note that [`Pointwise] and [`Compose] are
      not redundant with [`Einsum], because they enable much more shape inference: they do not specify
      the number of axes. The [axis_labels] use pseudo-labels local to the notation, to line up the axes.
      
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
    There is no case for unary pointwise operations because then the shape remains the same.
    Axes are broadcast-expanded on update to fit the incoming shape, except for [TerminalShape]'s
    batch axes, which are set to empty unless user-provided. *)
type shape_logic = 
  | Broadcast of compose_type * shape * shape
  (** 

     For [Broadcast (`Einsum (ls1, ls2, ls3), s1, s2)], the labels of [s1] and [s2] must match according
     to the [ls1], [ls2] lineup, and the resulting shape inherits the labels according to the [ls3] lineup.
  *)
  | TransposeShape of transpose_type * shape
  (** Permutes the axes of a shape. The simplest [TransposeShape] is to swap inputs with outputs of [s1],
      hence the name. *)
  | TerminalShape

(** Data required for a shape inference update step. A step should equilibrate information, passing it both
    top-down and bottom-up. The child should be identifiable within the parent via physical equality
    (allowing that a child fills both slots of a binary parent). *)
type shape_update_step = {
  shape: shape;
  shape_logic: shape_logic;
  parent_shape: shape option;
  parent_shape_logic: shape_logic option;
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

(* Design choice: tensor shape is decided after code is constructed, but when it is loaded.
   I.e. code needs to be re-executed with [Runcode.run_bytecode] or [Runnative.run_native]
   when the dimensions change. *)

let propagate_shapes (update: shape_update_step) =
  (* These are only for pointwise. *)
  let axis_labels sh1 sh2 = Map.merge sh1.axis_labels sh2.axis_labels ~f:(fun ~key ->
    function
    | `Both (l1, l2) ->
      if String.equal l1 l2 then Some l1
      else
        let error = "Axis label mismatch: "^l1^" vs "^l2^" for "^
                    (Sexp.to_string_hum @@ AxisKey.sexp_of_t key) in
         raise @@ Shape_error (error, sh1, sh2)
    | `Right l | `Left l -> Some l
  ) in
  (* FIXME: NOT IMPLEMENTED *)
  ignore (axis_labels, update)

let binop ~op_label ?(compose_op=`Pointwise) ~op_body ~grad_body m1 m2: t =
  let m1_l = m1.comp_node.label in
  let m1_l = if String.length m1_l > 11 then "n"^Int.to_string m1.node_id else m1_l in
  let m2_l = m2.comp_node.label in
  let m2_l = if String.length m2_l > 11 then "n"^Int.to_string m2.node_id else m2_l in
  let label = m1_l ^ op_label ^ m2_l in
  let comp_node = Node.create ~label in
  let node_id = comp_node.id in
  let axis_labels = Map.empty (module AxisKey) in
  let shape = { batch_shape=None; input_shape=None; output_shape=None; axis_labels; fixed=[];
                shape_of_node_id=node_id } in
  let shape_logic = Broadcast (compose_op, m1.shape, m2.shape) in
  (* let shape_update_step = { shape_update; shape; parent_shape; parent_shape_logic } in *)
  let m1_shape_update = {
    shape=m1.shape;
    shape_logic=m1.shape_logic;
    parent_shape=Some shape;
    parent_shape_logic=Some shape_logic;
  } in
  let m2_shape_update = {
    shape=m2.shape;
    shape_logic=m2.shape_logic;
    parent_shape=Some shape;
    parent_shape_logic=Some shape_logic;
  } in
  let local_shape_update = {
     shape; shape_logic; parent_shape=None; parent_shape_logic=None
   } in
  propagate_shapes m1_shape_update; propagate_shapes m2_shape_update;
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
  let local_shape_updates =
    Sequence.of_list @@
      if l2r_comp_order then [local_shape_update; m2_shape_update; m1_shape_update]
      else [local_shape_update; m1_shape_update; m2_shape_update] in
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
  let shape = { batch_shape=None; input_shape=None; output_shape=None; axis_labels; fixed=[];
                shape_of_node_id=node_id } in
  let shape_logic = TransposeShape(transpose_op, m.shape) in
  (* let shape_update_step = { shape_update; shape; parent_shape; parent_shape_logic } in *)
  let m_shape_update = {
    shape=m.shape;
    shape_logic=m.shape_logic;
    parent_shape=Some shape;
    parent_shape_logic=Some shape_logic;
  } in
  let local_shape_update = {
    shape; shape_logic; parent_shape=None; parent_shape_logic=None; } in
  propagate_shapes m_shape_update;
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
  let local_shape_updates = Sequence.of_list [local_shape_update; m_shape_update] in
  let subtree_shape_updates: shape_update_step Sequence.t =
    if m.processed then local_shape_updates
    else Sequence.append local_shape_updates m.subtree_shape_updates in
  m.processed <- true;
  {toplevel_forward; toplevel_backprop;
   forward_body=Some forward_body; backprop_body=Some backprop_body;
   init_values; init_grads; zero_grads;
   node_id; processed=false; comp_node; node; shape_logic; shape; subtree_shape_updates}

(* ********** User API below ********** *)

(** A terminal: a constant, a parameter, an input of the model. *)
let term ~label ?shape_spec ~(init_code:Ndarray.t Codelib.code) : t =
  let comp_node = Node.create ~label in
  let node_id = comp_node.id in
  let axis_labels = Map.empty (module AxisKey) in
  let shape =
    match shape_spec with
    | None -> { batch_shape=None; input_shape=None; output_shape=None; axis_labels; fixed=[];
                shape_of_node_id=node_id }
    | Some spec -> spec in
    (* FIXME: spec should be easy to provide partially. *)
  let shape_logic = TerminalShape in
  (* FIXME: not much of an updatable info *)
  let local_shape_update = {
     shape; shape_logic; parent_shape=None; parent_shape_logic=None;
   } in
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
  (* TODO(5): use dimensions inference and broadcasting. *)
  term ~label:(float_to_label v) 
    ~shape_spec:{batch_shape=Some []; input_shape=Some []; output_shape=Some [1];
                 axis_labels=Map.empty (module AxisKey);
                 fixed=[Batch_shape; Input_shape; Output_shape]; shape_of_node_id=0}
                 (* FIXME: this spec is broken *)
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

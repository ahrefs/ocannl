(** Construction of runtime-compiled code supporting backpropagation. *)

open Base

(** Information needed for compositional code generation. The code generation is suspended so that
    it can incorporate inferred shape information. *)
type t = {
  forward_body: (unit -> unit Codelib.code) option;
  (** Uses [option], i.e. [None] instead of [fun () -> .< () >.], to improve readability of generated code. *)
  init_values: unit -> unit Codelib.code;
  (** Initializes the values. *)
  init_grads: unit -> unit Codelib.code;
  (** Initializes the gradient data: typically, simply creates the ndarrays.
      Gradients are zeroed separately. The code is suspended so that it can incorporate shape inference. *)
  backprop_body: (unit -> unit Codelib.code) option;
  zero_grads: unit -> unit Codelib.code;
  (** Initializes the backpropagation phase. Computed once per backpropagation. *)
  node_id: int;
  comp_node: Ocannl_runtime.Node.t;
  (** This tracks the computation node as long as the model is not cross-compiled to a different
      process etc. *)
  node: Ocannl_runtime.Node.t Codelib.code;
  (** The node storing the computation results. [.!(t.node)] should equal [t.comp_node]. *)
  shape_logic: Shape.logic;
  (** How to do the last update of [t.shape] when finalizing the formula. *)
  shape: Shape.t;
  (** The eventual shape of [.!(t.node).value] and [.!(t.node).grad], incorporating the current state of
      shape inference. *)
  needs_gradient: bool;
  (** An optimization setting: whether gradients should be backpropagated into the formula. If any subformula
      needs gradients, this formula also needs gradients. *)
}

(** A global root is a formula that is not (currently) a subformula of another formula. *)
type global_root = {
  mutable forward_code: (unit -> unit) Codelib.code option;
  mutable forward: (unit -> unit) option;
  mutable backprop_code: (unit -> unit) Codelib.code option;
  mutable backprop: (unit -> unit) option;
  formula: t;
  subtree_shape_updates: Shape.update_step Sequence.t;
  (** We piggy-back on the code generation setup to arrange the updates. We perform each update twice
      to propagate information between all subformulas: first in postfix order while computing [t],
      second in prefix order by iterating over [t.subtree_shape_updates]. *)
}

(** If a formula with [node_id >= !first_session_id] is not among global roots, it must be a subformula
    of a global root. *)
let global_roots = ref @@ Map.empty (module Int)

(** A current session is the range of nodes from [!first_session_id] to [global.unique_id]. Subformulas
    with [node_id] before this range are not allowed in new formulas. *)
let first_session_id = ref 1

exception Session_error of string * t option

(* [reset_] and [create_] functions are the only direct users of [Ndarray] functions inside [Formula].
   The other uses are mediated by the [~op_body], [~grad_body] and [~init_code] arguments. *)
let reset_zeros n shape =
   Ndarray.(accum_unop_code ~accum:skip_arg_code ~op:(fun _ -> zero_code) ~lhs:n ~rhs:n
              (Shape.terminal_projections shape))
let reset_ones n shape =
  Ndarray.(accum_unop_code ~accum:skip_arg_code ~op:(fun _ -> one_code) ~lhs:n ~rhs:n
             (Shape.terminal_projections shape))
let create_value node shape =
   .< Ocannl_runtime.Node.(.~node.value <- create_array .~(Shape.to_dims_code shape)) >.
let create_grad node shape =
   .< Ocannl_runtime.Node.(.~node.grad <- create_array .~(Shape.to_dims_code shape)) >.

let binop ~op_label ?(compose_op=`Pointwise) ~op_body ~grad_body m1arg m2arg: t =
  let m1, m2 = if m1arg.node_id <= m2arg.node_id then m1arg, m2arg else m2arg, m1arg in
  (* Note: do not capture m1, m2 in any closure, so they can be GC'd. *)
  (if (m1.node_id < !first_session_id) then
    raise @@ Session_error ("The subformula is outside of current session", Some m1));
  (if (m2.node_id < !first_session_id) then
     raise @@ Session_error ("The subformula is outside of current session", Some m2));
  let m1_l = m1.comp_node.label in
  let m1_l = if String.length m1_l > 11 then "n"^Int.to_string m1.node_id else m1_l in
  let m2_l = m2.comp_node.label in
  let m2_l = if String.length m2_l > 11 then "n"^Int.to_string m2.node_id else m2_l in
  let label = m1_l ^ op_label ^ m2_l in
  let comp_node = Ocannl_runtime.Node.create ~label in
  let node_id = comp_node.id in
  let shape = Shape.{ batch=Unknown; input=Unknown; output=Unknown;
                      axis_labels=Map.empty (module AxisKey);
                     deduce_output_from_input=`Not_deduced } in
  let shape_logic = Shape.Broadcast (compose_op, m1.shape, m2.shape) in
  let local_shape_update = Shape.{ shape; logic=shape_logic } in
  Shape.propagate_shapes local_shape_update;
  let node = Codelib.genlet ~name:label .< Ocannl_runtime.Node.get node_id >. in
  let nv = .< .~node.value >. in
  let n1v = .< .~(m1.node).value >. in
  let n2v = .< .~(m2.node).value >. in
  let projections = Shape.derive_projections local_shape_update in
  let op_body() = op_body ~nv ~n1v ~n2v projections in
  let m1_processed = not @@ Map.mem !global_roots m1.node_id in
  let m2_processed = not @@ Map.mem !global_roots m2.node_id in
  (* The code needs to be included in the order it was computed! *)
  let m1_forward_body = m1.forward_body in
  let m2_forward_body = m2.forward_body in
  let forward_body =
    (match m1_processed, m1_forward_body, m2_processed, m2_forward_body with
    | true, _, true, _ | true, _, _, None | _, None, true, _ | _, None, _, None -> op_body
    | false, Some m1_body, false, Some m2_body ->
      fun () -> .< .~(m1_body()); .~(m2_body()); .~(op_body()) >.
    | _, _, false, Some m2_body -> fun () -> .< .~(m2_body()); .~(op_body()) >.
    | false, Some m1_body, _, _ -> fun () -> .< .~(m1_body()); .~(op_body()) >.)
  in
  let init_values_body() = create_value node shape in
  let m1_init_values = m1.init_values in
  let m2_init_values = m2.init_values in
  let init_values =
    (if m1_processed && m2_processed then init_values_body
    else if m1_processed then fun () -> .< .~(m2_init_values ()); .~(init_values_body ()) >.
    else if m2_processed then fun () -> .< .~(m1_init_values ()); .~(init_values_body ()) >.
    else fun () -> .< .~(m1_init_values ()); .~(m2.init_values ()); .~(init_values_body ()) >.) in
  let ng = .< .~node.grad >. in
  let n1g = .< .~(m1.node).grad >. in
  let n2g = .< .~(m2.node).grad >. in
  let zero_body() = reset_zeros ng shape in
  (* The order of zeroing gradients is irrelevant and multiple zeroing is fine, but we avoid it
     and keep the backprop order for readability. *)
  let needs_gradient = m1.needs_gradient || m2.needs_gradient in
  let m1_no_grad = m1_processed || not m1.needs_gradient in
  let m2_no_grad = m2_processed || not m2.needs_gradient in
  let m1_zero_grads = m1.zero_grads in
  let m2_zero_grads = m2.zero_grads in
  let zero_grads =
    if m1_no_grad && m2_no_grad then zero_body
    else if m1_no_grad then fun () -> .< .~(zero_body()); .~(m2_zero_grads()) >.
    else if m2_no_grad then fun () -> .< .~(zero_body()); .~(m1_zero_grads()) >.
    else fun () -> .< .~(zero_body()); .~(m2_zero_grads()); .~(m1_zero_grads()) >. in
  (* The code needs to be included in the reverse order to which it was computed! This guarantees
     that all ancestors of a node are backpropagated before the node is backpropagated, even for
     non-tree DAGs. *)
  let grad_body() = grad_body ~n1g ~n2g ~ng ~nv ~n1v ~n2v projections in
  (* For consistency / debuggability, we keep grad_body in backprop_body even when needs_gradients
     is false. *)
  let backprop_body =
    match m1_no_grad, m1.backprop_body, m2_no_grad, m2.backprop_body with
    | true, _, true, _ | true, _, _, None | _, None, true, _ | _, None, _, None -> grad_body
    | false, Some m1_body, false, Some m2_body ->
      fun () -> .< .~(grad_body()); .~(m1_body()); .~(m2_body()) >.
    | _, _, false, Some m2_body ->
      fun () -> .< .~(grad_body()); .~(m2_body())  >.
    | false, Some m1_body, _, _ ->
      fun () -> .< .~(grad_body()); .~(m1_body())  >.
    in
  let init_grads_body() = create_grad node shape in
  (* The order is not relevant, we keep the same order as in backprop for readability. *)
  let m1_init_grads = m1.init_grads in
  let m2_init_grads = m2.init_grads in
  let init_grads =
    if m1_no_grad && m2_no_grad then init_grads_body
    else if m1_no_grad then fun () -> .< .~(init_grads_body ()); .~(m2_init_grads ()) >.
    else if m2_no_grad then fun () -> .< .~(init_grads_body ()); .~(m1_init_grads ()) >.
    else fun () -> .< .~(init_grads_body ()); .~(m2_init_grads ()); .~(m1_init_grads ()) >. in
  (* The order is reverse to the order the updates were already executed for the first time. *)
  let local_shape_updates = Sequence.singleton local_shape_update in
  let subtree_shape_updates: Shape.update_step Sequence.t =
    if m1_processed && m2_processed then local_shape_updates
    else if m1_processed then Sequence.append local_shape_updates @@
      (Map.find_exn !global_roots m2.node_id).subtree_shape_updates
    else if m2_processed then Sequence.append local_shape_updates @@
      (Map.find_exn !global_roots m1.node_id).subtree_shape_updates
    else
      Sequence.(concat @@ of_list
                  [local_shape_updates; (Map.find_exn !global_roots m2.node_id).subtree_shape_updates;
                  (Map.find_exn !global_roots m1.node_id).subtree_shape_updates]) in

  (if not m1_processed then global_roots := Map.remove !global_roots m1.node_id);
  (if not m2_processed then global_roots := Map.remove !global_roots m2.node_id);
  let formula = {forward_body=Some forward_body; backprop_body=Some backprop_body;
                init_values; init_grads; zero_grads;
                node_id; comp_node; node; shape_logic; shape; needs_gradient} in
  let root = {forward_code=None; forward=None; backprop_code=None; backprop=None; 
              formula; subtree_shape_updates} in
  global_roots := Map.add_exn !global_roots ~key:node_id ~data:root;
  formula

let unop ~op_label ?init_shape ~transpose_op ~op_body ~grad_body m: t =
  (* Note: do not capture m in any closure, so it can be GC'd. *)
  let m_l = m.comp_node.label in
  let m_l = if String.length m_l > 11 then "n"^Int.to_string m.node_id else m_l in
  let label = op_label ^ m_l in
  let comp_node = Ocannl_runtime.Node.create ~label in
  let node_id = comp_node.id in

  let shape =
    match init_shape with
    | None ->
      Shape.{ batch=Unknown; input=Unknown; output=Unknown;
              axis_labels=Map.empty (module AxisKey);
              deduce_output_from_input=`Not_deduced }
    | Some shape -> shape in
  let shape_logic = Shape.Transpose(transpose_op, m.shape) in
  let local_shape_update = Shape.{ shape; logic=shape_logic } in
  Shape.propagate_shapes local_shape_update;

  let node = Codelib.genlet ~name:label .< Ocannl_runtime.Node.get node_id >. in
  let nv = .< .~node.value >. in
  let n1v = .< .~(m.node).value >. in
  let projections = Shape.derive_projections local_shape_update in
  let op_body() = op_body ~nv ~n1v projections in
  let m_processed = not @@ Map.mem !global_roots m.node_id in
  (* The code needs to be included in the order it was computed! *)

  let forward_body =
    match m_processed, m.forward_body with
    | true, _ | _, None -> op_body
    | false, Some m_body -> fun() -> .< .~(m_body()); .~(op_body()) >. in
  let m_init_values = m.init_values in
  let init_values = fun () -> .<
    .~(m_init_values ());
    .~(create_value node shape);
  >. in
  let ng = .< .~node.grad >. in
  let n1g = .< .~(m.node).grad >. in
  let zero_body() = reset_zeros ng shape in
  (* The order of zeroing gradients is irrelevant and multiple zeroing is fine, but we avoid it
       and keep the backprop order for readability. *)
  let needs_gradient = m.needs_gradient in
  let m_zero_grads = m.zero_grads in
  let zero_grads =
    if m_processed then zero_body
    else fun () -> .< .~(zero_body()); .~(m_zero_grads()) >. in
  let grad_body() = grad_body ~n1g ~ng ~nv ~n1v projections in
  (* The code needs to be included in the reverse order to which it was computed! *)
  let backprop_body =
    match m_processed, m.backprop_body with
    | true, _ | _, None -> grad_body
    | false, Some m_body -> fun () -> .< .~(grad_body()); .~(m_body()) >. in
  let init_grads_body = fun () -> create_grad node shape in
  (* The order is not relevant, we keep the same order as in backprop for readability. *)
  let m_init_grads = m.init_grads in
  let init_grads =
    if m_processed then init_grads_body
    else fun () -> .< .~(init_grads_body ()); .~(m_init_grads ()) >. in
  let local_shape_updates = Sequence.singleton local_shape_update in
  let subtree_shape_updates: Shape.update_step Sequence.t =
    if m_processed then local_shape_updates
    else Sequence.append local_shape_updates @@
    (Map.find_exn !global_roots m.node_id).subtree_shape_updates in

  (if not m_processed then global_roots := Map.remove !global_roots m.node_id);
  let formula = {forward_body=Some forward_body; backprop_body=Some backprop_body;
                 init_values; init_grads; zero_grads;
                 node_id; comp_node; node; shape_logic; shape; needs_gradient} in
  let root = {forward_code=None; forward=None; backprop_code=None; backprop=None; 
              formula; subtree_shape_updates} in
  global_roots := Map.add_exn !global_roots ~key:node_id ~data:root;
  formula

let term_needs_gradient (spec: Shape.term_spec) =
  match spec with
  | `Unknown -> true
  | `Constant _ -> false
  | `Data _ -> false
  | `Params _ -> true
  | `Unknown_batch_data _ -> false
  | `Deduced_params _ -> true

(** A terminal: a constant, a parameter, an input of the model. *)
let term ~label ?needs_gradient (spec: Shape.term_spec) ~op_body : t =
  let comp_node = Ocannl_runtime.Node.create ~label in
  let node_id = comp_node.id in
  let shape = Shape.of_term_spec spec in
  let needs_gradient =
    match needs_gradient with
    | None -> term_needs_gradient spec
    | Some setting -> setting in
  let shape_logic = Shape.Terminal in
  (* NOTE: this update does not do any work, but that might change in the future,
     and having it in the update sequence might help with debuggability. *)
  let local_shape_update = Shape.{ shape; logic=shape_logic } in
  Shape.propagate_shapes local_shape_update;

  let node = Codelib.genlet ~name:label .< Ocannl_runtime.Node.get node_id >. in
  let nv = .< .~node.value >. in
  (* Very unlikely someone will compute just the parameters. *)
  let forward_body = None in
  let init_values() = op_body ~nv shape in
  let ng = .< .~node.grad >. in
  let zero_grads() = reset_zeros ng shape in
  let backprop_body = None in
  (* Very unlikely someone will want dw/dw. *)
  let init_grads = fun () -> create_grad node shape in
  let subtree_shape_updates = Sequence.singleton local_shape_update in
  let formula = {forward_body; backprop_body;
                 init_values; init_grads; zero_grads;
                 node_id; comp_node; node; shape_logic; shape; needs_gradient} in
  let root = {forward_code=None; forward=None; backprop_code=None; backprop=None; 
              formula; subtree_shape_updates} in
  global_roots := Map.add_exn !global_roots ~key:node_id ~data:root;
  formula

let get_toplevel m =
  let forward_body = match m.forward_body with None -> .< () >. | Some body -> body() in
  let toplevel_forward = .< .~(m.init_values ()); fun () -> .~forward_body >. in
  let backprop_body = match m.backprop_body with None -> .< () >. | Some body -> body() in
  let ng = .< .~(m.node).Ocannl_runtime.Node.grad >. in
  let toplevel_backprop =
    if not m.needs_gradient then .< fun () -> () >.
    else .<
    .~(m.init_grads ());
    fun () ->
      .~(m.zero_grads());
      .~(reset_ones ng m.shape);
      .~backprop_body
      >. in
  toplevel_forward, toplevel_backprop

let sexp_of_t m =
  Sexp.message "Formula" [
    "label", String.sexp_of_t m.comp_node.label; "node_id", Int.sexp_of_t m.node_id;
  ]

include Comparator.Make(struct
    type nonrec t = t
    let compare m1 m2 = Int.compare m1.node_id m2.node_id
    let sexp_of_t = sexp_of_t
end)

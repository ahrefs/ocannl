(** Construction of runtime-compiled code supporting backpropagation. *)

open Base

(** Information needed for compositional code generation. The code generation is suspended so that
    it can incorporate inferred shape information. *)
type t = {
  forward_body: Code.t;
  init_values: Code.t;
  (** Initializes the values. *)
  init_grads: Code.t;
  (** Initializes the gradient data: typically, simply creates the arrays.
      Gradients are zeroed separately. The code is suspended so that it can incorporate shape inference. *)
  backprop_body: Code.t;
  zero_grads: Code.t;
  (** Initializes the backpropagation phase. Computed once per backpropagation. *)
  node_id: int;
  comp_node: Ocannl_runtime.Node.t;
  (** This tracks the computation node as long as the model is not cross-compiled to a different
      process etc. *)
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
  mutable forward_code: Code.program option;
  mutable backprop_code: Code.program option;
  formula: t;
  subtree_shape_updates: Shape.update_step Sequence.t;
  (** We piggy-back on the code generation setup to arrange the updates. We perform each update twice
      to propagate information between all subformulas: first in postfix order while computing [t],
      second in prefix order by iterating over [t.subtree_shape_updates]. *)
}

(** If a formula with [node_id >= !first_session_id] is not among global roots, it must be a subformula
    of a global root. *)
let global_roots = ref @@ Map.empty (module Int)

(** A current session is the range of nodes from [!first_session_id] to [Node.global.unique_id - 1],
    or an empty range if [!first_session_id = Node.global.unique_id].
    Subformulas with [node_id] before this range are not allowed in new formulas. *)
let first_session_id = ref 1

exception Session_error of string * t option

(** Prefix the input with the header information of all nodes within the current session. *)
let prefix_with_preamble content =
  let open Ocannl_runtime in
  let result = Buffer.create 16 in
  let ap = Buffer.add_string result in
  for i = !first_session_id to Node.global.unique_id - 1 do
    let n = NodeUI.node_header @@ Node.get i in
    ap"Node "; ap n; ap";\n";
  done;
  ap content; 
  Buffer.contents result

let session_error_printer = function
  | Session_error (msg, None) -> Some (prefix_with_preamble msg)
  | Session_error (msg, Some m) ->
    Some (prefix_with_preamble @@ "For #"^Int.to_string_hum m.node_id^": "^msg)
  | _ -> None

let () = Caml.Printexc.register_printer session_error_printer
  
let reset_zeros node field _shape =
  Code.Reset {tensor={node; field}; precision=Single; reset_op=`ConstantOfValue 0.0}

let reset_ones node field _shape =
  Code.Reset {tensor={node; field}; precision=Single; reset_op=`ConstantOfValue 1.0}

let create node field shape =
  Code.Create {tensor={node; field}; dims=(fun () -> Shape.to_dims shape); init_op=`Unspecified;
               precision=Single}

let max_sublabel_length = ref 25
               
let binop ~op_label ?(compose_op=`Pointwise) ~op_body ~grad_body m1arg m2arg: t =
  let m1, m2 = if m1arg.node_id <= m2arg.node_id then m1arg, m2arg else m2arg, m1arg in
  (* Note: do not capture m1, m2 in any closure, so they can be GC'd. *)
  (if (m1.node_id < !first_session_id) then
    raise @@ Session_error ("The subformula is outside of current session", Some m1));
  (if (m2.node_id < !first_session_id) then
     raise @@ Session_error ("The subformula is outside of current session", Some m2));
  let m1_l = m1.comp_node.label in
  let m1_l = if String.length m1_l > !max_sublabel_length then "n"^Int.to_string m1.node_id else m1_l in
  let m2_l = m2.comp_node.label in
  let m2_l = if String.length m2_l > !max_sublabel_length then "n"^Int.to_string m2.node_id else m2_l in
  let label = "(" ^ m1_l ^ op_label ^ m2_l ^ ")" in
  let n = Ocannl_runtime.Node.create ~label in
  let node_id = n.id in
  let shape = Shape.{ batch=Unknown; input=Unknown; output=Unknown;
                      axis_labels=Map.empty (module AxisKey);
                      deduce_output_from_input=`Not_deduced } in
  let shape_logic = Shape.Broadcast (compose_op, m1.shape, m2.shape) in
  let local_shape_update = Shape.{ shape; logic=shape_logic } in
  Shape.propagate_shapes local_shape_update;
  let n1 = m1.comp_node in
  let n2 = m2.comp_node  in
  let projections() = Shape.derive_projections local_shape_update in
  let op_body = op_body ~n ~n1 ~n2 projections in
  let m1_processed = not @@ Map.mem !global_roots m1.node_id in
  let m2_processed = not @@ Map.mem !global_roots m2.node_id in
  (* The code needs to be included in the order it was computed! *)
  let open Code in
  let forward_body =
    (match m1_processed, m1.forward_body, m2_processed, m2.forward_body with
    | true, _, true, _ | true, _, _, Noop | _, Noop, true, _ | _, Noop, _, Noop -> op_body
    | false, m1_body, false, m2_body -> Seq (ParHint (m1_body, m2_body), op_body)
    | _, _, false, m2_body -> Seq (m2_body, op_body)
    | false, m1_body, _, _ -> Seq(m1_body, op_body)) in
  let init_values_body = create n `Value shape in
  let init_values =
    if m1_processed && m2_processed then init_values_body
    else if m1_processed then Par (m2.init_values, init_values_body)
    else if m2_processed then Par (m1.init_values, init_values_body)
    else Par (Par (m1.init_values, m2.init_values), init_values_body) in
  let needs_gradient = m1.needs_gradient || m2.needs_gradient in
  let m1_no_grad = m1_processed || not m1.needs_gradient in
  let m2_no_grad = m2_processed || not m2.needs_gradient in
  let zero_body = if needs_gradient then reset_zeros n `Grad shape else Noop in
  (* The order of zeroing gradients is irrelevant and multiple zeroing is fine, but we avoid it
     and keep the backprop order for readability. *)
  let zero_grads =
    if m1_no_grad && m2_no_grad then zero_body
    else if m1_no_grad then Par (zero_body, m2.zero_grads)
    else if m2_no_grad then Par (zero_body, m1.zero_grads)
    else Par (zero_body, Par (m2.zero_grads, m1.zero_grads)) in
  (* The code needs to be included in the reverse order to which it was computed! This guarantees
     that all ancestors of a node are backpropagated before the node is backpropagated, even for
     non-tree DAGs. *)
  (*  *)
  let grad_body =
    if needs_gradient then
      grad_body ~n ~n1 ~n2 ~needs1:m1.needs_gradient ~needs2:m2.needs_gradient projections
    else Noop in
  let backprop_body =
    match m1_no_grad, m1.backprop_body, m2_no_grad, m2.backprop_body with
    | true, _, true, _ | true, _, _, Noop | _, Noop, true, _ | _, Noop, _, Noop -> grad_body
    | false, m1_body, false, m2_body -> Seq (grad_body, ParHint(m2_body, m1_body))
    | _, _, false, m2_body -> Seq (grad_body, m2_body)
    | false, m1_body, _, _ -> Seq (grad_body, m1_body) in
  let init_grads_body = create n `Grad shape in
  (* The order is not relevant, we keep the same order as in backprop for readability. *)
  let m1_init_grads = m1.init_grads in
  let m2_init_grads = m2.init_grads in
  let init_grads =
    if m1_no_grad && m2_no_grad then init_grads_body
    else if m1_no_grad then Par (init_grads_body, m2_init_grads)
    else if m2_no_grad then Par (init_grads_body, m1_init_grads)
    else Par (init_grads_body, Par (m2_init_grads, m1_init_grads))in
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
  let formula = {forward_body; backprop_body;
                init_values; init_grads; zero_grads;
                node_id; comp_node=n; shape_logic; shape; needs_gradient} in
  let root = {forward_code=None; backprop_code=None; 
              formula; subtree_shape_updates} in
  global_roots := Map.add_exn !global_roots ~key:node_id ~data:root;
  formula

let unop ~op_label ?init_shape ~transpose_op ~op_body ~grad_body m: t =
  (* Note: do not capture m in any closure, so it can be GC'd. *)
  let m_l = m.comp_node.label in
  let m_l = if String.length m_l > !max_sublabel_length then "n"^Int.to_string m.node_id else m_l in
  let label = op_label ^ "(" ^ m_l ^ ")" in
  let n = Ocannl_runtime.Node.create ~label in
  let node_id = n.id in

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

  let n1 = m.comp_node in
  let projections() = Shape.derive_projections local_shape_update in
  let op_body = op_body ~n ~n1 projections in
  let m_processed = not @@ Map.mem !global_roots m.node_id in
  (* The code needs to be included in the order it was computed! *)
  let open Code in
  let forward_body =
    match m_processed, m.forward_body with
    | true, _ | _, Noop -> op_body
    | false, m_body -> Seq (m_body, op_body) in
  let init_values = Par (m.init_values, create n `Value shape) in
  let needs_gradient = m.needs_gradient in
  let m_no_grad = m_processed || not m.needs_gradient in
  let zero_body = if needs_gradient then reset_zeros n `Grad shape else Noop in
  let zero_grads =
    if m_no_grad then zero_body
    else Par (zero_body, m.zero_grads) in
  let grad_body =
    if needs_gradient then grad_body ~n ~n1 projections else Noop in
  (* The code needs to be included in the reverse order to which it was computed! *)
  let backprop_body =
    match m_no_grad, m.backprop_body with
    | true, _ | _, Noop -> grad_body
    | false, m_body -> Seq (grad_body, m_body) in
  let init_grads_body = create n `Grad shape in
  let init_grads =
    if m_no_grad then init_grads_body
    else Par (init_grads_body, m.init_grads) in
  let local_shape_updates = Sequence.singleton local_shape_update in
  let subtree_shape_updates: Shape.update_step Sequence.t =
    if m_processed then local_shape_updates
    else Sequence.append local_shape_updates @@
      (Map.find_exn !global_roots m.node_id).subtree_shape_updates in

  (if not m_processed then global_roots := Map.remove !global_roots m.node_id);
  let formula = {forward_body; backprop_body;
                 init_values; init_grads; zero_grads;
                 node_id; comp_node=n; shape_logic; shape; needs_gradient} in
  let root = {forward_code=None; backprop_code=None; 
              formula; subtree_shape_updates} in
  global_roots := Map.add_exn !global_roots ~key:node_id ~data:root;
  formula

(** The default [needs_gradient] behavior. *)
let term_needs_gradient (spec: Shape.term_spec) =
  match spec with
  | `Unknown -> true
  | `Constant _ -> false
  | `Data _ -> false
  | `Params _ -> true
  | `Unknown_batch_data _ -> false
  | `Deduced_params _ -> true

(** A terminal: a constant, a parameter, an input of the model. *)
let term ~label ?needs_gradient (spec: Shape.term_spec) ~init_body : t =
  let n = Ocannl_runtime.Node.create ~label in
  let node_id = n.id in
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

  let open Code in
  let forward_body = Noop in
  let init_values = Par (create n `Value shape, init_body ~n `Value shape) in
  let zero_grads = reset_zeros n `Grad shape in
  let backprop_body = Noop in
  (* Very unlikely someone will want dw/dw. *)
  let init_grads = create n `Grad shape in
  let subtree_shape_updates = Sequence.singleton local_shape_update in
  let formula = {forward_body; backprop_body;
                 init_values; init_grads; zero_grads;
                 node_id; comp_node=n; shape_logic; shape; needs_gradient} in
  let root = {forward_code=None; backprop_code=None; 
              formula; subtree_shape_updates} in
  global_roots := Map.add_exn !global_roots ~key:node_id ~data:root;
  formula

let error_if_unknown_shape m =
  match m.shape with
  | {input=Unknown; _} -> raise @@ Session_error ("Shape of inputs is still unknown", Some m)
  | {output=Unknown; _} -> raise @@ Session_error ("Shape of outputs is still unknown", Some m)
  | {batch=Unknown; _} -> raise @@ Session_error ("Shape of batching is still unknown", Some m)
  | {output=Inferred []; _} ->
     raise @@ Session_error ("Shape of outputs is still empty -- missing shape information", Some m)
  | {input=_; output=_; batch=_; axis_labels=_; deduce_output_from_input=_} -> ()

let get_toplevel m =
  error_if_unknown_shape m;
  let open Code in
  let toplevel_forward = 
    {initialization=m.init_values; procedure=m.forward_body;
     routine={node=m.comp_node; field=`Forward};
     label="Forward #"^Int.to_string m.node_id^" "^m.comp_node.label} in
  let backprop =
    Seq (Par (m.zero_grads, reset_ones m.comp_node `Grad m.shape), m.backprop_body) in
  let toplevel_backprop =
    {initialization=m.init_grads; procedure=backprop;
     routine={node=m.comp_node; field=`Backprop};
     label="Backprop #"^Int.to_string m.node_id^" "^m.comp_node.label} in
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

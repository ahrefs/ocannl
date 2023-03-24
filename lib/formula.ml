(** Construction of runtime-compiled code supporting backpropagation. *)

open Base

type form = {
  backprop_body: Code.t;
  (** Performs backpropagation for the formula, which typically means adding partial gradients
      to the gradient tensor of the subformulas. *)
  needs_gradient: bool;
  (** An optimization setting: whether gradients should be backpropagated into the formula.
      If any subformula needs gradients, this formula also needs gradients. *)
} [@@deriving sexp]

(** Information needed for compositional code generation. The code generation is suspended so that
    it can incorporate inferred shape information. *)
type t = {
  forward_body: Code.t;
  (** Initializes the values. *)
  form: form option;
  id: int;
  node: NodeUI.t;
  (** This tracks the computation node as long as the model is not cross-compiled to a different
      process etc. *)
  shape_logic: Shape.logic;
  (** How to do the last update of [t.shape] when finalizing the formula. *)
  shape: Shape.t;
  (** The eventual shape of [.!(t.node).value] and [.!(t.node).grad], incorporating the current state of
      shape inference. *)
} [@@deriving sexp_of]

(** A global root is a formula that is not (currently) a subformula of another formula. *)
type global_root = {
  mutable forward_code: Code.program option;
  mutable backprop_code: Code.program option;
  formula: t;
}

(** If a formula with [id >= !first_session_id] is not among global roots, it must be a subformula
    of a global root. *)
let global_roots = ref @@ Map.empty (module Int)

(** We perform each update (at least) twice to propagate information between all subformulas:
    first in postfix order while computing [t], then in prefix order by iterating over this stack. *)
let session_shape_updates: Shape.update_step list ref = ref []

(** This code will usually be executed only once, after the shapes are inferred. But it will also
    be executed by each [Session.refresh_session ~regenerate:true] and 
    [Session.refresh_session ~reinit:true] call, except if [~force_no_init:true].
    Execution potentially in parallel. *)
let session_initializations: Code.t list ref = ref []
let session_initialized = ref 0

(** This code will be executed on each [Session.refresh_session ~run:true] call ([~run:true]
    is implicit), before any [forward] or [backprop] code. Execution potentially in parallel. *)
let session_prepare_step: Code.t list ref = ref []

(** A current session is the range of nodes from [!first_session_id] to [Node.global.unique_id - 1],
    or an empty range if [!first_session_id = Node.global.unique_id].
    Subformulas with [id] before this range are not allowed in new formulas. *)
let first_session_id = ref 1

exception Session_error of string * t option

(** Prefix the input with the header information of all nodes within the current session. *)
let prefix_with_preamble content =
  let open Ocannl_runtime in
  let result = Buffer.create 16 in
  let ap = Buffer.add_string result in
  for i = !first_session_id to Node.global.unique_id - 1 do
    let n = NodeUI.node_header @@ NodeUI.get i in
    ap"Node "; ap n; ap";\n";
  done;
  ap content; 
  Buffer.contents result

let session_error_printer = function
  | Session_error (msg, None) -> Some (prefix_with_preamble msg)
  | Session_error (msg, Some m) ->
    Some (prefix_with_preamble @@ "For #"^Int.to_string_hum m.id^": "^msg)
  | _ -> None

let () = Caml.Printexc.register_printer session_error_printer

let handle_error ?formula message =
  let exc = Session_error (message, formula) in
  Stdio.prerr_endline @@ Option.value_exn (session_error_printer exc);
  raise exc

let reset_zeros ~id field _shape =
  Code.Reset {tensor={id; field}; reset_op=`Constant_of_value 0.0}

let reset_ones ~id field _shape =
  Code.Reset {tensor={id; field}; reset_op=`Constant_of_value 1.0}

let create ~id ?(init_op=`Unspecified) field shape =
  Code.Create {tensor={id; field}; dims=(fun () -> Shape.to_dims shape); init_op}

let max_sublabel_length = ref 25

let binop ~op_label ?(compose_op=Shape.Pointwise_bin) ~op_body ~grad_body ~is_form m1 m2 =
  (* Note: do not capture m1, m2 in any closure, so they can be GC'd. *)
  (if (m1.id < !first_session_id) then
    raise @@ Session_error ("The subformula is outside of current session", Some m1));
  (if (m2.id < !first_session_id) then
     raise @@ Session_error ("The subformula is outside of current session", Some m2));
  let m1_first = m1.id <= m2.id in
  let m1_processed: bool =
    Option.is_some m1.form && not @@ Map.mem !global_roots m1.id in
  let m2_processed: bool =
    Option.is_some m2.form && (m2.id = m1.id || not @@ Map.mem !global_roots m2.id) in
  let children = [{NodeUI.sub_node_id=m1.id; computed_externally=m1_processed};
                  {sub_node_id=m2.id; computed_externally=m2_processed}] in
  let n = NodeUI.create_of_promoted_precision ~is_form m1.node.node m2.node.node
      ~op_label ~shape_spec:Unknown_shape ~children in
  let id = n.id in
  let shape = n.shape in
  let shape_logic = Shape.Broadcast (compose_op, m1.shape, m2.shape) in
  let local_shape_update = Shape.{ shape; logic=shape_logic } in
  Shape.(
    propagate_shapes local_shape_update;
    match m1.shape, m2.shape with
    | {batch=Given _; input=Given _; output=Given _; _},
      {batch=Given _; input=Given _; output=Given _; _} ->
      set_dims_type shape given
    | _ -> ()
  );
  session_shape_updates := local_shape_update :: !session_shape_updates;
  let n1 = m1.node in
  let n2 = m2.node  in
  let projections() = Shape.derive_projections local_shape_update in
  let op_body = op_body ~n ~n1 ~n2 projections in
  (* The code needs to be included in the order it was computed! *)
  let open Code in
  let forward_body =
    (match m1_processed, m1.forward_body, m2_processed, m2.forward_body with
    | true, _, true, _ | true, _, _, Noop | _, Noop, true, _ | _, Noop, _, Noop -> op_body
    | false, m1_body, false, m2_body when m1_first -> Seq (ParHint (m1_body, m2_body), op_body)
    | false, m1_body, false, m2_body -> Seq (ParHint (m2_body, m1_body), op_body)
    | _, _, false, m2_body -> Seq (m2_body, op_body)
    | false, m1_body, _, _ -> Seq(m1_body, op_body)) in
  let init_values_body = create ~id `Value shape in
  session_initializations := init_values_body :: !session_initializations;
  if not is_form then
    {forward_body; form=None; id; node=n; shape_logic; shape}
  else
    let form1, form2 = match m1.form, m2.form with
    | Some form1, Some form2 -> form1, form2
    | None, _ ->
      raise @@ Session_error ("binop ~is_form:true but subformula is non-form", Some m1)
    | _, None ->
      raise @@ Session_error ("binop ~is_form:true but subformula is non-form", Some m2) in
    let needs_gradient = form1.needs_gradient || form2.needs_gradient in
    let m1_no_grad = m1_processed || not form1.needs_gradient in
    let m2_no_grad = m2_processed || not form2.needs_gradient in
    if needs_gradient then
      session_prepare_step := reset_zeros ~id `Grad shape :: !session_prepare_step;
    (* The code needs to be included in the reverse order to which it was computed! This guarantees
       that all ancestors of a node are backpropagated before the node is backpropagated, even for
       non-tree DAGs. *)
    let grad_body = if needs_gradient then grad_body ~n ~n1 ~n2 projections else Noop in
    let grad_body =
      if form1.needs_gradient then grad_body
      else Code.remove_updates {id=m1.id; field=`Grad} grad_body in
    let grad_body =
      if form2.needs_gradient then grad_body
      else Code.remove_updates {id=m2.id; field=`Grad} grad_body in
    let backprop_body =
      match m1_no_grad, form1.backprop_body, m2_no_grad, form2.backprop_body with
      | true, _, true, _ | true, _, _, Noop | _, Noop, true, _ | _, Noop, _, Noop -> grad_body
      | false, m1_body, false, m2_body when m1_first -> Seq (grad_body, ParHint(m2_body, m1_body))
      | false, m1_body, false, m2_body -> Seq (grad_body, ParHint(m1_body, m2_body))
      | _, _, false, m2_body -> Seq (grad_body, m2_body)
      | false, m1_body, _, _ -> Seq (grad_body, m1_body) in
    if needs_gradient then
      session_initializations := create ~id `Grad shape :: !session_initializations;    
    (* The order is not relevant, we keep the same order as in backprop for readability. *)
    (if not m1_processed then global_roots := Map.remove !global_roots m1.id);
    (if not m2_processed then global_roots := Map.remove !global_roots m2.id);
    let form = Some {backprop_body; needs_gradient} in
    let formula = {forward_body; form; id; node=n; shape_logic; shape} in
    let root = {forward_code=None; backprop_code=None; formula} in
    global_roots := Map.add_exn !global_roots ~key:id ~data:root;
    formula

let unop ~op_label ?init_shape ~transpose_op ~op_body ~grad_body ~is_form m1: t =
  (* Note: do not capture m in any closure, so it can be GC'd. *)
  let m1_processed = Option.is_some m1.form && not @@ Map.mem !global_roots m1.id in
  let children = [{NodeUI.sub_node_id=m1.id; computed_externally=m1_processed}] in
  let n = NodeUI.create_of_same_precision_as ~is_form m1.node.node
      ~op_label ~shape_spec:Unknown_shape ~children in
  let id = n.id in

  let shape =
    match init_shape with
    | None ->
      { Shape.batch=Unknown; input=Unknown; output=Unknown;
        axis_labels=Map.empty (module Shape.AxisKey);
        deduce_within_shape_constraints=Not_constrained; id }
    | Some shape -> shape in
  let shape_logic = Shape.Transpose(transpose_op, m1.shape) in
  let local_shape_update = Shape.{ shape; logic=shape_logic } in
  Shape.(
    propagate_shapes local_shape_update;
    if Option.is_none init_shape then match m1.shape with
    | {batch=Given _; input=Given _; output=Given _; _} ->
      set_dims_type shape given
    | _ -> ()
  );
  session_shape_updates := local_shape_update :: !session_shape_updates;
  let n1 = m1.node in
  let projections() = Shape.derive_projections local_shape_update in
  let op_body = op_body ~n ~n1 projections in
  (* The code needs to be included in the order it was computed! *)
  let open Code in
  let forward_body =
    match m1_processed, m1.forward_body with
    | true, _ | _, Noop -> op_body
    | false, m_body -> Seq (m_body, op_body) in
  session_initializations := create ~id `Value shape :: !session_initializations;
  if not is_form then
    {forward_body; form=None; id; node=n; shape_logic; shape}
  else
    let form1 = match m1.form with
    | Some form -> form
    | None ->
      raise @@ Session_error ("binop ~is_form:true but subformula is non-form", Some m1) in
    let needs_gradient = form1.needs_gradient in
    let m1_no_grad = m1_processed || not form1.needs_gradient in
    if needs_gradient then
      session_prepare_step := reset_zeros ~id `Grad shape :: !session_prepare_step;
    let grad_body =
      if needs_gradient then grad_body ~n ~n1 projections else Noop in
    let grad_body =
      if form1.needs_gradient then grad_body
      else Code.remove_updates {id=m1.id; field=`Grad} grad_body in
    (* The code needs to be included in the reverse order to which it was computed! *)
    let backprop_body =
      match m1_no_grad, form1.backprop_body with
      | true, _ | _, Noop -> grad_body
      | false, m1_body -> Seq (grad_body, m1_body) in
    if needs_gradient then
      session_initializations := create ~id `Grad shape :: !session_initializations;    

    (if not m1_processed then global_roots := Map.remove !global_roots m1.id);
    let form = Some {backprop_body; needs_gradient} in
    let formula = {forward_body; form; id; node=n; shape_logic; shape} in
    let root = {forward_code=None; backprop_code=None; formula} in
    global_roots := Map.add_exn !global_roots ~key:id ~data:root;
    formula

(** The default [needs_gradient] behavior. *)
let term_needs_gradient (spec: Shape.term_spec) =
  match spec with
  | Unknown_shape -> true
  | Data _ -> false
  | Constant _ -> false
  | Params _ -> true
  | Transform _ -> false
  | Unknown_batch_data _ -> false
  | Deduced_params _ -> true

(** A terminal: a constant, a parameter, an input of the model. *)
let term ~label ?needs_gradient ~is_form (spec: Shape.term_spec) ~init_op =
  let n = NodeUI.create ~value_prec:Single ~grad_prec:Single ~is_form ()
      ~op_label:label ~shape_spec:spec ~children:[] in
  let id = n.id in
  let shape = n.shape in
  let shape_logic = Shape.Terminal in
  (* NOTE: this update does not do any work, but that might change in the future,
     and having it in the update sequence might help with debuggability. *)
  let local_shape_update = Shape.{ shape; logic=shape_logic } in
  Shape.propagate_shapes local_shape_update;
  session_shape_updates := local_shape_update :: !session_shape_updates;

  let open Code in
  let forward_body = Noop in
  session_initializations := create ~id ~init_op `Value shape :: !session_initializations;
  if not is_form then
    {forward_body; form=None; id; node=n; shape_logic; shape}
  else
    let needs_gradient =
      match needs_gradient with
      | None -> term_needs_gradient spec
      | Some setting -> setting in
    if needs_gradient then
      session_prepare_step := reset_zeros ~id `Grad shape :: !session_prepare_step;
    let backprop_body = Noop in
    (* Very unlikely someone will want dw/dw. *)
    if needs_gradient then
      session_initializations := create ~id `Grad shape :: !session_initializations;    
    let form = Some {backprop_body; needs_gradient} in
    let formula = {forward_body; form; id; node=n; shape_logic; shape} in
    let root = {forward_code=None; backprop_code=None; formula} in
    global_roots := Map.add_exn !global_roots ~key:id ~data:root;
    formula

let error_if_unknown_shape m =
  match m.shape with
  | {input=Unknown; _} -> raise @@ Session_error ("Shape of inputs is still unknown", Some m)
  | {output=Unknown; _} -> raise @@ Session_error ("Shape of outputs is still unknown", Some m)
  | {batch=Unknown; _} -> raise @@ Session_error ("Shape of batching is still unknown", Some m)
  | {output=Inferred []; _} ->
     raise @@ Session_error ("Shape of outputs is still empty -- missing shape information", Some m)
  | {input=_; output=_; batch=_; axis_labels=_; deduce_within_shape_constraints=_; id=_} -> ()

let get_toplevel m =
  error_if_unknown_shape m;
  let open Code in
  let toplevel_forward =
    Node_specific {procedure=m.forward_body; routine={id=m.id; field=`Forward};
                   label="Forward #"^Int.to_string m.id} in
  let backprop =
    Seq (reset_ones ~id:m.id `Grad m.shape,
         (Option.value_exn m.form).backprop_body) in
  let toplevel_backprop =
    Node_specific {procedure=backprop;
                   routine={id=m.id; field=`Backprop};
                   label="Backprop #"^Int.to_string m.id} in
  toplevel_forward, toplevel_backprop

(* FIXME: not inlining here gives an error about PrintBox.Simple.t_of_sexp missing *)
type printbox = (* PrintBox.Simple.t *)
    [ `Empty
    | `Hlist of printbox list
    | `Pad of printbox
    | `Table of printbox array array
    | `Text of string
    | `Tree of printbox * printbox list
    | `Vlist of printbox list ] [@@deriving sexp, compare]

let sexp_of_t m =
  (* TODO: output more *)
  Sexp.message "Formula" [
    "id", Int.sexp_of_t m.id;
    "op_label", String.sexp_of_t m.node.op_label;
  ]

include Comparator.Make(struct
    type nonrec t = t
    let compare m1 m2 = Int.compare m1.id m2.id
    let sexp_of_t = sexp_of_t
end)

let float_to_label v = Float.to_string_hum ~strip_zero:true v

let number ~is_form ?(axis_label="") c =
  (* Note: no axis label so that we do not conflict with user labels. *)
  term ~label:(float_to_label c) ~is_form
    (Constant {output_dims=[1]; axis_labels=axis_label}) ~init_op:(`Constant_of_value c)

let ndarray ~is_form ?(axis_labels="") ?label ?(batch_dims=[]) ?(input_dims=[]) ?(output_dims=[])
 values =
  let spec =
    match label, batch_dims, input_dims with
    | Some _, [], _ -> Shape.Params {input_dims; output_dims; axis_labels}
    | None, [], [] -> Constant {output_dims; axis_labels}
    | None, _, _ -> Transform {batch_dims; input_dims; output_dims; axis_labels}
    | _, _, [] -> Data {batch_dims; output_dims; axis_labels}
    | _, _::_, _::_ ->
      let sh = {Shape.batch=Given batch_dims; input=Given input_dims; output=Given output_dims;
                deduce_within_shape_constraints=Not_constrained;
                axis_labels=(Shape.axis_labels_of_spec axis_labels).labels; id= -1} in
      raise @@
      Shape.Shape_error ("Operation.ndarray: cannot provide all of [label], [batch_dims] and [input_dims]",
                         sh, sh) in
  let label =
    match label with
    | Some label -> label
    | None ->
      Caml.Format.pp_set_geometry Caml.Format.str_formatter
        ~max_indent:(!max_sublabel_length) ~margin:(!max_sublabel_length*2);
      let (!) = Array.of_list in
      let dims = Array.concat [!batch_dims; !output_dims; !input_dims] in
      let ndarr = Ocannl_runtime.Node.create_ndarray Single dims (`Fixed_constant values) in
      let (!) = List.length in
      NodeUI.pp_tensor_inline ~num_batch_axes: !batch_dims ~num_output_axes: !output_dims
        ~num_input_axes: !input_dims Caml.Format.str_formatter ndarr;
      Caml.Format.flush_str_formatter() in
  let label =
    if String.contains label '\n' then
      "c"^(NodeUI.dims_to_string @@ Array.concat_map [|batch_dims; output_dims; input_dims|] ~f:Array.of_list)
    else label in
  term ~is_form ~label spec ~init_op:(`Fixed_constant values)

module FDSL = struct
  let term = term ~is_form:true
  let number = number ~is_form:true
  let ndarray = ndarray ~is_form:true
end

module NFDSL = struct
  let term = term ~is_form:false
  let number = number ~is_form:false
  let ndarray = ndarray ~is_form:false
end

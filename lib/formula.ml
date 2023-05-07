(** Construction of runtime-compiled code supporting backpropagation. *)

open Base

type form = {
  backprop_body : Code.t;
      (** Performs backpropagation for the formula at each session step, which typically means adding
      partial gradients to the gradient tensor of the subformulas. *)
  needs_gradient : bool;
      (** An optimization setting: whether gradients should be backpropagated into the formula.
      If any subformula needs gradients, this formula also needs gradients. *)
}
[@@deriving sexp]

type t = {
  forward_body : Code.t;  (** Computes the values at each session step. *)
  form : form option;
  id : int;
  node : NodeUI.t;  (** Tracks the computation node. *)
  shape_logic : Shape.logic;
      (** How to do the last update of [t.shape] when finalizing the formula.
      It is stored with the formula for debugging (shape inference does not need to retrieve it). *)
  shape : Shape.t;
      (** The eventual shape of [.!(t.node).value] and [.!(t.node).grad], incorporating the current state of
      shape inference. *)
  cross_session_persistent : bool;
      (** A subformula is cross-session persistent if [forward_body] is [Noop], and the formula does
      not require data fetching. *)
}
[@@deriving sexp_of]
(** Information needed for compositional code generation. The code generation is suspended so that
    it can incorporate inferred shape information. *)

(** A global root is a formula that is not (currently) a subformula of another formula.

    If a formula with [id >= !first_session_id] is not among global roots, it must be a subformula
    of a global root. *)
let global_roots = ref @@ Map.empty (module Int)

(** We perform each update (at least) twice to propagate information between all subformulas:
    first in postfix order while computing [t], then in prefix order by iterating over this stack. *)
let session_shape_updates : Shape.update_step list ref = ref []

(** This code will usually be executed only once, after the shapes are inferred. But it will also
    be executed by each [Session.refresh_session ~regenerate:true] and 
    [Session.refresh_session ~reinit:true] call, except if [~force_no_init:true].
    Execution potentially in parallel. *)
let session_initializations : Code.create list ref = ref []

let session_initialized = ref 0

(** This code will be executed on each [Session.refresh_session ~run:true] call ([~run:true]
    is implicit), before any [forward] or [backprop] code. Execution potentially in parallel. *)
let session_prepare_forward : Code.t list ref = ref []

let session_prepare_backprop : Code.t list ref = ref []

(** A current session is the range of nodes from [!first_session_id] to [Node.global.unique_id - 1],
    or an empty range if [!first_session_id = Node.global.unique_id].
    Subformulas with [id] before this range are no longer updated by {!Session.SDSL.refresh_session}
    and can only be used in new formulas if they are cross-session-persistent: not depending on
    fetching operations. This condition is checked automatically. *)
let first_session_id = ref 1

let default_value_prec = ref NodeUI.single
let default_grad_prec = ref NodeUI.single

exception Session_error of string * t option [@@deriving sexp]

(** Prefix the input with the header information of all nodes within the current session. *)
let prefix_with_preamble content =
  let open Ocannl_runtime in
  let result = Buffer.create 16 in
  let ap = Buffer.add_string result in
  for id = !first_session_id to Node.global.unique_id - 1 do
    let n = NodeUI.get id in
    ap "Node ";
    ap @@ NodeUI.node_header n;
    if n.virtual_ then ap " (virtual)";
    ap ";\n"
  done;
  ap content;
  Buffer.contents result

let session_error_printer = function
  | Session_error (msg, None) -> Some (prefix_with_preamble msg)
  | Session_error (msg, Some m) -> Some (prefix_with_preamble @@ "For #" ^ Int.to_string_hum m.id ^ ": " ^ msg)
  | _ -> None

let () = Caml.Printexc.register_printer session_error_printer

let handle_error ?formula message =
  let exc = Session_error (message, formula) in
  Stdio.prerr_endline @@ Option.value_exn (session_error_printer exc);
  raise exc

let fetch_zeros ~id field _shape = Code.Fetch { tensor = { id; field }; fetch_op = Constant 0. }
let fetch_ones ~id field _shape = Code.Fetch { tensor = { id; field }; fetch_op = Constant 1. }

let create ~id ?(init_op = Code.Constant_fill [| 0.0 |]) field shape =
  { Code.tensor = { id; field }; dims = (fun () -> Shape.to_dims shape); init_op }

let max_sublabel_length = ref 25

let raw_binop ~zero_out ~accum ~lhs_id ~lhs_is_grad ~op ~rhs1_id ~rhs1_is_grad ~rhs2_id ~rhs2_is_grad ~logic =
  let n = NodeUI.get lhs_id in
  let n1 = NodeUI.get rhs1_id in
  let n2 = NodeUI.get rhs2_id in
  let shape = n.shape in
  let shape_logic = Shape.Broadcast (logic, n1.shape, n2.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic } in
  Shape.propagate_shapes local_shape_update;
  session_shape_updates := local_shape_update :: !session_shape_updates;
  let projections () = Shape.derive_projections local_shape_update in
  let lhs = Code.CDSL.(if lhs_is_grad then grad_of_id lhs_id else value_of_id lhs_id) in
  let rhs1 = Code.CDSL.(if rhs1_is_grad then grad_of_id rhs1_id else value_of_id rhs1_id) in
  let rhs2 = Code.CDSL.(if rhs2_is_grad then grad_of_id rhs2_id else value_of_id rhs2_id) in
  Code.Accum_binop { zero_out; accum; lhs; op; rhs1; rhs2; projections }

let raw_unop ~zero_out ~accum ~lhs_id ~lhs_is_grad ~op ~rhs_id ~rhs_is_grad ~logic =
  let n = NodeUI.get lhs_id in
  let n1 = NodeUI.get rhs_id in
  let shape = n.shape in
  let shape_logic = Shape.Transpose (logic, n1.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic } in
  Shape.propagate_shapes local_shape_update;
  session_shape_updates := local_shape_update :: !session_shape_updates;
  let projections () = Shape.derive_projections local_shape_update in
  let lhs = Code.CDSL.(if lhs_is_grad then grad_of_id lhs_id else value_of_id lhs_id) in
  let rhs = Code.CDSL.(if rhs_is_grad then grad_of_id rhs_id else value_of_id rhs_id) in
  Code.Accum_unop { zero_out; accum; lhs; op; rhs; projections }

let binop ~op_label ?desc_label ?(compose_op = Shape.Pointwise_bin) ~op_body ~grad_body ~is_form m1 m2 =
  (* Note: do not capture m1, m2 in any closure, so they can be GC'd. *)
  if m1.id < !first_session_id && not m1.cross_session_persistent then
    raise @@ Session_error ("The subformula is outside of current session", Some m1);
  if m2.id < !first_session_id && not m2.cross_session_persistent then
    raise @@ Session_error ("The subformula is outside of current session", Some m2);
  let m1_first = m1.id <= m2.id in
  let m1_processed : bool = Option.is_some m1.form && (not @@ Map.mem !global_roots m1.id) in
  let m2_processed : bool =
    Option.is_some m2.form && (m2.id = m1.id || (not @@ Map.mem !global_roots m2.id))
  in
  let children =
    [
      { NodeUI.sub_node_id = m1.id; computed_externally = m1_processed };
      { sub_node_id = m2.id; computed_externally = m2_processed };
    ]
  in
  let needs_gradient =
    match (m1.form, m2.form) with
    | Some form1, Some form2 -> form1.needs_gradient || form2.needs_gradient
    | Some form1, _ -> form1.needs_gradient
    | _, Some form2 -> form2.needs_gradient
    | _ -> false
  in
  let n =
    NodeUI.create_of_promoted_precision ~needs_gradient m1.node.node m2.node.node ~op_label ?desc_label
      ~children ()
  in
  let id = n.id in
  let shape = n.shape in
  let shape_logic = Shape.Broadcast (compose_op, m1.shape, m2.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic } in
  let n1 = m1.node in
  let n2 = m2.node in
  Shape.propagate_shapes local_shape_update;
  session_shape_updates := local_shape_update :: !session_shape_updates;
  let projections () = Shape.derive_projections local_shape_update in
  let op_body = op_body ~n ~n1 ~n2 ~projections in
  (* The code needs to be included in the order it was computed! *)
  let forward_body =
    Code.(
      match (m1_processed, m1.forward_body, m2_processed, m2.forward_body) with
      | true, _, true, _ | true, _, _, Noop | _, Noop, true, _ | _, Noop, _, Noop -> op_body
      | false, m1_body, false, m2_body when m1_first -> Seq (ParHint (m1_body, m2_body), op_body)
      | false, m1_body, false, m2_body -> Seq (ParHint (m2_body, m1_body), op_body)
      | _, _, false, m2_body -> Seq (m2_body, op_body)
      | false, m1_body, _, _ -> Seq (m1_body, op_body))
  in
  let init_values_body = create ~id Value shape in
  session_initializations := init_values_body :: !session_initializations;
  if not is_form then
    { forward_body; form = None; id; node = n; shape_logic; shape; cross_session_persistent = false }
  else
    let form1, form2 =
      match (m1.form, m2.form) with
      | Some form1, Some form2 -> (form1, form2)
      | None, _ -> raise @@ Session_error ("binop ~is_form:true but subformula is non-form", Some m1)
      | _, None -> raise @@ Session_error ("binop ~is_form:true but subformula is non-form", Some m2)
    in
    let m1_no_grad = m1_processed || not form1.needs_gradient in
    let m2_no_grad = m2_processed || not form2.needs_gradient in
    (if needs_gradient then
       let zero_grads = fetch_zeros ~id Grad shape in
       session_prepare_backprop := zero_grads :: !session_prepare_backprop);
    (* The code needs to be included in the reverse order to which it was computed! This guarantees
       that all ancestors of a node are backpropagated before the node is backpropagated, even for
       non-tree DAGs. *)
    let grad_body = if needs_gradient then grad_body ~n ~n1 ~n2 ~projections else Code.Noop in
    let grad_body =
      if form1.needs_gradient then grad_body else Code.remove_updates { id = m1.id; field = Grad } grad_body
    in
    let grad_body =
      if form2.needs_gradient then grad_body else Code.remove_updates { id = m2.id; field = Grad } grad_body
    in
    let backprop_body =
      match (m1_no_grad, form1.backprop_body, m2_no_grad, form2.backprop_body) with
      | true, _, true, _ | true, _, _, Noop | _, Noop, true, _ | _, Noop, _, Noop -> grad_body
      | false, m1_body, false, m2_body when m1_first -> Seq (grad_body, ParHint (m2_body, m1_body))
      | false, m1_body, false, m2_body -> Seq (grad_body, ParHint (m1_body, m2_body))
      | _, _, false, m2_body -> Seq (grad_body, m2_body)
      | false, m1_body, _, _ -> Seq (grad_body, m1_body)
    in
    if needs_gradient then session_initializations := create ~id Grad shape :: !session_initializations;
    (* The order is not relevant, we keep the same order as in backprop for readability. *)
    if not m1_processed then global_roots := Map.remove !global_roots m1.id;
    if not m2_processed then global_roots := Map.remove !global_roots m2.id;
    let form = Some { backprop_body; needs_gradient } in
    let formula =
      { forward_body; form; id; node = n; shape_logic; shape; cross_session_persistent = false }
    in
    global_roots := Map.add_exn !global_roots ~key:id ~data:formula;
    formula

let unop ~op_label ?desc_label ?init_shape ~transpose_op ~op_body ~grad_body ~is_form m1 : t =
  (* Note: do not capture m in any closure, so it can be GC'd. *)
  let m1_processed = Option.is_some m1.form && (not @@ Map.mem !global_roots m1.id) in
  let children = [ { NodeUI.sub_node_id = m1.id; computed_externally = m1_processed } ] in
  let needs_gradient = match m1.form with Some form1 -> form1.needs_gradient | None -> false in
  let n =
    NodeUI.create_of_same_precision_as ~needs_gradient m1.node.node ~op_label ?desc_label ~children ()
  in
  let id = n.id in
  let shape = n.shape in
  (match init_shape with
  | None -> ()
  | Some init ->
      let open Shape in
      shape.batch <- init.batch;
      shape.input <- init.input;
      shape.output <- init.output;
      shape.axis_labels <- init.axis_labels);
  let shape_logic = Shape.Transpose (transpose_op, m1.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic } in
  let n1 = m1.node in
  Shape.propagate_shapes local_shape_update;
  session_shape_updates := local_shape_update :: !session_shape_updates;
  let projections () = Shape.derive_projections local_shape_update in
  let op_body = op_body ~n ~n1 ~projections in
  (* The code needs to be included in the order it was computed! *)
  let forward_body =
    Code.(
      match (m1_processed, m1.forward_body) with
      | true, _ | _, Noop -> op_body
      | false, m_body -> Seq (m_body, op_body))
  in
  session_initializations := create ~id Value shape :: !session_initializations;
  if not is_form then
    { forward_body; form = None; id; node = n; shape_logic; shape; cross_session_persistent = false }
  else
    let form1 =
      match m1.form with
      | Some form -> form
      | None -> raise @@ Session_error ("binop ~is_form:true but subformula is non-form", Some m1)
    in
    let m1_no_grad = m1_processed || not form1.needs_gradient in
    (if needs_gradient then
       let zero_grads = fetch_zeros ~id Grad shape in
       session_prepare_backprop := zero_grads :: !session_prepare_backprop);
    let grad_body = if needs_gradient then grad_body ~n ~n1 ~projections else Code.Noop in
    let grad_body =
      if form1.needs_gradient then grad_body else Code.remove_updates { id = m1.id; field = Grad } grad_body
    in
    (* The code needs to be included in the reverse order to which it was computed! *)
    let backprop_body =
      match (m1_no_grad, form1.backprop_body) with
      | true, _ | _, Noop -> grad_body
      | false, m1_body -> Seq (grad_body, m1_body)
    in
    if needs_gradient then session_initializations := create ~id Grad shape :: !session_initializations;

    if not m1_processed then global_roots := Map.remove !global_roots m1.id;
    let form = Some { backprop_body; needs_gradient } in
    let formula =
      { forward_body; form; id; node = n; shape_logic; shape; cross_session_persistent = false }
    in
    global_roots := Map.add_exn !global_roots ~key:id ~data:formula;
    formula

(** A terminal: a constant, a parameter, an input of the model. *)
let term ~label ?desc_label ~needs_gradient ~is_form ?batch_dims ?input_dims ?output_dims ?axis_labels
    ?deduced ?init_op ?fetch_op () =
  if needs_gradient && not is_form then
    raise @@ Session_error ("Formula.term ~needs_gradient:true: a non-form formula cannot need gradient", None);
  let literal : bool =
    if needs_gradient then false
    else match (init_op, fetch_op) with Some (Code.Constant_fill [| _ |]), None -> true | _ -> false
  in
  let op_label : string = label in
  let n : NodeUI.t =
    NodeUI.create ~value_prec:!default_value_prec ~grad_prec:!default_grad_prec ~literal ~needs_gradient ()
      ~op_label ?desc_label ?batch_dims ?input_dims ?output_dims ?axis_labels ?deduced ~children:[] ()
  in
  let id = n.id in
  let shape = n.shape in
  let shape_logic = Shape.Terminal in
  (* NOTE: this update does not do any work, but that might change in the future,
     and having it in the update sequence might help with debuggability. *)
  let local_shape_update = Shape.{ shape; logic = shape_logic } in
  Shape.propagate_shapes local_shape_update;
  session_shape_updates := local_shape_update :: !session_shape_updates;

  let forward_body = Code.Noop in
  (* Note: we could embed the fetching code in the forward computation instead, but then we miss out
      on potential optimizations. E.g. fetching latency means it's important to do it early and
     in parallel. *)
  let init_op : Code.init_op =
    Option.value_or_thunk init_op ~default:(fun () -> Code.Constant_fill [| 0.0 |])
  in
  session_initializations := create ~id ~init_op Value shape :: !session_initializations;
  (if literal && Code.virtualize_settings.inline_constants then
     let fetch_op : Code.fetch_op =
       match init_op with Constant_fill [| c |] -> Constant c | _ -> assert false
     in
     let fetch = Code.Fetch { tensor = { id; field = Value }; fetch_op } in
     session_prepare_forward := fetch :: !session_prepare_forward);
  let cross_session_persistent =
    Code.(
      match fetch_op with
      | None -> true
      | Some fetch_op ->
          let fetch_op = fetch_op ~n in
          let fetch = Fetch { tensor = { id; field = Value }; fetch_op } in
          session_prepare_forward := fetch :: !session_prepare_forward;
          (match fetch_op with Constant _ -> () | _ -> n.cannot_be_virtual <- true);
          false)
  in
  if not is_form then
    { forward_body; form = None; id; node = n; shape_logic; shape; cross_session_persistent }
  else
    let backprop_body = Code.Noop in
    (if needs_gradient then
       let zero_grads = fetch_zeros ~id Grad shape in
       session_prepare_backprop := zero_grads :: !session_prepare_backprop);
    (* Very unlikely someone will want dw/dw. *)
    if needs_gradient then session_initializations := create ~id Grad shape :: !session_initializations;
    let form = Some { backprop_body; needs_gradient } in
    let formula = { forward_body; form; id; node = n; shape_logic; shape; cross_session_persistent } in
    global_roots := Map.add_exn !global_roots ~key:id ~data:formula;
    formula

let error_if_unknown_shape m =
  match m.shape with
  | { input = Unknown; _ } -> raise @@ Session_error ("Shape of inputs is still unknown", Some m)
  | { output = Unknown; _ } -> raise @@ Session_error ("Shape of outputs is still unknown", Some m)
  | { batch = Unknown; _ } -> raise @@ Session_error ("Shape of batching is still unknown", Some m)
  | { output = Inferred []; _ } ->
      raise @@ Session_error ("Shape of outputs is still empty -- missing shape information", Some m)
  | { input = _; output = _; batch = _; axis_labels = _; deduce_within_shape_constraints = _; id = _ } -> ()

let get_toplevel_forward m =
  error_if_unknown_shape m;
  Code.Block_comment ("Forward #" ^ Int.to_string m.id, m.forward_body)

let get_toplevel_backprop m =
  error_if_unknown_shape m;
  Code.Block_comment
    ( "Backprop #" ^ Int.to_string m.id,
      Seq (fetch_ones ~id:m.id Grad m.shape, (Option.value_exn m.form).backprop_body) )

(* FIXME: not inlining here gives an error about PrintBox.Simple.t_of_sexp missing *)
type printbox =
  (* PrintBox.Simple.t *)
  [ `Empty
  | `Hlist of printbox list
  | `Pad of printbox
  | `Table of printbox array array
  | `Text of string
  | `Tree of printbox * printbox list
  | `Vlist of printbox list ]
[@@deriving sexp, compare]

let sexp_of_t m =
  (* TODO: output more *)
  Sexp.message "Formula"
    [
      ("id", Int.sexp_of_t m.id);
      ("op_label", String.sexp_of_t m.node.op_label);
      ("desc_label", Option.sexp_of_t String.sexp_of_t m.node.desc_label);
    ]

include Comparator.Make (struct
  type nonrec t = t

  let compare m1 m2 = Int.compare m1.id m2.id
  let sexp_of_t = sexp_of_t
end)

let float_to_label v = Float.to_string_hum ~strip_zero:true v

let number ?desc_label ~is_form ?(axis_label = "") c =
  (* Note: no axis label so that we do not conflict with user labels. *)
  term ?desc_label ~label:(float_to_label c) ~is_form ~needs_gradient:false ~batch_dims:[] ~input_dims:[]
    ~output_dims:[ 1 ] ~axis_labels:axis_label ~init_op:(Constant_fill [| c |]) ()

let ndarray ?desc_label ~is_form ?(needs_gradient = false) ?(batch_dims = []) ?(input_dims = [])
    ?(output_dims = []) ?axis_labels ?label values =
  let label =
    match label with
    | Some label -> label
    | None ->
        Caml.Format.pp_set_geometry Caml.Format.str_formatter ~max_indent:!max_sublabel_length
          ~margin:(!max_sublabel_length * 2);
        let ( ! ) = Array.of_list in
        let dims = Array.concat [ !batch_dims; !output_dims; !input_dims ] in
        let ndarr = Ocannl_runtime.Node.create_ndarray Double dims (Constant_fill values) in
        let ( ! ) = List.length in
        NodeUI.pp_tensor_inline ~num_batch_axes:!batch_dims ~num_output_axes:!output_dims
          ~num_input_axes:!input_dims Caml.Format.str_formatter ndarr;
        Caml.Format.flush_str_formatter ()
  in
  let label =
    if String.contains label '\n' then
      "c" ^ NodeUI.dims_to_string
      @@ Array.concat_map [| batch_dims; output_dims; input_dims |] ~f:Array.of_list
    else label
  in
  term ?desc_label ~needs_gradient ~is_form ~batch_dims ~input_dims ~output_dims ?axis_labels
    ~deduced:Not_constrained ~label ~init_op:(Constant_fill values) ()

let params ?desc_label ?axis_labels ?input_dims ?output_dims ?deduced ?values ?value label =
  let init_op =
    match (values, value) with
    | Some _, Some _ -> invalid_arg "Formula.params: do not provide both ~values and ~value"
    | Some values, _ -> Code.Constant_fill values
    | _, Some value -> Code.Constant_fill [| value |]
    | None, None -> Standard_uniform
  in
  term ?desc_label ~needs_gradient:true ~is_form:true ~batch_dims:[] ?input_dims ?output_dims ?axis_labels
    ?deduced ~label ~init_op ()

module FDSL = struct
  let term = term ~is_form:true
  let number = number ~is_form:true
  let ndarray = ndarray ~is_form:true
  let params = params
end

module NFDSL = struct
  let term = term ~is_form:false
  let number = number ~is_form:false
  let ndarray = ndarray ~is_form:false
end

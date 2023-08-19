(** Construction of runtime-compiled code supporting backpropagation. *)

open Base
open Arrayjit

module LA = Low_level.Lazy_array

type diff = {
  grad : LA.t;
  zero_grads : High_level.t;
      (** Prepares for backpropagation. Always compile as: [backprop = Seq (zero_grads, backprop_body)]. *)
  backprop_body : High_level.t;
      (** Performs backpropagation for the tensor at each session step, which typically means adding
           partial gradients to the gradient tensor of the subtensors. *)
  needs_gradient : bool;
      (** An optimization setting: whether gradients should be backpropagated into the tensor.
          If any subtensor needs gradients, this tensor also needs gradients. *)
}
[@@deriving sexp_of]

type t = {
  forward_body : High_level.t;  (** Computes the values at each session step. *)
  diff : diff option;
  nondiff_forward_body : High_level.t;
      (** Same as [forward_body] if [diff] is [None], otherwise [Code.Noop]. *)
  id : int;  (** Same as [value.id]. *)
  value : LA.t;
  shape_logic : Shape.logic;
      (** How to do the last update of [t.shape] when finalizing the tensor.
          It is stored with the tensor for debugging (shape inference does not need to retrieve it). *)
  shape : Shape.t;
      (** The eventual shape of [.!(t.node).value] and [.!(t.node).grad], incorporating the current state of
          shape inference. *)
  cross_session_persistent : bool;
      (** A subtensor is cross-session persistent if [forward_body] is [Noop], and the tensor does
          not require data fetching. *)
}
[@@deriving sexp_of]
(** Information needed for compositional code generation. The code generation is suspended so that
    it can incorporate inferred shape information. *)

(** A global root is a tensor that is not (currently) a subtensor of another tensor.

    If a tensor with [id >= session_state.first_session_id] is not among global roots, it must be a subtensor
    of a global root. *)
let global_roots = ref @@ Map.empty (module Int)

(** We perform each update (at least) twice to propagate information between all subtensors:
    first in postfix order while computing [t], then in prefix order by iterating over this stack. *)
let session_shape_updates : Shape.update_step list ref = ref []

(** This code will usually be executed once, after the shapes are inferred. But it will also
    be executed by each [Session.refresh_session ~regenerate:true] and 
    [Session.refresh_session ~reinit:true] call, except if [~force_no_init:true].
    Execution potentially in parallel. *)
let session_initializations : Low_level.create list ref = ref []

let session_initialized = ref 0

(** This code will be executed on each [Session.refresh_session ~run:true] call ([~run:true]
    is implicit), after a [forward] and [backprop] step. Execution potentially in parallel. *)
let session_postprocess : High_level.t list ref = ref []

let default_value_prec = ref Ndarray.single
let default_grad_prec = ref Ndarray.single

exception Session_error of string * t option [@@deriving sexp]

let session_error_printer = function
  | Session_error (msg, None) -> Some msg
  | Session_error (msg, Some m) -> Some ("For #" ^ Int.to_string_hum m.id ^ ": " ^ msg)
  | _ -> None

let () = Caml.Printexc.register_printer session_error_printer

let fetch_zeros array shape =
  High_level.Fetch { array; fetch_op = Constant 0.; dims = (fun () -> Shape.to_dims shape) }

let fetch_ones array shape =
  High_level.Fetch { array; fetch_op = Constant 1.; dims = (fun () -> Shape.to_dims shape) }

let create ?(init_op = Low_level.Constant_fill [| 0.0 |]) array shape =
  { array; Low_level.dims = (fun () -> Shape.to_dims shape); init_op }

let max_sublabel_length = ref 25

let raw_binop ~zero_out ~accum ~m ~lhs_is_grad ~op ~m1 ~rhs1_is_grad ~m2 ~rhs2_is_grad ~logic =
  let shape = m.shape in
  let shape_logic = Shape.Broadcast (logic, m1.shape, m2.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic } in
  Shape.propagate_shapes local_shape_update;
  session_shape_updates := local_shape_update :: !session_shape_updates;
  let projections () = Shape.derive_projections local_shape_update in
  let lhs = if lhs_is_grad then m.value else (Option.value_exn m.diff).grad in
  let rhs1 = if rhs1_is_grad then m1.value else (Option.value_exn m1.diff).grad in
  let rhs2 = if rhs2_is_grad then m2.value else (Option.value_exn m2.diff).grad in
  High_level.Accum_binop { zero_out; accum; lhs; op; rhs1; rhs2; projections }

let raw_unop ~zero_out ~accum ~m ~lhs_is_grad ~op ~m1 ~rhs_is_grad ~logic =
  let shape = m.shape in
  let shape_logic = Shape.Transpose (logic, m1.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic } in
  Shape.propagate_shapes local_shape_update;
  session_shape_updates := local_shape_update :: !session_shape_updates;
  let projections () = Shape.derive_projections local_shape_update in
  let lhs = if lhs_is_grad then m.value else (Option.value_exn m.diff).grad in
  let rhs = if rhs_is_grad then m1.value else (Option.value_exn m1.diff).grad in
  High_level.Accum_unop { zero_out; accum; lhs; op; rhs; projections }

type session_state = { mutable first_session_id : int; mutable next_session_id : int }
(** A current session is the range of nodes from [session_state.first_session_id] to
    [session_state.next_session_id - 1] (possibly empty).

    TODO: Subtensors with [id] before this range are no longer updated
    and can only be used in new tensors if they are cross-session-persistent: not depending on
    fetching operations. This condition is checked automatically. *)

let session_state = { first_session_id = 0; next_session_id = 0 }

let binop ~op_label ?desc_label ?(compose_op = Shape.Pointwise_bin) ~op_body ~grad_body ~is_diff m1 m2 =
  (* Note: do not capture m1, m2 in any closure, so they can be GC'd. *)
  if m1.id < session_state.first_session_id && not m1.cross_session_persistent then
    raise @@ Session_error ("The subtensor is outside of current session", Some m1);
  if m2.id < session_state.first_session_id && not m2.cross_session_persistent then
    raise @@ Session_error ("The subtensor is outside of current session", Some m2);
  let m1_first = m1.id <= m2.id in
  let m1_processed : bool = Option.is_some m1.diff && (not @@ Map.mem !global_roots m1.id) in
  let m2_processed : bool =
    Option.is_some m2.diff && (m2.id = m1.id || (not @@ Map.mem !global_roots m2.id))
  in
  let needs_gradient =
    match (m1.diff, m2.diff) with
    | Some form1, Some form2 -> form1.needs_gradient || form2.needs_gradient
    | Some form1, _ -> form1.needs_gradient
    | _, Some form2 -> form2.needs_gradient
    | _ -> false
  in
  let fixme_fragile = session_state.next_session_id in
  let shape = Shape.make ~id:fixme_fragile () in
  let n =
    Code.create_node_promoted_precision ~needs_gradient m1.node.node m2.node.node ~op_label ?desc_label
      ~children shape
  in
  let id = n.id in
  let shape_logic = Shape.Broadcast (compose_op, m1.shape, m2.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic } in
  let n1 = m1.node in
  let n2 = m2.node in
  Shape.propagate_shapes local_update;
  session_shape_updates := local_shape_update :: !session_shape_updates;
  let projections () = Shape.derive_projections local_shape_update in
  let op_body = op_body ~n ~n1 ~n2 ~projections in
  (* The code needs to be included in the order it was computed! *)
  let forward_body =
    Code.(
      match
        ( m1_processed,
          (if is_diff then m1.forward_body else m1.nondiff_forward_body),
          m2_processed,
          if is_diff then m2.forward_body else m2.nondiff_forward_body )
      with
      | true, _, true, _ | true, _, _, Noop | _, Noop, true, _ | _, Noop, _, Noop -> op_body
      | false, m1_body, false, m2_body when m1_first -> Seq (ParHint (m1_body, m2_body), op_body)
      | false, m1_body, false, m2_body -> Seq (ParHint (m2_body, m1_body), op_body)
      | _, _, false, m2_body -> Seq (m2_body, op_body)
      | false, m1_body, _, _ -> Seq (m1_body, op_body))
  in
  let init_values_body = create ~id Value shape in
  session_initializations := init_values_body :: !session_initializations;
  if not is_diff then
    {
      forward_body;
      diff = None;
      nondiff_forward_body = forward_body;
      id;
      node = n;
      shape_logic;
      shape;
      cross_session_persistent = false;
    }
  else
    let diff1, diff2 =
      match (m1.diff, m2.diff) with
      | Some form1, Some form2 -> (form1, form2)
      | None, _ -> raise @@ Session_error ("binop ~is_diff:true but subtensor is non-diff", Some m1)
      | _, None -> raise @@ Session_error ("binop ~is_diff:true but subtensor is non-diff", Some m2)
    in
    let m1_no_grad = m1_processed || not diff1.needs_gradient in
    let m2_no_grad = m2_processed || not diff2.needs_gradient in
    let zero_grads =
      List.filter
        [
          (m1_no_grad, diff1.zero_grads); (m2_no_grad, diff2.zero_grads); (needs_gradient, fetch_zeros n shape);
        ]
        ~f:fst
      |> List.map ~f:snd |> High_level.sequential
    in
    (* The code needs to be included in the reverse order to which it was computed! This guarantees
       that all ancestors of a node are backpropagated before the node is backpropagated, even for
       non-tree DAGs. *)
    let grad_body = if needs_gradient then grad_body ~n ~n1 ~n2 ~projections else Code.Noop in
    let grad_body =
      if diff1.needs_gradient then grad_body else Code.remove_updates { id = m1.id; field = Grad } grad_body
    in
    let grad_body =
      if diff2.needs_gradient then grad_body else Code.remove_updates { id = m2.id; field = Grad } grad_body
    in
    let backprop_body =
      match (m1_no_grad, diff1.backprop_body, m2_no_grad, diff2.backprop_body) with
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
    let diff = Some { zero_grads; backprop_body; needs_gradient } in
    let tensor =
      {
        forward_body;
        diff;
        nondiff_forward_body = Code.Noop;
        id;
        node = n;
        shape_logic;
        shape;
        cross_session_persistent = false;
      }
    in
    global_roots := Map.add_exn !global_roots ~key:id ~data:tensor;
    tensor

let unop ~op_label ?desc_label ?init_shape ~transpose_op ~op_body ~grad_body ~is_diff m1 : t =
  (* Note: do not capture m in any closure, so it can be GC'd. *)
  let m1_processed = Option.is_some m1.diff && (not @@ Map.mem !global_roots m1.id) in
  let children = [ { Node.sub_node = m1.node; computed_externally = m1_processed } ] in
  let needs_gradient = match m1.diff with Some form1 -> form1.needs_gradient | None -> false in
  let fixme_fragile = session_state.next_session_id in
  let shape = Shape.make ~id:fixme_fragile () in
  let n =
    Code.create_node_same_precision_as ~needs_gradient m1.node.node ~op_label ?desc_label ~children shape
  in
  let id = n.id in
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
  Shape.propagate_shapes local_update;
  session_shape_updates := local_shape_update :: !session_shape_updates;
  let projections () = Shape.derive_projections local_shape_update in
  let op_body = op_body ~n ~n1 ~projections in
  (* The code needs to be included in the order it was computed! *)
  let forward_body =
    if m1_processed then op_body
    else if is_diff then Code.Seq (m1.forward_body, op_body)
    else Seq (m1.nondiff_forward_body, op_body)
  in
  session_initializations := create ~id Value shape :: !session_initializations;
  if not is_diff then
    {
      forward_body;
      diff = None;
      nondiff_forward_body = forward_body;
      id;
      node = n;
      shape_logic;
      shape;
      cross_session_persistent = false;
    }
  else
    let diff1 =
      match m1.diff with
      | Some diff -> diff
      | None -> raise @@ Session_error ("binop ~is_diff:true but subtensor is non-diff", Some m1)
    in
    let m1_no_grad = m1_processed || not diff1.needs_gradient in
    let zero_grads =
      List.filter [ (m1_no_grad, diff1.zero_grads); (needs_gradient, fetch_zeros n shape) ] ~f:fst
      |> List.map ~f:snd |> High_level.sequential
    in
    let grad_body = if needs_gradient then grad_body ~n ~n1 ~projections else Code.Noop in
    let grad_body =
      if diff1.needs_gradient then grad_body else Code.remove_updates { id = m1.id; field = Grad } grad_body
    in
    (* The code needs to be included in the reverse order to which it was computed! *)
    let backprop_body =
      match (m1_no_grad, diff1.backprop_body) with
      | true, _ | _, Noop -> grad_body
      | false, m1_body -> Seq (grad_body, m1_body)
    in
    if needs_gradient then session_initializations := create ~id Grad shape :: !session_initializations;

    if not m1_processed then global_roots := Map.remove !global_roots m1.id;
    let diff = Some { backprop_body; needs_gradient } in
    let tensor =
      {
        forward_body;
        diff;
        nondiff_forward_body = Code.Noop;
        id;
        node = n;
        shape_logic;
        shape;
        cross_session_persistent = false;
      }
    in
    global_roots := Map.add_exn !global_roots ~key:id ~data:tensor;
    tensor

(** A terminal: a constant, a parameter, an input of the model. *)
let term ~label ?desc_label ~needs_gradient ~is_diff ?batch_dims ?input_dims ?output_dims ?axis_labels
    ?deduced ?init_op ?fetch_op ?postprocess_op () =
  if needs_gradient && not is_diff then
    invalid_arg "Tensor.term ~needs_gradient:true: a non-diff tensor cannot need gradient";
  let literal : bool =
    if needs_gradient then false
    else
      match (init_op, fetch_op, postprocess_op) with
      | Some (Low_level.Constant_fill [| _ |]), None, None -> true
      | _ -> false
  in
  let op_label : string = label in
  let fixme_fragile = session_state.next_session_id in
  let shape = Shape.make ~id:fixme_fragile ?batch_dims ?input_dims ?output_dims ?axis_labels ?deduced () in
  let dims () = Shape.to_dims shape in
  let n : Code.node =
    Code.create_node ~value_prec:!default_value_prec ~grad_prec:!default_grad_prec ~literal ~needs_gradient
      ~op_label ?desc_label ~children:[] shape
  in
  let id = n.id in
  let shape_logic = Shape.Terminal in
  (* NOTE: this update does not do any work, but that might change in the future,
     and having it in the update sequence might help with debuggability. *)
  let local_shape_update = Shape.{ shape; logic = shape_logic } in
  Shape.propagate_shapes local_update;
  session_shape_updates := local_shape_update :: !session_shape_updates;

  let init_op : Code.init_op =
    Option.value_or_thunk init_op ~default:(fun () -> Low_level.Constant_fill [| 0.0 |])
  in
  session_initializations := create ~id ~init_op Value shape :: !session_initializations;
  let cross_session_persistent = Option.is_none fetch_op && Option.is_none postprocess_op in
  let forward_body =
    Option.value ~default:High_level.Noop fetch_op
    @@
    match fetch_op with
    | None ->
        if literal && Code.virtualize_settings.inline_constants then
          let fetch_op = match init_op with Constant_fill [| c |] -> Constant c | _ -> assert false in
          High_level.Fetch { tensor; fetch_op; dims }
        else Noop
    | Some fetch_op ->
        let fetch_op = fetch_op ~n in
        (match fetch_op with
        | Constant _ -> ()
        | _ ->
            n.annot.value_never_virtual <- true;
            n.annot.value_never_device_only <- true);
        Fetch { tensor; fetch_op; dims }
  in
  Option.iter postprocess_op ~f:(fun postprocess_op ->
      let postprocess_op = postprocess_op ~n in
      session_postprocess := postprocess_op :: !session_postprocess;
      n.annot.value_never_virtual <- true;
      n.annot.value_never_device_only <- true);
  if not is_diff then
    {
      forward_body;
      diff = None;
      nondiff_forward_body = forward_body;
      id;
      node = n;
      shape_logic;
      shape;
      cross_session_persistent;
    }
  else
    let backprop_body = Code.Noop in
    let zero_grads = if needs_gradient then fetch_zeros n shape else High_level.Noop in
    (* Very unlikely someone will want dw/dw. *)
    if needs_gradient then session_initializations := create ~id Grad shape :: !session_initializations;
    let diff = Some { zero_grads; backprop_body; needs_gradient } in
    let tensor =
      {
        forward_body;
        diff;
        nondiff_forward_body = Code.Noop;
        id;
        node = n;
        shape_logic;
        shape;
        cross_session_persistent;
      }
    in
    global_roots := Map.add_exn !global_roots ~key:id ~data:tensor;
    tensor

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
      Seq (fetch_ones ~id:m.id Grad m.shape, (Option.value_exn m.diff).backprop_body) )

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
  Sexp.message "Tensor"
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

let number ?desc_label ~is_diff ?(axis_label = "") c =
  (* Note: no axis label so that we do not conflict with user labels. *)
  term ?desc_label ~label:(float_to_label c) ~is_diff ~needs_gradient:false ~batch_dims:[] ~input_dims:[]
    ~output_dims:[ Shape.dim 1 ]
    ~axis_labels:axis_label ~init_op:(Constant_fill [| c |]) ()

let ndarray ?desc_label ~is_diff ?(needs_gradient = false) ?(batch_dims = []) ?(input_dims = [])
    ?(output_dims = []) ?axis_labels ?label values =
  let label =
    match label with
    | Some label -> label
    | None ->
        Caml.Format.pp_set_geometry Caml.Format.str_formatter ~max_indent:!max_sublabel_length
          ~margin:(!max_sublabel_length * 2);
        let ( ! ) = Array.of_list_map ~f:(fun d -> d.Shape.dim) in
        let dims = Array.concat [ !batch_dims; !output_dims; !input_dims ] in
        let ndarr = Ndarray.create Ndarray.double dims (Constant_fill values) in
        let ( ! ) = List.length in
        Ndarray.pp_array_inline ~num_batch_axes:!batch_dims ~num_output_axes:!output_dims
          ~num_input_axes:!input_dims Caml.Format.str_formatter ndarr;
        Caml.Format.flush_str_formatter ()
  in
  let label =
    if String.contains label '\n' then
      "c" ^ Shape.dims_to_string
      @@ Array.concat_map [| batch_dims; output_dims; input_dims |] ~f:Array.of_list
    else label
  in
  term ?desc_label ~needs_gradient ~is_diff ~batch_dims ~input_dims ~output_dims ?axis_labels
    ~deduced:Not_constrained ~label ~init_op:(Constant_fill values) ()

let params ?desc_label ?axis_labels ?input_dims ?output_dims ?deduced ?values ?value label =
  let init_op =
    match (values, value) with
    | Some _, Some _ -> invalid_arg "Tensor.params: do not provide both ~values and ~value"
    | Some values, _ -> Low_level.Constant_fill values
    | _, Some value -> Low_level.Constant_fill [| value |]
    | None, None -> Standard_uniform
  in
  term ?desc_label ~needs_gradient:true ~is_diff:true ~batch_dims:[] ?input_dims ?output_dims ?axis_labels
    ?deduced ~label ~init_op ()

module FDSL = struct
  let term = term ~is_diff:true
  let number = number ~is_diff:true
  let ndarray = ndarray ~is_diff:true
  let params = params
end

module NFDSL = struct
  let term = term ~is_diff:false
  let number = number ~is_diff:false
  let ndarray = ndarray ~is_diff:false
end

(** Construction of runtime-compiled code supporting backpropagation. *)

open Base
open Arrayjit
module LA = Low_level.Lazy_array

type diff = {
  grad : LA.t;
  zero_grads : High_level.t;
      (** Prepares for backpropagation. Always compile as: [backprop = Seq (zero_grads, backprop_body)]. *)
  backprop_body : High_level.t;
      (** Backpropagates for the tensor and its descendants; which typically means adding
          partial gradients to the gradient tensor of the subtensors, then for sub-subtensors etc. *)
}
[@@deriving sexp_of]

type t = {
  forward_body : High_level.t;
  diff : diff option;
  id : int;  (** Same as [value.id]. *)
  value : LA.t;
  shape : Shape.t;
      (** The eventual shape of [.!(t.node).value] and [.!(t.node).grad], incorporating the current state of
          shape inference. *)
  children : subtensor list;
}
[@@deriving sexp_of]
(** Information needed for compositional code generation. The code generation is suspended so that
    it can incorporate inferred shape information. *)

and subtensor = { subtensor : t; embedded : bool }

let rec sexp_of_t t =
  Sexp.message "Tensor"
    [
      ("id", Int.sexp_of_t t.id);
      ("label", sexp_of_string t.value.label);
      ("children", [%sexp_of: subtensor list] t.children);
    ]

and sexp_of_subtensor ch =
  Sexp.message "child" [ ("", sexp_of_t ch.subtensor); ("embedded", sexp_of_bool ch.embedded) ]

include Comparator.Make (struct
  type nonrec t = t

  let compare t1 t2 = Int.compare t1.id t2.id
  let sexp_of_t = sexp_of_t
end)

(** A forward root is a tensor that is not (currently) used to compute another tensor. *)
let forward_roots = ref @@ Map.empty (module Int)

(** A backprop root is a tensor with a gradient that is not (currently) receiving gradients from
    another tensor. I.e. it is not currently used to compute a tensor with a gradient. *)
let backprop_roots = ref @@ Map.empty (module Int)

(** We perform each update (at least) twice to propagate information between all subtensors:
    first in postfix order while computing [t], then in prefix order by iterating over this stack. *)
let session_shape_updates : Shape.update_step list ref = ref []

let session_initialized = ref 0
let default_value_prec = ref Ndarray.single
let default_grad_prec = ref Ndarray.single

exception Session_error of string * t option [@@deriving sexp]

let session_error_printer = function
  | Session_error (msg, None) -> Some msg
  | Session_error (msg, Some m) -> Some ("For #" ^ Int.to_string_hum m.id ^ ": " ^ msg)
  | _ -> None

let () = Caml.Printexc.register_printer session_error_printer

let fetch_zeros array shape =
  High_level.Fetch { array; fetch_op = Constant 0.; dims = lazy (Shape.to_dims shape) }

let fetch_ones array shape =
  High_level.Fetch { array; fetch_op = Constant 1.; dims = lazy (Shape.to_dims shape) }

let default_init_op = Low_level.Constant_fill [| 0.0 |]
let max_sublabel_length = ref 25

let raw_binop ~zero_out ~accum ~t ~lhs_is_grad ~op ~t1 ~rhs1_is_grad ~t2 ~rhs2_is_grad ~logic =
  let shape = t.shape in
  let shape_logic = Shape.Broadcast (logic, t1.shape, t2.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic } in
  Shape.propagate_shapes local_shape_update;
  session_shape_updates := local_shape_update :: !session_shape_updates;
  let projections = lazy (Shape.derive_projections local_shape_update) in
  let lhs = if lhs_is_grad then t.value else (Option.value_exn t.diff).grad in
  let rhs1 = if rhs1_is_grad then t1.value else (Option.value_exn t1.diff).grad in
  let rhs2 = if rhs2_is_grad then t2.value else (Option.value_exn t2.diff).grad in
  High_level.Accum_binop { zero_out; accum; lhs; op; rhs1; rhs2; projections }

let raw_unop ~zero_out ~accum ~t ~lhs_is_grad ~op ~t1 ~rhs_is_grad ~logic =
  let shape = t.shape in
  let shape_logic = Shape.Transpose (logic, t1.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic } in
  Shape.propagate_shapes local_shape_update;
  session_shape_updates := local_shape_update :: !session_shape_updates;
  let projections = lazy (Shape.derive_projections local_shape_update) in
  let lhs = if lhs_is_grad then t.value else (Option.value_exn t.diff).grad in
  let rhs = if rhs_is_grad then t1.value else (Option.value_exn t1.diff).grad in
  High_level.Accum_unop { zero_out; accum; lhs; op; rhs; projections }

type session_state = { mutable next_session_id : int }

let session_state = { next_session_id = 0 }

type grad_spec = Require_grad | Prohibit_grad | If_needed [@@deriving sexp, equal, variants]

let op ~op_label ?(desc_label = "") ?(compose_op = Shape.Pointwise_bin) ?(transpose_op = Shape.Pointwise_un)
    ~op_body ~grad_body ?(grad_spec = If_needed) make_shape ts =
  let ts = List.sort ts ~compare:(fun t1 t2 -> Int.ascending t1.id t2.id) in
  let fwd_embed = List.map ts ~f:(fun ti -> Map.mem !forward_roots ti.id) in
  List.iter2_exn ts fwd_embed ~f:(fun ti e -> if e then forward_roots := Map.remove !forward_roots ti.id);
  let children = List.map2_exn ts fwd_embed ~f:(fun ti embedded -> { subtensor = ti; embedded }) in
  let id = session_state.next_session_id in
  session_state.next_session_id <- session_state.next_session_id + 1;
  let shape = make_shape ~id in
  let vs = List.map ts ~f:(fun ti -> ti.value) in
  let dims = lazy (Shape.to_dims shape) in
  let label = op_label ^ if String.is_empty desc_label then "" else "/" ^ desc_label in
  let prec =
    List.map vs ~f:(fun v -> v.prec)
    |> List.reduce ~f:Ndarray.promote_prec
    |> Option.value ~default:!default_value_prec
  in
  let v = LA.create prec ~id ~label ~dims ~literal:false default_init_op in
  let rec shape_logics = function
    | [] -> [ Shape.Terminal ]
    | [ t1 ] -> [ Shape.Transpose (transpose_op, t1.shape) ]
    | [ t1; t2 ] -> [ Shape.Broadcast (compose_op, t1.shape, t2.shape) ]
    | t1 :: (t2 :: _ as ts) -> Shape.Broadcast (compose_op, t1.shape, t2.shape) :: shape_logics ts
  in
  let local_shape_updates = List.map ~f:(fun logic -> Shape.{ shape; logic }) @@ shape_logics ts in
  List.iter ~f:Shape.propagate_shapes local_shape_updates;
  session_shape_updates := local_shape_updates @ !session_shape_updates;
  let projections () =
    (* FIXME: ternary ops need an [rhs3] projection! I'll also convert to Lazy. *)
    Shape.derive_projections @@ List.hd_exn local_shape_updates
  in
  (* The code needs to be included in the order it was computed due to potential non-tree DAGs. *)
  let fwds = List.map2_exn ts fwd_embed ~f:(fun ti e -> if not e then High_level.Noop else ti.forward_body) in
  let forward_body = High_level.sequential @@ fwds @ [ op_body ~v ~vs ~projections ] in
  if
    is_prohibit_grad grad_spec
    || (Fn.non is_require_grad grad_spec && List.for_all ts ~f:(fun ti -> Option.is_none ti.diff))
  then (
    let tensor = { forward_body; diff = None; id; value = v; shape; children } in
    forward_roots := Map.add_exn !forward_roots ~key:id ~data:tensor;
    tensor)
  else
    let bck_embed = List.map ts ~f:(fun ti -> Map.mem !backprop_roots ti.id) in
    List.iter2_exn ts bck_embed ~f:(fun ti e -> if e then backprop_roots := Map.remove !backprop_roots ti.id);
    let gs = List.map ts ~f:(fun ti -> Option.map ti.diff ~f:(fun d -> d.grad)) in
    let g_prec =
      let f g = Option.map g ~f:(fun g -> g.LA.prec) in
      Option.value ~default:!default_grad_prec @@ List.reduce ~f:Ndarray.promote_prec @@ List.filter_map gs ~f
    in
    let g = LA.create g_prec ~id ~label:("grad " ^ label) ~dims ~literal:false default_init_op in
    let dcode ti = Option.value_map ti.diff ~default:High_level.Noop in
    let zero_grads =
      let f = dcode ~f:(fun diff -> diff.zero_grads) in
      let zeros =
        List.map2_exn (List.map ~f ts) bck_embed ~f:(fun z e -> if not e then High_level.Noop else z)
      in
      High_level.sequential @@ zeros @ [ fetch_zeros g shape ]
    in
    (* The code needs to be included in the reverse order to which it was computed! This guarantees
       that all ancestors of a node are backpropagated before the node is backpropagated, even for
       non-tree DAGs. *)
    let backprop_body =
      let f = dcode ~f:(fun diff -> diff.backprop_body) in
      let bcks =
        List.map2_exn (List.map ~f ts) bck_embed ~f:(fun z e -> if not e then High_level.Noop else z)
      in
      High_level.sequential @@ (grad_body ~v ~vs ~g ~gs ~projections :: List.rev bcks)
    in
    (* The order is not relevant, we keep the same order as in backprop for readability. *)
    let diff = Some { grad = g; zero_grads; backprop_body } in
    let tensor = { forward_body; diff; id; value = v; shape; children } in
    forward_roots := Map.add_exn !forward_roots ~key:id ~data:tensor;
    backprop_roots := Map.add_exn !backprop_roots ~key:id ~data:tensor;
    tensor

let binop ~op_label ?desc_label ?compose_op ~op_body ~grad_body ?grad_spec t1 t2 =
  let op_body ~v ~vs ~projections =
    match vs with [ v1; v2 ] -> op_body ~v ~v1 ~v2 ~projections | _ -> assert false
  in
  let grad_body ~v ~vs ~g ~gs ~projections =
    match (vs, gs) with
    | [ v1; v2 ], [ g1; g2 ] -> grad_body ~v ~v1 ~v2 ~g ~g1 ~g2 ~projections
    | _ -> assert false
  in
  op ~op_label ?desc_label ?compose_op ?transpose_op:None ~op_body ~grad_body ?grad_spec (Shape.make ())
    [ t1; t2 ]

let unop ~op_label ?desc_label ?compose_op ~op_body ~grad_body ?grad_spec t1 =
  let op_body ~v ~vs ~projections =
    match vs with [ v1 ] -> op_body ~v ~v1 ~projections | _ -> assert false
  in
  let grad_body ~v ~vs ~g ~gs ~projections =
    match (vs, gs) with [ v1 ], [ g1 ] -> grad_body ~v ~v1 ~g ~g1 ~projections | _ -> assert false
  in
  op ~op_label ?desc_label ?compose_op ?transpose_op:None ~op_body ~grad_body ?grad_spec (Shape.make ())
    [ t1 ]

(** A terminal: a constant, a parameter, an input of the model. *)
let term ~label ?desc_label ~grad_spec ?batch_dims ?input_dims ?output_dims ?axis_labels ?deduced ?init_op
    ?fetch_op () =
  let literal : bool =
    if is_require_grad grad_spec then false
    else match (init_op, fetch_op) with Some (Low_level.Constant_fill [| _ |]), None -> true | _ -> false
  in
  let op_body ~v ~vs:_ ~projections =
    let open High_level in
    let dims = lazy (projections ()).Indexing.lhs_dims in
    match fetch_op with
    | None ->
        if literal && Low_level.virtualize_settings.inline_constants then
          let fetch_op =
            match init_op with Some (Low_level.Constant_fill [| c |]) -> Constant c | _ -> assert false
          in
          Fetch { array = v; fetch_op; dims }
        else Noop
    | Some fetch_op ->
        let fetch_op = fetch_op ~v in
        (match fetch_op with
        | Constant _ -> ()
        | _ ->
            v.never_virtual <- true;
            v.never_device_only <- true);
        Fetch { array = v; fetch_op; dims }
  in
  let grad_body ~v:_ ~vs:_ ~g:_ ~gs:_ ~projections:_ = High_level.Noop in
  let make_shape = Shape.make ?batch_dims ?input_dims ?output_dims ?axis_labels ?deduced () in
  op ~op_label:label ?desc_label ?compose_op:None ?transpose_op:None ~op_body ~grad_body ~grad_spec make_shape
    []

let error_if_unknown_shape m =
  match m.shape with
  | { input = Unknown; _ } -> raise @@ Session_error ("Shape of inputs is still unknown", Some m)
  | { output = Unknown; _ } -> raise @@ Session_error ("Shape of outputs is still unknown", Some m)
  | { batch = Unknown; _ } -> raise @@ Session_error ("Shape of batching is still unknown", Some m)
  | { output = Inferred []; _ } ->
      raise @@ Session_error ("Shape of outputs is still empty -- missing shape information", Some m)
  | { input = _; output = _; batch = _; axis_labels = _; deduce_within_shape_constraints = _; id = _ } -> ()

let float_to_label v = Float.to_string_hum ~strip_zero:true v

let number ?desc_label ?(axis_label = "") ?(grad_spec = Prohibit_grad) c =
  (* Note: no axis label so that we do not conflict with user labels. *)
  term ?desc_label ~label:(float_to_label c) ~grad_spec ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ]
    ~axis_labels:axis_label ~init_op:(Constant_fill [| c |]) ()

let ndarray ?desc_label ?(grad_spec = Prohibit_grad) ?(batch_dims = []) ?(input_dims = []) ?(output_dims = [])
    ?axis_labels ?label values =
  let label =
    match label with
    | Some label -> label
    | None ->
        Caml.Format.pp_set_geometry Caml.Format.str_formatter ~max_indent:!max_sublabel_length
          ~margin:(!max_sublabel_length * 2);
        let dims = Array.concat_map [| batch_dims; output_dims; input_dims |] ~f:Array.of_list in
        let ndarr = Ndarray.create_array Ndarray.double ~dims (Constant_fill values) in
        let ( ! ) = List.length in
        Ndarray.pp_array_inline ~num_batch_axes:!batch_dims ~num_output_axes:!output_dims
          ~num_input_axes:!input_dims Caml.Format.str_formatter ndarr;
        Caml.Format.flush_str_formatter ()
  in
  let label =
    if String.contains label '\n' then
      "c" ^ Indexing.dims_to_string
      @@ Array.concat_map [| batch_dims; output_dims; input_dims |] ~f:Array.of_list
    else label
  in
  term ?desc_label ~grad_spec ~batch_dims ~input_dims ~output_dims ?axis_labels ~deduced:Not_constrained
    ~label ~init_op:(Constant_fill values) ()

let params ?desc_label ?axis_labels ?input_dims ?output_dims ?deduced ?values label =
  let init_op =
    match values with Some values -> Low_level.Constant_fill values | None -> Standard_uniform
  in
  term ?desc_label ~grad_spec:Require_grad ~batch_dims:[] ?input_dims ?output_dims ?axis_labels ?deduced
    ~label ~init_op ()

module TDSL = struct
  let term = term
  let number = number
  let ndarray = ndarray
  let params = params
end

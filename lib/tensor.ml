(** Construction of runtime-compiled code supporting backpropagation. *)

open Base
module Nd = Arrayjit.Ndarray
module LA = Arrayjit.Lazy_array
open Arrayjit

type diff = {
  grad : LA.t;
  zero_grads : Assignments.t;
      (** Prepares for backpropagation. Always compile as: [Seq (zero_grads, backprop)]. *)
  backprop : Assignments.t;
      (** Backpropagates for the tensor and its descendants; which typically means adding
          partial gradients to the gradient tensor of the subtensors, then for sub-subtensors etc. *)
}
[@@deriving sexp_of]

type t = {
  forward : Assignments.t;
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
      ("id", sexp_of_int t.id);
      ("label", sexp_of_string t.value.label);
      ("children", [%sexp_of: subtensor list] t.children);
    ]

and sexp_of_subtensor ch =
  Sexp.message "child"
    [ (if ch.embedded then ("", sexp_of_t ch.subtensor) else ("ref-id", sexp_of_int ch.subtensor.id)) ]

include Comparator.Make (struct
  type nonrec t = t

  let compare t1 t2 = Int.compare t1.id t2.id
  let sexp_of_t = sexp_of_t
end)

type session_state = {
  mutable next_id : int;
  mutable forward_roots : t Map.M(Int).t;
      (** A forward root is a tensor that is not (currently) used to compute another tensor. *)
  mutable backprop_roots : t Map.M(Int).t;
      (** A backprop root is a tensor with a gradient that is not (currently) receiving gradients from
          another tensor. I.e. it is not currently used to compute a tensor with a gradient. *)
  mutable shape_updates : Shape.update_step list;
      (** We perform each update (at least) twice to propagate information between all subtensors:
          first in postfix order while computing [t], then in prefix order by iterating over this stack. *)
}

let session_state =
  {
    next_id = 0;
    forward_roots = Map.empty (module Int);
    backprop_roots = Map.empty (module Int);
    shape_updates = [];
  }

let is_fwd_root t = Map.mem session_state.forward_roots t.id
let remove_fwd_root t = session_state.forward_roots <- Map.remove session_state.forward_roots t.id
let forward_roots () = session_state.forward_roots
let backprop_roots () = session_state.backprop_roots

let propagate_shape_updates () =
  List.iter ~f:Shape.propagate_shapes session_state.shape_updates;
  session_state.shape_updates <- []

let default_value_prec = ref Ops.single
let default_grad_prec = ref Ops.single

exception Session_error of string * t option [@@deriving sexp]

let session_error_printer = function
  | Session_error (msg, None) -> Some msg
  | Session_error (msg, Some m) -> Some ("For #" ^ Int.to_string_hum m.id ^ ": " ^ msg)
  | _ -> None

let () = Stdlib.Printexc.register_printer session_error_printer

let lazy_to_dims shape =
  lazy
    (propagate_shape_updates ();
     Shape.to_dims shape)

let lazy_projections shape_update =
  lazy
    (propagate_shape_updates ();
     Shape.derive_projections shape_update)

let fetch_zeros array shape = Assignments.Fetch { array; fetch_op = Constant 0.; dims = lazy_to_dims shape }
let fetch_ones array shape = Assignments.Fetch { array; fetch_op = Constant 1.; dims = lazy_to_dims shape }
let default_init_op = Ops.Constant_fill { values = [| 0.0 |]; strict = false }
let max_sublabel_length = ref 25

let raw_binop ~zero_out ~accum ~t ~lhs_is_grad ~op ~t1 ~rhs1_is_grad ~t2 ~rhs2_is_grad ~logic =
  let shape = t.shape in
  let shape_logic = Shape.Broadcast (logic, t1.shape, t2.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic } in
  Shape.propagate_shapes local_shape_update;
  session_state.shape_updates <- local_shape_update :: session_state.shape_updates;
  let projections = lazy_projections local_shape_update in
  let lhs = if lhs_is_grad then (Option.value_exn t.diff).grad else t.value in
  let rhs1 = if rhs1_is_grad then (Option.value_exn t1.diff).grad else t1.value in
  let rhs2 = if rhs2_is_grad then (Option.value_exn t2.diff).grad else t2.value in
  Assignments.Accum_binop { zero_out; accum; lhs; op; rhs1; rhs2; projections }

let raw_unop ~zero_out ~accum ~t ~lhs_is_grad ~op ~t1 ~rhs_is_grad ~logic =
  let shape = t.shape in
  let shape_logic = Shape.Transpose (logic, t1.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic } in
  Shape.propagate_shapes local_shape_update;
  session_state.shape_updates <- local_shape_update :: session_state.shape_updates;
  let projections = lazy_projections local_shape_update in
  let lhs = if lhs_is_grad then (Option.value_exn t.diff).grad else t.value in
  let rhs = if rhs_is_grad then (Option.value_exn t1.diff).grad else t1.value in
  Assignments.Accum_unop { zero_out; accum; lhs; op; rhs; projections }

type grad_spec = Require_grad | Prohibit_grad | If_needed [@@deriving sexp, equal, variants]

let op ~op_label ?(desc_label = "") ?(compose_op = Shape.Pointwise_bin) ?(transpose_op = Shape.Pointwise_un)
    ?(init_op = default_init_op) ~op_asn ~grad_asn ?(grad_spec = If_needed) make_shape ts =
  let ts = List.sort ts ~compare:(fun t1 t2 -> Int.ascending t1.id t2.id) in
  let fwd_embed = List.map ts ~f:is_fwd_root in
  List.iter2_exn ts fwd_embed ~f:(fun ti e -> if e then remove_fwd_root ti);
  let children = List.map2_exn ts fwd_embed ~f:(fun ti embedded -> { subtensor = ti; embedded }) in
  let id = session_state.next_id in
  session_state.next_id <- session_state.next_id + 1;
  let shape = make_shape ~id in
  let dims = lazy_to_dims shape in
  let label = op_label ^ if String.is_empty desc_label then "" else "#" ^ desc_label in
  let prec =
    List.map ts ~f:(fun ti -> ti.value.prec)
    |> List.reduce ~f:Ops.promote_prec
    |> Option.value ~default:!default_value_prec
  in
  let v = LA.create prec ~id ~label ~dims ~literal:false init_op in
  let rec shape_logics = function
    | [] -> [ Shape.Terminal init_op ]
    | [ t1 ] -> [ Shape.Transpose (transpose_op, t1.shape) ]
    | [ t1; t2 ] -> [ Shape.Broadcast (compose_op, t1.shape, t2.shape) ]
    | t1 :: (t2 :: _ as ts) -> Shape.Broadcast (compose_op, t1.shape, t2.shape) :: shape_logics ts
  in
  let local_shape_updates = List.map ~f:(fun logic -> Shape.{ shape; logic }) @@ shape_logics ts in
  List.iter ~f:Shape.propagate_shapes local_shape_updates;
  session_state.shape_updates <- local_shape_updates @ session_state.shape_updates;
  let projections = lazy_projections @@ List.hd_exn local_shape_updates in
  (* The code needs to be included in the order it was computed due to potential non-tree DAGs. *)
  let fwds = List.map2_exn ts fwd_embed ~f:(fun ti e -> if not e then Assignments.Noop else ti.forward) in
  let forward = Assignments.sequential @@ fwds @ [ op_asn ~v ~projections ] in
  if
    is_prohibit_grad grad_spec
    || (Fn.non is_require_grad grad_spec && List.for_all ts ~f:(fun ti -> Option.is_none ti.diff))
  then (
    let tensor = { forward; diff = None; id; value = v; shape; children } in
    session_state.forward_roots <- Map.add_exn session_state.forward_roots ~key:id ~data:tensor;
    tensor)
  else
    let bck_embed = List.map ts ~f:(fun ti -> Map.mem session_state.backprop_roots ti.id) in
    List.iter2_exn ts bck_embed ~f:(fun ti e ->
        if e then session_state.backprop_roots <- Map.remove session_state.backprop_roots ti.id);
    let g_prec =
      let f ti = Option.map ti.diff ~f:(fun d -> d.grad.LA.prec) in
      Option.value ~default:!default_grad_prec @@ List.reduce ~f:Ops.promote_prec @@ List.filter_map ts ~f
    in
    let grad_id = session_state.next_id in
    session_state.next_id <- session_state.next_id + 1;
    let g = LA.create g_prec ~id:grad_id ~label:("grad " ^ label) ~dims ~literal:false default_init_op in
    let dcode ti = Option.value_map ti.diff ~default:Assignments.Noop in
    let zero_grads =
      let f = dcode ~f:(fun diff -> diff.zero_grads) in
      let zeros =
        List.map2_exn (List.map ~f ts) bck_embed ~f:(fun z e -> if not e then Assignments.Noop else z)
      in
      Assignments.sequential @@ zeros @ [ fetch_zeros g shape ]
    in
    (* The code needs to be included in the reverse order to which it was computed! This guarantees
       that all ancestors of a node are backpropagated before the node is backpropagated, even for
       non-tree DAGs. *)
    let backprop =
      let f = dcode ~f:(fun diff -> diff.backprop) in
      let bcks =
        List.map2_exn (List.map ~f ts) bck_embed ~f:(fun z e -> if not e then Assignments.Noop else z)
      in
      Assignments.sequential @@ (grad_asn ~v ~g ~projections :: List.rev bcks)
    in
    (* The order is not relevant, we keep the same order as in backprop for readability. *)
    let diff = Some { grad = g; zero_grads; backprop } in
    let tensor = { forward; diff; id; value = v; shape; children } in
    session_state.forward_roots <- Map.add_exn session_state.forward_roots ~key:id ~data:tensor;
    session_state.backprop_roots <- Map.add_exn session_state.backprop_roots ~key:id ~data:tensor;
    tensor

let binop ~op_label ?desc_label ?compose_op ~op_asn ~grad_asn ?grad_spec t1 t2 =
  let op_asn ~v ~projections = op_asn ~v ~t1 ~t2 ~projections in
  let grad_asn ~v ~g ~projections = grad_asn ~v ~g ~t1 ~t2 ~projections in
  op ~op_label ?desc_label ?compose_op ?transpose_op:None ~op_asn ~grad_asn ?grad_spec (Shape.make ())
    [ t1; t2 ]

let unop ~op_label ?desc_label ?transpose_op ~op_asn ~grad_asn ?grad_spec t1 =
  let op_asn ~v ~projections = op_asn ~v ~t1 ~projections in
  let grad_asn ~v ~g ~projections = grad_asn ~v ~g ~t1 ~projections in
  op ~op_label ?desc_label ?compose_op:None ?transpose_op ~op_asn ~grad_asn ?grad_spec (Shape.make ()) [ t1 ]

(** A terminal: a constant, a parameter, an input of the model. *)
let term ~label ?desc_label ~grad_spec ?batch_dims ?input_dims ?output_dims ?axis_labels ?deduced ?init_op
    ?fetch_op () =
  let scalar_literal : bool =
    if is_require_grad grad_spec then false
    else
      match (init_op, fetch_op) with
      | Some (Ops.Constant_fill { values = [| _ |]; strict = _ }), None -> true
      | _ -> false
  in
  let op_asn ~v ~projections =
    let open Assignments in
    let dims =
      lazy
        (propagate_shape_updates ();
         (Lazy.force projections).Indexing.lhs_dims)
    in
    match fetch_op with
    | None ->
        if scalar_literal && Low_level.virtualize_settings.inline_constants then
          let fetch_op =
            match init_op with
            | Some (Ops.Constant_fill { values = [| c |]; _ }) -> Constant c
            | _ -> assert false
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
  let grad_asn ~v:_ ~g:_ ~projections:_ = Assignments.Noop in
  let make_shape = Shape.make ?batch_dims ?input_dims ?output_dims ?axis_labels ?deduced () in
  op ~op_label:label ?desc_label ?compose_op:None ?transpose_op:None ?init_op ~op_asn ~grad_asn ~grad_spec
    make_shape []

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
    ~axis_labels:axis_label
    ~init_op:(Constant_fill { values = [| c |]; strict = true })
    ()

let ndarray ?desc_label ?(grad_spec = Prohibit_grad) ?(batch_dims = []) ?(input_dims = []) ?(output_dims = [])
    ?axis_labels ?label ?(strict = true) values =
  let label =
    match label with
    | Some label -> label
    | None ->
        Stdlib.Format.pp_set_geometry Stdlib.Format.str_formatter ~max_indent:!max_sublabel_length
          ~margin:(!max_sublabel_length * 2);
        let dims = Array.concat_map [| batch_dims; output_dims; input_dims |] ~f:Array.of_list in
        let ndarr = Ndarray.create_array Ops.double ~dims (Constant_fill { values; strict }) in
        let ( ! ) = List.length in
        Ndarray.pp_array_inline ~num_batch_axes:!batch_dims ~num_output_axes:!output_dims
          ~num_input_axes:!input_dims Stdlib.Format.str_formatter ndarr;
        Stdlib.Format.flush_str_formatter ()
  in
  let label =
    if String.contains label '\n' then
      "c" ^ Indexing.dims_to_string
      @@ Array.concat_map [| batch_dims; output_dims; input_dims |] ~f:Array.of_list
    else label
  in
  term ?desc_label ~grad_spec ~batch_dims ~input_dims ~output_dims ?axis_labels ~deduced:Not_constrained
    ~label
    ~init_op:(Constant_fill { values; strict })
    ()

let param ?desc_label ?axis_labels ?input_dims ?output_dims ?deduced ?(strict = false) ?values label =
  let init_op =
    match values with Some values -> Ops.Constant_fill { values; strict } | None -> Standard_uniform
  in
  let t =
    term ?desc_label ~grad_spec:Require_grad ~batch_dims:[] ?input_dims ?output_dims ?axis_labels ?deduced
      ~label ~init_op ()
  in
  t.value.never_virtual <- true;
  t.value.never_device_only <- true;
  (* In principle, gradients can be device-only (in the global memory of the device). Gradients of param
     cannot be inlined because backpropagation and param update are usually separate computations. *)
  (Option.value_exn t.diff).grad.never_virtual <- true;
  t

let rec iter_embedded_arrays ~f t =
  f t.value;
  Option.iter t.diff ~f:(fun diff -> f diff.grad);
  List.iter ~f:(fun ch -> if ch.embedded then iter_embedded_arrays ~f ch.subtensor) t.children

(** *** Printing. *** *)

(** Converts ID, label and the dimensions of a node to a string. *)
let header t =
  let v_dims_s = LA.dims_to_string t.value in
  let g_dims_s = match t.diff with None -> "<no-grad>" | Some diff -> LA.dims_to_string diff.grad in
  let dims_s =
    if String.equal v_dims_s g_dims_s then "dims " ^ v_dims_s
    else "dims val " ^ v_dims_s ^ " grad " ^ g_dims_s
  in
  "#" ^ Int.to_string t.id ^ " " ^ t.value.label ^ " " ^ dims_s ^ " ["
  ^ String.concat ~sep:"," (List.map t.children ~f:(fun { subtensor = { id; _ }; _ } -> Int.to_string id))
  ^ "]"
(*^" "^PrintBox_text.to_string (PrintBox.Simple.to_box v.label)*)

type array_print_style =
  [ `Default
    (** The inner rectangles comprise both an input and an output axis, if available. Similarly,
      the outer rectangle comprises a second-from-end input axis and a second-from-end output axis,
      if available. At least one batch axis is output, when available.
      The axes that couldn't be output are printed at position/dimension [0]. *)
  | `N5_layout of string
    (** The string should provide exclusively non-negative integer pseudo-labels. The numbers [0]-[4] represent
      the priorities of the axes to be printed out, where the priorities correspond to, from highest:
      horizontal, vertical direction of the inner rectangle, horizontal, vertical direction of the outer
      rectangle, repetition (see also [Node.pp_print]). The numbers [n >= 5] stand for the actual
      positions [n - 5] within the corresponding axes. *)
  | `Label_layout of (string * int) list
    (** The association from axis labels to integers. The negative numbers [-5] to [-1] represent
      the priorities of the axes to be printed out, where the priorities correspond to, from highest:
      horizontal, vertical direction of the inner rectangle, horizontal, vertical direction of the outer
      rectangle, repetition (as above). The numbers [n >= 0] stand for the actual positions
      within the corresponding axes. Unspecified axes are printed at position [0]. *)
  | `Inline
    (** The tensors are printed linearly, in a bracketed manner, optionally prefixed with the labels
        specification. Note that the syntax causes ambiguity for 1-dimensional input axes (underscores are
        used for axes without explicit labels); when there is a 1-dimensional input axis, we output
        the labels specification even if there are no axis labels as a way to display the number of axes.
        The axis nesting is right-to-left (rightmost is innermost).
        The input axes are innermost and the batch axes outermost. The input axes use [,] as a separator
        and [()] as axis delimiters, but the delimiter for the outermost (i.e. leftmost) axis is omitted.
        The output axes use [;] as a separator and [[]] as axis delimiters (obligatory).
        The batch axes use [;] as a separator and [[||]] as axis delimiters (obligatory). *)
  ]
(** We print out up to 5 axes when printing a tensor, as a grid (outer rectangle) of (inner)
    rectangles, possibly repeated (screens). *)

let to_dag ?(single_node = false) ?entries_per_axis ~with_id ~with_value ~with_grad t =
  (* let la_to_box la = in *)
  let rec to_dag { subtensor = t; embedded } : PrintBox_utils.dag =
    let id = Int.to_string t.id in
    let children = if single_node then [] else List.map ~f:to_dag t.children in
    let prefix = "[" ^ id ^ "] " ^ t.value.label in
    let labels = Shape.axis_map_to_dims_index t.shape.axis_labels in
    let indices = Shape.default_display_indices t.shape in
    let txt = if with_id then prefix else t.value.label in
    let grad_txt diff =
      if String.is_substring (String.lowercase diff.grad.label) ~substring:"grad" then diff.grad.label
      else diff.grad.label ^ " Gradient"
    in
    match (not embedded, with_value, with_grad, t.value.array, t.diff) with
    | true, _, _, _, _ -> `Embed_subtree_ID (Int.to_string t.id)
    | _, false, false, _, _ | _, false, true, _, None -> `Subtree_with_ID (id, `Tree (`Text txt, children))
    | _, true, false, (lazy (Some v_array)), _ | _, true, true, (lazy (Some v_array)), None ->
        let node = `Box (Nd.render_array ~brief:true ~prefix ?entries_per_axis ~labels ~indices @@ v_array) in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, true, false, (lazy None), _ | _, true, true, (lazy None), None ->
        let node = `Text (txt ^ " <virtual>") in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, false, true, _, Some diff ->
        let prefix = prefix ^ " " ^ grad_txt diff in
        let node =
          match Lazy.force diff.grad.array with
          | Some g_array ->
              `Box (Nd.render_array ~brief:true ~prefix ?entries_per_axis ~labels ~indices g_array)
          | None -> `Text (prefix ^ " <virtual>")
        in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, true, true, (lazy (Some v_array)), Some diff ->
        let node =
          let value = Nd.render_array ~brief:true ~prefix ?entries_per_axis ~labels ~indices v_array in
          let grad =
            match Lazy.force diff.grad.array with
            | Some g_array ->
                `Box
                  (Nd.render_array ~brief:true ~prefix:(grad_txt diff) ?entries_per_axis ~labels ~indices
                     g_array)
            | None -> `Text (grad_txt diff ^ " <virtual>")
          in
          `Vlist (false, [ `Box value; grad ])
        in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, true, true, (lazy None), Some diff ->
        let node =
          let value = `Text (prefix ^ " " ^ t.value.label ^ " <virtual>") in
          let grad =
            match Lazy.force diff.grad.array with
            | Some g_array ->
                `Box
                  (Nd.render_array ~brief:true ~prefix:(grad_txt diff) ?entries_per_axis ~labels ~indices
                     g_array)
            | None -> `Text (grad_txt diff ^ " <virtual>")
          in
          `Vlist (false, [ value; grad ])
        in
        `Subtree_with_ID (id, `Tree (node, children))
  in
  to_dag { subtensor = t; embedded = true }

let to_printbox ?single_node ?entries_per_axis ?(with_id = false) ?(with_value = true) ~with_grad ~depth t =
  to_dag ?single_node ?entries_per_axis ~with_id ~with_value ~with_grad t |> PrintBox_utils.reformat_dag depth

let print ~with_grad ~with_code ?(with_low_level = false) (style : array_print_style) t =
  let sh = t.shape in
  let label = t.value.label in
  let prefix =
    "[" ^ Int.to_string t.id ^ "]: " ^ label ^ "shape "
    ^ Shape.to_string_hum ~style:`Axis_number_and_size sh
    ^ " "
  in
  let grad_txt diff =
    if String.is_substring (String.lowercase diff.grad.label) ~substring:"grad" then diff.grad.label
    else diff.grad.label ^ " Gradient"
  in
  let indices =
    match style with
    | `Default -> Shape.default_display_indices sh
    | `N5_layout priorities ->
        let f = function
          | Either.Second i -> i
          | First _ -> invalid_arg "`N5_layout requires integer-only labels"
        in
        let p_labels = Shape.(axis_labels_of_spec priorities).labels |> Map.map ~f in
        Shape.axis_map_to_dims_index p_labels
    | `Label_layout label_idcs ->
        let inv_labels =
          Map.to_alist sh.axis_labels |> List.map ~f:(fun (a, b) -> (b, a)) |> Map.of_alist (module String)
        in
        let inv_labels =
          match inv_labels with
          | `Duplicate_key l -> raise @@ Session_error ("`Label_layout found a repeating label: " ^ l, Some t)
          | `Ok inv_labels -> inv_labels
        in
        let idcs =
          List.map label_idcs ~f:(fun (l, i) ->
              match Map.find inv_labels l with
              | Some axis -> (axis, i)
              | None -> raise @@ Session_error ("`Label_layout label not found in shape: " ^ l, Some t))
        in
        Shape.axis_map_to_dims_index @@ Map.of_alist_exn (module Shape.AxisKey) idcs
    | `Inline -> [||]
  in
  let needs_spec =
    Fn.non Map.is_empty sh.axis_labels
    || Shape.(List.exists ~f:(( = ) 1) @@ list_of_dims @@ dims_of_kind Input sh)
  in
  let labels = Shape.axis_map_to_dims_index ~default:"" sh.axis_labels in
  let axes_spec = if needs_spec then Some (Shape.to_string_hum ~style:`Only_labels sh) else None in
  let num_axes kind = List.length Shape.(list_of_dims @@ dims_of_kind kind sh) in
  let num_batch_axes = num_axes Shape.AxisKey.Batch in
  let num_input_axes = num_axes Shape.AxisKey.Input in
  let num_output_axes = num_axes Shape.AxisKey.Output in
  (* TODO: code sharing with [to_dag] *)
  (if not @@ Lazy.is_val t.value.array then Stdlib.Format.printf "<not-in-yet>@ "
   else
     match (style, t.value.array) with
     | `Inline, (lazy None) -> Stdlib.Format.printf "<virtual>@ "
     | `Inline, (lazy (Some arr)) ->
         Nd.pp_array_inline Stdlib.Format.std_formatter ~num_batch_axes ~num_input_axes ~num_output_axes
           ?axes_spec arr
     | _, (lazy None) -> Stdlib.Format.printf "<virtual>@ "
     | _, (lazy (Some arr)) ->
         Nd.pp_array Stdlib.Format.std_formatter ~prefix ~labels ~indices arr;
         Stdlib.Format.print_newline ());
  if with_grad then
    Option.iter t.diff ~f:(fun diff ->
        if not @@ Lazy.is_val diff.grad.array then Stdlib.Format.printf "%s <not-in-yet>@ " (grad_txt diff)
        else
          match (style, diff.grad.array) with
          | `Inline, (lazy (Some arr)) ->
              Nd.pp_array_inline Stdlib.Format.std_formatter ~num_batch_axes ~num_input_axes ~num_output_axes
                ?axes_spec arr;
              Stdlib.Format.print_newline ()
          | _, (lazy (Some arr)) ->
              Nd.pp_array Stdlib.Format.std_formatter
                ~prefix:(prefix ^ " " ^ grad_txt diff)
                ~labels ~indices arr;
              Stdlib.Format.print_newline ()
          | _, (lazy None) -> Stdlib.Format.printf "%s <virtual>@ " (grad_txt diff));
  if with_code then (
    (match t.forward with
    | Noop -> ()
    | fwd_code -> Stdlib.Format.printf "Current forward body:@ %a@ " Assignments.fprint_code fwd_code);
    match t.diff with
    | Some { backprop = Noop; _ } -> ()
    | Some { backprop = bwd_code; _ } ->
        Stdlib.Format.printf "Current backprop body:@ %a@ " Assignments.fprint_code bwd_code
    | None -> ());
  if with_low_level then (
    (match t.forward with
    | Noop -> ()
    | fwd_code ->
        Stdlib.Format.printf "Current forward low-level body:@ %a@ " Low_level.fprint_code
        @@ Assignments.to_low_level fwd_code);
    match t.diff with
    | Some { backprop = Noop; _ } -> ()
    | Some { backprop = bwd_code; _ } ->
        Stdlib.Format.printf "Current backprop low-level body:@ %a@ " Low_level.fprint_code
        @@ Assignments.to_low_level bwd_code
    | None -> ());
  Stdio.printf "\n%!"

let print_forward_roots ~with_grad ~with_code (style : array_print_style) =
  List.iter (Map.to_alist ~key_order:`Increasing session_state.forward_roots) ~f:(fun (id, root) ->
      assert (id = root.id);
      print ~with_grad ~with_code style root)

let print_tree ?entries_per_axis ?(with_backend_info = false) ?(with_id = true) ?(with_value = true)
    ~with_grad ~depth t =
  (* FIXME: print backend info *)
  ignore with_backend_info;
  PrintBox_text.output Stdio.stdout @@ PrintBox_utils.dag_to_box @@ PrintBox_utils.boxify depth
  @@ to_dag ?entries_per_axis ~with_id ~with_value ~with_grad t

(** *** Accessors *** *)

let value_1d_points ?from_axis ~xdim t =
  Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_1d_points ?from_axis ~xdim arr)
  @@ Lazy.force t.value.array

let value_2d_points ?from_axis ~xdim ~ydim t =
  Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_2d_points ?from_axis ~xdim ~ydim arr)
  @@ Lazy.force t.value.array

let grad_1d_points ?from_axis ~xdim t =
  match t.diff with
  | None -> [||]
  | Some diff ->
      Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_1d_points ?from_axis ~xdim arr)
      @@ Lazy.force diff.grad.array

let grad_2d_points ?from_axis ~xdim ~ydim t =
  match t.diff with
  | None -> [||]
  | Some diff ->
      Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_2d_points ?from_axis ~xdim ~ydim arr)
      @@ Lazy.force diff.grad.array

let set_value t = Nd.set_from_float @@ Option.value_exn @@ Lazy.force t.value.array
let get_value t = Nd.get_as_float @@ Option.value_exn @@ Lazy.force t.value.array
let set_grad t = Nd.set_from_float @@ Option.value_exn @@ Lazy.force @@ (Option.value_exn t.diff).grad.array
let get_grad t = Nd.get_as_float @@ Option.value_exn @@ Lazy.force @@ (Option.value_exn t.diff).grad.array

let set_values t values =
  Ndarray.(reset (Constant_fill { values; strict = false }) @@ Option.value_exn @@ Lazy.force t.value.array)

module O = struct
  (** Get the value at the given indices. *)
  let ( .@{} ) = get_value

  let ( .@%{} ) = get_grad

  (** Set the value at the given indices. *)
  let ( .@{}<- ) = set_value

  let ( .@%{}<- ) = set_grad

  (** Get the value at the given index from a single-axis shape tensor. *)
  let ( .@[] ) t indx = get_value t [| indx |]

  let ( .@%[] ) t indx = get_grad t [| indx |]

  (** Set the value at the given index for a single-axis shape tensor. *)
  let ( .@[]<- ) t indx = set_value t [| indx |]

  let ( .@%[]<- ) t indx = set_grad t [| indx |]
end

(** Construction of runtime-compiled code supporting backpropagation. *)

open Base
module Nd = Arrayjit.Ndarray
module Tn = Arrayjit.Tnode
module Asgns = Arrayjit.Assignments
module Idx = Arrayjit.Indexing
module Debug_runtime = Arrayjit.Utils.Debug_runtime

[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

type diff = {
  grad : (Tn.t[@sexp.opaque]);
  zero_grads : Asgns.t;  (** Prepares for backpropagation. Always compile as: [Seq (zero_grads, backprop)]. *)
  backprop : Asgns.t;
      (** Backpropagates for the tensor and its descendants; which typically means adding
          partial gradients to the gradient tensor of the subtensors, then for sub-subtensors etc. *)
}
[@@deriving sexp_of]

type t = {
  forward : Asgns.t;
  diff : diff option;
  id : int;  (** Same as [value.id]. *)
  value : Tn.t;
  shape : Shape.t;
      (** The eventual shape of [t.value] and [t.diff.grad], incorporating the current state of
          shape inference. *)
  children : subtensor list;
}
(** Information needed for compositional code generation. The code generation is suspended so that
    it can incorporate inferred shape information. *)

and subtensor = { subtensor : t; embedded : bool }

let rec sexp_of_t t =
  Sexp.message "Tensor"
    [
      ("id", sexp_of_int t.id);
      ("label", [%sexp_of: string list] t.value.label);
      ("forward", [%sexp_of: Asgns.t] t.forward);
      ("diff", [%sexp_of: diff option] t.diff);
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
}

let session_state =
  { next_id = 0; forward_roots = Map.empty (module Int); backprop_roots = Map.empty (module Int) }

let is_fwd_root t = Map.mem session_state.forward_roots t.id
let remove_fwd_root t = session_state.forward_roots <- Map.remove session_state.forward_roots t.id
let forward_roots () = session_state.forward_roots
let backprop_roots () = session_state.backprop_roots
let default_value_prec = ref Arrayjit.Ops.single
let default_grad_prec = ref Arrayjit.Ops.single

exception Session_error of string * t option [@@deriving sexp]

let session_error_printer = function
  | Session_error (msg, None) -> Some msg
  | Session_error (msg, Some m) -> Some ("For #" ^ Int.to_string_hum m.id ^ ": " ^ msg)
  | _ -> None

let () = Stdlib.Printexc.register_printer session_error_printer
let lazy_to_dims shape = lazy (Shape.to_dims shape)
let fetch_zeros array shape = Asgns.Fetch { array; fetch_op = Constant 0.; dims = lazy_to_dims shape }
let fetch_ones array shape = Asgns.Fetch { array; fetch_op = Constant 1.; dims = lazy_to_dims shape }
let default_init_op = Arrayjit.Ops.Constant_fill { values = [| 0.0 |]; strict = false }
let max_sublabel_length = ref 25

let raw_binop ~initialize_neutral ~accum ~(t : t) ~(lhs_is_grad : bool) ~op ~(t1 : t) ~(rhs1_is_grad : bool)
    ~(t2 : t) ~rhs2_is_grad ~logic : Asgns.t =
  let shape = t.shape in
  let shape_logic = Shape.Broadcast (logic, t1.shape, t2.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic; id = get_update_id () } in
  Shape.propagate_shapes local_shape_update;
  let projections = lazy (Shape.derive_projections local_shape_update) in
  let lhs = if lhs_is_grad then (Option.value_exn t.diff).grad else t.value in
  let rhs1 = if rhs1_is_grad then (Option.value_exn t1.diff).grad else t1.value in
  let rhs2 = if rhs2_is_grad then (Option.value_exn t2.diff).grad else t2.value in
  Asgns.Accum_binop { initialize_neutral; accum; lhs; op; rhs1; rhs2; projections }

let raw_unop ~initialize_neutral ~accum ~(t : t) ~(lhs_is_grad : bool) ~op ~(t1 : t) ~(rhs_is_grad : bool)
    ~logic =
  let shape = t.shape in
  let shape_logic = Shape.Transpose (logic, t1.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic; id = get_update_id () } in
  Shape.propagate_shapes local_shape_update;
  let projections = lazy (Shape.derive_projections local_shape_update) in
  let lhs = if lhs_is_grad then (Option.value_exn t.diff).grad else t.value in
  let rhs = if rhs_is_grad then (Option.value_exn t1.diff).grad else t1.value in
  Asgns.Accum_unop { initialize_neutral; accum; lhs; op; rhs; projections }

type grad_spec = Require_grad | Prohibit_grad | If_needed [@@deriving sexp, equal, variants]

let op ~(label : string list) ?(compose_op = Shape.Pointwise_bin) ?(transpose_op = Shape.Pointwise_un)
    ?(init_op = default_init_op) ~op_asn ~grad_asn ?(grad_spec = If_needed) make_shape (orig_ts : t list) : t
    =
  let ordered_ts = List.dedup_and_sort orig_ts ~compare:(fun t1 t2 -> Int.ascending t1.id t2.id) in
  let children =
    List.folding_map orig_ts
      ~init:(Set.empty (module Int))
      ~f:(fun used ti ->
        (Set.add used ti.id, { subtensor = ti; embedded = is_fwd_root ti && not (Set.mem used ti.id) }))
  in
  let id = session_state.next_id in
  session_state.next_id <- session_state.next_id + 1;
  let shape = make_shape ~debug_name:(Tn.debug_name ~id ~label) ~id in
  let prec =
    List.map orig_ts ~f:(fun ti -> ti.value.prec)
    |> List.reduce ~f:Arrayjit.Ops.promote_prec
    |> Option.value ~default:!default_value_prec
  in
  let rec shape_logics = function
    | [] -> [ Shape.Terminal init_op ]
    | [ t1 ] -> [ Shape.Transpose (transpose_op, t1.shape) ]
    | [ t1; t2 ] -> [ Shape.Broadcast (compose_op, t1.shape, t2.shape) ]
    | t1 :: (t2 :: _ as ts) -> Shape.Broadcast (compose_op, t1.shape, t2.shape) :: shape_logics ts
  in
  let local_shape_updates =
    List.map ~f:(fun logic -> Shape.{ shape; logic; id = get_update_id () }) @@ shape_logics orig_ts
  in
  let dims = lazy_to_dims shape in
  List.iter ~f:Shape.propagate_shapes local_shape_updates;
  let projections = lazy (Shape.derive_projections @@ List.hd_exn local_shape_updates) in
  let v = Tn.create prec ~id ~label ~dims init_op in
  (* The code needs to be included in the order it was computed due to potential non-tree DAGs. *)
  let fwds = List.map ordered_ts ~f:(fun ti -> if is_fwd_root ti then ti.forward else Asgns.Noop) in
  let forward = Asgns.sequential @@ fwds @ [ op_asn ~v ~projections ] in
  List.iter ordered_ts ~f:(fun ti -> remove_fwd_root ti);
  if
    is_prohibit_grad grad_spec
    || (Fn.non is_require_grad grad_spec && List.for_all orig_ts ~f:(fun ti -> Option.is_none ti.diff))
  then (
    let tensor = { forward; diff = None; id; value = v; shape; children } in
    session_state.forward_roots <- Map.add_exn session_state.forward_roots ~key:id ~data:tensor;
    tensor)
  else
    let g_prec =
      let f ti = Option.map ti.diff ~f:(fun d -> d.grad.Tn.prec) in
      Option.value ~default:!default_grad_prec
      @@ List.reduce ~f:Arrayjit.Ops.promote_prec
      @@ List.filter_map orig_ts ~f
    in
    let grad_id = session_state.next_id in
    session_state.next_id <- session_state.next_id + 1;
    let g = Tn.create g_prec ~id:grad_id ~label:("grad" :: label) ~dims default_init_op in
    let dcode ti = Option.value_map ti.diff ~default:Asgns.Noop in
    let is_bck_root ti = Map.mem session_state.backprop_roots ti.id in
    let zero_grads =
      let zero_g = dcode ~f:(fun diff -> diff.zero_grads) in
      let zeros = List.map ordered_ts ~f:(fun ti -> if is_bck_root ti then zero_g ti else Asgns.Noop) in
      Asgns.sequential @@ zeros @ [ fetch_zeros g shape ]
    in
    (* The code needs to be included in the reverse order to which it was computed! This guarantees
       that all ancestors of a node are backpropagated before the node is backpropagated, even for
       non-tree DAGs. *)
    let backprop =
      let bprop = dcode ~f:(fun diff -> diff.backprop) in
      let bcks = List.map ordered_ts ~f:(fun ti -> if is_bck_root ti then bprop ti else Asgns.Noop) in
      Asgns.sequential @@ (grad_asn ~v ~g ~projections :: List.rev bcks)
    in
    List.iter ordered_ts ~f:(fun ti ->
        session_state.backprop_roots <- Map.remove session_state.backprop_roots ti.id);
    (* The order is not relevant, we keep the same order as in backprop for readability. *)
    let diff = Some { grad = g; zero_grads; backprop } in
    let tensor = { forward; diff; id; value = v; shape; children } in
    session_state.forward_roots <- Map.add_exn session_state.forward_roots ~key:id ~data:tensor;
    session_state.backprop_roots <- Map.add_exn session_state.backprop_roots ~key:id ~data:tensor;
    tensor

let binop ~label ?compose_op ~op_asn ~grad_asn ?grad_spec t1 t2 =
  let op_asn ~v ~projections = op_asn ~v ~t1 ~t2 ~projections in
  let grad_asn ~v ~g ~projections = grad_asn ~v ~g ~t1 ~t2 ~projections in
  op ~label ?compose_op ?transpose_op:None ~op_asn ~grad_asn ?grad_spec (Shape.make ()) [ t1; t2 ]

let unop ~label ?transpose_op ~op_asn ~grad_asn ?grad_spec t1 =
  let op_asn ~v ~projections = op_asn ~v ~t1 ~projections in
  let grad_asn ~v ~g ~projections = grad_asn ~v ~g ~t1 ~projections in
  op ~label ?compose_op:None ?transpose_op ~op_asn ~grad_asn ?grad_spec (Shape.make ()) [ t1 ]

(** A terminal: a constant, a parameter, an input of the model. *)
let term ~label ~grad_spec ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes ?deduced
    ?init_op ?fetch_op () =
  let op_asn ~v ~projections =
    let open Asgns in
    let dims = lazy (Lazy.force projections).Idx.lhs_dims in
    match (fetch_op, init_op) with
    | None, Some (Arrayjit.Ops.Constant_fill { values = [| _ |]; strict = _ })
      when not (is_require_grad grad_spec) ->
        (* The scalar literal case. *)
        let fetch_op =
          match init_op with
          | Some (Arrayjit.Ops.Constant_fill { values = [| c |]; _ }) -> Constant c
          | _ -> assert false
        in
        Fetch { array = v; fetch_op; dims }
    | None, _ -> Noop
    | Some fetch_op, _ ->
        let fetch_op = fetch_op ~v in
        (match fetch_op with
        | Constant _ | Slice _ | Embed_symbol _ -> ()
        | Imported _ ->
            (* Note: [Imported] can be used for merging across devices. But, some use cases of [Imported]
               will require a hosted tensor node. *)
            Tn.update_memory_mode v Materialized 22);
        Fetch { array = v; fetch_op; dims }
  in
  let grad_asn ~v:_ ~g:_ ~projections:_ = Asgns.Noop in
  let make_shape =
    Shape.make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes ?deduced ()
  in
  op ~label ?compose_op:None ?transpose_op:None ?init_op ~op_asn ~grad_asn ~grad_spec make_shape []

let float_to_label v = Float.to_string v

let number ?(label = []) ?axis_label ?(grad_spec = Prohibit_grad) c =
  (* Note: no axis label so that we do not conflict with user labels. *)
  let label = float_to_label c :: label in
  let init_op = Arrayjit.Ops.Constant_fill { values = [| c |]; strict = true } in
  let t = term ~label ~grad_spec ~batch_dims:[] ~input_dims:[] ~init_op in
  let t =
    match axis_label with
    | None -> t ~output_dims:[ 1 ] ()
    | Some axis_label -> t ~output_axes:[ (axis_label, 1) ] ()
  in
  Tn.update_memory_mode t.value Effectively_constant 24;
  t

let ndarray ?(label = []) ?(grad_spec = Prohibit_grad) ?batch_dims ?input_dims ?output_dims ?batch_axes
    ?input_axes ?output_axes ?(strict = true) values =
  let to_dim_list dims axes =
    Option.value ~default:[] @@ Option.first_some dims @@ Option.map axes ~f:(List.map ~f:snd)
  in
  let batch_ds = to_dim_list batch_dims batch_axes in
  let output_ds = to_dim_list output_dims output_axes in
  let input_ds = to_dim_list input_dims input_axes in
  let op_label =
    Stdlib.Format.pp_set_geometry Stdlib.Format.str_formatter ~max_indent:!max_sublabel_length
      ~margin:(!max_sublabel_length * 2);
    let dims = Array.concat_map [| batch_ds; output_ds; input_ds |] ~f:Array.of_list in
    let ndarr = Nd.create_array Arrayjit.Ops.double ~dims (Constant_fill { values; strict }) in
    let ( ! ) = List.length in
    Nd.pp_array_inline ~num_batch_axes:!batch_ds ~num_output_axes:!output_ds ~num_input_axes:!input_ds
      Stdlib.Format.str_formatter ndarr;
    Stdlib.Format.flush_str_formatter ()
  in
  let op_label =
    if String.contains op_label '\n' then
      "c" ^ Idx.dims_to_string @@ Array.concat_map [| batch_ds; output_ds; input_ds |] ~f:Array.of_list
    else op_label
  in
  let label = op_label :: label in
  let batch_dims = Option.first_some batch_dims @@ Option.some_if (Option.is_none batch_axes) [] in
  let input_dims = Option.first_some input_dims @@ Option.some_if (Option.is_none input_axes) [] in
  let output_dims = Option.first_some output_dims @@ Option.some_if (Option.is_none output_axes) [] in
  let t =
    term ~label ~grad_spec ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes
      ~deduced:Not_constrained
      ~init_op:(Constant_fill { values; strict })
      ()
  in
  Tn.update_memory_mode t.value Effectively_constant 24;
  t

let param ?input_dims ?output_dims ?input_axes ?output_axes ?deduced ?(strict = false) ?values label =
  let init_op =
    match values with
    | Some values -> Arrayjit.Ops.Constant_fill { values; strict }
    | None -> Standard_uniform
  in
  let t =
    term ~label:[ label ] ~grad_spec:Require_grad ~batch_dims:[] ?input_dims ?output_dims ?input_axes
      ?output_axes ?deduced ~init_op ()
  in
  let v = t.value in
  (* It is convenient to use the param syntax for volatiles (mutable inputs). *)
  Tn.update_memory_mode v (Hosted Nonconstant) 24;
  (* In principle, gradients can even be local, if a single jitted block does forward, backprop,
     and update computations. Use-cases needing [Materialized] gradients need to request that
     before any jitting. *)
  let g = (Option.value_exn t.diff).grad in
  Tn.update_memory_mode g Never_virtual 26;
  t

let rec iter_embedded_arrays ~f t =
  f t.value;
  Option.iter t.diff ~f:(fun diff -> f diff.grad);
  List.iter ~f:(fun ch -> if ch.embedded then iter_embedded_arrays ~f ch.subtensor) t.children

(** *** Printing. *** *)

(** Converts ID, label and the dimensions of a node to a string. *)
let header t =
  let v_dims_s = Tn.dims_to_string t.value in
  let g_dims_s = match t.diff with None -> "<no-grad>" | Some diff -> Tn.dims_to_string diff.grad in
  let dims_s =
    if String.equal v_dims_s g_dims_s then "dims " ^ v_dims_s
    else "dims val " ^ v_dims_s ^ " grad " ^ g_dims_s
  in
  "#" ^ Int.to_string t.id ^ " " ^ Tn.label t.value ^ " " ^ dims_s ^ " ["
  ^ String.concat ~sep:"," (List.map t.children ~f:(fun { subtensor = { id; _ }; _ } -> Int.to_string id))
  ^ "]"
(*^" "^PrintBox_text.to_string (PrintBox.Simple.to_box v.label)*)

let lazy_optional_payload ~present ~missing v =
  if Lazy.is_val v then
    match Lazy.force v with
    | Some p -> present p
    | None -> `Vlist (false, [ `Text (missing ()); `Text "<void>" ])
  else `Vlist (false, [ `Text (missing ()); `Text "<not-in-yet> " ])

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

let to_dag ?(single_node = false) ?entries_per_axis ~with_shape ~with_id ~with_value ~with_grad t =
  let rec to_dag { subtensor = t; embedded } : PrintBox_utils.dag =
    let id = Int.to_string t.id in
    let children = if single_node then [] else List.map ~f:to_dag t.children in
    let indices = Shape.default_display_indices t.shape in
    let labels = Shape.to_labels t.shape in
    let where_located a =
      match a.Tn.memory_mode with
      | None -> "<waiting>"
      | Some (m, prov) -> [%string "<%{Sexp.to_string_hum @@ Tn.sexp_of_memory_mode m} %{prov#Int}>"]
    in
    let txt =
      if with_id then "#" ^ id ^ " " ^ Tn.label t.value (* ^ " DEBUG: " ^ where_located t.value *)
      else Tn.label t.value
    in
    let grad_txt diff =
      let label = Tn.label diff.grad in
      let label =
        if String.is_substring (String.lowercase label) ~substring:"grad" then label else label ^ " Gradient"
      in
      if with_id then
        "#" ^ Int.to_string diff.grad.id ^ " " ^ label (* ^ " DEBUG: " ^ where_located diff.grad *)
      else label
    in
    let add_shape nodes =
      if with_shape then
        let shape = `Box (PrintBox.asprintf "%a" Sexp.pp_hum ([%sexp_of: Shape.t] t.shape)) in
        `Vlist (false, nodes @ [ shape ])
      else `Vlist (false, nodes)
    in
    match (not embedded, with_value, with_grad, t.diff) with
    | true, _, _, _ -> `Embed_subtree_ID (Int.to_string t.id)
    | _, false, false, _ | _, false, true, None ->
        `Subtree_with_ID (id, `Tree (add_shape [ `Text txt ], children))
    | _, true, false, _ | _, true, true, None ->
        let node =
          lazy_optional_payload t.value.array
            ~present:(fun v_array ->
              `Box (Nd.render_array ~brief:true ~prefix:txt ?entries_per_axis ~labels ~indices v_array))
            ~missing:(fun () -> txt ^ " " ^ where_located t.value)
        in
        `Subtree_with_ID (id, `Tree (add_shape [ node ], children))
    | _, false, true, Some diff ->
        let prefix = grad_txt diff in
        let node =
          match Lazy.force diff.grad.array with
          | Some g_array ->
              `Box (Nd.render_array ~brief:true ~prefix ?entries_per_axis ~labels ~indices g_array)
          | None -> `Text (prefix ^ " " ^ where_located diff.grad)
        in
        `Subtree_with_ID (id, `Tree (add_shape [ node ], children))
    | _, true, true, Some diff ->
        let node =
          let value =
            lazy_optional_payload t.value.array
              ~present:(fun v_array ->
                `Box (Nd.render_array ~brief:true ~prefix:txt ?entries_per_axis ~labels ~indices v_array))
              ~missing:(fun () -> txt ^ " " ^ where_located t.value)
          in
          let grad =
            lazy_optional_payload diff.grad.array
              ~present:(fun g_array ->
                `Box
                  (Nd.render_array ~brief:true ~prefix:(grad_txt diff) ?entries_per_axis ~labels ~indices
                     g_array))
              ~missing:(fun () -> grad_txt diff ^ " " ^ where_located diff.grad)
          in
          `Vlist (false, [ value; grad ])
        in
        `Subtree_with_ID (id, `Tree (add_shape [ node ], children))
  in
  to_dag { subtensor = t; embedded = true }

let to_printbox ?single_node ?entries_per_axis ?(with_id = false) ?(with_shape = false) ?(with_value = true)
    ~with_grad ~depth t =
  to_dag ?single_node ?entries_per_axis ~with_id ~with_shape ~with_value ~with_grad t
  |> PrintBox_utils.reformat_dag depth

let print ~with_grad ~with_code ?(with_low_level = false) (style : array_print_style) t =
  let sh = t.shape in
  let label = Tn.label t.value in
  let prefix =
    "[" ^ Int.to_string t.id ^ "]: " ^ label ^ " shape "
    ^ Shape.to_string_hum ~style:`Axis_number_and_size sh
    ^ " "
  in
  let grad_txt diff =
    let label = Tn.label diff.grad in
    if String.is_substring (String.lowercase label) ~substring:"grad" then label else label ^ " Gradient"
  in
  let labels = Shape.to_labels t.shape in
  let indices =
    match style with
    | `Default -> Shape.default_display_indices sh
    | `N5_layout priorities ->
        let f : (string, int) Either.t -> int = function
          | Either.Second i -> i
          | First _ -> invalid_arg "`N5_layout requires integer-only labels"
        in
        let p_labels = Shape.(axis_labels @@ axis_labels_of_spec priorities) in
        (Shape.axis_map_to_dims_index p_labels : (string, int) Either.t array) |> Array.map ~f
    | `Label_layout label_idcs ->
        let inv_labels =
          Array.mapi labels ~f:(fun i l -> (l, i)) |> Array.to_list |> Map.of_alist (module String)
        in
        let inv_labels =
          match inv_labels with
          | `Duplicate_key l -> raise @@ Session_error ("`Label_layout found a repeating label: " ^ l, Some t)
          | `Ok inv_labels -> inv_labels
        in
        let result = Array.create ~len:(Array.length labels) 0 in
        List.iter label_idcs ~f:(fun (l, priority) ->
            match Map.find inv_labels l with
            | Some pos -> result.(pos) <- priority
            | None -> raise @@ Session_error ("`Label_layout label not found in shape: " ^ l, Some t));
        result
    | `Inline -> [||]
  in
  let needs_spec =
    Array.exists ~f:(Fn.non String.is_empty) labels
    || Shape.(List.exists ~f:Row.(equal_dim @@ get_dim ~d:1 ()) sh.input.dims)
  in
  let axes_spec = if needs_spec then Some (Shape.to_string_hum ~style:`Only_labels sh) else None in
  let num_batch_axes = List.length sh.batch.dims in
  let num_input_axes = List.length sh.input.dims in
  let num_output_axes = List.length sh.output.dims in
  (* TODO: code sharing with [to_dag] *)
  (if not @@ Lazy.is_val t.value.array then Stdlib.Format.printf "%s <not-in-yet>@ " prefix
   else
     match (style, t.value.array) with
     | `Inline, (lazy None) -> Stdlib.Format.printf "<virtual>@ "
     | `Inline, (lazy (Some arr)) ->
         Nd.pp_array_inline (Stdlib.Format.get_std_formatter ()) ~num_batch_axes ~num_input_axes
           ~num_output_axes ?axes_spec arr
     | _, (lazy None) -> Stdlib.Format.printf "<virtual>@ "
     | _, (lazy (Some arr)) ->
         Nd.pp_array (Stdlib.Format.get_std_formatter ()) ~prefix ~labels ~indices arr;
         Stdlib.Format.print_newline ());
  if with_grad then
    Option.iter t.diff ~f:(fun diff ->
        if not @@ Lazy.is_val diff.grad.array then Stdlib.Format.printf "%s <not-in-yet>@ " (grad_txt diff)
        else
          match (style, diff.grad.array) with
          | `Inline, (lazy (Some arr)) ->
              Nd.pp_array_inline (Stdlib.Format.get_std_formatter ()) ~num_batch_axes ~num_input_axes
                ~num_output_axes ?axes_spec arr;
              Stdlib.Format.print_newline ()
          | _, (lazy (Some arr)) ->
              Nd.pp_array (Stdlib.Format.get_std_formatter ())
                ~prefix:(prefix ^ " " ^ grad_txt diff)
                ~labels ~indices arr;
              Stdlib.Format.print_newline ()
          | _, (lazy None) -> Stdlib.Format.printf "%s <virtual>@ " (grad_txt diff));
  if with_code then (
    (match t.forward with
    | Noop -> ()
    | fwd_code -> Stdlib.Format.printf "Current forward body:@ %a@ " (Asgns.fprint_hum ()) fwd_code);
    match t.diff with
    | Some { backprop = Noop; _ } -> ()
    | Some { backprop = bwd_code; _ } ->
        Stdlib.Format.printf "Current backprop body:@ %a@ " (Asgns.fprint_hum ()) bwd_code
    | None -> ());
  if with_low_level then (
    (match t.forward with
    | Noop -> ()
    | fwd_code ->
        Stdlib.Format.printf "Current forward low-level body:@ %a@ " (Arrayjit.Low_level.fprint_hum ())
        @@ Asgns.to_low_level fwd_code);
    match t.diff with
    | Some { backprop = Noop; _ } -> ()
    | Some { backprop = bwd_code; _ } ->
        Stdlib.Format.printf "Current backprop low-level body:@ %a@ " (Arrayjit.Low_level.fprint_hum ())
        @@ Asgns.to_low_level bwd_code
    | None -> ());
  Stdlib.Format.printf "\n%!"

let print_forward_roots ~with_grad ~with_code (style : array_print_style) =
  List.iter (Map.to_alist ~key_order:`Increasing session_state.forward_roots) ~f:(fun (id, root) ->
      assert (id = root.id);
      print ~with_grad ~with_code style root)

let print_tree ?entries_per_axis ?(with_backend_info = false) ?(with_id = true) ?(with_shape = false)
    ?(with_value = true) ~with_grad ~depth t =
  (* FIXME: print backend info *)
  ignore with_backend_info;
  PrintBox_text.output Stdio.stdout @@ PrintBox_utils.dag_to_box @@ PrintBox_utils.boxify depth
  @@ to_dag ?entries_per_axis ~with_id ~with_shape ~with_value ~with_grad t

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
  Nd.(reset (Constant_fill { values; strict = false }) @@ Option.value_exn @@ Lazy.force t.value.array)

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

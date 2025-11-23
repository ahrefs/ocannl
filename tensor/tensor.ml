open Base
module Lazy = Utils.Lazy
module Nd = Ir.Ndarray
module Tn = Ir.Tnode
module Asgns = Ir.Assignments
module Idx = Ir.Indexing

type ndarray = Nd.t
type tn = Tn.t
type tn_set = Set.M(Tn).t
type asgns = Asgns.t
type comp = Asgns.comp
type fetch_op = Asgns.fetch_op
type projections = { projections_debug : string; projections : Ir.Indexing.projections Lazy.t }

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_TENSOR=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_TENSOR"]

type diff = { grad : (Tn.t[@sexp.opaque]); zero_grads : Asgns.t; backprop : Asgns.comp }
[@@deriving sexp_of]

module rec Self : sig
  type t = {
    params : (t, Self_comparator.comparator_witness) Set.t;
    forward : comp;
    diff : diff option;
    id : int;
    value : tn;
    top_down_prec : bool;
    shape : Shape.t;
    children : subtensor list;
  }
  [@@deriving sexp_of]

  and subtensor = { subtensor : t; embedded : bool }

  val compare : t -> t -> int
end = struct
  type t = {
    params : (t, Self_comparator.comparator_witness) Set.t;
    forward : comp;
    diff : diff option;
    id : int;
    value : tn;
    top_down_prec : bool;
    shape : Shape.t;
    children : subtensor list;
  }

  and subtensor = { subtensor : t; embedded : bool }

  let rec sexp_of_t t =
    Sexp.message "Tensor"
      [
        ("id", sexp_of_int t.id);
        ("label", [%sexp_of: string list] t.value.label);
        ("forward", [%sexp_of: Asgns.comp] t.forward);
        ("diff", [%sexp_of: diff option] t.diff);
        ("children", [%sexp_of: subtensor list] t.children);
      ]

  and sexp_of_subtensor ch =
    Sexp.message "child"
      [
        (if ch.embedded then ("", sexp_of_t ch.subtensor)
         else ("ref-id", sexp_of_int ch.subtensor.id));
      ]

  let compare t1 t2 = Int.compare t1.id t2.id
end

and Self_comparator : (Comparator.S with type t := Self.t) = Comparator.Make (Self)

include Self
include Self_comparator

module T = struct
  include Self
  include Self_comparator
end

let sexp_of_comparator_witness _ = Sexp.Atom "comparator_witness"

type session_state = {
  mutable next_id : int;
  mutable forward_roots : t Map.M(Int).t;
  mutable backprop_roots : t Map.M(Int).t;
}

let session_state =
  { next_id = 0; forward_roots = Map.empty (module Int); backprop_roots = Map.empty (module Int) }

let is_fwd_root t = Map.mem session_state.forward_roots t.id
let remove_fwd_root t = session_state.forward_roots <- Map.remove session_state.forward_roots t.id
let is_bprop_root t = Map.mem session_state.backprop_roots t.id

let remove_bprop_root t =
  session_state.backprop_roots <- Map.remove session_state.backprop_roots t.id

let with_unchanged_roots ~f =
  let fwd_roots = session_state.forward_roots in
  let bprop_roots = session_state.backprop_roots in
  let finally () =
    session_state.forward_roots <- fwd_roots;
    session_state.backprop_roots <- bprop_roots
  in
  Exn.protectx ~f ~finally ()

let iter_embedded ~f t =
  Set.iter ~f t.forward.embedded_nodes;
  Option.iter t.diff ~f:(fun diff -> Set.iter ~f diff.backprop.embedded_nodes)

let%debug7_sexp rec init_params ?skip (t : t) : Asgns.comp =
  let more_embedded = ref @@ Set.empty (module Tn) in
  let params : t list =
    Set.to_list t.params
    |> (match skip with
       | None -> Fn.id
       | Some skip -> List.filter ~f:(fun p -> not (Map.mem skip p.value)))
       (* Compare to ordered_ts in op -- we need to sort to avoid computed-after-use bugs! *)
    |> List.sort ~compare:(fun p1 p2 -> Int.ascending p1.id p2.id)
  in
  let asgns =
    Asgns.Block_comment
      ( "init params for " ^ Tn.debug_name t.value,
        List.fold_right params ~init:Asgns.Noop ~f:(fun param acc ->
            if Set.is_empty param.params then Asgns.Seq (param.forward.asgns, acc)
            else
              let comp = init_params ?skip param in
              more_embedded := Set.union !more_embedded comp.Asgns.embedded_nodes;
              Seq (Seq (comp.Asgns.asgns, param.forward.asgns), acc)) )
  in
  let embedded_nodes =
    List.fold params ~init:!more_embedded ~f:(fun acc p -> Set.add acc p.value)
  in
  { Asgns.asgns; embedded_nodes }

let initial_default_prec =
  Ir.Ops.prec_of_string (Utils.get_global_arg ~default:"single" ~arg_name:"default_prec")

let default_value_prec = ref initial_default_prec
let default_grad_prec = ref initial_default_prec

exception Session_error of string * t option [@@deriving sexp]

let session_error_printer = function
  | Session_error (msg, None) -> Some msg
  | Session_error (msg, Some m) ->
      Some [%string "For #%{m.id#Int} %{Tn.debug_name m.value}: %{msg}"]
  | _ -> None

let () = Stdlib.Printexc.register_printer session_error_printer
let lazy_to_dims shape = lazy (Shape.to_dims shape)

let fetch_zeros array shape =
  Asgns.Fetch { array; fetch_op = Constant 0.; dims = lazy_to_dims shape }

let max_sublabel_length = ref 25

let raw_binop ~initialize_neutral ~accum ~(t : t) ~(lhs_is_grad : bool) ~op ~(t1 : t)
    ~(rhs1_is_grad : bool) ~(rhs1_is_merge : bool) ~(t2 : t) ~rhs2_is_grad ~rhs2_is_merge ~logic :
    Asgns.t =
  let shape = t.shape in
  let shape_logic = Shape.Broadcast (logic, t1.shape, t2.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic; id = get_update_id () } in
  Shape.propagate_shapes local_shape_update;
  let projections_debug = Shape.logic_to_spec shape_logic in
  let projections =
    { projections_debug; projections = lazy (Shape.derive_projections local_shape_update) }
  in
  let lhs = if lhs_is_grad then (Option.value_exn ~here:[%here] t.diff).grad else t.value in
  let rhs1 = if rhs1_is_grad then (Option.value_exn ~here:[%here] t1.diff).grad else t1.value in
  let rhs1 = if rhs1_is_merge then Asgns.Merge_buffer rhs1 else Node rhs1 in
  let rhs2 = if rhs2_is_grad then (Option.value_exn ~here:[%here] t2.diff).grad else t2.value in
  let rhs2 = if rhs2_is_merge then Asgns.Merge_buffer rhs2 else Node rhs2 in
  Asgns.Accum_op
    {
      initialize_neutral;
      accum;
      lhs;
      rhs = Binop { op; rhs1; rhs2 };
      projections = projections.projections;
      projections_debug = projections.projections_debug;
    }

let raw_ternop ~initialize_neutral ~accum ~(t : t) ~(lhs_is_grad : bool) ~op ~(t1 : t)
    ~(rhs1_is_grad : bool) ~(rhs1_is_merge : bool) ~(t2 : t) ~rhs2_is_grad ~rhs2_is_merge ~(t3 : t)
    ~rhs3_is_grad ~rhs3_is_merge ~logic : Asgns.t =
  let shape = t.shape in
  let shape_logic = Shape.Broadcast_tern (logic, t1.shape, t2.shape, t3.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic; id = get_update_id () } in
  Shape.propagate_shapes local_shape_update;
  let projections_debug = Shape.logic_to_spec shape_logic in
  let projections =
    { projections_debug; projections = lazy (Shape.derive_projections local_shape_update) }
  in
  let lhs = if lhs_is_grad then (Option.value_exn ~here:[%here] t.diff).grad else t.value in
  let rhs1 = if rhs1_is_grad then (Option.value_exn ~here:[%here] t1.diff).grad else t1.value in
  let rhs1 = if rhs1_is_merge then Asgns.Merge_buffer rhs1 else Node rhs1 in
  let rhs2 = if rhs2_is_grad then (Option.value_exn ~here:[%here] t2.diff).grad else t2.value in
  let rhs2 = if rhs2_is_merge then Asgns.Merge_buffer rhs2 else Node rhs2 in
  let rhs3 = if rhs3_is_grad then (Option.value_exn ~here:[%here] t3.diff).grad else t3.value in
  let rhs3 = if rhs3_is_merge then Asgns.Merge_buffer rhs3 else Node rhs3 in
  Asgns.Accum_op
    {
      initialize_neutral;
      accum;
      lhs;
      rhs = Ternop { op; rhs1; rhs2; rhs3 };
      projections = projections.projections;
      projections_debug = projections.projections_debug;
    }

let raw_unop ~initialize_neutral ~accum ~(t : t) ~(lhs_is_grad : bool) ~op ~(t1 : t)
    ~(rhs_is_grad : bool) ~(rhs_is_merge : bool) ~logic =
  let shape = t.shape in
  let shape_logic = Shape.Transpose (logic, t1.shape) in
  let local_shape_update = Shape.{ shape; logic = shape_logic; id = get_update_id () } in
  Shape.propagate_shapes local_shape_update;
  let projections_debug = Shape.logic_to_spec shape_logic in
  let projections =
    { projections_debug; projections = lazy (Shape.derive_projections local_shape_update) }
  in
  let lhs = if lhs_is_grad then (Option.value_exn ~here:[%here] t.diff).grad else t.value in
  let rhs = if rhs_is_grad then (Option.value_exn ~here:[%here] t1.diff).grad else t1.value in
  let rhs = if rhs_is_merge then Asgns.Merge_buffer rhs else Node rhs in
  Asgns.Accum_op
    {
      initialize_neutral;
      accum;
      lhs;
      rhs = Unop { op; rhs };
      projections = projections.projections;
      projections_debug = projections.projections_debug;
    }

type grad_spec = Require_grad | Prohibit_grad | If_needed [@@deriving sexp, equal, variants]

let%track7_sexp op ~(label : string list) ?(ternary_op = Shape.Pointwise_tern)
    ?(compose_op = Shape.Pointwise_bin) ?(transpose_op = Shape.Pointwise_un) ?terminal_op ~op_asn
    ~grad_asn ?(grad_spec = If_needed) ?(top_down_prec = false) make_shape (orig_ts : t list) : t =
  List.iter orig_ts ~f:(fun t ->
      if t.id >= session_state.next_id then
        raise
        @@ Session_error
             ( [%string
                 "Tensor #%{t.id#Int} %{Tn.debug_name t.value} has an id greater than the last id \
                  #%{session_state.next_id - 1#Int} -- check your uses of \
                  Tensor.unsafe_reinitialize, if all your uses are valid, report this as a bug."],
               Some t ));
  (* The code needs to be included in the order it was computed due to potential non-tree DAGs. *)
  let ordered_ts = List.dedup_and_sort orig_ts ~compare:(fun t1 t2 -> Int.ascending t1.id t2.id) in
  let id : int = session_state.next_id in
  session_state.next_id <- session_state.next_id + 1;
  let _session_state_next_id : int = session_state.next_id in
  let shape = make_shape ~debug_name:(Tn.get_debug_name ~id ~label ()) ~id in
  (* Split subtensors by whether they use top-down precision inference *)
  let top_down_ts = List.filter ordered_ts ~f:(fun t -> t.top_down_prec) in
  let delayed_prec_for default get =
    if top_down_prec then
      (* For top-down precision, don't promote from inputs *)
      Tn.Default default
    else
      (* For bottom-up precision, only promote from non-top-down subtensors *)
      let lazy_v_precs =
        List.filter_map ordered_ts ~f:(fun ti ->
            Option.map (get ti) ~f:(fun v ->
                if ti.top_down_prec then lazy (Tn.get_specified_prec v)
                else lazy (Some (Lazy.force v.prec))))
      in
      Tn.Inferred
        (lazy
          (List.filter_map lazy_v_precs ~f:Lazy.force
          |> List.reduce ~f:Ir.Ops.promote_prec
          |> Option.value ~default))
  in
  let delayed_prec = delayed_prec_for !default_value_prec (fun t -> Some t.value) in
  let terminal_logic () =
    let open Shape in
    (* Note: parameters will get their terminal logic set via Shape.set_terminal. *)
    let is_param = false in
    match terminal_op with
    | None -> Terminal { is_param; logic = Fetch (Asgns.Constant 0.0) }
    | Some (Fetch fetch_op) -> Terminal { is_param; logic = Fetch fetch_op }
    | Some (Data init_data) -> Terminal { is_param; logic = Data init_data }
  in
  let dims = lazy_to_dims shape in
  let padding = lazy (Shape.to_padding shape) in
  let v =
    match terminal_op with
    | Some (Shape.Data (Asgns.Reshape data)) ->
        Tn.create_with_reshape ~id ~label ~dims ~padding ~from_padded:false ~base_ndarray:data ()
    | Some (Shape.Data (Asgns.Keep_shape_no_padding data)) ->
        Tn.create_from_padded ~id ~label ~ndarray:data ~padding:None ()
    | Some (Shape.Data (Asgns.Padded { data; padding = padding_spec; padded_value })) ->
        let padding = Some (padding_spec, padded_value) in
        Tn.create_from_padded ~id ~label ~ndarray:data ~padding ()
    | Some (Shape.Fetch _) | None -> Tn.create delayed_prec ~id ~label ~dims ~padding ()
  in
  let update_infer_prec tn prec =
    (* Instead of just checking prec, we cross-check with dims (needed for code generation), to
       catch prec forcing bugs. *)
    if not (Lazy.is_val tn.Tn.dims) then Tn.update_infer_prec tn prec
  in
  (* Apply delayed top-down precision updates to parameter subtensors *)
  List.iter top_down_ts ~f:(fun ti -> update_infer_prec ti.value v.Tn.prec);
  let transpose_op =
    match transpose_op with
    | Uint4x32_to_prec _ -> Shape.Uint4x32_to_prec v.Tn.prec
    | _ -> transpose_op
  in
  let shape_logics = function
    | [] -> [ terminal_logic () ]
    | [ t1 ] -> [ Shape.Transpose (transpose_op, t1.shape) ]
    | [ t1; t2 ] -> [ Shape.Broadcast (compose_op, t1.shape, t2.shape) ]
    | [ t1; t2; t3 ] -> [ Shape.Broadcast_tern (ternary_op, t1.shape, t2.shape, t3.shape) ]
    | _ ->
        (* Let's implement what we need when we need it. *)
        assert false
  in
  let local_shape_updates =
    List.map ~f:(fun logic -> Shape.{ shape; logic; id = get_update_id () }) @@ shape_logics orig_ts
  in
  List.iter ~f:Shape.propagate_shapes local_shape_updates;
  let shape_update = List.hd_exn local_shape_updates in
  let projections_debug = Shape.logic_to_spec shape_update.logic in
  let projections =
    { projections_debug; projections = lazy (Shape.derive_projections shape_update) }
  in
  let embedded_nodes = ref @@ Set.singleton (module Tn) v in
  let children =
    List.folding_map orig_ts
      ~init:(Set.empty (module Int))
      ~f:(fun used ti ->
        let embedded = is_fwd_root ti && not (Set.mem used ti.id) in
        if embedded then
          embedded_nodes := Set.add (Set.union !embedded_nodes ti.forward.embedded_nodes) ti.value;
        (Set.add used ti.id, { subtensor = ti; embedded }))
  in
  let params = Set.union_list (module T) @@ List.map ordered_ts ~f:(fun ti -> ti.params) in

  let t =
    {
      params;
      forward = Asgns.empty_comp;
      diff = None;
      id;
      value = v;
      top_down_prec;
      shape;
      children;
    }
  in
  let fwds =
    List.filter_map ordered_ts ~f:(fun ti -> if is_fwd_root ti then Some ti.forward else None)
  in
  let forward = Asgns.sequence @@ fwds @ [ op_asn ~t ~projections ] in
  let forward =
    Asgns.
      { asgns = forward.asgns; embedded_nodes = Set.union forward.embedded_nodes !embedded_nodes }
  in
  List.iter ordered_ts ~f:(fun ti -> remove_fwd_root ti);
  let t = { t with forward } in
  if
    is_prohibit_grad grad_spec
    || Fn.non is_require_grad grad_spec
       && List.for_all orig_ts ~f:(fun ti -> Option.is_none ti.diff)
  then (
    session_state.forward_roots <- Map.add_exn session_state.forward_roots ~key:id ~data:t;
    t)
  else
    let get ti = Option.map ti.diff ~f:(fun d -> d.grad) in
    let delayed_prec = delayed_prec_for !default_grad_prec get in
    let grad_id = session_state.next_id in
    session_state.next_id <- session_state.next_id + 1;
    let g =
      Tn.create delayed_prec ~id:grad_id ~label:("grad" :: label) ~dims
        ~padding:(lazy (Shape.to_padding shape))
        ()
    in
    (* Apply delayed top-down precision updates to parameter gradient subtensors *)
    List.iter top_down_ts ~f:(fun ti ->
        Option.iter ti.diff ~f:(fun d -> update_infer_prec d.grad g.Tn.prec));
    let is_bck_root ti = Map.mem session_state.backprop_roots ti.id in
    let zero_grads =
      let zero_g ti =
        Option.value_map ti.diff ~default:Asgns.Noop ~f:(fun diff -> diff.zero_grads)
      in
      let zeros =
        List.map ordered_ts ~f:(fun ti -> if is_bck_root ti then zero_g ti else Asgns.Noop)
      in
      Asgns.sequential @@ zeros @ [ fetch_zeros g shape ]
    in
    let embedded_nodes = ref @@ Set.singleton (module Tn) g in
    (* The code needs to be included in the reverse order to which it was computed. This guarantees
       that all ancestors of a node are backpropagated before the node is backpropagated, even for
       non-tree DAGs. For repeating subtensors, the first one to-be-included should actually be
       included! That's why we reverse the order up-front prior to processing. *)
    let ordered_ts = List.rev ordered_ts in
    let bprop ti =
      Option.map ti.diff ~f:(fun diff ->
          embedded_nodes :=
            Set.add (Set.union !embedded_nodes diff.backprop.embedded_nodes) diff.grad;
          diff.backprop)
    in
    let bcks =
      List.filter_map ordered_ts ~f:(fun ti ->
          if is_bck_root ti && not (Set.mem t.params ti) then bprop ti else None)
    in
    let diff = Some { grad = g; zero_grads; backprop = Asgns.empty_comp } in
    let t = { t with diff } in
    let backprop = Asgns.sequence @@ (grad_asn ~t ~g ~projections :: bcks) in
    let backprop =
      {
        Asgns.asgns = backprop.asgns;
        embedded_nodes = Set.union backprop.embedded_nodes !embedded_nodes;
      }
    in
    List.iter ordered_ts ~f:(fun ti ->
        session_state.backprop_roots <- Map.remove session_state.backprop_roots ti.id);
    let diff = Some { grad = g; zero_grads; backprop } in
    let t = { t with diff } in
    session_state.forward_roots <- Map.add_exn session_state.forward_roots ~key:id ~data:t;
    session_state.backprop_roots <- Map.add_exn session_state.backprop_roots ~key:id ~data:t;
    t

type param_op_fun =
  ?input_dims:int list ->
  ?output_dims:int list ->
  ?input_axes:(string * int) list ->
  ?output_axes:(string * int) list ->
  ?deduced:Shape.deduce_within_shape ->
  unit ->
  t

type op_fun =
  ?label:string list ->
  ?top_down_prec:bool ->
  ?batch_dims:int list ->
  ?batch_axes:(string * int) list ->
  param_op_fun

let%track7_sexp binop ?op_label ?compose_op ~op_asn ~grad_asn ?grad_spec t1 t2 ?(label = [])
    ?top_down_prec ?batch_dims ?batch_axes ?input_dims ?output_dims ?input_axes ?output_axes
    ?deduced () : t =
  let op_asn ~t ~projections = op_asn ~t ~t1 ~t2 ~projections in
  let grad_asn ~t ~g ~projections = grad_asn ~t ~g ~t1 ~t2 ~projections in
  op
    ~label:(Option.to_list op_label @ label)
    ?compose_op ?transpose_op:None ~op_asn ~grad_asn ?grad_spec ?top_down_prec
    (Shape.make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes ?deduced
       ())
    [ t1; t2 ]

let%track7_sexp ternop ?op_label ?ternary_op ~op_asn ~grad_asn ?grad_spec t1 t2 t3 ?(label = [])
    ?top_down_prec ?batch_dims ?batch_axes ?input_dims ?output_dims ?input_axes ?output_axes
    ?deduced () : t =
  let op_asn ~t ~projections = op_asn ~t ~t1 ~t2 ~t3 ~projections in
  let grad_asn ~t ~g ~projections = grad_asn ~t ~g ~t1 ~t2 ~t3 ~projections in
  op
    ~label:(Option.to_list op_label @ label)
    ?ternary_op ?compose_op:None ~op_asn ~grad_asn ?grad_spec ?top_down_prec
    (Shape.make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes ?deduced
       ())
    [ t1; t2; t3 ]

let%track7_sexp unop ?op_label ?transpose_op ~op_asn ~grad_asn ?grad_spec t1 ?(label = [])
    ?top_down_prec ?batch_dims ?batch_axes ?input_dims ?output_dims ?input_axes ?output_axes
    ?deduced () : t =
  let op_asn ~t ~projections = op_asn ~t ~t1 ~projections in
  let grad_asn ~t ~g ~projections = grad_asn ~t ~g ~t1 ~projections in
  op
    ~label:(Option.to_list op_label @ label)
    ?compose_op:None ?transpose_op ~op_asn ~grad_asn ?grad_spec ?top_down_prec
    (Shape.make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes ?deduced
       ())
    [ t1 ]

let%track7_sexp term ?init_data ?fetch_op ?grad_spec ?(label = []) ?(top_down_prec = true)
    ?batch_dims ?batch_axes ?input_dims ?output_dims ?input_axes ?output_axes ?deduced () : t =
  let terminal_op =
    match (init_data, fetch_op) with
    | Some _, Some _ -> invalid_arg "Tensor.term: both init_data and fetch_op are provided"
    | Some init_data, None -> Some (Shape.Data init_data)
    | None, Some fetch_op -> Some (Shape.Fetch fetch_op)
    | None, None -> None
  in
  let op_asn ~t ~projections =
    let open Asgns in
    let dims = lazy (Lazy.force projections.projections).Idx.lhs_dims in
    match fetch_op with
    | None -> Asgns.empty_comp
    | Some
        (( Constant _ | Constant_bits _ | Slice _ | Embed_symbol _ | Embed_dim _ | Embed_self_id
         | Range_over_offsets | Constant_fill _ ) as fetch_op) ->
        Asgns.to_comp @@ Fetch { array = t.value; fetch_op; dims }
  in
  let grad_asn ~t:_ ~g:_ ~projections:_ = Asgns.empty_comp in
  let make_shape =
    Shape.make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes ?deduced ()
  in
  let grad_spec = Option.value grad_spec ~default:If_needed in
  (* Note: terminal_op is used for both tensor creation and shape inference. *)
  op ~label ?compose_op:None ?transpose_op:None ?terminal_op ~op_asn ~grad_asn ~grad_spec
    ~top_down_prec make_shape []

let float_to_label v = Float.to_string v |> String.chop_suffix_if_exists ~suffix:"."

let%track7_sexp number ?(label = []) ?axis_label ?(grad_spec = Prohibit_grad) c : t =
  (* Note: no axis label so that we do not conflict with user labels. *)
  let label = float_to_label c :: label in
  let fetch_op = Ir.Assignments.Constant c in
  let t = term ~label ~grad_spec ~batch_dims:[] ~input_dims:[] ~fetch_op in
  let t =
    match axis_label with
    | None -> t ~output_dims:[ 1 ] ()
    | Some axis_label -> t ~output_axes:[ (axis_label, 1) ] ()
  in
  Tn.update_memory_mode t.value Effectively_constant 24;
  (* FIXME: make this always pick a matching precision. *)
  Ir.Ops.(
    if exceeds_fp16_cutoff c then Tn.update_infer_prec ~only_if:is_up_to_fp16 t.value (lazy single));
  t

let%track7_sexp bits ?(label = []) ?axis_label ?(grad_spec = Prohibit_grad) i : t =
  (* Use Constant_bits for exact bit representation, primarily for uint4x32 *)
  let label = Int64.to_string i :: label in
  let fetch_op = Ir.Assignments.Constant_bits i in
  let t = term ~label ~grad_spec ~batch_dims:[] ~input_dims:[] ~fetch_op in
  let t =
    match axis_label with
    | None -> t ~output_dims:[ 1 ] ()
    | Some axis_label -> t ~output_axes:[ (axis_label, 1) ] ()
  in
  Tn.update_memory_mode t.value Effectively_constant 24;
  t

let constant_fill ~debug values =
  match Array.length values with
  | 0 -> (None, None)
  | 1 -> (None, Some (Asgns.Constant values.(0)))
  | n
    when n
         <= Int.of_string @@ Utils.get_global_arg ~default:"16" ~arg_name:"limit_constant_fill_size"
    ->
      (None, Some (Asgns.Constant_fill values))
  | _ ->
      let nd =
        Nd.create_array ~debug ~dims:[| Array.length values |] ~padding:None !default_value_prec
      in
      Nd.set_flat_values nd values;
      (Some (Asgns.Reshape nd), None)

let ndarray ?(grad_spec = Prohibit_grad) values ?(label = []) ?top_down_prec ?batch_dims ?batch_axes
    ?input_dims ?output_dims ?input_axes ?output_axes ?deduced () =
  let num_label =
    String.concat ~sep:"," @@ List.map ~f:float_to_label @@ Fn.flip List.take 5
    @@ Array.to_list values
  in
  let num_label = if Array.length values > 5 then num_label ^ "..." else num_label in
  let label = num_label :: label in
  let batch_dims = Option.first_some batch_dims @@ Option.some_if (Option.is_none batch_axes) [] in
  let input_dims = Option.first_some input_dims @@ Option.some_if (Option.is_none input_axes) [] in
  let output_dims =
    Option.first_some output_dims @@ Option.some_if (Option.is_none output_axes) []
  in
  let init_data, fetch_op = constant_fill ~debug:"Tensor.ndarray" values in
  (* Ideally, while the shape is known, the deduced argument will be used for verification. *)
  let t =
    term ?init_data ?fetch_op ~grad_spec ?batch_dims ?batch_axes ~label ?top_down_prec ?input_dims
      ?output_dims ?input_axes ?output_axes ?deduced ()
  in
  Tn.update_memory_mode t.value Effectively_constant 24;
  let max_abs = Array.fold values ~init:0. ~f:(fun acc v -> Float.(max acc @@ abs v)) in
  Ir.Ops.(
    if exceeds_fp16_cutoff max_abs then
      Tn.update_infer_prec ~only_if:is_up_to_fp16 t.value (lazy single));
  t

let term_init ?grad_spec values ?(label = []) ?top_down_prec ?batch_dims ?batch_axes ?input_dims
    ?output_dims ?input_axes ?output_axes ?deduced () =
  let init_data, fetch_op = constant_fill ~debug:"Tensor.term_init" values in
  term ?init_data ?fetch_op ?grad_spec ?batch_dims ?batch_axes ~label ?top_down_prec ?input_dims
    ?output_dims ?input_axes ?output_axes ?deduced ()

let%debug7_sexp param ~t (name : string) ?(more_label = []) ?input_dims ?output_dims ?input_axes
    ?output_axes ?deduced () : t =
  let t =
    t
      ?label:(Some (name :: more_label))
      ?top_down_prec:(Some true) ?batch_dims:(Some []) ?batch_axes:None ?input_dims ?output_dims
      ?input_axes ?output_axes ?deduced ()
  in
  let v = t.value in
  (* It is convenient to use the param syntax for volatiles (mutable embedded_nodes). *)
  Tn.update_memory_mode v (Hosted Nonconstant) 24;
  (* In principle, gradients can even be local, if a single jitted block does forward, backprop, and
     update computations. *)
  (match t.diff with
  | Some diff -> Tn.update_memory_mode diff.grad Never_virtual 26
  | None -> ());
  Shape.set_terminal ~is_param:(Option.is_some t.diff) t.shape;
  remove_fwd_root t;
  { t with params = Set.singleton (module T) t }

let debug_name t = Tn.debug_name t.value
let debug_grad t = Tn.debug_name (Option.value_exn t.diff).grad

let consume_forward_code t =
  if not @@ is_fwd_root t then
    raise
    @@ Session_error
         ( "Tensor.consume_forward_code: tensor is not a root for tnode: " ^ Tn.debug_name t.value
           ^ " (maybe you're trying to forward a param?)",
           Some t );
  (* Check if any non-embedded descendants of t are embedded in other roots *)
  let all_read = fst @@ Asgns.collect_nodes_guess_output t.forward.asgns in
  let non_embedded_descendants = Set.diff all_read t.forward.embedded_nodes in
  let other_roots =
    Map.data session_state.forward_roots |> List.filter ~f:(fun r -> r.id <> t.id)
  in
  let conflicting_roots =
    List.filter other_roots ~f:(fun root ->
        not (Set.is_empty (Set.inter non_embedded_descendants root.forward.embedded_nodes)))
  in
  if not @@ List.is_empty conflicting_roots then
    raise
    @@ Session_error
         ( [%string
             {|Tensor.consume_forward_code for %{debug_name t}:
found conflicting roots with shared non-embedded descendants: %{String.concat ~sep:", " @@ List.map ~f:debug_name conflicting_roots}|}],
           Some t );
  remove_fwd_root t;
  t.forward

let consume_backprop_code t =
  let diff =
    Option.value_or_thunk t.diff ~default:(fun () ->
        raise
        @@ Session_error
             ( "Tensor.consume_backprop_code: tensor is not differentiable for value: "
               ^ debug_name t,
               Some t ))
  in
  if not @@ is_bprop_root t then
    raise
    @@ Session_error
         ("Tensor.consume_backprop_code: tensor is not a root for tnode: " ^ debug_grad t, Some t);
  (* Check if any non-embedded grad descendants of t are embedded in other roots *)
  let all_read = fst @@ Asgns.collect_nodes_guess_output diff.backprop.asgns in
  let non_embedded_grad_descendants = Set.diff all_read diff.backprop.embedded_nodes in
  let other_roots =
    Map.data session_state.backprop_roots |> List.filter ~f:(fun r -> r.id <> t.id)
  in
  let conflicting_roots =
    List.filter other_roots ~f:(fun root ->
        match root.diff with
        | Some rdiff ->
            not
              (Set.is_empty (Set.inter non_embedded_grad_descendants rdiff.backprop.embedded_nodes))
        | None -> false)
  in
  if not @@ List.is_empty conflicting_roots then
    raise
    @@ Session_error
         ( [%string
             {|Tensor.consume_backprop_code for %{debug_grad t}:
found conflicting roots with shared non-embedded grad descendants: %{String.concat ~sep:", " @@ List.map ~f:debug_grad conflicting_roots}|}],
           Some t );
  remove_bprop_root t;
  diff.backprop

let random_seed = ref None

let set_random_seed ?seed () =
  let seed =
    Option.value ~default:42 @@ Option.first_some seed Utils.settings.fixed_state_for_init
  in
  let res = bits ~label:[ "random_seed" ] ~grad_spec:Prohibit_grad (Int64.of_int seed) in
  Tn.update_prec res.value Ir.Ops.uint4x32;
  random_seed := Some res

let rec get_random_seed () =
  match !random_seed with
  | Some res -> res
  | None ->
      set_random_seed ();
      get_random_seed ()

let%track5_sexp unsafe_reinitialize () : unit =
  session_state.next_id <- 0;
  session_state.forward_roots <- Map.empty (module Int);
  session_state.backprop_roots <- Map.empty (module Int);
  random_seed := None;
  Tn.Registry.clear Tn.registry;
  Shape.unsafe_reinitialize ()

let header t =
  let v_dims_s = Tn.dims_to_string t.value in
  let g_dims_s =
    match t.diff with None -> "<no-grad>" | Some diff -> Tn.dims_to_string diff.grad
  in
  let dims_s =
    if String.equal v_dims_s g_dims_s then "dims " ^ v_dims_s
    else "dims val " ^ v_dims_s ^ " grad " ^ g_dims_s
  in
  "#" ^ Int.to_string t.id ^ " " ^ Tn.label t.value ^ " " ^ dims_s ^ " ["
  ^ String.concat ~sep:","
      (List.map t.children ~f:(fun { subtensor = { id; _ }; _ } -> Int.to_string id))
  ^ "]"
(*^" "^PrintBox_text.to_string (PrintBox.Simple.to_box v.label)*)

let lazy_optional_payload ~force ~present ~missing v =
  if Lazy.is_val v.Tn.array || force then (
    Tn.do_read v;
    match Lazy.force v.array with
    | Some p -> present p
    | None -> `Vlist (false, [ `Text (missing ()); `Text "<void>" ]))
  else `Vlist (false, [ `Text (missing ()); `Text "<not-in-yet> " ])

type array_print_style =
  [ `Default | `Inline | `Label_layout of (string * int) list | `N5_layout of string ]
[@@deriving sexp_of]

let%debug5_sexp to_dag ?(single_node = false) ?(embedded_only = false) ?entries_per_axis ~force
    ~with_shape ~with_id ~with_value ~with_grad t =
  (* First scan to identify which tensors appear embedded anywhere *)
  let tensors_with_embedded_occurrence = Hash_set.create (module Int) in
  let rec scan_for_embedded { subtensor = t; embedded } =
    if embedded then Hash_set.add tensors_with_embedded_occurrence t.id;
    if not single_node then List.iter ~f:scan_for_embedded t.children
  in
  if not embedded_only then scan_for_embedded { subtensor = t; embedded = true };

  let visited = if embedded_only then None else Some (Hash_set.create (module Int)) in
  let rec to_dag { subtensor = t; embedded } : PrintBox_utils.dag =
    let id = Int.to_string t.id in
    let children = if single_node then [] else List.map ~f:to_dag t.children in
    let indices = Shape.default_display_indices t.shape in
    let labels = Shape.to_labels t.shape in
    let where_located a = Tn.(debug_memory_mode a.memory_mode) in
    let txt =
      if with_id then "#" ^ id ^ " " ^ Tn.label t.value (* ^ " DEBUG: " ^ where_located t.value *)
      else Tn.label t.value
    in
    let grad_txt diff =
      let label = Tn.label diff.grad in
      assert (String.is_prefix label ~prefix:"grad");
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
    let should_elide =
      if embedded_only then not embedded
      else if
        (* If this tensor appears embedded anywhere, use embedded logic for consistency *)
        Hash_set.mem tensors_with_embedded_occurrence t.id
      then not embedded
      else
        (* Only use visited tracking for tensors that are never embedded anywhere *)
        match visited with
        | None -> not embedded
        | Some visited_set ->
            if Hash_set.mem visited_set t.id then true
            else (
              Hash_set.add visited_set t.id;
              false)
    in
    let txt = txt ^ if (not should_elide) && not embedded then " non-emb" else "" in
    match (should_elide, with_value, with_grad, t.diff) with
    | true, _, _, _ -> `Embed_subtree_ID txt
    | _, false, false, _ | _, false, true, None ->
        `Subtree_with_ID (id, `Tree (add_shape [ `Text txt ], children))
    | _, true, false, _ | _, true, true, None ->
        let node =
          lazy_optional_payload t.value ~force
            ~present:(fun v_array ->
              Tn.do_read t.value;
              `Box
                (Nd.render_array ~brief:true ~prefix:txt ?entries_per_axis ~labels ~indices v_array))
            ~missing:(fun () -> txt ^ " " ^ where_located t.value)
        in
        `Subtree_with_ID (id, `Tree (add_shape [ node ], children))
    | _, false, true, Some diff ->
        let prefix =
          grad_txt diff ^ if (not should_elide) && not embedded then " non-emb" else ""
        in
        let node =
          if Lazy.is_val diff.grad.array then
            match Lazy.force diff.grad.array with
            | Some g_array ->
                Tn.do_read diff.grad;
                `Box
                  (Nd.render_array ~brief:true ~prefix ?entries_per_axis ~labels ~indices g_array)
            | None -> `Text (prefix ^ " " ^ where_located diff.grad)
          else `Text (prefix ^ " <not-in-yet> " ^ where_located diff.grad)
        in
        `Subtree_with_ID (id, `Tree (add_shape [ node ], children))
    | _, true, true, Some diff ->
        let node =
          let value =
            lazy_optional_payload t.value ~force
              ~present:(fun v_array ->
                Tn.do_read t.value;
                `Box
                  (Nd.render_array ~brief:true ~prefix:txt ?entries_per_axis ~labels ~indices
                     v_array))
              ~missing:(fun () -> txt ^ " " ^ where_located t.value)
          in
          let grad =
            lazy_optional_payload diff.grad ~force
              ~present:(fun g_array ->
                Tn.do_read diff.grad;
                `Box
                  (Nd.render_array ~brief:true ~prefix:(grad_txt diff) ?entries_per_axis ~labels
                     ~indices g_array))
              ~missing:(fun () -> grad_txt diff ^ " " ^ where_located diff.grad)
          in
          `Vlist (false, [ value; grad ])
        in
        `Subtree_with_ID (id, `Tree (add_shape [ node ], children))
  in
  to_dag { subtensor = t; embedded = true }

let to_printbox ?single_node ?embedded_only ?entries_per_axis ?(with_id = false) ?(force = false)
    ?(with_shape = false) ?(with_value = true) ~with_grad ~depth t =
  to_dag ?single_node ?embedded_only ?entries_per_axis ~with_id ~force ~with_shape ~with_value
    ~with_grad t
  |> PrintBox_utils.reformat_dag depth

let%debug_sexp log_debug_info ~from_log_level t =
  [%diagn_sexp
    let%diagn_sexp log_child { subtensor = _subtensor; embedded = _embedded } =
      [%logN_block
        from_log_level
          ((if _embedded then "Embedded " else "Non-embedded ") ^ Tn.debug_name _subtensor.value);
        Tn.log_debug_info ~from_log_level _subtensor.value]
    in
    [%logN_block
      from_log_level ("Tensor " ^ Tn.dims_to_string t.value);
      Tn.log_debug_info ~from_log_level t.value;
      Option.iter t.diff ~f:(fun diff ->
          [%log_block
            "Gradient";
            Tn.log_debug_info ~from_log_level diff.grad]);
      List.iter ~f:log_child t.children]]

let%debug5_sexp to_doc ~force ~with_grad ~with_code ?(with_low_level = false)
    (style : array_print_style) t =
  let sh = t.shape in
  let label = Tn.label t.value in
  let prefix_str =
    "[" ^ Int.to_string t.id ^ "]: " ^ label ^ " shape "
    ^ Shape.to_string_hum ~style:Row.Axis_number_and_size sh
    ^ " "
  in
  let grad_txt diff =
    let label = Tn.label diff.grad in
    assert (String.is_prefix label ~prefix:"grad");
    label
  in
  let labels = Shape.to_labels t.shape in
  let indices =
    match style with
    | `Default -> Shape.default_display_indices sh
    | `N5_layout priorities ->
        let f : Shape.axis_spec -> int = function
          | Shape.Fixed_index i -> i
          | Shape.Label _ -> invalid_arg "`N5_layout requires integer-only labels"
          | Shape.Conv_spec _ -> invalid_arg "`N5_layout does not support conv expressions"
        in
        let p_labels = Shape.(axis_labels @@ axis_labels_of_spec priorities) in
        (Shape.axis_map_to_dims_index p_labels : Shape.axis_spec array) |> Array.map ~f
    | `Label_layout label_idcs ->
        let inv_labels =
          Array.mapi labels ~f:(fun i l -> (l, i)) |> Array.to_list |> Map.of_alist (module String)
        in
        let inv_labels =
          match inv_labels with
          | `Duplicate_key l ->
              raise @@ Session_error ("`Label_layout found a repeating label: " ^ l, Some t)
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
  let axes_spec =
    if needs_spec then Some (Shape.to_string_hum ~style:Row.Only_labels sh) else None
  in
  let num_batch_axes = List.length sh.batch.dims in
  let num_input_axes = List.length sh.input.dims in
  let num_output_axes = List.length sh.output.dims in

  let open PPrint in
  (* Create document for tensor value *)
  let has_grad = with_grad && Option.is_some t.diff in
  let value_doc =
    if (not force) && not (Lazy.is_val t.value.array) then
      string prefix_str ^^ string " <not-in-yet>" ^^ break 1
    else
      match (style, Lazy.force t.value.array) with
      | _, None ->
          string prefix_str ^^ string " <not-hosted>" ^^ if has_grad then break 1 else empty
      | `Inline, Some arr ->
          Tn.do_read t.value;
          string prefix_str ^^ space
          ^^ Nd.to_doc_inline ~num_batch_axes ~num_input_axes ~num_output_axes ?axes_spec arr
          ^^ if has_grad then break 1 else empty
      | _, Some arr ->
          Tn.do_read t.value;
          Nd.to_doc ~prefix:prefix_str ~labels ~indices arr ^^ if has_grad then break 1 else empty
  in

  (* Create document for gradient *)
  let grad_doc =
    if with_grad then
      match t.diff with
      | Some diff -> (
          if (not force) && not (Lazy.is_val diff.grad.array) then
            string (grad_txt diff) ^^ string " <not-in-yet>"
          else
            match Lazy.force diff.grad.array with
            | None -> string (grad_txt diff) ^^ string " <not-hosted>"
            | Some arr -> (
                match style with
                | `Inline ->
                    Tn.do_read diff.grad;
                    string (grad_txt diff)
                    ^^ space
                    ^^ Nd.to_doc_inline ~num_batch_axes ~num_input_axes ~num_output_axes ?axes_spec
                         arr
                | `Default | `N5_layout _ | `Label_layout _ ->
                    Tn.do_read diff.grad;
                    let prefix = prefix_str ^ " " ^ grad_txt diff in
                    Nd.to_doc ~prefix ~labels ~indices arr))
      | None -> empty
    else empty
  in

  (* Create document for code *)
  let code_doc =
    if with_code then
      let fwd_doc =
        match t.forward.asgns with
        | Noop -> empty
        | fwd_code ->
            group (string "Current forward body:" ^^ nest 2 (hardline ^^ Asgns.to_doc () fwd_code))
            ^^ hardline
      in
      let bwd_doc =
        match t.diff with
        | Some { backprop = { asgns = Noop; _ }; _ } -> empty
        | Some { backprop = { asgns = bwd_code; _ }; _ } ->
            group (string "Current backprop body:" ^^ nest 2 (hardline ^^ Asgns.to_doc () bwd_code))
            ^^ hardline
        | None -> empty
      in
      fwd_doc ^^ bwd_doc
    else empty
  in

  (* Create document for low-level code *)
  let low_level_doc =
    if with_low_level then
      let fwd_doc =
        match t.forward.asgns with
        | Noop -> empty
        | fwd_code ->
            group
              (string "Current forward low-level body:"
              ^^ nest 2 (hardline ^^ Ir.Low_level.to_doc () (Asgns.to_low_level fwd_code)))
            ^^ hardline
      in
      let bwd_doc =
        match t.diff with
        | Some { backprop = { asgns = Noop; _ }; _ } -> empty
        | Some { backprop = { asgns = bwd_code; _ }; _ } ->
            group
              (string "Current backprop low-level body:"
              ^^ nest 2 (hardline ^^ Ir.Low_level.to_doc () (Asgns.to_low_level bwd_code)))
            ^^ hardline
        | None -> empty
      in
      fwd_doc ^^ bwd_doc
    else empty
  in

  (* Combine all documents and print *)
  group
    (value_doc ^^ grad_doc
    ^^ (if is_empty value_doc && is_empty grad_doc then empty else hardline)
    ^^ code_doc ^^ low_level_doc)

let print ?here ?(force = false) ~with_grad ~with_code ?(with_low_level = false)
    (style : array_print_style) t =
  Option.iter here ~f:(fun here ->
      Stdio.printf "HERE: %s\n%!" (Source_code_position.to_string here));
  let doc = to_doc ~force ~with_grad ~with_code ~with_low_level style t in
  PPrint.ToChannel.pretty 0.7 100 Stdio.stdout doc;
  Stdio.Out_channel.flush Stdio.stdout

let print_forward_roots ~with_grad ~with_code (style : array_print_style) =
  List.iter (Map.to_alist ~key_order:`Increasing session_state.forward_roots) ~f:(fun (id, root) ->
      assert (id = root.id);
      print ~with_grad ~with_code style root)

let print_tree ?here ?(force = false) ?entries_per_axis ?(with_backend_info = false)
    ?(with_id = true) ?(with_shape = false) ?(with_value = true) ?embedded_only ~with_grad ~depth t
    =
  Option.iter here ~f:(fun here ->
      Stdio.printf "HERE: %s\n%!" (Source_code_position.to_string here));
  (* FIXME: print backend info *)
  ignore with_backend_info;
  PrintBox_text.output Stdio.stdout @@ PrintBox_utils.dag_to_box @@ PrintBox_utils.boxify depth
  @@ to_dag ?entries_per_axis ?embedded_only ~with_id ~force ~with_shape ~with_value ~with_grad t

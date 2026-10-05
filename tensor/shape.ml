(** {1 Tensor shape types, shape inference, projection inference.} *)

open Base
module Lazy = Utils.Lazy
module Idx = Ir.Indexing
module Tn = Ir.Tnode

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_SHAPE=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_SHAPE"]

(** {2 Shape types and inference.} *)

(* Re-export types from Einsum_parser (which includes Einsum_types) *)
include Einsum_parser

type padding = Row.axis_padding array option [@@deriving sexp, equal]

type t = {
  mutable batch : Row.t;
  mutable input : Row.t;
  mutable output : Row.t;
  mutable batch_padding : padding;
  mutable input_padding : padding;
  mutable output_padding : padding;
  mutable padding_elem : float option option;
      (** The padding element for this shape's tensors. [None] means "unknown" (not yet determined),
          [Some (Some v)] means all operations use neutral element [v], [Some None] means different
          operations require different neutral elements (margin must be reset before each
          operation). *)
  id : int;  (** A node that has the same shape as this shape. *)
  debug_name : string;
}
[@@deriving equal, fields, sexp]

let row_of_kind = function `Batch -> batch | `Input -> input | `Output -> output

type deduce_within_shape = Not_constrained | Input_equals_output
[@@deriving compare, sexp, variants]

type delayed_var_ref = {
  var_ref : Ir.Indexing.variable_ref;
  mutable var : [ `Row of Row.row_var | `Dim of Row.dim_var | `Not_set_yet ];
}
[@@deriving equal, sexp_of]

let get_variable_ref ref_label =
  { var_ref = Ir.Indexing.{ ref_label; solved_dim = None; solved_sym = None }; var = `Not_set_yet }

type compose_type =
  | Pointwise_bin
  | Compose
  | Einsum of string * delayed_var_ref list
  | Defined_by_cd_logic
[@@deriving sexp_of, equal]

type transpose_type =
  | Transpose
  | Pointwise_un
  | Permute of string * delayed_var_ref list
  | Batch_slice of Idx.static_symbol
  | Uint4x32_to_prec of Ir.Ops.prec Lazy.t
  | Defined_by_cd_logic
[@@deriving equal, sexp_of]

type terminal_type = Data of Ir.Assignments.init_data | Fetch of Ir.Assignments.fetch_op
[@@deriving equal, sexp_of]

type ternary_type =
  | Pointwise_tern
  | Compose_accumulate
  | Defined_by_cd_logic
  | Einsum_tern of string * delayed_var_ref list
[@@deriving sexp_of, equal]

let einsum_of_spec spec =
  try Einsum_parser.einsum_of_spec spec
  with Einsum_parser.Parse_error msg ->
    raise @@ Utils.User_error ("Shape.einsum_of_spec: while parsing: " ^ spec ^ " error: " ^ msg)

type logic =
  | Broadcast of compose_type * t * t
  | Transpose of transpose_type * t
  | Broadcast_tern of ternary_type * t * t * t
  | Terminal of { is_param : bool; logic : terminal_type }
  | Block of { spec : string; delayed_vars : delayed_var_ref list; rhses : t list }
[@@deriving equal, sexp_of]

let logic_to_spec = function
  | Broadcast (Pointwise_bin, _, _)
  | Transpose (Pointwise_un, _)
  | Broadcast_tern (Pointwise_tern, _, _, _) ->
      "."
  | Broadcast (Compose, _, _) | Broadcast_tern (Compose_accumulate, _, _, _) -> "@"
  | Broadcast (Einsum (spec, _), _, _)
  | Transpose (Permute (spec, _), _)
  | Block { spec; _ }
  | Broadcast_tern (Einsum_tern (spec, _), _, _, _) ->
      spec
  | Transpose (Transpose, _) -> "T"
  | Transpose (Batch_slice _, _) -> "@|"
  | Transpose (Uint4x32_to_prec _, _) -> "U4x32"
  | Broadcast (Defined_by_cd_logic, _, _)
  | Transpose (Defined_by_cd_logic, _)
  | Broadcast_tern (Defined_by_cd_logic, _, _, _) ->
      "<cd_logic>"
  | Terminal _ -> "<terminal>"

module Update_id = struct
  module T = struct
    type t = Update_id of int [@@deriving equal, compare, hash, sexp]
  end

  include T
  include Comparator.Make (T)
end

type update_id = Update_id.t [@@deriving equal, compare, hash, sexp]

let update_uid = ref 0

let get_update_id () =
  Int.incr update_uid;
  Update_id.Update_id !update_uid

type update_step = {
  shape : t;
  logic : logic;
  id : update_id;
  mutable unsafe_projections : Idx.projections option;
  neutral_elem : float option;
      (** The neutral element of the operation's accumulation, fixed at creation: [Some v] when all
          assignment ops in the update step use the same neutral element [v], [None] when different
          operations have different neutral elements or when there are no accumulator operations.
          [derive_projections] reads it for the clamped-window decision (gh-504) and to commit the
          margins' neutral ([padding_elem]), so a step is only created once its operation's
          assignments are known -- see [Tensor.op]. *)
}
[@@deriving sexp_of]
(** Data required for a shape inference update step. Ideally, an update should be performed at least
    twice, the second time after all the other relevant updates have been performed for the first
    time. In OCANNL, this is achieved by performing updates both as the tensors are constructed, and
    via lazy callbacks as the corresponding [Ir.Indexing] dimensions and projections are first
    accessed. *)

type Row.error_trace += Shape_mismatch of t list

let with_error_trace = ref true

(** Converts an axes-keyed map into three arrays of values: batch axes, input axes, output axes. If
    the map is incomplete, the result will likely be invalid: gaps in the array are filled with an
    arbitrary one of the provided values. *)
let axis_map_to_dims_bio (type a) ?(default : a option) (idcs : a axis_map) =
  if Map.is_empty idcs then (([||], [||], [||]), ([||], [||], [||]))
  else
    let witness =
      match default with Some witness -> witness | None -> snd @@ Map.min_elt_exn idcs
    in
    let bch_axes, other =
      Map.partition_mapi idcs ~f:(fun ~key:{ in_axes; _ } ~data ->
          if Row.is_batch in_axes then Either.First data else Either.Second data)
    in
    let inp_axes, out_axes =
      Map.partition_mapi other ~f:(fun ~key:{ in_axes; _ } ~data ->
          if Row.is_input in_axes then Either.First data else Either.Second data)
    in
    let make_row axes =
      let back_axes, front_axes =
        Map.to_alist axes
        |> List.partition_map ~f:(fun ({ AxisKey.from_end; pos = i; _ }, v) ->
            if from_end then Either.First (i, v) else Second (i, v))
      in
      let back_size = List.fold back_axes ~init:0 ~f:(fun accu (i, _) -> max i accu) in
      let front_size = List.fold front_axes ~init:0 ~f:(fun accu (i, _) -> max i accu) in
      let back = Array.create ~len:back_size witness in
      let front = Array.create ~len:front_size witness in
      List.iter back_axes ~f:(fun (i, v) -> back.(back_size - i) <- v);
      List.iter front_axes ~f:(fun (i, v) -> front.(i - 1) <- v);
      (back, front)
    in
    let bch, beg_bch = make_row bch_axes in
    let inp, beg_inp = make_row inp_axes in
    let out, beg_out = make_row out_axes in
    ((bch, inp, out), (beg_bch, beg_inp, beg_out))

(** Converts an axes-keyed map into an array of values using the [force_to_dims] semantics of axes.
    If the map is incomplete and the [~default] is not given, the result might be invalid: gaps in
    the array are filled with an arbitrary one of the provided values. *)
let axis_map_to_dims_index (type a) ?(default : a option) (idcs : a axis_map) : a array =
  let (bch, inp, out), (beg_bch, beg_inp, beg_out) = axis_map_to_dims_bio ?default idcs in
  Array.concat [ beg_bch; bch; beg_out; out; beg_inp; inp ]

let axes_spec_to_dims_bio ~sh_id ~row_var_env ~dim_var_env:_ ~f labels =
  let (b_dims, i_dims, o_dims), (beg_b_dims, beg_i_dims, beg_o_dims) =
    axis_map_to_dims_bio labels.labels
  in
  let to_dim kind = Array.(Fn.compose to_list @@ map ~f:(f kind)) in
  let to_bcast kind ~implicit v beg_dims =
    let beg_dims = to_dim kind beg_dims in
    Option.value_map v ~default:(Row.Broadcastable, beg_dims) ~f:(fun vname ->
        let v = Hashtbl.find_or_add row_var_env vname ~default:(fun () -> Row.get_row_var ()) in
        Row.add_used_in_spec_or_compose v;
        (* A row omitted together with its kind separator reads as the context ellipsis, but unlike
           an explicit ellipsis it may be silently closed to an empty row (even on parameters) when
           nothing else constrains it. *)
        if implicit then Row.add_safe_to_guess v;
        (Row.Row_var v, beg_dims))
  in
  let to_row kind ~implicit v dims beg_dims =
    let bcast, beg_dims = to_bcast kind ~implicit v beg_dims in
    { Row.beg_dims; dims = to_dim kind dims; bcast; prov = Row.provenance ~sh_id ~kind }
  in
  let batch = to_row `Batch ~implicit:labels.implicit_batch labels.bcast_batch b_dims beg_b_dims in
  let input = to_row `Input ~implicit:labels.implicit_input labels.bcast_input i_dims beg_i_dims in
  let output = to_row `Output ~implicit:false labels.bcast_output o_dims beg_o_dims in
  (batch, input, output)

let einsum_slot_spec_to_dims_bio ?(fixed_index = `Pin_var) ~original_spec ~sh_id ~row_var_env
    ~dim_var_env labels =
  let proj_env_update = ref @@ Row.dim_map_empty in
  let extras = ref [] in
  let f kind = function
    | Label name ->
        Row.Var (Hashtbl.find_or_add dim_var_env name ~default:(fun () -> Row.get_var ~name ()))
    | Fixed_index _ when Poly.equal fixed_index `Unit_dim ->
        (* Product-space proxies (gh-512): a fixed-index axis has a pinned projection, so it is
           never a product axis regardless of its extent — the proxy pins it to dimension 1. *)
        Row.get_default_dim ~d:1 ()
    | Fixed_index i ->
        let var = Row.get_var () in
        let d = Row.Var var in
        proj_env_update := Map.add_exn !proj_env_update ~key:var ~data:(Idx.Fixed_idx i);
        extras :=
          Row.Dim_constr
            {
              d;
              constr = At_least_dim (i + 1);
              origin =
                [
                  {
                    lhs_name = original_spec;
                    lhs_kind = kind;
                    rhs_name = "einsum_slot_spec_to_dims_bio";
                    rhs_kind = kind;
                    operation = Some "At_least_dim";
                  };
                ];
            }
          :: !extras;
        d
    | Affine_spec { stride; over_label; conv; stride_offset } ->
        let stride_int =
          try Int.of_string stride
          with _ -> failwith ("Invalid stride value (expected integer): " ^ stride)
        in
        let over_dim =
          Row.Var
            (Hashtbl.find_or_add dim_var_env over_label ~default:(fun () ->
                 Row.get_var ~name:over_label ()))
        in
        let conv =
          Option.map conv ~f:(fun { dilation; kernel_label; use_padding } ->
              let dilation_int =
                try Int.of_string dilation
                with _ -> failwith ("Invalid dilation value (expected integer): " ^ dilation)
              in
              let use_padding_bool =
                match use_padding with
                | `True -> true
                | `False -> false
                | `Unspecified ->
                    failwith
                      "use_padding must be specified in convolution spec (use = for true, < for \
                       false)"
              in
              let kernel =
                Row.Var
                  (Hashtbl.find_or_add dim_var_env kernel_label ~default:(fun () ->
                       Row.get_var ~name:kernel_label ()))
              in
              { Row.dilation = dilation_int; kernel; use_padding = use_padding_bool })
        in
        Row.Affine { stride = stride_int; over = over_dim; conv; stride_offset }
    | Concat_spec labels ->
        let dims =
          List.map labels ~f:(fun label ->
              Row.Var
                (Hashtbl.find_or_add dim_var_env label ~default:(fun () ->
                     Row.get_var ~name:label ())))
        in
        Row.Concat dims
  in
  let result = axes_spec_to_dims_bio ~sh_id ~row_var_env ~dim_var_env ~f labels in
  (!extras, !proj_env_update, result)

type proj_axis_env = Idx.axis_index Row.dim_map [@@deriving sexp]

let check_dim_row_var_name_clash ~spec ~error_trace dim_var_env row_var_env =
  Hashtbl.iter_keys dim_var_env ~f:(fun name ->
      if Hashtbl.mem row_var_env name then
        raise
        @@ Row.Shape_error
             ( "Name '" ^ name
               ^ "' is used as both a dimension variable and a row variable in spec: " ^ spec,
               error_trace ))

let add_var_used_in_spec_or_compose row =
  match row with Row.Row_var v -> Row.add_used_in_spec_or_compose v | _ -> ()

let add_var_used_in_pointwise row =
  match row with Row.Row_var v -> Row.add_used_in_pointwise v | _ -> ()

(** Shared helper: bind each [delayed_var_ref] to the dim/row variable resolved for its label in
    [dim_var_env]/[row_var_env]. Returns the additional constraints implied by any pre-solved dims.
    Extracted to eliminate duplication across the unary-permute, binary-einsum, block, and
    ternary-einsum branches of [get_inequalities]. *)
let bind_delayed_vars_to_envs ~for_projections ~spec ~dim_var_env ~row_var_env ~error_shapes
    dim_refs =
  List.filter_map dim_refs ~f:(fun delayed_ref ->
      let label = delayed_ref.var_ref.ref_label in
      match Hashtbl.find dim_var_env label with
      | Some var -> (
          delayed_ref.var <- `Dim var;
          let origin =
            [
              {
                Row.lhs_name = label;
                lhs_kind = `Output;
                rhs_name = delayed_ref.var_ref.ref_label;
                rhs_kind = `Output;
                operation = Some "set_dim";
              };
            ]
          in
          match (delayed_ref.var_ref.solved_sym, delayed_ref.var_ref.solved_dim) with
          | _ when for_projections -> None
          | Some sym, _ -> Some (Row.Dim_eq { d1 = Row.Var var; d2 = Row.get_sym_dim sym; origin })
          | None, Some solved_dim ->
              Some
                (Row.Dim_eq { d1 = Row.Var var; d2 = Row.get_default_dim ~d:solved_dim (); origin })
          | None, None -> None)
      | None -> (
          match Hashtbl.find row_var_env label with
          | Some var ->
              delayed_ref.var <- `Row var;
              if Option.is_some delayed_ref.var_ref.solved_sym then
                raise
                @@ Row.Shape_error
                     ( "Variable " ^ label
                       ^ " is bound to a row variable but was set to a symbolic dimension \
                          (set_sym_dim does not support row variables)",
                       [ Shape_mismatch error_shapes ] );
              Option.map
                ~f:(fun solved_dim ->
                  Row.Rows_constr
                    {
                      r = [ Row.get_row_for_var Row.empty_provenance var ];
                      constr =
                        Row.Total_elems
                          { numerator = Num_elems solved_dim; divided_by = []; keep_axis = false };
                      origin =
                        [
                          {
                            Row.lhs_name = label;
                            lhs_kind = `Output;
                            rhs_name = delayed_ref.var_ref.ref_label;
                            rhs_kind = `Output;
                            operation = Some "set_dim";
                          };
                        ];
                    })
                (if for_projections then None else delayed_ref.var_ref.solved_dim)
          | None ->
              raise
              @@ Row.Shape_error
                   ( "Variable " ^ label ^ " not found in environments for spec: " ^ spec,
                     [ Shape_mismatch error_shapes ] )))

(** Shared n-ary einsum constraint builder used by [Broadcast (Einsum ...)],
    [Broadcast_tern (Einsum_tern ...)], and [Block]. Handles per-shape
    [einsum_slot_spec_to_dims_bio] calls, name-clash check, delayed-var binding, proj-env merge, and
    [Row_eq] emission.

    [rhs_slots] is a list of [(parsed_axis_labels, shape)] pairs, one per RHS. [result_label] is the
    operation string for LHS [Row_eq] origins (e.g. ["Broadcast RESULT"] or ["Block RESULT"]).
    [arg_label_prefix] is the prefix for RHS [Row_eq] origins; the index (1-based) is appended,
    yielding e.g. ["Broadcast ARGUMENT 1"]. [error_shapes] is the shape list for error context.

    Returns [(proj_env, bio_lhs, bio_rhs_list, inequalities)] where [bio_lhs] and [bio_rhs_list] are
    the batch/input/output row triples for the LHS and each RHS — needed by [Block] to compute
    discardable vars. *)
let einsum_n_constraints ~for_projections ~spec ~(rhs_slots : (parsed_axis_labels * t) list)
    ~(cur_sh : t) ~ls_lhs ~dim_refs ~result_label ~arg_label_prefix ~(error_shapes : t list)
    ?(lhs_constraints_first = false) () =
  let row_var_env = Hashtbl.create (module String) in
  let dim_var_env = Hashtbl.create (module String) in
  let rhs_results =
    List.mapi rhs_slots ~f:(fun idx (ls_rhs, (sh : t)) ->
        let extras, proj_env, bio =
          einsum_slot_spec_to_dims_bio ~original_spec:spec ~sh_id:sh.id ~row_var_env ~dim_var_env
            ls_rhs
        in
        (idx, sh, extras, proj_env, bio))
  in
  let extras_lhs, proj_env_lhs, ((b_lhs, i_lhs, o_lhs) as bio_lhs) =
    einsum_slot_spec_to_dims_bio ~original_spec:spec ~sh_id:(cur_sh : t).id ~row_var_env
      ~dim_var_env ls_lhs
  in
  check_dim_row_var_name_clash ~spec ~error_trace:[ Shape_mismatch error_shapes ] dim_var_env
    row_var_env;
  let extras_dim_refs =
    bind_delayed_vars_to_envs ~for_projections ~spec ~dim_var_env ~row_var_env ~error_shapes
      dim_refs
  in
  let proj_env =
    let combine ~key:_ _ _ = assert false in
    List.fold rhs_results ~init:proj_env_lhs ~f:(fun acc (_, _, _, proj_env_rhs, _) ->
        Map.merge_skewed ~combine acc proj_env_rhs)
  in
  let rhs_constraints =
    List.concat_map rhs_results ~f:(fun (idx, sh, extras, _, (b_rhs, i_rhs, o_rhs)) ->
        let arg_label = Printf.sprintf "%s %d" arg_label_prefix (idx + 1) in
        extras
        @ [
            Row.Row_eq
              {
                r1 = b_rhs;
                r2 = sh.batch;
                origin =
                  [
                    {
                      Row.lhs_name = spec;
                      lhs_kind = `Batch;
                      rhs_name = sh.debug_name;
                      rhs_kind = `Batch;
                      operation = Some arg_label;
                    };
                  ];
              };
            Row.Row_eq
              {
                r1 = i_rhs;
                r2 = sh.input;
                origin =
                  [
                    {
                      Row.lhs_name = spec;
                      lhs_kind = `Input;
                      rhs_name = sh.debug_name;
                      rhs_kind = `Input;
                      operation = Some arg_label;
                    };
                  ];
              };
            Row.Row_eq
              {
                r1 = o_rhs;
                r2 = sh.output;
                origin =
                  [
                    {
                      Row.lhs_name = spec;
                      lhs_kind = `Output;
                      rhs_name = sh.debug_name;
                      rhs_kind = `Output;
                      operation = Some arg_label;
                    };
                  ];
              };
          ])
  in
  let lhs_constraints =
    [
      Row.Row_eq
        {
          r1 = cur_sh.batch;
          r2 = b_lhs;
          origin =
            [
              {
                Row.lhs_name = cur_sh.debug_name;
                lhs_kind = `Batch;
                rhs_name = spec;
                rhs_kind = `Batch;
                operation = Some result_label;
              };
            ];
        };
      Row.Row_eq
        {
          r1 = cur_sh.input;
          r2 = i_lhs;
          origin =
            [
              {
                Row.lhs_name = cur_sh.debug_name;
                lhs_kind = `Input;
                rhs_name = spec;
                rhs_kind = `Input;
                operation = Some result_label;
              };
            ];
        };
      Row.Row_eq
        {
          r1 = cur_sh.output;
          r2 = o_lhs;
          origin =
            [
              {
                Row.lhs_name = cur_sh.debug_name;
                lhs_kind = `Output;
                rhs_name = spec;
                rhs_kind = `Output;
                operation = Some result_label;
              };
            ];
        };
    ]
  in
  let bio_rhs_list = List.map rhs_results ~f:(fun (_, _, _, _, bio) -> bio) in
  let ordered_constraints =
    if lhs_constraints_first then lhs_constraints @ rhs_constraints
    else rhs_constraints @ lhs_constraints
  in
  (proj_env, bio_lhs, bio_rhs_list, extras_dim_refs @ extras_lhs @ ordered_constraints)

(* For Block specs, compute discardable_vars: variables that are allowed to be 0. A variable v is
   discardable if: 1. v appears in a component of a Concat dimension on one side 2. For ALL shapes
   on the other side, there EXISTS an axis such that for ALL components of that axis, the complement
   of v's component has non-empty intersection. This is a four-quantifier condition: ∀shapes ∃axis
   ∀components: complement ∩ component ≠ ∅ *)
let compute_block_discardable_vars ~(this_side_rows : Row.t list)
    ~(other_side_shapes : Row.t list list) : Row.dim_var_set =
  (* Extract all dims from rows (both flanks; bcast no longer carries beg_dims). *)
  let dims_of_rows rows = List.concat_map rows ~f:(fun (row : Row.t) -> row.beg_dims @ row.dims) in
  (* Get component var sets for each axis. For non-Concat dims, there's one component with all vars
     of that dim. For Concat dims, each element is a separate component. *)
  let axis_components_of_dim (dim : Row.dim) : Row.dim_var_set list =
    match dim with
    | Row.Concat dims -> List.map dims ~f:Row.vars_of_dim
    | _ -> [ Row.vars_of_dim dim ]
  in
  (* For each shape on the other side, get its axes (each axis is a list of component var sets) *)
  let other_side_shape_axes : Row.dim_var_set list list list =
    List.map other_side_shapes ~f:(fun shape_rows ->
        List.map (dims_of_rows shape_rows) ~f:axis_components_of_dim)
  in
  (* For each Concat on this side, find discardable vars. Non-Concat dims cannot contribute
     discardable vars since their complement is empty. *)
  let this_side_dims = dims_of_rows this_side_rows in
  List.fold this_side_dims ~init:Row.dim_var_set_empty ~f:(fun acc dim ->
      match dim with
      | Row.Concat components ->
          let component_var_sets = List.map components ~f:Row.vars_of_dim in
          let all_concat_vars =
            List.fold component_var_sets ~init:Row.dim_var_set_empty ~f:Set.union
          in
          List.fold component_var_sets ~init:acc ~f:(fun acc2 component_vars ->
              let complement = Set.diff all_concat_vars component_vars in
              (* Check: for ALL shapes, EXISTS an axis where ALL components intersect complement *)
              let complement_covers_all_shapes =
                List.for_all other_side_shape_axes ~f:(fun shape_axes ->
                    List.exists shape_axes ~f:(fun axis_components ->
                        List.for_all axis_components ~f:(fun axis_comp_vars ->
                            not (Set.is_empty (Set.inter complement axis_comp_vars)))))
              in
              if complement_covers_all_shapes then Set.union acc2 component_vars else acc2)
      | _ -> acc)

let unused_shapes = Hash_set.create (module Int)

let%debug4_sexp get_inequalities ?(for_projections = false)
    ({ shape = cur_sh; logic; _ } as _upd : update_step) :
    proj_axis_env * Row.dim_var_set * Row.constraint_ list =
  Hash_set.remove unused_shapes cur_sh.id;
  let _debug_cur_sh : t = cur_sh in
  let _debug_logic : logic = logic in
  let open Row in
  let defaults ineqs = (dim_map_empty, dim_var_set_empty, ineqs) in
  let get_origin lhs_kind sh rhs_kind operation =
    Row.
      {
        lhs_name = cur_sh.debug_name;
        lhs_kind;
        rhs_name = sh.debug_name;
        rhs_kind;
        operation = Some operation;
      }
  in
  let mark_terminal ~is_param =
    [
      Terminal_row (is_param, cur_sh.batch, [ get_origin `Batch cur_sh `Batch "terminal" ]);
      Terminal_row (is_param, cur_sh.input, [ get_origin `Input cur_sh `Input "terminal" ]);
      Terminal_row (is_param, cur_sh.output, [ get_origin `Output cur_sh `Output "terminal" ]);
    ]
  in
  match logic with
  | Terminal { is_param; logic = Fetch Range_over_offsets } -> defaults @@ mark_terminal ~is_param
  | Terminal { is_param; logic = Fetch (Constant _) | Data (Broadcast _) } ->
      defaults @@ mark_terminal ~is_param
  | Terminal { is_param; logic = Fetch (Constant_bits _) } -> defaults @@ mark_terminal ~is_param
  | Terminal { is_param; logic = Data (Reshape nd) } ->
      ( dim_map_empty,
        dim_var_set_empty,
        Rows_constr
          {
            r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
            constr =
              (* [keep_axis]: a 1-element reshape prefers keeping a single dim-1 axis over
                 collapsing to a scalar when nothing else constrains the rows (gh-460). *)
              Total_elems
                {
                  numerator = Num_elems (Array.fold (Ir.Ndarray.dims nd) ~init:1 ~f:( * ));
                  divided_by = [];
                  keep_axis = true;
                };
            origin =
              [
                {
                  lhs_name = cur_sh.debug_name;
                  lhs_kind = `Batch;
                  rhs_name = "Reshape";
                  rhs_kind = `Output;
                  operation = Some "Total_elems";
                };
              ];
          }
        :: mark_terminal ~is_param )
  | Terminal { is_param; logic = Data (Keep_shape_no_padding nd) } ->
      (* FIXME: constrain padding to "not padded". *)
      ( dim_map_empty,
        dim_var_set_empty,
        (if for_projections then []
         else
           [
             Rows_constr
               {
                 r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
                 constr =
                   Exact
                     (Ir.Ndarray.dims nd
                     |> Array.map ~f:(fun d -> get_default_dim ~d ())
                     |> Array.to_list);
                 origin =
                   [
                     {
                       lhs_name = cur_sh.debug_name;
                       lhs_kind = `Batch;
                       rhs_name = "Keep_shape_no_padding";
                       rhs_kind = `Output;
                       operation = Some "Exact";
                     };
                   ];
               };
           ])
        @ mark_terminal ~is_param )
  | Terminal { is_param; logic = Data (Padded { data; padding; padded_value }) } ->
      ignore padded_value;
      ( dim_map_empty,
        dim_var_set_empty,
        (if for_projections then []
         else
           [
             Rows_constr
               {
                 r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
                 constr =
                   Exact
                     (* The ndarray is padded as given: the shape's (logical) dims exclude the
                        margins. The committed widths and neutral element themselves are enforced
                        against later demands via the tnode's forced padding lazy (see
                        [fresh_proj_ids]). *)
                     (Ir.Ndarray.dims data
                     |> Array.mapi ~f:(fun i d ->
                         let d =
                           if i < Array.length padding then
                             d - padding.(i).Ir.Ops.left - padding.(i).Ir.Ops.right
                           else d
                         in
                         get_default_dim ~d ())
                     |> Array.to_list);
                 origin =
                   [
                     {
                       lhs_name = cur_sh.debug_name;
                       lhs_kind = `Batch;
                       rhs_name = "Padded";
                       rhs_kind = `Output;
                       operation = Some "Exact";
                     };
                   ];
               };
           ])
        @ mark_terminal ~is_param )
  | Terminal { is_param; logic = Fetch (Constant_fill values) } ->
      let len = Array.length values in
      ( dim_map_empty,
        dim_var_set_empty,
        Rows_constr
          {
            r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
            constr = Total_elems { numerator = Num_elems len; divided_by = []; keep_axis = false };
            origin =
              [
                {
                  lhs_name = cur_sh.debug_name;
                  lhs_kind = `Batch;
                  rhs_name = "Constant_fill";
                  rhs_kind = `Output;
                  operation = Some "Total_elems";
                };
              ];
          }
        :: mark_terminal ~is_param )
  | Terminal { is_param; logic = Fetch (Slice { sliced = tn; batch_idx = _ }) } ->
      if Lazy.is_val tn.dims then
        ( dim_map_empty,
          dim_var_set_empty,
          (if for_projections then []
           else
             [
               Rows_constr
                 {
                   r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
                   constr =
                     Exact
                       (Lazy.force tn.dims |> Array.to_list |> List.tl_exn
                       |> List.map ~f:(fun d -> get_default_dim ~d ()));
                   origin =
                     [
                       {
                         lhs_name = cur_sh.debug_name;
                         lhs_kind = `Batch;
                         rhs_name = Tn.debug_name tn;
                         rhs_kind = `Output;
                         operation = Some "Slice";
                       };
                     ];
                 };
             ])
          @ mark_terminal ~is_param )
      else defaults @@ mark_terminal ~is_param
  | Terminal { is_param; logic = Fetch (Embed_symbol _) } -> defaults @@ mark_terminal ~is_param
  | Terminal { is_param; logic = Fetch (Embed_dim _) } -> defaults @@ mark_terminal ~is_param
  | Terminal { is_param; logic = Fetch Embed_self_id } -> defaults @@ mark_terminal ~is_param
  | Transpose (Transpose, sh) ->
      Hash_set.remove unused_shapes sh.id;
      defaults
      @@ [
           Row_ineq
             {
               res = cur_sh.batch;
               opnd = sh.batch;
               origin = [ get_origin `Batch sh `Batch "transpose" ];
             };
           Row_ineq
             {
               res = cur_sh.input;
               opnd = sh.output;
               origin = [ get_origin `Input sh `Output "transpose" ];
             };
           Row_ineq
             {
               res = cur_sh.output;
               opnd = sh.input;
               origin = [ get_origin `Output sh `Input "transpose" ];
             };
         ]
  | Transpose (Pointwise_un, sh) ->
      Hash_set.remove unused_shapes sh.id;
      add_var_used_in_pointwise cur_sh.input.bcast;
      add_var_used_in_pointwise sh.input.bcast;
      defaults
      @@ [
           Row_ineq
             {
               res = cur_sh.batch;
               opnd = sh.batch;
               origin = [ get_origin `Batch sh `Batch "pointwise_unary" ];
             };
           Row_ineq
             {
               res = cur_sh.input;
               opnd = sh.input;
               origin = [ get_origin `Input sh `Input "pointwise_unary" ];
             };
           Row_ineq
             {
               res = cur_sh.output;
               opnd = sh.output;
               origin = [ get_origin `Output sh `Output "pointwise_unary" ];
             };
         ]
  | Broadcast (Compose, sh1, sh2) ->
      Hash_set.remove unused_shapes sh1.id;
      Hash_set.remove unused_shapes sh2.id;
      add_var_used_in_spec_or_compose sh1.input.bcast;
      defaults
      @@ [
           Row_ineq
             {
               origin =
                 [
                   {
                     lhs_name = sh1.debug_name;
                     lhs_kind = `Input;
                     rhs_name = sh2.debug_name;
                     rhs_kind = `Output;
                     operation = Some "compose";
                   };
                 ];
               res = sh1.input;
               opnd = sh2.output;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh1 `Batch "compose" ];
               res = cur_sh.batch;
               opnd = sh1.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh2 `Batch "compose" ];
               res = cur_sh.batch;
               opnd = sh2.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh2 `Input "compose" ];
               res = cur_sh.input;
               opnd = sh2.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh1 `Output "compose" ];
               res = cur_sh.output;
               opnd = sh1.output;
             };
         ]
  | Broadcast (Pointwise_bin, sh1, sh2) ->
      Hash_set.remove unused_shapes sh1.id;
      Hash_set.remove unused_shapes sh2.id;
      add_var_used_in_pointwise cur_sh.input.bcast;
      add_var_used_in_pointwise sh1.input.bcast;
      add_var_used_in_pointwise sh2.input.bcast;
      defaults
      @@ [
           Row_ineq
             {
               origin = [ get_origin `Batch sh1 `Batch "pointwise_binary" ];
               res = cur_sh.batch;
               opnd = sh1.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh2 `Batch "pointwise_binary" ];
               res = cur_sh.batch;
               opnd = sh2.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh1 `Input "pointwise_binary" ];
               res = cur_sh.input;
               opnd = sh1.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh2 `Input "pointwise_binary" ];
               res = cur_sh.input;
               opnd = sh2.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh1 `Output "pointwise_binary" ];
               res = cur_sh.output;
               opnd = sh1.output;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh2 `Output "pointwise_binary" ];
               res = cur_sh.output;
               opnd = sh2.output;
             };
         ]
  | Broadcast_tern (Compose_accumulate, sh1, sh2, sh3) ->
      Hash_set.remove unused_shapes sh1.id;
      Hash_set.remove unused_shapes sh2.id;
      Hash_set.remove unused_shapes sh3.id;
      add_var_used_in_spec_or_compose sh1.input.bcast;
      defaults
      @@ [
           Row_ineq
             {
               origin =
                 [
                   {
                     lhs_name = sh1.debug_name;
                     lhs_kind = `Input;
                     rhs_name = sh2.debug_name;
                     rhs_kind = `Output;
                     operation = Some "compose_accumulate";
                   };
                 ];
               res = sh1.input;
               opnd = sh2.output;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh1 `Batch "compose_accumulate" ];
               res = cur_sh.batch;
               opnd = sh1.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh2 `Batch "compose_accumulate" ];
               res = cur_sh.batch;
               opnd = sh2.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh2 `Input "compose_accumulate" ];
               res = cur_sh.input;
               opnd = sh2.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh1 `Output "compose_accumulate" ];
               res = cur_sh.output;
               opnd = sh1.output;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh3 `Batch "compose_accumulate" ];
               res = cur_sh.batch;
               opnd = sh3.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh3 `Input "compose_accumulate" ];
               res = cur_sh.input;
               opnd = sh3.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh3 `Output "compose_accumulate" ];
               res = cur_sh.output;
               opnd = sh3.output;
             };
         ]
  | Broadcast_tern (Pointwise_tern, sh1, sh2, sh3) ->
      Hash_set.remove unused_shapes sh1.id;
      Hash_set.remove unused_shapes sh2.id;
      Hash_set.remove unused_shapes sh3.id;
      add_var_used_in_pointwise cur_sh.input.bcast;
      add_var_used_in_pointwise sh1.input.bcast;
      add_var_used_in_pointwise sh2.input.bcast;
      add_var_used_in_pointwise sh3.input.bcast;
      defaults
      @@ [
           Row_ineq
             {
               origin = [ get_origin `Batch sh1 `Batch "pointwise_ternary" ];
               res = cur_sh.batch;
               opnd = sh1.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh2 `Batch "pointwise_ternary" ];
               res = cur_sh.batch;
               opnd = sh2.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh3 `Batch "pointwise_ternary" ];
               res = cur_sh.batch;
               opnd = sh3.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh1 `Input "pointwise_ternary" ];
               res = cur_sh.input;
               opnd = sh1.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh2 `Input "pointwise_ternary" ];
               res = cur_sh.input;
               opnd = sh2.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh3 `Input "pointwise_ternary" ];
               res = cur_sh.input;
               opnd = sh3.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh1 `Output "pointwise_ternary" ];
               res = cur_sh.output;
               opnd = sh1.output;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh2 `Output "pointwise_ternary" ];
               res = cur_sh.output;
               opnd = sh2.output;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh3 `Output "pointwise_ternary" ];
               res = cur_sh.output;
               opnd = sh3.output;
             };
         ]
  | Broadcast (Defined_by_cd_logic, _, _)
  | Transpose (Defined_by_cd_logic, _)
  | Broadcast_tern (Defined_by_cd_logic, _, _, _) ->
      defaults @@ []
  | Transpose (Batch_slice ({ static_range; static_symbol; _ } as slice_sym), sh) ->
      (* A slice index and a symbolic extent (gh-490) validate incompatibly (strict [0, range) vs.
         inclusive [0, range]): reject a binding used as both. *)
      if slice_sym.Idx.used_as_extent then
        raise
        @@ Row.Shape_error
             ( "Static index " ^ Idx.symbol_ident static_symbol
               ^ " is used as a symbolic dimension extent and cannot also slice a batch axis",
               [] );
      slice_sym.Idx.used_as_slice <- true;
      Hash_set.remove unused_shapes sh.id;
      let slice_v = get_var () in
      let slice_var = Var slice_v in
      let proj_axis_env =
        Map.add_exn Row.dim_map_empty ~key:slice_v ~data:(Idx.Iterator static_symbol)
      in
      (* Expand a batch row instead of reducing one because even if the dimensions are known, the
         equations are also needed for projection inference. *)
      let expanded_batch =
        (* Slice axis is outer-anchored; prepend to beg_dims for both closed and open rows. *)
        {
          beg_dims = slice_var :: cur_sh.batch.beg_dims;
          dims = cur_sh.batch.dims;
          bcast = cur_sh.batch.bcast;
          prov = Row.provenance ~sh_id:cur_sh.id ~kind:`Batch;
        }
      in
      let get_origin kind =
        [
          {
            lhs_name = cur_sh.debug_name;
            lhs_kind = kind;
            rhs_name = sh.debug_name;
            rhs_kind = kind;
            operation = Some "Batch_slice";
          };
        ]
      in
      ( proj_axis_env,
        dim_var_set_empty,
        (Option.to_list (if for_projections then None else static_range)
        |> List.map ~f:(fun range ->
            Dim_eq
              {
                d1 = get_default_dim ~d:range ();
                d2 = slice_var;
                origin =
                  [
                    {
                      lhs_name = sh.debug_name;
                      lhs_kind = `Batch;
                      rhs_name = Idx.symbol_ident static_symbol;
                      rhs_kind = `Batch;
                      operation = Some "Slice";
                    };
                  ];
              }))
        @ [
            Row_eq { r1 = expanded_batch; r2 = sh.batch; origin = get_origin `Batch };
            Row_eq { r1 = cur_sh.input; r2 = sh.input; origin = get_origin `Input };
            Row_eq { r1 = cur_sh.output; r2 = sh.output; origin = get_origin `Output };
          ] )
  | Transpose (Permute (spec, dim_refs), sh) ->
      Hash_set.remove unused_shapes sh.id;
      add_var_used_in_spec_or_compose cur_sh.input.bcast;
      add_var_used_in_spec_or_compose sh.input.bcast;
      let ls_rhs, ls_lhs =
        match einsum_of_spec spec with
        | [ ls_rhs ], ls_lhs -> (ls_rhs, ls_lhs)
        | _ ->
            raise
            @@ Shape_error
                 ( "Invalid permutation spec (expected one argument): " ^ spec,
                   [ Shape_mismatch [ cur_sh; sh ] ] )
      in
      let row_var_env = Hashtbl.create (module String) in
      let dim_var_env = Hashtbl.create (module String) in

      let extras_rhs, proj_env_rhs, (b_rhs, i_rhs, o_rhs) =
        einsum_slot_spec_to_dims_bio ~original_spec:spec ~sh_id:sh.id ~row_var_env ~dim_var_env
          ls_rhs
      in
      let extras_lhs, proj_env_lhs, (b_lhs, i_lhs, o_lhs) =
        einsum_slot_spec_to_dims_bio ~original_spec:spec ~sh_id:cur_sh.id ~row_var_env ~dim_var_env
          ls_lhs
      in
      check_dim_row_var_name_clash ~spec
        ~error_trace:[ Shape_mismatch [ cur_sh; sh ] ]
        dim_var_env row_var_env;
      let extras_dim_refs =
        bind_delayed_vars_to_envs ~for_projections ~spec ~dim_var_env ~row_var_env
          ~error_shapes:[ cur_sh; sh ] dim_refs
      in
      let proj_env =
        let combine ~key:_ _ _ = assert false in
        Map.merge_skewed ~combine proj_env_rhs proj_env_lhs
      in
      ( proj_env,
        dim_var_set_empty,
        extras_dim_refs @ extras_rhs @ extras_lhs
        @ [
            Row_eq
              {
                r1 = cur_sh.batch;
                r2 = b_lhs;
                origin =
                  [
                    {
                      lhs_name = cur_sh.debug_name;
                      lhs_kind = `Batch;
                      rhs_name = spec;
                      rhs_kind = `Batch;
                      operation = Some "Permute RESULT";
                    };
                  ];
              };
            Row_eq
              {
                r1 = b_rhs;
                r2 = sh.batch;
                origin =
                  [
                    {
                      lhs_name = spec;
                      lhs_kind = `Batch;
                      rhs_name = sh.debug_name;
                      rhs_kind = `Batch;
                      operation = Some "Permute ARGUMENT";
                    };
                  ];
              };
            Row_eq
              {
                r1 = cur_sh.input;
                r2 = i_lhs;
                origin =
                  [
                    {
                      lhs_name = cur_sh.debug_name;
                      lhs_kind = `Input;
                      rhs_name = spec;
                      rhs_kind = `Input;
                      operation = Some "Permute RESULT";
                    };
                  ];
              };
            Row_eq
              {
                r1 = i_rhs;
                r2 = sh.input;
                origin =
                  [
                    {
                      lhs_name = spec;
                      lhs_kind = `Input;
                      rhs_name = sh.debug_name;
                      rhs_kind = `Input;
                      operation = Some "Permute ARGUMENT";
                    };
                  ];
              };
            Row_eq
              {
                r1 = cur_sh.output;
                r2 = o_lhs;
                origin =
                  [
                    {
                      lhs_name = cur_sh.debug_name;
                      lhs_kind = `Output;
                      rhs_name = spec;
                      rhs_kind = `Output;
                      operation = Some "Permute RESULT";
                    };
                  ];
              };
            Row_eq
              {
                r1 = o_rhs;
                r2 = sh.output;
                origin =
                  [
                    {
                      lhs_name = spec;
                      lhs_kind = `Output;
                      rhs_name = sh.debug_name;
                      rhs_kind = `Output;
                      operation = Some "Permute ARGUMENT";
                    };
                  ];
              };
          ] )
  | Transpose (Uint4x32_to_prec target_prec, sh) ->
      let var = get_var () in
      let coeff =
        Utils.safe_lazy [%string "Uint4x32 %{sh.id#Int} to_prec_of %{cur_sh.id#Int}"] (fun () ->
            16 / Ir.Ops.prec_in_bytes (Lazy.force target_prec))
      in
      defaults
      @@ [
           Rows_constr
             {
               r = [ sh.batch; sh.output; sh.input ];
               constr = Row.Exact [ Var var ];
               origin =
                 [
                   {
                     lhs_name = sh.debug_name;
                     lhs_kind = `Batch;
                     rhs_name = cur_sh.debug_name;
                     rhs_kind = `Batch;
                     operation = Some "Uint4x32_to_prec ARGUMENT axes exact";
                   };
                 ];
             };
           Rows_constr
             {
               r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
               constr =
                 Total_elems
                   {
                     (* [round_up]: the result total need not be a multiple of [coeff]; the counter
                        gets [ceil(total / coeff)] blocks and the last block is consumed partially
                        (tail-peeled in [Assignments] lowering). *)
                     numerator = Row.Strided_var { coeff; var; denom = 1; round_up = true };
                     divided_by = [];
                     keep_axis = false;
                   };
               origin =
                 [
                   {
                     lhs_name = cur_sh.debug_name;
                     lhs_kind = `Batch;
                     rhs_name = sh.debug_name;
                     rhs_kind = `Output;
                     operation = Some "Uint4x32_to_prec RESULT total elements";
                   };
                 ];
             };
         ]
  | Broadcast (Einsum (spec, dim_refs), sh1, sh2) ->
      Hash_set.remove unused_shapes sh1.id;
      Hash_set.remove unused_shapes sh2.id;
      add_var_used_in_spec_or_compose cur_sh.input.bcast;
      add_var_used_in_spec_or_compose sh1.input.bcast;
      add_var_used_in_spec_or_compose sh2.input.bcast;
      let ls_rhs1, ls_rhs2, ls_lhs =
        match einsum_of_spec spec with
        | [ ls_rhs1; ls_rhs2 ], ls_lhs -> (ls_rhs1, ls_rhs2, ls_lhs)
        | _ ->
            raise
            @@ Shape_error
                 ( "Invalid einsum spec (expected two arguments): " ^ spec,
                   [ Shape_mismatch [ cur_sh; sh1; sh2 ] ] )
      in
      let proj_env, _, _, inequalities =
        einsum_n_constraints ~for_projections ~spec
          ~rhs_slots:[ (ls_rhs1, sh1); (ls_rhs2, sh2) ]
          ~cur_sh ~ls_lhs ~dim_refs ~result_label:"Broadcast RESULT"
          ~arg_label_prefix:"Broadcast ARGUMENT" ~error_shapes:[ cur_sh; sh1; sh2 ]
          ~lhs_constraints_first:true ()
      in
      (proj_env, dim_var_set_empty, inequalities)
  | Block { spec; delayed_vars; rhses } ->
      List.iter rhses ~f:(fun sh -> Hash_set.remove unused_shapes sh.id);
      add_var_used_in_spec_or_compose cur_sh.input.bcast;
      List.iter rhses ~f:(fun sh -> add_var_used_in_spec_or_compose sh.input.bcast);
      let ls_rhses, ls_lhs =
        match einsum_of_spec spec with
        | ls_rhses, ls_lhs when List.length ls_rhses = List.length rhses -> (ls_rhses, ls_lhs)
        | _ ->
            raise
            @@ Shape_error
                 ( Printf.sprintf "Invalid block spec (expected %d arguments): %s"
                     (List.length rhses) spec,
                   [ Shape_mismatch (cur_sh :: rhses) ] )
      in
      let proj_env, (b_lhs, i_lhs, o_lhs), bio_rhs_list, inequalities =
        einsum_n_constraints ~for_projections ~spec ~rhs_slots:(List.zip_exn ls_rhses rhses) ~cur_sh
          ~ls_lhs ~dim_refs:delayed_vars ~result_label:"Block RESULT"
          ~arg_label_prefix:"Block ARGUMENT" ~error_shapes:(cur_sh :: rhses) ()
      in
      (* Compute discardable_vars for concat semantics: variables allowed to be 0. *)
      let rhs_shapes : Row.t list list =
        List.map bio_rhs_list ~f:(fun (b, inp, o) -> [ b; inp; o ])
      in
      let lhs_rows = [ b_lhs; i_lhs; o_lhs ] in
      let discardable_from_rhs =
        compute_block_discardable_vars ~this_side_rows:(List.concat rhs_shapes)
          ~other_side_shapes:[ lhs_rows ]
      in
      let discardable_from_lhs =
        compute_block_discardable_vars ~this_side_rows:lhs_rows ~other_side_shapes:rhs_shapes
      in
      let discardable_vars = Set.union discardable_from_rhs discardable_from_lhs in
      (proj_env, discardable_vars, inequalities)
  | Broadcast_tern (Einsum_tern (spec, dim_refs), sh1, sh2, sh3) ->
      Hash_set.remove unused_shapes sh1.id;
      Hash_set.remove unused_shapes sh2.id;
      Hash_set.remove unused_shapes sh3.id;
      add_var_used_in_spec_or_compose cur_sh.input.bcast;
      add_var_used_in_spec_or_compose sh1.input.bcast;
      add_var_used_in_spec_or_compose sh2.input.bcast;
      add_var_used_in_spec_or_compose sh3.input.bcast;
      let ls_rhs1, ls_rhs2, ls_rhs3, ls_lhs =
        match einsum_of_spec spec with
        | [ ls_rhs1; ls_rhs2; ls_rhs3 ], ls_lhs -> (ls_rhs1, ls_rhs2, ls_rhs3, ls_lhs)
        | _ ->
            raise
            @@ Shape_error
                 ( "Invalid einsum spec (expected three arguments): " ^ spec,
                   [ Shape_mismatch [ cur_sh; sh1; sh2; sh3 ] ] )
      in
      let proj_env, _, _, inequalities =
        einsum_n_constraints ~for_projections ~spec
          ~rhs_slots:[ (ls_rhs1, sh1); (ls_rhs2, sh2); (ls_rhs3, sh3) ]
          ~cur_sh ~ls_lhs ~dim_refs ~result_label:"Broadcast RESULT"
          ~arg_label_prefix:"Broadcast ARGUMENT" ~error_shapes:[ cur_sh; sh1; sh2; sh3 ] ()
      in
      (proj_env, dim_var_set_empty, inequalities)

let state = ref Row.empty_env

(* Depth of tensor-operation constructions in progress: [Tensor.op] raises it around its [op_asn]
   (see [with_construction_window]). While it is positive, finalizing inference is refused at the
   finalizer: the operation whose assignments are being built has not registered its constraints
   yet, so a [finish_inference] here would close its operands without them and commit that to the
   shapes, which no later step can undo (gh-ocannl-830). *)
let construction_depth = ref 0
let active_update_steps = ref []
let active_constraints = ref []

let infer_equal (sh1 : t) (sh2 : t) =
  Hash_set.remove unused_shapes sh1.id;
  Hash_set.remove unused_shapes sh2.id;
  let get_origin kind =
    [
      Row.
        {
          lhs_name = sh1.debug_name;
          lhs_kind = kind;
          rhs_name = sh2.debug_name;
          rhs_kind = kind;
          operation = Some "shape_equals";
        };
    ]
  in
  active_constraints :=
    Row.Row_eq { r1 = sh1.batch; r2 = sh2.batch; origin = get_origin `Batch }
    :: Row.Row_eq { r1 = sh1.input; r2 = sh2.input; origin = get_origin `Input }
    :: Row.Row_eq { r1 = sh1.output; r2 = sh2.output; origin = get_origin `Output }
    :: !active_constraints

(** The product-space proxy shape of an einsum-family operation (gh-512): the result's axes followed
    by the reduced-over (contracted) axes, matching the product-space order of [derive_projections]
    — result rows in [to_dims] order, then contracted axes by first occurrence over the RHS slots
    (each walked batch, output, input; [beg_dims], row variable, [dims]). Built by re-elaborating
    the einsum spec with fresh variables — the re-emitted operand equalities tie them to the
    operation's own solution — and appending the contracted axes to the result's input row (the last
    row in [to_dims] order). A [%cd] [*_pspace] intermediate's shape is [infer_equal]-ed to this
    proxy; the identity projection over the product space is derived from the projections by
    [Ir.Indexing.prod_project_for], which cross-checks the resulting dimensions against the product
    space at lowering.

    Raises [Row.Shape_error] when the product space is not expressible as a shape: a non-einsum
    operation, more than one reduced-over row variable, or a reduced-over row variable combined with
    an open result input row. *)
let product_space_shape ~(shape : t) ~(logic : logic) : t =
  let cur_sh = shape in
  let invalid msg error_shapes =
    raise @@ Row.Shape_error ("product_space_shape: " ^ msg, [ Shape_mismatch error_shapes ])
  in
  let spec, ls_rhs_list, ls_lhs, rhs_shapes =
    match logic with
    | Broadcast (Einsum (spec, _), sh1, sh2) -> (
        match einsum_of_spec spec with
        | [ ls1; ls2 ], ls_lhs -> (spec, [ ls1; ls2 ], ls_lhs, [ sh1; sh2 ])
        | _ -> invalid ("invalid binary einsum spec: " ^ spec) [ cur_sh; sh1; sh2 ])
    | Transpose (Permute (spec, _), sh1) -> (
        match einsum_of_spec spec with
        | [ ls1 ], ls_lhs -> (spec, [ ls1 ], ls_lhs, [ sh1 ])
        | _ -> invalid ("invalid permutation spec: " ^ spec) [ cur_sh; sh1 ])
    | Broadcast_tern (Einsum_tern (spec, _), sh1, sh2, sh3) -> (
        match einsum_of_spec spec with
        | [ ls1; ls2; ls3 ], ls_lhs -> (spec, [ ls1; ls2; ls3 ], ls_lhs, [ sh1; sh2; sh3 ])
        | _ -> invalid ("invalid ternary einsum spec: " ^ spec) [ cur_sh; sh1; sh2; sh3 ])
    | _ ->
        invalid
          "product-space intermediates (*_pspace) require an einsum-family operation (an einsum or \
           permutation spec)"
          [ cur_sh ]
  in
  let error_shapes = cur_sh :: rhs_shapes in
  let cannot msg =
    invalid ("cannot express the product space of " ^ spec ^ " as a shape: " ^ msg) error_shapes
  in
  (* Re-elaborate the spec with fresh (name-shared) variables and re-emit the operand equalities:
     the fresh variables converge to the operation's own solution, while the operation's
     delayed-variable bindings (Shape.set_dim targets) stay authoritative. *)
  let row_var_env = Hashtbl.create (module String) in
  let dim_var_env = Hashtbl.create (module String) in
  let constraints = ref [] in
  let elaborate ~label ls (sh : t) =
    let extras, _proj_env, ((b, i, o) as bio) =
      einsum_slot_spec_to_dims_bio ~original_spec:spec ~sh_id:sh.id ~row_var_env ~dim_var_env ls
    in
    let row_eq kind r1 r2 =
      Row.Row_eq
        {
          r1;
          r2;
          origin =
            [
              {
                Row.lhs_name = spec;
                lhs_kind = kind;
                rhs_name = sh.debug_name;
                rhs_kind = kind;
                operation = Some label;
              };
            ];
        }
    in
    constraints :=
      extras
      @ [ row_eq `Batch b sh.batch; row_eq `Input i sh.input; row_eq `Output o sh.output ]
      @ !constraints;
    bio
  in
  List.iteri (List.zip_exn ls_rhs_list rhs_shapes) ~f:(fun idx (ls, sh) ->
      ignore (elaborate ~label:(Printf.sprintf "Product-space ARGUMENT %d" (idx + 1)) ls sh));
  ignore (elaborate ~label:"Product-space RESULT" ls_lhs cur_sh);
  active_constraints := !constraints @ !active_constraints;
  (* The proxy's result rows: a SEPARATE, constraint-free elaboration (shared variable environments,
     so named variables coincide with the constrained ones) in which fixed-index axes are pinned to
     dimension 1 ([`Unit_dim]) — their projections are pinned, so they are never product axes
     regardless of their extent. The constrained elaboration above must not use [`Unit_dim]: it
     would pin the operation's actual fixed-index result axes. *)
  let _, _, (b_lhs, i_lhs, o_lhs) =
    einsum_slot_spec_to_dims_bio ~fixed_index:`Unit_dim ~original_spec:spec ~sh_id:cur_sh.id
      ~row_var_env ~dim_var_env ls_lhs
  in
  (* Contracted axes, from the parsed spec: labels (including convolution constituents and
     concatenation elements) and row variables of the RHS slots absent from the result, in
     first-occurrence order with each slot walked in the to_dims kind order (batch, output, input;
     leading axes, then the row variable, then trailing axes) — the order in which
     [derive_projections] first encounters them. Fixed-index axes are never product axes. *)
  let items_of_slot (ls : parsed_axis_labels) =
    let axis_items = function
      | Label l -> [ `Lbl l ]
      | Fixed_index _ -> []
      | Affine_spec { over_label; conv; _ } ->
          `Lbl over_label
          :: (match conv with Some { kernel_label; _ } -> [ `Lbl kernel_label ] | None -> [])
      | Concat_spec ls -> [ `Cnc ls ]
    in
    let row_items beg_specs bcast specs =
      List.concat_map beg_specs ~f:axis_items
      @ (match bcast with Some name -> [ `Rv name ] | None -> [])
      @ List.concat_map specs ~f:axis_items
    in
    row_items ls.given_beg_batch ls.bcast_batch ls.given_batch
    @ row_items ls.given_beg_output ls.bcast_output ls.given_output
    @ row_items ls.given_beg_input ls.bcast_input ls.given_input
  in
  let lhs_items = items_of_slot ls_lhs in
  let lhs_labels =
    Set.of_list
      (module String)
      (List.concat_map lhs_items ~f:(function `Lbl l -> [ l ] | `Cnc ls -> ls | `Rv _ -> []))
  in
  let lhs_rvars =
    Set.of_list
      (module String)
      (List.filter_map lhs_items ~f:(function `Rv name -> Some name | _ -> None))
  in
  let lookup_var name =
    Row.Var (Hashtbl.find_or_add dim_var_env name ~default:(fun () -> Row.get_var ~name ()))
  in
  let seen_labels = ref (Set.empty (module String)) in
  let seen_rvars = ref (Set.empty (module String)) in
  let contracted =
    List.concat_map ls_rhs_list ~f:(fun ls ->
        List.concat_map (items_of_slot ls) ~f:(function
          | `Lbl l ->
              if Set.mem lhs_labels l || Set.mem !seen_labels l then []
              else (
                seen_labels := Set.add !seen_labels l;
                [ `Dim (lookup_var l) ])
          | `Cnc ls ->
              let is_known l = Set.mem lhs_labels l || Set.mem !seen_labels l in
              if List.for_all ls ~f:is_known then []
              else if List.exists ls ~f:is_known then
                cannot
                  ("a concatenation axis is partially reduced over: " ^ String.concat ~sep:"^" ls)
              else (
                seen_labels := List.fold ls ~init:!seen_labels ~f:Set.add;
                [ `Dim (Row.Concat (List.map ls ~f:lookup_var)) ])
          | `Rv name ->
              if Set.mem lhs_rvars name || Set.mem !seen_rvars name then []
              else (
                seen_rvars := Set.add !seen_rvars name;
                let v =
                  Hashtbl.find_or_add row_var_env name ~default:(fun () -> Row.get_row_var ())
                in
                [ `Rvar v ])))
  in
  let contracted_dims_around idx =
    let dim_of = function `Dim d -> Some d | `Rvar _ -> None in
    ( List.filter_map (List.take contracted idx) ~f:dim_of,
      List.filter_map (List.drop contracted (idx + 1)) ~f:dim_of )
  in
  let input_row, output_row =
    match
      List.findi contracted ~f:(fun _ item -> match item with `Rvar _ -> true | _ -> false)
    with
    | None ->
        let extra = List.map contracted ~f:(function `Dim d -> d | `Rvar _ -> assert false) in
        (* Appending to [dims] places the contracted axes after an open row variable's expansion, so
           an open result input row needs no special case here. *)
        ({ i_lhs with dims = i_lhs.dims @ extra }, o_lhs)
    | Some (idx, `Rvar v) -> (
        if
          List.exists
            (List.drop contracted (idx + 1))
            ~f:(function `Rvar _ -> true | `Dim _ -> false)
        then
          cannot "the operation reduces over more than one row variable (unknown-rank axis groups)"
        else
          let before, after = contracted_dims_around idx in
          match (i_lhs.bcast, o_lhs.bcast) with
          | Row.Broadcastable, _ ->
              ( {
                  Row.beg_dims = i_lhs.beg_dims @ i_lhs.dims @ before;
                  bcast = Row.Row_var v;
                  dims = after;
                  prov = i_lhs.prov;
                },
                o_lhs )
          | Row.Row_var _, Row.Broadcastable when List.is_empty before && List.is_empty after ->
              (* Lone reduced-over row variable with an open result input row: place it in the
                 (closed) output row. The product order puts the result's input axes before the
                 reduced axes while this layout puts them after; a consistent layout is still
                 correct — every access goes through the same identity projection
                 ([Indexing.prod_project_for]), which pairs axes with product components by extent
                 (first-fit), and rejects a leftover on either side at lowering. *)
              ( i_lhs,
                {
                  Row.beg_dims = o_lhs.beg_dims @ o_lhs.dims;
                  bcast = Row.Row_var v;
                  dims = [];
                  prov = o_lhs.prov;
                } )
          | Row.Row_var _, _ ->
              cannot
                "the operation reduces over a row variable while the result's input row is open \
                 (and the reduced axes cannot go into the result's output row instead)")
    | Some (_, `Dim _) -> assert false
  in
  {
    batch = b_lhs;
    output = output_row;
    input = input_row;
    batch_padding = None;
    input_padding = None;
    output_padding = None;
    padding_elem = None;
    id = cur_sh.id;
    debug_name = "pspace:" ^ cur_sh.debug_name;
  }

(** Sets the dimension/total-elements for a delayed variable reference. For row variables, this
    creates a [Total_elems] constraint that will be reconciled during [finish_inference]. *)
let%track7_sexp set_dim (delayed_var_ref : delayed_var_ref) (dim : int) : unit =
  match delayed_var_ref with
  | { var_ref = { solved_sym = Some _; ref_label; _ }; _ } ->
      raise
      @@ Row.Shape_error
           ( "Cannot set a concrete dimension for variable reference with label " ^ ref_label
             ^ ": already set to a symbolic dimension",
             [] )
  | { var_ref = { solved_dim = Some dim2; _ }; _ } when dim2 = dim -> ()
  | { var_ref = { solved_dim = Some dim2; ref_label; _ }; _ } ->
      raise
      @@ Row.Shape_error
           ( "Cannot set dimension for variable reference with label " ^ ref_label,
             [ Row.Dim_mismatch [ Row.get_default_dim ~d:dim2 (); Row.get_default_dim ~d:dim () ] ]
           )
  | { var_ref = { solved_dim = None; _ }; var = `Not_set_yet; _ } ->
      delayed_var_ref.var_ref.solved_dim <- Some dim
  | { var_ref = { solved_dim = None; _ }; var = `Dim dim_var; _ } ->
      delayed_var_ref.var_ref.solved_dim <- Some dim;
      active_constraints :=
        Row.Dim_eq
          {
            d1 = Row.Var dim_var;
            d2 = Row.get_default_dim ~d:dim ();
            origin =
              [
                {
                  lhs_name = delayed_var_ref.var_ref.ref_label;
                  lhs_kind = `Output;
                  rhs_name = Int.to_string dim;
                  rhs_kind = `Output;
                  operation = Some "Shape.set_dim Dim";
                };
              ];
          }
        :: !active_constraints
  | { var_ref = { solved_dim = None; _ }; var = `Row row_var; _ } ->
      delayed_var_ref.var_ref.solved_dim <- Some dim;
      active_constraints :=
        Row.Rows_constr
          {
            (* TODO: actually, the Row.provenance should be the one of the shape that the row
               variable is in, should be stored in `Row and in env_row_var. *)
            r = [ Row.get_row_for_var Row.empty_provenance row_var ];
            constr = Total_elems { numerator = Num_elems dim; divided_by = []; keep_axis = false };
            origin =
              [
                {
                  lhs_name = delayed_var_ref.var_ref.ref_label;
                  lhs_kind = `Output;
                  rhs_name = Int.to_string dim;
                  rhs_kind = `Output;
                  operation = Some "Shape.set_dim Row";
                };
              ];
          }
        :: !active_constraints

(** Sets a delayed variable reference to a symbolic dimension (gh-490): the referenced dim variable
    is equated with the rigid symbolic extent [Sym sym]. Mirrors {!set_dim}; row variables are not
    supported (a symbolic total-elements constraint would require symbolic arithmetic). *)
let%track7_sexp set_sym_dim (delayed_var_ref : delayed_var_ref) (sym : Ir.Indexing.static_symbol) :
    unit =
  (* Eagerly validate the declared range (raises [Utils.User_error] if absent or non-positive). *)
  let sym_dim = Row.get_sym_dim sym in
  let origin =
    [
      {
        Row.lhs_name = delayed_var_ref.var_ref.ref_label;
        lhs_kind = `Output;
        rhs_name = Ir.Indexing.symbol_ident sym.Ir.Indexing.static_symbol;
        rhs_kind = `Output;
        operation = Some "Shape.set_sym_dim";
      };
    ]
  in
  match delayed_var_ref with
  | { var_ref = { solved_sym = Some sym2; _ }; _ } when Ir.Indexing.equal_static_symbol sym2 sym ->
      ()
  | { var_ref = { solved_sym = Some sym2; ref_label; _ }; _ } ->
      raise
      @@ Row.Shape_error
           ( "Cannot set symbolic dimension for variable reference with label " ^ ref_label
             ^ ": already set to a different symbolic dimension",
             [ Row.Dim_mismatch [ Row.get_sym_dim sym2; sym_dim ] ] )
  | { var_ref = { solved_dim = Some dim2; ref_label; _ }; _ } ->
      raise
      @@ Row.Shape_error
           ( "Cannot set symbolic dimension for variable reference with label " ^ ref_label
             ^ ": already set to a concrete dimension",
             [ Row.Dim_mismatch [ Row.get_default_dim ~d:dim2 (); sym_dim ] ] )
  | { var_ref = { solved_dim = None; _ }; var = `Not_set_yet; _ } ->
      delayed_var_ref.var_ref.solved_sym <- Some sym
  | { var_ref = { solved_dim = None; _ }; var = `Dim dim_var; _ } ->
      delayed_var_ref.var_ref.solved_sym <- Some sym;
      active_constraints :=
        Row.Dim_eq { d1 = Row.Var dim_var; d2 = sym_dim; origin } :: !active_constraints
  | { var_ref = { solved_dim = None; ref_label; _ }; var = `Row _; _ } ->
      raise
      @@ Row.Shape_error
           ( "Cannot set symbolic dimension for variable reference with label " ^ ref_label
             ^ ": it is bound to a row variable (set_sym_dim does not support row variables)",
             [] )

let set_equal delayed_ref1 delayed_ref2 =
  (* TODO: use provenance from the row variables once we have it there. *)
  match (delayed_ref1, delayed_ref2) with
  | { var_ref = { solved_sym = Some sym1; _ }; _ }, { var_ref = { solved_sym = Some sym2; _ }; _ }
    ->
      if Ir.Indexing.equal_static_symbol sym1 sym2 then ()
      else
        raise
        @@ Row.Shape_error
             ( "Cannot set equal dimensions for variable references bound to different symbolic \
                dimensions",
               [ Row.Dim_mismatch [ Row.get_sym_dim sym1; Row.get_sym_dim sym2 ] ] )
  | { var_ref = { solved_sym = Some sym; _ }; _ }, delayed_ref2 ->
      (* First is symbolic, second is not - set the second to match the first (raises if the second
         is already solved to a concrete dimension). *)
      set_sym_dim delayed_ref2 sym
  | delayed_ref1, { var_ref = { solved_sym = Some sym; _ }; _ } -> set_sym_dim delayed_ref1 sym
  | { var_ref = { solved_dim = Some dim1; _ }; _ }, { var_ref = { solved_dim = Some dim2; _ }; _ }
    ->
      if dim1 = dim2 then ()
      else
        raise
        @@ Row.Shape_error
             ( "Cannot set equal dimensions for variable references with different values",
               [
                 Row.Dim_mismatch [ Row.get_default_dim ~d:dim1 (); Row.get_default_dim ~d:dim2 () ];
               ] )
  | { var_ref = { solved_dim = Some dim; _ }; _ }, delayed_ref2 ->
      (* First is solved, second is not - set the second to match the first *)
      set_dim delayed_ref2 dim
  | delayed_ref1, { var_ref = { solved_dim = Some dim; _ }; _ } ->
      (* Second is solved, first is not - set the first to match the second *)
      set_dim delayed_ref1 dim
  | ( { var_ref = { solved_dim = None; ref_label = ref_label1; _ }; var = _; _ },
      { var_ref = { solved_dim = None; ref_label = ref_label2; _ }; var = `Not_set_yet; _ } )
  | ( { var_ref = { solved_dim = None; ref_label = ref_label1; _ }; var = `Not_set_yet; _ },
      { var_ref = { solved_dim = None; ref_label = ref_label2; _ }; var = _; _ } ) ->
      raise
      @@ Row.Shape_error
           ( "set_equal: insufficient information between labels " ^ ref_label1 ^ " and "
             ^ ref_label2,
             [] )
  | { var = `Dim dim_var1; _ }, { var = `Dim dim_var2; _ } ->
      (* Both are dimension variables - create equality constraint *)
      active_constraints :=
        Row.Dim_eq
          {
            d1 = Row.Var dim_var1;
            d2 = Row.Var dim_var2;
            origin =
              [
                {
                  lhs_name = delayed_ref1.var_ref.ref_label;
                  lhs_kind = `Output;
                  rhs_name = delayed_ref2.var_ref.ref_label;
                  rhs_kind = `Output;
                  operation = Some "Shape.set_equal Dim-Dim";
                };
              ];
          }
        :: !active_constraints
  | { var = `Row row_var1; _ }, { var = `Row row_var2; _ } ->
      (* Both are row variables - create row equality constraint *)
      active_constraints :=
        Row.Row_eq
          {
            r1 = Row.get_row_for_var Row.empty_provenance row_var1;
            r2 = Row.get_row_for_var Row.empty_provenance row_var2;
            origin =
              [
                {
                  lhs_name = delayed_ref1.var_ref.ref_label;
                  lhs_kind = `Output;
                  rhs_name = delayed_ref2.var_ref.ref_label;
                  rhs_kind = `Output;
                  operation = Some "Shape.set_equal Row-Row";
                };
              ];
          }
        :: !active_constraints
  | { var = `Dim dim_var; _ }, { var = `Row row_var; _ }
  | { var = `Row row_var; _ }, { var = `Dim dim_var; _ } ->
      (* One is dim var, one is row var - equality via Total_elems constraint *)
      active_constraints :=
        Row.Rows_constr
          {
            r = [ Row.get_row_for_var Row.empty_provenance row_var ];
            constr =
              Total_elems
                {
                  numerator =
                    Strided_var
                      {
                        coeff = Utils.safe_lazy "set_equal_dim_row" (fun () -> 1);
                        var = dim_var;
                        denom = 1;
                        round_up = false;
                      };
                  divided_by = [];
                  keep_axis = false;
                };
            origin =
              [
                {
                  lhs_name = delayed_ref1.var_ref.ref_label;
                  lhs_kind = `Output;
                  rhs_name = delayed_ref2.var_ref.ref_label;
                  rhs_kind = `Output;
                  operation = Some "Shape.set_equal Dim-Row or Row-Dim";
                };
              ];
          }
        :: !active_constraints

let set_scale ~factor delayed_ref_large delayed_ref_small =
  if factor < 1 then
    raise
    @@ Row.Shape_error ("Shape.set_scale: factor must be >= 1, got " ^ Int.to_string factor, []);
  (* Row-variable guard: reject any ref whose var is `Row _, regardless of which side is solved.
     Without this, the solved-side branches below would forward to set_dim on a Row ref, which
     silently emits a Rows_constr Total_elems constraint -- out of scope for this API. *)
  (match (delayed_ref_large.var, delayed_ref_small.var) with
  | `Row _, _ | _, `Row _ ->
      raise
      @@ Row.Shape_error
           ( "Shape.set_scale: requires two dimension variables; got at least one row variable \
              (labels " ^ delayed_ref_large.var_ref.ref_label ^ ", "
             ^ delayed_ref_small.var_ref.ref_label ^ ")",
             [] )
  | _ -> ());
  (* gh-490 v1: scaling relations over symbolic dimensions would require symbolic arithmetic. *)
  if
    Option.is_some delayed_ref_large.var_ref.solved_sym
    || Option.is_some delayed_ref_small.var_ref.solved_sym
  then
    raise
    @@ Row.Shape_error
         ( "Shape.set_scale: symbolic dimensions are not supported (labels "
           ^ delayed_ref_large.var_ref.ref_label ^ ", " ^ delayed_ref_small.var_ref.ref_label ^ ")",
           [] );
  if factor = 1 then set_equal delayed_ref_large delayed_ref_small
  else
    match (delayed_ref_large, delayed_ref_small) with
    | ( { var_ref = { solved_dim = Some d_large; _ }; _ },
        { var_ref = { solved_dim = Some d_small; _ }; _ } ) ->
        if d_large = factor * d_small then ()
        else
          raise
          @@ Row.Shape_error
               ( "Shape.set_scale: dimensions do not satisfy d_large = factor * d_small",
                 [
                   Row.Dim_mismatch
                     [
                       Row.get_default_dim ~d:d_large ();
                       Row.get_default_dim ~d:(factor * d_small) ();
                     ];
                 ] )
    | { var_ref = { solved_dim = Some d_large; ref_label; _ }; _ }, delayed_ref_small ->
        if d_large % factor <> 0 then
          raise
          @@ Row.Shape_error
               ( "Shape.set_scale: solved dimension " ^ Int.to_string d_large ^ " for " ^ ref_label
                 ^ " is not divisible by factor " ^ Int.to_string factor,
                 [] );
        set_dim delayed_ref_small (d_large / factor)
    | delayed_ref_large, { var_ref = { solved_dim = Some d_small; _ }; _ } ->
        set_dim delayed_ref_large (factor * d_small)
    | ( { var_ref = { solved_dim = None; ref_label = label_l; _ }; var = var_l; _ },
        { var_ref = { solved_dim = None; ref_label = label_s; _ }; var = var_s; _ } ) -> (
        match (var_l, var_s) with
        | `Dim dim_var_large, `Dim dim_var_small ->
            active_constraints :=
              Row.Dim_eq
                {
                  d1 = Row.Var dim_var_large;
                  d2 =
                    Row.Affine
                      {
                        stride = factor;
                        over = Row.Var dim_var_small;
                        conv = None;
                        stride_offset = 0;
                      };
                  origin =
                    [
                      {
                        lhs_name = label_l;
                        lhs_kind = `Output;
                        rhs_name = Int.to_string factor ^ "*" ^ label_s;
                        rhs_kind = `Output;
                        operation = Some "Shape.set_scale Dim-Dim";
                      };
                    ];
                }
              :: !active_constraints
        | `Not_set_yet, _ | _, `Not_set_yet ->
            raise
            @@ Row.Shape_error
                 ( "Shape.set_scale: insufficient information between labels " ^ label_l ^ " and "
                   ^ label_s,
                   [] )
        | `Row _, _ | _, `Row _ ->
            (* Defensive: already caught by the top-of-function row guard. *)
            assert false)

let set_terminal ~is_param (sh : t) =
  Hash_set.add unused_shapes sh.id;
  let get_origin kind =
    Row.
      {
        lhs_name = sh.debug_name;
        lhs_kind = kind;
        rhs_name = "(parameter)";
        rhs_kind = kind;
        operation = Some "set_terminal";
      }
  in
  active_constraints :=
    Row.Terminal_row (is_param, sh.batch, [ get_origin `Batch ])
    :: Row.Terminal_row (is_param, sh.input, [ get_origin `Input ])
    :: Row.Terminal_row (is_param, sh.output, [ get_origin `Output ])
    :: !active_constraints

(* Marks the shape's rows as resolved by their use sites (gh-ocannl-544): their row variables may
   close to the GLB of their use sites' rows at the final solver stages, where operation results
   close to an empty remainder by default. Substitutes through the current environment so that a
   variable already unified away transfers the mark to its representative; later unifications
   propagate it further inside [Row.unify_row]. *)
let set_resolve_at_use (sh : t) =
  let mark row =
    let row = Row.subst_row !state row in
    (match row.bcast with Row.Row_var v -> Row.add_resolve_at_use v | Row.Broadcastable -> ());
    List.iter (row.beg_dims @ row.dims) ~f:(fun d ->
        Set.iter (Row.vars_of_dim d) ~f:Row.add_resolve_at_use_dim)
  in
  mark sh.batch;
  mark sh.input;
  mark sh.output

let unsafe_reinitialize () =
  update_uid := 0;
  state := Row.empty_env;
  active_update_steps := [];
  active_constraints := [];
  construction_depth := 0

let iter_shapes update_step ~f =
  f update_step.shape;
  match update_step.logic with
  | Terminal _ -> ()
  | Transpose (_, sh1) -> f sh1
  | Broadcast (_, sh1, sh2) ->
      f sh1;
      f sh2
  | Broadcast_tern (_, sh1, sh2, sh3) ->
      f sh1;
      f sh2;
      f sh3
  | Block { rhses; _ } -> List.iter rhses ~f

let all_rows_w_origin update_step =
  let get_origin sh kind =
    Row.
      {
        lhs_name = sh.debug_name;
        lhs_kind = kind;
        rhs_name = "";
        rhs_kind = kind;
        operation = Some "remaining rows";
      }
  in
  let rows_sh sh =
    [
      (sh.batch, get_origin sh `Batch);
      (sh.input, get_origin sh `Input);
      (sh.output, get_origin sh `Output);
    ]
  in
  rows_sh update_step.shape
  @
  match update_step.logic with
  | Terminal _ -> []
  | Transpose (_, sh1) -> rows_sh sh1
  | Broadcast (_, sh1, sh2) -> rows_sh sh1 @ rows_sh sh2
  | Broadcast_tern (_, sh1, sh2, sh3) -> rows_sh sh1 @ rows_sh sh2 @ rows_sh sh3
  | Block { rhses; _ } -> List.concat_map rhses ~f:rows_sh

(** {3 Projection inference} *)

let fresh_proj_ids update =
  let resolved_padding = ref [] in
  let inferred_padding = ref [] in
  let no_padding = Ir.Ops.{ left = 0; right = 0 } in
  (* [locked = None]: the tensor node's layout is not committed yet — register the shape's padding
     (if any) as inferred. [locked = Some arr_opt]: the tnode's padding lazy is forced, so the
     buffer layout is committed — register the TNODE's padding (zeros when unpadded) as resolved for
     every axis, so that [Row.solve_proj_equations] rejects later demands exceeding it (a
     forced-at-[None] node used to register nothing, silently losing the halo; see
     [check_and_update_padding]). [offset] is the axis position within the tnode's padding array
     (axis order batch;output;input, cf. {!to_padding}). Returns the offset advanced past this
     row. *)
  let fetch_padding ~locked row row_padding ~offset =
    let dims = Array.of_list (row.Row.beg_dims @ row.Row.dims) in
    (match locked with
    | None ->
        Option.iter row_padding ~f:(fun padding ->
            Array.iter2_exn dims padding ~f:(fun d p ->
                match d with
                | Row.Dim { proj_id = Some proj_id; _ } | Row.Sym { sym_proj_id = Some proj_id; _ }
                  ->
                    inferred_padding := (proj_id, p) :: !inferred_padding
                | _ -> ()))
    | Some locked_arr ->
        Array.iteri dims ~f:(fun i d ->
            let p =
              match locked_arr with
              | Some arr when offset + i < Array.length arr -> arr.(offset + i)
              | _ -> no_padding
            in
            match d with
            | Row.Dim { proj_id = Some proj_id; _ } | Row.Sym { sym_proj_id = Some proj_id; _ } ->
                resolved_padding := (proj_id, p) :: !resolved_padding
            | _ -> ()));
    offset + Array.length dims
  in
  let fresh_shape (sh : t) =
    sh.batch <- Row.fresh_row_proj sh.batch;
    sh.input <- Row.fresh_row_proj sh.input;
    sh.output <- Row.fresh_row_proj sh.output;
    let locked =
      match Ir.Tnode.find ~id:sh.id with
      | Some tn when Lazy.is_val tn.Ir.Tnode.padding -> (
          (* [Lazy.is_val] can be true for a lazy currently being forced (forcing it raises
             [Lazy.Undefined]) — mid-force means this very derivation is part of committing the
             tnode's layout, so the padding is not settled yet: treat as not locked. *)
          try
            let p = Lazy.force tn.Ir.Tnode.padding in
            (* Reconcile the committed neutral element into the shape: nodes created with an
               already-padded layout (e.g. [wrap_padded]) commit their [padded_value] at creation
               while the shape's [padding_elem] starts unknown — without this, a later
               margin-reading consumer with a different neutral would commit its own neutral at the
               shape level unchecked and silently read the buffer's halo values (Codex P1 on PR
               #173). [update_padding_elem] then rejects conflicting consumers. *)
            (match p with
            | Some (_, v) -> (
                match sh.padding_elem with
                | None -> sh.padding_elem <- Some (Some v)
                | Some (Some v1) when Float.( = ) v1 v -> ()
                | Some _ ->
                    raise
                    @@ Row.Shape_error
                         ( [%string
                             "Conflicting padding neutral elements: the tensor's buffer committed \
                              margins holding %{v#Float}, but a different neutral was inferred for \
                              its shape. Materialize a separate copy of the tensor for the \
                              conflicting padded consumers (for a padded max-pool, \
                              Nn_blocks.max_pool2d_copy)"],
                           [ Shape_mismatch [ sh ] ] ))
            | None -> ());
            Some (Option.map ~f:fst p)
          with Lazy.Undefined -> None)
      | _ -> None
    in
    let offset = fetch_padding ~locked sh.batch sh.batch_padding ~offset:0 in
    let offset = fetch_padding ~locked sh.output sh.output_padding ~offset in
    ignore (fetch_padding ~locked sh.input sh.input_padding ~offset : int)
  in
  fresh_shape update.shape;
  (match update.logic with
  | Broadcast (Defined_by_cd_logic, _, _)
  | Transpose (Defined_by_cd_logic, _)
  | Broadcast_tern (Defined_by_cd_logic, _, _, _) ->
      ()
  | Terminal _ -> ()
  | Transpose (_, sh) -> fresh_shape sh
  | Broadcast (_, sh1, sh2) ->
      fresh_shape sh1;
      fresh_shape sh2
  | Broadcast_tern (_, sh1, sh2, sh3) ->
      fresh_shape sh1;
      fresh_shape sh2;
      fresh_shape sh3
  | Block { rhses; _ } -> List.iter rhses ~f:fresh_shape);
  (!resolved_padding, !inferred_padding)

let%debug4_sexp row_to_dims (row : Row.t) : int array =
  let open Row in
  let rec f = function
    | Dim { d; _ } -> d
    | Sym s ->
        (* Reconciliation with Tnode dims: a symbolic axis materializes at its maximum extent (the
           declared range of its static symbol), which sizes allocations and host transfers. *)
        sym_dim_extent_exn s
    | Var v ->
        raise
        @@ Row.Shape_error
             ( "Not enough shape information: unresolved variable "
               ^ Sexp.to_string_hum ([%sexp_of: dim_var] v),
               [ Row_mismatch [ row ] ] )
    | Affine _ ->
        (* FIXME: reconsider this, we could return the input dimension of the convolution. *)
        raise
        @@ Row.Shape_error
             ( "Not enough shape information: affine dimension cannot be converted to single int",
               [ Row_mismatch [ row ] ] )
    | Concat dims -> List.sum (module Int) dims ~f
  in
  match row with
  | { bcast = Row_var v; _ } ->
      raise
      @@ Row.Shape_error
           ( "Not enough shape information: unresolved row variable "
             ^ Sexp.to_string_hum ([%sexp_of: row_var] v),
             [ Row_mismatch [ row ] ] )
  | { beg_dims; dims; bcast = Broadcastable; prov = _ } -> Array.of_list_map (beg_dims @ dims) ~f

let to_dims_impl (sh : t) : int array =
  try Array.concat_map ~f:row_to_dims [| sh.batch; sh.output; sh.input |]
  with Row.Shape_error (s, trace) -> raise @@ Row.Shape_error (s, Shape_mismatch [ sh ] :: trace)

(** Computes the indexing into subtensors given the shape information of a tensor.
    [derive_projections] should only be invoked when the shapes are fully inferred already! Sets
    [unsafe_projections] as a side effect. Raises if projections were already computed. *)
let%debug4_sexp derive_projections (update_step : update_step) : unit =
  if Option.is_some update_step.unsafe_projections then
    raise
    @@ Utils.User_error "derive_projections: projections already computed for this update_step";
  let resolved_padding, _old_inferred_padding = fresh_proj_ids update_step in
  (* We will not use the old inferred padding so that we can derive precisely the padding
     contributed by this step. *)
  let _debug_update_step : update_step = update_step in
  let (proj_axis_env, discardable_vars, ineqs) :
      proj_axis_env * Row.dim_var_set * Row.constraint_ list =
    get_inequalities ~for_projections:true update_step
  in
  (* We need to solve the equations/inequalities one last time because of fresh row variables
     potentially generated by [get_inequalities]. Since the variables in the shapes must be
     substituted-out at this point, the global state is already an empty env, but in principle we
     want to only find a local solution to not contaminate projections across operations. *)
  let unsolved, local_env =
    Row.solve_inequalities ~stage:Stage1 ~discardable_vars ineqs Row.empty_env
  in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage2 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage3 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage4 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage5 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage6 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage7 unsolved local_env in
  assert (List.is_empty unsolved);
  let local_env = Row.populate_dim_proj_in_solved local_env in
  let rhs, skip_deriving =
    match update_step.logic with
    | Broadcast (Defined_by_cd_logic, _, _)
    | Transpose (Defined_by_cd_logic, _)
    | Broadcast_tern (Defined_by_cd_logic, _, _, _) ->
        ([], true)
    | Terminal _ -> ([], false)
    | Transpose (_, sh) -> ([ sh ], false)
    | Broadcast (_, sh1, sh2) -> ([ sh1; sh2 ], false)
    | Broadcast_tern (_, sh1, sh2, sh3) -> ([ sh1; sh2; sh3 ], false)
    | Block { rhses; _ } -> (rhses, false)
  in
  let terminal_rows_for_inputs =
    List.concat_map rhs ~f:(fun sh ->
        [
          Row.Terminal_row (false, sh.batch, []);
          Row.Terminal_row (false, sh.input, []);
          Row.Terminal_row (false, sh.output, []);
        ])
  in
  (* Important: ineqs must not be substituted / solved before getting proj_equations, because
     get_inequalities provides indexing information that is lost after substitution. *)
  let proj_eqs : Row.proj_equation list =
    Row.get_proj_equations (terminal_rows_for_inputs @ ineqs) proj_axis_env local_env
  in
  (* resolved_padding is passed for verification only - to check that operation padding doesn't
     exceed already-locked padding. It won't be used to set padding on shapes (get_dim_padding only
     returns inferred_padding of proj_env, which is for the current operation only). *)
  (* gh-504: a max/tropical-family accumulation (non-finite identity) lowers its padded ([=]-mode)
     window projections with clamped window bounds instead of demanding margins on the operand —
     skipping an out-of-range window position contributes the accumulation identity exactly. With
     no demand registered, [step_touches_margins] stays false for this step, so no neutral element
     is committed either: the operand composes freely with 0-neutral margin-touching consumers
     (padded convs — the Inception-block pattern). *)
  let clamp_padded =
    match update_step.neutral_elem with Some v -> not (Float.is_finite v) | None -> false
  in
  let proj_env : Row.proj_env =
    Row.solve_proj_equations ~clamp_padded ~resolved_padding ~inferred_padding:[] proj_eqs
  in
  let dims_of (sh : t) =
    sh.batch.beg_dims @ sh.batch.dims @ sh.output.beg_dims @ sh.output.dims @ sh.input.beg_dims
    @ sh.input.dims
  in
  let lhs = update_step.shape in
  let lhs_dims = to_dims_impl lhs in
  let rhs_dims = Array.of_list_map ~f:to_dims_impl rhs in
  let all_dims : Row.dim list = List.concat_map ~f:dims_of @@ (lhs :: rhs) in
  (* Note: the ordering will affect performance of naive backends. *)
  let all_product_projs : (Row.proj_id * int) list =
    Utils.unique_keep_first ~equal:(fun (p, _) (q, _) -> Row.equal_proj_id p q)
    @@ List.filter_map all_dims ~f:(Row.get_product_proj proj_env)
  in
  (* Deduplicate by iterator symbol. When Conv_input with d=1 over dimension is used, multiple
     proj_ids can share the same iterator (e.g., input height and kernel height). We keep the first
     occurrence for each unique iterator symbol. *)
  let all_product_projs_with_iters =
    List.map all_product_projs ~f:(fun (p, d) -> (p, d, Row.proj_to_iterator_exn proj_env p))
  in
  let unique_by_iterator =
    Utils.unique_keep_first
      ~equal:(fun (_, _, s1) (_, _, s2) -> Idx.equal_symbol s1 s2)
      all_product_projs_with_iters
  in
  (* gh-490 symbolic extents: associate each symbolic axis's product iterator with the axis's static
     symbol. The product-space entry is the maximum extent; lowering guards loop bodies with
     [iterator < value] when the symbol is bound. A maximum-one axis has no iterator, but retain its
     symbol under [None] so a consumer such as [Set_vec_unop] can reject the dynamic extent rather
     than silently treating it as the maximum (gh-ocannl-817). Distinct symbols sharing an iterator
     cannot arise (a [Sym] unifies only with itself), but check defensively. *)
  let extent_syms : (Idx.symbol option * Idx.static_symbol) list =
    let all =
      List.filter_map all_dims ~f:(fun dim ->
          match dim with
          | Row.Sym { sym; _ } ->
              let iter =
                Option.bind (Row.get_product_proj proj_env dim) ~f:(fun (p, _) ->
                    try Some (Row.proj_to_iterator_exn proj_env p) with Invalid_argument _ -> None)
              in
              Some (iter, sym)
          | _ -> None)
      |> List.dedup_and_sort ~compare:[%compare: Idx.symbol option * Idx.static_symbol]
    in
    List.iter all ~f:(fun (it, sym) ->
        List.iter all ~f:(fun (it2, sym2) ->
            match (it, it2) with
            | Some it, Some it2
              when Idx.equal_symbol it it2 && not (Idx.equal_static_symbol sym sym2) ->
                raise
                @@ Row.Shape_error
                     ( "derive_projections: distinct symbolic extents share a product iterator",
                       [ Shape_mismatch (lhs :: rhs) ] )
            | Some _, Some _ | Some _, None | None, Some _ | None, None -> ()));
    all
  in
  (* Most product projections occur as top-level dimensions of the result or operands. Constituents
     that occur only inside a compound projection do not: notably a unary convolution's kernel, and
     concat components. Include those solver-recorded iterators after the shape-derived axes. *)
  let all_product_dim_entries =
    Row.product_dim_iterators proj_env
    |> Utils.unique_keep_first ~equal:(fun (_, _, s1) (_, _, s2) -> Idx.equal_symbol s1 s2)
  in
  let unique_by_iterator =
    let seen =
      Set.of_list (module Idx.Symbol) (List.map unique_by_iterator ~f:(fun (_, _, s) -> s))
    in
    unique_by_iterator
    @ List.filter all_product_dim_entries ~f:(fun (_, _, s) -> not (Set.mem seen s))
  in
  let symbol_to_proj =
    Map.of_alist_exn
      (module Idx.Symbol)
      (List.map all_product_dim_entries ~f:(fun (p, d, s) -> (s, (p, d))))
  in
  (* Build connected components from Concat indices. Symbols that appear together in a Concat must
     be iterated together. We use union-find to group symbols into connected components. Include
     both product dimensions and Concat dimensions. Note: Concat can appear either as Row.Concat dim
     or as a regular Dim whose proj_id maps to Idx.Concat. *)
  let product_indices : Idx.axis_index list =
    List.filter_map all_dims ~f:(fun dim ->
        match dim with
        | Row.Concat _ -> Some (Row.get_dim_index proj_env dim)
        | Dim { proj_id = Some _; _ } -> (
            match Row.get_product_proj proj_env dim with
            | Some _ -> Some (Row.get_dim_index proj_env dim)
            | None -> (
                (* Also check if dim's projection maps to Idx.Concat *)
                try
                  match Row.get_dim_index proj_env dim with
                  | Idx.Concat _ as idx -> Some idx
                  | _ -> None
                with _ -> None))
        | _ -> (
            match Row.get_product_proj proj_env dim with
            | Some _ -> Some (Row.get_dim_index proj_env dim)
            | None -> None))
  in
  let concat_groups : Idx.symbol list list =
    List.filter_map product_indices ~f:(function Idx.Concat syms -> Some syms | _ -> None)
  in
  let unique_by_iterator =
    let seen =
      Set.of_list (module Idx.Symbol) (List.map unique_by_iterator ~f:(fun (_, _, s) -> s))
    in
    let missing_symbols =
      concat_groups |> List.concat
      |> Utils.unique_keep_first ~equal:Idx.equal_symbol
      |> List.filter ~f:(fun s -> not (Set.mem seen s))
    in
    let missing_entries =
      List.filter_map missing_symbols ~f:(fun s ->
          Map.find symbol_to_proj s |> Option.map ~f:(fun (p, d) -> (p, d, s)))
    in
    unique_by_iterator @ missing_entries
  in
  (* Union-find to group symbols that appear in the same Concat. Symbols within the same Concat must
     be in the same iteration group so that they are iterated SEQUENTIALLY (via unflat_lines), not
     NESTED. For (a, b) ++^ "a; b => a^b", i2 and i3 should be in the same group: - First iterate i2
     (RHS[0] active) - Then iterate i3 (RHS[1] active) This is achieved by making them part of the
     same product component. *)
  let symbol_classes : (Idx.symbol, Idx.symbol) Hashtbl.t = Hashtbl.create (module Idx.Symbol) in
  let find_repr sym =
    let rec loop s =
      match Hashtbl.find symbol_classes s with
      | None -> s
      | Some parent when Idx.equal_symbol parent s -> s
      | Some parent ->
          let repr = loop parent in
          Hashtbl.set symbol_classes ~key:s ~data:repr;
          repr
    in
    loop sym
  in
  let union sym1 sym2 =
    let r1 = find_repr sym1 and r2 = find_repr sym2 in
    if not (Idx.equal_symbol r1 r2) then Hashtbl.set symbol_classes ~key:r1 ~data:r2
  in
  (* Union all symbols within each Concat group to make them iterate sequentially *)
  List.iter concat_groups ~f:(fun syms ->
      match syms with [] -> () | first :: rest -> List.iter rest ~f:(fun s -> union first s));
  (* Group unique_by_iterator entries by their representative symbol *)
  let components : (Idx.symbol, (Row.proj_id * int * Idx.symbol) list) Hashtbl.t =
    Hashtbl.create (module Idx.Symbol)
  in
  List.iter unique_by_iterator ~f:(fun ((_, _, sym) as entry) ->
      let repr = find_repr sym in
      Hashtbl.update components repr ~f:(function None -> [ entry ] | Some l -> entry :: l));
  (* Convert to arrays, preserving order by first occurrence *)
  let seen_reprs = Hash_set.create (module Idx.Symbol) in
  let ordered_components =
    List.filter_map unique_by_iterator ~f:(fun (_, _, sym) ->
        let repr = find_repr sym in
        if Hash_set.mem seen_reprs repr then None
        else (
          Hash_set.add seen_reprs repr;
          Hashtbl.find components repr))
  in
  let product_components : Idx.component array =
    Array.of_list_map ordered_components ~f:(fun entries ->
        List.map entries ~f:(fun (_, d, s) -> (d, s)))
  in
  let indices_of_sh (sh : t) =
    Array.of_list_map ~f:(Row.get_dim_index proj_env)
    @@ List.concat
         [
           sh.batch.beg_dims;
           sh.batch.dims;
           sh.output.beg_dims;
           sh.output.dims;
           sh.input.beg_dims;
           sh.input.dims;
         ]
  in
  (* Extract padding from proj_env and set the padding fields on shapes *)
  let update_padding old_padding new_padding =
    let default = Ir.Ops.{ left = 0; right = 0 } in
    Option.value ~default
    @@ Option.merge old_padding new_padding ~f:(fun old new_p ->
        Ir.Ops.{ left = max old.left new_p.left; right = max old.right new_p.right })
  in
  let padding_of_row old_padding (row : Row.t) : padding =
    let paddings =
      (* Guard against insufficient shape information error. *)
      ignore (row_to_dims row);
      List.mapi (row.beg_dims @ row.dims) ~f:(fun i d ->
          update_padding
            (Option.map old_padding ~f:(fun p -> p.(i)))
            (Row.get_dim_padding proj_env d))
    in
    (* Only return Some if at least one dimension has non-zero padding *)
    if List.for_all paddings ~f:(fun p -> p.left = 0 && p.right = 0) then None
    else Some (Array.of_list paddings)
  in
  let set_padding (sh : t) : unit =
    Option.iter (padding_of_row sh.batch_padding sh.batch) ~f:(fun p -> sh.batch_padding <- Some p);
    Option.iter (padding_of_row sh.output_padding sh.output) ~f:(fun p ->
        sh.output_padding <- Some p);
    Option.iter (padding_of_row sh.input_padding sh.input) ~f:(fun p -> sh.input_padding <- Some p)
  in
  (* Whether THIS step demands nonzero padding on [sh] — i.e. actually reads (rhs) or writes (lhs)
     the margins. [Row.get_dim_padding] only reports the current operation's inferred padding. *)
  let step_touches_margins (sh : t) : bool =
    List.exists [ sh.batch; sh.output; sh.input ] ~f:(fun row ->
        List.exists (row.Row.beg_dims @ row.Row.dims) ~f:(fun d ->
            match Row.get_dim_padding proj_env d with
            | Some Ir.Ops.{ left; right } -> left > 0 || right > 0
            | None -> false))
  in
  let update_padding_elem (sh : t) : unit =
    (* The padding neutral element commits like the rest of the tensor's identity: the margins of a
       padded buffer permanently hold a single neutral value. Only operations that actually read the
       margins (demand nonzero padding on [sh] in this step) participate — a valid-window reader
       never sees the margins, so its accumulation neutral is irrelevant. A genuine conflict is
       rejected: the remedy is a materialized copy per neutral. *)
    if step_touches_margins sh then
      sh.padding_elem <-
        (match (sh.padding_elem, update_step.neutral_elem) with
        | None, None -> None (* Both unknown *)
        | None, Some v -> Some (Some v) (* First operation sets the value *)
        | Some (Some v1), Some v2 when Float.( = ) v1 v2 -> sh.padding_elem (* Consistent *)
        | Some (Some v1), Some v2 ->
            raise
            @@ Row.Shape_error
                 ( [%string
                     "Conflicting padding neutral elements: an operation touches the tensor's \
                      margins expecting %{v2#Float}, but they are committed to %{v1#Float} \
                      (demanded by an earlier operation). Materialize a separate copy of the \
                      tensor for one of the padded consumers (for a padded max-pool, \
                      Nn_blocks.max_pool2d_copy)"],
                   [ Shape_mismatch [ sh ] ] )
        | Some None, _ -> Some None (* Unreachable: conflicts raise instead. *)
        | Some _, None -> sh.padding_elem)
    (* Operation has no neutral elem, keep current *)
  in
  if skip_deriving then ()
  else (
    set_padding lhs;
    List.iter rhs ~f:set_padding;
    (* Update padding_elem based on the operation's neutral element: for rhs shapes the operation
       reads the margins; for the lhs, a padded write projection (scatter-style [=] output specs)
       commits the accumulation's neutral just the same — this keeps "padded implies a known neutral
       element" an invariant (cf. [to_padding]). *)
    update_padding_elem lhs;
    List.iter rhs ~f:update_padding_elem;
    let project_lhs = indices_of_sh lhs in
    let project_rhs = Array.of_list_map ~f:indices_of_sh rhs in
    let projections =
      try
        Idx.
          {
            components = product_components;
            lhs_dims;
            rhs_dims;
            project_lhs;
            project_rhs;
            extent_syms;
            debug_info =
              {
                spec = logic_to_spec update_step.logic;
                derived_for = sexp_of_update_step update_step;
                trace = [ ("derive_projections", Idx.unique_debug_id ()) ];
              };
          }
      with Row.Shape_error (s, trace) ->
        raise @@ Row.Shape_error (s, Shape_mismatch (lhs :: rhs) :: trace)
    in
    update_step.unsafe_projections <- Some projections)

(** {3 Shape inference} *)

let apply_env_t env sh =
  sh.batch <- Row.subst_row env sh.batch;
  sh.input <- Row.subst_row env sh.input;
  sh.output <- Row.subst_row env sh.output

(** Computes the product of dimensions in a row (both flanks). Returns [None] if any dimension is
    not yet resolved (still a variable or unresolved affine), or if the row ends in a row variable
    that is not yet resolved -- the row could still gain axes, so its product is unknown. *)
let compute_row_product env (row : Row.t) : int option =
  let rec product = function
    | [] -> Some 1
    | dim :: rest -> (
        let dim_val =
          match dim with
          | Row.Dim { d; _ } -> Some d
          | Row.Sym _ -> None (* Symbolic dims don't participate in row-product arithmetic. *)
          | Row.Var v -> Row.get_dim_val env v
          | Row.Affine _ -> None (* TODO: handle affine/convolution input dimensions *)
          | Row.Concat dims ->
              List.fold dims ~init:(Some 0) ~f:(fun acc d ->
                  match acc with
                  | None -> None
                  | Some sum -> (
                      match d with
                      | Row.Dim { d; _ } -> Some (sum + d)
                      | Row.Var v -> Option.map (Row.get_dim_val env v) ~f:(fun d -> sum + d)
                      | _ -> None))
        in
        match (dim_val, product rest) with
        | Some d, Some rest_product -> Some (d * rest_product)
        | _ -> None)
  in
  (* Fuel guards against reference cycles among row variables. *)
  let rec row_product fuel (row : Row.t) =
    match (product (row.beg_dims @ row.dims), row.bcast) with
    | None, _ -> None
    | Some p, Row.Broadcastable -> Some p
    | Some p, Row.Row_var v when fuel > 0 -> (
        match Row.get_row_from_env env v with
        | Some tail -> Option.map (row_product (fuel - 1) tail) ~f:(fun q -> p * q)
        | None -> None)
    | Some _, Row.Row_var _ -> None
  in
  row_product 100 row

(** Updates delayed variable references with inferred dimensions/row products from the environment.
    If [solved_dim] was previously set by [set_dim], we skip updating to preserve the user's
    constraint - consistency will be checked later in [finish_inference] when all constraints have
    been fully processed. *)
let update_delayed_var_refs env update_step =
  let update_var_ref_list var_refs =
    List.iter var_refs ~f:(fun delayed_ref ->
        match delayed_ref.var with
        | `Not_set_yet -> () (* Variable not bound yet, will be set later *)
        | `Dim dim_var -> (
            match Row.get_dim_val env dim_var with
            | Some d ->
                (* If solved_dim was set by set_dim, preserve it - consistency checked in
                   finish_inference. Otherwise, update from inference. *)
                if Option.is_none delayed_ref.var_ref.solved_dim then
                  delayed_ref.var_ref.solved_dim <- Some d
            | None -> () (* Not yet resolved *))
        | `Row row_var -> (
            match Row.get_row_from_env env row_var with
            | Some row -> (
                match compute_row_product env row with
                | Some product ->
                    (* If solved_dim was set by set_dim, preserve it - the Total_elems constraint
                       will be processed in finish_inference to reconcile. Otherwise, update. *)
                    if Option.is_none delayed_ref.var_ref.solved_dim then
                      delayed_ref.var_ref.solved_dim <- Some product
                | None -> () (* Row has unresolved dimensions *))
            | None -> () (* Row variable not yet resolved *)))
  in
  match update_step.logic with
  | Transpose (Permute (_, var_refs), _) -> update_var_ref_list var_refs
  | Broadcast (Einsum (_, var_refs), _, _) -> update_var_ref_list var_refs
  | Broadcast_tern (Einsum_tern (_, var_refs), _, _, _) -> update_var_ref_list var_refs
  | Block { delayed_vars; _ } -> update_var_ref_list delayed_vars
  | _ -> ()

let apply_env_step env update_step =
  iter_shapes update_step ~f:(apply_env_t env);
  update_delayed_var_refs env update_step

let%debug4_sexp propagate_shapes (update_step : update_step) : unit =
  (* Allow the derivation of constraints to depend on the shapes (currently, only Batch_slice
     does). *)
  iter_shapes update_step ~f:(apply_env_t !state);
  let _, discardable_vars, ineqs = get_inequalities update_step in
  active_update_steps := update_step :: !active_update_steps;
  active_constraints := ineqs @ !active_constraints;
  let ineqs', env = Row.solve_inequalities ~stage:Row.Stage1 ~discardable_vars ineqs !state in
  let _debug_remaining_constraints : Row.constraint_ list = ineqs' in
  apply_env_step env update_step;
  state := env

let with_construction_window f =
  Int.incr construction_depth;
  Exn.protect ~f ~finally:(fun () -> Int.decr construction_depth)

let%debug4_sexp finish_inference (() : unit) : unit =
  if !construction_depth > 0 then
    raise
    @@ Row.Shape_error
         ( "finish_inference: shape inference was finalized from within [op_asn], mid-construction \
            of a tensor operation (via Shape.to_dims, Shape.to_padding or the projections handle) \
            -- not supported: the operation's own constraints are not registered yet, so its \
            operands would close without them. Fix the [op_asn] and reinitialize the session.",
           [] );
  let unsolved =
    List.filter !active_constraints ~f:(function
      | Shape_row (r, _) | Terminal_row (_, r, _) ->
          not (List.exists (Row.row_shapes r) ~f:(Hash_set.mem unused_shapes))
      | _ -> true)
  in
  (* TODO: optimize to keep all needed information in unsolved, rather than starting with all
     constraints. *)
  let unsolved, env = Row.solve_inequalities ~stage:Stage2 unsolved !state in
  let unsolved, env = Row.solve_inequalities ~stage:Stage3 unsolved env in
  let all_update_rows =
    List.concat_map ~f:all_rows_w_origin !active_update_steps
    |> List.map ~f:(fun (r, o) -> (Row.subst_row env r, o))
    |> List.dedup_and_sort ~compare:(fun (r1, _) (r2, _) -> Row.compare r1 r2)
  in
  let unsolved =
    List.map ~f:(fun (ro, o) -> Row.Shape_row (ro, [ o ])) all_update_rows @ unsolved
  in
  let unsolved, env = Row.solve_inequalities ~stage:Stage4 unsolved env in
  let unsolved, env = Row.solve_inequalities ~stage:Stage5 unsolved env in
  let unsolved, env = Row.solve_inequalities ~stage:Stage6 unsolved env in
  let unsolved, env = Row.solve_inequalities ~stage:Stage7 unsolved env in
  assert (List.is_empty unsolved);
  let _active_update_steps : update_step list = !active_update_steps in
  List.iter ~f:(apply_env_step env) !active_update_steps;
  let _applied_update_steps : update_step list = !active_update_steps in
  (* Derive projections for all active update_steps, setting their unsafe_projections field. This
     must happen before clearing active_update_steps. *)
  List.iter !active_update_steps ~f:(fun update_step ->
      if Option.is_none update_step.unsafe_projections then derive_projections update_step);
  active_constraints := [];
  active_update_steps := [];
  (* There should not be any shape variables remaining in any inference-undergoing update steps. *)
  state := Row.empty_env

let to_dims (sh : t) =
  finish_inference ();
  to_dims_impl sh

let%track4_sexp to_padding (sh : t) : (Ir.Ops.axis_padding array * float) option =
  finish_inference ();
  try
    (* If any row has padding, we need to return padding for all dimensions. Use zero padding for
       rows without explicit padding. *)
    let no_padding = Ir.Ops.{ left = 0; right = 0 } in
    let get_padding_array row_opt row =
      match row_opt with
      | Some padding -> padding
      | None -> Array.create ~len:(List.length (row.Row.beg_dims @ row.Row.dims)) no_padding
    in
    let has_any_padding : bool =
      Option.is_some sh.batch_padding || Option.is_some sh.output_padding
      || Option.is_some sh.input_padding
    in
    if not has_any_padding then None
    else
      let batch : Row.axis_padding array = get_padding_array sh.batch_padding sh.batch in
      let output : Row.axis_padding array = get_padding_array sh.output_padding sh.output in
      let input : Row.axis_padding array = get_padding_array sh.input_padding sh.input in
      (* Padded implies a known neutral element: padding only arises from margin-touching
         accumulation projections, and [update_padding_elem] commits their neutral (conflicts are
         rejected there). *)
      let padded_value =
        match sh.padding_elem with
        | Some (Some v) -> v
        | Some None | None ->
            raise
            @@ Row.Shape_error
                 ( "Internal invariant violation: shape has committed padding but no neutral \
                    element; padding should only arise from accumulation projections that carry \
                    one",
                   [ Shape_mismatch [ sh ] ] )
      in
      Some (Array.concat [ batch; output; input ], padded_value)
  with Row.Shape_error (s, trace) -> raise @@ Row.Shape_error (s, Shape_mismatch [ sh ] :: trace)

let to_bases (sh : t) : string array =
  Array.concat_map ~f:(Row.row_to_bases !state) [| sh.batch; sh.output; sh.input |]

(* Per-kind dimension bases, so callers can check which row/axis a basis landed on (not just that a
   tag appears somewhere in the flattened list). Returns [(batch, input, output)]. *)
let to_bases_bio (sh : t) : string array * string array * string array =
  ( Row.row_to_bases !state sh.batch,
    Row.row_to_bases !state sh.input,
    Row.row_to_bases !state sh.output )

let sexp_of_error_trace = function
  | Shape_mismatch ts -> Sexp.List (Sexp.Atom "Shape_mismatch" :: List.map ts ~f:sexp_of_t)
  | error_trace -> Row.sexp_of_error_trace error_trace

let () =
  Sexplib0.Sexp_conv.Exn_converter.add [%extension_constructor Row.Shape_error] (function
    | Row.Shape_error (arg0, arg1) ->
        let res0 = sexp_of_string arg0 and res1 = sexp_of_list sexp_of_error_trace arg1 in
        Sexplib0.Sexp.List [ Sexplib0.Sexp.Atom "lib/shape.ml.Shape_error"; res0; res1 ]
    | _ -> assert false)

(** {2 Shape builders.} *)

let get_projections (update_step : update_step) : Idx.projections =
  finish_inference ();
  match update_step.unsafe_projections with
  | Some projections -> projections
  | None ->
      (* This shouldn't happen if finish_inference is working correctly, but derive projections just
         in case *)
      derive_projections update_step;
      Option.value_exn update_step.unsafe_projections

let make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes
    ?(deduced = Not_constrained) ~debug_name ~id () =
  let open Row in
  let f _kind d =
    (* Provenance split at construction (brief §Technical-issue-1, AC#4): an unannotated user axis
       is a [default] atom at every size, INCLUDING size 1 — an explicit user [1] does not stretch
       (it flags a forgotten/hidden dimension rather than silently broadcasting). Only scalar
       helpers, internal broadcast fill (e.g. [embed_self_id]), and rank-broadening synthesizers
       mint the claim-free broadcast top [1_(bcast_if_1)], and they do so explicitly via [~*_axes] /
       [get_bcast_dim], not through this default path. *)
    get_default_dim ~d ()
  in
  let make_dims kind ds =
    {
      beg_dims = [];
      dims = List.map ~f:(f kind) ds;
      bcast = Broadcastable;
      prov = provenance ~sh_id:id ~kind;
    }
  in
  let make_axes kind ds =
    {
      beg_dims = [];
      dims = List.map ~f:(fun (basis, d) -> get_dim ~d ~basis ()) ds;
      bcast = Broadcastable;
      prov = provenance ~sh_id:id ~kind;
    }
  in
  let make_unknown kind =
    {
      dims = [];
      beg_dims = [];
      bcast = Row_var (get_row_var ());
      prov = provenance ~sh_id:id ~kind;
    }
  in
  let batch =
    match (batch_dims, batch_axes) with
    | Some batch_dims, None -> make_dims `Batch batch_dims
    | None, Some batch_axes -> make_axes `Batch batch_axes
    | None, None -> make_unknown `Batch
    | Some _, Some _ -> invalid_arg "Shape.make: do not provide both batch_dims, batch_axes"
  in
  let input =
    match (input_dims, input_axes) with
    | Some input_dims, None -> make_dims `Input input_dims
    | None, Some input_axes -> make_axes `Input input_axes
    | None, None -> make_unknown `Input
    | Some _, Some _ -> invalid_arg "Shape.make: do not provide both input_dims, input_axes"
  in
  let output =
    match (output_dims, output_axes) with
    | Some output_dims, None -> make_dims `Output output_dims
    | None, Some output_axes -> make_axes `Output output_axes
    | None, None -> make_unknown `Output
    | Some _, Some _ -> invalid_arg "Shape.make: do not provide both output_dims, output_axes"
  in
  let result =
    {
      input;
      output;
      batch;
      id;
      debug_name;
      batch_padding = None;
      input_padding = None;
      output_padding = None;
      padding_elem = None;
    }
  in
  (match deduced with
  | Not_constrained -> ()
  | Input_equals_output -> (
      try
        let origin =
          [
            Row.
              {
                lhs_name = result.debug_name;
                lhs_kind = `Input;
                rhs_name = result.debug_name;
                rhs_kind = `Output;
                operation = Some "input_equals_output";
              };
          ]
        in
        let more_ineqs, env = Row.unify_row ~stage:Stage2 origin (input, output) !state in
        assert (List.is_empty more_ineqs);
        state := env
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ("Input_equals_output / " ^ s, Shape_mismatch [ result ] :: trace)));
  result

let shape_spec_to_dims_bio ~spec ~sh_id labels =
  let dim_var_env = Hashtbl.create (module String) in
  let f _kind = function
    | Label s when String.contains s '=' -> (
        let _label, dim =
          match String.split s ~on:'=' with
          | [ l; d ] -> (l, d)
          | _ -> invalid_arg "shape_spec_to_dims_bio: too many '='"
        in
        (* This is not a dimension basis i.e. unit! *)
        try Row.get_default_dim ~d:(Int.of_string dim) ()
        with _ -> invalid_arg "shape_spec_to_dims_bio: int expected after '='")
    | Label name ->
        Var (Hashtbl.find_or_add dim_var_env name ~default:(fun () -> Row.get_var ~name ()))
    | Fixed_index d -> Row.get_default_dim ~d ()
    | Affine_spec { stride; over_label; conv; stride_offset } ->
        let stride_int =
          try Int.of_string stride
          with _ -> failwith ("Invalid stride value (expected integer): " ^ stride)
        in
        let over_dim =
          Row.Var
            (Hashtbl.find_or_add dim_var_env over_label ~default:(fun () ->
                 Row.get_var ~name:over_label ()))
        in
        let conv =
          Option.map conv ~f:(fun { dilation; kernel_label; use_padding } ->
              let dilation_int =
                try Int.of_string dilation
                with _ -> failwith ("Invalid dilation value (expected integer): " ^ dilation)
              in
              let use_padding_bool =
                match use_padding with
                | `True -> true
                | `False -> false
                | `Unspecified ->
                    failwith
                      "use_padding must be specified in convolution spec (use = for true, < for \
                       false)"
              in
              let kernel =
                Row.Var
                  (Hashtbl.find_or_add dim_var_env kernel_label ~default:(fun () ->
                       Row.get_var ~name:kernel_label ()))
              in
              { Row.dilation = dilation_int; kernel; use_padding = use_padding_bool })
        in
        Row.Affine { stride = stride_int; over = over_dim; conv; stride_offset }
    | Concat_spec labels ->
        (* Convert each label in the concatenation to a dimension and wrap in Concat *)
        let dims =
          List.map labels ~f:(fun label ->
              Row.Var
                (Hashtbl.find_or_add dim_var_env label ~default:(fun () ->
                     Row.get_var ~name:label ())))
        in
        Row.Concat dims
  in
  let row_var_env = Hashtbl.create (module String) in
  let result = axes_spec_to_dims_bio ~sh_id ~row_var_env ~dim_var_env ~f labels in
  check_dim_row_var_name_clash ~spec ~error_trace:[] dim_var_env row_var_env;
  result

let of_spec ?(deduced = Not_constrained) ~debug_name ~id spec =
  let batch, input, output = shape_spec_to_dims_bio ~spec ~sh_id:id @@ axis_labels_of_spec spec in
  let result =
    {
      input;
      output;
      batch;
      id;
      debug_name;
      batch_padding = None;
      input_padding = None;
      output_padding = None;
      padding_elem = None;
    }
  in
  (match deduced with
  | Not_constrained -> ()
  | Input_equals_output -> (
      try
        let origin =
          [
            Row.
              {
                lhs_name = result.debug_name;
                lhs_kind = `Input;
                rhs_name = result.debug_name;
                rhs_kind = `Output;
                operation = Some "input_equals_output";
              };
          ]
        in
        let more_ineqs, env = Row.unify_row ~stage:Stage2 origin (input, output) !state in
        assert (List.is_empty more_ineqs);
        state := env
      with Row.Shape_error (s, trace) when !with_error_trace ->
        raise @@ Row.Shape_error ("of spec / " ^ s, Shape_mismatch [ result ] :: trace)));
  result

let to_string_hum ?(style = Row.Axis_size) (sh : t) =
  let n_outputs = List.length (sh.output.beg_dims @ sh.output.dims) in
  let n_batch = List.length (sh.batch.beg_dims @ sh.batch.dims) in
  let dims_to_string kind =
    let row = row_of_kind kind sh in
    let dims = row.beg_dims @ row.dims in
    String.concat ~sep:","
    @@ List.mapi dims ~f:(fun i d ->
        let num =
          match kind with `Input -> n_batch + n_outputs + i | `Output -> n_batch + i | `Batch -> i
        in
        match style with
        | Row.Only_bases | Axis_size | Projection_and_size -> Row.dim_to_string style d
        | Axis_number_and_size -> Int.to_string num ^ ":" ^ Row.dim_to_string style d)
  in
  let batch_dims = dims_to_string `Batch in
  let input_dims = dims_to_string `Input in
  let output_dims = dims_to_string `Output in
  if String.is_empty batch_dims && String.is_empty input_dims then output_dims
  else if String.is_empty batch_dims then input_dims ^ "->" ^ output_dims
  else if String.is_empty input_dims then batch_dims ^ "|" ^ output_dims
  else batch_dims ^ "|" ^ input_dims ^ "->" ^ output_dims

(** Given a fully-inferred shape, maps axes to their corresponding positions in an index using the
    [force_to_dims] semantics. *)
let axis_keys_to_idcs (sh : t) : int axis_map =
  if
    not
      (Row.is_broadcastable sh.input.bcast
      && Row.is_broadcastable sh.output.bcast
      && Row.is_broadcastable sh.batch.bcast)
  then raise @@ Utils.User_error "Shape.axis_keys_to_idcs: shape not fully inferred";
  let b_dims =
    (* Enumerate axes backwards across both flanks. *)
    Array.of_list_rev_mapi (sh.batch.beg_dims @ sh.batch.dims) ~f:(fun i _ ->
        AxisKey.{ in_axes = `Batch; pos = i + 1; from_end = true })
  in
  let i_dims =
    Array.of_list_rev_mapi (sh.input.beg_dims @ sh.input.dims) ~f:(fun i _ ->
        AxisKey.{ in_axes = `Input; pos = i + 1; from_end = true })
  in
  let o_dims =
    Array.of_list_rev_mapi (sh.output.beg_dims @ sh.output.dims) ~f:(fun i _ ->
        AxisKey.{ in_axes = `Output; pos = i + 1; from_end = true })
  in
  let idcs = Array.concat [ i_dims; o_dims; b_dims ] in
  Array.rev_inplace idcs;
  Map.of_alist_exn (module AxisKey) @@ Array.to_list @@ Array.mapi idcs ~f:(fun i key -> (key, i))

let%debug5_sexp default_display_indices (sh : t) : int array =
  let axes = axis_keys_to_idcs sh |> Map.map ~f:(fun _ -> 0) in
  let occupied = Array.create ~len:5 false in
  let set_occu prio =
    occupied.(prio + 5) <- true;
    prio
  in
  let occu prio = occupied.(prio + 5) in
  let num_input_axes = List.length (sh.input.beg_dims @ sh.input.dims) in
  let remaining =
    Stack.of_list
    @@ List.filter ~f:(Map.mem axes)
    @@ AxisKey.
         [
           { in_axes = `Input; from_end = true; pos = 1 };
           { in_axes = `Output; from_end = true; pos = 1 };
           { in_axes = `Input; from_end = true; pos = 2 };
           { in_axes = `Output; from_end = true; pos = 2 };
           (if num_input_axes > 1 then { in_axes = `Batch; from_end = true; pos = 1 }
            else { in_axes = `Output; from_end = true; pos = 3 });
           { in_axes = `Batch; from_end = true; pos = 1 };
           { in_axes = `Batch; from_end = true; pos = 2 };
           { in_axes = `Input; from_end = true; pos = 3 };
           { in_axes = `Output; from_end = true; pos = 3 };
           { in_axes = `Input; from_end = true; pos = 4 };
           { in_axes = `Output; from_end = true; pos = 4 };
           { in_axes = `Input; from_end = true; pos = 5 };
           { in_axes = `Output; from_end = true; pos = 5 };
         ]
  in
  let rec loop offset axes =
    if Stack.is_empty remaining || offset > 5 then axes
    else if Fn.non occu ~-offset then
      loop (offset + 1)
      @@ Map.change axes (Stack.pop_exn remaining) ~f:(Option.map ~f:(fun _ -> set_occu ~-offset))
    else loop (offset + 1) axes
  in
  let axes = loop 1 axes in
  axis_map_to_dims_index axes

let parse_n5_layout priorities =
  let f : Einsum_parser.axis_spec -> int = function
    | Fixed_index i -> i
    | Label _ -> invalid_arg "parse_n5_layout requires integer-only labels"
    | Affine_spec _ -> invalid_arg "parse_n5_layout does not support affine expressions"
    | Concat_spec _ -> invalid_arg "parse_n5_layout does not support concatenation expressions"
  in
  let p_labels = Einsum_parser.(axis_labels @@ axis_labels_of_spec priorities) in
  axis_map_to_dims_index p_labels |> Array.map ~f

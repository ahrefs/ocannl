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
  { var_ref = Ir.Indexing.{ ref_label; solved_dim = None }; var = `Not_set_yet }

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

type ternary_type = Pointwise_tern | Compose_accumulate | Defined_by_cd_logic
[@@deriving sexp, equal]

let einsum_of_spec spec =
  try Einsum_parser.einsum_of_spec spec
  with Einsum_parser.Parse_error msg ->
    raise @@ Utils.User_error ("Shape.einsum_of_spec: while parsing: " ^ spec ^ " error: " ^ msg)

type logic =
  | Broadcast of compose_type * t * t
  | Transpose of transpose_type * t
  | Broadcast_tern of ternary_type * t * t * t
  | Terminal of { is_param : bool; logic : terminal_type }
[@@deriving equal, sexp_of]

let logic_to_spec = function
  | Broadcast (Pointwise_bin, _, _)
  | Transpose (Pointwise_un, _)
  | Broadcast_tern (Pointwise_tern, _, _, _) ->
      "."
  | Broadcast (Compose, _, _) | Broadcast_tern (Compose_accumulate, _, _, _) -> "@"
  | Broadcast (Einsum (spec, _), _, _) | Transpose (Permute (spec, _), _) -> spec
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

type update_step = { shape : t; logic : logic; id : update_id } [@@deriving sexp_of]
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
  let to_bcast kind v beg_dims =
    let beg_dims = to_dim kind beg_dims in
    Option.value_map v ~default:(Row.Broadcastable, beg_dims) ~f:(fun vname ->
        let v = Hashtbl.find_or_add row_var_env vname ~default:(fun () -> Row.get_row_var ()) in
        Row.add_used_in_spec_or_compose v;
        (Row.Row_var { v; beg_dims }, []))
  in
  let to_row kind v dims beg_dims =
    let bcast, beg_dims = to_bcast kind v beg_dims in
    { Row.dims = beg_dims @ to_dim kind dims; bcast; prov = Row.provenance ~sh_id ~kind }
  in
  let batch = to_row `Batch labels.bcast_batch b_dims beg_b_dims in
  let input = to_row `Input labels.bcast_input i_dims beg_i_dims in
  let output = to_row `Output labels.bcast_output o_dims beg_o_dims in
  (batch, input, output)

let einsum_slot_spec_to_dims_bio ~original_spec ~sh_id ~row_var_env ~dim_var_env labels =
  let proj_env_update = ref @@ Row.dim_map_empty in
  let extras = ref [] in
  let f kind = function
    | Label name ->
        Row.Var (Hashtbl.find_or_add dim_var_env name ~default:(fun () -> Row.get_var ~name ()))
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
  in
  let result = axes_spec_to_dims_bio ~sh_id ~row_var_env ~dim_var_env ~f labels in
  (!extras, !proj_env_update, result)

type proj_axis_env = Idx.axis_index Row.dim_map [@@deriving sexp]

let add_var_used_in_spec_or_compose row =
  match row with Row.Row_var { v; _ } -> Row.add_used_in_spec_or_compose v | _ -> ()

let add_var_used_in_pointwise row =
  match row with Row.Row_var { v; _ } -> Row.add_used_in_pointwise v | _ -> ()

let unused_shapes = Hash_set.create (module Int)

let%debug4_sexp get_inequalities ?(for_projections = false)
    ({ shape = cur_sh; logic; id = _ } as _upd : update_step) : proj_axis_env * Row.constraint_ list
    =
  Hash_set.remove unused_shapes cur_sh.id;
  let _debug_cur_sh : t = cur_sh in
  let _debug_logic : logic = logic in
  let open Row in
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
  | Terminal { is_param; logic = Fetch Range_over_offsets } ->
      (Row.dim_map_empty, mark_terminal ~is_param)
  | Terminal { is_param; logic = Fetch (Constant _) } -> (Row.dim_map_empty, mark_terminal ~is_param)
  | Terminal { is_param; logic = Fetch (Constant_bits _) } ->
      (Row.dim_map_empty, mark_terminal ~is_param)
  | Terminal { is_param; logic = Data (Reshape nd) } ->
      ( dim_map_empty,
        Rows_constr
          {
            r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
            constr =
              Total_elems
                {
                  numerator = Num_elems (Array.fold (Ir.Ndarray.dims nd) ~init:1 ~f:( * ));
                  divided_by = [];
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
        (if for_projections then []
         else
           [
             Rows_constr
               {
                 r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
                 constr =
                   Exact
                     (Ir.Ndarray.dims nd |> Array.map ~f:(fun d -> get_dim ~d ()) |> Array.to_list);
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
      (* FIXME: constrain padding. *)
      ignore (padding, padded_value);
      ( dim_map_empty,
        (if for_projections then []
         else
           [
             Rows_constr
               {
                 r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
                 constr =
                   Exact
                     (Ir.Ndarray.dims data |> Array.map ~f:(fun d -> get_dim ~d ()) |> Array.to_list);
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
        Rows_constr
          {
            r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
            constr = Total_elems { numerator = Num_elems len; divided_by = [] };
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
          (if for_projections then []
           else
             [
               Rows_constr
                 {
                   r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
                   constr =
                     Exact
                       (Lazy.force tn.dims |> Array.to_list |> List.tl_exn
                       |> List.map ~f:(fun d -> get_dim ~d ()));
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
      else (Row.dim_map_empty, mark_terminal ~is_param)
  | Terminal { is_param; logic = Fetch (Embed_symbol _) } ->
      (Row.dim_map_empty, mark_terminal ~is_param)
  | Terminal { is_param; logic = Fetch (Embed_dim _) } ->
      (Row.dim_map_empty, mark_terminal ~is_param)
  | Terminal { is_param; logic = Fetch Embed_self_id } ->
      (Row.dim_map_empty, mark_terminal ~is_param)
  | Transpose (Transpose, sh) ->
      Hash_set.remove unused_shapes sh.id;
      ( Row.dim_map_empty,
        [
          Row_ineq
            {
              cur = cur_sh.batch;
              subr = sh.batch;
              origin = [ get_origin `Batch sh `Batch "transpose" ];
            };
          Row_ineq
            {
              cur = cur_sh.input;
              subr = sh.output;
              origin = [ get_origin `Input sh `Output "transpose" ];
            };
          Row_ineq
            {
              cur = cur_sh.output;
              subr = sh.input;
              origin = [ get_origin `Output sh `Input "transpose" ];
            };
        ] )
  | Transpose (Pointwise_un, sh) ->
      Hash_set.remove unused_shapes sh.id;
      add_var_used_in_pointwise cur_sh.input.bcast;
      add_var_used_in_pointwise sh.input.bcast;
      ( Row.dim_map_empty,
        [
          Row_ineq
            {
              cur = cur_sh.batch;
              subr = sh.batch;
              origin = [ get_origin `Batch sh `Batch "pointwise_unary" ];
            };
          Row_ineq
            {
              cur = cur_sh.input;
              subr = sh.input;
              origin = [ get_origin `Input sh `Input "pointwise_unary" ];
            };
          Row_ineq
            {
              cur = cur_sh.output;
              subr = sh.output;
              origin = [ get_origin `Output sh `Output "pointwise_unary" ];
            };
        ] )
  | Broadcast (Compose, sh1, sh2) ->
      Hash_set.remove unused_shapes sh1.id;
      Hash_set.remove unused_shapes sh2.id;
      add_var_used_in_spec_or_compose sh1.input.bcast;
      ( Row.dim_map_empty,
        [
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
              cur = sh1.input;
              subr = sh2.output;
            };
          Row_ineq
            {
              origin = [ get_origin `Batch sh1 `Batch "compose" ];
              cur = cur_sh.batch;
              subr = sh1.batch;
            };
          Row_ineq
            {
              origin = [ get_origin `Batch sh2 `Batch "compose" ];
              cur = cur_sh.batch;
              subr = sh2.batch;
            };
          Row_ineq
            {
              origin = [ get_origin `Input sh2 `Input "compose" ];
              cur = cur_sh.input;
              subr = sh2.input;
            };
          Row_ineq
            {
              origin = [ get_origin `Output sh1 `Output "compose" ];
              cur = cur_sh.output;
              subr = sh1.output;
            };
        ] )
  | Broadcast (Pointwise_bin, sh1, sh2) ->
      Hash_set.remove unused_shapes sh1.id;
      Hash_set.remove unused_shapes sh2.id;
      add_var_used_in_pointwise cur_sh.input.bcast;
      add_var_used_in_pointwise sh1.input.bcast;
      add_var_used_in_pointwise sh2.input.bcast;
      ( Row.dim_map_empty,
        [
          Row_ineq
            {
              origin = [ get_origin `Batch sh1 `Batch "pointwise_binary" ];
              cur = cur_sh.batch;
              subr = sh1.batch;
            };
          Row_ineq
            {
              origin = [ get_origin `Batch sh2 `Batch "pointwise_binary" ];
              cur = cur_sh.batch;
              subr = sh2.batch;
            };
          Row_ineq
            {
              origin = [ get_origin `Input sh1 `Input "pointwise_binary" ];
              cur = cur_sh.input;
              subr = sh1.input;
            };
          Row_ineq
            {
              origin = [ get_origin `Input sh2 `Input "pointwise_binary" ];
              cur = cur_sh.input;
              subr = sh2.input;
            };
          Row_ineq
            {
              origin = [ get_origin `Output sh1 `Output "pointwise_binary" ];
              cur = cur_sh.output;
              subr = sh1.output;
            };
          Row_ineq
            {
              origin = [ get_origin `Output sh2 `Output "pointwise_binary" ];
              cur = cur_sh.output;
              subr = sh2.output;
            };
        ] )
  | Broadcast_tern (Compose_accumulate, sh1, sh2, sh3) ->
      Hash_set.remove unused_shapes sh1.id;
      Hash_set.remove unused_shapes sh2.id;
      Hash_set.remove unused_shapes sh3.id;
      add_var_used_in_spec_or_compose sh1.input.bcast;
      ( Row.dim_map_empty,
        [
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
              cur = sh1.input;
              subr = sh2.output;
            };
          Row_ineq
            {
              origin = [ get_origin `Batch sh1 `Batch "compose_accumulate" ];
              cur = cur_sh.batch;
              subr = sh1.batch;
            };
          Row_ineq
            {
              origin = [ get_origin `Batch sh2 `Batch "compose_accumulate" ];
              cur = cur_sh.batch;
              subr = sh2.batch;
            };
          Row_ineq
            {
              origin = [ get_origin `Input sh2 `Input "compose_accumulate" ];
              cur = cur_sh.input;
              subr = sh2.input;
            };
          Row_ineq
            {
              origin = [ get_origin `Output sh1 `Output "compose_accumulate" ];
              cur = cur_sh.output;
              subr = sh1.output;
            };
          Row_ineq
            {
              origin = [ get_origin `Batch sh3 `Batch "compose_accumulate" ];
              cur = cur_sh.batch;
              subr = sh3.batch;
            };
          Row_ineq
            {
              origin = [ get_origin `Input sh3 `Input "compose_accumulate" ];
              cur = cur_sh.input;
              subr = sh3.input;
            };
          Row_ineq
            {
              origin = [ get_origin `Output sh3 `Output "compose_accumulate" ];
              cur = cur_sh.output;
              subr = sh3.output;
            };
        ] )
  | Broadcast_tern (Pointwise_tern, sh1, sh2, sh3) ->
      Hash_set.remove unused_shapes sh1.id;
      Hash_set.remove unused_shapes sh2.id;
      Hash_set.remove unused_shapes sh3.id;
      add_var_used_in_pointwise cur_sh.input.bcast;
      add_var_used_in_pointwise sh1.input.bcast;
      add_var_used_in_pointwise sh2.input.bcast;
      add_var_used_in_pointwise sh3.input.bcast;
      ( Row.dim_map_empty,
        [
          Row_ineq
            {
              origin = [ get_origin `Batch sh1 `Batch "pointwise_ternary" ];
              cur = cur_sh.batch;
              subr = sh1.batch;
            };
          Row_ineq
            {
              origin = [ get_origin `Batch sh2 `Batch "pointwise_ternary" ];
              cur = cur_sh.batch;
              subr = sh2.batch;
            };
          Row_ineq
            {
              origin = [ get_origin `Batch sh3 `Batch "pointwise_ternary" ];
              cur = cur_sh.batch;
              subr = sh3.batch;
            };
          Row_ineq
            {
              origin = [ get_origin `Input sh1 `Input "pointwise_ternary" ];
              cur = cur_sh.input;
              subr = sh1.input;
            };
          Row_ineq
            {
              origin = [ get_origin `Input sh2 `Input "pointwise_ternary" ];
              cur = cur_sh.input;
              subr = sh2.input;
            };
          Row_ineq
            {
              origin = [ get_origin `Input sh3 `Input "pointwise_ternary" ];
              cur = cur_sh.input;
              subr = sh3.input;
            };
          Row_ineq
            {
              origin = [ get_origin `Output sh1 `Output "pointwise_ternary" ];
              cur = cur_sh.output;
              subr = sh1.output;
            };
          Row_ineq
            {
              origin = [ get_origin `Output sh2 `Output "pointwise_ternary" ];
              cur = cur_sh.output;
              subr = sh2.output;
            };
          Row_ineq
            {
              origin = [ get_origin `Output sh3 `Output "pointwise_ternary" ];
              cur = cur_sh.output;
              subr = sh3.output;
            };
        ] )
  | Broadcast (Defined_by_cd_logic, _, _)
  | Transpose (Defined_by_cd_logic, _)
  | Broadcast_tern (Defined_by_cd_logic, _, _, _) ->
      (Row.dim_map_empty, [])
  | Transpose (Batch_slice { static_range; static_symbol }, sh) ->
      Hash_set.remove unused_shapes sh.id;
      let slice_v = get_var () in
      let slice_var = Var slice_v in
      let proj_axis_env =
        Map.add_exn Row.dim_map_empty ~key:slice_v ~data:(Idx.Iterator static_symbol)
      in
      (* Expand a batch row instead of reducing one because even if the dimensions are known, the
         equations are also needed for projection inference. *)
      let expanded_batch =
        match cur_sh.batch.bcast with
        | Broadcastable ->
            {
              dims = slice_var :: cur_sh.batch.dims;
              bcast = cur_sh.batch.bcast;
              prov = Row.provenance ~sh_id:cur_sh.id ~kind:`Batch;
            }
        | Row_var { v; beg_dims } ->
            {
              dims = cur_sh.batch.dims;
              bcast = Row_var { v; beg_dims = slice_var :: beg_dims };
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
        (Option.to_list (if for_projections then None else static_range)
        |> List.map ~f:(fun range ->
            Dim_eq
              {
                d1 = get_dim ~d:range ();
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
        | ls_rhs, None, ls_lhs -> (ls_rhs, ls_lhs)
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
      (* Bind delayed_var_refs to the variables after they are created *)
      let extras_dim_refs =
        List.filter_map dim_refs ~f:(fun delayed_ref ->
            let label = delayed_ref.var_ref.ref_label in
            (* Check if it's in one of the environments *)
            match Hashtbl.find dim_var_env label with
            | Some var ->
                delayed_ref.var <- `Dim var;
                Option.map
                  ~f:(fun solved_dim ->
                    Dim_eq
                      {
                        d1 = Row.Var var;
                        d2 = Row.get_dim ~d:solved_dim ();
                        origin =
                          [
                            {
                              lhs_name = label;
                              lhs_kind = `Output;
                              rhs_name = delayed_ref.var_ref.ref_label;
                              rhs_kind = `Output;
                              operation = Some "set_dim";
                            };
                          ];
                      })
                  (if for_projections then None else delayed_ref.var_ref.solved_dim)
            | None -> (
                match Hashtbl.find row_var_env label with
                | Some var ->
                    delayed_ref.var <- `Row var;
                    Option.map
                      ~f:(fun solved_dim ->
                        Rows_constr
                          {
                            r = [ Row.get_row_for_var Row.empty_provenance var ];
                            constr =
                              Total_elems { numerator = Num_elems solved_dim; divided_by = [] };
                            origin =
                              [
                                {
                                  lhs_name = label;
                                  lhs_kind = `Output;
                                  rhs_name = delayed_ref.var_ref.ref_label;
                                  rhs_kind = `Output;
                                  operation = Some "set_dim";
                                };
                              ];
                          })
                      delayed_ref.var_ref.solved_dim
                | None ->
                    raise
                    @@ Row.Shape_error
                         ( "Variable " ^ label ^ " not found in environments for spec: " ^ spec,
                           [ Shape_mismatch [ cur_sh; sh ] ] )))
      in
      let proj_env =
        let combine ~key:_ _ _ = assert false in
        Map.merge_skewed ~combine proj_env_rhs proj_env_lhs
      in
      ( proj_env,
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
      ( Row.dim_map_empty,
        [
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
                  { numerator = Row.Strided_var { coeff; var; denom = 1 }; divided_by = [] };
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
        ] )
  | Broadcast (Einsum (spec, dim_refs), sh1, sh2) ->
      Hash_set.remove unused_shapes sh1.id;
      Hash_set.remove unused_shapes sh2.id;
      add_var_used_in_spec_or_compose cur_sh.input.bcast;
      add_var_used_in_spec_or_compose sh1.input.bcast;
      add_var_used_in_spec_or_compose sh2.input.bcast;
      let ls_rhs1, ls_rhs2, ls_lhs =
        match einsum_of_spec spec with
        | ls_rhs1, Some ls_rhs2, ls_lhs -> (ls_rhs1, ls_rhs2, ls_lhs)
        | _, None, _ ->
            raise
            @@ Shape_error
                 ( "Invalid permutation spec (expected one argument): " ^ spec,
                   [ Shape_mismatch [ cur_sh; sh1; sh2 ] ] )
      in
      let row_var_env = Hashtbl.create (module String) in
      let dim_var_env = Hashtbl.create (module String) in
      let extras_rhs1, proj_env_rhs1, (b_rhs1, i_rhs1, o_rhs1) =
        einsum_slot_spec_to_dims_bio ~original_spec:spec ~sh_id:sh1.id ~row_var_env ~dim_var_env
          ls_rhs1
      in
      let extras_rhs2, proj_env_rhs2, (b_rhs2, i_rhs2, o_rhs2) =
        einsum_slot_spec_to_dims_bio ~original_spec:spec ~sh_id:sh2.id ~row_var_env ~dim_var_env
          ls_rhs2
      in
      let extras_lhs, proj_env_lhs, (b_lhs, i_lhs, o_lhs) =
        einsum_slot_spec_to_dims_bio ~original_spec:spec ~sh_id:cur_sh.id ~row_var_env ~dim_var_env
          ls_lhs
      in
      (* Bind delayed_var_refs to the variables after they are created *)
      (* TODO: refactor to avoid duplication with the one for unary einsum *)
      let extras_dim_refs =
        List.filter_map dim_refs ~f:(fun delayed_ref ->
            let label = delayed_ref.var_ref.ref_label in
            (* Check if it's in one of the environments *)
            match Hashtbl.find dim_var_env label with
            | Some var ->
                delayed_ref.var <- `Dim var;
                Option.map
                  ~f:(fun solved_dim ->
                    Dim_eq
                      {
                        d1 = Row.Var var;
                        d2 = Row.get_dim ~d:solved_dim ();
                        origin =
                          [
                            {
                              lhs_name = label;
                              lhs_kind = `Output;
                              rhs_name = delayed_ref.var_ref.ref_label;
                              rhs_kind = `Output;
                              operation = Some "set_dim";
                            };
                          ];
                      })
                  (if for_projections then None else delayed_ref.var_ref.solved_dim)
            | None -> (
                match Hashtbl.find row_var_env label with
                | Some var ->
                    delayed_ref.var <- `Row var;
                    Option.map
                      ~f:(fun solved_dim ->
                        Rows_constr
                          {
                            r = [ Row.get_row_for_var Row.empty_provenance var ];
                            constr =
                              Total_elems { numerator = Num_elems solved_dim; divided_by = [] };
                            origin =
                              [
                                {
                                  lhs_name = label;
                                  lhs_kind = `Output;
                                  rhs_name = delayed_ref.var_ref.ref_label;
                                  rhs_kind = `Output;
                                  operation = Some "set_dim";
                                };
                              ];
                          })
                      delayed_ref.var_ref.solved_dim
                | None ->
                    raise
                    @@ Row.Shape_error
                         ( "Variable " ^ label ^ " not found in environments for spec: " ^ spec,
                           [ Shape_mismatch [ cur_sh; sh1; sh2 ] ] )))
      in
      let proj_env =
        let combine ~key:_ _ _ = assert false in
        Map.merge_skewed ~combine proj_env_rhs1
        @@ Map.merge_skewed ~combine proj_env_rhs2 proj_env_lhs
      in
      (* Forget the old proj_env as it is not relevant after a propagate_shapes call completes. *)
      ( proj_env,
        extras_dim_refs @ extras_rhs1 @ extras_rhs2 @ extras_lhs
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
                      operation = Some "Broadcast RESULT";
                    };
                  ];
              };
            Row_eq
              {
                r1 = b_rhs1;
                r2 = sh1.batch;
                origin =
                  [
                    {
                      lhs_name = spec;
                      lhs_kind = `Batch;
                      rhs_name = sh1.debug_name;
                      rhs_kind = `Batch;
                      operation = Some "Broadcast ARGUMENT 1";
                    };
                  ];
              };
            Row_eq
              {
                r1 = b_rhs2;
                r2 = sh2.batch;
                origin =
                  [
                    {
                      lhs_name = spec;
                      lhs_kind = `Batch;
                      rhs_name = sh2.debug_name;
                      rhs_kind = `Batch;
                      operation = Some "Broadcast ARGUMENT 2";
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
                      operation = Some "Broadcast RESULT";
                    };
                  ];
              };
            Row_eq
              {
                r1 = i_rhs1;
                r2 = sh1.input;
                origin =
                  [
                    {
                      lhs_name = spec;
                      lhs_kind = `Input;
                      rhs_name = sh1.debug_name;
                      rhs_kind = `Input;
                      operation = Some "Broadcast ARGUMENT 1";
                    };
                  ];
              };
            Row_eq
              {
                r1 = i_rhs2;
                r2 = sh2.input;
                origin =
                  [
                    {
                      lhs_name = spec;
                      lhs_kind = `Input;
                      rhs_name = sh2.debug_name;
                      rhs_kind = `Input;
                      operation = Some "Broadcast ARGUMENT 2";
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
                      operation = Some "Broadcast RESULT";
                    };
                  ];
              };
            Row_eq
              {
                r1 = o_rhs1;
                r2 = sh1.output;
                origin =
                  [
                    {
                      lhs_name = spec;
                      lhs_kind = `Output;
                      rhs_name = sh1.debug_name;
                      rhs_kind = `Output;
                      operation = Some "Broadcast ARGUMENT 1";
                    };
                  ];
              };
            Row_eq
              {
                r1 = o_rhs2;
                r2 = sh2.output;
                origin =
                  [
                    {
                      lhs_name = spec;
                      lhs_kind = `Output;
                      rhs_name = sh2.debug_name;
                      rhs_kind = `Output;
                      operation = Some "Broadcast ARGUMENT 2";
                    };
                  ];
              };
          ] )

let state = ref Row.empty_env
let active_update_steps = ref []
let active_constraints = ref []

let%track7_sexp set_dim (delayed_var_ref : delayed_var_ref) (dim : int) : unit =
  match delayed_var_ref with
  | { var_ref = { solved_dim = Some dim2; _ }; _ } when dim2 = dim -> ()
  | { var_ref = { solved_dim = Some dim2; ref_label; _ }; _ } ->
      raise
      @@ Row.Shape_error
           ( "Cannot set dimension for variable reference with label " ^ ref_label,
             [ Row.Dim_mismatch [ Row.get_dim ~d:dim2 (); Row.get_dim ~d:dim () ] ] )
  | { var_ref = { solved_dim = None; _ }; var = `Not_set_yet } ->
      delayed_var_ref.var_ref.solved_dim <- Some dim
  | { var_ref = { solved_dim = None; _ }; var = `Dim dim_var } ->
      delayed_var_ref.var_ref.solved_dim <- Some dim;
      active_constraints :=
        Row.Dim_eq
          {
            d1 = Row.Var dim_var;
            d2 = Row.get_dim ~d:dim ();
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
  | { var_ref = { solved_dim = None; _ }; var = `Row row_var } ->
      delayed_var_ref.var_ref.solved_dim <- Some dim;
      active_constraints :=
        Row.Rows_constr
          {
            (* TODO: actually, the Row.provenance should be the one of the shape that the row
               variable is in, should be stored in `Row and in env_row_var. *)
            r = [ Row.get_row_for_var Row.empty_provenance row_var ];
            constr = Total_elems { numerator = Num_elems dim; divided_by = [] };
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

let set_equal delayed_ref1 delayed_ref2 =
  (* TODO: use provenance from the row variables once we have it there. *)
  match (delayed_ref1, delayed_ref2) with
  | { var_ref = { solved_dim = Some dim1; _ }; _ }, { var_ref = { solved_dim = Some dim2; _ }; _ }
    ->
      if dim1 = dim2 then ()
      else
        raise
        @@ Row.Shape_error
             ( "Cannot set equal dimensions for variable references with different values",
               [ Row.Dim_mismatch [ Row.get_dim ~d:dim1 (); Row.get_dim ~d:dim2 () ] ] )
  | { var_ref = { solved_dim = Some dim; _ }; _ }, delayed_ref2 ->
      (* First is solved, second is not - set the second to match the first *)
      set_dim delayed_ref2 dim
  | delayed_ref1, { var_ref = { solved_dim = Some dim; _ }; _ } ->
      (* Second is solved, first is not - set the first to match the second *)
      set_dim delayed_ref1 dim
  | ( { var_ref = { solved_dim = None; ref_label = ref_label1; _ }; var = _ },
      { var_ref = { solved_dim = None; ref_label = ref_label2; _ }; var = `Not_set_yet } )
  | ( { var_ref = { solved_dim = None; ref_label = ref_label1; _ }; var = `Not_set_yet },
      { var_ref = { solved_dim = None; ref_label = ref_label2; _ }; var = _ } ) ->
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
                      };
                  divided_by = [];
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

let unsafe_reinitialize () =
  update_uid := 0;
  state := Row.empty_env;
  active_update_steps := [];
  active_constraints := []

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

let apply_env_t env sh =
  sh.batch <- Row.subst_row env sh.batch;
  sh.input <- Row.subst_row env sh.input;
  sh.output <- Row.subst_row env sh.output

let rec compute_row_product env (row : Row.t) : int =
  match row.dims with
  | [] -> 1
  | dim :: rest ->
      let dim_val =
        match dim with
        | Row.Dim { d; _ } -> d
        | Row.Var v -> (
            match Row.get_dim_val env v with
            | Some d -> d
            | None -> 1 (* Variable not yet resolved *))
        | Row.Affine _ -> 1 (* TODO: handle affine/convolution input dimensions *)
      in
      dim_val * compute_row_product env { row with dims = rest }

let update_delayed_var_refs env update_step =
  let update_var_ref_list var_refs =
    List.iter var_refs ~f:(fun delayed_ref ->
        match delayed_ref.var with
        | `Not_set_yet -> () (* Variable not bound yet, will be set later *)
        | `Dim dim_var -> (
            match Row.get_dim_val env dim_var with
            | Some d ->
                Option.iter delayed_ref.var_ref.solved_dim ~f:(fun solved_dim ->
                    assert (solved_dim = d));
                delayed_ref.var_ref.solved_dim <- Some d
            | None -> () (* Not yet resolved *))
        | `Row row_var -> (
            match Row.get_row_from_env env row_var with
            | Some row ->
                let product = compute_row_product env row in
                Option.iter delayed_ref.var_ref.solved_dim ~f:(fun solved_dim ->
                    assert (solved_dim = product));
                delayed_ref.var_ref.solved_dim <- Some product
            | None -> () (* Not yet resolved *)))
  in
  match update_step.logic with
  | Transpose (Permute (_, var_refs), _) -> update_var_ref_list var_refs
  | Broadcast (Einsum (_, var_refs), _, _) -> update_var_ref_list var_refs
  | _ -> ()

let apply_env_step env update_step =
  iter_shapes update_step ~f:(apply_env_t env);
  update_delayed_var_refs env update_step

let%debug4_sexp propagate_shapes (update_step : update_step) : unit =
  (* Allow the derivation of constraints to depend on the shapes (currently, only Batch_slice
     does). *)
  iter_shapes update_step ~f:(apply_env_t !state);
  let _, ineqs = get_inequalities update_step in
  active_update_steps := update_step :: !active_update_steps;
  active_constraints := ineqs @ !active_constraints;
  let ineqs', env = Row.solve_inequalities ~stage:Row.Stage1 ineqs !state in
  let _debug_remaining_constraints : Row.constraint_ list = ineqs' in
  apply_env_step env update_step;
  state := env

let%debug4_sexp finish_inference (() : unit) : unit =
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
  active_constraints := [];
  active_update_steps := [];
  (* There should not be any shape variables remaining in any inference-undergoing update steps. *)
  state := Row.empty_env

let%debug4_sexp row_to_dims (row : Row.t) : int array =
  let open Row in
  let f = function
    | Dim { d; _ } -> d
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
  in
  match row with
  | { bcast = Row_var { v; _ }; _ } ->
      raise
      @@ Row.Shape_error
           ( "Not enough shape information: unresolved row variable "
             ^ Sexp.to_string_hum ([%sexp_of: row_var] v),
             [ Row_mismatch [ row ] ] )
  | { dims; bcast = Broadcastable; prov = _ } -> Array.of_list_map dims ~f

let to_dims (sh : t) : int array =
  finish_inference ();
  try Array.concat_map ~f:row_to_dims [| sh.batch; sh.output; sh.input |]
  with Row.Shape_error (s, trace) -> raise @@ Row.Shape_error (s, Shape_mismatch [ sh ] :: trace)

let to_padding (sh : t) : (Ir.Ops.axis_padding array * float) option =
  finish_inference ();
  (* FIXME: NOT IMPLEMENTED YET -- e.g. this should not be None if any of the padding isn't None.
     Also, the padded value should be inferred. *)
  try
    Option.map3 sh.batch_padding sh.output_padding sh.input_padding ~f:(fun batch output input ->
        (Array.concat [ batch; output; input ], 0.))
  with Row.Shape_error (s, trace) -> raise @@ Row.Shape_error (s, Shape_mismatch [ sh ] :: trace)

let to_labels (sh : t) : string array =
  Array.concat_map ~f:(Row.row_to_labels !state) [| sh.batch; sh.output; sh.input |]

let sexp_of_error_trace = function
  | Shape_mismatch ts -> Sexp.List (Sexp.Atom "Shape_mismatch" :: List.map ts ~f:sexp_of_t)
  | error_trace -> Row.sexp_of_error_trace error_trace

let () =
  Sexplib0.Sexp_conv.Exn_converter.add [%extension_constructor Row.Shape_error] (function
    | Row.Shape_error (arg0, arg1) ->
        let res0 = sexp_of_string arg0 and res1 = sexp_of_list sexp_of_error_trace arg1 in
        Sexplib0.Sexp.List [ Sexplib0.Sexp.Atom "lib/shape.ml.Shape_error"; res0; res1 ]
    | _ -> assert false)

(** *** Projection inference *** *)

let fresh_proj_ids update =
  let resolved_padding = ref [] in
  let inferred_padding = ref [] in
  let fetch_padding ~id row row_padding =
    let tn_opt = Ir.Tnode.find ~id in
    let is_resolved = match tn_opt with Some tn -> Lazy.is_val tn.padding | None -> false in
    Option.iter row_padding ~f:(fun padding ->
        Array.iter2_exn (Array.of_list row.Row.dims) padding ~f:(fun d p ->
            match d with
            | Row.Dim { proj_id = Some proj_id; _ } ->
                if is_resolved then resolved_padding := (proj_id, p) :: !resolved_padding
                else inferred_padding := (proj_id, p) :: !inferred_padding
            | _ -> ()))
  in
  let fresh_shape (sh : t) =
    sh.batch <- Row.fresh_row_proj sh.batch;
    sh.input <- Row.fresh_row_proj sh.input;
    sh.output <- Row.fresh_row_proj sh.output;
    fetch_padding ~id:sh.id sh.batch sh.batch_padding;
    fetch_padding ~id:sh.id sh.input sh.input_padding;
    fetch_padding ~id:sh.id sh.output sh.output_padding
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
      fresh_shape sh3);
  (!resolved_padding, !inferred_padding)

(** Computes the indexing into subtensors given the shape information of a tensor.
    [derive_projections] should only be invoked when the shapes are fully inferred already! *)
let%debug4_sexp derive_projections (update_step : update_step) : Idx.projections =
  finish_inference ();
  let resolved_padding, inferred_padding = fresh_proj_ids update_step in
  let _debug_update_step : update_step = update_step in
  let (proj_axis_env, ineqs) : proj_axis_env * Row.constraint_ list =
    get_inequalities ~for_projections:true update_step
  in
  (* We need to solve the equations/inequalities one last time because of fresh row variables
     potentially generated by [get_inequalities]. Since the variables in the shapes must be
     substituted-out at this point, the global state is already an empty env, but in principle we
     want to only find a local solution to not contaminate projections across operations. *)
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage1 ineqs Row.empty_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage2 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage3 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage4 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage5 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage6 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage7 unsolved local_env in
  assert (List.is_empty unsolved);
  let local_env = Row.populate_dim_proj_in_solved local_env in
  let rhs =
    match update_step.logic with
    | Broadcast (Defined_by_cd_logic, _, _)
    | Transpose (Defined_by_cd_logic, _)
    | Broadcast_tern (Defined_by_cd_logic, _, _, _) ->
        raise
        @@ Utils.User_error
             "Defined_by_cd_logic: use explicit ~logic annotations when defining this operation"
    | Terminal _ -> []
    | Transpose (_, sh) -> [ sh ]
    | Broadcast (_, sh1, sh2) -> [ sh1; sh2 ]
    | Broadcast_tern (_, sh1, sh2, sh3) -> [ sh1; sh2; sh3 ]
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
  let proj_env : Row.proj_env =
    Row.solve_proj_equations ~resolved_padding ~inferred_padding proj_eqs
  in
  let dims_of (sh : t) = sh.batch.dims @ sh.output.dims @ sh.input.dims in
  let lhs = update_step.shape in
  let lhs_dims = to_dims lhs in
  let rhs_dims = Array.of_list_map ~f:to_dims rhs in
  let all_dims : Row.dim list = List.concat_map ~f:dims_of @@ (lhs :: rhs) in
  (* Note: the ordering will affect performance of naive backends. *)
  let all_product_projs : (Row.proj_id * int) list =
    Utils.unique_keep_first ~equal:(fun (p, _) (q, _) -> Row.equal_proj_id p q)
    @@ List.filter_map all_dims ~f:(Row.get_product_proj proj_env)
  in
  let product_space : int array = Array.of_list_map all_product_projs ~f:snd in
  let product_iterators : Idx.symbol array =
    Array.of_list_map all_product_projs ~f:(fun (p, _) -> Row.proj_to_iterator_exn proj_env p)
  in
  let indices_of_sh (sh : t) =
    Array.of_list_map ~f:(Row.get_dim_index proj_env)
    @@ List.concat [ sh.batch.dims; sh.output.dims; sh.input.dims ]
  in
  try
    Idx.
      {
        product_space;
        lhs_dims;
        rhs_dims;
        product_iterators;
        project_lhs = indices_of_sh lhs;
        project_rhs = Array.of_list_map ~f:indices_of_sh rhs;
        debug_info =
          {
            spec = logic_to_spec update_step.logic;
            derived_for = sexp_of_update_step update_step;
            trace = [ ("derive_projections", Idx.unique_debug_id ()) ];
          };
      }
  with Row.Shape_error (s, trace) ->
    raise @@ Row.Shape_error (s, Shape_mismatch (lhs :: rhs) :: trace)

(** {2 Shape builders.} *)

let make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes
    ?(deduced = Not_constrained) ~debug_name ~id () =
  let open Row in
  let known_no_batch =
    match (batch_dims, batch_axes) with Some [], None -> true | None, Some [] -> true | _ -> false
  in
  let num_dim1_output = Option.to_list output_dims |> List.join |> List.count ~f:(fun d -> d = 1) in
  let f kind d =
    match kind with
    | `Batch | `Input -> get_dim ~d ()
    | `Output ->
        if (not known_no_batch) && num_dim1_output = 1 && d = 1 then
          let label = debug_name ^ "_output" in
          get_dim ~d ~label ()
        else get_dim ~d ()
  in
  let make_dims kind ds =
    { dims = List.map ~f:(f kind) ds; bcast = Broadcastable; prov = provenance ~sh_id:id ~kind }
  in
  let make_axes kind ds =
    {
      dims = List.map ~f:(fun (label, d) -> get_dim ~d ~label ()) ds;
      bcast = Broadcastable;
      prov = provenance ~sh_id:id ~kind;
    }
  in
  let make_unknown kind =
    {
      dims = [];
      bcast = Row_var { v = get_row_var (); beg_dims = [] };
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

let shape_spec_to_dims_bio labels =
  let dim_var_env = Hashtbl.create (module String) in
  let f _kind = function
    | Label s when String.contains s '=' -> (
        let _label, dim =
          match String.split s ~on:'=' with
          | [ l; d ] -> (l, d)
          | _ -> invalid_arg "shape_spec_to_dims_bio: too many '='"
        in
        (* This is not a dimension label i.e. unit! *)
        try Row.get_dim ~d:(Int.of_string dim) ()
        with _ -> invalid_arg "shape_spec_to_dims_bio: int expected after '='")
    | Label name ->
        Var (Hashtbl.find_or_add dim_var_env name ~default:(fun () -> Row.get_var ~name ()))
    | Fixed_index d -> Row.get_dim ~d ()
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
  in
  let row_var_env = Hashtbl.create (module String) in
  axes_spec_to_dims_bio ~row_var_env ~dim_var_env ~f labels

let of_spec ?(deduced = Not_constrained) ~debug_name ~id spec =
  let batch, input, output = shape_spec_to_dims_bio ~sh_id:id @@ axis_labels_of_spec spec in
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
  let n_outputs = List.length @@ sh.output.dims in
  let n_batch = List.length @@ sh.batch.dims in
  let dims_to_string kind =
    let dims = (row_of_kind kind sh).dims in
    String.concat ~sep:","
    @@ List.mapi dims ~f:(fun i d ->
        let num =
          match kind with `Input -> n_batch + n_outputs + i | `Output -> n_batch + i | `Batch -> i
        in
        match style with
        | Row.Only_labels | Axis_size | Projection_and_size -> Row.dim_to_string style d
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
    (* Enumerate axes backwards. *)
    Array.of_list_rev_mapi sh.batch.dims ~f:(fun i _ ->
        AxisKey.{ in_axes = `Batch; pos = i + 1; from_end = true })
  in
  let i_dims =
    Array.of_list_rev_mapi sh.input.dims ~f:(fun i _ ->
        AxisKey.{ in_axes = `Input; pos = i + 1; from_end = true })
  in
  let o_dims =
    Array.of_list_rev_mapi sh.output.dims ~f:(fun i _ ->
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
  let num_input_axes = List.length sh.input.dims in
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
  in
  let p_labels = Einsum_parser.(axis_labels @@ axis_labels_of_spec priorities) in
  axis_map_to_dims_index p_labels |> Array.map ~f

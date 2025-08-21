(** {1 Tensor shape types, shape inference, projection inference.} *)

open Base
module Lazy = Utils.Lazy
module Idx = Ir.Indexing

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_SHAPE=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_SHAPE"]

(** {2 Shape types and inference.} *)

(** An index pointing to any of a shape's axes, including the kind of the axis
    ([Batch, Input, Output]) and the position (which is counted from the end to facilitate
    broadcasting).

    Note the following inconsistency due to differing conventions in function notation and matrix
    notation: for label specifications and einsum notation, we write "batch|inputs->outputs", but
    when we convert a shape to an [Ndarray] index we do it in the order [[batch; outputs; inputs]].
*)
module AxisKey = struct
  module T = struct
    type kind = [ `Batch | `Input | `Output ] [@@deriving equal, compare, sexp, hash]

    type t = {
      in_axes : kind;
      pos : int;  (** Indices start at [1], counted from the end if [from_end] is true. *)
      from_end : bool;
          (** Axes are indexed from the front (rarely) or from the end (typically), to avoid
              reindexing when broadcasting. *)
    }
    [@@deriving equal, compare, sexp]
  end

  include T
  include Comparator.Make (T)
end

type 'a axis_map = 'a Map.M(AxisKey).t [@@deriving compare, sexp]

(** Specification for individual axes in the einsum notation. *)
type axis_spec =
  | Label of string  (** A variable axis label. *)
  | Fixed_index of int  (** A fixed index, used for projection. *)
  | Conv_spec of { stride : int; output_label : string; dilation : int; kernel_label : string }
      (** Convolution-style axis specification: stride*output + dilation*kernel. *)
[@@deriving compare, sexp]

type parsed_axis_labels = {
  bcast_batch : string option;
  bcast_input : string option;
  bcast_output : string option;
  given_batch : int;
  given_input : int;
  given_output : int;
  given_beg_batch : int;
  given_beg_input : int;
  given_beg_output : int;
  labels : axis_spec axis_map;
}
[@@deriving compare, sexp, fields]
(** The labels are strings assigned to [AxisKey] axes. Moreover the [bcast_] fields represent
    whether additional leading/middle axes are allowed (corresponding to the dot-ellipsis syntax for
    broadcasting). The string can be used to identify a row variable, and defaults to ["batch"],
    ["input"], ["output"] respectively when parsing ["..."]. The [given_] fields count the number of
    specified axes of the corresponding kind in [labels] where [from_end=true], [given_beg_] where
    [from_end=false]. *)

let axis_labels parsed = parsed.labels

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

type compose_type = Pointwise_bin | Compose | Einsum of string [@@deriving sexp, equal]

type transpose_type =
  | Transpose
  | Pointwise_un
  | Permute of string
  | Batch_slice of Idx.static_symbol
  | Uint4x32_to_prec of Ir.Ops.prec Lazy.t
  | Uint4x32_to_prec1 of Ir.Ops.prec Lazy.t
[@@deriving equal, sexp]

type terminal_type = Data of Ir.Assignments.init_data | Fetch of Ir.Assignments.fetch_op
[@@deriving equal, sexp_of]

type ternary_type = Pointwise_tern | Compose_accumulate [@@deriving sexp, equal]

let identifier ~multichar =
  let open Angstrom in
  if multichar then lift2 ( ^ ) (take_while1 Char.is_alpha) (take_while Char.is_alphanum)
  else Angstrom.satisfy Char.is_alpha >>| Char.to_string

let integer = Angstrom.(take_while1 Char.is_digit >>| Int.of_string)

let scaled_identifier ~multichar =
  let open Angstrom in
  integer <* char '*'
  >>= (fun coeff -> identifier ~multichar >>| fun id -> (coeff, id))
  <|> (identifier ~multichar >>| fun id -> (1, id))

let conv_term ~multichar =
  let open Angstrom in
  let* stride, output_label = scaled_identifier ~multichar in
  char '+' *> scaled_identifier ~multichar
  >>| (fun (dilation, kernel_label) -> Conv_spec { stride; output_label; dilation; kernel_label })
  <|>
  if stride <> 1 then
    return (Conv_spec { stride; output_label; dilation = 0; kernel_label = "_stride_only" })
  else fail "neither convolution nor strided iteration"

let opt_separators = Angstrom.take_while (fun c -> Char.is_whitespace c || Char.equal c ',')

let separators_with_comma =
  let open Angstrom in
  let* sep = opt_separators in
  if String.contains sep ',' then return () else fail "comma expected"

(** Parse a single axis specification that can be a label, fixed index, or conv expression. *)
let parse_single_axis_spec ~multichar =
  let open Angstrom in
  choice
    [
      conv_term ~multichar <?> "conv_term";
      integer >>| (fun i -> Fixed_index i) <?> "fixed_index";
      identifier ~multichar >>| (fun s -> Label s) <?> "label";
    ]

let axes_spec ~from_end ~multichar : _ Angstrom.t =
  let open Angstrom in
  let result =
    let p n i = if from_end then n - i else i + 1 in
    lift (fun l ->
        let n = List.length l in
        List.mapi l ~f:(fun i v -> (p n i, v)))
    @@ sep_by1
         (if multichar then separators_with_comma else opt_separators >>| ignore)
         (parse_single_axis_spec ~multichar)
  in
  opt_separators *> result <* opt_separators <?> "axes_spec"

let axis_labels_of_spec_parser ~multichar : parsed_axis_labels Angstrom.t =
  let open Angstrom in
  let combine ~key:_ _v1 _v2 = assert false in
  let axes_spec ~from_end =
    axes_spec ~from_end ~multichar <?> if from_end then "axes_spec" else "axes_spec_beg"
  in
  let ellipsis_spec = string "..." <|> (string ".." *> identifier ~multichar <* string "..") in
  let ellipsis_spec = ellipsis_spec <?> "ellipsis_spec" in
  let for_row ~kind in_axes beg_axes row_var_spec end_axes =
    let f from_end (pos, label) = (AxisKey.{ in_axes; pos; from_end }, label) in
    let from_beg = Map.of_alist_exn (module AxisKey) @@ List.map beg_axes ~f:(f false) in
    let from_end = Map.of_alist_exn (module AxisKey) @@ List.map end_axes ~f:(f true) in
    ( Option.map row_var_spec ~f:(fun rv -> if String.equal rv "..." then kind else rv),
      List.length end_axes,
      List.length beg_axes,
      Map.merge_skewed ~combine from_beg from_end )
  in
  let parse_row ~kind in_axes =
    let row = lift3 (for_row ~kind in_axes) in
    opt_separators
    *> (row (axes_spec ~from_end:false) (lift Option.some ellipsis_spec) (axes_spec ~from_end:true)
       <|> row (return []) (lift Option.some ellipsis_spec) (axes_spec ~from_end:true)
       <|> row (axes_spec ~from_end:false) (lift Option.some ellipsis_spec) (return [])
       <|> row (return []) (return None) (axes_spec ~from_end:true)
       <|> row (return []) (lift Option.some ellipsis_spec) (return []))
    <* opt_separators <?> "row_spec"
  in
  let default = Option.value ~default:(None, 0, 0, Map.empty (module AxisKey)) in
  let shape = lift3 (fun batch input output -> (default batch, default input, output)) in
  let p_b = lift Option.some @@ parse_row ~kind:"batch" `Batch <?> "batch_spec" in
  let p_i = lift Option.some @@ parse_row ~kind:"input" `Input <?> "input_spec" in
  let p_o = parse_row ~kind:"output" `Output <?> "output_spec" in
  let+ ( (bcast_batch, given_batch, given_beg_batch, batch_labels),
         (bcast_input, given_input, given_beg_input, input_labels),
         (bcast_output, given_output, given_beg_output, output_labels) ) =
    shape (return None) (p_i <* string "->") p_o
    <|> shape (p_b <* char '|') (p_i <* string "->") p_o
    <|> shape (p_b <* char '|') (return None) p_o
    <|> shape (return None) (return None) p_o
    <?> "shape_spec"
  in
  let labels =
    Map.merge_skewed ~combine input_labels @@ Map.merge_skewed ~combine output_labels batch_labels
  in
  {
    bcast_batch;
    bcast_input;
    bcast_output;
    given_batch;
    given_input;
    given_output;
    given_beg_batch;
    given_beg_input;
    given_beg_output;
    labels;
  }

let axis_labels_of_spec spec =
  let multichar = String.contains spec ',' in
  match
    Angstrom.(
      parse_string ~consume:Consume.All (axis_labels_of_spec_parser ~multichar <* end_of_input) spec)
  with
  | Ok result -> result
  | Error msg ->
      raise
      @@ Utils.User_error ("Shape.axis_labels_of_spec: while parsing: " ^ spec ^ " error: " ^ msg)

let einsum_of_spec_parser ~multichar : _ Angstrom.t =
  let open Angstrom in
  let p = axis_labels_of_spec_parser ~multichar in
  lift3
    (fun a b c -> (a, Some b, c))
    (p <?> "RHS1" <* char ';')
    (p <?> "RHS2")
    (string "=>" *> (p <?> "LHS"))
  <|> lift2 (fun a c -> (a, None, c)) (p <?> "RHS") (string "=>" *> (p <?> "LHS"))
  <?> "einsum_spec"

let einsum_of_spec spec =
  let multichar = String.contains spec ',' in
  match
    Angstrom.(
      parse_string ~consume:Consume.All (einsum_of_spec_parser ~multichar <* end_of_input) spec)
  with
  | Ok result -> result
  | Error msg ->
      raise @@ Utils.User_error ("Shape.einsum_of_spec: while parsing: " ^ spec ^ " error: " ^ msg)

type logic =
  | Broadcast of compose_type * t * t
  | Transpose of transpose_type * t
  | Broadcast_tern of ternary_type * t * t * t
  | Terminal of terminal_type
[@@deriving equal, sexp_of]

let logic_to_spec = function
  | Broadcast (Pointwise_bin, _, _)
  | Transpose (Pointwise_un, _)
  | Broadcast_tern (Pointwise_tern, _, _, _) ->
      "."
  | Broadcast (Compose, _, _) | Broadcast_tern (Compose_accumulate, _, _, _) -> "@"
  | Broadcast (Einsum spec, _, _) | Transpose (Permute spec, _) -> spec
  | Transpose (Transpose, _) -> "T"
  | Transpose (Batch_slice _, _) -> "@|"
  | Transpose (Uint4x32_to_prec _, _) -> "U4x32"
  | Transpose (Uint4x32_to_prec1 _, _) -> "U4x32_1"
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
        (Row.Row_var { v; beg_dims }, []))
  in
  let to_row kind v dims beg_dims =
    let bcast, beg_dims = to_bcast kind v beg_dims in
    { Row.dims = beg_dims @ to_dim kind dims; bcast; id = Row.row_id ~sh_id ~kind }
  in
  let batch = to_row `Batch labels.bcast_batch b_dims beg_b_dims in
  let input = to_row `Input labels.bcast_input i_dims beg_i_dims in
  let output = to_row `Output labels.bcast_output o_dims beg_o_dims in
  (batch, input, output)

let einsum_slot_spec_to_dims_bio ~generative ~sh_id ~row_var_env ~dim_var_env labels =
  let equal = Row.equal_kind in
  let proj_env_update = ref @@ Row.dim_map_empty in
  let extras = ref [] in
  let f kind = function
    | Label label ->
        Row.Var (Hashtbl.find_or_add dim_var_env label ~default:(fun () -> Row.get_var ~label ()))
    | Fixed_index 0 when Option.value ~default:false @@ List.Assoc.find generative ~equal kind ->
        Row.get_dim ~d:1 ()
    | Fixed_index i ->
        let var = Row.get_var () in
        let d = Row.Var var in
        proj_env_update := Map.add_exn !proj_env_update ~key:var ~data:(Idx.Fixed_idx i);
        extras := Row.Dim_constr { d; constr = At_least_dim (i + 1) } :: !extras;
        d
    | Conv_spec { stride; output_label; dilation; kernel_label } ->
        let output_dim =
          Row.Var
            (Hashtbl.find_or_add dim_var_env output_label ~default:(fun () ->
                 Row.get_var ~label:output_label ()))
        in
        let kernel_dim =
          if String.equal kernel_label "_stride_only" then
            (* For strided iteration (dilation=0), use fixed dimension 0 for kernel *)
            Row.get_dim ~d:0 ()
          else
            Row.Var
              (Hashtbl.find_or_add dim_var_env kernel_label ~default:(fun () ->
                   Row.get_var ~label:kernel_label ()))
        in
        Row.Conv_input { stride; output = output_dim; dilation; kernel = kernel_dim }
  in
  let result = axes_spec_to_dims_bio ~sh_id ~row_var_env ~dim_var_env ~f labels in
  (!extras, !proj_env_update, result)

type proj_axis_env = Idx.axis_index Row.dim_map [@@deriving sexp]

let%debug4_sexp get_inequalities ({ shape = cur_sh; logic; id = _ } as _upd : update_step) :
    proj_axis_env * Row.constraint_ list =
  let generative =
    [
      (`Batch, List.is_empty cur_sh.batch.dims);
      (`Input, List.is_empty cur_sh.input.dims);
      (`Output, List.is_empty cur_sh.output.dims);
    ]
  in
  let _debug_cur_sh : t = cur_sh in
  let _debug_logic : logic = logic in
  let open Row in
  let mark_terminal () =
    [ Terminal_row cur_sh.batch; Terminal_row cur_sh.input; Terminal_row cur_sh.output ]
  in
  match logic with
  | Terminal (Fetch Range_over_offsets) -> (Row.dim_map_empty, mark_terminal ())
  | Terminal (Fetch (Constant _)) -> (Row.dim_map_empty, mark_terminal ())
  | Terminal (Fetch (Constant_bits _)) -> (Row.dim_map_empty, mark_terminal ())
  | Terminal (Data (Reshape nd)) ->
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
          }
        :: mark_terminal () )
  | Terminal (Data (Keep_shape_no_padding nd)) ->
      (* FIXME: constrain padding to "not padded". *)
      ( dim_map_empty,
        Rows_constr
          {
            r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
            constr =
              Exact (Ir.Ndarray.dims nd |> Array.map ~f:(fun d -> get_dim ~d ()) |> Array.to_list);
          }
        :: mark_terminal () )
  | Terminal (Data (Padded { data; padding; padded_value })) ->
      (* FIXME: constrain padding. *)
      ignore (padding, padded_value);
      ( dim_map_empty,
        Rows_constr
          {
            r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
            constr =
              Exact (Ir.Ndarray.dims data |> Array.map ~f:(fun d -> get_dim ~d ()) |> Array.to_list);
          }
        :: mark_terminal () )
  | Terminal (Fetch (Constant_fill values)) ->
      let len = Array.length values in
      ( dim_map_empty,
        Rows_constr
          {
            r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
            constr = Total_elems { numerator = Num_elems len; divided_by = [] };
          }
        :: mark_terminal () )
  | Terminal (Fetch (Slice { sliced = tn; batch_idx = _ })) ->
      if Lazy.is_val tn.dims then
        ( dim_map_empty,
          Rows_constr
            {
              r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
              constr =
                Exact
                  (Lazy.force tn.dims |> Array.to_list |> List.tl_exn
                  |> List.map ~f:(fun d -> get_dim ~d ()));
            }
          :: mark_terminal () )
      else (Row.dim_map_empty, mark_terminal ())
  | Terminal (Fetch (Embed_symbol _)) -> (Row.dim_map_empty, mark_terminal ())
  | Terminal (Fetch Embed_self_id) -> (Row.dim_map_empty, mark_terminal ())
  | Transpose (Transpose, sh) ->
      ( Row.dim_map_empty,
        [
          Row_ineq { cur = cur_sh.batch; subr = sh.batch };
          Row_ineq { cur = cur_sh.input; subr = sh.output };
          Row_ineq { cur = cur_sh.output; subr = sh.input };
        ] )
  | Transpose (Pointwise_un, sh) ->
      ( Row.dim_map_empty,
        [
          Row_ineq { cur = cur_sh.batch; subr = sh.batch };
          Row_ineq { cur = cur_sh.input; subr = sh.input };
          Row_ineq { cur = cur_sh.output; subr = sh.output };
        ] )
  | Broadcast (Compose, sh1, sh2) ->
      ( Row.dim_map_empty,
        [
          Row_ineq { cur = sh1.input; subr = sh2.output };
          Row_ineq { cur = cur_sh.batch; subr = sh1.batch };
          Row_ineq { cur = cur_sh.batch; subr = sh2.batch };
          Row_ineq { cur = cur_sh.input; subr = sh2.input };
          Row_ineq { cur = cur_sh.output; subr = sh1.output };
        ] )
  | Broadcast (Pointwise_bin, sh1, sh2) ->
      ( Row.dim_map_empty,
        [
          Row_ineq { cur = cur_sh.batch; subr = sh1.batch };
          Row_ineq { cur = cur_sh.batch; subr = sh2.batch };
          Row_ineq { cur = cur_sh.input; subr = sh1.input };
          Row_ineq { cur = cur_sh.input; subr = sh2.input };
          Row_ineq { cur = cur_sh.output; subr = sh1.output };
          Row_ineq { cur = cur_sh.output; subr = sh2.output };
        ] )
  | Broadcast_tern (Compose_accumulate, sh1, sh2, sh3) ->
      ( Row.dim_map_empty,
        [
          Row_ineq { cur = sh1.input; subr = sh2.output };
          Row_ineq { cur = cur_sh.batch; subr = sh1.batch };
          Row_ineq { cur = cur_sh.batch; subr = sh2.batch };
          Row_ineq { cur = cur_sh.input; subr = sh2.input };
          Row_ineq { cur = cur_sh.output; subr = sh1.output };
          Row_ineq { cur = cur_sh.batch; subr = sh3.batch };
          Row_ineq { cur = cur_sh.input; subr = sh3.input };
          Row_ineq { cur = cur_sh.output; subr = sh3.output };
        ] )
  | Broadcast_tern (Pointwise_tern, sh1, sh2, sh3) ->
      ( Row.dim_map_empty,
        [
          Row_ineq { cur = cur_sh.batch; subr = sh1.batch };
          Row_ineq { cur = cur_sh.batch; subr = sh2.batch };
          Row_ineq { cur = cur_sh.batch; subr = sh3.batch };
          Row_ineq { cur = cur_sh.input; subr = sh1.input };
          Row_ineq { cur = cur_sh.input; subr = sh2.input };
          Row_ineq { cur = cur_sh.input; subr = sh3.input };
          Row_ineq { cur = cur_sh.output; subr = sh1.output };
          Row_ineq { cur = cur_sh.output; subr = sh2.output };
          Row_ineq { cur = cur_sh.output; subr = sh3.output };
        ] )
  | Transpose (Batch_slice { static_range; static_symbol }, sh) ->
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
              id = Row.row_id ~sh_id:cur_sh.id ~kind:`Batch;
            }
        | Row_var { v; beg_dims } ->
            {
              dims = cur_sh.batch.dims;
              bcast = Row_var { v; beg_dims = slice_var :: beg_dims };
              id = Row.row_id ~sh_id:cur_sh.id ~kind:`Batch;
            }
      in
      ( proj_axis_env,
        (Option.to_list static_range
        |> List.map ~f:(fun range -> Dim_eq { d1 = get_dim ~d:range (); d2 = slice_var }))
        @ [
            Row_eq { r1 = expanded_batch; r2 = sh.batch };
            Row_eq { r1 = cur_sh.input; r2 = sh.input };
            Row_eq { r1 = cur_sh.output; r2 = sh.output };
          ] )
  | Transpose (Permute spec, sh) ->
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
        einsum_slot_spec_to_dims_bio ~generative:[] ~sh_id:sh.id ~row_var_env ~dim_var_env ls_rhs
      in
      let extras_lhs, proj_env_lhs, (b_lhs, i_lhs, o_lhs) =
        einsum_slot_spec_to_dims_bio ~generative ~sh_id:cur_sh.id ~row_var_env ~dim_var_env ls_lhs
      in
      let proj_env =
        let combine ~key:_ _ _ = assert false in
        Map.merge_skewed ~combine proj_env_rhs proj_env_lhs
      in
      ( proj_env,
        extras_rhs @ extras_lhs
        @ [
            Row_ineq { cur = cur_sh.batch; subr = b_lhs };
            Row_ineq { cur = b_rhs; subr = sh.batch };
            Row_ineq { cur = cur_sh.input; subr = i_lhs };
            Row_ineq { cur = i_rhs; subr = sh.input };
            Row_ineq { cur = cur_sh.output; subr = o_lhs };
            Row_ineq { cur = o_rhs; subr = sh.output };
          ] )
  | Transpose (Uint4x32_to_prec target_prec, sh) ->
      let var = get_var () in
      let coeff =
        Utils.safe_lazy [%string "Uint4x32 %{sh.id#Int} to_prec_of %{cur_sh.id#Int}"] (fun () ->
            16 / Ir.Ops.prec_in_bytes (Lazy.force target_prec))
      in
      ( Row.dim_map_empty,
        [
          Rows_constr { r = [ sh.batch; sh.output; sh.input ]; constr = Row.Exact [ Var var ] };
          Rows_constr
            {
              r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
              constr =
                Total_elems
                  { numerator = Row.Strided_var { coeff; var; denom = 1 }; divided_by = [] };
            };
        ] )
  | Transpose (Uint4x32_to_prec1 _target_prec, sh) ->
      (* Non-vectorized version: preserves shape exactly *)
      ( Row.dim_map_empty,
        [
          Row_ineq { cur = cur_sh.batch; subr = sh.batch };
          Row_ineq { cur = sh.batch; subr = cur_sh.batch };
          Row_ineq { cur = cur_sh.input; subr = sh.input };
          Row_ineq { cur = sh.input; subr = cur_sh.input };
          Row_ineq { cur = cur_sh.output; subr = sh.output };
          Row_ineq { cur = sh.output; subr = cur_sh.output };
        ] )
  | Broadcast (Einsum spec, sh1, sh2) ->
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
        einsum_slot_spec_to_dims_bio ~generative:[] ~sh_id:sh1.id ~row_var_env ~dim_var_env ls_rhs1
      in
      let extras_rhs2, proj_env_rhs2, (b_rhs2, i_rhs2, o_rhs2) =
        einsum_slot_spec_to_dims_bio ~generative:[] ~sh_id:sh2.id ~row_var_env ~dim_var_env ls_rhs2
      in
      let extras_lhs, proj_env_lhs, (b_lhs, i_lhs, o_lhs) =
        einsum_slot_spec_to_dims_bio ~generative ~sh_id:cur_sh.id ~row_var_env ~dim_var_env ls_lhs
      in
      let proj_env =
        let combine ~key:_ _ _ = assert false in
        Map.merge_skewed ~combine proj_env_rhs1
        @@ Map.merge_skewed ~combine proj_env_rhs2 proj_env_lhs
      in
      (* Forget the old proj_env as it is not relevant after a propagate_shapes call completes. *)
      ( proj_env,
        extras_rhs1 @ extras_rhs2 @ extras_lhs
        @ [
            Row_ineq { cur = cur_sh.batch; subr = b_lhs };
            Row_ineq { cur = b_rhs1; subr = sh1.batch };
            Row_ineq { cur = b_rhs2; subr = sh2.batch };
            Row_ineq { cur = cur_sh.input; subr = i_lhs };
            Row_ineq { cur = i_rhs1; subr = sh1.input };
            Row_ineq { cur = i_rhs2; subr = sh2.input };
            Row_ineq { cur = cur_sh.output; subr = o_lhs };
            Row_ineq { cur = o_rhs1; subr = sh1.output };
            Row_ineq { cur = o_rhs2; subr = sh2.output };
          ] )

let state = ref Row.empty_env
let active_update_steps = ref []
let active_constraints = ref []

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

let all_rows update_step =
  let rows_sh sh = [ sh.batch; sh.input; sh.output ] in
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

let%debug4_sexp propagate_shapes (update_step : update_step) : unit =
  (* Allow the derivation of constraints to depend on the shapes (currently, only Batch_slice
     does). *)
  iter_shapes update_step ~f:(apply_env_t !state);
  let _, ineqs = get_inequalities update_step in
  active_update_steps := update_step :: !active_update_steps;
  let _debug_new_active_update_steps : update_step list = !active_update_steps in
  active_constraints := ineqs @ !active_constraints;
  let ineqs', env = Row.solve_inequalities ~stage:Row.Stage1 ineqs !state in
  let _debug_remaining_constraints : Row.constraint_ list = ineqs' in
  iter_shapes update_step ~f:(apply_env_t env);
  state := env

let%debug4_sexp finish_inference (() : unit) : unit =
  (* TODO: optimize to keep all needed information in unsolved, rather than starting with all
     constraints. *)
  let unsolved, env = Row.solve_inequalities ~stage:Stage2 !active_constraints !state in
  let unsolved, env = Row.solve_inequalities ~stage:Stage3 unsolved env in
  let all_update_rows =
    List.concat_map ~f:all_rows !active_update_steps
    |> List.map ~f:(Row.subst_row env)
    |> List.dedup_and_sort ~compare:Row.compare
  in
  let unsolved = List.map ~f:(fun r -> Row.Shape_row r) all_update_rows @ unsolved in
  let unsolved, env = Row.solve_inequalities ~stage:Stage4 unsolved env in
  let unsolved, env = Row.solve_inequalities ~stage:Stage5 unsolved env in
  let unsolved, env = Row.solve_inequalities ~stage:Stage6 unsolved env in
  let unsolved, env = Row.solve_inequalities ~stage:Stage7 unsolved env in
  assert (List.is_empty unsolved);
  let _active_update_steps : update_step list = !active_update_steps in
  List.iter ~f:(iter_shapes ~f:(apply_env_t env)) !active_update_steps;
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
    | Conv_input _ ->
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
  | { dims; bcast = Broadcastable; id = _ } -> Array.of_list_map dims ~f

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
    get_inequalities update_step
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
  (* Important: ineqs must not be substituted / solved before getting proj_equations, because
     get_inequalities provides indexing information that is lost after substitution. *)
  let proj_eqs : Row.proj_equation list = Row.get_proj_equations ineqs proj_axis_env local_env in
  let proj_env : Row.proj_env =
    Row.solve_proj_equations ~resolved_padding ~inferred_padding proj_eqs
  in
  let dims_of (sh : t) = sh.batch.dims @ sh.output.dims @ sh.input.dims in
  let lhs = update_step.shape in
  let rhs =
    match update_step.logic with
    | Terminal _ -> []
    | Transpose (_, sh) -> [ sh ]
    | Broadcast (_, sh1, sh2) -> [ sh1; sh2 ]
    | Broadcast_tern (_, sh1, sh2, sh3) -> [ sh1; sh2; sh3 ]
  in
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
  let make_dims kind ds =
    {
      dims = List.map ~f:(fun d -> get_dim ~d ()) ds;
      bcast = Broadcastable;
      id = row_id ~sh_id:id ~kind;
    }
  in
  let make_axes kind ds =
    {
      dims = List.map ~f:(fun (label, d) -> get_dim ~d ~label ()) ds;
      bcast = Broadcastable;
      id = row_id ~sh_id:id ~kind;
    }
  in
  let make_unknown kind =
    {
      dims = [];
      bcast = Row_var { v = get_row_var (); beg_dims = [] };
      id = row_id ~sh_id:id ~kind;
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
        let more_ineqs, env = Row.unify_row ~stage:Stage2 (input, output) !state in
        assert (List.is_empty more_ineqs);
        state := env
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ("Input_equals_output / " ^ s, Shape_mismatch [ result ] :: trace)));
  result

let shape_spec_to_dims_bio labels =
  let dim_var_env = Hashtbl.create (module String) in
  let f _kind = function
    | Label s when String.contains s '=' -> (
        let label, dim =
          match String.split s ~on:'=' with
          | [ l; d ] -> (l, d)
          | _ -> invalid_arg "shape_spec_to_dims_bio: too many '='"
        in
        try Row.get_dim ~d:(Int.of_string dim) ~label ()
        with _ -> invalid_arg "shape_spec_to_dims_bio: int expected after '='")
    | Label label ->
        Var (Hashtbl.find_or_add dim_var_env label ~default:(fun () -> Row.get_var ~label ()))
    | Fixed_index d -> Row.get_dim ~d ()
    | Conv_spec { stride; output_label; dilation; kernel_label } ->
        let output_dim =
          Row.Var
            (Hashtbl.find_or_add dim_var_env output_label ~default:(fun () ->
                 Row.get_var ~label:output_label ()))
        in
        let kernel_dim =
          if String.equal kernel_label "_stride_only" then
            (* For strided iteration (dilation=0), use fixed dimension 0 for kernel *)
            Row.get_dim ~d:0 ()
          else
            Row.Var
              (Hashtbl.find_or_add dim_var_env kernel_label ~default:(fun () ->
                   Row.get_var ~label:kernel_label ()))
        in
        Row.Conv_input { stride; output = output_dim; dilation; kernel = kernel_dim }
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
        let more_ineqs, env = Row.unify_row ~stage:Stage2 (input, output) !state in
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
             match kind with
             | `Input -> n_batch + n_outputs + i
             | `Output -> n_batch + i
             | `Batch -> i
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

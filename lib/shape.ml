(** Tensor shape types, shape inference, projection inference. *)

open Base
module Utils = Arrayjit.Utils

(** *** Shape types and inference *** *)

(** An index pointing to any of a shape's axes, including the kind of the axis ([Batch, Input, Output])
    and the position (which is counted from the end to facilitate broadcasting).

    Note the following inconsistency due to differing conventions in function notation and matrix notation:
    for label specifications and einsum notation, we write "batch|inputs->outputs", but when we convert
    a shape to an [Ndarray] index we do it in the order [[batch; outputs; inputs]]. *)
module AxisKey = struct
  module T = struct
    type kind = [ `Batch | `Input | `Output ] [@@deriving equal, compare, sexp, hash]

    type t = {
      in_axes : kind;
      from_end : int;
          (** Axes are indexed from the end, to avoid reindexing when broadcasting; starting with [1]. *)
    }
    [@@deriving equal, compare, sexp]

    let to_string key =
      (match key.in_axes with `Batch -> "bch" | `Input -> "inp" | `Output -> "out")
      ^ Int.to_string key.from_end
  end

  include T
  include Comparator.Make (T)
end

type 'a axis_map = 'a Map.M(AxisKey).t [@@deriving compare, sexp]

type parsed_axis_labels = {
  bcast_batch : string option;
  bcast_input : string option;
  bcast_output : string option;
  given_batch : int;
  given_input : int;
  given_output : int;
  labels : (string, int) Either.t axis_map;
}
[@@deriving compare, sexp, fields]
(** The labels are strings assigned to [AxisKey] axes. Moreover the [bcast_] fields represent whether
    additional leading axes are allowed (corresponding to the dot-ellipsis syntax for broadcasting).
    The string can be used to identify a row variable, and defaults to ["batch"],  ["input"], ["output"]
    respectively when parsing ["..."].
    The [given_] fields count the number of specified axes of the corresponding kind in [labels]. *)

let bcast_of_kind = function `Batch -> bcast_batch | `Input -> bcast_input | `Output -> bcast_output
let given_of_kind = function `Batch -> given_batch | `Input -> given_input | `Output -> given_output

type t = {
  mutable batch : Row.t;
  mutable input : Row.t;
  mutable output : Row.t;
  id : int;  (** A node that has the same shape as this shape. *)
  debug_name : string;
}
[@@deriving equal, fields, sexp]
(** The datatype from which the actual Tensor shapes are computed.

    Mutability is sufficient to perform inference, since there is no need for backtracking and
    no explicit unification variables for now. [Unknown] stands for "not yet specified". *)

let row_of_kind = function `Batch -> batch | `Input -> input | `Output -> output

let map_over_kind ~f kind sh =
  match kind with
  | `Batch -> { sh with batch = f sh.batch }
  | `Input -> { sh with input = f sh.input }
  | `Output -> { sh with output = f sh.output }

let update_kind ~f kind sh =
  match kind with
  | `Batch -> sh.batch <- f sh.batch
  | `Input -> sh.input <- f sh.input
  | `Output -> sh.output <- f sh.output

type deduce_within_shape = Not_constrained | Input_equals_output [@@deriving compare, sexp, variants]

type compose_type =
  | Pointwise_bin  (** NumPy-style broadcast matching batch, input and output axes, e.g. as in [s1 + s2]. *)
  | Compose
      (** Compose the outputs of the second shape with the inputs of the first shape, i.e. the shape of
      [fun x -> s1(s2(x))], or [s1 * s2] where [*] is the inner product (e.g. matrix multiply). *)
  | Einsum of string
      (** The [einsum] syntax: LABELS1;LABELS2=>LABELS3, where LABELSi are labels specifications.
      Note that currently [Compose] is not redundant with [Einsum], because it enables more shape
      inference: [Einsum] is limited to [Pointwise_bin]-like broadcasting, while [Compose] broadcasts
      inputs of the "operator" against outputs of the "operand" (matching up an arbitrary number of axes).
      The [axis_labels] use pseudo-labels local to the notation, to line up the axes.
      For [Einsum (ls1^";"^ls2^"=>"^ls3)], the symmetric difference / disjunctive union of [ls1] and [ls2]'s
      pseudo-labels should be equal to [ls3] pseudo-labels.

      Currently, we support two variants of the [einsum] syntax: either all the axes are provided,
      or all input, output axes are provided but none of the batch axes.
      Note: The "right-hand-side" is on the left! I.e. the syntax is "rhs=>lhs", "rhs1;rhs2=>lhs". *)
[@@deriving sexp, equal]

type transpose_type =
  | Transpose  (** Swaps inputs and outputs of a shape, preserves batch axes. *)
  | Pointwise_un  (** Preserves the shape. *)
  | Permute of string
      (** [Permute (ls1^"=>"^ls2)] is a variant of the [einsum] syntax [Einsum (ls1^";"^ls1^"=>"^ls2)].
      Note: The "right-hand-side" is on the left! I.e. the syntax is "rhs=>lhs", "rhs1;rhs2=>lhs". *)
  | Batch_slice of Arrayjit.Indexing.static_symbol  (** Removes the leftmost batch axis. *)
[@@deriving equal, sexp]

(** Parses a labels specification.

  * If [spec] contains any of: [' '; ','; '('; ')'], these characters are used as label separators.
    Otherwise, every character is a label.
  * If [spec] does not contain ["|"] nor ["->"], each label is of the kind [Output].
  * If [spec] doesn't contain ["|"], labels to the left of ["->"] are [Input] and to the right [Output].
  * Labels to the left of ["|"] are [Batch], and between ["|"] and ["->"] are [Input].

    The labels ["..ident.."], ["..."] (where [ident] does not contain any of the special characters)
    are only allowed at the first axis of a kind (i.e. last from-end).
    They are used to enable broadcasting for the axis kind in the einsum-related shape inference
    (like the ellipsis ["..."] in [numpy.einsum]), and are translated to row variables.
    The ellipsis ["..."] is context dependent: in the batch row it is the same as ["..batch.."],
    in the input row the same as ["..input.."], in the output row the same as ["..output.."].
    When the same row variable is used in multiple rows, the corresponding broadcasted axes are matched
    pointwise in the resulting operation.

    The label ["_"] is a place-holder: it is not output to the resulting map but aligns the axes
    of other labels. *)
let axis_labels_of_spec spec : parsed_axis_labels =
  let check_dot ~kind s =
    (* TODO: complain if the row variable specification contains special characters, e.g. [' '; ',']. *)
    if String.is_prefix s ~prefix:"..." then (Some kind, String.drop_prefix s 3)
    else if String.is_prefix s ~prefix:".." then
      let row_var_spec, s =
        match String.substr_index ~pos:2 s ~pattern:".." with
        | None -> invalid_arg "Shape.axis_labels_of_spec: unfinished row variable specification <..>"
        | Some end_pos -> (String.sub s ~pos:2 ~len:(end_pos - 2), String.drop_prefix s (end_pos + 2))
      in
      (Some row_var_spec, String.drop_prefix s 3)
    else (None, s)
  in
  let parse ~kind spec in_axes =
    let bcast, spec = check_dot ~kind @@ String.strip spec in
    ( bcast,
      let on = [ ' '; ','; '('; ')'; '\t'; '\r'; '\n' ] in
      let parse_label labels_num from_start s =
        let key = AxisKey.{ in_axes; from_end = labels_num - from_start } in
        if String.equal s "_" then None
        else try Some (key, Either.Second (Int.of_string s)) with _ -> Some (key, First s)
      in
      if List.exists ~f:(String.contains spec) on || String.for_all spec ~f:Char.is_digit then
        let labels = String.split_on_chars spec ~on |> List.filter ~f:(fun s -> not @@ String.is_empty s) in
        let labels_num = List.length labels in
        (labels_num, List.filter_mapi labels ~f:(parse_label labels_num) |> Map.of_alist_exn (module AxisKey))
      else
        let labels_num = String.length spec in
        ( labels_num,
          String.to_list spec |> List.map ~f:String.of_char
          |> List.filter_mapi ~f:(parse_label labels_num)
          |> Map.of_alist_exn (module AxisKey) ) )
  in
  let batch_spec, spec =
    match String.substr_index spec ~pattern:"|" with
    | Some end_bch ->
        ( String.sub ~pos:0 ~len:end_bch spec,
          String.sub ~pos:(end_bch + 1) ~len:(String.length spec - end_bch - 1) spec )
    | None -> ("", spec)
  in
  let input_spec, output_spec =
    match String.substr_index spec ~pattern:"->" with
    | Some end_inp ->
        ( String.sub ~pos:0 ~len:end_inp spec,
          String.sub ~pos:(end_inp + 2) ~len:(String.length spec - end_inp - 2) spec )
    | None -> ("", spec)
  in
  let bcast_batch, (given_batch, batch_labels) = parse ~kind:"batch" batch_spec `Batch in
  let bcast_input, (given_input, input_labels) = parse ~kind:"input" input_spec `Input in
  let bcast_output, (given_output, output_labels) = parse ~kind:"output" output_spec `Output in
  let labels =
    match Map.append ~lower_part:input_labels ~upper_part:output_labels with
    | `Ok m -> (
        match Map.append ~lower_part:batch_labels ~upper_part:m with `Ok r -> r | _ -> assert false)
    | _ -> assert false
  in
  { bcast_batch; bcast_input; bcast_output; given_batch; given_input; given_output; labels }

let einsum_of_spec spec =
  let rhs_spec, lhs_spec =
    match String.substr_index spec ~pattern:"=>" with
    | Some endp ->
        ( String.sub ~pos:0 ~len:endp spec,
          String.sub ~pos:(endp + 2) ~len:(String.length spec - endp - 2) spec )
    | None -> ("", spec)
  in
  let lhs_spec = String.strip lhs_spec in
  let rhs_spec = String.strip rhs_spec in
  if String.is_empty lhs_spec then invalid_arg ("einsum_of_spec: missing the result spec in " ^ spec);
  if String.is_empty rhs_spec then invalid_arg ("einsum_of_spec: missing the argument spec in " ^ spec);
  let rhs1_spec, rhs2_spec =
    match String.substr_index rhs_spec ~pattern:";" with
    | Some endp ->
        ( String.sub ~pos:0 ~len:endp rhs_spec,
          String.sub ~pos:(endp + 1) ~len:(String.length rhs_spec - endp - 1) rhs_spec )
    | None -> (rhs_spec, "")
  in
  let rhs1_spec = String.strip rhs1_spec in
  let rhs2_spec = String.strip rhs2_spec in
  let lhs_ls = axis_labels_of_spec lhs_spec in
  let rhs1_ls = axis_labels_of_spec rhs1_spec in
  if String.is_empty rhs2_spec then (rhs1_ls, None, lhs_ls)
  else (rhs1_ls, Some (axis_labels_of_spec rhs2_spec), lhs_ls)

(** How to propagate shape updates and do the last update of [Tensor.t.shape] when finalizing the tensor.
    Axes are broadcast-expanded on a bottom-up update to fit the incoming shape. *)
type logic =
  | Broadcast of compose_type * t * t
      (** Matches the shapes for a binary operation, allowing for broadcasting e.g. an axis of dimension 1
      does not conflict with a matching axis of a greater dimension.

      For [Broadcast (Einsum (ls1, ls2, ls3), s1, s2)], the labels of [s1] and [s2] must match according
      to the [ls1], [ls2] lineup, and the resulting shape inherits the labels according to the [ls3] lineup.
  *)
  | Transpose of transpose_type * t
      (** Permutes the axes of a shape. One case of [Transpose] is to swap inputs with outputs of [s1],
      hence the name. *)
  | Terminal of Arrayjit.Ops.init_op
      (** Extracts any available shape information from the initialization. E.g.
      for [File_mapped fn], opens the file [fn] to check its length. *)
[@@deriving equal, sexp]

let logic_to_spec = function
  | Broadcast (Pointwise_bin, _, _) | Transpose (Pointwise_un, _) -> "."
  | Broadcast (Compose, _, _) -> "@"
  | Broadcast (Einsum spec, _, _) | Transpose (Permute spec, _) -> spec
  | Transpose (Transpose, _) -> "T"
  | Transpose (Batch_slice _, _) -> "@|"
  | Terminal _ -> "<terminal>"

module Debug_runtime = Utils.Debug_PrintBox ()

type update_step = { shape : t; logic : logic } [@@deriving sexp]
(** Data required for a shape inference update step. Ideally, an update should be performed at least twice,
    the second time after all the other relevant updates have been performed for the first time.
    In OCANNL, this is achieved by performing updates both as the tensors are constructed, and via
    lazy callbacks as the corresponding [Arrayjit.Indexing] dimensions and projections are first accessed. *)

type Row.error_trace += Shape_mismatch of t list

let with_error_trace = ref true

(** Converts an axes-keyed map into three arrays of values: batch axes, input axes, output axes.
    If the map is incomplete, the result might be invalid: gaps in the array are filled with an arbitrary
    one of the provided values. *)
let axis_map_to_dims_bio (type a) ?(default : a option) (idcs : a axis_map) =
  if Map.is_empty idcs then ([||], [||], [||])
  else
    let witness = match default with Some witness -> witness | None -> snd @@ Map.min_elt_exn idcs in
    let bch_axes, other =
      Map.partition_mapi idcs ~f:(fun ~key:{ in_axes; _ } ~data ->
          if Row.is_batch in_axes then Either.First data else Either.Second data)
    in
    let inp_axes, out_axes =
      Map.partition_mapi other ~f:(fun ~key:{ in_axes; _ } ~data ->
          if Row.is_input in_axes then Either.First data else Either.Second data)
    in
    let bch_axes = Map.to_alist bch_axes |> List.map ~f:(fun ({ from_end = i; _ }, v) -> (i, v)) in
    let bch_size = List.fold bch_axes ~init:0 ~f:(fun accu (i, _) -> max i accu) in
    let bch = Array.create ~len:bch_size witness in
    List.iter bch_axes ~f:(fun (i, v) -> bch.(bch_size - i) <- v);
    let inp_axes = Map.to_alist inp_axes |> List.map ~f:(fun ({ from_end = i; _ }, v) -> (i, v)) in
    let inp_size = List.fold inp_axes ~init:0 ~f:(fun accu (i, _) -> max i accu) in
    let inp = Array.create ~len:inp_size witness in
    List.iter inp_axes ~f:(fun (i, v) -> inp.(inp_size - i) <- v);
    let out_axes = Map.to_alist out_axes |> List.map ~f:(fun ({ from_end = i; _ }, v) -> (i, v)) in
    let out_size = List.fold out_axes ~init:0 ~f:(fun accu (i, _) -> max i accu) in
    let out = Array.create ~len:out_size witness in
    List.iter out_axes ~f:(fun (i, v) -> out.(out_size - i) <- v);
    (bch, inp, out)

(** Converts an axes-keyed map into an array of values using the [force_to_dims] semantics of axes.
    If the map is incomplete and the [~default] is not given, the result might be invalid: gaps in
    the array are filled with an arbitrary one of the provided values. *)
let axis_map_to_dims_index (type a) ?(default : a option) (idcs : a axis_map) : a array =
  let bch, inp, out = axis_map_to_dims_bio ?default idcs in
  Array.concat [ bch; out; inp ]

let axes_spec_to_dims_bio ~sh_id ~row_var_env ~f labels =
  let b_dims, i_dims, o_dims = axis_map_to_dims_bio labels.labels in
  let vars = Hashtbl.create (module String) in
  let to_dim kind = Array.(Fn.compose to_list @@ map ~f:(f kind vars)) in
  let to_bcast v =
    Option.value_map v ~default:Row.Broadcastable ~f:(fun vname ->
        Hashtbl.find_or_add row_var_env vname ~default:(fun () -> Row.Row_var (Row.get_row_var ())))
  in
  (* let dims, bcast =Option.value v ~default:(Row.Broadcastable, in *)
  let batch =
    {
      Row.dims = to_dim `Batch b_dims;
      constr = Unconstrained;
      bcast = to_bcast labels.bcast_batch;
      id = Row.row_id ~sh_id ~kind:`Batch;
    }
  in
  let input =
    {
      Row.dims = to_dim `Input i_dims;
      constr = Unconstrained;
      bcast = to_bcast labels.bcast_input;
      id = Row.row_id ~sh_id ~kind:`Input;
    }
  in
  let output =
    {
      Row.dims = to_dim `Output o_dims;
      constr = Unconstrained;
      bcast = to_bcast labels.bcast_output;
      id = Row.row_id ~sh_id ~kind:`Output;
    }
  in
  (batch, input, output)

let einsum_slot_spec_to_dims_bio ~generative ~sh_id ~row_var_env labels =
  let equal = Row.equal_kind in
  let proj_env_update = ref @@ Row.dim_map_empty in
  let f kind vars = function
    | Either.First label ->
        Row.Var (Hashtbl.find_or_add vars label ~default:(fun () -> Row.get_var ~label ()))
    | Second 0 when Option.value ~default:false @@ List.Assoc.find generative ~equal kind ->
        Row.get_dim ~d:1 ()
    | Second i ->
        let var = Row.get_var () in
        proj_env_update := Map.add_exn !proj_env_update ~key:var ~data:(Arrayjit.Indexing.Fixed_idx i);
        Var var
  in
  let result = axes_spec_to_dims_bio ~f ~row_var_env ~sh_id labels in
  (!proj_env_update, result)

type proj_axis_env = Arrayjit.Indexing.axis_index Row.dim_map [@@deriving sexp]

let get_inequalities ({ shape = cur_sh; logic } : update_step) : proj_axis_env * Row.inequality list =
  let open Row in
  let dim_assoc_eqs assoc =
    List.Assoc.sort_and_group assoc ~compare:String.compare
    |> List.concat_map ~f:(function
         | _, [] -> assert false
         | _, d1 :: ds -> List.map ds ~f:(fun d2 -> Dim_eq { d1; d2 }))
  in
  let generative =
    [
      (`Batch, List.is_empty cur_sh.batch.dims);
      (`Input, List.is_empty cur_sh.input.dims);
      (`Output, List.is_empty cur_sh.output.dims);
    ]
  in
  match logic with
  | Terminal (Range_over_offsets | Standard_uniform | Constant_fill { strict = false; _ }) ->
      (Row.dim_map_empty, [])
  | Terminal (Constant_fill { values; strict = true }) ->
      let len = Array.length values in
      let io_dims =
        try List.map ~f:dim_to_int_exn @@ cur_sh.output.dims @ cur_sh.input.dims
        with Invalid_argument _ ->
          raise
          @@ Shape_error
               ( "unify_shapes Constant_fill strict: non-batch dimensions must be known",
                 [ Shape_mismatch [ cur_sh ] ] )
      in
      let batch_elems = len / abs (List.fold ~init:1 ~f:( * ) io_dims) in
      let b_row =
        {
          dims = [];
          constr = Total_elems batch_elems;
          bcast = Row_var (get_row_var ());
          id = row_id ~sh_id:cur_sh.id ~kind:`Batch;
        }
      in
      (dim_map_empty, [ Row_eq { r1 = b_row; r2 = cur_sh.batch } ])
  | Terminal (File_mapped (filename, prec)) ->
      let fd = Unix.openfile filename [ Unix.O_RDONLY ] 0o640 in
      let len = Unix.lseek fd 0 Unix.SEEK_END / Arrayjit.Ops.prec_in_bytes prec in
      Unix.close fd;
      let io_dims =
        try List.map ~f:dim_to_int_exn @@ cur_sh.output.dims @ cur_sh.input.dims
        with Invalid_argument _ ->
          raise
          @@ Shape_error
               ( "unify_shapes Constant_fill strict: non-batch dimensions must be known",
                 [ Shape_mismatch [ cur_sh ] ] )
      in
      let batch_elems = len / abs (List.fold ~init:1 ~f:( * ) io_dims) in
      let b_row =
        {
          dims = [];
          constr = Total_elems batch_elems;
          bcast = Row_var (get_row_var ());
          id = row_id ~sh_id:cur_sh.id ~kind:`Batch;
        }
      in
      (Row.dim_map_empty, [ Row_eq { r1 = b_row; r2 = cur_sh.batch } ])
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
  | Transpose (Batch_slice { static_range; static_symbol }, sh) ->
      if is_row_var sh.batch.bcast && is_row_var cur_sh.batch.bcast then
        (* Wait for more information *)
        (Row.dim_map_empty, [])
      else
        let slice_v = get_var () in
        let slice_var = Var slice_v in
        (* Note: at one point this code worked without marking [slice_var] as a solved projection.
            I don't know why this has not led to the axis being erroneusly expanded in the product space. *)
        let proj_axis_env =
          Map.add_exn Row.dim_map_empty ~key:slice_v ~data:(Arrayjit.Indexing.Iterator static_symbol)
        in
        (* Expand a batch row instead of reducing one because even if the dimensions are known,
           the equations are also needed for projection inference. *)
        let num_dims sh = List.length sh.batch.dims in
        if not @@ is_row_var cur_sh.batch.bcast then
          let expanded_batch =
            {
              dims = slice_var :: cur_sh.batch.dims;
              constr = Unconstrained;
              bcast = cur_sh.batch.bcast;
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
        else if not @@ is_row_var sh.batch.bcast then
          if num_dims cur_sh < num_dims sh then
            let matching_batch =
              {
                dims =
                  List.init (num_dims sh - num_dims cur_sh - 1) ~f:(fun _ -> Var (get_var ()))
                  @ cur_sh.batch.dims;
                constr = Unconstrained;
                bcast = Broadcastable;
                id = Row.row_id ~sh_id:cur_sh.id ~kind:`Batch;
              }
            in
            let expanded_batch =
              {
                dims = slice_var :: matching_batch.dims;
                constr = Unconstrained;
                bcast = Broadcastable;
                id = Row.row_id ~sh_id:cur_sh.id ~kind:`Batch;
              }
            in
            ( proj_axis_env,
              (Option.to_list static_range
              |> List.map ~f:(fun range -> Dim_eq { d1 = get_dim ~d:range (); d2 = slice_var }))
              @ [
                  Row_eq { r1 = matching_batch; r2 = cur_sh.batch };
                  Row_eq { r1 = expanded_batch; r2 = sh.batch };
                  Row_eq { r1 = cur_sh.input; r2 = sh.input };
                  Row_eq { r1 = cur_sh.output; r2 = sh.output };
                ] )
          else
            raise
            @@ Shape_error
                 ("Batch slice: the sliced tensor has too few batch axes", [ Shape_mismatch [ cur_sh; sh ] ])
        else
          (* Unfortunately, we cannot proceed if both rows have row variables --
             it would make it hard to connect the shapes once they are inferred with proj_axis_env. *)
          raise
          @@ Shape_error
               ( "Batch slice: inference with underspecified batch axes not supported yet",
                 [ Shape_mismatch [ cur_sh; sh ] ] )
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
      let proj_env_rhs, (b_rhs, i_rhs, o_rhs) =
        einsum_slot_spec_to_dims_bio ~generative:[] ~sh_id:sh.id ~row_var_env ls_rhs
      in
      let proj_env_lhs, (b_lhs, i_lhs, o_lhs) =
        einsum_slot_spec_to_dims_bio ~generative ~sh_id:cur_sh.id ~row_var_env ls_lhs
      in
      let label_groups = List.concat_map ~f:dims_label_assoc [ b_lhs; i_lhs; o_lhs; b_rhs; i_rhs; o_rhs ] in
      let proj_env =
        let combine ~key:_ _ _ = assert false in
        Map.merge_skewed ~combine proj_env_rhs proj_env_lhs
      in
      (* Forget the old proj_env as it is not relevant after a propagate_shapes call completes. *)
      ( proj_env,
        Row_ineq { cur = cur_sh.batch; subr = b_lhs }
        :: Row_ineq { cur = b_rhs; subr = sh.batch }
        :: Row_ineq { cur = cur_sh.input; subr = i_lhs }
        :: Row_ineq { cur = i_rhs; subr = sh.input }
        :: Row_ineq { cur = cur_sh.output; subr = o_lhs }
        :: Row_ineq { cur = o_rhs; subr = sh.output }
        :: dim_assoc_eqs label_groups )
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
      let proj_env_rhs1, (b_rhs1, i_rhs1, o_rhs1) =
        einsum_slot_spec_to_dims_bio ~generative:[] ~sh_id:sh1.id ~row_var_env ls_rhs1
      in
      let proj_env_rhs2, (b_rhs2, i_rhs2, o_rhs2) =
        einsum_slot_spec_to_dims_bio ~generative:[] ~sh_id:sh2.id ~row_var_env ls_rhs2
      in
      let proj_env_lhs, (b_lhs, i_lhs, o_lhs) =
        einsum_slot_spec_to_dims_bio ~generative ~sh_id:cur_sh.id ~row_var_env ls_lhs
      in
      let label_groups =
        List.concat_map ~f:dims_label_assoc
          [ b_lhs; i_lhs; o_lhs; b_rhs1; i_rhs1; o_rhs1; b_rhs2; i_rhs2; o_rhs2 ]
      in
      let proj_env =
        let combine ~key:_ _ _ = assert false in
        Map.merge_skewed ~combine proj_env_rhs1 @@ Map.merge_skewed ~combine proj_env_rhs2 proj_env_lhs
      in
      (* Forget the old proj_env as it is not relevant after a propagate_shapes call completes. *)
      ( proj_env,
        Row_ineq { cur = cur_sh.batch; subr = b_lhs }
        :: Row_ineq { cur = b_rhs1; subr = sh1.batch }
        :: Row_ineq { cur = b_rhs2; subr = sh2.batch }
        :: Row_ineq { cur = cur_sh.input; subr = i_lhs }
        :: Row_ineq { cur = i_rhs1; subr = sh1.input }
        :: Row_ineq { cur = i_rhs2; subr = sh2.input }
        :: Row_ineq { cur = cur_sh.output; subr = o_lhs }
        :: Row_ineq { cur = o_rhs1; subr = sh1.output }
        :: Row_ineq { cur = o_rhs2; subr = sh2.output }
        :: dim_assoc_eqs label_groups )

let indices_bio sh (type v) (arr : v array) =
  let n_batch = List.length sh.batch.dims in
  let batch : v Array.t = Array.sub arr ~pos:0 ~len:n_batch in
  let n_input = List.length sh.input.dims in
  let input = Array.sub arr ~pos:n_batch ~len:n_input in
  let n_output = List.length sh.output.dims in
  let output = Array.sub arr ~pos:(n_batch + n_input) ~len:n_output in
  (batch, input, output)

let state = ref Row.empty_env
let second_stage_inference = ref []

let deep_copy_update_step update_step =
  let upd sh = { sh with id = sh.id } in
  {
    shape = upd update_step.shape;
    logic =
      (match update_step.logic with
      | Terminal l -> Terminal l
      | Transpose (l, sh1) -> Transpose (l, upd sh1)
      | Broadcast (l, sh1, sh2) -> Broadcast (l, upd sh1, upd sh2));
  }

let iter_shapes update_step ~f =
  f update_step.shape;
  match update_step.logic with
  | Terminal _ -> ()
  | Transpose (_, sh1) -> f sh1
  | Broadcast (_, sh1, sh2) ->
      f sh1;
      f sh2

let apply_env update_step env =
  let f sh =
    sh.batch <- Row.subst_row env sh.batch;
    sh.input <- Row.subst_row env sh.input;
    sh.output <- Row.subst_row env sh.output
  in
  iter_shapes update_step ~f

let%debug_sexp propagate_shapes ?(is_complete = false) (update_step : update_step) : unit =
  if not @@ List.mem ~equal:phys_equal !second_stage_inference update_step then
    second_stage_inference := update_step :: !second_stage_inference;
  (* Allow the derivation of constraints to depend on the shapes (currently, only Batch_slice does). *)
  apply_env update_step !state;
  let _debug_initial : update_step = deep_copy_update_step update_step in
  let _, ineqs = get_inequalities update_step in
  let env = Row.solve_inequalities ~is_complete ineqs !state in
  (* A slight optimization: [finish_inference] will update the shapes at the end of inference. *)
  if (not is_complete) || Utils.settings.with_debug then apply_env update_step env;
  let _debug_result : update_step = deep_copy_update_step update_step in
  (* Debug_runtime.no_debug_if
     (equal _debug_initial.shape _debug_result.shape && equal_logic _debug_initial.logic _debug_result.logic); *)
  state := env

let finish_inference () =
  (* FIXME: do we need an extra round of inference? *)
  (* List.iter !second_stage_inference ~f:(propagate_shapes ~is_complete:true); *)
  List.iter !second_stage_inference ~f:(propagate_shapes ~is_complete:true);
  List.iter !second_stage_inference ~f:(Fn.flip apply_env !state);
  second_stage_inference := []

let row_to_dims row =
  let open Row in
  let row = Row.subst_row !state row in
  let f = function
    | Dim { d; _ } -> d
    | Var _ as dim ->
        raise @@ Shape_error ("Not enough shape information: unresolved variable", [ Dim_mismatch [ dim ] ])
  in
  match row with
  | { bcast = Row_var _; dims; _ } ->
      (* FIXME: DEBUG: *)
      (* raise @@ Shape_error ("Not enough shape information: unresolved row variable", [ Row_mismatch [ row ] ]) *)
      Array.of_list_map dims ~f
  | { dims; constr = _; bcast = Broadcastable; id = _ } -> Array.of_list_map dims ~f

(** Uses the matrix convention of putting the input axes last.
    Note: [force_to_dims] is "destructive": it closes shapes that remain incomplete after inference. *)
let to_dims (sh : t) : int array =
  try Array.concat_map ~f:row_to_dims [| sh.batch; sh.output; sh.input |]
  with Row.Shape_error (s, trace) -> raise @@ Row.Shape_error (s, Shape_mismatch [ sh ] :: trace)

(** Uses the matrix convention of putting the input axes last. *)
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

open Arrayjit.Indexing

let fresh_proj_ids update =
  let fresh_shape (sh : t) =
    sh.batch <- Row.fresh_row_proj sh.batch;
    sh.input <- Row.fresh_row_proj sh.input;
    sh.output <- Row.fresh_row_proj sh.output
  in
  fresh_shape update.shape;
  match update.logic with
  | Terminal _ -> ()
  | Transpose (_, sh) -> fresh_shape sh
  | Broadcast (_, sh1, sh2) ->
      fresh_shape sh1;
      fresh_shape sh2

(** Computes the indexing into subtensors given the shape information of a tensor. 
    [derive_projections] should only be invoked when the shapes are fully inferred already! *)
let derive_projections (update_step : update_step) : projections =
  (* FIXME: why is apply_env needed? Should be done already by finish_inference. *)
  (* apply_env update_step !state; *)
  fresh_proj_ids update_step;
  let proj_axis_env, ineqs = get_inequalities update_step in
  (* We need to solve the equations/inequalities one last time because of fresh row variables
     potentially generated by [get_inequalities]. Since the variables in the shapes must be substituted-out
     at this point, using the global state instead of empty env below would not change anything,
     but in principle we want to only find a local solution to not contaminate projections across operations. *)
  let local_env = Row.solve_inequalities ~is_complete:true ineqs Row.empty_env in
  let proj_eqs = Row.get_proj_equations ineqs proj_axis_env local_env in
  let proj_env = Row.solve_proj_equations proj_eqs in
  let dims_of (sh : t) = sh.batch.dims @ sh.output.dims @ sh.input.dims in
  let lhs = update_step.shape in
  let rhs =
    match update_step.logic with
    | Terminal _ -> []
    | Transpose (_, sh) -> [ sh ]
    | Broadcast (_, sh1, sh2) -> [ sh1; sh2 ]
  in
  let lhs_dims = to_dims lhs in
  let rhs_dims = Array.of_list_map ~f:to_dims rhs in
  let all_dims = List.concat_map ~f:dims_of @@ (lhs :: rhs) in
  (* Note: the ordering will affect performance of naive backends. *)
  let all_product_projs =
    Utils.unique_keep_first ~equal:(fun (p, _) (q, _) -> p = q)
    @@ List.filter_map all_dims ~f:(Row.get_product_proj proj_env)
  in
  let product_space = Array.of_list_map all_product_projs ~f:snd in
  let product_iterators =
    Array.of_list_map all_product_projs ~f:(fun (p, _) -> Row.proj_to_iterator proj_env p)
  in
  let indices_of_sh (sh : t) =
    Array.of_list_map ~f:(Row.get_proj_index proj_env)
    @@ List.concat [ sh.batch.dims; sh.output.dims; sh.input.dims ]
  in
  try
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
          trace = [ ("derive_projections", unique_debug_id ()) ];
        };
    }
  with Row.Shape_error (s, trace) -> raise @@ Row.Shape_error (s, Shape_mismatch (lhs :: rhs) :: trace)

let backprop_ith_arg ~from_1 projections =
  let project_lhs = projections.project_rhs.(from_1 - 1) in
  let project_rhs = Array.copy projections.project_rhs in
  project_rhs.(from_1 - 1) <- projections.project_lhs;
  let lhs_dims = projections.rhs_dims.(from_1 - 1) in
  let rhs_dims = Array.copy projections.rhs_dims in
  rhs_dims.(from_1 - 1) <- projections.lhs_dims;
  {
    product_space = projections.product_space;
    product_iterators = projections.product_iterators;
    lhs_dims;
    rhs_dims;
    project_lhs;
    project_rhs;
    debug_info =
      {
        projections.debug_info with
        trace =
          ("backprop_ith_arg " ^ Int.to_string from_1, unique_debug_id ()) :: projections.debug_info.trace;
      };
  }

(** *** Shape builders *** *)

let make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes
    ?(deduced = Not_constrained) ~debug_name ~id () =
  let open Row in
  let make_dims kind ds =
    {
      dims = List.map ~f:(fun d -> get_dim ~d ()) ds;
      constr = Unconstrained;
      bcast = Broadcastable;
      id = row_id ~sh_id:id ~kind;
    }
  in
  let make_axes kind ds =
    {
      dims = List.map ~f:(fun (label, d) -> get_dim ~d ~label ()) ds;
      constr = Unconstrained;
      bcast = Broadcastable;
      id = row_id ~sh_id:id ~kind;
    }
  in
  let make_unknown kind =
    { dims = []; constr = Unconstrained; bcast = Row_var (get_row_var ()); id = row_id ~sh_id:id ~kind }
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
  let result = { input; output; batch; id; debug_name } in
  (match deduced with
  | Not_constrained -> ()
  | Input_equals_output -> (
      try
        let more_ineqs, env = Row.unify_row (input, output) !state in
        assert (List.is_empty more_ineqs);
        state := env
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ("Input_equals_output / " ^ s, Shape_mismatch [ result ] :: trace)));
  result

let shape_spec_to_dims_bio labels =
  let f _kind vars = function
    | Either.First s when String.contains s '=' -> (
        let label, dim =
          match String.split s ~on:'=' with
          | [ l; d ] -> (l, d)
          | _ -> invalid_arg "shape_spec_to_dims_bio: too many '='"
        in
        try Row.get_dim ~d:(Int.of_string dim) ~label ()
        with _ -> invalid_arg "shape_spec_to_dims_bio: int expected after '='")
    | First label -> Var (Hashtbl.find_or_add vars label ~default:(fun () -> Row.get_var ~label ()))
    | Second d -> Row.get_dim ~d ()
  in
  let row_var_env = Hashtbl.create (module String) in
  axes_spec_to_dims_bio ~row_var_env ~f labels

let of_spec ?(deduced = Not_constrained) ~debug_name ~id spec =
  let batch, input, output = shape_spec_to_dims_bio ~sh_id:id @@ axis_labels_of_spec spec in
  let result = { input; output; batch; id; debug_name } in
  (match deduced with
  | Not_constrained -> ()
  | Input_equals_output -> (
      try
        let more_ineqs, env = Row.unify_row (input, output) !state in
        assert (List.is_empty more_ineqs);
        state := env
      with Row.Shape_error (s, trace) when !with_error_trace ->
        raise @@ Row.Shape_error ("of spec / " ^ s, Shape_mismatch [ result ] :: trace)));
  result

let to_string_hum ?(style = `Axis_size) (sh : t) =
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
           | `Only_labels | `Axis_size -> Row.dim_to_string style d
           | `Axis_number_and_size -> Int.to_string num ^ ":" ^ Row.dim_to_string style d)
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
    Array.of_list_rev_mapi sh.batch.dims ~f:(fun i _ -> AxisKey.{ in_axes = `Batch; from_end = i + 1 })
  in
  let i_dims =
    Array.of_list_rev_mapi sh.input.dims ~f:(fun i _ -> AxisKey.{ in_axes = `Input; from_end = i + 1 })
  in
  let o_dims =
    Array.of_list_rev_mapi sh.output.dims ~f:(fun i _ -> AxisKey.{ in_axes = `Output; from_end = i + 1 })
  in
  let idcs = Array.concat [ i_dims; o_dims; b_dims ] in
  Array.rev_inplace idcs;
  Map.of_alist_exn (module AxisKey) @@ Array.to_list @@ Array.mapi idcs ~f:(fun i key -> (key, i))

let default_display_indices sh =
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
           { in_axes = `Input; from_end = 1 };
           { in_axes = `Output; from_end = 1 };
           { in_axes = `Input; from_end = 2 };
           { in_axes = `Output; from_end = 2 };
           (if num_input_axes > 1 then { in_axes = `Batch; from_end = 1 }
            else { in_axes = `Output; from_end = 3 });
           { in_axes = `Batch; from_end = 1 };
           { in_axes = `Batch; from_end = 2 };
           { in_axes = `Input; from_end = 3 };
           { in_axes = `Output; from_end = 3 };
           { in_axes = `Input; from_end = 4 };
           { in_axes = `Output; from_end = 4 };
           { in_axes = `Input; from_end = 5 };
           { in_axes = `Output; from_end = 5 };
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

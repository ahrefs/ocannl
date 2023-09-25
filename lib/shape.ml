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
    type kind = Batch | Input | Output [@@deriving equal, compare, sexp, variants]

    type t = {
      in_axes : kind;
      from_end : int;
          (** Axes are indexed from the end, to avoid reindexing when broadcasting; starting with [1]. *)
    }
    [@@deriving equal, compare, sexp]

    let to_string key =
      (match key.in_axes with Batch -> "bch" | Input -> "inp" | Output -> "out")
      ^ Int.to_string key.from_end
  end

  include T
  include Comparator.Make (T)
end

type 'a axis_map = 'a Map.M(AxisKey).t [@@deriving compare, sexp]

type parsed_axis_labels = {
  bcast_batch : bool;
  bcast_input : bool;
  bcast_output : bool;
  given_batch : int;
  given_input : int;
  given_output : int;
  labels : (string, int) Either.t axis_map;
}
[@@deriving compare, sexp, fields]
(** The labels are strings assigned to [AxisKey] axes. Moreover the [bcast_] fields represent whether
    additional leading axes are allowed (corresponding to the dot-ellipsis syntax for broadcasting).
    The [given_] fields count the number of specified axes of the corresponding kind in [labels]. *)

let bcast_of_kind = function
  | AxisKey.Batch -> bcast_batch
  | AxisKey.Input -> bcast_input
  | AxisKey.Output -> bcast_output

let given_of_kind = function
  | AxisKey.Batch -> given_batch
  | AxisKey.Input -> given_input
  | AxisKey.Output -> given_output

type dim_var = { id : int; mutable label : string option [@compare.ignore] [@equal.ignore] [@hash.ignore] }
[@@deriving equal, hash, compare, sexp]

module Dim_var = struct
  type t = dim_var = {
    id : int;
    mutable label : string option; [@compare.ignore] [@equal.ignore] [@hash.ignore]
  }
  [@@deriving equal, hash, compare, sexp]

  include Comparator.Make (struct
    type nonrec t = t

    let compare = compare
    let sexp_of_t = sexp_of_t
  end)
end

(** A single axis in a shape. *)
type dim = Var of dim_var | Dim of { d : int; label : string option; proj_id : int }
[@@deriving equal, hash, compare, sexp, variants]

let uid = ref 0

let get_var ?label () =
  Int.incr uid;
  { id = !uid; label }

let get_dim ~d ?label () =
  Int.incr uid;
  Dim { d; proj_id = !uid; label }

(** A row specifies how axes of a single kind in a shape (the shape-kind) can adapt to other shapes. *)
type row =
  | Row_var of int  (** The shape-kind can be inferred to have more axes. *)
  | Broadcastable  (** The shape does not have more axes of this kind, but is "polymorphic". *)
  | Fixed  (** The shape fails in contexts expecting more axes of this kind -- it "stops broadcast". *)
[@@deriving equal, hash, compare, sexp, variants]

type dims_constraint =
  | Unconstrained
  | Total_elems of int  (** The shape-kind, inclusive of the further row spec, has this many elements. *)
[@@deriving equal, hash, compare, sexp, variants]

let get_row_var () =
  Int.incr uid;
  Row_var !uid

type dims = {
  dims : dim list;
  constr : dims_constraint;
  row : row;
  sh_id : Set.M(Int).t; [@equal.ignore] [@compare.ignore] [@hash.ignore]
}
[@@deriving equal, hash, compare, sexp]

type deduce_within_shape = Not_constrained | Input_equals_output [@@deriving compare, sexp, variants]

type t = {
  mutable batch : dims;
  mutable input : dims;
  mutable output : dims;
  id : int;  (** A node that has the same shape as this shape. *)
}
[@@deriving fields, sexp]
(** The datatype from which the actual Tensor shapes are computed.

    Mutability is sufficient to perform inference, since there is no need for backtracking and
    no explicit unification variables for now. [Unknown] stands for "not yet specified". *)

let dims_of_kind = function AxisKey.Batch -> batch | AxisKey.Input -> input | AxisKey.Output -> output

let map_over_kind ~f kind sh =
  match kind with
  | AxisKey.Batch -> { sh with batch = f sh.batch }
  | AxisKey.Input -> { sh with input = f sh.input }
  | AxisKey.Output -> { sh with output = f sh.output }

let update_kind ~f kind sh =
  match kind with
  | AxisKey.Batch -> sh.batch <- f sh.batch
  | AxisKey.Input -> sh.input <- f sh.input
  | AxisKey.Output -> sh.output <- f sh.output

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
[@@deriving sexp]

(** Parses a labels specification.

  * If [spec] contains any of: [' '; ','; '('; ')'], these characters are used as label separators.
    Otherwise, every character is a label.
  * If [spec] does not contain ["|"] nor ["->"], each label is of the kind [Output].
  * If [spec] doesn't contain ["|"], labels to the left of ["->"] are [Input] and to the right [Output].
  * Labels to the left of ["|"] are [Batch], and between ["|"] and ["->"] are [Input].

    The label ["..."] is only allowed at the first axis of a kind (i.e. last from-end).
    It is used to enable broadcasting for the axis kind in the einsum-related shape inference
    (like the ellipsis ["..."] in [numpy.einsum]).

    The label ["_"] is a place-holder: it is not output to the resulting map but aligns the axes
    of other labels. *)
let axis_labels_of_spec spec : parsed_axis_labels =
  let check_dot s =
    if String.length s > 3 && (Option.is_some @@ String.substr_index ~pos:3 s ~pattern:"...") then
      invalid_arg ("axis_labels_of_spec: dot only allowed at first axis of a kind: " ^ spec)
    else if String.is_prefix s ~prefix:"..." then (true, String.drop_prefix s 3)
    else (false, s)
  in
  let parse spec in_axes =
    let bcast, spec = check_dot @@ String.strip spec in
    ( bcast,
      let on = [ ' '; ','; '('; ')'; '\t'; '\r'; '\n' ] in
      let parse_label labels_num from_start s =
        let key = AxisKey.{ in_axes; from_end = labels_num - from_start } in
        if String.equal s "_" then None
        else try Some (key, Either.Second (Int.of_string s)) with _ -> Some (key, First s)
      in
      if List.exists ~f:(String.contains spec) on then
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
  let bcast_batch, (given_batch, batch_labels) = parse batch_spec Batch in
  let bcast_input, (given_input, input_labels) = parse input_spec Input in
  let bcast_output, (given_output, output_labels) = parse output_spec Output in
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
      (** Extracts any available shape information from the initialization from the initialization. E.g.
      for [File_mapped fn], opens the file [fn] to check its length. *)
[@@deriving sexp]

type update_step = { shape : t; logic : logic } [@@deriving sexp]
(** Data required for a shape inference update step. A step should equilibrate information, passing it both
    top-down and bottom-up. The child should be identifiable within the parent via physical equality
    (allowing that a child fills both slots of a binary parent). *)

type shape_error = Shape_mismatch of t list | Row_mismatch of dims list | Dim_mismatch of dim list
[@@deriving sexp]

exception Shape_error of string * shape_error list [@@deriving sexp]

let with_error_trace = ref true

(** Given a fully-inferred shape, maps axes to their corresponding positions in an index using the
    [force_to_dims] semantics. *)
let axis_keys_to_idcs (sh : t) : int axis_map =
  let b_dims =
    (* Enumerate axes backwards. *)
    Array.of_list_rev_mapi sh.batch.dims ~f:(fun i _ -> AxisKey.{ in_axes = Batch; from_end = i + 1 })
  in
  let i_dims =
    Array.of_list_rev_mapi sh.input.dims ~f:(fun i _ -> AxisKey.{ in_axes = Input; from_end = i + 1 })
  in
  let o_dims =
    Array.of_list_rev_mapi sh.output.dims ~f:(fun i _ -> AxisKey.{ in_axes = Output; from_end = i + 1 })
  in
  let idcs = Array.concat [ i_dims; o_dims; b_dims ] in
  Array.rev_inplace idcs;
  Map.of_alist_exn (module AxisKey) @@ Array.to_list @@ Array.mapi idcs ~f:(fun i key -> (key, i))

(** Converts an axes-keyed map into three arrays of values: batch axes, input axes, output axes.
    If the map is incomplete, the result might be invalid: gaps in the array are filled with an arbitrary
    one of the provided values. *)
let axis_map_to_dims_bio (type a) ?(default : a option) (idcs : a axis_map) =
  if Map.is_empty idcs then ([||], [||], [||])
  else
    let witness = match default with Some witness -> witness | None -> snd @@ Map.min_elt_exn idcs in
    let bch_axes, other =
      Map.partition_mapi idcs ~f:(fun ~key:{ in_axes; _ } ~data ->
          if AxisKey.is_batch in_axes then Either.First data else Either.Second data)
    in
    let inp_axes, out_axes =
      Map.partition_mapi other ~f:(fun ~key:{ in_axes; _ } ~data ->
          if AxisKey.is_input in_axes then Either.First data else Either.Second data)
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

type dim_env = dim Map.M(Dim_var).t [@@deriving sexp]
(** Note: The substituted variables can appear in the substitutions. *)

type row_env = dims Map.M(Int).t [@@deriving sexp]
(** Note: The substituted variables can appear in the substitutions. *)

type environment = { dim_env : dim_env; row_env : row_env; proj_env : int Map.M(Int).t } [@@deriving sexp]
type dim_eq = { d1 : dim; fix1 : bool; d2 : dim; fix2 : bool } [@@deriving sexp, equal, hash, compare]
type dim_eqs = dim_eq list [@@deriving sexp]

type row_eq = { r : dims; subr : dims } [@@deriving sexp, equal]
(** Where applicable, [subr] comes from a subtensor of [r]. *)

type row_eqs = row_eq list [@@deriving sexp, equal]

let rec subst_dim dim_env = function
  | Dim _ as d -> d
  | Var v as default -> Option.value ~default @@ Option.map ~f:(subst_dim dim_env) @@ Map.find dim_env v

let drop_from_end l n = List.rev @@ List.drop (List.rev l) n
let take_from_end l n = List.rev @@ List.take (List.rev l) n

let meet more_constr constr =
  match (more_constr, constr) with
  | Unconstrained, c -> c
  | c, Unconstrained -> c
  | (Total_elems n1 as c), Total_elems n2 when n1 = n2 -> c
  | Total_elems _, Total_elems _ -> raise @@ Shape_error ("Incompatible Total_elems constraints", [])

let dim_to_int_exn = function Dim { d; _ } -> d | Var _ -> invalid_arg "dim_to_int: dim still unknown"

let subst_row row_env ({ dims; constr; row; sh_id } as default) =
  match row with
  | Broadcastable | Fixed -> default
  | Row_var v -> (
      match Map.find row_env v with
      | None -> default
      | Some { dims = more_dims; constr = Unconstrained; row; sh_id = more_sh_id } ->
          { dims = more_dims @ dims; constr; row; sh_id = Set.union more_sh_id sh_id }
      | Some { dims = more_dims; constr = Total_elems m; row; sh_id = more_sh_id } ->
          let more_constr =
            if List.for_all dims ~f:is_dim then
              Total_elems (m * List.fold dims ~init:1 ~f:(fun n d -> n * dim_to_int_exn d))
            else Unconstrained (* Wait for more shape inference. *)
          in
          {
            dims = more_dims @ dims;
            constr = meet more_constr constr;
            row;
            sh_id = Set.union more_sh_id sh_id;
          })

let apply_constraint r env =
  let r = subst_row env.row_env r in
  match r.constr with
  | Unconstrained -> env
  | Total_elems n -> (
      match r.row with
      | Row_var _ -> env (* Wait for more shape inference. *)
      | Fixed | Broadcastable -> (
          let dims = List.map r.dims ~f:(subst_dim env.dim_env) in
          let vars, nonvars = List.partition_tf dims ~f:is_var in
          if List.length vars > 1 then env (* Wait for more shape inference. *)
          else
            let known = List.fold nonvars ~init:1 ~f:(fun n d -> n * dim_to_int_exn d) in
            match vars with
            | [] ->
                if n <> known then (
                  if Utils.settings.with_debug then
                    Stdlib.Format.printf "apply_constraint: shape error env=@ %a\n%!" Sexp.pp_hum
                      (sexp_of_environment env);
                  raise @@ Shape_error ("Total_elems constraint failed", [ Row_mismatch [ r ] ]))
                else env
            | [ Var v ] ->
                let rem = n / known in
                if rem = 0 then (
                  if Utils.settings.with_debug then
                    Stdlib.Format.printf "apply_constraint: shape error env=@ %a\n%!" Sexp.pp_hum
                      (sexp_of_environment env);
                  raise @@ Shape_error ("Total_elems constraint failed", [ Row_mismatch [ r ] ]))
                else { env with dim_env = Map.add_exn env.dim_env ~key:v ~data:(get_dim ~d:rem ()) }
            | _ -> assert false))

let eliminate_broadcastable = function (Row_var _ | Fixed) as d -> d | Broadcastable -> get_row_var ()

let rec unify_dims row_eqs env =
  match row_eqs with
  | [] -> env
  | { r; subr } :: row_eqs when equal_dims r subr -> apply_constraint r env |> unify_dims row_eqs
  | { r = { dims = []; row = Row_var v; _ } as rv; subr = rd as subr } :: row_eqs
  | { r = rd; subr = { dims = []; row = Row_var v; _ } as rv as subr } :: row_eqs -> (
      (* The tensor inherits broadcastability from its subtensors, but not from its use sites. *)
      let rd_is_subtensor = phys_equal rd subr in
      let rd = subst_row env.row_env rd in
      if equal_row rd.row rv.row && not (List.is_empty rd.dims) then (
        if Utils.settings.with_debug then
          Stdlib.Format.printf "unify_dims: occurs check: shape error env=@ %a\n%!" Sexp.pp_hum
            (sexp_of_environment env);
        raise @@ Shape_error ("unify_dims: occurs check: infinite number of axes", [ Row_mismatch [ rv; rd ] ]))
      else if equal_row rv.row rd.row then
        apply_constraint rv env |> apply_constraint rd |> unify_dims row_eqs
      else
        match Map.find env.row_env v with
        | None ->
            let data = if rd_is_subtensor then rd else { rd with row = eliminate_broadcastable rd.row } in
            assert (not @@ equal_row rv.row data.row);
            let row_env = Map.add_exn env.row_env ~key:v ~data in
            apply_constraint rv { env with row_env } |> unify_dims row_eqs
        | Some r' ->
            let row_eq = if rd_is_subtensor then { r = r'; subr = rd } else { r = rd; subr = r' } in
            apply_constraint rv env |> unify_dims (row_eq :: row_eqs))
  | {
      r = { dims = []; constr = constr1; row = Fixed; sh_id };
      subr = { dims = []; constr = constr2; row = Fixed | Broadcastable; sh_id = _ };
    }
    :: row_eqs
  | {
      r = { dims = []; constr = constr1; row = Broadcastable; sh_id = _ };
      subr = { dims = []; constr = constr2; row = Fixed; sh_id };
    }
    :: row_eqs ->
      let constr = meet constr1 constr2 in
      apply_constraint { dims = []; constr; row = Fixed; sh_id } env |> unify_dims row_eqs
  | ({ r = { dims = []; row = Fixed; _ }; subr = _ } as eq) :: _
  | ({ r = _; subr = { dims = []; row = Fixed; _ } } as eq) :: _ ->
      if Utils.settings.with_debug then
        Stdlib.Format.printf "unify_dims: Fixed-mode: shape error env=@ %a\n%!" Sexp.pp_hum
          (sexp_of_environment env);
      raise @@ Shape_error ("unify_dims: Fixed-mode axis number mismatch", [ Row_mismatch [ eq.r; eq.subr ] ])
  | (( { r = { dims = []; row = Broadcastable; _ }; subr = _ }
     | { r = _; subr = { dims = []; row = Broadcastable; _ } } ) as eq)
    :: row_eqs ->
      apply_constraint eq.r env |> apply_constraint eq.subr |> unify_dims row_eqs
  | ({
       r = { dims = _ :: _ as ds1; constr = constr1; row = r1; sh_id = sh_id1 };
       subr = { dims = _ :: _ as ds2; constr = constr2; row = r2; sh_id = sh_id2 };
     } as eq)
    :: row_eqs ->
      let constr = meet constr1 constr2 in
      let len1 = List.length ds1 and len2 = List.length ds2 in
      let suffix = min len1 len2 in
      let dims, row = if len2 > len1 then (ds2, r2) else (ds1, r1) in
      let ds1_suf = take_from_end ds1 suffix in
      let ds2_suf = take_from_end ds2 suffix in
      let dim_eqs =
        List.map2_exn ~f:(fun d1 d2 -> { d1; fix1 = is_fixed r1; d2; fix2 = is_fixed r2 }) ds1_suf ds2_suf
      in
      (try unify_dim dim_eqs env
       with Shape_error (s, trace) when !with_error_trace ->
         raise @@ Shape_error ("dim tail / " ^ s, Row_mismatch [ eq.r; eq.subr ] :: trace))
      |> apply_constraint { dims; constr; row; sh_id = Set.union sh_id1 sh_id2 }
      |> unify_dims
           ({
              r = { dims = drop_from_end ds1 suffix; constr = Unconstrained; row = r1; sh_id = sh_id1 };
              subr = { dims = drop_from_end ds2 suffix; constr = Unconstrained; row = r2; sh_id = sh_id2 };
            }
           :: row_eqs)

and unify_dim (dim_eqs : dim_eq list) (env : environment) : environment =
  match dim_eqs with
  | [] -> env
  | { d1 = Dim { label = Some l1; _ } as d1; d2 = Dim { label = Some l2; _ } as d2; fix1 = _; fix2 = _ } :: _
    when not (String.equal l1 l2) ->
      if Utils.settings.with_debug then
        Stdlib.Format.printf "unify_dim: different labels: shape error env=@ %a\n%!" Sexp.pp_hum
          (sexp_of_environment env);
      raise @@ Shape_error ("unify_dim: different labels", [ Dim_mismatch [ d1; d2 ] ])
  | {
      d1 = Dim { d = d1; label = _; proj_id = pid1 };
      d2 = Dim { d = d2; label = _; proj_id = pid2 };
      fix1 = _;
      fix2 = _;
    }
    :: dim_eqs
    when d1 = d2 ->
      let proj_env = Utils.union_add ~equal:Int.equal env.proj_env pid1 pid2 in
      unify_dim dim_eqs { env with proj_env }
  | { d1 = Dim { d = 1; _ }; d2 = Dim _; fix1 = false; fix2 = _ } :: dim_eqs -> unify_dim dim_eqs env
  | { d1 = Dim _; d2 = Dim { d = 1; _ }; fix1 = _; fix2 = false } :: dim_eqs -> unify_dim dim_eqs env
  | ({ d1 = Var v; d2; fix1; fix2 } | { d2 = Var v; d1 = d2; fix1 = fix2; fix2 = fix1 }) :: dim_eqs ->
      let dim_eqs =
        match Map.find env.dim_env v with None -> dim_eqs | Some d1 -> { d1; d2; fix1; fix2 } :: dim_eqs
      in
      let dim_env = Map.update env.dim_env v ~f:(fun _ -> subst_dim env.dim_env d2) in
      unify_dim dim_eqs { env with dim_env }
  | { d1; d2; fix1 = _; fix2 = _ } :: _ ->
      if Utils.settings.with_debug then
        Stdlib.Format.printf "unify_dim: shape error env=@ %a\n%!" Sexp.pp_hum (sexp_of_environment env);
      raise @@ Shape_error ("unify_dim", [ Dim_mismatch [ d1; d2 ] ])

let axes_spec_to_dims_bio ?b_row ?i_row ?o_row ~f labels =
  let b_dims, i_dims, o_dims = axis_map_to_dims_bio labels.labels in
  let vars = Hashtbl.create (module String) in
  let to_dim kind = Array.(Fn.compose to_list @@ map ~f:(f kind vars)) in
  let upd_row = function None, true -> Some (get_row_var ()) | old, true -> old | _, false -> None in
  let b_row = upd_row (b_row, labels.bcast_batch) in
  let i_row = upd_row (i_row, labels.bcast_input) in
  let o_row = upd_row (o_row, labels.bcast_output) in
  let to_row v = Option.value v ~default:Broadcastable in
  let batch =
    { dims = to_dim AxisKey.Batch b_dims; constr = Unconstrained; row = to_row b_row; sh_id = Utils.no_ints }
  in
  let input =
    { dims = to_dim AxisKey.Input i_dims; constr = Unconstrained; row = to_row i_row; sh_id = Utils.no_ints }
  in
  let output =
    { dims = to_dim AxisKey.Output o_dims; constr = Unconstrained; row = to_row o_row; sh_id = Utils.no_ints }
  in
  (b_row, i_row, o_row, batch, input, output)

let einsum_slot_spec_to_dims_bio ~generative ?b_row ?i_row ?o_row labels =
  let equal = AxisKey.equal_kind in
  let f kind vars = function
    | Either.First label -> Var (Hashtbl.find_or_add vars label ~default:(fun () -> get_var ~label ()))
    | Second 0 when Option.value ~default:false @@ List.Assoc.find generative ~equal kind -> get_dim ~d:1 ()
    | Second _ -> Var (get_var ())
  in
  axes_spec_to_dims_bio ?b_row ?i_row ?o_row ~f labels

let unify_shapes env { shape = cur_sh; logic } =
  let generative =
    AxisKey.
      [
        (Batch, List.is_empty cur_sh.batch.dims);
        (Input, List.is_empty cur_sh.input.dims);
        (Output, List.is_empty cur_sh.output.dims);
      ]
  in
  match logic with
  | Terminal (Range_over_offsets | Standard_uniform | Constant_fill { strict = false; _ }) -> env
  | Terminal (Constant_fill { values; strict = true }) -> (
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
        { dims = []; constr = Total_elems batch_elems; row = get_row_var (); sh_id = Utils.one_int cur_sh.id }
      in
      try unify_dims [ { r = b_row; subr = cur_sh.batch } ] env
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ("Constant_fill / " ^ s, Shape_mismatch [ cur_sh ] :: trace))
  | Terminal (File_mapped (filename, prec)) -> (
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
        { dims = []; constr = Total_elems batch_elems; row = get_row_var (); sh_id = Utils.one_int cur_sh.id }
      in
      try unify_dims [ { r = b_row; subr = cur_sh.batch } ] env
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ("File_mapped / " ^ s, Shape_mismatch [ cur_sh ] :: trace))
  | Transpose (Transpose, sh) -> (
      try
        unify_dims
          [
            { r = cur_sh.batch; subr = sh.batch };
            { r = cur_sh.input; subr = sh.output };
            { r = cur_sh.output; subr = sh.input };
          ]
          env
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ("Transpose / " ^ s, Shape_mismatch [ cur_sh; sh ] :: trace))
  | Transpose (Pointwise_un, sh) -> (
      try
        unify_dims
          [
            { r = cur_sh.batch; subr = sh.batch };
            { r = cur_sh.input; subr = sh.input };
            { r = cur_sh.output; subr = sh.output };
          ]
          env
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ("Pointwise unary / " ^ s, Shape_mismatch [ cur_sh; sh ] :: trace))
  | Broadcast (Compose, sh1, sh2) -> (
      try
        unify_dims
          [
            { r = cur_sh.batch; subr = sh1.batch };
            { r = cur_sh.batch; subr = sh2.batch };
            { r = cur_sh.input; subr = sh2.input };
            { r = cur_sh.output; subr = sh1.output };
            { r = sh1.input; subr = sh2.output };
          ]
          env
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ("Compose / " ^ s, Shape_mismatch [ cur_sh; sh1; sh2 ] :: trace))
  | Broadcast (Pointwise_bin, sh1, sh2) ->
      unify_dims
        [
          { r = cur_sh.batch; subr = sh1.batch };
          { r = cur_sh.batch; subr = sh2.batch };
          { r = cur_sh.input; subr = sh1.input };
          { r = cur_sh.input; subr = sh2.input };
          { r = cur_sh.output; subr = sh1.output };
          { r = cur_sh.output; subr = sh2.output };
        ]
        env
  | Transpose (Batch_slice { static_range; static_symbol }, sh) -> (
      if is_row_var sh.batch.row && is_row_var cur_sh.batch.row then (* Wait for more information *) env
      else
        let range_eq, batch_eq =
          let slice_var = Var (get_var ()) in
          if is_row_var sh.batch.row then
            let expanded_batch =
              {
                dims = slice_var :: cur_sh.batch.dims;
                constr = Unconstrained;
                row = cur_sh.batch.row;
                sh_id = Utils.one_int cur_sh.id;
              }
            in
            ( Option.to_list static_range
              |> List.map ~f:(fun range ->
                     { d1 = get_dim ~d:range (); fix1 = false; d2 = slice_var; fix2 = false }),
              { r = expanded_batch; subr = sh.batch } )
          else
            match sh.batch.dims with
            | [] ->
                raise
                @@ Shape_error
                     ("Batch slice: insufficent number of batch axes", [ Shape_mismatch [ cur_sh; sh ] ])
            | d2 :: dims ->
                let reduced_batch =
                  { dims; constr = Unconstrained; row = sh.batch.row; sh_id = Utils.one_int sh.id }
                in
                ( Option.to_list static_range
                  |> List.map ~f:(fun range ->
                         {
                           d1 = get_dim ~d:range ();
                           fix1 = is_fixed sh.batch.row;
                           d2;
                           fix2 = is_fixed cur_sh.batch.row;
                         }),
                  { r = cur_sh.batch; subr = reduced_batch } )
        in
        try
          unify_dim range_eq env |> apply_constraint cur_sh.batch
          |> unify_dims
               [ batch_eq; { r = cur_sh.input; subr = sh.input }; { r = cur_sh.output; subr = sh.output } ]
        with Shape_error (s, trace) when !with_error_trace ->
          raise
          @@ Shape_error
               ( [%string "Batch slice %{Arrayjit.Indexing.symbol_ident static_symbol} / %{s}"],
                 Shape_mismatch [ cur_sh; sh ] :: trace ))
  | Transpose (Permute spec, sh) -> (
      let ls_rhs, ls_lhs =
        match einsum_of_spec spec with
        | ls_rhs, None, ls_lhs -> (ls_rhs, ls_lhs)
        | _ ->
            raise
            @@ Shape_error
                 ( "Invalid permutation spec (expected one argument): " ^ spec,
                   [ Shape_mismatch [ cur_sh; sh ] ] )
      in
      let b_row, i_row, o_row, b_rhs, i_rhs, o_rhs = einsum_slot_spec_to_dims_bio ~generative:[] ls_rhs in
      let _, _, _, b_lhs, i_lhs, o_lhs =
        einsum_slot_spec_to_dims_bio ~generative ?b_row ?i_row ?o_row ls_lhs
      in
      try
        unify_dims
          [
            { r = cur_sh.batch; subr = b_lhs };
            { r = b_rhs; subr = sh.batch };
            { r = cur_sh.input; subr = i_lhs };
            { r = i_rhs; subr = sh.input };
            { r = cur_sh.output; subr = o_lhs };
            { r = o_rhs; subr = sh.output };
          ]
          env
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ([%string "Permute %{spec} / %{s}"], Shape_mismatch [ cur_sh; sh ] :: trace))
  | Broadcast (Einsum spec, sh1, sh2) -> (
      let ls_rhs1, ls_rhs2, ls_lhs =
        match einsum_of_spec spec with
        | ls_rhs1, Some ls_rhs2, ls_lhs -> (ls_rhs1, ls_rhs2, ls_lhs)
        | _, None, _ ->
            raise
            @@ Shape_error
                 ( "Invalid permutation spec (expected one argument): " ^ spec,
                   [ Shape_mismatch [ cur_sh; sh1; sh2 ] ] )
      in
      let b_row, i_row, o_row, b_rhs1, i_rhs1, o_rhs1 = einsum_slot_spec_to_dims_bio ~generative:[] ls_rhs1 in
      let b_row, i_row, o_row, b_rhs2, i_rhs2, o_rhs2 =
        einsum_slot_spec_to_dims_bio ~generative:[] ?b_row ?i_row ?o_row ls_rhs2
      in
      let _, _, _, b_lhs, i_lhs, o_lhs =
        einsum_slot_spec_to_dims_bio ~generative ?b_row ?i_row ?o_row ls_lhs
      in
      try
        unify_dims
          [
            { r = cur_sh.batch; subr = b_lhs };
            { r = b_rhs1; subr = sh1.batch };
            { r = b_rhs2; subr = sh2.batch };
            { r = cur_sh.input; subr = i_lhs };
            { r = i_rhs1; subr = sh1.input };
            { r = i_rhs2; subr = sh2.input };
            { r = cur_sh.output; subr = o_lhs };
            { r = o_rhs1; subr = sh1.output };
            { r = o_rhs2; subr = sh2.output };
          ]
          env
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ([%string "Einsum %{spec} / %{s}"], Shape_mismatch [ cur_sh; sh1; sh2 ] :: trace)
      )

let indices_bio sh (type v) (arr : v array) =
  let n_batch = List.length sh.batch.dims in
  let batch : v Array.t = Array.sub arr ~pos:0 ~len:n_batch in
  let n_input = List.length sh.input.dims in
  let input = Array.sub arr ~pos:n_batch ~len:n_input in
  let n_output = List.length sh.output.dims in
  let output = Array.sub arr ~pos:(n_batch + n_input) ~len:n_output in
  (batch, input, output)

let state =
  ref
    {
      dim_env = Map.empty (module Dim_var);
      row_env = Map.empty (module Int);
      proj_env = Map.empty (module Int);
    }

let propagate_shapes update_step =
  state := unify_shapes !state update_step;
  let upd row =
    let expanded = subst_row !state.row_env row in
    { expanded with dims = List.map expanded.dims ~f:(subst_dim !state.dim_env) }
  in
  let upd sh =
    sh.batch <- upd sh.batch;
    sh.input <- upd sh.input;
    sh.output <- upd sh.output
  in
  upd update_step.shape;
  match update_step.logic with
  | Terminal _ -> ()
  | Transpose (_, sh1) -> upd sh1
  | Broadcast (_, sh1, sh2) ->
      upd sh1;
      upd sh2

let rec force_row_to_dims =
  let rec f = function
    | Dim { d; _ } -> d
    | Var v as d -> (
        match Map.find !state.dim_env v with
        | None -> raise @@ Shape_error ("Dimensions still unknown", [ Dim_mismatch [ d ] ])
        | Some dim -> f dim)
  in
  function
  | { dims; constr; row = Row_var v; sh_id } -> (
      match Map.find !state.row_env v with
      | None ->
          let row_env =
            Map.add_exn !state.row_env ~key:v
              ~data:{ dims = []; constr = Unconstrained; row = Broadcastable; sh_id }
          in
          state := { !state with row_env };
          Array.of_list_map dims ~f
      | Some row2 -> force_row_to_dims { dims = row2.dims @ dims; constr; row = row2.row; sh_id })
  | { dims; constr = _; row = Broadcastable | Fixed; sh_id = _ } -> Array.of_list_map dims ~f

(** Uses the matrix convention of putting the input axes last.
    Note: [force_to_dims] is "destructive": it closes shapes that remain incomplete after inference. *)
let force_to_dims (sh : t) : int array =
  try Array.concat_map ~f:force_row_to_dims [| sh.batch; sh.output; sh.input |]
  with Shape_error (s, more) when !with_error_trace ->
    if Utils.settings.with_debug then
      Stdlib.Format.printf "force_to_dims: shape error global env=@ %a\n%!" Sexp.pp_hum
        (sexp_of_environment !state);
    raise @@ Shape_error ("Dimensions still unknown / " ^ s, Shape_mismatch [ sh ] :: more)

let rec row_to_labels env =
  let rec f = function
    | Dim { label = Some l; _ } -> l
    | Dim { label = None; _ } -> ""
    | Var v -> (
        match Map.find env.dim_env v with None -> Option.value v.label ~default:"" | Some dim -> f dim)
  in
  function
  | { dims; constr; row = Row_var v; sh_id } -> (
      match Map.find env.row_env v with
      | None -> Array.of_list_map dims ~f
      | Some row2 -> row_to_labels env { dims = row2.dims @ dims; constr; row = row2.row; sh_id })
  | { dims; constr = _; row = Broadcastable | Fixed; sh_id = _ } -> Array.of_list_map dims ~f

(** Uses the matrix convention of putting the input axes last. *)
let to_labels (sh : t) : string array =
  Array.concat_map ~f:(row_to_labels !state) [| sh.batch; sh.output; sh.input |]

(** *** Projection inference *** *)

open Arrayjit.Indexing

(** Computes the indexing into subtensors given the shape information of a tensor. 
    [derive_projections] should only be invoked when the shapes are fully inferred already! *)
let derive_projections update_step =
  let dims_of sh = sh.batch.dims @ sh.output.dims @ sh.input.dims in
  let lhs = update_step.shape in
  let project rhs =
    (* Close the shapes. *)
    let lhs_dims = force_to_dims lhs in
    let (_ : int array list) = List.map ~f:force_to_dims rhs in
    propagate_shapes update_step;
    let all_dims = List.concat_map ~f:dims_of @@ (lhs :: rhs) in
    let proj_repr proj_id = fst @@ Utils.union_find ~equal:Int.equal !state.proj_env ~key:proj_id ~rank:0 in
    let get_proj = function
      | Dim { d; proj_id; _ } -> (proj_repr proj_id, d)
      | Var _ as v ->
          raise
          @@ Shape_error
               ( "derive_projections: shape still not fully inferred",
                 [ Shape_mismatch (lhs :: rhs); Dim_mismatch [ v ] ] )
    in
    (* Note: the ordering will affect performance of naive backends. *)
    let all_projs =
      Utils.unique_keep_first ~equal:(fun (p, _) (q, _) -> p = q) @@ List.map all_dims ~f:get_proj
    in
    let product_iterators = List.map all_projs ~f:(fun (p, d) -> (p, opt_symbol d)) in
    let projections =
      Map.of_alist_exn (module Int) @@ List.map product_iterators ~f:(fun (p, s) -> (p, opt_iterator s))
    in
    let product_space = Array.filter ~f:iterated @@ Array.of_list_map all_projs ~f:snd in
    let product_iterators = Array.of_list @@ List.filter_map ~f:snd product_iterators in
    let f sh = Array.of_list_map (dims_of sh) ~f:(fun d -> Map.find_exn projections @@ fst @@ get_proj d) in
    {
      product_space;
      lhs_dims;
      product_iterators;
      project_lhs = f lhs;
      project_rhs = Array.of_list_map ~f rhs;
      debug_info = sexp_of_update_step update_step;
    }
  in
  match update_step.logic with
  | Terminal _ -> project []
  | Transpose (_, sh) -> project [ sh ]
  | Broadcast (_, sh1, sh2) -> project [ sh1; sh2 ]

let backprop_ith_arg ~from_1 projections =
  let project_lhs = projections.project_rhs.(from_1 - 1) in
  let project_rhs = Array.copy projections.project_rhs in
  project_rhs.(from_1 - 1) <- projections.project_lhs;
  { projections with project_lhs; project_rhs }

(** *** Shape builders *** *)

let make ?(fix_b = false) ?(fix_i = false) ?(fix_o = false) ?batch_dims ?input_dims ?output_dims ?batch_axes
    ?input_axes ?output_axes ?(deduced = Not_constrained) ~id () =
  let make_row fix = if fix then Fixed else Broadcastable in
  let make_dims fix ds =
    {
      dims = List.map ~f:(fun d -> get_dim ~d ()) ds;
      constr = Unconstrained;
      row = make_row fix;
      sh_id = Utils.one_int id;
    }
  in
  let make_axes fix ds =
    {
      dims = List.map ~f:(fun (label, d) -> get_dim ~d ~label ()) ds;
      constr = Unconstrained;
      row = make_row fix;
      sh_id = Utils.one_int id;
    }
  in
  let make_unknown () =
    { dims = []; constr = Unconstrained; row = get_row_var (); sh_id = Utils.one_int id }
  in
  let batch =
    match (batch_dims, batch_axes) with
    | Some batch_dims, None -> make_dims fix_b batch_dims
    | None, Some batch_axes -> make_axes fix_b batch_axes
    | None, None when not fix_b -> make_unknown ()
    | Some _, Some _ -> invalid_arg "Shape.make: do not provide both batch_dims, batch_axes"
    | None, None -> invalid_arg "Shape.make: do not provide fix_b:true for unknown batch axes"
  in
  let input =
    match (input_dims, input_axes) with
    | Some input_dims, None -> make_dims fix_i input_dims
    | None, Some input_axes -> make_axes fix_i input_axes
    | None, None when not fix_i -> make_unknown ()
    | Some _, Some _ -> invalid_arg "Shape.make: do not provide both input_dims, input_axes"
    | None, None -> invalid_arg "Shape.make: do not provide fix_b:true for unknown input axes"
  in
  let output =
    match (output_dims, output_axes) with
    | Some output_dims, None -> make_dims fix_o output_dims
    | None, Some output_axes -> make_axes fix_o output_axes
    | None, None when not fix_o -> make_unknown ()
    | Some _, Some _ -> invalid_arg "Shape.make: do not provide both output_dims, output_axes"
    | None, None -> invalid_arg "Shape.make: do not provide fix_b:true for unknown output axes"
  in
  let result = { input; output; batch; id } in
  (match deduced with
  | Not_constrained -> ()
  | Input_equals_output -> (
      try state := unify_dims [ { r = input; subr = output } ] !state
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ("Input_equals_output / " ^ s, Shape_mismatch [ result ] :: trace)));
  result

let shape_spec_to_dims_bio ?b_row ?i_row ?o_row labels =
  let f _kind vars = function
    | Either.First s when String.contains s '=' -> (
        let label, dim =
          match String.split s ~on:'=' with
          | [ l; d ] -> (l, d)
          | _ -> invalid_arg "shape_spec_to_dims_bio: too many '='"
        in
        try get_dim ~d:(Int.of_string dim) ~label ()
        with _ -> invalid_arg "shape_spec_to_dims_bio: int expected after '='")
    | First label -> Var (Hashtbl.find_or_add vars label ~default:(fun () -> get_var ~label ()))
    | Second d -> get_dim ~d ()
  in
  axes_spec_to_dims_bio ?b_row ?i_row ?o_row ~f labels

let of_spec ?(deduced = Not_constrained) ~id spec =
  let _, _, _, batch, input, output = shape_spec_to_dims_bio @@ axis_labels_of_spec spec in
  let result = { input; output; batch; id } in
  (match deduced with
  | Not_constrained -> ()
  | Input_equals_output -> (
      try state := unify_dims [ { r = input; subr = output } ] !state
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ("of spec / " ^ s, Shape_mismatch [ result ] :: trace)));
  result

(** A [stop_broadcast] mutates the partially-inferred shape of a tensor in-place, substituting-in
    a [Fixed] marker on the dimensions. This way we avoid introducing a new tensor. *)
let stop_broadcast sh =
  let fix = function
    | Broadcastable | Fixed -> Fixed
    | Row_var _ as row -> (
        try
          state :=
            unify_dims
              [
                {
                  r = { dims = []; constr = Unconstrained; row; sh_id = Utils.one_int sh.id };
                  subr = { dims = []; constr = Unconstrained; row = Fixed; sh_id = Utils.one_int sh.id };
                };
              ]
              !state;
          Fixed
        with Shape_error (s, trace) when !with_error_trace ->
          raise @@ Shape_error ("stop_broadcast / " ^ s, Shape_mismatch [ sh ] :: trace))
  in
  sh.batch <- { sh.batch with row = fix sh.batch.row };
  sh.input <- { sh.input with row = fix sh.input.row };
  sh.output <- { sh.output with row = fix sh.output.row }

let broadcast sh =
  let fix = function
    | Broadcastable | Fixed -> Broadcastable
    | Row_var _ as row -> (
        try
          state :=
            unify_dims
              [
                {
                  r = { dims = []; constr = Unconstrained; row; sh_id = Utils.one_int sh.id };
                  subr =
                    { dims = []; constr = Unconstrained; row = Broadcastable; sh_id = Utils.one_int sh.id };
                };
              ]
              !state;
          Broadcastable
        with Shape_error (s, trace) when !with_error_trace ->
          raise @@ Shape_error ("broadcast / " ^ s, Shape_mismatch [ sh ] :: trace))
  in
  sh.batch <- { sh.batch with row = fix sh.batch.row };
  sh.input <- { sh.input with row = fix sh.input.row };
  sh.output <- { sh.output with row = fix sh.output.row }

let to_string_hum ?(style = `Axis_size) sh =
  let n_outputs = List.length @@ sh.output.dims in
  let n_batch = List.length @@ sh.batch.dims in
  let dim_to_string = function
    | Dim { label = None; _ } when phys_equal style `Only_labels -> "_"
    | Dim { label = Some l; _ } when phys_equal style `Only_labels -> l
    | Dim { d; label = None; _ } -> Int.to_string d
    | Dim { d; label = Some l; _ } -> [%string "%{l}=%{d#Int}"]
    | Var { id; label = Some l } -> [%string "$%{id#Int}:%{l}"]
    | Var { id; label = None } -> "$" ^ Int.to_string id
  in
  let dims_to_string kind =
    let dims = (dims_of_kind kind sh).dims in
    String.concat ~sep:","
    @@ List.mapi dims ~f:(fun i d ->
           let num =
             match kind with Input -> n_batch + n_outputs + i | Output -> n_batch + i | Batch -> i
           in
           match style with
           | `Only_labels | `Axis_size -> dim_to_string d
           | `Axis_number_and_size -> Int.to_string num ^ ":" ^ dim_to_string d)
  in
  let batch_dims = dims_to_string Batch in
  let input_dims = dims_to_string Input in
  let output_dims = dims_to_string Output in
  if String.is_empty batch_dims && String.is_empty input_dims then output_dims
  else if String.is_empty batch_dims then input_dims ^ "->" ^ output_dims
  else if String.is_empty input_dims then batch_dims ^ "|" ^ output_dims
  else batch_dims ^ "|" ^ input_dims ^ "->" ^ output_dims

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
           { in_axes = Input; from_end = 1 };
           { in_axes = Output; from_end = 1 };
           { in_axes = Input; from_end = 2 };
           { in_axes = Output; from_end = 2 };
           (if num_input_axes > 1 then { in_axes = Batch; from_end = 1 }
            else { in_axes = Output; from_end = 3 });
           { in_axes = Batch; from_end = 1 };
           { in_axes = Batch; from_end = 2 };
           { in_axes = Input; from_end = 3 };
           { in_axes = Output; from_end = 3 };
           { in_axes = Input; from_end = 4 };
           { in_axes = Output; from_end = 4 };
           { in_axes = Input; from_end = 5 };
           { in_axes = Output; from_end = 5 };
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

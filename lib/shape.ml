(** Tensor shape types, shape inference, projection inference. *)

open Base

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
type dim = Var of dim_var | Dim of int | Scaled of { num : int; denom : int; dim : dim }
[@@deriving equal, hash, compare, sexp, variants]

let rec scale ~num ~denom ?(force_conv = false) dim : dim =
  let ratio = Num.(num_of_int num // num_of_int denom) in
  let rat_to_int f n = Big_int.int_of_big_int @@ f @@ Ratio.normalize_ratio @@ Num.ratio_of_num n in
  let to_num = rat_to_int Ratio.numerator_ratio in
  let to_denom = rat_to_int Ratio.denominator_ratio in
  let num = to_num ratio and denom = to_denom ratio in
  match dim with
  | Var _ -> Scaled { num; denom; dim }
  | Dim d ->
      let res = Num.(ratio */ num_of_int d) in
      if to_denom res = 1 || force_conv then Dim (to_num res / to_denom res) else Scaled { num; denom; dim }
  | Scaled { num; denom; dim } ->
      let ratio = Num.(ratio */ num_of_int num // num_of_int denom) in
      let num = to_num ratio and denom = to_denom ratio in
      if force_conv then scale ~num ~denom ~force_conv dim else Scaled { num; denom; dim }

(** A row specifies how axes of a single kind in a shape can adapt to other shapes. *)
type row =
  | Row_var of int  (** The shape can be inferred to have more axes. *)
  | Broadcastable  (** The shape does not have more axes of this kind, but is "polymorphic". *)
  | Fixed  (** The shape fails in contexts expecting more axes of this kind -- it "stops broadcast". *)
  | Total_elems of int * row
[@@deriving equal, hash, compare, sexp, variants]

let uid = ref 0

let get_var () =
  Int.incr uid;
  { id = !uid; label = None }

let get_row_var () =
  Int.incr uid;
  Row_var !uid

type dims = { dims : dim list; row : row } [@@deriving equal, hash, compare, sexp]

type deduce_within_shape =
  | Not_constrained
  | Input_equals_output
  | Input_to_output_scale of { num : int; denom : int }
[@@deriving compare, sexp, variants]

type t = {
  mutable batch : dims;
  mutable input : dims;
  mutable output : dims;
  mutable axis_labels : string axis_map;
  deduce_within_shape : deduce_within_shape;
      (** Intended mostly for terminal node cases where both [input] and [output] are initially
      unknown. It makes it trivial to implement dimension-preserving hidden layers: just set
      [deduce_within_shape=Input_equals_output]. *)
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

type shape_error = Shape_mismatch of t * t | Row_mismatch of dims * dims | Dim_mismatch of dim * dim
[@@deriving sexp]

exception Shape_error of string * shape_error [@@deriving sexp]

(** Given a fully-inferred shape, maps axes to their corresponding positions in an index using the
    [Shape.to_dims] semantics. *)
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

(** Converts an axes-keyed map into an array of values using the [Shape.to_dims] semantics of axes.
    If the map is incomplete and the [~default] is not given, the result might be invalid: gaps in
    the array are filled with an arbitrary one of the provided values. *)
let axis_map_to_dims_index (type a) ?(default : a option) (idcs : a axis_map) : a array =
  let bch, inp, out = axis_map_to_dims_bio ?default idcs in
  Array.concat [ bch; out; inp ]

(** Generate a label into a broadcasted axis given an einsum-like spec. Axes that are part of the spec
    do not count, so that we can use the labels to align axes across different shapes (lhs, rhs1,
    rhs2). *)
let gen_label_of_axis ?parsed_spec axis =
  let open AxisKey in
  let prefix, idx =
    match parsed_spec with
    | None -> ("_fix_", axis.from_end)
    | Some parsed_spec -> ("_", axis.from_end - given_of_kind axis.in_axes parsed_spec)
  in
  prefix ^ (match axis.in_axes with Batch -> "__b" | Input -> "__i" | Output -> "__o") ^ Int.to_string idx

(** Augment the pseudo-labels map of an einsum notation with the generated labels for broadcasted
    axes. *)
let axes_with_inf_labels ~all_labels ls_xhs =
  let rec loop more kind accu =
    let offset = given_of_kind kind ls_xhs in
    let axis = AxisKey.{ in_axes = kind; from_end = offset + more } in
    let label = gen_label_of_axis ~parsed_spec:ls_xhs axis in
    if not @@ Map.mem all_labels label then accu
    else loop (more + 1) kind @@ Map.add_exn accu ~key:axis ~data:(Either.First label)
  in
  let see kind accu = if bcast_of_kind kind ls_xhs then loop 1 kind accu else accu in
  AxisKey.(see Batch @@ see Input @@ see Output @@ ls_xhs.labels)

let axes_with_pseudo_labels =
  Map.mapi ~f:(fun ~key ~data ->
      match data with Either.First l -> l | Either.Second _ -> gen_label_of_axis key)

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

type dim_env = dim Map.M(Dim_var).t
(** Note: The substituted variables can appear in the substitutions. *)

type row_env = dims Map.M(Int).t
(** Note: The substituted variables can appear in the substitutions. *)

type dim_eq = { d1 : dim; fix1 : bool; d2 : dim; fix2 : bool }
type dim_eqs = dim_eq list
type row_eqs = (dims * dims) list

let rec subst_dim dim_env = function
  | Scaled { num; denom; dim } -> Scaled { num; denom; dim = subst_dim dim_env dim }
  | Dim _ as d -> d
  | Var v as default -> Option.value ~default @@ Option.map ~f:(subst_dim dim_env) @@ Map.find dim_env v

let drop_from_end l n = List.rev @@ List.drop (List.rev l) n
let take_from_end l n = List.rev @@ List.take (List.rev l) n

let rec subst_row row_env ({ dims; row } as default) =
  match row with
  | Broadcastable | Fixed -> default
  | Total_elems (n, row) ->
      let { dims; row } = subst_row row_env { dims; row } in
      { dims; row = Total_elems (n, row) }
  | Row_var v -> (
      match Map.find row_env v with
      | None -> default
      | Some { dims = more_dims; row } -> { dims = more_dims @ dims; row })

let dim_to_int_exn = function
  | Dim d -> d
  | Scaled { num; denom; dim } -> (
      match scale ~num ~denom ~force_conv:true dim with
      | Dim d -> d
      | _ -> invalid_arg "dim_to_int: dim still unknown")
  | Var _ -> invalid_arg "dim_to_int: dim still unknown"

let rec base_row_spec = function
  | (Broadcastable | Fixed | Row_var _) as row_spec -> row_spec
  | Total_elems (_, row_spec) -> base_row_spec row_spec

let rec normalize_row row (dim_env, row_env) : dim_env * row_env =
  match row with
  | { dims; row = Total_elems (n1, (Total_elems (n2, _) as r1)) } ->
      if n1 = n2 then normalize_row { dims; row = r1 } (dim_env, row_env)
      else raise @@ Shape_error ("Inconsistent Total_elems constraints on the row", Row_mismatch (row, row))
  | { dims; row = Total_elems (n, Row_var v) } -> (
      match Map.find row_env v with
      | None ->
          (* Wait for more shape inference. *)
          (dim_env, row_env)
      | Some { dims = dims2; row = row2 } ->
          normalize_row { dims = dims2 @ dims; row = Total_elems (n, row2) } (dim_env, row_env))
  | { dims; row = Total_elems (n, r) } when not @@ is_row_var r -> (
      let dims = List.map dims ~f:(subst_dim dim_env) in
      let vars, nonvars = List.partition_tf dims ~f:is_var in
      if List.is_empty nonvars || List.length vars > 1 then
        raise @@ Shape_error ("Not enough information to resolve Total_elems", Row_mismatch (row, row))
      else
        let total = List.map nonvars ~f:dim_to_int_exn |> List.reduce_exn ~f:( * ) in
        match vars with
        | [] ->
            if n <> total then raise @@ Shape_error ("Total_elems constraint failed", Row_mismatch (row, row))
            else (dim_env, row_env)
        | [ Var v ] ->
            let rem = n / total in
            if rem = 0 then raise @@ Shape_error ("Total_elems constraint failed", Row_mismatch (row, row))
            else (Map.add_exn dim_env ~key:v ~data:(Dim rem), row_env)
        | _ -> assert false)
  | _ -> (dim_env, row_env)

let rec unify_dims dim_env row_env dim_eqs (row_eqs : row_eqs) =
  match row_eqs with
  | [] -> unify_dim dim_env row_env dim_eqs
  | ({ dims = []; row = Row_var v }, r2) :: row_eqs | (r2, { dims = []; row = Row_var v }) :: row_eqs ->
      let row_eqs = match Map.find row_env v with None -> row_eqs | Some r1 -> (r1, r2) :: row_eqs in
      let row_env = Map.update row_env v ~f:(fun _ -> subst_row row_env r2) in
      unify_dims dim_env row_env dim_eqs row_eqs
  | ({ dims = []; row = Fixed }, { dims = []; row = _ }) :: row_eqs
  | ({ dims = []; row = _ }, { dims = []; row = Fixed }) :: row_eqs ->
      unify_dims dim_env row_env dim_eqs row_eqs
  | (({ dims = []; row = Fixed } as r1), r2) :: _ | (r2, ({ dims = []; row = Fixed } as r1)) :: _ ->
      raise @@ Shape_error ("unify_dims: Fixed-mode axis number mismatch", Row_mismatch (r1, r2))
  | ({ dims = []; row = Broadcastable }, _) :: row_eqs | (_, { dims = []; row = Broadcastable }) :: row_eqs ->
      unify_dims dim_env row_env dim_eqs row_eqs
  | (({ dims = []; row = Total_elems (n1, br1) } as r1), r2) :: row_eqs
  | (r2, ({ dims = []; row = Total_elems (n1, br1) } as r1)) :: row_eqs ->
      if n1 = 1 && not (is_row_var @@ base_row_spec br1) then
        unify_dims dim_env row_env dim_eqs @@ (({ dims = []; row = br1 }, r2) :: row_eqs)
      else if not (is_row_var @@ base_row_spec br1) then
        raise
        @@ Shape_error ("unify_dims: Not enough elements for a Total_elems constraint", Row_mismatch (r1, r2))
      else
        unify_dims dim_env row_env dim_eqs
        @@ (({ dims = []; row = br1 }, { dims = r2.dims; row = Total_elems (n1, r2.row) }) :: row_eqs)
  | (({ dims = _ :: _ as ds1; row = r1 } as row1), ({ dims = _ :: _ as ds2; row = r2 } as row2)) :: row_eqs ->
      let suffix = min (List.length ds1) (List.length ds2) in
      let ds1_suf = take_from_end ds1 suffix in
      let ds2_suf = take_from_end ds2 suffix in
      let br1 = base_row_spec r1 and br2 = base_row_spec r2 in
      let dim_eqs =
        List.map2_exn ~f:(fun d1 d2 -> { d1; fix1 = is_fixed br1; d2; fix2 = is_fixed br2 }) ds1_suf ds2_suf
        @ dim_eqs
      in
      unify_dims dim_env row_env dim_eqs
        (({ dims = drop_from_end ds1 suffix; row = br1 }, { dims = drop_from_end ds2 suffix; row = br2 })
        :: row_eqs)
      |> normalize_row row1 |> normalize_row row2

and unify_dim dim_env row_env dim_eqs : dim_env * row_env =
  match dim_eqs with
  | [] -> (dim_env, row_env)
  | ({ d1 = Scaled { num; denom; dim = Scaled _ as dim }; _ } as eq) :: dim_eqs ->
      unify_dim dim_env row_env @@ ({ eq with d1 = scale ~num ~denom dim } :: dim_eqs)
  | ({ d2 = Scaled { num; denom; dim = Scaled _ as dim }; _ } as eq) :: dim_eqs ->
      unify_dim dim_env row_env @@ ({ eq with d2 = scale ~num ~denom dim } :: dim_eqs)
  | ({ d1 = Scaled { num; denom; dim = Var _ as v }; d2; _ } as eq) :: dim_eqs ->
      unify_dim dim_env row_env @@ ({ eq with d1 = v; d2 = scale ~num:denom ~denom:num d2 } :: dim_eqs)
  | ({ d1; d2 = Scaled { num; denom; dim = Var _ as v }; _ } as eq) :: dim_eqs ->
      unify_dim dim_env row_env @@ ({ eq with d1 = scale ~num:denom ~denom:num d1; d2 = v } :: dim_eqs)
  | { d1 = Dim d1; d2 = Dim d2; fix1 = _; fix2 = _ } :: dim_eqs when d1 = d2 ->
      unify_dim dim_env row_env dim_eqs
  | { d1 = Dim 1; d2 = Dim _; fix1 = false; fix2 = _ } :: dim_eqs -> unify_dim dim_env row_env dim_eqs
  | { d1 = Dim _; d2 = Dim 1; fix1 = _; fix2 = false } :: dim_eqs -> unify_dim dim_env row_env dim_eqs
  | ({ d1 = Var v; d2; fix1; fix2 } | { d2 = Var v; d1 = d2; fix1 = fix2; fix2 = fix1 }) :: dim_eqs ->
      let dim_eqs =
        match Map.find dim_env v with None -> dim_eqs | Some d1 -> { d1; d2; fix1; fix2 } :: dim_eqs
      in
      let dim_env = Map.update dim_env v ~f:(fun _ -> subst_dim dim_env d2) in
      unify_dim dim_env row_env dim_eqs
  | ({ d1 = Scaled { num; denom; dim }; _ } as eq) :: dim_eqs ->
      unify_dim dim_env row_env @@ ({ eq with d1 = scale ~force_conv:true ~num ~denom dim } :: dim_eqs)
  | ({ d2 = Scaled { num; denom; dim }; _ } as eq) :: dim_eqs ->
      unify_dim dim_env row_env @@ ({ eq with d2 = scale ~force_conv:true ~num ~denom dim } :: dim_eqs)
  | { d1; d2; fix1 = _; fix2 = _ } :: _ -> raise @@ Shape_error ("unify_dim", Dim_mismatch (d1, d2))

let unify_shapes dim_env row_env { shape = cur_sh; logic } =
  match logic with
  | Terminal (Range_over_offsets | Standard_uniform | Constant_fill { strict = false; _ }) ->
      (dim_env, row_env)
  | Terminal (Constant_fill { values; strict = true }) ->
      let len = Array.length values in
      let io_dims =
        try List.map ~f:dim_to_int_exn @@ cur_sh.output.dims @ cur_sh.input.dims
        with Invalid_argument _ ->
          raise
          @@ Shape_error
               ( "unify_shapes Constant_fill strict: non-batch dimensions must be known",
                 Shape_mismatch (cur_sh, cur_sh) )
      in
      let batch_elems = len / abs (List.fold ~init:1 ~f:( * ) io_dims) in
      let b_row = { dims = []; row = Total_elems (batch_elems, get_row_var ()) } in
      unify_dims dim_env row_env [] [ (b_row, cur_sh.batch) ]
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
                 Shape_mismatch (cur_sh, cur_sh) )
      in
      let batch_elems = len / abs (List.fold ~init:1 ~f:( * ) io_dims) in
      let b_row = { dims = []; row = Total_elems (batch_elems, get_row_var ()) } in
      unify_dims dim_env row_env [] [ (cur_sh.batch, b_row) ]
  | Transpose (Transpose, sh) ->
      unify_dims dim_env row_env []
        [ (cur_sh.batch, sh.batch); (cur_sh.input, sh.output); (cur_sh.output, sh.input) ]
  | Transpose (Pointwise_un, sh) ->
      unify_dims dim_env row_env []
        [ (cur_sh.batch, sh.batch); (cur_sh.input, sh.input); (cur_sh.output, sh.output) ]
       (* | Broadcast (Compose, sh1, sh2) ->
          [ (cur_sh.batch, sh.batch); (cur_sh.input, sh.input); (cur_sh.output, sh.output) ]

        | Transpose (Batch_slice _, _)
        | Transpose ( _, _) *)
  | _ -> (dim_env, row_env)

let indices_bio sh (type v) (arr : v array) =
  let n_batch = List.length sh.batch.dims in
  let batch : v Array.t = Array.sub arr ~pos:0 ~len:n_batch in
  let n_input = List.length sh.input.dims in
  let input = Array.sub arr ~pos:n_batch ~len:n_input in
  let n_output = List.length sh.output.dims in
  let output = Array.sub arr ~pos:(n_batch + n_input) ~len:n_output in
  (batch, input, output)

type state = { mutable dim_env : dim_env; mutable row_env : row_env }

let state = { dim_env = Map.empty (module Dim_var); row_env = Map.empty (module Int) }

let update_state (dim_env, row_env) =
  state.dim_env <- dim_env;
  state.row_env <- row_env

let row_to_dims dim_env row_env =
  let rec f = function
    | Dim d -> d
    | Var v as d -> (
        match Map.find dim_env v with
        | None -> raise @@ Shape_error ("Dimensions still unknown", Dim_mismatch (d, d))
        | Some dim -> f dim)
    | Scaled { num; denom; dim } -> f @@ scale ~num ~denom ~force_conv:true dim
  in
  function
  | { dims; row = Row_var v } as row1 ->
      (match Map.find row_env v with
      | None -> state.row_env <- Map.add_exn state.row_env ~key:v ~data:{ dims; row = Broadcastable }
      | Some row2 -> update_state @@ unify_dims state.dim_env state.row_env [] [ (row1, row2) ]);
      Array.of_list_map dims ~f
  | { dims; row = _ } -> Array.of_list_map dims ~f

(** Uses the matrix convention of putting the input axes last. *)
let to_dims (sh : t) : int array =
  Array.concat_map ~f:(row_to_dims state.dim_env state.row_env) [| sh.batch; sh.output; sh.input |]

(** *** Projection inference *** *)

open Arrayjit.Indexing

(** Computes the indexing into subtensors given the shape information of a tensor. The processing
    mirrors [propagate_shapes], but [derive_projections] should only be invoked when the shapes
    are inferred already. *)
let derive_projections _shape_update = identity_projections

let backprop_ith_arg ~from_1 projections =
  let project_lhs = projections.project_rhs.(from_1 - 1) in
  let project_rhs = Array.copy projections.project_rhs in
  project_rhs.(from_1 - 1) <- projections.project_lhs;
  { projections with project_lhs; project_rhs }

let make ?(fix_b = false) ?(fix_i = false) ?(fix_o = false) ?batch_dims ?input_dims ?output_dims ?axis_labels
    ?deduced ~id () =
  let make_row fix = function _ when fix -> Fixed | None -> get_row_var () | Some _ -> Broadcastable in
  let make_dims fix ds = { dims = List.map ~f:dim @@ Option.value ~default:[] ds; row = make_row fix ds } in
  let batch = make_dims fix_b batch_dims in
  let input = make_dims fix_i input_dims in
  let output = make_dims fix_o output_dims in
  let deduce_within_shape = Option.value deduced ~default:Not_constrained in
  let axis_labels =
    match axis_labels with
    | None -> Map.empty (module AxisKey)
    | Some spec ->
        Map.map (axis_labels_of_spec spec).labels ~f:(function
          | Either.First label -> label
          | Second dim -> Int.to_string dim)
  in
  { input; output; batch; deduce_within_shape; axis_labels; id }

let to_string_hum ?(style = `Axis_size) sh =
  let n_outputs = List.length @@ sh.output.dims in
  let n_batch = List.length @@ sh.batch.dims in
  let rec dim_to_string = function
    | Dim d -> Int.to_string d
    | Var { id; label = Some l } -> [%string "$%{id#Int}:%{l}"]
    | Var { id; label = None } -> "$" ^ Int.to_string id
    | Scaled { num; denom; dim } -> [%string "%{num#Int}/%{denom#Int}*%{dim_to_string dim}"]
  in
  let dims_to_string kind =
    let dims = (dims_of_kind kind sh).dims in
    let n_dims = List.length dims in
    String.concat ~sep:","
    @@ List.mapi dims ~f:(fun i d ->
           let key = AxisKey.{ in_axes = kind; from_end = n_dims - i } in
           let num =
             match kind with Input -> n_batch + n_outputs + i | Output -> n_batch + i | Batch -> i
           in
           match (style, Map.find sh.axis_labels key) with
           | `Only_labels, None -> "_"
           | `Axis_size, None -> dim_to_string d
           | `Axis_number_and_size, None -> Int.to_string num ^ ":" ^ dim_to_string d
           | `Only_labels, Some l -> l
           | `Axis_size, Some l -> l ^ ":" ^ dim_to_string d
           | `Axis_number_and_size, Some l -> l ^ "=" ^ Int.to_string num ^ ":" ^ dim_to_string d)
  in
  let batch_dims = dims_to_string Batch in
  let input_dims = dims_to_string Input in
  let output_dims = dims_to_string Output in
  if String.is_empty batch_dims && String.is_empty input_dims then output_dims
  else if String.is_empty batch_dims then input_dims ^ "->" ^ output_dims
  else if String.is_empty input_dims then batch_dims ^ "|" ^ output_dims
  else batch_dims ^ "|" ^ input_dims ^ "->" ^ output_dims

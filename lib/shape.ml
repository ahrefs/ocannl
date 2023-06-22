(** Tensor shape types and inference. *)

open Base

(** An index pointing to any of a shape's axes, including the kind of the axis ([Batch, Input, Output])
    and the position (which is counted from the end to facilitate broadcasting).

    Note the following inconsistency due to differing conventions in function notation and matrix notation:
    for label specifications and einsum notation, we write "batch|inputs->outputs", but when we convert
    a shape to an [Code] index we do it in the order [[batch; outputs; inputs]]. *)
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
type dedicated_axis = Task_id | Sample_num [@@deriving equal, compare, sexp, variants]

type axis_special =
  | Dim  (** A "randomly accessed" or "frequently reduced" axis -- no optimization hint. *)
  | Dedicated of dedicated_axis  (** An axis whose iteration can have special treatment on some backends. *)
  | Frozen
      (** An axis that should be indexed at a single position during a single `refresh_session`:
          a dynamic index into it will be a [Frozen_recipient]. *)
[@@deriving equal, compare, sexp, variants]

type dim = { special : axis_special; dim : int } [@@deriving equal, compare, sexp]

let dim dim = { special = Dim; dim }
let parallel dim = { special = Dedicated Task_id; dim }
let minibatch dim = { special = Dedicated Sample_num; dim }
let dim_1 = function { special = Dedicated _; _ } -> false | { dim = 1; _ } -> true | _ -> false

type parsed_axis_labels = {
  bcast_batch : bool;
  bcast_input : bool;
  bcast_output : bool;
  given_batch : int;
  given_input : int;
  given_output : int;
  labels : (string, dim) Either.t axis_map;
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

let dim_to_string = function
  | { special = Dim; dim } -> Int.to_string dim
  | { special = Frozen; dim } -> "frozen " ^ Int.to_string dim
  | { special = Dedicated Task_id; dim } -> "parallel " ^ Int.to_string dim
  | { special = Dedicated Sample_num; dim } -> "minibatch " ^ Int.to_string dim

type dims =
  | Given of dim list
      (** User-provided dimensions. They will not change but will be broadcasted to bigger sizes. *)
  | Fixed of dim list
      (** User-provided dimensions that will fail if used in a different size context, even if broadcastable.
      Note that [Operation.stop_broadcast] implements additional shape logic:
      it converts the (bottom-up i.e. partially inferred) shape into a [Fixed] variant. *)
  | Inferred of dim list
      (** Dimensions that will itself change to a bigger size: they adapt to the broadcasted size. *)
  | Unknown
      (** User-provided and will be replaced through inference. Prefer using [Unknown] to [Inferred []]. *)
[@@deriving equal, compare, sexp, variants]

let map_dims ~f = function
  | Given dims -> Given (f dims)
  | Fixed dims -> Fixed (f dims)
  | Inferred dims -> Inferred (f dims)
  | Unknown -> Inferred (f [])

type deduce_dims = Not_constrained | Input_equals_output | Input_output_scale of float
[@@deriving compare, sexp, variants]

(** Converts dimensions according to the specification. Note that scalar axes (1D) are not scaled,
    for compatibility with broadcasting.

    Note that in practice [from] will be [Unknown] or [Inferred] dimensions, making it of little relevance
    how the [Given] and [Fixed] cases are interpreted here. *)
let deduce_dims from : deduce_dims -> dims = function
  | Not_constrained -> Unknown
  | Input_equals_output -> (
      match from with Given dims | Fixed dims -> Inferred dims | Inferred _ | Unknown -> from)
  | Input_output_scale sc -> (
      match from with
      | Unknown -> Unknown
      | Given dims | Fixed dims | Inferred dims ->
          Inferred
            (List.map dims ~f:(fun d ->
                 if d.dim = 1 || not (is_dim d.special) then d
                 else { d with dim = Float.(iround_exn ~dir:`Up @@ (sc * of_int d.dim)) })))

type t = {
  mutable batch : dims;
  mutable input : dims;
  mutable output : dims;
  mutable axis_labels : string axis_map;
  deduce_within_shape_constraints : deduce_dims;
      (** Intended for terminal node cases where both [input] and [output] are initially
      unknown. It makes it trivial to implement dimension-preserving hidden layers: just set
      [deduce_within_shape_constraints=Input_equals_output]. *)
  id : int;  (** A node that has the same shape as this shape. *)
}
[@@deriving fields, sexp]
(** The datatype from which the actual Code shapes are computed.

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

let list_of_dims = function Given ls | Fixed ls | Inferred ls -> ls | Unknown -> []

let shift_axes_of_kind kind sh ~f =
  Map.of_alist_exn (module AxisKey)
  @@ List.filter_map (Map.to_alist sh.axis_labels) ~f:(fun (({ in_axes; from_end }, v) as kv) ->
         if not @@ AxisKey.equal_kind kind in_axes then Some kv
         else
           let from_end = f from_end in
           if from_end < 1 then None else Some ({ in_axes; from_end }, v))

let append_all_axes ?prefix ?suffix ~main () =
  let affix, left =
    match (prefix, suffix) with
    | Some affix, None -> (affix, true)
    | None, Some affix -> (affix, false)
    | _ -> assert false
  in
  let ap affix dims = if left then list_of_dims affix @ dims else dims @ list_of_dims affix in
  let ap affix = function
    | Unknown -> Inferred (list_of_dims affix)
    | Inferred dims -> Inferred (ap affix dims)
    | Given dims -> Given (ap affix dims)
    | Fixed dims -> Fixed (ap affix dims)
  in
  let batch = ap affix.batch main.batch in
  let input = ap affix.input main.input in
  let output = ap affix.output main.output in
  let f ({ AxisKey.in_axes; from_end }, v) =
    let offset = List.length @@ list_of_dims @@ dims_of_kind in_axes (if left then main else affix) in
    ({ AxisKey.in_axes; from_end = from_end + offset }, v)
  in
  let axis_labels =
    if left then
      Map.to_alist affix.axis_labels |> List.map ~f
      |> Fn.flip List.append @@ Map.to_alist main.axis_labels
      |> Map.of_alist_exn (module AxisKey)
    else
      Map.to_alist main.axis_labels |> List.map ~f
      |> Fn.flip List.append @@ Map.to_alist affix.axis_labels
      |> Map.of_alist_exn (module AxisKey)
  in
  { main with batch; input; output; axis_labels }

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
  | Dynamic_index of {
      over_kind : AxisKey.kind;
      from_left : bool;
      other_axes_pointwise : bool;
      indexed_dims : int list option;
    }
      (** Uses RHS2 as an index into RHS1. The values of RHS2 along the last (output) axis are used to
      fix the RHS1 [over_kind] axes. If RHS1 has more [over_kind] axes than the size of RHS2's last
      axis, the remaining right axes are kept if [from_left] is true, otherwise the left axes
      are kept. The [Parallel] axes are preserved -- skipped over, never dynamically indexed.
      The fixed (indexed-into) axes are dropped from the shape of LHS. If RHS2 has more than one axis,
      [other_axes_pointwise] decides what to do with the other axes: if true, they are traversed
      pointwise with the corresponding axes of RHS1. For the pointwise alignment, we drop the last
      output axis of RHS2 and the fixed axes of RHS1. If [other_axes_pointwise] is false,
      the other axes of RHS2 are prepended to the remaining axes of RHS1 (of the corresponding
      kind). ([other_axes_pointwise] being true is akin to an inner product, being false is akin to
      an outer product.) If [indexed_dims] are not given, the shape of the indexed tensor (RHS1) must be given.
      Otherwise, [indexed_dims] provide the dimensions of the indexed axes.
      If otherwise unknown, [RHS2]'s output axis is inferred to be [Dim 1]. *)
[@@deriving sexp, equal]

type transpose_type =
  | Transpose  (** Swaps inputs and outputs of a shape, preserves batch axes. *)
  | Pointwise_un  (** Preserves the shape. *)
  | Permute of string
      (** [Permute (ls1^"=>"^ls2)] is a variant of the [einsum] syntax [Einsum (ls1^";"^ls1^"=>"^ls2)].
      Note: The "right-hand-side" is on the left! I.e. the syntax is "rhs=>lhs", "rhs1;rhs2=>lhs". *)
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
        else
          try Some (key, Either.Second { special = Dim; dim = Int.of_string s })
          with _ -> Some (key, First s)
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

(** How to propagate shape updates and do the last update of [Formula.t.shape] when finalizing the formula.
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
  | Terminal
[@@deriving sexp]

type update_step = { shape : t; logic : logic } [@@deriving sexp]
(** Data required for a shape inference update step. A step should equilibrate information, passing it both
    top-down and bottom-up. The child should be identifiable within the parent via physical equality
    (allowing that a child fills both slots of a binary parent). *)

exception Shape_error of string * t * t [@@deriving sexp]

(** Given a fully-inferred shape, maps axes to their corresponding positions in an index using the
    [Shape.to_dims] semantics. *)
let axis_keys_to_idcs (sh : t) : int axis_map =
  let b_dims =
    match sh.batch with
    | Unknown -> raise @@ Shape_error ("Batch dimensions still unknown", sh, sh)
    | Inferred dims | Given dims | Fixed dims ->
        (* Enumerate axes backwards. *)
        Array.of_list_mapi dims ~f:(fun i _ -> AxisKey.{ in_axes = Batch; from_end = i + 1 })
  in
  let i_dims =
    match sh.input with
    | Unknown -> raise @@ Shape_error ("Input dimensions still unknown", sh, sh)
    | Inferred dims | Given dims | Fixed dims ->
        Array.of_list_mapi dims ~f:(fun i _ -> AxisKey.{ in_axes = Input; from_end = i + 1 })
  in
  let o_dims =
    match sh.output with
    | Unknown -> raise @@ Shape_error ("Output dimensions still unknown", sh, sh)
    | Inferred dims | Given dims | Fixed dims ->
        Array.of_list_mapi dims ~f:(fun i _ -> AxisKey.{ in_axes = Output; from_end = i + 1 })
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

(** Splits the dimensions of a shape into a map from axes, putting at most one number in a [dims] of
    an axis. An empty [dims] list is an end-of-list sentinel: means that there are one fewer axes
    of the particular kind. *)
let to_axis_map (sh : t) : dims axis_map =
  let kind_dims kind =
    match dims_of_kind kind sh with
    | Unknown -> [ (AxisKey.{ in_axes = kind; from_end = 1 }, Unknown) ]
    | Inferred dims ->
        let n_dims = List.length dims in
        (AxisKey.{ in_axes = kind; from_end = n_dims + 1 }, Inferred [])
        :: List.rev_mapi dims ~f:(fun i d ->
               (AxisKey.{ in_axes = kind; from_end = n_dims - i }, Inferred [ d ]))
    | Given dims ->
        let n_dims = List.length dims in
        (AxisKey.{ in_axes = kind; from_end = n_dims + 1 }, Given [])
        :: List.rev_mapi dims ~f:(fun i d -> (AxisKey.{ in_axes = kind; from_end = n_dims - i }, Given [ d ]))
    | Fixed dims ->
        let n_dims = List.length dims in
        (AxisKey.{ in_axes = kind; from_end = n_dims + 1 }, Fixed [])
        :: List.rev_mapi dims ~f:(fun i d -> (AxisKey.{ in_axes = kind; from_end = n_dims - i }, Fixed [ d ]))
  in
  let b_dims = kind_dims Batch in
  let i_dims = kind_dims Input in
  let o_dims = kind_dims Output in
  Map.of_alist_exn (module AxisKey) @@ List.concat [ b_dims; i_dims; o_dims ]

(* Design choice: tensor shapes are decided while code is constructed, although not immediately.
   Due to mutable updates during shape inference, it is not possible to reuse the same formula with
   different shapes. The inference is finalized by invoking the [Formula.subtree_shape_updates] once
   on the root formula. *)

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

let set_dims_type sh typ =
  sh.batch <- typ (list_of_dims sh.batch);
  sh.input <- typ (list_of_dims sh.input);
  sh.output <- typ (list_of_dims sh.output)

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

let is_given_or_fixed dims = is_given dims || is_fixed dims

type eq_slot = [ `Lhs | `Rhs1 | `Rhs2 ] [@@deriving sexp]
type eqs_map = (string, (AxisKey.t * dims) list, Base.String.comparator_witness) Base.Map.t

type eqs_p_map =
  (string, (AxisKey.t * (dims, dim * dims) Either.t) list, Base.String.comparator_witness) Base.Map.t

let sexp_of_eqs_map (map : eqs_map) =
  Sexp.List (Map.to_alist map |> List.map ~f:[%sexp_of: string * (AxisKey.t * dims) list])

let sexp_of_eqs_p_map (map : eqs_p_map) =
  Sexp.List
    (Map.to_alist map |> List.map ~f:[%sexp_of: string * (AxisKey.t * (dims, dim * dims) Either.t) list])

type str_str_map = (string, string, Base.String.comparator_witness) Base.Map.t

let sexp_of_str_str_map (map : str_str_map) =
  Sexp.List (Map.to_alist map |> List.map ~f:[%sexp_of: string * string])

type str_dim_map = (string, dim, Base.String.comparator_witness) Base.Map.t

let sexp_of_str_dim_map (map : str_dim_map) =
  Sexp.List (Map.to_alist map |> List.map ~f:[%sexp_of: string * dim])

type str_fdim_map = (string, bool * dim, Base.String.comparator_witness) Base.Map.t

let sexp_of_str_fdim_map (map : str_fdim_map) =
  Sexp.List (Map.to_alist map |> List.map ~f:[%sexp_of: string * (bool * dim)])

type axis_dim_map = (AxisKey.t, dim, AxisKey.comparator_witness) Base.Map.t

let sexp_of_axis_dim_map (map : axis_dim_map) =
  Sexp.List (Map.to_alist map |> List.map ~f:[%sexp_of: AxisKey.t * dim])

type axis_fdim_map = (AxisKey.t, bool * dim, AxisKey.comparator_witness) Base.Map.t

let sexp_of_axis_fdim_map (map : axis_fdim_map) =
  Sexp.List (Map.to_alist map |> List.map ~f:[%sexp_of: AxisKey.t * (bool * dim)])

type axis_str_map = (AxisKey.t, string, AxisKey.comparator_witness) Base.Map.t

let sexp_of_axis_str_map (map : axis_str_map) =
  Sexp.List (Map.to_alist map |> List.map ~f:[%sexp_of: AxisKey.t * string])

type axis_plab_map = (AxisKey.t, (string, dim) Either.t, AxisKey.comparator_witness) Base.Map.t

let sexp_of_axis_plab_map (map : axis_plab_map) =
  Sexp.List (Map.to_alist map |> List.map ~f:[%sexp_of: AxisKey.t * (string, dim) Either.t])

let drop_keeping_special subs dims =
  let rec loop subs par = function
    | ({ special = Dedicated _; _ } as p) :: dims when subs > 0 -> loop subs (p :: par) dims
    | _ :: dims when subs > 0 -> loop (subs - 1) par dims
    | dims -> List.rev_append par dims
  in
  loop subs [] dims

let take_skipping_special subs dims =
  let rec loop acc subs = function
    | { special = Dedicated _; _ } :: dims when subs > 0 -> loop acc subs dims
    | dim :: dims when subs > 0 -> loop (dim :: acc) (subs - 1) dims
    | _ -> List.rev acc
  in
  loop [] subs dims

let take_keeping_special ~debug_sh1 ~debug_sh2 ?(indexed_dims = []) subs dims =
  let rec loop acc subs = function
    | indexed_dims, ({ special = Dedicated _; _ } as p) :: dims when subs > 0 ->
        loop (p :: acc) subs (indexed_dims, dims)
    | [], dim :: dims when subs > 0 -> loop (dim :: acc) (subs - 1) ([], dims)
    | ind_dim :: indexed_dims, dim :: dims when subs > 0 ->
        if not (ind_dim.dim = dim.dim) then
          raise (Shape_error ("Dynamic indexing: axis size requirement not met", debug_sh1, debug_sh2));
        loop (dim :: acc) (subs - 1) (indexed_dims, dims)
    | _ -> List.rev acc
  in
  loop [] subs (indexed_dims, dims)

let count_special_among subs dims =
  let rec loop acc subs = function
    | { special = Dedicated _; _ } :: dims when subs > 0 -> loop (acc + 1) subs dims
    | _ :: dims when subs > 0 -> loop acc (subs - 1) dims
    | _ -> acc
  in
  loop 0 subs dims

(** Performs a local step of shape inference, propagates information into and out of the parent shape
    and the child shape(s). *)
let rec propagate_shapes (update : update_step) =
  let pointwise_labels debug1 debug2 ls1 ls2 =
    Map.merge ls1 ls2 ~f:(fun ~key -> function
      | `Both (l1, l2) ->
          if String.equal l1 l2 then Some l1
          else
            let error =
              "Axis label mismatch: " ^ l1 ^ " vs " ^ l2 ^ " for " ^ Sexp.to_string_hum
              @@ AxisKey.sexp_of_t key
            in
            raise @@ Shape_error (error, debug1, debug2)
      | `Right l | `Left l -> Some l)
  in
  let broad_dim ~fixed_left ~fixed_right debug1 debug2 axis_key label = function
    | d1, d2 when equal_dim d1 d2 -> d1
    | { dim = 1; _ }, d when not fixed_left -> d
    | d, { dim = 1; _ } when not fixed_right -> d
    | d1, d2 ->
        let opt_label = match label with None -> "" | Some l -> " (" ^ l ^ ")" in
        let error =
          "Dimension mismatch for axis " ^ AxisKey.to_string axis_key ^ opt_label ^ ": " ^ dim_to_string d1
          ^ " vs. " ^ dim_to_string d2
        in
        raise @@ Shape_error (error, debug1, debug2)
  in
  (* If initially [lhs] is [Unknown], [rhs1] is [sh1] and [rhs2] is [sh2],
     then [lhs] becomes [broadcast_dims sh1 sh2]. *)
  let broadcast_dims sh1 sh2 kind labels sh1_dims sh2_dims =
    let rec broad_back_dims ~fixed_left ~fixed_right accu i = function
      | [], [] -> accu
      | [], dims when not fixed_left -> List.rev_append dims accu
      | dims, [] when not fixed_right -> List.rev_append dims accu
      | [], _ | _, [] ->
          let key = AxisKey.{ in_axes = kind; from_end = i } in
          let opt_label = match Map.find labels key with None -> "" | Some l -> " (" ^ l ^ ")" in
          let error = "Different number of axes around from-end " ^ AxisKey.to_string key ^ opt_label in
          raise @@ Shape_error (error, sh1, sh2)
      | d1 :: dims1, d2 :: dims2 ->
          let key = AxisKey.{ in_axes = kind; from_end = i } in
          broad_back_dims ~fixed_left ~fixed_right
            (broad_dim ~fixed_left ~fixed_right sh1 sh2 key (Map.find labels key) (d1, d2) :: accu)
            (i + 1) (dims1, dims2)
    in
    let broadcast_dims ~dims1 ~dims2 =
      broad_back_dims ~fixed_left:(is_fixed sh1_dims) ~fixed_right:(is_fixed sh2_dims) [] 1
        (List.rev dims1, List.rev dims2)
    in
    match (sh1_dims, sh2_dims) with
    | Unknown, Unknown -> Unknown
    | (Inferred dims | Given dims | Fixed dims), Unknown | Unknown, (Inferred dims | Given dims | Fixed dims)
      ->
        Inferred dims
    | Fixed dims1, Fixed dims2 -> Fixed (broadcast_dims ~dims1 ~dims2)
    | (Given dims1 | Fixed dims1), (Given dims2 | Fixed dims2) -> Given (broadcast_dims ~dims1 ~dims2)
    | (Inferred dims1 | Given dims1 | Fixed dims1), (Inferred dims2 | Given dims2 | Fixed dims2) ->
        Inferred (broadcast_dims ~dims1 ~dims2)
  in
  let cur_sh = update.shape in
  (* Note: does not work with arbitrary permutation as in einsum. *)
  let update_labels sh1 to_kind sh2 from_kind =
    pointwise_labels sh1 sh2 sh1.axis_labels
    @@ Map.map_keys_exn (module AxisKey) ~f:(fun k -> { k with in_axes = to_kind })
    @@ Map.filter_keys sh2.axis_labels ~f:AxisKey.(fun k -> equal_kind k.in_axes from_kind)
  in
  let broadcast_into ?(det = false) to_sh to_kind from_sh from_kind =
    match (dims_of_kind to_kind to_sh, dims_of_kind from_kind from_sh) with
    | ((Given _ | Fixed _) as into_dims), from_dims ->
        ignore @@ broadcast_dims to_sh from_sh to_kind to_sh.axis_labels into_dims from_dims;
        into_dims
    | into_dims, from_dims -> (
        to_sh.axis_labels <- update_labels to_sh to_kind from_sh from_kind;
        let result = broadcast_dims to_sh from_sh to_kind to_sh.axis_labels into_dims from_dims in
        match (det, from_dims, result) with
        | true, Fixed _, Inferred dims -> Fixed dims
        | true, Given _, Inferred dims -> Given dims
        | _ -> result)
  in
  let einsum_one_dim_opt debug_spec debug1 debug2 label terms =
    List.fold terms ~init:(false, None) ~f:(fun ((is_fixed, dim) as accu) (_axis, dims) ->
        match (dim, dims) with
        | _, (Inferred (_ :: _ :: _) | Given (_ :: _ :: _) | Fixed (_ :: _ :: _)) -> assert false
        | None, Unknown ->
            assert (not is_fixed);
            (false, None)
        | Some _, Unknown -> accu
        | None, (Inferred [ dim2 ] | Given [ dim2 ]) ->
            assert (not is_fixed);
            (false, Some dim2)
        | None, Fixed [ dim2 ] ->
            assert (not is_fixed);
            (true, Some dim2)
        | Some dim1, (Inferred [ dim2 ] | Given [ dim2 ]) when equal_dim dim1 dim2 -> accu
        | Some dim1, Fixed [ dim2 ] when equal_dim dim1 dim2 -> (true, dim)
        | Some { dim = 1; _ }, (Inferred [ dim2 ] | Given [ dim2 ]) when not is_fixed -> (false, Some dim2)
        | Some dim1, (Inferred [ dim2 ] | Given [ dim2 ] | Fixed [ dim2 ]) ->
            raise
            @@ Shape_error
                 ( ("Dimension mismatch " ^ dim_to_string dim1 ^ " vs. " ^ dim_to_string dim2
                  ^ " for einsum pseudo-label " ^ label ^ " of " ^ debug_spec
                   ^ if dim1.dim = 1 || dim2.dim = 1 then " (broadcast prevented)" else ""),
                   debug1,
                   debug2 )
        | _, Fixed [] ->
            raise
            @@ Shape_error
                 ( "Too few fixed axes at einsum pseudo-label " ^ label ^ " of " ^ debug_spec
                   ^ " (broadcast prevented)",
                   debug1,
                   debug2 )
        | _, (Inferred [] | Given []) when is_fixed ->
            raise
            @@ Shape_error
                 ( "Too few actual axes at einsum pseudo-label " ^ label ^ " of " ^ debug_spec
                   ^ " (broadcast prevented)",
                   debug1,
                   debug2 )
        | _, (Inferred [] | Given []) -> accu)
  in
  let einsum_one_dim debug_spec debug1 debug2 ~key ~data =
    match einsum_one_dim_opt debug_spec debug1 debug2 key data with
    | false, None -> (false, { special = Dim; dim = 1 } (* which can still be expanded/broadcasted *))
    | true, None -> assert false
    | is_fixed, Some dim -> (is_fixed, dim)
  in
  let einsum_to_dims orig_dims is_bcast fdims =
    let is_fixed, dims = Array.unzip fdims in
    let is_fixed = Array.exists is_fixed ~f:Fn.id in
    let dims = Array.to_list dims in
    match (orig_dims, is_fixed, is_bcast) with
    | _, true, _ -> Fixed dims
    | Inferred _, _, true -> Inferred dims
    | Fixed _, _, true -> Fixed dims
    | _ -> Given dims
  in
  let eqs_xhs debug_spec debug_sh ls_xhs sh_xhs =
    let eqs =
      Map.merge ls_xhs.labels sh_xhs ~f:(fun ~key:axis -> function
        | `Both (Either.First label, dim) -> Some (label, (axis, dim))
        | `Left (First label) -> Some (label, (axis, Inferred []))
        | `Both
            (Second { dim = at; _ }, (Given [ { dim; _ } ] | Fixed [ { dim; _ } ] | Inferred [ { dim; _ } ]))
          when at >= dim ->
            raise
            @@ Shape_error ("Specified dimension outside bounds for its axis: " ^ debug_spec, debug_sh, cur_sh)
        | `Both (Second _, dim) -> Some (gen_label_of_axis axis, (axis, dim))
        | `Left (Second d) -> Some (gen_label_of_axis axis, (axis, Inferred [ { d with dim = d.dim + 1 } ]))
        | `Right (Given [] | Fixed [] | Inferred [] | Unknown) -> None
        | `Right _dim when not (bcast_of_kind axis.in_axes ls_xhs) ->
            raise
            @@ Shape_error ("Too many axes to permute -- spec too short: " ^ debug_spec, debug_sh, cur_sh)
        (* Note: the too-few-axes error is reported when einsum_one_dim processes the result. *)
        | `Right dim -> Some (gen_label_of_axis ~parsed_spec:ls_xhs axis, (axis, dim)))
    in
    Map.of_alist_multi (module String) @@ Map.data eqs
  in
  let pseudo_to_labels_xhs xhs_labels sh =
    Map.merge xhs_labels sh.axis_labels ~f:(fun ~key:_ -> function
      | `Both (pseudo, label) -> Some (pseudo, label) | `Left _pseudo -> None | `Right _label -> assert false)
    |> Map.data
    |> Map.of_alist_exn (module String)
  in
  let all_axis_labels debug1 debug2 debug_spec pseudo_to_labels_1 pseudo_to_labels_2 =
    Map.merge pseudo_to_labels_1 pseudo_to_labels_2 ~f:(fun ~key:pseudo -> function
      | `Both (l1, l2) when String.equal l1 l2 -> Some l1
      | `Left l | `Right l -> Some l
      | `Both (l1, l2) ->
          let error =
            "Axis label mismatch: " ^ l1 ^ " vs " ^ l2 ^ " for pseudo label " ^ pseudo ^ " of spec "
            ^ debug_spec
          in
          raise @@ Shape_error (error, debug1, debug2))
  in
  match update.logic with
  | Terminal -> ()
  | Transpose (Transpose, sh) ->
      cur_sh.input <- broadcast_into ~det:true cur_sh Input sh Output;
      cur_sh.output <- broadcast_into ~det:true cur_sh Output sh Input;
      cur_sh.batch <- broadcast_into ~det:true cur_sh Batch sh Batch;
      sh.input <- broadcast_into sh Input cur_sh Output;
      sh.output <- broadcast_into sh Output cur_sh Input;
      sh.batch <- broadcast_into sh Batch cur_sh Batch
  | Transpose (Pointwise_un, sh) ->
      cur_sh.input <- broadcast_into ~det:true cur_sh Input sh Input;
      cur_sh.output <- broadcast_into ~det:true cur_sh Output sh Output;
      cur_sh.batch <- broadcast_into ~det:true cur_sh Batch sh Batch;
      sh.input <- broadcast_into sh Input cur_sh Input;
      sh.output <- broadcast_into sh Output cur_sh Output;
      sh.batch <- broadcast_into sh Batch cur_sh Batch
  | Transpose (Permute spec, sh) ->
      let ls_rhs, ls_lhs =
        match einsum_of_spec spec with
        | ls_rhs, None, ls_lhs -> (ls_rhs, ls_lhs)
        | _ -> raise @@ Shape_error ("Invalid permutation spec (expected one argument): " ^ spec, sh, cur_sh)
      in
      let sh_rhs : dims axis_map = to_axis_map sh in
      let sh_lhs : dims axis_map = to_axis_map cur_sh in
      let eqs_rhs : eqs_map = eqs_xhs spec sh ls_rhs sh_rhs in
      let eqs_lhs : eqs_map = eqs_xhs spec sh ls_lhs sh_lhs in
      let eqs : eqs_map =
        Map.merge eqs_rhs eqs_lhs ~f:(fun ~key:_label -> function
          | `Both (rhs, lhs) -> Some (rhs @ lhs) | `Left rhs -> Some rhs | `Right lhs -> Some lhs)
      in
      let label_dims : str_fdim_map = Map.mapi eqs ~f:(einsum_one_dim spec cur_sh sh) in
      let lhs_plabels : axis_plab_map = axes_with_inf_labels ~all_labels:label_dims ls_lhs in
      let lhs_labels : axis_str_map = axes_with_pseudo_labels lhs_plabels in
      (* To reassign labels across repeated pseudo-labels, we can forget the integers. *)
      let pseudo_to_labels_lhs : str_str_map = pseudo_to_labels_xhs lhs_labels cur_sh in
      let inferred_lhs : axis_fdim_map = Map.map lhs_labels ~f:(Map.find_exn label_dims) in
      let b_lhs, i_lhs, o_lhs = axis_map_to_dims_bio inferred_lhs in
      if is_inferred cur_sh.batch || is_unknown cur_sh.batch then
        cur_sh.batch <- einsum_to_dims cur_sh.batch ls_lhs.bcast_batch b_lhs;
      if is_inferred cur_sh.input || is_unknown cur_sh.input then
        cur_sh.input <- einsum_to_dims cur_sh.input ls_lhs.bcast_input i_lhs;
      if is_inferred cur_sh.output || is_unknown cur_sh.output then
        cur_sh.output <- einsum_to_dims cur_sh.output ls_lhs.bcast_output o_lhs;
      let rhs_plabels : axis_plab_map = axes_with_inf_labels ~all_labels:label_dims ls_rhs in
      let rhs_labels : axis_str_map = axes_with_pseudo_labels rhs_plabels in
      let pseudo_to_labels_rhs : str_str_map = pseudo_to_labels_xhs rhs_labels sh in
      let inferred_rhs : axis_fdim_map = Map.map rhs_labels ~f:(Map.find_exn label_dims) in
      let b_rhs, i_rhs, o_rhs = axis_map_to_dims_bio inferred_rhs in
      if is_inferred sh.batch || is_unknown sh.batch then
        sh.batch <- einsum_to_dims sh.batch ls_rhs.bcast_batch b_rhs;
      if is_inferred sh.input || is_unknown sh.input then
        sh.input <- einsum_to_dims sh.input ls_rhs.bcast_input i_rhs;
      if is_inferred sh.output || is_unknown sh.output then
        sh.output <- einsum_to_dims sh.output ls_rhs.bcast_output o_rhs;
      let all_axis_labels : str_str_map =
        all_axis_labels cur_sh sh spec pseudo_to_labels_lhs pseudo_to_labels_rhs
      in
      let lhs_axis_labels : axis_str_map = Map.filter_map lhs_labels ~f:(Map.find all_axis_labels) in
      cur_sh.axis_labels <- lhs_axis_labels;
      let rhs_axis_labels : axis_str_map = Map.filter_map rhs_labels ~f:(Map.find all_axis_labels) in
      sh.axis_labels <- rhs_axis_labels
  | Broadcast (Pointwise_bin, sh1, sh2) ->
      let up_labels = pointwise_labels sh1 sh2 sh1.axis_labels sh2.axis_labels in
      cur_sh.axis_labels <- up_labels;
      (* Note: will not work as expected (propagate givenness/fixedness) if the shape is pre-filled
         as [Inferred] instead of [Unknown]. *)
      if is_unknown cur_sh.input then
        cur_sh.input <- broadcast_dims sh1 sh2 AxisKey.Input up_labels sh1.input sh2.input
      else (
        cur_sh.input <- broadcast_into cur_sh Input sh1 Input;
        cur_sh.input <- broadcast_into cur_sh Input sh2 Input);
      if is_unknown cur_sh.output then
        cur_sh.output <- broadcast_dims sh1 sh2 AxisKey.Output up_labels sh1.output sh2.output
      else (
        cur_sh.output <- broadcast_into cur_sh Output sh1 Output;
        cur_sh.output <- broadcast_into cur_sh Output sh2 Output);
      if is_unknown cur_sh.batch then
        cur_sh.batch <- broadcast_dims sh1 sh2 AxisKey.Batch up_labels sh1.batch sh2.batch
      else (
        cur_sh.batch <- broadcast_into cur_sh Batch sh1 Batch;
        cur_sh.batch <- broadcast_into cur_sh Batch sh2 Batch);

      sh1.input <- broadcast_into sh1 Input cur_sh Input;
      sh1.output <- broadcast_into sh1 Output cur_sh Output;
      sh1.batch <- broadcast_into sh1 Batch cur_sh Batch;
      sh2.input <- broadcast_into sh2 Input cur_sh Input;
      sh2.output <- broadcast_into sh2 Output cur_sh Output;
      sh2.batch <- broadcast_into sh2 Batch cur_sh Batch
  | Broadcast (Compose, sh1, sh2) ->
      (* [sh2] is the value or the function that gets applied first: [cur_sh(x) = sh1(sh2(x))].
         I.e. [cur.I = sh2.I, cur.O = sh1.O, sh2.O = sh1.I]. *)
      cur_sh.input <- broadcast_into ~det:true cur_sh AxisKey.Input sh2 AxisKey.Input;
      cur_sh.output <- broadcast_into ~det:true cur_sh AxisKey.Output sh1 AxisKey.Output;
      if is_unknown cur_sh.batch then (
        let up_labels = update_labels cur_sh Batch sh1 Batch in
        cur_sh.axis_labels <- up_labels;
        let up_labels = update_labels cur_sh Batch sh2 Batch in
        cur_sh.axis_labels <- up_labels;
        cur_sh.batch <- broadcast_dims sh1 sh2 AxisKey.Batch up_labels sh1.batch sh2.batch)
      else (
        cur_sh.batch <- broadcast_into cur_sh Batch sh1 Batch;
        cur_sh.batch <- broadcast_into cur_sh Batch sh2 Batch);

      sh1.input <- broadcast_into sh1 Input sh2 Output;
      sh1.output <- broadcast_into sh1 Output cur_sh Output;
      sh1.batch <- broadcast_into sh1 Batch cur_sh Batch;
      sh2.input <- broadcast_into sh2 Input cur_sh Input;
      sh2.output <- broadcast_into sh2 Output sh1 Input;
      sh2.batch <- broadcast_into sh2 Batch cur_sh Batch;

      (* Always re-derive the output shape, to have the latest information. *)
      (* TODO: isn't it wasteful to discard the old sh1.output? *)
      if not @@ is_not_constrained sh1.deduce_within_shape_constraints then
        sh1.output <- deduce_dims sh2.input sh1.deduce_within_shape_constraints
        (* TODO(#37):
           if not @@ is_not_constrained sh1.deduce_input_from_output then
           sh1.input <- deduce_dims sh2.output sh1.deduce_input_from_output *)
  | Broadcast (Einsum spec, sh1, sh2) ->
      let ls_rhs1, ls_rhs2, ls_lhs =
        match einsum_of_spec spec with
        | ls_rhs1, Some ls_rhs2, ls_lhs -> (ls_rhs1, ls_rhs2, ls_lhs)
        | _ -> raise @@ Shape_error ("Invalid einsum spec (expected two arguments): " ^ spec, sh1, sh2)
      in
      let sh_rhs1 : dims axis_map = to_axis_map sh1 in
      let sh_rhs2 : dims axis_map = to_axis_map sh2 in
      let sh_lhs : dims axis_map = to_axis_map cur_sh in
      let eqs_rhs1 : eqs_map = eqs_xhs spec sh1 ls_rhs1 sh_rhs1 in
      let eqs_rhs2 : eqs_map = eqs_xhs spec sh2 ls_rhs2 sh_rhs2 in
      let eqs_lhs : eqs_map = eqs_xhs spec sh1 ls_lhs sh_lhs in
      let side_eq side (axis, dims) = ((side, axis), dims) in
      let eqs =
        Map.merge eqs_rhs1 eqs_lhs ~f:(fun ~key:_label -> function
          | `Both (rhs, lhs) ->
              Some (List.rev_map_append rhs ~f:(side_eq `Rhs1) @@ List.map lhs ~f:(side_eq `Lhs))
          | `Left rhs -> Some (List.map rhs ~f:(side_eq `Rhs1))
          | `Right lhs -> Some (List.map lhs ~f:(side_eq `Lhs)))
      in
      let eqs =
        Map.merge eqs_rhs2 eqs ~f:(fun ~key:_label -> function
          | `Both (rhs, more) -> Some (List.rev_map_append rhs ~f:(side_eq `Rhs2) more)
          | `Left rhs -> Some (List.map rhs ~f:(side_eq `Rhs2))
          | `Right more -> Some more)
      in
      let label_dims : str_fdim_map = Map.mapi eqs ~f:(einsum_one_dim spec sh1 sh2) in
      let lhs_plabels : axis_plab_map = axes_with_inf_labels ~all_labels:label_dims ls_lhs in
      let lhs_labels : axis_str_map = axes_with_pseudo_labels lhs_plabels in
      let pseudo_to_labels_lhs : str_str_map = pseudo_to_labels_xhs lhs_labels cur_sh in
      let inferred_lhs : axis_fdim_map = Map.map lhs_labels ~f:(Map.find_exn label_dims) in
      let b_lhs, i_lhs, o_lhs = axis_map_to_dims_bio inferred_lhs in
      if is_inferred cur_sh.batch || is_unknown cur_sh.batch then
        cur_sh.batch <- einsum_to_dims cur_sh.batch ls_lhs.bcast_batch b_lhs;
      if is_inferred cur_sh.input || is_unknown cur_sh.input then
        cur_sh.input <- einsum_to_dims cur_sh.input ls_lhs.bcast_input i_lhs;
      if is_inferred cur_sh.output || is_unknown cur_sh.output then
        cur_sh.output <- einsum_to_dims cur_sh.output ls_lhs.bcast_output o_lhs;
      let rhs1_plabels : axis_plab_map = axes_with_inf_labels ~all_labels:label_dims ls_rhs1 in
      let rhs1_labels : axis_str_map = axes_with_pseudo_labels rhs1_plabels in
      let pseudo_to_labels_rhs1 : str_str_map = pseudo_to_labels_xhs rhs1_labels sh1 in
      let inferred_rhs1 : axis_fdim_map = Map.map rhs1_labels ~f:(Map.find_exn label_dims) in
      let b_rhs1, i_rhs1, o_rhs1 = axis_map_to_dims_bio inferred_rhs1 in
      if is_inferred sh1.batch || is_unknown sh1.batch then
        sh1.batch <- einsum_to_dims sh1.batch ls_rhs1.bcast_batch b_rhs1;
      if is_inferred sh1.input || is_unknown sh1.input then
        sh1.input <- einsum_to_dims sh1.input ls_rhs1.bcast_input i_rhs1;
      if is_inferred sh1.output || is_unknown sh1.output then
        sh1.output <- einsum_to_dims sh1.output ls_rhs1.bcast_output o_rhs1;
      let rhs2_plabels : axis_plab_map = axes_with_inf_labels ~all_labels:label_dims ls_rhs2 in
      let rhs2_labels : axis_str_map = axes_with_pseudo_labels rhs2_plabels in
      let pseudo_to_labels_rhs2 : str_str_map = pseudo_to_labels_xhs rhs2_labels sh2 in
      let inferred_rhs2 : axis_fdim_map = Map.map rhs2_labels ~f:(Map.find_exn label_dims) in
      let b_rhs2, i_rhs2, o_rhs2 = axis_map_to_dims_bio inferred_rhs2 in
      if is_inferred sh2.batch || is_unknown sh2.batch then
        sh2.batch <- einsum_to_dims sh2.batch ls_rhs2.bcast_batch b_rhs2;
      if is_inferred sh2.input || is_unknown sh2.input then
        sh2.input <- einsum_to_dims sh2.input ls_rhs2.bcast_input i_rhs2;
      if is_inferred sh2.output || is_unknown sh2.output then
        sh2.output <- einsum_to_dims sh2.output ls_rhs2.bcast_output o_rhs2;
      let all_axis_labels1 : str_str_map =
        all_axis_labels cur_sh sh1 spec pseudo_to_labels_lhs pseudo_to_labels_rhs1
      in
      let all_axis_labels : str_str_map =
        all_axis_labels cur_sh sh2 spec all_axis_labels1 pseudo_to_labels_rhs2
      in
      let lhs_axis_labels : axis_str_map = Map.filter_map lhs_labels ~f:(Map.find all_axis_labels) in
      cur_sh.axis_labels <- lhs_axis_labels;
      let rhs1_axis_labels : axis_str_map = Map.filter_map rhs1_labels ~f:(Map.find all_axis_labels) in
      sh1.axis_labels <- rhs1_axis_labels;
      let rhs2_axis_labels : axis_str_map = Map.filter_map rhs2_labels ~f:(Map.find all_axis_labels) in
      sh2.axis_labels <- rhs2_axis_labels
  | Broadcast (Dynamic_index { over_kind; from_left; other_axes_pointwise; indexed_dims }, sh1, sh2) ->
      let subs =
        match List.last @@ list_of_dims sh2.output with
        | None ->
            sh2.output <- Given [ { special = Dim; dim = 1 } ];
            1
        | Some d -> d.dim
      in
      let check_no_axes indexed_dims =
        if List.length indexed_dims <> subs then
          raise
            (Shape_error
               ( "Dynamic indexing: provided dimensions and indices assume a different number of axes",
                 sh1,
                 sh2 ))
      in
      let reduced_sh2 =
        {
          sh2 with
          output = map_dims sh2.output ~f:List.drop_last_exn;
          axis_labels = shift_axes_of_kind AxisKey.Output sh2 ~f:(( - ) 1);
        }
      in
      let list_dims sh = list_of_dims @@ dims_of_kind over_kind sh in
      if
        Option.is_some indexed_dims
        && List.is_empty (list_dims sh1)
        && is_unknown (dims_of_kind over_kind cur_sh)
      then (* Wait for more information. *) ()
      else if Option.is_none indexed_dims && List.length (list_dims sh1) < subs then
        raise (Shape_error ("Insufficient indexed axes information for dynamic indexing", sh1, sh2))
      else if from_left then (
        let n_par_axes =
          match (indexed_dims, dims_of_kind over_kind sh1) with
          | Some indexed_dims, (Unknown | Inferred []) ->
              check_no_axes indexed_dims;
              (* Infer and fix the dimensions -- preserve "indexing from the left". *)
              let indexed_dims = List.map ~f:dim indexed_dims in
              let lhs_dims = list_dims cur_sh in
              update_kind ~f:(fun _ -> Given (indexed_dims @ lhs_dims)) over_kind sh1;
              0
          | None, _ ->
              let dims = list_dims sh1 in
              (* Fix the dimensions: we don't want new axes inferred to the left of indexed axes. *)
              update_kind ~f:(fun _ -> Given dims) over_kind sh1;
              count_special_among subs dims
          | Some indexed_dims, _ ->
              check_no_axes indexed_dims;
              let indexed_dims = List.map ~f:dim indexed_dims in
              let dims = list_dims sh1 in
              (* Verify the dimensions of the indexed axes. *)
              let _indexed_dims_with_par =
                take_keeping_special ~debug_sh1:sh1 ~debug_sh2:sh2 ~indexed_dims subs dims
              in
              (* Fix the dimensions: we don't want new axes inferred to the left of indexed axes. *)
              update_kind ~f:(fun _ -> Given dims) over_kind sh1;
              count_special_among subs dims
        in
        let sh1_size = List.length @@ list_dims sh1 in
        let reduced_dims over_dims = map_dims over_dims ~f:(drop_keeping_special subs) in
        let reduced_sh1 = map_over_kind over_kind ~f:reduced_dims sh1 in
        let drop_left from_end = if from_end > sh1_size - subs - n_par_axes then -1 else from_end in
        (* TODO: unfortunately we ignore labels on parallel axes. *)
        let reduced_sh1 = { reduced_sh1 with axis_labels = shift_axes_of_kind over_kind sh1 ~f:drop_left } in
        let extended_sh, logic =
          if other_axes_pointwise then (None, Broadcast (Pointwise_bin, reduced_sh1, reduced_sh2))
          else
            let extended_sh = append_all_axes ~prefix:reduced_sh2 ~main:reduced_sh1 () in
            (Some extended_sh, Transpose (Pointwise_un, extended_sh))
        in
        let update_other_axes = { shape = cur_sh; logic } in
        propagate_shapes update_other_axes;
        match extended_sh with
        | None ->
            (* Add back the indexed axes labels. *)
            let sh1_axis_labels =
              Map.fold sh1.axis_labels ~init:reduced_sh1.axis_labels
                ~f:(fun ~key:({ in_axes; from_end } as key) ~data labels ->
                  if AxisKey.equal_kind over_kind in_axes && from_end > sh1_size - subs - n_par_axes then
                    Map.add_exn labels ~key ~data
                  else labels)
            in
            sh1.axis_labels <- sh1_axis_labels
        | Some extended_sh ->
            (* FIXME: NOT IMPLEMENTED restoring inferred shapes and axis labels. *)
            ignore extended_sh)
      else (* FIXME: NOT IMPLEMENTED YET *)
        failwith "NOT IMPLEMENTED YET";
      let sh2_axis_labels = shift_axes_of_kind AxisKey.Output reduced_sh2 ~f:(( + ) 1) in
      let k1 = AxisKey.{ in_axes = Output; from_end = 1 } in
      let sh2_axis_labels =
        match Map.find sh2.axis_labels k1 with
        | None -> sh2_axis_labels
        | Some v -> Map.add_exn sh2_axis_labels ~key:k1 ~data:v
      in
      sh2.axis_labels <- sh2_axis_labels;
      (* FIXME: not sure about this behavior. *)
      if not @@ is_given_or_fixed sh2.output then
        sh2.output <- map_dims reduced_sh2.output ~f:(fun d -> d @ [ dim subs ])

(** Uses the matrix convention of putting the input axes last. *)
let to_dims (sh : t) : dim array =
  let b_dims =
    match sh.batch with
    | Unknown -> raise @@ Shape_error ("Batch dimensions still unknown", sh, sh)
    | Inferred dims | Given dims | Fixed dims -> Array.of_list dims
  in
  let i_dims =
    match sh.input with
    | Unknown -> raise @@ Shape_error ("Input dimensions still unknown", sh, sh)
    | Inferred dims | Given dims | Fixed dims -> Array.of_list dims
  in
  let o_dims =
    match sh.output with
    | Unknown -> raise @@ Shape_error ("Output dimensions still unknown", sh, sh)
    | Inferred dims | Given dims | Fixed dims -> Array.of_list dims
  in
  Array.concat [ b_dims; o_dims; i_dims ]

type symbol = Symbol of int [@@deriving compare, equal, sexp, hash, variants]

let unique_id = ref 1

let get_symbol () =
  let uid = !unique_id in
  Int.incr unique_id;
  Symbol uid

module CompareSymbol = struct
  type t = symbol = Symbol of int [@@deriving compare, equal, sexp, hash, variants]
end

module Symbol = struct
  include CompareSymbol
  include Comparator.Make (CompareSymbol)
end

let task_id_symbols = Hash_set.create (module Symbol)
let sample_num_symbols = Hash_set.create (module Symbol)

let get_sym_for_axis = function
  | Dim -> get_symbol ()
  | Dedicated Task_id ->
      let uid = get_symbol () in
      Hash_set.add task_id_symbols uid;
      uid
  | Dedicated Sample_num ->
      let uid = get_symbol () in
      Hash_set.add sample_num_symbols uid;
      uid
  | Frozen -> get_symbol ()

let task_id_sym = Hash_set.mem task_id_symbols
let sample_num_sym = Hash_set.mem sample_num_symbols
(* let iterate_sample_num = ref true *)

let fresh_symbol sym =
  if task_id_sym sym then get_sym_for_axis (Dedicated Task_id)
  else if sample_num_sym sym then get_sym_for_axis (Dedicated Sample_num)
  else get_symbol ()

let symbol_ident (Symbol s as sym) =
  if task_id_sym sym then "task_id_" ^ Int.to_string s
  else if sample_num_sym sym then "sample_num_" ^ Int.to_string s
  else "i" ^ Int.to_string s

let is_dedicated_any sym = task_id_sym sym || sample_num_sym sym
let is_dedicated_kind = function Task_id -> task_id_sym | Sample_num -> sample_num_sym


let iterated = function
  | { special = Dim; dim } when dim > 1 -> true
  (* | { special = Dedicated Sample_num; dim } when !iterate_sample_num && dim > 1 -> true *)
  | { special = Dedicated _; _ } -> true
  | _ -> false

let opt_symbol d = Option.some_if (iterated d) @@ get_sym_for_axis d.special

(** An index into a single axis for doing computations over multiple [Shape]-derived [Code]s. *)
type axis_index =
  | Fixed_idx of int  (** The specific position along an axis. *)
  | Iterator of symbol
      (** The given member of the [product_space] corresponding to some [product_iterators]. *)
  | Dynamic_recipient of symbol
      (** [Dynamic_recipient s] gets the index from the [s] member of an [Dynamic_provider sl]
      axis. *)
  | Dynamic_provider of { idcs : symbol array; target_dims : dim array }
      (** The values along a [Dynamic_provider {idcs; target_dims}] axis are read via "random access" by
      [Dynamic_recipient idx] axes for various values of [idx] in [idcs], modulo the corresponding
      dimension from [target_dims]. To prevent out-of-bound errors, set [target_dims] to the
      dimensions of the recipient axes. The contents stored with a [Dynamic_provider] are used during
      the translation from the high-level to the low-level representation, and later [Dynamic_provider]
      is used as a place-holder filled in with the axis number of the recipient. *)
  | Frozen_recipient of symbol
      (** Like [Dynamic_recipient] and [Fixed_idx], but the index value is not updated during an update step. *)
[@@deriving compare, equal, sexp, variants]

let opt_iterator = function None -> Fixed_idx 0 | Some sym -> Iterator sym

type str_osym_map = (string, symbol option, Base.String.comparator_witness) Base.Map.t

let sexp_of_str_osym_map (map : str_osym_map) =
  Sexp.List (Map.to_alist map |> List.map ~f:[%sexp_of: string * symbol option])

type axis_osym_map = (AxisKey.t, symbol option, AxisKey.comparator_witness) Base.Map.t

let sexp_of_axis_osym_map (map : axis_osym_map) =
  Sexp.List (Map.to_alist map |> List.map ~f:[%sexp_of: AxisKey.t * symbol option])

type projections = {
  product_space : dim array;
      (** The product space dimensions (concatentation of the relevant batch, output, input axes) with
      the same semantics as [to_dims], that an operation should parallelize (map-reduce) over. *)
  product_iterators : symbol array;
      (** The product space iterators (concatentation of the relevant batch, output, input axes)
      for iterating over the [product_space] axes, where same axes are at same array indices. *)
  project_lhs : axis_index array;
      (** A projection that takes an [product_space]-bound index and produces an index into the result of
      an operation. *)
  project_rhs1 : axis_index array;
      (** A projection that takes an [product_space]-bound index and produces an index into the (first)
      argument of an operation. *)
  project_rhs2 : axis_index array option;
      (** A projection that takes an [product_space]-bound index and produces an index into the second
      argument of a binary operation. *)
}
[@@deriving compare, equal, sexp]
(** All the information relevant for [Code] code generation contained in a completed [update_step]. *)

(** Projections for iterating over a terminal in [Code], or for a pointwise unary operator. *)
let identity_projections (product_space : dim array) =
  let product_iterators = Array.map product_space ~f:opt_symbol in
  let project_lhs = Array.map product_iterators ~f:opt_iterator in
  let product_space = Array.filter ~f:iterated product_space in
  let product_iterators = Array.filter_map ~f:Fn.id product_iterators in
  { product_space; product_iterators; project_lhs; project_rhs1 = project_lhs; project_rhs2 = None }

let indices_bio sh (type v) (arr : v array) =
  let n_batch = List.length @@ list_of_dims sh.batch in
  let batch : v Array.t = Array.sub arr ~pos:0 ~len:n_batch in
  let n_input = List.length @@ list_of_dims sh.input in
  let input = Array.sub arr ~pos:n_batch ~len:n_input in
  let n_output = List.length @@ list_of_dims sh.output in
  let output = Array.sub arr ~pos:(n_batch + n_input) ~len:n_output in
  (batch, input, output)

let project_broad d1 d2 =
  match (d1, d2) with
  | d1, d2 when equal_dim d1 d2 -> d1
  | { dim = 1; _ }, d | d, { dim = 1; _ } -> d
  | _ -> assert false

(* module Debug_runtime = Minidebug_runtime.Flushing (struct
     let debug_ch = Out_channel.stdout
   end) *)

let project_dyn_indexing ~(dynamic_syms : symbol array) ~(dedicated_syms : symbol array)
    (targets_and_skipped : dim array) : axis_index array =
  let remaining_dyn_syms, remaining_ded_syms, idcs =
    List.fold_left
      ~init:(Array.to_list dynamic_syms, Array.to_list dedicated_syms, [])
      (Array.to_list targets_and_skipped)
      ~f:(fun (dyn_syms, ded_syms, idcs) dim ->
        match dim.special with
        | Dedicated _ -> (dyn_syms, List.tl_exn ded_syms, Iterator (List.hd_exn ded_syms) :: idcs)
        | Frozen -> (List.tl_exn dyn_syms, ded_syms, Frozen_recipient (List.hd_exn dyn_syms) :: idcs)
        | Dim -> (List.tl_exn dyn_syms, ded_syms, Dynamic_recipient (List.hd_exn dyn_syms) :: idcs))
  in
  assert (List.is_empty remaining_dyn_syms);
  assert (List.is_empty remaining_ded_syms);
  Array.of_list_rev idcs

(** Computes the indexing into subformulas given the shape information of a formula. The processing
    mirrors [propagate_shapes], but [derive_projections] should only be invoked when the shapes
    are inferred already. *)
let rec derive_projections (shapes : update_step) : projections =
  (* Broadcasts symmetrically to iterate all axes. *)
  let broadcast_dims (sh1_dims : dims) (sh2_dims : dims) : dim list =
    let rec broad_back_dims accu = function
      | [], [] -> accu
      | dims, [] | [], dims -> List.rev_append dims accu
      | d1 :: dims1, d2 :: dims2 -> broad_back_dims (project_broad d1 d2 :: accu) (dims1, dims2)
    in
    match (sh1_dims, sh2_dims) with
    | Unknown, Unknown -> []
    | (Inferred dims | Given dims | Fixed dims), Unknown | Unknown, (Inferred dims | Given dims | Fixed dims)
      ->
        dims
    | (Inferred dims1 | Given dims1 | Fixed dims1), (Inferred dims2 | Given dims2 | Fixed dims2) ->
        broad_back_dims [] (List.rev dims1, List.rev dims2)
  in
  let broadcast_sh sh1 kind1 sh2 kind2 = broadcast_dims (dims_of_kind kind1 sh1) (dims_of_kind kind2 sh2) in
  let project_into_dims (product_idcs : symbol option list) (sh1_dims : dims) : axis_index list =
    let project_dim = function
      (* We preserve dedicated iterators even for 1-dimensional axes. *)
      | Some idx, { special = Dedicated _; _ } -> Iterator idx
      | _, { dim = 1; _ } | None, _ -> Fixed_idx 0
      | Some idx, _ -> Iterator idx
    in
    let rec project_dims ~is_fixed accu_idcs = function
      | [], [] -> accu_idcs
      | _idcs, [] ->
          assert (not is_fixed);
          accu_idcs
      | idx :: idcs, d1 :: dims1 -> project_dims ~is_fixed (project_dim (idx, d1) :: accu_idcs) (idcs, dims1)
      | _ ->
          (* Only reduced shapes, used internally, can have no output axes. *)
          (* FIXME: debug what's happening here. Maybe check for is_given. *)
          (* assert false *)
          accu_idcs
    in
    match sh1_dims with
    | Unknown ->
        assert (0 = List.length product_idcs);
        []
    | Inferred dims | Given dims -> project_dims ~is_fixed:false [] (List.rev product_idcs, List.rev dims)
    | Fixed dims -> project_dims ~is_fixed:true [] (List.rev product_idcs, List.rev dims)
  in
  let project_into product_idcs sh1 kind1 = project_into_dims product_idcs (dims_of_kind kind1 sh1) in
  let cur_sh = shapes.shape in
  (* Computes the corresponding dimension in the product space. *)
  let einsum_one_dim terms =
    List.fold terms ~init:(dim 1) ~f:(fun d ((_side, _axis), dims) ->
        match dims with
        | Either.First Unknown -> d
        (* | Second (_, Unknown) -> dim *)
        | First
            ( Inferred [ dim2 ]
            | Given [ dim2 ]
            | Fixed [ dim2 ] (* | Second (_, (Inferred [dim2] | Given [dim2] | Fixed [dim2])) *) )
          when equal_dim d dim2 ->
            d
        | First
            ( Inferred [ dim2 ]
            | Given [ dim2 ]
            | Fixed [ dim2 ] (* | Second (_, (Inferred [dim2] | Given [dim2] | Fixed [dim2])) *) )
          when d.dim = 1 ->
            dim2
        | First (Inferred [] | Given [] | Fixed [])
        (* | Second (_, (Inferred [] | Given [] | Fixed [])) -> dim *)
        | Second _ ->
            dim 1
        | _ -> assert false)
  in
  let map_with_dims dims idcs ~f =
    let rdims = List.rev @@ list_of_dims dims in
    let ridcs = List.take (Array.to_list @@ Array.rev idcs) @@ List.length rdims in
    List.rev @@ List.map2_exn rdims ridcs ~f
  in
  let eqs_xhs ls_xhs sh_xhs =
    let eqs =
      Map.merge ls_xhs.labels sh_xhs ~f:(fun ~key:axis -> function
        | `Both (Either.First label, dim) -> Some (label, (axis, Either.First dim))
        | `Left (Either.First label) -> Some (label, (axis, First (Inferred [])))
        | `Both (Either.Second pos, dim) -> Some (gen_label_of_axis axis, (axis, Second (pos, dim)))
        | `Left (Either.Second pos) -> Some (gen_label_of_axis axis, (axis, Second (pos, Inferred [])))
        | `Right (Given [] | Fixed [] | Inferred []) -> None
        | `Right _dim when not (bcast_of_kind axis.in_axes ls_xhs) -> assert false
        | `Right dim -> Some (gen_label_of_axis ~parsed_spec:ls_xhs axis, (axis, First dim)))
    in
    Map.of_alist_multi (module String) @@ Map.data eqs
  in
  let project_iterator d it = match d with { special = Dim | Frozen; dim = 1 } -> Fixed_idx 0 | _ -> it in
  let inferred_for_label label_iterators = function
    | Either.First label -> (
        match Map.find_exn label_iterators label with None -> Fixed_idx 0 | Some sym -> Iterator sym)
    | Second { special = _; dim = pos } -> Fixed_idx pos
  in

  (* For binary cases, we cannot rely on [cur_sh] containing all axes, since in principle it could
     have been restricted by an initial [Given] setting to efficiently implement map-reduce. *)
  match shapes.logic with
  | Terminal -> identity_projections @@ to_dims cur_sh
  | Transpose (Transpose, sh) ->
      let product_inp = broadcast_sh cur_sh Input sh Output in
      let iters_inp = List.map product_inp ~f:opt_symbol in
      let lhs_input = project_into iters_inp cur_sh Input in
      let product_out = broadcast_sh cur_sh Output sh Input in
      let iters_out = List.map product_out ~f:opt_symbol in
      let lhs_output = project_into iters_out cur_sh Output in
      let product_bch = broadcast_sh cur_sh Batch sh Batch in
      let iters_bch = List.map product_bch ~f:opt_symbol in
      let lhs_batch = project_into iters_bch cur_sh Batch in
      let rhs_input = project_into iters_out sh Input in
      let rhs_output = project_into iters_inp sh Output in
      let rhs_batch = project_into iters_bch sh Batch in
      let product_space = Array.of_list @@ List.concat [ product_bch; product_out; product_inp ] in
      let product_iterators = Array.of_list @@ List.concat [ iters_bch; iters_out; iters_inp ] in
      let project_lhs = Array.of_list @@ List.concat [ lhs_batch; lhs_output; lhs_input ] in
      let project_rhs1 = Array.of_list @@ List.concat [ rhs_batch; rhs_output; rhs_input ] in
      let product_space = Array.filter ~f:iterated product_space in
      let product_iterators = Array.filter_map ~f:Fn.id product_iterators in
      { product_space; product_iterators; project_lhs; project_rhs1; project_rhs2 = None }
  | Transpose (Pointwise_un, sh) ->
      let product_inp = broadcast_sh cur_sh Input sh Input in
      let iters_inp = List.map product_inp ~f:opt_symbol in
      let lhs_input = project_into iters_inp cur_sh Input in
      let product_out = broadcast_sh cur_sh Output sh Output in
      let iters_out = List.map product_out ~f:opt_symbol in
      let lhs_output = project_into iters_out cur_sh Output in
      let product_bch = broadcast_sh cur_sh Batch sh Batch in
      let iters_bch = List.map product_bch ~f:opt_symbol in
      let lhs_batch = project_into iters_bch cur_sh Batch in
      let rhs_input = project_into iters_inp sh Input in
      let rhs_output = project_into iters_out sh Output in
      let rhs_batch = project_into iters_bch sh Batch in
      let product_space = Array.of_list @@ List.concat [ product_bch; product_out; product_inp ] in
      let product_iterators = Array.of_list @@ List.concat [ iters_bch; iters_out; iters_inp ] in
      let project_lhs = Array.of_list @@ List.concat [ lhs_batch; lhs_output; lhs_input ] in
      let project_rhs1 = Array.of_list @@ List.concat [ rhs_batch; rhs_output; rhs_input ] in
      let product_space = Array.filter ~f:iterated product_space in
      let product_iterators = Array.filter_map ~f:Fn.id product_iterators in
      { product_space; product_iterators; project_lhs; project_rhs1; project_rhs2 = None }
  | Transpose (Permute spec, sh) ->
      let ls_rhs, ls_lhs =
        match einsum_of_spec spec with
        | ls_rhs, None, ls_lhs -> (ls_rhs, ls_lhs)
        | _ -> raise @@ Shape_error ("Invalid permutation (single-argument einsum) spec: " ^ spec, sh, cur_sh)
      in
      (* For einsum the product_space is precisely one-axis-per-label. *)
      let sh_rhs = to_axis_map sh in
      let sh_lhs = to_axis_map cur_sh in
      let eqs_p_rhs = eqs_xhs ls_rhs sh_rhs in
      let eqs_p_lhs = eqs_xhs ls_lhs sh_lhs in
      let side_eq side (axis, dims) = ((side, axis), dims) in
      let eqs_p =
        Map.merge eqs_p_rhs eqs_p_lhs ~f:(fun ~key:_label -> function
          | `Both (rhs, lhs) ->
              Some (List.rev_map_append rhs ~f:(side_eq `Rhs1) @@ List.map lhs ~f:(side_eq `Lhs))
          | `Left rhs -> Some (List.map rhs ~f:(side_eq `Rhs1))
          | `Right lhs -> Some (List.map lhs ~f:(side_eq `Lhs)))
      in
      let label_dims = Map.map eqs_p ~f:einsum_one_dim in
      let label_iterators = Map.map label_dims ~f:opt_symbol in
      (* TODO(100): here we allow the fixed index axes in the product space and avoid error
         only because [einsum_one_dim] outputs dimension 1 instead of that of the axis. *)
      let product_space = Array.of_list @@ Map.data label_dims in
      let product_iterators = Array.of_list @@ Map.data label_iterators in
      (* Inferred dims are not broadcasted-from-1, i.e. do not need Fixed_idx. But it doesn't hurt
         to treat them uniformly. *)
      let lhs_labels : axis_plab_map = axes_with_inf_labels ~all_labels:label_dims ls_lhs in
      let f = inferred_for_label label_iterators in
      let inferred_lhs = Map.map lhs_labels ~f in
      let b_lhs, i_lhs, o_lhs = axis_map_to_dims_bio inferred_lhs in
      let lhs_batch = map_with_dims cur_sh.batch b_lhs ~f:project_iterator in
      let lhs_input = map_with_dims cur_sh.input i_lhs ~f:project_iterator in
      let lhs_output = map_with_dims cur_sh.output o_lhs ~f:project_iterator in
      let rhs_labels = axes_with_inf_labels ~all_labels:label_dims ls_rhs in
      let inferred_rhs = Map.map rhs_labels ~f in
      let b_rhs, i_rhs, o_rhs = axis_map_to_dims_bio inferred_rhs in
      let rhs_batch = map_with_dims sh.batch b_rhs ~f:(fun d it -> if dim_1 d then Fixed_idx 0 else it) in
      let rhs_input = map_with_dims sh.input i_rhs ~f:(fun d it -> if dim_1 d then Fixed_idx 0 else it) in
      let rhs_output = map_with_dims sh.output o_rhs ~f:(fun d it -> if dim_1 d then Fixed_idx 0 else it) in
      let project_lhs = Array.of_list @@ List.concat [ lhs_batch; lhs_output; lhs_input ] in
      let project_rhs1 = Array.of_list @@ List.concat [ rhs_batch; rhs_output; rhs_input ] in
      let product_space = Array.filter ~f:iterated product_space in
      let product_iterators = Array.filter_map ~f:Fn.id product_iterators in
      { product_space; product_iterators; project_lhs; project_rhs1; project_rhs2 = None }
  | Broadcast (Pointwise_bin, sh1, sh2) ->
      let product_inp =
        match cur_sh.input with
        | Given _ | Unknown -> broadcast_sh sh1 Input sh2 Input
        | Fixed dims | Inferred dims -> dims
      in
      let iters_inp = List.map product_inp ~f:opt_symbol in
      let lhs1_input = project_into iters_inp cur_sh Input in
      let product_out =
        match cur_sh.output with
        | Given _ | Unknown -> broadcast_sh sh1 Output sh2 Output
        | Fixed dims | Inferred dims -> dims
      in
      let iters_out = List.map product_out ~f:opt_symbol in
      let lhs1_output = project_into iters_out cur_sh Output in
      let product_bch =
        match cur_sh.batch with
        | Given _ | Unknown -> broadcast_sh sh1 Batch sh2 Batch
        | Fixed dims | Inferred dims -> dims
      in
      let iters_bch = List.map product_bch ~f:opt_symbol in
      let lhs1_batch = project_into iters_bch cur_sh Batch in
      let rhs1_input = project_into iters_inp sh1 Input in
      let rhs1_output = project_into iters_out sh1 Output in
      let rhs1_batch = project_into iters_bch sh1 Batch in
      let rhs2_input = project_into iters_inp sh2 Input in
      let rhs2_output = project_into iters_out sh2 Output in
      let rhs2_batch = project_into iters_bch sh2 Batch in
      let product_space = Array.of_list @@ List.concat [ product_bch; product_out; product_inp ] in
      let product_iterators = Array.of_list @@ List.concat [ iters_bch; iters_out; iters_inp ] in
      let project_lhs = Array.of_list @@ List.concat [ lhs1_batch; lhs1_output; lhs1_input ] in
      let project_rhs1 = Array.of_list @@ List.concat [ rhs1_batch; rhs1_output; rhs1_input ] in
      let project_rhs2 = Some (Array.of_list @@ List.concat [ rhs2_batch; rhs2_output; rhs2_input ]) in
      let product_space = Array.filter ~f:iterated product_space in
      let product_iterators : symbol array = Array.filter_map ~f:Fn.id product_iterators in
      { product_space; product_iterators; project_lhs; project_rhs1; project_rhs2 }
  | Broadcast (Compose, sh1, sh2) ->
      (* [sh2] is the value or the function that gets applied first: [cur_sh(x) = sh1(sh2(x))].
         I.e. [cur.I = sh2.I, cur.O = sh1.O, sh2.O = sh1.I]. *)
      let product_inp = broadcast_sh cur_sh Input sh2 Input in
      let iters_inp = List.map product_inp ~f:opt_symbol in
      let lhs_input = project_into iters_inp cur_sh Input in
      let product_out = broadcast_sh cur_sh Output sh1 Output in
      let iters_out = List.map product_out ~f:opt_symbol in
      let lhs_output = project_into iters_out cur_sh Output in
      let product_bch =
        match cur_sh.batch with
        | Given _ | Unknown -> broadcast_sh sh1 Batch sh2 Batch
        | Fixed dims | Inferred dims -> dims
      in
      let iters_bch = List.map product_bch ~f:opt_symbol in
      let lhs1_batch = project_into iters_bch cur_sh Batch in

      let product_hid = broadcast_sh sh1 Input sh2 Output in
      let iters_hid = List.map product_hid ~f:opt_symbol in
      let rhs1_input = project_into iters_hid sh1 Input in
      let rhs1_output = project_into iters_out sh1 Output in
      let rhs1_batch = project_into iters_bch sh1 Batch in
      let rhs2_input = project_into iters_inp sh2 Input in
      let rhs2_output = project_into iters_hid sh2 Output in
      let rhs2_batch = project_into iters_bch sh2 Batch in
      let product_space =
        Array.of_list @@ List.concat [ product_bch; product_out; product_hid; product_inp ]
      in
      let product_iterators = Array.of_list @@ List.concat [ iters_bch; iters_out; iters_hid; iters_inp ] in
      let project_lhs = Array.of_list @@ List.concat [ lhs1_batch; lhs_output; lhs_input ] in
      let project_rhs1 = Array.of_list @@ List.concat [ rhs1_batch; rhs1_output; rhs1_input ] in
      let project_rhs2 = Some (Array.of_list @@ List.concat [ rhs2_batch; rhs2_output; rhs2_input ]) in
      let product_space = Array.filter ~f:iterated product_space in
      let product_iterators = Array.filter_map ~f:Fn.id product_iterators in
      { product_space; product_iterators; project_lhs; project_rhs1; project_rhs2 }
  | Broadcast (Einsum spec, sh1, sh2) ->
      let ls_rhs1, ls_rhs2, ls_lhs =
        match einsum_of_spec spec with
        | ls_rhs1, Some ls_rhs2, ls_lhs -> (ls_rhs1, ls_rhs2, ls_lhs)
        | _ -> raise @@ Shape_error ("Invalid (two-argument) einsum spec: " ^ spec, sh1, sh2)
      in
      (* For einsum the product_space is precisely one-axis-per-label. *)
      let sh_rhs1 = to_axis_map sh1 in
      let sh_rhs2 = to_axis_map sh2 in
      let sh_lhs = to_axis_map cur_sh in
      let eqs_rhs1 = eqs_xhs ls_rhs1 sh_rhs1 in
      let eqs_rhs2 = eqs_xhs ls_rhs2 sh_rhs2 in
      let eqs_lhs = eqs_xhs ls_lhs sh_lhs in
      let side_eq side (axis, dims) = ((side, axis), dims) in
      let eqs =
        Map.merge eqs_rhs1 eqs_lhs ~f:(fun ~key:_label -> function
          | `Both (rhs, lhs) ->
              Some (List.rev_map_append rhs ~f:(side_eq `Rhs1) @@ List.map lhs ~f:(side_eq `Lhs))
          | `Left rhs -> Some (List.map rhs ~f:(side_eq `Rhs1))
          | `Right lhs -> Some (List.map lhs ~f:(side_eq `Lhs)))
      in
      let eqs =
        Map.merge eqs_rhs2 eqs ~f:(fun ~key:_label -> function
          | `Both (rhs, more) -> Some (List.rev_map_append rhs ~f:(side_eq `Rhs2) more)
          | `Left rhs -> Some (List.map rhs ~f:(side_eq `Rhs2))
          | `Right more -> Some more)
      in
      let label_dims = Map.map eqs ~f:einsum_one_dim in
      let label_iterators = Map.map label_dims ~f:opt_symbol in
      let product_space = Array.of_list @@ Map.data label_dims in
      let product_iterators = Array.of_list @@ Map.data label_iterators in
      (* Inferred dims are not broadcasted-from-1, i.e. do not need Fixed_idx. But it doesn't hurt
         to treat them uniformly. *)
      let lhs_labels = axes_with_inf_labels ~all_labels:label_dims ls_lhs in
      let f = inferred_for_label label_iterators in
      let inferred_lhs = Map.map lhs_labels ~f in
      let b_lhs, i_lhs, o_lhs = axis_map_to_dims_bio inferred_lhs in
      let lhs_batch = map_with_dims cur_sh.batch b_lhs ~f:project_iterator in
      let lhs_input = map_with_dims cur_sh.input i_lhs ~f:project_iterator in
      let lhs_output = map_with_dims cur_sh.output o_lhs ~f:project_iterator in
      let rhs1_labels = axes_with_inf_labels ~all_labels:label_dims ls_rhs1 in
      let inferred_rhs1 = Map.map rhs1_labels ~f in
      let b_rhs1, i_rhs1, o_rhs1 = axis_map_to_dims_bio inferred_rhs1 in
      let rhs1_batch = map_with_dims sh1.batch b_rhs1 ~f:project_iterator in
      let rhs1_input = map_with_dims sh1.input i_rhs1 ~f:project_iterator in
      let rhs1_output = map_with_dims sh1.output o_rhs1 ~f:project_iterator in
      let rhs2_labels = axes_with_inf_labels ~all_labels:label_dims ls_rhs2 in
      let inferred_rhs2 = Map.map rhs2_labels ~f in
      let b_rhs2, i_rhs2, o_rhs2 = axis_map_to_dims_bio inferred_rhs2 in
      let rhs2_batch = map_with_dims sh2.batch b_rhs2 ~f:project_iterator in
      let rhs2_input = map_with_dims sh2.input i_rhs2 ~f:project_iterator in
      let rhs2_output = map_with_dims sh2.output o_rhs2 ~f:project_iterator in
      let project_lhs = Array.of_list @@ List.concat [ lhs_batch; lhs_output; lhs_input ] in
      let project_rhs1 = Array.of_list @@ List.concat [ rhs1_batch; rhs1_output; rhs1_input ] in
      let project_rhs2 = Some (Array.of_list @@ List.concat [ rhs2_batch; rhs2_output; rhs2_input ]) in
      let product_space = Array.filter ~f:iterated product_space in
      let product_iterators = Array.filter_map ~f:Fn.id product_iterators in
      { product_space; product_iterators; project_lhs; project_rhs1; project_rhs2 }
  | Broadcast (Dynamic_index { over_kind; from_left; other_axes_pointwise; _ }, sh1, sh2) ->
      let subs : int = (Option.value ~default:(dim 1) @@ List.last @@ list_of_dims sh2.output).dim in
      let output : dims = map_dims sh2.output ~f:List.drop_last_exn in
      let reduced_sh2 : t = { sh2 with output } in
      let reduced_sh2 : t =
        { reduced_sh2 with axis_labels = shift_axes_of_kind AxisKey.Output sh2 ~f:(( - ) 1) }
      in
      let sh1_size : int = List.length @@ list_of_dims @@ dims_of_kind over_kind sh1 in
      let reduced_sh1, targets_and_skipped, target_dims, logic =
        if from_left then
          let n_par_axes : int = dims_of_kind over_kind sh1 |> list_of_dims |> count_special_among subs in
          let reduced_dims (over_dims : dims) : dims = map_dims over_dims ~f:(drop_keeping_special subs) in
          let reduced_sh1 : t = map_over_kind over_kind ~f:reduced_dims sh1 in
          let target_dims : dim array =
            dims_of_kind over_kind sh1 |> list_of_dims |> take_skipping_special subs |> Array.of_list
          in
          let targets_and_skipped : dim array =
            dims_of_kind over_kind sh1 |> list_of_dims
            |> take_keeping_special ~debug_sh1:sh1 ~debug_sh2:sh2 subs
            |> Array.of_list
          in
          let drop_left (from_end : int) : int =
            if from_end > sh1_size - subs - n_par_axes then -1 else from_end
          in
          let reduced_sh1 : t =
            { reduced_sh1 with axis_labels = shift_axes_of_kind over_kind sh1 ~f:drop_left }
          in
          if other_axes_pointwise then
            ( reduced_sh1,
              targets_and_skipped,
              target_dims,
              Broadcast (Pointwise_bin, reduced_sh1, reduced_sh2) )
          else
            let extended_sh : t = append_all_axes ~prefix:reduced_sh2 ~main:reduced_sh1 () in
            (reduced_sh1, targets_and_skipped, target_dims, Transpose (Pointwise_un, extended_sh))
        else (* FIXME: NOT IMPLEMENTED YET *)
          failwith "NOT IMPLEMENTED YET"
      in
      let n_par_axes : int =
        Array.count targets_and_skipped ~f:(function { special = Dedicated _; _ } -> true | _ -> false)
      in
      let drop_left_par (proj : axis_index array) : axis_index array * symbol array =
        let result : axis_index array =
          Array.sub ~pos:n_par_axes ~len:(Array.length proj - n_par_axes) proj
        in
        let ded_syms : symbol array =
          Array.sub ~pos:0 ~len:n_par_axes proj
          |> Array.filter_map ~f:(function Iterator s when is_dedicated_any s -> Some s | _ -> None)
        in
        (result, ded_syms)
      in
      let drop_right_par (proj : axis_index array) : axis_index array * symbol array =
        let result : axis_index array = Array.sub ~pos:0 ~len:(Array.length proj - n_par_axes) proj in
        let ded_syms : symbol array =
          Array.sub ~pos:(Array.length proj - n_par_axes) ~len:n_par_axes proj
          |> Array.filter_map ~f:(function Iterator s when is_dedicated_any s -> Some s | _ -> None)
        in
        (result, ded_syms)
      in
      let update_other_axes : update_step = { shape = cur_sh; logic } in
      let reduced_projections : projections = derive_projections update_other_axes in
      let dynamic_syms : symbol array = Array.init subs ~f:(fun _ -> get_symbol ()) in
      let proj_b, proj_i, proj_o = indices_bio reduced_sh1 reduced_projections.project_rhs1 in
      let project_rhs1 : axis_index array =
        if from_left then
          match over_kind with
          | AxisKey.Batch ->
              let proj_b, dedicated_syms = drop_left_par proj_b in
              Array.concat
                [
                  project_dyn_indexing ~dynamic_syms ~dedicated_syms targets_and_skipped;
                  proj_b;
                  proj_o;
                  proj_i;
                ]
          | AxisKey.Input ->
              let proj_i, dedicated_syms = drop_left_par proj_i in
              Array.concat
                [
                  proj_b;
                  proj_o;
                  project_dyn_indexing ~dynamic_syms ~dedicated_syms targets_and_skipped;
                  proj_i;
                ]
          | AxisKey.Output ->
              let proj_o, dedicated_syms = drop_left_par proj_o in
              Array.concat
                [
                  proj_b;
                  project_dyn_indexing ~dynamic_syms ~dedicated_syms targets_and_skipped;
                  proj_o;
                  proj_i;
                ]
        else
          match over_kind with
          | AxisKey.Batch ->
              let proj_b, dedicated_syms = drop_right_par proj_b in
              Array.concat
                [
                  proj_b;
                  project_dyn_indexing ~dynamic_syms ~dedicated_syms targets_and_skipped;
                  proj_o;
                  proj_i;
                ]
          | AxisKey.Input ->
              let proj_i, dedicated_syms = drop_right_par proj_i in
              Array.concat
                [
                  proj_b;
                  proj_o;
                  proj_i;
                  project_dyn_indexing ~dynamic_syms ~dedicated_syms targets_and_skipped;
                ]
          | AxisKey.Output ->
              let proj_o, dedicated_syms = drop_right_par proj_o in
              Array.concat
                [
                  proj_b;
                  proj_o;
                  project_dyn_indexing ~dynamic_syms ~dedicated_syms targets_and_skipped;
                  proj_i;
                ]
      in
      let project_rhs2 =
        Some
          (Array.append
             (Option.value_exn reduced_projections.project_rhs2)
             [| Dynamic_provider { idcs = dynamic_syms; target_dims } |])
      in
      { reduced_projections with project_rhs1; project_rhs2 }

let backprop1 projections =
  { projections with project_lhs = projections.project_rhs1; project_rhs1 = projections.project_lhs }

let backprop2 projections =
  match projections.project_rhs2 with
  | None -> invalid_arg "Shape.backprop2: unary shapes (project_rhs2 is None)"
  | Some project_rhs2 ->
      { projections with project_lhs = project_rhs2; project_rhs2 = Some projections.project_lhs }

let backprop_unary projections =
  {
    projections with
    project_lhs = projections.project_rhs1;
    project_rhs1 = projections.project_lhs;
    project_rhs2 = Some projections.project_lhs;
  }

let derive_index ~product_syms ~(projection : axis_index array) =
  let sym_to_i =
    Array.mapi product_syms ~f:(fun i (Symbol s) -> (s, i)) |> Array.to_list |> Map.of_alist_exn (module Int)
  in
  let positions =
    Array.map projection ~f:(function
      | Iterator (Symbol s) when Map.mem sym_to_i s -> Either.First (Map.find_exn sym_to_i s)
      | (Fixed_idx _ | Dynamic_provider _ | Dynamic_recipient _ | Frozen_recipient _) as it -> Second it
      | Iterator s as it ->
          assert (is_dedicated_any s);
          Second it)
  in
  fun ~product -> Array.map positions ~f:(function First p -> product.(p) | Second it -> it)

let make ?batch_dims ?input_dims ?output_dims ?axis_labels ?deduced ~id () =
  let input = match input_dims with None -> Unknown | Some dims -> Given dims in
  let output = match output_dims with None -> Unknown | Some dims -> Given dims in
  let batch = match batch_dims with None -> Unknown | Some dims -> Given dims in
  let deduce_within_shape_constraints = Option.value deduced ~default:Not_constrained in
  let axis_labels =
    match axis_labels with
    | None -> Map.empty (module AxisKey)
    | Some spec ->
        Map.map (axis_labels_of_spec spec).labels ~f:(function
          | Either.First label -> label
          | Second { special; dim } ->
              (match special with
              | Dim -> ""
              | Frozen -> "frozen "
              | Dedicated Task_id -> "parallel "
              | Dedicated Sample_num -> "minibatch ")
              ^ Int.to_string dim)
  in
  { input; output; batch; deduce_within_shape_constraints; axis_labels; id }

let to_string_hum ?(style = `Axis_size) sh =
  let n_outputs = List.length @@ list_of_dims @@ dims_of_kind Output sh in
  let n_batch = List.length @@ list_of_dims @@ dims_of_kind Batch sh in
  let dims_to_string kind =
    let dims = list_of_dims @@ dims_of_kind kind sh in
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

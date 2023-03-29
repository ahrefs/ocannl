(** Tensor shape types and inference. *)

open Base

(** An index pointing to any of a shape's axes, including the kind of the axis ([Batch, Input, Output])
    and the position (which is counted from the end to facilitate broadcasting).
    
    Note the following inconsistency due to differing conventions in function notation and matrix notation:
    for label specifications and einsum notation, we write "batch|inputs->outputs", but when we convert
    a shape to an [Code] index we do it in the order [[batch; outputs; inputs]]. *)
module AxisKey = struct
  module T = struct
    type kind = 
      | Batch
      | Input
      | Output
    [@@deriving equal, compare, sexp, variants]
    type t = {
      in_axes: kind;
      from_end: int
      (** Axes are indexed from the end, to avoid reindexing when broadcasting; starting with [1]. *)
     } [@@deriving equal, compare, sexp]
     let to_string key = 
      (match key.in_axes with Batch -> "bch" | Input -> "inp" | Output -> "out") ^
      Int.to_string key.from_end
  end
  include T
  include Comparator.Make(T)
end

type 'a axis_map = 'a Map.M(AxisKey).t [@@deriving compare, sexp]

(** The labels are strings assigned to [AxisKey] axes. Moreover the [bcast_] fields represent whether
    additional leading axes are allowed (corresponding to the dot-ellipsis syntax for broadcasting).
    The [given_] fields count the number of specified axes of the corresponding kind in [labels].
    The [wildcard_] fields represent whether a wildcard suffix `%` was used, i.e. additional trailing
    axes are allowed (which in the case of `einsum` or `einsum1` operations should be summed over). *)
type parsed_axis_labels = {
  bcast_batch: bool;
  bcast_input: bool;
  bcast_output: bool;
  wildcard_batch: bool;
  wildcard_input: bool;
  wildcard_output: bool;
  given_batch: int;
  given_input: int;
  given_output: int;
  labels: string axis_map
} [@@deriving compare, sexp, fields]

let bcast_of_kind = function
  | AxisKey.Batch -> bcast_batch
  | AxisKey.Input -> bcast_input
  | AxisKey.Output -> bcast_output

let wildcard_of_kind = function
  | AxisKey.Batch -> wildcard_batch
  | AxisKey.Input -> wildcard_input
  | AxisKey.Output -> wildcard_output

let given_of_kind = function
  | AxisKey.Batch -> given_batch
  | AxisKey.Input -> given_input
  | AxisKey.Output -> given_output

type dims =
| Given of int list
(** User-provided dimensions. They will not change but will be broadcasted to bigger sizes. *)
| Fixed of int list
(** User-provided dimensions that will fail if used in a different size context, even if broadcastable.
    Note that [Operation.stop_broadcast] implements additional shape logic:
    it converts the (bottom-up i.e. partially inferred) shape into a [Fixed] variant. *)
| Inferred of int list
(** Dimensions that will itself change to a bigger size: they adapt to the broadcasted size. *)
| Unknown
(** User-provided and will be replaced through inference. Prefer using [Unknown] to [Inferred []]. *)
[@@deriving compare, sexp, variants]

type deduce_dims =
| Not_constrained
| Input_equals_output
| Input_output_scale of float
[@@deriving compare, sexp, variants]

(** Converts dimensions according to the specification. Note that scalar axes (1D) are not scaled,
    for compatibility with broadcasting.
    
    Note that in practice [from] will be [Unknown] or [Inferred] dimensions, making it of little relevance
    how the [Given] and [Fixed] cases are interpreted here. *)
let deduce_dims from: deduce_dims -> dims = function
| Not_constrained -> Unknown
| Input_equals_output ->
  (match from with
  | Given dims | Fixed dims -> Inferred dims
  | Inferred _ | Unknown -> from)
| Input_output_scale sc ->
  match from with
  | Unknown -> Unknown
  | (Given dims | Fixed dims | Inferred dims) -> Inferred (List.map dims ~f:(
      fun d -> if d = 1 then 1 else Float.(iround_exn ~dir:`Up @@ sc * of_int d)))

(** The datatype from which the actual Code shapes are computed.

    Mutability is sufficient to perform inference, since there is no need for backtracking and
    no explicit unification variables for now. [Unknown] stands for "not yet specified". *)
type t = {
  mutable batch: dims;
  mutable input: dims;
  mutable output: dims;
  mutable axis_labels: string axis_map;
  deduce_within_shape_constraints: deduce_dims;
  (** Intended for terminal node cases where both [input] and [output] are initially
      unknown. It makes it trivial to implement dimension-preserving hidden layers: just set
      [deduce_within_shape_constraints=Input_equals_output]. *)
  id: int;
  (** A node that has the same shape as this shape. *)
} [@@deriving fields, sexp]

let dims_of_kind = function
  | AxisKey.Batch -> batch
  | AxisKey.Input -> input
  | AxisKey.Output -> output

type compose_type =
  | Pointwise_bin
  (** NumPy-style broadcast matching batch, input and output axes, e.g. as in [s1 + s2]. *)
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
  | Transpose
  (** Swaps inputs and outputs of a shape, preserves batch axes. *)
  | Pointwise_un
  (** Preserves the shape. *)
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
  of other labels.
  
  The label ["%"] is only allowed as the last axis of a kind (i.e. first from-end). It is a
  non-broadcasting wildcard. *)
let axis_labels_of_spec spec: parsed_axis_labels =
  let check_dot s =
    if String.length s > 3 && Option.is_some @@ String.substr_index ~pos:3 s ~pattern:"..."
    then invalid_arg ("axis_labels_of_spec: dot only allowed at first axis of a kind: "^spec)
    else if String.is_prefix s ~prefix:"..." then true, String.drop_prefix s 3
    else false, s in
  let check_wildcard s =
    if String.length s > 0 &&
       Option.exists (String.index s '%') ~f:(fun p -> p < String.length s - 1)
    then invalid_arg ("axis_labels_of_spec: % only allowed at last axis of a kind: "^spec)
    else if String.is_suffix s ~suffix:"%" then true, String.drop_suffix s 1
    else false, s in
  let parse spec in_axes =
    let bcast, spec = check_dot @@ String.strip spec in
    let wildcard, spec = check_wildcard spec in
    bcast, wildcard,
    if List.exists ~f:(String.contains spec) [' '; ','; '('; ')'] then
      let labels = String.split_on_chars spec ~on:[' '; ','; '('; ')'] |>
                   List.filter ~f:(fun s -> not @@ String.is_empty s) in
      let labels_num = List.length labels in
      labels_num,
      List.foldi labels ~init:(Map.empty (module AxisKey))
        ~f:(fun from_start labels label ->
          if String.equal label "_" then labels
          else Map.add_exn labels ~key:AxisKey.{in_axes; from_end=labels_num - from_start} ~data:label)
    else
      let labels_num = String.length spec in
      labels_num,
      String.foldi spec ~init:(Map.empty (module AxisKey))
        ~f:(fun from_start labels label -> Map.add_exn labels 
               ~key:AxisKey.{in_axes; from_end=labels_num - from_start}
               ~data:(String.of_char label)) in
  let batch_spec, spec =
    match String.substr_index spec ~pattern:"|" with
    | Some end_bch -> String.sub ~pos:0 ~len:end_bch spec,
                      String.sub ~pos:(end_bch+1) ~len:(String.length spec - end_bch - 1) spec
    | None -> "", spec in
  let input_spec, output_spec =
    match String.substr_index spec ~pattern:"->" with
    | Some end_inp -> String.sub ~pos:0 ~len:end_inp spec,
                      String.sub ~pos:(end_inp+2) ~len:(String.length spec - end_inp - 2) spec
    | None -> "", spec in
  let batch_spec: string = batch_spec in
  let input_spec: string = input_spec in
  let output_spec: string = output_spec in
  let bcast_batch, wildcard_batch, (given_batch, batch_labels) = parse batch_spec Batch in
  let bcast_input, wildcard_input, (given_input, input_labels) = parse input_spec Input in
  let bcast_output, wildcard_output, (given_output, output_labels) = parse output_spec Output in
  let labels =
    match Map.append ~lower_part:input_labels ~upper_part:output_labels with
    | `Ok m -> (match Map.append ~lower_part:batch_labels ~upper_part:m with `Ok r -> r | _ -> assert false)
    | _ -> assert false in
  { bcast_batch; bcast_input; bcast_output; wildcard_batch; wildcard_input;
    wildcard_output; given_batch; given_input; given_output; labels }

let einsum_of_spec spec =
  let rhs_spec, lhs_spec =
    match String.substr_index spec ~pattern:"=>" with
    | Some endp -> String.sub ~pos:0 ~len:endp spec,
                      String.sub ~pos:(endp+2) ~len:(String.length spec - endp - 2) spec
    | None -> "", spec in
  (if String.is_empty lhs_spec then invalid_arg (
    "einsum_of_spec: missing the result spec in "^rhs_spec));
  (if String.is_empty rhs_spec then invalid_arg (
    "einsum_of_spec: missing the argument spec in "^rhs_spec));
  let rhs1_spec, rhs2_spec =
    match String.substr_index spec ~pattern:";" with
    | Some endp -> String.sub ~pos:0 ~len:endp spec,
                      String.sub ~pos:(endp+1) ~len:(String.length spec - endp - 1) spec
    | None -> spec, "" in
  let lhs_ls = axis_labels_of_spec lhs_spec in
  let rhs1_ls = axis_labels_of_spec rhs1_spec in
  if String.is_empty rhs2_spec then `Permute_unop (rhs1_ls, lhs_ls)
  else `Permute_binop (rhs1_ls, axis_labels_of_spec rhs2_spec, lhs_ls) 

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

(** Data required for a shape inference update step. A step should equilibrate information, passing it both
    top-down and bottom-up. The child should be identifiable within the parent via physical equality
    (allowing that a child fills both slots of a binary parent). *)
type update_step = {
  shape: t;
  logic: logic;
} [@@deriving sexp]

exception Shape_error of string * t * t [@@deriving sexp]

let list_of_dims = function
  | Given ls | Fixed ls | Inferred ls -> ls
  | Unknown -> []

(** Given a fully-inferred shape, maps axes to their corresponding positions in an index using the
    [Shape.to_dims] semantics. *)
let axis_keys_to_idcs (sh: t): int axis_map =
  let b_dims = match sh.batch with
    | Unknown -> raise @@ Shape_error ("Batch dimensions still unknown", sh, sh)
    | Inferred dims | Given dims | Fixed dims ->
      (* Enumerate axes backwards. *)
      Array.of_list_mapi dims ~f:(fun i _ -> AxisKey.{ in_axes=Batch; from_end=i + 1 }) in
  let i_dims = match sh.input with
    | Unknown -> raise @@ Shape_error ("Input dimensions still unknown", sh, sh)
    | Inferred dims | Given dims | Fixed dims ->
      Array.of_list_mapi dims ~f:(fun i _ -> AxisKey.{ in_axes=Input; from_end=i + 1 }) in
  let o_dims = match sh.output with
    | Unknown -> raise @@ Shape_error ("Output dimensions still unknown", sh, sh)
    | Inferred dims | Given dims | Fixed dims ->
      Array.of_list_mapi dims ~f:(fun i _ -> AxisKey.{ in_axes=Output; from_end=i + 1 }) in
  let idcs = Array.concat [i_dims; o_dims; b_dims] in
  Array.rev_inplace idcs;
  Map.of_alist_exn (module AxisKey) @@ Array.to_list @@ Array.mapi idcs ~f:(fun i key -> key, i)

(** Converts an axes-keyed map into three arrays of values: batch axes, input axes, output axes.
    If the map is incomplete, the result might be invalid: gaps in the array are filled with an arbitrary
    one of the provided values. *)
let axis_map_to_dims_bio (type a) ?(default:a option) (idcs: a axis_map) =
  if Map.is_empty idcs then [||], [||], [||]
  else
    let witness =
      match default with
      | Some witness -> witness
      | None -> snd @@ Map.min_elt_exn idcs in
    let bch_axes, other = Map.partition_mapi idcs ~f:(
        fun ~key:{in_axes; _} ~data ->
          if AxisKey.is_batch in_axes then Either.First data else Either.Second data) in
    let inp_axes, out_axes = Map.partition_mapi other ~f:(
        fun ~key:{in_axes; _} ~data ->
          if AxisKey.is_input in_axes then Either.First data else Either.Second data) in
    let bch_axes = Map.to_alist bch_axes |> List.map ~f:(fun ({from_end=i; _}, v) -> i, v) in
    let bch_size = List.fold bch_axes ~init:0 ~f:(fun accu (i,_) -> max i accu) in
    let bch = Array.create ~len:bch_size witness in
    List.iter bch_axes ~f:(fun (i,v) -> bch.(bch_size - i) <- v);
    let inp_axes = Map.to_alist inp_axes |> List.map ~f:(fun ({from_end=i; _}, v) -> i, v) in
    let inp_size = List.fold inp_axes ~init:0 ~f:(fun accu (i,_) -> max i accu) in
    let inp = Array.create ~len:inp_size witness in
    List.iter inp_axes ~f:(fun (i,v) -> inp.(inp_size - i) <- v);
    let out_axes = Map.to_alist out_axes |> List.map ~f:(fun ({from_end=i; _}, v) -> i, v) in
    let out_size = List.fold out_axes ~init:0 ~f:(fun accu (i,_) -> max i accu) in
    let out = Array.create ~len:out_size witness in
    List.iter out_axes ~f:(fun (i,v) -> out.(out_size - i) <- v);
    bch, inp, out

(** Converts an axes-keyed map into an array of values using the [Shape.to_dims] semantics of axes.
    If the map is incomplete and the [~default] is not given, the result might be invalid: gaps in
    the array are filled with an arbitrary one of the provided values. *)
let axis_map_to_dims_index (type a) ?(default:a option) (idcs: a axis_map): a array =
  let bch, inp, out = axis_map_to_dims_bio ?default idcs in
  Array.concat [bch; out; inp]

(** Splits the dimensions of a shape into a map from axes, putting at most one number in a [dims] of
    an axis. An empty [dims] list is an end-of-list sentinel: means that there are one fewer axes
    of the particular kind. *)
let to_axis_map (sh: t): dims axis_map =
  let kind_dims kind = match dims_of_kind kind sh with
    | Unknown ->
      [AxisKey.{ in_axes=kind; from_end=1 }, Unknown]
    | Inferred dims ->
      let n_dims = List.length dims in
      (AxisKey.{ in_axes=kind; from_end=n_dims + 1 }, Inferred [])::
      List.rev_mapi dims ~f:(fun i d -> AxisKey.{ in_axes=kind; from_end=n_dims - i }, Inferred [d])
    | Given dims ->
      let n_dims = List.length dims in
      (AxisKey.{ in_axes=kind; from_end=n_dims + 1 }, Given [])::
      List.rev_mapi dims ~f:(fun i d -> AxisKey.{ in_axes=kind; from_end=n_dims - i }, Given [d])
    | Fixed dims ->
      let n_dims = List.length dims in
      (AxisKey.{ in_axes=kind; from_end=n_dims + 1 }, Fixed [])::
      List.rev_mapi dims ~f:(fun i d -> AxisKey.{ in_axes=kind; from_end=n_dims - i }, Fixed [d]) in
  let b_dims = kind_dims Batch in
  let i_dims = kind_dims Input in
  let o_dims = kind_dims Output in
  Map.of_alist_exn (module AxisKey) @@ List.concat [b_dims; i_dims; o_dims]

(* Design choice: tensor shapes are decided while code is constructed, although not immediately.
   Due to mutable updates during shape inference, it is not possible to reuse the same formula with
   different shapes. The inference is finalized by invoking the [Formula.subtree_shape_updates] once
   on the root formula. *)
   
(** Generate a label into the broadcasted axis given an einsum-like spec. Axes that are part of the spec
    do not count, so that we use the labels to align axes across different shapes (lhs, rhs1, rhs2). *)
let gen_label_of_axis parsed_spec axis =
  let open AxisKey in
  let idx = axis.from_end - (given_of_kind axis.in_axes parsed_spec) in
  (match axis.in_axes with Batch -> "__b" | Input -> "__i" | Output -> "__o") ^ Int.to_string idx

let set_dims_type sh typ =
  sh.batch <- typ (list_of_dims sh.batch);
  sh.input <- typ (list_of_dims sh.input);
  sh.output <- typ (list_of_dims sh.output)

(** Augment the pseudo-labels map of an einsum notation with the generated labels for broadcasted axes. *)
let axes_with_inf_labels ~all_labels ls_xhs =
  let rec loop more kind accu =
    let offset = given_of_kind kind ls_xhs in
    let axis = AxisKey.{ in_axes=kind; from_end=offset + more } in
    let label = gen_label_of_axis ls_xhs axis in
    if not @@ Map.mem all_labels label then accu
    else loop (more+1) kind @@ Map.add_exn accu ~key:axis ~data:label in
  let see kind accu =
    if bcast_of_kind kind ls_xhs then loop 1 kind accu
    else accu in
  AxisKey.(see Batch @@ see Input @@ see Output @@ ls_xhs.labels)

let is_given_or_fixed dims = is_given dims || is_fixed dims

(* module Debug_runtime = Minidebug_runtime.PrintBox(struct let debug_ch = Stdio.stdout end) *)

(** Performs a local step of shape inference, propagates information into and out of the parent shape
    and the child shape(s). *)
let propagate_shapes (update: update_step) =
  let pointwise_labels debug1 debug2 ls1 ls2 = Map.merge ls1 ls2 ~f:(fun ~key ->
    function
    | `Both (l1, l2) ->
      if String.equal l1 l2 then Some l1
      else
        let error = "Axis label mismatch: "^l1^" vs "^l2^" for "^
                    (Sexp.to_string_hum @@ AxisKey.sexp_of_t key) in
         raise @@ Shape_error (error, debug1, debug2)
    | `Right l | `Left l -> Some l
  ) in
  let broad_dim ~fixed_left ~fixed_right debug1 debug2 axis_key label = function
    | d1, d2 when d1 = d2 -> d1
    | 1, d when not fixed_left -> d
    | d, 1 when not fixed_right -> d
    | d1, d2 ->
      let opt_label = match label with None -> "" | Some l -> " ("^l^")" in
      let error = "Dimension mismatch for axis "^AxisKey.to_string axis_key^opt_label^": "^
                  Int.to_string d1^" vs. "^Int.to_string d2 in
      raise @@ Shape_error (error, debug1, debug2) in
  (* If initially [lhs] is [Unknown], [rhs1] is [sh1] and [rhs2] is [sh2],
     then [lhs] becomes [broadcast_dims sh1 sh2]. *)
  let broadcast_dims sh1 sh2 kind labels sh1_dims sh2_dims =
    let rec broad_back_dims ~fixed_left ~fixed_right accu i = function
    | [], [] -> accu
    | [], dims when not fixed_left -> List.rev_append dims accu
    | dims, [] when not fixed_right -> List.rev_append dims accu
    | [], _ | _, [] ->
      let key = AxisKey.{in_axes=kind; from_end=i} in
      let opt_label = match Map.find labels key with None -> "" | Some l -> " ("^l^")" in
      let error = "Different number of axes around from-end "^AxisKey.to_string key^opt_label in
      raise @@ Shape_error (error, sh1, sh2)
    | d1::dims1, d2::dims2 ->
      let key = AxisKey.{in_axes=kind; from_end=i} in
      broad_back_dims ~fixed_left ~fixed_right
        (broad_dim ~fixed_left ~fixed_right sh1 sh2 key (Map.find labels key) (d1, d2)::accu)
        (i+1) (dims1, dims2) in
    let broadcast_dims ~dims1 ~dims2 =
      broad_back_dims ~fixed_left:(is_fixed sh1_dims) ~fixed_right:(is_fixed sh2_dims)
                    [] 1 (List.rev dims1, List.rev dims2) in
    match sh1_dims, sh2_dims with
      | Unknown, Unknown -> Unknown
      | (Inferred dims | Given dims| Fixed dims), Unknown
      | Unknown, (Inferred dims | Given dims | Fixed dims) -> Inferred dims
      | Fixed dims1, Fixed dims2 ->
        Fixed (broadcast_dims ~dims1 ~dims2)
      | (Given dims1 | Fixed dims1), (Given dims2 | Fixed dims2) ->
        Given (broadcast_dims ~dims1 ~dims2)
      | (Inferred dims1 | Given dims1 | Fixed dims1), (Inferred dims2 | Given dims2 | Fixed dims2) ->
        Inferred (broadcast_dims ~dims1 ~dims2) in
  let cur_sh = update.shape in
  (* Note: does not work with arbitrary permutation as in einsum. *)
  let update_labels sh1 to_kind sh2 from_kind =
    pointwise_labels sh1 sh2 sh1.axis_labels @@
    Map.map_keys_exn (module AxisKey) ~f:(fun k -> {k with in_axes=to_kind}) @@
    Map.filter_keys sh2.axis_labels ~f:AxisKey.(fun k -> equal_kind k.in_axes from_kind) in
  let broadcast_into ?(det=false) to_sh to_kind from_sh from_kind =
    match dims_of_kind to_kind to_sh, dims_of_kind from_kind from_sh with
    | (Given _ | Fixed _) as into_dims, from_dims ->
      ignore @@
        broadcast_dims to_sh from_sh to_kind to_sh.axis_labels into_dims from_dims;
      into_dims
    | into_dims, from_dims ->
      to_sh.axis_labels <- update_labels to_sh to_kind from_sh from_kind;
      let result = broadcast_dims to_sh from_sh to_kind to_sh.axis_labels into_dims from_dims in
      match det, from_dims, result with
      | true, Fixed _, Inferred dims -> Fixed dims
      | true, Given _, Inferred dims -> Given dims
      | _ -> result in
  let einsum_one_dim_opt debug_spec debug1 debug2 label terms =
    snd @@
    List.fold terms ~init:(false, None) ~f:(fun (is_fixed, dim as accu) (_axis, dims) ->
      match dim, dims with
        | _, (Inferred (_::_::_) | Given (_::_::_) | Fixed (_::_::_)) -> assert false
        | None, Unknown -> assert (not is_fixed); false, None
        | Some _, Unknown -> accu
        | None, (Inferred [dim2] | Given [dim2]) -> assert (not is_fixed); false, Some dim2
        | None, (Fixed [dim2]) -> assert (not is_fixed); true, Some dim2
        | Some dim1, (Inferred [dim2] | Given [dim2]) when dim1 = dim2 -> accu
        | Some dim1, (Fixed [dim2]) when dim1 = dim2 -> true, dim
        | Some 1, (Inferred [dim2] | Given [dim2]) when not is_fixed -> false, Some dim2
        | Some dim1, (Inferred [dim2] | Given [dim2] | Fixed [dim2]) ->
          raise @@ Shape_error ("Dimension mismatch "^Int.to_string_hum dim1^" vs. "^
                                Int.to_string_hum dim2^" for einsum pseudo-label "^label^" of "^debug_spec^
                                (if dim1 = 1 || dim2 = 1 then " (broadcast prevented)" else ""),
                                debug1, debug2)
        | _, Fixed [] ->
          raise @@ Shape_error ("Too few fixed axes at einsum pseudo-label "^label^" of "^debug_spec^
                                " (broadcast prevented)", debug1, debug2)
        | _, (Inferred [] | Given []) when is_fixed ->
          raise @@ Shape_error ("Too few actual axes at einsum pseudo-label "^label^" of "^debug_spec^
                                " (broadcast prevented)", debug1, debug2)
        | _, (Inferred [] | Given []) -> accu
      ) in
  let einsum_one_dim debug_spec debug1 debug2 ~key ~data =
    match einsum_one_dim_opt debug_spec debug1 debug2 key data with
    | None -> 1 (* which can still be expanded/broadcasted *)
    | Some dim -> dim in
  let to_inferred = Fn.compose inferred Array.to_list in
  let eqs_xhs debug_spec debug_sh ls_xhs sh_xhs =
    let eqs = Map.merge ls_xhs.labels sh_xhs ~f:(fun ~key:axis -> function
        | `Both (label, dim) -> Some (label, (axis, dim))
        | `Left label -> Some (label, (axis, Inferred []))
        | `Right (Given [] | Fixed [] | Inferred []) -> None
        | `Right _dim when not (bcast_of_kind axis.in_axes ls_xhs) -> raise @@ Shape_error (
            "Too many axes to permute -- spec too short: "^debug_spec, debug_sh, cur_sh)
            (* Note: the too-few-axes error is reported when einsum_one_dim processes the result. *)
        | `Right dim -> Some (gen_label_of_axis ls_xhs axis, (axis, dim))) in
    Map.of_alist_multi (module String) @@ Map.data eqs in
  let pseudo_to_labels_xhs xhs_labels sh =
    Map.merge xhs_labels sh.axis_labels ~f:(fun ~key:_ -> function
        | `Both (pseudo, label) -> Some (pseudo, label)
        | `Left _pseudo -> None
        | `Right _label -> assert false) |>
    Map.data |> Map.of_alist_exn (module String) in
  let all_axis_labels debug1 debug2 debug_spec pseudo_to_labels_1 pseudo_to_labels_2 =
    Map.merge pseudo_to_labels_1 pseudo_to_labels_2 ~f:(fun ~key:pseudo -> function
        | `Both (l1, l2) when String.equal l1 l2 -> Some l1
        | `Left l | `Right l -> Some l
        | `Both (l1, l2) ->
          let error = "Axis label mismatch: "^l1^" vs "^l2^" for pseudo label "^pseudo^
                      " of spec "^debug_spec in
          raise @@ Shape_error (error, debug1, debug2)
      ) in
  match update.logic with
  | Terminal -> ()
  | Transpose (Transpose, sh) ->
    cur_sh.input <- broadcast_into ~det:true cur_sh Input sh Output;
    cur_sh.output <- broadcast_into ~det:true cur_sh Output sh Input;
    cur_sh.batch <- broadcast_into ~det:true cur_sh Batch sh Batch;
    sh.input <- broadcast_into sh Input cur_sh Output;
    sh.output <- broadcast_into sh Output cur_sh Input;
    sh.batch <- broadcast_into sh Batch cur_sh Batch;

  | Transpose (Pointwise_un, sh) ->
    cur_sh.input <- broadcast_into ~det:true cur_sh Input sh Input;
    cur_sh.output <- broadcast_into ~det:true cur_sh Output sh Output;
    cur_sh.batch <- broadcast_into ~det:true cur_sh Batch sh Batch;
    sh.input <- broadcast_into sh Input cur_sh Input;
    sh.output <- broadcast_into sh Output cur_sh Output;
    sh.batch <- broadcast_into sh Batch cur_sh Batch;

  | Transpose (Permute spec, sh) ->
    let ls_rhs, ls_lhs = match einsum_of_spec spec with
    | `Permute_unop (ls_rhs, ls_lhs) -> ls_rhs, ls_lhs
    | _ -> raise @@
      Shape_error ("Invalid permutation spec (expected one argument): "^spec, sh, cur_sh) in
    let sh_rhs = to_axis_map sh in
    let sh_lhs = to_axis_map cur_sh in
    let eqs_rhs = eqs_xhs spec sh ls_rhs sh_rhs in
    let eqs_lhs = eqs_xhs spec sh ls_lhs sh_lhs in
    let eqs = Map.merge eqs_rhs eqs_lhs ~f:(fun ~key:_label -> function
        | `Both (rhs, lhs) -> Some (rhs @ lhs)
        | `Left rhs -> Some rhs
        | `Right lhs -> Some lhs) in
    let label_dims = Map.mapi eqs ~f:(einsum_one_dim spec cur_sh sh) in
    let lhs_labels = axes_with_inf_labels ~all_labels:label_dims ls_lhs in
    let pseudo_to_labels_lhs = pseudo_to_labels_xhs lhs_labels cur_sh in
    let inferred_lhs = Map.map lhs_labels ~f:(Map.find_exn label_dims) in
    let b_lhs, i_lhs, o_lhs = axis_map_to_dims_bio inferred_lhs in
    (if is_inferred cur_sh.batch || is_unknown cur_sh.batch then cur_sh.batch <- to_inferred b_lhs);
    (if is_inferred cur_sh.input || is_unknown cur_sh.input then cur_sh.input <- to_inferred i_lhs);
    (if is_inferred cur_sh.output || is_unknown cur_sh.output then cur_sh.output <- to_inferred o_lhs);
    let rhs_labels = axes_with_inf_labels ~all_labels:label_dims ls_rhs in
    let pseudo_to_labels_rhs = pseudo_to_labels_xhs rhs_labels sh in
    let inferred_rhs = Map.map rhs_labels ~f:(Map.find_exn label_dims) in
    let b_rhs, i_rhs, o_rhs = axis_map_to_dims_bio inferred_rhs in
    (if is_inferred sh.batch || is_unknown sh.batch then sh.batch <- to_inferred b_rhs);
    (if is_inferred sh.input || is_unknown sh.input then sh.input <- to_inferred i_rhs);
    (if is_inferred sh.output || is_unknown sh.output then sh.output <- to_inferred o_rhs);
    let all_axis_labels = all_axis_labels cur_sh sh spec pseudo_to_labels_lhs pseudo_to_labels_rhs in
    let lhs_axis_labels = Map.map lhs_labels ~f:(Map.find_exn all_axis_labels) in
    cur_sh.axis_labels <- lhs_axis_labels;
    let rhs_axis_labels = Map.map rhs_labels ~f:(Map.find_exn all_axis_labels) in
    sh.axis_labels <- rhs_axis_labels
    (* FIXME(87): handle givenness / fixedness propagation *)

  | Broadcast (Pointwise_bin, sh1, sh2) ->
    let up_labels = pointwise_labels sh1 sh2 sh1.axis_labels sh2.axis_labels in
    cur_sh.axis_labels <- up_labels;
    (* Note: will not work as expected (propagate givenness/fixedness) if the shape is pre-filled
       as [Inferred] instead of [Unknown]. *)
    (if is_unknown cur_sh.input then
      cur_sh.input <- broadcast_dims sh1 sh2 AxisKey.Input up_labels sh1.input sh2.input
    else (
      cur_sh.input <- broadcast_into cur_sh Input sh1 Input;
      cur_sh.input <- broadcast_into cur_sh Input sh2 Input;
    ));
    (if is_unknown cur_sh.output then
      cur_sh.output <- broadcast_dims sh1 sh2 AxisKey.Output up_labels sh1.output sh2.output
    else (
      cur_sh.output <- broadcast_into cur_sh Output sh1 Output;
      cur_sh.output <- broadcast_into cur_sh Output sh2 Output;
    ));
    (if is_unknown cur_sh.batch then
      cur_sh.batch <- broadcast_dims sh1 sh2 AxisKey.Batch up_labels sh1.batch sh2.batch
    else (
      cur_sh.batch <- broadcast_into cur_sh Batch sh1 Batch;
      cur_sh.batch <- broadcast_into cur_sh Batch sh2 Batch;
    ));
    
    sh1.input <- broadcast_into sh1 Input cur_sh Input;
    sh1.output <- broadcast_into sh1 Output cur_sh Output;
    sh1.batch <- broadcast_into sh1 Batch cur_sh Batch;
    sh2.input <- broadcast_into sh2 Input cur_sh Input;
    sh2.output <- broadcast_into sh2 Output cur_sh Output;
    sh2.batch <- broadcast_into sh2 Batch cur_sh Batch;

  | Broadcast (Compose, sh1, sh2) ->
    (** [sh2] is the value or the function that gets applied first: [cur_sh(x) = sh1(sh2(x))].
       I.e. [cur.I = sh2.I, cur.O = sh1.O, sh2.O = sh1.I]. *)
    cur_sh.input <- broadcast_into ~det:true cur_sh AxisKey.Input sh2 AxisKey.Input;
    cur_sh.output <- broadcast_into ~det:true cur_sh AxisKey.Output sh1 AxisKey.Output;
    (if is_unknown cur_sh.batch then
      let up_labels = update_labels cur_sh Batch sh1 Batch in
      cur_sh.axis_labels <- up_labels;
      let up_labels = update_labels cur_sh Batch sh2 Batch in
      cur_sh.axis_labels <- up_labels;
      cur_sh.batch <- broadcast_dims sh1 sh2 AxisKey.Batch up_labels sh1.batch sh2.batch
     else (
       cur_sh.batch <- broadcast_into cur_sh Batch sh1 Batch;
       cur_sh.batch <- broadcast_into cur_sh Batch sh2 Batch;
     ));
    
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
    let ls_rhs1, ls_rhs2, ls_lhs = match einsum_of_spec spec with
    | `Permute_binop (ls_rhs1, ls_rhs2, ls_lhs) -> ls_rhs1, ls_rhs2, ls_lhs
    | _ -> raise @@
      Shape_error ("Invalid einsum spec (expected two arguments): "^spec, sh1, sh2) in
    let sh_rhs1 = to_axis_map sh1 in
    let sh_rhs2 = to_axis_map sh2 in
    let sh_lhs = to_axis_map cur_sh in
    let eqs_rhs1 = eqs_xhs spec sh1 ls_rhs1 sh_rhs1 in
    let eqs_rhs2 = eqs_xhs spec sh2 ls_rhs2 sh_rhs2 in
    let eqs_lhs = eqs_xhs spec sh1 ls_lhs sh_lhs in
    let side_eq side (axis, dims) = (side, axis), dims in
    let eqs = Map.merge eqs_rhs1 eqs_lhs ~f:(fun ~key:_label -> function
        | `Both (rhs, lhs) -> Some (List.rev_map_append rhs ~f:(side_eq `Rhs1) @@ List.map lhs ~f:(side_eq `Lhs))
        | `Left rhs -> Some (List.map rhs ~f:(side_eq `Rhs1))
        | `Right lhs -> Some (List.map lhs ~f:(side_eq `Lhs))) in
    let eqs = Map.merge eqs_rhs2 eqs ~f:(fun ~key:_label -> function
        | `Both (rhs, more) -> Some (List.rev_map_append rhs ~f:(side_eq `Rhs2) more)
        | `Left rhs -> Some (List.map rhs ~f:(side_eq `Rhs2))
        | `Right more -> Some more) in
    let label_dims = Map.mapi eqs ~f:(einsum_one_dim spec sh1 sh2) in
    let lhs_labels = axes_with_inf_labels ~all_labels:label_dims ls_lhs in
    let pseudo_to_labels_lhs = pseudo_to_labels_xhs lhs_labels cur_sh in
    let inferred_lhs = Map.map lhs_labels ~f:(Map.find_exn label_dims) in
    let b_lhs, i_lhs, o_lhs = axis_map_to_dims_bio inferred_lhs in
    (if is_inferred cur_sh.batch || is_unknown cur_sh.batch then cur_sh.batch <- to_inferred b_lhs);
    (if is_inferred cur_sh.input || is_unknown cur_sh.input then cur_sh.input <- to_inferred i_lhs);
    (if is_inferred cur_sh.output || is_unknown cur_sh.output then cur_sh.output <- to_inferred o_lhs);
    let rhs1_labels = axes_with_inf_labels ~all_labels:label_dims ls_rhs1 in
    let pseudo_to_labels_rhs1 = pseudo_to_labels_xhs rhs1_labels sh1 in
    let inferred_rhs1 = Map.map rhs1_labels ~f:(Map.find_exn label_dims) in
    let b_rhs1, i_rhs1, o_rhs1 = axis_map_to_dims_bio inferred_rhs1 in
    (if is_inferred sh1.batch || is_unknown sh1.batch then sh1.batch <- to_inferred b_rhs1);
    (if is_inferred sh1.input || is_unknown sh1.input then sh1.input <- to_inferred i_rhs1);
    (if is_inferred sh1.output || is_unknown sh1.output then sh1.output <- to_inferred o_rhs1);
    let rhs2_labels = axes_with_inf_labels ~all_labels:label_dims ls_rhs2 in
    let pseudo_to_labels_rhs2 = pseudo_to_labels_xhs rhs2_labels sh2 in
    let inferred_rhs2 = Map.map rhs2_labels ~f:(Map.find_exn label_dims) in
    let b_rhs2, i_rhs2, o_rhs2 = axis_map_to_dims_bio inferred_rhs2 in
    (if is_inferred sh2.batch || is_unknown sh2.batch then sh2.batch <- to_inferred b_rhs2);
    (if is_inferred sh2.input || is_unknown sh2.input then sh2.input <- to_inferred i_rhs2);
    (if is_inferred sh2.output || is_unknown sh2.output then sh2.output <- to_inferred o_rhs2);
    let all_axis_labels1 = all_axis_labels cur_sh sh1 spec pseudo_to_labels_lhs pseudo_to_labels_rhs1 in
    let all_axis_labels = all_axis_labels cur_sh sh2 spec all_axis_labels1 pseudo_to_labels_rhs2 in
    let lhs_axis_labels = Map.map lhs_labels ~f:(Map.find_exn all_axis_labels) in
    cur_sh.axis_labels <- lhs_axis_labels;
    let rhs1_axis_labels = Map.map rhs1_labels ~f:(Map.find_exn all_axis_labels) in
    sh1.axis_labels <- rhs1_axis_labels;
    let rhs2_axis_labels = Map.map rhs2_labels ~f:(Map.find_exn all_axis_labels) in
    sh2.axis_labels <- rhs2_axis_labels
    (* FIXME(87): handle givenness / fixedness propagation *)


(** Uses the matrix convention of putting the input axes last. *)
let to_dims (sh: t): int array =
  let b_dims = match sh.batch with
    | Unknown -> raise @@ Shape_error ("Batch dimensions still unknown", sh, sh)
    | Inferred dims | Given dims | Fixed dims -> Array.of_list dims in
  let i_dims = match sh.input with
    | Unknown -> raise @@ Shape_error ("Input dimensions still unknown", sh, sh)
    | Inferred dims | Given dims | Fixed dims -> Array.of_list dims in
  let o_dims = match sh.output with
    | Unknown -> raise @@ Shape_error ("Output dimensions still unknown", sh, sh)
    | Inferred dims | Given dims | Fixed dims -> Array.of_list dims in
  Array.concat [b_dims; o_dims; i_dims]

type symbol = Symbol of int [@@deriving compare, sexp, variants]
let unique_id = ref 0
let get_symbol() = Int.incr unique_id; Symbol !unique_id

(** An index into a single axis for doing computations over multiple [Shape]-derived [Code]s. *)
type 'a axis_index =
| Fixed_idx of int
(** The specific position along an axis. *)
| Iterator of 'a
(** The given member of the [product_space] corresponding to some [product_iterators]. *)
[@@deriving compare, sexp, variants]

type symbolic_axis = symbol axis_index [@@deriving compare, sexp]

(** All the information relevant for [Code] code generation contained in a completed [update_step]. *)
type projections = {
  product_space: int array;
  (** The product space dimensions (concatentation of the relevant batch, output, input axes) with
      the same semantics as [to_dims], that an operation should parallelize (map-reduce) over. *)
  product_iterators: symbol array;
  (** The product space iterators (concatentation of the relevant batch, output, input axes)
      for iterating over the [product_space] axes, where same axes are at same array indices. *)
  project_lhs: symbolic_axis array;
  (** A projection that takes an [product_space]-bound index and produces an index into the result of
      an operation. *)
  project_rhs1: symbolic_axis array;
  (** A projection that takes an [product_space]-bound index and produces an index into the (first)
      argument of an operation. *)
  project_rhs2: symbolic_axis array option;
  (** A projection that takes an [product_space]-bound index and produces an index into the second
      argument of a binary operation. *)
} [@@deriving sexp]

(** Projections for iterating over a terminal in [Code], or for a pointwise unary operator. *)
let identity_projections product_space =
  let product_iterators = Array.map product_space ~f:(fun _ -> get_symbol()) in
  let project_lhs = Array.map product_iterators ~f:iterator in
  { product_space; product_iterators; project_lhs; project_rhs1=project_lhs; project_rhs2=None; }

(** Computes the indexing into subformulas given the shape information of a formula. The processing
    mirrors [propagate_shapes], but [derive_projections] should only be invoked when the shapes
    are inferred already. *)
let derive_projections (shapes: update_step) : projections =
  (* Broadcasts symmetrically to iterate all axes. *)
  let broadcast_dims sh1_dims sh2_dims =
    let rec broad_back_dims accu = function
    | [], [] -> accu
    | dims, [] | [], dims -> List.rev_append dims accu
    | d1::dims1, d2::dims2 -> broad_back_dims (max d1 d2::accu) (dims1, dims2) in
    match sh1_dims, sh2_dims with
      | Unknown, Unknown -> []
      | (Inferred dims | Given dims| Fixed dims), Unknown
      | Unknown, (Inferred dims | Given dims | Fixed dims) -> dims
      | (Inferred dims1 | Given dims1 | Fixed dims1), (Inferred dims2 | Given dims2 | Fixed dims2) ->
        broad_back_dims [] (List.rev dims1, List.rev dims2) in
  let broadcast_sh sh1 kind1 sh2 kind2 =
    broadcast_dims (dims_of_kind kind1 sh1) (dims_of_kind kind2 sh2) in
  (* The first arg is "into" we build the projection for, the second arg is the context. *)
  let broadcast_into_dims product_idcs sh1_dims sh2_dims =
    (* TODO: audit the use of [Iterator idx] / [Fixed_idx 0] wrt. the [project_lhs_verify] assert. *)
    let broad_dim idx = function
    | 1, 1 -> Iterator idx
    | 1, _d -> Fixed_idx 0
    | _ -> Iterator idx in
    let rec broad_back_dims accu_idcs = function
    | [], [], [] -> accu_idcs
    | _idcs, [], _dims -> accu_idcs
    | idcs, _dims, [] -> List.rev_map_append idcs accu_idcs ~f:iterator
    | idx::idcs, d1::dims1, d2::dims2 ->
      broad_back_dims (broad_dim idx (d1, d2)::accu_idcs) (idcs, dims1, dims2)
    | _ -> assert false in
    match sh1_dims, sh2_dims with
      | Unknown, Unknown ->
        assert (0 = List.length product_idcs); 
        []
      | (Inferred dims | Given dims| Fixed dims), Unknown
      | Unknown, (Inferred dims | Given dims | Fixed dims) ->
        assert (List.length dims = List.length product_idcs);
        List.map product_idcs ~f:iterator
      | (Inferred dims1 | Given dims1 | Fixed dims1), (Inferred dims2 | Given dims2 | Fixed dims2) ->
        broad_back_dims [] (List.rev product_idcs, List.rev dims1, List.rev dims2) in
  let broadcast_into product_idcs sh1 kind1 sh2 kind2 =
    broadcast_into_dims product_idcs (dims_of_kind kind1 sh1) (dims_of_kind kind2 sh2) in
  let cur_sh = shapes.shape in
  let einsum_one_dim terms =
    List.fold terms ~init:1 ~f:(fun dim ((_side, _axis), dims) ->
      match dims with
        | Unknown -> dim
        | Inferred [dim2] | Given [dim2] | Fixed [dim2] when dim = dim2 -> dim
        | Inferred [dim2] | Given [dim2] | Fixed [dim2] when dim = 1 -> dim2
        | Inferred [] | Given [] | Fixed [] -> dim
        | _ -> assert false
      ) in
  let map_with_dims dims idcs ~f =
    let rdims = List.rev @@ list_of_dims dims in
    let ridcs = List.take (Array.to_list @@ Array.rev idcs) @@ List.length rdims in
    List.rev @@ List.map2_exn rdims ridcs ~f in
  let eqs_xhs ls_xhs sh_xhs =
    let eqs = Map.merge ls_xhs.labels sh_xhs ~f:(fun ~key:axis -> function
        | `Both (label, dim) -> Some (label, (axis, dim))
        | `Left label -> Some (label, (axis, Inferred []))
        | `Right (Given [] | Fixed [] | Inferred []) -> None
        | `Right _dim when not (bcast_of_kind axis.in_axes ls_xhs) -> assert false
        | `Right dim -> Some (gen_label_of_axis ls_xhs axis, (axis, dim))) in
    Map.of_alist_multi (module String) @@ Map.data eqs in

  (* For binary cases, we cannot rely on [cur_sh] containing all axes, since in principle it could
     have been restricted by an initial [Given] setting to efficiently implement map-reduce. *)
  match shapes.logic with
  | Terminal -> identity_projections @@ to_dims cur_sh
  | Transpose (Transpose, sh) ->
    let product_inp = broadcast_sh cur_sh Input sh Output in
    let iters_inp = List.map product_inp ~f:(fun _ -> get_symbol()) in
    let lhs_input = broadcast_into iters_inp cur_sh Input sh Output in
    let product_out = broadcast_sh cur_sh Output sh Input in
    let iters_out = List.map product_out ~f:(fun _ -> get_symbol()) in
    let lhs_output = broadcast_into iters_out cur_sh Output sh Input in
    let product_bch = broadcast_sh cur_sh Batch sh Batch in
    let iters_bch = List.map product_bch ~f:(fun _ -> get_symbol()) in
    let lhs_batch = broadcast_into iters_bch cur_sh Batch sh Batch in
    let rhs_input = broadcast_into iters_out sh Input cur_sh Output in
    let rhs_output = broadcast_into iters_inp sh Output cur_sh Input in
    let rhs_batch = broadcast_into iters_bch sh Batch cur_sh Batch in
    let product_space =
      Array.of_list @@ List.concat [product_bch; product_out; product_inp] in
    let product_iterators =
      Array.of_list @@ List.concat [iters_bch; iters_out; iters_inp] in
    let project_lhs =
      Array.of_list @@ List.concat [lhs_batch; lhs_output; lhs_input] in
    let project_rhs1 =
      Array.of_list @@ List.concat [rhs_batch; rhs_output; rhs_input] in    
    { product_space; product_iterators; project_lhs; project_rhs1; project_rhs2 = None }

  | Transpose (Pointwise_un, sh) ->
    let product_inp = broadcast_sh cur_sh Input sh Input in
    let iters_inp = List.map product_inp ~f:(fun _ -> get_symbol()) in
    let lhs_input = broadcast_into iters_inp cur_sh Input sh Input in
    let product_out = broadcast_sh cur_sh Output sh Output in
    let iters_out = List.map product_out ~f:(fun _ -> get_symbol()) in
    let lhs_output = broadcast_into iters_out cur_sh Output sh Output in
    let product_bch = broadcast_sh cur_sh Batch sh Batch in
    let iters_bch = List.map product_bch ~f:(fun _ -> get_symbol()) in
    let lhs_batch = broadcast_into iters_bch cur_sh Batch sh Batch in
    let rhs_input = broadcast_into iters_inp sh Input cur_sh Input in
    let rhs_output = broadcast_into iters_out sh Output cur_sh Output in
    let rhs_batch = broadcast_into iters_bch sh Batch cur_sh Batch in
    let product_space =
      Array.of_list @@ List.concat [product_bch; product_out; product_inp] in
    let product_iterators =
      Array.of_list @@ List.concat [iters_bch; iters_out; iters_inp] in
    let project_lhs =
      Array.of_list @@ List.concat [lhs_batch; lhs_output; lhs_input] in
    let project_rhs1 =
      Array.of_list @@ List.concat [rhs_batch; rhs_output; rhs_input] in    
    { product_space; product_iterators; project_lhs; project_rhs1; project_rhs2 = None }

  | Transpose (Permute spec, sh) ->
    let ls_rhs, ls_lhs = match einsum_of_spec spec with
    | `Permute_unop (ls_rhs, ls_lhs) -> ls_rhs, ls_lhs
    | _ -> raise @@
      Shape_error ("Invalid permutation (single-argument einsum) spec: "^spec, sh, cur_sh) in
    (* For einsum the product_space is precisely one-axis-per-label. *)
    let sh_rhs = to_axis_map sh in
    let sh_lhs = to_axis_map cur_sh in
    let eqs_rhs = eqs_xhs ls_rhs sh_rhs in
    let eqs_lhs = eqs_xhs ls_lhs sh_lhs in
    let side_eq side (axis, dims) = (side, axis), dims in
    let eqs = Map.merge eqs_rhs eqs_lhs ~f:(fun ~key:_label -> function
        | `Both (rhs, lhs) -> Some (List.rev_map_append rhs ~f:(side_eq `Rhs1) @@ List.map lhs ~f:(side_eq `Lhs))
        | `Left rhs -> Some (List.map rhs ~f:(side_eq `Rhs1))
        | `Right lhs -> Some (List.map lhs ~f:(side_eq `Lhs))) in
    let label_dims = Map.map eqs ~f:einsum_one_dim in
    let label_iterators = Map.map eqs ~f:(fun _ -> get_symbol()) in
    let product_space = Array.of_list @@ Map.data label_dims in
    let product_iterators = Array.of_list @@ Map.data label_iterators in
    (* Inferred dims are not broadcasted-from-1, i.e. do not need Fixed_idx. But it doesn't hurt
       to treat them uniformly. *)
    let lhs_labels = axes_with_inf_labels ~all_labels:label_dims ls_lhs in
    let inferred_lhs = Map.map lhs_labels ~f:(Map.find_exn label_iterators) in
    let b_lhs, i_lhs, o_lhs = axis_map_to_dims_bio inferred_lhs in
    let lhs_batch = map_with_dims cur_sh.batch b_lhs ~f:(fun d s ->
      if d = 1 then Fixed_idx 0 else Iterator s) in
    let lhs_input = map_with_dims cur_sh.input i_lhs ~f:(fun d s ->
      if d = 1 then Fixed_idx 0 else Iterator s) in
    let lhs_output = map_with_dims cur_sh.output o_lhs ~f:(fun d s ->
      if d = 1 then Fixed_idx 0 else Iterator s) in
    let rhs_labels = axes_with_inf_labels ~all_labels:label_dims ls_rhs in
    let inferred_rhs = Map.map rhs_labels ~f:(Map.find_exn label_iterators) in
    let b_rhs, i_rhs, o_rhs = axis_map_to_dims_bio inferred_rhs in
    let rhs_batch = map_with_dims cur_sh.batch b_rhs ~f:(fun d s ->
        if d = 1 then Fixed_idx 0 else Iterator s) in
    let rhs_input = map_with_dims cur_sh.input i_rhs ~f:(fun d s ->
        if d = 1 then Fixed_idx 0 else Iterator s) in
    let rhs_output = map_with_dims cur_sh.output o_rhs ~f:(fun d s ->
        if d = 1 then Fixed_idx 0 else Iterator s) in
    let project_lhs =
      Array.of_list @@ List.concat [lhs_batch; lhs_output; lhs_input] in
    let project_rhs1 =
      Array.of_list @@ List.concat [rhs_batch; rhs_output; rhs_input] in    
    { product_space; product_iterators; project_lhs; project_rhs1; project_rhs2 = None }

  | Broadcast (Pointwise_bin, sh1, sh2) ->
    let product_inp =
      match cur_sh.input with
      | Given _ | Unknown -> broadcast_sh sh1 Input sh2 Input
      | Fixed dims | Inferred dims -> dims in
    let iters_inp = List.map product_inp ~f:(fun _ -> get_symbol()) in
    let lhs1_input = broadcast_into iters_inp cur_sh Input sh1 Input in
    let product_out =
      match cur_sh.output with
      | Given _ | Unknown -> broadcast_sh sh1 Output sh2 Output
      | Fixed dims | Inferred dims -> dims in
      let iters_out = List.map product_out ~f:(fun _ -> get_symbol()) in
    let lhs1_output = broadcast_into iters_out cur_sh Output sh1 Output in
    let product_bch =
      match cur_sh.batch with
      | Given _ | Unknown -> broadcast_sh sh1 Batch sh2 Batch
      | Fixed dims | Inferred dims -> dims in
    let iters_bch = List.map product_bch ~f:(fun _ -> get_symbol()) in
    let lhs1_batch = broadcast_into iters_bch cur_sh Batch sh1 Batch in
    let rhs1_input = broadcast_into iters_inp sh1 Input cur_sh Input in
    let rhs1_output = broadcast_into iters_out sh1 Output cur_sh Output in
    let rhs1_batch = broadcast_into iters_bch sh1 Batch sh2 Batch in
    let lhs2_input = broadcast_into iters_inp cur_sh Input sh2 Input in
    let lhs2_output = broadcast_into iters_out cur_sh Output sh2 Output in
    let lhs2_batch = broadcast_into iters_bch cur_sh Batch sh2 Batch in
    let rhs2_input = broadcast_into iters_inp sh2 Input cur_sh Input in
    let rhs2_output = broadcast_into iters_out sh2 Output cur_sh Output in
    let rhs2_batch = broadcast_into iters_bch sh2 Batch sh1 Batch in
    let product_space =
      Array.of_list @@ List.concat [product_bch; product_out; product_inp] in
    let product_iterators =
      Array.of_list @@ List.concat [iters_bch; iters_out; iters_inp] in
    let project_lhs =
      Array.of_list @@ List.concat [lhs1_batch; lhs1_output; lhs1_input] in
    let project_lhs_verify =
      Array.of_list @@ List.concat [lhs2_batch; lhs2_output; lhs2_input] in
    assert (Array.equal (fun a b -> compare_symbolic_axis a b = 0) project_lhs project_lhs_verify);
    let project_rhs1 =
      Array.of_list @@ List.concat [rhs1_batch; rhs1_output; rhs1_input] in    
    let project_rhs2 =
      Some (Array.of_list @@ List.concat [rhs2_batch; rhs2_output; rhs2_input]) in    
    { product_space; product_iterators; project_lhs; project_rhs1; project_rhs2 }

  | Broadcast (Compose, sh1, sh2) ->
    (** [sh2] is the value or the function that gets applied first: [cur_sh(x) = sh1(sh2(x))].
       I.e. [cur.I = sh2.I, cur.O = sh1.O, sh2.O = sh1.I]. *)
    let product_inp = broadcast_sh cur_sh Input sh2 Input in
    let iters_inp = List.map product_inp ~f:(fun _ -> get_symbol()) in
    let lhs_input = broadcast_into iters_inp cur_sh Input sh2 Input in
    let product_out = broadcast_sh cur_sh Output sh1 Output in
    let iters_out = List.map product_out ~f:(fun _ -> get_symbol()) in
    let lhs_output = broadcast_into iters_out cur_sh Output sh1 Output in
    let product_bch =
      match cur_sh.batch with
      | Given _ | Unknown -> broadcast_sh sh1 Batch sh2 Batch
      | Fixed dims | Inferred dims -> dims in
    let iters_bch = List.map product_bch ~f:(fun _ -> get_symbol()) in
    let lhs1_batch = broadcast_into iters_bch cur_sh Batch sh1 Batch in
    let lhs2_batch = broadcast_into iters_bch cur_sh Batch sh2 Batch in
    assert (List.equal (fun a b -> compare_symbolic_axis a b = 0) lhs1_batch lhs2_batch);

    let product_hid = broadcast_sh sh1 Input sh2 Output in
    let iters_hid = List.map product_hid ~f:(fun _ -> get_symbol()) in
    let rhs1_input = broadcast_into iters_hid sh1 Input sh2 Output in
    let rhs1_output = broadcast_into iters_out sh1 Output cur_sh Output in
    let rhs1_batch = broadcast_into iters_bch sh1 Batch sh2 Batch in
    let rhs2_input = broadcast_into iters_inp sh2 Input cur_sh Input in
    let rhs2_output = broadcast_into iters_hid sh2 Output sh1 Input in
    let rhs2_batch = broadcast_into iters_bch sh2 Batch sh1 Batch in
    let product_space =
      Array.of_list @@ List.concat [product_bch; product_out; product_hid; product_inp] in
    let product_iterators =
      Array.of_list @@ List.concat [iters_bch; iters_out; iters_hid; iters_inp] in
    let project_lhs =
      Array.of_list @@ List.concat [lhs1_batch; lhs_output; lhs_input] in
    let project_rhs1 =
      Array.of_list @@ List.concat [rhs1_batch; rhs1_output; rhs1_input] in    
    let project_rhs2 =
      Some (Array.of_list @@ List.concat [rhs2_batch; rhs2_output; rhs2_input]) in    
    { product_space; product_iterators; project_lhs; project_rhs1; project_rhs2 }

  | Broadcast (Einsum spec, sh1, sh2) ->
    let ls_rhs1, ls_rhs2, ls_lhs = match einsum_of_spec spec with
    | `Permute_binop (ls_rhs1, ls_rhs2, ls_lhs) -> ls_rhs1, ls_rhs2, ls_lhs
    | _ -> raise @@
      Shape_error ("Invalid (two-argument) einsum spec: "^spec, sh1, sh2) in
    (* For einsum the product_space is precisely one-axis-per-label. *)
    let sh_rhs1 = to_axis_map sh1 in
    let sh_rhs2 = to_axis_map sh2 in
    let sh_lhs = to_axis_map cur_sh in
    let eqs_rhs1 = eqs_xhs ls_rhs1 sh_rhs1 in
    let eqs_rhs2 = eqs_xhs ls_rhs2 sh_rhs2 in
    let eqs_lhs = eqs_xhs ls_lhs sh_lhs in
    let side_eq side (axis, dims) = (side, axis), dims in
    let eqs = Map.merge eqs_rhs1 eqs_lhs ~f:(fun ~key:_label -> function
        | `Both (rhs, lhs) ->
           Some (List.rev_map_append rhs ~f:(side_eq `Rhs1) @@ List.map lhs ~f:(side_eq `Lhs))
        | `Left rhs -> Some (List.map rhs ~f:(side_eq `Rhs1))
        | `Right lhs -> Some (List.map lhs ~f:(side_eq `Lhs))) in
    let eqs = Map.merge eqs_rhs2 eqs ~f:(fun ~key:_label -> function
        | `Both (rhs, more) -> Some (List.rev_map_append rhs ~f:(side_eq `Rhs2) more)
        | `Left rhs -> Some (List.map rhs ~f:(side_eq `Rhs2))
        | `Right more -> Some more) in
    let label_dims = Map.map eqs ~f:einsum_one_dim in
    let label_iterators = Map.map eqs ~f:(fun _ -> get_symbol()) in
    let product_space = Array.of_list @@ Map.data label_dims in
    let product_iterators = Array.of_list @@ Map.data label_iterators in
    (* Inferred dims are not broadcasted-from-1, i.e. do not need Fixed_idx. But it doesn't hurt
       to treat them uniformly. *)
    let lhs_labels = axes_with_inf_labels ~all_labels:label_dims ls_lhs in
    let inferred_lhs = Map.map lhs_labels ~f:(Map.find_exn label_iterators) in
    let b_lhs, i_lhs, o_lhs = axis_map_to_dims_bio inferred_lhs in
    let lhs_batch = map_with_dims cur_sh.batch b_lhs ~f:(fun d s ->
      if d = 1 then Fixed_idx 0 else Iterator s) in
    let lhs_input = map_with_dims cur_sh.input i_lhs ~f:(fun d s ->
      if d = 1 then Fixed_idx 0 else Iterator s) in
    let lhs_output = map_with_dims cur_sh.output o_lhs ~f:(fun d s ->
      if d = 1 then Fixed_idx 0 else Iterator s) in
    let rhs1_labels = axes_with_inf_labels ~all_labels:label_dims ls_rhs1 in
    let inferred_rhs1 = Map.map rhs1_labels ~f:(Map.find_exn label_iterators) in
    let b_rhs1, i_rhs1, o_rhs1 = axis_map_to_dims_bio inferred_rhs1 in
    let rhs1_batch = map_with_dims cur_sh.batch b_rhs1 ~f:(fun d s ->
      if d = 1 then Fixed_idx 0 else Iterator s) in
    let rhs1_input = map_with_dims cur_sh.input i_rhs1 ~f:(fun d s ->
      if d = 1 then Fixed_idx 0 else Iterator s) in
    let rhs1_output = map_with_dims cur_sh.output o_rhs1 ~f:(fun d s ->
      if d = 1 then Fixed_idx 0 else Iterator s) in
    let rhs2_labels = axes_with_inf_labels ~all_labels:label_dims ls_rhs2 in
    let inferred_rhs2 = Map.map rhs2_labels ~f:(Map.find_exn label_iterators) in
    let b_rhs2, i_rhs2, o_rhs2 = axis_map_to_dims_bio inferred_rhs2 in
    let rhs2_batch = map_with_dims cur_sh.batch b_rhs2 ~f:(fun d s ->
      if d = 1 then Fixed_idx 0 else Iterator s) in
    let rhs2_input = map_with_dims cur_sh.input i_rhs2 ~f:(fun d s ->
      if d = 1 then Fixed_idx 0 else Iterator s) in
    let rhs2_output = map_with_dims cur_sh.output o_rhs2 ~f:(fun d s ->
      if d = 1 then Fixed_idx 0 else Iterator s) in
    let project_lhs =
      Array.of_list @@ List.concat [lhs_batch; lhs_output; lhs_input] in
    let project_rhs1 =
      Array.of_list @@ List.concat [rhs1_batch; rhs1_output; rhs1_input] in    
    let project_rhs2 =
      Some (Array.of_list @@ List.concat [rhs2_batch; rhs2_output; rhs2_input]) in    
    { product_space; product_iterators; project_lhs; project_rhs1; project_rhs2 }

let backprop1 projections = {
  projections with project_lhs = projections.project_rhs1; project_rhs1 = projections.project_lhs;
}

let backprop2 projections =
  match projections.project_rhs2 with
  | None -> invalid_arg "Shape.backprop2: unary shapes (project_rhs2 is None)"
  | Some project_rhs2 -> 
    { projections with project_lhs = project_rhs2; project_rhs2 = Some projections.project_lhs }

let backprop_unary projections = {
  projections with project_lhs = projections.project_rhs1; project_rhs1 = projections.project_lhs;
                   project_rhs2 = Some projections.project_lhs;
}

let derive_index iterator_symbols (projection: symbolic_axis array) (type iterator):
  iterator array -> iterator axis_index array =
  let sym_to_i =
    Array.mapi iterator_symbols ~f:(fun i (Symbol s) -> s, i) |>
    Array.to_list |> Map.of_alist_exn (module Int) in
  let positions: int axis_index array = Array.map projection ~f:(
    function
    | Fixed_idx i -> Fixed_idx i
    | Iterator (Symbol s) -> Iterator (Map.find_exn sym_to_i s)
  ) in
  fun product -> Array.map positions ~f:(
      function Fixed_idx i -> Fixed_idx i | Iterator p -> Iterator product.(p))

let make ?batch_dims ?input_dims ?output_dims ?axis_labels ?deduced ~id () =
  let input = match input_dims with
    | None -> Unknown
    | Some dims -> Given dims in
  let output = match output_dims with
    | None -> Unknown
    | Some dims -> Given dims in
  let batch = match batch_dims with
    | None -> Unknown
    | Some dims -> Given dims in
  let deduce_within_shape_constraints = Option.value deduced ~default:Not_constrained in
  let axis_labels = match axis_labels with 
    | None -> Map.empty (module AxisKey)
    | Some spec -> (axis_labels_of_spec spec).labels in
  {input; output; batch; deduce_within_shape_constraints; axis_labels; id}

let to_string_hum ?(style=`Axis_size) sh =
  let n_outputs = List.length @@ list_of_dims @@ dims_of_kind Output sh in
  let n_batch = List.length @@ list_of_dims @@ dims_of_kind Batch sh in
  let dims_to_string kind =
    let dims = list_of_dims @@ dims_of_kind kind sh in
    let n_dims = List.length dims in
    String.concat ~sep:"," @@ List.mapi dims ~f:(fun i d ->
        let key = AxisKey.{in_axes=kind; from_end=n_dims - i} in
        let num = match kind with
        | Input -> n_batch + n_outputs + i
        | Output -> n_batch + i
        | Batch -> i in
        match style, Map.find sh.axis_labels key with
        | `Only_labels, None -> "_" 
        | `Axis_size, None -> Int.to_string d
        | `Axis_number_and_size, None -> Int.to_string num^":"^Int.to_string d
        | `Only_labels, Some l -> l
        | `Axis_size, Some l -> l ^":"^ Int.to_string d
        | `Axis_number_and_size, Some l -> l^"="^Int.to_string num^":"^Int.to_string d) in
  let batch_dims = dims_to_string Batch in
  let input_dims = dims_to_string Input in
  let output_dims = dims_to_string Output in
  if String.is_empty batch_dims && String.is_empty input_dims then output_dims
  else if String.is_empty batch_dims then input_dims^"->"^output_dims
  else if String.is_empty input_dims then batch_dims^"|"^output_dims
  else batch_dims^"|"^input_dims^"->"^output_dims

module CompareSymbol = struct
  type t = symbol
  let compare = compare_symbol
  let sexp_of_t = sexp_of_symbol
end
module Symbol = struct
  include CompareSymbol
  include Comparator.Make(CompareSymbol)
end

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
    type kind = Batch | Input | Output [@@deriving equal, compare, sexp, hash, variants]

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

module Dim_var = struct
  type t = { id : int; mutable label : string option [@compare.ignore] [@equal.ignore] [@hash.ignore] }
  [@@deriving equal, hash, compare, sexp]

  include Comparator.Make (struct
    type nonrec t = t

    let compare = compare
    let sexp_of_t = sexp_of_t
  end)
end

type dim_var = Dim_var.t [@@deriving equal, hash, compare, sexp]

(** A single axis in a shape. *)
type dim = Var of dim_var | Dim of { d : int; label : string option; proj_id : int }
[@@deriving equal, hash, compare, sexp, variants]

let uid = ref 0

let get_var ?label () : dim_var =
  Int.incr uid;
  { id = !uid; label }

let get_dim ~d ?label () =
  Int.incr uid;
  Dim { d; proj_id = !uid; label }

(** A row specifies how axes of a single kind in a shape (the shape-kind) can adapt to other shapes. *)
type row =
  | Row_var of int  (** The shape-kind can be inferred to have more axes. *)
  | Broadcastable  (** The shape does not have more axes of this kind, but is "polymorphic". *)
[@@deriving equal, hash, compare, sexp, variants]

type dims_constraint =
  | Unconstrained
  | Total_elems of int  (** The shape-kind, inclusive of the further row spec, has this many elements. *)
[@@deriving equal, hash, compare, sexp, variants]

let get_row_var () =
  Int.incr uid;
  Row_var !uid

module Row_id = struct
  type t = { sh_id : int; kind : AxisKey.kind } [@@deriving sexp, compare, equal, hash]

  include Comparator.Make (struct
    type nonrec t = t

    let compare = compare
    let sexp_of_t = sexp_of_t
  end)
end

type row_id = Row_id.t [@@deriving sexp, compare, equal, hash]

type dims = { dims : dim list; constr : dims_constraint; row : row; id : row_id }
[@@deriving equal, hash, compare, sexp]

type deduce_within_shape = Not_constrained | Input_equals_output [@@deriving compare, sexp, variants]

type t = {
  mutable batch : dims;
  mutable input : dims;
  mutable output : dims;
  id : int;  (** A node that has the same shape as this shape. *)
  debug_name : string;
}
[@@deriving equal, fields, sexp]
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
[@@deriving equal, sexp]

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
[@@deriving equal, sexp]

let logic_to_spec = function
  | Broadcast (Pointwise_bin, _, _) | Transpose (Pointwise_un, _) -> "."
  | Broadcast (Compose, _, _) -> "@"
  | Broadcast (Einsum spec, _, _) | Transpose (Permute spec, _) -> spec
  | Transpose (Transpose, _) -> "T"
  | Transpose (Batch_slice _, _) -> "@|"
  | Terminal _ -> "<terminal>"

type shape_error =
  | Shape_mismatch of t list
  | Row_mismatch of dims list
  | Dim_mismatch of dim list
  | Index_mismatch of Arrayjit.Indexing.axis_index list
[@@deriving sexp]

exception Shape_error of string * shape_error list [@@deriving sexp]

let dim_to_int_exn = function Dim { d; _ } -> d | Var _ -> invalid_arg "dim_to_int: dim still unknown"

let meet more_constr constr =
  match (more_constr, constr) with
  | Unconstrained, c -> c
  | c, Unconstrained -> c
  | (Total_elems n1 as c), Total_elems n2 when n1 = n2 -> c
  | Total_elems _, Total_elems _ -> raise @@ Shape_error ("Incompatible Total_elems constraints", [])

module Env : sig
  type 'a entry = { cur : 'a list; subr : 'a list; solved : 'a option } [@@deriving sexp]
  type dim_env = dim entry Map.M(Dim_var).t
  type row_env = dims entry Map.M(Int).t
  type proj_classes = int Map.M(Int).t [@@deriving sexp]

  type t = private {
    dim_env : dim_env;
    row_env : row_env;
    proj_classes : proj_classes;
    dim_rev_elim_order : dim_var list;
    row_rev_elim_order : int list;
  }
  [@@deriving sexp]

  val subst_dim : ?freshen_proj:bool -> t -> dim -> dim
  val occurs_dim : dim_var -> dim -> bool
  val subst_row : ?freshen_proj:bool -> t -> dims -> dims
  val occurs_row : int -> dims -> bool
  val update_dim : ?freshen_proj:bool -> is_complete:bool -> dim_var -> ?cur:dim -> ?subr:dim -> t -> t
  val update_row : ?freshen_proj:bool -> is_complete:bool -> int -> ?cur:dims -> ?subr:dims -> t -> t
  val apply_constraint : dims -> t -> t
  val update_proj_classes : int -> int -> t -> t
  val empty_env : t
  val with_proj_classes : proj_classes -> t -> t
  val merge_fresh_proj : is_complete:bool -> update:t -> state:t -> t
end = struct
  type 'a entry = { cur : 'a list; subr : 'a list; solved : 'a option } [@@deriving sexp]
  (** An entry implements inequalities [cur >= v >= subr] and/or an equality [v = solved]. *)

  type dim_env = dim entry Map.M(Dim_var).t [@@deriving sexp]
  type row_env = dims entry Map.M(Int).t [@@deriving sexp]
  type proj_classes = int Map.M(Int).t [@@deriving sexp]

  type t = {
    dim_env : dim_env;
    row_env : row_env;
    proj_classes : proj_classes;
    dim_rev_elim_order : dim_var list;
    row_rev_elim_order : int list;
  }
  [@@deriving sexp]
  (** Note that while we build up the partial sets of inequalities, the environment is not in solved form.
      It is only in resolved wrt. variables that are solved: [v -> e where Option.is_some e.solved]
      do not appear elsewhere in the environment. But once [finish_inference] is called, it becomes in
      solved form: variables later in the elimination order do not appear in entries for variables
      earlier in the elimination order. *)

  let s_dim_one v ~value ~in_ = match in_ with Var v2 when equal_dim_var v v2 -> value | _ -> in_

  let s_dim_one_in_entry v ~value { cur; subr; solved } =
    let upd m x = m x ~f:(fun in_ -> s_dim_one v ~value ~in_) in
    { cur = upd List.map cur; subr = upd List.map subr; solved = upd Option.map solved }

  let s_dim_one_in_row v ~value in_ =
    { in_ with dims = List.map in_.dims ~f:(fun in_ -> s_dim_one v ~value ~in_) }

  let s_dim_one_in_row_entry v ~value { cur; subr; solved } =
    let upd m x = m x ~f:(s_dim_one_in_row v ~value) in
    { cur = upd List.map cur; subr = upd List.map subr; solved = upd Option.map solved }

  let subst_dim ?(freshen_proj = false) env = function
    | Dim { d; label; proj_id = _ } when freshen_proj -> get_dim ~d ?label ()
    | Dim _ as d -> d
    | Var v as default ->
        Option.value ~default @@ Option.join @@ Option.map ~f:(fun e -> e.solved) @@ Map.find env.dim_env v

  let occurs_dim v = function Dim _ -> false | Var v' -> equal_dim_var v v'

  let s_row_one v ~value:{ dims = more_dims; constr = more_constr; row; id = _ } ~in_ =
    match in_ with
    | { dims; constr; row = Row_var v2; id } when v = v2 ->
        let more_constr =
          match more_constr with
          | Unconstrained -> Unconstrained
          | Total_elems m ->
              if List.for_all dims ~f:is_dim then
                Total_elems (m * List.fold dims ~init:1 ~f:(fun n d -> n * dim_to_int_exn d))
              else Unconstrained (* Wait for more shape inference. *)
        in
        { dims = more_dims @ dims; constr = meet more_constr constr; row; id }
    | _ -> in_

  let s_row_one_in_entry v ~value { cur; subr; solved } =
    let upd m x = m x ~f:(fun in_ -> s_row_one v ~value ~in_) in
    { cur = upd List.map cur; subr = upd List.map subr; solved = upd Option.map solved }

  let subst_row ?freshen_proj env { dims; constr; row; id } =
    let dims = List.map dims ~f:(subst_dim ?freshen_proj env) in
    match row with
    | Broadcastable -> { dims; constr; row; id }
    | Row_var v -> (
        match Map.find env.row_env v with
        | None | Some { solved = None; _ } -> { dims; constr; row; id }
        | Some { solved = Some { dims = more_dims; constr = Unconstrained; row; id = _ }; _ } ->
            { dims = more_dims @ dims; constr; row; id }
        | Some { solved = Some { dims = more_dims; constr = Total_elems m; row; id = _ }; _ } ->
            let more_constr =
              if List.for_all dims ~f:is_dim then
                Total_elems (m * List.fold dims ~init:1 ~f:(fun n d -> n * dim_to_int_exn d))
              else Unconstrained (* Wait for more shape inference. *)
            in
            { dims = more_dims @ dims; constr = meet more_constr constr; row; id })

  let occurs_row v = function { row = Row_var v'; _ } -> v = v' | _ -> false

  (* Note: [solve_dim_if_known] extracts a solution, it does not guarantee consistency checks for
     yet-unsolved inequality components. *)
  let solve_dim_if_known ~is_complete ~cur ~subr =
    if is_complete && List.length subr = 1 then Some (List.hd_exn subr)
    else
      let known = function Dim { d; _ } as dim -> Some (d, dim) | _ -> None in
      let compare (d1, _) (d2, _) = Int.ascending d1 d2 in
      let cur_knowns = List.sort ~compare @@ List.filter_map cur ~f:known in
      let subr_knowns = List.sort ~compare @@ List.filter_map subr ~f:known in
      let non1 (d, _) = d <> 1 in
      let non1_or_d ~d (d2, _) = d <> 1 && d <> d2 in
      let check_all_d_or_1 ~d =
        if List.exists ~f:(non1_or_d ~d) @@ cur_knowns @ subr_knowns then
          raise @@ Shape_error ("Conflicting dimensions inferred for an axis", [ Dim_mismatch (cur @ subr) ])
      in
      match (cur_knowns, List.rev subr_knowns) with
      | [], [] -> None
      | (1, dim) :: _, _ ->
          if List.exists ~f:non1 subr_knowns then
            raise
            @@ Shape_error
                 ("Axis expected to be dimension 1 actually is not dimension 1", [ Dim_mismatch subr ]);
          Some dim
      | (d, dim) :: _, [] when (* d > 1 && *) is_complete && List.is_empty subr ->
          check_all_d_or_1 ~d;
          Some dim
      | _, (d, dim) :: _ when d > 1 || (is_complete && List.length subr = List.length subr_knowns) ->
          check_all_d_or_1 ~d;
          Some dim
      | _, (_d, _) :: _ (* _d <= 1*) -> None
      | (d, _) :: _, _ ->
          if List.exists ~f:(non1_or_d ~d) @@ cur_knowns @ subr_knowns then (
            if Utils.settings.with_debug then
              Stdlib.Format.printf "WARNING: axis inferred to be dim-1 because of conflicting uses:@ %a\n%!"
                Sexp.pp_hum
                ([%sexp_of: dim list] @@ cur @ subr);
            Some (get_dim ~d:1 ()))
          else None

  let perhaps_eliminate_var ~is_complete v ~value:{ cur = v_cur; subr = v_subr; solved = v_solved } ~in_ =
    let subst ~v_side ~side =
      match (v_solved, List.partition_tf side ~f:(equal_dim @@ Var v)) with
      | _, ([], _) -> side
      | None, (v :: _, side) -> (if is_complete then [] else [ v ]) @ v_side @ side
      | Some dim, (_ :: _, side) -> dim :: side
    in
    match in_.solved with
    | Some (Var v2) when equal_dim_var v2 v && is_complete ->
        (None, { cur = v_cur @ in_.cur; subr = v_subr @ in_.subr; solved = v_solved })
    | in_sol ->
        let cur = subst ~v_side:v_cur ~side:in_.cur in
        let subr = subst ~v_side:v_subr ~side:in_.subr in
        let solved = solve_dim_if_known ~is_complete ~cur ~subr in
        (* [solved] cannot be [Some (Var v)] because v is already eliminated in subr. *)
        (Option.both in_sol solved, { cur; subr; solved = Option.first_some solved in_sol })

  let update_proj_classes pid1 pid2 env =
    { env with proj_classes = Utils.union_add ~equal:Int.equal env.proj_classes pid1 pid2 }

  let with_proj_classes proj_classes env = { env with proj_classes }

  (* Note: [unify_dim] will not resolve inequalities, requires another round of solving. *)
  let rec unify_dim ((dim1, dim2) as eq) env =
    match eq with
    | Dim { label = Some l1; _ }, Dim { label = Some l2; _ } when not (String.equal l1 l2) ->
        raise @@ Shape_error ("solved dimensions for axis: different labels", [ Dim_mismatch [ dim1; dim2 ] ])
    | Dim { d = d1; label = _; proj_id = pid1 }, Dim { d = d2; label = _; proj_id = pid2 } when d1 = d2 ->
        update_proj_classes pid1 pid2 env
    | Var v, d2 | d2, Var v -> (
        match Map.find env.dim_env v with
        | None ->
            let dim_env = Map.map env.dim_env ~f:(s_dim_one_in_entry v ~value:d2) in
            {
              env with
              dim_env = Map.add_exn dim_env ~key:v ~data:{ cur = []; subr = []; solved = Some d2 };
              row_env = Map.map env.row_env ~f:(s_dim_one_in_row_entry v ~value:d2);
              dim_rev_elim_order = v :: env.dim_rev_elim_order;
            }
        | Some { solved = Some d1; _ } -> unify_dim (d1, d2) env
        | Some { cur; subr; solved = None } ->
            let dim_env = Map.map env.dim_env ~f:(s_dim_one_in_entry v ~value:d2) in
            {
              env with
              dim_env = Map.update dim_env v ~f:(fun _ -> { cur; subr; solved = Some d2 });
              row_env = Map.map env.row_env ~f:(s_dim_one_in_row_entry v ~value:d2);
            })
    | _ ->
        (* Note: at the unify_dim stage, it's strict equality (no broadcasting). *)
        raise @@ Shape_error ("solved dimensions for axis: mismatch", [ Dim_mismatch [ dim1; dim2 ] ])

  let update_dim ?(freshen_proj = false) ~is_complete v ?cur ?subr env =
    (* Prevent the projection equivalences from leaking across [propagate_shapes update_step] invocations.
       Concluding that two axes have an equal size can span multiple update steps, and should not prevent
       them from being distinct axes in a product space. *)
    let no_v = List.filter ~f:(Fn.non @@ equal_dim (Var v)) in
    let cur = no_v @@ List.map ~f:(subst_dim ~freshen_proj env) @@ Option.to_list cur in
    let subr = no_v @@ List.map ~f:(subst_dim ~freshen_proj env) @@ Option.to_list subr in
    if List.is_empty cur && List.is_empty subr then env
    else
      match solve_dim_if_known ~is_complete ~cur ~subr with
      | Some value -> unify_dim (Var v, value) env
      | None -> (
          let value = { cur; subr; solved = None } in
          let elim = perhaps_eliminate_var ~is_complete v ~value in
          match Map.find env.dim_env v with
          | Some in_ ->
              (* Note: this is where the bulk of the work usually happens. *)
              let eq, data = elim ~in_ in
              let dim_env = Map.update env.dim_env v ~f:(fun _ -> data) in
              Option.fold eq ~init:{ env with dim_env } ~f:(Fn.flip unify_dim)
          | None ->
              let eqs = ref [] (* No fold_map in Map. *) in
              let dim_env =
                Map.map env.dim_env ~f:(fun in_ ->
                    let eq, entry = elim ~in_ in
                    eqs := Option.to_list eq @ !eqs;
                    entry)
              in
              let env = List.fold !eqs ~init:env ~f:(Fn.flip unify_dim) in
              {
                env with
                dim_env = Map.add_exn dim_env ~key:v ~data:value;
                dim_rev_elim_order = v :: env.dim_rev_elim_order;
              })

  (* Note: [solve_row_if_known] extracts a solution, it does not guarantee consistency checks for
     yet-unsolved inequality components. *)
  let solve_row_if_known ~is_complete ~cur ~subr =
    if is_complete && List.length subr = 1 then (None, Some (List.hd_exn subr))
    else
      let known = function
        | { dims; row = Broadcastable; _ } as dim -> Some (List.length dims, dim)
        | _ -> None
      in
      let unknown = function
        | { dims; row = Row_var _; _ } as dim -> Some (List.length dims, dim)
        | _ -> None
      in
      let compare (d1, _) (d2, _) = Int.ascending d1 d2 in
      let cur_knowns = List.sort ~compare @@ List.filter_map cur ~f:known in
      let subr_knowns = List.sort ~compare @@ List.filter_map subr ~f:known in
      let subr_unknowns = List.sort ~compare @@ List.filter_map subr ~f:unknown in
      match (cur_knowns, List.rev subr_knowns, List.rev subr_unknowns) with
      | [], [], _ -> (None, None)
      | (_, row) :: _, [], [] when is_complete -> (None, Some row)
      | (d1, _) :: _, _, (d2, _) :: _ when d1 < d2 ->
          raise
          @@ Shape_error ("Expected number of axes is smaller than the actual", [ Row_mismatch (cur @ subr) ])
      | (d1, _) :: _, (d2, _) :: _, _ when d1 < d2 ->
          raise
          @@ Shape_error ("Expected number of axes is smaller than the actual", [ Row_mismatch (cur @ subr) ])
      | (d1, row1) :: _, _, (d2, row2) :: _ when d1 = d2 -> (Some (row2, row1), Some row1)
      | (d1, row1) :: _, (d2, _) :: _, _ when d1 = d2 -> (None, Some row1)
      | _, (_, row) :: _, [] when is_complete -> (None, Some row)
      | _ -> (None, None)

  let meet_all = List.fold ~init:Unconstrained ~f:(fun c r -> meet c r.constr)

  let perhaps_eliminate_row_var ~is_complete v ~value:{ cur = v_cur; subr = v_subr; solved = v_solved } ~in_ =
    (* For now, this is the same as [perhaps_eliminate_var], except it can return two equations:
       one coming from [solve_row_if_known]. *)
    let subst ~v_side ~side =
      match (v_solved, List.partition_tf side ~f:(fun r -> equal_row r.row @@ Row_var v)) with
      | _, ([], _) -> side
      | None, (v :: _, side) -> (if is_complete then [] else [ v ]) @ v_side @ side
      | Some dim, (_ :: _, side) -> dim :: side
    in
    match in_.solved with
    | Some { row = Row_var v2; _ } when v2 = v && is_complete ->
        ([], { cur = v_cur @ in_.cur; subr = v_subr @ in_.subr; solved = v_solved })
    | in_sol ->
        let cur = subst ~v_side:v_cur ~side:in_.cur in
        let subr = subst ~v_side:v_subr ~side:in_.subr in
        let eq, solved = solve_row_if_known ~is_complete ~cur ~subr in
        (* [solved] cannot be [Some {row=Row_var v; _}] because v is already eliminated in subr. *)
        ( Option.(to_list eq @ to_list @@ both in_sol solved),
          { cur; subr; solved = Option.first_some solved in_sol } )

  let drop_from_end l n = List.rev @@ List.drop (List.rev l) n
  let take_from_end l n = List.rev @@ List.take (List.rev l) n

  let apply_constraint r env =
    let r = subst_row env r in
    match r.constr with
    | Unconstrained -> env
    | Total_elems n -> (
        match r.row with
        | Row_var _ -> env (* Wait for more shape inference. *)
        | Broadcastable -> (
            let dims = List.map r.dims ~f:(subst_dim env) in
            let vars, nonvars = List.partition_tf dims ~f:is_var in
            if List.length vars > 1 then env (* Wait for more shape inference. *)
            else
              let known = List.fold nonvars ~init:1 ~f:(fun n d -> n * dim_to_int_exn d) in
              match vars with
              | [] ->
                  if n <> known then (
                    if Utils.settings.with_debug then
                      Stdlib.Format.printf "Env.apply_constraint: shape error env=@ %a\n%!" Sexp.pp_hum
                        (sexp_of_t env);
                    raise @@ Shape_error ("Total_elems constraint failed", [ Row_mismatch [ r ] ]))
                  else env
              | [ Var v ] ->
                  let rem = n / known in
                  if rem = 0 then (
                    if Utils.settings.with_debug then
                      Stdlib.Format.printf "Env.apply_constraint: shape error env=@ %a\n%!" Sexp.pp_hum
                        (sexp_of_t env);
                    raise @@ Shape_error ("Total_elems constraint failed", [ Row_mismatch [ r ] ]))
                  else unify_dim (Var v, get_dim ~d:rem ()) env
              | _ -> assert false))

  let prefix_constraint ~drop row =
    match row.constr with
    | Unconstrained -> Unconstrained
    | Total_elems n ->
        let prefix = take_from_end row.dims drop in
        if List.exists prefix ~f:(Fn.non is_dim) then (* wait for more inference *) Unconstrained
        else
          let discarded = List.fold prefix ~init:1 ~f:(fun n d -> n * dim_to_int_exn d) in
          let result = n / discarded in
          if result <= 0 then raise @@ Shape_error ("Not enough elements", [ Row_mismatch [ row ] ])
          else Total_elems result

  (* Equate two rows, no broadcasting. Does not resolve inequalities, requires another round of solving. *)
  let rec unify_row ((row1, row2) as eq) env =
    let unify_prefix len =
      let dims1 = take_from_end row1.dims len and dims2 = take_from_end row2.dims len in
      List.fold ~init:env ~f:(Fn.flip unify_dim) @@ List.zip_exn dims1 dims2
    in
    match eq with
    | { row = Row_var v; dims = r1_dims; id; constr = _ }, r2
    | r2, { row = Row_var v; dims = r1_dims; id; constr = _ } -> (
        let r1_len = List.length r1_dims and r2_len = List.length r2.dims in
        if r1_len > r2_len then
          if is_row_var r2.row then unify_row (row2, row1) env
          else raise @@ Shape_error ("Number of axes mismatch", [ Row_mismatch [ row1; row2 ] ])
        else
          let env =
            try unify_prefix r1_len
            with Shape_error (s, trace) -> raise @@ Shape_error (s, Row_mismatch [ row1; row2 ] :: trace)
          in
          let constr = meet row1.constr @@ prefix_constraint ~drop:r1_len row2 in
          let value = { constr; row = r2.row; dims = drop_from_end r2.dims r1_len; id } in
          match Map.find env.row_env v with
          | None ->
              let row_env = Map.map env.row_env ~f:(s_row_one_in_entry v ~value) in
              {
                env with
                row_env = Map.add_exn row_env ~key:v ~data:{ cur = []; subr = []; solved = Some value };
                row_rev_elim_order = v :: env.row_rev_elim_order;
              }
          | Some { solved = Some r1; _ } -> unify_row (r1, value) env
          | Some { cur; subr; solved = None } ->
              let row_env = Map.map env.row_env ~f:(s_row_one_in_entry v ~value) in
              { env with row_env = Map.update row_env v ~f:(fun _ -> { cur; subr; solved = Some value }) })
    | ( { row = Broadcastable; dims = dims1; constr = constr1; id = _ },
        { row = Broadcastable; dims = dims2; constr = constr2; id = _ } ) ->
        let env =
          match List.zip dims1 dims2 with
          | Unequal_lengths ->
              raise @@ Shape_error ("Mismatching number of axes", [ Row_mismatch [ row1; row2 ] ])
          | Ok eqs -> List.fold ~init:env ~f:(Fn.flip unify_dim) eqs
        in
        apply_constraint { row1 with constr = meet constr1 constr2 } env

  let update_row ?(freshen_proj = false) ~is_complete v ?cur ?subr env =
    (* This is the same as [update_dim] except dealing with more potential side equations. *)
    let no_v =
      List.filter ~f:(fun r ->
          if equal_row (Row_var v) r.row then
            if List.is_empty r.dims then false
            else raise @@ Shape_error ("Infinite row via self-reference", [ Row_mismatch [ r ] ])
          else true)
    in
    let cur = no_v @@ List.map ~f:(subst_row ~freshen_proj env) @@ Option.to_list cur in
    let subr = no_v @@ List.map ~f:(subst_row ~freshen_proj env) @@ Option.to_list subr in
    if List.is_empty cur && List.is_empty subr then env
    else
      let guessed_id = (List.hd_exn @@ cur @ subr).id in
      match solve_row_if_known ~is_complete ~cur ~subr with
      | extra_eq, Some value ->
          let eqs =
            ({ dims = []; row = Row_var v; constr = Unconstrained; id = guessed_id }, value)
            :: Option.to_list extra_eq
          in
          List.fold ~init:env ~f:(Fn.flip unify_row) eqs
      | extra_eq, None -> (
          let value = { cur; subr; solved = None } in
          let elim = perhaps_eliminate_row_var ~is_complete v ~value in
          match Map.find env.row_env v with
          | Some in_ ->
              (* Note: this is where the bulk of the work usually happens. *)
              let eq, data = elim ~in_ in
              let eqs = eq @ Option.to_list extra_eq in
              let row_env = Map.update env.row_env v ~f:(fun _ -> data) in
              List.fold eqs ~init:{ env with row_env } ~f:(Fn.flip unify_row)
          | None ->
              let eqs = ref @@ Option.to_list extra_eq (* No fold_map in Map. *) in
              let row_env =
                Map.map env.row_env ~f:(fun in_ ->
                    let eq, entry = elim ~in_ in
                    eqs := eq @ !eqs;
                    entry)
              in
              let env = List.fold !eqs ~init:env ~f:(Fn.flip unify_row) in
              {
                env with
                row_env = Map.add_exn row_env ~key:v ~data:value;
                row_rev_elim_order = v :: env.row_rev_elim_order;
              })

  let empty_env =
    {
      dim_env = Map.empty (module Dim_var);
      row_env = Map.empty (module Int);
      proj_classes = Map.empty (module Int);
      (* The state's proj_classes come from the most recent propagate_shapes and are not used across calls
         to propagate_shapes. *)
      dim_rev_elim_order = [];
      row_rev_elim_order = [];
    }

  let merge_fresh_proj ~is_complete ~update ~state =
    let state =
      Map.fold ~init:state
        ~f:(fun ~key ~data env -> update_dim ~freshen_proj:true ~is_complete key data env)
        update.dim_env
    in
    let state =
      Map.fold ~init:state
        ~f:(fun ~key ~data env -> update_row ~freshen_proj:true ~is_complete key data env)
        update.row_env
    in
    state
end

type proj_environment = {
  proj_classes : Env.proj_classes;
  proj_env : Arrayjit.Indexing.axis_index Map.M(Dim_var).t;
}
[@@deriving sexp]

let empty_proj_environment = { proj_classes = Map.empty (module Int); proj_env = Map.empty (module Dim_var) }

type update_step = { shape : t; logic : logic; mutable env : proj_environment } [@@deriving sexp]
(** Data required for a shape inference update step. Ideally, an update should be performed at least twice,
    the second time after all the other relevant updates have been performed for the first time.
    In OCANNL, this is achieved by performing updates both as the tensors are constructed, and via
    lazy callbacks as the corresponding [Arrayjit.Indexing] dimensions and projections are first accessed. *)

let with_error_trace = ref true

type environment = Env.t [@@deriving sexp]
type dim_eq = { d1 : dim; d2 : dim } [@@deriving sexp, equal, hash, compare]
type dim_eqs = dim_eq list [@@deriving sexp]

type row_eq = { r : dims; subr : dims } [@@deriving sexp, equal]
(** Where applicable, [subr] comes from a subtensor of [r]. *)

type row_eqs = row_eq list [@@deriving sexp, equal]

module Debug_runtime = Utils.Debug_PrintBox ()

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

let axes_spec_to_dims_bio ?b_row ?i_row ?o_row ~sh_id ~f labels =
  let b_dims, i_dims, o_dims = axis_map_to_dims_bio labels.labels in
  let vars = Hashtbl.create (module String) in
  let to_dim kind = Array.(Fn.compose to_list @@ map ~f:(f kind vars)) in
  let upd_row = function None, true -> Some (get_row_var ()) | old, true -> old | _, false -> None in
  let b_row = upd_row (b_row, labels.bcast_batch) in
  let i_row = upd_row (i_row, labels.bcast_input) in
  let o_row = upd_row (o_row, labels.bcast_output) in
  let to_row v = Option.value v ~default:Broadcastable in
  let batch =
    {
      dims = to_dim AxisKey.Batch b_dims;
      constr = Unconstrained;
      row = to_row b_row;
      id = { sh_id; kind = AxisKey.Batch };
    }
  in
  let input =
    {
      dims = to_dim AxisKey.Input i_dims;
      constr = Unconstrained;
      row = to_row i_row;
      id = { sh_id; kind = AxisKey.Input };
    }
  in
  let output =
    {
      dims = to_dim AxisKey.Output o_dims;
      constr = Unconstrained;
      row = to_row o_row;
      id = { sh_id; kind = AxisKey.Output };
    }
  in
  (b_row, i_row, o_row, batch, input, output)

let einsum_slot_spec_to_dims_bio ~generative ?b_row ?i_row ?o_row ~sh_id labels =
  let equal = AxisKey.equal_kind in
  let proj_env_update = ref @@ Map.empty (module Dim_var) in
  let f kind vars = function
    | Either.First label -> Var (Hashtbl.find_or_add vars label ~default:(fun () -> get_var ~label ()))
    | Second 0 when Option.value ~default:false @@ List.Assoc.find generative ~equal kind -> get_dim ~d:1 ()
    | Second i ->
        let var = get_var () in
        proj_env_update := Map.add_exn !proj_env_update ~key:var ~data:(Arrayjit.Indexing.Fixed_idx i);
        Var var
  in
  let result = axes_spec_to_dims_bio ?b_row ?i_row ?o_row ~f ~sh_id labels in
  (!proj_env_update, result)

let unify_shapes (env : environment) ({ shape = cur_sh; logic; env = _ } as update_step : update_step) :
    environment =
  let row_eq_side kind row = { dims = []; constr = Unconstrained; row; id = { sh_id = cur_sh.id; kind } } in
  let row_eq ~kind_r ~r ~kind_subr ~subr =
    Option.to_list
    @@ Option.map2 r subr ~f:(fun r subr -> { r = row_eq_side kind_r r; subr = row_eq_side kind_subr subr })
  in
  let dims_label_assoc dims =
    let f = function Var { label = Some l; _ } as d -> Some (l, d) | _ -> None in
    List.filter_map dims.dims ~f
  in
  let dim_assoc_eqs assoc =
    List.Assoc.sort_and_group assoc ~compare:String.compare
    |> List.concat_map ~f:(function
         | _, [] -> assert false
         | _, d1 :: ds -> List.map ds ~f:(fun d2 -> { d1; d2 }))
  in
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
        {
          dims = [];
          constr = Total_elems batch_elems;
          row = get_row_var ();
          id = { sh_id = cur_sh.id; kind = Batch };
        }
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
        {
          dims = [];
          constr = Total_elems batch_elems;
          row = get_row_var ();
          id = { sh_id = cur_sh.id; kind = Batch };
        }
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
            { r = sh1.input; subr = sh2.output };
            { r = cur_sh.batch; subr = sh1.batch };
            { r = cur_sh.batch; subr = sh2.batch };
            { r = cur_sh.input; subr = sh2.input };
            { r = cur_sh.output; subr = sh1.output };
          ]
          env
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ("Compose / " ^ s, Shape_mismatch [ cur_sh; sh1; sh2 ] :: trace))
  | Broadcast (Pointwise_bin, sh1, sh2) -> (
      try
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
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ("Pointwise binary / " ^ s, Shape_mismatch [ cur_sh; sh1; sh2 ] :: trace))
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
                id = { sh_id = cur_sh.id; kind = Batch };
              }
            in
            ( Option.to_list static_range
              |> List.map ~f:(fun range -> { d1 = get_dim ~d:range (); d2 = slice_var }),
              { r = expanded_batch; subr = sh.batch } )
          else
            match sh.batch.dims with
            | [] ->
                raise
                @@ Shape_error
                     ("Batch slice: insufficent number of batch axes", [ Shape_mismatch [ cur_sh; sh ] ])
            | d2 :: dims ->
                let reduced_batch =
                  {
                    dims;
                    constr = Unconstrained;
                    row = sh.batch.row;
                    id = { sh_id = cur_sh.id; kind = Batch };
                  }
                in
                ( Option.to_list static_range |> List.map ~f:(fun range -> { d1 = get_dim ~d:range (); d2 }),
                  { r = cur_sh.batch; subr = reduced_batch } )
        in
        try
          unify_dim range_eq env |> Env.apply_constraint cur_sh.batch
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
      let proj_env_rhs, (b_row_rhs, i_row_rhs, o_row_rhs, b_rhs, i_rhs, o_rhs) =
        einsum_slot_spec_to_dims_bio ~generative:[] ~sh_id:sh.id ls_rhs
      in
      let proj_env_lhs, (b_row_lhs, i_row_lhs, o_row_lhs, b_lhs, i_lhs, o_lhs) =
        einsum_slot_spec_to_dims_bio ~generative ?b_row:b_row_rhs ?i_row:i_row_rhs ?o_row:o_row_rhs
          ~sh_id:cur_sh.id ls_lhs
      in
      let label_groups = List.concat_map ~f:dims_label_assoc [ b_lhs; i_lhs; o_lhs; b_rhs; i_rhs; o_rhs ] in
      let proj_env =
        let combine ~key:_ _ _ = assert false in
        Map.merge_skewed ~combine proj_env_rhs proj_env_lhs
      in
      (* Forget the old proj_env as it is not relevant after a propagate_shapes call completes. *)
      update_step.env <- { update_step.env with proj_env };
      try
        unify_dims
          ({ r = cur_sh.batch; subr = b_lhs } :: { r = b_rhs; subr = sh.batch }
           :: { r = cur_sh.input; subr = i_lhs } :: { r = i_rhs; subr = sh.input }
           :: { r = cur_sh.output; subr = o_lhs } :: { r = o_rhs; subr = sh.output }
           :: row_eq ~kind_r:Batch ~r:b_row_lhs ~kind_subr:Batch ~subr:b_row_rhs
          @ row_eq ~kind_r:Input ~r:i_row_lhs ~kind_subr:Input ~subr:i_row_rhs
          @ row_eq ~kind_r:Output ~r:o_row_lhs ~kind_subr:Output ~subr:o_row_rhs)
        @@ unify_dim (dim_assoc_eqs label_groups) env
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
      let proj_env_rhs1, (b_row_rhs1, i_row_rhs1, o_row_rhs1, b_rhs1, i_rhs1, o_rhs1) =
        einsum_slot_spec_to_dims_bio ~generative:[] ~sh_id:sh1.id ls_rhs1
      in
      let proj_env_rhs2, (b_row_rhs2, i_row_rhs2, o_row_rhs2, b_rhs2, i_rhs2, o_rhs2) =
        einsum_slot_spec_to_dims_bio ~generative:[] ?b_row:b_row_rhs1 ?i_row:i_row_rhs1 ?o_row:o_row_rhs1
          ~sh_id:sh2.id ls_rhs2
      in
      let proj_env_lhs, (b_row_lhs, i_row_lhs, o_row_lhs, b_lhs, i_lhs, o_lhs) =
        einsum_slot_spec_to_dims_bio ~generative ?b_row:b_row_rhs2 ?i_row:i_row_rhs2 ?o_row:o_row_rhs2
          ~sh_id:cur_sh.id ls_lhs
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
      update_step.env <- { update_step.env with proj_env };
      let eqs =
        { r = cur_sh.batch; subr = b_lhs } :: { r = b_rhs1; subr = sh1.batch }
        :: { r = b_rhs2; subr = sh2.batch } :: { r = cur_sh.input; subr = i_lhs }
        :: { r = i_rhs1; subr = sh1.input } :: { r = i_rhs2; subr = sh2.input }
        :: { r = cur_sh.output; subr = o_lhs } :: { r = o_rhs1; subr = sh1.output }
        :: { r = o_rhs2; subr = sh2.output }
        :: row_eq ~kind_r:Batch ~r:b_row_lhs ~kind_subr:Batch ~subr:b_row_rhs1
        @ row_eq ~kind_r:Input ~r:i_row_lhs ~kind_subr:Input ~subr:i_row_rhs1
        @ row_eq ~kind_r:Output ~r:o_row_lhs ~kind_subr:Output ~subr:o_row_rhs1
        @ row_eq ~kind_r:Batch ~r:b_row_lhs ~kind_subr:Batch ~subr:b_row_rhs2
        @ row_eq ~kind_r:Input ~r:i_row_lhs ~kind_subr:Input ~subr:i_row_rhs2
        @ row_eq ~kind_r:Output ~r:o_row_lhs ~kind_subr:Output ~subr:o_row_rhs2
      in
      try unify_dims eqs @@ unify_dim (dim_assoc_eqs label_groups) env
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

let state = ref Env.empty_env
let second_stage_inference = ref []

let rec close_row_broadcast env row =
  let row = Env.subst_row env row in
  let rec f env = function
    | Var v when Set.mem env.Env.broadcast.dim_vars v -> (
        match Map.find env.dim_env v with
        | None -> Env.update_dim v (get_dim ~d:1 ()) env
        | Some dim -> f env dim)
    | _ -> env
  in
  match row with
  | { dims; constr; row = Row_var v; id } when Set.mem env.Env.broadcast.row_vars v -> (
      match Map.find env.row_env v with
      | None ->
          let init = Env.update_row v { dims = []; constr; row = Broadcastable; id } env in
          List.fold dims ~f ~init
      | Some row -> close_row_broadcast env row)
  | { dims; _ } -> List.fold dims ~f ~init:env

(** Uses the matrix convention of putting the input axes last.
    Note: [force_to_dims] is "destructive": it closes shapes that remain incomplete after inference. *)
let close_shape_broadcast (sh : t) (env : environment) : environment =
  List.fold ~init:env ~f:close_row_broadcast [ sh.batch; sh.output; sh.input ]

let deep_copy_update_step update_step =
  let upd sh = { sh with id = sh.id } in
  {
    update_step with
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
    sh.batch <- Env.subst_row env sh.batch;
    sh.input <- Env.subst_row env sh.input;
    sh.output <- Env.subst_row env sh.output
  in
  iter_shapes update_step ~f

let%debug_sexp propagate_shapes (update_step : update_step) : unit =
  if not @@ List.mem ~equal:phys_equal !second_stage_inference update_step then
    second_stage_inference := update_step :: !second_stage_inference;
  (* Update dimension information coming from other propagation steps. *)
  apply_env update_step !state;
  let _debug_initial : update_step = deep_copy_update_step update_step in
  let env =
    unify_shapes
      (Env.with_proj_classes_and_broadcast update_step.env.proj_classes !state.broadcast Env.empty_env)
      update_step
  in
  (* Update both dimension and projections information (i.e. keep the update step's projections). *)
  apply_env update_step env;
  update_step.env <- { update_step.env with proj_classes = env.proj_classes };
  (* "Forget" the projections information of this propagation step to not contaminate other steps. *)
  let _debug_result : update_step = deep_copy_update_step update_step in
  let _debug_env_state : environment = !state in
  Debug_runtime.no_debug_if
    (equal _debug_initial.shape _debug_result.shape && equal_logic _debug_initial.logic _debug_result.logic);
  state := Env.merge_fresh_proj ~update:env ~state:!state

let finish_inference () =
  List.iter !second_stage_inference ~f:propagate_shapes;
  let f update_step =
    let f sh = state := close_shape_broadcast sh !state in
    iter_shapes update_step ~f;
    apply_env update_step !state
  in
  List.iter !second_stage_inference ~f;
  second_stage_inference := []

let row_to_dims row =
  let row = Env.subst_row !state row in
  let f = function
    | Dim { d; _ } -> d
    | Var _ as dim ->
        raise @@ Shape_error ("Not enough shape information: unresolved variable", [ Dim_mismatch [ dim ] ])
  in
  match row with
  | { row = Row_var _; _ } ->
      raise @@ Shape_error ("Not enough shape information: unresolved row variable", [ Row_mismatch [ row ] ])
  | { dims; constr = _; row = Broadcastable; id = _ } -> Array.of_list_map dims ~f

(** Uses the matrix convention of putting the input axes last.
    Note: [force_to_dims] is "destructive": it closes shapes that remain incomplete after inference. *)
let to_dims (sh : t) : int array =
  try Array.concat_map ~f:row_to_dims [| sh.batch; sh.output; sh.input |]
  with Shape_error (s, trace) -> raise @@ Shape_error (s, Shape_mismatch [ sh ] :: trace)

let rec row_to_labels env =
  let rec f = function
    | Dim { label = Some l; _ } -> l
    | Dim { label = None; _ } -> ""
    | Var v -> (
        match Map.find env.Env.dim_env v with None -> Option.value v.label ~default:"" | Some dim -> f dim)
  in
  function
  | { dims; constr; row = Row_var v; id } -> (
      match Map.find env.row_env v with
      | None -> Array.of_list_map dims ~f
      | Some row2 -> row_to_labels env { dims = row2.dims @ dims; constr; row = row2.row; id })
  | { dims; constr = _; row = Broadcastable; id = _ } -> Array.of_list_map dims ~f

(** Uses the matrix convention of putting the input axes last. *)
let to_labels (sh : t) : string array =
  Array.concat_map ~f:(row_to_labels !state) [| sh.batch; sh.output; sh.input |]

(** *** Projection inference *** *)

open Arrayjit.Indexing

(** Computes the indexing into subtensors given the shape information of a tensor. 
    [derive_projections] should only be invoked when the shapes are fully inferred already! *)
let derive_projections (update_step : update_step) : projections =
  let dims_of sh = sh.batch.dims @ sh.output.dims @ sh.input.dims in
  let lhs = update_step.shape in
  let project rhs =
    let lhs_dims = to_dims lhs in
    let rhs_dims = Array.of_list_map ~f:to_dims rhs in
    let all_dims = List.concat_map ~f:dims_of @@ (lhs :: rhs) in
    let proj_repr proj_id =
      fst @@ Utils.union_find ~equal:Int.equal update_step.env.proj_classes ~key:proj_id ~rank:0
    in
    (* Since shapes are already inferred, these variables unify directly with the proj_id of this operation. *)
    let constrained_projs =
      Map.to_alist update_step.env.proj_env
      |> List.filter_map ~f:(fun (v, idx) ->
             match Map.find !state.dim_env v with
             | Some (Dim { proj_id; _ }) -> Some (proj_id, idx)
             | other ->
                 if Utils.settings.with_debug then
                   Stdlib.Format.printf
                     "derive_projections: unresolved variable %a for projection constraints=@ %a\n%!"
                     Sexp.pp_hum (sexp_of_dim_var v) Sexp.pp_hum
                     ([%sexp_of: dim option] other);
                 None)
      |> Map.of_alist_multi (module Int)
      |> Map.map ~f:(Utils.unique_keep_first ~equal:equal_axis_index)
      |> Map.map ~f:(function
           | [] -> assert false
           | [ idx ] -> idx
           | idcs ->
               raise @@ Shape_error ("Multiple constraints on the same projection", [ Index_mismatch idcs ]))
    in
    let rec get_product_proj = function
      | Dim { proj_id; _ } when Map.mem constrained_projs proj_id -> None
      | Dim { d; proj_id; _ } -> if iterated d then Some (proj_repr proj_id, d) else None
      | Var v as dim -> (
          match Map.find !state.dim_env v with
          | None ->
              raise
              @@ Shape_error
                   ( "derive_projections: shape still not fully inferred",
                     [ Shape_mismatch (lhs :: rhs); Dim_mismatch [ dim ] ] )
          | Some dim -> get_product_proj dim)
    in
    (* Note: the ordering will affect performance of naive backends. *)
    let all_product_projs =
      Utils.unique_keep_first ~equal:(fun (p, _) (q, _) -> p = q)
      @@ List.filter_map all_dims ~f:get_product_proj
    in
    let product_iterators = List.map all_product_projs ~f:(fun (p, _) -> (p, get_symbol ())) in
    let product_space = Array.of_list_map all_product_projs ~f:snd in
    let rec get_slot_proj = function
      | Dim { proj_id; _ } when Map.mem constrained_projs proj_id -> Map.find_exn constrained_projs proj_id
      | Dim { d; proj_id; _ } ->
          if iterated d then
            Iterator (List.Assoc.find_exn product_iterators ~equal:Int.equal (proj_repr proj_id))
          else Fixed_idx 0
      | Var v as dim -> (
          match Map.find !state.dim_env v with
          | None ->
              raise
              @@ Shape_error
                   ( "derive_projections: shape still not fully inferred",
                     [ Shape_mismatch (lhs :: rhs); Dim_mismatch [ dim ] ] )
          | Some dim -> get_slot_proj dim)
    in
    let product_iterators = Array.of_list_map product_iterators ~f:snd in
    let f (sh : t) : axis_index array = Array.of_list_map (dims_of sh) ~f:get_slot_proj in
    {
      product_space;
      lhs_dims;
      rhs_dims;
      product_iterators;
      project_lhs = f lhs;
      project_rhs = Array.of_list_map ~f rhs;
      debug_info =
        {
          spec = logic_to_spec update_step.logic;
          derived_for = sexp_of_update_step update_step;
          trace = [ ("derive_projections", unique_debug_id ()) ];
        };
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
  let make_dims kind ds =
    {
      dims = List.map ~f:(fun d -> get_dim ~d ()) ds;
      constr = Unconstrained;
      row = Broadcastable;
      id = { sh_id = id; kind };
    }
  in
  let make_axes kind ds =
    {
      dims = List.map ~f:(fun (label, d) -> get_dim ~d ~label ()) ds;
      constr = Unconstrained;
      row = Broadcastable;
      id = { sh_id = id; kind };
    }
  in
  let make_unknown kind =
    { dims = []; constr = Unconstrained; row = get_row_var (); id = { sh_id = id; kind } }
  in
  let batch =
    match (batch_dims, batch_axes) with
    | Some batch_dims, None -> make_dims Batch batch_dims
    | None, Some batch_axes -> make_axes Batch batch_axes
    | None, None -> make_unknown Batch
    | Some _, Some _ -> invalid_arg "Shape.make: do not provide both batch_dims, batch_axes"
  in
  let input =
    match (input_dims, input_axes) with
    | Some input_dims, None -> make_dims Input input_dims
    | None, Some input_axes -> make_axes Input input_axes
    | None, None -> make_unknown Input
    | Some _, Some _ -> invalid_arg "Shape.make: do not provide both input_dims, input_axes"
  in
  let output =
    match (output_dims, output_axes) with
    | Some output_dims, None -> make_dims Output output_dims
    | None, Some output_axes -> make_axes Output output_axes
    | None, None -> make_unknown Output
    | Some _, Some _ -> invalid_arg "Shape.make: do not provide both output_dims, output_axes"
  in
  let result = { input; output; batch; id; debug_name } in
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

let of_spec ?(deduced = Not_constrained) ~debug_name ~id spec =
  let _, _, _, batch, input, output = shape_spec_to_dims_bio ~sh_id:id @@ axis_labels_of_spec spec in
  let result = { input; output; batch; id; debug_name } in
  (match deduced with
  | Not_constrained -> ()
  | Input_equals_output -> (
      try state := unify_dims [ { r = input; subr = output } ] !state
      with Shape_error (s, trace) when !with_error_trace ->
        raise @@ Shape_error ("of spec / " ^ s, Shape_mismatch [ result ] :: trace)));
  result

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

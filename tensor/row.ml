(** The row type, shape inference related types and constraint solving. *)

open Base

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_ROW=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_ROW"]

type axis_padding = Ir.Ops.axis_padding [@@deriving equal, sexp]

module Dim_var = struct
  type t = { id : int; name : string option [@compare.ignore] [@equal.ignore] [@hash.ignore] }
  [@@deriving equal, hash, compare, sexp]

  let to_string { id; name } =
    match name with None -> "$" ^ Int.to_string id | Some n -> [%string "$%{id#Int}:%{n}"]

  include Comparator.Make (struct
    type nonrec t = t

    let compare = compare
    let sexp_of_t = sexp_of_t
  end)
end

type dim_var = Dim_var.t [@@deriving equal, hash, compare, sexp]
type dim_cmp = Dim_var.comparator_witness
type dim_var_set = Set.M(Dim_var).t [@@deriving equal, hash, compare, sexp]
type 'a dim_map = 'a Map.M(Dim_var).t [@@deriving equal, sexp]

module Proj_id = struct
  type t = Proj_id of int [@@deriving equal, hash, compare, sexp]

  let to_string (Proj_id i) = Int.to_string i

  let fresh =
    let uid = ref 0 in
    fun () ->
      Int.incr uid;
      Proj_id !uid

  include Comparator.Make (struct
    type nonrec t = t

    let compare = compare
    let sexp_of_t = sexp_of_t
  end)
end

type proj_id = Proj_id.t [@@deriving equal, hash, compare, sexp]
type proj_cmp = Proj_id.comparator_witness
type proj_var_set = Set.M(Proj_id).t [@@deriving equal, sexp]
type 'a proj_map = 'a Map.M(Proj_id).t [@@deriving equal, sexp]

let proj_var_set_empty = Set.empty (module Proj_id)
let proj_map_empty = Map.empty (module Proj_id)
let dim_var_set_empty = Set.empty (module Dim_var)
let dim_map_empty = Map.empty (module Dim_var)

(* The reserved basis tag of the claim-free broadcast top: an axis with this tag broadcasts to
   any size while it remains size 1, and is an ordinary fixed axis otherwise. The top of the
   broadcast order is precisely [1_(bcast_if_1)]. Scalars and rank-broadening fill carry it. *)
let bcast_if_1 = "bcast_if_1"

(* The default basis tag the frontend supplies for any axis the user writes without naming a basis.
   It is an ordinary atom, incompatible with other named bases (including [bcast_if_1]). *)
let default_basis = "default"

type solved_dim = { d : int; basis : string; proj_id : proj_id option }
[@@deriving equal, hash, compare, sexp]

type convolution = { dilation : int; kernel : dim; use_padding : bool }
[@@deriving equal, hash, compare, sexp]

and dim =
  | Var of dim_var
  | Dim of solved_dim
  | Affine of { stride : int; over : dim; conv : convolution option; stride_offset : int }
  | Concat of dim list
[@@deriving equal, hash, compare, sexp]

let equal_dim d1 d2 =
  match (d1, d2) with
  | Dim { d = d1; basis = b1; proj_id = _ }, Dim { d = d2; basis = b2; proj_id = _ } ->
      d1 = d2 && String.equal b1 b2
  | _ -> equal_dim d1 d2

let uid = ref 0

let get_var ?name () : dim_var =
  Int.incr uid;
  { id = !uid; name }

(* [basis] is required (Option A totality): every minted dimension carries a tag so the compiler
   surfaces every construction site and forces a provenance decision. Use [get_bcast_dim] for the
   broadcast top (scalars, rank-broadening) and [get_default_dim] for unannotated user/derived
   atoms. *)
let get_dim ~d ~basis ?proj_id () =
  let proj_id = Option.map ~f:(fun p -> Proj_id.Proj_id p) proj_id in
  Dim { d; basis; proj_id }

(* Mint the claim-free broadcast top (or, at sizes > 1, an inert [bcast_if_1] atom). *)
let get_bcast_dim ~d ?proj_id () = get_dim ~d ~basis:bcast_if_1 ?proj_id ()

(* Mint an unannotated user/derived atom. *)
let get_default_dim ~d ?proj_id () = get_dim ~d ~basis:default_basis ?proj_id ()

(* The reserved tags ([default], [bcast_if_1]) carry no naming claim. *)
let is_reserved_basis b = String.equal b default_basis || String.equal b bcast_if_1

(* Combine the bases of two dimensions fused into one derived dimension (e.g. a convolution's
   stride and kernel). A named tag wins over a reserved (claim-free) tag; two distinct named tags
   are a conflict handled by [on_conflict]. *)
let merge_derived_basis ~on_conflict b1 b2 =
  if String.equal b1 b2 then b1
  else if is_reserved_basis b1 then b2
  else if is_reserved_basis b2 then b1
  else on_conflict ()

let effective_kernel_span ~dilation ~kernel_size =
  (* The effective kernel span with dilation is: 1 + (kernel_size - 1) * dilation This is the actual
     range of input indices covered by the kernel. *)
  1 + ((kernel_size - 1) * dilation)

let kernel_size_of_span ~dilation ~span =
  (* Inverse of effective_kernel_span. Given span, find kernel_size such that effective_kernel_span
     ~dilation ~kernel_size = span. Since effective_kernel_span = 1 + (kernel_size - 1) * dilation,
     we have: span - 1 = (kernel_size - 1) * dilation kernel_size = (span - 1) / dilation + 1 *)
  if (span - 1) % dilation <> 0 then None else Some (((span - 1) / dilation) + 1)

type 'a dim_hashtbl = 'a Hashtbl.M(Dim_var).t [@@deriving sexp]

let dim_hashtbl () = Hashtbl.create (module Dim_var)

type print_style = Only_bases | Axis_size | Axis_number_and_size | Projection_and_size
[@@deriving equal, compare, sexp]

(* Basis is total now: [Only_bases] prints the actual tag (including [default] and [bcast_if_1]) so
   the provenance split is inspectable. In size-oriented styles the reserved (claim-free) tags
   ([default] and [bcast_if_1]) — the display analog of the old blank [None] — print bare to keep
   the common shape display readable; only user-meaningful named tags print their [tag=] prefix. *)
let basis_size_prefix basis = if is_reserved_basis basis then "" else basis ^ "="

let solved_dim_to_string style { d; basis; proj_id } =
  match style with
  | Only_bases -> basis
  | Axis_size | Axis_number_and_size -> basis_size_prefix basis ^ Int.to_string d
  | Projection_and_size ->
      let size_part = Int.to_string d in
      let proj_part = match proj_id with None -> "" | Some pid -> "p" ^ Proj_id.to_string pid in
      basis_size_prefix basis ^ size_part ^ proj_part

let rec dim_to_string style = function
  | Dim solved_dim -> solved_dim_to_string style solved_dim
  | Var { id; name = Some n } -> [%string "$%{id#Int}:%{n}"]
  | Var { id; name = None } -> "$" ^ Int.to_string id
  | Affine { stride; over; conv; stride_offset } -> (
      let over_str = dim_to_string style over in
      let stride_str = if stride = 1 then over_str else Int.to_string stride ^ "*" ^ over_str in
      let offset_str =
        if stride_offset = 0 then stride_str else [%string "%{stride_str}+%{stride_offset#Int}"]
      in
      match conv with
      | None -> offset_str
      | Some { dilation; kernel; use_padding = _ } ->
          let kernel_str = dim_to_string style kernel in
          let kernel_str =
            if dilation = 1 then kernel_str else Int.to_string dilation ^ "*" ^ kernel_str
          in
          [%string "conv(%{offset_str}+%{kernel_str})"])
  | Concat dims ->
      let dims_str = List.map dims ~f:(dim_to_string style) |> String.concat ~sep:"^" in
      [%string "(%{dims_str})"]

module Row_var = struct
  type t = Row_var of int [@@deriving equal, hash, compare, sexp]

  include Comparator.Make (struct
    type nonrec t = t

    let compare = compare
    let sexp_of_t = sexp_of_t
  end)

  let get () =
    Int.incr uid;
    Row_var !uid
end

type row_var = Row_var.t [@@deriving equal, hash, compare, sexp]

let get_row_var = Row_var.get

(** A bcast specifies how axes of a single kind in a shape (i.e. the row) can adapt to other
    shapes. [Row_var v] absorbs only the middle gap between [t.beg_dims] and [t.dims];
    [Broadcastable] means no rank-broadening slack. *)
type bcast = Row_var of row_var | Broadcastable
[@@deriving equal, hash, compare, sexp, variants]

type kind = [ `Batch | `Input | `Output ] [@@deriving equal, compare, sexp, hash, variants]
type provenance_origin = { sh_id : int; kind : kind } [@@deriving sexp, compare, equal, hash]

(* List of origins, maintained as deduplicated and sorted *)
type provenance = provenance_origin list [@@deriving sexp, compare, equal, hash]

let empty_provenance = []
let provenance ~sh_id ~kind = [ { sh_id; kind } ]

(* Merge two provenances by combining and deduplicating their origins *)
let merge_provenance p1 p2 = List.dedup_and_sort ~compare:compare_provenance_origin (p1 @ p2)
(* let row_map_empty = Map.empty (module Provenance) *)

let provenance_shapes prov =
  List.map prov ~f:(fun origin -> origin.sh_id) |> List.dedup_and_sort ~compare:Int.compare

(** The row of axes of a single kind. [beg_dims] is the outer-left anchored flank (the first
    axes), [dims] is the outer-right anchored flank (the last axes). Both flanks may be non-empty
    for closed ([Broadcastable]) and open ([Row_var]) rows alike. *)
type t = { beg_dims : dim list; dims : dim list; bcast : bcast; prov : provenance }
[@@deriving equal, hash, compare, sexp]

type row = t [@@deriving equal, sexp]

let get_row_for_var prov v = { beg_dims = []; dims = []; bcast = Row_var v; prov }
let row_shapes row = provenance_shapes row.prov

let dims_basis_assoc dims =
  let f = function Var { name = Some n; _ } as d -> Some (n, d) | _ -> None in
  List.filter_map (dims.beg_dims @ dims.dims) ~f

type dim_constraint = Unconstrained_dim | At_least_dim of int
[@@deriving equal, hash, compare, sexp, variants]

type total_elems =
  | Num_elems of int
  | Strided_var of { coeff : int Utils.safe_lazy; var : dim_var; denom : int }
[@@deriving equal, hash, compare, sexp_of]

type row_constraint =
  | Unconstrained
  | Total_elems of { numerator : total_elems; divided_by : dim_var list }
  | Exact of dim list
[@@deriving equal, hash, compare, sexp_of, variants]

type constraint_origin = {
  lhs_name : string;
  lhs_kind : kind;
  rhs_name : string;
  rhs_kind : kind;
  operation : string option;
}
[@@deriving sexp_of, compare, equal]

(** An entry implements inequalities [res ⊑ v ⊑ opnd] and/or an equality [v = solved]. [res] and
    [opnd] must be sorted using the [@@deriving compare] comparison. *)
type dim_entry =
  | Solved_dim of dim
  | Bounds_dim of {
      is_in_param : bool;
      has_uniq_constr_unless : dim_var_set option;
      res : dim_var list;
      opnd : dim_var list;
      glb : dim option;
      constr : dim_constraint;
      origin : constraint_origin list;
    }
[@@deriving sexp_of]

type row_entry =
  | Solved_row of t
  | Bounds_row of {
      is_in_param : bool;
      res : row_var list;
      opnd : row_var list;
      glb : t option;
      constr : row_constraint;
      origin : constraint_origin list;
    }
[@@deriving sexp_of]

type dim_env = (dim_var, dim_entry) Utils.Tree_map.t

let sexp_of_dim_env env = Utils.Tree_map.sexp_of_t sexp_of_dim_var sexp_of_dim_entry env
let find_dim env var = Utils.Tree_map.find ~compare:compare_dim_var ~key:var env
let add_dim env ~key ~data = Utils.Tree_map.add ~compare:compare_dim_var ~key ~data env

(** Drops the [origin] field from a [dim_entry] to avoid redundant information when storing as sexp.
*)
let drop_origin (entry : dim_entry) : dim_entry =
  match entry with
  | Solved_dim _ -> entry
  | Bounds_dim { is_in_param; has_uniq_constr_unless; res; opnd; glb; constr; origin = _ } ->
      Bounds_dim { is_in_param; has_uniq_constr_unless; res; opnd; glb; constr; origin = [] }

type row_env = (row_var, row_entry) Utils.Tree_map.t

let sexp_of_row_env env = Utils.Tree_map.sexp_of_t sexp_of_row_var sexp_of_row_entry env
let find_row env var = Utils.Tree_map.find ~compare:compare_row_var ~key:var env
let add_row env ~key ~data = Utils.Tree_map.add ~compare:compare_row_var ~key ~data env

type environment = { dim_env : dim_env; row_env : row_env; discardable_vars : dim_var_set }
[@@deriving sexp_of]
(** The environment is only in resolved wrt. variables that are solved: [v -> Solved ...] do not
    appear elsewhere in the environment. In particular, per-dim and per-row constraints might not
    have been applied.

    [discardable_vars] are here for convenience -- see {!get_inequalities}. *)

let get_dim_val env var =
  match find_dim env.dim_env var with Some (Solved_dim (Dim { d; _ })) -> Some d | _ -> None

let get_row_from_env env var =
  match find_row env.row_env var with Some (Solved_row row) -> Some row | _ -> None

type constraint_ =
  | Dim_eq of { d1 : dim; d2 : dim; origin : constraint_origin list }
  | Row_eq of { r1 : t; r2 : t; origin : constraint_origin list }
  | Dim_ineq of { res : dim; opnd : dim; from_ : Sexp.t; origin : constraint_origin list }
  | Row_ineq of { res : t; opnd : t; origin : constraint_origin list }
  | Dim_constr of { d : dim; constr : dim_constraint; origin : constraint_origin list }
  | Rows_constr of { r : t list; constr : row_constraint; origin : constraint_origin list }
  | Terminal_dim of bool * dim * constraint_origin list
  | Terminal_row of bool * t * constraint_origin list
  | Shape_row of t * constraint_origin list
[@@deriving compare, equal, sexp_of, variants]

(** Drops the [origin] field from a [constraint_] to avoid redundant information when storing as
    sexp. *)
let drop_constraint_origin (c : constraint_) : constraint_ =
  match c with
  | Dim_eq { d1; d2; origin = _ } -> Dim_eq { d1; d2; origin = [] }
  | Row_eq { r1; r2; origin = _ } -> Row_eq { r1; r2; origin = [] }
  | Dim_ineq { res; opnd; from_; origin = _ } -> Dim_ineq { res; opnd; from_; origin = [] }
  | Row_ineq { res; opnd; origin = _ } -> Row_ineq { res; opnd; origin = [] }
  | Dim_constr { d; constr; origin = _ } -> Dim_constr { d; constr; origin = [] }
  | Rows_constr { r; constr; origin = _ } -> Rows_constr { r; constr; origin = [] }
  | Terminal_dim (b, d, _) -> Terminal_dim (b, d, [])
  | Terminal_row (b, r, _) -> Terminal_row (b, r, [])
  | Shape_row (r, _) -> Shape_row (r, [])

type stage = Stage1 | Stage2 | Stage3 | Stage4 | Stage5 | Stage6 | Stage7
[@@deriving sexp, equal, compare]

let is_stage2_up = function Stage1 -> false | _ -> true
let is_stage3_up = function Stage1 | Stage2 -> false | _ -> true
let is_stage4_up = function Stage1 | Stage2 | Stage3 -> false | _ -> true
let is_stage5_up = function Stage5 | Stage6 | Stage7 -> true | _ -> false
let is_stage6_up = function Stage6 | Stage7 -> true | _ -> false
let is_stage7 = function Stage7 -> true | _ -> false

module Idx = Ir.Indexing

type error_trace = ..

type error_trace +=
  | Row_mismatch of t list
  | Dim_mismatch of dim list
  | Index_mismatch of Idx.axis_index list
  | Rows_constr_failed of { constr : row_constraint }
  | Constraint_failed of constraint_

let sexp_of_error_trace = function
  | Row_mismatch rs -> Sexp.List (Sexp.Atom "Row_mismatch" :: List.map rs ~f:sexp_of_t)
  | Dim_mismatch ds -> Sexp.List (Sexp.Atom "Dim_mismatch" :: List.map ds ~f:sexp_of_dim)
  | Index_mismatch idcs ->
      Sexp.List (Sexp.Atom "Index_mismatch" :: List.map idcs ~f:Idx.sexp_of_axis_index)
  | Rows_constr_failed { constr } ->
      Sexp.List (Sexp.Atom "Rows_constr_failed" :: [ sexp_of_row_constraint constr ])
  | Constraint_failed constr ->
      Sexp.List (Sexp.Atom "Constraint_failed" :: [ sexp_of_constraint_ constr ])
  | _ -> Sexp.Atom "<outdated version of sexp_of_error_trace>"

exception Shape_error of string * error_trace list [@@deriving sexp_of]

type source = Direct | Equation | Res | Opnd [@@deriving equal, sexp]

(* Utility for merging origin lists - interleave and limit *)
let rec interleave l1 l2 =
  match (l1, l2) with [], l | l, [] -> l | h1 :: t1, h2 :: t2 -> h1 :: h2 :: interleave t1 t2

let merge_origins o1 o2 =
  let n = Int.of_string @@ Utils.get_global_arg ~default:"20" ~arg_name:"max_shape_error_origins" in
  (* First deduplicate each source independently to get unique items *)
  let unique1 = List.dedup_and_sort ~compare:compare_constraint_origin o1 in
  let unique2 = List.dedup_and_sort ~compare:compare_constraint_origin o2 in
  (* Interleave the unique items to preserve fairness *)
  let interleaved = interleave unique1 unique2 in
  (* Remove duplicates that appear in both sources while preserving order *)
  let seen = ref [] in
  let deduplicated =
    List.filter interleaved ~f:(fun item ->
        if List.mem !seen item ~equal:equal_constraint_origin then false
        else (
          seen := item :: !seen;
          true))
  in
  (* Take at most n items *)
  List.take deduplicated n

let rec dim_to_int_exn = function
  | Dim { d; _ } -> d
  | Var _ -> invalid_arg "dim_to_int: dim still unknown"
  | Affine _ -> invalid_arg "dim_to_int: affine dimension cannot be converted to single int"
  | Concat dims -> List.sum (module Int) dims ~f:dim_to_int_exn

let rec s_dim_one ?(keep_affine = false) v ~value ~in_ =
  match in_ with
  | Var v2 when equal_dim_var v v2 -> value
  | Affine { stride; over; conv; stride_offset } -> (
      let over = s_dim_one ~keep_affine v ~value ~in_:over in
      let conv =
        Option.map conv ~f:(fun { dilation; kernel; use_padding } ->
            { dilation; kernel = s_dim_one ~keep_affine v ~value ~in_:kernel; use_padding })
      in
      let result = Affine { stride; over; conv; stride_offset } in
      match result with
      | res when keep_affine -> res
      | Affine { stride = 1; over; conv = Some { use_padding = true; _ }; stride_offset = _ } ->
          over
      | Affine { stride = 1; over; conv = None; stride_offset = _ } -> over
      | Affine
          {
            stride;
            over = Dim s;
            conv = Some { dilation; kernel = Dim k; use_padding = false };
            stride_offset = _;
          } ->
          (* input_size = stride * (output_size - 1) + effective_kernel_span *)
          let span = effective_kernel_span ~dilation ~kernel_size:k.d in
          let basis =
            merge_derived_basis s.basis k.basis ~on_conflict:(fun () ->
                raise
                @@ Shape_error
                     ( "convolution: conflicting dimension bases between stride and kernel",
                       [ Dim_mismatch [ Dim s; Dim k ] ] ))
          in
          Dim { d = (stride * (s.d - 1)) + span; basis; proj_id = None }
      | Affine
          {
            stride;
            over = Dim s;
            conv = Some { kernel = Dim k; use_padding = true; _ };
            stride_offset = _;
          } ->
          let basis =
            merge_derived_basis s.basis k.basis ~on_conflict:(fun () ->
                raise
                @@ Shape_error
                     ( "convolution: conflicting dimension bases between stride and kernel",
                       [ Dim_mismatch [ Dim s; Dim k ] ] ))
          in
          Dim { d = s.d * stride; basis; proj_id = None }
      | Affine { stride; over = Dim s; conv = None; stride_offset = _ } ->
          Dim { d = s.d * stride; basis = s.basis; proj_id = None }
      | res -> res)
  | Concat dims -> (
      let dims = List.map dims ~f:(fun d -> s_dim_one ~keep_affine v ~value ~in_:d) in
      (* Filter out zero-dimension components *)
      let dims = List.filter dims ~f:(function Dim { d = 0; _ } -> false | _ -> true) in
      match dims with
      | [] -> Dim { d = 0; basis = default_basis; proj_id = None }
      | [ single ] -> single
      | res when keep_affine -> Concat res
      | _ ->
          if List.for_all dims ~f:(function Dim _ -> true | _ -> false) then (
            let solved_dims = List.filter_map dims ~f:(function Dim s -> Some s | _ -> None) in
            let total_d = List.sum (module Int) solved_dims ~f:(fun s -> s.d) in
            (* Reserved (claim-free) tags don't constrain the concat basis; all named tags must
               agree, and the concatenated axis takes that common named tag (else [default]). *)
            let named_bases =
              List.filter_map solved_dims ~f:(fun s ->
                  if is_reserved_basis s.basis then None else Some s.basis)
              |> List.dedup_and_sort ~compare:String.compare
            in
            let basis =
              match named_bases with
              | [] -> default_basis
              | [ b ] -> b
              | _ ->
                  raise
                  @@ Shape_error
                       ( "concat: conflicting dimension bases",
                         [ Dim_mismatch (List.map solved_dims ~f:(fun s -> Dim s)) ] )
            in
            Dim { d = total_d; basis; proj_id = None })
          else Concat dims)
  | Dim _ | Var _ -> in_

(* Helper functions for total_elems operations *)
let total_elems_to_string = function
  | Num_elems n -> Int.to_string n
  | Strided_var { coeff; var; denom } ->
      let coeff_string =
        if Utils.is_safe_val coeff then Int.to_string (Utils.safe_force coeff) else coeff.unique_id
      in
      let var_str = match var.name with Some n -> n | None -> "$" ^ Int.to_string var.id in
      if denom = 1 then [%string "%{coeff_string}*%{var_str}"]
      else [%string "(%{coeff_string}*%{var_str})/%{denom#Int}"]

let total_elems_divide t d =
  if d <= 0 then raise @@ Shape_error ([%string "Division by non-positive number: %{d#Int}"], [])
  else
    match t with
    | Num_elems n ->
        if n % d = 0 then Num_elems (n / d)
        else
          raise
          @@ Shape_error ([%string "Total_elems constraint: %{n#Int} not divisible by %{d#Int}"], [])
    | Strided_var { coeff; var; denom } -> Strided_var { coeff; var; denom = denom * d }

let safe_multiply coeff d = Utils.safe_map ~upd:[%string "*%{d#Int}"] ~f:(( * ) d) coeff

let total_elems_multiply t d =
  match t with
  | Num_elems n -> Num_elems (n * d)
  | Strided_var { coeff; var; denom } -> Strided_var { coeff = safe_multiply coeff d; var; denom }

let total_elems_known_zero = function
  | Num_elems n -> n <= 0
  | Strided_var { coeff; denom; _ } -> (
      denom <= 0 || match coeff.value with `Callback _ -> false | `Value v -> v <= 0)

(* Helper to remove a dimension variable from a list *)
let remove_var v vars = Utils.remove_elem ~equal:equal_dim_var v vars

(* For future flexibility *)
let dim_conjunction constr1 constr2 =
  match (constr1, constr2) with
  | Unconstrained_dim, _ -> Some ([], constr2)
  | _, Unconstrained_dim -> Some ([], constr1)
  | At_least_dim d1, At_least_dim d2 -> Some ([], At_least_dim (Int.max d1 d2))

let rec collect_dim_factors (known, vars) = function
  | Dim { d; _ } -> Some (d * known, vars)
  | Var v -> Some (known, v :: vars)
  | Affine { stride; over; conv = Some { use_padding = true; _ } | None; stride_offset = _ } ->
      Option.map
        (collect_dim_factors (known, vars) over)
        ~f:(fun (known, vars) -> (known * stride, vars))
  | _ -> None

let collect_factors dims =
  let f acc d = Result.of_option ~error:() @@ collect_dim_factors acc d in
  Result.ok @@ List.fold_result dims ~init:(1, []) ~f

let known_dims_product dims = match collect_factors dims with Some (_, []) -> true | _ -> false

let rec row_conjunction ~prov ~origin stage constr1 constr2 =
  let elems_mismatch n1 n2 =
    raise
    @@ Shape_error
         ( [%string
             "Total_elems constraint conflict: %{total_elems_to_string n1} vs. \
              %{total_elems_to_string n2}"],
           [] )
  in
  let late = is_stage2_up stage in
  match (constr1, constr2) with
  | _ when [%equal: row_constraint] constr1 constr2 -> Some ([], constr2)
  | Unconstrained, _ -> Some ([], constr2)
  | _, Unconstrained -> Some ([], constr1)
  | ( Total_elems { numerator = n1; divided_by = vars1 },
      Total_elems { numerator = n2; divided_by = vars2 } )
    when [%equal: dim_var list]
           (List.sort ~compare:compare_dim_var vars1)
           (List.sort ~compare:compare_dim_var vars2) -> (
      match (n1, n2) with
      | n1, n2 when [%equal: total_elems] n1 n2 -> Some ([], constr2)
      | Num_elems _, Num_elems _ ->
          (* Both are solved and different - this is a mismatch *)
          elems_mismatch n1 n2
      | Num_elems n, Strided_var { coeff; var; denom }
      | Strided_var { coeff; var; denom }, Num_elems n
        when late ->
          (* One is solved, one is not - we can derive an equation *)
          let coeff_val = Utils.safe_force coeff in
          (* The actual value represented is coeff * var / denom = n *)
          (* So var = n * denom / coeff *)
          if n * denom % coeff_val = 0 then
            Some
              ( [ Dim_eq { d1 = Var var; d2 = get_default_dim ~d:(n * denom / coeff_val) (); origin } ],
                constr1 )
          else
            (* n * denom is not divisible by coeff - this is a mismatch *)
            elems_mismatch n1 n2
      | ( Strided_var { coeff = c1; var = v1; denom = d1 },
          Strided_var { coeff = c2; var = v2; denom = d2 } )
        when late ->
          if equal_dim_var v1 v2 then
            (* Same variable but different coefficients/denominators - check if they're equal *)
            let val1 = Utils.safe_force c1 * d2 and val2 = Utils.safe_force c2 * d1 in
            if val1 <> val2 then elems_mismatch n1 n2 else Some ([], constr2)
          else
            (* Different variables - try to derive equations *)
            (* c1 * v1 / d1 = c2 * v2 / d2 *)
            (* c1 * v1 * d2 = c2 * v2 * d1 *)
            let c1_val = Utils.safe_force c1 in
            let c2_val = Utils.safe_force c2 in
            let lhs = c1_val * d2 in
            let rhs = c2_val * d1 in
            if lhs = rhs then Some ([], constr2)
            else if lhs % rhs = 0 then
              (* lhs = k * rhs, so c1 * v1 * d2 = k * c2 * v2 * d1 *)
              (* v1 = (k * c2 * d1) / (c1 * d2) * v2 *)
              let k = lhs / rhs in
              Some
                ( [
                    Dim_eq
                      {
                        d1 = Var v1;
                        d2 = Affine { stride = k; over = Var v2; conv = None; stride_offset = 0 };
                        origin;
                      };
                  ],
                  constr2 )
            else if rhs % lhs = 0 then
              (* rhs = k * lhs, so c2 * v2 * d1 = k * c1 * v1 * d2 *)
              (* v2 = (k * c1 * d2) / (c2 * d1) * v1 *)
              let k = rhs / lhs in
              Some
                ( [
                    Dim_eq
                      {
                        d1 = Var v2;
                        d2 = Affine { stride = k; over = Var v1; conv = None; stride_offset = 0 };
                        origin;
                      };
                  ],
                  constr1 )
            else
              (* Neither divides the other - we can't make progress *)
              (* Keep both constraints - this will be resolved later *)
              None
      | _ ->
          (* We don't want to force delayed values - keep both constraints *)
          None)
  | ( Total_elems { numerator = Strided_var { coeff = c1; var = v1; denom = _ }; divided_by = vars1 },
      constr2 )
    when List.mem vars1 v1 ~equal:equal_dim_var && late ->
      (* Variable appears in both numerator and denominator, they cancel out *)
      (* (c1 * v1 / d1) / (... * v1 * ...) = c1 / (d1 * ... * ...) *)
      let vars1' = remove_var v1 vars1 in
      row_conjunction ~prov ~origin stage
        (Total_elems { numerator = Num_elems (Utils.safe_force c1); divided_by = vars1' })
        constr2
  | ( constr2,
      Total_elems
        { numerator = Strided_var { coeff = c1; var = v1; denom = _ }; divided_by = vars1 } )
    when List.mem vars1 v1 ~equal:equal_dim_var && late ->
      let vars1' = remove_var v1 vars1 in
      row_conjunction ~prov ~origin stage
        (Total_elems { numerator = Num_elems (Utils.safe_force c1); divided_by = vars1' })
        constr2
  | ( Total_elems { numerator = n1; divided_by = vars1 },
      Total_elems { numerator = n2; divided_by = vars2 } ) ->
      (* Helper function to compute multiset difference *)
      let list_diff l1 l2 = List.fold l2 ~init:l1 ~f:(fun acc x -> remove_var x acc) in
      let vars1_only = list_diff vars1 vars2 in
      let vars2_only = list_diff vars2 vars1 in
      let extras ~keep_constr1 ?v1 ?v2 ~n1_val ~n2_val () =
        (* If we keep constr1, then it has fewer divided_by, i.e. vars1 ⊂ vars2. n1 / (product of
           vars1) = n2 / (product of vars2) Since vars1 ⊂ vars2, we have vars2 = vars1 ∪ vars2_only
           So: n1 / (product of vars1) = n2 / (product of vars1 × product of vars2_only) Thus: n1 =
           n2 / (product of vars2_only) Which means: product of vars2_only = n2 / n1 *)
        let extra_var = Option.to_list @@ if keep_constr1 then v1 else v2 in
        let num_var = if keep_constr1 then v2 else v1 in
        let diff_vars = extra_var @ if keep_constr1 then vars2_only else vars1_only in
        let n_big = if keep_constr1 then n2_val else n1_val in
        let n_small = if keep_constr1 then n1_val else n2_val in
        if n_small = 0 then
          raise @@ Shape_error ([%string "Division by zero in constraint solving"], [])
        else if n_big % n_small <> 0 then
          raise
          @@ Shape_error
               ([%string "Total_elems constraint: %{n_big#Int} not divisible by %{n_small#Int}"], [])
        else
          let quotient = n_big / n_small in
          if List.is_empty diff_vars then (
            (* No difference in variables but different numerators - this is a mismatch *)
            match num_var with
            | None -> elems_mismatch n1 n2
            | Some v ->
                if quotient <= 0 then elems_mismatch n1 n2;
                [ Dim_eq { d1 = Var v; d2 = get_default_dim ~d:quotient (); origin } ])
          else if quotient <= 0 && Option.is_none num_var then elems_mismatch n1 n2
          else if quotient = 1 && Option.is_none num_var then
            (* The difference variables must all be 1 *)
            List.map diff_vars ~f:(fun v ->
                Dim_eq { d1 = Var v; d2 = get_bcast_dim ~d:1 ~proj_id:42 (); origin })
          else
            (* The product of difference variables equals the quotient *)
            let r =
              {
                beg_dims = [];
                dims = List.map diff_vars ~f:(fun v -> Var v);
                bcast = Broadcastable;
                prov;
              }
            in
            let numerator =
              match num_var with
              | None -> Num_elems quotient
              | Some var ->
                  let coeff = Utils.{ value = `Value n_big; unique_id = Int.to_string n_big } in
                  Strided_var { coeff; var; denom = n_small }
            in
            [
              Rows_constr { r = [ r ]; constr = Total_elems { numerator; divided_by = [] }; origin };
            ]
      in
      let lazy_extras ~keep_constr1 ~num_var ?(extra_var = []) ~coeff ~denom () =
        (* If we keep constr1, then it has fewer divided_by, i.e. vars1 ⊂ vars2. n1 / (product of
           vars1) = n2 / (product of vars2) Since vars1 ⊂ vars2, we have vars2 = vars1 ∪ vars2_only
           So: n1 / (product of vars1) = n2 / (product of vars1 × product of vars2_only) Thus: n1 =
           n2 / (product of vars2_only) Which means: product of vars2_only = n2 / n1 *)
        let diff_vars = extra_var @ if keep_constr1 then vars2_only else vars1_only in
        (* The product of difference variables equals the quotient *)
        let r =
          {
            beg_dims = [];
            dims = List.map diff_vars ~f:(fun v -> Var v);
            bcast = Broadcastable;
            prov;
          }
        in
        let constr =
          Total_elems { numerator = Strided_var { coeff; var = num_var; denom }; divided_by = [] }
        in
        [ Rows_constr { r = [ r ]; constr; origin } ]
      in
      let extras ~keep_constr1 : _ option =
        match (n1, n2) with
        | Num_elems n1_val, Num_elems n2_val -> Some (extras ~keep_constr1 ~n1_val ~n2_val ())
        | ( Strided_var { coeff = c1; var = v1; denom = d1 },
            Strided_var { coeff = c2; var = v2; denom = d2 } )
          when equal_dim_var v1 v2 ->
            (* c1*v1/d1 = c2*v2/d2, and v1 = v2, so c1/d1 = c2/d2 *)
            if late then
              Some
                (extras ~keep_constr1
                   ~n1_val:(Utils.safe_force c1 * d2)
                   ~n2_val:(Utils.safe_force c2 * d1)
                   ())
            else None
        | Strided_var { coeff = c1; var = v1; denom = d1 }, Num_elems n2_val ->
            (* v1 from the numerator joins vars from the denominator. *)
            (* c1*v1/d1 = n2_val, so c1 = n2_val*d1/v1 *)
            if late then
              Some (extras ~keep_constr1 ~v1 ~n1_val:(Utils.safe_force c1) ~n2_val:(n2_val * d1) ())
            else if not keep_constr1 then
              Some (lazy_extras ~keep_constr1 ~num_var:v1 ~coeff:c1 ~denom:(n2_val * d1) ())
            else None
        | Num_elems n1_val, Strided_var { coeff = c2; var = v2; denom = d2 } ->
            if late then
              Some (extras ~keep_constr1 ~v2 ~n1_val:(n1_val * d2) ~n2_val:(Utils.safe_force c2) ())
            else if keep_constr1 then
              Some (lazy_extras ~keep_constr1 ~num_var:v2 ~coeff:c2 ~denom:(n1_val * d2) ())
            else None
        | ( Strided_var { coeff = c1; var = v1; denom = d1 },
            Strided_var { coeff = c2; var = v2; denom = d2 } ) ->
            if late then
              Some
                (extras ~keep_constr1 ~v1 ~v2
                   ~n1_val:(Utils.safe_force c1 * d2)
                   ~n2_val:(Utils.safe_force c2 * d1)
                   ())
            else None
      in
      if List.is_empty vars2_only then
        Option.map ~f:(fun x -> (x, constr2)) (extras ~keep_constr1:false)
      else if List.is_empty vars1_only then
        Option.map ~f:(fun x -> (x, constr1)) (extras ~keep_constr1:true)
      else None
  | Exact dims1, Exact dims2 ->
      if List.length dims1 <> List.length dims2 then
        raise
        @@ Shape_error
             ( "Exact row constraint length mismatch",
               [
                 Row_mismatch
                   [
                     { beg_dims = []; dims = dims1; bcast = Broadcastable; prov };
                     { beg_dims = []; dims = dims2; bcast = Broadcastable; prov };
                   ];
               ] )
      else
        let eqs = List.map2_exn dims1 dims2 ~f:(fun d1 d2 -> Dim_eq { d1; d2; origin }) in
        Some (eqs, constr1)
  | Total_elems { numerator; divided_by }, Exact dims
  | Exact dims, Total_elems { numerator; divided_by } -> (
      match collect_factors dims with
      | None -> None (* Give up on complex cases *)
      | Some (known_product, vars) -> (
          match numerator with
          | Num_elems n ->
              if n <= 0 then
                raise @@ Shape_error ([%string "Invalid Total_elems numerator: %{n#Int}"], [])
              else if known_product = 0 then
                raise @@ Shape_error ("Exact constraint has zero dimension", [])
              else if n % known_product <> 0 then
                raise
                @@ Shape_error
                     ( [%string
                         "Total_elems numerator %{n#Int} not divisible by Exact dimensions product \
                          %{known_product#Int}"],
                       [] )
              else
                let reminder = n / known_product in
                if reminder = 1 then
                  (* reminder is 1: equate all variables on both sides to 1 *)
                  let divided_by_eqs =
                    List.map divided_by ~f:(fun v ->
                        Dim_eq { d1 = Var v; d2 = get_bcast_dim ~d:1 ~proj_id:43 (); origin })
                  in
                  let exact_vars_eqs =
                    List.map vars ~f:(fun v ->
                        Dim_eq { d1 = Var v; d2 = get_bcast_dim ~d:1 ~proj_id:44 (); origin })
                  in
                  Some (divided_by_eqs @ exact_vars_eqs, Exact dims)
                else if List.is_empty divided_by && List.length vars = 1 && reminder > 0 then
                  (* divided_by is empty and there is only one dim variable in Exact dims *)
                  let v = List.hd_exn vars in
                  Some ([ Dim_eq { d1 = Var v; d2 = get_default_dim ~d:reminder (); origin } ], Exact dims)
                else if List.is_empty vars && List.length divided_by = 1 && reminder > 0 then
                  (* Exact dims contain only known dimensions and divided_by has exactly one
                     variable *)
                  let v = List.hd_exn divided_by in
                  Some ([ Dim_eq { d1 = Var v; d2 = get_default_dim ~d:reminder (); origin } ], Exact dims)
                else None
          | Strided_var { coeff; var; denom } ->
              if known_product = 0 then
                raise @@ Shape_error ("Exact constraint has zero dimension", [])
              else if late && List.is_empty vars && List.is_empty divided_by then
                (* Exact dims contain only known dimensions and divided_by is empty *)
                (* coeff * var / denom = known_product, so var = known_product * denom / coeff *)
                let coeff_val = Utils.safe_force coeff in
                if known_product * denom % coeff_val = 0 then
                  let d = known_product * denom / coeff_val in
                  Some ([ Dim_eq { d1 = Var var; d2 = get_default_dim ~d (); origin } ], Exact dims)
                else elems_mismatch numerator (Num_elems known_product)
              else if
                late
                && List.mem vars var ~equal:equal_dim_var
                && List.is_empty divided_by
                && List.length vars = 1
              then
                (* Simple case: (coeff * var / denom) = known_product * var, trivially satisfied if
                   coeff = denom *)
                if Utils.safe_force coeff = denom * known_product then Some ([], Exact dims)
                else
                  raise
                  @@ Shape_error
                       ( [%string
                           "Total_elems vs. exact dims mismatch: %{Utils.safe_force coeff#Int} * \
                            %{var#Dim_var} / %{denom#Int} = %{known_product#Int} * %{var#Dim_var}"],
                         [ Dim_mismatch dims ] )
              else if late && List.is_empty divided_by && List.length vars = 1 then
                (* Handle case: length vars = 1 and it's not equal var from Strided_var (while
                   divided_by is empty), derive the coefficient between the two *)
                let single_var = List.hd_exn vars in
                let coeff_val = Utils.safe_force coeff in
                (* coeff * var / denom = known_product * single_var *)
                (* So: var = (known_product * denom / coeff) * single_var *)
                if known_product * denom % coeff_val = 0 then
                  let coefficient = known_product * denom / coeff_val in
                  if coefficient = 1 then
                    (* Simple equality *)
                    Some ([ Dim_eq { d1 = Var var; d2 = Var single_var; origin } ], Exact dims)
                  else if coefficient > 1 then
                    (* Use Affine with stride and no convolution *)
                    Some
                      ( [
                          Dim_eq
                            {
                              d1 = Var var;
                              d2 =
                                Affine
                                  {
                                    stride = coefficient;
                                    over = Var single_var;
                                    conv = None;
                                    stride_offset = 0;
                                  };
                              origin;
                            };
                        ],
                        Exact dims )
                  else
                    (* coefficient <= 0, which is invalid *)
                    elems_mismatch numerator (Num_elems known_product)
                else
                  (* Not divisible, mismatch *)
                  elems_mismatch numerator (Num_elems known_product)
              else None))

let%track5_sexp rec apply_dim_constraint ~(source : source) ~(stage : stage) (dim : dim)
    (constr : dim_constraint) env : constraint_ list * dim_constraint =
  let extras, constr =
    match (dim, constr) with
    | Dim { d; _ }, At_least_dim d_min ->
        if d < d_min then
          raise
          @@ Shape_error
               ( "At_least_dim constraint failed, expected " ^ Int.to_string d_min,
                 [ Dim_mismatch [ dim ] ] )
        else ([], constr)
    | Affine { stride; over; conv; stride_offset = _ }, At_least_dim d_min -> (
        if d_min <= 0 then ([], Unconstrained_dim)
        else
          let quotient = if d_min % stride = 0 then d_min / stride else (d_min / stride) + 1 in
          match conv with
          | Some { dilation; kernel = Dim { d = d_k; _ }; use_padding = false } ->
              let d_min = d_min - (dilation * d_k) in
              if d_min <= 0 then ([], Unconstrained_dim)
              else
                let quotient =
                  if d_min % stride = 0 then d_min / stride else (d_min / stride) + 1
                in
                apply_dim_constraint ~source ~stage over (At_least_dim quotient) env
          | _ -> apply_dim_constraint ~source ~stage over (At_least_dim quotient) env)
    | Var v, _ -> (
        match find_dim env.dim_env v with
        | None -> ([], constr)
        | Some (Solved_dim _) -> assert false
        | Some (Bounds_dim bounds) -> (
            match (source, constr) with
            (* If source is [Res], then [constr] (target) is [Opnd]. *)
            | Res, (Unconstrained_dim | At_least_dim 1) -> ([], constr)
            | _ -> Option.value ~default:([], constr) @@ dim_conjunction constr bounds.constr))
    | Concat _dims, At_least_dim _d_min ->
        (* FIXME: reconsider if we can make progress *)
        (* For concatenation, the constraint applies to the sum of component dimensions. We don't
           propagate constraints to components here; they'll be resolved during unification. *)
        ([], constr)
    | _, Unconstrained_dim -> ([], constr)
  in
  (extras, constr)

exception Given_up

(* Mark variables in Total_elems constraints to prevent premature guessing. When a Total_elems has
   Strided_var { var; _ } in numerator and divided_by list, we mark var with has_uniq_constr_unless
   = Some divided_by. This prevents var from being guessed to 1 unless at least one divided_by
   variable is also prevented. *)
let mark_total_elems_vars (constr : row_constraint) env : environment =
  match constr with
  | Total_elems { numerator = Strided_var { var; _ }; divided_by } -> (
      let unless_set = Set.of_list (module Dim_var) divided_by in
      match find_dim env.dim_env var with
      | Some (Bounds_dim bounds) ->
          let has_uniq_constr_unless =
            match bounds.has_uniq_constr_unless with
            | None -> Some unless_set
            | Some existing -> Some (Set.union existing unless_set)
          in
          let bounds = Bounds_dim { bounds with has_uniq_constr_unless } in
          { env with dim_env = add_dim env.dim_env ~key:var ~data:bounds }
      | None ->
          let bounds =
            Bounds_dim
              {
                is_in_param = false;
                has_uniq_constr_unless = Some unless_set;
                res = [];
                opnd = [];
                glb = None;
                constr = Unconstrained_dim;
                origin = [];
              }
          in
          { env with dim_env = add_dim env.dim_env ~key:var ~data:bounds }
      | Some (Solved_dim _) -> env)
  | _ -> env

let reduce_row_constraint (constr : row_constraint) ~(beg_dims : dim list) ~(dims : dim list) :
    row_constraint =
  match constr with
  | Unconstrained -> Unconstrained
  | Total_elems { numerator; divided_by } -> (
      try
        let d, vars =
          match collect_factors (beg_dims @ dims) with
          | Some (d, vars) -> (d, vars)
          | None -> raise Given_up
        in
        (* Check if any vars appear in divided_by (multiset intersection) *)
        let has_common_var =
          List.exists vars ~f:(fun v -> List.mem divided_by v ~equal:equal_dim_var)
        in
        if has_common_var then Unconstrained
        else
          let numerator = total_elems_divide numerator d in
          if total_elems_known_zero numerator then
            raise
            @@ Shape_error
                 ( "reduce_row_constraint: Total_elems constraint failed, shape is too big",
                   [ Dim_mismatch (beg_dims @ dims) ] )
          else if d = 1 && List.is_empty vars then constr
          else Total_elems { numerator; divided_by = divided_by @ vars }
      with Given_up -> Unconstrained)
  | Exact exact_dims ->
      let beg_len = List.length beg_dims in
      let dims_len = List.length dims in
      let exact_len = List.length exact_dims in
      if beg_len + dims_len > exact_len then
        raise
        @@ Shape_error
             ( "reduce_row_constraint: Exact constraint failed, shape is too long",
               [ Dim_mismatch (beg_dims @ dims) ] )
      else Exact (List.take (List.drop exact_dims beg_len) (exact_len - beg_len - dims_len))

(* Inverts what [reduce_row_constraint] would do. *)
let _lift_row_constraint (constr : row_constraint) ~(beg_dims : dim list) ~(dims : dim list) :
    row_constraint =
  match constr with
  | Total_elems { numerator; divided_by } -> (
      let ds, vars =
        List.partition_map (beg_dims @ dims) ~f:(function
          | Dim { d; _ } -> Either.First d
          | Var v -> Either.Second v
          | Affine _ -> failwith "NOT IMPLEMENTED YET"
          | Concat _ -> failwith "NOT IMPLEMENTED YET")
      in
      (* Check if all vars are in divided_by - for multiset, need to count occurrences *)
      let remove_all vars from_list =
        List.fold vars ~init:(Some from_list) ~f:(fun acc v ->
            match acc with
            | None -> None
            | Some lst ->
                if List.mem lst v ~equal:equal_dim_var then Some (remove_var v lst) else None)
      in
      match remove_all vars divided_by with
      | None -> Unconstrained
      | Some remaining ->
          let d = List.fold ds ~init:1 ~f:( * ) in
          if d = 1 && List.is_empty vars then constr
          else Total_elems { numerator = total_elems_multiply numerator d; divided_by = remaining })
  | Unconstrained -> Unconstrained
  | Exact exact_dims -> Exact (beg_dims @ exact_dims @ dims)

(** Helper function to convert a list of rows to either a single row or information about multiple
    row variables. Returns Either.First with a single row if there are zero or one row variables.
    Returns Either.Second with (all_dims, row_vars) if there are multiple row variables, where
    all_dims is a concatenation of all dims and beg_dims in proper order, and row_vars is a list of
    (row_var * provenance) pairs. *)
let rows_to_row_or_vars (rows : row list) : (row, dim list * (row_var * provenance) list) Either.t =
  let rec collect_info before_dims row_vars rows =
    match rows with
    | [] -> (List.rev before_dims, List.rev row_vars)
    | row :: remaining_rows -> (
        match row.bcast with
        | Broadcastable ->
            (* Regular row: include both flanks. *)
            let new_before_dims =
              List.rev_append row.dims (List.rev_append row.beg_dims before_dims)
            in
            collect_info new_before_dims row_vars remaining_rows
        | Row_var v ->
            (* Row variable - collect it and continue *)
            let new_before_dims =
              List.rev_append row.dims (List.rev_append row.beg_dims before_dims)
            in
            let new_row_vars = (v, row.prov) :: row_vars in
            collect_info new_before_dims new_row_vars remaining_rows)
  in
  let all_dims, row_vars = collect_info [] [] rows in
  match row_vars with
  | [] ->
      (* No row variables found *)
      let first_prov =
        match
          List.find_map rows ~f:(function
            | { prov; _ }
              when List.exists prov ~f:(fun (origin : provenance_origin) ->
                       equal_kind origin.kind `Output) ->
                Some prov
            | _ -> None)
        with
        | None -> []
        | Some prov -> prov
      in
      Either.First { beg_dims = []; dims = all_dims; bcast = Broadcastable; prov = first_prov }
  | [ (v, prov) ] ->
      (* Exactly one row variable - reconstruct the proper row structure *)
      let rec reconstruct_single_var before_dims rows =
        match rows with
        | [] -> failwith "rows_to_row_or_vars: single row variable not found during reconstruction"
        | row :: remaining_rows -> (
            match row.bcast with
            | Broadcastable ->
                let acc =
                  List.rev_append row.dims (List.rev_append row.beg_dims before_dims)
                in
                reconstruct_single_var acc remaining_rows
            | Row_var found_v when equal_row_var found_v v ->
                let new_beg_dims = List.rev_append before_dims row.beg_dims in
                let after_dims =
                  List.concat_map remaining_rows ~f:(fun r -> r.beg_dims @ r.dims)
                in
                let new_dims = row.dims @ after_dims in
                { beg_dims = new_beg_dims; dims = new_dims; bcast = Row_var v; prov }
            | Row_var _ ->
                let acc =
                  List.rev_append row.dims (List.rev_append row.beg_dims before_dims)
                in
                reconstruct_single_var acc remaining_rows)
      in
      Either.First (reconstruct_single_var [] rows)
  | _ ->
      (* Multiple row variables *)
      Either.Second (all_dims, row_vars)

let row_of_var v prov = { beg_dims = []; dims = []; bcast = Row_var v; prov }

let unsolved_constraints env =
  let dims = Utils.Tree_map.to_alist env.dim_env in
  let rows = Utils.Tree_map.to_alist env.row_env in
  let dims =
    List.filter_map dims ~f:(fun (var, entry) ->
        match entry with
        | Solved_dim _ -> None
        | Bounds_dim { constr = Unconstrained_dim; _ } -> None
        | Bounds_dim { constr; origin; _ } -> Some (Dim_constr { d = Var var; constr; origin }))
  in
  let rows =
    List.filter_map rows ~f:(fun (var, entry) ->
        match entry with
        | Solved_row _ -> None
        | Bounds_row { constr = Unconstrained; _ } -> None
        | Bounds_row { constr; origin; _ } ->
            Some (Rows_constr { r = [ row_of_var var [] ]; constr; origin }))
  in
  dims @ rows

let check_empty_row ~origin r =
  if not (List.is_empty r.dims) then
    raise @@ Shape_error ("check_empty_row: row is not empty", [ Row_mismatch [ r ] ]);
  match r.bcast with
  | Broadcastable ->
      if List.is_empty r.beg_dims then []
      else
        raise
        @@ Shape_error
             ("check_empty_row: row is not empty (beg_dims)", [ Row_mismatch [ r ] ])
  | Row_var v ->
      if List.is_empty r.beg_dims then
        [
          Row_eq
            {
              r1 = row_of_var v r.prov;
              r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov = r.prov };
              origin;
            };
        ]
      else raise @@ Shape_error ("check_empty_row: row is not empty", [ Row_mismatch [ r ] ])

let s_dim_one_in_entry v ~value (in_ : dim_entry) : _ * dim_entry =
  let from_ = [%sexp_of: dim_var * dim_entry] (v, drop_origin in_) in
  match in_ with
  | Solved_dim in_ -> ([], Solved_dim (s_dim_one v ~value ~in_))
  | Bounds_dim { is_in_param; has_uniq_constr_unless; res; opnd; glb; constr; origin } ->
      let find_v side = List.partition_tf side ~f:(equal_dim_var v) in
      let res_v, res = find_v res in
      let opnd_v, opnd = find_v opnd in
      let ineqs0 =
        match (opnd_v, glb) with
        | _ :: _, Some glb -> [ Dim_ineq { res = glb; opnd = value; from_; origin } ]
        | _ -> []
      in
      let ineqs1 =
        if List.is_empty opnd_v then []
        else List.map res ~f:(fun res -> Dim_ineq { res = Var res; opnd = value; from_; origin })
      in
      let ineqs2 =
        if List.is_empty res_v then []
        else List.map opnd ~f:(fun opnd -> Dim_ineq { res = value; opnd = Var opnd; from_; origin })
      in
      ( ineqs0 @ ineqs1 @ ineqs2,
        Bounds_dim
          {
            is_in_param;
            has_uniq_constr_unless;
            res;
            opnd;
            glb = Option.map glb ~f:(fun in_ -> s_dim_one v ~value ~in_);
            constr;
            origin;
          } )

let s_dim_one_in_row v ~value in_ =
  {
    in_ with
    beg_dims = List.map in_.beg_dims ~f:(fun in_ -> s_dim_one v ~value ~in_);
    dims = List.map in_.dims ~f:(fun in_ -> s_dim_one v ~value ~in_);
  }

let reapply_rows_constr = ref false

let subst_row_constraint_impl ~subst_in_dim ~get_dim_val stage constr =
  let subst_total_elems_divided_by numerator divided_by =
    let substituted_divided_by = List.map divided_by ~f:(fun v -> subst_in_dim (Var v)) in
    match collect_factors substituted_divided_by with
    | Some (known_product, residual_vars) ->
        reapply_rows_constr := true;
        Total_elems
          { numerator = total_elems_divide numerator known_product; divided_by = residual_vars }
    | None ->
        (* Fall back to preserving the original constraint *)
        Total_elems { numerator; divided_by }
  in
  match constr with
  | Total_elems { numerator = Strided_var { coeff; var; denom }; divided_by }
    when is_stage2_up stage && Option.is_some (get_dim_val var) ->
      let dim = Option.value_exn (get_dim_val var) in
      let tot = Utils.safe_force coeff * dim in
      reapply_rows_constr := true;
      if tot % denom = 0 then subst_total_elems_divided_by (Num_elems (tot / denom)) divided_by
      else
        raise
        @@ Shape_error
             ( [%string
                 "Total_elems constraint: shape cannot be strided, %{tot#Int} not divisible by \
                  %{denom#Int}"],
               [ Rows_constr_failed { constr } ] )
  | Total_elems { numerator = Strided_var { coeff; var; denom }; divided_by }
    when not (equal_dim (Var var) (subst_in_dim (Var var))) -> (
      reapply_rows_constr := true;
      match subst_in_dim (Var var) with
      | Dim { d; _ } as value when is_stage2_up stage ->
          (* Stage 2+: Replace (coeff * v / denom) with (coeff * d / denom) *)
          let new_num = Utils.safe_force coeff * d in
          if new_num % denom = 0 then
            Total_elems { numerator = Num_elems (new_num / denom); divided_by }
          else
            raise
            @@ Shape_error
                 ( "s_dim_one_in_row_constr: Total_elems constraint failed: dimension is not \
                    divisible",
                   [ Dim_mismatch [ value ] ] )
      | Dim _ ->
          (* Stage 1: Don't force coeff yet, keep the constraint as-is *)
          Total_elems { numerator = Strided_var { coeff; var; denom }; divided_by }
      | Var v' -> Total_elems { numerator = Strided_var { coeff; var = v'; denom }; divided_by }
      | Affine _ ->
          (* FIXME: NOT IMPLEMENTED YET *)
          failwith "NOT IMPLEMENTED YET"
      | Concat _ ->
          (* FIXME: NOT IMPLEMENTED YET *)
          failwith "NOT IMPLEMENTED YET")
  | Total_elems { numerator; divided_by } -> subst_total_elems_divided_by numerator divided_by
  | Exact dims ->
      (* The constraint update does not affect its applicability, so we don't need to reapply it. *)
      Exact (List.map dims ~f:subst_in_dim)
  | Unconstrained -> constr

let s_dim_one_in_row_constr stage v ~value constr =
  let get_dim_val v' =
    if equal_dim_var v v' then match value with Dim { d; _ } -> Some d | _ -> None else None
  in
  subst_row_constraint_impl
    ~subst_in_dim:(fun in_ -> s_dim_one ~keep_affine:true v ~value ~in_)
    ~get_dim_val stage constr

let ineqs_from_reapply_rows_constr = ref []

let s_dim_one_in_row_entry stage v ~value ~key ~data =
  assert (not !reapply_rows_constr);
  let result =
    match data with
    | Solved_row in_ -> Solved_row (s_dim_one_in_row v ~value in_)
    | Bounds_row { is_in_param; res; opnd; glb; constr; origin } ->
        let constr = s_dim_one_in_row_constr stage v ~value constr in
        if !reapply_rows_constr then
          ineqs_from_reapply_rows_constr :=
            Rows_constr { r = [ row_of_var key [] ]; constr; origin }
            :: !ineqs_from_reapply_rows_constr;
        reapply_rows_constr := false;
        let glb = Option.map glb ~f:(s_dim_one_in_row v ~value) in
        Bounds_row { is_in_param; res; opnd; glb; constr; origin }
  in
  result

let rec vars_of_dim = function
  | Dim _ -> Set.empty (module Dim_var)
  | Var v -> Set.singleton (module Dim_var) v
  | Affine { over; conv = None; _ } -> vars_of_dim over
  | Affine { over; conv = Some { kernel; _ }; _ } ->
      Set.union (vars_of_dim over) (vars_of_dim kernel)
  | Concat dims -> Set.union_list (module Dim_var) (List.map dims ~f:vars_of_dim)

let subst_dim ?(keep_affine = false) env dim =
  let vars = vars_of_dim dim in
  List.fold (Set.elements vars) ~init:dim ~f:(fun acc v ->
      match find_dim env.dim_env v with
      | Some (Solved_dim d) -> s_dim_one ~keep_affine v ~value:d ~in_:acc
      | _ -> acc)

(** Substitute [value] for row variable [v] inside [in_], uniformly composing both flanks.

    If [in_]'s middle row variable matches [v], the result is

    [{ beg_dims = in_.beg_dims @ value.beg_dims; dims = value.dims @ in_.dims; bcast = value.bcast }]

    which preserves [in_]'s pinned leading flank whether [value] is closed or open. *)
let s_row_one v ~value:{ beg_dims = value_beg; dims = more_dims; bcast; prov = _ } ~in_ =
  match in_ with
  | { beg_dims; dims; bcast = Row_var v2; prov } when equal_row_var v v2 ->
      { beg_dims = beg_dims @ value_beg; dims = more_dims @ dims; bcast; prov }
  | _ -> in_

let s_row_one_in_row_constr _v ~value:_ ~in_ =
  match in_ with Unconstrained | Total_elems _ | Exact _ -> in_

let s_row_one_in_entry (v : row_var) ~(value : row) ~(in_ : row_entry) :
    constraint_ list * row_entry =
  match in_ with
  | Solved_row in_ -> ([], Solved_row (s_row_one v ~value ~in_))
  | Bounds_row { is_in_param; res; opnd; glb; constr; origin } ->
      (* TODO: audit code to ensure we don't lose the constraints associated with the bounds
         variables. *)
      let find_v side = List.partition_tf side ~f:(equal_row_var v) in
      let res_v, res = find_v res in
      let opnd_v, opnd = find_v opnd in
      let ineqs0 =
        match (opnd_v, glb) with
        | _ :: _, Some glb -> [ Row_ineq { res = glb; opnd = value; origin } ]
        | _ -> []
      in
      let ineqs1 =
        if List.is_empty opnd_v then []
        else
          List.map res ~f:(fun res ->
              Row_ineq { res = row_of_var res value.prov; opnd = value; origin })
      in
      let ineqs2 =
        if List.is_empty res_v then []
        else
          List.map opnd ~f:(fun opnd ->
              Row_ineq { opnd = row_of_var opnd value.prov; res = value; origin })
      in
      let constr = s_row_one_in_row_constr v ~value ~in_:constr in
      let glb = Option.map glb ~f:(fun in_ -> s_row_one v ~value ~in_) in
      (ineqs0 @ ineqs1 @ ineqs2, Bounds_row { is_in_param; res; opnd; glb; constr; origin })

let subst_row env ({ beg_dims; dims; bcast; prov } : t) : t =
  let s_dims = List.map ~f:(subst_dim env) in
  let beg_dims = s_dims beg_dims in
  let dims = s_dims dims in
  let default = { beg_dims; dims; bcast; prov } in
  match bcast with
  | Broadcastable -> default
  | Row_var v -> (
      match find_row env.row_env v with
      | None | Some (Bounds_row _) -> default
      | Some (Solved_row { beg_dims = []; dims = []; bcast = Row_var v2; _ })
        when equal_row_var v v2 ->
          default
      | Some (Solved_row ({ bcast = Row_var v2; _ } as r2)) when equal_row_var v v2 ->
          raise
          @@ Shape_error
               ("Infinite number of axes by self-reference", [ Row_mismatch [ default; r2 ] ])
      | Some (Solved_row { beg_dims = more_beg; dims = more_dims; bcast; prov = _ }) -> (
          (* Note: we assume env is idempotent (solved wrt. equalities). *)
          let more_beg = s_dims more_beg in
          let more_dims = s_dims more_dims in
          match bcast with
          | Broadcastable ->
              { beg_dims = beg_dims @ more_beg; dims = more_dims @ dims; bcast; prov }
          | Row_var _ ->
              {
                beg_dims = beg_dims @ more_beg;
                dims = more_dims @ dims;
                bcast;
                prov;
              }))

let subst_row_constraint stage env constr =
  subst_row_constraint_impl ~subst_in_dim:(subst_dim env) ~get_dim_val:(get_dim_val env) stage
    constr

let%track5_sexp rec apply_rows_constraint ~depth ~stage origin (rows : row list)
    (constr : row_constraint) env : constraint_ list * _ =
  if depth > 16 then ([], env)
  else
    (* Mark variables in Total_elems to prevent premature guessing *)
    let env = mark_total_elems_vars constr env in
    match rows_to_row_or_vars rows with
    | Either.First single_row -> apply_row_constraint ~depth stage origin single_row constr env
    | Either.Second (all_dims, row_vars) -> (
        match constr with
        | Exact dims when List.length dims < List.length all_dims ->
            (* Case 1: Exact dims has fewer axes than all_dims - raise mismatch *)
            raise
            @@ Shape_error
                 ("apply_rows_constraint: Exact constraint has too few axes", [ Row_mismatch rows ])
        | Exact dims when List.length dims = List.length all_dims ->
            (* Case 2: Exact dims has same length as all_dims - derive pairwise equations *)
            let dim_eqs = List.map2_exn dims all_dims ~f:(fun d1 d2 -> Dim_eq { d1; d2; origin }) in
            let row_eqs =
              List.map row_vars ~f:(fun (v, prov) ->
                  Row_eq
                    {
                      r1 = row_of_var v prov;
                      r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov };
                      origin;
                    })
            in
            (dim_eqs @ row_eqs, env)
        | Total_elems { numerator = Num_elems n; divided_by } -> (
            (* Case 3: Total_elems with known numerator *)
            match collect_factors all_dims with
            | None ->
                ([ Rows_constr { r = rows; constr; origin } ], env) (* Give up on complex cases *)
            | Some (known_product, product_vars) ->
                (* Move divided_by variables to the other side by combining with product_vars *)
                let all_product_vars = product_vars @ divided_by in
                if n % known_product <> 0 then
                  raise
                  @@ Shape_error
                       ( [%string
                           "Total_elems constraint: %{n#Int} not divisible by known product \
                            %{known_product#Int}"],
                         [] )
                else if n = known_product then
                  (* Equate all product vars to d=1 and add Total_elems 1 for each row var *)
                  let var_eqs =
                    List.map all_product_vars ~f:(fun v ->
                        Dim_eq { d1 = Var v; d2 = get_bcast_dim ~d:1 ~proj_id:45 (); origin })
                  in
                  let row_constrs =
                    List.map row_vars ~f:(fun (v, id) ->
                        Rows_constr
                          {
                            r = [ row_of_var v id ];
                            constr = Total_elems { numerator = Num_elems 1; divided_by = [] };
                            origin;
                          })
                  in
                  (var_eqs @ row_constrs, env)
                else
                  (* Cannot deduce no_further_axes, return unchanged *)
                  ([ Rows_constr { r = rows; constr; origin } ], env))
        | Exact [ single_dim ] -> (
            (* Handle exact single dimension constraint, preferring non-empty output rows. *)
            match List.rev rows with
            | { beg_dims = []; dims = []; bcast = Broadcastable; prov = _ } :: more_rows ->
                apply_rows_constraint ~depth:(depth + 1) ~stage origin (List.rev more_rows) constr
                  env
            | { beg_dims = []; dims = []; bcast = Row_var v; prov } :: more_rows
              when List.exists prov ~f:(fun (origin : provenance_origin) ->
                       equal_kind origin.kind `Input) ->
                let more_eqs, env =
                  apply_rows_constraint ~depth:(depth + 1) ~stage origin (List.rev more_rows) constr
                    env
                in
                ( Row_eq
                    {
                      r1 = row_of_var v prov;
                      r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov };
                      origin;
                    }
                  :: more_eqs,
                  env )
            | { beg_dims = []; dims = []; bcast = Row_var v; prov } :: more_rows
              when List.exists prov ~f:(fun (origin : provenance_origin) ->
                       equal_kind origin.kind `Output) ->
                ( Row_eq
                    {
                      r1 = row_of_var v prov;
                      r2 =
                        { beg_dims = []; dims = [ single_dim ]; bcast = Broadcastable; prov };
                      origin;
                    }
                  :: List.concat_map ~f:(check_empty_row ~origin) more_rows,
                  env )
            | { dims = _; bcast = Row_var _; prov; _ } :: _
              when List.exists prov ~f:(fun (origin : provenance_origin) ->
                       equal_kind origin.kind `Output) ->
                raise
                @@ Shape_error
                     ( "apply_rows_constraint: shape too big (non-empty output leading flank)",
                       [ Row_mismatch rows ] )
            | _ ->
                raise @@ Shape_error ("apply_rows_constraint: shape too big", [ Row_mismatch rows ])
            )
        | _ -> ([ Rows_constr { r = rows; constr; origin } ], env))

and apply_row_constraint ~depth stage origin (r : row) (constr : row_constraint) env :
    constraint_ list * _ =
  if depth > 16 then ([], env)
  else if is_unconstrained constr then ([], env)
  else
    (* Mark variables in Total_elems to prevent premature guessing *)
    let env = mark_total_elems_vars constr env in
    let constr = subst_row_constraint stage env constr in
    reapply_rows_constr := false;
    let reduce constr ~beg_dims ~dims =
      try reduce_row_constraint constr ~beg_dims ~dims
      with Shape_error (s, trace) -> raise @@ Shape_error (s, Row_mismatch [ r ] :: trace)
    in
    let extras, constr, env, stored, updated =
      match r with
      | { bcast = Broadcastable; _ } -> ([], constr, env, false, false)
      | { bcast = Row_var v; beg_dims; dims; _ } -> (
          match find_row env.row_env v with
          | Some (Solved_row _) -> ([], constr, env, false, false)
          | None ->
              ( [],
                constr,
                {
                  env with
                  row_env =
                    (let constr = reduce constr ~beg_dims ~dims in
                     add_row env.row_env ~key:v
                       ~data:
                         (Bounds_row
                            { is_in_param = false; constr; res = []; opnd = []; glb = None; origin }));
                },
                true,
                false )
          | Some (Bounds_row ({ constr = Unconstrained; _ } as bounds)) ->
              ( [],
                constr,
                {
                  env with
                  row_env =
                    add_row env.row_env ~key:v
                      ~data:(Bounds_row { bounds with constr = reduce constr ~beg_dims ~dims });
                },
                true,
                false )
          | Some (Bounds_row bounds) -> (
              let origin = merge_origins origin bounds.origin in
              match
                row_conjunction ~prov:r.prov ~origin stage (reduce constr ~beg_dims ~dims)
                  bounds.constr
              with
              | None -> ([], constr, env, false, false)
              | Some (extras, constr) ->
                  if phys_equal constr bounds.constr then (extras, constr, env, true, false)
                  else
                    ( extras,
                      constr,
                      {
                        env with
                        row_env =
                          add_row env.row_env ~key:v ~data:(Bounds_row { bounds with constr });
                      },
                      true,
                      true )))
    in
    match (r, constr) with
    | _ when stored && not updated -> (extras, env)
    | _, Unconstrained -> assert false
    | _, Total_elems { numerator = Strided_var { coeff; var; denom }; divided_by = [] }
      when is_stage2_up stage && Option.is_some (get_dim_val env var) ->
        let tot = Option.value_exn (get_dim_val env var) in
        let tot = Utils.safe_force coeff * tot / denom in
        apply_row_constraint ~depth:(depth + 1) stage origin r
          (Total_elems { numerator = Num_elems tot; divided_by = [] })
          env
    | ( { dims; bcast = Broadcastable; _ },
        Total_elems { numerator = Strided_var { coeff; var; denom }; divided_by = [] } )
      when is_stage2_up stage && known_dims_product dims ->
        let (d : int), _ = Option.value_exn (collect_factors dims) in
        let coeff : int = Utils.safe_force coeff in
        if denom * d % coeff = 0 then
          (Dim_eq { d1 = Var var; d2 = get_default_dim ~d:(denom * d / coeff) (); origin } :: extras, env)
        else
          raise
          @@ Shape_error
               ( [%string
                   "apply_row_constraint: Total_elems constraint failed: %{denom*d#Int} not \
                    divisible by %{coeff#Int}"],
                 [ Row_mismatch [ r ] ] )
    | { dims; bcast = Broadcastable; _ }, Total_elems { numerator; divided_by }
      when List.length divided_by <= 1 -> (
        try
          let d, vars =
            match collect_factors dims with Some (d, vars) -> (d, vars) | None -> raise Given_up
          in
          let numerator = total_elems_divide numerator d in
          if total_elems_known_zero numerator then
            raise
            @@ Shape_error
                 ( "apply_row_constraint: Total_elems constraint failed, shape is too big",
                   [ Dim_mismatch dims ] );
          match (numerator, vars, divided_by) with
          | Num_elems 1, [], [] -> (extras, env)
          | Num_elems _, [], [] ->
              raise
              @@ Shape_error
                   ( "apply_row_constraint: Total_elems constraint failed, shape is too small",
                     [ Row_mismatch [ r ] ] )
          | Num_elems n, [ v ], [] | Num_elems n, [], [ v ] ->
              (Dim_eq { d1 = Var v; d2 = get_default_dim ~d:n (); origin } :: extras, env)
          | Num_elems 1, vs1, vs2 ->
              ( List.map
                  ~f:(fun v -> Dim_eq { d1 = Var v; d2 = get_bcast_dim ~d:1 ~proj_id:46 (); origin })
                  (vs1 @ vs2)
                @ extras,
                env )
          | Strided_var { coeff; var; denom }, [], [ v ]
            when equal_dim_var var v && (Utils.is_safe_val coeff || is_stage2_up stage) ->
              (* Total = (coeff * v / denom) / v = coeff / denom *)
              if Utils.safe_force coeff % denom = 0 then
                ( Dim_eq { d1 = Var v; d2 = get_default_dim ~d:(Utils.safe_force coeff / denom) (); origin }
                  :: extras,
                  env )
              else if
                (* coeff not divisible by denom - keep as constraint *)
                stored
              then (extras, env)
              else (Rows_constr { r = [ r ]; constr; origin } :: extras, env)
          | _ ->
              if stored then (extras, env)
              else
                ( Rows_constr { r = [ r ]; constr; origin } :: extras,
                  env (* Wait for more shape inference. *) )
        with Given_up ->
          if stored then (extras, env)
          else
            ( Rows_constr { r = [ r ]; constr; origin } :: extras,
              env (* Wait for more shape inference. *) ))
    | { beg_dims; dims; bcast = Row_var v; _ }, Exact exact_dims
      when List.length beg_dims + List.length dims >= List.length exact_dims ->
        assert (not stored);
        if List.length dims + List.length beg_dims > List.length exact_dims then
          raise
          @@ Shape_error
               ( "apply_row_constraint: Exact constraint failed, shape is too long",
                 [ Row_mismatch [ r ] ] );
        ( Row_eq
            {
              r1 = row_of_var v r.prov;
              r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov = r.prov };
              origin;
            }
          :: List.map2_exn exact_dims (beg_dims @ dims) ~f:(fun d1 d2 -> Dim_eq { d1; d2; origin })
          @ extras,
          env )
    | ( { bcast = Row_var v; _ },
        Total_elems { numerator = Strided_var { coeff; var = _; denom }; divided_by = [] } )
      when is_stage2_up stage -> (
        (* Check if we have a GLB and if it meets our conditions *)
        match find_row env.row_env v with
        | Some (Bounds_row { glb = Some ({ dims = glb_dims; bcast = Broadcastable; _ } as glb); _ })
          when Utils.is_safe_val coeff && Utils.safe_force coeff > denom -> (
            (* Check if all GLB dimensions are known *)
            match collect_factors glb_dims with
            | Some (_known_product, []) ->
                (* Check if GLB has at most one dimension greater than 1 *)
                let greater_than_one =
                  List.filter glb_dims ~f:(function Dim { d; _ } -> d > 1 | _ -> false)
                in
                if List.length greater_than_one <= 1 then
                  (Row_eq { r1 = row_of_var v r.prov; r2 = glb; origin } :: extras, env)
                else if stored then (extras, env)
                else (Rows_constr { r = [ r ]; constr; origin } :: extras, env)
            | _ ->
                if stored then (extras, env)
                else (Rows_constr { r = [ r ]; constr; origin } :: extras, env))
        | _ ->
            if stored then (extras, env)
            else (Rows_constr { r = [ r ]; constr; origin } :: extras, env))
    | { bcast = Row_var _; _ }, _ | _, Total_elems { numerator = _; divided_by = _ } ->
        if stored then (extras, env)
        else
          ( Rows_constr { r = [ r ]; constr; origin } :: extras,
            env (* Wait for more shape inference. *) )
    | { beg_dims; dims; bcast = Broadcastable; _ }, Exact exact_dims ->
        assert (not stored);
        ( List.map2_exn exact_dims (beg_dims @ dims) ~f:(fun d1 d2 ->
              Dim_eq { d1; d2; origin })
          @ extras,
          env )

let rec dim_var_occurs_in_dim (v : dim_var) (d : dim) : bool =
  match d with
  | Var v' -> equal_dim_var v v'
  | Dim _ -> false
  | Affine { over; conv = None; _ } -> dim_var_occurs_in_dim v over
  | Affine { over; conv = Some { kernel; _ }; _ } ->
      dim_var_occurs_in_dim v over || dim_var_occurs_in_dim v kernel
  | Concat dims -> List.exists dims ~f:(dim_var_occurs_in_dim v)

(* --- Arithmetic normalization for [Concat] strict equality (consumed by [unify_dim] below). A
   concatenated axis denotes the SUM of its components, so two concats are equated arithmetically
   (cancel common components, reduce solved [Dim]s, bind/defer the residual) rather than by
   equal-arity structural matching. --- *)

(* Multiset cancellation of structurally-equal components from both sides. *)
let cancel_common_dims (xs : dim list) (ys : dim list) : dim list * dim list =
  let kept_rev, ys =
    List.fold xs ~init:([], ys) ~f:(fun (kept, ys) x ->
        match List.findi ys ~f:(fun _ y -> equal_dim x y) with
        | Some (i, _) -> (kept, List.filteri ys ~f:(fun j _ -> j <> i))
        | None -> (x :: kept, ys))
  in
  (List.rev kept_rev, ys)

(* Split components into solved [Dim]s and the rest (vars / affine / nested concat). *)
let partition_solved_dims (dims : dim list) : solved_dim list * dim list =
  List.partition_map dims ~f:(function Dim s -> Either.First s | d -> Either.Second d)

(* Reduce the basis tags of solved components: reserved (claim-free) tags ([default], [bcast_if_1])
   drop out; all named tags must agree (and the residual takes that named tag), otherwise [default].
   [None] flags a genuine named-tag conflict. *)
let reduce_solved_basis (solved : solved_dim list) : string option =
  match
    List.filter_map solved ~f:(fun s -> if is_reserved_basis s.basis then None else Some s.basis)
    |> List.dedup_and_sort ~compare:String.compare
  with
  | [] -> Some default_basis
  | [ b ] -> Some b
  | _ -> None

let%track6_sexp rec unify_dim ~stage origin (eq : dim * dim) env : constraint_ list * _ =
  let dim1 : dim = subst_dim env @@ fst eq and dim2 : dim = subst_dim env @@ snd eq in
  match (dim1, dim2) with
  | Dim { basis = b1; _ }, Dim { basis = b2; _ } when not (String.equal b1 b2) ->
      raise
      @@ Shape_error
           ("solved dimensions for axis: different bases", [ Dim_mismatch [ dim1; dim2 ] ])
  | Dim { d = d1; basis = _; _ }, Dim { d = d2; basis = _; _ } when d1 = d2 ->
      (* Basis is total now: equality of two solved dims requires both equal size and equal tag
         (the preceding arm already rejected unequal tags). There is no unspecified ([None]) side
         to propagate a basis onto, so the old upgrade-var pass is gone — variable solving records
         the already-total [Dim] exactly. *)
      ([], env)
  | Var v1, Var v2 when equal_dim_var v1 v2 -> ([], env)
  | ( Affine { stride = 1; over; conv = Some { use_padding = true; _ } | None; stride_offset = _ },
      dim )
  | ( dim,
      Affine { stride = 1; over; conv = Some { use_padding = true; _ } | None; stride_offset = _ } )
    ->
      unify_dim ~stage origin (over, dim) env
  | ( Affine
        {
          stride = stride1;
          over = over1;
          conv = Some { use_padding = true; _ } | None;
          stride_offset = _;
        },
      Affine
        {
          stride = stride2;
          over = over2;
          conv = Some { use_padding = true; _ } | None;
          stride_offset = _;
        } )
    when stride1 % stride2 = 0 || stride2 % stride1 = 0 ->
      (* Both have use_padding=true, we can simplify by dividing strides *)
      unify_dim ~stage origin
        ( Affine
            {
              stride = (if stride1 > stride2 then stride1 / stride2 else stride2 / stride1);
              over = (if stride1 > stride2 then over1 else over2);
              conv = None;
              stride_offset = 0;
            },
          if stride1 > stride2 then over2 else over1 )
        env
  | Affine { stride; over = Dim s; conv = None | Some { use_padding = true; _ }; _ }, dim
  | dim, Affine { stride; over = Dim s; conv = None | Some { use_padding = true; _ }; _ } ->
      (* stride_offset doesn't contribute to shapes when conv = None or use_padding = true *)
      unify_dim ~stage origin (get_dim ~d:(stride * s.d) ~basis:s.basis (), dim) env
  | Affine { stride; over; conv = None | Some { use_padding = true; _ }; _ }, Dim s
  | Dim s, Affine { stride; over; conv = None | Some { use_padding = true; _ }; _ } ->
      (* stride_offset doesn't contribute to shapes when conv = None or use_padding = true *)
      if s.d >= 0 && s.d % stride = 0 then
        unify_dim ~stage origin (get_dim ~d:(s.d / stride) ~basis:s.basis (), over) env
      else
        raise
        @@ Shape_error
             ("solved dimensions for axis: incompatible stride", [ Dim_mismatch [ dim1; dim2 ] ])
  | ( Affine
        { stride; over = Dim s; conv = Some { dilation; kernel = Dim k; use_padding = false }; _ },
      dim )
  | ( dim,
      Affine
        { stride; over = Dim s; conv = Some { dilation; kernel = Dim k; use_padding = false }; _ } )
    ->
      (* input_size = stride * (output_size - 1) + effective_kernel_span *)
      let span = effective_kernel_span ~dilation ~kernel_size:k.d in
      let basis =
        merge_derived_basis s.basis k.basis ~on_conflict:(fun () ->
            raise
            @@ Shape_error
                 ( "convolution: conflicting dimension bases between stride and kernel",
                   [ Dim_mismatch [ Dim s; Dim k ] ] ))
      in
      unify_dim ~stage origin (get_dim ~d:((stride * (s.d - 1)) + span) ~basis (), dim) env
  | Affine { stride; over; conv = Some { dilation; kernel = Dim k; use_padding = false }; _ }, Dim s
  | Dim s, Affine { stride; over; conv = Some { dilation; kernel = Dim k; use_padding = false }; _ }
    ->
      (* Reverse: solve for output_size given input_size s. input_size = stride * (output_size - 1)
         + effective_kernel_span output_size = (input_size - effective_kernel_span) / stride + 1 *)
      let span : int = effective_kernel_span ~dilation ~kernel_size:k.d in
      let basis =
        merge_derived_basis s.basis k.basis ~on_conflict:(fun () ->
            raise
            @@ Shape_error
                 ( "convolution: conflicting dimension bases between stride and kernel",
                   [ Dim_mismatch [ Dim s; Dim k ] ] ))
      in
      let numerator : int = s.d - span in
      if numerator >= 0 && numerator % stride = 0 then
        unify_dim ~stage origin (get_dim ~d:((numerator / stride) + 1) ~basis (), over) env
      else
        raise
        @@ Shape_error
             ("solved dimensions for axis: incompatible stride", [ Dim_mismatch [ dim1; dim2 ] ])
  | ( Affine
        { stride; over = Dim s; conv = Some { dilation; kernel = Var v; use_padding = false }; _ },
      Dim i )
  | ( Dim i,
      Affine
        { stride; over = Dim s; conv = Some { dilation; kernel = Var v; use_padding = false }; _ } )
    -> (
      (* Infer kernel dimension from the relationship: input_size = stride * (output_size - 1) +
         effective_kernel_span effective_kernel_span = input_size - stride * (output_size - 1) *)
      let span = i.d - (stride * (s.d - 1)) in
      match kernel_size_of_span ~dilation ~span with
      | Some k -> unify_dim ~stage origin (Var v, get_default_dim ~d:k ()) env
      | None ->
          raise
          @@ Shape_error
               ( "solved dimensions for axis: cannot infer kernel dimension",
                 [ Dim_mismatch [ dim1; dim2 ] ] ))
  | ( Affine { stride = s1; over = o1; conv = None | Some { use_padding = true; _ }; _ },
      Affine { stride = s2; over = o2; conv = None | Some { use_padding = true; _ }; _ } ) ->
      (* stride_offset doesn't contribute to shapes when conv = None or use_padding = true *)
      if s1 = s2 then unify_dim ~stage origin (o1, o2) env
      else if s1 >= s2 && s1 % s2 = 0 then
        unify_dim ~stage origin
          (o2, Affine { stride = s1 / s2; over = o1; conv = None; stride_offset = 0 })
          env
      else if s2 >= s1 && s2 % s1 = 0 then
        unify_dim ~stage origin
          (o1, Affine { stride = s2 / s1; over = o2; conv = None; stride_offset = 0 })
          env
      else
        raise
        @@ Shape_error
             ("solved dimensions for axis: unresolvable strides", [ Dim_mismatch [ dim1; dim2 ] ])
  | ( Affine
        {
          stride = s1;
          over = o1;
          conv = Some { dilation = d1; kernel = Dim k1; use_padding = false };
          _;
        },
      Affine
        {
          stride = s2;
          over = o2;
          conv = Some { dilation = d2; kernel = Dim k2; use_padding = false };
          _;
        } ) ->
      (* Both sides use: input = stride * (output - 1) + span *)
      let span1 = effective_kernel_span ~dilation:d1 ~kernel_size:k1.d
      and span2 = effective_kernel_span ~dilation:d2 ~kernel_size:k2.d in
      if s1 = s2 && span1 = span2 then unify_dim ~stage origin (o1, o2) env
      else if s1 >= s2 && s1 % s2 = 0 then
        let stride = s1 / s2 in
        let new_span = ((span1 - span2) / s2) + 1 in
        (* Encode new_span via a helper conv with dilation=1 *)
        let helper_kernel_size =
          Option.value ~default:new_span (kernel_size_of_span ~dilation:1 ~span:new_span)
        in
        let helper_conv =
          Some
            {
              dilation = 1;
              kernel = Dim { d = helper_kernel_size; basis = default_basis; proj_id = None };
              use_padding = false;
            }
        in
        unify_dim ~stage origin
          (o2, Affine { stride; over = o1; conv = helper_conv; stride_offset = 0 })
          env
      else if s2 >= s1 && s2 % s1 = 0 then
        let stride = s2 / s1 in
        let new_span = ((span2 - span1) / s1) + 1 in
        (* Encode new_span via a helper conv with dilation=1 *)
        let helper_kernel_size =
          Option.value ~default:new_span (kernel_size_of_span ~dilation:1 ~span:new_span)
        in
        let helper_conv =
          Some
            {
              dilation = 1;
              kernel = Dim { d = helper_kernel_size; basis = default_basis; proj_id = None };
              use_padding = false;
            }
        in
        unify_dim ~stage origin
          (o1, Affine { stride; over = o2; conv = helper_conv; stride_offset = 0 })
          env
      else
        (* FIXME: should keep the constraint as-is but currently unification must make progress. *)
        raise
        @@ Shape_error
             ("solved dimensions for axis: unresolvable strides", [ Dim_mismatch [ dim1; dim2 ] ])
  | ( Affine { stride = s1; over = o1; conv = None | Some { use_padding = true; _ }; _ },
      Affine
        {
          stride = s2;
          over = o2;
          conv = Some { dilation = d2; kernel = Dim k2; use_padding = false };
          _;
        } )
  | ( Affine
        {
          stride = s2;
          over = o2;
          conv = Some { dilation = d2; kernel = Dim k2; use_padding = false };
          _;
        },
      Affine { stride = s1; over = o1; conv = None | Some { use_padding = true; _ }; _ } ) ->
      (* conv = None and use_padding = true effectively have span 1 (identity on dimension);
         use_padding = false has effective_kernel_span *)
      let span1 = 1 and span2 = effective_kernel_span ~dilation:d2 ~kernel_size:k2.d in
      if s1 = s2 && span1 = span2 then unify_dim ~stage origin (o1, o2) env
      else if s1 >= s2 && s1 % s2 = 0 then
        let stride = s1 / s2 in
        let new_span = ((span1 - span2) / s2) + 1 in
        let helper_kernel_size =
          Option.value ~default:new_span (kernel_size_of_span ~dilation:1 ~span:new_span)
        in
        let helper_conv =
          Some
            {
              dilation = 1;
              kernel = Dim { d = helper_kernel_size; basis = default_basis; proj_id = None };
              use_padding = false;
            }
        in
        unify_dim ~stage origin
          (o2, Affine { stride; over = o1; conv = helper_conv; stride_offset = 0 })
          env
      else if s2 >= s1 && s2 % s1 = 0 then
        let stride = s2 / s1 in
        let new_span = ((span2 - span1) / s1) + 1 in
        let helper_kernel_size =
          Option.value ~default:new_span (kernel_size_of_span ~dilation:1 ~span:new_span)
        in
        let helper_conv =
          Some
            {
              dilation = 1;
              kernel = Dim { d = helper_kernel_size; basis = default_basis; proj_id = None };
              use_padding = false;
            }
        in
        unify_dim ~stage origin
          (o1, Affine { stride; over = o2; conv = helper_conv; stride_offset = 0 })
          env
      else
        raise
        @@ Shape_error
             ("solved dimensions for axis: unresolvable strides", [ Dim_mismatch [ dim1; dim2 ] ])
  | Affine { conv = Some { kernel = Var _; use_padding = false; _ }; _ }, _
  | _, Affine { conv = Some { kernel = Var _; use_padding = false; _ }; _ } ->
      (* Can't compute offset until kernel dimension is resolved; defer *)
      ([ Dim_eq { d1 = dim1; d2 = dim2; origin } ], env)
  | Var v, dim2 | dim2, Var v ->
      if dim_var_occurs_in_dim v dim2 then
        raise
        @@ Shape_error
             ( "occurs check failed: dimension variable occurs in its own definition",
               [ Dim_mismatch [ Var v; dim2 ] ] );
      let ineqs = ref [] in
      let f in_ =
        let more_ineqs, result = s_dim_one_in_entry v ~value:dim2 in_ in
        ineqs := more_ineqs @ !ineqs;
        result
      in
      ineqs_from_reapply_rows_constr := [];
      let env =
        match find_dim env.dim_env v with
        | None ->
            let dim_env = Utils.Tree_map.map ~f env.dim_env in
            {
              dim_env = add_dim dim_env ~key:v ~data:(Solved_dim dim2);
              row_env =
                Utils.Tree_map.mapi env.row_env ~f:(s_dim_one_in_row_entry stage v ~value:dim2);
              discardable_vars = env.discardable_vars;
            }
        | Some (Solved_dim _) -> assert false
        | Some (Bounds_dim { is_in_param = _; res; opnd; glb; constr; origin = origin1; _ } as in_)
          ->
            let origin = merge_origins origin origin1 in
            let from_ = [%sexp_of: dim_var * dim_entry] (v, drop_origin in_) in
            let dim_env = Utils.Tree_map.map env.dim_env ~f in
            List.iter res ~f:(fun res ->
                ineqs := Dim_ineq { res = Var res; opnd = dim2; from_; origin } :: !ineqs);
            List.iter opnd ~f:(fun opnd ->
                ineqs := Dim_ineq { res = dim2; opnd = Var opnd; from_; origin } :: !ineqs);
            Option.iter glb ~f:(fun glb ->
                ineqs := Dim_ineq { res = glb; opnd = dim2; from_; origin } :: !ineqs);
            let extras, constr = apply_dim_constraint ~source:Equation ~stage dim2 constr env in
            let extras =
              if is_unconstrained_dim constr then extras
              else Dim_constr { d = dim2; constr; origin } :: extras
            in
            ineqs := extras @ !ineqs;
            {
              dim_env = add_dim dim_env ~key:v ~data:(Solved_dim dim2);
              row_env =
                Utils.Tree_map.mapi env.row_env ~f:(s_dim_one_in_row_entry stage v ~value:dim2);
              discardable_vars = env.discardable_vars;
            }
      in
      ineqs := !ineqs_from_reapply_rows_constr @ !ineqs;
      ineqs_from_reapply_rows_constr := [];
      let dim_eqs, ineqs =
        List.partition_map !ineqs ~f:(function
          | Dim_eq { d1; d2; origin } -> Either.First ((d1, d2), origin)
          | ineq -> Either.Second ineq)
      in
      let f (ineqs, env) (ds, origin) =
        let more_ineqs, env = unify_dim ~stage origin ds env in
        (more_ineqs @ ineqs, env)
      in
      List.fold ~init:(ineqs, env) dim_eqs ~f
  | Concat [], d | d, Concat [] ->
      (* An empty concat is the size-0 axis. *)
      unify_dim ~stage origin (get_default_dim ~d:0 (), d) env
  | Concat [ x ], d | d, Concat [ x ] ->
      (* Single-component concat erases to its component. *)
      unify_dim ~stage origin (x, d) env
  | Concat xs, Concat ys ->
      (* Arithmetic equality of two concatenated axes. Cancel structurally-equal components, then
         reconcile the residual sums: a fully-solved side reduces to a [Dim] comparison; mixed
         solved/unsolved subtracts the solved part; an all-unsolved equal-length residual (the pure
         nested-stacking case, where corresponding fresh stack axes match positionally) pairs
         element-wise; anything still ambiguous is deferred until substitution resolves more. *)
      let xs, ys = cancel_common_dims xs ys in
      let solved_x, rest_x = partition_solved_dims xs in
      let solved_y, rest_y = partition_solved_dims ys in
      let sum_x = List.sum (module Int) solved_x ~f:(fun s -> s.d) in
      let sum_y = List.sum (module Int) solved_y ~f:(fun s -> s.d) in
      let basis_of solved =
        match reduce_solved_basis solved with
        | Some b -> b
        | None ->
            raise
            @@ Shape_error ("concat: conflicting dimension bases", [ Dim_mismatch [ dim1; dim2 ] ])
      in
      let basis_x = basis_of solved_x and basis_y = basis_of solved_y in
      let exceed () =
        raise
        @@ Shape_error
             ("concat: components exceed the concatenated size", [ Dim_mismatch [ dim1; dim2 ] ])
      in
      (match (rest_x, rest_y) with
      | [], [] ->
          unify_dim ~stage origin
            (get_dim ~d:sum_x ~basis:basis_x (), get_dim ~d:sum_y ~basis:basis_y ())
            env
      | [], _ ->
          let residual = sum_x - sum_y in
          if residual < 0 then exceed ();
          unify_dim ~stage origin (Concat rest_y, get_dim ~d:residual ~basis:basis_x ()) env
      | _, [] ->
          let residual = sum_y - sum_x in
          if residual < 0 then exceed ();
          unify_dim ~stage origin (Concat rest_x, get_dim ~d:residual ~basis:basis_y ()) env
      | _, _
        when List.is_empty solved_x && List.is_empty solved_y
             && List.length rest_x = List.length rest_y ->
          List.fold (List.zip_exn rest_x rest_y) ~init:([], env) ~f:(fun (acc, env) (a, b) ->
              let more, env = unify_dim ~stage origin (a, b) env in
              (more @ acc, env))
      | _, _ ->
          let canon rest sum basis =
            if sum = 0 then Concat rest else Concat (rest @ [ get_dim ~d:sum ~basis () ])
          in
          ( [ Dim_eq { d1 = canon rest_x sum_x basis_x; d2 = canon rest_y sum_y basis_y; origin } ],
            env ))
  | Concat xs, (Dim n as d) | (Dim n as d), Concat xs ->
      (* A concatenated axis equals a solved size: subtract the solved components and bind/defer the
         residual. *)
      let solved, rest = partition_solved_dims xs in
      let sum = List.sum (module Int) solved ~f:(fun s -> s.d) in
      let solved_basis =
        match reduce_solved_basis solved with
        | Some b -> b
        | None ->
            raise
            @@ Shape_error ("concat: conflicting dimension bases", [ Dim_mismatch [ dim1; dim2 ] ])
      in
      let residual = n.d - sum in
      if residual < 0 then
        raise
        @@ Shape_error
             ("concat: components exceed the concatenated size", [ Dim_mismatch [ dim1; dim2 ] ]);
      let residual_basis =
        merge_derived_basis solved_basis n.basis ~on_conflict:(fun () ->
            raise
            @@ Shape_error ("concat: conflicting dimension bases", [ Dim_mismatch [ dim1; dim2 ] ]))
      in
      (match rest with
      | [] -> unify_dim ~stage origin (get_dim ~d:sum ~basis:solved_basis (), d) env
      | [ c ] -> unify_dim ~stage origin (c, get_dim ~d:residual ~basis:residual_basis ()) env
      | _ ->
          (* Multiple unsolved components remain: defer the NORMALIZED residual (solved components
             subtracted), not the original equation, so the arithmetic progress is not lost and the
             same unsolved form is not requeued. *)
          ([ Dim_eq { d1 = Concat rest; d2 = get_dim ~d:residual ~basis:residual_basis (); origin } ],
            env))
  | Concat _, _ | _, Concat _ ->
      (* Concat against a not-yet-reducible dimension (e.g. an unresolved affine): defer rather than
         reject, so a later substitution can make progress. *)
      ([ Dim_eq { d1 = dim1; d2 = dim2; origin } ], env)
  | dim1, dim2 ->
      (* Note: at the unify_dim phase, it's strict equality (no broadcasting). *)
      raise @@ Shape_error ("solved dimensions for axis: mismatch", [ Dim_mismatch [ dim1; dim2 ] ])

let drop_from_end l n = List.rev @@ List.drop (List.rev l) n
let take_from_end (l : dim list) (n : int) : dim list = List.rev @@ List.take (List.rev l) n
let safe_to_guess = Hash_set.create (module Row_var)
let add_safe_to_guess v = Hash_set.add safe_to_guess v
let used_in_spec_or_compose = Hash_set.create (module Row_var)
let used_in_pointwise = Hash_set.create (module Row_var)
let add_used_in_spec_or_compose v = Hash_set.add used_in_spec_or_compose v
let add_used_in_pointwise v = Hash_set.add used_in_pointwise v

(* Persistent rank-relation graph, global like [global_template_cache] (row variable ids are never
   reused, so entries from unrelated inference problems are unreachable). An edge [v -> (w, k)]
   with [k >= 0] records the entailed fact [rank v >= rank w + k]: from solving
   [v := <w> ++ k known dims], from equal-flank row inequalities [<v>+flanks ⊑ <w>+flanks]
   (k = 0), and from the deficit (template) rule (k = the rank deficit > 0).
   Such facts are never retracted, unlike the [Bounds_row] res/opnd adjacency lists which drop
   entries when a variable is substituted away — mutually-growing row variables ping-pong between
   the store and the in-flight constraint list, so divergence detection needs this side table. *)
let global_rank_edges = Hashtbl.create (module Row_var)

(* Would adding [rank v >= rank w + k] close a cycle of positive total weight? A path [w ~> v] of
   total weight [s] entails [rank v >= rank v + s + k]: contradiction iff [s + k > 0]. Stored
   weights are nonnegative, so for [k > 0] reachability suffices, and for [k = 0] the path must
   contain a positive edge; DFS over (node, still-needs-positive) states. *)
let closes_positive_rank_cycle ~src:v ~dst:w k =
  let visited_need = Hash_set.create (module Row_var) in
  let visited_done = Hash_set.create (module Row_var) in
  let rec dfs u need_positive =
    (equal_row_var u v && not need_positive)
    ||
    let visited = if need_positive then visited_need else visited_done in
    (not (Hash_set.mem visited u))
    && (Hash_set.add visited u;
        List.exists (Hashtbl.find_multi global_rank_edges u) ~f:(fun (u', k') ->
            dfs u' (need_positive && k' = 0)))
  in
  dfs w (k = 0)

(* Record the entailed fact [rank v >= rank w + k]; raise if it makes the ranks unsatisfiable,
   which would otherwise diverge by minting ever-fresh template variables (rank-cycle check,
   transitive generalization of the one-step self-reference check). *)
let add_rank_edge ~rows v w k =
  if equal_row_var v w then (
    if k > 0 then
      raise @@ Shape_error ("Infinite number of axes by self-reference", [ Row_mismatch rows ]))
  else if closes_positive_rank_cycle ~src:v ~dst:w k then
    raise
    @@ Shape_error
         ("Infinite number of axes by rank cycle among row variables", [ Row_mismatch rows ])
  else
    let edges = Hashtbl.find_multi global_rank_edges v in
    if not (List.mem edges (w, k) ~equal:(fun (w1, k1) (w2, k2) -> equal_row_var w1 w2 && k1 = k2))
    then Hashtbl.add_multi global_rank_edges ~key:v ~data:(w, k)

(* Equate two rows, no broadcasting. Does not resolve inequalities. *)
let%debug5_sexp rec unify_row ~stage origin (eq : t * t) env : constraint_ list * _ =
  let rec solve (ineqs, env) : constraint_ -> constraint_ list * _ = function
    | Dim_eq { d1; d2; origin = eq_origin } ->
        let origin = merge_origins eq_origin origin in
        let more_ineqs, env = unify_dim ~stage origin (d1, d2) env in
        List.fold ~init:(ineqs, env) more_ineqs ~f:solve
    | Row_eq { r1; r2; origin = eq_origin } ->
        let origin = merge_origins eq_origin origin in
        let more_ineqs, env = unify_row ~stage origin (r1, r2) env in
        (more_ineqs @ ineqs, env)
    | ( Dim_ineq _ | Row_ineq _ | Dim_constr _ | Rows_constr _ | Terminal_dim _ | Terminal_row _
      | Shape_row _ ) as ineq ->
        (ineq :: ineqs, env)
  in
  let unify_suffix init dims1 dims2 len =
    let dims1 = take_from_end dims1 len and dims2 = take_from_end dims2 len in
    List.fold ~init ~f:(fun acc (d1, d2) ->
        let constr = Dim_eq { d1; d2; origin } in
        try solve acc constr
        with Shape_error (s, trace) -> raise @@ Shape_error (s, Constraint_failed constr :: trace))
    @@ List.zip_exn dims1 dims2
  in
  let r1 : t = subst_row env @@ fst eq and r2 : t = subst_row env @@ snd eq in
  let l = List.length in
  match (r1, r2) with
  | r1, r2 when equal_row r1 r2 -> ([], env)
  | ( { beg_dims = beg_dims1; dims = dims1; bcast = Row_var v1; prov = _ },
      { beg_dims = beg_dims2; dims = dims2; bcast = Row_var v2; prov = _ } )
    when equal_row_var v1 v2 ->
      let dims1_l = l dims1
      and dims2_l = l dims2
      and beg_dims1_l = l beg_dims1
      and beg_dims2_l = l beg_dims2 in
      if beg_dims1_l + dims1_l <> beg_dims2_l + dims2_l then
        raise
        @@ Shape_error ("Infinite number of axes by self-reference", [ Row_mismatch [ r1; r2 ] ])
      else if beg_dims1_l <> beg_dims2_l then
        (* Equal total flank lengths but shifted splits: the equation is l1.<v>.r1 = l2.<v>.r2
           whose surplus flank words rotate through [v]'s value (the word equation
           x ++ t = s ++ x), so the residue depends on [v]'s eventual length. Resolve by
           deferral into the closing policy: keep the equation in flight; if [v] is solved by
           other constraints, the substituted closed-closed (flat) check is exact; otherwise
           stage 6 closes [v] upward -- the least-material disjunct -- after which the
           re-emitted equation requires the two surplus words to be equal. Eagerly binding
           [v] to the empty row here would be unsound: a later [v = [3]] is jointly satisfiable
           with [3].<v> = <v>.[3]. (This case was once silently dropped, accepting the
           unsatisfiable [3].<v> = <v>.[5] with no dimension checked.) *)
        if is_stage6_up stage then
          (* Emission order matters: the driver reverses the accumulated list each round, so the
             closing binding must come SECOND here to be processed FIRST next round (same idiom
             as [solve_row_ineq]'s stage-6 branch); otherwise the re-emitted equation reproduces
             the pair against the still-unsolved variable and the fixpoint stalls. *)
          ( [
              Row_eq { r1; r2; origin };
              Row_eq
                {
                  r1 = row_of_var v1 r1.prov;
                  r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov = r1.prov };
                  origin;
                };
            ],
            env )
        else ([ Row_eq { r1; r2; origin } ], env)
      else
        let result = unify_suffix ([], env) dims1 dims2 @@ min dims1_l dims2_l in
        (* Leading flank: outer-anchor (the reversed-then-take_from_end pair below pairs the OUTER
           prefixes of each list elementwise; see [unify_suffix] which is symmetric and the
           double-reverse cancels). *)
        unify_suffix result (List.rev beg_dims1) (List.rev beg_dims2) @@ min beg_dims1_l beg_dims2_l
  | ({ beg_dims = beg_dims1; dims = dims1; bcast = Row_var v; prov = _ } as r1), r2
  | r2, ({ beg_dims = beg_dims1; dims = dims1; bcast = Row_var v; prov = _ } as r1) -> (
      let dims1_l : int = l dims1
      and dims2_l : int = l r2.dims
      and beg_dims1_l : int = l beg_dims1 in
      let beg_dims2_l : int = l r2.beg_dims in
      let prov = merge_provenance r1.prov r2.prov in
      let beg_dims_l = min beg_dims1_l beg_dims2_l in
      if dims1_l > dims2_l || (dims1_l = dims2_l && beg_dims1_l > beg_dims2_l) then
        if is_row_var r2.bcast then unify_row ~stage origin (r2, r1) env
        else raise @@ Shape_error ("Number of axes mismatch", [ Row_mismatch [ r1; r2 ] ])
      else
        let orig_rows = [ r1; r2 ] in
        let (beg_handled : bool), (ineqs, env), (value : row) =
          match r2.bcast with
          | Row_var v2 ->
              let beg_dims2 = r2.beg_dims in
              if Hash_set.mem safe_to_guess v2 then add_safe_to_guess v;
              if Hash_set.mem used_in_spec_or_compose v2 then add_used_in_spec_or_compose v;
              if Hash_set.mem used_in_pointwise v2 then add_used_in_pointwise v;
              if Hash_set.mem safe_to_guess v then add_safe_to_guess v2;
              if Hash_set.mem used_in_spec_or_compose v then add_used_in_spec_or_compose v2;
              if Hash_set.mem used_in_pointwise v then add_used_in_pointwise v2;
              let result =
                try unify_suffix ([], env) dims1 r2.dims dims1_l
                with Shape_error (s, trace) ->
                  raise @@ Shape_error (s, Row_mismatch orig_rows :: trace)
              in
              let dims = drop_from_end r2.dims dims1_l in
              if equal_row_var v v2 then
                if List.is_empty dims && l beg_dims2 = l beg_dims1 then
                  let value : row =
                    { beg_dims = []; dims; bcast = Row_var v; prov }
                  in
                  ( true,
                    unify_suffix result (List.rev beg_dims1) (List.rev beg_dims2) @@ l beg_dims2,
                    value )
                else
                  raise
                  @@ Shape_error
                       ("Infinite number of axes by self-reference", [ Row_mismatch orig_rows ])
              else
                let result =
                  unify_suffix result (List.rev beg_dims1) (List.rev beg_dims2) beg_dims_l
                in
                let value : row =
                  {
                    beg_dims = List.drop beg_dims2 beg_dims_l;
                    dims;
                    bcast = Row_var v2;
                    prov;
                  }
                in
                (beg_dims_l = l beg_dims1, result, value)
          | Broadcastable ->
              (* r2 is closed with structural flanks beg_dims2/dims2. We match r1.beg_dims1
                 outer-left and r1.dims1 outer-right against r2's flat axis list. The middle
                 (axes absorbed by v) preserves r2's structural split: anything remaining in
                 r2.beg_dims becomes value.beg_dims; anything remaining in r2.dims becomes
                 value.dims. r1's flanks may spill over into r2's other flank if r1's flank
                 length exceeds r2's matching flank length — handled below. *)
              let r2_flat = r2.beg_dims @ r2.dims in
              let r2_l = beg_dims2_l + dims2_l in
              if dims1_l + beg_dims1_l > r2_l then
                raise @@ Shape_error ("Number of axes mismatch", [ Row_mismatch [ r1; r2 ] ])
              else
                let beg_overlap = min beg_dims1_l beg_dims2_l in
                let end_overlap = min dims1_l dims2_l in
                let beg_spill = beg_dims1_l - beg_overlap in
                let end_spill = dims1_l - end_overlap in
                let value_beg_dims =
                  r2.beg_dims
                  |> Fn.flip List.drop beg_overlap
                  |> Fn.flip drop_from_end end_spill
                in
                let value_dims =
                  r2.dims
                  |> Fn.flip List.drop beg_spill
                  |> Fn.flip drop_from_end end_overlap
                in
                let result =
                  List.zip_exn beg_dims1 (List.take r2_flat beg_dims1_l)
                  @ List.zip_exn dims1 (take_from_end r2_flat dims1_l)
                  |> List.fold ~init:([], env) ~f:(fun acc (d1, d2) ->
                      solve acc (Dim_eq { d1; d2; origin }))
                in
                let value : row =
                  { beg_dims = value_beg_dims; dims = value_dims; bcast = Broadcastable; prov }
                in
                (true, result, value)
        in
        (* From now on, we have no use for un-reduced r2 since we deal with the row variable. *)
        let r2 : row = value in
        let ineqs : constraint_ list ref = ref ineqs in
        let f in_ =
          let more_ineqs, result = s_row_one_in_entry v ~(value : row) ~in_ in
          ineqs := more_ineqs @ !ineqs;
          result
        in
        let result env =
          let row_env = Utils.Tree_map.map env.row_env ~f in
          let unsolved, env =
            if beg_handled then (
              (match value.bcast with
              | Row_var u ->
                  add_rank_edge ~rows:orig_rows v u
                    (List.length value.beg_dims + List.length value.dims)
              | Broadcastable -> ());
              let constr =
                match find_row env.row_env v with
                | Some (Bounds_row { constr; origin; _ }) ->
                    [ Rows_constr { r = [ value ]; constr; origin } ]
                | _ -> []
              in
              (constr, { env with row_env = add_row row_env ~key:v ~data:(Solved_row value) }))
            else
              ( [
                  Row_eq
                    {
                      r1 =
                        {
                          beg_dims = List.drop beg_dims1 beg_dims_l;
                          dims = [];
                          bcast = Row_var v;
                          prov;
                        };
                      r2;
                      origin;
                    };
                ],
                env )
          in
          List.fold ~init:(unsolved, env) ~f:solve !ineqs
        in
        match find_row env.row_env v with
        | None -> result env
        | Some (Solved_row _) -> assert false
        | Some (Bounds_row { is_in_param = _; res; opnd; glb; constr; origin = origin1 }) ->
            let origin = merge_origins origin origin1 in
            let env =
              if beg_handled then (
                List.iter res ~f:(fun res ->
                    ineqs :=
                      Row_ineq { res = row_of_var res value.prov; opnd = r2; origin } :: !ineqs);
                List.iter opnd ~f:(fun opnd ->
                    ineqs :=
                      Row_ineq { opnd = row_of_var opnd value.prov; res = r2; origin } :: !ineqs);
                Option.iter glb ~f:(fun glb ->
                    ineqs := Row_ineq { res = glb; opnd = r2; origin } :: !ineqs);
                let extras, env = apply_row_constraint ~depth:0 stage origin value constr env in
                ineqs := extras @ !ineqs;
                env)
              else env
            in
            let _bound_elim_ineqs : constraint_ list = !ineqs in
            result env)
  | ( ({ bcast = Broadcastable; _ } as r1),
      ({ bcast = Broadcastable; _ } as r2) ) -> (
      (* Two closed rows must equal axis-by-axis, including both flanks. *)
      let r1_flat = r1.beg_dims @ r1.dims in
      let r2_flat = r2.beg_dims @ r2.dims in
      match List.zip r1_flat r2_flat with
      | Unequal_lengths ->
          raise @@ Shape_error ("Mismatching number of axes", [ Row_mismatch [ r1; r2 ] ])
      | Ok eqs ->
          List.fold ~init:([], env)
            ~f:(fun acc (d1, d2) -> solve acc (Dim_eq { d1; d2; origin }))
            eqs)

let%track5_sexp solve_dim_ineq ~(stage : stage) origin ~(res : dim) ~(opnd : dim) env :
    constraint_ list * _ =
  let nonredundant ?(more = []) (v : dim_var) (vs : dim_var list) : dim_var list =
    let _more : dim_var list = more in
    Utils.sorted_diff ~compare:compare_dim_var
      (List.dedup_and_sort ~compare:compare_dim_var (v :: vs))
      more
  in
  let rec cyclic ~opnd_v ~ress =
    (* TODO: it's somewhat inefficient *)
    List.exists ress ~f:(fun res_v ->
        equal_dim_var opnd_v res_v
        ||
        match find_dim env.dim_env res_v with
        | None | Some (Solved_dim (Dim _)) -> false
        | Some (Solved_dim (Var v)) -> equal_dim_var opnd_v v
        | Some (Solved_dim (Affine _)) -> false (* Affine dimensions can't be cyclic *)
        | Some (Solved_dim (Concat _)) -> false (* Concat dimensions can't be cyclic *)
        | Some (Bounds_dim { res = ress; _ }) -> cyclic ~opnd_v ~ress)
  in
  (* The relation enforced here is [res ⊑ opnd] (res the result side that refines, opnd the
     broadcastable operand side). The broadcast order is now a flat partial order: [res ⊑ opnd] iff
     they are equal as dims (same size AND same tag) or [opnd = 1_(bcast_if_1)] (the claim-free top,
     above everything). There is no longer a
     wildcard that matches any same-size dim. Inequality records no basis update by design: every
     dimension carries a concrete tag at construction time, so there is no unspecified ([None]) side
     to propagate onto — the leak that was latent under [None] is closed by construction. *)
  match (res, opnd) with
  | res, opnd when equal_dim res opnd -> ([], env)
  | _, Dim { d = 1; basis; _ } when String.equal basis bcast_if_1 ->
      (* opnd is the top: [res ⊑ top] for every res. Accept, record nothing. *)
      ([], env)
  | (Dim { d = 1; basis; _ } as res), _ when String.equal basis bcast_if_1 ->
      (* res is the top: [top ⊑ opnd] only by equality — the top refines only itself; force it. *)
      ([ Dim_eq { d1 = opnd; d2 = res; origin } ], env)
  | Dim { basis = b1; _ }, Dim { basis = b2; _ } when not (String.equal b1 b2) ->
      raise
      @@ Shape_error
           ("dimension comparison for axis: different bases", [ Dim_mismatch [ res; opnd ] ])
  | Affine _, _ | _, Affine _ -> ([ Dim_eq { d1 = opnd; d2 = res; origin } ], env)
  | Var res_v, Var opnd_v -> (
      match (find_dim env.dim_env res_v, find_dim env.dim_env opnd_v) with
      | Some (Bounds_dim { res = res1; _ }), _ when List.mem ~equal:equal_dim_var res1 opnd_v ->
          ([ Dim_eq { d1 = res; d2 = opnd; origin } ], env)
      | _, Some (Bounds_dim { opnd = opnd2; _ }) when List.mem ~equal:equal_dim_var opnd2 res_v ->
          ([ Dim_eq { d1 = res; d2 = opnd; origin } ], env)
      | None, None ->
          ( [],
            {
              env with
              dim_env =
                env.dim_env
                |> add_dim ~key:res_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param = false;
                            has_uniq_constr_unless = None;
                            glb = None;
                            res = [];
                            opnd = [ opnd_v ];
                            constr = Unconstrained_dim;
                            origin;
                          })
                |> add_dim ~key:opnd_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param = false;
                            has_uniq_constr_unless = None;
                            glb = None;
                            res = [ res_v ];
                            opnd = [];
                            constr = Unconstrained_dim;
                            origin;
                          });
            } )
      | Some (Solved_dim _), _ | _, Some (Solved_dim _) -> assert false
      | ( Some
            (Bounds_dim
               {
                 is_in_param;
                 res = res1;
                 opnd = opnd1;
                 glb = glb1;
                 constr = constr1;
                 origin = origin1;
                 _;
               } as in_),
          None ) ->
          let origin = merge_origins origin origin1 in
          let from_ = [%sexp_of: dim_var * dim_entry] (res_v, drop_origin in_) in
          let from_glb =
            Option.to_list glb1 |> List.map ~f:(fun res -> Dim_ineq { res; opnd; from_; origin })
          in
          let from_constr1, constr1 = apply_dim_constraint ~source:Opnd ~stage opnd constr1 env in
          let from_constr2, constr2 =
            apply_dim_constraint ~source:Res ~stage res Unconstrained_dim env
          in
          ( from_constr1 @ from_constr2 @ from_glb,
            {
              env with
              dim_env =
                env.dim_env
                |> add_dim ~key:res_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param;
                            has_uniq_constr_unless = None;
                            glb = glb1;
                            res = res1;
                            opnd = nonredundant opnd_v opnd1;
                            constr = constr1;
                            origin;
                          })
                |> add_dim ~key:opnd_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param = false;
                            has_uniq_constr_unless = None;
                            glb = None;
                            res = [ res_v ];
                            opnd = [];
                            constr = constr2;
                            origin;
                          });
            } )
      | ( Some
            (Bounds_dim
               {
                 is_in_param = _;
                 res = _;
                 opnd = [ opnd1 ];
                 glb = None;
                 constr = _;
                 origin = origin1;
                 _;
               }),
          Some
            (Bounds_dim
               {
                 is_in_param = _;
                 res = [ res2 ];
                 opnd = _;
                 glb = None;
                 constr = _;
                 origin = origin2;
                 _;
               }) )
        when is_stage2_up stage && equal_dim_var opnd_v opnd1 && equal_dim_var res_v res2 ->
          let origin = merge_origins origin (merge_origins origin1 origin2) in
          (* A heuristic to reduce template variables coming from e.g. einsum notation expansion. *)
          ([ Dim_eq { d1 = opnd; d2 = res; origin } ], env)
      | Some (Bounds_dim { res = ress; origin = origin1; _ }), Some (Bounds_dim _)
        when cyclic ~opnd_v ~ress ->
          let origin = merge_origins origin origin1 in
          ([ Dim_eq { d1 = opnd; d2 = res; origin } ], env)
      | ( None,
          Some
            (Bounds_dim
               {
                 is_in_param;
                 res = res2;
                 opnd = opnd2;
                 glb = glb2;
                 constr = constr2;
                 origin = origin2;
                 _;
               }) ) ->
          let origin = merge_origins origin origin2 in
          let from_constr1, constr1 =
            apply_dim_constraint ~source:Opnd ~stage opnd Unconstrained_dim env
          in
          let from_constr2, constr2 = apply_dim_constraint ~source:Res ~stage res constr2 env in
          ( from_constr2 @ from_constr1,
            {
              env with
              dim_env =
                env.dim_env
                |> add_dim ~key:res_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param;
                            has_uniq_constr_unless = None;
                            glb = None;
                            res = [];
                            opnd = [ opnd_v ];
                            constr = constr1;
                            origin;
                          })
                |> add_dim ~key:opnd_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param;
                            has_uniq_constr_unless = None;
                            glb = glb2;
                            res = nonredundant res_v res2;
                            opnd = opnd2;
                            constr = constr2;
                            origin;
                          });
            } )
      | ( Some
            (Bounds_dim
               {
                 is_in_param = iip1;
                 res = res1;
                 opnd = opnd1;
                 glb = glb1;
                 constr = constr1;
                 origin = origin1;
                 _;
               } as in_),
          Some
            (Bounds_dim
               {
                 is_in_param = iip2;
                 res = res2;
                 opnd = opnd2;
                 glb = glb2;
                 constr = constr2;
                 origin = origin2;
                 _;
               }) ) ->
          let origin = merge_origins origin (merge_origins origin1 origin2) in
          let from_ = [%sexp_of: dim_var * dim_var * dim_entry] (res_v, opnd_v, drop_origin in_) in
          let from_glb =
            Option.to_list glb1 |> List.map ~f:(fun res -> Dim_ineq { res; opnd; from_; origin })
          in
          let from_constr1, constr1 = apply_dim_constraint ~source:Opnd ~stage opnd constr1 env in
          let from_constr2, constr2 = apply_dim_constraint ~source:Res ~stage res constr2 env in
          ( from_constr1 @ from_constr2 @ from_glb,
            {
              env with
              dim_env =
                env.dim_env
                |> add_dim ~key:res_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param = iip1;
                            has_uniq_constr_unless = None;
                            glb = glb1;
                            res = res1;
                            opnd = nonredundant ~more:opnd2 opnd_v opnd1;
                            constr = constr1;
                            origin;
                          })
                |> add_dim ~key:opnd_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param = iip2;
                            has_uniq_constr_unless = None;
                            glb = glb2;
                            res = nonredundant ~more:res1 res_v res2;
                            opnd = opnd2;
                            constr = constr2;
                            origin;
                          });
            } ))
  | _, Var opnd_v -> (
      match find_dim env.dim_env opnd_v with
      | None ->
          ( [],
            {
              env with
              dim_env =
                add_dim env.dim_env ~key:opnd_v
                  ~data:
                    (Bounds_dim
                       {
                         is_in_param = false;
                         has_uniq_constr_unless = None;
                         glb = Some res;
                         res = [];
                         opnd = [];
                         constr = Unconstrained_dim;
                         origin;
                       });
            } )
      | Some (Solved_dim _) -> assert false
      | Some
          (Bounds_dim
             {
               is_in_param;
               res = res2;
               opnd = opnd2;
               glb = Some glb2;
               constr = constr2;
               origin = origin2;
               _;
             }) ->
          let origin = merge_origins origin origin2 in
          let glb, glb_forcing =
            match (res, glb2) with
            | Dim _, Dim _ when equal_dim res glb2 ->
                (* Same size and same tag: keep the existing bound. (Basis is total now, so there
                   is no wildcard "one side unspecified, prefer the other" case to handle here.) *)
                (res, [])
            | Dim _, Dim _ (* different size or different basis *) ->
                (* Intentional broadcast semantics: conflicting bases (or different sizes) demote
                   to the broadcast top 1_(bcast_if_1), meaning these axes are incompatible and
                   should be broadcast. This is NOT a bug — do not tighten to raise Shape_error. *)
                let glb = get_bcast_dim ~d:1 ~proj_id:47 () in
                (glb, [ Dim_eq { d1 = opnd; d2 = glb; origin } ])
            | Var _, _ | _, Var _ -> assert false
            | Affine _, _ | _, Affine _ -> assert false
            | Concat _, _ | _, Concat _ -> assert false
          in
          let from_constr, constr2 = apply_dim_constraint ~source:Res ~stage res constr2 env in
          ( from_constr @ glb_forcing,
            {
              env with
              dim_env =
                add_dim env.dim_env ~key:opnd_v
                  ~data:
                    (Bounds_dim
                       {
                         is_in_param;
                         has_uniq_constr_unless = None;
                         glb = Some glb;
                         res = res2;
                         opnd = opnd2;
                         constr = constr2;
                         origin;
                       });
            } )
      | Some
          (Bounds_dim
             {
               is_in_param;
               res = res2;
               opnd = opnd2;
               glb = None;
               constr = constr2;
               origin = origin2;
               _;
             } as in_) ->
          let origin = merge_origins origin origin2 in
          let from_ = [%sexp_of: dim_var * dim_entry] (opnd_v, drop_origin in_) in
          let from_constr, constr2 = apply_dim_constraint ~source:Res ~stage res constr2 env in
          ( from_constr
            @ List.map opnd2 ~f:(fun opnd_v -> Dim_ineq { res; opnd = Var opnd_v; from_; origin }),
            {
              env with
              dim_env =
                add_dim env.dim_env ~key:opnd_v
                  ~data:
                    (Bounds_dim
                       {
                         is_in_param;
                         has_uniq_constr_unless = None;
                         glb = Some res;
                         res = res2;
                         opnd = opnd2;
                         constr = constr2;
                         origin;
                       });
            } ))
  | Var _, Dim _ (* when d2 > 1 or based *) -> ([ Dim_eq { d1 = res; d2 = opnd; origin } ], env)
  | Concat dims1, Concat dims2 when List.length dims1 = List.length dims2 ->
      (* Element-wise unification of concatenated dimensions *)
      let eqs = List.map2_exn dims1 dims2 ~f:(fun d1 d2 -> Dim_eq { d1; d2; origin }) in
      (eqs, env)
  | _, Concat dims
    when List.count dims ~f:(function Var v -> Set.mem env.discardable_vars v | _ -> false)
         >= List.length dims - 1 ->
      (* Concat in opnd position with all-but-one (or all) components being discardable vars: preserve
         inequality so the solver can infer the variables *)
      ([ Dim_ineq { res; opnd; from_ = Sexp.List []; origin } ], env)
  | Concat _, _ | _, Concat _ ->
      (* Defer to dimension equality for concat with non-concat *)
      ([ Dim_eq { d1 = res; d2 = opnd; origin } ], env)
  | Dim _, Dim _ ->
      raise
      @@ Shape_error ("dimension comparison for axis: mismatch", [ Dim_mismatch [ res; opnd ] ])

let global_template_cache = Hashtbl.Poly.create ()

let%debug5_sexp solve_row_ineq ~(stage : stage) origin ~(res : t) ~(opnd : t) env :
    constraint_ list * _ =
  let nonredundant ?(more = []) v vs =
    Utils.sorted_diff ~compare:compare_row_var
      (List.dedup_and_sort ~compare:compare_row_var (v :: vs))
      more
  in
  let l = List.length in
  let res_dims_l : int = l res.dims and opnd_dims_l : int = l opnd.dims in
  let res_beg_dims = res.beg_dims and opnd_beg_dims = opnd.beg_dims in
  let res_beg_dims_l = l res_beg_dims and opnd_beg_dims_l = l opnd_beg_dims in
  let beg_dims_l = min res_beg_dims_l opnd_beg_dims_l in
  let dims_l = min res_dims_l opnd_dims_l in
  let from_ = sexp_of_constraint_ (drop_constraint_origin (Row_ineq { res; opnd; origin })) in
  (* Outer-anchor alignment on BOTH flanks: leading flank uses [List.take] (outer-left prefix);
     trailing flank uses [take_from_end] (outer-right suffix). This is symmetric with
     [unify_row]. *)
  let ineqs =
    List.map2_exn
      ~f:(fun res opnd -> Dim_ineq { res; opnd; from_; origin })
      (List.take res_beg_dims beg_dims_l)
      (List.take opnd_beg_dims beg_dims_l)
    @ List.map2_exn
        ~f:(fun res opnd -> Dim_ineq { res; opnd; from_; origin })
        (take_from_end res.dims dims_l) (take_from_end opnd.dims dims_l)
  in
  match (res, opnd) with
  | { bcast = Row_var v; prov; _ }, _ | _, { bcast = Row_var v; prov; _ }
    when is_stage6_up stage ->
      ( Row_ineq { res; opnd; origin }
        :: Row_eq
             {
               r1 = row_of_var v prov;
               r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov };
               origin;
             }
        :: ineqs,
        env )
  | res, opnd when equal_row res opnd -> ([], env)
  | { bcast = Row_var res_v; _ }, { bcast = Row_var opnd_v; _ }
    when equal_row_var res_v opnd_v ->
      if res_dims_l + res_beg_dims_l <> opnd_dims_l + opnd_beg_dims_l then
        raise
        @@ Shape_error ("Infinite number of axes by self-reference", [ Row_mismatch [ res; opnd ] ])
      else if res_dims_l <> opnd_dims_l then
        (* Equal totals but shifted splits around the shared variable: the residue is a pointwise
           chain through the variable's value, so it cannot be finalized here. Defer: if the
           variable is solved by other constraints the substituted check applies; otherwise the
           stage-6 branch above closes it upward (the least-material disjunct) and the constraint
           reduces to the closed-closed check below. (This case was once silently dropped,
           accepting the unsatisfiable [3].<v> <= <v>.[5].) *)
        (Row_ineq { res; opnd; origin } :: ineqs, env)
      else (ineqs, env)
  | { bcast = Row_var res_v; _ }, { bcast = Row_var opnd_v; _ }
    when res_dims_l = opnd_dims_l && res_beg_dims_l = opnd_beg_dims_l -> (
      add_rank_edge ~rows:[ res; opnd ] res_v opnd_v 0;
      match (find_row env.row_env res_v, find_row env.row_env opnd_v) with
      | Some (Bounds_row { res = res1; origin = origin1; _ }), _
        when List.mem ~equal:equal_row_var res1 opnd_v ->
          let origin = merge_origins origin origin1 in
          ( Row_eq { r1 = row_of_var opnd_v opnd.prov; r2 = row_of_var res_v res.prov; origin }
            :: ineqs,
            env )
      | _, Some (Bounds_row { opnd = opnd2; origin = origin2; _ })
        when List.mem ~equal:equal_row_var opnd2 res_v ->
          let origin = merge_origins origin origin2 in
          ( Row_eq { r1 = row_of_var opnd_v opnd.prov; r2 = row_of_var res_v res.prov; origin }
            :: ineqs,
            env )
      | ( Some (Bounds_row { opnd = [ opnd1 ]; origin = origin1; _ }),
          Some (Bounds_row { res = [ res2 ]; origin = origin2; _ }) )
        when is_stage2_up stage && equal_row_var opnd1 opnd_v && equal_row_var res2 res_v ->
          let origin = merge_origins origin (merge_origins origin1 origin2) in
          ( Row_eq { r1 = row_of_var opnd_v opnd.prov; r2 = row_of_var res_v res.prov; origin }
            :: ineqs,
            env )
      | Some (Bounds_row { opnd = opnd1; _ }), _ when List.mem ~equal:equal_row_var opnd1 opnd_v ->
          (ineqs, env)
      | _, Some (Bounds_row { res = res2; _ }) when List.mem ~equal:equal_row_var res2 res_v ->
          (ineqs, env)
      | None, None ->
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> add_row ~key:res_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param = false;
                            res = [];
                            opnd = [ opnd_v ];
                            glb = None;
                            constr = Unconstrained;
                            origin;
                          })
                |> add_row ~key:opnd_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param = false;
                            res = [ res_v ];
                            opnd = [];
                            glb = None;
                            constr = Unconstrained;
                            origin;
                          });
            } )
      | ( Some
            (Bounds_row
               {
                 is_in_param;
                 res = res1;
                 opnd = opnd1;
                 glb = glb1;
                 constr = constr1;
                 origin = origin1;
               }),
          None ) ->
          let origin = merge_origins origin origin1 in
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> add_row ~key:res_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param;
                            res = res1;
                            opnd = nonredundant opnd_v opnd1;
                            glb = glb1;
                            constr = constr1;
                            origin;
                          })
                |> add_row ~key:opnd_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param = false;
                            res = [ res_v ];
                            opnd = [];
                            glb = None;
                            constr = Unconstrained;
                            origin;
                          });
            } )
      | ( None,
          Some
            (Bounds_row
               {
                 is_in_param;
                 res = res2;
                 opnd = opnd2;
                 glb = glb2;
                 constr = constr2;
                 origin = origin2;
               }) ) ->
          let origin = merge_origins origin origin2 in
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> add_row ~key:opnd_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param;
                            res = nonredundant res_v res2;
                            opnd = opnd2;
                            glb = glb2;
                            constr = constr2;
                            origin;
                          })
                |> add_row ~key:res_v
                     ~data:
                       (Bounds_row
                          {
                            (* The bound shouldn't collapse on the param below. *)
                            is_in_param;
                            res = [];
                            opnd = [ opnd_v ];
                            glb = None;
                            constr = Unconstrained;
                            origin;
                          });
            } )
      | ( Some
            (Bounds_row
               {
                 is_in_param = iip1;
                 res = res1;
                 opnd = opnd1;
                 glb = glb1;
                 constr = constr1;
                 origin = origin1;
               }),
          Some
            (Bounds_row
               {
                 is_in_param = iip2;
                 res = res2;
                 opnd = opnd2;
                 glb = glb2;
                 constr = constr2;
                 origin = origin2;
               }) ) ->
          let origin = merge_origins origin (merge_origins origin1 origin2) in
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> add_row ~key:res_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param = iip1 || iip2;
                            res = res1;
                            opnd = nonredundant opnd_v opnd1;
                            glb = glb1;
                            constr = constr1;
                            origin;
                          })
                |> add_row ~key:opnd_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param = iip2;
                            res = nonredundant res_v res2;
                            opnd = opnd2;
                            glb = glb2;
                            constr = constr2;
                            origin;
                          });
            } )
      | Some (Solved_row _), _ | _, Some (Solved_row _) -> assert false)
  | { bcast = Row_var res_v; dims; _ }, _
    when res_dims_l + res_beg_dims_l < opnd_dims_l + opnd_beg_dims_l ->
      (* The template below commits [rank res_v >= rank opnd_v + deficit] with [deficit > 0]:
         record it (and detect rank cycles) before minting fresh template variables. *)
      (match opnd.bcast with
      | Broadcastable -> ()
      | Row_var opnd_v ->
          add_rank_edge ~rows:[ res; opnd ] res_v opnd_v
            (opnd_dims_l + opnd_beg_dims_l - (res_dims_l + res_beg_dims_l)));
      let budget = opnd_dims_l + opnd_beg_dims_l - (res_dims_l + res_beg_dims_l) in
      let more_dims_l = min budget @@ max 0 (opnd_dims_l - res_dims_l) in
      let more_dims : dim list =
        Array.(to_list @@ init more_dims_l ~f:(fun _ -> Var (get_var ())))
      in
      let budget = budget - more_dims_l in
      let more_beg_dims_l = min budget @@ max 0 (opnd_beg_dims_l - res_beg_dims_l) in
      let more_beg_dims : dim list =
        Array.(to_list @@ init more_beg_dims_l ~f:(fun _ -> Var (get_var ())))
      in
      (* The key of the template cache reflects that res_v will end up substituted by
         {dims=more_dims; bcast=Row_var templ_v}. TODO: should we cache more_dims also? *)
      let templ_key : row_var * int * int =
        (res_v, opnd_dims_l - res_dims_l, opnd_beg_dims_l - res_beg_dims_l)
      in
      let templ_v : row_var =
        Hashtbl.find_or_add global_template_cache templ_key ~default:get_row_var
      in
      if more_dims_l > 0 then add_safe_to_guess templ_v;
      let template : t =
        {
          beg_dims = res_beg_dims @ more_beg_dims;
          dims = more_dims @ dims;
          bcast = Row_var templ_v;
          prov = res.prov;
        }
      in
      (* We don't need to add any dimension inequalities, because they'll be captured by the extra
         row inequalities. *)
      ( [ Row_eq { r1 = res; r2 = template; origin }; Row_ineq { res = template; opnd; origin } ],
        env )
  | { bcast = Broadcastable; _ }, _ when res_dims_l + res_beg_dims_l < opnd_dims_l + opnd_beg_dims_l
    ->
      raise
      @@ Shape_error
           ( "Too many axes in an operand; maybe using * instead of *.?",
             [ Row_mismatch [ res; opnd ] ] )
  | { bcast; dims; prov = _; _ }, { bcast = Row_var opnd_v; _ }
    when opnd_dims_l <= res_dims_l && opnd_beg_dims_l <= res_beg_dims_l -> (
      (* GLB residue: drop the OUTER matched prefix from beg_dims and the OUTER matched suffix from
         dims — symmetric with the per-axis match. Note: no rank edge is recorded here — the
         entailed fact [rank res_v >= rank opnd_v - surplus] has a nonpositive weight, which the
         rank graph (nonnegative weights only) cannot represent. *)
      let r_res =
        {
          beg_dims = List.drop res.beg_dims beg_dims_l;
          dims = drop_from_end dims dims_l;
          bcast;
          prov = res.prov;
        }
      in
      match find_row env.row_env opnd_v with
      | None ->
          ( ineqs,
            {
              env with
              row_env =
                add_row env.row_env ~key:opnd_v
                  ~data:
                    (Bounds_row
                       {
                         is_in_param = false;
                         res = [];
                         opnd = [];
                         glb = Some r_res;
                         constr = Unconstrained;
                         origin;
                       });
            } )
      | Some
          (Bounds_row
             {
               is_in_param;
               res = res2;
               opnd = opnd2;
               glb = None;
               constr = constr2;
               origin = origin2;
             }) ->
          let origin = merge_origins origin origin2 in
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> add_row ~key:opnd_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param;
                            res = res2;
                            opnd = opnd2;
                            glb = Some r_res;
                            constr = constr2;
                            origin;
                          });
            } )
      | Some
          (Bounds_row
             {
               is_in_param;
               res = res2;
               opnd = opnd2;
               glb = Some glb2;
               constr = constr2;
               origin = origin2;
             }) -> (
          let origin = merge_origins origin origin2 in
          (* Row-level bound merge: prefer generality for broadcasting. The broadcast top
             [1_(bcast_if_1)] is most general; conflicting sizes or conflicting (named) bases
             generalize to that top — the deliberate-broadcast case (incompatible caps join to the
             top), not a basis-checking gap. The same join applies on both flanks. *)
          let join_dim d1 d2 =
            match (d1, d2) with
            | Dim { d = 1; basis; _ }, _ when String.equal basis bcast_if_1 -> d1
            | _, Dim { d = 1; basis; _ } when String.equal basis bcast_if_1 -> d2
            | Dim { d = n1; _ }, Dim { d = n2; _ } when n1 <> n2 -> get_bcast_dim ~d:1 ~proj_id:48 ()
            | Dim { basis = b1; _ }, Dim { basis = b2; _ } when not (String.equal b1 b2) ->
                get_bcast_dim ~d:1 ~proj_id:63 ()
            | ( Affine
                  {
                    stride;
                    over = Dim s;
                    conv = Some { use_padding = true; _ } | None;
                    stride_offset = _;
                  },
                Dim s' )
            | ( Dim s',
                Affine
                  {
                    stride;
                    over = Dim s;
                    conv = Some { use_padding = true; _ } | None;
                    stride_offset = _;
                  } )
              when stride * s.d <> s'.d ->
                get_bcast_dim ~d:1 ~proj_id:49 ()
            | ( Affine
                  {
                    stride;
                    over = Dim s;
                    conv = Some { kernel = Dim k; dilation; use_padding = false };
                    stride_offset = _;
                  },
                Dim s' )
              when (stride * (s.d - 1)) + effective_kernel_span ~dilation ~kernel_size:k.d
                   <> s'.d ->
                get_bcast_dim ~d:1 ~proj_id:50 ()
            | ( Dim s',
                Affine
                  {
                    stride;
                    over = Dim s;
                    conv = Some { kernel = Dim k; dilation; use_padding = false };
                    stride_offset = _;
                  } )
              when (stride * (s.d - 1)) + effective_kernel_span ~dilation ~kernel_size:k.d
                   <> s'.d ->
                get_bcast_dim ~d:1 ~proj_id:50 ()
            | ( Affine
                  {
                    stride = stride1;
                    over = Dim s1;
                    conv = Some { use_padding = true; _ } | None;
                    stride_offset = _;
                  },
                Affine
                  {
                    stride = stride2;
                    over = Dim s2;
                    conv = Some { use_padding = true; _ } | None;
                    stride_offset = _;
                  } )
              when stride1 * s1.d <> stride2 * s2.d ->
                get_bcast_dim ~d:1 ~proj_id:51 ()
            | ( Affine
                  {
                    stride = stride1;
                    over = Dim s1;
                    conv = Some { kernel = Dim k1; dilation = dilation1; use_padding = false };
                    stride_offset = _;
                  },
                Affine
                  {
                    stride = stride2;
                    over = Dim s2;
                    conv = Some { kernel = Dim k2; dilation = dilation2; use_padding = false };
                    stride_offset = _;
                  } )
              when (stride1 * (s1.d - 1))
                   + effective_kernel_span ~dilation:dilation1 ~kernel_size:k1.d
                   <> (stride2 * (s2.d - 1))
                      + effective_kernel_span ~dilation:dilation2 ~kernel_size:k2.d ->
                get_bcast_dim ~d:1 ~proj_id:52 ()
            | Var _, _ -> d1
            | _, Var _ -> d2
            | _, Dim _ -> d2
            | _ -> d1
          in
          (* Symmetric merge across both flanks: shorter on each side ("prefer generality"). *)
          let len1 = List.length r_res.dims and len2 = List.length glb2.dims in
          let glb_len = min len1 len2 in
          let beg_len1 = List.length r_res.beg_dims and beg_len2 = List.length glb2.beg_dims in
          let glb_beg_len = min beg_len1 beg_len2 in
          let glb_is_res =
            len1 + beg_len1 < len2 + beg_len2
            || (len1 + beg_len1 = len2 + beg_len2 && is_broadcastable res.bcast)
          in
          let glb_prov = if glb_is_res then r_res.prov else glb2.prov in
          (* Note: row-variable provenance is still kept on the shorter side; both flanks now merge
             correctly so the "we lose connection" caveat for the leading flank is gone. *)
          let glb_bcast = if glb_is_res then r_res.bcast else glb2.bcast in
          let glb_beg_dims =
            List.map2_exn
              (List.take r_res.beg_dims glb_beg_len)
              (List.take glb2.beg_dims glb_beg_len)
              ~f:join_dim
          in
          let glb_dims =
            List.map2_exn
              (take_from_end r_res.dims glb_len)
              (take_from_end glb2.dims glb_len)
              ~f:join_dim
          in
          let glb =
            { beg_dims = glb_beg_dims; dims = glb_dims; bcast = glb_bcast; prov = glb_prov }
          in
          let row_env =
            env.row_env
            |> add_row ~key:opnd_v
                 ~data:
                   (Bounds_row
                      {
                        is_in_param;
                        res = res2;
                        opnd = opnd2;
                        glb = Some glb;
                        constr = constr2;
                        origin;
                      })
          in
          match glb with
          | {
              beg_dims = [];
              dims = [] | [ Dim { d = 1; _ } ];
              bcast = Broadcastable;
              prov = _;
            } ->
              ( Row_eq { r1 = row_of_var opnd_v opnd.prov; r2 = glb; origin } :: ineqs,
                { env with row_env } )
          | _ -> (ineqs, { env with row_env }))
      | Some (Solved_row _) -> assert false)
  | _ when res_beg_dims_l > beg_dims_l && not (is_stage7 stage) ->
      (Row_ineq { res; opnd; origin } :: ineqs, env)
  | { bcast = Broadcastable; _ }, { bcast = Broadcastable; _ }
    when opnd_dims_l + opnd_beg_dims_l <= res_dims_l + res_beg_dims_l ->
      (* Both closed: the operand broadcasts only by inserting claim-free padding at its marker,
         so its EXPLICIT material always pins the corresponding result positions, aligned on the
         result's flat axis list from the outer edges; only the inserted middle positions are
         unconstrained. The structural beg-to-beg/dims-to-dims overlap pairs (in [ineqs]) are a
         subset of these flat pairs, so we emit the flat pairs alone. This replaces a vacuous
         accept that never compared cross-flank material, accepting the unsatisfiable
         [3].<closed> <= <closed>.[5] at equal ranks where no padding is inserted at all. *)
      let res_flat = res.beg_dims @ res.dims in
      ( List.map2_exn
          ~f:(fun res opnd -> Dim_ineq { res; opnd; from_; origin })
          (List.take res_flat opnd_beg_dims_l)
          opnd_beg_dims
        @ List.map2_exn
            ~f:(fun res opnd -> Dim_ineq { res; opnd; from_; origin })
            (take_from_end res_flat opnd_dims_l)
            opnd.dims,
        env )
  | { bcast = Row_var _; _ }, { bcast = Broadcastable; _ }
    when opnd_dims_l + opnd_beg_dims_l <= res_dims_l + res_beg_dims_l
         && (opnd_beg_dims_l > beg_dims_l || opnd_dims_l > dims_l) ->
      (* Open result, closed operand with explicit material outside the structural-flank overlap:
         the residual pairings depend on the variable's value. Defer -- at stage 6 the result row
         is closed upward and the constraint reduces to the closed-closed check above. *)
      (Row_ineq { res; opnd; origin } :: ineqs, env)
  | _, { bcast = Broadcastable; _ }
    when opnd_dims_l + opnd_beg_dims_l <= res_dims_l + res_beg_dims_l ->
      (ineqs, env)
  | { bcast = Row_var _ | Broadcastable; _ }, { bcast = Row_var _ | Broadcastable; _ } ->
      (Row_ineq { res; opnd; origin } :: ineqs, env)

(** Check if a dimension variable can be guessed to 1. A variable with has_uniq_constr_unless can
    only be guessed if at least one of the "unless" variables is also prevented from guessing (to
    break cycles). *)
let can_guess_dim_to_one env has_uniq_constr_unless =
  match has_uniq_constr_unless with
  | None -> true (* No restriction *)
  | Some unless_vars ->
      (* Can guess if at least one "unless" variable is also prevented from guessing *)
      Set.exists unless_vars ~f:(fun unless_v ->
          match find_dim env.dim_env unless_v with
          | Some (Bounds_dim { has_uniq_constr_unless = Some _; _ }) -> true
          | _ -> false)

let%debug5_sexp rec close_dim_terminal ~(stage : stage) ~is_param origin env (dim : dim) :
    constraint_ list =
  match dim with
  | Dim _ -> []
  | Var v -> (
      match find_dim env.dim_env v with
      | Some (Solved_dim _) -> assert false
      | Some (Bounds_dim { glb = Some glb; _ }) when is_stage4_up stage ->
          [ Dim_eq { d1 = dim; d2 = glb; origin } ]
      | Some
          (Bounds_dim
             { is_in_param; has_uniq_constr_unless; glb = None; constr = Unconstrained_dim; _ })
        when is_stage5_up stage ->
          (* Check if we can guess this variable to 1 (or 0 for discardable_vars) *)
          if not (can_guess_dim_to_one env has_uniq_constr_unless) then
            [ Terminal_dim (is_param, dim, origin) ]
          else if is_param || is_in_param then
            raise
            @@ Shape_error
                 ("You forgot to specify the hidden dimension(s) 1", [ Dim_mismatch [ dim ] ])
          else
            let guess_d = if Set.mem env.discardable_vars v then 0 else 1 in
            [ Dim_eq { d1 = dim; d2 = get_bcast_dim ~d:guess_d ~proj_id:53 (); origin } ]
      | _ when not (is_stage6_up stage) -> [ Terminal_dim (is_param, dim, origin) ]
      | _ -> [])
  | Affine _ ->
      (* The input dimension itself cannot be dim-1, and the over dimension doesn't become
         transitively terminal. *)
      []
  | Concat dims -> (
      (* For concatenation, filter out dimension-0 components and if one remains, close it. TODO:
         Consider guessing discardable_vars components to 0 at stage 6, but wait until needed. *)
      let non_zero_dims = List.filter dims ~f:(function Dim { d = 0; _ } -> false | _ -> true) in
      match non_zero_dims with
      | [ single ] -> close_dim_terminal ~stage ~is_param origin env single
      | _ -> [])

let last_dim_is dims p = match List.last dims with Some (Dim { d; _ }) -> p d | _ -> false

let _r_dims r = r.beg_dims @ r.dims

let row_var_is_in_param v env =
  match find_row env.row_env v with
  | Some (Bounds_row { is_in_param; _ }) -> is_in_param
  | _ -> false

let dim_var_is_in_param v env =
  match find_dim env.dim_env v with
  | Some (Bounds_dim { is_in_param; _ }) -> is_in_param
  | _ -> false

let is_safe_to_guess v =
  Hash_set.mem safe_to_guess v
  || (Hash_set.mem used_in_pointwise v && not (Hash_set.mem used_in_spec_or_compose v))

let%track5_sexp rec eliminate_rows_constraint ~depth stage origin ~glb (rows : row list)
    (constr : row_constraint) env : constraint_ list * environment =
  if depth > 4 then ([ Rows_constr { r = rows; constr; origin } ], env)
  else
    match rows_to_row_or_vars rows with
    | Either.First single_row ->
        eliminate_row_constraint ~depth:(depth + 1) stage origin ~terminal:false ~glb single_row
          constr env
    | Either.Second (_all_dims, row_vars) -> (
        let rev_row_vars = List.rev row_vars in
        match
          ( constr,
            List.findi rev_row_vars ~f:(fun _ (_, prov) ->
                List.exists prov ~f:(fun (origin : provenance_origin) ->
                    equal_kind origin.kind `Output)) )
        with
        | Total_elems _, Some (idx, (v, _id)) when is_stage3_up stage ->
            (* TODO: in stage 3, consider restricting to a strided dimension variable case. *)
            let other_vars : (row_var * provenance) list =
              List.filteri rev_row_vars ~f:(fun i _ -> i <> idx)
            in
            let other_eqs : constraint_ list =
              List.concat_map other_vars ~f:(fun (v, prov) ->
                  if
                    is_stage5_up stage
                    ||
                    match find_row env.row_env v with
                    | None
                    | Some (Bounds_row { is_in_param = false; glb = None; _ })
                    | Some (Bounds_row { glb = Some { dims = []; bcast = Broadcastable; _ }; _ }) ->
                        true
                    | _ -> false
                  then
                    let r1 = row_of_var v prov in
                    [
                      Row_eq
                        {
                          r1;
                          r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov };
                          origin;
                        };
                    ]
                  else [])
            in
            if is_stage5_up stage then
              let rows =
                List.map rows ~f:(function
                  | { bcast = Row_var v'; _ } as r when equal_row_var v' v -> r
                  | r ->
                      { r with beg_dims = []; dims = r.beg_dims @ r.dims; bcast = Broadcastable })
              in
              let ineqs, env =
                eliminate_rows_constraint ~depth:(depth + 1) stage origin ~glb rows constr env
              in
              (other_eqs @ ineqs, env)
            else (other_eqs @ [ Rows_constr { r = rows; constr; origin } ], env)
        | _ -> ([ Rows_constr { r = rows; constr; origin } ], env))

and eliminate_row_constraint ~depth stage origin ~terminal ~(glb : row option) (r : row)
    (constr : row_constraint) env : constraint_ list * environment =
  let keep_constr () =
    let ineqs, env = apply_row_constraint ~depth stage origin r constr env in
    List.fold ineqs ~init:([], env) ~f:(fun (ineqs, env) ineq ->
        match ineq with
        | Rows_constr { r = rows; constr; origin } ->
            let ineqs', env =
              eliminate_rows_constraint ~depth:(depth + 1) stage origin ~glb:None rows constr env
            in
            (ineqs @ ineqs', env)
        | ineq -> ([ ineq ], env))
  in
  match r with
  | { bcast = Broadcastable; _ } -> keep_constr ()
  | { beg_dims; dims; bcast = Row_var v; prov } -> (
      let r1 = row_of_var v prov in
      (* If glb is not provided from context, try to get it from the row environment. This is
         critical for non-terminal shapes where GLBs are populated through inequalities but wouldn't
         otherwise be available until Stage 6. However, we only use the environment GLB if it has
         fully resolved dimensions (no dimension variables), as partially resolved GLBs can prevent
         proper constraint resolution. *)
      let glb =
        match glb with
        | Some _ -> glb
        | None -> (
            match find_row env.row_env v with
            | Some (Bounds_row { glb = Some env_glb; _ }) -> (
                (* We need to substitute environment dimensions into the GLB to see if it's
                   resolved *)
                let env_glb = subst_row env env_glb in
                match collect_factors (env_glb.beg_dims @ env_glb.dims) with
                | Some (_, []) -> Some env_glb (* All dims are known constants after substitution *)
                | _ -> None (* GLB has unresolved dimension variables or collect_factors failed *))
            | _ -> None)
      in
      let opt_row_error () =
        if row_var_is_in_param v env && not (is_safe_to_guess v) then
          raise
          @@ Shape_error ("You forgot to specify the hidden dimension(s) 2", [ Row_mismatch [ r ] ])
      in
      let no_further_axes ~guess () =
        if guess then opt_row_error ();
        (* Close the bare row variable to empty Broadcastable; the original row's beg_dims is
           preserved at substitution time by s_row_one's uniform composition. *)
        Row_eq
          {
            r1;
            r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov };
            origin;
          }
      in
      (* Note: the reduced constraint applies to just the row variable. *)
      match reduce_row_constraint constr ~beg_dims ~dims with
      | Total_elems { numerator; divided_by } -> (
          let _divided_by : dim_var list = divided_by in
          match (numerator, divided_by, glb) with
          | Num_elems 1, vs, _ when is_stage5_up stage ->
              ( no_further_axes ~guess:false ()
                :: List.map vs ~f:(fun v ->
                    let d2 = get_bcast_dim ~d:1 ~proj_id:54 () in
                    Dim_eq { d1 = Var v; d2; origin }),
                env )
          | Num_elems d, [], None when d <> 1 && is_stage3_up stage ->
              let dim = get_default_dim ~d ~proj_id:55 () in
              ( [
                  Row_eq
                    {
                      r1;
                      r2 = { beg_dims = []; dims = [ dim ]; bcast = Broadcastable; prov };
                      origin;
                    };
                ],
                env )
          | Num_elems d, [], Some { dims; _ } when d <> 1 && last_dim_is dims (( = ) d) ->
              let dim = get_default_dim ~d ~proj_id:56 () in
              ( [
                  Row_eq
                    {
                      r1;
                      r2 = { beg_dims = []; dims = [ dim ]; bcast = Broadcastable; prov };
                      origin;
                    };
                ],
                env )
          | Num_elems _, [], Some glb ->
              let ineqs, env =
                apply_row_constraint ~depth:(depth + 1) stage origin
                  (if terminal then glb else r)
                  constr env
              in
              List.fold ineqs ~init:([], env) ~f:(fun (ineqs, env) ineq ->
                  match ineq with
                  | Rows_constr { r = rows; constr; origin } ->
                      let ineqs', env =
                        eliminate_rows_constraint ~depth stage origin ~glb:None rows constr env
                      in
                      (ineqs @ ineqs', env)
                  | ineq -> ([ ineq ], env))
          | Num_elems d, [ dv ], None when is_stage4_up stage ->
              ( no_further_axes ~guess:true ()
                :: [ Dim_eq { d1 = Var dv; d2 = get_default_dim ~d (); origin } ],
                env )
          | Num_elems d, [ v ], Some ({ dims; _ } as r2)
            when last_dim_is dims (fun d2 -> d % d2 = 0) ->
              let d2 = match List.last dims with Some (Dim { d; _ }) -> d | _ -> assert false in
              let row_eq =
                if d = d2 && is_stage5_up stage then no_further_axes ~guess:false ()
                else Row_eq { r1; r2; origin }
              in
              (row_eq :: [ Dim_eq { d1 = Var v; d2 = get_default_dim ~d:(d / d2) (); origin } ], env)
          | Strided_var { coeff; var; denom }, [], None
            when is_stage5_up stage
                 && (Utils.safe_force coeff > denom || denom % Utils.safe_force coeff <> 0) ->
              let coeff = Utils.safe_force coeff in
              let gcd = Utils.gcd coeff denom in
              let d = denom / gcd in
              let d2 = get_default_dim ~d () in
              let d3 = get_default_dim ~d:(coeff / gcd) () in
              (* opt_row_error (); *)
              ( [
                  Dim_eq { d1 = Var var; d2; origin };
                  Row_eq
                    {
                      r1;
                      r2 = { beg_dims = []; dims = [ d3 ]; bcast = Broadcastable; prov };
                      origin;
                    };
                ],
                env )
          | Strided_var { coeff; var; denom }, [], _
            when is_stage6_up stage && denom % Utils.safe_force coeff = 0 ->
              let d2 = get_default_dim ~d:(denom / Utils.safe_force coeff) () in
              if dim_var_is_in_param var env then
                raise
                @@ Shape_error
                     ( "You forgot to specify the hidden dimension(s) 3",
                       [ Dim_mismatch [ Var var ] ] )
              else ([ Dim_eq { d1 = Var var; d2; origin }; no_further_axes ~guess:true () ], env)
          | ( Strided_var { coeff; var; denom },
              [],
              Some ({ beg_dims = glb_beg; dims = glb_dims; bcast = _; prov = glb_prov } as glb) )
            when is_stage5_up stage && Utils.safe_force coeff > denom -> (
              (* Check if coeff > denom * product of known dimensions of the GLB. The constraint is:
                 coeff * var / denom = total_elements(row). So: var = total_elements * denom /
                 coeff. *)
              match collect_factors (glb_beg @ glb_dims) with
              | Some (known_product, []) ->
                  let coeff_val = Utils.safe_force coeff in
                  if coeff_val > denom * known_product then
                    ([ Row_eq { r1; r2 = glb; origin } ], env)
                  else
                    (* Equate the row variable to the dimensions of the GLB, and compute var from
                       the total elements *)
                    let var_value = known_product * denom / coeff_val in
                    ( [
                        Row_eq
                          {
                            r1;
                            r2 =
                              {
                                beg_dims = glb_beg;
                                dims = glb_dims;
                                bcast = Broadcastable;
                                prov = glb_prov;
                              };
                            origin;
                          };
                        Dim_eq { d1 = Var var; d2 = get_default_dim ~d:var_value (); origin };
                      ],
                      env )
              | _ -> keep_constr ())
          | Strided_var { coeff; var; denom }, _, _ when is_stage5_up stage ->
              let _var : dim_var = var in
              let _coeff : int = Utils.safe_force coeff in
              let _denom : int = denom in
              keep_constr ()
          | _ -> keep_constr ())
      | Exact dims ->
          ( [
              Row_eq
                {
                  r1;
                  r2 = { beg_dims = []; dims; bcast = Broadcastable; prov };
                  origin;
                };
            ],
            env )
      | Unconstrained -> ([], env))

let%track5_sexp close_row_terminal ~(stage : stage) ~is_param origin env
    ({ beg_dims; dims; bcast; prov } as _r : row) : constraint_ list =
  let suffix () = List.map dims ~f:(fun d -> Terminal_dim (is_param, d, origin)) in
  (* TODO: can this be simplified? Should we return the environment? *)
  match bcast with
  | Broadcastable ->
      if is_stage6_up stage then []
      else
        List.map beg_dims ~f:(fun d -> Terminal_dim (is_param, d, origin)) @ suffix ()
  | Row_var v -> (
      let term_dims () =
        List.map beg_dims ~f:(fun d -> Terminal_dim (is_param, d, origin)) @ suffix ()
      in
      let r1 : row = row_of_var v prov in
      (* Close the bare row variable to empty Broadcastable; the original row's beg_dims is
         preserved through substitution via s_row_one's uniform composition. *)
      let no_further_axes =
        Row_eq
          {
            r1;
            r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov };
            origin;
          }
      in
      match find_row env.row_env v with
      | Some (Bounds_row { is_in_param; glb = None; constr = Unconstrained; _ })
        when is_stage4_up stage ->
          if (is_param || is_in_param) && not (is_safe_to_guess v) then
            raise @@ Shape_error ("You forgot to specify the hidden dimension(s) 4", [])
          else (
            [%log6 "terminal row: closing", (_r : row)];
            no_further_axes :: term_dims ())
      | Some (Bounds_row { glb = None; constr; _ })
        when is_stage2_up stage && not (equal_row_constraint constr Unconstrained) ->
          let ineqs, _env =
            (* This is the constraint on the row variable, not on the original row. *)
            try
              eliminate_row_constraint ~depth:0 stage origin r1 ~terminal:true ~glb:None constr env
            with Shape_error (s, trace) -> raise @@ Shape_error (s, Row_mismatch [ r1 ] :: trace)
          in
          (* FIXME: at which stage should we drop the terminal row? *)
          let keep_terminal =
            if is_stage6_up stage then [] else [ Terminal_row (is_param, r1, origin) ]
          in
          ineqs @ term_dims () @ keep_terminal
      | Some (Solved_row _) -> assert false
      | Some (Bounds_row { glb = Some _; constr = Total_elems { numerator = Num_elems 1; _ }; _ })
        when is_stage3_up stage ->
          term_dims ()
      | Some (Bounds_row { glb = Some glb; origin = glb_origin; _ }) when is_stage3_up stage ->
          let origin = merge_origins origin glb_origin in
          Row_eq { r1; r2 = glb; origin } :: term_dims ()
      | _ when is_stage6_up stage -> []
      | _ ->
          [%log6 "terminal row: keeping", (_r : row), "as", (r1 : row)];
          Terminal_row (is_param, r1, origin) :: term_dims ())

let%debug5_sexp eliminate_dim_entry stage origin env v ~glb constr =
  let guess_dim () =
    if Set.mem env.discardable_vars v then get_default_dim ~d:0 ~proj_id:56 () else get_bcast_dim ~d:1 ~proj_id:59 ()
  in
  match (glb, constr) with
  | Some (Dim { d; _ } as glb), At_least_dim d2 when d2 > d ->
      raise
      @@ Shape_error
           ( [%string "dereferenced at dimension %{d2#Int}, higher than use site"],
             [ Dim_mismatch [ glb; Var v ] ] )
  | Some _, At_least_dim 1 ->
      (* Direct access at 0 is a strong heuristic for dimension 1 axis (e.g. result of a
         reduction). *)
      if is_stage7 stage then Some (Dim_eq { d1 = Var v; d2 = get_bcast_dim ~d:1 ~proj_id:57 (); origin })
      else None
  | Some glb, (At_least_dim _ | Unconstrained_dim) when is_stage6_up stage ->
      Some (Dim_eq { d1 = Var v; d2 = glb; origin })
  | None, At_least_dim d when is_stage7 stage ->
      Some (Dim_eq { d1 = Var v; d2 = get_default_dim ~d ~proj_id:58 (); origin })
  | None, _ when is_stage7 stage -> Some (Dim_eq { d1 = Var v; d2 = guess_dim (); origin })
  | _ -> None

let%track5_sexp process_shape_row ~(stage : stage) origin env
    ({ beg_dims; dims; bcast; prov } as r : row) : constraint_ list * _ =
  let final = is_stage7 stage in
  let rec finalize_dim_bounds = function
    | Dim _ -> []
    | Affine { over; conv = None; _ } -> finalize_dim_bounds over
    | Affine { over; conv = Some { kernel; _ }; _ } ->
        finalize_dim_bounds over @ finalize_dim_bounds kernel
    | Concat dims -> List.concat_map dims ~f:finalize_dim_bounds
    | Var v -> (
        let guess_dim () =
          if Set.mem env.discardable_vars v then get_default_dim ~d:0 ~proj_id:61 ()
          else get_bcast_dim ~d:1 ~proj_id:62 ()
        in
        match find_dim env.dim_env v with
        | Some (Bounds_dim { is_in_param = true; _ }) when final ->
            raise
            @@ Shape_error
                 ("You forgot to specify the hidden dimension(s) 5", [ Row_mismatch [ r ] ])
        | Some (Bounds_dim { glb; constr; has_uniq_constr_unless; _ })
          when is_stage4_up stage && can_guess_dim_to_one env has_uniq_constr_unless ->
            Option.to_list @@ eliminate_dim_entry stage origin env v ~glb constr
        | Some (Solved_dim _) -> assert false
        | Some (Bounds_dim { has_uniq_constr_unless; _ })
          when final && can_guess_dim_to_one env has_uniq_constr_unless ->
            [ Dim_eq { d1 = Var v; d2 = guess_dim (); origin } ]
        | None when final -> [ Dim_eq { d1 = Var v; d2 = guess_dim (); origin } ]
        | _ -> [])
  in
  let rec has_dim_var = function
    | Dim _ -> false
    | Affine { over; conv = None; _ } -> has_dim_var over
    | Affine { over; conv = Some { kernel; _ }; _ } -> has_dim_var over || has_dim_var kernel
    | Concat dims -> List.exists dims ~f:has_dim_var
    | Var _ -> true
  in
  let process_dims dims = List.concat_map dims ~f:finalize_dim_bounds in
  match bcast with
  | Broadcastable ->
      let keep =
        if (not final) && List.exists (beg_dims @ dims) ~f:has_dim_var then
          [ Shape_row (r, origin) ]
        else []
      in
      (keep @ process_dims beg_dims @ process_dims dims, env)
  | Row_var v -> (
      let dim_eqs = process_dims beg_dims @ process_dims dims in
      let r1 : row = row_of_var v prov in
      let empty_broadcastable : row =
        { beg_dims = []; dims = []; bcast = Broadcastable; prov }
      in
      match find_row env.row_env v with
      | Some (Bounds_row { glb = Some glb; constr = Unconstrained; _ }) when is_stage6_up stage ->
          (Row_eq { r1; r2 = glb; origin } :: dim_eqs, env)
      | Some (Bounds_row { constr = Unconstrained; _ }) when not final ->
          (Shape_row (r, origin) :: dim_eqs, env)
      | Some (Bounds_row { constr = Unconstrained; _ }) when final ->
          (Row_eq { r1; r2 = empty_broadcastable; origin } :: dim_eqs, env)
      | Some
          (Bounds_row
             {
               glb =
                 Some
                   ({
                      beg_dims = [];
                      dims = [] | [ Dim { d = 1; _ } ];
                      bcast = Broadcastable;
                      prov = _;
                    } as glb);
               _;
             }) ->
          (* That's a greatest-lower-bound (most specific) value for a row. *)
          (Row_eq { r1; r2 = glb; origin } :: dim_eqs, env)
      | Some (Bounds_row { glb; constr; _ }) ->
          let ineqs, env =
            try eliminate_row_constraint ~depth:0 stage origin r1 ~terminal:false ~glb constr env
            with Shape_error (s, trace) -> raise @@ Shape_error (s, Row_mismatch [ r1 ] :: trace)
          in
          let keep = if not final then [ Shape_row (r, origin) ] else [] in
          (keep @ ineqs @ dim_eqs, env)
      | Some (Solved_row _) -> assert false
      | _ when final ->
          (Row_eq { r1; r2 = empty_broadcastable; origin } :: dim_eqs, env)
      | _ -> (Shape_row (r, origin) :: dim_eqs, env))

let empty_env =
  {
    dim_env = Utils.Tree_map.empty;
    row_env = Utils.Tree_map.empty;
    discardable_vars = dim_var_set_empty;
  }

let update_dim_is_param d is_p env =
  if not is_p then env
  else
    match d with
    | Var v -> (
        match find_dim env.dim_env v with
        | None ->
            {
              env with
              dim_env =
                add_dim env.dim_env ~key:v
                  ~data:
                    (Bounds_dim
                       {
                         is_in_param = true;
                         has_uniq_constr_unless = None;
                         res = [];
                         opnd = [];
                         glb = None;
                         constr = Unconstrained_dim;
                         origin = [];
                       });
            }
        | Some (Bounds_dim b) ->
            let b = Bounds_dim { b with is_in_param = true } in
            { env with dim_env = add_dim env.dim_env ~key:v ~data:b }
        | _ -> env)
    | _ -> env

let update_row_is_param r is_p env =
  if not is_p then env
  else
    match r with
    | { bcast = Row_var v; _ } -> (
        match find_row env.row_env v with
        | None ->
            {
              env with
              row_env =
                add_row env.row_env ~key:v
                  ~data:
                    (Bounds_row
                       {
                         is_in_param = true;
                         res = [];
                         opnd = [];
                         glb = None;
                         constr = Unconstrained;
                         origin = [];
                       });
            }
        | Some (Bounds_row b) ->
            let b = Bounds_row { b with is_in_param = true } in
            { env with row_env = add_row env.row_env ~key:v ~data:b }
        | _ -> env)
    | _ -> env

let%debug4_sexp solve_inequalities ~(stage : stage)
    ?(discardable_vars : dim_var_set = dim_var_set_empty) (ineqs : constraint_ list) env :
    constraint_ list * _ =
  let env = { env with discardable_vars = Set.union discardable_vars env.discardable_vars } in
  let rec solve ineqs (env : environment) : constraint_ list * _ =
    (* Process a single constraint and return new constraints + updated env *)
    let process_constraint env ineq =
      match ineq with
      | Dim_eq { d1; d2; origin } ->
          let _ineq : constraint_ = ineq in
          (* Substituted inside unify_dim. *)
          unify_dim ~stage origin (d1, d2) env
      | Row_eq { r1; r2; origin } ->
          (* Substituted inside unify_row. *)
          unify_row ~stage origin (r1, r2) env
      | Dim_ineq { res; opnd; origin; _ } ->
          let _ineq : constraint_ = ineq in
          let res = subst_dim env res and opnd = subst_dim env opnd in
          solve_dim_ineq ~stage origin ~res ~opnd env
      | Row_ineq { res; opnd; origin } ->
          let _ineq : constraint_ = ineq in
          let res = subst_row env res and opnd = subst_row env opnd in
          solve_row_ineq ~stage origin ~res ~opnd env
      | Dim_constr { d; constr; origin } ->
          let d = subst_dim env d in
          let extras, constr = apply_dim_constraint ~source:Direct ~stage d constr env in
          let env =
            match (constr, d) with
            | Unconstrained_dim, _ | _, Dim _ | _, Affine _ | _, Concat _ -> env
            | _, Var v ->
                let data =
                  match find_dim env.dim_env v with
                  | Some (Solved_dim _) -> assert false
                  | Some (Bounds_dim bounds) ->
                      let origin = merge_origins origin bounds.origin in
                      Bounds_dim { bounds with constr; origin }
                  | None ->
                      Bounds_dim
                        {
                          is_in_param = false;
                          has_uniq_constr_unless = None;
                          constr;
                          glb = None;
                          res = [];
                          opnd = [];
                          origin;
                        }
                in
                { env with dim_env = add_dim env.dim_env ~key:v ~data }
          in
          (extras, env)
      | Rows_constr { r = rows; constr; origin } ->
          let constr : row_constraint = subst_row_constraint stage env constr in
          reapply_rows_constr := false;
          let substituted_rows = List.map rows ~f:(subst_row env) in
          let (more_ineqs : constraint_ list), env =
            if is_stage3_up stage then
              eliminate_rows_constraint ~depth:0 stage origin ~glb:None substituted_rows constr env
            else apply_rows_constraint ~depth:0 ~stage origin substituted_rows constr env
          in
          (more_ineqs, env)
      | Terminal_dim (is_param, d, origin) ->
          let env = update_dim_is_param d is_param env in
          let more_ineqs = close_dim_terminal ~stage ~is_param origin env @@ subst_dim env d in
          (more_ineqs, env)
      | Terminal_row (is_param, r, origin) ->
          let env = update_row_is_param r is_param env in
          let more_ineqs = close_row_terminal ~stage ~is_param origin env @@ subst_row env r in
          (more_ineqs, env)
      | Shape_row (r, origin) -> process_shape_row ~stage origin env @@ subst_row env r
    in

    (* Fold function that processes constraint, propagates origin, and handles errors *)
    let f (ineqs, env) ineq =
      try
        let more_ineqs, env = process_constraint env ineq in
        (more_ineqs @ ineqs, env)
      with Shape_error (s, trace) ->
        (* let f_out = Out_channel.open_text "shape_error.txt" in
           Stdlib.Printf.fprintf f_out "Shape_error: %s\nenv=\n%s\n%!" s
             (Sexp.to_string_hum @@ [%sexp_of: environment] env);
           Out_channel.close f_out; *)
        (* Add the failing constraint to the error trace *)
        raise @@ Shape_error (s, Constraint_failed ineq :: trace)
    in
    let ineqs', env = List.fold ineqs ~init:([], env) ~f in
    let ineqs' = List.rev ineqs' in
    if
      List.is_empty ineqs'
      || (List.length ineqs' = List.length ineqs && [%equal: constraint_ list] ineqs' ineqs)
    then (ineqs', env)
    else solve ineqs' env
  in
  solve ineqs env

let rec row_to_bases env =
  let rec f = function
    (* Basis is total: return the actual tag (including [default] and [bcast_if_1]) so the
       provenance split is visible to callers. *)
    | Dim { basis = b; _ } -> b
    | Var v -> (
        match find_dim env.dim_env v with
        | None | Some (Bounds_dim _) -> Option.value v.name ~default:""
        | Some (Solved_dim dim) -> f dim)
    | Affine _ -> ""
    | Concat dims -> List.map dims ~f |> String.concat ~sep:"^"
  in
  function
  | { beg_dims; dims; bcast = Row_var v; prov } -> (
      match find_row env.row_env v with
      | None | Some (Bounds_row _) -> Array.of_list_map (beg_dims @ dims) ~f
      | Some (Solved_row { beg_dims = beg_dims2; dims = dims2; bcast; _ }) ->
          row_to_bases env
            {
              beg_dims = beg_dims @ beg_dims2;
              dims = dims2 @ dims;
              bcast;
              prov;
            })
  | { beg_dims; dims; bcast = Broadcastable; prov = _ } ->
      Array.of_list_map (beg_dims @ dims) ~f

(** *** Projection inference *** *)

let fresh_row_proj r =
  let rec fresh_dim = function
    | Dim { d; basis; proj_id = _ } -> Dim { d; basis; proj_id = Some (Proj_id.fresh ()) }
    | Var _ as d -> d
    | Affine { stride; over; conv; stride_offset } ->
        let conv =
          Option.map conv ~f:(fun { dilation; kernel; use_padding } ->
              { dilation; kernel = fresh_dim kernel; use_padding })
        in
        Affine { stride; over = fresh_dim over; conv; stride_offset }
    | Concat dims -> Concat (List.map dims ~f:fresh_dim)
  in
  {
    r with
    beg_dims = List.map r.beg_dims ~f:fresh_dim;
    dims = List.map r.dims ~f:fresh_dim;
  }

let populate_dim_proj_in_solved env =
  let rec fresh_dim = function
    | Dim { d; basis; proj_id = None } -> Dim { d; basis; proj_id = Some (Proj_id.fresh ()) }
    | (Dim _ | Var _) as d -> d
    | Affine { stride; over; conv; stride_offset } ->
        let conv =
          Option.map conv ~f:(fun { dilation; kernel; use_padding } ->
              { dilation; kernel = fresh_dim kernel; use_padding })
        in
        Affine { stride; over = fresh_dim over; conv; stride_offset }
    | Concat dims -> Concat (List.map dims ~f:fresh_dim)
  in
  let fresh_row ({ beg_dims; dims; bcast; prov } : row) : row =
    let beg_dims = List.map beg_dims ~f:fresh_dim in
    let dims = List.map dims ~f:fresh_dim in
    { beg_dims; dims; bcast; prov }
  in
  let f_dim = function Solved_dim dim -> Solved_dim (fresh_dim dim) | entry -> entry in
  let f_row = function Solved_row row -> Solved_row (fresh_row row) | entry -> entry in
  {
    dim_env = Utils.Tree_map.map env.dim_env ~f:f_dim;
    row_env = Utils.Tree_map.map env.row_env ~f:f_row;
    discardable_vars = env.discardable_vars;
  }

(* let update_proj_classes pid1 pid2 proj_classes = Utils.union_add ~equal:Int.equal proj_classes
   pid1 pid2 *)

type convolution_proj = { dilation : int; kernel : proj; kernel_size : int; use_padding : bool }
[@@deriving compare, equal, sexp]

and proj =
  (* TODO: remove this variant Var to see if it breaks anything *)
  | Var of dim_var
  | Proj of proj_id * solved_dim
  | Solved of Idx.axis_index
  | Conv_input of {
      stride : int;
      over : proj;
      conv : convolution_proj option;
      stride_offset : int;
      mutable target_id : proj_id option;
    }
  | Concat of (proj_id * solved_dim) list
[@@deriving compare, equal, sexp]

type error_trace += Projection_mismatch of proj list

let sexp_of_error_trace = function
  | Projection_mismatch ps ->
      Sexp.List (Sexp.Atom "Projection_mismatch" :: List.map ps ~f:sexp_of_proj)
  | error_trace -> sexp_of_error_trace error_trace

type proj_to_index = Idx.axis_index Map.M(Proj_id).t [@@deriving sexp]
type proj_classes = Proj_id.t Map.M(Proj_id).t [@@deriving sexp]

type proj_env = {
  v_env : (dim_var, proj) Hashtbl.t;
  proj_to_index : proj_to_index;
  resolved_padding : axis_padding Map.M(Proj_id).t;
  inferred_padding : axis_padding Hashtbl.M(Proj_id).t;
  proj_classes : proj_classes;
  product_dim : int Map.M(Proj_id).t;
}
[@@deriving sexp_of]

type proj_equation = Proj_eq of proj * proj | Iterated of proj [@@deriving compare, equal, sexp]

let%track4_sexp get_proj_equations (inequalities : constraint_ list) proj_axis_env env :
    proj_equation list =
  (* The difference between to_proj and dim_to_proj is that here we do not have a projection
     environment. *)
  let rec to_proj : dim -> proj = function
    | Var v when Map.mem proj_axis_env v -> Solved (Map.find_exn proj_axis_env v)
    | Dim { d = 0; _ } -> Solved (Fixed_idx 0)
    | Dim ({ proj_id = Some proj_id; _ } as solved_dim) -> Proj (proj_id, solved_dim)
    | Dim { d = 1; basis = _; proj_id = None } ->
        (* d=1 dims created during constraint solving don't need iteration *)
        Solved (Fixed_idx 0)
    | Dim { d; basis; proj_id = None } ->
        raise
        @@ Shape_error
             ( "to_proj: Dim without proj_id (d=" ^ Int.to_string d ^ ", basis="
               ^ basis
               ^ ")",
               [] )
    | Affine { stride; over; conv = None; stride_offset } ->
        (* Strided iteration: no convolution *)
        Conv_input { stride; over = to_proj over; conv = None; stride_offset; target_id = None }
    | Affine { stride; over; conv = Some { dilation; kernel; use_padding }; stride_offset } ->
        let kernel_size =
          match subst_dim ~keep_affine:true env kernel with
          | Var v as dim ->
              raise
              @@ Shape_error
                   ( "projection_of_solved_dims: still not fully inferred for variable "
                     ^ Sexp.to_string_hum ([%sexp_of: dim_var] v),
                     [ Dim_mismatch [ dim ] ] )
          | Dim { d; _ } -> d
          | Affine _ as dim ->
              raise
              @@ Shape_error
                   ("projection_of_solved_dims: still not fully inferred", [ Dim_mismatch [ dim ] ])
          | Concat _ as dim ->
              raise
              @@ Shape_error
                   ( "projection_of_solved_dims: concat not supported as kernel",
                     [ Dim_mismatch [ dim ] ] )
        in
        Conv_input
          {
            stride;
            over = to_proj over;
            conv = Some { dilation; kernel = to_proj kernel; kernel_size; use_padding };
            stride_offset;
            target_id = None;
          }
    | d -> (
        match subst_dim ~keep_affine:true env d with
        | Dim { d = 0; _ } -> Solved (Fixed_idx 0)
        | Dim ({ proj_id = Some proj_id; _ } as solved_dim) -> Proj (proj_id, solved_dim)
        | Dim { d = 1; basis = _; proj_id = None } ->
            (* d=1 dims created during constraint solving don't need iteration *)
            Solved (Fixed_idx 0)
        | Dim { d; basis; proj_id = None } ->
            raise
            @@ Shape_error
                 ( "to_proj (subst): Dim without proj_id (d=" ^ Int.to_string d ^ ", basis="
                   ^ basis
                   ^ ")",
                   [] )
        | Var v when Map.mem proj_axis_env v -> Solved (Map.find_exn proj_axis_env v)
        | Var v -> Var v
        | Affine _ as affine -> to_proj affine
        | Concat dims ->
            let proj_dims =
              List.map dims ~f:(fun d ->
                  match subst_dim ~keep_affine:true env d with
                  | Dim ({ proj_id = Some proj_id; _ } as solved_dim) -> (proj_id, solved_dim)
                  | Dim { d; basis; proj_id = None } ->
                      let proj_id = Proj_id.fresh () in
                      (proj_id, { d; basis; proj_id = Some proj_id })
                  | _ ->
                      raise
                      @@ Shape_error
                           ("to_proj: concat component not fully solved", [ Dim_mismatch [ d ] ]))
            in
            Concat proj_dims)
  in
  let rec expand_dims = function
    | { beg_dims; dims; bcast = Row_var v; _ }
      when Utils.Tree_map.mem ~compare:compare_row_var ~key:v env.row_env -> (
        match find_row env.row_env v with
        | Some (Solved_row r) ->
            let more_dims = expand_dims r in
            beg_dims @ more_dims @ dims
        | _ -> beg_dims @ dims)
    | { beg_dims; dims; _ } -> beg_dims @ dims
  in
  let match_rows ~(with_broadcasting : bool) (r1 : row) (r2 : row) : proj_equation list =
    let dims1 : dim list = expand_dims r1 in
    let dims2 : dim list = expand_dims r2 in
    let len1 = List.length dims1 in
    let len = min len1 (List.length dims2) in
    let extras =
      if with_broadcasting then
        List.map ~f:(fun d -> Iterated (to_proj d)) @@ List.take dims1 (len1 - len)
      else []
    in
    extras
    @ (List.zip_exn (take_from_end dims1 len) (take_from_end dims2 len)
      |> List.map ~f:(fun (d1, d2) -> Proj_eq (to_proj d1, to_proj d2)))
  in
  let f = function
    | Dim_ineq { res = _; opnd = Dim ({ d = 1; proj_id = Some proj_id; _ } as solved_dim); _ } ->
        [ Proj_eq (Proj (proj_id, solved_dim), Solved (Fixed_idx 0)) ]
    | Dim_eq { d1; d2; origin = _ } | Dim_ineq { res = d1; opnd = d2; _ } ->
        [ Proj_eq (to_proj d1, to_proj d2) ]
    | Row_eq { r1; r2; origin = _ } -> match_rows ~with_broadcasting:false r1 r2
    | Row_ineq { res = r1; opnd = r2; origin = _ } ->
        match_rows ~with_broadcasting:true r1 r2
        |> List.concat_map ~f:(function
          | Proj_eq (proj1, (Proj (_, { d = 1; _ }) as proj2)) ->
              [ Iterated proj1; Proj_eq (proj2, Solved (Fixed_idx 0)) ]
          | eq -> [ eq ])
    | Terminal_dim (_is_param, d, _origin) -> [ Iterated (to_proj d) ]
    | Terminal_row (_is_param, r, _origin) ->
        List.map ~f:(fun d -> Iterated (to_proj d)) (expand_dims r)
    | Shape_row _ -> []
    | Rows_constr
        {
          r;
          constr = Total_elems { numerator = Strided_var { coeff; var; denom }; divided_by };
          origin = _;
        } -> (
        let divided_by =
          List.map divided_by ~f:(fun v -> subst_dim ~keep_affine:true env (Var v))
        in
        match collect_factors divided_by with
        | Some (known_product, residual_vars) ->
            assert (List.is_empty residual_vars);
            let coeff = Utils.safe_force coeff in
            let denom = known_product * denom in
            (* TODO: check if this is correct *)
            if coeff % denom = 0 then
              let stride = coeff / denom in
              match rows_to_row_or_vars @@ List.map ~f:(subst_row env) r with
              | Second _ -> assert false
              | Either.First { dims; _ } -> (
                  match List.rev dims with
                  | [] ->
                      [
                        (let output = subst_dim ~keep_affine:true env (Var var) in
                         Iterated (to_proj output));
                      ]
                  | inner :: other_dims ->
                      let output = subst_dim ~keep_affine:true env (Var var) in
                      let input = to_proj inner in
                      Iterated (to_proj output)
                      :: Proj_eq
                           ( to_proj
                               (Affine { stride; over = output; conv = None; stride_offset = 0 }),
                             input )
                      :: List.map other_dims ~f:(fun d -> Proj_eq (to_proj d, Solved Sub_axis)))
            else assert false
        | None -> [])
    | Dim_constr _ | Rows_constr _ -> []
  in
  List.concat_map inequalities ~f

let unknown_projection proj_id d =
  raise
  @@ Shape_error
       ([%string "projection_of_solved_dims: unknown projection: %{proj_id#Proj_id} %{d#Int}"], [])

let%track7_sexp get_proj_index (proj_env : proj_env) (proj : proj) : Idx.axis_index =
  let rec loop (proj : proj) : Idx.axis_index =
    match proj with
    | Proj (proj_id, { d; _ }) -> (
        let repr, _ =
          Utils.union_find ~equal:Proj_id.equal proj_env.proj_classes ~key:proj_id ~rank:0
        in
        match (d, Map.find proj_env.proj_to_index repr) with
        | _, Some i -> i
        | (0 | 1), None -> Idx.Fixed_idx 0
        | _ -> unknown_projection proj_id d)
    | Solved idx -> idx
    | Conv_input { stride; over; conv = None; stride_offset; target_id = _ } -> (
        (* Strided iteration: skip kernel computation since no convolution *)
        let over_idx = loop over in
        let symbols = ref [] in
        let offset = ref stride_offset in

        (* Expand over index - multiply by stride *)
        (match over_idx with
        | Idx.Fixed_idx i -> offset := !offset + (stride * i)
        | Idx.Sub_axis -> ()
        | Idx.Iterator s -> symbols := (stride, s) :: !symbols
        | Idx.Affine { symbols = over_syms; offset = over_offset } ->
            symbols := List.map over_syms ~f:(fun (c, s) -> (stride * c, s)) @ !symbols;
            offset := !offset + (stride * over_offset)
        | Idx.Concat syms -> symbols := List.map syms ~f:(fun s -> (stride, s)) @ !symbols);

        (* Combine and simplify symbols *)
        let symbols =
          !symbols
          |> List.filter ~f:(fun (c, _) -> c <> 0)
          |> List.sort ~compare:(fun (_, s1) (_, s2) -> Idx.compare_symbol s1 s2)
          |> List.group ~break:(fun (_, s1) (_, s2) -> not (Idx.equal_symbol s1 s2))
          |> List.map ~f:(fun group ->
              let s = snd (List.hd_exn group) in
              let coeff = List.sum (module Int) group ~f:fst in
              (coeff, s))
          |> List.filter ~f:(fun (c, _) -> c <> 0)
        in

        match symbols with
        | [] -> Idx.Fixed_idx !offset
        | [ (1, s) ] when !offset = 0 -> Idx.Iterator s
        | _ -> Idx.Affine { symbols; offset = !offset })
    | Conv_input
        {
          stride;
          over;
          conv = Some { dilation; kernel; kernel_size; use_padding };
          stride_offset;
          target_id;
        } -> (
        let over_idx = loop over in
        let kernel_idx = loop kernel in
        let symbols = ref [] in
        let offset = ref stride_offset in

        (* Expand over index - multiply by stride *)
        (match over_idx with
        | Idx.Fixed_idx i -> offset := !offset + (stride * i)
        | Idx.Sub_axis -> ()
        | Idx.Iterator s -> symbols := (stride, s) :: !symbols
        | Idx.Affine { symbols = over_syms; offset = over_offset } ->
            symbols := List.map over_syms ~f:(fun (c, s) -> (stride * c, s)) @ !symbols;
            offset := !offset + (stride * over_offset)
        | Idx.Concat syms -> symbols := List.map syms ~f:(fun s -> (stride, s)) @ !symbols);

        (match kernel_idx with
        | Idx.Fixed_idx i -> offset := !offset + (dilation * i)
        | Idx.Sub_axis -> ()
        | Idx.Iterator s -> symbols := (dilation, s) :: !symbols
        | Idx.Affine { symbols = kernel_syms; offset = kernel_offset } ->
            symbols := List.map kernel_syms ~f:(fun (c, s) -> (dilation * c, s)) @ !symbols;
            offset := !offset + (dilation * kernel_offset)
        | Idx.Concat syms -> symbols := List.map syms ~f:(fun s -> (dilation, s)) @ !symbols);

        (* Subtract padding if use_padding is true *)
        let offset =
          if use_padding then (
            (* Left padding smaller than right when split needed *)
            let right_padding = (kernel_size + 1) / 2 in
            let left_padding = kernel_size - right_padding in
            let operation_padding : axis_padding =
              Ir.Ops.{ left = left_padding; right = right_padding }
            in

            (* Check and update padding based on projection ID from over *)
            (let check_and_update_padding proj_id =
               let repr, _ =
                 Utils.union_find ~equal:Proj_id.equal proj_env.proj_classes ~key:proj_id ~rank:0
               in
               match Map.find proj_env.resolved_padding repr with
               | Some resolved_pad
                 when operation_padding.left > resolved_pad.left
                      || operation_padding.right > resolved_pad.right ->
                   raise
                   @@ Shape_error
                        ( [%string
                            "Operation padding (left=%{operation_padding.left#Int}, \
                             right=%{operation_padding.right#Int}) exceeds resolved padding \
                             (left=%{resolved_pad.left#Int}, right=%{resolved_pad.right#Int})"],
                          [ Projection_mismatch [ proj ] ] )
               | _ -> (
                   (* Update inferred padding to be sufficient for this operation, even if resolved
                      padding is present. *)
                   match Hashtbl.find proj_env.inferred_padding repr with
                   | Some existing_pad
                     when operation_padding.left > existing_pad.left
                          || operation_padding.right > existing_pad.right ->
                       let updated_pad : axis_padding =
                         Ir.Ops.
                           {
                             left = Int.max operation_padding.left existing_pad.left;
                             right = Int.max operation_padding.right existing_pad.right;
                           }
                       in
                       Hashtbl.set proj_env.inferred_padding ~key:repr ~data:updated_pad
                   | None -> Hashtbl.set proj_env.inferred_padding ~key:repr ~data:operation_padding
                   | Some _ -> (* Existing inferred padding is sufficient *) ())
             in
             match target_id with
             | Some proj_id -> check_and_update_padding proj_id
             | None -> () (* No target projection ID available to check *));

            !offset - left_padding)
          else !offset
        in

        (* Combine and simplify symbols *)
        let symbols =
          !symbols
          |> List.filter ~f:(fun (c, _) -> c <> 0)
          |> List.sort ~compare:(fun (_, s1) (_, s2) -> Idx.compare_symbol s1 s2)
          |> List.group ~break:(fun (_, s1) (_, s2) -> not (Idx.equal_symbol s1 s2))
          |> List.map ~f:(fun group ->
              let s = snd (List.hd_exn group) in
              let coeff = List.sum (module Int) group ~f:fst in
              (coeff, s))
          |> List.filter ~f:(fun (c, _) -> c <> 0)
        in

        match symbols with
        | [] -> Idx.Fixed_idx offset
        | [ (1, s) ] when offset = 0 -> Idx.Iterator s
        | _ -> Idx.Affine { symbols; offset })
    | Var v when Hashtbl.mem proj_env.v_env v -> loop (Hashtbl.find_exn proj_env.v_env v)
    | Var v ->
        raise
        @@ Shape_error
             ( "projection_of_solved_dims: still not fully inferred for variable "
               ^ Sexp.to_string_hum ([%sexp_of: dim_var] v),
               [ Projection_mismatch [ proj ] ] )
    | Concat proj_dims ->
        (* For concatenation, collect all component iterators *)
        let syms =
          List.concat_map proj_dims ~f:(fun (proj_id, { d; _ }) ->
              let repr, _ =
                Utils.union_find ~equal:Proj_id.equal proj_env.proj_classes ~key:proj_id ~rank:0
              in
              match (d, Map.find proj_env.proj_to_index repr) with
              | _, Some (Idx.Iterator s) -> [ s ]
              | (0 | 1), _ -> []
              | _, Some (Idx.Fixed_idx _) -> []
              | _, Some (Idx.Affine { symbols; _ }) -> List.map symbols ~f:snd
              | _, Some (Idx.Concat syms) -> syms
              | _, Some Idx.Sub_axis -> []
              | _, None -> [])
        in
        if List.is_empty syms then Idx.Fixed_idx 0 else Idx.Concat syms
  in
  loop proj

let rec dim_to_proj _proj_env : dim -> proj = function
  | Var v -> Var v
  | Dim ({ proj_id = Some proj_id; _ } as solved_dim) -> Proj (proj_id, solved_dim)
  | Dim s -> Proj (Proj_id.fresh (), s)
  | Affine { stride; over; conv = None; stride_offset } ->
      Conv_input
        { stride; over = dim_to_proj _proj_env over; conv = None; stride_offset; target_id = None }
  | Affine { stride; over; conv = Some { dilation; kernel; use_padding }; stride_offset } ->
      (* FIXME: is this sufficient? *)
      let kernel_size = match kernel with Dim { d; _ } -> d | _ -> assert false in
      Conv_input
        {
          stride;
          over = dim_to_proj _proj_env over;
          conv = Some { dilation; kernel = dim_to_proj _proj_env kernel; kernel_size; use_padding };
          stride_offset;
          target_id = None;
        }
  | Concat dims ->
      let proj_dims =
        List.map dims ~f:(fun d ->
            match d with
            | Dim ({ proj_id = Some proj_id; _ } as solved_dim) -> (proj_id, solved_dim)
            | Dim { d; basis; proj_id = None } ->
                let proj_id = Proj_id.fresh () in
                (proj_id, { d; basis; proj_id = Some proj_id })
            | _ -> failwith "dim_to_proj: Concat component not a solved dim")
      in
      Concat proj_dims

let get_dim_index proj_env =
  let loop = function
    | Dim { d; _ } when not @@ Idx.iterated d -> Idx.Fixed_idx 0
    | Dim { proj_id = None; _ } -> assert false
    | Var v when Hashtbl.mem proj_env.v_env v ->
        get_proj_index proj_env @@ Hashtbl.find_exn proj_env.v_env v
    | Var v as dim ->
        raise
        @@ Shape_error
             ( "projection_of_solved_dims: still not fully inferred for variable "
               ^ Sexp.to_string_hum ([%sexp_of: dim_var] v),
               [ Dim_mismatch [ dim ] ] )
    | Dim { proj_id = Some proj_id; d; _ } -> (
        let repr, _ =
          Utils.union_find ~equal:Proj_id.equal proj_env.proj_classes ~key:proj_id ~rank:0
        in
        match (d, Map.find proj_env.proj_to_index repr) with
        | _, Some i -> i
        | (0 | 1), None -> Fixed_idx 0
        | _ -> unknown_projection proj_id d)
    | Affine _ as dim -> get_proj_index proj_env (dim_to_proj proj_env dim)
    | Concat _ as dim -> get_proj_index proj_env (dim_to_proj proj_env dim)
  in
  loop

let%debug4_sexp solve_proj_equations (eqs : proj_equation list)
    ~(resolved_padding : (proj_id, axis_padding) List.Assoc.t)
    ~(inferred_padding : (proj_id, axis_padding) List.Assoc.t) : proj_env =
  let v_env = dim_hashtbl () in
  let p_solved = ref [] in
  let p_conv_input = ref [] in
  let verify_when_solved1 = ref [] in
  let verify_when_solved2 = ref [] in
  let p_dims = ref [] in
  let iterated_vars = ref [] in
  let p_concat_targets = ref [] in
  (* (target_pid, component proj_dims) for deferred Concat handling *)
  let p_concat_components = ref @@ Set.empty (module Proj_id) in
  (* proj_ids that are Concat components *)
  let proj_classes = ref @@ Map.empty (module Proj_id) in
  let rec loop (eq : proj_equation) : unit =
    match eq with
    | Proj_eq (Proj (p1, { d; _ }), Proj (p2, _)) when Proj_id.equal p1 p2 ->
        p_dims := (p1, d) :: !p_dims
    | Proj_eq (Var v1, Var v2) when equal_dim_var v1 v2 -> ()
    | Proj_eq ((Proj (p1, { d = d1; _ }) as proj1), (Proj (p2, { d = d2; _ }) as proj2)) ->
        if d1 <> d2 then
          raise
          @@ Shape_error
               ( "Conflicting dimensions for the same projection",
                 [ Projection_mismatch [ proj1; proj2 ] ] );
        p_dims := (p1, d1) :: !p_dims;
        proj_classes := Utils.union_add ~equal:Proj_id.equal !proj_classes p1 p2
    | Proj_eq (Proj (p, _), Solved idx) | Proj_eq (Solved idx, Proj (p, _)) ->
        p_solved := (p, idx) :: !p_solved
    | Proj_eq (Proj (p, _), (Conv_input c as conv_input))
    | Proj_eq ((Conv_input c as conv_input), Proj (p, _)) ->
        (match c.target_id with
        | Some pid when Proj_id.equal p pid -> ()
        | Some pid -> proj_classes := Utils.union_add ~equal:Proj_id.equal !proj_classes pid p
        | None -> c.target_id <- Some p);
        (* We will substitute variables in conv_input later *)
        p_conv_input := (p, conv_input) :: !p_conv_input
    | Proj_eq (Solved idx, ((Conv_input _ | Concat _) as conv_input))
    | Proj_eq (((Conv_input _ | Concat _) as conv_input), Solved idx) ->
        verify_when_solved1 := (idx, conv_input) :: !verify_when_solved1
    | Proj_eq
        ( (Conv_input { stride = stride1; over = over1; _ } as conv_input1),
          (Conv_input { stride = stride2; over = over2; _ } as conv_input2) )
      when stride1 = stride2 ->
        loop (Proj_eq (over1, over2));
        if equal_proj conv_input1 conv_input2 then ()
        else verify_when_solved2 := (conv_input1, conv_input2) :: !verify_when_solved2
    | Proj_eq ((Conv_input _ as conv_input1), ((Conv_input _ | Concat _) as conv_input2))
    | Proj_eq ((Concat _ as conv_input1), (Conv_input _ as conv_input2)) ->
        (* Conv_input vs Conv_input/Concat - defer verification *)
        if equal_proj conv_input1 conv_input2 then ()
        else verify_when_solved2 := (conv_input1, conv_input2) :: !verify_when_solved2
    | Proj_eq (Solved idx1, Solved idx2) when Idx.equal_axis_index idx1 idx2 -> ()
    | Proj_eq (Solved idx1, Solved idx2) ->
        raise
        @@ Shape_error
             ("Conflicting indices for the same axis/projection", [ Index_mismatch [ idx1; idx2 ] ])
    | Proj_eq (Var v1, Var v2) when equal_dim_var v1 v2 -> ()
    | Proj_eq (Var v, p) | Proj_eq (p, Var v) -> (
        match Hashtbl.find v_env v with
        | None -> Hashtbl.add_exn v_env ~key:v ~data:p
        | Some p2 -> loop (Proj_eq (p, p2)))
    | Iterated (Solved _) -> ()
    | Iterated (Proj (pid, { d; _ })) -> p_dims := (pid, d) :: !p_dims
    | Iterated (Conv_input { over; conv = None; _ }) ->
        (* Strided iteration: only process over since no convolution *)
        loop (Iterated over)
    | Iterated (Conv_input { over; conv = Some { kernel; _ }; _ }) ->
        loop (Iterated over);
        loop (Iterated kernel)
    | Iterated (Var v) -> (
        match Hashtbl.find v_env v with
        | None ->
            (* Defer: record that v needs iteration, will resolve after all equations processed *)
            iterated_vars := v :: !iterated_vars
        | Some proj -> loop @@ Iterated proj)
    | Iterated (Concat proj_dims) ->
        (* Each component of a concat needs iteration, including d=1 *)
        List.iter proj_dims ~f:(fun (pid, { d; _ }) ->
            p_dims := (pid, d) :: !p_dims;
            p_concat_components := Set.add !p_concat_components pid)
    | Proj_eq (Concat proj_dims1, Concat proj_dims2)
      when List.length proj_dims1 = List.length proj_dims2 ->
        (* Pairwise unification - projections must match structurally *)
        List.iter2_exn proj_dims1 proj_dims2 ~f:(fun (p1, s1) (p2, s2) ->
            p_concat_components := Set.add !p_concat_components p1;
            p_concat_components := Set.add !p_concat_components p2;
            loop (Proj_eq (Proj (p1, s1), Proj (p2, s2))))
    | Proj_eq (Concat proj_dims1, Concat proj_dims2) ->
        (* Mismatched Concat lengths - record all components for iteration *)
        List.iter proj_dims1 ~f:(fun (pid, { d; _ }) ->
            p_dims := (pid, d) :: !p_dims;
            p_concat_components := Set.add !p_concat_components pid);
        List.iter proj_dims2 ~f:(fun (pid, { d; _ }) ->
            p_dims := (pid, d) :: !p_dims;
            p_concat_components := Set.add !p_concat_components pid)
    | Proj_eq (Concat proj_dims, Proj (target_pid, _))
    | Proj_eq (Proj (target_pid, _), Concat proj_dims) ->
        (* Record components for iteration, including d=1 *)
        List.iter proj_dims ~f:(fun (pid, { d; _ }) ->
            p_dims := (pid, d) :: !p_dims;
            p_concat_components := Set.add !p_concat_components pid);
        (* Defer: target will get Concat index after components have iterators *)
        p_concat_targets := (target_pid, proj_dims) :: !p_concat_targets
  in
  let no_proj_assigned v =
    raise
    @@ Shape_error
         ( "Iterated variable has no projection assigned: "
           ^ Sexp.to_string_hum ([%sexp_of: dim_var] v),
           [] )
  in
  List.iter eqs ~f:loop;
  (* Process deferred iterated variables: they should now have projections assigned *)
  let pending_vars = !iterated_vars in
  iterated_vars := [];
  List.iter pending_vars ~f:(fun v ->
      match Hashtbl.find v_env v with
      | None -> no_proj_assigned v
      | Some proj -> loop @@ Iterated proj);
  (* Any variables added during the above iteration have no valid projection chain *)
  List.iter !iterated_vars ~f:no_proj_assigned;
  let projs = ref @@ Map.empty (module Proj_id) in
  let concat_reprs =
    Set.of_list
      (module Proj_id)
      (Set.to_list !p_concat_components
      |> List.map ~f:(fun p ->
          fst @@ Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0))
  in
  List.iter !p_solved ~f:(fun (p, idx) ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      if Idx.equal_axis_index idx (Idx.Fixed_idx 0) && Set.mem concat_reprs repr then ()
      else
        Utils.mref_add projs ~key:repr ~data:idx ~or_:(fun idx2 ->
            if not @@ Idx.equal_axis_index idx idx2 then
              raise
              @@ Shape_error
                   ("Multiple constraints on the same projection", [ Index_mismatch [ idx; idx2 ] ])));
  let product_dim = ref @@ Map.empty (module Proj_id) in
  let concat_dims =
    List.filter_map !p_dims ~f:(fun (p, d) ->
        if Set.mem !p_concat_components p then Some (p, d) else None)
  in
  (* Collect projection IDs that will get their index from Conv_input (target_id projections). These
     should NOT get fresh iterators from product_dim processing. *)
  let conv_input_targets =
    Set.of_list (module Proj_id)
    @@ List.filter_map !p_conv_input ~f:(fun (p, _) ->
        let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
        Some repr)
  in
  List.iter !p_dims ~f:(fun (p, d) ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      (* Include d=1 if it's a concat component - they need iterators for proper Concat indexing *)
      let needs_iterator = Idx.iterated d || (d = 1 && Set.mem !p_concat_components p) in
      if needs_iterator && (not @@ Map.mem !projs repr) then
        Utils.mref_add product_dim ~key:repr ~data:d ~or_:(fun d2 ->
            (* TODO: consider updating padding *)
            if d <> d2 then
              raise
              @@ Shape_error
                   ( [%string
                       "Conflicting dimensions for the same projection: %{p#Proj_id} %{d#Int} \
                        %{d2#Int}"],
                     [] )));
  List.iter concat_dims ~f:(fun (p, d) ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      if not @@ Map.mem !product_dim repr then product_dim := Map.set !product_dim ~key:repr ~data:d);
  (* Create fresh iterators for product dimensions, EXCEPT for those that will get their index from
     Conv_input (they will be processed later). *)
  Map.iteri !product_dim ~f:(fun ~key:p ~data:_ ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      if not (Set.mem conv_input_targets repr) then
        Utils.mref_add_missing projs repr ~f:(fun () -> Idx.(Iterator (get_symbol ()))));
  Set.iter concat_reprs ~f:(fun repr ->
      if Set.mem conv_input_targets repr then ()
      else
        match Map.find !projs repr with
        | Some (Idx.Iterator _) -> ()
        | _ -> projs := Map.set !projs ~key:repr ~data:Idx.(Iterator (get_symbol ())));

  (* Process p_conv_input to populate projs and compute padding *)
  let resolved_padding =
    Map.of_alist_exn (module Proj_id)
    @@ List.map resolved_padding ~f:(fun (p, pad) ->
        (fst @@ Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0, pad))
  in
  let inferred_padding =
    Hashtbl.of_alist_exn (module Proj_id)
    @@ List.map inferred_padding ~f:(fun (p, pad) ->
        (fst @@ Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0, pad))
  in

  let temp_proj_env =
    {
      v_env;
      proj_classes = !proj_classes;
      proj_to_index = !projs;
      inferred_padding;
      resolved_padding;
      product_dim = !product_dim;
    }
  in
  (* Process postponed Conv_input equations *)
  List.iter !p_conv_input ~f:(fun (p, conv_input) ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      let idx = get_proj_index temp_proj_env conv_input in
      Utils.mref_add projs ~key:repr ~data:idx ~or_:(fun idx2 ->
          if not @@ Idx.equal_axis_index idx idx2 then
            raise
            @@ Shape_error
                 ( "Multiple constraints on the same Conv_input projection",
                   [ Index_mismatch [ idx; idx2 ] ] )));

  (* Verify postponed equations *)
  List.iter !verify_when_solved1 ~f:(fun (idx, conv_input) ->
      try
        let conv_idx = get_proj_index temp_proj_env conv_input in
        if not @@ Idx.equal_axis_index idx conv_idx then
          raise
          @@ Shape_error
               ( "Cannot unify index with Conv_input projection",
                 [ Index_mismatch [ idx; conv_idx ] ] )
      with _ -> () (* Ignore errors for now *));

  List.iter !verify_when_solved2 ~f:(fun (conv_input1, conv_input2) ->
      try
        let idx1 = get_proj_index temp_proj_env conv_input1 in
        let idx2 = get_proj_index temp_proj_env conv_input2 in
        if not @@ Idx.equal_axis_index idx1 idx2 then
          raise
          @@ Shape_error
               ("Cannot unify two Conv_input projections", [ Index_mismatch [ idx1; idx2 ] ])
      with _ -> () (* Ignore errors for now *));

  (* Now create fresh iterators for product dimensions that still don't have an index. This is done
     after p_conv_input processing so Conv_input projections don't conflict. *)
  Map.iteri !product_dim ~f:(fun ~key:p ~data:_ ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      Utils.mref_add_missing projs repr ~f:(fun () -> Idx.(Iterator (get_symbol ()))));

  (* Process deferred concat targets: build Concat indices from component iterators *)
  List.iter !p_concat_targets ~f:(fun (target_pid, proj_dims) ->
      let target_repr, _ =
        Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:target_pid ~rank:0
      in
      let syms =
        List.filter_map proj_dims ~f:(fun (pid, { d; _ }) ->
            let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:pid ~rank:0 in
            match Map.find !projs repr with
            | Some (Idx.Iterator s) -> Some s
            | Some (Idx.Fixed_idx 0) when d = 0 -> None (* d=0 is invalid dimension, skip *)
            | _ when d = 0 -> None
            | _ ->
                raise
                @@ Shape_error
                     ( [%string
                         "Concat component projection %{pid#Proj_id} (d=%{d#Int}) has no iterator"],
                       [] ))
      in
      let expected_idx = if List.is_empty syms then Idx.Fixed_idx 0 else Idx.Concat syms in
      match Map.find !projs target_repr with
      | None -> projs := Map.set !projs ~key:target_repr ~data:expected_idx
      | Some existing_idx ->
          let ok =
            Idx.equal_axis_index existing_idx expected_idx
            ||
            match (existing_idx, expected_idx) with
            | Idx.Iterator s, Idx.Concat [ s' ] when Idx.equal_symbol s s' -> true
            | _ -> false
          in
          if not ok then
            raise
            @@ Shape_error
                 ( [%string
                     "Concat target projection %{target_pid#Proj_id} conflicts with existing index"],
                   [ Index_mismatch [ existing_idx; expected_idx ] ] ));

  {
    v_env;
    proj_classes = !proj_classes;
    proj_to_index = !projs;
    inferred_padding;
    resolved_padding;
    product_dim = !product_dim;
  }

let proj_repr proj_env p =
  fst @@ Utils.union_find ~equal:Proj_id.equal proj_env.proj_classes ~key:p ~rank:0

let get_product_proj proj_env dim =
  match dim with
  | Dim { proj_id = Some proj_id; d; _ } -> (
      let repr = proj_repr proj_env proj_id in
      if not (Map.mem proj_env.product_dim repr) then None
      else
        match Map.find proj_env.proj_to_index repr with
        | Some (Iterator _) -> Some (repr, d)
        | _ -> None)
  | Dim { proj_id = None; _ } -> None
  | Var v ->
      raise
      @@ Shape_error
           ( "projection_of_solved_dims: still not fully inferred for variable "
             ^ Sexp.to_string_hum ([%sexp_of: dim_var] v),
             [ Dim_mismatch [ dim ] ] )
  | Affine _ -> None
  | Concat _ -> None

let%debug6_sexp get_dim_padding (proj_env : proj_env) (dim : dim) : axis_padding option =
  match dim with
  | Dim { proj_id = Some proj_id; _ } ->
      let repr = proj_repr proj_env proj_id in
      Hashtbl.find proj_env.inferred_padding repr
  | _ -> assert false

let proj_to_iterator_exn proj_env p =
  match Map.find_exn proj_env.proj_to_index (proj_repr proj_env p) with
  | Iterator s -> s
  | _ -> invalid_arg "proj_to_iterator_exn"

let product_dim_iterators proj_env =
  Map.to_alist proj_env.product_dim
  |> List.filter_map ~f:(fun (p, d) ->
      match Map.find proj_env.proj_to_index p with
      | Some (Idx.Iterator s) -> Some (p, d, s)
      | _ -> None)

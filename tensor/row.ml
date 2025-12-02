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

type solved_dim = { d : int; label : string option; proj_id : proj_id option }
[@@deriving equal, hash, compare, sexp]

type convolution = { dilation : int; kernel : dim; use_padding : bool }
[@@deriving equal, hash, compare, sexp]

and dim =
  | Var of dim_var
  | Dim of solved_dim
  | Affine of { stride : int; over : dim; conv : convolution option; stride_offset : int }
[@@deriving equal, hash, compare, sexp]

let equal_dim d1 d2 =
  match (d1, d2) with
  | Dim { d = d1; label = l1; proj_id = _ }, Dim { d = d2; label = l2; proj_id = _ } ->
      d1 = d2 && Option.equal String.equal l1 l2
  | _ -> equal_dim d1 d2

let uid = ref 0

let get_var ?name () : dim_var =
  Int.incr uid;
  { id = !uid; name }

let get_dim ~d ?label ?proj_id () =
  let proj_id = Option.map ~f:(fun p -> Proj_id.Proj_id p) proj_id in
  Dim { d; label; proj_id }

type 'a dim_hashtbl = 'a Hashtbl.M(Dim_var).t [@@deriving sexp]

let dim_hashtbl () = Hashtbl.create (module Dim_var)

type print_style = Only_labels | Axis_size | Axis_number_and_size | Projection_and_size
[@@deriving equal, compare, sexp]

let solved_dim_to_string style { d; label; proj_id } =
  match style with
  | Only_labels -> ( match label with None -> "_" | Some l -> l)
  | Axis_size | Axis_number_and_size -> (
      let label_prefix = match label with None -> "" | Some l -> l ^ "=" in
      match proj_id with
      | None -> label_prefix ^ Int.to_string d
      | Some _ -> label_prefix ^ Int.to_string d)
  | Projection_and_size ->
      let label_part = match label with None -> "" | Some l -> l ^ "=" in
      let size_part = Int.to_string d in
      let proj_part = match proj_id with None -> "" | Some pid -> "p" ^ Proj_id.to_string pid in
      label_part ^ size_part ^ proj_part

let rec dim_to_string style = function
  | Dim { label = None; _ } when equal_print_style style Only_labels -> "_"
  | Dim { label = Some l; _ } when equal_print_style style Only_labels -> l
  | Dim { d; label = None; proj_id = None } when equal_print_style style Axis_size ->
      Int.to_string d
  | Dim { d; label = Some l; proj_id = None } when equal_print_style style Axis_size ->
      [%string "%{l}=%{d#Int}"]
  | Dim { d; label = None; proj_id = None } when equal_print_style style Axis_size ->
      Int.to_string d
  | Dim { d; label = Some l; proj_id = None } when equal_print_style style Axis_size ->
      [%string "%{l}=%{d#Int}"]
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

type bcast = Row_var of { v : row_var; beg_dims : dim list } | Broadcastable
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

type t = { dims : dim list; bcast : bcast; prov : provenance }
[@@deriving equal, hash, compare, sexp]

type row = t [@@deriving equal, sexp]

let get_row_for_var prov v = { dims = []; bcast = Row_var { v; beg_dims = [] }; prov }
let row_shapes row = provenance_shapes row.prov

let dims_label_assoc dims =
  let f = function Var { name = Some n; _ } as d -> Some (n, d) | _ -> None in
  List.filter_map dims.dims ~f

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

(** An entry implements inequalities [cur >= v >= subr] and/or an equality [v = solved]. [cur] and
    [subr] must be sorted using the [@@deriving compare] comparison. *)
type dim_entry =
  | Solved_dim of dim
  | Bounds_dim of {
      is_in_param : bool;
      has_uniq_constr_unless : dim_var_set option;
      cur : dim_var list;
      subr : dim_var list;
      lub : dim option;
      constr : dim_constraint;
      origin : constraint_origin list;
    }
[@@deriving sexp_of]

type row_entry =
  | Solved_row of t
  | Bounds_row of {
      is_in_param : bool;
      cur : row_var list;
      subr : row_var list;
      lub : t option;
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
  | Bounds_dim { is_in_param; has_uniq_constr_unless; cur; subr; lub; constr; origin = _ } ->
      Bounds_dim { is_in_param; has_uniq_constr_unless; cur; subr; lub; constr; origin = [] }

type row_env = (row_var, row_entry) Utils.Tree_map.t

let sexp_of_row_env env = Utils.Tree_map.sexp_of_t sexp_of_row_var sexp_of_row_entry env
let find_row env var = Utils.Tree_map.find ~compare:compare_row_var ~key:var env
let add_row env ~key ~data = Utils.Tree_map.add ~compare:compare_row_var ~key ~data env

type environment = { dim_env : dim_env; row_env : row_env } [@@deriving sexp_of]
(** The environment is only in resolved wrt. variables that are solved: [v -> Solved ...] do not
    appear elsewhere in the environment. In particular, per-dim and per-row constraints might not
    have been applied. *)

let get_dim_val env var =
  match find_dim env.dim_env var with Some (Solved_dim (Dim { d; _ })) -> Some d | _ -> None

let get_row_from_env env var =
  match find_row env.row_env var with Some (Solved_row row) -> Some row | _ -> None

type constraint_ =
  | Dim_eq of { d1 : dim; d2 : dim; origin : constraint_origin list }
  | Row_eq of { r1 : t; r2 : t; origin : constraint_origin list }
  | Dim_ineq of { cur : dim; subr : dim; from_ : Sexp.t; origin : constraint_origin list }
  | Row_ineq of { cur : t; subr : t; origin : constraint_origin list }
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
  | Dim_ineq { cur; subr; from_; origin = _ } -> Dim_ineq { cur; subr; from_; origin = [] }
  | Row_ineq { cur; subr; origin = _ } -> Row_ineq { cur; subr; origin = [] }
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

type source = Direct | Equation | Cur | Subr [@@deriving equal, sexp]

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

let dim_to_int_exn = function
  | Dim { d; _ } -> d
  | Var _ -> invalid_arg "dim_to_int: dim still unknown"
  | Affine _ -> invalid_arg "dim_to_int: affine dimension cannot be converted to single int"

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
          let extent = dilation * (k.d - 1) in
          Dim
            {
              d = (s.d * stride) + extent;
              label = Option.first_some s.label k.label;
              proj_id = None;
            }
      | Affine
          {
            stride;
            over = Dim s;
            conv = Some { kernel = Dim k; use_padding = true; _ };
            stride_offset = _;
          } ->
          Dim { d = s.d * stride; label = Option.first_some s.label k.label; proj_id = None }
      | Affine { stride; over = Dim s; conv = None; stride_offset = _ } ->
          Dim { d = s.d * stride; label = s.label; proj_id = None }
      | res -> res)
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
              ( [ Dim_eq { d1 = Var var; d2 = get_dim ~d:(n * denom / coeff_val) (); origin } ],
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
                [ Dim_eq { d1 = Var v; d2 = get_dim ~d:quotient (); origin } ])
          else if quotient <= 0 && Option.is_none num_var then elems_mismatch n1 n2
          else if quotient = 1 && Option.is_none num_var then
            (* The difference variables must all be 1 *)
            List.map diff_vars ~f:(fun v ->
                Dim_eq { d1 = Var v; d2 = get_dim ~d:1 ~proj_id:42 (); origin })
          else
            (* The product of difference variables equals the quotient *)
            let r =
              { dims = List.map diff_vars ~f:(fun v -> Var v); bcast = Broadcastable; prov }
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
        let r = { dims = List.map diff_vars ~f:(fun v -> Var v); bcast = Broadcastable; prov } in
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
                     { dims = dims1; bcast = Broadcastable; prov };
                     { dims = dims2; bcast = Broadcastable; prov };
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
                        Dim_eq { d1 = Var v; d2 = get_dim ~d:1 ~proj_id:43 (); origin })
                  in
                  let exact_vars_eqs =
                    List.map vars ~f:(fun v ->
                        Dim_eq { d1 = Var v; d2 = get_dim ~d:1 ~proj_id:44 (); origin })
                  in
                  Some (divided_by_eqs @ exact_vars_eqs, Exact dims)
                else if List.is_empty divided_by && List.length vars = 1 && reminder > 0 then
                  (* divided_by is empty and there is only one dim variable in Exact dims *)
                  let v = List.hd_exn vars in
                  Some ([ Dim_eq { d1 = Var v; d2 = get_dim ~d:reminder (); origin } ], Exact dims)
                else if List.is_empty vars && List.length divided_by = 1 && reminder > 0 then
                  (* Exact dims contain only known dimensions and divided_by has exactly one
                     variable *)
                  let v = List.hd_exn divided_by in
                  Some ([ Dim_eq { d1 = Var v; d2 = get_dim ~d:reminder (); origin } ], Exact dims)
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
                  Some ([ Dim_eq { d1 = Var var; d2 = get_dim ~d (); origin } ], Exact dims)
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
            (* If source is [Cur], then [constr] (target) is [Subr]. *)
            | Cur, (Unconstrained_dim | At_least_dim 1) -> ([], constr)
            | _ -> Option.value ~default:([], constr) @@ dim_conjunction constr bounds.constr))
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
                cur = [];
                subr = [];
                lub = None;
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
          | Affine _ -> failwith "NOT IMPLEMENTED YET")
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
            (* Regular row, add its dims and continue *)
            collect_info (List.rev_append row.dims before_dims) row_vars remaining_rows
        | Row_var { v; beg_dims } ->
            (* Row variable - collect it and continue *)
            let new_before_dims = List.rev_append row.dims (List.rev_append beg_dims before_dims) in
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
      Either.First { dims = all_dims; bcast = Broadcastable; prov = first_prov }
  | [ (v, prov) ] ->
      (* Exactly one row variable - reconstruct the proper row structure *)
      let rec reconstruct_single_var before_dims rows =
        match rows with
        | [] -> failwith "rows_to_row_or_vars: single row variable not found during reconstruction"
        | row :: remaining_rows -> (
            match row.bcast with
            | Broadcastable ->
                reconstruct_single_var (List.rev_append row.dims before_dims) remaining_rows
            | Row_var { v = found_v; beg_dims } when equal_row_var found_v v ->
                let new_beg_dims = List.rev_append before_dims beg_dims in
                let after_dims = List.concat_map remaining_rows ~f:(fun r -> r.dims) in
                let new_dims = row.dims @ after_dims in
                { dims = new_dims; bcast = Row_var { v; beg_dims = new_beg_dims }; prov }
            | Row_var _ -> reconstruct_single_var before_dims remaining_rows)
      in
      Either.First (reconstruct_single_var [] rows)
  | _ ->
      (* Multiple row variables *)
      Either.Second (all_dims, row_vars)

let row_of_var v prov = { dims = []; bcast = Row_var { v; beg_dims = [] }; prov }

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
  | Broadcastable -> []
  | Row_var { v; beg_dims } ->
      if List.is_empty beg_dims then
        [
          Row_eq
            {
              r1 = row_of_var v r.prov;
              r2 = { dims = []; bcast = Broadcastable; prov = r.prov };
              origin;
            };
        ]
      else raise @@ Shape_error ("check_empty_row: row is not empty", [ Row_mismatch [ r ] ])

let s_dim_one_in_entry v ~value (in_ : dim_entry) : _ * dim_entry =
  let from_ = [%sexp_of: dim_var * dim_entry] (v, drop_origin in_) in
  match in_ with
  | Solved_dim in_ -> ([], Solved_dim (s_dim_one v ~value ~in_))
  | Bounds_dim { is_in_param; has_uniq_constr_unless; cur; subr; lub; constr; origin } ->
      let find_v side = List.partition_tf side ~f:(equal_dim_var v) in
      let cur_v, cur = find_v cur in
      let subr_v, subr = find_v subr in
      let ineqs0 =
        match (subr_v, lub) with
        | _ :: _, Some lub -> [ Dim_ineq { cur = lub; subr = value; from_; origin } ]
        | _ -> []
      in
      let ineqs1 =
        if List.is_empty subr_v then []
        else List.map cur ~f:(fun cur -> Dim_ineq { cur = Var cur; subr = value; from_; origin })
      in
      let ineqs2 =
        if List.is_empty cur_v then []
        else List.map subr ~f:(fun subr -> Dim_ineq { cur = value; subr = Var subr; from_; origin })
      in
      ( ineqs0 @ ineqs1 @ ineqs2,
        Bounds_dim
          {
            is_in_param;
            has_uniq_constr_unless;
            cur;
            subr;
            lub = Option.map lub ~f:(fun in_ -> s_dim_one v ~value ~in_);
            constr;
            origin;
          } )

let s_dim_one_in_row v ~value in_ =
  { in_ with dims = List.map in_.dims ~f:(fun in_ -> s_dim_one v ~value ~in_) }

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
    | Bounds_row { is_in_param; cur; subr; lub; constr; origin } ->
        let constr = s_dim_one_in_row_constr stage v ~value constr in
        if !reapply_rows_constr then
          ineqs_from_reapply_rows_constr :=
            Rows_constr { r = [ row_of_var key [] ]; constr; origin }
            :: !ineqs_from_reapply_rows_constr;
        reapply_rows_constr := false;
        let lub = Option.map lub ~f:(s_dim_one_in_row v ~value) in
        Bounds_row { is_in_param; cur; subr; lub; constr; origin }
  in
  result

let rec vars_of_dim = function
  | Dim _ -> Set.empty (module Dim_var)
  | Var v -> Set.singleton (module Dim_var) v
  | Affine { over; conv = None; _ } -> vars_of_dim over
  | Affine { over; conv = Some { kernel; _ }; _ } ->
      Set.union (vars_of_dim over) (vars_of_dim kernel)

let subst_dim ?(keep_affine = false) env dim =
  let vars = vars_of_dim dim in
  List.fold (Set.elements vars) ~init:dim ~f:(fun acc v ->
      match find_dim env.dim_env v with
      | Some (Solved_dim d) -> s_dim_one ~keep_affine v ~value:d ~in_:acc
      | _ -> acc)

let s_row_one v ~value:{ dims = more_dims; bcast; prov = _ } ~in_ =
  match in_ with
  | { dims; bcast = Row_var { v = v2; beg_dims }; prov } when equal_row_var v v2 -> (
      match bcast with
      | Broadcastable -> { dims = beg_dims @ more_dims @ dims; bcast; prov }
      | Row_var { v = v3; beg_dims = more_beg_dims } ->
          {
            dims = more_dims @ dims;
            bcast = Row_var { v = v3; beg_dims = beg_dims @ more_beg_dims };
            prov;
          })
  | _ -> in_

let s_row_one_in_row_constr _v ~value:_ ~in_ =
  match in_ with Unconstrained | Total_elems _ | Exact _ -> in_

let s_row_one_in_entry (v : row_var) ~(value : row) ~(in_ : row_entry) :
    constraint_ list * row_entry =
  match in_ with
  | Solved_row in_ -> ([], Solved_row (s_row_one v ~value ~in_))
  | Bounds_row { is_in_param; cur; subr; lub; constr; origin } ->
      (* TODO: audit code to ensure we don't lose the constraints associated with the bounds
         variables. *)
      let find_v side = List.partition_tf side ~f:(equal_row_var v) in
      let cur_v, cur = find_v cur in
      let subr_v, subr = find_v subr in
      let ineqs0 =
        match (subr_v, lub) with
        | _ :: _, Some lub -> [ Row_ineq { cur = lub; subr = value; origin } ]
        | _ -> []
      in
      let ineqs1 =
        if List.is_empty subr_v then []
        else
          List.map cur ~f:(fun cur ->
              Row_ineq { cur = row_of_var cur value.prov; subr = value; origin })
      in
      let ineqs2 =
        if List.is_empty cur_v then []
        else
          List.map subr ~f:(fun subr ->
              Row_ineq { subr = row_of_var subr value.prov; cur = value; origin })
      in
      let constr = s_row_one_in_row_constr v ~value ~in_:constr in
      let lub = Option.map lub ~f:(fun in_ -> s_row_one v ~value ~in_) in
      (ineqs0 @ ineqs1 @ ineqs2, Bounds_row { is_in_param; cur; subr; lub; constr; origin })

let subst_row env ({ dims; bcast; prov } : t) : t =
  let s_dims = List.map ~f:(subst_dim env) in
  let dims = s_dims dims in
  let bcast =
    match bcast with
    | Row_var { v; beg_dims } -> Row_var { v; beg_dims = s_dims beg_dims }
    | Broadcastable -> Broadcastable
  in
  let default = { dims; bcast; prov } in
  match bcast with
  | Broadcastable -> default
  | Row_var { v; beg_dims } -> (
      match find_row env.row_env v with
      | None | Some (Bounds_row _) -> default
      | Some (Solved_row { dims = []; bcast = Row_var { v = v2; beg_dims = [] }; _ })
        when equal_row_var v v2 ->
          default
      | Some (Solved_row ({ bcast = Row_var { v = v2; _ }; _ } as r2)) when equal_row_var v v2 ->
          raise
          @@ Shape_error
               ("Infinite number of axes by self-reference", [ Row_mismatch [ default; r2 ] ])
      | Some (Solved_row { dims = more_dims; bcast; prov = _ }) -> (
          (* Note: we assume env is idempotent (solved wrt. equalities). *)
          match bcast with
          | Broadcastable ->
              { dims = beg_dims @ s_dims more_dims @ dims; bcast = Broadcastable; prov }
          | Row_var { v = v2; beg_dims = more_beg_dims } ->
              {
                dims = s_dims more_dims @ dims;
                bcast = Row_var { v = v2; beg_dims = beg_dims @ more_beg_dims };
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
                      r2 = { dims = []; bcast = Broadcastable; prov };
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
                        Dim_eq { d1 = Var v; d2 = get_dim ~d:1 ~proj_id:45 (); origin })
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
            | { dims = []; bcast = Broadcastable; prov = _ } :: more_rows ->
                apply_rows_constraint ~depth:(depth + 1) ~stage origin (List.rev more_rows) constr
                  env
            | { dims = []; bcast = Row_var { v; beg_dims = [] }; prov } :: more_rows
              when List.exists prov ~f:(fun (origin : provenance_origin) ->
                       equal_kind origin.kind `Input) ->
                let more_eqs, env =
                  apply_rows_constraint ~depth:(depth + 1) ~stage origin (List.rev more_rows) constr
                    env
                in
                ( Row_eq
                    {
                      r1 = row_of_var v prov;
                      r2 = { dims = []; bcast = Broadcastable; prov };
                      origin;
                    }
                  :: more_eqs,
                  env )
            | { dims = []; bcast = Row_var { v; beg_dims = [] }; prov } :: more_rows
              when List.exists prov ~f:(fun (origin : provenance_origin) ->
                       equal_kind origin.kind `Output) ->
                ( Row_eq
                    {
                      r1 = row_of_var v prov;
                      r2 = { dims = [ single_dim ]; bcast = Broadcastable; prov };
                      origin;
                    }
                  :: List.concat_map ~f:(check_empty_row ~origin) more_rows,
                  env )
            | { dims = _; bcast = Row_var { v = _; beg_dims = _ }; prov } :: _
              when List.exists prov ~f:(fun (origin : provenance_origin) ->
                       equal_kind origin.kind `Output) ->
                assert false
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
      | { bcast = Row_var { v; beg_dims }; dims; _ } -> (
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
                            { is_in_param = false; constr; cur = []; subr = []; lub = None; origin }));
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
          (Dim_eq { d1 = Var var; d2 = get_dim ~d:(denom * d / coeff) (); origin } :: extras, env)
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
              (Dim_eq { d1 = Var v; d2 = get_dim ~d:n (); origin } :: extras, env)
          | Num_elems 1, vs1, vs2 ->
              ( List.map
                  ~f:(fun v -> Dim_eq { d1 = Var v; d2 = get_dim ~d:1 ~proj_id:46 (); origin })
                  (vs1 @ vs2)
                @ extras,
                env )
          | Strided_var { coeff; var; denom }, [], [ v ]
            when equal_dim_var var v && (Utils.is_safe_val coeff || is_stage2_up stage) ->
              (* Total = (coeff * v / denom) / v = coeff / denom *)
              if Utils.safe_force coeff % denom = 0 then
                ( Dim_eq { d1 = Var v; d2 = get_dim ~d:(Utils.safe_force coeff / denom) (); origin }
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
    | { dims; bcast = Row_var { v; beg_dims }; _ }, Exact exact_dims
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
              r2 = { dims = []; bcast = Broadcastable; prov = r.prov };
              origin;
            }
          :: List.map2_exn exact_dims (beg_dims @ dims) ~f:(fun d1 d2 -> Dim_eq { d1; d2; origin })
          @ extras,
          env )
    | ( { bcast = Row_var { v; _ }; _ },
        Total_elems { numerator = Strided_var { coeff; var = _; denom }; divided_by = [] } )
      when is_stage2_up stage -> (
        (* Check if we have a LUB and if it meets our conditions *)
        match find_row env.row_env v with
        | Some (Bounds_row { lub = Some ({ dims = lub_dims; bcast = Broadcastable; _ } as lub); _ })
          when Utils.is_safe_val coeff && Utils.safe_force coeff > denom -> (
            (* Check if all LUB dimensions are known *)
            match collect_factors lub_dims with
            | Some (_known_product, []) ->
                (* Check if LUB has at most one dimension greater than 1 *)
                let greater_than_one =
                  List.filter lub_dims ~f:(function Dim { d; _ } -> d > 1 | _ -> false)
                in
                if List.length greater_than_one <= 1 then
                  (Row_eq { r1 = row_of_var v r.prov; r2 = lub; origin } :: extras, env)
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
    | { dims; bcast = Broadcastable; _ }, Exact exact_dims ->
        assert (not stored);
        (List.map2_exn exact_dims dims ~f:(fun d1 d2 -> Dim_eq { d1; d2; origin }) @ extras, env)

let rec dim_var_occurs_in_dim (v : dim_var) (d : dim) : bool =
  match d with
  | Var v' -> equal_dim_var v v'
  | Dim _ -> false
  | Affine { over; conv = None; _ } -> dim_var_occurs_in_dim v over
  | Affine { over; conv = Some { kernel; _ }; _ } ->
      dim_var_occurs_in_dim v over || dim_var_occurs_in_dim v kernel

let%debug5_sexp rec unify_dim ~stage origin (eq : dim * dim) env : constraint_ list * _ =
  let dim1 : dim = subst_dim env @@ fst eq and dim2 : dim = subst_dim env @@ snd eq in
  match (dim1, dim2) with
  | Dim { label = Some l1; _ }, Dim { label = Some l2; _ } when not (String.equal l1 l2) ->
      raise
      @@ Shape_error
           ("solved dimensions for axis: different labels", [ Dim_mismatch [ dim1; dim2 ] ])
  | Dim { d = d1; _ }, Dim { d = d2; _ } when d1 = d2 -> ([], env)
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
      unify_dim ~stage origin (get_dim ~d:(stride * s.d) (), dim) env
  | Affine { stride; over; conv = None | Some { use_padding = true; _ }; _ }, Dim s
  | Dim s, Affine { stride; over; conv = None | Some { use_padding = true; _ }; _ } ->
      (* stride_offset doesn't contribute to shapes when conv = None or use_padding = true *)
      if s.d >= 0 && s.d % stride = 0 then
        unify_dim ~stage origin (get_dim ~d:(s.d / stride) (), over) env
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
      (* stride_offset doesn't affect dimension - it just shifts which elements are accessed. Max
         index at stride_offset=stride-1: (s-1)*stride + (stride-1) + (k-1)*dilation So input dim =
         s * stride + (k-1) * dilation *)
      unify_dim ~stage origin (get_dim ~d:((stride * s.d) + (dilation * (k.d - 1))) (), dim) env
  | Affine { stride; over; conv = Some { dilation; kernel = Dim k; use_padding = false }; _ }, Dim s
  | Dim s, Affine { stride; over; conv = Some { dilation; kernel = Dim k; use_padding = false }; _ }
    ->
      (* Reverse: solve for over given input dim s. s = stride * over + dilation * (k - 1) over = (s
         - dilation * (k - 1)) / stride *)
      let kernel_extent = dilation * (k.d - 1) in
      let over_times_stride = s.d - kernel_extent in
      if over_times_stride >= 0 && over_times_stride % stride = 0 then
        unify_dim ~stage origin (get_dim ~d:(over_times_stride / stride) (), over) env
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
    ->
      (* Infer kernel dimension from the amount of contraction. i = s * stride + dilation * (k - 1)
         k = 1 + (i - s * stride) / dilation *)
      let kernel_extent = i.d - (stride * s.d) in
      if kernel_extent >= 0 && kernel_extent % dilation = 0 then
        let k = 1 + (kernel_extent / dilation) in
        unify_dim ~stage origin (Var v, get_dim ~d:k ()) env
      else
        raise
        @@ Shape_error
             ( "solved dimensions for axis: cannot infer kernel dimension",
               [ Dim_mismatch [ dim1; dim2 ] ] )
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
      (* stride_offset doesn't affect dimension - kernel extent is dilation * (k - 1) *)
      let extent1 = d1 * (k1.d - 1) and extent2 = d2 * (k2.d - 1) in
      if s1 = s2 && extent1 = extent2 then unify_dim ~stage origin (o1, o2) env
      else if s1 >= s2 && s1 % s2 = 0 then
        let stride = s1 / s2 in
        let new_extent = (extent1 - extent2) / s2 in
        (* Encode new_extent via a helper conv: dilation=1, kernel dim = new_extent + 1 *)
        let helper_conv =
          Some
            {
              dilation = 1;
              kernel = Dim { d = new_extent + 1; label = None; proj_id = None };
              use_padding = false;
            }
        in
        unify_dim ~stage origin
          (o2, Affine { stride; over = o1; conv = helper_conv; stride_offset = 0 })
          env
      else if s2 >= s1 && s2 % s1 = 0 then
        let stride = s2 / s1 in
        let new_extent = (extent2 - extent1) / s1 in
        (* Encode new_extent via a helper conv: dilation=1, kernel dim = new_extent + 1 *)
        let helper_conv =
          Some
            {
              dilation = 1;
              kernel = Dim { d = new_extent + 1; label = None; proj_id = None };
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
      (* conv = None and use_padding = true have extent 0; use_padding = false has extent dilation *
         (k-1) *)
      let extent1 = 0 and extent2 = d2 * (k2.d - 1) in
      if s1 = s2 && extent1 = extent2 then unify_dim ~stage origin (o1, o2) env
      else if s1 >= s2 && s1 % s2 = 0 then
        let stride = s1 / s2 in
        let new_extent = (extent1 - extent2) / s2 in
        let helper_conv =
          Some
            {
              dilation = 1;
              kernel = Dim { d = new_extent + 1; label = None; proj_id = None };
              use_padding = false;
            }
        in
        unify_dim ~stage origin
          (o2, Affine { stride; over = o1; conv = helper_conv; stride_offset = 0 })
          env
      else if s2 >= s1 && s2 % s1 = 0 then
        let stride = s2 / s1 in
        let new_extent = (extent2 - extent1) / s1 in
        let helper_conv =
          Some
            {
              dilation = 1;
              kernel = Dim { d = new_extent + 1; label = None; proj_id = None };
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
            }
        | Some (Solved_dim _) -> assert false
        | Some (Bounds_dim { is_in_param = _; cur; subr; lub; constr; origin = origin1; _ } as in_)
          ->
            let origin = merge_origins origin origin1 in
            let from_ = [%sexp_of: dim_var * dim_entry] (v, drop_origin in_) in
            let dim_env = Utils.Tree_map.map env.dim_env ~f in
            List.iter cur ~f:(fun cur ->
                ineqs := Dim_ineq { cur = Var cur; subr = dim2; from_; origin } :: !ineqs);
            List.iter subr ~f:(fun subr ->
                ineqs := Dim_ineq { cur = dim2; subr = Var subr; from_; origin } :: !ineqs);
            Option.iter lub ~f:(fun lub ->
                ineqs := Dim_ineq { cur = lub; subr = dim2; from_; origin } :: !ineqs);
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
  | ( { bcast = Row_var { v = v1; beg_dims = beg_dims1 }; dims = dims1; prov = _ },
      { bcast = Row_var { v = v2; beg_dims = beg_dims2 }; dims = dims2; prov = _ } )
    when equal_row_var v1 v2 ->
      let dims1_l = l dims1
      and dims2_l = l dims2
      and beg_dims1_l = l beg_dims1
      and beg_dims2_l = l beg_dims2 in
      if beg_dims1_l + dims1_l <> beg_dims2_l + dims2_l then
        raise
        @@ Shape_error ("Infinite number of axes by self-reference", [ Row_mismatch [ r1; r2 ] ]);
      let result = unify_suffix ([], env) dims1 dims2 @@ min dims1_l dims2_l in
      unify_suffix result (List.rev beg_dims1) (List.rev beg_dims2) @@ min beg_dims1_l beg_dims2_l
  | ({ bcast = Row_var { v; beg_dims = beg_dims1 }; dims = dims1; prov = _ } as r1), r2
  | r2, ({ bcast = Row_var { v; beg_dims = beg_dims1 }; dims = dims1; prov = _ } as r1) -> (
      let dims1_l : int = l dims1
      and dims2_l : int = l r2.dims
      and beg_dims1_l : int = l beg_dims1 in
      let beg_dims2_l : int =
        match r2.bcast with Row_var { beg_dims; _ } -> l beg_dims | Broadcastable -> 0
      in
      let prov = merge_provenance r1.prov r2.prov in
      let beg_dims_l = min beg_dims1_l beg_dims2_l in
      if dims1_l > dims2_l || (dims1_l = dims2_l && beg_dims1_l > beg_dims2_l) then
        if is_row_var r2.bcast then unify_row ~stage origin (r2, r1) env
        else raise @@ Shape_error ("Number of axes mismatch", [ Row_mismatch [ r1; r2 ] ])
      else
        let orig_rows = [ r1; r2 ] in
        let (beg_handled : bool), (ineqs, env), (value : row) =
          match r2.bcast with
          | Row_var { v = v2; beg_dims = beg_dims2 } ->
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
                  let bcast = Row_var { v; beg_dims = [] } in
                  let value : row = { bcast; dims; prov } in
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
                let bcast = Row_var { v = v2; beg_dims = List.drop beg_dims2 beg_dims_l } in
                let value : row = { bcast; dims; prov } in
                (beg_dims_l = l beg_dims1, result, value)
          | Broadcastable ->
              if dims1_l + beg_dims1_l > dims2_l then
                raise @@ Shape_error ("Number of axes mismatch", [ Row_mismatch [ r1; r2 ] ])
              else
                let dims = List.drop r2.dims beg_dims1_l |> Fn.flip drop_from_end dims1_l in
                let result =
                  List.zip_exn beg_dims1 (List.take r2.dims beg_dims1_l)
                  @ List.zip_exn dims1 (take_from_end r2.dims dims1_l)
                  |> List.fold ~init:([], env) ~f:(fun acc (d1, d2) ->
                      solve acc (Dim_eq { d1; d2; origin }))
                in
                let value : row = { bcast = Broadcastable; dims; prov } in
                (true, result, value)
        in
        (* From now on, we have no use for un-reduced r2 since we deal with the row variable. *)
        let r2 : row = value in
        let ineqs : constraint_ list ref = ref ineqs in
        let f in_ =
          let more_ineqs, result = s_row_one_in_entry v ~value:(value : row) ~in_ in
          ineqs := more_ineqs @ !ineqs;
          result
        in
        let result env =
          let row_env = Utils.Tree_map.map env.row_env ~f in
          let unsolved, env =
            if beg_handled then
              let constr =
                match find_row env.row_env v with
                | Some (Bounds_row { constr; origin; _ }) ->
                    [ Rows_constr { r = [ value ]; constr; origin } ]
                | _ -> []
              in
              (constr, { env with row_env = add_row row_env ~key:v ~data:(Solved_row value) })
            else
              ( [
                  Row_eq
                    {
                      r1 =
                        {
                          dims = [];
                          bcast = Row_var { v; beg_dims = List.drop beg_dims1 beg_dims_l };
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
        | Some (Bounds_row { is_in_param = _; cur; subr; lub; constr; origin = origin1 }) ->
            let origin = merge_origins origin origin1 in
            let env =
              if beg_handled then (
                List.iter cur ~f:(fun cur ->
                    ineqs :=
                      Row_ineq { cur = row_of_var cur value.prov; subr = r2; origin } :: !ineqs);
                List.iter subr ~f:(fun subr ->
                    ineqs :=
                      Row_ineq { subr = row_of_var subr value.prov; cur = r2; origin } :: !ineqs);
                Option.iter lub ~f:(fun lub ->
                    ineqs := Row_ineq { cur = lub; subr = r2; origin } :: !ineqs);
                let extras, env = apply_row_constraint ~depth:0 stage origin value constr env in
                ineqs := extras @ !ineqs;
                env)
              else env
            in
            let _bound_elim_ineqs : constraint_ list = !ineqs in
            result env)
  | ( ({ bcast = Broadcastable; dims = dims1; prov = _ } as r1),
      ({ bcast = Broadcastable; dims = dims2; prov = _ } as r2) ) -> (
      match List.zip dims1 dims2 with
      | Unequal_lengths ->
          raise @@ Shape_error ("Mismatching number of axes", [ Row_mismatch [ r1; r2 ] ])
      | Ok eqs ->
          List.fold ~init:([], env)
            ~f:(fun acc (d1, d2) -> solve acc (Dim_eq { d1; d2; origin }))
            eqs)

let%track5_sexp solve_dim_ineq ~(stage : stage) origin ~(cur : dim) ~(subr : dim) env :
    constraint_ list * _ =
  let nonredundant ?(more = []) (v : dim_var) (vs : dim_var list) : dim_var list =
    let _more : dim_var list = more in
    Utils.sorted_diff ~compare:compare_dim_var
      (List.dedup_and_sort ~compare:compare_dim_var (v :: vs))
      more
  in
  let rec cyclic ~subr_v ~curs =
    (* TODO: it's somewhat inefficient *)
    List.exists curs ~f:(fun cur_v ->
        equal_dim_var subr_v cur_v
        ||
        match find_dim env.dim_env cur_v with
        | None | Some (Solved_dim (Dim _)) -> false
        | Some (Solved_dim (Var v)) -> equal_dim_var subr_v v
        | Some (Solved_dim (Affine _)) -> false (* Affine dimensions can't be cyclic *)
        | Some (Bounds_dim { cur = curs; _ }) -> cyclic ~subr_v ~curs)
  in
  match (cur, subr) with
  | cur, subr when equal_dim cur subr -> ([], env)
  | Dim { label = Some l1; _ }, Dim { label = Some l2; _ } when not (String.equal l1 l2) ->
      raise
      @@ Shape_error
           ("dimension comparison for axis: different labels", [ Dim_mismatch [ cur; subr ] ])
  | Dim { d = d1; _ }, Dim { d = d2; _ } when d1 = d2 -> ([], env)
  | _, Dim { d = 1; label = None; _ } -> ([], env)
  | (Dim { d = 1; label = None; _ } as cur), _ -> ([ Dim_eq { d1 = subr; d2 = cur; origin } ], env)
  | Affine _, _ | _, Affine _ -> ([ Dim_eq { d1 = subr; d2 = cur; origin } ], env)
  | Var cur_v, Var subr_v -> (
      match (find_dim env.dim_env cur_v, find_dim env.dim_env subr_v) with
      | Some (Bounds_dim { cur = cur1; _ }), _ when List.mem ~equal:equal_dim_var cur1 subr_v ->
          ([ Dim_eq { d1 = cur; d2 = subr; origin } ], env)
      | _, Some (Bounds_dim { subr = subr2; _ }) when List.mem ~equal:equal_dim_var subr2 cur_v ->
          ([ Dim_eq { d1 = cur; d2 = subr; origin } ], env)
      | None, None ->
          ( [],
            {
              env with
              dim_env =
                env.dim_env
                |> add_dim ~key:cur_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param = false;
                            has_uniq_constr_unless = None;
                            lub = None;
                            cur = [];
                            subr = [ subr_v ];
                            constr = Unconstrained_dim;
                            origin;
                          })
                |> add_dim ~key:subr_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param = false;
                            has_uniq_constr_unless = None;
                            lub = None;
                            cur = [ cur_v ];
                            subr = [];
                            constr = Unconstrained_dim;
                            origin;
                          });
            } )
      | Some (Solved_dim _), _ | _, Some (Solved_dim _) -> assert false
      | ( Some
            (Bounds_dim
               {
                 is_in_param;
                 cur = cur1;
                 subr = subr1;
                 lub = lub1;
                 constr = constr1;
                 origin = origin1;
                 _;
               } as in_),
          None ) ->
          let origin = merge_origins origin origin1 in
          let from_ = [%sexp_of: dim_var * dim_entry] (cur_v, drop_origin in_) in
          let from_lub =
            Option.to_list lub1 |> List.map ~f:(fun cur -> Dim_ineq { cur; subr; from_; origin })
          in
          let from_constr1, constr1 = apply_dim_constraint ~source:Subr ~stage subr constr1 env in
          let from_constr2, constr2 =
            apply_dim_constraint ~source:Cur ~stage cur Unconstrained_dim env
          in
          ( from_constr1 @ from_constr2 @ from_lub,
            {
              env with
              dim_env =
                env.dim_env
                |> add_dim ~key:cur_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param;
                            has_uniq_constr_unless = None;
                            lub = lub1;
                            cur = cur1;
                            subr = nonredundant subr_v subr1;
                            constr = constr1;
                            origin;
                          })
                |> add_dim ~key:subr_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param = false;
                            has_uniq_constr_unless = None;
                            lub = None;
                            cur = [ cur_v ];
                            subr = [];
                            constr = constr2;
                            origin;
                          });
            } )
      | ( Some
            (Bounds_dim
               {
                 is_in_param = _;
                 cur = _;
                 subr = [ subr1 ];
                 lub = None;
                 constr = _;
                 origin = origin1;
                 _;
               }),
          Some
            (Bounds_dim
               {
                 is_in_param = _;
                 cur = [ cur2 ];
                 subr = _;
                 lub = None;
                 constr = _;
                 origin = origin2;
                 _;
               }) )
        when is_stage2_up stage && equal_dim_var subr_v subr1 && equal_dim_var cur_v cur2 ->
          let origin = merge_origins origin (merge_origins origin1 origin2) in
          (* A heuristic to reduce template variables coming from e.g. einsum notation expansion. *)
          ([ Dim_eq { d1 = subr; d2 = cur; origin } ], env)
      | Some (Bounds_dim { cur = curs; origin = origin1; _ }), Some (Bounds_dim _)
        when cyclic ~subr_v ~curs ->
          let origin = merge_origins origin origin1 in
          ([ Dim_eq { d1 = subr; d2 = cur; origin } ], env)
      | ( None,
          Some
            (Bounds_dim
               {
                 is_in_param;
                 cur = cur2;
                 subr = subr2;
                 lub = lub2;
                 constr = constr2;
                 origin = origin2;
                 _;
               }) ) ->
          let origin = merge_origins origin origin2 in
          let from_constr1, constr1 =
            apply_dim_constraint ~source:Subr ~stage subr Unconstrained_dim env
          in
          let from_constr2, constr2 = apply_dim_constraint ~source:Cur ~stage cur constr2 env in
          ( from_constr2 @ from_constr1,
            {
              env with
              dim_env =
                env.dim_env
                |> add_dim ~key:cur_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param;
                            has_uniq_constr_unless = None;
                            lub = None;
                            cur = [];
                            subr = [ subr_v ];
                            constr = constr1;
                            origin;
                          })
                |> add_dim ~key:subr_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param;
                            has_uniq_constr_unless = None;
                            lub = lub2;
                            cur = nonredundant cur_v cur2;
                            subr = subr2;
                            constr = constr2;
                            origin;
                          });
            } )
      | ( Some
            (Bounds_dim
               {
                 is_in_param = iip1;
                 cur = cur1;
                 subr = subr1;
                 lub = lub1;
                 constr = constr1;
                 origin = origin1;
                 _;
               } as in_),
          Some
            (Bounds_dim
               {
                 is_in_param = iip2;
                 cur = cur2;
                 subr = subr2;
                 lub = lub2;
                 constr = constr2;
                 origin = origin2;
                 _;
               }) ) ->
          let origin = merge_origins origin (merge_origins origin1 origin2) in
          let from_ = [%sexp_of: dim_var * dim_var * dim_entry] (cur_v, subr_v, drop_origin in_) in
          let from_lub =
            Option.to_list lub1 |> List.map ~f:(fun cur -> Dim_ineq { cur; subr; from_; origin })
          in
          let from_constr1, constr1 = apply_dim_constraint ~source:Subr ~stage subr constr1 env in
          let from_constr2, constr2 = apply_dim_constraint ~source:Cur ~stage cur constr2 env in
          ( from_constr1 @ from_constr2 @ from_lub,
            {
              env with
              dim_env =
                env.dim_env
                |> add_dim ~key:cur_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param = iip1;
                            has_uniq_constr_unless = None;
                            lub = lub1;
                            cur = cur1;
                            subr = nonredundant ~more:subr2 subr_v subr1;
                            constr = constr1;
                            origin;
                          })
                |> add_dim ~key:subr_v
                     ~data:
                       (Bounds_dim
                          {
                            is_in_param = iip2;
                            has_uniq_constr_unless = None;
                            lub = lub2;
                            cur = nonredundant ~more:cur1 cur_v cur2;
                            subr = subr2;
                            constr = constr2;
                            origin;
                          });
            } ))
  | _, Var subr_v -> (
      match find_dim env.dim_env subr_v with
      | None ->
          ( [],
            {
              env with
              dim_env =
                add_dim env.dim_env ~key:subr_v
                  ~data:
                    (Bounds_dim
                       {
                         is_in_param = false;
                         has_uniq_constr_unless = None;
                         lub = Some cur;
                         cur = [];
                         subr = [];
                         constr = Unconstrained_dim;
                         origin;
                       });
            } )
      | Some (Solved_dim _) -> assert false
      | Some
          (Bounds_dim
             {
               is_in_param;
               cur = cur2;
               subr = subr2;
               lub = Some lub2;
               constr = constr2;
               origin = origin2;
               _;
             }) ->
          let origin = merge_origins origin origin2 in
          let lub, lub_forcing =
            match (cur, lub2) with
            | Dim { d = d1; label = l1; _ }, Dim { d = d2; label = l2; _ }
              when d1 = d2 && Option.equal String.equal l1 l2 ->
                (cur, [])
            | Dim _, Dim _ (* when d1 <> d2 or l1 <> l2 *) ->
                let lub = get_dim ~d:1 ~proj_id:47 () in
                (lub, [ Dim_eq { d1 = subr; d2 = lub; origin } ])
                (* raise @@ Shape_error ( "dimension comparison for axis: upper bound mismatch", [
                   Dim_mismatch [ lub2; cur; subr ] ] ) *)
            | Var _, _ | _, Var _ -> assert false
            | Affine _, _ | _, Affine _ -> assert false
          in
          let from_constr, constr2 = apply_dim_constraint ~source:Cur ~stage cur constr2 env in
          ( from_constr @ lub_forcing,
            {
              env with
              dim_env =
                add_dim env.dim_env ~key:subr_v
                  ~data:
                    (Bounds_dim
                       {
                         is_in_param;
                         has_uniq_constr_unless = None;
                         lub = Some lub;
                         cur = cur2;
                         subr = subr2;
                         constr = constr2;
                         origin;
                       });
            } )
      | Some
          (Bounds_dim
             {
               is_in_param;
               cur = cur2;
               subr = subr2;
               lub = None;
               constr = constr2;
               origin = origin2;
               _;
             } as in_) ->
          let origin = merge_origins origin origin2 in
          let from_ = [%sexp_of: dim_var * dim_entry] (subr_v, drop_origin in_) in
          let from_constr, constr2 = apply_dim_constraint ~source:Cur ~stage cur constr2 env in
          ( from_constr
            @ List.map subr2 ~f:(fun subr_v -> Dim_ineq { cur; subr = Var subr_v; from_; origin }),
            {
              env with
              dim_env =
                add_dim env.dim_env ~key:subr_v
                  ~data:
                    (Bounds_dim
                       {
                         is_in_param;
                         has_uniq_constr_unless = None;
                         lub = Some cur;
                         cur = cur2;
                         subr = subr2;
                         constr = constr2;
                         origin;
                       });
            } ))
  | Var _, Dim _ (* when d2 > 1 or labeled *) -> ([ Dim_eq { d1 = cur; d2 = subr; origin } ], env)
  | Dim _, Dim _ ->
      raise
      @@ Shape_error ("dimension comparison for axis: mismatch", [ Dim_mismatch [ cur; subr ] ])

let global_template_cache = Hashtbl.Poly.create ()

let%debug5_sexp solve_row_ineq ~(stage : stage) origin ~(cur : t) ~(subr : t) env :
    constraint_ list * _ =
  let nonredundant ?(more = []) v vs =
    Utils.sorted_diff ~compare:compare_row_var
      (List.dedup_and_sort ~compare:compare_row_var (v :: vs))
      more
  in
  let l = List.length in
  let cur_dims_l : int = l cur.dims and subr_dims_l : int = l subr.dims in
  let cur_beg_dims =
    match cur.bcast with Row_var { beg_dims; _ } -> beg_dims | Broadcastable -> []
  in
  let subr_beg_dims =
    match subr.bcast with Row_var { beg_dims; _ } -> beg_dims | Broadcastable -> []
  in
  let cur_beg_dims_l = l cur_beg_dims and subr_beg_dims_l = l subr_beg_dims in
  let beg_dims_l = min cur_beg_dims_l subr_beg_dims_l in
  let dims_l = min cur_dims_l subr_dims_l in
  let from_ = sexp_of_constraint_ (drop_constraint_origin (Row_ineq { cur; subr; origin })) in
  let ineqs =
    List.map2_exn
      ~f:(fun cur subr -> Dim_ineq { cur; subr; from_; origin })
      (take_from_end cur_beg_dims beg_dims_l)
      (take_from_end subr_beg_dims beg_dims_l)
    @ List.map2_exn
        ~f:(fun cur subr -> Dim_ineq { cur; subr; from_; origin })
        (take_from_end cur.dims dims_l) (take_from_end subr.dims dims_l)
  in
  match (cur, subr) with
  | { dims = _; bcast = Row_var { v; _ }; prov }, _
  | _, { dims = _; bcast = Row_var { v; _ }; prov }
    when is_stage6_up stage ->
      ( Row_ineq { cur; subr; origin }
        :: Row_eq
             { r1 = row_of_var v prov; r2 = { dims = []; bcast = Broadcastable; prov }; origin }
        :: ineqs,
        env )
  | cur, subr when equal_row cur subr -> ([], env)
  | { bcast = Row_var { v = cur_v; _ }; _ }, { bcast = Row_var { v = subr_v; _ }; _ }
    when equal_row_var cur_v subr_v ->
      if cur_dims_l + cur_beg_dims_l = subr_dims_l + subr_beg_dims_l then (ineqs, env)
      else
        raise
        @@ Shape_error ("Infinite number of axes by self-reference", [ Row_mismatch [ cur; subr ] ])
  | { bcast = Row_var { v = cur_v; _ }; _ }, { bcast = Row_var { v = subr_v; _ }; _ }
    when cur_dims_l = subr_dims_l && cur_beg_dims_l = subr_beg_dims_l -> (
      match (find_row env.row_env cur_v, find_row env.row_env subr_v) with
      | Some (Bounds_row { cur = cur1; origin = origin1; _ }), _
        when List.mem ~equal:equal_row_var cur1 subr_v ->
          let origin = merge_origins origin origin1 in
          ( Row_eq { r1 = row_of_var subr_v subr.prov; r2 = row_of_var cur_v cur.prov; origin }
            :: ineqs,
            env )
      | _, Some (Bounds_row { subr = subr2; origin = origin2; _ })
        when List.mem ~equal:equal_row_var subr2 cur_v ->
          let origin = merge_origins origin origin2 in
          ( Row_eq { r1 = row_of_var subr_v subr.prov; r2 = row_of_var cur_v cur.prov; origin }
            :: ineqs,
            env )
      | ( Some (Bounds_row { subr = [ subr1 ]; origin = origin1; _ }),
          Some (Bounds_row { cur = [ cur2 ]; origin = origin2; _ }) )
        when is_stage2_up stage && equal_row_var subr1 subr_v && equal_row_var cur2 cur_v ->
          let origin = merge_origins origin (merge_origins origin1 origin2) in
          ( Row_eq { r1 = row_of_var subr_v subr.prov; r2 = row_of_var cur_v cur.prov; origin }
            :: ineqs,
            env )
      | Some (Bounds_row { subr = subr1; _ }), _ when List.mem ~equal:equal_row_var subr1 subr_v ->
          (ineqs, env)
      | _, Some (Bounds_row { cur = cur2; _ }) when List.mem ~equal:equal_row_var cur2 cur_v ->
          (ineqs, env)
      | None, None ->
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> add_row ~key:cur_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param = false;
                            cur = [];
                            subr = [ subr_v ];
                            lub = None;
                            constr = Unconstrained;
                            origin;
                          })
                |> add_row ~key:subr_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param = false;
                            cur = [ cur_v ];
                            subr = [];
                            lub = None;
                            constr = Unconstrained;
                            origin;
                          });
            } )
      | ( Some
            (Bounds_row
               {
                 is_in_param;
                 cur = cur1;
                 subr = subr1;
                 lub = lub1;
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
                |> add_row ~key:cur_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param;
                            cur = cur1;
                            subr = nonredundant subr_v subr1;
                            lub = lub1;
                            constr = constr1;
                            origin;
                          })
                |> add_row ~key:subr_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param = false;
                            cur = [ cur_v ];
                            subr = [];
                            lub = None;
                            constr = Unconstrained;
                            origin;
                          });
            } )
      | ( None,
          Some
            (Bounds_row
               {
                 is_in_param;
                 cur = cur2;
                 subr = subr2;
                 lub = lub2;
                 constr = constr2;
                 origin = origin2;
               }) ) ->
          let origin = merge_origins origin origin2 in
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> add_row ~key:subr_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param;
                            cur = nonredundant cur_v cur2;
                            subr = subr2;
                            lub = lub2;
                            constr = constr2;
                            origin;
                          })
                |> add_row ~key:cur_v
                     ~data:
                       (Bounds_row
                          {
                            (* The upper bound shouldn't collapse on the param below. *)
                            is_in_param;
                            cur = [];
                            subr = [ subr_v ];
                            lub = None;
                            constr = Unconstrained;
                            origin;
                          });
            } )
      | ( Some
            (Bounds_row
               {
                 is_in_param = iip1;
                 cur = cur1;
                 subr = subr1;
                 lub = lub1;
                 constr = constr1;
                 origin = origin1;
               }),
          Some
            (Bounds_row
               {
                 is_in_param = iip2;
                 cur = cur2;
                 subr = subr2;
                 lub = lub2;
                 constr = constr2;
                 origin = origin2;
               }) ) ->
          let origin = merge_origins origin (merge_origins origin1 origin2) in
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> add_row ~key:cur_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param = iip1 || iip2;
                            cur = cur1;
                            subr = nonredundant subr_v subr1;
                            lub = lub1;
                            constr = constr1;
                            origin;
                          })
                |> add_row ~key:subr_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param = iip2;
                            cur = nonredundant cur_v cur2;
                            subr = subr2;
                            lub = lub2;
                            constr = constr2;
                            origin;
                          });
            } )
      | Some (Solved_row _), _ | _, Some (Solved_row _) -> assert false)
  | { bcast = Row_var { v = cur_v; _ }; dims; _ }, _
    when cur_dims_l + cur_beg_dims_l < subr_dims_l + subr_beg_dims_l ->
      let budget = subr_dims_l + subr_beg_dims_l - (cur_dims_l + cur_beg_dims_l) in
      let more_dims_l = min budget @@ max 0 (subr_dims_l - cur_dims_l) in
      let more_dims : dim list =
        Array.(to_list @@ init more_dims_l ~f:(fun _ -> Var (get_var ())))
      in
      let budget = budget - more_dims_l in
      let more_beg_dims_l = min budget @@ max 0 (subr_beg_dims_l - cur_beg_dims_l) in
      let more_beg_dims : dim list =
        Array.(to_list @@ init more_beg_dims_l ~f:(fun _ -> Var (get_var ())))
      in
      (* The key of the template cache reflects that cur_v will end up substituted by
         {dims=more_dims; bcast=Row_var templ_v}. TODO: should we cache more_dims also? *)
      let templ_key : row_var * int * int =
        (cur_v, subr_dims_l - cur_dims_l, subr_beg_dims_l - cur_beg_dims_l)
      in
      let templ_v : row_var =
        Hashtbl.find_or_add global_template_cache templ_key ~default:get_row_var
      in
      if more_dims_l > 0 then add_safe_to_guess templ_v;
      let template : t =
        {
          dims = more_dims @ dims;
          bcast = Row_var { v = templ_v; beg_dims = cur_beg_dims @ more_beg_dims };
          prov = cur.prov;
        }
      in
      (* We don't need to add any dimension inequalities, because they'll be captured by the extra
         row inequalities. *)
      ( [ Row_eq { r1 = cur; r2 = template; origin }; Row_ineq { cur = template; subr; origin } ],
        env )
  | { bcast = Broadcastable; _ }, _ when cur_dims_l + cur_beg_dims_l < subr_dims_l + subr_beg_dims_l
    ->
      raise
      @@ Shape_error
           ( "Too many axes in a subtensor; maybe using * instead of *.?",
             [ Row_mismatch [ cur; subr ] ] )
  | { bcast; dims; prov = _ }, { bcast = Row_var { v = subr_v; _ }; _ }
    when subr_dims_l <= cur_dims_l && subr_beg_dims_l <= cur_beg_dims_l -> (
      let bcast =
        match bcast with
        | Row_var { v; beg_dims } -> Row_var { v; beg_dims = List.drop beg_dims beg_dims_l }
        | Broadcastable -> Broadcastable
      in
      let r_cur = { bcast; dims = drop_from_end dims dims_l; prov = cur.prov } in
      match find_row env.row_env subr_v with
      | None ->
          ( ineqs,
            {
              env with
              row_env =
                add_row env.row_env ~key:subr_v
                  ~data:
                    (Bounds_row
                       {
                         is_in_param = false;
                         cur = [];
                         subr = [];
                         lub = Some r_cur;
                         constr = Unconstrained;
                         origin;
                       });
            } )
      | Some
          (Bounds_row
             {
               is_in_param;
               cur = cur2;
               subr = subr2;
               lub = None;
               constr = constr2;
               origin = origin2;
             }) ->
          let origin = merge_origins origin origin2 in
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> add_row ~key:subr_v
                     ~data:
                       (Bounds_row
                          {
                            is_in_param;
                            cur = cur2;
                            subr = subr2;
                            lub = Some r_cur;
                            constr = constr2;
                            origin;
                          });
            } )
      | Some
          (Bounds_row
             {
               is_in_param;
               cur = cur2;
               subr = subr2;
               lub = Some lub2;
               constr = constr2;
               origin = origin2;
             }) -> (
          let origin = merge_origins origin origin2 in
          let len1 = List.length r_cur.dims and len2 = List.length lub2.dims in
          let lub_len = min len1 len2 in
          let lub_is_cur = len1 < len2 || (len1 = len2 && is_broadcastable cur.bcast) in
          let lub_prov = if lub_is_cur then r_cur.prov else lub2.prov in
          (* TODO: we lose connection here with the other bound if both have row variables. *)
          let lub_bcast = if lub_is_cur then r_cur.bcast else lub2.bcast in
          let lub_dims =
            List.map2_exn (take_from_end r_cur.dims lub_len) (take_from_end lub2.dims lub_len)
              ~f:(fun d1 d2 ->
                match (d1, d2) with
                (* Prefer dimensions without labels (more general), then prefer d=1 (more general
                   size) *)
                | Dim { d = 1; label = None; _ }, _ -> d1
                | _, Dim { d = 1; label = None; _ } -> d2
                | Dim { d = 1; label = Some _; _ }, Dim { label = None; _ } -> d2
                | Dim { label = None; _ }, Dim { d = 1; label = Some _; _ } -> d1
                | Dim { d = d1; _ }, Dim { d = d2; _ } when d1 <> d2 -> get_dim ~d:1 ~proj_id:48 ()
                | Dim { label = Some l1; _ }, Dim { label = Some l2; _ }
                  when not (String.equal l1 l2) ->
                    get_dim ~d:1 ~proj_id:63 ()
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
                    get_dim ~d:1 ~proj_id:49 ()
                | ( Affine
                      {
                        stride;
                        over = Dim s;
                        conv = Some { kernel = Dim k; dilation; use_padding = false };
                        stride_offset = _;
                      },
                    Dim s' )
                | ( Dim s',
                    Affine
                      {
                        stride;
                        over = Dim s;
                        conv = Some { kernel = Dim k; dilation; use_padding = false };
                        stride_offset = _;
                      } )
                  when (stride * s.d) + (dilation * (k.d - 1)) <> s'.d ->
                    get_dim ~d:1 ~proj_id:50 ()
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
                    get_dim ~d:1 ~proj_id:51 ()
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
                  when (stride1 * s1.d) + (dilation1 * (k1.d - 1))
                       <> (stride2 * s2.d) + (dilation2 * (k2.d - 1)) ->
                    get_dim ~d:1 ~proj_id:52 ()
                | Var _, _ -> d1
                | _, Var _ -> d2
                | _, Dim _ -> d2
                | _ -> d1)
          in
          let lub = { dims = lub_dims; bcast = lub_bcast; prov = lub_prov } in
          let row_env =
            env.row_env
            |> add_row ~key:subr_v
                 ~data:
                   (Bounds_row
                      {
                        is_in_param;
                        cur = cur2;
                        subr = subr2;
                        lub = Some lub;
                        constr = constr2;
                        origin;
                      })
          in
          match lub with
          | { dims = [] | [ Dim { d = 1; _ } ]; bcast = Broadcastable; prov = _ } ->
              ( Row_eq { r1 = row_of_var subr_v subr.prov; r2 = lub; origin } :: ineqs,
                { env with row_env } )
          | _ -> (ineqs, { env with row_env }))
      | Some (Solved_row _) -> assert false)
  | _ when cur_beg_dims_l > beg_dims_l && not (is_stage7 stage) ->
      (Row_ineq { cur; subr; origin } :: ineqs, env)
  | _, { bcast = Broadcastable; _ }
    when subr_dims_l + subr_beg_dims_l <= cur_dims_l + cur_beg_dims_l ->
      (ineqs, env)
  | { bcast = Row_var _ | Broadcastable; _ }, { bcast = Row_var _ | Broadcastable; _ } ->
      (Row_ineq { cur; subr; origin } :: ineqs, env)

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

let%debug5_sexp close_dim_terminal ~(stage : stage) ~is_param origin env (dim : dim) :
    constraint_ list =
  match dim with
  | Dim _ -> []
  | Var v -> (
      match find_dim env.dim_env v with
      | Some (Solved_dim _) -> assert false
      | Some
          (Bounds_dim
             { is_in_param; has_uniq_constr_unless; lub = None; constr = Unconstrained_dim; _ })
        when is_stage3_up stage ->
          (* Check if we can guess this variable to 1 *)
          if not (can_guess_dim_to_one env has_uniq_constr_unless) then
            [ Terminal_dim (is_param, dim, origin) ]
          else if is_param || is_in_param then
            raise
            @@ Shape_error
                 ("You forgot to specify the hidden dimension(s) 1", [ Dim_mismatch [ dim ] ])
          else [ Dim_eq { d1 = dim; d2 = get_dim ~d:1 ~proj_id:53 (); origin } ]
      | Some (Bounds_dim { lub = Some lub; _ }) when is_stage4_up stage ->
          [ Dim_eq { d1 = dim; d2 = lub; origin } ]
      | _ when not (is_stage5_up stage) -> [ Terminal_dim (is_param, dim, origin) ]
      | _ -> [])
  | Affine _ ->
      (* The input dimension itself cannot be dim-1, and the over dimension doesn't become
         transitively terminal. *)
      []

let last_dim_is dims p = match List.last dims with Some (Dim { d; _ }) -> p d | _ -> false

let r_dims r =
  match r.bcast with Broadcastable -> r.dims | Row_var { beg_dims; _ } -> beg_dims @ r.dims

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

let%track5_sexp rec eliminate_rows_constraint ~depth stage origin ~lub (rows : row list)
    (constr : row_constraint) env : constraint_ list * environment =
  if depth > 4 then ([ Rows_constr { r = rows; constr; origin } ], env)
  else
    match rows_to_row_or_vars rows with
    | Either.First single_row ->
        eliminate_row_constraint ~depth:(depth + 1) stage origin ~terminal:false ~lub single_row
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
                    | Some (Bounds_row { is_in_param = false; lub = None; _ })
                    | Some (Bounds_row { lub = Some { dims = []; bcast = Broadcastable; _ }; _ }) ->
                        true
                    | _ -> false
                  then
                    let r1 = row_of_var v prov in
                    [ Row_eq { r1; r2 = { dims = []; bcast = Broadcastable; prov }; origin } ]
                  else [])
            in
            if is_stage5_up stage then
              let rows =
                List.map rows ~f:(function
                  | { bcast = Row_var { v = v'; _ }; _ } as r when equal_row_var v' v -> r
                  | r -> { r with dims = r_dims r; bcast = Broadcastable })
              in
              let ineqs, env =
                eliminate_rows_constraint ~depth:(depth + 1) stage origin ~lub rows constr env
              in
              (other_eqs @ ineqs, env)
            else (other_eqs @ [ Rows_constr { r = rows; constr; origin } ], env)
        | _ -> ([ Rows_constr { r = rows; constr; origin } ], env))

and eliminate_row_constraint ~depth stage origin ~terminal ~(lub : row option) (r : row)
    (constr : row_constraint) env : constraint_ list * environment =
  let keep_constr () =
    let ineqs, env = apply_row_constraint ~depth stage origin r constr env in
    List.fold ineqs ~init:([], env) ~f:(fun (ineqs, env) ineq ->
        match ineq with
        | Rows_constr { r = rows; constr; origin } ->
            let ineqs', env =
              eliminate_rows_constraint ~depth:(depth + 1) stage origin ~lub:None rows constr env
            in
            (ineqs @ ineqs', env)
        | ineq -> ([ ineq ], env))
  in
  match r with
  | { bcast = Broadcastable; _ } -> keep_constr ()
  | { bcast = Row_var { v; beg_dims }; dims; prov } -> (
      let r1 = row_of_var v prov in
      let opt_row_error () =
        if row_var_is_in_param v env && not (is_safe_to_guess v) then
          raise
          @@ Shape_error ("You forgot to specify the hidden dimension(s) 2", [ Row_mismatch [ r ] ])
      in
      let no_further_axes ~guess () =
        if guess then opt_row_error ();
        Row_eq { r1; r2 = { dims = []; bcast = Broadcastable; prov }; origin }
      in
      (* Note: the reduced constraint applies to just the row variable. *)
      match reduce_row_constraint constr ~beg_dims ~dims with
      | Total_elems { numerator; divided_by } -> (
          let _divided_by : dim_var list = divided_by in
          match (numerator, divided_by, lub) with
          | Num_elems 1, vs, _ when is_stage5_up stage ->
              ( no_further_axes ~guess:false ()
                :: List.map vs ~f:(fun v ->
                    let d2 = get_dim ~d:1 ~proj_id:54 () in
                    Dim_eq { d1 = Var v; d2; origin }),
                env )
          | Num_elems d, [], None when d <> 1 && is_stage3_up stage ->
              let dim = get_dim ~d ~proj_id:55 () in
              ([ Row_eq { r1; r2 = { dims = [ dim ]; bcast = Broadcastable; prov }; origin } ], env)
          | Num_elems d, [], Some { dims; _ } when d <> 1 && last_dim_is dims (( = ) d) ->
              let dim = get_dim ~d ~proj_id:56 () in
              ([ Row_eq { r1; r2 = { dims = [ dim ]; bcast = Broadcastable; prov }; origin } ], env)
          | Num_elems _, [], Some lub ->
              let ineqs, env =
                apply_row_constraint ~depth:(depth + 1) stage origin
                  (if terminal then lub else r)
                  constr env
              in
              List.fold ineqs ~init:([], env) ~f:(fun (ineqs, env) ineq ->
                  match ineq with
                  | Rows_constr { r = rows; constr; origin } ->
                      let ineqs', env =
                        eliminate_rows_constraint ~depth stage origin ~lub:None rows constr env
                      in
                      (ineqs @ ineqs', env)
                  | ineq -> ([ ineq ], env))
          | Num_elems d, [ dv ], None when is_stage4_up stage ->
              ( no_further_axes ~guess:true ()
                :: [ Dim_eq { d1 = Var dv; d2 = get_dim ~d (); origin } ],
                env )
          | Num_elems d, [ v ], Some ({ dims; _ } as r2)
            when last_dim_is dims (fun d2 -> d % d2 = 0) ->
              let d2 = match List.last dims with Some (Dim { d; _ }) -> d | _ -> assert false in
              let row_eq =
                if d = d2 && is_stage5_up stage then no_further_axes ~guess:false ()
                else Row_eq { r1; r2; origin }
              in
              (row_eq :: [ Dim_eq { d1 = Var v; d2 = get_dim ~d:(d / d2) (); origin } ], env)
          | Strided_var { coeff; var; denom }, [], None
            when is_stage5_up stage
                 && (Utils.safe_force coeff > denom || denom % Utils.safe_force coeff <> 0) ->
              let coeff = Utils.safe_force coeff in
              let gcd = Utils.gcd coeff denom in
              let d = denom / gcd in
              let d2 = get_dim ~d () in
              let d3 = get_dim ~d:(coeff / gcd) () in
              (* opt_row_error (); *)
              ( [
                  Dim_eq { d1 = Var var; d2; origin };
                  Row_eq { r1; r2 = { dims = [ d3 ]; bcast = Broadcastable; prov }; origin };
                ],
                env )
          | Strided_var { coeff; var; denom }, [], _
            when is_stage6_up stage && denom % Utils.safe_force coeff = 0 ->
              let d2 = get_dim ~d:(denom / Utils.safe_force coeff) () in
              if dim_var_is_in_param var env then
                raise
                @@ Shape_error
                     ( "You forgot to specify the hidden dimension(s) 3",
                       [ Dim_mismatch [ Var var ] ] )
              else ([ Dim_eq { d1 = Var var; d2; origin }; no_further_axes ~guess:true () ], env)
          | ( Strided_var { coeff; var = _; denom },
              [],
              Some ({ dims = lub_dims; bcast = _; prov = lub_prov } as lub) )
            when is_stage5_up stage && Utils.safe_force coeff > denom -> (
              (* Check if coeff > denom * product of known dimensions of the LUB *)
              match collect_factors lub_dims with
              | Some (known_product, []) ->
                  let coeff_val = Utils.safe_force coeff in
                  if coeff_val > denom * known_product then
                    ([ Row_eq { r1; r2 = lub; origin } ], env)
                  else
                    (* Equate the row variable to the dimensions of the LUB *)
                    ( [
                        Row_eq
                          {
                            r1;
                            r2 = { dims = lub_dims; bcast = Broadcastable; prov = lub_prov };
                            origin;
                          };
                      ],
                      env )
              | _ -> keep_constr ())
          | Strided_var { coeff; var; denom }, _, _ when is_stage5_up stage ->
              let _var : dim_var = var in
              let _coeff : int = Utils.safe_force coeff in
              let _denom : int = denom in
              keep_constr ()
          | _ -> keep_constr ())
      | Exact dims -> ([ Row_eq { r1; r2 = { dims; bcast = Broadcastable; prov }; origin } ], env)
      | Unconstrained -> ([], env))

let%track5_sexp close_row_terminal ~(stage : stage) ~is_param origin env
    ({ dims; bcast; prov } as _r : row) : constraint_ list =
  let suffix () = List.map dims ~f:(fun d -> Terminal_dim (is_param, d, origin)) in
  (* TODO: can this be simplified? Should we return the environment? *)
  match bcast with
  | Broadcastable -> if is_stage6_up stage then [] else suffix ()
  | Row_var { v; beg_dims } -> (
      let term_dims () =
        List.map beg_dims ~f:(fun d -> Terminal_dim (is_param, d, origin)) @ suffix ()
      in
      let r1 : row = row_of_var v prov in
      let no_further_axes =
        Row_eq { r1; r2 = { dims = []; bcast = Broadcastable; prov }; origin }
      in
      match find_row env.row_env v with
      | Some (Bounds_row { is_in_param; lub = None; constr = Unconstrained; _ })
        when is_stage4_up stage ->
          if (is_param || is_in_param) && not (is_safe_to_guess v) then
            raise @@ Shape_error ("You forgot to specify the hidden dimension(s) 4", [])
          else (
            [%log6 "terminal row: closing", (_r : row)];
            no_further_axes :: term_dims ())
      | Some (Bounds_row { lub = None; constr; _ })
        when is_stage2_up stage && not (equal_row_constraint constr Unconstrained) ->
          let ineqs, _env =
            (* This is the constraint on the row variable, not on the original row. *)
            try
              eliminate_row_constraint ~depth:0 stage origin r1 ~terminal:true ~lub:None constr env
            with Shape_error (s, trace) -> raise @@ Shape_error (s, Row_mismatch [ r1 ] :: trace)
          in
          (* FIXME: at which stage should we drop the terminal row? *)
          let keep_terminal =
            if is_stage6_up stage then [] else [ Terminal_row (is_param, r1, origin) ]
          in
          ineqs @ term_dims () @ keep_terminal
      | Some (Solved_row _) -> assert false
      | Some (Bounds_row { lub = Some _; constr = Total_elems { numerator = Num_elems 1; _ }; _ })
        when is_stage3_up stage ->
          term_dims ()
      | Some (Bounds_row { lub = Some lub; origin = lub_origin; _ }) when is_stage3_up stage ->
          let origin = merge_origins origin lub_origin in
          Row_eq { r1; r2 = lub; origin } :: term_dims ()
      | _ when is_stage6_up stage -> []
      | _ ->
          [%log6 "terminal row: keeping", (_r : row), "as", (r1 : row)];
          Terminal_row (is_param, r1, origin) :: term_dims ())

let%debug5_sexp eliminate_dim_entry stage origin v ~lub constr =
  match (lub, constr) with
  | Some (Dim { d; _ } as lub), At_least_dim d2 when d2 > d ->
      raise
      @@ Shape_error
           ( [%string "dereferenced at dimension %{d2#Int}, higher than use site"],
             [ Dim_mismatch [ lub; Var v ] ] )
  | Some _, At_least_dim 1 ->
      (* Direct access at 0 is a strong heuristic for dimension 1 axis (e.g. result of a
         reduction). *)
      if is_stage7 stage then Some (Dim_eq { d1 = Var v; d2 = get_dim ~d:1 ~proj_id:57 (); origin })
      else None
  | Some lub, (At_least_dim _ | Unconstrained_dim) when is_stage6_up stage ->
      Some (Dim_eq { d1 = Var v; d2 = lub; origin })
  | None, At_least_dim d when is_stage7 stage ->
      Some (Dim_eq { d1 = Var v; d2 = get_dim ~d ~proj_id:58 (); origin })
  | None, _ when is_stage7 stage ->
      Some (Dim_eq { d1 = Var v; d2 = get_dim ~d:1 ~proj_id:59 (); origin })
  | _ -> None

let%track5_sexp process_shape_row ~(stage : stage) origin env ({ dims; bcast; prov } as r : row) :
    constraint_ list * _ =
  let final = is_stage7 stage in
  let rec finalize_upper_lower_bound = function
    | Dim _ -> []
    | Affine { over; conv = None; _ } -> finalize_upper_lower_bound over
    | Affine { over; conv = Some { kernel; _ }; _ } ->
        finalize_upper_lower_bound over @ finalize_upper_lower_bound kernel
    | Var v -> (
        match find_dim env.dim_env v with
        | Some (Bounds_dim { is_in_param = true; _ }) when final ->
            raise
            @@ Shape_error
                 ("You forgot to specify the hidden dimension(s) 5", [ Row_mismatch [ r ] ])
        | Some (Bounds_dim { lub; constr; has_uniq_constr_unless; _ })
          when is_stage4_up stage && can_guess_dim_to_one env has_uniq_constr_unless ->
            Option.to_list @@ eliminate_dim_entry stage origin v ~lub constr
        | Some (Solved_dim _) -> assert false
        | Some (Bounds_dim { has_uniq_constr_unless; _ })
          when final && can_guess_dim_to_one env has_uniq_constr_unless ->
            [ Dim_eq { d1 = Var v; d2 = get_dim ~d:1 ~proj_id:62 (); origin } ]
        | None when final -> [ Dim_eq { d1 = Var v; d2 = get_dim ~d:1 ~proj_id:60 (); origin } ]
        | _ -> [])
  in
  let rec has_dim_var = function
    | Dim _ -> false
    | Affine { over; conv = None; _ } -> has_dim_var over
    | Affine { over; conv = Some { kernel; _ }; _ } -> has_dim_var over || has_dim_var kernel
    | Var _ -> true
  in
  let process_dims dims = List.concat_map dims ~f:finalize_upper_lower_bound in
  match bcast with
  | Broadcastable ->
      let keep =
        if (not final) && List.exists dims ~f:has_dim_var then [ Shape_row (r, origin) ] else []
      in
      (keep @ process_dims dims, env)
  | Row_var { v; beg_dims } -> (
      let dim_eqs = process_dims beg_dims @ process_dims dims in
      let r1 : row = row_of_var v prov in
      match find_row env.row_env v with
      | Some (Bounds_row { lub = Some lub; constr = Unconstrained; _ }) when is_stage6_up stage ->
          (Row_eq { r1; r2 = lub; origin } :: dim_eqs, env)
      | Some (Bounds_row { constr = Unconstrained; _ }) when not final ->
          (Shape_row (r, origin) :: dim_eqs, env)
      | Some (Bounds_row { constr = Unconstrained; _ }) when final ->
          (Row_eq { r1; r2 = { dims = []; bcast = Broadcastable; prov }; origin } :: dim_eqs, env)
      | Some
          (Bounds_row
             {
               lub =
                 Some ({ dims = [] | [ Dim { d = 1; _ } ]; bcast = Broadcastable; prov = _ } as lub);
               _;
             }) ->
          (* That's a minimal / bottom value for a row. *)
          (Row_eq { r1; r2 = lub; origin } :: dim_eqs, env)
      | Some (Bounds_row { lub; constr; _ }) ->
          let ineqs, env =
            try eliminate_row_constraint ~depth:0 stage origin r1 ~terminal:false ~lub constr env
            with Shape_error (s, trace) -> raise @@ Shape_error (s, Row_mismatch [ r1 ] :: trace)
          in
          let keep = if not final then [ Shape_row (r, origin) ] else [] in
          (keep @ ineqs @ dim_eqs, env)
      | Some (Solved_row _) -> assert false
      | _ when final ->
          (Row_eq { r1; r2 = { dims = []; bcast = Broadcastable; prov }; origin } :: dim_eqs, env)
      | _ -> (Shape_row (r, origin) :: dim_eqs, env))

let empty_env = { dim_env = Utils.Tree_map.empty; row_env = Utils.Tree_map.empty }

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
                         cur = [];
                         subr = [];
                         lub = None;
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
    | { bcast = Row_var { v; _ }; _ } -> (
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
                         cur = [];
                         subr = [];
                         lub = None;
                         constr = Unconstrained;
                         origin = [];
                       });
            }
        | Some (Bounds_row b) ->
            let b = Bounds_row { b with is_in_param = true } in
            { env with row_env = add_row env.row_env ~key:v ~data:b }
        | _ -> env)
    | _ -> env

let%debug4_sexp solve_inequalities ~(stage : stage) (ineqs : constraint_ list) env :
    constraint_ list * _ =
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
      | Dim_ineq { cur; subr; origin; _ } ->
          let _ineq : constraint_ = ineq in
          let cur = subst_dim env cur and subr = subst_dim env subr in
          solve_dim_ineq ~stage origin ~cur ~subr env
      | Row_ineq { cur; subr; origin } ->
          let _ineq : constraint_ = ineq in
          let cur = subst_row env cur and subr = subst_row env subr in
          solve_row_ineq ~stage origin ~cur ~subr env
      | Dim_constr { d; constr; origin } ->
          let d = subst_dim env d in
          let extras, constr = apply_dim_constraint ~source:Direct ~stage d constr env in
          let env =
            match (constr, d) with
            | Unconstrained_dim, _ | _, Dim _ | _, Affine _ -> env
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
                          lub = None;
                          cur = [];
                          subr = [];
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
              eliminate_rows_constraint ~depth:0 stage origin ~lub:None substituted_rows constr env
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

let rec row_to_labels env =
  let rec f = function
    | Dim { label = Some l; _ } -> l
    | Dim { label = None; _ } -> ""
    | Var v -> (
        match find_dim env.dim_env v with
        | None | Some (Bounds_dim _) -> Option.value v.name ~default:""
        | Some (Solved_dim dim) -> f dim)
    | Affine _ -> ""
  in
  function
  | { dims; bcast = Row_var { v; beg_dims }; prov } -> (
      match find_row env.row_env v with
      | None | Some (Bounds_row _) -> Array.of_list_map (beg_dims @ dims) ~f
      | Some (Solved_row { dims = dims2; bcast = Broadcastable; _ }) ->
          row_to_labels env { dims = beg_dims @ dims2 @ dims; bcast = Broadcastable; prov }
      | Some (Solved_row { dims = dims2; bcast = Row_var { v = v2; beg_dims = beg_dims2 }; _ }) ->
          row_to_labels env
            {
              dims = dims2 @ dims;
              bcast = Row_var { v = v2; beg_dims = beg_dims @ beg_dims2 };
              prov;
            })
  | { dims; bcast = Broadcastable; prov = _ } -> Array.of_list_map dims ~f

(** *** Projection inference *** *)

let fresh_row_proj r =
  let rec fresh_dim = function
    | Dim { d; label; proj_id = _ } -> Dim { d; label; proj_id = Some (Proj_id.fresh ()) }
    | Var _ as d -> d
    | Affine { stride; over; conv; stride_offset } ->
        let conv =
          Option.map conv ~f:(fun { dilation; kernel; use_padding } ->
              { dilation; kernel = fresh_dim kernel; use_padding })
        in
        Affine { stride; over = fresh_dim over; conv; stride_offset }
  in
  let bcast =
    match r.bcast with
    | Row_var { v; beg_dims } -> Row_var { v; beg_dims = List.map beg_dims ~f:fresh_dim }
    | Broadcastable -> Broadcastable
  in
  { r with dims = List.map r.dims ~f:fresh_dim; bcast }

let populate_dim_proj_in_solved env =
  let rec fresh_dim = function
    | Dim { d; label; proj_id = None } -> Dim { d; label; proj_id = Some (Proj_id.fresh ()) }
    | (Dim _ | Var _) as d -> d
    | Affine { stride; over; conv; stride_offset } ->
        let conv =
          Option.map conv ~f:(fun { dilation; kernel; use_padding } ->
              { dilation; kernel = fresh_dim kernel; use_padding })
        in
        Affine { stride; over = fresh_dim over; conv; stride_offset }
  in
  let fresh_row ({ dims; bcast; prov } : row) : row =
    let dims = List.map dims ~f:fresh_dim in
    let bcast =
      match bcast with
      | Row_var { v; beg_dims } -> Row_var { v; beg_dims = List.map beg_dims ~f:fresh_dim }
      | Broadcastable -> Broadcastable
    in
    { dims; bcast; prov }
  in
  let f_dim = function Solved_dim dim -> Solved_dim (fresh_dim dim) | entry -> entry in
  let f_row = function Solved_row row -> Solved_row (fresh_row row) | entry -> entry in
  {
    dim_env = Utils.Tree_map.map env.dim_env ~f:f_dim;
    row_env = Utils.Tree_map.map env.row_env ~f:f_row;
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
    | Dim { d = 1; label = _; proj_id = None } ->
        (* d=1 dims created during constraint solving don't need iteration *)
        Solved (Fixed_idx 0)
    | Dim { d; label; proj_id = None } ->
        raise
        @@ Shape_error
             ( "to_proj: Dim without proj_id (d=" ^ Int.to_string d ^ ", label="
               ^ Option.value label ~default:"None"
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
        | Dim { d = 1; label = _; proj_id = None } ->
            (* d=1 dims created during constraint solving don't need iteration *)
            Solved (Fixed_idx 0)
        | Dim { d; label; proj_id = None } ->
            raise
            @@ Shape_error
                 ( "to_proj (subst): Dim without proj_id (d=" ^ Int.to_string d ^ ", label="
                   ^ Option.value label ~default:"None"
                   ^ ")",
                   [] )
        | Var v when Map.mem proj_axis_env v -> Solved (Map.find_exn proj_axis_env v)
        | Var v -> Var v
        | Affine _ as affine -> to_proj affine)
  in
  let rec expand_dims = function
    | { dims; bcast = Row_var { v; beg_dims }; _ }
      when Utils.Tree_map.mem ~compare:compare_row_var ~key:v env.row_env -> (
        match find_row env.row_env v with
        | Some (Solved_row r) ->
            let more_dims = expand_dims r in
            beg_dims @ more_dims @ dims
        | _ -> dims)
    | { dims; _ } -> dims
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
    | Dim_ineq { cur = _; subr = Dim ({ d = 1; proj_id = Some proj_id; _ } as solved_dim); _ } ->
        [ Proj_eq (Proj (proj_id, solved_dim), Solved (Fixed_idx 0)) ]
    | Dim_eq { d1; d2; origin = _ } | Dim_ineq { cur = d1; subr = d2; _ } ->
        [ Proj_eq (to_proj d1, to_proj d2) ]
    | Row_eq { r1; r2; origin = _ } -> match_rows ~with_broadcasting:false r1 r2
    | Row_ineq { cur = r1; subr = r2; origin = _ } ->
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
            offset := !offset + (stride * over_offset));

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
        { stride; over; conv = Some { dilation; kernel; kernel_size; use_padding }; stride_offset; target_id }
      -> (
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
            offset := !offset + (stride * over_offset));

        (match kernel_idx with
        | Idx.Fixed_idx i -> offset := !offset + (dilation * i)
        | Idx.Sub_axis -> ()
        | Idx.Iterator s -> symbols := (dilation, s) :: !symbols
        | Idx.Affine { symbols = kernel_syms; offset = kernel_offset } ->
            symbols := List.map kernel_syms ~f:(fun (c, s) -> (dilation * c, s)) @ !symbols;
            offset := !offset + (dilation * kernel_offset));

        (* Subtract padding if use_padding is true *)
        let offset =
          if use_padding then (
            (* Left padding smaller than right when split needed *)
            let right_padding = (kernel_size + 1) / 2 in
            let left_padding = kernel_size - right_padding in
            let operation_padding = Ir.Ops.{ left = left_padding; right = right_padding } in

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
               | Some _ -> (* Resolved padding is sufficient *) ()
               | None -> (
                   (* Update inferred padding to be sufficient for this operation *)
                   match Hashtbl.find proj_env.inferred_padding repr with
                   | Some existing_pad
                     when operation_padding.left > existing_pad.left
                          || operation_padding.right > existing_pad.right ->
                       let updated_pad =
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
    | Proj_eq (Solved idx, (Conv_input _ as conv_input))
    | Proj_eq ((Conv_input _ as conv_input), Solved idx) ->
        verify_when_solved1 := (idx, conv_input) :: !verify_when_solved1
    | Proj_eq
        ( (Conv_input { stride = stride1; over = over1; _ } as conv_input1),
          (Conv_input { stride = stride2; over = over2; _ } as conv_input2) )
      when stride1 = stride2 ->
        loop (Proj_eq (over1, over2));
        if equal_proj conv_input1 conv_input2 then ()
        else verify_when_solved2 := (conv_input1, conv_input2) :: !verify_when_solved2
    | Proj_eq ((Conv_input _ as conv_input1), (Conv_input _ as conv_input2)) ->
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
            let idx = Idx.(Iterator (get_symbol ())) in
            Hashtbl.add_exn v_env ~key:v ~data:(Solved idx)
        | Some proj -> loop @@ Iterated proj)
  in
  List.iter eqs ~f:loop;
  let projs = ref @@ Map.empty (module Proj_id) in
  List.iter !p_solved ~f:(fun (p, idx) ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      Utils.mref_add projs ~key:repr ~data:idx ~or_:(fun idx2 ->
          if not @@ Idx.equal_axis_index idx idx2 then
            raise
            @@ Shape_error
                 ("Multiple constraints on the same projection", [ Index_mismatch [ idx; idx2 ] ])));
  let product_dim = ref @@ Map.empty (module Proj_id) in
  List.iter !p_dims ~f:(fun (p, d) ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      if Idx.iterated d && (not @@ Map.mem !projs repr) then
        Utils.mref_add product_dim ~key:repr ~data:d ~or_:(fun d2 ->
            (* TODO: consider updating padding *)
            if d <> d2 then
              raise
              @@ Shape_error
                   ( [%string
                       "Conflicting dimensions for the same projection: %{p#Proj_id} %{d#Int} \
                        %{d2#Int}"],
                     [] )));
  Map.iteri !product_dim ~f:(fun ~key:p ~data:_ ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      Utils.mref_add_missing projs repr ~f:(fun () -> Idx.(Iterator (get_symbol ()))));

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
  | Dim { d; _ } when not @@ Idx.iterated d -> None
  | Dim { proj_id = Some proj_id; d; _ } -> (
      let repr = proj_repr proj_env proj_id in
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

let proj_to_iterator_exn proj_env p =
  match Map.find_exn proj_env.proj_to_index (proj_repr proj_env p) with
  | Iterator s -> s
  | _ -> invalid_arg "proj_to_iterator_exn"

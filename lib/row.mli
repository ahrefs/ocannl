(** The row type, shape inference related types and constraint solving. *)

open Base

type kind = [ `Batch | `Input | `Output ] [@@deriving equal, compare, sexp, hash, variants]
type dim_var [@@deriving equal, hash, compare, sexp]
type dim_cmp
type dim_var_set = (dim_var, dim_cmp) Base.Set.t [@@deriving equal, sexp]
type 'a dim_map = (dim_var, 'a, dim_cmp) Base.Map.t [@@deriving equal, sexp]

val get_var : ?label:string -> unit -> dim_var
val dim_var_set_empty : dim_var_set
val dim_map_empty : 'a dim_map

(** A single axis in a shape. *)
type dim = Var of dim_var | Dim of { d : int; label : string option; proj_id : int option }
[@@deriving equal, hash, compare, sexp, variants]

val get_dim : d:int -> ?label:string -> unit -> dim
val dim_to_int_exn : dim -> int
val dim_to_string : [> `Only_labels ] -> dim -> string

type row_id [@@deriving sexp, compare, equal, hash]
type row_cmp

val row_id : sh_id:int -> kind:kind -> row_id

type row_var [@@deriving sexp, compare, equal, hash]

val get_row_var : unit -> row_var

(** A bcast specifies how axes of a single kind in a shape (i.e. the row) can adapt to other shapes. *)
type bcast =
  | Row_var of row_var  (** The row can be inferred to have more axes. *)
  | Broadcastable  (** The shape does not have more axes of this kind, but is "polymorphic". *)
[@@deriving equal, hash, compare, sexp, variants]

type t = { dims : dim list; bcast : bcast; id : row_id } [@@deriving equal, hash, compare, sexp]

val dims_label_assoc : t -> (string * dim) list

type environment [@@deriving sexp]
type error_trace = ..

type error_trace +=
  | Row_mismatch of t list
  | Dim_mismatch of dim list
  | Index_mismatch of Arrayjit.Indexing.axis_index list

val sexp_of_error_trace : error_trace -> Sexp.t

exception Shape_error of string * error_trace list [@@deriving sexp_of]

type dim_constraint = Unconstrained_dim | At_least_dim of int
[@@deriving equal, hash, compare, sexp, variants]

type row_constraint =
  | Unconstrained
  | Total_elems of { numerator : int; divided_by : dim_var_set }
      (** The row or remainder of a row, inclusive of the further row spec, has this many elements. *)
[@@deriving equal, hash, compare, sexp, variants]

(** An entry implements inequalities [cur >= v >= subr] and/or an equality [v = solved]. [cur] and [subr] must
    be sorted using the [@@deriving compare] comparison. *)
type dim_entry =
  | Solved_dim of dim
  | Bounds_dim of { cur : dim_var list; subr : dim_var list; lub : dim option; constr : dim_constraint }
[@@deriving sexp]

type row_entry =
  | Solved_row of t
  | Bounds_row of { cur : row_var list; subr : row_var list; lub : t option; constr : row_constraint }
[@@deriving sexp]

type constraint_ =
  | Dim_eq of { d1 : dim; d2 : dim }
  | Row_eq of { r1 : t; r2 : t }
  | Reverse_eq of { r1 : t; r2 : t }
      (** If [r1] does not have a row variable, same as [Row_eq {r1={r1 with dims=List.rev r1.dims}; r2}]. *)
  | Dim_ineq of { cur : dim; subr : dim }
  | Row_ineq of { cur : t; subr : t }
  | Dim_constr of { d : dim; constr : dim_constraint }
  | Row_constr of { r : t; constr : row_constraint }
  | Terminal_dim of dim
  | Terminal_row of t
[@@deriving compare, equal, sexp, variants]

type stage = Stage1 | Stage2 | Stage3 | Stage4 [@@deriving sexp, equal, compare, variants]

val subst_row : environment -> t -> t
val unify_row : reverse:bool -> stage:stage -> t * t -> environment -> constraint_ list * environment
val empty_env : environment

val solve_inequalities :
  stage:stage ->
  active_update_rows:t list ->
  constraint_ list ->
  environment ->
  constraint_ list * environment

val row_to_labels : environment -> t -> string array

type proj [@@deriving compare, equal, sexp]
type proj_env [@@deriving sexp]

val fresh_row_proj : t -> t

type proj_equation =
  | Proj_eq of proj * proj  (** Two projections are the same, e.g. two axes share the same iterator. *)
  | Iterated of proj
      (** The projection needs to be an iterator even if an axis is not matched with another axis, e.g. for
          broadcasted-to axes of a tensor assigned a constant. *)
[@@deriving compare, equal, sexp]

val get_proj_equations :
  constraint_ list -> Arrayjit.Indexing.axis_index dim_map -> environment -> proj_equation list

val solve_proj_equations : proj_equation list -> proj_env
val get_proj_index : proj_env -> dim -> Arrayjit.Indexing.axis_index
val get_product_proj : proj_env -> dim -> (int * int) option
val proj_to_iterator : proj_env -> int -> Arrayjit.Indexing.symbol

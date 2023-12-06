(** The row type, shape inference related types and constraint solving. *)

open Base

type kind = [ `Batch | `Input | `Output ] [@@deriving equal, compare, sexp, hash, variants]
type dim_var [@@deriving equal, hash, compare, sexp]
type dim_cmp
type 'a dim_map = (dim_var, 'a, dim_cmp) Base.Map.t [@@deriving equal, sexp]

val get_var : ?label:string -> unit -> dim_var
val dim_map_empty : (dim_var, 'a, dim_cmp) Map.t

type 'a dim_hashtbl = (dim_var, 'a) Base.Hashtbl.t [@@deriving sexp]

val dim_hashtbl : unit -> 'a dim_hashtbl

(** A single axis in a shape. *)
type dim = Var of dim_var | Dim of { d : int; label : string option; proj_id : int option }
[@@deriving equal, hash, compare, sexp, variants]

val get_dim : d:int -> ?label:string -> unit -> dim
val dim_to_int_exn : dim -> int
val dim_to_string : [> `Only_labels ] -> dim -> string

type row_id [@@deriving sexp, compare, equal, hash]
type row_cmp

val row_id : sh_id:int -> kind:kind -> row_id
val row_map_empty : (row_id, 'a, row_cmp) Map.t

(** A bcast specifies how axes of a single kind in a shape (i.e. the row) can adapt to other shapes. *)
type bcast =
  | Row_var of int  (** The row can be inferred to have more axes. *)
  | Broadcastable  (** The shape does not have more axes of this kind, but is "polymorphic". *)
[@@deriving equal, hash, compare, sexp, variants]

val get_row_var : unit -> bcast

type dims_constraint =
  | Unconstrained
  | Total_elems of int  (** The shape-kind, inclusive of the further row spec, has this many elements. *)
[@@deriving equal, hash, compare, sexp, variants]

type t = { dims : dim list; constr : dims_constraint; bcast : bcast; id : row_id }
[@@deriving equal, hash, compare, sexp]

val dims_label_assoc : t -> (string * dim) list

type 'a entry = { cur : 'a list; subr : 'a list; solved : 'a option } [@@deriving sexp]
type dim_env = dim entry dim_map [@@deriving sexp]
type row_env = t entry Map.M(Int).t [@@deriving sexp]

type environment = private {
  dim_env : dim_env;
  row_env : row_env;
  dim_rev_elim_order : dim_var list;
  row_rev_elim_order : int list;
}
[@@deriving sexp]

type error_trace = ..

type error_trace +=
  | Row_mismatch of t list
  | Dim_mismatch of dim list
  | Index_mismatch of Arrayjit.Indexing.axis_index list

val sexp_of_error_trace : error_trace -> Sexp.t

exception Shape_error of string * error_trace list [@@deriving sexp_of]

val subst_dim : environment -> dim -> dim
val occurs_dim : dim_var -> dim -> bool
val subst_row : environment -> t -> t
val occurs_row : int -> t -> bool
val unify_dim : dim * dim -> environment -> environment
val update_dim : is_complete:bool -> dim_var -> ?cur:dim -> ?subr:dim -> environment -> environment
val unify_row : t * t -> environment -> environment
val update_row : is_complete:bool -> int -> ?cur:t -> ?subr:t -> environment -> environment
val apply_constraint : t -> environment -> environment
val add_dim_ineq : is_complete:bool -> cur:dim -> subr:dim -> environment -> environment
val add_row_ineq : is_complete:bool -> cur:t -> subr:t -> environment -> environment
val empty_env : environment

type inequality =
  | Dim_eq of { d1 : dim; d2 : dim }
  | Row_eq of { r1 : t; r2 : t }
  | Dim_ineq of { cur : dim; subr : dim }
  | Row_ineq of { cur : t; subr : t }
[@@deriving compare, equal, sexp]

val solve_inequalities : is_complete:bool -> inequality list -> environment -> environment
val row_to_labels : environment -> t -> string array

type proj [@@deriving compare, equal, sexp]
type proj_env [@@deriving sexp]

val fresh_proj : unit -> int
val fresh_row_proj : t -> t

val get_proj_equations :
  inequality list -> Arrayjit.Indexing.axis_index dim_map -> environment -> (proj * proj) list

val solve_proj_equations : (proj * proj) list -> proj_env
val get_proj_index : proj_env -> dim -> Arrayjit.Indexing.axis_index
val get_product_proj : proj_env -> dim -> (int * int) option
val proj_to_iterator : proj_env -> int -> Arrayjit.Indexing.symbol

(** The row type, shape inference related types and constraint solving. *)

open Base

type axis_padding = Ir.Ndarray.axis_padding [@@deriving equal, sexp]
type kind = [ `Batch | `Input | `Output ] [@@deriving equal, compare, sexp, hash, variants]
type dim_var [@@deriving equal, hash, compare, sexp]
type proj_id [@@deriving equal, hash, compare, sexp]
type dim_cmp
type dim_var_set = (dim_var, dim_cmp) Base.Set.t [@@deriving equal, sexp]
type 'a dim_map = (dim_var, 'a, dim_cmp) Base.Map.t [@@deriving equal, sexp]
type proj_cmp
type proj_var_set = (proj_id, proj_cmp) Base.Set.t [@@deriving equal, sexp]
type 'a proj_map = (proj_id, 'a, proj_cmp) Base.Map.t [@@deriving equal, sexp]

val get_var : ?label:string -> unit -> dim_var
val dim_var_set_empty : dim_var_set
val dim_map_empty : 'a dim_map
val proj_var_set_empty : proj_var_set
val proj_map_empty : 'a proj_map
val use_padding : bool ref

type solved_dim = { d : int; label : string option; proj_id : proj_id option }
[@@deriving equal, hash, compare, sexp]
(** A single axis in a shape. *)

type dim =
  | Var of dim_var
  | Dim of solved_dim
  | Conv_input of { stride : int; output : dim; dilation : int; kernel : dim }
      (** The offset is implicit, automatically derived. If [!use_padding] is [true], the offset is
          the left part of the dimensionality-preserving symmetric padding, otherwise it is 0. *)
[@@deriving equal, hash, compare, sexp, variants]

val get_dim : d:int -> ?label:string -> unit -> dim
val dim_to_int_exn : dim -> int

type print_style = Only_labels | Axis_size | Axis_number_and_size | Projection_and_size
[@@deriving equal, compare, sexp]

val solved_dim_to_string : print_style -> solved_dim -> string
val dim_to_string : print_style -> dim -> string

type row_id [@@deriving sexp, compare, equal, hash]
type row_cmp

val row_id : sh_id:int -> kind:kind -> row_id

type row_var [@@deriving sexp, compare, equal, hash]

val get_row_var : unit -> row_var

(** A bcast specifies how axes of a single kind in a shape (i.e. the row) can adapt to other shapes.
*)
type bcast =
  | Row_var of { v : row_var; beg_dims : dim list }
      (** The row can be inferred to have more axes. *)
  | Broadcastable  (** The shape does not have more axes of this kind, but is "polymorphic". *)
[@@deriving equal, hash, compare, sexp, variants]

type t = { dims : dim list; bcast : bcast; id : row_id } [@@deriving equal, hash, compare, sexp]

val dims_label_assoc : t -> (string * dim) list

type environment [@@deriving sexp]
type error_trace = ..

type error_trace +=
  | Row_mismatch of t list
  | Dim_mismatch of dim list
  | Index_mismatch of Ir.Indexing.axis_index list

val sexp_of_error_trace : error_trace -> Sexp.t

exception Shape_error of string * error_trace list [@@deriving sexp_of]

type dim_constraint = Unconstrained_dim | At_least_dim of int
[@@deriving equal, hash, compare, sexp, variants]

type row_constraint =
  | Unconstrained
  | Total_elems of { nominator : int; divided_by : dim_var_set }
      (** The rows, inclusive of the further row spec, have this many elements. *)
  | Exact of dim list
      (** The concatenated rows have these axes. *)
[@@deriving equal, hash, compare, sexp, variants]

(** An entry implements inequalities [cur >= v >= subr] and/or an equality [v = solved]. [cur] and
    [subr] must be sorted using the [@@deriving compare] comparison. *)
type dim_entry =
  | Solved_dim of dim
  | Bounds_dim of {
      cur : dim_var list;
      subr : dim_var list;
      lub : dim option;
      constr : dim_constraint;
    }
[@@deriving sexp]

type row_entry =
  | Solved_row of t
  | Bounds_row of {
      cur : row_var list;
      subr : row_var list;
      lub : t option;
      constr : row_constraint;
    }
[@@deriving sexp]

type constraint_ =
  | Dim_eq of { d1 : dim; d2 : dim }
  | Row_eq of { r1 : t; r2 : t }
  | Dim_ineq of { cur : dim; subr : dim }
  | Row_ineq of { cur : t; subr : t }
  | Dim_constr of { d : dim; constr : dim_constraint }
  | Rows_constr of { r : t list; constr : row_constraint }
      (** The constraint applies to the concatenation of the rows. *)
  | Terminal_dim of dim
  | Terminal_row of t
[@@deriving compare, equal, sexp, variants]

type stage = Stage1 | Stage2 | Stage3 | Stage4 | Stage5 | Stage6 | Stage7
[@@deriving sexp, equal, compare]

val subst_row : environment -> t -> t
val unify_row : stage:stage -> t * t -> environment -> constraint_ list * environment
val empty_env : environment
val eliminate_variables : environment -> t -> constraint_ list

val solve_inequalities :
  stage:stage -> constraint_ list -> environment -> constraint_ list * environment

val row_to_labels : environment -> t -> string array

type proj [@@deriving compare, equal, sexp]
type proj_env [@@deriving sexp_of]

val fresh_row_proj : t -> t

type proj_equation =
  | Proj_eq of proj * proj
      (** Two projections are the same, e.g. two axes share the same iterator. *)
  | Iterated of proj
      (** The projection needs to be an iterator even if an axis is not matched with another axis,
          e.g. for broadcasted-to axes of a tensor assigned a constant. *)
[@@deriving compare, equal, sexp]

val get_proj_equations :
  constraint_ list -> Ir.Indexing.axis_index dim_map -> environment -> proj_equation list

val solve_proj_equations :
  proj_equation list ->
  resolved_padding:(proj_id, axis_padding) List.Assoc.t ->
  inferred_padding:(proj_id, axis_padding) List.Assoc.t ->
  proj_env

val get_proj_index : proj_env -> proj -> Ir.Indexing.axis_index
val get_dim_index : proj_env -> dim -> Ir.Indexing.axis_index
val get_product_proj : proj_env -> dim -> (proj_id * int) option

val proj_to_iterator_exn : proj_env -> proj_id -> Ir.Indexing.symbol
(** [proj_to_iterator_exn proj_env p] returns the iterator for [p] in [proj_env]. Raises
    [Invalid_argument] if [p] is not an iterator. *)

(** Symbols and projections used to lower tensor indexing. *)

open Base

type symbol = Symbol of int [@@deriving compare, equal, sexp, hash, variants]

module Symbol : sig
  type t = symbol

  val compare : t -> t -> int
  val equal : t -> t -> bool
  val t_of_sexp : Sexp.t -> t
  val sexp_of_t : t -> Sexp.t
  val hash_fold_t : Ppx_hash_lib.Std.Hash.state -> t -> Ppx_hash_lib.Std.Hash.state
  val hash : t -> Ppx_hash_lib.Std.Hash.hash_value

  include Comparator.S with type t := t
end

val get_symbol : unit -> symbol

val discarding_symbols : (unit -> 'a) -> 'a
(** [discarding_symbols f] runs [f] and then rewinds the symbol counter to where it was: for dry
    runs whose minted symbols are all discarded (a schedule probe), so they shift no later minted
    name. Sound only when no symbol [f] mints outlives it. *)

val symbol_ident : symbol -> string

type 'a environment = 'a Map.M(Symbol).t [@@deriving sexp]

type static_symbol = {
  static_symbol : symbol;
  mutable static_range : int option;
  mutable used_as_extent : bool;
  mutable used_as_slice : bool;
}
[@@deriving compare, equal, sexp, hash]

type 'a bindings = Empty | Bind of static_symbol * (int -> 'a) bindings [@@deriving sexp_of]

val bound_symbols : 'a bindings -> static_symbol list

type ('r, 'idcs, 'p1, 'p2) variadic =
  | Result of 'r
  | Param_idx of int ref * (int -> 'r, int -> 'idcs, 'p1, 'p2) variadic
  | Param_1 of 'p1 option ref * ('p1 -> 'r, 'idcs, 'p1, 'p2) variadic
  | Param_2 of 'p2 option ref * ('p2 -> 'r, 'idcs, 'p1, 'p2) variadic
  | Param_2f :
      ('p2f -> 'p2) * 'p2f option ref * ('p2 -> 'r, 'idcs, 'p1, 'p2) variadic
      -> ('r, 'idcs, 'p1, 'p2) variadic

type unit_bindings = (unit -> unit) bindings [@@deriving sexp_of]
type lowered_bindings = (static_symbol, int ref) List.Assoc.t [@@deriving sexp_of]

val apply : ('r, 'idcs, 'p1, 'p2) variadic -> 'r
val lowered_bindings : 'a bindings -> ('b, 'a, 'p1, 'p2) variadic -> lowered_bindings
val find_exn : lowered_bindings -> static_symbol -> int ref
val get_static_symbol : ?static_range:int -> (int -> 'a) bindings -> static_symbol * 'a bindings

val fresh_launch_param : unit -> int ref
(** A launch parameter's cell as backend linking mints it: UNASSIGNED (gh-ocannl-940), holding a
    value outside every legal one, which {!validate_bound_value} refuses naming the parameter. A
    caller assigns it through [find_exn bindings sym := v]; a caller that means 0 writes 0. *)

val launch_param_assigned : int ref -> bool
(** Whether the cell holds a value someone assigned, rather than the one {!fresh_launch_param} left
    there. *)

val validate_bound_value : ?width64:bool -> static_symbol -> int -> unit
(** Bind-time validation of one launch-parameter value: refuses an unassigned one (gh-ocannl-940), a
    negative one, one outside the declared range (below [range] for an index, up to [range]
    inclusive for a symbolic extent) and, unless [width64], one past the int32 index width. *)

val validate_lowered_bindings : ?width64:bool -> lowered_bindings -> unit
(** {!validate_bound_value} over every parameter, reporting all the unassigned ones in one message
    first. *)

val dims_to_string : ?with_axis_numbers:bool -> int array -> string

type affine_index = private { symbols : (int * symbol) list; offset : int }
[@@deriving compare, equal, sexp_of]

type axis_index =
  | Fixed_idx of int
  | Iterator of symbol
  | Affine of affine_index
  | Sub_axis
  | Concat of symbol list
[@@deriving compare, equal, sexp]

val affine : symbols:(int * symbol) list -> offset:int -> axis_index
(** Coalesce repeated symbols, remove zero terms, and use [Fixed_idx] or [Iterator] whenever
    possible. The private payload prevents bypassing this normalization. *)

val axis_index_mentions_symbol : symbol -> axis_index -> bool
val axis_index_mentions_any : symbol list -> axis_index -> bool

type str_osym_map = (string, symbol option, String.comparator_witness) Map.t

type projections_debug = { spec : string; derived_for : Sexp.t; trace : (string * int) list }
[@@deriving sexp]

val unique_debug_id : unit -> int

type component = (int * symbol) list [@@deriving compare, equal, sexp]

type projections = {
  components : component array;
  lhs_dims : int array;
  rhs_dims : int array array;
  project_lhs : axis_index array;
  project_rhs : axis_index array array;
  extent_syms : (symbol option * static_symbol) list;
  debug_info : projections_debug;
}
[@@deriving compare, equal, sexp]

val iterated : int -> bool
val all_iterators : projections -> symbol list
val iterator_sizes : projections -> int Map.M(Symbol).t
val coalesce_affine_terms : (int * symbol) list -> (int * symbol) list
val affine_injective : symbol_range:(symbol -> int) -> axis_index array -> bool
val prod_project_for : projections -> dims:int array -> axis_index array
val reflect_projection : dims:int array -> projection:axis_index array -> axis_index

type variable_ref = private {
  ref_label : string;
  mutable solved_dim : int option;
  mutable solved_sym : static_symbol option;
}
[@@deriving sexp_of, equal]
(** Private (gh-ocannl-1573): read and matched freely, written only through the setters below. A
    solution is reached once: at most one of [solved_dim], [solved_sym] is ever set, and a set field
    keeps its value. Shape inference is the only writer ([Shape]'s capture references wrap these
    setters with the session's accounting); the cell is shared, so an [Assignments.Embed_dim] built
    before inference observes the solution it later reaches. *)

val variable_ref : string -> variable_ref
(** A fresh, unsolved reference with the given label. *)

val set_solved_dim : variable_ref -> int -> unit
(** Records a concrete solution. Setting the value it already holds is a no-op; a different value,
    or a reference already solved to a symbol, raises. *)

val set_solved_sym : variable_ref -> static_symbol -> unit
(** Records a symbolic-extent solution, guarded like {!set_solved_dim}. *)

module Doc_helpers : sig
  val ( ^^ ) : PPrint.document -> PPrint.document -> PPrint.document
  val ( !^ ) : string -> PPrint.document
  val int : int -> PPrint.document
  val comma_sep : PPrint.document
  val pp_comma : unit -> PPrint.document
  val pp_symbol : symbol -> PPrint.document
  val pp_static_symbol : static_symbol -> PPrint.document
  val pp_axis_index : axis_index -> PPrint.document
  val pp_indices : axis_index array -> PPrint.document
end

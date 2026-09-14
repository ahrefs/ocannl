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
val validate_bound_value : ?width64:bool -> static_symbol -> int -> unit
val validate_lowered_bindings : ?width64:bool -> lowered_bindings -> unit
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

type variable_ref = {
  ref_label : string;
  mutable solved_dim : int option;
  mutable solved_sym : static_symbol option;
}
[@@deriving sexp_of, equal]

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

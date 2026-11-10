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

val in_symbol_ident_namespace : string -> bool
(** Whether a name is spelled like {!symbol_ident} of some symbol, alone or behind a [_] suffix:
    ["i"], one or more digits, then the end or ["_"] and anything. The C-family emitters declare
    loop indices and static-index parameters under {!symbol_ident} and derive names from it (a
    pool-chunked [Grid] loop's [i<n>_chunk], [i<n>_lo], [i<n>_hi]), so their node-name minter
    reserves this namespace (gh-ocannl-2103). *)

type 'a environment = 'a Map.M(Symbol).t [@@deriving sexp]

type static_symbol = private {
  static_symbol : symbol;
  mutable static_range : int option;
  mutable used_as_extent : bool;
  mutable used_as_slice : bool;
}
[@@deriving compare, equal, sexp, hash]
(** Private (gh-ocannl-1624): read and matched freely, built by {!make_static_symbol} or
    {!get_static_symbol}, and written only through the setters below. The roles ([used_as_extent],
    [used_as_slice]) are shape-inference state of a caller-retained symbol: after its birth a role
    is written only by {!commit_role}, which takes {!shape_writer}. The declared range is not
    inference state; {!set_static_range} writes it. *)

val make_static_symbol :
  ?static_range:int -> ?used_as:[ `Extent | `Slice ] -> symbol -> static_symbol
(** A static symbol over [symbol], holding the role [used_as] from birth (default: none). A role
    given at birth is not a write to a symbol anything else holds yet, so it needs no writer --
    hand-built IR uses it for a symbolic extent no shape inference claimed. *)

val set_static_range : static_symbol -> int option -> unit
(** Re-declares the symbol's range. The analysis caches key on it, so a changed range misses. *)

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

val set_launch_value :
  ?width64:bool -> caller:string -> lowered_bindings -> static_symbol -> int -> unit
(** [set_launch_value ~caller bs sym v] validates [v] ({!validate_bound_value}) and writes it into
    [sym]'s cell of [bs]; a symbol [bs] does not bind is refused, naming [caller]. *)

val resolve_launch_values :
  caller:string ->
  hint:string ->
  ranged:(static_symbol -> string option) ->
  (static_symbol * int) list ->
  static_symbol list ->
  lowered_bindings
(** The launch values of a routine that a library entry point compiles and runs on the caller's
    behalf (gh-ocannl-1629): a fresh cell per symbol, the caller's values written into it (a symbol
    it does not bind, or given twice, is refused; the values themselves are validated by the run,
    once lowering has committed the role -- index or extent -- each is validated against), then
    every parameter left unassigned set to 0 -- unless [ranged] names its kind (a batch-slice index,
    a ranged static index, a symbolic extent), which is refused instead, naming each one and its
    kind and ending with [hint]: 0 is its first slice or empty extent, the silent default
    gh-ocannl-940 refuses. Hand the result to {!copy_launch_values} once the routine is linked. *)

val copy_launch_values : from:lowered_bindings -> lowered_bindings -> unit
(** [copy_launch_values ~from bs] writes into each cell of [bs] the value [from] holds for the same
    symbol; every symbol of [bs] must be bound by [from]. *)

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
    keeps its value. Shape inference is the only writer: the setters take {!shape_writer}, which
    [Shape] holds, and its capture references wrap them with the session's accounting
    (gh-ocannl-1624). The cell is shared, so an [Assignments.Embed_dim] built before inference
    observes the solution it later reaches. *)

val variable_ref : string -> variable_ref
(** A fresh, unsolved reference with the given label. *)

type shape_writer
(** The capability to write shape-inference state held by this module's records: solve a
    {!variable_ref}, commit a static symbol's role (gh-ocannl-1624). A process has at most one: the
    first {!claim_shape_writer} returns it and every later one raises. In a program linking the
    tensor frontend, [Shape] claims it as it initializes, so the setters taking it are [Shape]'s
    alone -- a write through them could not skip the session's accounting (liveness of a rewound
    reference, the open set, the mutation serial). *)

val claim_shape_writer : unit -> shape_writer
(** Returns the process's {!shape_writer}; raises [Utils.User_error] when it was already claimed. *)

val set_solved_dim : shape_writer -> variable_ref -> int -> unit
(** Records a concrete solution. Setting the value it already holds is a no-op; a different value,
    or a reference already solved to a symbol, raises. *)

val set_solved_sym : shape_writer -> variable_ref -> static_symbol -> unit
(** Records a symbolic-extent solution, guarded like {!set_solved_dim}. *)

type variable_ref_snapshot
(** A reference's solution at a checkpoint, bound to that reference (gh-ocannl-903). *)

val snapshot_variable_ref : variable_ref -> variable_ref_snapshot

val restore_variable_ref : shape_writer -> variable_ref_snapshot -> unit
(** Puts the recorded solution back on its reference -- unsolving it if it was unsolved: a rewind's
    write, so it takes the writer. *)

val variable_ref_snapshot_changed : variable_ref_snapshot -> bool
(** Whether the reference's solution differs from the recorded one: a diagnostic. *)

val commit_role : shape_writer -> static_symbol -> [ `Extent | `Slice ] -> unit
(** Sets the symbol's role flag. Committing the role it holds is a no-op; the other role raises (a
    binding cannot be validated both inclusively and strictly). [Shape] calls it only from a
    successful [finish_inference], for the claims [Row.claim_symbol_role] counted. *)

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

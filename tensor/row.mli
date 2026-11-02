(** The row type, shape inference related types and constraint solving. *)

open Base

type axis_padding = Ir.Ops.axis_padding [@@deriving equal, sexp]
type kind = [ `Batch | `Input | `Output ] [@@deriving equal, compare, sexp, hash, variants]
type dim_var [@@deriving equal, hash, compare, sexp]
type proj_id [@@deriving equal, hash, compare, sexp]
type dim_cmp
type dim_var_set = (dim_var, dim_cmp) Base.Set.t [@@deriving equal, sexp]
type 'a dim_map = (dim_var, 'a, dim_cmp) Base.Map.t [@@deriving equal, sexp]
type proj_cmp
type proj_var_set = (proj_id, proj_cmp) Base.Set.t [@@deriving equal, sexp]
type 'a proj_map = (proj_id, 'a, proj_cmp) Base.Map.t [@@deriving equal, sexp]

val get_var : ?name:string -> unit -> dim_var
val dim_var_set_empty : dim_var_set
val dim_map_empty : 'a dim_map
val proj_var_set_empty : proj_var_set
val proj_map_empty : 'a proj_map

val bcast_if_1 : string
(** The reserved basis tag of the claim-free broadcast top. An axis tagged [bcast_if_1] broadcasts
    to any size while it remains size 1 ([1_(bcast_if_1)] is the top of the broadcast order), and is
    an ordinary inert atom at sizes > 1. Scalars and rank-broadening fill carry it; a user may also
    write it deliberately as an explicit stretch-when-1 affordance. *)

val default_basis : string
(** The basis tag the frontend supplies for any axis the user writes without naming a basis. It is
    an ordinary atom, incompatible with every other named basis (including [bcast_if_1]). *)

val is_reserved_basis : string -> bool
(** Whether a basis tag is one of the reserved (claim-free) tags [default_basis] / [bcast_if_1],
    i.e. carries no user-meaningful naming claim. *)

type solved_dim = { d : int; basis : string; proj_id : proj_id option }
[@@deriving equal, hash, compare, sexp]
(** A single axis in a shape. [basis] is the (total) semantic annotation on the dimension (e.g.
    ["rgb"] on a size-3 axis); two dimensions that must agree in size must also agree on [basis].
    Unannotated user axes carry [default_basis]; the broadcast top carries [bcast_if_1]. [proj_id]
    is used for projection inference, and abused for provenance tracking during shape inference. *)

type convolution = { dilation : int; kernel : dim; use_padding : bool }
[@@deriving equal, hash, compare, sexp]
(** Convolution parameters. The offset is implicit, automatically derived. If [use_padding] is
    [true], the offset is the left part of the dimensionality-preserving symmetric padding,
    otherwise it is 0. *)

and dim =
  | Var of dim_var
  | Dim of solved_dim
  | Sym of sym_dim
      (** A symbolic dimension: a launch-time-bound extent (gh-490). During shape inference it is a
          rigid constant -- equal only to a [Sym] with the same static symbol; it never unifies with
          a concrete [Dim] (but the broadcast top [1_(bcast_if_1)] broadcasts to it). It is
          materialized at its declared maximum extent ([static_range]) for Tnode dims and
          projections. *)
  | Affine of { stride : int; over : dim; conv : convolution option; stride_offset : int }
      (** An affine transformation of a dimension. When [conv] is [None], this is a simple strided
          dimension. When [conv] is [Some], this includes convolution parameters. Invariants:
          [stride > 0], [dilation > 0] (when present), [0 <= stride_offset],
          [stride_offset < stride] on an axis without a window, and [stride_offset = 0] on an
          unpadded window; a padded window takes any offset, a tap moved past its margins reading
          the padding value. [Shape] refuses an einsum spec that breaks them (gh-ocannl-1618,
          gh-ocannl-1657). *)
  | Concat of dim list  (** Concatenation of multiple dimensions into a single axis. *)

and sym_dim = { sym : Ir.Indexing.static_symbol; sym_basis : string; sym_proj_id : proj_id option }
[@@deriving equal, hash, compare, sexp]

val get_dim : d:int -> basis:string -> ?proj_id:int -> unit -> dim
(** Mint a dimension with an explicit (required) basis tag. *)

val get_bcast_dim : d:int -> ?proj_id:int -> unit -> dim
(** Mint the claim-free broadcast top [d_(bcast_if_1)] (at [d = 1] the top of the broadcast order;
    at sizes > 1 an inert atom). Use at scalar / rank-broadening / GLB-demotion sites. *)

val get_default_dim : d:int -> ?proj_id:int -> unit -> dim
(** Mint an unannotated user/derived atom [d_(default)]. *)

val get_sym_dim : ?basis:string -> ?proj_id:int -> Ir.Indexing.static_symbol -> dim
(** Mint a symbolic dimension from a static-indexing symbol. The symbol must have a declared
    [static_range] (raises {!Utils.User_error} otherwise): the range is the maximum extent, used to
    size allocations and projections until launch-time extent binding lands. *)

(** {2 Session bookkeeping (gh-ocannl-1301)} *)

val mutation_serial : unit -> int
(** Counts writes a failed construction can leave behind in the inference session; see
    [Tensor.rewind_to_last_closure]. Only {!note_mutation} advances it. *)

val note_mutation : unit -> unit

val is_used_as_extent : Ir.Indexing.static_symbol -> bool
(** Whether the symbol's extent role is committed or claimed since the last closure. *)

val is_used_as_slice : Ir.Indexing.static_symbol -> bool

val claim_symbol_role : Ir.Indexing.static_symbol -> [ `Extent | `Slice ] -> unit
(** Records a role claim; it reaches the symbol's flag at the next successful closure. *)

val commit_symbol_roles : commit:(Ir.Indexing.static_symbol -> [ `Extent | `Slice ] -> unit) -> unit
(** Hands every pending claim to [commit] -- [Shape]'s [Ir.Indexing.commit_role] -- and empties the
    pending set. *)

val drop_symbol_roles : unit -> unit

val reset_transient : unit -> unit
(** Clears the solver's transient work state between solver calls. *)

type globals
(** The solver's module-level inference state -- the pending role claims, the safe-to-guess,
    use-site-resolution and concatenation-component sets, the rank graph and the template cache --
    as persistent values (gh-ocannl-903). *)

val capture_globals : unit -> globals
(** A pointer capture, O(1): what a checkpoint records. *)

val restore_globals : globals -> unit
(** Puts a capture back, as part of a rewind. *)

val globals_differences : globals -> globals -> string list
(** The components that differ between two captures (physically), by name: a diagnostic. *)

val sym_dim_extent_exn : sym_dim -> int

(** The maximum extent of a symbolic dimension: the declared range of its static symbol. *)

val dim_to_int_exn : dim -> int
val vars_of_dim : dim -> dim_var_set

type print_style = Only_bases | Axis_size | Axis_number_and_size | Projection_and_size
[@@deriving equal, compare, sexp]

val solved_dim_to_string : print_style -> solved_dim -> string

val dim_to_string : ?var_to_string:(dim_var -> string) -> print_style -> dim -> string
(** [var_to_string] renders the dimension variables, nested ones included (default: [$<id>]). *)

type provenance [@@deriving sexp, compare, equal, hash]

val provenance : sh_id:int -> kind:kind -> provenance
val empty_provenance : provenance
val merge_provenance : provenance -> provenance -> provenance

type row_var [@@deriving sexp, compare, equal, hash]

val get_row_var : unit -> row_var

(** A bcast specifies how axes of a single kind in a shape (i.e. the row) can adapt to other shapes.
*)
type bcast =
  | Row_var of row_var
      (** The row can be inferred to have more axes (between [t.beg_dims] and [t.dims]). *)
  | Broadcastable  (** The shape does not have more axes of this kind, but is "polymorphic". *)
[@@deriving equal, hash, compare, sexp, variants]

type t = {
  beg_dims : dim list;
      (** The outer-left anchored flank; the first dimensions of the row. May be non-empty for both
          closed and open rows. *)
  dims : dim list;
      (** The outer-right anchored flank; the last dimensions of the row. May be non-empty for both
          closed and open rows. *)
  bcast : bcast;
      (** [Row_var v] absorbs only the middle gap between [beg_dims] and [dims]; [Broadcastable]
          means no rank-broadening slack between the two anchored flanks. *)
  prov : provenance;
}
[@@deriving equal, hash, compare, sexp]

val dims_basis_assoc : t -> (string * dim) list
val get_row_for_var : provenance -> row_var -> t
val row_shapes : t -> int list

type environment [@@deriving sexp_of]

type dim_constraint = Unconstrained_dim | At_least_dim of int
[@@deriving equal, hash, compare, sexp_of, variants]

type total_elems =
  | Num_elems of int
  | Range_elems of { lo : int; hi : int }
      (** The total is any value in the inclusive range [lo..hi]. Arises only from substituting the
          solved variable of a round-up [Strided_var] (the leftover slack window); when the total is
          otherwise unconstrained, late stages prefer [hi]. *)
  | Strided_var of { coeff : int Utils.safe_lazy; var : dim_var; denom : int; round_up : bool }
      (** The total number of elements is (coefficient * variable) / denominator. When [round_up],
          the variable is instead the ceiling of the division: solving for the variable from a known
          total [n] gives [var = ceil(n * denom / coeff)], i.e. the constraint tolerates a final
          partial block ([coeff * (var - 1) < n * denom <= coeff * var]). Minted only by the
          [Uint4x32_to_prec] constraint, where the last 128-bit random block may be consumed
          partially. *)
[@@deriving equal, hash, compare, sexp_of]

type row_constraint =
  | Unconstrained
  | Total_elems of { numerator : total_elems; divided_by : dim_var list; keep_axis : bool }
      (** The rows, inclusive of the further row spec, have this many elements. The total is
          numerator / (product of divided_by variables). divided_by has multiset semantics - the
          same variable can appear multiple times. [keep_axis] means the constraint comes from a
          data terminal that prefers, when the constrained rows receive no other shape information,
          keeping a single dim-1 axis over closing the leftover row variable to the empty row
          ("identity reshape"); it is reset once the constraint is reduced by consuming axes or
          variables. *)
  | Exact of dim list  (** The concatenated rows have these axes. *)
[@@deriving equal, hash, compare, sexp_of]

type constraint_origin = {
  lhs_name : string;
  lhs_kind : kind;
  rhs_name : string;
  rhs_kind : kind;
  operation : string option;
}
[@@deriving sexp_of]

(** An entry implements inequalities [res ⊑ v ⊑ opnd] and/or an equality [v = solved]. [res] and
    [opnd] must be sorted using the [@@deriving compare] comparison. *)
type dim_entry =
  | Solved_dim of dim
  | Bounds_dim of {
      is_in_param : bool;
      has_uniq_constr_unless : dim_var_set option;
          (** If set, the variable should not be guessed 1 unless a variable from the set is also
              prevented from being guessed 1. *)
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

type constraint_ =
  | Dim_eq of { d1 : dim; d2 : dim; origin : constraint_origin list }
  | Row_eq of { r1 : t; r2 : t; origin : constraint_origin list }
  | Dim_ineq of { res : dim; opnd : dim; from_ : Sexp.t; origin : constraint_origin list }
  | Row_ineq of { res : t; opnd : t; origin : constraint_origin list }
  | Dim_constr of { d : dim; constr : dim_constraint; origin : constraint_origin list }
  | Rows_constr of { r : t list; constr : row_constraint; origin : constraint_origin list }
      (** The constraint applies to the concatenation of the rows. Note: broadcasting does not
          affect the constraint (i.e. there is no "subtyping", it resembles Row_eq). *)
  | Terminal_dim of bool * dim * constraint_origin list
      (** A terminal dimension with is_param flag indicating if it's a parameter requiring gradient.
      *)
  | Terminal_row of bool * t * constraint_origin list
      (** A row of the shape of a terminal tensor (i.e. a tensor that does not have sub-tensors).
          The bool flag indicates if it's a parameter requiring gradient. *)
  | Shape_row of t * constraint_origin list
      (** A row of a shape of interest. The bool flag indicates if it's a parameter requiring
          gradient. *)
[@@deriving compare, equal, sexp_of, variants]

type error_trace = ..

type error_trace +=
  | Row_mismatch of t list
  | Dim_mismatch of dim list
  | Index_mismatch of Ir.Indexing.axis_index list
  | Constraint_failed of constraint_

val sexp_of_error_trace : error_trace -> Sexp.t

exception Shape_error of string * error_trace list [@@deriving sexp_of]

(** Removed duplicate helper functions *)
type stage = Stage1 | Stage2 | Stage3 | Stage4 | Stage5 | Stage6 | Stage7
[@@deriving sexp, equal, compare]

val add_safe_to_guess : row_var -> unit
(** Mark a row variable as allowed to be empty even if it's in a parameter. *)

val add_used_in_spec_or_compose : row_var -> unit
(** Mark a row variable as used in an einsum spec or compose shape update. Meant specifically for
    input rows, to indicate that the variable should not be guessed empty when it ends up in a
    parameter. *)

val add_used_in_pointwise : row_var -> unit
(** Mark a row variable as used in a pointwise shape update. Meant specifically for input rows, to
    indicate that the variable can be guessed empty when it ends up in a parameter. *)

val add_concat_component : dim_var -> unit
(** Mark a dimension variable as naming a component of a concatenated einsum axis: closure refuses
    to guess it to dimension 1 (gh-ocannl-1656). *)

val add_resolve_at_use : row_var -> unit
(** Mark a row variable as resolved by its use sites: at the final solver stages it may close to the
    GLB of its use sites' rows, acquiring axes its own arguments do not have. Unmarked variables
    close to an empty remainder instead — a use site broadcasts an operation result in but cannot
    silently widen it beyond its arguments' shapes (gh-ocannl-544). Registered automatically for
    terminal rows (leaf tensors and parameters) and for parameter-initialization cones; requested
    explicitly by the [stretch] operation. *)

val is_resolve_at_use : row_var -> bool

val add_resolve_at_use_dim : dim_var -> unit
(** The dim-level counterpart of {!add_resolve_at_use}: a marked dimension variable may close to the
    GLB of its use sites; unmarked, unconstrained ones are guessed minimal (dim-1) instead.
    [At_least_dim]-constrained variables keep GLB closing regardless — direct indexing is a
    dim-carrying use. *)

val is_resolve_at_use_dim : dim_var -> bool
val subst_row : environment -> t -> t

val unify_row :
  stage:stage -> constraint_origin list -> t * t -> environment -> constraint_ list * environment

val empty_env : environment
val get_dim_val : environment -> dim_var -> int option

val get_dim_from_env : environment -> dim_var -> dim option
(** The variable's solution of any form (a concrete [Dim], a symbolic [Sym], ...), or [None] while
    it is unsolved (absent or only bounded). {!get_dim_val} is its concrete-size projection. *)

val get_row_from_env : environment -> row_var -> t option

val find_dim_entry : environment -> dim_var -> dim_entry option
(** A dimension variable's entry: solved, or its bounds. *)

val find_row_entry : environment -> row_var -> row_entry option
(** A row variable's entry: solved, or its bounds. *)

val unsolved_constraints : environment -> constraint_ list

val solve_inequalities :
  stage:stage ->
  ?discardable_vars:dim_var_set ->
  constraint_ list ->
  environment ->
  constraint_ list * environment
(** [discardable_vars] are safe to guess dimension = 0, they do not participate in projections for
    the update step the constraints were derived for. *)

val row_to_bases : environment -> t -> string array

type proj [@@deriving compare, equal, sexp]
type proj_env [@@deriving sexp_of]

val fresh_row_proj : t -> t

val equal_modulo_proj : t -> t -> bool
(** Row equality ignoring projection ids: they are scratch metadata each projection derivation
    re-mints ({!fresh_row_proj}), not part of a row's committed content. *)

val populate_dim_proj_in_solved : environment -> environment

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
  ?clamp_padded:bool ->
  proj_equation list ->
  resolved_padding:(proj_id, axis_padding) List.Assoc.t ->
  inferred_padding:(proj_id, axis_padding) List.Assoc.t ->
  proj_env
(** [resolved_padding] is used for verification only. [inferred_padding] is updated in [proj_env]
    even if [resolved_padding] is present.

    [clamp_padded] (default [false], gh-504): the operation's accumulation identity is non-finite
    (max/tropical family), so its padded ([=]-mode) window projections register no margin demand
    (and skip the exceeds-resolved check): they lower with clamped window bounds — the assignments
    lowering guards the accumulation to the operand's valid range, an out-of-range position
    contributing the accumulation identity, which is the same as not visiting it. *)

val get_proj_index : proj_env -> proj -> Ir.Indexing.axis_index
val get_dim_index : proj_env -> dim -> Ir.Indexing.axis_index
val get_product_proj : proj_env -> dim -> (proj_id * int) option

val get_dim_padding : proj_env -> dim -> axis_padding option
(** [resolved_padding] from [proj_env] is ignored (it is meant for verification only). *)

val proj_to_iterator_exn : proj_env -> proj_id -> Ir.Indexing.symbol
(** [proj_to_iterator_exn proj_env p] returns the iterator for [p] in [proj_env]. Raises
    [Invalid_argument] if [p] is not an iterator. Note that symbols go through substitution when
    lowering, so these iterators are not the same as the ones in the generated code. *)

val product_dim_iterators : proj_env -> (proj_id * int * Ir.Indexing.symbol) list
(** Iterated projections, including concat components with dim 1. *)

(** {2 Shape-scheme support (gh-ocannl-404)}

    What [Shape.Scheme] needs to carry a trace's constraints past its rewind and rename them per
    application. Dimension and row variables share one counter, monotonic within the session's range
    and within the reserved range of {!with_reserved_vars}. *)

val with_reserved_vars : (unit -> 'a) -> 'a
(** Runs the function with row and dimension variables minted from a reserved range (an id-neutral
    window, gh-ocannl-1751): every variable it mints must be discarded with it. *)

val var_mark : unit -> int
(** The id of the most recently minted variable: a variable with a larger id was minted since. *)

val dim_var_id : dim_var -> int
val row_var_id : row_var -> int
val discardable_vars : environment -> dim_var_set

type renaming = { rn_dim : dim_var -> dim_var; rn_row : row_var -> row_var }

val rename_dim : renaming -> dim -> dim
val rename_row : renaming -> t -> t

val rename_constraint : renaming -> constraint_ -> constraint_
(** Every variable position of every constructor; static symbols stay rigid. *)

val constraint_vars : constraint_ -> dim_var list * row_var list
val row_vars : t -> dim_var list * row_var list

val add_constraint_origin : constraint_origin -> constraint_ -> constraint_
(** Prepends a provenance entry. *)

val has_lazy_coefficient : constraint_ -> bool
(** Whether the constraint holds a [Strided_var] total, whose coefficient is a lazy over a node's
    precision. *)

type marks
(** The additions to the per-variable policy sets (and the pending symbol-role claims) since a
    {!capture_globals}. *)

val marks_since : globals -> marks

val add_marks : renaming -> marks -> unit
(** Re-adds the marks through the sets' writers, renaming the variables. *)

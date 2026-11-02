(** The nest view of raw lowered code shared by the recognizers of the algebraic-rewrite tier
    ({!Rewrites}): {!Online_softmax} and {!Prefix_scan} read a routine as its top-level statements,
    each either a perfect serial nest with one [Set] at the bottom or opaque to them, plus a census
    of who writes and reads which node. *)

val ( let* ) : 'a option -> ('a -> 'b option) -> 'b option

(** {1 Nests}

    Every [Accum_op] lowers to one top-level nest: serial loops from the outside in and a single
    [Set] at the bottom. A guard, a scan or a hardware-annotated loop anywhere in a statement makes
    it invisible to the recognizers. *)

type loop = { index : Indexing.symbol; from_ : int; to_ : int }

type nest = {
  loops : loop list;  (** Outermost first. *)
  tn : Tnode.t;
  idcs : Indexing.axis_index array;
  llsc : Low_level.scalar_t;
}

val nest_of : Low_level.t -> nest option

val opaque_effect : Tnode.t Affine.statement_effect -> bool
(** Whether an effect row is code the census cannot see, or that a rewrite must not move a write
    across. *)

val wrap : loop list -> Low_level.t -> Low_level.t
(** Serial loops around a body, the first loop outermost. *)

val mentions : Indexing.symbol -> Indexing.axis_index array -> bool
(** Whether a plain iterator of the symbol is among the indices. *)

val is_loop : nest -> Indexing.symbol -> bool

val reduction : nest -> (Ops.binop * Low_level.scalar_t) option
(** The accumulation operator and the accumulated operand of a nest [lhs[i] := lhs[i] op rhs]
    (either operand order for [Add] and [Max]) whose [rhs] does not read [lhs]. *)

val pointwise : nest -> bool
(** An elementwise definition: the value reads no cell of its own target. *)

val map_scalar :
  idx:(Indexing.axis_index -> Indexing.axis_index option) ->
  get:(Tnode.t -> Indexing.axis_index array -> Low_level.scalar_t option) ->
  Low_level.scalar_t ->
  (Low_level.scalar_t * bool) option
(** The expression with every index rewritten by [idx] and every read [get] answers replaced, and
    whether any read was replaced. A scope, local, dynamic or merge-buffer read declines. *)

(** {1 Routines} *)

type routine = {
  stmts : Low_level.t array;  (** The top-level statements, in order. *)
  nests : nest option array;
  writers : (Tnode.t, int list) Base.Hashtbl.t;
  reads : Base.Set.M(Tnode).t array;
  opaque : bool array;
}
(** The census is read off each statement's relations (gh-ocannl-1050), which enter loop, scan,
    guard and scope bodies alike. A merge-buffer read counts as a read of its source node. *)

val routine_of : Low_level.t -> routine

val writers : routine -> Tnode.t -> int list
(** The positions of the statements writing the node, ascending. *)

val reads_at : routine -> int -> Base.Set.M(Tnode).t

val definition : routine -> Tnode.t -> (int * nest) option
(** The one pointwise nest defining the node, if it is defined exactly once and elementwise. *)

val covers_whole : nest -> bool
(** Whether a nest writes every cell of its node once: instantiating a definition at arbitrary
    subscripts is sound only then. *)

val unfold :
  routine ->
  stop:Tnode.t list ->
  before:int ->
  depth:int ->
  Low_level.scalar_t ->
  (Low_level.scalar_t * int list) option
(** The expression a read stands for, unfolded through single-writer elementwise definitions and
    constant fills (short of the [stop] nodes) that write their whole node at a position before
    [before], with the positions of the definitions it went through. *)

(** {1 Index arithmetic} *)

val terms_of : Indexing.axis_index -> ((int * Indexing.symbol) list * int) option
(** An index as affine terms and an offset; [None] for a sub-axis or a concatenation. *)

val subst_idx :
  (Indexing.symbol -> Indexing.axis_index option) ->
  Indexing.axis_index ->
  Indexing.axis_index option
(** The index with each symbol the map answers replaced by its image, affinely. *)

val subst_scalar :
  ?get:(Tnode.t -> Indexing.axis_index array -> Low_level.scalar_t option) ->
  (Indexing.symbol -> Indexing.axis_index option) ->
  Low_level.scalar_t ->
  Low_level.scalar_t option

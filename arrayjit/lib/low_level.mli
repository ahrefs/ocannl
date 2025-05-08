(** {1 A for-loop-based array language and backend-agnostic optimization} *)

open Base

module Scope_id : sig
  type t = { tn : Tnode.t; scope_id : int } [@@deriving sexp_of, equal, hash, compare]
  type comparator_witness

  val comparator : (t, comparator_witness) Base.Comparator.t
end

type scope_id = Scope_id.t = { tn : Tnode.t; scope_id : int }
[@@deriving sexp_of, equal, hash, compare]

(** {2 Low-level representation} *)

(** Cases: [t] -- code, [float_t] -- single number at some precision. *)
type t =
  | Noop
  | Comment of string
  | Staged_compilation of (unit -> unit)
  | Seq of t * t
  | For_loop of { index : Indexing.symbol; from_ : int; to_ : int; body : t; trace_it : bool }
  | Zero_out of Tnode.t
  | Set of { tn : Tnode.t; idcs : Indexing.axis_index array; llv : float_t; mutable debug : string }
  | Set_local of scope_id * float_t
[@@deriving sexp_of, equal]

and float_t =
  | Local_scope of { id : scope_id; body : t; orig_indices : Indexing.axis_index array }
  | Get_local of scope_id
  | Get_global of Ops.global_identifier * Indexing.axis_index array option
  | Get of Tnode.t * Indexing.axis_index array
  | Ternop of Ops.ternop * float_t * float_t * float_t
  | Binop of Ops.binop * float_t * float_t
  | Unop of Ops.unop * float_t
  | Constant of float
  | Embed_index of Indexing.axis_index
[@@deriving sexp_of, equal, compare]

val apply_op : Ops.op -> float_t array -> float_t
val flat_lines : t list -> t list
val unflat_lines : t list -> t
val loop_over_dims : int array -> body:(Indexing.axis_index array -> t) -> t

(** {2 Optimization} *)

type virtualize_settings = {
  mutable enable_device_only : bool;
  mutable max_visits : int;
  mutable max_tracing_dim : int;
  mutable inline_scalar_constexprs : bool;
}

val virtualize_settings : virtualize_settings

type visits =
  | Visits of int
  | Recurrent
      (** A [Recurrent] visit is when there is an access prior to any assignment in an update. *)
[@@deriving sexp, equal, variants]

type traced_array = {
  tn : Tnode.t;
  mutable computations : (Indexing.axis_index array option * t) list;
      (** The computations (of the tensor node) are retrieved for optimization just as they are
          populated, so that the inlined code corresponds precisely to the changes to the arrays
          that would happen up till that point. Within the code blocks paired with an index tuple,
          all assignments and accesses must happen via the index tuple; if this is not the case for
          some assignment, the node cannot be virtual. Currently, we only allow for-loop symbols in
          assignment indices of virtual nodes. *)
  assignments : int array Base.Hash_set.t;
  accesses : (int array, visits) Base.Hashtbl.t;
  mutable zero_initialized : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;
      (** The node is read before it is written (i.e. it is recurrent). *)
  mutable read_only : bool;
  mutable is_scalar_constexpr : bool;
      (** True only if the tensor node has all axes of dimension 1, is either zeroed-out or assigned
          before accessed, is assigned at most once, and from an expression involving only constants
          or tensor nodes that were at the time is_scalar_constexpr. *)
}
[@@deriving sexp_of]

val get_node : (Tnode.t, traced_array) Base.Hashtbl.t -> Tnode.t -> traced_array
val optimize_integer_pow : bool ref

type traced_store = (Tnode.t, traced_array) Base.Hashtbl.t [@@deriving sexp_of]

type optimized = { traced_store : traced_store; llc : t; merge_node : Tnode.t option }
[@@deriving sexp_of]

val optimize :
  unoptim_ll_source:Stdlib.Format.formatter option ->
  ll_source:Stdlib.Format.formatter option ->
  name:string ->
  Indexing.static_symbol list ->
  t ->
  optimized

val input_and_output_nodes : optimized -> (Set.M(Tnode).t * Set.M(Tnode).t) * Tnode.t option
(** Inputs are the materialized read-only and read-before-write (within the code) non-constant
    non-merge nodes. They are inputs in a broad sense, as they could be recurrent nodes or
    parameters. Outputs are all the materialized nodes written-to by the code. The last returned
    component is the input merge node, if used in the code. *)

(** {2 Printing} *)

val code_hum_margin : int ref

val fprint_function_header :
  ?name:string ->
  ?static_indices:Indexing.static_symbol list ->
  unit ->
  Stdlib.Format.formatter ->
  unit

val get_ident_within_code : ?no_dots:bool -> ?blacklist:string list -> t array -> Tnode.t -> string

val fprint_cstyle :
  ?name:string ->
  ?static_indices:Indexing.static_symbol list ->
  unit ->
  Stdlib.Format.formatter ->
  t ->
  unit
(** Adheres more to the C syntax, outputs implicit type casts. *)

val fprint_hum :
  ?name:string ->
  ?static_indices:Indexing.static_symbol list ->
  unit ->
  Stdlib.Format.formatter ->
  t ->
  unit
(** Adheres more to the %cd syntax, does not output implicit type casts. *)

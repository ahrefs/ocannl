(** {1 A for-loop-based array language and backend-agnostic optimization} *)

open Base

(** {2 Global references} *)

module Scope_id : sig
  type t = { tn : Tnode.t; scope_id : int } [@@deriving sexp_of, equal, hash, compare]
  type comparator_witness

  val comparator : (t, comparator_witness) Base.Comparator.t
end

type scope_id = Scope_id.t = { tn : Tnode.t; scope_id : int }
[@@deriving sexp_of, equal, hash, compare]

(** {2 Low-level representation} *)

(** Cases: [t] -- code, [scalar_t] -- single number at some precision. *)
type t =
  | Noop
  | Comment of string
  | Staged_compilation of (unit -> PPrint.document)
  | Seq of t * t
  | For_loop of { index : Indexing.symbol; from_ : int; to_ : int; body : t; trace_it : bool }
  | Zero_out of Tnode.t
  | Set of {
      tn : Tnode.t;
      idcs : Indexing.axis_index array;
      llsc : scalar_t;
      mutable debug : string;
    }
  | Set_from_vec of {
      tn : Tnode.t;
      idcs : Indexing.axis_index array;
      length : int;
      vec_unop : Ops.vec_unop;
      arg : scalar_arg;
      mutable debug : string;
    }
  | Set_local of scope_id * scalar_t
[@@deriving sexp_of, equal]

and scalar_t =
  | Local_scope of { id : scope_id; body : t; orig_indices : Indexing.axis_index array }
  | Get_local of scope_id
  | Get of Tnode.t * Indexing.axis_index array
  | Get_merge_buffer of Tnode.t * Indexing.axis_index array
  | Ternop of Ops.ternop * scalar_arg * scalar_arg * scalar_arg
  | Binop of Ops.binop * scalar_arg * scalar_arg
  | Unop of Ops.unop * scalar_arg
  | Constant of float
  | Constant_bits of int64  (** Direct bit representation, primarily for uint4x32 *)
  | Embed_index of Indexing.axis_index
[@@deriving sexp_of, equal, compare]

and scalar_arg = scalar_t * Ops.prec [@@deriving sexp_of, equal, compare]
(** The argument precision is preserved in heterogeneous precision operation arguments, and is
    ignored (overridden) in homogeneous precision operations. *)

val scalar_precision : scalar_t -> Ops.prec
val apply_op : Ops.op -> scalar_t array -> scalar_t
val flat_lines : t list -> t list
val unflat_lines : t list -> t
val loop_over_dims : int array -> body:(Indexing.axis_index array -> t) -> t
val unroll_dims : int array -> body:(Indexing.axis_index array -> offset:int -> t) -> t

(** {2 Optimization} *)

type virtualize_settings = {
  mutable enable_device_only : bool;
  mutable max_visits : int;
  mutable max_tracing_dim : int;
  mutable inline_scalar_constexprs : bool;
  mutable inline_simple_computations : bool;
  mutable inline_complex_computations : bool;
}

val virtualize_settings : virtualize_settings

type visits =
  | Visits of int
  | Recurrent
      (** A [Recurrent] visit is when there is an access prior to any assignment in an update. *)
[@@deriving sexp, equal, variants]

type traced_array = {
  tn : Tnode.t;
  assignments : int array Base.Hash_set.t;
  accesses : (int array, visits) Base.Hashtbl.t;
  mutable zero_initialized_by_code : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;
      (** The node is read before it is written (i.e. it is recurrent). *)
  mutable read_only : bool;
      (** Surprisingly, the notions of read-only and of constant memory mode come apart: small
          hosted constants are not read-only because they are initialized on devices by being
          assigned to; and a volatile memory mode is read-only from the devices' perspective. *)
  mutable is_scalar_constexpr : bool;
      (** True only if the tensor node has all axes of dimension 1, is either zeroed-out or assigned
          before accessed, is assigned at most once, and from an expression involving only constants
          or tensor nodes that were at the time is_scalar_constexpr. *)
  mutable is_accessing : bool;
      (** False only if the tensor node is built from index embeddings and scalar constant
          expressions. *)
  mutable is_complex : bool;
      (** True only if the tensor node is built acciessing computations that are not a single
          getter. *)
}
[@@deriving sexp_of]

val get_node : (Tnode.t, traced_array) Base.Hashtbl.t -> Tnode.t -> traced_array
val optimize_integer_pow : bool ref

type traced_store = (Tnode.t, traced_array) Base.Hashtbl.t [@@deriving sexp_of]

type optimize_ctx = {
  computations : (Tnode.t, (Indexing.axis_index array option * t) list) Base.Hashtbl.t;
      (** The computations (of the tensor node) are retrieved for optimization just as they are
          populated, so that the inlined code corresponds precisely to the changes to the arrays
          that would happen up till that point. Within the code blocks paired with an index tuple,
          all assignments and accesses must happen via the index tuple; if this is not the case for
          some assignment, the node cannot be virtual. Currently, we only allow for-loop symbols in
          assignment indices of virtual nodes. *)
}
[@@deriving sexp_of]

type optimized = {
  traced_store : traced_store;
  optimize_ctx : optimize_ctx;
  llc : t;
  merge_node : Tnode.t option;
}
[@@deriving sexp_of]

val optimize :
  optimize_ctx ->
  unoptim_ll_source:(PPrint.document -> unit) option ->
  ll_source:(PPrint.document -> unit) option ->
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

val function_header_doc :
  ?name:string -> ?static_indices:Indexing.static_symbol list -> unit -> PPrint.document

val get_ident_within_code : ?no_dots:bool -> ?blacklist:string list -> t array -> Tnode.t -> string

val to_doc_cstyle :
  ?name:string -> ?static_indices:Indexing.static_symbol list -> unit -> t -> PPrint.document
(** Adheres more to the C syntax, outputs implicit type casts. *)

val to_doc :
  ?name:string -> ?static_indices:Indexing.static_symbol list -> unit -> t -> PPrint.document
(** Adheres to the %cd syntax. *)

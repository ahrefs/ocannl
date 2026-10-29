(** Assignment computations over tensor nodes. *)

open Base

type init_data =
  | Broadcast of float
      (** Host-backed scalar fill of the inferred shape, including its committed padding. Unlike
          [Reshape], this does not constrain the number of elements. *)
  | Reshape of Ndarray.t
  | Keep_shape_no_padding of Ndarray.t
  | Padded of { data : Ndarray.t; padding : Ops.axis_padding array; padded_value : float }
[@@deriving sexp_of, equal]

type buffer = Node of Tnode.t | Merge_buffer of Tnode.t [@@deriving sexp_of, equal]

type fetch_op =
  | Constant of float
  | Constant_bits of int64
  | Constant_fill of float array
  | Range_over_offsets
  | Slice of { batch_idx : Indexing.static_symbol; sliced : Tnode.t }
  | Embed_symbol of Indexing.static_symbol
  | Embed_self_id
  | Embed_dim of Indexing.variable_ref
[@@deriving sexp_of, equal]

type accum_rhs =
  | Ternop of { op : Ops.ternop; rhs1 : buffer; rhs2 : buffer; rhs3 : buffer }
  | Binop of { op : Ops.binop; rhs1 : buffer; rhs2 : buffer }
  | Unop of { op : Ops.unop; rhs : buffer }
  | Block of { op : Ops.unop; rhses : buffer array }
  | Rev_sides of { op : Ops.unop; lhses : buffer array }
[@@deriving sexp_of, equal]

type t =
  | Noop
  | Seq of t * t
  | Block_comment of string * t
  | Accum_op of {
      initialize_neutral : bool;
      accum : Ops.binop;
      lhs : Tnode.t;
      rhs : accum_rhs;
      projections : Indexing.projections Utils.Lazy.t;
      projections_debug : string;
    }
  | Set_vec_unop of {
      op : Ops.vec_unop;
      lhs : Tnode.t;
      rhs : buffer;
      projections : Indexing.projections Utils.Lazy.t;
      projections_debug : string;
    }
  | Fetch of { array : Tnode.t; fetch_op : fetch_op; dims : int array Utils.Lazy.t }
[@@deriving sexp_of]

type side = Fwd | Bprop [@@deriving sexp, compare, equal]

(** A code fragment generated from a tensor, identified by its owner node and side (gh-ocannl-1724):
    [(t.value, Fwd)] is [t]'s forward code; [(t.grad, Bprop)] is [t]'s backprop code, the code
    propagating {i from} [t.grad] into its children's gradients. *)
module Fragment : sig
  type t = Tnode.t * side [@@deriving compare, sexp_of]
  type comparator_witness

  val comparator : (t, comparator_witness) Comparator.t

  val describe : t -> string
  (** E.g. "the backprop code propagating from grad_t", for refusal messages. *)
end

type comp = {
  asgns : t;
  embedded_nodes : Set.M(Tnode).t;
      (** The nodes in {!field-asgns} that are not in [embedded_nodes] need to already be in
          contexts linked with the {!comp} -- an initialization claim, not ownership. *)
  owned : Set.M(Fragment).t;
      (** The tensor-generated code fragments this computation carries. [Context.compile] refuses a
          computation owning a fragment some routine of its compilation lineage already owns. *)
  requires : Set.M(Fragment).t;
      (** The backprop fragments its gradient writes flow into that it does not carry itself.
          [Context.run] refuses the routine unless a routine of its own lineage (never a sibling)
          owns each one. *)
}
[@@deriving sexp_of]
(** Computations based on assignments. [Tensor] construction fills in all three sets; zeroing,
    initialization and hand-written [%cd] assignments own and require nothing. *)

val comp :
  ?owned:Set.M(Fragment).t -> ?requires:Set.M(Fragment).t -> embedded:Set.M(Tnode).t -> t -> comp
(** The computation constructor; [owned] and [requires] default to empty. *)

val no_fragments : Set.M(Fragment).t
val to_comp : t -> comp
val empty_comp : comp
val context_nodes : plc:Tnode.Placements.t -> t -> Tnode.t_set
val collect_nodes_guess_output : t -> Tnode.t_set * Tnode.t_set
val collect_written : t -> Tnode.t_set
val sequence : comp list -> comp
val collect_neutral_elem : t -> float option
val to_low_level : ?static_indices:Indexing.static_symbol list -> t -> Low_level.t

val to_doc :
  ?name:string -> ?static_indices:Indexing.static_symbol list -> unit -> t -> PPrint.document

val get_name_exn : t -> string

val lower :
  ?rewrite_target:Rewrites.target ->
  Low_level.optimize_ctx ->
  unoptim_ll_source:(PPrint.document -> unit) option ->
  ll_source:(PPrint.document -> unit) option ->
  cd_source:(PPrint.document -> unit) option ->
  name:string ->
  Indexing.static_symbol list ->
  t ->
  Low_level.optimized

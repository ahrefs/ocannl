(** gh-ocannl-498 rematerialization: the budget-driven recompute-vs-store planner.

    A deterministic planning pass over {!Context}'s analyze-only surface -- it lowers and scores,
    never compiles, links or executes. It lives beside [context.ml] rather than inside it because it
    needs nothing of the context beyond {!Context.lowered_for_decisions}, {!Context.hardware_limits}
    and {!Context.decide_inline}. *)

(** A memory budget: what {!fit} plans the routine's placements against. *)
type t =
  | Bytes of int  (** Fit the routine's scored footprint under this many bytes. *)
  | Minimize  (** Take every flip that relieves footprint, whatever the recompute cost. *)
[@@deriving sexp_of]

type plan = {
  bp_baseline : Ir.Low_level.footprint;  (** The default-policy placement vector's score. *)
  bp_final : Ir.Low_level.footprint;  (** The score after the accepted flips. *)
  bp_flips : (Ir.Tnode.t * int * int) list;
      (** The accepted flips in acceptance order: the node demoted to recompute-at-use, the
          {e marginal} bytes it relieved on top of the flips accepted before it, and its
          recompute-cost bound. Flips committed as one joint group (see {!fit}) carry [0] each
          except the one that closed the group, which carries the group's whole relief — so the
          reliefs sum to exactly [bp_baseline - bp_final] either way. *)
  bp_considered : int;  (** Inline candidates individually scored. *)
  bp_dropped : int;  (** Inline candidates the [max_candidates] cut left unscored. *)
  bp_within_budget : bool;
      (** Whether [bp_final] meets the budget. Always [true] for {!Minimize}, which has no target to
          miss. A [false] here is a planning outcome, not an error: the selector reports that the
          decision vector cannot reach the budget rather than forcing illegal flips. *)
}
[@@deriving sexp_of]

val compare_relief_ratio : int -> int -> int -> int -> int
(** gh-ocannl-498: [compare_relief_ratio ra ca rb cb] compares the rationals [ra/ca] and [rb/cb]
    exactly, as {!fit} ranks candidates by footprint relief per unit of recompute cost. [ca] and
    [cb] must be positive; the numerators are byte counts and may be negative, since inlining a node
    can cost footprint rather than free it. Never cross-multiplies (the products of a byte count and
    a recompute cost can overflow) and never uses floats (the order must be bit-reproducible).
    Exposed for unit testing the ordering, including the sign cases where a continued-fraction
    descent over truncating division would otherwise invert it. *)

val footprint :
  ?name:string ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Ir.Low_level.footprint
(** The byte footprint {!Context.compile} of this routine from this context would imply under the
    default placement policy ({!Context.Backends.score_footprint}). Analyze-only and hermetic,
    exactly like {!Context.decision_surface}: lowering and optimization, no backend codegen, no
    linking, no effect on the context. *)

val fit :
  ?name:string ->
  ?max_candidates:int ->
  budget:t ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Context.t * plan
(** gh-ocannl-498 rematerialization: choose which materialized intermediates to demote to
    recompute-at-use — or, since gh-ocannl-616, to footprint-scoped scratch — so that the routine's
    scored footprint ({!footprint}) fits [budget], and return a child context that decides them so
    ({!Context.decide_inline} / {!Context.decide_footprint}) together with the plan.

    A deterministic planning pass, not a timed search: recompute-vs-store under a budget is
    decidable from the two cost sides — the recompute-cost bound each [`Inline] flip candidate
    carries ({!Ir.Low_level.field-flip_candidates}) and the footprint relief scored against the
    actual arena layout. Nothing is compiled, linked or executed; given the same code, config and
    placements the pass always chooses the same flips.

    The selection is greedy in two rounds. First every candidate is scored on its own against the
    baseline layout — footprint relief is not a function of the node's own size, since a node whose
    live span was already shared with another's frees no bytes by leaving. That solo relief only
    {e ranks}: candidates are ordered by relief per unit of recompute cost (an exact rational
    comparison, never a cross-multiplication that could overflow nor a float, so the order is
    bit-reproducible), zero-relief ones last.

    Round two accepts a prefix, re-scoring the {e cumulative} vector at each step, since inlining
    one node moves the others' spans. A candidate that adds nothing on top of the accepted set is
    not dropped but held {e speculatively}: relief is not additive in either direction, and two
    nodes pinning the same arena peak each free nothing alone yet free the whole range together.
    Each later candidate is then scored both with and without the held group, because a held flip
    can be actively harmful rather than merely unpaid, and judging every later candidate only in its
    company would let one bad hold mask a candidate that pays on its own. A group that beats the
    candidate alone is load-bearing and commits with it; a group that loses is harmful and is
    discarded, never reconsidered (this is a bounded planner, not a search over subsets); a group
    that ties is merely neutral and keeps being held — committing it would pay recompute for zero
    bytes, and discarding it would throw away a flip that may still be half of a later pair.
    Speculatives never joined by a paying flip are discarded at the end, so recompute is never paid
    for zero bytes. Acceptance stops as soon as the budget is met; {!Minimize} takes every flip that
    helps.

    [max_candidates] bounds the individually-scored candidates, keeping the cheapest-to-recompute
    ones; the count left unscored is reported as [bp_dropped] and logged under config
    [log_memory_budget], never silently dropped. It defaults to 32 for a {!Bytes} budget — which
    stops as soon as it is met, so the cut is a cost guard — and to {e unbounded} for {!Minimize},
    whose contract is every flip that still relieves footprint and whose config-only users
    ([memory_budget=minimize]) cannot raise a cap. Passing it explicitly bounds either kind, at two
    lowerings per candidate scored.

    The [`Inline] and [`Footprint] directions are considered — the opposite of the [`Materialize]
    chain {!Ir.Low_level.field-flip_candidates} feeds in [Train.tune_placements]. A node carrying
    both records is scored in each (their reliefs differ: an inlined reading keeps the template's
    leaves live up to the late consumer, a footprint-scoped one only its scratch), and the first
    direction that pays takes the node. Legality and observability are not this pass's to enforce
    and it does not try: the decisions record preferences, the virtualizer's
    [check_and_store_virtual] settles legality, and a rejected preference simply reproduces the
    materialized placement — which is why relief is scored from a real lowering rather than assumed.

    Raises {!Ir.Utils.User_error} when config [buffer_aliasing] is off: without the liveness planner
    every node is always-live and the score has nothing to do with what the allocator would do. *)

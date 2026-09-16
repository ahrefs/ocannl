(** The algebraic-rewrite tier over raw lowered code: pattern-directed substitutions that CHANGE the
    computation, where the schedule transforms only rearrange a fixed one. [Assignments.lower]
    applies the tier between [to_low_level] and [Low_level.optimize], ahead of the analyses, so the
    traced store and the placements see the rewritten routine and a rewrite may remove a node's
    definition outright.

    Each member is gated by a config key of its own (the [approximate] profile is the umbrella that
    turns them all on), classified [Code_borne]: the rewritten code carries the decision, so no
    digest needs the key. The tier is a static list -- explicit order, and no registration that a
    dropped unreferenced module could silently skip.

    Members feed each other (a rewrite can expose another's pattern, in either order), so the tier
    runs the enabled members in order until the code stops changing, judged by structural equality.
    The contract that makes this terminate: a member is idempotent on its own output, and every
    application consumes at least one instance of its pattern. A member that keeps rewriting past
    {!max_rounds} rounds is refused as malformed rather than looped on. *)

type rewrite = {
  name : string;  (** The config key that gates it. *)
  enabled : unit -> bool;
  apply : Low_level.t -> Low_level.t;
}

val tier : rewrite list
(** The members, in application order: {!Online_softmax} (gh-ocannl-483). *)

val max_rounds : int

val apply : Low_level.t -> Low_level.t
(** The enabled members to a fixpoint; the identity when none is enabled. *)

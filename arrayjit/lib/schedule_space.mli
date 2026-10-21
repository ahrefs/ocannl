(** The partial-schedule decision space (gh-ocannl-514 phases 1-2): the representation
    branch-and-bound schedule search operates on, per docs/proposals/gh-ocannl-514.md.

    A search node is a {e partial schedule shape} — some decisions committed, the rest open. Two
    kinds of decision level make up the space:

    - {b Placement levels}: per policy-decided tensor node, where its value lives on the
      inline/compute-at/materialize spectrum ({!placement}). The Inline/Materialize poles are the
      landed decision vector ([Context.decide_inline], [optimized.flip_candidates]); [Pl_stage_at]
      is the compute-at middle, represented here from the start so it is a decision level rather
      than a bolt-on — its instantiation (a [Schedule.Stage] at the given loop) lands with the later
      phases.
    - {b Family levels}: which sketch pipeline, and its parameters — represented as a refinement
      {!tree} whose choices may depend on earlier commitments (a staging shape constrains which
      geometries remain; a twin exists only for staged geometries). The sketch families factor into
      such trees instead of flat enumerations ([Autotune.matmul_sketch_tree] and
      [Autotune.conv_sketch_tree]); the seed lists are exactly the trees' {!leaves}.

    Phase 2 makes the tree carry {e legality over partial shapes}: each choice's children are judged
    with the same three-valued logic as [Schedule.op_legality], quantified over completions —
    {!child.Refuted} asserts no completion below the child is valid (with the violated constraint as
    witness, a decline explanation produced before any compilation, gh-ocannl-479), {!child.Unknown}
    never fathoms, and construction commits to a verdict at the moment the domain is built rather
    than leaving branches to fail at candidate compile. A fourth verdict, {!child.Excluded}, keeps
    the landed policy/legality separation (gh-ocannl-555: preferences suppress heuristic caps only,
    legality is non-negotiable) visible in the space itself: an [Excluded] branch is valid but
    suppressed by default-policy search economy — a driver may re-propose it by lifting the policy;
    it must never re-propose a [Refuted] one. *)

(** One tensor node's placement level. [Pl_stage_at s] materializes the node at loop [s]'s scope —
    recomputed per iteration of the loops outside [s], shared by the loops inside. [Pl_inline] and
    [Pl_materialize] are the poles the greedy flip chain already searches. Loop identity is by
    lowering-local symbol: persistence across fresh lowerings (schedule-cache rebinding) is the
    later phases' concern, like instantiation. *)
type placement = Pl_inline | Pl_stage_at of Indexing.symbol | Pl_materialize
[@@deriving sexp_of, compare, equal]

type ('l, 'a) child =
  | Child of ('l, 'a) tree Lazy.t  (** Feasible as far as construction can decide. *)
  | Unknown of string * ('l, 'a) tree Lazy.t
      (** The [Op_unknown] analogue: construction cannot decide (the witness says why — e.g. a
          permissively detected site whose mis-detections only candidate compilation rejects), so
          the subtree stays enumerable and a fathom must never prune it; its completions settle at
          candidate compile. *)
  | Excluded of string * ('l, 'a) child Lazy.t
      (** Valid but not proposed: default-policy search economy (a dominated or degenerate shape, a
          twin whose measured cost exceeds its possible win). The witness states the policy; the
          payload is the {e same judgment with only this policy lifted} — forcing it is the driver's
          lift operation ({!lift_excluded}), and the result remains subject to legality (a lifted
          branch may still be [Refuted] on other grounds). {!leaves}, {!enumerate} and the
          collectors treat the exclusion as opaque: they neither enumerate nor descend into the
          payload. *)
  | Refuted of string
      (** The [Op_illegal] analogue, quantified over completions: {e no} completion below this child
          satisfies the family's contract — an op precondition, a resource cap, or a
          statically-decided emission constraint refutes them all. The witness is the violated
          constraint: a decline explanation produced before any compilation. *)

and ('l, 'a) tree =
  | Leaf of 'a  (** A fully committed schedule shape — today, one sketch-seed parameter set. *)
  | Choice of { level : string; children : ('l * ('l, 'a) child) list }
      (** One open decision: [level] names it, each child one way to commit it — carried as ['l],
          the {e decision datum} identifying that commitment rather than a rendering of it
          (gh-ocannl-591: a consumer reading a decision back off a path matches on data, it never
          re-parses a display string; [Autotune.Family_decision] is the matmul family's such type,
          and its [to_label] with {!render_path} is the rendering). [level] names the node for
          display and grouping only — the datum is the identity, so renaming a level is a rendering
          change and cannot change what a path means. A [Choice] whose children are all
          [Refuted]/[Excluded] is an infeasible node ({!leaves} is empty). Levels appear in
          {e emission order} (the order candidates reach timing), not necessarily dependency order.
          Verdicts are decided when the parent is constructed — fathoming needs them without
          expansion — while feasible children's subtrees stay lazy: expanding one is a decision, not
          a side effect of construction. *)

val leaves : ('l, 'a) tree -> 'a list
(** All completions of [Child] and [Unknown] branches, in tree traversal order — the order the flat
    enumerations produced, so a factored family's [leaves] replaces its seed list drop-in. Forces
    the reachable subtrees. *)

val enumerate : ('l, 'a) tree -> ((string * 'l) list * 'a) list
(** {!leaves} with each completion's decision path — the committed [(level, decision)] vector that
    identifies the leaf; prefixes of these paths are the partial vectors interior nodes stand for.
*)

val refutations : ('l, 'a) tree -> ((string * 'l) list * string) list
(** Every [Refuted] child with its decision path (ending at the refuted commitment) and witness: the
    shape's pre-compilation decline explanations (gh-ocannl-479), in traversal order. *)

val exclusions : ('l, 'a) tree -> ((string * 'l) list * string) list
(** Every [Excluded] child with path and policy witness — the branches a driver could re-propose by
    lifting default-policy economy. *)

val unknowns : ('l, 'a) tree -> ((string * 'l) list * string) list
(** Every [Unknown] child with path and witness — the branches whose verdicts only candidate
    compilation settles; a fathom must treat them as feasible. *)

type search_stats = {
  st_expanded : int;  (** Choice nodes entered. *)
  st_scored : int;  (** Leaves the score hook priced. *)
  st_unscored : int;
      (** Leaves the score hook declined to price ([None]) — never winners. The hook's [None] covers
          both genuine no-coverage and caller-side rejections; a caller wanting the split tracks it
          inside the hook (as [Autotune.model_default] does). *)
  st_fathomed : int;  (** Subtrees pruned by the bound against the running threshold. *)
  st_refuted : int;  (** [Refuted] children encountered (never entered). *)
  st_excluded : int;  (** [Excluded] children encountered (never entered; a driver may lift). *)
  st_unknown : int;  (** [Unknown] children entered — the bound never fathoms them. *)
}
[@@deriving sexp_of]
(** The fathomed-vs-scored ledger (gh-ocannl-514 phase 6's evaluation data): how much of the space
    the verdicts and the bound dispatched without pricing. *)

val no_search_stats : search_stats

val search :
  ?bound:(path:(string * 'l) list -> ('l, 'a) tree -> float option) ->
  ?incumbent:float ->
  ?tie_break:('a -> int option) ->
  score:('a -> float option) ->
  ('l, 'a) tree ->
  ('a * float) option * search_stats
(** Branch-and-bound minimization over the refinement tree (gh-ocannl-514 phase 4). Depth-first in
    child order — the emission order — so on a bound-free run the returned minimum is the {e first}
    leaf achieving it, matching the flat enumerations' [min_elt] tie behavior that candidate
    ordering relies on. The running threshold starts at [incumbent] (an existing candidate's cost —
    e.g. the untuned default's model score, or a measured time) and tightens to each new best score;
    a leaf wins only by {e strict} improvement, preserving ties-to-the-incumbent.

    [bound] is the optimistic (lower-bound) cost of a subtree's completions: a subtree whose bound
    is at or above the threshold is fathomed — equality fathoms because displacement needs strict
    improvement. Soundness is the caller's contract ({!Cost_model.completion_floor}'s): a bound that
    can exceed a completion's true cost prunes true winners. [path] is the committed
    [(level, decision)] vector down to and including the judged child — the partial vector the
    subtree stands for (a prefix of {!enumerate}'s paths; [[]] for the root) — so a bound whose
    floor depends on the commitments (the placement levels, where {!Cost_model.completion_floor}'s
    [open_placement] narrows as [Pl_materialize] commitments accumulate) can price the exact node
    being judged; a commitment-invariant bound (the schedule-invariant family floor) ignores it.

    Verdict-fathoming and cost-fathoming are distinct relations. The [Op_unknown]-never-fathoms
    contract is about {e verdicts}: an [Unknown] child is never treated as [Refuted], so a directly
    encountered one is entered without consulting the bound (legality uncertainty is its dominant
    unknown, and a sound bound over possibly-nonexistent completions is vacuous there). A {e cost}
    bound, by contrast, may fathom any [Child] subtree — nested [Unknown]s included — because its
    soundness quantifies over every completion independently of how verdicts resolve: a subtree that
    cannot beat the incumbent even where legal is correctly pruned. [Refuted]/[Excluded] children
    are never entered (the construction-time fathoms, counted separately). Unscored leaves ([score]
    = [None]) are counted and never win: in the untuned regime a pick must have a model price, and
    the measured regime handles no-coverage candidates outside the bound-driven walk.

    [tie_break] (gh-ocannl-1254) orders leaves whose scores are {e exactly} equal: a leaf scoring
    the running best's score displaces that best leaf when both have a key and its key is strictly
    smaller; a leaf the key does not apply to ([None]) neither displaces nor is displaced by a tie.
    Ties keep going to the first leaf among equal keys, and never to a leaf against the [incumbent],
    which is no leaf and is displaced only by strict improvement. Keys are computed only for leaves
    that tie and for a best leaf a subtree's bound equals, each at most once. While a leaf with a
    key holds the threshold, a subtree whose bound equals it is entered rather than fathomed: a
    completion at the bound could win the tie. Under a keyless best leaf, as under the incumbent,
    equality fathoms. Without [tie_break] the search is exactly the first-strict-improvement walk
    above. *)

val lift_excluded : ('l, 'a) child -> ('l, 'a) child
(** The policy-lift operation: [Excluded (_, payload)] becomes the forced payload — the same
    judgment with only that one policy lifted, still subject to legality; any other child is
    returned unchanged. Lifting is per exclusion: a payload may itself contain further [Excluded]
    children deeper in its subtree. *)

val count_choices : ('l, 'a) tree -> int
(** Interior (decision) nodes reachable through [Child]/[Unknown]; forces them. *)

val depth : ('l, 'a) tree -> int
(** Longest reachable decision chain root-to-leaf; forces reachable subtrees. *)

val render_path : label:('l -> string) -> (string * 'l) list -> string
(** A decision path as ["level=label > level=label > …"], each decision rendered by [label]. The one
    display form for paths: rendering is a consumer's presentation step, never how a decision is
    identified. *)

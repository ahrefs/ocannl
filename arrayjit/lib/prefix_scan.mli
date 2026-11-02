(** The prefix-scan rewrite (gh-ocannl-1758): a member of the algebraic-rewrite tier over lowered
    code ({!Rewrites}), gated by the config key [prefix_scan] (default off, on in the [approximate]
    profile, pinned off in [reproducible]).

    A cumulative sum written as the triangular-selector contraction -- a lower-triangular 0/1 mask
    times the input, [a +* "i; i o => o" tri] (tensor puzzle 7) -- lowers to an O(n^2) reduction
    nest [y[.., o] += v(.., i) * tri[i, o]]. This pass replaces such a nest with one
    {!Ir.Low_level.t.Scan_loop} per row over [o], carrying the running reduction, O(n):

    - inclusive ([r <= q]): [s' = s ⊕ v[r := q]; y[.., q] = y[.., q] ⊕ s'];
    - exclusive ([r < q]): [y[.., q] = y[.., q] ⊕ s; s' = s ⊕ v[r := q]], the step at the scan's
      last position folding the neutral element instead behind an index guard: no output reads that
      state, and the composed form never read that payload (a [where]-selected one may be out of
      range there);

    [Forward] for a prefix ([r <= q], [r < q]), [Backward] for a suffix ([r >= q], [r > q]), where
    [r] is the nest's one reduced loop and [q] an output loop over the same range. Writing [y ⊕ s]
    keeps what the target held: the zero (or [-inf]) fill of a forward, the earlier contributions of
    a gradient. The cumsum's backward [a.grad[i] += g[o] * tri[i, o]] is the same shape with the
    roles exchanged -- a suffix -- so it becomes a [Backward] scan with no gradient written by hand.

    {b Recognition}, on one perfect serial nest [y[..] := y[..] ⊕ rhs] (either operand order), [⊕]
    one of [Add], [Max], [Min], with [rhs] not reading [y] and [y] a float (an integer target
    declines: the state would round its sums):
    - exactly one reduced loop [r], and an output iterator [q] of [y] whose loop has [r]'s range;
      every other loop indexes [y] (a row, wrapped around the scan);
    - [rhs] splits into a payload [v] and an index-only mask [M(r, q)]: for [Add], a product whose
      index-only factors multiply to an exact 0/1 mask, or [where M v 0] (or its complement
      [where M 0 v]); for [Max] / [Min], [where M v (∓inf)] (or the complement), or the tropical
      [v + L] with [L] in [{0, ∓inf}]. What the mask rejects is exactly the neutral element of [⊕]
      (a rejected arm read through stored definitions is held to the masks' exactness standard); a
      max over a multiplicative 0/1 mask is declined (its zeros are not neutral);
    - the mask is proven a triangle by evaluating it symbolically, following the definitions of the
      nodes it reads (range grids, broadcasts, comparisons, [not], [where]) back to [Embed_index]: a
      node followed has exactly one writer, an elementwise nest covering it whole ahead of the read,
      unpadded; the mask must be a function of [r - q] alone matching one of the four triangles over
      the whole range of [r - q]. Every intermediate integer must be exactly representable in every
      precision the mask was computed or stored at (an f16 range of more than 2048 cells declines);
    - [v] does not depend on [q] and does not read [y]. When [rhs] is a single read of a node at
      [y]'s precision, that node's elementwise definition stands in for it (the masked operand of a
      max is stored before it is reduced), provided nothing the definition reads is written between
      it and the reduction.

    Everything else -- a band, a shifted or anti-diagonal triangle, a mask varying with a row, two
    reduced loops, ranges that differ, a guard or clamp, a mask held in a host-initialized constant
    or computed by an earlier routine -- is left alone. The mask's and the masked operand's
    definitions are not removed: they stay as their nodes' definitions, and the optimizer drops them
    when nothing else reads them.

    {b Numerics.} The carried state is at the target's precision widened to f32 (f64 stays f64). A
    forward cumsum adds in the composed form's order, so it agrees bitwise on finite inputs; cummax
    and cummin agree exactly. A suffix scan regroups the sum (the composed form adds ascending from
    [q], the scan descending from the end). The composed product form also multiplies the rejected
    payloads by 0, so an infinite payload outside the triangle makes it NaN where the scan -- the
    cumulative sum the selector encodes -- stays finite. Hence an [approximate]-tier gate.

    The cummax backward is not this shape (its operand depends on the output position through the
    gradient gate) and stays composed. *)

val enabled : unit -> bool
(** Whether the rewrite applies: the programmatic override if one is set, else the config key
    [prefix_scan]. *)

val set_enabled : bool option -> unit
(** Programmatic override of the config key, for experiments and tests; [None] restores the key.
    Takes effect at the next lowering. *)

val state_owner : Tnode.t -> int option
(** The uid of the target node a carried state accumulates into, when the node is a state this
    module minted. *)

val reset : unit -> unit
(** Drops the memoized state nodes (the tier's session-reset hook; also run ahead of an
    accessibility snapshot). Sibling lowerings of one program mint the same nodes, keyed by the
    matched site -- the target, the match's ordinal among the scans into it, the precision -- which
    keeps them on one analysis-cache key. *)

val rewrite : Low_level.t -> Low_level.t
(** The pass over raw lowered code (no gate: the tier consults {!enabled}). Every eligible nest is
    rewritten in place; a routine without one is returned as is. Idempotent: a scan is not a nest.
*)

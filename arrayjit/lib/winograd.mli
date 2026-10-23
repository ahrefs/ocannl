(** The Winograd F(2x2, 3x3) convolution rewrite (gh-ocannl-505): a member of the algebraic-rewrite
    tier over lowered code ({!Rewrites}), gated by the config key [winograd_conv] (default off, on
    in the [approximate] profile, pinned off in [reproducible]).

    A 2-D 3x3 stride-1 convolution lowers to one nest
    [d[.., oh, ow, oc] += a[.., oh + kh, ow + kw, ic] * b[oc, kh, kw, ic]] (the input physically
    padded for a padded conv, so its reads are offset-free). The rewrite replaces that nest with
    four, over three minted nodes ([xi, nu] in [0..3] index the 4x4 transformed tile):

    - the kernel transform [U = G b G^T], per output and reduction channel, recomputed every call
      (so a trained kernel's update is always seen);
    - the input transform [V = B^T a B], per 4x4 input tile read at [(2 th, 2 tw)];
    - sixteen per-position contractions
      [M[xi, nu, .., th, tw, oc] += V[xi, nu, ..] * U[xi, nu, oc, ic]] behind a zeroing of [M] --
      the write loops in [d]'s axis order and the reduction innermost, the lowered shape of a
      product, which the matmul sketches then schedule;
    - the inverse transform [A^T M A] into each 2x2 output tile, ACCUMULATED into [d], so the
      zeroing ahead of the original nest (or any prior contents) keeps its meaning.

    An odd output extent is covered by Winograd on its even prefix and by the original statement,
    over fresh symbols, on the last row and the last column (disjoint strips). For an even covered
    extent [E] the last tile reads input coordinate [E + 1], exactly the direct form's maximum, so
    the rewrite reads nothing the direct form does not.

    {b Recognition}, on one perfect serial nest with a single [Set] of the form [d = d + x * y]
    (either order, or [fma]): exactly two axes of one operand read [o + k] (unit coefficients,
    offset 0), with [o] a plain output iterator of [d] and [k] a loop over [0..2] that the other
    operand reads plainly; each of those four symbols occurs only there (separability); besides
    them, exactly one output-channel symbol (in [d] and the kernel) and one reduction-channel symbol
    (in the input and the kernel), every other loop a batch axis of [d] and the input, all indices
    plain iterators or fixed; all loops from 0, both output extents at least 2; three distinct
    single-precision nodes. Everything else -- other windows, strides, dilations, padding offsets,
    precisions, channel rows of several axes -- is left alone. [Autotune.detect_conv] answers the
    scheduling question for the same nests but lives above this library (and refuses singleton axes,
    which this rewrite carries through: a batch of one is a fixed index here);
    [test/operations/winograd_conv] pins the two to agree on the eligible sites.

    Only the forward accumulation is rewritten: a conv's data gradient writes at affine indices and
    its weight gradient's "window" is the feature map, so both keep the direct form.

    The transformed-domain arithmetic changes rounding beyond reassociation (the kernel transform
    scales by halves and quarters, the tile transforms add and subtract), so results move within the
    [approximate] tier's tolerance; on inputs whose every partial sum is exactly representable the
    two forms agree bitwise, since every coefficient is dyadic. The rewrite is idempotent, as the
    tier's fixpoint requires: transformed reads index [2 t + i], the contractions have no window,
    and a remainder strip's output loop does not start at 0. *)

val enabled : unit -> bool
(** Whether the rewrite applies: the programmatic override if one is set, else the config key
    [winograd_conv]. *)

val set_enabled : bool option -> unit
(** Programmatic override of the config key, for experiments and tests; [None] restores the key.
    Takes effect at the next lowering. *)

val minted_role : Tnode.t -> [ `Kernel | `Input | `Product ] option
(** Whether a node is one of the rewrite's transformed buffers ([U], [V] or [M]), and which. The
    test is membership among the nodes this module minted. *)

val reset : unit -> unit
(** Drops the memoized minted nodes (the tier's session-reset hook; also run ahead of an
    accessibility snapshot). Sibling lowerings of one program mint the same nodes, keyed by the
    whole matched site, which keeps them on one analysis-cache key. *)

val rewrite : Low_level.t -> Low_level.t
(** The pass over raw lowered code (no gate: the tier consults {!enabled}). Every eligible conv nest
    is rewritten; a routine without one is returned as is. The minted nodes are declared
    [Never_virtual] with memory-mode provenance ["505:winograd-transform"]: inlined, [V] would be
    re-transformed once per output channel. *)

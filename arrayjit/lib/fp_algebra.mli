(** The licensed floating-point algebra over raw lowered code (gh-ocannl-998): a member of the
    {!Rewrites} tier, after {!Online_softmax}.

    The rewrites are identities over the reals that change IEEE results -- contraction of
    [a * b + c] into one [FMA], reassociation (constants, subtraction chains, the
    multiplication/division interchanges, integer-power unrolling) and the zero identities that
    infinities, NaNs and the sign of zero contradict -- each family gated by its licence key
    ({!Low_level.Fp_licence}, from [fp_contract], [fp_reassociate] and [fp_finite_math]). They used
    to run inside [Low_level.simplify_llc], after virtualization and again after scheduling, so
    contraction crossed inlining boundaries: an inlined producer [y := a * b] fused with its
    consumer's [y + c] while a materialized one rounded, and a placement flip changed results at
    uniform f32. On the raw lowering, one assignment at a time, the member never sees an inlined
    read, so OCANNL's own algebra is placement-neutral, and [simplify_llc] is exact everywhere it
    runs. The trade, taken on purpose: a fold that exists only after inlining (a consumer
    contracting its producer's product, a scalar constant node's value folding into a quotient) is
    lost. Backend compilers keep their own contraction licences ([cc_backend_fp_contract], NVRTC's
    [--use_fast_math]), which this does not govern.

    It runs after {!Online_softmax}: contraction turns the value reduction [o := o + w * v] into
    [fma (w, v, o)], which the probability hoist does not match. *)

val enabled : unit -> bool
(** Whether any licence is configured on. *)

val rewrite : Low_level.Fp_licence.t -> Low_level.t -> Low_level.t
(** Applies the licensed {!Low_level.fp_algebra_step} bottom-up to every floating expression, each
    rewrite re-normalized, so the result is a fixpoint: a second application changes nothing.
    Integer expressions are left to the exact canonicalizer. *)

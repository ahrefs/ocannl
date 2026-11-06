(** Interval (min/max) lattice shared between the per-tensor bounds summaries stored on {!Tnode.t}
    and the [interval_of] analysis over [Low_level.scalar_t]. See
    docs/design/interval-analysis-scalar-t.md. This is deliberately the single module defining the
    bounds record and its operations (binding constraint 9 of the proposal): [Tnode] stores plain
    [t] values; [Low_level] extends them with a symbol environment and source tracking.

    {b Invariants.}
    - {b Outwardness}: [lo] and [hi] always bound every possible (machine) value from below resp.
      above. Every arithmetic rule either produces exactly-representable endpoints or nudges the
      computed endpoints outward (see {!round_out}), so ordering-based folds are sound without an
      exactness requirement (binding constraint 6: "exact endpoints or outward rounding at every
      operation").
    - {b NaN-freedom}: a non-[top] interval asserts the value cannot be NaN. Any rule whose result
      may be NaN returns [top]. [lo = nan] never occurs.
    - [integral] asserts every possible {e machine} value is a mathematical integer (note that every
      float of magnitude >= 2^53 is an integer, so integrality survives float rounding).
    - [exact] asserts the endpoints represent the true bounds exactly; in particular
      [abs endpoint < 2^53] when [integral] (strict: 2^53 + 1 rounds {e to} 2^53, binding constraint
      6). Equality/singleton folds additionally require [exact]: two distinct integers above 2^53
      may round to the same float. *)

type t = { lo : float; hi : float; integral : bool; exact : bool } [@@deriving sexp, compare, equal]

val top : t
val is_top : t -> bool

val exact_int_cutoff : float
(** Strict exact-integer cutoff: [2^53]. Values with [abs v < exact_int_cutoff] read from integer
    storage convert to float exactly. *)

val point : float -> t
(** Constructs the singleton interval of a float value. NaN gives [top]; infinities give non-[exact]
    singletons (they never participate in folds). *)

val of_int : int -> t

val of_int_range : int -> int -> t
(** [of_int_range lo hi] is the interval of integers between [lo] and [hi] inclusive; requires
    [lo <= hi]. *)

val round_out : t -> t
(** Nudges endpoints outward by two ulps to absorb the round-to-nearest error of endpoint arithmetic
    (one ulp would do for a single operation; two is safety margin). Used whenever an arithmetic
    rule cannot guarantee exactly-representable endpoints, preserving the outwardness invariant. *)

val join : t -> t -> t
(** Lattice join (convex hull). *)

val inter : t -> t -> t
(** Lattice meet over the value sets; endpoint selection only, no arithmetic. The result can be
    empty ([lo > hi]) if the inputs are disjoint -- callers intersect facts known to hold
    simultaneously, so this does not arise for sound inputs. *)

val is_within : outer:t -> t -> bool
(** Value-set containment: every value admitted by [inner] is admitted by [outer]. Used to validate
    proposals against settled bounds. Conservative: outward-rounded [inner] endpoints can only cause
    false rejections, never false acceptance. *)

val is_singleton : t -> bool

(** {2 Truthiness}

    A derived view, not a lattice facet (see the proposal). Sound w.r.t. both the C emitters
    ([!= 0], [&&], ternary) and [Ops.interpret_*]: non-[top] intervals are NaN-free, so the
    C-vs-OCaml NaN divergences cannot arise on decided inputs. *)

val definitely_false : t -> bool
val definitely_true : t -> bool

(** {2 Machine ranges of precisions} *)

val dtype_range : Ops.prec -> t
(** The interval of all values representable (and thus of any machine result) at an integer
    precision; [top] for float and opaque-data precisions. Note the int64/uint64 upper endpoints
    round outward (2^63 - 1 is not a float) and are marked non-[exact]. *)

val at_prec : Ops.prec -> t -> t
(** [at_prec prec iv] makes [iv] valid for the machine value of an expression computed at (or
    converted into) precision [prec]: integer precisions pass a provably-in-range exact integral
    interval through and otherwise fall back to {!dtype_range} (out-of-range casts wrap); float
    precisions keep integrality and endpoint exactness only within the precision's exact-integer
    range, everything else widening to [top]. See the implementation for the full rule-by-rule
    rationale. *)

(** {2 Endpoint arithmetic}

    All rules take machine-valid operand intervals and return the {e real-arithmetic} result
    interval; callers apply {!at_prec} to account for the precision the operation actually computes
    in. Rules requiring finite endpoints return [top] otherwise (this also excludes operands that
    may be NaN, per the NaN-freedom invariant: [top] in, [top] out). *)

val add : t -> t -> t
val sub : t -> t -> t
val mul : t -> t -> t

val div : t -> t -> t
(** Division: [top] when the divisor interval contains 0 (or anything is unbounded); real-endpoint
    quotients otherwise, never exact. *)

val mod_ : t -> t -> t
(** Modulo, proposal rule: divisor an exact positive integral singleton [c] and a non-negative
    integral argument give [[0, min (hi, c-1)]]; anything else is [top] (avoids the C remainder sign
    traps). *)

val max_ : t -> t -> t
(** Max/Min: endpoint selection, no arithmetic error. [top] operands are rejected wholesale because
    the OCaml interpreter's [Float.max] and C's [fmax] disagree on NaN. *)

val min_ : t -> t -> t

val relu : t -> t
(** ReLU: [fmax(0, x)] maps NaN to 0 in C, and [Ops.interpret_unop] agrees, so even a [top] argument
    yields a genuine lower bound of 0. *)

val neg : t -> t

val trunc : t -> t
(** Truncation toward zero; monotone, exactly representable when the argument endpoint is. *)

val satur01 : t -> t
(** Saturation to [[0, 1]]: monotone clamping. [top] arguments stay [top]: the OCaml interpreter
    returns NaN for NaN (C's [fmax]/[fmin] chain would give a number), so a possibly-NaN argument
    admits no codomain bound valid on all execution paths. *)

val exp_like : t -> t
(** Codomain bound for [Exp]/[Exp2]: non-negative. NaN maps to NaN, so [top] stays [top]. *)

val abs_le_1 : t -> t
(** Codomain bound for [Sin]/[Cos]/[Tanh_approx]: [[-1, 1]] (libm and the GPU intrinsics respect
    it); NaN maps to NaN, so [top] stays [top]. *)

val sqrt_ : t -> t
(** Codomain bound for [Sqrt]: requires a provably non-negative argument (else NaN is possible). *)

val bool_range : t
(** The [[0, 1]]-valued operations (comparisons, [And]/[Or], [Not]): sound for any operands
    including NaN (all emitters produce 0 or 1). *)

val true_ : t
val false_ : t

(** {2 Comparison folding}

    Decisions rely on the outwardness invariant; equality-true additionally requires exact
    singletons (binding constraint 6). NaN safety: a possibly-NaN operand implies a [top] interval
    whose infinite endpoints never satisfy the strict inequalities, so a fold-to-true never fires on
    possibly-NaN operands; fold-to-false agrees with NaN comparison semantics (false) in both C and
    the OCaml interpreter. *)

val cmplt_decides : t -> t -> bool option
(** Whether [a < b] is decided: [Some true]/[Some false]/[None]. *)

val cmple_decides : t -> t -> bool option
(** Whether [a <= b] is decided. Unlike [cmplt_decides], the fold-to-true condition is non-strict,
    so infinite endpoints could satisfy it; requiring a finite endpoint on the deciding boundary
    keeps possibly-NaN ([top]) operands from folding to true ([NaN <= x] is false). *)

val cmpeq_decides : t -> t -> bool option
(** Whether [a = b] is decided. True requires both to be the same exact singleton; false requires
    disjointness (sound outward). *)

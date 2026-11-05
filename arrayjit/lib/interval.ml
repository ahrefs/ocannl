open Base
(** Interval (min/max) lattice shared between the per-tensor bounds summaries stored on {!Tnode.t}
    and the [interval_of] analysis over [Low_level.scalar_t]. See
    docs/proposals/interval-analysis-scalar-t.md. This is deliberately the single module defining
    the bounds record and its operations (binding constraint 9 of the proposal): [Tnode] stores
    plain [t] values; [Low_level] extends them with a symbol environment and source tracking.

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

let top = { lo = Float.neg_infinity; hi = Float.infinity; integral = false; exact = false }
let is_top { lo; hi; _ } = Float.(lo = neg_infinity) && Float.(hi = infinity)

(** Strict exact-integer cutoff: [2^53]. Values with [abs v < exact_int_cutoff] read from integer
    storage convert to float exactly. *)
let exact_int_cutoff = Ops.exact_int_cutoff

let endpoint_exact v = Float.is_finite v && Float.(abs v < exact_int_cutoff)

(** Constructs the singleton interval of a float value. NaN gives [top]; infinities give non-[exact]
    singletons (they never participate in folds). *)
let point v =
  if Float.is_nan v then top
  else { lo = v; hi = v; integral = Float.is_integer v; exact = Float.is_finite v }

let of_int i =
  let v = Float.of_int i in
  { lo = v; hi = v; integral = true; exact = endpoint_exact v }

(** [of_int_range lo hi] is the interval of integers between [lo] and [hi] inclusive; requires
    [lo <= hi]. *)
let of_int_range lo hi =
  assert (lo <= hi);
  let flo = Float.of_int lo and fhi = Float.of_int hi in
  { lo = flo; hi = fhi; integral = true; exact = endpoint_exact flo && endpoint_exact fhi }

(** Nudges endpoints outward by two ulps to absorb the round-to-nearest error of endpoint arithmetic
    (one ulp would do for a single operation; two is safety margin). Used whenever an arithmetic
    rule cannot guarantee exactly-representable endpoints, preserving the outwardness invariant. *)
let round_out iv =
  let down v = if Float.is_finite v then Float.one_ulp `Down (Float.one_ulp `Down v) else v in
  let up v = if Float.is_finite v then Float.one_ulp `Up (Float.one_ulp `Up v) else v in
  { iv with lo = down iv.lo; hi = up iv.hi; exact = false }

(** Lattice join (convex hull). *)
let join a b =
  {
    lo = Float.min a.lo b.lo;
    hi = Float.max a.hi b.hi;
    integral = a.integral && b.integral;
    exact = a.exact && b.exact;
  }

(** Lattice meet over the value sets; endpoint selection only, no arithmetic. The result can be
    empty ([lo > hi]) if the inputs are disjoint -- callers intersect facts known to hold
    simultaneously, so this does not arise for sound inputs. *)
let inter a b =
  {
    lo = Float.max a.lo b.lo;
    hi = Float.min a.hi b.hi;
    integral = a.integral || b.integral;
    exact = a.exact && b.exact;
  }

(** Value-set containment: every value admitted by [inner] is admitted by [outer]. Used to validate
    proposals against settled bounds. Conservative: outward-rounded [inner] endpoints can only cause
    false rejections, never false acceptance. *)
let is_within ~outer inner =
  Float.(inner.lo >= outer.lo)
  && Float.(inner.hi <= outer.hi)
  && ((not outer.integral) || inner.integral)

let is_singleton iv = iv.exact && Float.(iv.lo = iv.hi)

(** {2 Truthiness}

    A derived view, not a lattice facet (see the proposal). Sound w.r.t. both the C emitters
    ([!= 0], [&&], ternary) and [Ops.interpret_*]: non-[top] intervals are NaN-free, so the
    C-vs-OCaml NaN divergences cannot arise on decided inputs. *)

let definitely_false iv = Float.(iv.lo = 0.) && Float.(iv.hi = 0.)
let definitely_true iv = (not (is_top iv)) && (Float.(iv.lo > 0.) || Float.(iv.hi < 0.))

(** {2 Machine ranges of precisions} *)

(** The interval of all values representable (and thus of any machine result) at an integer
    precision; [top] for float and opaque-data precisions. Note the int64/uint64 upper endpoints
    round outward (2^63 - 1 is not a float) and are marked non-[exact]. *)
let dtype_range (prec : Ops.prec) =
  match Ops.integer_range prec with
  | Some (lo, hi) -> { lo; hi; integral = true; exact = endpoint_exact hi }
  | None -> top

(** [at_prec prec iv] makes [iv] valid for the machine value of an expression computed at (or
    converted into) precision [prec].

    - Integer precisions: a provably-in-range exact integral interval passes through; anything else
      falls back to {!dtype_range} -- out-of-range casts wrap (binding constraint 5), and in
      particular a lower bound that could cross zero at an unsigned precision widens to the full
      unsigned range rather than being trusted (binding constraint 7: the "crosses zero" rule,
      implemented as widening instead of an assert so it stays sound if an emitter ever produces
      such arithmetic). Note [dtype_range] is also correct for the float-to-integer conversions C
      leaves undefined out of range: all supported backends produce {e some} value of the type.
    - Float precisions: integrality and endpoint exactness survive only within the precision's
      exact-integer range (proposal lattice rules); everything else is [top] -- genuinely float
      computations never fold (Phase A policy), and float rounding could otherwise move machine
      values past real-arithmetic endpoints.
    - [Void_prec]/[Uint4x32_prec]: [top]. *)
let at_prec (prec : Ops.prec) iv =
  match prec with
  | Ops.Void_prec | Ops.Uint4x32_prec _ -> top
  | Ops.Byte_prec _ | Ops.Uint16_prec _ | Ops.Int32_prec _ | Ops.Uint32_prec _ | Ops.Int64_prec _
  | Ops.Uint64_prec _ ->
      let r = dtype_range prec in
      if iv.integral && iv.exact && Float.(iv.lo >= r.lo) && Float.(iv.hi <= r.hi) then iv else r
  | Ops.Half_prec _ | Ops.Bfloat16_prec _ | Ops.Fp8_prec _ | Ops.Single_prec _ | Ops.Double_prec _
    -> (
      match Ops.float_exact_int_limit prec with
      | Some lim
        when iv.integral && iv.exact && Float.(abs iv.lo <= lim) && Float.(abs iv.hi <= lim) ->
          iv
      | _ -> top)

(** {2 Endpoint arithmetic}

    All rules take machine-valid operand intervals and return the {e real-arithmetic} result
    interval; callers apply {!at_prec} to account for the precision the operation actually computes
    in. Rules requiring finite endpoints return [top] otherwise (this also excludes operands that
    may be NaN, per the NaN-freedom invariant: [top] in, [top] out). *)

let all_finite l = List.for_all l ~f:Float.is_finite

(* Exactness of add/sub/mul results: both operands exact and integral and the result endpoints below
   the cutoff -- integer arithmetic representable exactly in float. Otherwise nudge outward to
   absorb round-to-nearest (outwardness invariant). *)
let arith_result a b lo hi =
  let integral = a.integral && b.integral in
  let exact = a.exact && b.exact && integral && endpoint_exact lo && endpoint_exact hi in
  let iv = { lo; hi; integral; exact } in
  if exact then iv else round_out iv

let add a b =
  if all_finite [ a.lo; a.hi; b.lo; b.hi ] then arith_result a b (a.lo +. b.lo) (a.hi +. b.hi)
  else top

let sub a b =
  if all_finite [ a.lo; a.hi; b.lo; b.hi ] then arith_result a b (a.lo -. b.hi) (a.hi -. b.lo)
  else top

let mul a b =
  if all_finite [ a.lo; a.hi; b.lo; b.hi ] then
    let products = [ a.lo *. b.lo; a.lo *. b.hi; a.hi *. b.lo; a.hi *. b.hi ] in
    let lo = List.reduce_exn products ~f:Float.min in
    let hi = List.reduce_exn products ~f:Float.max in
    arith_result a b lo hi
  else top

(** Division: [top] when the divisor interval contains 0 (or anything is unbounded); real-endpoint
    quotients otherwise, never exact (integer division truncates and float division rounds; the
    result is only used through {!at_prec}, which widens integer-precision consumers to the dtype
    range since [integral] is cleared). *)
let div a b =
  if
    all_finite [ a.lo; a.hi; b.lo; b.hi ]
    && ((Float.(b.lo > 0.) && Float.(b.hi > 0.)) || (Float.(b.lo < 0.) && Float.(b.hi < 0.)))
  then
    let quotients = [ a.lo /. b.lo; a.lo /. b.hi; a.hi /. b.lo; a.hi /. b.hi ] in
    let lo = List.reduce_exn quotients ~f:Float.min in
    let hi = List.reduce_exn quotients ~f:Float.max in
    round_out { lo; hi; integral = false; exact = false }
  else top

(** Modulo, proposal rule: divisor an exact positive integral singleton [c] and a non-negative
    integral argument give [[0, min (hi, c-1)]]; anything else is [top] (avoids the C remainder sign
    traps). *)
let mod_ a b =
  if
    is_singleton b && b.integral
    && Float.(b.lo > 0.)
    && a.integral && a.exact
    && Float.(a.lo >= 0.)
    && Float.is_finite a.hi
  then
    let hi = Float.min a.hi (b.lo -. 1.) in
    { lo = 0.; hi; integral = true; exact = endpoint_exact hi }
  else top

(* Max/Min: endpoint selection, no arithmetic error. [top] operands are rejected wholesale because
   the OCaml interpreter's [Float.max] and C's [fmax] disagree on NaN. *)
let max_ a b =
  if is_top a || is_top b then top
  else
    {
      lo = Float.max a.lo b.lo;
      hi = Float.max a.hi b.hi;
      integral = a.integral && b.integral;
      exact = a.exact && b.exact;
    }

let min_ a b =
  if is_top a || is_top b then top
  else
    {
      lo = Float.min a.lo b.lo;
      hi = Float.min a.hi b.hi;
      integral = a.integral && b.integral;
      exact = a.exact && b.exact;
    }

(** ReLU: [fmax(0, x)] maps NaN to 0 in C, and [Ops.interpret_unop] agrees, so even a [top] argument
    yields a genuine lower bound of 0. *)
let relu a =
  {
    lo = Float.max 0. a.lo;
    hi = Float.max 0. a.hi;
    integral = a.integral;
    exact = (a.exact || is_top a) && Float.is_finite (Float.max 0. a.hi);
  }

let neg a = { a with lo = -.a.hi; hi = -.a.lo }

(** Truncation toward zero; monotone, exactly representable when the argument endpoint is. *)
let trunc a =
  if is_top a then top
  else
    let t v = Float.round_towards_zero v in
    {
      lo = t a.lo;
      hi = t a.hi;
      integral = true;
      exact = a.exact && endpoint_exact (t a.lo) && endpoint_exact (t a.hi);
    }

(** Saturation to [[0, 1]]: monotone clamping. [top] arguments stay [top]: the OCaml interpreter
    returns NaN for NaN (C's [fmax]/[fmin] chain would give a number), so a possibly-NaN argument
    admits no codomain bound valid on all execution paths. *)
let satur01 a =
  if is_top a then top
  else
    let clamp v = Float.max 0. (Float.min 1. v) in
    { lo = clamp a.lo; hi = clamp a.hi; integral = a.integral; exact = a.exact }

(** Codomain bound for [Exp]/[Exp2]: non-negative. NaN maps to NaN, so [top] stays [top]. *)
let exp_like a =
  if is_top a then top else { lo = 0.; hi = Float.infinity; integral = false; exact = false }

(** Codomain bound for [Sin]/[Cos]/[Tanh_approx]: [[-1, 1]] (libm and the GPU intrinsics respect
    it); NaN maps to NaN, so [top] stays [top]. *)
let abs_le_1 a = if is_top a then top else { lo = -1.; hi = 1.; integral = false; exact = false }

(** Codomain bound for [Sqrt]: requires a provably non-negative argument (else NaN is possible). *)
let sqrt_ a =
  if (not (is_top a)) && Float.(a.lo >= 0.) then
    { lo = 0.; hi = Float.infinity; integral = false; exact = false }
  else top

(** The [[0, 1]]-valued operations (comparisons, [And]/[Or], [Not]): sound for any operands
    including NaN (all emitters produce 0 or 1). *)
let bool_range = { lo = 0.; hi = 1.; integral = true; exact = true }

let true_ = { lo = 1.; hi = 1.; integral = true; exact = true }
let false_ = { lo = 0.; hi = 0.; integral = true; exact = true }

(** {2 Comparison folding}

    Decisions rely on the outwardness invariant; equality-true additionally requires exact
    singletons (binding constraint 6). NaN safety: a possibly-NaN operand implies a [top] interval
    whose infinite endpoints never satisfy the strict inequalities, so a fold-to-true never fires on
    possibly-NaN operands; fold-to-false agrees with NaN comparison semantics (false) in both C and
    the OCaml interpreter. *)

(** Whether [a < b] is decided: [Some true]/[Some false]/[None]. *)
let cmplt_decides a b =
  if Float.(a.hi < b.lo) then Some true else if Float.(a.lo >= b.hi) then Some false else None

(** Whether [a <= b] is decided. Unlike [cmplt_decides], the fold-to-true condition is non-strict,
    so infinite endpoints could satisfy it; requiring a finite endpoint on the deciding boundary
    keeps possibly-NaN ([top]) operands from folding to true ([NaN <= x] is false). *)
let cmple_decides a b =
  if Float.(a.hi <= b.lo) && (Float.is_finite a.hi || Float.is_finite b.lo) then Some true
  else if Float.(a.lo > b.hi) then Some false
  else None

(** Whether [a = b] is decided. True requires both to be the same exact singleton; false requires
    disjointness (sound outward). *)
let cmpeq_decides a b =
  if is_singleton a && is_singleton b && Float.(a.lo = b.lo) then Some true
  else if Float.(a.hi < b.lo) || Float.(b.hi < a.lo) then Some false
  else None

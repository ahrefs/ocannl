# Interval (min/max) analysis over scalar_t

**Date**: 2026-06-12, elaborated 2026-07-03, implemented 2026-07-04
**Status**: Implemented — Phase A and Phase B v1 (host-initialized, never-device-written
tensors) landed: `Interval` lattice module, `Low_level.interval_of` with the total symbol
env and env-scoped memo, comparison/`Where` folding in `simplify_llc`, the re-derived
`build_guarded_gather`, the `Tnode.bounds_state` propose/settle lifecycle with device-write
pinning and host-upload scans, and the executed-parity test
(`test/operations/test_bounds_folded_gather.ml`). One deviation from the draft lattice
rules: instead of gating ordering folds on `exact`, endpoints maintain an *outwardness
invariant* (outward 2-ulp rounding whenever endpoint arithmetic is not exactly
representable), which satisfies binding constraint 6's "exact endpoints **or** outward
rounding" and lets `dtype_range uint64`-style facts (inexact upper endpoint) still
discharge lower-bound conjuncts; `exact` remains required for equality/singleton folds.
The crosses-zero rule is implemented as sound widening to the unsigned dtype range in
`Interval.at_prec` rather than an assert. Originally seeded by the tinygrad
deep dive ([a-range-is-not-its-shape](../blog/a-range-is-not-its-shape.md), port area 3).
Judged there the best effort-to-payoff item of the six ports; no blocking dependency
([signed-index-precision](signed-index-precision.md) now blocks on *this*).

## Goal

An interval lattice over `Low_level.scalar_t` (and the integer analogue over
`Indexing.axis_index`), in the spirit of tinygrad's `vmin`/`vmax` derived property
(`tinygrad/uop/ops.py`, `_min_max`): `Constant` exact, `Embed_index` bounded by its
loop's statically-known extent, `Get` bounded by dtype range, arithmetic by interval
rules (add endpoints; extremal products; careful div/mod-by-constant cases). Exposed as
an analysis `simplify_llc`'s rewrite arms can consult to discharge validity masks,
prove indices in-bounds, and fold comparisons.

## Key points from the article's analysis

- Four existing efforts independently approximate this one analysis: stage-B
  injectivity in [#133](https://github.com/ahrefs/ocannl/issues/133) (range arithmetic
  over loop extents), read-before-write tracking in
  [#340](https://github.com/ahrefs/ocannl/issues/340), matcher side-conditions in
  [#343](https://github.com/ahrefs/ocannl/issues/343), and the landed surjectivity
  reasoning of [#420](https://github.com/ahrefs/ocannl/issues/420). Intervals are the
  unifying upgrade.
- Sequencing: before the schedule layer
  ([schedule-ir-optops](schedule-ir-optops.md)) — Padto is affordable in tinygrad
  *because* interval reasoning discharges most of the masks it introduces.
- Receiving site exists: `interval_of : scalar_t -> bounds` slots into `simplify_llc`'s
  world; loop extents are statically known from projections.
- **Symbol environment tracks all symbols.** `Embed_index` is where a symbol crosses from
  the index world into the value world, so the analysis threads `symbol → interval` —
  the abstract twin of `visit_llc`'s concrete `symbol → int` env. Completeness is by
  construction: every in-scope symbol at lowering time comes from a `For_loop`
  (`[from_, to_]`) or a static binding (`[0, static_range)`; top when the range is
  `None` — see [signed-index-precision](signed-index-precision.md) on bind-time
  validation). The one env serves both value analysis (`Embed_index` into guards and
  comparisons) and position analysis (`Get`/`Set` `idcs`, the #133/#420-style facts).
  Wiring is already in place: `optimize_proc` receives the full
  `static_indices : static_symbol list` (ranges included) and passes it to every
  candidate host pass; bounds are lost only to the per-pass idiom that strips the list
  to a bare-symbol membership `Set` (low_level.ml lines 613/796/1479 as of 2026-07-03).
  Seeding = building a `symbol → interval` map from the same list before the strip.
  **Enclosing guards narrow it too** (gh-ocannl-566): an `If` condition holds on every
  execution reaching its body, so `simplify_llc` refines the environment with the bounds
  a conjunction of integer-affine index comparisons implies (the shapes
  `Schedule.partition_breakpoints` reads) before recursing. Without it, construct-then-fold
  has a blind spot wherever a guard is provable only *under* a statement guard — the
  gh-ocannl-487 pipelined prefetch, whose `k := k+1` substitution widens the edge guard's
  interval past exactly the block the enclosing `If (k < to_)` excludes. A comparison
  contributes only when its machine evaluation is faithful to the mathematical integers
  over the incoming bounds (each side must survive `Interval.at_prec` unchanged at both
  the index precision and the condition's evaluation precision) — a single-precision
  guard `k <= 2^24` is also machine-true at `k = 2^24 + 1`, which rounds down.
  Verify at implementation time that `static_range` (a `mutable int option`) is settled
  before `optimize_proc` runs — dims are forced by lowering, so ranges derived from dims
  should be; assert rather than assume.
- **The memo must be env-scoped.** The same `scalar_t` subtree can be physically shared
  under different loop nests (immutable-tree sharing after rewrites; CSE/`Local_scope`
  reuse), and its interval differs per enclosing scope. Key the memo by (expression,
  symbol-env) or scope it per loop body — a global identity-keyed memo would return
  another scope's bounds.

## Lattice and rules (draft)

Domain: closed intervals over extended reals with an integrality flag,
`{ lo : float; hi : float; integral : bool }`. Top = `[-inf, +inf]` non-integral. No
bottom (expressions always evaluate). NaN policy: any expression that may evaluate to NaN
is top — folding decisions require NaN-freedom, which the integer fragment guarantees and
the float fragment mostly forfeits.

`axis_index` (exact, integral): `Fixed_idx i → [i,i]`; `Iterator s → env(s)`;
`Sub_axis → [0,0]`; `Affine {symbols; offset}` → `offset + Σ coeff·env(s)` with endpoints
picked by coefficient sign. `Concat` is gone by lowering.

`scalar_t`:
- `Embed_index idx` → interval of `idx`, integral.
- `Constant c` → `[c,c]`, integral iff `c` is a whole number; NaN → top. `Constant_bits` → top.
- `Get (tn, _)` → settled `Tnode` bounds (Phase B), else dtype range: integer precs give
  `[0, 2^w)` or `[-2^(w-1), 2^(w-1))` integral; float precs top. `Get_dynamic` likewise
  via the table's node. `Get_local`/`Local_scope` → top in Phase A.
- Binops: `Add`/`Sub`/`Mul` by endpoint arithmetic (extremal products). `Div`: top when
  the divisor interval contains 0, else endpoint rules. `Mod` by a constant `c > 0`:
  `[0, min(hi, c-1)]` when the argument is integral with `lo ≥ 0`, else top (avoids
  C-remainder sign traps). `Max`/`Min`: pointwise endpoints. `Relu_gate`/`Satur01_gate`:
  hull of `[0,0]` and the gated argument. Comparisons (`Cmplt`/`Cmpeq`/`Cmpne`) →
  `[0,1]` integral, folding to a point when intervals decide: `Cmplt` true iff
  `hi1 < lo2`, false iff `lo1 >= hi2`; `Cmpeq` false iff disjoint, true iff both are the
  same singleton. `And` → `[0,1]`, folds when either side is a decided point.
- Ternops: `Where (c, a, b)` → `a` when `c` folds true, `b` when false, else hull;
  `FMA` composes the `Mul`/`Add` rules.
- Unops: `Relu → [max(0,lo), max(0,hi)]`; `Trunc` → truncated endpoints, integral;
  monotone transcendentals map endpoints; non-monotone → codomain bounds or top.
- Precision annotations (`scalar_arg`): arithmetic at prec `p` preserves integrality
  claims only within `p`'s exact-integer range (fp16: 2048, fp32: 2^24, fp64: 2^53) —
  check the annotation before asserting `integral`, else drop the flag and widen.
  Machine-value soundness today: emitted index arithmetic is non-negative by
  construction (physical padding), so no unsigned-wrap modeling; keep a "lower bound
  could cross zero → top" assert until [signed-index-precision](signed-index-precision.md)
  lands.
- Exactness of the bound carrier: the float `lo`/`hi` cannot represent all int64/uint64
  values (dtype-range endpoints like 2^64-1 round). Carry an `exact : bool` bit, cleared
  whenever an endpoint is not exactly representable (in particular |v| >= 2^53).
  Range/ordering folds remain sound with outward-rounded endpoints; equality/singleton
  folds (`Cmpeq` true) are forbidden unless `exact` — two distinct integers may round to
  the same float.
- Float-arithmetic folding policy (Phase A): comparison folds fire only on facts derived
  from exact integral intervals (index arithmetic, integer `Get`s, whole constants).
  Intervals of genuinely float computations are carried conservatively (top, or
  outward-rounded endpoints) and never fold comparisons — real-endpoint arithmetic can
  underapproximate rounded fp results. Revisit with explicit outward rounding if a float
  consumer materializes.
- Unknown-op policy: conservative by construction, but via *exhaustive* matches whose
  currently-unhandled arms (`Or`, `ToPowOf`, `Threefry4x32_*`, `Satur01`, `Recip`,
  `Recip_sqrt`, `Tanh_approx`, `Not`, `Mul3`, ...) explicitly return top — not a `_ ->
  top` wildcard, so adding an op to `Ops` forces a conscious interval-rule decision at
  compile time.
- Truthiness is a derived view, not a lattice facet: definitely-false iff the interval
  is `[0,0]`, definitely-true iff `0` is outside `[lo, hi]`, else unknown. Comparison
  outputs are `[0,1]`-integral points so this is complete for `And`/`Where` over guard
  fragments; a separate "nonzero with mixed-sign range" facet is deferred until a
  consumer needs it.

Designated re-expression target (acceptance criterion): re-derive
`build_guarded_gather`'s three guard flavors. Construct the guard generically (lower
bound, upper bound, integrality conjunct) and let interval folding erase what the ids
precision proves — unsigned `Get` gives `[0, 2^w)` so the lower conjunct folds; integer
precs prove integrality so `Trunc` folds — leaving the current hand-written flavors as
emergent behavior. It is landed, executable, and already golden-tested
(`test_one_hot_embedding_lookup` asserts `Trunc` counts on both paths).

Pass-ordering caveat: the optimize pipeline applies `simplify_llc` *before*
`rewrite_one_hot_reductions` (low_level.ml `optimize_proc`), so a generically-built
guard is born after the only fold pass. Phase A therefore has `build_guarded_gather`
call the interval folder directly on the guard it constructs (local, no pipeline
change, no golden perturbation elsewhere); an additional post-rewrite simplify/fold
pass is the general fix if later rewrites also start emitting foldable code.

## Phasing

- **Phase A** (implementable now): `interval_of` over `axis_index` + `scalar_t` with the
  total symbol env (seeded from `static_indices` before the membership strip),
  env-scoped memo, rules above; consumers: comparison folding in `simplify_llc` and the
  gather-guard re-derivation.
- **Phase B**: `Tnode` vmin/vmax with the propose/settle/conflict lifecycle and
  host-write symmetry (below); consumer: full guard fold for runtime ids. Includes an
  audit of *all* host-write paths, not just `Context.set_values`/`from_host`: direct
  backend init/upload paths, persistence/checkpoint restore, and link-time `Host_inits`
  must participate in propose/settle/validate; device-to-device copies propagate the
  source node's bounds by join (no scan needed). **v1 scope restriction** (binding
  constraint 3 below): fold only on bounds of host-initialized, never-device-written
  tensors.
- **Phase C** (separate proposal, blocks on A):
  [signed-index-precision](signed-index-precision.md) with tnode-granular width
  selection; then logical-padding masks per [schedule-ir-optops](schedule-ir-optops.md).

## Interprocedural layer: vmin/vmax on `Tnode.t`

The `scalar_t`-level analysis is intra-routine; without more, every `Get`/`Get_dynamic`
degrades to dtype range at routine boundaries (a reader routine lowers its own comp; the
writer's defining expression is in a different comp — the tensor node is the only artifact
crossing between them). Store per-tensor scalar bounds (float `vmin`/`vmax` pair, exact for
integer values below 2^53) on `Tnode.t` as writer-derived summaries. Payoff: with ids
settled at `[0, vocab)`, the gh-343 gather guard folds *entirely* for genuinely runtime
ids, not just range-produced ones; same for data-dependent mask discharge.

Soundness design (third instance of the `delayed_prec` lifecycle pattern):

- **Multi-writer staleness**: a guard discharged against bounds, followed by a
  later-compiled wider writer, would leave already-generated code unsound (an OOB read,
  not a wrong zero). So: writers *propose* bounds (joined, like `Inferred` promotion);
  a reader that discharges a guard *settles* them (like forcing the prec lazy); a
  post-settlement writer whose interval does not fit is a compile-time error (like
  `update_prec` on a settled precision).
- **Host writes**: symmetric with compiled writers around the settlement point.
  *Pre-settlement*, `set_values`/`from_host` (and ndarray-backed `init_data` — proposed
  *lazily* when the `Host_inits` buffer is forced at link/upload time, not at tensor
  creation: `Reshape` inits deliberately wait for shape and padding inference, so an
  eager scan would force unresolved dims or miss padding added by `create_with_reshape`)
  act as writers: scan the uploaded values (O(n), trivial next to the copy) and *propose*
  the observed `[min, max]` into the join — or pin the tensor to top if scanning is
  disabled. Otherwise a later reader could settle narrow writer-derived bounds and fold a
  guard against host contents that were never inspected. A bonus: tensors created via
  `class_ids_of_int_list`-style host arrays get tight bounds automatically.
  *Post-settlement*, host writes validate against the settled bounds or error.
- **Self-referential writers** (read-modify-write across executions: param updates,
  accumulators) default to top — a fixpoint over unbounded repetitions is not attempted.
  The profitable cases (ids, range producers, one-hots) are write-once/functional.

## Binding constraints from the first implementation attempt (PR #87, reviewed 2026-07-04)

A full first implementation (Codex, PR #87 — discarded, starting over) was reviewed
in depth; its Phase A core was close to this spec, but the review surfaced design gaps
that this proposal must state as binding rules, not implementation advice. The redo must
satisfy all of these:

1. **Fail-safe guard construction.** Construct-then-fold is sound only when every
   generically-built conjunct is *correct unfolded* at its guard precision. `-1 < iv` at
   uint64 is not: it survives whenever `iv`'s interval is top (`Local_scope`/`Get_local`
   — e.g. a virtualized computed unsigned ids expression), and casting `-1.0` to
   `uint64_t` is C UB — the guard goes silently always-false. Keep precision-flavored
   construction for any conjunct whose unfolded form is wrong (the unsigned lower bound),
   or emit such conjuncts only when the fold succeeds. Folding is an optimization; it
   must never be a correctness obligation.
2. **Settlement is transitive and fires at every consumption point.** PR #87 settled only
   on full comparison/Where folds and leaked everywhere else: (a) writer-bound
   derivations read other nodes' *unsettled* candidates and dropped the interval's
   `sources` — a later legal widening of the source leaves a stale derived candidate that
   a reader then folds and settles (OOB with no error, ever); (b) the one-sided `And`/`Or`
   eliminations rewrote on a conjunct's bounds without settling them; (c) the `Where`
   interval rule returned the chosen branch's interval while dropping the *condition's*
   sources. Rule: every rewrite or proposal that consumes a node's bounds settles that
   node (the `sources` plumbing exists for exactly this); interval rules must propagate
   sources through branch selection.
3. **Proposals must be execution-anchored, not compile-anchored.** A compiled writer may
   never run, run after the reader, or run repeatedly. A per-`Set` *syntactic* self-read
   check is insufficient: `Seq(Set tmp = Get tn; Set tn = Get tmp + 1)` and two-routine
   cycles escape it, yielding bounds valid only for the first execution with no runtime
   validation of device writes. Pin to top any node whose value can transitively depend
   on its own prior state (dataflow closure, per routine and across routines), and
   account for the pre-first-write buffer contents (see 4). **Phase B v1 simplification
   (recommended): fold only on bounds of host-initialized, never-device-written tensors**
   — this covers the ids payoff with none of the above hazards.
4. **The domain of a node includes what the routine did not write.** A `Set` proposal is
   a fact about written elements only; allocation-time backend zero-init (memset — not
   represented as `Zero_out` in the llc) and the physical-padding halo fill are part of
   the node's value domain. Join `[0,0]` for backend-zero-initialized nodes and the
   padding fill value for padded tensors (host scans skip margins — the fill must be
   joined explicitly), unless surjective coverage is proven.
5. **Model the store conversion.** Writer intervals must be clamped/widened to the
   storage dtype's behavior: integer casts wrap (propose top unless the RHS is provably
   in range), float stores overflow to inf. Readers intersect candidates with dtype
   bounds. And bfloat16's exact-integer limit is **256** (7 mantissa bits), not fp16's
   2048 — witness: 257 is unrepresentable.
6. **Exactness discipline for all folds.** Ordering folds (`Cmplt`), not just
   equality/singleton folds, require `exact` endpoints or outward rounding at every
   operation — round-to-nearest endpoint arithmetic can round *inward* above 2^53.
   Host-scan exactness must be strict (`< 2^53`): the value 2^53 + 1 rounds *to* 2^53
   and would pass a `<=` test.
7. **The crosses-zero rule is not optional.** PR #87 skipped the "lower bound could cross
   zero at unsigned precision → top" assert entirely; real-arithmetic `Sub` at unsigned
   annotations folded by real ordering is latent unsoundness (no current emitter, but
   nothing prevents one). Implement the assert in `sub`/affine evaluation.
8. **Performance is part of the contract.** Host scans run per upload: gate them to
   nodes that can profit (integer precisions / gather-index candidates), fold directly
   over the bigarray in one pass (no boxed flat-values copy), and skip when settled-top.
   The memo is one table per env scope shared across queries — not a fresh assoc list
   per query — and bottom-up simplify folds at the top node only (children are already
   folded), never re-walking subtrees per nesting level.
9. **One lattice, one module.** The bounds record and its ops (top/point/join/contains,
   exactness) live in a single shared module that `Tnode` stores and the analysis extends
   with `sources`/env — PR #87's twin records with field-by-field converters is a drift
   trap (its two `point` functions already disagreed on finiteness).
10. **Executed parity for the folded path.** At least one test must *execute* a kernel
    whose guard folded via `Tnode` bounds — compile after the upload proposes — and
    compare values against the unfolded build. PR #87's suite only ever executed
    pre-bounds kernels and inspected folds via re-lowering, so a fold that produced
    wrong values would have passed everything. Structural assertions and `.expected`
    files (which the standalone test must ship, per repo convention) complement but do
    not replace this.

## Relations

[#133](https://github.com/ahrefs/ocannl/issues/133),
[#340](https://github.com/ahrefs/ocannl/issues/340),
[#343](https://github.com/ahrefs/ocannl/issues/343), landed #420;
[schedule-ir-optops](schedule-ir-optops.md) (downstream consumer);
[signed-index-precision](signed-index-precision.md) (**blocks on this**, revised
2026-07-03: intervals need no wrap modeling over the current IR — physical padding keeps
emitted index arithmetic non-negative by construction, so the "lower bound could cross
zero" rule is a cheap assert until masks land — while the signed migration waits for
intervals so tnode-granular width selection ships in its final form and the golden churn
is paid once).

## Acceptance criteria

- [x] Lattice and rules specified (draft above; single lattice with an integrality flag
      rather than two variants; widening not needed — loop extents are finite and
      static).
- [x] Re-expression target designated: `build_guarded_gather`'s guard flavors re-derived
      by folding a generically-constructed guard (implementation lands with Phase A).
- [x] Caching strategy decided: per-node memo, env-scoped (see the symbol-environment
      key point).
- [x] `Tnode` bounds lifecycle specified (propose/settle/conflict semantics mirroring
      `delayed_prec`; host-write symmetry around settlement; self-referential writers
      pinned to top) — implementation is Phase B.
- [x] All ten binding constraints from the PR #87 review satisfied by the redo (fail-safe
      guard construction; transitive settlement; execution anchoring / host-init-only v1;
      unwritten-domain joins — via whole-buffer scans including padding margins, plus
      device-write pinning making memset-zero domains moot in v1; store-conversion
      modeling in `Interval.at_prec`; exactness/outwardness for all folds; crosses-zero
      widening; gated single-pass scans + env-scoped shared memo; single lattice module
      `Interval`; executed parity of a bounds-folded kernel in
      `test/operations/test_bounds_folded_gather.ml`).

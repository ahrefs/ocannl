# Stage B: Virtualize Injective Affine Producers

Parent: [gh-ocannl-133.md](gh-ocannl-133.md)

Tracked by: https://github.com/ahrefs/ocannl/issues/133

## Goal

Allow virtual node inlining for producer positions with more than one non-static symbol in a
single affine index, but only when the producer's LHS map is proven injective over the
dropped producer loops.

The work has two linked parts:

1. Improve affine injectivity analysis in `Indexing.is_injective`.
2. Use that result in `check_idcs` and `make_subst` to inline sound affine producers.

The motivating forms come from convolution and pooling backprop projections, for example:

```text
idx = stride * oh + kh
idx = K * i + k
```

## Non-Goals

- Repeated-symbol diagonal support. That is [Stage A](gh-ocannl-133-stage-a.md), and Stage B
  builds on its substitution grouping and guard machinery.
- Loop-dropping substitution for non-injective producers. That is impossible in principle
  for accumulating setters because it loses fold contributions.
- Loop-free inversion of every injective affine map. Some inverses need quotient/remainder
  index operations that `axis_index` does not currently have.
- Exact bounded integer feasibility for all affine maps. Start with the mixed-radix
  criterion and add exact feasibility only if real workloads need it.

## Why Injectivity Is the Soundness Line

`inline_computation` drops loops whose symbols are bound in the substitution environment.
That is sound only when the requested cell corresponds to exactly one iteration of those
dropped loops.

For non-injective setters, the stored cell is a fold over a fiber of producer iterations. A
single substitution instance reconstructs one element of that fold and loses the rest. The
correct fallback would be guarded replay of the producer loop nest, not substitution.

## Affine Injectivity Analysis

### Per-Position Criterion

For one affine position:

```text
idx = c1 * s1 + ... + cm * sm + offset
```

Normalize first:

- coalesce repeated symbols by summing coefficients;
- drop zero coefficients;
- treat range-1 symbols as static;
- ignore offset for injectivity;
- sort remaining terms by ascending `abs(coeff)`.

The position is injective on its remaining symbols if, for every sorted term `k >= 2`:

```text
abs(c_k) >= 1 + sum_{i < k} abs(c_i) * (range_i - 1)
```

This mixed-radix test is sufficient, not necessary. It accepts practical stride/window forms
such as `2 * oh + wh` with `wh` range 2, and rejects overlapping forms such as `oh + kh`
when both ranges exceed 1.

The known incomplete case is:

```text
idx = 3 * a + 4 * b
a range = 3
b range = 2
```

The values are distinct, but the mixed-radix criterion rejects it. Exact bounded integer
feasibility can be added later behind the same interface.

### Whole-LHS Criterion

Use a pinning fixpoint:

1. Start with static and range-1 symbols pinned.
2. For each LHS position, remove already pinned terms from consideration.
3. If the residual position passes the per-position criterion, pin its remaining symbols.
4. Repeat until no more symbols are pinned.
5. The LHS is injective if all non-static symbols are pinned.

This accepts triangular cases such as:

```text
(s1, s1 + s2)
```

where the first position pins `s1`, then the second pins `s2`.

## Validation

`check_idcs` should receive or reconstruct loop widths for symbols bound by the producer
computation. With those widths:

- accept multi-symbol affine producer positions only if the whole-LHS pinning fixpoint
  succeeds;
- preserve early rejection for non-injective forms, ideally with a distinct provenance code;
- continue rejecting `Concat`;
- continue enforcing structural consistency across accesses to the virtualized tensor.

This validation does not guarantee every consumer can be inlined. `make_subst` may still
reject a specific call site with `Non_virtual 13` when the call-site structure cannot be
matched or solved.

## Substitution Construction

In `make_subst`, process producer positions in the pinning order found by validation:

1. `Iterator s`: bind `s` to the call-site index, using Stage A grouping and guards.
2. Structural affine match: if producer and call-site positions are affine with the same
   canonical coefficient list and equal offset, bind producer symbols pairwise to the
   call-site symbols.
3. Unit-coefficient solving: after substituting already pinned symbols, if exactly one
   unbound producer symbol remains with coefficient `+1` or `-1`, bind it to the residual
   affine expression and emit range guards for the solved symbol.
4. Otherwise reject this consumer with `Non_virtual 13`.

Any non-pinning repeated or redundant positions become consistency guards:

```text
subst(producer_position) = call_site_position
```

If any guard is introduced and the original lowering did not emit an init local because the
setter was injective and surjective, explicitly initialize the local. Do not rely on backend
implicit local zeroing.

## Lowering Payoff

Improving `Indexing.is_injective` is useful before affine inlining itself. Genuinely
injective scatters can lower to plain setters instead of neutral-init plus read-modify-write.

Example: max-pool backward with `stride = window` has an input-gradient index like
`2 * oh + wh`. The map is injective and can skip accumulation. This also helps inlining:
substitution-based affine inlining should only encounter setters whose dropped loops are
known not to accumulate.

## Fallbacks and Future IR Work

Non-injective setters can be inlined correctly only by replaying the producer loop nest and
guarding each write to the requested cell. Today's LLC can represent that, but it is usually
too expensive because it turns each read into a scan over the producer loop space.

Loop-free inversion of affine maps such as mixed-radix encodings needs index division and
remainder. `axis_index` currently has neither. A future IR extension could add forms such as:

```text
Quot of axis_index * int
Rem of axis_index * int
```

That would require updates to comparison, substitution, simplification, backend rendering,
and non-negative division semantics. It is not part of the first Stage B landing.

Plan of record: keep non-injective setters non-virtual. Do not add guarded replay in the
first Stage B landing; reconsider only if a real workload justifies the cost.

## Acceptance Criteria

- `Indexing.is_injective` returns true for:
  - `2 * oh + wh` with `wh` range 2;
  - `K * i + k` with `k` range `<= K`;
  - `3 * i + j` with `j` range `<= 3`;
  - two-position triangular maps such as `(s1, s1 + s2)`.
- `Indexing.is_injective` returns false for:
  - `i + j` with both ranges greater than 1;
  - `stride * o + k` when `k` range is greater than `stride`.
- The known incomplete case `3 * a + 4 * b` over ranges `(3, 2)` may remain classified as
  false.
- An injective pool-backward-style scatter lowers to a plain setter with no neutral-init
  pass.
- An injective affine scatter consumed through the same affine structure virtualizes with no
  intermediate buffer.
- A `2 * oh + wh`-style setter consumed at a plain iterator inlines with range guards and
  matches materialized execution on `sync_cc` and at least one GPU backend when available.
- A non-injective stride-1/kernel-3 convolution-input-gradient-style producer remains
  non-virtual under the relaxed checks, with a test pinning the reason.
- Existing Stage A tests and legacy single-symbol tests still pass.
- No `Concat` regression.

## Tests

Good starting locations:

- `test/operations/` for focused `Indexing.is_injective` unit coverage.
- `test/einsum/test_max_pool2d.ml` for pool-backward-style lowering and execution coverage.
- Existing inlining snapshot tests as a model for asserting that an intermediate buffer is
  removed.

Test groups:

- Injectivity unit cases for the mixed-radix and triangular criteria.
- Lowering snapshot for an injective scatter that should skip neutral initialization.
- Structural affine-match inlining.
- Unit-coefficient solving with range guards.
- Non-injective affine soundness pin.
- Regression for single-symbol inlining and Stage A diagonal inlining.
- Regression that `Concat` still raises the existing non-virtual path.

Standalone `.expected` files must include the standard two-line config lookup banner.

## Resolved Decisions

- Include unit-coefficient solving in Stage B, because pool-backward-style consumers are part
  of the target workload.
- Keep non-injective producers non-virtual. Do not add guarded replay in the first landing.
- Defer exact bounded integer feasibility. The mixed-radix criterion is the first accepted
  injectivity test.

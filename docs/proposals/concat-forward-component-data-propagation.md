# Concat forward: propagate concrete component data through projection derivation when a concat axis is constrained against a plain `Dim` axis

## Goal

A concrete (data-bearing) multi-component `Concat` (`++^`) output axis drops its
non-leading components' data when the concat axis is pointwise-combined with a
plain `Dim`-axis tensor of the same total size. The values that should appear at
the tail of the result axis are instead left at the operation's neutral element,
and middle components are misplaced.

This is a genuine correctness gap in concat **forward** execution, deterministic
and reproducible (confirmed by execution against `~/ocannl-staging` `master`).
It was surfaced by the retrospective of `task-887c4062` (Concat dim-solver
hardening, PR #66, durable learning #5) while building a high-level reachability
fixture for #66's AC1 GLB merge: the printed values were an unreliable witness
precisely *because* of this bug. Fixing it is a v0.7 release-robustness gate for
Concat shape & projections inference, and it unblocks the high-level AC1 GLB-merge
reachability fixture that currently can only be covered at the solver level.

## Acceptance Criteria

A concrete 3-component concat `(a, b, c) ++^ "ii;jj;kk => ii^jj^kk"` with
`a=[10;20]`, `b=[30;40;50]`, `c=[60;70]`, when pointwise-added against a plain
`Dim`-axis tensor of size 7 (zeros), produces `10 20 30 40 50 60 70` through
`Train.forward_once`. The reference symptom prior to the fix is `10 20 60 70 0 0 0`
(component `c` lands where `b` should, `b`'s data dropped, tail left at the neutral
element).

- [x] **Forward correctness restored.** The forward path correctly propagates all
  component data of a multi-component concat when the concat axis is constrained
  against a plain `Dim` axis by a broadcast binary op. The three reference cases
  that already work — a standalone concat, a unary consumer (`sin(...)`), and a
  scalar-broadcast add (`... + 0.0`) — continue to work.

- [x] **Middle components restored, not just the tail.** A free or concrete
  *middle* component (`(a, {m}, c) + fixed7`) produces correct output. (Pre-fix
  repro Case F produced `10 20 0 0 0 0 0`.) The fix must restore every misplaced
  component, regardless of position.

- [x] **Coverage across shapes.** Fixtures exercise: 2-component and 3+-component
  concats added against a `Dim` axis; the `Dim` operand on the **LHS** of the add
  (operand order); and nested concat (`flatten_concat`, from #66) added against a
  `Dim` axis. These need not all live in distinct tests, but the regression suite
  must witness more than the single 3-component-tail case.

- [x] **Hard-fail on a *missing* component size in `concat_offset_for`.** The
  current `Map.find iter_sizes s |> Option.value ~default:0` silently drops data
  when a component's iterator symbol is **absent** from `iter_sizes` (projection
  derivation dropped the component). Replace this silent default with an
  assertion / `Utils.User_error` that fires **only** when the iterator symbol is
  absent from the map. A component whose size is **present and resolved to `0`**
  (a genuinely empty component) is legitimate and must NOT trip the guard. The
  guard must distinguish "key absent from the map" (error) from "key present,
  value `0`" (allowed) — do not key the check off the `0` value. Apply this in
  **both** `loop_accum` (Block) and `loop_accum_rev` (Rev_sides) copies of
  `concat_offset_for`.

- [x] **Gradient direction audited and fixed.** Audit the `Rev_sides` /
  `loop_accum_rev` gradient path for the same offset corruption (it shares the
  `concat_offset_for` + projection-derivation machinery). If incoming gradient is
  misrouted to the wrong component under a `concat + Dim` shape, fix it. Add a
  gradient-through-`(concat + Dim)` fixture (the existing Tests 7/8 cover
  standalone-concat gradients, which are correct). If, after reproducing, the
  gradient path turns out genuinely unaffected, the fixture still ships as a
  regression guard and the audit finding is recorded in the test comments — this
  case is **not** split to a follow-up.
  **Audit finding**: The gradient path was NOT affected by the same bug. The root cause was
  in `low_level.ml`'s virtualizer (not in `concat_offset_for`), and the gradient path's
  `Rev_sides` For_loops write to component gradients directly (not Block-virtualized). Test
  5h ships as the regression guard (grad_q1=[1,1], grad_q2=[1,1,1], grad_q3=[1,1] ✓).

- [x] **High-level AC1 GLB-merge reachability fixture added.** With the forward
  bug fixed, add a high-level `%op` add-of-concat-vs-`Dim` (or
  pointwise-add-of-two-stacked-tensors) test to `test_block_tensor` that witnesses
  the #66 AC1 GLB merge between two `Concat` bounds with **real, asserted values**
  (not uninitialized memory). Re-point the stale deferring comments in
  `test_block_tensor.ml` (the block currently noting "an unrelated concat-forward
  concern" / "the printed result values are not a reliable witness") at the new
  fixture, since the witness is now reliable.
  **Done**: Test 5i (`glb_result = (r1,r2,r3 concat) + (s1,s2,s3 concat)`) produces
  [11, 22, 33, 44, 55, 66, 77] ✓. The stale deferring comments were replaced.

- [x] **Test 5e expectation corrected.** `test_block_tensor`'s Test 5e
  (`((a_e, b_e, {c_e}) ++^ …) + fixed7`) currently ships a `.expected` that
  encodes the bug (its trailing `1.00 2.00` is `a_e`'s data leaking, not `c_e`).
  Once the fix lands, this expectation reflects correct data propagation.

- [x] **Existing tests stay green.** `dune test` (or the project's test entry
  point) passes; no regression in the standalone-concat, unary-consumer, or
  gradient fixtures.

## Context

### How concat forward execution works now

The forward copy itself is correct — the corruption is **upstream**, in
projection derivation. The component write offsets are computed at lowering time
from the projection's per-component sizes; if those sizes are wrong or absent,
offsets are wrong and the tail of the axis is never written.

**Lowering — `arrayjit/lib/assignments.ml`:**
- `loop_accum` (Block / forward) and `loop_accum_rev` (Rev_sides / gradient) each
  build an `iter_sizes` map from `projections.product_space` ×
  `projections.product_iterators`, then define a local `concat_offset_for syms
  active` that walks the `Concat` symbol list accumulating `Map.find iter_sizes s
  |> Option.value ~default:0` to compute each component's write offset. The
  `~default:0` is the silent failure mode: a missing component size yields
  offset/size `0` and quietly drops data.
- The `Indexing.Concat syms` arm of `subst_index` (in each of the two `basecase`
  closures) resolves the active component to an `Iterator` (offset `0`) or an
  `Affine { symbols = [(1, s')]; offset }`.
- `is_allowed_by_concat` and `concat_syms_opt` gate which RHS/target participates
  per active component.

**Projection derivation (root) — `tensor/shape.ml`, `derive_projections`:**
- `all_dims` collects every dim across LHS + RHSes.
- `all_product_projs` maps each dim through `Row.get_product_proj proj_env` and
  deduplicates **by proj_id** via `Utils.unique_keep_first`. Crucially,
  `get_product_proj` (in `tensor/row.ml`) resolves through `proj_repr proj_env
  proj_id` — the union-find representative. So when the concat axis is unified
  against the add operand's plain `Dim 7` axis, the three component proj_ids can
  collapse onto a single representative, and the three distinct component sizes
  `2, 3, 2` no longer survive into `product_space`.
- `unique_by_iterator` further dedups by iterator symbol; `symbol_to_proj` /
  `product_dim_iterators` backfill concat component iterators; `concat_groups` +
  the union-find (`find_repr` / `union`) group concat symbols so components
  iterate **sequentially**; the grouped result becomes `product_space` /
  `product_iterators`.
- Net effect (hypothesis to confirm via diff, below): when the concat axis is
  constrained against a `Dim` axis, `product_space` no longer carries three
  distinct component sizes, so `concat_offset_for` reads the wrong per-component
  sizes (some defaulting to `0`).

**Operations — `tensor/operation.ml`:** `concat_sum` (a.k.a. `concat`, the `++^`
op label) builds `Asgns.Block { op; rhses }` for the forward direction and
`Asgns.Rev_sides { op; lhses }` for the gradient. `stack` introduces a fresh
leading axis on top of `concat_sum`.

**Solver interaction — `tensor/row.ml`:** the `Concat`-vs-`Dim` / `solve_dim_ineq`
/ `unify_dim` resolution touched by `task-887c4062` (PR #66) is what leaves behind
the proj_id mapping that the projection corruption is downstream of. The bug is
**not introduced by #66** — #66 changed only the dim *solver* arms, not the
projection-derivation or Block-lowering forward path — but it is *reached* by the
same `Concat`-vs-`Dim`-via-broadcast shapes that #66's AC1 cares about, which is
why it blocked the AC1 high-level reachability fixture.

### Reproduction reference (confirmed by execution on `master` @ PR #66 merge)

| Case | Expression (all components concrete) | Result | Verdict |
|------|--------------------------------------|--------|---------|
| A | `(a,b,c) ++^ "ii;jj;kk => ii^jj^kk"` standalone | `10 20 30 40 50 60 70` | correct |
| G | `sin((a,b,c) ++^ …)` | sin of all 7 | correct |
| H | `((a,b,c) ++^ …) + 0.0` (scalar broadcast) | `10 20 30 40 50 60 70` | correct |
| D | `((a,b,c) ++^ …) + fixed7` (`fixed7` size-7 `Dim` of zeros) | `10 20 60 70 0 0 0` | **WRONG** |
| F | `((a,{m},c) ++^ …) + fixed7` (free middle) | `10 20 0 0 0 0 0` | **WRONG** |

with `a=[10;20]`, `b=[30;40;50]`, `c=[60;70]`.

The free-parameter trailing case the retrospective originally described
(`(a,b,{c}) + fixed7`, uninitialized-memory tail) is the **same** root cause as
the concrete-`c` case — the two phrasings in durable-learning #5 are one bug,
observed once with a free `c` and once with a concrete `c`.

### Code pointers (by symbol; line numbers drift)

- `arrayjit/lib/assignments.ml`: `loop_accum`, `loop_accum_rev`,
  `concat_offset_for` (both copies), `is_allowed_by_concat`, the `Indexing.Concat
  syms` arm of `subst_index` (in both `basecase` closures), `basecase`.
- `tensor/shape.ml`: `derive_projections` — `all_dims`, `all_product_projs`,
  `unique_by_iterator`, `symbol_to_proj`, `concat_groups`, the union-find
  (`find_repr` / `union`), `product_space`, `product_iterators`, `indices_of_sh`.
- `tensor/row.ml`: `get_product_proj` (resolves through `proj_repr`),
  `proj_to_iterator_exn`, `product_dim_iterators`, `get_dim_index`; the
  `Concat`-vs-`Dim` / `solve_dim_ineq` / `unify_dim` resolution from #66.
- `tensor/operation.ml`: `concat_sum` / `concat`, `stack`, the `++^` op label
  (`Asgns.Block` forward, `Asgns.Rev_sides` gradient).
- Fixtures: `test/operations/test_block_tensor.{ml,expected}` — Test 5e
  (`((a_e,b_e,{c_e}) ++^ …) + fixed7`) is the shipped fixture that already
  exhibits the bug; the comment block immediately following it is the coder's note
  deferring the unblock to this task. Tests 7/8 cover standalone-concat gradients.

### Diagnostic starting point

The implementer should dump `proj_env`, `all_product_projs`, `product_space`, and
the resolved `project_lhs` `Concat` symbols for Case A (correct) vs Case D (wrong)
and diff them — that diff pins the exact proj_id collapse and tells whether the
fix belongs in `get_product_proj` (don't collapse component proj_ids across the
`Concat`-vs-`Dim` union), in `derive_projections` (preserve distinct component
sizes when deduplicating), or in the solver's `Concat`-vs-`Dim` mapping
(`tensor/row.ml`).

## Scope

**In scope:** the forward projection-derivation / lowering bug; the matching
`Rev_sides` gradient audit + fix (if affected); the `concat_offset_for` hard-fail
guard (both copies, missing-key only); regression fixtures (forward across shapes
+ operand order + nested concat, gradient-through-`concat + Dim`); the high-level
AC1 GLB-merge reachability fixture and re-pointing the stale `test_block_tensor`
comments; correcting Test 5e's `.expected`.

**Out of scope:** any change to #66's solver arms beyond what is strictly required
to stop the proj_id collapse; broader projection-derivation refactors unrelated to
the concat-vs-`Dim` interaction; performance work.

**Dependencies:** relates to `task-887c4062` (PR #66, merged) — this task is
downstream of it but does not depend on further #66 work. No blocking
dependencies.

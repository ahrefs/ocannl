# Ternary einsum notation / projection inference

## Goal

Support 3-operand einsum notation `"rhs1 ; rhs2 ; rhs3 => lhs"` in OCANNL's
`%cd` PPX, shape/projection inference, and a corresponding tensor-level entry
point, with working gradients. Removes the explicit "not supported yet"
blocker in `ppx_cd.ml` that references this issue.

Tracks [issue #305](https://github.com/ahrefs/ocannl/issues/305): "It's not
needed, but cool and not much work."

## Status update (2026-06-12)

- Issue #305 is still OPEN, milestone v1.0 (end of October 2026 per ROADMAP.md). Harness status: deferred. No implementation work has started.
- The blocker is unchanged: the `"einsum notation for ternary operators not supported yet, see issue #305"` error builder is at `tensor/ppx_cd.ml:1594-1595`, and `Shape.ternary_type` still has only `Pointwise_tern | Compose_accumulate | Defined_by_cd_logic` (`tensor/shape.mli:196-199`).
- All "verified in place" preconditions re-verified against current code: `parser.mly` `rhs_specs` recursion (line 266-268), `RHS3` slot machinery in `ppx_cd.ml`, `fma`/`where` via `Pointwise_tern` (`operation.ml:436,445`), `delayed_var_ref` in `shape.mli`, and `Ir.Ops.ternop` still only `Where | FMA` (`ops.ml:438-441`) — `Mul3` remains to be added under option (a).
- `shape.ml` has churned heavily since this proposal: broadcast-order reversal (LUB->GLB, meet->join; "⊑" now reads "refines"), dimension "label" -> "basis" rename, total basis with reserved `bcast_if_1` bottom, and the tensor-stacking `Block { spec; _ }` constructor (commit 58bfd6e5). The binary reference branch `Broadcast (Einsum (spec, dim_refs), sh1, sh2)` is now at `shape.ml:1054`, with the duplication-refactor TODO at line 1087. New constraint code must use the post-reversal join-semilattice vocabulary; einsum axis variables in spec strings are unaffected by the basis rename.
- The recently-added `Block` spec-carrying constructor in the shape-logic types is a fresh precedent for the proposed `Einsum_tern` addition (similar plumbing: constructor + constraint branch + projections branch).
- Cited test files (`test/ppx/test_ppx_op.ml`, `test/training/mlp_names.ml`, `test/ppx/test_ppx_name_conflict.ml`) all still exist.
- Verdict: proposal remains accurate and actionable; only the `shape.ml` reference line numbers and surrounding vocabulary have moved.

## Acceptance Criteria

*(Rescoped 2026-06-12 per the Complexity reassessment below: `Where` einsum
is in scope, FMA-einsum is out of scope; `Mul3`/`einsum3` is a sub-decision,
recommended in. The `"."`/`"@"` ternary `~logic` type bug cited by the design
review was fixed separately in commit `eec37739` — no AC needed for it beyond
a regression test if convenient.)*

- [ ] In `%cd` blocks, `where` (and `mul3`, if the Mul3 sub-decision is
  accepted) with `~logic:"<einsum-spec>"` no longer raises the
  `"einsum notation for ternary operators not supported yet, see issue #305"`
  error. The spec routes through ternary einsum shape inference and
  projection derivation. `fma` with an einsum spec remains an error (with a
  message explaining the accumulation-semantics ambiguity, no `#305`
  reference) unless Łukasz opts for allow-and-document.
- [ ] `Shape.ternary_type` gains a constructor representing ternary einsum
  (e.g. `Einsum_tern of string * delayed_var_ref list`), and
  `Shape.Broadcast_tern (Einsum_tern ..., sh1, sh2, sh3)` participates in
  constraint generation in `shape.ml` **via a shared n-ary helper extracted
  from the `Block` branch** (see Design review rec. 1), discharging the
  duplication TODO at `shape.ml:1087`. `logic_to_spec` and
  `update_delayed_var_refs` gain `Einsum_tern` cases (and `Block`, fixing
  the existing omission).
- [ ] Projections: for a 3-operand spec, the iteration product space is the
  union of all spec labels; labels absent from the LHS become reduction
  axes; a shared label appears once in the product. Verified by the
  numeric tests below rather than by inspecting projections directly.
- [ ] **Where semantics (select-before-reduce)**: `where` with spec
  `"p ; a ; b => out"` computes, at every point of the product space, the
  predicate-selected element of branch 2 or 3, and accumulates that into
  `out` (predicate evaluated per-element *before* reduction). Test against
  the decomposed reference: per-operand `einsum1` permutes + pointwise
  `where` + `++` reduce. Masks of 0/1 and small-integer data make float
  results exact.
- [ ] **Where gradient**: gradient flows to branches 2 and 3 masked by the
  predicate (`g2 =+ where v1 g 0`, `g3 =+ where v1 0 g` under the *same*
  projections as the forward); no gradient to the predicate. Test agreement
  with the decomposed reference on a small randomized case.
- [ ] Documented caveat: when a reduced axis is absent from a branch (branch
  broadcasts along it), the broadcast value is accumulated once per
  reduction iteration in which it is selected ("count-weighted") — this is
  the honest semantics of reduce-after-select and must be stated in the
  `where` doc comment.
- [ ] *(Mul3 sub-decision)* Pure-contraction chain: `"ij ; jk ; km => im"`
  produces the same numeric result as the binary chain
  `(a *+ "ij;jk=>ik" b) *+ "ik;km=>im" c` for small-integer-valued inputs
  (exact float arithmetic; random floats would need a tolerance since the
  fused kernel sums in a different order). This is the **falsifier** test
  for projection inference. Keep test dims tiny: the fused kernel is
  O(I·J·K·M).
- [ ] *(Mul3 sub-decision)* A batched test `"bij ; bjk ; bkm => bim"` matches
  the equivalent binary chain, and a gradient-agreement test on `a`, `b`,
  `c` against the binary chain.
- [ ] *(Mul3 sub-decision)* `einsum3`'s doc comment warns that for
  chain-structured contractions the fused kernel iterates the full product
  space (e.g. O(N⁴) for triple matmul vs O(N³) chained) and recommends
  binary chains there; ternary einsum is for shared-reduction patterns.
- [ ] An `operation.ml` entry point exposes ternary einsum from OCaml code,
  not only from `%cd` blocks: at minimum a spec-taking `where` variant
  (e.g. `where_einsum spec` or `?logic` on `where`), plus `einsum3` under
  the Mul3 sub-decision. Entry points accept `?capture_dims` like `einsum`.
- [ ] A clear `Shape_error` when the spec has ≠ 3 RHS for a ternary op
  (mirroring the binary "expected two arguments" message), with a negative
  test.
- [ ] No regression in existing binary-einsum and unary-permute tests
  (`test/ppx/test_ppx_op.ml`, `test/training/mlp_names.ml`,
  `test/ppx/test_ppx_name_conflict.ml`) nor in concat/stacking tests (the
  shared-helper refactor touches the `Block` path).
- [ ] The `[%expr ...]` error builder at the ternary-spec branch in
  `ppx_cd.ml` no longer references `#305`.

## Context

### What is already in place (verified)

- **Parser**: `tensor/parser.mly` rule `rhs_specs` already recurses on
  `SEMICOLON`, and `einsum_spec` returns
  `parsed_axis_labels list * parsed_axis_labels`. The list is unbounded —
  no parser change is required to accept 3+ RHS specs.
- **PPX RHS3 plumbing**: `tensor/ppx_cd.ml` has the full `RHS3` slot
  machinery: `projections_slot` includes `RHS3`, `ternary_op` lookup,
  `process_assign_ternop` and `process_raw_ternop` set up `setup_r3`,
  build `rhs_dims = [| rhs1_dims; rhs2_dims; rhs3_dims |]` and
  `project_rhs = [| project_rhs1; project_rhs2; project_rhs3 |]`, and emit
  `Ternop { op; rhs1; rhs2; rhs3 }` into `Accum_op`.
- **Assignments IR**: `arrayjit/lib/assignments.ml` `Ternop` is a buffer
  triple. Iteration order is determined by the `projections` field of
  `Accum_op`, *not* by the op constructor. So the IR is agnostic to whether
  the projections are pointwise or contraction-style — it will iterate
  whatever projection structure `derive_projections` produces.
- **Existing ternary ops**: `operation.ml` defines `fma` and `where` via
  `Tensor.ternop ~ternary_op:Pointwise_tern`. These are pointwise.
- **Ternary shape branches**: `Shape.Broadcast_tern` exists with
  `Pointwise_tern | Compose_accumulate | Defined_by_cd_logic` variants; the
  first two have constraint-generation branches in `shape.ml`.

### The actual blocker

The error message at the spec dispatch site for ternary ops in `ppx_cd.ml`
(in the branch handling `accu_op lhs (tern_op (rhs1, rhs2, rhs3) ~logic:"...")`):

```
"ppx_ocannl %%cd: expected <.> or <@>, found <%s> -- einsum notation for ternary
 operators not supported yet, see issue #305"
```

The branch only accepts `"."` (mapped to `Shape.Pointwise_bin`) and `"@"`
(mapped to `Shape.Compose`) as the `logic` payload — it has no `Shape.Einsum`
analog because `Shape.ternary_type` has no `Einsum` constructor.
*(Update 2026-06-12: those existing mappings are themselves a latent type bug —
`process_raw_ternop` passes `~logic` to `Tensor.raw_ternop`, which expects
`Shape.ternary_type` (`tensor/tensor.mli:96`), but `Pointwise_bin`/`Compose`
are `compose_type` constructors. Any `%cd` use of a ternary op with
`~logic:"."` or `~logic:"@"` generates ill-typed code today. The fix must map
`"."` → `Shape.Pointwise_tern` and `"@"` → `Shape.Compose_accumulate`,
matching the defaults in `ppx_shared.ml:514-515`.)* Compare the
binary case in the same file (a few lines earlier), which falls through to
`Shape.Einsum (logic, [])` for any other spec.

### What is missing (the real work)

1. **`Shape.ternary_type`** (`tensor/shape.mli`, `tensor/shape.ml`) needs a
   new constructor for ternary einsum, carrying the spec string and
   `delayed_var_ref list` (mirroring binary `Einsum`).
2. **Shape constraint generation** (`shape.ml`): a new branch in the big
   `match` that handles `Broadcast_tern (Einsum_tern (spec, dim_refs), sh1,
   sh2, sh3)`. The binary branch (`Broadcast (Einsum ...)`) is ~140 lines
   and already carries a `TODO: refactor to avoid duplication with the one
   for unary einsum`. A ternary copy is feasible but a refactor that
   parameterizes over arity (1/2/3 RHS via `parsed_axis_labels list`) would
   be cleaner. Choice left to implementer; refactor is preferred.
3. **`derive_projections`** for the ternary einsum branch — produce the
   union product space, with reduction axes for any label that is in some
   RHS but not in LHS. The binary case in the same function is the
   reference. *(Update 2026-06-12: this item is overstated.
   `derive_projections` (`shape.ml:1897`) is generic: it re-runs
   `get_inequalities ~for_projections:true` and only matches on
   `update_step.logic` to collect the RHS shape list, where the wildcard
   `Broadcast_tern (_, sh1, sh2, sh3)` (`shape.ml:1933`) already covers any
   new `ternary_type` constructor. No new `derive_projections` branch is
   needed; the projection product space falls out of the `proj_env` built by
   the new `get_inequalities` branch. The smaller real additions are:
   `logic_to_spec` (`shape.ml:94`) and `update_delayed_var_refs`
   (`shape.ml:2225-2228`) need `Einsum_tern` cases — the latter currently
   handles only `Permute` and binary `Einsum`, and notably omits `Block` as
   well.)*
4. **Logic dispatch in `ppx_cd.ml`**: replace the error builder so that any
   spec other than `"."`/`"@"` is wrapped as
   `Shape.Einsum_tern (logic, [])` (mirroring the binary case at the
   sibling branch).
5. **A pointwise primitive for the underlying scalar op.** This is the one
   subtlety. The existing `Ternop` primitives are `fma(a,b,c)=a*b+c` and
   `where(c,a,b)`. Neither cleanly expresses pure ternary contraction
   (which wants `a*b*c` accumulated via `+=`). Two options:

   - **(a)** Add a new ternary scalar op `Mul3` in `arrayjit/lib/ops.ml`
     and emit it from the new `einsum3` operation builder. This is the
     orthogonal, principled choice.
   - **(b)** Decompose ternary einsum at `operation.ml` into a chained
     binary einsum `(a *+ spec12 b) *+ spec23 c`, where the splits are
     derived from the parsed spec. The PPX still routes through the new
     ternary einsum shape-inference path (so projections are unified
     across all three operands at the shape level), but execution uses
     two binary kernels. Skips the scalar-op work but introduces an
     intermediate buffer.

   The proposal accepts either option. Recommendation: start with (a) for
   single-kernel execution and to keep the operation orthogonal; fall back
   to (b) only if backend codegen for `Mul3` proves disproportionate.

6. **Gradient**: each gradient of a ternary einsum is itself a ternary or
   binary einsum (e.g. `d/dA(A*B*C contracted)` is `B*C` contracted with the
   incoming gradient). Implement in `operation.ml` alongside `einsum`.

### Code pointers

- Blocker dispatch: `tensor/ppx_cd.ml`, function processing
  `accu_op lhs (tern_op (rhs1,rhs2,rhs3) ~logic)` — search for the
  `"einsum notation for ternary operators not supported yet"` string.
- Binary einsum logic-dispatch reference: same file, the analogous
  `process_raw_binop` dispatch ~30 lines above.
- Binary einsum shape constraints: `tensor/shape.ml`, branch
  `Broadcast (Einsum (spec, dim_refs), sh1, sh2)` — ~140 lines, marked
  with refactor TODO *(as of 2026-06-12: branch at line 1054, TODO at
  line 1087)*.
- Ternary shape constraints (existing): `tensor/shape.ml`, branches
  `Broadcast_tern (Compose_accumulate, ...)` and
  `Broadcast_tern (Pointwise_tern, ...)`.
- `compose_type` / `ternary_type` definitions: `tensor/shape.mli`.
- Ternary scalar ops registry: `tensor/ppx_cd.ml` `ternary_ops` Hashtbl
  (currently `where`, `fma`).
- Existing ternary tensor ops: `tensor/operation.ml`, `fma`, `where`.
- Binary einsum API: `tensor/operation.ml`, `einsum`, `pointmul`,
  `outer_sum` (search for `Shape.Einsum (spec, capture_dims)`).
- Tests to consult for the testing pattern: `test/ppx/test_ppx_op.ml`
  (uses `+*` infix einsum), `test/training/mlp_names.ml`.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

1. Add `Einsum_tern of string * delayed_var_ref list` to `Shape.ternary_type`
   in `shape.mli` / `shape.ml`, with a string-form representative for
   pretty-printing similar to existing variants.
2. In `shape.ml`, add a `Broadcast_tern (Einsum_tern (spec, dim_refs), sh1,
   sh2, sh3)` branch to constraint generation. Implementer's discretion
   whether to copy-adapt the binary branch or refactor binary+ternary onto
   a shared helper that takes a `parsed_axis_labels list`. Refactor is
   preferred because the binary branch already carries a TODO for it; the
   refactor delivers ternary "for free" and pays down debt.
3. ~~Add a `Broadcast_tern (Einsum_tern ..., ...)` branch to
   `derive_projections` (same file).~~ *(Update 2026-06-12: not needed —
   `derive_projections` is generic over `Broadcast_tern (_, ...)`; see the
   corrected item 3 in Context above. Instead, add `Einsum_tern` cases to
   `logic_to_spec` and `update_delayed_var_refs` in `shape.ml`.)*
4. In `ppx_cd.ml`, replace the error in the ternary-spec dispatch with the
   binary-style `Shape.Einsum_tern (logic, [])` wrapping (drop the `#305`
   reference from any remaining diagnostic).
5. Pick option (a) or (b) from Context §"What is missing" item 5. If (a):
   add `Mul3` (or similar) to `Ir.Ops.ternop` and supply the codegen for
   relevant backends (CPU C and CUDA at minimum; backends are listed in
   `arrayjit/lib/`).
6. Add `einsum3 spec t1 t2 t3` (or three-operand `*+`) in `operation.ml`,
   returning a tensor with the right `op_asn` and `grad_asn` per option (a)
   or (b).
7. Tests in `test/ppx/`:
   - `"ij;jk;km=>im"` matches the binary chain numerically.
   - `"bij;bjk;bkm=>bim"` batched.
   - Gradient agreement test against the binary chain.
   - At least one test that exercises `%cd` with a ternary einsum spec
     directly (verifying the PPX dispatch fix).
8. Run the existing test suite to confirm no regressions in binary einsum
   or `fma`/`where`.

## Scope

**In scope**:
- Single new ternary einsum variant in `Shape.ternary_type`.
- Shape inference, projection inference for 3-RHS einsum.
- PPX dispatch fix.
- One operation-level entry point with gradient.
- A minimal scalar primitive (new `Mul3` or chained-binary fallback).
- Tests covering the falsifier scenario (chain contraction equality) and
  gradient agreement.

**Out of scope**:
- N-ary (4+) einsum. The codepaths added should not preclude future
  generalization, but no N-ary work is performed here.
- New surface syntax beyond the existing `;`-separator convention.
- Refactoring binary/unary einsum infrastructure beyond what is needed to
  avoid copy-pasting the constraint branch a third time. (If the
  refactor becomes large, ship the duplication and file a follow-up.)
- Backend optimization of the new scalar op beyond functional correctness.

**Dependencies**: none.

**Effort note**: The original "small (1-2 days)" estimate assumed only the
PPX error needed removal. Verified investigation shows `Shape.ternary_type`
needs a new variant plus a new ~100-line shape-constraint branch (or a
refactor of the binary branch), `derive_projections` needs a new branch, and
a new scalar primitive plus its codegen are required for option (a). A more
realistic estimate is **medium (3-5 days)**, dominated by the shape.ml work
and tests. If the refactor is taken, the ceiling rises but unary/binary
einsum benefit. If the implementer prefers option (b) (decomposition into
chained binary einsums) they may be able to keep this closer to small, at
the cost of an intermediate buffer per call.

## Design review (2026-06-12)

**Verdict: sound-with-changes.** The decomposition of the work is right and
all preconditions check out, but the proposal misses an existing n-ary
precedent that changes the recommended implementation strategy, repeats a
latent type bug, and overstates the `derive_projections` work (fixed in
place above).

**Recommendations** (strongest first):

1. **Refactor target is the `Block` branch, not the binary branch.** The
   `Block { spec; delayed_vars; rhses }` branch in `get_inequalities`
   (`shape.ml:1288-1498`) is *already* n-ary einsum constraint generation:
   `einsum_of_spec`, per-RHS `einsum_slot_spec_to_dims_bio` over a list,
   delayed-var binding, and per-RHS `Row_eq` triples. The delayed-var
   binding code is duplicated verbatim between the binary `Einsum` branch
   (`shape.ml:1088-1141`) and `Block` (`shape.ml:1321-1372`); a copy-adapted
   ternary branch would make it a third copy. Extract the n-ary core (sans
   `compute_block_discardable_vars`, which is concat-specific) into a shared
   helper used by `Einsum`, `Einsum_tern`, and `Block`. The `Einsum_tern`
   branch then shrinks to ~30 lines and the existing TODO at
   `shape.ml:1087` is discharged. Given this, consider going straight to
   `Einsum_nary of string * delayed_var_ref list` semantics internally even
   if the public constructor stays ternary.
2. **Choose option (a) (`Mul3`) firmly, not as a soft default.** Option (b)
   does not serve the `%cd` path at all: the PPX emits a single
   `Ternop`-carrying `Accum_op`, so decomposition at `operation.ml` cannot
   apply there, and the falsifier test would then exercise the
   decomposition rather than ternary projections. `Mul3` codegen is trivial
   shared infix syntax in `ternop_c_syntax` (e.g. `("((", ") * (", ") * (",
   "))")`) — no per-backend builtins needed; also update `interpret_ternop`,
   `ternop_cd_syntax`, and `is_homogeneous_prec_ternop` (→ true).
3. **Gradients reuse forward projections; no spec manipulation needed.**
   Mirror `pointmul` (`operation.ml:71-82`): the grad of `einsum3` is
   `g1 =+ mul3 g v2 v3; g2 =+ mul3 v1 g v3; g3 =+ mul3 v1 v2 g` under the
   *same* projections — the slot-permutation machinery already does this for
   `fma`'s gradient. This requires registering `mul3` in
   `ppx_shared.ml` `ternary_ops` (with default logic `Pointwise_tern`) so
   `%cd` can emit it. Drop the proposal's "each gradient is itself a
   ternary or binary einsum" framing — constructing transposed specs is
   unnecessary and error-prone.
4. **Fix the latent `"."`/`"@"` type bug as part of this change** (see the
   in-place update under "The actual blocker"): the dispatch must produce
   `Shape.Pointwise_tern`/`Shape.Compose_accumulate`, and ideally a test
   should cover `fma ... ~logic:"."` in `%cd`, which cannot compile today.
   Add this to the acceptance criteria.
5. **Pin down spec-arity errors and capture-dims.** The new branch must
   raise a clear `Shape_error` when `einsum_of_spec` yields ≠ 3 RHS
   (mirroring the binary "expected two arguments" message), the
   `operation.ml` entry point should accept `?capture_dims` like `einsum`,
   and concat-`^` axes in ternary specs should behave exactly as in the
   binary branch (pass `dim_var_set_empty` for discardables; Block-style
   discardable-vars logic is *not* wanted here). Add a negative test for
   arity mismatch.

**Open decision points for Łukasz**:

- **Semantics of einsum logic for `Where`/`FMA`.** Once the PPX dispatch
  accepts arbitrary specs for any registered ternary op, `where`/`fma` with
  contraction specs become expressible, and their accumulation semantics
  are murky (e.g. FMA's `+ c` term is re-added on every reduction
  iteration). Allow and document, or restrict einsum logic to `Mul3`?
  *(Resolved 2026-06-12: `Where` IS allowed as a ternary einsum op — its
  select-before-reduce semantics is well-defined, see the Complexity
  reassessment below. FMA is dropped from einsum scope as low value; `fma`
  keeps only its existing `Pointwise_tern`/`Compose_accumulate` logics.)*
- **Refactor scope**: shared n-ary helper now (recommended; pays down the
  existing 2x duplication) vs. shipping a third near-duplicate branch with
  a follow-up issue.
- **Surface API**: standalone `einsum3 spec t1 t2 t3` (least invasive) vs. a
  three-operand infix in `TDSL.O`/`NTDSL.O`; and whether `update_var_ref_list`'s
  current omission of `Block` (`shape.ml:2225-2228`) is a separate bug worth
  filing while touching that match.

## Complexity reassessment (2026-06-12)

**Decisions recorded (Łukasz, 2026-06-12)**: `Where` is allowed as a ternary
einsum op. FMA-einsum is dropped from scope — `fma` keeps only its existing
non-einsum logics (`Pointwise_tern`, `Compose_accumulate`). Separately, the
latent `"."`/`"@"` ternary `~logic` type bug flagged by the design review has
since been fixed (commit `eec37739`, `ppx_cd.ml:1589-1590` now emit
`Shape.Pointwise_tern`/`Shape.Compose_accumulate`).

**Central question**: now that n-ary einsum constraint generation exists (the
`Block` branch, built for concatenation/stacking), is a dedicated ternary
einsum still worth the added complexity at all?

### Why `Block` does not subsume ternary einsum

The overlap is at the *shape/constraint* level only. At the *execution* level
the two are different animals: `Asgns.Block { op; rhses }` applies a unary op
to each RHS and accumulates each RHS **independently** into (a slice of) the
LHS — concatenation semantics. `Asgns.Ternop { op; rhs1; rhs2; rhs3 }`
combines three values **at each iteration point** of one product space. No
spec passed to `Block` can express `where(p,a,b)` or `a*b*c` per-point. So
"just use Block" is not an answer; the question is whether the per-point
ternary forms are worth their (now much smaller) cost.

### Path (a): implement, via a shared n-ary helper

What `Block` changed: the constraint-generation work — the proposal's
dominant cost item — is no longer "write a third ~140-line branch" but
"extract a helper both existing branches already want":

- **Shared helper extraction** (`get_inequalities`): the binary `Einsum`
  branch (`shape.ml:1054-1287`, ~233 lines) and the `Block` branch
  (`shape.ml:1288-1498`, ~211 lines) are the same algorithm — per-RHS
  `einsum_slot_spec_to_dims_bio` over a list, name-clash check, a ~50-line
  delayed-var-binding block duplicated verbatim, `proj_env` merge, per-RHS +
  LHS `Row_eq` triples — differing only in fixed-vs-list arity, the
  `compute_block_discardable_vars` computation (Block-only; pass
  `dim_var_set_empty` for einsum arities), and origin labels ("Broadcast
  ARGUMENT n" vs "Block ARGUMENT n", parameterize). Extraction: ~200-line
  helper replacing ~440 lines; net **shape.ml shrinks ~200 lines** and the
  TODO at `shape.ml:1087` is discharged. The `Einsum_tern` branch then costs
  ~15 lines. Main risk: `.expected` churn if origin labels shift — keep the
  binary branch's labels bit-identical.
- **No `derive_projections` work**: confirmed generic — line 1933 already
  destructures `Broadcast_tern (_, sh1, sh2, sh3)` for any `ternary_type`
  constructor; the projection space falls out of the `proj_env` returned by
  the new `get_inequalities` branch.
- **Small additions**: `Einsum_tern of string * delayed_var_ref list` in
  `ternary_type` (+ `logic_to_spec` case at `shape.ml:94`,
  `update_delayed_var_refs` case at `shape.ml:2225-2228`, folding in the
  `Block` omission fix) ≈ 15 lines. PPX dispatch: replace the error builder
  at `ppx_cd.ml:1591-1596` with `Shape.Einsum_tern (logic, [])` (gated per
  op; `fma` keeps erroring) ≈ 10 lines.
- **`Mul3` primitive** (sub-decision): ~6 touch points, all trivial infix —
  `ops.ml` (`ternop` constructor, `interpret_ternop` `v1 *. v2 *. v3`,
  `ternop_cd_syntax`, `ternop_c_syntax` `("((", ") * (", ") * (", "))")`,
  `is_homogeneous_prec_ternop` → true), the `Ops.[ Where; FMA ]` list at
  `c_syntax.ml:113`, and the `ternop_syntax` matches in
  `cuda_backend.ml:770` / `metal_backend.ml:522` (compiler-enforced). No
  builtins needed. ≈ 30 lines total.
- **`operation.ml` entry points**: `where` einsum variant (op_asn
  `v =:+ where v1 v2 v3`, grad as in the ACs) and `einsum3` (op_asn
  `v =:+ mul3 v1 v2 v3`, grads `g1 =+ mul3 g v2 v3` etc. under the *same*
  projections, mirroring `pointmul`/`fma` slot permutation) ≈ 25 lines, plus
  `mul3` registration in `ppx_shared.ml` `ternary_ops`.

**Total (a)**: ≈ 70-90 lines of new logic outside tests, riding on a
~200-line-net-negative refactor that is justified independently. Tests
~150-250 lines. Realistic effort: 3-4 days, dominated by refactor
verification and test churn — down from the pre-`Block` estimate where the
constraint branch alone was the medium-sized item.

### Path (b): don't implement — what users actually lose

Quantified against today's machinery (`low_level.ml` virtualization):

- **Pointwise-then-reduce composites already fuse.** Patterns like masked
  sum `(where mask x 0) ++ "ij=>i"` or `(a *. b *. c) ++ "..=>.."` build
  single-assignment pointwise intermediates whose cells are read once by
  the reducing consumer — within `virtualize_max_visits` (default 1), so
  they virtualize and inline into the reduction kernel. For these, ternary
  einsum buys **ergonomics only** (one spec vs 2-4 chained ops; axis
  realignment via `einsum1` permutes, themselves virtual).
- **Chain contractions are *better* without it.** For the falsifier example
  `"ij;jk;km=>im"`, the fused kernel iterates the full product space —
  O(I·J·K·M) — while the binary chain is O(I·J·K + I·K·M). For N=1000 that
  is ~500x more FLOPs. The headline use case is an algorithmic trap, not a
  win; the binary-chain "workaround" is the recommended implementation.
- **The genuine losses are two**:
  1. *3-operand contractions whose binary split forces a sizable
     intermediate at no FLOP advantage* (e.g. bilinear forms
     `"bi; i->jo; bj => b->o"`, where the chain's b×j×o intermediate costs
     the same total FLOPs as the fused b·i·j·o loop nest): whether the
     chain's intermediate materializes is config-dependent. Under the
     recommended config (`inline_complex_computations=false`,
     `ocannl_config.example:138`, kept off pending CSE per FIXME #351),
     accumulation self-accesses are counted, so *any* reduction-bearing
     intermediate exceeds `virtualize_max_visits` and materializes — fused
     ternary then saves a buffer and a kernel at equal FLOPs. Under the
     code default (`inline_complex_computations=true`), a read-once
     reduction intermediate can inline and the chain fuses anyway. Real
     but niche for a NN workload either way (`nn_blocks.ml`'s masked
     attention, the in-repo `where` user, is pointwise-then-softmax —
     ternary einsum would not even apply).
  2. *Single-assignment `%cd` forms*: `v =:@^ where v1 v2 v3 ~logic:"spec"`
     (masked max-reduce with the predicate evaluated pre-reduction, in one
     assignment with one guaranteed kernel) currently dies at the dispatch
     error. The composite workaround — pointwise `where` into an explicit
     intermediate, then `@^^`/`++` reduce — *also* fuses in the common case,
     because the read-once pointwise intermediate virtualizes; but it
     requires an extra named node, relies on the virtualizer rather than
     guaranteeing fusion, and needs `einsum1` pre-permutes whenever the
     predicate and branches disagree on axis order. So the loss here is
     ergonomic-plus-guarantee, not raw expressivity.

### Where-einsum semantics check (decision follow-through)

`where "p ; a ; b => out"` is well-defined and implementable with exactly the
`Mul3`-style projections: one product space over the union of all labels;
at each point, the predicate element (projection of spec 1) selects the
branch-2 or branch-3 element; the selected value is accumulated into `out`
(reduction over labels absent from `out`). Selection happens per-element
**before** reduction by construction — `Ternop` is evaluated inside the
iteration nest, accumulation outside it. Gradient: `g2 =+ where v1 g 0;
g3 =+ where v1 0 g` under the same forward projections — per iteration
point, the incoming gradient cell routes to whichever branch was selected;
no gradient to the predicate (correct: it is piecewise constant). One
documented caveat: a branch that broadcasts along a reduced axis has its
value accumulated once per selected iteration (count-weighted) — the honest
semantics of reduce-after-select. With non-`+` accumulators in raw `%cd`,
gradients are the user's responsibility as usual.

### Recommendation: IMPLEMENT, via the shared n-ary helper

Load-bearing reasons:

1. **The marginal cost collapsed.** Post-`Block`, the once-dominant
   constraint-generation work reduces to a helper extraction that is
   independently justified (discharges the `shape.ml:1087` TODO, removes
   ~200 net lines, fixes the `update_var_ref_list` omissions); projections
   and PPX `RHS3` plumbing need ~zero and ~10 lines respectively; `Mul3` is
   ~30 trivial lines with no per-backend builtins.
2. **`Where`-einsum has clean semantics and a real, if modest, payoff.**
   Select-before-reduce in a single spec (and in single `%cd` assignments
   with any accumulator, e.g. `=:@^`) replaces explicit-intermediate
   composites that fuse only at the virtualizer's discretion; its gradient
   reuses forward projections with no spec manipulation. The decided-in op
   is also the one whose semantics survive reduction unambiguously
   (unlike FMA's re-added `+ c`).
3. **It closes a v1.0-milestoned issue at its cheapest-ever price**, and the
   alternative (CLOSE) would still leave the helper refactor worth doing,
   at which point the ternary branch is ~15 lines of marginal code.

Required honesty in the deliverable: `einsum3`'s documentation (and the
proposal's falsifier test) must warn that chain-structured contractions
belong in binary chains — ternary einsum is for shared-reduction patterns
and fused selection, not a replacement for contraction sequencing.

**Remaining sub-decisions for Łukasz**:

- **Include `Mul3`/`einsum3`?** Recommended yes (~30 lines + entry point;
  it is the issue's title feature and the gradient story is clean), but a
  `Where`-only first cut is coherent if product contraction feels
  speculative — the shape/PPX machinery is op-agnostic.
- **`fma` with einsum spec**: targeted compile-time error in `%cd`
  (recommended, matching "FMA out of scope") vs. allow-and-document the
  count-weighted `+ c` semantics.
- **Surface API naming**: `?logic` parameter on `where` vs. a separate
  `where_einsum spec`; `einsum3` as standalone vs. a three-operand infix.

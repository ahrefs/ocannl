# Tensor stacking operation (expansion+concat einsum specs) and syntax

## Goal

OCANNL has no first-class way to **stack** N tensors along a *new* axis (result
rank = operand rank + 1). Users can currently only concatenate along an
*existing* axis. This task adds a tensor **stacking** operation built on the
existing einsum machinery — for N operands, generate on the fly (1) an
**expansion** einsum spec that unsqueezes each operand with a fresh leading
dim-1 axis, and (2) a **concatenation** einsum spec that concatenates the
expanded intermediates along that new axis — and exposes block-literal **syntax**
that desugars to the operation.

This is the user-selected "in-between easy approach" (2026-06-08): it sits
between the abandoned block-tensor-literal task (`task-fe1c593d`) and the
abandoned direct-`Concat`-axis N-ary framing (`task-e548a1b1`), and supersedes
both. Because it routes through `einsum1` + `concat`, it is aligned with — not a
competitor to — `gh-ocannl-49` (the einsum-extension concat operator `^`), and
deliberately avoids the harder Variant-A direct-`Concat`-axis projection/lowering.

Relates to: `gh-ocannl-49`, `gh-ocannl-421`.

## Acceptance Criteria

1. **A named stacking operation exists.** A reusable, callable operation (e.g.
   `stack` / `stack_along ~axis_kind`) lives in `tensor/operation.ml` and is
   exposed through the DSL module surface (consistent with how `einsum1` /
   `concat` are exposed in `NTDSL` / `TDSL.O`). It performs the on-the-fly
   expansion-spec + concat-spec generation internally (unsqueeze each operand via
   `einsum1`, then `concat` along the fresh axis). The block-literal PPX desugars
   to **this** operation rather than emitting inline `einsum1` + `concat`.

2. **Result rank grows by one** (true stacking — a new axis), as opposed to
   block-concat along an existing axis. For operands with `output_dims:[3]`, a
   2-element stack produces a result with axes `0:2, 1:3` (the existing
   `test_block_tensor` fixture asserts exactly this).

3. **Surface dispatch by container kind is preserved** (matches the documented
   `docs/syntax_extensions.md` design):
   - list `[a; b]` → new **output** axis,
   - array `[|a; b|]` → new **batch** axis,
   - top-level 2+ tuple `(a, b)` → new **input** axis,
   - single-element `[a]` → degenerate unsqueeze.

4. **Numeric-literal routing is unchanged.** The first-leaf heuristic
   (`is_ndarray_constant_expr`) still routes numeric block literals to the
   existing `TDSL.ndarray` path; the `(2.0 : basis)` type-annotation ndarray
   form that master added independently continues to work.

5. **The work lands on `lukstafi/ocannl-staging`** (the fork stays a fork; PR
   opened with `--repo lukstafi/ocannl-staging`). Whether the result is an
   updated PR #21 or a fresh PR is cosmetic.

6. **Tests pass.** The `test/operations/test_block_tensor` fixture (13 cases,
   both `.ml` and re-baselined `.expected`) passes, and `rope_test.expected` is
   re-baselined where the new codegen changes its output. No regressions in the
   einsum / operations test suites.

7. The redundant `Concat = Concat` `unify_dim` change from PR #21 is **not**
   re-introduced — that unification already landed on master in a more evolved
   form (around `tensor/row.ml` `unify_dim`), so this part of the seed branch is
   dropped on rebase.

## Context

How things work now:

- **`tensor/operation.ml`** — `einsum1 ?capture_dims spec` (a `Tensor.unop` with
  `Shape.Permute`) and `concat_sum` / `concat` (a `Tensor.blockop` over an array
  of rhs tensors) already exist and are exposed through the DSL modules
  (`NTDSL`, `TDSL.O`, `Initial_NTDSL`). These are the primitives the stacking
  operation composes; no new shape-solver or runtime primitive is needed.

- **`tensor/ppx_op.ml`** — the `%op` PPX. On the seed branch
  (`origin/restage/ludics/task-fe1c593d-s4/root` @ `8ec198560fbc`) the function
  `translate_block_tensor ~loc ~loop ~label ~opt_label axis_kind elems` builds:
  - a per-axis-kind **unsqueeze spec** (RHS `0` mints a fresh size-1 axis), e.g.
    output kind `"...|...->... => ...|...->0,..."`, applied as
    `einsum1 <unsqueeze_spec> e` to each operand;
  - a **concat spec** with fresh labels `bt0^bt1^…`, e.g. output kind
    `"...|...-> bt0,...; ...|...-> bt1,... => ...|...-> bt0^bt1,..."`, applied as
    `concat <concat_spec> [| unsqueezed… |]`.
  The dispatch arms (`is_ndarray_constant_expr` first-leaf check, then
  `translate_block_tensor … `Output` / `Batch` / `Input`) live further down the
  same file. On master, the same dispatch region was independently rewritten to
  add the `(2.0 : basis)` type-annotation ndarray form (`Ptyp_constr` →
  `TDSL.number ~axis_basis:…`).

- **`docs/syntax_extensions.md`** documents the list/array/tuple → output/batch/
  input axis-kind mapping. `docs/block-tensor-literal-proposal.md` is prior
  background — leave it as-is (not retired without explicit instruction).

- **`test/operations/test_block_tensor.{ml,expected}`** — the seed branch adds a
  13-case fixture proving rank growth. Reuse it as the regression anchor.

- **PR #21** (`lukstafi/ocannl-staging`, CLOSED, "block tensor literal syntax",
  restage of `ahrefs/ocannl#450`) already implements the unsqueeze+concat
  algorithm — on inspection it *is* tensor stacking, because it performs the
  unsqueeze. The genuine addition this task makes beyond PR #21 is factoring the
  inline codegen into the named, callable operation (AC 1).

## Approach

*Suggested approach — agents may deviate if they find a better path.*

This is largely a **rebase-and-factor** job, not a rewrite:

1. **Rebase the seed branch.** Take `origin/restage/ludics/task-fe1c593d-s4/root`
   (@ `8ec198560fbc`) and reconcile it against current `master`. A 3-way merge
   reports exactly **one** content conflict: `tensor/ppx_op.ml`, in the dispatch
   arm that master rewrote to add the `(2.0 : basis)` ndarray form. Reconcile by
   hand so both features coexist: keep master's basis-annotation ndarray form
   **and** the block/stack dispatch (`is_ndarray_constant_expr` first-leaf check
   routing numeric literals to `ndarray`, non-numeric to `translate_block_tensor`
   with the `Output` / `Batch` / `Input` axis kind). Do **not** re-apply the seed
   branch's `tensor/row.ml` `Concat = Concat` `unify_dim` hunk — it already landed
   on master. (The seed branch is old and its raw diff against master is noisy
   with unrelated deletions; rely on the merge-base 3-way merge, which auto-merges
   everything except the one `ppx_op.ml` arm.)

2. **Factor out the named operation** (the user's "using the new operation", AC
   1). Extract the on-the-fly expansion-spec + concat-spec generation that PR #21
   inlines in the PPX into a reusable operation in `tensor/operation.ml` (working
   name `stack` / `stack_along ~axis_kind`, final name a coder choice), exposed
   through the DSL modules alongside `einsum1` / `concat`. Have
   `translate_block_tensor` desugar to that operation rather than emitting
   `einsum1` + `concat` inline. The spec-generation logic and the `einsum1`/
   `concat` primitives already exist, so this is mechanical relocation plus a
   public signature.

3. **Re-baseline expectations** against the rebased build:
   `test/operations/test_block_tensor.expected` and `rope_test.expected` (and any
   other fixtures the new codegen shifts). Run the einsum + operations suites.

## Scope

In scope:
- Reconciled `tensor/ppx_op.ml` (basis-annotation form + block/stack dispatch).
- Named stacking operation in `tensor/operation.ml`, exposed in the DSL modules.
- `test/operations/test_block_tensor.{ml,expected}` and re-baselined fixtures.
- PR against `lukstafi/ocannl-staging` (pass `--repo` explicitly).

Out of scope / non-goals:
- The Variant-A direct-`Concat`-axis projection/lowering (`task-e548a1b1`,
  abandoned).
- Re-introducing the `Concat = Concat` `unify_dim` change (already on master).
- Retiring `docs/block-tensor-literal-proposal.md`.
- The `gh-ocannl-49` `^` concat-operator surface (aligned but separate work).

Open minor choices (deferrable to the coder; none blocking):
- The operation's public name and whether it lives in `TDSL.O`.
- Whether the result lands as an updated PR #21 or a fresh PR (cosmetic).

Supersedes `task-fe1c593d` and `task-e548a1b1` (both abandoned).

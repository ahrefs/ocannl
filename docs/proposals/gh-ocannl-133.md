# Virtualize Repeated-Symbol and Affine-Indexed Producers

Tracked by: https://github.com/ahrefs/ocannl/issues/133

## Status

Issue #133 is still open. The feature has not landed.

The implementation still rejects the two producer index patterns that motivated the issue:

- `Non_virtual 5`: repeated dynamic symbols in the producer's index vector, such as a
  diagonal setter with indices `[| Iterator i; Iterator i |]`.
- `Non_virtual 51`: more than one non-static symbol inside a single `Affine` index
  position, such as `stride * oh + kh`.

`make_subst` in `inline_computation` still assumes the producer index vector yields one
substitution binding per position. Removing the validation checks without changing
substitution construction would be unsafe: duplicate symbols can crash environment
construction, and multi-symbol affine setters can silently lose fold contributions unless
the index map is injective.

## Split Plan

This proposal is now an umbrella with two child proposals:

- [Stage A: Virtualize Repeated-Symbol Producers](gh-ocannl-133-stage-a.md) handles
  diagonal and partially diagonal producers by allowing repeated symbols and adding equality
  guards at inlined read sites.
- [Stage B: Virtualize Injective Affine Producers](gh-ocannl-133-stage-b.md) first improves
  affine injectivity analysis, then uses that result to safely inline injective affine
  producers.

Recommended landing order:

```text
#134 -> Stage A -> Stage B
```

#134 is not required for isolated Stage A target tests. It is recommended first to keep
concerns separated: #134 fixes shared-loop ownership and cleanup bookkeeping under the
existing index rules, while #133 then changes which index patterns are admissible for
virtualization.

## Scope Boundary

Stage A is the immediate closure candidate for the diagonal half of #133. It does not accept
multi-symbol affine producer positions.

Stage B covers affine producer positions whose dropped loops are proven injective. It keeps
non-injective loop-dropping substitution out of scope because that algorithm loses fold
contributions. It also defers quotient/remainder index IR needed for loop-free inversion of
some affine maps.

`Concat` indices remain out of scope in both stages. They must still be eliminated before
virtualization; preserve the current `Non_virtual 52` rejection.

## Shared Context

Relevant files:

- `arrayjit/lib/low_level.ml`
  - `check_and_store_virtual` / `check_idcs`: validates whether a producer can be stored as
    virtual.
  - `inline_computation` / `make_subst`: builds the symbol substitution environment used
    during inlining.
  - `subst`: already expands nested affine substitutions; it is not the main blocker.
- `arrayjit/lib/indexing.ml`
  - `axis_index`: `Fixed_idx`, `Iterator`, `Affine`, `Sub_axis`, `Concat`.
  - `is_injective`: currently treats every multi-symbol affine LHS position as
    non-injective.
- `arrayjit/lib/assignments.ml`
  - lowering emits neutral initialization plus accumulating setters when projections are
    not injective.
- `docs/lowering_and_inlining.md`
  - documents the current affine limitation.

## Follow-Up After Both Stages

After #134, Stage A, and Stage B land, add a joint regression where a diagonal tensor and an
independent element-wise tensor are produced in the same loop and both are consumed
downstream.

# Proposal: Rename "dimensions label" to "basis", make them more usable

**Issue:** [#298](https://github.com/ahrefs/ocannl/issues/298)
**Milestone:** v0.6.4
**Related:** [#255](https://github.com/ahrefs/ocannl/issues/255) (audit label checking), [axis-labels proposal](axis-labels.md) (separate axis naming feature)

## Goal

Rename the `label` field on `solved_dim` (dimension semantic annotations like "rgb", "features") to `basis` throughout the codebase, and rename the corresponding user-facing API parameters (`~axis_label` to `~axis_basis`). This clarifies the distinction between dimension basis (semantic unit of an axis) and other uses of "label" in OCANNL (tnode debug names, einsum pseudo-labels). Additionally, introduce a `dimensions` record type to make basis+size pairs a first-class concept.

## Acceptance Criteria

- [ ] Rename `solved_dim.label` field to `solved_dim.basis` in the type definition and all references in `tensor/row.ml` and `tensor/row.mli`
- [ ] Rename `~axis_label` parameter to `~axis_basis` in `Tensor.number`, `Tensor.bits`, and `Operation.range` (and their DSL module wrappers `TDSL`, `NTDSL`)
- [ ] Rename `~label` parameter on `Row.get_dim` to `~basis`
- [ ] Rename `dims_label_assoc` to `dims_basis_assoc` in `Row` module
- [ ] Rename `row_to_labels` to `row_to_bases` in `Row` module (and `Shape.to_labels` to `Shape.to_bases`)
- [ ] Rename `Only_labels` print style variant to `Only_bases`
- [ ] Update PPX code (`ppx_op.ml`, `ppx_cd.ml`, `ppx_shared.ml`) to emit the renamed API calls
- [ ] Do NOT rename: (a) `Tn.t.label : string list` (tnode debug names -- unrelated), (b) `einsum_types.Label` constructor (einsum variable identifiers -- unrelated), (c) `parsed_axis_labels` / `axis_labels_of_spec` (einsum parsing infrastructure -- these are pseudo-labels local to einsum notation, not dimension basis)
- [ ] Update documentation: `docs/shape_inference.md`, `docs/syntax_extensions.md`, `ROADMAP.md` references to "dimension units" / "dimension labels"
- [ ] Update the axis-labels proposal (`docs/proposals/axis-labels.md`) to use "basis" terminology
- [ ] All existing tests pass with no regressions
- [ ] Consider introducing a `dimensions` record type `{ size : int; basis : string option }` as a convenience alias or replacement for the current `solved_dim` fields `d` and `basis` -- evaluate whether this simplifies user-facing APIs

## Context

### Current state

The `label` field lives on `solved_dim` in `tensor/row.ml` (line 63):

```ocaml
type solved_dim = { d : int; label : string option; proj_id : proj_id option }
```

It provides semantic annotations on dimensions (e.g., a dimension of size 3 with label "rgb" means "this axis represents RGB channels"). Two dimensions that must agree in size must also agree on their label. Labels propagate through affine/concat operations.

### Three distinct uses of "label" in OCANNL

1. **Dimension labels (basis)** on `solved_dim` -- THIS task's target
2. **Tnode debug labels** (`Tn.t.label : string list`) -- NOT renamed, these are debug names
3. **Einsum pseudo-labels** (`einsum_types.Label of string`) -- NOT renamed, these are notation-local identifiers

### Scope estimate

The rename touches approximately:
- **Core types**: `tensor/row.ml` (~15 occurrences), `tensor/row.mli` (~5)
- **Shape infrastructure**: `tensor/shape.ml` (~10), `tensor/shape.mli` (~3)
- **User API**: `tensor/tensor.ml` (~6 `axis_label` refs), `tensor/tensor.mli` (~2)
- **PPX**: `tensor/ppx_op.ml` (~3), `tensor/ppx_cd.ml` (~3), `tensor/ppx_shared.ml` (~8)
- **Operations**: `tensor/operation.ml` (~3)
- **Tests**: ~9 files with ~124 total occurrences of related patterns (many are tnode labels or einsum labels that should NOT be renamed)
- **Documentation**: `ROADMAP.md`, `docs/shape_inference.md`, `docs/syntax_extensions.md`, `docs/proposals/axis-labels.md`

### The `dimensions` record question

The issue suggests introducing `type dimensions = { d : int; basis : string option }` combining size and basis. Currently `solved_dim` already has this plus `proj_id`. Options:

- **Minimal**: just rename `label` to `basis` in `solved_dim`, no new type
- **New convenience type**: expose `type dimensions = { size : int; basis : string option }` as a user-facing type for shape specification APIs (e.g., `Shape.make ~output_dims:[{ size=3; basis=Some "rgb" }]`), while `solved_dim` retains its internal `proj_id` field

The minimal approach is recommended for this task. A new `dimensions` record can be added as a follow-up if the API benefit is clear after the rename settles.

### Obligatory basis question

The issue asks "Maybe make units obligatory!" -- this should NOT be done in this task. Making basis obligatory would be a breaking API change affecting all existing code that creates dimensions without labels. It can be considered as a separate enhancement after the audit in #255 strengthens label/basis checking.

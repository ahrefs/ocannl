# Axis Labels (as opposed to dimension units)

## Motivation

OCANNL's shape system identifies axes positionally and provides optional **dimension labels** (aka "units" or "basis") as semantic annotations on individual dimensions. These labels constrain matching -- axes with different labels cannot be unified -- but do not name axes themselves. This means there is no way to refer to a specific axis of a tensor by name after construction.

Axis labels would allow naming axes explicitly (e.g., `batch`, `seq_len`, `hidden`), similar to PyTorch's named tensors or xarray's dimension names. This could improve:
- **Readability**: operations like transpose and reshape can reference axes by name instead of position.
- **Safety**: catch permutation errors at the type/shape level (e.g., accidentally swapping `seq_len` and `hidden`).
- **Interoperability**: named axes are a common convention in the broader ML ecosystem.

The README (line 92) and ROADMAP (lines 181, 224) list this as a post-1.0 consideration. GitHub issue [#298](https://github.com/ahrefs/ocannl/issues/298) tracks the related but distinct task of renaming dimension labels to "basis" and making them more usable.

## Current State

### Dimension labels (the existing mechanism)

Dimensions carry an optional `label : string option` in `solved_dim` (`tensor/row.ml`, line 63):

```ocaml
type solved_dim = { d : int; label : string option; proj_id : proj_id option }
```

These labels act as semantic units: two dimensions that must agree in size must also agree on label (if both are labeled). A label like `"rgb"` on a dimension of size 3 means "this axis represents RGB channels." Labels need not be unique within a row and are inferred during shape inference. They are *not* an axis selection mechanism (README line 16).

### Einsum pseudo-labels

The einsum notation (e.g., `"...s | h d; ...t | h d => ...s | t -> h"`) uses single- or multi-character identifiers to line up axes across operands. These are local to the notation -- they identify which axes correspond to each other within a single einsum spec, but do not persist on the resulting tensor. Parsed as `Label of string` in `einsum_types.ml` (line 18).

### Shape spec labels vs dimension labels

In `shape_spec_to_dims_bio` (`tensor/shape.ml`, around line 2343), the parser distinguishes two uses of labels in shape specifications:
- `name=42` syntax: sets a dimension to a fixed size (the name is an axis label, currently discarded -- comment says "This is not a dimension label i.e. unit!").
- Plain `name` syntax: creates or references a dimension variable (for inference).

This is the only place where "axis labels" (as distinct from dimension labels) are acknowledged in the current code, but they are not stored or propagated.

### Key files

- `tensor/row.ml` -- `dim`, `solved_dim`, `dim_var`, constraint solving, label matching (lines 1599, 2092)
- `tensor/shape.ml` -- shape inference, `shape_spec_to_dims_bio`, einsum integration
- `tensor/einsum_types.ml` -- `axis_spec`, `parsed_axis_labels`, `AxisKey`
- `tensor/shape.mli` -- public API, notes on dimension labels vs axis labels (line 196)
- `docs/shape_inference.md` -- explanation of the label system (line 49)
- `docs/workshop-paper-proposal.md` -- design rationale for positional axes (line 32)

## Proposed Change

Introduce an **optional axis label** mechanism that is distinct from the existing dimension labels (units/basis). An axis label names the axis itself (its role in the tensor), while a dimension label names the semantic unit of the axis's size.

### Design questions to resolve

1. **Where do axis labels live?** The natural place is on the `dim` type or as a parallel field in `Row.t`. Unlike dimension labels which annotate a solved dimension, axis labels should be present even on unsolved `Var` dimensions. A possible extension:
   - Add an `axis_label : string option` field alongside each `dim` entry in a row, or
   - Extend the row type to carry a parallel list of axis labels.

2. **Uniqueness within a row**: Axis labels should be unique within a single row (batch, input, or output). Two axes in the same row cannot share an axis label. This is unlike dimension labels, which need not be unique.

3. **Inference and propagation**: When two axes are unified (via shape inference), their axis labels should be unified too -- if both are labeled, they must match; if only one is labeled, the label propagates. This mirrors the existing dimension-label unification logic but applies to axis identity rather than dimension semantics.

4. **Interaction with einsum notation**: Einsum pseudo-labels already name axes locally. The question is whether einsum labels should automatically become axis labels on the result tensor. This would be a natural extension but changes the semantics of einsum specs (currently ephemeral labels would become persistent).

5. **Interaction with row variables and broadcasting**: When a row is extended by broadcasting, the new axes have no axis labels. When a row variable is substituted, axis labels on the substituted axes should be preserved.

6. **API surface**: Axis labels could be:
   - Set via shape specs (e.g., `"batch seq_len | hidden -> hidden"` where these become axis labels), or
   - Set programmatically on tensor construction, or
   - Both.

### What axis labels enable

- **Named transpose**: `transpose tensor ~from:"seq_len" ~to_:"hidden"` instead of positional indices.
- **Named indexing**: select or slice axes by name.
- **Better error messages**: "axis 'seq_len' (size 128) does not match axis 'hidden' (size 64)" instead of positional references.
- **Documentation**: tensor shapes are self-describing.

### Design tension

The workshop paper proposal (`docs/workshop-paper-proposal.md`, line 51) articulates a deliberate design rationale for positional axes:
- Positional representation builds on mathematical tradition.
- Row variable inference is ambiguity-free with positional axes but problematic with named axes.
- Optional dimension units provide semantic safety without requiring unique axis names.

Axis labels should therefore remain **strictly optional** and not change the semantics of shape inference for unlabeled axes. They are an additional layer of checking, not a replacement for positional identification.

## Scope

**In scope:**
- Design of the axis label representation and its integration with existing types.
- Rules for axis label inference and unification.
- Interaction with einsum notation, broadcasting, and row variables.
- Backwards compatibility: all existing code without axis labels continues to work unchanged.

**Out of scope:**
- Renaming dimension labels to "basis" (tracked separately in [#298](https://github.com/ahrefs/ocannl/issues/298)).
- Shape schemes / polymorphism for tensor functions (separate post-1.0 item).
- Dynamic axis selection or advanced named-tensor algebra (e.g., xarray-style alignment).

**Dependencies:**
- Issue [#298](https://github.com/ahrefs/ocannl/issues/298) (rename labels to basis) should ideally land first to reduce terminology confusion between "dimension label" and "axis label."
- No hard blockers from other tasks.

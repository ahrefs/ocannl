# Axis Labels Design (as opposed to dimension units)

## Motivation

OCANNL's shape system identifies axes positionally and provides optional **dimension labels** (aka "units" or "basis") as semantic annotations on individual dimensions. These labels constrain matching -- axes with different labels cannot be unified -- but do not name axes themselves. This means there is no way to refer to a specific axis of a tensor by name after construction.

Axis labels would allow naming axes explicitly (e.g., `batch`, `seq_len`, `hidden`), similar to PyTorch's named tensors or xarray's dimension names. This could improve:
- **Readability**: operations like transpose and reshape can reference axes by name instead of position.
- **Safety**: catch permutation errors at the type/shape level (e.g., accidentally swapping `seq_len` and `hidden`).
- **Interoperability**: named axes are a common convention in the broader ML ecosystem.

The README (line 92) and ROADMAP (lines 181, 224) list this as a post-1.0 consideration. GitHub issue [#298](https://github.com/ahrefs/ocannl/issues/298) tracks the related but distinct task of renaming dimension labels to "basis" and making them more usable.

## Current State

### Three kinds of "labels" in OCANNL today

1. **Dimension labels (units/basis)** on `solved_dim` (`tensor/row.ml`, line 63):

   ```ocaml
   type solved_dim = { d : int; label : string option; proj_id : proj_id option }
   ```

   These act as semantic units: two dimensions that must agree in size must also agree on label (if both are labeled). A label like `"rgb"` on a dimension of size 3 means "this axis represents RGB channels." Labels need not be unique within a row and are inferred during shape inference. They are *not* an axis selection mechanism.

2. **Einsum pseudo-labels** in the notation `"...s | h d; ...t | h d => ...s | t -> h"`. These are single- or multi-character identifiers local to the notation, parsed as `Label of string` in `einsum_types.ml` (line 18). They identify which axes correspond to each other within a single einsum spec but do **not** persist on the resulting tensor.

3. **Shape-spec position labels** in `shape_spec_to_dims_bio` (`tensor/shape.ml`, around line 2343). The `name=42` syntax sets a dimension to a fixed size. The name part is an axis label but is currently **discarded** -- the comment says "This is not a dimension label i.e. unit!" This is the only place where axis labels (distinct from dimension labels) are acknowledged in code, but they are not stored or propagated.

### Existing misleading API names

The current codebase has parameters that use "axis" terminology but actually set **dimension labels**, not axis labels:

- `Tensor.number ?axis_label` and `Tensor.bits ?axis_label`: these pass the string to `output_axes`, which calls `Shape.make ~output_axes:[(label, 1)]`, which passes to `get_dim ~d ~label ()` -- i.e., it sets a **dimension label** (unit), not an axis label in the sense proposed here.
- `Shape.make ?batch_axes ?input_axes ?output_axes`: documented as "these are dimension labels and not axis labels" (shape.mli line 196). The `(string * int) list` pairs set dimension labels.

This terminological collision must be addressed by the implementation. See the API compatibility section below.

### Key files

- `tensor/row.ml` -- `dim`, `solved_dim`, `dim_var`, constraint solving, label matching (lines 1599, 2092)
- `tensor/shape.ml` -- shape inference, `shape_spec_to_dims_bio`, einsum integration
- `tensor/einsum_types.ml` -- `axis_spec`, `parsed_axis_labels`, `AxisKey`
- `tensor/shape.mli` -- public API, notes on dimension labels vs axis labels (line 196)
- `docs/shape_inference.md` -- explanation of the label system (line 49)
- `docs/workshop-paper-proposal.md` -- design rationale for positional axes (line 32)

## Design Decisions

### Decision 1: Axis labels live on `Row.t` as parallel metadata

Axis labels belong at the **row level**, not on the `dim` type. Rationale:

- A `dim_var` can be shared across multiple axes in an einsum spec (e.g., `i` in `"ij;jk=>ik"` appears in different rows of different operands). The axis label names the *position in a specific tensor*, not the dimension value.
- A `solved_dim` already carries `label` (dimension label/unit) and `proj_id`. Adding axis labels to `dim` conflates two distinct concerns: what the dimension measures (unit) vs. what role the axis plays (identity).
- `Row.t` naturally represents "the axes of a particular kind in a particular tensor" -- the right granularity for axis naming.

**Chosen representation**: add an `axis_labels : string option list` field to `Row.t`:

```ocaml
type t = { dims : dim list; bcast : bcast; prov : provenance; axis_labels : string option list }
```

And extend `bcast` to carry labels on `beg_dims`:

```ocaml
type bcast =
  | Row_var of { v : row_var; beg_dims : dim list; beg_axis_labels : string option list }
  | Broadcastable
```

**Semantics of `axis_labels`**:
- `axis_labels = []` means "unset / don't check" -- fully backward compatible. This is the default for all existing code.
- `axis_labels = [Some "a"; None; Some "c"]` means "3 axes; first is labeled 'a', second is unlabeled, third is labeled 'c'."
- When `axis_labels` is non-empty, it must have the same length as `dims`. This invariant is maintained by the infrastructure and validated when labels are first attached.
- We never auto-populate `axis_labels` with `[None; None; ...]` when the axis count becomes known from dims. Labels remain `[]` until someone explicitly provides them.

### Decision 2: Axis labels are unique within a single row

Two axes in the same row (batch, input, or output) cannot share the same axis label. This is enforced at row construction time (when labels are first set) and after inference updates merge labels.

The same axis label name *may* appear in different rows of the same shape (e.g., an axis `"time"` in both batch and input) -- this is unusual but not prohibited, since rows are independent namespaces. Importantly, using the same axis label name in two different rows does **not** imply any shape-inference relationship between those rows. Axis labels never drive unification or create constraints -- they are checked only where positional unification already occurs. Cross-row uniqueness may be added later if experience shows it's needed.

### Decision 3: Propagation rules for unification

When two rows are unified (via `unify_row`), axis labels are unified position-by-position using the following rules:

| Row 1 label | Row 2 label | Result | Action |
|---|---|---|---|
| `[]` (unset) | anything | other's labels | Inherit |
| anything | `[]` (unset) | first's labels | Inherit |
| `Some a` | `Some a` | `Some a` | Keep (match) |
| `Some a` | `Some b` where `a <> b` | **error** | `Shape_error` with both labels |
| `Some a` | `None` | `Some a` | Propagate |
| `None` | `Some b` | `Some b` | Propagate |
| `None` | `None` | `None` | Keep (both unlabeled) |

When lists differ in length (due to broadcasting inserting new axes), the shorter list is left-padded with `None` values before comparison. This matches broadcasting semantics where new axes appear on the left.

### Decision 4: Einsum pseudo-labels remain ephemeral by default

Einsum pseudo-labels do **not** automatically become axis labels on the result tensor. They remain local alignment identifiers within a single spec. Rationale:

- Automatically persisting them would change semantics for all existing einsum operations.
- Many einsum variable names are terse/positional (`i`, `j`, `k`) and not meaningful as persistent axis names.
- Users would have no way to suppress unwanted axis labels from einsum specs.

However, the `name=42` syntax in shape specs (which is currently discarded) will now store the name as an axis label. This is the natural place for explicit axis labeling in spec strings:

```
"seq_len=128 | hidden=64 -> hidden=64"
```

In the above, `seq_len` and `hidden` are axis labels. The `=N` part sets the dimension size. Each `hidden=64` independently creates a fixed dimension of size 64 with the axis label `"hidden"` stored on its respective row (input and output). The axis labels are stored independently -- having the same axis label name in two different rows does **not** create any shape-inference constraint between them. If the user needs input and output dimensions to be linked, they should use the existing dimension-variable syntax (e.g., `"hidden | hidden -> hidden"` without `=N`, which shares a `dim_var` across rows and forces unification through the existing mechanism).

Future work may add a `~persist_labels:bool` parameter to einsum operations, but this is out of scope for the initial implementation.

### Decision 5: Broadcasting and row variables

- **Broadcasting inserts unlabeled axes.** When a row is extended by broadcasting (prepending axes from the left), the new axes get `None` axis labels. Broadcasting never invents axis labels.
- **Row variable substitution preserves labels.** When a row variable is resolved to concrete axes, axis labels on the existing `beg_dims` and `dims` portions of the row are preserved. The substituted axes from the row variable get `None` labels unless the substituting row itself has axis labels.
- **Permute/transpose moves labels with axes.** See the operation semantics section below.

### Decision 6: API naming and compatibility

The existing `?axis_label` parameters on `Tensor.number` and `Tensor.bits` and the `*_axes` parameters on `Shape.make` actually set **dimension labels** (units), not axis labels in the new sense. This terminology collision is addressed as follows:

**Phase 1 (this task)**: Introduce new, distinctly named parameters for true axis labels. Do not rename or change the existing parameters (to avoid breaking existing code):

- `Shape.make` gains `?batch_axis_labels:(string option list)`, `?input_axis_labels:(string option list)`, `?output_axis_labels:(string option list)` -- these set axis labels on the respective rows.
- `Shape.get_axis_labels : t -> [` Batch | ` Input | ` Output] -> string option list` -- retrieves axis labels for a row.
- `Shape.set_axis_labels : t -> [` Batch | ` Input | ` Output] -> string option list -> unit` -- sets axis labels on a mutable row.

**Phase 2 (coordinated with #298)**: When dimension labels are renamed to "basis", the existing `*_axes` and `?axis_label` parameters can be deprecated and replaced with clearer names. This is tracked separately and not a blocker.

**Documentation**: All new APIs clearly document that axis labels are distinct from dimension labels. The existing `*_axes` parameter docs are updated to note the distinction.

## Operation Semantics

### Pointwise operations (add, multiply, etc.)

Axis labels are unified pairwise between operands and the result, following the propagation table above. This applies to all `Broadcast`-type shape operations.

**Example**: `x + y` where `x` has output axis labels `[Some "a"; Some "b"]` and `y` has `[Some "a"; None]` produces output axis labels `[Some "a"; Some "b"]`.

### Transpose (`Transpose` shape type)

`Transpose` swaps input and output rows. Axis labels move with their rows:

- If input had labels `[Some "x"; Some "y"]` and output had `[Some "a"]`, after transpose the input has `[Some "a"]` and the output has `[Some "x"; Some "y"]`.

### Permute (`Permute` shape type, einsum unary `"spec => result"`)

Permute reorders axes according to an einsum spec. Axis labels are carried along with their axes during reordering. The mapping is determined by the einsum variable correspondence:

**Example**: Input axes labeled `[Some "h"; Some "w"]` with spec `"hw => wh"` produces `[Some "w"; Some "h"]`.

Implementation: when building the LHS row from the einsum spec, each axis position on the LHS inherits the axis label from the corresponding RHS position (matched through the shared einsum variable). If the RHS position has no axis label, the LHS position also has none.

Permute operations use equations (not inequalities), so they do NOT permit broadcasting. This means axis labels are carried 1:1 without needing to handle broadcasting padding.

### Binary einsum

In binary einsum (`"spec1; spec2 => result_spec"`), axis labels on the result follow the LHS of the spec. Specifically:

- **Surviving axes**: If an axis on RHS1 or RHS2 has an axis label and that axis survives into the result (appears in the LHS spec), the label propagates to the corresponding result position.
- **Contracted axes**: Axes that appear in the RHS but not the LHS are contracted (reduced). Their axis labels are consumed and do not appear in the result.
- **Conflict resolution**: If both RHS1 and RHS2 contribute an axis label for the same result position (through the same einsum variable), the labels must match. If they differ, it is a `Shape_error`.

### Batch slice (`Batch_slice`)

Removes the leftmost batch axis. The remaining batch axis labels shift: `[Some "a"; Some "b"; Some "c"]` becomes `[Some "b"; Some "c"]`.

### Block operations

Block operations (constructing tensors from sub-tensors) do not propagate axis labels from the sub-tensors. The result's axis labels come from the block's spec, if any. If the block spec uses `name=N` syntax, those names become axis labels.

### Concat (`a^b` in specs)

Concat combines multiple axes into one. The result is a single axis position. The axis label for the concatenated result is `None` by default -- the component labels from the `a^b` spec remain local alignment names. Users can explicitly name the concatenated axis via the result spec.

### Operations where axis labels are dropped

- **Any operation that changes rank in a way that is not a pure permutation**: if the shape logic cannot establish a 1:1 correspondence between input and output axes, axis labels are dropped (set to `[]`).
- **`Defined_by_cd_logic`**: shapes defined by `%cd` extension logic do not propagate axis labels, since the shape semantics are opaque.
- **`Uint4x32_to_prec`**: precision conversion changes element count; axis labels are dropped.

## Error Message Contract

### When axis labels appear in errors

Axis label information is included in `Shape_error` messages whenever it is available. Specifically:

1. **Axis label mismatch**: When two rows are unified and have conflicting axis labels at the same position, the error message names both labels:
   ```
   Shape_error: axis label mismatch at output position 2:
     'seq_len' (from shape 'query', size 128)
     vs 'hidden' (from shape 'key', size 64)
   ```

2. **Dimension mismatch with labels**: When a dimension size mismatch occurs between axes that have axis labels, the labels are included for context:
   ```
   Shape_error: dimension mismatch at output position 1:
     axis 'hidden' has size 64 (from shape 'query')
     vs axis 'hidden' has size 128 (from shape 'key')
   ```
   Note: axis labels agreeing but sizes disagreeing is possible and reported distinctly.

3. **Row mismatch with labels**: When row shapes cannot be unified and axis labels are present, the labels are listed alongside sizes:
   ```
   Shape_error: row mismatch (output):
     shape 'query': [seq_len:128, hidden:64]
     shape 'key':   [batch:32, hidden:64]
   ```

### Fallback for unlabeled axes

When axis labels are not present (`axis_labels = []` or position is `None`), the error message falls back to the existing behavior: positional references with dimension label (unit) if available, or just size.

### Distinction between axis-label errors and dimension-label errors

Axis-label mismatches and dimension-label mismatches are reported as **separate error conditions**:

- **Axis-label mismatch**: "axis label mismatch: 'seq_len' vs 'hidden'" -- the axis positions have conflicting names.
- **Dimension-label mismatch**: "solved dimensions for axis: different labels" (existing message) -- the dimension values have conflicting units.

Both can occur at once (axes with different names AND different units). Since axis labels are checked as part of `unify_row` (after positional alignment is determined but within the same call), the axis label mismatch is detected and reported first. If both errors would occur, only the axis label error is raised -- fixing it may also reveal or resolve the dimension label issue.

### Display format

For axis labels in printed shapes and error messages, axis labels use a colon separator to distinguish from dimension labels which use equals:

- Axis label: `seq_len:128` (axis name, then size)
- Dimension label: `rgb=3` (dimension unit, then size)
- Both present: `seq_len:rgb=3` (axis name, then dimension unit, then size)

The `row_to_labels` function (`tensor/row.ml`) is updated to prefer axis labels over dimension labels when both are available.

## Scope

**In scope:**
- Design of the axis label representation and its integration with existing types (`Row.t`, `bcast`).
- Concrete propagation rules for all operation types.
- Error message specification for axis label mismatches.
- API naming that avoids collision with existing misleading `*_axes` parameters.
- Backwards compatibility: all existing code without axis labels continues to work unchanged.

**Out of scope:**
- Renaming dimension labels to "basis" (tracked separately in [#298](https://github.com/ahrefs/ocannl/issues/298)).
- Shape schemes / polymorphism for tensor functions (separate post-1.0 item).
- Dynamic axis selection or advanced named-tensor algebra (e.g., xarray-style alignment).
- Making einsum pseudo-labels automatically persist (potential future `~persist_labels:bool`).
- PPX syntax extensions for axis labels (deferred until runtime representation is stable).

**Dependencies:**
- Issue [#298](https://github.com/ahrefs/ocannl/issues/298) (rename labels to basis) should ideally land first to reduce terminology confusion. Not a hard blocker.
- No hard blockers from other tasks.

## Design Tension

The workshop paper proposal (`docs/workshop-paper-proposal.md`, line 51) articulates a deliberate design rationale for positional axes:
- Positional representation builds on mathematical tradition.
- Row variable inference is ambiguity-free with positional axes but problematic with named axes.
- Optional dimension units provide semantic safety without requiring unique axis names.

Axis labels therefore remain **strictly optional** and do not change the semantics of shape inference for unlabeled axes. They are an additional layer of checking, not a replacement for positional identification.

### When axis labels are checked

Axis labels are checked **inline during row unification** (`unify_row`), not as a separate post-inference pass. When `unify_row` is called, it first performs existing dimension unification (positional matching, broadcasting, constraint generation), and then -- in the same call -- merges axis labels position-by-position using the propagation table from Decision 3. If a label conflict is detected, a `Shape_error` is raised immediately.

This inline approach means:
- Axis label errors are reported at the point of the operation that caused the conflict, with full provenance context available.
- Axis labels never influence which dimensions are unified or how row variables are resolved. The positional/dimensional logic runs first within `unify_row`; the label merge is a validation step on the resulting alignment.
- Adding axis labels to existing code can only add new error checks (label mismatches), never change inference outcomes or the order in which dimensions are solved.

## Implementation Roadmap

### Phase 1: Core types and propagation (this task)
1. Add `axis_labels` to `Row.t` and `beg_axis_labels` to `bcast.Row_var`.
2. Add helper functions: `validate_axis_labels`, `merge_axis_labels`, `axis_labels_to_string`.
3. Thread axis labels through `unify_row` and row substitution.
4. Update `Shape.make` with `*_axis_labels` parameters.
5. Update `shape_spec_to_dims_bio` to store `name=N` as axis labels.
6. Update `row_to_labels` to prefer axis labels.
7. Add error messages for axis label mismatches.
8. Tests for basic construction, propagation, mismatch errors, backward compatibility.

### Phase 2: Full operation coverage
1. Implement axis label propagation for `Permute`, `Transpose`, binary einsum, `Batch_slice`.
2. Add drop-labels logic for rank-changing operations.
3. Update `to_string_hum` and print styles.
4. Tests for each operation type.

### Phase 3: API and syntax (after runtime representation is stable)
1. Add axis label arguments to `Tensor.term`, `Tensor.number`, etc.
2. Coordinate with #298 to clean up `*_axes` / `?axis_label` naming.
3. PPX sugar for axis labels in `%op` and `%cd`.
4. Update `docs/syntax_extensions.md` with examples.

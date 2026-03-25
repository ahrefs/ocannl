# Shape Schemes for Tensor Functions

GitHub issue: [ahrefs/ocannl#404](https://github.com/ahrefs/ocannl/issues/404)

## Motivation

OCANNL's shape system is monomorphic: all row and dimension variables are existential. When a tensor function (e.g., `mlp`, `softmax`, `multi_head_attention`) is called multiple times, each call site independently re-runs shape inference from scratch. This leads to:

1. **Late error detection.** Shape errors inside a function body surface only when the function is applied, not at definition time. If a function is called N times, the same internal error can appear N times with confusing provenance.

2. **Redundant inference work.** Each application re-derives the same internal shape relationships. For functions called in loops or across layers, this multiplies constraint-solving effort.

3. **Poor error messages.** When a shape error occurs deep inside a function, the user sees internal shape variables and constraints rather than a clear message about which argument violated the function's shape requirements.

Shape schemes address this by tracing the function once at definition time with symbolic tensors, capturing the resulting shape constraints as a reusable template, and instantiating that template (with freshened variables) at each call site. This is the approach described in `docs/shape_inference.md` (line 9) as future work, analogous to Hindley-Milner type schemes.

## Current State

### Shape inference pipeline

Shape inference is a multi-stage pipeline in `tensor/shape.ml`:
- `propagate_shapes` runs Stage 1 constraint solving online as tensors are constructed.
- `finish_inference` runs Stages 2-7 on demand (triggered when dimensions or projections are needed, typically at jit time).
- `derive_projections` freshens projection IDs per-operation and re-solves projection constraints to prevent cross-operation contamination.

Global mutable state tracks the inference context:
- `state : Row.env ref` -- the substitution environment.
- `active_update_steps : update_step list ref` -- all operations registered for inference.
- `active_constraints : Row.constraint_ list ref` -- accumulated inequality constraints.

### The `%op` PPX and the `()` boundary

The `%op` syntax extension in `tensor/ppx_op.ml` handles functions with a `()` separator (lines 418-456). Parameters before `()` are configuration (not transformed); parameters after `()` are tensor arguments (transformed with TDSL operators). Examples from `lib/nn_blocks.ml`:

```ocaml
let%op mlp ~label ~hid_dims () = ...          (* config: label, hid_dims; tensors: implicit via returned closure *)
let%op softmax ~spec ?temperature () = ...     (* config: spec, temperature; tensors: via returned closure *)
let%op multi_head_attention ~label ~num_heads ~d_k ~d_v ... () ~train_step ?mask x = ...
```

The `()` boundary is the natural point for shape scheme capture: configuration parameters determine the function's shape constraints, and the `()` call is when those constraints become fixed.

### Tensor node namespaces (prerequisite)

Issue #372 (tensor node namespaces) was completed today. Namespaces allow creating tensors in isolated ID spaces, which is needed for tracing: the symbolic tensors created during scheme capture must not pollute the global tensor registry. The scheme capture would create tensors in a temporary abstract namespace, trace the function body, capture constraints, then discard the namespace.

### Key types

- `update_step` (shape.ml:122): records one shape-inference operation with its `shape`, `logic`, projections, and neutral element.
- `Row.constraint_` (row.mli:107-154): `Dim_eq`, `Row_eq`, `Dim_ineq`, `Row_ineq`, `Terminal_dim`, `Terminal_row` -- the constraint vocabulary.
- `delayed_var_ref` (shape.mli:96-119): captures shape variables at definition time, used by `set_dim`/`set_equal` for manual shape binding in functions like `multi_head_attention`.

### Freshening infrastructure

`fresh_proj_ids` (shape.ml, around line 1727) already freshens projection IDs within shapes to prevent cross-operation contamination. Shape schemes need a broader version that freshens all dim_vars and row_vars across a constraint set. Partial infrastructure exists in `Row.solve_inequalities` (which applies environment substitutions), but a standalone constraint-level substitution is likely needed.

## Proposed Change

Add optional shape scheme support for tensor functions defined with `%op` and a `()` separator.

### Shape scheme representation

A shape scheme captures the constraint template from a single traced execution:
- The shapes of tracing input tensors (symbolic arguments).
- The shapes of tracing output tensors (function results).
- The set of constraints generated during tracing (raw, pre-Stage-1).
- All dim_vars and row_vars that must be freshened at instantiation.
- Any `update_step` entries from the traced execution.

### Scheme capture (at definition time)

When a `%op` function with `()` is defined:
1. Save the current global shape inference state.
2. Create tracing tensors with fully symbolic shapes in an abstract namespace.
3. Execute the function body with the tracing tensors.
4. Collect all constraints and update steps generated during execution.
5. Store them as the function's shape scheme.
6. Restore the global state, discarding the tracing namespace.

Shape errors during tracing are caught at definition time -- this is one of the main benefits.

### Scheme instantiation (at each call site)

When the function is applied:
1. Freshen all dim_vars and row_vars in the scheme (new variable instances).
2. Apply the substitution to all stored constraints.
3. Add equalities between freshened tracing-input shapes and actual argument shapes.
4. Inject the freshened constraints into the global inference state via `propagate_shapes`.
5. Execute the function body normally (to construct the computation graph with real tensor nodes).

The body still executes at each call site for computation graph construction. The scheme only pre-validates and accelerates the shape inference portion.

### Acceptance criteria

- Tensor functions with `%op` and `()` can capture shape schemes at definition time.
- Shape schemes are stored as constraint templates with symbolic variables.
- At each call site, schemes are instantiated with freshened variables.
- Shape errors internal to the function are caught at definition time.
- Argument-mismatch errors are reported at the call site with clear provenance.
- Existing `nn_blocks.ml` functions (`mlp`, `softmax`, `multi_head_attention`, etc.) work with shape schemes.
- The feature is opt-in: functions without `()` or without explicit annotation work as before.
- No regression in existing tests.

### Edge cases

- **Configuration-dependent shapes.** In `mlp ~hid_dims:[32; 16] ()`, different configurations produce different schemes. This is natural since the scheme is captured after configuration parameters are bound (at the `()` call).
- **Variable-arity returns.** Functions may return tuples or closures returning tensors. The scheme must capture output shapes for all tensors produced during tracing.
- **Side effects on shape variables.** Functions like `multi_head_attention` call `Shape.set_dim` and `Shape.set_equal` during execution. These side effects must be captured during tracing and replayed during instantiation.
- **Higher-order tensor functions.** A function taking another tensor function as argument would need nested schemes. Initially, only first-order functions need support.
- **Multi-stage inference interaction.** The scheme should capture raw constraints plus Stage 1 equalities, so that equalities propagate immediately at instantiation while raw constraints ensure correctness through later stages.

## Scope

**In scope:**
- Shape scheme type definition and storage.
- Tracing infrastructure (capture and restore of inference state).
- Constraint freshening across full constraint sets.
- Scheme instantiation at call sites.
- PPX modifications to `%op` for scheme-aware code generation.
- Validation with existing `nn_blocks.ml` functions.

**Out of scope:**
- Full abstract interpretation (skipping body re-execution entirely, copying traced tensor graphs). This is the more ambitious version described in `docs/shape_inference.md` and would be a follow-up.
- Higher-order shape schemes (functions of functions).
- Scheme serialization or cross-module sharing.

**Dependencies:**
- gh-ocannl-372 (tensor node namespaces) -- prerequisite, completed.
- Touches the same PPX infrastructure as gh-ocannl-348 (simplify `%cd` syntax).

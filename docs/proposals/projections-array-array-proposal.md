# Proposal: Convert `projections` from `axis_index array` to `axis_index array array`

**Issue**: [gh-ocannl-421](https://github.com/ahrefs/ocannl/issues/421)
**Author**: Draft proposal (worker agent)
**Date**: 2026-03-03
**Status**: Draft
**Blocks**: gh-ocannl-398 (RoPE), transformer chain

## Summary

This proposal details the conversion of the `projections` record type so that `project_lhs` becomes `axis_index array array` and `project_rhs` becomes `axis_index array array array`, where the new middle array dimension indexes **block (concatenation) components**. Non-concatenation operations use singleton outer arrays, preserving backward compatibility. This change makes concatenation a first-class structural concept in projections rather than an encoding via the `Concat` variant of `axis_index`, and unlocks using `%cd` syntax for defining concatenation-involving operations (currently impossible because `%cd` only handles fixed RHS slots).

---

## 1. Current State Analysis

### 1.1 The `projections` type (`arrayjit/lib/indexing.ml`, lines 136-157)

```ocaml
type projections = {
  product_space : int list array;
  lhs_dims : int array;
  rhs_dims : int array array;
  product_iterators : symbol list array;
  project_lhs : axis_index array;           (* product space -> LHS index *)
  project_rhs : axis_index array array;     (* project_rhs.(i) -> RHS_i index *)
  debug_info : projections_debug;
}
```

The `product_space` and `product_iterators` fields already use `int list` and `symbol list` (not scalars) to accommodate concatenation dimensions, where multiple symbols/dimensions share a single product-space axis. However, `project_lhs` and `project_rhs` remain flat arrays, and the connection between concatenation components and RHS tensors is encoded implicitly via the `Concat of symbol list` variant in `axis_index`.

### 1.2 The `Concat` variant (`arrayjit/lib/indexing.ml`, lines 117-119)

```ocaml
| Concat of symbol list
    (** This axis is formed by concatenating multiple axes, each represented
        by an iterator symbol. Concat indices are eliminated during lowering. *)
```

When a dimension is concatenated (e.g., `a^b^c`), shape inference produces a `Concat [s_a; s_b; s_c]` axis index in `project_lhs`. Each symbol `s_x` corresponds to one RHS tensor's contribution to that concatenated axis.

### 1.3 Block lowering (`arrayjit/lib/assignments.ml`, lines 280-396)

The `loop_accum` function handles `Block` RHS by:

1. **Extracting `concat_syms_opt`** (line 309-317): scanning `project_lhs` for a single `Concat` variant, extracting its symbol list, and checking it matches the number of RHS tensors.
2. **Substituting indices** (lines 326-364): for each block iteration, the active concat component is determined by checking which concat symbol is in `block_iters`, then resolved to an `Iterator` (or `Affine` with offset).
3. **Filtering RHS** (lines 373-384): for each RHS tensor, checking whether its concat symbol is active in the current block iteration via `allow_by_concat`.
4. **Empty block handling**: raising `Empty_block` exception when a product iterator is active but not in `block_iters`, allowing the loop structure to skip inactive blocks.

The `loop_accum_rev` function (lines 452-628) mirrors this for `Rev_sides` (gradient of concatenation), with similar concat resolution and RHS filtering logic.

### 1.4 PPX slot access (`tensor/ppx_cd.ml`, lines 200-232)

The `%cd` macro generates projection access code for fixed slots:

```ocaml
let project_p_slot debug loc slot =
  match slot with
  | LHS  -> [%expr p.project_lhs]
  | RHS1 -> [%expr p.project_rhs.(0)]
  | RHS2 -> [%expr p.project_rhs.(1)]
  | RHS3 -> [%expr p.project_rhs.(2)]
  | Scalar -> [%expr [| Ir.Indexing.Fixed_idx 0 |]]
  ...
```

This hardcoded slot system means `%cd` cannot generate code for block operations where the number of RHS tensors is dynamic.

### 1.5 Shape derivation (`tensor/shape.ml`, lines 1984-2058)

The `derive_projections` function builds projections from shape inference results:

```ocaml
let indices_of_sh (sh : t) =
  Array.of_list_map ~f:(Row.get_dim_index proj_env)
  @@ List.concat [ sh.batch.dims; sh.output.dims; sh.input.dims ]
in
...
project_lhs = indices_of_sh lhs;
project_rhs = Array.of_list_map ~f:indices_of_sh rhs;
```

`Row.get_dim_index` (in `tensor/row.ml`, line 3961) delegates to `get_proj_index`, which for `Concat` dimensions (line 3911-3927) collects all component iterator symbols into a flat `Idx.Concat syms` value.

### 1.6 Surjectivity/injectivity checks (`arrayjit/lib/indexing.ml`)

- `is_surjective` (line 164): conservatively returns `false` for multiple LHS Concat axes (line 192-195). For single Concat axes, it collects symbols from the `Concat` variant and checks coverage.
- `is_injective` (line 243): collects symbols from `Concat` variants into the LHS symbol set and checks partition properties against product iterator sets.

### 1.7 `concat_sum` operation (`tensor/operation.ml`, lines 448-478)

Currently bypasses `%cd` entirely, manually building `Asgns.Accum_op { rhs = Block { ... } }` for the forward pass and `Asgns.Accum_op { rhs = Rev_sides { ... } }` for the backward pass.

---

## 2. Proposed Changes

### 2.1 New `projections` type

```ocaml
type projections = {
  product_space : int list array;
  lhs_dims : int array;
  rhs_dims : int array array;
  product_iterators : symbol list array;
  project_lhs : axis_index array array;
      (** [project_lhs.(block_component).(axis)] — for non-concat operations,
          this is always [|single_projection|]. For concat operations, each
          element is the projection for one block component. *)
  project_rhs : axis_index array array array;
      (** [project_rhs.(rhs_index).(block_component).(axis)] — the middle
          dimension indexes block components, matching [project_lhs]. *)
  debug_info : projections_debug;
}
```

**Key invariant**: For non-concatenation operations, `project_lhs` has exactly one element (`[| proj |]`) and each `project_rhs.(i)` has exactly one element (`[| proj_i |]`). For concatenation operations with `n` components, `project_lhs` has `n` elements, and each `project_rhs.(i)` has `n` elements (some of which may be "empty" / inactive for that component).

### 2.2 Representation of inactive block components

When a block component does not involve a particular RHS tensor, we need a way to signal "skip this RHS for this component." Options:

- **Option A**: Use a sentinel value, e.g., `[||]` (empty array) for the axis_index array of an inactive component.
- **Option B**: Introduce a new type `block_component = Active of axis_index array | Inactive`.
- **Option C**: Use `axis_index array option array` for the middle dimension.

**Recommendation**: Option A (`[||]` as sentinel). It requires no new types, is simple to check (`Array.is_empty`), and naturally extends the existing array-based representation. The lowering code already has an `Empty_block` exception mechanism; `[||]` is the structural equivalent.

### 2.3 Handling of the `Concat` variant

After this change, the `Concat` variant of `axis_index` becomes unnecessary for new projections — concatenation is represented structurally. However, we should **deprecate rather than immediately remove** it:

- Phase 1-3 keep `Concat` in the type but stop producing it in new projection derivations.
- Code that pattern-matches on `Concat` can be gradually simplified.
- Final removal can happen in a follow-up once all uses are confirmed eliminated.

### 2.4 Pre-computed offsets

Currently, `concat_offset_for` in `assignments.ml` computes concatenation offsets at lowering time by walking the symbol list. With the new structure, offsets are **pre-computed during projection derivation** in `shape.ml`. Each block component's LHS projection already has the correct offset baked into its `Affine` or `Iterator` index.

For example, concatenating `a` (dim 3) and `b` (dim 2):
- **Current**: `project_lhs = [| ...; Concat [s_a; s_b]; ... |]`, and lowering computes offset 0 for `s_a`, offset 3 for `s_b`.
- **Proposed**: `project_lhs = [| [| ...; Iterator s_a; ... |]; [| ...; Affine {symbols=[(1,s_b)]; offset=3}; ... |] |]`

---

## 3. File-by-File Change Targets

### 3.1 `arrayjit/lib/indexing.ml`

| Function/Type | Change |
|---|---|
| `projections` type | `project_lhs : axis_index array array`, `project_rhs : axis_index array array array` |
| `identity_projections` | Wrap results in singleton: `project_lhs = [| proj |]`, `project_rhs = [| [| proj |] |]` |
| `is_surjective` | Iterate over block components; surjective iff union of all components covers LHS dims |
| `is_injective` | Check no two components write to overlapping LHS positions |
| `reflect_projection` | Handle `axis_index array array` — apply to each component or require singleton |
| `pp_indices` (Doc_helpers) | Display multi-component projections with block-component grouping |

### 3.2 `arrayjit/lib/assignments.ml`

| Function | Change |
|---|---|
| `loop_accum` (Block path) | Replace `concat_syms_opt` extraction + `allow_by_concat` filtering with iteration over `project_lhs.(component)` and `project_rhs.(i).(component)`. The `subst_index` function no longer needs a `Concat` case. |
| `loop_accum_rev` (Rev_sides path) | Mirror changes. `target_projections` construction simplifies. |
| `loop_accum` (Unop/Binop/Ternop paths) | Unwrap singleton: `project_lhs.(0)`, `project_rhs.(i).(0)`. |
| `loop_set_vec_unop` | Unwrap singleton. |
| `can_skip_accumulation` | Per-component check or require singleton. |
| `is_total` | Per-component check or require singleton. |

### 3.3 `tensor/shape.ml`

| Function | Change |
|---|---|
| `derive_projections` (~line 2038-2058) | For non-Block operations: wrap in singleton. For Block operations: produce multi-component projections with pre-computed offsets per component. |
| `indices_of_sh` (~line 1984) | No change needed for non-concat. For concat, called per-component with the appropriate RHS shape. |
| Block shape logic (~lines 1279-1430) | May need adjustments to track per-component projection info. |

### 3.4 `tensor/row.ml`

| Function | Change |
|---|---|
| `get_proj_index` Concat case (~line 3911) | Instead of producing `Idx.Concat syms`, produce per-component `axis_index array` values that will be assembled into the outer array by `derive_projections`. |

### 3.5 `tensor/ppx_cd.ml`

| Function | Change |
|---|---|
| `project_p_slot` (~line 200) | Return `p.project_lhs.(0)` for LHS, `p.project_rhs.(i).(0)` for RHS_i (singleton unwrap for non-block ops). |
| New: block iteration support | Add mechanism for `%cd` to iterate over block components (Phase 5). |
| `projections_slot` type | Consider adding `RHS_BLOCK` variant or a loop construct (Phase 5). |

### 3.6 `tensor/operation.ml`

| Function | Change |
|---|---|
| `concat_sum` (~line 448) | Phase 6: optionally rewrite using `%cd` syntax. |

---

## 4. Phase-by-Phase Implementation Plan

### Phase 1: Change the type and add singleton wrapping (Day 1-2)

**Goal**: The type changes, all code compiles, all tests pass. No semantic change.

1. Change `projections` type in `indexing.ml`.
2. Update `identity_projections` to wrap in singletons.
3. Update all projection **creation** sites in `shape.ml` (`derive_projections`) to wrap in singletons.
4. Update all projection **access** sites:
   - `assignments.ml`: `projections.project_lhs` -> `projections.project_lhs.(0)`, `projections.project_rhs.(i)` -> `projections.project_rhs.(i).(0)` in non-Block paths.
   - `ppx_cd.ml`: `p.project_lhs` -> `p.project_lhs.(0)`, `p.project_rhs.(i)` -> `p.project_rhs.(i).(0)`.
   - `indexing.ml`: `is_surjective`, `is_injective`, `reflect_projection` — operate on `.(0)`.
5. For Block/Rev_sides paths in `assignments.ml`, temporarily keep existing logic operating on `.(0)` (these still use `Concat`-encoded projections).

**Verification**: All existing tests pass. `dune test` green. No behavioral change.

### Phase 2: Produce multi-component projections for Block operations (Day 3-4)

**Goal**: Shape inference produces multi-component projections. Block lowering consumes them.

1. In `shape.ml` `derive_projections`, for `Block` operations:
   - Instead of producing `Concat syms` in `project_lhs`, produce `n` component arrays (one per RHS).
   - Each component array has the same shape as a non-concat projection, but with the concat axis replaced by `Iterator s_i` (or `Affine { symbols=[(1,s_i)]; offset=cum_offset }`) for the `i`-th component.
   - `project_rhs`: for each RHS `j`, produce `n` component arrays. Component `k` is `[||]` (inactive) if `k != j`, and the normal projection for RHS `j` if `k == j`.

2. In `row.ml` `get_proj_index`, the `Concat` case can either:
   - (a) Return a new structured result that `derive_projections` unpacks, or
   - (b) `derive_projections` calls `get_dim_index` per-component instead of per-shape.

   Option (b) is cleaner: for Block operations, instead of `indices_of_sh` which calls `get_dim_index` on the full shape (including Concat dims), we call a new `per_component_indices_of_sh` that replaces each Concat dim with the specific component's iterator + offset.

3. Update `assignments.ml` Block lowering (`loop_accum`):
   - Remove `concat_syms_opt` extraction.
   - Remove `Concat` case from `subst_index`.
   - Instead of filtering RHS by `allow_by_concat`, iterate over components: for component `k`, use `project_lhs.(k)` and for each RHS `i`, use `project_rhs.(i).(k)` (skip if `[||]`).
   - The `for_loop` / `basecase` structure changes: instead of product-space iteration selecting which block is active (via `block_iters` membership), the component index is explicit.

4. Update `loop_accum_rev` similarly.

**Verification**: All existing tests pass. Concat tests exercise the new path.

### Phase 3: Simplify surjectivity/injectivity for multi-component projections (Day 4-5)

**Goal**: `is_surjective` and `is_injective` work correctly with multi-component projections.

1. `is_surjective`: A multi-component projection is surjective if, for each LHS position, at least one component writes to it. For disjoint block concatenation (the common case), this means checking that the union of component ranges covers the LHS dimension.

2. `is_injective`: A multi-component projection is injective if no two components write to the same LHS position. For disjoint block concatenation, this is automatically true (non-overlapping offset ranges).

3. Helper: `is_disjoint_block proj` — checks that block components have non-overlapping LHS ranges. When true, surjectivity reduces to: each component is surjective over its sub-range, and the sub-ranges tile the LHS.

**Verification**: Initialization optimization still works correctly for concat operations.

### Phase 4: Update `%cd` PPX for block components (Day 5-6)

**Goal**: `%cd` can generate code that works with multi-component projections.

For the immediate term, the simplest approach is:

1. **Singleton unwrap**: `project_p_slot` generates `p.project_lhs.(0)` and `p.project_rhs.(i).(0)` — this works for all non-block operations and is the minimal change.

2. **Block-aware code generation** (stretch): Add a new `%cd` mechanism for iterating over block components. Design options:
   - **Approach A: Loop emission**. Add a special slot `RHS_BLOCK(i)` that generates:
     ```ocaml
     Array.iteri p.project_lhs ~f:(fun _component_idx lhs_proj ->
       let rhs_proj = p.project_rhs.(i).(_component_idx) in
       if not (Array.is_empty rhs_proj) then ...)
     ```
   - **Approach B: Array-of-tensors parameter**. Extend `%cd` syntax to accept `rhses : buffer array` and generate iteration code.
   - **Approach C: Deferred**. Keep `concat_sum` using the raw API for now. The value of this change is primarily architectural (cleaner projection type), not necessarily requiring `%cd` rewrite of concat operations immediately.

**Recommendation**: Implement the singleton unwrap (4.1) in this task. Defer block-aware `%cd` generation (4.2) to a follow-up, since the primary value of gh-ocannl-421 is in making the projection type correct and simplifying the lowering code.

### Phase 5: Clean up and deprecate `Concat` variant (Day 7)

1. Verify that no code path produces `Concat` axis indices anymore.
2. Add a deprecation warning or comment on the `Concat` variant.
3. Remove `Concat` handling from `subst_index` in assignments.ml (now dead code).
4. Simplify `reflect_projection` (remove `Concat` case).
5. Simplify `is_surjective` and `is_injective` (remove `Concat`-specific logic).

### Phase 6: Migrate `concat_sum` (Day 7-8, stretch)

1. If block-aware `%cd` is implemented, rewrite `concat_sum` and its gradient using `%cd`.
2. If not, verify that `concat_sum` works correctly with the new projection structure when using the raw API.
3. Update `accum_rhs` type: the `Block` and `Rev_sides` variants may be simplified or generalized.

---

## 5. Risk Assessment and Edge Cases

### 5.1 Performance

- **Extra indirection**: Non-concat operations now access `project_lhs.(0).(axis)` instead of `project_lhs.(axis)` — one extra array dereference. This is in a hot loop during lowering (which happens once per compilation), not during runtime execution, so impact should be negligible.
- **Lowering code size**: Multi-component projections may produce more lowering code (one block per component). This matches the current behavior where `for_loop` in `loop_accum` already iterates over product-space entries.

### 5.2 `Rev_sides` semantics

The current `Rev_sides` path swaps LHS and RHS: `lhs` becomes the read tensor, `lhses` become the write targets. With multi-component projections:
- `project_lhs.(k)` gives the read indices for component `k`.
- `project_rhs.(i).(k)` gives the write indices for target `i` at component `k`.
- Active/inactive components work the same way (`[||]` means skip).

This is a natural extension; no special handling needed beyond the same component iteration.

### 5.3 Nested/multi-axis concatenation

A spec like `"a;b;c => a^b, c^d"` with two concatenated axes is currently not well-supported (produces two `Concat` variants in `project_lhs`, causing `concat_syms_opt` to fail). With multi-component projections, each concatenated axis contributes independently to the component structure. If there are `m` components on axis 1 and `n` components on axis 2, we get `m * n` block components (Cartesian product). This is a natural extension but should be verified with tests.

### 5.4 `product_space` and `product_iterators` interaction

Currently, `product_space` and `product_iterators` use list-of-lists for concat dimensions. With multi-component projections, the product-space iteration structure could potentially simplify (each component has well-defined scalar product-space entries). However, the `for_loop` structure in `loop_accum` depends on the list-of-lists to generate the branching loop. Changing this is a separate concern and should not be done in this task — the existing product-space structure is compatible with multi-component projections.

### 5.5 Backward compatibility of serialized projections

The `projections` type derives `sexp`, so any serialized projections will break. This should not be an issue since projections are computed at compile time and not persisted across versions.

### 5.6 `Set_vec_unop` path

The `loop_set_vec_unop` function (assignments.ml ~line 640) accesses `projections.project_lhs` and `projections.project_rhs.(0)` and explicitly rejects `Concat` (line 660). With the new type, this path simply uses `.(0)` (it only supports unary operations, never block operations).

### 5.7 Mixed concat and non-concat axes

A projection may have some axes that are concatenated and others that are not. In the multi-component representation, the non-concat axes have identical values across all components, and only the concat axis differs. This is naturally represented.

---

## 6. Testing Strategy

### 6.1 Existing tests (must all pass at every phase)

- `test/operations/test_concat_graph.ml` — 2-way, 3-way, and 3-way with unit-dim concatenation with forward and backward passes.
- `test/operations/test_concat_ppx.ml` — `%op`-syntax concat smoke tests.
- All other operation tests (einsum, pointwise, convolution, etc.) — must be unaffected by singleton wrapping.

### 6.2 New tests to add

1. **Singleton unwrap verification**: Add a test that creates a non-concat operation, inspects the projection, and verifies `Array.length proj.project_lhs = 1`.
2. **Multi-component verification**: Add a test that creates a concat operation, inspects the projection, and verifies `Array.length proj.project_lhs = num_components`.
3. **Inactive component**: Verify that for a 3-way concat `a;b;c => a^b^c`, `project_rhs.(0).(1)` and `project_rhs.(0).(2)` are `[||]` (inactive for RHS 0 at components 1 and 2).
4. **Pre-computed offsets**: Verify that `project_lhs.(1)` for a 2-way concat contains the correct offset (sum of dim 0's size).
5. **Surjectivity check**: Verify that `is_surjective` returns `true` for disjoint block concatenations.
6. **Injectivity check**: Verify that `is_injective` returns `true` for disjoint block concatenations.
7. **Gradient correctness**: Run numerical gradient checks on concat operations (already partially covered by `test_concat_graph.ml`).

### 6.3 Regression testing

- Run `dune test` after each phase.
- Verify that generated lowered code (from `%cd` debug output) is structurally equivalent before and after the change for non-concat operations.

---

## 7. Acceptance Criteria Checklist

- [ ] `project_lhs` type is `axis_index array array`
- [ ] `project_rhs` type is `axis_index array array array`
- [ ] Non-concatenation operations use singleton outer arrays
- [ ] `identity_projections` produces singleton-wrapped projections
- [ ] `derive_projections` in `shape.ml` produces multi-component projections for Block operations
- [ ] Block lowering in `assignments.ml` iterates over components instead of resolving `Concat`
- [ ] `Rev_sides` lowering updated for multi-component projections
- [ ] `is_surjective` handles multi-component projections correctly
- [ ] `is_injective` handles multi-component projections correctly
- [ ] `%cd` PPX generates `.(0)` access for non-block operations
- [ ] `Concat` variant deprecated (not produced by new code paths)
- [ ] Existing concatenation tests pass (2-way, 3-way, unit-dim)
- [ ] Existing non-concatenation tests pass (no regression)
- [ ] No performance regression in lowering
- [ ] **Stretch**: `concat_sum` rewritten using `%cd` syntax
- [ ] **Stretch**: `%cd` supports block-component iteration for new concat-involving operations

---

## 8. Open Questions

1. **Should `product_space` and `product_iterators` also be restructured?** Currently they use `int list array` and `symbol list array` where the inner list indexes concat components. With multi-component projections, these could become scalar arrays (`int array` and `symbol array`) per component. However, this is a larger change and may not be necessary — the existing structure is compatible.

2. **Cartesian product for multi-axis concatenation**: If a spec has two concatenated axes with `m` and `n` components respectively, should we produce `m * n` block components? Or is multi-axis concatenation not supported / not needed?

3. **`accum_rhs` type simplification**: After this change, should `Block` and `Rev_sides` be unified with `Unop`/`Binop`/`Ternop`? The distinction is that Block has a variable number of RHS tensors, which the type system still needs to distinguish.

4. **Interaction with tiling (gh-ocannl-350)**: Tiling transforms operate on projections. Multi-component projections should be tiled per-component, which is a natural extension. Confirm no conflicts with the loop-hoisting work in slot 3.

# Product Dimension Implementation Status

## Summary

This document tracks the implementation of the `Prod of dim list` construct in OCANNL's shape inference system. This construct represents an axis that is a product of other axes, useful for modeling concatenation or multi-axis views.

## Completed Changes

### 1. Documentation (shape_inference.md)

- ✅ Added documentation about the `Prod` construct in the type definition section
- ✅ Added a new section "Product dimensions (Prod)" explaining:
  - Purpose (concatenation and multi-axis views)
  - Semantics (dimension = product of constituent dimensions)
  - Projection behavior (respects order, row-major indexing)
  - Interaction with other inference features
  - Planned einsum notation syntax (`i&j`)

### 2. Row Module Interface (row.mli)

- ✅ Added helper functions:
  - `val dim_vars : dim -> dim_var list`
  - `val is_solved_dim : dim -> bool`

### 3. Row Module Implementation (row.ml)

- ✅ Updated `dim_to_int_exn` to handle `Prod` by computing the product recursively
- ✅ Updated `apply_dim_constraint` to handle `Prod` with `At_least_dim` constraints
- ✅ Updated `reduce_row_constraint` to extract dimensions and variables from nested products
- ✅ Updated `_lift_row_constraint` similarly

## Remaining Work

### 1. Row Module Implementation (row.ml)

The following functions still need `Prod` cases added:

#### Pattern matching functions

- `s_dim_one` - needs to substitute inside Prod constituents
- `subst_dim` - needs to recursively substitute in Prod constituents
- `s_dim_one_in_row_constr` - handle when value is Prod
- `unify_dim` - handle unification of Prod dimensions
- `solve_dim_ineq` - handle inequalities involving Prod
- `close_dim_terminal` - handle terminal Prod dimensions
- `last_dim_is` - check if last dim is a specific Prod
- `row_to_labels` - generate labels for Prod dimensions
- `fresh_row_proj` - handle projection IDs in Prod
- `get_proj_index` - handle projection indices for Prod
- Various functions in `apply_row_constraint`

#### Implementation suggestions

1. **For substitution functions**: Recursively apply substitution to all constituents
2. **For unification**: Two Prods unify if they have the same structure and corresponding dimensions unify
3. **For inequalities**: `Prod ds1 >= Prod ds2` if they have compatible structure and element-wise inequalities hold
4. **For projections**: Generate composite projections that respect the row-major ordering of constituents

### 2. Helper Function Implementations

```ocaml
let get_prod dims = Prod dims

let rec dim_vars = function
  | Var v -> [v]
  | Dim _ -> []
  | Prod dims -> List.concat_map dims ~f:dim_vars

let rec is_solved_dim = function
  | Var _ -> false
  | Dim _ -> true
  | Prod dims -> List.for_all dims ~f:is_solved_dim
```

### 3. Shape Module Updates

- Update `row_to_dims` to handle Prod
- Any other functions that pattern match on dim

### 4. Einsum Notation Extension

- Add parser support for `&` operator in einsum specs
- Update `einsum_slot_spec_to_dims_bio` to parse and create Prod dimensions
- Handle the semantics of product axes in einsum operations

### 5. Testing

- Create comprehensive tests for:
  - Basic Prod dimension inference
  - Nested Prod structures
  - Prod in constraints
  - Prod in projections
  - Einsum notation with `&`
  - Edge cases (empty Prod, single-element Prod, etc.)

## Design Considerations

1. **Broadcasting**: A Prod can broadcast only if all its constituents can broadcast (i.e., are dimension-1)
2. **Labels**: How should labels work with Prod? Concatenate constituent labels? New label?
3. **Projections**: The projection for a Prod should decompose the flat index into constituent indices
4. **Normalization**: Should we normalize `Prod [Dim d]` to `Dim d`? `Prod []` to `Dim 1`?
5. **Flattening**: Should nested Prods be flattened? `Prod [Prod [a; b]; c]` → `Prod [a; b; c]`?

### Key Design Decisions Made

1. **Prod comparison**: Two Prod dimensions are considered equal only if they have the exact same structure (`List.equal equal_dim`)
2. **Prod in inequalities**: When a Prod is fully solved (all constituents are concrete dimensions), we expand it to its product value for comparison
3. **Unsolved Prod**: When Prod contains variables, we generally cannot compare it in inequalities and raise appropriate errors
4. **LUB computation**: For different Prod structures or Prod vs Dim, we default to dimension-1 when values don't match

The remaining functions follow similar patterns - they need to handle Prod by either:

- Recursing into the constituents (for functions like `row_to_labels`)
- Computing the product when fully solved (for projection-related functions)
- Preserving the structure (for functions like `fresh_row_proj`)

## Next Steps

1. ✅ Implement the helper functions in row.ml
2. Systematically go through each linter error and add Prod handling
3. Update shape.ml for any needed changes
4. Design and implement einsum notation parsing for `&`
5. Create a comprehensive test suite
6. Review and refine the design based on testing

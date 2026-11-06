# Proposal: Accept `=:` syntax in `%cd` for the `Fetch` constructor

**Issue:** [ahrefs/ocannl#209](https://github.com/ahrefs/ocannl/issues/209)
**Status:** Draft proposal

## Summary

`Assignments.to_doc` outputs `Fetch` operations using `=:` syntax (e.g. `x =: 0.;`), but the `%cd` PPX parser cannot parse these back. This proposal adds parser clauses so that simple `Fetch` operations written with `=:` in `%cd` blocks are accepted, closing the round-trip gap for readable output.

## Current State

### What `to_doc` outputs for `Fetch` (assignments.ml:940-941)

```ocaml
| Fetch { array; fetch_op; dims = _ } ->
    string (ident array) ^^ string " =: " ^^ doc_of_fetch_op fetch_op ^^ string ";" ^^ break 1
```

The `doc_of_fetch_op` function (assignments.ml:832-847) formats each variant:

| `fetch_op` variant | Output example |
|---|---|
| `Constant f` | `x =: 0.;` |
| `Constant_bits i` | `x =: bits(42LL);` |
| `Constant_fill values` | `x =: constant_fill([1., 2., 3.]);` |
| `Range_over_offsets` | `x =: range_over_offsets();` |
| `Slice { batch_idx; sliced }` | `x =: y @\| i;` |
| `Embed_symbol s` | `x =: !@i;` |
| `Embed_self_id` | `x =: self_id();` |
| `Embed_dim { ref_label; _ }` | `x =: (dim label);` |

### How `=:` is currently parsed in `%cd` (ppx_cd.ml)

The `=:` operator currently serves two purposes in the parser:

1. **Vec unary ops** (ppx_cd.ml:1509-1519, 1645-1647): `lhs =: vec_unary_op rhs` maps to `Set_vec_unop`.
2. **Assignment accumulation** (ppx_shared.ml:508): `"=:"` in `assignment_ops` maps to `(false, Arg2)`, creating an `Accum_op` with identity-like semantics.

Neither path produces a `Fetch` constructor. `Fetch` is only generated internally by `tensor.ml` (lines 177, 559) during tensor construction, never from user-written `%cd` syntax.

### The `is_assignment` guard (ppx_shared.ml:396-399)

```ocaml
let is_assignment ident =
  String.length ident > 1
  && Char.equal ident.[0] '='
  && (not @@ List.mem [ "=="; "==="; "=>"; "==>"; "=>>" ] ident ~equal:String.equal)
```

`"=:"` passes this guard (starts with `=`, length > 1, not in exclusion list).

## Proposed Changes

### Scope: Constant fetch only (phase 1)

The most useful and lowest-risk subset is supporting `lhs =: <float_literal>;` which maps to `Fetch { array = lhs; fetch_op = Constant f; dims }`. This covers the most common `Fetch` usage (zero-initialization, constant fills).

Other `fetch_op` variants (`Slice`, `Embed_symbol`, etc.) involve information that is harder to infer from syntax alone (batch indices, static symbols, dimension refs). These can be added in follow-up work.

### 1. Add parser clauses in `ppx_cd.ml`

Add new match clauses in the `transl` function, **before** the existing `=:` vec_unop clauses (around line 1644), to catch `lhs =: <constant>`:

```ocaml
(* Fetch: lhs =: <float_constant> *)
| [%expr [%e? lhs] =: [%e? { pexp_desc = Pexp_constant (Pconst_float (_, _)); _ } as rhs]]
  when proj_in_scope ->
    process_fetch_constant ~lhs ~rhs ()

(* Fetch: lhs =: <int_constant> (coerced to float) *)
| [%expr [%e? lhs] =: [%e? { pexp_desc = Pexp_constant (Pconst_integer (_, _)); _ } as rhs]]
  when proj_in_scope ->
    process_fetch_constant ~lhs ~rhs ()
```

### 2. Add `process_fetch_constant` helper in `ppx_cd.ml`

Add this helper near the other `process_*` functions (around line 847):

```ocaml
let process_fetch_constant ~lhs ~rhs () =
  let lhs_result = loop ~proj_in_scope:true lhs in
  let setup_l = setup_array ~punned ~bad_pun_hints ~for_slot:LHS lhs_result in
  let body_for_lhs =
    [%expr
      [%e shape_infer loc lhs_result];
      Some
        (Ir.Assignments.Fetch
           {
             array = lhs;
             fetch_op = Constant [%e rhs];
             dims = lazy (Ir.Shape.to_dims (Lazy.force lhs.Ir.Tnode.dims));
           })]
  in
  assignment ~punned ~lhs:setup_l ~rhses:[] ~body_for_lhs ()
in
```

**Key design decisions:**
- `dims` is derived from the LHS tensor's shape, matching how `tensor.ml:fetch_zeros` works.
- No RHS setup is needed since the RHS is a literal constant, not a tensor.
- `rhses:[]` because `Fetch` has no tensor right-hand sides.

### 3. File locations

| File | Change |
|---|---|
| `tensor/ppx_cd.ml` ~line 847 | Add `process_fetch_constant` helper |
| `tensor/ppx_cd.ml` ~line 1644 | Add match clauses for `=:` with constant literals |

### No changes needed in:
- `arrayjit/lib/assignments.ml` -- `Fetch` type and `to_doc` already correct
- `tensor/ppx_shared.ml` -- `assignment_ops` already has `"=:"` entry
- `arrayjit/lib/ops.ml` -- `assign_op_cd_syntax` already handles `Arg2 -> "=:"`

## Ordering Considerations

The new clauses must appear **before** the existing `=:` vec_unop clauses. The pattern matching order matters:

1. `lhs =: <float_literal>` -- NEW (Fetch constant)
2. `lhs =: <int_literal>` -- NEW (Fetch constant, coerced)
3. `lhs =: vec_un_op rhs ~projections:...` -- existing (Set_vec_unop with projections)
4. `lhs =: vec_un_op rhs` -- existing (Set_vec_unop without projections)

There is no ambiguity because the new clauses match a bare literal on the RHS, while vec_unop clauses require an identifier (the op name) followed by an argument.

## Test Plan

1. **Unit test**: Add a test in `test/operations/` that uses `%cd` with `=: <constant>`:
   ```ocaml
   let%cd x =: 0.
   let%cd y =: 42.
   ```
   Verify these compile and produce the expected `Fetch` assignments.

2. **Round-trip test**: Generate `to_string_hum` output for a `Fetch` constant, verify the output matches the `%cd` input syntax.

3. **Regression**: Ensure existing `=:` vec_unop usage still works (run existing test suite).

4. **Edge cases**:
   - Negative constants: `x =: -1.0` (may parse as unary negation applied to literal)
   - Integer constants: `x =: 0` (should coerce to float)

## Risk Assessment

**Low risk.** This is an additive change:
- New match clauses are pattern-specific (literal constants only) and cannot conflict with existing clauses
- No existing behavior is modified
- The `Fetch` constructor is well-established in the codebase

**Potential concerns:**
1. **Negative literals**: OCaml parses `-1.0` as `Pexp_apply(-, 1.0)` not `Pexp_constant(-1.0)`. May need an additional clause for `=: - <literal>` or accept only non-negative constants initially.
2. **Overlap with gh-ocannl-348** (slot 3, `%cd` simplifications): That task may restructure the parser. Coordinate to avoid merge conflicts. However, since this adds new clauses rather than modifying existing ones, conflicts should be minimal.
3. **`rhses:[]`**: The `assignment` helper may expect at least one RHS setup. Need to verify it handles an empty `rhses` list gracefully.

## Future Extensions

After the constant case is proven, subsequent PRs could add:
- `lhs =: rhs @| idx` for `Slice`
- `lhs =: !@sym` for `Embed_symbol`
- `lhs =: range_over_offsets()` for `Range_over_offsets`
- `lhs =: constant_fill([...])` for `Constant_fill`

Each of these requires understanding how to resolve the needed compile-time information (static symbols, dimension refs) from the syntactic context available in `%cd`.

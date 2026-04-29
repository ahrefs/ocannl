# Proposal: Track whether Local_scope variables need initialization

**Issue**: https://github.com/ahrefs/ocannl/issues/340

## Goal

Eliminate unnecessary zero-initialization of `Local_scope` scalar variables in generated C code. Currently, every `Local_scope` variable is unconditionally zero-initialized (e.g., `float v42 = 0.0;`), but initialization is only needed when the variable is **recurrent** -- i.e., read before its first write (accumulator pattern: `v42 = v42 + delta`). The tracing infrastructure already detects this via `read_before_write` on `traced_array`; the missing piece is threading that flag into the `Local_scope` IR node and using it in code generation.

## Acceptance Criteria

- A `needs_init : bool` field is added to the `Local_scope` variant in `scalar_t` (both `low_level.ml` and `low_level.mli`).
- The field is populated from `traced_array.read_before_write` during `Local_scope` creation in `virtual_llc`.
- When `needs_init = false`, `c_syntax.ml` emits the declaration without zero-initialization (e.g., `float v42;` instead of `float v42 = (float)0;`).
- When `needs_init = true`, zero-initialization is preserved (no regression for accumulator patterns).
- The `TODO(#340)` comment in `c_syntax.ml` line 519 is removed.
- All existing tests pass (no regression).
- Conservative default: if the traced_array lookup fails, `needs_init` defaults to `true`.

## Context

### Current state: unconditional zero-init

In `c_syntax.ml` lines 518-521, every `Local_scope` variable gets zero-initialized:

```ocaml
let init_zero =
  (* TODO(#340): only do this in the rare cases where the computation is accumulating *)
  let prefix, postfix = B.convert_precision ~from:Ops.int32 ~to_:scope_prec in
  string " = " ^^ string prefix ^^ string "0" ^^ string postfix
in
```

This produces C code like `float v42_some_var = (float)0;` even when the variable is immediately assigned before any read.

### Existing detection infrastructure

The tracing pass (`visit_llc`, called first in `optimize_proc` at line 1394) already tracks whether each tensor node is read before written:

- **`traced_array.read_before_write`** (line 153): a `mutable bool` field, set to `true` at line 452 when any access position is `Recurrent`.
- **`visits` type** (line 145): `Visits of int | Recurrent` -- a position is marked `Recurrent` via the `visit` function (line 191) when accessed before being assigned.
- The `traced_store` hashtable, populated by `visit_llc`, is passed to `virtual_llc` where `Local_scope` is created.

### Local_scope creation site

In `virtual_llc` (line 775), `Local_scope` is created at lines 838-841:

```ocaml
let id = get_scope tn in
Option.value ~default:llsc
@@ Option.map (inline_computation ~id computations_table traced static_indices indices)
     ~f:(fun body -> Local_scope { id; body; orig_indices = indices })
```

The `traced` variable (of type `traced_array`) is already in scope here -- obtained at line 835 via `get_node traced_store tn`. Its `read_before_write` field is already populated by the earlier tracing pass.

### Downstream passes

After `virtual_llc`, the IR passes through:
1. `cleanup_virtual_llc` -- propagates `Local_scope` unchanged (line 933, 942)
2. `simplify_llc` -- may eliminate `Local_scope` nodes (lines 1044-1053) or propagate via `{ opts with body = ... }`
3. `eliminate_common_subexpressions` -- may replace duplicate `Local_scope` with `Get_local` (lines 1304-1311)

All of these use record update (`{ opts with ... }`) or pattern match with wildcards, so adding a new field requires only that propagation code preserves it. The simplification pass that eliminates `Local_scope` entirely (lines 1044-1050) doesn't need the field since the local variable ceases to exist.

### Backend scope

Only `c_syntax.ml` handles `Local_scope` for code generation (lines 514-528 for code gen, line 646 for debug output). The CUDA and Metal backends do not directly pattern-match on `Local_scope`.

### Related issue

gh-ocannl-420 addresses unnecessary `Zero_out` for arrays (a different level of initialization). This issue (#340) addresses unnecessary zero-init for scalar local variables. They are complementary optimizations.

## Approach

### 1. Add `needs_init` field to `Local_scope`

In `low_level.ml` line 53 and `low_level.mli` line 45, extend the record:

```ocaml
| Local_scope of { id : scope_id; body : t; orig_indices : Indexing.axis_index array; needs_init : bool }
```

### 2. Populate at creation site

In `virtual_llc` at line 841, use `traced.read_before_write`:

```ocaml
~f:(fun body -> Local_scope { id; body; orig_indices = indices; needs_init = traced.read_before_write })
```

At line 842-844 (the existing `Local_scope` propagation case), preserve the field via `{ opts with body = ... }` (already the pattern used).

### 3. Update c_syntax.ml

At line 514, add `needs_init` to the pattern match. At lines 518-521, conditionally emit zero-init:

```ocaml
| Local_scope { id = { tn = { prec = scope_prec; _ }; scope_id } as id; body; orig_indices = _; needs_init } ->
    let scope_prec = Lazy.force scope_prec in
    let num_typ = string (B.typ_of_prec scope_prec) in
    let init_zero =
      if needs_init then
        let prefix, postfix = B.convert_precision ~from:Ops.int32 ~to_:scope_prec in
        string " = " ^^ string prefix ^^ string "0" ^^ string postfix
      else
        empty
    in
```

Remove the `TODO(#340)` comment.

### 4. Update all pattern matches in low_level.ml

~20 locations in `low_level.ml` pattern-match on `Local_scope`. Most fall into two categories:

- **Wildcard the field** (read-only traversals): lines 72, 202, 222, 243, 396, 564, 1200, 1270-1271. These use `{ id; body; _ }` or similar -- already wildcard `orig_indices`, so no change needed if already using `_`.
- **Propagate the field** (transformations that rebuild `Local_scope`): lines 724-730, 933-942, 978, 1028-1053, 1304-1311. Most use `{ opts with body = ... }` which automatically preserves `needs_init`. The explicit record constructions at lines 724-730 and 1311 need `needs_init` added.

### 5. Handle inline_computation (lines 724-730)

This creates a new `Local_scope` inside `inline_computation` when transforming nested scopes. The `needs_init` from the original should be preserved:

```ocaml
| Local_scope { id; body; orig_indices; needs_init } ->
    Local_scope
      {
        id;
        body = Option.value_exn ~here:[%here] @@ loop env body;
        orig_indices = Array.map ~f:(subst env) orig_indices;
        needs_init;
      }
```

### 6. Handle CSE pass (lines 1304-1311)

The CSE pass rebuilds `Local_scope` after processing the body. Propagate `needs_init`:

```ocaml
| Local_scope { id; body; orig_indices; needs_init } ->
    ...
    let result = Local_scope { id; body; orig_indices; needs_init } in
```

### 7. Update expected test outputs

If any `.expected` test files show the zero-initialization pattern for non-recurrent `Local_scope` variables, they will need updating to reflect the removed `= (float)0` initializers. Run the test suite and update `.expected` files as needed.

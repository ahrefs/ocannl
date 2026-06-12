# Proposal: Track whether Local_scope variables need initialization

**Issue**: https://github.com/ahrefs/ocannl/issues/340

## Status update (2026-06-12)

- Issue is OPEN. The GitHub milestone reads v0.8, but ROADMAP.md (the authority) lists #340 under v0.7.2 "Compiler optimizations" (was due mid-April 2026, now past due).
- Not implemented: `Local_scope` still has no `needs_init` field (`low_level.ml:54`, `low_level.mli:48`), and the `TODO(#340)` comment is still present in `c_syntax.ml` — now at line 602 (was 519).
- New since the proposal was written: cross-statement CSE hoisting landed in `low_level.ml` (commits e48ec84f, 98410c21, f88fbdf8). It introduced a `Declare_local of scope_id` statement variant (`low_level.ml:50`) that splits a hoisted `Local_scope` into a declaration + body; `Declare_local` is a *second* unconditional zero-init site in `c_syntax.ml` (lines 585–591) that this design must also cover (or conservatively keep initializing hoisted declarations).
- The pass pipeline gained `hoist_cross_statement_cse` after `eliminate_common_subexpressions` (`low_level.ml:1632`); it is an additional downstream pass that must propagate the new field.
- Line numbers cited below have drifted (low_level.ml and c_syntax.ml both grew); they have been refreshed in place as of 2026-06-12.
- The design itself remains valid and unblocked; no part of the approach is invalidated.

## Goal

Eliminate unnecessary zero-initialization of `Local_scope` scalar variables in generated C code. Currently, every `Local_scope` variable is unconditionally zero-initialized (e.g., `float v42 = 0.0;`), but initialization is only needed when the variable is **recurrent** -- i.e., read before its first write (accumulator pattern: `v42 = v42 + delta`). The tracing infrastructure already detects this via `read_before_write` on `traced_array`; the missing piece is threading that flag into the `Local_scope` IR node and using it in code generation. *(Update 2026-06-12: this premise is wrong in a load-bearing way — `read_before_write` is only set when `not (Tn.known_virtual tn)`, and setting it also forces `Materialized` provenance 36 (`low_level.ml:454-458`), which makes `Tn.known_non_virtual` true and thus prevents `Local_scope` creation in `virtual_llc`. Consequently `traced.read_before_write` is identically `false` at every `Local_scope` creation site; see the Design review below for the corrected data source.)*

## Acceptance Criteria

- A `needs_init : bool` field is added to the `Local_scope` variant in `scalar_t` (both `low_level.ml` and `low_level.mli`).
- The field is populated from `traced_array.read_before_write` during `Local_scope` creation in `virtual_llc`. *(Update 2026-06-12: invalid — that flag is always `false` where `Local_scope` is created (see above), so this criterion amounts to unconditionally dropping the initializer. Replace with: `needs_init` is derived from a read-before-first-write scan of the scope body.)*
- When `needs_init = false`, `c_syntax.ml` emits the declaration without zero-initialization (e.g., `float v42;` instead of `float v42 = (float)0;`).
- When `needs_init = true`, zero-initialization is preserved (no regression for accumulator patterns).
- The `TODO(#340)` comment in `c_syntax.ml` line 602 is removed.
- All existing tests pass (no regression).
- Conservative default: if the traced_array lookup fails, `needs_init` defaults to `true`.

## Context

### Current state: unconditional zero-init

In `c_syntax.ml` lines 601-606, every `Local_scope` variable gets zero-initialized:

```ocaml
let init_zero =
  (* TODO(#340): only do this in the rare cases where the computation is accumulating *)
  let prefix, postfix = B.convert_precision ~from:Ops.int32 ~to_:scope_prec in
  string " = " ^^ string prefix ^^ string "0" ^^ string postfix
in
```

This produces C code like `float v42_some_var = (float)0;` even when the variable is immediately assigned before any read.

### Existing detection infrastructure

The tracing pass (`visit_llc`, called first in `optimize_proc` at line 1625) already tracks whether each tensor node is read before written:

- **`traced_array.read_before_write`** (line 154): a `mutable bool` field, set to `true` at line 455 when any access position is `Recurrent`.
- **`visits` type** (line 146): `Visits of int | Recurrent` -- a position is marked `Recurrent` via the `visit` function (line 192) when accessed before being assigned.
- The `traced_store` hashtable, populated by `visit_llc`, is passed to `virtual_llc` where `Local_scope` is created.

### Local_scope creation site

In `virtual_llc` (line 780), `Local_scope` is created at lines 843-847:

```ocaml
let id = get_scope tn in
Option.value ~default:llsc
@@ Option.map (inline_computation ~id computations_table traced static_indices indices)
     ~f:(fun body -> Local_scope { id; body; orig_indices = indices })
```

The `traced` variable (of type `traced_array`) is already in scope here -- obtained at line 841 via `get_node traced_store tn`. Its `read_before_write` field is already populated by the earlier tracing pass.

### Downstream passes

After `virtual_llc`, the IR passes through:
1. `cleanup_virtual_llc` -- propagates `Local_scope` unchanged (lines 940-949)
2. `simplify_llc` -- may eliminate `Local_scope` nodes (lines 1053-1062) or propagate via `{ opts with body = ... }`
3. `eliminate_common_subexpressions` -- may replace duplicate `Local_scope` with `Get_local` (lines 1324-1340)
4. `hoist_cross_statement_cse` (line 1586) -- *(Update 2026-06-12: new pass)* may split a `Local_scope` into a `Declare_local` statement followed by the body, hoisted to a common ancestor scope. `Declare_local` has its own unconditional zero-init in `c_syntax.ml` (lines 585-591), so `needs_init` must either be threaded into `Declare_local` too, or hoisted declarations conservatively keep zero-init.

All of these use record update (`{ opts with ... }`) or pattern match with wildcards, so adding a new field requires only that propagation code preserves it. The simplification pass that eliminates `Local_scope` entirely (lines 1053-1059) doesn't need the field since the local variable ceases to exist.

### Backend scope

Only `c_syntax.ml` handles `Local_scope` for code generation (lines 597-614 for code gen, line 732 for debug output; plus the `Declare_local` declaration at lines 585-591). The CUDA and Metal backends do not directly pattern-match on `Local_scope`.

### Related issue

gh-ocannl-420 addresses unnecessary `Zero_out` for arrays (a different level of initialization). This issue (#340) addresses unnecessary zero-init for scalar local variables. They are complementary optimizations.

## Approach

### 1. Add `needs_init` field to `Local_scope`

In `low_level.ml` line 54 and `low_level.mli` line 48, extend the record:

```ocaml
| Local_scope of { id : scope_id; body : t; orig_indices : Indexing.axis_index array; needs_init : bool }
```

### 2. Populate at creation site

In `virtual_llc` at line 847, use `traced.read_before_write`:

```ocaml
~f:(fun body -> Local_scope { id; body; orig_indices = indices; needs_init = traced.read_before_write })
```

At lines 848-850 (the existing `Local_scope` propagation case), preserve the field via `{ opts with body = ... }` (already the pattern used).

*(Update 2026-06-12: as noted above, `traced.read_before_write` is always `false` here. Either populate from `Hashtbl.exists traced.accesses ~f:is_recurrent` (un-gated), or — preferred — drop the field and compute `needs_init` syntactically from the body; see Design review.)*

### 3. Update c_syntax.ml

At line 597, add `needs_init` to the pattern match. At lines 601-606, conditionally emit zero-init:

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

- **Wildcard the field** (read-only traversals): lines 73, 203, 223, 244, 399, 568, 1209, 1291-1292. These use `{ id; body; _ }` or similar -- already wildcard `orig_indices`, so no change needed if already using `_`.
- **Propagate the field** (transformations that rebuild `Local_scope`): lines 729-736, 940-949, 985, 1037-1062, 1324-1331, plus the new `hoist_cross_statement_cse` pass (line 1586). Most use `{ opts with body = ... }` which automatically preserves `needs_init`. The explicit record constructions at lines 729-736 and 1331 need `needs_init` added.

### 5. Handle inline_computation (lines 729-736)

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

### 6. Handle CSE pass (lines 1324-1331) and cross-statement hoisting

The CSE pass rebuilds `Local_scope` after processing the body. Propagate `needs_init`:

```ocaml
| Local_scope { id; body; orig_indices; needs_init } ->
    ...
    let result = Local_scope { id; body; orig_indices; needs_init } in
```

*(Update 2026-06-12)*: the new `hoist_cross_statement_cse` pass (`low_level.ml:1586`) splits hoisted `Local_scope` nodes into `Declare_local of scope_id` + body. To get the benefit for hoisted scopes, either extend `Declare_local` to carry `needs_init` (and skip the zero-init at `c_syntax.ml:585-591` when false), or accept conservative zero-init for hoisted declarations in the first iteration.

### 7. Update expected test outputs

If any `.expected` test files show the zero-initialization pattern for non-recurrent `Local_scope` variables, they will need updating to reflect the removed `= (float)0` initializers. Run the test suite and update `.expected` files as needed.

## Design review (2026-06-12)

**Verdict: sound-with-changes.** The emission mechanism (conditional `init_zero`) is right; the proposed data source is wrong, and there is a simpler architecture than field-threading.

**Why the data source is wrong.** `visit_llc` sets `read_before_write` only under `not (Tn.known_virtual tn)` and simultaneously forces `Materialized` (provenance 36) — and `Materialized` makes `Tn.known_non_virtual` true (`tnode.ml:251-252`), which blocks `Local_scope` creation. So at the creation site the flag is *identically false*, and the proposal as written silently degenerates to "never zero-init". That is unsafe in exactly the corner where the initializer is load-bearing: a tensor already `Virtual` (e.g. shared across routines via `optimize_ctx`) with recurrent accesses skips the line-454 branch, gets inlined, and `inline_computation` can then produce a body like `Set_local (id, ... Get_local id ...)` with no prior write — today that reads a defined `0`; without the initializer it is C undefined behavior.

**Recommendations:**

1. **Derive `needs_init` from a syntactic read-before-first-write scan of the scope body** ("does `Get_local id` occur before the first definitely-executed `Set_local id`?"). The statement IR is branch-free and `For_loop` bounds are static ints, so the scan is *precise*: writes under a `For_loop` are guaranteed when `to_ >= from_`; reads count anywhere. Do not use `traced_array` at all.
2. **Prefer computing the scan at emission time in `c_syntax.ml`** (export a helper `Low_level.reads_scope_before_set : scope_id -> t -> bool`), rather than adding a field. This eliminates steps 1, 2, 4, 5, 6 of the Approach (no threading through ~20 pattern matches), and is correct *by construction* after every downstream body rewrite — notably `simplify_llc:1056`'s substitution, which merges `Set_local (id, 0.); Set_local (id, f id)` and is the main case where the saved initializer is not just a dead store.
3. **`Declare_local`:** compute the same scan at hoist time over the body being split off (`hoist_shared_locals` has it in hand, `low_level.ml:1558-1570`) and store the bool in `Declare_local`; or, acceptably, keep conservative zero-init for hoisted declarations — hoisted accumulator bodies start with `Set_local (id, 0.)`, so the decl-init is a dead store there anyway.
4. **Add a regression test for the known-virtual + recurrent corner** (explicitly mark a tn `Virtual`, accumulate with `=+` and no init). If that configuration is considered illegal, make `visit_llc` reject it loudly instead of papering over it with the decl initializer.
5. **Temper expectations in the goal/criteria:** since `inline_computation:706` rewrites `Zero_out` to `Set_local (id, 0.)`, most surviving `Local_scope` bodies self-initialize, and the dropped `= (float)0` is a dead store any C/CUDA/Metal compiler already eliminates. The real payoff is cleaner generated code; this is a cleanliness item, not a perf item, and the UB-safety analysis is the hard requirement.

**Open decision points for Łukasz:**
- Emission-time scan in `c_syntax.ml` (no IR change) vs. a `needs_init` field threaded through the passes. The review recommends the former; the field variant is acceptable if populated by the same body scan at creation *and* at `hoist_cross_statement_cse`.
- Whether `Declare_local` participates in the first iteration or conservatively keeps `= 0`.
- Whether "known-virtual tensor with recurrent accesses" is a supported configuration; if not, assert in `visit_llc` rather than relying on zero-init semantics.

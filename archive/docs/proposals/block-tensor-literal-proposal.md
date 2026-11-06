# Proposal: Block tensor literal syntax `[ta; tb]`, `(ta, tb)`, `[|ta; tb|]`

**Task**: task-fe1c593d
**Author**: Draft proposal (worker agent)
**Date**: 2026-03-25
**Status**: Draft
**Depends on**: gh-ocannl-49 (done), relates to gh-ocannl-421

## Summary

Add syntactic sugar to the `%op` PPX extension so that OCaml list, tuple, and array literal syntax containing tensor expressions desugars to `concat` calls, constructing block tensors by concatenating along a new leading axis. Lists `[ta; tb]` concatenate along a new output axis, tuples `(ta, tb)` along a new input axis, and arrays `[|ta; tb|]` along a new batch axis. This reuses the existing ndarray constant disambiguation: if all leaves are numeric literals, the existing `ndarray_op` path is used; if any leaf is a tensor expression, the new block tensor path is used.

---

## 1. Current State

### 1.1 Existing ndarray constant handling

In `ppx_op.ml` (lines 353-355), the `translate` function matches `Pexp_array` and `Pexp_construct ("::", ...)` (list syntax) and routes them to `ndarray_op`, which calls `ndarray_constant` in `ppx_shared.ml`. That function traverses nested list/tuple/array expressions, expecting numeric literal leaves, and produces shape dimensions plus a flat array of values.

### 1.2 Existing `++^` concatenation handling

In `ppx_op.ml` (lines 171-209), the `translate` function matches `(e1, e2, ...) ++^ "spec"` and generates `concat ?label spec [| e1'; e2'; ... |]`. The tuple elements are recursively translated, and the explicit spec string controls axis mapping.

### 1.3 Block tensor literal syntax (documented but unimplemented)

`syntax_extensions.md` (lines 552-566) documents the planned syntax where `[ta; tb]`, `(ta, tb)`, and `[|ta; tb|]` introduce a new leading axis and concatenate along it. The spec is auto-generated rather than user-supplied.

---

## 2. Proposed Design

### 2.1 Disambiguation strategy

Add a helper function `is_ndarray_constant_expr` that checks whether an expression is structurally an ndarray constant (nested list/tuple/array of numeric literals). The check is shallow: inspect the first leaf encountered by following the list/tuple/array backbone. If the first leaf is a float literal, integer literal, or a nested list/tuple/array (recurse), treat the whole expression as an ndarray constant. Otherwise treat it as a block tensor expression.

This is simpler and more robust than attempting `ndarray_constant` and catching failures, because `ndarray_constant` does not raise on non-constant leaves -- it passes them through as "expressions that compute a number."

```ocaml
(* In ppx_shared.ml or ppx_op.ml *)
let rec is_ndarray_constant_expr expr =
  match expr.pexp_desc with
  | Pexp_constant (Pconst_float _) | Pexp_constant (Pconst_integer _) -> true
  | Pexp_tuple (e :: _) | Pexp_array (e :: _) -> is_ndarray_constant_expr e
  | Pexp_construct ({ txt = Lident "::"; _ }, _) ->
      let elems = collect_list [] expr in
      (match elems with e :: _ -> is_ndarray_constant_expr e | [] -> true)
  | Pexp_prefix ("-", e) -> is_ndarray_constant_expr e  (* negative literals *)
  | _ -> false
```

### 2.2 New match cases in `translate`

Replace the existing ndarray constant match at lines 353-355 with expanded logic:

```ocaml
  (* List syntax: [e1; e2; ...] *)
  | { pexp_desc = Pexp_construct ({ txt = Lident "::"; _ }, _); _ } as list_expr ->
      if is_ndarray_constant_expr list_expr then
        (no_vbs, ndarray_op ?label ~ndarray_fn:[%expr TDSL.ndarray] list_expr)
      else
        let elems = collect_list [] list_expr in
        translate_block_tensor_list ~loc ~loop ~label ~opt_label `Output elems

  (* Array syntax: [|e1; e2; ...|] *)
  | { pexp_desc = Pexp_array elems; _ } ->
      if is_ndarray_constant_expr expr then
        (no_vbs, ndarray_op ?label ~ndarray_fn:[%expr TDSL.ndarray] expr)
      else
        translate_block_tensor_list ~loc ~loop ~label ~opt_label `Batch elems

  (* Tuple syntax: (e1, e2, ...) -- only for 2+ elements *)
  | { pexp_desc = Pexp_tuple elems; _ } when List.length elems >= 2 ->
      if is_ndarray_constant_expr expr then
        (no_vbs, ndarray_op ?label ~ndarray_fn:[%expr TDSL.ndarray] expr)
      else
        translate_block_tensor_list ~loc ~loop ~label ~opt_label `Input elems
```

The tuple case is placed **after** all `++^` patterns and operator application patterns (which also match tuples), so it only triggers for bare tuples containing tensor expressions.

### 2.3 Spec generation

The `translate_block_tensor_list` function generates a concat spec string and emits a `concat` call:

```ocaml
let translate_block_tensor_list ~loc ~loop ~label ~opt_label axis_kind elems =
  match elems with
  | [] ->
      (no_vbs,
       Ast_builder.Default.pexp_extension ~loc
       @@ Location.error_extensionf ~loc
            "ppx_ocannl %%op: block tensor requires at least one component")
  | _ ->
      let n = List.length elems in
      let vbss, translated = List.unzip (List.map elems ~f:loop) in
      let labels = List.mapi elems ~f:(fun i _ ->
        (* Use _bt0, _bt1, ... to avoid clashing with user axis labels *)
        "_bt" ^ Int.to_string i) in
      let lhs_parts = String.concat ~sep:"^" labels in
      let spec = match axis_kind with
        | `Output ->
            (* Each: ...|...->_btI, ... ; result: ...|...->_bt0^_bt1^..., ... *)
            String.concat ~sep:"; "
              (List.map labels ~f:(fun l -> "...|...-> " ^ l ^ ", ..."))
            ^ " => ...|...-> " ^ lhs_parts ^ ", ..."
        | `Input ->
            (* Each: ...|_btI, ...->... ; result: ...|_bt0^_bt1^..., ...->... *)
            String.concat ~sep:"; "
              (List.map labels ~f:(fun l -> "...| " ^ l ^ ", ...->..."))
            ^ " => ...| " ^ lhs_parts ^ ", ...->..."
        | `Batch ->
            (* Each: _btI, ...|...->... ; result: _bt0^_bt1^..., ...|...->... *)
            String.concat ~sep:"; "
              (List.map labels ~f:(fun l -> l ^ ", ...|...->..."))
            ^ " => " ^ lhs_parts ^ ", ...|...->..."
      in
      let spec_expr = substitute_identifiers_in_einsum_spec ~loc spec in
      let rhses_array = Ast_builder.Default.pexp_array ~loc translated in
      (reduce_vbss vbss,
       [%expr concat ?label:[%e opt_expr ~loc label] [%e spec_expr] [%e rhses_array]])
```

### 2.4 How nesting works

Nested block tensors `[[ta; tb]; [tc; td]]` work naturally because:
1. The outer list `[inner1; inner2]` is detected as a block tensor (first element is a list, not a numeric literal).
2. Each inner element `[ta; tb]` is recursively translated via `loop`, which itself calls `translate` and hits the block tensor list case.
3. The inner calls produce `concat` expressions (tensors), which become the operands of the outer `concat`.

Result: two nested `concat` calls, producing a tensor with two new leading output axes.

### 2.5 Single-element case

`[ta]` is valid and produces a concat with a single component, effectively an unsqueeze (adds a size-1 leading axis). The spec `"...|...-> _bt0, ... => ...|...-> _bt0, ..."` is an identity concat that adds the axis.

### 2.6 Mixed leaves

`[1.0; ta]` -- the first leaf is a float literal, so `is_ndarray_constant_expr` returns `true`, and the expression is treated as an ndarray constant. This matches the existing behavior: `ndarray_constant` will try to interpret `ta` as a number expression (which would produce a type error if `ta` is a tensor). This is acceptable because:
- If the user wants to include a scalar constant alongside tensor expressions, they can wrap it: `[!.1.0; ta]` (where `!.` creates a scalar tensor).
- The disambiguation is based on the first leaf, which is simple and predictable.

Alternative: check ALL leaves instead of just the first. This would make `[1.0; ta]` a block tensor (since `ta` is not a constant). However, this requires traversing the entire structure and could be confusing -- the user might intend an ndarray constant with a computed element.

**Recommendation**: Check only the first leaf (simple, fast, predictable). Document that mixing literal and tensor expressions requires wrapping the literals with `!.` to force tensor interpretation.

---

## 3. Implementation Plan

### Phase 1: `is_ndarray_constant_expr` helper

Add the helper function in `ppx_op.ml` (or `ppx_shared.ml` if reuse is desired). It traverses the list/tuple/array backbone to the first leaf and checks if it is a numeric literal.

**Files**: `tensor/ppx_op.ml`

### Phase 2: Replace ndarray constant match cases

Replace lines 353-355 in `ppx_op.ml` with the expanded disambiguation logic. Add the tuple case after existing operator patterns.

**Placement considerations**:
- The list case (`Pexp_construct ("::", ...)`) currently at line 354 must be replaced in-place.
- The array case (`Pexp_array`) currently at line 353 must be replaced in-place.
- The tuple case (`Pexp_tuple`) must be added near the end of the match, after all `++^` patterns (lines 171-258), record patterns (lines 259-352), and the `**.` pattern (lines 361-366), but before the generic `Pexp_apply` case (line 394). Actually, it should go after `Pexp_apply` too, since `Pexp_apply` handles function calls like `f (a, b)` where the tuple is an argument. Wait -- in OCaml's AST, `f (a, b)` has the tuple inside `Pexp_apply`'s argument list, not as a standalone `Pexp_tuple`. A bare `(a, b)` at expression level IS `Pexp_tuple`. So the tuple case can go after line 366 (after `**. `) and before `Pexp_apply` (line 394) -- actually no, `Pexp_apply` patterns at lines 367-382 specifically match `op (e2, e3)` forms where the tuple is part of the application. A bare tuple `(ta, tb)` without being applied to anything would not match `Pexp_apply` patterns. So placing the tuple case between the operator-application cases and the generic `Pexp_apply` case should be safe.

Actually, looking more carefully: the binary ops case at line 367-373 matches `op (e2, e3)` which is `Pexp_apply (op, [(Nolabel, Pexp_tuple [e2; e3])])` -- the tuple is inside the apply. A bare `(ta, tb)` would be `Pexp_tuple [ta; tb]` without an enclosing apply. So there is no conflict.

**Recommended placement**: Replace lines 353-355 with the list and array cases. Add the tuple case right after (before line 356).

### Phase 3: `translate_block_tensor_list` helper

Add the helper function that generates the spec string and emits the `concat` call. This function:
1. Validates non-empty element list
2. Recursively translates each element via `loop`
3. Generates axis labels `_bt0`, `_bt1`, ...
4. Builds the spec string based on axis kind
5. Calls `substitute_identifiers_in_einsum_spec` on the spec
6. Emits `[%expr concat ?label:... spec_expr rhses_array]`

**Files**: `tensor/ppx_op.ml`

### Phase 4: Testing

Add a new test file `test/operations/test_block_tensor.ml` (or extend `test_concat_graph.ml`) with:

1. **Basic list (output axis)**: `let%op stacked = [x1; x2]` -- stack two vectors
2. **Basic tuple (input axis)**: `let%op cat_input = (x1, x2)` -- concatenate along input axis
3. **Basic array (batch axis)**: `let%op cat_batch = [|x1; x2|]` -- concatenate along batch axis
4. **Nesting**: `let%op block = [[x1; x2]; [x3; x4]]` -- 2x2 block matrix
5. **Single element**: `let%op unsqueezed = [x1]` -- adds a size-1 axis
6. **Gradient flow**: verify gradients for all cases above
7. **Regression**: existing ndarray constants `[1.0; 2.0; 3.0]`, `(1.0, 2.0)`, `[|1.0; 2.0|]` still work

### Phase 5: Documentation

Update `syntax_extensions.md` lines 552-566:
- Change "Block tensor syntax (upcoming)" to "Block tensor syntax"
- Add concrete examples showing the desugared form
- Document the disambiguation rule (first-leaf check)
- Document the `!.` wrapping for mixing scalars with tensors

---

## 4. Spec String Details

### 4.1 Output axis (list `[ta; tb]`)

For 2 components:
```
"...|...-> _bt0, ...; ...|...-> _bt1, ... => ...|...-> _bt0^_bt1, ..."
```

Each component gets a new leading output axis of size matching its existing shape, and they are concatenated. The `...` broadcast handles all existing axes.

### 4.2 Input axis (tuple `(ta, tb)`)

For 2 components:
```
"...| _bt0, ...->...; ...| _bt1, ...->... => ...| _bt0^_bt1, ...->..."
```

### 4.3 Batch axis (array `[|ta; tb|]`)

For 2 components:
```
"_bt0, ...|...->...; _bt1, ...|...->... => _bt0^_bt1, ...|...->..."
```

### 4.4 Why `_bt` prefix

The labels `_bt0`, `_bt1`, etc. use a `_bt` prefix (for "block tensor") to avoid clashing with user-chosen axis labels like `a`, `b`, `x`, etc. The einsum parser supports multi-character labels, so this works.

### 4.5 Correctness argument

Each component tensor contributes one "slice" along the new axis. The `^` operator concatenates these slices. The broadcast `...` ensures all other axes are aligned. This is exactly equivalent to:
1. Unsqueezing each component to add a size-1 leading axis
2. Concatenating along that axis

The `concat` runtime function handles this in a single operation via the `Block` lowering path.

---

## 5. Edge Cases and Risks

### 5.1 Conflict with `Pexp_tuple` in operator applications

OCaml's AST represents `f (a, b)` as `Pexp_apply (f, [(Nolabel, Pexp_tuple [a; b])])`. The tuple is inside the apply node, not at the top level. A bare `(a, b)` at expression level is `Pexp_tuple [a; b]`. The `translate` function processes `Pexp_apply` by recursively calling `loop` on the arguments, so when it encounters the tuple argument inside an apply, it will hit the tuple case and potentially misinterpret it as a block tensor.

**Mitigation**: This is not a problem because:
- Binary operators at lines 367-382 match specific patterns with tuple arguments and handle them before the generic `Pexp_apply` case.
- The generic `Pexp_apply` case (line 394) calls `loop` on each argument individually. When it calls `loop arg_expr` on a tuple argument `(a, b)`, the recursive call WOULD hit the new tuple block tensor case.
- However, this would only happen for function calls like `my_fn (tensor1, tensor2)` which in OCaml is multi-argument application with a tuple. The user presumably intends the tuple as multiple arguments, not as a block tensor.

**Revised mitigation**: Only match `Pexp_tuple` at the **top level** of a `translate` call (i.e., when the tuple is the entire expression), not inside argument positions. This can be done by adding a `~is_top_of_expression` flag, OR by simply not matching `Pexp_tuple` in `loop` and only matching it in the `translate` wrapper.

Actually, wait. Looking at the code more carefully: the `loop` function IS `translate` (line 75: `let loop = translate ... ~is_toplevel:false`). Every recursive call goes through the same match. So if we add a `Pexp_tuple` case, it will match tuples inside function arguments too.

**Better mitigation**: Do NOT add a standalone `Pexp_tuple` match case. Instead, only handle tuples when they appear at the top level of a `%op` expression or as a standalone statement. This can be done by gating the tuple case on `is_toplevel`:

```ocaml
  | { pexp_desc = Pexp_tuple elems; _ } when is_toplevel && not (is_ndarray_constant_expr expr) ->
      translate_block_tensor_list ~loc ~loop ~label ~opt_label `Input elems
```

But `is_toplevel` is `false` for recursive calls, which means `((ta, tb), (tc, td))` would not work for the inner tuples. The outer tuple would match (if at top level), but inner elements `(ta, tb)` would not be translated as block tensors.

**Alternative approach**: Leave the tuple case out entirely for now. The user can always use the explicit `++^` syntax for input-axis concatenation: `(ta, tb) ++^ "...| _bt0, ...->...; ...| _bt1, ...->... => ...| _bt0^_bt1, ...->..."`. Lists and arrays are less ambiguous because:
- Lists: only used for ndarray constants or block tensors (no other OCaml construct uses list literals at expression level in a meaningful way inside `%op`).
- Arrays: same as lists.
- Tuples: used for multi-argument function calls, pairs, grouping, etc.

**Recommendation**: Implement list and array block tensor syntax. Defer tuple syntax due to ambiguity with multi-argument function calls. Document this decision. If tuple syntax is desired later, it can be added with a disambiguating marker (e.g., a type annotation or attribute).

### 5.2 Empty containers

`[]`, `()`, `[||]` should produce clear PPX error messages.

### 5.3 Interaction with `Pexp_tuple` for single-element

OCaml does not have 1-tuples at the syntax level (parenthesized expression is just the expression). So `(ta)` is just `ta` -- no tuple node in the AST. This means single-element input-axis block tensors are impossible with tuple syntax, which is fine.

### 5.4 Three-way+ concatenation backprop

The gh-ocannl-49 bug fix (now completed) ensures 3-way concatenation works correctly. Block tensor syntax for 3+ components is safe.

---

## 6. Revised Scope

Given the tuple ambiguity analysis in section 5.1, the recommended scope for this task is:

1. **List syntax `[ta; tb; ...]`** -- output axis block tensor -- IMPLEMENT
2. **Array syntax `[|ta; tb; ...|]`** -- batch axis block tensor -- IMPLEMENT
3. **Tuple syntax `(ta, tb, ...)`** -- input axis block tensor -- DEFER (document the ambiguity and recommend `++^` with explicit spec for input-axis concatenation)

This reduces risk while delivering the most useful forms. Lists and arrays have no ambiguity in `%op` blocks because the only other use of list/array literals is ndarray constants, which is cleanly disambiguated.

---

## 7. Files Changed

| File | Change |
|------|--------|
| `tensor/ppx_op.ml` | Add `is_ndarray_constant_expr`, `translate_block_tensor_list`, replace lines 353-355 |
| `tensor/ppx_shared.ml` | Possibly move `is_ndarray_constant_expr` here if useful for `ppx_cd.ml` |
| `docs/syntax_extensions.md` | Update lines 552-566 to reflect implemented syntax |
| `test/operations/test_block_tensor.ml` | New test file for block tensor syntax |
| `test/operations/dune` | Add new test executable |

---

## 8. Estimated Effort

Small-medium (2-3 days)

- Day 1: Implement `is_ndarray_constant_expr` and `translate_block_tensor_list` in `ppx_op.ml`. Replace the ndarray constant match cases. (~1 day)
- Day 2: Write tests for list and array block tensor syntax, including nesting and gradients. Verify no ndarray constant regression. (~1 day)
- Day 3: Update documentation. Handle edge cases (empty, single-element). (~0.5 day)

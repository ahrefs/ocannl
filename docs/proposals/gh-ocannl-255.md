# Proposal: Audit and Strengthen Dimension Label Checking and Inference

**Issue:** [#255](https://github.com/ahrefs/ocannl/issues/255)
**Milestone:** v0.6.4
**Related:** [#298](https://github.com/ahrefs/ocannl/issues/298) (rename label -> basis)

## Summary

Dimension labels (optional string annotations on tensor axes, e.g. "batch", "features") are intended
to catch semantic mistakes like accidental transposition. The current implementation has several gaps
where labels are silently dropped or not checked, reducing their protective value. This proposal
covers an audit of all label-handling code paths and targeted fixes, plus dedicated test coverage.

## Audit Scope and Key Code Locations

All code is in `tensor/row.ml` unless otherwise noted.

### 1. Label checking in `unify_dim` (line 1596-1603)

Labels are only checked when **both** dims are solved (`Dim`-`Dim`). The `Var v, dim2` fallback
case (line 1827) stores the solved dim via `Solved_dim dim2` but does **not** check whether a
previously-associated label on the variable conflicts with the label on `dim2`. This means:

```
Var{name="x"} unifies with Dim{d=4; label=Some "batch"} -> OK, stores solved
Var{name="x"} unifies with Dim{d=4; label=Some "features"} -> second call sees Dim-Dim, caught
```

The indirect path works because Var is immediately substituted. However, if a Var is unified with
an **unlabeled** Dim first, then later a labeled Dim with the same size, the label is lost:

```
Var unifies with Dim{d=4; label=None} -> stores Solved_dim(Dim{d=4; label=None})
Later: Dim{d=4; label=None} vs Dim{d=4; label=Some "batch"} -> passes (sizes match, line 1603)
```

**Gap confirmed:** Line 1603 (`when d1 = d2 -> ([], env)`) short-circuits on matching sizes
without checking labels. Two solved dims with the same size but different labels (one `None`,
one `Some`) will not raise an error.

### 2. LUB (least upper bound) computation (line 2360-2367, 2783-2792)

Two separate LUB paths handle labels:

- **`dim_comparison_for_axis`** (line 2360): When two solved dims have the same size but different
  labels, falls through to the `Dim _, Dim _` branch (line 2363) which forces `d=1` as the LUB.
  This is a silent demotion rather than an error -- the conflicting label dims get collapsed to
  size 1, potentially causing subtle shape mismatches later.

- **Row-level LUB** (line 2783-2792): Conflicting labels produce `get_dim ~d:1 ~proj_id:63 ()`,
  silently discarding both labels and demoting to size 1.

### 3. Concat label handling (line 422-436)

When all concat components are solved, `List.find_map solved_dims ~f:(fun s -> s.label)` (line 434)
picks the **first non-None label** arbitrarily. If components have different labels, this is
silently ignored.

Similarly, in convolution/affine substitution (lines 408, 418), `Option.first_some s.label k.label`
takes the first available label without checking for conflicts.

### 4. Dimension equality (line 76-80)

`equal_dim` for solved dims compares both `d` and `label` (ignoring `proj_id`). This is correct --
it means equal dims require matching labels. But the `unify_dim` function's fast path at line 1603
only checks `d1 = d2`, bypassing label comparison.

### 5. `dim_comparison_for_axis` broadcast case (line 2097-2098)

Broadcasting with `Dim{d=1; label=None}` is allowed freely (line 2097). A `Dim{d=1; label=Some _}`
would NOT match this pattern and would flow to the Var case. This is actually correct behavior --
labeled singleton dims are semantically meaningful.

### 6. Einsum / shape spec layer (`tensor/shape.ml`, `tensor/einsum_types.ml`)

Einsum axis labels (e.g. "b" in `"b i -> b o"`) are a **different concept** from dimension labels.
Einsum labels become `dim_var` names, not `solved_dim.label` values. Dimension labels enter the
system through `get_dim ~d ~label ()` calls at shape.ml lines 2265-2274, and through `~axis_label`
on `Tensor.number`/`Tensor.bits` (tensor.ml lines 575-599) which flows to `~output_axes`.

### 7. No existing test coverage

Searched all files in `test/` -- the word "label" appears only in contexts of debug/display labels,
tensor names, or print styles. There are **zero tests** for dimension label checking, propagation,
or conflict detection.

## Proposed Changes

### Fix 1: Check labels in `unify_dim` solved-dim fast path (line 1603)

Change from:
```ocaml
| Dim { d = d1; _ }, Dim { d = d2; _ } when d1 = d2 -> ([], env)
```
To:
```ocaml
| Dim { d = d1; label = l1; _ }, Dim { d = d2; label = l2; _ }
  when d1 = d2 && Option.value ~default:true (Option.map2 ~f:String.equal l1 l2) ->
  (* Propagate label if one side has it and the other doesn't *)
  ([], env)
```

This matches the pattern already used at line 2361 and ensures that `Dim{d=4; label=None}` vs
`Dim{d=4; label=Some "batch"}` is accepted (label propagates from the labeled side), while
`Dim{d=4; label=Some "batch"}` vs `Dim{d=4; label=Some "features"}` is caught by the existing
check at line 1599.

**Risk:** Low. The current behavior already passes through -- this just adds the guard to
prevent falling through to the wrong branch in edge cases.

### Fix 2: Warn or error on LUB label conflicts

In the LUB computation (lines 2360-2367 and 2783-2792), the silent demotion to `d=1` when labels
conflict is intentional broadcast semantics -- it means "these axes are incompatible, so collapse
to size 1." This is actually correct behavior for broadcasting. **No change needed**, but add a
comment documenting this design choice.

### Fix 3: Check concat component labels (line 434)

Replace the arbitrary first-label selection with a consistency check:
```ocaml
let labels = List.filter_map solved_dims ~f:(fun s -> s.label) in
let label = List.hd labels in
(* Check all labels are consistent *)
if not (List.for_all labels ~f:(fun l ->
    Option.value ~default:true (Option.map label ~f:(String.equal l))))
then
  raise @@ Shape_error
    ("concat: conflicting dimension labels", [Dim_mismatch (List.map solved_dims ~f:(fun s -> Dim s))]);
Dim { d = total_d; label; proj_id = None }
```

**Risk:** Medium. If any existing code concatenates differently-labeled dims, this will break.
Need to audit existing usage first. Can start with a warning instead of an error.

### Fix 4: Check convolution label conflicts (lines 408, 418)

Replace `Option.first_some s.label k.label` with a check:
```ocaml
let label = match s.label, k.label with
  | Some l1, Some l2 when not (String.equal l1 l2) ->
    raise @@ Shape_error ("convolution: conflicting labels between stride and kernel", ...)
  | l1, l2 -> Option.first_some l1 l2
in
```

**Risk:** Low. Kernel dims are rarely labeled in practice.

### Fix 5: Global label-size registry

**Recommendation: Defer.** A global registry adds complexity and makes the system less flexible
(e.g., dynamic batch sizes). The unification-time checking proposed above is sufficient for
catching semantic errors. File as a separate issue if desired.

## Testing Strategy

Create a new test file `test/einsum/test_dimension_labels.ml` with cases:

1. **Same label, same size**: Two dims labeled "batch" with size 32 unify successfully
2. **Different labels, same size**: "batch" vs "features" with same size raises `Shape_error`
3. **One labeled, one unlabeled, same size**: Label propagates (no error)
4. **Label propagation through variable**: Var unified with labeled dim, then checked against another labeled dim
5. **Concat with conflicting labels**: Raises error (after Fix 3)
6. **Concat with consistent labels**: Label preserved
7. **LUB with conflicting labels**: Produces d=1 (document this is expected)
8. **Broadcast with labeled singleton**: Labeled d=1 is preserved
9. **Affine/convolution label propagation**: Label survives stride computation

Tests should use the `Row` module directly (unit-test style) rather than going through the full
tensor API, for isolation. A few integration tests through `Tensor.term` with `~output_axes`
would also be valuable.

## Sequencing

1. Add tests first (expect current behavior, documenting gaps)
2. Apply Fix 1 (unify_dim label check) -- update expected test outputs
3. Apply Fix 4 (convolution label check) -- low risk
4. Apply Fix 3 (concat label check) -- run full test suite, audit breakage
5. Add LUB documentation comments (Fix 2)
6. Coordinate with #298 (basis rename) -- these changes should land first

## Risk Assessment

- **Breaking existing code:** The main risk is Fix 3 (concat labels). Existing code that
  concatenates differently-labeled dims would break. Mitigation: search for `Concat` usage
  in the codebase, start with warnings.
- **Performance:** Adding `Option.map2` to the unify_dim fast path adds negligible cost
  (one option comparison per dim pair).
- **Interaction with #298:** The basis rename touches the same `label` fields. This proposal
  should merge first; #298 is a mechanical rename afterward.

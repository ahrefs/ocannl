# Proposal: Elide Unneeded Local_scope Zero Initialization

**Issue:** https://github.com/ahrefs/ocannl/issues/340

## Status

Verified on 2026-06-19 against local HEAD `00f8ae3f` and the live GitHub issue.

- Issue #340 is still open and assigned to milestone `v0.7`.
- `ROADMAP.md` lists `Local_scope` initialization tracking under the still-open v0.7 compiler
  optimization work.
- `Local_scope` still carries only `id`, `body`, and `orig_indices` in
  `arrayjit/lib/low_level.ml` and `arrayjit/lib/low_level.mli`.
- `Declare_local of scope_id` exists and is used by cross-statement CSE hoisting.
- `arrayjit/lib/c_syntax.ml` still emits unconditional `= 0` initializers for both
  `Declare_local` and scalar `Local_scope` declarations, and the `TODO(#340)` is still present.

## Summary

Generated C currently initializes every scalar local produced by `Local_scope`:

```c
float v42 = (float)0;
```

That initializer is only semantically required when the local can be read before its first write, as
in an accumulator:

```c
v42 = v42 + delta;
```

Most `Local_scope` bodies assign the local before reading it, often because `Zero_out` was already
rewritten to an explicit `Set_local (id, 0.)` inside the body. In those cases the declaration
initializer is dead clutter:

```c
float v42;
v42 = (float)0;
...
```

Cross-statement CSE can also hoist a `Local_scope` body into a standalone `Declare_local` followed
by the body. That declaration has the same initialization problem, so this proposal treats
`Local_scope` and `Declare_local` as one codegen surface: both must use the same body-derived
decision.

The original issue suggested adding a `needs_init` field to `Local_scope` and populating it from
`traced_array.read_before_write`. That data source is not sound for current code. The better fix is
to derive the answer from the final body itself: emit `= 0` only if a syntactic scan of the body
finds `Get_local id` before the first definitely executed `Set_local id`.

## Decision

Use a body scan, not `traced_array.read_before_write`, as the source of truth.

The scan should answer:

> Does this `scope_id` have a possible read before the first definitely executed write?

For the current low-level statement IR, this can be precise enough to be used directly at codegen:

- `Set_local (id, value)` first scans `value` for `Get_local id`; then, if the assignment target is
  `id`, marks the local as written.
- `Get_local id` before that mark means the declaration initializer is required.
- `Seq` scans left to right.
- `For_loop` scans the body. A write inside the loop is definitely executed only when the loop runs
  at least once (`to_ >= from_` under the current integer-bound convention); reads inside the loop
  still count.
- `Set`, `Set_from_vec`, scalar expressions, nested `Local_scope`, and nested op arguments are
  scanned for scalar reads. Nested `Local_scope` binders should not be confused with the target id.
- `Declare_local`, `Zero_out`, `Comment`, `Staged_compilation`, and `Noop` do not write the target
  local.

This avoids threading a new boolean through many `Local_scope` pattern matches and remains correct
after downstream rewrites such as simplification and common-subexpression elimination. The one place
that must store the result is `Declare_local`, because hoisting splits the declaration from the body
that justifies it.

## Why read_before_write Is the Wrong Signal

`visit_llc` sets `traced_array.read_before_write` only for recurrent accesses when the tensor node
is not already known virtual. The same branch also updates the tensor memory mode to `Materialized`.
Materialized nodes are known non-virtual, which prevents `Local_scope` creation for that tensor.

That means `traced.read_before_write` is false at the normal `Local_scope` creation site for exactly
the class of locals this optimization wants to inspect. Using it would silently turn into "never
initialize local scopes", which is unsafe for supported or accidentally reachable recurrent virtual
cases.

## Implementation Plan

1. Add a helper in `arrayjit/lib/low_level.ml`, exported from `low_level.mli`:

   ```ocaml
   val reads_scope_before_set : scope_id -> t -> bool
   ```

   The helper should be pure and operate on the final `Low_level.t` body. It should scan statements
   and scalar expressions structurally, using `Scope_id.equal` for target-local identity.

2. In `arrayjit/lib/c_syntax.ml`, update the `Local_scope` declaration emission:

   ```ocaml
   let init_zero =
     if Low_level.reads_scope_before_set id body then
       let prefix, postfix = B.convert_precision ~from:Ops.int32 ~to_:scope_prec in
       string " = " ^^ string prefix ^^ string "0" ^^ string postfix
     else empty
   in
   ```

   Remove the `TODO(#340)` comment.

3. Extend `Declare_local` so hoisted locals use the same initialization decision:

   ```ocaml
   | Declare_local of { id : scope_id; needs_init : bool }
   ```

   In `hoist_shared_locals`, compute `needs_init` from the hoisted body before recording the
   insertion:

   ```ocaml
   let needs_init = reads_scope_before_set canonical_id body in
   insertions := (first_user, [ Declare_local { id = canonical_id; needs_init }; body ])
                 :: !insertions
   ```

   Then update `c_syntax.ml` so `Declare_local` uses the same conditional initializer as
   `Local_scope`.

4. Update every `Declare_local` pattern match and constructor site in `low_level.ml`,
   `low_level.mli`, and `c_syntax.ml`.

   Existing read-only traversals can usually switch from `Declare_local id` to
   `Declare_local { id; _ }`. Any equality, hashing, debug-printing, and pretty-printing paths must
   include or intentionally ignore `needs_init` according to their existing role. Code generation
   must honor it.

5. Do not add `needs_init` to `Local_scope`.

   A field would need to be maintained across `inline_computation`, `virtual_llc`,
   `cleanup_virtual_llc`, `simplify_llc`, `eliminate_common_subexpressions`, and
   `hoist_cross_statement_cse`. Computing from the final body is simpler and less fragile. For the
   hoisted case, the field belongs on `Declare_local` because the body has already been split out.

## Acceptance Criteria

- Non-recurrent `Local_scope` declarations in generated C omit the `= 0` initializer.
- Recurrent `Local_scope` declarations keep the `= 0` initializer.
- Non-recurrent hoisted `Declare_local` declarations omit the `= 0` initializer.
- Recurrent hoisted `Declare_local` declarations keep the `= 0` initializer.
- The `TODO(#340)` in `arrayjit/lib/c_syntax.ml` is removed.
- `Declare_local` carries and honors its own initialization decision.
- The helper is covered by focused tests for:
  - write-before-read: no initializer needed;
  - read-before-write: initializer needed;
  - `Set_local (id, Binop (... Get_local id ...))`: initializer needed;
  - leading `Set_local (id, Constant 0.)` before later reads: no declaration initializer needed;
  - loop body reads and writes, including an empty-loop case if the current `For_loop` bound
    convention permits one.
- Existing tests pass with `OCANNL_BACKEND=sync_cc dune runtest`.

## Suggested Regression Test

Add focused tests under `test/` or `arrayjit/test/` that build small low-level computations and
print the generated C for both inline and hoisted cases:

1. A local whose body starts with `Set_local (id, Constant 0.)`, followed by later reads. The
   emitted declaration should be `float v;`, not `float v = (float)0;`.
2. A local whose first assignment reads itself, such as `Set_local (id, Binop (Add, Get_local id,
   Constant 1.))`. The emitted declaration must keep `= (float)0`.
3. A duplicated `Local_scope` expression that cross-statement CSE hoists into `Declare_local; body`,
   with a write-before-read body. The hoisted declaration should omit `= 0`.
4. The same hoisted shape with a read-before-write body. The hoisted declaration must keep `= 0`.

If there is already a generated-code expectation test around local scopes, extend that instead of
adding a new harness.

## Non-Goals

- This proposal does not change array-level `Zero_out` elision; that work is tracked separately and
  was largely handled under #420.
- This proposal does not alter tensor memory-mode inference.
- This proposal does not try to prove performance impact. The main payoff is cleaner generated code
  and preserving defined C semantics only where the initializer is actually needed.

## Validation Notes

Current local code supports the body-scan design:

- `Local_scope` creation happens after tracing and constructs a body that can already contain
  `Set_local` and `Get_local`.
- Simplification can rewrite local-scope bodies, so a codegen-time body scan sees the post-rewrite
  truth.
- Cross-statement CSE hoisting extracts `Local_scope` bodies into `Declare_local; body`, which is
  why `Declare_local` needs its own explicit `needs_init` payload.
- CUDA and Metal backends do not directly emit `Local_scope`; this proposal is scoped to the shared
  C syntax path used by the compiled backends.

# Widen dynamic gather index cast to `Ops.index_prec ()` for very large table axes

## Goal

The generated C for the guarded dynamic gather (gh-ocannl-343's one-hot →
embedding-lookup rewrite) casts the runtime table index with a hardcoded
`((int)(...))`. `int` is 32-bit, so a table/vocabulary index exceeding `INT_MAX`
is truncated before it reaches the row-major offset arithmetic. For very large
table/vocab axes this silently reads the wrong row.

This is the explicit non-blocking follow-up the reviewer raised in gh-ocannl-343
review round 2 (and the coder acknowledged): "the `((int)(...))` cast is narrower
than `Ops.index_prec ()` — follow-up for very large table axes." Widen the cast so
the gather index tracks the same width as the rest of the project's index
arithmetic.

Relates to gh-ocannl-343. Distinct from sibling tasks task-73617488 (one-hot
virtualizer exemption) and task-76458e75 (merged) — do not conflate.

## Acceptance Criteria

1. **Cast widened to the index precision's C type.** The `Get_dynamic` load arm of
   `pp_scalar` in `arrayjit/lib/c_syntax.ml` renders the dynamic index cast to
   `B.typ_of_prec (Ops.index_prec ())` (i.e. `uint64_t` under
   `Utils.settings.large_models`, else `uint32_t`) instead of the hardcoded
   `((int)(...))`. The cast width therefore auto-follows the `large_models` switch,
   matching `loop_index_type` / `arg_int_prefix` already used for loop counters and
   buffer-offset arithmetic in the same module.

2. **OOB guard semantics untouched.** `build_guarded_gather` in
   `arrayjit/lib/low_level.ml` is not modified. Its signed-comparison guard
   (`guard_prec = Ops.double`; `lower = Cmplt(-1, iv)`, `upper = Cmplt(iv,
   class_count)`, `is_integral = Cmpeq(iv, Trunc iv)`) keeps its current precision
   and structure. The widened cast must not re-narrow or otherwise alter the guard's
   signedness. Verifiable: a diff that touches only the codegen render site (and the
   test), not the guard builder.

3. **Index value carries full precision end-to-end (sub-item 2).** Confirm that
   widening the *outer* cast is sufficient — i.e. that the index value `iv` is not
   already truncated upstream before the cast. `dyn_expr` is `pp_scalar iprec iv`,
   and `iprec` is whatever precision the matched index expression carried in the
   source IR (`match_one_hot_contribution` returns the `Cmpeq` operand's precision
   verbatim; it is not forced to `Ops.index_prec ()`). The change must include
   evidence — a test exercising a table whose relevant index exceeds the 32-bit
   boundary (or, where running that is impractical, a generated-C inspection) — that
   a `> 2^31` index survives correctly through the gather (and, where `large_models`
   makes `index_prec` 64-bit, that a `> 2^63`-relevant path is not re-narrowed). If
   `iprec` itself is found to be narrow on the large-vocab path, the fix must widen
   that too so the index is not truncated before the cast; if `iprec` is already
   wide enough, this AC is satisfied by the inspection alone, recorded in the task
   Notes / test.

4. **Brittle test assertion updated (sub-item 1).** The structural assertion in
   `test/operations/test_one_hot_embedding_lookup.ml` that currently checks
   `String.is_substring c ~substring:"((int)("` is updated so it passes against the
   widened cast while preserving its intent (a guarded dynamic table read is present:
   it pairs the cast substring with the ternary `"?"` and asserts the vocabulary
   reduction loop is gone). Either update the literal to the new cast text
   (`((uint32_t)(` under default settings) or make the assertion match the
   dynamic-read shape more structurally; the surrounding explanatory comment (and the
   comment at the `debug_float` `Get_dynamic` arm in `c_syntax.ml` that mentions
   `table[(int)idx]`) is kept accurate.

5. **All three backends covered with no per-backend change.** The fix lives in the
   shared `pp_scalar` functor body in `c_syntax.ml`; CC, Metal, and CUDA inherit it
   (none override `Get_dynamic`). No backend-specific code is added.

6. **Existing gh-ocannl-343 behaviour preserved.** The one-hot → embedding-lookup
   rewrite, the OOB-index-yields-zero-row behaviour, and the vocabulary-loop
   elimination continue to pass (`test_one_hot_embedding_lookup` green on the C
   backend).

## Context

How the dynamic gather is rendered today:

- **`arrayjit/lib/c_syntax.ml`**, `pp_scalar`, `Get_dynamic { tn; idcs; dyn_axis;
  dyn_value = iv, iprec }` arm. It computes `dyn_expr = pp_scalar iprec iv`, then:

  ```ocaml
  let dyn_idx_doc = string "((int)(" ^^ dyn_expr ^^ string "))" in
  ```

  and hands `dyn_idx_doc` to `pp_array_offset_dyn` to splice into the row-major
  offset at `dyn_axis`. `iprec` is in scope but the cast ignores it — `int` is
  hardcoded. `pp_array_offset_dyn` consumes the doc unchanged and needs no edit.

- The functor exposes `typ_of_prec = Ops.c_typ_of_prec` (field `B.typ_of_prec`),
  plus `loop_index_type` / `arg_int_prefix` which already select `uint64_t`
  vs `uint32_t` by `Utils.settings.large_models`. `Ops.index_prec ()` returns
  `uint64` under `large_models`, else `uint32` (`arrayjit/lib/ops.ml`).
  `Ops.c_typ_of_prec` maps `Uint64_prec -> "uint64_t"`, `Uint32_prec ->
  "uint32_t"`, etc. So `B.typ_of_prec (Ops.index_prec ())` yields the matching C
  type name for the cast.

- **`arrayjit/lib/low_level.ml`**, `build_guarded_gather`, wraps the gather in
  `Ternop (Where, (in_range, _), (gather, value_prec), (Constant 0., value_prec))`.
  The whole guard is built in `Ops.double` precisely so `-1 < idx` does not wrap (the
  `[[unsigned-index-precision]]` learning). Because the C ternary short-circuits, the
  truncating cast only ever runs on an index the guard already proved is in
  `[0, class_count)` and integral — so casting through an *unsigned* `index_prec`
  type is safe here even though the guard itself must stay signed. **The guard and
  the cast are in different files, different IRs, and the cast runs strictly after
  the guard admits the value; widening the cast cannot touch the guard.** This task
  does not modify `build_guarded_gather`.

- **`arrayjit/lib/low_level.ml`**, `match_one_hot_contribution`, returns the matched
  index expression *with its source precision* (`match_cmpeq` returns `(b, pb)` /
  `(a, pa)` verbatim). That precision becomes `iprec` on the `Get_dynamic`'s
  `dyn_value`. It is not coerced to `index_prec ()`, which is exactly why AC 3
  requires verifying the value is not narrow before the outer cast.

- **`test/operations/test_one_hot_embedding_lookup.ml`** asserts (against generated
  C read from `build_files/embedded_fwd.c`) that the dynamic read contains
  `((int)(` AND a ternary `"?"`, and that no `<= 3` vocabulary loop bound survives.
  The `((int)(` substring will no longer appear once the cast widens — this is the
  brittle assertion AC 4 updates. The `.expected` companion inspects the C file but
  does not hardcode `(int)` in a way that breaks (verify during implementation).

## Approach

*Suggested approach — agents may deviate if they find a better path.*

In `pp_scalar`'s `Get_dynamic` arm in `c_syntax.ml`, replace the hardcoded literal:

```ocaml
let dyn_idx_doc = string "((int)(" ^^ dyn_expr ^^ string "))" in
```

with a cast to the index precision's C type, e.g.:

```ocaml
let idx_typ = B.typ_of_prec (Ops.index_prec ()) in
let dyn_idx_doc = string ("((" ^ idx_typ ^ ")(") ^^ dyn_expr ^^ string "))" in
```

Then for AC 3, inspect how `iprec` is set for the large-vocab path
(`match_one_hot_contribution` → `Get_dynamic.dyn_value`). If it already tracks a
wide enough precision (it should, under `large_models`), record that and add the
end-to-end test/inspection. If it is narrow on the relevant path, widen the index
value precision at its source so the value is not truncated before the cast.

Finally, update the `test_one_hot_embedding_lookup.ml` assertion to match the new
cast rendering (or to a more structural dynamic-read check) and refresh the
explanatory comments at both `Get_dynamic` arms in `c_syntax.ml`.

## Scope

**In scope:** the single render-site change in `c_syntax.ml`; the upstream `iprec`
end-to-end verification (and widening only if found narrow); the test assertion
update; comment touch-ups. **Out of scope:** any change to `build_guarded_gather` or
the guard's `Ops.double` signed-comparison semantics; per-backend overrides; the
sibling tasks task-73617488 and task-76458e75.

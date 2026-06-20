# Stage B unit-solve guard: debug-safe `Where` value logging

Related task: `task-9658aac9` (surfaced by the retrospective of `task-4bf2df02`,
gh-ocannl-133 Stage B).

Parent context: [gh-ocannl-133-stage-b.md](gh-ocannl-133-stage-b.md)

## Goal

Stage B's unit-coefficient solving inlines an injective affine producer by emitting a range
guard as a `Where` ternary whose **then**-branch is the producer read at the unit-solved index:

```text
Where (range_cond, Get (producer, solved_idx), Get_local id)
```

For a non-matching kept-loop iteration, `solved_idx` can derive an out-of-bounds flat offset.
At normal execution this is harmless: the backend lowers `Where` to a C/CUDA ternary
`cond ? then : else`, which **short-circuits**, so `producer[solved_idx]` is dereferenced only
when `range_cond` holds. The one path that actually dereferences out of bounds today is
**debug value-logging**: `c_syntax.ml`'s `debug_float` for `Ternop Where` collects *all three*
branch values as `printf` arguments evaluated **unconditionally**. With ppx_minidebug value
logging enabled, the then-branch's `Get (producer, oob_idx)` becomes an unconditional
out-of-bounds read — a genuine, reproducible fault, not a theoretical one.

This is the exact hazard class gh-343 already recognized and fixed for the sibling `Get_dynamic`
one-hot gather construct: see the comment at `debug_float`'s `Get_dynamic` arm in
`c_syntax.ml` ("a `Where`'s `debug_float` collects all three branch values as printf arguments
evaluated unconditionally, so returning the raw `table[...]` access here would read out of
bounds for ids the surrounding guard is meant to exclude").

Close the only currently-reachable out-of-bounds path by making `debug_float` not dereference a
`Where`'s conditionally-evaluated branches out of bounds, mirroring the gh-343 remedy. Defer the
full structural soundness fix (index clamping or an IR branch so the discarded read is never
constructed) until an eager/predicated backend that materializes both `Where` arms actually
lands.

## Acceptance Criteria

1. **No unconditional out-of-bounds producer read in debug value-logging of a unit-solve
   `Where` guard.** When ppx_minidebug value logging is enabled, the `debug_float` rendering of
   a Stage B unit-solve `Where (range_cond, Get (producer, solved_idx), Get_local id)` does not
   dereference `producer` at `solved_idx` unconditionally — the dereference is either gated by
   the guard condition (short-circuited, as the runtime ternary is) or replaced by an
   always-safe surrogate (e.g. logging the condition and/or index instead of the raw read),
   consistent with the gh-343 `Get_dynamic` precedent.

2. **Matching iterations' logged values are unchanged.** For iterations where the guard holds,
   the value reported by debug logging is still the producer read (no loss of debug fidelity for
   the in-range case). Whatever the fix prints for the off-range branch must not corrupt or
   misreport the in-range value.

3. **Runtime numerics and IR unchanged.** The fix touches only the debug-logging rendering path
   (`c_syntax.ml` `debug_float`). It does **not** change `try_unit_solve` / `range_guards` index
   derivation in `low_level.ml`, the emitted guard IR (`inline_computation`'s `Where` over
   `range_conds`), or backend `Where` lowering (`cuda_backend.ml` / `metal_backend.ml`
   `ternop_syntax`). Existing non-debug execution parity (`affine_lowering.ml::ac6`,
   `virtual_affine.ml`) continues to pass byte-for-byte.

4. **Regression coverage pins the previously-hazardous shape under debug logging.** A test
   constructs (or reuses) a Stage B unit-solve scatter→copy whose virtualized form emits the
   `Where` range guard, enables debug value-logging, and asserts the hazard is closed. Per the
   gh-343 precedent (commit `18964416`, which read the generated C from the build directory and
   asserted on the guarded-gather text), the assertion may be over the generated debug-logging C
   (the `Where` branch reads are not emitted as unconditional `printf` accessor arguments) or
   over the emitted IR / `debug_float` document, whichever is more robust — the runtime ternary
   short-circuits regardless, so a structural/text assertion is preferred over trying to fault at
   runtime. Parity assertions belong with `affine_lowering.ml::ac6`; structural assertions belong
   in `virtual_affine.ml`, matching where the task scope places them.

5. **Deferred-soundness comment at the guard site.** A brief comment is added at the unit-solve
   guard emission site in `low_level.ml` (the `Set`/`Set_local` arm of `inline_computation`'s
   inner `loop`, where `guarded` is folded as
   `Ternop (Ops.Where, (cond, index_prec), (acc, value_prec), (Get_local id, value_prec))`)
   documenting: (a) the then-branch producer read is evaluated structurally at the unit-solved
   index even for non-matching kept-loop iterations; (b) the runtime `Where` ternary
   short-circuits, so arithmetic execution does not fault; (c) the debug-logging path is made safe
   by this change; (d) full structural soundness (index clamping / an IR branch) is deferred until
   an eager/predicated backend that evaluates both `Where` arms lands.

### AC verification reachability

All named paths (`arrayjit/lib/c_syntax.ml`, `arrayjit/lib/low_level.ml`,
`test/operations/affine_lowering.ml`, `test/operations/virtual_affine.ml`) live inside
`git -C ~/ocannl-staging` and are reachable via ordinary find/grep and post-merge commit SHAs;
no out-of-git-context path is involved.

## Context

How it works now:

- **Guard emission** — `arrayjit/lib/low_level.ml`, `inline_computation`. `try_unit_solve`
  binds the single ±1-coefficient unbound symbol and pushes
  `(uc, rest_axis, rhs_axis, range)` onto `range_guards`. In the `Set`/`Set_local` arm of the
  inner `loop`, the producer read is computed unconditionally (`inlined = loop_scalar env llsc`),
  then wrapped: `conds` (from `range_guards` via the `lt`/`add_offset` helpers, all in
  `index_prec`) are folded into
  `Ternop (Ops.Where, (cond, index_prec), (acc, value_prec), (Get_local id, value_prec))`.
  The then-arm `acc` carries the producer `Get` at the unit-solved index. This is the comment
  site for AC 5.

- **Runtime lowering (short-circuits, safe)** — `cuda_backend.ml` `ternop_syntax`:
  `Ops.Where, _ -> cond ? v2 : v3`; `metal_backend.ml` `Ops.Where` (select);
  `c_syntax.ml` `pp_scalar`'s `Ops.Where` arm. These are out of scope (AC 3).

- **Debug value-logging (the hazard)** — `c_syntax.ml` `debug_float`, the
  `Ternop (op, (v1, v1_prec), (v2, v2_prec), (v3, v3_prec))` arm. For `Ops.Where` it recurses
  `debug_float prec v2` (then) and `debug_float prec v3` (else) and concatenates their `printf`
  accessor argument lists, so both branch reads are emitted as unconditional arguments. The
  then-branch's `Get (producer, oob_idx)` falls through to the `Get (tn, idcs)` leaf, which emits
  an unconditional `producer[offset]` access. This is the fix site.

- **The gh-343 precedent (the template for the fix)** — in the same `debug_float`, the
  `Get_dynamic` leaf carries the gh-343 comment and logs the always-safe dynamic *index* value
  (`get_ident tn ^ "@dyn_idx"`) instead of the table read, precisely because the enclosing
  `Where`'s `debug_float` evaluates all branches unconditionally. The difference here: the
  unit-solve then-branch is a plain `Get`, not a distinct `Get_dynamic` leaf — so the per-leaf
  trick is not directly transplantable, and the safe behavior must instead be applied at the
  `Where` level (where the conditional-evaluation structure is visible).

- **Test scaffolding** — `test/operations/affine_lowering.ml::ac6` runs the
  `dst[2*oh+wh] = src[oh,wh]` scatter consumed at a plain iterator (and a triangular variant),
  comparing virtualized vs materialized execution; it has the build helpers (`run_scatter_then_copy`,
  `run_triangular`) usable to construct the hazardous shape. `test/operations/virtual_affine.ml`
  holds the structural IR walkers (`count_guard_ops` counting `Where`/`Cmplt`, `count_get`,
  `walk_t`/`walk_s`) and the `case_unit_solve_plain` / `case_triangular` cases that already
  assert the range guard is emitted — the natural home for a structural assertion that the
  guarded read is not in an unconditional debug position. gh-343's generated-C assertion
  approach (read the build-directory `.c` with `output_debug_files_in_build_directory` /
  `output_debug_files_in_run_directory` enabled) is the precedent for an end-to-end debug-path
  assertion.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

The narrowest sound change lives entirely in `c_syntax.ml`'s `debug_float`, in the
`Ternop` arm's `Ops.Where` case. Make the rendering not emit the conditionally-evaluated
then/else branch reads as unconditional `printf` accessor arguments. Because the then-branch is a
plain `Get` (unlike gh-343's distinct `Get_dynamic` leaf), the fix is applied at the `Where`
level rather than the leaf. Two candidate shapes, both faithful to gh-343's spirit (log a
safe surrogate; never dereference a guarded branch unconditionally):

- **Surrogate logging** — for a `Where`, log the condition value and a non-dereferencing label
  for the branches (analogous to `Get_dynamic`'s `@dyn_idx`), so no out-of-range branch is read.
  Simplest; loses the in-range branch's numeric value in the log unless the condition is
  separately reflected.
- **Short-circuiting render** — emit the branch values through the C ternary itself (so the C
  `?:` short-circuit gates the dereference) rather than as separate, eagerly-evaluated `printf`
  arguments. Preserves in-range fidelity (AC 2) at the cost of a more involved doc construction.

The choice between these (and exactly how to preserve AC-2 fidelity) is a real design decision,
so this proposal does **not** prescribe one — the plan phase should pick. The remaining two
changes are mechanical: the AC-5 comment at the `low_level.ml` guard site, and the AC-4
regression test (prefer a structural/generated-C assertion over a runtime fault, since the
runtime ternary short-circuits).

## Scope

In scope:
- The `debug_float` `Ternop`/`Where` fix in `arrayjit/lib/c_syntax.ml`.
- A deferred-soundness comment at the unit-solve guard site in `arrayjit/lib/low_level.ml`.
- A regression test pinning the previously-hazardous shape under debug logging (parity in
  `test/operations/affine_lowering.ml::ac6`, structural in `test/operations/virtual_affine.ml`).

Out of scope (explicit non-goals from the resolved task):
- Changing `try_unit_solve` / `range_guards` index derivation (option A, not chosen).
- Index clamping or an IR-level branch so the discarded read is never constructed (deferred full
  soundness; revisit when an eager/predicated backend lands).
- Backend `Where` lowering (`cuda_backend.ml` / `metal_backend.ml` `ternop_syntax`).
- The unsigned-index range-guard fix (commit `e28c4060`) — orthogonal, already landed.

Dependencies: none beyond Stage B already being merged.

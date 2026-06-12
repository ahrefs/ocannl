# Proposal: Implement local let-bindings in the `%cd` syntax

**Issue:** [ahrefs/ocannl#80](https://github.com/ahrefs/ocannl/issues/80)
**Milestone:** v1.0
**Status:** Draft proposal

## Status update (2026-06-12)

- Issue [#80](https://github.com/ahrefs/ocannl/issues/80) is **OPEN** (state reason: REOPENED), milestone v1.0 (due 2026-06-30); harness task is `deferred`.
- Still unimplemented: the `Pexp_let` hard error is intact, now at `tensor/ppx_cd.ml:1850-1859` (was 1832-1842), with the commented-out shallow-rewrite sketch at lines 1860-1862. Line references in the body have been refreshed.
- **gh-ocannl-348 ("Simplify translations from the `%cd` syntax") is now CLOSED**, so the coordination concern in "Related task" is resolved: the `transl` calling convention survived unchanged (`translate` at line 490, `loop = transl ~bad_pun_hints` at line 497), and the recommended closure-captured-env threading still applies as written.
- Other verified anchors: `expr_type` at lines 6-20, `projections_slot` at line 24, `result` at lines 77-93 — all unchanged. The `%cd` consumer counts still hold (71 occurrences in `tensor/operation.ml`, 3 in `lib/train.ml`). The `test/ppx/` diff-against-expected harness (`test_ppx_op.ml` / `test_ppx_op_expected.ml`) is intact.
- Line drift inside `ppx_cd.ml` (from the basis-rename and axis-annotation commits `d980b82f`, `b2cee000`, `c4918bd1`): special-purpose identifiers now at ~1052-1096, catchall prefix/suffix slot heuristic at ~1099-1111, `%op`'s `Pexp_let` handler at `ppx_op.ml:654-664`.
- Repo-wide renames since April 2026 (broadcast LUB→GLB reversal, "label"→"basis", "invalid"→"discardable") touched `ppx_cd.ml` only superficially; none of the design's assumptions are invalidated.
- The design, acceptance criteria, and the six ambiguities remain current; no implementation work has started.

## Goal

Replace the hard error at `tensor/ppx_cd.ml:1850-1859` with a real translation
of `let ... in ...` (and `let ... and ... in ...`, `let _ = ... in ...`,
`let rec ... in ...`) inside `%cd` blocks. The translator must thread a
small lexical environment that records, for each let-bound identifier, the
inferred `expr_type` ("kind": `Code`, `Array`, `Tensor`, `Value_of_tensor`,
`Grad_of_tensor`, ...) and the inferred `projections_slot` (`LHS` / `RHS1` /
... / `Undet`) of its RHS, so that uses of the identifier in the body
resolve to the same kind/slot they would have had if inlined. After the
change, refactoring a `%cd` block to introduce intermediate names should be
behaviour-preserving.

## Acceptance Criteria

- [ ] **Simple let inlines.** A unit test compiles a `%cd` block of the form
      `let%cd asgn ~t ~t1 ~t2 ~projections = let s = v1 + v2 in lhs =:+ s` and
      asserts the produced `Assignments.comp` is structurally equivalent to
      the inlined form `lhs =:+ v1 + v2` (i.e. identical `Accum_binop` /
      `Accum_unop` shape, identical projections lazy thunk, identical
      embedded-nodes set). *Mutation falsifier:* removing the env extension
      so `s` falls into the catchall `Pexp_ident` case at `ppx_cd.ml:1099`
      makes `s` resolve via the prefix/suffix heuristic to `slot = Undet`
      and `typ = Unknown`, which produces a different `comp` (or fails to
      compile because the LHS slot can no longer be derived).
- [ ] **Slot is correctly inherited.** A `%cd` block that binds an
      array-slotted RHS, e.g. `let r = v1 in lhs =:+ r * v2` (RHS1 slot via
      `v1`), produces the same `comp` as `lhs =:+ v1 * v2`. The slot
      recorded on `Pexp_let` flows to the `Pexp_ident` lookup and into
      `setup_array ~for_slot:RHS1`. *Mutation falsifier:* dropping the env
      entry forces the body to fall back on the prefix/suffix heuristic
      (`ppx_cd.ml:1104-1109`), under which a name like `r` matches none of
      the `lhs_*`/`rhs1_*`/... patterns and resolves to `Undet`; the
      generated assignment loses its slot tag and the projections it derives
      no longer match the inlined version.
- [ ] **Naming-convention vs env precedence is locked.** A second test
      defines `let lhs_intermediate = v1 + v2 in rhs1 =:+ lhs_intermediate`
      where the binding's RHS slot (LHS, inherited from `v1 + v2`'s
      computation context) and the name's prefix-heuristic slot (LHS via
      `lhs_*`) happen to agree, plus a third test where the binding's RHS
      slot disagrees with the prefix heuristic (e.g.
      `let lhs_x = v1 in ...` -- `v1` says RHS1, `lhs_*` says LHS). The
      resolved precedence rule (env-wins, see Ambiguity 1) is asserted by
      the test. *Mutation falsifier:* swapping the precedence in the
      `Pexp_ident` lookup flips the chosen slot and the test fails.
- [ ] **Kind is tracked across types.** A test binds a `Code`-typed RHS
      (e.g. an `if`-branch returning a `comp`) and uses the binding in
      sequencing position; a separate test binds a `Tensor`-typed RHS and
      uses the binding where a tensor is expected. Mixing kinds incorrectly
      (e.g. using a `Code`-bound name in tensor position) raises a PPX-time
      error whose message names the binding and its actual kind.
      *Mutation falsifier:* a kind-erasing implementation that records only
      `slot` would let the misuse compile and fail later (or silently
      miscompile); the assertion on the PPX error message catches this.
- [ ] **Nested let.** `let x = v1 + v2 in let y = x + v3 in lhs =:+ y`
      compiles to the same `comp` as `lhs =:+ (v1 + v2) + v3`. Nested env
      extension works to arbitrary depth (a 3-deep test should suffice).
      *Mutation falsifier:* an implementation that handles only the
      outermost let (re-enters the catchall for the inner let-body)
      regresses the inner binding to `Unknown`/`Undet` and produces a
      different `comp`.
- [ ] **Shadowing.** `let x = v1 in let x = v2 in lhs =:+ x` resolves the
      inner `x` to `v2`'s slot/kind, not `v1`'s, and produces the same
      `comp` as `lhs =:+ v2`. *Mutation falsifier:* using `Map.add` instead
      of `Map.set` (or otherwise failing to overwrite) keeps the outer
      binding visible and the test fails.
- [ ] **Parallel `let ... and ...`.** A test using
      `let x = v1 and y = v2 in lhs =:+ x + y` confirms each binding is
      visible in the body but not in the other binding's RHS (standard
      OCaml scoping). *Mutation falsifier:* threading the env left-to-right
      so that `y`'s RHS sees `x` (incorrect scoping) is caught when the
      tests are extended with `let x = v1 and y = x in ...`, which under
      the buggy implementation would resolve `y` to RHS1 instead of treating
      `x` as a free variable resolved by the surrounding scope.
- [ ] **Discard binding.** `let _ = comment_msg in lhs =:+ v1 + v2`
      compiles cleanly (no unused-var warning, no env entry needed) and
      produces the same `comp` as the bare `lhs =:+ v1 + v2`. The discard
      pattern is recognized and skipped without polluting the env.
      *Mutation falsifier:* if the binding is naively translated as
      `let _ = ... in ...` with the LHS being a non-`comp` expression
      (e.g. a string), the OCaml compiler raises a type error -- the test
      catches that.
- [ ] **`vbs` lifting still works through let-bindings.** A binding whose
      RHS contains an inline tensor declaration (e.g.
      `let foo = { hint } * v1 in lhs =:+ foo`) still surfaces the punning
      `value_binding` to the top of the `%cd` block via the `vbs`
      mechanism. *Mutation falsifier:* if the let-body's `vbs` are
      discarded (or the RHS's `vbs` are not merged into the binding's
      result), the inline declaration is lost and the produced expression
      references an undefined identifier -- a compile-time failure caught
      by the test.
- [ ] **Unused let-bindings do not warn.** `let _x = v1 in lhs =:+ v2`
      compiles without the `unused-var-strict` / `unused-value-declaration`
      warning (i.e. the PPX does not insert a synthetic use of `_x`).
      *Mutation falsifier:* an implementation that wraps the body in
      `let _ = _x in ...` to suppress unused-var warnings would itself
      trigger a warning under `-w +26`; alternatively, dropping the
      underscore convention recognition would emit the warning directly.
- [ ] **No regression in the existing `%cd` corpus.** `dune build` and
      `dune runtest` succeed across the repo. Specifically: `tensor/`,
      `lib/train.ml`, `tensor/operation.ml` (which is the largest in-tree
      consumer with 71 `%cd` invocations), `bin/compilation_speed.ml`, the
      training examples under `test/training/`, and the existing
      `test/ppx/` PPX expansion tests. *Mutation falsifier:* any change
      that disturbs the existing recursion paths (e.g. accidentally
      threading a default-empty env that overrides the catchall heuristic
      for free variables) breaks identifier resolution for names like
      `learning_rate`, `momentum`, etc. that are bound outside the `%cd`
      block.

## Context

### Issue history and reversal

The user originally closed this issue on 2024-08-27 with "This is needless
complexity," then reopened it on 2025-08-11 with a refined design: "This
does make sense. The environment needed will hold a kind (e.g. non-diff
tensor, code, array etc.) and slot (LHS, RHS1 etc.) of the bindings."
The proposal honours that refinement -- the env carries `expr_type *
projections_slot`, not just one or the other.

### OCANNL audit pause

Per harness memory, **autonomous OCANNL work is paused** for the user's
hands-on quality audit (see `project_ocannl_quality_audit.md`). This
proposal is filed for record but the task should be left in
`deferred_launch` state; it should not auto-start. PPX work in particular
is fiddly and benefits from review before agents begin.

### Where the relevant code lives today

The seeded design pointers in the task file are accurate as of this audit:

- **Error site:** `tensor/ppx_cd.ml:1850-1859` *(Update 2026-06-12: was
  1832-1842)* -- the `Pexp_let` arm in
  `transl` builds a `Location.error_extensionf` with the message
  "let-in: local let-bindings not implemented yet". A commented-out sketch
  on lines 1860-1862 shows the obvious-but-wrong shallow rewrite (recurse
  into bindings and body, rebuild `Pexp_let`); the comment is correct that
  this is not enough -- it doesn't track kind/slot.
- **`expr_type` definition:** `tensor/ppx_cd.ml:6-20`. Variants are `Code
  { is_commented }`, `Array`, `Value_of_tensor of expression`,
  `Grad_of_tensor of expression`, `Tensor`, `Unknown`, `Merge_value of
  expression`, `Merge_grad of expression`, `No_grad_tensor_intro { name;
  name_expr; extra_args }`, `Function`. Note that several variants carry
  expression payloads -- the env value type must accommodate that or
  collapse them carefully (see Ambiguity 4).
- **`projections_slot` definition:** `tensor/ppx_cd.ml:24` -- `LHS | RHS1
  | RHS2 | RHS3 | Scalar | Nonslot | Undet`.
- **`result` type:** `tensor/ppx_cd.ml:77-93` -- the value `transl`
  returns. Includes `vbs : value_binding list`, `typ`, `slot`, `expr`, and
  `array_opt_of_code`.
- **`transl` entry:** `tensor/ppx_cd.ml:490-497`. Currently parameterised
  by `~bad_pun_hints ~proj_in_scope`. The `loop` alias on line 497 is just
  `transl ~bad_pun_hints` (so all interior recursion calls go through
  `loop ~proj_in_scope`).
- **`loop` call sites:** ~50 call sites inside `transl`, mostly of the
  form `loop ~proj_in_scope expr1`. (Verified by `grep -c "loop ~proj"`.)
  Threading a new `~env` parameter would touch every one of them. See
  "Approach" for a less invasive option.
- **Slot heuristic for free identifiers:** `tensor/ppx_cd.ml:1099-1111`.
  After the special-case identifiers (`lhs`, `v`, `g`, `rhs1`, `t`, `t1`,
  `v1`, `g1`, ...), unrecognized identifiers are run through prefix/suffix
  patterns: `lhs_*` / `*_lhs` -> LHS, `rhs1_*` / `*_rhs1` -> RHS1, ...,
  `rhs_*` / `*_rhs` -> RHS1, otherwise Undet. The let-binding lookup must
  decide whether to override or coexist with this heuristic (see
  Ambiguity 1).
- **Special-purpose identifiers:** `tensor/ppx_cd.ml:1052-1096` -- `lhs`,
  `v`, `g`, `rhs1`, `t`, `t1`, `v1`, `g1`, `rhs2`, `t2`, `v2`, `g2`,
  `rhs3`, `t3`, `v3`, `g3`. Each maps to a fixed `(typ, slot)` pair.
  These should NOT be shadowable by a user `let lhs = ... in ...` (see
  Ambiguity 5) -- or if they should, the rule needs to be explicit.
- **`%op` reference implementation:** `tensor/ppx_op.ml:654-664`. `%op`'s
  `Pexp_let` handler is structurally simple: recurse into each binding's
  RHS, recurse into the body, reduce all `vbs`, rebuild the `Pexp_let`.
  It does NOT track an environment. This works for `%op` because every
  expression has type `Tensor.t` -- there is no kind/slot distinction to
  preserve. **The `%op` precedent is informative for the recursion shape
  and `vbs` plumbing, but is NOT structurally portable for the env-tracking
  part.** That has to be designed fresh for `%cd`.
- **Documentation invariants:** `docs/syntax_extensions.md:221-311`
  documents the slot-detection naming convention and the special
  identifiers. The implementation must preserve both. Section
  "Projection slot detection by naming convention" (lines 275-311)
  should be updated to mention let-binding behaviour.

### Existing in-tree consumers

`%cd` is used in `lib/train.ml` (3 occurrences), `tensor/operation.ml` (71
occurrences -- the largest consumer, defining all primitive op gradients),
and `bin/compilation_speed.ml` (1 occurrence). A spot-check of
`tensor/operation.ml`'s `add`, `sub`, `mul`, `pointmul` definitions shows
that **none of the existing `%cd` blocks use local `let-in`** -- they
sequence assignments with `;`. So this feature is purely additive: no
existing call sites need to be migrated.

### Test infrastructure precedent

PPX expansion tests in this repo use a "diff `_actual.ml` against
`_expected.ml`" pattern: see `test/ppx/test_ppx_op.ml` paired with
`test/ppx/test_ppx_op_expected.ml`, with the diff harness in
`test/ppx/dune` (lines 12-40) and the `pp.ml` driver. The new test for
`%cd` let-bindings should follow this convention: a new
`test_ppx_cd.ml` (containing the let-binding cases above) and a
`test_ppx_cd_expected.ml` (the post-expansion ground truth), wired up
identically. End-to-end "the expanded code actually runs and produces the
right `Assignments.comp`" is verified separately by extending one of the
`test/operations/` runners with a small assertion, OR by building a
fixture in `test/ppx/test_ppx_cd.ml` that calls into the expanded code.

### `vbs` and the punning lifecycle

The `result.vbs` field carries `value_binding`s introduced by inline
tensor declarations (e.g. `{ hint }`). They are lifted to the top of the
`%cd` block (see `tensor/ppx_cd.ml:171-181`). When a let-body's RHS
contains an inline declaration, the resulting `vbs` must propagate up
through the `Pexp_let` reconstruction -- otherwise the lifted
`value_binding` will reference identifiers no longer in scope. The `%op`
implementation already does this correctly (`reduce_vbss` on `ppx_op.ml:664`);
`%cd` must do the same plus carry the env.

### Related task

`gh-ocannl-348` ("Simplify `%cd` syntax translations") is in scope of
the same area. If it lands first, the env-threading touch points may
shrink. If this proposal lands first, gh-ocannl-348 may need to consider
the new env parameter. Coordinate by reading whichever is in flight at
implementation time. *(Update 2026-06-12: #348 is now closed; the
`transl`/`loop` calling convention is unchanged, so no coordination is
needed anymore.)*

## Approach

### High-level plan

1. Define the env type:

   ```ocaml
   type binding_info = { typ : expr_type; slot : projections_slot }
   type env = binding_info Map.M(String).t
   let empty_env : env = Map.empty (module String)
   ```

2. Decide how to thread `env` (see Ambiguity 2). Recommended:
   make `env` a mutable `ref` (or pair of `Hashtbl` + scope-pop list)
   captured by the `transl` closure, since `transl` already lives inside
   the `let translate ?ident_label expr =` closure (line 490) that
   captures the per-call `punned` table. This avoids touching ~50 `loop`
   call sites and matches the existing pattern. The trade-off
   (correctness under exception unwinding, scope-pop ordering for `let
   ... and ...`) is local to the let-handler.

3. New `Pexp_let` arm (replacing lines 1850-1859):

   ```ocaml
   | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
       let recurse_binding binding =
         let res = loop ~proj_in_scope binding.pvb_expr in
         let name_opt = pat_var_name binding.pvb_pat in
         (name_opt, res, { binding with pvb_expr = res.expr })
       in
       (* For let rec, install bindings into env BEFORE translating RHSes
          (with kind=Unknown / slot=Undet placeholders); for nonrec, install
          bindings AFTER translating RHSes; for parallel and, translate all
          RHSes in the OUTER env and install all bindings together. *)
       ...
       let body_res = loop ~proj_in_scope body in
       (* Pop bindings from env. *)
       {
         vbs = reduce_vbss (List.map binding_results ~f:(fun r -> r.vbs)
                            @ [ body_res.vbs ]);
         typ = body_res.typ;
         slot = body_res.slot;
         expr = { expr with pexp_desc = Pexp_let (recflag, new_bindings, body_res.expr) };
         array_opt_of_code = body_res.array_opt_of_code;
       }
   ```

4. New `Pexp_ident` lookup. In the catchall identifier branch (line
   1099) check the env first. If found, return a `result` with the
   recorded `typ` and `slot`. If not found, fall back to the existing
   prefix/suffix heuristic. This gives "env-wins" precedence (see
   Ambiguity 1).

5. Discard pattern (`let _ = ...`) and underscored names (`let _x =
   ...`): translate the RHS for its side effects but do not insert an
   env entry for `_`; for `_x` insert with the underscore-prefixed name
   so uses (rare) still resolve.

6. Tuple / record / constructor patterns on the LHS: out of scope for
   v1 -- raise a PPX-time error "let-in: only simple variable patterns
   are supported in `%cd` for now". This keeps the pattern-matching
   surface small. The task's acceptance criteria do not require complex
   patterns; if needed they can be added in a follow-up.

7. Documentation: extend `docs/syntax_extensions.md` (after the
   "Projection slot detection by naming convention" section) with a
   short subsection "Local let-bindings" describing precedence rules,
   recursive let support (per Ambiguity 3), parallel `and` semantics,
   and the discard / underscore conventions.

### Falsifiers' relationship to the implementation

Each acceptance criterion above is paired with a specific mutation that
breaks it -- review should run all of them as `dune test` cases plus
spot mutations in `transl` to verify the test catches the breakage.

## Ambiguities

The implementer should resolve these with the user before starting code
work:

1. **Naming-convention vs env precedence for slot detection.** When
   `let lhs_x = v1 in ...` is followed by use of `lhs_x` in the body,
   the name's prefix heuristic says LHS but the binding's RHS slot says
   RHS1. **Recommendation: env-wins** (intentional bindings override
   accidental name shape), because that's the principle of least
   surprise once the user has explicitly bound a name. **Alternatives:**
   heuristic-wins (preserves the documented invariant for users who
   relied on naming for shape inference), or hard-error on conflict.

2. **Env threading mechanism.** Closure-captured mutable ref vs explicit
   `~env` parameter on every `loop` call. **Recommendation: closure-captured
   mutable** (matches the existing `punned : (string, _) Hashtbl.t`
   pattern in the same closure scope, avoids touching ~50 call sites).
   **Alternative:** explicit threading is more functional and easier to
   reason about, but the diff is much larger and gh-ocannl-348 may
   refactor the calling convention anyway.

3. **Recursive let (`let rec`).** `%op` allows `let rec` (its handler
   doesn't distinguish). For `%cd`, recursive bindings of `comp` values
   are unlikely to be useful and would require placeholder kind/slot
   during RHS translation. **Recommendation: support `let rec` syntactically
   but treat the binding as `Unknown`/`Undet` during its own RHS
   translation** (so referencing the recursive name inside its own RHS
   falls into the catchall heuristic). **Alternatives:** disallow with a
   clear PPX error, or fully support with a dedicated fixpoint pass
   (probably overkill).

4. **`expr_type` payload variants.** Variants like `Value_of_tensor of
   expression`, `No_grad_tensor_intro { name; name_expr; ... }`, and
   `Function` carry expression payloads. **Question:** when a let-bound
   RHS has one of these as its `typ`, do we record the full payload in
   the env (so the body sees the same `expr` for `Value_of_tensor` /
   `Grad_of_tensor`), or do we collapse to a payload-less proxy (e.g.
   `Tensor` for tensor-shaped values, `Unknown` for `Function`)?
   **Recommendation: record the full payload.** Anything less loses
   information needed by the gradient-grad-of, value-of, and merge-of
   code paths.

5. **Shadowing of special identifiers.** Should `let lhs = ... in ...`
   override the special-purpose identifier `lhs` (which currently maps
   to LHS slot, Array kind)? **Recommendation: hard-error** -- shadowing
   special-purpose identifiers is almost certainly a user mistake. The
   PPX error message should suggest renaming.

6. **`vbs` lifting scope.** When a let-bound RHS contains an inline
   tensor declaration (`let foo = { hint } * v1 in ...`), the lifted
   `value_binding` for `hint` is currently lifted to the TOP of the
   whole `%cd` block, where it is bound for the entire block's lifetime.
   **Question:** is that the desired scope, or should it be lifted only
   to the surrounding `let`'s scope (i.e. wrapped in the rebuilt
   `Pexp_let`)? **Recommendation: keep top-level lifting** (matches
   `%op` behaviour and matches the current convention that punned
   tensors are first-class block-level entities), but document this
   explicitly in `docs/syntax_extensions.md` because it's surprising.

(6 ambiguities; the worker may collapse 3+5 if the user prefers a
single "shadowing rule" answer.)

## Estimated effort

Medium (3-4 days), per the task file. Distribution:
- Day 1: env type, threading mechanism, simple let case + tests.
- Day 2: nested, shadowing, parallel `and`, discard/underscore.
- Day 3: edge cases (kind tracking, payload variants, vbs lifting),
  documentation.
- Day 4: review, regression sweep, fix-ups from PPX error message
  polish.

## Out of scope

- Complex patterns on the LHS (tuples, records, constructors).
- `let-binding`s in the LHS position of `=`-style assignment operators
  (the LHS must remain a simple slot expression).
- Module-level `let` inside `%cd` (already handled by `Pexp_letmodule`,
  no env tracking needed there).
- gh-ocannl-348's broader simplification work.

# Proposal: Blacklist primitive operator names when selecting identifier names

**Task**: gh-ocannl-383
**Issue**: https://github.com/ahrefs/ocannl/issues/383

## Goal

Ensure generated C code never uses C math function names or C keywords as variable
identifiers, and update the introductory slides to reflect current identifier naming.

The core blacklist infrastructure already exists (commit 132144b1, May 2025). This
task completes the work by extending coverage to C keywords and verifying slides.

## Acceptance Criteria

- C math function names (exp, log, sqrt, fma, fmaf, fmaxf, fminf, copysignf, etc.)
  remain blacklisted via the dynamic extraction in `c_syntax.ml` `ident_blacklist`.
- C language keywords (int, float, for, if, return, etc.) and C99 additions are added
  to the blacklist so that tensor labels matching these names get disambiguated.
- The `ident_blacklist` in `c_syntax.ml` is the single source of truth; no other
  `get_ident_within_code` call sites need modification (the `assignments.ml` and
  `low_level.ml` pretty-printers produce OCaml-level IR, not C code).
- The introductory slides (`docs/slides-basics_backprop_training_codegen.md`) are
  reviewed and updated if any generated code examples show pre-blacklist naming.
- Existing tests pass without regression.

## Context

### Current implementation

1. **`ident_blacklist`** (`arrayjit/lib/c_syntax.ml` lines 107-172): Dynamically
   extracts C math function names by probing `Ops.unop_c_syntax`, `binop_c_syntax`,
   `ternop_c_syntax`, and `vec_unop_c_syntax` across all precisions. Any result
   ending with `(` is treated as a function call and the name is collected.

2. **Blacklist consumption** (`c_syntax.ml` line 233): Passed to
   `Low_level.get_ident_within_code ~blacklist:B.ident_blacklist`.

3. **Disambiguation** (`tnode.ml` `styled_ident`, lines 536-558): When a label
   appears in the `repeating_*_idents` hash tables (which include blacklisted names
   seeded with a sentinel ID of -1), the identifier is prefixed with `n<id>_`.

### Gap: no C keyword coverage

The dynamic extraction only captures math function names. C keywords like `int`,
`float`, `for`, `if`, `return` are not blacklisted. While unlikely as tensor labels,
PPX-derived labels come from OCaml variable names, and some overlap exists (e.g.
`for`, `if`, `mod`). Adding a static keyword list closes this gap.

### Slides status

The slides at `docs/slides-basics_backprop_training_codegen.md` already use
disambiguated variable names (e.g., `n35`, `w3_grad`) and C function names appear
only as function calls (`fma(`, `fmaf(`, `fmaxf(`). No pre-blacklist naming issues
are visible, but the slides should be verified against current codegen output after
the keyword extension.

### Approach

1. Add a `c_keywords` list in `c_syntax.ml` containing C89/C99 reserved words and
   append it to the dynamically extracted `ident_blacklist`.
2. Optionally add CUDA/Metal reserved words behind a comment or gated on backend,
   for future-proofing.
3. Regenerate or manually verify slide code examples against current output.
4. Run the full test suite to confirm no regressions.

### Code pointers

| Location | Role |
|----------|------|
| `arrayjit/lib/c_syntax.ml:107-172` | `ident_blacklist` definition |
| `arrayjit/lib/c_syntax.ml:233` | Blacklist passed to `get_ident_within_code` |
| `arrayjit/lib/low_level.ml:1425-1470` | `get_ident_within_code` with blacklist seeding |
| `arrayjit/lib/tnode.ml:536-558` | `styled_ident` disambiguation logic |
| `arrayjit/lib/tnode.ml:523-534` | `no_grad_ident_label` label extraction |
| `docs/slides-basics_backprop_training_codegen.md` | Introductory slides |

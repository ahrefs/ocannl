# Proposal: Blacklist primitive operator names when selecting identifier names

**Task**: gh-ocannl-383
**Issue**: https://github.com/ahrefs/ocannl/issues/383

## Status update (2026-06-12)

- Issue #383 is still OPEN on GitHub, milestone v0.7.
- The core claims hold against current code: `ident_blacklist` is still at `arrayjit/lib/c_syntax.ml:107-172` (dynamic extraction across `Ops.[ byte; half; single; double ]` precisions, including `vec_unop_c_syntax` for `Uint4x32_to_prec_uniform`), and is still consumed at `c_syntax.ml:233` via `Low_level.get_ident_within_code ~no_dots:true ~blacklist:B.ident_blacklist`.
- The gap is unchanged: no C keyword list exists in `c_syntax.ml` — the proposed `c_keywords` extension has NOT landed. No commits since April 2026 touch the blacklist.
- Line-number drift in helpers: `Low_level.get_ident_within_code` is now at `low_level.ml:1656`; `Tnode.no_grad_ident_label` at `tnode.ml:440`; `Tnode.styled_ident` at `tnode.ml:453`. (Code pointers table below updated.)
- The slides (`docs/slides-basics_backprop_training_codegen.md`) were updated since (streams/data-parallel section removed, zeroing example cleaned up under #420); their generated-code excerpts still show only disambiguated/clean identifiers (`n35`, `w3_grad`, `relu`) with C math functions appearing only as calls — consistent with the "Slides status" section below.
- Remaining work is exactly the Approach below: add the static C keyword list (plus optional CUDA/Metal reserved words), re-verify slides, run tests.

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
   *(Update 2026-06-12: the probing iterates over hardcoded constructor lists,
   not the full op types, and these have already drifted: the unop list ends at
   `Not` and omits `Uint4x32_to_prec_uniform1` (added later, `ops.ml:421`),
   whose codegen emits builtin calls named `uint4x32_to_<prec>_uniform(` —
   distinct from the vec variant's `..._uniform_vec(` that *is* probed. So the
   "dynamic" extraction silently misses ops added after the lists were
   written.)*

2. **Blacklist consumption** (`c_syntax.ml` line 233): Passed to
   `Low_level.get_ident_within_code ~blacklist:B.ident_blacklist`.

3. **Disambiguation** (`tnode.ml` `styled_ident`, now at line 453): When a label
   appears in the `repeating_*_idents` hash tables (which include blacklisted names
   seeded with a sentinel ID of -1), the identifier is prefixed with `n<id>_`.

### Gap: no C keyword coverage

The dynamic extraction only captures math function names. C keywords like `int`,
`float`, `for`, `if`, `return` are not blacklisted. While unlikely as tensor labels,
PPX-derived labels come from OCaml variable names, and some overlap exists.
*(Update 2026-06-12: the original examples `for`, `if`, `mod` are wrong — those
are OCaml keywords and can never be PPX-derived labels from `let`-bound names.
The real overlap set is C keywords that are valid OCaml value identifiers:
`int`, `float`, `double`, `char`, `long`, `short`, `return`, `signed`,
`unsigned`, `void`, `auto`, `break`, `case`, `const`, `continue`, `default`,
`enum`, `extern`, `goto`, `register`, `sizeof`, `static`, `switch`, `typedef`,
`union`, `volatile`, `inline`, `restrict`. Additionally, labels supplied as
explicit `~label` strings are arbitrary and can be any keyword, including
`for`/`if`.)* Adding a static keyword list closes this gap.

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
| `arrayjit/lib/low_level.ml:1656` | `get_ident_within_code` with blacklist seeding |
| `arrayjit/lib/tnode.ml:453` | `styled_ident` disambiguation logic |
| `arrayjit/lib/tnode.ml:440` | `no_grad_ident_label` label extraction |
| `docs/slides-basics_backprop_training_codegen.md` | Introductory slides |

## Design review (2026-06-12)

**Verdict: sound-with-changes.** The architecture is right — `ident_blacklist`
in the per-backend `C_syntax_config` is the single choke point
(`c_syntax.ml:233` → `low_level.ml:1656-1664` seeding → `tnode.ml:453`
disambiguation), and the `assignments.ml`/`low_level.ml` pretty-printer call
sites are correctly identified as out of scope. But the proposal under-scopes
the blacklist contents and ignores that the existing "dynamic" extraction has
already gone stale.

Key design fact the recommendations rest on: **over-blacklisting is nearly
free.** A blacklisted name is seeded with sentinel id `-1`
(`low_level.ml:1660-1664`), so a colliding label merely gains an `n<id>_`
prefix (`tnode.ml:472`) — a cosmetic cost. There is no reason to keep the
list minimal; the only regression risk is identifier churn in `.expected`
files that embed generated code, which only triggers if an existing test
label collides with a newly added entry (none plausible for keywords).

**Recommendations** (strongest first):

1. **Make the op-name extraction auto-derived instead of extending hardcoded
   lists.** The constructor lists in `c_syntax.ml:107-172` already drifted
   (missing `Uint4x32_to_prec_uniform1`; see in-place update above), and
   #305's proposed `Mul3` would be the next silent omission.
   `ppx_variants_conv` is already a preprocessor for `arrayjit/lib`
   (`arrayjit/lib/dune:15`): add `[@@deriving variants]` to
   `unop`/`binop`/`ternop`/`vec_unop` in `ops.ml` and enumerate all
   constructors via the generated `Variants_of_*` module (all constructors
   are nullary). This removes the maintenance hazard permanently and is the
   right answer to "blacklist primitive operator names" in the issue title.
2. **Backend-specific reserved words are required for Metal, not optional.**
   MSL keywords include `kernel`, `device`, `constant`, `thread`,
   `threadgroup`, `half`, `uint` — and `kernel`, `half`, `constant`,
   `device` are *highly plausible* tensor labels in NN code (conv kernels!).
   Since `ident_blacklist` is a config-module value, have
   `metal_backend.ml`'s `C_syntax_config` append an MSL list to the
   inherited one. CUDA reserved words (`__global__` etc.) are implausible
   labels; a short list (`warpSize`, `threadIdx`, `blockIdx`, `blockDim`,
   `gridDim`) is cheap insurance.
3. **Blacklist scaffolding identifiers, not just keywords and op names**:
   names the generated code itself declares or references — `log_file`,
   `log_file_name` (`c_syntax.ml:203`), the typedef names `uint32_t`/
   `uint64_t` used in loop headers (a local variable with that name makes
   the subsequent `for (uint32_t i0 = ...` ill-formed), and builtin helper
   names from `builtins.c` not reachable via op-syntax probing.
4. **Add a falsifier test.** The acceptance criteria currently only require
   "existing tests pass", which cannot detect the gap being fixed. Add a
   standalone test (per CLAUDE.md, under `test/operations` with an
   `.expected` file) that builds tensors labeled e.g. `return`, `int`,
   `kernel`, `half` (via explicit `~label` or let-bound names) and compiles
   a routine on the configured backend — failing today on Metal for
   `kernel`/`half`, and on all C backends for `int`/`return`.

**Open decision points for Łukasz**:

- Auto-derivation via `[@@deriving variants]` on the op types (small,
  one-time, removes a footgun) vs. just fixing the hardcoded lists and
  accepting future drift.
- Whether the static keyword list should be C-only in `Pure_C_config` with
  per-backend appends (recommended), or one union list shared by all
  backends (simpler, slightly more cosmetic prefixing).
- Whether slide verification (the issue's second sentence) needs anything
  beyond the already-done freshness check — the slides currently show only
  clean identifiers, so this is likely a no-op; confirm and close that part
  of the issue text.

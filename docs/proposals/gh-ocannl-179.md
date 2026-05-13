# C-syntax tracing printf: clean line breaks and indentation

## Goal

Generated tracing log statements (`fprintf` / `printf` / Metal `log_debug`) in
the C, CUDA, and Metal backends currently produce ugly line breaks and
indentation when PPrint's pretty-printer wraps long calls. The cause is that
each `pp_log_statement` implementation builds its document from raw `^^`
concatenation only, with no `group` / `nest` / `break` guidance — so when the
call exceeds the line width, PPrint breaks at arbitrary, visually unhelpful
points.

This is a cosmetic-only fix to the formatting of generated source code. The
generated code already compiles and runs correctly.

Issue: <https://github.com/ahrefs/ocannl/issues/179>

## Acceptance Criteria

- [ ] `fprintf` log statements produced by the C backend
      (`Pure_C_config.pp_log_statement` in `arrayjit/lib/c_syntax.ml`) lay out
      cleanly when the call exceeds the line width: arguments either all on one
      line (when they fit) or each broken to its own line at a consistent
      indent, instead of arbitrary mid-call breaks.
- [ ] `printf` log statements produced by the CUDA backend
      (`Cuda_syntax_config.pp_log_statement` in
      `arrayjit/lib/cuda_backend.ml`) lay out cleanly under the same rule.
- [ ] `os_log` / `log_debug` statements produced by the Metal backend
      (`Metal_syntax_config.pp_log_statement` in
      `arrayjit/lib/metal_backend.ml`) lay out cleanly under the same rule.
- [ ] Generated code still compiles for all three backends (the change is
      purely about whitespace, not tokens).
- [ ] No regressions in existing tests; expected-output files (if any) are
      updated to reflect the improved layout.

## Context

The three `pp_log_statement` implementations all share the same shape:
build a string-prefix doc (e.g. `"fprintf(log_file, "`, `"printf("`,
`"<obj>.log_debug("`), then a quoted format-string literal, then a
comma-separated list of argument docs, then `rparen ^^ semi`. Today they
wire these together with `^^` and a hard `space` between args:

C backend (`arrayjit/lib/c_syntax.ml`, `Pure_C_config.pp_log_statement`):

```ocaml
log_file_check ^^ string "fprintf(log_file, "
^^ dquotes (string base_message_literal)
^^ (if List.is_empty args_docs then empty else comma ^^ space)
^^ separate (comma ^^ space) args_docs
^^ rparen ^^ semi
```

CUDA backend (`arrayjit/lib/cuda_backend.ml`,
`Cuda_syntax_config.pp_log_statement`):

```ocaml
string "printf("
^^ dquotes (string format_string_literal)
^^ comma ^^ space
^^ separate (comma ^^ space) all_args
^^ rparen ^^ semi
```

Metal backend (`arrayjit/lib/metal_backend.ml`,
`Metal_syntax_config.pp_log_statement`):

```ocaml
string metal_log_object_name ^^ string ".log_debug(" ^^ base_doc ^^ comma ^^ space
^^ separate (comma ^^ space) args_docs
^^ rparen ^^ semi
```

The same file (`c_syntax.ml`) already uses PPrint's `group` / `nest` /
`ifflat` / `break` combinators idiomatically elsewhere, e.g. for ternary,
binary, and call-like expressions around lines 177–200, 331, 344–346, 805.
Those existing call sites are good reference style for the fix.

The pretty-printer is invoked in `filter_and_prepend_builtins` with
`PPrint.ToBuffer.pretty 1.0 110 ...` (line ~240), so the target line width
is 110 columns and the ribbon ratio is 1.0 — which is what `group` will
respect when deciding flat-vs-broken.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

Replace the hard `space` separators with PPrint's soft-break primitives, and
wrap each `pp_log_statement` body in a `group` so the whole call is laid out
flat when it fits or fully broken when it doesn't. Concretely, for each
backend:

1. Replace `comma ^^ space` between arguments with `comma ^^ break 1` inside
   `separate` (i.e. `separate (comma ^^ break 1) args_docs`).
2. Wrap the argument list in `nest N (break 0 ^^ ...)` so broken lines are
   indented relative to the `printf(` opening — `N = 4` matches the
   indentation already used for similar call-like constructs in `c_syntax.ml`.
3. Wrap the entire statement (after any leading guard like `log_file_check`)
   in `group (...)` so PPrint chooses flat or broken atomically.

A typical resulting shape is:

```ocaml
group
  (string "fprintf(log_file, "
  ^^ dquotes (string base_message_literal)
  ^^ (if List.is_empty args_docs then empty
      else comma ^^ nest 4 (break 1 ^^ separate (comma ^^ break 1) args_docs))
  ^^ rparen ^^ semi)
```

For the C backend, keep `log_file_check` outside the `group` (it's a guard,
not part of the call). For the CUDA backend, the format-string-literal
argument is always present, so the `if List.is_empty` guard is unnecessary.
For the Metal backend, preserve the existing branch on
`List.is_empty args_docs` (the no-args case has no comma at all).

After the change, regenerate any captured-output test fixtures and visually
inspect a sample of generated source to confirm the layout is readable.

## Scope

In scope:

- `Pure_C_config.pp_log_statement` in `arrayjit/lib/c_syntax.ml`.
- `Cuda_syntax_config.pp_log_statement` in `arrayjit/lib/cuda_backend.ml`.
- `Metal_syntax_config.pp_log_statement` in `arrayjit/lib/metal_backend.ml`.
- Updating any expected-output test fixtures whose recorded log lines change
  whitespace.

Out of scope:

- Any change to log-statement semantics, argument list, or format string.
- Refactoring the duplicated structure across the three backends into a
  shared helper (could be a follow-up; the duplication is small and the
  per-backend differences — file guard, log_id prefix, no-args branch — are
  real).
- Reworking unrelated pretty-printing in the same files.
- gh-ocannl-160 ("make tracing more efficient or abstract"): tracked
  separately; this proposal only touches layout.

Dependencies: none.

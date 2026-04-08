# Fix broken test suite: OCANNL

**Task**: task-bb30d0be  
**Project**: OCANNL  
**Date**: 2026-04-07  
**Status**: Ready to implement

## Goal

Restore `dune runtest` to exit code 0 by fixing two independent test failures introduced in recent
commits.

## Acceptance Criteria

- `dune runtest` exits with code 0 (all tests pass)
- `arrayjit/test/test_ndarray_binary_io.expected` includes the two log-level debug lines that the
  current code always emits when `log_level=0`
- `tensor/shape.ml` `derive_projections` no longer crashes with a duplicate-key exception when
  building `symbol_to_proj` for convolution/pooling cases where output_size=1

## Context

Two unrelated test failures exist as of 2026-04-07:

### Failure 1: `arrayjit/test/test_ndarray_binary_io` — output mismatch

`dune runtest` diff shows two lines missing from the `.expected` file:

```
+Retrieving commandline, environment, or config file variable ocannl_log_level
+Found 0, in the config file
```

Root cause: commit `b3262b2c` added `|| equal_string n "log_level"` to the `with_debug` condition
in `arrayjit/lib/utils.ml:get_global_arg` (line 164). This causes the `log_level` lookup itself to
always emit debug output, regardless of the current log level — it is a self-referential special
case. The `.expected` file for `test_ndarray_binary_io` was written after this commit but does not
include these lines. The `test_max_pool2d.expected` file already contains them (confirmed), so only
`test_ndarray_binary_io.expected` needs updating.

### Failure 2: `test/einsum/test_max_pool2d` — fatal exception

```
Fatal error: exception ("Map.of_alist_exn: duplicate key" (Symbol 42))
Called from Ocannl_tensor__Shape.derive_projections in file "tensor/shape.ml", lines 1899-1901
```

Root cause: `derive_projections` builds `symbol_to_proj` via `Map.of_alist_exn` over the result of
`Row.product_dim_iterators`. For the `test_max_pool2d_output_dim_1` test case (3x3 input, window=3,
stride=2 → 1x1 output), two different `proj_id`s both map to the same iterator symbol (Symbol 42).
When the output dimension is 1, the `oh` dimension collapses and shares an iterator with one of the
kernel dimensions, producing a duplicate key. `Map.of_alist_exn` panics on duplicates.

This was introduced in commit `9100f654` (Concat projection unification). The `symbol_to_proj` map
is subsequently used only via `Map.find` in `missing_entries` (line ~1943), so keeping just the
first occurrence per symbol is semantically correct.

## Approach

### Fix 1: Update expected file (minimal, preferred)

Prepend the two log-level lines to
`arrayjit/test/test_ndarray_binary_io.expected`:

```
Retrieving commandline, environment, or config file variable ocannl_log_level
Found 0, in the config file
PASS: Byte
...
```

This is the minimal fix that matches the current intended behavior (the `log_level` self-logging was
deliberately added in `b3262b2c`). Removing the `|| equal_string n "log_level"` condition would be
a behavioral regression and is out of scope for this bug fix.

### Fix 2: Replace `Map.of_alist_exn` with `Map.of_alist_reduce` in `derive_projections`

In `tensor/shape.ml` at lines 1898–1901, change:

```ocaml
let symbol_to_proj =
  Map.of_alist_exn
    (module Idx.Symbol)
    (Row.product_dim_iterators proj_env |> List.map ~f:(fun (p, d, s) -> (s, (p, d))))
in
```

to:

```ocaml
let symbol_to_proj =
  Map.of_alist_reduce
    (module Idx.Symbol)
    (Row.product_dim_iterators proj_env |> List.map ~f:(fun (p, d, s) -> (s, (p, d))))
    ~f:(fun first _second -> first)
in
```

The `~f:(fun first _second -> first)` keeps the first mapping when duplicates occur, which is
correct because the map is only queried to fill in entries for symbols not already present in
`unique_by_iterator` (which was already deduplicated by symbol). The underlying cause — why two
proj_ids share the same iterator symbol for 1x1 output — may be a deeper shape inference issue
worth investigating separately, but that is out of scope here.

## Files to Change

| File | Change |
|------|--------|
| `arrayjit/test/test_ndarray_binary_io.expected` | Prepend 2 log-level lines |
| `tensor/shape.ml` lines 1899–1901 | `Map.of_alist_exn` → `Map.of_alist_reduce ~f:(fun first _ -> first)` |

## Verification

After applying both fixes:

```
dune runtest
```

Should exit with code 0 with no diff output.

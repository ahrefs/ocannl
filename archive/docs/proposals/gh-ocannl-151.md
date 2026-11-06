# Proposal: Restore `Low_level.Fill`

GitHub issue: [ahrefs/ocannl#151](https://github.com/ahrefs/ocannl/issues/151)

## Preamble: Discarded

This proposal should not be implemented as written. `Zero_out` already captures the only currently
valuable special case: whole-buffer zeroing, which maps naturally to allocation zeroing,
declaration `= {0}`, byte-wise memset, CUDA memset, and Metal blit fill. General `Fill { value }`
would add a low-level IR case and more match arms without a measured non-zero bulk-fill use case.

If arbitrary non-zero constant fills later show up as a real bottleneck, revisit the design with
profiling and a concrete backend scheduling plan. Until then, keep `Zero_out` as the explicit
zero-specialized IR node and do not pursue gh-ocannl-151.

**Status, validated 2026-06-19**: Open. GitHub currently places the issue in milestone `v0.7`
with due date 2026-06-24. The current code still has `Zero_out` as the only bulk-initialization
IR node; there is no `Fill` variant, and non-zero constants still lower to explicit element loops.

## Summary

Restore a first-class low-level `Fill` instruction so constant tensor initialization remains a
single semantic operation until backend lowering. Today the compiler loses that intent:

- `Fetch { fetch_op = Constant 0.0 }` lowers to `Zero_out`.
- `Fetch { fetch_op = Constant c }` for non-zero `c` lowers to `loop_over_dims` plus `Set`.
- Reduction neutral initialization uses the same path, so common neutral values such as `1.0`,
  `infinity`, and `neg_infinity` are also expanded before the backend can choose a bulk strategy.

The first win is IR clarity: all constant fills become one operation. The second win is performance:
backends can map fills to their native mechanisms where that is actually available, while keeping a
correct loop fallback everywhere.

This proposal deliberately separates those two steps. Replacing `Zero_out` with `Fill` is a
compiler-IR change. Emitting `cuMemset*` or a Metal blit for runtime fills is a backend scheduling
change, because CUDA and Metal fills happen outside device kernel source; they cannot be obtained by
blindly printing `memset` inside `c_syntax.ml`.

## Why This Is Safe To Revisit

`Fill` was removed in commit `5e808c79` on 2023-05-09:

> Remove `low_level.Fill`, it has tricky semantics
> It was naively and wrongly ignoring `Parallel` / `Task_id`.

That warning was valid for the old IR. A whole-buffer fill inside a per-task parallel region could
clobber work owned by other tasks. The old low-level `Parallel`/`Task_id` execution model is no
longer present in `arrayjit/lib/low_level.ml`, `assignments.ml`, or `indexing.ml`. The current
high-level `Ocannl.Parallel` API is unrelated user-facing data-parallel orchestration; it is not the
removed low-level indexing construct.

The old removal rationale is therefore no longer load-bearing, but the replacement still needs a
clear rule: `Fill` is a whole logical tensor-node write. It must only be emitted where the current
code would already write the whole node by `Zero_out` or by a full `loop_over_dims` constant loop.

## Current Ground Truth

The following facts were checked against local HEAD on 2026-06-19:

- `arrayjit/lib/low_level.ml` and `.mli` define `Zero_out of Tnode.t`; there is no `Fill`.
- `assignments.ml` lowers `Fetch { Constant 0.0 }` to `Low_level.Zero_out array`.
- `assignments.ml` lowers non-zero `Constant c` to `Low_level.loop_over_dims ... Set(Constant c)`.
- `initialize_neutral` emits `Fetch { Constant neutral_value }` before accumulation when an init is
  needed, so neutral values use the same lowering.
- `c_syntax.ml` handles `Zero_out` by expanding it to a generated element loop.
- gh-ocannl-420 has already landed: the first function-scope `Zero_out` can be elided when a local
  declaration's `= {0}` already covers it. The relevant state is `zero_out_seen`,
  `zero_out_loop_redundant`, and `zero_initialized_by_code`.
- The shared slab API already exposes `memset_zero` in `backend_intf.ml`; `backends.ml` uses it for
  allocation-time zeroing.
- CUDA implements allocation-time zeroing with `Cu.Stream.memset_d8`.
- Metal implements allocation-time zeroing with `Me.BlitCommandEncoder.fill_buffer`.
- `backend_intf.buffer_loc = { pool_id; offset }` means backend bulk fill APIs should address a
  byte range inside a pool, not assume one allocation per tensor node.

## Goal

Represent every full-tensor constant initialization as:

```ocaml
| Fill of { tn : Tnode.t; value : float }
```

Then lower that instruction as efficiently as each backend can support:

- zero fills may use byte-wise bulk zeroing,
- non-zero fills may use typed loops or backend kernels,
- unsupported cases must keep the existing element-loop semantics.

`Constant_bits` is intentionally out of scope for the first pass. It can keep the current loop
lowering until there is a bit-pattern fill design that is correct for packed and non-floating
precisions.

## Non-Goals

- Do not redesign reduction initialization or projection inference.
- Do not use `memset` for arbitrary floating-point values. Byte-wise `memset` is only correct when
  the target byte pattern is intentionally repeated byte by byte, most importantly zero.
- Do not assume every `Fill` can be a host-side API call. Local arrays and virtualized computations
  still need generated code.
- Do not rework pool allocation. This proposal consumes the existing `{ pool_id; offset }` contract.

## Design

### 1. Replace `Zero_out` With `Fill`

Add `Fill of { tn : Tnode.t; value : float }` to `Low_level.t` and remove `Zero_out` rather than
keeping two spellings for the same write. The replacement is mechanical in most passes:

- read/write collection: `Fill` writes `tn` and reads nothing,
- substitution and scalar simplification: unchanged payload,
- structural equality: compare both `tn` and `value`,
- pretty-printers: print `fill <tn> <value>;`,
- constant validation: validate `value` against `tn.prec` exactly as `Constant value` is validated.

The one semantic caveat is tracing: keep the existing `zero_initialized_by_code` behavior only for
`Fill { value = 0.0 }`. Non-zero fills initialize memory, but they do not justify C declarations
with `= {0}` and do not let allocation skip zeroing for the same reason.

### 2. Lower Constants To `Fill`

In `assignments.ml`, change constant fetch lowering to:

```ocaml
| Fetch { array; fetch_op = Constant c; dims = _ } ->
    default_padding_before array @@ Low_level.Fill { tn = array; value = c }
```

This preserves the current padding order: padding reset code still runs before the full logical
fill, exactly as it does before `Zero_out` today. `Constant_bits` remains on the existing loop path.

### 3. Preserve gh-ocannl-420 First-Touch Elision

The `Zero_out` handler in `c_syntax.ml` currently has important first-touch logic:

- `zero_out_seen` distinguishes the first zero from a later re-zero,
- `zero_out_loop_redundant` checks whether a local declaration already emitted `= {0}`,
- zeroing inside loops or `Local_scope` bodies is never treated as redundant.

Generalize this logic only for `Fill { value = 0.0 }`. Later zero fills must still execute.
Non-zero fills are never declaration-elidable by this mechanism.

### 4. Backend Lowering Policy

Stage 1 should make every backend correct by lowering `Fill` through existing generated loops:

- zero `Fill` lowers to the same loop as `Zero_out` after first-touch elision,
- non-zero `Fill` lowers to the same typed loop that `assignments.ml` emits today,
- virtualized `Fill` becomes `Set_local (id, Constant value)`.

Stage 2 adds actual bulk operations where the backend execution model supports them:

- **CC / generated C**: zero fills for addressable buffers may use `memset(ptr, 0, bytes)`.
  Non-zero floating fills should remain typed loops unless a precision-specific bulk fill helper is
  added.
- **CUDA**: device memory zero fills should use a scheduled host-side backend operation, likely by
  extending the backend slab interface beyond allocation-time `memset_zero` or by introducing a
  fill task in the routine schedule. Do not emit `memset` inside CUDA kernel source.
- **Metal**: zero fills should use `Me.BlitCommandEncoder.fill_buffer` against the resolved
  `{ pool_id; offset; size }` byte range. Non-zero fills need a compute fill kernel or a typed loop
  fallback.

The important invariant is that bulk fill operates on the tensor node's resolved buffer range. With
pooled buffers, the target is not just `pool_id`; it is `pool_id + offset` for exactly
`Tnode.size_in_bytes tn`.

### 5. Allocation Interaction

Keep `zero_initialized_by_code` narrowly defined: it records a first write by generated code that
zeros the node. `backends.ml` currently uses it to skip allocation-time zeroing. That remains valid
for `Fill 0.0`, but not for non-zero `Fill`.

If a future backend schedules zero `Fill` as a separate `memset_zero` task, treat that task as the
code initialization for the same purpose. The skip must be per tensor node and per byte range; it
must not zero or skip neighboring tensors in the same pool.

## Acceptance Criteria

- `Low_level.t` has `Fill { tn; value }`; `Zero_out` is removed or reduced to a temporary migration
  alias with no remaining steady-state uses.
- `Fetch { fetch_op = Constant c }` lowers to `Fill` for all floating constants.
- `Fetch { fetch_op = Constant_bits _ }` remains correct and is explicitly left out of this change.
- Tracing, virtualization, CSE, equality, validation, read/write analysis, and pretty-printers all
  handle `Fill`.
- `Fill { value = 0.0 }` preserves gh-ocannl-420 first-touch elision.
- Non-zero `Fill` does not set `zero_initialized_by_code` and does not trigger declaration `= {0}`.
- Padding reset order is unchanged.
- All generated C/CUDA/Metal remains correct with loop fallback before backend-specific bulk fill
  optimizations are added.
- `.expected` files that print low-level IR use `fill ... 0;` instead of `zero_out ...;`.
- Existing tests pass with `OCANNL_BACKEND=sync_cc dune runtest`.

## Test Plan

Run targeted tests first:

```sh
OCANNL_BACKEND=sync_cc dune exec test/einsum/test_accumulation_semantics.exe
OCANNL_BACKEND=sync_cc dune exec test/operations/zero_out_local_decl.exe
OCANNL_BACKEND=sync_cc dune exec test/operations/test_slice_alias.exe
```

Then run the normal suite:

```sh
OCANNL_BACKEND=sync_cc dune runtest
```

Regenerate affected standalone `.expected` files from Dune output or `dune promote`, preserving the
two-line config lookup banner required by this repository's test convention.

## Implementation Checklist

- Update `low_level.ml` / `.mli` with `Fill`.
- Replace all `Zero_out` matches in `low_level.ml`.
- Update `assignments.ml` constant lowering.
- Generalize `c_syntax.ml` first-touch zero elision from `Zero_out` to zero-valued `Fill`.
- Update tests and expected outputs.
- Add or adjust focused tests for:
  - zero fill,
  - non-zero fill,
  - neutral initialization with a non-zero neutral value,
  - re-zero after a prior write,
  - local declaration elision from gh-ocannl-420.
- Only after the semantic migration is green, add backend bulk-fill tasks for CUDA and Metal.

## Open Questions

- Should `Fill` carry `float`, or should it eventually carry a richer constant type that can include
  `Constant_bits`? Start with `float` to keep the migration small.
- Where should backend runtime fills live: as an extension of `Backend_intf.Slab_alloc`, as a new
  task kind in the schedule, or as a backend-private procedure wrapper around compiled kernels?
  The answer determines how `cuMemset*` and Metal blits are sequenced with routine execution.
- For non-zero fills, is a generated typed loop sufficient, or do CUDA and Metal need dedicated fill
  kernels for common neutral values such as `1.0`?

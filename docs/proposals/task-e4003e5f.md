# Slice/sub-tensor as alias view (minimal copy): convert Fetch.Slice to buffer aliasing

**Task**: task-e4003e5f (subtask 293a of [gh-ocannl-293](gh-ocannl-293.md))
**Date**: 2026-06-12 (supersedes the 2026-04-25 harness elaboration)
**Status**: Blocked on [#344](https://github.com/ahrefs/ocannl/issues/344) (universal pool
allocator — see [gh-ocannl-344.md](gh-ocannl-344.md)); design ready otherwise.

## Goal

`Fetch { fetch_op = Slice _ }` in `arrayjit/lib/assignments.ml` currently materializes the
sliced sub-tensor via a set/get copy loop. Convert it to alias the parent's buffer at an
offset: the sub-tensor shares backing storage with its parent, and slicing performs no
copy.

## Why blocked on #344

The current `'buffer_ptr buffer = { ptr; size_in_bytes }` (`backend_intf.ml:7`, verified
2026-06-12) carries no offset; `buffer_ptr` is an opaque per-backend pointer. Without the
#344 refactor (tensors addressed as `(pool_id, byte_offset)` within pool buffers), every
backend would need ad-hoc offset bolted onto its pointer type (CUDA `CUdeviceptr +
size_t`, Metal `MTLBuffer + NSUInteger`, C `void* + size_t`). With #344, alias resolution
becomes one branch in the buffer-resolution layer: an alias is the parent's
`(pool_id, offset + delta)`.

**Constraint to flag into #344's design**: the pool allocator's free/reclaim pass must be
alias-aware — it must not reclaim a slab while alias entries reference it, and must skip
alias entries themselves (they don't own backing storage). If #344's implementation
doesn't account for this, this task effectively becomes a co-PR with it.

## Design

1. **Tnode metadata**: add `mutable alias_of : (Tn.t * int (* byte offset *)) option` to
   `Tnode.t` — *orthogonal* to `memory_mode`. An alias's mode (Hosted/On_device/...) is
   independently meaningful, and the `update_memory_mode` lattice is a careful state
   machine that shouldn't absorb an extra dimension. (Alternative: a context-side
   `Map.M(Tnode).t -> alias_info` side-table; costs locality, saves a word per tnode.)
2. **Lowering** (`assignments.ml`, the `Fetch {fetch_op = Slice {batch_idx; sliced}}`
   branch): when the slice is alias-eligible (parent materialized, leading-axis slice,
   compatible padding — see below), set `array.alias_of <- Some (sliced, offset)` and
   emit no copy loop; otherwise fall back to the existing materializing loop. The `Slice`
   constructor stays in the AST; only lowering changes.
3. **Buffer resolution** (`backends.ml` `alloc_if_needed`): new branch — if
   `key.alias_of = Some (parent, delta)`, ensure the parent is resolved in `ctx_arrays`
   (recurse), then enter the alias's buffer as parent's base + `delta` instead of calling
   `alloc_*`. Downstream `Map.find ctx_arrays tn` lookups stay uniform; no kernel-emit
   changes per backend, since all backends reach indexing through `low_level.ml`'s
   `Indexing.Affine` uniformly.
4. **Context accounting**: an alias is "in context" iff its parent is — adjust
   `is_in_context_force` / `context_nodes` in `assignments.ml` and
   `verify_prior_context` in the backends.
5. **Lifetime**: `alias_of` holds a strong OCaml reference to the parent tnode, and both
   entries live in the same context's `ctx_arrays`, so GC-level lifetime is already
   safe. The pool-allocator reclaim pass is the remaining hazard (see constraint above).

## Alias eligibility (fall back to copy otherwise)

- **Leading-axis slices only**: `Slice` slices axis 0 (`Iterator idx` is prepended), so
  the alias offset is `idx * stride_of_axis_0` on contiguous row-major storage.
  Non-leading or strided views are out of scope (293b-or-later territory).
- **Padding**: the lowering now wraps fetches in `default_padding_before` (2026-05
  padding work — this postdates the original elaboration and makes the hazard concrete).
  Alias only when parent and child padding are both absent or the child's padding
  matches the corresponding parent slab; otherwise materialize.
- **Virtual / Effectively_constant parents**: no backing buffer to alias — fall through
  to the materializing loop (it already handles this).
- **Merge-buffer RHS**: aliases should be ineligible as `Merge_buffer` sources —
  guard or materialize. Note `device_to_device` now returns a transfer routine, so the
  guard belongs in routine construction.

## Semantics change (open question, default chosen)

Today `Fetch (Slice _)` is a copy: writes through the slice do **not** mutate the parent.
After this change they would. Default: accept aliasing as the new documented semantics
(that is what "minimal copying" means), with a changelog note. The conservative
alternative — aliasing only as a compiler optimization when no write through the slice
can be proven — requires write-tracking and is not proposed here.

## Code pointers (verified 2026-06-12)

- `arrayjit/lib/assignments.ml:37` — `Slice of { batch_idx; sliced }` constructor;
  `:729` — the materializing lowering to replace.
- `arrayjit/lib/tnode.ml:22` — `memory_mode` variants; no alias metadata exists today.
- `arrayjit/lib/backend_intf.ml:7` — offset-less `buffer` type (the #344 gate); `:26` —
  `Alloc_buffer`.
- `arrayjit/lib/backends.ml` — `alloc_if_needed`, the resolution site.
- `arrayjit/lib/indexing.ml` — `Affine` already supports `Σ(coeff·sym) + offset`; no new
  indexing math needed.
- `tensor/operation.ml` — surface `slice` / `@|`; `lib/train.ml` — `rebatch` (callers,
  unchanged).
- `test/operations/check_slice_shapes.ml` — only existing end-to-end Slice test; it
  checks shapes/printouts, not copy-vs-alias.

## Acceptance Criteria

- [ ] Alias-eligible `Fetch.Slice` produces no copy loop in lowered code and no fresh
      allocation (assert via allocator instrumentation or generated-code inspection in
      `build_files/`).
- [ ] Mutation-visibility test: writing through a slice is observed via the parent (and
      vice versa). **Mandatory** — this is the observable semantics change.
- [ ] Ineligible slices (padding mismatch, virtual parent) still work via the fallback
      copy loop.
- [ ] `test/operations/check_slice_shapes` passes unchanged; `dune build && dune runtest`
      green on `sync_cc`/`multicore_cc`, Metal, and (where available) CUDA.

## Open questions

1. Confirm the gate: wait for #344, or absorb the minimal `buffer_ptr → (base, offset)`
   plumbing here? Default: stay blocked on #344 (the milestone chain v0.7 → v1.0 already
   implies the ordering).
2. `alias_of` as a `Tnode.t` field vs side-table (default: field).
3. Aliasing-as-semantics vs aliasing-as-optimization (default: semantics, documented).

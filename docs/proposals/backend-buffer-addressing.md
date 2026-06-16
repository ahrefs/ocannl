# Proposal: Backend buffer addressing — `(pool_id, offset)` locations, device-owned pools, and the allocator seam (with stream→device fold)

**Date**: 2026-06-16
**Status**: Proposed. Behavior-preserving refactor of `arrayjit/lib/backend_intf.ml` and the
backend layer. Foundation extracted from [gh-ocannl-344.md](gh-ocannl-344.md) (universal pool
allocator) after a design deep-dive; see "Relationship to other work". Companion to the dead-code
removal in [streams-cleanup.md](streams-cleanup.md).

## Summary

Two coupled changes to the backend interface, both **behavior-preserving** (no change to
generated code, allocation count, or device memory; only the *representation* of a buffer
handle changes, and backend signatures shrink):

1. **Fold `stream` into `device`.** With one compute stream per device (the end state of the
   first-class-streams removal), the surviving stream fields move onto the device and the
   multi-stream scaffolding goes away.
2. **Replace `buffer_ptr` with `buffer_loc = { pool_id : int; offset : int }`.** Per-tnode
   buffers become backend-agnostic integer locations resolved against a backend-private pool
   table. This drops the `'buffer_ptr` type parameter from the shared types and introduces a
   shared **allocator seam** that mints pool ids — initially with a trivial
   one-pool-per-tnode policy that is byte-for-byte equivalent to today's allocation.

The point is to land the *addressing contract* and the *interface simplification* on their
own — cheaply, with no behavior change — so that (a) pooling ([#344](gh-ocannl-344.md)) becomes
a policy swap behind the seam, (b) alias views ([task-e4003e5f.md](task-e4003e5f.md)) unblock on
just the contract, and (c) buffer handles become deterministic integers, which is a real
debuggability win for logs and `.expected` tests.

## Motivation

- **Debuggability / reproducibility.** Today a buffer is an opaque backend pointer
  (`Metal.Buffer.t` / `CUdeviceptr` / `void*`) that varies per run and sexps as noise. A
  `{ pool_id; offset }` of deterministically-assigned integers is stable across runs, diffable,
  and meaningful in logs and `.expected` files.
- **Interface simplification.** `'buffer_ptr` is threaded through `device`, `stream`, `context`,
  and `ctx_arrays` as one of four type parameters. The `finalize` signature in `backends.mli`
  already carries a note that "this type will get simpler with modular explicits." Making the
  per-tnode handle a monomorphic integer record removes that parameter entirely — no modular
  explicits required.
- **Foundation for pooling and alias views.** The `(pool_id, offset)` contract is exactly what
  the universal pool allocator (#344) and minimal-copy slicing (task-e4003e5f, subtask of #293)
  need. Landing it as a behavior-preserving refactor de-risks both: the expensive, behavior-
  changing work (bump arenas, Metal binding fix) is confined to a later, Metal-scoped step.
- **Completes the streams removal.** The stream→device fold is the positive counterpart to
  streams-cleanup.md's dead-code removal: once the multi-stream bookkeeping is gone, a "stream"
  is just the device's one compute queue, so its fields belong on the device.

## Background: why this is behavior-preserving

There are no peer compute streams to coordinate (one compute stream per device), and the shared
layer never *interprets* a buffer pointer — it only stores it, passes it to backend ops, and
logs it. Verified 2026-06-16:

- `link_compiled` (the binding/dispatch that actually dereferences pointers) is **backend-side**
  (`backend_impl.ml:219`, implemented in `cc_backend.ml:379`, `cuda_backend.ml`,
  `metal_backend.ml`). The shared `backends.ml` `link` (`:324-341`) just threads `ctx_arrays`
  through to it.
- The only shared-layer mentions of `buffer_ptr` are in `to_host` / `from_host` / `copy`
  (`backends.ml:57-217`), where the pointer is (a) stored in the map, (b) handed to a backend
  transfer op, and (c) `[%log]`-ged. None dereference it.

So replacing the pointer with an integer location and moving the pointer into a backend-private
table changes representation only. With the trivial one-pool-per-tnode allocator policy
(below), every tnode still gets its own slab at offset 0, so memory, allocation count, and
generated code are identical to today.

## Design

### 1. Fold `stream` into `device`

Current (`backend_intf.ml:92-164`): a per-device `device_ref` holding `constant_buffer_cache`
and a `streams` weak-dynarray; a per-stream `stream_ref` holding `runner`, `merge_buffer`,
`stream_id`, `allocated_buffer`, `updating_for`, `updating_for_merge_buffer`; and a `context`
whose `stream` field points at a `stream_ref`.

After the fold:

```
type ('dev, 'runner, 'event) device = {
  dev : 'dev;  ordinal : int;  device_id : int;
  runner : 'runner;                                    (* was stream *)
  merge_buffer : buffer_loc option ref;                (* was stream; see §reserved pool *)
  updating_for : 'event Hashtbl.M(Tnode).t;            (* was stream — writer events *)
  updating_for_merge_buffer : (Tnode.t * 'event option) option;  (* was stream *)
  pools : pool_table;                                  (* NEW — backend-private base table, §2 *)
  constant_buffer_cache : buffer_loc Hashtbl.M(Tnode).t;  (* loc, not ptr *)
  (* gone: streams dynarray, stream_id, the stream.device back-pointer *)
}
and ('dev, 'runner, 'event, 'oc) context = {
  device : ('dev, 'runner, 'event) device;            (* was: stream *)
  parent; ctx_buffers; finalized; optimize_ctx; merge_buffer_node;
}
```

- **Coherence is preserved by relocation, not rewrite.** `updating_for` (per-tnode writer
  events), `merge_buffer`, and the `'event` machinery move from `stream` to `device`
  verbatim. They are still needed — not for intra-device cross-stream coherence (which is gone)
  but for **cross-device coherence**: `device_to_device` waits on the producer device's
  most-recent-write event for the tnode and on the transfer-completion event. With one compute
  runner per device, `updating_for` keyed on the device is exactly right.
- **Forward-compatible with prefetch.** A future automatic prefetch/transfer stream is just a
  second `runner` (+ its own staging slot) on the device — the fixed-role model, not a return
  to open-ended multi-streaming.
- **Doc fix.** Update the `'event` comment (`backend_intf.ml:75-77`) from "across streams" to
  "across devices/queues."

This overlaps streams-cleanup.md's "merge stream into device" item (its AC 3); treat that item
as realized here. The dead-code removals streams-cleanup.md lists (config type, sharing
variants, `round_robin`, `suggested_num_streams`) are **out of scope here** and remain its
mandate.

### 2. `buffer_ptr` → `buffer_loc`, with `'base` behind the backend

```
type buffer_loc = { pool_id : int; offset : int } [@@deriving sexp, compare, equal]
type ctx_buffers = buffer_loc Map.M(Tnode).t       (* was ctx_arrays : 'buffer_ptr ctx_arrays *)
```

- **Size is not in the record** — it comes from `tn.size_in_bytes` (`tnode.ml:57`). A pool's
  total size lives in the allocator's pool table (§3), not per tnode.
- **The backend pointer relocates, it does not vanish.** Today `ctx_arrays : tnode → buffer_ptr`
  holds N backend pointers. After: `ctx_buffers : tnode → buffer_loc` is pure integers, and the
  backend owns a private `pool_id → 'base` table (the `pools` field on `device`). Pointer
  cardinality drops from N tnodes to K pools, and **`'base` no longer appears in any shared
  type** — the `'buffer_ptr` parameter leaves `device`/`context`/`ctx_buffers`.
- **`pool_id` is device-local.** A context is on one device; cross-device `copy` resolves each
  side against its own device's `pools`. No global `(device, pool_id)` key is needed.
- **Naming.** `buffer_loc` supersedes #344's proposed `buffer_offset` (which implied
  offset-only). `ctx_arrays → ctx_buffers` is the long-pending rename (it was #344 AC 3); it
  rides this type change because every reference site is edited anyway.
- **Aliases fall out for free** (needed by task-e4003e5f): an alias is
  `{ pool_id = parent.pool_id; offset = parent.offset + delta }` — the same record.

### 3. The allocator seam (shared) and the backend slab API

The offset/bump arithmetic is backend-agnostic and id minting must be deterministic, so the
allocator is a **shared** module that *calls* backend slab primitives — not a per-backend
functor. The backend's only job is to vend slabs and resolve addresses.

**Backend API (int-in / int-out)** — replaces the per-tnode `Alloc_buffer` module type
(`backend_intf.ml:26-41`: `alloc_buffer` / `alloc_array` / `alloc_zeros` / `free_buffer` are
all removed):

```
val alloc_pool  : dev -> pool_id:int -> size_in_bytes:int -> ?mode:Tnode.memory_mode -> alignment:int -> unit
val free_pool   : dev -> pool_id:int -> unit          (* drops table entry + frees slab *)
val memset_zero : dev -> pool_id:int -> offset:int -> size_in_bytes:int -> unit
(* dispatch resolution stays inside link_compiled — same backend-side locus as today;
   only its argument changes: ctx_buffers of locs instead of ctx_arrays of ptrs. The backend
   looks up pool_id → 'base in its own table and binds base + offset. *)
```

The `?mode` argument migrates from per-tnode alloc to `alloc_pool` (storage-mode segregation is
a per-pool property — exactly where Metal wants it).

**Shared seam.** A module owning a deterministic per-device `pool_id` counter and the
`ctx_buffers` entries:

```
allocate ctx tn =
  let id = fresh_pool_id ctx.device in
  Backend.alloc_pool ctx.device ~pool_id:id ~size_in_bytes:(Lazy.force tn.size_in_bytes) ~mode ~alignment;
  { pool_id = id; offset = 0 }
```

This replaces the `alloc_array`/`buffer_ptr` step inside `alloc_if_needed`
(`backends.ml:489-524`). **Phase-1 policy is one-pool-per-tnode, offset ≡ 0** — behavior-
identical to today. The shared-side pool-state table (`pool_id → { total_size; bump_offset;
mode }`) is **not** built here; it is purely additive and lands with the bump policy in #344.

- **Determinism.** The counter increments in `alloc_if_needed`'s existing deterministic tnode
  order, so pool ids and offsets are reproducible — the debuggability win. Ownership of the id
  space stays in the shared layer for exactly this reason (a backend-minted id risks
  hash/alloc-order nondeterminism).
- **Conformance test.** In this policy `pool_id` is in 1:1 correspondence with the old
  `buffer_ptr` (it indexes the same single slab, offset 0). That both this policy and the future
  bump-arena policy sit behind the identical seam + contract is the proof the seam is placed
  correctly.

### 4. Merge buffer as a reserved pool

The per-device `merge_buffer` becomes a **reserved single-tenant pool** addressed uniformly as
`{ pool_id = <reserved>; offset = 0 }`. This keeps merge buffers out of any future arena (they
need pointer stability across transfer-routine invocations) while still expressing them in the
one addressing scheme. `merge_buffer_node` and the transfer path are otherwise unchanged.

### 5. Backend-specific touch-ups

- **Metal log table re-key.** `metal_backend.ml`'s `stream_logs` hashtable (`:161`) is keyed by
  `stream_id` purely as a handle for its async `MTLLogState` callback's captured-log ref — not
  for scheduling. Re-key it on `device_id`, and move the GC finalizer (`:223`) from stream
  finalization to device finalization. CUDA/cc need no change (CUDA captures kernel `printf` at
  process stdout; cc has no per-stream log table).
- **Test inspection hook.** `metal_backend.mli:8` currently exposes `buffer_ptr = Metal.Buffer.t`
  "so tests can inspect the storage mode." Replace that leaked type with a targeted
  `storage_mode_of_pool : dev -> pool_id -> mode`. Storage mode is now a per-pool property, so
  this is the better surface anyway.
- **Existential `Wrapper`** in `context.ml:11-23` sheds its `'buffer_ptr` member; `Context.t`
  stays abstract and non-parametric — **no user-facing API change** (`buffer_ptr` was never
  user-facing; it lived behind `Context.t`'s existential and the `Backends_deprecated` alias).

## Acceptance Criteria

1. `stream` is folded into `device`: `device.streams` and `stream_id` are gone; `runner`,
   `merge_buffer`, `updating_for`, `updating_for_merge_buffer` live on `device`;
   `context.stream` becomes `context.device`.
2. The `'event` writer-tracking and `device_to_device` cross-device coherence are preserved
   (relocated onto `device`), with the doc comment updated to "across devices/queues."
3. `buffer_ptr` is replaced by `buffer_loc = { pool_id : int; offset : int }`; `ctx_arrays` is
   renamed to `ctx_buffers : buffer_loc Map.M(Tnode).t`.
4. `'base`/`'buffer_ptr` appears in **no** shared type; the backend owns a private
   `pool_id → 'base` table on its `device`. The `('buffer_ptr, …)` parameter is dropped from
   `device`/`context` signatures.
5. The per-tnode `Alloc_buffer` interface (`alloc_buffer`/`alloc_array`/`alloc_zeros`/
   `free_buffer`) is replaced by the slab API (`alloc_pool` / `free_pool` / `memset_zero`),
   int-in/int-out.
6. A shared allocator seam mints deterministic per-device `pool_id`s; its phase-1 body is the
   one-pool-per-tnode policy. Generated code, allocation count, and device memory are unchanged
   from before the refactor.
7. The merge buffer is a reserved single-tenant pool; `constant_buffer_cache` stores
   `buffer_loc`.
8. Metal's `stream_logs` is re-keyed on `device_id`; `metal_backend.mli` exposes
   `storage_mode_of_pool` instead of a concrete `buffer_ptr`.
9. `Context.t` remains abstract and non-parametric; no user-facing signature changes.
10. All existing tests pass; backend `.expected` outputs change only where buffer handles are
    printed (opaque pointer → deterministic `{ pool_id; offset }`), not in generated code.

## Relationship to other work

- **streams-cleanup.md** — sibling/prerequisite. It removes multi-stream *dead code* (config
  type, `sharing` variants, `round_robin`, `suggested_num_streams`, residual hashtables); this
  proposal does the *positive* consolidation (fold surviving fields into `device`) plus the
  buffer-addressing contract. Land streams-cleanup first or together; this realizes its "merge
  stream into device" item.
- **gh-ocannl-333 (remove hosted memory)** — adjacent; both simplify the backend/memory layer
  and can land around the same time. No hard ordering with this proposal.
- **gh-ocannl-344 (universal pool allocator)** — *downstream*. After this lands, the bump-arena
  policy + Metal binding-limit fix is a **policy swap behind this seam**: a new `allocate` body
  (per-context arena + per-device constant pool) plus Metal in-shader `pool_base + offset`,
  touching neither the `buffer_loc` contract nor the slab API nor CUDA/C generated code. #344 is
  trimmed to that capability and depends on this proposal.
- **task-e4003e5f (slice as alias view, subtask 293a)** — *downstream, unblocked earlier by
  this split*. Alias views need only the `buffer_loc` addressing contract
  (`{ pool_id; offset + delta }`), **not** the bump pooling. Its "blocked on #344" gate is
  re-pointed to this proposal's contract.

## Out of scope (these are gh-ocannl-344 or later)

- Bump-arena allocation policy (per-context working arena, per-device constant pool, append-bump
  on exhaustion) and the shared-side pool-state table.
- Metal binding-limit fix (bind pools once, offsets as runtime args, in-shader `pool_base +
  offset`), Metal untracked hazard mode, storage-mode pool segregation.
- `large_models` / uint32-vs-uint64 offset width.
- Liveness-based offset reuse and buffer eviction (the "thick allocator").
- The dead-code removals owned by streams-cleanup.md.

# Backend singletons, wrapped contexts, and Context.copy

**Date**: 2026-07-06
**Status**: Implemented (branch `backend-singletons`).

## Motivation

`Context.copy` was a `failwith` stub, and could not be otherwise: `Backends.fresh_backend`
applied the backend functors **per call** (generatively for CUDA/Metal via `Fresh ()`), and
`Context.t` hid the result behind an existential wrapper. Two independently created contexts on
the same backend therefore had genuinely *distinct* backend-context types — no amount of
matching inside `copy` could recover an equality that did not exist. This also duplicated
per-instance device state: each root context re-enumerated devices and kept its own
`constant_buffer_cache` for the same physical device (while Metal's pool table was already
module-global, so the per-call isolation was partial to begin with).

## Why per-call functors existed, and what replaces them

The generative design followed a multiday bug hunt: a tnode-keyed backend cache leaked between
tests, and `Tensor.unsafe_reinitialize` resets the tnode id counter, so a stale cache entry
*aliased* a fresh tnode that reused its id. Discarding the whole backend module per
`fresh_backend` call cured the backend instances of that bug class — at the cost of nameable
types — but could not cure the class itself (any tnode-keyed table elsewhere that outlives a
reinitialization had the same bug; `Context`'s id-keyed for-print proxy table was a live
instance).

The root-cause fix makes the workaround unnecessary: tnode identity is now a hidden
process-unique `Tnode.uid` that no reinitialization resets; the presentational `id` still
restarts at 0 for deterministic printing. Every tnode-keyed map, set and cache in the process
keys on `uid`, so stale entries can never alias fresh nodes — they are just unreachable garbage.
Correctness no longer depends on which caches get wiped. (Residual concern is memory only:
singleton caches like `constant_buffer_cache` now persist across reinitializations within a
process; if that ever matters, a non-load-bearing clear-hook registry is the remedy.)

## The design

One instantiation per backend for the whole process, at the top of `backends.ml`:

- `Sync_cc_b`, `Multicore_cc_b`, `Cuda_b`, `Metal_b` — `Raise_backend` /
  `Make_device_backend_from_lowered` applied once; the impl-level `Fresh ()` functors became
  plain `Impl` modules. Instantiation must not touch drivers or hardware, so device
  discovery/driver init stays **lazy inside the impls**, forced at first `get_device` (PR #94
  review): cudajit is a depopt — the library being installed does not imply a usable driver —
  and a CPU-only run must not depend on GPU runtimes (`static_properties` became `unit ->
  Sexp.t` for the same reason). On platforms without the corresponding library the
  dune-`select`ed `Lowered_backend_missing` stub is instantiated instead — harmless at module
  init, raising on use. Either failure mode surfaces at first device use, which is exactly where
  `Context.auto`'s try-in-order fallback catches it, matching the retired per-call
  `fresh_backend` semantics.
- `type backend = Sync_cc | Multicore_cc | Cuda | Metal` with `get_backend` (non-generative
  successor of `fresh_backend`; same config-driven name resolution) and `backend_module` (the
  erased first-class-module view, for consumers that thread a single backend through and never
  need to re-correlate contexts: the raw-API tests, `Parallel`).
- The central piece, a **closed disjunction over context types** — one variant, no existential:

  ```ocaml
  type wrapped_context =
    | Sync_cc_ctx of Sync_cc_b.context
    | Multicore_cc_ctx of Multicore_cc_b.context
    | Cuda_ctx of Cuda_b.context
    | Metal_ctx of Metal_b.context
  ```

  Matching two values on the same constructor recovers type equality directly, which is what
  lets `Context.copy` fall onto the backend-specific `device_to_device` when both contexts come
  from the same backend, and lets contexts created by *independent* roots unify (the singleton
  types are the same).

`Context.t` holds a `wrapped_context`. Generic operations need no per-backend code: read-only
paths use `Backends.query` and context-updating paths use `Backends.with_backend` — higher-rank
dispatch helpers written once (4 match arms each) that pass the singleton module plus the
concrete context to a polymorphic callback and rebuild the same constructor around the result.
Since `Backend_intf.context` is a transparent polymorphic record, fields like `ctx_buffers`,
`optimize_ctx`, `merge_buffer_node` are readable inside the callbacks without any dispatch
beyond the helper's.

`Backend_intf` and `Backend_impl` module types are **unchanged**: the variant layer is purely
additive on top of them. In particular `Context.copy` consumes the existing
`Backend.device_to_device` (returning a transfer routine whose linked context carries
`merge_buffer_node`), so the gh-ocannl-288 static verification story is inherited, not
re-implemented.

## Context.copy semantics

`copy ?into_merge_buffer ~src ~dst tn` returns the updated destination context:

- **Same backend** (same constructor): dispatch to `Backend.device_to_device`. `Some r` — run
  the transfer schedule (ordered on `dst`'s stream; reads await via `to_host` as usual) and
  rewrap `r.context` into `dst` (for `~into_merge_buffer:Copy` this is what carries
  `merge_buffer_node = Some tn` into the next compile's static merge-node check). `None` with
  the node present in `src` but absent in `dst` — `init_from_device` (allocate + copy). `None`
  with the node absent from `src` — fall back to the host round-trip, which serves host-init
  literals and for-print proxies; under `Copy` this raises instead (a merge buffer cannot be
  filled host-side).
- **Cross backend**: host round-trip (`from_host dst (to_host src)`); `Copy` raises.

## What this unblocks (deferred)

- Port `lib/parallel.ml` and the raw-API tests (`merge_buffer_static_verification`,
  `shard_transfer`, `test_buffer_loc`, `test_cuda_pool_offset`) off the erased Backend API onto
  Context-level transfers; needs a Context-level notion of "same device, distinct stream"
  contexts.
- Then demote `Backend_intf.routine` to an internal link-result and hoist
  `ctx_buffers`/`merge_buffer_node`/`finalized` onto `Context.t` — mechanical once nothing
  public consumes them.
- If cross-process cache memory growth ever matters: clear-hook registry invoked by
  `Tensor.unsafe_reinitialize` (memory hygiene only; correctness is covered by `uid`).

## Relations

[context-scoped-memory-modes](../../../docs/design/context-scoped-memory-modes.md) (contexts-as-values groundwork;
the optimize_ctx that `make_context` seeds), gh-ocannl-288 (merge-buffer static verification
that `copy` preserves), gh-ocannl-333 (on-demand host access that the fallback path reuses).

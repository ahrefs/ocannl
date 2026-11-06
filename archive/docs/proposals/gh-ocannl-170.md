# CUDA pinned host transfers

Issue: [#170](https://github.com/ahrefs/ocannl/issues/170)

## Current state

CUDA host transfers now sit behind explicit `Context.to_host`/`from_host`
operations and the slab allocator. `cuda_backend.ml` passes ordinary Bigarrays
to cudajit's asynchronous copy functions. OCANNL has no pinned-host allocation
API, and the cudajit dependency must own any binding and lifetime-safe Bigarray
wrapper.

The old plan proposed an untyped `void *` staging buffer per stream. That would
add an extra CPU copy to every transfer and makes D2H completion/lifetime easy
to get wrong. It should not be the default design.

## Goal

Determine whether pinned host buffers materially improve large CUDA transfers,
then expose the smallest safe mechanism that wins.

## Direction

1. Add or use a cudajit abstraction that allocates a GC-rooted Bigarray backed
   by pinned memory (or safely registers an existing Bigarray). The abstraction
   must make ownership and freeing explicit.
2. Benchmark pageable, directly pinned, and—only if necessary—staged transfers
   across representative sizes and repeated transfers.
3. Integrate an opt-in pinned path at the `Context`/CUDA boundary. Preserve the
   pageable path for small or one-off transfers and systems with tight pinned
   memory limits.
4. Keep source buffers alive until the queued copy completes; D2H data must not
   be copied out of staging before the stream event completes.

## Completion criteria

- cudajit has a tested, lifetime-safe pinned host buffer API.
- Benchmarks report bandwidth and latency, including the extra CPU-copy cost
  of staging.
- OCANNL uses pinned memory only where measurements justify it and has a
  bounded/clear release policy.
- CUDA transfer and persistence tests pass through both the ordinary and
  pinned paths.

If direct pinned Bigarrays do not beat the current path for OCANNL's transfer
shapes, close the issue with the measurements rather than carrying speculative
buffer machinery.

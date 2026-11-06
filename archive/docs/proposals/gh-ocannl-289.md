# Make memory reporting for CUDA more meaningful

## Goal

Replace the CUDA backend's `cu_mem_get_info`-based memory reporter with
per-allocation tracking, so the number returned by `get_used_memory` reflects
**bytes allocated by OCANNL on this device**, not the system-wide GPU memory
pressure.

Source: https://github.com/ahrefs/ocannl/issues/289

Today, `Cuda_backend.get_used_memory` returns `total - free` from
`Cu.Device.get_free_and_total_mem ()`, which is the entire GPU device's
occupancy (every CUDA process on the system, plus driver overhead). That makes
the benchmark table from the issue report numbers in the billions of bytes,
while the CC backend reports thousands for the same workload. The two are not
comparable, which defeats the purpose of having `get_used_memory` exposed via
the `Backend_intf.Backend_device_common` signature in the first place.

## Acceptance Criteria

- [ ] `Cuda_backend.get_used_memory` no longer calls
  `Cu.Device.get_free_and_total_mem`. It returns the running total of bytes
  OCANNL has allocated through the backend's allocator, less the bytes
  reclaimed via `mem_free` / GC finalisation.
- [ ] Memory numbers reported for a CUDA run are in the same order of
  magnitude as CC backend numbers for the same workload (i.e. roughly the sum
  of `prec_in_bytes * dims_product` over the live tnodes, plus any auxiliary
  buffers like merge buffers and constant caches).
- [ ] Existing CUDA tests / benchmarks continue to build and run on Nvidia
  hardware. (Verification is manual on the user's RTX 3050 desktop —
  `dune runtest` does not exercise CUDA.)

## Context

### The interface, and how peer backends already implement it

`Backend_intf.Backend_device_common` declares:

```ocaml
val get_used_memory : device -> int
(** Returns (an upper bound of) the memory used for arrays, in bytes. *)
```

Two of the three real backends already follow the same convention — increment
an `Atomic.t` counter on every allocation, decrement via `Gc.finalise` on the
buffer object:

- **CC backend** (`backend_impl.ml` `No_device_buffer_and_copying`): module-level
  `let used_memory = Atomic.make 0`; `alloc_impl ~size_in_bytes` does
  `Atomic.fetch_and_add used_memory size_in_bytes` *and* attaches a
  `Gc.finalise` that subtracts the same number when the buffer pointer is
  collected. `get_used_memory ()` just reads the atomic. The
  `For_add_scheduler` exposure is plumbed through `schedulers.ml`'s
  `get_used_memory _device = get_used_memory ()`.
- **Metal backend** (`metal_backend.ml`): `let allocated_memory = Atomic.make 0`
  at module scope; a `track_allocation` helper called from `alloc_buffer`,
  `alloc_array`, `alloc_zeros` does the increment plus
  `Gc.finalise` decrement. `get_used_memory _device = Atomic.get
  allocated_memory`. `static_properties` even surfaces the value.

The CUDA backend (`cuda_backend.ml`) is the odd one out: its
`Alloc_buffer` module allocates via `Cu.Deviceptr.mem_alloc` /
`Cu.Deviceptr.mem_free` without bookkeeping, and its `get_used_memory` reads
the device-wide free/total instead.

### Allocation sites in `cuda_backend.ml`

All `Cu.Deviceptr.mem_alloc` / `mem_free` calls happen inside the file:

- `Alloc_buffer.alloc_buffer` — sized buffer alloc, with optional in-place
  free of the previous buffer when growing.
- `Alloc_buffer.alloc_array` — alloc by `prec * dims`.
- `Alloc_buffer.alloc_zeros` — alloc by `prec * dims`, then `memset_d8`.
- `Alloc_buffer.free_buffer = Some (fun _stream ptr -> Cu.Deviceptr.mem_free ptr)`
  — the explicit free path (called by the framework, e.g. on context
  finalisation).
- `opt_alloc_merge_buffer` (in `Fresh ()`) — frees the old merge buffer if any,
  allocates a new one of `size_in_bytes`.
- `finalize_device` — iterates `device.constant_buffer_cache` and frees each
  buffer pointer.

These six sites cover every byte that flows through `mem_alloc` /
`mem_free` in the backend.

### `Cu.Deviceptr.t` and finalisers

`Cu.Deviceptr.t` is a value type wrapping a CUDA pointer; the OCaml runtime
does not have a finaliser on it by default (unlike Metal's
`Me.Buffer.t`, which is GC-tracked through ARC). So the Metal trick of "attach
a finaliser to the buffer object" works there because the buffer object is
itself a heap value. For the CUDA backend, we must rely on the **explicit**
free paths (`Alloc_buffer.free_buffer`, `opt_alloc_merge_buffer`,
`finalize_device`) to decrement the counter, and accept that pointers leaked
past those paths will leave the counter slightly inflated. That matches the
"upper bound" wording in the docstring of `get_used_memory`.

This is why approach (1) from the task elaboration — track allocations
manually — is the right choice for CUDA: approach (2) would require
iterating over per-device live tnodes, a layer the backend does not currently
own, while approach (1) is a five-line change that mirrors the CC and Metal
shape exactly.

### Per-device vs aggregated

Both peer backends use a single module-level atomic and ignore the `device`
argument. The CUDA backend should do the same for parity. If per-device
breakdown is desired later, that is a follow-up that touches the interface
(currently `device -> int`) and all three backends symmetrically — out of
scope here.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

In `arrayjit/lib/cuda_backend.ml`:

1. Add a module-level counter near the top, alongside `initialized_devices`:

   ```ocaml
   let allocated_memory = Atomic.make 0
   ```

2. Add small helpers (private to the file) that wrap `mem_alloc` / `mem_free`
   and adjust the counter:

   ```ocaml
   let mem_alloc_tracked ~size_in_bytes =
     let ptr = Cu.Deviceptr.mem_alloc ~size_in_bytes in
     ignore (Atomic.fetch_and_add allocated_memory size_in_bytes : int);
     ptr

   let mem_free_tracked ~size_in_bytes ptr =
     Cu.Deviceptr.mem_free ptr;
     ignore (Atomic.fetch_and_add allocated_memory (-size_in_bytes) : int)
   ```

   The free helper requires a known size; for the explicit paths in
   `Alloc_buffer.alloc_buffer` (resizing case) and `opt_alloc_merge_buffer`,
   the framework already keeps the old buffer's `size_in_bytes` in scope.

3. Route every `mem_alloc` / `mem_free` site listed above through the tracked
   helpers. For `alloc_array` / `alloc_zeros`, `size_in_bytes` is computed
   inline; for the explicit free in `Alloc_buffer.free_buffer`, the size is
   not currently passed in, which means the framework's `free_buffer`
   signature `stream -> buffer_ptr -> unit` does not surface it. Two options:

   a. Track sizes in a side table keyed by raw pointer (hash on
      `Cu.Deviceptr.string_of`), populated on alloc and consulted on free.
      Slightly more code, fully accurate.

   b. Disable `free_buffer` by setting it to `None` (Metal does this) and
      rely on the framework's higher-level allocator-reuse pattern plus
      the explicit decrement at the resize and merge-buffer paths to keep
      the counter approximately correct. Look at how the framework actually
      drives `free_buffer` before choosing this — if it's only called
      in narrow contexts (e.g. context finalisation), option (b) may
      diverge.

   The implementor should pick (a) unless inspection of the framework's
   `free_buffer` callers shows option (b) is fine. Option (a) is preferable
   because it preserves an existing feature flag.

   For `finalize_device`'s iteration over `constant_buffer_cache`, the same
   side table (or a parallel size cache stored alongside the cache entries)
   gives the size to subtract.

4. Replace `get_used_memory`:

   ```ocaml
   let get_used_memory _device = Atomic.get allocated_memory
   ```

   The `set_ctx` call and `Cu.Device.get_free_and_total_mem` go away.

5. Optional polish: surface `allocated_memory` in `static_properties` as Metal
   does (`Sexp.Atom (Int.to_string (Atomic.get allocated_memory))` under an
   `allocated_memory` key) so debug dumps show it.

6. Optional polish: keep the system-level reporter under a different name
   (e.g. `get_total_gpu_memory`) if any user code wants the raw
   `cu_mem_get_info` figure. The issue does not request this, so it is
   acceptable to drop it entirely; revisit if a downstream call site shows up.

### Verification

- `dune build && dune runtest` from `~/ocannl-staging` — must pass.
  `dune runtest` does not exercise CUDA, so the build is the main check.
- Manual on the RTX 3050 desktop: run a benchmark that previously reported
  the billions-of-bytes number from the issue's table and confirm the new
  number is in the same order of magnitude as CC's report for the same
  workload. The exact benchmark binary is not pinned in the issue;
  `bin/`, `bench/`, and `test/` should be scanned for one that exercises the
  CUDA backend and prints `get_used_memory`. If none is obvious, a small
  ad-hoc snippet that allocates a known number of bytes and reads back
  `get_used_memory` is sufficient — that's a unit-of-accounting check.

## Scope

**In scope**:

- The five-or-six allocation/free sites in `arrayjit/lib/cuda_backend.ml`.
- The `get_used_memory` function in the same file.
- Optional `static_properties` surfacing.

**Out of scope**:

- Pool allocator integration (gh-ocannl-344, currently deferred). When that
  lands, "still in pool but not in use" buffers will count as allocated by
  the OS and by this counter — that's a separate accuracy question.
- Per-device or per-stream breakdown of the counter.
- Any change to `Backend_intf.Backend_device_common.get_used_memory`'s
  signature or to the CC / Metal backends.
- Any `cuMemAllocHost` / pinned-memory reporting (gh-ocannl-170 territory).
- Any new public API; the existing `get_used_memory` is the contract.

**Dependencies**: none. Self-contained inside `cuda_backend.ml`.

**PR destination**: `ahrefs/ocannl`, branch off `master`.

**Hardware**: implementation is doable without a GPU; final verification of
the "comparable to CC" criterion needs the RTX 3050 desktop.

# Proposal: Use `cuMemAllocHost` for pinned staging buffers in CUDA backend

Task: gh-ocannl-170
Issue: https://github.com/ahrefs/ocannl/issues/170

## Goal

Speed up CPU-GPU memory transfers in the CUDA backend by using pinned (page-locked) host memory for staging buffers. Pinned memory enables DMA transfers and can be 2-3x faster for host-device copies. Following the CUDA documentation's warning, pinned memory is used sparingly -- only for reusable staging buffers, not for all bigarrays.

## Acceptance Criteria

- A per-stream pinned staging buffer is allocated via `cuMemAllocHost` in the CUDA backend, used as an intermediary for `from_host` and `to_host` transfers.
- The staging buffer is grown lazily (allocated on first use, reallocated when a larger transfer is needed) and freed when the stream is finalized.
- Regular host-side bigarrays (created by `Ndarray.create_array`) remain unpinned -- no changes to `Ndarray` module.
- `cuMemAllocHost` and `cuMemFreeHost` bindings are added to `ocaml-cudajit` (in `cuda_ffi/bindings.ml`, `src/cuda.ml`, `src/cuda.mli`).
- Transfers via the staging buffer are correct: data round-trips through `from_host` -> kernel -> `to_host` produce identical results to the current implementation.
- All existing CUDA tests pass with no regression.
- The feature can be disabled via an environment variable or global setting for debugging purposes (e.g., `OCANNL_NO_PINNED_STAGING=1`).

## Context

### Current transfer path

Host-device transfers in the CUDA backend go through `Cu.Stream.memcpy_H_to_D` and `Cu.Stream.memcpy_D_to_H`, which are async wrappers around `cuMemcpyHtoDAsync` and `cuMemcpyDtoHAsync`. These operate on regular (pageable) host memory from OCaml bigarrays created via `Bigarray.Genarray.create`.

When source host memory is pageable, the CUDA driver must internally copy it to a pinned staging buffer before DMA transfer. This double-copy is the performance bottleneck this task eliminates.

### Key code locations

| Component | File | Line(s) | Relevance |
|-----------|------|---------|-----------|
| `from_host` | `arrayjit/lib/cuda_backend.ml` | 258-261 | H2D transfer: calls `Cu.Stream.memcpy_H_to_D` with bigarray |
| `to_host` | `arrayjit/lib/cuda_backend.ml` | 263-266 | D2H transfer: calls `Cu.Stream.memcpy_D_to_H` with bigarray |
| `alloc_if_needed` | `arrayjit/lib/backends.ml` | 470-559 | Initial H2D for constants; calls `Device.from_host` |
| `Ndarray.create_array` | `arrayjit/lib/ndarray.ml` | 470-482 | Host array creation via `Bigarray.Genarray.create` (unchanged) |
| Device buffer alloc | `arrayjit/lib/cuda_backend.ml` | 58-88 | `Alloc_buffer` module: `mem_alloc`, `alloc_zeros`, `free_buffer` |
| Stream finalization | `arrayjit/lib/cuda_backend.ml` | 130-135 | `finalize_device`: frees cross-stream buffers |
| Merge buffer pattern | `arrayjit/lib/cuda_backend.ml` | 121-128 | `opt_alloc_merge_buffer`: lazy grow pattern (model for staging buffer) |
| cudajit `Deviceptr` | `ocaml-cudajit/src/cuda.ml` | ~1489 | `mem_alloc` / `mem_free`: pattern for new `mem_alloc_host` / `mem_free_host` |
| cudajit FFI bindings | `ocaml-cudajit/cuda_ffi/bindings.ml` | 43-48 | `cu_mem_alloc` / `cu_mem_alloc_async`: pattern for new binding |

### Dependency: ocaml-cudajit bindings

`cuMemAllocHost` and `cuMemFreeHost` are **not yet bound** in ocaml-cudajit. The gh-ocaml-cudajit-5 task (broader pinned memory API: `Hostptr` module, `cuMemHostAlloc`, `cuMemHostRegister`, etc.) was planned but its code has not landed in the ocaml-cudajit main branch. This task requires a minimal subset: just `cuMemAllocHost` and `cuMemFreeHost`.

### Merge buffer as design precedent

The existing `opt_alloc_merge_buffer` in `cuda_backend.ml` (line 121-128) demonstrates the lazy-grow pattern for per-stream device-side buffers. The pinned staging buffer follows the same pattern on the host side: allocate on first use, grow when needed, free on finalization.

## Approach

### Step 1: Add minimal `cuMemAllocHost`/`cuMemFreeHost` bindings to ocaml-cudajit

**File: `ocaml-cudajit/cuda_ffi/bindings.ml`**

Add two new FFI bindings after the existing `cu_mem_free_async` (line 112):

```ocaml
let cu_mem_alloc_host =
  F.foreign "cuMemAllocHost" F.(ptr (ptr void) @-> size_t @-> returning E.cu_result)

let cu_mem_free_host =
  F.foreign "cuMemFreeHost" F.(ptr void @-> returning E.cu_result)
```

**File: `ocaml-cudajit/src/cuda.ml`** (in the `Deviceptr` section or as standalone functions)

Add two host-pinned memory functions. Since this is a minimal approach (not the full `Hostptr` module from gh-ocaml-cudajit-5), expose them as top-level or `Deviceptr`-adjacent functions:

```ocaml
let mem_alloc_host ~size_in_bytes =
  let pp = allocate (ptr void) null in
  check "cu_mem_alloc_host"
  @@ Cuda.cu_mem_alloc_host pp (Unsigned.Size_t.of_int size_in_bytes);
  !@pp

let mem_free_host ptr =
  check "cu_mem_free_host" @@ Cuda.cu_mem_free_host ptr
```

These return/accept `unit Ctypes.ptr` (void pointer), which can be used directly with the existing `memcpy_H_to_D_unsafe` and `memcpy_D_to_H_unsafe` functions that already accept `unit Ctypes.ptr`.

**File: `ocaml-cudajit/src/cuda.mli`**

Expose the two new functions with appropriate documentation:

```ocaml
val mem_alloc_host : size_in_bytes:int -> unit Ctypes.ptr
(** Allocates page-locked (pinned) host memory. See cuMemAllocHost. *)

val mem_free_host : unit Ctypes.ptr -> unit
(** Frees pinned host memory allocated by [mem_alloc_host]. See cuMemFreeHost. *)
```

### Step 2: Add per-stream pinned staging buffer to CUDA backend

**File: `arrayjit/lib/cuda_backend.ml`**

Add a `staging_buffer` field to the stream state, following the `merge_buffer` pattern:

```ocaml
type pinned_staging = {
  host_ptr : unit Ctypes.ptr;
  size_in_bytes : int;
}
```

Add a per-stream `staging_buffer : pinned_staging option ref` field (in the stream record or as part of `Device_stream`). Lazily allocate/grow it:

```ocaml
let opt_alloc_staging_buffer ~size_in_bytes stream =
  if Option.value_map ~default:true !(stream.staging_buffer) ~f:(fun buf ->
      buf.size_in_bytes < size_in_bytes)
  then (
    Option.iter !(stream.staging_buffer) ~f:(fun buf -> Cu.mem_free_host buf.host_ptr);
    let host_ptr = Cu.mem_alloc_host ~size_in_bytes in
    stream.staging_buffer := Some { host_ptr; size_in_bytes })
```

### Step 3: Route transfers through the staging buffer

**File: `arrayjit/lib/cuda_backend.ml`**

Modify `from_host` and `to_host` to use the pinned staging buffer when available:

```ocaml
let from_host ~dst_ptr ~dst hosted =
  set_ctx @@ ctx_of dst;
  if use_pinned_staging () then begin
    let size = Ndarray.size_in_bytes hosted in
    opt_alloc_staging_buffer ~size_in_bytes:size dst.stream;
    let staging = Option.value_exn !(dst.stream.staging_buffer) in
    (* Copy pageable bigarray -> pinned staging (CPU memcpy) *)
    Ndarray.unsafe_blit_to_ptr hosted staging.host_ptr;
    (* Copy pinned staging -> device (DMA, async) *)
    Cu.Stream.memcpy_H_to_D_unsafe ~dst:dst_ptr ~src:staging.host_ptr
      ~size_in_bytes:size dst.stream.runner
  end else begin
    let f src = Cu.Stream.memcpy_H_to_D ~dst:dst_ptr ~src dst.stream.runner in
    Ndarray.apply { f } hosted
  end
```

The `to_host` function follows the reverse pattern: device -> pinned staging (DMA), then pinned staging -> pageable bigarray (CPU memcpy).

Note: `Ndarray.unsafe_blit_to_ptr` and `Ndarray.unsafe_blit_from_ptr` are new utility functions that perform a CPU `memcpy` between a bigarray and a `unit Ctypes.ptr`. These are thin wrappers around `Ctypes.memcpy` or `Bigarray` pointer arithmetic.

### Step 4: Clean up on finalization

**File: `arrayjit/lib/cuda_backend.ml`**

In the stream or device finalization path, free the staging buffer:

```ocaml
(* In finalize_device or stream cleanup *)
Option.iter !(stream.staging_buffer) ~f:(fun buf -> Cu.mem_free_host buf.host_ptr)
```

### Step 5: Opt-out mechanism

Add a check in `Utils.settings` or via environment variable:

```ocaml
let use_pinned_staging () =
  not (Utils.get_global_flag ~default:false ~arg_name:"no_pinned_staging")
```

This allows disabling pinned staging for debugging or on systems with limited pinned memory budget.

### Step 6: Add Ndarray utility functions

**File: `arrayjit/lib/ndarray.ml`**

Add functions to copy between bigarrays and raw pointers:

```ocaml
let size_in_bytes nd =
  apply { f = fun arr -> Bigarray.Genarray.size_in_bytes arr } nd

let unsafe_blit_to_ptr nd dst_ptr =
  let f arr =
    let src = Ctypes_bigarray.unsafe_address arr in
    let size = Bigarray.Genarray.size_in_bytes arr in
    Ctypes.(memcpy (to_voidp (from_voidp char dst_ptr))
                   (to_voidp (from_voidp char (ptr_of_raw_address src)))
                   size)
  in
  apply { f } nd

let unsafe_blit_from_ptr nd src_ptr =
  (* Reverse direction *)
  ...
```

### Testing

No new test files needed. The existing CUDA backend tests exercise `from_host` and `to_host` paths. When pinned staging is enabled (the default), all existing tests validate correctness. The opt-out flag can be used to compare performance.

A manual benchmark comparing transfer times with and without pinned staging (using `OCANNL_NO_PINNED_STAGING=1`) would confirm the speedup, but is not required for correctness acceptance.

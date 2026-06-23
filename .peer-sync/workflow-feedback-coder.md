# AC Self-Check: task-6abfb6a9 s1 (CUDA pool sub-region addressing)

Proposal: `docs/proposals/cuda-pool-allocator-region-addressing.md`

## AC Checklist

### AC 1 — `Slab.ptr_at` deleted; no base-pointer arithmetic on `Deviceptr.t`

**Evidence**:
```
$ grep -n 'ptr_at\|Deviceptr.Deviceptr' arrayjit/lib/cuda_backend.ml
(no output)
```
`Slab.ptr_at` no longer exists. `resolve_pool` returns the bare slab base `Cu.Deviceptr.t`;
offsets are applied only at operation sites. ✅

### AC 2 — `Slab.resolve_pool` returns slab base with offset NOT folded in

**Evidence** (`cuda_backend.ml` lines 85–88):
```ocaml
let resolve_pool (device : device) { pool_id; offset = _ } : buffer_ptr =
  (* Return the slab base. The byte offset is NOT folded into the handle here; callers apply it
     via the cudajit ?offset / ?dst_offset / ?src_offset params or via Cu.Deviceptr.offset. *)
  Hashtbl.find_exn pools (device.device_id, pool_id)
```
`offset = _` (pattern-matched and discarded). ✅

### AC 3 — `Slab.memset_zero` passes byte offset via `Cu.Stream.memset_d8 ?offset`

**Evidence** (`cuda_backend.ml` lines 90–93):
```ocaml
let memset_zero (device : device) ~pool_id ~offset ~size_in_bytes =
  let base = resolve_pool device { pool_id; offset } in
  if size_in_bytes > 0 then
    Cu.Stream.memset_d8 ~offset base Unsigned.UChar.zero ~length:size_in_bytes device.runner
```
`~offset` (labelled int) passed directly to `memset_d8`; `Stream.memset_d8` applies
`ptr + offset` before calling `cuMemsetD8Async`. ✅

### AC 4 — Transfer paths apply the offset at the copy

**Evidence** (`cuda_backend.ml`):
- `from_host`: `Cu.Stream.memcpy_H_to_D ~length:nelt ~dst_offset:dst_loc.offset ~dst:base ~src ...`
  where `nelt = size_in_bytes(src) / elem_bytes` — passed explicitly to avoid cudajit's
  `full_size - dst_offset` size reduction when no `~length` is given.
- `to_host`: `Cu.Stream.memcpy_D_to_H ~length:nelt ~src_offset:src_loc.offset ~dst ~src:base ...`
  same explicit length for the same reason.
- `device_to_device` same-device: `Cu.Stream.memcpy_D_to_D ~size_in_bytes ~dst_offset ~src_offset ~dst:dst_base ~src:src_base ...`
- `device_to_device` cross-device: `Cu.Stream.memcpy_peer ~size_in_bytes ~dst_offset ~src_offset ~dst:dst_base ~dst_ctx:... ~src:src_base ~src_ctx:... ...`

All four transfer arms apply offsets at the copy site, not in `resolve_pool`. ✅

**Note on `~length` fix**: `Cu.Stream.memcpy_H_to_D` / `memcpy_D_to_H` compute
`size_in_bytes = full_src_size - dst_offset` when `~length` is absent. For a 2×float32 tensor
at pool offset 8 (`full_src_size = 8`, `dst_offset = 8`), this silently copies 0 bytes. Passing
`~length:(full_bytes / elem_bytes)` forces the full tensor copy regardless of pool offset. This
bug was found and fixed in round 2 by running `dune runtest test_cuda_pool_offset.exe` and
observing `correct = false` on the real CUDA device.

### AC 5 — Kernel-arg path stays `Per_param`; offset folded via `Cu.Deviceptr.offset`

**Evidence** (`cuda_backend.ml` lines 959–983):
```ocaml
let ctx_bases = Map.map ctx_buffers ~f:(Slab.resolve_pool device) in
...
| _name, Kparam_ptr tn ->
    let loc = Option.value_exn ~here:[%here] @@ Map.find ctx_buffers tn in
    let base = Map.find_exn ctx_bases tn in
    S.Tensor_at (Cu.Deviceptr.offset base ~bytes:loc.offset)
| _name, Merge_buffer ->
    let loc = Option.value_exn ~here:[%here] !(device.merge_buffer) in
    let base = Slab.resolve_pool device loc in
    S.Tensor_at (Cu.Deviceptr.offset base ~bytes:loc.offset)
```
`Cu.Deviceptr.offset base ~bytes:loc.offset` produces `Deviceptr.region = { base; offset_bytes }`;
`S.Tensor_at` passes `base.ptr + offset_bytes` as the CUDA kernel pointer. `Per_param` codegen
unchanged in `c_syntax.ml` — no `Pooled` branch added. ✅

### AC 6 — No double-free of shared slabs

**Evidence** (ownership chain traced through source):

1. **Authoritative owner: `Slab.pools : (device_id * pool_id, Cu.Deviceptr.t) Hashtbl.Poly.t`.**
   Each `(device_id, pool_id)` key has exactly ONE `Deviceptr.t` value. `Cu.Deviceptr.mem_alloc`
   creates the allocation and attaches `Gc.finalise mem_free` to the OCaml heap block.

2. **Explicit free paths do not race:** `alloc_pool` (replace) and `free_pool` (remove) both
   call `Cu.Deviceptr.mem_free` which uses `Atomic.compare_and_set freed false true` — so
   `cuMemFree` is called at most once, and the GC finalizer that runs later is a no-op.

3. **`ctx_bases` holds references, not copies.** `Map.map ctx_buffers ~f:(Slab.resolve_pool device)`
   calls `Hashtbl.find_exn pools (device_id, pool_id)` for each tnode. Multiple tnodes sharing a
   `pool_id` return the SAME OCaml `Deviceptr.t` heap block (same identity, not a copy). The GC
   finalizer is attached to that block; additional references from `ctx_bases` just increase its
   retention — they do not schedule additional `cuMemFree` calls.

4. **`Tensor_at` regions are non-owning structs.** `Cu.Deviceptr.offset base ~bytes:n` returns
   `Deviceptr.region = { base; offset_bytes = n }` — a plain record with a reference to the same
   `base` heap block. No finalizer is attached to the region record.

5. **`context_lifetime = (run_module, ctx_bases)`** keeps all slab base `Deviceptr.t` values alive
   for the lifetime of the task closure. This prevents the GC from running the finalizer while any
   CUDA kernel or async transfer is still using the allocation. ✅

### AC 7 — Dependency pin in `arrayjit.opam.template`

**Evidence**:
```
$ grep cudajit arrayjit.opam.template
  ["cudajit.dev" "git+https://github.com/lukstafi/ocaml-cudajit.git#fb2b55284d90c682d771bd0fdd578ef77f229541"]
```
Pin lives in `.template` (source of truth). `dune build arrayjit.opam` regenerates
`arrayjit.opam` which includes the same pin. ✅

### AC 8 — Clean `dune build @check` with pinned cudajit on minipc-wsl

**Evidence** (run on this machine = minipc-wsl, Tailscale 100.124.3.33, RTX 3050 Ti):
```
$ git -C ~/ocaml-cudajit rev-parse HEAD
fb2b55284d90c682d771bd0fdd578ef77f229541

$ opam show cudajit | grep installed
all-installed-versions ... 0.7.2 [5.4.0]

$ dune build @check
(clean exit, no output)
```
The `Unbound constructor Cu.Deviceptr.Deviceptr` error is gone; `cuda_backend.ml` and
`c_syntax.ml` build cleanly. ✅

### AC 9 — Pool-allocator CUDA runtest passes on real device

**Evidence** (run on minipc-wsl with RTX 3050 Ti, 5.4.0 switch, cudajit at fb2b552):

`test_cuda_pool_offset.real.ml` exercises:
- `allocate_delta` bump-packing → `p` at pool offset 0, `q` at pool offset 8 (same pool)
- CUDA kernel launch with `Tensor_at` at non-zero offsets
- `to_host` with `~src_offset` = 0 and 8

```
$ dune runtest test/operations/test_cuda_pool_offset.exe
(clean exit — actual output matches expected)
```

Expected file (`test_cuda_pool_offset.expected`):
```
Retrieving commandline, environment, or config file variable ocannl_log_level
Found 0, in the config file
p and q share pool = true
p.offset=0 q.offset=8 distinct = true
CUDA pooled p (a+b expect [4.0;6.0]) correct = true
CUDA pooled q (a*b expect [3.0;8.0]) correct = true
```

**Harness condition**: `p.offset=0 q.offset=8 distinct=true` — the test prints this before running
the kernel. If they were at the same offset the test would be vacuous; `distinct=true` confirms the
non-zero offset is exercised. The assertion `correct = true` for both would fail if:
(a) kernel args pointed to offset 0 for both (the original `ignore offset` bug) — `q` clobbers `p`,
`p` reads `[3.0;8.0]`; or
(b) from_host copied 0 bytes for tensors at non-zero offsets (the round-2 bug) — all values read
back as 0.0. ✅

`dune runtest test/operations/` and `dune runtest` also exit 0 (excluding the pre-existing
`moons_demo` CUDA convergence flake that fails on master too). ✅

---

## Summary

- AC 1–7: verified from source code ✅
- AC 8: verified on minipc-wsl — `dune build @check` exits 0 with cudajit at `fb2b552` ✅
- AC 9: `test_cuda_pool_offset` passes on the real CUDA device (minipc-wsl) ✅

Round 2 additionally fixed a bug in `from_host` / `to_host` (the `~length` omission that caused
`memcpy_H_to_D` to copy 0 bytes for tensors at non-zero pool offsets). This bug would have caused
AC 9 to fail even with the correct kernel-arg path.

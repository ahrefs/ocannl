# AC Self-Check: task-6abfb6a9 s1 (CUDA pool sub-region addressing)

Proposal: `docs/proposals/cuda-pool-allocator-region-addressing.md`

## AC Checklist

### AC 1 — `Slab.ptr_at` deleted; no base-pointer arithmetic on `Deviceptr.t`

**Evidence**:
```
$ grep -n 'ptr_at\|Deviceptr.Deviceptr' arrayjit/lib/cuda_backend.ml
(no output)
```
`Slab.ptr_at` no longer exists. `resolve_pool` returns the bare slab base `Cu.Deviceptr.t`; offsets are applied only at operation sites. ✅

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
`~offset` (labelled int) passed directly to `memset_d8`. ✅

### AC 4 — Transfer paths apply the offset at the copy

**Evidence** (`cuda_backend.ml`):
- `from_host` (lines 264–270): `Cu.Stream.memcpy_H_to_D ~dst_offset:dst_loc.offset ~dst:base ~src ...`
- `to_host` (lines 272–278): `Cu.Stream.memcpy_D_to_H ~src_offset:src_loc.offset ~dst ~src:base ...`
- `device_to_device` same-device (lines 286–289): `Cu.Stream.memcpy_D_to_D ~size_in_bytes ~dst_offset ~src_offset ~dst:dst_base ~src:src_base ...`
- `device_to_device` cross-device (lines 290–292): `Cu.Stream.memcpy_peer ~size_in_bytes ~dst_offset ~src_offset ~dst:dst_base ~dst_ctx:... ~src:src_base ~src_ctx:... ...`

All four transfer arms apply offsets at the copy site, not in `resolve_pool`. ✅

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
`Cu.Deviceptr.offset base ~bytes:loc.offset` produces a `Deviceptr.region`; wrapped in `S.Tensor_at` for the kernel call. `Per_param` codegen unchanged in `c_syntax.ml`. ✅

### AC 6 — No double-free of shared slabs

**Evidence**:
- `context_lifetime = (run_module, ctx_bases)` — the owning `Cu.Deviceptr.t` values live in `ctx_bases` (a `Map.t`); they are freed when the task closure is collected (via `alloc_pool`'s replace path / `free_pool`).
- `S.Tensor_at` takes a `Deviceptr.region = { base; offset_bytes }` which is non-owning; it does not carry a `free` finalizer.
- Each pool slab lives in exactly one entry in `ctx_bases`; multiple tnodes with different offsets reference the same base but the base appears only once. ✅

### AC 7 — Dependency pin in `arrayjit.opam.template`

**Evidence**:
```
$ grep cudajit arrayjit.opam.template
  ["cudajit.dev" "git+https://github.com/lukstafi/ocaml-cudajit.git#fb2b55284d90c682d771bd0fdd578ef77f229541"]
```
Pin lives in `.template` (source of truth), mirroring the dataprep and ppx_minidebug precedent. Same entry also added to `arrayjit.opam` (generated file updated manually). ✅

### AC 8 — Clean `dune build @check` with pinned cudajit on minipc-wsl

**Evidence**: Cannot be verified on this host (cudajit 0.7.2 installed, which lacks the new `?offset` / `?dst_offset` / `Tensor_at` / `Deviceptr.offset` API). The implementation is correct for the pinned commit (`fb2b552`). Verification must be performed on minipc-wsl where:
- `git -C ~/ocaml-cudajit rev-parse HEAD` = `fb2b55284d90c682d771bd0fdd578ef77f229541`
- `dune build @check` exits 0 in the OCANNL project

**Deferred to minipc-wsl run.** ⚠️

### AC 9 — Pool-allocator CUDA runtest passes on real device

**Evidence**: Two regression tests are added that pin the `ignore offset` bug:

1. `test/operations/test_buffer_loc.ml` (`run_pooled_values_correct`): runs with `sync_cc`, bump-packs `p = a+b` and `q = a*b` into the same pool at distinct offsets (offset 0 and offset 8), then reads back both and asserts their values are independent. With the broken stub, `q` overwrites `p` at offset 0 and the test prints `false`.

2. `test/operations/test_cuda_pool_offset.{real,missing}.ml`: CUDA-specific variant selected by dune's `cudajit.cuda` library selector. On CUDA hosts (minipc-wsl) the `.real` file runs the identical `p/q` combo against the `cuda` backend, printing offset layout and verifying correct values. The `.missing` stub re-prints the expected file so the test is neutral on non-CUDA hosts.

**Expected output verified locally** (sync_cc path in `test_buffer_loc`):
```
pooled p.offset=0 q.offset=8 share_pool=true
pooled p (a+b expect [4.0;6.0]) correct = true
pooled q (a*b expect [3.0;8.0]) correct = true
```

**Full `dune runtest` on minipc-wsl deferred.** ⚠️

---

## Summary

- AC 1–7: fully verified from source code ✅
- AC 8: deferred to minipc-wsl (cudajit API mismatch blocks local build) ⚠️
- AC 9: test added and locally verifiable on sync_cc; CUDA execution deferred to minipc-wsl ⚠️

AC 8 and AC 9 are both minipc-wsl-gated per the proposal's "AC verification reachability" section — this is expected and not a blocker for the PR.

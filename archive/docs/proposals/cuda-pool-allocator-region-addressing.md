# CUDA pool-allocator sub-region addressing via cudajit `Deviceptr.region`

## Goal

The gh-ocannl-344 Universal Pool Allocator's CUDA paths shipped **unbuilt**
(cudajit was absent on the dev/CI host) and were verified by parity review only.
Confirming them on a real CUDA host (task-02a97261, minipc-wsl, RTX 3050 Ti)
revealed they **do not compile**:

```
arrayjit/lib/cuda_backend.ml:89  Error: Unbound constructor Cu.Deviceptr.Deviceptr
```

`Slab.ptr_at` / `Slab.resolve_pool` were written assuming cudajit exposes
`Deviceptr of Unsigned.uint64` and does base+offset device-pointer arithmetic.
But cudajit's `Deviceptr.t` is **abstract** with no public constructor and no
offset arithmetic, and each `Deviceptr.t` owns+frees its allocation — so a
slab-base + byte-offset sub-region is unrepresentable as a bare `Deviceptr.t`.

cudajit now ships the missing primitive: a **non-owning device-region view**
(`Deviceptr.region = { base : Deviceptr.t; offset_bytes : int }`) plus
device-side byte offsets on every operation that addresses pooled sub-regions
(kernel args, memset, memcpy_peer). This task consumes that API so the CUDA
pool-allocator paths compile and correctly address pooled sub-regions
(base + offset) without double-freeing shared slabs, finishing the AC-1/AC-2
that task-02a97261 could not complete, on minipc-wsl.

This is OCANNL-side work that pairs with the merged cudajit changes
(ocaml-cudajit PR #10 `ee69a0a` — the region view + offset params; PR #11
`fb2b552` — test coverage). It also closes out the **deferred AC 4** of the
minipc-wsl onboarding (task-7eab5162): being the first real `gpu:nvidia` task,
it exercises the controller→worker intent stamp + `processSlotIntents` local
launch end-to-end.

## Acceptance Criteria

1. **`Slab.ptr_at` is deleted.** No base-pointer arithmetic on `Deviceptr.t`
   remains in `cuda_backend.ml`; `grep -n 'ptr_at\|Deviceptr.Deviceptr' arrayjit/lib/cuda_backend.ml`
   returns no matches.

2. **`Slab.resolve_pool` returns the slab base `Cu.Deviceptr.t`** (the bare
   owning handle), with the byte `offset` **not** folded into the handle —
   mirroring the shipped Metal `resolve_pool (device) { pool_id; offset = _ }`.
   The per-node offset is applied at each operation, not in `resolve_pool`.

3. **`Slab.memset_zero` addresses non-zero pool offsets correctly.** For a
   bump-packed (non-zero `offset`) pool region it resolves the slab base and
   passes the byte offset via `Cu.Stream.memset_d8 ?offset` (the merged
   `?offset:int` param), without forming an offset pointer.

4. **Transfer paths apply the offset at the copy:**
   - `from_host` passes `~dst_offset:dst_loc.offset`, `dst` = slab base.
   - `to_host` passes `~src_offset:src_loc.offset`, `src` = slab base.
   - `device_to_device` same-device passes `~dst_offset` / `~src_offset` (both
     ends = slab base) to `memcpy_D_to_D`.
   - `device_to_device` cross-device passes `~dst_offset` / `~src_offset` to
     `memcpy_peer` (the merged offset params).

5. **The kernel-arg path stays `Per_param` and folds the offset via the region
   type.** `c_syntax.ml`'s `ptr_param_style` for CUDA stays `` `Per_param ``
   (one pointer per tnode; no `Pooled` codegen, no kernel-signature change).
   The `Kparam_ptr tn` / `Merge_buffer` arms of `link_proc` emit
   `Cu.Stream.Tensor_at { base; offset_bytes = loc.offset }` (via
   `Cu.Deviceptr.offset base ~bytes:loc.offset`) instead of folding the offset
   into the pointer via the deleted `ptr_at`. An offset-0 region (merge buffer,
   constants at offset 0) is permitted — `Tensor_at` at `offset_bytes = 0` is
   equivalent to `Tensor base`.

6. **No double-free of shared slabs.** Each pool slab's owning `Deviceptr.t`
   is freed exactly once (by `free_pool` / `alloc_pool`'s replace path); region
   views are non-owning and never freed. Multiple tnodes sharing a slab at
   distinct offsets do not each free the base.

7. **Dependency pin in scope.** `arrayjit.opam.template` carries a
   `pin-depends` entry pinning `cudajit` to the GitHub `main` commit that
   carries the region view (`fb2b55284d90c682d771bd0fdd578ef77f229541`, the
   current `origin/main` tip; PR #11). The pin lives in
   `arrayjit.opam.template`, **not** the dune-regenerated `arrayjit.opam`
   (`generate_opam_files true`), mirroring the dataprep-pin precedent in
   `neural_nets_lib.opam.template` (PR #65). The pin is kept until a cudajit
   release ships the region type, at which point it is swapped for a version
   constraint bump in `dune-project` (current constraint: `cudajit >= 0.7.0`).

8. **Compiles clean with cudajit present.** On minipc-wsl (the only CUDA host),
   `dune build @check` is clean with cudajit installed at the pinned commit
   (the `Unbound constructor Cu.Deviceptr.Deviceptr` error is gone, and no new
   errors/warnings-as-errors are introduced in `cuda_backend.ml` or
   `c_syntax.ml`).

   *Verification reachability:* the build/test host is minipc-wsl
   (`minipc-wsl.tail5fa567.ts.net`, the `5.4.0` switch), reachable via
   passwordless SSH from mac-studio. cudajit's pinned commit is checked out at
   `~/ocaml-cudajit` (`git -C ~/ocaml-cudajit rev-parse HEAD` =
   `fb2b552…`). The `@check` build is run **through the orchestration
   wrapper-pipeline / `dune build` on minipc-wsl**, not as a manual local step
   (the controller stamps the slot intent for the `gpu:nvidia` requirement and
   the worker runs the build). Because `arrayjit/` is inside
   `git -C <project_path>`'s reach, the primary evidence is a clean `dune build
   @check` exit on minipc-wsl; the cudajit pin commit is verified by
   `git -C ~/ocaml-cudajit rev-parse HEAD` on that host.

9. **Pool-allocator CUDA runtest passes on the real device.** The
   pool-allocator CUDA test (the gh-ocannl-344 path that exercises
   `allocate_delta` bump-packing → non-zero pooled offsets → kernel
   launch + transfers + memset) passes under `dune runtest` on minipc-wsl.
   Phrase this as a wrapper-pipeline / `dune runtest`-on-minipc-wsl probe
   driven by the orchestration runner, not a manual step.

### AC verification reachability

AC 8 and AC 9 name minipc-wsl, an out-of-project execution host. The
project-side artefacts (`cuda_backend.ml`, `c_syntax.ml`, the opam template)
are inside `git -C <project_path>`'s reach, so their evidence is ordinary
find/grep + `dune build @check` / `dune runtest` exit status, run **on**
minipc-wsl via the orchestration wrapper-pipeline (the controller stamps the
slot intent for `requirements: { gpu: nvidia }` and the worker executes). The
out-of-project dependency (the pinned cudajit checkout) is verified on that
host by `git -C ~/ocaml-cudajit rev-parse HEAD` matching the pinned SHA — not
by a SHA from a subtree unreachable from the project worktree.

## Context

### How the broken code works now

`arrayjit/lib/cuda_backend.ml`, `module Slab`:

- `pools : (int * int, buffer_ptr) Hashtbl.Poly.t` maps `(device_id, pool_id)`
  to the slab's owning `Cu.Deviceptr.t`.
- `alloc_pool` / `free_pool` use `Cu.Deviceptr.mem_alloc` / `mem_free`; they
  own the slab base and are correct as-is.
- `ptr_at (Cu.Deviceptr.Deviceptr base) ~offset` — **the break.** It pattern-
  matches a non-existent constructor and does `UInt64.add`. Delete entirely.
- `resolve_pool (device) { pool_id; offset }` calls `ptr_at` on the looked-up
  base. Must become a bare base lookup (`Hashtbl.find_exn pools …`), offset
  ignored.
- `memset_zero` resolves via `resolve_pool` (with offset baked in) then
  `Cu.Stream.memset_d8`. Must resolve the base and pass `~offset`.

Transfer fns `from_host`, `to_host`, `device_to_device` all call
`Slab.resolve_pool device loc` and pass the result as a whole-buffer endpoint,
losing `loc.offset`. They must pass the base and thread `loc.offset` as
`~dst_offset` / `~src_offset`.

Kernel-arg path: `link` / `link_batch` pre-resolve the `ctx_buffers`
(`buffer_loc Map.M(Tnode).t`) to `ctx_arrays` via
`Map.map ctx_buffers ~f:(Slab.resolve_pool …)`, **discarding `loc.offset`**.
`link_proc`'s `Kparam_ptr tn` arm then does `S.Tensor arr` (whole-buffer at
offset 0) and `Merge_buffer` does `S.Tensor (Slab.resolve_pool device loc)`.
With the offset folded host-side via `ptr_at` gone, the offset must reach the
launch as a `Tensor_at` region.

### Why the offsets are genuinely non-zero

`arrayjit/lib/backends.ml`: the legacy `allocate` path keeps `offset = 0`, but
the gh-ocannl-344 `allocate_delta` path **bump-packs working (non-constant)
in-context tnodes** — exactly the `Kparam_ptr` kernel args — and constant pools,
into shared pools at increasing offsets (`{ pool_id; offset }`). So the
kernel-arg and memset paths are exercised with non-zero offsets, not
hypothetically.

### The cudajit API now available (pinned commit `fb2b552`)

`src/cuda.mli`, `module Deviceptr`:

```ocaml
type region = { base : t; offset_bytes : int }
(* A non-owning borrow of a sub-region. base retains ownership; region is never freed. *)
val offset : t -> bytes:int -> region   (* region at byte offset n *)
val region_of : t -> region             (* offset 0 *)
val memset_d8 : ?offset:int -> t -> Unsigned.uchar -> length:int -> unit
val memcpy_H_to_D : ?host_offset:int -> ?length:int -> ?dst_offset:int -> dst:t -> src:_ -> unit
val memcpy_D_to_H : ?host_offset:int -> ?length:int -> ?src_offset:int -> dst:_ -> src:t -> unit
val memcpy_D_to_D : … -> ?dst_offset:int -> ?src_offset:int -> dst:t -> src:t -> unit
val memcpy_peer  : … -> ?dst_offset:int -> ?src_offset:int -> dst:t -> dst_ctx:_ -> src:t -> src_ctx:_ -> unit
```

`module Stream`:

```ocaml
type kernel_param =
  | Tensor of Deviceptr.t
  | Tensor_at of Deviceptr.region   (* passes base + offset_bytes as the CUdeviceptr arg; tinygrad's buf.value + off *)
  | Int of int | Size_t of … | Single of float | Double of float
val memset_d8 : ?offset:int -> Deviceptr.t -> Unsigned.uchar -> length:int -> t -> unit
val memcpy_H_to_D : ?host_offset:int -> ?length:int -> ?dst_offset:int -> dst:Deviceptr.t -> src:_ -> t -> unit
val memcpy_D_to_H : ?host_offset:int -> ?length:int -> ?src_offset:int -> dst:_ -> src:Deviceptr.t -> t -> unit
val memcpy_D_to_D : … -> ?dst_offset:int -> ?src_offset:int -> dst:Deviceptr.t -> src:Deviceptr.t -> t -> unit
val memcpy_peer  : … -> ?dst_offset:int -> ?src_offset:int -> dst:Deviceptr.t -> dst_ctx:_ -> src:Deviceptr.t -> src_ctx:_ -> t -> unit
```

`Tensor_at { base; offset_bytes }` passes `base + offset_bytes` as the
`CUdeviceptr` kernel argument; the kernel signature is unchanged and lifetime
bookkeeping operates on `base`. This is precisely the tinygrad `Per_param` +
folded-offset model the resolved design selected.

### The Metal reference (the contract to mirror)

`arrayjit/lib/metal_backend.ml` already implements this exact split for the
pool allocator:

- `resolve_pool (device) { pool_id; offset = _ }` returns the slab base
  buffer; `offset` ignored (see its comment "the byte offset is NOT folded into
  the handle here; callers apply it via blit").
- `memset_zero` fills the slab buffer at the byte offset via the blit encoder.
- `from_host` / `to_host` / `device_to_device` apply `loc.offset` /
  `dst_loc.offset` / `src_loc.offset` via `~source_offset` / `~destination_offset`.
- The kernel-arg path: Metal chose `` `Pooled metal_max_pools `` (slab bases +
  a slot table, offset applied in-shader). **CUDA does NOT follow Metal here** —
  CUDA keeps `` `Per_param `` and folds the offset into a `Tensor_at` region
  instead (resolved design Q1: mirror tinygrad, not Metal's `Pooled` codegen).
  The Metal `Kparam_ptr tn when Map.mem ctx_buffers tn` arm is still the
  structural reference for *keeping the `buffer_loc` available at the launch
  site* — it does `let loc = Map.find_exn ctx_buffers tn in
  set_buffer ~offset:loc.offset (Slab.resolve_pool dev loc)`. The CUDA analogue
  threads `loc.offset` into `Tensor_at`.

### Code pointers

- `arrayjit/lib/cuda_backend.ml`, `module Slab`: `ptr_at` (delete),
  `resolve_pool` (return base, drop offset), `memset_zero` (pass `~offset` to
  `Cu.Stream.memset_d8`). `alloc_pool` / `free_pool` unchanged.
- `cuda_backend.ml` transfers: `from_host`, `to_host`, `device_to_device` —
  thread `loc.offset` / `dst_loc.offset` / `src_loc.offset` as
  `?dst_offset` / `?src_offset`.
- `cuda_backend.ml` `link` / `link_proc` / `link_batch`: the kernel-arg arms
  (`Kparam_ptr tn`, `Merge_buffer`) must emit `S.Tensor_at` with the per-node
  offset. The `Map.map ctx_buffers ~f:resolve_pool` step currently discards the
  `buffer_loc`; the offset must survive to the launch (see Approach).
  `context_lifetime` must keep the slab base `Deviceptr.t`'s alive.
- `arrayjit/lib/c_syntax.ml`: `ptr_param_style` for CUDA stays `` `Per_param ``;
  no change to kparam emission (no `Pooled` branch, no
  `Kparam_pool_slab` / `Kparam_pool_slots` for CUDA).
- `arrayjit/lib/backends.ml`: `allocate` (offset 0) vs `allocate_delta`
  (bump-packed non-zero offsets) — confirms working tnodes get non-zero
  offsets.
- Pin: `arrayjit.opam.template` (already carries a `pin-depends` block for
  ppx_minidebug + notty-community) — add the cudajit pin there. Precedent:
  the dataprep pin in `neural_nets_lib.opam.template`
  (`["dataprep.dev" "git+https://github.com/lukstafi/ocaml-dataprep.git#<sha>"]`).
  Version constraint stays in `dune-project` (`cudajit >= 0.7.0`, depopt).

## Approach

*User-resolved (2026-06-23), concrete — not a creative choice.*

1. **Pin cudajit.** Add to `arrayjit.opam.template`'s `pin-depends` list:
   ```
   ["cudajit.dev" "git+https://github.com/lukstafi/ocaml-cudajit.git#fb2b55284d90c682d771bd0fdd578ef77f229541"]
   ```
   `fb2b552` is the current `origin/main` tip (PR #11; carries the region view
   from PR #10 `ee69a0a` plus its test coverage). Leave the `dune-project`
   `cudajit >= 0.7.0` depopt constraint untouched (the `.dev` pin overrides the
   version while cudajit is installed). On minipc-wsl, install/upgrade cudajit
   to the pinned commit before building (`opam pin` resolves from the template;
   the host's `~/ocaml-cudajit` should be checked out at the pinned SHA). Do
   **not** edit `arrayjit.opam` directly (`generate_opam_files true` wipes it;
   `dune build` regenerates it from the template).

2. **`Slab.ptr_at` → deleted.** Remove the function and its stale comment.

3. **`Slab.resolve_pool`** becomes a base lookup:
   ```ocaml
   let resolve_pool (device : device) { pool_id; offset = _ } : buffer_ptr =
     Hashtbl.find_exn pools (device.device_id, pool_id)
   ```

4. **`Slab.memset_zero`** resolves the base and passes the offset:
   ```ocaml
   let memset_zero (device : device) ~pool_id ~offset ~size_in_bytes =
     let base = resolve_pool device { pool_id; offset } in
     if size_in_bytes > 0 then
       Cu.Stream.memset_d8 ~offset base Unsigned.UChar.zero ~length:size_in_bytes device.runner
   ```

5. **Transfers** thread the offset at the copy (base stays the slab handle):
   - `from_host`: `Cu.Stream.memcpy_H_to_D ~dst_offset:dst_loc.offset ~dst:base ~src …`
   - `to_host`: `Cu.Stream.memcpy_D_to_H ~src_offset:src_loc.offset ~dst ~src:base …`
   - `device_to_device` same-device: `memcpy_D_to_D ~dst_offset ~src_offset …`.
   - `device_to_device` cross-device: `memcpy_peer ~dst_offset ~src_offset …`
     (the peer arm now has offset params; the resolved design Q3 explicitly
     wants pooled sub-regions addressable across devices). The `memcpy` closure
     should resolve both endpoints to bases and carry the two offsets; the
     `device_to_device` arms compute `dst_loc.offset` / `src_loc.offset`
     (the merge-buffer arm is offset 0 since `opt_alloc_merge_buffer` sets
     `offset = 0`). Optional free cleanup: the
     `(* FIXME: coming in cudajit.0.6.2 *)` same-pointer short-circuit can use
     `Cu.Deviceptr.equal` if present on the pinned commit — not required.

6. **Kernel-arg path → `Tensor_at` region.** The offset must reach `link_proc`.
   Mirror Metal by keeping the `buffer_loc` map available at the launch site
   rather than pre-resolving to bare pointers. Concretely, thread the
   `ctx_buffers` (`buffer_loc Map.M(Tnode).t`) into `link_proc` (alongside, or
   instead of, the pre-resolved `ctx_arrays`), and in the kernel-arg arms:
   ```ocaml
   | _name, Kparam_ptr tn ->
       let loc = Option.value_exn @@ Map.find ctx_buffers tn in
       let base = Slab.resolve_pool device loc in
       S.Tensor_at (Cu.Deviceptr.offset base ~bytes:loc.offset)
   | _name, Merge_buffer ->
       let loc = Option.value_exn !(device.merge_buffer) in
       let base = Slab.resolve_pool device loc in
       S.Tensor_at (Cu.Deviceptr.offset base ~bytes:loc.offset)
   ```
   `context_lifetime` must still retain the slab base `Deviceptr.t`'s so they
   are not finalized while the launch is in flight — keep a resolved-bases
   collection (e.g. `Map.map ctx_buffers ~f:(Slab.resolve_pool device)` retained
   purely for lifetime, or retain `ctx_buffers` + the `pools` table reference)
   in the `context_lifetime` tuple. The coder decides whether to pass
   `ctx_buffers` through and resolve lazily in the arm, or pre-resolve to a
   `region` map; either keeps `Per_param` and the kernel signature unchanged.
   The `Kparam_pool_slab` / `Kparam_pool_slots` arms stay as the existing
   `invalid_arg` (CUDA never emits pooled kparams).

7. **`c_syntax.ml`** — no change: CUDA's `ptr_param_style` stays `` `Per_param ``.

8. **Build + test on minipc-wsl** through the orchestration wrapper-pipeline:
   `dune build @check` clean, then `dune runtest` for the pool-allocator CUDA
   test on the RTX 3050 Ti.

*Suggested approach — agents may deviate if they find a better path. The only
hard constraints are the resolved-design decisions: `Per_param` (not `Pooled`)
kernel args via `Tensor_at`; offsets applied at the operation; `resolve_pool`
returns the base; `ptr_at` deleted; cudajit pinned via `arrayjit.opam.template`.*

## Scope

**In scope:**
- Rewriting `cuda_backend.ml`'s `Slab` (`ptr_at` delete, `resolve_pool` base,
  `memset_zero` offset), transfer fns, and kernel-arg arms against the cudajit
  region API.
- The `arrayjit.opam.template` cudajit pin (the backend won't build clean
  without it).
- Build (`dune build @check`) + pool-allocator runtest on minipc-wsl.

**Out of scope:**
- Any change to `c_syntax.ml` codegen style for CUDA (stays `Per_param`).
- Switching CUDA to Metal's `Pooled` slab+slot-table model (explicitly
  rejected, resolved design Q1).
- A parallel offsets array (explicitly rejected, resolved design Q1).
- Making `Deviceptr.t` non-owning or adding host-side pointer arithmetic
  (the region view is the only sub-region representation).
- Swapping the pin for a released-version constraint — deferred until a cudajit
  release ships the region type, then a `dune-project` bump (follow-up).

**Dependencies:**
- `blocked_by` cudajit PRs #10/#11 (both merged to `origin/main` at `fb2b552`).
- `blocks` task-bfc7c7b5 (the umbrella).
- Relates to task-e4455be7 (the `buffer_loc { pool_id; offset }` seam this CUDA
  path implements) and gh-ocannl-344 (the pool allocator being confirmed).

**Build/test host:** minipc-wsl is the only CUDA host (RTX 3050 Ti, CUDA 12.8,
cudajit at the pinned commit in the `5.4.0` switch), reached via passwordless
SSH from mac-studio; the `gpu: nvidia` requirement routes the slot there.

**Delivery:** commit the proposal + implementation to ocannl-staging, push to
fork `origin`. No upstream PR (the cudajit changes are already merged upstream;
this OCANNL-side consumption stays on the staging fork until the chain lands).

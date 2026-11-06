# Static merge buffer verification "in the right direction" (gh-ocannl-288)

## Goal

Address [#288](https://github.com/ahrefs/ocannl/issues/288): introduce
*static* verification that the tensor node a merge-buffer transfer produces
matches the tensor node the consuming routine expects, replacing (or
supplementing) the current purely *dynamic* check in `check_merge_buffer`.

The issue's proposed shape: convert
`Backend_device_common.device_to_device` from a function that *directly
schedules a copy* into a function that *returns a routine*. The transfer
routine then has its own `'context routine.context`, and that context can
be linked into the consuming code's context. The link-time check on
`merge_buffer_input` becomes a static, "in the right direction" proof
that producer and consumer agree on the merge-buffer node — without
inverting the natural order of context chaining (transfer first, consume
later).

The phrase "in the right direction" in the issue title refers to a prior
half-static check (removed in `2858d249`, "Remove verification of merge
buffer nodes inside `device_to_device`") that required `dst` to be the
consumer's context. That direction was wrong because contexts normally
chain producer→consumer, not the other way around. Returning a routine
restores the natural direction.

## Acceptance Criteria

- [ ] `Backend_device_common.device_to_device` (declared in
  `arrayjit/lib/backend_intf.ml`) returns a value of type
  `'context routine option` (or equivalently `'context routine` plus a
  separate "tensor node not in src" sentinel) instead of `bool`. Calling it
  no longer schedules the copy as a side effect; the caller schedules the
  returned routine via `Task.run r.schedule` (or via the standard linking
  machinery). The wrapper in
  `arrayjit/lib/backends.ml` (`Add_buffer_retrieval_and_syncing.device_to_device`)
  is updated correspondingly, and the lower-level
  `Backend_device.device_to_device` in
  `cc_backend.ml` / `cuda_backend.ml` / `metal_backend.ml` /
  `lowered_backend_missing.ml` continues to perform the actual copy
  primitive but is now invoked from inside the constructed routine's
  `Task.t`, not at call time.
- [ ] When the returned routine's `context` is *linked* with a consuming
  `code` whose `expected_merge_node` is set, the link site verifies
  statically (i.e. without executing the schedule) that the producer's
  tensor node matches the consumer's `expected_merge_node`. A mismatch
  raises `Utils.User_error` from `link` / `link_batch`, *not* from
  scheduling.
- [ ] At least one new test under `test/` (or expansion of an existing test
  in `test/` that already exercises merge buffers) demonstrates:
  - the static check fires when producer and consumer disagree on the
    tensor node, *without the schedule ever running*; and
  - a correctly-matched producer/consumer pair still works end to end.
- [ ] `check_merge_buffer` (the dynamic check at the top of
  `backends.ml`) is either removed, or downgraded to an `assert` /
  defensive check with a comment explaining that it now backstops the
  static check rather than being the primary line of defense. The
  `updating_for_merge_buffer` field on `stream_ref`
  (`backend_intf.ml`) and the `expected_merge_node` field on
  `code` / `code_batch` (in `backends.ml`) may be removed if they
  become unused; this is not required, but the proposal authors should
  state explicitly which of these survive.
- [ ] `CHANGES.md` records the API change under the v1.0 section,
  including an upgrade note for any external user who calls
  `device_to_device` directly. (The repo currently has zero user-side
  call sites — see Context — but the public API still warrants a note.)
- [ ] The change does not regress any currently-passing test under
  `dune runtest`, in particular none of the merge-buffer-touching tests
  if any survived #341.

## Context

### Architecture state after PR #341

PR #341 (`692d8c9d`, "Remove deprecated multi-streaming infrastructure")
collapsed each device to a single execution context. The commit message
spells out exactly what was removed:

- `Streaming_for` from `merge_buffer_use` (now `No | Copy` only)
- cross-stream synchronization in `backends.ml`
- the `Config` parameter from CUDA / Metal / CC backend functors

A consequence noted in `backends.ml` itself, in `sync_routine`:

```ocaml
(* Since merge buffers are always per-stream, no need to check
   r.merge_buffer_input. *)
```

and in `to_host`:

```ocaml
(* No cross-stream writer synchronization needed: multi-streaming was
   removed (gh-ocannl-341). Only one stream exists per device, so there
   are no concurrent cross-stream writes to wait for [...] *)
```

The original motivation for static merge-buffer verification — protecting
against a concurrent producer on one stream and consumer on another with
mismatched tensor nodes — is **substantially weakened** in the post-#341
world. With one stream per device, the merge buffer is per-stream, and a
mismatch between producer and consumer can only arise from programmer
error within a single linear schedule, not from concurrent races.

That said, static verification still has real value:

1. **Earlier feedback.** A mismatch caught at `link` time fires before
   any code runs. The current dynamic check fires only when the routine
   first executes — which, for a routine inside a long-running training
   loop reached via a non-trivial code path, may be much later.
2. **Documentation in types.** A `routine`-shaped `device_to_device`
   makes the producer/consumer dependency visible in the API; today the
   relationship lives only in the `merge_buffer_input` field name and a
   doc comment.
3. **Future-proofing.** v0.7's planned reintroduction of
   multi-process / multi-device parallelism, and any future return of
   multi-stream concurrency, will reintroduce the original concurrent
   scenario. A static check shipped now will protect those use cases by
   construction.

### What `device_to_device` looks like today

The function still directly schedules the transfer (the issue's premise
holds). In `arrayjit/lib/backends.ml`,
`Add_buffer_retrieval_and_syncing.device_to_device`:

```ocaml
let%track3_sexp device_to_device (tn : Tn.t) ~into_merge_buffer
    ~(dst : Backend.context) ~(src : Backend.context) =
  match Map.find src.ctx_arrays tn with
  | None -> false
  | Some s_arr -> (
      wait_for_ready ~dst ~src tn;
      match into_merge_buffer with
      | No -> ...
      | Copy ->
          Backend.(device_to_device tn ~into_merge_buffer
            ~dst_ptr:None ~dst ~src_ptr:s_arr ~src);
          update_writer_event dst @@ Merge_buffer tn;
          true)
```

The signature in `backend_intf.ml`:

```ocaml
val device_to_device :
  Tnode.t -> into_merge_buffer:merge_buffer_use ->
  dst:context -> src:context -> bool
```

The dynamic check in `backends.ml`, `check_merge_buffer`:

```ocaml
let check_merge_buffer stream ~code_node =
  match (stream.updating_for_merge_buffer, code_node) with
  | _, None -> ()
  | Some (actual, _), Some expected when Tnode.equal actual expected -> ()
  | _ -> raise @@ Utils.User_error ("Merge buffer mismatch ...")
```

This is invoked via `Task.prepend` inside `link` and `link_batch` —
i.e. it runs as the first work of the consumer's schedule, not at link
time.

### User-side call sites

There are **zero** call sites of `device_to_device` outside
`arrayjit/lib/`. PR #341 removed all of them, including the historical
`Train.parallel_update` data-parallel helper. The only mentions in
`lib/`/`bin/` are dead-stub references (e.g.
`arrayjit/lib/context.ml:258`, a placeholder `failwith`) and doc-comment
links (`arrayjit/lib/tnode.ml:20`).

This means the API change has no in-repo user impact today; the only
consumers are `init_from_device` and the no-op self-copy path inside
`backends.ml` itself, which can be migrated alongside the change.

### Relation to the sibling task `concise-merge-buffer-transfers`

The sibling proposal (`docs/proposals/concise-merge-buffer-transfers.md`,
covering README v1.0 line "Concise syntax for transfers into the merge
buffer") adds an additive `Backend.merge_from : 'context routine ->
src:'context -> bool` combinator that wraps `device_to_device` plus
`Task.run`. Its scope section explicitly defers #288 as complementary,
separate work.

The two tasks are *not* the same:

- `concise-merge-buffer-transfers` is **additive sugar** at the *call
  site* — it does not change `device_to_device`'s shape.
- This task (#288) is a **breaking change** to `device_to_device`'s
  shape that enables the static check.

If both ship, the natural composition is: `merge_from` becomes the
high-level call-site form, and internally it links the new routine-
returning `device_to_device` into the consuming routine. So this task is
*downstream* of the sibling, in the sense that the sibling sets the
preferred user-facing surface; but they can ship in either order.

### Code pointers (by symbol)

- `arrayjit/lib/backend_intf.ml`:
  - `merge_buffer_use` type (now `No | Copy` after #341)
  - `'context routine` record (with `merge_buffer_input`)
  - `Backend_device_common.device_to_device` signature
  - `stream_ref.updating_for_merge_buffer` (used by the dynamic check)
- `arrayjit/lib/backends.ml`:
  - `check_merge_buffer` (top of file) — the dynamic check this task
    aims to replace or supplement
  - `Add_buffer_retrieval_and_syncing.device_to_device` — the wrapper to
    convert
  - `link` / `link_batch` — where the static check should be installed,
    replacing the `Task.prepend` of `check_merge_buffer`
  - `code.expected_merge_node` / `code_batch.expected_merge_nodes` —
    what the static check compares against
  - `init_from_device` — internal user of `device_to_device` to migrate
- `arrayjit/lib/cc_backend.ml`, `cuda_backend.ml`, `metal_backend.ml`:
  per-backend `device_to_device` primitives (lower-level signature with
  `dst_ptr` / `src_ptr`); these stay the same shape but get invoked from
  inside a constructed `Task.t` rather than directly.
- `arrayjit/lib/lowered_backend_missing.ml:134` — stub backend; mirror
  whatever signature change lands.
- `git show 2858d249` — the commit that *removed* the previous
  half-static check; useful prior art for the implementer.

## Approach

*Tentative — agents may deviate.*

The recommended sequence is open-ended, but here is a plausible shape:

1. Change the high-level
   `Add_buffer_retrieval_and_syncing.device_to_device`
   in `backends.ml` to return `Backend.context routine option`. The
   constructed routine's `schedule : Task.t` performs the
   `wait_for_ready` + low-level
   `Backend.device_to_device` + `update_writer_event` work that the
   current implementation does inline. The routine's
   `merge_buffer_input` field is set to `Some tn` for the `Copy` case
   and `None` for `No`. The `inputs` set contains `tn`, the `outputs`
   set is empty (or `{tn}` for the `No` case).
2. Mirror the signature change in `backend_intf.ml`
   (`Backend_device_common.device_to_device`).
3. At `link` (and `link_batch`) time, when the consumer's
   `expected_merge_node` is `Some _`, additionally accept an optional
   producing-routine argument (or, equivalently, fold the producer
   routine's context into the consumer's context-chaining and check at
   that fold). Compare the producer routine's `merge_buffer_input`
   against the consumer's `expected_merge_node` *at link time*; raise
   `User_error` on mismatch.
4. Either leave `check_merge_buffer` in place as a defensive backstop
   (recommended for the first cut: static check is best-effort because
   only linked producer/consumer pairs are caught statically; ad-hoc
   transfers without a downstream link still need dynamic verification),
   or remove it once the test suite confirms the static check covers all
   currently-exercised paths. The proposal does not mandate removal.
5. Migrate `init_from_device` (the only in-tree user-facing-shaped call
   site) and the self-copy fast-path in `device_to_device`'s `No` branch
   to the new return type.
6. Coordinate with `concise-merge-buffer-transfers` if it has already
   landed: re-implement `Backend.merge_from` on top of the new
   routine-returning form (the change is local, and the call-site
   ergonomics from that proposal stay the same).

A *non-goal* of this approach: re-introducing the previous "require
`dst` to be the consumer's context" check (commit `2858d249`). That was
the half-static check the issue called out as backwards. The new check
runs at `link` between *producer routine* and *consumer code*, not at
`device_to_device` call time.

A point genuinely up to the implementer: whether the static check is
**mandatory** at link time (refuses to link a consumer with
`expected_merge_node = Some n` unless a producer with matching
`merge_buffer_input` has been linked first) or **best-effort** (checks
when a producer has been linked, otherwise falls through to the dynamic
check). The mandatory form is stricter but may force pervasive plumbing;
the best-effort form is the natural starting point.

## Scope

**In scope:**

- Change of `Backend_device_common.device_to_device` return type in
  `backend_intf.ml`.
- Corresponding implementation change in `backends.ml`'s
  `Add_buffer_retrieval_and_syncing.device_to_device`.
- Static check at `link` / `link_batch`.
- Migration of `init_from_device` and any other in-tree call site.
- Test demonstrating the static check fires before scheduling.
- `CHANGES.md` entry.

**Out of scope:**

- Concise call-site syntax (`Backend.merge_from`) — sibling task
  `concise-merge-buffer-transfers`.
- Reintroducing data-parallel `Train.parallel_update` or any other
  user-side merge-buffer call site — that is gated on v0.7 multi-process
  / multi-device parallelism work.
- Removing `merge_buffer_use` or the `Copy` variant.
- Touching the per-backend lower-level `device_to_device` primitives
  (`cc_backend.ml` etc.) other than to keep them callable from inside the
  new routine's schedule.
- Memory-mode sharing audit (#291), Universal Pool Allocator (#344),
  scope-init tracking (#340) — sibling v1.0 safety items, separate.

**Dependencies:** none hard. Soft coordination with
`concise-merge-buffer-transfers` (sibling proposal at
`docs/proposals/concise-merge-buffer-transfers.md`) — the two should
agree on the final user-facing surface. Either can land first.

**Open question for the user (not blocking this proposal but worth
flagging in review):** post-#341, the original concurrent-streams
motivation for static verification is gone. The remaining benefits
(earlier feedback, types-as-documentation, future-proofing) are real but
narrower. If the user prefers to defer this until v0.7 multi-process
parallelism returns concurrent producers/consumers to the picture, that
is a coherent alternative; the issue's premise about
`device_to_device` "directly scheduling" still holds, but the urgency
is reduced.

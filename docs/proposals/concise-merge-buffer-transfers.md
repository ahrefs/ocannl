# Concise syntax for merge buffer transfers

## Status update (2026-06-12)

- Still wanted: ROADMAP.md lists "Concise syntax for merge buffer transfers" under the v1.0 Ergonomics section. No `merge_from`-style helper exists yet, and there are still zero in-repo user-side call sites of `device_to_device` outside `arrayjit/lib`.
- Major API change invalidates the code snippets below: `device_to_device` no longer schedules the copy and returns `bool` — it now builds and returns a transfer routine, `context routine option` (commit 1162646d; signature at `arrayjit/lib/backend_intf.ml:307`). The caller runs `Task.run r.schedule` (or links a consumer against `r.context`).
- gh-ocannl-288 is CLOSED/COMPLETED: static merge-buffer verification landed as `check_merge_buffer_static` in `arrayjit/lib/backends.ml`, performed at link time against the `merge_buffer_node` recorded on the transfer routine's returned context. The "verification is dynamic-only today" framing and the "#288 is complementary, can ship without it" argument are obsolete; the dynamic `check_merge_buffer` remains as a residual run-time check.
- A related ergonomic helper landed for the non-merge case: `init_from_device` (`backend_intf.ml:328`, `backends.ml:~201`) wraps `device_to_device ~into_merge_buffer:No` for first-time placement.
- `merge_buffer_use = No | Copy` confirmed (`backend_intf.ml:43`); `Streaming_for` is indeed gone (#341 cleanup, commit 692d8c9d).
- Remaining work: redesign the sketch against the routine-returning API — a `merge_from`-style combinator would now run the transfer routine's schedule and then the consumer's, and should obtain/preserve the static verification by linking the consumer against the transfer routine's context. Acceptance criteria mentioning a `bool` return need adjusting accordingly.

## Goal

Provide a concise API for the "transfer a tensor node into another routine's
merge buffer, then run the routine" pattern, replacing the current
two-step `device_to_device ~into_merge_buffer:Copy ... ; Task.run merge.schedule`
sequence. Tracks the v1.0 README/ROADMAP item *"Concise syntax for transfers
into the merge buffer since we know which tensor node is transferred and
where to."*

The system already knows two things the user is forced to repeat at every
call site:

1. **Which tensor node** the merge buffer is for — the destination
   `routine.merge_buffer_input` field already records this.
2. **Where to put it** — running the destination routine is what consumes the
   merge buffer; transfers without a subsequent `Task.run` of the consuming
   routine are pointless.

## Acceptance Criteria

- [ ] A new function (or pair of closely related functions) in
  `arrayjit/lib/backends.ml` that performs "transfer a tensor node from a
  source context into a destination routine's merge buffer, and schedule that
  routine" as a single call. The function takes the destination
  `Backend.context routine` and the source `Backend.context` (and any other
  inputs implied by ergonomics), and reads the tensor node from
  `routine.merge_buffer_input`.
- [ ] Calling the new helper raises a clear `User_error` (or returns `false`)
  when `routine.merge_buffer_input = None`, when the tensor node is absent
  from the source context, or when the existing dynamic `check_merge_buffer`
  sees a mismatch — i.e. all current dynamic safety checks remain in force.
- [ ] At least one call site demonstrates the new API. Concretely:
  - Either a unit/integration test under `test/` or a demo under `bin/` that
    exercises a multi-context merge-buffer transfer using the new helper, or
  - A reintroduced `lib/train.ml` data-parallel helper (the historical
    `merge_grads`/`merge_loss` shape) using the new API as its primitive.
  - The chosen call site builds and passes `dune runtest`.
- [ ] Token-count win at the call site is measurable: the new form is at
  least **40% shorter** in source tokens than the equivalent
  `device_to_device ~into_merge_buffer:Copy ...; Task.run ...; check…`
  pair, when written naturally. (The historical `merge_grads` body in
  `Train.parallel_update` was ~5 lines per merge step plus a `Streaming_for`
  branch; the new form should fit in 1–2 lines.)
- [ ] The new helper is exported from `arrayjit/lib/backends.mli` (or from
  `Ir.Backend_intf` as part of the `Backend` signature, whichever matches the
  surrounding style) with a short doc comment that points to the underlying
  `device_to_device` for users who need the lower-level form.
- [ ] Existing `device_to_device` call sites and the `merge_buffer_use` type
  are not removed or renamed — this is additive sugar, not a breaking change.
- [ ] `CHANGES.md` records the addition under the v1.0 (or current) section.

## Context

### How merge buffer transfers work today

*(Update 2026-06-12: this section describes the pre-1162646d API. `device_to_device` now
returns a transfer routine — `context routine option` — instead of scheduling the copy and
returning `bool`; see the Status update above and `arrayjit/lib/backend_intf.ml:307`.)*

The low-level primitive is in `arrayjit/lib/backends.ml`,
`Add_buffer_retrieval_and_syncing.device_to_device`:

```ocaml
let%track3_sexp device_to_device (tn : Tn.t) ~into_merge_buffer ~(dst : Backend.context)
    ~(src : Backend.context) =
  match Map.find src.ctx_arrays tn with
  | None -> false
  | Some s_arr -> (
      wait_for_ready ~dst ~src tn;
      match into_merge_buffer with
      | No -> ...
      | Copy ->
          Backend.(device_to_device tn ~into_merge_buffer ~dst_ptr:None ~dst ~src_ptr:s_arr ~src);
          update_writer_event dst @@ Merge_buffer tn;
          true)
```

The signature is exposed via `Backend_intf.Backend_device_common`:

```ocaml
val device_to_device :
  Tnode.t -> into_merge_buffer:merge_buffer_use -> dst:context -> src:context -> bool
```

*(Update 2026-06-12: the return type is now `context routine option` — the function builds
a transfer routine that the caller schedules.)*

with `type merge_buffer_use = No | Copy` (`backend_intf.ml`, after the
multi-stream cleanup in PR #341 removed the `Streaming_for` variant).

The merge-buffer side of the destination routine is described by
`'context routine.merge_buffer_input : Tnode.t option` (`backend_intf.ml`),
populated at link time in `backends.ml` via
`Low_level.input_and_output_nodes`. The *consistency check* at run time lives
in `check_merge_buffer` (`backends.ml`, top of file): it compares
`stream.updating_for_merge_buffer` against the routine's expected
merge-buffer node and raises `User_error` on mismatch.

The `%cd` syntax extension already exposes a concise *read* side via the
`.merge` pseudo-field — `[%cd p.grad =+ p.grad.merge]`
(`docs/syntax_extensions.md` § *Referencing arrays*,
`docs/anatomy_of_a_backend.md` § merge buffers). What is missing is a
correspondingly concise *transfer* side.

### Historical user pattern

Before multi-streaming was removed (PR #341, commit `692d8c9d`),
`lib/train.ml` contained the canonical user-facing pattern. Reconstructed
from `git show 77b7d5a9~1:lib/train.ml`:

```ocaml
let merge_grads ~(from : int) ~(to_ : int) : unit =
  Array.iteri all_params ~f:(fun i p ->
      let grad_merge =
        Option.value_exn ~here:[%here] grad_merges_to.(to_).(i)
      in
      let into_merge_buffer, streaming = mbuf_use grad_merge.schedule in
      assert (
        Backend.device_to_device (Option.value_exn ~here:[%here] p.diff).grad
          ~into_merge_buffer ~dst:ctxs.(to_) ~src:ctxs.(from));
      if not streaming then Task.run grad_merge.schedule)
```

After #341 the `streaming` branch goes away; the residual verbose pattern is:

```ocaml
assert (
  Backend.device_to_device (Option.value_exn ~here:[%here] p.diff).grad
    ~into_merge_buffer:Copy ~dst:grad_merge.context ~src:src_ctx);
Task.run grad_merge.schedule
```

The user must (a) hand-write the tensor node `(Option.value_exn p.diff).grad`,
even though `grad_merge.merge_buffer_input` already names exactly that node;
(b) thread the destination context as `~dst:grad_merge.context` and (c)
remember to run the routine. There are currently **zero** in-repo call sites
of `device_to_device` outside `arrayjit/lib`, because the user-side helper
that used to exercise it was removed with multi-streaming. New call sites
will return when v0.7.x reintroduces multi-process / multi-device data
parallelism, or sooner if a federation/transformer demo wants merge buffers.

### Relation to gh-ocannl-288 (static merge buffer verification)

*(Update 2026-06-12: #288 is CLOSED/COMPLETED. Static verification now exists as
`check_merge_buffer_static` in `backends.ml`, checked at link time against the
`merge_buffer_node` carried by the context that the `device_to_device` transfer routine
returns. The paragraph below is the pre-#288 state of play; the dynamic check remains
as a residual run-time guard. The new sugar should be designed to flow through the
static check — i.e. link consumers against the transfer routine's context.)*

#288 asks for *static* (compile-time / shape-time) verification that
merge-buffer producers and consumers agree on the tensor node. The current
codebase already performs *dynamic* verification in `check_merge_buffer`
(`backends.ml`, line ~15) and in `update_writer_event`'s
`Merge_buffer` branch. The new sugar can ship without #288: it
relies on the same `merge_buffer_input` field, and any mismatch will be
caught dynamically by the existing check. #288 is complementary — it
upgrades the safety net but does not gate this ergonomic change.

### Code pointers (by symbol, not line)

- `arrayjit/lib/backend_intf.ml`: types `merge_buffer_use`, `routine`
  (with `merge_buffer_input`), and the `Backend_device_common` signature
  declaring `device_to_device`.
- `arrayjit/lib/backends.ml`:
  - `Add_buffer_retrieval_and_syncing.device_to_device` — the existing
    transfer wrapper that handles `wait_for_ready`, `update_writer_event`,
    and the `Copy` branch.
  - `check_merge_buffer` — dynamic consistency check; the new helper should
    flow through paths that preserve this guarantee.
  - `Add_buffer_retrieval_and_syncing.sync_routine` — shows the existing
    pattern for wrapping a routine's `schedule` with pre/post tasks; the new
    helper may use a similar shape.
- `arrayjit/lib/backends.mli` — currently exposes only `finalize` and
  `fresh_backend`. The signature `Backend_device_common.device_to_device`
  flows out via `Ir.Backend_intf`. The new helper should be exposed in the
  same place as `device_to_device`.
- `lib/train.ml` — historically the primary consumer; likely site for the
  demonstration call-site once data-parallelism returns. (The current file
  has no merge-buffer code post-#341.)
- `docs/syntax_extensions.md` § *Referencing arrays* — the existing
  `tensor.merge` pseudo-field; the proposal complements but does not modify
  it.
- `docs/anatomy_of_a_backend.md` § merge buffers — narrative description of
  the `Copy` mode and dynamic verification.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

*(Update 2026-06-12: the sketch below predates the routine-returning `device_to_device`.
An updated `merge_from` would pattern-match on `device_to_device tn ~into_merge_buffer:Copy
~dst:r.context ~src` returning `Some transfer`, run `Task.run transfer.schedule` then
`Task.run r.schedule`, and ideally arrange for `r` to be linked against `transfer.context`
so the static merge-buffer check (#288, landed) applies. The ergonomic motivation is
unchanged — if anything the raw pattern is now slightly longer.)*

Add a new helper to `Add_buffer_retrieval_and_syncing` in `backends.ml`
roughly along these lines:

```ocaml
(** Transfer the routine's [merge_buffer_input] node from [src] into [r]'s
    merge buffer, then schedule [r]. Returns [false] without scheduling if
    the routine has no merge-buffer input or the node is absent from [src].
    Raises [User_error] on a merge-buffer node mismatch (delegated to the
    existing dynamic check). *)
val merge_from : 'context routine -> src:'context -> bool
```

Implementation sketch:

```ocaml
let merge_from (r : Backend.context routine) ~(src : Backend.context) : bool =
  match r.merge_buffer_input with
  | None -> false
  | Some tn ->
      if device_to_device tn ~into_merge_buffer:Copy ~dst:r.context ~src then begin
        Task.run r.schedule;
        true
      end else false
```

so the user-side call collapses from

```ocaml
assert (
  Backend.device_to_device (Option.value_exn p.diff).grad
    ~into_merge_buffer:Copy ~dst:grad_merge.context ~src);
Task.run grad_merge.schedule
```

(roughly 90 source tokens) to

```ocaml
assert (Backend.merge_from grad_merge ~src)
```

(roughly 12 tokens; ~85% reduction at the call site).

Open detail (left to the implementer): whether the helper should additionally
expose a "transfer-only, no-run" variant
(`val merge_into : 'context routine -> src:'context -> bool`) for users who
want to schedule the transfer and run the routine separately (e.g. interleave
multiple transfers before consuming any). The historical `Train.parallel_update`
code did exactly this (transfer all, then run the merge routine implicitly via
a sync), so the split form has at least one motivating use case. The
recommended initial cut: ship `merge_from` only; add `merge_into` if a real
call site needs it.

PPX-level changes are *not* recommended here. The existing `.merge`
pseudo-field already covers the read side; the transfer side is plain backend
API, and a `%cd`-level extension would entangle two layers (assignment IR vs
runtime scheduling) for marginal additional concision. A combinator on
`routine` is the lighter-weight choice and matches surrounding style
(`sync_routine`, `link`, etc.).

## Scope

**In scope:**
- New combinator(s) in `backends.ml` plus mirrored declaration in the
  signature exposing `device_to_device`.
- One demonstration call-site (test, bin example, or reintroduced
  `Train` helper).
- `CHANGES.md` entry; a short paragraph in `docs/anatomy_of_a_backend.md` or
  `docs/syntax_extensions.md` if the agent judges it useful.

**Out of scope:**
- Static merge-buffer verification (#288) — complementary, separate task.
- A `%cd`-level extension for the transfer side.
- Removing or deprecating `device_to_device` itself.
- Reintroducing data-parallel `Train.parallel_update` — touched only if it
  is the natural home for the demonstration call-site, and even then only
  the merge-buffer-call shape, not the broader scheduler.
- Pool-allocator / loop-hoisting changes (separate v0.7.2 / v0.8 work).

**Dependencies:** none. Builds on existing `merge_buffer_input` and the
dynamic `check_merge_buffer` already in `backends.ml`.

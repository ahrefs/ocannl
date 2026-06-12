# Re-elaborate sharding primitives after the multi-streaming cleanup (#341)

**Task**: task-a2c331e9 (subtask 293b of [gh-ocannl-293](gh-ocannl-293.md))
**Date**: 2026-06-12 (refreshes the 2026-04-25 harness elaboration)
**Status**: Decided — user selected **Outcome 2 (per-shard backend contexts)** on
2026-06-12. Follow-through: 293c (`task-2445dd1c`) re-elaborated against the sketch below;
GH issue #293 milestone aligned to v1.0; Outcome 3 recorded as a v1.x revisit note (no
tracking task).
**Blocks**: task-2445dd1c (293c, training-loop integration).

## Goal

Decide the post-#341 sharding story for OCANNL. The original Phase 2 sketch of #293
(`shard`, `gather`, automatic gradient synchronization) assumed cross-stream sharing as
a language feature; [#341](https://github.com/ahrefs/ocannl/issues/341) (closed,
completed) removed that. Three candidate outcomes:

1. **Close as not-planned** — sharding is out of scope for the single-process model.
2. **Per-shard backend contexts** — one stream/queue/domain per shard,
   host-orchestrated synchronization.
3. **Multi-process orchestration** — shards as separate processes.

## Selected verdict: Outcome 2 (per-shard backend contexts)

> **Verdict (2026-06-12)**: the user selected **Outcome 2**. The recommendation below
> stands as the rationale of record. The accompanying decisions: align the #293 GH issue
> milestone to **v1.0** (matching ROADMAP), and keep Outcome 3 (multi-process) as a
> one-line v1.x revisit note rather than a tracked task — see "Questions for the user"
> below for the resolved answers.

### The key factual correction

#341 removed cross-stream *coherence*, not multi-streaming itself. Verified against the
current code (2026-06-12):

Still present and load-bearing for sharding:
- **Multiple streams per device**: `backend_intf.ml:99` `device_ref.streams` is a
  `weak_dynarray`; `new_stream` (`backend_intf.ml:281`) is part of the backend
  interface. CUDA creates real streams (`Cu.Stream.create ~non_blocking:true`), Metal
  creates command queues, the `Multicore` scheduler spawns one domain per stream
  (`schedulers.ml:174` `Domain.spawn worker`).
- **Multi-device**: `num_devices`, `get_device ~ordinal`, per-device
  `constant_buffer_cache` (note: the April elaboration called this
  `device_buffer_cache`; it has since been renamed).
- **Per-stream merge buffer**: `merge_buffer_use = No | Copy` (`backend_intf.ml:43`)
  and `device_to_device tn ~into_merge_buffer:Copy ~dst ~src` — the one cross-stream
  primitive that survived. **Update since April**: `device_to_device` now *returns a
  transfer routine* (`backend_intf.ml:307`, merge-buffer static-verification work,
  2026-05) rather than scheduling a copy imperatively. This strengthens the Outcome 2
  story: gather/grad-sync steps are routines composable with compute routines under the
  same static verification.
- **Host-side sync**: `await`, `sync`, `will_wait_for`, `all_work`.

Removed by #341 (do not design against): *automatic* cross-stream writer tracking
(`wait_for_all` and the cross-stream branch of `update_writer_event`),
`Shared_cross_streams` sharing, the `Streaming_for` merge-buffer variant, the
stream-count `config`, and `train.ml`'s `round_robin` driver. Two streams writing the
same tnode is now undefined behavior — so shards must own disjoint buffers, with the
merge-buffer copy as the only cross-stream channel. That is exactly the Outcome 2 shape.

**Factual correction (verified 2026-06-12)**: `wait_for_ready` was *not* removed — it
survives at `backends.ml:48` and is invoked from the `device_to_device` transfer path
(`backends.ml:150,183,209`) as an *explicit*, host-orchestrated cross-stream wait. Only
the *implicit* writer tracking went away. This strengthens Outcome 2: the explicit
"dst stream waits for src stream's transfer to be ready" primitive a per-shard design
needs is already in the interface, no resurrection required.

### Why not Outcome 1

Closing as not-planned would walk away from working infrastructure: the primitives a
per-shard design needs are all still in place; only the *implicit* coherence died, and
the redesign never needed it. Outcome 1 is defensible only as a strategic statement
("OCANNL doesn't compete on distributed training") — a product call, not a technical
conclusion.

### Why not Outcome 3

OCANNL has no in-tree multi-process machinery (no fork/IPC/Eio in `arrayjit/` or
`lib/`); building job submission, cross-process init, and checkpointing just to enable
sharding is a milestone-sized greenfield project for benefits Outcome 2 already delivers
in-process. Revisit if multi-node training becomes a v1.x goal.

## Outcome 2 design sketch (input for 293c's elaboration)

```ocaml
(* lib/train.ml or new lib/parallel.ml *)
val shard_along : axis:int -> n_shards:int -> Tensor.t -> Tensor.t array
val gather : axis:int -> Tensor.t array -> Tensor.t
val grad_sync : Tensor.t array -> unit  (* all-reduce via merge-buffer copies *)
```

- Staging: (1) one stream per shard on one device (data parallelism on a single GPU /
  one domain per shard on CPU); (2) shards across devices via `get_device ~ordinal`
  (`shard i` on `device (i mod num_devices)`).
- Synchronization is explicit at `gather`/`grad_sync` boundaries: per-shard
  `device_to_device ~into_merge_buffer:Copy` routines + `await`. No new tnode sharing
  mode needed — each shard's tnodes live on exactly one stream by construction.
- `Train.grad_update ~setup_for_parallel` (`lib/train.ml:84`) still exists as a hook
  from the old driver and is a natural integration point.
- Dependency on 293a (slice-as-alias, [task-e4003e5f.md](task-e4003e5f.md)): not a hard
  blocker — sharding can ship with copy-on-shard initially — but alias views make
  `shard_along` allocation-free; natural sequencing remains 293a → 293b → 293c.

## Edge cases for the 293c elaboration

- Per-shard RNG seeding (`Embed_self_id` / random-seed machinery) must diverge per
  shard, or data-parallel shards compute identical batches.
- Backward through `gather` is `shard` and vice versa; expressible in `%cd`, but a fused
  `grad_body` may be warranted.
- Gradient reduction (sum vs mean) across shards lives in a `Train.grad_sync` helper,
  not in backends.
- Pipeline parallelism (heterogeneous shards) is 293c territory; this design only fixes
  the primitive layer.

## Questions for the user (the verdict itself) — resolved 2026-06-12

1. **Commit to Outcome 2?** → **Yes, Outcome 2.** Per-shard backend contexts with
   host-orchestrated synchronization. Outcome 1 (drop distributed training) was not
   chosen.
2. **Milestones**: align the GH issue? → **Align the GH issue to v1.0** (matching
   ROADMAP). Done — issue #293 milestone moved from v0.8 to v1.0.
3. **Outcome 3 later?** → **Add the one-line note.** Recorded as: "Revisit Outcome 3
   (multi-process orchestration) for multi-node training in v1.x" — see "Why not
   Outcome 3" above and the umbrella ([gh-ocannl-293.md](gh-ocannl-293.md)). No tracking
   task yet.

## Acceptance Criteria

- [x] User picks an outcome (this document records the recommendation, evidence, and the
      selected verdict: Outcome 2 — see "Selected verdict" and the resolved questions).
- [x] On Outcome 2: 293c (`task-2445dd1c`) re-elaborated against the sketch above; GH
      issue milestone aligned with ROADMAP (moved to v1.0).
- [ ] On Outcome 1: 293b and 293c closed as not-planned; umbrella #293 closed; ROADMAP
      v1.0 entry removed. — N/A: Outcome 1 was not selected.

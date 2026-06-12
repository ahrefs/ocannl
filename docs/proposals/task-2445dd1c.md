# Training-loop sharding integration (data/pipeline parallelism)

**Task**: task-2445dd1c (subtask 293c of [gh-ocannl-293](gh-ocannl-293.md))
**Date**: 2026-06-12 (re-elaborated after 293b's verdict)
**Status**: Ready — 293b ([task-a2c331e9.md](task-a2c331e9.md)) selected **Outcome 2
(per-shard backend contexts)** on 2026-06-12. This task is now elaborated against that
verdict's design sketch.

## Scope (from the parent issue's Phase 3)

Support common sharding patterns at the training-loop layer (`lib/train.ml`): data
parallelism on the batch dimension first, pipeline parallelism layer-wise as a possible
follow-up split. The previous driver, `round_robin`, was removed with the multi-streaming
cleanup (#341); its remaining hook is `Train.grad_update ~setup_for_parallel`
(`lib/train.ml:84`).

## Design (resolved: build on 293b's Outcome 2)

293b *designed* the primitive layer (it was an elaboration-only verdict task and ships no
code): shards are explicit per-shard backend contexts (one stream/queue/domain per
shard), each owning disjoint tnode buffers, with the per-stream merge buffer
(`device_to_device tn ~into_merge_buffer:Copy`, now a transfer *routine*) as the only
cross-stream channel, and `wait_for_ready` (`backends.ml:48`) as the explicit
host-orchestrated synchronization point. **293c owns both halves of the implementation**:
it builds the sharding primitives per the 293b sketch *and* wires them into the training
loop. (293b records the design but commits no code, so no other open task is responsible
for these APIs — they live here.)

The primitives 293c implements (in `lib/parallel.ml` or `lib/train.ml`), per the 293b
sketch:

```ocaml
val shard_along : axis:int -> n_shards:int -> Tensor.t -> Tensor.t array
val gather : axis:int -> Tensor.t array -> Tensor.t
val grad_sync : Tensor.t array -> unit  (* all-reduce via merge-buffer copies *)
```

### Data parallelism (first deliverable)

1. **Shard placement.** For `n_shards` on a backend with `num_devices` devices, place
   `shard i` on `get_device ~ordinal:(i mod num_devices)`, one `new_stream` per shard.
   On a single device this is one stream per shard (data parallelism on one GPU); on CPU
   one `Multicore` domain per shard.
2. **Per-shard routines.** Compile the forward+backward graph once per shard against that
   shard's context, so each shard's intermediate tnodes live on exactly one stream by
   construction (no `Shared_cross_streams` mode needed — that died with #341).
3. **Gradient synchronization.** Between the per-shard backward pass and the optimizer
   step, run `grad_sync`: an all-reduce of each parameter's gradient across shards,
   implemented as `device_to_device ~into_merge_buffer:Copy` transfer routines + the
   reduction, gated by `wait_for_ready`. Reduction mode (sum vs mean) is a `Train`-level
   helper argument, not a backend concern.
4. **Driver.** An epoch/batch driver that splits each batch along the batch axis with
   `shard_along`, runs the per-shard routines, calls `grad_sync`, then a single optimizer
   step on the synchronized gradients. This replaces what `round_robin` used to do;
   `Train.grad_update ~setup_for_parallel:true` (`lib/train.ml:84`) is the integration
   hook.
5. **Per-shard RNG.** Data-parallel shards must seed RNG divergently (`Embed_self_id` /
   random-seed machinery) or every shard computes the identical batch. The driver sets a
   per-shard seed offset.

### Pipeline parallelism (second, harder half — may split out)

Heterogeneous shards running different layers, with activations/gradients handed between
stages over merge-buffer transfers. This is strictly more complex (stage scheduling,
micro-batching, bubble management) and may be split into its own subtask if it grows; the
data-parallel deliverable above does not depend on it.

## Out of scope

- The *design* of the sharding primitives (which outcome, which backend mechanisms) —
  that was 293b's verdict ([task-a2c331e9.md](task-a2c331e9.md)). Their *implementation*
  is in-scope here (see Design above); 293b ships no code.
- Slice-as-alias-view ([task-e4003e5f.md](task-e4003e5f.md), independent; alias views
  make `shard_along` allocation-free but are not a hard blocker — sharding can ship
  copy-on-shard initially).
- Multi-process / multi-node orchestration (Outcome 3, deferred to a v1.x revisit).

## Dependencies

- **293b verdict**: satisfied (Outcome 2 selected 2026-06-12).
- **293a (`task-e4003e5f`)**: soft — copy-on-shard works without it; alias views are an
  optimization. Natural sequencing 293a → 293b → 293c stands but 293c is not blocked.

## Acceptance Criteria

- [x] The sharding primitives `shard_along` / `gather` / `grad_sync` are implemented (in
      new `lib/parallel.ml`) per the 293b sketch — this task owns them, as 293b was
      verdict-only. `shard_along` / `gather` are public functions; `grad_sync` is realized
      as the data-parallel session's all-reduce (exposed on the session handle as
      `grad_sync : unit -> unit`, and run by `step`). Its signature differs from the
      sketch's `Tensor.t array -> unit` because the raw-backend layer needs the per-shard
      *contexts* (which a bare `Tensor.t array` does not carry); the all-reduce core is
      tested in isolation in `test/operations/shard_transfer.ml`.
- [x] Those primitives are wired into a data-parallel training driver
      (`Parallel.data_parallel`).
- [x] Per-shard forward+backward routines compile against per-shard backend contexts
      (one stream per shard via `new_stream` / `get_device`); each shard owns disjoint
      tnode buffers (on single-device unified-memory backends a shared hosted tnode cannot
      hold distinct per-shard data, so each shard rebuilds the model over its own tnodes).
- [x] `grad_sync` runs between backward and the optimizer step, all-reducing parameter
      gradients across shards via `device_to_device ~into_merge_buffer:Copy` transfer
      routines (internally gated by `wait_for_ready`) + a `%cd` accumulation, with a
      selectable `Sum` / `Mean` reduction.
- [x] Per-shard RNG seeding diverges: each shard's graph is built after
      `set_random_seed ~seed:(base_seed + shard_id)` (scoped by `Tensor.with_saved_random_seed`
      so the caller's global seed is restored), and the data slices are themselves distinct
      (`shard_along`). Verified by the *driver-level* RNG test in
      `test/training/data_parallel.ml` ("driver routes per-shard seed into RNG", which flips
      if the driver stops routing `base_seed` into the shard seeds) plus a transient-mutation
      check ("global random-seed singleton preserved across data_parallel").
- [x] A `test/training/` test (`data_parallel.ml`) trains a small model data-parallel
      across 2 shards and checks parameter parity against the single-shard baseline.
- [x] Pipeline parallelism is **split into a follow-up task** with a recorded rationale
      (see "Pipeline parallelism" below).

## Pipeline parallelism — split to a follow-up (decision, 2026-06-12)

Pipeline parallelism is **not** delivered in 293c; it is split into a dedicated follow-up
subtask of #293. Rationale: it is strictly more complex than data parallelism (heterogeneous
shards running different layers, a stage scheduler, activation/gradient hand-off contracts
between stages, micro-batching, and bubble management), it shares **none** of the
data-parallel all-reduce machinery, and the data-parallel deliverable does not depend on it.
The data-parallel primitives in `lib/parallel.ml` (`shard_along` / `gather` / merge-buffer
transfers) are reusable building blocks for it but impose no design constraint. No tracking
task file is created yet (the work is recorded here under the #293 umbrella); open one when
pipeline parallelism is scheduled.

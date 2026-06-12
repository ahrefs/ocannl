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

293b fixed the primitive layer: shards are explicit per-shard backend contexts (one
stream/queue/domain per shard), each owning disjoint tnode buffers, with the per-stream
merge buffer (`device_to_device tn ~into_merge_buffer:Copy`, now a transfer *routine*) as
the only cross-stream channel, and `wait_for_ready` (`backends.ml:48`) as the explicit
host-orchestrated synchronization point. 293c wires those into the training loop.

The primitives 293b proposes for 293c to consume (`lib/parallel.ml` or `lib/train.ml`):

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

- The sharding primitives themselves (`shard_along` / `gather` / `grad_sync`) — those are
  293b territory ([task-a2c331e9.md](task-a2c331e9.md)); 293c consumes them.
- Slice-as-alias-view ([task-e4003e5f.md](task-e4003e5f.md), independent; alias views
  make `shard_along` allocation-free but are not a hard blocker — sharding can ship
  copy-on-shard initially).
- Multi-process / multi-node orchestration (Outcome 3, deferred to a v1.x revisit).

## Dependencies

- **293b verdict**: satisfied (Outcome 2 selected 2026-06-12).
- **293a (`task-e4003e5f`)**: soft — copy-on-shard works without it; alias views are an
  optimization. Natural sequencing 293a → 293b → 293c stands but 293c is not blocked.

## Acceptance Criteria

- [ ] `shard_along` / `gather` / `grad_sync` (from 293b) are wired into a data-parallel
      training driver in `lib/train.ml` (or new `lib/parallel.ml`).
- [ ] Per-shard forward+backward routines compile against per-shard backend contexts
      (one stream/queue/domain per shard); each shard owns disjoint tnode buffers.
- [ ] `grad_sync` runs between backward and the optimizer step, all-reducing parameter
      gradients across shards via merge-buffer transfer routines + `wait_for_ready`, with
      a selectable sum/mean reduction.
- [ ] Per-shard RNG seeding diverges so data-parallel shards process distinct batch
      slices.
- [ ] A `test/training/` test trains a small model data-parallel across ≥2 shards and
      checks parity (within tolerance) against the single-shard baseline.
- [ ] Pipeline parallelism is either delivered or explicitly split into a follow-up task
      with a recorded rationale.

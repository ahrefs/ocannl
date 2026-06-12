# Training-loop sharding integration (data/pipeline parallelism)

**Task**: task-2445dd1c (subtask 293c of [gh-ocannl-293](gh-ocannl-293.md))
**Date**: 2026-06-12
**Status**: Blocked on 293b's verdict ([task-a2c331e9.md](task-a2c331e9.md)). This is
deliberately a stub — elaborating it before the verdict would be wasted work.

## Scope (from the parent issue's Phase 3)

Support common sharding patterns at the training-loop layer (`lib/train.ml`): data
parallelism on the batch dimension, pipeline parallelism layer-wise. The previous
driver, `round_robin`, was removed with the multi-streaming cleanup (#341); its
remaining hook is `Train.grad_update ~setup_for_parallel` (`lib/train.ml:84`).

## Decision tree (resume condition: 293b is decided)

- 293b closes as **not-planned** → close this task too; the #293 umbrella closes.
- 293b picks **per-shard backend contexts** (the current recommendation) → elaborate
  this task as: wire `shard_along` / `gather` / `grad_sync` into the training loop —
  per-shard forward+backward routines, `grad_sync` between backward and optimizer step,
  epoch/batch drivers replacing what `round_robin` used to do. Pipeline parallelism
  (different shards run different layers) is the second, harder half and may split out.
- 293b picks **multi-process orchestration** → elaborate against that design instead
  (out-of-process coordination layer; currently considered unlikely).

## Out of scope

- The sharding primitives themselves (293b territory).
- Slice-as-alias-view ([task-e4003e5f.md](task-e4003e5f.md), independent).

## Acceptance Criteria

- [ ] TBD — written during re-elaboration once 293b's verdict is in.

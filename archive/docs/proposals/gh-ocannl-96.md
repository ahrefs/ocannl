# Training checkpointing

Issue: [#96](https://github.com/ahrefs/ocannl/issues/96)
ROADMAP: v1.0 completeness

## Goal

Resume a training run in a fresh process with the same model parameters,
optimizer state, step-dependent behavior, and user-supplied loop state.

## Current state

`Persistence.save/load/restore` already transfers arbitrary tnode sets through
an explicit `Context.t`, supports namespaces, and validates precision/shape.
The missing piece is not another tensor file format:

- `Train.sgd_one` creates momentum tensors inside `%cd` without returning
  first-class optimizer state;
- loop progress and host data-generator RNG state belong to user code;
- restore matches `(namespace, id)`, so a fresh process must rebuild the same
  graph deterministically before restoring, or use an explicit name-mapping
  migration layer.

The old proposal's global `Train.checkpoint_save path` hid these constraints
and prescribed a fixed set of epoch/loss-list fields. Checkpoint state must be
explicit instead.

## Direction

1. Introduce an optimizer handle (or additive SGD constructor) that exposes
   both the update computation and its persistent state tnodes. Preserve the
   existing `sgd_update` convenience API.
2. Define a small checkpoint manifest that combines a tensor set with
   versioned user metadata. The metadata codec is supplied by the training
   program; OCANNL should not define universal epoch, sampler, or history
   fields.
3. Document the reconstruction contract: same model code/configuration,
   deterministic tnode creation and namespace, then `Persistence.restore`.
   Cross-version/name-based migration is separate work.
4. Include every stateful tensor used by the run: parameters, optimizer state,
   explicit train-step tensors, and the graph's random seed where relevant.
   Host `Random.State` or dataset cursors live in metadata.

## Completion criteria

- A momentum optimizer exposes a complete, inspectable persistent-state set.
- A two-process test compares uninterrupted `N+M` training with
  save-after-`N`, rebuild, restore, and `M` more steps on the same backend.
  Parameters, momentum, next-step loss, and step-dependent random behavior
  match under a stated deterministic contract.
- Missing/mismatched tensors, incompatible manifest versions, and wrong model
  reconstruction fail with useful messages.
- The format and compatibility boundary are documented.
- Checkpointing remains context-explicit and works without global registries.

Distributed/sharded checkpoints, cross-version migration, and arbitrary
external tracker history are follow-ups.

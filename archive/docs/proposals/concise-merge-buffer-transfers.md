# Concise merge-buffer transfers

ROADMAP item: concise syntax for transfers into the merge buffer.

## Current state

`Backend.device_to_device ... ~into_merge_buffer:Copy` now returns a transfer
routine. Its returned context records the transferred node, and
`Backend.link transfer.context consumer_code` statically checks that the
consumer reads the same merge-buffer node.

`lib/parallel.ml` contains the first real user of this protocol. Its private
`merge_transfer` helper schedules the transfer, links the consumer against the
transfer context, and schedules the consumer. The old proposal predated this
API and proposed accepting an already-linked consumer routine; that would be
the wrong abstraction because the static check happens while linking.

## Goal

Extract the proven `lib/parallel.ml` pattern into the backend API if another
caller needs it. The helper should compose a transfer with an **unlinked
consumer code value**, preserving the static merge-node check and returning
one routine (or an equivalent scheduleable value).

This is deliberately a small convenience, not new transfer semantics.

## Completion criteria

- A backend helper represents “copy `tn` from `src` into `dst`'s merge buffer,
  then run this consumer code” without exposing the intermediate context.
- The helper links the consumer through the transfer context; mismatched
  merge-buffer nodes still fail at link time.
- `lib/parallel.ml` uses the helper, removing its private orchestration
  duplicate.
- A test covers success, a missing source node, and a mismatched consumer.
- Low-level `device_to_device` remains available for callers that need to
  compose more than one routine.

## Scope guidance

Do not add syntax to `%cd`: `.merge` is already the concise read side. Do not
accept an already-linked routine, add token-count targets, or resurrect the
removed streaming modes. If `lib/parallel.ml` remains the only caller, keeping
the helper private is preferable to adding API surface; revisit this proposal
when a second caller appears.

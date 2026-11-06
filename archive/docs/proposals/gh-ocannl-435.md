# Simply and NanoDO deep dive

Issue: [#435](https://github.com/ahrefs/ocannl/issues/435)

## Goal

Compare Simply and NanoDO with OCANNL's user-facing model/training layer and
extract a small number of actionable lessons for `lib/`.

## Updated OCANNL baseline

The comparison must include the current library, not the 2026-03 snapshot:
decoder-only blocks and positional encodings, GPT-2 pretrained inference,
precision policy, safetensors/persistence, data-parallel training, and the
schedule/autotuning layer now exist. Missing breadth in attention, optimizer,
sharding, inference caching, and RL should be described without assuming each
gap belongs in core OCANNL.

## Deliverable

Record the exact Simply and NanoDO revisions and write a concise note covering:

- parameter ownership and module composition;
- shape/einsum expression and agent readability;
- attention, FFN, normalization, positional, and KV-cache choices;
- mixed precision, sharding, training, checkpoint loading, and RL boundaries;
- what NanoDO deliberately omits to stay small;
- where OCANNL's compiled DSL changes the trade-off rather than merely lagging
  a JAX implementation.

End with no more than a handful of recommendations, each classified as:
already covered, a concrete candidate with a validation use case, or
intentionally outside `lib/`. Code pointers and evidence matter more than an
eleven-axis compliance matrix.

The study is complete when the durable note is committed and #435 receives a
summary/link. Follow-up issues are optional and should be filed only for
recommendations with a clear acceptance test.

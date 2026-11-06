# Small-transformer digit addition

Issue: [#427](https://github.com/ahrefs/ocannl/issues/427)

## Goal

Reproduce a published small-transformer addition result closely enough to
separate framework limitations from training-recipe details.

## Corrected framing

OCANNL already has decoder-only training examples, causal attention, positional
encodings, parallel schedules, and tensor-core paths. This is now an
algorithmic-learning experiment, not infrastructure bring-up.

“Grokking” should not be claimed from an on-the-fly infinite data stream.
Either reproduce held-out addition accuracy on generated examples, or use a
fixed training subset and separately show the delayed train/generalization
transition that the term grokking denotes.

Likewise, do not require a 500–1000 parameter count while using generic block
defaults that exceed it. Select one reference paper/configuration, report
OCANNL's exact trainable count, and explain any architectural deviation.

## Direction

- Generate reversed-result addition sequences and mask loss to the answer
  region.
- Start from `transformer_names.ml`, but match the selected reference's
  embeddings, normalization, weight tying, and optimizer rather than assuming
  SGD plus a generic `decoder_only_block` is equivalent.
- Measure exact-sequence accuracy on fresh examples and include per-digit/carry
  diagnostics when convergence fails.
- Separate a fast deterministic data/model smoke test from the long training
  reproduction under `@slow`.

## Completion criteria

- Data encoding, masking, and exact-match evaluation have regular tests.
- A fixed-seed slow run reaches a stated held-out accuracy on the selected
  digit width and records model size and training budget.
- The checked output distinguishes reproduction, partial reproduction, and a
  negative result; it does not weaken CI to a token-level threshold and call it
  addition.
- The write-up cites the precise reference result and documents deviations.

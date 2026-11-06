# LSTM example

Issue: [#60](https://github.com/ahrefs/ocannl/issues/60)
ROADMAP: v0.9 examples

## Goal

Add a reusable LSTM cell and demonstrate recurrent parameter sharing with a
small Names language-model example.

## Current codebase fit

`%op`'s `()` lifting boundary creates a cell's parameters once; repeatedly
applying the resulting closure shares them across unrolled time steps.
`Operation.slice` and `Operation.stack` provide timestep extraction and output
assembly. The Names and transformer examples provide data, loss, and sampling
scaffolding.

A new sigmoid primitive is not required for this task. The stable identity
`sigmoid x = 0.5 * (tanh (0.5 * x) + 1)` uses the existing tanh operation and
keeps backend scope small.

## Direction

- Add `lstm_cell` with the standard input, forget, output, and candidate gates,
  returning `(h, c)`.
- Add a fixed-length unroll helper only if it is clearer than a short loop in
  the example. Keep sequence packing/padding policy outside the cell.
- Verify sharing by parameter identities/count, not only by observing similar
  outputs.
- Adapt the Names setup for training and deterministic evaluation. Put the
  convergence run under `@slow`; keep cell shape/gradient/sharing tests in the
  regular suite.

## Completion criteria

- Cell forward values and gradients match a small independent reference.
- An unroll over multiple steps has one parameter set, not one set per step,
  and returns correctly shaped stacked outputs.
- A fixed-seed slow example improves held-out next-character loss over a simple
  baseline and can sample through the trained recurrent model.
- Public API docs explain state shapes and parameter-sharing behavior.

Generated names are useful qualitative output, not a sufficient regression
assertion. Bidirectional/multilayer LSTMs, packed variable-length sequences,
and a general recurrent runtime are follow-ups.

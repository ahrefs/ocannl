# Reproduce “Growing Sparse Computational Graphs with RNNs”

Issue: [#182](https://github.com/ahrefs/ocannl/issues/182)

## Goal

Reproduce one small task from the Bonsai RNN work: train a tiny recurrent
model, prune small weights while retaining accuracy, and visualize the
surviving learned connectivity.

This is an exploration of recurrent graphs and parameter mutation, not a
sparse-runtime feature.

## Current codebase fit

- Recurrent cells can be unrolled with shared parameters using `%op`'s `()`
  lifting boundary; `Operation.stack` can reassemble per-step outputs.
- Host observation and mutation are context-mediated:
  `Context.get_values`/`set_values`.
- There is no generic sparse graph rewrite. Setting a dense parameter to zero
  does **not** remove an edge from the tensor graph, so `Tensor.to_printbox`
  cannot honestly serve as a “pruned graph” visualization by itself.

## Direction

- Start with the delay task (`y[t] = x[t-2]`) and a minimal custom or tanh RNN
  cell. Reuse an LSTM only if #60 has landed and does not obscure the small
  learned circuit.
- Train with a sparsity-promoting penalty, sweep pruning thresholds on held-out
  data, and select the sparsest model that retains the agreed accuracy.
- Render a derived connectivity view from the pruned weight matrices (text,
  DOT, or SVG). The artifact should omit zero-weight edges and label the
  remaining weights; it need not pretend the underlying dense tensor graph was
  rewritten.
- Keep the long convergence run under `@slow`; use a small deterministic test
  for pruning, sparsity accounting, and visualization.

## Completion criteria

- The unpruned model solves the selected deterministic task on held-out data.
- One-shot magnitude pruning removes a substantial majority of weights while
  retaining near-baseline accuracy; report the achieved numbers rather than
  baking in the blog's best result before measuring OCANNL's model.
- A checked artifact depicts the nonzero recurrent connectivity.
- A regular test covers the mechanics, and an `@slow` target reproduces the
  training result from a fixed seed.

Quantization and execution-time exploitation of sparsity are separate work.

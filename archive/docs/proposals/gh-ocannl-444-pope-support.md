# PoPE (Polar Position Embeddings)

Issue: [#444](https://github.com/ahrefs/ocannl/issues/444)

## Goal

Add PoPE as a position-embedding strategy for self-attention. PoPE maps each
projected scalar to a nonnegative magnitude and a position-dependent phase,
so a content width `p` becomes a score width `2p`.

## Current state

`Nn_blocks.position_embedding` supports learned, sinusoidal, RoPE, and no
position embedding. `rope_frequencies`, `position_indices`, and
`Operation.interleave` provide most of the shape machinery. PoPE remains
explicitly deferred in `nn_blocks.ml`; no softplus operation exists.

The old plan assumed PoPE could reuse the `d_k` projection axis and then
manually reset it. That conflates the content and score widths. The
implementation must give them distinct inferred axes: Q/K project to
`d_k / 2`, PoPE interleaves to `d_k`, and the attention score contracts over
the expanded axis. V remains width `d_v`.

## Direction

- Implement a `pope` tensor combinator beside `rope`, using
  `interleave (magnitude * cos phase) (magnitude * sin phase)`.
- Start with a composed, numerically stable softplus if it behaves correctly
  on all backends. Add a native primitive only if stress tests or generated
  code justify the extra cross-backend surface.
- Generalize self-attention's Q/K projection-width plumbing without changing
  RoPE or non-positional callers.
- Keep PoPE out of cross-attention unless a concrete model calls for it.

## Completion criteria

- Shape tests prove content width `d_k/2` expands to score width `d_k`, with a
  clear error for odd `d_k`.
- Small numerical tests compare PoPE and its gradients with an independent
  reference, including large positive/negative softplus inputs.
- Attention forward/backprop tests cover PoPE while existing RoPE and
  transformer tests remain unchanged.
- Documentation states which width `rope_frequencies` expects for PoPE.

Do not add separate projection parameters, backend primitives, or transformer
variants unless the smallest implementation demonstrates they are necessary.

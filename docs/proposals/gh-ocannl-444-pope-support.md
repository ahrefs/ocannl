# Proposal: PoPE (Polar Position Embeddings) support

## Goal

Add PoPE (arXiv:2509.10534) as a position embedding variant in OCANNL's transformer blocks. PoPE applies `softplus` to Q/K elements independently to produce magnitudes, which are then paired with position-only phases (from RoPE-style frequency/position angles). This maps `d` scalars to `2d` real values, doubling the per-head dimensionality, unlike RoPE which rotates pairs in-place.

## Acceptance Criteria

- [ ] Add `softplus` unary op to `tensor/operation.ml` with forward `log(1 + exp(x))` and gradient `sigmoid(x) = 1/(1 + exp(-x))`
- [ ] Add `PoPE of { freqs : Tensor.t; positions : Tensor.t }` variant to `position_embedding` type in `nn_blocks.ml`
- [ ] Support separate projection width in `multi_head_attention`: when PoPE is used, Q/K are projected to `d_k/2` (the "content" dimension), then PoPE doubles it back to `d_k` via magnitude-phase pairing
- [ ] PoPE transform: given projected Q/K of width `d_k/2`, compute `mu = softplus(x)` for magnitudes, then output `interleave(mu * cos(angle), mu * sin(angle))` where angles come from `positions * freqs` (reusing existing `rope_frequencies` and `position_indices`)
- [ ] The score computation uses the full `d_k`-width Q/K (after PoPE expansion), so the existing einsum and `sqrt(dim d)` scaling work unchanged
- [ ] Gradients flow correctly through softplus and the magnitude-phase computation
- [ ] No regression in existing tests (`attention_test.ml`, `rope_test.ml`, `transformer_test.ml`)
- [ ] New test verifying PoPE attention forward pass shape, gradient flow, and numerical sanity (softplus outputs are non-negative)
- [ ] `assert (d_k mod 2 = 0)` when PoPE is used (same as RoPE)

## Context

### Current state (verified)

- **RoPE is fully implemented**: `position_embedding` type already has `RoPE of { freqs; positions }` variant. The `rope` function, `rope_frequencies`, `position_indices`, `interleave`/`deinterleave_even`/`deinterleave_odd` are all in place and tested.
- **`multi_head_attention`** (`nn_blocks.ml:181-209`): Takes `~d_k ~d_v` params, projects Q/K/V via inline params `{ w_q }`, `{ w_k }`, `{ w_v }`. The dim `d` is set to `d_k` at line 200. RoPE is applied between projection and score computation (lines 190-193).
- **Softplus is NOT available** as a primitive unary op. Available unary ops: `relu`, `sat01`, `exp`, `log`, `exp2`, `log2`, `sin`, `cos`, `sqrt`, `tanh`, `recip`, `recip_sqrt`. The `%cd` DSL for assignment-level code does not have `softplus` or `sigmoid` as primitives.
- **Interleave/deinterleave ops** are available in `tensor/operation.ml` and exposed in the DSL. They handle even/odd pair splitting/merging on the last output axis.
- **Existing PoPE placeholder**: A comment at `nn_blocks.ml:129-131` explicitly defers PoPE to #444.
- **#398 proposal** (`docs/proposals/gh-ocannl-398.md`): Confirms PoPE was deferred because it doubles dimensionality and requires decoupling projection width from head width.

### Key difference from RoPE

RoPE rotates existing (x_even, x_odd) pairs in-place -- the output has the same dimensionality as the input. PoPE instead:
1. Takes a `d/2`-dimensional content vector (from a narrower Q/K projection)
2. Computes magnitude `mu_i = softplus(x_i)` for each of the `d/2` components
3. Computes phase `theta_i = positions * freqs_i` (same as RoPE frequencies)
4. Outputs `d`-dimensional vector: `interleave(mu * cos(theta), mu * sin(theta))`

This means the Q/K projection weights `w_q`, `w_k` must project to `d_k/2` output dimensions when PoPE is active, while the score einsum still operates on the full `d_k` width (after PoPE expansion).

## Approach

### 1. Add `softplus` unary op (`tensor/operation.ml`)

Follow the pattern of existing unary ops (e.g., `relu` at line 166). Since `softplus(x) = log(1 + exp(x))` and its gradient is `sigmoid(x) = 1/(1+exp(-x))`, and these don't exist as `%cd` primitives, implement using the existing `exp` and `log` primitives at the tensor op level:

```
let softplus x = log(1 + exp(x))
```

For gradient, express sigmoid as a composition: `exp(x) / (1 + exp(x))`, or equivalently reuse the forward value: `1 - exp(-softplus(x))`. This avoids needing a new `%cd` primitive -- softplus can be defined as a composite operation using existing `log` and `exp` ops. However, for numerical stability and efficiency, it is better to add `softplus` and `sigmoid` as native `%cd` assignments primitives in `arrayjit/lib/assignments.ml`, mirroring how `relu` and `relu_gate` are defined.

Decision point: composite vs native. Native is preferred for numerical stability (avoids overflow for large x). The implementer should add `Softplus` and `Sigmoid` to the low-level assignment ops.

### 2. Add `PoPE` variant to `position_embedding` type

```ocaml
type position_embedding =
  | Learned_additive
  | Sinusoidal_additive of { enc_encoding : Tensor.t; dec_encoding : Tensor.t }
  | RoPE of { freqs : Tensor.t; positions : Tensor.t }
  | PoPE of { freqs : Tensor.t; positions : Tensor.t }
  | No_pos_embed
```

### 3. Implement `pope` transform function

In `nn_blocks.ml`, alongside the existing `rope` function:

```ocaml
let%op pope ~freqs ~positions x =
  let cos_a = cos (positions *. freqs) in
  let sin_a = sin (positions *. freqs) in
  let mu = softplus x in
  interleave (mu *. cos_a) (mu *. sin_a)
```

Note: `x` here has output dim `d_k/2`, `freqs` has output dim `d_k/2`, result has output dim `d_k`.

### 4. Modify `multi_head_attention` for PoPE

The key change: when `pos_embed = PoPE _`, the projection dimension for Q/K is `d_k/2` instead of `d_k`. This requires setting the `d` shape dimension differently before vs after PoPE expansion:

- Project Q/K with output width `d_k/2` (set dim constraint before projection)
- Apply PoPE to expand to `d_k`
- Score computation uses full `d_k` as before

Implementation: add a `d_proj` local variable that is `d_k/2` for PoPE and `d_k` otherwise, and use `Shape.set_dim` appropriately on a separate dimension variable for the projection.

### 5. Update `transformer` and decoder blocks

Add `PoPE` case alongside `RoPE` in pattern matches throughout `transformer_encoder_block`, `transformer_decoder_block`, `transformer_encoder`, `transformer_decoder`, and `transformer`. These are straightforward -- PoPE behaves like RoPE in that it is applied inside attention (not as additive embedding).

### 6. Tests

Add `test/operations/pope_test.ml` (or extend `rope_test.ml`):
- Verify shape: input `d_k/2` -> output `d_k` after PoPE
- Verify softplus non-negativity of magnitudes
- Verify gradient flow through the full attention pipeline with PoPE
- Compare against hand-computed values for a small example

### Prerequisites

- **Softplus as `%cd` primitive**: Requires changes to `arrayjit/lib/assignments.ml` to add `Softplus` and `Sigmoid` (or `Sigmoid_gate`) to the low-level assignment language, then backend codegen support in CPU and CUDA/Metal backends. This is the main prerequisite work.
- **No external blockers**: RoPE (#398) is complete. The infrastructure (interleave, deinterleave, frequencies, position indices) is all in place.

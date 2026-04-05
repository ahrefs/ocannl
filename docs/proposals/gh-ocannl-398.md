# Proposal: RoPE and other non-learned position embeddings

## Goal

Add non-learned (computed) position embedding support to OCANNL's transformer blocks, specifically RoPE (Rotary Position Embeddings), PoPE (Polar Coordinate Position Embeddings), and sinusoidal positional encoding. This is a prerequisite for loading and running modern LLMs (Llama, Gemma) that rely on RoPE.

## Acceptance Criteria

- [ ] Implement RoPE as a function in `nn_blocks.ml` that applies rotary embeddings to query/key tensors
- [ ] RoPE frequency tensors are non-learned (`grad_spec:Prohibit_grad`)
- [ ] Implement PoPE variant (arXiv:2509.10534) -- clarify whether PoPE differs from RoPE only in attention score normalization (see note below)
- [ ] Implement sinusoidal positional encoding as a non-learned additive embedding
- [ ] Add a `position_embedding` strategy type and integrate it into `multi_head_attention` and the `transformer` function
- [ ] Assert even `d_k` at construction time (RoPE requires pairs)
- [ ] Base frequency is configurable (default 10000, some models use 500000)
- [ ] Gradients flow correctly through the rotation (sin/cos gradients already correct in `operation.ml`)
- [ ] No regression in existing tests (`attention_test.ml`, `transformer_test.ml`)
- [ ] New test verifying RoPE output shape, gradient flow, and ideally comparison against a reference

## Context

### Key code pointers (verified against current source)

- **`lib/nn_blocks.ml`** -- `multi_head_attention` at lines 115-134, `transformer` at lines 202-220. Position encoding is currently learned via inline params `{ pos_encoding }` at line 218. RoPE/PoPE would be applied to Q/K tensors inside `multi_head_attention` (after projection, before score computation), while sinusoidal replaces the additive `{ pos_encoding }` in `transformer`.
- **`tensor/operation.ml`** -- `sin` (line 216), `cos` (line 227) with correct gradients. `interleave` (lines 336-346) implements `2*i` / `2*i+1` indexing via einsum logic; its gradient performs the inverse (de-interleave), extracting even/odd elements.
- **`tensor/tensor.mli`** -- `grad_spec` type at line 126: `Require_grad | Prohibit_grad | If_needed`.
- **`arrayjit/lib/assignments.ml`** -- `Constant_fill` (line 27) for filling tensors with precomputed float arrays; `Embed_symbol` (line 38) for embedding static symbols.

### Even/odd pair access

RoPE requires splitting a vector into even/odd pairs. The `interleave` gradient logic already implements de-interleaving (`2*i => i` and `2*i+1 => i`). A dedicated `deinterleave` operation (or reshape to `[..., d/2, 2]`) is needed. The reshape approach may be cleaner if OCANNL's shape system supports it; otherwise, a new `deinterleave` unop using the existing einsum `2*i` pattern is straightforward.

### PoPE clarification needed

The task elaboration notes that expanding PoPE's formula `||x|| * RoPE(x/||x||)` yields the same rotation as plain RoPE. The paper's actual contribution may be in how attention scores are computed (decoupled content/position matching), not in the embedding rotation itself. The implementer should read section 3 of arXiv:2509.10534 carefully to determine whether PoPE requires:
1. Only a different attention score formula (modifying the dot-product computation), or
2. A genuinely different rotation applied to Q/K vectors.

This does not block starting -- the RoPE and sinusoidal work is independent, and PoPE can be finalized once the paper is read.

### Integration pattern

The `multi_head_attention` function currently takes Q/K/V projections and computes scores directly. The integration point is between projection and score computation (lines 117-121 in `nn_blocks.ml`):

```
let q = { w_q } * x in
let k = { w_k } * x in
(* --> apply RoPE/PoPE to q and k here <-- *)
let scores = (q +* k ...) /. sqrt (dim d) in
```

A `position_embedding` variant type controls which strategy is used. The `transformer` function needs a corresponding parameter to switch between learned additive encoding and the new strategies.

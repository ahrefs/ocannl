# Proposal: Transformer Inference for a Small Open-Weights Model

## Status update (2026-06-12)

- Issue #377 is OPEN, milestone **v0.7.1** (GH due date 2026-03-14 has lapsed; ROADMAP.md
  still lists "Transformer inference demo (#377)" under v0.7.1).
- **Superseded by `docs/proposals/gh-ocannl-377.md`** ("GPT-2 Small Inference Pipeline",
  2026-04-15), which covers the same scope against the same issue with fresher context.
  This file should be treated as the historical first elaboration.
- Several premises have since landed, invalidating "what's missing" claims below:
  - `decoder_only_block` and a `decoder_only` stack now exist in `lib/nn_blocks.ml`
    (lines 231 and 245), with `~mask` and `pos_embed` support; they arrived with the
    decoder-only Names-dataset example (`test/training/transformer_names.ml`, PR #57).
    Caveat: the block is **post-layer-norm**; GPT-2 needs a pre-norm variant, so that gap
    stands.
  - RoPE and sinusoidal position embeddings landed (#398 CLOSED, commit `9d943eb9`;
    PoPE deferred to #444) — the claim "no RoPE code exists in lib/" is no longer true.
  - A BPE tokenizer now exists: `Dataprep.Bpe` in `ocaml-dataprep/lib/bpe.ml`, compatible
    with HuggingFace tokenizer.json and explicitly supporting GPT-2 (also Gemma 3, Qwen 3,
    SmolLM2). See `docs/proposals/dataprep-tokenizer-integration.md`. The "minimal inline
    BPE" fallback is obsolete.
- Still missing: GeLU/SiLU/Sigmoid/Erf activations, weight conversion from safetensors,
  embedding lookup for a 50K vocab, and the inference script itself (`gpt2_inference.ml`
  does not exist).
- `Persistence.{save,load,restore}` (#373) is unchanged and remains the weight-loading seam.
- nn_blocks.ml line numbers drifted: `softmax` is now at 107, `multi_head_attention` 179,
  `layer_norm` 208, `transformer_encoder_block` 217, `transformer` 309.

## Goal

Build an end-to-end inference pipeline that loads pre-trained GPT-2 small (124M parameters) weights, tokenizes a text prompt, runs a forward pass through a decoder-only transformer, and generates coherent text autoregressively. This demonstrates OCANNL's ability to run real pre-trained models and serves as a flagship example for the v0.7.1 milestone.

GitHub issue: https://github.com/ahrefs/ocannl/issues/377
ROADMAP.md v0.7.1: "Transformer inference demo (#377)"

## Acceptance Criteria

- [ ] GPT-2 small (124M) weights are converted to OCANNL checkpoint format via a Python helper script, and loaded using `Persistence.load`/`Persistence.restore`
- [ ] A decoder-only transformer architecture matching GPT-2 small is implemented: 12 layers, 12 heads, d_model=768, d_ff=3072, pre-layer-norm, GeLU activation, learned positional embeddings
- [ ] GeLU activation is added to `nn_blocks.ml` (tanh approximation using existing ops)
- [ ] A `decoder_only_block` and `decoder_only_transformer` are added to `nn_blocks.ml` (pre-layer-norm, masked self-attention, GeLU FFN, no cross-attention)
- [ ] Tokenization uses the `Dataprep.Bpe_tokenizer` from `ocaml-dataprep` (or a minimal BPE implementation if the dataprep tokenizer is not yet available for GPT-2's vocabulary)
- [ ] The inference script produces coherent, recognizable English text from a given prompt (e.g., "The quick brown fox")
- [ ] Forward pass logits match HuggingFace GPT-2 reference output within float32 tolerance (~1e-4 relative error) on a short test sequence
- [ ] Autoregressive generation supports temperature and top-k sampling
- [ ] The example runs end-to-end on CPU in a reasonable time (< 30 seconds for generating 50 tokens)
- [ ] The script is located at `test/training/gpt2_inference.ml` with a corresponding `.expected` file for CI

## Context

### What exists now

**Tensor persistence** (`lib/persistence.ml`): `save`, `load`, and `restore` are fully implemented with a binary checkpoint format (s-expression header + raw tensor data). `load` creates fresh tnodes from file; `restore` overwrites existing tnode data. This is the foundation for weight loading.

**Transformer blocks** (`lib/nn_blocks.ml`):
- `multi_head_attention` (line 115): separate Q/K/V projections, optional mask, temperature scaling, dropout
- `layer_norm` (line 136): learnable gamma/beta normalization
- `transformer_encoder_block` (line 145): self-attention + FFN with post-layer-norm (close to what's needed but uses post-norm and ReLU)
- `transformer` (line 202): encoder-decoder architecture (not decoder-only)
- `softmax` (line 106): numerically stable with temperature

**No decoder-only block in nn_blocks.ml**: The `transformer_encoder_block` is the closest match but uses post-layer-norm and lacks a `~mask` parameter. The Names transformer proposal (watch-ocannl-README-md-369aadb4) specifies adding `decoder_only_block` to nn_blocks.ml, but it has not been implemented yet. *(Update 2026-06-12: implemented — `decoder_only_block` (post-norm, with `~mask`) and `decoder_only` are now in nn_blocks.ml; a pre-norm variant is still needed for GPT-2.)*

**No GeLU/SiLU activations**: Only ReLU, Exp, Log, Sin, Cos, Sqrt, Neg, Tanh_approx exist. GeLU can be composed: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.

**RoPE** (gh-ocannl-398): The issue is closed but no RoPE code exists in `lib/`. *(Update 2026-06-12: RoPE and sinusoidal embeddings now exist in nn_blocks.ml via the `position_embedding` type, commit `9d943eb9`.)* Not needed for GPT-2 (which uses learned positional embeddings), but would be needed for LLaMA/Gemma follow-up.

**Tokenizer** (dataprep): The `Dataprep` package is already an opam dependency. `Dataprep.Names` provides character-level tokenization. A BPE tokenizer (`Dataprep.Bpe_tokenizer`) is being developed in `ocaml-dataprep`. If not ready, a minimal GPT-2 BPE decoder can be implemented inline. *(Update 2026-06-12: it is ready — `Dataprep.Bpe` reads HuggingFace tokenizer.json and supports GPT-2.)*

### Architecture: GPT-2 Small

```
vocab_size: 50257
n_positions: 1024
n_embd: 768
n_layer: 12
n_head: 12
d_k = d_v = 64 (768 / 12)
d_ff: 3072 (4 * 768)
activation: GeLU
layer_norm: pre-norm (before attention and FFN)
positional: learned absolute embeddings
output: tied to token embedding (weight sharing)
total parameters: ~124M (~500MB float32)
```

### Key technical challenges

1. **Pre-layer-norm block**: The existing `transformer_encoder_block` uses post-norm. GPT-2 needs:
   ```
   x = x + mha(ln1(x))    -- pre-norm attention
   x = x + ffn(ln2(x))    -- pre-norm FFN
   ```

2. **Fused QKV projection**: GPT-2's `c_attn.weight` is a single (768 x 2304) matrix computing Q, K, V together. OCANNL's `multi_head_attention` uses separate `w_q`, `w_k`, `w_v`. The weight converter must split the fused weight into three (768 x 768) matrices.

3. **Weight mapping**: GPT-2 HuggingFace safetensors use names like `h.0.attn.c_attn.weight`. The Python converter maps these to OCANNL tensor labels and writes the checkpoint file. The OCANNL side then constructs the model graph (which assigns tnode IDs) and uses `Persistence.restore` to fill in the weights.

4. **Embedding lookup vs one-hot**: For vocab_size=50257, one-hot encoding is impractical (50K-dimensional vectors). Instead, use an integer-indexed embedding lookup. This may require a small utility or a creative use of einsum to select rows from the embedding matrix.

5. **Token embedding tying**: GPT-2 reuses the input embedding matrix as the output projection (logits = hidden @ embedding^T). The checkpoint converter and model construction must share this weight.

### Approach

**Phase 1 -- nn_blocks.ml additions**:
- Add `gelu` activation function
- Add `decoder_only_block` with pre-layer-norm, masked self-attention, configurable FFN activation
- Add `decoder_only_transformer` that stacks N blocks with embedding + final layer norm

**Phase 2 -- Weight conversion**:
- Python script (`scripts/convert_gpt2_weights.py`) that downloads GPT-2 small from HuggingFace, splits fused QKV, and writes OCANNL checkpoint format
- The script must produce a checkpoint file matching the tnode IDs/labels that the OCaml model graph will create

**Phase 3 -- Tokenization bridge**:
- If `Dataprep.Bpe_tokenizer` supports GPT-2 vocab: use it directly
- Otherwise: minimal GPT-2 BPE implementation reading the published `vocab.bpe` and `encoder.json` files

**Phase 4 -- Inference script** (`test/training/gpt2_inference.ml`):
- Construct GPT-2 model graph
- Load weights from checkpoint
- Encode prompt -> run forward pass -> sample next token -> loop
- Print generated text

**Phase 5 -- Validation**:
- Compare logits against HuggingFace reference on a fixed prompt
- Verify greedy decoding produces identical output

### Dependencies

- **gh-ocannl-373** (Tensor persistence): DONE -- `Persistence.{save,load,restore}` available
- **watch-ocannl-README-md-ebb316ea** (Tokenizers): In progress in `ocaml-dataprep`. Can work around with inline BPE if not ready.
- **watch-ocannl-README-md-369aadb4** (Names transformer): Not yet implemented, but this task will independently add `decoder_only_block` to nn_blocks.ml (the Names example can then reuse it) *(Update 2026-06-12: landed the other way around — the Names example (PR #57) brought `decoder_only_block` into nn_blocks.ml, which GPT-2 work can now reuse/adapt)*
- **gh-ocannl-398** (RoPE): Not needed for GPT-2. Relevant for future LLaMA/Gemma work.

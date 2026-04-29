# Proposal: GPT-2 Small Inference Pipeline

## Goal

Build an end-to-end inference pipeline for GPT-2 Small (124M parameters) that loads pre-trained weights into OCANNL, tokenizes input text, runs a decoder-only transformer forward pass, and generates coherent text autoregressively. This is the v0.7.1 milestone "Transformer inference demo (#377)" and serves as OCANNL's first real-world pre-trained model example.

GitHub issue: https://github.com/ahrefs/ocannl/issues/377

## Acceptance Criteria

- [ ] GeLU activation added to `lib/nn_blocks.ml` using the tanh approximation composed from existing ops: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
- [ ] `decoder_only_block` added to `lib/nn_blocks.ml`: pre-layer-norm, masked self-attention, GeLU FFN, configurable activation, no cross-attention
- [ ] `decoder_only_transformer` added to `lib/nn_blocks.ml`: stacks N blocks with final layer norm, accepts learned positional embeddings
- [ ] Python weight conversion script (`scripts/convert_gpt2_weights.py`) downloads GPT-2 Small from HuggingFace, splits fused QKV weights, transposes Conv1D-style matrices, and writes OCANNL checkpoint format (s-expression header + binary)
- [ ] Weights loaded via `Persistence.restore` into the constructed OCANNL model graph
- [ ] Tokenization uses `Dataprep.Bpe.from_pretrained "openai-community/gpt2"` for encode/decode
- [ ] Autoregressive generation loop: forward pass, sample from last-position logits, temperature scaling, greedy and top-k sampling modes
- [ ] Forward-pass logits on a fixed test prompt match HuggingFace GPT-2 reference output within float32 tolerance (~1e-4 relative error per logit)
- [ ] Generated text from "The quick brown fox" is coherent, recognizable English
- [ ] Example runs on CPU (C backend) and completes generation of 50 tokens in < 60 seconds
- [ ] Inference script at `test/training/gpt2_inference.ml` with `.expected` file for CI regression testing
- [ ] Token embedding weight tying: input embedding matrix reused as output projection (logits = hidden @ embedding^T)

## Context

### Available infrastructure

**Tensor persistence** (`lib/persistence.ml`): Fully implemented `save`, `load`, and `restore` with binary checkpoint format. `restore` matches by tnode ID and overwrites hosted buffers, verifying precision and dimension compatibility. This is the weight-loading mechanism.

**Transformer blocks** (`lib/nn_blocks.ml`):
- `multi_head_attention` (line 181): Separate Q/K/V projections via `w_q`, `w_k`, `w_v` params, scaled dot-product, optional causal mask, temperature, dropout, RoPE support
- `layer_norm` (line 211): Learnable gamma/beta, configurable epsilon
- `softmax` (line 107): Numerically stable with temperature, axis-specified via `~spec`
- `mlp` (line 89): Variable-depth MLP with linear output, uses ReLU
- `transformer_encoder_block` (line 220): Post-norm self-attention + FFN (close but uses post-norm and ReLU; no mask parameter)
- `dropout` (line 82): Controlled by train_step presence

**Decoder-only pattern** already demonstrated in `test/training/fsm_transformer.ml`: A decoder-only model with masked self-attention, learned positional embeddings, causal mask, separate training and inference compilation, and shared weight parameters between the two. This is the primary reference for how to structure the GPT-2 inference script.

**BPE tokenizer** (`Dataprep.Bpe` in `ocaml-dataprep`): Fully implemented, supports HuggingFace `tokenizer.json` format. API: `from_pretrained "openai-community/gpt2"` for download+load, `encode t string -> int array`, `decode t ids -> string`, `vocab_size t -> int`. Compatible with GPT-2's byte-level BPE (50257 vocab).

**One-hot encoding** (`nn_blocks.ml` line 61): `one_hot_of_int_list ~num_classes` converts integer indices to one-hot tensors. Usable for embedding lookup at inference time (single token at a time: 50257 floats = ~200KB, feasible).

**Activations available**: ReLU, Exp, Log, Sin, Cos, Sqrt, Neg, Tanh, Sat01, Recip, Recip_sqrt. No GeLU or SiLU.

### GPT-2 Small architecture

| Parameter | Value |
|-----------|-------|
| vocab_size | 50257 |
| n_positions | 1024 |
| n_embd (d_model) | 768 |
| n_layer | 12 |
| n_head | 12 |
| d_k = d_v | 64 (768/12) |
| d_ff | 3072 (4 * 768) |
| activation | GeLU (tanh approx) |
| normalization | Pre-layer-norm |
| positional | Learned absolute embeddings |
| output projection | Tied to token embedding |
| total params | ~124M (~500MB fp32) |

### Key technical challenges

1. **Embedding lookup**: With vocab_size=50257, full-sequence one-hot encoding is memory-prohibitive. For autoregressive inference (one token at a time), single-token one-hot is feasible (~200KB). For the initial prompt, process tokens sequentially or in a small batch. This avoids the need for #343 (dynamic indexing).

2. **Pre-layer-norm block**: The existing `transformer_encoder_block` uses post-norm. GPT-2 needs: `x = x + mha(ln1(x))` then `x = x + gelu_ffn(ln2(x))`. A new `decoder_only_block` must be added.

3. **Fused QKV weight splitting**: HuggingFace GPT-2 stores `c_attn.weight` as a single (768, 2304) Conv1D-style matrix. The Python converter must: (a) transpose from (in, out) to (out, in), and (b) split into three (768, 768) matrices for separate `w_q`, `w_k`, `w_v`.

4. **Weight ID mapping**: OCANNL identifies tensors by integer ID, not name. The workflow is: (1) construct the OCANNL model graph (assigns tnode IDs deterministically), (2) export a mapping of (label -> ID) from the OCaml side, (3) the Python converter writes the checkpoint using those IDs.

5. **Token embedding tying**: GPT-2 ties input embeddings and output projection. In OCANNL, the model construction must reference the same weight param for both `w_embed * one_hot_input` and `w_embed * hidden` (logits).

6. **No KV cache**: Initial implementation recomputes the full attention over the entire sequence at each generation step. This is O(n^2) per step but correct. KV caching is a follow-up optimization.

### Dependencies

- **Tensor persistence (#373)**: DONE -- `Persistence.{save,load,restore}` fully implemented
- **BPE tokenizer**: DONE -- `Dataprep.Bpe` supports GPT-2 via `from_pretrained`
- **RoPE (#398)**: Not needed for GPT-2 (uses learned positional embeddings)
- **#343 (dynamic indexing)**: Not needed -- single-token one-hot is sufficient for inference

## Approach

### Phase 1: nn_blocks.ml additions (GeLU + decoder-only blocks)

**Add GeLU activation** as a composable function:
```ocaml
let gelu x =
  let%op result =
    x *. !.0.5 *. (!.1.0 + tanh (sqrt !.(2.0 /. Float.pi) *. (x + !.0.044715 *. (x *. x *. x))))
  in result
```
This uses the standard tanh approximation and composes entirely from existing operations (pointmul, add, tanh, sqrt, pointpow).

**Add `decoder_only_block`**: Pre-layer-norm masked self-attention + FFN with configurable activation:
```ocaml
let decoder_only_block ~label ~num_heads ~d_k ~d_v ~d_ff
    ?(epsilon = 1e-5) ?(activation = relu) ?(pos_embed = No_pos_embed) () =
  let mha = multi_head_attention ~label:("mha" :: label)
    ~num_heads ~d_k ~d_v ~pos_embed () in
  let ln1 = layer_norm ~label:("ln1" :: label) ~epsilon () in
  let ln2 = layer_norm ~label:("ln2" :: label) ~epsilon () in
  fun ~train_step ~mask x ->
    let x = x + mha ~train_step ~mask (ln1 x) in
    x + (fun h -> { w2 } * activation ({ w1; o = [d_ff] } * h + { b1 = 0. })) (ln2 x) + { b2 = 0. }
```
This mirrors the `fsm_transformer.ml` pattern but with pre-norm and configurable activation.

**Add `decoder_only_transformer`**: Stacks N blocks with a final layer norm:
```ocaml
let decoder_only_transformer ~label ~num_layers ~num_heads ~d_k ~d_v ~d_ff
    ?(epsilon = 1e-5) ?(activation = relu) ?(pos_embed = No_pos_embed) () =
  let layers = List.init num_layers ~f:(fun i ->
    decoder_only_block ~label:(("layer" ^ Int.to_string i) :: label)
      ~num_heads ~d_k ~d_v ~d_ff ~epsilon ~activation ~pos_embed ()) in
  let ln_f = layer_norm ~label:("ln_f" :: label) ~epsilon () in
  fun ~train_step ~mask x ->
    let h = List.fold layers ~init:x ~f:(fun x layer -> layer ~train_step ~mask x) in
    ln_f h
```

### Phase 2: Weight conversion script

Python script `scripts/convert_gpt2_weights.py`:
1. Download GPT-2 Small from HuggingFace using `transformers` library
2. Extract state dict, iterate over parameter names
3. For each `h.{i}.attn.c_attn.weight`: transpose (768,2304) -> (2304,768), split into Q/K/V (768,768) each
4. For each `h.{i}.attn.c_attn.bias`: split into Q/K/V (768,) each
5. For each `h.{i}.mlp.c_fc.weight` and `c_proj.weight`: transpose Conv1D style (in,out) -> (out,in)
6. Write OCANNL checkpoint format:
   - Build s-expression header with tensor metadata (ID, label, prec=Single, dims, offset, byte_length)
   - Write binary float32 data contiguously
7. The script accepts a `--id-mapping` JSON file that maps OCANNL tensor labels to integer IDs

The ID mapping is generated by a small OCaml helper that constructs the GPT-2 model graph and prints the label-to-ID mapping for all parameters.

### Phase 3: Inference script (`test/training/gpt2_inference.ml`)

Structure follows the `fsm_transformer.ml` pattern:

1. **Construct model graph**: Build GPT-2 architecture using the new `decoder_only_transformer` block, with `gelu` activation, d_model=768, 12 layers, 12 heads
2. **Load weights**: Use `Persistence.restore` to fill in pre-trained weights from the converted checkpoint file
3. **Tokenize input**: Use `Dataprep.Bpe.from_pretrained "openai-community/gpt2"` to encode the prompt
4. **Embedding**: Convert token IDs to one-hot vectors, multiply by embedding matrix
5. **Add positional encoding**: Add learned positional embeddings (loaded from checkpoint)
6. **Forward pass**: Run through all 12 transformer blocks with causal mask
7. **Output projection**: Multiply final hidden state by embedding matrix transpose (weight tying)
8. **Sampling**: Extract logits for last position, apply temperature, optionally top-k filter, sample from softmax
9. **Generation loop**: Append sampled token, re-run forward pass on growing sequence

The `.expected` file will test greedy decoding with a fixed prompt to ensure deterministic output for CI.

### Phase 4: Validation

1. **Logit comparison**: Python script that runs the same prompt through HuggingFace GPT-2 and dumps logits. The OCaml test compares against these reference logits.
2. **Greedy generation match**: Verify that greedy decoding (temperature=0) produces the same token sequence as the HuggingFace reference for a short prompt.
3. **Qualitative check**: Generated text from open prompts is coherent English.

### File layout

```
lib/nn_blocks.ml              -- add gelu, decoder_only_block, decoder_only_transformer
scripts/convert_gpt2_weights.py  -- weight conversion from HuggingFace to OCANNL checkpoint
scripts/dump_gpt2_reference.py   -- dump reference logits for validation
test/training/gpt2_inference.ml  -- main inference example
test/training/gpt2_inference.expected  -- CI regression output
```

### Estimated effort

Large (7-10 days):
- Days 1-2: GeLU activation, `decoder_only_block`, `decoder_only_transformer` in nn_blocks.ml. Unit tests.
- Day 3: Python weight conversion script. ID mapping helper.
- Day 4: Weight loading integration -- construct model, restore weights, verify shapes match.
- Day 5: Tokenizer integration -- encode prompt, decode output, handle special tokens.
- Days 6-7: Autoregressive generation loop -- forward pass, sampling, text output.
- Days 8-9: Validation against HuggingFace reference. Debug numerical differences.
- Day 10: CI test, documentation, cleanup.

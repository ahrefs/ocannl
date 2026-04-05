# Proposal: Transformer for the Names Dataset

## Goal

Add a complete decoder-only autoregressive transformer example that trains on the Names dataset and generates plausible names. This serves as both a test of OCANNL's transformer building blocks and a tutorial demonstrating end-to-end model construction, training, and inference.

GitHub issue: https://github.com/ahrefs/ocannl/issues/57
Related proposal: `gh-ocannl-57.md` (detailed implementation notes)

## Acceptance Criteria

- [ ] A `decoder_only_block` and `decoder_only` function are added to `lib/nn_blocks.ml` — masked self-attention + FFN, no cross-attention, following the `transformer_encoder_block` pattern but accepting a `~mask` parameter
- [ ] A training script `test/training/transformer_names.ml` implements a character-level decoder-only transformer on the Names dataset
- [ ] The `Dataprep.Names` module is extended (or the script includes local helpers) to provide full-sequence access (name -> padded index sequence), not just bigrams
- [ ] Causal masking is correctly applied: lower-triangular mask so position s attends only to positions 0..s
- [ ] Teacher forcing training: input = sequence[0..n-2], target = sequence[1..n-1], cross-entropy loss
- [ ] Loss decreases during training to a value significantly below random baseline (~3.33 nats for 28-char vocabulary)
- [ ] Autoregressive generation works: start from `.` token, sample next character from softmax distribution, feed back as input, stop at `.` or max length
- [ ] Generated names are recognizably name-like (better than random, better than bigram-only baseline)
- [ ] Expected output file `test/training/transformer_names.expected` with epoch loss targets for CI regression testing
- [ ] If RoPE (gh-ocannl-398) is merged, optionally use it instead of learned positional embeddings; otherwise use learned positional embeddings

## Context

### What exists

**Bigram MLP** (`test/training/bigram_mlp.ml`): Character-level bigram model using `Dataprep.Names` for data loading, one-hot encoding, SGD training with LR schedule, and CDF-based multinomial sampling for generation. This is the primary pattern to follow.

**Encoder-decoder transformer** (`lib/nn_blocks.ml` lines 202-240): Full encoder-decoder with learned positional embeddings, `transformer_with_loss` wrapper for teacher forcing. Not suitable for this task — we need decoder-only (no encoder, no cross-attention).

**Decoder-only prototype** (`test/operations/layer_norm_test.ml` lines 4-13): A `mini_decoder_block` that is exactly a decoder-only block (masked self-attention + FFN, no cross-attention). Currently used only in a forward-pass shape test, not for training.

**`transformer_encoder_block`** (`lib/nn_blocks.ml` line 145): Self-attention + FFN with pre-norm. Very close to what's needed — just lacks `~mask` parameter. The decoder-only block is essentially this block with causal masking passed through.

**Causal mask** construction pattern from `test/operations/layer_norm_test.ml` and `transformer_test.ml`: `NTDSL.init` with `if s >= t then 1. else 0.`.

**Names dataset** (`Dataprep.Names`): ~32K names, 28-char vocabulary (`.` start/end, ` ` space, a-z), `dict_size = 28`, `char_index`, `char_to_one_hot`, `letters_with_dot`. Currently only provides bigrams via `get_all_bigrams()`.

**RoPE status**: Implemented on branch `ludics/gh-ocannl-398-s4/root` but not yet merged to master. The example should use learned positional embeddings by default, with RoPE as an optional enhancement once merged.

### Architecture decisions

- **Decoder-only** (GPT-style): no encoder, causal self-attention only
- **Character-level**: no tokenizer needed, uses existing 28-char vocabulary
- **Fixed context length** (`ctx_len = 16`): names padded to fixed length with `.` tokens
- **One-hot encoding**: OCANNL doesn't support integer embedding lookup; use one-hot * weight matrix
- **SGD optimizer**: only SGD is available in OCANNL (no Adam)
- **Toy scale**: `d_model=32, num_heads=4, d_ff=64, num_layers=2` (~10-20K parameters)

### Dimension layout

OCANNL's 3-row shape system (batch, input, output):
- Token tensors: `batch=[batch_size; seq_len]`, input=`[]`, output=`[vocab_size]` or `[d_model]`
- Causal mask: `batch=[seq_len]`, input=`[seq_len]`, output=`[]`
- Positional embedding: `batch=[seq_len]`, output=`[d_model]`

### Key risks

- SGD may converge slowly on transformers (mitigated by toy scale + LR schedule)
- One-hot sequences use ~54MB for full dataset (mitigated by reasonable batch sizes)
- `Dataprep.Names` may need changes in the external `ocaml-dataprep` package, or sequence conversion can be done locally in the training script using `char_index`

## Files Changed

| File | Change |
|------|--------|
| `lib/nn_blocks.ml` | Add `decoder_only_block` and `decoder_only` (extract from `layer_norm_test.ml` prototype, add dropout) |
| `test/training/transformer_names.ml` | New: complete training + generation script |
| `test/training/transformer_names.expected` | New: expected output for CI |
| `test/training/dune` | Add `transformer_names` test stanza |

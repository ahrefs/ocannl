# Proposal: Reproduce Small-Size Transformer Digit Addition Results

## Goal

Demonstrate that OCANNL's transformer building blocks can train a tiny transformer (~500-1000 parameters) to learn multi-digit addition from scratch, reproducing the "grokking" phenomenon where accuracy suddenly jumps after extended training. This validates OCANNL as a viable framework for algorithmic learning tasks and serves as a precursor to the decoder-only autoregressive transformer example (gh-ocannl-57).

## Acceptance Criteria

1. **Data generator**: A function that generates random N-digit addition problems as token sequences of the form `A+B=R` where R is the reversed-digit sum (LSB first). Vocab: digits 0-9, `+`, `=`, PAD (13 tokens). Configurable digit count (3-10). On-the-fly generation (no static dataset).

2. **Model**: A decoder-only transformer using existing `multi_head_attention`, `mlp`, and causal mask infrastructure from `nn_blocks.ml`, with ~500-1000 trainable parameters. Hyperparameters: 1 layer, 1 attention head, d_model=16, d_ff=32 (tunable).

3. **Training**: SGD training loop following the `fsm_transformer.ml` pattern (manual epoch/batch loop, `grad_update` + `sgd_update`, linear LR schedule). On-the-fly batch generation. Training for sufficient steps (10k-50k+) to observe grokking.

4. **Evaluation**: Periodic evaluation on freshly generated test problems. Report exact-match accuracy (all result digits correct). The test should pass a threshold demonstrating the model learned to add (e.g., >80% exact-match accuracy on 3-digit addition).

5. **Integration**: New file `test/training/digit_addition.ml` with corresponding dune test entry. Runs as a standard `dune test` target.

## Context

### Existing Building Blocks

- **`multi_head_attention`** (`lib/nn_blocks.ml:181-209`): Q/K/V projections, scaled dot-product attention, multi-head via einsum, optional causal mask, dropout. Supports RoPE and sinusoidal position embeddings.

- **`mlp`** (`lib/nn_blocks.ml:89-96`): Variable-depth MLP with ReLU activations and linear output. Used as the FFN in transformer blocks.

- **`softmax`** (`lib/nn_blocks.ml:107-113`): Numerically stable (max subtraction), axis-specified via `~spec`.

- **`layer_norm`** (`lib/nn_blocks.ml:211-218`): Learnable gamma/beta. Note: `fsm_transformer.ml` omits layer norm for tiny models because recentered weights + normalization can kill gradient signal. We should experiment with and without.

- **Causal mask**: Lower-triangular matrix pattern used in `fsm_transformer.ml:103-108` and `transformer_test.ml:57-69`.

- **Initialization**: `kaiming` and `xavier` initializers available in `nn_blocks.ml:31-53`. Alternatively, the manual recentering approach from `fsm_transformer.ml:168-174` (shift uniform[0,1) to centered range).

### Reference Implementation Pattern: `fsm_transformer.ml`

This is the closest existing example. It trains a single-block decoder-only transformer on FSM next-state prediction. Key patterns to follow:

- **Data**: One-hot encoded sequences, flat arrays loaded via `Tn.set_values` (lines 51-59).
- **Model**: Manual assembly of `multi_head_attention` + `mlp` with residual connections, no layer norm (lines 116-124).
- **Compilation**: Separate training and inference compilation paths sharing weights (lines 128-194).
- **Training loop**: Manual epoch/batch iteration, `grad_update` + `sgd_update`, linear LR warmdown (lines 200-226).
- **Evaluation**: Separate inference routine compiled from same model, argmax prediction (lines 236-258).
- **Weight init**: Post-init recentering to avoid exp overflow in attention (lines 168-174).

### Reference Implementation Pattern: `bigram_mlp.ml`

Training loop with batched data, LR scheduling, SGD, and inference via `%cd` for weight sharing.

### Papers

1. Havinga "gpt-acc-jax": ~777 parameter transformer learns 10-digit addition
2. Zhu "A 456-Parameter Transformer Solves 10-Digit Addition": even smaller model

Key insight from papers: reversed output digits (LSB first) are critical — the model can propagate carry information left-to-right through the autoregressive sequence.

## Approach

### 1. Data Generation

```
Sequence format: d₁d₂...dₙ+d₁d₂...dₙ=rₘrₘ₋₁...r₁[PAD...]
```

- Input: two N-digit numbers A, B (zero-padded to N digits)
- Output: sum digits reversed (LSB first), zero-padded to N+1 positions (to handle carry overflow)
- Total sequence length: N + 1 + N + 1 + (N+1) = 3N + 3
- Vocab: 0-9 (digits), 10 (+), 11 (=), 12 (PAD) — 13 tokens
- One-hot encode each token position
- Generate batches on-the-fly using `Random.State`

For a decoder-only (GPT-style) setup: the model sees the full concatenated sequence and is trained to predict the next token at every position. Loss is computed only on the result portion (after `=`), since the input portion is deterministic given the problem.

### 2. Model Architecture

Follow `fsm_transformer.ml` pattern — manual single-block decoder-only transformer:

```
input → embedding(13 → d_model) + learned_pos_encoding → 
  residual(multi_head_attention(causal_mask)) →
  residual(mlp(d_ff)) →
  output_projection(d_model → 13)
```

Starting hyperparameters (targeting ~500-800 parameters):
- `d_model = 16`
- `num_heads = 1` (or 2 with `d_k = 8`)
- `d_ff = 32`
- `num_layers = 1`
- Sequence length: 3*N + 3 (e.g., 12 for N=3)

Parameter count estimate for N=3 (seq_len=12):
- Embedding: 13 * 16 = 208
- Pos encoding: 12 * 16 = 192
- W_q, W_k, W_v: 3 * 16 * 16 = 768 (may be too many — reduce d_k)
- W_o: 16 * 16 = 256
- FFN W1: 16 * 32 = 512, b1: 32
- FFN W_out: 32 * 16 = 512
- Output projection: 16 * 13 = 208

This exceeds the 777-parameter target. To reduce: use d_k = d_v = 8 (or lower), reduce d_ff, or use weight tying between embedding and output projection. The exact parameter budget will require experimentation.

### 3. Training Loop

- SGD with linear LR warmdown (following `fsm_transformer.ml` and `bigram_mlp.ml`)
- On-the-fly batch generation: each batch is freshly generated random addition problems
- Batch size: 32-64
- Training steps: 10k-50k (grokking can take many apparently unproductive steps)
- Every 500-1000 steps: evaluate on 1000 fresh test problems
- Track and print: epoch loss, exact-match accuracy (all result digits correct)

### 4. Evaluation

- Generate 1000 fresh random addition problems
- Run inference (forward-only, sharing trained weights)
- For each problem: argmax decode the result tokens, compare to ground truth
- Exact-match accuracy: all result digits must be correct
- Print sample predictions for qualitative inspection
- Test passes if accuracy exceeds threshold (e.g., >80% for 3-digit addition)

### 5. File Structure

- `test/training/digit_addition.ml` — main training script
- Add dune test entry in `test/training/dune` (no external data dependencies, only `ocannl` library needed)

### 6. Development Sequence

1. Implement data generator and verify tokenization correctness
2. Wire up model following `fsm_transformer.ml` pattern
3. Implement training loop with loss tracking
4. Add evaluation with exact-match accuracy
5. Tune hyperparameters for grokking behavior
6. Set regression thresholds for CI (conservative, not requiring full grokking)

### Known Risks

- **CUDA single-threaded kernels**: All CUDA kernels currently run with `grid_dim=1, block_dim=1`. This will make GPU training very slow for this task. CPU backend is likely faster for this model size.
- **Parameter budget**: Hitting exactly 500-800 parameters while maintaining trainability may require multiple rounds of hyperparameter tuning.
- **Grokking timing**: The phase transition can occur very late in training. CI test thresholds should be set conservatively (partial learning, not full grokking) to avoid flaky tests.
- **Weight initialization**: As seen in `fsm_transformer.ml`, OCANNL's default uniform[0,1) init causes attention overflow. Must use recentering or the `kaiming`/`xavier` initializers from `nn_blocks.ml`.

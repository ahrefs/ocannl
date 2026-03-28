# Decoder-Only Autoregressive Transformer on Names Dataset

## Motivation

OCANNL has full transformer building blocks (`multi_head_attention`, `layer_norm`, `mlp`,
`softmax`) and character-level bigram examples on the Names dataset, but no transformer
training example. The ROADMAP places a "Transformer toy example (#57)" under v0.6.4:
a fully working decoder-only autoregressive transformer as a Names dataset language model.

GitHub issue: https://github.com/ahrefs/ocannl/issues/57

## Current State

**Transformer blocks** in `lib/nn_blocks.ml`:
- `multi_head_attention` (line 115): Q/K/V projections, scaled dot-product with optional `~mask`, multi-head via einsum `"... s | h d; ... t | h d => ... s | t -> h"`, dropout support
- `layer_norm` (line 136): learnable gamma/beta, configurable epsilon
- `mlp` (line 89): variable-depth MLP with linear output layer
- `softmax` (line 107): numerically stable (max subtraction), temperature parameter, axis-specified via `~spec`
- `transformer_encoder_block` (line 145): self-attention + FFN with pre-norm — close to what we need but lacks `~mask`
- `transformer_decoder_block` (line 170): masked self-attention + cross-attention + FFN — includes cross-attention we don't want

**Decoder-only prototype** in `test/operations/layer_norm_test.ml` (lines 4-13):
```ocaml
let%op mini_decoder_block ~label ~num_heads ~d_k ~d_v ~d_ff ?(epsilon = 1e-5) () =
  let masked_mha = multi_head_attention ~label ~num_heads ~d_k ~d_v () in
  let ffn = mlp ~label ~hid_dims:[ d_ff ] () in
  let ln1 = layer_norm ~label ~epsilon () in
  let ln2 = layer_norm ~label ~epsilon () in
  fun ~train_step target ~mask ->
    let x1 = ln1 (target + masked_mha ~train_step ~mask target) in
    ln2 (x1 + ffn x1)
```
This is exactly a decoder-only block (masked self-attention + FFN, no cross-attention). Currently used only in a forward-pass test, not for training.

**Causal mask pattern** from `test/operations/layer_norm_test.ml` (lines 44-49):
```ocaml
let mask = NTDSL.init ~l:"mask" ~prec:Ir.Ops.single
    ~b:[ tgt_seq_len ] ~i:[ tgt_seq_len ] ~o:[]
    ~f:(function [| s; t |] -> if s >= t then 1. else 0. | ...)
```
Lower-triangular: query at position s attends to key at position t only when s >= t.

**Names dataset** (`Dataprep.Names`):
- `read_names ()` — list of ~32K names
- `dict_size = 28` — vocabulary: `.` (start/end), ` ` (padding), a-z
- `char_index`, `char_to_one_hot` — conversion utilities
- `letters_with_dot` — ordered character list for decoding
- Only provides bigrams via `get_all_bigrams ()` — no sequence-level API

**Bigram training examples** (`test/training/bigram.ml`, `bigram_mlp.ml`):
- Established patterns: `IDX.get_static_symbol` for batching, `one_hot_of_int_list` for encoding, `TDSL.rebatch` for tensor creation, `@|` for batch indexing
- Cross-entropy: `neg (log (softmax_probs *. one_hot_target))`, reduced via `++ "...|... => 0"`
- SGD with learning rate schedule
- Autoregressive generation: multinomial sampling via accumulated CDF over `letters_with_dot`

## Proposed Change

### 1. Add `decoder_only_block` and `decoder_only` to `lib/nn_blocks.ml`

Extract the `mini_decoder_block` pattern from `layer_norm_test.ml` into a reusable building block, adding dropout support. Add a stacking function:

```ocaml
(** Decoder-only transformer block: masked self-attention + FFN with pre-norm LayerNorm.
    Like [transformer_encoder_block] but with causal masking via [~mask]. *)
let%op decoder_only_block ~label ~num_heads ~d_k ~d_v ~d_ff
    ?(epsilon = 1e-5) ?(dropout_rate = 0.0) () =
  let masked_mha =
    multi_head_attention ~label:("masked_mha" :: label)
      ~num_heads ~d_k ~d_v ~dropout_rate () in
  let ffn = mlp ~label:("ffn" :: label) ~hid_dims:[ d_ff ] () in
  let ln1 = layer_norm ~label:("ln1" :: label) ~epsilon () in
  let ln2 = layer_norm ~label:("ln2" :: label) ~epsilon () in
  fun ~train_step x ~mask ->
    let x1 = ln1 (x + masked_mha ~train_step ~mask x) in
    ln2 (x1 + ffn x1)

(** Stack of decoder-only blocks. *)
let decoder_only ~label ~num_layers ~num_heads ~d_k ~d_v ~d_ff
    ?epsilon ?dropout_rate () =
  let layers = List.init num_layers ~f:(fun i ->
      decoder_only_block
        ~label:(("layer" ^ Int.to_string i) :: label)
        ~num_heads ~d_k ~d_v ~d_ff ?epsilon ?dropout_rate ()) in
  fun ~train_step x ~mask ->
    List.fold layers ~init:x ~f:(fun x layer -> layer ~train_step x ~mask)
```

Place these between the existing `transformer_encoder_block` (line 154) and `cross_attention` (line 155), since they follow the encoder block pattern but add masking.

### 2. Create `test/training/transformer_names.ml`

A complete training script following the `bigram_mlp.ml` pattern:

**Data preparation** — convert names to fixed-length sequences with teacher forcing:
```ocaml
let ctx_len = 16  (* captures names up to 14 chars + start/end *)

(* "emma" -> ['.'; 'e'; 'm'; 'm'; 'a'; '.'; '.'; ...] padded to ctx_len *)
(* input:  positions 0..ctx_len-2  (teacher forcing input) *)
(* target: positions 1..ctx_len-1  (next-token prediction target) *)
let name_to_sequences name =
  let chars = '.' :: String.to_list name @ ['.'] in
  let padded = List.take (chars @ List.init ctx_len ~f:(fun _ -> '.')) ctx_len in
  let input_indices = List.take padded (ctx_len - 1) |> List.map ~f:char_index in
  let target_indices = List.tl_exn (List.take padded ctx_len) |> List.map ~f:char_index in
  (input_indices, target_indices)
```

Each name produces one training example of `ctx_len - 1` tokens. With ~32K names this gives ~32K training examples.

**Model architecture** (toy-scale for fast iteration):
```
d_model = 32, num_heads = 4, d_ff = 64, num_layers = 2, ctx_len = 16
vocab_size = 28 (Dataprep.Names.dict_size)
```
~10-20K parameters. Structure:
1. Token embedding: `{ tok_embed; o = [d_model] } * one_hot_input` — (28 -> 32)
2. Positional embedding: `embedded + { pos_embed }` — learned, shape (ctx_len-1) x d_model
3. Decoder stack: `decoder_only ~num_layers:2 ~num_heads:4 ...`
4. Output projection: `{ w_out } * decoder_output` — (32 -> 28)

**Loss**: cross-entropy with softmax, following bigram pattern:
```ocaml
let log_probs = log (softmax ~spec:"... | v" () logits) in
let loss = neg (log_probs *. one_hot_target) ++ "...|... => 0" in
let batch_loss = (loss ++ "...|... => 0") /. !..batch_size
```

**Training loop**: SGD with learning rate decay, batched over names using `IDX.get_static_symbol` and `@|` indexing. Follow `bigram_mlp.ml` for the optimizer setup pattern.

**Autoregressive generation**: extend bigram generation to full-context inference:
```ocaml
(* Build inference computation: set context, run forward, sample from last position *)
let generate_name ~infer_step ~ctx ~logits_tensor =
  (* Start with '.' at position 0, rest padded *)
  let context = Array.create ~len:(ctx_len - 1) (char_index '.') in
  let rec aux pos =
    if pos >= ctx_len - 1 then ... (* max length reached *)
    else begin
      (* Set one-hot context into input tensor *)
      set_context_tensor ctx context;
      Train.run ctx infer_step;
      (* Sample from position pos using CDF over logits at that position *)
      let next_char = sample_from_logits logits_tensor pos in
      if Char.equal next_char '.' && pos > 0 then
        extract_name context 1 (pos - 1)
      else begin
        context.(pos) <- char_index next_char;
        aux (pos + 1)
      end
    end
  in
  aux 1
```

The inference computation reuses the trained model parameters but creates a separate routine with batch_size=1 and no gradient computation. The CDF-based sampling follows the pattern from `bigram_mlp.ml` lines 106-120.

### 3. Update `test/training/dune`

Add a test stanza following the existing pattern:
```
(test
 (name transformer_names)
 (package neural_nets_lib)
 (modules transformer_names)
 (deps
  ocannl_config
  (env_var OCANNL_BACKEND))
 (libraries ocannl dataprep)
 (preprocess
  (pps ppx_here ppx_ocannl)))
```

### 4. Add expected output file

Create `test/training/transformer_names.expected` with epoch loss targets and a few generated names. The generated names should be recognizably name-like (better than random, better than bigram-only).

## Dimension Layout in OCANNL's Shape System

OCANNL uses a 3-row shape system: batch dims, input dims, output dims. For the transformer:

- **Batch**: `[batch_size; seq_len]` — sequence positions are batch dimensions (matching `layer_norm_test.ml` line 37: `~batch_dims:[ batch_size; tgt_seq_len ]`)
- **Input**: `[]` — no input dimensions after embedding
- **Output**: `[d_model]` for hidden states, `[vocab_size]` for logits

The causal mask has shape `~b:[seq_len] ~i:[seq_len] ~o:[]` — both its dimensions are batch dimensions, aligning with the sequence positions in Q and K.

Token embedding maps `[vocab_size]` output dims to `[d_model]` output dims via matrix multiply. Positional embedding is a learnable tensor with batch dim `[seq_len]` and output dim `[d_model]`, broadcast-added to the embedded tokens.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| SGD convergence on transformer (Adam typically preferred) | Slow/no convergence | Toy scale + LR schedule; the bigram_mlp shows SGD works for this data. If needed, reduce model size further |
| Memory with one-hot sequences: 32K x 15 x 28 = ~13M floats | ~54MB, may stress Metal #344 buffer limits | Start with C backend; reduce batch_size or dataset size if needed |
| Positional embedding shape mismatch with variable-length names | Shape errors | Fixed ctx_len with '.' padding — all sequences same length |
| Sequence dimension as batch dim — interactions with attention einsum | Incorrect attention patterns | Verified: `layer_norm_test.ml` uses exactly this layout and it works |
| One-hot encoding is memory-inefficient for sequences | Large tensors | Acceptable for toy example; future work could add embedding lookup |

## Out of Scope

- **Integer embedding lookup** — currently OCANNL tensors are float-only, so we use one-hot * weight matrix. A sparse/integer embedding would be a separate feature.
- **Adam optimizer** — only SGD is available. Adding Adam is tracked separately.
- **Tokenizer integration** — the issue explicitly defers this to v0.7 with huggingface-tokenizers bindings.
- **Open-weights model support** (GPT-2, Gemma) — explicitly deferred to v0.7.
- **RoPE** — tracked in gh-ocannl-398, this example uses learned positional embeddings.

## Files Changed

| File | Change |
|------|--------|
| `lib/nn_blocks.ml` | Add `decoder_only_block` and `decoder_only` functions |
| `test/training/transformer_names.ml` | New: complete training + generation example |
| `test/training/transformer_names.expected` | New: expected output for CI |
| `test/training/dune` | Add `transformer_names` test stanza |

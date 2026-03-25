# Dataprep Tokenizer Integration into OCANNL

## Motivation

The v0.7.1 milestone item (README line 73) tracks tokenizer availability: "Tokenizers are being developed in the spin-off project ocaml-dataprep." The tokenizer implementation itself is happening in `ocaml-dataprep` (gh-ocaml-dataprep-2). This task covers the OCANNL-side integration: ensuring the dataprep BPE tokenizer works smoothly in OCANNL's training and inference pipelines, and checking off the README milestone item.

Without this integration, the downstream transformer inference task (watch-ocannl-README-md-177e6c0e) cannot process text input.

## Current State

**Dependency already exists**: OCANNL's `neural_nets_lib.opam` already depends on `dataprep >= 0.1.0`, and `dune-project` lists it. The `Dataprep` library is used in tests:
- `test/training/bigram.ml` -- uses `Dataprep.Names` for character-level tokenization
- `test/training/bigram_mlp.ml` -- same
- `test/training/circles_conv.ml` -- uses `Dataprep.Circles`
- `test/training/moons_demo.ml` -- uses `Dataprep.Half_moons`

**Existing helper**: `lib/nn_blocks.ml` has `one_hot_of_int_list ~num_classes` which converts integer class indices to one-hot tensors. This is used by the bigram examples but is not suitable for subword tokenization (vocabulary sizes of 50K-256K make one-hot encoding impractical -- embedding lookups are needed instead).

**No token-to-tensor bridge exists**: There is no utility to convert a tokenized sequence (int array from the BPE tokenizer) into an OCANNL tensor suitable for embedding lookup in transformer models.

Key files:
- `lib/nn_blocks.ml` -- NN building blocks, including `one_hot_of_int_list`
- `neural_nets_lib.opam` -- already depends on dataprep
- `README.md` line 73 -- the milestone checkbox
- `ROADMAP.md` lines 92-93 -- tokenizer bindings entry

## Proposed Change

### 1. Token-to-tensor utility

Add a helper in `lib/nn_blocks.ml` (or a new `lib/tokenizer_utils.ml`) that converts token ID arrays from the dataprep BPE tokenizer into OCANNL tensors:

- Convert `int array` (from `Dataprep.Bpe_tokenizer.encode`) to an integer-valued OCANNL tensor
- Support padding/truncation to a fixed sequence length (needed for batched inference)
- Provide a batched variant for multiple sequences

This bridges `Dataprep.Bpe_tokenizer.encode` output with the embedding layer input expected by transformer models.

### 2. Integration test

Add a test (e.g., `test/training/tokenizer_roundtrip.ml`) that:
- Loads a tokenizer via `Dataprep.Bpe_tokenizer.from_pretrained`
- Encodes sample text to token IDs
- Converts to OCANNL tensor
- Verifies the tensor has the expected shape and values

### 3. README/ROADMAP update

Once the tokenizer is functional and the integration test passes:
- Check off the README line 73 milestone item
- Update ROADMAP.md lines 92-93 to reflect completion

### Acceptance criteria

- [ ] A utility function converts BPE tokenizer output (`int array`) to OCANNL tensors with padding/truncation
- [ ] At least one integration test demonstrates the dataprep tokenizer working with OCANNL tensors
- [ ] README line 73 checkbox is checked
- [ ] ROADMAP tokenizer entry is marked complete

### Edge cases to consider

- **Large vocabulary IDs**: Token IDs can be up to 256K (Gemma). The tensor representation must use an integer-compatible precision.
- **Variable-length sequences**: Batched inference requires padding. The utility should accept a `max_len` parameter and handle truncation/padding.
- **Special tokens**: BOS/EOS tokens may need to be prepended/appended before tensor conversion, depending on model requirements. This should be the caller's responsibility (the utility just converts IDs to tensors).

## Scope

**In scope:**
- Token-to-tensor conversion utility in OCANNL
- Integration test with the dataprep BPE tokenizer
- README/ROADMAP checkbox updates

**Out of scope:**
- The BPE tokenizer implementation itself (that is gh-ocaml-dataprep-2)
- Embedding layer implementation (that is part of the transformer inference task)
- Training a tokenizer

**Dependencies:**
- Blocked by `gh-ocaml-dataprep-2` (BPE tokenizer implementation must be available first)
- Blocks `watch-ocannl-README-md-177e6c0e` (transformer inference needs tokenized input)

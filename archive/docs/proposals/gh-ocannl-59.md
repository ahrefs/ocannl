# Add makemore example progression (gh-ocannl-59)

## Goal

Turn OCANNL's two existing character-level Names examples (`bigram.ml`,
`bigram_mlp.ml`) plus the newly merged `transformer_names.ml` into a coherent
**makemore progression** that mirrors Andrej Karpathy's *Neural Networks: Zero
to Hero* lecture series. Each "part" targets one lecture and builds on the
previous, culminating in an end-to-end reference that introduces OCANNL's
syntax, shape inference, and autodiff through increasingly sophisticated
autoregressive language models.

Issue: https://github.com/ahrefs/ocannl/issues/59

> "That would be the second-simplest complete example, after the
> `micrograd README` example." — @lukstafi

This proposal is the *umbrella* task that covers Parts 1–4 (with Part 5 a
stretch). Part 6 (transformer) is already in-tree as `transformer_names.ml`
(gh-ocannl-57, merged `03722947`) and the RNN variant is gh-ocannl-60
(proposal merged `7beeb877`).

## Acceptance Criteria

- [ ] Part 1 (Bigram) — existing `test/training/bigram.ml` is audited against
      Karpathy's counts-only + neural formulation. Any drift from the
      reference (loss formula, normalization, generation) is either fixed or
      documented with a comment explaining the intentional difference.
- [ ] Part 2 (MLP with multi-char context) — a new
      `test/training/mlp_names.ml` (or the current `bigram_mlp.ml` renamed and
      extended) implements a Bengio-style MLP: learned character embeddings,
      fixed context window of `block_size = 3`, concatenated embeddings fed
      through a hidden layer with `tanh`, and a linear output layer. Train /
      dev / test split (80 / 10 / 10, deterministic) is included.
- [ ] Part 3 (BatchNorm MLP + init) — a new
      `test/training/mlp_bn_names.ml` extends Part 2 with `batch_norm1d`
      between the hidden linear and `tanh`, and uses `kaiming` init for the
      hidden weights. A new `batch_norm1d` is added to `lib/nn_blocks.ml`,
      normalizing over the batch axis only (not spatial axes as
      `batch_norm2d` does).
- [ ] Part 4 (How OCANNL compiles gradients) — a short markdown tutorial
      (`docs/makemore_tutorial.md` or a new section in `docs/syntax_extensions.md`)
      walks through the generated backward code for the Part 3 MLP using
      `output_cd_file:true` (or equivalent flag), showing what OCANNL emits
      in place of Karpathy's manual `.backward()`. No new OCANNL code — this
      is a documentation deliverable that references existing emission.
- [ ] Each training example prints (i) final train / dev loss, (ii) a few
      generated names. Each has a matching `.expected` file so the CI
      diff-test catches regressions, and a `dune` entry under
      `test/training/dune`.
- [ ] A tutorial section (either a new top-level `docs/makemore.md` or a
      subsection within an existing docs page) links each OCANNL example to
      the corresponding Karpathy lecture video and to the PyTorch snippet
      from the upstream makemore repo. The tutorial calls out OCANNL-specific
      affordances that replace PyTorch idioms: einsum reductions for the
      softmax denominator, shape inference eliminating explicit reshape,
      `%op` / `%cd` replacing `.backward()`.
- [ ] All new / extended examples build and run on the CPU backend with
      `dune runtest` (parity with existing `bigram_mlp.ml`, which compiles
      cleanly under Metal but carries a `FIXME(#344)` about Metal buffer
      argument limits — the same comment is carried forward).

### Stretch (defer to a follow-up if not straightforward)

- [ ] Part 5 (WaveNet) — hierarchical 1D dilated causal convolutions on the
      Names character stream. Feasibility depends on whether a 1D-channel
      projection through `conv2d` with one spatial dimension set to 1 works
      cleanly (no dilation support exists in `conv2d` at HEAD — grep for
      `dilation` in `lib/nn_blocks.ml` / `tensor/operation.ml` returns
      nothing). If 1D dilated convs require new einsum support, the coder
      **files a follow-up** rather than expanding this task.

## Context

### Verified against HEAD (commit `7beeb877`, branch `master`)

**Existing examples (the makemore progression, current state):**

- `test/training/bigram.ml` — **Part 1 equivalent.** Uses
  `Dataprep.Names.get_all_bigrams () |> bigrams_to_indices`, one-hot encodes
  via `Nn_blocks.one_hot_of_int_list`, learns a single weight matrix `w`
  (implicitly `[vocab_size, vocab_size]` via shape inference), applies
  `exp(w * input + 1)` and normalizes with einsum `"...|... => ...|0"`,
  computes `-log(output_probs)` as cross-entropy, runs 11 epochs of SGD with
  `learning_rate = 1`. Generates 20 names. Minor gap vs Karpathy's lecture:
  Karpathy first shows the counts-only (non-gradient) version, then the
  equivalent neural network — `bigram.ml` only shows the latter. Audit /
  polish step: decide whether to add a short comment pointing this out.

- `test/training/bigram_mlp.ml` — **Part 2 partial.** Uses *the same bigram
  pairs* as `bigram.ml` (single-character context), passes through
  `relu ({b1} + {w1} * input)` then `exp ({w2} * ... + 1)` and normalizes.
  Hidden dim 24, 5 epochs, custom learning-rate schedule, no train/dev/test
  split. **Not Karpathy's Part 2**: missing the multi-character context
  window, learned character embeddings, and the dev/test split — all of
  which are central to the lecture. Part 2 work therefore is substantial
  even though a file exists. The file should be renamed (e.g., to
  `mlp_names.ml` — the existing `bigram_mlp.ml` name actively misleads
  about the architecture) and rewritten, or kept as a pedagogical stepping
  stone and a fresh `mlp_names.ml` added alongside it. Recommended:
  **rename** (keep a single Part 2 example to avoid duplicating Names
  data-prep scaffolding).

- `test/training/transformer_names.ml` — **Part 6 equivalent, already
  merged.** Provides reusable scaffolding: `ctx_len = 16`, `eff_seq_len`,
  `vocab_size = Dataprep.Names.dict_size`, `name_to_sequences` (pads with
  `' '`, prefixes with `'.'`, teacher-forcing offset), `prepare_dataset`,
  `seqs_to_flat_one_hot` (pre-encodes a full batch into a flat float32
  buffer). Parts 2 and 3 should **reuse** `name_to_sequences` adapted to
  `block_size = 3`: the mechanism is the same (sliding window over
  `pad * block_size @ name @ [end_of_name]` then pair each window with the
  next character).

**`lib/nn_blocks.ml` helpers already available** (by name, not line number):

- `one_hot_of_int_list ~num_classes` — builds a one-hot tensor from an
  `int list`. Used by both existing `bigram*` examples.
- `mlp_layer ~label ~hid_dim ()` — `relu ({w} * x + {b; o = [hid_dim]})`.
  The building block for Part 2's MLP.
- `mlp ~label ~hid_dims ()` — stacks `mlp_layer` + final linear. Could be
  used directly for Part 2 if we only need `relu` activations; Part 3 swaps
  `relu` for `tanh` and inserts `batch_norm1d`, so it won't reuse `mlp` as-is.
- `kaiming init_f ()`, `xavier init_f ()` — lifted initializers (Part 3
  uses `kaiming`).
- `softmax ~spec` — reduce_specified_axes + stable softmax. The existing
  bigram examples compute softmax manually inline; Parts 2/3 should switch
  to `softmax` for clarity.
- `batch_norm2d ~label ?epsilon ?momentum () ~train_step x` — normalizes
  over `o | h, w, ..c..`. **Running-stats FIXME still present** at HEAD:
  the `momentum` argument is ignored, and `train_step = None` falls through
  to use learned `gamma` / `beta` directly rather than population
  statistics. `batch_norm1d` will inherit this limitation (acceptable for a
  tutorial example; see *Design decisions* below).

**Dataset (`~/ocaml-dataprep/lib/names.ml`, installed as opam `dataprep.0.1.0`):**

- `read_names : unit -> string list` — 32K-name SSA list.
- `letters`, `letters_with_dot` — alphabet; `dict_size = 28`
  (`. space a..z`).
- `char_index : char -> int`, `char_to_one_hot`, `char_to_index_tbl`.
- `get_all_bigrams`, `bigrams_to_indices` — used by current `bigram.ml`.
- **No** pre-built `block_size`-context generator, **no** train/dev/test
  split. Parts 2/3 add these locally (or, if the coder prefers, upstream
  them to `ocaml-dataprep`; scope note below).

**PPX / tensor DSL primitives** (`tensor/operation.ml`):

- `tanh` is a proper primitive with op-label `tanh`, exported on `NTDSL` /
  `TDSL`. Safe to use directly in Parts 2/3.
- **No `sigmoid` primitive** (confirmed by grep); not required for this
  task — `tanh` is Karpathy's chosen activation for the Part 3 saturation
  demonstration.
- Shape inference lets us write `{ emb; o = [embed_dim] } * input_one_hot`
  and let OCANNL infer the `vocab_size` axis from the one-hot input — the
  embedding table idiom used by the existing `bigram_mlp.ml`.

### Design decisions surfaced by verification

1. **Embedding lookup style.** Karpathy uses advanced indexing
   (`C[x]`) for the embedding table. OCANNL's current idiom (used by
   `bigram.ml` / `bigram_mlp.ml`) is one-hot × matrix multiply, which is
   semantically equivalent and already works through shape inference and
   backprop. An index-lookup formulation via `Fetch` could be more
   efficient but is **not a goal of this task** — it would be a separate
   performance improvement applicable beyond makemore. **Decision:** stay
   with one-hot × matmul; file a follow-up only if the Part 2/3 batch sizes
   expose noticeable slowdowns.

2. **`batch_norm1d` running-stats.** The existing `batch_norm2d` ignores
   `momentum` and has a `FIXME` for population statistics. `batch_norm1d`
   should mirror the 2D version's structure (same parameter lifting,
   `gamma` / `beta`, `epsilon`, `momentum` threaded through but unused)
   rather than diverging. A shared running-stats fix is out of scope here;
   **document the limitation in the new block's ocamldoc** and reference
   the 2D FIXME.

3. **Part 2 file naming.** Current `bigram_mlp.ml` predates `transformer_names.ml`.
   Its name refers to its *input-context width* (bigram = 1-char input)
   rather than its architecture (MLP). Once Part 2 adopts
   `block_size = 3`, the filename is actively wrong. Recommended rename:
   `bigram_mlp.ml` → `mlp_names.ml`. The associated `.expected` file and
   `dune` stanza follow. This *is* a visible workflow change and deserves
   a line in `CHANGES.md`.

4. **Batch-size discipline.** `block_size = 3` expands ~32 k names with
   avg ~6 chars to ~200 k training contexts vs bigram_mlp's ~230 k bigrams
   (comparable). Reuse `batch_size = 1000` from `bigram_mlp.ml` as a safe
   starting point; the coder may tune.

5. **Part 4 depth.** The tutorial should (a) show how to set
   `output_cd_file:true` (or whichever debug flag dumps the emitted code),
   (b) walk through the generated backward for a simple sub-expression
   (e.g., `tanh(W @ x + b)`), (c) map each emitted line to the
   corresponding manual-backward line from Karpathy's lecture. Aim for
   ~200–400 words; don't try to re-teach autodiff.

6. **WaveNet / Part 5.** OCANNL's `conv2d` has no dilation support and
   `tensor/operation.ml` has no causal-mask einsum helpers. Implementing
   dilated causal 1D convs requires either extending the einsum (new
   `stride` variants) or emulating dilation by re-indexing the input.
   Neither is small. **Keep as stretch**; file a dedicated follow-up if
   the coder attempts it and hits friction.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

### Phase 1 (Part 1) — audit existing `bigram.ml`

Low-effort review. Compare the loss formula and generation loop against
Karpathy's lecture 1 neural-network formulation. If drift is found, fix
minimally; otherwise add one or two tutorial-targeted comments. No new
files.

### Phase 2 (Part 2) — rewrite as multi-char context MLP

```ocaml
let block_size = 3
let embed_dim = 10
let hid_dim = 200

(* Data prep: reuse name_to_sequences's padding style but with block-size window. *)
let name_to_contexts name =
  let padded = List.init block_size ~f:(fun _ -> '.') @ String.to_list name @ ['.'] in
  (* slide block_size + 1 window; produce (context_chars, target_char) pairs *)
  ...

(* Model:
   input: one-hot of shape [batch | block_size, vocab_size]
   embed: { C; o = [embed_dim] } * input, giving [batch | block_size, embed_dim]
   flatten block_size × embed_dim → one MLP input via einsum:
     "... | b, e => ... | (b * e)"     (concat along a fused axis)
   hidden: tanh ({w1} * flat + {b1; o = [hid_dim]})
   logits: {w2} * hidden + {b2}        (shape inference fills vocab_size)
   loss:  log-softmax cross-entropy (reuse the numerically-stable form from
          transformer_names.ml rather than the older manual softmax in bigram.ml)
*)
```

Add a deterministic 80/10/10 split using `Dataprep.Names.read_names ()`
piped through a fixed-seed shuffle before tokenization.

### Phase 3 (Part 3) — BatchNorm MLP

Add to `lib/nn_blocks.ml` just below `batch_norm2d`:

```ocaml
(** Batch normalization for MLPs - normalizes over the batch dimension only.
    See {!batch_norm2d} for the running-statistics FIXME; this block inherits
    the same limitation. *)
let%op batch_norm1d ~label ?(epsilon = 1e-5) ?(momentum = 0.9) () ~train_step x =
  let _ = momentum in
  (* x has shape [... | ..c..] - batch axes on the left, channel axes on the right *)
  let mean = (x ++ "..o.. | ..c.. => 0 | ..c..") /. (dim o) in
  let centered = x - mean in
  let variance = ((centered *. centered) ++ "..o.. | ..c.. => 0 | ..c..") /. (dim o) in
  let std_dev = sqrt (variance + !.epsilon) in
  let normalized = centered /. std_dev in
  match train_step with
  | Some _ -> ({ gamma = 1. } *. normalized) + { beta = 0. }
  | None -> (gamma *. normalized) + beta
```

*(The exact einsum dimension naming may need adjustment — `dim o` is the
right-hand-side reduction dimension in the 2D version; reuse whatever name
the existing 2D implementation uses. Coder verifies at build time.)*

Part 3 example (`mlp_bn_names.ml`) reuses Part 2's data-prep verbatim and
the model becomes:

```
input → embed → flatten → linear (kaiming) → batch_norm1d → tanh → linear → logits
```

### Phase 4 (Part 4) — documentation

Add `docs/makemore_tutorial.md` (or a new section in `syntax_extensions.md`).
The "compiled backward" walkthrough should use the Part 3 model because
it's the lecture's focus, but the actual generated code to quote can come
from a tiny sub-expression for readability.

### Phase 5 — linking tutorial

Final 200–400 words of `docs/makemore_tutorial.md`: a table mapping each
OCANNL example to its Karpathy video + the PyTorch repo file. Cross-link
from `README.md`'s examples section.

### Phase 6 (stretch) — WaveNet

Decide-and-defer decision. If the coder concludes that dilated 1D conv
requires einsum work, they file a follow-up issue (tentatively tagged
`example-wavenet`) and close this task with Parts 1–4 complete.

## Scope

**In scope**

- `lib/nn_blocks.ml`: add `batch_norm1d` mirroring `batch_norm2d`.
- `test/training/bigram.ml`: audit / polish only.
- `test/training/bigram_mlp.ml` → `test/training/mlp_names.ml`: rename and
  rewrite for `block_size = 3` Bengio-style MLP.
- `test/training/mlp_bn_names.ml` (new): Part 3 BatchNorm variant.
- Corresponding `.expected` files and `test/training/dune` stanzas.
- `docs/makemore_tutorial.md` (new) — Parts 4 + linking tutorial.
- `CHANGES.md` entry covering the rename, `batch_norm1d`, and the new
  examples.

**Out of scope** (file follow-ups if hit)

- Fixing `batch_norm2d` running statistics (applies to `batch_norm1d`
  too — same underlying framework gap).
- Upstreaming `block_size`-windowed context generation or a deterministic
  train/dev/test split helper to `ocaml-dataprep`. Keep them local to
  `mlp_names.ml` for now; upstream in a separate dataprep PR if they
  prove useful elsewhere.
- `Fetch`-based embedding lookup as a performance optimization.
- 1D dilated causal convolutions / WaveNet — defer per Phase 6.
- RNN / LSTM / GRU variants — covered by gh-ocannl-60.
- Transformer variant — already in-tree as `transformer_names.ml`.

**Dependencies**

- `relates_to: gh-ocannl-57` — **merged at HEAD**; the transformer example
  is the Part 6 continuation that this tutorial should link to. The
  `relates_to` pointer can remain.
- `relates_to: gh-ocannl-60` (LSTM example, proposal merged) — cross-link
  both ways once the LSTM example lands; no hard dependency.
- No hard `blocked_by` — all prerequisites are in-tree.

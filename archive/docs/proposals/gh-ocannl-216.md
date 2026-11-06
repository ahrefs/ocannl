# Examples from Fleuret's Deep Learning Course (and dataflowr)

## Goal

GitHub issue: https://github.com/ahrefs/ocannl/issues/216 (label: `explore`, milestone: v1.1).

The user has linked François Fleuret's *Deep Learning Course* (fleuret.org/dlc/, 13 PyTorch-based lectures) and the *Deep Learning Do It Yourself!* course (dataflowr.github.io) as candidate sources for OCANNL example translations. The intent expressed in the issue body is twofold:

1. **Validate** that OCANNL's API can express idioms taught in mainstream PyTorch courses.
2. **Build a small library of educational examples** that someone arriving from a PyTorch background can read.

This proposal is **artifact-only** by user direction — the user has paused autonomous OCANNL work pending a hands-on quality audit (see harness memory, 2026-04-29). The point of writing it now is to convert an open-ended "translate ~13 lectures" backlog into a concrete, defensible, per-file set of acceptance criteria that can be queued cleanly when work resumes. No implementation should follow until the user explicitly unblocks.

## Acceptance Criteria

The proposal is **complete** (this artifact accepted) when each of these is true:

- **AC-A. A named example shortlist exists.** A specific, finite list of OCANNL example files to add (committed below in *Approach*), each with the source lecture/notebook it derives from. The list is *closed* (no "and similar"), justified relative to OCANNL's existing example coverage, and sized so a single reasonable PR is feasible per file.
- **AC-B. Per-file ACs are specified.** For each proposed example, an explicit acceptance contract: what the script must print/return, what numerical target counts as "correct," and the comparison target against the PyTorch reference. (See *Per-Example AC Template* below.)
- **AC-C. The 5f24bd63 overlap is resolved.** `watch-ocannl-README-md-5f24bd63` (MNIST/CIFAR convnets) is already **done** (`mnist_conv.ml`, `cifar_conv.ml` shipped under `proposal: docs/proposals/convnet-examples.md`). This proposal explicitly does not re-cover that ground; basic image classification with CNNs is removed from scope and the rationale is recorded.
- **AC-D. API-gap reporting protocol is fixed.** A decision is recorded for *how* gaps surfaced during translation get filed (one summary umbrella issue with checkboxed sub-items vs. one issue per gap) and *what threshold* triggers filing.
- **AC-E. Out-of-scope items are listed.** "Translate all 13 lectures," "build a tutorial site," "achieve PyTorch parity benchmarks," and "non-correctness PyTorch comparisons" are explicitly excluded.

(Implementation of the examples themselves is *not* an AC of this proposal — it is the body of the follow-up tasks the proposal carves out.)

## Context

### Existing OCANNL example coverage (read first)

`test/training/`:
- `moons_demo.ml` — MLP on 2D moons (binary classification)
- `mlp_names.ml`, `mlp_bn_names.ml` — MLP with/without batch norm on character-name dataset
- `bigram.ml` — character-level bigram model
- `circles_conv.ml` — LeNet on synthetic 16×16 circles (3 classes)
- `mnist_conv.ml` — LeNet on MNIST (shipped via gh-ocannl-54 / 5f24bd63)
- `cifar_conv.ml` — deeper CNN on CIFAR-10 (shipped via gh-ocannl-54 / 5f24bd63)
- `transformer_names.ml` — small transformer LM on names
- `fsm_transformer.ml` — transformer learning a finite-state machine task

### Existing OCANNL building blocks (`lib/nn_blocks.ml`)

Identified via top-level definitions:

- **Init**: `kaiming`, `xavier`, `normal`, plus `*_at` counter-indexed variants
- **Loss helpers**: `one_hot_of_int_list`, `softmax`
- **MLP / CNN**: `mlp`, `conv2d`, `max_pool2d`, `lenet`, `vgg_block`, `resnet_block`, `conv_bn_relu`
- **Normalization**: `layer_norm`, `batch_norm2d` (has FIXME at line 327: no running statistics for inference)
- **Attention / transformer**: `multi_head_attention`, `transformer_encoder`, `transformer_decoder`, `decoder_only`, RoPE (`rope_frequencies`, `apply_rope` patterns), `sinusoidal_position_encoding`
- **Regularization**: `dropout`

### Conspicuously **absent** (likely API-gap territory)

- **No recurrent blocks** — no `rnn_cell`, `lstm_cell`, `gru_cell`, no sequence-unrolled training pattern. Fleuret's lectures cover RNNs explicitly; an LSTM example would either require building these blocks first or would surface an honest "we recommend the transformer path instead" answer.
- **No VAE / GAN / autoencoder** examples or helpers — generative-model lectures would land entirely in user code.
- **No data-augmentation utilities** — Fleuret's CIFAR lectures lean on standard PyTorch `torchvision.transforms`. OCANNL would either inline minimal augmentation or skip it.
- **`batch_norm2d` inference path** is incomplete (no running stats). Any example relying on test-time BN reproduction would expose this.

### How "correctness" is established for example translations

OCANNL's existing example tests use `.expected` files that pin **printed output** — typically a coarse decreasing-loss trajectory or a final accuracy band. They do **not** attempt bit-equivalence with PyTorch. Per-file ACs in this proposal follow the same convention: a *band* on final loss / accuracy that is loose enough to absorb floating-point divergence and tight enough to detect real regressions.

### Why a finite shortlist (not "5–10 examples")

The original task elaboration suggested "5–10 representative examples." After auditing existing coverage, the residual *uncovered* surface is much narrower than 5–10 lectures' worth — most of Fleuret's syllabus (tensor ops, MLP, CNN image classification, attention/transformer, regularization, embeddings) is already represented in `test/training/`. Continuing to spec "5–10 examples" would either duplicate existing work or push into RNN/generative territory that requires *new building blocks*, not new examples. The shortlist below reflects this audit.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

### Per-Example AC Template

Each proposed example, when implemented, must satisfy:

1. **File location**: `test/training/<name>.ml`, registered in `test/training/dune` as a `(test ...)` stanza.
2. **Source attribution**: header comment naming the Fleuret lecture (or dataflowr notebook) it derives from, with URL.
3. **Numerical contract**: a `.expected` file pinning a printed loss/accuracy trajectory at a *fixed* random seed and *fixed* step count, with a stated tolerance band derived from running the PyTorch reference 3+ times and taking [min−ε, max+ε] of its final metric. (Bands, not exact equality.)
4. **Runtime budget**: under 60 seconds on the CPU backend at the specified step count, so it functions as a regression test. Larger reference runs may be left as commented-out alternatives.
5. **API-gap log**: any place where translation required a workaround, a `(* API-gap: ... *)` comment, gathered into the umbrella issue per AC-D.

### Proposed Shortlist

The shortlist is **3 examples**, deliberately small, chosen so that each one exercises a part of OCANNL's API not already tested by an existing example *and* maps cleanly to a specific Fleuret lecture. Each is independently mergeable.

#### Example 1 — `autoencoder_mnist.ml`

- **Source**: Fleuret Lecture 8 (autoencoders) / dataflowr "Module 8a" equivalent.
- **Why it's not redundant**: existing MNIST example (`mnist_conv.ml`) is a *classifier*. An MLP/conv autoencoder reuses the dataset but exercises a fundamentally different training shape (reconstruction loss, no labels) and validates that OCANNL handles tied or mirrored encoder/decoder topologies cleanly.
- **Per-file AC**:
  - Trains an encoder (MLP or conv) → bottleneck → decoder on MNIST images.
  - Reconstruction MSE on a held-out set drops below a fixed threshold within N epochs (threshold + N to be set during implementation, anchored to PyTorch reference run).
  - `.expected` pins the loss-decrease trajectory.
- **Likely API gaps to log**: weight tying (if attempted), `transposed conv2d` / upsampling (if conv variant chosen — `nn_blocks` currently has no transpose-conv block).

#### Example 2 — `vae_mnist.ml`

- **Source**: Fleuret Lecture 11 (generative models — VAE section) / dataflowr "Module 8b".
- **Why it's not redundant**: there is **no generative example** in the current suite. A VAE is the smallest defensible generative model — it stays within the existing optimizer/loss machinery (just KL + reconstruction) and avoids the adversarial-training scaffolding that a GAN would need.
- **Per-file AC**:
  - Trains an encoder → (μ, log σ²) → reparameterized sample → decoder on MNIST.
  - Combined loss (reconstruction + β·KL) decreases monotonically over the pinned trajectory.
  - Generates samples by decoding `z ~ N(0, I)` and prints a small ASCII or label-distribution summary (no image-file I/O required for the test).
  - `.expected` pins the loss trajectory.
- **Likely API gaps to log**: convenience for sampling from a parameterized normal; KL-divergence computation against a unit Gaussian (will likely be inlined and worth promoting to a helper if pattern repeats).

#### Example 3 — `lstm_or_recurrent_<task>.ml` *(conditional — see below)*

- **Source**: Fleuret Lecture 11 (RNN/LSTM section) / dataflowr "Module 6".
- **Status**: this example is **proposed conditionally** because OCANNL has no recurrent building blocks today. Two sub-options:
  - **3a. Build minimal LSTM cell + char-level sequence task** — e.g., palindrome detection or simple sequence-tagging. This is a substantial addition (`lstm_cell` block in `nn_blocks.ml`, plus an unrolled-training pattern). Likely *not* a single example PR — would itself be a follow-up task with its own proposal.
  - **3b. Skip RNN coverage entirely**, with a written rationale that OCANNL's intended path for sequence tasks is the transformer family (already covered by `transformer_names.ml` and `fsm_transformer.ml`), and Fleuret's RNN lecture is intentionally not translated.
- **Decision deferred to user** — surfaced as Q3 below. The proposal commits the user to picking 3a or 3b, but does not pre-commit to 3a's implementation cost.

### What is *not* on the shortlist (and why)

- **MLP / classification basics** — covered by `mlp_names.ml`, `moons_demo.ml`.
- **CNN image classification (MNIST, CIFAR)** — covered by `mnist_conv.ml`, `cifar_conv.ml` (5f24bd63 done).
- **Attention / transformer LM** — covered by `transformer_names.ml`, `fsm_transformer.ml`.
- **Word embeddings / bigram** — covered by `bigram.ml`.
- **Batch-norm demo** — covered by `mlp_bn_names.ml`.
- **GAN** — too much new scaffolding (alternating optimizers, gradient penalty for stability) for an "example translation" task. If pursued, it deserves its own proposal.
- **Modern self-supervised methods (SimCLR, MAE, etc.)** — out of scope of Fleuret's 2020-vintage lectures.
- **Lecture-by-lecture parity** — the goal is API validation, not pedagogical reproduction.

### API-Gap Reporting Protocol (resolves AC-D)

**Recommendation**: **single umbrella issue**, opened against `ahrefs/ocannl` at the start of implementation work, titled e.g. *"API gaps surfaced by Fleuret-course example translations (gh-ocannl-216 follow-up)"*, with a checkbox list. Each gap that is encountered during the translation of one of the shortlist examples gets:

- A checkbox entry in the umbrella issue with a one-line description and a code pointer to the workaround.
- A `(* API-gap: see #<umbrella-issue> *)` comment at the workaround site.
- **Promotion criterion**: any gap that requires more than a 10-line workaround, *or* would change a public signature in `nn_blocks.ml` / `tensor/operation.ml`, gets promoted to its own dedicated issue (and the umbrella entry links to it).

Rationale for umbrella-by-default: separate issues per gap fragment the discussion before the user has decided which gaps are worth fixing; one umbrella keeps the proposal-author's findings collected and the user can spin off issues at audit time. The 10-line / public-signature threshold prevents the umbrella from absorbing genuinely big gaps that deserve their own design discussion.

## Scope

**In scope:**
- This proposal artifact (the document you are reading).
- The shortlist as defined above and the per-file AC template.
- The API-gap reporting protocol decision.
- A clean carve-out from the already-completed 5f24bd63 work.

**Out of scope:**
- Implementing any of the shortlisted examples — paused pending user audit; each will be a separate task when unblocked.
- Choosing 13 examples / lecture-by-lecture parity.
- Building a tutorial site or rendered HTML companion.
- Comparing OCANNL to PyTorch beyond what's needed to verify per-file numerical correctness (no benchmark tables, no qualitative reviews).
- New `nn_blocks.ml` building blocks (LSTM cell, transposed conv, etc.) — if needed by a shortlist example, those become their own predecessor tasks.
- Fixing the existing `batch_norm2d` running-stats FIXME — separate concern; flag-only if it blocks an example.

**Dependencies:**
- **Subsumes/relates-to** `watch-ocannl-README-md-5f24bd63` (already done; non-overlap is established by removing CNN classification from this scope).
- **Related** `gh-ocannl-182` (Bonsai networks reproduction) — same "translate from another framework" pattern; if both proceed, share the API-gap umbrella-issue convention.
- **Potentially blocked by** future RNN-cell work, *only* if user picks option 3a.

## Open Questions for the User (block scheduling implementation)

These are the genuine ambiguities the user should resolve before the shortlist becomes implementation-ready:

1. **Example-count target.** Is the shortlist of 3 (autoencoder, VAE, optional RNN) acceptable, or does the user want a different size — e.g., add a GAN, drop the VAE, etc.? The original task hinted at 5–10; the audit argues for 3.
2. **Course selection.** Fleuret only? Fleuret + dataflowr? Or topic-driven without fixed allegiance to either course (the current proposal effectively does the third)?
3. **RNN handling (Example 3).** Pick 3a (build LSTM cell + example) or 3b (skip, document the transformer-first stance). If 3a: this becomes its own task with its own proposal; this proposal does not pre-authorize the building-block work.
4. **API-gap filing.** Confirm the "umbrella issue + 10-line / public-signature promotion threshold" protocol, or specify an alternative (e.g., "every gap is its own issue, no umbrella").

Until these are resolved, no implementation tasks should be cut from this proposal.

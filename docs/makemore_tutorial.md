# makemore: a character-level language-model progression

[makemore](https://github.com/karpathy/makemore) is a series of progressively
more sophisticated autoregressive character-level language models from Andrej
Karpathy's *Neural Networks: Zero to Hero* lecture series. All of them train on
the same dataset — ~32k first names from the [SSA
names](https://github.com/karpathy/makemore/blob/master/names.txt) list — and
ask the same question: *given a prefix of characters, what is the next
character?*

This tutorial walks through an OCANNL implementation of that progression,
mirroring Karpathy's structure part-by-part. Each part produces a standalone
runnable example under `test/training/`:

1. [Part 1 — Bigram](#part-1--bigram) → [`bigram.ml`](../test/training/bigram.ml)
2. [Part 2 — Bengio MLP](#part-2--bengio-mlp) → [`mlp_names.ml`](../test/training/mlp_names.ml)
3. [Part 3 — BatchNorm MLP](#part-3--batchnorm-mlp) → [`mlp_bn_names.ml`](../test/training/mlp_bn_names.ml)
4. [Part 4 — How OCANNL compiles gradients](#part-4--how-ocannl-compiles-gradients)
5. [Part 5 — WaveNet](#part-5--wavenet) *(stretch goal — deferred)*
6. [Part 6 — Transformer](#part-6--transformer) → [`transformer_names.ml`](../test/training/transformer_names.ml)

All examples run on the CPU (`OCANNL_BACKEND=sync_cc dune runtest
test/training/`) and use `Utils.settings.fixed_state_for_init <- Some 3` so
that `.expected` fixtures stay deterministic across runs.

## Dataset and vocabulary

The Names dataset is provided by the opam package
[`dataprep`](https://github.com/ahrefs/ocaml-dataprep). It exposes a dict of
size 28: the 26 lowercase letters plus `.` (used as both beginning- and
end-of-name marker) and a space (used to pad short names where a fixed-length
sequence is required). The helpers we use across the examples:

- `Dataprep.Names.read_names : unit -> string list` — returns ~32k names.
- `Dataprep.Names.char_index : char -> int` — alphabet → index.
- `Dataprep.Names.letters_with_dot : char list` — index → character.

The examples below each split the name list deterministically into
80% / 10% / 10% train / dev / test via a Fisher–Yates shuffle with a fixed
seed (`split_seed = 42`).

## Part 1 — Bigram

**Karpathy:** [*Building makemore Part 1: bigrams, probabilities,
negative-log-likelihood*](https://www.youtube.com/watch?v=PaCmpygFfXo)

The bigram model predicts the next character given only the *previous*
character. Each training example is a pair `(char_prev, char_next)`.
[`bigram.ml`](../test/training/bigram.ml) implements the neural-net formulation
directly — one weight matrix `w` of shape `[vocab_size -> vocab_size]`, softmax
per row, cross-entropy against the one-hot target. Karpathy's lecture shows a
counts-only warm-up notebook step first; we skip that and go straight to the
neural formulation to keep the training + generation loop compact.

The softmax denominator is expressed as an einsum reduction:

```ocaml
let counts = exp (({ w } + 1) * input) in
counts /. (counts ++ "...|... => ...|0")
```

In PyTorch that would be `F.softmax(w @ input, dim=-1)`. In OCANNL the
`...|... => ...|0` einsum reduces the *output* axis to size 1 by summing,
leaving batch and other axes untouched. This is the same pattern you'll see
repeated in each subsequent part, plus a numerical-stability `max` subtraction
starting from Part 2.

Sampling is a CDF-based walk over the 28 outputs (`counts[i] / sum(counts)`),
terminated when `.` is emitted.

Karpathy's lecture additionally applies a small `0.01 * (W ** 2).mean ()`
regularizer. We omit it — the loss thresholds in `bigram.expected` already
accommodate its absence — and keep the example focused on the core softmax +
cross-entropy + generation triangle.

## Part 2 — Bengio MLP

**Karpathy:** [*Building makemore Part 2: MLP*](https://www.youtube.com/watch?v=TCH_1BHY58I)

[`mlp_names.ml`](../test/training/mlp_names.ml) is the Bengio et al. 2003
neural probabilistic language model: a learned character embedding table `c`
of shape `[vocab_size -> embed_dim]` (here `embed_dim = 10`), a fixed context
window of `block_size = 3` preceding characters, and an MLP over the
concatenated embeddings:

```
     ┌─ C[x_{t-2}] ─┐
x →  │  C[x_{t-1}]  │ → tanh (W1 @ concat + b1) → (W2 @ hidden + b2) → softmax
     └─ C[x_t]     ─┘
```

OCANNL's shape inference replaces Karpathy's explicit `.view(-1, n_embd *
block_size)` reshape. We keep `block_size` as a *batch* axis of the embedded
tensor and use an einsum contraction with `+*` to fuse `block_size` and
`embed_dim` into the hidden weight:

```ocaml
let%op embed input = { c; o = [ embed_dim ] } * input in
(* embed : batch=[batch_size, block_size], output=[embed_dim] *)

let%op hidden x =
  tanh
    ((embed x +* { w1 } "bs|->e; |se->h => b|->h" [ "s"; "e" ])
    + { b1; o = [ hid_dim ] })
(* hidden : batch=[batch_size], output=[hid_dim] *)
```

The einsum spec reads "LHS has batch axes `b,s` and output `e`; RHS (`w1`) has
input axes `s,e` and output `h`; result has batch `b` and output `h`." The
repeated `s` and `e` are the contracted axes — same semantics as PyTorch's
matrix multiply over a flattened `[block_size * embed_dim]` view, but here
expressed without a reshape.

The cross-entropy loss uses the numerically-stable log-softmax (subtract per-
row max before exp) — identical to the transformer example and Karpathy's
later lectures.

### Weight initialization

OCANNL's default parameter initializer is a centered, scaled `uniform1`
distribution over `[-0.25, 0.25)`. It keeps the arbitrary-shape behavior of the
non-vectorized `uniform1` path while avoiding all-positive weights. For an MLP
with `tanh`, that centered default avoids saturating the preactivation and
trapping SGD at a high-loss plateau.

Karpathy's lecture handles the same pain point a different way, which is
itself the entry point to Part 3.

Under the fixed seed, `mlp_names.ml` converges to a final train/dev/test NLL
of ~2.49 over 15 epochs. The three sampled names (`hadasi`, `koun`, `kinre`)
are recognizably Names-like.

## Part 3 — BatchNorm MLP

**Karpathy:** [*Building makemore Part 3: Activations & Gradients,
BatchNorm*](https://www.youtube.com/watch?v=P6sfmUTpUmc)

[`mlp_bn_names.ml`](../test/training/mlp_bn_names.ml) extends Part 2 by
inserting [`Nn_blocks.batch_norm1d`](../lib/nn_blocks.ml) between the hidden
linear and the `tanh`. The new block mirrors the existing `batch_norm2d` but
normalizes over the batch axis only (no spatial axes to reduce). Its einsum
is:

```ocaml
let mean = (x ++ "..o.. | ..c.. => 0 | ..c.." [ "o" ]) /. dim o in
```

where `o` is the captured batch-axis length and `..c..` is the channel row
variable passed through unchanged.

The model becomes:

```
embed → (w1 +* ... + b1) → batch_norm1d → tanh → (w2 * hidden + b2)
```

The hidden weight `w1` uses Kaiming-normal initialization — `normal1 ()`
sampled from a standard normal then scaled by `sqrt(scale_sq / fan_in)` per
[`Nn_blocks.kaiming`](../lib/nn_blocks.ml). Threading `train_step` through
the layers requires the unit-closure pattern from `nn_blocks.ml`:

```ocaml
let bn1 = Nn_blocks.batch_norm1d ~label:[ "bn1" ] () in
let%op mk_hidden () ~train_step x =
  tanh
    (bn1 ~train_step
       ((embed x +* { w1 = kaiming normal1 () }
           "bs|->e; |se->h => b|->h" [ "s"; "e" ])
       + { b1; o = [ hid_dim ] }))
in
let hidden = mk_hidden ()
```

`let%op mk_hidden () ~train_step x = ...` lifts the inline `{ w1 = … }` and
`{ b1 }` parameters to the `()` closure scope — they are constructed *once*
when `mk_hidden ()` is called, and shared across every subsequent invocation
(training, dev/test eval, inference generation). The explicit Kaiming
initializer on `w1` overrides the default centered `uniform1` initializer so
Kaiming's fan-in scale is preserved; `c`, `b1`, `w2`, and `b2` use the default
`[-0.25, 0.25)` range.

### Known limitation — running statistics

`batch_norm1d` inherits `batch_norm2d`'s FIXME: running statistics are not
implemented, so `momentum` is ignored and the inference path
(`~train_step:None`) falls through to `(gamma *. normalized) + beta` computed
from *batch* statistics rather than population statistics. For a single-
example inference batch, `mean == x`, `centered == 0`, `normalized == 0`, so
the output collapses to `beta` regardless of input. Generation quality
degrades accordingly — `mlp_bn_names.ml`'s three sampled names are noticeably
noisier than Part 2's (`ria`, `ehnlk`, `lc` under the fixed seed). The tutorial
leaves this as a pedagogical demonstration of why running statistics matter;
the framework-level fix is tracked as a follow-up to this task.

## Part 4 — How OCANNL compiles gradients

**Karpathy:** [*Building makemore Part 4: becoming a backprop
ninja*](https://www.youtube.com/watch?v=q8SA3rM6ckI)

Part 4 of the lecture series replaces `loss.backward()` with a hand-written
backward pass through the Part 3 MLP — the reader implements gradient
formulas for each forward operation.

OCANNL does the equivalent automatically. The `%cd` / `%op` syntax extensions
generate both forward and backward code at `Train.grad_update` time. To read
the generated code for yourself, enable the debug flag:

```bash
OCANNL_BACKEND=sync_cc \
  dune exec test/training/mlp_bn_names.exe \
  -- --ocannl_output_debug_files_in_build_directory=true
```

After the run, `_build/default/test/training/build_files/` holds three files
per compiled routine:

1. `*.cd` — the high-level assignment IR (forward + backward interleaved),
2. `*.ll` — the low-level for-loop IR, and
3. `*.c` / `*.cu` / `*.metal` — the backend-specific source passed to the
   compiler.

Skim the `.cd` file for the training routine and look at the sub-expression
for `tanh(w1 +* embed x + b1)`. Karpathy's lecture has the reader write, by
hand:

```python
dh           = (1 - h ** 2) * dhpreact     # d tanh / d preact
dhpreact     = dh                          # alias for clarity
dw1          = embed.T @ dhpreact          # weight gradient
dembed       = dhpreact @ w1.T             # backprop into embed
```

In OCANNL the emitted `.cd` expresses the same four gradient terms as
explicit assignment statements, annotated by the tensor label they correspond
to. The `tanh` derivative in particular surfaces as an elementwise
multiplication of the upstream gradient by `1 - h**2`, matching the manual
formula line-for-line. The difference is that you don't maintain the
gradient bookkeeping — the PPX threads it for every operation you wrote in
`%op` / `%cd`, and re-threads it when you change the model.

Treat Part 4 as the bridge from "I could hand-write a gradient" to "the
compiler has already written it" — the same mental model, without the upkeep.

## Part 5 — WaveNet

**Karpathy:** [*Building makemore Part 5: Building a
WaveNet*](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)

Part 5 introduces hierarchical 1D dilated causal convolutions. OCANNL's
`conv2d` has no dilation support at HEAD, and the einsum does not yet have
first-class stride-dilation primitives for the 1D case. A WaveNet example is
therefore deferred to a follow-up issue — implementing it is a meaningful
einsum / lowering extension, not a tutorial polish.

## Part 6 — Transformer

**Karpathy:** Context covered across *Zero to Hero* Parts 7–9 (intro to
language modelling → GPT from scratch), condensed into the decoder-only GPT.

[`transformer_names.ml`](../test/training/transformer_names.ml) is a
masked-attention decoder-only transformer trained on the Names dataset with a
16-character context and teacher-forced targets. It was merged as part of
[`gh-ocannl-57`](https://github.com/ahrefs/ocannl/issues/57) and is the
continuation of this tutorial — once you've worked through Parts 2 and 3, the
transformer replaces the fixed-context-window MLP with self-attention and
adds positional encoding + FFN residual blocks.

The same data-prep primitives (`name_to_sequences`, `seqs_to_flat_one_hot`)
are visible there, adapted to a full sequence instead of a sliding window.

---

*Mapping summary*

1. Part 1 → [`bigram.ml`](../test/training/bigram.ml)
2. Part 2 → [`mlp_names.ml`](../test/training/mlp_names.ml)
3. Part 3 → [`mlp_bn_names.ml`](../test/training/mlp_bn_names.ml)
4. Part 4 → *this page, §Part 4*
5. Part 5 → *deferred — follow-up issue*
6. Part 6 → [`transformer_names.ml`](../test/training/transformer_names.ml)

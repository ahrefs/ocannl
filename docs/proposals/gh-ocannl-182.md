# Reproduce "Growing Bonsai Networks with RNNs" (gh-ocannl-182)

## Status update (2026-06-12)

- Issue [ahrefs/ocannl#182](https://github.com/ahrefs/ocannl/issues/182) is still OPEN (label: explore), milestone v1.0. Per ROADMAP.md (authoritative) v1.0 targets end of October 2026; the GitHub milestone due date (June 2026) lags.
- The Ludics harness task records a start on 2026-04-30, but nothing has landed: there is no `test/training/bonsai_rnn.ml`, no related branch, and no pruning helper in `lib/train.ml`.
- gh-ocannl-59 (makemore) is now CLOSED — `test/training/mlp_names.ml`, `mlp_bn_names.ml` and `batch_norm1d` in `lib/nn_blocks.ml` landed, giving extra training-loop scaffolding to crib from (also new: `fsm_transformer.ml` from gh-ocannl-116, conv examples).
- gh-ocannl-60 (LSTM example) is still OPEN (milestone v0.7.1); no `lstm_cell` exists in `lib/nn_blocks.ml`, so the recommended fallback — a local vanilla `tanh`-RNN cell — applies.
- Cited APIs re-verified at HEAD: `Tn.set_values`/`get_values` (`arrayjit/lib/tnode.ml:752`/`758`), `to_dag`/`to_printbox` (`tensor/tensor.ml`), `Train.run_once`/`init_params`, `Tensor.params`, `Utils.settings.fixed_state_for_init`, `relu`/`sat01`/`tanh` exported and still no `sigmoid` primitive, still no Graphviz exporter.
- Repo-wide renames since April 2026 (broadcast-order reversal LUB→GLB, "label"→"basis") do not touch any identifier this proposal relies on.
- The entire implementation remains to do.

## Goal

Reproduce the central result from Casey Primozic's blog post
[*Growing Sparse Computational Graphs with RNNs*](https://cprimozic.net/blog/growing-sparse-computational-graphs-with-rnns/)
(originally implemented in Tinygrad): train a small RNN on a deterministic
sequence task, apply magnitude pruning to drive >90% of weights to zero, and
render the surviving sub-network as a human-interpretable computational graph.
This is an *explore* / `v1.0` task — the value is demonstrating that OCANNL's
static-graph compilation model accommodates recurrent training, post-training
parameter mutation, and graph-level visualization of the trained-and-pruned
artifact.

Issue: https://github.com/ahrefs/ocannl/issues/182

## Acceptance Criteria

- [ ] **Task and dataset**: a deterministic binary-sequence task is implemented
      in the example. The reference baseline is the blog's `y[n] = x[n-2]`
      task (output equals the input from two timesteps prior, on a stream of
      uniform-random bits). Training data is generated on the fly from a target
      program; no external dataset is added.
- [ ] **RNN training**: the example builds, trains, and converges on the chosen
      task. Convergence is checked numerically (cross-entropy or MSE on a
      held-out validation stream of >=10k tokens). The expected output file
      records the final loss and validation accuracy for the regression
      harness.
- [ ] **Validation accuracy floor (pre-prune)**: the trained (un-pruned) RNN
      reaches **>=99% per-token validation accuracy** on the
      `y[n] = x[n-2]` task. (For more complex tasks chosen by the implementer,
      the analogous floor is "perfect or near-perfect" — same bar as the blog's
      "perfect validation accuracy" claim.)
- [ ] **Magnitude pruning**: a post-training pruning pass zeroes every weight
      in every parameter tensor whose absolute value falls below a configurable
      threshold. Parameter mutation goes through `Tn.set_values` (or an
      equivalent host-side write) on the parameter tensors after training
      completes; pruning is one-shot, not interleaved with the optimizer.
- [ ] **Sparsity floor (post-prune)**: at the highest threshold for which the
      pruned network still meets the post-prune accuracy floor (next AC),
      **>=80% of all trainable scalar weights are zero**. (The blog reports
      ">90%" on its tasks; 80% is the conservative falsifiable bar — treat
      a result of 90%+ as a successful match to the blog.)
- [ ] **Post-prune accuracy floor**: after pruning to the chosen threshold,
      validation accuracy on >=10k held-out tokens stays **>=99%** (i.e. at
      most 1% degradation from the pre-prune floor on the `y[n] = x[n-2]`
      task; the same "still correctly models the target" criterion the blog
      uses).
- [ ] **Sparsity-aware regularization (pick one explicitly)**: either
      (a) plain L1 weight decay on the parameter tensors during training, with
      the L1 coefficient tuned to satisfy the sparsity AC, OR
      (b) the blog's threshold-shifted L1
      `T(x) = tanh((|x| - t) * s) - tanh(-t * s)` (penalises non-zero weights
      uniformly above threshold `t`). Whichever path is chosen, the example
      documents it and the proposal's "Open questions" item is resolved by the
      implementer's commit message / inline comment.
- [ ] **Graph visualization of the pruned network**: after pruning, the
      surviving sub-network is rendered to a static artifact committed as part
      of the example's output (or as an `.expected` snapshot). Acceptable
      forms (in order of preference, implementer picks):
      1. A Graphviz `.dot` file written to disk (matching the blog's medium),
         possibly accompanied by a rendered PNG / SVG in `docs/`.
      2. A `Tensor.to_printbox` text-art rendering of the pruned forward graph,
         saved to a `.expected` file under `test/training/`.
      Either form must visibly omit pruned-out weights / dead branches — i.e.
      the visualization reflects the *post-prune* sub-graph, not the dense
      template.
- [ ] **Reproducibility**: the example is seeded
      (`Utils.settings.fixed_state_for_init`), runs deterministically, and the
      regression harness checks the recorded loss / accuracy / sparsity-%
      to within a documented tolerance.
- [ ] **Where the demo lives**: `test/training/bonsai_rnn.ml` (+ `.expected`
      file + `dune` entry), parallel to existing `test/training/*` examples
      (`bigram.ml`, `transformer_names.ml`, etc.). It runs under the standard
      regression backend (CPU `multicore_cc` or `sync_cc`), not GPU-only.

## Context

### What the blog actually does

The blog demonstrates a workflow, not a single network: train a tiny
hand-designed RNN cell with a sparsity-promoting regularizer, prune away the
overwhelming majority of weights, and stare at the resulting sub-graph until
the task's structure becomes legible (e.g. `y[n] = x[n-2]` resolves to a
two-stage delay line). Five tasks are listed in the post (delay, repeat-loop,
balanced parens, run-length pattern, four-mode logic gate). Networks have 1-8
neurons. The sparsity result (>90% of weights zero at "perfect" task accuracy)
is the headline.

Specifics worth keeping in mind during implementation:

- **Custom cell**, not LSTM/GRU. The blog uses an architecture that "decouples
  output from hidden state" with a custom activation tuned for binary outputs.
  An LSTM cell would also work and lets the implementer reuse the
  building-block proposal in `gh-ocannl-60.md` if it lands first; a small
  vanilla `tanh`-RNN cell (`h_t = tanh(W_x x_t + W_h h_{t-1} + b)`) is the
  smallest scope that still reproduces the qualitative result. Choice is
  intentionally left open — see Open questions.
- **Sparsity regularizer matters**. Plain L1 over-penalises large weights and
  doesn't drive small weights to *exactly zero* (only to "small"). The blog's
  threshold-shifted L1 does. Either is acceptable here as long as the sparsity
  AC is met; this is the most likely friction point in implementation.
- **Pruning is one-shot**. After training, weights below threshold are set to
  zero. The blog also experiments with quantising the surviving weights — that
  is explicitly out of scope (see Scope).

### Existing OCANNL building blocks (verified at HEAD; re-verified 2026-06-12)

Reuse these — do not invent parallel infrastructure.

- **Recurrent unrolling pattern**: the proposal in
  `docs/proposals/gh-ocannl-60.md` (deferred at time of writing) details how
  parameter-shared cells are unrolled across timesteps using OCANNL's
  unit-parameter lifting. Section *Approach B* there is the canonical recipe;
  this task should mirror it, possibly cell-for-cell if `gh-ocannl-60` lands
  first. The mechanism does not require new framework features.
- **Building blocks** — `lib/nn_blocks.ml`. Reference layers: `mlp_layer`,
  `multi_head_attention`, `decoder_only_block`. Convention is
  `let%op name ~label ~cfg () = fun args -> ...` with `{ w }` / `{ b = 0.; o = [...] }`
  declarations for parameter lifting.
- **Activations** — `tensor/operation.ml` exports `relu`, `sat01`, `tanh`. No
  `sigmoid` primitive; if needed, define locally
  `let%op sigmoid x = recip (1 + exp (neg x))` (same advice as
  `gh-ocannl-60.md`).
- **Training loop** — `lib/train.ml` provides `Train.run_once`, parameter set
  enumeration via `Tensor.params`, host-side parameter access via `Tn.get_value`
  / `Tn.set_value` / `Tn.set_values`. `init_params` and `forward` show the
  hosted-vs-virtual memory mode plumbing required to actually mutate parameters
  from OCaml after compilation.
- **Parameter mutation API** — `arrayjit/lib/tnode.ml` `set_values` and
  `set_value`: write a flat `float array` (or single value) into a parameter's
  host-side buffer; the function bumps the memory mode to
  `Hosted Nonconstant` so the next `host_to_device` propagates the change. This
  is the pruning hook: iterate `t.params`, read each tensor, threshold,
  `set_values` back.
- **Graph rendering** — `tensor/tensor.ml` `to_dag` / `to_printbox` produce a
  `PrintBox_utils.dag` with optional value / gradient rendering, suitable for
  text-art visualization of the forward graph. **No Graphviz exporter exists**;
  if AC option (1) is picked, the implementer either adds a small
  `dag -> dot` converter alongside `PrintBox_utils.ml` or generates the `.dot`
  string directly from a walk of the tensor's `children`.
- **Existing language-modeling scaffolds** — `test/training/bigram.ml` and
  `test/training/transformer_names.ml` show the seeding /
  `Tensor.unsafe_reinitialize` / data-tensor / SGD / `.expected` pattern. The
  bonsai example follows the same shape; the only novel pieces are the
  recurrent unroll, the regularizer, the pruning pass, and the visualization
  artifact.

### What is genuinely new for this task

1. **Sparsity-promoting regularizer**. Requires either threading L1 into the
   loss (one-line `params |> Set.fold ~init:0. ~f:(fun acc p -> acc + abs p)`)
   or implementing the threshold-shifted variant.
2. **One-shot pruning pass**. Requires walking `t.Tensor.params`, calling
   `Tn.get_values`, applying `Array.map (fun w -> if abs w < threshold then 0.
   else w)`, calling `Tn.set_values`. Plus correctly invalidating any cached
   device-side copies (`set_values` already updates memory mode; verify the
   next `host_to_device` actually fires).
3. **Sparsity measurement**. Counting zero weights post-prune and reporting the
   percentage in the example output / `.expected` file.
4. **Visualization of the pruned graph**. Either a new `dot` exporter or a
   `to_printbox` snapshot, captured *after* pruning so the dead weights /
   branches read as zero / are visibly absent.

### Related work

- **gh-ocannl-60** (LSTM example) — the recurrent-unroll mechanics overlap
  heavily; the bonsai example may consume `lstm_cell` if 60 lands first, or
  define its own simpler vanilla-RNN cell. The two tasks are not blocked on
  each other; they share idioms.
- **gh-ocannl-59** (makemore examples) — language-modeling siblings; useful
  reference for data-tensor / training-loop scaffolding but not a dependency.
  *(Update 2026-06-12: now closed — `mlp_names.ml` and `mlp_bn_names.ml`
  landed in `test/training/`.)*

## Scope

**In scope**

- `test/training/bonsai_rnn.ml` + `.expected` + `dune` entry.
- A vanilla-RNN (or LSTM, implementer's choice) cell — local to the example
  unless it cleanly factors into `lib/nn_blocks.ml`.
- A small post-training pruning helper, either local to the example or in
  `lib/train.ml` (e.g. `Train.magnitude_prune ~threshold t`).
- A graph-visualization artifact (Graphviz `.dot` OR `to_printbox` snapshot).
- Sparsity-promoting regularizer wired into the loss.

**Out of scope** (file follow-up issues if they bite)

- Quantisation of surviving weights (the blog's secondary experiment).
- Sparse / structured-sparse runtime — pruned-to-zero weights still occupy
  memory and consume FLOPs; this proposal is purely about *demonstrating*
  sparsity, not exploiting it for performance.
- Generalising the pruning pass into a reusable training utility beyond what
  this one example needs.
- A full Graphviz export module for arbitrary OCANNL tensors (a `.dot`
  exporter for *this* example may grow into a library later, but the proposal
  doesn't promise that).
- Reproducing more than one of the blog's five tasks. The implementer may add
  more if it's cheap; only the `y[n] = x[n-2]` baseline is required.
- Multi-layer / bidirectional RNN, GRU.
- Interactive visualization (the blog's logic-analyzer overlay).

**Dependencies**

- Soft / shared-idiom relationship with **gh-ocannl-60**; not a hard blocker.
  If `gh-ocannl-60` merges first, this task can reuse `lstm_cell`. If not, this
  task ships its own minimal vanilla-RNN cell.

## Open questions for the implementer

These are surfaced here rather than in the task's `Questions` section because
they are *design choices*, not *user input required*. The implementer's commit
message or inline comment should record the decision.

1. **Cell architecture**: vanilla `tanh`-RNN (smallest scope, ~10 lines of
   `let%op`) vs. LSTM (reuse / share `gh-ocannl-60`'s `lstm_cell`) vs. blog's
   custom cell (closest to the blog, but requires defining the custom
   activation). Default recommendation: vanilla `tanh`-RNN for the baseline
   `y[n] = x[n-2]` task — minimal scope, sufficient to hit the ACs.
2. **Regularizer**: plain L1 (default, simpler) vs. threshold-shifted L1 (faithful
   to blog, harder to tune). If plain L1 fails to meet the >=80% sparsity AC at
   the post-prune accuracy floor, escalate to the threshold variant.
3. **Visualization medium**: Graphviz `.dot` (matches blog, requires writing a
   small `dag -> dot` walker) vs. `to_printbox` snapshot in `.expected`
   (matches OCANNL's existing test idiom, no new code). Default
   recommendation: `to_printbox` for the regression test, optionally a `.dot`
   committed alongside if the implementer wants to mirror the blog's aesthetic.

# Add Nn_blocks.cross_entropy_loss helper

## Status update (2026-06-12)

- Not implemented yet: there is no `cross_entropy_loss` in `lib/nn_blocks.ml`, and `transformer_with_loss` (now at line 347) still computes `log (softmax ~spec:"... | v" () logits)` — the numerical instability this proposal targets is still present.
- Line references drifted and are corrected in the table below: `reduce_specified_axes` is at lines 98-103, `softmax` at 107-115, `transformer_with_loss` at 347-361.
- Motivation has strengthened: two more training examples landed since (test/training/`fsm_transformer.ml` and `transformer_names.ml`, plus `mlp_names.ml`/`mlp_bn_names.ml`), each hand-rolling a softmax + NLL loss variant.
- The `softmax` helper gained a `?temperature` parameter but its `~spec` convention and subtract-max pattern are unchanged; the proposed design still applies as written.
- No GitHub issue tracks this; it is a standalone ergonomics proposal.

## Goal

Cross-entropy loss from logits is reimplemented in every training example with slight variations, some numerically unstable (e.g. `log(softmax(x))` in `transformer_with_loss`). A canonical `cross_entropy_loss` helper in `Nn_blocks` eliminates this boilerplate and provides a numerically stable implementation using the log-softmax trick.

## Acceptance Criteria

1. `Nn_blocks.cross_entropy_loss` computes cross-entropy loss from raw logits and one-hot targets using numerically stable log-softmax (subtract max before exp).
2. Accepts a `~spec` parameter following the same convention as `Nn_blocks.softmax` (specifies which axes are the class/vocabulary axes).
3. Supports an optional `~mask` parameter for excluding positions (e.g. padding tokens) from the loss.
4. Supports an optional `~normalize_by` parameter for custom normalization (e.g. valid token count instead of batch size).
5. Without mask or normalize_by, the function returns the summed scalar loss (reduced to 0-dimensional).
6. `transformer_with_loss` in `nn_blocks.ml` is refactored to use the new helper, fixing its current numerical instability.
7. The function is documented with an ocamldoc comment following the style of existing helpers.

## Context

### Key files

| File | Lines | Relevance |
|------|-------|-----------|
| `lib/nn_blocks.ml:98-103` | `reduce_specified_axes` | Generates reduction einsum spec from user-facing axis spec -- reuse for the new function |
| `lib/nn_blocks.ml:107-115` | `softmax` | Existing helper whose `~spec` convention and numerical stability pattern (subtract max) should be followed |
| `lib/nn_blocks.ml:347-361` | `transformer_with_loss` | Uses naive `log(softmax(x))` -- primary refactoring target |
| `test/training/mnist_conv.ml:104-108` | MNIST loss | Uses `softmax + epsilon` pattern -- could optionally adopt the helper |
| `test/training/bigram.ml:51-52` | Bigram loss | Manual softmax + NLL pattern |
| `test/training/fsm_transformer.ml:122-123` | FSM loss | Another NLL pattern variant |

*(Update 2026-06-12: line numbers refreshed; `test/training/transformer_names.ml:~130` is a further NLL reimplementation site that landed since this was written.)*

### Numerical stability

The existing `softmax` helper (line 107-115) already implements the subtract-max trick:
```ocaml
let max_vals = x_scaled @^^ spec in
let exp_vals = exp (x_scaled - max_vals) in
exp_vals /. (exp_vals ++ spec)
```

The `cross_entropy_loss` function should use the same approach but compute `log_probs` directly as `shifted - log_sum_exp` rather than taking `log(softmax(x))`, which loses numerical precision.

### DSL patterns

- `%op` functions use labeled arguments and `()` unit for parameter lifting
- Inline parameters `{ name }` are learnable -- the loss function should NOT introduce any
- `@^^` is max-reduce, `++` is sum-reduce, `*. ` is pointwise multiply
- `reduce_specified_axes` converts a spec like `"...|v"` into `"...|v => ...|0"`

## Approach

*Suggested approach -- agents may deviate if they find a better path.*

Add `cross_entropy_loss` to `lib/nn_blocks.ml` near the existing `softmax` function (around line 116):

```ocaml
let%op cross_entropy_loss ~spec ?mask ?normalize_by () ~logits ~targets =
  let reduce_spec = reduce_specified_axes spec in
  let max_logits = logits @^^ reduce_spec in
  let shifted = logits - max_logits in
  let log_sum_exp = log (exp shifted ++ reduce_spec) in
  let log_probs = shifted - log_sum_exp in
  let nll = neg ((targets *. log_probs) ++ reduce_spec) in
  let masked_nll = match mask with None -> nll | Some m -> nll *. m in
  let total = masked_nll ++ "...|... => 0" in
  match normalize_by with None -> total | Some n -> total /. n
```

Then refactor `transformer_with_loss` to call it:
```ocaml
let loss = cross_entropy_loss ~spec:"... | v" () ~logits ~targets:tgt_target in
```

## Scope

**In scope:**
- New `cross_entropy_loss` function in `lib/nn_blocks.ml`
- Refactoring `transformer_with_loss` to use it
- Documenting the new function

**Out of scope:**
- Refactoring training examples (they serve as tutorials showing the manual pattern)
- Label smoothing or other advanced loss variants
- General loss function framework
- Changes to the `softmax` function itself

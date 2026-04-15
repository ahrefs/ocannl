# Proposal: Polish MNIST and CIFAR Classifier Examples

**Task:** gh-ocannl-54 — Add examples: MNIST and CIFAR classifiers
**Issue:** https://github.com/ahrefs/ocannl/issues/54
**Milestone:** v0.7.1

## Goal

Finalize the MNIST and CIFAR-10 classifier examples so they serve as polished, well-documented
tutorials for OCANNL's CNN pipeline. The core implementations already exist
(`test/training/mnist_conv.ml`, `test/training/cifar_conv.ml`) and pass regression tests. This
task completes the remaining acceptance criteria: ensuring `dataprep` integration for CIFAR data
loading, updating the README checkpoint, and verifying documentation quality.

## Acceptance Criteria

1. **MNIST classifier trains and achieves reasonable accuracy.**
   - Already satisfied: `mnist_conv.ml` trains LeNet on MNIST, reaches 27.9% on 2K-sample
     regression subset (above 25% threshold, well above 10% random chance). Full-run mode
     documented to target >95% with 60K samples / 20 epochs.

2. **CIFAR-10 classifier trains and achieves reasonable accuracy.**
   - Already satisfied: `cifar_conv.ml` trains LeNet on CIFAR-10, reaches 19.9% on 2K-sample
     regression subset (above 15% threshold). Full-run mode documented to target >60% with wider
     channels and 50K samples / 50 epochs.

3. **Examples use the `dataprep` package for data loading.**
   - MNIST: satisfied — uses `Dataprep.Mnist.load`.
   - CIFAR: partially satisfied — uses custom `Conv_data.load_cifar10` because
     `Dataprep.Cifar10.load` is broken (downloads Python pickle format instead of binary).
     Either fix the `Dataprep.Cifar10` loader in `ocaml-dataprep`, or document the workaround
     as intentional. The current approach is reasonable and self-contained.

4. **Examples are added as test executables in `test/training/dune`.**
   - Already satisfied: both `mnist_conv` and `cifar_conv` have `(test ...)` stanzas with
     correct dependencies (`ocannl`, `dataprep`, `conv_data`, `unix`).

5. **README.md checkbox for convnet examples can be checked off.**
   - Not yet done: `README.md` line 72 still shows `- [ ] Add convnet examples: MNIST and CIFAR.`
   - Action: change to `- [x]`.

## Context

### Existing code (all committed to `master`)

| File | Purpose | Lines |
|------|---------|-------|
| `test/training/mnist_conv.ml` | MNIST LeNet classifier with train/eval | 204 |
| `test/training/cifar_conv.ml` | CIFAR-10 LeNet classifier with train/eval | 215 |
| `test/training/conv_data.ml` | int8→float32 conversion, CIFAR binary loader | 165 |
| `test/training/mnist_conv.expected` | Regression test expected output | 24 |
| `test/training/cifar_conv.expected` | Regression test expected output | 24 |
| `test/training/dune` | Build stanzas for both tests | relevant stanzas at lines 54-74 |

### Architecture used

Both examples use `Nn_blocks.lenet` (5x5 conv → relu → pool → 5x5 conv → relu → pool → fc120 → fc84 → logits) with `softmax` cross-entropy loss and SGD with linear learning rate decay. The earlier `circles_conv.ml` row variable mismatch issue documented in the task elaboration has been resolved — `lenet` composes `conv2d` and `max_pool2d` successfully.

### Data pipeline

- **MNIST**: `Dataprep.Mnist.load` → `Conv_data.mnist_images_to_float32` (adds channel dim, normalizes to [0,1]) → `Conv_data.labels_to_int_list` → `Nn_blocks.one_hot_of_int_list`
- **CIFAR-10**: `Conv_data.load_cifar10` (custom binary downloader/parser, CHW→HWC) → `Conv_data.cifar_images_to_float32` (normalizes to [-0.5, 0.5] for zero-centered activations) → same label pipeline

### Related proposals

- `docs/proposals/convnet-examples.md` — earlier proposal that guided the initial implementation.

## Approach

Since the core implementation is complete and passing tests, the remaining work is minimal:

### 1. Check README checkbox
Change `README.md` line 72 from `- [ ]` to `- [x]` for the convnet examples item.

### 2. Verify CIFAR `dataprep` criterion
The acceptance criterion says "Examples use the `dataprep` package for data loading." The CIFAR
example uses a custom loader (`Conv_data.load_cifar10`) because the `dataprep` CIFAR loader is
broken. Two options:
- **Option A (recommended):** Accept the current `Conv_data.load_cifar10` as sufficient — it
  downloads, caches, and parses CIFAR-10 binary data correctly. Add a comment noting this is a
  workaround for the broken `Dataprep.Cifar10` loader. (Already documented in `cifar_conv.ml`
  header and `conv_data.ml` docstring.)
- **Option B:** Fix `Dataprep.Cifar10.load` in `ocaml-dataprep` to use the binary format. This
  is a separate task in a different repo.

### 3. Review documentation quality
Both files already have:
- Module-level doc comments explaining purpose, regression vs full-run modes, and manual validation commands
- Inline comments explaining each step (data loading, conversion, model setup, loss, training loop, evaluation)
- Clear separation of configuration constants with full-run values in comments

### 4. Verify `dune runtest` passes
Run `OCANNL_BACKEND=sync_cc dune runtest` to confirm both examples produce expected output. (They are already committed with `.expected` files that match.)

### Estimated effort
Minimal — 1-2 hours. The substantive implementation work is done. This is checkbox-checking
and final review.

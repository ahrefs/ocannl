# Add Convnet Examples: MNIST and CIFAR Classifiers

## Motivation

OCANNL has comprehensive CNN building blocks (`conv2d`, `max_pool2d`, `lenet`, `vgg_block`,
`resnet_block`, etc.) and dataset loaders (`Dataprep.Mnist`, `Dataprep.Cifar10`), but no
training examples that connect them. The only CNN training example is `circles_conv.ml`, which
uses a synthetic 16x16 single-channel dataset. Real-world image classification examples
(MNIST, CIFAR-10) are needed both as tutorials and as integration tests for the CNN pipeline.

GitHub issue: https://github.com/ahrefs/ocannl/issues/54

## Current State

**CNN layers** are implemented in `lib/nn_blocks.ml`:
- `conv2d` (line 255): einsum-based 2D convolution with padding/stride options
- `max_pool2d` (line 294): max pooling with no-padding mode (`<` spec)
- `lenet` (line 371): full LeNet architecture (conv5→pool→conv5→pool→fc120→fc84→logits)
- Deeper architectures available: `vgg_block`, `resnet_block`, `conv_bn_relu`

**Dataset loaders** exist in the `dataprep` package:
- `Dataprep.Mnist.load ~fashion_mnist:false` returns `int8_unsigned` bigarrays: images `[N; 28; 28]`, labels `[N]`
- `Dataprep.Cifar10.load ()` returns `int8_unsigned` bigarrays: images `[N; 32; 32; 3]`, labels `[N]`
- Both handle downloading and caching automatically

**Working CNN example** at `test/training/circles_conv.ml`:
- Uses `Nn_blocks.lenet` with `Dataprep.Circles` (16x16, 1-channel, 3 classes)
- Trains with SGD, cross-entropy loss, converges to loss ~0.01 in 1000 epochs
- Established patterns: `IDX.get_static_symbol` for batching, `TDSL.rebatch`, `one_hot_of_int_list`, `softmax` + log-based cross-entropy

**Gap**: No code converts `int8_unsigned` bigarrays to OCANNL's `float32`-based `Ndarray.t`. The circles
dataset sidesteps this by providing `float32` directly via `generate_single_prec`.

## Proposed Change

Add two training scripts following the `circles_conv.ml` pattern:

### MNIST (`test/training/mnist_conv.ml`)

- Load data via `Dataprep.Mnist.load ~fashion_mnist:false`
- Convert `int8_unsigned [N; 28; 28]` images to `float32 [N; 28; 28; 1]` with `/255.0` normalization
- Convert `int8_unsigned [N]` labels to `int list` for `one_hot_of_int_list ~num_classes:10`
- Use `Nn_blocks.lenet` (default params: out_channels1=6, out_channels2=16)
- Dimension flow: 28→28→14→14→7→fc120→fc84→10 (verified: works cleanly with padding=true conv and stride-2 pool)
- Cross-entropy loss, SGD with learning rate decay
- Target: training loss converges, test accuracy >95% for LeNet on MNIST

### CIFAR-10 (`test/training/cifar_conv.ml`)

- Load data via `Dataprep.Cifar10.load ()`
- Convert `int8_unsigned [N; 32; 32; 3]` to `float32` with `/255.0` normalization (channel dim already present)
- Use a deeper architecture than LeNet — CIFAR-10 with 10 classes and 3-channel 32x32 input requires more capacity.
  Options using existing blocks: `vgg_block` (multiple conv layers per stage), `conv_bn_relu` chains, or `resnet_block`.
- Dimension flow with 32x32: 32→32→16→16→8 (stride-2 pool stages), works cleanly
- Target: test accuracy >60% for a simple CNN

### Data conversion utility

Each script needs `int8_unsigned → float32` conversion with normalization. This can be:
- Inline in each script (simpler, no coupling)
- A shared helper in `Dataprep` or a local module (if the pattern proves reusable)

The decision should be made during implementation based on how much code overlaps.

### Build integration

- Add both tests to `test/training/dune` following the existing `(test ...)` stanza pattern
- Create `.expected` files capturing loss-decrease output for regression testing

### Practical considerations

- **Dataset size**: Full MNIST is 60K images (~188MB as float32), CIFAR is 50K (~614MB). For CI,
  use a subset or low epoch count with relaxed convergence targets.
- **Network dependency**: `Dataprep` downloads on first run. Tests should document this requirement.
- **`batch_norm2d` limitation**: Current implementation lacks running statistics (noted as FIXME at
  line 327). This may affect test-time accuracy if batch norm is used in the CIFAR model.
- **Stale comment in `circles_conv.ml`**: The header (lines 6-16) describes a row variable mismatch
  workaround, but `lenet` is actually used successfully on line 81. The comment should be updated or
  removed.

## Scope

**In scope:**
- MNIST training script with LeNet
- CIFAR-10 training script with appropriate architecture
- Data conversion (int8→float32) in each script
- `.expected` files and dune integration
- Test accuracy evaluation (not just training loss)

**Out of scope:**
- Data augmentation
- Fashion-MNIST variant (trivial follow-up: `~fashion_mnist:true`)
- Hyperparameter search / achieving SOTA accuracy
- Changes to `Dataprep` library interface
- Changes to `Nn_blocks` CNN building blocks

**Dependencies:**
- None blocking. All required building blocks and data loaders exist.
- Related: `watch-ocannl-README-md-9e031df7` (names transformer example, same milestone family)

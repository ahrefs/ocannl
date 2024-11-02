# ocannl

NOTE TO POTENTIAL CONTRIBUTORS: reach out so I can adjust my work style -- start using branches for refactoring. Otherwise you face frustration as the code might be broken. Tagged versions of the code are guaranteed to work as well as the given stage of the project permitted.

OCANNL is sponsored by [Ahrefs](https://ocaml.org/success-stories/peta-byte-scale-web-crawler)! [Visit the Ahrefs website.](https://ahrefs.com/)

## OCANNL -- OCaml Compiles Algorithms for Neural Networks Learning

* A from-scratch, compiled Deep Learning framework.
* Implements backpropagation (i.e. first-order reverse mode autodiff) and shape inference.
* The long-term goal is to provide several "low-level" backends, aiming to seek inspiration from projects such as [TinyGrad](https://github.com/tinygrad/tinygrad), [TVM](https://github.com/apache/tvm), [Luminal](https://github.com/jafioti/luminal).
  * OCANNL starts with a high-level representation, but can compile everything down to `for` loops.
* The library users can compile any amount of code into a routine (i.e. a compilation unit). The user decides explicitly what the scope of a compilation unit is, by putting together the corresponding code. Depending on the use case:
  * the whole training update step can be a single routine,
  * or the step can be composed of a gradient update routine (a forward pass and a backprop pass) and a params update routine (e.g. SGD with momentum, ADAM, etc.),
  * or the user can compile parts of a model separately, manually composing the corresponding forward pass code and the backprop code.
* Tensor axes are split into kinds: batch, input and output. Tensor dimensions have optional labels.
  * The labels ensure a more precise semantics for dimension matching.
  * In the future we might introduce axis labels as an alternative to positional axis selection, it would be a separate naming mechanism.
* OCANNL has full support for the `einsum` notation, integrated with shape inference. Supports static indexing, with a built-in operation to take a slice of the batch axes, integrated with shape inference. Extensible to more static indexing patterns as needs arise.
  * OCANNL does not have dynamic indexing (using the last axis of one tensor as indices into another tensor). If it's needed, it can be added (we had a prototype once, removed to reduce complexity). Then it would also be integrated with shape inference.
* OCANNL has a suite of tutorials doubling as tests with inline expectations.
* OCANNL offers two main levels of abstraction.
  * Differentiable computations, centered around the [`%op`](lib/ppx_op.ml) syntax extension.
    * `%op` stands for "operation", it's meant to express tensors: `Tensor.t`, and tensor functions.
  * Plain computations, centered around the [`%cd`](lib/ppx_cd.ml) syntax extension. It integrates the `arrayjit` backend library with shape inference.
    * `%cd` stands for "code", it's meant to express assignment computations: `Assignments.comp`.
* The support for mixed-precision computations is upcoming.
  * E.g. higher-precision network components, or gradients at a higher precision than values.
  * Currently (v0.3), you can select the precision, and individual computation nodes track their precision, but mixing precisions might break things.
* Should be easily extensible.
* Model surgery should be starightforward (not sure if we are there yet).
* It's a feature, not a bug!
  * To scale a tensor by a number, always use pointwise-multiplication, e.g. `2*.m` or `m*.2`.
  * Matrix-multiplying a tensor `m` by a constant number, e.g. `m*2`, broadcasts the number to the shape of the input axes of the tensor. This results in an output-axes-only tensor (multi-axis-vector) that is the scaled sum over the input axes of the tensor `m`.
  * Matrix-multiplying a constant number by a tensor `m`, e.g. `2*m`, broadcasts the number to the shape of the output axes of the tensor. This results in a tensor whose inputs are of the same shape as the inputs of `m`, and the output shape is 1D (scalar), that is the scaled sum over the output axes of the tensor `m`.
  * The matrix-multiply operation behaves pointwise along the batch axes.

## Usage

[API documentation entry point](https://ahrefs.github.io/ocannl/dev/).

A possible route to learning OCANNL:

1. Read [the introductory slides](docs/OCANNL-slides-basics_backprop_training_loop_codegen.pdf).
2. Get some basic grasp of the aims and design of the project by reading or skimming files in [test/](test/) and [bin/](bin/).
3. Read the syntax extensions documentation [lib/syntax_extensions.md](lib/syntax_extensions.md).
4. Read the introductory part of the shape inference documentation [lib/shape_inference.md](lib/shape_inference.md).
5. Improve your understanding by reading or skimming: [lib/shape.mli](lib/shape.mli), [lib/tensor.mli](lib/tensor.mli), [lib/operation.ml](lib/operation.ml), [arrayjit/lib/backend_types.ml](arrayjit/lib/backend_types.ml), [lib/train.ml](lib/train.ml), and [lib/nn_blocks.ml](lib/nn_blocks.ml).
6. Read [arrayjit/lib/anatomy_of_a_backend.md](arrayjit/lib/anatomy_of_a_backend.md).
7. Read the implementation overview:
   1. Shape inference details [lib/shape_inference.md](lib/shape_inference.md).
   2. Backend-independent optimizations [arrayjit/lib/lowering_and_inlining.md](arrayjit/lib/lowering_and_inlining.md) -- _lowering_ means translating (compiling) from the high-level representation (as assignments) to the low-level representation.
   3. More documentation to come.

### Using the tracing debugger with CUDA computations

To use debugging as provided by configuring `Utils.settings.debug_log_from_routines <- true` with the `cuda` backend, you need to wrap the code scheduling tasks and synchronizing `cuda` devices with `Utils.capture_stdout_logs`. The reason is that CUDA kernels are allowed to use `printf`, but not `fprintf` -- the driver dumps the printing buffer of a device to `stdout` at certain times (e.g. when synchronizing the device). For an example, see the implementation of `Train.example_train_loop`. Specifically, it wraps two sections: the call to `Train.parallel_update`, and the body of the returned `infer_callback`.

IMPORTANT: due to potential bugs, debug logging from CUDA in complex settings currently only works as intended for _very_ small computation sizes.

## Upcoming milestones

This is very tentative.

* 0.5: stream-to-stream synchronization at the buffer level.
  * Need to add support for CUDA events to cudajit, and add `Condition`-based events for CPU backends.
  * Overhaul of the backend interfaces, both user-facing but especially internal: full code sharing.
  * Also: Apple Metal backend, either here 0.5.x or later 0.7.x.
* 0.6: Replicate the scaffolding from [llm.c](https://github.com/karpathy/llm.c) for training GPT-2.
  * More of primitive numeric operations.
  * Useful building blocks for models in [lib/nn_blocks.ml](lib/nn_blocks.ml).
  * A language model example.
  * Port (translate or bind) the Python files from [llm.c](https://github.com/karpathy/llm.c) to implement tokenization, data loading and saving etc.
  * At the end of 0.6.x, we should have an apples-to-apples benchmark comparing OCANNL to [llm.c](https://github.com/karpathy/llm.c) for both CPU and GPU.
* 0.7: Optimize performance -- low hanging fruit.
  * First harvested from [Fast Multidimensional Matrix Multiplication on CPU from Scratch](https://siboehm.com/articles/22/Fast-MMM-on-CPU).
  * Then harvested from [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM).
  * Finally from [llm.c](https://github.com/karpathy/llm.c).
  * These will require splitting a routine into multiple CUDA kernels.
* 0.8: A new abstraction layer automating compilation/linking, execution, and some data transfers.
  * E.g. host-device transfers: copy from host if host update is later than the previous device update.
  * Concise syntax for transfers into the merge buffer since we know which tensor node is transferred and where to.
  * At the end of 0.8.x, OCANNL has a REPL.
* 0.9: Hopefully-efficient expressivity: block tensors, convolution.
  * Requires extending expressivity of projections and the generalized einsum notation.
  * Then, we can add convnet building blocks and corresponding examples starting with MNIST.
  * Verify or rethink usefulness of dimension labels, and whether to introduce axis labels.
* 0.10: Optimize performance: program search.
  * Instead of dynamic scheduling as in tinygrad, we can schedule statically by program search.
  * We should also reproduce the search that tinygrad is doing.
  * Check which optimizations are missing against the implementation of [llm.c](https://github.com/karpathy/llm.c).
* 1.0: Few documentation gaps, some degree of feature completeness.
  * Feature completeness demonstrated by resolving / implementing a few of the $\color{green}{\text{explore}}$ issues.

### Releases

For more details, see [CHANGES](CHANGES.md).

* **0.4.1 Half precision, mixed precision, CUDA virtual devices** (virtual devices renamed to streams in 0.4.2)
  * Half precision. Maybe improvements for mixed-precision computations.
  * Resolve remaining issues with the new scheduler.
  * Initial version of [lib/nn_blocks.ml](lib/nn_blocks.ml).
* **v0.4 merge buffers, C-syntax backend builder**: a significant refactoring of the API.
* **v0.3 shape inference, jitted routines**: a major rewrite of the whole project.
  * **v0.3.3**: continuous integration and opam release.
  * **v0.3.2**: new shape inference feature: tracking leftmost axes -- complete inference for splicing, ellipsis-in-the-middle allowed in einsum notation.
  * **v0.3.1**: sanitizing code inclusion (rootness checks).
  * **v0.3.0**: declarative shape inference; replaced the session interface with a "jitted code routines" API. Cuda defunct.
* **v0.2 inching toward GPU**:
  * **v0.2.1 naive-cuda**: a Cuda backend where blocks and threads are exposed via dedicated axis types.
  * **v0.2.0 stack-as-device**: treating the C function stack as the "device memory".
* **v0.1 GCCJIT backend**:
  * **v0.1.2**: multicore computations using a thread-local "task id" index.
  * **v0.1.1**: inlining scalar constants, improved inlining for virtual nodes.
  * **v0.1.0**: a `Gccjit` backend, single and double precision floats, code compiled as a monolithic update step function.
* **v0.0 untagged**: basic design around shape inference, high-level and low-level code representation. Now-abandoned Meta-OCaml and OCaml backends.

## Why not just use [OWL](https://ocaml.xyz/)?

OCANNL follows different design choices than [OWL](https://ocaml.xyz/). For example:

* OCANNL is not functorized.
* OCANNL has fewer abstraction layers.
* OCANNL has a more powerful shape inference.
* OCANNL only supports backpropagation, while OWL supports full forward and backward auto-diff.
* Some aspects are more centralized in OCANNL than in OWL and form the "infrastructure":
  * Tensor indexing mechanisms are not extensible, other than changing OCANNL code.
  * Shape inference is fully handled by OCANNL and not extensible, other than changing OCANNL code.
  * [`Tensor`](lib/tensor.ml) implements "putting pieces together".
  * [`Train`](lib/train.ml) has the optimization "frontend" and utilities.
  * [`arrayjit`](arrayjit/), which may one day become a standalone library: generates the code, performs backend-agnostic optimizations (_virtual nodes_ whose computation is inlined), implements the backends.
* Some aspects that are more core to OWL are less encapsulated in OCANNL, so it should be more natural to extend them.
  * Specifically, [`Operation`](lib/operation.ml) and [`Train`](lib/train.ml) are just collections of functions.
* OCANNL provides lower-level compilation backends than OWL, it is more self-contained in this sense.

## Installation

Although the project is called `ocannl`, the main package is called `neural_nets_lib`, to avoid the (opam linter's) complaint that the name can be confused with other packages. This also clarifies that `ocannl` is composed of `arrayjit` and `neural_nets_lib`.

The dependency on `ocaml-cudajit` is optional, so you have to install it first to enable the Cuda backend.
